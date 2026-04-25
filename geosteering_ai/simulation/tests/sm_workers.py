# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_workers.py                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Workers Numba + Fortran               ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : concurrent.futures.ProcessPoolExecutor + subprocess        ║
# ║  Dependências: numpy, geosteering_ai.simulation.simulate_multi            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Encapsula a execução paralela dos dois simuladores seguindo o         ║
# ║    paradigma Central Master-Plan, Parallel Execution (Worker Sandboxes). ║
# ║    Cada worker opera num subprocesso isolado — o Python Numba JIT        ║
# ║    reusa o cache `@njit` após warmup único por processo; o Fortran       ║
# ║    tatu.x é copiado para um TemporaryDirectory exclusivo.                 ║
# ║                                                                           ║
# ║  FLUXO DE EXECUÇÃO                                                        ║
# ║    ┌──────────────────────────────────────────────────────────────────┐  ║
# ║    │ Master (QThread)                                                 │  ║
# ║    │  ├─ chunk_models → n_workers batches                            │  ║
# ║    │  └─ ProcessPoolExecutor:                                         │  ║
# ║    │      ├─ Worker 0: warmup JIT → loop chunk → stack H             │  ║
# ║    │      ├─ Worker 1: idem                                           │  ║
# ║    │      └─ Worker K: idem                                           │  ║
# ║    └──────────────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  SINAIS QT                                                                ║
# ║    • progress_update(done, total, mod_per_h)  — cada 100 modelos         ║
# ║    • worker_finished(worker_id, n_done)       — fim de cada chunk        ║
# ║    • finished_all(result_dict)                — término da execução      ║
# ║    • error(msg)                                — erros irrecuperáveis   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Workers assíncronos para simulação Numba e Fortran.

O módulo expõe ``SimulationThread`` (subclasse ``QThread``) que orquestra
um ``ProcessPoolExecutor`` com workers sandbox, além de funções
``run_numba_chunk`` e ``run_fortran_chunk`` (picklable para IPC).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sm_qt_compat import QObject, QThread, Signal

# ──────────────────────────────────────────────────────────────────────────
# Estruturas de configuração
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SimRequest:
    """Pedido de simulação — parâmetros comuns aos dois backends.

    Attributes:
        frequencies_hz: Lista de frequências em Hz.
        tr_spacings_m: Lista de espaçamentos T-R em metros.
        dip_degs: Lista de ângulos de dip em graus.
        positions_z: Array ``(n_pos,)`` de profundidades em metros.
        backend: ``"numba"`` ou ``"fortran"``.
        n_workers: Número de processos sandbox.
        n_threads: Threads por worker (OpenMP/Numba).
        hankel_filter: ``"werthmuller_201pt"`` | ``"kong_61pt"`` | ``"anderson_801pt"``.
        h1: Altura 1º ponto-médio T-R acima da 1ª interface (m).
        tj: Janela de investigação (m).
        p_med: Passo entre medidas (m).
        fortran_binary: Caminho absoluto ao executável ``tatu.x``.
    """

    frequencies_hz: List[float]
    tr_spacings_m: List[float]
    dip_degs: List[float]
    positions_z: np.ndarray
    backend: str = "numba"
    n_workers: int = 4
    n_threads: int = 4
    hankel_filter: str = "werthmuller_201pt"
    h1: float = 10.0
    tj: float = 120.0
    p_med: float = 0.2
    fortran_binary: str = ""
    save_raw: bool = False
    output_dir: str = ""
    output_filename: str = "sim_output"


# ──────────────────────────────────────────────────────────────────────────
# Worker picklable — Numba JIT sandbox
# ──────────────────────────────────────────────────────────────────────────


def run_numba_chunk(
    worker_id: int,
    chunk: List[dict],
    positions_z: np.ndarray,
    freqs: List[float],
    trs: List[float],
    dips: List[float],
    hankel_filter: str,
    n_threads: int,
    progress_q: Optional[Any] = None,
    block_size: int = 100,
    total_hint: Optional[int] = None,
) -> Tuple[int, np.ndarray, np.ndarray, float]:
    """Executa ``simulate_multi`` num chunk de modelos dentro de um worker.

    O ambiente ``NUMBA_NUM_THREADS`` é definido ANTES do import de numba,
    garantindo que o thread pool do subprocesso respeite o valor solicitado.
    A primeira chamada é descartada (warmup JIT).

    Args:
        progress_q: ``multiprocessing.Queue`` opcional. Quando fornecido, o
            worker publica um ``dict`` a cada ``block_size`` modelos com os
            campos ``{done, total, block_start, block_end, block_s, mean_s,
            eta_s}`` para que a QThread master reemita como log estruturado.
        block_size: Modelos por bloco de log (padrão 100, seguindo
            ``fifthBuildTIVModels.py``).
        total_hint: Quando fornecido, sobrescreve ``len(chunk)`` na mensagem
            (usado em modo sequencial para que o log exiba o total global).

    Returns:
        Tupla ``(worker_id, H_stack, z_obs, elapsed_sec)`` onde
        ``H_stack`` tem shape ``(n_chunk, nTR, nAng, n_pos, nf, 9)``.
    """
    os.environ["NUMBA_NUM_THREADS"] = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)

    from geosteering_ai.simulation import SimulationConfig, simulate_multi

    positions_z = np.asarray(positions_z, dtype=np.float64)
    cfg = SimulationConfig(
        frequency_hz=float(freqs[0]),
        tr_spacing_m=float(trs[0]),
        hankel_filter=hankel_filter,
        backend="numba",
        num_threads=n_threads,
    )

    # ── Warmup JIT (1 chamada descartada por processo) ───────────────────
    if chunk:
        m0 = chunk[0]
        _ = simulate_multi(
            rho_h=np.asarray(m0["rho_h"], dtype=np.float64),
            rho_v=np.asarray(m0["rho_v"], dtype=np.float64),
            esp=np.asarray(m0["thicknesses"], dtype=np.float64),
            positions_z=positions_z,
            frequencies_hz=freqs,
            tr_spacings_m=trs,
            dip_degs=dips,
            cfg=cfg,
            hankel_filter=hankel_filter,
        )
        # Publica evento de warmup concluído (útil p/ warmup feedback em E8).
        if progress_q is not None:
            try:
                progress_q.put({"event": "warmup_done", "worker_id": worker_id})
            except Exception:
                pass

    total = int(total_hint) if total_hint is not None else len(chunk)
    t0 = time.perf_counter()
    block_start_t = t0
    H_list: List[np.ndarray] = []
    z_obs_ref: Optional[np.ndarray] = None
    for i, m in enumerate(chunk):
        r = simulate_multi(
            rho_h=np.asarray(m["rho_h"], dtype=np.float64),
            rho_v=np.asarray(m["rho_v"], dtype=np.float64),
            esp=np.asarray(m["thicknesses"], dtype=np.float64),
            positions_z=positions_z,
            frequencies_hz=freqs,
            tr_spacings_m=trs,
            dip_degs=dips,
            cfg=cfg,
            hankel_filter=hankel_filter,
        )
        H_list.append(r.H_tensor)
        if z_obs_ref is None:
            z_obs_ref = r.z_obs

        # Log estruturado a cada `block_size` modelos (modo n_workers=1).
        if progress_q is not None and (i + 1) % block_size == 0:
            now = time.perf_counter()
            block_s = now - block_start_t
            mean_s = block_s / float(block_size)
            remaining = max(0, total - (i + 1))
            eta_s = mean_s * remaining
            try:
                progress_q.put(
                    {
                        "event": "block",
                        "worker_id": worker_id,
                        "done": i + 1,
                        "total": total,
                        "block_start": i + 1 - block_size + 1,
                        "block_end": i + 1,
                        "block_s": block_s,
                        "mean_s": mean_s,
                        "eta_s": eta_s,
                    }
                )
            except Exception:
                pass
            block_start_t = now

    elapsed = time.perf_counter() - t0
    if H_list:
        H_stack = np.stack(H_list, axis=0)
    else:
        # Shape 6D consistente com o caso não-vazio (0, nTR, nAng, n_pos, nf, 9) —
        # evita erro "arrays must have same number of dimensions" no concatenate
        # do master quando um worker recebe chunk vazio (n_workers > n_models).
        n_pos = int(positions_z.shape[0])
        nf = len(freqs)
        n_tr = len(trs)
        n_ang = len(dips)
        H_stack = np.empty((0, n_tr, n_ang, n_pos, nf, 9), dtype=np.complex128)
    z_obs_out = z_obs_ref if z_obs_ref is not None else np.empty((0,), dtype=np.float64)
    return worker_id, H_stack, z_obs_out, elapsed


# ──────────────────────────────────────────────────────────────────────────
# Worker picklable — Fortran tatu.x sandbox
# ──────────────────────────────────────────────────────────────────────────


def _write_model_in(
    path: str,
    freqs: List[float],
    dips: List[float],
    h1: float,
    tj: float,
    p_med: float,
    trs: List[float],
    filename: str,
    n_layers: int,
    rho_h: Sequence[float],
    rho_v: Sequence[float],
    thicknesses: Sequence[float],
    current_model: int,
    max_models: int,
    use_arbitrary_freq: int,
    filter_type: int,
) -> None:
    """Escreve um ``model.in`` compatível com tatu.x v10.0.

    Layout conforme ``docs/reference/documentacao_simulador_fortran_otimizado.md``.
    Para multi-frequência (nf>1), ``use_arbitrary_freq=1`` é mandatório.
    """
    nf = len(freqs)
    ntheta = len(dips)
    nTR = len(trs)
    lines: List[str] = []
    lines.append(f"{nf}                !nf\n")
    for f in freqs:
        lines.append(f"{f:.6f}\n")
    lines.append(f"{ntheta}                !ntheta\n")
    for d in dips:
        lines.append(f"{d:.6f}\n")
    lines.append(f"{h1:.6f}           !h1\n")
    lines.append(f"{tj:.6f}           !tj\n")
    lines.append(f"{p_med:.6f}        !p_med\n")
    lines.append(f"{nTR}                !nTR\n")
    for t in trs:
        lines.append(f"{t:.6f}\n")
    lines.append(f"{filename}              !nome dos arquivos de saída\n")
    lines.append(f"{n_layers}                !número de camadas\n")
    for j in range(n_layers):
        rh = round(float(rho_h[j]), 2)
        rv = round(float(rho_v[j]), 2)
        if rv < rh:
            rv = rh
        if j == 0:
            lines.append(f"{rh}    {rv}     !resistividades horizontal e vertical\n")
        else:
            lines.append(f"{rh}    {rv}\n")
    for j, t in enumerate(thicknesses):
        tv = round(float(t), 2)
        if j == 0:
            lines.append(f"{tv}              !espessuras das n-2 camadas\n")
        else:
            lines.append(f"{tv}\n")
    lines.append(
        f"{current_model} {max_models}         !modelo atual e o número máximo de modelos\n"
    )
    lines.append(f"{use_arbitrary_freq}                 !F5 use_arbitrary_freq\n")
    lines.append("0                 !F7 use_tilted_antennas\n")
    lines.append("0                 !F6 use_compensation\n")
    lines.append(
        f"{filter_type}                 !filter_type (0=Werthmuller 1=Kong 2=Anderson)\n"
    )
    with open(path, "w") as f:
        f.writelines(lines)


def run_fortran_chunk(
    worker_id: int,
    chunk: List[dict],
    freqs: List[float],
    trs: List[float],
    dips: List[float],
    h1: float,
    tj: float,
    p_med: float,
    fortran_binary: str,
    filter_type: int,
    n_threads: int,
    model_start: int,
    model_end: int,
    progress_q: Optional[Any] = None,
    block_size: int = 100,
    total_hint: Optional[int] = None,
) -> Tuple[int, List[Tuple[str, str]], float]:
    """Executa tatu.x num chunk de modelos em TemporaryDirectory sandbox.

    Cada modelo reescreve o ``model.in`` do sandbox e chama ``subprocess.run``.
    Arquivos ``.dat``/``.out`` gerados são movidos para ``cwd`` com prefixo
    ``w{worker_id}_`` para posterior merge pelo master.

    Returns:
        ``(worker_id, [(base, prefixed)], elapsed_sec)``.
    """
    env = {**os.environ, "OMP_NUM_THREADS": str(n_threads)}
    results: List[Tuple[str, str]] = []
    original_dir = os.getcwd()
    use_arbitrary_freq = 1 if len(freqs) > 1 else 0
    filename_base = "sim_batch"

    total = int(total_hint) if total_hint is not None else len(chunk)
    t0 = time.perf_counter()
    block_start_t = t0
    with tempfile.TemporaryDirectory(prefix=f"sm_w{worker_id}_") as tmpdir:
        tatu_dst = os.path.join(tmpdir, "tatu.x")
        shutil.copy2(fortran_binary, tatu_dst)
        os.chmod(tatu_dst, 0o755)
        model_in_path = os.path.join(tmpdir, "model.in")

        for i, m in enumerate(chunk):
            current_idx = model_start + i
            _write_model_in(
                path=model_in_path,
                freqs=freqs,
                dips=dips,
                h1=h1,
                tj=tj,
                p_med=p_med,
                trs=trs,
                filename=filename_base,
                n_layers=int(m["n_layers"]),
                rho_h=m["rho_h"],
                rho_v=m["rho_v"],
                thicknesses=m["thicknesses"],
                current_model=current_idx,
                max_models=model_end,
                use_arbitrary_freq=use_arbitrary_freq,
                filter_type=filter_type,
            )
            try:
                subprocess.run(
                    [tatu_dst],
                    cwd=tmpdir,
                    env=env,
                    capture_output=True,
                    check=True,
                    timeout=120,
                )
            except subprocess.CalledProcessError as e:
                # Falha do Fortran: aborta o chunk e retorna o que foi feito
                raise RuntimeError(
                    f"[Worker {worker_id}] Fortran retornou erro no modelo "
                    f"{current_idx}: {e.stderr.decode('utf-8', 'replace')[:300]}"
                )

            # Log estruturado a cada `block_size` modelos (modo n_workers=1).
            if progress_q is not None and (i + 1) % block_size == 0:
                now = time.perf_counter()
                block_s = now - block_start_t
                mean_s = block_s / float(block_size)
                remaining = max(0, total - (i + 1))
                eta_s = mean_s * remaining
                try:
                    progress_q.put(
                        {
                            "event": "block",
                            "worker_id": worker_id,
                            "done": i + 1,
                            "total": total,
                            "block_start": i + 1 - block_size + 1,
                            "block_end": i + 1,
                            "block_s": block_s,
                            "mean_s": mean_s,
                            "eta_s": eta_s,
                        }
                    )
                except Exception:
                    pass
                block_start_t = now

        # Move artefatos gerados para cwd com prefixo único por worker
        for f in Path(tmpdir).glob("*.dat"):
            new_name = f"w{worker_id}_{f.name}"
            shutil.move(str(f), os.path.join(original_dir, new_name))
            results.append((f.name, new_name))
        for f in Path(tmpdir).glob("*.out"):
            new_name = f"w{worker_id}_{f.name}"
            shutil.move(str(f), os.path.join(original_dir, new_name))
            results.append((f.name, new_name))

    elapsed = time.perf_counter() - t0
    return worker_id, results, elapsed


# ──────────────────────────────────────────────────────────────────────────
# QThread orquestrador
# ──────────────────────────────────────────────────────────────────────────


class SimulationThread(QThread):
    """Thread Qt que orquestra o ProcessPoolExecutor.

    Emite sinais:
      * ``progress_update(done, total, mod_per_h)`` — a cada worker finalizado
      * ``log(str)`` — mensagens informativas
      * ``finished_all(dict)`` — resultado consolidado
      * ``error(str)`` — erros irrecuperáveis

    O resultado consolidado contém:
      * ``backend``: "numba" ou "fortran"
      * ``H_stack``: ndarray (n_models, nTR, nAng, n_pos, nf, 9) — Numba
      * ``z_obs``:  ndarray (nAng, n_pos) — Numba
      * ``dat_files``: list[str] — Fortran
      * ``elapsed``: float total wall clock em segundos
      * ``throughput_mod_h``: float
    """

    progress_update = Signal(int, int, float)
    log = Signal(str)
    finished_all = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        req: SimRequest,
        models: List[dict],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._req = req
        self._models = models
        self._stopped = False

    def request_stop(self) -> None:
        """Solicita interrupção cooperativa (entre chunks)."""
        self._stopped = True

    def _emit_block_log(self, evt: Dict[str, Any]) -> None:
        """Converte um evento de ``progress_q`` em mensagem de log estruturada.

        Formato exatamente igual ao de ``fifthBuildTIVModels.py:1287-1290``:
        ``"  Modelos {A}–{B} de {T} | bloco={S}s | média={M}s/modelo | ETA={E}min"``.
        """
        try:
            self.log.emit(
                f"  Modelos {int(evt['block_start']):>4d}–{int(evt['block_end']):<4d} "
                f"de {int(evt['total'])} | bloco={float(evt['block_s']):6.1f}s | "
                f"média={float(evt['mean_s']):.3f}s/modelo | "
                f"ETA={float(evt['eta_s']) / 60.0:.1f}min"
            )
        except Exception:
            pass

    def run(self) -> None:  # noqa: C901 — single orquestrador, legível
        req = self._req
        models = self._models
        n_total = len(models)
        if n_total == 0:
            self.error.emit("Lista de modelos vazia.")
            return

        # Chunking uniforme com resto distribuído. Limita n_workers ao
        # número de modelos — evita workers vazios que retornariam arrays
        # 1D (dtype complex128) incompatíveis com o concatenate 6D do
        # reagrupamento (bug: "input arrays must have same number of
        # dimensions, 6 vs 1").
        requested_workers = max(1, int(req.n_workers))
        n_workers = min(requested_workers, n_total)
        if n_workers < requested_workers:
            self.log.emit(
                f"  [INFO] n_workers reduzido de {requested_workers} para "
                f"{n_workers} (= n_modelos). Evita workers ociosos."
            )
        chunk_size = n_total // n_workers
        remainder = n_total % n_workers
        batches: List[Tuple[int, List[dict], int, int]] = []
        start = 0
        for w in range(n_workers):
            extra = 1 if w < remainder else 0
            end = start + chunk_size + extra
            batches.append((w, models[start:end], start, end))
            start = end

        # Log estruturado em blocos de 100 modelos é habilitado apenas em modo
        # sequencial (n_workers=1), replicando fielmente o formato usado por
        # ``fifthBuildTIVModels.py``. Em paralelo, workers concluem em ordem
        # não-monotônica e o log de blocos se tornaria confuso.
        use_progress_q = n_workers == 1

        t0 = time.perf_counter()
        done = 0
        throughput = 0.0
        H_parts: Dict[int, np.ndarray] = {}
        z_obs_ref: Optional[np.ndarray] = None
        dat_files: List[str] = []

        # Setup opcional do canal IPC de progresso.
        progress_q = None
        mp_manager = None
        if use_progress_q:
            try:
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                mp_manager = ctx.Manager()
                progress_q = mp_manager.Queue()
                self.log.emit(
                    "Modo sequencial (1 worker) — log estruturado em blocos de 100 modelos."
                )
            except Exception as exc:
                self.log.emit(
                    f"  [AVISO] Não foi possível criar fila de progresso ({exc}); "
                    "log estruturado desabilitado."
                )
                progress_q = None
                use_progress_q = False

        try:
            if req.backend == "numba":
                self.log.emit(
                    f"Numba — {n_workers} workers × {req.n_threads} threads "
                    f"({n_total} modelos)."
                )
                if n_workers > 1:
                    # Warmup JIT feedback inline: avisa antes do primeiro
                    # future.result() que silenciaria 10-30 s de compilação.
                    self.log.emit(
                        f"  Aquecendo JIT em {n_workers} workers "
                        f"(~10–30 s no primeiro modelo de cada worker)…"
                    )
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(
                            run_numba_chunk,
                            wid,
                            chunk,
                            req.positions_z,
                            req.frequencies_hz,
                            req.tr_spacings_m,
                            req.dip_degs,
                            req.hankel_filter,
                            req.n_threads,
                            progress_q if use_progress_q else None,
                            100,
                            n_total if use_progress_q else None,
                        ): (wid, chunk, s, e)
                        for wid, chunk, s, e in batches
                    }

                    if use_progress_q:
                        # Modo sequencial: drena progress_q enquanto o único
                        # future roda (single future, usamos list() para pegar).
                        (fut,) = list(futures.keys())
                        warmup_logged = False
                        while not fut.done():
                            if self._stopped:
                                break
                            try:
                                evt = progress_q.get(timeout=0.5)
                            except Exception:
                                continue
                            if not isinstance(evt, dict):
                                continue
                            if evt.get("event") == "warmup_done":
                                if not warmup_logged:
                                    self.log.emit(
                                        "  JIT aquecido — iniciando processamento."
                                    )
                                    warmup_logged = True
                                continue
                            if evt.get("event") == "block":
                                self._emit_block_log(evt)
                                dt = max(time.perf_counter() - t0, 1e-6)
                                throughput = float(evt["done"]) / dt * 3600.0
                                self.progress_update.emit(
                                    int(evt["done"]), n_total, throughput
                                )
                        wid, H_stack, z_obs, elapsed_w = fut.result()
                        H_parts[wid] = H_stack
                        if z_obs_ref is None and z_obs.size > 0:
                            z_obs_ref = z_obs
                        done = H_stack.shape[0]
                        dt = max(time.perf_counter() - t0, 1e-6)
                        throughput = done / dt * 3600.0
                        self.progress_update.emit(done, n_total, throughput)
                        self.log.emit(
                            f"  Worker {wid}: {H_stack.shape[0]} modelos "
                            f"em {elapsed_w:.2f}s"
                        )
                    else:
                        for future in as_completed(futures):
                            if self._stopped:
                                break
                            wid, H_stack, z_obs, elapsed_w = future.result()
                            H_parts[wid] = H_stack
                            if z_obs_ref is None and z_obs.size > 0:
                                z_obs_ref = z_obs
                            done += H_stack.shape[0]
                            dt = max(time.perf_counter() - t0, 1e-6)
                            throughput = done / dt * 3600.0
                            self.progress_update.emit(done, n_total, throughput)
                            self.log.emit(
                                f"  Worker {wid}: {H_stack.shape[0]} modelos "
                                f"em {elapsed_w:.2f}s"
                            )
                # Reagrupa preservando ordem dos workers (id asc). Filtra
                # arrays vazios para evitar ValueError "arrays must have
                # same number of dimensions" quando algum worker não
                # processou modelos.
                valid_parts = [
                    H_parts[k]
                    for k in sorted(H_parts.keys())
                    if H_parts[k].ndim == 6 and H_parts[k].shape[0] > 0
                ]
                if valid_parts:
                    H_full = (
                        valid_parts[0]
                        if len(valid_parts) == 1
                        else np.concatenate(valid_parts, axis=0)
                    )
                else:
                    H_full = np.empty((0,), dtype=np.complex128)
                elapsed_total = time.perf_counter() - t0
                # Linha-resumo de finalização com 89 traços (mesmo padrão de
                # fifthBuildTIVModels.py:1294).
                self.log.emit("─" * 89)
                self.log.emit(
                    f"Tempo de execução (treinamento): "
                    f"{elapsed_total / 3600.0:.4f} horas  ({elapsed_total:.1f} s)"
                )
                self.finished_all.emit(
                    {
                        "backend": "numba",
                        "H_stack": H_full,
                        "z_obs": z_obs_ref,
                        "elapsed": elapsed_total,
                        "throughput_mod_h": n_total / max(elapsed_total, 1e-6) * 3600.0,
                        "n_models": n_total,
                    }
                )
                return

            # ── Fortran ──────────────────────────────────────────────────
            if not req.fortran_binary or not os.path.isfile(req.fortran_binary):
                raise FileNotFoundError(
                    f"Binário tatu.x não encontrado: {req.fortran_binary!r}"
                )
            filter_type_map = {
                "werthmuller_201pt": 0,
                "kong_61pt": 1,
                "anderson_801pt": 2,
            }
            filter_type = filter_type_map.get(req.hankel_filter, 0)

            self.log.emit(
                f"Fortran — {n_workers} workers × {req.n_threads} threads "
                f"({n_total} modelos)."
            )
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        run_fortran_chunk,
                        wid,
                        chunk,
                        req.frequencies_hz,
                        req.tr_spacings_m,
                        req.dip_degs,
                        req.h1,
                        req.tj,
                        req.p_med,
                        req.fortran_binary,
                        filter_type,
                        req.n_threads,
                        s + 1,
                        e,
                        progress_q if use_progress_q else None,
                        100,
                        n_total if use_progress_q else None,
                    ): (wid, chunk, s, e)
                    for wid, chunk, s, e in batches
                }

                if use_progress_q:
                    (fut,) = list(futures.keys())
                    while not fut.done():
                        if self._stopped:
                            break
                        try:
                            evt = progress_q.get(timeout=0.5)
                        except Exception:
                            continue
                        if isinstance(evt, dict) and evt.get("event") == "block":
                            self._emit_block_log(evt)
                            dt = max(time.perf_counter() - t0, 1e-6)
                            throughput = float(evt["done"]) / dt * 3600.0
                            self.progress_update.emit(
                                int(evt["done"]), n_total, throughput
                            )
                    wid, files, elapsed_w = fut.result()
                    dat_files.extend([nn for (_, nn) in files])
                    done = n_total
                    dt = max(time.perf_counter() - t0, 1e-6)
                    throughput = done / dt * 3600.0
                    self.progress_update.emit(done, n_total, throughput)
                    self.log.emit(f"  Worker {wid}: chunk processado em {elapsed_w:.2f}s")
                else:
                    for future in as_completed(futures):
                        if self._stopped:
                            break
                        wid, files, elapsed_w = future.result()
                        dat_files.extend([nn for (_, nn) in files])
                        done += len(futures[future][1])
                        dt = max(time.perf_counter() - t0, 1e-6)
                        throughput = done / dt * 3600.0
                        self.progress_update.emit(done, n_total, throughput)
                        self.log.emit(
                            f"  Worker {wid}: chunk processado em {elapsed_w:.2f}s"
                        )
            elapsed_total = time.perf_counter() - t0
            self.log.emit("─" * 89)
            self.log.emit(
                f"Tempo de execução (treinamento): "
                f"{elapsed_total / 3600.0:.4f} horas  ({elapsed_total:.1f} s)"
            )
            self.finished_all.emit(
                {
                    "backend": "fortran",
                    "dat_files": dat_files,
                    "elapsed": elapsed_total,
                    "throughput_mod_h": n_total / max(elapsed_total, 1e-6) * 3600.0,
                    "n_models": n_total,
                }
            )
        except Exception as exc:  # pragma: no cover — protegido por sinais
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()[:2000]}")
        finally:
            # Encerra o Manager (se criado) para liberar o subprocess auxiliar.
            if mp_manager is not None:
                try:
                    mp_manager.shutdown()
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────
# QThread para I/O de artefatos (evita travamento da GUI em _on_sim_finished)
# ──────────────────────────────────────────────────────────────────────────


class SaveArtifactsThread(QThread):
    """Thread Qt dedicada a exportar ``.dat`` + ``.out`` fora da main thread.

    Antes desta classe, a gravação dos artefatos ocorria dentro de
    ``_on_sim_finished`` na main Qt thread — resultava em freeze da GUI por
    15–60 s em simulações com n_models=3000+ porque ``write_dat_from_tensor``
    tem 5 loops aninhados em Python puro (1.8M iterações típicas) + I/O de
    ~300 MB via ``tofile``.

    Esta QThread delega toda a operação para um worker Qt separado,
    mantendo a event loop responsiva. O MainWindow conecta:

      * ``saved(dict)``: {"dat", "out", "elapsed_s"} — sucesso
      * ``error(str)``: mensagem de erro

    O objeto deve ser armazenado como atributo do parent (MainWindow) para
    sobreviver ao retorno do handler — caso contrário o GC destrói o
    thread antes de ``run()`` completar.
    """

    saved = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        dat_path: str,
        out_path: str,
        H_stack: np.ndarray,
        z_obs: np.ndarray,
        n_dips: int,
        n_freqs: int,
        nmaxmodel: int,
        angles: List[float],
        freqs_hz: List[float],
        nmeds_per_angle: List[int],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._dat_path = dat_path
        self._out_path = out_path
        self._H_stack = H_stack
        self._z_obs = z_obs
        self._n_dips = n_dips
        self._n_freqs = n_freqs
        self._nmaxmodel = nmaxmodel
        self._angles = angles
        self._freqs_hz = freqs_hz
        self._nmeds_per_angle = nmeds_per_angle

    def run(self) -> None:  # pragma: no cover — I/O side-effect
        try:
            from .sm_io import write_dat_from_tensor, write_out_file

            t0 = time.perf_counter()
            write_dat_from_tensor(self._dat_path, self._H_stack, self._z_obs)
            dat_elapsed = time.perf_counter() - t0
            write_out_file(
                self._out_path,
                n_dips=self._n_dips,
                n_freqs=self._n_freqs,
                nmaxmodel=self._nmaxmodel,
                angles=self._angles,
                freqs_hz=self._freqs_hz,
                nmeds_per_angle=self._nmeds_per_angle,
            )
            total_elapsed = time.perf_counter() - t0
            self.saved.emit(
                {
                    "dat": self._dat_path,
                    "out": self._out_path,
                    "elapsed_s": total_elapsed,
                    "dat_elapsed_s": dat_elapsed,
                }
            )
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()[:1500]}")


__all__ = [
    "SaveArtifactsThread",
    "SimRequest",
    "SimulationThread",
    "run_fortran_chunk",
    "run_numba_chunk",
]
