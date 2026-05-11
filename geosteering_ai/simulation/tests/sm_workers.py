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

import multiprocessing as _mp
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sm_qt_compat import QObject, QThread, Signal

# ──────────────────────────────────────────────────────────────────────────
# Pool persistente — elimina overhead de spawn/import/JIT por simulação
# ──────────────────────────────────────────────────────────────────────────

# Flag de controle: True dentro de um worker após o inicializador rodar.
# False no processo principal — impede warmup redundante.
_WORKER_INITIALIZED: bool = False

# Pool singleton reaproveitado entre simulações consecutivas.
_PERSISTENT_POOL: Optional[ProcessPoolExecutor] = None
_PERSISTENT_POOL_CONFIG: Optional[Tuple[int, int, str]] = None


def _numba_init_worker(n_threads: int, hankel_filter: str) -> None:
    """Inicializador leve do pool — roda UMA vez por worker ao spawnar.

    Configura variáveis de ambiente de threading (OMP/KMP) e retorna em
    < 100 ms. O aquecimento do JIT Numba é feito por ``_run_numba_warmup_task``
    (submetido como Future pelo ``PoolWarmupThread``), o que garante que o
    warmup seja cancelável e não bloqueie o shutdown do aplicativo.

    Histórico:
        v2.10–v2.24: warmup usava ``esp=[5.0, 5.0]`` (shape inválida) no
        próprio inicializador. Falha silenciosa via ValueError →
        ``_WORKER_INITIALIZED = True`` sem warmup real → cold-start em prod.

        v2.25: corrigiu o shape mas manteve ``simulate_multi`` no
        inicializador. Efeito colateral: inicializador levava 35–38 s (JIT
        frio) ou 6 s (cache em disco) e NÃO podia ser cancelado. O handler
        ``atexit`` do ``ProcessPoolExecutor`` chama ``shutdown(wait=True)``
        em todos os pools rastreados — com workers presos no inicializador,
        isso causa travamento garantido ao fechar o aplicativo.

        v2.26: warmup movido para ``_run_numba_warmup_task`` (Future).
        Inicializador retorna em < 100 ms; cancelamento via
        ``cancel_futures=True`` é garantido; sem hang no shutdown.

    Note:
        NÃO setamos ``NUMBA_NUM_THREADS`` aqui — ver docstring de
        ``_acquire_numba_pool`` para a razão completa. ``OMP_NUM_THREADS``
        (BLAS/MKL) é seguro setar porque não conflita com o config do Numba.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "2")
    os.environ.setdefault("KMP_WARNINGS", "FALSE")
    # _WORKER_INITIALIZED permanece False até _run_numba_warmup_task concluir.


def _run_numba_warmup_task(n_threads: int, hankel_filter: str) -> bool:
    """Aquece o JIT Numba no worker — submetido como Future pelo PoolWarmupThread.

    Ao contrário do inicializador, Futures são canceláveis via
    ``cancel_futures=True`` no shutdown do pool — eliminando o travamento
    ao fechar o aplicativo que ocorria quando o warmup ficava no
    inicializador (v2.25).

    Executa ``simulate_multi`` com shapes representativas e corretas:

      ┌────────────────────────────────────────────────────────────────┐
      │  n_layers = 10  →  esp.shape = (n-2,) = (8,)                 │
      │  positions_z = 50 pontos (suficiente para ativar JIT)         │
      │  Warmup A: single-combo (nTR×nAng = 1) → kernel cached       │
      │  Warmup B: multi-combo  (nTR×nAng = 4) → FLAT prange         │
      └────────────────────────────────────────────────────────────────┘

    50 posições (vs. 600 em v2.25): JIT é ativado pela PRIMEIRA chamada,
    não pelo tamanho do array. Arrays grandes só adicionam latência ao
    warmup quente (~6 s → ~2 s com 50 posições) sem benefício de bytecode.

    Histórico:
        v2.10–v2.24: Bug de shape → ValueError silencioso → warm falso.
        v2.25: shape correto, mas warmup no inicializador → hang no shutdown.
        v2.26: warmup como Future, shape correto, 50 posições.

    Args:
        n_threads: Threads Numba (mascaramento via ``SimulationConfig``).
        hankel_filter: Filtro Hankel — deve coincidir com o da simulação real
            para garantir warm hit no primeiro uso.

    Returns:
        True se o warmup completou sem exceção; False caso contrário.
        Quando False, ``run_numba_chunk`` executa o warmup secundário no
        primeiro chunk real (fora de ``t0_sim``).
    """
    global _WORKER_INITIALIZED
    if _WORKER_INITIALIZED:
        return True
    warmup_failed = False
    try:
        import numpy as _np

        from geosteering_ai.simulation import SimulationConfig, simulate_multi

        _cfg = SimulationConfig(
            backend="numba", num_threads=n_threads, hankel_filter=hankel_filter
        )
        # n_layers=10 → esp.shape=(n-2,)=(8,) — correto para _validate_multi_inputs.
        _rho = _np.array(
            [1.0, 10.0, 1.0, 50.0, 1.0, 100.0, 1.0, 5.0, 20.0, 1.0],
            dtype=_np.float64,
        )
        _esp = _np.full(8, 3.0, dtype=_np.float64)
        # 50 posições: ativa JIT (qualquer tamanho ≥ 1 serve) e aquece
        # alocadores moderadamente sem overhead excessivo no path quente.
        _pos = _np.linspace(-5.0, 20.0, 50, dtype=_np.float64)
        # Warmup A — single-combo → _simulate_positions_njit_cached
        simulate_multi(
            rho_h=_rho,
            rho_v=_rho,
            esp=_esp,
            positions_z=_pos,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cfg=_cfg,
        )
        # Warmup B — multi-combo → _simulate_combined_prange_flat
        simulate_multi(
            rho_h=_rho,
            rho_v=_rho,
            esp=_esp,
            positions_z=_pos,
            frequencies_hz=[20000.0, 40000.0],
            tr_spacings_m=[1.0, 2.0],
            dip_degs=[0.0],
            cfg=_cfg,
        )
    except Exception:
        # Mantém _WORKER_INITIALIZED=False para que run_numba_chunk execute
        # o warmup secundário no primeiro chunk real (fora de t0_sim).
        warmup_failed = True
    if not warmup_failed:
        _WORKER_INITIALIZED = True
    return not warmup_failed


def _noop() -> None:
    """Função vazia usada para sincronizar o pool sem carga real."""


def _acquire_numba_pool(
    n_workers: int, n_threads: int, hankel_filter: str
) -> ProcessPoolExecutor:
    """Retorna (ou cria) o pool persistente com os parâmetros especificados.

    Se o pool existente tiver configuração diferente, o antigo é encerrado
    sem esperar antes de criar o novo.

    Note (v2.16):
        Sprint 15.1 — antes do spawn dos workers, setamos
        ``NUMBA_NUM_THREADS = n_threads`` no env do processo pai. Os workers
        spawn herdam este env, e Numba lê o valor durante a primeira
        ``import numba`` no worker (que é disparada pelo unpickle do próprio
        ``_numba_init_worker``). Resultado: o pool de threads do Numba dentro
        de cada worker nasce dimensionado em ``n_threads``, e
        ``numba.set_num_threads(n_threads)`` em ``simulate_multi`` apenas
        confirma esse mascaramento (sem RuntimeError).

        Em v2.15 (commits 0f92035 + e1c8864), este passo foi removido sob a
        teoria de que ``set_num_threads`` resolveria sozinho. Mas como o
        pool nasce com ``cpu_count()`` (16 em hyperthreaded), e
        ``set_num_threads`` falha silenciosamente em chamadas posteriores
        em alguns estados internos, workers acabavam rodando com threads
        ativas em estado indefinido — provocando regressão 4–8×.

        Mantemos ``OMP_NUM_THREADS = n_threads`` para BLAS/MKL.
    """
    global _PERSISTENT_POOL, _PERSISTENT_POOL_CONFIG
    cfg_key = (n_workers, n_threads, hankel_filter)
    if _PERSISTENT_POOL is None or _PERSISTENT_POOL_CONFIG != cfg_key:
        if _PERSISTENT_POOL is not None:
            try:
                _PERSISTENT_POOL.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                _PERSISTENT_POOL.shutdown(wait=False)
            except Exception:
                pass
        # ── FIX v2.16: dimensionar pool de threads do Numba no spawn ─────
        # Setado no PAI antes do spawn — herdado pelos workers. Numba lê
        # NUMBA_NUM_THREADS na primeira import (durante o spawn dos workers).
        os.environ["NUMBA_NUM_THREADS"] = str(n_threads)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "2")
        os.environ.setdefault("KMP_WARNINGS", "FALSE")
        # ────────────────────────────────────────────────────────────────
        _PERSISTENT_POOL = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_mp.get_context("spawn"),
            initializer=_numba_init_worker,
            initargs=(n_threads, hankel_filter),
        )
        _PERSISTENT_POOL_CONFIG = cfg_key
    return _PERSISTENT_POOL


def release_numba_pool() -> None:
    """Encerra o pool persistente de forma limpa.

    Deve ser chamado em ``closeEvent()`` da janela principal para liberar
    os subprocessos antes do encerramento do aplicativo.
    """
    global _PERSISTENT_POOL, _PERSISTENT_POOL_CONFIG
    if _PERSISTENT_POOL is not None:
        try:
            _PERSISTENT_POOL.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            _PERSISTENT_POOL.shutdown(wait=False)
        except Exception:
            pass
        _PERSISTENT_POOL = None
        _PERSISTENT_POOL_CONFIG = None


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

    O número ativo de threads Numba é controlado via ``cfg.num_threads``
    (mascaramento interno com ``numba.set_num_threads``). Não alteramos
    ``NUMBA_NUM_THREADS`` em env var — ver nota em ``_numba_init_worker``.
    A primeira chamada é descartada (warmup JIT) quando o pool initializer
    não a executou.

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
    # OMP_NUM_THREADS: limita BLAS/MKL por worker (seguro setar após import Numba).
    # NUMBA_NUM_THREADS NÃO é setado aqui — ver docstring de _numba_init_worker.
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "2")
    os.environ.setdefault("KMP_WARNINGS", "FALSE")

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
    # Pulado quando _numba_init_worker já executou o warmup via pool initializer
    # (pool persistente). Em modo legado (ProcessPoolExecutor efêmero) ou em
    # workers criados sem initializer, _WORKER_INITIALIZED permanece False e o
    # warmup ocorre normalmente aqui.
    if chunk and not _WORKER_INITIALIZED:
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
    # Publica evento de warmup concluído — sempre, para que o master
    # atualize o log independentemente do caminho tomado acima.
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
    # v2.11 — sinais de pausa/retomada/cancelamento cooperativo
    paused = Signal()
    resumed = Signal()
    cancelled = Signal()

    def __init__(
        self,
        req: SimRequest,
        models: List[dict],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._req = req
        self._models = models
        # Flag legada de cancelamento — preservada para retro-compat.
        # request_stop() = "pare ao final do chunk em curso" (cancelamento).
        self._stopped = False
        # v2.11 — flag separada para "cancelamento agressivo" pedido pelo
        # botão Cancel da UI. Distingue cancel intencional vs stop legado.
        self._cancel_requested = False
        # v2.11 — Event para pausa cooperativa. Quando setado (default),
        # a thread roda livremente; quando clear, a thread bloqueia em wait().
        # Uso em pontos checkpoint dentro do loop run().
        import threading as _threading

        self._pause_event = _threading.Event()
        self._pause_event.set()
        self._is_paused: bool = False

    def request_stop(self) -> None:
        """Solicita interrupção cooperativa (entre chunks).

        Mantém comportamento v2.10 — usado pelo botão "Parar" legado.
        Para cancelamento explícito v2.11, ver :meth:`request_cancel`.
        """
        self._stopped = True

    def request_pause(self) -> None:
        """Solicita pausa cooperativa (v2.11).

        A thread bloqueia em pontos checkpoint dentro do loop ``run()``
        até :meth:`request_resume` ou :meth:`request_cancel`. Idempotente.
        """
        if self._is_paused:
            return
        self._is_paused = True
        self._pause_event.clear()
        self.paused.emit()

    def request_resume(self) -> None:
        """Retoma a execução após :meth:`request_pause` (v2.11). Idempotente."""
        if not self._is_paused:
            return
        self._is_paused = False
        self._pause_event.set()
        self.resumed.emit()

    def request_cancel(self) -> None:
        """Solicita cancelamento agressivo da simulação (v2.11).

        Diferente de ``request_stop``, este método também libera a thread
        de qualquer pausa em andamento, garantindo que o cleanup ocorra
        imediatamente. Emite ``cancelled`` quando o cleanup conclui.
        """
        self._cancel_requested = True
        self._stopped = True  # também sinaliza para o loop legado
        # Libera qualquer wait pendente em pause checkpoint.
        self._pause_event.set()

    @property
    def is_paused(self) -> bool:
        """Estado atual da pausa cooperativa (v2.11)."""
        return self._is_paused

    @property
    def is_cancelled(self) -> bool:
        """Estado atual do cancelamento (v2.11)."""
        return self._cancel_requested

    def _wait_if_paused(self) -> None:
        """Bloqueia se houver pausa solicitada (v2.11). Chamado em checkpoints.

        Usar timeout interno em ``Event.wait`` permitiria checagem de
        cancelamento durante a pausa — mas como ``request_cancel`` faz
        ``_pause_event.set()`` para liberar o wait, basta um wait simples.
        """
        if not self._pause_event.is_set():
            self._pause_event.wait()

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
        # t0_sim: redefinido no branch Numba APÓS todos os workers completarem
        # _numba_init_worker (spawn + JIT). Throughput usa t0_sim para excluir
        # overhead de pool creation (~10–12 s no cold start).
        # t0 permanece como referência de tempo total (pool + simulação).
        t0_sim: Optional[float] = None
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
                # ── Sprint 17.1 (v2.17): diagnóstico de paralelismo ──────
                # Loga topologia da CPU e detecta oversubscrição (workers
                # × threads > cores físicos), que causa perda 30-50% em
                # CPUs com Hyperthreading/SMT. Esta era a CAUSA RAIZ da
                # regressão remanescente em produção GUI (38k mod/h):
                # defaults v2.16 produziam 4w × 4t = 16 threads em 8 cores
                # físicos = 2× oversubscrição.
                try:
                    from geosteering_ai.simulation import detect_cpu_topology

                    _logical, _phys, _has_ht = detect_cpu_topology()
                    _total_threads = n_workers * int(req.n_threads)
                    _ht_tag = " (HT/SMT)" if _has_ht else ""
                    self.log.emit(
                        f"CPU: {_phys} cores físicos · {_logical} threads "
                        f"lógicas{_ht_tag} · alvo paralelo: {_total_threads} threads"
                    )
                    if _total_threads > _phys:
                        factor = _total_threads / max(1, _phys)
                        self.log.emit(
                            f"  ⚠ AVISO: oversubscrição {factor:.2f}× "
                            f"({_total_threads} threads > {_phys} cores físicos). "
                            f"Performance pode degradar 30-50%. Recomendado: "
                            f"reduzir workers ou threads para {_phys} threads totais."
                        )
                except Exception:
                    pass
                self.log.emit(
                    f"Numba — {n_workers} workers × {req.n_threads} threads "
                    f"({n_total} modelos)."
                )
                # Pool persistente: workers já aquecidos pelo _run_numba_warmup_task
                # (Future submetido por PoolWarmupThread na abertura da GUI).
                # Spawn/import/JIT só ocorrem na PRIMEIRA simulação ou quando
                # n_workers/n_threads/hankel_filter mudam.
                pool = _acquire_numba_pool(n_workers, req.n_threads, req.hankel_filter)
                # ── Sprint 18.1 (v2.18) / v2.26: aguarda workers, define t0_sim ──
                # Os _noop ficam na fila FIFO ATRÁS dos _run_numba_warmup_task já
                # submetidos por PoolWarmupThread. Quando _noop retorna, todos os
                # warmup futures já concluíram → JIT pronto → t0_sim correto.
                # Pool warm (2.ª simulação mesma sessão) → retorna em ~0 ms.
                # Pool frio (JIT ainda em andamento) → bloqueia até warmup futures
                # concluírem, depois retorna imediatamente. t0_sim excluí overhead.
                _init_futs = [pool.submit(_noop) for _ in range(n_workers)]
                for _if in _init_futs:
                    try:
                        _if.result(timeout=120)
                    except Exception:
                        pass
                t0_sim = time.perf_counter()
                _warmup_s = t0_sim - t0
                if _warmup_s > 0.5:
                    self.log.emit(
                        f"  Pool aquecido em {_warmup_s:.1f}s (spawn + JIT). "
                        "Throughput mede apenas simulação daqui em diante."
                    )
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
                        # v2.11: pausa cooperativa — bloqueia se _pause_event
                        # foi clear via request_pause(). request_cancel libera.
                        self._wait_if_paused()
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
                            dt = max(time.perf_counter() - (t0_sim or t0), 1e-6)
                            throughput = float(evt["done"]) / dt * 3600.0
                            self.progress_update.emit(
                                int(evt["done"]), n_total, throughput
                            )
                    wid, H_stack, z_obs, elapsed_w = fut.result()
                    H_parts[wid] = H_stack
                    if z_obs_ref is None and z_obs.size > 0:
                        z_obs_ref = z_obs
                    done = H_stack.shape[0]
                    dt = max(time.perf_counter() - (t0_sim or t0), 1e-6)
                    throughput = done / dt * 3600.0
                    self.progress_update.emit(done, n_total, throughput)
                    self.log.emit(
                        f"  Worker {wid}: {H_stack.shape[0]} modelos "
                        f"em {elapsed_w:.2f}s"
                    )
                else:
                    for future in as_completed(futures):
                        # v2.11: pausa cooperativa entre workers concluídos.
                        self._wait_if_paused()
                        if self._stopped:
                            break
                        wid, H_stack, z_obs, elapsed_w = future.result()
                        H_parts[wid] = H_stack
                        if z_obs_ref is None and z_obs.size > 0:
                            z_obs_ref = z_obs
                        done += H_stack.shape[0]
                        dt = max(time.perf_counter() - (t0_sim or t0), 1e-6)
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
                # sim_elapsed: apenas tempo de simulação (exclui warmup de pool).
                # t0_sim é definido após os _noop confirmarem init dos workers.
                sim_elapsed = time.perf_counter() - (t0_sim or t0)
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
                        "throughput_mod_h": n_total / max(sim_elapsed, 1e-6) * 3600.0,
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
                        # v2.11: pausa cooperativa Fortran path (sequencial).
                        self._wait_if_paused()
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
                    self.log.emit(
                        f"  Worker {wid}: chunk processado em {elapsed_w:.2f}s"
                    )
                else:
                    for future in as_completed(futures):
                        # v2.11: pausa cooperativa Fortran path (paralelo).
                        self._wait_if_paused()
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
            # v2.11 — emit cancelled após cleanup se foi cancelamento explícito.
            # Diferencia stop legado (apenas _stopped) de cancel intencional.
            if self._cancel_requested:
                self.cancelled.emit()
            # Encerra o Manager (se criado) para liberar o subprocess auxiliar.
            if mp_manager is not None:
                try:
                    mp_manager.shutdown()
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────
# QThread para pré-aquecimento do pool Numba em background (v2.18)
# ──────────────────────────────────────────────────────────────────────────


class PoolWarmupThread(QThread):
    """Pré-aquece o pool Numba persistente em background para eliminar cold-start.

    Quando ``SimulationPage`` é aberta, esta thread cria/reusa o pool com os
    defaults recomendados e submete ``_run_numba_warmup_task`` como Future
    para cada worker. Na primeira simulação real, o pool já está pronto:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  Sem pre-warm (v2.17):                                              │
    │    Usuário clica "Simular" → cold pool → overhead de JIT incluído   │
    │    no throughput → 38k mod/h reportado (vs. 85k real)               │
    │                                                                      │
    │  Com pre-warm via inicializador (v2.18–v2.25):                      │
    │    GUI abre → workers spawnam → _numba_init_worker (35 s JIT frio)  │
    │    → pool "pronto" após 35 s; saída da app durante init = HANG      │
    │                                                                      │
    │  Com pre-warm via Future (v2.26):                                   │
    │    GUI abre → workers spawnam em < 1 s (init leve) → warmup future  │
    │    submetido por worker → JIT ocorre no future (cancelável)          │
    │    → saída limpa a qualquer momento (cancel_futures=True)            │
    │    → t0_sim ainda excluí JIT (noops ficam na fila atrás dos warm.)  │
    └──────────────────────────────────────────────────────────────────────┘

    Ordenamento da fila de futures que garante t0_sim correto:

      PoolWarmupThread submete:   [warm_1, warm_2, ..., warm_N]
      SimulationThread submete:   [noop_1, noop_2, ..., noop_N, chunk_1, ...]
                                  └── ficam atrás dos warms na fila FIFO ──┘
      Quando noop_N retorna → todos os warms já concluíram → t0_sim correto.

    Signals:
        warmup_done(elapsed_s, n_workers, n_threads): Pool e JIT prontos.
        warmup_error(msg): Falha não-fatal (warmup secundário em run_numba_chunk).

    Note:
        Usa ``hankel_filter="werthmuller_201pt"`` (padrão) para pre-warm.
        Se o filtro real for diferente, o pool é recriado na simulação
        (custo único). Para a configuração padrão, o warm hit é garantido.
    """

    warmup_done = Signal(float, int, int)  # elapsed_s, n_workers, n_threads
    warmup_error = Signal(str)

    def __init__(
        self,
        n_workers: int,
        n_threads: int,
        hankel_filter: str = "werthmuller_201pt",
        parent: Optional["QObject"] = None,
    ) -> None:
        super().__init__(parent)
        self._n_workers = n_workers
        self._n_threads = n_threads
        self._hankel_filter = hankel_filter

    def run(self) -> None:
        """Cria pool, submete _run_numba_warmup_task por worker e aguarda."""
        t0 = time.perf_counter()
        try:
            pool = _acquire_numba_pool(
                self._n_workers, self._n_threads, self._hankel_filter
            )
            # Submete warmup como Future — um por worker.
            # Futures são canceláveis (cancel_futures=True no shutdown);
            # o inicializador não é — essa é a diferença central v2.26.
            futs = [
                pool.submit(
                    _run_numba_warmup_task, self._n_threads, self._hankel_filter
                )
                for _ in range(self._n_workers)
            ]
            for f in futs:
                f.result(timeout=120)
            self.warmup_done.emit(
                time.perf_counter() - t0, self._n_workers, self._n_threads
            )
        except Exception as exc:
            self.warmup_error.emit(str(exc))


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


# ──────────────────────────────────────────────────────────────────────────
# Pré-aquecimento de cache Numba JIT (v2.9)
# ──────────────────────────────────────────────────────────────────────────


class NumbaPrimer(QThread):
    """Pré-aquece o cache Numba JIT em background no startup do GUI.

    Executa ``simulate_multi`` com um modelo trivial no processo principal,
    escrevendo artefatos compilados (.nbi/.nbc) em disco via ``cache=True``.
    Quando a primeira simulação real iniciar, os workers subprocessos
    carregarão o cache do disco (~3–5 s) em vez de recompilar do zero
    (~90–110 s após atualização de ambiente Python/Numba).

    Signals:
        primer_done(float): Tempo elapsed em segundos. Valor baixo (<15 s)
            indica que o cache já estava em disco; valor alto indica que o
            compilador LLVM precisou recompilar as funções @njit(cache=True).
        primer_failed(str): Emitido se Numba não estiver disponível ou se
            ``simulate_multi`` lançar exceção. O argumento é a msg de erro.

    Note:
        O primer usa ``NUMBA_NUM_THREADS=1`` para não disputar CPU com a GUI.
        Os workers de simulação usam ``n_threads`` configurado pelo usuário.
        Após o primer concluir, o cache em disco é válido para qualquer
        quantidade de threads (Numba compila por assinatura de tipos, não por
        contagem de threads).
    """

    primer_done = Signal(float)
    primer_failed = Signal(str)

    def run(self) -> None:
        """Configura env OMP, importa simulate_multi e roda 1 modelo trivial."""
        # NÃO setar NUMBA_NUM_THREADS/OMP_NUM_THREADS aqui — modificaria o ambiente
        # do processo pai, envenenando workers filhos (spawn herda o env): eles
        # inicializariam o pool Numba com max=1 thread e falhariam em
        # set_num_threads(N>1) com "must be between 1 and 1".
        # O SimulationConfig(num_threads=1) abaixo já garante thread única via
        # _numba.set_num_threads(1) dentro de simulate_multi — sem afetar o env.
        os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "2")
        os.environ.setdefault("KMP_WARNINGS", "FALSE")

        try:
            from geosteering_ai.simulation import SimulationConfig, simulate_multi
        except Exception as exc:
            self.primer_failed.emit(str(exc))
            return

        t0 = time.perf_counter()
        try:
            cfg = SimulationConfig(backend="numba", num_threads=1)
            simulate_multi(
                rho_h=np.array([1.0, 10.0, 1.0], dtype=np.float64),
                rho_v=np.array([1.0, 10.0, 1.0], dtype=np.float64),
                esp=np.array([5.0, 5.0], dtype=np.float64),
                positions_z=np.array([0.0], dtype=np.float64),
                frequencies_hz=[20000.0],
                tr_spacings_m=[1.0],
                dip_degs=[0.0],
                cfg=cfg,
            )
        except Exception as exc:
            self.primer_failed.emit(str(exc))
            return

        self.primer_done.emit(time.perf_counter() - t0)


__all__ = [
    "NumbaPrimer",
    "SaveArtifactsThread",
    "SimRequest",
    "SimulationThread",
    "_acquire_numba_pool",
    "_noop",
    "release_numba_pool",
    "run_fortran_chunk",
    "run_numba_chunk",
]
