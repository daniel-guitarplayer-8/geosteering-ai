# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/simulate.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Subcomando `simulate` da CLI                               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.56 — geometria/chunk JAX + tempo)        ║
# ║  Versão      : v2.56                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-10                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : argparse + simulate_multi / simulate_batch                 ║
# ║  Dependências: numpy, geosteering_ai.cli._{exec,backend,table,hwinfo}      ║
# ║  Padrão      : Hexagonal — adapter externo do simulador                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Handler do subcomando ``simulate``: gera N modelos sintéticos 1D       ║
# ║    5-camada, executa no backend escolhido (``numba`` padrão / ``jax``     ║
# ║    GPU), apresenta os resultados numa tabela ASCII (throughput, tempo,    ║
# ║    paralelismo, hardware, NaN/Inf) e opcionalmente grava ``.npz`` ou      ║
# ║    ``.dat``/``.out`` (22-col conforme ``geosteering-physics.md`` §4).     ║
# ║                                                                           ║
# ║  RECURSOS v2.53                                                           ║
# ║    --backend {numba,jax} · --dtype · --jax-strategy · --warmup ·          ║
# ║    --format {npz,dat,none} · --json · --repeat N · --compare-backends     ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    handle_simulate: handler do subcomando, recebe argparse.Namespace     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Subcomando ``simulate`` da CLI Geosteering AI (Sprint v2.53).

Gera modelos sintéticos com perfis 1D randomizados e os executa no backend
escolhido (Numba CPU ou JAX GPU), apresentando os resultados numa tabela
ASCII e, opcionalmente, gravando ``.npz`` ou ``.dat``/``.out`` 22-col.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from time import perf_counter
from typing import List, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _parse_float_list(text: str | None, default: list[float]) -> list[float]:
    """Interpreta string CSV de floats; retorna ``default`` se None ou inválido.

    Aceita vírgula ou ponto-e-vírgula como separador. Espaços são ignorados.

    Args:
        text: string no formato ``"1.0,2.0,3.0"`` ou ``None``.
        default: lista retornada quando ``text`` é None, vazio ou inválido.

    Returns:
        Lista de floats parseados, ou ``default`` em caso de erro.

    Example:
        >>> _parse_float_list("2000, 20000, 100000", [20000.0])
        [2000.0, 20000.0, 100000.0]
        >>> _parse_float_list(None, [20000.0])
        [20000.0]
    """
    if not text or not text.strip():
        return default
    try:
        parsed = [
            float(v.strip()) for v in text.replace(";", ",").split(",") if v.strip()
        ]
        return parsed if parsed else default
    except ValueError:
        logger.warning(
            "Valor inválido em lista de floats: %r — usando default %s", text, default
        )
        return default


def _build_random_models(
    n_models: int,
    seed: int,
    *,
    geometry: str = "per-model",
    n_geometries: int | None = None,
    quantize_step: float | None = None,
) -> list[dict]:
    """Gera N modelos sintéticos 1D 5-camada para smoke/benchmark.

    Cada modelo é um dict ``{rho_h, rho_v, esp}`` com:
      - 5 camadas (3 internas + 2 semi-espaços)
      - resistividades em [1.0, 100.0] Ω·m
      - razão TIV ``rho_v / rho_h`` em [1.0, 3.0]
      - espessuras das camadas internas em [2.0, 10.0] m

    Args:
        n_models: número de modelos a gerar.
        seed: semente do gerador (reprodutibilidade).
        geometry: modo de amostragem de ``esp`` (batchabilidade no JAX):
            ``"per-model"`` (DEFAULT, comportamento legado — esp único por
            modelo), ``"templates"`` (K geometrias replicadas → agrupável,
            JAX satura) ou ``"quantized"`` (esp arredondado → parcial).
        n_geometries: (só ``templates``) K geometrias distintas.
        quantize_step: (só ``quantized``) passo de quantização (m).

    Returns:
        Lista de dicts ``[{rho_h, rho_v, esp}, ...]``, com cada array
        em ``np.float64`` (float32 não é suportado pelo simulador
        Fortran-equivalente).

    Raises:
        ValueError: se ``n_models <= 0`` (numpy.random.default_rng
            aceita 0 mas retorna lista vazia, sem erro); ou ``geometry``
            inválido / ``quantize_step <= 0``.

    Note:
        Range de resistividades alinhado aos modelos canônicos do
        projeto (oklahoma_3, devine_8, etc.). O modo ``per-model`` preserva
        EXATAMENTE o stream rng legado (esp sorteado no loop); ``templates``/
        ``quantized`` pré-computam ``esp`` via
        :func:`geosteering_ai.cli._exec.sample_geometry`. O Numba é indiferente
        ao modo — só o JAX-grouped se beneficia do compartilhamento.

    Example:
        >>> models = _build_random_models(2, seed=42)
        >>> len(models)
        2
        >>> sorted(models[0].keys())
        ['esp', 'rho_h', 'rho_v']
    """
    rng = np.random.default_rng(seed)
    # per-model: preserva o stream legado (esp no loop). templates/quantized:
    # pré-computa esp_all (consome rng ANTES do loop de rho).
    esp_all = None
    if geometry != "per-model":
        from geosteering_ai.cli._exec import sample_geometry

        esp_all = sample_geometry(
            rng,
            n_models,
            3,
            mode=geometry,
            n_geometries=n_geometries,
            quantize_step=quantize_step,
        )
    models = []
    for i in range(n_models):
        rho_h = rng.uniform(1.0, 100.0, size=5).astype(np.float64)
        ratio = rng.uniform(1.0, 3.0, size=5).astype(np.float64)
        rho_v = rho_h * ratio
        esp = (
            esp_all[i]
            if esp_all is not None
            else rng.uniform(2.0, 10.0, size=3).astype(np.float64)
        )
        models.append({"rho_h": rho_h, "rho_v": rho_v, "esp": esp})
    return models


def _build_stats(
    *,
    backend: str,
    device: str,
    throughput_mod_h: float,
    throughput_std: float,
    elapsed_s: float,
    repeat: int,
    n_models: int,
    n_pos: int,
    frequencies_hz: Sequence[float],
    dip_degs: Sequence[float],
    tr_spacings_m: Sequence[float],
    workers: int | None,
    threads: int | None,
    dtype: str,
    jax_strategy: str,
    n_geometry_groups: int | None,
    finitude: Mapping[str, object],
    reason: str | None = None,
    warmup_s: float | None = None,
    total_s: float | None = None,
) -> dict:
    """Monta o dict de estatísticas consumido por ``build_result_rows``/JSON.

    Args:
        backend/device: backend EFETIVO + device (após eventual fallback).
        throughput_mod_h/throughput_std: mediana e desvio do throughput (mod/h).
        elapsed_s: tempo da MEDIANA das rodadas hot (s) — coerente com o throughput.
        repeat: nº de rodadas hot medidas.
        n_models/n_pos: dimensões da carga.
        frequencies_hz/dip_degs/tr_spacings_m: eixos de geometria.
        workers/threads: paralelismo Numba (None = auto).
        dtype/jax_strategy: parâmetros do path JAX.
        n_geometry_groups: grupos de geometria (JAX) ou None.
        finitude: dict ``{nan_count, inf_count, all_finite}``.
        reason: motivo do fallback de backend (e.g. jax→numba não-agrupável) ou
            None se o backend efetivo == solicitado.
        warmup_s: tempo da fase de warmup (s) — transparência v2.56.
        total_s: tempo total do handler (s) — explica o gap vs ``time real``.

    Returns:
        dict plano (JSON-serializável) com todas as chaves de resultado.
    """
    return {
        "backend": backend,
        "device": device,
        "throughput_mod_h": throughput_mod_h,
        "throughput_std": throughput_std,
        "elapsed_s": elapsed_s,
        "warmup_s": warmup_s,
        "total_s": total_s,
        "repeat": repeat,
        "n_models": n_models,
        "n_pos": n_pos,
        "n_freqs": len(frequencies_hz),
        "n_dips": len(dip_degs),
        "n_trs": len(tr_spacings_m),
        "workers": workers,
        "threads": threads,
        "dtype": dtype,
        "jax_strategy": jax_strategy,
        "n_geometry_groups": n_geometry_groups,
        "reason": reason,
        "nan_count": finitude.get("nan_count"),
        "inf_count": finitude.get("inf_count"),
        "all_finite": finitude.get("all_finite"),
    }


def _save_simulation_output(
    out_dir: Path,
    out_format: str,
    H6: np.ndarray,
    models: Sequence[Mapping[str, np.ndarray]],
    positions_z: np.ndarray,
    frequencies_hz: Sequence[float],
    dip_degs: Sequence[float],
    *,
    quiet: bool,
) -> None:
    """Grava o tensor H em ``.npz`` ou ``.dat``/``.out`` (22-col §4).

    Args:
        out_dir: diretório de saída (criado se necessário).
        out_format: ``"npz"`` ou ``"dat"`` (``"none"`` não chega aqui).
        H6: tensor ``(n_models, nTR, nAng, n_pos, nf, 9)`` complex.
        models: lista de modelos (para mapear res_h/res_v no z_obs).
        positions_z: ``(n_pos,)`` profundidades (m).
        frequencies_hz/dip_degs: eixos para os metadados ``.out``.
        quiet: suprime o log de confirmação.

    Raises:
        OSError: falha de escrita (propagada ao handler → exit 1).

    Note:
        ``.dat`` segue o layout 22-col de ``geosteering-physics.md`` §4:
        col1=zobs, col2/3=res_h/res_v (resistividade REAL na camada de cada
        z_obs via :func:`_exec.rho_at_obs_from_batch`), col4-21=Re/Im das 9
        componentes (Hxx..Hzz). H é gravado CRU (mesmo estado do `.dat` do
        ``tatu.x``). Re-legível por ``data.loading.load_binary_dat``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_format == "npz":
        # FIX v2.53: grava o tensor REAL (H_stack), não repr(result). O bug
        # legado lia getattr(result, "H", None) → sempre None → salvava repr.
        out_file = out_dir / "simulate_results.npz"
        np.savez(out_file, H=H6)
        if not quiet:
            logger.info("Resultados (.npz) gravados em: %s", out_file)
        return

    # ── out_format == "dat" — .dat/.out 22-col conforme geosteering-physics §4
    from geosteering_ai.cli._exec import models_to_batch, rho_at_obs_from_batch
    from geosteering_ai.simulation.io.tensor_dat import (
        write_dat_from_tensor,
        write_out_file,
    )

    n_models, _nTR, nAng, n_pos, nf, _ = H6.shape
    rho_h_batch, rho_v_batch, esp_batch = models_to_batch(models)
    # res_h/res_v NA camada de cada z_obs (col2/col3) — (n_models, n_pos).
    rho_h_obs, rho_v_obs = rho_at_obs_from_batch(
        positions_z, rho_h_batch, rho_v_batch, esp_batch
    )
    # Broadcast no eixo de dip → (n_models, nAng, n_pos) (write_dat espera isso).
    rho_h_obs3 = np.broadcast_to(rho_h_obs[:, None, :], (n_models, nAng, n_pos))
    rho_v_obs3 = np.broadcast_to(rho_v_obs[:, None, :], (n_models, nAng, n_pos))
    z_obs = np.broadcast_to(
        np.asarray(positions_z, dtype=np.float64)[None, :], (nAng, n_pos)
    )

    dat_file = out_dir / "simulate_results.dat"
    out_file = out_dir / "simulate_results.out"
    write_dat_from_tensor(
        str(dat_file),
        H6,
        z_obs,
        rho_h_obs3,
        rho_v_obs3,
        model_id_start=1,
    )
    # .out sidecar — nmeds = n_pos por ângulo (grid compartilhado entre dips).
    # Nota: o layout 22-col do .dat é §4-conforme em TODOS os casos (re-legível
    # por load_binary_dat, leitura plana por stride). O .out clássico (4 linhas)
    # descreve nt/nf/nmaxmodel mas NÃO o eixo de TR; para nTR>1 os registros de
    # cada TR ficam concatenados no stream (ordem Fortran) — use o .dat direto.
    write_out_file(
        str(out_file),
        n_dips=nAng,
        n_freqs=nf,
        nmaxmodel=n_models,
        angles=list(dip_degs),
        freqs_hz=list(frequencies_hz),
        nmeds_per_angle=[n_pos] * nAng,
    )
    if not quiet:
        logger.info("Resultados (.dat/.out 22-col) gravados em: %s", out_dir)


def _emit_results(args: argparse.Namespace, stats: dict, hw: dict) -> None:
    """Imprime a tabela ASCII e/ou o JSON dos resultados (stdout — D9 exc.).

    Args:
        args: namespace (usa ``as_json`` e ``quiet``).
        stats: dict de estatísticas (:func:`_build_stats`).
        hw: dict de hardware (:func:`_hwinfo.collect_hardware_info`).

    Note:
        ``--json`` emite SOMENTE o JSON (stdout limpo, apto a ``| jq``) e
        suprime a tabela; sem ``--json``, a tabela é exibida (a menos que
        ``--quiet``). Usa ``print`` (exceção D9 documentada — stdout de CLI).
    """
    if args.as_json:
        payload = {"result": stats, "hardware": hw}
        print(json.dumps(payload, ensure_ascii=False, default=str))
        return
    if not args.quiet:
        from geosteering_ai.cli._table import build_result_rows, render_kv_table

        rows = build_result_rows(stats, hw)
        print(render_kv_table("RESULTADO — geosteering-cli simulate", rows))


def _run_compare_backends(
    args: argparse.Namespace,
    models: Sequence[dict],
    positions_z: np.ndarray,
    frequencies_hz: Sequence[float],
    dip_degs: Sequence[float],
    tr_spacings_m: Sequence[float],
) -> int:
    """Delega a comparação numba×jax ao helper compartilhado (DRY).

    Mantido como wrapper fino sobre
    :func:`geosteering_ai.cli._exec.run_compare_backends` para preservar a
    assinatura local usada por :func:`handle_simulate`.

    Args:
        args: namespace do subcomando (usa workers/threads/dtype/jax_strategy/
            warmup/as_json/quiet/n_pos).
        models/positions_z/frequencies_hz/dip_degs/tr_spacings_m: contexto.

    Returns:
        Exit code propagado de ``run_compare_backends`` (0 | 1).
    """
    from geosteering_ai.cli._exec import run_compare_backends

    return run_compare_backends(
        models=list(models),
        positions_z=positions_z,
        frequencies_hz=frequencies_hz,
        dip_degs=dip_degs,
        tr_spacings_m=tr_spacings_m,
        n_pos=args.n_pos,
        workers=args.workers,
        threads=args.threads,
        dtype=args.dtype,
        jax_strategy=args.jax_strategy,
        warmup=args.warmup,
        as_json=args.as_json,
        quiet=args.quiet,
        title="COMPARAÇÃO DE BACKENDS — simulate",
    )


def handle_simulate(args: argparse.Namespace) -> int:
    """Handler do subcomando ``simulate``.

    Constrói N modelos randomizados, resolve o backend (numba/jax), executa
    ``--repeat`` rodadas hot (mediana do throughput), apresenta a tabela ASCII
    (+ JSON opcional) e grava ``--out`` no ``--format`` escolhido.

    Args:
        args: ``argparse.Namespace`` com os campos do subparser ``simulate``
            (models, n_pos, workers, threads, seed, frequencies, dips,
            tr_spacings, out, out_format, quiet, backend, dtype, jax_strategy,
            warmup, as_json, repeat, compare_backends).

    Returns:
        Exit code: 0 (sucesso) | 1 (erro tratado: ValueError/RuntimeError/OSError).

    Raises:
        Não propaga — falhas viram exit code 1 com ``logger.error(exc_info=True)``.

    Note:
        O caminho Numba preserva ``simulate_multi(models=...)`` (pool de workers)
        SEM regressão; o caminho JAX usa o dispatcher parity-tested
        ``simulate_batch``. Imports pesados são lazy (via ``_exec``).

    Example:
        $ geosteering-cli simulate --models 8 --n-pos 100 --backend numba
        $ geosteering-cli simulate --models 64 --backend jax --dtype complex64
        $ geosteering-cli simulate --models 3 --out /tmp/g --format dat
        $ geosteering-cli simulate --models 8 --compare-backends --json
    """
    t_total_start = perf_counter()  # v2.56 — cobre lazy imports + build + warmup + runs
    from geosteering_ai.cli._exec import (
        finitude_stats,
        resolve_backend_preflight,
        resolve_jax_chunk_size,
        resolve_requested_backend,
        run_once,
        warmup_backend,
    )
    from geosteering_ai.cli._hwinfo import collect_hardware_info

    frequencies_hz = _parse_float_list(getattr(args, "frequencies", None), [20000.0])
    dip_degs = _parse_float_list(getattr(args, "dips", None), [0.0])
    tr_spacings_m = _parse_float_list(getattr(args, "tr_spacings", None), [1.0])
    positions_z = np.linspace(-5.0, 5.0, args.n_pos).astype(np.float64)

    if not args.quiet:
        logger.info(
            "Gerando %d modelos sintéticos (seed=%d, n_pos=%d)...",
            args.models,
            args.seed,
            args.n_pos,
        )
    models = _build_random_models(
        args.models,
        args.seed,
        geometry=getattr(args, "geometry", "per-model"),
        n_geometries=getattr(args, "n_geometries", None),
        quantize_step=getattr(args, "quantize_step", None),
    )

    # ── --compare-backends: caminho dedicado (numba vs jax) ──────────────
    if getattr(args, "compare_backends", False):
        return _run_compare_backends(
            args, models, positions_z, frequencies_hz, dip_degs, tr_spacings_m
        )

    # ── Resolve backend com PRÉ-VOO de geometria (v2.55) + warmup opcional ─
    # Normaliza o solicitado (None→numba+DeprecationWarning; spec 0003) e então
    # o pré-voo conta os grupos de geometria com NumPy puro ANTES de tocar JAX:
    # se não-agrupável (ou auto resolve numba), roda Numba SEM iniciar o CUDA.
    requested_backend = resolve_requested_backend(args)
    backend, device, preflight_reason = resolve_backend_preflight(
        requested_backend, models, quiet=args.quiet
    )
    # Auto-chunk anti-OOM (v2.56) — high-config jax fragmenta o eixo de modelos.
    n_configs = len(frequencies_hz) * len(tr_spacings_m) * len(dip_degs)
    jax_chunk = resolve_jax_chunk_size(
        backend, n_configs, explicit=getattr(args, "jax_chunk_size", None)
    )
    # Warmup cronometrado (v2.56): SKIP warmup_backend no jax (shape errado, peso
    # morto); manter no numba (opt-in via --warmup).
    t_warmup_start = perf_counter()
    if args.warmup and backend != "jax":
        warmup_backend(
            backend,
            n_pos=args.n_pos,
            dtype=args.dtype,
            jax_strategy=args.jax_strategy,
            dip_degs=dip_degs,
            tr_spacings_m=tr_spacings_m,
            freqs_hz=frequencies_hz,
        )
    warmup_s = perf_counter() - t_warmup_start

    # ── Execução cronometrada (--repeat rodadas hot → mediana) ───────────
    repeat = max(1, int(args.repeat))
    throughputs: List[float] = []
    elapseds: List[float] = []
    last_H6: np.ndarray | None = None
    last_groups: int | None = None
    effective_backend = backend
    # Motivo inicial = pré-voo (jax→numba por geometria); o dispatcher pode
    # sobrescrever se houver fallback adicional no run.
    fallback_reason: str | None = preflight_reason
    try:
        for _ in range(repeat):
            H6, elapsed, groups, eff, reason = run_once(
                backend,
                models,
                positions_z,
                frequencies_hz=frequencies_hz,
                dip_degs=dip_degs,
                tr_spacings_m=tr_spacings_m,
                workers=args.workers,
                threads=args.threads,
                dtype=args.dtype,
                jax_strategy=args.jax_strategy,
                jax_chunk_size_models=jax_chunk,
            )
            throughputs.append((args.models / elapsed) * 3600.0 if elapsed > 0 else 0.0)
            elapseds.append(elapsed)
            last_H6, last_groups = H6, groups
            effective_backend = eff
            # Fallback do dispatcher (eff != backend) sobrescreve; senão preserva
            # o motivo do pré-voo (preflight_reason já em fallback_reason).
            if eff != backend:
                fallback_reason = reason
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Erro durante simulação: %s", exc, exc_info=True)
        return 1

    if last_H6 is None:  # pragma: no cover — repeat>=1 garante uma rodada
        logger.error("Nenhuma rodada executada.")
        return 1

    thr_median = statistics.median(throughputs)
    thr_std = statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    elapsed_median = statistics.median(elapseds)  # v2.56 — coerente com thr_median
    total_s = perf_counter() - t_total_start
    # device EFETIVO: se o jax caiu para numba (geometria não-agrupável), é CPU.
    effective_device = "gpu" if effective_backend == "jax" else "cpu"

    stats = _build_stats(
        backend=effective_backend,
        device=effective_device,
        throughput_mod_h=thr_median,
        throughput_std=thr_std,
        elapsed_s=elapsed_median,
        repeat=repeat,
        n_models=args.models,
        n_pos=args.n_pos,
        frequencies_hz=frequencies_hz,
        dip_degs=dip_degs,
        tr_spacings_m=tr_spacings_m,
        workers=args.workers,
        threads=args.threads,
        dtype=args.dtype,
        jax_strategy=args.jax_strategy,
        n_geometry_groups=last_groups,
        finitude=finitude_stats(last_H6),
        reason=fallback_reason,
        warmup_s=warmup_s,
        total_s=total_s,
    )
    hw = collect_hardware_info(want_gpu=(backend == "jax"))
    _emit_results(args, stats, hw)

    # ── Gravação --out (--format npz|dat; none = só tabela) ──────────────
    if args.out is not None and args.out_format != "none":
        try:
            _save_simulation_output(
                Path(args.out),
                args.out_format,
                last_H6,
                models,
                positions_z,
                frequencies_hz,
                dip_degs,
                quiet=args.quiet,
            )
        except OSError as exc:
            logger.error("Erro ao gravar arquivo de saída: %s", exc)
            return 1

    return 0
