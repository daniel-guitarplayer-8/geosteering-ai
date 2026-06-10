# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/simulate.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Subcomando `simulate` da CLI                               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.30 — multi-dim)                         ║
# ║  Versão      : v2.30                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-10                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : argparse + simulate_multi (geosteering_ai.simulation)      ║
# ║  Dependências: numpy, geosteering_ai.simulation (lazy import dentro       ║
# ║                de handle_simulate para acelerar `--help`)                 ║
# ║  Padrão      : Hexagonal — adapter externo do simulate_multi              ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementa o handler do subcomando ``simulate``: gera N modelos       ║
# ║    sintéticos 1D 5-camada com perfis randomizados (rho_h, rho_v, esp)    ║
# ║    e executa ``simulate_multi`` em batch. Reutiliza a infraestrutura      ║
# ║    de auto-detect de paralelismo da Sprint v2.23 A.2                      ║
# ║    (``recommend_default_parallelism``).                                    ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    handle_simulate: handler do subcomando, recebe argparse.Namespace     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Subcomando ``simulate`` da CLI Geosteering AI (Sprint v2.24 — I2.6).

Gera modelos sintéticos com perfis 1D randomizados e executa
``simulate_multi`` em batch. Reutiliza a infraestrutura de auto-detect
de paralelismo da Sprint v2.23 A.2 (``recommend_default_parallelism``).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

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
    geometry: str = "per-model",
    n_geometries: int | None = None,
) -> list[dict]:
    """Gera N modelos sintéticos 1D 5-camada para smoke/benchmark.

    Cada modelo é um dict ``{rho_h, rho_v, esp}`` com:
      - 5 camadas (3 internas + 2 semi-espaços)
      - resistividades em [1.0, 100.0] Ω·m
      - razão TIV ``rho_v / rho_h`` em [1.0, 3.0]
      - espessuras das camadas internas em [2.0, 10.0] m

    O parâmetro ``geometry`` controla a BATCHABILIDADE da ``esp`` no JAX
    (espelha ``data.synthetic_generator``):

      ┌──────────────┬──────────────────────────────────────────────────────┐
      │ per-model    │ esp ÚNICA por modelo (stream rng LEGADO, bit-idêntico  │
      │  (default)   │   p/ seeds existentes) → cada modelo = 1 grupo → no    │
      │              │   JAX cai p/ Numba (não-agrupável)                     │
      │ templates    │ K geometrias DISTINTAS replicadas round-robin →        │
      │              │   POUCOS grupos GRANDES → vmap satura a GPU            │
      │ quantized    │ esp arredondada → muitos modelos compartilham → parcial│
      └──────────────┴──────────────────────────────────────────────────────┘

    Args:
        n_models: número de modelos a gerar.
        seed: semente do gerador (reprodutibilidade).
        geometry: ``"per-model"`` | ``"templates"`` | ``"quantized"`` (amostragem
            de ``esp`` via :func:`geosteering_ai.cli._exec.sample_geometry`).
        n_geometries: (só ``templates``) K geometrias distintas; None = auto.

    Returns:
        Lista de dicts ``[{rho_h, rho_v, esp}, ...]``, com cada array
        em ``np.float64`` (float32 não é suportado pelo simulador
        Fortran-equivalente).

    Note:
        ``per-model`` preserva o stream rng legado (esp sorteada no loop após
        rho_h/rho_v) — zero surpresa p/ seeds existentes. ``templates``/
        ``quantized`` sorteiam a ``esp`` em bloco (via ``sample_geometry``) e o
        ``rho`` por modelo (formação fixa, resistividades variadas).

    Example:
        >>> models = _build_random_models(2, seed=42)
        >>> len(models)
        2
        >>> sorted(models[0].keys())
        ['esp', 'rho_h', 'rho_v']
    """
    rng = np.random.default_rng(seed)
    if geometry == "per-model":
        # Stream LEGADO (bit-idêntico): rho_h, rho_v=rho_h·ratio, esp — POR modelo.
        models = []
        for _ in range(n_models):
            rho_h = rng.uniform(1.0, 100.0, size=5).astype(np.float64)
            rho_v = rho_h * rng.uniform(1.0, 3.0, size=5).astype(np.float64)
            esp = rng.uniform(2.0, 10.0, size=3).astype(np.float64)
            models.append({"rho_h": rho_h, "rho_v": rho_v, "esp": esp})
        return models

    # templates/quantized — ``esp`` em bloco (compartilhada); ``rho`` por modelo.
    from geosteering_ai.cli._exec import sample_geometry

    esp_batch = sample_geometry(
        rng, n_models, 3, mode=geometry, n_geometries=n_geometries
    )
    models = []
    for i in range(n_models):
        rho_h = rng.uniform(1.0, 100.0, size=5).astype(np.float64)
        rho_v = rho_h * rng.uniform(1.0, 3.0, size=5).astype(np.float64)
        models.append({"rho_h": rho_h, "rho_v": rho_v, "esp": esp_batch[i]})
    return models


def handle_simulate(args: argparse.Namespace) -> int:
    """Handler do subcomando ``simulate``.

    Constrói N modelos randomizados, executa ``simulate_multi`` em
    batch (com paralelismo auto-detectado quando args não especificam),
    e opcionalmente grava o tensor H em ``.npz``.

    Args:
        args: ``argparse.Namespace`` com campos:

            - ``models`` (int): número de modelos a gerar/simular
            - ``n_pos`` (int): pontos de profundidade por modelo
            - ``workers`` (int | None): workers paralelos
            - ``threads`` (int | None): threads Numba por worker
            - ``seed`` (int): semente do gerador
            - ``frequencies`` (str | None): CSV de Hz (ex: ``"2000,20000"``)
            - ``dips`` (str | None): CSV de graus (ex: ``"0,15,30"``)
            - ``tr_spacings`` (str | None): CSV de metros (ex: ``"0.5,1.0"``)
            - ``out`` (str | None): diretório de saída (None = memória)
            - ``quiet`` (bool): suprime logs informativos

    Returns:
        Exit code:

        - 0: sucesso (todos os modelos simulados e gravados)
        - 1: erro tratado (ValueError, RuntimeError, OSError)

    Raises:
        Não propaga exceções — todas as falhas viram exit code 1
        com ``logger.error(..., exc_info=True)``.

    Note:
        Logging é configurado em ``main.py`` (centralizado). Aqui
        apenas obtemos o logger do módulo. Imports pesados de
        simulação são lazy para preservar ``--help`` rápido.

    Example:
        Em produção via CLI::

            $ geosteering-cli simulate --models 10 --n-pos 100 --quiet
            $ geosteering-cli simulate --frequencies 2000,20000 --dips 0,15,30
            $ geosteering-cli simulate --tr-spacings 0.5,1.0,1.5 --models 50
    """
    # ── Motor compartilhado (cli._exec) — wiring p/ AMBOS os simuladores
    #    (Numba JIT + JAX GPU) via o pré-voo TLS-safe da spec 0003. Imports
    #    leves (numpy puro); a simulação pesada é lazy DENTRO de run_once.
    from geosteering_ai.cli._exec import (
        finitude_stats,
        resolve_backend_preflight,
        resolve_jax_chunk_size,
        resolve_requested_backend,
        run_compare_backends,
        run_once,
        warmup_backend,
    )

    t_handler = time.perf_counter()  # p/ tempo total (handler) na tabela ASCII

    # ── Parâmetros multi-dim (Sprint v2.30) + flags (spec 0003 + v2.53-v2.56) ──
    frequencies_hz = _parse_float_list(getattr(args, "frequencies", None), [20000.0])
    dip_degs = _parse_float_list(getattr(args, "dips", None), [0.0])
    tr_spacings_m = _parse_float_list(getattr(args, "tr_spacings", None), [1.0])
    n_pos = int(args.n_pos)
    positions_z = np.linspace(-5.0, 5.0, n_pos).astype(np.float64)
    n_models = int(args.models)
    geometry = getattr(args, "geometry", "per-model")
    repeat = max(1, int(getattr(args, "repeat", 1) or 1))
    dtype = getattr(args, "dtype", "complex128")
    jax_strategy = getattr(args, "jax_strategy", "bucketed")
    quiet = bool(getattr(args, "quiet", False))
    as_json = bool(getattr(args, "json", False))
    workers = getattr(args, "workers", None)
    threads = getattr(args, "threads", None)

    if not quiet:
        logger.info(
            "Gerando %d modelos sintéticos (seed=%d, n_pos=%d, geometry=%s)...",
            n_models,
            args.seed,
            n_pos,
            geometry,
        )
    try:
        models = _build_random_models(n_models, args.seed, geometry=geometry)
    except ValueError as exc:
        logger.error("Erro ao gerar modelos: %s", exc, exc_info=True)
        return 1

    # ── --compare-backends: numba × jax lado-a-lado (DRY com benchmark) ───────
    if bool(getattr(args, "compare_backends", False)):
        return run_compare_backends(
            models=models,
            positions_z=positions_z,
            frequencies_hz=frequencies_hz,
            dip_degs=dip_degs,
            tr_spacings_m=tr_spacings_m,
            n_pos=n_pos,
            workers=workers,
            threads=threads,
            dtype=dtype,
            jax_strategy=jax_strategy,
            warmup=bool(getattr(args, "warmup", False)),
            as_json=as_json,
            quiet=quiet,
            title="simulate --compare-backends",
        )

    # ── Resolve o backend com PRÉ-VOO TLS-safe (spec 0003) ────────────────────
    # jax + geometria não-agrupável → numba SEM tocar CUDA (anti-crash TLS);
    # auto → árvore medida do dispatcher; numba → numba. ``--backend`` ausente
    # → numba + DeprecationWarning (default mudará p/ auto).
    requested = resolve_requested_backend(args)
    backend, device, reason = resolve_backend_preflight(requested, models, quiet=quiet)
    n_configs = len(frequencies_hz) * len(tr_spacings_m) * len(dip_degs)
    jax_chunk = resolve_jax_chunk_size(
        backend, n_configs, explicit=getattr(args, "jax_chunk_size", None)
    )
    if not quiet:
        logger.info(
            "Backend: %s (device=%s) · %d modelos × %d pos × %d freq × %d dips × %d TR",
            backend,
            device,
            n_models,
            n_pos,
            len(frequencies_hz),
            len(dip_degs),
            len(tr_spacings_m),
        )

    # ── Warmup opcional (JIT/XLA) antes da medição ───────────────────────────
    warmup_s = 0.0
    if bool(getattr(args, "warmup", False)):
        t_warm = time.perf_counter()
        warmup_backend(
            backend,
            n_pos=n_pos,
            dtype=dtype,
            jax_strategy=jax_strategy,
            dip_degs=dip_degs,
            tr_spacings_m=tr_spacings_m,
            freqs_hz=frequencies_hz,
        )
        warmup_s = time.perf_counter() - t_warm

    # ── Rodadas cronometradas (--repeat) — guarda o MELHOR tempo ──────────────
    timings: list[float] = []
    h6 = None
    effective = backend
    eff_reason = reason
    n_groups = None
    try:
        for _ in range(repeat):
            h6, elapsed, n_groups, effective, run_reason = run_once(
                backend,
                models,
                positions_z,
                frequencies_hz=frequencies_hz,
                dip_degs=dip_degs,
                tr_spacings_m=tr_spacings_m,
                workers=workers,
                threads=threads,
                dtype=dtype,
                jax_strategy=jax_strategy,
                jax_chunk_size_models=jax_chunk,
            )
            timings.append(elapsed)
            if run_reason:
                eff_reason = run_reason
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Erro durante simulação: %s", exc, exc_info=True)
        return 1

    best = min(timings) if timings else 0.0
    throughput = (n_models / best) * 3600.0 if best > 0 else 0.0
    total_s = time.perf_counter() - t_handler
    fin = finitude_stats(h6) if h6 is not None else {}

    # ── Saída JSON (backend EFETIVO concreto — nunca 'auto'; AC-4.1) ──────────
    if as_json:
        import json

        payload = {
            "command": "simulate",
            "models": n_models,
            "n_pos": n_pos,
            "geometry": geometry,
            "backend": effective,
            "device": device,
            "repeat": repeat,
            "best_elapsed_s": best,
            "warmup_s": warmup_s,
            "total_s": total_s,
            "throughput_mod_h": throughput,
            "n_freq": len(frequencies_hz),
            "n_dips": len(dip_degs),
            "n_tr": len(tr_spacings_m),
            "n_geometry_groups": n_groups,
            "finitude": fin or None,
        }
        if eff_reason:
            payload["reason"] = eff_reason
        print(json.dumps(payload))  # stdout limpo (logs vão p/ stderr)
    elif not quiet:
        # ── Tabela ASCII de resultados (devolve a apresentação do CLI rico) ───
        # Reusa os renderizadores PUROS (_table/_hwinfo). stdout = a tabela; os
        # logs de progresso já foram p/ stderr (D9 — exceção de CLI documentada).
        from geosteering_ai.cli._hwinfo import collect_hardware_info
        from geosteering_ai.cli._table import build_result_rows, render_kv_table

        hw = collect_hardware_info(want_gpu=(effective == "jax"))
        stats = {
            "backend": effective,
            "device": device,
            "reason": eff_reason,
            "throughput_mod_h": throughput,
            "elapsed_s": best,
            "repeat": repeat,
            "warmup_s": warmup_s if warmup_s > 0 else None,
            "total_s": total_s,
            "n_models": n_models,
            "n_pos": n_pos,
            "n_freqs": len(frequencies_hz),
            "n_dips": len(dip_degs),
            "n_trs": len(tr_spacings_m),
            "dtype": dtype,
            "jax_strategy": jax_strategy,
            "n_geometry_groups": n_groups,
            "workers": workers,
            "threads": threads,
            "nan_count": fin.get("nan_count"),
            "inf_count": fin.get("inf_count"),
            "all_finite": fin.get("all_finite"),
        }
        print(render_kv_table("SIMULATE — RESULTADO", build_result_rows(stats, hw)))

    # ── Gravação (--out + --format npz|dat|none) ──────────────────────────────
    fmt = getattr(args, "format", "npz")
    if args.out is not None and fmt != "none" and h6 is not None:
        try:
            _save_results(
                args.out, fmt, h6, models, positions_z, frequencies_hz, dip_degs, n_pos
            )
            if not quiet:
                logger.info("Resultados (%s) gravados em: %s", fmt, args.out)
        except OSError as exc:
            logger.error("Erro ao gravar arquivo de saída: %s", exc)
            return 1

    return 0


def _save_results(
    out: str,
    fmt: str,
    h6: np.ndarray,
    models: list[dict],
    positions_z: np.ndarray,
    frequencies_hz: list[float],
    dip_degs: list[float],
    n_pos: int,
) -> None:
    """Grava o tensor H em ``.npz`` ou ``.dat`` (22-col Fortran-compat) + ``.out``.

    ``dat`` segue o layout de ``geosteering-physics.md`` §4 (col1=zobs, col2/3=
    res_h/res_v REAIS na camada de cada z, col4-21=Re/Im Hxx..Hzz). Reusa os
    escritores canônicos (``simulation.io.tensor_dat``) e o mapeamento z→camada
    de ``cli._exec`` — zero recomputação de física (só serializa o H6 já pronto).

    Args:
        out: diretório de saída (criado se ausente).
        fmt: ``"npz"`` (tensor H) ou ``"dat"`` (22-col binário + .out ASCII).
        h6: tensor ``(n_models, nTR, nAng, n_pos, nf, 9)`` complexo.
        models: lista de dicts ``{rho_h, rho_v, esp}`` (para col2/3 do .dat).
        positions_z: ``(n_pos,)`` profundidades (m).
        frequencies_hz / dip_degs: eixos (para o cabeçalho .out).
        n_pos: nº de posições por modelo.

    Raises:
        OSError: falha de I/O (o handler converte em exit code 1).
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "dat":
        from geosteering_ai.cli._exec import models_to_batch, rho_at_obs_from_batch
        from geosteering_ai.simulation.io.tensor_dat import (
            write_dat_from_tensor,
            write_out_file,
        )

        n_models = len(models)
        n_ang = len(dip_degs)
        rho_h_b, rho_v_b, esp_b = models_to_batch(models)
        rho_h_obs, rho_v_obs = rho_at_obs_from_batch(
            positions_z, rho_h_b, rho_v_b, esp_b
        )
        # Broadcast no eixo de ângulos (write_dat_from_tensor espera (n,nAng,n_pos)).
        rho_h_obs_b = np.broadcast_to(rho_h_obs[:, None, :], (n_models, n_ang, n_pos))
        rho_v_obs_b = np.broadcast_to(rho_v_obs[:, None, :], (n_models, n_ang, n_pos))
        write_dat_from_tensor(
            str(out_dir / "simulate_results.dat"),
            h6,
            positions_z,
            rho_h_at_obs=rho_h_obs_b,
            rho_v_at_obs=rho_v_obs_b,
        )
        write_out_file(
            str(out_dir / "simulate_results.out"),
            n_dips=n_ang,
            n_freqs=len(frequencies_hz),
            nmaxmodel=n_models,
            angles=list(dip_degs),
            freqs_hz=list(frequencies_hz),
            nmeds_per_angle=[n_pos] * n_ang,
        )
    else:  # npz — tensor H REAL (fix v2.53: era repr(result))
        np.savez(out_dir / "simulate_results.npz", H=h6)
