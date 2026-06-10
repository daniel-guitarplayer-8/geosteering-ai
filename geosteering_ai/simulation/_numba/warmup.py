# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_numba/warmup.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend Numba JIT CPU (warmup)         ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-01 (v2.52 — redução de warmup Numba JIT CPU)      ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : Numba 0.65.x (LLVM JIT)                                    ║
# ║  Dependências: numba (opcional — skip gracioso), multi_forward            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Pré-compila o caminho de PRODUÇÃO do simulador Numba CPU              ║
# ║    (`simulate_multi(backend="numba")`) ANTES do workload real,           ║
# ║    eliminando o cold-start de compilação LLVM na 1ª simulação. Aquece     ║
# ║    os kernels `parallel=True` (prange) que DOMINAM o tempo de produção:   ║
# ║    `_simulate_combined_prange_flat`, `_simulate_positions_njit_cached`,   ║
# ║    `precompute_common_arrays_cache` (+ dipolos/propagação INLINADOS).     ║
# ║                                                                           ║
# ║  GAP RESOLVIDO (v2.52)                                                    ║
# ║    O warmup legado `_warmup_numba_tier2_sync` aquece só hmd_tiv/vmd via   ║
# ║    o caminho de CALLBACK JAX (3 posições) → (a) REQUER JAX; (b) NUNCA     ║
# ║    aquece os batch kernels paralelos. Este módulo fecha o gap E é         ║
# ║    JAX-INDEPENDENTE (roda em deploy Numba-only).                          ║
# ║                                                                           ║
# ║  DESIGN (2 chamadas — dispatch mutuamente exclusiva)                      ║
# ║    A dispatch de `simulate_multi` é exclusiva em `n_combos`: multi-combo  ║
# ║    (≥2 TR) → prange-flat; single-combo (1 TR) → cached. Logo aquecemos    ║
# ║    AMBOS com 2 chamadas tiny. Arrays via `np.full`/`np.linspace`          ║
# ║    (writeable, C-contíguos) → nenhuma 2ª especialização (guard verde).    ║
# ║                                                                           ║
# ║  INVIOLÁVEL                                                               ║
# ║    O warmup SÓ RODA kernels EXISTENTES — não altera numérica nem o        ║
# ║    hot-path. Paridade Fortran <1e-6 (Numba é a referência) preservada.   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Warmup Numba-nativo do simulador CPU — pré-compila o caminho de produção.

Expõe :func:`warmup_numba_simulator` (parametrizável) e
:func:`warmup_numba_simulator_from_config` (deriva de um ``SimulationConfig``)
para pré-aquecer os kernels JIT Numba (incluindo os `parallel=True`/prange que
dominam produção) + popular o cache de disco `.nbc`, de modo que a 1ª simulação
real não pague o cold-start de compilação LLVM. JAX-INDEPENDENTE.

Example:
    Aquecer o caminho Numba de produção::

        >>> from geosteering_ai.simulation._numba.warmup import warmup_numba_simulator
        >>> info = warmup_numba_simulator(n_layers=5, n_positions=200, verbose=True)
        >>> "_simulate_combined_prange_flat" in info["functions_warmed"]
        True

Note:
    Os dipolos (`hmd_tiv`/`vmd`) são chamados njit-a-njit (INLINADOS) dentro dos
    kernels prange → seu código É compilado (e persiste no `.nbc` do kernel pai),
    mas o ``.signatures`` do dispatcher standalone permanece 0 (só popula em
    entrada via Python — caminho de callback JAX). A métrica de cobertura usa os
    kernels Python-dispatcháveis (flat/cached/precompute).
    Ref: docs/reports/warmup_analysis_jit_2026-05-12.md.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.simulation.config import SimulationConfig

logger = logging.getLogger(__name__)

__all__ = [
    "warmup_numba_simulator",
    "warmup_numba_simulator_from_config",
]

# ── Valores sintéticos do modelo de warmup ────────────────────────────────
# Modelo isotrópico homogêneo (espelha _jax/warmup.py): rho idêntico + esp
# idêntico. np.full/np.linspace garantem arrays WRITEABLE C-contíguos →
# mesma assinatura de produção → nenhuma 2ª especialização de dipolo.
_WARMUP_RHO_OHM_M: float = 10.0
_WARMUP_ESP_M: float = 5.0


def _tracked_dispatchers() -> dict:
    """Dispatchers `@njit` Python-dispatcháveis rastreados (`.signatures`).

    Returns:
        ``dict[str, Dispatcher]`` dos kernels paralelos de produção cujo
        ``.signatures`` popula em entrada via Python (os que o warmup compila):
        ``_simulate_combined_prange_flat``, ``_simulate_positions_njit_cached``,
        ``_simulate_positions_njit_cached_tiled`` (só com ``use_tiled_positions``),
        ``precompute_common_arrays_cache``. Os dipolos (inlinados) ficam de fora
        da métrica (ver NOTE do módulo).
    """
    from geosteering_ai.simulation._numba.kernel import precompute_common_arrays_cache
    from geosteering_ai.simulation.forward import (
        _simulate_combined_prange_flat,
        _simulate_positions_njit_cached,
        _simulate_positions_njit_cached_tiled,
    )

    return {
        "_simulate_combined_prange_flat": _simulate_combined_prange_flat,
        "_simulate_positions_njit_cached": _simulate_positions_njit_cached,
        "_simulate_positions_njit_cached_tiled": _simulate_positions_njit_cached_tiled,
        "precompute_common_arrays_cache": precompute_common_arrays_cache,
    }


def warmup_numba_simulator(
    *,
    n_layers: int = 5,
    n_positions: int = 200,
    n_models: int = 1,
    frequencies_hz: Sequence[float] = (20000.0,),
    tr_spacings_m: Sequence[float] = (1.0,),
    dip_degs: Sequence[float] = (0.0,),
    threads: Optional[int] = None,
    hankel_filter: str = "werthmuller_201pt",
    complex_dtype: str = "complex128",
    base_cfg: "Optional[SimulationConfig]" = None,
    verbose: bool = False,
) -> dict:
    """Pré-compila o caminho de produção do simulador Numba CPU.

    Dispara DUAS chamadas ``simulate_multi(backend="numba")`` sintéticas que
    forçam a compilação LLVM dos kernels `parallel=True`/prange + a árvore de
    funções inlinadas (dipolos, propagação, geometria, Hankel), populando o
    cache de disco `.nbc`.

    Fluxo:
      ┌────────────────────────────────────────────────────────────────────┐
      │  HAS_NUMBA? não → skip gracioso ({skipped: True})                   │
      │     ↓ sim                                                            │
      │  modelo sintético homogêneo (rho=10, esp=5; arrays writeable)       │
      │     ↓                                                                │
      │  snapshot .signatures (antes)                                        │
      │     ↓                                                                │
      │  (1) simulate_multi multi-combo (tr=2) → prange-flat + precompute   │
      │  (2) simulate_multi single-combo (tr=1) → cached                    │
      │     ↓  → compila + persiste .nbc                                    │
      │  snapshot (depois) → functions_warmed                                │
      └────────────────────────────────────────────────────────────────────┘

    Args:
        n_layers: Camadas do modelo (inclui 2 semi-espaços). Default 5.
        n_positions: Posições do grid (>1 p/ ativar prange). Default 200.
        n_models: Reservado (caminho single-model in-process; sem pool). Default 1.
        frequencies_hz: Frequências (Hz) a aquecer. Default ``(20000.0,)``.
        tr_spacings_m: TR base; o warmup adiciona um 2º TR internamente p/ cobrir
            o prange-flat (multi-combo). Default ``(1.0,)``.
        dip_degs: Ângulos dip (°). Default ``(0.0,)``.
        threads: Nº de threads (via ``cfg.num_threads``; NÃO seta ``NUMBA_NUM_THREADS``).
            ``None`` = auto (-1). O thread-count é máscara runtime, não assinatura de
            compile → não causa recompile.
        hankel_filter: Filtro Hankel. Default werthmuller_201pt.
        complex_dtype: ``"complex128"`` (paridade, default) ou ``"complex64"``.
        base_cfg: :class:`SimulationConfig` de produção opcional. Se fornecido, é
            reusado (via ``dataclasses.replace(cfg, backend="numba")``) — honra os
            SELETORES de kernel (``use_flat_prange``/``use_tiled_positions``/
            ``parallel``) → aquece o kernel EXATO de produção; ``n_positions`` vem
            do cfg. ``None`` (default) → cfg mínimo dos kwargs (caminho default:
            flat + cached não-tiled).
        verbose: Loga timing + kernels aquecidos.

    Returns:
        ``dict`` com:
          - ``skipped`` (bool), ``reason`` (str|None) — skip se Numba ausente;
          - ``functions_warmed`` (list[str]) — kernels com ≥1 assinatura pós-warmup;
          - ``n_signatures`` (dict[str,int]) — contagem por kernel rastreado;
          - ``elapsed_s`` (float) — tempo de warmup;
          - ``threads`` (int) — ``numba.get_num_threads()`` em runtime;
          - ``cache_dir`` (str|None) — ``NUMBA_CACHE_DIR``;
          - ``n_layers``/``n_positions``/``n_models`` (int).

    Raises:
        ValueError: Se ``n_layers < 2`` ou ``n_positions < 2`` ou ``n_models < 1``.

    Example:
        >>> info = warmup_numba_simulator(n_layers=5, n_positions=200)
        >>> info["skipped"]
        False

    Note:
        Os dipolos (`hmd_tiv`/`vmd`) são INLINADOS njit-a-njit → não aparecem em
        ``functions_warmed`` (seu código compila dentro de flat/cached). NÃO os
        chame direto daqui (risco de 2ª especialização). See Also:
        :func:`warmup_numba_simulator_from_config`.
    """
    # Se `base_cfg` foi passado (caminho from_config), o grid e os campos de
    # seleção de kernel (use_flat_prange/use_tiled_positions/parallel) vêm DELE
    # → aquece o kernel EXATO de produção (não só o default).
    if base_cfg is not None:
        n_positions = int(base_cfg.n_positions)
    if n_layers < 2:
        raise ValueError(f"n_layers deve ser >= 2 (recebido {n_layers}).")
    if n_positions < 10:
        # SimulationConfig exige n_positions >= 10 (config.py::__post_init__);
        # o prange já dispara com n_pos>1, mas casamos o mínimo de produção.
        raise ValueError(f"n_positions deve ser >= 10 (recebido {n_positions}).")
    if n_models < 1:
        raise ValueError(f"n_models deve ser >= 1 (recebido {n_models}).")

    # ── Guard: Numba ausente → skip gracioso ─────────────────────────────────
    from geosteering_ai.simulation._numba import HAS_NUMBA

    if not HAS_NUMBA:
        logger.info("warmup_numba_simulator: Numba ausente — skip gracioso.")
        return {
            "skipped": True,
            "reason": "numba_absent",
            "functions_warmed": [],
            "n_signatures": {},
            "elapsed_s": 0.0,
            "threads": 0,
            "cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
            "n_layers": n_layers,
            "n_positions": n_positions,
            "n_models": n_models,
        }

    import numba

    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.multi_forward import simulate_multi

    # ── Modelo sintético HOMOGÊNEO (arrays WRITEABLE C-contíguos) ────────────
    n_esp = max(n_layers - 2, 0)
    rho_h = np.full(n_layers, _WARMUP_RHO_OHM_M, dtype=np.float64)
    rho_v = np.full(n_layers, _WARMUP_RHO_OHM_M, dtype=np.float64)
    esp = np.full(n_esp, _WARMUP_ESP_M, dtype=np.float64)
    total_thick = _WARMUP_ESP_M * n_esp if n_esp > 0 else _WARMUP_ESP_M
    positions_z = np.linspace(-1.0, total_thick + 1.0, n_positions)

    # Config: se `base_cfg` foi passado, reusa-o (preservando os seletores de
    # kernel) forçando só backend="numba"; senão constrói o mínimo dos kwargs.
    if base_cfg is not None:
        import dataclasses

        cfg = dataclasses.replace(base_cfg, backend="numba")
    elif threads is not None:
        # `num_threads` condicional (não-default só quando o caller pede threads).
        cfg = SimulationConfig(
            backend="numba",
            dtype=complex_dtype,
            hankel_filter=hankel_filter,
            n_positions=int(n_positions),
            num_threads=int(threads),
        )
    else:
        cfg = SimulationConfig(
            backend="numba",
            dtype=complex_dtype,
            hankel_filter=hankel_filter,
            n_positions=int(n_positions),
        )

    freqs = list(frequencies_hz)
    tr_base = list(tr_spacings_m)
    dips = list(dip_degs)

    tracked = _tracked_dispatchers()
    t0 = time.perf_counter()

    # Numba `simulate_multi` é SÍNCRONO — a 1ª chamada compila+roda (bloqueante);
    # não há sync lazy como no JAX. As 2 chamadas abaixo cobrem ambos os branches.
    # (1) Multi-combo (2 TR distintos) → _simulate_combined_prange_flat + precompute.
    tr_multi = tr_base + [tr_base[0] * 2.0] if len(tr_base) == 1 else tr_base
    simulate_multi(
        rho_h,
        rho_v,
        esp,
        positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=tr_multi,
        dip_degs=dips,
        cfg=cfg,
    )

    # (2) Single-combo (1 TR) → _simulate_positions_njit_cached.
    simulate_multi(
        rho_h,
        rho_v,
        esp,
        positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=[tr_base[0]],
        dip_degs=[dips[0]],
        cfg=cfg,
    )

    elapsed_s = time.perf_counter() - t0
    sig_after = {name: len(fn.signatures) for name, fn in tracked.items()}
    functions_warmed = [name for name, c in sig_after.items() if c > 0]

    try:
        run_threads = int(numba.get_num_threads())
    except Exception:  # noqa: BLE001 — diagnóstico best-effort
        run_threads = -1

    info = {
        "skipped": False,
        "reason": None,
        "functions_warmed": functions_warmed,
        "n_signatures": sig_after,
        "elapsed_s": elapsed_s,
        "threads": run_threads,
        "cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        "n_layers": n_layers,
        "n_positions": int(n_positions),
        "n_models": n_models,
    }
    if verbose:
        logger.info(
            "[warmup-numba] n_layers=%d n_pos=%d dtype=%s threads=%d → %d kernels "
            "(%s) em %.2fs (cache=%s)",
            n_layers,
            int(n_positions),
            complex_dtype,
            run_threads,
            len(functions_warmed),
            ", ".join(functions_warmed),
            elapsed_s,
            info["cache_dir"],
        )
    return info


def warmup_numba_simulator_from_config(
    cfg: "SimulationConfig",
    *,
    n_layers: int = 5,
    n_models: int = 1,
    dip_degs: Sequence[float] = (0.0,),
    verbose: bool = False,
) -> dict:
    """Aquece o caminho Numba EXATO de um ``SimulationConfig`` de produção.

    Conveniência para CI/produção: passa o ``cfg`` REAL via ``base_cfg`` →
    :func:`warmup_numba_simulator` reusa TODOS os campos do cfg (inclusive os
    seletores de kernel ``use_flat_prange``/``use_tiled_positions``/``parallel``/
    ``flat_prange_min_combos``) forçando só ``backend="numba"``. Assim aquece o
    kernel EXATO que a produção usará (não apenas o caminho default). ``n_layers``/
    ``n_models``/``dip_degs`` (propriedades do modelo, não do cfg) passam à parte.

    Args:
        cfg: :class:`SimulationConfig` de produção (todos os campos honrados).
        n_layers: Camadas do modelo de warmup. Default 5.
        n_models: Reservado. Default 1.
        dip_degs: Ângulos dip a aquecer. Default ``(0.0,)``.
        verbose: Loga timing + kernels aquecidos.

    Returns:
        ``dict`` — idêntico a :func:`warmup_numba_simulator`.

    Example:
        >>> from geosteering_ai.simulation.config import SimulationConfig
        >>> cfg = SimulationConfig(backend="numba", n_positions=200)
        >>> info = warmup_numba_simulator_from_config(cfg)

    Note:
        See Also: :func:`warmup_numba_simulator`.
    """
    return warmup_numba_simulator(
        n_layers=n_layers,
        n_models=n_models,
        frequencies_hz=(float(cfg.frequency_hz),),
        tr_spacings_m=(float(cfg.tr_spacing_m),),
        dip_degs=dip_degs,
        base_cfg=cfg,
        verbose=verbose,
    )
