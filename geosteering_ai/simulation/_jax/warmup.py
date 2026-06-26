# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/warmup.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend JAX (warmup do kernel GPU)     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-01 (v2.51 — redução de warmup JAX GPU)            ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : JAX 0.7.x (XLA)                                            ║
# ║  Dependências: jax (opcional — skip gracioso), multi_forward, forward_pure║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Pré-compila o caminho BUCKETED de PRODUÇÃO do simulador JAX GPU       ║
# ║    (``use_native_dipoles=True``) ANTES do workload real, eliminando o    ║
# ║    cold-start de compilação XLA na 1ª simulação. Aquece o cache JIT       ║
# ║    in-process (``_BUCKET_JIT_CACHE``) E popula o cache de disco           ║
# ║    persistente (XLA escreve o HLO compilado) → ganho cross-run.          ║
# ║                                                                           ║
# ║  GAP RESOLVIDO (v2.51)                                                    ║
# ║    O warmup legado ``_warmup_numba_tier2_sync`` usa                       ║
# ║    ``use_native_dipoles=False`` → aquece só o path Numba pure_callback,  ║
# ║    NUNCA o kernel JAX nativo de produção. Este módulo fecha o gap.        ║
# ║                                                                           ║
# ║  ESTRATÉGIA (dispatch REAL, não AOT)                                      ║
# ║    Dispara um ``simulate_multi_jax_batched`` tiny SINTÉTICO → o path real ║
# ║    garante HLO bit-idêntico ao de produção (cache-hit garantido) e        ║
# ║    reproduz todos os shapes compilados. (AOT ``.lower().compile()`` seria ║
# ║    frágil pela estrutura vmapped — ver NOTE em :func:`warmup_jax_simulator`)║
# ║                                                                           ║
# ║  INVIOLÁVEL                                                               ║
# ║    O warmup SÓ pré-compila kernels EXISTENTES — não altera numérica nem   ║
# ║    o hot-path. Paridade Fortran <1e-13 c128 estruturalmente preservada.  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Warmup JAX-nativo do simulador GPU — pré-compila o caminho bucketed.

Expõe :func:`warmup_jax_simulator` (parametrizável) e
:func:`warmup_jax_simulator_from_config` (deriva de um ``SimulationConfig``)
para pré-aquecer o cache JIT de bucket + o cache de disco persistente, de modo
que a 1ª simulação real não pague o cold-start de compilação XLA.

Example:
    Aquecer a config de inversão/inferência canônica (grid de 600 posições)::

        >>> from geosteering_ai.simulation._jax.warmup import warmup_jax_simulator
        >>> info = warmup_jax_simulator(n_layers=3, n_positions=600, verbose=True)
        >>> info["buckets_warmed"] >= 1
        True

Note:
    Warmup é PLENO apenas quando ``positions_z``/``n_layers``/``dip_degs``/
    ``tr_spacings_m``/``freqs_hz``/``complex_dtype``/``n_models`` batem com os
    valores de PRODUÇÃO — cada um governa um shape compilado distinto (o kernel
    bucketed recompila por shape de ``z_bucket``). Para o caso single/few-geometria
    (inversão/inferência), aqueça o grid EXATO. Para data-gen multi-geometria
    aleatória, o warmup cobre os buckets modais e prima o cache de disco, mas não
    elimina 100% das recompilações (ver ``docs/reports/`` — shape-explosion).
    Ref: docs/reference/plano_simulador_python_jax_numba.md.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.simulation.config import SimulationConfig

logger = logging.getLogger(__name__)

__all__ = [
    "warmup_jax_shapes",
    "warmup_jax_simulator",
    "warmup_jax_simulator_from_config",
]

# ── Valores sintéticos do modelo de warmup ────────────────────────────────
# Modelo isotrópico homogêneo: rho idêntico em todas as camadas + esp idêntico
# entre modelos → geometria HOMOGÊNEA → caminho BUCKETED (o de produção GPU).
_WARMUP_RHO_OHM_M: float = 10.0
_WARMUP_ESP_M: float = 5.0


def warmup_jax_simulator(
    *,
    n_layers: int = 3,
    n_positions: int = 600,
    hankel_filter: str = "werthmuller_201pt",
    complex_dtype: str = "complex128",
    dip_degs: Sequence[float] = (0.0,),
    tr_spacings_m: Sequence[float] = (1.0,),
    freqs_hz: Sequence[float] = (20000.0,),
    n_models: int = 1,
    jax_strategy: str = "bucketed",
    positions_z: Optional[np.ndarray] = None,
    esp_template: Optional[Union[np.ndarray, Sequence[float]]] = None,
    verbose: bool = False,
) -> dict:
    """Pré-compila o caminho bucketed de PRODUÇÃO do simulador JAX GPU.

    Dispara um ``simulate_multi_jax_batched`` sintético (modelo isotrópico
    homogêneo) que força a compilação XLA de cada bucket ``(ct, cr, n, npt,
    dtype)`` × shape de ``z_bucket`` da config alvo, populando o cache JIT
    in-process (``_BUCKET_JIT_CACHE``) E o cache de disco persistente.

    Fluxo:
      ┌────────────────────────────────────────────────────────────────────┐
      │  HAS_JAX? não → skip gracioso ({skipped: True})                     │
      │     ↓ sim                                                            │
      │  monta batch sintético homogêneo (rho=10, esp=5) → bucketed         │
      │     ↓                                                                │
      │  snapshot get_jit_cache_info() (antes)                              │
      │     ↓                                                                │
      │  simulate_multi_jax_batched(...) REAL + sync (.H_tensor.shape)      │
      │     ↓  → compila buckets + persiste HLO em disco                    │
      │  snapshot (depois) → delta = buckets aquecidos                      │
      └────────────────────────────────────────────────────────────────────┘

    Args:
        n_layers: Camadas do modelo (inclui 2 semi-espaços). Default 3.
            Governa o nº máximo de buckets ``(ct, cr) ∈ {0..n-1}²``.
        n_positions: Posições do grid de medição. Default 600 (escala produção).
            Governa os shapes de ``z_bucket`` — DEVE bater com produção.
        hankel_filter: Filtro Hankel (``npt`` deriva dele). Default werthmuller_201pt.
        complex_dtype: ``"complex128"`` (paridade, default) ou ``"complex64"``.
            CADA dtype é um cache key separado (dupla compilação se ambos usados).
        dip_degs: Ângulos dip (°) a aquecer. Cada dip → bucketset distinto.
        tr_spacings_m: Espaçamentos T-R (m) a aquecer.
        freqs_hz: Frequências (Hz) a aquecer (vmapadas internamente).
        n_models: Modelos no batch de warmup. Default 1 (inversão/inferência).
            DEVE bater com produção (governa o shape vmapped ``(n_models, ...)``).
        jax_strategy: ``"bucketed"`` (default, produção GPU) ou ``"unified"``.
        positions_z: Grid explícito ``(n_positions,)``. ``None`` → grid sintético
            cobrindo o modelo. Forneça o grid EXATO de produção p/ warmup pleno.
        esp_template: Espessuras EXPLÍCITAS ``(n_layers-2,)`` de UMA geometria real
            (1 dos K templates determinísticos da produção). ``None`` (default) →
            geometria homogênea sintética (CLI/inversão). Forneça o template real +
            ``positions_z`` p/ os buckets ``(ct,cr)`` baterem com a 1ª sim do SM
            (shape-matching). Usado por :func:`warmup_jax_shapes`.
        verbose: Loga timing + buckets aquecidos.

    Returns:
        ``dict`` com:
          - ``skipped`` (bool), ``reason`` (str|None) — skip se JAX ausente;
          - ``buckets_warmed`` (int) — delta do cache bucketed (cai p/
            ``programs_warmed`` se 0, p/ não enganar em ``jax_strategy="unified"``);
          - ``programs_warmed`` (int) — delta STRATEGY-AGNÓSTICO de
            ``total_xla_programs`` (bucketed+unified+chunked);
          - ``total_xla_programs`` (int) — total no cache pós-warmup;
          - ``n_distinct_shapes`` (int) — = buckets_warmed nesta chamada;
          - ``elapsed_s`` (float) — tempo de warmup;
          - ``jit_cache_info`` (dict) — snapshot pós-warmup;
          - ``persisted`` (bool) — cache de disco persistente ativo.

    Raises:
        ValueError: Se ``n_layers < 2`` ou ``n_positions < 1`` ou ``n_models < 1``.

    Example:
        >>> info = warmup_jax_simulator(n_layers=5, n_positions=600, n_models=64)
        >>> info["skipped"]
        False

    Note:
        **Por que dispatch real e não AOT**: o artefato compilado de produção é o
        wrapper ``jax.vmap``-sobre-modelos de ``_get_bucket_jit`` invocado com o
        shape vmapped ``(n_models, n_pos_bucket, nf, 9)``. Reproduzir esse
        lowering à mão via ``.lower()`` é frágil (in_axes, eta-stack, scatter); o
        dispatch real garante HLO bit-idêntico → cache-hit garantido.
        See Also: :func:`warmup_jax_simulator_from_config`.
    """
    if n_layers < 2:
        raise ValueError(f"n_layers deve ser >= 2 (recebido {n_layers}).")
    if n_positions < 1:
        raise ValueError(f"n_positions deve ser >= 1 (recebido {n_positions}).")
    if n_models < 1:
        raise ValueError(f"n_models deve ser >= 1 (recebido {n_models}).")

    # ── Guard: JAX ausente → skip gracioso (CI CPU / dev sem jax) ────────────
    from geosteering_ai.simulation._jax import HAS_JAX

    if not HAS_JAX:
        logger.info("warmup_jax_simulator: JAX ausente — skip gracioso.")
        return {
            "skipped": True,
            "reason": "jax_absent",
            "buckets_warmed": 0,
            "total_xla_programs": 0,
            "n_distinct_shapes": 0,
            "elapsed_s": 0.0,
            "jit_cache_info": {},
            "persisted": False,
        }

    import jax

    from geosteering_ai.simulation._jax.forward_pure import get_jit_cache_info
    from geosteering_ai.simulation._jax.multi_forward import (
        simulate_multi_jax_batched,
    )
    from geosteering_ai.simulation.config import SimulationConfig

    # ── Modelo sintético HOMOGÊNEO (isotrópico) → caminho bucketed ───────────
    # rho idêntico em todas as camadas + esp idêntico entre modelos garante
    # geometria homogênea (1 grupo) → kernel bucketed de produção.
    n_esp = max(n_layers - 2, 0)
    rho_h_batch = np.full((n_models, n_layers), _WARMUP_RHO_OHM_M, dtype=np.float64)
    rho_v_batch = np.full((n_models, n_layers), _WARMUP_RHO_OHM_M, dtype=np.float64)
    if esp_template is None:
        # Geometria HOMOGÊNEA sintética (default — inversão/inferência/CLI canônico).
        esp_batch = np.full((n_models, n_esp), _WARMUP_ESP_M, dtype=np.float64)
    else:
        # Geometria EXPLÍCITA (shape-matching): tile do template real (1 das K formas
        # determinísticas da produção) → buckets (ct,cr) idênticos aos da 1ª sim real.
        esp_row = np.asarray(esp_template, dtype=np.float64).reshape(-1)
        if esp_row.shape[0] != n_esp:
            raise ValueError(
                f"esp_template tem {esp_row.shape[0]} espessuras, mas n_layers={n_layers} "
                f"exige n_esp={n_esp} (= n_layers-2)."
            )
        esp_batch = np.broadcast_to(esp_row, (n_models, n_esp)).copy()

    if positions_z is None:
        if esp_template is None:
            total_thick = _WARMUP_ESP_M * n_esp if n_esp > 0 else _WARMUP_ESP_M
        else:
            total_thick = float(np.asarray(esp_batch[0]).sum()) or _WARMUP_ESP_M
        positions_z = np.linspace(-1.0, total_thick + 1.0, n_positions)
    else:
        positions_z = np.asarray(positions_z, dtype=np.float64)

    cfg = SimulationConfig(
        backend="jax",
        dtype=complex_dtype,
        jax_strategy=jax_strategy,
        hankel_filter=hankel_filter,
        n_positions=int(positions_z.shape[0]),
    )

    info_before = get_jit_cache_info()
    t0 = time.perf_counter()

    res = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=list(freqs_hz),
        tr_spacings_m=list(tr_spacings_m),
        dip_degs=list(dip_degs),
        cfg=cfg,
        hankel_filter=hankel_filter,
    )
    # `.H_tensor` já é numpy (np.asarray no batched) → força sync/block.
    _ = res.H_tensor.shape

    elapsed_s = time.perf_counter() - t0
    info_after = get_jit_cache_info()
    # `programs_warmed` é STRATEGY-AGNÓSTICO (delta de total_xla_programs cobre
    # bucketed + unified + chunked). `buckets_warmed` mede só o cache bucketed
    # (default de produção) — com jax_strategy="unified" o delta bucketed é 0,
    # então `buckets_warmed` cai p/ `programs_warmed` p/ não reportar 0 enganoso.
    bucketed_delta = int(info_after["bucketed_size"]) - int(
        info_before["bucketed_size"]
    )
    programs_warmed = int(info_after["total_xla_programs"]) - int(
        info_before["total_xla_programs"]
    )
    buckets_warmed = bucketed_delta if bucketed_delta > 0 else programs_warmed

    # Persistência: o cache de disco está ativo se a env var/config foi setada.
    # getattr p/ a flag dinâmica do jax.config (mypy-safe + version-guarded).
    import os

    persisted = bool(os.environ.get("JAX_COMPILATION_CACHE_DIR")) or bool(
        getattr(jax.config, "jax_compilation_cache_dir", None)
    )

    if verbose:
        logger.info(
            "[warmup-jax] n_layers=%d n_pos=%d dtype=%s strategy=%s n_models=%d "
            "→ %d buckets em %.2fs (total_xla=%d, persisted=%s)",
            n_layers,
            int(positions_z.shape[0]),
            complex_dtype,
            jax_strategy,
            n_models,
            buckets_warmed,
            elapsed_s,
            int(info_after["total_xla_programs"]),
            persisted,
        )

    return {
        "skipped": False,
        "reason": None,
        "buckets_warmed": buckets_warmed,
        "programs_warmed": programs_warmed,
        "total_xla_programs": int(info_after["total_xla_programs"]),
        "n_distinct_shapes": buckets_warmed,
        "elapsed_s": elapsed_s,
        "jit_cache_info": info_after,
        "persisted": persisted,
    }


def warmup_jax_simulator_from_config(
    cfg: "SimulationConfig",
    *,
    n_layers: int = 3,
    n_models: int = 1,
    dip_degs: Sequence[float] = (0.0,),
    verbose: bool = False,
) -> dict:
    """Aquece a config EXATA de produção derivando os kwargs de um ``SimulationConfig``.

    Conveniência para CI/inferência: extrai ``n_positions``/``dtype``/
    ``hankel_filter``/``jax_strategy``/``frequency_hz``/``tr_spacing_m`` do
    ``cfg`` e chama :func:`warmup_jax_simulator`. ``n_layers`` e ``n_models`` não
    fazem parte do ``SimulationConfig`` (são propriedades do modelo/batch) →
    passados à parte.

    Args:
        cfg: :class:`SimulationConfig` da produção (define grid, dtype, filtro,
            estratégia, freq, TR).
        n_layers: Camadas do modelo de warmup. Default 3.
        n_models: Modelos no batch de warmup. Default 1 (inversão/inferência).
        dip_degs: Ângulos dip a aquecer. Default ``(0.0,)``.
        verbose: Loga timing + buckets aquecidos.

    Returns:
        ``dict`` — idêntico a :func:`warmup_jax_simulator`.

    Example:
        >>> from geosteering_ai.simulation.config import SimulationConfig
        >>> cfg = SimulationConfig(backend="jax", n_positions=600)
        >>> info = warmup_jax_simulator_from_config(cfg)

    Note:
        See Also: :func:`warmup_jax_simulator`.
    """
    return warmup_jax_simulator(
        n_layers=n_layers,
        n_positions=int(cfg.n_positions),
        hankel_filter=cfg.hankel_filter,
        complex_dtype=cfg.dtype,
        dip_degs=dip_degs,
        tr_spacings_m=(float(cfg.tr_spacing_m),),
        freqs_hz=(float(cfg.frequency_hz),),
        n_models=n_models,
        jax_strategy=cfg.jax_strategy,
        verbose=verbose,
    )


def warmup_jax_shapes(
    specs: Sequence[dict],
    *,
    cancel_cb: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    verbose: bool = False,
) -> dict:
    """Aquece os SHAPES EXATOS de uma config do SM (lista de specs determinísticos).

    Diferente de :func:`warmup_jax_simulator` (geometria homogênea sintética), aqui cada
    ``spec`` carrega a geometria EXPLÍCITA de UM template determinístico da produção (via
    ``build_warmup_specs``), de modo que os programas XLA compilados sejam **bit-idênticos**
    aos que a 1ª simulação real dispara → cache-hit total. Itera os specs no caminho JAX
    DIRETO (``warmup_jax_simulator`` com ``esp_template``), reportando progresso por spec e
    cooperando com cancelamento ENTRE specs (NUNCA dentro do kernel — atômico).

    Cada ``spec`` é um dict com as chaves: ``esp_template`` (Sequence[float], n_layers-2),
    ``n_layers`` (int), ``n_models`` (int — dim-líder do vmap = chunk balanceado),
    ``positions_z`` (np.ndarray), ``freqs_hz``/``tr_spacings_m``/``dip_degs`` (Sequence),
    ``complex_dtype`` (str), ``hankel_filter`` (str), e opcional ``jax_strategy`` (str).

    Args:
        specs: lista de specs de warmup (de ``build_warmup_specs``). Vazia → no-op.
        cancel_cb: callable ``() -> bool``; se retornar ``True`` ENTRE specs, para cedo.
        progress_cb: callable ``(done, total)`` chamado após CADA spec aquecido.
        verbose: loga o agregado (programas aquecidos, tempo, cancelado).

    Returns:
        dict agregado: ``skipped``/``reason`` (JAX ausente), ``programs_warmed`` (delta de
        ``total_xla_programs``), ``specs_warmed``/``specs_total``, ``elapsed_s``,
        ``cancelled`` (bool), ``persisted`` (bool), ``total_xla_programs`` (pós-warmup).

    Note:
        SÓ pré-compila (dispatch REAL, nunca AOT) — não altera numérica nem hot-path.
        Paridade Fortran <1e-12 estruturalmente preservada. See Also:
        :func:`warmup_jax_simulator`, ``gui/services/jax_warmup_spec.build_warmup_specs``.
    """
    n_total = len(specs)
    from geosteering_ai.simulation._jax import HAS_JAX

    if not HAS_JAX:
        logger.info("warmup_jax_shapes: JAX ausente — skip gracioso.")
        return {
            "skipped": True,
            "reason": "jax_absent",
            "programs_warmed": 0,
            "specs_warmed": 0,
            "specs_total": n_total,
            "elapsed_s": 0.0,
            "cancelled": False,
            "persisted": False,
            "total_xla_programs": 0,
        }

    import os

    import jax

    from geosteering_ai.simulation._jax.forward_pure import get_jit_cache_info

    info_before = get_jit_cache_info()
    t0 = time.perf_counter()
    done = 0
    cancelled = False
    for spec in specs:
        if cancel_cb is not None and cancel_cb():
            cancelled = True
            break
        pz = np.asarray(spec["positions_z"], dtype=np.float64)
        # Caminho JAX DIRETO (bypassa o dispatcher) com a geometria REAL do template.
        warmup_jax_simulator(
            n_layers=int(spec["n_layers"]),
            n_positions=int(pz.shape[0]),
            hankel_filter=str(spec["hankel_filter"]),
            complex_dtype=str(spec["complex_dtype"]),
            dip_degs=tuple(spec["dip_degs"]),
            tr_spacings_m=tuple(spec["tr_spacings_m"]),
            freqs_hz=tuple(spec["freqs_hz"]),
            n_models=int(spec["n_models"]),
            jax_strategy=str(spec.get("jax_strategy", "bucketed")),
            positions_z=pz,
            esp_template=np.asarray(spec["esp_template"], dtype=np.float64),
            verbose=False,
        )
        done += 1
        if progress_cb is not None:
            progress_cb(done, n_total)

    elapsed_s = time.perf_counter() - t0
    info_after = get_jit_cache_info()
    programs_warmed = int(info_after["total_xla_programs"]) - int(
        info_before["total_xla_programs"]
    )
    persisted = bool(os.environ.get("JAX_COMPILATION_CACHE_DIR")) or bool(
        getattr(jax.config, "jax_compilation_cache_dir", None)
    )

    if verbose:
        logger.info(
            "[warmup-jax-shapes] %d/%d specs → %d programas em %.2fs "
            "(cancelled=%s, persisted=%s, total_xla=%d)",
            done,
            n_total,
            programs_warmed,
            elapsed_s,
            cancelled,
            persisted,
            int(info_after["total_xla_programs"]),
        )

    return {
        "skipped": False,
        "reason": None,
        "programs_warmed": programs_warmed,
        "specs_warmed": done,
        "specs_total": n_total,
        "elapsed_s": elapsed_s,
        "cancelled": cancelled,
        "persisted": persisted,
        "total_xla_programs": int(info_after["total_xla_programs"]),
    }
