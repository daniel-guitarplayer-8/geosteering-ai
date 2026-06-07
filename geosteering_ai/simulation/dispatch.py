# -*- coding: utf-8 -*-
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/dispatch.py                                       ║
# ║  ------------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                           ║
# ║  Subsistema  : Simulador — Dispatcher batched (JAX GPU ⇄ Numba CPU)          ║
# ║  Autor       : Daniel Leal                                                   ║
# ║  Criação     : 2026-05-31 (Sprint B — A-jax-gpu-dispatcher)                  ║
# ║  Status      : Produção                                                      ║
# ║  Framework   : NumPy + JAX (lazy) + Numba                                    ║
# ║  ------------------------------------------------------------------------    ║
# ║  FINALIDADE                                                                   ║
# ║    Roteia uma simulação BATCHED (n_models modelos) entre o caminho JAX       ║
# ║    GPU (bucketed/grouped) e o Numba CPU (16w×4t), codificando a ÁRVORE DE    ║
# ║    DECISÃO MEDIDA nos relatórios v2.45–v2.47 — sem que o caller precise      ║
# ║    escolher o backend manualmente.                                          ║
# ║                                                                              ║
# ║  ÁRVORE DE DECISÃO (backend="auto")                                          ║
# ║    ┌────────────────────────────────────────────────────────────────────┐  ║
# ║    │  GPU disponível + n_models≥32 + geometria agrupável → JAX bucketed  │  ║
# ║    │  sem GPU  OU  n_models<32  OU  geometria não-agrupável → Numba 16w4t │  ║
# ║    │  NUNCA o kernel JAX `unified` em high-config (guard/erro — OOM 80GB) │  ║
# ║    └────────────────────────────────────────────────────────────────────┘  ║
# ║                                                                              ║
# ║  REUSO (não reinventa)                                                       ║
# ║    • simulate_multi_jax_batched_grouped + group_by_geometry (Sprint A)       ║
# ║    • simulate_multi (Numba, referência de paridade)                          ║
# ║    • jax.devices()[0].platform == "gpu" (auto-detect)                        ║
# ║                                                                              ║
# ║  PARIDADE: o dispatcher SÓ roteia — não toca a física. JAX = grouped/        ║
# ║  bucketed (parity <1e-13 c128); Numba = simulate_multi (referência).         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Dispatcher batched JAX GPU ⇄ Numba CPU — :func:`simulate_batch`.

Expõe :func:`simulate_batch` que recebe um batch de modelos e roteia para o
backend ótimo via a árvore de decisão medida (``backend="auto"``), ou força um
backend específico (``"jax"`` / ``"numba"``) com guardas de segurança.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes da árvore de decisão (documentação física/de performance — D10)
# ──────────────────────────────────────────────────────────────────────────────
# Crossover de OCUPAÇÃO da GPU: abaixo de 32 modelos no batch, o vmap não satura
# a A6000 e o JAX perde para o Numba (medido v2.45/v2.47 — 0.53× a n=16, 1.89× a
# n≥32). Default do limiar `n_models_gpu_threshold`.
_N_MODELS_GPU_THRESHOLD: int = 32

# Limiar de AGRUPABILIDADE: se mais de 50% dos modelos têm geometria (esp) própria,
# o caminho JAX-grouped degeneraria em ~1 modelo/grupo (sem ganho de batch). Acima
# disso → Numba (medido Sprint A).
_GROUPABLE_RATIO_MAX: float = 0.5

# Regime "high-config" onde o kernel JAX `unified` (heterogêneo) estoura VRAM
# (medido: 80 GB a 18 cfg × 600 pos). Usado pelo guard anti-unified.
_HIGH_CONFIG_MIN_POS: int = 300
_HIGH_CONFIG_MIN_CONFIGS: int = 9


def _jax_gpu_available() -> bool:
    """Detecta se o JAX enxerga um device GPU (auto-detect do dispatcher).

    Returns:
        bool: ``True`` se ``jax`` está instalado E ``jax.devices()[0].platform``
            é ``"gpu"``. ``False`` em qualquer falha (jax ausente, sem GPU, erro
            de inicialização) — rota segura para Numba CPU.

    Note:
        Import LAZY de ``jax`` — este módulo deve funcionar em ambientes
        Numba-only (CI CPU). Ref: ``_jacobian.py:604`` usa o mesmo predicado.
    """
    try:
        import jax

        return bool(jax.devices()[0].platform == "gpu")
    except Exception:  # noqa: BLE001 — jax ausente / sem device / init error → Numba
        return False


def _is_high_config(n_pos: int, n_configs: int) -> bool:
    """Regime onde o kernel JAX ``unified`` estoura VRAM (guard anti-OOM)."""
    return n_pos >= _HIGH_CONFIG_MIN_POS and n_configs >= _HIGH_CONFIG_MIN_CONFIGS


def _resolve_backend(
    backend: str,
    n_models: int,
    esp_batch: np.ndarray,
    *,
    numba_fallback: bool,
    n_models_gpu_threshold: int,
) -> Tuple[str, str, Optional[int]]:
    """Resolve o backend efetivo via a árvore de decisão (helper puro, testável).

    Args:
        backend: ``"auto"`` | ``"jax"`` | ``"jax_gpu"`` | ``"numba"``.
        n_models: nº de modelos no batch.
        esp_batch: ``(n_models, n-2)`` — geometrias (p/ medir agrupabilidade).
        numba_fallback: se ``True``, força ``jax`` cai p/ Numba quando não-agrupável.
        n_models_gpu_threshold: limiar de ocupação GPU (default 32).

    Returns:
        Tupla ``(backend_efetivo, motivo, n_geometry_groups)``.

    Raises:
        ValueError: ``backend`` inválido.

    Note:
        Pura (sem GPU, exceto ``_jax_gpu_available``) — cada ramo é testável
        mockando ``_jax_gpu_available``. See Also: :func:`simulate_batch`.
    """
    from geosteering_ai.simulation._jax.multi_forward import group_by_geometry

    esp_np = np.asarray(esp_batch, dtype=np.float64)
    n_esp = esp_np.shape[1] if esp_np.ndim == 2 else 0
    n_groups: Optional[int] = len(group_by_geometry(esp_np)) if n_esp > 0 else None
    # n_esp==0 (≤2 camadas) → geometria trivialmente compartilhada (1 grupo).
    groupable = (n_groups is None) or (n_groups <= _GROUPABLE_RATIO_MAX * n_models)

    if backend == "numba":
        return "numba", "backend forçado: numba", n_groups

    if backend in ("jax", "jax_gpu"):
        # Forçado-jax: preserva o comportamento Sprint A (warns + fallback opt-in).
        if not _jax_gpu_available():
            logger.warning(
                "simulate_batch: backend=jax forçado SEM GPU JAX — execução em CPU "
                "é lenta; considere backend='numba' ou 'auto'."
            )
        if n_models < n_models_gpu_threshold:
            logger.warning(
                "simulate_batch: backend=jax com n_models=%d < %d — GPU subocupada "
                "(crossover de ocupação ~n≥%d).",
                n_models,
                n_models_gpu_threshold,
                n_models_gpu_threshold,
            )
        if numba_fallback and not groupable:
            reason = (
                f"geometria mal-agrupável (n_grupos={n_groups}/{n_models}) → fallback "
                "p/ Numba 16w×4t (JAX-grouped degeneraria em per-model)"
            )
            logger.warning("simulate_batch: %s", reason)
            return "numba", reason, n_groups
        return "jax", "backend forçado: jax (grouped/bucketed)", n_groups

    if backend == "auto":
        # Árvore ESTRITA (medida): cada condição que desfavorece a GPU → Numba.
        if not _jax_gpu_available():
            return "numba", "auto: sem GPU JAX → Numba CPU", n_groups
        if n_models < n_models_gpu_threshold:
            return (
                "numba",
                f"auto: n_models={n_models} < {n_models_gpu_threshold} "
                "(GPU subocupada) → Numba",
                n_groups,
            )
        if not groupable:
            return (
                "numba",
                f"auto: geometria não-agrupável (n_grupos={n_groups}/{n_models}) "
                "→ Numba",
                n_groups,
            )
        return (
            "jax",
            f"auto: GPU + n_models≥{n_models_gpu_threshold} + agrupável "
            "→ JAX bucketed (grouped)",
            n_groups,
        )

    raise ValueError(f"backend inválido: {backend!r}. Use 'auto' | 'jax' | 'numba'.")


def _simulate_batch_numba(
    rho_h_batch: np.ndarray,
    rho_v_batch: np.ndarray,
    esp_batch: np.ndarray,
    positions_z: np.ndarray,
    *,
    frequencies_hz: Sequence[float],
    tr_spacings_m: Sequence[float],
    dip_degs: Sequence[float],
    hankel_filter: str,
) -> np.ndarray:
    """Caminho Numba: itera por modelo via ``simulate_multi`` (paraleliza por prange).

    Returns:
        ``H_tensor`` ``(n_models, nTR, nAng, n_pos, nf, 9)`` complex128.

    Note:
        Referência de paridade (<1e-12 vs Fortran). Cada chamada usa o pool
        Numba 16w×4t. Extraído de ``synthetic_generator.generate_batch`` (Sprint A).
    """
    from geosteering_ai.simulation import SimulationConfig, simulate_multi

    n_models = rho_h_batch.shape[0]
    n_pos = positions_z.shape[0]
    nf, nTR, nAng = len(frequencies_hz), len(tr_spacings_m), len(dip_degs)
    sim_cfg = SimulationConfig(backend="numba", parallel=True)

    H6 = np.empty((n_models, nTR, nAng, n_pos, nf, 9), dtype=np.complex128)
    for i in range(n_models):
        res = simulate_multi(
            rho_h=rho_h_batch[i],
            rho_v=rho_v_batch[i],
            esp=esp_batch[i],
            positions_z=positions_z,
            frequencies_hz=list(frequencies_hz),
            tr_spacings_m=list(tr_spacings_m),
            dip_degs=list(dip_degs),
            cfg=sim_cfg,
            hankel_filter=hankel_filter,
        )
        # 1 modelo (sem `models=`) → MultiSimulationResult (arm com H_tensor) da
        # union; `getattr` evita o erro union-attr do mypy + checa em runtime.
        h_i = getattr(res, "H_tensor", None)
        if h_i is None:  # pragma: no cover — guarda defensiva
            raise TypeError("simulate_multi retornou tipo sem H_tensor.")
        H6[i] = np.asarray(h_i)
    return H6


def simulate_batch(
    rho_h_batch: np.ndarray,
    rho_v_batch: np.ndarray,
    esp_batch: np.ndarray,
    positions_z: np.ndarray,
    *,
    frequencies_hz: Optional[Sequence[float]] = None,
    tr_spacings_m: Optional[Sequence[float]] = None,
    dip_degs: Optional[Sequence[float]] = None,
    backend: str = "auto",
    numba_fallback: bool = True,
    n_models_gpu_threshold: int = _N_MODELS_GPU_THRESHOLD,
    dtype: str = "complex128",
    jax_chunk_size_models: Optional[int] = None,
    jax_strategy: str = "bucketed",
    hankel_filter: str = "werthmuller_201pt",
) -> Tuple[np.ndarray, dict]:
    """Dispatcher batched — roteia JAX GPU ⇄ Numba CPU pela árvore de decisão.

    Estrutura do dispatcher:
      ┌────────────────────────────────────────────────────────────────────┐
      │  backend="auto"  →  _resolve_backend (árvore medida)               │
      │     ├─ jax    →  simulate_multi_jax_batched_grouped (bucketed)     │
      │     └─ numba  →  _simulate_batch_numba (simulate_multi per-model)  │
      │  Guard: jax_strategy="unified" + high-config → ValueError (OOM)    │
      └────────────────────────────────────────────────────────────────────┘

    Args:
        rho_h_batch: ``(n_models, n)`` float — resistividades horizontais (Ω·m).
        rho_v_batch: ``(n_models, n)`` float — resistividades verticais (Ω·m).
        esp_batch: ``(n_models, n-2)`` float — espessuras internas (m). PODE variar
            entre modelos (a agrupabilidade é medida por :func:`group_by_geometry`).
        positions_z: ``(n_pos,)`` float — profundidades TVD compartilhadas (m).
        frequencies_hz: lista de frequências (Hz). Default ``[20000.0]``.
        tr_spacings_m: lista de espaçamentos T-R (m). Default ``[1.0]``.
        dip_degs: lista de ângulos dip (°). Default ``[0.0]``.
        backend: ``"auto"`` (default, árvore) | ``"jax"`` | ``"numba"``.
        numba_fallback: forçado-jax cai p/ Numba quando geometria não-agrupável.
        n_models_gpu_threshold: limiar de ocupação GPU (default 32).
        dtype: dtype complexo do path JAX (``"complex128"`` default — paridade).
        jax_chunk_size_models: chunk do eixo de modelos (VRAM) no path JAX.
        jax_strategy: ``"bucketed"`` (default/seguro). ``"unified"`` é GUARDADO em
            high-config (OOM).
        hankel_filter: nome do filtro Hankel.

    Returns:
        Tupla ``(H_tensor, info)``:
          - ``H_tensor``: ``(n_models, nTR, nAng, n_pos, nf, 9)`` complex.
          - ``info``: dict com ``backend`` (efetivo), ``reason``, ``n_geometry_groups``,
            ``elapsed_s``.

    Raises:
        ValueError: ``backend`` inválido; OU ``jax_strategy="unified"`` em high-config
            (guard anti-OOM — use ``"bucketed"`` / o caminho grouped).

    Note:
        O caminho JAX SEMPRE usa o grouped (bucketed por grupo de geometria) →
        o kernel ``unified`` NUNCA é atingido por construção. Paridade <1e-13
        c128 preservada (só roteia). See Also:
        :func:`geosteering_ai.simulation.simulate_multi_jax_batched_grouped`.
    """
    rho_h_np = np.asarray(rho_h_batch, dtype=np.float64)
    rho_v_np = np.asarray(rho_v_batch, dtype=np.float64)
    esp_np = np.asarray(esp_batch, dtype=np.float64)
    n_models = rho_h_np.shape[0]

    freqs = list(frequencies_hz) if frequencies_hz is not None else [20000.0]
    trs = list(tr_spacings_m) if tr_spacings_m is not None else [1.0]
    dips = list(dip_degs) if dip_degs is not None else [0.0]
    n_configs = len(freqs) * len(trs) * len(dips)

    # ── Resolve o backend efetivo (árvore de decisão) ────────────────────────
    resolved, reason, n_groups = _resolve_backend(
        backend,
        n_models,
        esp_np,
        numba_fallback=numba_fallback,
        n_models_gpu_threshold=n_models_gpu_threshold,
    )

    # ── Guard anti-unified: NUNCA o kernel unified em high-config (OOM 80 GB) ─
    if (
        resolved == "jax"
        and jax_strategy == "unified"
        and _is_high_config(positions_z.shape[0], n_configs)
    ):
        raise ValueError(
            "simulate_batch: jax_strategy='unified' é PROIBIDO em high-config "
            f"(n_pos={positions_z.shape[0]}, n_configs={n_configs}) — estoura VRAM "
            "(80 GB medido). Use jax_strategy='bucketed' (caminho grouped)."
        )

    # ── Roteamento ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if resolved == "jax":
        from geosteering_ai.simulation import (
            SimulationConfig,
            simulate_multi_jax_batched_grouped,
        )

        sim_cfg = SimulationConfig(
            backend="jax",
            jax_strategy="bucketed",  # FORÇA bucketed — guard anti-unified
            dtype=dtype,
            jax_chunk_size_models=jax_chunk_size_models,
            # n_workers/threads explícitos (INERTES no JAX — vmap/XLA não os
            # consomem) só para PULAR o auto-detect de paralelismo Numba do
            # __post_init__ (config.py:834), que loga "Sprint v2.23 A.2 —
            # auto-detect: n_workers=…" de forma inútil/ruidosa no caminho JAX.
            n_workers=1,
            threads_per_worker=1,
        )
        H6, grp_info = simulate_multi_jax_batched_grouped(
            rho_h_np,
            rho_v_np,
            esp_np,
            positions_z,
            frequencies_hz=freqs,
            tr_spacings_m=trs,
            dip_degs=dips,
            cfg=sim_cfg,
            hankel_filter=hankel_filter,
        )
        H6 = np.asarray(H6)
        n_groups = grp_info.get("n_groups", n_groups)
    else:  # numba
        H6 = _simulate_batch_numba(
            rho_h_np,
            rho_v_np,
            esp_np,
            positions_z,
            frequencies_hz=freqs,
            tr_spacings_m=trs,
            dip_degs=dips,
            hankel_filter=hankel_filter,
        )
    elapsed = time.perf_counter() - t0

    info = {
        "backend": resolved,
        "reason": reason,
        "n_geometry_groups": n_groups,
        "elapsed_s": elapsed,
    }
    return H6, info


__all__ = ["simulate_batch"]
