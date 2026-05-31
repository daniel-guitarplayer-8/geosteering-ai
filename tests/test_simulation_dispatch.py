# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_dispatch.py                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Sprint B — dispatcher batched JAX GPU ⇄ Numba CPU          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-31                                                 ║
# ║  Status      : Produção (gate de merge)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes Sprint B — `simulate_batch` + árvore de decisão `_resolve_backend`.

Cobre: (1) cada ramo da árvore (mockando GPU on/off); (2) guard anti-unified
(high-config → ValueError); (3) paridade cross-backend JAX vs Numba (<1e-10);
(4) o caminho do dispatcher casa com `simulate_multi_jax_batched_grouped`.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

jax_only = pytest.mark.skipif(not HAS_JAX, reason="JAX não instalado")

from geosteering_ai.simulation import dispatch  # noqa: E402
from geosteering_ai.simulation import simulate_batch  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# (1) Árvore de decisão — _resolve_backend (pura, mock de GPU)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve(backend, n, esp, gpu, *, numba_fallback=True, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(dispatch, "_jax_gpu_available", lambda: gpu)
    else:
        dispatch._jax_gpu_available = lambda: gpu  # fallback (não-isolado)
    return dispatch._resolve_backend(
        backend, n, esp, numba_fallback=numba_fallback, n_models_gpu_threshold=32
    )


def test_tree_auto_gpu_n64_homogeneous_to_jax(monkeypatch):
    esp = np.tile([5.0, 3.0], (64, 1))
    b, reason, ng = _resolve("auto", 64, esp, True, monkeypatch=monkeypatch)
    assert b == "jax" and ng == 1


def test_tree_auto_n_below_threshold_to_numba(monkeypatch):
    esp = np.tile([5.0, 3.0], (16, 1))
    b, _, _ = _resolve("auto", 16, esp, True, monkeypatch=monkeypatch)
    assert b == "numba"


def test_tree_auto_heterogeneous_to_numba(monkeypatch):
    esp = np.array([5.0, 3.0])[None] * np.random.default_rng(0).uniform(
        0.8, 1.2, (64, 2)
    )
    b, _, ng = _resolve("auto", 64, esp, True, monkeypatch=monkeypatch)
    assert b == "numba" and ng == 64


def test_tree_auto_no_gpu_to_numba(monkeypatch):
    esp = np.tile([5.0, 3.0], (64, 1))
    b, _, _ = _resolve("auto", 64, esp, False, monkeypatch=monkeypatch)
    assert b == "numba"


def test_tree_numba_forced(monkeypatch):
    esp = np.tile([5.0, 3.0], (64, 1))
    b, _, _ = _resolve("numba", 64, esp, True, monkeypatch=monkeypatch)
    assert b == "numba"


def test_tree_jax_forced_heterogeneous_fallback(monkeypatch):
    esp = np.array([5.0, 3.0])[None] * np.random.default_rng(1).uniform(
        0.8, 1.2, (64, 2)
    )
    b, _, _ = _resolve("jax", 64, esp, True, monkeypatch=monkeypatch)
    assert b == "numba"  # numba_fallback p/ não-agrupável


def test_tree_jax_forced_no_fallback_stays_jax(monkeypatch):
    esp = np.array([5.0, 3.0])[None] * np.random.default_rng(2).uniform(
        0.8, 1.2, (64, 2)
    )
    b, _, _ = _resolve(
        "jax", 64, esp, True, numba_fallback=False, monkeypatch=monkeypatch
    )
    assert b == "jax"


def test_tree_invalid_backend_raises(monkeypatch):
    esp = np.tile([5.0], (4, 1))
    with pytest.raises(ValueError, match="backend inválido"):
        _resolve("bogus", 4, esp, True, monkeypatch=monkeypatch)


# ─────────────────────────────────────────────────────────────────────────────
# (2) Guard anti-unified (high-config → ValueError)
# ─────────────────────────────────────────────────────────────────────────────
@jax_only
def test_guard_unified_high_config_raises(monkeypatch):
    """jax_strategy='unified' em high-config (n_pos≥300, n_configs≥9) → ValueError."""
    monkeypatch.setattr(dispatch, "_jax_gpu_available", lambda: True)
    n = 32
    rh = np.random.default_rng(0).uniform(1, 100, (n, 3))
    esp = np.tile([5.0], (n, 1))
    pz = np.linspace(-5, 5, 600)
    with pytest.raises(ValueError, match="unified.*PROIBIDO"):
        simulate_batch(
            rh,
            rh,
            esp,
            pz,
            frequencies_hz=[2e4, 5e4, 1e5],
            tr_spacings_m=[0.5, 1.0],
            dip_degs=[0.0, 30.0, 60.0],  # 18 configs
            backend="jax",
            jax_strategy="unified",
        )


def test_guard_unified_low_config_allowed(monkeypatch):
    """unified em low-config NÃO dispara o guard (só high-config OOMa)."""
    # n_pos pequeno → não high-config → sem guard (roteia normalmente p/ numba/jax).
    monkeypatch.setattr(dispatch, "_jax_gpu_available", lambda: False)  # → numba
    n = 4
    rh = np.random.default_rng(0).uniform(1, 100, (n, 3))
    esp = np.tile([5.0], (n, 1))
    H, info = simulate_batch(
        rh,
        rh,
        esp,
        np.linspace(-2, 12, 10),
        frequencies_hz=[2e4],
        backend="auto",
        jax_strategy="unified",
    )
    assert info["backend"] == "numba"  # sem GPU → numba; guard não dispara


# ─────────────────────────────────────────────────────────────────────────────
# (3-4) Paridade cross-backend + dispatcher casa com grouped direto
# ─────────────────────────────────────────────────────────────────────────────
@jax_only
def test_dispatch_jax_vs_numba_parity(monkeypatch):
    """simulate_batch(jax) vs (numba) — mesmos modelos → <1e-10 (invariante)."""
    monkeypatch.setattr(dispatch, "_jax_gpu_available", lambda: True)
    rng = np.random.default_rng(7)
    n, npos = 6, 20
    rh = rng.uniform(1, 100, (n, 3))
    esp = np.tile([5.0], (n, 1))
    pz = np.linspace(-5, 5, npos)
    kw = dict(frequencies_hz=[2e4, 5e4], tr_spacings_m=[1.0], dip_degs=[0.0, 30.0])
    H_jax, ij = simulate_batch(rh, rh, esp, pz, backend="jax", **kw)
    H_num, inu = simulate_batch(rh, rh, esp, pz, backend="numba", **kw)
    assert ij["backend"] == "jax" and inu["backend"] == "numba"
    assert np.abs(H_jax - H_num).max() < 1e-10


@jax_only
def test_dispatch_jax_matches_grouped_direct(monkeypatch):
    """O caminho JAX do dispatcher é bit-idêntico a simulate_multi_jax_batched_grouped."""
    from geosteering_ai.simulation import (
        SimulationConfig,
        simulate_multi_jax_batched_grouped,
    )

    monkeypatch.setattr(dispatch, "_jax_gpu_available", lambda: True)
    rng = np.random.default_rng(3)
    n, npos = 8, 16
    rh = rng.uniform(1, 100, (n, 3))
    esp = np.tile([5.0], (n, 1))
    pz = np.linspace(-2, 10, npos)
    kw = dict(frequencies_hz=[2e4], tr_spacings_m=[1.0], dip_degs=[0.0])
    H_disp, _ = simulate_batch(rh, rh, esp, pz, backend="jax", **kw)
    cfg = SimulationConfig(backend="jax", jax_strategy="bucketed", dtype="complex128")
    H_grp, _ = simulate_multi_jax_batched_grouped(rh, rh, esp, pz, cfg=cfg, **kw)
    assert np.abs(H_disp - np.asarray(H_grp)).max() == 0.0  # mesmo caminho
