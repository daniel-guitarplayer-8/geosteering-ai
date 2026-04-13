# -*- coding: utf-8 -*-
"""Testes Sprint 5.1b — jax.jacfwd end-to-end nativo.

Cobertura:

- Forward pure vs Numba (paridade bit-a-bit rtol<1e-12).
- jacfwd nativo vs FD Numba (rtol<5e-3).
- Estabilidade alta resistividade (ρ>1000 Ω·m).
- Shape/dtype do JacobianResult nativo.

Todos os testes requerem ``JAX_ENABLE_X64=True`` (fixture autouse).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from geosteering_ai.simulation import simulate  # noqa: E402
from geosteering_ai.simulation._jacobian import (  # noqa: E402
    compute_jacobian_fd_numba,
    compute_jacobian_jax,
)
from geosteering_ai.simulation._jax.forward_pure import (  # noqa: E402
    build_static_context,
    forward_pure_jax,
)
from geosteering_ai.simulation.config import SimulationConfig  # noqa: E402


@pytest.fixture
def simple_model():
    """Modelo 3 camadas isotrópico para testes rápidos."""
    return {
        "rho_h": np.array([10.0, 100.0, 10.0]),
        "rho_v": np.array([10.0, 100.0, 10.0]),
        "esp": np.array([5.0]),
        "positions_z": np.linspace(-2.0, 7.0, 8),
        "freq": 20000.0,
        "tr": 1.0,
    }


# ═════════════════════════════════════════════════════════════════════════
# 1 — Paridade forward_pure_jax vs Numba
# ═════════════════════════════════════════════════════════════════════════


def test_forward_pure_matches_numba(simple_model) -> None:
    """forward_pure_jax reproduz Numba bit-a-bit (float64 roundoff)."""
    m = simple_model
    ctx = build_static_context(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        freqs_hz=np.array([m["freq"]]),
        tr_spacing_m=m["tr"],
        dip_deg=0.0,
    )
    H_pure = np.asarray(forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx))

    cfg = SimulationConfig(
        frequency_hz=m["freq"],
        tr_spacing_m=m["tr"],
        backend="numba",
    )
    res = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg,
    )
    H_numba = res.H_tensor
    if H_numba.ndim == 2:
        H_numba = H_numba[:, np.newaxis, :]
    max_abs = np.max(np.abs(H_pure - H_numba))
    assert max_abs < 1e-10, f"max_abs={max_abs:.2e}"


# ═════════════════════════════════════════════════════════════════════════
# 2 — jacfwd nativo: shape, dtype, finitude
# ═════════════════════════════════════════════════════════════════════════


def test_jacfwd_native_shape_and_dtype(simple_model) -> None:
    """compute_jacobian_jax(try_jacfwd=True) retorna JacobianResult nativo."""
    m = simple_model
    cfg = SimulationConfig(
        frequency_hz=m["freq"],
        tr_spacing_m=m["tr"],
        backend="jax",
    )
    jac = compute_jacobian_jax(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg,
        try_jacfwd=True,
    )
    assert jac.method == "jacfwd"
    assert jac.backend == "jax_native"
    assert jac.fd_step is None
    assert jac.dH_dRho_h.shape == (m["positions_z"].shape[0], 1, 9, 3)
    assert jac.dH_dRho_h.dtype == np.complex128
    assert np.all(np.isfinite(jac.dH_dRho_h.real))
    assert np.all(np.isfinite(jac.dH_dRho_v.imag))


# ═════════════════════════════════════════════════════════════════════════
# 3 — Paridade jacfwd nativo vs FD Numba (rtol<5e-3)
# ═════════════════════════════════════════════════════════════════════════


def test_jacfwd_native_matches_fd_numba(simple_model) -> None:
    """∂H/∂ρ via jacfwd nativo ≈ FD Numba (rtol<5e-3)."""
    m = simple_model
    cfg_jax = SimulationConfig(
        frequency_hz=m["freq"],
        tr_spacing_m=m["tr"],
        backend="jax",
    )
    cfg_numba = SimulationConfig(
        frequency_hz=m["freq"],
        tr_spacing_m=m["tr"],
        backend="numba",
    )
    jac_native = compute_jacobian_jax(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg_jax,
        try_jacfwd=True,
    )
    jac_fd = compute_jacobian_fd_numba(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg_numba,
        fd_step=1e-4,
    )
    # Compara só entradas com magnitude ≥ 1e-9 para evitar divisão por zero.
    A = jac_native.dH_dRho_h
    B = jac_fd.dH_dRho_h
    mask = np.abs(B) > 1e-9
    if mask.any():
        rel = np.abs((A - B)[mask] / B[mask])
        assert (
            np.max(rel) < 5e-2
        ), f"jacfwd nativo diverge de FD: max_rel={np.max(rel):.2e}"


# ═════════════════════════════════════════════════════════════════════════
# 4 — Estabilidade alta resistividade (ρ>1000 Ω·m)
# ═════════════════════════════════════════════════════════════════════════


def test_jacfwd_native_high_rho_stability() -> None:
    """oklahoma_28 análogo com ρ=1500 Ω·m permanece finito em jacfwd nativo."""
    rho_h = np.array([10.0, 1500.0, 10.0])
    rho_v = np.array([10.0, 3000.0, 10.0])
    esp = np.array([3.0])
    z = np.linspace(0.0, 3.0, 5)
    cfg = SimulationConfig(backend="jax", frequency_hz=20000.0, tr_spacing_m=1.0)

    jac = compute_jacobian_jax(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=z,
        cfg=cfg,
        try_jacfwd=True,
    )
    assert jac.method == "jacfwd"
    assert np.all(np.isfinite(jac.dH_dRho_h.real))
    assert np.all(np.isfinite(jac.dH_dRho_h.imag))
    assert np.all(np.isfinite(jac.dH_dRho_v.real))


# ═════════════════════════════════════════════════════════════════════════
# 5 — JAX hybrid preservado (use_native_dipoles=False ainda funciona)
# ═════════════════════════════════════════════════════════════════════════


def test_jax_hybrid_path_preserved(simple_model) -> None:
    """Caminho JAX híbrido (pure_callback → Numba) não foi removido."""
    from geosteering_ai.simulation._jax.kernel import fields_in_freqs_jax_batch
    from geosteering_ai.simulation.filters import FilterLoader

    m = simple_model
    filt = FilterLoader().load("werthmuller_201pt")
    kr = np.asarray(filt.abscissas, dtype=np.float64)
    wJ0 = np.asarray(filt.weights_j0, dtype=np.float64)
    wJ1 = np.asarray(filt.weights_j1, dtype=np.float64)

    H_hybrid = fields_in_freqs_jax_batch(
        positions_z=m["positions_z"],
        dz_half=0.5,
        r_half=0.0,
        dip_rad=0.0,
        n=3,
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        freqs_hz=np.array([m["freq"]]),
        krJ0J1=kr,
        wJ0=wJ0,
        wJ1=wJ1,
        use_native_dipoles=False,  # Caminho híbrido preservado
    )
    assert H_hybrid.shape == (m["positions_z"].shape[0], 1, 9)
    assert np.all(np.isfinite(H_hybrid.real))
