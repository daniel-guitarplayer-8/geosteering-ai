# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_geosignals.py                                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0017-sm-plots-complete (Fatia 6d)                          ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM MVVM — geosinais + perfis ρ/λ + geologia no resultado   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-07                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre o núcleo PURO da Fatia 6d: fórmulas dos geosinais BYTE-FIÉIS,     ║
# ║    perfis ρ/λ (step), geologia exposta por _run_simulation, e paridade     ║
# ║    geosinal numba×jax <1e-12 (gated GPU).                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da Fatia 6d (núcleo PURO) — geosinais byte-fiéis, perfis ρ/λ, geologia."""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.gui.services.derived import (
    GEOSIGNALS,
    compute_geosignal,
    lambda_profile,
    rho_profile,
)
from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation


def _gpu_available() -> bool:
    try:
        from geosteering_ai.simulation.dispatch import _jax_gpu_available

        return bool(_jax_gpu_available())
    except Exception:
        return False


_needs_gpu = pytest.mark.skipif(not _gpu_available(), reason="requer GPU JAX")


# ════════════════════════════════════════════════════════════════════════════
# Geosinais — fórmulas byte-fiéis (índices 0=Hxx..8=Hzz)
# ════════════════════════════════════════════════════════════════════════════
def _h9(**comps: complex) -> np.ndarray:
    """Monta um h9 (1, 9) com as componentes nomeadas (resto 0)."""
    idx = {"Hxx": 0, "Hxy": 1, "Hxz": 2, "Hyx": 3, "Hyy": 4, "Hzz": 8}
    h = np.zeros((1, 9), dtype=np.complex128)
    for name, val in comps.items():
        h[0, idx[name]] = val
    return h


def test_geosignal_formulas_byte_faithful():
    h = _h9(Hxx=2 + 0j, Hyy=1 + 0j, Hxz=4 + 0j, Hzz=2 + 0j, Hxy=3 + 0j, Hyx=1 + 0j)
    assert np.isclose(compute_geosignal("USD", h)[0], 2.0)  # Hxx/Hyy
    assert np.isclose(compute_geosignal("UAD", h)[0], 1.0)  # Hxx−Hyy
    assert np.isclose(compute_geosignal("UHR", h)[0], 2.0)  # Hxz/Hzz
    assert np.isclose(compute_geosignal("UHA", h)[0], 2.0)  # Hxz−Hzz
    assert np.isclose(compute_geosignal("U3DF", h)[0], 0.5)  # (3−1)/(3+1)


def test_geosignal_all_names_and_errors():
    h = np.ones((2, 9), dtype=np.complex128)
    for name in GEOSIGNALS:
        out = compute_geosignal(name, h)
        assert out.shape == (2,)
    with pytest.raises(ValueError):
        compute_geosignal("BOGUS", h)
    with pytest.raises(ValueError):
        compute_geosignal("USD", np.ones((2, 3), dtype=np.complex128))


# ════════════════════════════════════════════════════════════════════════════
# Perfis ρ/λ — step via cumsum(esp)
# ════════════════════════════════════════════════════════════════════════════
def test_rho_profile_step():
    rho = np.array([1.0, 10.0, 100.0])
    esp = np.array([8.0])  # 3 camadas, 1 interna; interfaces [0, 8]
    z = np.array([-1.0, 0.0, 4.0, 8.0, 10.0])
    out = rho_profile(z, rho, esp)
    assert list(out) == [1.0, 10.0, 10.0, 100.0, 100.0]


def test_lambda_profile_step_clipped():
    rho_h = np.array([1.0, 10.0, 4.0])
    rho_v = np.array([2.0, 10.0, 16.0])  # λ = √2, 1, 2
    esp = np.array([8.0])
    z = np.array([-1.0, 4.0, 9.0])
    out = lambda_profile(z, rho_h, rho_v, esp)
    assert np.allclose(out, [np.sqrt(2.0), 1.0, 2.0])
    assert np.all(out >= 1.0)  # TIV: λ ≥ 1


# ════════════════════════════════════════════════════════════════════════════
# Geologia exposta por _run_simulation (perfis ρ/λ na galeria)
# ════════════════════════════════════════════════════════════════════════════
def test_run_simulation_exposes_geology_fixed():
    out = _run_simulation(
        SimRequest(geology_mode="fixed", n_models=2, frequencies_hz=(20000.0,), tj=10.0)
    )
    assert out["n_models"] == 2
    geo = out["geology"]
    assert len(geo) == 2
    assert geo[0]["rho_h"].shape == (3,) and geo[0]["thicknesses"].shape == (1,)


def test_run_simulation_exposes_geology_manual():
    out = _run_simulation(
        SimRequest(
            geology_mode="manual",
            n_models=2,
            manual_n_layers=3,
            manual_thicknesses=(8.0,),
            manual_rho_h=(1.0, 10.0, 100.0),
            manual_rho_v=(2.0, 20.0, 200.0),
            frequencies_hz=(20000.0,),
            tj=10.0,
        )
    )
    geo = out["geology"]
    assert len(geo) == 2
    assert list(geo[0]["rho_h"]) == [1.0, 10.0, 100.0]


# ════════════════════════════════════════════════════════════════════════════
# Paridade geosinal numba×jax <1e-12 (gated GPU)
# ════════════════════════════════════════════════════════════════════════════
@_needs_gpu
def test_geosignal_parity_numba_vs_jax():
    base = dict(geology_mode="fixed", n_models=2, frequencies_hz=(20000.0,), tj=10.0)
    h_n = _run_simulation(SimRequest(backend="numba", **base))["H6"]
    h_j = _run_simulation(SimRequest(backend="jax", **base))["H6"]
    for name in GEOSIGNALS:
        g_n = compute_geosignal(name, h_n)
        g_j = compute_geosignal(name, h_j)
        assert np.max(np.abs(g_n - g_j)) < 1e-12, name
