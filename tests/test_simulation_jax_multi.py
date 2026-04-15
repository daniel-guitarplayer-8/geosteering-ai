# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_multi.py                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes Sprint 11-JAX — simulate_multi_jax paridade Numba   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-15 (Sprint 11-JAX, PR #23 / v1.5.0-alpha)         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes Sprint 11-JAX — paridade `simulate_multi_jax` vs `simulate_multi` Numba.

Valida que a implementação JAX produz resultados numericamente equivalentes
à referência Numba em 4 configurações × 3 modelos = 12 cenários.

Gate: ``max_abs_err < 1e-12``. Como o path JAX reutiliza `forward_pure_jax`
(JAX native end-to-end), diferenças bit-exact são esperadas em <1e-13.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

jax_required = pytest.mark.skipif(not HAS_JAX, reason="JAX não instalado")


# ──────────────────────────────────────────────────────────────────────────────
# Teste 1 — API shape e imports
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
def test_sprint11_jax_api_shape():
    """`simulate_multi_jax` retorna H_tensor shape (nTR, nAngles, n_pos, nf, 9)."""
    from geosteering_ai.simulation import MultiSimulationResultJAX, simulate_multi_jax

    rho_h = np.array([1.0, 20.0, 1.0])
    rho_v = np.array([1.0, 40.0, 1.0])
    esp = np.array([2.4384])
    positions_z = np.linspace(-5, 5, 10)

    result = simulate_multi_jax(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0, 40000.0],  # nf=2
        tr_spacings_m=[0.5, 1.0, 1.5],  # nTR=3
        dip_degs=[0.0, 30.0],  # nAngles=2
    )

    assert isinstance(result, MultiSimulationResultJAX)
    assert result.H_tensor.shape == (3, 2, 10, 2, 9)
    assert result.z_obs.shape == (2, 10)
    assert result.rho_h_at_obs.shape == (2, 10)
    assert result.rho_v_at_obs.shape == (2, 10)


# ──────────────────────────────────────────────────────────────────────────────
# Teste 2 — Dedup de cache por hordist
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
def test_sprint11_jax_dedup_vertical():
    """Poço vertical (dip=0°, 3 TR): 1 único hordist cache."""
    from geosteering_ai.simulation import simulate_multi_jax

    rho_h = np.array([1.0, 20.0, 1.0])
    rho_v = np.array([1.0, 40.0, 1.0])
    esp = np.array([2.4384])
    positions_z = np.linspace(-5, 5, 10)

    result = simulate_multi_jax(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[0.5, 1.0, 1.5],  # 3 TR
        dip_degs=[0.0],  # dip=0 → hordist=0 para todos
    )
    assert (
        result.unique_hordist_count == 1
    ), f"Poço vertical deveria ter 1 cache único; obtido {result.unique_hordist_count}"


# ──────────────────────────────────────────────────────────────────────────────
# Teste 3 — Paridade JAX vs Numba em 4 configs × 3 modelos = 12 cenários
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
@pytest.mark.parametrize(
    "model_name",
    ["oklahoma_3", "oklahoma_5", "devine_8"],
)
@pytest.mark.parametrize(
    "config",
    [
        (1, 1),  # single TR, single angle
        (3, 1),  # multi TR, single angle (vertical)
        (1, 3),  # single TR, multi angle
        (2, 2),  # multi TR × multi angle
    ],
)
def test_sprint11_jax_parity_vs_numba(model_name, config):
    """Paridade <1e-12 simulate_multi_jax vs simulate_multi em 12 cenários."""
    from geosteering_ai.simulation import simulate_multi, simulate_multi_jax
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    nTR, nAngles = config
    m = get_canonical_model(model_name)

    # TR spacings + ângulos
    tr_list = [0.5, 1.0, 1.5][:nTR] if nTR > 1 else [1.0]
    dip_list = [0.0, 30.0, 60.0][:nAngles] if nAngles > 1 else [0.0]

    rho_h = np.asarray(m.rho_h, dtype=np.float64)
    rho_v = np.asarray(m.rho_v, dtype=np.float64)
    esp = np.asarray(m.esp, dtype=np.float64)
    positions_z = np.linspace(m.min_depth - 2, m.max_depth + 2, 20)

    # ── Numba (baseline) ───────────────────────────────────────────────────
    res_numba = simulate_multi(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=tr_list,
        dip_degs=dip_list,
    )

    # ── JAX ────────────────────────────────────────────────────────────────
    res_jax = simulate_multi_jax(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=tr_list,
        dip_degs=dip_list,
    )

    # Shapes devem ser idênticos
    assert res_jax.H_tensor.shape == res_numba.H_tensor.shape

    # Paridade <1e-12 (em prática observa-se ~1e-14)
    max_err = float(np.max(np.abs(res_jax.H_tensor - res_numba.H_tensor)))
    assert (
        max_err < 1e-12
    ), f"{model_name} {nTR}TR×{nAngles}ang: max_abs_err = {max_err} > 1e-12"


# ──────────────────────────────────────────────────────────────────────────────
# Teste 4 — to_single() roundtrip
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
def test_sprint11_jax_to_single_roundtrip():
    """MultiSimulationResultJAX.to_single() retorna SimulationResult válido."""
    from geosteering_ai.simulation import simulate_multi_jax
    from geosteering_ai.simulation.forward import SimulationResult

    rho_h = np.array([1.0, 20.0, 1.0])
    rho_v = np.array([1.0, 40.0, 1.0])
    esp = np.array([2.4384])
    positions_z = np.linspace(-5, 5, 10)

    result = simulate_multi_jax(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )

    single = result.to_single()
    assert isinstance(single, SimulationResult)
    assert single.H_tensor.shape == (10, 1, 9)


# ──────────────────────────────────────────────────────────────────────────────
# Teste 5 — Validações fail-fast (paridade com simulate_multi)
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
def test_sprint11_jax_empty_lists_raise():
    """Listas vazias devem raise ValueError."""
    from geosteering_ai.simulation import simulate_multi_jax

    rho_h = np.array([1.0, 20.0, 1.0])
    rho_v = np.array([1.0, 40.0, 1.0])
    esp = np.array([2.4384])
    positions_z = np.linspace(-5, 5, 10)

    with pytest.raises(ValueError, match="tr_spacings_m vazio"):
        simulate_multi_jax(
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            positions_z=positions_z,
            tr_spacings_m=[],
            dip_degs=[0.0],
        )

    with pytest.raises(ValueError, match="dip_degs vazio"):
        simulate_multi_jax(
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            positions_z=positions_z,
            tr_spacings_m=[1.0],
            dip_degs=[],
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
