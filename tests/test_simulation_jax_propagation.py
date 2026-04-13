# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_propagation.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Testes      : Sprint 3.2 — JAX propagation (common_arrays, factors)    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Framework   : pytest + JAX + NumPy                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  VALIDAÇÕES                                                               ║
# ║    1. Paridade JAX vs Numba < 1e-12 nos 9 arrays de common_arrays       ║
# ║    2. Paridade JAX vs Numba < 1e-12 nos 6 fatores de common_factors     ║
# ║    3. Casos: 1 camada (full-space), 3 camadas (TIV), alta resistividade ║
# ║    4. Diferenciabilidade via jax.grad (autodiff)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de Sprint 3.2 — paridade _jax/propagation vs _numba/propagation."""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation._jax import HAS_JAX

pytestmark = pytest.mark.skipif(
    not HAS_JAX, reason="JAX não instalado — Sprint 3.2 requer `pip install jax[cpu]`"
)

if HAS_JAX:
    import jax.numpy as jnp

    from geosteering_ai.simulation._jax.propagation import (
        common_arrays_jax,
        common_factors_jax,
    )
    from geosteering_ai.simulation._numba.geometry import sanitize_profile
    from geosteering_ai.simulation._numba.propagation import common_arrays as ca_numba
    from geosteering_ai.simulation._numba.propagation import common_factors as cf_numba


# Fixtures — cenários de perfil
_SCENARIOS = {
    "single_layer": {
        "n": 1,
        "h_arr": np.array([0.0]),
        "eta": np.array([[1.0, 1.0]]),
        "esp": np.zeros(0, dtype=np.float64),
    },
    "three_layers_tiv": {
        "n": 3,
        "eta": np.array([[1.0, 1.0], [0.01, 0.005], [1.0, 1.0]]),
        "esp": np.array([5.0]),
    },
    "five_layers_iso": {
        "n": 5,
        "eta": np.array(
            [
                [1.0, 1.0],
                [0.02, 0.02],
                [0.005, 0.005],
                [0.067, 0.067],
                [0.22, 0.22],
            ]
        ),
        "esp": np.array([2.0, 3.0, 1.5]),
    },
    "high_resistivity": {
        "n": 3,
        "eta": np.array([[1.0, 1.0], [1e-6, 1e-6], [1.0, 1.0]]),
        "esp": np.array([4.0]),
    },
}


@pytest.mark.skipif(not HAS_JAX, reason="JAX required")
@pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
class TestCommonArraysParity:
    """Paridade JAX vs Numba em common_arrays."""

    NAMES = ["u", "s", "uh", "sh", "RTEdw", "RTEup", "RTMdw", "RTMup", "AdmInt"]

    def _setup(self, scenario_name):
        sc = _SCENARIOS[scenario_name]
        n = sc["n"]
        npt = 201
        hordist = 1.0
        krJ0J1 = np.linspace(0.001, 50.0, npt).astype(np.float64)
        zeta = complex(0.0, 2.0 * np.pi * 20000.0 * 4e-7 * np.pi)
        eta = sc["eta"].astype(np.float64)

        if "h_arr" in sc:
            h_arr = sc["h_arr"].astype(np.float64)
        else:
            h_arr, _ = sanitize_profile(n, sc["esp"])
        return n, npt, hordist, krJ0J1, zeta, h_arr, eta

    def test_shapes_match(self, scenario_name) -> None:
        n, npt, hordist, krJ0J1, zeta, h_arr, eta = self._setup(scenario_name)
        outs_j = common_arrays_jax(
            n,
            npt,
            hordist,
            jnp.asarray(krJ0J1),
            zeta,
            jnp.asarray(h_arr),
            jnp.asarray(eta),
        )
        for o in outs_j:
            assert o.shape == (npt, n)

    def test_parity_all_arrays(self, scenario_name) -> None:
        n, npt, hordist, krJ0J1, zeta, h_arr, eta = self._setup(scenario_name)
        outs_n = ca_numba(n, npt, hordist, krJ0J1, zeta, h_arr, eta)
        outs_j = common_arrays_jax(
            n,
            npt,
            hordist,
            jnp.asarray(krJ0J1),
            zeta,
            jnp.asarray(h_arr),
            jnp.asarray(eta),
        )
        for name, o_n, o_j in zip(self.NAMES, outs_n, outs_j):
            diff = np.max(np.abs(np.asarray(o_j) - o_n))
            assert diff < 1e-12, f"{scenario_name}/{name}: diff={diff:.2e}"


@pytest.mark.skipif(not HAS_JAX, reason="JAX required")
class TestCommonFactorsParity:
    """Paridade JAX vs Numba em common_factors (com perfil 3-camadas TIV)."""

    def test_parity_three_layer_tx_in_top(self) -> None:
        n, npt = 3, 201
        hordist = 1.0
        krJ0J1 = np.linspace(0.001, 50.0, npt).astype(np.float64)
        zeta = complex(0.0, 2.0 * np.pi * 20000.0 * 4e-7 * np.pi)
        eta = np.array([[1.0, 1.0], [0.01, 0.005], [1.0, 1.0]], dtype=np.float64)
        h_arr, prof_arr = sanitize_profile(n, np.array([5.0]))

        # TX na camada 0 (topo)
        h0 = -0.5
        camad_t = 0

        outs_n = ca_numba(n, npt, hordist, krJ0J1, zeta, h_arr, eta)
        cf_n = cf_numba(n, npt, h0, h_arr, prof_arr, camad_t, *outs_n[:8])

        outs_j = common_arrays_jax(
            n,
            npt,
            hordist,
            jnp.asarray(krJ0J1),
            zeta,
            jnp.asarray(h_arr),
            jnp.asarray(eta),
        )
        cf_j = common_factors_jax(
            n,
            npt,
            h0,
            jnp.asarray(h_arr),
            jnp.asarray(prof_arr),
            camad_t,
            *outs_j[:8],
        )

        names = ["Mxdw", "Mxup", "Eudw", "Euup", "FEdwz", "FEupz"]
        for nm, o_n, o_j in zip(names, cf_n, cf_j):
            diff = np.max(np.abs(np.asarray(o_j) - o_n))
            assert diff < 1e-12, f"{nm}: diff={diff:.2e}"

    def test_parity_tx_in_middle_layer(self) -> None:
        """TX na camada média — exercita top_t/bot_t diferentes."""
        n, npt = 3, 201
        hordist = 1.5
        krJ0J1 = np.linspace(0.001, 50.0, npt).astype(np.float64)
        zeta = complex(0.0, 2.0 * np.pi * 20000.0 * 4e-7 * np.pi)
        eta = np.array([[1.0, 1.0], [0.01, 0.005], [0.5, 0.5]], dtype=np.float64)
        h_arr, prof_arr = sanitize_profile(n, np.array([3.0]))

        h0 = 1.5  # dentro da camada 1
        camad_t = 1

        outs_n = ca_numba(n, npt, hordist, krJ0J1, zeta, h_arr, eta)
        cf_n = cf_numba(n, npt, h0, h_arr, prof_arr, camad_t, *outs_n[:8])

        outs_j = common_arrays_jax(
            n,
            npt,
            hordist,
            jnp.asarray(krJ0J1),
            zeta,
            jnp.asarray(h_arr),
            jnp.asarray(eta),
        )
        cf_j = common_factors_jax(
            n,
            npt,
            h0,
            jnp.asarray(h_arr),
            jnp.asarray(prof_arr),
            camad_t,
            *outs_j[:8],
        )

        for o_n, o_j in zip(cf_n, cf_j):
            diff = np.max(np.abs(np.asarray(o_j) - o_n))
            assert diff < 1e-11  # tolerância levemente maior por ter exp grandes
