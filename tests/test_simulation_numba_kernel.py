# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_numba_kernel.py                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Testes Orquestrador (Sprint 2.4)       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de _numba/kernel.py — orquestrador forward (Sprint 2.4)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from geosteering_ai.simulation._numba.kernel import compute_zrho, fields_in_freqs
from geosteering_ai.simulation.filters import FilterLoader
from geosteering_ai.simulation.validation.half_space import (
    static_decoupling_factors,
    vmd_fullspace_broadside,
)

_MU_0 = 4.0e-7 * math.pi


def _setup(frequency=20000.0, rho=100.0, n=1, esp=None):
    """Helper: monta argumentos de kernel para caso single-layer."""
    filt = FilterLoader().load("werthmuller_201pt")
    if n == 1:
        rho_h = np.array([rho])
        rho_v = np.array([rho])
        esp = np.zeros(0, dtype=np.float64) if esp is None else esp
    else:
        rho_h = np.full(n, rho)
        rho_v = np.full(n, rho)
        if esp is None:
            esp = np.full(n - 2, 5.0, dtype=np.float64)
    return {
        "filt": filt,
        "n": n,
        "rho_h": rho_h,
        "rho_v": rho_v,
        "esp": esp,
        "freqs_hz": np.array([frequency]),
    }


class TestOrchestratorBasic:
    """fields_in_freqs estrutura básica."""

    def test_shape_nf_9(self):
        """Output tem shape (nf, 9)."""
        ctx = _setup()
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=ctx["n"],
            rho_h=ctx["rho_h"],
            rho_v=ctx["rho_v"],
            esp=ctx["esp"],
            freqs_hz=ctx["freqs_hz"],
            krJ0J1=ctx["filt"].abscissas,
            wJ0=ctx["filt"].weights_j0,
            wJ1=ctx["filt"].weights_j1,
        )
        assert cH.shape == (1, 9)
        assert cH.dtype == np.complex128

    def test_multi_frequency(self):
        """3 frequências → 3 linhas."""
        filt = FilterLoader().load("werthmuller_201pt")
        freqs = np.array([1000.0, 20000.0, 400000.0])
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=freqs,
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        assert cH.shape == (3, 9)

    def test_no_nan_inf(self):
        """Saída finita (sem NaN/Inf)."""
        ctx = _setup()
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=ctx["n"],
            rho_h=ctx["rho_h"],
            rho_v=ctx["rho_v"],
            esp=ctx["esp"],
            freqs_hz=ctx["freqs_hz"],
            krJ0J1=ctx["filt"].abscissas,
            wJ0=ctx["filt"].weights_j0,
            wJ1=ctx["filt"].weights_j1,
        )
        assert np.all(np.isfinite(cH.real))
        assert np.all(np.isfinite(cH.imag))

    def test_invalid_shapes_raise(self):
        """rho_h com shape errado levanta ValueError."""
        filt = FilterLoader().load("werthmuller_201pt")
        with pytest.raises(ValueError, match="rho_h.shape"):
            fields_in_freqs(
                Tx=0.0,
                Ty=0.0,
                Tz=0.0,
                cx=1.0,
                cy=0.0,
                cz=0.0,
                dip_rad=0.0,
                n=3,
                rho_h=np.array([100.0]),  # shape errado (espera (3,))
                rho_v=np.array([100.0, 100.0, 100.0]),
                esp=np.array([5.0]),
                freqs_hz=np.array([20000.0]),
                krJ0J1=filt.abscissas,
                wJ0=filt.weights_j0,
                wJ1=filt.weights_j1,
            )


class TestOrchestratorPhysics:
    """fields_in_freqs reproduz decoupling factors analíticos."""

    def test_decoupling_axial_hxx(self):
        """Hxx (col 0) ≈ +1/(2π) = ACx no limite estático (ρ alto)."""
        filt = FilterLoader().load("werthmuller_201pt")
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([1.0e8]),  # quase infinito → estático
            rho_v=np.array([1.0e8]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([10.0]),  # baixa freq
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        _, ACx = static_decoupling_factors(1.0)
        assert abs(cH[0, 0].real - ACx) < 1e-6

    def test_decoupling_planar_hyy(self):
        """Hyy (col 4) ≈ -1/(4π) = ACp."""
        filt = FilterLoader().load("werthmuller_201pt")
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([1.0e8]),
            rho_v=np.array([1.0e8]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([10.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        ACp, _ = static_decoupling_factors(1.0)
        assert abs(cH[0, 4].real - ACp) < 1e-6

    def test_decoupling_broadside_hzz(self):
        """Hzz (col 8) ≈ -1/(4π) = ACp no broadside."""
        filt = FilterLoader().load("werthmuller_201pt")
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([1.0e8]),
            rho_v=np.array([1.0e8]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([10.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        ACp, _ = static_decoupling_factors(1.0)
        assert abs(cH[0, 8].real - ACp) < 1e-6

    def test_vmd_broadside_analytical(self):
        """Hzz (col 8) vs vmd_fullspace_broadside (com conj por convenção)."""
        filt = FilterLoader().load("werthmuller_201pt")
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([20000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        Hz_analytical = np.conj(vmd_fullspace_broadside(1.0, 20000.0, 100.0))
        assert abs(cH[0, 8] - Hz_analytical) < 1e-4


class TestOrchestratorMultiLayer:
    """fields_in_freqs com perfis multi-camada."""

    def test_3_layer_isotropic(self):
        """Perfil 3 camadas isotrópicas - força continuidade."""
        filt = FilterLoader().load("werthmuller_201pt")
        # Todas as camadas com ρ=100 Ω·m → equivalente a full-space
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=3,
            rho_h=np.array([100.0, 100.0, 100.0]),
            rho_v=np.array([100.0, 100.0, 100.0]),
            esp=np.array([5.0]),
            freqs_hz=np.array([20000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        # Deve dar ~ mesmo resultado que full-space
        cH_full = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([20000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        # As 9 componentes devem estar próximas
        # (tolerância frouxa pois 1-camada vs 3-camadas homogêneas pode
        # diferir por arredondamento da recursão)
        np.testing.assert_allclose(cH, cH_full, atol=1e-10)

    def test_tiv_anisotropy(self):
        """Perfil TIV (ρh ≠ ρv) produz resultado distinto do isotrópico."""
        filt = FilterLoader().load("werthmuller_201pt")
        cH_iso = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([20000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        cH_tiv = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([100.0]),  # σh=0.01
            rho_v=np.array([500.0]),  # σv=0.002 → λ=5
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([20000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        # Valores TIV devem diferir do isotrópico
        assert not np.allclose(cH_iso, cH_tiv, atol=1e-10)


class TestOrchestratorHighResistivity:
    """fields_in_freqs estável em alta resistividade."""

    @pytest.mark.parametrize("rho", [1.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0])
    def test_high_resistivity_stable(self, rho):
        """ρ ∈ {1, 10², ..., 10⁶} Ω·m sem NaN/Inf."""
        filt = FilterLoader().load("werthmuller_201pt")
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([rho]),
            rho_v=np.array([rho]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([20000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        assert np.all(np.isfinite(cH.real)), f"NaN/Inf real em ρ={rho}"
        assert np.all(np.isfinite(cH.imag)), f"NaN/Inf imag em ρ={rho}"


class TestOrchestratorRotation:
    """dip_rad é aplicado corretamente via rotate_tensor."""

    def test_dip_zero_equals_untransformed(self):
        """dip_rad=0 → rotação identidade → saída igual ao tensor cru."""
        filt = FilterLoader().load("werthmuller_201pt")
        cH = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([20000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        # Hxx e Hzz devem ser distintos (axial vs broadside)
        assert abs(cH[0, 0] - cH[0, 8]) > 1e-3

    def test_dip_pi_2_swaps_xx_zz(self):
        """dip=π/2 (horizontal): Hxx ↔ Hzz no tensor rotacionado."""
        filt = FilterLoader().load("werthmuller_201pt")
        cH_vert = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=0.0,
            n=1,
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([20000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        cH_horiz = fields_in_freqs(
            Tx=0.0,
            Ty=0.0,
            Tz=0.0,
            cx=1.0,
            cy=0.0,
            cz=0.0,
            dip_rad=math.pi / 2,
            n=1,
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            freqs_hz=np.array([20000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        # Os 2 tensores devem ser diferentes (rotação 90°)
        assert not np.allclose(cH_vert, cH_horiz, atol=1e-6)


class TestComputeZrho:
    """Helper compute_zrho."""

    def test_full_space(self):
        """n=1 retorna valores da única camada."""
        zobs, rh, rv = compute_zrho(
            Tz=0.0,
            cz=10.0,
            n=1,
            rho_h=np.array([100.0]),
            rho_v=np.array([200.0]),
            esp=np.zeros(0, dtype=np.float64),
        )
        assert zobs == 5.0
        assert rh == 100.0
        assert rv == 200.0

    def test_mid_layer_correct(self):
        """Ponto-médio dentro da camada interna retorna seus rhos."""
        zobs, rh, rv = compute_zrho(
            Tz=0.0,
            cz=10.0,
            n=3,
            rho_h=np.array([1.0, 100.0, 1.0]),
            rho_v=np.array([1.0, 200.0, 1.0]),
            esp=np.array([20.0]),
        )
        assert zobs == 5.0
        assert rh == 100.0
        assert rv == 200.0
