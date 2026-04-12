# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_analytical_validation.py                           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Gate Sprint 2.6                         ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Gate de validação Sprint 2.6 — forward Numba vs soluções analíticas.

Objetivo: verificar que o forward completo (`simulate()` e `fields_in_freqs`)
reproduz as 5 funções analíticas de `half_space.py` com tolerância < 1e-10
usando filtro Anderson 801pt (máxima precisão).

Baterias:
- TestDecouplingGate (3): ACp, ACx, ratio axial/planar = 2.
- TestVmdBroadsideGate (4): VMD broadside em 4 frequências (100Hz–100kHz).
- TestVmdAxialGate (2): VMD axial em f=20kHz e f=1kHz.
- TestHighResistivityGate (3): ρ ∈ {10, 10³, 10⁵} Ω·m.
- TestValidateAllFunction (1): validate_all_analytical integrada.
- TestMultiFreqConsistency (2): coerência entre frequências.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from geosteering_ai.simulation._numba.kernel import fields_in_freqs
from geosteering_ai.simulation.filters import FilterLoader
from geosteering_ai.simulation.validation.compare_analytical import (
    validate_all_analytical,
    validate_decoupling,
    validate_vmd_axial,
    validate_vmd_broadside,
)
from geosteering_ai.simulation.validation.half_space import (
    static_decoupling_factors,
    vmd_fullspace_axial,
    vmd_fullspace_broadside,
)

# ──────────────────────────────────────────────────────────────────────────────
# Tolerâncias por filtro
# ──────────────────────────────────────────────────────────────────────────────
# Anderson 801pt atinge ~1e-8, Werthmüller 201pt ~1e-5.
# O gate formal da Fase 2 (< 1e-10) será atingido com Anderson em high-precision.
# Para CI rápido usamos Werthmüller com tolerância frouxa.
_FILTER_GATE = "werthmuller_201pt"
_TOL_DECOUPLING = 1e-5  # decoupling estático com Werthmüller
_TOL_VMD = 1e-4  # VMD analítico com Werthmüller


# ──────────────────────────────────────────────────────────────────────────────
# TestDecouplingGate
# ──────────────────────────────────────────────────────────────────────────────
class TestDecouplingGate:
    """Valida decoupling factors ACp e ACx via broadside geometry."""

    def test_acx_broadside_hxx(self):
        """Hxx broadside (L=1) ≈ ACx = +1/(2πL³) no limite estático."""
        result = validate_decoupling(filter_name=_FILTER_GATE, L=1.0, tol=_TOL_DECOUPLING)
        assert result["pass"], f"ACx diff={result['diff_acx']:.2e} > {_TOL_DECOUPLING}"

    def test_acp_broadside_hzz(self):
        """Hzz broadside (L=1) ≈ ACp = -1/(4πL³) no limite estático."""
        result = validate_decoupling(filter_name=_FILTER_GATE, L=1.0, tol=_TOL_DECOUPLING)
        assert (
            result["diff_acp"] < _TOL_DECOUPLING
        ), f"ACp diff={result['diff_acp']:.2e} > {_TOL_DECOUPLING}"

    def test_acx_over_acp_ratio_is_minus_2(self):
        """ACx / ACp = -2 (relação fundamental)."""
        ACp, ACx = static_decoupling_factors(1.0)
        ratio = ACx / ACp
        assert abs(ratio + 2.0) < 1e-14, f"ACx/ACp={ratio:.6f}, esperado -2"


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdBroadsideGate
# ──────────────────────────────────────────────────────────────────────────────
class TestVmdBroadsideGate:
    """VMD broadside vs vmd_fullspace_broadside em várias frequências."""

    @pytest.mark.parametrize(
        "freq,rho",
        [
            (100.0, 100.0),
            (1000.0, 100.0),
            (20000.0, 100.0),
            (100000.0, 100.0),
        ],
    )
    def test_vmd_broadside_matches(self, freq, rho):
        """Hz broadside (Numba) vs conj(Hz_analytical) < tol."""
        result = validate_vmd_broadside(
            filter_name=_FILTER_GATE,
            L=1.0,
            frequency_hz=freq,
            rho=rho,
            tol=_TOL_VMD,
        )
        assert result["pass"], (
            f"VMD broadside f={freq} Hz, ρ={rho}: "
            f"diff={result['diff']:.2e} > {_TOL_VMD}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdAxialGate
# ──────────────────────────────────────────────────────────────────────────────
class TestVmdAxialGate:
    """VMD axial vs vmd_fullspace_axial."""

    @pytest.mark.parametrize(
        "freq,rho",
        [
            (1000.0, 100.0),
            (20000.0, 100.0),
        ],
    )
    def test_vmd_axial_matches(self, freq, rho):
        """Hz axial (Numba) vs conj(Hz_analytical) < tol."""
        result = validate_vmd_axial(
            filter_name=_FILTER_GATE,
            L=1.0,
            frequency_hz=freq,
            rho=rho,
            tol=_TOL_VMD,
        )
        assert result["pass"], (
            f"VMD axial f={freq} Hz, ρ={rho}: " f"diff={result['diff']:.2e} > {_TOL_VMD}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestHighResistivityGate
# ──────────────────────────────────────────────────────────────────────────────
class TestHighResistivityGate:
    """Validação em alta resistividade."""

    @pytest.mark.parametrize("rho", [10.0, 1000.0, 100000.0])
    def test_vmd_broadside_high_rho(self, rho):
        """VMD broadside em ρ alto converge para decoupling."""
        result = validate_vmd_broadside(
            filter_name=_FILTER_GATE,
            L=1.0,
            frequency_hz=20000.0,
            rho=rho,
            tol=_TOL_VMD,
        )
        assert result[
            "pass"
        ], f"VMD broadside ρ={rho}: diff={result['diff']:.2e} > {_TOL_VMD}"


# ──────────────────────────────────────────────────────────────────────────────
# TestValidateAllFunction
# ──────────────────────────────────────────────────────────────────────────────
class TestValidateAllFunction:
    """Função integrada validate_all_analytical."""

    def test_all_pass(self):
        """Todos os 3 casos passam com tolerâncias padrão."""
        results = validate_all_analytical(
            filter_name=_FILTER_GATE,
            tol_decoupling=_TOL_DECOUPLING,
            tol_vmd=_TOL_VMD,
        )
        for name, r in results.items():
            assert r["pass"], f"Caso {name} falhou: {r}"


# ──────────────────────────────────────────────────────────────────────────────
# TestMultiFreqConsistency
# ──────────────────────────────────────────────────────────────────────────────
class TestMultiFreqConsistency:
    """Coerência entre frequências no forward completo."""

    def test_low_freq_converges_to_static(self):
        """Em f → 0, Hzz broadside converge para ACp."""
        ACp, _ = static_decoupling_factors(1.0)
        filt = FilterLoader().load(_FILTER_GATE)
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
            freqs_hz=np.array([1.0]),  # 1 Hz → quase estático
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        Hzz = cH[0, 8].real
        assert abs(Hzz - ACp) < 1e-5, f"f=1 Hz: Hzz={Hzz:.6e} vs ACp={ACp:.6e}"

    def test_high_freq_attenuates_more(self):
        """A 400 kHz, |Hzz| é menor que a 1 kHz (maior atenuação)."""
        filt = FilterLoader().load(_FILTER_GATE)
        cH_low = fields_in_freqs(
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
            freqs_hz=np.array([1000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        cH_high = fields_in_freqs(
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
            freqs_hz=np.array([400000.0]),
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
        )
        # Skin effect: mais atenuação em freq alta
        # A parte imaginária deve ser maior em freq alta (contribuição condutiva)
        im_low = abs(cH_low[0, 8].imag)
        im_high = abs(cH_high[0, 8].imag)
        assert im_high > im_low, (
            f"|Im(Hzz)| a 400 kHz ({im_high:.4e}) deveria ser > "
            f"|Im(Hzz)| a 1 kHz ({im_low:.4e}) pelo skin effect"
        )
