# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_forward.py                                         ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Testes API Forward (Sprint 2.5)        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da API pública simulate() (Sprint 2.5).

Baterias:
- TestSimulateBasic (6): shapes, dtype, result type, multi-freq, configs.
- TestSimulatePhysics (5): axial ACx, constância em full-space, 3 camadas.
- TestSimulateHighRes (3): estabilidade em ρ alto + NaN-free.
- TestSimulateGeometry (3): dip=0° vs dip=30°, multi-posição.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from geosteering_ai.simulation import SimulationConfig, simulate
from geosteering_ai.simulation.forward import SimulationResult
from geosteering_ai.simulation.validation.half_space import static_decoupling_factors


class TestSimulateBasic:
    """Testes estruturais da API simulate."""

    def test_returns_simulation_result(self):
        """Retorno é SimulationResult."""
        result = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
        )
        assert isinstance(result, SimulationResult)

    def test_shape_single_freq(self):
        """Shape (n_pos, 1, 9) para single-freq."""
        result = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.linspace(-2, 2, 10),
            frequency_hz=20000.0,
        )
        assert result.H_tensor.shape == (10, 1, 9)
        assert result.H_tensor.dtype == np.complex128
        assert result.z_obs.shape == (10,)
        assert result.rho_h_at_obs.shape == (10,)
        assert result.rho_v_at_obs.shape == (10,)

    def test_shape_multi_freq(self):
        """Shape (n_pos, 3, 9) para multi-freq via config."""
        cfg = SimulationConfig(frequencies_hz=[20000.0, 100000.0, 400000.0])
        result = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
            cfg=cfg,
        )
        assert result.H_tensor.shape == (1, 3, 9)
        assert result.freqs_hz.shape == (3,)

    def test_z_obs_is_midpoint(self):
        """z_obs é o ponto-médio TR para dip=0°."""
        result = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([5.0, 10.0]),
            tr_spacing_m=1.0,
        )
        np.testing.assert_allclose(result.z_obs, [5.0, 10.0])

    def test_rho_at_obs_correct_in_fullspace(self):
        """Full-space: rho_h_at_obs == rho_h[0] para todas as posições."""
        result = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([200.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.linspace(-5, 5, 10),
        )
        np.testing.assert_allclose(result.rho_h_at_obs, 100.0)
        np.testing.assert_allclose(result.rho_v_at_obs, 200.0)

    def test_cfg_preserved_in_result(self):
        """O config usado é preservado no resultado."""
        cfg = SimulationConfig(frequency_hz=40000.0)
        result = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
            cfg=cfg,
        )
        assert result.cfg.frequency_hz == 40000.0


class TestSimulatePhysics:
    """Testes de correto físico."""

    def test_fullspace_axial_hzz_equals_acx(self):
        """Hzz com ferramenta vertical (dip=0°) em full-space ≈ ACx."""
        _, ACx = static_decoupling_factors(1.0)
        result = simulate(
            rho_h=np.array([1.0e8]),  # quase vácuo → limite estático
            rho_v=np.array([1.0e8]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
            frequency_hz=10.0,  # baixa freq
            tr_spacing_m=1.0,
        )
        Hzz = result.H_tensor[0, 0, 8].real
        assert (
            abs(Hzz - ACx) < 1e-4
        ), f"Hzz={Hzz:.6f} vs ACx={ACx:.6f} diff={abs(Hzz - ACx):.2e}"

    def test_fullspace_constant_across_positions(self):
        """Full-space isotrópico: tensor constante em todas as posições."""
        result = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.linspace(-10, 10, 20),
            frequency_hz=20000.0,
        )
        # Todas as posições devem dar o mesmo Hzz
        Hzz_vals = result.H_tensor[:, 0, 8]
        spread = np.abs(Hzz_vals - Hzz_vals[0]).max()
        assert spread < 1e-12, f"spread Hzz across positions = {spread:.2e}"

    def test_fullspace_all_finite(self):
        """Full-space com posições de -10 a +10 não gera NaN/Inf."""
        result = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.linspace(-10, 10, 50),
        )
        assert np.all(np.isfinite(result.H_tensor)), "NaN/Inf no H_tensor"

    def test_3_layer_contrast_varies_with_position(self):
        """3 camadas contrastantes: Hzz varia com z (não constante)."""
        result = simulate(
            rho_h=np.array([1.0, 100.0, 1.0]),
            rho_v=np.array([1.0, 100.0, 1.0]),
            esp=np.array([5.0]),
            positions_z=np.linspace(-2, 7, 50),
        )
        Hzz = result.H_tensor[:, 0, 8]
        # O tensor deve variar à medida que a ferramenta passa pelas camadas
        spread = np.abs(Hzz.max() - Hzz.min())
        assert (
            spread > 1e-3
        ), f"Hzz deveria variar com z em 3 camadas, spread={spread:.2e}"

    def test_tiv_vs_isotropic_different(self):
        """TIV (ρh ≠ ρv) dá resultado diferente do isotrópico."""
        result_iso = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
        )
        result_tiv = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([500.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
        )
        assert not np.allclose(result_iso.H_tensor, result_tiv.H_tensor, atol=1e-10)


class TestSimulateHighRes:
    """Estabilidade em alta resistividade."""

    @pytest.mark.parametrize("rho", [1.0, 1000.0, 100000.0, 1000000.0])
    def test_high_rho_no_nan(self, rho):
        """ρ até 10⁶ Ω·m sem NaN/Inf."""
        result = simulate(
            rho_h=np.array([rho]),
            rho_v=np.array([rho]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
        )
        assert np.all(np.isfinite(result.H_tensor))


class TestSimulateGeometry:
    """Testes de geometria (dip, espaçamento)."""

    def test_dip_changes_tensor(self):
        """dip=0° produz tensor diferente de dip=30°."""
        r0 = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
            dip_deg=0.0,
        )
        r30 = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
            dip_deg=30.0,
        )
        assert not np.allclose(r0.H_tensor, r30.H_tensor, atol=1e-6)

    def test_different_spacing_different_result(self):
        """L=0.5 m vs L=2.0 m dá resultados diferentes."""
        r_short = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
            tr_spacing_m=0.5,
        )
        r_long = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
            tr_spacing_m=2.0,
        )
        assert not np.allclose(r_short.H_tensor, r_long.H_tensor, atol=1e-10)

    def test_hankel_filter_override(self):
        """Override de filtro Hankel funciona (Kong vs Werthmüller)."""
        r_w = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
            hankel_filter="werthmuller_201pt",
        )
        r_k = simulate(
            rho_h=np.array([100.0]),
            rho_v=np.array([100.0]),
            esp=np.zeros(0, dtype=np.float64),
            positions_z=np.array([0.0]),
            hankel_filter="kong_61pt",
        )
        # Valores devem ser próximos mas não idênticos (filtros diferentes)
        Hzz_w = r_w.H_tensor[0, 0, 8]
        Hzz_k = r_k.H_tensor[0, 0, 8]
        # Devem concordar em ~1e-3 (ambos são filtros de boa qualidade)
        assert abs(Hzz_w - Hzz_k) < 1e-3
        # Mas não bit-exatos (filtros com npt diferente)
        assert abs(Hzz_w - Hzz_k) > 1e-10
