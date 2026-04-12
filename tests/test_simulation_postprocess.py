# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_postprocess.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Testes Pós-processamento Sprint 2.2   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida os módulos de pós-processamento da Sprint 2.2:                 ║
# ║                                                                           ║
# ║      1. apply_compensation (F6) — compensação midpoint CDR              ║
# ║      2. apply_tilted_antennas (F7) — projeção para antenas inclinadas   ║
# ║                                                                           ║
# ║    Testes miram identidade (casos triviais), ortogonalidade            ║
# ║    (β=0 → Hzz, β=90° φ=0° → Hxz) e reciprocidade numérica.            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes F6/F7 pós-processamento (Sprint 2.2).

Baterias:

- **TestCompensationBasic** (6 testes): shapes, identidade, opt-in.
- **TestCompensationPhysics** (4 testes): fórmula CDR, dB/graus, zeros.
- **TestTiltedBasic** (5 testes): shapes, β=0, β=90 projeções.
- **TestTiltedOrtogonality** (5 testes): 4 casos canônicos + combinações.

Total: 20 testes.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from geosteering_ai.simulation.postprocess import (
    apply_compensation,
    apply_tilted_antennas,
)


# ──────────────────────────────────────────────────────────────────────────────
# TestCompensationBasic — 6 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestCompensationBasic:
    """Testes básicos de apply_compensation."""

    def test_output_shapes_1_pair(self):
        """1 par de compensação → output shape (1, ntheta, nmeds, nf, 9)."""
        H = np.random.randn(3, 1, 10, 1, 9).astype(np.complex128)
        H_comp, dphi, dalpha = apply_compensation(H, ((0, 1),))
        assert H_comp.shape == (1, 1, 10, 1, 9)
        assert dphi.shape == (1, 1, 10, 1, 9)
        assert dalpha.shape == (1, 1, 10, 1, 9)

    def test_output_shapes_multi_pair(self):
        """2 pares → dim 0 do output = 2."""
        H = np.random.randn(4, 2, 5, 3, 9).astype(np.complex128)
        H_comp, _, _ = apply_compensation(H, ((0, 1), (2, 3)))
        assert H_comp.shape == (2, 2, 5, 3, 9)

    def test_dtypes(self):
        """H_comp=complex128, dphi/dalpha=float64."""
        H = np.ones((2, 1, 5, 1, 9), dtype=np.complex128)
        H_comp, dphi, dalpha = apply_compensation(H, ((0, 1),))
        assert H_comp.dtype == np.complex128
        assert dphi.dtype == np.float64
        assert dalpha.dtype == np.float64

    def test_identity_when_near_equals_far(self):
        """Se H_near == H_far, H_comp == H_near, dphi=0, dalpha=0."""
        H = np.full((2, 1, 3, 1, 9), 1.5 + 0.5j, dtype=np.complex128)
        H_comp, dphi, dalpha = apply_compensation(H, ((0, 1),))
        assert np.allclose(H_comp[0], H[0])
        assert np.allclose(dphi, 0.0)
        assert np.allclose(dalpha, 0.0)

    def test_empty_comp_pairs_raises(self):
        """comp_pairs=() levanta ValueError."""
        H = np.ones((2, 1, 3, 1, 9), dtype=np.complex128)
        with pytest.raises(ValueError, match="vazio"):
            apply_compensation(H, ())

    def test_invalid_pair_index_raises(self):
        """Índice fora do range de n_tr levanta ValueError."""
        H = np.ones((2, 1, 3, 1, 9), dtype=np.complex128)
        with pytest.raises(ValueError, match="fora do range"):
            apply_compensation(H, ((0, 5),))


# ──────────────────────────────────────────────────────────────────────────────
# TestCompensationPhysics — 4 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestCompensationPhysics:
    """Testes da fórmula CDR e unidades."""

    def test_h_comp_is_average(self):
        """H_comp = 0.5 * (H_near + H_far)."""
        H_near = np.full((1, 1, 3, 1, 9), 1.0 + 0.0j, dtype=np.complex128)
        H_far = np.full((1, 1, 3, 1, 9), 3.0 + 0.0j, dtype=np.complex128)
        H = np.concatenate([H_near, H_far], axis=0)
        H_comp, _, _ = apply_compensation(H, ((0, 1),))
        assert np.allclose(H_comp, 2.0 + 0.0j)

    def test_attenuation_db_factor_2(self):
        """|H_near|=1, |H_far|=2 → dB = 20·log10(0.5) = -6.0206."""
        H = np.ones((2, 1, 3, 1, 9), dtype=np.complex128)
        H[1] = 2.0 + 0j
        _, _, dalpha = apply_compensation(H, ((0, 1),))
        expected = 20.0 * math.log10(0.5)
        np.testing.assert_allclose(dalpha, expected, atol=1e-10)

    def test_phase_diff_90_deg(self):
        """arg(1+0j) - arg(0+1j) = -π/2 → -90°."""
        H_near = np.full((1, 1, 3, 1, 9), 1.0 + 0.0j, dtype=np.complex128)
        H_far = np.full((1, 1, 3, 1, 9), 0.0 + 1.0j, dtype=np.complex128)
        H = np.concatenate([H_near, H_far], axis=0)
        _, dphi, _ = apply_compensation(H, ((0, 1),))
        np.testing.assert_allclose(dphi, -90.0, atol=1e-10)

    def test_zero_far_produces_nan_atten(self):
        """|H_far|=0 → atten=NaN (guarded por np.where)."""
        H = np.ones((2, 1, 3, 1, 9), dtype=np.complex128)
        H[1] = 0.0 + 0.0j  # far = 0
        _, _, dalpha = apply_compensation(H, ((0, 1),))
        # Quando far=0, ratio=NaN → 20·log10(NaN) = NaN
        assert np.all(np.isnan(dalpha))


# ──────────────────────────────────────────────────────────────────────────────
# TestTiltedBasic — 5 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestTiltedBasic:
    """Testes básicos de apply_tilted_antennas."""

    def test_output_shape(self):
        """Shape (n_tilted, *prefix)."""
        H = np.zeros((2, 5, 1, 9), dtype=np.complex128)
        out = apply_tilted_antennas(H, ((0.0, 0.0), (45.0, 0.0), (90.0, 0.0)))
        assert out.shape == (3, 2, 5, 1)

    def test_dtype_complex128(self):
        H = np.zeros((5, 9), dtype=np.complex128)
        out = apply_tilted_antennas(H, ((0.0, 0.0),))
        assert out.dtype == np.complex128

    def test_beta_zero_returns_hzz(self):
        """β=0° → H_tilted = Hzz puro (independentemente de φ)."""
        H = np.zeros((5, 9), dtype=np.complex128)
        H[:, 8] = 2.5 + 1.5j  # Hzz
        H[:, 2] = 99.0  # Hxz (irrelevante para β=0)
        H[:, 5] = -99.0  # Hyz (irrelevante)
        out = apply_tilted_antennas(H, ((0.0, 0.0), (0.0, 90.0)))
        assert np.allclose(out[0], H[:, 8])
        assert np.allclose(out[1], H[:, 8])

    def test_beta_90_phi_0_returns_hxz(self):
        """β=90°, φ=0° → H_tilted = Hxz puro."""
        H = np.zeros((5, 9), dtype=np.complex128)
        H[:, 2] = 1.0 + 0.5j  # Hxz
        H[:, 5] = 99.0  # Hyz irrelevante
        H[:, 8] = 99.0  # Hzz irrelevante
        out = apply_tilted_antennas(H, ((90.0, 0.0),))
        np.testing.assert_allclose(out[0], H[:, 2], atol=1e-12)

    def test_beta_90_phi_90_returns_hyz(self):
        """β=90°, φ=90° → H_tilted = Hyz puro."""
        H = np.zeros((5, 9), dtype=np.complex128)
        H[:, 5] = 1.0 + 0.5j  # Hyz
        H[:, 2] = 99.0  # Hxz irrelevante
        H[:, 8] = 99.0  # Hzz irrelevante
        out = apply_tilted_antennas(H, ((90.0, 90.0),))
        np.testing.assert_allclose(out[0], H[:, 5], atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# TestTiltedOrtogonality — 5 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestTiltedOrtogonality:
    """Testes de ortogonalidade e combinações canônicas."""

    def test_beta_45_phi_0_is_linear_combo(self):
        """β=45°, φ=0° → H = cos(45)·Hzz + sin(45)·Hxz."""
        H = np.zeros((1, 9), dtype=np.complex128)
        H[0, 2] = 4.0 + 0j  # Hxz
        H[0, 8] = 2.0 + 0j  # Hzz
        out = apply_tilted_antennas(H, ((45.0, 0.0),))
        expected = math.cos(math.pi / 4) * 2.0 + math.sin(math.pi / 4) * 4.0
        np.testing.assert_allclose(out[0, 0], expected, atol=1e-12)

    def test_beta_45_phi_90_uses_hyz(self):
        """β=45°, φ=90° → H = cos(45)·Hzz + sin(45)·Hyz (cos(90)=0)."""
        H = np.zeros((1, 9), dtype=np.complex128)
        H[0, 2] = 99.0  # Hxz NÃO deve aparecer
        H[0, 5] = 4.0 + 0j  # Hyz
        H[0, 8] = 2.0 + 0j  # Hzz
        out = apply_tilted_antennas(H, ((45.0, 90.0),))
        expected = math.cos(math.pi / 4) * 2.0 + math.sin(math.pi / 4) * 4.0
        np.testing.assert_allclose(out[0, 0], expected, atol=1e-12)

    def test_phi_180_negates_x(self):
        """φ=180° → cos(180)=-1 → H = cos(β)·Hzz - sin(β)·Hxz."""
        H = np.zeros((1, 9), dtype=np.complex128)
        H[0, 2] = 4.0 + 0j  # Hxz
        H[0, 8] = 2.0 + 0j  # Hzz
        out = apply_tilted_antennas(H, ((90.0, 180.0),))
        # cos(90)=0, sin(90)=1, cos(180)=-1 → -Hxz
        np.testing.assert_allclose(out[0, 0], -H[0, 2], atol=1e-12)

    def test_multiple_configs_independent(self):
        """Múltiplas configs produzem resultados independentes."""
        H = np.zeros((3, 9), dtype=np.complex128)
        H[:, 8] = 1.0 + 0j
        H[:, 2] = 2.0 + 0j
        out = apply_tilted_antennas(H, ((0.0, 0.0), (90.0, 0.0), (45.0, 0.0)))
        # out[0] = Hzz = 1
        np.testing.assert_allclose(out[0], 1.0 + 0j)
        # out[1] = Hxz = 2
        np.testing.assert_allclose(out[1], 2.0 + 0j)
        # out[2] = cos(45)*1 + sin(45)*2
        expected = math.cos(math.pi / 4) + math.sin(math.pi / 4) * 2.0
        np.testing.assert_allclose(out[2], expected, atol=1e-12)

    def test_empty_configs_raises(self):
        """tilted_configs=() levanta ValueError."""
        H = np.zeros((5, 9), dtype=np.complex128)
        with pytest.raises(ValueError, match="vazio"):
            apply_tilted_antennas(H, ())
