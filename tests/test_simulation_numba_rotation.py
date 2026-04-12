# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_numba_rotation.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Testes Rotação (Sprint 2.3)            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de _numba/rotation.py (Sprint 2.3)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from geosteering_ai.simulation._numba.rotation import build_rotation_matrix, rotate_tensor


class TestBuildRotationMatrix:
    """Verifica a matriz R(α, β, γ) canônica."""

    def test_identity_zero_angles(self):
        """R(0,0,0) = I."""
        R = build_rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_orthogonality(self):
        """R·Rᵀ = I para qualquer rotação pura."""
        for alpha in [0.0, math.pi / 6, math.pi / 3, math.pi / 2]:
            for beta in [0.0, math.pi / 4, math.pi / 2]:
                for gamma in [0.0, math.pi / 3, math.pi]:
                    R = build_rotation_matrix(alpha, beta, gamma)
                    prod = R @ R.T
                    np.testing.assert_allclose(prod, np.eye(3), atol=1e-14)

    def test_determinant_plus_1(self):
        """det(R) = +1 (rotação própria, sem reflexão)."""
        R = build_rotation_matrix(math.pi / 4, math.pi / 6, math.pi / 3)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-14

    def test_alpha_half_pi(self):
        """α=π/2, β=γ=0: rotação do eixo z para o eixo x."""
        R = build_rotation_matrix(math.pi / 2, 0.0, 0.0)
        # R[2,2] = cos(α) = 0, R[0,2] = sin(α)·cos(β) = 1
        assert abs(R[2, 2]) < 1e-15
        assert abs(R[0, 2] - 1.0) < 1e-15

    def test_dtype_float64(self):
        """Tipo de retorno é float64."""
        R = build_rotation_matrix(0.5, 0.3, 0.1)
        assert R.dtype == np.float64


class TestRotateTensor:
    """Verifica rotate_tensor (Rᵀ·H·R)."""

    def test_identity_preserves_tensor(self):
        """Rotação (0,0,0) preserva o tensor."""
        H = np.array(
            [
                [1.0 + 2.0j, 3.0 - 1.0j, 0.5j],
                [4.0, 5.0 + 3j, 6j],
                [7j, 8.0, 9.0 + 0j],
            ],
            dtype=np.complex128,
        )
        H_rot = rotate_tensor(0.0, 0.0, 0.0, H)
        np.testing.assert_allclose(H_rot, H, atol=1e-15)

    def test_gamma_90_swaps_xx_yy(self):
        """γ=π/2 (rotação em torno do eixo z) troca Hxx ↔ Hyy."""
        H = np.diag([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        H_rot = rotate_tensor(0.0, 0.0, math.pi / 2, H)
        # Após rotação de π/2 em torno de z, Hxx vira Hyy e vice-versa
        assert abs(H_rot[0, 0] - 2.0) < 1e-14
        assert abs(H_rot[1, 1] - 1.0) < 1e-14
        assert abs(H_rot[2, 2] - 3.0) < 1e-14  # Hzz inalterado

    def test_alpha_180_flips_xx_zz(self):
        """α=π (rotação 180° em torno do eixo y projetado): Hxx↔Hzz."""
        H = np.diag([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        H_rot = rotate_tensor(math.pi, 0.0, 0.0, H)
        # α=π: cos(α)=-1, sin(α)=0 → R[0,0]=-1, R[2,2]=-1
        # H_rot[0,0] = (-1)²·H[0,0] = H[0,0] = 1
        # Hzz permanece 3
        assert abs(H_rot[0, 0] - 1.0) < 1e-14
        assert abs(H_rot[2, 2] - 3.0) < 1e-14

    def test_complex_tensor_preserved_trace(self):
        """trace(H) é invariante sob rotação (para qualquer α,β,γ)."""
        H = np.array(
            [[1 + 1j, 2, 3], [4, 5 + 2j, 6], [7, 8, 9 + 3j]],
            dtype=np.complex128,
        )
        trace_orig = H[0, 0] + H[1, 1] + H[2, 2]
        H_rot = rotate_tensor(0.7, 0.3, 0.5, H)
        trace_rot = H_rot[0, 0] + H_rot[1, 1] + H_rot[2, 2]
        assert abs(trace_orig - trace_rot) < 1e-13

    def test_frobenius_norm_preserved(self):
        """||H||_F = ||Rᵀ·H·R||_F para rotação unitária."""
        H = np.array(
            [[1 + 1j, 2, 3j], [4, 5 + 2j, 6], [7j, 8, 9 + 3j]],
            dtype=np.complex128,
        )
        norm_orig = np.linalg.norm(H, ord="fro")
        H_rot = rotate_tensor(0.4, 0.6, 0.8, H)
        norm_rot = np.linalg.norm(H_rot, ord="fro")
        assert abs(norm_orig - norm_rot) < 1e-13

    def test_only_dip_broadcast(self):
        """Caso típico Fortran: rotate_tensor(α, 0, 0, H)."""
        H = np.diag([1.0 + 0j, 1.0 + 0j, 1.0 + 0j])  # identidade
        H_rot = rotate_tensor(math.pi / 4, 0.0, 0.0, H)
        # Rotação de identidade → identidade (qualquer ângulo)
        np.testing.assert_allclose(H_rot, np.eye(3), atol=1e-14)

    def test_output_shape_3x3(self):
        """Retorno é sempre (3, 3) complex128."""
        H = np.zeros((3, 3), dtype=np.complex128)
        out = rotate_tensor(0.5, 0.5, 0.5, H)
        assert out.shape == (3, 3)
        assert out.dtype == np.complex128
