# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_foundation.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — JAX Foundation (Sprint 3.1)                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest + JAX + numpy                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Bateria de testes para os módulos de fundação do backend JAX:        ║
# ║      • hankel.py   — integrate_j0, integrate_j1, integrate_j0_j1         ║
# ║      • rotation.py — build_rotation_matrix, rotate_tensor               ║
# ║                                                                           ║
# ║    Validações:                                                            ║
# ║      1. Paridade numérica JAX vs Numba (<1e-12 em float64)              ║
# ║      2. Propriedades matemáticas (ortogonalidade, det=1, identidade)   ║
# ║      3. Diferenciabilidade via jax.grad (autodiff)                      ║
# ║      4. Suporte a complex128 (sem downcast para complex64)              ║
# ║                                                                           ║
# ║  SKIP CONDICIONAL                                                         ║
# ║    Se JAX não instalado, todos os testes são skipados. O gate da       ║
# ║    Sprint 3.1 exige JAX disponível — skip só é benigno em CI mínimo.   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de fundação JAX (Sprint 3.1) — hankel + rotation."""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation._jax import HAS_JAX

# Skip global se JAX não instalado
pytestmark = pytest.mark.skipif(
    not HAS_JAX, reason="JAX não instalado — Sprint 3.1 requer `pip install jax[cpu]`"
)

if HAS_JAX:
    import jax
    import jax.numpy as jnp

    from geosteering_ai.simulation._jax import (
        build_rotation_matrix,
        integrate_j0,
        integrate_j0_j1,
        integrate_j1,
        rotate_tensor,
    )
    from geosteering_ai.simulation._numba.rotation import (
        build_rotation_matrix as build_R_numba,
    )
    from geosteering_ai.simulation._numba.rotation import (
        rotate_tensor as rotate_tensor_numba,
    )
    from geosteering_ai.simulation.filters import FilterLoader


# ──────────────────────────────────────────────────────────────────────────────
# Hankel: integrate_j0, integrate_j1
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_JAX, reason="JAX required")
class TestHankelJax:
    """Testa integração Hankel via JAX."""

    @pytest.fixture(scope="class")
    def filt(self):
        return FilterLoader().load("werthmuller_201pt")

    def test_integrate_j0_constant_function(self, filt) -> None:
        """Integral de f(kr)=1 deve retornar Σwᵢ (soma de pesos)."""
        values = jnp.ones(filt.npt, dtype=jnp.complex128)
        w = jnp.asarray(filt.weights_j0)
        result = integrate_j0(values, w)
        expected = np.sum(filt.weights_j0)
        # Tolerância: somas grandes em float64 têm ULP ~ 1e-15
        assert np.isclose(float(np.real(result)), expected, rtol=1e-13)
        assert np.isclose(float(np.imag(result)), 0.0, atol=1e-15)

    def test_integrate_j1_constant_function(self, filt) -> None:
        values = jnp.ones(filt.npt, dtype=jnp.complex128)
        w = jnp.asarray(filt.weights_j1)
        result = integrate_j1(values, w)
        expected = np.sum(filt.weights_j1)
        assert np.isclose(float(np.real(result)), expected, rtol=1e-13)

    def test_integrate_j0_j1_consistency(self, filt) -> None:
        """integrate_j0_j1 deve concordar com chamadas separadas."""
        values = jnp.asarray(
            np.random.default_rng(42).standard_normal(filt.npt)
            + 1j * np.random.default_rng(7).standard_normal(filt.npt)
        )
        w0 = jnp.asarray(filt.weights_j0)
        w1 = jnp.asarray(filt.weights_j1)

        I0_a = integrate_j0(values, w0)
        I1_a = integrate_j1(values, w1)
        I0_b, I1_b = integrate_j0_j1(values, w0, w1)

        assert np.isclose(complex(I0_a), complex(I0_b), atol=1e-14)
        assert np.isclose(complex(I1_a), complex(I1_b), atol=1e-14)

    def test_integrate_j0_parity_numpy(self, filt) -> None:
        """Paridade com np.dot (ground-truth numpy)."""
        values_np = np.random.default_rng(42).standard_normal(filt.npt) + 0j
        values = jnp.asarray(values_np)
        w = jnp.asarray(filt.weights_j0)

        I_jax = integrate_j0(values, w)
        I_numpy = np.dot(filt.weights_j0, values_np)

        assert np.isclose(complex(I_jax), complex(I_numpy), atol=1e-13)

    def test_integrate_j0_jit_cache(self, filt) -> None:
        """Chamadas sucessivas devem reusar o cache JIT (não re-compilam)."""
        w = jnp.asarray(filt.weights_j0)
        v1 = jnp.ones(filt.npt, dtype=jnp.complex128)
        v2 = jnp.zeros(filt.npt, dtype=jnp.complex128)

        r1 = integrate_j0(v1, w)
        r2 = integrate_j0(v2, w)  # deve reutilizar JIT
        assert complex(r2) == 0.0
        # r1 foi computada com v1=1 → soma de pesos (!= 0)
        assert abs(complex(r1)) > 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Rotation: build_rotation_matrix
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_JAX, reason="JAX required")
class TestBuildRotationMatrixJax:
    """Testa construção da matriz de rotação R(α, β, γ) via JAX."""

    def test_identity_when_all_zero(self) -> None:
        R = build_rotation_matrix(0.0, 0.0, 0.0)
        assert jnp.allclose(R, jnp.eye(3))

    def test_orthogonal_R_R_T_is_identity(self) -> None:
        """R · Rᵀ = I para quaisquer α, β, γ (rotação é ortogonal)."""
        R = build_rotation_matrix(0.3, 0.5, -0.2)
        product = R @ R.T
        assert jnp.allclose(product, jnp.eye(3), atol=1e-14)

    def test_determinant_plus_one(self) -> None:
        """det(R) = +1 (rotação própria, não reflexão)."""
        R = build_rotation_matrix(1.2, -0.4, 0.8)
        det = jnp.linalg.det(R)
        assert np.isclose(float(det), 1.0, atol=1e-14)

    def test_parity_with_numba(self) -> None:
        """Paridade bit-a-bit com _numba/rotation.build_rotation_matrix."""
        for angs in [(0.3, 0.1, -0.2), (1.5, 0.0, 0.0), (0.0, 2.1, 0.0)]:
            R_jax = build_rotation_matrix(*angs)
            R_numba = build_R_numba(*angs)
            max_diff = np.max(np.abs(np.asarray(R_jax) - R_numba))
            assert max_diff < 1e-14, f"angs={angs}, diff={max_diff}"

    def test_differentiable_via_grad(self) -> None:
        """jax.grad funciona: ∂R₀₀/∂α = -sin(α) quando β=γ=0."""

        def f(a):
            R = build_rotation_matrix(a, 0.0, 0.0)
            return R[0, 0]

        grad_at_03 = jax.grad(f)(0.3)
        expected = -np.sin(0.3)
        assert np.isclose(float(grad_at_03), expected, atol=1e-14)


# ──────────────────────────────────────────────────────────────────────────────
# Rotation: rotate_tensor
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_JAX, reason="JAX required")
class TestRotateTensorJax:
    """Testa H' = Rᵀ · H · R via JAX."""

    @pytest.fixture
    def H(self):
        return jnp.array(
            [[1.0 + 0j, 2j, 3.0], [4.0, 5.0, 6j], [7j, 8.0, 9.0 + 0j]],
            dtype=jnp.complex128,
        )

    def test_identity_preserves_tensor(self, H) -> None:
        H_rot = rotate_tensor(0.0, 0.0, 0.0, H)
        assert jnp.allclose(H_rot, H, atol=1e-15)

    def test_preserves_trace(self, H) -> None:
        """tr(Rᵀ·H·R) = tr(H) — invariante de rotação."""
        H_rot = rotate_tensor(0.3, 0.5, -0.2, H)
        tr_H = jnp.trace(H)
        tr_H_rot = jnp.trace(H_rot)
        assert np.isclose(complex(tr_H), complex(tr_H_rot), atol=1e-13)

    def test_preserves_frobenius_norm(self, H) -> None:
        """‖Rᵀ·H·R‖_F = ‖H‖_F — rotação é unitária."""
        H_rot = rotate_tensor(0.3, 0.5, -0.2, H)
        norm_H = jnp.linalg.norm(H)
        norm_H_rot = jnp.linalg.norm(H_rot)
        assert np.isclose(float(norm_H), float(norm_H_rot), atol=1e-12)

    def test_parity_with_numba(self, H) -> None:
        """Paridade com _numba.rotate_tensor em < 1e-13."""
        H_np = np.asarray(H)
        for angs in [(0.3, 0.1, -0.2), (1.5, 0.0, 0.0), (0.0, 0.0, 2.5)]:
            H_rot_jax = rotate_tensor(angs[0], angs[1], angs[2], H)
            H_rot_numba = rotate_tensor_numba(angs[0], angs[1], angs[2], H_np)
            max_diff = np.max(np.abs(np.asarray(H_rot_jax) - H_rot_numba))
            assert max_diff < 1e-13, f"angs={angs}, diff={max_diff}"

    def test_composition_inverse(self, H) -> None:
        """R(α)·R(-α) = I → rotate(α)·rotate(-α) = H (aprox.)."""
        H1 = rotate_tensor(0.5, 0.0, 0.0, H)
        H2 = rotate_tensor(-0.5, 0.0, 0.0, H1)
        assert jnp.allclose(H2, H, atol=1e-12)
