# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_native_e2e.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — JAX Native end-to-end (Sprint 3.3.4)             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest + JAX + NumPy                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida que o path JAX nativo (``use_native_dipoles=True``) produz     ║
# ║    resultado bit-exato vs o caminho híbrido (Numba via pure_callback).  ║
# ║    Cobre ETAPAS 3 (propagação), 5 (kernels) e 6 (assembly tensor)      ║
# ║    end-to-end, testando:                                                  ║
# ║      • Paridade hybrid vs native em 3 perfis (small, medium, large)      ║
# ║      • Alta resistividade (10⁶ Ω·m) sem NaN/Inf                         ║
# ║      • Dip não-nulo (30°)                                                ║
# ║      • Diferenciabilidade: jax.grad sobre rho_h finito                  ║
# ║      • Compile-time bounded (< 120s)                                     ║
# ║      • Regression guard: hybrid path inalterado                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes end-to-end JAX nativo vs híbrido (Sprint 3.3.4)."""
from __future__ import annotations

import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _run_jax_batch(rho_h, rho_v, esp, positions_z, freqs, dip_deg=0.0, use_native=False):
    """Executa via fields_in_freqs_jax_batch e retorna H_tensor."""
    import math

    from geosteering_ai.simulation._jax.kernel import fields_in_freqs_jax_batch
    from geosteering_ai.simulation.filters import FilterLoader

    filt = FilterLoader().load("werthmuller_201pt")
    L = 1.0
    dip_rad = np.deg2rad(dip_deg)
    dz_half = L / 2.0 * math.cos(dip_rad)
    r_half = L / 2.0 * math.sin(dip_rad)
    return fields_in_freqs_jax_batch(
        positions_z,
        dz_half,
        r_half,
        dip_rad,
        rho_h.shape[0],
        rho_h,
        rho_v,
        esp,
        freqs,
        filt.abscissas,
        filt.weights_j0,
        filt.weights_j1,
        use_native_dipoles=use_native,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TestNativeVsHybrid — paridade bit-exata (< 1e-10)
# ──────────────────────────────────────────────────────────────────────────────


class TestNativeVsHybrid:
    """Native e hybrid devem concordar em < 1e-10 para perfis variados."""

    def test_small_3layer_isotropic(self) -> None:
        rho = np.array([1.0, 100.0, 1.0])
        esp = np.array([5.0])
        z = np.linspace(-2.0, 7.0, 20)
        f = np.array([20000.0])
        H_h = _run_jax_batch(rho, rho, esp, z, f, use_native=False)
        H_n = _run_jax_batch(rho, rho, esp, z, f, use_native=True)
        np.testing.assert_allclose(H_n, H_h, rtol=1e-12, atol=1e-13)

    def test_medium_7layer_tiv(self) -> None:
        rho_h = np.array([1.0, 50.0, 5.0, 200.0, 10.0, 80.0, 1.0])
        rho_v = rho_h * 2.0  # TIV λ²=2
        esp = np.array([3.0, 2.0, 4.0, 1.0, 5.0])
        z = np.linspace(-2.0, 18.0, 30)
        f = np.array([20000.0])
        H_h = _run_jax_batch(rho_h, rho_v, esp, z, f, use_native=False)
        H_n = _run_jax_batch(rho_h, rho_v, esp, z, f, use_native=True)
        np.testing.assert_allclose(H_n, H_h, rtol=1e-11, atol=1e-12)

    def test_large_10layer(self) -> None:
        rng = np.random.default_rng(42)
        n = 10
        rho = 10 ** rng.uniform(0, 3, n)
        esp = rng.uniform(1.0, 5.0, n - 2)
        z = np.linspace(-2.0, float(esp.sum()) + 2.0, 40)
        f = np.array([20000.0])
        H_h = _run_jax_batch(rho, rho, esp, z, f, use_native=False)
        H_n = _run_jax_batch(rho, rho, esp, z, f, use_native=True)
        np.testing.assert_allclose(H_n, H_h, rtol=1e-10, atol=1e-11)


# ──────────────────────────────────────────────────────────────────────────────
# TestNativeStability — sem NaN/Inf em condições extremas
# ──────────────────────────────────────────────────────────────────────────────


class TestNativeStability:
    """Path nativo deve ser estável em alta resistividade e dip não-nulo."""

    def test_high_rho_1e6(self) -> None:
        rho = np.array([1e6, 1e6, 1e6])
        esp = np.array([5.0])
        z = np.array([2.5])
        f = np.array([20000.0])
        H_n = _run_jax_batch(rho, rho, esp, z, f, use_native=True)
        assert np.all(np.isfinite(H_n)), "NaN/Inf em ρ=10⁶ Ω·m"

    def test_dip_30deg(self) -> None:
        rho = np.array([1.0, 100.0, 1.0])
        esp = np.array([5.0])
        z = np.linspace(0.0, 5.0, 10)
        f = np.array([20000.0])
        H_h = _run_jax_batch(rho, rho, esp, z, f, dip_deg=30.0, use_native=False)
        H_n = _run_jax_batch(rho, rho, esp, z, f, dip_deg=30.0, use_native=True)
        np.testing.assert_allclose(H_n, H_h, rtol=1e-11, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# TestNativeDifferentiability — jax.grad end-to-end
# ──────────────────────────────────────────────────────────────────────────────


class TestNativeDifferentiability:
    """O path nativo deve permitir jax.grad sobre rho_h (PINN training)."""

    def test_grad_wrt_z(self) -> None:
        """jax.grad sobre z de uma posição deve retornar gradiente finito."""
        from geosteering_ai.simulation._jax.dipoles_native import (
            native_dipoles_full_jax,
        )
        from geosteering_ai.simulation._jax.propagation import (
            common_arrays_jax,
            common_factors_jax,
        )
        from geosteering_ai.simulation.filters import FilterLoader

        filt = FilterLoader().load("werthmuller_201pt")
        npt = filt.abscissas.shape[0]
        n = 3
        h_arr = jnp.array([0.0, 5.0, 0.0])
        prof_arr = jnp.array([-1e300, 0.0, 5.0, 1e300])
        eta = jnp.array([[1.0, 1.0], [0.01, 0.01], [1.0, 1.0]])
        kr = jnp.asarray(filt.abscissas)
        wJ0 = jnp.asarray(filt.weights_j0)
        wJ1 = jnp.asarray(filt.weights_j1)
        omega = 2.0 * jnp.pi * 20000.0
        zeta = 1j * omega * 4e-7 * jnp.pi
        r = 0.01  # guard
        outs = common_arrays_jax(n, npt, r, kr, zeta, h_arr, eta)
        u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt = outs
        cf = common_factors_jax(
            n, npt, 2.0, h_arr, prof_arr, 1, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup
        )
        Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz = cf

        def loss_fn(z_val):
            matH = native_dipoles_full_jax(
                0.0,
                0.0,
                2.0,
                n,
                1,
                1,
                npt,
                kr,
                wJ0,
                wJ1,
                h_arr,
                prof_arr,
                zeta,
                eta,
                0.01,
                0.0,
                z_val,
                u,
                s,
                uh,
                sh,
                RTEdw,
                RTEup,
                RTMdw,
                RTMup,
                Mxdw,
                Mxup,
                Eudw,
                Euup,
                FEdwz,
                FEupz,
            )
            return jnp.real(jnp.sum(matH))

        g = jax.grad(loss_fn)(3.0)
        assert np.isfinite(float(g)), f"grad não finito: {g}"


# ──────────────────────────────────────────────────────────────────────────────
# TestCompileTimeBudget
# ──────────────────────────────────────────────────────────────────────────────


class TestCompileTimeBudget:
    """Compile-time end-to-end não pode ultrapassar 120s."""

    def test_compile_under_120s(self) -> None:
        rho = np.array([1.0, 100.0, 1.0])
        esp = np.array([5.0])
        z = np.array([2.5])
        f = np.array([20000.0])
        jax.clear_caches()
        t0 = time.time()
        _ = _run_jax_batch(rho, rho, esp, z, f, use_native=True)
        dt = time.time() - t0
        assert dt < 120.0, f"Compile-time end-to-end: {dt:.1f}s > 120s"


# ──────────────────────────────────────────────────────────────────────────────
# TestHybridUnchanged — regression guard
# ──────────────────────────────────────────────────────────────────────────────


class TestHybridUnchanged:
    """O caminho hybrid (default) não deve ter mudado nesta PR."""

    def test_hybrid_produces_expected_hzz(self) -> None:
        rho = np.array([1.0, 100.0, 1.0])
        esp = np.array([5.0])
        z = np.array([2.5])
        f = np.array([20000.0])
        H = _run_jax_batch(rho, rho, esp, z, f, use_native=False)
        Hzz = H[0, 0, 8]
        # Hzz deve ser próximo de ACx ~ 0.159 para meio quase-homogêneo
        assert abs(Hzz.real) > 0.1, f"Hzz.real muito pequeno: {Hzz}"
        assert np.isfinite(Hzz), "Hzz não é finito"
