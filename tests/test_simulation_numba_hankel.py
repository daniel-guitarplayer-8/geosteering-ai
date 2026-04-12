# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_numba_hankel.py                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Testes Hankel Helpers (Sprint 2.3)     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de _numba/hankel.py (Sprint 2.3)."""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation._numba.hankel import (
    integrate_j0,
    integrate_j0_j1,
    integrate_j1,
    prepare_kr,
)


class TestPrepareKr:
    """Escala das abscissas do filtro Hankel."""

    def test_basic_scaling(self):
        abs_f = np.array([0.1, 1.0, 10.0])
        kr = prepare_kr(2.0, abs_f)
        np.testing.assert_allclose(kr, [0.05, 0.5, 5.0])

    def test_r_guard_when_hordist_zero(self):
        """hordist < eps → usa r=0.01."""
        abs_f = np.array([1.0])
        kr = prepare_kr(1e-15, abs_f)  # praticamente zero
        assert kr[0] == 100.0  # 1.0 / 0.01

    def test_dtype_float64(self):
        abs_f = np.array([1.0], dtype=np.float64)
        kr = prepare_kr(1.0, abs_f)
        assert kr.dtype == np.float64


class TestIntegrateJ0:
    """integrate_j0 soma discreta."""

    def test_zero_f_returns_zero(self):
        """f=0 → integral=0."""
        f = np.zeros(10, dtype=np.complex128)
        w = np.ones(10, dtype=np.float64)
        result = integrate_j0(f, w, 1.0)
        assert result == 0.0 + 0.0j

    def test_constant_f_gives_sum_divided_by_r(self):
        """f=const → integral = const·Σw/r."""
        f = np.full(5, 2.0 + 0j, dtype=np.complex128)
        w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # soma = 15
        result = integrate_j0(f, w, 2.0)
        expected = 2.0 * 15.0 / 2.0  # = 15
        assert abs(result - expected) < 1e-14

    def test_complex_f(self):
        """f complex → resultado complex."""
        f = np.array([1 + 1j, 2, 3 - 1j], dtype=np.complex128)
        w = np.array([1.0, 1.0, 1.0])
        result = integrate_j0(f, w, 1.0)
        # Σ = (1+1j) + 2 + (3-1j) = 6+0j
        expected = 6.0 + 0.0j
        assert abs(result - expected) < 1e-14


class TestIntegrateJ1:
    """integrate_j1 soma discreta."""

    def test_constant_f_gives_sum_divided_by_r(self):
        f = np.full(3, 4.0 + 0j, dtype=np.complex128)
        w = np.array([0.5, 0.5, 1.0])  # soma = 2.0
        result = integrate_j1(f, w, 4.0)
        expected = 4.0 * 2.0 / 4.0  # = 2
        assert abs(result - expected) < 1e-14

    def test_zero_f_returns_zero(self):
        f = np.zeros(7, dtype=np.complex128)
        w = np.ones(7)
        assert integrate_j1(f, w, 3.0) == 0.0 + 0.0j


class TestIntegrateJ0J1:
    """integrate_j0_j1 retorna ambas as integrais."""

    def test_matches_individual_calls(self):
        """integrate_j0_j1(f, wJ0, wJ1, r) == (integrate_j0(...), integrate_j1(...))."""
        f = np.array([1 + 0.5j, 2 - 0.3j, 3, 0.5j], dtype=np.complex128)
        wJ0 = np.array([0.1, 0.2, 0.3, 0.4])
        wJ1 = np.array([0.5, 0.6, 0.7, 0.8])
        r = 2.5

        combined = integrate_j0_j1(f, wJ0, wJ1, r)
        separate = (integrate_j0(f, wJ0, r), integrate_j1(f, wJ1, r))

        assert abs(combined[0] - separate[0]) < 1e-14
        assert abs(combined[1] - separate[1]) < 1e-14

    def test_returns_tuple_of_2(self):
        f = np.zeros(5, dtype=np.complex128)
        w = np.ones(5)
        result = integrate_j0_j1(f, w, w, 1.0)
        assert len(result) == 2
        assert result[0] == 0 + 0j
        assert result[1] == 0 + 0j


class TestHankelFilterIntegration:
    """Integração com FilterLoader real."""

    def test_werthmuller_201pt_works(self):
        """Helpers funcionam com filtro Werthmüller 201pt."""
        from geosteering_ai.simulation.filters import FilterLoader

        filt = FilterLoader().load("werthmuller_201pt")
        kr = prepare_kr(1.0, filt.abscissas)
        assert kr.shape == filt.abscissas.shape
        assert kr.dtype == np.float64

        # Integral de f=1: depende do filtro, mas deve ser finito
        f = np.ones(filt.abscissas.shape[0], dtype=np.complex128)
        int0 = integrate_j0(f, filt.weights_j0, 1.0)
        int1 = integrate_j1(f, filt.weights_j1, 1.0)
        assert np.isfinite(int0.real)
        assert np.isfinite(int1.real)
