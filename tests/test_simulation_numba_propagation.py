# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_numba_propagation.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto    : Geosteering AI v2.0                                         ║
# ║  Subsistema : Simulador Python — Sprint 2.1 (propagation Numba)          ║
# ║  Autor      : Daniel Leal                                                 ║
# ║  Criação    : 2026-04-11                                                 ║
# ║  Framework  : pytest + NumPy                                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida as duas funções portadas de Fortran para Python+Numba na     ║
# ║    Sprint 2.1 (`common_arrays` e `common_factors`), verificando:       ║
# ║                                                                           ║
# ║      1. Shapes, dtypes, contiguidade e não-mutação dos inputs.          ║
# ║      2. Limite 1-camada homogêneo (s = u, RT = 0).                      ║
# ║      3. Limite TIV com razão de anisotropia λ = σh/σv = 4.              ║
# ║      4. Invariantes da recursão (|RT| ≤ 1, condições terminais).        ║
# ║      5. Fatores de onda em common_factors (shapes, signos).             ║
# ║      6. Compilação JIT dual-mode (Numba opcional).                      ║
# ║                                                                           ║
# ║  ESTRATÉGIA                                                              ║
# ║    Os testes funcionam independentemente da presença de Numba. O      ║
# ║    sinalizador `HAS_NUMBA` é exposto em `_numba/propagation.py` e     ║
# ║    usado em `TestNumbaCompilation` para mensagens informativas.        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do backend Numba — Sprint 2.1 (propagation)."""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation._numba.propagation import (
    HAS_NUMBA,
    common_arrays,
    common_factors,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE REFERÊNCIA
# ──────────────────────────────────────────────────────────────────────────────
# Caso "LWD típico" usado em vários testes:
#   frequência 20 kHz, ρ = 1 Ω·m, espaçamento 1 m, 201 pontos de quadratura.
_F_LWD: float = 20000.0
_MU_0: float = 4.0e-7 * np.pi
_ZETA_LWD: complex = 1j * 2.0 * np.pi * _F_LWD * _MU_0
_NPT_WER: int = 201
_KR_MIN: float = 0.001
_KR_MAX: float = 50.0


def _make_krJ0J1(npt: int) -> np.ndarray:
    """Cria um grid logarítmico de abscissas para testes de unidade.

    Note:
        Em produção, `krJ0J1` vem do `FilterLoader`. Para testes
        isolados, um grid linear simples é suficiente e mais rápido.
    """
    return np.linspace(_KR_MIN, _KR_MAX, npt)


# ──────────────────────────────────────────────────────────────────────────────
# GRUPO 1: SHAPES, DTYPES, CONTIGUIDADE, NÃO-MUTAÇÃO
# ──────────────────────────────────────────────────────────────────────────────
class TestCommonArraysShapes:
    """Validações estruturais de `common_arrays`."""

    @pytest.fixture
    def outputs(self) -> tuple:
        """Saída de common_arrays para um caso 3-camadas TIV."""
        n, npt = 3, _NPT_WER
        h = np.array([1.0, 5.0, 0.0], dtype=np.float64)
        eta = np.array([[1.0, 1.0], [0.01, 0.005], [1.0, 1.0]], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        return common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)

    def test_returns_nine_arrays(self, outputs: tuple) -> None:
        """common_arrays deve retornar exatamente 9 arrays."""
        assert len(outputs) == 9

    def test_all_outputs_complex128(self, outputs: tuple) -> None:
        """Todos os 9 arrays devem ser complex128."""
        for i, arr in enumerate(outputs):
            assert (
                arr.dtype == np.complex128
            ), f"Array {i} tem dtype {arr.dtype}, esperado complex128"

    def test_all_outputs_shape_npt_n(self, outputs: tuple) -> None:
        """Todos os 9 arrays devem ter shape (npt, n) = (201, 3)."""
        for i, arr in enumerate(outputs):
            assert arr.shape == (
                _NPT_WER,
                3,
            ), f"Array {i} tem shape {arr.shape}, esperado (201, 3)"

    def test_inputs_not_mutated(self) -> None:
        """common_arrays não deve modificar h, eta ou krJ0J1."""
        n, npt = 2, _NPT_WER
        h = np.array([2.0, 0.0], dtype=np.float64)
        eta = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        h_copy = h.copy()
        eta_copy = eta.copy()
        kr_copy = kr.copy()
        _ = common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)
        assert np.array_equal(h, h_copy), "h foi modificado"
        assert np.array_equal(eta, eta_copy), "eta foi modificado"
        assert np.array_equal(kr, kr_copy), "krJ0J1 foi modificado"


# ──────────────────────────────────────────────────────────────────────────────
# GRUPO 2: LIMITE 1-CAMADA HOMOGÊNEO ISOTRÓPICO
# ──────────────────────────────────────────────────────────────────────────────
class TestCommonArraysHomogeneousLimit:
    """Validações no limite 1-camada homogêneo isotrópico (σh = σv)."""

    @pytest.fixture
    def outputs_iso(self) -> tuple:
        n, npt = 1, _NPT_WER
        h = np.array([0.0], dtype=np.float64)
        eta = np.array([[1.0, 1.0]], dtype=np.float64)  # σh = σv = 1
        kr = _make_krJ0J1(npt)
        return common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)

    def test_s_equals_u_for_isotropic(self, outputs_iso: tuple) -> None:
        """Isotrópico: λ = 1 → s = sqrt(1)·sqrt(kr² + iωμσ) = u."""
        u, s = outputs_iso[0], outputs_iso[1]
        # Comparação bit-exata (não usar np.isclose porque atol≠0)
        assert np.array_equal(s, u), (
            f"s ≠ u no limite isotrópico. " f"|s-u|.max() = {np.abs(s-u).max():.2e}"
        )

    def test_u_squared_matches_analytical(self, outputs_iso: tuple) -> None:
        """u² deve ser kr² + zeta·σh (formulação equivalente a -kh²).

        Note:
            Tolerância `rtol=1e-13` (não bit-exata) porque `sqrt(x)·sqrt(x)`
            em float64 acumula arredondamento ~2 ULPs — tipicamente
            ~1e-15 em valores normalizados, mas pode chegar a 1e-13 para
            valores grandes (kr² ≈ 2500 no extremo do grid).
        """
        u = outputs_iso[0]
        kr = _make_krJ0J1(_NPT_WER)
        kr_squared = kr * kr
        expected_u_squared = kr_squared + _ZETA_LWD * 1.0  # σh = 1
        assert np.allclose(
            u[:, 0] * u[:, 0],
            expected_u_squared,
            rtol=1e-13,
            atol=0.0,
        )

    def test_admint_equals_u_over_zeta(self, outputs_iso: tuple) -> None:
        """AdmInt = u / zeta (definição direta).

        Note:
            Usa ``np.testing.assert_allclose`` com rtol=1e-13 em vez de
            ``np.array_equal`` porque o Numba JIT (LLVM) pode reordenar
            operações float, gerando diferenças na última casa decimal —
            comportamento esperado e documentado no Sprint 2.1.
        """
        u = outputs_iso[0]
        AdmInt = outputs_iso[8]
        np.testing.assert_allclose(AdmInt, u / _ZETA_LWD, rtol=1e-13)

    def test_rtedw_zero_for_single_layer(self, outputs_iso: tuple) -> None:
        """Sem interface inferior → RTEdw[:, n-1] = 0."""
        RTEdw = outputs_iso[4]
        assert np.all(RTEdw == 0)

    def test_rtmdw_zero_for_single_layer(self, outputs_iso: tuple) -> None:
        """Sem interface inferior → RTMdw[:, n-1] = 0."""
        RTMdw = outputs_iso[6]
        assert np.all(RTMdw == 0)


# ──────────────────────────────────────────────────────────────────────────────
# GRUPO 3: LIMITE TIV COM ANISOTROPIA λ = σh/σv = 4
# ──────────────────────────────────────────────────────────────────────────────
class TestCommonArraysTIVLimit:
    """Validações com meio TIV (anisotropia vertical)."""

    @pytest.fixture
    def outputs_tiv(self) -> tuple:
        n, npt = 1, _NPT_WER
        h = np.array([0.0], dtype=np.float64)
        # σh = 0.4, σv = 0.1 → λ = σh/σv = 4 (ρh=2.5, ρv=10)
        eta = np.array([[0.4, 0.1]], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        return common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)

    def test_s_squared_formula_tiv(self, outputs_tiv: tuple) -> None:
        """s² = λ·(kr² + zeta·σv) em meio TIV (forma fechada).

        Note:
            Tolerância `rtol=1e-13` pelo mesmo motivo de
            `test_u_squared_matches_analytical` (arredondamento
            acumulado em `sqrt(lambda)·sqrt(kr²+zeta·σv)`).
        """
        s = outputs_tiv[1]
        kr = _make_krJ0J1(_NPT_WER)
        lamb2 = 0.4 / 0.1  # λ = 4
        expected_s_squared = lamb2 * (kr * kr + _ZETA_LWD * 0.1)
        assert np.allclose(
            s[:, 0] * s[:, 0],
            expected_s_squared,
            rtol=1e-13,
            atol=0.0,
        )

    def test_s_not_equal_u_for_anisotropic(self, outputs_tiv: tuple) -> None:
        """Anisotrópico: s ≠ u (a não ser em λ = 1)."""
        u, s = outputs_tiv[0], outputs_tiv[1]
        assert not np.array_equal(s, u), "s == u em meio anisotrópico — sinal de bug"

    def test_u_unchanged_by_anisotropy(self, outputs_tiv: tuple) -> None:
        """u depende só de σh, não de σv (verificação cruzada)."""
        # Computa u com σv diferente mas σh igual, deve dar mesmo u.
        n, npt = 1, _NPT_WER
        h = np.array([0.0], dtype=np.float64)
        eta_other = np.array([[0.4, 0.05]], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        u_ref = outputs_tiv[0]
        u_other = common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta_other)[0]
        assert np.array_equal(u_ref, u_other)


# ──────────────────────────────────────────────────────────────────────────────
# GRUPO 4: INVARIANTES DA RECURSÃO (3+ CAMADAS)
# ──────────────────────────────────────────────────────────────────────────────
class TestCommonArraysRecursionInvariants:
    """Validações de estabilidade/convergência da recursão RT."""

    @pytest.fixture
    def outputs_3layer(self) -> tuple:
        """Modelo 3-camadas contrastantes: 1 Ω·m, 100 Ω·m, 1 Ω·m."""
        n, npt = 3, _NPT_WER
        h = np.array([2.0, 5.0, 0.0], dtype=np.float64)
        eta = np.array([[1.0, 1.0], [0.01, 0.005], [1.0, 1.0]], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        return common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)

    def test_rtedw_last_is_zero(self, outputs_3layer: tuple) -> None:
        """Terminal da recursão bottom-up: RTEdw[:, n-1] = 0."""
        RTEdw = outputs_3layer[4]
        assert np.all(RTEdw[:, -1] == 0)

    def test_rteup_first_is_zero(self, outputs_3layer: tuple) -> None:
        """Terminal da recursão top-down: RTEup[:, 0] = 0."""
        RTEup = outputs_3layer[5]
        assert np.all(RTEup[:, 0] == 0)

    def test_rtedw_bounded(self, outputs_3layer: tuple) -> None:
        """|RTEdw[:, i]| ≤ 1 para toda camada (estabilidade física)."""
        RTEdw = outputs_3layer[4]
        # Tolerância de 1e-12 para erros numéricos de arredondamento
        assert np.all(np.abs(RTEdw) <= 1.0 + 1e-12)

    def test_rtmup_bounded(self, outputs_3layer: tuple) -> None:
        """|RTMup[:, i]| ≤ 1 para toda camada (estabilidade física)."""
        RTMup = outputs_3layer[7]
        assert np.all(np.abs(RTMup) <= 1.0 + 1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# GRUPO 5: common_factors — SHAPES E TIPOS
# ──────────────────────────────────────────────────────────────────────────────
class TestCommonFactorsShapes:
    """Validações estruturais de `common_factors`."""

    @pytest.fixture
    def factors(self) -> tuple:
        """Fatores de onda para 3 camadas, transmissor na camada 1."""
        n, npt = 3, _NPT_WER
        h = np.array([1.0, 5.0, 0.0], dtype=np.float64)
        eta = np.array([[1.0, 1.0], [0.01, 0.005], [1.0, 1.0]], dtype=np.float64)
        prof = np.array([0.0, 1.0, 6.0, 6.0], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        outs = common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)
        u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, _ = outs
        return common_factors(
            n,
            npt,
            3.5,
            h,
            prof,
            1,
            u,
            s,
            uh,
            sh,
            RTEdw,
            RTEup,
            RTMdw,
            RTMup,
        )

    def test_returns_six_arrays(self, factors: tuple) -> None:
        assert len(factors) == 6

    def test_all_shapes_are_1d_npt(self, factors: tuple) -> None:
        for i, arr in enumerate(factors):
            assert arr.shape == (
                _NPT_WER,
            ), f"Fator {i} tem shape {arr.shape}, esperado ({_NPT_WER},)"

    def test_all_dtypes_complex128(self, factors: tuple) -> None:
        for i, arr in enumerate(factors):
            assert arr.dtype == np.complex128


# ──────────────────────────────────────────────────────────────────────────────
# GRUPO 6: common_factors — CONSISTÊNCIA FÍSICA NO LIMITE 1-CAMADA
# ──────────────────────────────────────────────────────────────────────────────
class TestCommonFactorsHomogeneousSelfConsistency:
    """Validações em meio 1-camada (RT = 0 → fatores degeneram)."""

    @pytest.fixture
    def factors_single_layer(self) -> tuple:
        """1 camada, transmissor no meio (z = 5 m)."""
        n, npt = 1, _NPT_WER
        h = np.array([10.0], dtype=np.float64)  # camada com espessura
        eta = np.array([[1.0, 1.0]], dtype=np.float64)
        # prof[0] = 0 (topo), prof[1] = 10 (fundo)
        prof = np.array([0.0, 10.0], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        outs = common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)
        u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, _ = outs
        return common_factors(
            n,
            npt,
            5.0,
            h,
            prof,
            0,
            u,
            s,
            uh,
            sh,
            RTEdw,
            RTEup,
            RTMdw,
            RTMup,
        )

    def test_mxdw_reduces_to_exp_minus_s_dz(self, factors_single_layer: tuple) -> None:
        """Com RT = 0, Mxdw = exp(-s·(bot-h0)) = exp(-s·5)."""
        n, npt = 1, _NPT_WER
        h = np.array([10.0], dtype=np.float64)
        eta = np.array([[1.0, 1.0]], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        outs = common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)
        s = outs[1]
        Mxdw = factors_single_layer[0]
        # Mxdw = exp(-s·5) / (1 - 0·0·exp) = exp(-s·5)
        expected = np.exp(-s[:, 0] * 5.0)
        assert np.allclose(Mxdw, expected, atol=1e-14, rtol=1e-14)

    def test_mxup_reduces_to_exp_plus_s_dz(self, factors_single_layer: tuple) -> None:
        """Com RT = 0, Mxup = exp(s·(top-h0)) = exp(-s·5)."""
        n, npt = 1, _NPT_WER
        h = np.array([10.0], dtype=np.float64)
        eta = np.array([[1.0, 1.0]], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        outs = common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)
        s = outs[1]
        Mxup = factors_single_layer[1]
        # Mxup = exp(s·(0 - 5)) = exp(-s·5)
        expected = np.exp(s[:, 0] * (0.0 - 5.0))
        assert np.allclose(Mxup, expected, atol=1e-14, rtol=1e-14)

    def test_eudw_matches_mxdw_when_s_equals_u(self, factors_single_layer: tuple) -> None:
        """Em meio isotrópico (s = u), Eudw e Mxdw diferem só por sinal
        do RT (ambos 0 aqui), então devem ser iguais."""
        Mxdw = factors_single_layer[0]
        Eudw = factors_single_layer[2]
        assert np.allclose(Mxdw, Eudw, atol=1e-14, rtol=1e-14)

    def test_fedwz_reduces_to_exp_minus_u_dz(self, factors_single_layer: tuple) -> None:
        """Com RT = 0, FEdwz = exp(-u·(bot-h0)) = exp(-u·5)."""
        n, npt = 1, _NPT_WER
        h = np.array([10.0], dtype=np.float64)
        eta = np.array([[1.0, 1.0]], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        outs = common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)
        u = outs[0]
        FEdwz = factors_single_layer[4]
        expected = np.exp(-u[:, 0] * 5.0)
        assert np.allclose(FEdwz, expected, atol=1e-14, rtol=1e-14)


# ──────────────────────────────────────────────────────────────────────────────
# GRUPO 7: COMPILAÇÃO DUAL-MODE NUMBA
# ──────────────────────────────────────────────────────────────────────────────
class TestNumbaCompilation:
    """Valida o comportamento dual-mode (Numba opcional)."""

    def test_has_numba_is_bool(self) -> None:
        """HAS_NUMBA deve ser bool (True ou False)."""
        assert isinstance(HAS_NUMBA, bool)

    def test_functions_callable_in_current_mode(self) -> None:
        """Independente de ter Numba, as funções são chamáveis.

        Note:
            Se Numba está disponível, este teste também valida a
            compilação JIT (primeira chamada gera cache). Se não,
            valida o no-op decorator.
        """
        n, npt = 1, 61  # usa npt=61 para ser rápido
        h = np.array([0.0], dtype=np.float64)
        eta = np.array([[1.0, 1.0]], dtype=np.float64)
        kr = _make_krJ0J1(npt)
        outs = common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)
        # Só verifica que não lançou exceção e retornou 9 items
        assert len(outs) == 9
        if HAS_NUMBA:
            # Segunda chamada deve ser cacheada (não gera código novo)
            outs2 = common_arrays(n, npt, 1.0, kr, _ZETA_LWD, h, eta)
            assert len(outs2) == 9
