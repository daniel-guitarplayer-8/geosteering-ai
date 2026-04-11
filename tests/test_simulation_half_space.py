# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_half_space.py                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto    : Geosteering AI v2.0                                         ║
# ║  Subsistema : Simulador Python — Validação Half-Space (Sprint 1.3)       ║
# ║  Autor      : Daniel Leal                                                 ║
# ║  Criação    : 2026-04-11                                                 ║
# ║  Framework  : pytest + NumPy                                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garante que as soluções analíticas fechadas de referência para      ║
# ║    EM 1D satisfazem todas as propriedades físicas esperadas:           ║
# ║                                                                           ║
# ║      1. Constantes de decoupling ACp/ACx bit-exatas com CLAUDE.md     ║
# ║      2. Skin depth com fórmula universal e dependência 1/√f, √ρ       ║
# ║      3. Número de onda com Im(k) > 0 (atenuação com r)                ║
# ║      4. VMD axial ≠ 0 e limite estático → ACx·m                       ║
# ║      5. VMD broadside ≠ 0 e limite estático → ACp·m                   ║
# ║      6. Razão axial/broadside → -2 em ω → 0 (normalização correta)    ║
# ║      7. Decaimento com frequência (skin effect)                        ║
# ║      8. Simetria de escala em L e moment_Am2                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de `geosteering_ai.simulation.validation.half_space`."""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation.validation import (
    MU_0,
    skin_depth,
    static_decoupling_factors,
    vmd_fullspace_axial,
    vmd_fullspace_broadside,
    wavenumber_quasi_static,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE REFERÊNCIA (bit-exato com CLAUDE.md)
# ──────────────────────────────────────────────────────────────────────────────
# Valores aproximados documentados no CLAUDE.md para L = 1 m:
#   ACp = -1/(4π L³) ≈ -0.079577
#   ACx = +1/(2π L³) ≈ +0.159155
#
# Os valores abaixo são calculados diretamente da fórmula fechada (Python
# float64 puro) para comparação bit-exata sem depender da função testada.
_PI = np.pi
_ACP_L1 = -1.0 / (4.0 * _PI)  # ≈ -0.079577471545947...
_ACX_L1 = 1.0 / (2.0 * _PI)  # ≈ +0.159154943091895...

# Tolerâncias numéricas para comparação float64 em diferentes contextos:
#   _TOL_BITEXACT: comparação bit-a-bit (usada para limites estáticos)
#   _TOL_PHYSICAL: comparação com valor físico esperado (tolerância 1e-12)
_TOL_BITEXACT: float = 0.0
_TOL_PHYSICAL: float = 1.0e-12


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE DECOUPLING FACTORS (CASO 1)
# ──────────────────────────────────────────────────────────────────────────────
class TestStaticDecouplingFactors:
    """Decoupling factors ACp e ACx no limite estático (ω → 0)."""

    def test_l1_acp_matches_claude_md(self) -> None:
        """ACp(L=1) = -1/(4π) bit-exato."""
        ACp, _ = static_decoupling_factors(L=1.0)
        assert ACp == _ACP_L1

    def test_l1_acx_matches_claude_md(self) -> None:
        """ACx(L=1) = +1/(2π) bit-exato."""
        _, ACx = static_decoupling_factors(L=1.0)
        assert ACx == _ACX_L1

    def test_ratio_acx_to_neg_acp_equals_2(self) -> None:
        """ACx / (-ACp) = 2 para qualquer L (invariância geométrica)."""
        ACp, ACx = static_decoupling_factors(L=1.0)
        assert ACx / (-ACp) == 2.0

    def test_scaling_with_l_cube(self) -> None:
        """ACp e ACx escalam como 1/L³."""
        ACp1, ACx1 = static_decoupling_factors(L=1.0)
        ACp2, ACx2 = static_decoupling_factors(L=2.0)
        # L=2 → escalar 1/8 (1/L³)
        assert ACp2 == pytest.approx(ACp1 / 8.0, rel=_TOL_PHYSICAL)
        assert ACx2 == pytest.approx(ACx1 / 8.0, rel=_TOL_PHYSICAL)

    def test_sign_convention(self) -> None:
        """ACp negativo, ACx positivo (convenção geofísica)."""
        ACp, ACx = static_decoupling_factors(L=1.0)
        assert ACp < 0
        assert ACx > 0

    def test_rejects_zero_L(self) -> None:
        with pytest.raises(AssertionError, match="L=0"):
            static_decoupling_factors(L=0.0)

    def test_rejects_negative_L(self) -> None:
        with pytest.raises(AssertionError, match="L=-1"):
            static_decoupling_factors(L=-1.0)


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE SKIN DEPTH (CASO 2)
# ──────────────────────────────────────────────────────────────────────────────
class TestSkinDepth:
    """Profundidade de penetração em meio condutor homogêneo."""

    def test_scalar_input(self) -> None:
        """Entrada escalar retorna escalar (não array 0-d)."""
        delta = skin_depth(20000.0, 1.0)
        assert isinstance(delta, float)
        assert delta > 0

    def test_formula_closed_form(self) -> None:
        """δ = √(ρ / (π f μ₀)) bit-exato."""
        f, rho = 20000.0, 1.0
        delta = skin_depth(f, rho)
        expected = np.sqrt(rho / (np.pi * f * MU_0))
        assert delta == expected

    def test_inverse_freq_dependence(self) -> None:
        """δ ∝ 1/√f — dobrando a frequência, δ cai por √2."""
        d1 = skin_depth(20000.0, 1.0)
        d2 = skin_depth(40000.0, 1.0)
        assert d2 == pytest.approx(d1 / np.sqrt(2.0), rel=_TOL_PHYSICAL)

    def test_sqrt_resistivity_dependence(self) -> None:
        """δ ∝ √ρ — quadruplicando ρ, δ dobra."""
        d1 = skin_depth(20000.0, 1.0)
        d2 = skin_depth(20000.0, 4.0)
        assert d2 == pytest.approx(d1 * 2.0, rel=_TOL_PHYSICAL)

    def test_typical_lwd_20khz_1ohm(self) -> None:
        """Caso típico LWD: f=20 kHz, ρ=1 Ω·m → δ ≈ 3.56 m."""
        delta = skin_depth(20000.0, 1.0)
        assert delta == pytest.approx(3.559, abs=0.01)

    def test_array_broadcast(self) -> None:
        """Entrada array retorna array do mesmo shape."""
        freqs = np.array([2000.0, 20000.0, 200000.0])
        deltas = skin_depth(freqs, 1.0)
        assert isinstance(deltas, np.ndarray)
        assert deltas.shape == (3,)
        # Decrescente em frequência
        assert deltas[0] > deltas[1] > deltas[2]

    def test_rejects_zero_frequency(self) -> None:
        with pytest.raises(AssertionError, match="frequency_hz"):
            skin_depth(0.0, 1.0)

    def test_rejects_negative_resistivity(self) -> None:
        with pytest.raises(AssertionError, match="resistivity_ohm_m"):
            skin_depth(20000.0, -1.0)


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE NÚMERO DE ONDA QUASI-ESTÁTICO (CASO 3)
# ──────────────────────────────────────────────────────────────────────────────
class TestWavenumberQuasiStatic:
    """Número de onda complexo k no limite quasi-estático."""

    def test_returns_complex(self) -> None:
        k = wavenumber_quasi_static(20000.0, 1.0)
        assert isinstance(k, complex)

    def test_positive_imaginary_part(self) -> None:
        """Im(k) > 0 garante decaimento do campo com r crescente."""
        k = wavenumber_quasi_static(20000.0, 1.0)
        assert k.imag > 0

    def test_equal_real_and_imag(self) -> None:
        """Para k² = iωμ₀σ (imaginário puro positivo), Re(k) == Im(k)."""
        k = wavenumber_quasi_static(20000.0, 1.0)
        assert k.real == pytest.approx(k.imag, rel=_TOL_PHYSICAL)

    def test_k_magnitude_times_delta_equals_sqrt2(self) -> None:
        """|k| · δ = √2 (relação fundamental quasi-estática)."""
        delta = skin_depth(20000.0, 1.0)
        k = wavenumber_quasi_static(20000.0, 1.0)
        assert abs(k) * delta == pytest.approx(np.sqrt(2.0), rel=1e-10)

    def test_scales_with_sqrt_frequency(self) -> None:
        """|k| ∝ √f — quadruplicando f, |k| dobra."""
        k1 = wavenumber_quasi_static(20000.0, 1.0)
        k2 = wavenumber_quasi_static(80000.0, 1.0)
        assert abs(k2) == pytest.approx(2.0 * abs(k1), rel=_TOL_PHYSICAL)

    def test_rejects_zero_frequency(self) -> None:
        with pytest.raises(AssertionError):
            wavenumber_quasi_static(0.0, 1.0)

    def test_rejects_zero_resistivity(self) -> None:
        with pytest.raises(AssertionError):
            wavenumber_quasi_static(20000.0, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE VMD FULL-SPACE AXIAL (CASO 4)
# ──────────────────────────────────────────────────────────────────────────────
class TestVmdFullspaceAxial:
    """Campo VMD em full-space isotrópico, geometria axial (θ=0)."""

    def test_returns_complex(self) -> None:
        H = vmd_fullspace_axial(1.0, 20000.0, 1.0)
        assert isinstance(H, complex)

    def test_nonzero_for_typical_case(self) -> None:
        """LWD típico: f=20kHz, ρ=1 Ω·m, L=1 m → campo não nulo."""
        H = vmd_fullspace_axial(1.0, 20000.0, 1.0)
        assert abs(H) > 0

    def test_static_limit_matches_acx(self) -> None:
        """Limite ω → 0 (f muito baixo): Re(H) = ACx, Im(H) ≈ 0."""
        H = vmd_fullspace_axial(L=1.0, frequency_hz=1.0e-6, resistivity_ohm_m=1.0)
        _, ACx = static_decoupling_factors(L=1.0)
        # Em f muito baixo, a parte real converge para ACx exatamente.
        assert H.real == pytest.approx(ACx, abs=1.0e-10)
        # A parte imaginária é muito pequena (∝ kL).
        assert abs(H.imag) < 1.0e-6

    def test_moment_linearity(self) -> None:
        """H escala linearmente com o momento do dipolo."""
        H1 = vmd_fullspace_axial(1.0, 20000.0, 1.0, moment_Am2=1.0)
        H2 = vmd_fullspace_axial(1.0, 20000.0, 1.0, moment_Am2=5.0)
        assert H2 == pytest.approx(5.0 * H1, rel=_TOL_PHYSICAL)

    def test_l_cube_scaling_static(self) -> None:
        """No limite estático, H escala como 1/L³."""
        H1 = vmd_fullspace_axial(L=1.0, frequency_hz=1e-6, resistivity_ohm_m=1.0)
        H2 = vmd_fullspace_axial(L=2.0, frequency_hz=1e-6, resistivity_ohm_m=1.0)
        assert H2.real == pytest.approx(H1.real / 8.0, rel=1.0e-10)

    def test_skin_effect_attenuation(self) -> None:
        """Alta frequência → módulo menor (pele EM contrai campo)."""
        # Escolhemos ρ=1 e L grande para observar atenuação clara.
        H_low = vmd_fullspace_axial(L=5.0, frequency_hz=1000.0, resistivity_ohm_m=1.0)
        H_high = vmd_fullspace_axial(
            L=5.0,
            frequency_hz=500000.0,
            resistivity_ohm_m=1.0,
        )
        assert abs(H_high) < abs(H_low)

    def test_rejects_zero_L(self) -> None:
        with pytest.raises(AssertionError, match="L=0"):
            vmd_fullspace_axial(0.0, 20000.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# TESTES DE VMD FULL-SPACE BROADSIDE (CASO 5)
# ──────────────────────────────────────────────────────────────────────────────
class TestVmdFullspaceBroadside:
    """Campo VMD em full-space isotrópico, geometria broadside (θ=π/2)."""

    def test_returns_complex(self) -> None:
        H = vmd_fullspace_broadside(1.0, 20000.0, 1.0)
        assert isinstance(H, complex)

    def test_nonzero_for_typical_case(self) -> None:
        H = vmd_fullspace_broadside(1.0, 20000.0, 1.0)
        assert abs(H) > 0

    def test_static_limit_matches_acp(self) -> None:
        """Limite ω → 0: Re(H) = ACp (negativo), Im(H) ≈ 0."""
        H = vmd_fullspace_broadside(
            L=1.0,
            frequency_hz=1.0e-6,
            resistivity_ohm_m=1.0,
        )
        ACp, _ = static_decoupling_factors(L=1.0)
        assert H.real == pytest.approx(ACp, abs=1.0e-10)
        assert abs(H.imag) < 1.0e-6

    def test_static_limit_is_negative(self) -> None:
        """ACp < 0 → broadside static Re < 0."""
        H = vmd_fullspace_broadside(1.0, 1e-6, 1.0)
        assert H.real < 0

    def test_moment_linearity(self) -> None:
        H1 = vmd_fullspace_broadside(1.0, 20000.0, 1.0, moment_Am2=1.0)
        H2 = vmd_fullspace_broadside(1.0, 20000.0, 1.0, moment_Am2=3.0)
        assert H2 == pytest.approx(3.0 * H1, rel=_TOL_PHYSICAL)


# ──────────────────────────────────────────────────────────────────────────────
# TESTES CROSS-CUTTING (relações entre casos)
# ──────────────────────────────────────────────────────────────────────────────
class TestCrossCuttingRelations:
    """Relações que conectam múltiplos casos analíticos.

    Note:
        Estes testes não validam um único caso, mas a **consistência
        cruzada** entre os 5 casos analíticos. Se um dos casos estiver
        errado, pelo menos uma dessas relações falhará.
    """

    def test_static_limit_ratio_axial_broadside(self) -> None:
        """No limite estático, H_axial / H_broadside = -2."""
        H_ax = vmd_fullspace_axial(L=1.0, frequency_hz=1e-6, resistivity_ohm_m=1.0)
        H_br = vmd_fullspace_broadside(
            L=1.0,
            frequency_hz=1e-6,
            resistivity_ohm_m=1.0,
        )
        ratio = H_ax.real / H_br.real
        assert ratio == pytest.approx(-2.0, abs=1.0e-10)

    def test_decoupling_factors_vs_vmd_static(self) -> None:
        """ACp/ACx (caso 1) == limites estáticos de VMD (casos 4, 5)."""
        ACp, ACx = static_decoupling_factors(L=1.0)
        H_ax = vmd_fullspace_axial(1.0, 1e-6, 1.0).real
        H_br = vmd_fullspace_broadside(1.0, 1e-6, 1.0).real
        assert H_ax == pytest.approx(ACx, abs=1e-10)
        assert H_br == pytest.approx(ACp, abs=1e-10)

    def test_skin_depth_vs_wavenumber(self) -> None:
        """|k| · δ = √2 vincula casos 2 e 3."""
        for f in [1e3, 1e4, 1e5, 1e6]:
            for rho in [0.1, 1.0, 100.0, 1000.0]:
                delta = skin_depth(f, rho)
                k = wavenumber_quasi_static(f, rho)
                assert abs(k) * delta == pytest.approx(
                    np.sqrt(2.0),
                    rel=1e-10,
                ), f"Falha em f={f}, ρ={rho}"

    def test_vmd_axial_frequency_independence_static(self) -> None:
        """Em f → 0, o valor independe de ρ (só depende da geometria)."""
        H_rho1 = vmd_fullspace_axial(1.0, 1e-6, 1.0)
        H_rho100 = vmd_fullspace_axial(1.0, 1e-6, 100.0)
        # Ambos convergem para ACx com alta precisão.
        assert H_rho1.real == pytest.approx(H_rho100.real, abs=1e-10)
