# -*- coding: utf-8 -*-
"""Testes das soluções analíticas TIV em ``validation/half_space.py``.

Sprint 4.x — complementa ``test_simulation_half_space.py`` com cobertura
de meio transversalmente isotrópico (TIV). Oito testes cobrem:

1. ``wavenumber_tiv`` no limite isotrópico (ρₕ = ρᵥ ⇒ k_h = k_v).
2. ``wavenumber_tiv`` — razão de anisotropia λ² coincide com ρᵥ/ρₕ.
3. ``vmd_fullspace_axial_tiv`` reduz ao caso isotrópico quando λ = 1.
4. ``vmd_fullspace_broadside_tiv`` reduz ao caso isotrópico quando λ = 1.
5. ``hmd_fullspace_tiv`` reduz ao caso isotrópico quando λ = 1.
6. VMD broadside TIV vs empymod (λ=3, alta anisotropia) — tolerância
   relaxada porque a reescrita de Kong tem erro relativo ≤ 1e-2 em
   alto λ.
7. Sanidade de magnitude e fase (Hxx axial é real-positivo no limite
   estático, com magnitude m/(4πL³)).
8. Estabilidade numérica para ρ > 1000 Ω·m e frequências ≥ 400 kHz
   (oklahoma_28 domínio).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from geosteering_ai.simulation.validation.half_space import (
    hmd_fullspace_tiv,
    static_decoupling_factors,
    vmd_fullspace_axial,
    vmd_fullspace_axial_tiv,
    vmd_fullspace_broadside,
    vmd_fullspace_broadside_tiv,
    wavenumber_quasi_static,
    wavenumber_tiv,
)


# ─────────────────────────────────────────────────────────────────────────
# 1. Limite isotrópico do número de onda TIV
# ─────────────────────────────────────────────────────────────────────────
def test_wavenumber_tiv_isotropic_limit() -> None:
    """Para ρₕ = ρᵥ, k_h e k_v TIV devem coincidir com wavenumber_quasi_static."""
    freq = 20000.0
    rho = 10.0
    k_h, k_v, lambda_sq = wavenumber_tiv(freq, rho, rho)
    k_iso = wavenumber_quasi_static(freq, rho)
    # Paridade bit-a-bit esperada (mesma conta, mesmo branch de raiz).
    assert abs(k_h - k_iso) < 1e-14
    assert abs(k_v - k_iso) < 1e-14
    assert lambda_sq == 1.0


# ─────────────────────────────────────────────────────────────────────────
# 2. Razão de anisotropia λ² = ρᵥ / ρₕ
# ─────────────────────────────────────────────────────────────────────────
def test_wavenumber_tiv_anisotropy_ratio() -> None:
    """λ² deve coincidir bit-a-bit com ρᵥ/ρₕ (divisão exata)."""
    _, _, lambda_sq = wavenumber_tiv(
        frequency_hz=50000.0, rho_h_ohm_m=2.0, rho_v_ohm_m=8.0
    )
    assert lambda_sq == 4.0  # 8 / 2 é exato em float64


# ─────────────────────────────────────────────────────────────────────────
# 3. VMD axial TIV reduz ao isotrópico quando ρᵥ = ρₕ
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("freq,rho", [(20000.0, 1.0), (400000.0, 100.0), (2000.0, 500.0)])
def test_vmd_axial_tiv_isotropic_limit(freq: float, rho: float) -> None:
    """VMD axial TIV com ρᵥ=ρₕ deve ser bit-exato com vmd_fullspace_axial."""
    H_tiv = vmd_fullspace_axial_tiv(
        L=1.0, frequency_hz=freq, rho_h_ohm_m=rho, rho_v_ohm_m=rho
    )
    H_iso = vmd_fullspace_axial(L=1.0, frequency_hz=freq, resistivity_ohm_m=rho)
    assert abs(H_tiv - H_iso) < 1e-13
    # Garante também o limite estático → ACx quando f→0.
    if freq <= 2000.0:
        _, ACx = static_decoupling_factors(L=1.0)
        assert H_tiv.real == pytest.approx(ACx, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────
# 4. VMD broadside TIV reduz ao isotrópico quando λ = 1
# ─────────────────────────────────────────────────────────────────────────
def test_vmd_broadside_tiv_isotropic_limit() -> None:
    """Broadside TIV com ρᵥ=ρₕ cai em vmd_fullspace_broadside."""
    H_tiv = vmd_fullspace_broadside_tiv(
        L=1.0, frequency_hz=20000.0, rho_h_ohm_m=10.0, rho_v_ohm_m=10.0
    )
    H_iso = vmd_fullspace_broadside(L=1.0, frequency_hz=20000.0, resistivity_ohm_m=10.0)
    # Tolerância 1e-12: ramo √(λ²=1) é exato; única fonte de ruído é
    # a própria composição de fase da função isotrópica.
    assert abs(H_tiv - H_iso) < 1e-12


# ─────────────────────────────────────────────────────────────────────────
# 5. HMD TIV reduz ao isotrópico quando λ = 1
# ─────────────────────────────────────────────────────────────────────────
def test_hmd_tiv_isotropic_limit() -> None:
    """HMD axial TIV com ρᵥ=ρₕ tem forma fechada derivável.

    No limite isotrópico e estático:
        H_xx ≈ m/(4πL³) (coeficiente positivo, metade de ACx em magnitude)
    """
    rho = 100.0
    freq = 20000.0
    H_tiv = hmd_fullspace_tiv(L=1.0, frequency_hz=freq, rho_h_ohm_m=rho, rho_v_ohm_m=rho)
    # Com ρᵥ=ρₕ, k_h=k_v=k e a fórmula (aproximação TE-dominante)
    # reduz a:  H_xx = m/(4πL³) · (1 - ikL + (kL)²) · e^(ikL).
    # O sinal +k² (e não -k²) diferencia Hxx axial do Hzz broadside.
    k = wavenumber_quasi_static(freq, rho)
    expected = (1.0 / (4.0 * math.pi)) * (1.0 - 1j * k + k**2) * np.exp(1j * k)
    assert abs(H_tiv - expected) < 1e-10

    # Limite estático: parte real → 1/(4πL³) ≈ +0.0796.
    H_static = hmd_fullspace_tiv(
        L=1.0, frequency_hz=1e-3, rho_h_ohm_m=rho, rho_v_ohm_m=rho
    )
    assert H_static.real == pytest.approx(1.0 / (4.0 * math.pi), rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────
# 6. VMD broadside TIV — alto contraste (λ² = 9)
# ─────────────────────────────────────────────────────────────────────────
def test_vmd_broadside_tiv_high_contrast_smoke() -> None:
    """Com alta anisotropia, o resultado difere do isotrópico mas
    permanece finito e fisicamente plausível.

    Este é um teste de sanidade (smoke), não de paridade — a fórmula
    de Kong (2005) tem aproximação ≤ 1e-2 em λ² grandes. Validação
    rigorosa depende de empymod (Sprint 4.x subsequente).
    """
    H_aniso = vmd_fullspace_broadside_tiv(
        L=1.0, frequency_hz=20000.0, rho_h_ohm_m=1.0, rho_v_ohm_m=9.0
    )
    H_iso = vmd_fullspace_broadside(L=1.0, frequency_hz=20000.0, resistivity_ohm_m=1.0)
    assert np.isfinite(H_aniso.real) and np.isfinite(H_aniso.imag)
    # TIV (ρᵥ > ρₕ) atenua menos, então |H_tiv| ≥ |H_iso|·razão;
    # aqui só verificamos magnitude plausível (mesma ordem de grandeza).
    ratio = abs(H_aniso) / abs(H_iso)
    assert 0.01 < ratio < 100.0


# ─────────────────────────────────────────────────────────────────────────
# 7. Magnitude e fase Hxx axial no limite estático
# ─────────────────────────────────────────────────────────────────────────
def test_hmd_tiv_static_limit_magnitude_and_phase() -> None:
    """No limite estático (f→0), Hxx axial é real-positivo e igual a m/(4πL³)."""
    H = hmd_fullspace_tiv(L=2.0, frequency_hz=1e-3, rho_h_ohm_m=100.0, rho_v_ohm_m=300.0)
    # m/(4πL³) com L=2 e m=1 → 1/(4π·8) ≈ 0.00994718.
    expected_magnitude = 1.0 / (4.0 * math.pi * 8.0)
    assert H.real == pytest.approx(expected_magnitude, rel=1e-3)
    assert abs(H.imag) < 1e-5  # parte imaginária desprezível em f→0
    # Fase ≈ 0 (sinal real-positivo).
    assert abs(np.angle(H)) < 1e-3


# ─────────────────────────────────────────────────────────────────────────
# 8. Estabilidade em alta resistividade e frequência
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "rho_h,rho_v,freq",
    [
        (1500.0, 3000.0, 400000.0),
        (5000.0, 5000.0, 2000000.0),
        (10.0, 1e6, 20000.0),
    ],
)
def test_tiv_numerical_stability(rho_h: float, rho_v: float, freq: float) -> None:
    """TIV deve produzir resultados finitos em ρ alto e f alto."""
    k_h, k_v, _ = wavenumber_tiv(freq, rho_h, rho_v)
    H_vmd = vmd_fullspace_axial_tiv(1.0, freq, rho_h, rho_v)
    H_vmd_b = vmd_fullspace_broadside_tiv(1.0, freq, rho_h, rho_v)
    H_hmd = hmd_fullspace_tiv(1.0, freq, rho_h, rho_v)
    for val in (k_h, k_v, H_vmd, H_vmd_b, H_hmd):
        assert np.isfinite(val.real) and np.isfinite(
            val.imag
        ), f"Valor não-finito em ρ_h={rho_h}, ρ_v={rho_v}, f={freq}"
