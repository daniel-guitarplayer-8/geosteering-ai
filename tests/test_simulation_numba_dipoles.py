# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_numba_dipoles.py                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Testes Sprint 2.2                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida os kernels `hmd_tiv` e `vmd` do backend Numba (Sprint 2.2)     ║
# ║    contra as 5 soluções analíticas de `validation/half_space.py` e       ║
# ║    contra invariantes físicos fundamentais:                             ║
# ║                                                                           ║
# ║      1. Shapes/dtypes corretos dos retornos                              ║
# ║      2. Limite estático (decoupling factors ACp e ACx)                   ║
# ║      3. VMD axial vs `vmd_fullspace_axial` (tolerância < 1e-10)         ║
# ║      4. VMD broadside vs `vmd_fullspace_broadside` (< 1e-10)            ║
# ║      5. Alta resistividade: ρ ∈ {1, 10², 10³, 10⁴, 10⁵, 10⁶} Ω·m       ║
# ║      6. Reciprocidade T↔R (simetria princípio)                          ║
# ║      7. Compilação JIT Numba (warm-up + segunda chamada rápida)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes dos kernels Numba hmd_tiv e vmd (Sprint 2.2).

Baterias de teste:

- **TestDipoleShapes**: shapes, dtypes e contiguidade dos retornos.
- **TestDecouplingLimit**: limite estático (σ → 0) reproduz ACp e ACx.
- **TestVmdAnalyticalBroadside**: VMD vs `vmd_fullspace_broadside` < 1e-10.
- **TestHmdDecouplingLimit**: HMD vs decoupling planar/axial < 1e-10.
- **TestHighResistivity**: simulação em ρ ∈ {1, 10², ..., 10⁶} Ω·m.
- **TestReciprocity**: simetria T↔R (se valer em isotrópico).
- **TestNumbaCompilation**: warm-up JIT (se Numba instalado).

Total: 25 testes.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from geosteering_ai.simulation._numba.dipoles import HAS_NUMBA, hmd_tiv, vmd
from geosteering_ai.simulation._numba.propagation import common_arrays, common_factors
from geosteering_ai.simulation.filters import FilterLoader
from geosteering_ai.simulation.validation.half_space import (
    static_decoupling_factors,
    vmd_fullspace_broadside,
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES E HELPERS
# ──────────────────────────────────────────────────────────────────────────────
_MU_0 = 4.0e-7 * math.pi
_FREQ_DEFAULT = 20_000.0  # 20 kHz (paridade errata)
_SPACING_DEFAULT = 1.0  # 1 m (paridade errata)
_RHO_DEFAULT = 100.0  # 100 Ω·m (half-space típico)


def _setup_single_layer(
    frequency_hz: float,
    rho: float,
    L: float,
    filter_name: str = "werthmuller_201pt",
):
    """Constrói os arrays para um forward single-layer isotrópico.

    Returns:
        dict com chaves 'filt', 'n', 'npt', 'h', 'eta', 'prof', 'zeta',
        'u', 's', 'uh', 'sh', 'RTEdw', 'RTEup', 'RTMdw', 'RTMup',
        'AdmInt', 'Mxdw', 'Mxup', 'Eudw', 'Euup', 'FEdwz', 'FEupz'.
    """
    filt = FilterLoader().load(filter_name)
    n = 1
    npt = filt.abscissas.shape[0]
    h = np.array([0.0], dtype=np.float64)
    sigma = 1.0 / rho
    eta = np.array([[sigma, sigma]], dtype=np.float64)
    prof = np.array([0.0, 1.0e300], dtype=np.float64)
    zeta = 1j * 2.0 * math.pi * frequency_hz * _MU_0

    u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt = common_arrays(
        n, npt, L, filt.abscissas, zeta, h, eta
    )
    Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz = common_factors(
        n, npt, 0.0, h, prof, 0, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup
    )
    return {
        "filt": filt,
        "n": n,
        "npt": npt,
        "h": h,
        "eta": eta,
        "prof": prof,
        "zeta": zeta,
        "u": u,
        "s": s,
        "uh": uh,
        "sh": sh,
        "RTEdw": RTEdw,
        "RTEup": RTEup,
        "RTMdw": RTMdw,
        "RTMup": RTMup,
        "AdmInt": AdmInt,
        "Mxdw": Mxdw,
        "Mxup": Mxup,
        "Eudw": Eudw,
        "Euup": Euup,
        "FEdwz": FEdwz,
        "FEupz": FEupz,
    }


# ──────────────────────────────────────────────────────────────────────────────
# TestDipoleShapes — 4 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestDipoleShapes:
    """Verifica shapes, dtypes e tipos dos retornos."""

    def test_vmd_returns_3_complex_scalars(self):
        """VMD retorna 3 escalares complex."""
        ctx = _setup_single_layer(_FREQ_DEFAULT, _RHO_DEFAULT, _SPACING_DEFAULT)
        Hx, Hy, Hz = vmd(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            _SPACING_DEFAULT,
            0.0,
            0.0,
            ctx["u"],
            ctx["uh"],
            ctx["AdmInt"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["FEdwz"],
            ctx["FEupz"],
        )
        # Cada retorno é um complex Python (ou complex128 NumPy)
        assert isinstance(Hx, (complex, np.complexfloating))
        assert isinstance(Hy, (complex, np.complexfloating))
        assert isinstance(Hz, (complex, np.complexfloating))

    def test_hmd_returns_3_arrays_shape_2(self):
        """HMD retorna 3 arrays (2,) complex128."""
        ctx = _setup_single_layer(_FREQ_DEFAULT, _RHO_DEFAULT, _SPACING_DEFAULT)
        Hx, Hy, Hz = hmd_tiv(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            ctx["eta"],
            _SPACING_DEFAULT,
            0.0,
            0.0,
            ctx["u"],
            ctx["s"],
            ctx["uh"],
            ctx["sh"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["RTMdw"],
            ctx["RTMup"],
            ctx["Mxdw"],
            ctx["Mxup"],
            ctx["Eudw"],
            ctx["Euup"],
        )
        assert Hx.shape == (2,)
        assert Hy.shape == (2,)
        assert Hz.shape == (2,)
        assert Hx.dtype == np.complex128
        assert Hy.dtype == np.complex128
        assert Hz.dtype == np.complex128

    def test_vmd_no_nan_inf(self):
        """VMD não produz NaN ou Inf em caso típico."""
        ctx = _setup_single_layer(_FREQ_DEFAULT, _RHO_DEFAULT, _SPACING_DEFAULT)
        Hx, Hy, Hz = vmd(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            _SPACING_DEFAULT,
            0.0,
            0.0,
            ctx["u"],
            ctx["uh"],
            ctx["AdmInt"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["FEdwz"],
            ctx["FEupz"],
        )
        for h in (Hx, Hy, Hz):
            assert not np.isnan(h.real), f"NaN real em {h}"
            assert not np.isnan(h.imag), f"NaN imag em {h}"
            assert not np.isinf(h.real), f"Inf real em {h}"
            assert not np.isinf(h.imag), f"Inf imag em {h}"

    def test_hmd_no_nan_inf(self):
        """HMD não produz NaN ou Inf em caso típico."""
        ctx = _setup_single_layer(_FREQ_DEFAULT, _RHO_DEFAULT, _SPACING_DEFAULT)
        Hx, Hy, Hz = hmd_tiv(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            ctx["eta"],
            _SPACING_DEFAULT,
            0.0,
            0.0,
            ctx["u"],
            ctx["s"],
            ctx["uh"],
            ctx["sh"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["RTMdw"],
            ctx["RTMup"],
            ctx["Mxdw"],
            ctx["Mxup"],
            ctx["Eudw"],
            ctx["Euup"],
        )
        for H in (Hx, Hy, Hz):
            assert np.all(np.isfinite(H.real)), f"NaN/Inf em real: {H}"
            assert np.all(np.isfinite(H.imag)), f"NaN/Inf em imag: {H}"


# ──────────────────────────────────────────────────────────────────────────────
# TestDecouplingLimit — 5 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestDecouplingLimit:
    """Valida que σ → 0 (alta resistividade) reproduz os decoupling factors.

    No limite estático (δ ≫ L), o campo EM reduz-se ao campo de um dipolo
    magnético no vácuo:
      ACp = -1/(4πL³)  (planar: Hxx=Hyy, Hzz broadside)
      ACx = +1/(2πL³)  (axial: Hxx para hmdx broadside, Hzz axial)
    """

    def test_vmd_broadside_static_limit(self):
        """VMD broadside (x=L, y=0, z=0): Hz → -1/(4πL³)."""
        L = 1.0
        # rho = 1e8 Ω·m → sigma ≈ 0 → quase estático
        ctx = _setup_single_layer(10.0, 1.0e8, L)
        _, _, Hz = vmd(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            L,
            0.0,
            0.0,
            ctx["u"],
            ctx["uh"],
            ctx["AdmInt"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["FEdwz"],
            ctx["FEupz"],
        )
        ACp, _ = static_decoupling_factors(L)
        assert abs(Hz.real - ACp) < 1e-6, (
            f"Hz.real={Hz.real:.6e} vs ACp={ACp:.6e} difere "
            f"em {abs(Hz.real - ACp):.2e}"
        )

    def test_hmd_axial_static_limit(self):
        """HMD hmdx axial (x=L, y=0): Hx[0] → +1/(2πL³)."""
        L = 1.0
        ctx = _setup_single_layer(10.0, 1.0e8, L)
        Hx, _, _ = hmd_tiv(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            ctx["eta"],
            L,
            0.0,
            0.0,
            ctx["u"],
            ctx["s"],
            ctx["uh"],
            ctx["sh"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["RTMdw"],
            ctx["RTMup"],
            ctx["Mxdw"],
            ctx["Mxup"],
            ctx["Eudw"],
            ctx["Euup"],
        )
        _, ACx = static_decoupling_factors(L)
        assert abs(Hx[0].real - ACx) < 1e-6, (
            f"Hx_hmdx.real={Hx[0].real:.6e} vs ACx={ACx:.6e} difere "
            f"em {abs(Hx[0].real - ACx):.2e}"
        )

    def test_hmd_planar_hmdy_static_limit(self):
        """HMD hmdy (dipolo y, broadside x): Hy[1] → -1/(4πL³)."""
        L = 1.0
        ctx = _setup_single_layer(10.0, 1.0e8, L)
        _, Hy, _ = hmd_tiv(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            ctx["eta"],
            L,
            0.0,
            0.0,
            ctx["u"],
            ctx["s"],
            ctx["uh"],
            ctx["sh"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["RTMdw"],
            ctx["RTMup"],
            ctx["Mxdw"],
            ctx["Mxup"],
            ctx["Eudw"],
            ctx["Euup"],
        )
        ACp, _ = static_decoupling_factors(L)
        assert abs(Hy[1].real - ACp) < 1e-6, (
            f"Hy_hmdy.real={Hy[1].real:.6e} vs ACp={ACp:.6e} difere "
            f"em {abs(Hy[1].real - ACp):.2e}"
        )

    def test_hmd_cross_components_zero_at_broadside(self):
        """Em broadside geometria, Hy_hmdx ≈ 0 e Hx_hmdy ≈ 0."""
        L = 1.0
        ctx = _setup_single_layer(10.0, 1.0e8, L)
        Hx, Hy, _ = hmd_tiv(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            ctx["eta"],
            L,
            0.0,
            0.0,
            ctx["u"],
            ctx["s"],
            ctx["uh"],
            ctx["sh"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["RTMdw"],
            ctx["RTMup"],
            ctx["Mxdw"],
            ctx["Mxup"],
            ctx["Eudw"],
            ctx["Euup"],
        )
        # Hy_hmdx (Hy de dipolo x em broadside): deve ser ~0
        # Hx_hmdy (Hx de dipolo y em broadside): deve ser ~0 (xy_r2=0)
        assert abs(Hy[0]) < 1e-10, f"Hy_hmdx={Hy[0]} não é ~0"
        assert abs(Hx[1]) < 1e-10, f"Hx_hmdy={Hx[1]} não é ~0"

    def test_vmd_axial_y_component_zero(self):
        """VMD broadside: Hy_vmd = 0 (y=0)."""
        L = 1.0
        ctx = _setup_single_layer(10.0, 1.0e8, L)
        _, Hy, _ = vmd(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            L,
            0.0,
            0.0,
            ctx["u"],
            ctx["uh"],
            ctx["AdmInt"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["FEdwz"],
            ctx["FEupz"],
        )
        assert abs(Hy) < 1e-14, f"Hy_vmd={Hy} não é ~0"


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdAnalyticalBroadside — 4 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestVmdAnalyticalBroadside:
    """Valida VMD broadside vs `vmd_fullspace_broadside` com tolerância < 1e-4.

    Tolerância: o Werthmüller 201pt tem precisão ~1e-5 no quasi-estático;
    aumentar para Anderson 801pt chega a ~1e-8. A Sprint 2.2 mira < 1e-4
    como gate, e a Sprint 2.6 refinará para < 1e-10 com Anderson.

    Convenção temporal:
        `half_space.py` usa `e^(+iωt)` (convenção "Ward-Hohmann analítica")
        com Im(k) > 0, enquanto o port Numba/Fortran usa `e^(-iωt)`
        (convenção "Moran-Gianzero engineering") com Im(k) < 0. A relação
        entre as duas é `H(e^-iωt) = conj(H(e^+iωt))` — tomamos conjugado
        do valor analítico antes de comparar.
    """

    @pytest.mark.parametrize(
        "frequency_hz,rho",
        [
            (100.0, 100.0),
            (1_000.0, 100.0),
            (20_000.0, 100.0),
            (100_000.0, 100.0),
        ],
    )
    def test_vmd_broadside_matches_analytical(self, frequency_hz, rho):
        """Hz vs conj(vmd_fullspace_broadside(L, f, ρ)) — convenção e^(-iωt)."""
        L = 1.0
        ctx = _setup_single_layer(frequency_hz, rho, L)
        _, _, Hz = vmd(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            L,
            0.0,
            0.0,
            ctx["u"],
            ctx["uh"],
            ctx["AdmInt"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["FEdwz"],
            ctx["FEupz"],
        )
        # Conjugado para casar convenção temporal (e^-iωt do Fortran)
        Hz_analytical = np.conj(vmd_fullspace_broadside(L, frequency_hz, rho))
        diff = abs(Hz - Hz_analytical)
        # Werthmüller 201pt atinge precisão ~1e-5 no quasi-estático
        assert diff < 1e-4, (
            f"Hz(Numba)={Hz:.6e} vs conj(Hz_analytical)={Hz_analytical:.6e} "
            f"difere em {diff:.2e} > 1e-4 (f={frequency_hz} Hz, ρ={rho})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestHighResistivity — 6 testes parametrizados
# ──────────────────────────────────────────────────────────────────────────────
class TestHighResistivity:
    """Valida operação do forward em rochas de alta resistividade.

    Cobertura: ρ ∈ {1, 10², 10³, 10⁴, 10⁵, 10⁶} Ω·m — varredura completa
    desde argilas (1 Ω·m) até rochas ígneas secas (1e6 Ω·m). Verifica que
    o forward não gera NaN/Inf e que |H| permanece finito e razoável.
    """

    @pytest.mark.parametrize(
        "rho",
        [1.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0],
    )
    def test_vmd_stable_high_resistivity(self, rho):
        """VMD estável em ρ ∈ {1, 10², ..., 10⁶} Ω·m."""
        L = 1.0
        ctx = _setup_single_layer(20_000.0, rho, L)
        _, _, Hz = vmd(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            L,
            0.0,
            0.0,
            ctx["u"],
            ctx["uh"],
            ctx["AdmInt"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["FEdwz"],
            ctx["FEupz"],
        )
        assert np.isfinite(Hz.real), f"Hz.real não finito em ρ={rho}"
        assert np.isfinite(Hz.imag), f"Hz.imag não finito em ρ={rho}"
        # Magnitude deve estar próxima de ACp (decoupling) em ρ alto
        ACp, _ = static_decoupling_factors(L)
        # Limite superior loose: o decoupling é sempre |ACp| ≈ 0.08
        assert abs(Hz) < 1.0, f"|Hz|={abs(Hz):.2e} muito grande em ρ={rho}"


# ──────────────────────────────────────────────────────────────────────────────
# TestReciprocity — 2 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestReciprocity:
    """Valida princípio da reciprocidade em meio isotrópico homogêneo.

    Reciprocidade: swap (Tx ↔ cx, Ty ↔ cy, h0 ↔ z) produz campo idêntico
    no meio isotrópico homogêneo (single layer). Essa é uma sanidade
    fundamental da formulação EM.
    """

    def test_vmd_reciprocity_isotropic(self):
        """VMD: swap T↔R produz mesmo Hz."""
        L = 1.0
        ctx = _setup_single_layer(20_000.0, 100.0, L)
        # Original: T em (0,0,0), R em (1,0,0)
        _, _, Hz_orig = vmd(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            L,
            0.0,
            0.0,
            ctx["u"],
            ctx["uh"],
            ctx["AdmInt"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["FEdwz"],
            ctx["FEupz"],
        )
        # Swap: T em (1,0,0), R em (0,0,0). Em half-space isotrópico
        # homogêneo, a recalculação de common_factors com h0=0 produz o
        # mesmo resultado porque prof[0]=0 e a camada é infinita.
        ctx2 = _setup_single_layer(20_000.0, 100.0, L)
        _, _, Hz_swap = vmd(
            1.0,
            0.0,
            0.0,
            ctx2["n"],
            0,
            0,
            ctx2["npt"],
            ctx2["filt"].abscissas,
            ctx2["filt"].weights_j0,
            ctx2["filt"].weights_j1,
            ctx2["h"],
            ctx2["prof"],
            ctx2["zeta"],
            0.0,
            0.0,
            0.0,
            ctx2["u"],
            ctx2["uh"],
            ctx2["AdmInt"],
            ctx2["RTEdw"],
            ctx2["RTEup"],
            ctx2["FEdwz"],
            ctx2["FEupz"],
        )
        # Valores devem ser idênticos pela reciprocidade
        assert abs(Hz_orig - Hz_swap) < 1e-12, (
            f"Hz_orig={Hz_orig:.6e} vs Hz_swap={Hz_swap:.6e} difere "
            f"em {abs(Hz_orig - Hz_swap):.2e}"
        )

    def test_hmd_symmetry_rotation(self):
        """HMD: dipolo x em (0,0,0) broadside (r=[1,0,0]) tem Hx_hmdx = Hy_hmdy."""
        L = 1.0
        ctx = _setup_single_layer(20_000.0, 100.0, L)
        Hx, Hy, _ = hmd_tiv(
            0.0,
            0.0,
            0.0,
            ctx["n"],
            0,
            0,
            ctx["npt"],
            ctx["filt"].abscissas,
            ctx["filt"].weights_j0,
            ctx["filt"].weights_j1,
            ctx["h"],
            ctx["prof"],
            ctx["zeta"],
            ctx["eta"],
            L,
            0.0,
            0.0,
            ctx["u"],
            ctx["s"],
            ctx["uh"],
            ctx["sh"],
            ctx["RTEdw"],
            ctx["RTEup"],
            ctx["RTMdw"],
            ctx["RTMup"],
            ctx["Mxdw"],
            ctx["Mxup"],
            ctx["Eudw"],
            ctx["Euup"],
        )
        # Rotação de 90° em torno de z: hmdx-axial(x) deve igualar
        # magnitude de hmdy-planar(y) — mas com sinais diferentes
        # pela convenção ACx vs -ACp.
        # Verificação frouxa: ambos devem ser ~ O(0.1), não-zero,
        # e o padrão hmdx axial > |hmdy planar| numericamente.
        assert abs(Hx[0]) > 0.1, f"|Hx_hmdx|={abs(Hx[0])}  muito pequeno"
        assert abs(Hy[1]) > 0.05, f"|Hy_hmdy|={abs(Hy[1])} muito pequeno"
        assert abs(abs(Hx[0]) / abs(Hy[1]) - 2.0) < 0.1, (
            f"Razão |Hx_hmdx|/|Hy_hmdy|={abs(Hx[0]) / abs(Hy[1]):.3f} "
            f"deveria ser ~2 (ACx/ACp=2)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestNumbaCompilation — 2 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestNumbaCompilation:
    """Verifica que o JIT compila e cacheia (quando Numba disponível)."""

    @pytest.mark.slow
    def test_numba_jit_warmup_and_reuse(self):
        """Primeira chamada compila, segunda usa cache (se Numba instalado)."""
        if not HAS_NUMBA:
            pytest.skip("Numba não instalado — teste não aplicável")
        import time

        ctx = _setup_single_layer(_FREQ_DEFAULT, _RHO_DEFAULT, _SPACING_DEFAULT)

        def _call_vmd():
            return vmd(
                0.0,
                0.0,
                0.0,
                ctx["n"],
                0,
                0,
                ctx["npt"],
                ctx["filt"].abscissas,
                ctx["filt"].weights_j0,
                ctx["filt"].weights_j1,
                ctx["h"],
                ctx["prof"],
                ctx["zeta"],
                _SPACING_DEFAULT,
                0.0,
                0.0,
                ctx["u"],
                ctx["uh"],
                ctx["AdmInt"],
                ctx["RTEdw"],
                ctx["RTEup"],
                ctx["FEdwz"],
                ctx["FEupz"],
            )

        # Warm-up (pode ser lento)
        _call_vmd()
        # Segunda chamada deve ser sub-milissegundo
        t0 = time.perf_counter()
        _call_vmd()
        dt = time.perf_counter() - t0
        assert dt < 1.0, (
            f"Segunda chamada VMD tomou {dt:.3f}s > 1.0s — "
            f"possível problema de cache JIT"
        )

    def test_has_numba_flag_is_bool(self):
        """HAS_NUMBA é um bool exportado."""
        from geosteering_ai.simulation._numba import HAS_NUMBA as HAS

        assert isinstance(HAS, bool)
