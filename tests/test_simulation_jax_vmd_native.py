# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_vmd_native.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — JAX Native VMD Dipoles (Sprint 3.3.3)            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest + JAX + NumPy                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Testes do port JAX nativo dos 6 casos geométricos do `vmd()`         ║
# ║    (Vertical Magnetic Dipole) via `lax.switch`. Valida:                  ║
# ║      • Paridade bit-exata (rtol=1e-12) vs referência NumPy pura          ║
# ║        (replica `_numba/dipoles.py:856-945`)                              ║
# ║      • Dispatcher reusa `compute_case_index_jax` (já testado na         ║
# ║        Sprint 3.3.2)                                                      ║
# ║      • Estabilidade em alta resistividade (10³..10⁶ Ω·m)                ║
# ║      • Diferenciabilidade via `jax.grad` w.r.t. `z`                     ║
# ║      • Compile-time bounded (< 90s primeira chamada)                     ║
# ║      • IMPLEMENTATION_STATUS reflete completude                          ║
# ║                                                                           ║
# ║  ESCOPO SPRINT 3.3.3                                                      ║
# ║    Esta sprint implementa a ETAPA 5 do `vmd()` em JAX nativo — os       ║
# ║    kernels KtezJ0/KtedzzJ1 para os 6 casos geométricos. As ETAPAS 3    ║
# ║    (propagação dos potenciais TEdwz/TEupz entre camadas via loops      ║
# ║    Numba 787-854) e 6 (assembly final Hx/Hy/Hz via Σ KtedzzJ1·kr²) ║
# ║    seguem no caminho híbrido. Wiring end-to-end é Sprint 3.3.4.         ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • _jax/dipoles_native.py — implementação (Sprint 3.3.3)              ║
# ║    • _numba/dipoles.py:856-945 — referência ETAPA 5 VMD                 ║
# ║    • magneticdipoles.f08:527-623 — referência Fortran                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da ETAPA 5 nativa do VMD em JAX (6 casos via `lax.switch`)."""
from __future__ import annotations

import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from geosteering_ai.simulation._jax.dipoles_native import (  # noqa: E402
    IMPLEMENTATION_STATUS,
    _vmd_full_jax,
    _vmd_kernel_case1_jax,
    _vmd_kernel_case2_jax,
    _vmd_kernel_case3_jax,
    _vmd_kernel_case4_jax,
    _vmd_kernel_case5_jax,
    _vmd_kernel_case6_jax,
    compute_case_index_jax,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — inputs sintéticos realistas
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def vmd_synth_inputs():
    """Inputs sintéticos para os 6 kernels VMD (npt=32)."""
    npt = 32
    rng = np.random.default_rng(2026)

    def _c(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex128
        ) * 0.1

    return {
        "npt": npt,
        "TEdwz_r": jnp.asarray(_c((npt,)) + 0.5),
        "TEupz_r": jnp.asarray(_c((npt,)) + 0.5),
        "u_r": jnp.asarray(_c((npt,)) + 0.1),
        "RTEdw_r": jnp.asarray(_c((npt,))),
        "RTEup_r": jnp.asarray(_c((npt,))),
        "AdmInt_r": jnp.asarray(_c((npt,)) + 0.5),
        "FEdwz": jnp.asarray(_c((npt,)) + 0.3),
        "FEupz": jnp.asarray(_c((npt,)) + 0.3),
        "wJ0": jnp.asarray(rng.normal(size=npt)),
        "wJ1": jnp.asarray(rng.normal(size=npt)),
        "z": 5.0,
        "h0": 4.0,
        "prof_r": 3.0,
        "prof_r1": 8.0,
        "h_r": 5.0,
    }


def _call_vmd_case(case_fn, inputs):
    """Invoca um kernel `_vmd_kernel_caseN_jax` com a assinatura uniforme."""
    return case_fn(
        inputs["TEdwz_r"],
        inputs["TEupz_r"],
        inputs["u_r"],
        inputs["RTEdw_r"],
        inputs["RTEup_r"],
        inputs["AdmInt_r"],
        inputs["FEdwz"],
        inputs["FEupz"],
        inputs["wJ0"],
        inputs["wJ1"],
        inputs["z"],
        inputs["h0"],
        inputs["prof_r"],
        inputs["prof_r1"],
        inputs["h_r"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Referência NumPy pura — replica `_numba/dipoles.py:856-945` (ETAPA 5 VMD)
# ──────────────────────────────────────────────────────────────────────────────


def _numpy_vmd_case(case_id: int, inputs) -> tuple[np.ndarray, np.ndarray]:
    """Computa (KtezJ0, KtedzzJ1) na referência NumPy pura para um dado caso."""
    TEdwz = np.asarray(inputs["TEdwz_r"])
    TEupz = np.asarray(inputs["TEupz_r"])
    u = np.asarray(inputs["u_r"])
    RTEdw = np.asarray(inputs["RTEdw_r"])
    RTEup = np.asarray(inputs["RTEup_r"])
    AdmInt = np.asarray(inputs["AdmInt_r"])
    FEdwz = np.asarray(inputs["FEdwz"])
    FEupz = np.asarray(inputs["FEupz"])
    wJ0 = np.asarray(inputs["wJ0"])
    wJ1 = np.asarray(inputs["wJ1"])
    z = inputs["z"]
    h0 = inputs["h0"]
    pr = inputs["prof_r"]
    pr1 = inputs["prof_r1"]
    hr = inputs["h_r"]

    if case_id == 1:
        fac = TEupz * np.exp(u * z)
        KtezJ0 = fac * wJ0
        KtedzzJ1 = AdmInt * fac * wJ1
    elif case_id == 2:
        a = np.exp(u * (z - pr1))
        b = RTEup * np.exp(-u * (z - pr + hr))
        fac = TEupz * (a + b)
        KtezJ0 = fac * wJ0
        KtedzzJ1 = AdmInt * TEupz * (a - b) * wJ1
    elif case_id == 3:
        a = np.exp(u * (z - h0))
        b = RTEup * FEupz * np.exp(-u * (z - pr))
        c = RTEdw * FEdwz * np.exp(u * (z - pr1))
        fac = TEupz * (a + b + c)
        KtezJ0 = fac * wJ0
        KtedzzJ1 = AdmInt * TEupz * (a - b + c) * wJ1
    elif case_id == 4:
        a = np.exp(-u * (z - h0))
        b = RTEup * FEupz * np.exp(-u * (z - pr))
        c = RTEdw * FEdwz * np.exp(u * (z - pr1))
        fac = TEdwz * (a + b + c)
        KtezJ0 = fac * wJ0
        KtedzzJ1 = -AdmInt * TEdwz * (a + b - c) * wJ1
    elif case_id == 5:
        a = np.exp(-u * (z - pr))
        b = RTEdw * np.exp(u * (z - pr1 - hr))
        fac = TEdwz * (a + b)
        KtezJ0 = fac * wJ0
        KtedzzJ1 = -AdmInt * TEdwz * (a - b) * wJ1
    elif case_id == 6:
        fac = TEdwz * np.exp(-u * (z - pr))
        KtezJ0 = fac * wJ0
        KtedzzJ1 = -AdmInt * fac * wJ1
    else:
        raise ValueError(f"case_id inválido: {case_id}")
    return KtezJ0, KtedzzJ1


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdCaseParity — paridade bit-exata por caso (rtol < 1e-12)
# ──────────────────────────────────────────────────────────────────────────────


class TestVmdCaseParity:
    """Cada kernel VMD JAX deve ser bit-exato vs referência NumPy."""

    @pytest.mark.parametrize(
        "case_id,case_fn",
        [
            (1, _vmd_kernel_case1_jax),
            (2, _vmd_kernel_case2_jax),
            (3, _vmd_kernel_case3_jax),
            (4, _vmd_kernel_case4_jax),
            (5, _vmd_kernel_case5_jax),
            (6, _vmd_kernel_case6_jax),
        ],
    )
    def test_parity_case(self, vmd_synth_inputs, case_id, case_fn) -> None:
        KtezJ0_j, KtedzzJ1_j = _call_vmd_case(case_fn, vmd_synth_inputs)
        KtezJ0_n, KtedzzJ1_n = _numpy_vmd_case(case_id, vmd_synth_inputs)
        np.testing.assert_allclose(np.asarray(KtezJ0_j), KtezJ0_n, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            np.asarray(KtedzzJ1_j), KtedzzJ1_n, rtol=1e-12, atol=1e-14
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdDispatcher — `_vmd_full_jax(idx, ...)` deve igualar chamada direta
# ──────────────────────────────────────────────────────────────────────────────


class TestVmdDispatcher:
    """O dispatcher `lax.switch` deve produzir resultado idêntico ao kernel direto."""

    @pytest.mark.parametrize("case_idx", list(range(6)))
    def test_dispatch_matches_direct_call(self, vmd_synth_inputs, case_idx) -> None:
        direct_fn = [
            _vmd_kernel_case1_jax,
            _vmd_kernel_case2_jax,
            _vmd_kernel_case3_jax,
            _vmd_kernel_case4_jax,
            _vmd_kernel_case5_jax,
            _vmd_kernel_case6_jax,
        ][case_idx]
        K0_direct, K1_direct = _call_vmd_case(direct_fn, vmd_synth_inputs)
        K0_disp, K1_disp = _vmd_full_jax(
            case_idx,
            vmd_synth_inputs["TEdwz_r"],
            vmd_synth_inputs["TEupz_r"],
            vmd_synth_inputs["u_r"],
            vmd_synth_inputs["RTEdw_r"],
            vmd_synth_inputs["RTEup_r"],
            vmd_synth_inputs["AdmInt_r"],
            vmd_synth_inputs["FEdwz"],
            vmd_synth_inputs["FEupz"],
            vmd_synth_inputs["wJ0"],
            vmd_synth_inputs["wJ1"],
            vmd_synth_inputs["z"],
            vmd_synth_inputs["h0"],
            vmd_synth_inputs["prof_r"],
            vmd_synth_inputs["prof_r1"],
            vmd_synth_inputs["h_r"],
        )
        np.testing.assert_allclose(
            np.asarray(K0_disp), np.asarray(K0_direct), rtol=1e-13, atol=1e-15
        )
        np.testing.assert_allclose(
            np.asarray(K1_disp), np.asarray(K1_direct), rtol=1e-13, atol=1e-15
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdCaseIndexReuse — compute_case_index_jax já testado em HMD
# ──────────────────────────────────────────────────────────────────────────────


class TestVmdCaseIndexReuse:
    """Garante que VMD reusa o mesmo `compute_case_index_jax` da Sprint 3.3.2."""

    def test_same_function_same_indices(self) -> None:
        # 6 posições canônicas — mesmo mapeamento usado no HMD
        cases = [
            (0, 1, 3, 2.0, 5.0, 0),  # camadR=0, camadT=1
            (1, 2, 3, 4.0, 7.0, 1),  # camadR<camadT (e camadR != 0)
            (1, 1, 3, 4.0, 5.0, 2),  # mesma camada, z<=h0
            (1, 1, 3, 6.0, 5.0, 3),  # mesma camada, z>h0
            (1, 0, 3, 5.0, 2.0, 4),  # camadR>camadT, não última
            (2, 0, 3, 9.0, 2.0, 5),  # camadR=n-1
        ]
        for camad_r, camad_t, n, z, h0, expected in cases:
            assert compute_case_index_jax(camad_r, camad_t, n, z, h0) == expected


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdHighResistivityStability — sem NaN/Inf em ρ ∈ {10³..10⁶} Ω·m
# ──────────────────────────────────────────────────────────────────────────────


class TestVmdHighResistivityStability:
    """Em meios de alta resistividade, `u` é grande e `exp(-u·dz)` ≈ 0; nenhum NaN."""

    @pytest.mark.parametrize("rho_scale", [1e3, 1e4, 1e5, 1e6])
    def test_no_nan_inf_high_rho(self, rho_scale, vmd_synth_inputs) -> None:
        # Multiplica u por sqrt(1/ρ_scale) — proxy para alta resistividade
        scaled = dict(vmd_synth_inputs)
        scaled["u_r"] = vmd_synth_inputs["u_r"] / np.sqrt(rho_scale)
        for case_fn in [
            _vmd_kernel_case1_jax,
            _vmd_kernel_case2_jax,
            _vmd_kernel_case3_jax,
            _vmd_kernel_case4_jax,
            _vmd_kernel_case5_jax,
            _vmd_kernel_case6_jax,
        ]:
            K0, K1 = _call_vmd_case(case_fn, scaled)
            assert np.all(np.isfinite(np.asarray(K0))), f"NaN/Inf em K0, ρ={rho_scale}"
            assert np.all(np.isfinite(np.asarray(K1))), f"NaN/Inf em K1, ρ={rho_scale}"


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdDifferentiability — jax.grad(_vmd_full_jax, argnums=z) finito
# ──────────────────────────────────────────────────────────────────────────────


class TestVmdDifferentiability:
    """`jax.grad` sobre `z` deve retornar gradiente finito em todos os 6 casos."""

    @pytest.mark.parametrize("case_idx", list(range(6)))
    def test_grad_wrt_z(self, vmd_synth_inputs, case_idx) -> None:
        def _scalar_loss(z_val):
            K0, K1 = _vmd_full_jax(
                case_idx,
                vmd_synth_inputs["TEdwz_r"],
                vmd_synth_inputs["TEupz_r"],
                vmd_synth_inputs["u_r"],
                vmd_synth_inputs["RTEdw_r"],
                vmd_synth_inputs["RTEup_r"],
                vmd_synth_inputs["AdmInt_r"],
                vmd_synth_inputs["FEdwz"],
                vmd_synth_inputs["FEupz"],
                vmd_synth_inputs["wJ0"],
                vmd_synth_inputs["wJ1"],
                z_val,
                vmd_synth_inputs["h0"],
                vmd_synth_inputs["prof_r"],
                vmd_synth_inputs["prof_r1"],
                vmd_synth_inputs["h_r"],
            )
            return jnp.real(jnp.sum(K0) + jnp.sum(K1))

        grad_fn = jax.grad(_scalar_loss)
        g = grad_fn(vmd_synth_inputs["z"])
        assert np.isfinite(float(g)), f"grad não finito no caso {case_idx}: {g}"


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdCompileTimeBudget — primeira chamada deve compilar em < 90s
# ──────────────────────────────────────────────────────────────────────────────


class TestVmdCompileTimeBudget:
    """Regression guard — compile-time não pode disparar."""

    def test_first_call_compiles_under_90s(self, vmd_synth_inputs) -> None:
        # Limpa caches para forçar recompile (best effort)
        jax.clear_caches()
        t0 = time.time()
        for case in range(6):
            _vmd_full_jax(
                case,
                vmd_synth_inputs["TEdwz_r"],
                vmd_synth_inputs["TEupz_r"],
                vmd_synth_inputs["u_r"],
                vmd_synth_inputs["RTEdw_r"],
                vmd_synth_inputs["RTEup_r"],
                vmd_synth_inputs["AdmInt_r"],
                vmd_synth_inputs["FEdwz"],
                vmd_synth_inputs["FEupz"],
                vmd_synth_inputs["wJ0"],
                vmd_synth_inputs["wJ1"],
                vmd_synth_inputs["z"],
                vmd_synth_inputs["h0"],
                vmd_synth_inputs["prof_r"],
                vmd_synth_inputs["prof_r1"],
                vmd_synth_inputs["h_r"],
            )
        dt = time.time() - t0
        assert dt < 90.0, f"VMD compile-time disparou: {dt:.1f}s > 90s"


# ──────────────────────────────────────────────────────────────────────────────
# TestVmdImplementationStatus — sanity check do dict de status
# ──────────────────────────────────────────────────────────────────────────────


class TestVmdImplementationStatus:
    """Garante que IMPLEMENTATION_STATUS reflete a Sprint 3.3.3."""

    def test_status_marks_vmd_complete(self) -> None:
        assert "_vmd_full_jax" in IMPLEMENTATION_STATUS
        assert "✅" in IMPLEMENTATION_STATUS["_vmd_full_jax"]
        assert "Sprint 3.3.3" in IMPLEMENTATION_STATUS["_vmd_full_jax"]

    def test_status_marks_vmd_kernels_complete(self) -> None:
        assert "_vmd_kernel_case1_jax..case6_jax" in IMPLEMENTATION_STATUS
        assert "✅" in IMPLEMENTATION_STATUS["_vmd_kernel_case1_jax..case6_jax"]
