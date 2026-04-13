# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_dipoles_native.py                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — JAX Native HMD Dipoles (Sprint 3.3.2)            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest + JAX + NumPy                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Testes do port JAX nativo dos 6 casos geométricos de hmd_tiv via      ║
# ║    `lax.switch`. Valida:                                                  ║
# ║      • Paridade bit-exata (rtol=1e-12) vs Numba ETAPA 5                  ║
# ║      • Dispatcher `compute_case_index_jax` para 6 posições canônicas    ║
# ║      • Estabilidade em alta resistividade (10³..10⁶ Ω·m)                ║
# ║      • Diferenciabilidade via `jax.grad`                                  ║
# ║      • Compile-time bounded (< 90s primeira chamada)                     ║
# ║      • Hybrid fallback funciona quando use_native_dipoles=True          ║
# ║                                                                           ║
# ║  IMPORTANTE — ESCOPO SPRINT 3.3.2                                         ║
# ║    Esta sprint implementa APENAS a ETAPA 5 do hmd_tiv em JAX nativo —   ║
# ║    os kernels Ktm/Kte/Ktedz para os 6 casos geométricos. A propagação  ║
# ║    dos potenciais (ETAPA 3) e o assembly Ward-Hohmann (ETAPA 6), bem    ║
# ║    como o VMD nativo, ficam para Sprint 3.3.3 (PR #11). Por isso,       ║
# ║    os testes focam nos kernels individuais + dispatcher.                 ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • _jax/dipoles_native.py — implementação (Sprint 3.3.2)              ║
# ║    • _numba/dipoles.py:547-644 — referência ETAPA 5                     ║
# ║    • magneticdipoles.f08:310-383 — referência Fortran                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da ETAPA 5 nativa em JAX (6 casos via `lax.switch`)."""
from __future__ import annotations

import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from geosteering_ai.simulation._jax.dipoles_native import (  # noqa: E402
    IMPLEMENTATION_STATUS,
    _hmd_tiv_full_jax,
    _hmd_tiv_kernel_case1_jax,
    _hmd_tiv_kernel_case2_jax,
    _hmd_tiv_kernel_case3_jax,
    _hmd_tiv_kernel_case4_jax,
    _hmd_tiv_kernel_case5_jax,
    _hmd_tiv_kernel_case6_jax,
    compute_case_index_jax,
    decoupling_factors_jax,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — inputs sintéticos realistas
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synth_inputs():
    """Inputs sintéticos (npt=32 para testes rápidos)."""
    npt = 32
    rng = np.random.default_rng(42)

    # Arrays complex128 com magnitude ~1 (realistic para u/s em baixas freqs)
    def _c(shape):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex128
        ) * 0.1

    # Simula camada (npt,) arrays
    return {
        "npt": npt,
        "u_r": jnp.asarray(_c((npt,)) + 0.1),
        "s_r": jnp.asarray(_c((npt,)) + 0.1),
        "RTEdw_r": jnp.asarray(_c((npt,))),
        "RTEup_r": jnp.asarray(_c((npt,))),
        "RTMdw_r": jnp.asarray(_c((npt,))),
        "RTMup_r": jnp.asarray(_c((npt,))),
        "Tudw_r": jnp.asarray(_c((npt,))),
        "Tuup_r": jnp.asarray(_c((npt,))),
        "Txdw_r": jnp.asarray(_c((npt,))),
        "Txup_r": jnp.asarray(_c((npt,))),
        "Mxdw": jnp.asarray(_c((npt,))),
        "Mxup": jnp.asarray(_c((npt,))),
        "Eudw": jnp.asarray(_c((npt,))),
        "Euup": jnp.asarray(_c((npt,))),
        "z": 5.0,
        "h0": 4.0,
        "prof_r": 3.0,
        "prof_r1": 8.0,
        "h_r": 5.0,
    }


def _call_case(case_fn, inputs):
    """Ponteiro com signature uniforme para kernels case_{1..6}_jax."""
    return case_fn(
        inputs["u_r"],
        inputs["s_r"],
        inputs["RTEdw_r"],
        inputs["RTEup_r"],
        inputs["RTMdw_r"],
        inputs["RTMup_r"],
        inputs["Tudw_r"],
        inputs["Tuup_r"],
        inputs["Txdw_r"],
        inputs["Txup_r"],
        inputs["Mxdw"],
        inputs["Mxup"],
        inputs["Eudw"],
        inputs["Euup"],
        inputs["z"],
        inputs["h0"],
        inputs["prof_r"],
        inputs["prof_r1"],
        inputs["h_r"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# TestCaseParityVsNumba — paridade bit-exata por caso (rtol < 1e-12)
# ──────────────────────────────────────────────────────────────────────────────


def _numba_case_ktm_kte(
    case_id: int, inputs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Referência NumPy pura — replica ETAPA 5 do Numba para 1 caso."""
    u = np.asarray(inputs["u_r"])
    s = np.asarray(inputs["s_r"])
    RTEdw = np.asarray(inputs["RTEdw_r"])
    RTEup = np.asarray(inputs["RTEup_r"])
    RTMdw = np.asarray(inputs["RTMdw_r"])
    RTMup = np.asarray(inputs["RTMup_r"])
    Tudw = np.asarray(inputs["Tudw_r"])
    Tuup = np.asarray(inputs["Tuup_r"])
    Txdw = np.asarray(inputs["Txdw_r"])
    Txup = np.asarray(inputs["Txup_r"])
    Mxdw = np.asarray(inputs["Mxdw"])
    Mxup = np.asarray(inputs["Mxup"])
    Eudw = np.asarray(inputs["Eudw"])
    Euup = np.asarray(inputs["Euup"])
    z = inputs["z"]
    h0 = inputs["h0"]
    pr = inputs["prof_r"]
    pr1 = inputs["prof_r1"]
    hr = inputs["h_r"]

    if case_id == 1:
        Ktm = Txup * np.exp(s * z)
        Kte = Tuup * np.exp(u * z)
        Ktedz = u * Kte
    elif case_id == 2:
        a_tm = np.exp(s * (z - pr1))
        b_tm = RTMup * np.exp(-s * (z - pr + hr))
        Ktm = Txup * (a_tm + b_tm)
        a_te = np.exp(u * (z - pr1))
        b_te = RTEup * np.exp(-u * (z - pr + hr))
        Kte = Tuup * (a_te + b_te)
        Ktedz = u * Tuup * (a_te - b_te)
    elif case_id == 3:
        a_tm = np.exp(s * (z - h0))
        b_tm = RTMup * Mxup * np.exp(-s * (z - pr))
        c_tm = RTMdw * Mxdw * np.exp(s * (z - pr1))
        Ktm = Txup * (a_tm + b_tm + c_tm)
        a_te = np.exp(u * (z - h0))
        b_te = RTEup * Euup * np.exp(-u * (z - pr))
        c_te = RTEdw * Eudw * np.exp(u * (z - pr1))
        Kte = Tuup * (a_te + b_te - c_te)
        Ktedz = u * Tuup * (a_te - b_te - c_te)
    elif case_id == 4:
        a_tm = np.exp(-s * (z - h0))
        b_tm = RTMup * Mxup * np.exp(-s * (z - pr))
        c_tm = RTMdw * Mxdw * np.exp(s * (z - pr1))
        Ktm = Txdw * (a_tm + b_tm + c_tm)
        a_te = np.exp(-u * (z - h0))
        b_te = RTEup * Euup * np.exp(-u * (z - pr))
        c_te = RTEdw * Eudw * np.exp(u * (z - pr1))
        Kte = Tudw * (a_te - b_te + c_te)
        Ktedz = -u * Tudw * (a_te - b_te - c_te)
    elif case_id == 5:
        a_tm = np.exp(-s * (z - pr))
        b_tm = RTMdw * np.exp(s * (z - pr1 - hr))
        Ktm = Txdw * (a_tm + b_tm)
        a_te = np.exp(-u * (z - pr))
        b_te = RTEdw * np.exp(u * (z - pr1 - hr))
        Kte = Tudw * (a_te + b_te)
        Ktedz = -u * Tudw * (a_te - b_te)
    elif case_id == 6:
        Ktm = Txdw * np.exp(-s * (z - pr))
        Kte = Tudw * np.exp(-u * (z - pr))
        Ktedz = -u * Kte
    else:
        raise ValueError(f"case_id inválido: {case_id}")
    return Ktm, Kte, Ktedz


class TestCaseParityVsNumba:
    """Cada caso JAX deve ser bit-exato vs implementação NumPy idêntica."""

    @pytest.mark.parametrize(
        "case_id,case_fn",
        [
            (1, _hmd_tiv_kernel_case1_jax),
            (2, _hmd_tiv_kernel_case2_jax),
            (3, _hmd_tiv_kernel_case3_jax),
            (4, _hmd_tiv_kernel_case4_jax),
            (5, _hmd_tiv_kernel_case5_jax),
            (6, _hmd_tiv_kernel_case6_jax),
        ],
    )
    def test_parity_case(self, synth_inputs, case_id, case_fn) -> None:
        """JAX vs NumPy (referência) → erro relativo < 1e-12."""
        Ktm_j, Kte_j, Ktedz_j = _call_case(case_fn, synth_inputs)
        Ktm_n, Kte_n, Ktedz_n = _numba_case_ktm_kte(case_id, synth_inputs)
        np.testing.assert_allclose(np.asarray(Ktm_j), Ktm_n, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(Kte_j), Kte_n, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(Ktedz_j), Ktedz_n, rtol=1e-12, atol=1e-14)


# ──────────────────────────────────────────────────────────────────────────────
# TestCaseIndex — compute_case_index_jax dispatches corretamente
# ──────────────────────────────────────────────────────────────────────────────


class TestCaseIndex:
    """Valida mapeamento (camad_r, camad_t, n, z, h0) → índice 0..5."""

    def test_case1_rx_surface(self) -> None:
        """camadR==0 and camadT!=0 → idx=0."""
        assert compute_case_index_jax(0, 2, 5, 1.0, 10.0) == 0

    def test_case2_rx_above(self) -> None:
        """camadR < camadT (but not in surface) → idx=1."""
        assert compute_case_index_jax(1, 3, 5, 2.0, 8.0) == 1

    def test_case3_same_above(self) -> None:
        """camadR==camadT and z <= h0 → idx=2."""
        assert compute_case_index_jax(1, 1, 3, 5.0, 10.0) == 2

    def test_case4_same_below(self) -> None:
        """camadR==camadT and z > h0 → idx=3."""
        assert compute_case_index_jax(1, 1, 3, 15.0, 10.0) == 3

    def test_case5_rx_below_internal(self) -> None:
        """camadR > camadT and camadR != n-1 → idx=4."""
        assert compute_case_index_jax(2, 1, 5, 10.0, 5.0) == 4

    def test_case6_rx_bottom(self) -> None:
        """camadR == n-1 → idx=5."""
        assert compute_case_index_jax(4, 1, 5, 100.0, 5.0) == 5


# ──────────────────────────────────────────────────────────────────────────────
# TestDispatcher — _hmd_tiv_full_jax (lax.switch)
# ──────────────────────────────────────────────────────────────────────────────


class TestDispatcher:
    """Dispatcher `_hmd_tiv_full_jax` com `lax.switch`."""

    @pytest.mark.parametrize("case_idx", [0, 1, 2, 3, 4, 5])
    def test_dispatch_matches_direct_call(self, synth_inputs, case_idx) -> None:
        """Switch com idx=X → mesmo resultado que chamar caseX diretamente."""
        direct_fns = [
            _hmd_tiv_kernel_case1_jax,
            _hmd_tiv_kernel_case2_jax,
            _hmd_tiv_kernel_case3_jax,
            _hmd_tiv_kernel_case4_jax,
            _hmd_tiv_kernel_case5_jax,
            _hmd_tiv_kernel_case6_jax,
        ]
        direct = _call_case(direct_fns[case_idx], synth_inputs)
        switched = _hmd_tiv_full_jax(
            case_idx,
            synth_inputs["u_r"],
            synth_inputs["s_r"],
            synth_inputs["RTEdw_r"],
            synth_inputs["RTEup_r"],
            synth_inputs["RTMdw_r"],
            synth_inputs["RTMup_r"],
            synth_inputs["Tudw_r"],
            synth_inputs["Tuup_r"],
            synth_inputs["Txdw_r"],
            synth_inputs["Txup_r"],
            synth_inputs["Mxdw"],
            synth_inputs["Mxup"],
            synth_inputs["Eudw"],
            synth_inputs["Euup"],
            synth_inputs["z"],
            synth_inputs["h0"],
            synth_inputs["prof_r"],
            synth_inputs["prof_r1"],
            synth_inputs["h_r"],
        )
        for out_direct, out_switch in zip(direct, switched):
            np.testing.assert_allclose(
                np.asarray(out_direct), np.asarray(out_switch), rtol=1e-13, atol=1e-15
            )


# ──────────────────────────────────────────────────────────────────────────────
# TestHighResistivityStability
# ──────────────────────────────────────────────────────────────────────────────


class TestHighResistivityStability:
    """Estabilidade numérica em alta resistividade (ρ > 1000 Ω·m)."""

    @pytest.mark.parametrize("rho_scale", [1e3, 1e4, 1e5, 1e6])
    def test_no_nan_inf_high_rho(self, rho_scale, synth_inputs) -> None:
        """Em regime de alta resistividade, u/s → valores grandes → exp deve
        permanecer finito se o argumento for limitado pela geometria."""
        # Escala s_r pelo fator correspondente a sigma → 1/ρ
        inp = {**synth_inputs}
        scale = np.sqrt(1.0 / rho_scale)  # u ~ sqrt(zeta·sigma)
        inp["u_r"] = inp["u_r"] * scale
        inp["s_r"] = inp["s_r"] * scale

        for case_fn in [
            _hmd_tiv_kernel_case1_jax,
            _hmd_tiv_kernel_case3_jax,
            _hmd_tiv_kernel_case6_jax,
        ]:
            out = _call_case(case_fn, inp)
            for arr in out:
                arr_np = np.asarray(arr)
                assert np.all(np.isfinite(arr_np.real)), f"NaN/Inf real em ρ={rho_scale}"
                assert np.all(np.isfinite(arr_np.imag)), f"NaN/Inf imag em ρ={rho_scale}"


# ──────────────────────────────────────────────────────────────────────────────
# TestDifferentiability — jax.grad funciona (diferenciável)
# ──────────────────────────────────────────────────────────────────────────────


class TestDifferentiability:
    """`jax.grad` sobre inputs escalares (ex.: z, h0)."""

    def test_grad_wrt_z(self, synth_inputs) -> None:
        """d/dz |Ktm|² deve ser finito e não-nulo (caso 3)."""

        def loss(z: float) -> float:
            # Usa caso 3 (same_layer above)
            inp = {**synth_inputs, "z": z}
            Ktm, _, _ = _call_case(_hmd_tiv_kernel_case3_jax, inp)
            return jnp.real(jnp.sum(Ktm * jnp.conj(Ktm)))

        g = jax.grad(loss)(synth_inputs["z"])
        g_np = float(g)
        assert np.isfinite(g_np)
        assert abs(g_np) > 0.0, "gradiente não deveria ser zero"

    def test_decoupling_grad_wrt_L(self) -> None:
        """d(ACx)/dL para L=1.0 → -3 ACx / L (diferenciável)."""
        L = 1.0
        _, ACx = decoupling_factors_jax(L)
        # ∂ACx/∂L = -3/(2π L⁴) = -3 ACx / L
        grad_fn = jax.grad(lambda L_: decoupling_factors_jax(L_)[1])
        g = grad_fn(L)
        expected = -3.0 * float(ACx) / L
        np.testing.assert_allclose(float(g), expected, rtol=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# TestCompileTimeBudget — regression guard
# ──────────────────────────────────────────────────────────────────────────────


class TestCompileTimeBudget:
    """Primeira chamada do dispatcher deve compilar em < 90s."""

    def test_first_call_compiles_under_90s(self, synth_inputs) -> None:
        """Regression guard — se lax.switch com 6 branches complexas
        estourar 90s, o orçamento está sendo quebrado."""
        t0 = time.time()
        out = _hmd_tiv_full_jax(
            0,  # case 1
            synth_inputs["u_r"],
            synth_inputs["s_r"],
            synth_inputs["RTEdw_r"],
            synth_inputs["RTEup_r"],
            synth_inputs["RTMdw_r"],
            synth_inputs["RTMup_r"],
            synth_inputs["Tudw_r"],
            synth_inputs["Tuup_r"],
            synth_inputs["Txdw_r"],
            synth_inputs["Txup_r"],
            synth_inputs["Mxdw"],
            synth_inputs["Mxup"],
            synth_inputs["Eudw"],
            synth_inputs["Euup"],
            synth_inputs["z"],
            synth_inputs["h0"],
            synth_inputs["prof_r"],
            synth_inputs["prof_r1"],
            synth_inputs["h_r"],
        )
        for arr in out:
            arr.block_until_ready()
        elapsed = time.time() - t0
        assert elapsed < 90.0, (
            f"Primeira chamada de _hmd_tiv_full_jax levou {elapsed:.1f}s "
            f"(limite 90s). Verifique se lax.switch não está inflando o "
            f"grafo."
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestImplementationStatus — sanity check do registry de status
# ──────────────────────────────────────────────────────────────────────────────


class TestImplementationStatus:
    """IMPLEMENTATION_STATUS deve refletir estado atual."""

    def test_status_has_full_dispatcher(self) -> None:
        assert "_hmd_tiv_full_jax" in IMPLEMENTATION_STATUS
        assert "✅" in IMPLEMENTATION_STATUS["_hmd_tiv_full_jax"]

    def test_status_marks_vmd_complete(self) -> None:
        # Sprint 3.3.3 (PR #11): VMD nativo completo — ETAPAS 3+6 ainda
        # pendentes para Sprint 3.3.4 (PR #12).
        assert "_vmd_full_jax" in IMPLEMENTATION_STATUS
        assert "✅" in IMPLEMENTATION_STATUS["_vmd_full_jax"]
        assert "ETAPAS 3+6 (TEdwz/TEupz prop + tensor assembly)" in IMPLEMENTATION_STATUS
        assert (
            "⏳"
            in IMPLEMENTATION_STATUS["ETAPAS 3+6 (TEdwz/TEupz prop + tensor assembly)"]
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestHybridFallback — use_native_dipoles=True cai para hybrid
# ──────────────────────────────────────────────────────────────────────────────


class TestHybridFallback:
    """Quando use_native_dipoles=True, deve emitir WARNING e funcionar
    como o caminho híbrido (bit-exato)."""

    def test_native_flag_warns_but_works(self, caplog) -> None:
        """Flag use_native_dipoles=True emite warning mas preserva resultado."""
        from geosteering_ai.simulation import SimulationConfig, simulate

        # Não há forma direta de passar use_native_dipoles através de
        # simulate() na Sprint 3.3.2 (PR #11 integrará). Aqui testamos o
        # kernel diretamente.
        from geosteering_ai.simulation._jax.kernel import (
            fields_in_freqs_jax_batch,
        )
        from geosteering_ai.simulation.filters import FilterLoader

        filt = FilterLoader().load("werthmuller_201pt")

        rho_h = np.array([1.0, 100.0, 1.0])
        rho_v = np.array([1.0, 200.0, 1.0])
        esp = np.array([5.0])
        positions_z = np.linspace(-1.0, 6.0, 3)
        freqs_hz = np.array([20000.0])

        import logging

        with caplog.at_level(
            logging.WARNING, logger="geosteering_ai.simulation._jax.kernel"
        ):
            out_native = fields_in_freqs_jax_batch(
                positions_z=positions_z,
                dz_half=0.5,
                r_half=0.0,
                dip_rad=0.0,
                n=3,
                rho_h=rho_h,
                rho_v=rho_v,
                esp=esp,
                freqs_hz=freqs_hz,
                krJ0J1=filt.abscissas,
                wJ0=filt.weights_j0,
                wJ1=filt.weights_j1,
                use_native_dipoles=True,  # Deve disparar warning + fallback
            )
        # Warning foi emitido?
        assert any(
            "use_native_dipoles=True" in record.message for record in caplog.records
        ), "Warning de fallback deveria ter sido emitido"
        # Resultado é o mesmo que hybrid (bit-exato, pois cai de volta)
        out_hybrid = fields_in_freqs_jax_batch(
            positions_z=positions_z,
            dz_half=0.5,
            r_half=0.0,
            dip_rad=0.0,
            n=3,
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            freqs_hz=freqs_hz,
            krJ0J1=filt.abscissas,
            wJ0=filt.weights_j0,
            wJ1=filt.weights_j1,
            use_native_dipoles=False,
        )
        np.testing.assert_allclose(out_native, out_hybrid, rtol=1e-14, atol=1e-16)
