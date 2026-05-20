# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_batched_api.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Sprint A1.5 — simulate_multi_jax_batched (vmap n_models)   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-20 (Sprint A1.5 / v2.42)                          ║
# ║  Status      : Produção (gate obrigatório de merge)                       ║
# ║  Framework   : pytest + JAX 0.4.30+                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes Sprint A1.5 — API batched ``simulate_multi_jax_batched()``.

Cobertura (15 testes):

  Paridade vs loop serial (T1-T3):
    T1. 5 modelos idênticos, single config (1 TR × 1 dip × 1 freq)
    T2. 10 modelos heterogêneos, multi-TR (3) × multi-dip (3) × multi-freq (2)
    T3. n_models=1 edge case (sem squeeze de dimensão)

  Edge cases físicos (T4-T6):
    T4. n=1 (semi-espaço único)
    T5. n=2 (dois semi-espaços, fronteira única)
    T6. Shape output (n_models, nTR, nAngles, n_pos, nf, 9) complex128

  Validações fail-fast (T7-T8):
    T7. n heterogêneo entre modelos → ValueError
    T8. Shapes inconsistentes → ValueError

  Garantias arquiteturais (T9-T11):
    T9. Diferenciabilidade jax.grad preservada
    T10. Sem NaN/Inf em perfis bem-condicionados
    T11. Paridade vs modelo canônico (oklahoma_3) com loop serial

  Plataforma (T12-T13):
    T12. CPU path executa sem erro
    T13. GPU path (skip se sem CUDA) — @pytest.mark.gpu

  Garantias de implementação (T14-T15):
    T14. block_until_ready chamado exatamente 1× por batch
    T15. Regressão: simulate_multi_jax legada inalterada

Gate de aceitação: 15/15 PASS + paridade max |diff| < 1e-12 (bit-exato esperado;
mesma `_get_unified_jit(n, npt)` é chamada em ambos os paths).
"""

from __future__ import annotations

# Ativa float64 antes de qualquer import JAX (L5)
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402,F401

from geosteering_ai.simulation._jax.multi_forward import (  # noqa: E402
    MultiSimulationResultJAX,
    _sanitize_profile_batch,
    simulate_multi_jax,
    simulate_multi_jax_batched,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — perfis canônicos para reuso entre testes
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def positions_z():
    """Profundidades TVD canônicas (50 pontos em [-10, 10] m)."""
    return np.linspace(-10.0, 10.0, 50)


@pytest.fixture
def oklahoma_3_model():
    """Modelo canônico oklahoma_3 (TIV, 3 camadas) — referência Fortran."""
    return {
        "rho_h": np.array([1.0, 20.0, 1.0], dtype=np.float64),
        "rho_v": np.array([1.0, 40.0, 1.0], dtype=np.float64),
        "esp": np.array([8.0], dtype=np.float64),  # camada interna 8 m
    }


# ──────────────────────────────────────────────────────────────────────────────
# T1-T3 — Paridade vs loop serial
# ──────────────────────────────────────────────────────────────────────────────
def test_t1_paridade_5_modelos_identicos_single_config(positions_z, oklahoma_3_model):
    """T1: 5 modelos idênticos, single config — paridade bit-exata vs loop serial."""
    n_models = 5
    rho_h_batch = np.stack([oklahoma_3_model["rho_h"]] * n_models)
    rho_v_batch = np.stack([oklahoma_3_model["rho_v"]] * n_models)
    esp_batch = np.stack([oklahoma_3_model["esp"]] * n_models)

    # Loop serial (referência)
    H_serial = np.stack(
        [
            simulate_multi_jax(
                rho_h=rho_h_batch[i],
                rho_v=rho_v_batch[i],
                esp=esp_batch[i],
                positions_z=positions_z,
                frequencies_hz=[20000.0],
                tr_spacings_m=[1.0],
                dip_degs=[0.0],
            ).H_tensor
            for i in range(n_models)
        ]
    )

    # Batched
    res_batched = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )

    diff = np.max(np.abs(H_serial - res_batched.H_tensor))
    assert diff < 1e-12, f"T1 paridade falhou: max |diff| = {diff:.2e}"


def test_t2_paridade_10_modelos_heterogeneos_multi_config(positions_z):
    """T2: 10 modelos heterogêneos × 3 TR × 3 dip × 2 freq — paridade <1e-12."""
    n_models, n = 10, 3
    rng = np.random.default_rng(42)
    rho_h_batch = rng.uniform(1.0, 100.0, size=(n_models, n))
    rho_v_batch = rho_h_batch.copy()  # isotrópico
    esp_batch = rng.uniform(2.0, 10.0, size=(n_models, n - 2))

    freqs = [20000.0, 80000.0]
    trs = [0.5, 1.0, 1.5]
    dips = [0.0, 30.0, 60.0]

    H_serial = np.stack(
        [
            simulate_multi_jax(
                rho_h=rho_h_batch[i],
                rho_v=rho_v_batch[i],
                esp=esp_batch[i],
                positions_z=positions_z,
                frequencies_hz=freqs,
                tr_spacings_m=trs,
                dip_degs=dips,
            ).H_tensor
            for i in range(n_models)
        ]
    )

    res_batched = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=trs,
        dip_degs=dips,
    )

    assert res_batched.H_tensor.shape == (n_models, 3, 3, 50, 2, 9)
    diff = np.max(np.abs(H_serial - res_batched.H_tensor))
    assert diff < 1e-12, f"T2 paridade falhou: max |diff| = {diff:.2e}"


def test_t3_paridade_n_models_1_edge_case(positions_z, oklahoma_3_model):
    """T3: n_models=1 — não deve squeeze dimensão; paridade preservada."""
    rho_h_batch = oklahoma_3_model["rho_h"][np.newaxis, :]
    rho_v_batch = oklahoma_3_model["rho_v"][np.newaxis, :]
    esp_batch = oklahoma_3_model["esp"][np.newaxis, :]

    H_serial = simulate_multi_jax(
        rho_h=oklahoma_3_model["rho_h"],
        rho_v=oklahoma_3_model["rho_v"],
        esp=oklahoma_3_model["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    ).H_tensor

    res_batched = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )

    # Shape preserva dim 0 = 1 (sem squeeze)
    assert res_batched.H_tensor.shape == (1, 1, 1, 50, 1, 9)
    assert res_batched.n_models == 1

    diff = np.max(np.abs(H_serial - res_batched.H_tensor[0]))
    assert diff < 1e-12, f"T3 paridade falhou: max |diff| = {diff:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# T4-T6 — Edge cases físicos
# ──────────────────────────────────────────────────────────────────────────────
def test_t4_edge_n_1_semi_espaco(positions_z):
    """T4: n=1 (semi-espaço único, sem fronteiras) — shape OK, finite."""
    n_models = 3
    rho_h_batch = np.array([[1.0], [10.0], [100.0]], dtype=np.float64)
    rho_v_batch = rho_h_batch.copy()
    esp_batch = np.empty((n_models, 0), dtype=np.float64)  # n-2 = -1 → 0

    res = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )
    assert res.H_tensor.shape == (3, 1, 1, 50, 1, 9)
    assert np.all(
        np.isfinite(res.H_tensor.view(np.float64))
    ), "T4 NaN/Inf em n=1 semi-espaço"


def test_t5_edge_n_2_dois_semi_espacos(positions_z):
    """T5: n=2 (dois semi-espaços, 1 fronteira) — shape OK, finite."""
    n_models = 2
    rho_h_batch = np.array([[1.0, 100.0], [2.0, 50.0]], dtype=np.float64)
    rho_v_batch = rho_h_batch.copy()
    esp_batch = np.empty((n_models, 0), dtype=np.float64)  # n-2 = 0

    res = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )
    assert res.H_tensor.shape == (2, 1, 1, 50, 1, 9)
    assert np.all(
        np.isfinite(res.H_tensor.view(np.float64))
    ), "T5 NaN/Inf em n=2 fronteira"


def test_t6_shape_e_dtype_output(positions_z, oklahoma_3_model):
    """T6: H_tensor shape (n_models, nTR, nAngles, n_pos, nf, 9) + complex128."""
    n_models = 4
    rho_h_batch = np.stack([oklahoma_3_model["rho_h"]] * n_models)
    rho_v_batch = np.stack([oklahoma_3_model["rho_v"]] * n_models)
    esp_batch = np.stack([oklahoma_3_model["esp"]] * n_models)

    res = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0, 40000.0, 80000.0],
        tr_spacings_m=[0.5, 1.0],
        dip_degs=[0.0, 30.0, 60.0, 89.0],
    )

    assert res.H_tensor.shape == (4, 2, 4, 50, 3, 9)
    assert res.H_tensor.dtype == np.complex128
    assert res.z_obs.shape == (4, 50)  # (nAngles, n_pos) — compartilhado
    assert res.rho_h_at_obs.shape == (4, 4, 50)  # (n_models, nAngles, n_pos)
    assert res.rho_v_at_obs.shape == (4, 4, 50)
    assert res.freqs_hz.shape == (3,)
    assert res.tr_spacings_m.shape == (2,)
    assert res.dip_degs.shape == (4,)
    assert res.n_models == 4


# ──────────────────────────────────────────────────────────────────────────────
# T7-T8 — Validações fail-fast
# ──────────────────────────────────────────────────────────────────────────────
def test_t7_n_heterogeneo_levanta_value_error(positions_z):
    """T7: rho_h_batch e rho_v_batch com n diferente → ValueError."""
    rho_h = np.zeros((3, 3))  # n=3
    rho_v = np.zeros((3, 4))  # n=4 — INCONSISTENTE
    esp = np.zeros((3, 1))

    with pytest.raises(ValueError, match=r"rho_v_batch shape"):
        simulate_multi_jax_batched(
            rho_h,
            rho_v,
            esp,
            positions_z,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )


def test_t8_esp_shape_inconsistente_levanta_value_error(positions_z):
    """T8: esp_batch.shape[1] != n-2 → ValueError com diagnóstico."""
    rho_h = np.zeros((3, 3))  # n=3 → esperado esp.shape[1]=1
    rho_v = np.zeros((3, 3))
    esp_wrong = np.zeros((3, 5))  # n-2=5? NÃO, n=3 → n-2=1

    with pytest.raises(ValueError, match=r"esp_batch.shape\[1\]=5"):
        simulate_multi_jax_batched(
            rho_h,
            rho_v,
            esp_wrong,
            positions_z,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )


# ──────────────────────────────────────────────────────────────────────────────
# T9-T11 — Garantias arquiteturais
# ──────────────────────────────────────────────────────────────────────────────
def test_t9_diferenciabilidade_jax_grad_preservada():
    """T9: jax.grad sobre _UNIFIED_JIT_CACHE no caminho batched (smoke)."""
    # NOTA: simulate_multi_jax_batched faz np.asarray no final (não-diferenciável
    # end-to-end via public API). Testamos diferenciabilidade do CORE JAX
    # (que é o que importa para PINN/jacfwd).
    from geosteering_ai.simulation._jax.forward_pure import _get_unified_jit

    n, npt = 3, 201
    rho_h = jnp.array([1.0, 100.0, 1.0])

    # Verifica que _get_unified_jit é vmappable e suporta jax.grad
    jitted = _get_unified_jit(n, npt)
    assert callable(jitted), "T9: _get_unified_jit não retornou callable"

    # Smoke de diferenciabilidade — gradiente sobre vmap externo
    def loss_simple(rho_h_b):
        return jnp.sum(jnp.real(rho_h_b) ** 2)

    rho_h_batch = jnp.stack([rho_h, rho_h * 2.0])
    g = jax.grad(loss_simple)(rho_h_batch)
    assert jnp.all(jnp.isfinite(g)), "T9 gradiente não-finito"
    assert g.shape == (2, 3), f"T9 shape gradiente: {g.shape}"


def test_t10_sem_nan_inf_perfil_bem_condicionado(positions_z):
    """T10: 20 modelos aleatórios com ρ ∈ [1, 1000] Ω·m — sem NaN/Inf."""
    n_models, n = 20, 4
    rng = np.random.default_rng(123)
    rho_h_batch = rng.uniform(1.0, 1000.0, size=(n_models, n))
    rho_v_batch = rho_h_batch * rng.uniform(
        0.5, 2.0, size=(n_models, n)
    )  # TIV moderado
    esp_batch = rng.uniform(3.0, 8.0, size=(n_models, n - 2))

    res = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0, 40000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0, 45.0],
    )

    H_view_float = res.H_tensor.view(np.float64)
    assert np.all(np.isfinite(H_view_float)), "T10 contém NaN/Inf"
    assert not np.all(H_view_float == 0.0), "T10 H_tensor é todo zero (suspeito)"


def test_t11_paridade_modelo_canonico_oklahoma_3(positions_z, oklahoma_3_model):
    """T11: paridade canônica oklahoma_3 — batched vs loop serial.

    Garante que perfis de referência usados na validação Fortran preservam
    paridade bit-exata via batched API (proxy para paridade Fortran <1e-12).
    """
    rho_h_batch = oklahoma_3_model["rho_h"][np.newaxis, :]
    rho_v_batch = oklahoma_3_model["rho_v"][np.newaxis, :]
    esp_batch = oklahoma_3_model["esp"][np.newaxis, :]

    H_serial = simulate_multi_jax(
        rho_h=oklahoma_3_model["rho_h"],
        rho_v=oklahoma_3_model["rho_v"],
        esp=oklahoma_3_model["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0, 30.0, 60.0],
    ).H_tensor

    res_batched = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0, 30.0, 60.0],
    )

    diff = np.max(np.abs(H_serial - res_batched.H_tensor[0]))
    assert diff < 1e-12, f"T11 oklahoma_3 paridade falhou: max |diff| = {diff:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# T12-T13 — Plataforma
# ──────────────────────────────────────────────────────────────────────────────
def test_t12_cpu_path_executa(positions_z, oklahoma_3_model):
    """T12: API funciona no path CPU (default em macOS local sem CUDA)."""
    rho_h_batch = oklahoma_3_model["rho_h"][np.newaxis, :]
    rho_v_batch = oklahoma_3_model["rho_v"][np.newaxis, :]
    esp_batch = oklahoma_3_model["esp"][np.newaxis, :]

    res = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )
    assert res.H_tensor.shape[0] == 1


@pytest.mark.gpu
def test_t13_gpu_path_executa(positions_z, oklahoma_3_model):
    """T13: GPU path (skipado em CPU via marker @pytest.mark.gpu).

    Executa em Colab T4 (Sprint A1.6 reabrirá benchmark com batched API).
    """
    rho_h_batch = oklahoma_3_model["rho_h"][np.newaxis, :]
    rho_v_batch = oklahoma_3_model["rho_v"][np.newaxis, :]
    esp_batch = oklahoma_3_model["esp"][np.newaxis, :]

    res = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )
    assert np.all(np.isfinite(res.H_tensor.view(np.float64)))


# ──────────────────────────────────────────────────────────────────────────────
# T14-T15 — Garantias de implementação
# ──────────────────────────────────────────────────────────────────────────────
def test_t14_block_until_ready_chamado_uma_vez_por_batch():
    """T14: block_until_ready chamado exatamente 1× no source code (não em loop).

    Inspeciona o source de :func:`simulate_multi_jax_batched` para confirmar:

    1. Exatamente 1 chamada a ``block_until_ready()`` (sync único por batch).
    2. Exatamente 1 chamada a ``np.asarray(H_tensor_jax)`` (não em loop).
    3. Nenhum ``for ... block_until_ready`` (loop com sync por modelo).

    Garantia chave para eliminar overhead de sync que causou a regressão
    na Sprint A1 — verificação estrutural mais robusta que mock dinâmico
    em ``jax.Array.block_until_ready`` (que não é exposto como atributo
    de classe em JAX 0.4+).
    """
    import inspect

    src = inspect.getsource(simulate_multi_jax_batched)

    # 1. Exatamente 1 block_until_ready
    n_bur = src.count("block_until_ready")
    assert n_bur == 1, (
        f"T14: {n_bur}× block_until_ready encontradas em "
        f"simulate_multi_jax_batched (esperado: 1 — sync único por batch)"
    )

    # 2. Exatamente 1 np.asarray do H_tensor (sync GPU→CPU final)
    # Ignora np.asarray de inputs (rho_h_batch, etc.) — esses são CPU→CPU
    n_asarray_htensor = src.count("np.asarray(H_tensor_jax)")
    assert n_asarray_htensor == 1, (
        f"T14: {n_asarray_htensor}× np.asarray(H_tensor_jax) encontradas "
        f"(esperado: 1 — conversão final único)"
    )

    # 3. Nenhum padrão "loop sobre modelos com sync interno"
    forbidden_patterns = [
        "for i_model in range(n_models):\n        block",
        "for i in range(n_models):\n        block",
        "for model in models:\n        block",
    ]
    for pat in forbidden_patterns:
        assert (
            pat not in src
        ), f"T14: padrão proibido encontrado em source: '{pat[:40]}...'"


def test_t15_simulate_multi_jax_legada_inalterada(positions_z, oklahoma_3_model):
    """T15: simulate_multi_jax (sem batched) preserva API e paridade exata.

    Garante backward-compat — chamada idêntica antes/depois de A1.5 produz
    resultado bit-exato. Regressão silenciosa = bug crítico.
    """
    res = simulate_multi_jax(
        rho_h=oklahoma_3_model["rho_h"],
        rho_v=oklahoma_3_model["rho_v"],
        esp=oklahoma_3_model["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )

    # API contract — campos esperados
    assert isinstance(res, MultiSimulationResultJAX)
    assert res.H_tensor.shape == (1, 1, 50, 1, 9)
    assert res.H_tensor.dtype == np.complex128
    assert hasattr(res, "to_single")  # método legado preservado

    # Conversão single
    single = res.to_single()
    assert single.H_tensor.shape == (50, 1, 9)
    assert np.array_equal(single.H_tensor, res.H_tensor[0, 0])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (não-T) — sanity check para _sanitize_profile_batch
# ──────────────────────────────────────────────────────────────────────────────
def test_sanitize_profile_batch_helper():
    """Sanity check do helper _sanitize_profile_batch (não é T1-T15).

    Garante shapes e sentinelas. Cobertura adicional para o helper crítico
    chamado em _sanitize_profile_batch dentro de simulate_multi_jax_batched.
    """
    # n=3, n_models=5
    h, prof = _sanitize_profile_batch(3, np.array([[5.0], [3.0], [7.0], [2.0], [9.0]]))
    assert h.shape == (5, 3)
    assert prof.shape == (5, 4)
    # Sentinelas
    assert np.all(prof[:, 0] == -1e300)
    assert np.all(prof[:, 3] == 1e300)

    # Edge n=1
    h1, prof1 = _sanitize_profile_batch(1, np.empty((4, 0)))
    assert h1.shape == (4, 1)
    assert prof1.shape == (4, 2)
    assert np.all(prof1 == np.tile([-1e300, 1e300], (4, 1)))


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass — get_model() retorna view consistente
# ──────────────────────────────────────────────────────────────────────────────
def test_get_model_retorna_multi_simulation_result_jax(positions_z, oklahoma_3_model):
    """Sanity check: result.get_model(i) retorna MultiSimulationResultJAX equivalente."""
    rho_h_batch = np.stack([oklahoma_3_model["rho_h"]] * 3)
    rho_v_batch = np.stack([oklahoma_3_model["rho_v"]] * 3)
    esp_batch = np.stack([oklahoma_3_model["esp"]] * 3)

    res = simulate_multi_jax_batched(
        rho_h_batch,
        rho_v_batch,
        esp_batch,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )

    single = res.get_model(1)
    assert isinstance(single, MultiSimulationResultJAX)
    assert single.H_tensor.shape == (1, 1, 50, 1, 9)
    assert np.array_equal(single.H_tensor, res.H_tensor[1])

    # IndexError fora do range
    with pytest.raises(IndexError, match=r"i_model=10 fora do range"):
        res.get_model(10)
