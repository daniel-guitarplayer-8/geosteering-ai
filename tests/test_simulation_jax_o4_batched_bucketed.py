# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_o4_batched_bucketed.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Sprint O4 (v2.44) — batched-bucketed (vmap modelos)        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-29 (Sprint O4)                                    ║
# ║  Status      : Produção (gate obrigatório de merge)                       ║
# ║  Framework   : pytest + JAX 0.4.30+                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes Sprint O4 — path BUCKETED de ``simulate_multi_jax_batched``.

O Sprint O4 faz ``simulate_multi_jax_batched`` usar o kernel BUCKETED (em vez
do ``_get_unified_jit`` hardcodado) quando a geometria é COMPARTILHADA entre
os modelos do batch (só ``rho`` varia — regime PINN/on-the-fly). Isso eleva o
throughput on-the-fly (gargalo #1 do simulador JAX GPU) reusando os kernels de
bucket compilados 1× e aplicando-os a todos os modelos via ``jax.vmap``.

Cobertura:

  Paridade (gate primário — mesmo kernel ⇒ bit-exato):
    • batched-bucketed vs loop serial ``simulate_multi_jax`` (bucketed), n=3/5/10
    • batched-bucketed vs batched-unified (fallback) — <1e-13
    • multi-dim (multi-TR × multi-dip × multi-freq) vs serial

  Fallback seguro (zero regressão):
    • geometria HETEROGÊNEA (esp varia entre modelos) → warning + unified

  Edge cases:
    • n=1 (semi-espaço) e n=2 (dois semi-espaços) via bucketed

  Garantias arquiteturais:
    • diferenciabilidade (jax.grad) através do vmap-modelos bucketed (PINN)
    • 1 único block_until_ready no helper bucketed (sync por batch, não modelo)

Gate de aceitação: 100% PASS + paridade max |diff| < 1e-13 (c128).
"""

from __future__ import annotations

import logging

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from geosteering_ai.simulation import (  # noqa: E402
    SimulationConfig,
    simulate_multi_jax,
)
from geosteering_ai.simulation._jax.multi_forward import (  # noqa: E402
    _build_H_tensor_batched_bucketed,
    _forward_config_buckets_over_models,
    simulate_multi_jax_batched,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _make_shared_geom_batch(n: int, n_models: int, seed: int = 0):
    """Batch com GEOMETRIA COMPARTILHADA (esp idêntico) e rho heterogêneo.

    Regime PINN/on-the-fly: cada modelo tem rho_h/rho_v distintos mas o mesmo
    perfil de espessuras ``esp`` (→ mesmos camad_t/camad_r → bucketing único).
    """
    rng = np.random.default_rng(seed)
    rho_h = rng.uniform(1.0, 100.0, size=(n_models, n))
    rho_v = rho_h.copy()  # isotrópico (suficiente p/ paridade de kernel)
    esp_row = rng.uniform(2.0, 10.0, size=max(n - 2, 0))
    esp = np.tile(esp_row[None, :], (n_models, 1))  # COMPARTILHADO entre modelos
    return rho_h, rho_v, esp


_CFG_BUCKETED = SimulationConfig(backend="jax", jax_strategy="bucketed")
_CFG_UNIFIED = SimulationConfig(backend="jax", jax_strategy="unified")


# ──────────────────────────────────────────────────────────────────────────────
# Paridade — gate primário
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [3, 5, 10])
def test_o4_batched_bucketed_paridade_vs_loop_serial(n):
    """batched-bucketed == loop serial simulate_multi_jax(bucketed) <1e-13.

    Mesmo kernel ``_get_bucket_jit`` em ambos → esperado bit-exato (~0). Cobre
    n=3/5/10 (modelos geológicos realistas, multi-bucket).
    """
    n_models = 6
    positions_z = np.linspace(-10.0, 10.0, 120)
    rho_h, rho_v, esp = _make_shared_geom_batch(n, n_models, seed=n)

    H_serial = np.stack(
        [
            simulate_multi_jax(
                rho_h=rho_h[i],
                rho_v=rho_v[i],
                esp=esp[i],
                positions_z=positions_z,
                frequencies_hz=[20000.0],
                tr_spacings_m=[1.0],
                dip_degs=[0.0],
                cfg=_CFG_BUCKETED,
            ).H_tensor
            for i in range(n_models)
        ]
    )

    res = simulate_multi_jax_batched(
        rho_h,
        rho_v,
        esp,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=_CFG_BUCKETED,
    )

    assert res.H_tensor.shape == (n_models, 1, 1, 120, 1, 9)
    diff = np.max(np.abs(H_serial - res.H_tensor))
    assert diff < 1e-13, f"n={n}: batched-bucketed vs serial max|diff|={diff:.2e}"


def test_o4_batched_bucketed_vs_batched_unified():
    """batched-bucketed == batched-unified (fallback) <1e-13 (geom homogênea)."""
    n, n_models = 5, 8
    positions_z = np.linspace(-10.0, 10.0, 100)
    rho_h, rho_v, esp = _make_shared_geom_batch(n, n_models, seed=7)
    kw = dict(
        frequencies_hz=[20000.0, 80000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )

    H_b = simulate_multi_jax_batched(
        rho_h, rho_v, esp, positions_z, cfg=_CFG_BUCKETED, **kw
    ).H_tensor
    H_u = simulate_multi_jax_batched(
        rho_h, rho_v, esp, positions_z, cfg=_CFG_UNIFIED, **kw
    ).H_tensor

    diff = np.max(np.abs(H_b - H_u))
    assert diff < 1e-13, f"bucketed vs unified batched: max|diff|={diff:.2e}"


def test_o4_multidim_paridade():
    """Multi-TR(3) × multi-dip(3) × multi-freq(2) — bucketed vs serial <1e-13.

    Cobre o loop Python sobre configs + dedup por hordist + vmap-freq interno.
    """
    n, n_models = 5, 5
    positions_z = np.linspace(-10.0, 10.0, 60)
    rho_h, rho_v, esp = _make_shared_geom_batch(n, n_models, seed=11)
    freqs = [20000.0, 80000.0]
    trs = [0.5, 1.0, 1.5]
    dips = [0.0, 30.0, 60.0]

    H_serial = np.stack(
        [
            simulate_multi_jax(
                rho_h=rho_h[i],
                rho_v=rho_v[i],
                esp=esp[i],
                positions_z=positions_z,
                frequencies_hz=freqs,
                tr_spacings_m=trs,
                dip_degs=dips,
                cfg=_CFG_BUCKETED,
            ).H_tensor
            for i in range(n_models)
        ]
    )

    res = simulate_multi_jax_batched(
        rho_h,
        rho_v,
        esp,
        positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=trs,
        dip_degs=dips,
        cfg=_CFG_BUCKETED,
    )

    assert res.H_tensor.shape == (n_models, 3, 3, 60, 2, 9)
    diff = np.max(np.abs(H_serial - res.H_tensor))
    assert diff < 1e-13, f"multi-dim bucketed vs serial: max|diff|={diff:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# Fallback seguro — geometria heterogênea
# ──────────────────────────────────────────────────────────────────────────────
def test_o4_geometria_heterogenea_fallback_unified(caplog):
    """esp heterogêneo + jax_strategy='bucketed' → warning + resultado == unified.

    Garante ZERO regressão e ZERO resultado fisicamente errado: geometria
    divergente NÃO pode usar bucketing compartilhado; o dispatcher emite
    warning e cai para o path unified (correto).
    """
    n, n_models = 4, 4
    positions_z = np.linspace(-10.0, 10.0, 50)
    rng = np.random.default_rng(3)
    rho_h = rng.uniform(1.0, 100.0, size=(n_models, n))
    rho_v = rho_h.copy()
    esp = rng.uniform(2.0, 10.0, size=(n_models, n - 2))  # HETEROGÊNEO
    kw = dict(frequencies_hz=[20000.0], tr_spacings_m=[1.0], dip_degs=[0.0])

    with caplog.at_level(
        logging.WARNING, logger="geosteering_ai.simulation._jax.multi_forward"
    ):
        H_req_bucketed = simulate_multi_jax_batched(
            rho_h, rho_v, esp, positions_z, cfg=_CFG_BUCKETED, **kw
        ).H_tensor

    # 1. Warning de fallback foi emitido.
    assert any(
        "heterog" in rec.message.lower() for rec in caplog.records
    ), "esperado warning de geometria heterogênea (fallback unified)"

    # 2. Resultado idêntico ao unified explícito (fallback correto).
    H_unified = simulate_multi_jax_batched(
        rho_h, rho_v, esp, positions_z, cfg=_CFG_UNIFIED, **kw
    ).H_tensor
    diff = np.max(np.abs(H_req_bucketed - H_unified))
    assert diff < 1e-13, f"fallback heterogêneo != unified: max|diff|={diff:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases — n=1 / n=2
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [1, 2])
def test_o4_edge_n1_n2_bucketed(n):
    """n=1 (semi-espaço) e n=2 via bucketed — finite + paridade vs serial.

    Regressão da colisão de chave _CTX_CACHE n=1/n=2 (esp vazio em ambos):
    se a chave colidisse, n=2 receberia ctx de n=1 (h_arr (1,)) → crash.
    """
    n_models = 3
    positions_z = np.linspace(-5.0, 5.0, 30)
    rng = np.random.default_rng(n)
    rho_h = rng.uniform(1.0, 100.0, size=(n_models, n))
    rho_v = rho_h.copy()
    esp = np.empty((n_models, max(n - 2, 0)), dtype=np.float64)

    res = simulate_multi_jax_batched(
        rho_h,
        rho_v,
        esp,
        positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=_CFG_BUCKETED,
    )
    assert res.H_tensor.shape == (n_models, 1, 1, 30, 1, 9)
    assert np.all(
        np.isfinite(res.H_tensor.view(np.float64))
    ), f"n={n}: NaN/Inf no path bucketed"

    H_serial = np.stack(
        [
            simulate_multi_jax(
                rho_h=rho_h[i],
                rho_v=rho_v[i],
                esp=esp[i],
                positions_z=positions_z,
                frequencies_hz=[20000.0],
                tr_spacings_m=[1.0],
                dip_degs=[0.0],
                cfg=_CFG_BUCKETED,
            ).H_tensor
            for i in range(n_models)
        ]
    )
    diff = np.max(np.abs(H_serial - res.H_tensor))
    assert diff < 1e-13, f"n={n} edge bucketed vs serial: max|diff|={diff:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# Garantias arquiteturais
# ──────────────────────────────────────────────────────────────────────────────
def test_o4_diferenciabilidade_bucketed_kernel():
    """jax.grad flui através do vmap-modelos bucketed (relevante p/ PINN).

    Testa o núcleo diferenciável ``_forward_config_buckets_over_models``
    (jnp puro), não a API pública (que sincroniza p/ NumPy).
    """
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context_cached,
    )

    n, n_models = 3, 4
    positions_z = np.linspace(-8.0, 8.0, 20)
    ctx = build_static_context_cached(
        rho_h=np.full(n, 10.0),
        rho_v=np.full(n, 10.0),
        esp=np.array([6.0]),
        positions_z=positions_z,
        freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0,
        dip_deg=0.0,
        strategy="bucketed",
        complex_dtype="complex128",
    )
    rng = np.random.default_rng(0)
    rho_h_batch = jnp.asarray(rng.uniform(1.0, 100.0, size=(n_models, n)))
    rho_v_batch = rho_h_batch

    def scalar_out(rho_h_b):
        H = _forward_config_buckets_over_models(
            ctx=ctx,
            rho_h_batch_jnp=rho_h_b,
            rho_v_batch_jnp=rho_v_batch,
            n=n,
            npt=ctx.npt,
            n_models=n_models,
            n_pos=positions_z.shape[0],
            nf=1,
            complex_dtype="complex128",
        )
        return jnp.sum(jnp.abs(H))

    grad = jax.grad(scalar_out)(rho_h_batch)
    assert grad.shape == (n_models, n)
    assert np.all(np.isfinite(np.asarray(grad))), "grad com NaN/Inf"
    assert np.any(np.asarray(grad) != 0.0), "grad identicamente nulo (suspeito)"


def test_o4_block_until_ready_unico_bucketed():
    """Gate T14 (bucketed): 1 único block_until_ready + np.asarray no helper.

    Sync por BATCH, nunca por modelo nem por config. Inspeção de source dos
    helpers bucketed (_build_H_tensor_batched_bucketed + sub-helper de config).
    """
    import inspect

    src = inspect.getsource(_build_H_tensor_batched_bucketed) + inspect.getsource(
        _forward_config_buckets_over_models
    )
    # Conta a CHAMADA (`.block_until_ready()`), não menções em docstring/prosa.
    n_bur = src.count(".block_until_ready()")
    assert (
        n_bur == 1
    ), f"bucketed: {n_bur}× .block_until_ready() (esperado 1 — sync único/batch)"
    n_asarray = src.count("np.asarray(H_tensor_jax)")
    assert (
        n_asarray == 1
    ), f"bucketed: {n_asarray}× np.asarray(H_tensor_jax) (esperado 1)"


# ──────────────────────────────────────────────────────────────────────────────
# chunk_size_models — invariância (OOM fix Cenário H)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("chunk", [None, 4, 8, 16, 5])
def test_o4_chunk_size_models_invariante(chunk):
    """jax_chunk_size_models não altera o resultado — bit-exato vs monolítico.

    Chunking apenas reordena o loop Python sobre fatias de modelos (mesma
    matemática por modelo). Inclui chunk=5 (NÃO divisor de 16 → última fatia
    parcial) para validar o caso de fatia menor. Gate: max|diff| == 0.
    """
    n, n_models = 5, 16
    positions_z = np.linspace(-10.0, 10.0, 80)
    rho_h, rho_v, esp = _make_shared_geom_batch(n, n_models, seed=42)
    kw = dict(frequencies_hz=[20000.0], tr_spacings_m=[1.0], dip_degs=[0.0])

    # Referência monolítica (chunk=None).
    H_mono = simulate_multi_jax_batched(
        rho_h, rho_v, esp, positions_z, cfg=_CFG_BUCKETED, **kw
    ).H_tensor

    cfg_chunk = SimulationConfig(
        backend="jax", jax_strategy="bucketed", jax_chunk_size_models=chunk
    )
    H_chunk = simulate_multi_jax_batched(
        rho_h, rho_v, esp, positions_z, cfg=cfg_chunk, **kw
    ).H_tensor

    assert H_chunk.shape == H_mono.shape == (n_models, 1, 1, 80, 1, 9)
    diff = np.max(np.abs(H_mono - H_chunk))
    assert (
        diff == 0.0
    ), f"chunk={chunk}: chunking alterou resultado max|diff|={diff:.2e}"


def test_o4_chunk_size_models_unified_path():
    """Chunking também funciona no fallback unified (geometria heterogênea).

    Tolerância <1e-13 (não bit-exato): o kernel unified usa vmap+fori_loop, e
    o XLA pode reordenar reduções conforme o tamanho do batch (n_models vs
    chunk) → diferença ~ULP float64 (~7e-15), fisicamente idêntica. O path
    bucketed, por construção independente-por-modelo, é bit-exato (==0, ver
    test_o4_chunk_size_models_invariante).
    """
    n, n_models = 4, 10
    positions_z = np.linspace(-10.0, 10.0, 50)
    rng = np.random.default_rng(5)
    rho_h = rng.uniform(1.0, 100.0, size=(n_models, n))
    rho_v = rho_h.copy()
    esp = rng.uniform(2.0, 10.0, size=(n_models, n - 2))  # HETEROGÊNEO → unified
    kw = dict(frequencies_hz=[20000.0], tr_spacings_m=[1.0], dip_degs=[0.0])

    H_mono = simulate_multi_jax_batched(
        rho_h, rho_v, esp, positions_z, cfg=_CFG_UNIFIED, **kw
    ).H_tensor
    cfg_chunk = SimulationConfig(
        backend="jax", jax_strategy="unified", jax_chunk_size_models=3
    )
    H_chunk = simulate_multi_jax_batched(
        rho_h, rho_v, esp, positions_z, cfg=cfg_chunk, **kw
    ).H_tensor

    diff = np.max(np.abs(H_mono - H_chunk))
    assert diff < 1e-13, f"chunk unified alterou resultado: max|diff|={diff:.2e}"
