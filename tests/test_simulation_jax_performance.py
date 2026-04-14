# -*- coding: utf-8 -*-
"""Testes Sprint 7.x — Performance do `forward_pure_jax` otimizado.

Cobertura:

- Cache JIT reutilizado (5 chamadas consecutivas rápidas após warmup).
- Paridade bit-a-bit vs Numba pós-otimização.
- CPU < 500 ms/modelo (gate Sprint 7.4).
- Estabilidade alta ρ após otimização.

Requer ``JAX_ENABLE_X64=True``.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("jax")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from geosteering_ai.simulation import simulate  # noqa: E402
from geosteering_ai.simulation._jax.forward_pure import (  # noqa: E402
    build_static_context,
    clear_jit_cache,
    forward_pure_jax,
    get_jit_cache_info,
    set_jit_cache_maxsize,
)
from geosteering_ai.simulation.config import SimulationConfig  # noqa: E402


@pytest.fixture
def model_oklahoma_3():
    """Oklahoma 3 canônico, 100 posições, 1 frequência."""
    rho_h = np.array([10.0, 100.0, 10.0])
    rho_v = np.array([10.0, 100.0, 10.0])
    esp = np.array([5.0])
    z = np.linspace(-2.0, 7.0, 100)
    return rho_h, rho_v, esp, z


# ═════════════════════════════════════════════════════════════════════════
# 1 — Cache JIT reutilizado (chamadas consecutivas rápidas após warmup)
# ═════════════════════════════════════════════════════════════════════════


def test_forward_pure_jit_cache_reused(model_oklahoma_3) -> None:
    """Após 1a chamada (compile), 5 chamadas seguintes < 300ms cada."""
    rho_h, rho_v, esp, z = model_oklahoma_3
    ctx = build_static_context(
        rho_h,
        rho_v,
        esp,
        z,
        freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0,
        dip_deg=0.0,
    )
    # Warmup (trace + compile)
    H = forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx)
    H.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(5):
        H = forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx)
        H.block_until_ready()
    elapsed_per_call = (time.perf_counter() - t0) / 5

    # Gate: cache reutilizado → < 300ms/chamada em CPU Intel i9.
    # Hardwares mais modestos podem precisar de >100ms — gate frouxo.
    assert (
        elapsed_per_call < 0.3
    ), f"Cache JIT NÃO reutilizado: {elapsed_per_call*1000:.0f}ms/chamada"


# ═════════════════════════════════════════════════════════════════════════
# 2 — Paridade bit-a-bit vs Numba pós-otimização
# ═════════════════════════════════════════════════════════════════════════


def test_forward_pure_matches_numba_post_optim(model_oklahoma_3) -> None:
    """forward_pure_jax otimizado == Numba (max_abs < 1e-10)."""
    rho_h, rho_v, esp, z = model_oklahoma_3
    ctx = build_static_context(
        rho_h,
        rho_v,
        esp,
        z,
        freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0,
        dip_deg=0.0,
    )
    H_jax = np.asarray(forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx))

    cfg = SimulationConfig(frequency_hz=20000.0, tr_spacing_m=1.0, backend="numba")
    res = simulate(rho_h=rho_h, rho_v=rho_v, esp=esp, positions_z=z, cfg=cfg)
    H_n = res.H_tensor
    if H_n.ndim == 2:
        H_n = H_n[:, np.newaxis, :]

    max_abs = float(np.max(np.abs(H_jax - H_n)))
    assert max_abs < 1e-10, f"Paridade regrediu pós-otimização: max_abs={max_abs:.2e}"


# ═════════════════════════════════════════════════════════════════════════
# 3 — CPU < 500 ms/modelo (gate Sprint 7.4)
# ═════════════════════════════════════════════════════════════════════════


def test_forward_pure_cpu_under_500ms(model_oklahoma_3) -> None:
    """CPU oklahoma_3 100 pos < 500ms pós-warmup."""
    rho_h, rho_v, esp, z = model_oklahoma_3
    ctx = build_static_context(
        rho_h,
        rho_v,
        esp,
        z,
        freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0,
        dip_deg=0.0,
    )
    H = forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx)
    H.block_until_ready()

    t0 = time.perf_counter()
    H = forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx)
    H.block_until_ready()
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"Sprint 7.4 gate falhou: {elapsed*1000:.0f}ms/modelo > 500ms"


# ═════════════════════════════════════════════════════════════════════════
# 4 — Estabilidade alta ρ pós-otimização (ρ>1000 Ω·m)
# ═════════════════════════════════════════════════════════════════════════


def test_forward_pure_high_rho_post_optim() -> None:
    """ρ=1500 Ω·m → H_tensor finito em todas componentes."""
    rho_h = np.array([10.0, 1500.0, 10.0])
    rho_v = np.array([10.0, 3000.0, 10.0])
    esp = np.array([3.0])
    z = np.linspace(0.0, 3.0, 50)
    ctx = build_static_context(
        rho_h,
        rho_v,
        esp,
        z,
        freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0,
        dip_deg=0.0,
    )
    H = forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx)
    H_np = np.asarray(H)
    assert np.all(np.isfinite(H_np.real))
    assert np.all(np.isfinite(H_np.imag))


# ═════════════════════════════════════════════════════════════════════════
# 5 — LRU cache: clear, maxsize, eviction (PR #14e — Sprint 7.x+)
# ═════════════════════════════════════════════════════════════════════════


def test_jit_cache_clear_and_info() -> None:
    """clear_jit_cache + get_jit_cache_info retornam valores coerentes."""
    clear_jit_cache()
    info = get_jit_cache_info()
    assert info["n_entries"] == 0
    assert info["maxsize"] >= 1
    assert info["keys"] == []


def test_jit_cache_populates_after_forward() -> None:
    """Após forward_pure_jax, cache contém >= 1 bucket."""
    clear_jit_cache()
    set_jit_cache_maxsize(64)
    rho_h = np.array([10.0, 100.0, 10.0])
    rho_v = np.array([10.0, 100.0, 10.0])
    esp = np.array([5.0])
    z = np.linspace(-2.0, 7.0, 50)
    ctx = build_static_context(
        rho_h, rho_v, esp, z, freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0, dip_deg=0.0,
    )
    H = forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx)
    H.block_until_ready()
    info = get_jit_cache_info()
    assert info["n_entries"] >= 1, "Cache deveria ter entradas após forward"
    clear_jit_cache()


def test_jit_cache_eviction_lru() -> None:
    """Com maxsize=3 e oklahoma_28 (44 buckets), cache estabiliza em 3."""
    clear_jit_cache()
    set_jit_cache_maxsize(3)

    # Oklahoma 28 — usa get_canonical_model
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model("oklahoma_28")
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 100)
    ctx = build_static_context(
        m.rho_h, m.rho_v, m.esp, z,
        freqs_hz=np.array([20000.0]), tr_spacing_m=1.0, dip_deg=0.0,
    )
    H = forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx)
    H.block_until_ready()
    info = get_jit_cache_info()
    # Após execução com 44 buckets e maxsize=3, esperamos exatamente 3.
    assert info["n_entries"] == 3, (
        f"LRU não evictou corretamente: cache tem {info['n_entries']} "
        "entradas com maxsize=3"
    )
    # Restaura default.
    set_jit_cache_maxsize(64)
    clear_jit_cache()


def test_jit_cache_parity_after_eviction() -> None:
    """Paridade bit-a-bit vs Numba mantida mesmo após eviction LRU."""
    clear_jit_cache()
    set_jit_cache_maxsize(3)

    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )
    from geosteering_ai.simulation.config import SimulationConfig

    m = get_canonical_model("oklahoma_5")
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, 80)
    ctx = build_static_context(
        m.rho_h, m.rho_v, m.esp, z,
        freqs_hz=np.array([20000.0]), tr_spacing_m=1.0, dip_deg=0.0,
    )
    H_jax = np.asarray(forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx))

    cfg = SimulationConfig(frequency_hz=20000.0, tr_spacing_m=1.0, backend="numba")
    res = simulate(rho_h=m.rho_h, rho_v=m.rho_v, esp=m.esp, positions_z=z, cfg=cfg)
    H_n = res.H_tensor
    if H_n.ndim == 2:
        H_n = H_n[:, np.newaxis, :]

    max_abs = float(np.max(np.abs(H_jax - H_n)))
    assert max_abs < 1e-10, (
        f"Paridade regrediu após eviction LRU: max_abs={max_abs:.2e}"
    )
    set_jit_cache_maxsize(64)
    clear_jit_cache()


def test_set_jit_cache_maxsize_validation() -> None:
    """set_jit_cache_maxsize rejeita valores < 1."""
    with pytest.raises(ValueError):
        set_jit_cache_maxsize(0)
    with pytest.raises(ValueError):
        set_jit_cache_maxsize(-5)


# ═════════════════════════════════════════════════════════════════════════
# 6 — Sprint 8: warmup_all_buckets + forward_pure_jax_chunked (PR #14f)
# ═════════════════════════════════════════════════════════════════════════


def test_warmup_all_buckets_returns_bucket_count() -> None:
    """warmup_all_buckets retorna número de buckets compilados."""
    from geosteering_ai.simulation._jax.forward_pure import warmup_all_buckets

    clear_jit_cache()
    rho_h = np.array([10.0, 100.0, 10.0])
    rho_v = np.array([10.0, 100.0, 10.0])
    esp = np.array([5.0])
    z = np.linspace(-2.0, 7.0, 50)
    ctx = build_static_context(
        rho_h, rho_v, esp, z, freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0, dip_deg=0.0,
    )
    n = warmup_all_buckets(ctx)
    assert n >= 1
    info = get_jit_cache_info()
    assert info["n_entries"] == n
    clear_jit_cache()


def test_forward_pure_jax_chunked_parity() -> None:
    """forward_pure_jax_chunked retorna resultado bit-a-bit idêntico ao default."""
    from geosteering_ai.simulation._jax.forward_pure import (
        forward_pure_jax_chunked,
    )

    clear_jit_cache()
    rho_h = np.array([10.0, 50.0, 5.0])
    rho_v = np.array([10.0, 100.0, 5.0])
    esp = np.array([4.0])
    z = np.linspace(-1.0, 8.0, 64)
    ctx = build_static_context(
        rho_h, rho_v, esp, z, freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0, dip_deg=0.0,
    )
    H_default = np.asarray(forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx))
    H_chunk = np.asarray(
        forward_pure_jax_chunked(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx, chunk_size=16)
    )
    max_diff = float(np.max(np.abs(H_default - H_chunk)))
    assert max_diff < 1e-13, (
        f"chunked diverge do default: max_diff={max_diff:.2e}"
    )
    clear_jit_cache()


def test_forward_pure_jax_chunked_small_passes_through() -> None:
    """n_pos <= chunk_size: passa direto (1 chunk)."""
    from geosteering_ai.simulation._jax.forward_pure import (
        forward_pure_jax_chunked,
    )

    rho_h = np.array([10.0, 100.0, 10.0])
    rho_v = np.array([10.0, 100.0, 10.0])
    esp = np.array([5.0])
    z = np.linspace(-2.0, 7.0, 10)
    ctx = build_static_context(
        rho_h, rho_v, esp, z, freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0, dip_deg=0.0,
    )
    H = forward_pure_jax_chunked(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx, chunk_size=64)
    assert H.shape == (10, 1, 9)
    H.block_until_ready()


def test_forward_pure_jax_chunked_validates_chunk_size() -> None:
    """chunk_size < 1 levanta ValueError."""
    from geosteering_ai.simulation._jax.forward_pure import (
        forward_pure_jax_chunked,
    )

    rho_h = np.array([10.0, 100.0, 10.0])
    rho_v = np.array([10.0, 100.0, 10.0])
    esp = np.array([5.0])
    z = np.linspace(-2.0, 7.0, 20)
    ctx = build_static_context(
        rho_h, rho_v, esp, z, freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0, dip_deg=0.0,
    )
    with pytest.raises(ValueError):
        forward_pure_jax_chunked(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx, chunk_size=0)


# ═════════════════════════════════════════════════════════════════════════
# 7 — Sprint 9: forward_pure_jax_pmap (PR #14f)
# ═════════════════════════════════════════════════════════════════════════


def test_forward_pure_jax_pmap_single_device() -> None:
    """forward_pure_jax_pmap com n_devices=1 funciona (ambiente mono-GPU/CPU)."""
    from geosteering_ai.simulation._jax.forward_pure import forward_pure_jax_pmap

    n_devices = jax.local_device_count()
    # Ambiente mono-device: testa com batch igual a n_devices (=1).
    rho_h = np.array([10.0, 100.0, 10.0])
    rho_v = np.array([10.0, 100.0, 10.0])
    esp = np.array([5.0])
    z = np.linspace(-2.0, 7.0, 20)
    ctx = build_static_context(
        rho_h, rho_v, esp, z, freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0, dip_deg=0.0,
    )
    # Cria batch com shape (n_devices, n_layers).
    rho_h_batch = jax.numpy.stack([ctx.rho_h_jnp] * n_devices, axis=0)
    rho_v_batch = jax.numpy.stack([ctx.rho_v_jnp] * n_devices, axis=0)

    H_pmap = forward_pure_jax_pmap(rho_h_batch, rho_v_batch, ctx)
    assert H_pmap.shape == (n_devices, 20, 1, 9)
    assert np.all(np.isfinite(np.asarray(H_pmap).real))


def test_forward_pure_jax_pmap_mismatch_raises() -> None:
    """pmap rejeita batch com shape[0] != n_devices."""
    from geosteering_ai.simulation._jax.forward_pure import forward_pure_jax_pmap

    n_devices = jax.local_device_count()
    rho_h = np.array([10.0, 100.0, 10.0])
    rho_v = np.array([10.0, 100.0, 10.0])
    esp = np.array([5.0])
    z = np.linspace(-2.0, 7.0, 10)
    ctx = build_static_context(
        rho_h, rho_v, esp, z, freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0, dip_deg=0.0,
    )
    # Batch deliberadamente com shape[0]=n_devices+1 → deve raise.
    wrong = jax.numpy.stack([ctx.rho_h_jnp] * (n_devices + 1), axis=0)
    with pytest.raises(ValueError, match="n_devices"):
        forward_pure_jax_pmap(wrong, wrong, ctx)
