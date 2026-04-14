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
    forward_pure_jax,
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
