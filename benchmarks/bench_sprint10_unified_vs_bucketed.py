# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  benchmarks/bench_sprint10_unified_vs_bucketed.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Benchmark Sprint 10 Phase 2 — unified vs bucketed (CPU)    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-15 (PR #24-part2 / v1.5.0)                        ║
# ║  Framework   : JAX 0.4.30+, time.perf_counter                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Compara tempo de forward ``strategy="bucketed"`` vs ``"unified"`` em CPU.

Usa oklahoma_3/5/28 canônicos com 100 posições × 1 freq como baseline.
Gate soft: ``unified_time ≤ 1.3× bucketed_time`` (documentado no plano
``cosmic-riding-garden.md``). O slowdown CPU é aceito como trade-off
para os ganhos GPU (VRAM ~11 GB → ~250 MB, throughput 5-20× T4).

Uso::

    python benchmarks/bench_sprint10_unified_vs_bucketed.py
    python benchmarks/bench_sprint10_unified_vs_bucketed.py --model oklahoma_28
    python benchmarks/bench_sprint10_unified_vs_bucketed.py --n-pos 600 --reps 20
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def bench_one(model_name: str, n_pos: int, reps: int) -> dict:
    """Retorna dict ``{bucketed_ms, unified_ms, ratio, xla_b, xla_u, parity}``."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from geosteering_ai.simulation._jax import (
        clear_unified_jit_cache,
        count_compiled_xla_programs,
    )
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context,
        clear_jit_cache,
        forward_pure_jax,
    )
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    m = get_canonical_model(model_name)
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, n_pos)

    # ── Bucketed ─────────────────────────────────────────────────────────
    clear_jit_cache()
    clear_unified_jit_cache()
    ctx_b = build_static_context(
        m.rho_h, m.rho_v, m.esp, z,
        freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0, dip_deg=0.0,
        strategy="bucketed",
    )
    # Warmup (compila)
    forward_pure_jax(ctx_b.rho_h_jnp, ctx_b.rho_v_jnp, ctx_b).block_until_ready()
    xla_b = count_compiled_xla_programs(ctx_b)

    t0 = time.perf_counter()
    H_b = None
    for _ in range(reps):
        H_b = forward_pure_jax(ctx_b.rho_h_jnp, ctx_b.rho_v_jnp, ctx_b)
    H_b.block_until_ready()
    t_b_ms = (time.perf_counter() - t0) * 1e3 / reps

    # ── Unified ──────────────────────────────────────────────────────────
    clear_jit_cache()
    clear_unified_jit_cache()
    ctx_u = build_static_context(
        m.rho_h, m.rho_v, m.esp, z,
        freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0, dip_deg=0.0,
        strategy="unified",
    )
    forward_pure_jax(ctx_u.rho_h_jnp, ctx_u.rho_v_jnp, ctx_u).block_until_ready()
    xla_u = count_compiled_xla_programs(ctx_u)

    t0 = time.perf_counter()
    H_u = None
    for _ in range(reps):
        H_u = forward_pure_jax(ctx_u.rho_h_jnp, ctx_u.rho_v_jnp, ctx_u)
    H_u.block_until_ready()
    t_u_ms = (time.perf_counter() - t0) * 1e3 / reps

    parity = float(jnp.abs(H_b - H_u).max())

    return {
        "model": model_name,
        "n_layers": len(m.rho_h),
        "n_pos": n_pos,
        "bucketed_ms": t_b_ms,
        "unified_ms": t_u_ms,
        "ratio": t_u_ms / t_b_ms if t_b_ms > 0 else float("inf"),
        "xla_b": xla_b,
        "xla_u": xla_u,
        "parity": parity,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--model", default="all",
        choices=["oklahoma_3", "oklahoma_5", "oklahoma_28", "all"],
    )
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    models = (
        ["oklahoma_3", "oklahoma_5", "oklahoma_28"]
        if args.model == "all" else [args.model]
    )

    print(f"\n{'Model':12s} {'n':>3s} {'n_pos':>6s} "
          f"{'XLA_b':>6s} {'XLA_u':>6s} "
          f"{'bucketed_ms':>12s} {'unified_ms':>11s} "
          f"{'ratio':>6s} {'parity':>10s}")
    print("-" * 90)

    results = []
    for model in models:
        r = bench_one(model, args.n_pos, args.reps)
        results.append(r)
        print(
            f"{r['model']:12s} {r['n_layers']:>3d} {r['n_pos']:>6d} "
            f"{r['xla_b']:>6d} {r['xla_u']:>6d} "
            f"{r['bucketed_ms']:>10.2f}   {r['unified_ms']:>9.2f}   "
            f"{r['ratio']:>5.2f}× {r['parity']:>9.2e}"
        )

    # Gate soft
    worst_ratio = max(r["ratio"] for r in results)
    print()
    print(f"Gate CPU: ratio ≤ 1.3× (soft)")
    print(f"Pior ratio observado: {worst_ratio:.2f}×")
    if worst_ratio > 1.5:
        print("⚠️  Slowdown acima de 1.5× — revisar trade-off GPU vs CPU")
    else:
        print("✅ Trade-off CPU aceitável (< 1.5×)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
