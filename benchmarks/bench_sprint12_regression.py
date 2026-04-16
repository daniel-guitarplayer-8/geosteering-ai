# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  benchmarks/bench_sprint12_regression.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Benchmark Sprint 12 — matriz regressão 600×3 + vmap_real   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-16 (PR #25 / v1.6.0)                              ║
# ║  Framework   : pytest-free CLI (argparse) + JAX                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Benchmark Sprint 12 — matriz completa `(n_pos × nf × nTR × nAng × strategy)`.

Mede throughput, latência, XLA count, paridade e (quando disponível) VRAM
para 4 modelos canônicos em 3 estratégias:

    1. ``bucketed``   — caminho legacy (default v1.5.0)
    2. ``unified``    — 1 XLA program (Sprint 10 Phase 2)
    3. ``vmap_real``  — vmap aninhado (Sprint 12, opt-in)

Modos:

    --matrix short  → 24 pontos (CI-friendly, ~5 min CPU)
    --matrix full   → 192 pontos (análise manual, ~2 h CPU / ~15 min T4)

CLI
---
.. code-block:: bash

    # CPU short (CI)
    python benchmarks/bench_sprint12_regression.py --backend cpu \\
        --matrix short --out results/sprint12_cpu_short.csv

    # T4 full (Colab Pro+)
    python benchmarks/bench_sprint12_regression.py --backend gpu \\
        --matrix full --out results/sprint12_gpu_full.csv

    # Com profiler XLA em 5 configs críticas
    python benchmarks/bench_sprint12_regression.py --backend gpu \\
        --matrix critical --profile-out /tmp/sprint12_trace
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 12 regression matrix benchmark — bucketed vs unified vs vmap_real"
    )
    p.add_argument(
        "--matrix",
        choices=["short", "full", "critical"],
        default="short",
        help="short=24 pts; full=192 pts; critical=5 pts críticos com profiler",
    )
    p.add_argument(
        "--backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help="JAX backend (apenas informativo — runtime usa jax.default_backend())",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["oklahoma_3", "oklahoma_5", "oklahoma_15", "oklahoma_28"],
        help="Modelos canônicos a benchmarkar",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/sprint12_benchmark.csv"),
        help="Caminho de saída do CSV",
    )
    p.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="Diretório para jax.profiler traces (se --matrix critical)",
    )
    p.add_argument(
        "--repeats", type=int, default=5, help="Repetições pós-warmup (default 5)"
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Configurações de matriz
# ──────────────────────────────────────────────────────────────────────────────
# short — 24 pontos (CI)
#   n_pos ∈ {100, 600}, nf ∈ {1, 3}, nTR=1, nAng=1, strategies∈{bucketed,unified,vmap_real}
SHORT_GRID = list(
    itertools.product(
        [100, 600],  # n_pos
        [1, 3],  # nf
        [1],  # nTR
        [1],  # nAng
        ["bucketed", "unified", "vmap_real"],  # strategy
    )
)

# full — 192 pontos (análise)
FULL_GRID = list(
    itertools.product(
        [100, 300, 600],  # n_pos
        [1, 3],  # nf
        [1, 3],  # nTR
        [1, 5],  # nAng
        ["bucketed", "unified"],  # 2 strategies × 4 modelos = 96; vmap_real adiciona
    )
)

# critical — 5 configs críticas para jax.profiler
CRITICAL_CONFIGS = [
    # (n_pos, nf, nTR, nAng, strategy) — meta: investigar regressão 600×3
    (100, 1, 1, 1, "unified"),  # baseline (gate G3 = 5.46×)
    (600, 1, 1, 1, "unified"),  # grande_n_pos
    (100, 3, 1, 1, "unified"),  # multi_freq
    (600, 3, 1, 1, "unified"),  # produção (regressão 0.93×)
    (100, 1, 3, 5, "vmap_real"),  # multi-TR/multi-ang com vmap_real
]


def _get_vram_mb() -> Optional[float]:
    """Retorna VRAM GPU atual em MB; None em CPU."""
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return float(out.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _measure_one(
    model_name: str,
    n_pos: int,
    nf: int,
    nTR: int,
    nAng: int,
    strategy: str,
    repeats: int,
) -> Dict:
    """Roda uma config e retorna métricas. `strategy` ∈ {bucketed,unified,vmap_real}."""
    import jax

    from geosteering_ai.simulation._jax.forward_pure import (
        clear_jit_cache,
        clear_unified_jit_cache,
    )
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    jax.config.update("jax_enable_x64", True)
    clear_jit_cache()
    clear_unified_jit_cache()

    m = get_canonical_model(model_name)
    z = np.linspace(m.min_depth - 2, m.max_depth + 2, n_pos)
    freqs = (
        np.logspace(4, np.log10(1e5), nf).tolist()
        if nf > 1
        else [20000.0]
    )
    tr_list = np.linspace(0.5, 2.0, nTR).tolist() if nTR > 1 else [1.0]
    dip_list = np.linspace(0.0, 60.0, nAng).tolist() if nAng > 1 else [0.0]

    # Configuração baseada em strategy
    if strategy == "vmap_real":
        cfg = SimulationConfig(jax_strategy="unified", jax_vmap_real=True)
    else:
        cfg = SimulationConfig(jax_strategy=strategy, jax_vmap_real=False)

    vram_before = _get_vram_mb()

    # Warmup (1 call para compilação + sync)
    t_compile_start = time.perf_counter()
    res_warm = simulate_multi_jax(
        rho_h=m.rho_h,
        rho_v=m.rho_v,
        esp=m.esp,
        positions_z=z,
        frequencies_hz=freqs,
        tr_spacings_m=tr_list,
        dip_degs=dip_list,
        cfg=cfg,
    )
    _ = float(np.abs(res_warm.H_tensor).sum())  # força completude do compute
    compile_ms = (time.perf_counter() - t_compile_start) * 1e3

    # Medição: `repeats` reps pós-warmup
    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = simulate_multi_jax(
            rho_h=m.rho_h,
            rho_v=m.rho_v,
            esp=m.esp,
            positions_z=z,
            frequencies_hz=freqs,
            tr_spacings_m=tr_list,
            dip_degs=dip_list,
            cfg=cfg,
        )
        _ = float(np.abs(res.H_tensor).sum())
        latencies.append(time.perf_counter() - t0)

    lat_ms = np.median(latencies) * 1e3
    throughput = n_pos * nf * nTR * nAng * 3600 / lat_ms * 1000  # mod/h
    vram_after = _get_vram_mb()
    vram_delta = (vram_after - vram_before) if vram_before is not None else None

    return {
        "model": model_name,
        "n_pos": n_pos,
        "nf": nf,
        "nTR": nTR,
        "nAng": nAng,
        "strategy": strategy,
        "latency_ms": round(lat_ms, 3),
        "throughput_mod_h": int(throughput),
        "compile_ms": round(compile_ms, 1),
        "vram_delta_mb": round(vram_delta, 1) if vram_delta is not None else "N/A",
        "shape": str(tuple(res.H_tensor.shape)),
    }


def _write_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    grid = {
        "short": SHORT_GRID,
        "full": FULL_GRID,
        "critical": [(*c, "critical") for c in CRITICAL_CONFIGS],  # marker only
    }[args.matrix]

    if args.matrix == "critical":
        # Critical runs: 1 model only (oklahoma_28), 5 configs, profiler ON
        grid = [c[:5] for c in grid]
        args.models = ["oklahoma_28"]

    print(
        f"Sprint 12 benchmark — matrix={args.matrix} backend={args.backend} "
        f"total={len(grid) * len(args.models)} pontos"
    )

    rows = []
    for i, (n_pos, nf, nTR, nAng, strat) in enumerate(grid):
        for model in args.models:
            try:
                metrics = _measure_one(
                    model_name=model,
                    n_pos=n_pos,
                    nf=nf,
                    nTR=nTR,
                    nAng=nAng,
                    strategy=strat,
                    repeats=args.repeats,
                )
                rows.append(metrics)
                print(
                    f"  [{i+1}/{len(grid)}] {model} n_pos={n_pos} nf={nf} "
                    f"nTR={nTR} nAng={nAng} {strat:>10}: "
                    f"{metrics['latency_ms']:6.1f} ms "
                    f"{metrics['throughput_mod_h']:>8} mod/h"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {model} {strat}: {e}")

    if rows:
        _write_csv(rows, args.out)
        print(f"\nResultados salvos em {args.out}")


if __name__ == "__main__":
    main()
