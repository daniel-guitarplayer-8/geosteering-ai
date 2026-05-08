"""Sprint v2.22 — Benchmark FLAT prange vs legacy v2.21.

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — Sprint v2.22 Benchmark
═══════════════════════════════════════════════════════════════════════════

Mede throughput (mod/h) com e sem ``cfg.use_flat_prange`` em três
cenários canônicos do roadmap:

  • Cenário E (n_pos=600, nf=1, 1TR, 1ang):
        baseline v2.21 ≈ 122k mod/h — meta: sem regressão (≥120k)
  • Cenário B (multi-TR/Ang, nf=1):
        baseline v2.21 ≈ 303k mod/h — meta: ≥600k (≥2× ganho)
  • Cenário F (nf=4, 1TR, 1ang):
        baseline v2.21 ≈ 100k mod/h — meta: ≥1.3× (130k+)

Uso:

    python benchmarks/bench_v22_flat_prange.py --scenario E
    python benchmarks/bench_v22_flat_prange.py --scenario B
    python benchmarks/bench_v22_flat_prange.py --scenario F
    python benchmarks/bench_v22_flat_prange.py --all
    python benchmarks/bench_v22_flat_prange.py --all --runs 5

Cada cenário roda ``--runs`` vezes (default 3) e reporta mediana ± stdev.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass

import numpy as np

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.multi_forward import simulate_multi


# ──────────────────────────────────────────────────────────────────────────
# Modelo de produção (5 camadas TIV — representativo do treinamento)
# ──────────────────────────────────────────────────────────────────────────


def _model_production():
    """5 camadas TIV — representativo de geosteering em formação típica."""
    return dict(
        rho_h=np.array([5.0, 50.0, 100.0, 50.0, 5.0]),
        rho_v=np.array([5.0, 50.0, 200.0, 50.0, 5.0]),
        esp=np.array([1.5, 2.0, 3.0]),
    )


# ──────────────────────────────────────────────────────────────────────────
# Cenários do roadmap
# ──────────────────────────────────────────────────────────────────────────


SCENARIOS = {
    "E": dict(
        desc="n_pos=600, nf=1, 1TR, 1ang (producao LWD)",
        n_pos=600,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        frequencies_hz=[20000.0],
        baseline_mod_h=122_000,
        target_mod_h=120_000,
    ),
    "B": dict(
        desc="n_pos=200, nf=1, 3TR, 4ang (multi-array LWD)",
        n_pos=200,
        tr_spacings_m=[0.5, 1.0, 1.5],
        dip_degs=[0.0, 30.0, 60.0, 89.0],
        frequencies_hz=[20000.0],
        baseline_mod_h=303_000,
        target_mod_h=600_000,
    ),
    "F": dict(
        desc="n_pos=600, nf=4, 1TR, 1ang (ARC multi-freq)",
        n_pos=600,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        frequencies_hz=[2000.0, 20000.0, 100000.0, 400000.0],
        baseline_mod_h=100_000,
        target_mod_h=130_000,
    ),
}


@dataclass
class BenchResult:
    scenario: str
    use_flat: bool
    times_s: list
    median_s: float
    stdev_s: float
    mod_h: float
    n_pos: int


def run_one(scenario_id: str, use_flat: bool, runs: int = 3) -> BenchResult:
    """Roda 1 cenário com flag específico e retorna throughput."""
    if runs < 1:
        raise ValueError(f"runs must be >= 1 (got {runs})")
    cfg = SimulationConfig(use_flat_prange=use_flat, parallel=True)
    s = SCENARIOS[scenario_id]
    model = _model_production()
    positions_z = np.linspace(-2.0, 8.0, s["n_pos"])

    kw = dict(
        rho_h=model["rho_h"],
        rho_v=model["rho_v"],
        esp=model["esp"],
        positions_z=positions_z,
        tr_spacings_m=s["tr_spacings_m"],
        dip_degs=s["dip_degs"],
        frequencies_hz=s["frequencies_hz"],
        cfg=cfg,
    )

    # Warmup (JIT compilation)
    _ = simulate_multi(**kw)

    # Medição
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = simulate_multi(**kw)
        times.append(time.perf_counter() - t0)

    median_s = statistics.median(times)
    stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0

    # Throughput: 1 modelo geológico por simulação completa
    # Para benchmark de throughput LWD em geração de dataset
    mod_h = 3600.0 / median_s

    return BenchResult(
        scenario=scenario_id,
        use_flat=use_flat,
        times_s=times,
        median_s=median_s,
        stdev_s=stdev_s,
        mod_h=mod_h,
        n_pos=s["n_pos"],
    )


def print_comparison(scenario_id: str, runs: int):
    """Roda legacy + flat e imprime tabela de comparação."""
    s = SCENARIOS[scenario_id]
    print(f"\n=== Cenário {scenario_id}: {s['desc']} ===")
    print(f"Baseline v2.21: {s['baseline_mod_h']:,} mod/h | "
          f"Meta: {s['target_mod_h']:,} mod/h")

    res_legacy = run_one(scenario_id, use_flat=False, runs=runs)
    res_flat = run_one(scenario_id, use_flat=True, runs=runs)

    speedup = res_flat.mod_h / res_legacy.mod_h
    target_pass = res_flat.mod_h >= s["target_mod_h"]
    no_regression = res_flat.mod_h >= s["baseline_mod_h"] * 0.9

    print(f"  legacy v2.21:  {res_legacy.median_s*1000:7.2f} ms ± {res_legacy.stdev_s*1000:5.2f} | "
          f"{res_legacy.mod_h:>10,.0f} mod/h")
    print(f"  flat   v2.22:  {res_flat.median_s*1000:7.2f} ms ± {res_flat.stdev_s*1000:5.2f} | "
          f"{res_flat.mod_h:>10,.0f} mod/h")
    print(f"  speedup:       {speedup:.2f}×")

    status_target = "✓ MET" if target_pass else "✗ MISS"
    status_reg = "✓ OK" if no_regression else "✗ REGRESSION"
    print(f"  meta target:   {status_target} | regression: {status_reg}")
    return res_legacy, res_flat


def main():
    parser = argparse.ArgumentParser(description="Bench Sprint v2.22 FLAT prange")
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()),
        help="Cenário (E, B, F)",
    )
    parser.add_argument("--all", action="store_true", help="Rodar todos os cenários")
    parser.add_argument("--runs", type=int, default=3, help="Runs por cenário")
    args = parser.parse_args()

    if args.all:
        scenarios_to_run = list(SCENARIOS.keys())
    elif args.scenario:
        scenarios_to_run = [args.scenario]
    else:
        parser.error("Use --scenario X ou --all")

    print(f"# Sprint v2.22 FLAT prange benchmark — runs={args.runs}")
    print(f"# Hardware: {sys.platform} | Python {sys.version_info.major}.{sys.version_info.minor}")

    all_results = []
    for sid in scenarios_to_run:
        res = print_comparison(sid, args.runs)
        all_results.append((sid, res))

    print("\n=== Sumário ===")
    for sid, (legacy, flat) in all_results:
        speedup = flat.mod_h / legacy.mod_h
        print(f"  {sid}: legacy {legacy.mod_h:>10,.0f} → flat {flat.mod_h:>10,.0f} mod/h "
              f"({speedup:.2f}×)")


if __name__ == "__main__":
    main()
