# -*- coding: utf-8 -*-
"""Sprint v2.37 F1 — Benchmark heurística FLAT/não-FLAT.

Compara:
- **with-heuristic** (default v2.37): `use_flat_prange=True`,
  `flat_prange_min_combos=2`. Single-combo (Cenário E) força não-FLAT.
- **force-flat** (v2.22.4–v2.36 default): `use_flat_prange=True`,
  `flat_prange_min_combos=1`. Sempre FLAT.
- **force-non-flat** (v2.21 legacy): `use_flat_prange=False`.

Objetivo: validar que a heurística (default v2.37) iguala o caminho
não-FLAT em single-combo, recuperando o ~+15% perdido em v2.22.4+.

Uso:
    python benchmarks/bench_v237_flat_heuristic.py [--runs 5]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Optional

import numpy as np

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.multi_forward import simulate_multi


_MODEL = {
    "rho_h": np.array([5.0, 50.0, 100.0, 50.0, 5.0]),
    "rho_v": np.array([5.0, 50.0, 200.0, 50.0, 5.0]),
    "esp": np.array([1.5, 2.0, 3.0]),
}


_SCENARIOS = {
    "E (1 combo)": dict(
        n_pos=600,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        frequencies_hz=[20000.0],
        n_eff_combos=1,
    ),
    "F (4 combos, multi-freq)": dict(
        n_pos=600,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        frequencies_hz=[2000.0, 20000.0, 100000.0, 400000.0],
        n_eff_combos=4,
    ),
    "B (12 combos, multi-array)": dict(
        n_pos=200,
        tr_spacings_m=[0.5, 1.0, 1.5],
        dip_degs=[0.0, 30.0, 60.0, 89.0],
        frequencies_hz=[20000.0],
        n_eff_combos=12,
    ),
}


def _run_one(scenario: dict, cfg: SimulationConfig) -> float:
    positions_z = np.linspace(-2.0, 8.0, scenario["n_pos"])
    t0 = time.perf_counter()
    simulate_multi(
        rho_h=_MODEL["rho_h"],
        rho_v=_MODEL["rho_v"],
        esp=_MODEL["esp"],
        positions_z=positions_z,
        frequencies_hz=scenario["frequencies_hz"],
        tr_spacings_m=scenario["tr_spacings_m"],
        dip_degs=scenario["dip_degs"],
        cfg=cfg,
    )
    return time.perf_counter() - t0


def _bench_scenario(name: str, scenario: dict, runs: int) -> dict:
    configs = {
        "heuristic (v2.37)": SimulationConfig(
            use_flat_prange=True, flat_prange_min_combos=2, parallel=True
        ),
        "force-flat (v2.22.4)": SimulationConfig(
            use_flat_prange=True, flat_prange_min_combos=1, parallel=True
        ),
        "force-non-flat (v2.21)": SimulationConfig(
            use_flat_prange=False, parallel=True
        ),
    }

    # Warmup (1 run de cada para JIT compilar)
    for cfg in configs.values():
        _run_one(scenario, cfg)

    results = {}
    for label, cfg in configs.items():
        times = [_run_one(scenario, cfg) for _ in range(runs)]
        med = statistics.median(times)
        results[label] = {
            "median_s": med,
            "mod_h": 3600.0 / med if med > 0 else 0,
        }

    return {
        "scenario": name,
        "n_eff_combos": scenario["n_eff_combos"],
        "configs": results,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)

    print("=== Sprint v2.37 F1 — Heurística FLAT/não-FLAT ===")
    print(f"Runs por config: {args.runs}\n")
    print(
        f"{'Cenário':<28} {'combos':>7} "
        f"{'heuristic mod/h':>17} {'force-flat':>13} {'non-flat':>11} "
        f"{'Δ heur vs flat':>15}"
    )
    print("─" * 100)

    for name, sc in _SCENARIOS.items():
        r = _bench_scenario(name, sc, args.runs)
        heuristic = r["configs"]["heuristic (v2.37)"]["mod_h"]
        force_flat = r["configs"]["force-flat (v2.22.4)"]["mod_h"]
        non_flat = r["configs"]["force-non-flat (v2.21)"]["mod_h"]
        delta_pct = (heuristic / force_flat - 1.0) * 100.0 if force_flat else 0
        print(
            f"{name:<28} {r['n_eff_combos']:>7} "
            f"{heuristic:>17,.0f} {force_flat:>13,.0f} {non_flat:>11,.0f} "
            f"{delta_pct:>+13.2f}%"
        )

    print("\nInterpretação:")
    print("  - Cenário E: heuristic ≈ non-flat > force-flat (recupera regressão)")
    print("  - Cenário F/B: heuristic ≈ force-flat (heurística NÃO atua)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
