# -*- coding: utf-8 -*-
"""Sprint v2.36 O2 — Benchmark tile/block vs legacy.

Mede throughput (mod/h) de ``_simulate_positions_njit_cached_tiled`` para
``tile_size ∈ {1, 2, 4, 8, 16}`` vs versão não-tiled
(``use_tiled_positions=False``) no caminho single-TR legado (`use_flat_prange=False`).

Critério de promoção (Sprint v2.36 plano):
- Paridade bit-exata vs legacy (validada em test_simulation_v236_tile_block.py)
- Ganho ≥ +5% em ao menos um tile_size, em ao menos um perfil → promover
- Caso contrário, REVERT da feature (mantém implementação isolada por opt-in)

Uso:
    python benchmarks/bench_v236_tile_block.py [--runs 5] [--n-pos 200]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Optional

import numpy as np

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.forward import simulate


_PROFILES = {
    "small (3 cam, 100 pos)": {
        "rho_h": np.array([10.0, 50.0, 10.0]),
        "rho_v": np.array([10.0, 50.0, 10.0]),
        "esp": np.array([5.0]),
        "n_pos": 100,
    },
    "medium (5 cam, 200 pos)": {
        "rho_h": np.array([10.0, 30.0, 100.0, 30.0, 10.0]),
        "rho_v": np.array([15.0, 50.0, 100.0, 50.0, 15.0]),
        "esp": np.array([2.0, 3.0, 2.0]),
        "n_pos": 200,
    },
    "large (5 cam, 600 pos)": {
        "rho_h": np.array([10.0, 30.0, 100.0, 30.0, 10.0]),
        "rho_v": np.array([15.0, 50.0, 100.0, 50.0, 15.0]),
        "esp": np.array([2.0, 3.0, 2.0]),
        "n_pos": 600,
    },
}


def _run_one(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    positions_z: np.ndarray,
    use_tiled: bool,
    tile_size: int,
) -> float:
    """Roda 1 simulação e retorna elapsed (s)."""
    cfg = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=use_tiled,
        tile_size=tile_size,
        parallel=True,
    )
    t0 = time.perf_counter()
    simulate(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        cfg=cfg,
    )
    return time.perf_counter() - t0


def _bench(profile_name: str, profile: dict, runs: int, tile_size: int) -> dict:
    rho_h = profile["rho_h"]
    rho_v = profile["rho_v"]
    esp = profile["esp"]
    n_pos = profile["n_pos"]
    positions_z = np.linspace(-5.0, 5.0, n_pos).astype(np.float64)

    # Warmup (1 run de cada para JIT compilar)
    _run_one(rho_h, rho_v, esp, positions_z, False, 1)
    _run_one(rho_h, rho_v, esp, positions_z, True, tile_size)

    legacy_times = [
        _run_one(rho_h, rho_v, esp, positions_z, False, 1) for _ in range(runs)
    ]
    tiled_times = [
        _run_one(rho_h, rho_v, esp, positions_z, True, tile_size)
        for _ in range(runs)
    ]

    legacy_med = statistics.median(legacy_times)
    tiled_med = statistics.median(tiled_times)
    speedup = legacy_med / tiled_med if tiled_med > 0 else float("inf")

    return {
        "profile": profile_name,
        "tile_size": tile_size,
        "n_pos": n_pos,
        "legacy_med_s": legacy_med,
        "tiled_med_s": tiled_med,
        "legacy_mod_h": 3600.0 / legacy_med if legacy_med > 0 else 0,
        "tiled_mod_h": 3600.0 / tiled_med if tiled_med > 0 else 0,
        "speedup": speedup,
        "delta_pct": (speedup - 1.0) * 100.0,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--tile-sizes",
        type=str,
        default="2,4,8,16",
        help="CSV de tile_sizes a testar",
    )
    args = parser.parse_args(argv)

    tile_sizes = [int(x) for x in args.tile_sizes.split(",")]

    print(f"\n=== Sprint v2.36 O2 — Tile/Block Benchmark ===")
    print(f"Runs por config: {args.runs}")
    print(f"Tile sizes: {tile_sizes}\n")
    print(
        f"{'Perfil':<28} {'tile':>5} {'legacy mod/h':>15} {'tiled mod/h':>15} "
        f"{'delta %':>10}"
    )
    print("─" * 80)

    results = []
    for prof_name, prof in _PROFILES.items():
        for ts in tile_sizes:
            r = _bench(prof_name, prof, args.runs, ts)
            results.append(r)
            print(
                f"{r['profile']:<28} {r['tile_size']:>5} "
                f"{r['legacy_mod_h']:>15,.0f} {r['tiled_mod_h']:>15,.0f} "
                f"{r['delta_pct']:>+9.2f}%"
            )

    # Resumo: ganho máximo
    max_gain = max(results, key=lambda r: r["delta_pct"])
    print("\n" + "─" * 80)
    print(
        f"Ganho máximo: {max_gain['profile']} @ tile_size={max_gain['tile_size']} "
        f"→ {max_gain['delta_pct']:+.2f}%"
    )
    if max_gain["delta_pct"] >= 5.0:
        print("✓ Critério de promoção (≥+5%) atingido")
    else:
        print("✗ Critério de promoção NÃO atingido — manter opt-in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
