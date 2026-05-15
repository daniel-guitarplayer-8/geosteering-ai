# -*- coding: utf-8 -*-
"""Sprint v2.37 F4 — Benchmark cross-platform reproduzível.

Captura fingerprint do hardware/ambiente + mede 3 cenários canônicos
(E, B, F) em 4 configurações de paralelismo (cfg.parallel × use_flat_prange
× heurística v2.37 × tile/block auto v2.37). Emite JSON estruturado para
agregação posterior em `docs/PERFORMANCE_BASELINE.md` § "Cross-Platform".

Uso:
    python benchmarks/bench_v237_cross_platform.py [--runs 5] [--out FILE]

O JSON resultante (em `benchmarks/results/` por default) inclui:
- Fingerprint: CPU brand, cores phys/log, cache sizes, RAM, OS, Python,
  Numba, NumPy versions.
- Resultados por cenário × config: mediana, p25, p75, stdev, mod_h.
- Hash determinístico do código relevante (forward.py + multi_forward.py +
  config.py) para vincular resultados a uma versão exata.

Cobertura mínima planejada (após múltiplas máquinas):
- Intel Core i7/i9 (HT)         — esta máquina baseline
- AMD Ryzen/EPYC (SMT)          — pendente
- Apple Silicon M1/M2/M3        — pendente
- ARM Neoverse (cloud)          — pendente

Sprint v2.37 F4 entrega a INFRA + 1 medição (Intel i9-9980HK). Outras
arquiteturas dependem de acesso a hardware externo (TODO Sprint v2.38).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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
    "E (single-combo)": dict(
        n_pos=600,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        frequencies_hz=[20000.0],
        n_eff_combos=1,
    ),
    "F (4 combos multi-freq)": dict(
        n_pos=600,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        frequencies_hz=[2000.0, 20000.0, 100000.0, 400000.0],
        n_eff_combos=4,
    ),
    "B (12 combos multi-array)": dict(
        n_pos=200,
        tr_spacings_m=[0.5, 1.0, 1.5],
        dip_degs=[0.0, 30.0, 60.0, 89.0],
        frequencies_hz=[20000.0],
        n_eff_combos=12,
    ),
}


_CONFIGS = {
    "v2.21 (non-FLAT, no tile)": dict(
        use_flat_prange=False,
        flat_prange_min_combos=1,
        use_tiled_positions=False,
    ),
    "v2.22.4 (FLAT, no tile)": dict(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=False,
    ),
    "v2.37 heuristic (default)": dict(
        use_flat_prange=True,
        flat_prange_min_combos=2,
        use_tiled_positions=False,
    ),
    "v2.37 heuristic + auto-tile": dict(
        use_flat_prange=True,
        flat_prange_min_combos=2,
        use_tiled_positions=True,
        tile_size_auto=True,
    ),
}


def _hardware_fingerprint() -> dict:
    """Captura identificação reproduzível do hardware/ambiente."""
    import subprocess

    fp = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "python": platform.python_version(),
    }

    # macOS: sysctl tem detalhes de CPU mais ricos
    if platform.system() == "Darwin":
        try:
            keys = [
                "machdep.cpu.brand_string",
                "hw.physicalcpu",
                "hw.logicalcpu",
                "hw.l1dcachesize",
                "hw.l2cachesize",
                "hw.l3cachesize",
                "hw.memsize",
            ]
            for k in keys:
                out = subprocess.check_output(
                    ["sysctl", "-n", k], text=True, timeout=5
                ).strip()
                fp[k.replace(".", "_")] = out
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Linux: /proc/cpuinfo
    elif platform.system() == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            for line in cpuinfo.splitlines():
                if "model name" in line:
                    fp["model_name"] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass

    # psutil para portabilidade
    try:
        import psutil

        fp["cores_physical"] = psutil.cpu_count(logical=False)
        fp["cores_logical"] = psutil.cpu_count(logical=True)
        fp["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    # Versões de bibliotecas críticas
    try:
        import numba

        fp["numba"] = numba.__version__
    except ImportError:
        fp["numba"] = None
    import numpy

    fp["numpy"] = numpy.__version__

    return fp


def _code_hash() -> str:
    """SHA256 dos módulos críticos para reproduzibilidade."""
    h = hashlib.sha256()
    root = Path(__file__).parent.parent / "geosteering_ai" / "simulation"
    for fname in ("forward.py", "multi_forward.py", "config.py"):
        p = root / fname
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _run_one(scenario: dict, cfg: SimulationConfig) -> float:
    positions_z = np.linspace(-2.0, 8.0, scenario["n_pos"]).astype(np.float64)
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


def _bench(scenario_name: str, scenario: dict, runs: int) -> dict:
    """Bench 1 cenário × 4 configs."""
    out = {}
    for cfg_name, cfg_kwargs in _CONFIGS.items():
        cfg = SimulationConfig(parallel=True, **cfg_kwargs)
        _run_one(scenario, cfg)  # warmup
        times = [_run_one(scenario, cfg) for _ in range(runs)]
        med = statistics.median(times)
        out[cfg_name] = {
            "median_s": med,
            "p25_s": statistics.quantiles(times, n=4)[0] if len(times) >= 4 else med,
            "p75_s": statistics.quantiles(times, n=4)[-1] if len(times) >= 4 else med,
            "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
            "mod_h": 3600.0 / med if med > 0 else 0,
            "runs": runs,
        }
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    fp = _hardware_fingerprint()
    code = _code_hash()

    print("=== Sprint v2.37 F4 — Cross-Platform Benchmark ===")
    print(f"\nHardware fingerprint:")
    for k, v in fp.items():
        print(f"  {k}: {v}")
    print(f"\nCode hash (forward+multi_forward+config): {code}")
    print(f"\nRunning {len(_SCENARIOS)} scenarios × {len(_CONFIGS)} configs × {args.runs} runs...")

    results = {
        "fingerprint": fp,
        "code_hash": code,
        "scenarios": {},
    }

    print(
        f"\n{'Cenário':<28} {'config':<32} {'mod/h':>12} {'p25':>10} {'p75':>10}"
    )
    print("─" * 100)

    for sc_name, sc in _SCENARIOS.items():
        sc_results = _bench(sc_name, sc, args.runs)
        results["scenarios"][sc_name] = sc_results
        for cfg_name, r in sc_results.items():
            p25_mod_h = 3600.0 / r["p25_s"] if r["p25_s"] > 0 else 0
            p75_mod_h = 3600.0 / r["p75_s"] if r["p75_s"] > 0 else 0
            # p25/p75 invertem em mod/h: tempo p75 = throughput p25
            print(
                f"{sc_name:<28} {cfg_name:<32} {r['mod_h']:>12,.0f} "
                f"{p75_mod_h:>10,.0f} {p25_mod_h:>10,.0f}"
            )
        print("─" * 100)

    # Persistir JSON
    out_path = args.out
    if out_path is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        machine_tag = fp.get("machdep_cpu_brand_string", platform.machine())
        machine_tag = machine_tag.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")[:40]
        out_path = str(
            results_dir
            / f"v237_cross_platform_{machine_tag}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
        )
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n✓ JSON gravado em: {out_path}")

    # Summary: comparação v2.37 default vs v2.22.4 default (recovery check)
    print("\n=== Summary v2.37 vs v2.22.4 ===")
    for sc_name, sc_results in results["scenarios"].items():
        v22 = sc_results.get("v2.22.4 (FLAT, no tile)", {}).get("mod_h", 0)
        v37 = sc_results.get("v2.37 heuristic (default)", {}).get("mod_h", 0)
        v37_tile = sc_results.get("v2.37 heuristic + auto-tile", {}).get("mod_h", 0)
        if v22 > 0:
            delta = (v37 / v22 - 1.0) * 100.0
            delta_tile = (v37_tile / v22 - 1.0) * 100.0
            print(
                f"  {sc_name:<28} v2.22={v22:,.0f}  v2.37={v37:,.0f}  ({delta:+.2f}%)  "
                f"+tile={v37_tile:,.0f}  ({delta_tile:+.2f}%)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
