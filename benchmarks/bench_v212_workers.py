# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  benchmarks/bench_v212_workers.py                                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto    : Geosteering AI v2.0                                         ║
# ║  Subsistema : Benchmark v2.12 — Workers Nativos (Sprint 12.4)            ║
# ║  Autor      : Daniel Leal                                                 ║
# ║  Criação    : 2026-04-30                                                 ║
# ║  Framework  : stdlib (time, argparse) + numpy                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Benchmark dos 4 modos de execução (A/B/C/D) da nova API              ║
# ║    `simulate_multi(models=[...], n_workers=N, threads_per_worker=K)`.   ║
# ║    Imprime tabela com tempo, throughput (mod/h) e speedup vs Modo A.   ║
# ║                                                                           ║
# ║  CRITÉRIO DE SUCESSO (relatório §3, 2026-04-30)                          ║
# ║    Modo D ≥ 1.3× Modo B em CPU 8-core para 30k modelos.                ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    # Bench rápido (~1 min): 200 modelos, n_pos=50.                      ║
# ║    python benchmarks/bench_v212_workers.py --n 200                     ║
# ║                                                                           ║
# ║    # Bench completo (~30 min): 30k modelos, n_pos=600.                  ║
# ║    python benchmarks/bench_v212_workers.py --n 30000 --n-pos 600       ║
# ║                                                                           ║
# ║    # Cenário customizado: subset de modos.                              ║
# ║    python benchmarks/bench_v212_workers.py --n 1000 --modes A B D      ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • docs/reports/v2.12_workers_nativos_2026-04-30.md                   ║
# ║    • geosteering_ai/simulation/_workers.py                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Benchmark dos 4 modos de execução do `simulate_multi` (v2.12).

Executa simulações sintéticas em batch para comparar tempo e throughput
de Modo A (Single), B (Multi-Thread), C (Workers serial) e D (Hybrid).

Example:
    Bench rápido para validação::

        $ python benchmarks/bench_v212_workers.py --n 200
        Modo                                Tempo (s)        Mod/h    Speedup
        ───────────────────────────────────────────────────────────────────────
        A — Single (1w × 1t)                    18.4         39130       1.00x
        B — Multi-Thread (1w × 8t)               5.2        138462       3.54x
        C — Workers (4w × 1t)                    5.8        124138       3.17x
        D — Hybrid (4w × 2t)  ★ DEFAULT          4.1        175610       4.49x

Note:
    Os números variam por hardware. Em CPU 8-core moderno, espera-se
    Modo D ≥ 1.3× Modo B (critério de sucesso v2.12). Em CPU 4-core,
    a vantagem do Modo D pode ser menor (eff_threads=1 → reduz a Modo C).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from geosteering_ai.simulation import (
    SimulationConfig,
    release_pool,
    simulate_multi,
)


# ──────────────────────────────────────────────────────────────────────────────
# Geração de modelos canônicos
# ──────────────────────────────────────────────────────────────────────────────
def make_models(n: int, seed: int = 42) -> List[Dict[str, np.ndarray]]:
    """Gera N modelos tri-camada com perturbações suaves de resistividade.

    Modelo: 3 camadas (semi-espaço/camada/semi-espaço) com
    rho_h = rho_v ∈ [5, 15] Ω·m e espessuras ∈ [3, 8] m.

    Args:
        n: Quantidade de modelos.
        seed: Semente do RNG (default 42 — reprodutível).

    Returns:
        Lista de N dicts com chaves ``rho_h``, ``rho_v``, ``esp``.
    """
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(n):
        rho_inner = float(rng.uniform(5.0, 15.0))
        esp = float(rng.uniform(3.0, 8.0))
        models.append(
            {
                "rho_h": np.array([1.0, rho_inner, 1.0], dtype=np.float64),
                "rho_v": np.array([1.0, rho_inner, 1.0], dtype=np.float64),
                "esp": np.array([esp], dtype=np.float64),
            }
        )
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Configuração dos 4 modos
# ──────────────────────────────────────────────────────────────────────────────
MODES_DEFAULT: List[Tuple[str, str, int, int]] = [
    ("A", "A — Single (1w × 1t)", 1, 1),
    ("B", "B — Multi-Thread (1w × Nt)", 1, -1),  # -1 = cpu_count
    ("C", "C — Workers (4w × 1t)", 4, 1),
    ("D", "D — Hybrid (4w × Kt)  ★ DEFAULT", 4, -1),  # -1 = auto
]


def _resolve_threads(n_workers: int, n_threads_hint: int) -> int:
    """Resolve `n_threads_hint == -1` para `cpu_count // n_workers`."""
    if n_threads_hint > 0:
        return n_threads_hint
    cpu = os.cpu_count() or 1
    return max(1, cpu // max(1, n_workers))


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────
def run_benchmark(
    models: List[Dict[str, np.ndarray]],
    positions_z: np.ndarray,
    selected: List[str],
) -> List[Tuple[str, str, int, int, float, float]]:
    """Executa os 4 modos e retorna lista de resultados.

    Returns:
        Lista de tuplas ``(mode_id, mode_name, n_workers, n_threads,
        elapsed_s, mod_per_h)``.
    """
    cfg = SimulationConfig(backend="numba", parallel=True)
    results = []
    for mode_id, mode_name, n_w, n_t_hint in MODES_DEFAULT:
        if mode_id not in selected:
            continue
        n_t = _resolve_threads(n_w, n_t_hint)
        t0 = time.perf_counter()
        res = simulate_multi(
            models=models,
            positions_z=positions_z,
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            frequencies_hz=[20000.0],
            cfg=cfg,
            n_workers=n_w,
            threads_per_worker=n_t,
        )
        elapsed = time.perf_counter() - t0
        n_models = res.H_stack.shape[0]
        mod_per_h = n_models / max(elapsed, 1e-9) * 3600.0
        results.append(
            (mode_id, mode_name, res.n_workers, res.n_threads, elapsed, mod_per_h)
        )
        # Cleanup do pool entre modos para garantir start "frio" justo.
        # (em cenário real o pool é reutilizado; aqui medimos custo total
        # incluindo spawn).
        release_pool()
    return results


def print_table(
    results: List[Tuple[str, str, int, int, float, float]],
) -> None:
    """Imprime tabela formatada com Modo / Tempo / Mod/h / Speedup."""
    print()
    print(f"{'Modo':<40} {'Tempo (s)':>12} {'Mod/h':>14} {'Speedup':>10}")
    print("─" * 78)
    if not results:
        print("Nenhum modo selecionado.")
        return
    baseline = results[0][4]  # tempo do primeiro modo (geralmente A)
    for _, name, _n_w, _n_t, elapsed, mod_per_h in results:
        speedup = baseline / max(elapsed, 1e-9)
        print(f"{name:<40} {elapsed:>12.2f} {mod_per_h:>14.0f} {speedup:>9.2f}x")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark v2.12 — Workers Nativos (4 modos A/B/C/D)"
    )
    parser.add_argument(
        "--n", type=int, default=200, help="Número de modelos (default 200)"
    )
    parser.add_argument(
        "--n-pos",
        type=int,
        default=50,
        help="Número de posições por modelo (default 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semente RNG (default 42)"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
        help="Modos a rodar (default todos)",
    )
    args = parser.parse_args()

    print(f"Geosteering AI — Benchmark v2.12 (Workers Nativos)")
    print(f"  Modelos: {args.n}")
    print(f"  Posições por modelo: {args.n_pos}")
    print(f"  CPU count: {os.cpu_count()}")
    print(f"  Modos selecionados: {', '.join(args.modes)}")

    models = make_models(args.n, seed=args.seed)
    positions_z = np.linspace(-2.0, 7.0, args.n_pos)

    results = run_benchmark(models, positions_z, args.modes)
    print_table(results)

    # Critério de sucesso v2.12: Modo D ≥ 1.3× Modo B.
    by_mode = {r[0]: r[4] for r in results}
    if "B" in by_mode and "D" in by_mode:
        ratio = by_mode["B"] / max(by_mode["D"], 1e-9)
        criteria_met = ratio >= 1.3
        print(f"Critério v2.12: Modo D vs B → {ratio:.2f}x  ", end="")
        print("✅ ATINGIDO" if criteria_met else "⚠️  ABAIXO DO ALVO 1.3x")
        return 0 if criteria_met else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
