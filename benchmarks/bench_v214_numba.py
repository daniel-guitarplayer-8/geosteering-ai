#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  benchmarks/bench_v214_numba.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Benchmark Formal — Otimizações Numba v2.14                ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-01                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: numpy, argparse, time                                      ║
# ║  Referência  : docs/CHANGELOG.md v2.14 — Benchmark formal adiado v2.14   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Medir e validar ganhos de performance das otimizações Numba v2.14    ║
# ║    (Sprints 13.1, 13.2, 13.3, 13.4) em 4 cenários relevantes de produção║
# ║    (single-freq, multi-freq, multi-TR×ang, PINN), comparando contra      ║
# ║    baseline v2.12 (pré-otimizações) e v2.13 (pré-fastmath).              ║
# ║                                                                           ║
# ║  CENÁRIOS BENCHMARK                                                       ║
# ║    ┌─────────────────────────────────────────────────────────────────┐  ║
# ║    │  Cenário A — single-freq, 30k modelos (worker pool)            │  ║
# ║    │    Métrica: models/hour throughput                              │  ║
# ║    │    Baseline v2.12: ~1M mod/h                                   │  ║
# ║    │    Meta: zero regressão vs v2.12 (cache_persistent=False)       │  ║
# ║    │                                                                 │  ║
# ║    │  Cenário B — multi-freq (10 freqs), 30k modelos (pool)         │  ║
# ║    │    Métrica: modelos/hora (todas 10 freqs)                      │  ║
# ║    │    Baseline v2.12: ~200k mod/h (solo freq)                     │  ║
# ║    │    Meta v2.13 Sprint 13.1 (prange freq): ≥1.5× speedup         │  ║
# ║    │    Meta v2.14 Sprint 13.4 (fastmath): +5-10% vs v2.13          │  ║
# ║    │                                                                 │  ║
# ║    │  Cenário C — multi-TR (3) × multi-ang (5), 5k models           │  ║
# ║    │    Métrica: modelos/hora (todas TRs×ângs)                      │  ║
# ║    │    Baseline v2.13 (Python serial TR×ang): ~50k mod/h           │  ║
# ║    │    Meta v2.14 Sprint 13.3 (prange TR×ang): ≥1.3× speedup       │  ║
# ║    │    Motivação: elimina 24 transições Python→Numba               │  ║
# ║    │                                                                 │  ║
# ║    │  Cenário D — PINN (50 chamadas, cache_persistent=True)         │  ║
# ║    │    Métrica: ms/chamada (menor melhor)                          │  ║
# ║    │    Baseline v2.13 (sem cache): ~50ms/chamada                   │  ║
# ║    │    Meta v2.14 Sprint 13.2 (cache hit): ≥5× speedup             │  ║
# ║    │    Motivação: cache cross-call (precompute) reutiliza          │  ║
# ║    └─────────────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  USO CLI                                                                  ║
# ║    python benchmarks/bench_v214_numba.py --scenario A --models 30000     ║
# ║    python benchmarks/bench_v214_numba.py --scenario B --freqs 10         ║
# ║    python benchmarks/bench_v214_numba.py --scenario C                    ║
# ║    python benchmarks/bench_v214_numba.py --scenario D                    ║
# ║    python benchmarks/bench_v214_numba.py --all                           ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Benchmark formal v2.14 — 4 cenários de produção para Sprints 13.1-13.4."""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.simulation import (
        release_numba_cache,
        release_pool,
        simulate_multi,
    )
except ImportError as e:
    print(f"ERRO: Não foi possível importar simulador. {e}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Configuração Canonical
# ──────────────────────────────────────────────────────────────────────────────
def _canonical_3layer() -> Dict[str, np.ndarray]:
    """Modelo 3 camadas para todos os cenários."""
    return dict(
        rho_h=np.array([10.0, 1.0, 10.0]),
        rho_v=np.array([10.0, 1.0, 10.0]),
        esp=np.array([5.0]),
        positions_z=np.linspace(-2.0, 7.0, 30),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cenário A — Single-freq, 30k modelos (pool workers)
# ──────────────────────────────────────────────────────────────────────────────
def benchmark_scenario_a(n_models: int = 30000, n_workers: int = 4) -> Tuple[float, float]:
    """Cenário A: single-freq, N modelos via worker pool.

    Args:
        n_models: Número de modelos. Default 30k.
        n_workers: Workers pool. Default 4.

    Returns:
        Tupla (elapsed_s, throughput_mod_per_h).
    """
    m = _canonical_3layer()

    # Gera N modelos idênticos (para simplificar)
    models = [m.copy() for _ in range(n_models)]

    # Timing
    t0 = time.perf_counter()
    try:
        result = simulate_multi(
            models=models,
            positions_z=m["positions_z"],
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            n_workers=n_workers,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0

        throughput = n_models / elapsed * 3600  # modelos/hora
        return elapsed, throughput
    finally:
        release_pool()
        release_numba_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Cenário B — Multi-freq (N freqs), 30k modelos
# ──────────────────────────────────────────────────────────────────────────────
def benchmark_scenario_b(
    n_models: int = 30000, n_freqs: int = 10, n_workers: int = 4
) -> Tuple[float, float]:
    """Cenário B: multi-freq (N freqs), N modelos via worker pool.

    Args:
        n_models: Número de modelos. Default 30k.
        n_freqs: Número de frequências. Default 10.
        n_workers: Workers pool. Default 4.

    Returns:
        Tupla (elapsed_s, throughput_mod_per_h).
    """
    m = _canonical_3layer()

    # Gera N modelos idênticos
    models = [m.copy() for _ in range(n_models)]

    # Multi-freq
    freqs = np.logspace(np.log10(20000.0), np.log10(500000.0), n_freqs).tolist()

    t0 = time.perf_counter()
    try:
        result = simulate_multi(
            models=models,
            positions_z=m["positions_z"],
            frequencies_hz=freqs,
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            n_workers=n_workers,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0

        # Throughput: modelos (com todas freqs)
        throughput = n_models / elapsed * 3600
        return elapsed, throughput
    finally:
        release_pool()
        release_numba_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Cenário C — Multi-TR (3) × multi-ângulo (5), 5k modelos
# ──────────────────────────────────────────────────────────────────────────────
def benchmark_scenario_c(n_models: int = 5000, n_workers: int = 4) -> Tuple[float, float]:
    """Cenário C: multi-TR×multi-ang (3×5=15 combos), N modelos.

    Testa prange combinado TR×ângulo (Sprint 13.3).

    Args:
        n_models: Número de modelos. Default 5k.
        n_workers: Workers pool. Default 4.

    Returns:
        Tupla (elapsed_s, throughput_mod_per_h).
    """
    m = _canonical_3layer()

    # Gera N modelos
    models = [m.copy() for _ in range(n_models)]

    # Multi-TR × multi-ângulo
    tr_spacings = [0.5, 1.0, 1.5]  # nTR=3
    dip_degs = [0.0, 15.0, 30.0, 45.0, 60.0]  # nAngles=5

    t0 = time.perf_counter()
    try:
        result = simulate_multi(
            models=models,
            positions_z=m["positions_z"],
            frequencies_hz=[20000.0, 40000.0],
            tr_spacings_m=tr_spacings,
            dip_degs=dip_degs,
            n_workers=n_workers,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0

        throughput = n_models / elapsed * 3600
        return elapsed, throughput
    finally:
        release_pool()
        release_numba_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Cenário D — PINN (50 chamadas, cache_persistent=True)
# ──────────────────────────────────────────────────────────────────────────────
def benchmark_scenario_d(n_iterations: int = 50) -> Tuple[float, float]:
    """Cenário D: 50 chamadas single-model com cache_persistent=True.

    Testa cache cross-call (Sprint 13.2).

    Args:
        n_iterations: Número de chamadas. Default 50.

    Returns:
        Tupla (elapsed_s, ms_per_call).
    """
    m = _canonical_3layer()

    # Warmup (1 chamada)
    try:
        simulate_multi(
            **m,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cache_persistent=True,
        )
    except Exception:
        pass

    # Timing de N iterações com cache_persistent
    t0 = time.perf_counter()
    try:
        for _ in range(n_iterations):
            simulate_multi(
                **m,
                frequencies_hz=[20000.0],
                tr_spacings_m=[1.0],
                dip_degs=[0.0],
                cache_persistent=True,
            )
        t1 = time.perf_counter()
        elapsed = t1 - t0

        ms_per_call = (elapsed / n_iterations) * 1000
        return elapsed, ms_per_call
    finally:
        release_numba_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """CLI runner para benchmarks v2.14."""
    parser = argparse.ArgumentParser(
        description="Benchmark formal v2.14 — 4 cenários Numba (Sprints 13.1-13.4)"
    )
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C", "D"],
        help="Cenário específico (A/B/C/D) ou --all",
    )
    parser.add_argument(
        "--all", action="store_true", help="Rodar todos os 4 cenários"
    )
    parser.add_argument(
        "--models", type=int, default=30000, help="Número de modelos (Cenários A/B/C)"
    )
    parser.add_argument(
        "--freqs", type=int, default=10, help="Número de frequências (Cenário B)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Workers pool (Cenários A/B/C)"
    )
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Benchmark v2.14 — Otimizações Numba JIT (Sprints 13.1-13.4)  ║")
    print("║  Simulador Python — Geosteering AI v2.0                       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    # Definir quais cenários rodar
    scenarios = []
    if args.all:
        scenarios = ["A", "B", "C", "D"]
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        parser.print_help()
        return

    # Cabeçalho tabela
    print(
        f"{'Cenário':<10} {'Modelos':<12} {'Freqs':<8} {'Workers':<10} "
        f"{'Elapsed':<12} {'Throughput':<20}"
    )
    print("-" * 82)

    # Rodar cenários
    try:
        if "A" in scenarios:
            print(f"Cenário A (single-freq)... ", end="", flush=True)
            elapsed, throughput = benchmark_scenario_a(
                n_models=args.models, n_workers=args.workers
            )
            print(
                f"A         {args.models:<12} 1        {args.workers:<10} "
                f"{elapsed:<12.2f} {throughput:<20.1f} mod/h"
            )

        if "B" in scenarios:
            print(f"Cenário B (multi-freq)... ", end="", flush=True)
            elapsed, throughput = benchmark_scenario_b(
                n_models=args.models, n_freqs=args.freqs, n_workers=args.workers
            )
            print(
                f"B         {args.models:<12} {args.freqs:<8} {args.workers:<10} "
                f"{elapsed:<12.2f} {throughput:<20.1f} mod/h"
            )

        if "C" in scenarios:
            print(f"Cenário C (multi-TR×ang)... ", end="", flush=True)
            elapsed, throughput = benchmark_scenario_c(
                n_models=min(args.models // 6, 5000), n_workers=args.workers
            )
            print(
                f"C         {min(args.models // 6, 5000):<12} 2        {args.workers:<10} "
                f"{elapsed:<12.2f} {throughput:<20.1f} mod/h"
            )

        if "D" in scenarios:
            print(f"Cenário D (PINN cache)... ", end="", flush=True)
            elapsed, ms_per_call = benchmark_scenario_d(n_iterations=50)
            print(
                f"D         1            1        1          "
                f"{elapsed:<12.2f} {ms_per_call:<20.2f} ms/call"
            )

        print("-" * 82)
        print()
        print("✓ Benchmark concluído com sucesso.")
        print(f"  Recomendação: validar ganhos vs v2.12/v2.13 em hardware local.")

    except Exception as e:
        print(f"\n✗ ERRO durante benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
