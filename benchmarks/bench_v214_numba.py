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
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np


def _configure_threads(threads_per_worker: int) -> None:
    """Configura variáveis de ambiente de threading antes do import Numba.

    Define ``OMP_NUM_THREADS`` e ``NUMBA_NUM_THREADS`` para controlar o
    paralelismo intra-worker. O total de threads em uso é
    ``n_workers × threads_per_worker``; ajuste em função do número real de
    cores físicos para evitar oversubscription.

    Args:
        threads_per_worker: Threads por worker (1 para Cenário D, 2 default
            para A/B/C). Range válido: ``[1, 16]``.

    Note:
        Esta função deve ser chamada ANTES de qualquer ``import numba`` ou
        ``simulate_multi`` para que Numba leia o env var corretamente. O
        v2.15 chama isto no topo de ``main()``, antes do worker pool.
    """
    if not 1 <= threads_per_worker <= 16:
        raise ValueError(
            f"threads_per_worker={threads_per_worker} fora de [1, 16]"
        )
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["NUMBA_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)

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
# Número de pontos de medição (n_positions) — exposto pelo CLI via
# ``--n-positions``. Default 30 (microbenchmark rápido para CI). Em produção
# (GUI do Simulation Manager), o valor típico é 600 (sequência LWD), e o
# Cenário E permite reproduzir esse caso real.
_N_POSITIONS: int = 30


def _canonical_3layer(n_positions: int | None = None) -> Dict[str, np.ndarray]:
    """Modelo 3 camadas para todos os cenários.

    Args:
        n_positions: Sobrescreve ``_N_POSITIONS`` se informado. Default
            ``None`` (usa valor global setado pelo CLI).

    Returns:
        Dict com `rho_h`, `rho_v`, `esp`, `positions_z`.
    """
    npos = int(n_positions) if n_positions is not None else _N_POSITIONS
    return dict(
        rho_h=np.array([10.0, 1.0, 10.0]),
        rho_v=np.array([10.0, 1.0, 10.0]),
        esp=np.array([5.0]),
        positions_z=np.linspace(-2.0, 7.0, npos),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Threads por worker — exposto pelo CLI, propagado a simulate_multi explicitamente
# ──────────────────────────────────────────────────────────────────────────────
# Usar variável global lida pelo main() é mais simples que passar args através
# das 4 funções benchmark_scenario_*. Default 2 (4 workers × 2 = 8 = cores físicos
# em hardware de 8 cores). Sobrescrita por --threads-per-worker no CLI.
_THREADS_PER_WORKER: int = 2


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
            threads_per_worker=_THREADS_PER_WORKER,
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
            threads_per_worker=_THREADS_PER_WORKER,
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
            threads_per_worker=_THREADS_PER_WORKER,
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
# Cenário E — Production scale (n_positions=600, single-freq)
# ──────────────────────────────────────────────────────────────────────────────
def benchmark_scenario_e(
    n_models: int = 300, n_workers: int = 4, n_positions: int = 600
) -> Tuple[float, float]:
    """Cenário E: production scale, n_positions=600 (típico LWD).

    Reproduz a configuração real da GUI do Simulation Manager (sequência
    LWD com ~600 pontos por modelo) para detectar regressões que não
    aparecem no microbenchmark Cenário A (30 pts).

    Histórico v2.16: adicionado para validar fix da regressão de threading
    masking (commits 0f92035 + e1c8864 da v2.15) que reduziu produção da
    GUI de 200k mod/h → 25–38k mod/h. Pós-fix esperado: ≥150k mod/h.

    Args:
        n_models: Número de modelos. Default 300 (20× menos que Cenário A
            porque cada modelo é 20× mais lento com 600 pts).
        n_workers: Workers pool. Default 4.
        n_positions: Pontos de medição por modelo. Default 600 (production).

    Returns:
        Tupla (elapsed_s, throughput_mod_per_h).
    """
    m = _canonical_3layer(n_positions=n_positions)

    # Gera N modelos idênticos
    models = [m.copy() for _ in range(n_models)]

    t0 = time.perf_counter()
    try:
        result = simulate_multi(
            models=models,
            positions_z=m["positions_z"],
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            threads_per_worker=_THREADS_PER_WORKER,
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
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """CLI runner para benchmarks v2.14+ (Sprint 19.2 — defaults CPU-aware)."""
    # Sprint 19.2 (v2.19): defaults automáticos via topologia de CPU.
    # Em CPU 8C/16T HT, recomenda 4w × 2t = 8 threads (cores físicos),
    # alinhado com a GUI v2.17 e evitando oversubscrição em hot paths
    # com saturação de ALU (típico para Numba @njit numérico).
    try:
        from geosteering_ai.simulation import (
            detect_cpu_topology,
            recommend_default_parallelism,
        )

        _default_workers, _default_threads = recommend_default_parallelism()
        _, _physical_cores, _has_ht = detect_cpu_topology()
    except Exception:
        # Fallback conservador caso a função ainda não esteja exposta.
        _default_workers = 4
        _default_threads = 2
        _physical_cores = os.cpu_count() or 8
        _has_ht = False

    parser = argparse.ArgumentParser(
        description="Benchmark formal v2.14+ — 5 cenários Numba (Sprints 13.1-13.4 + 19.2)"
    )
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C", "D", "E"],
        help="Cenário específico (A/B/C/D/E) ou --all",
    )
    parser.add_argument(
        "--all", action="store_true", help="Rodar todos os 5 cenários (A–E)"
    )
    parser.add_argument(
        "--models",
        type=int,
        default=30000,
        help="Número de modelos (Cenários A/B/C/E). Cenário E usa min(--models, 300) "
        "por padrão para manter tempo razoável com 600 pts.",
    )
    parser.add_argument(
        "--freqs", type=int, default=10, help="Número de frequências (Cenário B)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers,
        help=f"Workers pool (default auto={_default_workers}; "
        f"baseado em {_physical_cores} cores físicos)",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=_default_threads,
        help=f"Threads intra-worker (default auto={_default_threads}). "
        f"Total = workers × threads-per-worker. Recomendação CPU-aware "
        f"(Sprint 19.2 v2.19) evita oversubscrição em CPUs com HT/SMT.",
    )
    parser.add_argument(
        "--n-positions",
        type=int,
        default=30,
        help="Pontos de medição por modelo (default 30 microbench, 600 production). "
        "Aplicável a Cenários A/B/C/D. Cenário E sempre usa 600 (override).",
    )
    args = parser.parse_args()

    # Sprint 19.2 (v2.19): warning de oversubscrição.
    # Quando workers × threads > cores_físicos, threads competem pelas mesmas
    # ALUs e o throughput cai (até 2-3× em loops numéricos pesados como
    # _simulate_positions_njit). Em CPUs com HT/SMT (M1/M2, Ryzen, Core i),
    # logical_cores é o dobro de physical_cores mas hyperthreads não ajudam
    # cargas saturadas em ALU como Numba njit numérico denso.
    _total_threads = args.workers * args.threads_per_worker
    if _total_threads > _physical_cores:
        ht_label = " (HT/SMT ativo)" if _has_ht else ""
        print(
            f"  ATENCAO: {args.workers}w x {args.threads_per_worker}t = "
            f"{_total_threads} threads em {_physical_cores} cores fisicos"
            f"{ht_label}. Oversubscricao pode degradar performance em "
            f"2-3x. Recomendado: {_default_workers}w x {_default_threads}t "
            f"= {_default_workers * _default_threads} threads."
        )
        print()

    # ── Configurar threading ANTES de qualquer trabalho Numba ────────────────
    # Setar env vars OMP_NUM_THREADS / NUMBA_NUM_THREADS / MKL_NUM_THREADS
    # antes de instanciar worker pool, garantindo que Numba JIT compile com
    # o número correto de threads. Sem isto, Numba detecta cpu_count() e
    # cria oversubscription quando combinado com worker pool.
    _configure_threads(args.threads_per_worker)

    # ── Propagar para benchmark_scenario_*  ─────────────────────────────────
    # Os 3 cenários A/B/C passam ``threads_per_worker=_THREADS_PER_WORKER``
    # explicitamente em ``simulate_multi(...)``, evitando o erro
    # ``Cannot set NUMBA_NUM_THREADS to a different value once threads have
    # been launched`` que ocorre quando ``run_batch`` calcula ``eff_threads``
    # baseado em ``cpu_count`` heurística e diverge do env var.
    global _THREADS_PER_WORKER, _N_POSITIONS
    _THREADS_PER_WORKER = int(args.threads_per_worker)
    _N_POSITIONS = int(args.n_positions)

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Benchmark v2.14 — Otimizações Numba JIT (Sprints 13.1-13.4)  ║")
    print("║  Simulador Python — Geosteering AI v2.0                       ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(
        f"  Threading: workers={args.workers} × threads/worker="
        f"{args.threads_per_worker} → {args.workers * args.threads_per_worker} "
        f"threads totais"
    )
    print()

    # Definir quais cenários rodar
    scenarios = []
    if args.all:
        scenarios = ["A", "B", "C", "D", "E"]
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

        if "E" in scenarios:
            # Cenário E: production scale (600 pts). Usa min(args.models, 300)
            # para manter tempo razoável; pode ser sobrescrito via --models.
            n_models_e = min(args.models, 300) if args.all else args.models
            n_pos_e = 600  # override fixo do Cenário E
            print(f"Cenário E (production 600 pts)... ", end="", flush=True)
            elapsed, throughput = benchmark_scenario_e(
                n_models=n_models_e, n_workers=args.workers, n_positions=n_pos_e
            )
            print(
                f"E         {n_models_e:<12} 1        {args.workers:<10} "
                f"{elapsed:<12.2f} {throughput:<20.1f} mod/h "
                f"({n_pos_e} pts)"
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
