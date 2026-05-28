# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  benchmarks/bench_numba_vs_jax_gpu.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Bench Sprint O2 — Numba vs JAX (complex128/complex64)     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-28 (Sprint O2)                                     ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : Python 3.13 + JAX 0.4.30+ + Numba 0.65+                   ║
# ║  Dependências: numpy, jax, numba, geosteering_ai.simulation              ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Mede throughput (mod/h) de Numba vs JAX nos Cenários A/E/H em         ║
# ║    complex128 (default) e complex64 (Sprint O2 opt-in).                  ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    PY=/path/to/python                                                     ║
# ║    $PY benchmarks/bench_numba_vs_jax_gpu.py \                            ║
# ║        --scenarios A,E,H --n-a 200 --n-e 50 --n-h 10 \                   ║
# ║        --dtype complex64                                                  ║
# ║                                                                           ║
# ║    Default --dtype complex128 (baseline). Flag --dtype complex64         ║
# ║    ativa Sprint O2 path (2× menor footprint, esperado speedup GPU).      ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Sprint O2 — _COMPLEX_DTYPE_MAP em forward_pure.py                  ║
# ║    • SimulationConfig.dtype — config.py:361                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Bench Sprint O2 — Numba vs JAX nos cenários A/E/H com dtype opt-in.

Cenários (definidos por convenção do projeto v2.30+):
    A — Single position (sanity): 1 modelo × n_pos × 1 freq × 1 TR × 1 dip
    E — Sequential positions: 1 modelo × 600 pos × 1 freq × 1 TR × 1 dip
    H — Multi-dim stress: 1 modelo × n_pos × 8 freq × 8 TR × 8 dip

Métrica: mod/h (modelos por hora) calculado como n_iters / wall_time_s × 3600.

Saída: tabela CSV em `benchmarks/results/bench_O2_<scenario>_<dtype>.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# Garante x64 antes de qualquer import JAX
os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Modelo padrão para bench (oklahoma_5 — 5 camadas, fidelidade média)
# ──────────────────────────────────────────────────────────────────────────────


def _get_default_model():
    """Retorna modelo oklahoma_5 (compromisso entre profundidade e custo)."""
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    return get_canonical_model("oklahoma_5")


# ──────────────────────────────────────────────────────────────────────────────
# Cenários A/E/H
# ──────────────────────────────────────────────────────────────────────────────


def _scenario_A(n: int):
    """Single position bench (sanity)."""
    return {
        "n_positions": n,
        "frequencies_hz": [20000.0],
        "tr_spacings_m": [1.0],
        "dip_degs": [0.0],
    }


def _scenario_E(n: int):
    """Sequential positions (600pos típico)."""
    return {
        "n_positions": n,
        "frequencies_hz": [20000.0],
        "tr_spacings_m": [1.0],
        "dip_degs": [0.0],
    }


def _scenario_H(n: int):
    """Multi-dim 8×8×8 stress test.

    NOTA: Range valido de dip é [0, 89]° (paridade Fortran). 90° e 105° foram
    substituidos por 70° e 85° (8 valores espacados sem violar range).
    """
    return {
        "n_positions": n,
        "frequencies_hz": list(np.logspace(3, np.log10(2e5), 8).tolist()),
        "tr_spacings_m": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5],
        "dip_degs": [0.0, 15.0, 30.0, 45.0, 60.0, 70.0, 80.0, 85.0],
    }


_SCENARIO_BUILDERS = {"A": _scenario_A, "E": _scenario_E, "H": _scenario_H}


# ──────────────────────────────────────────────────────────────────────────────
# Bench runners
# ──────────────────────────────────────────────────────────────────────────────


def _bench_numba(
    rho_h,
    rho_v,
    esp,
    positions_z,
    frequencies_hz,
    tr_spacings_m,
    dip_degs,
    n_iters: int,
    n_workers: int = 4,
    threads_per_worker: int = 16,
) -> float:
    """Mede tempo wall de simulate_multi (Numba) com n_iters chamadas.

    Sprint O3p2 (v2.43): default agora é `parallel=True` com 4 workers ×
    16 threads (64 logical cores Threadripper). Antes era `parallel=False`
    (single-thread), o que subdimensionava o baseline Numba e enviesava a
    comparação contra JAX.

    Args:
        n_workers: Workers Numba (process pool). Default 4.
        threads_per_worker: Threads por worker. Default 16.

    Returns:
        Tempo total em segundos.
    """
    from geosteering_ai.simulation import SimulationConfig, simulate_multi

    cfg = SimulationConfig(
        backend="numba",
        parallel=True,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
    )
    # Warmup
    _ = simulate_multi(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=frequencies_hz,
        tr_spacings_m=tr_spacings_m,
        dip_degs=dip_degs,
        cfg=cfg,
    )

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = simulate_multi(
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            positions_z=positions_z,
            frequencies_hz=frequencies_hz,
            tr_spacings_m=tr_spacings_m,
            dip_degs=dip_degs,
            cfg=cfg,
        )
    return time.perf_counter() - t0


def _bench_jax(
    rho_h,
    rho_v,
    esp,
    positions_z,
    frequencies_hz,
    tr_spacings_m,
    dip_degs,
    n_iters: int,
    dtype: str,
    strategy: str = "unified",
) -> float:
    """Mede tempo wall de simulate_multi_jax com n_iters chamadas.

    Args:
        dtype: 'complex128' (default) ou 'complex64' (Sprint O2 opt-in).
        strategy: 'bucketed' ou 'unified' (default 'unified' = 1 JIT por n,npt).

    Returns: tempo total em segundos.
    """
    from geosteering_ai.simulation import SimulationConfig, simulate_multi_jax

    cfg = SimulationConfig(
        backend="jax",
        dtype=dtype,
        jax_strategy=strategy,
        parallel=False,
    )

    # Warmup (compila JIT)
    res_warm = simulate_multi_jax(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=frequencies_hz,
        tr_spacings_m=tr_spacings_m,
        dip_degs=dip_degs,
        cfg=cfg,
    )
    _ = np.asarray(res_warm.H_tensor)  # bloqueia sync GPU→CPU

    t0 = time.perf_counter()
    for _ in range(n_iters):
        res = simulate_multi_jax(
            rho_h=rho_h,
            rho_v=rho_v,
            esp=esp,
            positions_z=positions_z,
            frequencies_hz=frequencies_hz,
            tr_spacings_m=tr_spacings_m,
            dip_degs=dip_degs,
            cfg=cfg,
        )
        _ = np.asarray(res.H_tensor)
    return time.perf_counter() - t0


def _bench_jax_batched(
    rho_h,
    rho_v,
    esp,
    positions_z,
    frequencies_hz,
    tr_spacings_m,
    dip_degs,
    n_iters: int,
    dtype: str,
    strategy: str = "unified",
    n_models: int = 4,
) -> float:
    """Bench JAX usando ``simulate_multi_jax_batched`` (vmap sobre modelos).

    Sprint O3p2 (v2.43): caminho batched amortiza ``build_static_context``
    e ``np.asarray`` GPU→CPU sync entre N modelos via ``jax.vmap``. 1 JIT
    XLA para o batch inteiro — elimina compilação por modelo.

    Args:
        rho_h, rho_v, esp: arrays 1D do modelo de referência.
            ``rho_h_batch`` = ``rho_h`` repetido ``n_models`` vezes.
        n_models: Número de modelos no batch (default 4).
        dtype: 'complex128' ou 'complex64'.
        strategy: 'unified' (default, único compatível com batched).

    Returns:
        Tempo total em segundos.

    Note:
        Cenário H (8×8×8) NÃO é compatível — pode causar OOM. Manter loop
        legacy via ``_bench_jax``.
    """
    from geosteering_ai.simulation import SimulationConfig
    from geosteering_ai.simulation._jax.multi_forward import (
        simulate_multi_jax_batched,
    )

    cfg = SimulationConfig(
        backend="jax",
        dtype=dtype,
        jax_strategy=strategy,
        parallel=False,
    )

    # Stack rho/esp ao longo do eixo n_models (mesmo modelo replicado).
    rho_h_batch = np.tile(np.asarray(rho_h)[None, :], (n_models, 1))
    rho_v_batch = np.tile(np.asarray(rho_v)[None, :], (n_models, 1))
    esp_batch = np.tile(np.asarray(esp)[None, :], (n_models, 1))

    # Warmup
    res_warm = simulate_multi_jax_batched(
        rho_h_batch=rho_h_batch,
        rho_v_batch=rho_v_batch,
        esp_batch=esp_batch,
        positions_z=positions_z,
        frequencies_hz=frequencies_hz,
        tr_spacings_m=tr_spacings_m,
        dip_degs=dip_degs,
        cfg=cfg,
    )
    # Block until ready — força sync GPU→CPU.
    try:
        import jax

        jax.block_until_ready(res_warm.H_tensor)
    except Exception:
        _ = np.asarray(res_warm.H_tensor)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        res = simulate_multi_jax_batched(
            rho_h_batch=rho_h_batch,
            rho_v_batch=rho_v_batch,
            esp_batch=esp_batch,
            positions_z=positions_z,
            frequencies_hz=frequencies_hz,
            tr_spacings_m=tr_spacings_m,
            dip_degs=dip_degs,
            cfg=cfg,
        )
        try:
            import jax

            jax.block_until_ready(res.H_tensor)
        except Exception:
            _ = np.asarray(res.H_tensor)
    return time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bench Sprint O2 — Numba vs JAX (complex128/complex64)."
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="A,E,H",
        help="Lista CSV de cenários: A, E, H (default: A,E,H).",
    )
    parser.add_argument("--n-a", type=int, default=200, help="n_pos para cenário A")
    parser.add_argument("--n-e", type=int, default=50, help="n_pos para cenário E")
    parser.add_argument("--n-h", type=int, default=10, help="n_pos para cenário H")
    parser.add_argument(
        "--n-iters", type=int, default=5, help="Iterações por bench (default 5)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="complex128",
        choices=["complex128", "complex64"],
        help="Dtype complexo do path JAX (default complex128).",
    )
    parser.add_argument(
        "--jax-strategies",
        type=str,
        default="unified",
        help="Lista CSV de strategies JAX (bucketed,unified). Default: unified.",
    )
    parser.add_argument(
        "--skip-numba",
        action="store_true",
        help="Pula bench Numba (apenas JAX).",
    )
    # Sprint O3p2 (v2.43) — flags batched + numba tuning
    parser.add_argument(
        "--batched",
        action="store_true",
        help=(
            "Sprint O3p2: bench JAX via simulate_multi_jax_batched (vmap "
            "sobre n_models). NÃO suportado em cenário H (OOM)."
        ),
    )
    parser.add_argument(
        "--n-models-batch",
        type=int,
        default=4,
        help="Sprint O3p2: n_models no batch (apenas com --batched). Default 4.",
    )
    parser.add_argument(
        "--numba-workers",
        type=int,
        default=4,
        help="Sprint O3p2: Numba n_workers (default 4 — 64 logical cores).",
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=16,
        help="Sprint O3p2: Numba threads_per_worker (default 16).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Caminho do CSV de saída. Default: benchmarks/results/bench_O2_<dtype>.csv",
    )
    args = parser.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    strategies = [s.strip() for s in args.jax_strategies.split(",") if s.strip()]

    # ── Modelo ───────────────────────────────────────────────────────────────
    m = _get_default_model()
    print(f"[bench-O2] modelo: {m.name} | n_layers={m.n_layers}")
    print(f"[bench-O2] dtype JAX: {args.dtype} | strategies: {strategies}")
    print(f"[bench-O2] iterações: {args.n_iters} por bench")
    print()

    # ── CSV output ────────────────────────────────────────────────────────────
    if args.output_csv is None:
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(exist_ok=True)
        out_csv = out_dir / f"bench_O2_{args.dtype}.csv"
    else:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = ["scenario,backend,strategy,dtype,n_pos,n_iters,wall_s,mod_per_h"]

    # ── Bench loop ────────────────────────────────────────────────────────────
    n_map = {"A": args.n_a, "E": args.n_e, "H": args.n_h}
    for scen in scenarios:
        if scen not in _SCENARIO_BUILDERS:
            print(f"[bench-O2] cenário {scen!r} desconhecido, pulando.")
            continue
        n_pos = n_map[scen]
        params = _SCENARIO_BUILDERS[scen](n_pos)
        positions_z = np.linspace(m.min_depth - 1.0, m.max_depth + 1.0, n_pos)

        common = dict(
            rho_h=m.rho_h,
            rho_v=m.rho_v,
            esp=m.esp,
            positions_z=positions_z,
            frequencies_hz=params["frequencies_hz"],
            tr_spacings_m=params["tr_spacings_m"],
            dip_degs=params["dip_degs"],
            n_iters=args.n_iters,
        )

        nf = len(params["frequencies_hz"])
        ntr = len(params["tr_spacings_m"])
        nda = len(params["dip_degs"])
        print(
            f"[bench-O2] Cenário {scen}: n_pos={n_pos}, nf={nf}, nTR={ntr}, nAng={nda}"
        )

        # Numba — Sprint O3p2: tuning explícito via CLI.
        if not args.skip_numba:
            try:
                t_n = _bench_numba(
                    n_workers=args.numba_workers,
                    threads_per_worker=args.numba_threads,
                    **common,
                )
                mod_h_n = args.n_iters / t_n * 3600.0
                print(
                    f"  [numba {args.numba_workers}w×{args.numba_threads}t] "
                    f"wall={t_n:.3f}s | mod/h={mod_h_n:.2e}"
                )
                rows.append(
                    f"{scen},numba,{args.numba_workers}w{args.numba_threads}t,"
                    f"complex128,{n_pos},{args.n_iters},"
                    f"{t_n:.6f},{mod_h_n:.6e}"
                )
            except Exception as e:
                print(f"  [numba] FALHA: {e}")

        # JAX (por estratégia) — caminho loop legacy
        for strat in strategies:
            try:
                t_j = _bench_jax(dtype=args.dtype, strategy=strat, **common)
                mod_h_j = args.n_iters / t_j * 3600.0
                print(
                    f"  [jax/{strat}] dtype={args.dtype} wall={t_j:.3f}s | "
                    f"mod/h={mod_h_j:.2e}"
                )
                rows.append(
                    f"{scen},jax,{strat},{args.dtype},{n_pos},{args.n_iters},"
                    f"{t_j:.6f},{mod_h_j:.6e}"
                )
            except Exception as e:
                print(f"  [jax/{strat}] FALHA: {e}")

        # JAX batched — Sprint O3p2 (skip Cenário H: OOM).
        if args.batched:
            if scen == "H":
                print(
                    f"  [jax/batched] cenário H ignorado (potencial OOM). "
                    f"Use loop legacy."
                )
            else:
                for strat in strategies:
                    if strat != "unified":
                        print(
                            f"  [jax/batched/{strat}] strategy != unified — "
                            f"pulando (batched requer unified)."
                        )
                        continue
                    try:
                        t_b = _bench_jax_batched(
                            dtype=args.dtype,
                            strategy=strat,
                            n_models=args.n_models_batch,
                            **common,
                        )
                        # mod/h normalizado pelo número de modelos do batch.
                        mod_h_b = args.n_iters * args.n_models_batch / t_b * 3600.0
                        print(
                            f"  [jax/batched/{strat}] n_models="
                            f"{args.n_models_batch} dtype={args.dtype} "
                            f"wall={t_b:.3f}s | mod/h={mod_h_b:.2e}"
                        )
                        rows.append(
                            f"{scen},jax_batched,{strat}_x{args.n_models_batch},"
                            f"{args.dtype},{n_pos},{args.n_iters},"
                            f"{t_b:.6f},{mod_h_b:.6e}"
                        )
                    except Exception as e:
                        print(f"  [jax/batched/{strat}] FALHA: {e}")
        print()

    # ── Escreve CSV ───────────────────────────────────────────────────────────
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[bench-O2] CSV salvo em: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
