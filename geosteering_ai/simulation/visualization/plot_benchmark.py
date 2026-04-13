# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/visualization/plot_benchmark.py               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Plot de Benchmarks (Sprint 2.8)        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : matplotlib                                                 ║
# ║  Dependências: matplotlib, numpy                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Gráficos comparativos de throughput (mod/h) entre diferentes          ║
# ║    configurações (backend, parallel, filtro) e entre perfis de           ║
# ║    benchmark (small/medium/large). Consome objetos :class:`BenchmarkResult`║
# ║    do módulo `benchmarks.bench_forward`.                                  ║
# ║                                                                           ║
# ║  LAYOUT                                                                   ║
# ║    • plot_benchmark_comparison:                                          ║
# ║      - Painel 1: bar chart throughput (mod/h) por configuração          ║
# ║      - Painel 2: linha de speedup vs baseline Fortran                   ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • geosteering_ai/simulation/benchmarks/bench_forward.py               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Plotagem de resultados de benchmarks do simulador Python.

Fornece gráficos de comparação entre configurações (backend, threads,
filtro Hankel) e entre perfis (small/medium/large). Integra com os
objetos :class:`BenchmarkResult` produzidos por `bench_forward.py`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:  # pragma: no cover — matplotlib é dep opcional
    _HAS_MPL = False
    plt = None  # type: ignore[assignment]
    Figure = object  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    # Import apenas para type hints (evita ciclo em runtime).
    from geosteering_ai.simulation.benchmarks.bench_forward import BenchmarkResult

logger = logging.getLogger(__name__)

# Baseline Fortran (mod/h) — derivada dos benchmarks Sprint 2.7 e da
# documentação do simulador Fortran v10.0 (i9-9980HK 8 threads OpenMP).
_FORTRAN_BASELINE_MOD_H: float = 58_856.0


# ──────────────────────────────────────────────────────────────────────────────
# plot_benchmark_comparison — figura principal
# ──────────────────────────────────────────────────────────────────────────────


def plot_benchmark_comparison(
    results: Iterable["BenchmarkResult"],
    *,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (14.0, 6.0),
    show_fortran_baseline: bool = True,
) -> Figure:
    """Gera figura comparativa de throughput entre benchmarks.

    A figura tem 2 painéis lado-a-lado:
      1. **Throughput bar chart**: throughput em mod/h por benchmark,
         com barra de erro baseada em ``time_std_seconds`` (se
         disponível).
      2. **Speedup vs Fortran**: linha com marcadores mostrando o
         percentual do baseline Fortran atingido por cada benchmark.

    Args:
        results: Iterável de :class:`BenchmarkResult`. Cada resultado
            vira uma barra/ponto nos painéis, identificado por
            ``result.profile_name``.
        title: Título da figura. Se None, usa
            ``"Benchmark Forward — Simulador Python"``.
        figsize: Tamanho da figura em polegadas. Default: (14, 6).
        show_fortran_baseline: Se True, desenha linha horizontal
            vermelha tracejada no patamar do Fortran baseline.

    Returns:
        :class:`matplotlib.figure.Figure` pronta para ``savefig`` ou
        ``show``. Não chama ``plt.show()`` internamente.

    Raises:
        ImportError: Se matplotlib não estiver instalado.
        ValueError: Se ``results`` estiver vazio.

    Example:
        >>> from geosteering_ai.simulation.benchmarks.bench_forward import (
        ...     run_all_sizes
        ... )
        >>> from geosteering_ai.simulation.visualization import (
        ...     plot_benchmark_comparison
        ... )
        >>> results = run_all_sizes()
        >>> fig = plot_benchmark_comparison(results)
        >>> fig.savefig("benchmark_comparison.png", dpi=200)
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib é necessário para plot_benchmark_comparison. "
            "Instale via `pip install matplotlib`."
        )

    results = list(results)
    if not results:
        raise ValueError("`results` vazio — nada para plotar.")

    # ── Extração de dados ────────────────────────────────────────
    names = [r.profile_name for r in results]
    throughputs = np.array([r.throughput_models_per_hour for r in results])
    stds = np.array([getattr(r, "time_std_seconds", 0.0) or 0.0 for r in results])
    # Propagação de erro: σ(throughput) ≈ throughput · σ(t)/t
    mean_times = np.array([r.mean_time_seconds for r in results])
    mean_times_safe = np.where(mean_times > 0, mean_times, 1.0)
    throughput_stds = throughputs * (stds / mean_times_safe)

    # Speedup vs Fortran (%)
    fortran_pct = 100.0 * throughputs / _FORTRAN_BASELINE_MOD_H

    # ── Figura com 2 painéis ─────────────────────────────────────
    fig, (ax_tp, ax_sp) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"wspace": 0.30}
    )
    fig.suptitle(title or "Benchmark Forward — Simulador Python", fontsize=14, y=1.0)

    # Painel 1: throughput (mod/h) em barras
    x = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    ax_tp.bar(
        x,
        throughputs,
        yerr=throughput_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
    )
    if show_fortran_baseline:
        ax_tp.axhline(
            y=_FORTRAN_BASELINE_MOD_H,
            color="firebrick",
            linestyle="--",
            linewidth=1.5,
            label=f"Fortran ({_FORTRAN_BASELINE_MOD_H:,.0f} mod/h)",
        )
        ax_tp.legend(loc="best")
    ax_tp.set_xticks(x)
    ax_tp.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax_tp.set_ylabel("Throughput (modelos/hora)")
    ax_tp.set_title("Throughput por perfil")
    ax_tp.grid(axis="y", linestyle=":", alpha=0.5)

    # Painel 2: % do Fortran baseline
    ax_sp.plot(
        x,
        fortran_pct,
        marker="o",
        markersize=9,
        color="steelblue",
        linewidth=2.0,
    )
    ax_sp.axhline(y=100.0, color="firebrick", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_sp.axhline(
        y=68.0,
        color="darkorange",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label="Meta Python (40k ≈ 68% Fortran)",
    )
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax_sp.set_ylabel("% do baseline Fortran")
    ax_sp.set_title("Throughput relativo ao Fortran")
    ax_sp.grid(axis="y", linestyle=":", alpha=0.5)
    ax_sp.legend(loc="best", fontsize=9)

    # Anotações com valores exatos sobre cada barra (painel 1)
    for i, (tp, pct) in enumerate(zip(throughputs, fortran_pct)):
        ax_tp.annotate(
            f"{tp:,.0f}",
            xy=(i, tp),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
        ax_sp.annotate(
            f"{pct:.1f}%",
            xy=(i, pct),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


__all__ = ["plot_benchmark_comparison"]
