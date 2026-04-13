# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/visualization/plot_benchmark_advanced.py      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Plotagens de Benchmark e Diagnóstico (Sprint 2.10+)      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : matplotlib, numpy, time                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FUNÇÕES                                                                  ║
# ║    • plot_speedup_curve(n_threads_list, throughputs)                     ║
# ║    • plot_filter_convergence(model, filters=[...])                       ║
# ║    • plot_error_heatmap(H_ref, H_test, z_obs, freqs)                    ║
# ║    • plot_component_times(component_names, times)                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Plotagens de benchmark e diagnóstico do simulador Python."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

logger = logging.getLogger(__name__)


def _require_mpl():
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib é necessário. Instale via `pip install matplotlib`."
        )


# ──────────────────────────────────────────────────────────────────────────────
# plot_speedup_curve — strong scaling
# ──────────────────────────────────────────────────────────────────────────────


def plot_speedup_curve(
    n_threads_list: Iterable[int],
    throughputs_mod_h: Iterable[float],
    *,
    reference_baseline: Optional[float] = None,
    label: str = "Numba",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 6.0),
) -> "Figure":
    """Plota curva de speedup (strong scaling) vs número de threads.

    Args:
        n_threads_list: Lista de n_threads testados.
        throughputs_mod_h: Throughput correspondente em mod/h.
        reference_baseline: Valor de referência (mod/h) p/ linha horizontal
            tracejada (default: Fortran OpenMP ~58856).
        label: Rótulo da série.
        title: Título.
        figsize: Tamanho.

    Returns:
        Figure com 2 subplots: throughput absoluto + speedup relativo.

    Example:
        >>> n = [1, 2, 4, 8, 16]
        >>> tp = [25000, 47000, 85000, 150000, 180000]
        >>> fig = plot_speedup_curve(n, tp, reference_baseline=58856)
    """
    _require_mpl()

    n_list = np.asarray(list(n_threads_list), dtype=np.int64)
    tp_list = np.asarray(list(throughputs_mod_h), dtype=np.float64)

    if n_list.size == 0:
        raise ValueError("n_threads_list vazio")

    speedup_rel = tp_list / tp_list[0]  # relativo a 1 thread

    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title or f"Strong scaling — {label}", fontsize=13, y=0.98)

    # Painel 1: throughput absoluto
    ax_abs.plot(n_list, tp_list, marker="o", color="steelblue", linewidth=1.8)
    if reference_baseline is not None:
        ax_abs.axhline(
            reference_baseline,
            color="firebrick",
            linestyle="--",
            linewidth=1.4,
            label=f"Ref ({reference_baseline:,.0f} mod/h)",
        )
        ax_abs.legend(loc="best")
    ax_abs.set_xlabel("Número de threads")
    ax_abs.set_ylabel("Throughput (mod/h)")
    ax_abs.set_title("Throughput absoluto")
    ax_abs.set_xscale("log")
    ax_abs.set_xticks(n_list)
    ax_abs.set_xticklabels([str(n) for n in n_list])
    ax_abs.grid(True, linestyle=":", alpha=0.5)

    # Painel 2: speedup relativo + linha ideal (linear)
    ax_rel.plot(
        n_list, speedup_rel, marker="o", color="darkorange", linewidth=1.8, label="Medido"
    )
    ax_rel.plot(
        n_list,
        n_list.astype(float),
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="Ideal (linear)",
    )
    ax_rel.set_xlabel("Número de threads")
    ax_rel.set_ylabel("Speedup relativo (× 1 thread)")
    ax_rel.set_title("Eficiência paralela")
    ax_rel.set_xscale("log")
    ax_rel.set_yscale("log")
    ax_rel.set_xticks(n_list)
    ax_rel.set_xticklabels([str(n) for n in n_list])
    ax_rel.legend(loc="best", fontsize=9)
    ax_rel.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_filter_convergence — Werthmüller vs Kong vs Anderson
# ──────────────────────────────────────────────────────────────────────────────


def plot_filter_convergence(
    canonical_model_name: str = "oklahoma_3",
    *,
    frequency_hz: float = 20000.0,
    tr_spacing_m: float = 1.0,
    n_positions: int = 100,
    component: str = "Hzz",
    reference_filter: str = "anderson_801pt",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12.0, 6.0),
) -> "Figure":
    """Plota convergência dos filtros Hankel (vs filtro de referência).

    Mostra o erro relativo de Kong 61pt e Werthmüller 201pt contra
    Anderson 801pt (considerado como ground-truth de precisão máxima).
    Útil para decidir qual filtro usar em cada caso de aplicação.

    Args:
        canonical_model_name: Modelo canônico para testar.
        frequency_hz: Frequência.
        tr_spacing_m: Espaçamento TR.
        n_positions: Número de posições.
        component: Componente do tensor analisado.
        reference_filter: Filtro ground-truth (default Anderson 801pt).
        title: Título.
        figsize: Tamanho.

    Returns:
        Figure com 2 painéis: componente H em cada filtro + erro relativo.
    """
    _require_mpl()

    from geosteering_ai.simulation import SimulationConfig, simulate
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    _COMPS = {
        "Hxx": 0,
        "Hxy": 1,
        "Hxz": 2,
        "Hyx": 3,
        "Hyy": 4,
        "Hyz": 5,
        "Hzx": 6,
        "Hzy": 7,
        "Hzz": 8,
    }
    ic = _COMPS[component]

    model = get_canonical_model(canonical_model_name)  # type: ignore[arg-type]
    positions_z = np.linspace(model.min_depth - 2.0, model.max_depth + 2.0, n_positions)

    filters = ["kong_61pt", "werthmuller_201pt", "anderson_801pt"]
    results = {}
    for filt_name in filters:
        cfg = SimulationConfig(
            frequency_hz=frequency_hz,
            tr_spacing_m=tr_spacing_m,
            hankel_filter=filt_name,
            parallel=False,
        )
        res = simulate(
            rho_h=model.rho_h,
            rho_v=model.rho_v,
            esp=model.esp,
            positions_z=positions_z,
            cfg=cfg,
        )
        results[filt_name] = res.H_tensor[:, 0, ic]

    ref = results[reference_filter]

    fig, (ax_h, ax_err) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(
        title or f"Convergência de filtros Hankel — {model.title} — {component}",
        fontsize=13,
        y=0.98,
    )

    colors = {
        "kong_61pt": "steelblue",
        "werthmuller_201pt": "darkorange",
        "anderson_801pt": "firebrick",
    }

    # Painel 1: Re(H)
    for fname, H in results.items():
        label = fname.replace("_", " ")
        ax_h.plot(H.real, positions_z, label=label, color=colors[fname], linewidth=1.5)
    ax_h.invert_yaxis()
    ax_h.set_xlabel(f"Re({component})")
    ax_h.set_ylabel("Profundidade (m)")
    ax_h.set_title(f"Re({component}) nos 3 filtros")
    ax_h.legend(loc="best", fontsize=9)
    ax_h.grid(True, linestyle=":", alpha=0.5)

    # Painel 2: erro relativo vs Anderson
    for fname, H in results.items():
        if fname == reference_filter:
            continue
        rel_err = np.abs(H - ref) / (np.abs(ref) + 1e-12)
        ax_err.semilogx(
            rel_err,
            positions_z,
            label=f"{fname} vs {reference_filter}",
            color=colors[fname],
            linewidth=1.5,
        )
    ax_err.set_xlabel(r"$|H_{filt} - H_{ref}| / |H_{ref}|$")
    ax_err.set_title("Erro relativo")
    ax_err.legend(loc="best", fontsize=9)
    ax_err.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_error_heatmap — erro relativo posição × frequência
# ──────────────────────────────────────────────────────────────────────────────


def plot_error_heatmap(
    H_reference: np.ndarray,
    H_test: np.ndarray,
    *,
    z_obs: Optional[np.ndarray] = None,
    freqs_hz: Optional[np.ndarray] = None,
    component: str = "Hzz",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
    cmap: str = "magma",
) -> "Figure":
    """Heatmap de erro relativo entre dois backends/resultados.

    Visualiza onde (posição × frequência) os backends divergem — útil
    para validação cruzada Fortran ↔ Numba ↔ JAX ↔ empymod.

    Args:
        H_reference: (n_positions, nf, 9) — resultado de referência.
        H_test: (n_positions, nf, 9) — resultado testado.
        z_obs: (n_positions,) — profundidades.
        freqs_hz: (nf,) — frequências.
        component: Componente analisado.
        title: Título.
        figsize: Tamanho.
        cmap: Colormap.
    """
    _require_mpl()

    if H_reference.shape != H_test.shape:
        raise ValueError("H_reference e H_test devem ter o mesmo shape")

    _COMPS = {
        "Hxx": 0,
        "Hxy": 1,
        "Hxz": 2,
        "Hyx": 3,
        "Hyy": 4,
        "Hyz": 5,
        "Hzx": 6,
        "Hzy": 7,
        "Hzz": 8,
    }
    ic = _COMPS[component]

    ref = H_reference[:, :, ic]
    test = H_test[:, :, ic]
    rel_err = np.abs(test - ref) / (np.abs(ref) + 1e-12)

    if z_obs is None:
        z_obs = np.arange(H_reference.shape[0], dtype=np.float64)
    if freqs_hz is None:
        freqs_hz = np.arange(H_reference.shape[1], dtype=np.float64)

    fig, ax = plt.subplots(figsize=figsize)
    # Log-scale do erro para visualizar ordens de magnitude
    with np.errstate(divide="ignore"):
        log_err = np.log10(np.maximum(rel_err, 1e-16))

    im = ax.pcolormesh(
        freqs_hz,
        z_obs,
        log_err,
        shading="auto",
        cmap=cmap,
        vmin=-16,
        vmax=0,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Profundidade (m)")
    ax.set_title(title or f"Erro relativo (log10) — {component} — test vs referência")

    cbar = plt.colorbar(im, ax=ax, label=r"$\log_{10}|H_{test}-H_{ref}|/|H_{ref}|$")
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_component_times — tempo por componente (propagation, dipoles, etc.)
# ──────────────────────────────────────────────────────────────────────────────


def plot_component_times(
    component_names: List[str],
    times_s: List[float],
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 5.5),
) -> "Figure":
    """Bar chart horizontal dos tempos gastos em cada componente do kernel.

    Útil para identificar hot spots restantes após otimização. Mede
    em millisegundos o tempo por chamada de common_arrays,
    common_factors, hmd_tiv, vmd, rotate_tensor.

    Args:
        component_names: Nomes dos componentes.
        times_s: Tempos em segundos (por chamada).
        title: Título.
        figsize: Tamanho.
    """
    _require_mpl()

    names = list(component_names)
    times_ms = np.asarray(times_s) * 1000.0
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    order = np.argsort(times_ms)[::-1]
    names_sorted = [names[i] for i in order]
    times_sorted = times_ms[order]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names_sorted))
    bars = ax.barh(y_pos, times_sorted, color=colors, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Tempo (ms/chamada)")
    ax.set_title(title or "Tempo por componente do kernel")
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    # Anotações com valores
    for i, (bar, t) in enumerate(zip(bars, times_sorted)):
        ax.annotate(
            f"{t:.2f} ms",
            xy=(t, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def measure_component_times(
    n_layers: int = 22,
    n_positions: int = 601,
    n_iter: int = 30,
) -> dict:
    """Mede tempos por componente do kernel (para plot_component_times).

    Returns:
        Dict ``{component_name: mean_time_s}`` com medições.
    """
    from geosteering_ai.simulation._numba.dipoles import hmd_tiv, vmd
    from geosteering_ai.simulation._numba.geometry import sanitize_profile
    from geosteering_ai.simulation._numba.propagation import (
        common_arrays,
        common_factors,
    )
    from geosteering_ai.simulation._numba.rotation import rotate_tensor
    from geosteering_ai.simulation.filters import FilterLoader

    filt = FilterLoader().load("werthmuller_201pt")
    np.random.seed(42)
    rho_h = np.exp(np.random.uniform(np.log(0.5), np.log(500), n_layers))
    rho_v = rho_h * 1.5
    esp = np.ones(n_layers - 2) * 2.0
    h_arr, prof_arr = sanitize_profile(n_layers, esp)
    eta = np.stack([1.0 / rho_h, 1.0 / rho_v], axis=-1)
    zeta = 1j * 2.0 * np.pi * 20000.0 * 4.0e-7 * np.pi

    # Warmup JIT
    outs = common_arrays(n_layers, 201, 0.01, filt.abscissas, zeta, h_arr, eta)
    cf = common_factors(n_layers, 201, 0.0, h_arr, prof_arr, 10, *outs[:8])

    # Medições
    def _timeit(fn):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        return (time.perf_counter() - t0) / n_iter

    t_ca = _timeit(
        lambda: common_arrays(n_layers, 201, 0.01, filt.abscissas, zeta, h_arr, eta)
    )
    t_cf = _timeit(
        lambda: common_factors(n_layers, 201, 0.0, h_arr, prof_arr, 10, *outs[:8])
    )
    t_hmd = _timeit(
        lambda: hmd_tiv(
            0.0,
            0.0,
            0.0,
            n_layers,
            10,
            10,
            201,
            filt.abscissas,
            filt.weights_j0,
            filt.weights_j1,
            h_arr,
            prof_arr,
            zeta,
            eta,
            0.01,
            0.0,
            0.0,
            *outs[:8],
            *cf[:4],
        )
    )
    t_vmd = _timeit(
        lambda: vmd(
            0.0,
            0.0,
            0.0,
            n_layers,
            10,
            10,
            201,
            filt.abscissas,
            filt.weights_j0,
            filt.weights_j1,
            h_arr,
            prof_arr,
            zeta,
            0.01,
            0.0,
            0.0,
            outs[0],
            outs[2],
            outs[8],
            outs[4],
            outs[5],
            cf[4],
            cf[5],
        )
    )
    matH = np.eye(3, dtype=np.complex128)
    t_rot = _timeit(lambda: rotate_tensor(0.3, 0.0, 0.0, matH))

    return {
        "common_arrays": t_ca,
        "common_factors": t_cf,
        "hmd_tiv": t_hmd,
        "vmd": t_vmd,
        "rotate_tensor": t_rot,
    }


__all__ = [
    "plot_speedup_curve",
    "plot_filter_convergence",
    "plot_error_heatmap",
    "plot_component_times",
    "measure_component_times",
]
