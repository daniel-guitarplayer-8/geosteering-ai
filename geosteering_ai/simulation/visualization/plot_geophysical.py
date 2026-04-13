# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/visualization/plot_geophysical.py             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Plotagens Geofísicas Avançadas (Sprint 2.10+)            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FUNÇÕES                                                                  ║
# ║    • plot_pseudosection(results_by_angle, ...)                           ║
# ║    • plot_polar_directivity(results_by_angle, ...)                       ║
# ║    • plot_nyquist(results_by_freq, ...)                                 ║
# ║    • plot_tornado(base_result, sweeps, ...)                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Plotagens geofísicas avançadas — pseudosection, polar, Nyquist, tornado."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

logger = logging.getLogger(__name__)

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


def _require_mpl():
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib é necessário. Instale via `pip install matplotlib`."
        )


# ──────────────────────────────────────────────────────────────────────────────
# plot_pseudosection — Re(H) posição × ângulo (anisotropia visualizada)
# ──────────────────────────────────────────────────────────────────────────────


def plot_pseudosection(
    results_by_angle: Dict[float, object],
    *,
    component: str = "Hzz",
    freq_idx: int = 0,
    use_real: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 7.0),
    cmap: str = "RdBu_r",
) -> "Figure":
    """Pseudo-section — H(posição, ângulo) como heatmap.

    Mostra como o tensor H varia com a profundidade (eixo y) e o
    ângulo de dip/azimute da ferramenta (eixo x). Anisotropia TIV
    aparece como assimetria no heatmap.

    Args:
        results_by_angle: Dict ``{angle_deg: SimulationResult}``.
        component: Componente do tensor.
        freq_idx: Frequência.
        use_real: Se True, plota Re(H); senão Im(H).
        title: Título.
        figsize: Tamanho.
        cmap: Colormap.
    """
    _require_mpl()

    angles = sorted(results_by_angle.keys())
    if not angles:
        raise ValueError("results_by_angle vazio")

    ic = _COMPS[component]
    first = results_by_angle[angles[0]]
    z_obs = first.z_obs
    n_positions = z_obs.size

    # Monta matriz (n_positions, n_angles)
    data = np.empty((n_positions, len(angles)), dtype=np.float64)
    for i_a, a in enumerate(angles):
        r = results_by_angle[a]
        H = r.H_tensor[:, freq_idx, ic]
        data[:, i_a] = H.real if use_real else H.imag

    vmax = np.max(np.abs(data))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        np.asarray(angles),
        z_obs,
        data,
        shading="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.invert_yaxis()
    label = "Re" if use_real else "Im"
    ax.set_xlabel(r"Dip $\theta$ (graus)")
    ax.set_ylabel("Profundidade (m)")
    freq = float(first.freqs_hz[freq_idx])
    freq_label = f"{freq/1000:.3g} kHz" if freq < 1e6 else f"{freq/1e6:.3g} MHz"
    ax.set_title(title or f"Pseudo-section {label}({component}) @ f = {freq_label}")
    plt.colorbar(im, ax=ax, label=f"{label}({component})")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_polar_directivity — |H(θ)| polar
# ──────────────────────────────────────────────────────────────────────────────


def plot_polar_directivity(
    results_by_angle: Dict[float, object],
    *,
    position_idx: int = 0,
    freq_idx: int = 0,
    component: str = "Hzz",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
) -> "Figure":
    """Polar plot da diretividade da ferramenta — |H(θ)|.

    Em uma única posição vertical, varre o dip θ e plota o módulo de H
    em coordenadas polares. Útil para visualizar como a ferramenta
    "vê" camadas em função da inclinação.

    Args:
        results_by_angle: Dict ``{angle_deg: SimulationResult}``.
        position_idx: Índice da posição a analisar.
        freq_idx: Frequência.
        component: Componente do tensor.
        title: Título.
        figsize: Tamanho.
    """
    _require_mpl()

    angles_deg = sorted(results_by_angle.keys())
    if not angles_deg:
        raise ValueError("results_by_angle vazio")

    ic = _COMPS[component]
    magnitudes = np.array(
        [
            np.abs(results_by_angle[a].H_tensor[position_idx, freq_idx, ic])
            for a in angles_deg
        ]
    )
    angles_rad = np.deg2rad(angles_deg)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(
        angles_rad,
        magnitudes,
        marker="o",
        color="steelblue",
        linewidth=2.0,
        markersize=5,
    )
    ax.fill(angles_rad, magnitudes, alpha=0.2, color="steelblue")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(
        title or f"Diretividade |{component}(θ)| — pos. {position_idx}",
        pad=20,
        fontsize=12,
    )
    ax.grid(True)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_nyquist — Re(H) vs Im(H) em frequência variável
# ──────────────────────────────────────────────────────────────────────────────


def plot_nyquist(
    result,
    *,
    position_idx: int = 0,
    component: str = "Hzz",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
) -> "Figure":
    """Nyquist plot — Re(H) vs Im(H) para múltiplas frequências.

    Mostra a trajetória do tensor H no plano complexo conforme a
    frequência varia. Loops indicam ressonância de modos TE/TM.

    Args:
        result: SimulationResult com ``freqs_hz`` contendo >= 2 freqs.
        position_idx: Posição a analisar.
        component: Componente.
        title: Título.
        figsize: Tamanho.
    """
    _require_mpl()

    ic = _COMPS[component]
    H = result.H_tensor[position_idx, :, ic]  # (nf,) complex
    freqs = result.freqs_hz

    if H.size < 2:
        raise ValueError(f"Nyquist precisa >= 2 frequências; obteve {H.size}")

    fig, ax = plt.subplots(figsize=figsize)

    # Color-code por frequência (logscale se variar > 1 ordem)
    if freqs.max() / freqs.min() > 10:
        log_freqs = np.log10(freqs)
        norm = (log_freqs - log_freqs.min()) / (log_freqs.max() - log_freqs.min())
    else:
        norm = (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-12)
    colors = plt.cm.viridis(norm)

    for i in range(H.size - 1):
        ax.plot(
            [H[i].real, H[i + 1].real],
            [H[i].imag, H[i + 1].imag],
            color=colors[i],
            linewidth=1.5,
        )
    ax.scatter(H.real, H.imag, c=freqs, cmap="viridis", s=30, zorder=3)

    # Anotações na primeira e última frequência
    ax.annotate(
        f"f = {freqs[0]/1000:.2g} kHz",
        xy=(H[0].real, H[0].imag),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
    )
    ax.annotate(
        f"f = {freqs[-1]/1000:.2g} kHz",
        xy=(H[-1].real, H[-1].imag),
        xytext=(10, -15),
        textcoords="offset points",
        fontsize=9,
    )

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(f"Re({component})")
    ax.set_ylabel(f"Im({component})")
    ax.set_title(
        title or f"Nyquist — {component} @ pos. {position_idx}",
        fontsize=12,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_tornado — sensibilidade de H a perturbações em cada camada
# ──────────────────────────────────────────────────────────────────────────────


def plot_tornado(
    base_value: float,
    sweeps: List[Tuple[str, float, float]],
    *,
    baseline_label: str = "Base",
    metric_label: str = r"$|H|$",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 7.0),
) -> "Figure":
    """Tornado chart — sensibilidade de uma métrica a perturbações.

    Cada linha é uma variável perturbada; a barra mostra o range
    entre (mínimo, máximo) da métrica. Ordenado por amplitude
    decrescente — dá a hierarquia de sensibilidade.

    Args:
        base_value: Valor da métrica no modelo baseline.
        sweeps: Lista de ``(var_name, low_value, high_value)``.
        baseline_label: Rótulo para linha vertical baseline.
        metric_label: Rótulo do eixo x.
        title: Título.
        figsize: Tamanho.

    Example:
        >>> # Sensibilidade de |Hzz| à variação de ρ em cada camada
        >>> sweeps = [
        ...     ("Camada 0 (ρh)", 0.008, 0.012),
        ...     ("Camada 1 (ρh)", 0.092, 0.108),
        ...     ("Camada 2 (ρh)", 0.0085, 0.0115),
        ... ]
        >>> fig = plot_tornado(0.01, sweeps)
    """
    _require_mpl()

    if not sweeps:
        raise ValueError("sweeps vazio")

    # Ordena por amplitude decrescente
    ranges = np.array([abs(hi - lo) for _, lo, hi in sweeps])
    order = np.argsort(ranges)[::-1]
    sorted_sweeps = [sweeps[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(sorted_sweeps))

    for i, (name, lo, hi) in enumerate(sorted_sweeps):
        min_val = min(lo, hi)
        max_val = max(lo, hi)
        # Barra da variação (de min a max)
        ax.barh(
            i,
            max_val - min_val,
            left=min_val,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        # Marcadores dos extremos
        ax.plot([lo], [i], "o", color="firebrick", markersize=8, zorder=3)
        ax.plot([hi], [i], "o", color="seagreen", markersize=8, zorder=3)

    ax.axvline(
        base_value,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"{baseline_label} = {base_value:.4g}",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in sorted_sweeps], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(metric_label)
    ax.set_title(title or "Tornado chart — sensibilidade por variável")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


__all__ = [
    "plot_pseudosection",
    "plot_polar_directivity",
    "plot_nyquist",
    "plot_tornado",
]
