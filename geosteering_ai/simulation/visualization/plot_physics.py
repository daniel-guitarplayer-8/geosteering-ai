# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/visualization/plot_physics.py                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Plotagens Físicas (Sprint 2.10+)       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : matplotlib                                                 ║
# ║  Dependências: matplotlib, numpy                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Plotagens físicas complementares ao tensor H — visualizações          ║
# ║    relevantes à interpretação geofísica (skin depth, atenuação/fase,    ║
# ║    FV/GS, sensitivity kernel).                                            ║
# ║                                                                           ║
# ║  FUNÇÕES                                                                  ║
# ║    • plot_skin_depth_heatmap(freqs, rhos)                                ║
# ║    • plot_attenuation_phase(result, component="Hzz")                     ║
# ║    • plot_feature_views(result)  — Re/Im/|H|/arg(H)                      ║
# ║    • plot_geosignals(result)     — razões simétrica/antissimétrica      ║
# ║    • plot_sensitivity_kernel(result, dH_drho)  [usa Jacobiano]         ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Ward & Hohmann (1988) §4 — formulação EM em meios estratificados  ║
# ║    • Nabighian (1988) — skin depth quasi-estático                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Plotagens físicas complementares do simulador EM 1D TIV."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from geosteering_ai.simulation.validation.half_space import skin_depth

logger = logging.getLogger(__name__)

# Epsilon para guard de denominador — suficiente para float64 e compatível
# com validação física do projeto (>= 1e-15).
_EPS_DENOM: float = 1e-12

# Componentes do tensor H (ordem compatível com SimulationResult.H_tensor)
_COMPONENT_INDEX = {
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


def _make_log_norm(arr):
    import matplotlib.colors as mcolors

    vmin = max(float(np.min(arr)), 1e-3)
    vmax = float(np.max(arr))
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


# ──────────────────────────────────────────────────────────────────────────────
# plot_skin_depth_heatmap — profundidade de investigação
# ──────────────────────────────────────────────────────────────────────────────


def plot_skin_depth_heatmap(
    freqs_hz: Optional[np.ndarray] = None,
    resistivities: Optional[np.ndarray] = None,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 6.0),
    cmap: str = "viridis",
) -> "Figure":
    """Heatmap de skin depth delta(f, rho) em metros.

    A skin depth é a profundidade de penetração 1/e do campo EM em
    meio homogêneo quasi-estático. Indica a profundidade de investigação
    típica da ferramenta LWD.
    """
    _require_mpl()

    if freqs_hz is None:
        freqs_hz = np.logspace(np.log10(100.0), np.log10(2.0e6), 60)
    if resistivities is None:
        resistivities = np.logspace(-1.0, 5.0, 60)

    F, R = np.meshgrid(freqs_hz, resistivities, indexing="xy")
    delta = skin_depth(F, R)

    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(
        F,
        R,
        delta,
        shading="auto",
        cmap=cmap,
        norm=_make_log_norm(delta),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel(r"Resistividade $\rho$ ($\Omega\cdot$m)")
    ax.set_title(title or r"Skin depth $\delta(f, \rho) = \sqrt{\rho/(\pi f \mu_0)}$")

    levels = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    cs = ax.contour(
        F,
        R,
        delta,
        levels=levels,
        colors="white",
        linewidths=0.8,
        alpha=0.7,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%gm")

    cbar = plt.colorbar(pcm, ax=ax, label=r"Skin depth $\delta$ (m)")
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_attenuation_phase — métricas LWD clássicas
# ──────────────────────────────────────────────────────────────────────────────


def plot_attenuation_phase(
    result,
    *,
    component: str = "Hzz",
    reference_component: Optional[str] = None,
    freq_idx: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 8.0),
) -> "Figure":
    """Plota atenuação (dB) e phase-shift (graus) ao longo do poço."""
    _require_mpl()

    ic = _COMPONENT_INDEX[component]
    H = result.H_tensor[:, freq_idx, ic]
    z_obs = result.z_obs

    if reference_component is not None:
        ic_ref = _COMPONENT_INDEX[reference_component]
        H_ref = result.H_tensor[:, freq_idx, ic_ref]
        atten_db = 20.0 * np.log10(np.abs(H) / np.maximum(np.abs(H_ref), _EPS_DENOM))
        phase_deg = np.rad2deg(np.angle(H) - np.angle(H_ref))
        ylabel_a = rf"$|H_{{{component}}}|/|H_{{{reference_component}}}|$ (dB)"
        ylabel_p = (
            rf"$\arg(H_{{{component}}}) - \arg(H_{{{reference_component}}})$ (graus)"
        )
    else:
        atten_db = 20.0 * np.log10(np.maximum(np.abs(H), _EPS_DENOM))
        phase_deg = np.rad2deg(np.angle(H))
        ylabel_a = rf"$|H_{{{component}}}|$ (dB)"
        ylabel_p = rf"$\arg(H_{{{component}}})$ (graus)"

    fig, (ax_a, ax_p) = plt.subplots(
        1, 2, figsize=figsize, sharey=True, gridspec_kw={"wspace": 0.15}
    )

    freq = float(result.freqs_hz[freq_idx])
    freq_label = f"{freq/1000:.3g} kHz" if freq < 1e6 else f"{freq/1e6:.3g} MHz"
    fig.suptitle(
        title or f"Atenuação e Fase — {component} @ f = {freq_label}",
        fontsize=13,
        y=0.98,
    )

    ax_a.plot(atten_db, z_obs, color="steelblue", linewidth=1.5)
    ax_a.invert_yaxis()
    ax_a.set_xlabel(ylabel_a)
    ax_a.set_ylabel("Profundidade (m)")
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.set_title("Atenuação")

    ax_p.plot(phase_deg, z_obs, color="darkorange", linewidth=1.5)
    ax_p.set_xlabel(ylabel_p)
    ax_p.grid(True, linestyle=":", alpha=0.5)
    ax_p.set_title("Phase-shift")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_feature_views — Re(H), Im(H), |H|, arg(H)
# ──────────────────────────────────────────────────────────────────────────────


def plot_feature_views(
    result,
    *,
    component: str = "Hzz",
    freq_idx: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14.0, 6.0),
) -> "Figure":
    """Plota as 4 FV (Feature Views) clássicas: Re, Im, |H|, arg(H)."""
    _require_mpl()

    ic = _COMPONENT_INDEX[component]
    H = result.H_tensor[:, freq_idx, ic]
    z_obs = result.z_obs

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    freq = float(result.freqs_hz[freq_idx])
    freq_label = f"{freq/1000:.3g} kHz" if freq < 1e6 else f"{freq/1e6:.3g} MHz"
    fig.suptitle(
        title or f"Feature Views — {component} @ f = {freq_label}",
        fontsize=13,
        y=1.00,
    )

    views = [
        (H.real, "Re(H)", "steelblue"),
        (H.imag, "Im(H)", "firebrick"),
        (np.abs(H), r"$|H|$", "seagreen"),
        (np.rad2deg(np.angle(H)), r"arg(H) (graus)", "purple"),
    ]
    for ax, (data, label, color) in zip(axes, views):
        ax.plot(data, z_obs, color=color, linewidth=1.5)
        ax.set_xlabel(label)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.invert_yaxis()

    axes[0].set_ylabel("Profundidade (m)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_geosignals — razões simétrica/antissimétrica
# ──────────────────────────────────────────────────────────────────────────────


def plot_geosignals(
    result,
    *,
    freq_idx: int = 0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 7.0),
) -> "Figure":
    """Plota Geosinais (razões simétricas e antissimétricas).

    Geosinais são as features direção-sensitivas usadas em geosteering.
    """
    _require_mpl()

    Hxx = result.H_tensor[:, freq_idx, 0]
    Hxz = result.H_tensor[:, freq_idx, 2]
    Hzx = result.H_tensor[:, freq_idx, 6]
    Hzz = result.H_tensor[:, freq_idx, 8]
    z_obs = result.z_obs

    gs_anti = (Hxz - Hzx) / (Hxx + Hzz + _EPS_DENOM)
    gs_sym = (Hxz + Hzx) / (Hxz - Hzx + _EPS_DENOM)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    freq = float(result.freqs_hz[freq_idx])
    freq_label = f"{freq/1000:.3g} kHz" if freq < 1e6 else f"{freq/1e6:.3g} MHz"
    fig.suptitle(
        title or f"Geosinais (direção-sensitivas) @ f = {freq_label}",
        fontsize=13,
        y=0.98,
    )

    ax1.plot(gs_anti.real, z_obs, color="steelblue", label="Re", linewidth=1.5)
    ax1.plot(
        gs_anti.imag,
        z_obs,
        color="firebrick",
        label="Im",
        linewidth=1.5,
        linestyle="--",
    )
    ax1.axvline(0.0, color="black", linestyle=":", alpha=0.5)
    ax1.set_xlabel(r"$(H_{xz} - H_{zx}) / (H_{xx} + H_{zz})$")
    ax1.set_ylabel("Profundidade (m)")
    ax1.set_title("GS antissimétrico (direção)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.invert_yaxis()

    ax2.plot(gs_sym.real, z_obs, color="seagreen", label="Re", linewidth=1.5)
    ax2.plot(
        gs_sym.imag,
        z_obs,
        color="purple",
        label="Im",
        linewidth=1.5,
        linestyle="--",
    )
    ax2.set_xlabel(r"$(H_{xz} + H_{zx}) / (H_{xz} - H_{zx})$")
    ax2.set_title("GS simétrico")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_sensitivity_kernel — ∂H/∂rho visualizado (heatmap)
# ──────────────────────────────────────────────────────────────────────────────


def plot_sensitivity_kernel(
    result,
    dH_drho: np.ndarray,
    *,
    component: str = "Hzz",
    freq_idx: int = 0,
    use_imag: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
    cmap: str = "RdBu_r",
) -> "Figure":
    """Heatmap do sensitivity kernel (posição x camada)."""
    _require_mpl()

    if dH_drho.ndim != 2:
        raise ValueError(
            f"dH_drho deve ser 2D (n_positions, n_layers), obtido {dH_drho.shape}"
        )

    z_obs = result.z_obs
    n_layers = dH_drho.shape[1]

    data = dH_drho.imag if use_imag else dH_drho.real
    vmax = np.max(np.abs(data))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data.T,
        aspect="auto",
        extent=(z_obs[0], z_obs[-1], n_layers - 0.5, -0.5),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )

    freq = float(result.freqs_hz[freq_idx])
    label = "Im" if use_imag else "Re"
    ax.set_xlabel("Profundidade da medição (m)")
    ax.set_ylabel("Índice da camada do perfil")
    ax.set_title(
        title
        or rf"Sensitivity kernel $\partial_\rho {label}({component})$"
        rf" @ f = {freq/1000:.3g} kHz"
    )

    plt.colorbar(im, ax=ax, label=rf"{label}$(\partial H / \partial \rho)$")
    fig.tight_layout()
    return fig


__all__ = [
    "plot_skin_depth_heatmap",
    "plot_attenuation_phase",
    "plot_feature_views",
    "plot_geosignals",
    "plot_sensitivity_kernel",
]
