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


# ──────────────────────────────────────────────────────────────────────────────
# Constante — epsilon para guard de denominador (compatível float64)
# ──────────────────────────────────────────────────────────────────────────────
_EPS_DENOM: float = 1e-12


# ──────────────────────────────────────────────────────────────────────────────
# plot_apparent_resistivity_curves — curvas LWD industriais (ρ_a vs TVD)
# ──────────────────────────────────────────────────────────────────────────────


def plot_apparent_resistivity_curves(
    result,
    *,
    components: Tuple[str, ...] = ("Hxx", "Hyy", "Hzz"),
    freq_indices: Optional[Iterable[int]] = None,
    mu0: float = 4e-7 * np.pi,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 8.0),
    show_true_rho: bool = True,
) -> "Figure":
    """Curvas de resistividade aparente vs TVD — padrão LWD industrial.

    Converte amplitudes do tensor H em resistividade aparente ρ_a(f, z)
    via expressão canônica de LWD (Moran-Gianzero 1979):

        ρ_a ≈ (ω · μ₀ · L²) / (2 · |H_ij · 4π · L³|)

    onde L = spacing T-R. Produz painel duplo: (esquerda) perfil ρ_h/ρ_v
    verdadeiro (log-scale, depth invertida), (direita) curvas ρ_a das
    componentes selecionadas vs TVD, sobrepostas por frequência.

    Convenção LWD: interpretação direta é válida em camadas homogêneas
    espessas; em boundaries e meios finos, ρ_a desvia da verdadeira ρ
    (fenômeno de polarização de camada). Este gráfico é o padrão
    industrial de visualização para navegação em tempo real (Schlumberger
    GeoSteering Pilot, Halliburton GeoForce).

    Args:
        result: :class:`SimulationResult` do simulador Python.
        components: Componentes do tensor a exibir (padrão: diagonais).
        freq_indices: Índices das frequências em ``result.freqs_hz``.
            Se ``None``, usa todas.
        mu0: Permeabilidade magnética do vácuo (H/m). Default 4π×10⁻⁷.
        title: Título da figura.
        figsize: Tamanho.
        show_true_rho: Se True, sobrepõe ρ_h/ρ_v verdadeiras no painel
            direito para comparação visual (industry-standard).

    Returns:
        :class:`matplotlib.figure.Figure` com 2 painéis lado a lado.

    Note:
        A conversão ρ_a é simplificada — em produção, o LWD service
        company aplica correções de invasão, dip, e espessura de camada
        antes de reportar. Este plot é apenas para verificação visual
        da sensibilidade do simulador.

    Example:
        >>> fig = plot_apparent_resistivity_curves(result,
        ...     components=("Hxx", "Hzz"), freq_indices=[0, 1])
        >>> fig.savefig("apparent_resistivity.png", dpi=300)
    """
    _require_mpl()

    H = np.asarray(result.H_tensor)
    z_obs = np.asarray(result.z_obs)
    freqs = np.asarray(result.freqs_hz)
    L = float(result.cfg.tr_spacing_m)

    if freq_indices is None:
        freq_indices = list(range(freqs.size))
    freq_indices = list(freq_indices)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(
        title or f"Resistividade aparente LWD — L = {L:.2f} m",
        fontsize=13,
        y=0.99,
    )

    # ── Painel esquerdo: ρ_h / ρ_v verdadeiras ────────────────────────
    ax_true = axes[0]
    ax_true.semilogx(
        result.rho_h_at_obs,
        z_obs,
        color="steelblue",
        linewidth=1.8,
        label=r"$\rho_h$ verdadeira",
    )
    ax_true.semilogx(
        result.rho_v_at_obs,
        z_obs,
        color="darkorange",
        linewidth=1.5,
        linestyle="--",
        label=r"$\rho_v$ verdadeira",
    )
    ax_true.invert_yaxis()
    ax_true.set_xlabel(r"Resistividade ($\Omega \cdot m$)")
    ax_true.set_ylabel("TVD (m)")
    ax_true.set_title("Perfil verdadeiro")
    ax_true.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_true.legend(loc="best", fontsize=9)

    # ── Painel direito: ρ_a(f) por componente ─────────────────────────
    ax_app = axes[1]
    if show_true_rho:
        ax_app.semilogx(
            result.rho_h_at_obs,
            z_obs,
            color="gray",
            linewidth=1.0,
            linestyle=":",
            alpha=0.6,
            label=r"$\rho_h$ (ref.)",
        )

    cmap = plt.get_cmap("viridis")
    for ic, comp in enumerate(components):
        idx = _COMPS[comp]
        for jf, fi in enumerate(freq_indices):
            H_ij = H[:, fi, idx]
            amp = np.abs(H_ij) + _EPS_DENOM
            # ρ_a ≈ ω μ₀ L² / (2 · amp · 4π L³) = f μ₀ / (4·amp·L)
            omega = 2.0 * np.pi * freqs[fi]
            rho_a = (omega * mu0) / (4.0 * amp * L + _EPS_DENOM) * L
            color = cmap(
                0.15
                + 0.7
                * (ic * len(freq_indices) + jf)
                / max(1, len(components) * len(freq_indices))
            )
            freq_hz = freqs[fi]
            flabel = (
                f"{freq_hz/1000:.1f} kHz" if freq_hz < 1e6 else f"{freq_hz/1e6:.1f} MHz"
            )
            ax_app.semilogx(
                rho_a,
                z_obs,
                color=color,
                linewidth=1.5,
                label=f"{comp} @ {flabel}",
            )

    ax_app.set_xlabel(r"$\rho_a$ aparente ($\Omega \cdot m$)")
    ax_app.set_title("Resistividade aparente")
    ax_app.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_app.legend(loc="best", fontsize=8, ncol=1)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_geosignal_response_vs_dip — GS USD/UAD/UHR/UHA em função de dip
# ──────────────────────────────────────────────────────────────────────────────


def plot_geosignal_response_vs_dip(
    results_by_dip: Dict[float, object],
    *,
    freq_idx: int = 0,
    z_target: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 7.0),
) -> "Figure":
    """Resposta de GeoSignals (USD/UAD/UHR/UHA) em função de dip relativo.

    Plota as 4 razões de GeoSignal canônicas do LWD direcional para
    boundary-mapping, em função do ângulo de dip relativo da ferramenta
    em relação à camada. Cada GS isola uma direção de gradiente:

      • USD = (Hxx - Hyy) / (Hxx + Hyy) — simétrica planar
      • UAD = (Hxz - Hzx) / (Hxz + Hzx + ε)  — antissimétrica cross
      • UHR = Re(Hxy / Hzz) — acoplamento horizontal normalizado
      • UHA = Re(Hxz / Hzz) — acoplamento axial normalizado

    O dip relativo φ é o ângulo entre o eixo z da ferramenta e a normal
    à camada (φ=0° → paralelo, φ=90° → perpendicular). A sensibilidade
    direcional do boundary mapping aparece como picos/zeros em valores
    específicos de φ.

    Args:
        results_by_dip: Dict ``{dip_graus: SimulationResult}`` — um
            resultado por ângulo de dip.
        freq_idx: Índice da frequência.
        z_target: Profundidade (m) alvo para extração. Se ``None``, usa
            o meio do perfil.
        title: Título.
        figsize: Tamanho.

    Returns:
        Figure com 4 painéis (2×2) — um por GS.

    Example:
        >>> # Rodar simulação para cada dip
        >>> dips = np.linspace(0, 90, 19)
        >>> results = {float(d): simulate(..., cfg=cfg_with_dip(d))
        ...            for d in dips}
        >>> fig = plot_geosignal_response_vs_dip(results)
    """
    _require_mpl()

    if not results_by_dip:
        raise ValueError("results_by_dip vazio")

    dips = sorted(results_by_dip.keys())
    _sample = results_by_dip[dips[0]]
    z_obs = np.asarray(_sample.z_obs)
    if z_target is None:
        z_idx = z_obs.size // 2
    else:
        z_idx = int(np.argmin(np.abs(z_obs - z_target)))
    z_actual = float(z_obs[z_idx])

    Hxx = np.array(
        [np.complex128(results_by_dip[d].H_tensor[z_idx, freq_idx, 0]) for d in dips]
    )
    Hyy = np.array(
        [np.complex128(results_by_dip[d].H_tensor[z_idx, freq_idx, 4]) for d in dips]
    )
    Hzz = np.array(
        [np.complex128(results_by_dip[d].H_tensor[z_idx, freq_idx, 8]) for d in dips]
    )
    Hxy = np.array(
        [np.complex128(results_by_dip[d].H_tensor[z_idx, freq_idx, 1]) for d in dips]
    )
    Hxz = np.array(
        [np.complex128(results_by_dip[d].H_tensor[z_idx, freq_idx, 2]) for d in dips]
    )
    Hzx = np.array(
        [np.complex128(results_by_dip[d].H_tensor[z_idx, freq_idx, 6]) for d in dips]
    )

    usd = (Hxx - Hyy) / (Hxx + Hyy + _EPS_DENOM)
    uad = (Hxz - Hzx) / (Hxz + Hzx + _EPS_DENOM)
    uhr = Hxy / (Hzz + _EPS_DENOM)
    uha = Hxz / (Hzz + _EPS_DENOM)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    freq = float(_sample.freqs_hz[freq_idx])
    flabel = f"{freq/1000:.3g} kHz" if freq < 1e6 else f"{freq/1e6:.3g} MHz"
    fig.suptitle(
        title or f"GeoSignals vs dip relativo — z = {z_actual:.2f} m, " f"f = {flabel}",
        fontsize=13,
        y=0.99,
    )

    gs_curves = [
        (axes[0, 0], usd, "USD — $(H_{xx} - H_{yy})/(H_{xx} + H_{yy})$", "steelblue"),
        (axes[0, 1], uad, "UAD — $(H_{xz} - H_{zx})/(H_{xz} + H_{zx})$", "firebrick"),
        (axes[1, 0], uhr, r"UHR — $H_{xy}/H_{zz}$", "seagreen"),
        (axes[1, 1], uha, r"UHA — $H_{xz}/H_{zz}$", "darkorange"),
    ]
    for ax, curve, lbl, color in gs_curves:
        ax.plot(dips, np.real(curve), color=color, linewidth=1.8, label="Re")
        ax.plot(
            dips,
            np.imag(curve),
            color=color,
            linewidth=1.3,
            linestyle="--",
            alpha=0.7,
            label="Im",
        )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Dip relativo (graus)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


__all__ = [
    "plot_pseudosection",
    "plot_polar_directivity",
    "plot_nyquist",
    "plot_tornado",
    "plot_apparent_resistivity_curves",
    "plot_geosignal_response_vs_dip",
]
