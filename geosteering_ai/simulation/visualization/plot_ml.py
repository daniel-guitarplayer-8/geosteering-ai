# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/visualization/plot_ml.py                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Plotagens para integração ML/DL (Sprint 2.10+)           ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Integração entre saídas do simulador Python e o pipeline v2.0 de     ║
# ║    Deep Learning (geosteering_ai/data, noise, visualization existente).║
# ║    Reutiliza convenções já estabelecidas no pacote principal.          ║
# ║                                                                           ║
# ║  FUNÇÕES                                                                  ║
# ║    • plot_augmentation_preview(result, noise_kind, snr_db)              ║
# ║      Mostra canal limpo vs com ruído (pipeline on-the-fly v2.0)        ║
# ║    • plot_uncertainty_bands(y_samples, z_obs)                           ║
# ║      Visualiza UQ (MC Dropout / Ensemble / INN) em formato bands      ║
# ║                                                                           ║
# ║  NOTA SOBRE INTEGRAÇÃO                                                    ║
# ║    Em vez de re-implementar augmentation (que existe em                 ║
# ║    geosteering_ai/noise/), estas plotagens CONSOMEM o output do        ║
# ║    pipeline existente quando disponível. Se o usuário tem um tensor   ║
# ║    ruidoso, passa-o diretamente; se não, aplicamos ruído simples       ║
# ║    in-situ para preview.                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Plotagens para integração com pipeline ML/DL do Geosteering AI v2.0."""
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
# plot_augmentation_preview — canal limpo vs ruidoso (pipeline v2.0)
# ──────────────────────────────────────────────────────────────────────────────


def plot_augmentation_preview(
    result,
    *,
    component: str = "Hzz",
    freq_idx: int = 0,
    noise_std_rel: float = 0.05,
    rng_seed: int = 42,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12.0, 6.0),
) -> "Figure":
    """Preview de data augmentation — tensor limpo vs ruidoso.

    Aplica ruído gaussiano multiplicativo (paridade com pipeline
    on-the-fly de v2.0 — ``geosteering_ai/noise/functions.py``
    gaussian_mult) e plota ambos lado a lado. Útil para validar
    visualmente que o augmentation preserva features relevantes.

    Args:
        result: SimulationResult do simulador Python.
        component: Componente do tensor.
        freq_idx: Frequência.
        noise_std_rel: Desvio padrão relativo do ruído (ex.: 0.05 = 5%).
        rng_seed: Seed para reprodutibilidade.
        title: Título.
        figsize: Tamanho.

    Returns:
        Figure com 4 subplots: Re(H) limpo, Re(H) ruidoso, Im(H) limpo,
        Im(H) ruidoso.

    Note:
        Para integração completa com pipeline v2.0, o caller pode
        passar `result` já pré-processado por DataPipeline. Aqui
        fazemos apenas augmentation visualmente representativo.
    """
    _require_mpl()

    ic = _COMPS[component]
    H = result.H_tensor[:, freq_idx, ic]
    z_obs = result.z_obs

    rng = np.random.default_rng(rng_seed)
    noise_re = rng.normal(0.0, noise_std_rel * np.abs(H.real), H.size)
    noise_im = rng.normal(0.0, noise_std_rel * np.abs(H.imag), H.size)
    H_noisy = (H.real + noise_re) + 1j * (H.imag + noise_im)

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    freq = float(result.freqs_hz[freq_idx])
    freq_label = f"{freq/1000:.3g} kHz" if freq < 1e6 else f"{freq/1e6:.3g} MHz"
    fig.suptitle(
        title
        or f"Augmentation preview — {component} @ {freq_label} — "
        f"noise_std = {noise_std_rel*100:.1f}%",
        fontsize=12,
        y=0.99,
    )

    pairs = [
        (H.real, "Re(H) limpo", "steelblue"),
        (H_noisy.real, "Re(H) ruidoso", "#7FB3D5"),
        (H.imag, "Im(H) limpo", "firebrick"),
        (H_noisy.imag, "Im(H) ruidoso", "#F5B7B1"),
    ]
    for ax, (data, label, color) in zip(axes, pairs):
        ax.plot(data, z_obs, color=color, linewidth=1.5)
        ax.set_xlabel(label)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.invert_yaxis()

    axes[0].set_ylabel("Profundidade (m)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_uncertainty_bands — UQ bands (MC Dropout / Ensemble / INN)
# ──────────────────────────────────────────────────────────────────────────────


def plot_uncertainty_bands(
    y_samples: np.ndarray,
    z_obs: np.ndarray,
    *,
    y_true: Optional[np.ndarray] = None,
    target_label: str = r"$\rho$ (Ω·m)",
    quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 7.0),
    use_log_x: bool = True,
) -> "Figure":
    """Plota bandas de incerteza (UQ) de inversão LWD.

    Formato canônico de saída de métodos de UQ do pipeline v2.0:
      - MC Dropout (geosteering_ai/inference/uncertainty.py)
      - Ensemble (N modelos com seeds diferentes)
      - INN (Invertible Neural Networks — posterior sampling)

    Args:
        y_samples: (n_samples, n_positions) — amostras da posterior.
        z_obs: (n_positions,) — profundidades.
        y_true: Opcional — ground truth para sobrepor.
        target_label: Rótulo do eixo x (ex.: ρh em Ω·m).
        quantiles: Quantis para bandas (low, mid, high). Default p5/p50/p95.
        title: Título.
        figsize: Tamanho.
        use_log_x: Se True, semilog no eixo x (apropriado para ρ).

    Returns:
        Figure com mediana + banda de incerteza + ground truth opcional.

    Example:
        >>> # y_samples: (100 MC runs, 300 positions)
        >>> fig = plot_uncertainty_bands(y_samples, z_obs, y_true=rho_true)
    """
    _require_mpl()

    if y_samples.ndim != 2:
        raise ValueError(
            f"y_samples deve ser 2D (n_samples, n_positions), obtido {y_samples.shape}"
        )

    q_lo, q_mid, q_hi = quantiles
    lo = np.quantile(y_samples, q_lo, axis=0)
    mid = np.quantile(y_samples, q_mid, axis=0)
    hi = np.quantile(y_samples, q_hi, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    # Banda
    ax.fill_betweenx(
        z_obs,
        lo,
        hi,
        color="steelblue",
        alpha=0.25,
        label=f"Banda p{int(q_lo*100)}–p{int(q_hi*100)}",
    )
    # Mediana
    ax.plot(
        mid,
        z_obs,
        color="steelblue",
        linewidth=2.0,
        label=f"Mediana (p{int(q_mid*100)})",
    )
    # Ground truth
    if y_true is not None:
        ax.plot(
            y_true,
            z_obs,
            color="firebrick",
            linewidth=1.5,
            linestyle="--",
            label="Ground truth",
        )

    ax.invert_yaxis()
    if use_log_x:
        ax.set_xscale("log")
    ax.set_xlabel(target_label)
    ax.set_ylabel("Profundidade (m)")
    ax.set_title(title or "Bandas de incerteza — inversão EM 1D")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# plot_pinn_loss_decomposition — decomposição L_total em termos PINN
# ──────────────────────────────────────────────────────────────────────────────


def plot_pinn_loss_decomposition(
    loss_history: dict,
    *,
    components: Tuple[str, ...] = ("loss_data", "loss_physics", "loss_continuity"),
    log_scale: bool = True,
    show_total: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 6.5),
) -> "Figure":
    """Decomposição temporal da função objetivo PINN (L_total = Σ L_i).

    Integra com os 8 cenários PINN do pipeline v2.0
    (``geosteering_ai/losses/pinns.py``) exibindo como cada termo
    da função objetivo composta evolui ao longo do treino. Cada
    cenário PINN pode ter seu peso relativo — esta plotagem expõe
    visualmente o balanço:

        L_total = λ_data · L_data + λ_phys · L_physics + λ_cont · L_continuity

    Útil para:
      - Diagnóstico de colapso de gradiente (um termo domina)
      - Curriculum PINN (terms devem convergir em momentos distintos)
      - Sanity check do weighting adaptativo (ex.: GradNorm, SoftAdapt)

    Args:
        loss_history: Dict ``{"epochs": [...], "loss_data": [...],
            "loss_physics": [...], ...}``. Compatível com o callback
            de v2.0 ``training/callbacks.py::PINNLossLogger``.
        components: Quais termos plotar (deve existir em ``loss_history``).
            Default cobre o caso PINN-canônico.
        log_scale: Se True, usa escala log no eixo y (padrão para
            funções objetivo que caem 3-6 ordens de magnitude).
        show_total: Se True, traça também L_total como linha tracejada.
        title: Título.
        figsize: Tamanho.

    Returns:
        Figure com painel principal (curvas por termo) + painel inferior
        com razões relativas L_i / L_total ao longo do treino.

    Raises:
        KeyError: Se ``loss_history`` não contiver ``"epochs"`` ou algum
            dos ``components``.

    Note:
        Para integração com TensorBoard ou Weights & Biases, o caller
        pode persistir ``loss_history`` via callback e chamar esta
        plotagem para relatórios pós-treino.

    Example:
        >>> history = {
        ...     "epochs": list(range(100)),
        ...     "loss_data": np.exp(-0.05 * np.arange(100)) + 1e-3,
        ...     "loss_physics": 0.5 * np.exp(-0.03 * np.arange(100)),
        ...     "loss_continuity": 0.1 * np.exp(-0.02 * np.arange(100)),
        ... }
        >>> fig = plot_pinn_loss_decomposition(history)
    """
    _require_mpl()

    if "epochs" not in loss_history:
        raise KeyError("loss_history deve conter chave 'epochs'")
    epochs = np.asarray(loss_history["epochs"], dtype=np.float64)

    # Valida presença dos componentes
    for comp in components:
        if comp not in loss_history:
            raise KeyError(f"loss_history não contém '{comp}'")

    curves = {c: np.asarray(loss_history[c], dtype=np.float64) for c in components}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    fig.suptitle(
        title or "Decomposição da função objetivo PINN — L_total = Σ L_i",
        fontsize=13,
        y=0.99,
    )

    # ── Painel 1: curvas individuais + total ──────────────────────────
    ax_main = axes[0]
    palette = ["steelblue", "firebrick", "seagreen", "darkorange", "purple"]
    total = np.zeros_like(epochs)
    for i, (comp, values) in enumerate(curves.items()):
        color = palette[i % len(palette)]
        label = comp.replace("loss_", r"$\mathcal{L}_{\mathrm{") + r"}}$"
        ax_main.plot(
            epochs,
            np.maximum(values, 1e-15),
            color=color,
            linewidth=1.8,
            label=label,
        )
        total = total + np.maximum(values, 0.0)

    if show_total:
        ax_main.plot(
            epochs,
            np.maximum(total, 1e-15),
            color="black",
            linewidth=2.2,
            linestyle="--",
            alpha=0.8,
            label=r"$\mathcal{L}_{\mathrm{total}}$",
        )

    if log_scale:
        ax_main.set_yscale("log")
    ax_main.set_ylabel("Valor da função objetivo")
    ax_main.set_title("Evolução por termo")
    ax_main.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_main.legend(loc="upper right", fontsize=9)

    # ── Painel 2: razões relativas L_i / L_total ──────────────────────
    ax_ratio = axes[1]
    total_safe = np.maximum(total, 1e-12)
    for i, (comp, values) in enumerate(curves.items()):
        color = palette[i % len(palette)]
        ratio = values / total_safe
        ax_ratio.plot(epochs, ratio, color=color, linewidth=1.5)

    ax_ratio.set_xlabel("Época")
    ax_ratio.set_ylabel(r"$\mathcal{L}_i / \mathcal{L}_{\mathrm{total}}$")
    ax_ratio.set_title("Contribuição relativa por termo")
    ax_ratio.set_ylim(0.0, 1.05)
    ax_ratio.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


__all__ = [
    "plot_augmentation_preview",
    "plot_uncertainty_bands",
    "plot_pinn_loss_decomposition",
]
