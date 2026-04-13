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


__all__ = [
    "plot_augmentation_preview",
    "plot_uncertainty_bands",
]
