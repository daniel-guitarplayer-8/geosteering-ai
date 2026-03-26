# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/geosteering.py                                     ║
# ║  Bloco: 9 — Visualization (C72)                                          ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • plot_curtain: curtain plot 2D de resistividade vs profundidade       ║
# ║    • plot_dtb_profile: perfil DTB true vs predicted                       ║
# ║    • plot_geosteering_dashboard: painel 4-quadrantes de geosteering      ║
# ║                                                                            ║
# ║  Dependencias: numpy, matplotlib (lazy import)                            ║
# ║  Exports: ~3 funcoes — ver __all__                                        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C72)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Visualizacoes especificas para geosteering — curtain, DTB, dashboard.

Funcoes de plotagem para operacoes de geosteering em cenarios de
inversao de resistividade 1D. Todas as funcoes importam matplotlib
de forma lazy para suportar ambientes headless.

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  VISUALIZACOES DE GEOSTEERING                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. Curtain Plot (plot_curtain)                                        │
    │     Mapa 2D: eixo X = posicao lateral, eixo Y = profundidade          │
    │     Cor = resistividade predita (log10). Incerteza como bandas.       │
    │                                                                         │
    │  2. DTB Profile (plot_dtb_profile)                                     │
    │     Perfil 1D: DTB true (preto) vs DTB predicted (azul)               │
    │     Eixo X = posicao lateral, eixo Y = DTB (scaled domain)            │
    │                                                                         │
    │  3. Geosteering Dashboard (plot_geosteering_dashboard)                 │
    │     4 paineis: DTB error, R2 evolution, latencia, confianca           │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Example:
    >>> from geosteering_ai.visualization.geosteering import (
    ...     plot_curtain, plot_dtb_profile, plot_geosteering_dashboard,
    ... )
    >>> plot_curtain(y_pred, z_obs=z, show=False)
    >>> plot_dtb_profile(dtb_true, dtb_pred, show=False)

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Matplotlib importado LAZY dentro de cada funcao.
    Referenciado em:
        - visualization/__init__.py: re-exportado como API publica
        - evaluation/geosteering_report.py: figuras para secao 5
        - tests/test_visualization_geosteering.py: testes de plotagem
    Ref: docs/ARCHITECTURE_v2.md secao 9 (Visualization).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Curtain plot 2D ---
    "plot_curtain",
    # --- DTB profile ---
    "plot_dtb_profile",
    # --- Dashboard 4-quadrantes ---
    "plot_geosteering_dashboard",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de visualizacao geosteering
# ──────────────────────────────────────────────────────────────────────

# Resolucao default para salvamento de figuras (DPI)
_DEFAULT_DPI = 300

# Colormap para curtain plot de resistividade
_DEFAULT_CMAP = "viridis"

# Cores para consistencia visual entre modulos
_COLOR_TRUE = "black"
_COLOR_PRED = "#1f77b4"       # azul matplotlib default
_COLOR_UNCERTAINTY = "#aec7e8"  # azul claro para bandas de incerteza
_COLOR_DTB_TRUE = "black"
_COLOR_DTB_PRED = "#d62728"    # vermelho para DTB predito
_COLOR_ERROR = "#ff7f0e"       # laranja para barras de erro
_COLOR_LATENCY = "#2ca02c"     # verde para latencia
_COLOR_CONFIDENCE = "#9467bd"  # roxo para confianca

# Nomes de componentes de resistividade
_COMPONENT_NAMES = ("rho_h", "rho_v")

# Epsilon para estabilidade numerica (float32)
_EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════
# CURTAIN PLOT — Mapa 2D de resistividade predita vs profundidade
#
# Eixo X: posicao lateral (amostras ou z_obs se fornecido)
# Eixo Y: profundidade (pontos na sequencia)
# Cor: resistividade em log10 (Ohm.m)
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  Curtain Plot: Resistividade Predita                               │
# │  ┌───────────────────────────────────────────────┐ ┌──────┐       │
# │  │░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓███████████████████████████│ │ 3.0  │       │
# │  │░░░░░░░░▓▓▓▓▓▓▓▓▓█████████████████████████████│ │      │       │
# │  │░░░░░░░░░░░░▓▓▓▓▓██████████████████████████████│ │ 2.0  │ log10│
# │  │░░░░░░░░░░░░░░░░▓▓▓████████████████████████████│ │      │       │
# │  │░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓███████████████████│ │ 1.0  │       │
# │  └───────────────────────────────────────────────┘ └──────┘       │
# │       Posicao lateral (pontos)                                     │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_curtain(
    y_pred: np.ndarray,
    *,
    z_obs: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    component: int = 0,
    title: str = "Curtain Plot",
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C72: Curtain plot 2D de resistividade predita vs profundidade.

    Gera um mapa 2D color-coded onde o eixo X representa a posicao
    lateral (amostras) e o eixo Y a profundidade (pontos na sequencia).
    A cor codifica a resistividade predita em log10 (Ohm.m).

    Se ``uncertainty`` for fornecido, sobrepoe contornos de incerteza
    (desvio-padrao) como isolinhas sobre o curtain plot.

    Args:
        y_pred: Array (N, seq_len, n_channels) com resistividade
            predita em dominio log10. Canal selecionado via ``component``.
        z_obs: Array opcional (N,) com posicoes laterais de observacao
            (ex: measured depth). Se None, usa indices [0, 1, ..., N-1].
        uncertainty: Array opcional (N, seq_len, n_channels) com
            desvio-padrao da predicao. Se fornecido, overlay de
            contornos de incerteza (isolinhas a 1 sigma).
        component: Indice do canal a plotar (0 = rho_h, 1 = rho_v).
            Default: 0.
        title: Titulo do plot. Default: "Curtain Plot".
        save_path: Caminho completo para salvar figura (None = nao salva).
        show: Se True, chama ``plt.show()``. Default: True.
        dpi: Resolucao DPI. Default: 300.

    Raises:
        ValueError: Se ``component`` >= n_channels.
        ValueError: Se ``y_pred`` nao for 3D.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import numpy as np
        >>> y_pred = np.random.randn(100, 600, 2)
        >>> plot_curtain(y_pred, component=0, show=False)

    Note:
        Matplotlib importado LAZY.
        Dados esperados em log10 (TARGET_SCALING = "log10").
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - evaluation/geosteering_report.py: secao 5 (Figuras)
        Ref: docs/ARCHITECTURE_v2.md secao 9.
    """
    # --- Validacao de entrada ---
    if y_pred.ndim != 3:
        raise ValueError(
            f"Esperado array 3D (N, seq_len, channels), "
            f"recebido ndim={y_pred.ndim}, shape={y_pred.shape}"
        )

    n_samples, seq_len, n_channels = y_pred.shape

    if component >= n_channels:
        raise ValueError(
            f"component={component} fora do range [0, {n_channels - 1}]. "
            f"Array tem {n_channels} canais."
        )

    # --- Lazy import matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    logger.info(
        "Plotando curtain plot — component=%d (%s), "
        "N=%d amostras, seq_len=%d",
        component,
        _COMPONENT_NAMES[component] if component < len(_COMPONENT_NAMES) else f"canal_{component}",
        n_samples, seq_len,
    )

    # --- Extrair dados do canal selecionado ---
    # data shape: (N, seq_len) — cada linha e uma amostra
    data = y_pred[:, :, component]  # (N, seq_len)

    # --- Eixo de posicao lateral ---
    if z_obs is not None:
        x_axis = z_obs
        x_label = "Posicao Lateral (z_obs)"
    else:
        x_axis = np.arange(n_samples)
        x_label = "Amostra"

    # --- Eixo de profundidade ---
    depth_axis = np.arange(seq_len)

    # --- Criar figura ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Curtain plot como pcolormesh
    # Transpor data para (seq_len, N) pois Y=profundidade, X=posicao
    im = ax.pcolormesh(
        x_axis, depth_axis, data.T,
        cmap=_DEFAULT_CMAP,
        shading="auto",
    )

    # Colorbar
    comp_name = (
        _COMPONENT_NAMES[component]
        if component < len(_COMPONENT_NAMES)
        else f"canal_{component}"
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(
        rf"$\log_{{10}}(\rho_{{{comp_name}}})$ [$\Omega\cdot$m]",
        fontsize=11,
    )

    # --- Overlay de incerteza (contornos) ---
    if uncertainty is not None:
        if uncertainty.shape == y_pred.shape:
            unc_data = uncertainty[:, :, component].T  # (seq_len, N)
            # Contornos de 1-sigma
            contour = ax.contour(
                x_axis, depth_axis, unc_data,
                levels=5,
                colors="white",
                linewidths=0.8,
                alpha=0.6,
            )
            ax.clabel(contour, inline=True, fontsize=7, fmt="%.3f")
            logger.info("Contornos de incerteza sobrepostos (5 niveis).")
        else:
            logger.warning(
                "Shape de uncertainty (%s) incompativel com y_pred (%s). "
                "Contornos de incerteza ignorados.",
                uncertainty.shape, y_pred.shape,
            )

    # --- Labels e formatacao ---
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Profundidade [pontos]", fontsize=12)
    ax.set_title(
        f"{title} — {comp_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.invert_yaxis()  # profundidade cresce para baixo

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("Curtain plot salvo em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info(
        "plot_curtain concluido — component=%d, %d amostras",
        component, n_samples,
    )


# ════════════════════════════════════════════════════════════════════════
# DTB PROFILE — Perfil 1D de Distance-to-Boundary
#
# DTB true (preto solido) vs DTB predicted (vermelho tracejado),
# com eixo X = posicao lateral e eixo Y = DTB (dominio escalado).
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  DTB Profile: True vs Predicted                                    │
# │                                                                     │
# │  ──── True (preto)                                                 │
# │  ---- Pred (vermelho)                                              │
# │                                                                     │
# │  DTB │   ╱\   ╱──╲                                                │
# │      │  ╱  ╲_╱    ╲___╱──╲                                        │
# │      │ ╱                    ╲                                      │
# │      └──────────────────────────→ Posicao lateral                  │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_dtb_profile(
    dtb_true: np.ndarray,
    dtb_pred: np.ndarray,
    *,
    z_obs: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C72: Perfil de DTB (Distance-to-Boundary) true vs predicted.

    Plota o perfil 1D de DTB com sobreposicao de valores verdadeiros
    (preto solido) e preditos (vermelho tracejado). Se os arrays
    forem 2D (N, seq_len), plota a media sobre amostras com banda
    de desvio-padrao sombreada.

    Args:
        dtb_true: Array com DTB verdadeiro no dominio escalado.
            Shape 1D (seq_len,) para amostra unica, ou
            2D (N, seq_len) para multiplas amostras (media plotada).
        dtb_pred: Array com DTB predito. Mesmo shape de ``dtb_true``.
        z_obs: Array opcional com posicoes laterais. Se None, usa
            indices. Deve ter comprimento = seq_len.
        save_path: Caminho para salvar figura (None = nao salva).
        show: Se True, chama ``plt.show()``. Default: True.
        dpi: Resolucao DPI. Default: 300.

    Raises:
        ValueError: Se shapes de ``dtb_true`` e ``dtb_pred`` forem
            incompativeis.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import numpy as np
        >>> dtb_true = np.random.randn(600)
        >>> dtb_pred = dtb_true + np.random.randn(600) * 0.1
        >>> plot_dtb_profile(dtb_true, dtb_pred, show=False)

    Note:
        Matplotlib importado LAZY.
        DTB em dominio escalado (log10 ou normalizado).
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - evaluation/geosteering_report.py: secao 5 (Figuras)
        Ref: docs/ARCHITECTURE_v2.md secao 9.
    """
    # --- Validacao de entrada ---
    if dtb_true.shape != dtb_pred.shape:
        raise ValueError(
            f"Shape mismatch: dtb_true={dtb_true.shape}, "
            f"dtb_pred={dtb_pred.shape}"
        )

    # --- Lazy import matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    # --- Determinar se e amostra unica ou multiplas ---
    is_multi = dtb_true.ndim == 2
    if is_multi:
        # Media e desvio-padrao sobre amostras
        n_samples = dtb_true.shape[0]
        seq_len = dtb_true.shape[1]
        true_mean = np.mean(dtb_true, axis=0)
        true_std = np.std(dtb_true, axis=0)
        pred_mean = np.mean(dtb_pred, axis=0)
        pred_std = np.std(dtb_pred, axis=0)
    else:
        n_samples = 1
        seq_len = dtb_true.shape[0]
        true_mean = dtb_true
        true_std = None
        pred_mean = dtb_pred
        pred_std = None

    logger.info(
        "Plotando DTB profile — N=%d amostras, seq_len=%d",
        n_samples, seq_len,
    )

    # --- Eixo de posicao ---
    if z_obs is not None:
        x_axis = z_obs[:seq_len]
        x_label = "Posicao Lateral (z_obs)"
    else:
        x_axis = np.arange(seq_len)
        x_label = "Profundidade [pontos]"

    # --- Criar figura ---
    fig, ax = plt.subplots(figsize=(12, 5))

    # True DTB
    ax.plot(
        x_axis, true_mean,
        color=_COLOR_DTB_TRUE, linewidth=1.5,
        label="DTB True",
    )

    # Banda de desvio-padrao (true, se multiplas amostras)
    if true_std is not None:
        ax.fill_between(
            x_axis,
            true_mean - true_std,
            true_mean + true_std,
            alpha=0.15,
            color=_COLOR_DTB_TRUE,
        )

    # Predicted DTB
    ax.plot(
        x_axis, pred_mean,
        color=_COLOR_DTB_PRED, linewidth=1.2,
        linestyle="--",
        label="DTB Predicted",
    )

    # Banda de desvio-padrao (predicted, se multiplas amostras)
    if pred_std is not None:
        ax.fill_between(
            x_axis,
            pred_mean - pred_std,
            pred_mean + pred_std,
            alpha=0.15,
            color=_COLOR_DTB_PRED,
        )

    # --- Labels e formatacao ---
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("DTB (dominio escalado)", fontsize=12)
    ax.set_title(
        "Distance-to-Boundary — True vs Predicted",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("DTB profile salvo em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info(
        "plot_dtb_profile concluido — %d amostras, seq_len=%d",
        n_samples, seq_len,
    )


# ════════════════════════════════════════════════════════════════════════
# GEOSTEERING DASHBOARD — 4 paineis com metricas operacionais
#
# Painel com 4 quadrantes para visao geral do desempenho geosteering:
#
# ┌─────────────────────────────────┬─────────────────────────────────┐
# │  (1) DTB Error Distribution    │  (2) R2 per Component           │
# │  Histograma de |dtb_err|       │  Bar chart: rho_h, rho_v, DTB  │
# ├─────────────────────────────────┼─────────────────────────────────┤
# │  (3) Latency Profile           │  (4) Confidence Level           │
# │  Latencia em ms                │  Bar: look-ahead accuracy       │
# └─────────────────────────────────┴─────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_geosteering_dashboard(
    metrics: Dict[str, Any],
    *,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C72: Dashboard de 4 paineis com metricas de geosteering.

    Gera um painel com 4 subplots organizados em grade 2x2:

        (1) DTB Error: histograma da distribuicao de erros de DTB
        (2) R2 Evolution: barras de R2 por componente
        (3) Latency: indicador de latencia de inferencia
        (4) Confidence: acuracia de look-ahead e deteccao de interfaces

    Todos os campos sao opcionais no dicionario ``metrics``. Paineis
    sem dados exibem mensagem informativa.

    Args:
        metrics: Dicionario com metricas de geosteering. Chaves esperadas:
            - "dtb_errors": np.ndarray 1D com erros DTB absolutos
            - "r2_rh": float, R2 para rho_h
            - "r2_rv": float, R2 para rho_v
            - "r2_global": float, R2 global
            - "latency_ms": float, latencia media em ms
            - "look_ahead_accuracy": float, fracao [0, 1]
            - "n_interfaces_detected": int
            - "n_interfaces_total": int
        save_path: Caminho para salvar figura (None = nao salva).
        show: Se True, chama ``plt.show()``. Default: True.
        dpi: Resolucao DPI. Default: 300.

    Raises:
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> metrics_dict = {
        ...     "r2_rh": 0.95, "r2_rv": 0.92, "r2_global": 0.935,
        ...     "latency_ms": 12.5, "look_ahead_accuracy": 0.87,
        ...     "n_interfaces_detected": 17, "n_interfaces_total": 20,
        ... }
        >>> plot_geosteering_dashboard(metrics_dict, show=False)

    Note:
        Matplotlib importado LAZY.
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - evaluation/geosteering_report.py: secao 5 (Figuras)
        Ref: docs/ARCHITECTURE_v2.md secao 9.
    """
    # --- Lazy import matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    logger.info(
        "Plotando geosteering dashboard — %d chaves de metricas",
        len(metrics),
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Painel 1: DTB Error Distribution (top-left) ─────────────────
    ax1 = axes[0, 0]
    dtb_errors = metrics.get("dtb_errors")
    if dtb_errors is not None and isinstance(dtb_errors, np.ndarray) and dtb_errors.size > 0:
        ax1.hist(
            dtb_errors.flatten(), bins=40, density=True,
            alpha=0.7, color=_COLOR_ERROR, edgecolor="white", linewidth=0.5,
        )
        mu_dtb = float(np.mean(dtb_errors))
        std_dtb = float(np.std(dtb_errors))
        ax1.axvline(
            x=mu_dtb, color="red", linewidth=1.5, linestyle="--",
            label=f"mean={mu_dtb:.4f}",
        )
        ax1.set_xlabel("DTB Error (absoluto)", fontsize=10)
        ax1.set_ylabel("Densidade", fontsize=10)
        ax1.set_title(
            f"DTB Error Distribution\n(mean={mu_dtb:.4f}, std={std_dtb:.4f})",
            fontsize=11, fontweight="bold",
        )
        ax1.legend(fontsize=8)
    else:
        ax1.text(
            0.5, 0.5, "DTB Error\nnao disponivel",
            ha="center", va="center", fontsize=12,
            transform=ax1.transAxes, color="gray",
        )
        ax1.set_title("DTB Error Distribution", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # ── Painel 2: R2 per Component (top-right) ──────────────────────
    ax2 = axes[0, 1]
    r2_labels = []
    r2_values = []
    for key, label in [("r2_rh", "rho_h"), ("r2_rv", "rho_v"), ("r2_global", "Global")]:
        val = metrics.get(key)
        if val is not None:
            r2_labels.append(label)
            r2_values.append(float(val))

    if r2_values:
        bars = ax2.bar(
            r2_labels, r2_values,
            color=[_COLOR_PRED, _COLOR_DTB_PRED, _COLOR_LATENCY][:len(r2_values)],
            alpha=0.8, edgecolor="white", linewidth=1.2,
        )
        # Anotacoes de valor em cada barra
        for bar_obj, val in zip(bars, r2_values):
            ax2.text(
                bar_obj.get_x() + bar_obj.get_width() / 2.0,
                bar_obj.get_height() + 0.01,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax2.set_ylim(0.0, min(1.1, max(r2_values) + 0.15))
        ax2.set_ylabel("R2", fontsize=10)
    else:
        ax2.text(
            0.5, 0.5, "R2 metricas\nnao disponiveis",
            ha="center", va="center", fontsize=12,
            transform=ax2.transAxes, color="gray",
        )
    ax2.set_title("R2 por Componente", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # ── Painel 3: Latency (bottom-left) ─────────────────────────────
    ax3 = axes[1, 0]
    latency = metrics.get("latency_ms")
    if latency is not None and float(latency) >= 0:
        lat_val = float(latency)
        # Gauge-style: barra horizontal unica
        ax3.barh(
            ["Latencia"], [lat_val],
            color=_COLOR_LATENCY, alpha=0.8,
            edgecolor="white", linewidth=1.2,
            height=0.5,
        )
        ax3.text(
            lat_val + 0.5, 0, f"{lat_val:.1f} ms",
            ha="left", va="center", fontsize=11, fontweight="bold",
        )
        ax3.set_xlabel("Latencia (ms)", fontsize=10)
        # Limites horizontais com margem
        ax3.set_xlim(0, max(lat_val * 1.5, 10.0))
    else:
        ax3.text(
            0.5, 0.5, "Latencia\nnao disponivel",
            ha="center", va="center", fontsize=12,
            transform=ax3.transAxes, color="gray",
        )
    ax3.set_title("Latencia de Inferencia", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="x")

    # ── Painel 4: Confidence / Look-Ahead (bottom-right) ────────────
    ax4 = axes[1, 1]
    la_accuracy = metrics.get("look_ahead_accuracy")
    n_detected = metrics.get("n_interfaces_detected")
    n_total = metrics.get("n_interfaces_total")

    if la_accuracy is not None:
        la_val = float(la_accuracy)
        categories = ["Look-Ahead Accuracy"]
        values = [la_val]
        colors = [_COLOR_CONFIDENCE]

        bars4 = ax4.bar(
            categories, values,
            color=colors, alpha=0.8,
            edgecolor="white", linewidth=1.2,
        )
        for bar_obj, val in zip(bars4, values):
            ax4.text(
                bar_obj.get_x() + bar_obj.get_width() / 2.0,
                bar_obj.get_height() + 0.01,
                f"{val:.2%}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        # Informacao de interfaces detectadas
        if n_detected is not None and n_total is not None:
            ax4.text(
                0.5, 0.15,
                f"Interfaces: {n_detected}/{n_total}",
                ha="center", va="center", fontsize=10,
                transform=ax4.transAxes, color="gray",
            )

        ax4.set_ylim(0.0, 1.15)
        ax4.set_ylabel("Accuracy", fontsize=10)
    else:
        ax4.text(
            0.5, 0.5, "Look-Ahead Accuracy\nnao disponivel",
            ha="center", va="center", fontsize=12,
            transform=ax4.transAxes, color="gray",
        )
    ax4.set_title("Nivel de Confianca", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    # ── Titulo geral e layout ────────────────────────────────────────
    fig.suptitle(
        "Geosteering Dashboard",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("Geosteering dashboard salvo em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_geosteering_dashboard concluido — %d paineis", 4)
