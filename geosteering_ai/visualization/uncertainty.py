# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/uncertainty.py                                      ║
# ║  Bloco: 9 — Visualization                                                ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • plot_uncertainty_histograms: histogramas de residuos com fit Gauss   ║
# ║    • plot_confidence_bands: perfil com banda de confianca 95%             ║
# ║    • plot_calibration_curve: cobertura esperada vs observada              ║
# ║                                                                            ║
# ║  Dependencias: numpy, matplotlib (lazy), scipy.stats (lazy)              ║
# ║                inference/uncertainty.py (UncertaintyResult via            ║
# ║                TYPE_CHECKING)                                              ║
# ║  Exports: ~3 funcoes — ver __all__                                        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C61: UQ visualization)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Visualizacao de incerteza — histogramas, bandas de confianca, calibracao.

Funcoes de plotagem para quantificacao de incerteza em inversao geofisica:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  VISUALIZACOES DE INCERTEZA                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. Histogramas de residuos (plot_uncertainty_histograms)               │
    │     2 subplots lado a lado (rho_h, rho_v), fit Gaussiano overlay       │
    │     Residuos = y_true - y_pred (em log10 decades)                      │
    │                                                                         │
    │  2. Bandas de confianca (plot_confidence_bands)                         │
    │     Perfil 1D: true (preto), mean (azul), +/- 2sigma (azul claro)     │
    │     2 subplots: rho_h e rho_v para uma amostra                         │
    │                                                                         │
    │  3. Curva de calibracao (plot_calibration_curve)                        │
    │     X = nivel de confianca esperado (50%, 68%, 95%, 99%)               │
    │     Y = fracao real de valores dentro dos limites                       │
    │     Diagonal = calibracao perfeita                                      │
    └─────────────────────────────────────────────────────────────────────────┘

Matplotlib e scipy.stats sao importados de forma lazy dentro de cada funcao
para suportar ambientes headless e minimizar dependencias no import.

Example:
    >>> from geosteering_ai.visualization.uncertainty import (
    ...     plot_uncertainty_histograms,
    ...     plot_confidence_bands,
    ...     plot_calibration_curve,
    ... )
    >>> plot_uncertainty_histograms(y_true, y_pred, show=False)
    >>> plot_confidence_bands(y_true, y_pred, unc_result, sample_idx=0)
    >>> plot_calibration_curve(y_true, y_pred_mean, y_pred_std)

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Matplotlib importado LAZY dentro de cada funcao.
    Referenciado em:
        - visualization/__init__.py: re-exportado como API publica
        - inference/uncertainty.py: UncertaintyResult consumido aqui
        - tests/test_visualization_uncertainty.py: testes de plotagem
    Ref: docs/ARCHITECTURE_v2.md secao 9 (Visualization).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.inference.uncertainty import UncertaintyResult

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Histogramas de residuos ---
    "plot_uncertainty_histograms",
    # --- Bandas de confianca ---
    "plot_confidence_bands",
    # --- Curva de calibracao ---
    "plot_calibration_curve",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de visualizacao de incerteza
# ──────────────────────────────────────────────────────────────────────

# Nomes default das componentes de resistividade
# Canal 0 = rho_h (horizontal), Canal 1 = rho_v (vertical)
_DEFAULT_COMPONENT_NAMES: Tuple[str, ...] = ("rho_h", "rho_v")

# Resolucao default para salvamento de figuras (DPI)
_DEFAULT_DPI = 300

# Numero default de bins para histogramas
_DEFAULT_N_BINS_HIST = 50

# Niveis de confianca para curva de calibracao (%)
# Padrao: 50%, 68% (~1sigma), 90%, 95% (~2sigma), 99% (~3sigma)
_DEFAULT_CONFIDENCE_LEVELS: Tuple[float, ...] = (
    0.50, 0.68, 0.80, 0.90, 0.95, 0.99,
)

# Cores para consistencia visual
_COLOR_TRUE = "black"
_COLOR_MEAN = "#1f77b4"        # azul matplotlib default
_COLOR_CI_BAND = "#aec7e8"     # azul claro para banda CI
_COLOR_HIST_RHO_H = "#1f77b4"  # azul para rho_h
_COLOR_HIST_RHO_V = "#d62728"  # vermelho para rho_v
_COLOR_GAUSS_FIT = "#2ca02c"   # verde para fit gaussiano
_COLOR_CALIBRATION = "#ff7f0e" # laranja para curva de calibracao
_COLOR_DIAGONAL = "gray"       # diagonal de referencia

# Epsilon para estabilidade numerica (float32)
_EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════
# HISTOGRAMAS DE RESIDUOS — 2 subplots com fit Gaussiano
#
# Residuos = y_true - y_pred (em log10 decades).
# Cada componente (rho_h, rho_v) em subplot separado.
# Overlay: curva Gaussiana ajustada (mu, sigma) via scipy.stats.norm.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  [  Histograma rho_h  ] [  Histograma rho_v  ]                    │
# │  Residuo (decades)       Residuo (decades)                         │
# │  mu=0.002, s=0.081       mu=-0.001, s=0.094                       │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_uncertainty_histograms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    component_names: Sequence[str] = _DEFAULT_COMPONENT_NAMES,
    n_bins: int = _DEFAULT_N_BINS_HIST,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C61: Histogramas de residuos por componente com fit Gaussiano.

    Gera 2 subplots lado a lado, um por componente de resistividade
    (rho_h, rho_v). Cada histograma mostra a distribuicao dos residuos
    (y_true - y_pred) em log10 decades, com overlay de curva Gaussiana
    ajustada (media, desvio-padrao).

    Args:
        y_true: Array (N, seq_len, n_channels) com resistividade
            verdadeira em log10. Canal 0 = rho_h, canal 1 = rho_v.
        y_pred: Array (N, seq_len, n_channels) com resistividade
            predita em log10. Mesmo shape de ``y_true``.
        component_names: Nomes das componentes para labels dos subplots.
            Default: ``("rho_h", "rho_v")``.
        n_bins: Numero de bins do histograma. Default: 50.
        save_path: Caminho completo para salvar figura (None = nao salva).
            Exemplo: ``"/path/to/residual_histograms.png"``.
        show: Se True, chama ``plt.show()`` ao final. Default: True.
        dpi: Resolucao em DPI para salvamento. Default: 300.

    Raises:
        ValueError: Se shapes de ``y_true`` e ``y_pred`` forem incompativeis.
        ValueError: Se ``y_true`` nao for 3D.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import numpy as np
        >>> y_true = np.random.randn(100, 600, 2)
        >>> y_pred = y_true + np.random.randn(100, 600, 2) * 0.1
        >>> plot_uncertainty_histograms(y_true, y_pred, show=False)

    Note:
        Matplotlib e scipy.stats importados LAZY.
        Residuos em log10 decades: 0.1 = ~0.1 decada = ~26% em Ohm.m.
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - evaluation/metrics.py: complementa MetricsReport com visual
        Ref: docs/ARCHITECTURE_v2.md secao 9.
    """
    # --- Validacao de entrada ---
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if y_true.ndim != 3:
        raise ValueError(
            f"Esperado array 3D (N, seq_len, channels), "
            f"recebido ndim={y_true.ndim}, shape={y_true.shape}"
        )

    n_channels = y_true.shape[-1]

    # --- Lazy imports ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    try:
        from scipy.stats import norm
    except ImportError:
        logger.warning(
            "scipy nao instalado — fit Gaussiano desabilitado. "
            "Instale com: pip install scipy"
        )
        norm = None  # type: ignore[assignment]

    logger.info(
        "Plotando histogramas de residuos — %d canais, %d amostras",
        n_channels, y_true.shape[0],
    )

    # --- Cores por componente ---
    channel_colors = [_COLOR_HIST_RHO_H, _COLOR_HIST_RHO_V]
    # Extender caso haja mais canais que cores default
    while len(channel_colors) < n_channels:
        channel_colors.append("#9467bd")  # roxo fallback

    # --- Criar figura ---
    fig, axes = plt.subplots(
        1, n_channels,
        figsize=(6 * n_channels, 5),
        squeeze=False,
    )

    for ch in range(n_channels):
        ax = axes[0, ch]

        # Residuos = true - pred (flatten para histograma)
        residuals = (y_true[:, :, ch] - y_pred[:, :, ch]).flatten()

        # Histograma normalizado (densidade)
        ax.hist(
            residuals,
            bins=n_bins,
            density=True,
            alpha=0.7,
            color=channel_colors[ch],
            edgecolor="white",
            linewidth=0.5,
            label="Residuos",
        )

        # Fit Gaussiano overlay
        mu = float(np.mean(residuals))
        sigma = float(np.std(residuals))
        # Garantir sigma > 0 para fit
        sigma = max(sigma, _EPS)

        if norm is not None:
            x_fit = np.linspace(
                mu - 4 * sigma, mu + 4 * sigma, 200,
            )
            y_fit = norm.pdf(x_fit, loc=mu, scale=sigma)
            ax.plot(
                x_fit, y_fit,
                color=_COLOR_GAUSS_FIT,
                linewidth=2,
                linestyle="--",
                label=f"Gauss (mu={mu:.4f}, s={sigma:.4f})",
            )

        # Linha vertical em zero (residuo ideal)
        ax.axvline(
            x=0, color="gray", linewidth=0.8, linestyle=":",
            alpha=0.6,
        )

        # Labels
        name = (
            component_names[ch]
            if ch < len(component_names)
            else f"canal_{ch}"
        )
        ax.set_xlabel(f"Residuo log10 [{name}] (decades)")
        ax.set_ylabel("Densidade")
        ax.set_title(
            f"Residuos — {name}\n"
            f"(mu={mu:.4f}, sigma={sigma:.4f}, N={len(residuals):,})"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Distribuicao de Residuos por Componente",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("Histograma salvo em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_uncertainty_histograms concluido — %d canais", n_channels)


# ════════════════════════════════════════════════════════════════════════
# BANDAS DE CONFIANCA — Perfil 1D com CI 95%
#
# Para uma amostra selecionada, plota perfil de resistividade com
# banda de confianca (mean +/- 2*sigma) sombreada em azul claro.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  [   rho_h com CI 95%   ] [   rho_v com CI 95%   ]               │
# │  ─── True (preto)         ─── True (preto)                        │
# │  ─── Mean (azul)          ─── Mean (azul)                         │
# │  ░░░ CI 95% (azul claro)  ░░░ CI 95% (azul claro)                │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_confidence_bands(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty_result: UncertaintyResult,
    *,
    sample_idx: int = 0,
    component_names: Sequence[str] = _DEFAULT_COMPONENT_NAMES,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C61: Perfil de resistividade com bandas de confianca 95%.

    Plota para uma amostra selecionada:
        - True: perfil verdadeiro em preto solido
        - Mean: predicao media (MC Dropout ou Ensemble) em azul
        - CI 95%: banda sombreada (ci_lower a ci_upper) em azul claro

    2 subplots lado a lado: rho_h (esquerda) e rho_v (direita).

    Args:
        y_true: Array (N, seq_len, n_channels) com resistividade
            verdadeira em log10.
        y_pred: Array (N, seq_len, n_channels) com predicao pontual
            em log10. Usado apenas para referencia (nao plotado se
            ``uncertainty_result.mean`` esta disponivel).
        uncertainty_result: UncertaintyResult contendo mean, std,
            ci_lower, ci_upper. Obtido via UncertaintyEstimator.
        sample_idx: Indice da amostra a plotar. Default: 0.
        component_names: Nomes das componentes. Default: ``("rho_h", "rho_v")``.
        save_path: Caminho para salvar figura (None = nao salva).
        show: Se True, chama ``plt.show()``. Default: True.
        dpi: Resolucao DPI. Default: 300.

    Raises:
        ValueError: Se ``sample_idx`` estiver fora do range.
        ValueError: Se shapes forem incompativeis entre y_true e uncertainty_result.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> from geosteering_ai.inference.uncertainty import UncertaintyEstimator
        >>> estimator = UncertaintyEstimator(method="mc_dropout")
        >>> result = estimator.estimate(model, x_test, n_samples=30)
        >>> plot_confidence_bands(y_true, y_pred, result, sample_idx=5)

    Note:
        Matplotlib importado LAZY.
        CI 95% provem de UncertaintyResult (mean +/- 1.96*std).
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - inference/uncertainty.py: UncertaintyResult consumido aqui
        Ref: docs/ARCHITECTURE_v2.md secao 9.
    """
    # --- Validacao de entrada ---
    n_samples_available = y_true.shape[0]
    if sample_idx < 0 or sample_idx >= n_samples_available:
        raise ValueError(
            f"sample_idx={sample_idx} fora do range [0, {n_samples_available - 1}]"
        )

    if y_true.shape != uncertainty_result.mean.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"uncertainty mean={uncertainty_result.mean.shape}"
        )

    # --- Lazy import ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    n_channels = y_true.shape[-1]
    seq_len = y_true.shape[1]
    depth_axis = np.arange(seq_len)  # eixo de profundidade (pontos)

    logger.info(
        "Plotando bandas de confianca — amostra %d, %d canais, "
        "method=%s, n_forward=%d",
        sample_idx, n_channels,
        uncertainty_result.method, uncertainty_result.n_samples,
    )

    # --- Extrair dados da amostra selecionada ---
    true_sample = y_true[sample_idx]                        # (seq, ch)
    mean_sample = uncertainty_result.mean[sample_idx]        # (seq, ch)
    ci_lo_sample = uncertainty_result.ci_lower[sample_idx]   # (seq, ch)
    ci_hi_sample = uncertainty_result.ci_upper[sample_idx]   # (seq, ch)
    std_sample = uncertainty_result.std[sample_idx]          # (seq, ch)

    # --- Criar figura ---
    fig, axes = plt.subplots(
        1, n_channels,
        figsize=(6 * n_channels, 8),
        sharey=True,
        squeeze=False,
    )

    for ch in range(n_channels):
        ax = axes[0, ch]

        # Banda CI 95% (sombreada)
        ax.fill_betweenx(
            depth_axis,
            ci_lo_sample[:, ch],
            ci_hi_sample[:, ch],
            alpha=0.3,
            color=_COLOR_CI_BAND,
            label="CI 95%",
        )

        # True (preto solido)
        ax.plot(
            true_sample[:, ch], depth_axis,
            color=_COLOR_TRUE, linewidth=1.5,
            label="True",
        )

        # Mean prediction (azul)
        ax.plot(
            mean_sample[:, ch], depth_axis,
            color=_COLOR_MEAN, linewidth=1.2, linestyle="--",
            label=f"Mean ({uncertainty_result.method})",
        )

        # Labels
        name = (
            component_names[ch]
            if ch < len(component_names)
            else f"canal_{ch}"
        )
        ax.set_xlabel(r"$\log_{10}(\rho)$ [$\Omega\cdot$m]")
        if ch == 0:
            ax.set_ylabel("Profundidade [pontos]")
        ax.set_title(
            f"{name} — mean std={float(np.mean(std_sample[:, ch])):.4f}"
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.invert_yaxis()  # profundidade cresce para baixo
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Bandas de Confianca 95% — Amostra {sample_idx}\n"
        f"({uncertainty_result.method}, "
        f"n={uncertainty_result.n_samples} forward passes)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("Bandas de confianca salvas em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info(
        "plot_confidence_bands concluido — amostra %d, method=%s",
        sample_idx, uncertainty_result.method,
    )


# ════════════════════════════════════════════════════════════════════════
# CURVA DE CALIBRACAO — Cobertura esperada vs observada
#
# Para cada nivel de confianca (50%, 68%, 95%, 99%), computa a fracao
# real de valores verdadeiros dentro dos limites previstos.
# Calibracao perfeita: diagonal (expected = observed).
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  1.0 ┤                                          ╱ diagonal        │
# │      │                                       ╱                     │
# │  0.8 ┤                                    ╱                        │
# │      │                     ●──────── ╱                             │
# │  0.6 ┤               ●────────╱                                   │
# │      │          ●────────╱                                         │
# │  0.4 ┤     ●────────╱                                             │
# │      │  ╱                                                          │
# │  0.2 ┤╱                  ● = observado                             │
# │      │                                                             │
# │  0.0 ├────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤          │
# │      0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0      │
# │                 Cobertura Esperada                                 │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    *,
    confidence_levels: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C61: Curva de calibracao — cobertura esperada vs observada.

    Para cada nivel de confianca, computa o intervalo previsto
    (mean +/- z * std) e verifica a fracao de valores verdadeiros
    contidos nesse intervalo. Calibracao perfeita: linha diagonal.

    Um modelo sobre-confiante fica ABAIXO da diagonal (observado < esperado).
    Um modelo sub-confiante fica ACIMA da diagonal (observado > esperado).

    Args:
        y_true: Array (N, seq_len, n_channels) com valores verdadeiros
            em log10.
        y_pred_mean: Array (N, seq_len, n_channels) com media das
            predicoes em log10. Pode ser obtido de UncertaintyResult.mean.
        y_pred_std: Array (N, seq_len, n_channels) com desvio-padrao
            das predicoes em log10. De UncertaintyResult.std.
        confidence_levels: Sequencia de niveis de confianca esperados
            (entre 0 e 1). Default: ``(0.50, 0.68, 0.80, 0.90, 0.95, 0.99)``.
        save_path: Caminho para salvar figura (None = nao salva).
        show: Se True, chama ``plt.show()``. Default: True.
        dpi: Resolucao DPI. Default: 300.

    Raises:
        ValueError: Se shapes forem incompativeis entre y_true e y_pred_mean.
        ValueError: Se shapes forem incompativeis entre y_pred_mean e y_pred_std.
        ImportError: Se matplotlib nao estiver instalado.
        ImportError: Se scipy nao estiver instalado.

    Example:
        >>> result = estimator.estimate(model, x_test, n_samples=50)
        >>> plot_calibration_curve(
        ...     y_true, result.mean, result.std, show=False,
        ... )

    Note:
        Matplotlib e scipy.stats importados LAZY.
        Usa scipy.stats.norm.ppf(alpha) para converter nivel de confianca
        em fator z (quantil da Normal padrao).
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - inference/uncertainty.py: UncertaintyResult.mean/.std
        Ref: docs/ARCHITECTURE_v2.md secao 9.
        Ref: Kuleshov, V. et al. (2018). Accurate Uncertainties for Deep
            Learning Using Calibrated Regression. ICML.
    """
    # --- Validacao de entrada ---
    if y_true.shape != y_pred_mean.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"y_pred_mean={y_pred_mean.shape}"
        )
    if y_pred_mean.shape != y_pred_std.shape:
        raise ValueError(
            f"Shape mismatch: y_pred_mean={y_pred_mean.shape}, "
            f"y_pred_std={y_pred_std.shape}"
        )

    # --- Lazy imports ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    try:
        from scipy.stats import norm
    except ImportError:
        logger.error(
            "scipy nao instalado. Instale com: pip install scipy. "
            "Necessario para norm.ppf() na curva de calibracao."
        )
        raise

    # --- Niveis de confianca ---
    if confidence_levels is None:
        confidence_levels = _DEFAULT_CONFIDENCE_LEVELS

    logger.info(
        "Plotando curva de calibracao — %d niveis, shape=%s",
        len(confidence_levels), y_true.shape,
    )

    # --- Computar cobertura observada para cada nivel ---
    # Clampar std para evitar divisao por zero
    std_safe = np.maximum(y_pred_std, _EPS)

    observed_coverages: List[float] = []
    for level in confidence_levels:
        # Fator z: para CI de nivel `level`, o quantil e (1+level)/2
        # Exemplo: level=0.95 → alpha=0.975 → z=1.96
        alpha = (1.0 + level) / 2.0
        z = float(norm.ppf(alpha))

        # Limites do intervalo de confianca
        lower = y_pred_mean - z * std_safe
        upper = y_pred_mean + z * std_safe

        # Fracao de valores verdadeiros dentro dos limites
        inside = np.logical_and(y_true >= lower, y_true <= upper)
        coverage = float(np.mean(inside))
        observed_coverages.append(coverage)

        logger.debug(
            "Nivel %.0f%%: z=%.3f, cobertura observada=%.4f",
            level * 100, z, coverage,
        )

    expected = list(confidence_levels)
    observed = observed_coverages

    # --- Criar figura ---
    fig, ax = plt.subplots(figsize=(7, 7))

    # Diagonal de referencia (calibracao perfeita)
    ax.plot(
        [0, 1], [0, 1],
        color=_COLOR_DIAGONAL, linewidth=1.5, linestyle="--",
        label="Calibracao perfeita",
        zorder=1,
    )

    # Curva de calibracao observada
    ax.plot(
        expected, observed,
        color=_COLOR_CALIBRATION, linewidth=2.5,
        marker="o", markersize=8,
        label="Observado",
        zorder=2,
    )

    # Anotacoes de cada ponto
    for exp, obs in zip(expected, observed):
        ax.annotate(
            f"  {obs:.2%}",
            xy=(exp, obs),
            fontsize=8,
            color=_COLOR_CALIBRATION,
        )

    # Labels e formatacao
    ax.set_xlabel("Cobertura Esperada", fontsize=12)
    ax.set_ylabel("Cobertura Observada", fontsize=12)
    ax.set_title(
        "Curva de Calibracao — Incerteza Preditiva",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("Curva de calibracao salva em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info(
        "plot_calibration_curve concluido — %d niveis de confianca",
        len(confidence_levels),
    )
