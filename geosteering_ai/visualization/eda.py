# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/eda.py                                              ║
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
# ║    • Analise exploratoria de dados (EDA) para o pipeline de inversao      ║
# ║    • Distribuicoes de features (histogramas)                              ║
# ║    • Matriz de correlacao (heatmap)                                       ║
# ║    • Boxplots por feature                                                 ║
# ║    • Serie temporal de uma sequencia exemplo                              ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), numpy, matplotlib (lazy)      ║
# ║  Exports: ~1 (plot_eda_summary)                                           ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9.3                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C15/C26)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Analise exploratoria de dados — distribuicoes, correlacoes, series.

Gera 4 paineis de EDA para dados de entrada do pipeline de inversao:
    1. Histogramas de distribuicao por feature
    2. Matriz de correlacao (heatmap)
    3. Boxplots para detectar outliers por feature
    4. Serie temporal de uma sequencia exemplo

Aceita dados 2D (n_samples, n_features) ou 3D (n_samples, seq_len, n_features).
No caso 3D, as features sao agregadas (flatten ou media) conforme o painel.

Example:
    >>> from geosteering_ai.visualization import plot_eda_summary
    >>> import numpy as np
    >>> data = np.random.randn(100, 600, 5)
    >>> plot_eda_summary(data, feature_names=["Hxx", "Hyy", "Hzz", "GS1", "GS2"])

Note:
    Matplotlib importado de forma lazy (suporta ambientes headless).
    Referenciado em:
        - data/pipeline.py: EDA pos-loading
        - notebooks: analise exploratoria interativa
    Ref: docs/ARCHITECTURE_v2.md secao 9.3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Funcao principal ---
    "plot_eda_summary",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de visualizacao
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_DPI = 300
_HIST_BINS = 50
_FIGSIZE_SUMMARY = (16, 12)
_CMAP_CORR = "coolwarm"     # colormap para matriz de correlacao
_MAX_FEATURES_BOXPLOT = 20  # limite de features no boxplot para legibilidade


# ──────────────────────────────────────────────────────────────────────
# D2: Funcoes auxiliares — preparacao de dados para EDA
# ──────────────────────────────────────────────────────────────────────
def _flatten_to_2d(data: np.ndarray) -> np.ndarray:
    """Achata dados 3D (n, seq, feat) para 2D (n*seq, feat).

    Para dados ja 2D, retorna sem modificacao.

    Args:
        data: Array 2D (n_samples, n_features) ou
            3D (n_samples, seq_len, n_features).

    Returns:
        Array 2D (n_total, n_features) achatado.

    Note:
        Ref: utils/validation.py para validacao de shapes.
    """
    if data.ndim == 3:
        n_samples, seq_len, n_features = data.shape
        return data.reshape(-1, n_features)
    return data


def _generate_feature_names(n_features: int) -> List[str]:
    """Gera nomes genericos para features sem nome.

    Args:
        n_features: Numero de features.

    Returns:
        Lista de nomes no formato ["feat_0", "feat_1", ...].

    Note:
        Utilizado quando feature_names nao e fornecido pelo usuario.
    """
    return [f"feat_{i}" for i in range(n_features)]


# ──────────────────────────────────────────────────────────────────────
# D2: Funcao principal — sumario EDA completo
# ──────────────────────────────────────────────────────────────────────
def plot_eda_summary(
    data: np.ndarray,
    *,
    config: Optional[PipelineConfig] = None,
    feature_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> None:
    """Gera sumario visual EDA (Exploratory Data Analysis).

    Produz 4 paineis em uma unica figura:
        1. Histogramas — distribuicao de cada feature (com KDE opcional)
        2. Correlacao — heatmap da matriz de correlacao entre features
        3. Boxplots  — dispersao e outliers por feature
        4. Sequencia — serie temporal de uma amostra exemplo (se 3D)

    Aceita dados 2D (n_samples, n_features) ou 3D (n_samples, seq_len,
    n_features). No caso 3D, histogramas e boxplots usam dados achatados,
    e o painel de sequencia mostra a primeira amostra completa.

    Args:
        data: Array 2D (n_samples, n_features) ou
            3D (n_samples, seq_len, n_features) com dados de entrada.
        config: PipelineConfig opcional para metadados no titulo.
            Se fornecido, feature_view e model_type sao incluidos.
        feature_names: Lista de nomes das features (default None).
            Se None, nomes genericos sao gerados automaticamente.
        save_dir: Diretorio para salvar figuras (None = nao salva).
            Cria o diretorio se nao existir.
        dpi: Resolucao em DPI para salvamento (default 300).
        show: Se True, chama plt.show() ao final (default True).

    Raises:
        ValueError: Se data nao for 2D ou 3D.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100, 600, 5)
        >>> names = ["Hxx_re", "Hyy_re", "Hzz_re", "USD", "UHR"]
        >>> plot_eda_summary(data, feature_names=names, show=False)

    Note:
        Matplotlib importado de forma lazy (suporta ambientes headless).
        Ref: docs/ARCHITECTURE_v2.md secao 9.3.
    """
    # --- Validacao de dimensao ---
    if data.ndim not in (2, 3):
        msg = f"Esperado array 2D ou 3D, recebido ndim={data.ndim}."
        raise ValueError(msg)

    # --- Import lazy do matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    # --- Determinar n_features e nomes ---
    n_features = data.shape[-1]
    if feature_names is None:
        feature_names = _generate_feature_names(n_features)
    elif len(feature_names) != n_features:
        logger.warning(
            "feature_names tem %d nomes, mas dados tem %d features. "
            "Usando nomes genericos.",
            len(feature_names), n_features,
        )
        feature_names = _generate_feature_names(n_features)

    # --- Dados achatados para histogramas/boxplots/correlacao ---
    data_flat = _flatten_to_2d(data)
    logger.info(
        "EDA: shape original=%s, achatado=%s, n_features=%d",
        data.shape, data_flat.shape, n_features,
    )

    # --- Titulo base ---
    base_title = "EDA Summary"
    if config is not None:
        base_title = (
            f"EDA — FV={config.feature_view}, "
            f"Model={config.model_type}"
        )

    # --- Criar figura com 4 paineis (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=_FIGSIZE_SUMMARY)

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Painel 1 (sup-esq): Histogramas de distribuicao              │
    # └────────────────────────────────────────────────────────────────┘
    ax_hist = axes[0, 0]
    for i in range(n_features):
        ax_hist.hist(
            data_flat[:, i],
            bins=_HIST_BINS,
            alpha=0.6,
            label=feature_names[i],
            density=True,  # normalizado para comparacao entre features
        )
    ax_hist.set_xlabel("Valor")
    ax_hist.set_ylabel("Densidade")
    ax_hist.set_title("Distribuicao de Features")
    ax_hist.legend(fontsize=7, loc="upper right", ncol=max(1, n_features // 4))
    ax_hist.grid(True, alpha=0.3)

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Painel 2 (sup-dir): Matriz de correlacao (heatmap)           │
    # └────────────────────────────────────────────────────────────────┘
    ax_corr = axes[0, 1]
    # Correlacao de Pearson entre features
    corr_matrix = np.corrcoef(data_flat.T)  # (n_features, n_features)
    im = ax_corr.imshow(
        corr_matrix,
        cmap=_CMAP_CORR,
        vmin=-1.0, vmax=1.0,
        aspect="auto",
    )
    ax_corr.set_xticks(range(n_features))
    ax_corr.set_yticks(range(n_features))
    ax_corr.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
    ax_corr.set_yticklabels(feature_names, fontsize=7)
    ax_corr.set_title("Matriz de Correlacao")
    fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    # Anotar valores de correlacao nas celulas
    for i in range(n_features):
        for j in range(n_features):
            ax_corr.text(
                j, i, f"{corr_matrix[i, j]:.2f}",
                ha="center", va="center", fontsize=6,
                color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
            )

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Painel 3 (inf-esq): Boxplots por feature                    │
    # └────────────────────────────────────────────────────────────────┘
    ax_box = axes[1, 0]
    n_plot = min(n_features, _MAX_FEATURES_BOXPLOT)
    box_data = [data_flat[:, i] for i in range(n_plot)]
    bp = ax_box.boxplot(
        box_data,
        labels=feature_names[:n_plot],
        patch_artist=True,
        showfliers=True,
        flierprops={"markersize": 2, "alpha": 0.3},
    )
    # Colorir boxes com paleta suave
    colors = plt.cm.Set3(np.linspace(0, 1, n_plot))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax_box.set_xlabel("Feature")
    ax_box.set_ylabel("Valor")
    ax_box.set_title("Boxplots (Outlier Detection)")
    ax_box.tick_params(axis="x", rotation=45)
    ax_box.grid(True, alpha=0.3, axis="y")
    if n_features > _MAX_FEATURES_BOXPLOT:
        logger.warning(
            "Apenas %d de %d features exibidas no boxplot (limite de legibilidade)",
            _MAX_FEATURES_BOXPLOT, n_features,
        )

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Painel 4 (inf-dir): Serie temporal de amostra exemplo        │
    # └────────────────────────────────────────────────────────────────┘
    ax_seq = axes[1, 1]
    if data.ndim == 3:
        # Plotar primeira amostra completa (seq_len pontos por feature)
        sample = data[0]  # (seq_len, n_features)
        seq_axis = np.arange(sample.shape[0])
        for i in range(n_features):
            ax_seq.plot(
                seq_axis, sample[:, i],
                linewidth=0.8, alpha=0.8,
                label=feature_names[i],
            )
        ax_seq.set_xlabel("Ponto na Sequencia")
        ax_seq.set_ylabel("Valor")
        ax_seq.set_title("Serie Temporal — Amostra 0")
        ax_seq.legend(fontsize=7, loc="upper right", ncol=max(1, n_features // 4))
    else:
        # Dados 2D: scatter das primeiras 2 features
        if n_features >= 2:
            ax_seq.scatter(
                data_flat[:, 0], data_flat[:, 1],
                s=1, alpha=0.3, c="steelblue",
            )
            ax_seq.set_xlabel(feature_names[0])
            ax_seq.set_ylabel(feature_names[1])
            ax_seq.set_title(f"Scatter: {feature_names[0]} vs {feature_names[1]}")
        else:
            ax_seq.hist(data_flat[:, 0], bins=_HIST_BINS, color="steelblue")
            ax_seq.set_xlabel(feature_names[0])
            ax_seq.set_title(f"Distribuicao: {feature_names[0]}")
    ax_seq.grid(True, alpha=0.3)

    # --- Layout e titulo global ---
    fig.suptitle(base_title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    # --- Diretorio de salvamento ---
    if save_dir is not None:
        save_path_obj = Path(save_dir)
        save_path_obj.mkdir(parents=True, exist_ok=True)
        fname = save_path_obj / "eda_summary.png"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        logger.info("EDA summary salvo em: %s", fname)

    # --- Exibir se solicitado ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_eda_summary concluido: %d features, %s", n_features, data.shape)
