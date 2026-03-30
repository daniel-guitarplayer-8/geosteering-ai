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
# ║    • Distribuicoes de features (histogramas com KDE, splits sobrepostos)  ║
# ║    • Matriz de correlacao (heatmap com threshold e metodos Pearson/       ║
# ║      Spearman)                                                             ║
# ║    • Perfis de amostra (componentes EM vs profundidade z_obs)             ║
# ║    • Comparacao train/val/test (boxplots para detectar data leakage)      ║
# ║    • Heatmap de sensibilidade (variancia ou gradiente por profundidade)   ║
# ║    • Sumario 4-paineis (histogramas, correlacao, boxplots, serie)         ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), numpy, scipy.stats (opt),     ║
# ║                matplotlib (lazy)                                           ║
# ║  Exports: ~6 funcoes — ver __all__                                        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 21.2 (EDA avancado)                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C15/C26)         ║
# ║    v2.0.0 (2026-03) — Fase IV: 5 funcoes EDA avancado adicionadas       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Analise exploratoria de dados — distribuicoes, correlacoes, perfis, splits, sensibilidade.

Modulo EDA avancado com 6 funcoes para diagnostico do pipeline de inversao:

   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  6 Funcoes EDA (Ref: docs/ARCHITECTURE_v2.md secao 21.2):                  │
   │                                                                              │
   │  Funcao                         │ Proposito                    │ Input      │
   │  ───────────────────────────────┼──────────────────────────────┼────────────│
   │  plot_eda_summary               │ Sumario 4-paineis (v2.0.0)  │ 2D ou 3D   │
   │  plot_feature_distributions     │ Histogramas + KDE overlay    │ 2D/3D/dict │
   │  plot_correlation_heatmap       │ Correlacao com threshold     │ 2D ou 3D   │
   │  plot_sample_profiles           │ Componentes EM vs z_obs      │ 3D only    │
   │  plot_train_val_test_comparison │ Boxplot splits (leakage)     │ dict only  │
   │  plot_sensitivity_heatmap       │ Features x profundidade      │ 3D only    │
   │                                                                              │
   │  Todas aceitam config: PipelineConfig (opcional) para metadados no titulo.  │
   │  Matplotlib importado lazy (headless-safe).                                 │
   │  NUNCA print() — logging.getLogger(__name__).                               │
   └──────────────────────────────────────────────────────────────────────────────┘

Example:
    >>> from geosteering_ai.visualization.eda import plot_eda_summary
    >>> import numpy as np
    >>> data = np.random.randn(100, 600, 5)
    >>> plot_eda_summary(data, feature_names=["z", "ReHxx", "ImHxx", "ReHzz", "ImHzz"])

    >>> from geosteering_ai.visualization.eda import plot_feature_distributions
    >>> splits = {"train": data[:60], "val": data[60:80], "test": data[80:]}
    >>> plot_feature_distributions(splits, show=False)

Note:
    Matplotlib importado de forma lazy (suporta ambientes headless).
    scipy.stats.gaussian_kde usado para KDE overlay (fallback sem scipy).
    Referenciado em:
        - data/pipeline.py: EDA pos-loading
        - notebooks: analise exploratoria interativa
        - training/loop.py: diagnostico pre-treinamento
    Ref: docs/ARCHITECTURE_v2.md secao 21.2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # ── Sumario original (v2.0.0) ────────────────────────────────────
    "plot_eda_summary",
    # ── EDA avancado — Fase IV (5 funcoes) ────────────────────────────
    "plot_feature_distributions",
    "plot_correlation_heatmap",
    "plot_sample_profiles",
    "plot_train_val_test_comparison",
    "plot_sensitivity_heatmap",
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
_CMAP_CORR = "coolwarm"  # colormap para matriz de correlacao
_CMAP_SENSITIVITY = "viridis"  # colormap para heatmap de sensibilidade
_MAX_FEATURES_BOXPLOT = 20  # limite de features no boxplot para legibilidade
_DEFAULT_N_SAMPLES = 3  # amostras exemplo para plot_sample_profiles
_SPLIT_COLORS = {  # cores consistentes para splits train/val/test
    "train": "#1f77b4",  # azul — padrao matplotlib tab:blue
    "val": "#ff7f0e",  # laranja — tab:orange
    "test": "#2ca02c",  # verde — tab:green
}
_VALID_CORR_METHODS = {"pearson", "spearman"}
_VALID_SENSITIVITY_METRICS = {"variance", "gradient"}


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
        logger.error("matplotlib nao instalado. Instale com: pip install matplotlib")
        raise

    # --- Determinar n_features e nomes ---
    n_features = data.shape[-1]
    if feature_names is None:
        feature_names = _generate_feature_names(n_features)
    elif len(feature_names) != n_features:
        logger.warning(
            "feature_names tem %d nomes, mas dados tem %d features. "
            "Usando nomes genericos.",
            len(feature_names),
            n_features,
        )
        feature_names = _generate_feature_names(n_features)

    # --- Dados achatados para histogramas/boxplots/correlacao ---
    data_flat = _flatten_to_2d(data)
    logger.info(
        "EDA: shape original=%s, achatado=%s, n_features=%d",
        data.shape,
        data_flat.shape,
        n_features,
    )

    # --- Titulo base ---
    base_title = "EDA Summary"
    if config is not None:
        base_title = f"EDA — FV={config.feature_view}, " f"Model={config.model_type}"

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
        vmin=-1.0,
        vmax=1.0,
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
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
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
            _MAX_FEATURES_BOXPLOT,
            n_features,
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
                seq_axis,
                sample[:, i],
                linewidth=0.8,
                alpha=0.8,
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
                data_flat[:, 0],
                data_flat[:, 1],
                s=1,
                alpha=0.3,
                c="steelblue",
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


# ════════════════════════════════════════════════════════════════════════════
# SECAO: HELPERS COMPARTILHADOS (Fase IV)
# ════════════════════════════════════════════════════════════════════════════
# Funcoes auxiliares reutilizadas pelas 5 funcoes EDA avancado.
# Isolam logica comum de resolucao de entrada, lazy import, e
# finalizacao de figuras (save/show/close).
# Ref: DRY — evitar duplicacao de boilerplate entre 5 funcoes.
# ──────────────────────────────────────────────────────────────────────────


def _lazy_import_plt():
    """Import lazy de matplotlib.pyplot com mensagem de erro util.

    Returns:
        modulo matplotlib.pyplot.

    Raises:
        ImportError: Se matplotlib nao estiver instalado.

    Note:
        Chamado no inicio de cada funcao de plot.
        Suporta ambientes headless (Agg backend).
    """
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        logger.error("matplotlib nao instalado. Instale com: pip install matplotlib")
        raise


def _resolve_feature_names(
    n_features: int,
    feature_names: Optional[List[str]] = None,
) -> List[str]:
    """Resolve nomes de features: usa fornecidos ou gera genericos.

    Args:
        n_features: Numero de features nos dados.
        feature_names: Lista fornecida pelo usuario (pode ser None).

    Returns:
        Lista de nomes com exatamente n_features elementos.

    Note:
        Se feature_names tem tamanho diferente de n_features,
        loga warning e gera nomes genericos.
    """
    if feature_names is None:
        return _generate_feature_names(n_features)
    if len(feature_names) != n_features:
        logger.warning(
            "feature_names tem %d nomes, mas dados tem %d features. "
            "Usando nomes genericos.",
            len(feature_names),
            n_features,
        )
        return _generate_feature_names(n_features)
    return list(feature_names)


def _finalize_figure(
    fig,
    plt_module,
    *,
    save_dir: Optional[str],
    filename: str,
    dpi: int,
    show: bool,
) -> None:
    """Finaliza figura: tight_layout + save + show/close.

    Args:
        fig: Figure matplotlib a finalizar.
        plt_module: Modulo matplotlib.pyplot (para show/close).
        save_dir: Diretorio para salvar (None = nao salva).
        filename: Nome do arquivo PNG (sem diretorio).
        dpi: Resolucao para salvamento.
        show: Se True, chama plt.show(); se False, fecha figura.

    Note:
        Cria diretorio se nao existir.
        Ref: Padrao reutilizado em todas as 6 funcoes EDA.
    """
    fig.tight_layout()
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fpath = save_path / filename
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        logger.info("Figura salva em: %s", fpath)
    if show:
        plt_module.show()
    else:
        plt_module.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCAO 1 — plot_feature_distributions
# ════════════════════════════════════════════════════════════════════════════
# Histogramas com KDE overlay para cada feature. Aceita um unico array
# (2D ou 3D) ou um dicionario de splits {"train": ..., "val": ..., "test": ...}.
# Quando dict, cada split eh sobreposto com cor diferente para detectar
# distribuicoes deslocadas (indicador de data leakage ou viés de split).
#
# KDE (Kernel Density Estimation) usa scipy.stats.gaussian_kde se
# disponivel; caso contrario, fallback para histograma puro.
#
# Componentes EM do pipeline (Re/Im de Hxx e Hzz) tipicamente tem
# distribuicoes multimodais por causa de contrastes geologicos — KDE
# evidencia esses modos melhor que histogramas puros.
#
# Ref: docs/ARCHITECTURE_v2.md secao 21.2 (plot_feature_distributions).
# ──────────────────────────────────────────────────────────────────────────


def plot_feature_distributions(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    *,
    config: Optional[PipelineConfig] = None,
    feature_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> None:
    """Histogramas com KDE overlay para cada feature, opcionalmente com splits.

    Gera uma grade de subplots (1 por feature) com histogramas normalizados
    e curva KDE sobreposta. Se ``data`` for um dicionario com chaves
    train/val/test, cada split e desenhado com cor diferente para
    facilitar a deteccao visual de distribuicoes deslocadas (data leakage).

    A KDE usa scipy.stats.gaussian_kde quando disponivel. Se scipy nao
    estiver instalado, o fallback eh histograma puro (sem curva suave).

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Layout tipico (5 features, 1 split):                                   │
    │                                                                          │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
    │  │  z_obs   │ │ Re(Hxx) │ │ Im(Hxx) │ │ Re(Hzz) │ │ Im(Hzz) │         │
    │  │  hist+KDE│ │ hist+KDE│ │ hist+KDE│ │ hist+KDE│ │ hist+KDE│         │
    │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘         │
    │                                                                          │
    │  Layout tipico (5 features, 3 splits):                                  │
    │  Cada subplot sobrepoe train (azul), val (laranja), test (verde)        │
    └──────────────────────────────────────────────────────────────────────────┘

    Args:
        data: Dados de entrada. Dois formatos aceitos:
            - np.ndarray 2D (n_samples, n_feat) ou 3D (n_models, seq_len, n_feat)
            - dict com chaves "train", "val", "test" → cada valor np.ndarray
        config: PipelineConfig opcional para metadados no titulo.
        feature_names: Nomes das features (default: generados).
        save_dir: Diretorio para salvar PNG (None = nao salva).
        dpi: Resolucao DPI (default 300).
        show: Se True, exibe interativamente (default True).

    Raises:
        ValueError: Se array tem ndim diferente de 2 ou 3.

    Example:
        >>> import numpy as np
        >>> from geosteering_ai.visualization.eda import plot_feature_distributions
        >>> data = np.random.randn(20, 600, 5).astype(np.float32)
        >>> plot_feature_distributions(data, show=False)

        >>> splits = {"train": data[:12], "val": data[12:16], "test": data[16:]}
        >>> plot_feature_distributions(splits, show=False)

    Note:
        Referenciado em:
            - data/pipeline.py: diagnostico pos-split
            - tests/test_visualization.py: TestPlotFeatureDistributions
        Ref: docs/ARCHITECTURE_v2.md secao 21.2.
        KDE fallback: se scipy nao instalado, histograma puro.
        Multimodalidade em componentes EM eh esperada (contrastes geologicos).
    """
    plt = _lazy_import_plt()

    # ── Resolver entrada: dict de splits ou array unico ──────────────
    if isinstance(data, dict):
        splits = data
        # ── Validar ndim de cada split (2D ou 3D) ───────────────────
        for split_name, arr in splits.items():
            if not isinstance(arr, np.ndarray) or arr.ndim not in (2, 3):
                ndim = arr.ndim if isinstance(arr, np.ndarray) else "N/A"
                raise ValueError(
                    f"splits['{split_name}'] deve ser np.ndarray 2D ou 3D, "
                    f"recebido ndim={ndim}."
                )
        # Inferir n_features do primeiro split disponivel
        first_arr = next(iter(splits.values()))
        n_features = first_arr.shape[-1]
    elif isinstance(data, np.ndarray):
        if data.ndim not in (2, 3):
            raise ValueError(f"Esperado array 2D ou 3D, recebido ndim={data.ndim}.")
        splits = {"data": data}
        n_features = data.shape[-1]
    else:
        raise ValueError(
            f"data deve ser np.ndarray ou dict, recebido {type(data).__name__}."
        )

    names = _resolve_feature_names(n_features, feature_names)

    # ── KDE: tentar importar scipy ───────────────────────────────────
    try:
        from scipy.stats import gaussian_kde

        _has_kde = True
    except ImportError:
        _has_kde = False
        logger.info("scipy nao disponivel — KDE desabilitado, usando histograma puro.")

    # ── Layout: uma coluna por feature, ate 5 colunas por linha ──────
    n_cols = min(n_features, 5)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for feat_idx in range(n_features):
        ax = axes[feat_idx // n_cols, feat_idx % n_cols]
        for split_name, arr in splits.items():
            # ── Achatar se 3D ────────────────────────────────────────
            flat = _flatten_to_2d(arr) if arr.ndim == 3 else arr
            col_data = flat[:, feat_idx]
            color = _SPLIT_COLORS.get(split_name, "#7f7f7f")

            # ── Histograma normalizado ───────────────────────────────
            ax.hist(
                col_data,
                bins=_HIST_BINS,
                alpha=0.4,
                density=True,
                color=color,
                label=split_name,
            )

            # ── KDE overlay (se scipy disponivel) ────────────────────
            # KDE suaviza a distribuicao empirica, evidenciando modos
            # multimodais tipicos de dados EM com contrastes geologicos.
            if _has_kde and len(col_data) > 1:
                try:
                    kde = gaussian_kde(col_data.astype(np.float64))
                    x_grid = np.linspace(col_data.min(), col_data.max(), 200)
                    ax.plot(x_grid, kde(x_grid), color=color, linewidth=1.5)
                except np.linalg.LinAlgError:
                    # Dados constantes → KDE falha (variancia zero)
                    logger.debug(
                        "KDE falhou para feature %d (%s) — dados constantes.",
                        feat_idx,
                        names[feat_idx],
                    )

        ax.set_title(names[feat_idx], fontsize=10)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidade")
        ax.grid(True, alpha=0.3)
        if len(splits) > 1:
            ax.legend(fontsize=7)

    # ── Esconder axes vazios ─────────────────────────────────────────
    for idx in range(n_features, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    title = "Distribuicao de Features"
    if config is not None:
        title += f" — FV={config.feature_view}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    logger.info(
        "plot_feature_distributions: %d features, %d splits, KDE=%s",
        n_features,
        len(splits),
        _has_kde,
    )
    _finalize_figure(
        fig,
        plt,
        save_dir=save_dir,
        filename="eda_feature_distributions.png",
        dpi=dpi,
        show=show,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCAO 2 — plot_correlation_heatmap
# ════════════════════════════════════════════════════════════════════════════
# Matriz de correlacao standalone com anotacoes numericas e threshold.
# Suporta Pearson (default) e Spearman (robusto a outliers).
#
# Para dados EM de inversao, a correlacao entre Re(Hxx) e Im(Hxx) e
# tipicamente alta (~0.7-0.9) pois ambas derivam do mesmo tensor H.
# Correlacoes inesperadamente baixas (<0.3) entre componentes do mesmo
# tensor indicam possivel problema de preprocessing ou split.
#
# O threshold filtra anotacoes: valores com |r| < threshold sao
# exibidos em branco/cinza para reduzir ruido visual e destacar
# apenas correlacoes significativas.
#
# Ref: docs/ARCHITECTURE_v2.md secao 21.2 (plot_correlation_heatmap).
# ──────────────────────────────────────────────────────────────────────────


def plot_correlation_heatmap(
    data: np.ndarray,
    *,
    config: Optional[PipelineConfig] = None,
    feature_names: Optional[List[str]] = None,
    method: str = "pearson",
    threshold: float = 0.0,
    save_dir: Optional[str] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> None:
    """Heatmap de correlacao com anotacoes numericas e threshold de filtragem.

    Calcula a matriz de correlacao entre features usando Pearson (default)
    ou Spearman (robusto a outliers e relacoes nao-lineares). Anota cada
    celula com o valor numerico; valores com |r| < threshold sao mostrados
    em cinza claro para reduzir ruido visual.

    Para dados EM de inversao 1D, esperamos:
      - Re(Hxx) × Im(Hxx): alta correlacao (~0.7-0.9) — mesmo tensor
      - Re(Hzz) × Im(Hzz): alta correlacao (~0.7-0.9) — mesmo tensor
      - z_obs × Re(Hxx): moderada (~0.3-0.6) — atenuacao com profundidade
      - Correlacoes inesperadas (<0.3 ou >0.95) merecem investigacao

    Args:
        data: Array 2D (n_samples, n_feat) ou 3D (n_models, seq_len, n_feat).
            3D e achatado internamente para 2D.
        config: PipelineConfig opcional para metadados no titulo.
        feature_names: Nomes das features.
        method: Metodo de correlacao: "pearson" (default) ou "spearman".
            Pearson mede correlacao linear; Spearman mede correlacao
            monotonica (rank-based), mais robusto a outliers.
        threshold: Limiar minimo para exibir anotacao (default 0.0 = todas).
            Valores com |r| < threshold sao anotados em cinza claro.
        save_dir: Diretorio para salvar PNG.
        dpi: Resolucao DPI (default 300).
        show: Se True, exibe interativamente.

    Raises:
        ValueError: Se method nao e "pearson" ou "spearman".
        ValueError: Se data.ndim nao e 2 ou 3.

    Example:
        >>> import numpy as np
        >>> from geosteering_ai.visualization.eda import plot_correlation_heatmap
        >>> data = np.random.randn(50, 600, 5).astype(np.float32)
        >>> plot_correlation_heatmap(data, method="spearman", threshold=0.3, show=False)

    Note:
        Referenciado em:
            - tests/test_visualization.py: TestPlotCorrelationHeatmap
        Ref: docs/ARCHITECTURE_v2.md secao 21.2.
        Spearman requer scipy.stats.spearmanr (fallback para Pearson se ausente).
        Correlacoes Re/Im do mesmo tensor sao fisicamente esperadas.
    """
    if method not in _VALID_CORR_METHODS:
        raise ValueError(
            f"method='{method}' invalido. Validos: {sorted(_VALID_CORR_METHODS)}"
        )

    if isinstance(data, np.ndarray) and data.ndim not in (2, 3):
        raise ValueError(f"Esperado array 2D ou 3D, recebido ndim={data.ndim}.")

    plt = _lazy_import_plt()
    data_flat = _flatten_to_2d(data) if data.ndim == 3 else data
    n_features = data_flat.shape[1]
    names = _resolve_feature_names(n_features, feature_names)

    # ── Calcular matriz de correlacao ────────────────────────────────
    if method == "pearson":
        corr_matrix = np.corrcoef(data_flat.T)
    else:
        # ── Spearman: correlacao de ranks (robusto a outliers) ───────
        # Usa scipy.stats.spearmanr se disponivel; fallback para
        # Pearson sobre ranks manuais se scipy ausente.
        try:
            from scipy.stats import spearmanr

            # ── scipy.stats.spearmanr retorna SpearmanrResult (>=1.7) ──
            # .statistic eh float para 2 features, ndarray para >2.
            # Usar np.asarray para normalizar ambos os casos.
            result = spearmanr(data_flat)
            corr_arr = np.asarray(
                result.statistic if hasattr(result, "statistic") else result.correlation
            )
            if corr_arr.ndim < 2:
                # ── 2-feature case: escalar → reconstruir 2x2 simetrica ─
                v = float(corr_arr)
                corr_matrix = np.array([[1.0, v], [v, 1.0]])
            else:
                corr_matrix = corr_arr
        except ImportError:
            logger.warning("scipy nao instalado — fallback para Pearson sobre ranks.")
            ranks = np.apply_along_axis(
                lambda col: np.argsort(np.argsort(col)).astype(np.float64),
                axis=0,
                arr=data_flat,
            )
            corr_matrix = np.corrcoef(ranks.T)

    # ── Plotar heatmap ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, n_features * 1.2), max(5, n_features)))
    im = ax.imshow(
        corr_matrix,
        cmap=_CMAP_CORR,
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Anotar valores com threshold filtering ───────────────────────
    # Valores com |r| >= threshold em preto/branco (contraste),
    # valores com |r| < threshold em cinza claro (reduz ruido visual).
    for i in range(n_features):
        for j in range(n_features):
            val = corr_matrix[i, j]
            if abs(val) >= threshold:
                color = "white" if abs(val) > 0.5 else "black"
            else:
                color = "#cccccc"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    title = f"Correlacao ({method.capitalize()})"
    if threshold > 0:
        title += f" — threshold={threshold}"
    if config is not None:
        title += f" — FV={config.feature_view}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    logger.info(
        "plot_correlation_heatmap: %d features, method=%s, threshold=%.2f",
        n_features,
        method,
        threshold,
    )
    _finalize_figure(
        fig,
        plt,
        save_dir=save_dir,
        filename="eda_correlation_heatmap.png",
        dpi=dpi,
        show=show,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCAO 3 — plot_sample_profiles
# ════════════════════════════════════════════════════════════════════════════
# Plota perfis de amostra: componentes EM vs indice de medicao (proxy
# para profundidade z_obs). Cada subplot mostra uma amostra completa
# (seq_len pontos) com todas as features sobrepostas.
#
# Este plot eh o equivalente ao "log de poco" do pipeline: permite
# visualizar como as componentes EM (Re/Im de Hxx e Hzz) variam ao
# longo do perfil de medicao, identificando:
#   - Contrastes de resistividade (steps abruptos em Re/Im)
#   - Zonas de transicao (rampas suaves)
#   - Anomalias (spikes, dropouts)
#   - Coerencia entre componentes do mesmo tensor
#
# Requer dados 3D (n_models, seq_len, n_features) — cada modelo
# geologico produz um perfil completo de 600 medicoes.
#
# Ref: docs/ARCHITECTURE_v2.md secao 21.2 (plot_sample_profiles).
# ──────────────────────────────────────────────────────────────────────────


def plot_sample_profiles(
    data: np.ndarray,
    *,
    config: Optional[PipelineConfig] = None,
    feature_names: Optional[List[str]] = None,
    n_samples: int = _DEFAULT_N_SAMPLES,
    save_dir: Optional[str] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> None:
    """Plota perfis de amostra: componentes EM vs indice de medicao.

    Visualiza N amostras do dataset, cada uma em um subplot separado,
    com todas as features sobrepostas. O eixo X representa o indice
    de medicao (proxy para profundidade z_obs, SPACING_METERS=1.0 m),
    e cada curva representa uma feature (componente EM ou z_obs).

    Equivalente ao "log de poco" do pipeline de inversao. Permite
    identificar visualmente:
      - Contrastes de resistividade → degraus abruptos em Re/Im
      - Zonas de transicao → rampas suaves entre camadas
      - Anomalias → spikes de ruido ou dropouts de telemetria
      - Coerencia → Re(Hxx) e Im(Hxx) devem ter shapes similares

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Layout (n_samples=3, 5 features):                                      │
    │                                                                          │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │  Amostra 0: z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)        │   │
    │  │  Amostra 1: ...                                                 │   │
    │  │  Amostra 2: ...                                                 │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    └──────────────────────────────────────────────────────────────────────────┘

    Args:
        data: Array 3D (n_models, seq_len, n_features). DEVE ser 3D.
            Layout P1 baseline: [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)].
        config: PipelineConfig opcional para metadados.
        feature_names: Nomes das features.
        n_samples: Numero de amostras a plotar (default 3).
            Se maior que n_models, usa min(n_samples, n_models).
        save_dir: Diretorio para salvar PNG.
        dpi: Resolucao DPI (default 300).
        show: Se True, exibe interativamente.

    Raises:
        ValueError: Se data.ndim != 3.

    Example:
        >>> import numpy as np
        >>> from geosteering_ai.visualization.eda import plot_sample_profiles
        >>> data = np.random.randn(10, 600, 5).astype(np.float32)
        >>> plot_sample_profiles(data, n_samples=2, show=False)

    Note:
        Referenciado em:
            - tests/test_visualization.py: TestPlotSampleProfiles
        Ref: docs/ARCHITECTURE_v2.md secao 21.2.
        Requer dados 3D — 2D nao tem eixo temporal para perfil.
        seq_len=600 corresponde a 600 medicoes × SPACING_METERS=1.0 m.
    """
    if data.ndim != 3:
        raise ValueError(
            f"plot_sample_profiles requer dados 3D (n_models, seq_len, n_feat), "
            f"recebido ndim={data.ndim}."
        )

    plt = _lazy_import_plt()
    n_models, seq_len, n_features = data.shape
    n_plot = min(n_samples, n_models)
    names = _resolve_feature_names(n_features, feature_names)
    seq_axis = np.arange(seq_len)

    fig, axes = plt.subplots(
        n_plot,
        1,
        figsize=(14, 3.5 * n_plot),
        sharex=True,
        squeeze=False,
    )

    for sample_idx in range(n_plot):
        ax = axes[sample_idx, 0]
        sample = data[sample_idx]  # (seq_len, n_features)

        for feat_idx in range(n_features):
            # ── Cada feature em cor distinta com linewidth fino ──────
            # Cores do ciclo matplotlib default (tab10), adequado para
            # ate 10 features (P1=5, P4 com GS ate ~15).
            ax.plot(
                seq_axis,
                sample[:, feat_idx],
                linewidth=0.8,
                alpha=0.85,
                label=names[feat_idx],
            )

        ax.set_ylabel("Valor")
        ax.set_title(f"Amostra {sample_idx}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(
            fontsize=7,
            loc="upper right",
            ncol=max(1, n_features // 3),
        )

    axes[-1, 0].set_xlabel("Indice de Medicao (proxy z_obs)")

    title = "Perfis de Amostra — Componentes EM vs Profundidade"
    if config is not None:
        title += f" — {config.model_type}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    logger.info(
        "plot_sample_profiles: %d amostras, %d features, seq_len=%d",
        n_plot,
        n_features,
        seq_len,
    )
    _finalize_figure(
        fig,
        plt,
        save_dir=save_dir,
        filename="eda_sample_profiles.png",
        dpi=dpi,
        show=show,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCAO 4 — plot_train_val_test_comparison
# ════════════════════════════════════════════════════════════════════════════
# Boxplots lado a lado para cada feature, agrupados por split
# (train, val, test). Diagnostico rapido para detectar data leakage:
#
# Sintomas de leakage:
#   - Distribuicoes identicas entre splits → possivel split por amostra
#     em vez de por modelo geologico (violacao P1)
#   - Distribuicoes drasticamente diferentes → viés de selecao no split
#   - Val/test com range maior que train → inversao do esperado
#
# O pipeline Geosteering AI DEVE splittar por modelo geologico (P1),
# o que garante que cada modelo inteiro vai para um unico split.
# Splits por amostra randomica misturam medicoes do mesmo modelo
# entre splits, causando leakage por dependencia temporal.
#
# Ref: docs/ARCHITECTURE_v2.md secao 21.2 (plot_train_val_test_comparison).
# ──────────────────────────────────────────────────────────────────────────


def plot_train_val_test_comparison(
    splits: Dict[str, np.ndarray],
    *,
    config: Optional[PipelineConfig] = None,
    feature_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> None:
    """Boxplots comparativos train/val/test por feature para detectar leakage.

    Produz uma grade de subplots com um boxplot por feature. Cada subplot
    mostra os boxplots de train, val e test lado a lado, permitindo
    comparacao visual rapida entre splits.

    Diagnostico de data leakage:
      ┌──────────────────────────────────────────────────────────────────┐
      │  Sintoma                            │ Causa Provavel            │
      │  ──────────────────────────────────┼────────────────────────── │
      │  Distribuicoes identicas            │ Split por amostra (P1!)  │
      │  Val/test range > train             │ Viés de selecao          │
      │  Outliers apenas em 1 split         │ Modelos geologicos raros │
      │  Mediana deslocada (>1σ)            │ Feature nao-estacionaria │
      └──────────────────────────────────────────────────────────────────┘

    Args:
        splits: Dicionario com chaves "train", "val", "test" (pelo menos 2).
            Cada valor eh np.ndarray 2D ou 3D.
        config: PipelineConfig opcional para metadados.
        feature_names: Nomes das features.
        save_dir: Diretorio para salvar PNG.
        dpi: Resolucao DPI (default 300).
        show: Se True, exibe interativamente.

    Raises:
        TypeError: Se splits nao e dict.

    Example:
        >>> import numpy as np
        >>> from geosteering_ai.visualization.eda import plot_train_val_test_comparison
        >>> d = {"train": np.random.randn(6,600,5), "val": np.random.randn(2,600,5),
        ...      "test": np.random.randn(2,600,5)}
        >>> plot_train_val_test_comparison(d, show=False)

    Note:
        Referenciado em:
            - tests/test_visualization.py: TestPlotTrainValTestComparison
        Ref: docs/ARCHITECTURE_v2.md secao 21.2.
        Split por modelo geologico (P1) garante independencia entre splits.
        Leakage por split de amostra eh o erro mais comum em pipelines EM.
    """
    if not isinstance(splits, dict):
        raise TypeError(f"splits deve ser dict, recebido {type(splits).__name__}.")

    plt = _lazy_import_plt()

    # ── Verificar splits esperados ───────────────────────────────────
    expected = {"train", "val", "test"}
    present = set(splits.keys())
    missing = expected - present
    if missing:
        logger.warning(
            "Splits faltando: %s. Presentes: %s",
            sorted(missing),
            sorted(present),
        )

    # ── Inferir n_features do primeiro split ─────────────────────────
    first_arr = next(iter(splits.values()))
    n_features = first_arr.shape[-1]
    names = _resolve_feature_names(n_features, feature_names)

    # ── Achatar todos os splits para 2D ──────────────────────────────
    flat_splits = {}
    for split_name, arr in splits.items():
        flat_splits[split_name] = _flatten_to_2d(arr) if arr.ndim == 3 else arr

    # ── Layout: 1 subplot por feature ────────────────────────────────
    n_cols = min(n_features, 5)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    split_names = sorted(flat_splits.keys())
    for feat_idx in range(n_features):
        ax = axes[feat_idx // n_cols, feat_idx % n_cols]
        box_data = []
        box_labels = []
        box_colors = []

        for sn in split_names:
            col = flat_splits[sn][:, feat_idx]
            box_data.append(col)
            box_labels.append(sn)
            box_colors.append(_SPLIT_COLORS.get(sn, "#7f7f7f"))

        bp = ax.boxplot(
            box_data,
            labels=box_labels,
            patch_artist=True,
            showfliers=True,
            flierprops={"markersize": 2, "alpha": 0.3},
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(names[feat_idx], fontsize=10)
        ax.set_ylabel("Valor")
        ax.grid(True, alpha=0.3, axis="y")

    # ── Esconder axes vazios ─────────────────────────────────────────
    for idx in range(n_features, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    title = "Comparacao Train/Val/Test por Feature"
    if config is not None:
        title += f" — split_by_model={config.split_by_model}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    logger.info(
        "plot_train_val_test_comparison: %d features, splits=%s",
        n_features,
        split_names,
    )
    _finalize_figure(
        fig,
        plt,
        save_dir=save_dir,
        filename="eda_train_val_test_comparison.png",
        dpi=dpi,
        show=show,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCAO 5 — plot_sensitivity_heatmap
# ════════════════════════════════════════════════════════════════════════════
# Heatmap 2D de sensibilidade: features (eixo Y) x profundidade (eixo X).
# Duas metricas disponiveis:
#   - variance: variancia por posicao ao longo dos modelos
#   - gradient: gradiente medio (|df/dz|) por posicao
#
# Interpretacao fisica:
#   - Alta variancia numa posicao → grande diversidade de respostas EM
#     naquela profundidade (fronteira de camada, contraste de resistividade)
#   - Variancia uniforme → formacao homogenea
#   - Gradiente alto → transicao abrupta (boundary detection)
#   - Gradiente baixo → zona suave (interior de camada)
#
# Requer dados 3D (n_models, seq_len, n_features). A variancia eh
# calculada ao longo do eixo 0 (modelos), produzindo (seq_len, n_feat).
#
# Ref: docs/ARCHITECTURE_v2.md secao 21.2 (plot_sensitivity_heatmap).
# ──────────────────────────────────────────────────────────────────────────


def plot_sensitivity_heatmap(
    data: np.ndarray,
    *,
    config: Optional[PipelineConfig] = None,
    feature_names: Optional[List[str]] = None,
    metric: str = "variance",
    save_dir: Optional[str] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> None:
    """Heatmap de sensibilidade: features x profundidade (variancia ou gradiente).

    Calcula uma metrica de sensibilidade por posicao temporal e por feature,
    produzindo um heatmap 2D (n_features x seq_len) que mostra onde o sinal
    EM varia mais ao longo dos modelos geologicos.

    Metricas disponiveis:
      ┌──────────────────────────────────────────────────────────────────┐
      │  Metrica    │ Formula            │ Detecta                      │
      │  ──────────┼────────────────────┼───────────────────────────── │
      │  variance  │ Var(x, axis=0)     │ Diversidade de resposta EM   │
      │  gradient  │ mean(|dx/dz|, ax=0)│ Transicoes abruptas (beds)   │
      └──────────────────────────────────────────────────────────────────┘

    Interpretacao fisica:
      - Alta variancia numa posicao → fronteira de camada, contraste
        de resistividade entre modelos geologicos diferentes
      - Variancia uniforme → formacao homogenea, sem contraste
      - Gradiente alto → transicao abrupta (boundary, interface)
      - Gradiente baixo → interior de camada suave

    Args:
        data: Array 3D (n_models, seq_len, n_features). DEVE ser 3D.
            Variancia/gradiente calculados ao longo do eixo 0 (modelos).
        config: PipelineConfig opcional para metadados.
        feature_names: Nomes das features.
        metric: Metrica de sensibilidade: "variance" (default) ou "gradient".
        save_dir: Diretorio para salvar PNG.
        dpi: Resolucao DPI (default 300).
        show: Se True, exibe interativamente.

    Raises:
        ValueError: Se data.ndim != 3.
        ValueError: Se metric nao e "variance" ou "gradient".

    Example:
        >>> import numpy as np
        >>> from geosteering_ai.visualization.eda import plot_sensitivity_heatmap
        >>> data = np.random.randn(20, 600, 5).astype(np.float32)
        >>> plot_sensitivity_heatmap(data, metric="variance", show=False)

    Note:
        Referenciado em:
            - tests/test_visualization.py: TestPlotSensitivityHeatmap
        Ref: docs/ARCHITECTURE_v2.md secao 21.2.
        Requer 3D — variancia ao longo de modelos nao faz sentido em 2D.
        seq_len=600 × SPACING_METERS=1.0 → eixo X em metros.
    """
    if data.ndim != 3:
        raise ValueError(
            f"plot_sensitivity_heatmap requer dados 3D (n_models, seq_len, n_feat), "
            f"recebido ndim={data.ndim}."
        )
    if metric not in _VALID_SENSITIVITY_METRICS:
        raise ValueError(
            f"metric='{metric}' invalido. Validos: {sorted(_VALID_SENSITIVITY_METRICS)}"
        )

    plt = _lazy_import_plt()
    n_models, seq_len, n_features = data.shape
    names = _resolve_feature_names(n_features, feature_names)

    # ── Guard: gradient requer seq_len >= 2 (np.diff retorna vazio) ──
    if metric == "gradient" and seq_len < 2:
        raise ValueError(
            f"metric='gradient' requer seq_len >= 2, recebido seq_len={seq_len}."
        )

    # ── Calcular metrica de sensibilidade ────────────────────────────
    if metric == "variance":
        # ── Variancia por posicao ao longo dos modelos ───────────────
        # shape: (seq_len, n_features) — cada posicao tem uma variancia
        # calculada sobre os n_models modelos geologicos.
        sensitivity = np.var(data, axis=0)  # (seq_len, n_features)
        metric_label = "Variancia"
    else:
        # ── Gradiente medio: mean(|df/dz|) por posicao ──────────────
        # np.diff ao longo do eixo temporal (axis=1) calcula
        # a diferenca entre medicoes consecutivas (proxy para df/dz
        # com SPACING_METERS=1.0). Media absoluta sobre modelos (axis=0)
        # destaca posicoes com transicoes abruptas.
        gradients = np.abs(np.diff(data, axis=1))  # (n_models, seq_len-1, n_feat)
        sensitivity = np.mean(gradients, axis=0)  # (seq_len-1, n_features)
        metric_label = "Gradiente Medio |df/dz|"

    # ── Plotar heatmap (features x posicao) ──────────────────────────
    # Transpor para (n_features, seq_len) — features no eixo Y,
    # posicao/profundidade no eixo X, facilitando leitura como log.
    fig, ax = plt.subplots(figsize=(14, max(3, n_features * 0.8)))
    im = ax.imshow(
        sensitivity.T,  # (n_features, seq_len)
        aspect="auto",
        cmap=_CMAP_SENSITIVITY,
        interpolation="nearest",
    )
    ax.set_xlabel("Indice de Medicao (proxy z_obs)")
    ax.set_ylabel("Feature")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(names, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label=metric_label)

    title = f"Sensibilidade ({metric_label})"
    if config is not None:
        title += f" — {config.model_type}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    logger.info(
        "plot_sensitivity_heatmap: %d modelos, %d features, seq_len=%d, metric=%s",
        n_models,
        n_features,
        seq_len,
        metric,
    )
    _finalize_figure(
        fig,
        plt,
        save_dir=save_dir,
        filename="eda_sensitivity_heatmap.png",
        dpi=dpi,
        show=show,
    )
