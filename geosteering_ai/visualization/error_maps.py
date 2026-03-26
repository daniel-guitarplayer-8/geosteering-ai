# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/error_maps.py                                      ║
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
# ║    • Heatmap 2D de erro absoluto (sample x depth) por componente          ║
# ║    • Grafico de barras de RMSE por banda de resistividade                 ║
# ║    • Perfil espacial de erro RMSE vs indice de profundidade               ║
# ║    • Diagnostico visual para identificar regioes problematicas            ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), numpy, matplotlib (lazy)      ║
# ║  Exports: ~3 (plot_error_heatmap, plot_error_by_band, plot_spatial_error) ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9.6                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C51/C53/C60 viz)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Mapas de erro — heatmap 2D, barras por banda, perfil espacial.

Tres visualizacoes complementares para diagnostico de erro de inversao:

    1. Heatmap 2D (sample x depth) — identifica regioes espaciais com erro
       sistematico (e.g., proximidade de interfaces, bordas de sequencia)
    2. Barras por banda — RMSE segregado por faixa de resistividade, revela
       vieses de dominio (e.g., erro maior em rho < 1 ohm.m)
    3. Perfil espacial — RMSE medio por indice de profundidade, detecta
       degradacao nas bordas de sequencia ou em zonas de transicao

Todas as funcoes usam matplotlib via import lazy e aceitam ``save_path``
opcional para exportacao direta em PNG/PDF/SVG.

Example:
    >>> from geosteering_ai.visualization.error_maps import plot_error_heatmap
    >>> import numpy as np
    >>> y_true = np.random.randn(50, 600, 2)
    >>> y_pred = y_true + np.random.randn(50, 600, 2) * 0.1
    >>> plot_error_heatmap(y_true, y_pred, component=0, show=False)

Note:
    Matplotlib importado de forma lazy (suporta ambientes headless).
    Referenciado em:
        - evaluation/advanced.py: error_by_resistivity_band, spatial_error_profile
        - training/loop.py: diagnostico pos-treinamento
    Ref: docs/ARCHITECTURE_v2.md secao 9.6.
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
    # --- Funcoes de visualizacao de erro ---
    "plot_error_heatmap",
    "plot_error_by_band",
    "plot_spatial_error",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de visualizacao
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_DPI = 300
_FIGSIZE_HEATMAP = (14, 6)     # largura x altura para heatmap
_FIGSIZE_BAR = (10, 5)         # largura x altura para barras por banda
_FIGSIZE_SPATIAL = (10, 5)     # largura x altura para perfil espacial
_EPS = 1e-12                   # eps para float32 (NUNCA 1e-30)

# Nomes das componentes de resistividade (indice 0 e 1)
_COMPONENT_NAMES = {
    0: r"$\rho_h$ (horizontal)",
    1: r"$\rho_v$ (vertical)",
}

# Colormap para heatmap de erro — sequential warm (vermelho = erro alto)
_CMAP_ERROR = "hot_r"

# Cores para graficos de barras
_COLOR_BAR = "#3498db"          # azul para barras de RMSE
_COLOR_SPATIAL = "#e74c3c"      # vermelho para perfil espacial
_COLOR_SPATIAL_FILL = "#e74c3c" # vermelho com transparencia para fill


# ──────────────────────────────────────────────────────────────────────
# D2: Funcao principal — heatmap 2D de erro absoluto
# ──────────────────────────────────────────────────────────────────────
def plot_error_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    component: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """Plota heatmap 2D (sample x depth) de erro absoluto de inversao.

    Cada pixel (i, j) representa |y_true[i, j, c] - y_pred[i, j, c]|
    para a componente ``c`` (0 = rho_h, 1 = rho_v). Cores quentes
    indicam regioes com erro elevado. Util para identificar padroes
    espaciais de erro (e.g., bordas de sequencia, proximidade de
    interfaces geologicas).

    Args:
        y_true: Array (n_samples, seq_len, n_components) com resistividade
            verdadeira em log10. Minimo 2 componentes (rho_h, rho_v).
        y_pred: Array (n_samples, seq_len, n_components) com resistividade
            predita em log10. Mesmo shape de y_true.
        component: Indice da componente a plotar (default 0 = rho_h).
            0 para rho_h (resistividade horizontal), 1 para rho_v (vertical).
        title: Titulo customizado (default None = titulo automatico).
            Se None, titulo inclui nome da componente e RMSE medio.
        save_path: Caminho para salvar a figura (None = nao salva).
            Diretorios pais criados automaticamente.
        show: Se True, chama plt.show() ao final (default True).
        dpi: Resolucao em DPI para salvamento (default 300).

    Raises:
        ValueError: Se shapes de y_true e y_pred sao incompativeis.
        ValueError: Se component esta fora do range valido.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import numpy as np
        >>> y_true = np.random.randn(20, 600, 2)
        >>> y_pred = y_true + np.random.randn(20, 600, 2) * 0.05
        >>> plot_error_heatmap(y_true, y_pred, component=0, show=False)
        >>> plot_error_heatmap(y_true, y_pred, component=1, show=False)

    Note:
        Erro computado em dominio log10 (mesmo dominio do treinamento).
        Ref: docs/ARCHITECTURE_v2.md secao 9.6 (error maps).
    """
    # --- Validacao de entrada ---
    if y_true.shape != y_pred.shape:
        msg = (
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}. "
            "Ambos devem ter shape identico (n, seq_len, n_components)."
        )
        raise ValueError(msg)

    if y_true.ndim != 3:
        msg = (
            f"Esperado array 3D (n, seq_len, n_components), "
            f"recebido ndim={y_true.ndim}."
        )
        raise ValueError(msg)

    n_components = y_true.shape[-1]
    if component < 0 or component >= n_components:
        msg = (
            f"component={component} fora do range valido [0, {n_components - 1}]. "
            f"Array tem {n_components} componentes."
        )
        raise ValueError(msg)

    # --- Import lazy do matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    # --- Computar erro absoluto ---
    # D7: Erro no dominio log10 (dominio do treinamento)
    abs_error = np.abs(y_true[:, :, component] - y_pred[:, :, component])
    rmse_total = float(np.sqrt(np.mean(abs_error ** 2) + _EPS))

    n_samples, seq_len = abs_error.shape
    logger.info(
        "plot_error_heatmap: component=%d, shape=(%d, %d), RMSE=%.6f",
        component, n_samples, seq_len, rmse_total,
    )

    # --- Nome da componente ---
    comp_name = _COMPONENT_NAMES.get(component, f"Componente {component}")

    # --- Titulo ---
    if title is None:
        title = f"Erro Absoluto — {comp_name} (RMSE={rmse_total:.4f})"

    # --- Criar figura ---
    fig, ax = plt.subplots(1, 1, figsize=_FIGSIZE_HEATMAP)

    im = ax.imshow(
        abs_error,
        aspect="auto",
        cmap=_CMAP_ERROR,
        interpolation="nearest",
        origin="upper",
    )

    ax.set_xlabel("Indice de Profundidade", fontsize=11)
    ax.set_ylabel("Amostra", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Colorbar com label
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|y_{true} - y_{pred}|$ (log10 $\Omega\cdot$m)", fontsize=10)

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path_obj), dpi=dpi, bbox_inches="tight")
        logger.info("Error heatmap salvo em: %s", save_path_obj)

    # --- Exibir se solicitado ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_error_heatmap concluido: component=%d, RMSE=%.6f",
                component, rmse_total)


# ──────────────────────────────────────────────────────────────────────
# D2: Grafico de barras — RMSE por banda de resistividade
# ──────────────────────────────────────────────────────────────────────
def plot_error_by_band(
    band_results: Dict[str, float],
    *,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """Plota grafico de barras de RMSE por banda de resistividade.

    Cada barra representa o RMSE medio para amostras dentro de uma
    faixa especifica de resistividade (e.g., "0.1-1.0", "1.0-10.0",
    "10.0-100.0" ohm.m). Revela vieses de dominio onde o modelo
    performa pior (tipicamente em resistividades muito baixas ou altas).

    Args:
        band_results: Dicionario {nome_banda: rmse_valor}.
            Chaves sao strings descritivas (e.g., "0.1-1.0 ohm.m").
            Valores sao RMSE float para cada banda. Output tipico de
            ``evaluation.advanced.error_by_resistivity_band()``.
        title: Titulo customizado (default None = "RMSE por Banda de Resistividade").
        save_path: Caminho para salvar a figura (None = nao salva).
            Diretorios pais criados automaticamente.
        show: Se True, chama plt.show() ao final (default True).
        dpi: Resolucao em DPI para salvamento (default 300).

    Raises:
        ValueError: Se band_results estiver vazio.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> bands = {"0.1-1.0": 0.12, "1.0-10.0": 0.05, "10.0-100.0": 0.08}
        >>> plot_error_by_band(bands, show=False)

    Note:
        Bandas ordenadas pela posicao no dicionario (preserva ordem de insercao).
        Ref: docs/ARCHITECTURE_v2.md secao 9.6 (error by band).
    """
    # --- Validacao de entrada ---
    if not band_results:
        msg = "band_results esta vazio. Fornecer pelo menos uma banda."
        raise ValueError(msg)

    # --- Import lazy do matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    band_names = list(band_results.keys())
    rmse_values = np.array([band_results[k] for k in band_names], dtype=np.float64)

    logger.info(
        "plot_error_by_band: %d bandas, RMSE range [%.6f, %.6f]",
        len(band_names), rmse_values.min(), rmse_values.max(),
    )

    # --- Titulo ---
    if title is None:
        title = "RMSE por Banda de Resistividade"

    # --- Criar figura ---
    fig, ax = plt.subplots(1, 1, figsize=_FIGSIZE_BAR)

    x_pos = np.arange(len(band_names))
    bars = ax.bar(
        x_pos, rmse_values,
        color=_COLOR_BAR, edgecolor="white", linewidth=0.8,
        alpha=0.85,
    )

    # D7: Anotar valor de RMSE acima de cada barra
    for bar_obj, val in zip(bars, rmse_values):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2.0,
            bar_obj.get_height() + rmse_values.max() * 0.02,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_names, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Banda de Resistividade", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path_obj), dpi=dpi, bbox_inches="tight")
        logger.info("Error by band salvo em: %s", save_path_obj)

    # --- Exibir se solicitado ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_error_by_band concluido: %d bandas", len(band_names))


# ──────────────────────────────────────────────────────────────────────
# D2: Perfil espacial — RMSE vs indice de profundidade
# ──────────────────────────────────────────────────────────────────────
def plot_spatial_error(
    error_profile: np.ndarray,
    *,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """Plota perfil espacial de RMSE vs indice de profundidade.

    Mostra como o erro medio varia ao longo da dimensao de profundidade
    (seq_len). Util para detectar degradacao nas bordas de sequencia
    (efeito de borda em modelos convolucionais) ou concentracao de erro
    em zonas de transicao entre camadas geologicas.

    Args:
        error_profile: Array 1D (seq_len,) com RMSE medio por indice
            de profundidade. Output tipico de
            ``evaluation.advanced.spatial_error_profile()``.
        title: Titulo customizado (default None = "Perfil Espacial de Erro").
        save_path: Caminho para salvar a figura (None = nao salva).
            Diretorios pais criados automaticamente.
        show: Se True, chama plt.show() ao final (default True).
        dpi: Resolucao em DPI para salvamento (default 300).

    Raises:
        ValueError: Se error_profile nao for 1D.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import numpy as np
        >>> profile = np.random.uniform(0.01, 0.15, 600)
        >>> plot_spatial_error(profile, show=False)

    Note:
        Perfil tipicamente computado como RMSE medio sobre todas as
        amostras para cada indice de profundidade.
        Ref: docs/ARCHITECTURE_v2.md secao 9.6 (spatial error).
    """
    # --- Validacao de entrada ---
    if error_profile.ndim != 1:
        msg = (
            f"Esperado array 1D (seq_len,), recebido ndim={error_profile.ndim}, "
            f"shape={error_profile.shape}."
        )
        raise ValueError(msg)

    # --- Import lazy do matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    seq_len = error_profile.shape[0]
    depth_axis = np.arange(seq_len)
    rmse_mean = float(np.mean(error_profile))
    rmse_max = float(np.max(error_profile))
    rmse_max_idx = int(np.argmax(error_profile))

    logger.info(
        "plot_spatial_error: seq_len=%d, RMSE medio=%.6f, max=%.6f (idx=%d)",
        seq_len, rmse_mean, rmse_max, rmse_max_idx,
    )

    # --- Titulo ---
    if title is None:
        title = f"Perfil Espacial de Erro (RMSE medio={rmse_mean:.4f})"

    # --- Criar figura ---
    fig, ax = plt.subplots(1, 1, figsize=_FIGSIZE_SPATIAL)

    # Area preenchida sob a curva para visualizacao do envelope de erro
    ax.fill_between(
        depth_axis, 0, error_profile,
        color=_COLOR_SPATIAL_FILL, alpha=0.2,
    )
    ax.plot(
        depth_axis, error_profile,
        color=_COLOR_SPATIAL, linewidth=1.2, label="RMSE por Profundidade",
    )

    # D7: Linha horizontal de RMSE medio
    ax.axhline(
        y=rmse_mean, color="gray", linewidth=1.0, linestyle="--",
        label=f"RMSE medio={rmse_mean:.4f}",
    )

    # D7: Marcar ponto de erro maximo
    ax.plot(
        rmse_max_idx, rmse_max,
        marker="v", markersize=10, color="darkred",
        label=f"Max RMSE={rmse_max:.4f} (idx={rmse_max_idx})",
    )

    ax.set_xlabel("Indice de Profundidade", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, seq_len - 1)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path_obj), dpi=dpi, bbox_inches="tight")
        logger.info("Spatial error salvo em: %s", save_path_obj)

    # --- Exibir se solicitado ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_spatial_error concluido: seq_len=%d", seq_len)
