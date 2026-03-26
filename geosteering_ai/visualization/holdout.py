# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/holdout.py                                          ║
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
# ║    • Plotagem de amostras holdout: true vs predicted (rho_h, rho_v)       ║
# ║    • Perfis de resistividade em escala log10 com deteccao de interfaces   ║
# ║    • 2 subplots por amostra: rho_h (azul) e rho_v (vermelho)            ║
# ║    • Suporte a salvamento em disco (save_dir) e exibicao interativa      ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), numpy, matplotlib (lazy)      ║
# ║  Exports: ~1 (plot_holdout_samples)                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9.1                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C42B)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Plotagem de amostras holdout — comparacao true vs predicted.

Gera figuras com 2 subplots por amostra:
    - Esquerda: rho_h (resistividade horizontal) — true (preto) vs pred (azul)
    - Direita:  rho_v (resistividade vertical)   — true (preto) vs pred (vermelho)

Ambos em escala log10. Deteccao automatica de interfaces de camada
(transicoes staircase) nos perfis de resistividade verdadeira.

Example:
    >>> from geosteering_ai.visualization import plot_holdout_samples
    >>> plot_holdout_samples(y_true, y_pred, n_samples=3, dpi=150)

Note:
    Matplotlib e importado de forma lazy para suportar ambientes headless.
    Referenciado em:
        - training/loop.py: avaliacao pos-treinamento
        - evaluation/metrics.py: relatorio visual de holdout
    Ref: docs/ARCHITECTURE_v2.md secao 9.1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Funcao principal ---
    "plot_holdout_samples",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de visualizacao
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_N_SAMPLES = 5
_DEFAULT_DPI = 300
_FIGSIZE_PER_SAMPLE = (12, 4)  # largura x altura por amostra

# Cores para consistencia visual com legado (C42B)
_COLOR_TRUE = "black"
_COLOR_PRED_RHO_H = "#1f77b4"  # azul matplotlib default
_COLOR_PRED_RHO_V = "#d62728"  # vermelho matplotlib default


# ──────────────────────────────────────────────────────────────────────
# D2: Funcoes auxiliares — deteccao de interfaces de camada
# ──────────────────────────────────────────────────────────────────────
def _detect_interfaces(profile: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Detecta indices de interfaces de camada em perfil staircase.

    Uma interface e detectada onde a diferenca absoluta entre pontos
    consecutivos excede ``eps``, indicando transicao entre camadas
    geologicas com resistividades distintas.

    Args:
        profile: Array 1D (seq_len,) com valores de resistividade log10.
        eps: Limiar minimo para detectar transicao (default 1e-12).
            Valor 1e-12 adequado para float32; NUNCA usar 1e-30.

    Returns:
        Array 1D com indices das interfaces detectadas.

    Note:
        Ref: docs/ARCHITECTURE_v2.md secao 9.1 (deteccao de interfaces).
    """
    diffs = np.abs(np.diff(profile))
    return np.where(diffs > eps)[0]


# ──────────────────────────────────────────────────────────────────────
# D2: Funcao principal — plotagem de amostras holdout
# ──────────────────────────────────────────────────────────────────────
def plot_holdout_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    config: Optional[PipelineConfig] = None,
    n_samples: int = _DEFAULT_N_SAMPLES,
    save_dir: Optional[str] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> None:
    """Plota comparacao holdout: perfis de resistividade true vs predicted.

    Cria uma figura com 2 subplots por amostra:
        - Esquerda: rho_h (resistividade horizontal) — true vs predicted
        - Direita:  rho_v (resistividade vertical)   — true vs predicted

    Ambos em escala log10. Linhas verticais tracejadas cinza indicam
    interfaces de camada detectadas no perfil verdadeiro.

    Args:
        y_true: Array (n, seq_len, 2) com resistividade verdadeira em log10.
            Canal 0 = rho_h, canal 1 = rho_v.
        y_pred: Array (n, seq_len, 2) com resistividade predita em log10.
            Mesmo shape de y_true.
        config: PipelineConfig opcional para titulo e metadados.
            Se fornecido, o titulo inclui model_type e loss_type.
        n_samples: Numero de amostras a plotar (default 5).
            Limitado automaticamente ao tamanho do dataset.
        save_dir: Diretorio para salvar figuras (None = nao salva).
            Cria o diretorio se nao existir.
        dpi: Resolucao em DPI para salvamento (default 300).
        show: Se True, chama plt.show() ao final (default True).

    Raises:
        ValueError: Se y_true e y_pred tiverem shapes incompativeis.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> import numpy as np
        >>> y_true = np.random.randn(10, 600, 2)
        >>> y_pred = np.random.randn(10, 600, 2)
        >>> plot_holdout_samples(y_true, y_pred, n_samples=3, show=False)

    Note:
        Matplotlib importado de forma lazy (suporta ambientes headless).
        Ref: docs/ARCHITECTURE_v2.md secao 9.1.
    """
    # --- Validacao de entrada ---
    if y_true.shape != y_pred.shape:
        msg = (
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}. "
            "Ambos devem ter shape (n, seq_len, 2)."
        )
        raise ValueError(msg)

    if y_true.ndim != 3 or y_true.shape[-1] != 2:
        msg = (
            f"Esperado shape (n, seq_len, 2), recebido {y_true.shape}. "
            "Canal 0 = rho_h, canal 1 = rho_v."
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

    # --- Limitar n_samples ao tamanho do dataset ---
    n_available = y_true.shape[0]
    n_samples = min(n_samples, n_available)
    logger.info("Plotando %d amostras holdout (de %d disponiveis)", n_samples, n_available)

    # --- Titulo base ---
    base_title = "Holdout"
    if config is not None:
        base_title = f"{config.model_type} — {config.loss_type}"

    # --- Selecao de indices uniformemente espacados ---
    indices = np.linspace(0, n_available - 1, n_samples, dtype=int)

    # --- Diretorio de salvamento ---
    save_path_obj = None
    if save_dir is not None:
        save_path_obj = Path(save_dir)
        save_path_obj.mkdir(parents=True, exist_ok=True)
        logger.info("Figuras serao salvas em: %s", save_path_obj)

    # --- Gerar figuras ---
    seq_len = y_true.shape[1]
    depth_axis = np.arange(seq_len)  # eixo de profundidade (pontos)

    for fig_idx, sample_idx in enumerate(indices):
        fig, axes = plt.subplots(
            1, 2,
            figsize=_FIGSIZE_PER_SAMPLE,
            sharey=True,
        )

        true_sample = y_true[sample_idx]  # (seq_len, 2)
        pred_sample = y_pred[sample_idx]  # (seq_len, 2)

        # --- Detectar interfaces no perfil verdadeiro ---
        interfaces_h = _detect_interfaces(true_sample[:, 0])
        interfaces_v = _detect_interfaces(true_sample[:, 1])

        # --- Subplot esquerdo: rho_h ---
        ax_h = axes[0]
        ax_h.plot(
            true_sample[:, 0], depth_axis,
            color=_COLOR_TRUE, linewidth=1.5, label="True",
        )
        ax_h.plot(
            pred_sample[:, 0], depth_axis,
            color=_COLOR_PRED_RHO_H, linewidth=1.2, linestyle="--", label="Pred",
        )
        # Interfaces como linhas horizontais tracejadas cinza
        for iface in interfaces_h:
            ax_h.axhline(y=iface, color="gray", linewidth=0.5, linestyle=":")
        ax_h.set_xlabel(r"$\log_{10}(\rho_h)$ [$\Omega\cdot$m]")
        ax_h.set_ylabel("Profundidade [pontos]")
        ax_h.set_title(r"$\rho_h$ (horizontal)")
        ax_h.legend(loc="lower right", fontsize=8)
        ax_h.invert_yaxis()  # profundidade cresce para baixo
        ax_h.grid(True, alpha=0.3)

        # --- Subplot direito: rho_v ---
        ax_v = axes[1]
        ax_v.plot(
            true_sample[:, 1], depth_axis,
            color=_COLOR_TRUE, linewidth=1.5, label="True",
        )
        ax_v.plot(
            pred_sample[:, 1], depth_axis,
            color=_COLOR_PRED_RHO_V, linewidth=1.2, linestyle="--", label="Pred",
        )
        for iface in interfaces_v:
            ax_v.axhline(y=iface, color="gray", linewidth=0.5, linestyle=":")
        ax_v.set_xlabel(r"$\log_{10}(\rho_v)$ [$\Omega\cdot$m]")
        ax_v.set_title(r"$\rho_v$ (vertical)")
        ax_v.legend(loc="lower right", fontsize=8)
        # invert_yaxis ja aplicado em ax_h (sharey=True compartilha)
        ax_v.grid(True, alpha=0.3)

        # --- Titulo da figura ---
        fig.suptitle(
            f"{base_title} — Amostra {sample_idx}",
            fontsize=12,
            fontweight="bold",
        )
        fig.tight_layout()

        # --- Salvar se solicitado ---
        if save_path_obj is not None:
            fname = save_path_obj / f"holdout_sample_{sample_idx:04d}.png"
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")
            logger.info("Salvo: %s", fname)

    # --- Exibir se solicitado ---
    if show:
        plt.show()
    else:
        plt.close("all")

    logger.info("plot_holdout_samples concluido: %d amostras", n_samples)
