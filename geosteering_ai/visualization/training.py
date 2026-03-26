# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/training.py                                        ║
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
# ║    • Curvas de treinamento: loss, R², LR schedule, noise curriculum       ║
# ║    • 4 subplots adaptativos (paineis 3 e 4 condicionais a chaves)        ║
# ║    • Plot isolado de LR schedule para analise detalhada                   ║
# ║    • Suporte a salvamento em disco (save_path) e exibicao interativa      ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), numpy, matplotlib (lazy)      ║
# ║  Exports: ~2 (plot_training_history, plot_lr_schedule)                    ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9.5                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C59 training curves)        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Curvas de treinamento — loss, R-squared, LR schedule, noise level.

Gera figuras de diagnostico pos-treinamento para analisar convergencia,
schedule de learning rate e curriculum de ruido:

    1. Train/Val Loss     — curvas de perda por epoca (ambos splits)
    2. R² por Epoca       — coeficiente de determinacao (se 'r2' em history)
    3. LR Schedule        — evolucao do learning rate (se 'lr' em history)
    4. Noise Level        — curriculum de ruido (se 'noise_level' em history)

Paineis 3 e 4 sao exibidos condicionalmente — se a chave nao existir no
dicionario history, o subplot e substituido por um placeholder informativo.

Example:
    >>> from geosteering_ai.visualization.training import plot_training_history
    >>> history = model.fit(...).history
    >>> plot_training_history(history, show=False, save_path="training.png")

Note:
    Matplotlib importado de forma lazy (suporta ambientes headless).
    Referenciado em:
        - training/loop.py: diagnostico pos-treinamento
        - evaluation/metrics.py: relatorio visual de convergencia
    Ref: docs/ARCHITECTURE_v2.md secao 9.5.
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
    # --- Funcoes de plotagem de treinamento ---
    "plot_training_history",
    "plot_lr_schedule",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de visualizacao
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_DPI = 300
_FIGSIZE_HISTORY = (16, 10)   # largura x altura para 4 subplots (2x2)
_FIGSIZE_LR = (10, 4)        # largura x altura para plot isolado de LR
_EPS = 1e-12                  # eps para float32 (NUNCA 1e-30)

# Cores para consistencia visual
_COLOR_TRAIN_LOSS = "#1f77b4"    # azul — train loss
_COLOR_VAL_LOSS = "#ff7f0e"      # laranja — val loss
_COLOR_TRAIN_R2 = "#2ca02c"      # verde — train R²
_COLOR_VAL_R2 = "#d62728"        # vermelho — val R²
_COLOR_LR = "#9467bd"            # roxo — learning rate
_COLOR_NOISE = "#e377c2"         # rosa — noise level


# ──────────────────────────────────────────────────────────────────────
# D2: Funcoes auxiliares — deteccao de chaves no history
# ──────────────────────────────────────────────────────────────────────
def _find_key(history: Dict[str, list], candidates: List[str]) -> Optional[str]:
    """Encontra a primeira chave presente no history dentre os candidatos.

    Necessario porque Keras pode nomear metricas de formas diferentes
    dependendo da versao e da configuracao (e.g., 'lr' vs 'learning_rate',
    'r2' vs 'r2_score' vs 'R2Score').

    Args:
        history: Dicionario history do Keras (chave -> lista de valores).
        candidates: Lista de nomes candidatos em ordem de prioridade.

    Returns:
        Nome da primeira chave encontrada, ou None se nenhuma existir.

    Note:
        Ref: training/loop.py para nomes de metricas registradas.
    """
    for key in candidates:
        if key in history:
            return key
    return None


def _plot_placeholder(ax: object, message: str) -> None:
    """Exibe mensagem informativa em subplot quando dados nao disponiveis.

    Utilizado nos paineis condicionais (LR schedule, noise level) quando
    a chave correspondente nao existe no dicionario history.

    Args:
        ax: Eixo matplotlib onde a mensagem sera exibida.
        message: Texto informativo (e.g., "LR nao registrado no history").

    Note:
        Ref: visualization/eda.py para pattern similar de placeholder.
    """
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=11, color="gray", style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


# ──────────────────────────────────────────────────────────────────────
# D2: Funcao principal — curvas de treinamento (4 subplots)
# ──────────────────────────────────────────────────────────────────────
def plot_training_history(
    history: Dict[str, list],
    *,
    config: Optional[PipelineConfig] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """Plota curvas de treinamento em 4 subplots adaptativos.

    Gera uma figura 2x2 com:
        1. (sup-esq) Train/Val Loss — curvas de perda por epoca
        2. (sup-dir) R² por Epoca — coeficiente de determinacao (condicional)
        3. (inf-esq) LR Schedule — evolucao do learning rate (condicional)
        4. (inf-dir) Noise Level — curriculum de ruido (condicional)

    Paineis condicionais exibem placeholder informativo quando a chave
    correspondente nao existe no dicionario history.

    Args:
        history: Dicionario history do Keras (output de model.fit().history).
            Chaves obrigatorias: 'loss'.
            Chaves opcionais: 'val_loss', 'r2', 'val_r2', 'lr', 'noise_level'.
        config: PipelineConfig opcional para metadados no titulo.
            Se fornecido, model_type e loss_type sao incluidos no titulo.
        save_path: Caminho para salvar a figura (None = nao salva).
            Extensao determina formato (png, pdf, svg). Diretorios pais
            sao criados automaticamente.
        show: Se True, chama plt.show() ao final (default True).
        dpi: Resolucao em DPI para salvamento (default 300).

    Raises:
        ValueError: Se 'loss' nao estiver presente no history.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> history = {"loss": [1.0, 0.5, 0.3], "val_loss": [1.1, 0.6, 0.4]}
        >>> plot_training_history(history, show=False)
        >>> # Com save:
        >>> plot_training_history(history, save_path="curves.png", show=False)

    Note:
        Matplotlib importado de forma lazy (suporta ambientes headless).
        Ref: docs/ARCHITECTURE_v2.md secao 9.5.
    """
    # --- Validacao de entrada ---
    if "loss" not in history:
        msg = (
            "Chave 'loss' obrigatoria no dicionario history. "
            f"Chaves disponiveis: {list(history.keys())}"
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

    # --- Detectar chaves disponiveis ---
    loss_key = "loss"
    val_loss_key = _find_key(history, ["val_loss"])
    r2_key = _find_key(history, ["r2", "r2_score", "R2Score"])
    val_r2_key = _find_key(history, ["val_r2", "val_r2_score", "val_R2Score"])
    lr_key = _find_key(history, ["lr", "learning_rate"])
    noise_key = _find_key(history, ["noise_level", "noise_level_max"])

    epochs = np.arange(1, len(history[loss_key]) + 1)
    logger.info(
        "plot_training_history: %d epocas, chaves=%s",
        len(epochs), list(history.keys()),
    )

    # --- Titulo base ---
    base_title = "Training History"
    if config is not None:
        base_title = f"{config.model_type} — {config.loss_type}"

    # --- Criar figura 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=_FIGSIZE_HISTORY)

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Painel 1 (sup-esq): Train/Val Loss                           │
    # │   Curvas de perda por epoca. Val loss tracejado se disponivel.│
    # └────────────────────────────────────────────────────────────────┘
    ax_loss = axes[0, 0]
    ax_loss.plot(
        epochs, history[loss_key],
        color=_COLOR_TRAIN_LOSS, linewidth=1.5, label="Train Loss",
    )
    if val_loss_key is not None:
        ax_loss.plot(
            epochs, history[val_loss_key],
            color=_COLOR_VAL_LOSS, linewidth=1.5, linestyle="--",
            label="Val Loss",
        )
        # D7: Marcar melhor val_loss com estrela
        val_losses = np.array(history[val_loss_key])
        best_epoch = int(np.argmin(val_losses))
        ax_loss.plot(
            epochs[best_epoch], val_losses[best_epoch],
            marker="*", markersize=12, color=_COLOR_VAL_LOSS,
            label=f"Best Val (ep {best_epoch + 1})",
        )
    ax_loss.set_xlabel("Epoca")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Train / Val Loss")
    ax_loss.legend(fontsize=8, loc="upper right")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xlim(epochs[0], epochs[-1])

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Painel 2 (sup-dir): R² por Epoca                             │
    # │   Condicional — placeholder se r2 nao registrado.            │
    # └────────────────────────────────────────────────────────────────┘
    ax_r2 = axes[0, 1]
    if r2_key is not None:
        ax_r2.plot(
            epochs, history[r2_key],
            color=_COLOR_TRAIN_R2, linewidth=1.5, label=f"Train {r2_key}",
        )
        if val_r2_key is not None:
            ax_r2.plot(
                epochs, history[val_r2_key],
                color=_COLOR_VAL_R2, linewidth=1.5, linestyle="--",
                label=f"Val {val_r2_key}",
            )
        ax_r2.axhline(y=1.0, color="gray", linewidth=0.5, linestyle=":")
        ax_r2.set_xlabel("Epoca")
        ax_r2.set_ylabel(r"$R^2$")
        ax_r2.set_title(r"$R^2$ por Epoca")
        ax_r2.legend(fontsize=8, loc="lower right")
        ax_r2.grid(True, alpha=0.3)
        ax_r2.set_xlim(epochs[0], epochs[-1])
    else:
        # D13: Branch — R² nao disponivel → placeholder
        _plot_placeholder(ax_r2, r"$R^2$ nao registrado no history")
        ax_r2.set_title(r"$R^2$ por Epoca (N/A)")
        logger.info("R2 nao encontrado no history — painel 2 como placeholder")

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Painel 3 (inf-esq): LR Schedule                              │
    # │   Condicional — placeholder se lr nao registrado.            │
    # └────────────────────────────────────────────────────────────────┘
    ax_lr = axes[1, 0]
    if lr_key is not None:
        lr_values = np.array(history[lr_key], dtype=np.float64)
        ax_lr.plot(
            epochs, lr_values,
            color=_COLOR_LR, linewidth=1.5, label="Learning Rate",
        )
        ax_lr.set_xlabel("Epoca")
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.set_title("LR Schedule")
        ax_lr.legend(fontsize=8, loc="upper right")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.set_xlim(epochs[0], epochs[-1])
        # D7: Escala log se LR varia mais que 1 ordem de magnitude
        lr_range = lr_values.max() / max(lr_values.min(), _EPS)
        if lr_range > 10.0:
            ax_lr.set_yscale("log")
    else:
        # D13: Branch — LR nao disponivel → placeholder
        _plot_placeholder(ax_lr, "LR nao registrado no history\n(usar LearningRateScheduler callback)")
        ax_lr.set_title("LR Schedule (N/A)")
        logger.info("LR nao encontrado no history — painel 3 como placeholder")

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Painel 4 (inf-dir): Noise Level (Curriculum)                  │
    # │   Condicional — placeholder se noise_level nao registrado.   │
    # └────────────────────────────────────────────────────────────────┘
    ax_noise = axes[1, 1]
    if noise_key is not None:
        noise_values = np.array(history[noise_key], dtype=np.float64)
        ax_noise.fill_between(
            epochs, 0, noise_values,
            color=_COLOR_NOISE, alpha=0.3,
        )
        ax_noise.plot(
            epochs, noise_values,
            color=_COLOR_NOISE, linewidth=1.5, label="Noise Level",
        )
        ax_noise.set_xlabel("Epoca")
        ax_noise.set_ylabel("Noise Level (A/m)")
        ax_noise.set_title("Curriculum Noise")
        ax_noise.legend(fontsize=8, loc="upper left")
        ax_noise.grid(True, alpha=0.3)
        ax_noise.set_xlim(epochs[0], epochs[-1])
        ax_noise.set_ylim(bottom=0)
    else:
        # D13: Branch — noise nao disponivel → placeholder
        _plot_placeholder(
            ax_noise,
            "Noise level nao registrado no history\n(usar UpdateNoiseLevelCallback)",
        )
        ax_noise.set_title("Curriculum Noise (N/A)")
        logger.info("Noise level nao encontrado no history — painel 4 como placeholder")

    # --- Layout e titulo global ---
    fig.suptitle(base_title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path_obj), dpi=dpi, bbox_inches="tight")
        logger.info("Training history salvo em: %s", save_path_obj)

    # --- Exibir se solicitado ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_training_history concluido: %d epocas", len(epochs))


# ──────────────────────────────────────────────────────────────────────
# D2: Funcao secundaria — plot isolado de LR schedule
# ──────────────────────────────────────────────────────────────────────
def plot_lr_schedule(
    history: Dict[str, list],
    *,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """Plota evolucao isolada do learning rate por epoca.

    Plot dedicado para analise detalhada do LR schedule, com escala
    logaritmica automatica quando o LR varia mais de 1 ordem de
    magnitude. Exibe anotacao do LR minimo e maximo.

    Args:
        history: Dicionario history do Keras.
            Deve conter 'lr' ou 'learning_rate'.
        save_path: Caminho para salvar a figura (None = nao salva).
            Extensao determina formato (png, pdf, svg). Diretorios pais
            sao criados automaticamente.
        show: Se True, chama plt.show() ao final (default True).
        dpi: Resolucao em DPI para salvamento (default 300).

    Raises:
        ValueError: Se 'lr' ou 'learning_rate' nao estiver no history.
        ImportError: Se matplotlib nao estiver instalado.

    Example:
        >>> history = {"loss": [1.0, 0.5], "lr": [1e-3, 3e-4]}
        >>> plot_lr_schedule(history, show=False)

    Note:
        Para LR schedule completo no contexto de training curves, use
        ``plot_training_history`` que inclui LR como painel 3.
        Ref: docs/ARCHITECTURE_v2.md secao 9.5.
    """
    # --- Import lazy do matplotlib ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise

    # --- Detectar chave de LR ---
    lr_key = _find_key(history, ["lr", "learning_rate"])
    if lr_key is None:
        msg = (
            "Nenhuma chave de LR encontrada no history. "
            "Chaves esperadas: 'lr' ou 'learning_rate'. "
            f"Chaves disponiveis: {list(history.keys())}"
        )
        raise ValueError(msg)

    lr_values = np.array(history[lr_key], dtype=np.float64)
    epochs = np.arange(1, len(lr_values) + 1)
    logger.info("plot_lr_schedule: %d epocas, LR range [%.2e, %.2e]",
                len(epochs), lr_values.min(), lr_values.max())

    # --- Criar figura ---
    fig, ax = plt.subplots(1, 1, figsize=_FIGSIZE_LR)

    ax.plot(
        epochs, lr_values,
        color=_COLOR_LR, linewidth=1.8, label="Learning Rate",
    )

    # D7: Marcar LR maximo e minimo
    idx_max = int(np.argmax(lr_values))
    idx_min = int(np.argmin(lr_values))
    ax.plot(epochs[idx_max], lr_values[idx_max],
            marker="^", markersize=10, color="green",
            label=f"Max LR={lr_values[idx_max]:.2e} (ep {idx_max + 1})")
    ax.plot(epochs[idx_min], lr_values[idx_min],
            marker="v", markersize=10, color="red",
            label=f"Min LR={lr_values[idx_min]:.2e} (ep {idx_min + 1})")

    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_title("LR Schedule", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(epochs[0], epochs[-1])

    # D7: Escala log se LR varia mais que 1 ordem de magnitude
    lr_range = lr_values.max() / max(lr_values.min(), _EPS)
    if lr_range > 10.0:
        ax.set_yscale("log")

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path_obj), dpi=dpi, bbox_inches="tight")
        logger.info("LR schedule salvo em: %s", save_path_obj)

    # --- Exibir se solicitado ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_lr_schedule concluido: %d epocas", len(epochs))
