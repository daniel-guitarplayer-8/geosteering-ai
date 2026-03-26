# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/optuna_viz.py                                      ║
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
# ║    • Visualizacoes de resultados de otimizacao Optuna (C62)               ║
# ║    • 4 plots: historico, importancia, contorno, coordenadas paralelas    ║
# ║    • Funcao agregadora plot_optuna_results gera todas as 4 figuras       ║
# ║    • Lazy imports para matplotlib E optuna (graceful degradation)        ║
# ║                                                                            ║
# ║  Dependencias: optuna (lazy), matplotlib (lazy), numpy                   ║
# ║                config.py (PipelineConfig via TYPE_CHECKING)              ║
# ║  Exports: ~4 funcoes — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C62 Optuna visualization)  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Visualizacoes de otimizacao Optuna — historico, importancia, contorno, paralelas.

Gera figuras de diagnostico pos-HPO para analisar convergencia e sensibilidade
dos hiperparametros otimizados pelo Optuna:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  VISUALIZACOES OPTUNA (C62)                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. Optimization History (plot_optimization_history)                    │
    │     Objetivo por trial com linha best-so-far sobreposta.               │
    │     Detecta convergencia visual e trials descartados por pruning.      │
    │                                                                         │
    │  2. Parameter Importances (plot_param_importances)                      │
    │     Barra horizontal de importancia relativa por hiperparametro.       │
    │     Requer >= 2 trials completos para fManova.                         │
    │                                                                         │
    │  3. Contour Plot (plot_contour)                                         │
    │     Mapa de contorno 2D para os 2 parametros mais importantes.         │
    │     Revela interacoes e regioes otimas no espaco de busca.             │
    │                                                                         │
    │  4. Parallel Coordinate (plot_parallel_coordinate)                      │
    │     Coordenadas paralelas coloridas por valor do objetivo.             │
    │     Identifica faixas de valores associadas a bom desempenho.          │
    │                                                                         │
    │  Wrapper: plot_optuna_results — gera todas as 4 de uma vez.            │
    └─────────────────────────────────────────────────────────────────────────┘

Optuna e matplotlib sao importados de forma lazy dentro de cada funcao
para suportar ambientes sem essas dependencias instaladas.

Example:
    >>> from geosteering_ai.visualization.optuna_viz import plot_optuna_results
    >>> plot_optuna_results(study, save_dir="/path/to/plots", show=False)

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Optuna e matplotlib importados LAZY dentro de cada funcao.
    Referenciado em:
        - visualization/__init__.py: re-exportado como API publica
        - training/loop.py: diagnostico pos-HPO
        - tests/test_visualization_optuna.py: testes de plotagem
    Ref: docs/ARCHITECTURE_v2.md secao 9 (Visualization).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Funcao agregadora: gera todas as 4 figuras ---
    "plot_optuna_results",
    # --- Funcoes individuais por tipo de plot ---
    "plot_optimization_history",
    "plot_param_importances",
    "plot_contour",
    "plot_parallel_coordinate",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de visualizacao Optuna
# ──────────────────────────────────────────────────────────────────────

# Resolucao default para salvamento de figuras (DPI)
_DEFAULT_DPI = 300

# Tamanhos de figura por tipo de plot
_FIGSIZE_HISTORY = (12, 5)       # largura x altura — optimization history
_FIGSIZE_IMPORTANCES = (10, 6)   # largura x altura — param importances
_FIGSIZE_CONTOUR = (10, 8)       # largura x altura — contour plot
_FIGSIZE_PARALLEL = (14, 6)      # largura x altura — parallel coordinate

# Numero minimo de trials completos para computar importancias
# fManova requer >= 2 trials para funcionar
_MIN_TRIALS_IMPORTANCES = 2

# Numero minimo de trials completos para contour plot
# Precisa de pontos suficientes para interpolar superficie
_MIN_TRIALS_CONTOUR = 3

# Cores para consistencia visual
_COLOR_TRIAL = "#1f77b4"          # azul — valor objetivo por trial
_COLOR_BEST = "#d62728"           # vermelho — melhor valor acumulado
_COLOR_BAR = "#2ca02c"            # verde — barras de importancia
_COLOR_PRUNED = "#cccccc"         # cinza claro — trials podados

# Epsilon para estabilidade numerica (float32)
_EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────
# D2: Helper — import lazy do optuna com mensagem informativa
# ──────────────────────────────────────────────────────────────────────
def _import_optuna() -> Any:
    """Importa optuna de forma lazy com mensagem de erro amigavel.

    Encapsula o import do optuna e do modulo de visualizacao para
    evitar ImportError em ambientes sem optuna instalado.

    Returns:
        Modulo optuna importado.

    Raises:
        ImportError: Se optuna nao estiver instalado, com instrucoes
            de instalacao.

    Note:
        Ref: geosteering_ai/config.py para USE_OPTUNA flag.
    """
    try:
        import optuna
        return optuna
    except ImportError:
        logger.error(
            "optuna nao instalado. Instale com: pip install optuna. "
            "Visualizacoes Optuna requerem optuna >= 3.0."
        )
        raise ImportError(
            "optuna nao instalado. Instale com: pip install optuna"
        )


def _import_matplotlib() -> Any:
    """Importa matplotlib.pyplot de forma lazy com mensagem de erro amigavel.

    Encapsula o import do matplotlib para suportar ambientes headless
    e sem biblioteca grafica instalada.

    Returns:
        Modulo matplotlib.pyplot importado.

    Raises:
        ImportError: Se matplotlib nao estiver instalado, com instrucoes
            de instalacao.

    Note:
        Ref: visualization/training.py para pattern identico.
    """
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.error(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )
        raise ImportError(
            "matplotlib nao instalado. Instale com: pip install matplotlib"
        )


def _get_completed_trials(study: Any) -> List[Any]:
    """Retorna apenas trials com status COMPLETE do estudo Optuna.

    Filtra trials podados (PRUNED) e falhos (FAIL) para funcoes que
    requerem valores de objetivo validos.

    Args:
        study: Objeto optuna.study.Study com historico de otimizacao.

    Returns:
        Lista de trials com status COMPLETE.

    Note:
        Ref: optuna.trial.TrialState para estados possiveis.
    """
    optuna = _import_optuna()
    return [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]


def _get_top_params(
    study: Any,
    n_top: int = 2,
) -> List[str]:
    """Retorna os n parametros mais importantes do estudo.

    Usa optuna.importance.get_param_importances para rankear e
    seleciona os top-n. Fallback para primeiros n parametros do
    melhor trial se importancia nao puder ser computada.

    Args:
        study: Objeto optuna.study.Study com historico de otimizacao.
        n_top: Numero de parametros top a retornar. Default: 2.

    Returns:
        Lista com nomes dos n parametros mais importantes.

    Note:
        Requer >= 2 trials COMPLETE para fManova.
        Ref: optuna.importance.get_param_importances.
    """
    optuna = _import_optuna()
    completed = _get_completed_trials(study)

    if len(completed) < _MIN_TRIALS_IMPORTANCES:
        # Fallback: usar parametros do melhor trial
        logger.warning(
            "Apenas %d trials completos (minimo %d para importancia). "
            "Usando parametros do melhor trial como fallback.",
            len(completed), _MIN_TRIALS_IMPORTANCES,
        )
        best_params = list(study.best_trial.params.keys())
        return best_params[:n_top]

    try:
        importances = optuna.importance.get_param_importances(study)
        param_names = list(importances.keys())
        return param_names[:n_top]
    except Exception as exc:
        logger.warning(
            "Falha ao computar importancia de parametros: %s. "
            "Usando parametros do melhor trial como fallback.",
            exc,
        )
        best_params = list(study.best_trial.params.keys())
        return best_params[:n_top]


# ════════════════════════════════════════════════════════════════════════
# OPTIMIZATION HISTORY — Objetivo por trial com best-so-far
#
# Plota valor do objetivo para cada trial (scatter), com linha
# sobreposta mostrando o melhor valor acumulado ate aquele trial.
# Trials podados sao mostrados como marcadores cinza (se aplicavel).
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Obj ┤                                                              │
# │ 0.8 ┤  o                                                          │
# │ 0.6 ┤    o   o                                                    │
# │ 0.4 ┤  ────o──────o                                               │
# │ 0.2 ┤            o──────────o   ← best-so-far                     │
# │ 0.1 ┤                    o────────────── ← convergencia            │
# │     ├────┬────┬────┬────┬────┬────┬────┤                          │
# │     0    5   10   15   20   25   30                                │
# │                Trial Number                                        │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_optimization_history(
    study: Any,
    *,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C62: Historico de otimizacao — objetivo por trial com best-so-far.

    Plota scatter dos valores do objetivo para cada trial completo,
    com linha sobreposta mostrando o melhor valor acumulado. Permite
    visualizar convergencia e identificar trials de alto desempenho.

    Trials podados (PRUNED) sao plotados como marcadores cinza para
    indicar onde o pruner interrompeu a avaliacao.

    Args:
        study: Objeto ``optuna.study.Study`` com historico de otimizacao.
            Deve conter ao menos 1 trial COMPLETE.
        save_path: Caminho completo para salvar figura (None = nao salva).
            Exemplo: ``"/path/to/optuna_history.png"``.
            Extensao determina formato (png, pdf, svg). Diretorios pais
            sao criados automaticamente.
        show: Se True, chama ``plt.show()`` ao final. Default: True.
        dpi: Resolucao em DPI para salvamento. Default: 300.

    Raises:
        ImportError: Se optuna ou matplotlib nao estiverem instalados.
        ValueError: Se o estudo nao contiver trials completos.

    Example:
        >>> import optuna
        >>> study = optuna.create_study(direction="minimize")
        >>> study.optimize(objective, n_trials=50)
        >>> plot_optimization_history(study, show=False)

    Note:
        Optuna e matplotlib importados LAZY.
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - visualization/optuna_viz.py: chamado por plot_optuna_results
        Ref: docs/ARCHITECTURE_v2.md secao 9.
    """
    optuna = _import_optuna()
    plt = _import_matplotlib()

    # --- Separar trials por estado ---
    completed = _get_completed_trials(study)
    if not completed:
        raise ValueError(
            "Estudo Optuna nao contem trials completos. "
            "Execute study.optimize() antes de visualizar."
        )

    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]

    logger.info(
        "plot_optimization_history: %d trials completos, %d podados",
        len(completed), len(pruned),
    )

    # --- Extrair dados ---
    trial_numbers = [t.number for t in completed]
    trial_values = [t.value for t in completed]

    # Best-so-far (acumulado)
    # D7: direcao do estudo determina se best e min ou max
    is_minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    best_so_far: List[float] = []
    current_best = float("inf") if is_minimize else float("-inf")
    for val in trial_values:
        if is_minimize:
            current_best = min(current_best, val)
        else:
            current_best = max(current_best, val)
        best_so_far.append(current_best)

    # --- Criar figura ---
    fig, ax = plt.subplots(1, 1, figsize=_FIGSIZE_HISTORY)

    # Trials podados (cinza, marker 'x')
    if pruned:
        pruned_numbers = [t.number for t in pruned]
        # Pruned trials nao tem valor final — plotar na posicao do ultimo
        # intermediate value, se disponivel
        pruned_values = []
        for t in pruned:
            if t.intermediate_values:
                # D7: ultimo valor intermediario registrado antes do pruning
                last_step = max(t.intermediate_values.keys())
                pruned_values.append(t.intermediate_values[last_step])
            else:
                pruned_values.append(None)

        # Filtrar trials com valor disponivel
        valid_pruned = [
            (n, v) for n, v in zip(pruned_numbers, pruned_values)
            if v is not None
        ]
        if valid_pruned:
            pn, pv = zip(*valid_pruned)
            ax.scatter(
                pn, pv,
                color=_COLOR_PRUNED, marker="x", s=40,
                alpha=0.5, label="Pruned", zorder=1,
            )

    # Trials completos (scatter azul)
    ax.scatter(
        trial_numbers, trial_values,
        color=_COLOR_TRIAL, marker="o", s=30,
        alpha=0.7, label="Objective Value", zorder=2,
    )

    # Best-so-far (linha vermelha)
    ax.plot(
        trial_numbers, best_so_far,
        color=_COLOR_BEST, linewidth=2.0,
        label="Best So Far", zorder=3,
    )

    # D7: Marcar melhor trial com estrela
    best_idx = int(np.argmin(trial_values) if is_minimize
                    else np.argmax(trial_values))
    ax.plot(
        trial_numbers[best_idx], trial_values[best_idx],
        marker="*", markersize=15, color=_COLOR_BEST,
        label=f"Best (trial {trial_numbers[best_idx]}, "
              f"val={trial_values[best_idx]:.6f})",
        zorder=4,
    )

    # Labels e formatacao
    ax.set_xlabel("Trial Number", fontsize=11)
    ax.set_ylabel("Objective Value", fontsize=11)
    ax.set_title(
        f"Optimization History ({len(completed)} trials, "
        f"{'minimize' if is_minimize else 'maximize'})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("Optimization history salvo em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info(
        "plot_optimization_history concluido: %d trials, best=%.6f",
        len(completed), trial_values[best_idx],
    )


# ════════════════════════════════════════════════════════════════════════
# PARAMETER IMPORTANCES — Barra horizontal de importancia relativa
#
# Usa fManova (default do Optuna) para computar importancia de cada
# hiperparametro. Exibe barra horizontal ordenada do mais ao menos
# importante, com valor numerico anotado em cada barra.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  learning_rate  ████████████████████████  0.412                    │
# │  dropout_rate   ██████████████████       0.298                     │
# │  n_layers       █████████████            0.201                     │
# │  batch_size     ████████                 0.089                     │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_param_importances(
    study: Any,
    *,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C62: Importancia de hiperparametros — barra horizontal ordenada.

    Usa fManova (default do Optuna) para computar importancia relativa
    de cada hiperparametro otimizado. Exibe barras horizontais ordenadas
    do mais ao menos importante, com valor numerico anotado.

    Requer ao menos 2 trials COMPLETE para computar importancias.

    Args:
        study: Objeto ``optuna.study.Study`` com historico de otimizacao.
            Deve conter >= 2 trials COMPLETE.
        save_path: Caminho completo para salvar figura (None = nao salva).
            Exemplo: ``"/path/to/param_importances.png"``.
        show: Se True, chama ``plt.show()`` ao final. Default: True.
        dpi: Resolucao em DPI para salvamento. Default: 300.

    Raises:
        ImportError: Se optuna ou matplotlib nao estiverem instalados.
        ValueError: Se o estudo nao contiver trials completos suficientes.

    Example:
        >>> plot_param_importances(study, save_path="importances.png")

    Note:
        Optuna e matplotlib importados LAZY.
        fManova requer >= 2 trials COMPLETE para funcionar.
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - visualization/optuna_viz.py: chamado por plot_optuna_results
        Ref: docs/ARCHITECTURE_v2.md secao 9.
        Ref: Akiba, T. et al. (2019). Optuna: A Next-gen HPO Framework.
    """
    optuna = _import_optuna()
    plt = _import_matplotlib()

    completed = _get_completed_trials(study)
    if len(completed) < _MIN_TRIALS_IMPORTANCES:
        raise ValueError(
            f"Importancia requer >= {_MIN_TRIALS_IMPORTANCES} trials "
            f"completos. Estudo contem apenas {len(completed)}."
        )

    logger.info(
        "plot_param_importances: computando importancia com %d trials",
        len(completed),
    )

    # --- Computar importancias ---
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:
        logger.error(
            "Falha ao computar importancia de parametros: %s", exc,
        )
        raise

    if not importances:
        logger.warning("Nenhuma importancia computada — estudo sem parametros?")
        return

    # --- Ordenar por importancia (descendente para plot) ---
    param_names = list(importances.keys())
    param_values = list(importances.values())

    # Invertir para que o mais importante fique no topo
    param_names_sorted = list(reversed(param_names))
    param_values_sorted = list(reversed(param_values))

    # --- Criar figura ---
    n_params = len(param_names_sorted)
    fig_height = max(3, 0.5 * n_params + 2)  # D7: altura adaptativa
    fig, ax = plt.subplots(
        1, 1, figsize=(_FIGSIZE_IMPORTANCES[0], fig_height),
    )

    # Barra horizontal
    y_positions = np.arange(n_params)
    bars = ax.barh(
        y_positions, param_values_sorted,
        color=_COLOR_BAR, alpha=0.8, edgecolor="white",
        height=0.6,
    )

    # Anotacoes de valor numerico em cada barra
    for i, (bar, val) in enumerate(zip(bars, param_values_sorted)):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=9,
        )

    # Labels e formatacao
    ax.set_yticks(y_positions)
    ax.set_yticklabels(param_names_sorted, fontsize=10)
    ax.set_xlabel("Relative Importance", fontsize=11)
    ax.set_title(
        f"Hyperparameter Importances ({len(completed)} trials)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, max(param_values_sorted) * 1.15)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("Param importances salvo em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info(
        "plot_param_importances concluido: %d parametros, "
        "top=%s (%.3f)",
        n_params, param_names[0], param_values[0],
    )


# ════════════════════════════════════════════════════════════════════════
# CONTOUR PLOT — Mapa 2D para top 2 parametros
#
# Plota mapa de contorno 2D do objetivo em funcao dos 2 parametros
# mais importantes. Cada ponto e um trial completo; a superficie
# e interpolada via scatter tricontourf.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  param_2  ┤                                                        │
# │      0.5  ┤   ░░░▒▒▓▓██                                          │
# │      0.4  ┤  ░░▒▒▓▓████  ← regiao otima                         │
# │      0.3  ┤ ░░▒▒▓▓█████                                          │
# │      0.2  ┤  ░░▒▒▓▓████                                          │
# │      0.1  ┤   ░░░▒▒▓▓██                                          │
# │           ├────┬────┬────┬────┬────┤                              │
# │           1e-4 3e-4 1e-3 3e-3 1e-2                                │
# │                    param_1                                         │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_contour(
    study: Any,
    *,
    params: Optional[Tuple[str, str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C62: Contour plot 2D para os 2 parametros mais importantes.

    Plota mapa de contorno do objetivo em funcao de 2 hiperparametros,
    com scatter overlay dos trials. Por default, seleciona automaticamente
    os 2 parametros mais importantes via fManova.

    Cada ponto e um trial completo. A superficie de contorno e
    interpolada via ``tricontourf`` (triangulacao de Delaunay).

    Args:
        study: Objeto ``optuna.study.Study`` com historico de otimizacao.
            Deve conter >= 3 trials COMPLETE.
        params: Tupla com nomes dos 2 parametros a plotar.
            Se None, seleciona automaticamente os 2 mais importantes.
            Exemplo: ``("learning_rate", "dropout_rate")``.
        save_path: Caminho completo para salvar figura (None = nao salva).
            Exemplo: ``"/path/to/contour.png"``.
        show: Se True, chama ``plt.show()`` ao final. Default: True.
        dpi: Resolucao em DPI para salvamento. Default: 300.

    Raises:
        ImportError: Se optuna ou matplotlib nao estiverem instalados.
        ValueError: Se o estudo nao contiver trials completos suficientes.
        ValueError: Se parametros especificados nao existirem no estudo.

    Example:
        >>> plot_contour(study, params=("learning_rate", "dropout_rate"))
        >>> # Auto-selecao dos 2 mais importantes:
        >>> plot_contour(study, save_path="contour.png", show=False)

    Note:
        Optuna e matplotlib importados LAZY.
        Requer >= 3 trials COMPLETE para interpolacao.
        Parametros categoricos sao mapeados a indices inteiros.
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - visualization/optuna_viz.py: chamado por plot_optuna_results
        Ref: docs/ARCHITECTURE_v2.md secao 9.
    """
    _import_optuna()
    plt = _import_matplotlib()

    completed = _get_completed_trials(study)
    if len(completed) < _MIN_TRIALS_CONTOUR:
        raise ValueError(
            f"Contour plot requer >= {_MIN_TRIALS_CONTOUR} trials "
            f"completos. Estudo contem apenas {len(completed)}."
        )

    # --- Determinar parametros ---
    if params is None:
        top_params = _get_top_params(study, n_top=2)
        if len(top_params) < 2:
            raise ValueError(
                "Contour plot requer >= 2 parametros no estudo. "
                f"Encontrados: {top_params}"
            )
        param_x, param_y = top_params[0], top_params[1]
    else:
        param_x, param_y = params

    # Validar que parametros existem
    all_params = set()
    for t in completed:
        all_params.update(t.params.keys())

    for p in (param_x, param_y):
        if p not in all_params:
            raise ValueError(
                f"Parametro '{p}' nao encontrado no estudo. "
                f"Parametros disponiveis: {sorted(all_params)}"
            )

    logger.info(
        "plot_contour: param_x='%s', param_y='%s', %d trials",
        param_x, param_y, len(completed),
    )

    # --- Extrair dados ---
    x_vals: List[float] = []
    y_vals: List[float] = []
    z_vals: List[float] = []

    # D7: Mapear categoricos a indices para permitir contour
    x_categories: Optional[List[str]] = None
    y_categories: Optional[List[str]] = None

    for t in completed:
        if param_x not in t.params or param_y not in t.params:
            continue
        x_raw = t.params[param_x]
        y_raw = t.params[param_y]

        # Detectar categoricos (strings)
        if isinstance(x_raw, str):
            if x_categories is None:
                x_categories = sorted(
                    set(
                        tt.params[param_x]
                        for tt in completed
                        if param_x in tt.params
                        and isinstance(tt.params[param_x], str)
                    )
                )
            x_vals.append(float(x_categories.index(x_raw)))
        else:
            x_vals.append(float(x_raw))

        if isinstance(y_raw, str):
            if y_categories is None:
                y_categories = sorted(
                    set(
                        tt.params[param_y]
                        for tt in completed
                        if param_y in tt.params
                        and isinstance(tt.params[param_y], str)
                    )
                )
            y_vals.append(float(y_categories.index(y_raw)))
        else:
            y_vals.append(float(y_raw))

        z_vals.append(t.value)

    if len(x_vals) < _MIN_TRIALS_CONTOUR:
        raise ValueError(
            f"Trials insuficientes com ambos parametros ({param_x}, {param_y}) "
            f"definidos. Encontrados: {len(x_vals)}, minimo: {_MIN_TRIALS_CONTOUR}."
        )

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    z_arr = np.array(z_vals)

    # --- Criar figura ---
    fig, ax = plt.subplots(1, 1, figsize=_FIGSIZE_CONTOUR)

    # Contour preenchido via tricontourf (nao requer grid regular)
    try:
        tcf = ax.tricontourf(x_arr, y_arr, z_arr, levels=15, cmap="viridis")
        fig.colorbar(tcf, ax=ax, label="Objective Value")
    except Exception as exc:
        # D13: Branch — tricontourf falha com pontos colineares
        logger.warning(
            "tricontourf falhou (%s) — plotando apenas scatter.", exc,
        )

    # Scatter overlay dos trials
    scatter = ax.scatter(
        x_arr, y_arr,
        c=z_arr, cmap="viridis", edgecolors="white",
        linewidths=0.5, s=50, zorder=5,
    )

    # D7: Marcar melhor trial
    is_minimize = True
    try:
        import optuna as _opt
        is_minimize = (
            study.direction == _opt.study.StudyDirection.MINIMIZE
        )
    except Exception:
        pass

    best_idx = int(np.argmin(z_arr) if is_minimize else np.argmax(z_arr))
    ax.plot(
        x_arr[best_idx], y_arr[best_idx],
        marker="*", markersize=18, color=_COLOR_BEST,
        markeredgecolor="white", markeredgewidth=1.0,
        label=f"Best (obj={z_arr[best_idx]:.6f})",
        zorder=6,
    )

    # Labels
    x_label = param_x
    y_label = param_y
    if x_categories is not None:
        ax.set_xticks(range(len(x_categories)))
        ax.set_xticklabels(x_categories, rotation=45, ha="right")
        x_label = f"{param_x} (categorical)"
    if y_categories is not None:
        ax.set_yticks(range(len(y_categories)))
        ax.set_yticklabels(y_categories)
        y_label = f"{param_y} (categorical)"

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(
        f"Contour — {param_x} vs {param_y} ({len(x_vals)} trials)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("Contour plot salvo em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info(
        "plot_contour concluido: %s vs %s, %d trials",
        param_x, param_y, len(x_vals),
    )


# ════════════════════════════════════════════════════════════════════════
# PARALLEL COORDINATE — Coordenadas paralelas coloridas por objetivo
#
# Cada trial e uma linha passando por eixos paralelos (1 por parametro).
# Cor indica valor do objetivo (azul = bom, vermelho = ruim).
# Identifica faixas de valores associadas a bom desempenho.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  lr      dropout   n_layers  batch_sz  │ Objective                │
# │  │         │         │         │       │                           │
# │  ├─────────┼─────────┼─────────┤       │  ▓▓▓ 0.10 (bom)         │
# │  │╲        │        ╱│╲        │       │  ░░░ 0.50 (ruim)        │
# │  │ ╲───────┼──────╱  │ ╲───────┤       │                          │
# │  │  ╲      │    ╱    │  ╲      │       │                          │
# │  ├─────────┼─────────┼─────────┤       │                          │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_parallel_coordinate(
    study: Any,
    *,
    params: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C62: Coordenadas paralelas coloridas pelo valor do objetivo.

    Plota cada trial como uma linha polilinha passando por eixos
    paralelos (um por hiperparametro), colorida pelo valor do objetivo.
    Permite identificar visualmente faixas de parametros associadas
    a bom desempenho.

    Args:
        study: Objeto ``optuna.study.Study`` com historico de otimizacao.
            Deve conter >= 1 trial COMPLETE.
        params: Lista de nomes de parametros a incluir. Se None,
            inclui todos os parametros do melhor trial.
        save_path: Caminho completo para salvar figura (None = nao salva).
            Exemplo: ``"/path/to/parallel.png"``.
        show: Se True, chama ``plt.show()`` ao final. Default: True.
        dpi: Resolucao em DPI para salvamento. Default: 300.

    Raises:
        ImportError: Se optuna ou matplotlib nao estiverem instalados.
        ValueError: Se o estudo nao contiver trials completos.

    Example:
        >>> plot_parallel_coordinate(study, show=False)
        >>> # Com parametros especificos:
        >>> plot_parallel_coordinate(
        ...     study,
        ...     params=["learning_rate", "dropout_rate", "n_layers"],
        ...     save_path="parallel.png",
        ... )

    Note:
        Optuna e matplotlib importados LAZY.
        Parametros categoricos sao mapeados a indices inteiros.
        Referenciado em:
            - visualization/__init__.py: re-exportado
            - visualization/optuna_viz.py: chamado por plot_optuna_results
        Ref: docs/ARCHITECTURE_v2.md secao 9.
    """
    optuna = _import_optuna()
    plt = _import_matplotlib()
    from matplotlib.collections import LineCollection  # noqa: E402

    completed = _get_completed_trials(study)
    if not completed:
        raise ValueError(
            "Estudo Optuna nao contem trials completos. "
            "Execute study.optimize() antes de visualizar."
        )

    # --- Determinar parametros a plotar ---
    if params is None:
        params = list(study.best_trial.params.keys())

    if not params:
        raise ValueError(
            "Nenhum parametro disponivel para parallel coordinate. "
            "O melhor trial nao possui parametros."
        )

    n_params = len(params)
    logger.info(
        "plot_parallel_coordinate: %d parametros, %d trials",
        n_params, len(completed),
    )

    # --- Mapear categoricos e extrair valores ---
    # Para cada parametro, categoricos sao mapeados a indices
    category_maps: dict = {}  # param_name -> {value: index}
    for p in params:
        cat_values = set()
        for t in completed:
            if p in t.params and isinstance(t.params[p], str):
                cat_values.add(t.params[p])
        if cat_values:
            sorted_cats = sorted(cat_values)
            category_maps[p] = {v: i for i, v in enumerate(sorted_cats)}

    # Extrair matriz de valores (trials x params) + vetor de objetivos
    data_rows: List[List[float]] = []
    obj_values: List[float] = []

    for t in completed:
        row: List[float] = []
        skip = False
        for p in params:
            if p not in t.params:
                skip = True
                break
            val = t.params[p]
            if p in category_maps:
                row.append(float(category_maps[p].get(val, 0)))
            else:
                row.append(float(val))
        if not skip:
            data_rows.append(row)
            obj_values.append(t.value)

    if not data_rows:
        logger.warning(
            "Nenhum trial com todos os parametros %s definidos.", params,
        )
        return

    data = np.array(data_rows)  # (n_trials, n_params)
    objectives = np.array(obj_values)

    # --- Normalizar cada eixo para [0, 1] ---
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    # Evitar divisao por zero em parametros constantes
    ranges[ranges < _EPS] = 1.0
    data_norm = (data - mins) / ranges

    # --- Criar figura ---
    fig, ax = plt.subplots(1, 1, figsize=_FIGSIZE_PARALLEL)

    # Normalizar objetivos para colormap
    obj_min = objectives.min()
    obj_max = objectives.max()
    obj_range = obj_max - obj_min
    if obj_range < _EPS:
        obj_norm = np.zeros_like(objectives)
    else:
        obj_norm = (objectives - obj_min) / obj_range

    # D7: Inverter colormap se minimize (azul = bom = valor baixo)
    is_minimize = True
    try:
        is_minimize = (
            study.direction == optuna.study.StudyDirection.MINIMIZE
        )
    except Exception:
        pass

    cmap = plt.cm.viridis_r if is_minimize else plt.cm.viridis

    # Plotar cada trial como polilinha
    x_positions = np.arange(n_params)
    for i in range(len(data_norm)):
        color = cmap(obj_norm[i])
        ax.plot(
            x_positions, data_norm[i],
            color=color, alpha=0.4, linewidth=1.2,
        )

    # Eixos verticais
    for j in range(n_params):
        ax.axvline(x=j, color="gray", linewidth=0.5, alpha=0.5)

    # Labels dos eixos com valores originais
    ax.set_xticks(x_positions)
    x_labels: List[str] = []
    for j, p in enumerate(params):
        if p in category_maps:
            cats = sorted(category_maps[p].keys())
            x_labels.append(f"{p}\n({', '.join(cats[:3])}{'...' if len(cats) > 3 else ''})")
        else:
            x_labels.append(f"{p}\n[{mins[j]:.4g}, {maxs[j]:.4g}]")
    ax.set_xticklabels(x_labels, fontsize=8, ha="center")

    ax.set_ylabel("Normalized Value", fontsize=11)
    ax.set_title(
        f"Parallel Coordinate ({len(data_rows)} trials, "
        f"{n_params} params)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=obj_min, vmax=obj_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Objective Value", fontsize=10)

    fig.tight_layout()

    # --- Salvar se solicitado ---
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=dpi, bbox_inches="tight")
        logger.info("Parallel coordinate salvo em: %s", save_p)

    # --- Exibir ou fechar ---
    if show:
        plt.show()
    else:
        plt.close(fig)

    logger.info(
        "plot_parallel_coordinate concluido: %d trials, %d params",
        len(data_rows), n_params,
    )


# ════════════════════════════════════════════════════════════════════════
# WRAPPER — Gera todas as 4 visualizacoes de uma vez
#
# Funcao de conveniencia que chama as 4 funcoes individuais em sequencia,
# salvando cada figura em save_dir com nomes padronizados.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  plot_optuna_results(study, save_dir="plots/optuna/")              │
# │  ├── 1. optuna_history.png       (plot_optimization_history)       │
# │  ├── 2. optuna_importances.png   (plot_param_importances)          │
# │  ├── 3. optuna_contour.png       (plot_contour)                    │
# │  └── 4. optuna_parallel.png      (plot_parallel_coordinate)        │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def plot_optuna_results(
    study: Any,
    *,
    save_dir: Optional[str] = None,
    show: bool = True,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """C62: Gera todas as 4 visualizacoes Optuna de uma vez.

    Funcao de conveniencia que chama em sequencia:
        1. ``plot_optimization_history`` — objetivo por trial
        2. ``plot_param_importances`` — importancia de parametros
        3. ``plot_contour`` — mapa 2D dos top 2 parametros
        4. ``plot_parallel_coordinate`` — coordenadas paralelas

    Cada funcao e chamada dentro de um try/except individual para
    que falha em um plot nao impeça a geracao dos demais.

    Args:
        study: Objeto ``optuna.study.Study`` com historico de otimizacao.
            Deve conter ao menos 1 trial COMPLETE.
        save_dir: Diretorio para salvar as 4 figuras (None = nao salva).
            Exemplo: ``"/path/to/plots/optuna/"``.
            Arquivos gerados:
                - ``optuna_history.png``
                - ``optuna_importances.png``
                - ``optuna_contour.png``
                - ``optuna_parallel.png``
        show: Se True, chama ``plt.show()`` em cada plot. Default: True.
        dpi: Resolucao em DPI para salvamento. Default: 300.

    Raises:
        ImportError: Se optuna ou matplotlib nao estiverem instalados.
            Erro emitido ANTES de qualquer plot — falha rapida.

    Example:
        >>> import optuna
        >>> study = optuna.create_study(direction="minimize")
        >>> study.optimize(objective, n_trials=50)
        >>> from geosteering_ai.visualization.optuna_viz import plot_optuna_results
        >>> plot_optuna_results(study, save_dir="plots/optuna/", show=False)

    Note:
        Optuna e matplotlib importados LAZY.
        Cada sub-plot e independente: falha em importances nao impede
        contour ou parallel. Warnings sao emitidos para plots que falharam.
        Referenciado em:
            - visualization/__init__.py: re-exportado como API publica
            - training/loop.py: diagnostico pos-HPO
        Ref: docs/ARCHITECTURE_v2.md secao 9.
    """
    # Verificar dependencias antes de iniciar qualquer plot
    _import_optuna()
    _import_matplotlib()

    completed = _get_completed_trials(study)
    logger.info(
        "plot_optuna_results: gerando 4 visualizacoes para %d trials completos",
        len(completed),
    )

    # --- Preparar caminhos ---
    save_paths = {
        "history": None,
        "importances": None,
        "contour": None,
        "parallel": None,
    }
    if save_dir is not None:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        save_paths["history"] = str(save_dir_path / "optuna_history.png")
        save_paths["importances"] = str(
            save_dir_path / "optuna_importances.png"
        )
        save_paths["contour"] = str(save_dir_path / "optuna_contour.png")
        save_paths["parallel"] = str(save_dir_path / "optuna_parallel.png")

    n_success = 0
    n_failed = 0

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Plot 1: Optimization History                                   │
    # │   Objetivo por trial com best-so-far. Requer >= 1 trial.     │
    # └────────────────────────────────────────────────────────────────┘
    try:
        plot_optimization_history(
            study,
            save_path=save_paths["history"],
            show=show,
            dpi=dpi,
        )
        n_success += 1
    except Exception as exc:
        logger.warning("Optimization history falhou: %s", exc)
        n_failed += 1

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Plot 2: Parameter Importances                                  │
    # │   Barra horizontal. Requer >= 2 trials (fManova).            │
    # └────────────────────────────────────────────────────────────────┘
    try:
        plot_param_importances(
            study,
            save_path=save_paths["importances"],
            show=show,
            dpi=dpi,
        )
        n_success += 1
    except Exception as exc:
        logger.warning("Param importances falhou: %s", exc)
        n_failed += 1

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Plot 3: Contour Plot                                           │
    # │   Mapa 2D para top 2 parametros. Requer >= 3 trials.         │
    # └────────────────────────────────────────────────────────────────┘
    try:
        plot_contour(
            study,
            save_path=save_paths["contour"],
            show=show,
            dpi=dpi,
        )
        n_success += 1
    except Exception as exc:
        logger.warning("Contour plot falhou: %s", exc)
        n_failed += 1

    # ┌────────────────────────────────────────────────────────────────┐
    # │ Plot 4: Parallel Coordinate                                    │
    # │   Polilinhas coloridas por objetivo. Requer >= 1 trial.      │
    # └────────────────────────────────────────────────────────────────┘
    try:
        plot_parallel_coordinate(
            study,
            save_path=save_paths["parallel"],
            show=show,
            dpi=dpi,
        )
        n_success += 1
    except Exception as exc:
        logger.warning("Parallel coordinate falhou: %s", exc)
        n_failed += 1

    logger.info(
        "plot_optuna_results concluido: %d/4 plots gerados, %d falhas",
        n_success, n_failed,
    )
    if save_dir is not None:
        logger.info("Figuras salvas em: %s", save_dir)
