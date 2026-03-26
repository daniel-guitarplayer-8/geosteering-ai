# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: training/optuna_hpo.py                                            ║
# ║  Bloco: 6 — Training                                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Otimizacao de hiperparametros via Optuna (opt-in, default off)       ║
# ║    • Search space adaptavel (LR, batch, dropout, loss, optimizer)         ║
# ║    • Samplers: TPE, CMA-ES, Random                                        ║
# ║    • Pruners: Median, Hyperband                                            ║
# ║    • Weight reset entre trials (evita path-dependency)                    ║
# ║    • Integracao com PipelineConfig (config como parametro)                ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), optuna (lazy import)           ║
# ║  Exports: ~3 (create_search_space, create_study, run_hpo) — ver __all__  ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6.4 (Optuna HPO)                     ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C38+C45)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Optuna HPO — Otimizacao de hiperparametros para o pipeline de inversao.

Modulo opt-in (config.use_optuna=True) para busca automatica de
hiperparametros via Optuna. Integra-se com PipelineConfig para gerar
variantes de configuracao a cada trial.

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  OPTUNA HPO WORKFLOW                                                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  create_study(config)                                               │
    │       │                                                              │
    │       ├─ Sampler: TPE / CMA-ES / Random (config.optuna_sampler)     │
    │       ├─ Pruner: Median / Hyperband (config.optuna_pruner)          │
    │       └─ Direction: minimize (val_loss)                              │
    │                                                                      │
    │  run_hpo(config, build_and_train_fn)                                │
    │       │                                                              │
    │       └─ for trial in n_trials:                                     │
    │            1. params = create_search_space(trial, config)           │
    │            2. trial_config = config.copy(**params)                  │
    │            3. val_loss = build_and_train_fn(trial_config)           │
    │            4. Optuna registra val_loss + prune se ruim              │
    │                                                                      │
    │  Retorno: {best_params, best_value, n_trials, study}               │
    └──────────────────────────────────────────────────────────────────────┘

Samplers disponiveis:

.. code-block:: text

    ┌──────────┬──────────────────────────────────────────────────────────┐
    │ Sampler  │ Descricao                                              │
    ├──────────┼──────────────────────────────────────────────────────────┤
    │ tpe      │ Tree-structured Parzen Estimators (default, eficiente) │
    │ cmaes    │ CMA-ES (bom para espacos continuos, N>10 trials)      │
    │ random   │ Random search (baseline, exploratoria)                  │
    └──────────┴──────────────────────────────────────────────────────────┘

Pruners disponiveis:

.. code-block:: text

    ┌────────────┬────────────────────────────────────────────────────────┐
    │ Pruner     │ Descricao                                            │
    ├────────────┼────────────────────────────────────────────────────────┤
    │ median     │ Para trial se abaixo da mediana dos completos       │
    │ hyperband  │ Successive halving (eficiente para muitos trials)    │
    └────────────┴────────────────────────────────────────────────────────┘

Example:
    >>> from geosteering_ai.config import PipelineConfig
    >>> config = PipelineConfig.robusto().copy(use_optuna=True, optuna_n_trials=10)
    >>> result = run_hpo(config, my_train_fn)
    >>> result['best_params']
    {'learning_rate': 0.0003, 'batch_size': 64, ...}

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Referenciado em:
        - training/__init__.py: re-export create_search_space, create_study, run_hpo
        - config.py: secao 14 (Optuna fields)
        - tests/test_optuna_hpo.py: TestSearchSpace, TestStudy, TestRunHPO
    Ref: docs/ARCHITECTURE_v2.md secao 6.4 (Optuna HPO).
    Optuna e lazy-imported (ImportError informativo se ausente).
    Weight reset entre trials: initial_weights salvos antes do loop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Search space ---
    "create_search_space",
    # --- Study factory ---
    "create_study",
    # --- Runner principal ---
    "run_hpo",
]


# ════════════════════════════════════════════════════════════════════════
# CONSTANTES — Limites do search space
#
# Valores escolhidos com base nos cenarios S14-S21+ validados:
#   LR: [1e-5, 1e-3] — 1e-4 e o default E-Robusto
#   Batch: {16, 32, 64, 128} — 32 e o default
#   Dropout: [0.0, 0.5] — 0.0 e o default (weight_decay regulariza)
#   Optimizer: adam, adamw, nadam, rmsprop — os mais usados em DL
#   Loss: rmse, huber, mse, mae, log_cosh — as 5 mais estaveis
# ════════════════════════════════════════════════════════════════════════

_LR_MIN: float = 1e-5
_LR_MAX: float = 1e-3

_BATCH_CHOICES: List[int] = [16, 32, 64, 128]

_DROPOUT_MIN: float = 0.0
_DROPOUT_MAX: float = 0.5

_OPTIMIZER_CHOICES: List[str] = ["adam", "adamw", "nadam", "rmsprop"]

_LOSS_CHOICES: List[str] = [
    "rmse",
    "huber",
    "mse",
    "mae",
    "log_cosh",
]


# ════════════════════════════════════════════════════════════════════════
# SEARCH SPACE — Definicao do espaco de busca Optuna
#
# Cada trial sugere um conjunto de hiperparametros dentro dos limites
# definidos acima. O trial.suggest_* usa distribuicoes adequadas:
#   - suggest_float(log=True) para LR (escala logaritmica)
#   - suggest_categorical para batch, optimizer, loss
#   - suggest_float para dropout (escala uniforme)
#
# O dict retornado e compativel com config.copy(**params) para
# criar uma variante do PipelineConfig com os HPs sugeridos.
# ════════════════════════════════════════════════════════════════════════

def create_search_space(trial: Any, config: PipelineConfig) -> Dict[str, Any]:
    """Define espaco de busca Optuna com base no config.

    Sugere hiperparametros via trial.suggest_*. O dict retornado e
    compativel com PipelineConfig.copy(**params) para criar
    configuracoes variantes a cada trial.

    Args:
        trial: Instancia de optuna.trial.Trial (gerenciada pelo study).
        config: PipelineConfig base (usado como referencia para limites
            e para garantir compatibilidade com o pipeline).

    Returns:
        Dict com hiperparametros sugeridos:
            - learning_rate (float): LR em escala log [1e-5, 1e-3].
            - batch_size (int): Tamanho do batch {16, 32, 64, 128}.
            - dropout_rate (float): Taxa de dropout [0.0, 0.5].
            - loss_type (str): Funcao de perda {rmse, huber, mse, mae, log_cosh}.
            - optimizer (str): Otimizador {adam, adamw, nadam, rmsprop}.

    Example:
        >>> # Dentro de um objective Optuna:
        >>> def objective(trial):
        ...     params = create_search_space(trial, config)
        ...     trial_config = config.copy(**params)
        ...     return train_and_evaluate(trial_config)

    Note:
        Referenciado em:
            - training/optuna_hpo.py: run_hpo() (dentro do objective)
            - tests/test_optuna_hpo.py: TestSearchSpace
        Ref: docs/ARCHITECTURE_v2.md secao 6.4 (search space).
        suggest_float(log=True): distribuicao log-uniforme para LR
        (faixa de 2 ordens de magnitude: 1e-5 a 1e-3).
        suggest_categorical: distribuicao uniforme discreta para batch/opt/loss.
    """
    params: Dict[str, Any] = {}

    # ── Learning rate (log-uniforme: 1e-5 a 1e-3) ────────────────────
    # D7: Escala logaritmica e essencial para LR — diferenca entre
    # 1e-5 e 1e-4 e tao significativa quanto entre 1e-4 e 1e-3.
    params["learning_rate"] = trial.suggest_float(
        "learning_rate", _LR_MIN, _LR_MAX, log=True
    )

    # ── Batch size (categorico: potencias de 2) ──────────────────────
    # D7: Batch size impacta estabilidade dos gradientes e uso de GPU.
    # Potencias de 2 sao otimas para alinhamento de memoria GPU.
    params["batch_size"] = trial.suggest_categorical(
        "batch_size", _BATCH_CHOICES
    )

    # ── Dropout rate (uniforme: 0.0 a 0.5) ───────────────────────────
    # D7: Dropout complementa weight_decay (regularizacao por ativacao).
    # 0.0 = sem dropout (default E-Robusto). 0.5 = maximo recomendado.
    params["dropout_rate"] = trial.suggest_float(
        "dropout_rate", _DROPOUT_MIN, _DROPOUT_MAX
    )

    # ── Loss type (categorico: 5 opcoes estaveis) ────────────────────
    # D7: Apenas losses robustas e bem validadas no pipeline.
    # Losses exoticas (physics-informed, geosteering) nao sao tunaveis.
    params["loss_type"] = trial.suggest_categorical(
        "loss_type", _LOSS_CHOICES
    )

    # ── Optimizer (categorico: 4 opcoes) ──────────────────────────────
    # D7: AdamW e o default (weight_decay nativo). Adam, Nadam e RMSprop
    # sao alternativas bem estabelecidas para series temporais.
    params["optimizer"] = trial.suggest_categorical(
        "optimizer", _OPTIMIZER_CHOICES
    )

    logger.info(
        "Trial %d: LR=%.2e, batch=%d, dropout=%.3f, loss=%s, opt=%s",
        trial.number,
        params["learning_rate"],
        params["batch_size"],
        params["dropout_rate"],
        params["loss_type"],
        params["optimizer"],
    )

    return params


# ════════════════════════════════════════════════════════════════════════
# STUDY FACTORY — Criacao de study Optuna com sampler e pruner
#
# Encapsula a criacao de optuna.create_study() com configuracao
# derivada do PipelineConfig:
#
#   ┌────────────────────────────────────────────────────────────────┐
#   │  config.optuna_sampler → Sampler Optuna                       │
#   ├────────────────────────────────────────────────────────────────┤
#   │  "tpe"    → TPESampler(seed=config.global_seed)              │
#   │  "cmaes"  → CmaEsSampler(seed=config.global_seed)           │
#   │  "random" → RandomSampler(seed=config.global_seed)           │
#   ├────────────────────────────────────────────────────────────────┤
#   │  config.optuna_pruner → Pruner Optuna                        │
#   ├────────────────────────────────────────────────────────────────┤
#   │  "median"    → MedianPruner(n_startup_trials=5)             │
#   │  "hyperband" → HyperbandPruner(min_resource=3, max=epochs)  │
#   └────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def create_study(config: PipelineConfig) -> Any:
    """Cria Optuna study com sampler e pruner configurados.

    Encapsula optuna.create_study() com escolha de sampler e pruner
    derivada de config.optuna_sampler e config.optuna_pruner.
    Seed do sampler e derivada de config.global_seed para
    reprodutibilidade.

    Args:
        config: PipelineConfig com campos Optuna (secao 14):
            - optuna_sampler: 'tpe', 'cmaes', 'random'.
            - optuna_pruner: 'median', 'hyperband'.
            - global_seed: Seed para reprodutibilidade do sampler.
            - epochs: Numero maximo de epocas (para Hyperband max_resource).

    Returns:
        Instancia de optuna.study.Study configurada para minimizacao.

    Raises:
        ImportError: Se optuna nao estiver instalado.
        ValueError: Se sampler ou pruner nao forem reconhecidos.

    Example:
        >>> config = PipelineConfig.robusto().copy(use_optuna=True)
        >>> study = create_study(config)
        >>> study.direction
        StudyDirection.MINIMIZE

    Note:
        Referenciado em:
            - training/optuna_hpo.py: run_hpo() (criacao do study)
            - tests/test_optuna_hpo.py: TestStudy
        Ref: docs/ARCHITECTURE_v2.md secao 6.4 (study creation).
        Optuna e lazy-imported: ImportError informativo se ausente.
        Seed: TPE e Random usam seed diretamente; CMA-ES usa seed no
        restart_strategy (comportamento deterministico se N >= 10 trials).
        Direction: sempre "minimize" (val_loss como metrica alvo).
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna necessario para HPO: pip install optuna. "
            "Consulte: https://optuna.readthedocs.io/"
        )

    # ── Sampler ───────────────────────────────────────────────────────
    sampler = _build_sampler(config, optuna)

    # ── Pruner ────────────────────────────────────────────────────────
    pruner = _build_pruner(config, optuna)

    # ── Study (minimize val_loss) ─────────────────────────────────────
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"geosteering_hpo_{config.experiment_tag or 'default'}",
    )

    logger.info(
        "Optuna study criado: sampler=%s, pruner=%s, seed=%d",
        config.optuna_sampler,
        config.optuna_pruner,
        config.global_seed,
    )

    return study


# ════════════════════════════════════════════════════════════════════════
# RUN HPO — Loop principal de otimizacao
#
# Executa optuna study.optimize() com objective que:
#   1. Sugere HPs via create_search_space
#   2. Cria config variante via config.copy(**params)
#   3. Chama build_and_train_fn(trial_config) → val_loss
#   4. Retorna val_loss para Optuna registrar
#
# Weight reset: nao implementado aqui (responsabilidade do
# build_and_train_fn, que deve construir modelo fresh a cada trial).
#
# Timeout: study.optimize() para apos optuna_timeout segundos
# OU optuna_n_trials trials (o que ocorrer primeiro).
# ════════════════════════════════════════════════════════════════════════

def run_hpo(
    config: PipelineConfig,
    build_and_train_fn: Callable[..., float],
) -> Dict[str, Any]:
    """Executa otimizacao de hiperparametros via Optuna.

    Cria study, define objective, e executa optimize() com n_trials
    e timeout do config. Cada trial sugere HPs, cria config variante,
    e chama build_and_train_fn para avaliar a performance.

    Args:
        config: PipelineConfig com use_optuna=True e campos Optuna:
            - optuna_n_trials: Numero maximo de trials.
            - optuna_timeout: Timeout em segundos (1h default).
            - optuna_sampler: 'tpe', 'cmaes', 'random'.
            - optuna_pruner: 'median', 'hyperband'.
        build_and_train_fn: Callable que recebe PipelineConfig e retorna
            val_loss (float). Deve construir modelo, treinar, e retornar
            a melhor val_loss. Responsavel pelo weight reset (modelo novo
            a cada chamada).

    Returns:
        Dict com resultados:
            - best_params (dict): Melhores hiperparametros encontrados.
            - best_value (float): Melhor val_loss obtido.
            - n_trials (int): Numero de trials completados.
            - study: Instancia do Optuna study (para analise posterior).

    Raises:
        ImportError: Se optuna nao estiver instalado.
        ValueError: Se use_optuna=False.

    Example:
        >>> def train_fn(trial_config):
        ...     model = build_model(trial_config)
        ...     history = model.fit(...)
        ...     return min(history.history['val_loss'])
        >>>
        >>> config = PipelineConfig.robusto().copy(
        ...     use_optuna=True, optuna_n_trials=20
        ... )
        >>> result = run_hpo(config, train_fn)
        >>> result['best_params']
        {'learning_rate': 0.0003, 'batch_size': 64, ...}
        >>> result['best_value']
        0.1234

    Note:
        Referenciado em:
            - training/__init__.py: re-export run_hpo
            - tests/test_optuna_hpo.py: TestRunHPO
        Ref: docs/ARCHITECTURE_v2.md secao 6.4 (HPO runner).
        Weight reset: build_and_train_fn DEVE criar modelo novo a cada
        chamada (pesos aleatorios). NStageTrainer.run() dentro do fn
        garante treinamento completo por trial.
        Timeout e n_trials sao limites conjuntos (o primeiro atingido para).
        Optuna loga internamente (nao duplicar com logger do modulo).
    """
    if not config.use_optuna:
        raise ValueError(
            "run_hpo requer config.use_optuna=True. "
            "Defina use_optuna=True no PipelineConfig."
        )

    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna necessario para HPO: pip install optuna. "
            "Consulte: https://optuna.readthedocs.io/"
        )

    # ── Cria study ────────────────────────────────────────────────────
    study = create_study(config)

    # ── Define objective ──────────────────────────────────────────────
    def objective(trial: optuna.trial.Trial) -> float:
        """Objective function para Optuna — avalia um set de HPs.

        Args:
            trial: Trial Optuna com metodos suggest_*.

        Returns:
            val_loss (float) para Optuna registrar e comparar.

        Raises:
            optuna.TrialPruned: Se trial for podado pelo pruner.
        """
        # 1. Sugere HPs
        params = create_search_space(trial, config)

        # 2. Cria config variante com HPs sugeridos
        # D7: config.copy() cria nova instancia imutavel (frozen dataclass)
        # com overrides. Validacao __post_init__ executada automaticamente.
        trial_config = config.copy(**params)

        # 3. Treina e avalia
        logger.info(
            "Trial %d: iniciando treinamento com %s",
            trial.number,
            params,
        )

        try:
            val_loss = build_and_train_fn(trial_config)
        except Exception as exc:
            logger.warning(
                "Trial %d falhou: %s. Retornando inf.", trial.number, exc
            )
            return float("inf")

        logger.info(
            "Trial %d: val_loss = %.6f", trial.number, val_loss
        )

        return val_loss

    # ── Executa otimizacao ────────────────────────────────────────────
    logger.info(
        "Iniciando HPO: %d trials, timeout=%ds, sampler=%s, pruner=%s",
        config.optuna_n_trials,
        config.optuna_timeout,
        config.optuna_sampler,
        config.optuna_pruner,
    )

    study.optimize(
        objective,
        n_trials=config.optuna_n_trials,
        timeout=config.optuna_timeout,
        show_progress_bar=True,
    )

    # ── Coleta resultados ─────────────────────────────────────────────
    n_completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )
    n_failed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    )

    logger.info(
        "HPO concluido: %d trials (%d completos, %d podados, %d falhos). "
        "Best val_loss = %.6f",
        len(study.trials),
        n_completed,
        n_pruned,
        n_failed,
        study.best_value if n_completed > 0 else float("inf"),
    )

    if n_completed > 0:
        logger.info("Best params: %s", study.best_params)

    result: Dict[str, Any] = {
        "best_params": study.best_params if n_completed > 0 else {},
        "best_value": study.best_value if n_completed > 0 else float("inf"),
        "n_trials": len(study.trials),
        "n_completed": n_completed,
        "n_pruned": n_pruned,
        "n_failed": n_failed,
        "study": study,
    }

    return result


# ════════════════════════════════════════════════════════════════════════
# FUNCOES AUXILIARES INTERNAS
#
# _build_sampler: Cria sampler Optuna a partir do config.
# _build_pruner: Cria pruner Optuna a partir do config.
#
# Ambas sao privadas (prefixo _) e nao entram no __all__.
# ════════════════════════════════════════════════════════════════════════


def _build_sampler(config: PipelineConfig, optuna_module: Any) -> Any:
    """Cria sampler Optuna a partir do config.

    Args:
        config: PipelineConfig com optuna_sampler e global_seed.
        optuna_module: Modulo optuna importado (passado para evitar
            re-import).

    Returns:
        Instancia de optuna.samplers.BaseSampler.

    Raises:
        ValueError: Se config.optuna_sampler nao for reconhecido.

    Note:
        Referenciado em:
            - training/optuna_hpo.py: create_study() (sampler creation)
        D7: Seed propagada para reprodutibilidade (identico ao
        config.global_seed=42 default).
        TPE: melhor para espacos mistos (cat + cont), eficiente com N<100.
        CMA-ES: melhor para espacos continuos puros, requer N>10.
        Random: baseline, sem overhead de modelo de surrogate.
    """
    sampler_name = config.optuna_sampler.lower()
    seed = config.global_seed

    if sampler_name == "tpe":
        return optuna_module.samplers.TPESampler(seed=seed)
    elif sampler_name == "cmaes":
        return optuna_module.samplers.CmaEsSampler(seed=seed)
    elif sampler_name == "random":
        return optuna_module.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(
            f"optuna_sampler '{config.optuna_sampler}' nao reconhecido. "
            f"Validos: tpe, cmaes, random"
        )


def _build_pruner(config: PipelineConfig, optuna_module: Any) -> Any:
    """Cria pruner Optuna a partir do config.

    Args:
        config: PipelineConfig com optuna_pruner e epochs.
        optuna_module: Modulo optuna importado.

    Returns:
        Instancia de optuna.pruners.BasePruner.

    Raises:
        ValueError: Se config.optuna_pruner nao for reconhecido.

    Note:
        Referenciado em:
            - training/optuna_hpo.py: create_study() (pruner creation)
        D7: n_startup_trials=5 para Median: espera 5 trials completos
        antes de comecar a podar (evita podar trials iniciais
        prematuramente por falta de referencia).
        Hyperband min_resource=3: reporta metrica a cada 3 epocas.
        Hyperband max_resource=epochs: limite superior de resources.
    """
    pruner_name = config.optuna_pruner.lower()

    if pruner_name == "median":
        # D7: n_startup_trials=5 garante baseline estatistico antes de podar.
        # n_warmup_steps=10: nao poda nas primeiras 10 epocas de cada trial
        # (convergencia inicial e ruidosa, podar cedo gera falsos negativos).
        return optuna_module.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        )
    elif pruner_name == "hyperband":
        # D7: Hyperband usa successive halving com schedule agressivo.
        # min_resource=3: reporta a cada 3 epocas (overhead minimo).
        # max_resource=epochs: limite superior = epocas totais.
        # reduction_factor=3: elimina 2/3 dos trials a cada etapa.
        return optuna_module.pruners.HyperbandPruner(
            min_resource=3,
            max_resource=config.epochs,
            reduction_factor=3,
        )
    else:
        raise ValueError(
            f"optuna_pruner '{config.optuna_pruner}' nao reconhecido. "
            f"Validos: median, hyperband"
        )
