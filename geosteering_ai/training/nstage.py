# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: training/nstage.py                                               ║
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
# ║    • NStageTrainer: treinamento em estagios progressivos de ruido (S21+)  ║
# ║    • Stage 1: convergencia clean (nstage_stage1_epochs, noise = 0)        ║
# ║    • Stages 2..N: noise_k, lr_k, patience_k auto-calculados              ║
# ║    • Mini-curriculum opcional dentro de cada stage (stage_ramp_fraction)   ║
# ║    • TrainingResult: container para resultado unificado multi-stage       ║
# ║    • Merge de histories Keras em historico unico                          ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), TensorFlow 2.x / Keras        ║
# ║  Exports: ~2 (TrainingResult, NStageTrainer) — ver __all__               ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6.2 (N-Stage Training)               ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C43 PARTE 7B)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""NStageTrainer — Treinamento em estagios progressivos de ruido.

Separa convergencia clean (Stage 1) de adaptacao progressiva a ruido
(Stages 2..N). Cada estagio tem noise_level, learning_rate e patience
auto-calculados a partir dos parametros do config:

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  N-STAGE TRAINING (S21+ Unificado)                                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Stage 1 (Clean Convergence):                                       │
    │    noise = 0.0                                                       │
    │    lr    = config.learning_rate (1e-4)                              │
    │    epochs = nstage_stage1_epochs (15)                               │
    │    patience = early_stopping_patience (60)                          │
    │                                                                      │
    │  Stage k (k = 2..N):                                                │
    │    noise_k    = noise_max × (k-1)/(N-1)                            │
    │    lr_k       = learning_rate × stage_lr_decay^(k-1)               │
    │    patience_k = nstage_base_patience + (k-2) × 5                   │
    │    epochs_k   = epochs - nstage_stage1_epochs  (dividido entre N-1)│
    │                                                                      │
    │  Mini-curriculum (opcional, por stage):                              │
    │    Primeiros stage_ramp_fraction × epochs_k com noise crescente    │
    │    Restante com noise constante (noise_k)                           │
    │                                                                      │
    │  Exemplo N=3, noise_max=0.08, LR=1e-4, decay=0.5:                 │
    │    Stage 1: noise=0.00, LR=1e-4                                     │
    │    Stage 2: noise=0.04, LR=5e-5                                     │
    │    Stage 3: noise=0.08, LR=2.5e-5                                   │
    └──────────────────────────────────────────────────────────────────────┘

Example:
    >>> from geosteering_ai.config import PipelineConfig
    >>> config = PipelineConfig.nstage(n=3)
    >>> trainer = NStageTrainer(config)
    >>> params = trainer.compute_stage_params(2)
    >>> params['noise_level']
    0.04
    >>> params['learning_rate']
    5e-05

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Referenciado em:
        - training/__init__.py: re-export NStageTrainer, TrainingResult
        - config.py: secao 10 (N-Stage fields), preset nstage()
        - tests/test_nstage.py: TestNStageTrainer (compute, run, merge)
    Ref: docs/ARCHITECTURE_v2.md secao 6.2 (N-Stage Training).
    Mutuamente exclusivo com curriculum (validado em PipelineConfig.__post_init__).
    S21+: TWO_STAGE generalizado para N stages com auto-calculo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Container de resultado ---
    "TrainingResult",
    # --- Treinador N-Stage ---
    "NStageTrainer",
]


# ════════════════════════════════════════════════════════════════════════
# TRAINING RESULT — Container para historico multi-stage
#
# Unifica os resultados de model.fit() de multiplos estagios em um
# unico objeto com historico mesclado e metadados por estagio.
#
# Conteudo:
#   - merged_history: dict com listas por metrica (loss, val_loss, etc.)
#   - stage_histories: lista de histories individuais (1 por stage)
#   - stage_params: lista de dicts com parametros de cada stage
#   - total_epochs: total de epocas executadas (soma de todos os stages)
#   - best_val_loss: melhor val_loss global entre todos os stages
# ════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingResult:
    """Resultado unificado de treinamento multi-stage.

    Encapsula historicos Keras de todos os stages e metadados de cada
    estagio (noise, LR, patience, epocas executadas).

    Attributes:
        merged_history: Historico mesclado com todas as metricas concatenadas.
            Chaves: 'loss', 'val_loss', etc. Valores: listas de floats.
        stage_histories: Historicos individuais de cada stage (Keras dicts).
        stage_params: Parametros computados para cada stage (noise, LR, etc.).
        total_epochs: Total de epocas executadas em todos os stages.
        best_val_loss: Melhor val_loss encontrada em qualquer stage.

    Example:
        >>> result = TrainingResult(
        ...     merged_history={'loss': [0.5, 0.3, 0.2], 'val_loss': [0.6, 0.4, 0.3]},
        ...     stage_histories=[{'loss': [0.5]}, {'loss': [0.3, 0.2]}],
        ...     stage_params=[{'noise_level': 0.0}, {'noise_level': 0.08}],
        ...     total_epochs=3,
        ...     best_val_loss=0.3,
        ... )

    Note:
        Referenciado em:
            - training/nstage.py: NStageTrainer.run() (retorno principal)
            - training/__init__.py: re-export TrainingResult
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        merged_history e compativel com Keras history.history format.
    """
    merged_history: Dict[str, List[float]] = field(default_factory=dict)
    stage_histories: List[Dict[str, List[float]]] = field(default_factory=list)
    stage_params: List[Dict[str, Any]] = field(default_factory=list)
    total_epochs: int = 0
    best_val_loss: float = float("inf")


# ════════════════════════════════════════════════════════════════════════
# N-STAGE TRAINER — Treinamento progressivo multi-estagio
#
# Implementa o loop de treinamento N-Stage (S21+ Unificado):
#
# ┌────────────────────────────────────────────────────────────────────┐
# │  FLUXO DO NSTAGETRAINER                                           │
# ├────────────────────────────────────────────────────────────────────┤
# │                                                                    │
# │  __init__(config) → valida flags, computa distribuicao de epocas  │
# │       │                                                            │
# │  compute_stage_params(k) → {noise_k, lr_k, patience_k, epochs_k} │
# │       │                                                            │
# │  run(model, loss_fn, ...) → loop stages 1..N                     │
# │       │                                                            │
# │       ├─ Stage 1: noise=0, lr=base, fit(stage1_epochs)           │
# │       │                                                            │
# │       ├─ Stage k=2..N:                                            │
# │       │    1. noise_level_var.assign(noise_k)                     │
# │       │    2. Recompile model com lr_k                            │
# │       │    3. build_callbacks_fn(patience_k)                      │
# │       │    4. model.fit(epochs_k)                                 │
# │       │    5. Merge history                                       │
# │       │                                                            │
# │       └─ return TrainingResult (merged)                           │
# └────────────────────────────────────────────────────────────────────┘
#
# Auto-calculo por stage (formulas S21+):
#   noise_k    = noise_max × (k-1) / (N-1)        — linear 0..max
#   lr_k       = learning_rate × decay^(k-1)       — exponential decay
#   patience_k = base_patience + (k-2) × 5         — +5 por stage extra
#   epochs_k   = remaining_epochs / (N-1)           — distribuicao uniforme
#
# Mini-curriculum (stage_ramp_fraction):
#   Dentro de cada stage k>=2, os primeiros ramp_fraction × epochs_k
#   epocas tem noise crescente de 0 ate noise_k. Isso suaviza a
#   transicao entre stages (evita salto abrupto de noise).
# ════════════════════════════════════════════════════════════════════════

class NStageTrainer:
    """Treinador N-Stage: ruido progressivo entre estagios.

    Stage 1 converge em dados limpos (noise = 0). Stages subsequentes
    aumentam progressivamente o noise, diminuem o LR e ajustam patience,
    permitindo ao modelo se adaptar gradualmente a niveis crescentes de
    ruido sem perder a convergencia obtida em dados limpos.

    Attributes:
        config: PipelineConfig com flags N-Stage (secao 10).
        n_stages: Numero de estagios (config.n_training_stages).
        _remaining_epochs: Epocas disponiveis para stages 2..N.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig.nstage(n=3)
        >>> trainer = NStageTrainer(config)
        >>> # Em producao: result = trainer.run(model, loss_fn, ...)

    Note:
        Referenciado em:
            - training/__init__.py: re-export NStageTrainer
            - config.py: preset nstage() cria config com use_nstage=True
            - tests/test_nstage.py: TestNStageTrainer
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (N-Stage Training).
        Mutuamente exclusivo com curriculum noise (use_curriculum).
        PipelineConfig.__post_init__ valida a exclusividade.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Inicializa NStageTrainer com configuracao validada.

        Args:
            config: PipelineConfig com use_nstage=True e campos N-Stage
                (n_training_stages, nstage_stage1_epochs, stage_lr_decay,
                nstage_base_patience, use_stage_mini_curriculum,
                stage_ramp_fraction, noise_level_max, learning_rate, epochs).

        Raises:
            ValueError: Se use_nstage=False ou n_training_stages < 2.

        Note:
            Referenciado em:
                - training/nstage.py: NStageTrainer (constructor)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Validacao: use_nstage DEVE ser True, n_stages >= 2.
            _remaining_epochs = epochs - nstage_stage1_epochs.
        """
        if not config.use_nstage:
            raise ValueError(
                "NStageTrainer requer config.use_nstage=True. "
                "Use PipelineConfig.nstage() ou defina use_nstage=True."
            )
        if config.n_training_stages < 2:
            raise ValueError(
                f"n_training_stages deve ser >= 2, recebido: {config.n_training_stages}"
            )

        self.config = config
        self.n_stages: int = config.n_training_stages

        # Epocas restantes para stages 2..N (divididas uniformemente)
        self._remaining_epochs: int = max(
            config.epochs - config.nstage_stage1_epochs, 0
        )

        logger.info(
            "NStageTrainer inicializado: %d stages, Stage 1 = %d ep (clean), "
            "Stages 2..%d = %d ep total, noise_max = %.4f, lr_decay = %.2f",
            self.n_stages,
            config.nstage_stage1_epochs,
            self.n_stages,
            self._remaining_epochs,
            config.noise_level_max,
            config.stage_lr_decay,
        )

    # ════════════════════════════════════════════════════════════════════
    # COMPUTE STAGE PARAMS — Auto-calculo de noise, LR, patience, epochs
    #
    # Formulas S21+ (validadas experimentalmente):
    #
    #   Stage 1 (clean):
    #     noise   = 0.0
    #     lr      = config.learning_rate
    #     patience = config.early_stopping_patience
    #     epochs  = config.nstage_stage1_epochs
    #
    #   Stage k (k >= 2):
    #     noise_k    = noise_max × (k-1) / (N-1)
    #     lr_k       = learning_rate × decay^(k-1)
    #     patience_k = base_patience + (k-2) × 5
    #     epochs_k   = remaining_epochs / (N-1)
    #
    # Exemplo N=3, noise_max=0.08, LR=1e-4, decay=0.5:
    #   Stage 1: noise=0.00, LR=1.0e-4, patience=60
    #   Stage 2: noise=0.04, LR=5.0e-5, patience=30
    #   Stage 3: noise=0.08, LR=2.5e-5, patience=35
    # ════════════════════════════════════════════════════════════════════

    def compute_stage_params(self, stage_idx: int) -> Dict[str, Any]:
        """Computa noise_level, learning_rate, patience e epochs para stage k.

        Args:
            stage_idx: Indice do stage (1-based). Stage 1 = clean convergence.

        Returns:
            Dict com chaves:
                - noise_level (float): Nivel de ruido para o stage.
                - learning_rate (float): Taxa de aprendizado para o stage.
                - patience (int): Early stopping patience para o stage.
                - epochs (int): Numero de epocas para o stage.
                - use_mini_curriculum (bool): Se mini-curriculum esta ativo.
                - ramp_epochs (int): Epocas de rampa dentro do stage (0 se off).

        Raises:
            ValueError: Se stage_idx < 1 ou > n_stages.

        Example:
            >>> config = PipelineConfig.nstage(n=3)
            >>> trainer = NStageTrainer(config)
            >>> p1 = trainer.compute_stage_params(1)
            >>> p1['noise_level']
            0.0
            >>> p2 = trainer.compute_stage_params(2)
            >>> p2['noise_level']
            0.04

        Note:
            Referenciado em:
                - training/nstage.py: NStageTrainer.run() (loop principal)
                - tests/test_nstage.py: TestComputeStageParams
            Ref: docs/ARCHITECTURE_v2.md secao 6.2 (formulas S21+).
            Stage 1 sempre clean (noise=0, LR=base, patience=ES patience).
            Stages 2..N: noise linear, LR exponencial, patience crescente.
        """
        if stage_idx < 1 or stage_idx > self.n_stages:
            raise ValueError(
                f"stage_idx deve estar em [1, {self.n_stages}], "
                f"recebido: {stage_idx}"
            )

        cfg = self.config

        # ── Stage 1: convergencia clean (sem ruido) ───────────────────
        if stage_idx == 1:
            return {
                "noise_level": 0.0,
                "learning_rate": cfg.learning_rate,
                "patience": cfg.early_stopping_patience,
                "epochs": cfg.nstage_stage1_epochs,
                "use_mini_curriculum": False,
                "ramp_epochs": 0,
            }

        # ── Stages 2..N: ruido progressivo ────────────────────────────
        k = stage_idx
        n = self.n_stages

        # noise_k = noise_max × (k-1) / (N-1) — linear de 0 a noise_max
        noise_level = cfg.noise_level_max * (k - 1) / (n - 1)

        # lr_k = learning_rate × decay^(k-1) — decaimento exponencial
        learning_rate = cfg.learning_rate * (cfg.stage_lr_decay ** (k - 1))

        # patience_k = base_patience + (k-2) × 5 — incremento por stage
        patience = cfg.nstage_base_patience + (k - 2) * 5

        # epochs_k = remaining_epochs / (N-1) — distribuicao uniforme
        epochs_per_stage = self._remaining_epochs // (n - 1)

        # Mini-curriculum: rampa curta dentro do stage
        use_mini = cfg.use_stage_mini_curriculum
        ramp_epochs = 0
        if use_mini:
            # stage_ramp_fraction × epochs_k epocas de rampa (noise 0→noise_k)
            ramp_epochs = int(cfg.stage_ramp_fraction * epochs_per_stage)

        return {
            "noise_level": noise_level,
            "learning_rate": learning_rate,
            "patience": patience,
            "epochs": epochs_per_stage,
            "use_mini_curriculum": use_mini,
            "ramp_epochs": ramp_epochs,
        }

    # ════════════════════════════════════════════════════════════════════
    # RUN — Loop principal de treinamento multi-stage
    #
    # Executa o treinamento completo em N stages:
    #
    # ┌───────────────────────────────────────────────────────────────┐
    # │  for stage_idx in range(1, N+1):                             │
    # │    1. params = compute_stage_params(stage_idx)               │
    # │    2. noise_level_var.assign(params['noise_level'])          │
    # │    3. Recompile model com params['learning_rate']            │
    # │    4. callbacks = build_callbacks_fn(params['patience'])     │
    # │    5. history = model.fit(..., epochs=params['epochs'])      │
    # │    6. Merge history → result                                 │
    # │  return TrainingResult                                       │
    # └───────────────────────────────────────────────────────────────┘
    #
    # Observacoes:
    #   - model.fit() usa initial_epoch para continuar contagem global
    #   - noise_level_var e tf.Variable compartilhada com train_map_fn
    #   - Recompile necessario para atualizar LR (Keras nao permite
    #     alterar LR sem recompilar em todas as versoes)
    #   - build_callbacks_fn recria callbacks com patience atualizado
    # ════════════════════════════════════════════════════════════════════

    def run(
        self,
        model: Any,
        loss_fn: Any,
        metrics_list: List[Any],
        train_ds: Any,
        val_ds: Any,
        build_callbacks_fn: Callable[[int], List[Any]],
        noise_level_var: Any,
    ) -> TrainingResult:
        """Executa treinamento N-Stage completo.

        Para cada stage: ajusta noise, recompila modelo com novo LR,
        constroi callbacks com patience atualizado, e executa model.fit().
        Retorna TrainingResult com historicos mesclados.

        Args:
            model: Modelo Keras compilado (tf.keras.Model).
            loss_fn: Funcao de perda (Keras loss ou callable).
            metrics_list: Lista de metricas Keras para compilacao.
            train_ds: tf.data.Dataset de treino (com train_map_fn aplicado).
            val_ds: tf.data.Dataset de validacao.
            build_callbacks_fn: Callable que recebe patience (int) e retorna
                lista de callbacks Keras para o stage. Deve criar callbacks
                frescos a cada chamada (evita estado residual).
            noise_level_var: tf.Variable(float32) compartilhada com
                train_map_fn para controle dinamico do nivel de ruido.

        Returns:
            TrainingResult com historico mesclado de todos os stages.

        Raises:
            RuntimeError: Se model.fit() falhar em qualquer stage.

        Example:
            >>> # Uso tipico (simplificado — em producao via TrainingLoop):
            >>> config = PipelineConfig.nstage(n=2)
            >>> trainer = NStageTrainer(config)
            >>> result = trainer.run(
            ...     model=model, loss_fn='mse', metrics_list=['mae'],
            ...     train_ds=train_ds, val_ds=val_ds,
            ...     build_callbacks_fn=lambda p: [EarlyStopping(patience=p)],
            ...     noise_level_var=noise_var,
            ... )

        Note:
            Referenciado em:
                - training/__init__.py: orquestrado por TrainingLoop
                - tests/test_nstage.py: TestNStageRun
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            TensorFlow importado lazy (dentro do metodo) para compatibilidade
            com ambientes sem GPU e para testes unitarios com mock.
            noise_level_var.assign() atualiza noise para train_map_fn via
            tf.data.map closure (leitura em tempo de execucao do grafo).
        """
        import tensorflow as tf  # D10: lazy import TF

        cfg = self.config
        result = TrainingResult()
        global_epoch = 0  # Contador global de epocas (para initial_epoch)

        logger.info(
            "Iniciando N-Stage Training: %d stages, %d epocas totais",
            self.n_stages,
            cfg.epochs,
        )

        for stage_idx in range(1, self.n_stages + 1):
            # ── 1. Computa parametros do stage ────────────────────────
            params = self.compute_stage_params(stage_idx)
            result.stage_params.append(params)

            stage_noise = params["noise_level"]
            stage_lr = params["learning_rate"]
            stage_patience = params["patience"]
            stage_epochs = params["epochs"]

            logger.info(
                "Stage %d/%d: noise=%.4f, lr=%.2e, patience=%d, epochs=%d",
                stage_idx,
                self.n_stages,
                stage_noise,
                stage_lr,
                stage_patience,
                stage_epochs,
            )

            # ── 2. Atualiza noise_level_var (lido pelo train_map_fn) ─
            # D7: assign() atualiza tf.Variable no grafo; train_map_fn
            # le esse valor a cada batch via closure sobre a Variable.
            noise_level_var.assign(stage_noise)
            logger.info(
                "Stage %d: noise_level_var atualizada para %.4f",
                stage_idx,
                stage_noise,
            )

            # ── 3. Recompile modelo com novo LR ──────────────────────
            # D7: Keras requer recompilacao para alterar optimizer/LR.
            # Pesos do modelo sao PRESERVADOS (compile nao reseta pesos).
            optimizer = _build_optimizer(cfg, stage_lr)
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics_list,
            )
            logger.info(
                "Stage %d: modelo recompilado com LR=%.2e (optimizer=%s)",
                stage_idx,
                stage_lr,
                cfg.optimizer,
            )

            # ── 4. Constroi callbacks frescos para o stage ────────────
            # D7: callbacks devem ser recriados (EarlyStopping, etc.
            # mantem estado interno que nao deve persistir entre stages).
            callbacks = build_callbacks_fn(stage_patience)

            # ── 5. Mini-curriculum: rampa de noise dentro do stage ────
            if params["use_mini_curriculum"] and params["ramp_epochs"] > 0:
                mini_cb = _MiniCurriculumCallback(
                    noise_level_var=noise_level_var,
                    target_noise=stage_noise,
                    ramp_epochs=params["ramp_epochs"],
                )
                callbacks.append(mini_cb)
                logger.info(
                    "Stage %d: mini-curriculum ativo (rampa %d ep, "
                    "0.0 → %.4f)",
                    stage_idx,
                    params["ramp_epochs"],
                    stage_noise,
                )

            # ── 6. model.fit() para o stage ───────────────────────────
            # initial_epoch: continua contagem global (Keras usa para
            # logs, TensorBoard e LR schedules relativos).
            try:
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=global_epoch + stage_epochs,
                    initial_epoch=global_epoch,
                    callbacks=callbacks,
                    verbose=1 if cfg.verbose else 0,
                )
            except Exception as exc:
                logger.error(
                    "Stage %d: model.fit() falhou: %s", stage_idx, exc
                )
                raise RuntimeError(
                    f"N-Stage Training falhou no Stage {stage_idx}: {exc}"
                ) from exc

            # ── 7. Registra resultado do stage ────────────────────────
            stage_hist = history.history
            result.stage_histories.append(stage_hist)

            # Numero real de epocas executadas (pode ser < stage_epochs
            # se EarlyStopping parou antecipadamente)
            actual_epochs = len(stage_hist.get("loss", []))
            global_epoch += actual_epochs

            logger.info(
                "Stage %d: concluido em %d/%d epocas",
                stage_idx,
                actual_epochs,
                stage_epochs,
            )

            # ── 8. Atualiza best_val_loss ─────────────────────────────
            val_losses = stage_hist.get("val_loss", [])
            if val_losses:
                stage_best = min(val_losses)
                if stage_best < result.best_val_loss:
                    result.best_val_loss = stage_best
                    logger.info(
                        "Stage %d: novo best val_loss = %.6f",
                        stage_idx,
                        stage_best,
                    )

            # ── 9. Restaura noise ao nivel do stage (pos mini-curriculum) ─
            # D7: Mini-curriculum pode ter alterado noise_level_var;
            # restauramos ao valor nominal do stage para consistencia.
            noise_level_var.assign(stage_noise)

        # ── Merge de historicos ────────────────────────────────────────
        result.merged_history = _merge_histories(result.stage_histories)
        result.total_epochs = global_epoch

        logger.info(
            "N-Stage Training concluido: %d stages, %d epocas totais, "
            "best val_loss = %.6f",
            self.n_stages,
            result.total_epochs,
            result.best_val_loss,
        )

        return result


# ════════════════════════════════════════════════════════════════════════
# FUNCOES AUXILIARES INTERNAS
#
# _build_optimizer: Cria optimizer Keras a partir do config e LR override.
# _merge_histories: Concatena dicts de history de multiplos stages.
# _MiniCurriculumCallback: Callback para rampa de noise dentro de um stage.
#
# Todas sao privadas (prefixo _) e nao entram no __all__.
# ════════════════════════════════════════════════════════════════════════


def _build_optimizer(config: PipelineConfig, learning_rate: float) -> Any:
    """Cria optimizer Keras a partir do config com LR override.

    Args:
        config: PipelineConfig com campo optimizer (nome do optimizer).
        learning_rate: Taxa de aprendizado para este stage (override).

    Returns:
        Instancia de tf.keras.optimizers.Optimizer.

    Raises:
        ValueError: Se config.optimizer nao for reconhecido.

    Note:
        Referenciado em:
            - training/nstage.py: NStageTrainer.run() (recompile por stage)
        Ref: docs/ARCHITECTURE_v2.md secao 6.1 (optimizer factory).
        Lazy import TF (dentro da funcao) para compatibilidade CPU-only.
        weight_decay aplicado apenas para AdamW (Keras 3.x nativo).
    """
    import tensorflow as tf  # D10: lazy import TF

    _OPTIMIZERS = {
        "adam": lambda: tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "adamw": lambda: tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=config.weight_decay,
        ),
        "sgd": lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate),
        "rmsprop": lambda: tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        ),
        "nadam": lambda: tf.keras.optimizers.Nadam(
            learning_rate=learning_rate
        ),
        "adagrad": lambda: tf.keras.optimizers.Adagrad(
            learning_rate=learning_rate
        ),
    }

    opt_name = config.optimizer.lower()
    if opt_name not in _OPTIMIZERS:
        raise ValueError(
            f"Optimizer '{config.optimizer}' nao reconhecido. "
            f"Validos: {list(_OPTIMIZERS.keys())}"
        )

    optimizer = _OPTIMIZERS[opt_name]()

    # D7: gradient clipping aplicado diretamente no optimizer Keras 3.x
    # (clipnorm e parametro do optimizer, nao callback)
    if config.use_gradient_clipping:
        optimizer.clipnorm = config.gradient_clip_norm

    logger.info(
        "Optimizer criado: %s, lr=%.2e, clipnorm=%s",
        opt_name,
        learning_rate,
        config.gradient_clip_norm if config.use_gradient_clipping else "off",
    )

    return optimizer


def _merge_histories(
    stage_histories: List[Dict[str, List[float]]],
) -> Dict[str, List[float]]:
    """Concatena historicos Keras de multiplos stages.

    Args:
        stage_histories: Lista de dicts (Keras history.history) por stage.

    Returns:
        Dict unico com listas concatenadas por metrica.

    Example:
        >>> h1 = {'loss': [0.5, 0.4], 'val_loss': [0.6, 0.5]}
        >>> h2 = {'loss': [0.3, 0.2], 'val_loss': [0.4, 0.3]}
        >>> merged = _merge_histories([h1, h2])
        >>> merged['loss']
        [0.5, 0.4, 0.3, 0.2]

    Note:
        Referenciado em:
            - training/nstage.py: NStageTrainer.run() (pos-loop)
        Metricas que existem em apenas alguns stages sao preenchidas
        com [] para stages onde nao existem (sem NaN padding).
    """
    if not stage_histories:
        return {}

    # Coleta todas as chaves de metricas presentes em qualquer stage
    all_keys: set = set()
    for hist in stage_histories:
        all_keys.update(hist.keys())

    merged: Dict[str, List[float]] = {key: [] for key in sorted(all_keys)}

    for hist in stage_histories:
        for key in merged:
            merged[key].extend(hist.get(key, []))

    return merged


class _MiniCurriculumCallback:
    """Callback para rampa de noise dentro de um stage (mini-curriculum).

    Dentro de um stage k>=2, os primeiros ramp_epochs epocas tem noise
    crescente linearmente de 0.0 ate target_noise. Apos a rampa,
    noise permanece constante em target_noise.

    Implementa a interface de callback Keras (on_epoch_begin).

    Attributes:
        noise_level_var: tf.Variable compartilhada com train_map_fn.
        target_noise: Nivel de noise alvo para o stage.
        ramp_epochs: Numero de epocas de rampa.
        _stage_start_epoch: Epoca global no inicio do stage (capturada
            no primeiro on_epoch_begin).

    Note:
        Referenciado em:
            - training/nstage.py: NStageTrainer.run() (append a callbacks)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (mini-curriculum).
        Classe privada (prefixo _) — nao exportada no __all__.
        Herda de tf.keras.callbacks.Callback via lazy import.
    """

    def __init__(
        self,
        noise_level_var: Any,
        target_noise: float,
        ramp_epochs: int,
    ) -> None:
        """Inicializa callback de mini-curriculum.

        Args:
            noise_level_var: tf.Variable(float32) para controle de noise.
            target_noise: Nivel de noise alvo (final da rampa).
            ramp_epochs: Numero de epocas para a rampa (0 = sem rampa).

        Note:
            Referenciado em:
                - training/nstage.py: NStageTrainer.run() (instanciacao)
        """
        import tensorflow as tf  # D10: lazy import TF

        # Herda de Callback para compatibilidade com Keras
        super().__init__()

        self.noise_level_var = noise_level_var
        self.target_noise = target_noise
        self.ramp_epochs = ramp_epochs
        self._stage_start_epoch: Optional[int] = None

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Ajusta noise no inicio de cada epoca (rampa linear).

        Args:
            epoch: Indice da epoca (global, 0-based — Keras convencao).
            logs: Dict de logs Keras (nao utilizado).

        Note:
            Referenciado em:
                - training/nstage.py: Keras chama automaticamente
            D7: Epoca relativa ao stage = epoch - _stage_start_epoch.
            Noise cresce linearmente: noise_k × (ep_rel / ramp_epochs).
            Apos ramp_epochs: noise constante em target_noise.
        """
        # Captura epoca de inicio do stage (primeira chamada)
        if self._stage_start_epoch is None:
            self._stage_start_epoch = epoch

        # Epoca relativa ao stage (0-based)
        ep_rel = epoch - self._stage_start_epoch

        if ep_rel < self.ramp_epochs:
            # Rampa linear: 0 → target_noise em ramp_epochs
            # D7: eps=1e-12 previne divisao por zero (embora ramp_epochs >= 1)
            frac = ep_rel / max(self.ramp_epochs, 1)
            current_noise = self.target_noise * frac
        else:
            # Pos-rampa: noise constante
            current_noise = self.target_noise

        self.noise_level_var.assign(current_noise)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Hook de fim de epoca (noop — interface Keras requer).

        Args:
            epoch: Indice da epoca.
            logs: Dict de logs Keras.
        """


# Registra _MiniCurriculumCallback como subclasse de tf.keras.callbacks.Callback
# via monkey-patch na primeira importacao de TF (lazy compatibility).
# Isso evita importar TF no escopo do modulo.
def _register_callback_base() -> type:
    """Retorna _MiniCurriculumCallback com base Callback do Keras.

    Returns:
        Classe _MiniCurriculumCallback que herda de tf.keras.callbacks.Callback.

    Note:
        Referenciado em:
            - training/nstage.py: _MiniCurriculumCallback (metaclasse)
        D10: Lazy import para evitar importar TF no escopo do modulo.
        A classe interna herda de Callback para que Keras a reconheca
        no loop de treinamento.
    """
    import tensorflow as tf  # D10: lazy import TF

    class _MiniCurriculumCallbackKeras(tf.keras.callbacks.Callback):
        """Callback Keras para mini-curriculum dentro de um stage.

        Herda de tf.keras.callbacks.Callback para compatibilidade total
        com o loop de treinamento Keras (model.fit).

        See Also:
            _MiniCurriculumCallback (versao sem heranca, docs completos).
        """

        def __init__(
            self,
            noise_level_var: Any,
            target_noise: float,
            ramp_epochs: int,
        ) -> None:
            super().__init__()
            self.noise_level_var = noise_level_var
            self.target_noise = target_noise
            self.ramp_epochs = ramp_epochs
            self._stage_start_epoch: Optional[int] = None

        def on_epoch_begin(
            self, epoch: int, logs: Optional[Dict] = None
        ) -> None:
            if self._stage_start_epoch is None:
                self._stage_start_epoch = epoch

            ep_rel = epoch - self._stage_start_epoch

            if ep_rel < self.ramp_epochs:
                frac = ep_rel / max(self.ramp_epochs, 1)
                current_noise = self.target_noise * frac
            else:
                current_noise = self.target_noise

            self.noise_level_var.assign(current_noise)

    return _MiniCurriculumCallbackKeras


# Override: usa versao Keras quando TF esta disponivel
try:
    _MiniCurriculumCallback = _register_callback_base()  # type: ignore[misc]
except ImportError:
    # TF nao disponivel (testes unitarios sem TF) — usa versao sem heranca
    pass
