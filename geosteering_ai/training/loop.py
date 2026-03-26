# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: training/loop.py                                                 ║
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
# ║    • TrainingLoop: orquestrador de compile + fit + causal finetuning       ║
# ║    • TrainingResult: dataclass com historico, tempo, metricas finais       ║
# ║    • Optimizer factory: adam, adamw, sgd, rmsprop, nadam, adagrad          ║
# ║    • Mixed precision support (opt-in via config.use_mixed_precision)       ║
# ║    • History merging para multi-stage training (N-Stage S21+)             ║
# ║    • Causal finetuning: refina modelo acausal para inferencia causal      ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), tensorflow (lazy import)       ║
# ║  Exports: ~2 simbolos (TrainingLoop, TrainingResult) — ver __all__        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6, legado C43                        ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C43)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""TrainingLoop — Orquestrador de treinamento do pipeline de inversao.

Encapsula o ciclo completo: compile → fit → causal finetuning (opcional).
Todas as decisoes de otimizacao sao derivadas de PipelineConfig (NUNCA
globals().get()). TensorFlow importado DENTRO de metodos (lazy import)
para compatibilidade com ambientes CPU-only e testes leves.

Fluxo de decisao:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  TrainingLoop.run(model, loss_fn, metrics, train_ds, val_ds, cbs)  │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. compile(model, loss_fn, metrics)                                │
    │     ├─ _create_optimizer(learning_rate)                              │
    │     │  ├─ "adam"   → tf.keras.optimizers.Adam                       │
    │     │  ├─ "adamw"  → tf.keras.optimizers.AdamW + weight_decay       │
    │     │  ├─ "sgd"    → tf.keras.optimizers.SGD + momentum             │
    │     │  ├─ "rmsprop"→ tf.keras.optimizers.RMSprop                    │
    │     │  ├─ "nadam"  → tf.keras.optimizers.Nadam                      │
    │     │  └─ "adagrad"→ tf.keras.optimizers.Adagrad                    │
    │     ├─ gradient_clipping (clip_norm se habilitado)                   │
    │     └─ mixed precision (LossScaleOptimizer se habilitado)           │
    │                                                                      │
    │  2. fit(model, train_ds, val_ds, callbacks)                         │
    │     ├─ model.fit(epochs, callbacks, verbose)                        │
    │     └─ registra history + training_time                              │
    │                                                                      │
    │  3. config.use_causal_mode?                                          │
    │     ├─ SIM: _causal_finetune(model, train_ds, val_ds)              │
    │     │       (LR reduzido, epochs curtos, preserva convergencia)     │
    │     └─ NAO: retorna TrainingResult                                  │
    └──────────────────────────────────────────────────────────────────────┘

Example:
    >>> from geosteering_ai import PipelineConfig
    >>> from geosteering_ai.training.loop import TrainingLoop
    >>>
    >>> config = PipelineConfig.robusto()
    >>> loop = TrainingLoop(config)
    >>> result = loop.run(model, loss_fn, metrics, train_ds, val_ds, cbs)
    >>> result.training_time
    1234.56

Note:
    Referenciado em:
        - geosteering_ai/__init__.py: import conceitual de TrainingLoop
        - training/__init__.py: re-export de TrainingLoop, TrainingResult
        - models/registry.py: docstring referencia training/loop.py
        - data/pipeline.py: PreparedData docstring referencia TrainingLoop
    Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
    Legado C43 (479 linhas) — compile+fit+SKIP_TRAINING+causal finetuning.
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Classe principal (loop de treinamento) ---
    "TrainingLoop",
    # --- Resultado do treinamento ---
    "TrainingResult",
]


# ════════════════════════════════════════════════════════════════════════════
# SECAO: TRAINING RESULT — Resultado imutavel do treinamento
# ════════════════════════════════════════════════════════════════════════════
# Encapsula o resultado do ciclo compile → fit (→ causal finetuning).
# history: dicionario epoch-by-epoch de metricas (loss, val_loss, etc.).
# training_time: duracao total em segundos (incluindo causal finetuning).
# final_epoch: ultima epoca executada (pode ser < epochs se early stopping).
# best_epoch: epoca com menor val_loss (-1 se nao monitorado).
# best_val_loss: valor do val_loss na best_epoch (inf se nao monitorado).
# final_metrics: metricas do ultimo epoch (dict extraido do history).
# ────────────────────────────────────────────────────────────────────────


@dataclass
class TrainingResult:
    """Resultado de uma execucao de treinamento.

    Container imutavel com historico epoch-by-epoch, tempo total,
    epoca final, melhor epoca (por val_loss), e metricas finais.
    Retornado por TrainingLoop.run() e TrainingLoop.fit().

    Attributes:
        history (dict): Metricas epoch-by-epoch. Chaves tipicas:
            "loss", "val_loss", "lr", e metricas customizadas.
            Cada valor e uma lista de floats indexada por epoch.
        training_time (float): Duracao total em segundos (float).
            Inclui fit() + causal finetuning (se aplicavel).
        final_epoch (int): Ultima epoca executada. Pode ser menor
            que config.epochs se EarlyStopping foi acionado.
        best_epoch (int): Epoca com menor val_loss. -1 se val_loss
            nao foi monitorado ou nao disponivel no historico.
        best_val_loss (float): Valor de val_loss na best_epoch.
            float('inf') se nao disponivel.
        final_metrics (dict): Metricas do ultimo epoch extraidas
            do history. Exemplo: {"loss": 0.05, "val_loss": 0.06}.

    Example:
        >>> result = TrainingResult(
        ...     history={"loss": [0.5, 0.3], "val_loss": [0.6, 0.4]},
        ...     training_time=120.5,
        ...     final_epoch=2,
        ...     best_epoch=2,
        ...     best_val_loss=0.4,
        ... )
        >>> result.best_val_loss
        0.4

    Note:
        Referenciado em:
            - training/loop.py: TrainingLoop.run() (retorno principal)
            - training/loop.py: TrainingLoop.fit() (popula campos)
        Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
        best_epoch e best_val_loss sao extraidos do history["val_loss"]
        via argmin. Se val_loss nao disponivel, permanecem default.
    """

    history: Dict[str, List[float]] = field(default_factory=dict)
    training_time: float = 0.0
    final_epoch: int = 0
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    final_metrics: Dict[str, float] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CONSTANTES DE CAUSAL FINETUNING
# ════════════════════════════════════════════════════════════════════════════
# Hiperparametros do estagio de causal finetuning.
# LR_DECAY_FACTOR: fator de reducao do LR em relacao ao LR principal.
#   Valor baixo (0.1) evita destruir convergencia do treinamento acausal.
# FINETUNE_EPOCHS: numero de epocas do finetuning causal.
#   Valor conservador (10) suficiente para adaptar padding sem overfitting.
# ────────────────────────────────────────────────────────────────────────

_CAUSAL_FINETUNE_LR_DECAY: float = 0.1
"""Fator de reducao do LR para causal finetuning (LR * 0.1)."""

_CAUSAL_FINETUNE_EPOCHS: int = 10
"""Numero de epocas para causal finetuning."""


# ════════════════════════════════════════════════════════════════════════════
# SECAO: TRAINING LOOP — Orquestrador de treinamento
# ════════════════════════════════════════════════════════════════════════════
# Classe principal que encapsula o ciclo compile → fit → finetuning.
# Todas as decisoes sao derivadas de PipelineConfig:
#   - optimizer: tipo, LR, weight_decay, gradient clipping
#   - mixed precision: LossScaleOptimizer (opt-in)
#   - causal finetuning: ativado por config.use_causal_mode
#   - verbose: nivel de logging durante model.fit()
#
# TensorFlow importado DENTRO dos metodos (lazy import) para:
#   1. Evitar import-time GPU allocation em ambientes CPU-only
#   2. Permitir testes leves sem GPU disponivel
#   3. Compatibilidade com importacao parcial do pacote
# ────────────────────────────────────────────────────────────────────────


class TrainingLoop:
    """Orquestrador de treinamento: compile, fit, e causal finetuning.

    Recebe PipelineConfig como unico parametro de configuracao e deriva
    todas as decisoes de otimizacao a partir dos campos do config.
    Gerencia optimizer, mixed precision, gradient clipping, e history
    merging para multi-stage training.

    Attributes:
        config (PipelineConfig): Configuracao do pipeline (imutavel).
        history (dict or None): Historico acumulado de metricas epoch-by-epoch.
            None antes do primeiro fit(). Atualizado a cada fit() via merge.
        training_time (float): Tempo total de treinamento em segundos.
            Acumulado entre chamadas a fit() (para multi-stage).

    Example:
        >>> from geosteering_ai import PipelineConfig
        >>> from geosteering_ai.training.loop import TrainingLoop
        >>>
        >>> config = PipelineConfig.robusto()
        >>> loop = TrainingLoop(config)
        >>> loop.compile(model, loss_fn, metrics_list)
        >>> result = loop.fit(model, train_ds, val_ds, callbacks)
        >>> result.training_time  # tempo em segundos
        1234.56

    Note:
        Referenciado em:
            - geosteering_ai/__init__.py: docstring (exemplo de uso)
            - models/registry.py: docstring referencia TrainingLoop.run()
            - data/pipeline.py: PreparedData docstring referencia TrainingLoop
        Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
        Legado C43: compile+fit+causal finetuning em ~479 linhas imperativas.
        Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
        NUNCA usar globals().get() — PipelineConfig e o ponto unico de verdade.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Inicializa TrainingLoop com configuracao do pipeline.

        Args:
            config: PipelineConfig com todos os hiperparametros de
                treinamento (LR, epochs, optimizer, gradient clipping, etc.).
                Deve ser uma instancia validada (errata verificada no
                __post_init__).

        Note:
            Referenciado em:
                - training/loop.py: run(), compile(), fit()
            Ref: docs/ARCHITECTURE_v2.md secao 6.
            history e None ate o primeiro fit(). training_time e acumulado
            entre chamadas a fit() para suportar N-Stage training (S21+).
        """
        self.config = config
        self.history: Optional[Dict[str, List[float]]] = None
        self.training_time: float = 0.0

    # ──────────────────────────────────────────────────────────────────
    # SECAO: OPTIMIZER FACTORY
    # ──────────────────────────────────────────────────────────────────
    # Cria optimizer Keras com base em config.optimizer.
    # Suporta 6 tipos: adam, adamw, sgd, rmsprop, nadam, adagrad.
    # Gradient clipping via clipnorm (quando habilitado no config).
    # AdamW inclui weight_decay para regularizacao implicita.
    # SGD inclui momentum=0.9 e Nesterov para convergencia acelerada.
    # ──────────────────────────────────────────────────────────────────

    def _create_optimizer(
        self,
        learning_rate: Optional[float] = None,
    ) -> Any:
        """Cria optimizer Keras com base em config.optimizer.

        Factory que mapeia o nome do optimizer (string) para a classe
        Keras correspondente, aplicando LR, weight_decay, e gradient
        clipping conforme configurado.

        Args:
            learning_rate: Taxa de aprendizado. Se None, usa
                config.learning_rate. Permite override para causal
                finetuning e N-Stage (LR reduzido por stage).

        Returns:
            tf.keras.optimizers.Optimizer: Instancia do optimizer Keras
            com LR, weight_decay e clipnorm configurados.

        Raises:
            ValueError: Se config.optimizer nao for reconhecido (embora
                PipelineConfig.__post_init__ ja valide os tipos aceitos).

        Example:
            >>> loop = TrainingLoop(config)
            >>> opt = loop._create_optimizer()         # usa config.learning_rate
            >>> opt = loop._create_optimizer(lr=1e-5)  # override para finetuning

        Note:
            Referenciado em:
                - training/loop.py: compile() (chamada principal)
                - training/loop.py: _causal_finetune() (LR reduzido)
            Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
            Gradient clipping via clipnorm (config.gradient_clip_norm)
            quando config.use_gradient_clipping=True.
            AdamW: weight_decay parametro dedicado (regularizacao L2 implicita).
            SGD: momentum=0.9 + nesterov=True (aceleracao de convergencia).
            Validacao de config.optimizer feita em PipelineConfig.__post_init__
            (valores aceitos: adam, adamw, sgd, rmsprop, nadam, adagrad).
        """
        import tensorflow as tf  # lazy import — D10 compatibilidade CPU-only

        lr = learning_rate if learning_rate is not None else self.config.learning_rate

        # ── Gradient clipping (clip_norm global) ────────────────────
        # Aplicado via clipnorm no optimizer Keras. Previne explosao de
        # gradientes em treinamento com noise alto (cenario E-Robusto).
        clip_kwargs: Dict[str, Any] = {}
        if self.config.use_gradient_clipping:
            clip_kwargs["clipnorm"] = self.config.gradient_clip_norm
            logger.debug(
                "Gradient clipping ativo: clipnorm=%.4f",
                self.config.gradient_clip_norm,
            )

        # ── Factory de optimizer ────────────────────────────────────
        # Mapeamento config.optimizer (str) → classe Keras.
        opt_name = self.config.optimizer.lower()

        if opt_name == "adam":
            # Adam padrao sem weight_decay
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                **clip_kwargs,
            )
        elif opt_name == "adamw":
            # AdamW com weight_decay (regularizacao L2 decoupled)
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=self.config.weight_decay,
                **clip_kwargs,
            )
        elif opt_name == "sgd":
            # SGD com momentum Nesterov para convergencia acelerada
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=0.9,
                nesterov=True,
                **clip_kwargs,
            )
        elif opt_name == "rmsprop":
            # RMSprop — adaptativo, bom para RNNs
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=lr,
                **clip_kwargs,
            )
        elif opt_name == "nadam":
            # Nadam — Adam + Nesterov momentum
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate=lr,
                **clip_kwargs,
            )
        elif opt_name == "adagrad":
            # Adagrad — LR adaptativo por parametro
            optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=lr,
                **clip_kwargs,
            )
        else:
            # Fallback defensivo (PipelineConfig.__post_init__ ja valida)
            raise ValueError(
                f"Optimizer '{self.config.optimizer}' nao reconhecido. "
                f"Validos: adam, adamw, sgd, rmsprop, nadam, adagrad."
            )

        logger.info(
            "Optimizer criado: %s (LR=%.2e, clip=%s)",
            opt_name,
            lr,
            f"{self.config.gradient_clip_norm:.2f}"
            if self.config.use_gradient_clipping
            else "off",
        )
        return optimizer

    # ──────────────────────────────────────────────────────────────────
    # SECAO: COMPILE — Compilacao do modelo Keras
    # ──────────────────────────────────────────────────────────────────
    # Reune optimizer + loss + metrics e compila o modelo Keras.
    # Mixed precision (opt-in): wrapa optimizer com LossScaleOptimizer
    # para treinar em float16 com escalamento automatico de gradientes.
    # Evita underflow em mixed precision mantendo master weights em fp32.
    # ──────────────────────────────────────────────────────────────────

    def compile(
        self,
        model: Any,
        loss_fn: Callable,
        metrics_list: Optional[List[Any]] = None,
        *,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Compila modelo Keras com optimizer, loss, e metricas.

        Cria o optimizer via _create_optimizer(), aplica mixed precision
        se habilitado, e chama model.compile(). O modelo fica pronto
        para model.fit() apos esta chamada.

        Args:
            model: Modelo Keras (tf.keras.Model) nao compilado.
                Espera-se que ja tenha input/output shapes definidos
                (construido via ModelRegistry.build(config)).
            loss_fn: Funcao de perda Keras-compatible. Pode ser string
                ("mse"), funcao (rmse_loss), ou instancia de tf.keras.losses.
                Tipicamente obtida via LossFactory.build_combined(config).
            metrics_list: Lista de metricas Keras para monitorar.
                Se None, usa lista vazia. Exemplos: ["mae"], ou
                instancias customizadas (R2Score, PerComponentMetric).
            learning_rate: Override do LR para compilacao. Se None,
                usa config.learning_rate. Util para causal finetuning.

        Example:
            >>> loop = TrainingLoop(config)
            >>> loop.compile(model, loss_fn, ["mae"])
            >>> # modelo pronto para fit()

        Note:
            Referenciado em:
                - training/loop.py: run() (passo 1)
                - training/loop.py: _causal_finetune() (recompila com LR baixo)
            Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
            Mixed precision via tf.keras.mixed_precision.LossScaleOptimizer.
            Ativado quando config.use_mixed_precision=True.
            Policy global deve ser setada ANTES de build_model() (nao aqui).
        """
        import tensorflow as tf  # lazy import

        # ── Criar optimizer ──────────────────────────────────────────
        optimizer = self._create_optimizer(learning_rate=learning_rate)

        # ── Mixed precision (opt-in) ────────────────────────────────
        # Wrapa optimizer com LossScaleOptimizer para float16 training.
        # Escala loss para cima antes do backward pass (evita underflow)
        # e escala gradientes de volta antes do optimizer step.
        if self.config.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            logger.info(
                "Mixed precision ativo: LossScaleOptimizer wrapping %s",
                self.config.optimizer,
            )

        # ── Compilar modelo ──────────────────────────────────────────
        if metrics_list is None:
            metrics_list = []

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics_list,
        )

        # ── Logging de confirmacao ───────────────────────────────────
        n_params = model.count_params() if hasattr(model, "count_params") else "?"
        logger.info(
            "Modelo compilado: %s params, loss=%s, %d metricas, optimizer=%s",
            f"{n_params:,}" if isinstance(n_params, int) else n_params,
            getattr(loss_fn, "__name__", str(loss_fn)),
            len(metrics_list),
            self.config.optimizer,
        )

    # ──────────────────────────────────────────────────────────────────
    # SECAO: FIT — Execucao do treinamento
    # ──────────────────────────────────────────────────────────────────
    # Executa model.fit() com train_ds, val_ds, callbacks.
    # Registra tempo total e atualiza historico acumulado.
    # Suporta multi-stage via initial_epoch e history merging.
    # verbose derivado de config.verbose (1 = progresso, 0 = silencioso).
    # ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        model: Any,
        train_ds: Any,
        val_ds: Any,
        callbacks: Optional[List[Any]] = None,
        *,
        epochs: Optional[int] = None,
        initial_epoch: int = 0,
    ) -> TrainingResult:
        """Executa model.fit() e registra historico + tempo de treinamento.

        Chama model.fit() com os datasets e callbacks fornecidos. Mede
        o tempo total de execucao e extrai metricas finais do historico.
        O historico e acumulado (merged) entre chamadas para suportar
        N-Stage training (multiplas chamadas a fit() sequenciais).

        Args:
            model: Modelo Keras compilado (apos self.compile()).
            train_ds: tf.data.Dataset para treinamento. Deve estar
                configurado com batch, prefetch, e map_fn (noise, FV,
                GS, scale) conforme DataPipeline.build_train_map_fn().
            val_ds: tf.data.Dataset para validacao. Processado offline
                (FV+GS+scale sem noise). Pode ser None se sem validacao.
            callbacks: Lista de Keras callbacks. Tipicamente inclui
                EarlyStopping, ModelCheckpoint, UpdateNoiseLevelCallback.
                Se None, usa lista vazia.
            epochs: Numero maximo de epocas. Se None, usa config.epochs.
                Permite override para causal finetuning ou N-Stage.
            initial_epoch: Epoch inicial (0-indexed). Para N-Stage,
                Stage 2 comeca em initial_epoch = Stage 1 final_epoch.

        Returns:
            TrainingResult: Resultado com history, training_time,
            final_epoch, best_epoch, best_val_loss, final_metrics.

        Example:
            >>> loop = TrainingLoop(config)
            >>> loop.compile(model, loss_fn, metrics)
            >>> result = loop.fit(model, train_ds, val_ds, cbs)
            >>> result.final_epoch
            150

        Note:
            Referenciado em:
                - training/loop.py: run() (passo 2)
                - training/loop.py: _causal_finetune() (fit curto)
            Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
            verbose: 1 se config.verbose=True, 0 caso contrario.
            History merging: concatena listas de metricas epoch-by-epoch
            para preservar historico completo em multi-stage.
            training_time e acumulado (self.training_time +=) para
            capturar tempo total em cenarios multi-fit.
        """
        if callbacks is None:
            callbacks = []

        n_epochs = epochs if epochs is not None else self.config.epochs
        verbose_level = 1 if self.config.verbose else 0

        logger.info(
            "Iniciando fit: epochs=%d (initial_epoch=%d), batch_size=%d, "
            "callbacks=%d",
            n_epochs,
            initial_epoch,
            self.config.batch_size,
            len(callbacks),
        )

        # ── Execucao cronometrada ────────────────────────────────────
        t_start = time.perf_counter()

        keras_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=n_epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=verbose_level,
        )

        elapsed = time.perf_counter() - t_start
        self.training_time += elapsed  # acumulado para multi-stage

        # ── Extrair historico como dict de listas ────────────────────
        new_history: Dict[str, List[float]] = {
            k: list(v) for k, v in keras_history.history.items()
        }

        # ── Merge com historico existente (multi-stage) ──────────────
        # Em N-Stage, cada fit() adiciona suas epocas ao historico global.
        # Chaves novas sao criadas; chaves existentes sao estendidas.
        self.history = _merge_histories(self.history, new_history)

        # ── Extrair final_epoch, best_epoch, best_val_loss ──────────
        # final_epoch: epoca real (initial_epoch + len(history["loss"]))
        actual_epochs_run = len(new_history.get("loss", []))
        final_epoch = initial_epoch + actual_epochs_run

        best_epoch, best_val_loss = _extract_best_epoch(self.history)

        # ── Metricas do ultimo epoch ─────────────────────────────────
        final_metrics: Dict[str, float] = {}
        for key, values in new_history.items():
            if values:
                final_metrics[key] = float(values[-1])

        logger.info(
            "Fit concluido: %d epocas em %.1f s | "
            "final_epoch=%d | best_epoch=%d | best_val_loss=%.6f",
            actual_epochs_run,
            elapsed,
            final_epoch,
            best_epoch,
            best_val_loss,
        )

        return TrainingResult(
            history=self.history,
            training_time=self.training_time,
            final_epoch=final_epoch,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            final_metrics=final_metrics,
        )

    # ──────────────────────────────────────────────────────────────────
    # SECAO: CAUSAL FINETUNING
    # ──────────────────────────────────────────────────────────────────
    # Estagio opcional: refina modelo treinado em modo acausal para
    # inferencia causal (geosteering realtime). Usa LR reduzido
    # (LR * _CAUSAL_FINETUNE_LR_DECAY) para nao destruir convergencia.
    # Poucas epocas (_CAUSAL_FINETUNE_EPOCHS) suficientes para adaptar
    # padding/masking sem overfitting. Ativado quando
    # config.use_causal_mode=True (auto-derivado de inference_mode).
    # ──────────────────────────────────────────────────────────────────

    def _causal_finetune(
        self,
        model: Any,
        loss_fn: Callable,
        metrics_list: Optional[List[Any]],
        train_ds: Any,
        val_ds: Any,
        callbacks: Optional[List[Any]] = None,
        *,
        initial_epoch: int = 0,
    ) -> TrainingResult:
        """Executa causal finetuning com LR reduzido.

        Recompila o modelo com LR = config.learning_rate * decay_factor
        e treina por poucas epocas adicionais. Preserva a convergencia
        do treinamento acausal enquanto adapta o modelo para masking
        causal (padding, lookback window).

        Args:
            model: Modelo Keras ja treinado (pos fit() principal).
            loss_fn: Funcao de perda (mesma do treinamento principal).
            metrics_list: Lista de metricas (mesma do treinamento principal).
            train_ds: tf.data.Dataset para treinamento.
            val_ds: tf.data.Dataset para validacao.
            callbacks: Lista de callbacks para finetuning. Se None,
                usa lista vazia (sem EarlyStopping no finetuning).
            initial_epoch: Epoch inicial para continuidade do historico.

        Returns:
            TrainingResult: Resultado com historico merged (inclui
            epocas do finetuning causal no history global).

        Note:
            Referenciado em:
                - training/loop.py: run() (passo 3, condicional)
            Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
            LR_DECAY = 0.1 → LR_finetune = config.learning_rate * 0.1.
            EPOCHS = 10 → suficiente para adaptar padding causal.
            Legado C43: PARTE 8 — causal finetuning com LR * 0.1.
        """
        finetune_lr = self.config.learning_rate * _CAUSAL_FINETUNE_LR_DECAY
        finetune_epochs = initial_epoch + _CAUSAL_FINETUNE_EPOCHS

        logger.info(
            "Causal finetuning: LR=%.2e (decay=%.2f), epochs=%d, "
            "initial_epoch=%d",
            finetune_lr,
            _CAUSAL_FINETUNE_LR_DECAY,
            _CAUSAL_FINETUNE_EPOCHS,
            initial_epoch,
        )

        # ── Recompilar com LR reduzido ──────────────────────────────
        self.compile(
            model,
            loss_fn,
            metrics_list,
            learning_rate=finetune_lr,
        )

        # ── Fit curto para adaptar causal masking ────────────────────
        result = self.fit(
            model,
            train_ds,
            val_ds,
            callbacks=callbacks,
            epochs=finetune_epochs,
            initial_epoch=initial_epoch,
        )

        logger.info(
            "Causal finetuning concluido: best_val_loss=%.6f (epoch %d)",
            result.best_val_loss,
            result.best_epoch,
        )
        return result

    # ──────────────────────────────────────────────────────────────────
    # SECAO: RUN — Pipeline completo (compile + fit + finetuning)
    # ──────────────────────────────────────────────────────────────────
    # Entry point principal. Executa o ciclo completo:
    #   1. compile(model, loss_fn, metrics)
    #   2. fit(model, train_ds, val_ds, callbacks)
    #   3. Se config.use_causal_mode: causal finetuning
    # Retorna TrainingResult com historico, tempo, e metricas finais.
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        model: Any,
        loss_fn: Callable,
        metrics_list: Optional[List[Any]],
        train_ds: Any,
        val_ds: Any,
        callbacks: Optional[List[Any]] = None,
    ) -> TrainingResult:
        """Pipeline completo: compile + fit + causal finetuning (opcional).

        Entry point principal do TrainingLoop. Executa:
        1. compile() — cria optimizer, aplica mixed precision, compila
        2. fit() — treina com datasets e callbacks
        3. Se config.use_causal_mode: _causal_finetune() com LR reduzido

        Args:
            model: Modelo Keras nao compilado (output do ModelRegistry).
            loss_fn: Funcao de perda. Tipicamente de LossFactory.build_combined().
            metrics_list: Lista de metricas Keras.
            train_ds: tf.data.Dataset para treinamento.
            val_ds: tf.data.Dataset para validacao.
            callbacks: Lista de Keras callbacks. Se None, usa lista vazia.

        Returns:
            TrainingResult: Resultado com history (incluindo finetuning
            se aplicavel), training_time total, best_epoch, best_val_loss,
            e final_metrics.

        Example:
            >>> config = PipelineConfig.robusto()
            >>> loop = TrainingLoop(config)
            >>> result = loop.run(model, loss_fn, metrics, train_ds, val_ds, cbs)
            >>> result.best_val_loss
            0.0523

        Note:
            Referenciado em:
                - geosteering_ai/__init__.py: docstring (exemplo de uso)
                - models/registry.py: docstring referencia TrainingLoop.run()
            Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
            Causal finetuning ativado quando config.use_causal_mode=True
            (auto-derivado de inference_mode="realtime" no __post_init__).
            Para N-Stage, chamar fit() multiplas vezes em vez de run().
        """
        logger.info(
            "TrainingLoop.run(): model_type=%s, optimizer=%s, LR=%.2e, "
            "epochs=%d, mixed_precision=%s, causal=%s",
            self.config.model_type,
            self.config.optimizer,
            self.config.learning_rate,
            self.config.epochs,
            self.config.use_mixed_precision,
            self.config.use_causal_mode,
        )

        # ── Passo 1: Compile ─────────────────────────────────────────
        self.compile(model, loss_fn, metrics_list)

        # ── Passo 2: Fit principal ───────────────────────────────────
        result = self.fit(
            model,
            train_ds,
            val_ds,
            callbacks=callbacks,
        )

        # ── Passo 3: Causal finetuning (condicional) ────────────────
        # Ativado quando config.use_causal_mode=True. Recompila com LR
        # reduzido e treina por poucas epocas adicionais. Preserva
        # convergencia do treinamento acausal.
        if self.config.use_causal_mode:
            logger.info(
                "use_causal_mode=True: iniciando causal finetuning "
                "(a partir de epoch %d)",
                result.final_epoch,
            )
            result = self._causal_finetune(
                model,
                loss_fn,
                metrics_list,
                train_ds,
                val_ds,
                callbacks=callbacks,
                initial_epoch=result.final_epoch,
            )

        logger.info(
            "TrainingLoop.run() concluido: %.1f s total, "
            "best_val_loss=%.6f (epoch %d), final_epoch=%d",
            result.training_time,
            result.best_val_loss,
            result.best_epoch,
            result.final_epoch,
        )
        return result


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCOES AUXILIARES (module-level)
# ════════════════════════════════════════════════════════════════════════════
# _merge_histories: concatena historicos epoch-by-epoch de multiplas
#   chamadas a fit() (N-Stage training). Chaves novas sao criadas;
#   chaves existentes tem suas listas estendidas.
# _extract_best_epoch: localiza a epoca com menor val_loss no historico
#   acumulado. Retorna (-1, inf) se val_loss nao disponivel.
# ────────────────────────────────────────────────────────────────────────


def _merge_histories(
    existing: Optional[Dict[str, List[float]]],
    new: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """Merge dois historicos de treinamento (epoch-by-epoch).

    Concatena listas de metricas do historico existente com as do
    novo fit(). Permite acumular historico entre multiplas chamadas
    a fit() (N-Stage training, causal finetuning).

    Args:
        existing: Historico acumulado anterior. None na primeira chamada.
        new: Historico do fit() mais recente.

    Returns:
        Dict com metricas concatenadas. Chaves novas sao criadas;
        chaves existentes tem suas listas estendidas.

    Example:
        >>> h1 = {"loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}
        >>> h2 = {"loss": [0.3], "val_loss": [0.4]}
        >>> merged = _merge_histories(h1, h2)
        >>> merged["loss"]
        [0.5, 0.4, 0.3]

    Note:
        Referenciado em:
            - training/loop.py: TrainingLoop.fit() (history merging)
        Ref: docs/ARCHITECTURE_v2.md secao 6 (Training / N-Stage).
        Defensivo: trata existing=None como dict vazio.
        Chaves presentes apenas em existing sao preservadas.
    """
    if existing is None:
        return dict(new)

    merged = {k: list(v) for k, v in existing.items()}
    for key, values in new.items():
        if key in merged:
            merged[key].extend(values)  # concatena epocas
        else:
            merged[key] = list(values)  # chave nova
    return merged


def _extract_best_epoch(
    history: Optional[Dict[str, List[float]]],
) -> tuple:
    """Extrai best_epoch e best_val_loss do historico acumulado.

    Localiza a epoca (1-indexed) com menor val_loss. Se val_loss
    nao esta disponivel no historico, retorna (-1, inf).

    Args:
        history: Historico acumulado de metricas. None se nenhum
            fit() foi executado.

    Returns:
        Tuple (best_epoch, best_val_loss):
            - best_epoch (int): Epoca 1-indexed com menor val_loss.
              -1 se val_loss nao disponivel.
            - best_val_loss (float): Valor de val_loss na best_epoch.
              float('inf') se nao disponivel.

    Example:
        >>> h = {"loss": [0.5, 0.3], "val_loss": [0.6, 0.4]}
        >>> _extract_best_epoch(h)
        (2, 0.4)

    Note:
        Referenciado em:
            - training/loop.py: TrainingLoop.fit() (calculo de best_epoch)
        Ref: docs/ARCHITECTURE_v2.md secao 6.
        Epoch 1-indexed: argmin + 1. Compativel com Keras convention.
    """
    if history is None:
        return -1, float("inf")

    val_losses = history.get("val_loss", [])
    if not val_losses:
        return -1, float("inf")

    # argmin sobre val_loss acumulado — epoch 1-indexed
    best_idx = int(min(range(len(val_losses)), key=lambda i: val_losses[i]))
    best_val_loss = float(val_losses[best_idx])
    best_epoch = best_idx + 1  # 1-indexed

    return best_epoch, best_val_loss
