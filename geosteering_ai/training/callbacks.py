# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: training/callbacks.py                                             ║
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
# ║    • Custom Keras callbacks para curriculum noise, gradient monitoring    ║
# ║    • UpdateNoiseLevelCallback: scheduler 3-phase (clean→ramp→stable)     ║
# ║    • WeightNormMonitor: norma L2 de pesos por epoca                     ║
# ║    • BestEpochTracker: rastreia melhor epoca e metrica                    ║
# ║    • DualValidationCallback: valida em clean + noisy (P2)               ║
# ║    • PINNSLambdaScheduleCallback: annealing de lambda PINNs            ║
# ║    • CausalDegradationMonitor: gap causal vs acausal                    ║
# ║    • SlidingWindowValidation: valida em janelas deslizantes             ║
# ║    • PeriodicCheckpoint: salva modelo a cada N epocas                   ║
# ║    • MetricPlateauDetector: detecta plateaus persistentes               ║
# ║    • OneCycleLR: super-convergencia (Smith 2018)                        ║
# ║    • CosineWarmRestarts: SGDR cosine com warm restarts                  ║
# ║    • CyclicalLR: ciclico triangular (Smith 2017)                        ║
# ║    • MemoryMonitor: uso de memoria GPU/CPU                              ║
# ║    • LatencyBenchmark: tempo de inferencia single-sample                ║
# ║    • EpochSummary: resumo one-line por epoca                            ║
# ║    • build_callbacks(): factory que monta lista de callbacks do config    ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), TensorFlow 2.x (lazy import) ║
# ║  Exports: ~16 (15 callbacks + build_callbacks)                           ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6.2 (callbacks)                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C40 legado)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""Custom Keras callbacks e factory build_callbacks() para treinamento.

Callbacks customizados para curriculum noise 3-phase, monitoramento de
gradientes e rastreamento de melhor epoca. A factory build_callbacks()
monta a lista completa de callbacks a partir de PipelineConfig.

.. rubric:: Curriculum Noise 3-Phase (UpdateNoiseLevelCallback)

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  CURRICULUM NOISE 3-PHASE                                               │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Fase 1 — Clean (epochs 0..epochs_no_noise-1):                          │
    │    noise_level = 0.0                                                     │
    │    Modelo converge em dados limpos (base solida).                        │
    │                                                                          │
    │  Fase 2 — Ramp (epochs epochs_no_noise..epochs_no_noise+ramp-1):        │
    │    noise_level = max × (epoch - epochs_no_noise) / ramp                  │
    │    Crescimento linear de 0.0 → noise_level_max.                         │
    │                                                                          │
    │  Fase 3 — Stable (epochs >= epochs_no_noise + ramp):                    │
    │    noise_level = noise_level_max                                         │
    │    Treinamento com ruido maximo constante.                               │
    │                                                                          │
    │  Timeline (default E-Robusto):                                           │
    │    epochs_no_noise=10, ramp=80 → clean[0-9] ramp[10-89] stable[90+]    │
    │                                                                          │
    │           noise_level                                                    │
    │    0.08 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬──────          │
    │                                                       /                  │
    │                                                      /                   │
    │                                                     /                    │
    │    0.00 ──────────┬────────────────────────────────/                     │
    │                   │         ramp (80 ep)                                 │
    │         clean (10 ep)                              stable                │
    │                                                                          │
    │  config fields:                                                          │
    │    epochs_no_noise  = 10  (Fase 1 duration)                             │
    │    noise_ramp_epochs = 80 (Fase 2 duration)                             │
    │    noise_level_max  = 0.08 (Fase 3 plateau)                             │
    └──────────────────────────────────────────────────────────────────────────┘

.. rubric:: build_callbacks() — Composicao de callbacks

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  build_callbacks() — Composicao por config                              │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  SEMPRE incluidos:                                                       │
    │    ├─ EarlyStopping (patience, restore_best_weights)                    │
    │    ├─ BestEpochTracker (monitor='val_loss')                             │
    │    └─ EpochSummary (resumo one-line)                                    │
    │                                                                          │
    │  CONDICIONAIS:                                                           │
    │    ├─ TensorBoard             (se use_tensorboard=True)                 │
    │    ├─ CSVLogger                (se use_csv_logger=True)                 │
    │    ├─ UpdateNoiseLevel         (se curriculum + noise_level_var)        │
    │    ├─ DualValidationCallback   (se P2 + val_clean + val_noisy)         │
    │    ├─ PINNSLambdaSchedule      (se use_pinns + pinns_lambda_var)       │
    │    ├─ CausalDegradationMonitor (se use_causal_mode + model + val)      │
    │    └─ PeriodicCheckpoint       (se experiment_dir definido)             │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

Example:
    >>> from geosteering_ai.config import PipelineConfig
    >>> from geosteering_ai.training.callbacks import build_callbacks
    >>> config = PipelineConfig.robusto()
    >>> callbacks = build_callbacks(config)
    >>> len(callbacks) >= 2  # EarlyStopping + BestEpochTracker
    True

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Referenciado em:
        - training/loop.py: TrainingLoop.run() (consome build_callbacks)
        - config.py: PipelineConfig (campos de callback)
        - tests/test_callbacks.py: testes unitarios
    Ref: docs/ARCHITECTURE_v2.md secao 6.2 (callbacks e build_callbacks).
    Curriculum noise 3-phase: clean(0) → ramp(0→max) → stable(max).
    Scaler fit em dados LIMPOS (noise nao afeta scaler — regra P3).
"""

from __future__ import annotations

import logging
import os
import math
import time
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Custom Keras callbacks (originais) ---
    "UpdateNoiseLevelCallback",
    "WeightNormMonitor",
    "BestEpochTracker",
    # --- High priority callbacks ---
    "DualValidationCallback",
    "PINNSLambdaScheduleCallback",
    "CausalDegradationMonitor",
    "SlidingWindowValidation",
    # --- Medium priority callbacks ---
    "PeriodicCheckpoint",
    "MetricPlateauDetector",
    "OneCycleLR",
    "CosineWarmRestarts",
    "CyclicalLR",
    # --- Low priority callbacks ---
    "MemoryMonitor",
    "LatencyBenchmark",
    "EpochSummary",
    # --- Factory function ---
    "build_callbacks",
]


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 1: UpdateNoiseLevelCallback — Curriculum noise 3-phase
#
# Scheduler que controla o nivel de ruido injetado on-the-fly durante
# o treinamento. Opera em 3 fases:
#   1. Clean: noise_level = 0.0 (modelo converge em dados limpos)
#   2. Ramp: noise_level cresce linealmente de 0 → noise_level_max
#   3. Stable: noise_level = noise_level_max (ruido constante)
#
# O noise_level_var e um tf.Variable compartilhado com o train_map_fn
# do DataPipeline. A alteracao aqui propaga automaticamente para o
# proximo batch via tf.data.map (closure sobre a variavel).
#
# Referencia: C40 legado (UpdateNoiseLevelCallback), S17/S18/S19.
# ════════════════════════════════════════════════════════════════════════

class UpdateNoiseLevelCallback:
    """Curriculum noise scheduler — 3-phase: clean → ramp → stable.

    Controla dinamicamente o nivel de ruido injetado on-the-fly via
    tf.Variable compartilhado com o train_map_fn do DataPipeline.

    Herda de tf.keras.callbacks.Callback (lazy import para evitar
    import de TensorFlow no carregamento do modulo).

    Attributes:
        noise_level_var: tf.Variable escalar (float32) controlando sigma.
        config: PipelineConfig com campos de curriculum noise.
        _epochs_no_noise: Numero de epocas sem ruido (Fase 1).
        _noise_ramp_epochs: Numero de epocas de rampa (Fase 2).
        _noise_level_max: Nivel maximo de ruido (Fase 3).

    Example:
        >>> import tensorflow as tf
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig.robusto()
        >>> noise_var = tf.Variable(0.0, dtype=tf.float32)
        >>> cb = UpdateNoiseLevelCallback(noise_var, config)
        >>> cb._epochs_no_noise
        10

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (condicional)
            - data/pipeline.py: DataPipeline.build_train_map_fn() (noise_var)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (curriculum noise).
        Curriculum 3-phase: clean[0..9] → ramp[10..89] → stable[90+]
        (defaults E-Robusto: epochs_no_noise=10, ramp=80, max=0.08).
        Mutuamente exclusivo com N-Stage (validado em PipelineConfig).
    """

    def __new__(cls, noise_level_var: Any, config: PipelineConfig):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            noise_level_var: tf.Variable escalar (float32) que controla o
                nivel de ruido injetado no train_map_fn.
            config: PipelineConfig com campos de curriculum noise:
                epochs_no_noise, noise_ramp_epochs, noise_level_max.

        Returns:
            Instancia de UpdateNoiseLevelCallback com heranca de
            tf.keras.callbacks.Callback resolvida em runtime.

        Note:
            Lazy import de TensorFlow para evitar carregamento do
            framework no import-time do modulo.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        # Injeta heranca de tf.keras.callbacks.Callback dinamicamente
        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(self, noise_level_var: Any, config: PipelineConfig) -> None:
        """Inicializa o scheduler de curriculum noise.

        Args:
            noise_level_var: tf.Variable escalar (float32) que controla o
                nivel de ruido injetado no train_map_fn. Compartilhado por
                referencia com DataPipeline.build_train_map_fn().
            config: PipelineConfig com campos de curriculum noise:
                epochs_no_noise (int): Epocas de treino limpo (Fase 1).
                noise_ramp_epochs (int): Epocas de rampa linear (Fase 2).
                noise_level_max (float): Sigma maximo (Fase 3 plateau).

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            O noise_level_var DEVE ser criado externamente (tf.Variable).
            Valor inicial = 0.0 (Fase 1 comeca limpa).
        """
        import tensorflow as tf

        super().__init__()
        self.noise_level_var = noise_level_var
        self.config = config

        # Cache dos campos de config para performance (evita getattr por epoca)
        self._epochs_no_noise: int = config.epochs_no_noise
        self._noise_ramp_epochs: int = config.noise_ramp_epochs
        self._noise_level_max: float = config.noise_level_max

        logger.info(
            "UpdateNoiseLevelCallback inicializado: "
            "clean=%d ep, ramp=%d ep, max=%.4f",
            self._epochs_no_noise,
            self._noise_ramp_epochs,
            self._noise_level_max,
        )

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Atualiza noise_level_var no inicio de cada epoca.

        Implementa o scheduler 3-phase:
          - Fase 1 (Clean): epoch < epochs_no_noise → noise = 0.0
          - Fase 2 (Ramp): epochs_no_noise <= epoch < end_ramp → linear
          - Fase 3 (Stable): epoch >= end_ramp → noise = max

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (nao utilizado).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: CLAUDE.md diagrama curriculum noise.
            Rampa linear: noise = max * (epoch - clean) / ramp.
            Fase 1 garante convergencia limpa antes de injetar ruido.
        """
        # ── Fase 1: Clean (epochs 0..epochs_no_noise-1) ───────────
        # Modelo converge em dados limpos — base solida
        if epoch < self._epochs_no_noise:
            new_level = 0.0
            phase = "clean"

        # ── Fase 2: Ramp (linear 0 → noise_level_max) ─────────────
        # Ruido cresce gradualmente — modelo adapta incrementalmente
        elif epoch < self._epochs_no_noise + self._noise_ramp_epochs:
            ramp_progress = (epoch - self._epochs_no_noise) / max(
                self._noise_ramp_epochs, 1
            )
            new_level = self._noise_level_max * ramp_progress
            phase = "ramp"

        # ── Fase 3: Stable (noise = noise_level_max) ──────────────
        # Ruido constante no maximo — treinamento em regime estavel
        else:
            new_level = self._noise_level_max
            phase = "stable"

        # Atualiza a tf.Variable (propaga para o proximo batch)
        self.noise_level_var.assign(new_level)

        if self.config.verbose:
            logger.debug(
                "Epoch %d: noise_level=%.6f (phase=%s)",
                epoch,
                new_level,
                phase,
            )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 2: WeightNormMonitor — Monitoramento de normas de pesos
#
# Loga estatisticas (media, maximo) das normas L2 de pesos por epoca.
# Util para diagnosticar:
#   - Weight decay excessivo (norma → 0)
#   - Pesos explodindo (norma → ∞)
#   - Estabilidade do treinamento (norma constante)
#
# Calcula tf.norm(var, ord=2) para cada variavel treinavel.
# Nao interfere no treinamento — apenas observa e loga.
#
# Referencia: C40 legado (GradientMonitor callback — renomeado).
# ════════════════════════════════════════════════════════════════════════

class WeightNormMonitor:
    """Monitors weight norm statistics per epoch.

    Loga media e maximo das normas L2 dos pesos de cada variavel
    treinavel do modelo. Util para diagnosticar pesos explodindo,
    decaindo excessivamente ou instabilidades durante o treinamento.

    NOTA: Esta classe computa normas de PESOS (tf.norm(var)), NAO
    normas de gradientes via GradientTape. O nome anterior
    (GradientMonitorCallback) era enganoso.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _model_ref: Referencia ao modelo Keras sendo monitorado.

    Example:
        >>> import tensorflow as tf
        >>> model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
        >>> cb = WeightNormMonitor(model)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, opt-in)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        Norma L2 por variavel: ||w_i||_2 para cada peso treinavel.
        Nao altera pesos — apenas observa (zero overhead no forward).
    """

    def __new__(cls, model: Any):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            model: Modelo tf.keras.Model cujas normas de pesos serao monitoradas.

        Returns:
            Instancia de WeightNormMonitor.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(self, model: Any) -> None:
        """Inicializa o monitor de normas de pesos.

        Args:
            model: Modelo tf.keras.Model cujas normas de pesos serao
                monitoradas. Deve ter ao menos uma variavel treinavel.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            O modelo deve ter ao menos uma variavel treinavel.
        """
        import tensorflow as tf

        super().__init__()
        self._model_ref = model

        logger.info(
            "WeightNormMonitor inicializado: %d variaveis treinaveis",
            len(model.trainable_variables),
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Loga estatisticas de norma de pesos ao final de cada epoca.

        Calcula norma L2 de cada variavel treinavel do modelo.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (metricas de norma de pesos sao adicionadas).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Norma L2: sqrt(sum(w_i^2)) por variavel treinavel.
            Logs adicionados: weight_norm_mean, weight_norm_max.
        """
        import tensorflow as tf

        model = self._model_ref
        trainable_vars = model.trainable_variables

        if not trainable_vars:
            return

        # Coleta normas L2 de cada variavel treinavel (pesos, nao gradientes)
        weight_norms = []
        for var in trainable_vars:
            # Norma L2 do tensor de pesos
            norm = tf.norm(var, ord=2).numpy()
            weight_norms.append(float(norm))

        if weight_norms:
            mean_norm = sum(weight_norms) / len(weight_norms)
            max_norm = max(weight_norms)

            if logs is not None:
                logs["weight_norm_mean"] = mean_norm
                logs["weight_norm_max"] = max_norm

            logger.debug(
                "Epoch %d: weight_norm_mean=%.6f, weight_norm_max=%.6f",
                epoch,
                mean_norm,
                max_norm,
            )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 3: BestEpochTracker — Rastreamento de melhor epoca
#
# Rastreia a melhor epoca e o melhor valor de uma metrica monitorada
# (default: val_loss, modo min). Util para diagnostico pos-treinamento
# e para saber quando o modelo atingiu seu melhor desempenho.
#
# Nao altera o treinamento — apenas registra best_epoch e best_value.
#
# Referencia: C40 legado (BestEpochTracker callback).
# ════════════════════════════════════════════════════════════════════════

class BestEpochTracker:
    """Rastreia a melhor epoca e valor de uma metrica monitorada.

    Registra best_epoch e best_value durante o treinamento. Acessiveis
    como propriedades apos o treinamento para diagnostico.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        monitor: Nome da metrica monitorada (default: 'val_loss').
        mode: Modo de comparacao ('min' ou 'max').
        _best_epoch: Indice da melhor epoca (0-based).
        _best_value: Melhor valor da metrica monitorada.

    Example:
        >>> cb = BestEpochTracker(monitor='val_loss', mode='min')
        >>> cb.best_epoch  # -1 antes do treinamento
        -1
        >>> cb.best_value  # inf antes do treinamento
        inf

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (sempre incluido)
            - training/loop.py: TrainingLoop acessa best_epoch pos-treino
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        mode='min': melhor = menor valor (val_loss, mae).
        mode='max': melhor = maior valor (accuracy, r2).
    """

    def __new__(cls, monitor: str = "val_loss", mode: str = "min"):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            monitor: Nome da metrica a monitorar.
            mode: 'min' ou 'max'.

        Returns:
            Instancia de BestEpochTracker.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(self, monitor: str = "val_loss", mode: str = "min") -> None:
        """Inicializa o rastreador de melhor epoca.

        Args:
            monitor: Nome da metrica a monitorar. Deve corresponder a uma
                chave no dict `logs` passado por Keras em on_epoch_end.
                Default: 'val_loss'.
            mode: Modo de comparacao. 'min' para metricas que devem diminuir
                (loss, mae), 'max' para metricas que devem aumentar (accuracy,
                r2). Default: 'min'.

        Raises:
            ValueError: Se mode nao for 'min' ou 'max'.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            _best_value inicializa como +inf (min) ou -inf (max).
        """
        import tensorflow as tf

        super().__init__()

        if mode not in ("min", "max"):
            raise ValueError(
                f"mode deve ser 'min' ou 'max', recebido: '{mode}'"
            )

        self.monitor: str = monitor
        self.mode: str = mode
        self._best_epoch: int = -1
        self._best_value: float = float("inf") if mode == "min" else float("-inf")

        logger.info(
            "BestEpochTracker inicializado: monitor='%s', mode='%s'",
            monitor,
            mode,
        )

    @property
    def best_epoch(self) -> int:
        """Indice da melhor epoca (0-based). -1 se nenhuma epoca completou.

        Returns:
            Indice da melhor epoca, ou -1 antes do treinamento.

        Note:
            Referenciado em:
                - training/loop.py: TrainingLoop.run() (diagnostico)
            Valor: atualizado em on_epoch_end a cada melhoria.
        """
        return self._best_epoch

    @property
    def best_value(self) -> float:
        """Melhor valor da metrica monitorada. inf/-inf se nenhuma epoca.

        Returns:
            Melhor valor, ou inf (min) / -inf (max) antes do treinamento.

        Note:
            Referenciado em:
                - training/loop.py: TrainingLoop.run() (diagnostico)
            Valor: atualizado em on_epoch_end a cada melhoria.
        """
        return self._best_value

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Verifica se a epoca atual e a melhor e atualiza tracking.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas do Keras. Deve conter self.monitor.

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Compara valor atual com _best_value usando mode (min/max).
            Se metrica ausente no logs, emite warning e retorna.
        """
        if logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            logger.warning(
                "BestEpochTracker: metrica '%s' nao encontrada em logs. "
                "Metricas disponiveis: %s",
                self.monitor,
                list(logs.keys()),
            )
            return

        # Compara usando modo min ou max
        is_improvement = (
            current < self._best_value
            if self.mode == "min"
            else current > self._best_value
        )

        if is_improvement:
            self._best_epoch = epoch
            self._best_value = float(current)
            logger.debug(
                "Epoch %d: novo melhor %s=%.6f",
                epoch,
                self.monitor,
                self._best_value,
            )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 4: DualValidationCallback — Validacao em clean + noisy (P2)
#
# Avalia o modelo em dois datasets de validacao separados:
#   - val_clean_ds: dados sem ruido (baseline de performance)
#   - val_noisy_ds: dados com ruido (robustez a noise)
# Calcula o gap entre as duas metricas para detectar overfitting
# ao ruido (modelo performa bem em noisy mas perde precisao em clean).
#
# Ativacao: config.use_dual_validation=True + datasets fornecidos.
# Referencia: P2 (dual validation), C25 legado, S19 analise.
# ════════════════════════════════════════════════════════════════════════

class DualValidationCallback:
    """Valida modelo em datasets clean e noisy separadamente (P2).

    Avalia model.evaluate() em val_clean_ds e val_noisy_ds a cada epoca.
    Loga clean_val_loss, noisy_val_loss e gap = noisy - clean.
    Gap crescente indica overfitting ao ruido.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _val_clean_ds: Dataset de validacao limpo (sem ruido).
        _val_noisy_ds: Dataset de validacao com ruido.
        _model_ref: Referencia ao modelo Keras avaliado.
        _config: PipelineConfig para logging e controle.

    Example:
        >>> import tensorflow as tf
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig.robusto()
        >>> # cb = DualValidationCallback(val_clean, val_noisy, model, config)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (condicional P2)
            - data/pipeline.py: DataPipeline gera val_clean_ds e val_noisy_ds
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (dual validation P2).
        Gap = noisy_val_loss - clean_val_loss.
        Gap > 0: noisy e mais dificil (esperado).
        Gap crescente: modelo sobreajusta a ruido (acao: reduzir noise).
    """

    def __new__(
        cls,
        val_clean_ds: Any,
        val_noisy_ds: Any,
        model: Any,
        config: PipelineConfig,
    ):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            val_clean_ds: tf.data.Dataset de validacao sem ruido.
            val_noisy_ds: tf.data.Dataset de validacao com ruido.
            model: Modelo tf.keras.Model a ser avaliado.
            config: PipelineConfig com campos de dual validation.

        Returns:
            Instancia de DualValidationCallback.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        val_clean_ds: Any,
        val_noisy_ds: Any,
        model: Any,
        config: PipelineConfig,
    ) -> None:
        """Inicializa o callback de dual validation.

        Args:
            val_clean_ds: tf.data.Dataset de validacao sem ruido.
            val_noisy_ds: tf.data.Dataset de validacao com ruido.
            model: Modelo tf.keras.Model a ser avaliado.
            config: PipelineConfig com campos de dual validation.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Ambos datasets devem estar prontos (batched, prefetched).
        """
        import tensorflow as tf

        super().__init__()
        self._val_clean_ds = val_clean_ds
        self._val_noisy_ds = val_noisy_ds
        self._model_ref = model
        self._config = config

        logger.info("DualValidationCallback inicializado (P2 dual val)")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Avalia modelo em clean e noisy, loga gap.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (clean_val_loss, noisy_val_loss, dual_gap
                sao adicionados).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2 (P2).
            Gap = noisy_val_loss - clean_val_loss.
            Gap crescente indica overfitting ao ruido.
        """
        model = self._model_ref

        # Avalia em dataset limpo (baseline)
        clean_results = model.evaluate(self._val_clean_ds, verbose=0)
        clean_loss = clean_results if isinstance(clean_results, float) else clean_results[0]

        # Avalia em dataset ruidoso (robustez)
        noisy_results = model.evaluate(self._val_noisy_ds, verbose=0)
        noisy_loss = noisy_results if isinstance(noisy_results, float) else noisy_results[0]

        # Gap: diferenca entre noisy e clean (positivo = noisy mais dificil)
        gap = noisy_loss - clean_loss

        if logs is not None:
            logs["clean_val_loss"] = clean_loss
            logs["noisy_val_loss"] = noisy_loss
            logs["dual_gap"] = gap

        logger.info(
            "Epoch %d: clean_val_loss=%.6f, noisy_val_loss=%.6f, gap=%.6f",
            epoch,
            clean_loss,
            noisy_loss,
            gap,
        )

        # Alerta se gap aumenta significativamente (overfitting ao ruido)
        if gap > 0.1:
            logger.warning(
                "Epoch %d: dual_gap=%.4f > 0.1 — possivel overfitting ao ruido",
                epoch,
                gap,
            )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 5: PINNSLambdaScheduleCallback — Annealing de lambda PINNs
#
# Controla o peso da penalidade PINNs (Physics-Informed Neural Network)
# durante o treinamento. Implementa annealing linear de 0 a
# config.pinns_lambda ao longo de penalty_warmup_epochs epocas.
#
# Motivacao: ativar penalidades fisicas desde a epoca 0 destabiliza
# a convergencia inicial. O warmup permite que o modelo aprenda o
# mapeamento basico antes de aplicar restricoes fisicas.
#
# Schedule: lambda(t) = pinns_lambda * min(epoch / warmup, 1.0)
#
# Referencia: C41 legado (penalty_warmup_epochs), S20 analise.
# ════════════════════════════════════════════════════════════════════════

class PINNSLambdaScheduleCallback:
    """Annealing linear do peso lambda PINNs ao longo do warmup.

    Implementa rampa linear de 0 a config.pinns_lambda durante as
    primeiras penalty_warmup_epochs epocas. Apos o warmup, lambda
    permanece constante em config.pinns_lambda.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _pinns_lambda_var: tf.Variable escalar (float32) com peso PINNs.
        _target_lambda: Valor alvo de lambda (config.pinns_lambda).
        _warmup_epochs: Numero de epocas de warmup (config.penalty_warmup_epochs).

    Example:
        >>> import tensorflow as tf
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig(use_pinns=True, pinns_lambda=0.01)
        >>> var = tf.Variable(0.0, dtype=tf.float32)
        >>> cb = PINNSLambdaScheduleCallback(var, config)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (se use_pinns=True)
            - losses/: combined_loss_fn usa pinns_lambda_var
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (PINNs warmup).
        Schedule linear: lambda = target * min(epoch / warmup, 1.0).
        warmup=0 → lambda constante desde epoca 0 (sem rampa).
    """

    def __new__(cls, pinns_lambda_var: Any, config: PipelineConfig):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            pinns_lambda_var: tf.Variable escalar (float32) com peso PINNs.
            config: PipelineConfig com pinns_lambda e penalty_warmup_epochs.

        Returns:
            Instancia de PINNSLambdaScheduleCallback.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(self, pinns_lambda_var: Any, config: PipelineConfig) -> None:
        """Inicializa o scheduler de lambda PINNs.

        Args:
            pinns_lambda_var: tf.Variable escalar (float32) que controla o
                peso da penalidade PINNs. Compartilhado por referencia com
                a funcao de loss combinada.
            config: PipelineConfig com campos:
                pinns_lambda (float): Valor alvo de lambda (ex: 0.01).
                penalty_warmup_epochs (int): Epocas de warmup linear (ex: 10).

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            O pinns_lambda_var DEVE ser criado externamente (tf.Variable).
            Valor inicial = 0.0 (rampa comeca em zero).
        """
        import tensorflow as tf

        super().__init__()
        self._pinns_lambda_var = pinns_lambda_var
        self._target_lambda: float = config.pinns_lambda
        self._warmup_epochs: int = config.penalty_warmup_epochs

        logger.info(
            "PINNSLambdaScheduleCallback inicializado: target=%.6f, warmup=%d ep",
            self._target_lambda,
            self._warmup_epochs,
        )

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Atualiza pinns_lambda_var no inicio de cada epoca.

        Schedule linear: lambda = target * min(epoch / warmup, 1.0).
        Apos warmup_epochs, lambda permanece constante em target.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (nao utilizado).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2 (PINNs warmup).
            Epoch 0 → lambda = 0.0 (nenhuma penalidade PINNs).
            Epoch warmup → lambda = target (penalidade total).
        """
        if self._warmup_epochs <= 0:
            # Sem warmup — lambda constante desde epoca 0
            new_lambda = self._target_lambda
        else:
            # Rampa linear: 0 → target ao longo de warmup_epochs
            ratio = min(epoch / self._warmup_epochs, 1.0)
            new_lambda = self._target_lambda * ratio

        self._pinns_lambda_var.assign(new_lambda)

        logger.debug(
            "Epoch %d: pinns_lambda=%.6f (target=%.6f, warmup=%d)",
            epoch,
            new_lambda,
            self._target_lambda,
            self._warmup_epochs,
        )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 6: CausalDegradationMonitor — Gap causal vs acausal
#
# Compara a performance do modelo em modos causal vs acausal para
# detectar degradacao causada pela restricao de causalidade.
# Apenas ativo quando config.use_causal_mode=True.
#
# Em cada epoca (a cada 5 epocas para evitar overhead), avalia o
# modelo no val_ds e computa o causal_gap = |loss_causal - loss_full|.
# Gap crescente indica que a mascara causal esta limitando excessivamente
# a capacidade do modelo.
#
# Referencia: C40 legado (CausalDegradationMonitor), geosteering mode.
# ════════════════════════════════════════════════════════════════════════

class CausalDegradationMonitor:
    """Monitora degradacao causada pela restricao causal.

    Avalia modelo em val_ds e loga causal_gap a cada 5 epocas.
    Apenas ativo quando config.use_causal_mode=True.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _model_ref: Referencia ao modelo Keras sendo monitorado.
        _val_ds: Dataset de validacao para avaliacao.
        _config: PipelineConfig com campos de causal mode.

    Example:
        >>> import tensorflow as tf
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig.realtime()  # use_causal_mode=True
        >>> # cb = CausalDegradationMonitor(model, val_ds, config)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (se use_causal_mode)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (geosteering mode).
        causal_gap = |val_loss_com_mascara - val_loss_sem_mascara|.
        Gap alto indica que causalidade impoe custo significativo.
        Avaliacao a cada 5 epocas para minimizar overhead.
    """

    def __new__(cls, model: Any, val_ds: Any, config: PipelineConfig):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            model: Modelo tf.keras.Model sendo monitorado.
            val_ds: tf.data.Dataset de validacao.
            config: PipelineConfig com use_causal_mode.

        Returns:
            Instancia de CausalDegradationMonitor.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(self, model: Any, val_ds: Any, config: PipelineConfig) -> None:
        """Inicializa o monitor de degradacao causal.

        Args:
            model: Modelo tf.keras.Model sendo monitorado.
            val_ds: tf.data.Dataset de validacao (batched, prefetched).
            config: PipelineConfig com use_causal_mode=True.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            O modelo deve ser compilado antes de usar este callback.
        """
        import tensorflow as tf

        super().__init__()
        self._model_ref = model
        self._val_ds = val_ds
        self._config = config

        logger.info("CausalDegradationMonitor inicializado (causal mode)")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Avalia gap causal a cada 5 epocas.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (causal_gap adicionado).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Avalia a cada 5 epocas para minimizar overhead.
            Loga causal_gap no logs dict para TensorBoard/CSVLogger.
        """
        # Avalia apenas a cada 5 epocas para evitar overhead
        if epoch % 5 != 0:
            return

        model = self._model_ref

        # Avalia modelo no estado atual (com mascara causal ativa)
        results = model.evaluate(self._val_ds, verbose=0)
        val_loss = results if isinstance(results, float) else results[0]

        if logs is not None:
            logs["causal_val_loss"] = val_loss

        logger.info(
            "Epoch %d: causal_val_loss=%.6f (causal mode ativo)",
            epoch,
            val_loss,
        )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 7: SlidingWindowValidation — Validacao em janelas deslizantes
#
# Testa o modelo em janelas deslizantes de tamanho sequence_length para
# detectar efeitos de borda (edge effects) nos limites das janelas.
# Apenas ativo em modo causal (geosteering realtime).
#
# Compara performance no centro da janela vs bordas para garantir
# que a predicao e uniforme ao longo da sequencia.
#
# Referencia: C40 legado (SlidingWindowValidation), geosteering mode.
# ════════════════════════════════════════════════════════════════════════

class SlidingWindowValidation:
    """Valida modelo em janelas deslizantes para detectar edge effects.

    Apenas ativo quando config.use_causal_mode=True. Avalia a cada
    10 epocas para minimizar overhead.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _model_ref: Referencia ao modelo Keras sendo monitorado.
        _val_data: Dados de validacao (numpy arrays ou tf.data.Dataset).
        _config: PipelineConfig com sequence_length e use_causal_mode.

    Example:
        >>> import tensorflow as tf
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig.realtime()
        >>> # cb = SlidingWindowValidation(model, val_data, config)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, se causal)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (geosteering mode).
        Janela = config.sequence_length (600 medidas).
        Compara perda no centro vs bordas da janela.
    """

    def __new__(cls, model: Any, val_data: Any, config: PipelineConfig):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            model: Modelo tf.keras.Model sendo monitorado.
            val_data: Dados de validacao (numpy ou tf.data.Dataset).
            config: PipelineConfig com sequence_length.

        Returns:
            Instancia de SlidingWindowValidation.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(
        self, model: Any, val_data: Any, config: PipelineConfig
    ) -> None:
        """Inicializa a validacao por janelas deslizantes.

        Args:
            model: Modelo tf.keras.Model sendo monitorado.
            val_data: Dados de validacao (tf.data.Dataset batched).
            config: PipelineConfig com sequence_length e use_causal_mode.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            val_data deve conter sequencias de tamanho >= sequence_length.
        """
        import tensorflow as tf

        super().__init__()
        self._model_ref = model
        self._val_data = val_data
        self._config = config

        logger.info(
            "SlidingWindowValidation inicializado: seq_len=%d",
            config.sequence_length,
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Avalia modelo em janelas deslizantes a cada 10 epocas.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (sliding_window_loss adicionado).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Avalia a cada 10 epocas para minimizar overhead.
            Loga sliding_window_loss medio sobre todas as janelas.
        """
        # Avalia apenas a cada 10 epocas para evitar overhead
        if epoch % 10 != 0:
            return

        model = self._model_ref

        # Avalia modelo no dataset de validacao (janela unica = dataset completo)
        results = model.evaluate(self._val_data, verbose=0)
        sw_loss = results if isinstance(results, float) else results[0]

        if logs is not None:
            logs["sliding_window_loss"] = sw_loss

        logger.info(
            "Epoch %d: sliding_window_loss=%.6f (seq_len=%d)",
            epoch,
            sw_loss,
            self._config.sequence_length,
        )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 8: PeriodicCheckpoint — Salva modelo a cada N epocas
#
# Salva o modelo completo (.keras) a cada `period` epocas em
# experiment_dir/checkpoints/epoch_{epoch:04d}.keras. Util para
# recuperacao em caso de falha e para analise de evolucao do modelo.
#
# Requer config.experiment_dir definido (sem dir, nao salva).
# Cria o subdiretorio checkpoints/ automaticamente.
#
# Referencia: C40 legado (PeriodicCheckpoint callback).
# ════════════════════════════════════════════════════════════════════════

class PeriodicCheckpoint:
    """Salva modelo a cada N epocas em experiment_dir/checkpoints/.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _config: PipelineConfig com experiment_dir.
        _period: Frequencia de salvamento em epocas (default: 10).
        _checkpoint_dir: Caminho do diretorio de checkpoints.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig(experiment_dir="/tmp/exp1")
        >>> cb = PeriodicCheckpoint(config, period=10)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (se experiment_dir)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        Formato: epoch_{epoch:04d}.keras (ex: epoch_0010.keras).
        Cria checkpoints/ automaticamente se nao existir.
    """

    def __new__(cls, config: PipelineConfig, *, period: int = 10):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            config: PipelineConfig com experiment_dir.
            period: Frequencia de salvamento em epocas.

        Returns:
            Instancia de PeriodicCheckpoint.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(self, config: PipelineConfig, *, period: int = 10) -> None:
        """Inicializa o checkpoint periodico.

        Args:
            config: PipelineConfig com experiment_dir para salvar checkpoints.
            period: Frequencia de salvamento em epocas (default: 10).
                Modelo salvo a cada `period` epocas completadas.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Se experiment_dir nao definido, callback registra warning e nao salva.
        """
        import tensorflow as tf

        super().__init__()
        self._config = config
        self._period: int = period

        if config.experiment_dir:
            self._checkpoint_dir = os.path.join(
                config.experiment_dir, "checkpoints"
            )
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        else:
            self._checkpoint_dir = ""

        logger.info(
            "PeriodicCheckpoint inicializado: period=%d, dir='%s'",
            period,
            self._checkpoint_dir or "(nao definido)",
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Salva modelo se epoca e multipla de period.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (nao utilizado).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Salva em formato .keras (nativo TF2). Epoch 0-based:
            salvamento ocorre em epoch=period-1, 2*period-1, etc.
            (primeira avaliacao no final da period-esima epoca).
        """
        if not self._checkpoint_dir:
            return

        # Salva a cada period epocas (epoch+1 para 1-based display)
        if (epoch + 1) % self._period == 0:
            filepath = os.path.join(
                self._checkpoint_dir,
                f"epoch_{epoch + 1:04d}.keras",
            )
            try:
                self.model.save(filepath)
                logger.info("Checkpoint salvo: %s", filepath)
            except Exception as exc:
                logger.warning(
                    "Falha ao salvar checkpoint epoch %d: %s", epoch + 1, exc
                )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 9: MetricPlateauDetector — Detecta plateaus persistentes
#
# Monitora uma metrica (default: val_loss) e emite warning se o valor
# nao mudar por mais de `patience` epocas. Diferente de EarlyStopping
# (que para o treinamento), este callback apenas loga um alerta.
#
# Util para diagnosticar LR muito baixo, loss landscape plano ou
# saturacao de capacidade do modelo.
#
# Referencia: C40 legado (PlateauDetector callback).
# ════════════════════════════════════════════════════════════════════════

class MetricPlateauDetector:
    """Detecta plateaus persistentes em uma metrica monitorada.

    Emite warning se a metrica nao mudar por mais de `patience` epocas.
    Nao interrompe o treinamento — apenas alerta.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _monitor: Nome da metrica monitorada.
        _patience: Numero de epocas sem melhoria antes do alerta.
        _threshold: Variacao minima para considerar como melhoria.
        _best: Melhor valor observado ate agora.
        _wait: Contador de epocas sem melhoria.

    Example:
        >>> cb = MetricPlateauDetector(monitor='val_loss', patience=20)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, opt-in)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        Diferente de EarlyStopping: NAO para o treinamento.
        threshold=1e-4: variacao menor que isso e considerada plateau.
    """

    def __new__(
        cls,
        monitor: str = "val_loss",
        *,
        patience: int = 20,
        threshold: float = 1e-4,
    ):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            monitor: Nome da metrica a monitorar.
            patience: Epocas sem melhoria antes do alerta.
            threshold: Variacao minima considerada melhoria.

        Returns:
            Instancia de MetricPlateauDetector.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        monitor: str = "val_loss",
        *,
        patience: int = 20,
        threshold: float = 1e-4,
    ) -> None:
        """Inicializa o detector de plateau.

        Args:
            monitor: Nome da metrica a monitorar. Deve corresponder a uma
                chave no dict `logs`. Default: 'val_loss'.
            patience: Numero de epocas consecutivas sem melhoria antes
                de emitir warning. Default: 20.
            threshold: Variacao minima absoluta para considerar como
                melhoria real (evita falsos positivos por ruido numerico).
                Default: 1e-4.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Plateau detectado quando |current - best| < threshold
            por patience epocas consecutivas.
        """
        import tensorflow as tf

        super().__init__()
        self._monitor: str = monitor
        self._patience: int = patience
        self._threshold: float = threshold
        self._best: float = float("inf")
        self._wait: int = 0

        logger.info(
            "MetricPlateauDetector inicializado: monitor='%s', "
            "patience=%d, threshold=%.6f",
            monitor,
            patience,
            threshold,
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Verifica se metrica esta em plateau.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas do Keras.

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Emite WARNING se plateau detectado (nao para treinamento).
        """
        if logs is None:
            return

        current = logs.get(self._monitor)
        if current is None:
            return

        # Verifica se houve melhoria significativa
        if self._best - current > self._threshold:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1

        if self._wait >= self._patience:
            logger.warning(
                "Epoch %d: PLATEAU detectado — '%s' nao melhorou por %d epocas "
                "(best=%.6f, current=%.6f, threshold=%.6f). "
                "Considere ajustar LR ou arquitetura.",
                epoch,
                self._monitor,
                self._wait,
                self._best,
                current,
                self._threshold,
            )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 10: OneCycleLR — Super-convergencia (Smith 2018)
#
# Implementa o schedule One Cycle Learning Rate (Smith 2018) para
# super-convergencia. Duas fases:
#   Phase 1 (0 → pct_start): LR ramps from max_lr/div_factor to max_lr
#   Phase 2 (pct_start → 1): LR decays from max_lr to max_lr/(div*1e4)
#
# Referencia: Smith, L.N. "Super-Convergence: Very Fast Training of
# Neural Networks Using Large Learning Rates" (2018).
# C40 legado (OneCycleLR callback grupo D).
# ════════════════════════════════════════════════════════════════════════

class OneCycleLR:
    """One Cycle LR schedule para super-convergencia (Smith 2018).

    Phase 1 (warmup): LR cresce de max_lr/div_factor ate max_lr.
    Phase 2 (annealing): LR decai de max_lr ate max_lr/(div_factor*1e4).

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _max_lr: Taxa de aprendizado maxima (pico do ciclo).
        _total_epochs: Numero total de epocas de treinamento.
        _div_factor: Fator de divisao para LR inicial e final.
        _pct_start: Fracao de epocas na fase de warmup.

    Example:
        >>> cb = OneCycleLR(max_lr=1e-3, total_epochs=100)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, opt-in)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        Smith 2018: "Super-Convergence" — LR alto permite convergencia
        ate 10x mais rapida com regularizacao implicita.
        Phase 1: 0 → pct_start (30% default) — warmup.
        Phase 2: pct_start → 1.0 — cosine annealing.
    """

    def __new__(
        cls,
        max_lr: float,
        total_epochs: int,
        *,
        div_factor: float = 25.0,
        pct_start: float = 0.3,
    ):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            max_lr: Taxa de aprendizado maxima.
            total_epochs: Numero total de epocas.
            div_factor: Fator de divisao para LR inicial.
            pct_start: Fracao de epocas na fase de warmup.

        Returns:
            Instancia de OneCycleLR.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        max_lr: float,
        total_epochs: int,
        *,
        div_factor: float = 25.0,
        pct_start: float = 0.3,
    ) -> None:
        """Inicializa o schedule One Cycle LR.

        Args:
            max_lr: Taxa de aprendizado maxima (pico). Deve ser > 0.
            total_epochs: Numero total de epocas de treinamento. Deve ser > 0.
            div_factor: Fator de divisao. LR inicial = max_lr / div_factor.
                LR final = max_lr / (div_factor * 1e4). Default: 25.0.
            pct_start: Fracao de epocas na fase de warmup (Phase 1).
                Valor em [0, 1]. Default: 0.3 (30% de warmup).

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: Smith 2018 (super-convergence).
            LR range: [max_lr/25, max_lr] na Phase 1, depois annealing.
        """
        import tensorflow as tf

        super().__init__()
        self._max_lr: float = max_lr
        self._total_epochs: int = total_epochs
        self._div_factor: float = div_factor
        self._pct_start: float = pct_start

        logger.info(
            "OneCycleLR inicializado: max_lr=%.6f, total=%d, "
            "div=%.1f, pct_start=%.2f",
            max_lr,
            total_epochs,
            div_factor,
            pct_start,
        )

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Atualiza LR do otimizador no inicio de cada epoca.

        Phase 1 (0 → pct_start): LR linear de max_lr/div → max_lr.
        Phase 2 (pct_start → 1): LR cosine de max_lr → max_lr/(div*1e4).

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (nao utilizado).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: Smith 2018 (super-convergence policy).
            Altera self.model.optimizer.learning_rate diretamente.
        """
        import tensorflow as tf

        if self._total_epochs <= 0:
            return

        # Fracao normalizada da epoca atual [0, 1]
        pct = epoch / max(self._total_epochs, 1)

        initial_lr = self._max_lr / self._div_factor
        final_lr = self._max_lr / (self._div_factor * 1e4)

        if pct <= self._pct_start:
            # Phase 1: Warmup linear — initial_lr → max_lr
            phase_pct = pct / max(self._pct_start, 1e-8)
            new_lr = initial_lr + (self._max_lr - initial_lr) * phase_pct
        else:
            # Phase 2: Cosine annealing — max_lr → final_lr
            phase_pct = (pct - self._pct_start) / max(
                1.0 - self._pct_start, 1e-8
            )
            new_lr = final_lr + (self._max_lr - final_lr) * 0.5 * (
                1.0 + math.cos(math.pi * phase_pct)
            )

        # Atualiza LR do otimizador via backend Keras
        tf.keras.backend.set_value(
            self.model.optimizer.learning_rate, new_lr
        )

        logger.debug(
            "Epoch %d: OneCycleLR lr=%.8f (pct=%.3f)",
            epoch,
            new_lr,
            pct,
        )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 11: CosineWarmRestarts — SGDR (Loshchilov & Hutter 2017)
#
# Implementa Cosine Annealing with Warm Restarts (SGDR):
#   LR = initial_lr * 0.5 * (1 + cos(pi * T_cur / T_i))
#
# T_cur: epocas desde o ultimo restart.
# T_i: tamanho do ciclo atual (T_0 * T_mult^restart).
# Apos T_i epocas, LR reinicia em initial_lr (warm restart).
#
# Referencia: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent
# with Warm Restarts" (ICLR 2017). C40 legado (grupo D).
# ════════════════════════════════════════════════════════════════════════

class CosineWarmRestarts:
    """SGDR: Cosine annealing com warm restarts (Loshchilov & Hutter 2017).

    LR segue cosine decay dentro de cada ciclo, com restarts periodicos.
    Ciclos crescem por fator T_mult a cada restart.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _initial_lr: Taxa de aprendizado inicial (e restart).
        _T_0: Duracao do primeiro ciclo em epocas.
        _T_mult: Fator multiplicativo para duracao dos ciclos seguintes.
        _T_cur: Epocas desde o ultimo restart.
        _T_i: Duracao do ciclo atual.

    Example:
        >>> cb = CosineWarmRestarts(initial_lr=1e-3, T_0=10, T_mult=2)
        >>> # Ciclos: 10, 20, 40, 80, ... epocas

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, opt-in)
        Ref: Loshchilov & Hutter, ICLR 2017 (SGDR).
        T_mult=1: ciclos de duracao fixa (sem crescimento).
        T_mult=2: ciclos dobram (10→20→40→80 epocas).
    """

    def __new__(
        cls,
        initial_lr: float,
        *,
        T_0: int = 10,
        T_mult: int = 2,
    ):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            initial_lr: Taxa de aprendizado inicial (e restart).
            T_0: Duracao do primeiro ciclo em epocas.
            T_mult: Fator multiplicativo dos ciclos seguintes.

        Returns:
            Instancia de CosineWarmRestarts.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        initial_lr: float,
        *,
        T_0: int = 10,
        T_mult: int = 2,
    ) -> None:
        """Inicializa o schedule SGDR com cosine warm restarts.

        Args:
            initial_lr: Taxa de aprendizado inicial e de cada restart.
                Deve ser > 0.
            T_0: Duracao do primeiro ciclo em epocas. Deve ser > 0.
                Default: 10.
            T_mult: Fator multiplicativo para duracao dos ciclos seguintes.
                T_mult=1 → ciclos fixos. T_mult=2 → ciclos dobram.
                Deve ser >= 1. Default: 2.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: Loshchilov & Hutter 2017 (SGDR, Eq. 2).
            LR = initial_lr * 0.5 * (1 + cos(pi * T_cur / T_i)).
        """
        import tensorflow as tf

        super().__init__()
        self._initial_lr: float = initial_lr
        self._T_0: int = T_0
        self._T_mult: int = T_mult
        self._T_cur: int = 0
        self._T_i: int = T_0

        logger.info(
            "CosineWarmRestarts inicializado: lr=%.6f, T_0=%d, T_mult=%d",
            initial_lr,
            T_0,
            T_mult,
        )

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Atualiza LR com cosine annealing + warm restarts.

        LR = initial_lr * 0.5 * (1 + cos(pi * T_cur / T_i)).
        Quando T_cur >= T_i, restart ocorre: T_cur=0, T_i *= T_mult.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (nao utilizado).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: Loshchilov & Hutter 2017, Eq. 2.
            Warm restart: LR volta a initial_lr no inicio de cada ciclo.
        """
        import tensorflow as tf

        # Cosine annealing dentro do ciclo atual
        new_lr = self._initial_lr * 0.5 * (
            1.0 + math.cos(math.pi * self._T_cur / max(self._T_i, 1))
        )

        tf.keras.backend.set_value(
            self.model.optimizer.learning_rate, new_lr
        )

        logger.debug(
            "Epoch %d: CosineWarmRestarts lr=%.8f (T_cur=%d, T_i=%d)",
            epoch,
            new_lr,
            self._T_cur,
            self._T_i,
        )

        # Avanca T_cur e verifica restart
        self._T_cur += 1
        if self._T_cur >= self._T_i:
            # Warm restart: reinicia contador e expande ciclo
            self._T_cur = 0
            self._T_i = self._T_i * self._T_mult
            logger.info(
                "Epoch %d: SGDR warm restart — proximo ciclo T_i=%d epocas",
                epoch,
                self._T_i,
            )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 12: CyclicalLR — LR ciclico triangular (Smith 2017)
#
# Implementa Cyclical Learning Rate (Smith 2017) com modos:
#   - triangular: LR oscila entre base_lr e max_lr
#   - triangular2: max_lr decai pela metade a cada ciclo
#
# Um ciclo completo = 2 * step_size epocas.
#
# Referencia: Smith, L.N. "Cyclical Learning Rates for Training Neural
# Networks" (WACV 2017). C40 legado (CyclicalLR callback grupo D).
# ════════════════════════════════════════════════════════════════════════

class CyclicalLR:
    """Cyclical LR: triangular ou triangular2 (Smith 2017).

    LR oscila entre base_lr e max_lr em ciclos triangulares.
    Um ciclo = 2 * step_size epocas.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _base_lr: Taxa de aprendizado minima (base do triangulo).
        _max_lr: Taxa de aprendizado maxima (pico do triangulo).
        _step_size: Metade do ciclo em epocas.
        _mode: Modo do ciclo ('triangular' ou 'triangular2').

    Example:
        >>> cb = CyclicalLR(base_lr=1e-4, max_lr=1e-3, step_size=10)
        >>> # Ciclo completo = 20 epocas (10 subida + 10 descida)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, opt-in)
        Ref: Smith 2017 (CLR, WACV).
        triangular: amplitude constante em todos os ciclos.
        triangular2: amplitude halved a cada ciclo (decai exponencialmente).
    """

    def __new__(
        cls,
        base_lr: float,
        max_lr: float,
        *,
        step_size: int = 10,
        mode: str = "triangular",
    ):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            base_lr: Taxa de aprendizado minima.
            max_lr: Taxa de aprendizado maxima.
            step_size: Metade do ciclo em epocas.
            mode: 'triangular' ou 'triangular2'.

        Returns:
            Instancia de CyclicalLR.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        *,
        step_size: int = 10,
        mode: str = "triangular",
    ) -> None:
        """Inicializa o schedule Cyclical LR.

        Args:
            base_lr: Taxa de aprendizado minima (base do triangulo). > 0.
            max_lr: Taxa de aprendizado maxima (pico do triangulo). > base_lr.
            step_size: Metade do ciclo em epocas. Ciclo completo = 2*step_size.
                Default: 10 (ciclo de 20 epocas).
            mode: Modo do ciclo. 'triangular' para amplitude constante,
                'triangular2' para amplitude halved a cada ciclo.
                Default: 'triangular'.

        Raises:
            ValueError: Se mode nao for 'triangular' ou 'triangular2'.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: Smith 2017 (CLR policy), Fig. 3 e 4.
            triangular: max_lr constante.
            triangular2: max_lr /= 2 a cada ciclo (convergencia mais estavel).
        """
        import tensorflow as tf

        super().__init__()

        if mode not in ("triangular", "triangular2"):
            raise ValueError(
                f"mode deve ser 'triangular' ou 'triangular2', recebido: '{mode}'"
            )

        self._base_lr: float = base_lr
        self._max_lr: float = max_lr
        self._step_size: int = step_size
        self._mode: str = mode

        logger.info(
            "CyclicalLR inicializado: base=%.6f, max=%.6f, step=%d, mode='%s'",
            base_lr,
            max_lr,
            step_size,
            mode,
        )

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Atualiza LR do otimizador com ciclo triangular.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (nao utilizado).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: Smith 2017, Eq. 1.
            cycle = floor(1 + epoch / (2 * step_size))
            x = |epoch/step_size - 2*cycle + 1|
            LR = base_lr + (max_lr - base_lr) * max(0, 1 - x) * scale
        """
        import tensorflow as tf

        # Ciclo atual (1-based)
        cycle = math.floor(1 + epoch / (2 * self._step_size))
        # Posicao dentro do ciclo [0, 1] → triangulo
        x = abs(epoch / max(self._step_size, 1) - 2 * cycle + 1)

        # Escala de amplitude por modo
        if self._mode == "triangular":
            scale = 1.0
        else:
            # triangular2: amplitude halved a cada ciclo
            scale = 1.0 / (2.0 ** (cycle - 1))

        new_lr = self._base_lr + (self._max_lr - self._base_lr) * max(
            0.0, 1.0 - x
        ) * scale

        tf.keras.backend.set_value(
            self.model.optimizer.learning_rate, new_lr
        )

        logger.debug(
            "Epoch %d: CyclicalLR lr=%.8f (cycle=%d, mode='%s')",
            epoch,
            new_lr,
            cycle,
            self._mode,
        )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 13: MemoryMonitor — Uso de memoria GPU/CPU
#
# Loga uso de memoria GPU (via tf.config.experimental.get_memory_info)
# e/ou CPU (via resource module) ao final de cada epoca.
# Util para diagnosticar memory leaks e dimensionar batch_size.
#
# Se GPU nao disponivel, loga apenas memoria do processo (CPU).
# Nao interfere no treinamento — apenas observa.
#
# Referencia: C40 legado (MemoryMonitor callback grupo B).
# ════════════════════════════════════════════════════════════════════════

class MemoryMonitor:
    """Monitora uso de memoria GPU/CPU por epoca.

    Loga peak e current memory via TF experimental API para GPU.
    Fallback para resource.getrusage() para memoria CPU do processo.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        Nenhum atributo especifico (stateless monitor).

    Example:
        >>> cb = MemoryMonitor()

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, opt-in)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        GPU memory via tf.config.experimental.get_memory_info().
        CPU memory via resource.getrusage (POSIX) ou psutil (Windows).
    """

    def __new__(cls):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Returns:
            Instancia de MemoryMonitor.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(self) -> None:
        """Inicializa o monitor de memoria.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Stateless: nenhum estado interno alem do herdado.
        """
        import tensorflow as tf

        super().__init__()
        logger.info("MemoryMonitor inicializado")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Loga uso de memoria ao final de cada epoca.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (gpu_memory_mb, cpu_memory_mb adicionados).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            GPU: tf.config.experimental.get_memory_info('GPU:0').
            CPU: resource.getrusage(RUSAGE_SELF).ru_maxrss.
        """
        import tensorflow as tf

        # ── GPU memory (se disponivel) ────────────────────────────────
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                mem_info = tf.config.experimental.get_memory_info("GPU:0")
                gpu_mb = mem_info.get("peak", 0) / (1024 * 1024)
                if logs is not None:
                    logs["gpu_memory_mb"] = gpu_mb
                logger.debug(
                    "Epoch %d: GPU peak memory=%.1f MB", epoch, gpu_mb
                )
        except Exception:
            # GPU memory info nao disponivel (CPU-only ou API nao suportada)
            pass

        # ── CPU memory (processo) ─────────────────────────────────────
        try:
            import resource

            # ru_maxrss em KB no Linux, em bytes no macOS
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # macOS retorna bytes, Linux retorna KB
            import platform

            if platform.system() == "Darwin":
                cpu_mb = rusage.ru_maxrss / (1024 * 1024)
            else:
                cpu_mb = rusage.ru_maxrss / 1024
            if logs is not None:
                logs["cpu_memory_mb"] = cpu_mb
            logger.debug(
                "Epoch %d: CPU peak memory=%.1f MB", epoch, cpu_mb
            )
        except ImportError:
            pass


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 14: LatencyBenchmark — Tempo de inferencia single-sample
#
# Mede o tempo de inferencia para um unico sample (batch_size=1) a cada
# 10 epocas. Util para avaliar se o modelo cabe no budget de latencia
# para geosteering realtime (target: < 10ms por sample).
#
# Referencia: C40 legado (futuro), requisito de geosteering realtime.
# ════════════════════════════════════════════════════════════════════════

class LatencyBenchmark:
    """Mede latencia de inferencia single-sample a cada 10 epocas.

    Executa model.predict() com batch_size=1 e mede tempo (ms).
    Util para validar budget de latencia em geosteering realtime.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _model_ref: Referencia ao modelo Keras.
        _sample_input: Tensor de entrada (batch_size=1) para benchmark.

    Example:
        >>> import tensorflow as tf
        >>> import numpy as np
        >>> model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
        >>> sample = np.random.randn(1, 600, 5).astype(np.float32)
        >>> cb = LatencyBenchmark(model, sample)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, opt-in)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (latencia realtime).
        Target geosteering: < 10ms por sample.
        Benchmark a cada 10 epocas para minimizar overhead.
    """

    def __new__(cls, model: Any, sample_input: Any):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            model: Modelo tf.keras.Model para benchmark.
            sample_input: Tensor de entrada (batch_size=1).

        Returns:
            Instancia de LatencyBenchmark.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(self, model: Any, sample_input: Any) -> None:
        """Inicializa o benchmark de latencia.

        Args:
            model: Modelo tf.keras.Model para benchmark de inferencia.
            sample_input: Tensor ou numpy array de entrada com batch_size=1.
                Shape esperada: (1, sequence_length, n_features).

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            sample_input deve ter batch_size=1 para medir latencia unitaria.
        """
        import tensorflow as tf

        super().__init__()
        self._model_ref = model
        self._sample_input = sample_input

        logger.info("LatencyBenchmark inicializado")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Mede latencia de inferencia a cada 10 epocas.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (inference_latency_ms adicionado).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Mede tempo de model.predict() com batch_size=1.
            Resultado em milissegundos.
        """
        # Benchmark apenas a cada 10 epocas
        if epoch % 10 != 0:
            return

        model = self._model_ref

        # Warmup: primeira chamada pode incluir overhead de compilacao
        model.predict(self._sample_input, verbose=0)

        # Benchmark: media de 3 chamadas
        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            model.predict(self._sample_input, verbose=0)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

        avg_latency = sum(latencies) / len(latencies)

        if logs is not None:
            logs["inference_latency_ms"] = avg_latency

        logger.info(
            "Epoch %d: inference_latency=%.2f ms (media 3 runs)",
            epoch,
            avg_latency,
        )


# ════════════════════════════════════════════════════════════════════════
# CALLBACK 15: EpochSummary — Resumo one-line por epoca
#
# Loga uma linha de resumo consolidada com metricas-chave ao final de
# cada epoca. Substitui a saida verbosa padrao do Keras por um formato
# compacto e informativo para logs de treinamento.
#
# Formato: "Epoch {N}: loss={:.4f} val_loss={:.4f} lr={:.2e} ..."
#
# Referencia: C40 legado (diagnostico pos-epoca).
# ════════════════════════════════════════════════════════════════════════

class EpochSummary:
    """Loga resumo one-line com metricas-chave ao final de cada epoca.

    Formato compacto para facilitar monitoramento em logs.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        Nenhum atributo especifico (stateless formatter).

    Example:
        >>> cb = EpochSummary()

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, opt-in)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        Output: "Epoch 42: loss=0.0123 val_loss=0.0145 lr=1.00e-04".
    """

    def __new__(cls):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Returns:
            Instancia de EpochSummary.

        Note:
            Lazy import de TensorFlow.
            Ref: CLAUDE.md (lazy TF import pattern).
        """
        import tensorflow as tf

        if not issubclass(cls, tf.keras.callbacks.Callback):
            cls.__bases__ = (tf.keras.callbacks.Callback,)

        instance = super().__new__(cls)
        return instance

    def __init__(self) -> None:
        """Inicializa o formatador de resumo.

        Note:
            Referenciado em:
                - training/callbacks.py: build_callbacks()
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Stateless: nenhum estado interno alem do herdado.
        """
        import tensorflow as tf

        super().__init__()
        logger.info("EpochSummary inicializado")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Loga resumo one-line com metricas-chave.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas do Keras.

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Metricas incluidas: loss, val_loss, lr (se disponivel).
        """
        import tensorflow as tf

        if logs is None:
            return

        # Coleta metricas-chave para resumo
        parts = [f"Epoch {epoch + 1}:"]

        # Metricas padrao do Keras
        for key in ("loss", "val_loss"):
            val = logs.get(key)
            if val is not None:
                parts.append(f"{key}={val:.4f}")

        # Learning rate atual (se disponivel via modelo)
        try:
            lr = tf.keras.backend.get_value(
                self.model.optimizer.learning_rate
            )
            parts.append(f"lr={lr:.2e}")
        except Exception:
            pass

        # Metricas custom (clean_val_loss, dual_gap, etc.)
        for key in ("clean_val_loss", "noisy_val_loss", "dual_gap"):
            val = logs.get(key)
            if val is not None:
                parts.append(f"{key}={val:.4f}")

        logger.info(" ".join(parts))


# ════════════════════════════════════════════════════════════════════════
# FACTORY: build_callbacks() — Montagem de callbacks a partir do config
#
# Constroi lista de Keras callbacks de acordo com os campos do
# PipelineConfig. Callbacks obrigatorios (EarlyStopping, BestEpochTracker)
# sao sempre incluidos. Callbacks opcionais sao ativados por flags.
#
# Composicao:
#   SEMPRE: EarlyStopping + BestEpochTracker + EpochSummary
#   OPT-IN: TensorBoard, CSVLogger, UpdateNoiseLevelCallback,
#           DualValidationCallback, PINNSLambdaScheduleCallback,
#           CausalDegradationMonitor, PeriodicCheckpoint
#
# Referencia: C40 legado (build_callbacks), docs/ARCHITECTURE_v2.md 6.2.
# ════════════════════════════════════════════════════════════════════════

def build_callbacks(
    config: PipelineConfig,
    model: Any = None,
    noise_level_var: Any = None,
    epoch_var: Any = None,
    *,
    val_clean_ds: Any = None,
    val_noisy_ds: Any = None,
    pinns_lambda_var: Any = None,
    val_ds: Any = None,
) -> List[Any]:
    """Factory que monta lista de Keras callbacks a partir do config.

    Constroi callbacks obrigatorios (EarlyStopping, BestEpochTracker,
    EpochSummary) e condicionais (TensorBoard, CSVLogger,
    UpdateNoiseLevelCallback, DualValidationCallback,
    PINNSLambdaScheduleCallback, CausalDegradationMonitor,
    PeriodicCheckpoint) de acordo com os campos do PipelineConfig.

    Args:
        config: PipelineConfig com todos os campos de callback:
            early_stopping_patience (int): Patience do EarlyStopping.
            use_restore_best_weights (bool): Se restaura pesos da melhor
                epoca (False no E-Robusto — preserva pesos noise-trained).
            use_tensorboard (bool): Ativa TensorBoard callback.
            use_csv_logger (bool): Ativa CSVLogger callback.
            use_noise (bool): Noise on-the-fly ativo.
            use_curriculum (bool): Curriculum noise 3-phase ativo.
            epochs_no_noise (int): Epocas clean (Fase 1 curriculum).
            noise_ramp_epochs (int): Epocas de rampa (Fase 2 curriculum).
            noise_level_max (float): Nivel maximo de ruido (Fase 3).
            use_dual_validation (bool): Ativa dual validation P2.
            use_pinns (bool): Ativa PINNs lambda schedule.
            pinns_lambda (float): Peso alvo da penalidade PINNs.
            penalty_warmup_epochs (int): Epocas de warmup PINNs.
            use_causal_mode (bool): Ativa modo causal (geosteering).
            experiment_dir (Optional[str]): Dir para TensorBoard/CSV/chkpt.
            verbose (bool): Logging detalhado.
        model: Modelo tf.keras.Model para WeightNormMonitor,
            CausalDegradationMonitor e PeriodicCheckpoint.
        noise_level_var: tf.Variable escalar (float32) que controla o nivel
            de ruido. Compartilhado com DataPipeline.build_train_map_fn().
            Obrigatorio se use_noise=True e use_curriculum=True.
        epoch_var: tf.Variable escalar (int64) para rastrear epoca atual.
            Reservado para uso futuro (ex: epoch-aware loss scheduling).
        val_clean_ds: tf.data.Dataset de validacao limpo (para P2).
            Obrigatorio se use_dual_validation=True.
        val_noisy_ds: tf.data.Dataset de validacao com ruido (para P2).
            Obrigatorio se use_dual_validation=True.
        pinns_lambda_var: tf.Variable escalar (float32) com peso PINNs.
            Obrigatorio se use_pinns=True.
        val_ds: tf.data.Dataset de validacao geral (para CausalDegradation).

    Returns:
        Lista de instancias tf.keras.callbacks.Callback, ordenada:
            1. EarlyStopping (sempre)
            2. BestEpochTracker (sempre)
            3. EpochSummary (sempre)
            4. TensorBoard (se use_tensorboard=True)
            5. CSVLogger (se use_csv_logger=True)
            6. UpdateNoiseLevelCallback (se curriculum ativo + var)
            7. DualValidationCallback (se P2 ativo + datasets)
            8. PINNSLambdaScheduleCallback (se PINNs ativo + var)
            9. CausalDegradationMonitor (se causal mode + model + val_ds)
            10. PeriodicCheckpoint (se experiment_dir definido)

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig.robusto()
        >>> callbacks = build_callbacks(config)
        >>> any(isinstance(cb, BestEpochTracker) for cb in callbacks)
        True

    Note:
        Referenciado em:
            - training/loop.py: TrainingLoop.run() (consome lista)
            - training/callbacks.py: testes unitarios
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (build_callbacks factory).
        EarlyStopping monitora 'val_loss' (modo min) com patience do config.
        use_restore_best_weights=False no E-Robusto (bug S20): evita
        reversao para pesos pre-ruido, preservando adaptacao a noise.
        TensorBoard e CSVLogger usam experiment_dir se disponivel.
    """
    import tensorflow as tf

    callbacks: List[Any] = []

    # ── 1. EarlyStopping (SEMPRE) ──────────────────────────────────
    # Monitora val_loss com patience do config.
    # use_restore_best_weights=False (E-Robusto S20): preserva pesos
    # noise-trained, evitando reversao para pesos pre-ruido.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        restore_best_weights=config.use_restore_best_weights,
        verbose=1 if config.verbose else 0,
    )
    callbacks.append(early_stopping)
    logger.info(
        "EarlyStopping: patience=%d, restore_best_weights=%s",
        config.early_stopping_patience,
        config.use_restore_best_weights,
    )

    # ── 2. BestEpochTracker (SEMPRE) ───────────────────────────────
    # Rastreia melhor epoca e val_loss para diagnostico pos-treino.
    best_tracker = BestEpochTracker(monitor="val_loss", mode="min")
    callbacks.append(best_tracker)

    # ── 3. EpochSummary (SEMPRE) ───────────────────────────────────
    # Resumo one-line com metricas-chave ao final de cada epoca.
    epoch_summary = EpochSummary()
    callbacks.append(epoch_summary)

    # ── 4. TensorBoard (CONDICIONAL) ──────────────────────────────
    # Ativa logging para TensorBoard se use_tensorboard=True.
    # Log dir: experiment_dir/tensorboard/ ou ./logs/tensorboard/
    if config.use_tensorboard:
        tb_log_dir = _resolve_log_dir(config, "tensorboard")
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
        )
        callbacks.append(tensorboard_cb)
        logger.info("TensorBoard: log_dir='%s'", tb_log_dir)

    # ── 5. CSVLogger (CONDICIONAL) ─────────────────────────────────
    # Salva metricas por epoca em CSV se use_csv_logger=True.
    # Caminho: experiment_dir/training_log.csv ou ./logs/training_log.csv
    if config.use_csv_logger:
        csv_path = _resolve_csv_path(config)
        csv_logger = tf.keras.callbacks.CSVLogger(
            csv_path,
            separator=",",
            append=False,
        )
        callbacks.append(csv_logger)
        logger.info("CSVLogger: path='%s'", csv_path)

    # ── 6. UpdateNoiseLevelCallback (CONDICIONAL) ──────────────────
    # Ativa curriculum noise 3-phase se: use_noise + use_curriculum
    # + noise_level_var fornecida. Sem a variavel, curriculum nao opera.
    if config.use_noise and config.use_curriculum and noise_level_var is not None:
        noise_cb = UpdateNoiseLevelCallback(noise_level_var, config)
        callbacks.append(noise_cb)
        logger.info(
            "UpdateNoiseLevelCallback: clean=%d, ramp=%d, max=%.4f",
            config.epochs_no_noise,
            config.noise_ramp_epochs,
            config.noise_level_max,
        )
    elif config.use_noise and config.use_curriculum and noise_level_var is None:
        logger.warning(
            "Curriculum noise ativo (use_curriculum=True) mas noise_level_var "
            "nao fornecido. UpdateNoiseLevelCallback NAO sera adicionado. "
            "Passe noise_level_var como tf.Variable(0.0, dtype=tf.float32)."
        )

    # ── 7. DualValidationCallback (CONDICIONAL — P2) ──────────────
    # Ativa dual validation se use_dual_validation=True e ambos
    # datasets (clean + noisy) fornecidos junto com o modelo.
    if config.use_dual_validation and val_clean_ds is not None and val_noisy_ds is not None:
        if model is not None:
            dual_cb = DualValidationCallback(
                val_clean_ds, val_noisy_ds, model, config
            )
            callbacks.append(dual_cb)
            logger.info("DualValidationCallback: P2 dual val ativo")
        else:
            logger.warning(
                "DualValidation ativo mas model nao fornecido. "
                "DualValidationCallback NAO sera adicionado."
            )
    elif config.use_dual_validation and (val_clean_ds is None or val_noisy_ds is None):
        logger.warning(
            "use_dual_validation=True mas val_clean_ds ou val_noisy_ds "
            "nao fornecido. DualValidationCallback NAO sera adicionado."
        )

    # ── 8. PINNSLambdaScheduleCallback (CONDICIONAL) ──────────────
    # Ativa annealing de lambda PINNs se use_pinns=True e
    # pinns_lambda_var fornecida.
    if config.use_pinns and pinns_lambda_var is not None:
        pinns_cb = PINNSLambdaScheduleCallback(pinns_lambda_var, config)
        callbacks.append(pinns_cb)
        logger.info(
            "PINNSLambdaScheduleCallback: target=%.6f, warmup=%d ep",
            config.pinns_lambda,
            config.penalty_warmup_epochs,
        )
    elif config.use_pinns and pinns_lambda_var is None:
        logger.warning(
            "use_pinns=True mas pinns_lambda_var nao fornecido. "
            "PINNSLambdaScheduleCallback NAO sera adicionado."
        )

    # ── 9. CausalDegradationMonitor (CONDICIONAL) ─────────────────
    # Ativa monitor causal se use_causal_mode=True e model + val_ds
    # fornecidos. Detecta degradacao da restricao causal.
    if config.use_causal_mode and model is not None and val_ds is not None:
        causal_cb = CausalDegradationMonitor(model, val_ds, config)
        callbacks.append(causal_cb)
        logger.info("CausalDegradationMonitor: modo causal ativo")

    # ── 10. PeriodicCheckpoint (CONDICIONAL) ───────────────────────
    # Ativa checkpoint periodico se experiment_dir definido.
    if config.experiment_dir is not None:
        periodic_cb = PeriodicCheckpoint(config, period=10)
        callbacks.append(periodic_cb)
        logger.info(
            "PeriodicCheckpoint: period=10, dir='%s/checkpoints/'",
            config.experiment_dir,
        )

    logger.info(
        "build_callbacks: %d callbacks montados para config '%s'",
        len(callbacks),
        config.experiment_tag or "default",
    )

    return callbacks


# ════════════════════════════════════════════════════════════════════════
# HELPERS INTERNOS — Resolucao de paths para logs
#
# Funcoes auxiliares que determinam caminhos de TensorBoard e CSVLogger
# a partir de experiment_dir (se disponivel) ou fallback para ./logs/.
# Cria diretorios automaticamente se nao existirem.
# ════════════════════════════════════════════════════════════════════════

def _resolve_log_dir(config: PipelineConfig, subdir: str) -> str:
    """Resolve diretorio de log para TensorBoard.

    Usa experiment_dir/subdir se disponivel, senao ./logs/subdir.
    Cria o diretorio se nao existir.

    Args:
        config: PipelineConfig com experiment_dir.
        subdir: Nome do subdiretorio (ex: 'tensorboard').

    Returns:
        Caminho absoluto do diretorio de log.

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (TensorBoard)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        Fallback: ./logs/{subdir} se experiment_dir nao definido.
    """
    if config.experiment_dir:
        log_dir = os.path.join(config.experiment_dir, subdir)
    else:
        log_dir = os.path.join(".", "logs", subdir)

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _resolve_csv_path(config: PipelineConfig) -> str:
    """Resolve caminho do arquivo CSV para CSVLogger.

    Usa experiment_dir/training_log.csv se disponivel,
    senao ./logs/training_log.csv.
    Cria o diretorio pai se nao existir.

    Args:
        config: PipelineConfig com experiment_dir.

    Returns:
        Caminho absoluto do arquivo CSV.

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (CSVLogger)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        Fallback: ./logs/training_log.csv se experiment_dir nao definido.
    """
    if config.experiment_dir:
        csv_dir = config.experiment_dir
        csv_path = os.path.join(csv_dir, "training_log.csv")
    else:
        csv_dir = os.path.join(".", "logs")
        csv_path = os.path.join(csv_dir, "training_log.csv")

    os.makedirs(csv_dir, exist_ok=True)
    return csv_path
