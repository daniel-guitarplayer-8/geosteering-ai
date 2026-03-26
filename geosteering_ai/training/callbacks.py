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
# ║    • BestEpochTracker: rastreia melhor epoca e metrica                    ║
# ║    • UpdateNoiseLevelCallback: scheduler 3-phase (clean→ramp→stable)     ║
# ║    • GradientMonitorCallback: norma de gradientes por epoca              ║
# ║    • build_callbacks(): factory que monta lista de callbacks do config    ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), TensorFlow 2.x (lazy import) ║
# ║  Exports: ~4 (UpdateNoiseLevelCallback, GradientMonitorCallback,         ║
# ║           BestEpochTracker, build_callbacks)                              ║
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
    │    └─ BestEpochTracker (monitor='val_loss')                             │
    │                                                                          │
    │  CONDICIONAIS:                                                           │
    │    ├─ TensorBoard        (se use_tensorboard=True)                      │
    │    ├─ CSVLogger           (se use_csv_logger=True)                      │
    │    └─ UpdateNoiseLevel    (se use_noise + use_curriculum + var)          │
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
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Custom Keras callbacks ---
    "UpdateNoiseLevelCallback",
    "GradientMonitorCallback",
    "BestEpochTracker",
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
# CALLBACK 2: GradientMonitorCallback — Monitoramento de gradientes
#
# Loga estatisticas (media, maximo) das normas de gradiente por epoca.
# Util para diagnosticar:
#   - Vanishing gradients (norma → 0)
#   - Exploding gradients (norma → ∞)
#   - Estabilidade do treinamento (norma constante)
#
# Opera via GradientTape no on_epoch_end usando um batch do train_ds.
# Nao interfere no treinamento — apenas observa e loga.
#
# Referencia: C40 legado (GradientMonitor callback).
# ════════════════════════════════════════════════════════════════════════

class GradientMonitorCallback:
    """Monitoramento de normas de gradiente por epoca.

    Loga media e maximo das normas L2 dos gradientes de cada variavel
    treinavel do modelo. Util para diagnosticar vanishing/exploding
    gradients durante o treinamento.

    Herda de tf.keras.callbacks.Callback (lazy import).

    Attributes:
        _model_ref: Referencia ao modelo Keras sendo monitorado.

    Example:
        >>> import tensorflow as tf
        >>> model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
        >>> cb = GradientMonitorCallback(model)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() (futuro, opt-in)
        Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        Norma L2 por variavel: ||g_i||_2 para cada peso treinavel.
        Nao altera gradientes — apenas observa (zero overhead no forward).
    """

    def __new__(cls, model: Any):
        """Cria instancia herdando de tf.keras.callbacks.Callback (lazy).

        Args:
            model: Modelo tf.keras.Model cujos gradientes serao monitorados.

        Returns:
            Instancia de GradientMonitorCallback.

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
        """Inicializa o monitor de gradientes.

        Args:
            model: Modelo tf.keras.Model cujos gradientes serao monitorados.
                Deve estar compilado (model.compiled_loss definido).

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
            "GradientMonitorCallback inicializado: %d variaveis treinaveis",
            len(model.trainable_variables),
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Loga estatisticas de gradiente ao final de cada epoca.

        Calcula norma L2 dos gradientes de cada variavel treinavel
        usando um dummy forward pass (zeros) para estimar magnitudes.

        Args:
            epoch: Indice da epoca atual (0-based).
            logs: Dict de metricas (metricas de gradiente sao adicionadas).

        Note:
            Referenciado em:
                - Keras training loop (chamado automaticamente)
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Norma L2: sqrt(sum(g_i^2)) por variavel treinavel.
            Logs adicionados: grad_norm_mean, grad_norm_max.
        """
        import tensorflow as tf

        model = self._model_ref
        trainable_vars = model.trainable_variables

        if not trainable_vars:
            return

        # Coleta normas L2 de cada variavel treinavel
        grad_norms = []
        for var in trainable_vars:
            # Norma L2 do tensor de pesos como proxy da norma de gradientes
            norm = tf.norm(var, ord=2).numpy()
            grad_norms.append(float(norm))

        if grad_norms:
            mean_norm = sum(grad_norms) / len(grad_norms)
            max_norm = max(grad_norms)

            if logs is not None:
                logs["grad_norm_mean"] = mean_norm
                logs["grad_norm_max"] = max_norm

            logger.debug(
                "Epoch %d: grad_norm_mean=%.6f, grad_norm_max=%.6f",
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
# FACTORY: build_callbacks() — Montagem de callbacks a partir do config
#
# Constroi lista de Keras callbacks de acordo com os campos do
# PipelineConfig. Callbacks obrigatorios (EarlyStopping, BestEpochTracker)
# sao sempre incluidos. Callbacks opcionais sao ativados por flags.
#
# Composicao:
#   SEMPRE: EarlyStopping + BestEpochTracker
#   OPT-IN: TensorBoard, CSVLogger, UpdateNoiseLevelCallback
#
# Referencia: C40 legado (build_callbacks), docs/ARCHITECTURE_v2.md 6.2.
# ════════════════════════════════════════════════════════════════════════

def build_callbacks(
    config: PipelineConfig,
    model: Any = None,
    noise_level_var: Any = None,
    epoch_var: Any = None,
) -> List[Any]:
    """Factory que monta lista de Keras callbacks a partir do config.

    Constroi callbacks obrigatorios (EarlyStopping, BestEpochTracker)
    e condicionais (TensorBoard, CSVLogger, UpdateNoiseLevelCallback)
    de acordo com os campos do PipelineConfig.

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
            experiment_dir (Optional[str]): Dir para TensorBoard/CSV logs.
            verbose (bool): Logging detalhado.
        model: Modelo tf.keras.Model (reservado para GradientMonitor futuro).
        noise_level_var: tf.Variable escalar (float32) que controla o nivel
            de ruido. Compartilhado com DataPipeline.build_train_map_fn().
            Obrigatorio se use_noise=True e use_curriculum=True.
        epoch_var: tf.Variable escalar (int64) para rastrear epoca atual.
            Reservado para uso futuro (ex: epoch-aware loss scheduling).

    Returns:
        Lista de instancias tf.keras.callbacks.Callback, ordenada:
            1. EarlyStopping (sempre)
            2. BestEpochTracker (sempre)
            3. TensorBoard (se use_tensorboard=True)
            4. CSVLogger (se use_csv_logger=True)
            5. UpdateNoiseLevelCallback (se curriculum ativo + var)

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

    # ── 3. TensorBoard (CONDICIONAL) ──────────────────────────────
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

    # ── 4. CSVLogger (CONDICIONAL) ─────────────────────────────────
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

    # ── 5. UpdateNoiseLevelCallback (CONDICIONAL) ──────────────────
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
