# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: training/adaptation.py                                           ║
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
# ║    • C69: Domain Adaptation de dados sinteticos para dados de campo        ║
# ║    • DomainAdapter: orquestrador de estrategias de adaptacao               ║
# ║    • fine_tune: congela backbone, treina head com LR reduzido              ║
# ║    • progressive: descongela camadas progressivamente (head → backbone)    ║
# ║    • AdaptationResult: dataclass com historico e metricas da adaptacao     ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), tensorflow (lazy import)       ║
# ║  Exports: ~2 simbolos (DomainAdapter, AdaptationResult) — ver __all__    ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6 (Training), legado C69             ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C69 Domain Adaptation)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""DomainAdapter — Adaptacao de dominio de dados sinteticos para dados de campo.

Modelos treinados em dados sinteticos (forward-modeled) podem sofrer domain
shift quando aplicados a dados de campo reais. Este modulo fornece estrategias
para adaptar (fine-tune) modelos pre-treinados usando datasets de campo,
preservando o conhecimento adquirido durante o treinamento sintetico.

Estrategias suportadas:

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  DOMAIN ADAPTATION — Sintetico → Campo                             │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Strategy: "fine_tune"                                               │
    │  ┌────────────────────────────────────────────────────────────────┐  │
    │  │  1. Congela freeze_ratio × total_layers (backbone)            │  │
    │  │  2. Compila com LR_base × lr_factor (reduzido)                │  │
    │  │  3. model.fit(field_train_ds, epochs, EarlyStopping)          │  │
    │  │  4. Descongela TODAS as camadas ao final                      │  │
    │  └────────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │  Strategy: "progressive"                                             │
    │  ┌────────────────────────────────────────────────────────────────┐  │
    │  │  1. Congela freeze_ratio × total_layers (backbone)            │  │
    │  │  2. Divide epochs em N blocos (N = camadas congeladas)        │  │
    │  │  3. Loop: descongela 1 camada, treina bloco_epochs            │  │
    │  │     LR decai a cada bloco (discriminative fine-tuning)        │  │
    │  │  4. Descongela TODAS as camadas ao final                      │  │
    │  └────────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │  AdaptationResult (retorno):                                         │
    │    strategy, epochs_trained, initial_val_loss, final_val_loss,       │
    │    history (metricas epoch-by-epoch)                                  │
    └──────────────────────────────────────────────────────────────────────┘

Example:
    >>> from geosteering_ai.config import PipelineConfig
    >>> from geosteering_ai.training.adaptation import DomainAdapter
    >>>
    >>> config = PipelineConfig.robusto()
    >>> adapter = DomainAdapter(config)
    >>> result = adapter.adapt(
    ...     model, field_train_ds, field_val_ds,
    ...     strategy="fine_tune", epochs=50, lr_factor=0.1,
    ... )
    >>> result.final_val_loss
    0.042

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Referenciado em:
        - training/__init__.py: re-export DomainAdapter, AdaptationResult
        - config.py: PipelineConfig.learning_rate (LR base para lr_factor)
    Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
    Domain adaptation e relevante para geosteering em campo real (Morales
    v5.0.15+), onde dados sinteticos forward-modeled diferem da resposta
    EM medida em campo (efeitos de borehole, mud filtrate, anisotropia 3D).
    eps = 1e-12 (float32 safe, NUNCA 1e-30).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Resultado da adaptacao ---
    "AdaptationResult",
    # --- Classe principal (adaptacao de dominio) ---
    "DomainAdapter",
]


# ════════════════════════════════════════════════════════════════════════
# CONSTANTES — Defaults e limites de seguranca
#
# _VALID_STRATEGIES: estrategias suportadas para adaptacao de dominio.
#   "fine_tune" e a mais simples e eficaz para datasets de campo pequenos.
#   "progressive" e mais sofisticada, adequada para datasets maiores.
# _MIN_FREEZE_RATIO / _MAX_FREEZE_RATIO: limites para fracao de freeze.
#   Abaixo de 0.1, praticamente nenhuma camada congelada (sem protecao).
#   Acima de 0.99, praticamente todas congeladas (nenhuma adaptacao).
# _MIN_LR_FACTOR: limite inferior para fator de LR. Abaixo de 1e-6,
#   o LR efetivo seria negligivel (nenhuma atualizacao de pesos).
# _EPS: epsilon numerico para float32 (NUNCA 1e-30).
# ════════════════════════════════════════════════════════════════════════

_VALID_STRATEGIES = ("fine_tune", "progressive")
"""Estrategias validas para domain adaptation."""

_MIN_FREEZE_RATIO: float = 0.1
"""Fracao minima de camadas a congelar (seguranca)."""

_MAX_FREEZE_RATIO: float = 0.99
"""Fracao maxima de camadas a congelar (seguranca)."""

_MIN_LR_FACTOR: float = 1e-6
"""Fator minimo de reducao de LR (evita LR efetivo zero)."""

_EPS: float = 1e-12
"""Epsilon numerico para float32 (NUNCA 1e-30)."""

_PROGRESSIVE_LR_DECAY_PER_BLOCK: float = 0.8
"""Fator de decay de LR entre blocos da estrategia progressive."""


# ════════════════════════════════════════════════════════════════════════
# ADAPTATION RESULT — Container de resultado da adaptacao
#
# Encapsula metricas e historico de uma execucao de domain adaptation.
# Compativel com Keras history.history (dict de listas por metrica).
# strategy: identifica qual estrategia foi usada ("fine_tune"/"progressive").
# epochs_trained: total de epocas efetivamente executadas (pode ser menor
#   que o solicitado se EarlyStopping foi acionado).
# initial_val_loss / final_val_loss: permitem calcular a melhoria relativa.
# history: metricas epoch-by-epoch (loss, val_loss, etc.).
# ════════════════════════════════════════════════════════════════════════


@dataclass
class AdaptationResult:
    """Resultado de uma execucao de domain adaptation.

    Container com metricas de antes/depois da adaptacao, estrategia usada,
    epocas treinadas, e historico completo de metricas epoch-by-epoch.
    Retornado por DomainAdapter.adapt().

    Attributes:
        strategy (str): Estrategia utilizada. Valores validos:
            "fine_tune" (congela backbone, treina head) ou
            "progressive" (descongela camadas progressivamente).
        epochs_trained (int): Total de epocas efetivamente executadas.
            Pode ser menor que ``epochs`` se EarlyStopping foi acionado.
        initial_val_loss (float): Valor de val_loss ANTES da adaptacao
            (avaliacao do modelo pre-treinado no field_val_ds).
        final_val_loss (float): Valor de val_loss APOS a adaptacao
            (melhor val_loss alcancada durante o treinamento).
        history (dict): Historico epoch-by-epoch de metricas Keras.
            Chaves tipicas: "loss", "val_loss", "lr". Valores: listas
            de floats. Compativel com Keras history.history format.

    Example:
        >>> result = AdaptationResult(
        ...     strategy="fine_tune",
        ...     epochs_trained=30,
        ...     initial_val_loss=0.15,
        ...     final_val_loss=0.042,
        ...     history={"loss": [0.1, 0.08], "val_loss": [0.15, 0.042]},
        ... )
        >>> result.final_val_loss < result.initial_val_loss
        True

    Note:
        Referenciado em:
            - training/adaptation.py: DomainAdapter.adapt() (retorno)
            - training/__init__.py: re-export AdaptationResult
        Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
        history e compativel com Keras history.history format para
        integracao direta com callbacks e visualizacao.
    """

    strategy: str = "fine_tune"
    epochs_trained: int = 0
    initial_val_loss: float = float("inf")
    final_val_loss: float = float("inf")
    history: Dict[str, List[float]] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════
# DOMAIN ADAPTER — Adaptacao de dominio sintetico → campo
#
# Orquestra o processo de adaptar um modelo pre-treinado em dados
# sinteticos para dados de campo reais. O domain shift entre sintetico
# e campo e comum em geosteering:
#
#   - Dados sinteticos: forward-modeled, 1D layered, sem borehole effects
#   - Dados de campo: efeitos 3D, mud filtrate, anisotropia, noise real
#
# Estrategias implementadas:
#
# ┌────────────────────────────────────────────────────────────────────┐
# │  FINE_TUNE                                                         │
# │    • Congela primeiras freeze_ratio × N camadas (backbone)        │
# │    • Treina camadas restantes (head) com LR × lr_factor           │
# │    • EarlyStopping com patience = min(15, epochs//3)              │
# │    • Simples, eficaz para datasets de campo pequenos (~100-1K)    │
# ├────────────────────────────────────────────────────────────────────┤
# │  PROGRESSIVE                                                       │
# │    • Inicia como fine_tune (backbone congelado)                   │
# │    • A cada bloco de epocas, descongela 1 camada do backbone      │
# │    • LR decai a cada bloco (discriminative fine-tuning)           │
# │    • Mais controle, adequado para datasets maiores (~1K-10K)      │
# └────────────────────────────────────────────────────────────────────┘
#
# Ambas as estrategias descongelam TODAS as camadas ao final para
# garantir que o modelo retorna ao estado "fully trainable".
# ════════════════════════════════════════════════════════════════════════


class DomainAdapter:
    """C69: Domain adaptation de dados sinteticos para dados de campo.

    Adapta modelo pre-treinado em dados sinteticos (forward-modeled) para
    dados de campo reais. O domain shift e mitigado via fine-tuning com
    camadas congeladas (preservando features aprendidas no backbone) e
    LR reduzido (evitando catastrofic forgetting).

    Strategies:
        - fine_tune: congela backbone (exceto ultimas N camadas), retreina
          com LR reduzido em field data. Simples e eficaz para datasets
          de campo pequenos (~100-1K amostras).
        - progressive: descongela camadas progressivamente do head ao
          backbone, treinando cada bloco por algumas epocas com LR
          decrescente. Mais sofisticado para datasets maiores.

    Attributes:
        config (PipelineConfig): Configuracao do pipeline (imutavel).
        _logger (logging.Logger): Logger estruturado para o modulo.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> from geosteering_ai.training.adaptation import DomainAdapter
        >>>
        >>> config = PipelineConfig.robusto()
        >>> adapter = DomainAdapter(config)
        >>> result = adapter.adapt(model, field_train_ds, field_val_ds)
        >>> result.strategy
        'fine_tune'

    Note:
        Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
        Referenciado em:
            - training/__init__.py: re-export DomainAdapter
            - config.py: PipelineConfig.learning_rate (LR base)
        Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
        Domain shift sintetico→campo: efeitos de borehole, mud filtrate,
        anisotropia 3D, noise EM real vs. gaussiano simulado.
        eps = 1e-12 (float32 safe, NUNCA 1e-30).
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Inicializa DomainAdapter com configuracao do pipeline.

        Args:
            config: PipelineConfig validada. Campos relevantes:
                - learning_rate: LR base (sera multiplicado por lr_factor)
                - optimizer: tipo de optimizer para recompile
                - epochs: referencia para duracao (adapt usa argumento proprio)
                - loss_type: perda para recompile (mantida do treinamento)

        Note:
            Referenciado em:
                - training/adaptation.py: adapt(), _fine_tune(), _progressive_unfreeze()
            Ref: docs/ARCHITECTURE_v2.md secao 6.
            PipelineConfig e o ponto unico de verdade (NUNCA globals().get()).
        """
        self.config = config
        self._logger = logging.getLogger(__name__)

    # ──────────────────────────────────────────────────────────────────
    # SECAO: ADAPT — Ponto de entrada principal
    # ──────────────────────────────────────────────────────────────────
    # Valida parametros, avalia modelo pre-treinado no field_val_ds
    # (initial_val_loss), despacha para a estrategia selecionada,
    # e empacota o resultado em AdaptationResult.
    #
    # Fluxo:
    #   1. Validacao de strategy, freeze_ratio, lr_factor
    #   2. Avaliacao pre-adaptacao (initial_val_loss)
    #   3. Despacho: _fine_tune() ou _progressive_unfreeze()
    #   4. Empacota AdaptationResult com metricas pre/pos
    # ──────────────────────────────────────────────────────────────────

    def adapt(
        self,
        model: Any,
        field_train_ds: Any,
        field_val_ds: Any,
        *,
        strategy: str = "fine_tune",
        epochs: int = 50,
        lr_factor: float = 0.1,
        freeze_ratio: float = 0.8,
    ) -> AdaptationResult:
        """Executa domain adaptation no modelo pre-treinado.

        Avalia o modelo no field_val_ds (pre-adaptacao), aplica a
        estrategia selecionada, e retorna AdaptationResult com metricas
        de antes/depois e historico completo.

        Args:
            model: Modelo Keras pre-treinado em dados sinteticos.
                Deve estar compilado (optimizer + loss definidos).
                tf.keras.Model com layers acessiveis via model.layers.
            field_train_ds: tf.data.Dataset com dados de campo para
                treinamento. Deve estar batched e, se aplicavel,
                com FV+GS+scaling ja aplicados via InferencePipeline.
            field_val_ds: tf.data.Dataset com dados de campo para
                validacao. Mesmo formato que field_train_ds.
            strategy: Estrategia de adaptacao. Valores validos:
                "fine_tune" (congela backbone, treina head) ou
                "progressive" (descongela progressivamente).
                Default: "fine_tune".
            epochs: Numero maximo de epocas para adaptacao.
                Default: 50. EarlyStopping pode encerrar antes.
            lr_factor: Fator multiplicativo para o LR base
                (config.learning_rate × lr_factor). Valores tipicos:
                0.01-0.1. Default: 0.1.
            freeze_ratio: Fracao de camadas a congelar (0.0 a 1.0).
                0.8 = congela 80% das camadas (backbone), treina 20%
                (head). Default: 0.8.

        Returns:
            AdaptationResult com strategy, epochs_trained, initial_val_loss,
            final_val_loss, e history (epoch-by-epoch metrics dict).

        Raises:
            ValueError: Se strategy nao esta em _VALID_STRATEGIES.
            ValueError: Se freeze_ratio fora de [_MIN_FREEZE_RATIO, _MAX_FREEZE_RATIO].
            ValueError: Se lr_factor < _MIN_LR_FACTOR.
            ValueError: Se epochs < 1.

        Note:
            Referenciado em:
                - training/adaptation.py: DomainAdapter (API publica)
                - training/__init__.py: docstring de exemplo
            Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
            O modelo e avaliado com model.evaluate() ANTES da adaptacao
            para estabelecer baseline (initial_val_loss).
            Todas as camadas sao descongeladas ao final, independente
            do resultado da adaptacao.
        """
        import tensorflow as tf  # Lazy import — D7: evita import-time GPU alloc

        # ── Validacao de parametros ─────────────────────────────────
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy deve ser um de {_VALID_STRATEGIES}, "
                f"recebido: '{strategy}'"
            )
        if not (_MIN_FREEZE_RATIO <= freeze_ratio <= _MAX_FREEZE_RATIO):
            raise ValueError(
                f"freeze_ratio deve estar em [{_MIN_FREEZE_RATIO}, "
                f"{_MAX_FREEZE_RATIO}], recebido: {freeze_ratio}"
            )
        if lr_factor < _MIN_LR_FACTOR:
            raise ValueError(
                f"lr_factor deve ser >= {_MIN_LR_FACTOR}, "
                f"recebido: {lr_factor}"
            )
        if epochs < 1:
            raise ValueError(
                f"epochs deve ser >= 1, recebido: {epochs}"
            )

        # ── LR efetivo para adaptacao ──────────────────────────────
        adaptation_lr = self.config.learning_rate * lr_factor

        self._logger.info(
            "Domain adaptation: strategy='%s', epochs=%d, "
            "lr_factor=%.6f (LR efetivo=%.2e), freeze_ratio=%.2f",
            strategy,
            epochs,
            lr_factor,
            adaptation_lr,
            freeze_ratio,
        )

        # ── Avaliacao pre-adaptacao (baseline) ─────────────────────
        # D7: model.evaluate retorna [loss, metric1, metric2, ...]
        pre_eval = model.evaluate(field_val_ds, verbose=0)
        # Se evaluate retorna lista, loss e o primeiro elemento
        initial_val_loss = float(
            pre_eval[0] if isinstance(pre_eval, (list, tuple)) else pre_eval
        )
        self._logger.info(
            "Val loss pre-adaptacao: %.6f", initial_val_loss
        )

        # ── Despacho para estrategia ───────────────────────────────
        t0 = time.monotonic()

        if strategy == "fine_tune":
            history = self._fine_tune(
                model, field_train_ds, field_val_ds,
                epochs=epochs,
                lr=adaptation_lr,
                freeze_ratio=freeze_ratio,
            )
        else:  # strategy == "progressive" (validado acima)
            history = self._progressive_unfreeze(
                model, field_train_ds, field_val_ds,
                epochs=epochs,
                lr=adaptation_lr,
                freeze_ratio=freeze_ratio,
            )

        elapsed = time.monotonic() - t0

        # ── Computa metricas finais ────────────────────────────────
        val_loss_history = history.get("val_loss", [])
        final_val_loss = (
            min(val_loss_history) if val_loss_history
            else initial_val_loss
        )
        epochs_trained = len(history.get("loss", []))

        self._logger.info(
            "Domain adaptation concluida em %.1fs: %d epocas, "
            "val_loss %.6f → %.6f (delta=%.6f)",
            elapsed,
            epochs_trained,
            initial_val_loss,
            final_val_loss,
            final_val_loss - initial_val_loss,
        )

        return AdaptationResult(
            strategy=strategy,
            epochs_trained=epochs_trained,
            initial_val_loss=initial_val_loss,
            final_val_loss=final_val_loss,
            history=history,
        )

    # ──────────────────────────────────────────────────────────────────
    # SECAO: FINE_TUNE — Congela backbone, treina head com LR reduzido
    # ──────────────────────────────────────────────────────────────────
    # Estrategia simples e eficaz para datasets de campo pequenos.
    #
    # Fluxo:
    #   1. Identifica camadas treinaveis (layer.trainable = True)
    #   2. Congela primeiras freeze_ratio × N camadas
    #   3. Recompila modelo com optimizer atualizado (LR reduzido)
    #   4. Treina com EarlyStopping (patience conservador)
    #   5. Descongela TODAS as camadas ao final
    #
    # A fracao freeze_ratio define a fronteira backbone/head:
    #   freeze_ratio=0.8 → 80% congelado (backbone), 20% livre (head)
    #   freeze_ratio=0.5 → 50% congelado, 50% livre
    #
    # EarlyStopping patience = min(15, epochs // 3):
    #   Conservador para evitar overfitting em datasets pequenos.
    #   Para epochs=50 → patience=15. Para epochs=20 → patience=6.
    # ──────────────────────────────────────────────────────────────────

    def _fine_tune(
        self,
        model: Any,
        train_ds: Any,
        val_ds: Any,
        *,
        epochs: int,
        lr: float,
        freeze_ratio: float,
    ) -> Dict[str, List[float]]:
        """Congela backbone, treina head com LR reduzido.

        Estrategia classica de fine-tuning: preserva features do backbone
        (camadas convolucionais/recorrentes iniciais) e adapta apenas as
        camadas finais (head) ao novo dominio. O LR reduzido evita
        catastrofic forgetting nas camadas descongeladas.

        Args:
            model: Modelo Keras compilado. Sera recompilado com novo LR.
            train_ds: tf.data.Dataset de treinamento (field data, batched).
            val_ds: tf.data.Dataset de validacao (field data, batched).
            epochs: Numero maximo de epocas para fine-tuning.
            lr: Learning rate efetivo (ja multiplicado por lr_factor).
            freeze_ratio: Fracao de camadas a congelar (0.0 a 1.0).

        Returns:
            dict: Historico Keras (history.history) com metricas
            epoch-by-epoch. Chaves: "loss", "val_loss", etc.

        Note:
            Referenciado em:
                - training/adaptation.py: DomainAdapter.adapt() (despacho)
            Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
            TODAS as camadas sao descongeladas ao final (bloco finally)
            para garantir que o modelo nao permanece parcialmente congelado.
            EarlyStopping monitora "val_loss" com patience conservador.
        """
        import tensorflow as tf  # Lazy import — evita import-time GPU alloc

        # ── Salvar estado original de trainable ────────────────────
        original_trainable = [
            layer.trainable for layer in model.layers
        ]

        try:
            # ── Congelar camadas do backbone ───────────────────────
            n_layers = len(model.layers)
            n_freeze = int(n_layers * freeze_ratio)
            # D7: garante pelo menos 1 camada livre para adaptacao
            n_freeze = min(n_freeze, n_layers - 1)

            for i, layer in enumerate(model.layers):
                layer.trainable = i >= n_freeze  # True para head, False para backbone

            n_trainable = sum(1 for l in model.layers if l.trainable)
            n_frozen = n_layers - n_trainable
            self._logger.info(
                "Fine-tune: %d/%d camadas congeladas, %d treinaveis, LR=%.2e",
                n_frozen, n_layers, n_trainable, lr,
            )

            # ── Recompilar com LR reduzido ─────────────────────────
            # D7: preserva loss e metrics do modelo original
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=lr,
                    clipnorm=1.0,  # D7: gradient clipping conservador
                ),
                loss=model.loss,
                metrics=model.compiled_metrics._metrics if hasattr(
                    model, "compiled_metrics"
                ) and model.compiled_metrics is not None else None,
            )

            # ── EarlyStopping conservador ──────────────────────────
            patience = min(15, max(3, epochs // 3))
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]

            self._logger.info(
                "Fine-tune fit: epochs=%d, patience=%d",
                epochs, patience,
            )

            # ── Treinamento ────────────────────────────────────────
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )

            return dict(history.history)

        finally:
            # ── Restaurar trainable para TODAS as camadas ──────────
            # D7: garante que modelo nao permanece parcialmente congelado
            for layer, was_trainable in zip(
                model.layers, original_trainable
            ):
                layer.trainable = was_trainable

            self._logger.info(
                "Fine-tune: trainable restaurado para estado original"
            )

    # ──────────────────────────────────────────────────────────────────
    # SECAO: PROGRESSIVE UNFREEZE — Descongela camadas progressivamente
    # ──────────────────────────────────────────────────────────────────
    # Estrategia gradual inspirada em ULMFiT (Howard & Ruder, 2018):
    #
    # ┌────────────────────────────────────────────────────────────────┐
    # │  Bloco 1: head (ultimas 20% camadas) — LR = lr                │
    # │  Bloco 2: descongela 1 camada — LR = lr × 0.8                │
    # │  Bloco 3: descongela 1 camada — LR = lr × 0.64               │
    # │  ...                                                           │
    # │  Bloco N: descongela ultima camada — LR = lr × 0.8^(N-1)     │
    # └────────────────────────────────────────────────────────────────┘
    #
    # Discriminative fine-tuning: camadas mais profundas (proximas ao
    # input) aprendem features mais genericas e devem ser ajustadas
    # com LR menor. Camadas mais rasas (proximas ao output) capturam
    # features task-specific e toleram LR maior.
    #
    # Epocas por bloco = epochs // n_unfreeze_steps (minimo 2).
    # Se sobrar epochs, redistribui para o ultimo bloco.
    # ──────────────────────────────────────────────────────────────────

    def _progressive_unfreeze(
        self,
        model: Any,
        train_ds: Any,
        val_ds: Any,
        *,
        epochs: int,
        lr: float,
        freeze_ratio: float,
    ) -> Dict[str, List[float]]:
        """Descongela camadas progressivamente do head ao backbone.

        Inspirado em discriminative fine-tuning (ULMFiT): cada bloco de
        epocas descongela uma camada adicional com LR progressivamente
        menor. Camadas profundas (features genericas) recebem LR menor
        que camadas rasas (features task-specific).

        Args:
            model: Modelo Keras compilado. Sera recompilado a cada bloco.
            train_ds: tf.data.Dataset de treinamento (field data, batched).
            val_ds: tf.data.Dataset de validacao (field data, batched).
            epochs: Numero total de epocas (dividido entre blocos).
            lr: Learning rate inicial (ja multiplicado por lr_factor).
            freeze_ratio: Fracao de camadas inicialmente congeladas.

        Returns:
            dict: Historico Keras mesclado (concatenacao de todos os blocos).
            Chaves: "loss", "val_loss", etc. Valores: listas de floats.

        Note:
            Referenciado em:
                - training/adaptation.py: DomainAdapter.adapt() (despacho)
            Ref: docs/ARCHITECTURE_v2.md secao 6 (Training).
            TODAS as camadas sao descongeladas ao final (bloco finally).
            LR decai por _PROGRESSIVE_LR_DECAY_PER_BLOCK (0.8) a cada bloco.
            Epocas por bloco = max(2, epochs // n_unfreeze_steps).
        """
        import tensorflow as tf  # Lazy import — evita import-time GPU alloc

        # ── Salvar estado original de trainable ────────────────────
        original_trainable = [
            layer.trainable for layer in model.layers
        ]

        try:
            n_layers = len(model.layers)
            n_freeze = int(n_layers * freeze_ratio)
            n_freeze = min(n_freeze, n_layers - 1)

            # ── Congelar backbone ──────────────────────────────────
            for i, layer in enumerate(model.layers):
                layer.trainable = i >= n_freeze

            # ── Calcular blocos de unfreezing ──────────────────────
            # D7: cada bloco descongela 1 camada, do head para o backbone
            # Limita n_unfreeze_steps para nao exceder camadas congeladas
            # e para garantir pelo menos 2 epocas por bloco
            n_unfreeze_steps = max(1, min(
                n_freeze,
                epochs // 2,  # Pelo menos 2 epocas por bloco
            ))
            epochs_per_block = max(2, epochs // n_unfreeze_steps)

            self._logger.info(
                "Progressive unfreeze: %d camadas congeladas, "
                "%d blocos de %d epocas, LR_decay_per_block=%.2f",
                n_freeze, n_unfreeze_steps, epochs_per_block,
                _PROGRESSIVE_LR_DECAY_PER_BLOCK,
            )

            # ── Historico mesclado (concatenacao de blocos) ────────
            merged_history: Dict[str, List[float]] = {}
            current_lr = lr
            total_epochs_used = 0

            for block_idx in range(n_unfreeze_steps):
                # ── Determinar epocas deste bloco ──────────────────
                remaining_epochs = epochs - total_epochs_used
                if remaining_epochs <= 0:
                    # D7: nao restam epocas, encerra loop
                    break

                # Ultimo bloco usa epocas restantes
                if block_idx == n_unfreeze_steps - 1:
                    block_epochs = remaining_epochs
                else:
                    block_epochs = min(epochs_per_block, remaining_epochs)

                # ── Descongelar 1 camada (exceto primeiro bloco) ───
                if block_idx > 0:
                    # D7: descongelar da camada mais proxima ao head
                    # em direcao ao backbone (indice decrescente)
                    unfreeze_idx = n_freeze - block_idx
                    if 0 <= unfreeze_idx < n_layers:
                        model.layers[unfreeze_idx].trainable = True
                        self._logger.info(
                            "Bloco %d: descongela camada %d ('%s'), LR=%.2e",
                            block_idx + 1,
                            unfreeze_idx,
                            model.layers[unfreeze_idx].name,
                            current_lr,
                        )

                # ── Recompilar com LR atualizado ───────────────────
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=current_lr,
                        clipnorm=1.0,
                    ),
                    loss=model.loss,
                    metrics=model.compiled_metrics._metrics if hasattr(
                        model, "compiled_metrics"
                    ) and model.compiled_metrics is not None else None,
                )

                # ── Callbacks por bloco ────────────────────────────
                patience = min(10, max(2, block_epochs // 2))
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=patience,
                        restore_best_weights=True,
                        verbose=0,
                    ),
                ]

                self._logger.info(
                    "Bloco %d/%d: epochs=%d, LR=%.2e, patience=%d, "
                    "trainable=%d/%d",
                    block_idx + 1,
                    n_unfreeze_steps,
                    block_epochs,
                    current_lr,
                    patience,
                    sum(1 for l in model.layers if l.trainable),
                    n_layers,
                )

                # ── Treinamento do bloco ───────────────────────────
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=block_epochs,
                    callbacks=callbacks,
                    verbose=0,
                )

                # ── Merge historico ────────────────────────────────
                block_history = history.history
                for key, values in block_history.items():
                    if key not in merged_history:
                        merged_history[key] = []
                    merged_history[key].extend(values)

                total_epochs_used += len(block_history.get("loss", []))

                # ── Decay LR para proximo bloco ────────────────────
                # D7: discriminative fine-tuning — camadas mais profundas
                # recebem LR menor (features mais genericas)
                current_lr *= _PROGRESSIVE_LR_DECAY_PER_BLOCK

            self._logger.info(
                "Progressive unfreeze concluido: %d epocas em %d blocos",
                total_epochs_used, n_unfreeze_steps,
            )

            return merged_history

        finally:
            # ── Restaurar trainable para TODAS as camadas ──────────
            for layer, was_trainable in zip(
                model.layers, original_trainable
            ):
                layer.trainable = was_trainable

            self._logger.info(
                "Progressive unfreeze: trainable restaurado para estado original"
            )
