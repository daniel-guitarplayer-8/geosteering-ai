# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: training/metrics.py                                              ║
# ║  Bloco: 6 — Training                                                     ║
# ║                                                                           ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning    ║
# ║  Autor: Daniel Leal                                                       ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)        ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec)║
# ║  Pacote: geosteering_ai (pip installable)                                ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                 ║
# ║                                                                           ║
# ║  Proposito:                                                               ║
# ║    • 3 Keras custom metrics: R2Score, PerComponentMetric,                ║
# ║      AnisotropyRatioError                                                ║
# ║    • 5 funcoes numpy re-exportadas de evaluation.metrics (DRY):         ║
# ║      compute_r2, compute_rmse, compute_mae, compute_mbe, compute_mape  ║
# ║    • build_metrics(config) — factory de metricas para model.compile()   ║
# ║                                                                           ║
# ║  Dependencias: config.py (PipelineConfig via TYPE_CHECKING)              ║
# ║  Exports: 9 simbolos — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 7.2 (metricas de avaliacao)         ║
# ║                                                                           ║
# ║  Historico:                                                               ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Metricas Keras e numpy para avaliacao do pipeline de inversao.

3 custom Keras metrics para monitoramento durante treinamento
(model.compile) e 5 funcoes numpy para avaliacao pos-treinamento.

Metricas Keras:
    - R2Score: coeficiente de determinacao global (R^2).
    - PerComponentMetric: RMSE por canal de saida (rho_h ou rho_v).
    - AnisotropyRatioError: erro medio na razao de anisotropia rho_v/rho_h.

Funcoes numpy:
    - compute_r2: R^2 sobre arrays completos.
    - compute_rmse: Root Mean Squared Error.
    - compute_mae: Mean Absolute Error.
    - compute_mbe: Mean Bias Error (detecta bias sistematico).
    - compute_mape: Mean Absolute Percentage Error.

Factory:
    - build_metrics(config): retorna lista de metricas para model.compile().

Example:
    >>> from geosteering_ai.training.metrics import build_metrics, compute_rmse
    >>> metrics = build_metrics(config)
    >>> model.compile(optimizer="adam", loss="mse", metrics=metrics)
    >>> rmse_val = compute_rmse(y_true, y_pred)

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Lazy TF import: ``import tensorflow as tf`` dentro de classes/funcoes,
    NAO no topo do modulo (permite uso CPU-only das funcoes numpy).
    eps = 1e-12 para float32 (NUNCA 1e-30 — Errata v5.0.15).
    Referenciado em:
        - training/__init__.py: re-export de build_metrics
        - config.py: output_channels, verbose (parametros de build_metrics)
    Ref: docs/ARCHITECTURE_v2.md secao 7.2 (metricas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ────────────────────────────────────────────────────────────────────────
# DRY: funcoes numpy de avaliacao importadas de evaluation.metrics
# (implementacao canonica). Re-exportadas via __all__ para manter
# backward compatibility com consumidores de training.metrics.
# ────────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.metrics import (  # noqa: E402
    compute_r2,
    compute_rmse,
    compute_mae,
    compute_mbe,
    compute_mape,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Keras custom metrics ---
    "R2Score",
    "PerComponentMetric",
    "AnisotropyRatioError",
    # --- Funcoes numpy de avaliacao ---
    "compute_r2",
    "compute_rmse",
    "compute_mae",
    "compute_mbe",
    "compute_mape",
    # --- Factory ---
    "build_metrics",
]

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════════
# Epsilon seguro para float32 — protege contra divisao por zero.
# NUNCA usar 1e-30 (causa subnormais em float32, gradientes explodidos).
# Ref: Errata v5.0.15, IEEE 754 float32 min normal ≈ 1.175e-38.
# ──────────────────────────────────────────────────────────────────────────

EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════════
# SECAO: KERAS CUSTOM METRICS
# ════════════════════════════════════════════════════════════════════════════
# Metricas customizadas que herdam de tf.keras.metrics.Metric para
# integrar com model.compile() e model.fit(). Cada metrica acumula
# estatisticas por batch via update_state() e computa o resultado
# final via result(). reset_state() limpa acumuladores entre epocas.
#
# Todas usam lazy TF import (import tensorflow as tf DENTRO da classe)
# para permitir importacao CPU-only do modulo (funcoes numpy puras).
#
# Metricas disponiveis:
#   ┌──────────────────────────┬───────────────────────────────────────────┐
#   │ Metrica                  │ Formula                                   │
#   ├──────────────────────────┼───────────────────────────────────────────┤
#   │ R2Score                  │ 1 - SSE / (SST + eps)                    │
#   │ PerComponentMetric       │ sqrt(MSE_i + eps) para canal i           │
#   │ AnisotropyRatioError     │ mean|ratio_true - ratio_pred|            │
#   └──────────────────────────┴───────────────────────────────────────────┘
#
# Ref: docs/ARCHITECTURE_v2.md secao 7.2, Keras custom metrics guide.
# ──────────────────────────────────────────────────────────────────────────


class R2Score:
    """Coeficiente de determinacao R^2 como metrica Keras.

    Acumula SSE (Sum of Squared Errors) e SST (Sum of Squared Total)
    ao longo dos batches e computa R^2 = 1 - SSE / (SST + eps) no
    final da epoca. Valores proximos de 1.0 indicam bom ajuste;
    valores negativos indicam pior que media.

    A classe e construida dinamicamente como subclasse de
    ``tf.keras.metrics.Metric`` no __init_subclass__ — isso permite
    lazy import do TensorFlow.

    Attributes:
        sum_sq_error (tf.Variable): Acumulador SSE (soma dos erros ao
            quadrado entre y_true e y_pred).
        sum_sq_total (tf.Variable): Acumulador SST (soma dos desvios
            ao quadrado de y_true em relacao a media global).

    Example:
        >>> metric = R2Score()
        >>> metric.update_state(y_true, y_pred)
        >>> r2 = metric.result().numpy()
        >>> assert -1.0 <= r2 <= 1.0 or r2 < -1.0  # pode ser < -1

    Note:
        SSE = sum((y_true - y_pred)^2), SST = sum((y_true - mean(y_true))^2).
        eps = 1e-12 para evitar divisao por zero quando SST ~ 0.
        Referenciado em:
            - build_metrics(): sempre incluido na lista de metricas.
        Ref: docs/ARCHITECTURE_v2.md secao 7.2.
    """

    def __init__(self, name: str = "r2_score", **kwargs):
        """Inicializa R2Score com acumuladores zerados.

        Args:
            name: Nome da metrica exibido no training log.
            **kwargs: Argumentos adicionais para tf.keras.metrics.Metric.

        Note:
            Lazy TF import: tensorflow importado apenas neste metodo.
        """
        import tensorflow as tf  # Lazy import — CPU-only safe

        # Heranca dinamica para evitar import de tf no topo do modulo
        if not isinstance(self, tf.keras.metrics.Metric):
            # Re-instancia como subclasse de Metric
            self.__class__ = type(
                "R2Score",
                (tf.keras.metrics.Metric,),
                {
                    "update_state": R2Score.update_state,
                    "result": R2Score.result,
                    "reset_state": R2Score.reset_state,
                },
            )
            tf.keras.metrics.Metric.__init__(self, name=name, **kwargs)

        # Acumuladores para SSE e SST
        self.sum_sq_error = self.add_weight(
            name="sum_sq_error", initializer="zeros"
        )
        self.sum_sq_total = self.add_weight(
            name="sum_sq_total", initializer="zeros"
        )

        logger.debug("R2Score metric inicializada (name=%s)", name)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Acumula SSE e SST para o batch atual.

        Args:
            y_true: Tensor de valores verdadeiros. Shape (batch, seq, channels)
                ou (batch, channels).
            y_pred: Tensor de predicoes. Mesmo shape que y_true.
            sample_weight: Pesos por amostra (opcional, nao implementado).

        Note:
            SSE += sum((y_true - y_pred)^2) sobre todo o batch.
            SST += sum((y_true - mean(y_true))^2) sobre todo o batch.
            mean(y_true) e calculada POR BATCH (aproximacao valida para
            batches grandes ou quando a media global e estavel).
        """
        import tensorflow as tf  # Lazy import

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # SSE: soma dos erros quadraticos (pred vs true)
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        # SST: soma dos desvios quadraticos (true vs media do batch)
        y_mean = tf.reduce_mean(y_true)
        ss_tot = tf.reduce_sum(tf.square(y_true - y_mean))

        self.sum_sq_error.assign_add(ss_res)
        self.sum_sq_total.assign_add(ss_tot)

    def result(self):
        """Computa R^2 = 1 - SSE / (SST + eps).

        Returns:
            tf.Tensor: Escalar R^2. Proximo de 1.0 = bom ajuste,
            0.0 = prediz a media, negativo = pior que media.

        Note:
            eps = 1e-12 protege contra divisao por zero.
        """
        return 1.0 - self.sum_sq_error / (self.sum_sq_total + EPS)

    def reset_state(self):
        """Zera acumuladores SSE e SST para proxima epoca.

        Note:
            Chamado automaticamente pelo Keras no inicio de cada epoca
            (ou manualmente via metric.reset_states()).
        """
        self.sum_sq_error.assign(0.0)
        self.sum_sq_total.assign(0.0)


class PerComponentMetric:
    """RMSE por canal de saida (rho_h ou rho_v) como metrica Keras.

    Monitora a performance do modelo para cada componente de resistividade
    individualmente. Canal 0 = rho_h (horizontal), canal 1 = rho_v (vertical).
    Util para diagnosticar se o modelo esta sub-performando em uma componente
    especifica (ex: rho_v geralmente mais dificil de inverter).

    Acumula MSE (Mean Squared Error) do canal selecionado e retorna
    sqrt(MSE + eps) via result().

    Attributes:
        component_idx (int): Indice do canal monitorado (0=rho_h, 1=rho_v).
        sum_sq (tf.Variable): Acumulador da soma dos erros quadraticos.
        count (tf.Variable): Contador de amostras acumuladas.

    Example:
        >>> rmse_rh = PerComponentMetric(0, name="rmse_rh")
        >>> rmse_rv = PerComponentMetric(1, name="rmse_rv")
        >>> rmse_rh.update_state(y_true, y_pred)
        >>> print(rmse_rh.result().numpy())  # RMSE do canal 0

    Note:
        Para output_channels=2: canal 0 = rho_h, canal 1 = rho_v.
        Para output_channels=4/6: canais adicionais (DTB, incerteza).
        Referenciado em:
            - build_metrics(): sempre incluido para canais 0 e 1.
        Ref: docs/ARCHITECTURE_v2.md secao 7.2.
    """

    def __init__(self, component_idx: int, name: str = None, **kwargs):
        """Inicializa PerComponentMetric para um canal especifico.

        Args:
            component_idx: Indice do canal de saida a monitorar.
                0 = rho_h (resistividade horizontal).
                1 = rho_v (resistividade vertical).
            name: Nome da metrica. Se None, auto-gera "rmse_c{idx}".
            **kwargs: Argumentos adicionais para tf.keras.metrics.Metric.

        Raises:
            ValueError: Se component_idx < 0.

        Note:
            Lazy TF import: tensorflow importado apenas neste metodo.
        """
        import tensorflow as tf  # Lazy import — CPU-only safe

        if component_idx < 0:
            raise ValueError(
                f"component_idx deve ser >= 0, recebido: {component_idx}"
            )

        self.component_idx = component_idx

        if name is None:
            name = f"rmse_c{component_idx}"

        # Heranca dinamica para evitar import de tf no topo do modulo
        if not isinstance(self, tf.keras.metrics.Metric):
            self.__class__ = type(
                "PerComponentMetric",
                (tf.keras.metrics.Metric,),
                {
                    "update_state": PerComponentMetric.update_state,
                    "result": PerComponentMetric.result,
                    "reset_state": PerComponentMetric.reset_state,
                    "get_config": PerComponentMetric.get_config,
                },
            )
            tf.keras.metrics.Metric.__init__(self, name=name, **kwargs)

        # Acumuladores para MSE por componente
        self.sum_sq = self.add_weight(
            name="sum_sq", initializer="zeros"
        )
        self.count = self.add_weight(
            name="count", initializer="zeros"
        )

        logger.debug(
            "PerComponentMetric inicializada (idx=%d, name=%s)",
            component_idx, name,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Acumula erro quadratico para o canal selecionado.

        Args:
            y_true: Tensor de valores verdadeiros. Ultimo eixo = channels.
            y_pred: Tensor de predicoes. Mesmo shape que y_true.
            sample_weight: Pesos por amostra (opcional, nao implementado).

        Note:
            Extrai y_true[..., idx] e y_pred[..., idx] para o canal
            selecionado e acumula sum((true_i - pred_i)^2) e count.
        """
        import tensorflow as tf  # Lazy import

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Extrai canal especifico (ultimo eixo = channels)
        true_i = y_true[..., self.component_idx]
        pred_i = y_pred[..., self.component_idx]

        # Acumula erro quadratico e contagem
        batch_sq = tf.reduce_sum(tf.square(true_i - pred_i))
        batch_count = tf.cast(tf.size(true_i), tf.float32)

        self.sum_sq.assign_add(batch_sq)
        self.count.assign_add(batch_count)

    def result(self):
        """Computa RMSE = sqrt(MSE + eps) para o canal monitorado.

        Returns:
            tf.Tensor: Escalar RMSE do canal. Quanto menor, melhor.

        Note:
            eps = 1e-12 garante estabilidade numerica quando MSE ~ 0.
        """
        import tensorflow as tf  # Lazy import

        mse = self.sum_sq / (self.count + EPS)
        return tf.sqrt(mse + EPS)

    def reset_state(self):
        """Zera acumuladores para proxima epoca.

        Note:
            Chamado automaticamente pelo Keras no inicio de cada epoca.
        """
        self.sum_sq.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Serializa configuracao para save/load do modelo.

        Returns:
            dict: Configuracao com component_idx para reconstrucao.

        Note:
            Necessario para tf.keras.models.load_model() com custom metrics.
        """
        config = {"component_idx": self.component_idx, "name": self.name}
        return config


class AnisotropyRatioError:
    """Erro na razao de anisotropia rho_v/rho_h como metrica Keras.

    A razao de anisotropia lambda = rho_v / rho_h e um parametro critico
    em geosteering: lambda > 1 indica anisotropia vertical (folhelhos),
    lambda ~ 1 indica isotropia (arenitos). Erro nesta razao impacta
    diretamente a identificacao de limites de camada.

    Acumula |ratio_true - ratio_pred| ao longo dos batches e retorna
    a media via result().

    Attributes:
        sum_abs_error (tf.Variable): Acumulador do erro absoluto total
            na razao de anisotropia.
        count (tf.Variable): Contador de amostras acumuladas.

    Example:
        >>> metric = AnisotropyRatioError()
        >>> metric.update_state(y_true, y_pred)
        >>> print(metric.result().numpy())  # MAE da razao

    Note:
        Canal 0 = rho_h (denominador), canal 1 = rho_v (numerador).
        eps = 1e-12 protege contra divisao por zero em rho_h ~ 0.
        Ativada apenas quando output_channels >= 2.
        Referenciado em:
            - build_metrics(): incluido quando output_channels >= 2.
        Ref: docs/ARCHITECTURE_v2.md secao 7.2.
    """

    def __init__(self, name: str = "anisotropy_ratio_error", **kwargs):
        """Inicializa AnisotropyRatioError com acumuladores zerados.

        Args:
            name: Nome da metrica exibido no training log.
            **kwargs: Argumentos adicionais para tf.keras.metrics.Metric.

        Note:
            Lazy TF import: tensorflow importado apenas neste metodo.
        """
        import tensorflow as tf  # Lazy import — CPU-only safe

        # Heranca dinamica para evitar import de tf no topo do modulo
        if not isinstance(self, tf.keras.metrics.Metric):
            self.__class__ = type(
                "AnisotropyRatioError",
                (tf.keras.metrics.Metric,),
                {
                    "update_state": AnisotropyRatioError.update_state,
                    "result": AnisotropyRatioError.result,
                    "reset_state": AnisotropyRatioError.reset_state,
                },
            )
            tf.keras.metrics.Metric.__init__(self, name=name, **kwargs)

        # Acumuladores para erro na razao de anisotropia
        self.sum_abs_error = self.add_weight(
            name="sum_abs_error", initializer="zeros"
        )
        self.count = self.add_weight(
            name="count", initializer="zeros"
        )

        logger.debug("AnisotropyRatioError metric inicializada (name=%s)", name)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Acumula erro absoluto na razao rho_v/rho_h.

        Args:
            y_true: Tensor de valores verdadeiros. Shape (..., channels)
                onde canal 0 = rho_h, canal 1 = rho_v.
            y_pred: Tensor de predicoes. Mesmo shape que y_true.
            sample_weight: Pesos por amostra (opcional, nao implementado).

        Note:
            ratio = rho_v / (rho_h + eps).
            erro = |ratio_true - ratio_pred|.
            eps = 1e-12 no denominador para estabilidade numerica.
        """
        import tensorflow as tf  # Lazy import

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Extrai rho_h (canal 0) e rho_v (canal 1)
        rho_h_true = y_true[..., 0]
        rho_v_true = y_true[..., 1]
        rho_h_pred = y_pred[..., 0]
        rho_v_pred = y_pred[..., 1]

        # Razao de anisotropia: lambda = rho_v / rho_h
        ratio_true = rho_v_true / (rho_h_true + EPS)
        ratio_pred = rho_v_pred / (rho_h_pred + EPS)

        # Erro absoluto na razao
        abs_error = tf.reduce_sum(tf.abs(ratio_true - ratio_pred))
        batch_count = tf.cast(tf.size(rho_h_true), tf.float32)

        self.sum_abs_error.assign_add(abs_error)
        self.count.assign_add(batch_count)

    def result(self):
        """Computa MAE da razao de anisotropia.

        Returns:
            tf.Tensor: Escalar mean|ratio_true - ratio_pred|.
            Quanto menor, melhor a preservacao da anisotropia.

        Note:
            eps = 1e-12 no denominador para evitar divisao por zero.
        """
        return self.sum_abs_error / (self.count + EPS)

    def reset_state(self):
        """Zera acumuladores para proxima epoca.

        Note:
            Chamado automaticamente pelo Keras no inicio de cada epoca.
        """
        self.sum_abs_error.assign(0.0)
        self.count.assign(0.0)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FACTORY DE METRICAS (build_metrics)
# ════════════════════════════════════════════════════════════════════════════
# Constroi a lista de metricas Keras para model.compile() baseado na
# configuracao do pipeline. Metricas incluidas:
#
#   SEMPRE:
#     - R2Score (global)
#     - PerComponentMetric(0, "rmse_rh")  →  RMSE rho_h
#     - PerComponentMetric(1, "rmse_rv")  →  RMSE rho_v
#
#   CONDICIONAL (output_channels >= 2):
#     - AnisotropyRatioError  →  erro na razao rho_v/rho_h
#
#   CONDICIONAL (config.verbose):
#     - tf.keras.metrics.MeanAbsoluteError  →  MAE global
#
# Ref: docs/ARCHITECTURE_v2.md secao 7.2 (build_metrics factory).
# ──────────────────────────────────────────────────────────────────────────


def build_metrics(config: PipelineConfig) -> List:
    """Constroi lista de metricas Keras para model.compile().

    Seleciona metricas baseado na configuracao do pipeline:
    - Sempre inclui R2Score, RMSE por componente (rho_h, rho_v).
    - AnisotropyRatioError se output_channels >= 2.
    - MAE adicional se config.verbose (logging detalhado).

    Args:
        config: Configuracao do pipeline. Atributos utilizados:
            - output_channels (int): numero de canais de saida (2, 4, 6).
            - verbose (bool): se True, adiciona MAE global.

    Returns:
        Lista de instancias tf.keras.metrics.Metric prontas para
        model.compile(metrics=...).

    Example:
        >>> config = PipelineConfig.baseline()
        >>> metrics = build_metrics(config)
        >>> len(metrics)  # R2 + rmse_rh + rmse_rv + anisotropy = 4
        4
        >>> model.compile(optimizer="adam", loss="mse", metrics=metrics)

    Note:
        Lazy TF import: tensorflow importado apenas dentro desta funcao.
        PipelineConfig e acessado via TYPE_CHECKING (sem import circular).
        Referenciado em:
            - training/loop.py: TrainingLoop.compile_model().
        Ref: docs/ARCHITECTURE_v2.md secao 7.2.
    """
    import tensorflow as tf  # Lazy import — CPU-only safe

    metrics: List = []

    # --- Sempre: R2Score global ---
    metrics.append(R2Score(name="r2_score"))
    logger.info("Metrica adicionada: R2Score")

    # --- Sempre: RMSE por componente (rho_h e rho_v) ---
    metrics.append(PerComponentMetric(0, name="rmse_rh"))
    metrics.append(PerComponentMetric(1, name="rmse_rv"))
    logger.info("Metricas adicionadas: rmse_rh (canal 0), rmse_rv (canal 1)")

    # --- Condicional: AnisotropyRatioError (requer 2+ canais) ---
    if config.output_channels >= 2:
        metrics.append(AnisotropyRatioError(name="anisotropy_ratio_error"))
        logger.info(
            "Metrica adicionada: AnisotropyRatioError "
            "(output_channels=%d >= 2)", config.output_channels,
        )

    # --- Condicional: MAE global (modo verbose) ---
    if config.verbose:
        metrics.append(tf.keras.metrics.MeanAbsoluteError(name="mae"))
        logger.info(
            "Metrica adicionada: MeanAbsoluteError (verbose=%s)",
            config.verbose,
        )

    logger.info(
        "build_metrics: %d metricas construidas para output_channels=%d, "
        "verbose=%s",
        len(metrics), config.output_channels, config.verbose,
    )

    return metrics
