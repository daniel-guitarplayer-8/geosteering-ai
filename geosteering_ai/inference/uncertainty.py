# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: inference/uncertainty.py                                          ║
# ║  Bloco: 7 — Inference                                                    ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • UncertaintyResult: dataclass com mean, std, CI 95% das predicoes     ║
# ║    • UncertaintyEstimator: quantificacao de incerteza via MC Dropout ou   ║
# ║      ensemble de modelos. Lazy TF import em todos os metodos.             ║
# ║    • MC Dropout: N forward passes com dropout ativo (training=True)       ║
# ║    • Ensemble: N modelos independentes, media e std das predicoes         ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig via TYPE_CHECKING), numpy        ║
# ║                TensorFlow importado LAZY dentro de cada metodo            ║
# ║  Exports: ~2 (UncertaintyResult, UncertaintyEstimator) — ver __all__     ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C67: UQ)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Quantificacao de incerteza — MC Dropout e Ensemble.

Fornece dois metodos de estimativa de incerteza para predicoes de inversao
geofisica:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  METODOS DE QUANTIFICACAO DE INCERTEZA                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  MC Dropout (Gal & Ghahramani, 2016):                                  │
    │    model(x, training=True) × N  → stack → mean, std                    │
    │    Requer dropout layers no modelo (ativo via training=True)            │
    │                                                                         │
    │  Ensemble:                                                              │
    │    [model_1(x), model_2(x), ..., model_K(x)] → stack → mean, std      │
    │    Requer K modelos treinados independentemente                         │
    │                                                                         │
    │  Saida: UncertaintyResult(mean, std, ci_lower, ci_upper, method, n)    │
    │         CI 95% = mean ± 1.96 × std (Normal assumption)                 │
    └─────────────────────────────────────────────────────────────────────────┘

Resultados no dominio log10 (nao Ohm.m) para preservar simetria gaussiana
das predicoes — desvio-padrao em log10 decades e interpretavel diretamente.

Example:
    >>> from geosteering_ai.inference.uncertainty import UncertaintyEstimator
    >>> estimator = UncertaintyEstimator(method="mc_dropout")
    >>> result = estimator.estimate(model, x_test, n_samples=50)
    >>> result.mean.shape   # (N, seq, channels)
    >>> result.ci_upper     # mean + 1.96 * std

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    TensorFlow importado LAZY dentro de cada metodo (nao no topo).
    Referenciado em:
        - inference/__init__.py: re-exportado como API publica
        - inference/pipeline.py: InferencePipeline.predict(return_uncertainty=True)
        - visualization/uncertainty.py: plot_confidence_bands recebe UncertaintyResult
        - tests/test_uncertainty.py: TestUncertaintyEstimator
    Ref: docs/ARCHITECTURE_v2.md secao 6 (Inference).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Dataclass de resultado ---
    "UncertaintyResult",
    # --- Classe principal de estimativa ---
    "UncertaintyEstimator",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de incerteza
# ──────────────────────────────────────────────────────────────────────

# Fator z para intervalo de confianca 95% (distribuicao Normal padrao)
# P(|Z| <= 1.96) = 0.95 para Z ~ N(0,1)
_Z_95 = 1.96

# Epsilon para estabilidade numerica (float32)
# NUNCA usar 1e-30 — valor correto para float32 e 1e-12
_EPS = 1e-12

# Numero default de amostras MC Dropout
# 30 forward passes: bom trade-off entre precisao e custo computacional
# (Gal & Ghahramani 2016 recomendam >= 20)
_DEFAULT_MC_SAMPLES = 30

# Metodos de incerteza validos
_VALID_METHODS = ("mc_dropout", "ensemble")


# ════════════════════════════════════════════════════════════════════════
# UNCERTAINTY RESULT — Container tipado para resultados de UQ
#
# Armazena estatisticas agregadas (mean, std, CI) de multiplas forward
# passes (MC Dropout) ou multiplos modelos (Ensemble). Todos os arrays
# tem shape identico: (N, seq_len, n_channels).
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  UncertaintyResult                                                  │
# ├──────────────┬──────────────────────────────────────────────────────┤
# │  mean        │ (N, seq, ch) — media das predicoes                  │
# │  std         │ (N, seq, ch) — desvio-padrao das predicoes          │
# │  ci_lower    │ (N, seq, ch) — limite inferior CI 95%               │
# │  ci_upper    │ (N, seq, ch) — limite superior CI 95%               │
# │  method      │ str — "mc_dropout" | "ensemble"                     │
# │  n_samples   │ int — numero de forward passes usadas               │
# └──────────────┴──────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

@dataclass
class UncertaintyResult:
    """Container para resultados de quantificacao de incerteza.

    Armazena media, desvio-padrao e intervalos de confianca 95% (CI)
    calculados a partir de multiplas predicoes (MC Dropout ou Ensemble).
    Todos os arrays compartilham o mesmo shape: ``(N, seq_len, n_channels)``.

    Attributes:
        mean: np.ndarray de shape (N, seq_len, n_channels). Media pontual
            das predicoes agregadas. Em dominio log10.
        std: np.ndarray de shape (N, seq_len, n_channels). Desvio-padrao
            das predicoes. Unidade: decadas log10 (0.1 = ~0.1 decada).
        ci_lower: np.ndarray de shape (N, seq_len, n_channels). Limite
            inferior do intervalo de confianca 95%: mean - 1.96 * std.
        ci_upper: np.ndarray de shape (N, seq_len, n_channels). Limite
            superior do intervalo de confianca 95%: mean + 1.96 * std.
        method: Nome do metodo utilizado: ``"mc_dropout"`` ou ``"ensemble"``.
        n_samples: Numero de forward passes (MC) ou modelos (Ensemble)
            usados para computar as estatisticas.

    Example:
        >>> result = UncertaintyResult(
        ...     mean=np.zeros((10, 600, 2)),
        ...     std=np.ones((10, 600, 2)) * 0.1,
        ...     ci_lower=np.zeros((10, 600, 2)) - 0.196,
        ...     ci_upper=np.zeros((10, 600, 2)) + 0.196,
        ...     method="mc_dropout",
        ...     n_samples=30,
        ... )
        >>> result.mean.shape
        (10, 600, 2)

    Note:
        CI 95% assume distribuicao Normal: mean +/- 1.96 * std.
        Resultados em dominio log10 (nao Ohm.m) para preservar simetria.
        Referenciado em:
            - inference/uncertainty.py: UncertaintyEstimator retorna UncertaintyResult
            - visualization/uncertainty.py: plot_confidence_bands recebe UncertaintyResult
        Ref: docs/ARCHITECTURE_v2.md secao 6.
    """

    mean: np.ndarray       # (N, seq_len, n_channels) — media das predicoes
    std: np.ndarray        # (N, seq_len, n_channels) — desvio-padrao
    ci_lower: np.ndarray   # (N, seq_len, n_channels) — CI 95% inferior
    ci_upper: np.ndarray   # (N, seq_len, n_channels) — CI 95% superior
    method: str            # "mc_dropout" | "ensemble"
    n_samples: int         # numero de forward passes ou modelos usados


# ════════════════════════════════════════════════════════════════════════
# UNCERTAINTY ESTIMATOR — MC Dropout e Ensemble
#
# Quantifica incerteza epistemica via dois metodos:
#   1. MC Dropout: N forward passes com training=True → dropout ativo
#   2. Ensemble: K modelos treinados, cada um faz 1 forward pass
#
# Ambos produzem um stack de predicoes → mean + std → CI 95%.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  FLUXO MC DROPOUT                                                  │
# ├─────────────────────────────────────────────────────────────────────┤
# │  x → model(x, training=True) × N → stack(N, batch, seq, ch)       │
# │      → mean(axis=0)  → (batch, seq, ch) = predicao media           │
# │      → std(axis=0)   → (batch, seq, ch) = incerteza epistemica     │
# │      → mean ± 1.96*std → CI 95%                                    │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  FLUXO ENSEMBLE                                                    │
# ├─────────────────────────────────────────────────────────────────────┤
# │  x → [m1(x), m2(x), ..., mK(x)] → stack(K, batch, seq, ch)       │
# │      → mean(axis=0) → predicao media                               │
# │      → std(axis=0)  → incerteza inter-modelo                       │
# │      → mean ± 1.96*std → CI 95%                                    │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

class UncertaintyEstimator:
    """C67: Quantificacao de incerteza via MC Dropout ou Ensemble.

    Encapsula a logica de multiplas forward passes (MC Dropout) ou
    agregacao de multiplos modelos (Ensemble) para estimar incerteza
    epistemica nas predicoes de inversao geofisica.

    Attributes:
        method: Metodo de incerteza selecionado (``"mc_dropout"`` ou
            ``"ensemble"``). Determina qual sub-metodo e invocado por
            ``estimate()``.

    Example:
        >>> from geosteering_ai.inference.uncertainty import UncertaintyEstimator
        >>> estimator = UncertaintyEstimator(method="mc_dropout")
        >>> result = estimator.estimate(model, x_test, n_samples=50)
        >>> result.mean.shape   # (N, seq, channels)
        (100, 600, 2)
        >>> result.std.mean()   # media da incerteza
        0.0812

    Note:
        TensorFlow e importado LAZY dentro de cada metodo — nao no topo
        do modulo. Isso garante compatibilidade com ambientes sem GPU
        e imports rapidos em testes.
        Referenciado em:
            - inference/__init__.py: re-exportado
            - inference/pipeline.py: InferencePipeline pode compor UncertaintyEstimator
            - visualization/uncertainty.py: plot_confidence_bands usa UncertaintyResult
        Ref: docs/ARCHITECTURE_v2.md secao 6.
    """

    # ────────────────────────────────────────────────────────────────
    # D2: Inicializacao — valida metodo de incerteza
    # ────────────────────────────────────────────────────────────────

    def __init__(self, method: str = "mc_dropout") -> None:
        """Inicializa UncertaintyEstimator com metodo selecionado.

        Args:
            method: Metodo de quantificacao de incerteza. Valores aceitos:
                - ``"mc_dropout"``: Monte Carlo Dropout (Gal & Ghahramani 2016).
                  Requer modelo com dropout layers. N forward passes com
                  ``training=True`` para ativar dropout em inferencia.
                - ``"ensemble"``: Ensemble de modelos. Requer lista de K
                  modelos treinados independentemente. Cada modelo faz 1
                  forward pass; estatisticas agregadas sobre K predicoes.
                Default: ``"mc_dropout"``.

        Raises:
            ValueError: Se ``method`` nao for ``"mc_dropout"`` ou ``"ensemble"``.

        Note:
            Referenciado em:
                - UncertaintyEstimator.estimate(): despacha para sub-metodo
            Ref: docs/ARCHITECTURE_v2.md secao 6.
        """
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Metodo invalido: {method!r}. "
                f"Valores aceitos: {_VALID_METHODS}"
            )

        self.method = method
        logger.info("UncertaintyEstimator inicializado — method=%s", method)

    # ────────────────────────────────────────────────────────────────
    # D2: MC Dropout — N forward passes com dropout ativo
    # ────────────────────────────────────────────────────────────────

    def estimate_mc_dropout(
        self,
        model: object,
        x: np.ndarray,
        *,
        n_samples: int = _DEFAULT_MC_SAMPLES,
    ) -> UncertaintyResult:
        """Estima incerteza via Monte Carlo Dropout.

        Executa ``n_samples`` forward passes com ``training=True`` para
        manter dropout ativo durante inferencia. Coleta predicoes em um
        stack e computa media, desvio-padrao e intervalos de confianca.

        Args:
            model: Modelo Keras treinado (tf.keras.Model). DEVE conter
                dropout layers para que MC Dropout produza variabilidade.
                Se nao houver dropout, todas as predicoes serao identicas
                e std sera ~0.
            x: Array de entrada com shape ``(N, seq_len, n_features)``.
                Dados ja preprocessados (FV + GS + scaled).
            n_samples: Numero de forward passes com dropout ativo.
                Mais amostras = estimativa mais precisa, porem mais lenta.
                Recomendado: >= 20 (Gal & Ghahramani 2016). Default: 30.

        Returns:
            UncertaintyResult com mean, std e CI 95% das predicoes.
            Todos os arrays tem shape ``(N, seq_len, n_channels)`` onde
            n_channels = n_targets (tipicamente 2: rho_h, rho_v).

        Raises:
            ValueError: Se ``x`` nao for 3D.
            ValueError: Se ``n_samples`` < 2.

        Note:
            TensorFlow importado LAZY dentro deste metodo.
            Predicoes no dominio log10 — std em decadas log10.
            Referenciado em:
                - UncertaintyEstimator.estimate(): dispatch mc_dropout
            Ref: Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian
                Approximation. ICML.
        """
        import tensorflow as tf  # lazy import — D7: nunca no topo

        if x.ndim != 3:
            raise ValueError(
                f"Esperado array 3D (N, seq_len, features), "
                f"recebido ndim={x.ndim}, shape={x.shape}"
            )
        if n_samples < 2:
            raise ValueError(
                f"n_samples deve ser >= 2 para estimar variancia, "
                f"recebido: {n_samples}"
            )

        logger.info(
            "MC Dropout: %d forward passes, input shape=%s",
            n_samples, x.shape,
        )

        # Converter para tensor TF float32 uma unica vez
        x_tensor = tf.constant(x, dtype=tf.float32)

        # Coletar N predicoes com dropout ativo (training=True)
        predictions_list: List[np.ndarray] = []
        for i in range(n_samples):
            # training=True ativa dropout/batch_norm em modo treinamento
            pred = model(x_tensor, training=True)  # type: ignore[operator]
            predictions_list.append(pred.numpy())

        # Stack → (n_samples, N, seq_len, n_channels)
        predictions_stack = np.stack(predictions_list, axis=0)

        # Estatisticas agregadas ao longo do eixo de amostras MC
        mean = np.mean(predictions_stack, axis=0)     # (N, seq, ch)
        std = np.std(predictions_stack, axis=0)        # (N, seq, ch)

        # Clampar std para evitar CI degenerado com std=0
        std = np.maximum(std, _EPS)

        # Intervalo de confianca 95%: mean +/- 1.96 * std
        ci_lower = mean - _Z_95 * std
        ci_upper = mean + _Z_95 * std

        logger.info(
            "MC Dropout concluido — mean_std=%.6f, "
            "mean_range=[%.4f, %.4f]",
            float(np.mean(std)),
            float(np.min(mean)),
            float(np.max(mean)),
        )

        return UncertaintyResult(
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method="mc_dropout",
            n_samples=n_samples,
        )

    # ────────────────────────────────────────────────────────────────
    # D2: Ensemble — K modelos independentes, 1 forward pass cada
    # ────────────────────────────────────────────────────────────────

    def estimate_ensemble(
        self,
        models: List[object],
        x: np.ndarray,
    ) -> UncertaintyResult:
        """Estima incerteza via Ensemble de modelos.

        Executa 1 forward pass por modelo, coleta predicoes de K modelos
        treinados independentemente, e computa media e desvio-padrao
        inter-modelo.

        Args:
            models: Lista de K modelos Keras treinados (tf.keras.Model).
                Cada modelo deve ter a mesma arquitetura de saida
                (mesmo n_targets). Recomendado K >= 3 para estimativa
                estatisticamente significativa.
            x: Array de entrada com shape ``(N, seq_len, n_features)``.
                Dados ja preprocessados (FV + GS + scaled).

        Returns:
            UncertaintyResult com mean, std e CI 95% das predicoes.
            n_samples sera len(models).

        Raises:
            ValueError: Se ``x`` nao for 3D.
            ValueError: Se ``models`` tiver menos de 2 elementos.

        Note:
            TensorFlow importado LAZY dentro deste metodo.
            Cada modelo faz 1 forward pass com ``training=False`` (modo
            inferencia — dropout desativado, batch_norm em modo eval).
            Referenciado em:
                - UncertaintyEstimator.estimate(): dispatch ensemble
            Ref: Lakshminarayanan, B. et al. (2017). Simple and Scalable
                Predictive Uncertainty Estimation using Deep Ensembles.
                NeurIPS.
        """
        import tensorflow as tf  # lazy import — D7: nunca no topo

        if x.ndim != 3:
            raise ValueError(
                f"Esperado array 3D (N, seq_len, features), "
                f"recebido ndim={x.ndim}, shape={x.shape}"
            )
        if len(models) < 2:
            raise ValueError(
                f"Ensemble requer >= 2 modelos, recebido: {len(models)}"
            )

        n_models = len(models)
        logger.info(
            "Ensemble: %d modelos, input shape=%s",
            n_models, x.shape,
        )

        # Converter para tensor TF float32 uma unica vez
        x_tensor = tf.constant(x, dtype=tf.float32)

        # Coletar predicao de cada modelo (training=False — modo inferencia)
        predictions_list: List[np.ndarray] = []
        for idx, m in enumerate(models):
            pred = m(x_tensor, training=False)  # type: ignore[operator]
            predictions_list.append(pred.numpy())
            logger.debug("Modelo %d/%d — pred shape=%s", idx + 1, n_models, pred.shape)

        # Stack → (K, N, seq_len, n_channels)
        predictions_stack = np.stack(predictions_list, axis=0)

        # Estatisticas inter-modelo
        mean = np.mean(predictions_stack, axis=0)     # (N, seq, ch)
        std = np.std(predictions_stack, axis=0)        # (N, seq, ch)

        # Clampar std para evitar CI degenerado
        std = np.maximum(std, _EPS)

        # Intervalo de confianca 95%
        ci_lower = mean - _Z_95 * std
        ci_upper = mean + _Z_95 * std

        logger.info(
            "Ensemble concluido — %d modelos, mean_std=%.6f, "
            "mean_range=[%.4f, %.4f]",
            n_models,
            float(np.mean(std)),
            float(np.min(mean)),
            float(np.max(mean)),
        )

        return UncertaintyResult(
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method="ensemble",
            n_samples=n_models,
        )

    # ────────────────────────────────────────────────────────────────
    # D2: Dispatch — delega para sub-metodo correto
    # ────────────────────────────────────────────────────────────────

    def estimate(
        self,
        model_or_models: object,
        x: np.ndarray,
        *,
        n_samples: int = _DEFAULT_MC_SAMPLES,
    ) -> UncertaintyResult:
        """Estima incerteza delegando para o metodo configurado.

        Dispatch automatico baseado em ``self.method``:
            - ``"mc_dropout"``: chama ``estimate_mc_dropout(model, x, n_samples)``
            - ``"ensemble"``:  chama ``estimate_ensemble(models, x)``

        Args:
            model_or_models: Para MC Dropout: um unico modelo Keras.
                Para Ensemble: lista de modelos Keras.
            x: Array de entrada com shape ``(N, seq_len, n_features)``.
                Dados ja preprocessados (FV + GS + scaled).
            n_samples: Numero de forward passes para MC Dropout.
                Ignorado para Ensemble (usa len(models)). Default: 30.

        Returns:
            UncertaintyResult com mean, std e CI 95%.

        Raises:
            ValueError: Se ``self.method`` for invalido (nao deveria ocorrer
                se __init__ validou corretamente).
            TypeError: Se tipo de ``model_or_models`` nao corresponder ao
                metodo selecionado.

        Example:
            >>> # MC Dropout
            >>> estimator = UncertaintyEstimator(method="mc_dropout")
            >>> result = estimator.estimate(model, x_test, n_samples=50)
            >>>
            >>> # Ensemble
            >>> estimator = UncertaintyEstimator(method="ensemble")
            >>> result = estimator.estimate([model_1, model_2, model_3], x_test)

        Note:
            Referenciado em:
                - inference/__init__.py: API publica do subpacote
                - visualization/uncertainty.py: consome UncertaintyResult
            Ref: docs/ARCHITECTURE_v2.md secao 6.
        """
        if self.method == "mc_dropout":
            # MC Dropout espera um unico modelo
            if isinstance(model_or_models, (list, tuple)):
                raise TypeError(
                    "MC Dropout espera um unico modelo, nao uma lista. "
                    "Use method='ensemble' para lista de modelos."
                )
            return self.estimate_mc_dropout(
                model_or_models, x, n_samples=n_samples,
            )

        if self.method == "ensemble":
            # Ensemble espera lista de modelos
            if not isinstance(model_or_models, (list, tuple)):
                raise TypeError(
                    "Ensemble espera uma lista de modelos, nao um unico modelo. "
                    "Use method='mc_dropout' para modelo unico."
                )
            return self.estimate_ensemble(model_or_models, x)

        # Fallback — nao deveria chegar aqui se __init__ validou
        raise ValueError(
            f"Metodo desconhecido: {self.method!r}. "
            f"Valores aceitos: {_VALID_METHODS}"
        )

    # ────────────────────────────────────────────────────────────────
    # D2: Representacao — info concisa para logging/debug
    # ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Representacao concisa do estimador para logging.

        Note:
            Exibe metodo selecionado.
        """
        return f"UncertaintyEstimator(method={self.method!r})"
