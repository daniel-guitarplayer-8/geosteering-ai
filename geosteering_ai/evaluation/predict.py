# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/predict.py                                            ║
# ║  Bloco: 8 — Evaluation                                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • PredictionResult dataclass: container para predicoes e inversas      ║
# ║    • predict_test: gera predicoes no test set com scaling inverso         ║
# ║    • Retorna tanto dominio log10 (scaled) quanto Ohm.m (linear)          ║
# ║    • Requer TensorFlow (lazy import) para model.predict()                ║
# ║                                                                            ║
# ║  Dependencias: numpy (principal), tensorflow (lazy, para model.predict)   ║
# ║  Exports: ~2 (PredictionResult, predict_test)                            ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8.4                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Predicoes no test set com scaling inverso.

Gera predicoes usando model.predict() e converte do dominio scaled (log10)
de volta para dominio linear (Ohm.m), retornando ambos em um container
PredictionResult. Suporta apenas target_scaling = "log10" (o unico scaling
validado pela errata v5.0.15).

Pipeline de conversao:

    model.predict(x_test) → y_pred_scaled (log10)
                                │
                         10^y_pred_scaled → y_pred_ohm (Ohm.m)

ERRATA: eps = 1e-12 para float32 (NUNCA 1e-30).
ERRATA: target_scaling = "log10" (NUNCA "log").

Note:
    Referenciado em:
        - evaluation/__init__.py: re-exports PredictionResult, predict_test
        - tests/test_evaluation.py: testes de predicao
    Ref: docs/ARCHITECTURE_v2.md secao 8.4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Container de predicoes ---
    "PredictionResult",
    # --- Funcao principal ---
    "predict_test",
]


# ════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════
# Epsilon seguro para float32 — protege contra underflow em conversao.
# NUNCA usar 1e-30 (causa subnormais em float32).
# Ref: Errata v5.0.15, IEEE 754 float32 min normal ~ 1.175e-38.
# ────────────────────────────────────────────────────────────────────────

EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════
# DATACLASS DE RESULTADO
#
# Container para predicoes em ambos dominios (log10 e Ohm.m), permitindo
# avaliacoes em dominio scaled (metricas) e linear (visualizacao).
#
# Ref: docs/reference/metricas.md
# ════════════════════════════════════════════════════════════════════════


@dataclass
class PredictionResult:
    """Container para predicoes e valores reais em dois dominios.

    Armazena tanto o dominio scaled (log10) quanto o dominio linear
    (Ohm.m) para facilitar avaliacoes e visualizacoes. As metricas de
    erro devem ser computadas no dominio log10 (y_true_scaled, y_pred_scaled).
    O dominio Ohm.m e util para visualizacao de perfis de resistividade.

    Attributes:
        y_true_scaled: Array (N, seq, 2) com valores reais em log10.
            Canal 0 = log10(rho_h), canal 1 = log10(rho_v).
        y_pred_scaled: Array (N, seq, 2) com predicoes em log10.
        y_true_ohm: Array (N, seq, 2) com valores reais em Ohm.m.
            Canal 0 = rho_h, canal 1 = rho_v.
        y_pred_ohm: Array (N, seq, 2) com predicoes em Ohm.m.

    Example:
        >>> result = predict_test(model, test_ds, config)
        >>> result.y_pred_scaled.shape
        (100, 600, 2)
        >>> result.y_pred_ohm.min()
        0.15

    Note:
        Referenciado em:
            - evaluation/predict.py: predict_test retorna PredictionResult
            - evaluation/metrics.py: compute_all_metrics recebe y_true_scaled
            - evaluation/advanced.py: funcoes avancadas recebem y_true_scaled
        Ref: docs/ARCHITECTURE_v2.md secao 8.4.
    """

    y_true_scaled: np.ndarray
    y_pred_scaled: np.ndarray
    y_true_ohm: np.ndarray
    y_pred_ohm: np.ndarray


# ════════════════════════════════════════════════════════════════════════
# FUNCAO PRINCIPAL — predict_test
#
# Gera predicoes no test set e converte para ambos dominios. Aceita
# tf.data.Dataset ou tupla (x_test, y_test) como input. Usa lazy import
# de TensorFlow para model.predict().
#
# Ref: docs/reference/inference.md
# ════════════════════════════════════════════════════════════════════════


def predict_test(
    model,
    test_ds,
    config: "PipelineConfig",
) -> PredictionResult:
    """Gera predicoes no test set com scaling inverso (log10 → Ohm.m).

    Aceita tf.data.Dataset (batched) ou tupla (x_test, y_test) como
    input. Extrai y_true do dataset, gera y_pred via model.predict,
    e converte ambos para dominio linear (Ohm.m) via 10^y.

    Args:
        model: Modelo Keras treinado (tf.keras.Model). Deve aceitar
            x_test como input e retornar predicoes (N, seq, 2).
        test_ds: tf.data.Dataset batched com (x, y) ou tupla (x, y)
            onde x tem shape (N, seq, n_features) e y tem shape
            (N, seq, 2) em dominio log10.
        config: PipelineConfig com metadados do pipeline. Usado para
            validar target_scaling e logging.

    Returns:
        PredictionResult com predicoes e valores reais em ambos dominios.

    Raises:
        ImportError: Se TensorFlow nao esta instalado.
        ValueError: Se target_scaling != "log10" ou shapes invalidos.

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-exports predict_test
            - tests/test_evaluation.py: testes de predicao
        Ref: docs/ARCHITECTURE_v2.md secao 8.4.
    """
    # D7: lazy import de TensorFlow — nao requerido no nivel do modulo
    try:
        import tensorflow as tf  # noqa: F811
    except ImportError as e:
        raise ImportError(
            "predict_test requer TensorFlow. "
            "Instale com: pip install tensorflow"
        ) from e

    # D7: validacao de target_scaling — DEVE ser "log10" (errata v5.0.15)
    if config.target_scaling != "log10":
        raise ValueError(
            f"predict_test suporta apenas target_scaling='log10', "
            f"recebido '{config.target_scaling}'. "
            f"Errata v5.0.15: TARGET_SCALING DEVE ser 'log10'."
        )

    # D7: extrai x e y do dataset (suporta tf.data.Dataset e tupla)
    if isinstance(test_ds, tf.data.Dataset):
        # Itera sobre batches e concatena
        x_batches = []
        y_batches = []
        for x_batch, y_batch in test_ds:
            x_batches.append(np.asarray(x_batch))
            y_batches.append(np.asarray(y_batch))
        x_test = np.concatenate(x_batches, axis=0)
        y_true_scaled = np.concatenate(y_batches, axis=0)
    elif isinstance(test_ds, (tuple, list)) and len(test_ds) == 2:
        x_test = np.asarray(test_ds[0])
        y_true_scaled = np.asarray(test_ds[1])
    else:
        raise ValueError(
            "test_ds deve ser tf.data.Dataset ou tupla (x, y). "
            f"Recebido: {type(test_ds)}"
        )

    # D7: validacao de shapes
    if x_test.ndim != 3:
        raise ValueError(
            f"Shape esperado (N, seq, n_features) para x_test, "
            f"recebido {x_test.shape}"
        )
    if y_true_scaled.ndim != 3 or y_true_scaled.shape[-1] != 2:
        raise ValueError(
            f"Shape esperado (N, seq, 2) para y_true, "
            f"recebido {y_true_scaled.shape}"
        )

    logger.info(
        "predict_test: gerando predicoes para %d amostras, "
        "seq_len=%d, n_features=%d",
        x_test.shape[0],
        x_test.shape[1],
        x_test.shape[2],
    )

    # D7: model.predict gera predicoes em dominio log10
    y_pred_scaled = model.predict(x_test, verbose=0)
    y_pred_scaled = np.asarray(y_pred_scaled, dtype=np.float64)
    y_true_scaled = np.asarray(y_true_scaled, dtype=np.float64)

    # D7: conversao log10 → linear (Ohm.m): rho = 10^y
    y_true_ohm = np.power(10.0, y_true_scaled)
    y_pred_ohm = np.power(10.0, y_pred_scaled)

    logger.info(
        "predict_test: predicoes geradas — "
        "y_pred_scaled range=[%.4f, %.4f], "
        "y_pred_ohm range=[%.4f, %.4f]",
        float(np.min(y_pred_scaled)),
        float(np.max(y_pred_scaled)),
        float(np.min(y_pred_ohm)),
        float(np.max(y_pred_ohm)),
    )

    return PredictionResult(
        y_true_scaled=y_true_scaled,
        y_pred_scaled=y_pred_scaled,
        y_true_ohm=y_true_ohm,
        y_pred_ohm=y_pred_ohm,
    )
