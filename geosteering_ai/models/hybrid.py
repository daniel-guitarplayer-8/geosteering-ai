# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/hybrid.py                                                  ║
# ║  Bloco: 3e — Arquiteturas Hibridas (CNN+LSTM)                            ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • CNN_LSTM: extrator CNN seguido de LSTM para modelagem temporal       ║
# ║    • CNN_BiLSTM_ED: encoder-decoder com CNN + BiLSTM bidirecional        ║
# ║    • Combina receptive field local (CNN) + memoria longa (LSTM)           ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: 2 funcoes — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.5, legado C32                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Arquiteturas hibridas CNN+LSTM para inversao resistividade.

CNN_LSTM: conv1D features → LSTM temporal modeling.
CNN_BiLSTM_ED: encoder CNN + BiLSTM + decoder CNN (acausal only).

Note:
    Referenciado em:
        - models/registry.py: _REGISTRY['CNN_LSTM'], _REGISTRY['CNN_BiLSTM_ED']
        - tests/test_models.py: TestHybrid
    Legado C32 (978 linhas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CNN_LSTM
# ════════════════════════════════════════════════════════════════════════════
# CNN extrai features locais; LSTM captura dependencias temporais longas.
# Padrao: 3-4 Conv1D (encoder) → 2-3 LSTM → projecao.
# Causalmente compativel (LSTM forward + Conv causal).
# ──────────────────────────────────────────────────────────────────────────


def build_cnn_lstm(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi CNN_LSTM: extrator CNN + modelagem temporal LSTM.

    3 Conv1D capturam padroes locais; 2 LSTM modelam dependencias
    de longo alcance ao longo dos 600 pontos de medicao.

    Arquitetura:
        Input → CNN(32) → CNN(64) → CNN(128) → LSTM(128) → LSTM(64) → Out

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: CNN_LSTM seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['CNN_LSTM']
            - tests/test_models.py: TestHybrid.test_cnn_lstm_forward
        CNN e compativel com causal_mode; LSTM e nativo causal.
        Legado C32 build_cnn_lstm().
    """
    import tensorflow as tf
    from geosteering_ai.models.blocks import output_projection

    ap = config.arch_params or {}
    cnn_filters = ap.get("cnn_filters", [32, 64, 128])
    lstm_units = ap.get("lstm_units", [128, 64])
    kernel_size = ap.get("kernel_size", 3)
    dr = config.dropout_rate
    causal = config.use_causal_mode
    pad = "causal" if causal else "same"
    l2 = config.l2_weight if config.use_l2_regularization else 0.0
    reg = tf.keras.regularizers.L2(l2) if l2 > 0.0 else None

    logger.info(
        "build_cnn_lstm: n_feat=%d, cnn=%s, lstm=%s",
        config.n_features, cnn_filters, lstm_units,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    # ── CNN encoder ───────────────────────────────────────────────────
    for n_filt in cnn_filters:
        x = tf.keras.layers.Conv1D(
            n_filt, kernel_size, padding=pad,
            kernel_regularizer=reg, use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        if dr > 0.0:
            x = tf.keras.layers.Dropout(dr)(x)

    # ── LSTM temporal modeling ────────────────────────────────────────
    for units in lstm_units:
        x = tf.keras.layers.LSTM(
            units, return_sequences=True,
            dropout=dr, kernel_regularizer=reg,
        )(x)

    out = output_projection(
        x, config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="CNN_LSTM")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CNN_BILSTM_ED — ENCODER-DECODER
# ════════════════════════════════════════════════════════════════════════════
# Encoder: CNN comprime features → BiLSTM modela dependencias bidirecionais.
# Decoder: segundo CNN de-comprime e projeta para output_channels.
# CAUSAL_INCOMPATIBLE: BiLSTM usa dados futuros.
# ──────────────────────────────────────────────────────────────────────────


def build_cnn_bilstm_ed(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi CNN_BiLSTM_ED: encoder (CNN+BiLSTM) + decoder (CNN).

    CAUSAL_INCOMPATIBLE: BiLSTM bidirecional usa dados futuros.
    Melhor performance offline para datasets grandes.

    Arquitetura:
        Input → CNN_enc → BiLSTM → CNN_dec → Output

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: CNN_BiLSTM_ED seq2seq acausal.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['CNN_BiLSTM_ED']
            - tests/test_models.py: TestHybrid.test_cnn_bilstm_ed_forward
        CAUSAL_INCOMPATIBLE — apenas inference_mode='offline'.
        Legado C32 build_cnn_bilstm_ed().
    """
    import tensorflow as tf
    from geosteering_ai.models.blocks import output_projection

    ap = config.arch_params or {}
    enc_filters = ap.get("enc_filters", [32, 64])
    bilstm_units = ap.get("bilstm_units", 128)
    dec_filters = ap.get("dec_filters", [64, 32])
    kernel_size = ap.get("kernel_size", 3)
    dr = config.dropout_rate
    l2 = config.l2_weight if config.use_l2_regularization else 0.0
    reg = tf.keras.regularizers.L2(l2) if l2 > 0.0 else None

    if config.use_causal_mode:
        logger.warning("CNN_BiLSTM_ED e CAUSAL_INCOMPATIBLE. use_causal_mode ignorado.")

    logger.info(
        "build_cnn_bilstm_ed: n_feat=%d, enc=%s, bilstm=%d, dec=%s",
        config.n_features, enc_filters, bilstm_units, dec_filters,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    # ── Encoder CNN ───────────────────────────────────────────────────
    for n_filt in enc_filters:
        x = tf.keras.layers.Conv1D(
            n_filt, kernel_size, padding="same",
            kernel_regularizer=reg, use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

    # ── BiLSTM bidirecional ───────────────────────────────────────────
    lstm = tf.keras.layers.LSTM(bilstm_units, return_sequences=True, dropout=dr)
    x = tf.keras.layers.Bidirectional(lstm, merge_mode="concat")(x)
    if dr > 0.0:
        x = tf.keras.layers.Dropout(dr)(x)

    # ── Decoder CNN ───────────────────────────────────────────────────
    for n_filt in dec_filters:
        x = tf.keras.layers.Conv1D(
            n_filt, kernel_size, padding="same",
            kernel_regularizer=reg, use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

    out = output_projection(
        x, config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="CNN_BiLSTM_ED")


__all__ = ["build_cnn_lstm", "build_cnn_bilstm_ed"]
