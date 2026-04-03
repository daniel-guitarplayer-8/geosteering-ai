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
# ║  Exports: 3 funções — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.5, legado C32                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementação inicial (CNN_LSTM, CNN_BiLSTM_ED)   ║
# ║    v2.0.1 (2026-04) — +ResNeXt_LSTM (3 arquiteturas)                    ║
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
    de longo alcance ao longo dos pontos de medição (config.sequence_length).

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
        config.n_features,
        cnn_filters,
        lstm_units,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    # ── CNN encoder ───────────────────────────────────────────────────
    for n_filt in cnn_filters:
        x = tf.keras.layers.Conv1D(
            n_filt,
            kernel_size,
            padding=pad,
            kernel_regularizer=reg,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        if dr > 0.0:
            x = tf.keras.layers.Dropout(dr)(x)

    # ── LSTM temporal modeling ────────────────────────────────────────
    for units in lstm_units:
        x = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            dropout=dr,
            kernel_regularizer=reg,
        )(x)

    out = output_projection(
        x,
        config.output_channels,
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
        config.n_features,
        enc_filters,
        bilstm_units,
        dec_filters,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    # ── Encoder CNN ───────────────────────────────────────────────────
    for n_filt in enc_filters:
        x = tf.keras.layers.Conv1D(
            n_filt,
            kernel_size,
            padding="same",
            kernel_regularizer=reg,
            use_bias=False,
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
            n_filt,
            kernel_size,
            padding="same",
            kernel_regularizer=reg,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="CNN_BiLSTM_ED")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: RESNEXT_LSTM — HIBRIDO RESNEXT + LSTM
# ════════════════════════════════════════════════════════════════════════════
# Combina ResNeXt (grouped convolutions para extracao multi-escala de
# features locais) com LSTM (modelagem de dependencias de longo alcance).
#
# ResNeXt captura padroes espaciais multi-escala nas componentes EM
# (alta/media/baixa frequencia) via cardinalidade C=32, enquanto LSTM
# modela a evolução temporal ao longo dos pontos de profundidade (seq_len).
#
# Causal-compatible: LSTM eh nativo forward-only; Conv1D usa padding='causal'.
# Vantagem sobre CNN_LSTM: grouped convolutions sao mais eficientes e
# capturam melhor features multi-escala do tensor EM (Hxx/Hzz/Hxz/Hzx).
#
# Ref: Xie et al. (2017) CVPR — ResNeXt.
#      Hochreiter & Schmidhuber (1997) — LSTM.
# ──────────────────────────────────────────────────────────────────────────


def build_resnext_lstm(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi ResNeXt_LSTM: extrator ResNeXt + modelagem temporal LSTM.

    3 blocos ResNeXt bottleneck (grouped conv) capturam features
    multi-escala, seguidos por 2 camadas LSTM para dependencias
    temporais ao longo do perfil de resistividade.

    Arquitetura:
      ┌──────────────────────────────────────────────────────────────┐
      │  Input (B, seq_len, n_features)                             │
      │    ↓                                                        │
      │  ResNeXtBlock(64,  C=32, d=4, k=3) — features locais       │
      │  ResNeXtBlock(128, C=32, d=4, k=3) — features multi-escala │
      │  ResNeXtBlock(256, C=32, d=4, k=3) — features abstratas    │
      │    ↓                                                        │
      │  LSTM(128, return_sequences=True) — temporal longo          │
      │  LSTM(64,  return_sequences=True) — temporal refinado       │
      │    ↓                                                        │
      │  Output: Dense(output_channels, 'linear')                   │
      │  Output (B, seq_len, output_channels)                       │
      └──────────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com:
            - n_features, sequence_length, output_channels
            - use_causal_mode, dropout_rate
            - arch_params: override granular:
                - resnext_filters (list, default [64, 128, 256])
                - lstm_units (list, default [128, 64])
                - cardinality (int, default 32)
                - group_width (int, default 4)
                - kernel_size (int, default 3)

    Returns:
        tf.keras.Model: ResNeXt_LSTM seq2seq hibrido.

    Example:
        >>> config = PipelineConfig(model_type="ResNeXt_LSTM")
        >>> model = build_resnext_lstm(config)
        >>> assert model.output_shape == (None, 600, 2)

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['ResNeXt_LSTM']
            - tests/test_models.py: TestHybrid.test_resnext_lstm_forward
        Causal mode: Conv1D usa padding='causal'; LSTM eh nativo causal.
        Ref: Xie et al. (2017) CVPR + Hochreiter & Schmidhuber (1997).
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import output_projection

    ap = config.arch_params or {}
    resnext_filters = ap.get("resnext_filters", [64, 128, 256])
    lstm_units = ap.get("lstm_units", [128, 64])
    cardinality = ap.get("cardinality", 32)
    group_width = ap.get("group_width", 4)
    kernel_size = ap.get("kernel_size", 3)
    dr = config.dropout_rate
    causal = config.use_causal_mode
    pad = "causal" if causal else "same"
    l2 = config.l2_weight if config.use_l2_regularization else 0.0
    reg = tf.keras.regularizers.L2(l2) if l2 > 0.0 else None

    logger.info(
        "build_resnext_lstm: n_feat=%d, resnext=%s, lstm=%s, C=%d",
        config.n_features,
        resnext_filters,
        lstm_units,
        cardinality,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    # ── ResNeXt encoder (grouped convolution bottleneck) ──────────────
    for n_filt in resnext_filters:
        intermediate = cardinality * group_width
        skip = x

        # Bottleneck: 1×1 → grouped k×1 → 1×1
        x = tf.keras.layers.Conv1D(
            intermediate,
            1,
            padding="same",
            kernel_regularizer=reg,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv1D(
            intermediate,
            kernel_size,
            padding=pad,
            groups=cardinality,
            kernel_regularizer=reg,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv1D(
            n_filt,
            1,
            padding="same",
            kernel_regularizer=reg,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Skip connection
        in_ch = getattr(skip.shape, "__getitem__", lambda _: None)(-1)
        if in_ch is None or in_ch != n_filt:
            skip = tf.keras.layers.Conv1D(n_filt, 1, padding="same")(skip)

        x = tf.keras.layers.Add()([skip, x])
        x = tf.keras.layers.Activation("relu")(x)

        if dr > 0.0:
            x = tf.keras.layers.Dropout(dr)(x)

    # ── LSTM temporal modeling ────────────────────────────────────────
    for units in lstm_units:
        x = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            dropout=dr,
            kernel_regularizer=reg,
        )(x)

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="ResNeXt_LSTM")


__all__ = ["build_cnn_lstm", "build_cnn_bilstm_ed", "build_resnext_lstm"]
