# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/rnn.py                                                     ║
# ║  Bloco: 3d — Redes Recorrentes (LSTM, BiLSTM)                            ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • LSTM: causal nativo, processa sequencia esquerda→direita             ║
# ║    • BiLSTM: acausal, processa em ambas direcoes (offline only)           ║
# ║    • Saida seq2seq via return_sequences=True em todas as camadas          ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: 2 funcoes — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.4, legado C31                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""LSTM e BiLSTM para inversao resistividade — seq2seq.

LSTM: causal-native (only forward), compativel com realtime.
BiLSTM: CAUSAL_INCOMPATIBLE — use apenas em modo offline.

Note:
    Referenciado em:
        - models/registry.py: _REGISTRY['LSTM'], _REGISTRY['BiLSTM']
        - tests/test_models.py: TestRNN
    Legado C31 (813 linhas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: LSTM
# ════════════════════════════════════════════════════════════════════════════
# N camadas LSTM empilhadas com return_sequences=True (seq2seq).
# Nativo causal: cada passo t so usa informacao de t'<=t.
# Adicionar Skip Connection entre camadas LSTM via Dense projecao.
# ──────────────────────────────────────────────────────────────────────────


def build_lstm(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi LSTM empilhado para inversao seq2seq causal.

    N camadas LSTM com return_sequences=True preservam a dimensao temporal.
    Regularizacao via recurrent_dropout e kernel_regularizer.

    Arquitetura:
        Input → LSTM_1(units, ret_seq=True) → LSTM_2 → ... → Dense(out_ch)

    Args:
        config: PipelineConfig com:
            - n_features, sequence_length, output_channels
            - arch_params: units, n_layers, recurrent_dropout

    Returns:
        tf.keras.Model: LSTM seq2seq causal.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['LSTM']
            - tests/test_models.py: TestRNN.test_lstm_forward
        LSTM e causal-native — sem necesidade de use_causal_mode.
        BiLSTM NAO e causal — veja build_bilstm().
        Legado C31 build_lstm().
    """
    import tensorflow as tf
    from geosteering_ai.models.blocks import output_projection

    ap = config.arch_params or {}
    units = ap.get("units", 128)
    n_layers = ap.get("n_layers", 3)
    rec_dropout = ap.get("recurrent_dropout", 0.0)
    dr = config.dropout_rate
    l2 = config.l2_weight if config.use_l2_regularization else 0.0
    reg = tf.keras.regularizers.L2(l2) if l2 > 0.0 else None

    logger.info(
        "build_lstm: n_feat=%d, units=%d, n_layers=%d",
        config.n_features, units, n_layers,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    for layer_i in range(n_layers):
        x = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            dropout=dr,
            recurrent_dropout=rec_dropout,
            kernel_regularizer=reg,
        )(x)
        if dr > 0.0:
            x = tf.keras.layers.Dropout(dr)(x)
        logger.debug("LSTM layer %d: units=%d", layer_i + 1, units)

    out = output_projection(
        x, config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="LSTM")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: BILSTM
# ════════════════════════════════════════════════════════════════════════════
# BiLSTM: LSTM bidirecional (forward + backward) — CAUSAL_INCOMPATIBLE.
# Usa informacao do futuro (backward pass) — apenas modo offline.
# Melhor performance que LSTM unidirecional para dados acausais.
# ──────────────────────────────────────────────────────────────────────────


def build_bilstm(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi BiLSTM bidirecional para inversao seq2seq offline.

    CAUSAL_INCOMPATIBLE: backward LSTM usa dados futuros.
    Melhor performance que LSTM unidirecional para treinamento offline.

    Arquitetura:
        Input → Bidirectional(LSTM_1) → Bidirectional(LSTM_2) → ... → Dense

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: BiLSTM seq2seq (acausal).

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['BiLSTM']
            - tests/test_models.py: TestRNN.test_bilstm_forward
        CAUSAL_INCOMPATIBLE: usar apenas inference_mode='offline'.
        Output units por camada = 2 * units (forward + backward concat).
        Legado C31 build_bilstm().
    """
    import tensorflow as tf
    from geosteering_ai.models.blocks import output_projection

    ap = config.arch_params or {}
    units = ap.get("units", 128)
    n_layers = ap.get("n_layers", 3)
    merge_mode = ap.get("merge_mode", "concat")
    rec_dropout = ap.get("recurrent_dropout", 0.0)
    dr = config.dropout_rate
    l2 = config.l2_weight if config.use_l2_regularization else 0.0
    reg = tf.keras.regularizers.L2(l2) if l2 > 0.0 else None

    logger.info(
        "build_bilstm: n_feat=%d, units=%d, n_layers=%d, merge=%s",
        config.n_features, units, n_layers, merge_mode,
    )

    if config.use_causal_mode:
        logger.warning(
            "BiLSTM e CAUSAL_INCOMPATIBLE. use_causal_mode=True sera ignorado."
        )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    for layer_i in range(n_layers):
        lstm = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            dropout=dr,
            recurrent_dropout=rec_dropout,
            kernel_regularizer=reg,
        )
        x = tf.keras.layers.Bidirectional(lstm, merge_mode=merge_mode)(x)
        if dr > 0.0:
            x = tf.keras.layers.Dropout(dr)(x)
        logger.debug("BiLSTM layer %d: units=%d (out=%d)", layer_i + 1, units, x.shape[-1])

    out = output_projection(
        x, config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="BiLSTM")


__all__ = ["build_lstm", "build_bilstm"]
