# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/tcn.py                                                     ║
# ║  Bloco: 3c — Temporal Convolutional Networks                              ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • 2 arquiteturas TCN nativas causais para geosteering realtime         ║
# ║    • TCN: empilhamento de dilatacoes exponenciais (1,2,4,...2^k)          ║
# ║    • TCN_Advanced: multi-scale stacks + atencao + SE                      ║
# ║    • Causal nativo: sem look-ahead, compativel com InferencePipeline      ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: 2 funcoes — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.3, legado C30                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""TCN e TCN_Advanced para inversao seq2seq causalmente correta.

Ambas as arquiteturas sao causal-native: todo Conv1D usa padding='causal'.
Dilation doubling: 1, 2, 4, 8, ..., 2^(n_levels-1).
Campo receptivo total = (kernel_size - 1) * 2 * sum(dilations) + 1.

Note:
    Referenciado em:
        - models/registry.py: _REGISTRY['TCN'], _REGISTRY['TCN_Advanced']
        - tests/test_models.py: TestTCN
    Ref: Bai et al. (2018) arXiv:1803.01271 — An Empirical Evaluation of TCNs.
    Legado C30 (949 linhas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: TCN — TEMPORAL CONVOLUTIONAL NETWORK
# ════════════════════════════════════════════════════════════════════════════
# TCN empilha tcn_residual_block com dilation doubling (1, 2, 4, 8...).
# N_LEVELS dilation levels controlam o campo receptivo.
# Para seq_len=600 e kernel=3: 8 niveis → campo receptivo = 255.
# Sempre causal (padding='causal') — nativo para geosteering realtime.
# ──────────────────────────────────────────────────────────────────────────


def build_tcn(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi TCN com dilation doubling para inversao seq2seq causal.

    N_LEVELS blocos residuais causais com dilations [1, 2, 4, ..., 2^(N-1)].
    Campo receptivo: (kernel-1) * 2 * (2^N - 1) + 1.

    Arquitetura:
        ┌────────────────────────────────────────────────────────┐
        │  Input (batch, 600, n_feat)                           │
        │  ↓ Stem: Conv(filters, k=1) — projecao de entrada     │
        │  ↓ TCNBlock(filters, d=1)                             │
        │  ↓ TCNBlock(filters, d=2)                             │
        │  ↓ TCNBlock(filters, d=4)                             │
        │  ...                                                   │
        │  ↓ TCNBlock(filters, d=2^(N-1))                       │
        │  ↓ Output: Conv(out_ch, k=1)                          │
        └────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com:
            - n_features, sequence_length, output_channels
            - use_causal_mode: True (TCN e sempre causal)
            - arch_params: filters, n_levels, kernel_size, dropout_rate

    Returns:
        tf.keras.Model: TCN causal seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['TCN']
            - tests/test_models.py: TestTCN.test_tcn_forward
        TCN e causal-native: use_causal_mode nao altera comportamento.
        Ref: docs/ARCHITECTURE_v2.md secao 5.3.
    """
    import tensorflow as tf
    from geosteering_ai.models.blocks import tcn_residual_block, output_projection

    ap = config.arch_params or {}
    filters = ap.get("filters", 64)
    n_levels = ap.get("n_levels", 8)
    kernel_size = ap.get("kernel_size", 3)
    dr = ap.get("dropout_rate", config.dropout_rate)

    logger.info(
        "build_tcn: n_feat=%d, filters=%d, n_levels=%d, k=%d",
        config.n_features, filters, n_levels, kernel_size,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── Stem: projecao de canais de entrada ───────────────────────────
    x = tf.keras.layers.Conv1D(filters, 1, padding="same")(inp)

    # ── TCN stack com dilation doubling ───────────────────────────────
    for level in range(n_levels):
        dilation = 2 ** level
        x = tcn_residual_block(
            x, filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout_rate=dr,
            l1=config.l1_weight if config.use_l1_regularization else 0.0,
            l2=config.l2_weight if config.use_l2_regularization else 0.0,
        )
        logger.debug("TCN level %d: dilation=%d", level, dilation)

    out = output_projection(
        x, config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="TCN")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: TCN_ADVANCED — MULTI-SCALE + ATENCAO
# ════════════════════════════════════════════════════════════════════════════
# TCN_Advanced adiciona:
#   1. Multi-scale stacks: 2+ sequencias de dilation doubling
#   2. SE block entre stacks (recalibracao por canal)
#   3. Atencao temporal leve (opcional) no topo
# ──────────────────────────────────────────────────────────────────────────


def build_tcn_advanced(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi TCN_Advanced: multi-scale stacks + SE + atencao.

    Empilha N_STACKS × N_LEVELS TCNBlocks, com SE block entre stacks.
    Adiciona atencao temporal leve no final (opcional via arch_params).

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: TCN_Advanced causal seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['TCN_Advanced']
            - tests/test_models.py: TestTCN.test_tcn_advanced_forward
        Legado C30 build_tcn_advanced().
    """
    import tensorflow as tf
    from geosteering_ai.models.blocks import (
        tcn_residual_block, se_block, self_attention_block, output_projection
    )

    ap = config.arch_params or {}
    filters = ap.get("filters", 64)
    n_stacks = ap.get("n_stacks", 2)
    n_levels = ap.get("n_levels", 6)
    kernel_size = ap.get("kernel_size", 3)
    use_attention = ap.get("use_attention", True)
    use_se = ap.get("use_se", True)
    num_heads = ap.get("num_heads", 4)
    dr = ap.get("dropout_rate", config.dropout_rate)

    logger.info(
        "build_tcn_advanced: n_feat=%d, n_stacks=%d, n_levels=%d",
        config.n_features, n_stacks, n_levels,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = tf.keras.layers.Conv1D(filters, 1, padding="same")(inp)

    for stack_i in range(n_stacks):
        for level in range(n_levels):
            dilation = 2 ** level
            x = tcn_residual_block(
                x, filters,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                dropout_rate=dr,
                l1=config.l1_weight if config.use_l1_regularization else 0.0,
                l2=config.l2_weight if config.use_l2_regularization else 0.0,
            )
        if use_se:
            x = se_block(x, reduction=config.se_reduction)
        logger.debug("TCN_Advanced stack %d complete", stack_i + 1)

    if use_attention:
        x = self_attention_block(
            x, num_heads=num_heads, key_dim=filters // num_heads,
            use_causal_mask=True,  # TCN_Advanced sempre causal
        )

    out = output_projection(
        x, config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="TCN_Advanced")


__all__ = ["build_tcn", "build_tcn_advanced"]
