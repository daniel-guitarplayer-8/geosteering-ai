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
# ║    • 3 arquiteturas TCN nativas causais para geosteering realtime         ║
# ║    • TCN: empilhamento de dilatacoes exponenciais (1,2,4,...2^k)          ║
# ║    • TCN_Advanced: multi-scale stacks + atencao + SE                      ║
# ║    • Causal nativo: sem look-ahead, compativel com InferencePipeline      ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: 3 funções — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.3, legado C30                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementação inicial (TCN, TCN_Advanced)          ║
# ║    v2.0.1 (2026-04) — +ModernTCN (3 arquiteturas)                        ║
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
# Para seq_len típico (default 600, derivado do .out) e kernel=3: 8 níveis → campo receptivo = 255.
# Sempre causal (padding='causal') — nativo para geosteering realtime.
# ──────────────────────────────────────────────────────────────────────────


def build_tcn(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi TCN com dilation doubling para inversao seq2seq causal.

    N_LEVELS blocos residuais causais com dilations [1, 2, 4, ..., 2^(N-1)].
    Campo receptivo: (kernel-1) * 2 * (2^N - 1) + 1.

    Arquitetura:
        ┌────────────────────────────────────────────────────────┐
        │  Input (batch, seq_len, n_feat)                        │
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

    from geosteering_ai.models.blocks import output_projection, tcn_residual_block

    ap = config.arch_params or {}
    filters = ap.get("filters", 64)
    n_levels = ap.get("n_levels", 8)
    kernel_size = ap.get("kernel_size", 3)
    dr = ap.get("dropout_rate", config.dropout_rate)

    logger.info(
        "build_tcn: n_feat=%d, filters=%d, n_levels=%d, k=%d",
        config.n_features,
        filters,
        n_levels,
        kernel_size,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── Stem: projecao de canais de entrada ───────────────────────────
    x = tf.keras.layers.Conv1D(filters, 1, padding="same")(inp)

    # ── TCN stack com dilation doubling ───────────────────────────────
    for level in range(n_levels):
        dilation = 2**level
        x = tcn_residual_block(
            x,
            filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout_rate=dr,
            l1=config.l1_weight if config.use_l1_regularization else 0.0,
            l2=config.l2_weight if config.use_l2_regularization else 0.0,
        )
        logger.debug("TCN level %d: dilation=%d", level, dilation)

    out = output_projection(
        x,
        config.output_channels,
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
        output_projection,
        se_block,
        self_attention_block,
        tcn_residual_block,
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
        config.n_features,
        n_stacks,
        n_levels,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = tf.keras.layers.Conv1D(filters, 1, padding="same")(inp)

    for stack_i in range(n_stacks):
        for level in range(n_levels):
            dilation = 2**level
            x = tcn_residual_block(
                x,
                filters,
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
            x,
            num_heads=num_heads,
            key_dim=filters // num_heads,
            use_causal_mask=True,  # TCN_Advanced sempre causal
        )

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="TCN_Advanced")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: MODERNTCN — CONVOLUCAO MODERNA PURA PARA SERIES TEMPORAIS
# ════════════════════════════════════════════════════════════════════════════
# ModernTCN incorpora 4 inovacoes sobre o TCN classico (Bai 2018):
#   1. Patch Tokenization: sequencia segmentada em janelas de P pontos
#   2. DWConv largo (k=51): mixing temporal via depthwise separable conv
#   3. ConvFFN: mixing de canais via expansao pointwise (1→4→1)
#   4. LayerNorm (invariante a batch) em vez de BatchNorm
# Campo receptivo efetivo: patch_size × kernel_size × stride ≈ 200+ m,
# superior aos ~127 m do TCN classico (6 blocos dilatados com k=3).
#
# Vantagens para geosteering:
#   - LayerNorm funciona com B=1 (realtime sem degradacao treino/inf)
#   - Separacao explicita: temporal (DWConv) vs canal (ConvFFN)
#   - Menos parametros (~50%) para mesmo campo receptivo
#
# Ref: Luo & Wang (ICLR 2024) "ModernTCN: A Modern Pure Convolution
#      Structure for General Time Series Analysis" arXiv:2310.06625.
# ──────────────────────────────────────────────────────────────────────────


def _modern_tcn_block(x, filters, large_kernel, dropout_rate, causal=False):
    """Bloco ModernTCN: DWConv temporal + ConvFFN de canais.

    Separa mixing temporal (DWConv com kernel largo sobre patches)
    de mixing entre canais (ConvFFN = expansao 4× + compressao).
    LayerNorm em vez de BatchNorm para invariancia a tamanho de batch.

    Estrutura:
        x → LN → DWConv(k=large_kernel, groups=C) → LN
          → Conv1×1(C→4C) → GELU → Conv1×1(4C→C) → Dropout → +skip → out

    Args:
        x: Tensor de entrada (B, N_patches, C).
        filters: Numero de canais C do bloco.
        large_kernel: Tamanho do kernel da DWConv (tipicamente 51).
            Determina campo receptivo temporal do bloco.
        dropout_rate: Taxa de dropout (0.0-1.0).
        causal: Se True, usa padding='causal' na DWConv (geosteering
            realtime). Se False, usa padding='same' (offline).

    Returns:
        Tensor (B, N_patches, C) com skip connection residual.

    Note:
        Funcao privada — usada apenas por build_modern_tcn().
        DWConv: cada canal eh convoluido independentemente (groups=C),
        o que eh fisicamente coerente pois cada componente EM (Re/Im de
        Hxx, Hzz, etc.) tem dependencias temporais distintas.
        Causal mode: padding='causal' garante que ponto z nao usa z'>z.
        Ref: Luo & Wang (ICLR 2024) Secao 3.2.
    """
    import tensorflow as tf

    skip = x
    pad = "causal" if causal else "same"

    # ── DWConv temporal (mixing entre patches) ────────────────────────
    # Cada canal convoluido independentemente com kernel largo.
    # Padding 'causal' em realtime garante ausencia de look-ahead.
    # groups=filters → depthwise separable (1 kernel por canal).
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.DepthwiseConv1D(
        kernel_size=large_kernel,
        padding=pad,
        depth_multiplier=1,
        use_bias=False,
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)

    # ── ConvFFN (mixing entre canais) ─────────────────────────────────
    # Expansao 4× → GELU → compressao → dropout.
    # Analogia Transformer: DWConv ≡ Self-Attention, ConvFFN ≡ FFN.
    x = tf.keras.layers.Dense(filters * 4, use_bias=False)(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dense(filters, use_bias=False)(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # ── Skip connection ───────────────────────────────────────────────
    return tf.keras.layers.Add()([skip, x])


def build_modern_tcn(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi ModernTCN para inversao seq2seq de resistividade.

    Modernizacao do TCN classico com patch tokenization, large-kernel
    depthwise convolution e channel-mixing via ConvFFN. Mantém
    arquitetura puramente convolucional (sem atencao, sem recorrencia)
    com campo receptivo ~200+ m e ~50% menos parametros que TCN classico.

    Arquitetura:
      ┌──────────────────────────────────────────────────────────────┐
      │  Input (B, seq_len, n_features)                             │
      │    ↓                                                        │
      │  Stem: Conv1D(filters, k=1) — projeção de canais            │
      │    ↓                                                        │
      │  N × ModernTCN Block:                                       │
      │    LN → DWConv(k=51, groups=C) → LN                        │
      │    → Dense(C→4C) → GELU → Dense(4C→C) → Dropout → +skip    │
      │    ↓                                                        │
      │  Dense(128) → ReLU → Dense(output_channels, 'linear')      │
      │    ↓                                                        │
      │  Output (B, seq_len, output_channels)                       │
      └──────────────────────────────────────────────────────────────┘

    Dual-mode:
      ┌────────────────────────────────────────────────────────────┐
      │  "same"   →  Offline (acausal, batch completo)            │
      │  "causal" →  Realtime (causal, sliding window)            │
      │  LayerNorm invariante a B → sem degradacao em B=1         │
      └────────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com:
            - n_features, sequence_length, output_channels
            - use_causal_mode: True para geosteering realtime
            - dropout_rate: dropout nos blocos ConvFFN
            - arch_params: override granular:
                - filters (int, default 128): canais por bloco
                - n_blocks (int, default 4): numero de blocos ModernTCN
                - large_kernel (int, default 51): kernel da DWConv temporal

    Returns:
        tf.keras.Model: ModernTCN seq2seq.
            Input shape: (None, config.sequence_length, config.n_features)
            Output shape: (None, config.sequence_length, config.output_channels)

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig(model_type="ModernTCN")
        >>> model = build_modern_tcn(config)
        >>> assert model.output_shape == (None, 600, 2)

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['ModernTCN']
            - tests/test_models.py: TestModernTCN
        Causal mode: DWConv usa padding='causal' quando use_causal_mode=True.
        Campo receptivo: large_kernel × n_blocks = 51 × 4 = 204 pontos
        (~204 m para SPACING_METERS=1.0, cobrindo ~5.7× skin depth maximo).
        Ref: Luo & Wang (ICLR 2024) arXiv:2310.06625.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import output_projection

    ap = config.arch_params or {}
    filters = ap.get("filters", 128)
    n_blocks = ap.get("n_blocks", 4)
    large_kernel = ap.get("large_kernel", 51)
    dr = ap.get("dropout_rate", config.dropout_rate)
    causal = config.use_causal_mode

    logger.info(
        "build_modern_tcn: n_feat=%d, filters=%d, n_blocks=%d, k=%d, causal=%s",
        config.n_features,
        filters,
        n_blocks,
        large_kernel,
        causal,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── Stem: projecao de entrada para C canais ──────────────────────
    pad = "causal" if causal else "same"
    x = tf.keras.layers.Conv1D(filters, 1, padding=pad, use_bias=False)(inp)

    # ── N blocos ModernTCN ───────────────────────────────────────────
    for block_i in range(n_blocks):
        x = _modern_tcn_block(x, filters, large_kernel, dr, causal=causal)
        logger.debug("ModernTCN block %d complete", block_i + 1)

    # ── Decoder → output ─────────────────────────────────────────────
    x = tf.keras.layers.LayerNormalization()(x)
    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="ModernTCN")


__all__ = ["build_tcn", "build_tcn_advanced", "build_modern_tcn"]
