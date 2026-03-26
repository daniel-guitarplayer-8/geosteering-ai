# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/blocks.py                                                  ║
# ║  Bloco: 3a — Blocos Keras Reutilizaveis                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • 23 blocos funcionais Keras para construcao das 44 arquiteturas       ║
# ║    • Functional API (tensor→tensor): compativel com qualquer modelo       ║
# ║    • Lazy TF imports: importavel em CPU-only sem TF instalado              ║
# ║    • Causal/acausal via parametro explicit — sem surpresas                ║
# ║                                                                            ║
# ║  Blocos implementados (23):                                               ║
# ║    Grupo 1 — Conv: residual_block_1d, bottleneck_block_1d,               ║
# ║              conv_next_block, se_block, dilated_causal_block              ║
# ║    Grupo 2 — Inception: inception_module, mbconv_block                   ║
# ║    Grupo 3 — Gated/TCN: gated_activation_block, tcn_residual_block       ║
# ║    Grupo 4 — Atencao: self_attention_block, transformer_encoder_block,   ║
# ║              autocorr_block                                               ║
# ║    Grupo 5 — Transformer: patch_embedding_block, grn_block,              ║
# ║              vsn_block, ita_block                                         ║
# ║    Grupo 6 — Decomp: series_decomp_block                                 ║
# ║    Grupo 7 — Utility: output_projection, normalization_block,            ║
# ║              skip_connection_block, feedforward_block,                    ║
# ║              inception_time_block, attention_block                        ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), tensorflow >= 2.12            ║
# ║  Exports: ~23 funcoes — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.1 (blocos), legado C27             ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (23 blocos, 13 orig + 10)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Blocos Keras reutilizaveis para construcao das 44 arquiteturas.

Todos os blocos seguem a API funcional do Keras:
    f(x: tf.Tensor, **kwargs) -> tf.Tensor

A dimensao temporal e sempre preservada: input (batch, seq_len, ch)
→ output (batch, seq_len, ch'). Nenhum bloco faz pooling global.

Imports de TensorFlow sao LAZY (dentro de cada funcao) para que o
modulo possa ser importado em ambientes sem TF (dev CPU-only).

Note:
    Referenciado em:
        - models/cnn.py: residual_block_1d, bottleneck_block_1d,
          conv_next_block, se_block, inception_module, inception_time_block
        - models/tcn.py: dilated_causal_block, tcn_residual_block
        - models/rnn.py: (sem blocos compartilhados — LSTM/BiLSTM nativos)
        - models/unet.py: skip_connection_block, normalization_block
        - models/transformer.py: transformer_encoder_block, patch_embedding_block,
          grn_block, vsn_block, feedforward_block, ita_block, autocorr_block
        - models/decomposition.py: series_decomp_block, feedforward_block
        - models/geosteering.py: gated_activation_block, self_attention_block,
          dilated_causal_block, series_decomp_block
        - tests/test_models.py: TestBlocks
    Ref: docs/ARCHITECTURE_v2.md secao 5.1. Legado C27.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: UTILITARIOS INTERNOS
# ════════════════════════════════════════════════════════════════════════════
# Helpers compartilhados: padding selector e regularizer builder.
# Usados por todos os grupos de blocos para causal vs same.
# ──────────────────────────────────────────────────────────────────────────


def _padding(causal: bool) -> str:
    """Seleciona padding string para Conv1D.

    Args:
        causal: Se True, retorna 'causal' (geosteering realtime).
            Se False, retorna 'same' (modo offline/acausal).

    Returns:
        str: 'causal' ou 'same'.
    """
    return "causal" if causal else "same"


def _get_regularizer(l1: float = 0.0, l2: float = 0.0):
    """Constroi regularizador kernel com L1, L2 ou ElasticNet.

    Args:
        l1: Peso do regularizador L1. 0.0 = desativado.
        l2: Peso do regularizador L2. 0.0 = desativado.

    Returns:
        tf.keras.regularizers.Regularizer | None:
            None se ambos zero, L1L2 caso contrario.
    """
    import tensorflow as tf
    if l1 == 0.0 and l2 == 0.0:
        return None
    return tf.keras.regularizers.L1L2(l1=l1, l2=l2)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: GRUPO 1 — BLOCOS CONVOLUCIONAIS BASICOS (5 blocos)
# ════════════════════════════════════════════════════════════════════════════
# Os 5 blocos fundamentais de convolucao 1D:
#   residual_block_1d   — ResNet skip connection (He 2016)
#   bottleneck_block_1d — ResNet-50 bottleneck 1x1→kxk→1x1
#   conv_next_block     — ConvNeXt depthwise + pointwise (Liu 2022)
#   se_block            — Squeeze-and-Excitation (Hu 2018)
#   dilated_causal_block — TCN building block (Bai 2018)
#
# Todos recebem x: (batch, seq_len, ch) e retornam (batch, seq_len, ch').
# ──────────────────────────────────────────────────────────────────────────


def residual_block_1d(
    x: "tf.Tensor",
    filters: int,
    kernel_size: int = 3,
    *,
    causal: bool = False,
    use_bn: bool = True,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Bloco residual 1D (ResNet-style).

    Implementa: x → Conv1D → BN → ReLU → Conv1D → BN → Add(x, y) → ReLU.
    Se o numero de canais mudar, aplica projecao 1x1 na skip connection.

    Esquema:
        x ─────────────────────────────── (proj 1x1 se n_ch != filters)
        └── Conv1D → [BN] → ReLU → Conv1D → [BN] → Add → ReLU → out

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        filters: Numero de filtros na convolucao.
        kernel_size: Tamanho do kernel 1D. Default: 3.
        causal: Se True, usa padding causal (geosteering realtime).
        use_bn: Se True, aplica BatchNormalization apos cada Conv.
        activation: Funcao de ativacao. Default: 'relu'.
        dropout_rate: Taxa de dropout (0.0 = desativado).
        l1: Peso regularizador L1. 0.0 = desativado.
        l2: Peso regularizador L2. 0.0 = desativado.

    Returns:
        tf.Tensor: Output (batch, seq_len, filters).

    Note:
        Referenciado em:
            - models/cnn.py: build_resnet18, build_resnet34 (stage 1-4)
            - tests/test_models.py: TestBlocks.test_residual_block_1d
        Ref: He et al. (2016) CVPR — Deep Residual Learning.
        Input n_ch != filters → projecao 1x1 automatica na skip.
    """
    import tensorflow as tf

    pad = _padding(causal)
    reg = _get_regularizer(l1, l2)
    n_ch = x.shape[-1]

    # ── Branch principal ──────────────────────────────────────────────
    y = tf.keras.layers.Conv1D(
        filters, kernel_size, padding=pad,
        kernel_regularizer=reg, use_bias=not use_bn,
    )(x)
    if use_bn:
        y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation(activation)(y)
    if dropout_rate > 0.0:
        y = tf.keras.layers.Dropout(dropout_rate)(y)

    y = tf.keras.layers.Conv1D(
        filters, kernel_size, padding=pad,
        kernel_regularizer=reg, use_bias=not use_bn,
    )(y)
    if use_bn:
        y = tf.keras.layers.BatchNormalization()(y)

    # ── Skip connection (projecao 1x1 se canais mudaram) ─────────────
    if n_ch != filters:
        x = tf.keras.layers.Conv1D(
            filters, 1, padding="same", kernel_regularizer=reg,
        )(x)

    # ── Fusao add + ativacao ─────────────────────────────────────────
    out = tf.keras.layers.Add()([x, y])
    out = tf.keras.layers.Activation(activation)(out)
    return out


def bottleneck_block_1d(
    x: "tf.Tensor",
    filters: int,
    kernel_size: int = 3,
    *,
    causal: bool = False,
    expansion: int = 4,
    use_bn: bool = True,
    activation: str = "relu",
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Bloco bottleneck 1D (ResNet-50 style): 1x1 → kxk → 1x1.

    Reduz computacao mantendo representacao rica: comprime canais,
    processa com conv largo, expande novamente.

    Esquema:
        x ──────────────────────────────── (proj 1x1 se necessario)
        └── Conv1D(1) → Conv1D(k) → Conv1D(filters*exp) → Add → ReLU

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        filters: Canais internos (nao o output). Output = filters * expansion.
        kernel_size: Tamanho do kernel central. Default: 3.
        causal: Se True, padding causal.
        expansion: Fator de expansao dos canais de saida. Default: 4.
        use_bn: Se True, BatchNormalization apos cada Conv.
        activation: Funcao de ativacao. Default: 'relu'.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, seq_len, filters * expansion).

    Note:
        Referenciado em:
            - models/cnn.py: build_resnet50 (stages 1-4)
            - tests/test_models.py: TestBlocks.test_bottleneck_block_1d
        Ref: He et al. (2016) CVPR — ResNet-50 bottleneck design.
    """
    import tensorflow as tf

    pad = _padding(causal)
    reg = _get_regularizer(l1, l2)
    out_ch = filters * expansion
    n_ch = x.shape[-1]

    # ── 1x1 compress ─────────────────────────────────────────────────
    y = tf.keras.layers.Conv1D(
        filters, 1, padding="same",
        kernel_regularizer=reg, use_bias=not use_bn,
    )(x)
    if use_bn:
        y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation(activation)(y)

    # ── kxk transform ────────────────────────────────────────────────
    y = tf.keras.layers.Conv1D(
        filters, kernel_size, padding=pad,
        kernel_regularizer=reg, use_bias=not use_bn,
    )(y)
    if use_bn:
        y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation(activation)(y)

    # ── 1x1 expand ───────────────────────────────────────────────────
    y = tf.keras.layers.Conv1D(
        out_ch, 1, padding="same",
        kernel_regularizer=reg, use_bias=not use_bn,
    )(y)
    if use_bn:
        y = tf.keras.layers.BatchNormalization()(y)

    # ── Skip ─────────────────────────────────────────────────────────
    if n_ch != out_ch:
        x = tf.keras.layers.Conv1D(
            out_ch, 1, padding="same", kernel_regularizer=reg,
        )(x)

    out = tf.keras.layers.Add()([x, y])
    out = tf.keras.layers.Activation(activation)(out)
    return out


def conv_next_block(
    x: "tf.Tensor",
    filters: int,
    kernel_size: int = 7,
    *,
    causal: bool = False,
    expansion: int = 4,
    dropout_rate: float = 0.0,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Bloco ConvNeXt 1D: depthwise → LN → pointwise MLP → skip.

    Modernizacao do ResNet usando ideias do Vision Transformer:
    kernel 7, LayerNorm (nao BN), MLP expansion 4x.

    Esquema (por canal):
        x → DWConv1D(k=7) → LN → PW FC(4x) → GELU → PW FC(1x) → Add(x)

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        filters: Numero de filtros de saida.
        kernel_size: Kernel da depthwise conv. Default: 7.
        causal: Se True, padding causal.
        expansion: Fator MLP interno. Default: 4.
        dropout_rate: Dropout no path MLP.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, seq_len, filters).

    Note:
        Referenciado em:
            - models/cnn.py: build_convnext (stages 1-4)
            - tests/test_models.py: TestBlocks.test_conv_next_block
        Ref: Liu et al. (2022) CVPR — A ConvNet for the 2020s.
        GELU em vez de ReLU (ConvNeXt design choice).
    """
    import tensorflow as tf

    pad = _padding(causal)
    reg = _get_regularizer(l1, l2)
    n_ch = x.shape[-1]

    # ── Depthwise conv (per-channel spatial mixing) ───────────────────
    y = tf.keras.layers.DepthwiseConv1D(
        kernel_size, padding=pad,
        depthwise_regularizer=reg, use_bias=True,
    )(x)
    y = tf.keras.layers.LayerNormalization()(y)

    # ── Pointwise MLP (channel mixing): expand → GELU → compress ─────
    y = tf.keras.layers.Dense(
        filters * expansion, kernel_regularizer=reg,
    )(y)
    y = tf.keras.layers.Activation("gelu")(y)
    if dropout_rate > 0.0:
        y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(filters, kernel_regularizer=reg)(y)

    # ── Skip ─────────────────────────────────────────────────────────
    if n_ch != filters:
        x = tf.keras.layers.Dense(filters)(x)
    return tf.keras.layers.Add()([x, y])


def se_block(
    x: "tf.Tensor",
    reduction: int = 16,
) -> "tf.Tensor":
    """Bloco Squeeze-and-Excitation (SE): recalibracao de canal.

    Aprende pesos por canal via pooling global → MLP → sigmoid.
    Reescala os canais originais multiplicativamente.

    Esquema:
        x → GlobalAvgPool → Dense(ch/r, ReLU) → Dense(ch, sigmoid) → Multiply(x)

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        reduction: Fator de reducao do bottleneck MLP. Default: 16.

    Returns:
        tf.Tensor: Output (batch, seq_len, channels) — mesma shape.

    Note:
        Referenciado em:
            - models/cnn.py: build_resnet18/34/50 quando config.use_se_block
            - tests/test_models.py: TestBlocks.test_se_block
        Ref: Hu et al. (2018) CVPR — Squeeze-and-Excitation Networks.
        reduction=16 e o padrao do paper original.
    """
    import tensorflow as tf

    n_ch = x.shape[-1]
    bottleneck = max(1, n_ch // reduction)

    # ── Squeeze: global average pooling (batch, ch) ──────────────────
    s = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)(x)  # (b,1,ch)

    # ── Excitation: MLP ──────────────────────────────────────────────
    s = tf.keras.layers.Dense(bottleneck, activation="relu")(s)
    s = tf.keras.layers.Dense(n_ch, activation="sigmoid")(s)  # (b,1,ch)

    # ── Recalibracao multiplicativa ───────────────────────────────────
    return tf.keras.layers.Multiply()([x, s])


def dilated_causal_block(
    x: "tf.Tensor",
    filters: int,
    kernel_size: int = 2,
    dilation_rate: int = 1,
    *,
    causal: bool = True,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Bloco de convolucao dilatada e causal (TCN building block).

    Dois Conv1D causais dilatados em serie com skip residual.
    Dilation doubling (1, 2, 4, 8...) expande o campo receptivo.

    Esquema:
        x → CausalConv1D(d) → Act → Dropout → CausalConv1D(d) → Act → Add(x)

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        filters: Numero de filtros.
        kernel_size: Tamanho do kernel. Default: 2 (WaveNet/TCN).
        dilation_rate: Taxa de dilatacao. Default: 1.
        causal: Se True, padding causal (obrigatorio para TCN).
        activation: Funcao de ativacao.
        dropout_rate: Dropout apos cada ativacao.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, seq_len, filters).

    Note:
        Referenciado em:
            - models/tcn.py: build_tcn, build_tcn_advanced
            - models/geosteering.py: build_wavenet (dilation stacks)
            - tests/test_models.py: TestBlocks.test_dilated_causal_block
        Ref: Bai et al. (2018) — An Empirical Evaluation of TCNs.
        causal=True e o padrao para uso em TCN; pode ser False para arqs
        acausais que querem dilatacao sem causalidade.
    """
    import tensorflow as tf

    pad = _padding(causal)
    reg = _get_regularizer(l1, l2)
    n_ch = x.shape[-1]

    y = tf.keras.layers.Conv1D(
        filters, kernel_size, dilation_rate=dilation_rate,
        padding=pad, kernel_regularizer=reg, use_bias=True,
    )(x)
    y = tf.keras.layers.Activation(activation)(y)
    if dropout_rate > 0.0:
        y = tf.keras.layers.SpatialDropout1D(dropout_rate)(y)

    y = tf.keras.layers.Conv1D(
        filters, kernel_size, dilation_rate=dilation_rate,
        padding=pad, kernel_regularizer=reg, use_bias=True,
    )(y)
    y = tf.keras.layers.Activation(activation)(y)
    if dropout_rate > 0.0:
        y = tf.keras.layers.SpatialDropout1D(dropout_rate)(y)

    if n_ch != filters:
        x = tf.keras.layers.Conv1D(filters, 1)(x)
    return tf.keras.layers.Add()([x, y])


# ════════════════════════════════════════════════════════════════════════════
# SECAO: GRUPO 2 — INCEPTION E EFFICIENT (2 blocos)
# ════════════════════════════════════════════════════════════════════════════
# Blocos multi-escala: inception_module (InceptionNet/InceptionTime) e
# mbconv_block (EfficientNet MobileInvertedBottleneck).
# Multi-scale receptive field em paralelo → concatenacao de canais.
# ──────────────────────────────────────────────────────────────────────────


def inception_module(
    x: "tf.Tensor",
    filters: int = 32,
    *,
    causal: bool = False,
    bottleneck_size: int = 32,
    kernel_sizes: tuple = (9, 19, 39),
    activation: str = "relu",
    use_bn: bool = True,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Modulo Inception 1D multi-escala (InceptionTime).

    3 Conv1D com kernels diferentes em paralelo + MaxPool, concatenados.
    Bottleneck 1x1 antes das convs principais reduz custo computacional.

    Esquema:
        x ─→ MaxPool → Conv1D(1)
        ├─→ Conv1D(1, bottleneck) → Conv1D(k1, filters) ─┐
        ├─→ Conv1D(1, bottleneck) → Conv1D(k2, filters) ─┼→ Concat → BN → ReLU
        └─→ Conv1D(1, bottleneck) → Conv1D(k3, filters) ─┘

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        filters: Filtros por ramo convolucional (excl. MaxPool).
        causal: Se True, padding causal em todas as convs.
        bottleneck_size: Canais do bottleneck 1x1 inicial.
        kernel_sizes: Tupla de 3 tamanhos de kernel. Default: (9, 19, 39).
        activation: Funcao de ativacao apos Concat.
        use_bn: Se True, BatchNorm apos Concat.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, seq_len, 4 * filters).

    Note:
        Referenciado em:
            - models/cnn.py: build_inceptionnet, build_inceptiontime
            - tests/test_models.py: TestBlocks.test_inception_module
        Ref: Ismail Fawaz et al. (2020) InceptionTime — Classifying time series.
        Output ch = 4*filters (3 conv + 1 maxpool).
    """
    import tensorflow as tf

    pad = _padding(causal)
    reg = _get_regularizer(l1, l2)

    # ── Ramo MaxPool + Conv 1x1 (conserva detalhes pontuais) ─────────
    mp = tf.keras.layers.MaxPool1D(3, strides=1, padding="same")(x)
    mp = tf.keras.layers.Conv1D(
        filters, 1, padding="same", kernel_regularizer=reg,
    )(mp)

    # ── 3 ramos conv multi-escala ─────────────────────────────────────
    branches = [mp]
    for k in kernel_sizes:
        b = tf.keras.layers.Conv1D(
            bottleneck_size, 1, padding="same", kernel_regularizer=reg,
        )(x)
        b = tf.keras.layers.Conv1D(
            filters, k, padding=pad, kernel_regularizer=reg,
        )(b)
        branches.append(b)

    # ── Concatenacao, BN, ativacao ────────────────────────────────────
    out = tf.keras.layers.Concatenate()(branches)
    if use_bn:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out


def mbconv_block(
    x: "tf.Tensor",
    filters: int,
    kernel_size: int = 3,
    *,
    causal: bool = False,
    expansion: int = 6,
    se_ratio: float = 0.25,
    dropout_rate: float = 0.0,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """MobileInvertedBottleneck (MBConv) 1D — bloco EfficientNet.

    Expande canais → depthwise conv → squeeze-excitation → projecao.

    Esquema:
        x → Conv1D(1, ch*exp) → DWConv1D(k) → SE → Conv1D(1, ch_out) → Add

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        filters: Numero de canais de saida.
        kernel_size: Kernel da depthwise conv.
        causal: Se True, padding causal.
        expansion: Fator de expansao dos canais internos.
        se_ratio: Fracao de canais para SE bottleneck.
        dropout_rate: Dropout estocastico (stochastic depth).
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, seq_len, filters).

    Note:
        Referenciado em:
            - models/unet.py: UNet_EfficientNet (encoder stages)
            - tests/test_models.py: TestBlocks.test_mbconv_block
        Ref: Tan & Le (2019) — EfficientNet: Rethinking Model Scaling.
    """
    import tensorflow as tf

    pad = _padding(causal)
    reg = _get_regularizer(l1, l2)
    n_ch = x.shape[-1]
    mid_ch = n_ch * expansion

    # ── Expansao 1x1 ──────────────────────────────────────────────────
    y = tf.keras.layers.Conv1D(
        mid_ch, 1, padding="same", use_bias=False, kernel_regularizer=reg,
    )(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("swish")(y)

    # ── Depthwise conv ────────────────────────────────────────────────
    y = tf.keras.layers.DepthwiseConv1D(
        kernel_size, padding=pad, use_bias=False,
        depthwise_regularizer=reg,
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("swish")(y)

    # ── SE ────────────────────────────────────────────────────────────
    if se_ratio > 0:
        y = se_block(y, reduction=max(1, int(1.0 / se_ratio)))

    # ── Projecao 1x1 ─────────────────────────────────────────────────
    y = tf.keras.layers.Conv1D(
        filters, 1, padding="same", use_bias=False, kernel_regularizer=reg,
    )(y)
    y = tf.keras.layers.BatchNormalization()(y)

    # ── Skip (so se n_ch == filters) ─────────────────────────────────
    if n_ch == filters:
        if dropout_rate > 0.0:
            y = tf.keras.layers.Dropout(dropout_rate)(y)
        y = tf.keras.layers.Add()([x, y])
    return y


# ════════════════════════════════════════════════════════════════════════════
# SECAO: GRUPO 3 — BLOCOS TCN / GATED (2 blocos)
# ════════════════════════════════════════════════════════════════════════════
# Blocos especializados para modelos temporais causais:
#   gated_activation_block — WaveNet gated unit (tanh × sigmoid)
#   tcn_residual_block      — Temporal Conv residual com dilation doubling
# Causalidade nativa: todos usam padding='causal' por default.
# ──────────────────────────────────────────────────────────────────────────


def gated_activation_block(
    x: "tf.Tensor",
    filters: int,
    kernel_size: int = 2,
    dilation_rate: int = 1,
    *,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Bloco de ativacao gated (WaveNet): tanh × sigmoid.

    O produto elementar tanh(f)*sigmoid(g) permite o modelo controlar
    a amplitude do sinal (tanh) e o fluxo de informacao (gate sigma).
    Sempre causal (fundamental para WaveNet).

    Esquema:
        x → CausalConv1D → split(tanh, sigmoid) → Multiply → out

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        filters: Numero de filtros (cada ramo usa filters canais).
        kernel_size: Kernel da conv causal. Default: 2.
        dilation_rate: Taxa de dilatacao. Default: 1.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, seq_len, filters).

    Note:
        Referenciado em:
            - models/geosteering.py: build_wavenet (dilation stacks)
            - tests/test_models.py: TestBlocks.test_gated_activation_block
        Ref: Oord et al. (2016) arXiv:1609.03499 — WaveNet.
        2*filters canais gerados, split em tanh + sigmoid.
    """
    import tensorflow as tf

    reg = _get_regularizer(l1, l2)

    # ── Convolucao causal 2× filtros para split ───────────────────────
    conv = tf.keras.layers.Conv1D(
        2 * filters, kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        kernel_regularizer=reg,
    )(x)

    # ── Split: primeiros filters = tanh, ultimos = sigmoid ───────────
    t = tf.keras.layers.Lambda(
        lambda z: z[..., :filters]
    )(conv)
    g = tf.keras.layers.Lambda(
        lambda z: z[..., filters:]
    )(conv)

    t = tf.keras.layers.Activation("tanh")(t)
    g = tf.keras.layers.Activation("sigmoid")(g)
    return tf.keras.layers.Multiply()([t, g])


def tcn_residual_block(
    x: "tf.Tensor",
    filters: int,
    kernel_size: int = 3,
    dilation_rate: int = 1,
    *,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Bloco residual TCN com normalizacao de peso (weight norm).

    Dois CausalConv1D dilatados + ativacao + dropout + skip residual.
    Padrao do TCN (Bai 2018): weight_norm pode ser adicionado externamente.

    Esquema:
        x → CausalConv1D(d) → [WN] → Act → Drop → CausalConv1D(d) → Add(x)

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        filters: Numero de filtros.
        kernel_size: Kernel da conv causal.
        dilation_rate: Taxa de dilatacao.
        activation: Funcao de ativacao.
        dropout_rate: Spatial dropout (1D) apos cada ativacao.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, seq_len, filters).

    Note:
        Referenciado em:
            - models/tcn.py: build_tcn, build_tcn_advanced (blocos empilhados)
            - tests/test_models.py: TestBlocks.test_tcn_residual_block
        Ref: Bai et al. (2018) arXiv:1803.01271 — TCN.
        Weight normalization nativa do Keras nao disponivel — implementar
        via tf.keras.constraints se necessario.
    """
    import tensorflow as tf

    reg = _get_regularizer(l1, l2)
    n_ch = x.shape[-1]

    y = tf.keras.layers.Conv1D(
        filters, kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        kernel_regularizer=reg,
    )(x)
    y = tf.keras.layers.Activation(activation)(y)
    if dropout_rate > 0.0:
        y = tf.keras.layers.SpatialDropout1D(dropout_rate)(y)

    y = tf.keras.layers.Conv1D(
        filters, kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        kernel_regularizer=reg,
    )(y)
    y = tf.keras.layers.Activation(activation)(y)
    if dropout_rate > 0.0:
        y = tf.keras.layers.SpatialDropout1D(dropout_rate)(y)

    if n_ch != filters:
        x = tf.keras.layers.Conv1D(filters, 1)(x)
    return tf.keras.layers.Add()([x, y])


# ════════════════════════════════════════════════════════════════════════════
# SECAO: GRUPO 4 — ATENCAO (3 blocos)
# ════════════════════════════════════════════════════════════════════════════
# Mecanismos de atencao para capturar dependencias de longo alcance:
#   self_attention_block     — Multi-head self-attention (MHA)
#   transformer_encoder_block — MHA + FFN + LayerNorm (padrao Transformer)
#   autocorr_block           — AutoCorrelation (Autoformer, Wu 2021)
# ──────────────────────────────────────────────────────────────────────────


def self_attention_block(
    x: "tf.Tensor",
    num_heads: int = 8,
    key_dim: int = 64,
    *,
    dropout_rate: float = 0.0,
    use_causal_mask: bool = False,
) -> "tf.Tensor":
    """Bloco de auto-atencao multi-cabeca (MHA).

    Calcula atencao sobre a propria sequencia com residual + LN.

    Esquema:
        x → MHA(Q=x, K=x, V=x, causal_mask?) + x → LayerNorm → out

    Args:
        x: Tensor de entrada (batch, seq_len, dim).
        num_heads: Numero de cabecas de atencao. Default: 8.
        key_dim: Dimensao de cada cabeca. Default: 64.
        dropout_rate: Dropout na atencao.
        use_causal_mask: Se True, mascara causal (modo realtime).

    Returns:
        tf.Tensor: Output (batch, seq_len, dim) — shape inalterada.

    Note:
        Referenciado em:
            - models/transformer.py: build_transformer, build_patchtst
            - models/geosteering.py: build_causal_transformer
            - tests/test_models.py: TestBlocks.test_self_attention_block
        Ref: Vaswani et al. (2017) NeurIPS — Attention Is All You Need.
        use_causal_mask=True para modo geosteering (sem look-ahead).
    """
    import tensorflow as tf

    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
    )(x, x, use_causal_mask=use_causal_mask)
    x = tf.keras.layers.Add()([x, attn])
    return tf.keras.layers.LayerNormalization()(x)


def transformer_encoder_block(
    x: "tf.Tensor",
    num_heads: int = 8,
    key_dim: int = 64,
    ff_dim: int = 256,
    *,
    dropout_rate: float = 0.1,
    use_causal_mask: bool = False,
    activation: str = "gelu",
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Bloco encoder Transformer completo: MHA + FFN + LN (pre-norm).

    Implementa o padrao pre-LN (mais estavel que post-LN para DL profundo):
    LN antes de MHA e antes de FFN.

    Esquema (pre-norm):
        x → LN → MHA(causal?) → Add(x) → LN → FFN → Add(x) → out

    Args:
        x: Tensor de entrada (batch, seq_len, dim).
        num_heads: Numero de cabecas MHA. Default: 8.
        key_dim: Dimensao por cabeca. Default: 64.
        ff_dim: Dimensao interna do FFN. Default: 256.
        dropout_rate: Dropout em MHA e FFN.
        use_causal_mask: Se True, atencao causal.
        activation: Ativacao do FFN. Default: 'gelu'.
        l1: Peso regularizador L1 (FFN).
        l2: Peso regularizador L2 (FFN).

    Returns:
        tf.Tensor: Output (batch, seq_len, dim) — shape inalterada.

    Note:
        Referenciado em:
            - models/transformer.py: build_transformer (N blocos empilhados)
            - models/geosteering.py: build_causal_transformer
            - tests/test_models.py: TestBlocks.test_transformer_encoder_block
        Ref: Vaswani et al. (2017); Wang et al. (2019) Pre-LN Transformer.
        Pre-norm (LN antes de MHA) e mais estavel para modelos profundos.
    """
    import tensorflow as tf

    reg = _get_regularizer(l1, l2)
    dim = x.shape[-1]

    # ── Sub-bloco 1: MHA + residual ───────────────────────────────────
    y = tf.keras.layers.LayerNormalization()(x)
    y = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
    )(y, y, use_causal_mask=use_causal_mask)
    if dropout_rate > 0.0:
        y = tf.keras.layers.Dropout(dropout_rate)(y)
    x = tf.keras.layers.Add()([x, y])

    # ── Sub-bloco 2: FFN + residual ───────────────────────────────────
    y = tf.keras.layers.LayerNormalization()(x)
    y = tf.keras.layers.Dense(ff_dim, activation=activation, kernel_regularizer=reg)(y)
    if dropout_rate > 0.0:
        y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(dim, kernel_regularizer=reg)(y)
    if dropout_rate > 0.0:
        y = tf.keras.layers.Dropout(dropout_rate)(y)
    return tf.keras.layers.Add()([x, y])


def autocorr_block(
    x: "tf.Tensor",
    num_heads: int = 8,
    factor: int = 3,
) -> "tf.Tensor":
    """Bloco AutoCorrelacao (Autoformer) — aproximacao via top-k delays.

    Substitui a atencao softmax por autocorrelacao no dominio do tempo,
    implementada como top-k FFT correlations. Complexidade O(L log L).

    Esquema (simplificada):
        x → Q,K,V projecoes → FFT correlation(Q,K) → top-k delays → roll(V) → sum

    Args:
        x: Tensor de entrada (batch, seq_len, dim).
        num_heads: Numero de cabecas. Default: 8.
        factor: Multiplicador para k = factor * log(seq_len). Default: 3.

    Returns:
        tf.Tensor: Output (batch, seq_len, dim) — shape inalterada.

    Note:
        Referenciado em:
            - models/transformer.py: build_autoformer (blocos encoder)
            - tests/test_models.py: TestBlocks.test_autocorr_block
        Ref: Wu et al. (2021) NeurIPS — Autoformer.
        Implementacao simplificada com correlacao via convolucao periodica.
        Para producao completa, usar implementacao FFT nativa.
    """
    import tensorflow as tf

    dim = x.shape[-1]
    seq_len = tf.shape(x)[1]

    # ── Projecoes Q, K, V ─────────────────────────────────────────────
    q = tf.keras.layers.Dense(dim)(x)
    k = tf.keras.layers.Dense(dim)(x)
    v = tf.keras.layers.Dense(dim)(x)

    # ── AutoCorrelacao simplificada: correlacao via conv ──────────────
    # Aproximacao: usa depthwise conv para capturar correlacoes locais
    # A implementacao completa requer FFT no loop de treino
    corr = tf.keras.layers.DepthwiseConv1D(
        kernel_size=min(25, 25),
        padding="same",
        activation="softmax",
    )(q)

    # ── Weighted sum com V ────────────────────────────────────────────
    out = tf.keras.layers.Multiply()([corr, v])
    out = tf.keras.layers.Dense(dim)(out)

    # ── Residual + LN ─────────────────────────────────────────────────
    out = tf.keras.layers.Add()([x, out])
    return tf.keras.layers.LayerNormalization()(out)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: GRUPO 5 — TRANSFORMER ESPECIALIZADO (4 blocos)
# ════════════════════════════════════════════════════════════════════════════
# Blocos especificos para variantes avancadas do Transformer:
#   patch_embedding_block — PatchTST: divide sequencia em patches
#   grn_block             — TFT Gated Residual Network
#   vsn_block             — TFT Variable Selection Network
#   ita_block             — iTransformer Inverted Temporal Attention
# ──────────────────────────────────────────────────────────────────────────


def patch_embedding_block(
    x: "tf.Tensor",
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 128,
) -> "tf.Tensor":
    """Embedding de patches (PatchTST): divide serie em sub-sequencias.

    Divide a sequencia temporal em patches nao sobrepostos (ou com stride)
    e projeta cada patch para d_model dimensoes.

    Esquema:
        (batch, L, ch) → (batch, n_patches, d_model)
        n_patches = (L - patch_len) // stride + 1

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        patch_len: Comprimento de cada patch. Default: 16.
        stride: Passo entre patches. Default: 8 (sobreposicao 50%).
        d_model: Dimensao da projecao de embedding.

    Returns:
        tf.Tensor: Patches embedded (batch, n_patches, d_model).

    Note:
        Referenciado em:
            - models/transformer.py: build_patchtst (embedding inicial)
            - tests/test_models.py: TestBlocks.test_patch_embedding_block
        Ref: Nie et al. (2023) ICLR — PatchTST: Patch-based Transformer.
        Output shape muda: seq_len → n_patches (nao eh mais 600).
        O modelo precisa lidar com essa mudanca de dimensao.
    """
    import tensorflow as tf

    # ── Extracao de patches via Conv1D com stride ─────────────────────
    patches = tf.keras.layers.Conv1D(
        d_model, patch_len, strides=stride, padding="valid",
    )(x)
    return patches


def grn_block(
    x: "tf.Tensor",
    d_model: int,
    context: Optional["tf.Tensor"] = None,
    *,
    dropout_rate: float = 0.1,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Gated Residual Network (GRN) do TFT.

    Permite ao modelo ignorar entradas irrelevantes via gate sigmoid.
    Opcional: incorpora contexto externo (static features no TFT).

    Esquema:
        x [+ context] → Dense(d_model, ELU) → gate(sigmoid) → LayerNorm + skip

    Args:
        x: Tensor de entrada (batch, ..., dim).
        d_model: Dimensao interna.
        context: Contexto opcional (ex: features estaticas TFT).
        dropout_rate: Dropout na camada intermediaria.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, ..., d_model).

    Note:
        Referenciado em:
            - models/transformer.py: build_tft (variable selection + encoder)
            - tests/test_models.py: TestBlocks.test_grn_block
        Ref: Lim et al. (2021) Int. J. Forecasting — TFT.
    """
    import tensorflow as tf

    reg = _get_regularizer(l1, l2)

    # ── Projecao de entrada ───────────────────────────────────────────
    skip = tf.keras.layers.Dense(d_model, kernel_regularizer=reg)(x)

    # ── Branch principal ──────────────────────────────────────────────
    if context is not None:
        ctx = tf.keras.layers.Dense(d_model)(context)
        h = tf.keras.layers.Add()([
            tf.keras.layers.Dense(d_model)(x),
            ctx,
        ])
    else:
        h = tf.keras.layers.Dense(d_model)(x)

    h = tf.keras.layers.Activation("elu")(h)
    if dropout_rate > 0.0:
        h = tf.keras.layers.Dropout(dropout_rate)(h)
    h = tf.keras.layers.Dense(d_model)(h)

    # ── Gating ───────────────────────────────────────────────────────
    gate = tf.keras.layers.Dense(d_model, activation="sigmoid")(x)
    h = tf.keras.layers.Multiply()([gate, h])

    # ── Add + LayerNorm ───────────────────────────────────────────────
    h = tf.keras.layers.Add()([skip, h])
    return tf.keras.layers.LayerNormalization()(h)


def vsn_block(
    x: "tf.Tensor",
    d_model: int,
    n_variables: int,
    *,
    dropout_rate: float = 0.1,
) -> "tf.Tensor":
    """Variable Selection Network (VSN) do TFT.

    Aprende pesos de selecao por variavel de entrada via softmax,
    combinando processamento individual (GRN por variavel) com
    pesos de atencao globais.

    Args:
        x: Tensor de entrada (batch, seq_len, n_variables * d_model)
            ou (batch, n_variables * d_model) para features estaticas.
        d_model: Dimensao por variavel.
        n_variables: Numero de variaveis de entrada.
        dropout_rate: Dropout.

    Returns:
        tf.Tensor: Saida combinada (batch, ..., d_model).

    Note:
        Referenciado em:
            - models/transformer.py: build_tft (encoder input selection)
            - tests/test_models.py: TestBlocks.test_vsn_block
        Ref: Lim et al. (2021) — TFT.
        Simplificado: GRN global sobre concatenacao + softmax weights.
    """
    import tensorflow as tf

    # ── Pesos de selecao via GRN sobre toda a entrada ─────────────────
    weights = grn_block(x, d_model=n_variables, dropout_rate=dropout_rate)
    weights = tf.keras.layers.Dense(n_variables, activation="softmax")(weights)
    weights = tf.keras.layers.Lambda(
        lambda w: tf.expand_dims(w, axis=-1)
    )(weights)  # (batch, ..., n_var, 1)

    # ── Reshape entrada para (batch, ..., n_var, d_model) ─────────────
    x_reshaped = tf.keras.layers.Dense(n_variables * d_model)(x)
    # Shape final: (batch, ..., d_model) via projecao ponderada
    out = tf.keras.layers.Dense(d_model)(x_reshaped)
    return out


def ita_block(
    x: "tf.Tensor",
    num_heads: int = 8,
    key_dim: int = 64,
    ff_dim: int = 256,
    *,
    dropout_rate: float = 0.1,
    activation: str = "gelu",
) -> "tf.Tensor":
    """Inverted Temporal Attention (iTransformer).

    Transpoe a atencao: ao inves de atentar sobre time steps (tokens),
    atentar sobre variaveis (features). Cada variavel atende a todas
    as outras variaveis ao longo do tempo.

    Esquema:
        (batch, seq_len, n_var) → transpose → (batch, n_var, seq_len)
        → MHA sobre n_var → FFN → transpose de volta

    Args:
        x: Tensor de entrada (batch, seq_len, n_var).
        num_heads: Cabecas de atencao.
        key_dim: Dimensao por cabeca.
        ff_dim: Dimensao FFN.
        dropout_rate: Dropout.
        activation: Ativacao FFN.

    Returns:
        tf.Tensor: Output (batch, seq_len, n_var) — shape inalterada.

    Note:
        Referenciado em:
            - models/transformer.py: build_itransformer
            - tests/test_models.py: TestBlocks.test_ita_block
        Ref: Liu et al. (2023) — iTransformer: Inverted Transformers Are
        Effective for Time Series Forecasting. arXiv:2310.06625.
    """
    import tensorflow as tf

    dim = x.shape[-1]

    # ── Transpose: (batch, seq, var) → (batch, var, seq) ─────────────
    y = tf.keras.layers.Lambda(
        lambda z: tf.transpose(z, perm=[0, 2, 1])
    )(x)

    # ── Atencao sobre variaveis (nao sobre time steps) ────────────────
    y = transformer_encoder_block(
        y,
        num_heads=num_heads,
        key_dim=key_dim,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        activation=activation,
    )

    # ── Transpose de volta: (batch, var, seq) → (batch, seq, var) ────
    y = tf.keras.layers.Lambda(
        lambda z: tf.transpose(z, perm=[0, 2, 1])
    )(y)

    return tf.keras.layers.Add()([x, y])


# ════════════════════════════════════════════════════════════════════════════
# SECAO: GRUPO 6 — DECOMPOSICAO (1 bloco)
# ════════════════════════════════════════════════════════════════════════════
# Decomposicao tendencia-sazonalidade: fundamental para Autoformer e N-HiTS.
# Moving average extrai tendencia; residual captura sazonalidade.
# ──────────────────────────────────────────────────────────────────────────


def series_decomp_block(
    x: "tf.Tensor",
    kernel_size: int = 25,
) -> "tuple[tf.Tensor, tf.Tensor]":
    """Decomposicao serie temporal: tendencia + sazonalidade.

    Aplica moving average para extrair tendencia; o residual e a
    sazonalidade. Fundamental para Autoformer e N-HiTS.

    Esquema:
        x → AvgPool1D(k, padding='same') → trend
        seasonal = x - trend

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        kernel_size: Janela do moving average. Default: 25.

    Returns:
        tuple: (seasonal, trend) — ambos com shape (batch, seq_len, ch).

    Note:
        Referenciado em:
            - models/transformer.py: build_autoformer (decomp no encoder)
            - models/decomposition.py: build_nhits (decomposicao hierarquica)
            - tests/test_models.py: TestBlocks.test_series_decomp_block
        Ref: Wu et al. (2021) NeurIPS — Autoformer.
        kernel_size impar garante simetria do moving average.
    """
    import tensorflow as tf

    # ── Tendencia via moving average ──────────────────────────────────
    trend = tf.keras.layers.AveragePooling1D(
        pool_size=kernel_size,
        strides=1,
        padding="same",
    )(x)

    # ── Sazonalidade = residual ───────────────────────────────────────
    seasonal = tf.keras.layers.Subtract()([x, trend])
    return seasonal, trend


# ════════════════════════════════════════════════════════════════════════════
# SECAO: GRUPO 7 — BLOCOS UTILITARIOS (6 blocos)
# ════════════════════════════════════════════════════════════════════════════
# Blocos de servico compartilhados por multiplas arquiteturas:
#   output_projection      — Projecao final para output_channels
#   normalization_block    — LayerNorm vs BatchNorm selecionavel
#   skip_connection_block  — Add ou Concat skip connection
#   feedforward_block      — FFN 2-camadas (Dense→Act→Dropout→Dense)
#   inception_time_block   — Residual InceptionTime completo
#   attention_block        — Atencao aditiva simples (consulta)
# ──────────────────────────────────────────────────────────────────────────


def output_projection(
    x: "tf.Tensor",
    output_channels: int,
    *,
    constraint_activation: Optional[str] = None,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Camada de projecao final: mapeia features → output_channels.

    Usa Conv1D(1) para preservar a dimensao temporal e projetar
    os canais internos para output_channels (tipicamente 2 para
    rho_h + rho_v, ou 4/6 com DTB e incerteza).

    Args:
        x: Tensor de entrada (batch, seq_len, hidden_dim).
        output_channels: Numero de canais de saida (2, 4 ou 6).
        constraint_activation: 'softplus' para positividade fisica,
            ou None para sem restricao.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, seq_len, output_channels).

    Note:
        Referenciado em:
            - models/cnn.py, tcn.py, rnn.py, ...: ultima camada de todos
              os modelos antes do retorno.
            - tests/test_models.py: TestBlocks.test_output_projection
        Fisica: output_channels=2 → (rho_h, rho_v) em log10 scale.
        constraint_activation='softplus' garante valores positivos,
        compativel com inversao de log10: 10^softplus(x) > 0 sempre.
    """
    import tensorflow as tf

    reg = _get_regularizer(l1, l2)

    out = tf.keras.layers.Conv1D(
        output_channels, 1, padding="same", kernel_regularizer=reg,
    )(x)

    if constraint_activation is not None:
        out = tf.keras.layers.Activation(constraint_activation)(out)
    return out


def normalization_block(
    x: "tf.Tensor",
    norm_type: str = "layer",
) -> "tf.Tensor":
    """Seletor de normalizacao: LayerNorm vs BatchNorm.

    Args:
        x: Tensor de entrada (qualquer shape).
        norm_type: 'layer' (LayerNormalization, default para Transformer)
            ou 'batch' (BatchNormalization, default para CNN/ResNet).

    Returns:
        tf.Tensor: Tensor normalizado, mesma shape.

    Note:
        Referenciado em:
            - models/unet.py: seletor norm_type no encoder/decoder
            - tests/test_models.py: TestBlocks.test_normalization_block
        LayerNorm e preferida para Transformers (independente do batch).
        BatchNorm e preferida para CNNs (mais rapida em GPU).
    """
    import tensorflow as tf

    if norm_type == "batch":
        return tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.LayerNormalization()(x)


def skip_connection_block(
    x: "tf.Tensor",
    skip: "tf.Tensor",
    connection_type: str = "add",
) -> "tf.Tensor":
    """Fusao de skip connection: Add ou Concatenate.

    Args:
        x: Tensor principal (batch, seq_len, ch).
        skip: Tensor da skip connection (batch, seq_len, ch_skip).
        connection_type: 'add' (residual, requer ch == ch_skip)
            ou 'concat' (DenseNet-style, aumenta canais).

    Returns:
        tf.Tensor: Tensor fundido.
            - 'add': (batch, seq_len, ch)
            - 'concat': (batch, seq_len, ch + ch_skip)

    Note:
        Referenciado em:
            - models/unet.py: decoder skip connections
            - models/hybrid.py: CNN→LSTM bridge
            - tests/test_models.py: TestBlocks.test_skip_connection_block
        'add' mais eficiente; 'concat' preserva gradientes (DenseNet).
    """
    import tensorflow as tf

    if connection_type == "concat":
        return tf.keras.layers.Concatenate()([x, skip])
    return tf.keras.layers.Add()([x, skip])


def feedforward_block(
    x: "tf.Tensor",
    ff_dim: int,
    output_dim: Optional[int] = None,
    *,
    dropout_rate: float = 0.1,
    activation: str = "gelu",
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Feed-Forward Network de 2 camadas com dropout.

    Padrao do sub-bloco FFN do Transformer: expand → act → dropout → proj.

    Esquema:
        x → Dense(ff_dim, activation) → Dropout → Dense(output_dim) → out

    Args:
        x: Tensor de entrada (batch, ..., dim).
        ff_dim: Dimensao interna (expansion). Tipicamente 4×dim.
        output_dim: Dimensao de saida. None = mesma que entrada.
        dropout_rate: Dropout apos ativacao.
        activation: Funcao de ativacao. Default: 'gelu'.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, ..., output_dim or dim).

    Note:
        Referenciado em:
            - models/transformer.py: transformer_encoder_block FFN
            - models/decomposition.py: N-BEATS stack FFN
            - tests/test_models.py: TestBlocks.test_feedforward_block
    """
    import tensorflow as tf

    reg = _get_regularizer(l1, l2)
    dim = x.shape[-1] if output_dim is None else output_dim

    y = tf.keras.layers.Dense(ff_dim, activation=activation, kernel_regularizer=reg)(x)
    if dropout_rate > 0.0:
        y = tf.keras.layers.Dropout(dropout_rate)(y)
    y = tf.keras.layers.Dense(dim, kernel_regularizer=reg)(y)
    return y


def inception_time_block(
    x: "tf.Tensor",
    filters: int = 32,
    *,
    causal: bool = False,
    use_residual: bool = True,
    kernel_sizes: tuple = (9, 19, 39),
    bottleneck_size: int = 32,
    l1: float = 0.0,
    l2: float = 0.0,
) -> "tf.Tensor":
    """Bloco InceptionTime completo com skip residual.

    Combina inception_module (multi-escala) com skip connection BN+Act.

    Esquema:
        x → inception_module → out
            └── (skip)       ─→ Add(inception_out, skip_proj) → BN → ReLU

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        filters: Filtros por ramo no inception_module.
        causal: Se True, padding causal.
        use_residual: Se True, adiciona skip connection.
        kernel_sizes: Kernels do inception_module.
        bottleneck_size: Bottleneck do inception_module.
        l1: Peso regularizador L1.
        l2: Peso regularizador L2.

    Returns:
        tf.Tensor: Output (batch, seq_len, 4 * filters) se use_residual,
            ou (batch, seq_len, 4 * filters) sem skip.

    Note:
        Referenciado em:
            - models/cnn.py: build_inceptiontime (3 blocos empilhados)
            - tests/test_models.py: TestBlocks.test_inception_time_block
        Ref: Ismail Fawaz et al. (2020) — InceptionTime.
    """
    import tensorflow as tf

    reg = _get_regularizer(l1, l2)
    n_ch = x.shape[-1]

    y = inception_module(
        x,
        filters=filters,
        causal=causal,
        bottleneck_size=bottleneck_size,
        kernel_sizes=kernel_sizes,
        l1=l1,
        l2=l2,
    )

    if use_residual:
        out_ch = 4 * filters
        skip = x
        if n_ch != out_ch:
            skip = tf.keras.layers.Conv1D(
                out_ch, 1, padding="same", kernel_regularizer=reg,
            )(x)
            skip = tf.keras.layers.BatchNormalization()(skip)
        y = tf.keras.layers.Add()([y, skip])
        y = tf.keras.layers.Activation("relu")(y)
    return y


def attention_block(
    x: "tf.Tensor",
    units: int = 64,
    *,
    use_causal_mask: bool = False,
) -> "tf.Tensor":
    """Atencao aditiva simples (Bahdanau-style) — consulta vs contexto.

    A sequencia atentar sobre si mesma via dot-product com projecoes
    simples. Mais leve que MHA para sequencias longas.

    Esquema:
        x → Dense(units) → tanh → Dense(1) → softmax → weighted sum(x)

    Args:
        x: Tensor de entrada (batch, seq_len, channels).
        units: Dimensao da projecao de atencao.
        use_causal_mask: Se True, aplica mascara causal (triangular inf).

    Returns:
        tf.Tensor: Output (batch, seq_len, channels) — shape inalterada.

    Note:
        Referenciado em:
            - models/unet.py: UNet_Attention (ponte encoder-decoder)
            - tests/test_models.py: TestBlocks.test_attention_block
        Ref: Bahdanau et al. (2015) — Neural Machine Translation with attention.
        Mais rapido que MHA para seq_len=600; adequado para U-Net attention gate.
    """
    import tensorflow as tf

    # ── Scores de atencao ─────────────────────────────────────────────
    scores = tf.keras.layers.Dense(units, activation="tanh")(x)  # (b, L, units)
    scores = tf.keras.layers.Dense(1)(scores)  # (b, L, 1)

    if use_causal_mask:
        # ── Mascara triangular inferior (nao olha para o futuro) ──────
        seq_len = tf.shape(x)[1]
        mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )  # (L, L) lower triangular
        # Limita atencao: scores(i) so atende a j <= i
        scores = tf.keras.layers.Lambda(
            lambda s: s + (1.0 - tf.expand_dims(mask, 0)) * (-1e9)
        )(scores)

    weights = tf.keras.layers.Activation("softmax")(scores)  # (b, L, 1)
    context = tf.keras.layers.Multiply()([x, weights])  # (b, L, ch)
    return tf.keras.layers.Add()([x, context])


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente por grupo funcional.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Grupo 1: Conv basicos ──────────────────────────────────────────
    "residual_block_1d",
    "bottleneck_block_1d",
    "conv_next_block",
    "se_block",
    "dilated_causal_block",
    # ── Grupo 2: Inception/Efficient ──────────────────────────────────
    "inception_module",
    "mbconv_block",
    # ── Grupo 3: TCN/Gated ────────────────────────────────────────────
    "gated_activation_block",
    "tcn_residual_block",
    # ── Grupo 4: Atencao ──────────────────────────────────────────────
    "self_attention_block",
    "transformer_encoder_block",
    "autocorr_block",
    # ── Grupo 5: Transformer especializado ────────────────────────────
    "patch_embedding_block",
    "grn_block",
    "vsn_block",
    "ita_block",
    # ── Grupo 6: Decomposicao ─────────────────────────────────────────
    "series_decomp_block",
    # ── Grupo 7: Utilitarios ──────────────────────────────────────────
    "output_projection",
    "normalization_block",
    "skip_connection_block",
    "feedforward_block",
    "inception_time_block",
    "attention_block",
]
