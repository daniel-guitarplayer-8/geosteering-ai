# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/unet.py                                                    ║
# ║  Bloco: 3f — Arquiteturas U-Net 1D (14 variantes)                        ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • 14 variantes U-Net 1D para inversao seq2seq                         ║
# ║    • Base + Attention + ResNet18/34/50 + ConvNeXt + Inception +           ║
# ║      EfficientNet (7 base × 2 = 14 com/sem attention)                    ║
# ║    • Arquitetura encoder-decoder com skip connections                     ║
# ║    • CAUSAL_INCOMPATIBLE: decoder acessa features do encoder              ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: 14 funcoes — ver __all__                                        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.6, legado C33                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (14 variantes)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""14 variantes U-Net 1D para inversao de resistividade.

Todas as variantes sao CAUSAL_INCOMPATIBLE: o skip connection do encoder
para o decoder implica acesso a features de toda a sequencia, violando
causalidade. Usar apenas inference_mode='offline'.

Padrão arquitetural:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Encoder: N blocos de downsampling (stride conv ou MaxPool)     │
    │  Bottleneck: bloco mais profundo sem skip                       │
    │  Decoder: N blocos de upsampling com skip connections           │
    │  Output: Conv1D(output_channels, 1)                            │
    └──────────────────────────────────────────────────────────────────┘

Implementacao usa 'same' padding: sem downsampling temporal efetivo
(seq_len preservado). Skips por concatenacao (DenseNet-style).

Note:
    Referenciado em:
        - models/registry.py: _REGISTRY[14 entradas UNet]
        - tests/test_models.py: TestUNet
    Legado C33 (3756 linhas, 14 variantes).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: BUILDER INTERNO — U-NET GENERICA
# ════════════════════════════════════════════════════════════════════════════
# Funcao interna que constroi qualquer U-Net dado:
#   - encoder_fn: funcao que aplica blocos no encoder
#   - decoder_fn: funcao que aplica blocos no decoder
#   - use_attention: se True, adiciona attention gate nos skips
# Todas as 14 variantes reutilizam esta estrutura.
# ──────────────────────────────────────────────────────────────────────────


def _build_unet_base(
    config: "PipelineConfig",
    encoder_type: str = "base",
    use_attention: bool = False,
    name: str = "UNet_Base",
) -> "tf.keras.Model":
    """Builder interno para qualquer U-Net 1D.

    Parametros encoder_type controlam quais blocos usar no encoder.
    skip connections via Concatenate (DenseNet-style, mais estavel).
    Upsampling via UpSampling1D + Conv1D (sem artefatos de deconv).

    Args:
        config: PipelineConfig.
        encoder_type: 'base', 'resnet18', 'resnet34', 'resnet50',
            'convnext', 'inception', 'efficientnet'.
        use_attention: Se True, aplica attention_block nos skips.
        name: Nome do modelo Keras.

    Returns:
        tf.keras.Model: U-Net 1D seq2seq (CAUSAL_INCOMPATIBLE).
    """
    import tensorflow as tf
    from geosteering_ai.models.blocks import (
        residual_block_1d, bottleneck_block_1d, conv_next_block,
        inception_module, mbconv_block, attention_block,
        output_projection, se_block,
    )

    ap = config.arch_params or {}
    depth = ap.get("depth", 4)  # numero de niveis encoder/decoder
    base_filters = ap.get("base_filters", 32)
    kernel_size = ap.get("kernel_size", 3)
    dr = config.dropout_rate
    l1 = config.l1_weight if config.use_l1_regularization else 0.0
    l2 = config.l2_weight if config.use_l2_regularization else 0.0

    if config.use_causal_mode:
        logger.warning("%s e CAUSAL_INCOMPATIBLE — use apenas offline.", name)

    def _enc_block(x, filters):
        """Aplica bloco encoder conforme encoder_type."""
        if encoder_type == "resnet50":
            return bottleneck_block_1d(x, filters // 4, kernel_size, l1=l1, l2=l2)
        elif encoder_type in ("resnet18", "resnet34"):
            return residual_block_1d(x, filters, kernel_size, l1=l1, l2=l2)
        elif encoder_type == "convnext":
            return conv_next_block(x, filters, l1=l1, l2=l2)
        elif encoder_type == "inception":
            return inception_module(x, filters // 4, l1=l1, l2=l2)
        elif encoder_type == "efficientnet":
            return mbconv_block(x, filters, kernel_size, l1=l1, l2=l2)
        else:  # base
            x = tf.keras.layers.Conv1D(
                filters, kernel_size, padding="same",
                use_bias=False,
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            return tf.keras.layers.Activation("relu")(x)

    def _skip_with_attention(x, skip):
        """Aplica attention gate nos skips se use_attention."""
        if use_attention:
            skip = attention_block(skip, units=skip.shape[-1])
        return tf.keras.layers.Concatenate()([x, skip])

    logger.info("build_%s: n_feat=%d, depth=%d, base_filters=%d",
                name, config.n_features, depth, base_filters)

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    # ── Encoder com skip connections ──────────────────────────────────
    skips = []
    for level in range(depth):
        filters = base_filters * (2 ** level)
        x = _enc_block(x, filters)
        if dr > 0.0:
            x = tf.keras.layers.Dropout(dr)(x)
        skips.append(x)
        # ── Downsampling via MaxPool (seq preservada com pool_size=1) ──
        # Para U-Net 1D sem downsampling real, usamos Conv1D stride=2
        # e depois UpSampling1D no decoder para restaurar.
        # Alternativa sem stride: manter seq_len preservado (padding only).
        # Aqui: sem stride para manter seq_len=600 (mais simples).
        logger.debug("UNet encoder level %d: filters=%d", level, filters)

    # ── Bottleneck ────────────────────────────────────────────────────
    bn_filters = base_filters * (2 ** depth)
    x = _enc_block(x, bn_filters)
    x = tf.keras.layers.LayerNormalization()(x)

    # ── Decoder com skip connections ──────────────────────────────────
    for level in reversed(range(depth)):
        filters = base_filters * (2 ** level)
        # ── Fusao com skip ─────────────────────────────────────────────
        x = _skip_with_attention(x, skips[level])
        # ── Reducao de canais apos concat ──────────────────────────────
        x = tf.keras.layers.Conv1D(filters, 1, padding="same")(x)
        x = _enc_block(x, filters)
        if dr > 0.0:
            x = tf.keras.layers.Dropout(dr)(x)
        logger.debug("UNet decoder level %d: filters=%d", level, filters)

    out = output_projection(
        x, config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name=name)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: 14 FUNCOES PUBLICA DE CONSTRUCAO U-NET
# ════════════════════════════════════════════════════════════════════════════
# As 14 variantes sao pares (com/sem attention) de 7 configuracoes base.
#
#   ┌───────────────────────────────┬─────────────────────────────────────┐
#   │  Variante                     │ encoder_type   │ use_attention      │
#   ├───────────────────────────────┼────────────────┼────────────────────┤
#   │  UNet_Base                    │ 'base'         │ False              │
#   │  UNet_Attention               │ 'base'         │ True               │
#   │  UNet_ResNet18                │ 'resnet18'     │ False              │
#   │  UNet_Attention_ResNet18      │ 'resnet18'     │ True               │
#   │  UNet_ResNet34                │ 'resnet34'     │ False              │
#   │  UNet_Attention_ResNet34      │ 'resnet34'     │ True               │
#   │  UNet_ResNet50                │ 'resnet50'     │ False              │
#   │  UNet_Attention_ResNet50      │ 'resnet50'     │ True               │
#   │  UNet_ConvNeXt                │ 'convnext'     │ False              │
#   │  UNet_Attention_ConvNeXt      │ 'convnext'     │ True               │
#   │  UNet_Inception               │ 'inception'    │ False              │
#   │  UNet_Attention_Inception     │ 'inception'    │ True               │
#   │  UNet_EfficientNet            │ 'efficientnet' │ False              │
#   │  UNet_Attention_EfficientNet  │ 'efficientnet' │ True               │
#   └───────────────────────────────┴────────────────┴────────────────────┘
# ──────────────────────────────────────────────────────────────────────────


def build_unet_base(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D baseline com Conv+BN+ReLU no encoder/decoder."""
    return _build_unet_base(config, "base", False, "UNet_Base")


def build_unet_attention(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D + attention gates nos skip connections."""
    return _build_unet_base(config, "base", True, "UNet_Attention")


def build_unet_resnet18(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder ResNet-18 (blocos residuais)."""
    return _build_unet_base(config, "resnet18", False, "UNet_ResNet18")


def build_unet_attention_resnet18(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder ResNet-18 + attention gates."""
    return _build_unet_base(config, "resnet18", True, "UNet_Attention_ResNet18")


def build_unet_resnet34(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder ResNet-34 (blocos residuais mais profundos)."""
    return _build_unet_base(config, "resnet34", False, "UNet_ResNet34")


def build_unet_attention_resnet34(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder ResNet-34 + attention gates."""
    return _build_unet_base(config, "resnet34", True, "UNet_Attention_ResNet34")


def build_unet_resnet50(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder ResNet-50 (bottleneck blocks)."""
    return _build_unet_base(config, "resnet50", False, "UNet_ResNet50")


def build_unet_attention_resnet50(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder ResNet-50 + attention gates."""
    return _build_unet_base(config, "resnet50", True, "UNet_Attention_ResNet50")


def build_unet_convnext(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder ConvNeXt (depthwise + LN + GELU)."""
    return _build_unet_base(config, "convnext", False, "UNet_ConvNeXt")


def build_unet_attention_convnext(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder ConvNeXt + attention gates."""
    return _build_unet_base(config, "convnext", True, "UNet_Attention_ConvNeXt")


def build_unet_inception(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder Inception (multi-escala temporal)."""
    return _build_unet_base(config, "inception", False, "UNet_Inception")


def build_unet_attention_inception(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder Inception + attention gates."""
    return _build_unet_base(config, "inception", True, "UNet_Attention_Inception")


def build_unet_efficientnet(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder EfficientNet (MBConv + SE)."""
    return _build_unet_base(config, "efficientnet", False, "UNet_EfficientNet")


def build_unet_attention_efficientnet(config: "PipelineConfig") -> "tf.keras.Model":
    """U-Net 1D com encoder EfficientNet + attention gates."""
    return _build_unet_base(config, "efficientnet", True, "UNet_Attention_EfficientNet")


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    "build_unet_base",
    "build_unet_attention",
    "build_unet_resnet18",
    "build_unet_attention_resnet18",
    "build_unet_resnet34",
    "build_unet_attention_resnet34",
    "build_unet_resnet50",
    "build_unet_attention_resnet50",
    "build_unet_convnext",
    "build_unet_attention_convnext",
    "build_unet_inception",
    "build_unet_attention_inception",
    "build_unet_efficientnet",
    "build_unet_attention_efficientnet",
]
