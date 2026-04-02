# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/cnn.py                                                     ║
# ║  Bloco: 3b — Arquiteturas CNN (ResNet, ConvNeXt, Inception, CNN_1D)       ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • 7 arquiteturas CNN para inversao 1D seq2seq (batch, 600, ch)        ║
# ║    • ResNet-18★ (default), ResNet-34, ResNet-50 (residual deep)           ║
# ║    • ConvNeXt (depthwise + LN moderno), CNN_1D (baseline simples)        ║
# ║    • InceptionNet, InceptionTime (multi-escala temporal)                  ║
# ║    • Causal/acausal via config.use_causal_mode                           ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: ~7 funcoes — ver __all__                                        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.2 (CNN), legado C28               ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (7 arquiteturas)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Arquiteturas CNN 1D para inversao de resistividade (seq2seq).

Todas as funcoes recebem `config: PipelineConfig` e retornam
`tf.keras.Model` com:
    - Input: (batch, config.sequence_length, config.n_features)
    - Output: (batch, config.sequence_length, config.output_channels)

Imports de TensorFlow sao lazy (dentro de cada funcao).

Note:
    Referenciado em:
        - models/registry.py: ModelRegistry._build_cnn()
        - tests/test_models.py: TestCNN
    Ref: docs/ARCHITECTURE_v2.md secao 5.2. Legado C28 (3 arqs originais)
    + C29 (CNN_1D).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: UTILITARIOS INTERNOS
# ════════════════════════════════════════════════════════════════════════════
# Funcoes auxiliares compartilhadas por todas as arquiteturas CNN:
#   _stem(): stem conv1D inicial
#   _apply_se(): aplicacao condicional de SE block
#   _get_reg(): regularizador kernel a partir de config
# ──────────────────────────────────────────────────────────────────────────


def _get_reg(config: "PipelineConfig"):
    """Constroi regularizador a partir de config.

    Args:
        config: PipelineConfig com use_l1/l2_regularization e pesos.

    Returns:
        tf.keras.regularizers.Regularizer | None.
    """
    import tensorflow as tf

    l1 = config.l1_weight if config.use_l1_regularization else 0.0
    l2 = config.l2_weight if config.use_l2_regularization else 0.0
    if l1 == 0.0 and l2 == 0.0:
        return None
    return tf.keras.regularizers.L1L2(l1=l1, l2=l2)


def _stem_block(x, filters: int, kernel_size: int, causal: bool, reg):
    """Stem inicial: Conv → BN → ReLU.

    Args:
        x: Tensor de entrada (batch, seq_len, n_features).
        filters: Filtros do stem.
        kernel_size: Kernel do stem.
        causal: Se True, padding causal.
        reg: Regularizador kernel.

    Returns:
        tf.Tensor: (batch, seq_len, filters).
    """
    import tensorflow as tf

    pad = "causal" if causal else "same"
    x = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding=pad,
        kernel_regularizer=reg,
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


# ════════════════════════════════════════════════════════════════════════════
# SECAO: RESNET-18 (DEFAULT ★)
# ════════════════════════════════════════════════════════════════════════════
# ResNet-18 adaptado para seq2seq 1D: 4 stages, 2 blocos residuais por stage.
# Filtros: 64 → 128 → 256 → 512. Sem downsampling temporal (padding='same').
# Tier 1: validado, estavel, melhor tradeoff params/performance.
# Ref: He et al. (2016) CVPR — Deep Residual Learning.
# ──────────────────────────────────────────────────────────────────────────


def build_resnet18(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi ResNet-18 1D adaptado para inversao seq2seq.

    4 stages com 2 blocos residuais cada. Preserva seq_len = 600.
    Default ★ do pipeline (Tier 1, validado, estavel).

    Arquitetura:
        ┌─────────────────────────────────────────────────────┐
        │  Input (batch, 600, n_feat)                        │
        │  ↓ Stem: Conv(64, k=7) + BN + ReLU                │
        │  ↓ Stage 1: 2× ResBlock(64,  k=3)                 │
        │  ↓ Stage 2: 2× ResBlock(128, k=3)                 │
        │  ↓ Stage 3: 2× ResBlock(256, k=3)                 │
        │  ↓ Stage 4: 2× ResBlock(512, k=3)                 │
        │  ↓ Output: Conv(output_ch, k=1)                   │
        │  Output (batch, 600, output_channels)              │
        └─────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com:
            - sequence_length: 600 (errata)
            - n_features: canais de entrada
            - output_channels: 2, 4 ou 6
            - use_causal_mode: padding causal se realtime
            - use_se_block: SE nos blocos residuais
            - se_reduction: fator SE
            - dropout_rate: dropout nos blocos
            - arch_params: override granular (filters, stem_k, etc.)

    Returns:
        tf.keras.Model: Modelo ResNet-18 seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['ResNet_18']
            - tests/test_models.py: TestCNN.test_resnet18_forward
        Causal mode: padding='causal' em todos os Conv1D.
        SE block: aplicado apos segundo Conv de cada ResBlock se ativo.
        Ref: docs/ARCHITECTURE_v2.md secao 5.2.1.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import (
        output_projection,
        residual_block_1d,
        se_block,
    )

    # ── Resolucao de hiperparametros (config + arch_params overrides) ─
    ap = config.arch_params or {}
    stage_filters = ap.get("stage_filters", [64, 128, 256, 512])
    stem_filters = ap.get("stem_filters", 64)
    stem_k = ap.get("stem_k", 7)
    blocks_per_stage = ap.get("blocks_per_stage", [2, 2, 2, 2])
    kernel_size = ap.get("kernel_size", 3)
    causal = config.use_causal_mode
    reg = _get_reg(config)
    dr = config.dropout_rate
    use_se = config.use_se_block

    logger.info(
        "build_resnet18: n_feat=%d, out_ch=%d, causal=%s, se=%s",
        config.n_features,
        config.output_channels,
        causal,
        use_se,
    )

    # ── Input ─────────────────────────────────────────────────────────
    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── Stem ─────────────────────────────────────────────────────────
    x = _stem_block(inp, stem_filters, stem_k, causal, reg)

    # ── 4 Stages com blocos residuais ────────────────────────────────
    for stage_i, (n_filt, n_blocks) in enumerate(zip(stage_filters, blocks_per_stage)):
        for _ in range(n_blocks):
            x = residual_block_1d(
                x,
                n_filt,
                kernel_size,
                causal=causal,
                dropout_rate=dr,
                l1=config.l1_weight if config.use_l1_regularization else 0.0,
                l2=config.l2_weight if config.use_l2_regularization else 0.0,
            )
        if use_se:
            x = se_block(x, reduction=config.se_reduction)
        logger.debug("ResNet18 Stage %d: filters=%d, se=%s", stage_i + 1, n_filt, use_se)

    # ── Output projection ─────────────────────────────────────────────
    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )

    return tf.keras.Model(inputs=inp, outputs=out, name="ResNet_18")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: RESNET-34
# ════════════════════════════════════════════════════════════════════════════
# ResNet-34: stages [3, 4, 6, 3] blocos — mais profundo que ResNet-18.
# Bom para datasets grandes com 1000+ modelos geologicos.
# Ref: He et al. (2016) CVPR.
# ──────────────────────────────────────────────────────────────────────────


def build_resnet34(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi ResNet-34 1D: 4 stages com [3,4,6,3] blocos residuais.

    Mais profundo que ResNet-18 para datasets grandes (1000+ modelos).

    Args:
        config: PipelineConfig (mesmos campos que build_resnet18).

    Returns:
        tf.keras.Model: ResNet-34 seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['ResNet_34']
            - tests/test_models.py: TestCNN.test_resnet34_forward
        Ref: He et al. (2016) CVPR — ResNet-34 config: [3,4,6,3].
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import (
        output_projection,
        residual_block_1d,
        se_block,
    )

    ap = config.arch_params or {}
    stage_filters = ap.get("stage_filters", [64, 128, 256, 512])
    stem_filters = ap.get("stem_filters", 64)
    stem_k = ap.get("stem_k", 7)
    blocks_per_stage = ap.get("blocks_per_stage", [3, 4, 6, 3])  # ResNet-34
    kernel_size = ap.get("kernel_size", 3)
    causal = config.use_causal_mode
    reg = _get_reg(config)
    dr = config.dropout_rate
    use_se = config.use_se_block

    logger.info(
        "build_resnet34: n_feat=%d, out_ch=%d", config.n_features, config.output_channels
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = _stem_block(inp, stem_filters, stem_k, causal, reg)

    for stage_i, (n_filt, n_blocks) in enumerate(zip(stage_filters, blocks_per_stage)):
        for _ in range(n_blocks):
            x = residual_block_1d(
                x,
                n_filt,
                kernel_size,
                causal=causal,
                dropout_rate=dr,
                l1=config.l1_weight if config.use_l1_regularization else 0.0,
                l2=config.l2_weight if config.use_l2_regularization else 0.0,
            )
        if use_se:
            x = se_block(x, reduction=config.se_reduction)

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="ResNet_34")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: RESNET-50
# ════════════════════════════════════════════════════════════════════════════
# ResNet-50: usa blocos bottleneck (1x1→3x3→1x1) para maior eficiencia.
# Canais de saida dos stages: 256, 512, 1024, 2048 (4x expansion).
# Ref: He et al. (2016) CVPR.
# ──────────────────────────────────────────────────────────────────────────


def build_resnet50(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi ResNet-50 1D: bottleneck blocks [3,4,6,3].

    Maior capacidade via bottleneck (1x1 compress + 3x3 + 1x1 expand).
    Stages: 256, 512, 1024, 2048 canais (expansion=4).

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: ResNet-50 seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['ResNet_50']
            - tests/test_models.py: TestCNN.test_resnet50_forward
        Ref: He et al. (2016) CVPR — ResNet-50 com bottleneck.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import (
        bottleneck_block_1d,
        output_projection,
        se_block,
    )

    ap = config.arch_params or {}
    stage_filters = ap.get("stage_filters", [64, 128, 256, 512])  # pre-expansion
    stem_filters = ap.get("stem_filters", 64)
    blocks_per_stage = ap.get("blocks_per_stage", [3, 4, 6, 3])
    kernel_size = ap.get("kernel_size", 3)
    expansion = ap.get("expansion", 4)
    causal = config.use_causal_mode
    use_se = config.use_se_block

    logger.info(
        "build_resnet50: n_feat=%d, out_ch=%d", config.n_features, config.output_channels
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = _stem_block(inp, stem_filters, 7, causal, _get_reg(config))

    for stage_i, (n_filt, n_blocks) in enumerate(zip(stage_filters, blocks_per_stage)):
        for _ in range(n_blocks):
            x = bottleneck_block_1d(
                x,
                n_filt,
                kernel_size,
                causal=causal,
                expansion=expansion,
                l1=config.l1_weight if config.use_l1_regularization else 0.0,
                l2=config.l2_weight if config.use_l2_regularization else 0.0,
            )
        if use_se:
            x = se_block(x, reduction=config.se_reduction)

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="ResNet_50")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CONVNEXT
# ════════════════════════════════════════════════════════════════════════════
# ConvNeXt 1D: modernizacao do ResNet com ideias do ViT.
# Depthwise conv (k=7), LayerNorm, GELU, expansion MLP 4x.
# Stages: 96 → 192 → 384 → 768 filtros (padrao ConvNeXt-Tiny).
# Ref: Liu et al. (2022) CVPR — A ConvNet for the 2020s.
# ──────────────────────────────────────────────────────────────────────────


def build_convnext(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi ConvNeXt 1D (Tiny variant) para inversao seq2seq.

    Substitui Conv+BN+ReLU por DepthwiseConv+LN+GELU+MLP.
    Tier 1 para dados grandes com melhor generalizacao que ResNet.

    Arquitetura:
        Input → Patchify(4) → Stage1(96, 3b) → Stage2(192, 3b)
              → Stage3(384, 9b) → Stage4(768, 3b) → Output

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: ConvNeXt 1D seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['ConvNeXt']
            - tests/test_models.py: TestCNN.test_convnext_forward
        Ref: Liu et al. (2022) — ConvNeXt. Stage [3,3,9,3] = Tiny.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import conv_next_block, output_projection

    ap = config.arch_params or {}
    stage_filters = ap.get("stage_filters", [96, 192, 384, 768])
    blocks_per_stage = ap.get("blocks_per_stage", [3, 3, 9, 3])
    stem_k = ap.get("stem_k", 4)
    kernel_size = ap.get("kernel_size", 7)
    causal = config.use_causal_mode
    dr = config.dropout_rate

    logger.info(
        "build_convnext: n_feat=%d, out_ch=%d", config.n_features, config.output_channels
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── Patchify stem (ConvNeXt usa stride=4 no stem) ─────────────────
    # Para seq2seq, stride=1 para preservar comprimento temporal
    pad = "causal" if causal else "same"
    x = tf.keras.layers.Conv1D(stage_filters[0], stem_k, padding=pad, use_bias=False)(inp)
    x = tf.keras.layers.LayerNormalization()(x)

    # ── 4 Stages ─────────────────────────────────────────────────────
    for stage_i, (n_filt, n_blocks) in enumerate(zip(stage_filters, blocks_per_stage)):
        # ── Downsampling entre stages (preservar seq via downsampling de canais) ──
        if stage_i > 0 and x.shape[-1] != n_filt:
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.Conv1D(n_filt, 1, padding="same")(x)

        for _ in range(n_blocks):
            x = conv_next_block(
                x,
                n_filt,
                kernel_size=kernel_size,
                causal=causal,
                dropout_rate=dr,
                l1=config.l1_weight if config.use_l1_regularization else 0.0,
                l2=config.l2_weight if config.use_l2_regularization else 0.0,
            )

    x = tf.keras.layers.LayerNormalization()(x)

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="ConvNeXt")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: INCEPTIONNET
# ════════════════════════════════════════════════════════════════════════════
# InceptionNet 1D: modulos inception multi-escala com 3 branches paralelos.
# Kernels (9, 19, 39) capturam dependencias de curto, medio e longo prazo.
# Ref: Szegedy et al. (2014) Inception / GoogLeNet.
# ──────────────────────────────────────────────────────────────────────────


def build_inceptionnet(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi InceptionNet 1D com 4 modulos inception empilhados.

    Cada modulo combina 3 Conv1D com kernels diferentes + MaxPool,
    capturando dependencias temporais em multiplas escalas simultaneamente.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: InceptionNet 1D seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['InceptionNet']
            - tests/test_models.py: TestCNN.test_inceptionnet_forward
        Output ch = 4 * filters_per_branch por modulo.
        Ref: Szegedy et al. (2014) — GoogLeNet/Inception.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import inception_module, output_projection

    ap = config.arch_params or {}
    n_modules = ap.get("n_modules", 4)
    filters = ap.get("filters", 32)
    bottleneck = ap.get("bottleneck_size", 32)
    kernel_sizes = ap.get("kernel_sizes", (9, 19, 39))
    causal = config.use_causal_mode

    logger.info(
        "build_inceptionnet: n_feat=%d, out_ch=%d",
        config.n_features,
        config.output_channels,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    for i in range(n_modules):
        x = inception_module(
            x,
            filters=filters,
            causal=causal,
            bottleneck_size=bottleneck,
            kernel_sizes=tuple(kernel_sizes),
            l1=config.l1_weight if config.use_l1_regularization else 0.0,
            l2=config.l2_weight if config.use_l2_regularization else 0.0,
        )
        logger.debug("InceptionNet module %d: out_ch=%d", i + 1, x.shape[-1])

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="InceptionNet")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: INCEPTIONTIME
# ════════════════════════════════════════════════════════════════════════════
# InceptionTime 1D: blocos inception_time_block com skip residual.
# Padrao do TSAI/InceptionTime para classificacao de series temporais.
# Ref: Ismail Fawaz et al. (2020).
# ──────────────────────────────────────────────────────────────────────────


def build_inceptiontime(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi InceptionTime 1D: 3 inception blocks com residual.

    Inspirado no InceptionTime original (classificacao → adaptado para
    regressao seq2seq). Residuais a cada 3 modulos inception.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: InceptionTime seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['InceptionTime']
            - tests/test_models.py: TestCNN.test_inceptiontime_forward
        Ref: Ismail Fawaz et al. (2020) Data Mining and Knowledge Discovery.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import inception_time_block, output_projection

    ap = config.arch_params or {}
    n_blocks = ap.get("n_blocks", 3)
    filters = ap.get("filters", 32)
    kernel_sizes = ap.get("kernel_sizes", (9, 19, 39))
    bottleneck = ap.get("bottleneck_size", 32)
    causal = config.use_causal_mode

    logger.info(
        "build_inceptiontime: n_feat=%d, out_ch=%d",
        config.n_features,
        config.output_channels,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    for i in range(n_blocks):
        x = inception_time_block(
            x,
            filters=filters,
            causal=causal,
            use_residual=True,
            kernel_sizes=tuple(kernel_sizes),
            bottleneck_size=bottleneck,
            l1=config.l1_weight if config.use_l1_regularization else 0.0,
            l2=config.l2_weight if config.use_l2_regularization else 0.0,
        )
        logger.debug("InceptionTime block %d: out_ch=%d", i + 1, x.shape[-1])

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="InceptionTime")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CNN_1D (BASELINE SIMPLES)
# ════════════════════════════════════════════════════════════════════════════
# CNN_1D: 6 camadas Conv1D simetricas [32,64,128,128,64,32] (encoder-decoder).
# Baseline mais simples para comparacao rapida. Sem blocos residuais.
# Implementacao direta, sem dependencies de blocks.py.
# Legado: C29 (509 linhas).
# ──────────────────────────────────────────────────────────────────────────


def build_cnn1d(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi CNN_1D baseline: 6 Conv1D simetricas (encoder-decoder).

    6 camadas Conv1D com filtros [32, 64, 128, 128, 64, 32]:
    encoder expande (32→64→128) e decoder comprime (128→64→32).
    Sem residual, sem atencao — mais rapido, menor capacity.

    Arquitetura:
        Input → Conv(32) → Conv(64) → Conv(128) → Conv(128)
              → Conv(64) → Conv(32) → Output(out_ch)

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: CNN_1D seq2seq baseline.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['CNN_1D']
            - tests/test_models.py: TestCNN.test_cnn1d_forward
        Baseline para benchmarking: mais simples que ResNet-18.
        Legado C29: 6 camadas simetricas [32,64,128,128,64,32].
    """
    import tensorflow as tf

    ap = config.arch_params or {}
    filter_list = ap.get("filter_list", [32, 64, 128, 128, 64, 32])
    kernel_size = ap.get("kernel_size", 3)
    activation = ap.get("activation", "relu")
    use_bn = ap.get("use_bn", True)
    causal = config.use_causal_mode
    pad = "causal" if causal else "same"
    reg = _get_reg(config)
    dr = config.dropout_rate

    logger.info(
        "build_cnn1d: n_feat=%d, out_ch=%d, filters=%s",
        config.n_features,
        config.output_channels,
        filter_list,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    for i, n_filt in enumerate(filter_list):
        x = tf.keras.layers.Conv1D(
            n_filt,
            kernel_size,
            padding=pad,
            kernel_regularizer=reg,
            use_bias=not use_bn,
        )(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        if dr > 0.0:
            x = tf.keras.layers.Dropout(dr)(x)

    out = tf.keras.layers.Conv1D(
        config.output_channels,
        1,
        padding="same",
        activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="CNN_1D")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: RESNEXT — AGGREGATED RESIDUAL TRANSFORMATIONS
# ════════════════════════════════════════════════════════════════════════════
# ResNeXt (Xie et al. 2017) estende o ResNet com "grouped convolutions":
# em vez de um unico caminho (1 Conv1D largo), divide em C caminhos
# paralelos (cardinality) com Conv1D menores e agrega por concatenação.
#
# Vantagem sobre ResNet: melhor trade-off params/performance.
# Para inversão EM 1D, cada "caminho" captura padroes de escala diferente
# nas componentes do tensor H (alta/media/baixa frequencia espacial).
#
# Adaptacao 1D para geosteering:
#   - Cardinalidade C=32 (default) com largura d=4 por grupo
#   - 4 stages [64, 128, 256, 512] filtros (como ResNet-18)
#   - Compativel com modo causal (padding='causal')
#
# Ref: Xie et al. (2017) "Aggregated Residual Transformations for Deep
#      Neural Networks" CVPR — ResNeXt-50 32×4d.
# ──────────────────────────────────────────────────────────────────────────


def _resnext_block_1d(
    x, filters, cardinality, group_width, kernel_size, causal, dropout_rate, reg
):
    """Bloco ResNeXt 1D com grouped convolutions.

    Estrutura bottleneck com C caminhos paralelos (grouped convolution):
      x → Conv1D(C*d, k=1) → BN → ReLU
        → Conv1D(C*d, k=3, groups=C) → BN → ReLU
        → Conv1D(filters, k=1) → BN → +skip → ReLU

    Args:
        x: Tensor de entrada (B, N, C_in).
        filters: Canais de saida do bloco.
        cardinality: Numero de grupos paralelos C (default 32).
        group_width: Largura de cada grupo d (default 4).
        kernel_size: Tamanho do kernel (default 3).
        causal: Se True, padding causal.
        dropout_rate: Taxa de dropout.
        reg: Regularizador kernel.

    Returns:
        Tensor (B, N, filters) com skip connection residual.

    Note:
        Funcao privada — usada apenas por build_resnext().
        Grouped convolution: 32 grupos × 4 canais = 128 canais intermediarios.
        Ref: Xie et al. (2017) Fig. 3c — grouped convolution pathway.
    """
    import tensorflow as tf

    pad = "causal" if causal else "same"
    intermediate = cardinality * group_width

    skip = x

    # ── Bottleneck: 1×1 → grouped 3×k → 1×1 ─────────────────────────
    # 1×1 reduce para C*d canais
    x = tf.keras.layers.Conv1D(
        intermediate,
        1,
        padding="same",
        kernel_regularizer=reg,
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Grouped convolution: C grupos, cada um com d canais
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

    # 1×1 projeta para filters
    x = tf.keras.layers.Conv1D(
        filters,
        1,
        padding="same",
        kernel_regularizer=reg,
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # ── Skip connection ───────────────────────────────────────────────
    in_channels = getattr(skip.shape, "__getitem__", lambda _: None)(-1)
    if in_channels is None or in_channels != filters:
        skip = tf.keras.layers.Conv1D(filters, 1, padding="same")(skip)

    x = tf.keras.layers.Add()([skip, x])
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_resnext(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi ResNeXt 1D para inversao seq2seq de resistividade.

    ResNeXt adapta o ResNet com grouped convolutions (cardinalidade C=32,
    largura d=4), obtendo melhor tradeoff parametros/performance. Cada
    grupo captura padroes de escala espacial diferente nas componentes EM.

    Arquitetura:
      ┌──────────────────────────────────────────────────────────────┐
      │  Input (B, 600, n_features)                                 │
      │    ↓                                                        │
      │  Stem: Conv1D(64, k=7) → BN → ReLU                         │
      │    ↓                                                        │
      │  Stage 1: 2× ResNeXtBlock(64,  C=32, d=4)                 │
      │  Stage 2: 2× ResNeXtBlock(128, C=32, d=4)                 │
      │  Stage 3: 2× ResNeXtBlock(256, C=32, d=4)                 │
      │  Stage 4: 2× ResNeXtBlock(512, C=32, d=4)                 │
      │    ↓                                                        │
      │  Output: Conv1D(output_ch, k=1)                             │
      │  Output (B, 600, output_channels)                           │
      └──────────────────────────────────────────────────────────────┘

    Dual-mode:
      ┌────────────────────────────────────────────────────────────┐
      │  "same"   →  Offline (acausal, batch completo)            │
      │  "causal" →  Realtime (causal, geosteering)               │
      └────────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com:
            - n_features, sequence_length, output_channels
            - use_causal_mode, use_se_block, dropout_rate
            - arch_params: override granular:
                - stage_filters (list, default [64, 128, 256, 512])
                - blocks_per_stage (list, default [2, 2, 2, 2])
                - cardinality (int, default 32): grupos paralelos
                - group_width (int, default 4): canais por grupo

    Returns:
        tf.keras.Model: ResNeXt seq2seq.

    Example:
        >>> config = PipelineConfig(model_type="ResNeXt")
        >>> model = build_resnext(config)
        >>> assert model.output_shape == (None, 600, 2)

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['ResNeXt']
            - tests/test_models.py: TestResNeXt
        Causal mode: padding='causal' em todas as Conv1D.
        Cardinalidade 32×4d: 32 caminhos paralelos com 4 canais cada,
        equivalente em FLOPs a ResNet mas com performance superior.
        Ref: Xie et al. (2017) CVPR — "Aggregated Residual Transformations".
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import output_projection, se_block

    ap = config.arch_params or {}
    stage_filters = ap.get("stage_filters", [64, 128, 256, 512])
    blocks_per_stage = ap.get("blocks_per_stage", [2, 2, 2, 2])
    cardinality = ap.get("cardinality", 32)
    group_width = ap.get("group_width", 4)
    kernel_size = ap.get("kernel_size", 3)
    causal = config.use_causal_mode
    reg = _get_reg(config)
    dr = config.dropout_rate
    use_se = config.use_se_block

    logger.info(
        "build_resnext: n_feat=%d, C=%d, d=%d, causal=%s",
        config.n_features,
        cardinality,
        group_width,
        causal,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── Stem ──────────────────────────────────────────────────────────
    x = _stem_block(inp, stage_filters[0], 7, causal, reg)

    # ── 4 Stages com blocos ResNeXt ──────────────────────────────────
    for stage_i, (n_filt, n_blocks) in enumerate(zip(stage_filters, blocks_per_stage)):
        for _ in range(n_blocks):
            x = _resnext_block_1d(
                x,
                n_filt,
                cardinality,
                group_width,
                kernel_size,
                causal,
                dr,
                reg,
            )
        if use_se:
            x = se_block(x, reduction=config.se_reduction)
        logger.debug("ResNeXt Stage %d: filters=%d", stage_i + 1, n_filt)

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="ResNeXt")


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    "build_resnet18",
    "build_resnet34",
    "build_resnet50",
    "build_convnext",
    "build_inceptionnet",
    "build_inceptiontime",
    "build_cnn1d",
    "build_resnext",
]
