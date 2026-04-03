# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/transformer.py                                             ║
# ║  Bloco: 3g — Arquiteturas Transformer (6 variantes)                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • 6 arquiteturas Transformer para inversao resistividade               ║
# ║    • Transformer (vanilla), Simple_TFT, TFT (full), PatchTST             ║
# ║    • Autoformer (autocorr), iTransformer (inverted attention)             ║
# ║    • Adaptaveis para causal/acausal via config.use_causal_mode            ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: 6 funcoes — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.7, legado C34                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (6 variantes)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""6 arquiteturas Transformer 1D para inversao resistividade.

Todas implementam seq2seq: Input (batch, seq_len, n_feat) → Output (batch, seq_len, out_ch).
seq_len = config.sequence_length (default 600, derivado do .out).
Positional encoding: learned (simples e eficaz para series fisicas).

Note:
    Referenciado em:
        - models/registry.py: _REGISTRY[Transformer..iTransformer]
        - tests/test_models.py: TestTransformer
    Legado C34 (2374 linhas, 6 arquiteturas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

import tensorflow as tf

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: POSITIONAL ENCODING
# ════════════════════════════════════════════════════════════════════════════
# Aprendido (learned) em vez de sinoidal — mais flexivel para dados LWD
# onde a distancia entre amostras pode variar (diferentes velocidades de
# perfuracao). Implementado como Keras Layer para compatibilidade com
# Keras 3.x (Colab), onde KerasTensor nao pode ser passado para tf.*.
# ──────────────────────────────────────────────────────────────────────────


class _LearnedPositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding aprendido via Embedding lookup.

    Encapsulado como Layer para compatibilidade com Keras 3.x (Colab),
    onde KerasTensor nao pode ser passado para funcoes tf.* diretamente.
    Dentro de call(), os tensores sao concretos e tf.shape/tf.range funcionam.

    Args:
        max_len: Comprimento maximo da sequencia (ex: 600).
        d_model: Dimensao do modelo (embedding size).

    Note:
        Ref: Vaswani et al. (2017) — positional encoding, aqui learned
        em vez de sinoidal (mais flexivel para espacamento LWD variavel).
    """

    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(max_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = self.embedding(tf.range(seq_len))
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


# ════════════════════════════════════════════════════════════════════════════
# SECAO: TRANSFORMER (VANILLA)
# ════════════════════════════════════════════════════════════════════════════
# N blocos TransformerEncoder empilhados com positional encoding aprendido.
# Pre-LN (mais estavel que post-LN para modelos profundos).
# Ref: Vaswani et al. (2017) NeurIPS.
# ──────────────────────────────────────────────────────────────────────────


def build_transformer(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi Transformer vanilla para inversao seq2seq.

    N blocos Transformer encoder (MHA + FFN + pre-LN) com
    positional encoding aprendido e projecao final seq2seq.

    Arquitetura:
        Input → Dense(d_model) → PosEnc → N × TransformerBlock → Out

    Args:
        config: PipelineConfig com:
            - n_features, sequence_length, output_channels
            - use_causal_mode: se True, mascara causal na atencao
            - arch_params: n_layers, d_model, num_heads, ff_dim

    Returns:
        tf.keras.Model: Transformer seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['Transformer']
            - tests/test_models.py: TestTransformer.test_transformer_forward
        Ref: Vaswani et al. (2017) — Attention Is All You Need.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import output_projection, transformer_encoder_block

    ap = config.arch_params or {}
    n_layers = ap.get("n_layers", 4)
    d_model = ap.get("d_model", 128)
    num_heads = ap.get("num_heads", 8)
    ff_dim = ap.get("ff_dim", 256)
    dr = ap.get("dropout_rate", config.dropout_rate)
    causal = config.use_causal_mode

    logger.info(
        "build_transformer: n_feat=%d, d_model=%d, n_layers=%d, heads=%d",
        config.n_features,
        d_model,
        n_layers,
        num_heads,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = tf.keras.layers.Dense(d_model)(inp)  # projecao de entrada

    # ── Positional encoding aprendido ─────────────────────────────────
    x = _LearnedPositionalEncoding(config.sequence_length, d_model)(x)
    if dr > 0.0:
        x = tf.keras.layers.Dropout(dr)(x)

    # ── N blocos Transformer ──────────────────────────────────────────
    for i in range(n_layers):
        x = transformer_encoder_block(
            x,
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            ff_dim=ff_dim,
            dropout_rate=dr,
            use_causal_mask=causal,
            l1=config.l1_weight if config.use_l1_regularization else 0.0,
            l2=config.l2_weight if config.use_l2_regularization else 0.0,
        )
        logger.debug("Transformer block %d", i + 1)

    x = tf.keras.layers.LayerNormalization()(x)
    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="Transformer")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: SIMPLE_TFT — TFT SIMPLIFICADO
# ════════════════════════════════════════════════════════════════════════════
# TFT simplificado: GRN para projecao de entrada + Transformer encoder.
# Sem Variable Selection Network completo (mais leve que TFT full).
# Ref: Lim et al. (2021) — TFT.
# ──────────────────────────────────────────────────────────────────────────


def build_simple_tft(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi Simple_TFT: GRN de entrada + Transformer encoder.

    Versao simplificada do TFT: GRN para selecao de features + N blocos
    TransformerEncoder. Sem VSN completo — mais eficiente para P1 (5 feat).

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: Simple_TFT seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['Simple_TFT']
            - tests/test_models.py: TestTransformer.test_simple_tft_forward
        Ref: Lim et al. (2021) Int. J. Forecasting — TFT.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import (
        grn_block,
        output_projection,
        transformer_encoder_block,
    )

    ap = config.arch_params or {}
    n_layers = ap.get("n_layers", 3)
    d_model = ap.get("d_model", 64)
    num_heads = ap.get("num_heads", 4)
    ff_dim = ap.get("ff_dim", 128)
    dr = ap.get("dropout_rate", config.dropout_rate)
    causal = config.use_causal_mode

    logger.info("build_simple_tft: d_model=%d, n_layers=%d", d_model, n_layers)

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── GRN de entrada: selecao de features ──────────────────────────
    x = grn_block(inp, d_model=d_model, dropout_rate=dr)

    # ── Positional encoding ───────────────────────────────────────────
    x = _LearnedPositionalEncoding(config.sequence_length, d_model)(x)

    # ── N blocos Transformer ──────────────────────────────────────────
    for _ in range(n_layers):
        x = transformer_encoder_block(
            x,
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            ff_dim=ff_dim,
            dropout_rate=dr,
            use_causal_mask=causal,
        )

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="Simple_TFT")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: TFT — TEMPORAL FUSION TRANSFORMER COMPLETO
# ════════════════════════════════════════════════════════════════════════════
# TFT completo: VSN + GRN + encoder Transformer + gated output.
# Mais pesado mas melhor para dados com features heterogeneas (P3/P4).
# Ref: Lim et al. (2021).
# ──────────────────────────────────────────────────────────────────────────


def build_tft(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi TFT (Temporal Fusion Transformer) completo.

    VSN seleciona features relevantes; GRN processa; Transformer modela
    dependencias temporais; gating controla fluxo de informacao.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: TFT completo seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['TFT']
            - tests/test_models.py: TestTransformer.test_tft_forward
        Ref: Lim et al. (2021) — TFT. Melhor para P3/P4 (features hetero).
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import (
        grn_block,
        output_projection,
        transformer_encoder_block,
        vsn_block,
    )

    ap = config.arch_params or {}
    n_layers = ap.get("n_layers", 3)
    d_model = ap.get("d_model", 64)
    num_heads = ap.get("num_heads", 4)
    ff_dim = ap.get("ff_dim", 128)
    dr = ap.get("dropout_rate", config.dropout_rate)
    causal = config.use_causal_mode
    n_var = config.n_features

    logger.info("build_tft: d_model=%d, n_layers=%d, n_var=%d", d_model, n_layers, n_var)

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── VSN: variable selection ───────────────────────────────────────
    x = vsn_block(inp, d_model=d_model, n_variables=n_var, dropout_rate=dr)

    # ── GRN de entrada ────────────────────────────────────────────────
    x = grn_block(x, d_model=d_model, dropout_rate=dr)

    # ── Positional encoding ───────────────────────────────────────────
    x = _LearnedPositionalEncoding(config.sequence_length, d_model)(x)

    # ── N blocos Transformer ──────────────────────────────────────────
    for _ in range(n_layers):
        x = transformer_encoder_block(
            x,
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            ff_dim=ff_dim,
            dropout_rate=dr,
            use_causal_mask=causal,
        )

    # ── Gated output ──────────────────────────────────────────────────
    gate = tf.keras.layers.Dense(d_model, activation="sigmoid")(x)
    x = tf.keras.layers.Multiply()([x, gate])

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="TFT")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: PATCHTST
# ════════════════════════════════════════════════════════════════════════════
# PatchTST: divide a serie em patches de comprimento L, processa cada
# patch como um token via Transformer. Melhor eficiencia para seq longas.
# Ref: Nie et al. (2023) ICLR.
# ──────────────────────────────────────────────────────────────────────────


def build_patchtst(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi PatchTST: Transformer sobre patches temporais.

    Divide a serie em patches (tokens) → Transformer → reconstroi.
    Mais eficiente para seq_len longo (e.g. 600) que Transformer sobre pontos individuais.

    Nota: PatchTST usa seq_len diferente internamente (n_patches).
    Projecao final mapeia de volta para sequence_length via Dense.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: PatchTST seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['PatchTST']
            - tests/test_models.py: TestTransformer.test_patchtst_forward
        Ref: Nie et al. (2023) ICLR — PatchTST.
        Output: (batch, sequence_length, output_channels) via reshape.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import (
        output_projection,
        patch_embedding_block,
        transformer_encoder_block,
    )

    ap = config.arch_params or {}
    patch_len = ap.get("patch_len", 16)
    stride = ap.get("stride", 8)
    n_layers = ap.get("n_layers", 3)
    d_model = ap.get("d_model", 128)
    num_heads = ap.get("num_heads", 8)
    ff_dim = ap.get("ff_dim", 256)
    dr = ap.get("dropout_rate", config.dropout_rate)
    causal = config.use_causal_mode

    logger.info(
        "build_patchtst: n_feat=%d, patch_len=%d, stride=%d, d_model=%d",
        config.n_features,
        patch_len,
        stride,
        d_model,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── Patch embedding: (batch, seq, feat) → (batch, n_patch, d_model)
    x = patch_embedding_block(inp, patch_len=patch_len, stride=stride, d_model=d_model)

    # ── Transformer sobre patches ─────────────────────────────────────
    n_patches = x.shape[1]  # pode ser None se dynamico
    for _ in range(n_layers):
        x = transformer_encoder_block(
            x,
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            ff_dim=ff_dim,
            dropout_rate=dr,
            use_causal_mask=causal,
        )

    # ── Projecao de volta para (batch, seq_len, out_ch) ───────────────
    # Flatten patches e projetar para seq_len * out_ch → reshape
    x = tf.keras.layers.Flatten()(x)  # (batch, n_patches * d_model)
    x = tf.keras.layers.Dense(
        config.sequence_length * config.output_channels,
    )(x)
    out = tf.keras.layers.Reshape((config.sequence_length, config.output_channels))(x)

    if config.use_physical_constraint_layer:
        out = tf.keras.layers.Activation(config.constraint_activation)(out)

    return tf.keras.Model(inputs=inp, outputs=out, name="PatchTST")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: AUTOFORMER
# ════════════════════════════════════════════════════════════════════════════
# Autoformer: decomposicao tendencia-sazonalidade + AutoCorrelation.
# Decomp substituem AddNorm no Transformer padrao.
# Ref: Wu et al. (2021) NeurIPS.
# ──────────────────────────────────────────────────────────────────────────


def build_autoformer(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi Autoformer: decomp + autocorrelacao para inversao.

    Cada bloco realiza: decomp(x) + autocorr(seasonal) + trend acumulada.
    Adequado para dados LWD com tendencias suaves + variacao rapida.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: Autoformer seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['Autoformer']
            - tests/test_models.py: TestTransformer.test_autoformer_forward
        Ref: Wu et al. (2021) NeurIPS — Autoformer.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import (
        autocorr_block,
        feedforward_block,
        output_projection,
        series_decomp_block,
    )

    ap = config.arch_params or {}
    n_layers = ap.get("n_layers", 2)
    d_model = ap.get("d_model", 64)
    num_heads = ap.get("num_heads", 4)
    ff_dim = ap.get("ff_dim", 128)
    decomp_k = ap.get("decomp_kernel", 25)
    dr = ap.get("dropout_rate", config.dropout_rate)

    logger.info("build_autoformer: d_model=%d, n_layers=%d", d_model, n_layers)

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = tf.keras.layers.Dense(d_model)(inp)

    trend_cum = tf.keras.layers.Lambda(lambda z: tf.zeros_like(z))(x)

    for _ in range(n_layers):
        # ── AutoCorrelation ───────────────────────────────────────────
        seasonal, trend = series_decomp_block(x, decomp_k)
        seasonal = autocorr_block(seasonal, num_heads=num_heads)
        trend_cum = tf.keras.layers.Add()([trend_cum, trend])

        # ── FFN ───────────────────────────────────────────────────────
        y = feedforward_block(seasonal, ff_dim=ff_dim, dropout_rate=dr)
        seasonal, trend = series_decomp_block(y, decomp_k)
        trend_cum = tf.keras.layers.Add()([trend_cum, trend])
        x = tf.keras.layers.Add()([seasonal, trend_cum])

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="Autoformer")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: ITRANSFORMER
# ════════════════════════════════════════════════════════════════════════════
# iTransformer: inverte atencao temporal → atencao sobre variaveis.
# Cada variavel (feature) atende a todas as outras variaveis.
# Ref: Liu et al. (2023) arXiv:2310.06625.
# ──────────────────────────────────────────────────────────────────────────


def build_itransformer(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi iTransformer: atencao invertida sobre variaveis.

    Em vez de atentar sobre time steps, atentar sobre features.
    Melhor para P4 (geosinais) onde as relacoes entre features sao
    mais informativas que dependencias temporais locais.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: iTransformer seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['iTransformer']
            - tests/test_models.py: TestTransformer.test_itransformer_forward
        Ref: Liu et al. (2023) — iTransformer. ICLR 2024.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import ita_block, output_projection

    ap = config.arch_params or {}
    n_layers = ap.get("n_layers", 3)
    num_heads = ap.get("num_heads", 4)
    key_dim = ap.get("key_dim", 32)
    ff_dim = ap.get("ff_dim", 128)
    dr = ap.get("dropout_rate", config.dropout_rate)

    logger.info("build_itransformer: n_layers=%d, n_feat=%d", n_layers, config.n_features)

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    for _ in range(n_layers):
        x = ita_block(
            x,
            num_heads=num_heads,
            key_dim=key_dim,
            ff_dim=ff_dim,
            dropout_rate=dr,
        )

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="iTransformer")


__all__ = [
    "build_transformer",
    "build_simple_tft",
    "build_tft",
    "build_patchtst",
    "build_autoformer",
    "build_itransformer",
]
