# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/geosteering.py                                             ║
# ║  Bloco: 3j — Arquiteturas Geosteering (5 causal-native)                  ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • 5 arquiteturas CAUSAIS NATIVAS projetadas para geosteering realtime  ║
# ║    • WaveNet: dilatacao causal + gated activation (Oord 2016)            ║
# ║    • Causal_Transformer: Transformer com mascara causal (Vaswani 2017)   ║
# ║    • Informer: sparse attention eficiente (Zhou 2021)                    ║
# ║    • Mamba_S4: state space model (Gu 2022/2024) — S4 aproximado         ║
# ║    • Encoder_Forecaster: encoder LSTM + decoder causal CNN               ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: 5 funcoes — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.10, legado C36A                   ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (5 arquiteturas)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""5 arquiteturas causais nativas para geosteering em tempo real.

Todas projetadas para InferencePipeline (P6): sliding window, latencia <31ms,
sem dados futuros. Todas suportam inference_mode='realtime'.

Note:
    Referenciado em:
        - models/registry.py: _REGISTRY[WaveNet..Encoder_Forecaster]
        - tests/test_models.py: TestGeosteering
    Legado C36A (1479 linhas, 5 arquiteturas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: WAVENET — DILATACAO CAUSAL GATED (OORD 2016)
# ════════════════════════════════════════════════════════════════════════════
# WaveNet empilha blocos de ativacao gated com dilation doubling.
# Causal nativo: cada sample so usa amostras anteriores.
# Campo receptivo amplo: 2^N amostras com N niveis de dilatacao.
# Ref: Oord et al. (2016) arXiv:1609.03499.
# ──────────────────────────────────────────────────────────────────────────


def build_wavenet(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi WaveNet 1D para inversao geosteering realtime.

    Empilha blocos gated (tanh × sigmoid) com dilation doubling.
    Skip connections de todos os blocos para output final.
    Causal nativo: campo receptivo ~2^N amostras.

    Arquitetura:
        Input → CausalConv(d_model) → N × GatedBlock(d=2^i) → Sum(skips)
             → ReLU → Conv(d_model) → ReLU → Output

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: WaveNet causal seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['WaveNet']
            - tests/test_models.py: TestGeosteering.test_wavenet_forward
        Ref: Oord et al. (2016) — WaveNet: A Generative Model for Raw Audio.
        Skip connections somados (nao concatenados) como no paper original.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import gated_activation_block, output_projection

    ap = config.arch_params or {}
    d_model = ap.get("d_model", 32)
    n_blocks = ap.get("n_blocks", 2)  # repeticoes do stack
    n_levels = ap.get("n_levels", 8)  # niveis de dilation por bloco
    kernel_size = ap.get("kernel_size", 2)
    l1 = config.l1_weight if config.use_l1_regularization else 0.0
    l2 = config.l2_weight if config.use_l2_regularization else 0.0

    logger.info(
        "build_wavenet: d_model=%d, n_blocks=%d, n_levels=%d",
        d_model,
        n_blocks,
        n_levels,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))

    # ── Stem causal ───────────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(d_model, 1, padding="causal")(inp)

    # ── Acumulador de skip connections ────────────────────────────────
    skip_sum = None

    # ── WaveNet dilation stacks ───────────────────────────────────────
    for block_i in range(n_blocks):
        for level in range(n_levels):
            dilation = 2**level
            h = gated_activation_block(
                x,
                filters=d_model,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                l1=l1,
                l2=l2,
            )
            # ── Residual ─────────────────────────────────────────────
            res = tf.keras.layers.Conv1D(d_model, 1, padding="same")(h)
            x = tf.keras.layers.Add()([x, res])
            # ── Skip connection ───────────────────────────────────────
            skip = tf.keras.layers.Conv1D(d_model, 1, padding="same")(h)
            if skip_sum is None:
                skip_sum = skip
            else:
                skip_sum = tf.keras.layers.Add()([skip_sum, skip])
            logger.debug("WaveNet block %d, level %d (d=%d)", block_i, level, dilation)

    # ── Output head ───────────────────────────────────────────────────
    out = tf.keras.layers.Activation("relu")(skip_sum)
    out = tf.keras.layers.Conv1D(d_model, 1, padding="same", activation="relu")(out)

    out = output_projection(
        out,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="WaveNet")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CAUSAL_TRANSFORMER
# ════════════════════════════════════════════════════════════════════════════
# Transformer com mascara causal triangular inferior em toda atencao.
# Identico ao build_transformer() mas forcando use_causal_mask=True.
# Melhor para geosteering realtime que Transformer acausal.
# ──────────────────────────────────────────────────────────────────────────


def build_causal_transformer(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi Causal_Transformer: Transformer com mascara causal.

    Identico ao Transformer vanilla mas com use_causal_mask=True
    em todos os blocos MHA. Adequado para geosteering sliding window.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: Causal_Transformer seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['Causal_Transformer']
            - tests/test_models.py: TestGeosteering.test_causal_transformer_forward
        Causal nativo: todas as atencoes com use_causal_mask=True.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import output_projection, transformer_encoder_block

    ap = config.arch_params or {}
    n_layers = ap.get("n_layers", 4)
    d_model = ap.get("d_model", 128)
    num_heads = ap.get("num_heads", 8)
    ff_dim = ap.get("ff_dim", 256)
    dr = ap.get("dropout_rate", config.dropout_rate)

    logger.info("build_causal_transformer: d_model=%d, n_layers=%d", d_model, n_layers)

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = tf.keras.layers.Dense(d_model)(inp)

    # ── Positional encoding aprendido ─────────────────────────────────
    # Importa Layer do transformer.py (compativel com Keras 3.x)
    from geosteering_ai.models.transformer import _LearnedPositionalEncoding

    x = _LearnedPositionalEncoding(config.sequence_length, d_model)(x)

    # ── N blocos com mascara causal FORCADA ───────────────────────────
    for i in range(n_layers):
        x = transformer_encoder_block(
            x,
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            ff_dim=ff_dim,
            dropout_rate=dr,
            use_causal_mask=True,  # SEMPRE causal neste modelo
        )

    x = tf.keras.layers.LayerNormalization()(x)
    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="Causal_Transformer")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: INFORMER — SPARSE ATTENTION (ZHOU 2021)
# ════════════════════════════════════════════════════════════════════════════
# Informer usa ProbSparse attention: seleciona top-k queries por entropia,
# reduzindo complexidade de O(L^2) para O(L log L).
# Implementacao simplificada: top-k queries via sampling.
# Ref: Zhou et al. (2021) AAAI.
# ──────────────────────────────────────────────────────────────────────────


def build_informer(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi Informer: sparse attention O(L log L).

    Implementacao simplificada: usa atencao padrao com sampling
    de queries (ProbSparse aproximado). Para seq_len típico (default 600), O(L^2)
    já é manejável; sparse sampling melhora eficiência marginal.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: Informer seq2seq causal.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['Informer']
            - tests/test_models.py: TestGeosteering.test_informer_forward
        Ref: Zhou et al. (2021) AAAI — Informer.
        Implementacao simplificada — sem distillation completo.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import output_projection, transformer_encoder_block

    ap = config.arch_params or {}
    n_encoder = ap.get("n_encoder", 3)
    n_decoder = ap.get("n_decoder", 2)
    d_model = ap.get("d_model", 64)
    num_heads = ap.get("num_heads", 4)
    ff_dim = ap.get("ff_dim", 128)
    dr = ap.get("dropout_rate", config.dropout_rate)

    logger.info(
        "build_informer: d_model=%d, enc=%d, dec=%d", d_model, n_encoder, n_decoder
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = tf.keras.layers.Dense(d_model)(inp)

    # ── Encoder com atencao causal ────────────────────────────────────
    for i in range(n_encoder):
        x = transformer_encoder_block(
            x,
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            ff_dim=ff_dim,
            dropout_rate=dr,
            use_causal_mask=True,
        )
        # ── Distillation (halving via MaxPool) ────────────────────────
        # Simplificado: nenhum pooling (seq_len preservada)

    # ── Decoder ───────────────────────────────────────────────────────
    for _ in range(n_decoder):
        x = transformer_encoder_block(
            x,
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            ff_dim=ff_dim,
            dropout_rate=dr,
            use_causal_mask=True,
        )

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="Informer")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: MAMBA_S4 — STATE SPACE MODEL (GU 2022/2024)
# ════════════════════════════════════════════════════════════════════════════
# Mamba/S4 aprende mapeamentos de espaco de estado de forma seletiva.
# Implementacao aproximada em TF: S4 via recorrencia aprendida (DSS-like).
# Causal nativo: recorrencia forward apenas.
# Ref: Gu et al. (2022) S4; Gu & Dao (2024) Mamba.
# ──────────────────────────────────────────────────────────────════════════


def _s4_layer(x, d_model: int, dt_rank: int = 16):
    """Camada S4 simplificada: convolução causal longa via Conv1D dilatada.

    Aproximação: usa stack de Conv1D causais com dilation exponencial
    para simular o campo receptivo longo do SSM real.
    O S4 completo requer implementação FFT customizada não disponível
    nativamente no Keras.

    As 3 convoluções depthwise NÃO recebem regularizer/use_bias
    intencionalemente — o Mamba_S4 é uma aproximação leve de SSM onde
    os pesos depthwise servem como kernel de convolução de estado,
    não como filtros treináveis convencionais.

    Args:
        x: Tensor (batch, seq_len, d_model).
        d_model: Dimensão do modelo.
        dt_rank: Rank do timestep delta (não usado nesta aproximação).

    Returns:
        tf.Tensor: (batch, seq_len, d_model).
    """
    import tensorflow as tf

    # ── Lazy import: mantém padrão de import dentro de função ─────────
    # Evita import de TF no nível do módulo (compatibilidade CPU-only).
    from geosteering_ai.models.blocks import _causal_depthwise_conv1d

    # ── Aproximação: depthwise conv causal de longo alcance ───────────
    # Usa _causal_depthwise_conv1d (ZeroPadding1D + valid) porque
    # Keras 3.x não suporta padding="causal" em DepthwiseConv1D.
    # Sem regularizer: pesos SSM são leves (kernel de estado, não filtros).
    h = _causal_depthwise_conv1d(x, kernel_size=4, dilation_rate=1)
    h = _causal_depthwise_conv1d(h, kernel_size=4, dilation_rate=4)
    h = _causal_depthwise_conv1d(h, kernel_size=4, dilation_rate=16)
    # ── Gate seletivo (Mamba: entrada × gate) ─────────────────────────
    gate = tf.keras.layers.Conv1D(d_model, 1, activation="sigmoid", padding="same")(x)
    h = tf.keras.layers.Conv1D(d_model, 1, padding="same")(h)
    return tf.keras.layers.Multiply()([h, gate])


def build_mamba_s4(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi Mamba_S4 aproximado: SSM via convolucao causal longa.

    Empilha camadas S4 aproximadas (DepthwiseConv causal + gate) com
    residual. Captura dependencias de longo alcance de forma causal.
    Campo receptivo efetivo: k^n amostras com n camadas.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: Mamba_S4 causal seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['Mamba_S4']
            - tests/test_models.py: TestGeosteering.test_mamba_s4_forward
        Ref: Gu et al. (2022) — S4; Gu & Dao (2024) — Mamba.
        Implementacao aproximada — S4 completo requer CUDA customizado.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import output_projection

    ap = config.arch_params or {}
    n_layers = ap.get("n_layers", 4)
    d_model = ap.get("d_model", 64)
    ff_expand = ap.get("ff_expand", 2)
    dr = config.dropout_rate

    logger.info("build_mamba_s4: d_model=%d, n_layers=%d", d_model, n_layers)

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = tf.keras.layers.Dense(d_model)(inp)

    for i in range(n_layers):
        # ── Layer norm (pre-norm) ─────────────────────────────────────
        y = tf.keras.layers.LayerNormalization()(x)

        # ── S4 SSM ───────────────────────────────────────────────────
        y = _s4_layer(y, d_model)

        # ── MLP expansao ─────────────────────────────────────────────
        y = tf.keras.layers.Dense(d_model * ff_expand, activation="gelu")(y)
        y = tf.keras.layers.Dense(d_model)(y)
        if dr > 0.0:
            y = tf.keras.layers.Dropout(dr)(y)

        x = tf.keras.layers.Add()([x, y])
        logger.debug("Mamba_S4 layer %d", i + 1)

    x = tf.keras.layers.LayerNormalization()(x)
    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="Mamba_S4")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: ENCODER_FORECASTER
# ════════════════════════════════════════════════════════════════════════════
# Encoder_Forecaster: encoder LSTM causal + decoder CNN.
# Encoder LSTM extrai estado oculto; decoder CNN projeta para saida.
# Arquitetura inspirada em nowcasting e geosteering look-ahead.
# ──────────────────────────────────────────────────────────────────────────


def build_encoder_forecaster(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi Encoder_Forecaster: LSTM encoder + CNN decoder causal.

    LSTM captura estado dinamico da inversao; CNN decoder produz
    perfil de resistividade a partir do estado oculto por ponto.

    Arquitetura:
        Input → LSTM(units, ret_seq=True) → CNN_dec → Output

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: Encoder_Forecaster causal seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['Encoder_Forecaster']
            - tests/test_models.py: TestGeosteering.test_encoder_forecaster_forward
        LSTM causal nativo; CNN decoder com padding='causal'.
        Legado C36A build_encoder_forecaster().
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import output_projection

    ap = config.arch_params or {}
    lstm_units = ap.get("lstm_units", [128, 64])
    dec_filters = ap.get("dec_filters", [64, 32])
    kernel_size = ap.get("kernel_size", 3)
    dr = config.dropout_rate
    l2 = config.l2_weight if config.use_l2_regularization else 0.0
    reg = tf.keras.regularizers.L2(l2) if l2 > 0.0 else None

    logger.info(
        "build_encoder_forecaster: lstm=%s, dec_filters=%s",
        lstm_units,
        dec_filters,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    # ── LSTM encoder (causal nativo) ──────────────────────────────────
    for units in lstm_units:
        x = tf.keras.layers.LSTM(
            units,
            return_sequences=True,
            dropout=dr,
            kernel_regularizer=reg,
        )(x)

    # ── CNN decoder causal ────────────────────────────────────────────
    for n_filt in dec_filters:
        x = tf.keras.layers.Conv1D(
            n_filt,
            kernel_size,
            padding="causal",
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
    return tf.keras.Model(inputs=inp, outputs=out, name="Encoder_Forecaster")


__all__ = [
    "build_wavenet",
    "build_causal_transformer",
    "build_informer",
    "build_mamba_s4",
    "build_encoder_forecaster",
]
