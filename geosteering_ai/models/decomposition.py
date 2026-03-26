# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/decomposition.py                                           ║
# ║  Bloco: 3h — Decomposicao (N-BEATS, N-HiTS)                             ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • N-BEATS: stacks de MLP com basis expansion (trend + seasonality)    ║
# ║    • N-HiTS: hierarquico multi-escala com pooling expressivo             ║
# ║    • Adaptados de previsao de series para inversao seq2seq                ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: 2 funcoes — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.8, legado C35                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""N-BEATS e N-HiTS adaptados para inversao 1D seq2seq.

Ambos usam residual stacking: cada bloco aprende parte do sinal,
subtrai sua contribuicao (residual learning), e passa o residual adiante.

Note:
    Referenciado em:
        - models/registry.py: _REGISTRY['N_BEATS'], _REGISTRY['N_HiTS']
        - tests/test_models.py: TestDecomposition
    Legado C35 (1083 linhas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: N-BEATS — NEURAL BASIS EXPANSION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
# N-BEATS empilha blocos MLP com basis expansion.
# Cada bloco: MLP → (backcast, forecast); residual = input - backcast.
# Adaptacao para inversao: output e a soma das forecasts de todos os blocos.
# Ref: Oreshkin et al. (2020) ICLR.
# ──────────────────────────────────────────────────────────────────────────


def _nbeats_block(x, n_hidden, n_layers_per_block, output_dim, expansion_coef_dim):
    """Bloco interno N-BEATS: MLP + basis expansion.

    Args:
        x: Tensor (batch, seq_len * n_feat) — flattened.
        n_hidden: Unidades por camada MLP.
        n_layers_per_block: Numero de camadas MLP.
        output_dim: Dimensao de saida total (seq_len * out_ch).
        expansion_coef_dim: Dimensao dos coeficientes de basis.

    Returns:
        tuple: (backcast, forecast) — ambos com shape (batch, output_dim).
    """
    import tensorflow as tf

    h = x
    for _ in range(n_layers_per_block):
        h = tf.keras.layers.Dense(n_hidden, activation="relu")(h)

    # ── Basis expansion ───────────────────────────────────────────────
    theta_b = tf.keras.layers.Dense(expansion_coef_dim, use_bias=False)(h)
    theta_f = tf.keras.layers.Dense(expansion_coef_dim, use_bias=False)(h)

    # ── Projecao para backcast e forecast ────────────────────────────
    # Camada Dense trainable=False simula basis fixo (Keras3 compat fix)
    backcast = tf.keras.layers.Dense(output_dim, use_bias=False)(theta_b)
    forecast = tf.keras.layers.Dense(output_dim, use_bias=False)(theta_f)
    return backcast, forecast


def build_nbeats(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi N-BEATS para inversao seq2seq.

    N_STACKS stacks, cada com N_BLOCKS blocos MLP. Residual doubling:
    residual(t) = input(t) - backcast(t). Forecast acumulada = soma dos
    forecasts de todos os blocos.

    Adaptacao: input = (batch, seq_len, n_feat) → flatten por step.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: N-BEATS seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['N_BEATS']
            - tests/test_models.py: TestDecomposition.test_nbeats_forward
        Ref: Oreshkin et al. (2020) ICLR — N-BEATS.
    """
    import tensorflow as tf

    ap = config.arch_params or {}
    n_stacks = ap.get("n_stacks", 2)
    n_blocks = ap.get("n_blocks", 3)
    n_hidden = ap.get("n_hidden", 256)
    n_layers = ap.get("n_layers_per_block", 4)
    expansion_dim = ap.get("expansion_coef_dim", 32)

    seq_len = config.sequence_length
    n_feat = config.n_features
    out_ch = config.output_channels
    input_dim = seq_len * n_feat
    output_dim = seq_len * out_ch

    logger.info(
        "build_nbeats: n_stacks=%d, n_blocks=%d, n_hidden=%d",
        n_stacks, n_blocks, n_hidden,
    )

    inp = tf.keras.Input(shape=(seq_len, n_feat))

    # ── Flatten input para MLP ────────────────────────────────────────
    x_flat = tf.keras.layers.Reshape((input_dim,))(inp)

    # ── Projecao inicial se input_dim != output_dim ───────────────────
    residual = tf.keras.layers.Dense(output_dim)(x_flat)

    forecast_sum = tf.keras.layers.Lambda(lambda z: tf.zeros_like(z))(residual)

    for stack_i in range(n_stacks):
        for block_i in range(n_blocks):
            backcast, forecast = _nbeats_block(
                residual, n_hidden, n_layers, output_dim, expansion_dim
            )
            residual = tf.keras.layers.Subtract()([residual, backcast])
            forecast_sum = tf.keras.layers.Add()([forecast_sum, forecast])
            logger.debug("N-BEATS stack %d, block %d", stack_i + 1, block_i + 1)

    # ── Reshape forecast para (batch, seq_len, out_ch) ────────────────
    out = tf.keras.layers.Reshape((seq_len, out_ch))(forecast_sum)

    if config.use_physical_constraint_layer:
        out = tf.keras.layers.Activation(config.constraint_activation)(out)

    return tf.keras.Model(inputs=inp, outputs=out, name="N_BEATS")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: N-HiTS — NEURAL HIERARCHICAL INTERPOLATION
# ════════════════════════════════════════════════════════════════════════════
# N-HiTS: hierarquico multi-resolucao com max pooling expressivo.
# Cada stack usa uma escala diferente (pooling size crescente).
# Ref: Challu et al. (2023) AAAI.
# ──────────────────────────────────────────────────────────────────────────


def build_nhits(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi N-HiTS: hierarquico multi-escala para inversao.

    3 stacks com scales [1, 4, 8] (pooling crescente).
    Cada stack foca em componentes de frequencia diferentes:
    scale=1 → alta freq, scale=8 → baixa freq (tendencia).

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: N-HiTS seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['N_HiTS']
            - tests/test_models.py: TestDecomposition.test_nhits_forward
        Ref: Challu et al. (2023) AAAI — N-HiTS.
    """
    import tensorflow as tf
    from geosteering_ai.models.blocks import series_decomp_block

    ap = config.arch_params or {}
    n_stacks = ap.get("n_stacks", 3)
    pool_sizes = ap.get("pool_sizes", [1, 4, 8])
    n_hidden = ap.get("n_hidden", 256)
    n_layers = ap.get("n_layers", 2)
    expansion_dim = ap.get("expansion_coef_dim", 32)

    seq_len = config.sequence_length
    n_feat = config.n_features
    out_ch = config.output_channels
    output_dim = seq_len * out_ch

    # Garantir que pool_sizes tenha n_stacks elementos
    while len(pool_sizes) < n_stacks:
        pool_sizes.append(pool_sizes[-1] * 2)
    pool_sizes = pool_sizes[:n_stacks]

    logger.info(
        "build_nhits: n_stacks=%d, pool_sizes=%s, n_hidden=%d",
        n_stacks, pool_sizes, n_hidden,
    )

    inp = tf.keras.Input(shape=(seq_len, n_feat))
    x = inp

    forecast_sum = tf.keras.layers.Lambda(
        lambda z: tf.zeros((tf.shape(z)[0], seq_len, out_ch))
    )(inp)

    for stack_i, pool_size in enumerate(pool_sizes):
        # ── Decomposicao: stack foca em componente especifica ─────────
        if pool_size > 1:
            _, trend = series_decomp_block(x, kernel_size=min(pool_size * 2 + 1, 25))
            x_input = trend
        else:
            x_input = x

        # ── Pooling expressivo ────────────────────────────────────────
        if pool_size > 1:
            x_pool = tf.keras.layers.AveragePooling1D(pool_size, padding="same")(x_input)
        else:
            x_pool = x_input

        # ── MLP stack ─────────────────────────────────────────────────
        x_flat = tf.keras.layers.Flatten()(x_pool)
        h = x_flat
        for _ in range(n_layers):
            h = tf.keras.layers.Dense(n_hidden, activation="relu")(h)

        # ── Forecast desta escala ─────────────────────────────────────
        forecast_flat = tf.keras.layers.Dense(output_dim)(h)
        forecast = tf.keras.layers.Reshape((seq_len, out_ch))(forecast_flat)
        forecast_sum = tf.keras.layers.Add()([forecast_sum, forecast])
        logger.debug("N-HiTS stack %d: pool_size=%d", stack_i + 1, pool_size)

    out = forecast_sum
    if config.use_physical_constraint_layer:
        out = tf.keras.layers.Activation(config.constraint_activation)(out)

    return tf.keras.Model(inputs=inp, outputs=out, name="N_HiTS")


__all__ = ["build_nbeats", "build_nhits"]
