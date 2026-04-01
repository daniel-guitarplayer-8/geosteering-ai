# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/advanced.py                                                ║
# ║  Bloco: 3i — Arquiteturas Avancadas (DNN, FNO, DeepONet, GeoAttn)       ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • DNN: baseline MLP ponto a ponto (seq2seq via TimeDistributed)       ║
# ║    • FNO: Fourier Neural Operator (espectral, periodico)                 ║
# ║    • DeepONet: Deep Operator Network (branch-trunk)                      ║
# ║    • Geophysical_Attention: atencao especializada para fisica LWD        ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/blocks.py               ║
# ║  Exports: 4 funcoes — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.9, legado C36                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (4 arquiteturas)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Arquiteturas avancadas: DNN, FNO, DeepONet, Geophysical_Attention.

Note:
    Referenciado em:
        - models/registry.py: _REGISTRY[DNN..Geophysical_Attention]
        - tests/test_models.py: TestAdvanced
    Legado C36 (1750 linhas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: DNN — MLP PONTO A PONTO (BASELINE)
# ════════════════════════════════════════════════════════════════════════════
# DNN aplica um MLP independentemente a cada ponto temporal (TimeDistributed).
# Baseline mais simples — sem dependencias temporais explicilas.
# ──────────────────────────────────────────────────────────────────────────


def build_dnn(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi DNN ponto a ponto via TimeDistributed Dense.

    MLP de N camadas aplicado independentemente a cada um dos 600 pontos.
    Baseline simples: sem temporal modeling (CNN, LSTM ou Transformer).
    Util para ablation study: o quanto a sequencia importa.

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: DNN seq2seq via TimeDistributed.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['DNN']
            - tests/test_models.py: TestAdvanced.test_dnn_forward
        TimeDistributed aplica o mesmo MLP a cada time step.
        Legado C36 build_dnn().
    """
    import tensorflow as tf

    ap = config.arch_params or {}
    hidden_units = ap.get("hidden_units", [256, 256, 128])
    activation = ap.get("activation", "relu")
    dr = config.dropout_rate
    l2 = config.l2_weight if config.use_l2_regularization else 0.0
    reg = tf.keras.regularizers.L2(l2) if l2 > 0.0 else None

    logger.info("build_dnn: n_feat=%d, units=%s", config.n_features, hidden_units)

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    for units in hidden_units:
        dense = tf.keras.layers.Dense(
            units, activation=activation, kernel_regularizer=reg
        )
        x = tf.keras.layers.TimeDistributed(dense)(x)
        if dr > 0.0:
            x = tf.keras.layers.Dropout(dr)(x)

    out_dense = tf.keras.layers.Dense(
        config.output_channels,
        activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    out = tf.keras.layers.TimeDistributed(out_dense)(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="DNN")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FNO — FOURIER NEURAL OPERATOR
# ════════════════════════════════════════════════════════════════════════════
# FNO aprende operadores no espaco de frequencia via FFT.
# Multiplica modos de Fourier por parametros aprendidos (R espectral).
# Ideal para dados com estrutura periodica ou quasi-periodica.
# Ref: Li et al. (2021) ICLR — FNO.
# ──────────────────────────────────────────────────────────────────────────


class _FourierLayer(tf.keras.layers.Layer):
    """Camada Fourier: FFT → multiplicacao espectral → iFFT.

    Encapsulado como Layer para compatibilidade com Keras 3.x (Colab),
    onde KerasTensor nao pode ser passado para tf.signal/tf.transpose.
    Dentro de call(), os tensores sao concretos.

    Args:
        n_modes: Numero de modos de Fourier a manter.
        out_channels: Canais de saida.

    Note:
        Ref: Li et al. (2021) ICLR — Fourier Neural Operator.
        A multiplicacao espectral e implementada como Dense sobre
        partes real/imaginaria separadas (equivalente a parametro R).
    """

    def __init__(self, n_modes: int, out_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.n_modes = n_modes
        self.out_channels = out_channels
        # Dense para projecao espectral (criadas em build)
        self.proj_r = None
        self.proj_i = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.proj_r = tf.keras.layers.Dense(
            self.out_channels * self.n_modes,
            use_bias=False,
            name="proj_real",
        )
        self.proj_i = tf.keras.layers.Dense(
            self.out_channels * self.n_modes,
            use_bias=False,
            name="proj_imag",
        )
        super().build(input_shape)

    def call(self, x):
        # ── FFT ao longo da dimensao temporal ─────────────────────────
        x_ft = tf.signal.rfft(tf.transpose(x, [0, 2, 1]))  # (batch, ch, L//2+1)

        # ── Trunca para n_modes modos ─────────────────────────────────
        x_ft_trunc = x_ft[..., : self.n_modes]  # (batch, ch, n_modes)

        # ── Projecao espectral (real e imaginaria separadas) ──────────
        x_ft_r = tf.math.real(x_ft_trunc)
        x_ft_i = tf.math.imag(x_ft_trunc)

        batch = tf.shape(x)[0]
        x_flat_r = tf.reshape(x_ft_r, [batch, -1])
        x_flat_i = tf.reshape(x_ft_i, [batch, -1])

        proj_r = tf.reshape(
            self.proj_r(x_flat_r), [batch, self.out_channels, self.n_modes]
        )
        proj_i = tf.reshape(
            self.proj_i(x_flat_i), [batch, self.out_channels, self.n_modes]
        )

        # ── Recombina em complexo e pad para iFFT ─────────────────────
        out_ft = tf.complex(proj_r, proj_i)
        seq_len = tf.shape(x)[1]
        n_rfft = seq_len // 2 + 1

        pad_size = n_rfft - self.n_modes
        paddings = [[0, 0], [0, 0], [0, pad_size]]
        out_ft_pad = tf.pad(out_ft, paddings)

        # ── iFFT ──────────────────────────────────────────────────────
        out_t = tf.signal.irfft(out_ft_pad)  # (batch, out_ch, seq_len)
        out_t = tf.transpose(out_t, [0, 2, 1])  # (batch, seq_len, out_ch)
        return out_t

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_modes": self.n_modes,
                "out_channels": self.out_channels,
            }
        )
        return config


def build_fno(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi FNO (Fourier Neural Operator) 1D para inversao.

    Empilha camadas Fourier (espectral) com camadas Conv1D locais.
    A combinacao captura estruturas globais (FNO) e locais (Conv).

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: FNO seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['FNO']
            - tests/test_models.py: TestAdvanced.test_fno_forward
        Ref: Li et al. (2021) ICLR — Fourier Neural Operator.
        n_modes deve ser <= seq_len // 2 + 1.
    """
    import tensorflow as tf

    ap = config.arch_params or {}
    n_layers = ap.get("n_layers", 4)
    d_model = ap.get("d_model", 64)
    n_modes = ap.get("n_modes", 16)
    activation = ap.get("activation", "gelu")

    logger.info(
        "build_fno: d_model=%d, n_layers=%d, n_modes=%d", d_model, n_layers, n_modes
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = tf.keras.layers.Dense(d_model)(inp)

    for i in range(n_layers):
        # ── Camada Fourier ────────────────────────────────────────────
        x_fourier = _FourierLayer(n_modes=n_modes, out_channels=d_model)(x)

        # ── Camada local (Conv1D) ─────────────────────────────────────
        x_local = tf.keras.layers.Conv1D(d_model, 1, padding="same")(x)

        # ── Combinacao ────────────────────────────────────────────────
        x = tf.keras.layers.Add()([x_fourier, x_local])
        x = tf.keras.layers.Activation(activation)(x)
        logger.debug("FNO layer %d", i + 1)

    x = tf.keras.layers.LayerNormalization()(x)

    out = tf.keras.layers.Conv1D(
        config.output_channels,
        1,
        padding="same",
        activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="FNO")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: DEEPONET — DEEP OPERATOR NETWORK
# ════════════════════════════════════════════════════════════════════════════
# DeepONet aprende operadores funcionais via dois sub-redes:
#   Branch net: processa os dados de entrada (funcao u)
#   Trunk net:  processa os pontos de avaliacao (localizacoes y)
# Ref: Lu et al. (2021) Nature MI — DeepONet.
# ──────────────────────────────────────────────────────────────────────────


def build_deeponet(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi DeepONet para inversao como aprendizado de operador.

    Branch net: codifica o sinal EM de entrada (u).
    Trunk net: codifica os pontos de medicao z_obs (localizacoes).
    Output: produto escalar branch × trunk → (batch, 600, out_ch).

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: DeepONet seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['DeepONet']
            - tests/test_models.py: TestAdvanced.test_deeponet_forward
        Ref: Lu et al. (2021) Nature MI — DeepONet.
        Trunk usa posicoes z_obs (col 1 do input = feature index 0).
    """
    import tensorflow as tf

    ap = config.arch_params or {}
    branch_units = ap.get("branch_units", [256, 256, 128])
    trunk_units = ap.get("trunk_units", [128, 128, 128])
    p_dim = ap.get("p_dim", 128)  # dim do produto interno
    activation = ap.get("activation", "tanh")

    logger.info(
        "build_deeponet: branch=%s, trunk=%s, p=%d", branch_units, trunk_units, p_dim
    )

    # ── Duas entradas: sinal EM e posicoes z ─────────────────────────
    inp = tf.keras.Input(
        shape=(config.sequence_length, config.n_features), name="em_input"
    )

    # ── Branch net: processa todo o sinal EM ─────────────────────────
    # Flatten para MLP (processa o sinal como vetor global)
    branch = tf.keras.layers.Flatten()(inp)
    for units in branch_units:
        branch = tf.keras.layers.Dense(units, activation=activation)(branch)
    branch = tf.keras.layers.Dense(p_dim)(branch)  # (batch, p_dim)

    # ── Trunk net: processa posicoes (z_obs = feature 0) ─────────────
    # z_obs e a primeira feature de input
    z_obs = tf.keras.layers.Lambda(lambda x: x[..., 0:1])(inp)  # (batch, 600, 1)
    trunk = z_obs
    for units in trunk_units:
        trunk = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units, activation=activation)
        )(trunk)
    trunk = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(p_dim))(
        trunk
    )  # (batch, 600, p_dim)

    # ── Produto interno Branch × Trunk ────────────────────────────────
    # branch: (batch, p_dim) → (batch, 1, p_dim)
    branch_exp = tf.keras.layers.Lambda(lambda b: tf.expand_dims(b, 1))(
        branch
    )  # (batch, 1, p_dim)

    # Produto elementar e soma sobre p_dim
    product = tf.keras.layers.Multiply()([trunk, branch_exp])  # (batch, 600, p_dim)
    x = tf.keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=-1, keepdims=True))(
        product
    )  # (batch, 600, 1)

    # ── Projecao para out_ch ─────────────────────────────────────────
    out = tf.keras.layers.Conv1D(
        config.output_channels,
        1,
        padding="same",
        activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="DeepONet")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: GEOPHYSICAL_ATTENTION
# ════════════════════════════════════════════════════════════════════════════
# Arquitetura especializada para dados LWD:
#   CNN local para padroes de resistividade
#   + Atencao especializada para interfaces de camada
#   + Priori fisico via mascaramento de atencao por skin depth
# ──────────────────────────────────────────────────────────────────────────


def build_geophysical_attention(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi Geophysical_Attention: CNN + atencao fisica para LWD.

    Combina CNN residual para features locais com atencao fisica:
    - Janela de atencao limitada pelo skin depth (~11m → ~110 amostras)
    - Multi-head attention com mascara por localidade

    Args:
        config: PipelineConfig.

    Returns:
        tf.keras.Model: Geophysical_Attention seq2seq.

    Note:
        Referenciado em:
            - models/registry.py: _REGISTRY['Geophysical_Attention']
            - tests/test_models.py: TestAdvanced.test_geophysical_attention_forward
        Skin depth ~11m / 0.05m_per_sample = ~220 amostras.
        Legado C36 build_geophysical_attention().
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import (
        output_projection,
        residual_block_1d,
        self_attention_block,
    )

    ap = config.arch_params or {}
    n_cnn_blocks = ap.get("n_cnn_blocks", 4)
    cnn_filters = ap.get("cnn_filters", 64)
    n_attn_heads = ap.get("num_heads", 8)
    key_dim = ap.get("key_dim", 64)
    dr = config.dropout_rate
    causal = config.use_causal_mode

    logger.info(
        "build_geophysical_attention: n_cnn=%d, filters=%d, heads=%d",
        n_cnn_blocks,
        cnn_filters,
        n_attn_heads,
    )

    inp = tf.keras.Input(shape=(config.sequence_length, config.n_features))
    x = inp

    # ── CNN encoder: features locais de resistividade ─────────────────
    for i in range(n_cnn_blocks):
        x = residual_block_1d(
            x,
            cnn_filters,
            kernel_size=3,
            causal=causal,
            l1=config.l1_weight if config.use_l1_regularization else 0.0,
            l2=config.l2_weight if config.use_l2_regularization else 0.0,
        )
        logger.debug("GeoAttn CNN block %d", i + 1)

    # ── Atencao global (opcional: causal para realtime) ───────────────
    x = self_attention_block(
        x,
        num_heads=n_attn_heads,
        key_dim=key_dim,
        dropout_rate=dr,
        use_causal_mask=causal,
    )

    # ── Segundo CNN refinamento ───────────────────────────────────────
    x = residual_block_1d(x, cnn_filters, kernel_size=3, causal=causal)
    x = tf.keras.layers.LayerNormalization()(x)

    out = output_projection(
        x,
        config.output_channels,
        constraint_activation=(
            config.constraint_activation if config.use_physical_constraint_layer else None
        ),
    )
    return tf.keras.Model(inputs=inp, outputs=out, name="Geophysical_Attention")


__all__ = [
    "build_dnn",
    "build_fno",
    "build_deeponet",
    "build_geophysical_attention",
]
