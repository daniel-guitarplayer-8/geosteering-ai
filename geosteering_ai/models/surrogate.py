# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/surrogate.py                                               ║
# ║  Bloco: 3b — SurrogateNet: Forward Model Neural para PINNs               ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (ponto unico de verdade)                ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • SurrogateNet: rede TCN dilatada que aprende rho(z) → H_EM(z)        ║
# ║    • build_surrogate(): factory que constroi o modelo a partir do config  ║
# ║    • Campo receptivo configuravel (~127m) para dependencia nao-local      ║
# ║    • Suporte a K componentes EM configuraveis (Modo A/B/C)               ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig)                                 ║
# ║  Exports: ~2 funcoes/classes — ver __all__                                ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 18.5 (SurrogateNet)                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.2 (2026-04) — Implementacao inicial (Passo 2, Modo B)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""SurrogateNet — rede TCN que aprende o forward model rho(z) → H_EM(z).

Arquitetura baseada em Temporal Convolutional Network (TCN) com dilatacoes
exponenciais para capturar dependencias nao-locais do campo EM:

  ┌─────────────────────────────────────────────────────────────────┐
  │  SurrogateNet — Forward Model Neural Surrogate                 │
  │                                                                 │
  │  Input: (B, N, 2) — [log10(rho_h), log10(rho_v)]              │
  │    ↓                                                            │
  │  TCN Encoder (6 blocos dilatados):                             │
  │    Bloco 1: Conv1D(64,  k=3, d=1)  → campo rec. = 2m         │
  │    Bloco 2: Conv1D(128, k=3, d=2)  → campo rec. = 6m         │
  │    Bloco 3: Conv1D(128, k=3, d=4)  → campo rec. = 14m        │
  │    Bloco 4: Conv1D(256, k=3, d=8)  → campo rec. = 30m        │
  │    Bloco 5: Conv1D(256, k=3, d=16) → campo rec. = 62m        │
  │    Bloco 6: Conv1D(256, k=3, d=32) → campo rec. = 126m       │
  │    ↓                                                            │
  │  Decoder: Dense(128) → Dense(64) → Dense(2*K, 'linear')       │
  │    ↓                                                            │
  │  Output: (B, N, 2*K) — Re+Im de K componentes EM              │
  └─────────────────────────────────────────────────────────────────┘

O campo receptivo de ~126m cobre ~3x o skin depth maximo (delta ≈ 35.7m
para rho=100 Ohm.m, f=20kHz), garantindo que o modelo captura os efeitos
de camadas adjacentes (shoulder beds) no campo EM medido.

Ref: Oord et al. (2016) WaveNet; Bai et al. (2018) TCN benchmark.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente: factory, constantes.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    "build_surrogate",
    "build_surrogate_modern",
]


# ════════════════════════════════════════════════════════════════════════════
# SECAO: BLOCO TCN DILATADO PARA SURROGATE
# ════════════════════════════════════════════════════════════════════════════
# Cada bloco TCN consiste em: Conv1D dilatada → BatchNorm → ReLU → Dropout.
# O padding "causal" garante que a saida no ponto z nao usa informacao
# futura (z' > z), permitindo uso em modo streaming.
# A skip connection (residual) estabiliza o treinamento em redes profundas.
#
# Campo receptivo por bloco: rf = (kernel_size - 1) × dilation
# Campo receptivo total: sum(rf_i) + 1 para 6 blocos com d=1,2,4,8,16,32
# = (2×1 + 2×2 + 2×4 + 2×8 + 2×16 + 2×32) + 1 = 127
# ──────────────────────────────────────────────────────────────────────────


def _surrogate_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    """Bloco TCN dilatado para SurrogateNet.

    Estrutura:
      x → Conv1D(causal, dilatado) → BatchNorm → ReLU → Dropout → + skip → out

    Args:
        x: Tensor de entrada (B, N, C_in).
        filters: Numero de filtros Conv1D.
        kernel_size: Tamanho do kernel (default 3).
        dilation_rate: Fator de dilatacao (1, 2, 4, 8, 16, 32).
        dropout_rate: Taxa de dropout (0.0-1.0).

    Returns:
        Tensor de saida (B, N, filters) com skip connection residual.

    Note:
        Funcao privada — usada apenas por build_surrogate().
        Ref: Bai et al. (2018) "An Empirical Evaluation of Generic
             Convolutional and Recurrent Networks for Sequence Modeling".
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    # ── Conv1D causal dilatada ────────────────────────────────────
    # padding="causal" garante que a saida no ponto t depende
    # somente de entradas em t' <= t (sem vazamento temporal).
    conv = layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        kernel_initializer="he_normal",
    )(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.ReLU()(conv)
    if dropout_rate > 0:
        conv = layers.Dropout(dropout_rate)(conv)

    # ── Skip connection (residual) ────────────────────────────────
    # Se C_in != filters, projecao 1x1 para compatibilizar shapes.
    # Usa getattr para Keras 3.x: KerasTensor.shape[-1] pode ser None
    # durante graph construction — nesse caso, projeta sempre (seguro).
    in_channels = getattr(x.shape, "__getitem__", lambda _: None)(-1)
    if in_channels is None or in_channels != filters:
        x = layers.Conv1D(filters, 1, padding="same")(x)

    # Nota: ReLU pos-adicao omitido intencionalmente — o proximo bloco
    # TCN inicia com Conv1D → BN → ReLU, funcionando como pre-activation
    # residual (He et al. 2016 Identity Mappings). Isso permite que o
    # gradiente flua sem atenuacao pela skip connection limpa.
    return layers.Add()([x, conv])


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FACTORY — BUILD_SURROGATE
# ════════════════════════════════════════════════════════════════════════════
# Factory que constroi o SurrogateNet a partir do PipelineConfig.
# O numero de canais de saida (2*K) eh determinado automaticamente
# pela lista config.surrogate_output_components.
# ──────────────────────────────────────────────────────────────────────────


def build_surrogate(
    config: "PipelineConfig",
    *,
    dropout_rate: float = 0.1,
) -> "tf.keras.Model":
    """Constroi SurrogateNet TCN para forward model rho → H_EM.

    Cria um modelo Keras que mapeia perfis de resistividade para
    campos EM, usando TCN dilatado para capturar dependencias
    nao-locais (shoulder bed effects).

    Arquitetura:
      ┌──────────────────────────────────────────────────────────────┐
      │  Input (B, None, 2) — [log10(rho_h), log10(rho_v)]        │
      │    ↓                                                        │
      │  6 × TCN Block: Conv1D → BN → ReLU → Dropout → Skip       │
      │    dilatacoes: [1, 2, 4, 8, 16, 32]                       │
      │    filtros:    [64, 128, 128, 256, 256, 256]               │
      │    campo receptivo total: 127 pontos (~127 m)              │
      │    ↓                                                        │
      │  Dense(128) → ReLU → Dense(64) → ReLU                     │
      │    ↓                                                        │
      │  Dense(n_outputs, 'linear')                                 │
      │    ↓                                                        │
      │  Output (B, None, 2*K) — Re+Im de K componentes            │
      └──────────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com:
            - surrogate_output_components: Lista de componentes EM.
              K componentes → 2*K canais de saida.
              Ex: ["XX", "ZZ"] → 4 canais,
                  ["XX", "ZZ", "XZ", "ZX"] → 8 canais.
        dropout_rate: Taxa de dropout nos blocos TCN (default 0.1).

    Returns:
        tf.keras.Model: Modelo funcional Keras.
            Input shape: (None, None, 2)
            Output shape: (None, None, 2*K)

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        >>> model = build_surrogate(config)
        >>> model.summary()
        >>> assert model.output_shape == (None, None, 4)

    Note:
        Referenciado em:
            - losses/pinns.py: make_surrogate_physics_loss() (modo neural)
            - tests/test_surrogate.py: TestBuildSurrogate
        Ref: docs/ARCHITECTURE_v2.md secao 18.5.
             Bai et al. (2018) — TCN benchmark.
             Oord et al. (2016) — WaveNet (causal dilated convolutions).
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    n_components = len(config.surrogate_output_components)
    n_outputs = 2 * n_components  # Re + Im para cada componente

    # ── Input: (B, None, 2) — sequencia de resistividade ──────────
    # None permite sequencia de comprimento variavel (multi-angulo).
    # Canal 0: log10(rho_h), Canal 1: log10(rho_v).
    inp = layers.Input(shape=(None, 2), name="surrogate_input")

    # ── TCN Encoder: 6 blocos com dilatacao exponencial ────────────
    # Dilatacoes: [1, 2, 4, 8, 16, 32] → campo receptivo = 127 pontos.
    # Para SPACING_METERS=1.0, isso corresponde a ~127 m de profundidade,
    # cobrindo ~3× o skin depth maximo (delta ≈ 35.7m para rho=100, f=20kHz).
    #
    # Filtros: crescem de 64 → 256 para capturar features progressivamente
    # mais abstratas (camadas finas → transicoes → tendencias regionais).
    _tcn_config = [
        # (filters, dilation_rate)
        (64, 1),  # Bloco 1: features locais (1-3 m)
        (128, 2),  # Bloco 2: features de curto alcance (3-7 m)
        (128, 4),  # Bloco 3: features de medio alcance (7-15 m)
        (256, 8),  # Bloco 4: shoulder bed effects (15-31 m)
        (256, 16),  # Bloco 5: efeitos regionais (31-63 m)
        (256, 32),  # Bloco 6: dependencias de longo alcance (63-127 m)
    ]

    x = inp
    for filters, dilation in _tcn_config:
        x = _surrogate_tcn_block(
            x,
            filters,
            kernel_size=3,
            dilation_rate=dilation,
            dropout_rate=dropout_rate,
        )

    # ── Decoder: projecao para canais de saida ─────────────────────
    # Dense(128) → Dense(64) → Dense(2*K): transicao gradual de features
    # abstratas para canais Re/Im fisicos.
    x = layers.Dense(128, activation="relu", name="decoder_dense1")(x)
    x = layers.Dense(64, activation="relu", name="decoder_dense2")(x)

    # ── Output: (B, None, 2*K) — linear (sem ativacao) ────────────
    # Re(H) e Im(H) podem ser positivos ou negativos, portanto
    # a ativacao final deve ser linear (sem restricao de range).
    out = layers.Dense(n_outputs, activation="linear", name="surrogate_output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="SurrogateNet")

    logger.info(
        "SurrogateNet construido: %d componentes (%s), %d canais saida, "
        "%d parametros, campo receptivo ~127 pontos",
        n_components,
        "/".join(config.surrogate_output_components),
        n_outputs,
        model.count_params(),
    )

    return model


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FACTORY — BUILD_SURROGATE_MODERN (ModernTCN)
# ════════════════════════════════════════════════════════════════════════════
# SurrogateNet v2 baseado em ModernTCN: large-kernel DWConv + ConvFFN.
# Campo receptivo ~200+ m (vs ~127 m do TCN classico), ~50% menos params.
# LayerNorm invariante a batch — sem degradacao em B=1 (realtime).
# Ref: Luo & Wang (ICLR 2024) arXiv:2310.06625.
# ──────────────────────────────────────────────────────────────────────────


def _surrogate_modern_block(x, filters, large_kernel, dropout_rate):
    """Bloco ModernTCN para SurrogateNet v2.

    Estrutura:
      x → LN → DWConv(k=large_kernel) → LN → Dense(4C) → GELU
        → Dense(C) → Dropout → +skip → out

    Args:
        x: Tensor de entrada (B, N, C).
        filters: Numero de canais C.
        large_kernel: Tamanho do kernel DWConv (default 51).
        dropout_rate: Taxa de dropout (0.0-1.0).

    Returns:
        Tensor (B, N, C) com skip connection residual.

    Note:
        Funcao privada — usada apenas por build_surrogate_modern().
        DWConv processa cada canal (Re/Im de cada componente EM)
        independentemente, respeitando a fisicidade do sinal.
        Ref: Luo & Wang (ICLR 2024) Secao 3.2.
    """
    import tensorflow as tf

    skip = x

    # ── DWConv temporal ───────────────────────────────────────────────
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.DepthwiseConv1D(
        kernel_size=large_kernel,
        padding="causal",
        depth_multiplier=1,
        use_bias=False,
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)

    # ── ConvFFN (mixing de canais) ────────────────────────────────────
    x = tf.keras.layers.Dense(filters * 4, use_bias=False)(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dense(filters, use_bias=False)(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # ── Projecao skip se dimensoes incompativeis ──────────────────────
    in_channels = getattr(skip.shape, "__getitem__", lambda _: None)(-1)
    if in_channels is None or in_channels != filters:
        skip = tf.keras.layers.Dense(filters, use_bias=False)(skip)

    return tf.keras.layers.Add()([skip, x])


def build_surrogate_modern(
    config: "PipelineConfig",
    *,
    dropout_rate: float = 0.1,
) -> "tf.keras.Model":
    """Constroi SurrogateNet v2 (ModernTCN) para forward model rho → H_EM.

    Versao modernizada do SurrogateNet com large-kernel DWConv,
    LayerNorm e ConvFFN. Campo receptivo ~200+ m (vs ~127 m do TCN
    classico) e ~50% menos parametros.

    Arquitetura:
      ┌──────────────────────────────────────────────────────────────┐
      │  Input (B, None, 2) — [log10(rho_h), log10(rho_v)]        │
      │    ↓                                                        │
      │  Stem: Conv1D(128, k=1) — projecao de canais                │
      │    ↓                                                        │
      │  4 × ModernTCN Block:                                       │
      │    LN → DWConv(k=51, causal) → LN                          │
      │    → Dense(C→4C) → GELU → Dense(4C→C) → Drop → +skip      │
      │    ↓                                                        │
      │  LN → Dense(128) → ReLU → Dense(64) → ReLU                │
      │    ↓                                                        │
      │  Dense(2*K, 'linear')                                       │
      │    ↓                                                        │
      │  Output (B, None, 2*K) — Re+Im de K componentes EM        │
      └──────────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com surrogate_output_components.
        dropout_rate: Taxa de dropout nos blocos (default 0.1).

    Returns:
        tf.keras.Model: SurrogateNet v2 (ModernTCN).
            Input shape: (None, None, 2)
            Output shape: (None, None, 2*K)

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        >>> model = build_surrogate_modern(config)
        >>> assert model.output_shape == (None, None, 4)

    Note:
        Referenciado em:
            - losses/pinns.py: make_surrogate_physics_loss() (modo neural)
            - tests/test_surrogate.py: TestBuildSurrogateModern
        Campo receptivo: 51 × 4 blocos = 204 pontos (~204 m).
        Cobre ~5.7× skin depth maximo (delta ≈ 35.7m, rho=100, f=20kHz).
        Ref: Luo & Wang (ICLR 2024) arXiv:2310.06625.
    """
    import tensorflow as tf

    n_components = len(config.surrogate_output_components)
    n_outputs = 2 * n_components

    filters = 128
    n_blocks = 4
    large_kernel = 51

    inp = tf.keras.layers.Input(shape=(None, 2), name="surrogate_modern_input")

    # ── Stem ──────────────────────────────────────────────────────────
    x = tf.keras.layers.Conv1D(filters, 1, padding="same", use_bias=False)(inp)

    # ── ModernTCN blocks ──────────────────────────────────────────────
    for _ in range(n_blocks):
        x = _surrogate_modern_block(x, filters, large_kernel, dropout_rate)

    # ── Decoder ───────────────────────────────────────────────────────
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="modern_decoder1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="modern_decoder2")(x)
    out = tf.keras.layers.Dense(
        n_outputs, activation="linear", name="surrogate_modern_output"
    )(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="SurrogateNet_Modern")

    logger.info(
        "SurrogateNet_Modern construido: %d componentes (%s), %d canais saida, "
        "%d parametros, campo receptivo ~204 pontos",
        n_components,
        "/".join(config.surrogate_output_components),
        n_outputs,
        model.count_params(),
    )

    return model
