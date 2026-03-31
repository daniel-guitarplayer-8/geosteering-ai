# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: noise/functions.py                                                ║
# ║  Bloco: 2c — Noise On-The-Fly                                             ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Funcoes TF de noise on-the-fly para tf.data.map (4 tipos core)       ║
# ║    • NOISE_FN_MAP: registro tipo→funcao (extensivel)                      ║
# ║    • apply_noise_tf(): dispatcher que mixa N tipos com pesos              ║
# ║    • apply_raw_em_noise(): versao numpy para uso offline/testes           ║
# ║    • create_noise_level_var(): factory para tf.Variable compartilhado     ║
# ║                                                                            ║
# ║  Dependencias: tensorflow, numpy, config.py (PipelineConfig)              ║
# ║  Exports: ~7 simbolos — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4.5 (noise)                          ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CONSTANTES E TIPOS
# ════════════════════════════════════════════════════════════════════════════
# Definicoes de tipos e constantes para o sistema de noise.
# O noise eh aplicado APENAS nas 4 colunas EM (Re/Im de Hxx e Hzz),
# NUNCA na coluna z_obs (primeira coluna dos INPUT_FEATURES).
# Ref: Errata v5.0.15, CLAUDE.md (noise on-the-fly obrigatorio com FV/GS).
# ──────────────────────────────────────────────────────────────────────────

#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  4 Tipos de Noise Core (TF on-the-fly):                                │
#   │                                                                          │
#   │  Tipo           │ Formula                    │ Dominio Fisico            │
#   │  ───────────────┼────────────────────────────┼───────────────────────────│
#   │  gaussian       │ x + N(0, σ²)              │ Ruido eletronico aditivo  │
#   │  multiplicative │ x · (1 + N(0, σ²))        │ Erro de ganho do amp.    │
#   │  uniform        │ x + U(-σ, σ)              │ Quantizacao ADC           │
#   │  dropout        │ x · Bernoulli(1-σ)/(1-σ)  │ Dropout de canal/amostra │
#   │                                                                          │
#   │  σ = noise_level_var (tf.Variable, controlado pelo curriculum)          │
#   │  Aplicado ANTES de FV e GS (fidelidade LWD)                            │
#   │  Colunas protegidas (0:n_protected) NUNCA recebem noise:                 │
#   │    P1: n_protected=1 → [z_obs]                                          │
#   │    P2: n_protected=2 → [theta_norm, z_obs]                              │
#   │    P3: n_protected=2 → [f_norm, z_obs]                                  │
#   │    P2+P3: n_protected=3 → [theta_norm, f_norm, z_obs]                   │
#   └──────────────────────────────────────────────────────────────────────────┘

# Tipo callable para funcoes de noise TF.
# Assinatura: (x: tf.Tensor, noise_level: tf.Variable, n_protected: int) -> tf.Tensor
NoiseFnType = Callable  # tf.Tensor, tf.Variable, int -> tf.Tensor


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCOES DE NOISE TENSORFLOW
# ════════════════════════════════════════════════════════════════════════════
# Cada funcao recebe um tensor 3D (batch, seq_len, n_feat) e um
# tf.Variable com o nivel de ruido atual (controlado pelo curriculum).
# O noise eh aplicado APENAS nas colunas EM (indices n_protected: em diante),
# preservando as primeiras n_protected colunas intactas (theta?, freq?, z_obs).
# n_protected = config.n_prefix + 1 (default 1 para P1 baseline).
# Ref: Cadeia on-the-fly: raw → noise → FV → GS → scale → modelo.
# ──────────────────────────────────────────────────────────────────────────


def _add_gaussian_noise_tf(x, noise_level, n_protected=1):
    """Noise gaussiano aditivo: x + N(0, sigma^2).

    Modela ruido eletronico aditivo do receptor LWD. Tipo mais
    comum e fisicamente motivado para medidas EM.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat). Layout P1:
            [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz), ...GS].
            Layout P2+P3: [theta_norm?, f_norm?, z_obs, Re(Hxx), ...].
        noise_level: tf.Variable escalar com sigma atual.
        n_protected: Colunas iniciais protegidas (parametros conhecidos
            + z_obs). Default 1 (P1). P2: 2, P3: 2, P2+P3: 3.
            Calculado como config.n_prefix + 1.

    Returns:
        tf.Tensor: x com noise gaussiano nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["gaussian"].
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    # ── Separar colunas protegidas das features EM ────────────────────
    protected = x[:, :, :n_protected]  # (batch, seq, n_prot) — NUNCA noise
    em_feats = x[:, :, n_protected:]  # (batch, seq, n_em) — recebe noise
    noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level,
        dtype=tf.float32,
    )
    return tf.concat([protected, em_feats + noise], axis=-1)


def _add_multiplicative_noise_tf(x, noise_level, n_protected=1):
    """Noise multiplicativo (speckle): x * (1 + N(0, sigma^2)).

    Modela erros de ganho do amplificador LWD e variacao de
    sensibilidade do receptor com temperatura/pressao.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise multiplicativo nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["multiplicative"].
        Ref: Noise tipo "speckle" no catalogo C12 legado.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level,
        dtype=tf.float32,
    )
    return tf.concat([protected, em_feats * (1.0 + noise)], axis=-1)


def _add_uniform_noise_tf(x, noise_level, n_protected=1):
    """Noise uniforme: x + U(-sigma, sigma).

    Modela erro de quantizacao ADC (resolucao finita do conversor
    analogico-digital).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise uniforme nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["uniform"].
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    noise = tf.random.uniform(
        shape=tf.shape(em_feats),
        minval=-noise_level,
        maxval=noise_level,
        dtype=tf.float32,
    )
    return tf.concat([protected, em_feats + noise], axis=-1)


def _add_dropout_noise_tf(x, noise_level, n_protected=1):
    """Noise dropout: mascara aleatoria de canais/amostras.

    Modela perda de dados por falha de telemetria ou dropout de
    canal durante perfuracao. Usa inverted dropout para manter
    escala esperada.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar (probabilidade de dropout).
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com dropout nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["dropout"].
        Inverted dropout: divide por (1-p) para manter escala.
        Se noise_level >= 1.0, retorna zeros (protecao).
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Inverted dropout: mask * x / (1-p) ────────────────────────────
    keep_prob = tf.maximum(1.0 - noise_level, 1e-6)
    mask = tf.cast(
        tf.random.uniform(shape=tf.shape(em_feats)) < keep_prob,
        dtype=tf.float32,
    )
    em_dropped = em_feats * mask / keep_prob
    return tf.concat([protected, em_dropped], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCOES DE NOISE CORE — 5 TIPOS ADICIONAIS
# ════════════════════════════════════════════════════════════════════════════
# Complementam os 4 tipos originais (gaussian, multiplicative, uniform,
# dropout) com 5 fenomenos fisicos presentes em 100% dos pocos LWD reais.
#
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  5 Tipos CORE Adicionais:                                               │
#   │                                                                          │
#   │  Tipo             │ Formula                         │ Fenomeno Fisico    │
#   │  ─────────────────┼─────────────────────────────────┼────────────────────│
#   │  drift            │ x + cumsum(N(0,σ²)) · φ        │ Deriva termica     │
#   │  depth_dependent  │ x + N(0, σ·(1+α·idx/L))        │ Atenuacao com z    │
#   │  spikes           │ x + Bernoulli(p)·N(0,5σ)       │ Interferencia EM   │
#   │  pink             │ FFT→1/f→IFFT (σ scaled)         │ Flicker eletronico │
#   │  saturation       │ clip(x, P0.5, P99.5)            │ Sobrecarga ADC     │
#   │                                                                          │
#   │  σ = noise_level_var, φ = decay factor (0.95), α = depth scale          │
#   │  Todos preservam z_obs (col 0) e operam on-the-fly via tf.data.map     │
#   └──────────────────────────────────────────────────────────────────────────┘
#
# Ref: docs/reference/noise_catalog.md secao 2 (CORE).
# ──────────────────────────────────────────────────────────────────────────


def _add_drift_noise_tf(x, noise_level, n_protected=1):
    """Noise de deriva termica: cumsum de incrementos aleatorios.

    Modela a deriva lenta da eletronica do receptor LWD causada por
    variacao de temperatura/pressao no fundo do poco (>150°C).
    O cumsum gera uma tendencia temporalmente correlacionada que
    cresce monotonicamente (random walk), simulando calibracao
    que se degrada ao longo do perfil de medicao.

    Formula: x + cumsum(N(0, σ²)) × φ
    Onde φ = 0.95 (fator de escala global — atenua ligeiramente
    a amplitude total do drift). Amplitude tipica em seq_len=600:
    ~0.95 × sqrt(600) × σ ≈ 23σ. Para σ=0.05, drift ≈ ±1.2 A/m.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar com sigma base.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com drift acumulado nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["drift"].
        Ref: noise_catalog.md tipo #5 (drift), legado C12 phi=0.95.
        O drift eh um random walk escalado — amplitude cresce com
        sqrt(seq_len). Fator φ=0.95 reduz em 5% a amplitude total.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Incrementos aleatorios (random walk steps) ───────────────────
    increments = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level,
        dtype=tf.float32,
    )
    # ── Cumsum ao longo do eixo temporal (axis=1) ────────────────────
    # Produz tendencia temporalmente correlacionada (drift).
    # Fator de decaimento phi evita crescimento ilimitado.
    drift = tf.cumsum(increments, axis=1) * 0.95
    return tf.concat([protected, em_feats + drift], axis=-1)


def _add_depth_dependent_noise_tf(x, noise_level, n_protected=1):
    """Noise dependente de profundidade: sigma cresce com indice temporal.

    Modela a atenuacao EM com profundidade (skin depth): quanto mais
    fundo no poco, maior a atenuacao do sinal EM e menor o SNR.
    O noise eh aditivo gaussiano com sigma variavel:
    σ(z) = σ_base × (1 + α × z/seq_len)

    Onde α = 1.0 (sigma dobra do topo ao fundo do perfil).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar com sigma base.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise crescente em profundidade.

    Note:
        Referenciado em: NOISE_FN_MAP["depth_dependent"].
        Ref: noise_catalog.md tipo #8, Errata skin depth secao 4.6.
        alpha=1.0: no fundo do perfil, sigma = 2 × sigma_base.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    seq_len = tf.cast(tf.shape(em_feats)[1], tf.float32)
    # ── Fator de escala: cresce linearmente de 1.0 a 2.0 ────────────
    # shape: (1, seq_len, 1) — broadcasta em batch e features
    indices = tf.cast(tf.range(tf.shape(em_feats)[1]), tf.float32)
    scale = 1.0 + indices / tf.maximum(seq_len, 1.0)
    scale = tf.reshape(scale, (1, -1, 1))  # (1, seq_len, 1)
    noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level,
        dtype=tf.float32,
    )
    return tf.concat([protected, em_feats + noise * scale], axis=-1)


def _add_spikes_noise_tf(x, noise_level, n_protected=1):
    """Noise de spikes: outliers transitorios por interferencia EM.

    Modela picos impulsivos causados por descargas eletrostaticas,
    EMI de motores do rig, ou glitches na telemetria. Picos sao raros
    (p=0.1%) mas de alta magnitude (5× sigma_base).

    Formula: x + Bernoulli(0.001) × N(0, 5σ)

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar com sigma base.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com spikes esporadicos nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["spikes"].
        Ref: noise_catalog.md tipo #21 (spikes).
        Probabilidade 0.001 e magnitude 5σ sao defaults do legado C12.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Mascara de Bernoulli: p=0.001 de spike ──────────────────────
    mask = tf.cast(
        tf.random.uniform(shape=tf.shape(em_feats)) < 0.001,
        dtype=tf.float32,
    )
    # ── Magnitude do spike: 5× sigma_base ───────────────────────────
    spike_values = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level * 5.0,
        dtype=tf.float32,
    )
    return tf.concat([protected, em_feats + mask * spike_values], axis=-1)


def _add_pink_noise_tf(x, noise_level, n_protected=1):
    """Noise rosa (1/f): espectro decai como 1/frequencia.

    Modela flicker noise eletronico presente em todos os receptores
    analogicos LWD. O espectro 1/f domina nas baixas frequencias,
    criando flutuacoes lentas de baseline que sao dificeis de
    distinguir de verdadeiras variacoes de resistividade.

    Implementacao: ruido branco modulado por envelope 1/f no
    dominio do tempo (aproximacao eficiente sem FFT no grafo TF).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar com sigma base.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise 1/f nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["pink"].
        Ref: noise_catalog.md tipo #9 (pink).
        Aproximacao temporal: ruido branco filtrado por media movel
        ponderada (eficiente para tf.data.map, sem tf.signal.fft).
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Ruido branco base ────────────────────────────────────────────
    white = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level,
        dtype=tf.float32,
    )
    # ── Aproximacao 1/f: cumsum + decaimento (low-pass) ──────────────
    # cumsum produz espectro ~1/f² (brownian); media com branco → 1/f
    brownian = tf.cumsum(white, axis=1)
    # Normalizar brownian para mesma escala que white
    std_b = tf.math.reduce_std(brownian, axis=1, keepdims=True)
    std_b = tf.maximum(std_b, 1e-12)
    brownian_norm = brownian / std_b * noise_level
    # Mix 50/50 branco + browniano ≈ espectro 1/f
    pink = 0.5 * white + 0.5 * brownian_norm
    return tf.concat([protected, em_feats + pink], axis=-1)


def _saturation_clip(em_feats, noise_level):
    """Helper: clipping per-sequence para saturation (chamado apenas se σ>0)."""
    import tensorflow as tf

    mean = tf.reduce_mean(em_feats, axis=1, keepdims=True)
    std = tf.math.reduce_std(em_feats, axis=1, keepdims=True)
    std = tf.maximum(std, 1e-12)
    n_sigma = tf.maximum(3.0 - noise_level * 20.0, 1.5)
    lo = mean - n_sigma * std
    hi = mean + n_sigma * std
    return tf.clip_by_value(em_feats, lo, hi)


def _add_saturation_noise_tf(x, noise_level, n_protected=1):
    """Noise de saturacao: clipping em percentil do ADC.

    Modela a sobrecarga do receptor quando o sinal EM excede a
    faixa dinamica do conversor analogico-digital (ADC). Valores
    acima/abaixo do percentil 99.5/0.5 sao clipados.

    O noise_level controla a agressividade do clipping:
    percentil = 100 - noise_level × 50 (0.05 → P97.5).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar (controla clipping).
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com valores extremos clipados.

    Note:
        Referenciado em: NOISE_FN_MAP["saturation"].
        Ref: noise_catalog.md tipo #7 (saturation).
        noise_level=0.05 → clip no percentil 97.5/2.5.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Guard: noise_level=0 → retorna x inalterado ─────────────────
    # Saturation sempre clipa; sem guard, seria ativo na fase clean
    # do curriculum quando noise_level_var=0.0.
    em_clipped = tf.cond(
        noise_level <= 0.0,
        lambda: em_feats,
        lambda: _saturation_clip(em_feats, noise_level),
    )
    return tf.concat([protected, em_clipped], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCOES DE NOISE GEOFISICO LWD — 6 TIPOS (R1-R6)
# ════════════════════════════════════════════════════════════════════════════
# Efeitos geofisicos especializados que ocorrem especificamente no
# ambiente de perfuracao LWD. Maior delta entre dados sinteticos e
# dados de campo reais.
#
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  6 Tipos Geofisicos LWD (R1-R6):                                       │
#   │                                                                          │
#   │  ID │ Tipo                      │ Efeito Fisico                         │
#   │  ───┼───────────────────────────┼───────────────────────────────────────│
#   │  R1 │ shoulder_bed              │ Leakage H de camadas adjacentes       │
#   │  R2 │ borehole_effect           │ Rugosidade/washout do poco            │
#   │  R3 │ mud_invasion              │ Filtrado de lama altera rho aparente  │
#   │  R4 │ anisotropy_misalignment   │ Eixo ferramenta ≠ eixo TIV           │
#   │  R5 │ formation_heterogeneity   │ Variabilidade intra-camada            │
#   │  R6 │ telemetry                 │ Erro MWD: dropout + bit error         │
#   │                                                                          │
#   │  ATENCAO: R5 eh o UNICO tipo que perturba targets (rho_h, rho_v).     │
#   │  No pipeline v2.0, perturbacao de targets eh aplicada separadamente    │
#   │  via config.use_target_perturbation. Aqui, R5 opera apenas nas         │
#   │  features EM (como todos os outros tipos neste modulo).                │
#   └──────────────────────────────────────────────────────────────────────────┘
#
# Ref: docs/reference/noise_catalog.md secao 5 (R1-R6).
# ──────────────────────────────────────────────────────────────────────────


def _add_shoulder_bed_noise_tf(x, noise_level, n_protected=1):
    """R1 — Shoulder bed effect: leakage H de camadas adjacentes.

    Em camadas finas (<1 metro), a medida EM sofre influencia das
    camadas adjacentes (shoulder beds). Isso manifesta como uma
    suavizacao/borrando das transicoes abruptas de resistividade.

    Implementacao: media movel de janela 3 + noise aditivo, simulando
    o efeito de resolucao vertical finita da ferramenta LWD.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com suavizacao shoulder bed + noise.

    Note:
        Referenciado em: NOISE_FN_MAP["shoulder_bed"].
        Ref: noise_catalog.md R1. Wang et al. (2018).
        Kernel 3 ≈ 3 medicoes ≈ 3 metros de resolucao vertical.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Media movel kernel=3 (resolucao vertical da ferramenta) ──────
    # Simula resposta da ferramenta que integra sobre ~3 metros.
    # Pad com replicate nas bordas para preservar shape.
    padded = tf.pad(em_feats, [[0, 0], [1, 1], [0, 0]], mode="REFLECT")
    smoothed = (padded[:, :-2, :] + padded[:, 1:-1, :] + padded[:, 2:, :]) / 3.0
    # ── Mix: (1-σ)×original + σ×smoothed + noise ────────────────────
    blend = (1.0 - noise_level) * em_feats + noise_level * smoothed
    noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level * 0.5,
        dtype=tf.float32,
    )
    return tf.concat([protected, blend + noise], axis=-1)


def _add_borehole_effect_noise_tf(x, noise_level, n_protected=1):
    """R2 — Borehole effect: distorcao pelo fluido e rugosidade do poco.

    A rugosidade do poco (washout) e o fluido de perfuracao (lama)
    modificam o campo EM medido. O efeito eh proporcional ao
    diametro do poco — quanto maior o washout, maior a distorcao.

    Implementacao: noise multiplicativo + aditivo combinado,
    simulando variacao de acoplamento ferramenta-formacao.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com efeito borehole combinado.

    Note:
        Referenciado em: NOISE_FN_MAP["borehole_effect"].
        Ref: noise_catalog.md R2. Constable et al. (2016).
        Combina efeito multiplicativo (ganho) + aditivo (offset).
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Efeito multiplicativo (variacao de ganho) ────────────────────
    gain_noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level * 0.5,
        dtype=tf.float32,
    )
    # ── Efeito aditivo (offset por acoplamento) ─────────────────────
    offset_noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level * 0.5,
        dtype=tf.float32,
    )
    result = em_feats * (1.0 + gain_noise) + offset_noise
    return tf.concat([protected, result], axis=-1)


def _add_mud_invasion_noise_tf(x, noise_level, n_protected=1):
    """R3 — Mud invasion: filtrado de lama altera resistividade aparente.

    O filtrado de lama invade a formacao proxima ao poco, criando
    uma zona invadida com resistividade diferente da formacao virgem.
    O efeito eh um fator multiplicativo que atenua o sinal EM.

    Implementacao: fator multiplicativo = 1 - noise_level × U(0,1),
    simulando atenuacao variavel por invasao.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com atenuacao por invasao.

    Note:
        Referenciado em: NOISE_FN_MAP["mud_invasion"].
        Ref: noise_catalog.md R3. Fator default 0.8 no legado.
        noise_level=0.05 → atenuacao de 0-5% (conservador).
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Fator de atenuacao uniforme por amostra ─────────────────────
    # Cada posicao temporal tem atenuacao ligeiramente diferente
    # (invasao nao eh uniforme ao longo do poco).
    attenuation = 1.0 - noise_level * tf.random.uniform(
        shape=tf.shape(em_feats),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )
    return tf.concat([protected, em_feats * attenuation], axis=-1)


def _add_anisotropy_misalignment_noise_tf(x, noise_level, n_protected=1):
    """R4 — Anisotropy misalignment: eixo da ferramenta ≠ eixo TIV.

    Quando a ferramenta nao esta perfeitamente alinhada com o eixo
    de simetria TIV da formacao, as componentes Hxx e Hzz se misturam.
    O efeito eh uma rotacao parcial no espaco das componentes EM.

    Implementacao: mistura cruzada entre pares de colunas EM.
    delta = noise_level controla a magnitude da mistura.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com mistura cruzada Hxx↔Hzz.

    Note:
        Referenciado em: NOISE_FN_MAP["anisotropy_misalignment"].
        Ref: noise_catalog.md R4. delta_deg=2° no legado.
        noise_level=0.05 → ~3° de desalinhamento.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Mistura cruzada: cada coluna recebe epsilon de vizinhas ──────
    # Aproximacao linearizada de rotacao: col_i += δ × col_((i+2)%n)
    # Para 4 EM cols: Re(Hxx)↔Re(Hzz), Im(Hxx)↔Im(Hzz)
    # Para n_em < 4, shift=2 pode ser no-op (shift mod n_em = 0).
    # Nesse caso, adiciona noise gaussiano como fallback seguro.
    n_em = tf.shape(em_feats)[2]
    shift = tf.roll(em_feats, shift=2, axis=2)
    # ── Fallback: se n_em < 4, roll nao mistura — adicionar gaussian ─
    mixed = tf.cond(
        n_em >= 4,
        lambda: em_feats + noise_level * shift,
        lambda: em_feats
        + tf.random.normal(
            shape=tf.shape(em_feats),
            mean=0.0,
            stddev=noise_level,
            dtype=tf.float32,
        ),
    )
    return tf.concat([protected, mixed], axis=-1)


def _add_formation_heterogeneity_noise_tf(x, noise_level, n_protected=1):
    """R5 — Formation heterogeneity: variabilidade intra-camada.

    Modela a variabilidade natural de resistividade dentro de uma
    mesma camada geologica. Mesmo em formacoes "homogeneas", existe
    variabilidade de 5-15% causada por variacao de porosidade,
    cimentacao, saturacao de fluidos.

    NOTA: R5 eh o UNICO tipo que no legado tambem perturbava targets
    (rho_h, rho_v). No pipeline v2.0, este modulo opera APENAS nas
    features EM. Perturbacao de targets usa config.use_target_perturbation.

    Implementacao: noise multiplicativo com autocorrelacao espacial
    (low-pass filtered), simulando variacao gradual de propriedades.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com variabilidade de formacao nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["formation_heterogeneity"].
        Ref: noise_catalog.md R5.
        ATENCAO: aqui opera APENAS em features EM.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Variacao multiplicativa com autocorrelacao ───────────────────
    # Ruido branco → cumsum → normalizar → fator multiplicativo
    white = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level,
        dtype=tf.float32,
    )
    # Low-pass via cumsum normalizado (autocorrelacao)
    smooth = tf.cumsum(white, axis=1)
    std_s = tf.math.reduce_std(smooth, axis=1, keepdims=True)
    std_s = tf.maximum(std_s, 1e-12)
    smooth_norm = smooth / std_s * noise_level
    # Fator multiplicativo: 1 + variacao
    factor = 1.0 + smooth_norm
    return tf.concat([protected, em_feats * factor], axis=-1)


def _add_telemetry_noise_tf(x, noise_level, n_protected=1):
    """R6 — Telemetry noise: erros de transmissao MWD/LWD.

    O sistema de telemetria mud-pulse transmite dados do fundo do
    poco para a superficie. Erros incluem:
    - Dropouts (perda completa de pacote): p = noise_level × 0.02
    - Bit errors (corrupcao): ruido gaussiano residual

    Implementacao: combinacao de dropout esparso + noise aditivo leve.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com erros de telemetria.

    Note:
        Referenciado em: NOISE_FN_MAP["telemetry"].
        Ref: noise_catalog.md R6.
        Dropout 0.1% + BER 1e-4 sao valores tipicos LWD.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Dropout de telemetria: pacotes inteiros zerados ──────────────
    # Probabilidade de perda proporcional a noise_level
    drop_prob = noise_level * 0.02
    keep = tf.cast(
        tf.random.uniform(shape=(tf.shape(em_feats)[0], tf.shape(em_feats)[1], 1))
        >= drop_prob,
        dtype=tf.float32,
    )
    em_dropped = em_feats * keep
    # ── Bit errors: ruido gaussiano residual (1/10 do sigma) ────────
    bit_noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level * 0.1,
        dtype=tf.float32,
    )
    return tf.concat([protected, em_dropped + bit_noise], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: REGISTRO DE FUNCOES (NOISE_FN_MAP)
# ════════════════════════════════════════════════════════════════════════════
# Dicionario tipo→funcao para extensibilidade. Novos tipos de noise
# podem ser adicionados registrando aqui sem alterar o dispatcher.
# 15 tipos totais: 4 originais + 5 CORE + 6 LWD (R1-R6).
# Ref: Factory Pattern (CLAUDE.md secao Code Patterns).
# ──────────────────────────────────────────────────────────────────────────

NOISE_FN_MAP: Dict[str, NoiseFnType] = {
    # ── 4 tipos originais (v2.0.0) ──────────────────────────────────
    "gaussian": _add_gaussian_noise_tf,
    "multiplicative": _add_multiplicative_noise_tf,
    "uniform": _add_uniform_noise_tf,
    "dropout": _add_dropout_noise_tf,
    # ── 5 CORE adicionais (Fase II) ─────────────────────────────────
    "drift": _add_drift_noise_tf,
    "depth_dependent": _add_depth_dependent_noise_tf,
    "spikes": _add_spikes_noise_tf,
    "pink": _add_pink_noise_tf,
    "saturation": _add_saturation_noise_tf,
    # ── 6 Geofisicos LWD R1-R6 (Fase II) ────────────────────────────
    "shoulder_bed": _add_shoulder_bed_noise_tf,
    "borehole_effect": _add_borehole_effect_noise_tf,
    "mud_invasion": _add_mud_invasion_noise_tf,
    "anisotropy_misalignment": _add_anisotropy_misalignment_noise_tf,
    "formation_heterogeneity": _add_formation_heterogeneity_noise_tf,
    "telemetry": _add_telemetry_noise_tf,
}

# Tipos validos — consultado por PipelineConfig e testes.
VALID_NOISE_TYPES = frozenset(NOISE_FN_MAP.keys())


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FACTORY PARA TF.VARIABLE
# ════════════════════════════════════════════════════════════════════════════
# Cria o tf.Variable compartilhado entre pipeline (tf.data.map)
# e callback (UpdateNoiseLevelCallback). Este variable eh a ponte
# entre o curriculum schedule e a injecao de noise no grafo TF.
# Ref: Skill geosteering-v2 secao 7.3 (curriculum learning).
# ──────────────────────────────────────────────────────────────────────────


def create_noise_level_var(
    initial_value: float = 0.0,
    name: str = "noise_level",
) -> "tf.Variable":
    """Cria tf.Variable escalar para nivel de noise compartilhado.

    Este variable eh a ponte central entre o curriculum schedule
    (UpdateNoiseLevelCallback) e o pipeline de noise (tf.data.map).
    Deve ser criado UMA VEZ e passado para ambos.

    Args:
        initial_value: Valor inicial do noise level. Default: 0.0
            (fase clean do curriculum). Para noise constante sem
            curriculum, usar config.noise_level_max.
        name: Nome do variable no grafo TF.

    Returns:
        tf.Variable: Escalar float32, trainable=False.

    Example:
        >>> from geosteering_ai.noise import create_noise_level_var
        >>> noise_var = create_noise_level_var(0.0)
        >>> noise_var.assign(0.05)  # atualizado pelo callback

    Note:
        Referenciado em:
            - training/loop.py (cria variable)
            - training/callbacks.py (UpdateNoiseLevelCallback atualiza)
            - data/pipeline.py (build_train_map_fn le)
        trainable=False: NAO participa do backprop.
        Fase clean do curriculum comeca com 0.0.
    """
    import tensorflow as tf

    return tf.Variable(
        initial_value,
        dtype=tf.float32,
        trainable=False,
        name=name,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECAO: DISPATCHER TF (MULTI-TIPO)
# ════════════════════════════════════════════════════════════════════════════
# Aplica mix de N tipos de noise com pesos normalizados.
# Chamado dentro de tf.data.map para noise on-the-fly.
# Suporta config.noise_types=["gaussian"] (simples) ou
# config.noise_types=["gaussian","multiplicative"] (mixto).
# Ref: Cadeia on-the-fly: raw → noise → FV → GS → scale.
# ──────────────────────────────────────────────────────────────────────────


def apply_noise_tf(
    x: "tf.Tensor",
    noise_level_var: "tf.Variable",
    noise_types: List[str],
    noise_weights: List[float],
    n_protected: int = 1,
) -> "tf.Tensor":
    """Aplica composicao sequencial de noise types no tensor de features.

    Dispatcher que resolve cada tipo contra NOISE_FN_MAP e aplica
    sequencialmente (cada tipo opera sobre o resultado do anterior).
    Pesos controlam o noise_level efetivo de cada tipo.
    Se noise_level_var == 0 (fase clean), retorna x inalterado.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat). Layout P1:
            [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz), ...].
            Layout P2+P3: [theta_norm?, f_norm?, z_obs, Re(Hxx), ...].
        noise_level_var: tf.Variable escalar com sigma atual.
        noise_types: Lista de tipos. Ex: ["gaussian", "multiplicative"].
        noise_weights: Pesos correspondentes. Ex: [0.8, 0.2].
            Pesos controlam noise_level efetivo de cada tipo:
            noise_level_efetivo = noise_level × (w_i / w_max).
        n_protected: Colunas iniciais protegidas (parametros conhecidos
            + z_obs). Default 1 (P1). P2: 2, P3: 2, P2+P3: 3.
            Calculado como config.n_prefix + 1.

    Returns:
        tf.Tensor: x com noise composto aplicado sequencialmente.

    Raises:
        ValueError: Se algum tipo nao esta em NOISE_FN_MAP.

    Example:
        >>> from geosteering_ai.noise import apply_noise_tf, create_noise_level_var
        >>> noise_var = create_noise_level_var(0.05)
        >>> x_noisy = apply_noise_tf(x, noise_var, ["gaussian"], [1.0])
        >>> # P2+P3: proteger 3 colunas (theta, freq, z)
        >>> x_noisy = apply_noise_tf(x, noise_var, ["gaussian"], [1.0], n_protected=3)

    Note:
        Referenciado em:
            - data/pipeline.py: build_train_map_fn() (Step 1 do map_fn)
        Composicao sequencial: gaussiano + multiplicativo = primeiro
        adiciona, depois escala. Fisicamente correto vs media ponderada.
        Fase clean: noise_level_var=0.0 → todas funcoes produzem x+0 ≈ x.
    """
    import tensorflow as tf

    # ── Validar tipos ─────────────────────────────────────────────────
    for nt in noise_types:
        if nt not in NOISE_FN_MAP:
            raise ValueError(
                f"Noise type '{nt}' desconhecido. "
                f"Validos: {sorted(NOISE_FN_MAP.keys())}"
            )

    # ── Tipo unico (otimizacao — evita loop e concat) ────────────────
    if len(noise_types) == 1:
        return NOISE_FN_MAP[noise_types[0]](x, noise_level_var, n_protected)

    # ── Composicao sequencial de N tipos ──────────────────────────────
    # Fisicamente correto: cada noise type opera sobre o resultado do
    # anterior. Pesos controlam noise_level efetivo de cada tipo:
    #   noise_level_efetivo = noise_level × (w_i / w_max)
    # Composicao sequencial preserva semantica de cada tipo:
    #   gaussiano + multiplicativo = primeiro adiciona, depois escala.
    w_max = max(noise_weights)
    result = x
    for nt, w in zip(noise_types, noise_weights):
        # Escalar noise_level pelo peso normalizado relativo ao maximo
        scaled_level = noise_level_var * tf.constant(w / w_max, dtype=tf.float32)
        result = NOISE_FN_MAP[nt](result, scaled_level, n_protected)

    return result


# ════════════════════════════════════════════════════════════════════════════
# SECAO: VERSAO NUMPY (OFFLINE / TESTES)
# ════════════════════════════════════════════════════════════════════════════
# Versao numpy de apply_raw_em_noise para uso em testes, EDA,
# e cenarios offline. Semantica identica a versao TF para tipo unico.
# NOTA: para multi-tipo, numpy usa blending ponderado enquanto TF usa
# composicao sequencial — distribuicoes diferem levemente nesse caso.
# noise aplicado APENAS nas colunas EM (indices n_protected:), z preservado.
# NAO usar em produção com FV/GS ativos (usar on-the-fly TF).
# ──────────────────────────────────────────────────────────────────────────


def apply_raw_em_noise(
    x: np.ndarray,
    noise_level: float = 0.05,
    noise_types: Optional[List[str]] = None,
    noise_weights: Optional[List[float]] = None,
    seed: Optional[int] = None,
    n_protected: int = 1,
) -> np.ndarray:
    """Aplica noise nas colunas EM de um array numpy.

    Versao offline/testes. Aplica noise aditivo gaussiano (default)
    ou mix de tipos nas colunas EM, preservando as primeiras
    ``n_protected`` colunas intactas (parametros conhecidos + z_obs).

    Para uso em producao com FV/GS ativos, usar a versao TF
    via apply_noise_tf dentro de tf.data.map (on-the-fly).

    Args:
        x: Array 3D (n_seq, seq_len, n_feat) com raw features.
            Layout P1: [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)].
            Layout P2+P3: [theta_norm, f_norm, z_obs, Re(Hxx), ...].
        noise_level: Desvio padrao do noise. Default: 0.05.
        noise_types: Lista de tipos. Default: ["gaussian"].
        noise_weights: Pesos correspondentes. Default: [1.0].
        seed: Seed para reproducibilidade. None = aleatorio.
        n_protected: Numero de colunas iniciais protegidas do noise.
            Default: 1 (apenas z_obs para P1 baseline).
            P2: 2 (theta_norm + z_obs).
            P3: 2 (f_norm + z_obs).
            P2+P3: 3 (theta_norm + f_norm + z_obs).
            Regra: n_protected = config.n_prefix + 1.

    Returns:
        np.ndarray: Copia de x com noise nas colunas EM (indices n_protected:).

    Raises:
        ValueError: Se x.ndim != 3 ou tipo desconhecido.

    Example:
        >>> import numpy as np
        >>> from geosteering_ai.noise import apply_raw_em_noise
        >>> x = np.random.randn(10, 600, 5).astype(np.float32)
        >>> x_noisy = apply_raw_em_noise(x, noise_level=0.05)
        >>> np.testing.assert_array_equal(x_noisy[:,:,0], x[:,:,0])  # z preservado
        >>> # P2+P3: proteger 3 colunas (theta, freq, z)
        >>> x7 = np.random.randn(10, 600, 7).astype(np.float32)
        >>> x7_noisy = apply_raw_em_noise(x7, noise_level=0.05, n_protected=3)

    Note:
        Referenciado em:
            - tests/test_noise.py (validacao offline)
            - visualization/holdout.py (plots clean vs noisy)
            - tests/test_data_pipeline.py: TestThetaFreqInjection
        AVISO: NAO usar em producao com FV/GS ativos.
        Cadeia correta eh on-the-fly: raw → noise_tf → FV_tf → GS_tf → scale.
        v2.0.1: Adicionado n_protected para suportar P2/P3 (theta/freq prefixo).
    """
    if x.ndim != 3:
        raise ValueError(
            f"x deve ser 3D (n_seq, seq_len, n_feat), recebido ndim={x.ndim}"
        )

    if noise_types is None:
        noise_types = ["gaussian"]
    if noise_weights is None:
        noise_weights = [1.0]

    rng = np.random.RandomState(seed)
    result = x.copy()

    # ── Separar colunas protegidas das features EM ───────────────────
    # n_protected=1 (P1): col 0 = z_obs protegida, cols 1: = EM ruidoso
    # n_protected=2 (P2): cols 0-1 = theta+z protegidas, cols 2: = EM
    # n_protected=3 (P2+P3): cols 0-2 = theta+freq+z, cols 3: = EM
    # Colunas protegidas NAO recebem noise (parametros conhecidos).
    em_feats = result[:, :, n_protected:]  # vista — modifica in-place
    w_sum = sum(noise_weights)

    for nt, w in zip(noise_types, noise_weights):
        w_norm = w / w_sum

        if nt == "gaussian":
            # ── Gaussiano aditivo: N(0, sigma^2) ─────────────────────
            noise = rng.normal(0.0, noise_level, size=em_feats.shape)
            em_feats += w_norm * noise.astype(em_feats.dtype)

        elif nt == "multiplicative":
            # ── Multiplicativo (speckle): x * (1 + N(0, sigma^2)) ────
            noise = rng.normal(0.0, noise_level, size=em_feats.shape)
            em_feats[:] = (1.0 - w_norm) * em_feats + w_norm * em_feats * (
                1.0 + noise.astype(em_feats.dtype)
            )

        elif nt == "uniform":
            # ── Uniforme: U(-sigma, sigma) ────────────────────────────
            noise = rng.uniform(-noise_level, noise_level, size=em_feats.shape)
            em_feats += w_norm * noise.astype(em_feats.dtype)

        elif nt == "dropout":
            # ── Dropout: mascara aleatoria ────────────────────────────
            keep_prob = max(1.0 - noise_level, 1e-6)
            mask = (rng.uniform(size=em_feats.shape) < keep_prob).astype(em_feats.dtype)
            em_feats[:] = (1.0 - w_norm) * em_feats + w_norm * em_feats * mask / keep_prob

        # ── 5 CORE adicionais (Fase II) ─────────────────────────────
        elif nt == "drift":
            # ── Deriva termica: cumsum de incrementos × decay ────────
            increments = rng.normal(0.0, noise_level, size=em_feats.shape)
            drift = np.cumsum(increments, axis=1) * 0.95
            em_feats += w_norm * drift.astype(em_feats.dtype)

        elif nt == "depth_dependent":
            # ── Noise crescente com profundidade: σ(z) = σ×(1+z/L) ──
            seq_len_local = em_feats.shape[1]
            scale = 1.0 + np.arange(seq_len_local).reshape(1, -1, 1) / max(
                seq_len_local, 1
            )
            noise = rng.normal(0.0, noise_level, size=em_feats.shape)
            em_feats += w_norm * (noise * scale).astype(em_feats.dtype)

        elif nt == "spikes":
            # ── Outliers transitorios: Bernoulli(p=0.001) × 5σ ──────
            mask = (rng.uniform(size=em_feats.shape) < 0.001).astype(em_feats.dtype)
            spike_values = rng.normal(0.0, noise_level * 5.0, size=em_feats.shape)
            em_feats += w_norm * (mask * spike_values).astype(em_feats.dtype)

        elif nt == "pink":
            # ── Flicker noise 1/f: mix branco + browniano ────────────
            white = rng.normal(0.0, noise_level, size=em_feats.shape)
            brownian = np.cumsum(white, axis=1)
            std_b = np.std(brownian, axis=1, keepdims=True)
            std_b = np.maximum(std_b, 1e-12)
            brownian_norm = brownian / std_b * noise_level
            pink = 0.5 * white + 0.5 * brownian_norm
            em_feats += w_norm * pink.astype(em_feats.dtype)

        elif nt == "saturation":
            # ── Clipping ADC: clip em n_sigma do desvio padrao ───────
            mean_val = np.mean(em_feats, axis=1, keepdims=True)
            std_val = np.std(em_feats, axis=1, keepdims=True)
            std_val = np.maximum(std_val, 1e-12)
            n_sigma = max(3.0 - noise_level * 20.0, 1.5)
            lo = mean_val - n_sigma * std_val
            hi = mean_val + n_sigma * std_val
            em_feats[:] = np.clip(em_feats, lo, hi)

        # ── 6 Geofisicos LWD R1-R6 (Fase II) ───────────────────────
        elif nt == "shoulder_bed":
            # ── R1: media movel kernel=3 + noise aditivo leve ────────
            padded = np.pad(em_feats, ((0, 0), (1, 1), (0, 0)), mode="reflect")
            smoothed = (padded[:, :-2, :] + padded[:, 1:-1, :] + padded[:, 2:, :]) / 3.0
            blend = (1.0 - noise_level) * em_feats + noise_level * smoothed
            noise = rng.normal(0.0, noise_level * 0.5, size=em_feats.shape)
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * (blend + noise)).astype(
                em_feats.dtype
            )

        elif nt == "borehole_effect":
            # ── R2: multiplicativo (ganho) + aditivo (offset) ────────
            gain = rng.normal(0.0, noise_level * 0.5, size=em_feats.shape)
            offset = rng.normal(0.0, noise_level * 0.5, size=em_feats.shape)
            noisy = em_feats * (1.0 + gain) + offset
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * noisy).astype(
                em_feats.dtype
            )

        elif nt == "mud_invasion":
            # ── R3: atenuacao multiplicativa uniforme ────────────────
            atten = 1.0 - noise_level * rng.uniform(0, 1, size=em_feats.shape)
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * em_feats * atten).astype(
                em_feats.dtype
            )

        elif nt == "anisotropy_misalignment":
            # ── R4: mistura cruzada entre colunas EM ─────────────────
            shift = np.roll(em_feats, shift=2, axis=2)
            mixed = em_feats + noise_level * shift
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * mixed).astype(
                em_feats.dtype
            )

        elif nt == "formation_heterogeneity":
            # ── R5: variacao multiplicativa com autocorrelacao ────────
            white = rng.normal(0.0, noise_level, size=em_feats.shape)
            smooth = np.cumsum(white, axis=1)
            std_s = np.std(smooth, axis=1, keepdims=True)
            std_s = np.maximum(std_s, 1e-12)
            smooth_norm = smooth / std_s * noise_level
            factor = 1.0 + smooth_norm
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * em_feats * factor).astype(
                em_feats.dtype
            )

        elif nt == "telemetry":
            # ── R6: dropout esparso + bit error gaussiano ────────────
            drop_prob = noise_level * 0.02
            keep = (
                rng.uniform(size=(em_feats.shape[0], em_feats.shape[1], 1)) >= drop_prob
            ).astype(em_feats.dtype)
            bit_noise = rng.normal(0.0, noise_level * 0.1, size=em_feats.shape)
            noisy = em_feats * keep + bit_noise
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * noisy).astype(
                em_feats.dtype
            )

        else:
            raise ValueError(
                f"Noise type '{nt}' desconhecido. "
                f"Validos: {sorted(NOISE_FN_MAP.keys())}"
            )

    return result


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Registro ──────────────────────────────────────────────────────
    "NOISE_FN_MAP",
    "VALID_NOISE_TYPES",
    # ── Factory ───────────────────────────────────────────────────────
    "create_noise_level_var",
    # ── Dispatcher TF ─────────────────────────────────────────────────
    "apply_noise_tf",
    # ── Numpy (offline/testes) ────────────────────────────────────────
    "apply_raw_em_noise",
]
