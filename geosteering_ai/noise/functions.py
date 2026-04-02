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
# ║    • Funcoes TF de noise on-the-fly para tf.data.map (34 tipos)           ║
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
#   │  34 Tipos de Noise (TF on-the-fly):                                     │
#   │                                                                          │
#   │  GRUPO │ Tipo                     │ Formula                │ Fenomeno   │
#   │  ──────┼──────────────────────────┼────────────────────────┼────────────│
#   │  ORIG  │ gaussian                 │ x + N(0,σ²)           │ Elet.adit. │
#   │  ORIG  │ multiplicative           │ x·(1+N(0,σ²))         │ Ganho amp  │
#   │  ORIG  │ uniform                  │ x + U(-σ,σ)           │ Quant. ADC │
#   │  ORIG  │ dropout                  │ x·Bern(1-σ)/(1-σ)     │ Dropout    │
#   │  CORE  │ drift                    │ cumsum(N)·φ            │ Deriva T   │
#   │  CORE  │ depth_dependent          │ N(0,σ·(1+αz/L))       │ Aten. z    │
#   │  CORE  │ spikes                   │ Bern(p)·N(0,5σ)       │ EMI picos  │
#   │  CORE  │ pink                     │ 0.5W+0.5Brown          │ Flicker    │
#   │  CORE  │ saturation               │ clip(x,Plo,Phi)        │ ADC clip   │
#   │  CORE+ │ varying                  │ N·U(σ_min,σ_max)·|x|  │ Heterosced │
#   │  CORE+ │ gaussian_local           │ N(0,pct·|x|)          │ Calib.loc  │
#   │  CORE+ │ gaussian_global          │ N(0,pct·std_glob)     │ Calib.glob │
#   │  CORE+ │ speckle                  │ x·(1+N(0,σ²))         │ Ganho²     │
#   │  CORE+ │ quantization             │ round(x/q)·q          │ ADC bits   │
#   │  R1-R6 │ shoulder_bed             │ MA(3)+N               │ Shoulder   │
#   │  R1-R6 │ borehole_effect          │ gain+offset            │ Borehole   │
#   │  R1-R6 │ mud_invasion             │ x·(1-σ·U)             │ Invasion   │
#   │  R1-R6 │ anisotropy_misalignment  │ col_i+=δ·col_j        │ TIV axis   │
#   │  R1-R6 │ formation_heterogeneity  │ x·(1+smooth)          │ Hetero.    │
#   │  R1-R6 │ telemetry                │ drop+bit_err           │ MWD link   │
#   │  EXT   │ cross_talk               │ Re+=ε·Im, Im+=ε·Re    │ Coupling   │
#   │  EXT   │ orientation              │ cos(θ)c0+sin(θ)c1     │ Mandrel    │
#   │  EXT   │ emi_noise                │ Σ A·sin(2πkft)        │ EMI 60Hz   │
#   │  EXT   │ freq_dependent           │ N(0,σ·f^α)            │ Freq.dep.  │
#   │  EXT   │ noise_floor              │ N(0,floor)             │ Det.limit  │
#   │  EXT   │ proportional             │ N(0,0.03·|x|)·scale   │ Prop. 3%   │
#   │  EXT   │ reim_diff                │ Re:σ, Im:1.5σ          │ Im noisier │
#   │  EXT   │ component_diff           │ Hxx:1.0σ, Hzz:0.8σ   │ Antena Δ   │
#   │  EXT   │ gaussian_keras           │ N(0,σ)                │ Keras compat│
#   │  EXT   │ motion                   │ x·(1+A·sin(2πft/L))   │ BHA vib    │
#   │  EXT   │ thermal                  │ N(0,0.3σ)             │ Johnson-N  │
#   │  EXT   │ phase_shift              │ rot(Re,Im,φ)          │ Demod err  │
#   │  R7    │ bha_vibration            │ shift·sin(2πt/L+φ)    │ Lateral    │
#   │  R8    │ eccentricity             │ x·(1+ecc·cos(2πt/L))  │ Ecc. tool  │
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
    # ── Bit errors: ruído gaussiano residual (1/10 do sigma) ────────
    # Mascarado por keep: amostras dropadas ficam ZERADAS (pacote perdido
    # real — downstream model deve aprender a lidar com zeros).
    # Sem máscara, bit noise preencheria gaps com ruído aleatório,
    # comprometendo a semântica física de perda de telemetria.
    bit_noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level * 0.1,
        dtype=tf.float32,
    )
    return tf.concat([protected, em_dropped + bit_noise * keep], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCOES DE NOISE CORE MISSING — 5 TIPOS
# ════════════════════════════════════════════════════════════════════════════
# 5 tipos que completam o catalogo CORE de noise encontrado em 100% dos
# sistemas de aquisicao LWD. Cada tipo modela um aspecto distinto do
# erro de medicao EM que nao eh capturado pelos 9 tipos anteriores.
#
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  5 Tipos CORE Missing:                                                  │
#   │                                                                          │
#   │  Tipo             │ Formula                         │ Fenomeno Fisico    │
#   │  ─────────────────┼─────────────────────────────────┼────────────────────│
#   │  varying          │ x + N·U(σ_min,σ_max)·|x|       │ Heteroscedastico  │
#   │  gaussian_local   │ x + N(0,pct·|x|)               │ Erro calib. local │
#   │  gaussian_global  │ x + N(0,pct·std_global)         │ Erro calib. global│
#   │  speckle          │ x·(1+N(0,σ²))                  │ Variacao ganho    │
#   │  quantization     │ round(x/q)·q                   │ Resolucao ADC     │
#   │                                                                          │
#   │  Adicionados ANTES da secao LWD R1-R6 por afinidade funcional.         │
#   └──────────────────────────────────────────────────────────────────────────┘
#
# Ref: docs/reference/noise_catalog.md secao 3 (CORE missing).
# ──────────────────────────────────────────────────────────────────────────


def _add_varying_noise_tf(x, noise_level, n_protected=1):
    """Noise heteroscedastico: incerteza dependente do sinal.

    Modela a incerteza de medicao que varia com a magnitude do sinal.
    Em receptores LWD reais, o SNR nao eh constante — sinais fracos
    (alta resistividade, grande spacing) tem erro relativo maior que
    sinais fortes. O sigma de cada amostra eh amostrado de U(σ_min, σ_max)
    e multiplicado por |x|, criando noise proporcional ao sinal.

    Formula: x + N(0,1) × U(σ_min, σ_max) × |x|
    Onde σ_min = 0.01 × noise_level, σ_max = noise_level.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar com sigma base.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise heteroscedastico nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["varying"].
        Ref: noise_catalog.md tipo varying.
        σ_min=0.01×σ garante noise minimo mesmo em sinais fortes.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Sigma variavel por amostra: U(σ_min, σ_max) ─────────────────
    # Cada posicao (batch, seq, feat) recebe sigma diferente,
    # simulando incerteza nao-estacionaria do receptor LWD.
    sigma_min = 0.01 * noise_level
    sigma_max = noise_level
    sigma_local = tf.random.uniform(
        shape=tf.shape(em_feats),
        minval=sigma_min,
        maxval=tf.maximum(sigma_max, sigma_min + 1e-12),
        dtype=tf.float32,
    )
    noise = tf.random.normal(shape=tf.shape(em_feats), dtype=tf.float32)
    noisy_em = em_feats + noise * sigma_local * tf.abs(em_feats)
    return tf.concat([protected, noisy_em], axis=-1)


def _add_gaussian_local_noise_tf(x, noise_level, n_protected=1):
    """Noise gaussiano local: erro proporcional ao sinal local.

    Modela erro de calibracao local do receptor LWD onde a incerteza
    de cada medicao eh proporcional a magnitude absoluta daquela
    medicao especifica. Comum em amplificadores com ganho variavel.

    Formula: x + N(0, pct × |x|) onde pct = noise_level.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar (pct do sinal).
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise proporcional ao sinal local.

    Note:
        Referenciado em: NOISE_FN_MAP["gaussian_local"].
        Ref: noise_catalog.md tipo gaussian_local.
        pct=0.05 → erro de ~5% da magnitude local.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Noise proporcional a |x| local ──────────────────────────────
    # stddev = pct × |x|, onde pct = noise_level (tipicamente 0.05).
    stddev = noise_level * tf.abs(em_feats)
    noise = tf.random.normal(shape=tf.shape(em_feats), dtype=tf.float32)
    noisy_em = em_feats + noise * stddev
    return tf.concat([protected, noisy_em], axis=-1)


def _add_gaussian_global_noise_tf(x, noise_level, n_protected=1):
    """Noise gaussiano global: erro proporcional ao desvio padrao global.

    Modela erro de medicao do sistema LWD onde a incerteza eh constante
    e proporcional a dispersao global do sinal. Tipico de sistemas com
    calibracao fixa (auto-range desabilitado).

    Formula: x + N(0, pct × std_global) onde std_global = std(em_feats).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar (pct da dispersao global).
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise proporcional ao desvio padrao global.

    Note:
        Referenciado em: NOISE_FN_MAP["gaussian_global"].
        Ref: noise_catalog.md tipo gaussian_global.
        pct=0.05 → erro de ~5% do std global das features EM.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Desvio padrao global das features EM ────────────────────────
    # Calculado sobre todo o batch para representar a escala do sinal.
    std_global = tf.math.reduce_std(em_feats)
    std_global = tf.maximum(std_global, 1e-12)
    noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level * std_global,
        dtype=tf.float32,
    )
    noisy_em = em_feats + noise
    return tf.concat([protected, noisy_em], axis=-1)


def _add_speckle_noise_tf(x, noise_level, n_protected=1):
    """Noise speckle: variacao de ganho do amplificador com sigma².

    Modela flutuacoes de ganho do amplificador LWD causadas por
    variacao termica. Similar ao multiplicativo, mas com sigma
    ao quadrado para representar noise de ganho (gain noise) que
    escala quadraticamente com a instabilidade do sistema.

    Formula: x × (1 + N(0, noise_level²)).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com speckle noise (sigma quadrado) nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["speckle"].
        Ref: noise_catalog.md tipo speckle.
        sigma² para noise_level=0.05 → stddev=0.0025 (muito leve).
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Speckle: ganho multiplicativo com σ² ─────────────────────────
    # noise_level² produz efeito mais suave que multiplicative puro.
    noise = tf.random.normal(
        shape=tf.shape(em_feats),
        mean=0.0,
        stddev=noise_level * noise_level,
        dtype=tf.float32,
    )
    noisy_em = em_feats * (1.0 + noise)
    return tf.concat([protected, noisy_em], axis=-1)


def _add_quantization_noise_tf(x, noise_level, n_protected=1):
    """Noise de quantizacao: resolucao finita do ADC.

    Modela o erro de discretizacao do conversor analogico-digital (ADC)
    com resolucao finita. O sinal continuo eh arredondado para o
    multiplo mais proximo do quantum q, introduzindo erro uniforme
    de magnitude maxima q/2.

    Formula: round(x / q) × q, onde q = noise_level × 0.1.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar (controla tamanho do quantum).
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x quantizado nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["quantization"].
        Ref: noise_catalog.md tipo quantization.
        q = 0.005 para noise_level=0.05 (resolucao de ~12 bits em range ±10).
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Quantizacao: arredondamento para multiplo de q ───────────────
    # q = noise_level × 0.1 garante granularidade proporcional ao noise.
    # tf.round pode nao ser diferenciavel, mas noise nao entra no backprop.
    q = noise_level * 0.1
    q = tf.maximum(q, 1e-12)  # protecao contra divisao por zero
    quantized = tf.round(em_feats / q) * q
    return tf.concat([protected, quantized], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCOES DE NOISE EXTENDED — 12 TIPOS
# ════════════════════════════════════════════════════════════════════════════
# Efeitos avancados de noise presentes em cenarios especificos de campo
# LWD. Incluem acoplamento entre canais, efeitos de orientacao da
# ferramenta, interferencia eletromagnetica, e efeitos termicos.
#
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  12 Tipos Extended:                                                      │
#   │                                                                          │
#   │  Tipo             │ Formula                         │ Fenomeno Fisico    │
#   │  ─────────────────┼─────────────────────────────────┼────────────────────│
#   │  cross_talk       │ Re+=ε·Im, Im+=ε·Re             │ Acoplamento cap.   │
#   │  orientation      │ cos(θ)c0+sin(θ)c1              │ Rotacao mandrel    │
#   │  emi_noise        │ Σ A·sin(2πkft)                 │ EMI 60Hz harmonics │
#   │  freq_dependent   │ N(0,σ·f^α)                     │ Piso freq-dep.     │
#   │  noise_floor      │ N(0,floor)                     │ Limite deteccao    │
#   │  proportional     │ N(0,0.03·|x|)·scale            │ Erro prop. ~3%     │
#   │  reim_diff        │ Re:σ, Im:1.5σ                  │ Im mais ruidoso    │
#   │  component_diff   │ Hxx:1.0σ, Hzz:0.8σ            │ Sensib. antena Δ   │
#   │  gaussian_keras   │ N(0,σ)                         │ Keras GaussianNoise│
#   │  motion           │ x·(1+A·sin(2πft/L))            │ Vibracao BHA       │
#   │  thermal          │ N(0,0.3σ)                      │ Johnson-Nyquist    │
#   │  phase_shift      │ rot(Re,Im,φ)                   │ Erro demodulador   │
#   │                                                                          │
#   │  Todos preservam z_obs (col 0) e operam on-the-fly via tf.data.map     │
#   └──────────────────────────────────────────────────────────────────────────┘
#
# Ref: docs/reference/noise_catalog.md secao 4 (Extended).
# ──────────────────────────────────────────────────────────────────────────


def _add_cross_talk_noise_tf(x, noise_level, n_protected=1):
    """Noise de cross-talk: acoplamento capacitivo Re↔Im.

    Modela o vazamento de sinal entre os canais in-phase (Re) e
    quadrature (Im) causado por acoplamento capacitivo nos cabos
    de sinal da ferramenta LWD. O efeito eh proporcional ao sinal
    do canal vizinho.

    Formula: Re += ε×Im, Im += ε×Re, onde ε = noise_level × 0.4.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com cross-talk Re↔Im nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["cross_talk"].
        Ref: noise_catalog.md tipo cross_talk.
        ε=0.02 para noise_level=0.05 (~2% de vazamento entre canais).
        Para 4 EM cols: [Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)].
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Cross-talk: colunas pares (Re) vazam para impares (Im) e vice-versa
    # epsilon = noise_level × 0.4 controla magnitude do acoplamento.
    epsilon = noise_level * 0.4
    n_em = tf.shape(em_feats)[2]
    # Shift circular de 1 posicao: Re↔Im adjacentes
    shifted = tf.roll(em_feats, shift=1, axis=2)
    noisy_em = em_feats + epsilon * shifted
    return tf.concat([protected, noisy_em], axis=-1)


def _add_orientation_noise_tf(x, noise_level, n_protected=1):
    """Noise de orientacao: rotacao do mandrel da ferramenta.

    Modela a rotacao do mandrel durante a perfuracao que mistura
    as componentes EM em pares. A ferramenta LWD rota continuamente
    (~60-120 RPM), e erros no giroscopio/magnetometro de orientacao
    produzem mistura entre pares de colunas.

    Formula: col0' = cos(θ)×col0 + sin(θ)×col1
             col1' = -sin(θ)×col0 + cos(θ)×col1
    Onde θ = noise_level × 0.04 radianos (~2° para σ=0.05).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com rotacao entre pares de colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["orientation"].
        Ref: noise_catalog.md tipo orientation.
        θ=0.002 rad para noise_level=0.05 (~0.11°).
        Rotacao aplicada a pares consecutivos de colunas EM.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Angulo de rotacao proporcional ao noise_level ────────────────
    theta = noise_level * 0.04
    cos_t = tf.cos(theta)
    sin_t = tf.sin(theta)
    # ── Rotacao aplicada a pares consecutivos (0,1), (2,3), ... ─────
    # Para n_em impar, ultima coluna permanece inalterada.
    n_em = tf.shape(em_feats)[2]
    n_pairs = n_em // 2
    # Reshape para operar em pares: (batch, seq, n_pairs, 2)
    paired = tf.reshape(
        em_feats[:, :, : n_pairs * 2],
        (tf.shape(em_feats)[0], tf.shape(em_feats)[1], n_pairs, 2),
    )
    col0 = paired[:, :, :, 0]  # (batch, seq, n_pairs)
    col1 = paired[:, :, :, 1]
    rot0 = cos_t * col0 + sin_t * col1
    rot1 = -sin_t * col0 + cos_t * col1
    rotated = tf.stack([rot0, rot1], axis=-1)  # (batch, seq, n_pairs, 2)
    rotated_flat = tf.reshape(
        rotated,
        (tf.shape(em_feats)[0], tf.shape(em_feats)[1], n_pairs * 2),
    )
    # Concatenar coluna impar restante se existir
    remainder = em_feats[:, :, n_pairs * 2 :]
    noisy_em = tf.concat([rotated_flat, remainder], axis=-1)
    return tf.concat([protected, noisy_em], axis=-1)


def _add_emi_noise_tf(x, noise_level, n_protected=1):
    """Noise de interferencia eletromagnetica (EMI): harmonicos 60Hz.

    Modela interferencia eletromagnetica de equipamentos do rig
    (motores, geradores) que operam a 60Hz (ou 50Hz). O noise
    eh composto por 3 harmonicos (60, 120, 180 Hz) com amplitude
    decrescente.

    Formula: Σ(amplitude × sin(2π × k × f × t)) para k=1..3
    Onde f=60Hz, amplitude = noise_level × 0.2 / k.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com interferencia EMI 60Hz nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["emi_noise"].
        Ref: noise_catalog.md tipo emi_noise.
        3 harmonicos (60, 120, 180 Hz) com amplitude 1/k.
        Fase aleatoria por harmonic para evitar cancelamento.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Eixo temporal normalizado [0, 1) ────────────────────────────
    seq_len = tf.cast(tf.shape(em_feats)[1], tf.float32)
    t = tf.cast(tf.range(tf.shape(em_feats)[1]), tf.float32) / tf.maximum(seq_len, 1.0)
    t = tf.reshape(t, (1, -1, 1))  # (1, seq_len, 1) — broadcasta
    # ── Soma de 3 harmonicos com fase aleatoria ─────────────────────
    base_freq = 60.0
    amplitude = noise_level * 0.2
    emi = tf.zeros_like(em_feats)
    for k in range(1, 4):
        phase = tf.random.uniform(
            shape=(1, 1, tf.shape(em_feats)[2]),
            minval=0.0,
            maxval=2.0 * 3.14159265,
            dtype=tf.float32,
        )
        harmonic = (amplitude / tf.cast(k, tf.float32)) * tf.sin(
            2.0 * 3.14159265 * tf.cast(k, tf.float32) * base_freq * t + phase
        )
        emi = emi + harmonic
    noisy_em = em_feats + emi
    return tf.concat([protected, noisy_em], axis=-1)


def _add_freq_dependent_noise_tf(x, noise_level, n_protected=1):
    """Noise dependente de frequencia: piso de noise cresce com posicao.

    Modela o piso de noise que aumenta com a frequencia de operacao
    da ferramenta. Em LWD, frequencias mais altas sofrem maior
    atenuacao e portanto menor SNR. Simplificacao: noise cresce
    com posicao na sequencia (proxy para profundidade/frequencia).

    Formula: x + N(0, σ × f^α) onde α=0.5, f = posicao normalizada.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise freq-dependent nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["freq_dependent"].
        Ref: noise_catalog.md tipo freq_dependent.
        α=0.5 → noise cresce como sqrt(posicao) (sublinear).
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Fator posicional: (1 + idx/L)^0.5 ──────────────────────────
    # Cresce de 1.0 a sqrt(2) ao longo da sequencia.
    seq_len = tf.cast(tf.shape(em_feats)[1], tf.float32)
    indices = tf.cast(tf.range(tf.shape(em_feats)[1]), tf.float32)
    f_factor = tf.sqrt(1.0 + indices / tf.maximum(seq_len, 1.0))
    f_factor = tf.reshape(f_factor, (1, -1, 1))  # (1, seq_len, 1)
    noise = tf.random.normal(
        shape=tf.shape(em_feats), mean=0.0, stddev=noise_level, dtype=tf.float32
    )
    noisy_em = em_feats + noise * f_factor
    return tf.concat([protected, noisy_em], axis=-1)


def _add_noise_floor_noise_tf(x, noise_level, n_protected=1):
    """Noise floor: limite de deteccao do instrumento.

    Modela o piso de noise constante do receptor LWD, independente
    da magnitude do sinal. Representa o limite fundamental de
    deteccao do sistema (thermal noise + noise de quantizacao).

    Formula: x + N(0, floor_value) onde floor_value = noise_level × 1e-8 / 0.05.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise floor constante nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["noise_floor"].
        Ref: noise_catalog.md tipo noise_floor.
        floor_value=2e-7 para noise_level=0.05 (~10 nV em campo H).
        Constante — nao depende da magnitude do sinal.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Noise floor constante, escalado por noise_level ──────────────
    floor_value = noise_level * 1e-8 / 0.05
    noise = tf.random.normal(
        shape=tf.shape(em_feats), mean=0.0, stddev=floor_value, dtype=tf.float32
    )
    noisy_em = em_feats + noise
    return tf.concat([protected, noisy_em], axis=-1)


def _add_proportional_noise_tf(x, noise_level, n_protected=1):
    """Noise proporcional: erro ~3% da magnitude do sinal.

    Modela o erro proporcional tipico de ferramentas LWD comerciais
    (especificacao de fabrica: ~3% do valor lido). O noise eh
    proporcional a |x| com fator fixo de 3%, escalado pelo noise_level.

    Formula: x + N(0, 0.03 × |x|) × (noise_level / 0.05).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com erro proporcional ~3% nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["proportional"].
        Ref: noise_catalog.md tipo proportional.
        3% eh especificacao tipica de ferramentas Schlumberger/Halliburton.
        noise_level=0.05 → exatamente 3%. noise_level=0.1 → 6%.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Erro proporcional: 3% × |x| × scale ─────────────────────────
    scale = noise_level / 0.05
    stddev = 0.03 * tf.abs(em_feats) * scale
    noise = tf.random.normal(shape=tf.shape(em_feats), dtype=tf.float32)
    noisy_em = em_feats + noise * stddev
    return tf.concat([protected, noisy_em], axis=-1)


def _add_reim_diff_noise_tf(x, noise_level, n_protected=1):
    """Noise Re/Im diferencial: Im mais ruidoso que Re.

    Modela a assimetria de noise entre canais in-phase (Re) e
    quadrature (Im). Em receptores LWD, o canal Im tipicamente
    tem SNR ~30% menor que Re devido a demodulacao e filtragem
    menos eficientes no caminho quadrature.

    Formula: Re += N(0, σ), Im += N(0, 1.5σ).
    Split por colunas pares (Re) e impares (Im).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise diferencial Re/Im nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["reim_diff"].
        Ref: noise_catalog.md tipo reim_diff.
        Im recebe 50% mais noise que Re (fator 1.5).
        Colunas pares = Re, impares = Im (layout P1: Re,Im,Re,Im).
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Noise diferencial: Re (σ) e Im (1.5σ) ──────────────────────
    # Mascara de escala: colunas pares=1.0, impares=1.5
    n_em = tf.shape(em_feats)[2]
    # Pares (0,2,4...)=1.0 (Re), ímpares (1,3,5...)=1.5 (Im)
    scale = tf.where(
        tf.equal(tf.math.mod(tf.range(n_em), 2), 0),
        tf.ones(n_em, dtype=tf.float32),
        tf.ones(n_em, dtype=tf.float32) * 1.5,
    )
    scale = tf.reshape(scale, (1, 1, -1))  # (1, 1, n_em)
    noise = tf.random.normal(
        shape=tf.shape(em_feats), mean=0.0, stddev=noise_level, dtype=tf.float32
    )
    noisy_em = em_feats + noise * scale
    return tf.concat([protected, noisy_em], axis=-1)


def _add_component_diff_noise_tf(x, noise_level, n_protected=1):
    """Noise por componente: Hxx e Hzz com sensibilidades diferentes.

    Modela a diferenca de sensibilidade entre antenas coplanares (Hxx)
    e coaxiais (Hzz) da ferramenta LWD. A antena coaxial tipicamente
    tem melhor SNR (~20% menor noise) por geometria de acoplamento.

    Formula: primeiras 2 cols EM × σ×1.0, ultimas 2 cols × σ×0.8.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise diferenciado Hxx/Hzz.

    Note:
        Referenciado em: NOISE_FN_MAP["component_diff"].
        Ref: noise_catalog.md tipo component_diff.
        Hxx (planar): fator 1.0. Hzz (axial): fator 0.8.
        Para <4 EM cols, primeiras metade=1.0, segunda metade=0.8.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Escala por componente: Hxx=1.0, Hzz=0.8 ─────────────────────
    n_em = tf.shape(em_feats)[2]
    half = n_em // 2
    # Primeiras 'half' colunas (Hxx): fator 1.0
    # Ultimas colunas (Hzz): fator 0.8
    scale_hxx = tf.ones((1, 1, half), dtype=tf.float32) * 1.0
    scale_hzz = tf.ones((1, 1, n_em - half), dtype=tf.float32) * 0.8
    scale = tf.concat([scale_hxx, scale_hzz], axis=-1)
    noise = tf.random.normal(
        shape=tf.shape(em_feats), mean=0.0, stddev=noise_level, dtype=tf.float32
    )
    noisy_em = em_feats + noise * scale
    return tf.concat([protected, noisy_em], axis=-1)


def _add_gaussian_keras_noise_tf(x, noise_level, n_protected=1):
    """Noise gaussiano puro (Keras GaussianNoise equivalente).

    Identico ao tipo 'gaussian', mas existe como entrada separada
    no catalogo para compatibilidade com usuarios que referenciam
    tf.keras.layers.GaussianNoise. Semantica identica: x + N(0, σ).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise gaussiano puro nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["gaussian_keras"].
        Identico a NOISE_FN_MAP["gaussian"] — alias para catalogo.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    noise = tf.random.normal(
        shape=tf.shape(em_feats), mean=0.0, stddev=noise_level, dtype=tf.float32
    )
    return tf.concat([protected, em_feats + noise], axis=-1)


def _add_motion_noise_tf(x, noise_level, n_protected=1):
    """Noise de movimento: vibracao BHA via modulacao sinusoidal.

    Modela a vibracao mecanica do BHA (Bottom Hole Assembly) durante
    a perfuracao. A vibracao lateral produz modulacao periodica do
    sinal EM captado, com frequencia tipica de ~5 ciclos por perfil.

    Formula: x × (1 + amplitude × sin(2π × freq × t / seq_len))
    Onde amplitude = noise_level × 0.4, freq = 5.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com modulacao sinusoidal de vibracao BHA.

    Note:
        Referenciado em: NOISE_FN_MAP["motion"].
        Ref: noise_catalog.md tipo motion.
        freq=5 → ~5 ciclos de vibracao por perfil de 600 amostras.
        amplitude=0.02 para noise_level=0.05 (~2% modulacao).
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Modulacao sinusoidal: 5 ciclos por sequencia ────────────────
    amplitude = noise_level * 0.4
    freq = 5.0
    seq_len = tf.cast(tf.shape(em_feats)[1], tf.float32)
    t = tf.cast(tf.range(tf.shape(em_feats)[1]), tf.float32)
    modulation = amplitude * tf.sin(
        2.0 * 3.14159265 * freq * t / tf.maximum(seq_len, 1.0)
    )
    modulation = tf.reshape(modulation, (1, -1, 1))  # (1, seq_len, 1)
    noisy_em = em_feats * (1.0 + modulation)
    return tf.concat([protected, noisy_em], axis=-1)


def _add_thermal_noise_tf(x, noise_level, n_protected=1):
    """Noise termico: Johnson-Nyquist do receptor LWD.

    Modela o ruido termico fundamental (Johnson-Nyquist) gerado pela
    resistencia dos circuitos do receptor LWD a ~175°C (temperatura
    tipica de fundo de poco). Simplificado para N(0, 0.3σ) representando
    sqrt(4×k_B×T×R×Δf) / referencia em unidades normalizadas.

    Formula: x + N(0, σ_thermal), σ_thermal = noise_level × 0.3.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com noise termico Johnson-Nyquist.

    Note:
        Referenciado em: NOISE_FN_MAP["thermal"].
        Ref: noise_catalog.md tipo thermal. Johnson (1928).
        Fator 0.3 calibrado para T≈175°C, R≈50Ω, Δf≈1kHz.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Noise termico: gaussiano com σ reduzido ─────────────────────
    sigma_thermal = noise_level * 0.3
    noise = tf.random.normal(
        shape=tf.shape(em_feats), mean=0.0, stddev=sigma_thermal, dtype=tf.float32
    )
    noisy_em = em_feats + noise
    return tf.concat([protected, noisy_em], axis=-1)


def _add_phase_shift_noise_tf(x, noise_level, n_protected=1):
    """Noise de phase shift: erro do demodulador Re/Im.

    Modela o erro de fase do demodulador que rota as componentes
    Re e Im por um angulo aleatorio. Em receptores LWD, o PLL
    (Phase-Locked Loop) pode ter jitter de fase que mistura Re↔Im.

    Formula: Re' = Re×cos(φ) - Im×sin(φ)
             Im' = Re×sin(φ) + Im×cos(φ)
    Onde φ = noise_level × 0.1 radianos.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com rotacao de fase Re/Im nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["phase_shift"].
        Ref: noise_catalog.md tipo phase_shift.
        φ=0.005 rad para noise_level=0.05 (~0.29°).
        Rotacao aplicada a pares (Re,Im) consecutivos.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Angulo de fase aleatorio por batch ──────────────────────────
    phi = noise_level * 0.1
    cos_p = tf.cos(phi)
    sin_p = tf.sin(phi)
    # ── Rotacao em pares (Re, Im) consecutivos ──────────────────────
    n_em = tf.shape(em_feats)[2]
    n_pairs = n_em // 2
    paired = tf.reshape(
        em_feats[:, :, : n_pairs * 2],
        (tf.shape(em_feats)[0], tf.shape(em_feats)[1], n_pairs, 2),
    )
    re = paired[:, :, :, 0]
    im = paired[:, :, :, 1]
    re_rot = re * cos_p - im * sin_p
    im_rot = re * sin_p + im * cos_p
    rotated = tf.stack([re_rot, im_rot], axis=-1)
    rotated_flat = tf.reshape(
        rotated,
        (tf.shape(em_feats)[0], tf.shape(em_feats)[1], n_pairs * 2),
    )
    remainder = em_feats[:, :, n_pairs * 2 :]
    noisy_em = tf.concat([rotated_flat, remainder], axis=-1)
    return tf.concat([protected, noisy_em], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCOES DE NOISE GEOSTEERING — 2 TIPOS (R7-R8)
# ════════════════════════════════════════════════════════════════════════════
# Efeitos especificos de geosteering: vibracao lateral do BHA e
# excentricidade da ferramenta no poco. Estes efeitos sao mais
# pronunciados em pocos horizontais/direcionais.
#
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  2 Tipos Geosteering (R7-R8):                                           │
#   │                                                                          │
#   │  ID │ Tipo           │ Efeito Fisico                                    │
#   │  ───┼────────────────┼──────────────────────────────────────────────────│
#   │  R7 │ bha_vibration  │ Deslocamento lateral sinusoidal do sensor        │
#   │  R8 │ eccentricity   │ Ganho variavel por excentricidade no borehole    │
#   └──────────────────────────────────────────────────────────────────────────┘
#
# Ref: docs/reference/noise_catalog.md secao 6 (R7-R8).
# ──────────────────────────────────────────────────────────────────────────


def _add_bha_vibration_noise_tf(x, noise_level, n_protected=1):
    """R7 — BHA vibration: deslocamento lateral sinusoidal do sensor.

    Modela a vibracao lateral do BHA que desloca o sensor EM em
    relacao a formacao. Em pocos direcionais/horizontais, o BHA
    vibra com frequencia proporcional ao RPM (~1-2 Hz). O
    deslocamento produz perturbacao aditiva sinusoidal com fase
    aleatoria.

    Formula: x + shift × sin(2π × t / seq_len + random_phase)
    Onde shift = noise_level × 0.5.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com vibracao lateral sinusoidal.

    Note:
        Referenciado em: NOISE_FN_MAP["bha_vibration"].
        Ref: noise_catalog.md R7.
        shift=0.025 para noise_level=0.05 (~2.5% do sinal).
        1 ciclo por sequencia com fase aleatoria.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Vibracao sinusoidal com fase aleatoria ──────────────────────
    shift = noise_level * 0.5
    seq_len = tf.cast(tf.shape(em_feats)[1], tf.float32)
    t = tf.cast(tf.range(tf.shape(em_feats)[1]), tf.float32)
    # Fase aleatoria por feature para diversidade
    phase = tf.random.uniform(
        shape=(1, 1, tf.shape(em_feats)[2]),
        minval=0.0,
        maxval=2.0 * 3.14159265,
        dtype=tf.float32,
    )
    vibration = shift * tf.sin(
        2.0 * 3.14159265 * tf.reshape(t, (1, -1, 1)) / tf.maximum(seq_len, 1.0) + phase
    )
    noisy_em = em_feats + vibration
    return tf.concat([protected, noisy_em], axis=-1)


def _add_eccentricity_noise_tf(x, noise_level, n_protected=1):
    """R8 — Eccentricity: ganho variavel por ferramenta nao-centrada.

    Modela o efeito de excentricidade da ferramenta no borehole.
    Quando a ferramenta nao esta centrada, o acoplamento EM com
    a formacao varia periodicamente com a posicao angular, criando
    modulacao de ganho cossenoidal.

    Formula: x × (1 + ecc × cos(2π × t / seq_len + phase))
    Onde ecc = noise_level × 0.2.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.
        n_protected: Colunas iniciais protegidas. Default 1 (P1).

    Returns:
        tf.Tensor: x com modulacao de ganho por excentricidade.

    Note:
        Referenciado em: NOISE_FN_MAP["eccentricity"].
        Ref: noise_catalog.md R8.
        ecc=0.01 para noise_level=0.05 (~1% modulacao de ganho).
        1 ciclo por sequencia com fase aleatoria.
        Colunas 0:n_protected preservadas (parametros conhecidos + z_obs).
    """
    import tensorflow as tf

    protected = x[:, :, :n_protected]
    em_feats = x[:, :, n_protected:]
    # ── Modulacao cossenoidal por excentricidade ─────────────────────
    ecc = noise_level * 0.2
    seq_len = tf.cast(tf.shape(em_feats)[1], tf.float32)
    t = tf.cast(tf.range(tf.shape(em_feats)[1]), tf.float32)
    phase = tf.random.uniform(
        shape=(1, 1, tf.shape(em_feats)[2]),
        minval=0.0,
        maxval=2.0 * 3.14159265,
        dtype=tf.float32,
    )
    modulation = 1.0 + ecc * tf.cos(
        2.0 * 3.14159265 * tf.reshape(t, (1, -1, 1)) / tf.maximum(seq_len, 1.0) + phase
    )
    noisy_em = em_feats * modulation
    return tf.concat([protected, noisy_em], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: REGISTRO DE FUNCOES (NOISE_FN_MAP)
# ════════════════════════════════════════════════════════════════════════════
# Dicionario tipo→funcao para extensibilidade. Novos tipos de noise
# podem ser adicionados registrando aqui sem alterar o dispatcher.
# 34 tipos totais: 4 originais + 5 CORE + 5 CORE missing + 6 LWD (R1-R6)
#                  + 12 Extended + 2 Geosteering (R7-R8).
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
    # ── 5 CORE missing (Fase III) ──────────────────────────────────────
    "varying": _add_varying_noise_tf,
    "gaussian_local": _add_gaussian_local_noise_tf,
    "gaussian_global": _add_gaussian_global_noise_tf,
    "speckle": _add_speckle_noise_tf,
    "quantization": _add_quantization_noise_tf,
    # ── 6 Geofisicos LWD R1-R6 (Fase II) ────────────────────────────
    "shoulder_bed": _add_shoulder_bed_noise_tf,
    "borehole_effect": _add_borehole_effect_noise_tf,
    "mud_invasion": _add_mud_invasion_noise_tf,
    "anisotropy_misalignment": _add_anisotropy_misalignment_noise_tf,
    "formation_heterogeneity": _add_formation_heterogeneity_noise_tf,
    "telemetry": _add_telemetry_noise_tf,
    # ── 12 Extended (Fase III) ───────────────────────────────────────
    "cross_talk": _add_cross_talk_noise_tf,
    "orientation": _add_orientation_noise_tf,
    "emi_noise": _add_emi_noise_tf,
    "freq_dependent": _add_freq_dependent_noise_tf,
    "noise_floor": _add_noise_floor_noise_tf,
    "proportional": _add_proportional_noise_tf,
    "reim_diff": _add_reim_diff_noise_tf,
    "component_diff": _add_component_diff_noise_tf,
    "gaussian_keras": _add_gaussian_keras_noise_tf,
    "motion": _add_motion_noise_tf,
    "thermal": _add_thermal_noise_tf,
    "phase_shift": _add_phase_shift_noise_tf,
    # ── 2 Geosteering R7-R8 (Fase III) ──────────────────────────────
    "bha_vibration": _add_bha_vibration_noise_tf,
    "eccentricity": _add_eccentricity_noise_tf,
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
            noisy = em_feats * keep + bit_noise * keep  # bit noise só onde não dropou
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * noisy).astype(
                em_feats.dtype
            )

        # ── 5 CORE missing (Fase III) ──────────────────────────────
        elif nt == "varying":
            # ── Heteroscedastico: N·U(σ_min,σ_max)·|x| ────────────
            sigma_min = 0.01 * noise_level
            sigma_max = max(noise_level, sigma_min + 1e-12)
            sigma_local = rng.uniform(sigma_min, sigma_max, size=em_feats.shape)
            noise = rng.normal(0.0, 1.0, size=em_feats.shape)
            em_feats += w_norm * (noise * sigma_local * np.abs(em_feats)).astype(
                em_feats.dtype
            )

        elif nt == "gaussian_local":
            # ── Erro proporcional ao sinal local: N(0,pct·|x|) ─────
            stddev = noise_level * np.abs(em_feats)
            noise = rng.normal(0.0, 1.0, size=em_feats.shape)
            em_feats += w_norm * (noise * stddev).astype(em_feats.dtype)

        elif nt == "gaussian_global":
            # ── Erro proporcional ao std global: N(0,pct·std_glob) ──
            std_global = max(np.std(em_feats), 1e-12)
            noise = rng.normal(0.0, noise_level * std_global, size=em_feats.shape)
            em_feats += w_norm * noise.astype(em_feats.dtype)

        elif nt == "speckle":
            # ── Gain noise com σ²: x·(1+N(0,σ²)) ───────────────────
            noise = rng.normal(0.0, noise_level * noise_level, size=em_feats.shape)
            noisy = em_feats * (1.0 + noise)
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * noisy).astype(
                em_feats.dtype
            )

        elif nt == "quantization":
            # ── ADC quantization: round(x/q)·q ─────────────────────
            q = max(noise_level * 0.1, 1e-12)
            quantized = np.round(em_feats / q) * q
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * quantized).astype(
                em_feats.dtype
            )

        # ── 12 Extended (Fase III) ─────────────────────────────────
        elif nt == "cross_talk":
            # ── Acoplamento capacitivo Re↔Im: shift circular ────────
            epsilon = noise_level * 0.4
            shifted = np.roll(em_feats, shift=1, axis=2)
            noisy = em_feats + epsilon * shifted
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * noisy).astype(
                em_feats.dtype
            )

        elif nt == "orientation":
            # ── Rotacao mandrel: cos(θ)c0+sin(θ)c1 por pares ───────
            theta = noise_level * 0.04
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            n_em_local = em_feats.shape[2]
            n_pairs = n_em_local // 2
            result_em = em_feats.copy()
            for p in range(n_pairs):
                c0 = em_feats[:, :, 2 * p]
                c1 = em_feats[:, :, 2 * p + 1]
                result_em[:, :, 2 * p] = cos_t * c0 + sin_t * c1
                result_em[:, :, 2 * p + 1] = -sin_t * c0 + cos_t * c1
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * result_em).astype(
                em_feats.dtype
            )

        elif nt == "emi_noise":
            # ── EMI 60Hz: 3 harmonicos sinusoidais ──────────────────
            seq_len_local = em_feats.shape[1]
            t = np.arange(seq_len_local).reshape(1, -1, 1) / max(seq_len_local, 1)
            amplitude = noise_level * 0.2
            emi = np.zeros_like(em_feats)
            for k in range(1, 4):
                phase = rng.uniform(0, 2 * np.pi, size=(1, 1, em_feats.shape[2]))
                emi += (amplitude / k) * np.sin(2 * np.pi * k * 60.0 * t + phase)
            em_feats += w_norm * emi.astype(em_feats.dtype)

        elif nt == "freq_dependent":
            # ── Noise freq-dependent: σ·sqrt(1+idx/L) ──────────────
            seq_len_local = em_feats.shape[1]
            indices = np.arange(seq_len_local).reshape(1, -1, 1)
            f_factor = np.sqrt(1.0 + indices / max(seq_len_local, 1))
            noise = rng.normal(0.0, noise_level, size=em_feats.shape)
            em_feats += w_norm * (noise * f_factor).astype(em_feats.dtype)

        elif nt == "noise_floor":
            # ── Piso de noise constante: N(0, floor) ────────────────
            floor_value = noise_level * 1e-8 / 0.05
            noise = rng.normal(0.0, floor_value, size=em_feats.shape)
            em_feats += w_norm * noise.astype(em_feats.dtype)

        elif nt == "proportional":
            # ── Erro proporcional ~3%: N(0,0.03·|x|)·scale ─────────
            scale = noise_level / 0.05
            stddev = 0.03 * np.abs(em_feats) * scale
            noise = rng.normal(0.0, 1.0, size=em_feats.shape)
            em_feats += w_norm * (noise * stddev).astype(em_feats.dtype)

        elif nt == "reim_diff":
            # ── Im mais ruidoso que Re: pares=1.0, impares=1.5 ─────
            n_em_local = em_feats.shape[2]
            scale = np.ones(n_em_local, dtype=np.float32)
            scale[1::2] = 1.5  # colunas impares (Im) recebem 50% mais
            scale = scale.reshape(1, 1, -1)
            noise = rng.normal(0.0, noise_level, size=em_feats.shape)
            em_feats += w_norm * (noise * scale).astype(em_feats.dtype)

        elif nt == "component_diff":
            # ── Hxx(1.0σ) vs Hzz(0.8σ): split por metade ──────────
            n_em_local = em_feats.shape[2]
            half = n_em_local // 2
            scale = np.ones(n_em_local, dtype=np.float32)
            scale[half:] = 0.8  # Hzz (segunda metade) menos ruidoso
            scale = scale.reshape(1, 1, -1)
            noise = rng.normal(0.0, noise_level, size=em_feats.shape)
            em_feats += w_norm * (noise * scale).astype(em_feats.dtype)

        elif nt == "gaussian_keras":
            # ── Keras GaussianNoise equivalente: x+N(0,σ) ──────────
            noise = rng.normal(0.0, noise_level, size=em_feats.shape)
            em_feats += w_norm * noise.astype(em_feats.dtype)

        elif nt == "motion":
            # ── Vibracao BHA sinusoidal: x·(1+A·sin(2πft/L)) ───────
            amplitude = noise_level * 0.4
            freq = 5.0
            seq_len_local = em_feats.shape[1]
            t = np.arange(seq_len_local).reshape(1, -1, 1)
            modulation = amplitude * np.sin(
                2.0 * np.pi * freq * t / max(seq_len_local, 1)
            )
            noisy = em_feats * (1.0 + modulation)
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * noisy).astype(
                em_feats.dtype
            )

        elif nt == "thermal":
            # ── Johnson-Nyquist: N(0, 0.3σ) ────────────────────────
            sigma_thermal = noise_level * 0.3
            noise = rng.normal(0.0, sigma_thermal, size=em_feats.shape)
            em_feats += w_norm * noise.astype(em_feats.dtype)

        elif nt == "phase_shift":
            # ── Demodulator phase error: rotacao Re/Im por φ ────────
            phi = noise_level * 0.1
            cos_p = np.cos(phi)
            sin_p = np.sin(phi)
            n_em_local = em_feats.shape[2]
            n_pairs = n_em_local // 2
            result_em = em_feats.copy()
            for p in range(n_pairs):
                re = em_feats[:, :, 2 * p]
                im = em_feats[:, :, 2 * p + 1]
                result_em[:, :, 2 * p] = re * cos_p - im * sin_p
                result_em[:, :, 2 * p + 1] = re * sin_p + im * cos_p
            em_feats[:] = ((1.0 - w_norm) * em_feats + w_norm * result_em).astype(
                em_feats.dtype
            )

        # ── 2 Geosteering R7-R8 (Fase III) ────────────────────────
        elif nt == "bha_vibration":
            # ── R7: deslocamento lateral sinusoidal ─────────────────
            shift = noise_level * 0.5
            seq_len_local = em_feats.shape[1]
            t = np.arange(seq_len_local).reshape(1, -1, 1)
            phase = rng.uniform(0, 2 * np.pi, size=(1, 1, em_feats.shape[2]))
            vibration = shift * np.sin(2.0 * np.pi * t / max(seq_len_local, 1) + phase)
            em_feats += w_norm * vibration.astype(em_feats.dtype)

        elif nt == "eccentricity":
            # ── R8: modulacao de ganho cossenoidal ──────────────────
            ecc = noise_level * 0.2
            seq_len_local = em_feats.shape[1]
            t = np.arange(seq_len_local).reshape(1, -1, 1)
            phase = rng.uniform(0, 2 * np.pi, size=(1, 1, em_feats.shape[2]))
            modulation = 1.0 + ecc * np.cos(
                2.0 * np.pi * t / max(seq_len_local, 1) + phase
            )
            noisy = em_feats * modulation
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
