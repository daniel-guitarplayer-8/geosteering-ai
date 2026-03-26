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
#   │  z_obs (col 0) NUNCA recebe noise                                       │
#   └──────────────────────────────────────────────────────────────────────────┘

# Tipo callable para funcoes de noise TF.
# Assinatura: (x: tf.Tensor, noise_level: tf.Variable) -> tf.Tensor
NoiseFnType = Callable  # tf.Tensor, tf.Variable -> tf.Tensor


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCOES DE NOISE TENSORFLOW
# ════════════════════════════════════════════════════════════════════════════
# Cada funcao recebe um tensor 3D (batch, seq_len, n_feat) e um
# tf.Variable com o nivel de ruido atual (controlado pelo curriculum).
# O noise eh aplicado APENAS nas colunas EM (indices 1: em diante),
# preservando z_obs (indice 0) intacto.
# Ref: Cadeia on-the-fly: raw → noise → FV → GS → scale → modelo.
# ──────────────────────────────────────────────────────────────────────────


def _add_gaussian_noise_tf(x, noise_level):
    """Noise gaussiano aditivo: x + N(0, sigma^2).

    Modela ruido eletronico aditivo do receptor LWD. Tipo mais
    comum e fisicamente motivado para medidas EM.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat). Layout:
            [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz), ...GS].
        noise_level: tf.Variable escalar com sigma atual.

    Returns:
        tf.Tensor: x com noise gaussiano nas colunas EM (indices 1:).

    Note:
        Referenciado em: NOISE_FN_MAP["gaussian"].
        z_obs (col 0) preservado. GS columns tambem recebem noise
        (correto — GS sao derivados dos EM ruidosos).
    """
    import tensorflow as tf
    # ── Separar z_obs (col 0) das features EM (cols 1:) ──────────────
    z_obs = x[:, :, :1]           # (batch, seq, 1) — NUNCA noise
    em_feats = x[:, :, 1:]        # (batch, seq, n_feat-1) — recebe noise
    noise = tf.random.normal(
        shape=tf.shape(em_feats), mean=0.0, stddev=noise_level,
        dtype=tf.float32,
    )
    return tf.concat([z_obs, em_feats + noise], axis=-1)


def _add_multiplicative_noise_tf(x, noise_level):
    """Noise multiplicativo (speckle): x * (1 + N(0, sigma^2)).

    Modela erros de ganho do amplificador LWD e variacao de
    sensibilidade do receptor com temperatura/pressao.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.

    Returns:
        tf.Tensor: x com noise multiplicativo nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["multiplicative"].
        Ref: Noise tipo "speckle" no catalogo C12 legado.
    """
    import tensorflow as tf
    z_obs = x[:, :, :1]
    em_feats = x[:, :, 1:]
    noise = tf.random.normal(
        shape=tf.shape(em_feats), mean=0.0, stddev=noise_level,
        dtype=tf.float32,
    )
    return tf.concat([z_obs, em_feats * (1.0 + noise)], axis=-1)


def _add_uniform_noise_tf(x, noise_level):
    """Noise uniforme: x + U(-sigma, sigma).

    Modela erro de quantizacao ADC (resolucao finita do conversor
    analogico-digital).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar.

    Returns:
        tf.Tensor: x com noise uniforme nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["uniform"].
    """
    import tensorflow as tf
    z_obs = x[:, :, :1]
    em_feats = x[:, :, 1:]
    noise = tf.random.uniform(
        shape=tf.shape(em_feats), minval=-noise_level, maxval=noise_level,
        dtype=tf.float32,
    )
    return tf.concat([z_obs, em_feats + noise], axis=-1)


def _add_dropout_noise_tf(x, noise_level):
    """Noise dropout: mascara aleatoria de canais/amostras.

    Modela perda de dados por falha de telemetria ou dropout de
    canal durante perfuracao. Usa inverted dropout para manter
    escala esperada.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        noise_level: tf.Variable escalar (probabilidade de dropout).

    Returns:
        tf.Tensor: x com dropout nas colunas EM.

    Note:
        Referenciado em: NOISE_FN_MAP["dropout"].
        Inverted dropout: divide por (1-p) para manter escala.
        Se noise_level >= 1.0, retorna zeros (protecao).
    """
    import tensorflow as tf
    z_obs = x[:, :, :1]
    em_feats = x[:, :, 1:]
    # ── Inverted dropout: mask * x / (1-p) ────────────────────────────
    keep_prob = tf.maximum(1.0 - noise_level, 1e-6)
    mask = tf.cast(
        tf.random.uniform(shape=tf.shape(em_feats)) < keep_prob,
        dtype=tf.float32,
    )
    em_dropped = em_feats * mask / keep_prob
    return tf.concat([z_obs, em_dropped], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: REGISTRO DE FUNCOES (NOISE_FN_MAP)
# ════════════════════════════════════════════════════════════════════════════
# Dicionario tipo→funcao para extensibilidade. Novos tipos de noise
# podem ser adicionados registrando aqui sem alterar o dispatcher.
# O pipeline resolve noise_types do config contra este registro.
# Ref: Factory Pattern (CLAUDE.md secao Code Patterns).
# ──────────────────────────────────────────────────────────────────────────

NOISE_FN_MAP: Dict[str, NoiseFnType] = {
    "gaussian": _add_gaussian_noise_tf,
    "multiplicative": _add_multiplicative_noise_tf,
    "uniform": _add_uniform_noise_tf,
    "dropout": _add_dropout_noise_tf,
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
        initial_value, dtype=tf.float32, trainable=False, name=name,
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
) -> "tf.Tensor":
    """Aplica mix de noise types no tensor de features.

    Dispatcher que resolve cada tipo contra NOISE_FN_MAP e aplica
    com pesos normalizados. Se noise_level_var == 0 (fase clean),
    retorna x inalterado (otimizacao).

    Args:
        x: Tensor 3D (batch, seq_len, n_feat). Layout:
            [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz), ...].
        noise_level_var: tf.Variable escalar com sigma atual.
        noise_types: Lista de tipos. Ex: ["gaussian", "multiplicative"].
        noise_weights: Pesos correspondentes. Ex: [0.8, 0.2].
            Normalizados internamente para somar 1.0.

    Returns:
        tf.Tensor: x com noise mixto aplicado.

    Raises:
        ValueError: Se algum tipo nao esta em NOISE_FN_MAP.

    Example:
        >>> from geosteering_ai.noise import apply_noise_tf, create_noise_level_var
        >>> noise_var = create_noise_level_var(0.05)
        >>> x_noisy = apply_noise_tf(x, noise_var, ["gaussian"], [1.0])

    Note:
        Referenciado em:
            - data/pipeline.py: build_train_map_fn() (Step 1 do map_fn)
        Pesos normalizados: [0.6, 0.4] → [0.6, 0.4] (ja normalizado).
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
        return NOISE_FN_MAP[noise_types[0]](x, noise_level_var)

    # ── Mix de N tipos com pesos normalizados ─────────────────────────
    w_sum = sum(noise_weights)
    result = tf.zeros_like(x)
    for nt, w in zip(noise_types, noise_weights):
        w_norm = tf.constant(w / w_sum, dtype=tf.float32)
        noisy = NOISE_FN_MAP[nt](x, noise_level_var)
        result = result + w_norm * noisy

    return result


# ════════════════════════════════════════════════════════════════════════════
# SECAO: VERSAO NUMPY (OFFLINE / TESTES)
# ════════════════════════════════════════════════════════════════════════════
# Versao numpy de apply_raw_em_noise para uso em testes, EDA,
# e cenarios offline. Semantica identica a versao TF:
# noise aplicado APENAS nas colunas EM (indices 1:), z preservado.
# NAO usar em produção com FV/GS ativos (usar on-the-fly TF).
# ──────────────────────────────────────────────────────────────────────────


def apply_raw_em_noise(
    x: np.ndarray,
    noise_level: float = 0.05,
    noise_types: Optional[List[str]] = None,
    noise_weights: Optional[List[float]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Aplica noise nas colunas EM de um array numpy.

    Versao offline/testes. Aplica noise aditivo gaussiano (default)
    ou mix de tipos nas colunas EM (indices 1: em diante), preservando
    z_obs (indice 0) intacto.

    Para uso em produção com FV/GS ativos, usar a versao TF
    via apply_noise_tf dentro de tf.data.map (on-the-fly).

    Args:
        x: Array 3D (n_seq, seq_len, n_feat) com raw features.
            Layout: [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)].
        noise_level: Desvio padrao do noise. Default: 0.05.
        noise_types: Lista de tipos. Default: ["gaussian"].
        noise_weights: Pesos correspondentes. Default: [1.0].
        seed: Seed para reproducibilidade. None = aleatorio.

    Returns:
        np.ndarray: Copia de x com noise nas colunas EM.

    Raises:
        ValueError: Se x.ndim != 3 ou tipo desconhecido.

    Example:
        >>> import numpy as np
        >>> from geosteering_ai.noise import apply_raw_em_noise
        >>> x = np.random.randn(10, 600, 5).astype(np.float32)
        >>> x_noisy = apply_raw_em_noise(x, noise_level=0.05)
        >>> assert x_noisy[:, :, 0] == pytest.approx(x[:, :, 0])  # z preservado

    Note:
        Referenciado em:
            - tests/test_noise.py (validacao offline)
            - visualization/holdout.py (plots clean vs noisy)
        AVISO: NAO usar em producao com FV/GS ativos.
        Cadeia correta eh on-the-fly: raw → noise_tf → FV_tf → GS_tf → scale.
    """
    if x.ndim != 3:
        raise ValueError(f"x deve ser 3D (n_seq, seq_len, n_feat), recebido ndim={x.ndim}")

    if noise_types is None:
        noise_types = ["gaussian"]
    if noise_weights is None:
        noise_weights = [1.0]

    rng = np.random.RandomState(seed)
    result = x.copy()

    # ── Separar z_obs (col 0) das features EM (cols 1:) ──────────────
    em_feats = result[:, :, 1:]  # vista (nao copia) — modifica in-place
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
            em_feats[:] = (
                (1.0 - w_norm) * em_feats
                + w_norm * em_feats * (1.0 + noise.astype(em_feats.dtype))
            )

        elif nt == "uniform":
            # ── Uniforme: U(-sigma, sigma) ────────────────────────────
            noise = rng.uniform(-noise_level, noise_level, size=em_feats.shape)
            em_feats += w_norm * noise.astype(em_feats.dtype)

        elif nt == "dropout":
            # ── Dropout: mascara aleatoria ────────────────────────────
            keep_prob = max(1.0 - noise_level, 1e-6)
            mask = (rng.uniform(size=em_feats.shape) < keep_prob).astype(em_feats.dtype)
            em_feats[:] = (
                (1.0 - w_norm) * em_feats
                + w_norm * em_feats * mask / keep_prob
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
