# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/sampling.py                                                  ║
# ║  Bloco: 2e — Oversampling de Alta Resistividade (Estrategia B)            ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (ponto unico de verdade)                ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • compute_rho_max_per_sequence(): max rho por sequencia (para sampling) ║
# ║    • oversample_high_rho(): repete sequencias de alta rho N vezes          ║
# ║    • filter_by_rho_max(): filtra dataset por threshold de rho              ║
# ║    • Suporte para curriculum de rho via RhoCurriculumCallback              ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig)                                 ║
# ║  Exports: ~3 funcoes — ver __all__                                        ║
# ║  Ref: docs/physics/perspectivas.md (Estrategia B — alta resistividade)    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""Sampling — Oversampling de alta resistividade e filtro por rho.

Equilibra a representacao de modelos geologicos de alta resistividade
no dataset de treinamento. Por padrao, modelos com rho > 100 Ohm.m
sao sub-representados, fazendo com que a rede ignore gradientes
dessas amostras. Duas estrategias (mutuamente exclusivas):

Estrategia 1 — Oversampling (estatico):
    Repete sequencias com max(rho_h) > threshold N vezes no dataset.
    Efeito: modelo de alta rho aparece 3x mais no treinamento.
    Vantagem: simples, sem callback. Desvantagem: dataset maior.

Estrategia 2 — Curriculum de rho (dinamico):
    Fase easy: so amostras com max(rho) < rho_max_start.
    Fase ramp: rho_max cresce linearmente por epoca.
    Fase full: todas as amostras incluidas.
    Vantagem: progressivo, melhor convergencia.
    Desvantagem: requer callback + filter dinâmico no tf.data.

Motivacao fisica:
    Para f=20 kHz, skin depth cresce com rho: delta ~ sqrt(rho).
    Em rho > 100 Ohm.m, o sinal EM medido se aproxima do
    acoplamento direto (campo livre), tornando a inversao ambigua.
    Oversampling/curriculum forcam a rede a ver mais exemplos
    dessa regiao critica, melhorando a convergencia.

Referencia: Relatorio Estrategia B (alta resistividade).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
__all__ = [
    "compute_rho_max_per_sequence",
    "oversample_high_rho",
    "filter_by_rho_max",
]


# ════════════════════════════════════════════════════════════════════════════
# SECAO: COMPUTACAO DE RHO MAX POR SEQUENCIA
# ════════════════════════════════════════════════════════════════════════════
# Calcula max(rho_h) para cada sequencia no dataset.
# Usado por oversampling (threshold) e curriculum (filtro dinamico).
# rho_h (canal 0 de y) eh usado por ser mais sensivel a camadas
# horizontais — o canal principal para geosteering.
# ──────────────────────────────────────────────────────────────────────────


def compute_rho_max_per_sequence(
    y: np.ndarray,
) -> np.ndarray:
    """Computa max(rho_h) para cada sequencia.

    Args:
        y: Targets 3D (n_seq, seq_len, n_channels) em Ohm.m.
            Canal 0 eh rho_h (horizontal resistivity).
            DEVE ser chamado ANTES de target_scaling (log10).

    Returns:
        np.ndarray: max(rho_h) por sequencia, shape (n_seq,).

    Example:
        >>> y = np.random.uniform(1, 1000, (50, 600, 2))
        >>> rho_max = compute_rho_max_per_sequence(y)
        >>> rho_max.shape  # (50,)

    Note:
        Referenciado em:
            - data/sampling.py: oversample_high_rho() (selecao de indices)
            - data/sampling.py: filter_by_rho_max() (filtro dinamico)
            - data/pipeline.py: DataPipeline.prepare() (pre-target_scaling)
            - tests/test_sampling.py: TestComputeRhoMax
        Ref: Estrategia B (oversampling alta resistividade).
        Usa rho_h (canal 0) — mais sensivel a camadas horizontais.
        CHAMAR ANTES de target_scaling (log10 distorceria o threshold).
    """
    return np.max(y[:, :, 0], axis=1)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: OVERSAMPLING ESTATICO
# ════════════════════════════════════════════════════════════════════════════
# Repete sequencias de alta rho N vezes no dataset.
# Implementado como repeticao de indices (eficiente em memoria).
#
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  ANTES do oversampling:                                          │
#   │    Dataset: [seq0(rho=5), seq1(rho=10), seq2(rho=500)]          │
#   │    Distribuicao: 2 baixa rho, 1 alta rho                        │
#   │                                                                  │
#   │  DEPOIS do oversampling (threshold=100, factor=3):               │
#   │    Dataset: [seq0, seq1, seq2, seq2, seq2]                       │
#   │    Distribuicao: 2 baixa rho, 3 alta rho (3x oversampled)       │
#   │    → Alta rho agora representa 60% do dataset                    │
#   └──────────────────────────────────────────────────────────────────┘
#
# Ref: Estrategia B — oversampling estatico.
# ──────────────────────────────────────────────────────────────────────────


def oversample_high_rho(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oversampling de sequencias com alta resistividade.

    Repete sequencias onde max(rho_h) > threshold N vezes.
    Os arrays x, y, z sao expandidos com as copias. Usado ANTES
    de target_scaling (y em Ohm.m).

    Args:
        x: Features 3D (n_seq, seq_len, n_feat).
        y: Targets 3D (n_seq, seq_len, n_channels) em Ohm.m.
        z: Profundidade 2D (n_seq, seq_len) em metros.
        config: PipelineConfig. Atributos usados:
            - config.rho_oversampling_threshold: limite em Ohm.m.
            - config.rho_oversampling_factor: vezes a repetir.

    Returns:
        Tuple (x_new, y_new, z_new) com sequencias repetidas.

    Example:
        >>> config = PipelineConfig(use_rho_oversampling=True)
        >>> x_new, y_new, z_new = oversample_high_rho(x, y, z, config)
        >>> x_new.shape[0] >= x.shape[0]  # True

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline.prepare() (antes de target_scaling)
            - tests/test_sampling.py: TestOversampleHighRho
        Ref: Estrategia B.
        CHAMAR ANTES de target_scaling para que threshold funcione em Ohm.m.
        Oversampling eh mutuamente exclusivo com curriculum de rho.
    """
    threshold = config.rho_oversampling_threshold
    factor = config.rho_oversampling_factor

    rho_max = compute_rho_max_per_sequence(y)
    high_rho_mask = rho_max > threshold
    n_high = int(high_rho_mask.sum())

    if n_high == 0:
        logger.info(
            "oversample_high_rho: nenhuma sequencia com rho_max > %.1f — "
            "dataset inalterado (%d sequencias)",
            threshold,
            len(y),
        )
        return x, y, z

    # ── Indices das sequencias de alta rho ────────────────────────────
    high_idx = np.where(high_rho_mask)[0]

    # ── Repetir (factor-1) vezes (original ja conta como 1) ──────────
    extra_idx = np.tile(high_idx, factor - 1)

    # ── Concatenar indices originais + extras ─────────────────────────
    all_idx = np.concatenate([np.arange(len(y)), extra_idx])

    logger.info(
        "oversample_high_rho: %d sequencias com rho_max > %.1f "
        "(de %d total). Oversampled %dx → dataset expandido de %d para %d.",
        n_high,
        threshold,
        len(y),
        factor,
        len(y),
        len(all_idx),
    )

    return x[all_idx], y[all_idx], z[all_idx]


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FILTRO POR RHO MAX (CURRICULUM)
# ════════════════════════════════════════════════════════════════════════════
# Filtra sequencias onde max(rho_h) > rho_max_threshold.
# Usado pelo RhoCurriculumCallback para controlar quais amostras
# sao incluidas no treinamento em cada epoca.
#
# O filtro opera sobre rho_max pre-computado (array 1D), evitando
# recomputar max(rho) a cada epoca.
#
# Ref: Estrategia B — curriculum de resistividade.
# ──────────────────────────────────────────────────────────────────────────


def filter_by_rho_max(
    rho_max_per_seq: np.ndarray,
    rho_threshold: float,
) -> np.ndarray:
    """Retorna mascara booleana para sequencias com rho_max <= threshold.

    Usado pelo curriculum de rho para filtrar progressivamente.

    Args:
        rho_max_per_seq: max(rho_h) por sequencia, shape (n_seq,).
        rho_threshold: limite de rho_max para inclusao.

    Returns:
        np.ndarray: Mascara booleana (n_seq,). True = incluir.

    Example:
        >>> rho_max = np.array([10, 50, 200, 5000])
        >>> mask = filter_by_rho_max(rho_max, 100.0)
        >>> mask  # array([True, True, False, False])

    Note:
        Referenciado em:
            - training/callbacks.py: RhoCurriculumCallback.on_epoch_begin()
            - tests/test_sampling.py: TestFilterByRhoMax
        Ref: Estrategia B — curriculum de rho.
    """
    return rho_max_per_seq <= rho_threshold
