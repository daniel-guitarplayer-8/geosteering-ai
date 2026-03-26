# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/realtime_comparison.py                                ║
# ║  Bloco: 8 — Evaluation (C70)                                            ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • ModeComparisonResult dataclass: container para metricas comparativas ║
# ║    • compare_modes: quantifica degradacao do modo offline para realtime   ║
# ║    • Computa delta R2, delta RMSE e latencia media de inferencia          ║
# ║    • NumPy-only: sem dependencia de TensorFlow para avaliacao             ║
# ║                                                                            ║
# ║  Dependencias: numpy (unica dependencia externa)                          ║
# ║  Exports: ~2 (ModeComparisonResult, compare_modes)                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8.7                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C70)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Comparacao de desempenho entre modos offline e realtime.

Quantifica a degradacao de metricas ao migrar da inferencia offline
(acausal, sequencia completa) para inferencia realtime (causal,
sliding-window). Util para determinar se o modelo atende aos
requisitos de qualidade em cenarios de geosteering em tempo real.

Metricas comparadas:

    ┌──────────────┬──────────────────────────────────────────────────────┐
    │ Metrica      │ Interpretacao                                        │
    ├──────────────┼──────────────────────────────────────────────────────┤
    │ r2_offline   │ R2 com predicao acausal (sequencia completa)         │
    │ r2_realtime  │ R2 com predicao causal (sliding window)              │
    │ delta_r2     │ r2_offline - r2_realtime (>0 = degradacao)           │
    │ rmse_offline │ RMSE da predicao offline (log10 decades)             │
    │ rmse_realtime│ RMSE da predicao realtime (log10 decades)            │
    │ delta_rmse   │ rmse_realtime - rmse_offline (>0 = degradacao)       │
    │ latency_ms   │ Latencia media de inferencia no modo realtime        │
    └──────────────┴──────────────────────────────────────────────────────┘

ERRATA: eps = 1e-12 para float32 (NUNCA 1e-30).

Note:
    Referenciado em:
        - evaluation/__init__.py: re-exports ModeComparisonResult, compare_modes
        - evaluation/geosteering_report.py: alimenta secao 3 (Offline vs Realtime)
        - tests/test_evaluation.py: testes unitarios
    Ref: docs/ARCHITECTURE_v2.md secao 8.7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Container de comparacao offline vs realtime ---
    "ModeComparisonResult",
    # --- Funcao principal de comparacao ---
    "compare_modes",
]

# ════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════
# Epsilon seguro para float32 — protege contra divisao por zero em R2.
# NUNCA usar 1e-30 (causa subnormais em float32, gradientes explodidos).
# Ref: Errata v5.0.15, IEEE 754 float32 min normal ≈ 1.175e-38.
# ────────────────────────────────────────────────────────────────────────

EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════
# DATACLASS: ModeComparisonResult
#
# Container imutavel com metricas comparativas entre os modos de
# inferencia offline (acausal) e realtime (causal). Inclui deltas
# e latencia media para avaliacao rapida de viabilidade.
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │ ModeComparisonResult                                                │
# ├──────────────────────────────────────────────────────────────────────┤
# │  r2_offline    float    R2 offline (acausal)                        │
# │  r2_realtime   float    R2 realtime (causal)                        │
# │  delta_r2      float    r2_offline - r2_realtime (degradacao)       │
# │  rmse_offline  float    RMSE offline (log10)                        │
# │  rmse_realtime float    RMSE realtime (log10)                       │
# │  delta_rmse    float    rmse_realtime - rmse_offline (degradacao)   │
# │  latency_ms    float    latencia media em ms (-1.0 se desconhecida)│
# └──────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModeComparisonResult:
    """Resultado da comparacao entre modos offline e realtime.

    Container imutavel contendo metricas pareadas (R2, RMSE) para
    ambos os modos de inferencia, deltas de degradacao, e latencia
    media de inferencia no modo realtime.

    Attributes:
        r2_offline: Coeficiente de determinacao R2 no modo offline
            (acausal, sequencia completa). Valor ideal: 1.0.
        r2_realtime: Coeficiente de determinacao R2 no modo realtime
            (causal, sliding window). Esperado <= r2_offline.
        delta_r2: Degradacao de R2 (r2_offline - r2_realtime).
            Positivo indica perda de qualidade no modo realtime.
        rmse_offline: RMSE no modo offline (em log10 decades).
            Valor ideal: 0.0.
        rmse_realtime: RMSE no modo realtime (em log10 decades).
            Esperado >= rmse_offline.
        delta_rmse: Aumento de RMSE (rmse_realtime - rmse_offline).
            Positivo indica degradacao de precisao no modo realtime.
        latency_ms: Latencia media de inferencia no modo realtime
            em milissegundos. -1.0 se nao informada.

    Example:
        >>> result = compare_modes(y_true, y_pred_offline, y_pred_realtime)
        >>> result.delta_r2  # degradacao R2
        0.023
        >>> result.latency_ms
        -1.0

    Note:
        Referenciado em:
            - evaluation/realtime_comparison.py: compare_modes retorna esta classe
            - evaluation/geosteering_report.py: secao 3 (Offline vs Realtime)
        Ref: docs/ARCHITECTURE_v2.md secao 8.7.
    """

    r2_offline: float
    r2_realtime: float
    delta_r2: float
    rmse_offline: float
    rmse_realtime: float
    delta_rmse: float
    latency_ms: float

    def to_dict(self) -> dict:
        """Converte resultado para dicionario serializavel.

        Returns:
            Dicionario com todos os campos da comparacao.

        Note:
            Referenciado em:
                - evaluation/geosteering_report.py: generate_geosteering_report
            Ref: docs/ARCHITECTURE_v2.md secao 8.7.
        """
        return {
            "r2_offline": self.r2_offline,
            "r2_realtime": self.r2_realtime,
            "delta_r2": self.delta_r2,
            "rmse_offline": self.rmse_offline,
            "rmse_realtime": self.rmse_realtime,
            "delta_rmse": self.delta_rmse,
            "latency_ms": self.latency_ms,
        }


# ════════════════════════════════════════════════════════════════════════
# FUNCOES AUXILIARES — R2 e RMSE (NumPy puro)
#
# Reimplementadas localmente para evitar dependencia circular com
# evaluation/metrics.py (que pode importar coisas desnecessarias).
# Mesma formula, apenas float escalar.
#
# Ref: docs/reference/metricas.md
# ════════════════════════════════════════════════════════════════════════

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 escalar sobre arrays flattened.

    R2 = 1 - SS_res / SS_tot, com protecao contra SS_tot = 0.

    Args:
        y_true: Array com valores verdadeiros (qualquer shape).
        y_pred: Array com predicoes (mesmo shape que y_true).

    Returns:
        Escalar R2 (float). Retorna 0.0 se SS_tot < EPS.

    Note:
        Referenciado em:
            - evaluation/realtime_comparison.py: compare_modes
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
    """
    yt = y_true.flatten().astype(np.float64)
    yp = y_pred.flatten().astype(np.float64)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot < EPS:
        # Variancia nula — R2 indefinido, retorna 0.0
        logger.warning(
            "SS_tot < EPS (%.2e). Variancia nula — R2 retornado como 0.0.",
            ss_tot,
        )
        return 0.0
    return 1.0 - ss_res / ss_tot


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE escalar sobre arrays flattened.

    RMSE = sqrt(mean((y - yhat)^2)).

    Args:
        y_true: Array com valores verdadeiros (qualquer shape).
        y_pred: Array com predicoes (mesmo shape que y_true).

    Returns:
        Escalar RMSE (float).

    Note:
        Referenciado em:
            - evaluation/realtime_comparison.py: compare_modes
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
    """
    yt = y_true.flatten().astype(np.float64)
    yp = y_pred.flatten().astype(np.float64)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


# ════════════════════════════════════════════════════════════════════════
# FUNCAO PRINCIPAL: compare_modes
#
# Compara predicoes offline vs realtime usando R2 e RMSE. Computa
# deltas de degradacao e registra resultados via logging.
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │ Fluxo de comparacao                                                │
# │                                                                     │
# │  y_true ─┬─ R2(y_true, y_offline) ─── r2_offline ──┐              │
# │          │                                           │ delta_r2    │
# │          └─ R2(y_true, y_realtime) ── r2_realtime ──┘              │
# │                                                                     │
# │  y_true ─┬─ RMSE(y_true, y_offline) ─ rmse_offline ──┐            │
# │          │                                             │ delta_rmse│
# │          └─ RMSE(y_true, y_realtime) ─ rmse_realtime ─┘            │
# └──────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def compare_modes(
    y_true: np.ndarray,
    y_pred_offline: np.ndarray,
    y_pred_realtime: np.ndarray,
    *,
    latency_ms: Optional[float] = None,
    config: Optional[PipelineConfig] = None,
) -> ModeComparisonResult:
    """Quantifica degradacao de desempenho do modo offline para realtime.

    Compara predicoes offline (acausal, sequencia completa) contra
    predicoes realtime (causal, sliding window) usando R2 e RMSE.
    Opcionalmente recebe latencia media de inferencia.

    Os deltas sao definidos de forma que valores positivos indicam
    degradacao no modo realtime:
        - delta_r2 = r2_offline - r2_realtime (R2 diminui)
        - delta_rmse = rmse_realtime - rmse_offline (RMSE aumenta)

    Args:
        y_true: Array (N, seq_len, n_channels) com resistividade
            verdadeira em dominio log10. Canal 0 = rho_h, canal 1 = rho_v.
        y_pred_offline: Array (N, seq_len, n_channels) com predicoes
            do modo offline (acausal). Mesmo shape de ``y_true``.
        y_pred_realtime: Array (N, seq_len, n_channels) com predicoes
            do modo realtime (causal). Mesmo shape de ``y_true``.
        latency_ms: Latencia media de inferencia no modo realtime,
            em milissegundos. Se None, registrado como -1.0.
        config: PipelineConfig opcional. Se fornecido, usa
            ``config.model_type`` para logging contextualizado.

    Returns:
        ModeComparisonResult com metricas pareadas e deltas.

    Raises:
        ValueError: Se shapes de y_true, y_pred_offline e y_pred_realtime
            forem incompativeis.

    Example:
        >>> import numpy as np
        >>> y = np.random.randn(50, 600, 2)
        >>> off = y + np.random.randn(50, 600, 2) * 0.05
        >>> rt = y + np.random.randn(50, 600, 2) * 0.08
        >>> result = compare_modes(y, off, rt, latency_ms=12.3)
        >>> result.delta_r2 > 0  # realtime degrada R2
        True

    Note:
        Metricas em dominio log10 (TARGET_SCALING = "log10").
        Referenciado em:
            - evaluation/__init__.py: re-export
            - evaluation/geosteering_report.py: secao 3
            - tests/test_evaluation.py: test_compare_modes
        Ref: docs/ARCHITECTURE_v2.md secao 8.7.
    """
    # ── Validacao de shapes ──────────────────────────────────────────
    if y_true.shape != y_pred_offline.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"y_pred_offline={y_pred_offline.shape}"
        )
    if y_true.shape != y_pred_realtime.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"y_pred_realtime={y_pred_realtime.shape}"
        )

    # ── Identificacao do modelo para logging ─────────────────────────
    model_label = "desconhecido"
    if config is not None:
        model_label = getattr(config, "model_type", "desconhecido")

    logger.info(
        "Comparando modos offline vs realtime — modelo=%s, "
        "shape=%s, N=%d amostras",
        model_label, y_true.shape, y_true.shape[0],
    )

    # ── Computar metricas offline ────────────────────────────────────
    r2_off = _compute_r2(y_true, y_pred_offline)
    rmse_off = _compute_rmse(y_true, y_pred_offline)

    # ── Computar metricas realtime ───────────────────────────────────
    r2_rt = _compute_r2(y_true, y_pred_realtime)
    rmse_rt = _compute_rmse(y_true, y_pred_realtime)

    # ── Computar deltas de degradacao ────────────────────────────────
    # delta_r2 > 0 indica que offline e melhor (R2 diminuiu no realtime)
    delta_r2 = r2_off - r2_rt
    # delta_rmse > 0 indica degradacao (RMSE aumentou no realtime)
    delta_rmse = rmse_rt - rmse_off

    # ── Latencia ─────────────────────────────────────────────────────
    effective_latency = latency_ms if latency_ms is not None else -1.0

    # ── Construir resultado ──────────────────────────────────────────
    result = ModeComparisonResult(
        r2_offline=r2_off,
        r2_realtime=r2_rt,
        delta_r2=delta_r2,
        rmse_offline=rmse_off,
        rmse_realtime=rmse_rt,
        delta_rmse=delta_rmse,
        latency_ms=effective_latency,
    )

    # ── Logging detalhado ────────────────────────────────────────────
    logger.info(
        "Comparacao concluida — modelo=%s:\n"
        "  R2  offline=%.6f  realtime=%.6f  delta=%.6f\n"
        "  RMSE offline=%.6f  realtime=%.6f  delta=%.6f\n"
        "  Latencia: %.2f ms",
        model_label,
        r2_off, r2_rt, delta_r2,
        rmse_off, rmse_rt, delta_rmse,
        effective_latency,
    )

    # ── Avisos de degradacao significativa ────────────────────────────
    if delta_r2 > 0.05:
        logger.warning(
            "Degradacao significativa de R2: delta=%.4f (>0.05). "
            "Considerar ajuste da janela causal ou re-treinamento.",
            delta_r2,
        )
    if delta_rmse > 0.1:
        logger.warning(
            "Degradacao significativa de RMSE: delta=%.4f (>0.1 decades). "
            "Considerar ajuste de hiperparametros para modo causal.",
            delta_rmse,
        )

    return result
