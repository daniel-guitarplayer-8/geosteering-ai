# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/metrics.py                                             ║
# ║  Bloco: 8 — Evaluation                                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • MetricsReport dataclass: container para metricas de avaliacao        ║
# ║    • compute_all_metrics: calcula R2, RMSE, MAE, MBE, MAPE (global +     ║
# ║      por componente rho_h/rho_v)                                          ║
# ║    • evaluate_predictions: wrapper com logging formatado                   ║
# ║    • NumPy-only: sem dependencia de TensorFlow para avaliacao             ║
# ║                                                                            ║
# ║  Dependencias: numpy (unica dependencia externa)                          ║
# ║  Exports: ~5 (MetricsReport, compute_all_metrics, evaluate_predictions,  ║
# ║           compute_r2, compute_rmse)                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8.1                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C42)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Metricas de avaliacao para inversao de resistividade.

Calcula metricas padrao (R2, RMSE, MAE, MBE, MAPE) para predicoes
de resistividade em dominio log10, com decomposicao por componente
(rho_h = resistividade horizontal, rho_v = resistividade vertical).

Todas as metricas sao calculadas em NumPy (pos-treinamento, sem TF).
Inputs esperados em dominio log10 (TARGET_SCALING = "log10").

Metricas implementadas:

    ┌──────────┬────────────────────────────────────────────────────────────┐
    │ Metrica  │ Formula                                                    │
    ├──────────┼────────────────────────────────────────────────────────────┤
    │ R2       │ 1 - SS_res / SS_tot                                       │
    │ RMSE     │ sqrt(mean((y - yhat)^2))                                  │
    │ MAE      │ mean(|y - yhat|)                                          │
    │ MBE      │ mean(yhat - y) — bias medio (positivo = superestima)      │
    │ MAPE     │ mean(|y - yhat| / (|y| + eps)) * 100                     │
    └──────────┴────────────────────────────────────────────────────────────┘

ERRATA: eps = 1e-12 para float32 (NUNCA 1e-30).

Note:
    Referenciado em:
        - evaluation/__init__.py: re-exports MetricsReport, compute_all_metrics,
          evaluate_predictions
        - evaluation/comparison.py: compare_models usa MetricsReport
        - tests/test_evaluation.py: testes unitarios
    Ref: docs/ARCHITECTURE_v2.md secao 8.1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Container de metricas ---
    "MetricsReport",
    # --- Funcoes de calculo ---
    "compute_all_metrics",
    "evaluate_predictions",
    "compute_r2",
    "compute_rmse",
    "compute_mae",
    "compute_mbe",
    "compute_mape",
]

# ════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════
# Epsilon seguro para float32 — protege contra divisao por zero em MAPE.
# NUNCA usar 1e-30 (causa subnormais em float32, gradientes explodidos).
# Ref: Errata v5.0.15, IEEE 754 float32 min normal ≈ 1.175e-38.
# ────────────────────────────────────────────────────────────────────────

EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════
# METRICAS INDIVIDUAIS — Funcoes puras NumPy
#
# Cada funcao recebe arrays 1D ou N-D (flattened internamente quando
# necessario) e retorna um escalar float. Inputs esperados em dominio
# log10 (como produzido por apply_target_scaling("log10")).
#
# Ref: docs/reference/metricas.md
# ════════════════════════════════════════════════════════════════════════

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de determinacao R2 (global).

    R2 = 1 - SS_res / SS_tot, onde SS_res = sum((y - yhat)^2) e
    SS_tot = sum((y - mean(y))^2). R2 = 1.0 indica predicao perfeita;
    R2 = 0.0 indica predicao tao boa quanto a media; R2 < 0 indica
    predicao pior que a media.

    Args:
        y_true: Array com valores reais (qualquer shape, flattened).
        y_pred: Array com predicoes (mesmo shape que y_true).

    Returns:
        R2 como float escalar. Retorna 0.0 se SS_tot == 0 (variancia
        nula — todos os targets identicos).

    Raises:
        ValueError: Se y_true e y_pred tem shapes diferentes.

    Note:
        Referenciado em:
            - evaluation/metrics.py: compute_all_metrics (R2 global e por canal)
            - evaluation/comparison.py: compare_models (ranking por R2)
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes incompativeis: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Variancia nula (targets constantes): R² indefinido, retorna 0.0
    if ss_tot == 0.0:
        logger.warning("compute_r2: variancia nula (ss_tot=0). Retornando 0.0.")
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    RMSE = sqrt(mean((y - yhat)^2)). Em dominio log10, RMSE ~ 0.01
    indica erro de ~2.3% em resistividade linear (10^0.01 ≈ 1.023).

    Args:
        y_true: Array com valores reais (qualquer shape, flattened).
        y_pred: Array com predicoes (mesmo shape que y_true).

    Returns:
        RMSE como float escalar.

    Raises:
        ValueError: Se y_true e y_pred tem shapes diferentes.

    Note:
        Referenciado em:
            - evaluation/metrics.py: compute_all_metrics (RMSE global e por canal)
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes incompativeis: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    MAE = mean(|y - yhat|). Menos sensivel a outliers que RMSE. Em
    dominio log10, MAE indica o erro absoluto medio em decadas.

    Args:
        y_true: Array com valores reais (qualquer shape, flattened).
        y_pred: Array com predicoes (mesmo shape que y_true).

    Returns:
        MAE como float escalar.

    Raises:
        ValueError: Se y_true e y_pred tem shapes diferentes.

    Note:
        Referenciado em:
            - evaluation/metrics.py: compute_all_metrics (MAE global)
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes incompativeis: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Bias Error.

    MBE = mean(yhat - y). Positivo indica superestimativa sistematica;
    negativo indica subestimativa. Util para diagnosticar vieses na
    predicao de resistividade (ex: modelo sistematicamente superestima
    rho_v em camadas resistivas).

    Args:
        y_true: Array com valores reais (qualquer shape, flattened).
        y_pred: Array com predicoes (mesmo shape que y_true).

    Returns:
        MBE como float escalar.

    Raises:
        ValueError: Se y_true e y_pred tem shapes diferentes.

    Note:
        Referenciado em:
            - evaluation/metrics.py: compute_all_metrics (MBE global)
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes incompativeis: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )
    return float(np.mean(y_pred - y_true))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    MAPE = mean(|y - yhat| / (|y| + eps)) * 100. Metrica relativa que
    expressa o erro como percentual do valor real. Protegida contra
    divisao por zero via eps = 1e-12.

    Nota: MAPE pode ser inflado para targets proximos de zero em
    dominio log10 (log10(1) = 0). Para resistividades ~ 1 Ohm.m,
    o denominador sera ~ eps, gerando MAPE alto. Interpretar com
    cautela nesse regime.

    Args:
        y_true: Array com valores reais (qualquer shape, flattened).
        y_pred: Array com predicoes (mesmo shape que y_true).

    Returns:
        MAPE como float (percentual, 0 a inf).

    Raises:
        ValueError: Se y_true e y_pred tem shapes diferentes.

    Note:
        Referenciado em:
            - evaluation/metrics.py: compute_all_metrics (MAPE global)
        ERRATA: eps = 1e-12 (NUNCA 1e-30) para estabilidade float32.
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes incompativeis: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + EPS)) * 100.0)


# ════════════════════════════════════════════════════════════════════════
# METRICS REPORT — Container dataclass para metricas
#
# Agrega metricas globais e por componente (rho_h = canal 0,
# rho_v = canal 1) num unico objeto serializavel. Suporta:
#   - to_dict(): exportacao para JSON/logging
#   - summary(): string formatada para terminal/notebook
#
# Inputs esperados em dominio log10 com shape (n, seq, 2):
#   canal 0 = log10(rho_h) — resistividade horizontal
#   canal 1 = log10(rho_v) — resistividade vertical
#
# Ref: docs/reference/metricas.md
# ════════════════════════════════════════════════════════════════════════

@dataclass
class MetricsReport:
    """Container para metricas de avaliacao de inversao de resistividade.

    Armazena metricas globais (sobre ambos canais concatenados) e
    per-component (rho_h e rho_v separados). Todos os valores sao
    calculados em dominio log10 (TARGET_SCALING = "log10").

    Attributes:
        r2: Coeficiente R2 global (ambos canais). 1.0 = perfeito.
        r2_rh: R2 para rho_h (resistividade horizontal, canal 0).
        r2_rv: R2 para rho_v (resistividade vertical, canal 1).
        rmse: RMSE global (ambos canais) em dominio log10.
        rmse_rh: RMSE para rho_h em dominio log10.
        rmse_rv: RMSE para rho_v em dominio log10.
        mae: MAE global em dominio log10.
        mbe: MBE global (positivo = superestimativa).
        mape: MAPE global em percentual.

    Example:
        >>> report = MetricsReport(
        ...     r2=0.98, r2_rh=0.985, r2_rv=0.975,
        ...     rmse=0.02, rmse_rh=0.018, rmse_rv=0.022,
        ...     mae=0.015, mbe=-0.001, mape=1.5,
        ... )
        >>> report.to_dict()["r2"]
        0.98
        >>> print(report.summary())  # doctest: +SKIP

    Note:
        Referenciado em:
            - evaluation/metrics.py: compute_all_metrics (retorno)
            - evaluation/metrics.py: evaluate_predictions (retorno)
            - evaluation/comparison.py: compare_models (input)
            - evaluation/__init__.py: re-export
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
    """
    r2: float
    r2_rh: float
    r2_rv: float
    rmse: float
    rmse_rh: float
    rmse_rv: float
    mae: float
    mbe: float
    mape: float

    def to_dict(self) -> Dict[str, float]:
        """Converte para dicionario (serializacao JSON/logging).

        Retorna dict com chaves correspondentes aos nomes dos atributos
        e valores float. Util para salvar metricas em JSON, CSV ou
        integrar com ferramentas de experiment tracking.

        Returns:
            Dicionario com 9 metricas nomeadas.

        Note:
            Referenciado em:
                - evaluation/comparison.py: compare_models (acesso por metrica)
                - training/callbacks.py: logging de metricas periodicas
            Ref: docs/ARCHITECTURE_v2.md secao 8.1.
        """
        return {
            "r2": self.r2,
            "r2_rh": self.r2_rh,
            "r2_rv": self.r2_rv,
            "rmse": self.rmse,
            "rmse_rh": self.rmse_rh,
            "rmse_rv": self.rmse_rv,
            "mae": self.mae,
            "mbe": self.mbe,
            "mape": self.mape,
        }

    def summary(self) -> str:
        """String formatada para exibicao em terminal/notebook.

        Gera um resumo de metricas em formato tabular com alinhamento
        por coluna. Inclui metricas globais e per-component.

        Returns:
            String multi-linha com metricas formatadas.

        Note:
            Referenciado em:
                - evaluation/metrics.py: evaluate_predictions (logging)
            Ref: docs/ARCHITECTURE_v2.md secao 8.1.
        """
        lines = [
            "=" * 60,
            "  METRICAS DE AVALIACAO (dominio log10)",
            "=" * 60,
            f"  R2  (global) : {self.r2:>10.6f}",
            f"  R2  (rho_h)  : {self.r2_rh:>10.6f}",
            f"  R2  (rho_v)  : {self.r2_rv:>10.6f}",
            "-" * 60,
            f"  RMSE (global): {self.rmse:>10.6f}",
            f"  RMSE (rho_h) : {self.rmse_rh:>10.6f}",
            f"  RMSE (rho_v) : {self.rmse_rv:>10.6f}",
            "-" * 60,
            f"  MAE  (global): {self.mae:>10.6f}",
            f"  MBE  (global): {self.mbe:>10.6f}",
            f"  MAPE (global): {self.mape:>10.4f} %",
            "=" * 60,
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# COMPUTE ALL METRICS — Calcula todas as metricas de avaliacao
#
# Recebe arrays 3D (n, seq, 2) em dominio log10 e retorna MetricsReport
# com metricas globais e per-component. Shape esperado:
#   canal 0 = log10(rho_h) — resistividade horizontal
#   canal 1 = log10(rho_v) — resistividade vertical
#
# Metricas globais sao calculadas sobre ambos canais concatenados.
# Metricas per-component sao calculadas sobre cada canal separadamente.
#
# Ref: docs/ARCHITECTURE_v2.md secao 8.1
# ════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> MetricsReport:
    """Calcula todas as metricas padrao para inversao de resistividade.

    Recebe predicoes e ground-truth em dominio log10, shape (n, seq, 2),
    onde canal 0 = log10(rho_h) e canal 1 = log10(rho_v). Retorna
    MetricsReport com 9 metricas (5 globais + 4 per-component).

    Args:
        y_true: np.ndarray de shape (n, seq, 2). Valores verdadeiros
            de resistividade em dominio log10. Canal 0 = rho_h
            (horizontal), canal 1 = rho_v (vertical).
        y_pred: np.ndarray de shape (n, seq, 2). Predicoes do modelo
            no mesmo dominio e shape que y_true.

    Returns:
        MetricsReport com todas as metricas calculadas.

    Raises:
        ValueError: Se shapes sao incompativeis, ou se a ultima
            dimensao nao e 2 (rho_h + rho_v).

    Example:
        >>> y_true = np.random.randn(10, 600, 2).astype(np.float32)
        >>> y_pred = y_true + np.random.randn(10, 600, 2) * 0.01
        >>> report = compute_all_metrics(y_true, y_pred)
        >>> report.r2 > 0.99
        True

    Note:
        Referenciado em:
            - evaluation/metrics.py: evaluate_predictions (chamada interna)
            - evaluation/__init__.py: re-export
            - tests/test_evaluation.py: testes de shapes e valores
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
        ERRATA: TARGET_SCALING = "log10" (NUNCA "log").
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes incompativeis: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )

    if y_true.ndim != 3 or y_true.shape[-1] != 2:
        raise ValueError(
            f"Shape esperado: (n, seq, 2), recebido: {y_true.shape}. "
            f"Canal 0 = rho_h, canal 1 = rho_v."
        )

    # ── Per-component: rho_h (canal 0) e rho_v (canal 1) ──
    # Separa canais para metricas individuais. Resistividade horizontal
    # (rho_h) e vertical (rho_v) possuem faixas dinamicas diferentes
    # e devem ser avaliadas separadamente para diagnostico.
    y_true_rh = y_true[..., 0]  # shape (n, seq) — log10(rho_h)
    y_pred_rh = y_pred[..., 0]
    y_true_rv = y_true[..., 1]  # shape (n, seq) — log10(rho_v)
    y_pred_rv = y_pred[..., 1]

    # ── R2 per-component ──
    r2_rh = compute_r2(y_true_rh, y_pred_rh)
    r2_rv = compute_r2(y_true_rv, y_pred_rv)

    # ── RMSE per-component ──
    rmse_rh = compute_rmse(y_true_rh, y_pred_rh)
    rmse_rv = compute_rmse(y_true_rv, y_pred_rv)

    # ── Global: ambos canais concatenados ──
    # Flatten para calcular metricas sobre todos os elementos.
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    r2_global = compute_r2(y_true_flat, y_pred_flat)
    rmse_global = compute_rmse(y_true_flat, y_pred_flat)
    mae_global = compute_mae(y_true_flat, y_pred_flat)
    mbe_global = compute_mbe(y_true_flat, y_pred_flat)
    mape_global = compute_mape(y_true_flat, y_pred_flat)

    return MetricsReport(
        r2=r2_global,
        r2_rh=r2_rh,
        r2_rv=r2_rv,
        rmse=rmse_global,
        rmse_rh=rmse_rh,
        rmse_rv=rmse_rv,
        mae=mae_global,
        mbe=mbe_global,
        mape=mape_global,
    )


# ════════════════════════════════════════════════════════════════════════
# EVALUATE PREDICTIONS — Wrapper com logging formatado
#
# Calcula metricas via compute_all_metrics e registra no logging module
# com formatacao estruturada. Identifica o dataset avaliado pelo nome.
#
# Ref: docs/ARCHITECTURE_v2.md secao 8.1
# ════════════════════════════════════════════════════════════════════════

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dataset_name: str = "test",
) -> MetricsReport:
    """Avalia predicoes e registra metricas via logging.

    Wrapper sobre compute_all_metrics que adiciona logging estruturado.
    Loga resumo formatado identificando o dataset avaliado (ex: "test",
    "val_clean", "val_noisy").

    Args:
        y_true: np.ndarray de shape (n, seq, 2). Valores verdadeiros
            em dominio log10.
        y_pred: np.ndarray de shape (n, seq, 2). Predicoes do modelo.
        dataset_name: Nome do dataset para identificacao no log.
            Exemplos: "test", "val_clean", "val_noisy", "holdout".

    Returns:
        MetricsReport com todas as metricas calculadas.

    Raises:
        ValueError: Se shapes sao incompativeis (delegado a
            compute_all_metrics).

    Example:
        >>> y_true = np.random.randn(5, 600, 2).astype(np.float32)
        >>> y_pred = y_true + 0.01
        >>> report = evaluate_predictions(y_true, y_pred, dataset_name="test")
        >>> report.r2 > 0.0
        True

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-export
            - inference/pipeline.py: InferencePipeline.evaluate()
            - training/loop.py: TrainingLoop.evaluate_final()
        Ref: docs/ARCHITECTURE_v2.md secao 8.1.
        Logging: usa logger.info (NUNCA print).
    """
    logger.info("Avaliando predicoes no dataset '%s'...", dataset_name)
    logger.info(
        "  Shapes: y_true=%s, y_pred=%s",
        y_true.shape,
        y_pred.shape,
    )

    report = compute_all_metrics(y_true, y_pred)

    # ── Logging formatado ──
    logger.info(
        "Metricas [%s]:\n%s",
        dataset_name,
        report.summary(),
    )

    # ── Diagnostico de bias ──
    # MBE > 0: modelo superestima resistividade (tendencia para rho alto)
    # MBE < 0: modelo subestima resistividade (tendencia para rho baixo)
    if abs(report.mbe) > 0.05:
        logger.warning(
            "Bias significativo detectado no dataset '%s': MBE=%.4f "
            "(limiar: 0.05). Investigar se ha desbalanceamento nas "
            "faixas de resistividade do treinamento.",
            dataset_name,
            report.mbe,
        )

    return report
