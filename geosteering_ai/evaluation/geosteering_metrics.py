# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/geosteering_metrics.py                                ║
# ║  Bloco: 8 — Evaluation (C71)                                            ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • GeoMetrics dataclass: metricas especificas para geosteering          ║
# ║    • compute_geosteering_metrics: calcula DTB error, look-ahead accuracy  ║
# ║    • Deteccao de interfaces geologicas via gradiente de resistividade     ║
# ║    • NumPy-only: sem dependencia de TensorFlow para avaliacao             ║
# ║                                                                            ║
# ║  Dependencias: numpy (unica dependencia externa)                          ║
# ║  Exports: ~2 (GeoMetrics, compute_geosteering_metrics)                   ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8.8                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C71)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Metricas especificas para geosteering — DTB, look-ahead, interfaces.

Complementa metricas basicas (R2, RMSE) e avancadas (interface_metrics,
coherence) com metricas especificamente relevantes para operacoes de
geosteering em tempo real:

    ┌─────────────────────────────┬────────────────────────────────────────┐
    │ Metrica                     │ Proposito                              │
    ├─────────────────────────────┼────────────────────────────────────────┤
    │ dtb_error_mean              │ Erro medio DTB (Distance to Boundary) │
    │ dtb_error_std               │ Desvio-padrao do erro DTB             │
    │ look_ahead_accuracy         │ Fracao de interfaces detectadas       │
    │                             │ N pontos antes (antecipacao)           │
    │ n_interfaces_detected       │ Interfaces corretamente antecipadas   │
    │ n_interfaces_total          │ Total de interfaces verdadeiras       │
    └─────────────────────────────┴────────────────────────────────────────┘

    Fluxo de avaliacao geosteering:

    ┌─────────────┐     ┌───────────────────┐     ┌──────────────────┐
    │  y_true     │────→│ detectar          │────→│  GeoMetrics      │
    │  y_pred     │     │ interfaces (grad) │     │  (dataclass)     │
    │  dtb_true?  │────→│ calcular DTB err  │     │                  │
    │  dtb_pred?  │     │ calcular look-ah. │     │                  │
    └─────────────┘     └───────────────────┘     └──────────────────┘

DTB (Distance to Boundary) e um label derivado da geometria do modelo
geologico que indica a distancia vertical ate a interface mais proxima.
Quando dtb_true/dtb_pred nao sao fornecidos, os campos DTB recebem
NaN e sao omitidos da avaliacao.

ERRATA: eps = 1e-12 para float32 (NUNCA 1e-30).

Note:
    Referenciado em:
        - evaluation/__init__.py: re-exports GeoMetrics, compute_geosteering_metrics
        - evaluation/geosteering_report.py: secao 2 (Geosteering Metrics)
        - tests/test_evaluation.py: testes unitarios
    Ref: docs/ARCHITECTURE_v2.md secao 8.8.
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
    # --- Container de metricas geosteering ---
    "GeoMetrics",
    # --- Funcao principal ---
    "compute_geosteering_metrics",
]

# ════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════
# Epsilon seguro para float32 — protege contra divisao por zero.
# NUNCA usar 1e-30 (causa subnormais em float32, gradientes explodidos).
# Ref: Errata v5.0.15, IEEE 754 float32 min normal ≈ 1.175e-38.
# ────────────────────────────────────────────────────────────────────────

EPS = 1e-12

# Gradiente minimo (em log10 decades/ponto) para considerar uma
# transicao como interface geologica. Valor calibrado empiricamente
# para modelos 1D com resistividade em log10.
_DEFAULT_INTERFACE_THRESHOLD = 0.5


# ════════════════════════════════════════════════════════════════════════
# DATACLASS: GeoMetrics
#
# Container para metricas especificas de geosteering. Campos DTB
# sao NaN quando dtb_true/dtb_pred nao sao fornecidos.
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │ GeoMetrics                                                          │
# ├──────────────────────────────────────────────────────────────────────┤
# │  dtb_error_mean        float    erro medio DTB (scaled domain)     │
# │  dtb_error_std         float    desvio-padrao DTB                  │
# │  look_ahead_accuracy   float    fracao interfaces antecipadas      │
# │  n_interfaces_detected int      interfaces corretamente detectadas │
# │  n_interfaces_total    int      total de interfaces verdadeiras    │
# └──────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GeoMetrics:
    """Metricas especificas para operacoes de geosteering.

    Container imutavel com metricas relevantes para geosteering:
    erro DTB (Distance-to-Boundary), acuracia de antecipacao de
    interfaces (look-ahead), e contagem de interfaces detectadas.

    Attributes:
        dtb_error_mean: Erro medio de DTB no dominio escalado (log10).
            NaN se dtb_true/dtb_pred nao foram fornecidos.
        dtb_error_std: Desvio-padrao do erro DTB no dominio escalado.
            NaN se dtb_true/dtb_pred nao foram fornecidos.
        look_ahead_accuracy: Fracao de interfaces geologicas verdadeiras
            que foram antecipadas pelo modelo no minimo ``look_ahead_window``
            pontos antes da posicao real. Valor entre 0.0 e 1.0.
        n_interfaces_detected: Numero de interfaces verdadeiras
            corretamente antecipadas pelo modelo.
        n_interfaces_total: Numero total de interfaces verdadeiras
            detectadas nos dados de referencia (y_true).

    Example:
        >>> metrics = compute_geosteering_metrics(y_true, y_pred)
        >>> metrics.look_ahead_accuracy
        0.85
        >>> metrics.n_interfaces_total
        20

    Note:
        Referenciado em:
            - evaluation/geosteering_metrics.py: compute_geosteering_metrics
            - evaluation/geosteering_report.py: secao 2
        Ref: docs/ARCHITECTURE_v2.md secao 8.8.
    """

    dtb_error_mean: float
    dtb_error_std: float
    look_ahead_accuracy: float
    n_interfaces_detected: int
    n_interfaces_total: int

    def to_dict(self) -> dict:
        """Converte metricas para dicionario serializavel.

        Returns:
            Dicionario com todos os campos.

        Note:
            Referenciado em:
                - evaluation/geosteering_report.py: generate_geosteering_report
            Ref: docs/ARCHITECTURE_v2.md secao 8.8.
        """
        return {
            "dtb_error_mean": self.dtb_error_mean,
            "dtb_error_std": self.dtb_error_std,
            "look_ahead_accuracy": self.look_ahead_accuracy,
            "n_interfaces_detected": self.n_interfaces_detected,
            "n_interfaces_total": self.n_interfaces_total,
        }


# ════════════════════════════════════════════════════════════════════════
# FUNCOES AUXILIARES — Deteccao de interfaces e look-ahead
#
# Interfaces geologicas sao detectadas via gradiente da resistividade
# ao longo da sequencia. Um ponto e considerado interface se
# |grad(y)| > threshold.
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │ Deteccao de Interface via Gradiente                                │
# │                                                                     │
# │  y_true[:,i] ───────────┐                                          │
# │                          │ grad = |y[i+1] - y[i]|                  │
# │  y_true[:,i+1] ─────────┘                                          │
# │                                                                     │
# │  Se grad > threshold → posicao i e interface                       │
# └──────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def _detect_interfaces(
    y: np.ndarray,
    threshold: float,
) -> list:
    """Detecta posicoes de interfaces geologicas via gradiente.

    Interface definida como ponto onde |y[i+1] - y[i]| > threshold
    em qualquer canal de resistividade.

    Args:
        y: Array 2D (seq_len, n_channels) com resistividade em log10.
        threshold: Limiar minimo de gradiente para considerar interface.

    Returns:
        Lista de indices (int) onde interfaces foram detectadas.

    Note:
        Referenciado em:
            - evaluation/geosteering_metrics.py: _compute_look_ahead
        Ref: docs/ARCHITECTURE_v2.md secao 8.8.
    """
    if y.ndim == 1:
        # Canal unico — expandir para 2D
        y = y[:, np.newaxis]

    # Gradiente absoluto entre pontos consecutivos
    grad = np.abs(np.diff(y, axis=0))  # (seq_len-1, n_channels)
    # Interface se qualquer canal excede o threshold
    max_grad = np.max(grad, axis=-1)  # (seq_len-1,)
    # Indices onde gradiente > threshold
    interface_positions = np.where(max_grad > threshold)[0].tolist()

    return interface_positions


def _compute_look_ahead(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    look_ahead_window: int,
    interface_threshold: float,
) -> tuple:
    """Computa acuracia de antecipacao de interfaces (look-ahead).

    Para cada interface verdadeira na posicao ``p``, verifica se o
    modelo prediz uma mudanca significativa (gradiente > threshold)
    em algum ponto do intervalo ``[p - look_ahead_window, p]``.

    Args:
        y_true: Array (N, seq_len, n_channels) com valores verdadeiros.
        y_pred: Array (N, seq_len, n_channels) com predicoes.
        look_ahead_window: Numero de pontos antes da interface para
            verificar antecipacao.
        interface_threshold: Limiar de gradiente para deteccao.

    Returns:
        Tupla (n_detected, n_total, accuracy) onde:
            - n_detected: interfaces antecipadas corretamente
            - n_total: total de interfaces verdadeiras
            - accuracy: n_detected / n_total (0.0 se n_total = 0)

    Note:
        Referenciado em:
            - evaluation/geosteering_metrics.py: compute_geosteering_metrics
        Ref: docs/ARCHITECTURE_v2.md secao 8.8.
    """
    n_samples = y_true.shape[0]
    total_interfaces = 0
    detected_interfaces = 0

    for i in range(n_samples):
        # Detectar interfaces verdadeiras na amostra i
        true_interfaces = _detect_interfaces(
            y_true[i], interface_threshold,
        )
        total_interfaces += len(true_interfaces)

        if not true_interfaces:
            continue

        # Detectar interfaces preditas na amostra i
        pred_interfaces = set(_detect_interfaces(
            y_pred[i], interface_threshold,
        ))

        # Para cada interface verdadeira, verificar se o modelo
        # antecipou (prediz interface em [pos - window, pos])
        for pos in true_interfaces:
            # Janela de antecipacao: [pos - window, pos] (inclusivo)
            window_start = max(0, pos - look_ahead_window)
            # Verificar se alguma interface predita cai na janela
            anticipated = any(
                p in pred_interfaces
                for p in range(window_start, pos + 1)
            )
            if anticipated:
                detected_interfaces += 1

    # Acuracia: fracao de interfaces antecipadas
    accuracy = 0.0
    if total_interfaces > 0:
        accuracy = detected_interfaces / total_interfaces

    return detected_interfaces, total_interfaces, accuracy


# ════════════════════════════════════════════════════════════════════════
# FUNCAO PRINCIPAL: compute_geosteering_metrics
#
# Calcula DTB error (se fornecido), look-ahead accuracy, e contagem
# de interfaces. Retorna GeoMetrics dataclass.
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │ Pipeline de metricas geosteering                                   │
# │                                                                     │
# │  y_true, y_pred ──→ _compute_look_ahead() ──→ look-ahead accuracy │
# │                                                                     │
# │  dtb_true, dtb_pred ──→ mean/std(|error|) ──→ DTB error stats     │
# │  (opcional)                                                         │
# │                                                                     │
# │  Todos ──→ GeoMetrics(...)                                          │
# └──────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def compute_geosteering_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dtb_true: Optional[np.ndarray] = None,
    dtb_pred: Optional[np.ndarray] = None,
    look_ahead_window: int = 10,
    interface_threshold: float = _DEFAULT_INTERFACE_THRESHOLD,
    config: Optional[PipelineConfig] = None,
) -> GeoMetrics:
    """Computa metricas especificas para geosteering.

    Calcula acuracia de antecipacao de interfaces (look-ahead) e
    opcionalmente erro de DTB (Distance-to-Boundary). As metricas
    quantificam a capacidade do modelo de antecipar mudancas
    geologicas em cenarios de geosteering.

    A deteccao de interfaces usa o gradiente absoluto da resistividade:
    um ponto e classificado como interface se |y[i+1] - y[i]| >
    ``interface_threshold`` em qualquer canal.

    Para look-ahead, uma interface verdadeira na posicao ``p`` e
    considerada antecipada se o modelo prediz interface em algum
    ponto do intervalo ``[p - look_ahead_window, p]``.

    Args:
        y_true: Array (N, seq_len, n_channels) com resistividade
            verdadeira em dominio log10. Canal 0 = rho_h, canal 1 = rho_v.
        y_pred: Array (N, seq_len, n_channels) com predicoes em
            dominio log10. Mesmo shape de ``y_true``.
        dtb_true: Array opcional (N, seq_len) com DTB verdadeiro
            (Distance-to-Boundary) em dominio escalado. Se None,
            campos DTB no resultado serao NaN.
        dtb_pred: Array opcional (N, seq_len) com DTB predito.
            Deve ser fornecido junto com ``dtb_true``.
        look_ahead_window: Numero de pontos antes da interface
            para verificar antecipacao. Default: 10.
        interface_threshold: Limiar minimo de gradiente (em log10
            decades/ponto) para considerar uma transicao como
            interface geologica. Default: 0.5.
        config: PipelineConfig opcional. Se fornecido, usa
            ``config.model_type`` para logging contextualizado.

    Returns:
        GeoMetrics com DTB error (ou NaN), look-ahead accuracy e
        contagem de interfaces.

    Raises:
        ValueError: Se shapes de y_true e y_pred forem incompativeis.
        ValueError: Se dtb_true fornecido sem dtb_pred (ou vice-versa).

    Example:
        >>> import numpy as np
        >>> y_true = np.random.randn(50, 600, 2)
        >>> y_pred = y_true + np.random.randn(50, 600, 2) * 0.1
        >>> metrics = compute_geosteering_metrics(y_true, y_pred)
        >>> 0.0 <= metrics.look_ahead_accuracy <= 1.0
        True

    Note:
        Metricas em dominio log10 (TARGET_SCALING = "log10").
        Referenciado em:
            - evaluation/__init__.py: re-export
            - evaluation/geosteering_report.py: secao 2
            - tests/test_evaluation.py: test_compute_geosteering_metrics
        Ref: docs/ARCHITECTURE_v2.md secao 8.8.
    """
    # ── Validacao de shapes ──────────────────────────────────────────
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    # ── Validacao DTB: ambos ou nenhum ───────────────────────────────
    if (dtb_true is None) != (dtb_pred is None):
        raise ValueError(
            "dtb_true e dtb_pred devem ser ambos fornecidos ou ambos None. "
            f"Recebido: dtb_true={'fornecido' if dtb_true is not None else 'None'}, "
            f"dtb_pred={'fornecido' if dtb_pred is not None else 'None'}"
        )

    # ── Identificacao do modelo para logging ─────────────────────────
    model_label = "desconhecido"
    if config is not None:
        model_label = getattr(config, "model_type", "desconhecido")

    logger.info(
        "Computando metricas geosteering — modelo=%s, shape=%s, "
        "look_ahead=%d, threshold=%.3f",
        model_label, y_true.shape, look_ahead_window, interface_threshold,
    )

    # ── DTB Error (se fornecido) ─────────────────────────────────────
    if dtb_true is not None and dtb_pred is not None:
        dtb_error = np.abs(
            dtb_true.astype(np.float64) - dtb_pred.astype(np.float64)
        )
        dtb_mean = float(np.mean(dtb_error))
        dtb_std = float(np.std(dtb_error))
        logger.info(
            "DTB error — mean=%.6f, std=%.6f (N=%d pontos)",
            dtb_mean, dtb_std, dtb_error.size,
        )
    else:
        # DTB nao fornecido — campos NaN
        dtb_mean = float("nan")
        dtb_std = float("nan")
        logger.info("DTB arrays nao fornecidos — campos DTB serao NaN.")

    # ── Look-ahead Accuracy ──────────────────────────────────────────
    n_detected, n_total, accuracy = _compute_look_ahead(
        y_true, y_pred, look_ahead_window, interface_threshold,
    )

    logger.info(
        "Look-ahead accuracy — detectadas=%d/%d, accuracy=%.4f, "
        "window=%d pontos",
        n_detected, n_total, accuracy, look_ahead_window,
    )

    # ── Construir resultado ──────────────────────────────────────────
    result = GeoMetrics(
        dtb_error_mean=dtb_mean,
        dtb_error_std=dtb_std,
        look_ahead_accuracy=accuracy,
        n_interfaces_detected=n_detected,
        n_interfaces_total=n_total,
    )

    logger.info(
        "Metricas geosteering concluidas — modelo=%s: "
        "DTB_mean=%.4f, DTB_std=%.4f, look_ahead=%.4f (%d/%d)",
        model_label,
        result.dtb_error_mean, result.dtb_error_std,
        result.look_ahead_accuracy,
        result.n_interfaces_detected, result.n_interfaces_total,
    )

    return result
