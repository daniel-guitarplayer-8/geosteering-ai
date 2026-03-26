# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.evaluation                                     ║
# ║  Bloco: 8 — Evaluation                                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║                                                                            ║
# ║  Modulos: metrics.py, comparison.py                                       ║
# ║  Proposito: Metricas de avaliacao pos-treinamento e comparacao de modelos ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8                                     ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (2 modulos)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de avaliacao — metricas e comparacao de modelos.

Fornece funcionalidades de avaliacao pos-treinamento para o pipeline
de inversao geofisica:

    predicoes → metricas (R2, RMSE, MAE, MBE, MAPE) → comparacao

Modulos:
    metrics         MetricsReport dataclass, compute_all_metrics,
                    evaluate_predictions (com logging formatado)
    comparison      ComparisonResult dataclass, compare_models
                    (ranking multi-modelo por metrica)

Principios:
    - NumPy-only: metricas pos-treinamento nao requerem TensorFlow
    - Metricas por componente: rho_h (canal 0) e rho_v (canal 1) separados
    - Logging estruturado: NUNCA print(), sempre logging module
    - PipelineConfig como parametro (para acesso a metadados, se necessario)

Referencia: docs/ARCHITECTURE_v2.md secao 8.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports: metrics.py — metricas de avaliacao pos-treinamento
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.metrics import (
    MetricsReport,
    compute_all_metrics,
    evaluate_predictions,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: comparison.py — comparacao e ranking de modelos
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.comparison import (
    ComparisonResult,
    compare_models,
)

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente por modulo
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- metrics.py: metricas pos-treinamento ---
    "MetricsReport",
    "compute_all_metrics",
    "evaluate_predictions",
    # --- comparison.py: comparacao de modelos ---
    "ComparisonResult",
    "compare_models",
]
