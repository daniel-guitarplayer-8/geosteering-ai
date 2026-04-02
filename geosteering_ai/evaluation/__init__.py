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
# ║  Modulos: metrics.py, comparison.py, advanced.py, predict.py,            ║
# ║           manifest.py, report.py, realtime_comparison.py,               ║
# ║           geosteering_metrics.py, geosteering_report.py, dod.py        ║
# ║  Proposito: Metricas de avaliacao pos-treinamento e comparacao de modelos ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8                                     ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (2 modulos)                  ║
# ║    v2.0.0 (2026-03) — Adicionados advanced.py (C50-C55) e predict.py    ║
# ║    v2.0.0 (2026-03) — Adicionados manifest.py (C64) e report.py (C65)  ║
# ║    v2.0.0 (2026-03) — Adicionados realtime_comparison.py (C70),       ║
# ║                        geosteering_metrics.py (C71),                    ║
# ║                        geosteering_report.py (C73)                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de avaliacao — metricas, comparacao, analises avancadas e predicoes.

Fornece funcionalidades de avaliacao pos-treinamento para o pipeline
de inversao geofisica:

    predicoes → metricas (R2, RMSE, MAE, MBE, MAPE) → comparacao
                    │
              metricas avancadas: interfaces, bandas, anisotropia,
              perfil espacial, coerencia TIV, estabilidade

Modulos:
    metrics         MetricsReport dataclass, compute_all_metrics,
                    evaluate_predictions (com logging formatado)
    comparison      ComparisonResult dataclass, compare_models
                    (ranking multi-modelo por metrica)
    advanced        InterfaceReport, CoherenceReport, StabilityReport,
                    interface_metrics, error_by_resistivity_band,
                    error_by_anisotropy, spatial_error_profile,
                    physical_coherence_check, stability_analysis
    predict         PredictionResult, predict_test
                    (predicoes com scaling inverso log10 → Ohm.m)
    manifest        create_manifest, save_manifest, load_manifest
                    (manifesto JSON do experimento para reprodutibilidade)
    report          generate_report
                    (relatorio Markdown automatizado pos-treinamento)
    realtime_comparison
                    ModeComparisonResult, compare_modes
                    (comparacao offline vs realtime: delta R2/RMSE, latencia)
    geosteering_metrics
                    GeoMetrics, compute_geosteering_metrics
                    (DTB error, look-ahead accuracy, deteccao de interfaces)
    geosteering_report
                    generate_geosteering_report
                    (relatorio Markdown especifico para geosteering)

Principios:
    - NumPy-only: metricas pos-treinamento nao requerem TensorFlow
      (exceto predict_test e stability_analysis que fazem lazy import)
    - Metricas por componente: rho_h (canal 0) e rho_v (canal 1) separados
    - Logging estruturado: NUNCA print(), sempre logging module
    - PipelineConfig como parametro (para acesso a metadados, se necessario)

Referencia: docs/ARCHITECTURE_v2.md secao 8.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports: advanced.py — metricas avancadas (C50-C55)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.advanced import (
    CoherenceReport,
    InterfaceReport,
    StabilityReport,
    error_by_anisotropy,
    error_by_resistivity_band,
    interface_metrics,
    physical_coherence_check,
    spatial_error_profile,
    stability_analysis,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: comparison.py — comparacao e ranking de modelos
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.comparison import ComparisonResult, compare_models

# ──────────────────────────────────────────────────────────────────────
# Imports: config_report.py — relatorio pre-treinamento (C42A adaptado)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.config_report import generate_config_report

# ──────────────────────────────────────────────────────────────────────
# Imports: dod.py — Picasso DOD (Depth of Detection) analitico
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.dod import (
    DODResult,
    compute_dod_anisotropy,
    compute_dod_contrast,
    compute_dod_dip,
    compute_dod_frequency,
    compute_dod_map,
    compute_dod_snr,
    compute_dod_standard,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: geosteering_metrics.py — metricas especificas geosteering (C71)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.geosteering_metrics import (
    GeoMetrics,
    compute_geosteering_metrics,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: geosteering_report.py — relatorio Markdown geosteering (C73)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.geosteering_report import generate_geosteering_report

# ──────────────────────────────────────────────────────────────────────
# Imports: manifest.py — manifesto JSON do experimento (C64)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.manifest import (
    create_manifest,
    load_manifest,
    save_manifest,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: metrics.py — metricas de avaliacao pos-treinamento
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.metrics import (
    MetricsReport,
    compute_all_metrics,
    evaluate_predictions,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: predict.py — predicoes com scaling inverso
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.predict import PredictionResult, predict_test

# ──────────────────────────────────────────────────────────────────────
# Imports: realtime_comparison.py — comparacao offline vs realtime (C70)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.realtime_comparison import (
    ModeComparisonResult,
    compare_modes,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: report.py — relatorio Markdown automatizado (C65)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.evaluation.report import generate_report

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
    # --- advanced.py: metricas avancadas (C50-C55) ---
    "InterfaceReport",
    "CoherenceReport",
    "StabilityReport",
    "interface_metrics",
    "error_by_resistivity_band",
    "error_by_anisotropy",
    "spatial_error_profile",
    "physical_coherence_check",
    "stability_analysis",
    # --- predict.py: predicoes com scaling inverso ---
    "PredictionResult",
    "predict_test",
    # --- manifest.py: manifesto JSON do experimento (C64) ---
    "create_manifest",
    "save_manifest",
    "load_manifest",
    # --- report.py: relatorio Markdown automatizado (C65) ---
    "generate_report",
    # --- realtime_comparison.py: comparacao offline vs realtime (C70) ---
    "ModeComparisonResult",
    "compare_modes",
    # --- geosteering_metrics.py: metricas geosteering (C71) ---
    "GeoMetrics",
    "compute_geosteering_metrics",
    # --- geosteering_report.py: relatorio geosteering (C73) ---
    "generate_geosteering_report",
    # --- config_report.py: relatorio pre-treinamento (C42A adaptado) ---
    "generate_config_report",
    # --- dod.py: Picasso DOD (Depth of Detection) analitico ---
    "DODResult",
    "compute_dod_standard",
    "compute_dod_contrast",
    "compute_dod_snr",
    "compute_dod_frequency",
    "compute_dod_anisotropy",
    "compute_dod_dip",
    "compute_dod_map",
]
