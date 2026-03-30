# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.visualization                                  ║
# ║  Bloco: 9 — Visualization                                                ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Modulos: holdout.py, picasso.py, eda.py, realtime.py, training.py,     ║
# ║           error_maps.py, export.py, optuna_viz.py, geosteering.py     ║
# ║  Proposito: Visualizacoes para inversao geofisica — holdout, DOD, EDA,   ║
# ║             monitoramento realtime, curvas de treinamento, mapas de erro, ║
# ║             exportacao batch                                              ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (4 modulos)                  ║
# ║    v2.0.0 (2026-03) — Adicionados training.py, error_maps.py, export.py║
# ║    v2.0.0 (2026-03) — Adicionado optuna_viz.py (C62 Optuna viz)      ║
# ║    v2.0.0 (2026-03) — Adicionado geosteering.py (C72 geo viz)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de visualizacao — holdout, Picasso DOD, EDA, realtime, training, error maps, export, Optuna.

Modulos:
    holdout     Comparacao true vs predicted em amostras holdout
    picasso     Mapas Picasso DOD (Depth of Detection) por contraste
    eda         Analise exploratoria: distribuicoes, correlacoes, boxplots
    realtime    Monitoramento ao vivo para inferencia realtime
    training    Curvas de treinamento: loss, R², LR schedule, noise level
    error_maps  Heatmap 2D de erro, barras por banda, perfil espacial
    export      Exportacao batch de figuras em multiplos formatos
    optuna_viz  Visualizacoes Optuna: historico, importancia, contorno, paralelas
    geosteering Curtain plot 2D, DTB profile, geosteering dashboard 4-quadrantes

Todas as funcoes recebem ``config: PipelineConfig`` como parametro opcional.
Matplotlib e importado de forma lazy (dentro de cada funcao) para ambientes
sem display grafico.

Referencia: docs/ARCHITECTURE_v2.md secao 9.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports: eda.py — analise exploratoria de dados (sumario + 5 avancados)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.eda import (
    plot_correlation_heatmap,
    plot_eda_summary,
    plot_feature_distributions,
    plot_sample_profiles,
    plot_sensitivity_heatmap,
    plot_train_val_test_comparison,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: error_maps.py — heatmap, barras por banda, perfil espacial
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.error_maps import (
    plot_error_by_band,
    plot_error_heatmap,
    plot_spatial_error,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: export.py — batch export de figuras
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.export import export_all_figures, save_figure

# ──────────────────────────────────────────────────────────────────────
# Imports: geosteering.py — curtain, DTB, dashboard (C72)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.geosteering import (
    plot_curtain,
    plot_dtb_profile,
    plot_geosteering_dashboard,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: holdout.py — comparacao true vs predicted
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.holdout import (
    plot_holdout_clean_noisy,
    plot_holdout_samples,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: optuna_viz.py — visualizacoes Optuna (C62)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.optuna_viz import (
    plot_contour,
    plot_optimization_history,
    plot_optuna_results,
    plot_parallel_coordinate,
    plot_param_importances,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: picasso.py — mapas Picasso DOD
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.picasso import plot_picasso_dod

# ──────────────────────────────────────────────────────────────────────
# Imports: realtime.py — monitoramento ao vivo
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.realtime import RealtimeMonitor

# ──────────────────────────────────────────────────────────────────────
# Imports: training.py — curvas de treinamento (C59)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.training import plot_lr_schedule, plot_training_history

# ──────────────────────────────────────────────────────────────────────
# Imports: uncertainty.py — visualizacao de incerteza (histogramas, CI, calibracao)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.uncertainty import (
    plot_calibration_curve,
    plot_confidence_bands,
    plot_uncertainty_histograms,
)

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente por modulo
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- holdout.py: comparacao true vs predicted + clean vs noisy ---
    "plot_holdout_samples",
    "plot_holdout_clean_noisy",
    # --- picasso.py: mapas Picasso DOD ---
    "plot_picasso_dod",
    # --- eda.py: analise exploratoria (sumario + 5 avancados) ---
    "plot_eda_summary",
    "plot_feature_distributions",
    "plot_correlation_heatmap",
    "plot_sample_profiles",
    "plot_train_val_test_comparison",
    "plot_sensitivity_heatmap",
    # --- realtime.py: monitoramento ao vivo ---
    "RealtimeMonitor",
    # --- training.py: curvas de treinamento (C59) ---
    "plot_training_history",
    "plot_lr_schedule",
    # --- error_maps.py: heatmap, barras, perfil espacial ---
    "plot_error_heatmap",
    "plot_error_by_band",
    "plot_spatial_error",
    # --- export.py: batch export de figuras ---
    "export_all_figures",
    "save_figure",
    # --- uncertainty.py: histogramas, bandas CI, calibracao ---
    "plot_uncertainty_histograms",
    "plot_confidence_bands",
    "plot_calibration_curve",
    # --- optuna_viz.py: visualizacoes Optuna (C62) ---
    "plot_optuna_results",
    "plot_optimization_history",
    "plot_param_importances",
    "plot_contour",
    "plot_parallel_coordinate",
    # --- geosteering.py: curtain, DTB, dashboard (C72) ---
    "plot_curtain",
    "plot_dtb_profile",
    "plot_geosteering_dashboard",
]
