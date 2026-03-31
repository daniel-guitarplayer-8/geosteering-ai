# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.data                                            ║
# ║  Bloco: 2 — Preparacao de Dados                                           ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║                                                                            ║
# ║  Modulos: loading.py, splitting.py, feature_views.py, geosignals.py,     ║
# ║           scaling.py, pipeline.py                                          ║
# ║  Cadeia: raw → parse → decoupling → split → FV → GS → scale → tf.data   ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4                                     ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (6 modulos)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de dados — loading, splitting, FV, GS, scaling, pipeline.

Implementa a cadeia completa de preparacao de dados para o pipeline de
inversao geofisica:

    raw → parse → decoupling → split → FV → GS → scale → tf.data

Modulos:
    loading         Parsing de .out/.dat, decoupling EM, mapeamento 22-col
    splitting       Split por modelo geologico (P1), DataSplits container
    feature_views   6 transformacoes de Feature View (identity, log, razao, etc.)
    geosignals      Geosinais derivados (USD, UHR, UHA, UAD, U3DF)
    scaling         Target scaling (8 metodos), scaler fit/transform
    pipeline        DataPipeline orquestrador + PreparedData container

Principios:
    - Split por modelo geologico, NUNCA por amostra (P1)
    - Scaler fitado em dados LIMPOS (P3)
    - FV/GS computados APOS noise on-the-fly (fidelidade fisica)
    - GS veem ruido como LWD real

Referencia: docs/ARCHITECTURE_v2.md secao 4.3.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports: boundaries.py — DTB (Distance to Boundary) labels (P5)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.boundaries import (
    apply_dtb_scaling,
    build_extended_targets,
    compute_dtb_for_dataset,
    compute_dtb_labels,
    detect_boundaries,
    inverse_dtb_scaling,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: feature_views.py — 6 transformacoes de Feature View
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.feature_views import VALID_VIEWS, apply_feature_view

# ──────────────────────────────────────────────────────────────────────
# Imports: geosignals.py — geosinais derivados (USD, UHR, etc.)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.geosignals import (
    FAMILY_DEPS,
    compute_expanded_features,
    compute_geosignals,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: inspection.py — inspecao de dados pre-treinamento (C26A adaptado)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.inspection import export_inspection_csv, inspect_data_splits

# ──────────────────────────────────────────────────────────────────────
# Imports: loading.py — parsing de arquivos .out/.dat e decoupling EM
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.loading import (
    COL_MAP_22,
    EM_COMPONENTS,
    AngleGroup,
    OutMetadata,
    apply_decoupling,
    load_binary_dat,
    load_dataset,
    parse_out_metadata,
    segregate_by_angle,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: pipeline.py — orquestrador DataPipeline + PreparedData
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.pipeline import DataPipeline, PreparedData

# ──────────────────────────────────────────────────────────────────────
# Imports: sampling.py — oversampling alta rho (Estrategia B)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.sampling import (
    compute_rho_max_per_sequence,
    filter_by_rho_max,
    oversample_high_rho,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: scaling.py — target scaling e scaler fit/transform
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.scaling import (
    apply_target_scaling,
    create_scaler,
    fit_scaler,
    inverse_target_scaling,
    transform_features,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: second_order.py — features de 2o grau (Estrategia C)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.second_order import (
    compute_second_order_features,
    compute_second_order_features_tf,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: splitting.py — split por modelo geologico (P1)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.splitting import DataSplits, apply_split, split_model_ids

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente por modulo
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- pipeline.py: orquestrador principal ---
    "DataPipeline",
    "PreparedData",
    # --- loading.py: parsing e decoupling ---
    "AngleGroup",
    "OutMetadata",
    "load_binary_dat",
    "load_dataset",
    "parse_out_metadata",
    "apply_decoupling",
    "segregate_by_angle",
    "COL_MAP_22",
    "EM_COMPONENTS",
    # --- splitting.py: split por modelo geologico (P1) ---
    "DataSplits",
    "split_model_ids",
    "apply_split",
    # --- feature_views.py: transformacoes FV ---
    "apply_feature_view",
    "VALID_VIEWS",
    # --- geosignals.py: geosinais derivados ---
    "compute_expanded_features",
    "compute_geosignals",
    "FAMILY_DEPS",
    # --- boundaries.py: DTB (Distance to Boundary) labels (P5) ---
    "detect_boundaries",
    "compute_dtb_labels",
    "apply_dtb_scaling",
    "inverse_dtb_scaling",
    "build_extended_targets",
    "compute_dtb_for_dataset",
    # --- inspection.py: inspecao de dados (C26A adaptado) ---
    "inspect_data_splits",
    "export_inspection_csv",
    # --- scaling.py: target scaling e scaler ---
    "apply_target_scaling",
    "inverse_target_scaling",
    "create_scaler",
    "fit_scaler",
    "transform_features",
    # --- sampling.py: oversampling alta rho (Estrategia B) ---
    "compute_rho_max_per_sequence",
    "oversample_high_rho",
    "filter_by_rho_max",
    # --- second_order.py: features de 2o grau (Estrategia C) ---
    "compute_second_order_features",
    "compute_second_order_features_tf",
]
