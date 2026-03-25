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
# Imports: loading.py — parsing de arquivos .out/.dat e decoupling EM
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.loading import (
    AngleGroup,
    OutMetadata,
    load_binary_dat,
    load_dataset,
    parse_out_metadata,
    apply_decoupling,
    COL_MAP_22,
    EM_COMPONENTS,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: splitting.py — split por modelo geologico (P1)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.splitting import DataSplits, split_model_ids, apply_split

# ──────────────────────────────────────────────────────────────────────
# Imports: feature_views.py — 6 transformacoes de Feature View
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.feature_views import apply_feature_view, VALID_VIEWS

# ──────────────────────────────────────────────────────────────────────
# Imports: geosignals.py — geosinais derivados (USD, UHR, etc.)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.geosignals import (
    compute_expanded_features,
    compute_geosignals,
    FAMILY_DEPS,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: scaling.py — target scaling e scaler fit/transform
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.scaling import (
    apply_target_scaling,
    inverse_target_scaling,
    create_scaler,
    fit_scaler,
    transform_features,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: pipeline.py — orquestrador DataPipeline + PreparedData
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.data.pipeline import DataPipeline, PreparedData

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
    # --- scaling.py: target scaling e scaler ---
    "apply_target_scaling",
    "inverse_target_scaling",
    "create_scaler",
    "fit_scaler",
    "transform_features",
]
