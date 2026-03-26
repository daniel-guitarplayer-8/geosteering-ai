# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.inference                                      ║
# ║  Bloco: 7 — Inference                                                    ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Modulos: pipeline.py, realtime.py, export.py                             ║
# ║  Cadeia: raw → FV → GS → scale → model.predict → inverse_scale           ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6                                     ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (3 modulos)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de inferencia — pipeline, realtime, export.

Implementa a cadeia completa de inferencia para o pipeline de inversao
geofisica, desde dados brutos ate predicoes em Ohm.m:

    raw → FV_transform → GS_transform → scale → model.predict → inverse_scale

Modulos:
    pipeline        InferencePipeline: cadeia completa FV+GS+scalers+modelo
    realtime        RealtimeInference: sliding window para geosteering
    export          Exportacao para SavedModel, TFLite, ONNX

Principios:
    - Toda funcao recebe config: PipelineConfig (NUNCA globals)
    - Scalers e transformacoes identicas ao treinamento (P3/P6)
    - InferencePipeline serializavel via joblib + .keras + .yaml
    - RealtimeInference com buffer circular para inferencia causal

Referencia: docs/ARCHITECTURE_v2.md secao 6.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports: pipeline.py — cadeia completa de inferencia
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.inference.pipeline import InferencePipeline

# ──────────────────────────────────────────────────────────────────────
# Imports: realtime.py — sliding window para geosteering
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.inference.realtime import RealtimeInference

# ──────────────────────────────────────────────────────────────────────
# Imports: export.py — exportacao de modelos
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.inference.export import (
    export_saved_model,
    export_tflite,
    export_onnx,
)

# ──────────────────────────────────────────────────────────────────────
# Imports: uncertainty.py — quantificacao de incerteza (MC Dropout, Ensemble)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.inference.uncertainty import (
    UncertaintyResult,
    UncertaintyEstimator,
)

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente por modulo
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- pipeline.py: cadeia completa de inferencia ---
    "InferencePipeline",
    # --- realtime.py: sliding window para geosteering ---
    "RealtimeInference",
    # --- export.py: exportacao de modelos ---
    "export_saved_model",
    "export_tflite",
    "export_onnx",
    # --- uncertainty.py: quantificacao de incerteza ---
    "UncertaintyResult",
    "UncertaintyEstimator",
]
