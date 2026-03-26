# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.training                                        ║
# ║  Bloco: 6 — Training Loop, Callbacks, Metrics, N-Stage, Optuna            ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║                                                                            ║
# ║  Modulos: loop, callbacks, metrics, nstage, optuna_hpo                    ║
# ║  Exports: ~10 simbolos publicos — ver __all__                             ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 7                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de treinamento — loop, callbacks, metricas, N-Stage, Optuna.

API principal:
    TrainingLoop(config).run(...)     — pipeline completo de treinamento
    build_callbacks(config, ...)       — factory de callbacks Keras
    build_metrics(config)              — factory de metricas Keras
    NStageTrainer(config).run(...)     — treinamento multi-stage
    run_hpo(config, fn)                — otimizacao Optuna (opt-in)

Uso tipico:
    >>> from geosteering_ai.training import TrainingLoop, build_callbacks, build_metrics
    >>> loop = TrainingLoop(config)
    >>> result = loop.run(model, loss_fn, build_metrics(config), train_ds, val_ds, cbs)
"""
from geosteering_ai.training.loop import TrainingLoop, TrainingResult
from geosteering_ai.training.callbacks import (
    UpdateNoiseLevelCallback,
    WeightNormMonitor,
    BestEpochTracker,
    build_callbacks,
)
from geosteering_ai.training.metrics import (
    R2Score,
    PerComponentMetric,
    AnisotropyRatioError,
    build_metrics,
)
from geosteering_ai.training.nstage import NStageTrainer, NStageResult
from geosteering_ai.training.optuna_hpo import run_hpo

__all__ = [
    # ── Loop ──────────────────────────────────────────────────────────────
    "TrainingLoop",
    "TrainingResult",
    # ── N-Stage ───────────────────────────────────────────────────────────
    "NStageTrainer",
    "NStageResult",
    # ── Callbacks ─────────────────────────────────────────────────────────
    "UpdateNoiseLevelCallback",
    "WeightNormMonitor",
    "BestEpochTracker",
    "build_callbacks",
    # ── Metrics ───────────────────────────────────────────────────────────
    "R2Score",
    "PerComponentMetric",
    "AnisotropyRatioError",
    "build_metrics",
    # ── Optuna HPO ────────────────────────────────────────────────────────
    "run_hpo",
]
