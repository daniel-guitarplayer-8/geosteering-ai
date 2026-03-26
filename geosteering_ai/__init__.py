# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: geosteering_ai/__init__.py                                       ║
# ║  Bloco: 0 — Raiz do Pacote                                               ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable via pyproject.toml)              ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Subpacotes: config, data, noise, models, losses, training, inference,    ║
# ║              evaluation, visualization, utils                              ║
# ║  Exports: ~30 simbolos publicos (classes + factories + metadados)         ║
# ║  Ref: docs/ARCHITECTURE_v2.md                                             ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (Bloco 1: config + estrutura)║
# ║    v2.0.0 (2026-03) — Re-exports de todos os subpacotes (Blocos 1-9)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Geosteering AI — Pipeline de Inversao Geofisica 1D com Deep Learning.

Reproduz, com fidelidade fisica, a inversao eletromagnetica em tempo real
atraves de arquiteturas de Deep Learning. Suporta componentes EM + geosinais
e/ou Feature Views como features para inversao de resistividade em cenarios
de inferencia offline (acausal) e realtime causal para geosteering em
ambientes ruidosos (on-the-fly).

Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
Autor: Daniel Leal.

Modulos:
    config          PipelineConfig dataclass (ponto unico de verdade)
    data            Loading, splitting, FV, GS, scaling, DataPipeline
    noise           Noise on-the-fly (gaussian, multiplicative, curriculum)
    models          44 arquiteturas (39 standard + 5 geosteering)
    losses          26 funcoes de perda (13 gen + 4 geo + 9 adv)
    training        TrainingLoop, callbacks, N-Stage, metricas
    inference       InferencePipeline, SlidingWindowInference, export
    evaluation      Metricas, comparacao, sensibilidade
    visualization   Holdout plots, training curves, Picasso DOD, EDA
    utils           Logger, timer, validation, formatting, system, io

Example:
    >>> from geosteering_ai import PipelineConfig
    >>> from geosteering_ai import DataPipeline
    >>> from geosteering_ai import build_model, build_loss_fn
    >>>
    >>> config = PipelineConfig.from_yaml("configs/robusto.yaml")
    >>> pipeline = DataPipeline(config)
    >>> data = pipeline.prepare("/path/to/dataset")
    >>> model = build_model(config)
    >>> loss_fn = build_loss_fn(config)
"""

import logging as _logging

_logger = _logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Metadados do pacote
# ──────────────────────────────────────────────────────────────────────
__version__ = "2.0.0"
__author__ = "Daniel Leal"
__framework__ = "TensorFlow 2.x / Keras"

# ──────────────────────────────────────────────────────────────────────
# Bloco 1: config — PipelineConfig (sempre disponivel, sem TF)
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.config import PipelineConfig

# ──────────────────────────────────────────────────────────────────────
# Bloco 2: data — DataPipeline, PreparedData, loading, splitting, etc.
#   Dependencia: numpy, scipy, scikit-learn (sem TF obrigatorio no import)
# ──────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.data import DataPipeline, PreparedData
except ImportError as _e:  # pragma: no cover
    _logger.debug("data subpackage nao disponivel: %s", _e)
    DataPipeline = None  # type: ignore[assignment, misc]
    PreparedData = None  # type: ignore[assignment, misc]

# ──────────────────────────────────────────────────────────────────────
# Bloco 2c: noise — apply_noise_tf, CurriculumSchedule
#   Dependencia: TF para funcoes tf.*, numpy para offline
# ──────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.noise import (
        apply_noise_tf,
        apply_raw_em_noise,
        CurriculumSchedule,
        create_noise_level_var,
    )
except ImportError as _e:  # pragma: no cover
    _logger.debug("noise subpackage nao disponivel: %s", _e)
    apply_noise_tf = None  # type: ignore[assignment]
    apply_raw_em_noise = None  # type: ignore[assignment]
    CurriculumSchedule = None  # type: ignore[assignment, misc]
    create_noise_level_var = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────
# Bloco 3: models — ModelRegistry, build_model, list_available_models
#   Dependencia: TF para construcao efetiva do modelo
# ──────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.models import (
        ModelRegistry,
        build_model,
        get_model_info,
        is_causal_compatible,
        list_available_models,
    )
except ImportError as _e:  # pragma: no cover
    _logger.debug("models subpackage nao disponivel: %s", _e)
    ModelRegistry = None  # type: ignore[assignment, misc]
    build_model = None  # type: ignore[assignment]
    get_model_info = None  # type: ignore[assignment]
    is_causal_compatible = None  # type: ignore[assignment]
    list_available_models = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────
# Bloco 4: losses — LossFactory, build_loss_fn, list_available_losses
#   Dependencia: TF para funcoes de perda
# ──────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.losses import (
        LossFactory,
        build_loss_fn,
        list_available_losses,
        VALID_LOSS_TYPES,
    )
except ImportError as _e:  # pragma: no cover
    _logger.debug("losses subpackage nao disponivel: %s", _e)
    LossFactory = None  # type: ignore[assignment, misc]
    build_loss_fn = None  # type: ignore[assignment]
    list_available_losses = None  # type: ignore[assignment]
    VALID_LOSS_TYPES = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────
# Bloco 6: training — TrainingLoop, NStageTrainer, build_callbacks
#   Dependencia: TF/Keras para callbacks e loop de treinamento
# ──────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.training import (
        TrainingLoop,
        TrainingResult,
        NStageTrainer,
        NStageResult,
        build_callbacks,
        build_metrics,
    )
except ImportError as _e:  # pragma: no cover
    _logger.debug("training subpackage nao disponivel: %s", _e)
    TrainingLoop = None  # type: ignore[assignment, misc]
    TrainingResult = None  # type: ignore[assignment, misc]
    NStageTrainer = None  # type: ignore[assignment, misc]
    NStageResult = None  # type: ignore[assignment, misc]
    build_callbacks = None  # type: ignore[assignment]
    build_metrics = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────
# Bloco 7: inference — InferencePipeline, RealtimeInference, export
#   Dependencia: TF para model.predict e exportacao
# ──────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.inference import (
        InferencePipeline,
        RealtimeInference,
        export_saved_model,
        export_tflite,
        export_onnx,
    )
except ImportError as _e:  # pragma: no cover
    _logger.debug("inference subpackage nao disponivel: %s", _e)
    InferencePipeline = None  # type: ignore[assignment, misc]
    RealtimeInference = None  # type: ignore[assignment, misc]
    export_saved_model = None  # type: ignore[assignment]
    export_tflite = None  # type: ignore[assignment]
    export_onnx = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────
# Bloco 8: evaluation — MetricsReport, compute_all_metrics, compare
#   Dependencia: numpy-only (sem TF), mas pode falhar se numpy ausente
# ──────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.evaluation import (
        MetricsReport,
        compute_all_metrics,
        evaluate_predictions,
        ComparisonResult,
        compare_models,
    )
except ImportError as _e:  # pragma: no cover
    _logger.debug("evaluation subpackage nao disponivel: %s", _e)
    MetricsReport = None  # type: ignore[assignment, misc]
    compute_all_metrics = None  # type: ignore[assignment]
    evaluate_predictions = None  # type: ignore[assignment]
    ComparisonResult = None  # type: ignore[assignment, misc]
    compare_models = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────
# Bloco 9: visualization — holdout, picasso, eda, realtime
#   Dependencia: matplotlib (opcional). Lazy imports internos.
# ──────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.visualization import (
        plot_holdout_samples,
        plot_picasso_dod,
        plot_eda_summary,
        RealtimeMonitor,
    )
except ImportError as _e:  # pragma: no cover
    _logger.debug("visualization subpackage nao disponivel: %s", _e)
    plot_holdout_samples = None  # type: ignore[assignment]
    plot_picasso_dod = None  # type: ignore[assignment]
    plot_eda_summary = None  # type: ignore[assignment]
    RealtimeMonitor = None  # type: ignore[assignment, misc]

# ──────────────────────────────────────────────────────────────────────
# Bloco 5: utils — logger, timer, validation, formatting, system, io
#   Dependencia: stdlib apenas (sem TF)
# ──────────────────────────────────────────────────────────────────────
try:
    from geosteering_ai.utils import get_logger, setup_logger, set_all_seeds
except ImportError as _e:  # pragma: no cover
    _logger.debug("utils subpackage nao disponivel: %s", _e)
    get_logger = None  # type: ignore[assignment]
    setup_logger = None  # type: ignore[assignment]
    set_all_seeds = None  # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente por subpacote
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # ── Metadados do pacote ──────────────────────────────────────────
    "__version__",
    "__author__",
    "__framework__",
    # ── config (Bloco 1) ─────────────────────────────────────────────
    "PipelineConfig",
    # ── data (Bloco 2) ───────────────────────────────────────────────
    "DataPipeline",
    "PreparedData",
    # ── noise (Bloco 2c) ─────────────────────────────────────────────
    "apply_noise_tf",
    "apply_raw_em_noise",
    "CurriculumSchedule",
    "create_noise_level_var",
    # ── models (Bloco 3) ─────────────────────────────────────────────
    "ModelRegistry",
    "build_model",
    "get_model_info",
    "is_causal_compatible",
    "list_available_models",
    # ── losses (Bloco 4) ─────────────────────────────────────────────
    "LossFactory",
    "build_loss_fn",
    "list_available_losses",
    "VALID_LOSS_TYPES",
    # ── training (Bloco 6) ───────────────────────────────────────────
    "TrainingLoop",
    "TrainingResult",
    "NStageTrainer",
    "NStageResult",
    "build_callbacks",
    "build_metrics",
    # ── inference (Bloco 7) ──────────────────────────────────────────
    "InferencePipeline",
    "RealtimeInference",
    "export_saved_model",
    "export_tflite",
    "export_onnx",
    # ── evaluation (Bloco 8) ─────────────────────────────────────────
    "MetricsReport",
    "compute_all_metrics",
    "evaluate_predictions",
    "ComparisonResult",
    "compare_models",
    # ── visualization (Bloco 9) ──────────────────────────────────────
    "plot_holdout_samples",
    "plot_picasso_dod",
    "plot_eda_summary",
    "RealtimeMonitor",
    # ── utils (Bloco 5 — subset principal) ───────────────────────────
    "get_logger",
    "setup_logger",
    "set_all_seeds",
]
