# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.losses                                         ║
# ║  Bloco: 4 — 26 Loss Functions + LossFactory                              ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║                                                                            ║
# ║  Modulos: catalog (26 funcoes), factory (LossFactory + build_loss_fn),  ║
# ║           pinns (3 cenarios PINN + TIV + lambda schedule)               ║
# ║  Exports: ~12 simbolos publicos — ver __all__                            ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de loss functions — 26 losses + LossFactory.

API principal:
    build_loss_fn(config)         — factory central (ponto de uso)
    LossFactory                   — fachada OO
    list_available_losses()       — catalogo das 26 losses
    VALID_LOSS_TYPES              — lista de nomes validos

Uso tipico:
    >>> from geosteering_ai.losses import build_loss_fn
    >>> from geosteering_ai.config import PipelineConfig
    >>> config = PipelineConfig(loss_type="log_scale_aware")
    >>> loss_fn = build_loss_fn(config)
    >>> model.compile(loss=loss_fn, optimizer="adam")
"""
from geosteering_ai.losses.factory import (
    VALID_LOSS_TYPES,
    LossFactory,
    build_loss_fn,
    list_available_losses,
)
from geosteering_ai.losses.pinns import (
    VALID_LAMBDA_SCHEDULES,
    VALID_PINNS_SCENARIOS,
    build_pinns_loss,
    compute_lambda_schedule,
    make_maxwell_physics_loss,
    make_oracle_physics_loss,
    make_surrogate_physics_loss,
    make_tiv_constraint_loss,
)

__all__ = [
    # ── Factory ───────────────────────────────────────────────────────
    "LossFactory",
    "VALID_LOSS_TYPES",
    "build_loss_fn",
    "list_available_losses",
    # ── PINNs ─────────────────────────────────────────────────────────
    "VALID_PINNS_SCENARIOS",
    "VALID_LAMBDA_SCHEDULES",
    "build_pinns_loss",
    "compute_lambda_schedule",
    "make_oracle_physics_loss",
    "make_surrogate_physics_loss",
    "make_maxwell_physics_loss",
    "make_tiv_constraint_loss",
]
