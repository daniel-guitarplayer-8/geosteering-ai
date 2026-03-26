# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.losses                                         ║
# ║  Bloco: 4 — 26 Loss Functions + LossFactory                              ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║                                                                            ║
# ║  Modulos: catalog (26 funcoes), factory (LossFactory + build_loss_fn)    ║
# ║  Exports: ~6 simbolos publicos — ver __all__                              ║
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
    LossFactory,
    VALID_LOSS_TYPES,
    build_loss_fn,
    list_available_losses,
)

__all__ = [
    "LossFactory",
    "VALID_LOSS_TYPES",
    "build_loss_fn",
    "list_available_losses",
]
