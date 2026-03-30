# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.models                                          ║
# ║  Bloco: 3 — Arquiteturas de Modelos (44 total)                           ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║                                                                            ║
# ║  Modulos: blocks, cnn, tcn, rnn, hybrid, unet, transformer,              ║
# ║           decomposition, advanced, geosteering, registry                  ║
# ║  Exports: ~16 simbolos publicos principais — ver __all__                  ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de modelos — 44 arquiteturas + ModelRegistry.

API principal:
    build_model(config)           — factory central (ponto de uso)
    ModelRegistry                 — fachada OO
    list_available_models()       — catalogo das 44 arquiteturas
    is_causal_compatible(name)    — verifica modo realtime
    get_model_info(name)          — metadados por arquitetura

Uso tipico:
    >>> from geosteering_ai.models import build_model
    >>> from geosteering_ai.config import PipelineConfig
    >>> config = PipelineConfig.baseline()
    >>> # model = build_model(config)  # requer TF (Colab)

Note:
    Referenciado em:
        - training/loop.py: from geosteering_ai.models import build_model
        - tests/test_models.py: import geosteering_ai.models
    Ref: docs/ARCHITECTURE_v2.md secao 5.
"""

# ── Blocos utilitarios (importavel sem TF — funcoes com lazy TF inside) ───
from geosteering_ai.models.blocks import (
    bottleneck_block_1d,
    conv_next_block,
    dilated_causal_block,
    feedforward_block,
    film_layer,
    normalization_block,
    output_projection,
    residual_block_1d,
    se_block,
    skip_connection_block,
    static_injection_stem,
)

# ── Registry (API principal — imports lazy internamente) ──────────────────
from geosteering_ai.models.registry import (
    ModelRegistry,
    build_model,
    get_model_info,
    is_causal_compatible,
    list_available_models,
)

# ── D8: Exports publicos ─────────────────────────────────────────────────
__all__ = [
    # ── Registry / Factory ────────────────────────────────────────────
    "ModelRegistry",
    "build_model",
    "get_model_info",
    "is_causal_compatible",
    "list_available_models",
    # ── Blocos (utilitarios) ──────────────────────────────────────────
    "residual_block_1d",
    "bottleneck_block_1d",
    "conv_next_block",
    "se_block",
    "dilated_causal_block",
    "output_projection",
    "normalization_block",
    "skip_connection_block",
    "feedforward_block",
    # ── Static Injection (Abordagens B/C — P2/P3) ────────────────────
    "static_injection_stem",
    "film_layer",
]
