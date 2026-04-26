# -*- coding: utf-8 -*-
"""Sub-pacote de backends de plotagem para o Simulation Manager v2.6.

Disponibiliza 4 backends pluggable via `make_canvas(backend, ...)`:

- ``PlotBackend.MATPLOTLIB`` — default, sempre disponível
- ``PlotBackend.PYQTGRAPH`` — interativo, GPU opcional
- ``PlotBackend.PLOTLY`` — HTML hover rich (requer PyQt6-WebEngine)
- ``PlotBackend.VISPY`` — GL puro experimental

Imports lazy: cada backend só carrega suas deps quando solicitado.
"""
from __future__ import annotations

from .base import (
    AxisConfig,
    PlotBackend,
    PlotCanvas,
    SubplotHandle,
    available_backends,
    make_canvas,
)

__all__ = [
    "AxisConfig",
    "PlotBackend",
    "PlotCanvas",
    "SubplotHandle",
    "available_backends",
    "make_canvas",
]
