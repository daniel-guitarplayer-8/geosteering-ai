# -*- coding: utf-8 -*-
"""Shim de retrocompat — re-exporta ``gui.plot_backends.base`` (spec 0006).

TRANSITÓRIO: a implementação canônica vive em
:mod:`geosteering_ai.gui.plot_backends.base`. Removido quando o SM migrar (0011).
"""

from __future__ import annotations

from geosteering_ai.gui.plot_backends.base import (
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
