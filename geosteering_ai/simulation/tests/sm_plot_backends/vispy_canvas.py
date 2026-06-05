# -*- coding: utf-8 -*-
"""Shim de retrocompat — re-exporta ``gui.plot_backends.vispy_canvas`` (spec 0006).

TRANSITÓRIO: a implementação canônica vive em
:mod:`geosteering_ai.gui.plot_backends.vispy_canvas`. Removido quando o SM migrar (0011).
"""

from __future__ import annotations

from geosteering_ai.gui.plot_backends.vispy_canvas import VispyCanvas

__all__ = ["VispyCanvas"]
