# -*- coding: utf-8 -*-
"""Shim de retrocompat — re-exporta ``gui.plot_backends.mpl_canvas`` (spec 0006).

TRANSITÓRIO: a implementação canônica vive em
:mod:`geosteering_ai.gui.plot_backends.mpl_canvas`. Removido quando o SM migrar (0011).
"""

from __future__ import annotations

from geosteering_ai.gui.plot_backends.mpl_canvas import MatplotlibCanvas

__all__ = ["MatplotlibCanvas"]
