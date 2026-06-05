# -*- coding: utf-8 -*-
"""Shim de retrocompat — re-exporta ``gui.persistence.plot_cache`` (spec 0007).

TRANSITÓRIO: a implementação canônica vive em
:mod:`geosteering_ai.gui.persistence.plot_cache` (relocada na spec 0007 via git mv).
Mantido para não quebrar ``from .sm_plot_cache import …`` (monólito) nem
``tests/test_simulation_lru_cache.py``. Removido quando o SM migrar (0011).
"""

from __future__ import annotations

from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache, default_max_bytes

__all__ = ["LRUPlotCache", "default_max_bytes"]
