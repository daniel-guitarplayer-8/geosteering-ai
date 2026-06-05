# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_plot_backends/__init__.py             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — Backends de plotagem (shim retrocompat)             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-26 · 2026-06-05 (relocado p/ gui/, spec 0006)       ║
# ║  Status      : Produção (shim de retrocompatibilidade)                    ║
# ║  Dependências: geosteering_ai.gui.plot_backends                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Shim de retrocompatibilidade. A abstração multi-backend de plotagem    ║
# ║    (PlotCanvas ABC + factory + 4 backends) foi RELOCADA na spec 0006 para  ║
# ║    o pacote de PRODUÇÃO ``geosteering_ai/gui/plot_backends/`` (para que SM ║
# ║    e Studio a compartilhem, e a GUI deixe de viver em ``tests/``). Este    ║
# ║    pacote re-exporta dali — os importadores existentes do monólito         ║
# ║    (``from .sm_plot_backends import …`` e o submódulo direto               ║
# ║    ``…sm_plot_backends.pyqtgraph_canvas``) continuam SEM alteração.        ║
# ║                                                                           ║
# ║  PADRÃO                                                                   ║
# ║    Strangler Fig (ADR-S01) — idêntico ao ``sm_qt_compat`` (spec 0004).     ║
# ║                                                                           ║
# ║  EXPORTS (re-export de gui.plot_backends — objetos IDÊNTICOS)             ║
# ║    PlotCanvas, PlotBackend, AxisConfig, SubplotHandle,                    ║
# ║    make_canvas, available_backends                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Shim de retrocompat — re-exporta os backends de plotagem de ``gui.plot_backends``.

A implementação canônica vive em :mod:`geosteering_ai.gui.plot_backends` (relocada
na spec 0006). Mantido para não quebrar o monólito do Simulation Manager, que faz
``from .sm_plot_backends import make_canvas`` e também acessa o submódulo
``…sm_plot_backends.pyqtgraph_canvas`` diretamente. Por isso o shim é um PACOTE
(com submódulos), não um módulo plano.

TODO (Fase 1 — spec 0011): TRANSITÓRIO. Quando o Simulation Manager migrar para
``geosteering_ai/gui/`` e seus imports apontarem para ``geosteering_ai.gui.plot_backends``,
este shim (e a inversão de camada "código em ``tests/`` importando ``gui/``") deve
ser REMOVIDO.
"""

from __future__ import annotations

from geosteering_ai.gui.plot_backends import (
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
