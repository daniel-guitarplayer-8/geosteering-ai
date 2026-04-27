# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_plot_backends/pyqtgraph_canvas.py     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Backend PyQtGraph — modo interativo (GPU opcional, 60fps em 1000 curvas) ║
# ║  Status      : Produção (v2.7a)                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  VANTAGENS                                                                ║
# ║    • Qt-native — sem overhead de canvas embedding                         ║
# ║    • InfiniteLine para crosshair sincronizado (trivial)                   ║
# ║    • SignalProxy + ScatterPlotItem para hover tooltips                    ║
# ║    • setYLink replica sharey                                              ║
# ║    • ImageExporter / SVGExporter cobrem PNG/SVG                          ║
# ║                                                                           ║
# ║  LIMITAÇÕES                                                               ║
# ║    • Sem suporte LaTeX (apenas HTML Qt)                                   ║
# ║    • Qualidade de export inferior ao matplotlib                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  CORREÇÃO v2.7a (bug fix)                                                 ║
# ║    Tema dark/light agora é aplicado POR INSTÂNCIA via setBackground() e   ║
# ║    setPen() nos PlotItems — pg.setConfigOption() global foi removido.     ║
# ║    Imports PyQt6 hardcoded substituídos por sm_qt_compat (suporta         ║
# ║    PyQt6 e PySide6 como fallback).                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Backend PyQtGraph — interativo de alta performance."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..sm_qt_compat import QtCore, QtGui
from .base import AxisConfig, PlotCanvas, SubplotHandle

__all__ = ["PyQtGraphCanvas"]


class PyQtGraphCanvas(PlotCanvas):
    """Canvas PyQtGraph baseado em GraphicsLayoutWidget.

    Note:
        Crosshair sincronizado é trivial via ``InfiniteLine`` + ``setYLink``.
        Hover tooltips via ``SignalProxy`` no ``sigMouseMoved``.
        Tema dark/light aplicado por instância (v2.7a) — pg.setConfigOption
        global não é mais usado para evitar efeito colateral em outros
        canvases PyQtGraph abertos simultaneamente.
    """

    def __init__(
        self,
        parent: Any = None,
        figsize: Tuple[float, float] = (14, 9),
        style: Any = None,
    ) -> None:
        import pyqtgraph as pg

        # antialias global é inócuo (afeta qualidade, não cores)
        pg.setConfigOption("antialias", True)

        self._pg = pg
        self._glw = pg.GraphicsLayoutWidget(parent=parent)
        self._style = style
        self._plots: Dict[Tuple[int, int], Any] = {}
        self._proxies: List[Any] = []  # mantém referências vivas (anti-GC)

        theme = getattr(style, "theme", "light")
        self._dark = theme == "dark"
        self._apply_theme_to_instance(self._dark)

    # ── Helpers de tema por instância ─────────────────────────────────────

    def _apply_theme_to_instance(self, dark: bool) -> None:
        """Aplica tema dark/light ao GraphicsLayoutWidget e seus PlotItems.

        Evita ``pg.setConfigOption()`` global — afeta apenas esta instância,
        permitindo que outros canvases PyQtGraph coexistam com temas diferentes.
        """
        bg = "#1e1e1e" if dark else "#ffffff"
        fg = "#d4d4d4" if dark else "#000000"
        grid_alpha = 0.3

        # Fundo do widget
        self._glw.setBackground(bg)

        # Atualizar todos os PlotItems já criados
        for p in self._plots.values():
            self._style_plot_item(p, fg, grid_alpha)

    def _style_plot_item(self, p: Any, fg: str, grid_alpha: float = 0.3) -> None:
        """Aplica cor de eixo/texto a um PlotItem individual."""
        pen = self._pg.mkPen(color=fg, width=1)
        for axis_name in ("left", "bottom", "right", "top"):
            ax = p.getAxis(axis_name)
            if ax is not None:
                ax.setPen(pen)
                ax.setTextPen(pen)
        p.showGrid(x=True, y=True, alpha=grid_alpha)

    # ── Interface PlotCanvas ───────────────────────────────────────────────

    def widget(self) -> Any:
        return self._glw

    def clear(self) -> None:
        self._glw.clear()
        self._plots.clear()
        self._proxies.clear()

    def draw(self) -> None:
        # PyQtGraph renderiza imediatamente (paint event Qt) — no-op
        return None

    def save(self, path: str, dpi: int = 150) -> None:  # noqa: ARG002
        from pathlib import Path

        ext = Path(path).suffix.lower()
        if ext == ".svg":
            from pyqtgraph.exporters import SVGExporter

            exp = SVGExporter(self._glw.scene())
        else:
            from pyqtgraph.exporters import ImageExporter

            exp = ImageExporter(self._glw.scene())
        exp.export(path)

    def add_subplot_grid(
        self,
        rows: int,
        cols: int,
        sharey: bool = True,
        width_ratios: Optional[Sequence[float]] = None,  # noqa: ARG002
        height_ratios: Optional[Sequence[float]] = None,  # noqa: ARG002
    ) -> List[List[SubplotHandle]]:
        fg = "#d4d4d4" if self._dark else "#000000"
        plots: List[List[Any]] = []
        first: Optional[Any] = None
        for r in range(rows):
            row_plots: List[Any] = []
            for c in range(cols):
                p = self._glw.addPlot(row=r, col=c)
                p.invertY(True)  # profundidade cresce para baixo
                self._style_plot_item(p, fg)
                if sharey and first is not None:
                    p.setYLink(first)
                if first is None and sharey:
                    first = p
                self._plots[(r, c)] = p
                row_plots.append(p)
            plots.append(row_plots)
        return plots

    def plot_line(
        self,
        ax: SubplotHandle,
        x: np.ndarray,
        y: np.ndarray,
        *,
        label: str = "",
        color: Optional[str] = None,
        linewidth: float = 1.5,
        linestyle: str = "-",
    ) -> None:
        pen_color = color if color else ("#d4d4d4" if self._dark else "#1f4ea8")
        # Mapear linestyle matplotlib → Qt.PenStyle
        style_map = {
            "-": QtCore.Qt.PenStyle.SolidLine,
            "--": QtCore.Qt.PenStyle.DashLine,
            ":": QtCore.Qt.PenStyle.DotLine,
            "-.": QtCore.Qt.PenStyle.DashDotLine,
        }
        pen = self._pg.mkPen(
            pen_color,
            width=linewidth,
            style=style_map.get(linestyle, QtCore.Qt.PenStyle.SolidLine),
        )
        ax.plot(x, y, pen=pen, name=label or None)

    def add_hline(
        self,
        ax: SubplotHandle,
        y: float,
        *,
        color: str = "#7f7f7f",
        linestyle: str = "--",  # noqa: ARG002  # PyQtGraph usa InfiniteLine (sempre DashLine)
        alpha: float = 0.4,
        linewidth: float = 0.8,
    ) -> None:
        c = QtGui.QColor(color)
        c.setAlphaF(alpha)
        line = self._pg.InfiniteLine(
            pos=y,
            angle=0,
            pen=self._pg.mkPen(c, width=linewidth, style=2),  # 2 = DashLine
        )
        ax.addItem(line)

    def set_axis_config(self, ax: SubplotHandle, cfg: AxisConfig) -> None:
        if cfg.title:
            ax.setTitle(cfg.title)
        if cfg.xlabel:
            ax.setLabel("bottom", cfg.xlabel)
        if cfg.ylabel:
            ax.setLabel("left", cfg.ylabel)
        ax.invertY(cfg.invert_y)
        ax.showGrid(x=cfg.grid, y=cfg.grid, alpha=0.3)
        if cfg.log_x:
            ax.setLogMode(x=True, y=False)
        if cfg.log_y:
            ax.setLogMode(x=False, y=True)

    def set_dark_mode(self, dark: bool) -> None:
        """Aplica tema dark/light por instância (sem afetar outros canvases)."""
        self._dark = dark
        self._apply_theme_to_instance(dark)

    def reset_zoom(self) -> None:
        for p in self._plots.values():
            p.autoRange()
