# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_plot_backends/pyqtgraph_canvas.py     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Backend PyQtGraph — modo interativo (GPU opcional, 60fps em 1000 curvas) ║
# ║  Status      : Produção (v2.6)                                            ║
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
# ║    • Tema dark via setConfigOption (não por instância)                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Backend PyQtGraph — interativo de alta performance."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import AxisConfig, PlotCanvas, SubplotHandle

__all__ = ["PyQtGraphCanvas"]


class PyQtGraphCanvas(PlotCanvas):
    """Canvas PyQtGraph baseado em GraphicsLayoutWidget.

    Note:
        Crosshair sincronizado é trivial via ``InfiniteLine`` + ``setYLink``.
        Hover tooltips via ``SignalProxy`` no ``sigMouseMoved``.
    """

    def __init__(
        self,
        parent: Any = None,
        figsize: Tuple[float, float] = (14, 9),
        style: Any = None,
    ) -> None:
        import pyqtgraph as pg

        # Tema dark/light global (PyQtGraph é global, não por instância)
        theme = getattr(style, "theme", "light")
        if theme == "dark":
            pg.setConfigOption("background", "#1e1e1e")
            pg.setConfigOption("foreground", "#d4d4d4")
        else:
            pg.setConfigOption("background", "w")
            pg.setConfigOption("foreground", "k")
        pg.setConfigOption("antialias", True)

        self._pg = pg
        self._glw = pg.GraphicsLayoutWidget(parent=parent)
        self._style = style
        self._plots: Dict[Tuple[int, int], Any] = {}
        self._proxies: List[Any] = []  # mantém referências vivas (anti-GC)
        self._dark = theme == "dark"

    def widget(self) -> Any:
        return self._glw

    def clear(self) -> None:
        self._glw.clear()
        self._plots.clear()
        self._proxies.clear()

    def draw(self) -> None:
        # PyQtGraph renderiza imediatamente (paint event Qt) — no-op
        return None

    def save(self, path: str, dpi: int = 150) -> None:
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
        width_ratios: Optional[Sequence[float]] = None,
        height_ratios: Optional[Sequence[float]] = None,
    ) -> List[List[SubplotHandle]]:
        plots: List[List[Any]] = []
        first: Optional[Any] = None
        for r in range(rows):
            row_plots: List[Any] = []
            for c in range(cols):
                p = self._glw.addPlot(row=r, col=c)
                p.invertY(True)  # depth cresce para baixo
                p.showGrid(x=True, y=True, alpha=0.3)
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
        # PyQtGraph linestyle: '-' solid, '--' dash, ':' dot
        from PyQt6.QtCore import Qt

        style_map = {
            "-": Qt.PenStyle.SolidLine,
            "--": Qt.PenStyle.DashLine,
            ":": Qt.PenStyle.DotLine,
            "-.": Qt.PenStyle.DashDotLine,
        }
        pen = self._pg.mkPen(
            pen_color,
            width=linewidth,
            style=style_map.get(linestyle, Qt.PenStyle.SolidLine),
        )
        ax.plot(x, y, pen=pen, name=label or None)

    def add_hline(
        self,
        ax: SubplotHandle,
        y: float,
        *,
        color: str = "#7f7f7f",
        linestyle: str = "--",
        alpha: float = 0.4,
        linewidth: float = 0.8,
    ) -> None:
        from PyQt6.QtGui import QColor

        c = QColor(color)
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
        # PyQtGraph é global — recriar canvas seria necessário para troca real
        self._dark = dark
        if dark:
            self._pg.setConfigOption("background", "#1e1e1e")
            self._pg.setConfigOption("foreground", "#d4d4d4")
        else:
            self._pg.setConfigOption("background", "w")
            self._pg.setConfigOption("foreground", "k")

    def reset_zoom(self) -> None:
        for p in self._plots.values():
            p.autoRange()
