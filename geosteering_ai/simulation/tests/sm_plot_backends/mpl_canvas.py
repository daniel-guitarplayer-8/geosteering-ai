# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_plot_backends/mpl_canvas.py           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Backend Matplotlib                    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Status      : Produção (v2.6)                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementação concreta de PlotCanvas usando Matplotlib +               ║
# ║    FigureCanvasQTAgg. É o backend default — sempre disponível, qualidade  ║
# ║    de publicação SVG/PDF, suporte LaTeX. Wraps a infraestrutura legada    ║
# ║    do `EMCanvas` (sm_plots.py:255-321) preservando comportamento bit-exato║
# ║    para os 1464+ testes existentes.                                       ║
# ║                                                                           ║
# ║  ESTRATÉGIA                                                               ║
# ║    • Composição: contém um `FigureCanvasQTAgg` interno + `Figure`         ║
# ║    • `widget()` retorna o FigureCanvas (QWidget) para inserção em layout  ║
# ║    • `add_subplot_grid` usa `GridSpec` com `sharey=axes[0,0]` opcional    ║
# ║    • `set_dark_mode` ajusta facecolor, tick colors, spine colors          ║
# ║                                                                           ║
# ║  COMPATIBILIDADE COM EMCanvas LEGADO                                      ║
# ║    Esta classe não substitui EMCanvas — ela co-existe. A função           ║
# ║    `plot_tensor_full` em sm_plots.py mantém branch para os dois tipos:    ║
# ║    isinstance(canvas, PlotCanvas) usa API nova; EMCanvas usa caminho     ║
# ║    legado direto via canvas.figure.add_subplot.                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Backend Matplotlib (default) para o Simulation Manager v2.6."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .base import AxisConfig, PlotCanvas, SubplotHandle

__all__ = ["MatplotlibCanvas"]


class MatplotlibCanvas(PlotCanvas):
    """Implementação Matplotlib do PlotCanvas — backend default.

    Wraps `FigureCanvasQTAgg` + `Figure` em um QWidget compatível com Qt.
    Reaproveita 100% da lógica do `EMCanvas` legado para máxima preservação.

    Attributes:
        figure: matplotlib.figure.Figure
        canvas: FigureCanvasQTAgg (também é QWidget)
        style: PlotStyle opcional aplicado em construção

    Example:
        >>> from sm_plot_backends import make_canvas, PlotBackend
        >>> canvas = make_canvas(PlotBackend.MATPLOTLIB, parent=self)
        >>> axes = canvas.add_subplot_grid(3, 6, sharey=True)
        >>> canvas.plot_line(axes[0][0], x, y, label="Re(Hxx)")
        >>> canvas.draw()
    """

    def __init__(
        self,
        parent: Any = None,
        figsize: Tuple[float, float] = (14, 9),
        style: Any = None,
    ) -> None:
        """Constrói o canvas matplotlib com Figure interna.

        Args:
            parent: Widget Qt pai (passado ao FigureCanvasQTAgg).
            figsize: Tamanho da figura em polegadas (largura, altura).
            style: Instância opcional de PlotStyle de sm_plots.py.
        """
        # Imports lazy — evita custo de importar matplotlib se backend não usado
        import matplotlib

        matplotlib.use("QtAgg", force=False)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._figure = Figure(figsize=figsize, dpi=getattr(style, "dpi", 100))
        self._canvas = FigureCanvasQTAgg(self._figure)
        if parent is not None:
            self._canvas.setParent(parent)
        self._style = style
        self._gridspec = None  # GridSpec atual (se houver)
        # Aplica tema inicial conforme PlotStyle
        theme = getattr(style, "theme", "light")
        if theme == "dark":
            self.set_dark_mode(True)

    # ── Propriedades de compatibilidade com EMCanvas legado ───────────────

    @property
    def figure(self):
        """Acesso direto à matplotlib.Figure (compat com código legado)."""
        return self._figure

    @property
    def canvas(self):
        """Acesso direto ao FigureCanvasQTAgg (compat com código legado)."""
        return self._canvas

    @property
    def style(self):
        """PlotStyle aplicado neste canvas (pode ser None)."""
        return self._style

    # ── PlotCanvas API ────────────────────────────────────────────────────

    def widget(self) -> Any:
        """Retorna FigureCanvasQTAgg (QWidget) para layout Qt."""
        return self._canvas

    def clear(self) -> None:
        """Limpa todos os axes e elementos da figura."""
        self._figure.clear()
        self._gridspec = None

    def draw(self) -> None:
        """Renderiza canvas (idempotente — usa draw_idle para não bloquear)."""
        self._canvas.draw_idle()

    def save(self, path: str, dpi: int = 150) -> None:
        """Salva como PNG/PDF/SVG conforme extensão do path.

        Args:
            path: Caminho do arquivo de saída.
            dpi: Resolução em DPI (default 150). Para publicação use 300.
        """
        self._figure.savefig(path, dpi=dpi, bbox_inches="tight")

    def add_subplot_grid(
        self,
        rows: int,
        cols: int,
        sharey: bool = True,
        width_ratios: Optional[Sequence[float]] = None,
        height_ratios: Optional[Sequence[float]] = None,
    ) -> List[List[SubplotHandle]]:
        """Cria grid de subplots via GridSpec; retorna matriz de Axes.

        v2.6 P1: ``sharey=True`` faz todos os subplots compartilharem eixo Y
        (depth comum) — zoom em qualquer subplot propaga para todos.
        """
        from matplotlib.gridspec import GridSpec

        gs_kwargs = {"figure": self._figure}
        if width_ratios is not None:
            gs_kwargs["width_ratios"] = list(width_ratios)
        if height_ratios is not None:
            gs_kwargs["height_ratios"] = list(height_ratios)
        gs_kwargs["wspace"] = 0.32
        gs_kwargs["hspace"] = 0.38

        self._gridspec = GridSpec(rows, cols, **gs_kwargs)
        axes_grid: List[List[Any]] = []
        first_ax: Optional[Any] = None
        for r in range(rows):
            row_axes: List[Any] = []
            for c in range(cols):
                kwargs = {}
                if sharey and first_ax is not None:
                    kwargs["sharey"] = first_ax
                ax = self._figure.add_subplot(self._gridspec[r, c], **kwargs)
                if first_ax is None and sharey:
                    first_ax = ax
                row_axes.append(ax)
            axes_grid.append(row_axes)
        return axes_grid

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
        """Adiciona curva via ``ax.plot()``."""
        ax.plot(x, y, label=label, color=color, linewidth=linewidth, linestyle=linestyle)

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
        """Adiciona linha horizontal via ``ax.axhline()``."""
        ax.axhline(y, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)

    def set_axis_config(self, ax: SubplotHandle, cfg: AxisConfig) -> None:
        """Aplica AxisConfig (title, labels, invert_y, grid, log scales)."""
        if cfg.title:
            ax.set_title(cfg.title)
        if cfg.xlabel:
            ax.set_xlabel(cfg.xlabel)
        if cfg.ylabel:
            ax.set_ylabel(cfg.ylabel)
        if cfg.invert_y and not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.grid(cfg.grid, alpha=0.35)
        if cfg.log_x:
            ax.set_xscale("log")
        if cfg.log_y:
            ax.set_yscale("log")

    def set_dark_mode(self, dark: bool) -> None:
        """Ajusta facecolor, tick colors, spine colors para tema dark/light.

        v2.6 U1+P1: integração visual com tema dark da UI VSCode-style.
        """
        if dark:
            bg = "#1e1e1e"
            fg = "#d4d4d4"
            grid_color = "#3c3c3c"
        else:
            bg = "#ffffff"
            fg = "#000000"
            grid_color = "#cccccc"
        self._figure.set_facecolor(bg)
        for ax in self._figure.axes:
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, which="both")
            for spine in ax.spines.values():
                spine.set_edgecolor(fg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            if ax.get_title():
                ax.title.set_color(fg)
            ax.grid(color=grid_color, alpha=0.5)

    def set_legend(self, ax: SubplotHandle, visible: bool = True) -> None:
        """Mostra/oculta legenda no subplot."""
        if visible:
            ax.legend(loc="best", fontsize=8)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    def reset_zoom(self) -> None:
        """Reseta zoom de todos os axes ao range automático."""
        for ax in self._figure.axes:
            ax.relim()
            ax.autoscale_view()
        self.draw()
