# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_plot_backends/vispy_canvas.py         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Backend Vispy — GL puro (experimental)                                   ║
# ║  Status      : Stub limitado (v2.6)                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  VANTAGENS                                                                ║
# ║    • GPU OpenGL — 1M+ pontos a 60fps                                      ║
# ║    • Adequado para >100k pontos (caso de benchmark 30k modelos)           ║
# ║                                                                           ║
# ║  LIMITAÇÕES (recomendação do agente de pesquisa: ROI baixo)               ║
# ║    • Sem axes nativos (~400 LOC para reproduzir labels matplotlib)        ║
# ║    • API de baixo nível (Visual, Transform manuais)                       ║
# ║    • set_axis_config: NotImplementedError                                 ║
# ║    • Para uso típico (10k pontos), PyQtGraph oferece melhor balanço      ║
# ║                                                                           ║
# ║  USO PRETENDIDO                                                           ║
# ║    Exclusivamente para benchmark de visualização real-time com 30k+      ║
# ║    modelos — quando matplotlib e PyQtGraph não conseguem manter 30fps.   ║
# ║    Marcado como "(experimental)" no combo de Preferências.                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Backend Vispy — stub experimental para casos extremos (>100k pontos)."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .base import AxisConfig, PlotCanvas, SubplotHandle

__all__ = ["VispyCanvas"]


class VispyCanvas(PlotCanvas):
    """Canvas Vispy — implementação stub para casos extremos.

    Note:
        Esta é uma implementação **experimental** com features mínimas.
        Para uso normal, prefira Matplotlib (default) ou PyQtGraph.

        ``set_axis_config`` lança ``NotImplementedError`` — Vispy não
        tem axes nativos como matplotlib. Reproduzir labels exigiria
        ~400 LOC de Visual customizados.
    """

    def __init__(
        self,
        parent: Any = None,
        figsize: Tuple[float, float] = (14, 9),
        style: Any = None,
    ) -> None:
        try:
            from vispy import scene  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Backend Vispy requer pacote 'vispy'. " "Instale com: pip install vispy"
            ) from exc
        from vispy import scene
        from vispy.scene import SceneCanvas

        bgcolor = "#1e1e1e" if getattr(style, "theme", "light") == "dark" else "white"
        self._scene = scene
        self._canvas = SceneCanvas(
            keys="interactive",
            bgcolor=bgcolor,
            parent=parent,
            size=(int(figsize[0] * 80), int(figsize[1] * 80)),
        )
        self._grid = None
        self._views: List[List[Any]] = []
        self._style = style
        self._dark = bgcolor == "#1e1e1e"

    def widget(self) -> Any:
        return self._canvas.native

    def clear(self) -> None:
        if self._grid is not None:
            for view_row in self._views:
                for vb in view_row:
                    if vb is not None and vb.parent is self._grid:
                        vb.parent = None
            self._views = []
        self._grid = None

    def draw(self) -> None:
        self._canvas.update()

    def save(self, path: str, dpi: int = 150) -> None:
        from pathlib import Path

        ext = Path(path).suffix.lower()
        if ext != ".png":
            raise NotImplementedError(
                f"Vispy stub só exporta PNG (recebido: {ext}). "
                "Use Matplotlib para SVG/PDF."
            )
        img = self._canvas.render()
        try:
            from imageio import imwrite

            imwrite(path, img)
        except ImportError:
            # Fallback usando vispy interno (write_png)
            try:
                from vispy.io import write_png

                write_png(path, img)
            except Exception as exc:
                raise RuntimeError(
                    f"Falha ao salvar PNG via Vispy (instale imageio): {exc}"
                ) from exc

    def add_subplot_grid(
        self,
        rows: int,
        cols: int,
        sharey: bool = True,
        width_ratios: Optional[Sequence[float]] = None,
        height_ratios: Optional[Sequence[float]] = None,
    ) -> List[List[SubplotHandle]]:
        self._grid = self._canvas.central_widget.add_grid()
        self._views = []
        for r in range(rows):
            row_views: List[Any] = []
            for c in range(cols):
                vb = self._grid.add_view(
                    row=r,
                    col=c,
                    bgcolor="#252526" if self._dark else "white",
                    border_color="#3c3c3c" if self._dark else "#cccccc",
                )
                vb.camera = self._scene.PanZoomCamera(aspect=None)
                vb.camera.flip = (False, True, False)  # depth invertido (Y down)
                row_views.append(vb)
            self._views.append(row_views)
        return self._views

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
        from vispy.scene import visuals

        col = color if color else ("#74b9ff" if self._dark else "#1f4ea8")
        # Normaliza color hex → rgba
        from matplotlib.colors import to_rgba

        rgba = to_rgba(col)
        pos = np.column_stack([np.asarray(x), np.asarray(y)]).astype(np.float32)
        line = visuals.Line(
            pos=pos,
            color=rgba,
            width=linewidth,
            method="gl",
            antialias=True,
        )
        ax.add(line)

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
        from matplotlib.colors import to_rgba
        from vispy.scene import visuals

        rgba = to_rgba(color, alpha=alpha)
        # Vispy não tem range automático antes de plot — usa intervalo amplo
        x_range = np.array([-1e9, 1e9], dtype=np.float32)
        pos = np.column_stack([x_range, np.array([y, y], dtype=np.float32)])
        line = visuals.Line(pos=pos, color=rgba, width=linewidth, method="gl")
        ax.add(line)

    def set_axis_config(self, ax: SubplotHandle, cfg: AxisConfig) -> None:
        # Vispy stub: features avançadas de eixo não implementadas.
        # Para títulos/labels reais, use Matplotlib ou PyQtGraph.
        if cfg.title or cfg.xlabel or cfg.ylabel:
            # No-op silencioso — features ausentes neste stub
            pass
        if cfg.invert_y:
            # Já configurado em add_subplot_grid via flip=(False, True, False)
            pass

    def set_dark_mode(self, dark: bool) -> None:
        self._dark = dark
        bgcolor = "#1e1e1e" if dark else "white"
        try:
            self._canvas.bgcolor = bgcolor
        except Exception:
            pass

    def reset_zoom(self) -> None:
        for row_views in self._views:
            for vb in row_views:
                if vb is not None and hasattr(vb, "camera"):
                    try:
                        vb.camera.set_range()
                    except Exception:
                        pass
