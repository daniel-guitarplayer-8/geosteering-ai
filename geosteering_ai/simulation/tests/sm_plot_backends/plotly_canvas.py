# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_plot_backends/plotly_canvas.py        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Backend Plotly — HTML interativo via QWebEngineView                      ║
# ║  Status      : Produção (v2.6, requer PyQt6-WebEngine)                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  VANTAGENS                                                                ║
# ║    • Hover tooltips ricos via hovertemplate (nativo)                      ║
# ║    • Zoom box / pan profissional                                          ║
# ║    • shared_yaxes em make_subplots replica sharey                         ║
# ║    • Export HTML standalone (compartilhável)                              ║
# ║                                                                           ║
# ║  LIMITAÇÕES                                                               ║
# ║    • Requer PyQt6-WebEngine (~150 MB)                                     ║
# ║    • Latência de setHtml (~50-200ms)                                      ║
# ║    • >50k pontos exige Scattergl                                          ║
# ║    • Sem suporte LaTeX nativo (MathJax via cdn)                           ║
# ║                                                                           ║
# ║  ESTRATÉGIA DE PERFORMANCE                                                ║
# ║    Bufferiza traces no self._fig durante plot_line(); chama setHtml UMA  ║
# ║    vez em draw() — não por linha. Fluxo:                                  ║
# ║      add_subplot_grid → make_subplots cria layout                        ║
# ║      plot_line × N    → fig.add_trace acumula                            ║
# ║      draw()           → fig.to_html + view.setHtml (single render)       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Backend Plotly — HTML interativo embedded em QWebEngineView."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .base import AxisConfig, PlotCanvas, SubplotHandle

__all__ = ["PlotlyCanvas"]


class PlotlyCanvas(PlotCanvas):
    """Canvas Plotly via QWebEngineView — modo análise/sharing.

    Note:
        Requer ``PyQt6-WebEngine`` instalado:
        ``pip install PyQt6-WebEngine``
    """

    def __init__(
        self,
        parent: Any = None,
        figsize: Tuple[float, float] = (14, 9),
        style: Any = None,
    ) -> None:
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Backend Plotly requer PyQt6-WebEngine. "
                "Instale com: pip install PyQt6-WebEngine"
            ) from exc
        from PyQt6.QtWebEngineWidgets import QWebEngineView

        self._view = QWebEngineView(parent)
        self._fig = None
        self._style = style
        self._figsize = figsize
        self._template = (
            "plotly_dark"
            if getattr(style, "theme", "light") == "dark"
            else "plotly_white"
        )
        self._subplot_map: List[List[Tuple[int, int]]] = []
        self._dark = self._template == "plotly_dark"

    def widget(self) -> Any:
        return self._view

    def clear(self) -> None:
        self._fig = None
        self._subplot_map = []
        self._view.setHtml("<html><body style='margin:0;'></body></html>")

    def draw(self) -> None:
        if self._fig is None:
            return
        # Single setHtml — performance crítica
        html = self._fig.to_html(
            include_plotlyjs="cdn",
            config={"displayModeBar": True, "responsive": True},
        )
        self._view.setHtml(html)

    def save(self, path: str, dpi: int = 150) -> None:
        if self._fig is None:
            raise RuntimeError(
                "Nenhum plot para salvar — use add_subplot_grid + plot_line antes."
            )
        from pathlib import Path

        ext = Path(path).suffix.lower()
        if ext in (".html", ".htm"):
            self._fig.write_html(path, include_plotlyjs="cdn")
        elif ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf"):
            try:
                # Requer kaleido instalado
                self._fig.write_image(path, scale=dpi / 100.0)
            except Exception as exc:
                raise RuntimeError(
                    f"Export Plotly para {ext} requer 'kaleido' instalado. "
                    f"Instale com: pip install kaleido. Erro: {exc}"
                ) from exc
        else:
            raise ValueError(f"Extensão não suportada pelo Plotly: {ext}")

    def add_subplot_grid(
        self,
        rows: int,
        cols: int,
        sharey: bool = True,
        width_ratios: Optional[Sequence[float]] = None,
        height_ratios: Optional[Sequence[float]] = None,
    ) -> List[List[SubplotHandle]]:
        from plotly.subplots import make_subplots

        kwargs = {
            "rows": rows,
            "cols": cols,
            "shared_yaxes": sharey,
            "horizontal_spacing": 0.04,
            "vertical_spacing": 0.06,
        }
        if width_ratios is not None:
            total = float(sum(width_ratios))
            kwargs["column_widths"] = [w / total for w in width_ratios]
        if height_ratios is not None:
            total = float(sum(height_ratios))
            kwargs["row_heights"] = [h / total for h in height_ratios]

        self._fig = make_subplots(**kwargs)
        # Aplica template (dark/light) e tamanho
        self._fig.update_layout(
            template=self._template,
            width=int(self._figsize[0] * 80),
            height=int(self._figsize[1] * 80),
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        # Y reverso para depth (camadas crescem para baixo)
        self._fig.update_yaxes(autorange="reversed")
        # Retorna matriz de tuples (row, col) — handles 1-indexed do Plotly
        self._subplot_map = [[(r + 1, c + 1) for c in range(cols)] for r in range(rows)]
        return self._subplot_map

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
        if self._fig is None:
            raise RuntimeError("add_subplot_grid deve ser chamado antes de plot_line")
        import plotly.graph_objs as go

        r, c = ax  # type: ignore[misc]
        dash_map = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
        self._fig.add_trace(
            go.Scatter(
                x=np.asarray(x).tolist(),
                y=np.asarray(y).tolist(),
                mode="lines",
                name=label or "",
                line=dict(
                    color=color,
                    width=linewidth,
                    dash=dash_map.get(linestyle, "solid"),
                ),
                hovertemplate=("z=%{y:.2f} m<br>" "valor=%{x:.4e}" "<extra></extra>"),
            ),
            row=r,
            col=c,
        )

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
        if self._fig is None:
            return
        r, c = ax  # type: ignore[misc]
        self._fig.add_hline(
            y=y,
            line_dash="dash",
            line_color=color,
            line_width=linewidth,
            opacity=alpha,
            row=r,
            col=c,
        )

    def set_axis_config(self, ax: SubplotHandle, cfg: AxisConfig) -> None:
        if self._fig is None:
            return
        r, c = ax  # type: ignore[misc]
        if cfg.title:
            # Plotly não tem title por subplot — usa annotation no topo
            pass
        if cfg.xlabel:
            self._fig.update_xaxes(title_text=cfg.xlabel, row=r, col=c)
        if cfg.ylabel:
            self._fig.update_yaxes(title_text=cfg.ylabel, row=r, col=c)
        if cfg.invert_y:
            self._fig.update_yaxes(autorange="reversed", row=r, col=c)
        if cfg.log_x:
            self._fig.update_xaxes(type="log", row=r, col=c)
        if cfg.log_y:
            self._fig.update_yaxes(type="log", row=r, col=c)
        self._fig.update_xaxes(showgrid=cfg.grid, row=r, col=c)
        self._fig.update_yaxes(showgrid=cfg.grid, row=r, col=c)

    def set_dark_mode(self, dark: bool) -> None:
        self._template = "plotly_dark" if dark else "plotly_white"
        self._dark = dark
        if self._fig is not None:
            self._fig.update_layout(template=self._template)

    def reset_zoom(self) -> None:
        if self._fig is not None:
            self._fig.update_layout(
                xaxis=dict(autorange=True), yaxis=dict(autorange="reversed")
            )
            self.draw()
