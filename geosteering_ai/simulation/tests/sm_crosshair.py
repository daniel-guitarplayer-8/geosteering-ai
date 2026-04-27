# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_crosshair.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Crosshair sincronizado (blitting)     ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Status      : Produção (v2.6b)                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    CrosshairManager mantém uma linha vertical (e opcionalmente            ║
# ║    horizontal) sincronizada entre N axes do mesmo Figure matplotlib       ║
# ║    via blitting (draw_artist + blit) para manter >30 fps em 18 subplots.  ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    cm = CrosshairManager(figcanvas, axes_list)                            ║
# ║    cm.enable()        # ativa hover crosshair                             ║
# ║    cm.disable()       # desativa e limpa cache                            ║
# ║    cm.toggle()        # alterna estado                                    ║
# ║                                                                           ║
# ║  PERFORMANCE                                                              ║
# ║    Blitting: redraw apenas das linhas (não da figura inteira). Cache      ║
# ║    `_bg` é reconstruído em resize/draw_event (hooks abaixo).              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""CrosshairManager — crosshair sincronizado entre N axes via blitting (v2.6b)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class CrosshairManager:
    """Crosshair sincronizado entre múltiplos axes matplotlib via blitting.

    Attributes:
        canvas: ``FigureCanvasQTAgg`` do matplotlib.
        axes: lista de ``Axes`` que compartilharão o crosshair.
        show_horizontal: se ``True``, desenha também linha horizontal.

    Note:
        Blitting funciona melhor com ``Axes`` que tenham extensões similares.
        Para axes com sharey, o crosshair vertical sincroniza naturalmente.
    """

    def __init__(
        self,
        canvas: Any,
        axes: List[Any],
        *,
        show_horizontal: bool = False,
        color: str = "#dcdcaa",
        linewidth: float = 0.8,
    ) -> None:
        self.canvas = canvas
        self.axes = list(axes)
        self.show_horizontal = bool(show_horizontal)
        self._color = color
        self._linewidth = linewidth
        self._enabled = False
        self._bg: Dict[Any, Any] = {}
        self._lines_v: Dict[Any, Any] = {}
        self._lines_h: Dict[Any, Any] = {}
        # connection ids
        self._cid_motion: Optional[int] = None
        self._cid_resize: Optional[int] = None
        self._cid_draw: Optional[int] = None
        self._cid_leave: Optional[int] = None

    # ── API pública ───────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Ativa o crosshair (cria linhas invisíveis + hooks de eventos)."""
        if self._enabled or not self.axes:
            return
        self._create_lines()
        self._cid_motion = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_resize = self.canvas.mpl_connect(
            "resize_event", self._on_resize_or_draw
        )
        self._cid_draw = self.canvas.mpl_connect("draw_event", self._on_resize_or_draw)
        self._cid_leave = self.canvas.mpl_connect("axes_leave_event", self._on_leave)
        self._enabled = True
        # Captura background na próxima draw
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def disable(self) -> None:
        """Desativa o crosshair e remove linhas + hooks."""
        if not self._enabled:
            return
        for cid in (
            self._cid_motion,
            self._cid_resize,
            self._cid_draw,
            self._cid_leave,
        ):
            if cid is not None:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._cid_motion = self._cid_resize = None
        self._cid_draw = self._cid_leave = None
        self._remove_lines()
        self._bg.clear()
        self._enabled = False
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def toggle(self) -> None:
        """Alterna o estado entre ativo/inativo."""
        if self._enabled:
            self.disable()
        else:
            self.enable()

    def update_axes(self, axes: List[Any]) -> None:
        """Substitui a lista de axes (após replot que recriou subplots).

        Mantém o estado enabled/disabled — se ativo, recria linhas no
        novo conjunto de axes.
        """
        was_enabled = self._enabled
        if was_enabled:
            self.disable()
        self.axes = list(axes)
        self._bg.clear()
        if was_enabled:
            self.enable()

    # ── Internos ──────────────────────────────────────────────────────────

    def _create_lines(self) -> None:
        for ax in self.axes:
            try:
                ln_v = ax.axvline(
                    x=0,
                    color=self._color,
                    linewidth=self._linewidth,
                    linestyle="--",
                    alpha=0.0,
                    animated=True,
                )
                self._lines_v[ax] = ln_v
                if self.show_horizontal:
                    ln_h = ax.axhline(
                        y=0,
                        color=self._color,
                        linewidth=self._linewidth,
                        linestyle="--",
                        alpha=0.0,
                        animated=True,
                    )
                    self._lines_h[ax] = ln_h
            except Exception:
                continue

    def _remove_lines(self) -> None:
        for ln in list(self._lines_v.values()) + list(self._lines_h.values()):
            try:
                ln.remove()
            except Exception:
                pass
        self._lines_v.clear()
        self._lines_h.clear()

    def _on_resize_or_draw(self, _event: Any) -> None:
        # Recaptura background de cada axes — invalidate cache
        self._bg.clear()
        for ax in self.axes:
            try:
                self._bg[ax] = self.canvas.copy_from_bbox(ax.bbox)
            except Exception:
                pass

    def _on_motion(self, event: Any) -> None:
        if event is None or event.inaxes is None:
            return
        if event.xdata is None and event.ydata is None:
            return
        x = event.xdata
        y = event.ydata
        # Garante backgrounds capturados
        if not self._bg:
            self._on_resize_or_draw(None)
        # Para cada axes, restaura BG e desenha line vertical em x
        for ax in self.axes:
            ln_v = self._lines_v.get(ax)
            if ln_v is None:
                continue
            try:
                bg = self._bg.get(ax)
                if bg is None:
                    continue
                self.canvas.restore_region(bg)
                if x is not None:
                    ln_v.set_xdata([x, x])
                    ln_v.set_alpha(0.85)
                    ax.draw_artist(ln_v)
                if self.show_horizontal:
                    ln_h = self._lines_h.get(ax)
                    if ln_h is not None and y is not None and ax is event.inaxes:
                        ln_h.set_ydata([y, y])
                        ln_h.set_alpha(0.85)
                        ax.draw_artist(ln_h)
                self.canvas.blit(ax.bbox)
            except Exception:
                continue

    def _on_leave(self, _event: Any) -> None:
        # Apaga crosshair ao sair da área de plot
        for ax in self.axes:
            try:
                bg = self._bg.get(ax)
                if bg is None:
                    continue
                self.canvas.restore_region(bg)
                self.canvas.blit(ax.bbox)
            except Exception:
                continue


__all__ = ["CrosshairManager"]
