# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/results_view.py                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ResultsView — galeria do ensemble (View Qt)                ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0011d)         ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — galeria do ensemble (Fatia 4)                    ║
# ║  Framework   : Qt6 via gui.qt_compat + gui.plot_backends                   ║
# ║  Padrão      : View (MVVM) — binding ao ResultsViewModel; sem lógica       ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Barra de seletores (componente Hxx..Hzz, plot-kind Re/Im/|H|/fase,      ║
# ║    índices TR/dip/freq) + paginação (◀ ▶) + uma GRADE (add_subplot_grid)   ║
# ║    com as curvas dos modelos da página. Liga-se a ``changed`` (seletor) e  ║
# ║    ``results_changed`` (novo resultado) do ResultsViewModel e re-plota.    ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ResultsView                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ResultsView`` — galeria Qt do ensemble (grade + seletores + paginação) (0011d)."""

from __future__ import annotations

import math
from typing import Any

from apps.sim_manager.perspectives.simulation.results_viewmodel import (
    COMPONENT_NAMES,
    PLOT_KINDS,
    ResultsViewModel,
)
from geosteering_ai.gui.plot_backends import AxisConfig, PlotBackend, make_canvas
from geosteering_ai.gui.qt_compat import QtWidgets

__all__ = ["ResultsView"]

# Rótulos amigáveis dos plot-kinds (mesma ordem de PLOT_KINDS).
_KIND_LABELS = {"re": "Re", "im": "Im", "mag": "|H|", "phase": "Fase (°)"}
_MAX_COLS = 4  # colunas máximas da grade da galeria


class ResultsView(QtWidgets.QWidget):  # type: ignore[misc] # QtWidgets é Any → mypy
    """Galeria do ensemble: seletores + paginação + grade de curvas, ligada ao VM.

    Args:
        vm: o :class:`ResultsViewModel` (a View se liga a ``changed``/``results_changed``).
        parent: widget pai opcional.

    Note:
        A View não computa curvas — só lê ``vm.curve_for(m)``/``vm.depth`` e plota.
        Os seletores empurram para o VM (que clampa); o VM emite → a View re-monta.
    """

    def __init__(self, vm: ResultsViewModel, parent: Any = None) -> None:
        super().__init__(parent)
        self._vm = vm

        # ── Seletores ────────────────────────────────────────────────────────
        self._cmp = QtWidgets.QComboBox()
        self._cmp.addItems(list(COMPONENT_NAMES))
        self._cmp.setToolTip("Componente do tensor H (Hxx..Hzz).")
        self._kind = QtWidgets.QComboBox()
        self._kind.addItems([_KIND_LABELS[k] for k in PLOT_KINDS])
        self._kind.setToolTip("Modo de plot: Re / Im / magnitude / fase.")
        self._tr = QtWidgets.QSpinBox()
        self._tr.setPrefix("TR ")
        self._dip = QtWidgets.QSpinBox()
        self._dip.setPrefix("dip ")
        self._freq = QtWidgets.QSpinBox()
        self._freq.setPrefix("f ")

        # ── Paginação ────────────────────────────────────────────────────────
        self._prev = QtWidgets.QPushButton("◀")
        self._next = QtWidgets.QPushButton("▶")
        self._page_lbl = QtWidgets.QLabel("—")

        # ── Canvas (galeria via grade de subplots) ───────────────────────────
        self._canvas = make_canvas(PlotBackend.MATPLOTLIB, parent=self)
        # Fundo escuro p/ casar com o tema (o canvas não é alcançável pelo QSS).
        self._canvas.set_dark_mode(True)

        # ── Layout ────────────────────────────────────────────────────────────
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Comp.:"))
        bar.addWidget(self._cmp)
        bar.addWidget(QtWidgets.QLabel("Modo:"))
        bar.addWidget(self._kind)
        bar.addWidget(self._tr)
        bar.addWidget(self._dip)
        bar.addWidget(self._freq)
        bar.addStretch(1)
        bar.addWidget(self._prev)
        bar.addWidget(self._page_lbl)
        bar.addWidget(self._next)
        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(bar)
        root.addWidget(self._canvas.widget(), stretch=1)

        # ── Binding ──────────────────────────────────────────────────────────
        self._cmp.currentIndexChanged.connect(self._on_cmp_changed)
        self._kind.currentIndexChanged.connect(self._on_kind_changed)
        self._tr.valueChanged.connect(self._on_tr_changed)
        self._dip.valueChanged.connect(self._on_dip_changed)
        self._freq.valueChanged.connect(self._on_freq_changed)
        self._prev.clicked.connect(self._on_prev)
        self._next.clicked.connect(self._on_next)
        self._vm.changed.connect(self._render)
        self._vm.results_changed.connect(self._render)
        self._render()

    # ── Slots de seletor (empurram p/ o VM; o VM clampa e re-emite) ──────────
    def _on_cmp_changed(self, idx: int) -> None:
        self._vm.component_index = idx

    def _on_kind_changed(self, idx: int) -> None:
        self._vm.plot_kind = PLOT_KINDS[idx] if 0 <= idx < len(PLOT_KINDS) else "re"

    def _on_tr_changed(self, value: int) -> None:
        self._vm.tr_index = value

    def _on_dip_changed(self, value: int) -> None:
        self._vm.dip_index = value

    def _on_freq_changed(self, value: int) -> None:
        self._vm.freq_index = value

    def _on_prev(self) -> None:
        self._vm.page = self._vm.page - 1

    def _on_next(self) -> None:
        self._vm.page = self._vm.page + 1

    # ── Render ───────────────────────────────────────────────────────────────
    def _sync_controls(self) -> None:
        """Reflete o estado do VM nos widgets (com sinais bloqueados — sem loop)."""
        dims = self._vm.dims
        # config spinners: range [0, dim-1]; só habilitados quando dim>1.
        n_tr, n_ang, n_f = (dims[1], dims[2], dims[4]) if dims else (1, 1, 1)
        for spin, hi, cur in (
            (self._tr, n_tr, self._vm.tr_index),
            (self._dip, n_ang, self._vm.dip_index),
            (self._freq, n_f, self._vm.freq_index),
        ):
            spin.blockSignals(True)
            spin.setRange(0, max(0, hi - 1))
            spin.setValue(cur)
            spin.setEnabled(hi > 1)
            spin.blockSignals(False)
        for combo, idx in ((self._cmp, self._vm.component_index),):
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._kind.blockSignals(True)
        self._kind.setCurrentIndex(PLOT_KINDS.index(self._vm.plot_kind))
        self._kind.blockSignals(False)
        # paginação
        n_pages = self._vm.n_pages
        self._page_lbl.setText(f"{self._vm.page + 1}/{n_pages}" if n_pages else "—")
        self._prev.setEnabled(self._vm.page > 0)
        self._next.setEnabled(self._vm.page < n_pages - 1)

    def _render(self, *_: Any) -> None:
        """Re-monta a grade da galeria (modelos da página) + sincroniza seletores."""
        self._sync_controls()
        self._canvas.clear()
        models = self._vm.page_models()
        depth = self._vm.depth
        if not models or depth is None:
            self._canvas.draw()  # galeria vazia (sem resultado)
            return
        # grade rows×cols p/ os modelos da página.
        cols = min(_MAX_COLS, len(models))
        rows = math.ceil(len(models) / cols)
        grid = self._canvas.add_subplot_grid(rows, cols)
        kind_lbl = _KIND_LABELS[self._vm.plot_kind]
        comp = self._vm.component_name
        for pos, model_idx in enumerate(models):
            ax = grid[pos // cols][pos % cols]
            curve = self._vm.curve_for(model_idx)
            self._canvas.plot_line(ax, curve, depth, label=f"#{model_idx}")
            self._canvas.set_axis_config(
                ax,
                AxisConfig(
                    title=f"#{model_idx} — {kind_lbl} {comp}",
                    xlabel=f"{kind_lbl} {comp}",
                    ylabel="z (m)",
                    invert_y=True,
                ),
            )
        self._canvas.draw()
