# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/results_view.py                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ResultsView — galeria do ensemble (View Qt)                ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0011d / 0017)  ║
# ║  Versão      : v0.2                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — galeria do ensemble (Fatia 4 + 6d)              ║
# ║  Framework   : Qt6 via gui.qt_compat + gui.plot_backends                   ║
# ║  Padrão      : View (MVVM) — binding ao ResultsViewModel; sem lógica       ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Barra de seletores (canal Hxx..Hzz/geosinais, plot-kind Re/Im/|H|/      ║
# ║    fase, modo curva/perfil ρ/anisotropia λ/heatmap, índices TR/dip/freq)   ║
# ║    + paginação (◀ ▶) + animation bar + GRADE (add_subplot_grid) com as     ║
# ║    curvas/perfis dos modelos da página, OU um heatmap do ensemble inteiro. ║
# ║    Liga-se a ``changed``/``results_changed`` do ResultsViewModel; re-plota. ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ResultsView                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ResultsView`` — galeria Qt do ensemble (grade/heatmap + seletores) (0011d/0017)."""

from __future__ import annotations

import math
from typing import Any

from apps.sim_manager.perspectives.simulation.results_viewmodel import (
    CHANNELS,
    PLOT_KINDS,
    PLOT_MODES,
    ResultsViewModel,
)
from geosteering_ai.gui.plot_backends import (
    AxisConfig,
    PlotBackend,
    available_backends,
    make_canvas,
)
from geosteering_ai.gui.qt_compat import QtWidgets
from geosteering_ai.gui.shell.widgets.animation_bar import AnimationBar

__all__ = ["ResultsView"]

# Rótulos amigáveis dos plot-kinds (mesma ordem de PLOT_KINDS).
_KIND_LABELS = {"re": "Re", "im": "Im", "mag": "|H|", "phase": "Fase (°)"}
# Rótulos amigáveis dos modos da galeria (mesma ordem de PLOT_MODES).
_MODE_LABELS = {
    "curve": "Curvas",
    "rho": "Perfil ρ",
    "lambda": "Anisotropia λ",
    "heatmap": "Heatmap ensemble",
}
_MAX_COLS = 4  # colunas máximas da grade da galeria


class ResultsView(QtWidgets.QWidget):  # type: ignore[misc] # QtWidgets é Any → mypy
    """Galeria do ensemble: seletores + paginação + grade/heatmap, ligada ao VM.

    Args:
        vm: o :class:`ResultsViewModel` (a View se liga a ``changed``/``results_changed``).
        parent: widget pai opcional.

    Note:
        A View não computa curvas/perfis — só lê ``vm.curve_for``/``vm.rho_curves_for``/
        ``vm.ensemble_image`` e plota. Os seletores empurram para o VM (que clampa);
        o VM emite → a View re-monta.
    """

    def __init__(self, vm: ResultsViewModel, parent: Any = None) -> None:
        super().__init__(parent)
        self._vm = vm

        # ── Seletores ────────────────────────────────────────────────────────
        # Backend de plot (matplotlib ↔ pyqtgraph ↔ …): só os disponíveis no env.
        self._backend = QtWidgets.QComboBox()
        self._backend.addItems([b.value for b in available_backends()])
        self._backend.setCurrentText(vm.plot_backend.value)
        self._backend.setToolTip(
            "Motor de plotagem: PyQtGraph (interativo, default) ou matplotlib (publicação)."
        )
        # Modo da galeria: curvas / perfil ρ / anisotropia λ / heatmap do ensemble.
        self._mode = QtWidgets.QComboBox()
        self._mode.addItems([_MODE_LABELS[m] for m in PLOT_MODES])
        self._mode.setToolTip("O que plotar: curvas do canal, perfis ρ/λ ou heatmap.")
        # Canal: 9 componentes (Hxx..Hzz) + 5 geosinais (USD..U3DF).
        self._cmp = QtWidgets.QComboBox()
        self._cmp.addItems(list(CHANNELS))
        self._cmp.setToolTip("Canal: componente do tensor H (Hxx..Hzz) ou geosinal.")
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

        # ── Animation bar (varre o ensemble — single-model playback) ──────────
        self._anim = AnimationBar()
        self._anim.frame_changed.connect(self._on_focus_changed)

        # ── Canvas (galeria via grade de subplots) ───────────────────────────
        # Cria o canvas com FALLBACK p/ matplotlib se o backend pedido não tiver
        # deps (ex.: .session com "plotly" sem WebEngine) — reconcilia o VM p/ o
        # backend efetivo (combo segue via _sync_controls no _render inicial).
        self._canvas, self._active_backend = self._build_canvas(vm.plot_backend)
        if vm.plot_backend != self._active_backend:
            vm.plot_backend = self._active_backend  # binding ainda não conectado

        # ── Layout ────────────────────────────────────────────────────────────
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Plot:"))
        bar.addWidget(self._backend)
        bar.addWidget(QtWidgets.QLabel("Modo:"))
        bar.addWidget(self._mode)
        bar.addWidget(QtWidgets.QLabel("Canal:"))
        bar.addWidget(self._cmp)
        bar.addWidget(self._kind)
        bar.addWidget(self._tr)
        bar.addWidget(self._dip)
        bar.addWidget(self._freq)
        bar.addStretch(1)
        bar.addWidget(self._prev)
        bar.addWidget(self._page_lbl)
        bar.addWidget(self._next)
        self._root = QtWidgets.QVBoxLayout(self)
        self._root.addLayout(bar)
        self._root.addWidget(self._canvas.widget(), stretch=1)
        self._root.addWidget(self._anim)

        # ── Binding ──────────────────────────────────────────────────────────
        self._backend.currentTextChanged.connect(self._on_backend_changed)
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
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
    def _on_backend_changed(self, text: str) -> None:
        try:
            self._vm.plot_backend = PlotBackend(text)
        except ValueError:
            pass  # texto inválido (não deveria ocorrer — combo só lista válidos)

    def _on_mode_changed(self, idx: int) -> None:
        self._vm.plot_mode = PLOT_MODES[idx] if 0 <= idx < len(PLOT_MODES) else "curve"

    def _on_cmp_changed(self, idx: int) -> None:
        self._vm.channel_index = idx

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

    def _on_focus_changed(self, frame: int) -> None:
        """Animation bar avançou → foca o modelo e leva a página até ele (playback)."""
        self._vm.focus_model = frame
        # A página segue o modelo em foco → o playback varre TODO o ensemble e o
        # subplot focado é destacado (▶) nos modos curva/perfil.
        self._vm.page = frame // self._vm.page_size

    # ── Render ───────────────────────────────────────────────────────────────
    def _sync_controls(self) -> None:
        """Reflete o estado do VM nos widgets (com sinais bloqueados — sem loop)."""
        mode = self._vm.plot_mode
        is_profile = mode in ("rho", "lambda")
        is_heatmap = mode == "heatmap"
        dims = self._vm.dims
        # config spinners: range [0, dim-1]; habilitados quando dim>1 E não-perfil
        # (perfis ρ/λ vêm da geologia — independem de TR/dip/freq).
        n_tr, n_ang, n_f = (dims[1], dims[2], dims[4]) if dims else (1, 1, 1)
        for spin, hi, cur in (
            (self._tr, n_tr, self._vm.tr_index),
            (self._dip, n_ang, self._vm.dip_index),
            (self._freq, n_f, self._vm.freq_index),
        ):
            spin.blockSignals(True)
            spin.setRange(0, max(0, hi - 1))
            spin.setValue(cur)
            spin.setEnabled(hi > 1 and not is_profile)
            spin.blockSignals(False)
        # canal + kind: relevantes p/ curvas e heatmap; irrelevantes p/ perfis ρ/λ.
        self._cmp.blockSignals(True)
        self._cmp.setCurrentIndex(self._vm.channel_index)
        self._cmp.setEnabled(not is_profile)
        self._cmp.blockSignals(False)
        self._kind.blockSignals(True)
        self._kind.setCurrentIndex(PLOT_KINDS.index(self._vm.plot_kind))
        self._kind.setEnabled(not is_profile)
        self._kind.blockSignals(False)
        self._mode.blockSignals(True)
        self._mode.setCurrentIndex(PLOT_MODES.index(mode))
        self._mode.blockSignals(False)
        self._backend.blockSignals(True)
        self._backend.setCurrentText(self._vm.plot_backend.value)
        self._backend.blockSignals(False)
        # paginação: por-modelo (curvas/perfis); o heatmap mostra TODOS de uma vez.
        n_pages = self._vm.n_pages
        self._page_lbl.setText(f"{self._vm.page + 1}/{n_pages}" if n_pages else "—")
        self._prev.setEnabled(self._vm.page > 0 and not is_heatmap)
        self._next.setEnabled(self._vm.page < n_pages - 1 and not is_heatmap)
        # animation bar: varre os modelos (1 por frame); só faz sentido com ≥2 modelos
        # e fora do heatmap (que já mostra TODOS os modelos de uma vez).
        self._anim.set_frame_count(self._vm.n_models)
        self._anim.set_frame(self._vm.focus_model)
        self._anim.setEnabled(self._vm.n_models > 1 and not is_heatmap)

    def _build_canvas(self, backend: PlotBackend) -> tuple[Any, PlotBackend]:
        """Cria um canvas do ``backend``, com FALLBACK p/ matplotlib se faltar dep.

        ``make_canvas`` levanta ``ImportError`` p/ backend sem deps (ex.: "plotly"
        sem WebEngine — alcançável via ``.session``). Aqui caímos p/ MATPLOTLIB
        (sempre disponível) em vez de quebrar a galeria. Já aplica ``set_dark_mode``.

        Args:
            backend: backend desejado.

        Returns:
            ``(canvas, backend_efetivo)`` — ``backend_efetivo`` = MATPLOTLIB no fallback.
        """
        try:
            canvas = make_canvas(backend, parent=self)
            effective = backend
        except ImportError:
            effective = PlotBackend.MATPLOTLIB
            canvas = make_canvas(effective, parent=self)
        canvas.set_dark_mode(True)
        return canvas, effective

    def _rebuild_canvas(self) -> None:
        """Recria o canvas quando o backend muda (substitui no layout, sem leak).

        Usa :meth:`_build_canvas` (fallback p/ matplotlib se as deps faltarem) e
        RECONCILIA o VM com o backend efetivo — assim ``_active_backend`` sempre
        avança (sem re-tentar o backend quebrado a cada ``_render``).
        """
        old = self._canvas
        self._canvas, self._active_backend = self._build_canvas(self._vm.plot_backend)
        self._root.replaceWidget(old.widget(), self._canvas.widget())
        old.widget().setParent(None)
        old.widget().deleteLater()
        # Fallback ocorreu → reconcilia o VM (combo/estado convergem; sem recursão:
        # plot_backend passa a == _active_backend, então o próximo _render não rebuilda).
        if self._vm.plot_backend != self._active_backend:
            self._vm.plot_backend = self._active_backend

    def _render(self, *_: Any) -> None:
        """Re-monta a galeria conforme o modo (curva/perfil/heatmap) + sincroniza."""
        self._sync_controls()
        # Backend trocado → recria o canvas ANTES de plotar.
        if self._vm.plot_backend != self._active_backend:
            self._rebuild_canvas()
        self._canvas.clear()
        if not self._vm.has_result or self._vm.depth is None:
            self._canvas.draw()  # galeria vazia (sem resultado)
            return
        mode = self._vm.plot_mode
        if mode == "heatmap":
            self._render_heatmap()
        elif mode in ("rho", "lambda"):
            self._render_profiles(mode)
        else:
            self._render_curves()
        self._canvas.draw()

    def _grid_for_page(self) -> tuple[list, list]:
        """Cria a grade rows×cols p/ os modelos da página atual; retorna (grid, models)."""
        models = self._vm.page_models()
        cols = min(_MAX_COLS, max(1, len(models)))
        rows = math.ceil(len(models) / cols) if models else 1
        grid = self._canvas.add_subplot_grid(rows, cols)
        return grid, models

    def _render_curves(self) -> None:
        """Modo 'curve': uma curva (componente OU geosinal) por modelo da página."""
        grid, models = self._grid_for_page()
        cols = len(grid[0]) if grid else 1
        depth = self._vm.depth
        kind_lbl = _KIND_LABELS[self._vm.plot_kind]
        ch = self._vm.channel_name
        for pos, model_idx in enumerate(models):
            ax = grid[pos // cols][pos % cols]
            curve = self._vm.curve_for(model_idx)
            self._canvas.plot_line(ax, curve, depth, label=f"#{model_idx}")
            self._canvas.set_axis_config(
                ax,
                AxisConfig(
                    title=f"{self._focus_mark(model_idx)}#{model_idx} — {kind_lbl} {ch}",
                    xlabel=f"{kind_lbl} {ch}",
                    ylabel="z (m)",
                    invert_y=True,
                ),
            )

    def _render_profiles(self, mode: str) -> None:
        """Modo 'rho'/'lambda': perfil step ρ(z)/λ(z) por modelo (usa a geologia)."""
        grid, models = self._grid_for_page()
        cols = len(grid[0]) if grid else 1
        for pos, model_idx in enumerate(models):
            ax = grid[pos // cols][pos % cols]
            if mode == "rho":
                data = self._vm.rho_curves_for(model_idx)
                if data is None:
                    self._empty_subplot(ax, f"#{model_idx} — sem geologia")
                    continue
                z, rho_h, rho_v = data
                self._canvas.plot_step(ax, rho_h, z, label="ρh", color="#6366f1")
                self._canvas.plot_step(
                    ax, rho_v, z, label="ρv", color="#e0833b", linestyle="--"
                )
                cfg = AxisConfig(
                    title=f"{self._focus_mark(model_idx)}#{model_idx} — ρ(z)",
                    xlabel="ρ (Ω·m)",
                    ylabel="z (m)",
                    invert_y=True,
                    log_x=True,
                )
            else:  # lambda
                data = self._vm.lambda_curve_for(model_idx)
                if data is None:
                    self._empty_subplot(ax, f"#{model_idx} — sem geologia")
                    continue
                z, lam = data
                self._canvas.plot_step(ax, lam, z, label="λ", color="#22c55e")
                cfg = AxisConfig(
                    title=f"{self._focus_mark(model_idx)}#{model_idx} — λ(z)",
                    xlabel="λ = √(ρv/ρh)",
                    ylabel="z (m)",
                    invert_y=True,
                )
            self._canvas.set_axis_config(ax, cfg)

    def _render_heatmap(self) -> None:
        """Modo 'heatmap': imagem (n_models × n_pos) do canal/transform do ensemble."""
        grid = self._canvas.add_subplot_grid(1, 1, sharey=False)
        ax = grid[0][0]
        res = self._vm.ensemble_image()
        if res is None:
            self._empty_subplot(ax, "sem ensemble")
            return
        img, extent, label = res
        try:
            handle = self._canvas.plot_image(ax, img, extent=extent, cmap="viridis")
            self._canvas.set_colorbar(ax, handle, label=label)
        except NotImplementedError:
            # plotly/vispy não suportam imagem → aviso claro (sem crash).
            self._empty_subplot(
                ax, f"heatmap indisponível ({self._active_backend.value})"
            )
            return
        self._canvas.set_axis_config(
            ax,
            AxisConfig(
                title=f"Ensemble — {label}",
                xlabel="z (m)",
                ylabel="modelo #",
                invert_y=False,  # a imagem já posiciona os modelos via extent/origin
                grid=False,
            ),
        )

    def _focus_mark(self, model_idx: int) -> str:
        """``"▶ "`` se ``model_idx`` é o modelo em foco da animação (≥2 modelos), senão ``""``."""
        if self._vm.n_models > 1 and model_idx == self._vm.focus_model:
            return "▶ "
        return ""

    def _empty_subplot(self, ax: Any, title: str) -> None:
        """Configura um subplot vazio com um título (estado 'sem dados')."""
        self._canvas.set_axis_config(ax, AxisConfig(title=title, invert_y=False))
