# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/antigravity_window.py                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : AntigravityMainWindow — shell estilo Google Antigravity    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell MVVM (spec 0013)                               ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de shell (reusável SM + Studio)         ║
# ║  Framework   : Qt6 via gui.qt_compat (QSplitter + QStackedWidget)          ║
# ║  Dependências: gui.shell.main_window_base, gui.shell.widgets               ║
# ║  Padrão      : View (MVVM) — host de perspectivas (activity rail + stack)  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Casca ``QMainWindow`` com layout inspirado no Google Antigravity IDE:    ║
# ║    activity rail vertical (nav) │ perspectivas empilhadas (``QStackedWidget``)║
# ║    │ secondary sidebar (Histórico/Log/Artifacts) + status bar com accent.   ║
# ║    Reusa a lógica de registro/lazy-build de ``MainWindowBase`` via os        ║
# ║    accessors ``_host_*`` (substitui o ``QTabWidget`` por rail+stack). NÃO    ║
# ║    toca ViewModels (Princípio X). Reusável pelo SM e pelo futuro Studio.     ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    AntigravityMainWindow                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``AntigravityMainWindow`` — shell rail + stacked + sidebar (estilo Antigravity, 0013)."""

from __future__ import annotations

from typing import Any

from geosteering_ai.gui.qt_compat import Qt, QtWidgets
from geosteering_ai.gui.shell.main_window_base import MainWindowBase
from geosteering_ai.gui.shell.perspective import Perspective
from geosteering_ai.gui.shell.widgets.activity_bar import ActivityBar
from geosteering_ai.gui.shell.widgets.secondary_sidebar import SecondarySidebar

__all__ = ["AntigravityMainWindow"]


class AntigravityMainWindow(MainWindowBase):  # type: ignore[misc] # QtWidgets é Any
    """Shell estilo Antigravity: activity rail + perspectivas empilhadas + sidebar.

    Substitui o host de abas de :class:`MainWindowBase` por um ``QStackedWidget``
    navegado por uma :class:`ActivityBar` vertical, com uma
    :class:`SecondarySidebar` (Histórico/Log/Artifacts) à direita. A lógica de
    registro/ordenação/lazy-build é herdada da base (via os accessors ``_host_*``).

    Attributes:
        sidebar: a :class:`SecondarySidebar` compartilhada (a perspectiva ativa
            envia log/histórico aqui — usado pela Fatia 6a).

    Note:
        Perspectivas com ``enabled = False`` (scaffolds "em breve") aparecem na
        rail como itens cinza não-clicáveis — tornam a ORGANIZAÇÃO de todos os
        recursos visível antes de cada um ser implementado (Fatias 6c-6i).
    """

    _RAIL_WIDTH = 52

    def _init_host(self) -> None:
        """Host Antigravity: [activity rail] [ stack | secondary sidebar ]."""
        self._rail = ActivityBar()
        self._rail.setFixedWidth(self._RAIL_WIDTH)
        self._stack = QtWidgets.QStackedWidget()
        self.sidebar = SecondarySidebar()
        # Expõe a sidebar no contexto (extras) → a perspectiva ativa liga seu
        # log/histórico aqui (Fatia 6a) sem acoplar o ViewModel ao shell.
        self.ctx.extras["secondary_sidebar"] = self.sidebar

        inner = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        inner.addWidget(self._stack)
        inner.addWidget(self.sidebar)
        inner.setStretchFactor(0, 1)
        inner.setStretchFactor(1, 0)
        inner.setChildrenCollapsible(False)
        inner.setSizes([820, 300])

        central = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(central)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        hbox.addWidget(self._rail)
        hbox.addWidget(inner, 1)
        self.setCentralWidget(central)

        self._rail.selected.connect(self._on_rail_selected)
        self._init_status_fields()

    def _init_status_fields(self) -> None:
        """Campos PERMANENTES da BottomBar — paridade com o monólito (Lote 2).

        Espelha a status bar do SM monolítico: ``Exp`` · ``● Estado`` · ``Elapsed`` ·
        ``Throughput`` · ``Cache`` · ``Plot`` · ``Binding``. Ficam à direita
        (``addPermanentWidget``). ``Binding`` é sempre-ligado (do ``qt_compat``); os
        demais são atualizados pela perspectiva ativa via os setters expostos em
        ``ctx.extras["status_bar"]`` (sem acoplar o ViewModel ao shell).
        """
        from geosteering_ai.gui.qt_compat import QT_BINDING

        bar = self.statusBar()
        self._sb_exp = QtWidgets.QLabel("Exp: —")
        self._sb_state = QtWidgets.QLabel("● ocioso")
        self._sb_elapsed = QtWidgets.QLabel("Elapsed: —")
        self._sb_throughput = QtWidgets.QLabel("Throughput: —")
        self._sb_cache = QtWidgets.QLabel("Cache: 0")
        self._sb_plot = QtWidgets.QLabel("Plot: —")
        self._sb_binding = QtWidgets.QLabel(f"Binding: {QT_BINDING or '—'}")
        for widget in (
            self._sb_exp,
            self._sb_state,
            self._sb_elapsed,
            self._sb_throughput,
            self._sb_cache,
            self._sb_plot,
            self._sb_binding,
        ):
            widget.setProperty("role", "hint")
            bar.addPermanentWidget(widget)
        # Setters expostos à perspectiva (None-safe: a perspectiva guarda contra
        # teardown do widget). ``set_state`` recebe a string já formatada (com ●).
        self.ctx.extras["status_bar"] = {
            "set_exp": lambda text: self._sb_exp.setText(f"Exp: {text}"),
            "set_state": lambda text: self._sb_state.setText(str(text)),
            "set_elapsed": lambda text: self._sb_elapsed.setText(f"Elapsed: {text}"),
            "set_throughput": (
                lambda text: self._sb_throughput.setText(f"Throughput: {text}")
            ),
            "set_cache": lambda text: self._sb_cache.setText(f"Cache: {text}"),
            "set_plot": lambda text: self._sb_plot.setText(f"Plot: {text}"),
        }

    # ── Accessors do host (rail + stack) ───────────────────────────────────
    def _host_count(self) -> int:
        return int(self._stack.count())

    def _host_widget(self, index: int) -> Any:
        return self._stack.widget(index)

    def _host_current_index(self) -> int:
        return int(self._stack.currentIndex())

    def _host_insert(
        self, index: int, container: Any, perspective: Perspective
    ) -> None:
        """Insere o container no stack + um item correspondente na activity rail."""
        self._stack.insertWidget(index, container)
        glyph = getattr(perspective, "icon_glyph", "") or (perspective.title[:1] or "●")
        enabled = bool(getattr(perspective, "enabled", True))
        self._rail.insert_item(index, glyph, perspective.title, enabled=enabled)
        # A 1ª perspectiva HABILITADA vira a ativa (rail marcada + stack nela).
        if enabled and self._enabled_count() == 1:
            self._rail.set_current(index)
            self._stack.setCurrentIndex(index)

    # ── Internos ───────────────────────────────────────────────────────────
    def _on_rail_selected(self, index: int) -> None:
        """Slot da rail — troca a perspectiva ativa no stack (lazy-build)."""
        self._stack.setCurrentIndex(index)
        self._ensure_built(index)

    def _enabled_count(self) -> int:
        """Quantas perspectivas habilitadas já foram registradas (para 1ª seleção)."""
        return sum(
            1 for p in self._by_container.values() if bool(getattr(p, "enabled", True))
        )
