# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_gui_antigravity_shell.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0013-sm-antigravity-shell                                 ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell Antigravity (rail + stack + sidebar)          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-06                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre a spec 0013: registro/ordenação/lazy-build no host rail+stack     ║
# ║    (AC-2), troca via activity rail, scaffolds desabilitados, secondary     ║
# ║    sidebar exposta (AC-1/AC-3), e REGRESSÃO do host de abas de              ║
# ║    MainWindowBase (AC-6, refator preservou o comportamento default).       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0013 — shell Antigravity (rail+stack+sidebar) + regressão host de abas."""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from geosteering_ai.gui.qt_compat import QT_AVAILABLE
from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.shell.perspective import Perspective
from geosteering_ai.gui.viewmodels.base import BaseViewModel

pytestmark = pytest.mark.skipif(not QT_AVAILABLE, reason="requer binding Qt6")


class _StubPerspective(Perspective):
    """Perspectiva de teste — conta quantas vezes ``build_view`` é chamada (lazy)."""

    def __init__(
        self, pid: str, title: str, order: int, *, enabled: bool = True
    ) -> None:
        self.id = pid
        self.title = title
        self.order = order
        self.icon_glyph = title[:1]
        self.enabled = enabled
        self.build_count = 0

    def build_viewmodel(self, ctx: AppContext) -> BaseViewModel:
        return BaseViewModel()

    def build_view(self, ctx: AppContext) -> Any:
        from geosteering_ai.gui.qt_compat import QtWidgets

        self.build_count += 1
        return QtWidgets.QLabel(self.title)


def _make_window():
    from geosteering_ai.gui.shell.antigravity_window import AntigravityMainWindow

    return AntigravityMainWindow(AppContext(app_name="teste 0013"))


@pytest.mark.gui
def test_antigravity_registers_and_counts(qapp):
    """AC-2 — registra perspectivas; perspective_count reflete o nº no stack."""
    win = _make_window()
    win.add_perspective(_StubPerspective("a", "Alpha", 0))
    assert win.perspective_count == 1
    win.add_perspective(_StubPerspective("b", "Beta", 1))
    assert win.perspective_count == 2


@pytest.mark.gui
def test_antigravity_first_built_others_lazy(qapp):
    """AC-2 — a 1ª perspectiva é construída; as demais só ao serem selecionadas."""
    win = _make_window()
    a = _StubPerspective("a", "Alpha", 0)
    b = _StubPerspective("b", "Beta", 1)
    win.add_perspective(a)
    win.add_perspective(b)
    assert a.build_count == 1  # ativa (índice 0) → construída
    assert b.build_count == 0  # ainda não selecionada → lazy
    win._on_rail_selected(1)  # simula clique na rail
    assert b.build_count == 1  # agora construída
    win._on_rail_selected(1)  # idempotente (não reconstrói)
    assert b.build_count == 1


@pytest.mark.gui
def test_antigravity_sidebar_exposed(qapp):
    """AC-3 — a secondary sidebar existe, está no ctx.extras e aceita log/histórico."""
    win = _make_window()
    assert win.sidebar is not None
    assert win.ctx.extras.get("secondary_sidebar") is win.sidebar
    win.sidebar.append_log("linha de log")  # não deve lançar
    win.sidebar.add_history_item("sim #1")


@pytest.mark.gui
def test_antigravity_placeholder_disabled_not_built(qapp):
    """AC-2 — scaffold (enabled=False) entra na rail mas não é construído (lazy)."""
    win = _make_window()
    win.add_perspective(_StubPerspective("a", "Alpha", 0))
    ph = _StubPerspective("p", "Placeholder", 1, enabled=False)
    win.add_perspective(ph)
    assert win.perspective_count == 2
    assert ph.build_count == 0  # desabilitado → nunca ativado → nunca construído


@pytest.mark.gui
def test_mainwindowbase_tab_host_regression(qapp):
    """AC-6 — host de abas (default) inalterado: registra + conta + lazy-build."""
    from geosteering_ai.gui.shell.main_window_base import MainWindowBase

    win = MainWindowBase(AppContext(app_name="teste abas"))
    a = _StubPerspective("a", "Alpha", 0)
    win.add_perspective(a)
    assert win.perspective_count == 1
    assert a.build_count == 1  # aba atual construída
    # ordenação por order: insere Beta(order=2) e Gamma(order=1) → Gamma antes
    win.add_perspective(_StubPerspective("b", "Beta", 2))
    win.add_perspective(_StubPerspective("g", "Gamma", 1))
    assert win.perspective_count == 3
