# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_gui_mvvm_base.py                                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0005-gui-mvvm-base                                         ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — base MVVM (VMSignal/BaseViewModel/Perspective/MWBase) ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre os ACs da spec 0005. Os testes de VMSignal/BaseViewModel/        ║
# ║    Perspective são PUROS (sem pytest-qt). A FRONTEIRA de import (módulos  ║
# ║    puros NÃO carregam Qt) é verificada por subprocess. MainWindowBase é    ║
# ║    testado com QApplication offscreen (headless).                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0005 — base MVVM (VMSignal · BaseViewModel · Perspective · MWBase)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Qt headless — mesmo padrão de tests/test_simulation_parameters_seed.py:26. Setado no
# IMPORT do módulo (antes da coleta) → o QApplication dos testes Qt sobe offscreen mesmo
# sem DISPLAY/xvfb e sem conflitar com pytest-qt na sessão geral.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ════════════════════════════════════════════════════════════════════════════
# RF-1 — VMSignal (puro, sem Qt)
# ════════════════════════════════════════════════════════════════════════════
def test_vmsignal_connect_emit():
    """AC-1.2 — connect + emit entrega o valor ao callback."""
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    s = VMSignal()
    rec: list = []
    s.connect(rec.append)
    s.emit(42)
    assert rec == [42]


def test_vmsignal_connect_idempotent_and_disconnect():
    """AC-1.3 — connect idempotente; disconnect para de notificar (no-op se ausente)."""
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    s = VMSignal()
    rec: list = []
    cb = rec.append
    s.connect(cb)
    s.connect(cb)  # idempotente — não duplica
    assert len(s) == 1
    s.emit(1)
    assert rec == [1]  # uma única entrega
    s.disconnect(cb)
    s.disconnect(cb)  # no-op (já desconectado)
    s.emit(2)
    assert rec == [1]  # nada novo


def test_vmsignal_multiple_args():
    """AC-1.4 — emit repassa múltiplos args/kwargs ao callback."""
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    s = VMSignal()
    rec: list = []
    s.connect(lambda *a, **k: rec.append((a, k)))
    s.emit("name", 3.14, unit="Hz")
    assert rec == [(("name", 3.14), {"unit": "Hz"})]


def test_vmsignal_isolates_callback_exception(caplog):
    """AC-1.5 — exceção em um callback NÃO impede os demais (e é logada)."""
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    s = VMSignal()
    rec: list = []

    def boom(_):
        raise ValueError("falha proposital")

    s.connect(boom)
    s.connect(rec.append)
    s.emit(7)
    assert rec == [7]  # o segundo callback recebeu apesar do 1º falhar
    assert any(
        "VMSignal" in r.message or "isolado" in r.message for r in caplog.records
    )


def test_vmsignal_clear_removes_all_callbacks():
    """AC-1.6 — clear() remove TODOS os callbacks (emit posterior é no-op)."""
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    s = VMSignal()
    rec: list = []
    s.connect(rec.append)
    s.connect(lambda _: rec.append("b"))
    assert len(s) == 2
    s.clear()
    assert len(s) == 0
    s.emit(99)
    assert rec == []  # nenhum callback recebeu


# ════════════════════════════════════════════════════════════════════════════
# RF-2 — BaseViewModel (puro, sem Qt)
# ════════════════════════════════════════════════════════════════════════════
def _make_counter_vm():
    from geosteering_ai.gui.viewmodels.base import BaseViewModel

    class CounterVM(BaseViewModel):
        _STATE_FIELDS = ("_value",)

        def __init__(self) -> None:
            super().__init__()
            self._value = 0

        @property
        def value(self) -> int:
            return self._value

        @value.setter
        def value(self, v: int) -> None:
            self._set("_value", v)

    return CounterVM


def test_baseviewmodel_property_emits_changed():
    """AC-2.2 — setar uma property via _set emite changed(name, value)."""
    CounterVM = _make_counter_vm()
    vm = CounterVM()
    rec: list = []
    vm.changed.connect(lambda name, value: rec.append((name, value)))
    vm.value = 5
    assert rec == [("_value", 5)]
    assert vm.value == 5
    # set para o MESMO valor não re-emite (no-op)
    vm.value = 5
    assert rec == [("_value", 5)]


def test_baseviewmodel_to_from_dict_roundtrip():
    """AC-2.3 — to_dict/from_dict preservam o estado serializável."""
    CounterVM = _make_counter_vm()
    vm = CounterVM()
    vm.value = 9
    data = vm.to_dict()
    assert data == {"_value": 9}
    vm2 = CounterVM.from_dict(data)
    assert vm2.to_dict() == {"_value": 9}
    assert vm2.value == 9


def test_baseviewmodel_from_dict_missing_field_keeps_default():
    """from_dict com campo ausente mantém o default (forward-compat de snapshots)."""
    CounterVM = _make_counter_vm()
    vm = CounterVM.from_dict({})  # nada serializado
    assert vm.value == 0  # default do __init__ preservado, sem KeyError


def test_baseviewmodel_set_array_state_does_not_crash():
    """_set com estado de array NumPy NÃO levanta ValueError ('==' ambíguo) e emite.

    Regressão da correção crítica: a comparação de deduplicação usa ``_equal``
    (seguro), então um ViewModel com estado de array funciona sem override e sem
    crash. O comportamento default é conservador (re-emite — nunca perde uma
    notificação).
    """
    import numpy as np

    from geosteering_ai.gui.viewmodels.base import BaseViewModel

    class ArrayVM(BaseViewModel):
        _STATE_FIELDS = ("_curve",)

        def __init__(self) -> None:
            super().__init__()
            self._curve = np.zeros(3)

        @property
        def curve(self) -> np.ndarray:
            return self._curve

        @curve.setter
        def curve(self, v: np.ndarray) -> None:
            self._set("_curve", v)

    vm = ArrayVM()
    rec: list = []
    vm.changed.connect(lambda name, value: rec.append(name))
    vm.curve = np.array([1.0, 2.0, 3.0])  # NÃO deve crashar
    assert rec == ["_curve"]
    assert np.array_equal(vm.curve, np.array([1.0, 2.0, 3.0]))


# ════════════════════════════════════════════════════════════════════════════
# RF-3 — Perspective ABC
# ════════════════════════════════════════════════════════════════════════════
def test_perspective_is_abstract():
    """AC-3.1 — subclasse sem build_view/build_viewmodel não instancia (TypeError)."""
    from geosteering_ai.gui.shell.perspective import Perspective

    class Incompleta(Perspective):
        id = "x"

    with pytest.raises(TypeError):
        Incompleta()  # abstratos não implementados


def test_perspective_defaults_and_concrete():
    """AC-3.3 — concreta instancia; on_activate/on_close têm default não-abstrato."""
    from geosteering_ai.gui.shell.context import AppContext
    from geosteering_ai.gui.shell.perspective import Perspective
    from geosteering_ai.gui.viewmodels.base import BaseViewModel

    class Concreta(Perspective):
        id, title, icon, order = "sim", "Simulação", "flask", 0

        def build_view(self, ctx):  # noqa: ANN001
            return object()  # placeholder (não-Qt no teste puro)

        def build_viewmodel(self, ctx: AppContext) -> BaseViewModel:
            return BaseViewModel()

    p = Concreta()
    assert p.on_activate() is None
    assert p.on_close() is True
    assert isinstance(p.build_viewmodel(AppContext()), BaseViewModel)


def test_perspective_on_close_can_veto():
    """on_close pode VETAR o fechamento retornando False (contrato blueprint §5.2)."""
    from geosteering_ai.gui.shell.context import AppContext
    from geosteering_ai.gui.shell.perspective import Perspective
    from geosteering_ai.gui.viewmodels.base import BaseViewModel

    class VetoPerspective(Perspective):
        id, title = "veto", "Veto"

        def build_view(self, ctx):  # noqa: ANN001
            return object()

        def build_viewmodel(self, ctx: AppContext) -> BaseViewModel:
            return BaseViewModel()

        def on_close(self) -> bool:
            return False  # estado sujo → veta o fechamento

    assert VetoPerspective().on_close() is False


# ════════════════════════════════════════════════════════════════════════════
# RNF-3 — Fronteira de import: módulos puros NÃO carregam Qt (subprocess)
# ════════════════════════════════════════════════════════════════════════════
def test_pure_mvvm_modules_do_not_import_qt():
    """AC-1.1/2.1/3.2 — importar a base pura NÃO importa PyQt6/PySide6 (Princípio X)."""
    code = (
        "import sys\n"
        "import geosteering_ai.gui.viewmodels.signal\n"
        "import geosteering_ai.gui.viewmodels.base\n"
        "import geosteering_ai.gui.shell  # AppContext + Perspective (Qt-free)\n"
        "import geosteering_ai.gui.shell.perspective\n"
        "import geosteering_ai.gui.shell.context\n"
        "bad = [m for m in ('PyQt6', 'PySide6') if m in sys.modules]\n"
        "assert not bad, f'Qt importado pela base pura: {bad}'\n"
        "print('PURE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-1000:]
    assert "PURE_OK" in proc.stdout


def test_main_window_base_not_reexported_from_shell_init():
    """RNF-3 (fronteira) — MainWindowBase (View Qt) NÃO é exposto por gui.shell.

    Garante que ``import geosteering_ai.gui.shell`` permaneça Qt-free: a casca Qt
    vive só no submódulo ``gui.shell.main_window_base`` (acesso explícito).
    """
    import geosteering_ai.gui.shell as shell

    assert "MainWindowBase" not in getattr(shell, "__all__", [])
    with pytest.raises(ImportError):
        from geosteering_ai.gui.shell import MainWindowBase  # noqa: F401


# ════════════════════════════════════════════════════════════════════════════
# RF-5 — MainWindowBase (Qt, headless offscreen)
# ════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def offscreen_app():
    """QApplication headless (offscreen) — sem necessidade de DISPLAY/xvfb."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from geosteering_ai.gui.qt_compat import QT_AVAILABLE, QtWidgets

    if not QT_AVAILABLE:
        pytest.skip("nenhum binding Qt6 no ambiente")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


def _make_recording_perspective(order: int = 0):
    from geosteering_ai.gui.qt_compat import QtWidgets
    from geosteering_ai.gui.shell.perspective import Perspective
    from geosteering_ai.gui.viewmodels.base import BaseViewModel

    class RecPerspective(Perspective):
        def __init__(self, order_: int) -> None:
            self.id = f"rec{order_}"
            self.title = f"Rec {order_}"
            self.icon = ""
            self.order = order_
            self.n_activate = 0
            self.n_build_view = 0
            self.calls: list[str] = []  # ordem de chamadas (activate antes de build)

        def build_view(self, ctx):  # noqa: ANN001
            self.n_build_view += 1
            self.calls.append("build")
            return QtWidgets.QLabel(self.title)

        def build_viewmodel(self, ctx) -> BaseViewModel:  # noqa: ANN001
            return BaseViewModel()

        def on_activate(self) -> None:
            self.n_activate += 1
            self.calls.append("activate")

    return RecPerspective(order)


@pytest.mark.gui
def test_main_window_base_hosts_perspectives_lazy(offscreen_app):
    """AC-5.1/5.2 — add_perspective adiciona aba; build é LAZY (no on_activate)."""
    from geosteering_ai.gui.shell.context import AppContext
    from geosteering_ai.gui.shell.main_window_base import MainWindowBase

    mw = MainWindowBase(AppContext(app_name="Teste 0005"))
    try:
        assert mw.windowTitle() == "Teste 0005"
        assert mw.statusBar() is not None

        p0 = _make_recording_perspective(order=0)
        mw.add_perspective(p0)
        assert mw.perspective_count == 1
        # a primeira aba é a atual → construída imediatamente (lazy-on-activate)
        assert p0.n_build_view == 1
        assert p0.n_activate == 1
        assert p0.calls == ["activate", "build"]  # on_activate precede build_view

        # segunda perspectiva: NÃO construída até ser ativada
        p1 = _make_recording_perspective(order=1)
        mw.add_perspective(p1)
        assert mw.perspective_count == 2
        assert p1.n_build_view == 0  # lazy: ainda não ativada

        mw._tabs.setCurrentIndex(1)  # ativa a 2ª aba
        assert p1.n_build_view == 1
        assert p1.n_activate == 1

        # reativar a 1ª aba NÃO reconstrói (idempotente)
        mw._tabs.setCurrentIndex(0)
        assert p0.n_build_view == 1
    finally:
        mw.deleteLater()


@pytest.mark.gui
def test_main_window_base_orders_by_perspective_order(offscreen_app):
    """add_perspective insere mantendo a ordem por ``perspective.order``."""
    from geosteering_ai.gui.shell.context import AppContext
    from geosteering_ai.gui.shell.main_window_base import MainWindowBase

    mw = MainWindowBase(AppContext())
    try:
        mw.add_perspective(_make_recording_perspective(order=2))  # "Rec 2"
        mw.add_perspective(
            _make_recording_perspective(order=0)
        )  # "Rec 0" → vai p/ índice 0
        assert mw._tabs.tabText(0) == "Rec 0"
        assert mw._tabs.tabText(1) == "Rec 2"
    finally:
        mw.deleteLater()
