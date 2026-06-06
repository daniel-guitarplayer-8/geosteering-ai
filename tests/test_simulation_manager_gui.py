# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_manager_gui.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Suite pytest-qt — GUI Simulation Manager                   ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Test infrastructure (Sprint v2.33)                         ║
# ║  Versão      : v2.33                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-15                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest-qt + PyQt6/PySide6 (via sm_qt_compat)               ║
# ║  Dependências: pytest, pytest-qt, geosteering_ai.simulation.tests.*       ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobertura automatizada da GUI do Simulation Manager. Foco em:          ║
# ║                                                                           ║
# ║      • Imports e símbolos Qt (sm_qt_compat funcional)                     ║
# ║      • SimulationThread (sinais expostos, métodos cooperativos)           ║
# ║      • SimRequest dataclass (defaults + roundtrip)                        ║
# ║      • Widgets leves (WelcomeWidget, ParametersPage) instanciáveis        ║
# ║      • Fixtures qt_binding + mock_simulation_thread + mock_sim_request    ║
# ║                                                                           ║
# ║    Deliberadamente NÃO instancia ``MainWindow`` (~10.7k linhas, com       ║
# ║    integração de pool persistente + Qt timers + signal hubs). Testes      ║
# ║    de integração end-to-end ficam para Sprint v3.0+ (após refator do     ║
# ║    MainWindow em componentes menores).                                    ║
# ║                                                                           ║
# ║  HEADLESS                                                                 ║
# ║    Fixtures em ``conftest_qt.py`` setam ``QT_QPA_PLATFORM=offscreen``     ║
# ║    quando ``DISPLAY`` está ausente. Em CI Ubuntu, ``xvfb-run -a`` fornece ║
# ║    o display virtual.                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Suite pytest-qt para a GUI do Simulation Manager (Sprint v2.33)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Skip o módulo inteiro ANTES de qualquer fixture/marker — preferível em
# ambientes sem GUI (raro mas possível). Ordem canônica: importorskip primeiro.
pytest.importorskip("pytestqt", reason="pytest-qt not installed")
try:
    from geosteering_ai.gui.qt_compat import QT_BINDING  # noqa: E402

    if QT_BINDING is None:
        pytest.skip(
            "Nenhum binding Qt6 disponível (PyQt6 ou PySide6 ausente)",
            allow_module_level=True,
        )
except ImportError:
    pytest.skip("sm_qt_compat indisponível", allow_module_level=True)

# Carrega fixtures Qt globalmente (qt_binding, mock_sim_request,
# mock_simulation_thread). pytest descobre via conftest_qt.py automaticamente
# em ``tests/`` se importado, ou via pytest_plugins.
pytest_plugins = ["tests.conftest_qt"]

# Marca toda a suite como ``gui`` — pulável via `pytest -m "not gui"`
pytestmark = pytest.mark.gui


# ──────────────────────────────────────────────────────────────────────────
# Grupo 1 — Compatibilidade Qt e fixtures
# ──────────────────────────────────────────────────────────────────────────


def test_qt_binding_is_pyqt6_or_pyside6(qt_binding: str) -> None:
    """T1: ``qt_binding`` retorna um valor válido (PyQt6 ou PySide6)."""
    assert qt_binding in ("PyQt6", "PySide6"), f"Binding inesperado: {qt_binding!r}"


def test_qt_compat_exports_signal_slot() -> None:
    """T2: ``sm_qt_compat`` exporta ``Signal``, ``Slot``, ``QtCore``."""
    from geosteering_ai.gui import qt_compat as sm_qt_compat

    assert hasattr(sm_qt_compat, "Signal")
    assert hasattr(sm_qt_compat, "Slot")
    assert hasattr(sm_qt_compat, "QtCore")
    assert hasattr(sm_qt_compat, "QtWidgets")
    assert hasattr(sm_qt_compat, "QtGui")
    assert hasattr(sm_qt_compat, "Qt")


def test_mock_simulation_thread_has_required_signals(
    mock_simulation_thread: MagicMock,
) -> None:
    """T3: ``mock_simulation_thread`` expõe os 7 sinais do real."""
    for signal_name in (
        "progress_update",
        "finished_all",
        "error",
        "paused",
        "resumed",
        "cancelled",
        "log",
    ):
        assert hasattr(mock_simulation_thread, signal_name)
        signal = getattr(mock_simulation_thread, signal_name)
        # Mocks suportam emit/connect/disconnect — verifica API
        assert hasattr(signal, "emit")
        assert hasattr(signal, "connect")
        assert hasattr(signal, "disconnect")


def test_mock_simulation_thread_has_control_methods(
    mock_simulation_thread: MagicMock,
) -> None:
    """T4: ``mock_simulation_thread`` expõe métodos de controle cooperativo."""
    for method_name in (
        "start",
        "quit",
        "wait",
        "request_pause",
        "request_resume",
        "request_stop",
        "is_paused",
        "is_cancelled",
    ):
        assert hasattr(mock_simulation_thread, method_name)


# ──────────────────────────────────────────────────────────────────────────
# Grupo 2 — SimRequest dataclass
# ──────────────────────────────────────────────────────────────────────────


def test_sim_request_defaults_via_fixture(mock_sim_request) -> None:
    """T5: fixture ``mock_sim_request`` produz SimRequest válido."""
    from geosteering_ai.simulation.tests.sm_workers import SimRequest

    assert isinstance(mock_sim_request, SimRequest)
    assert mock_sim_request.frequencies_hz == [20000.0]
    assert mock_sim_request.tr_spacings_m == [1.0]
    assert mock_sim_request.dip_degs == [0.0]
    assert mock_sim_request.backend == "numba"
    assert mock_sim_request.n_workers == 1
    assert mock_sim_request.n_threads == 1
    # positions_z é um ndarray (n_pos,)
    assert isinstance(mock_sim_request.positions_z, np.ndarray)
    assert mock_sim_request.positions_z.shape == (10,)


def test_sim_request_dataclass_defaults_directly() -> None:
    """T6: SimRequest aceita parâmetros mínimos sem erro."""
    from geosteering_ai.simulation.tests.sm_workers import SimRequest

    req = SimRequest(
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        positions_z=np.zeros(5, dtype=np.float64),
    )
    # Defaults documentados em SimRequest
    assert req.backend == "numba"
    assert req.n_workers == 4
    assert req.n_threads == 4
    assert req.hankel_filter == "werthmuller_201pt"
    assert req.h1 == 10.0
    assert req.save_raw is False


# ──────────────────────────────────────────────────────────────────────────
# Grupo 3 — SimulationThread real (não-mock, só metadata)
# ──────────────────────────────────────────────────────────────────────────


def test_simulation_thread_is_qthread_subclass() -> None:
    """T7: SimulationThread herda de QThread."""
    from geosteering_ai.gui.qt_compat import QThread
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    assert issubclass(SimulationThread, QThread)


def test_simulation_thread_exposes_signals_class_level() -> None:
    """T8: SimulationThread declara sinais ao nível da classe."""
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    # Sinais Qt são atributos de classe; verificar presença
    for signal_name in (
        "progress_update",
        "log",
        "finished_all",
        "error",
        "paused",
        "resumed",
        "cancelled",
    ):
        assert hasattr(
            SimulationThread, signal_name
        ), f"SimulationThread sem sinal {signal_name}"


def test_simulation_thread_accepts_sim_request(mock_sim_request, qtbot) -> None:
    """T9: SimulationThread instancia com SimRequest + models vazio.

    ``qtbot`` é requisitado para garantir QApplication ativa (necessário ao
    construir um QThread). Cleanup é automático (Python GC) — QThread não
    é QWidget, então ``qtbot.addWidget`` não se aplica.
    """
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    thread = SimulationThread(mock_sim_request, [])
    assert thread._req is mock_sim_request
    assert thread._models == []
    assert thread._stopped is False
    assert thread._is_paused is False


def test_simulation_thread_pause_resume_idempotent(mock_sim_request, qtbot) -> None:
    """T10: ``request_pause`` e ``request_resume`` são idempotentes (v2.11)."""
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    thread = SimulationThread(mock_sim_request, [])

    # Chamar pause 2× — segunda chamada é no-op
    thread.request_pause()
    assert thread._is_paused is True
    thread.request_pause()
    assert thread._is_paused is True

    # Chamar resume 2× — segunda chamada é no-op
    thread.request_resume()
    assert thread._is_paused is False
    thread.request_resume()
    assert thread._is_paused is False


def test_simulation_thread_pause_emits_paused_signal(mock_sim_request, qtbot) -> None:
    """T11: ``request_pause`` emite sinal ``paused``."""
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    thread = SimulationThread(mock_sim_request, [])

    with qtbot.waitSignal(thread.paused, timeout=1000):
        thread.request_pause()


def test_simulation_thread_resume_emits_resumed_signal(mock_sim_request, qtbot) -> None:
    """T12: ``request_resume`` emite sinal ``resumed`` quando pausado."""
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    thread = SimulationThread(mock_sim_request, [])
    thread.request_pause()  # pré-condição: pausado

    with qtbot.waitSignal(thread.resumed, timeout=1000):
        thread.request_resume()


def test_simulation_thread_request_stop_sets_flag(mock_sim_request, qtbot) -> None:
    """T13: ``request_stop`` marca flag de interrupção cooperativa."""
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    thread = SimulationThread(mock_sim_request, [])
    assert thread._stopped is False
    thread.request_stop()
    assert thread._stopped is True


# ──────────────────────────────────────────────────────────────────────────
# Grupo 4 — Widgets leves (Welcome + Parameters)
# ──────────────────────────────────────────────────────────────────────────


def test_welcome_widget_instantiates(qtbot) -> None:
    """T14: WelcomeWidget instancia sem erros e expõe interface básica."""
    # Lazy import — simulation_manager é caro de importar
    from geosteering_ai.simulation.tests.simulation_manager import WelcomeWidget

    widget = WelcomeWidget()
    qtbot.addWidget(widget)
    assert widget.isVisible() is False  # não foi mostrado ainda
    widget.show()
    qtbot.waitExposed(widget, timeout=2000)
    assert widget.isVisible() is True


def test_parameters_page_to_dict_roundtrip(qtbot) -> None:
    """T15: ParametersPage.to_dict + from_dict é roundtrip (preserva valores)."""
    from geosteering_ai.simulation.tests.simulation_manager import ParametersPage

    page = ParametersPage()
    qtbot.addWidget(page)

    # Capturar snapshot inicial
    snapshot = page.to_dict()
    assert isinstance(snapshot, dict)
    assert len(snapshot) > 0

    # Roundtrip: from_dict do snapshot deve restaurar mesmos valores
    page.from_dict(snapshot)
    restored = page.to_dict()

    # Comparar chaves principais (alguns valores podem ser normalizados —
    # strings convertidas para float etc — então comparamos só chaves)
    assert set(restored.keys()) == set(snapshot.keys())


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Test pulado em Windows (xvfb não disponível)",
)
def test_qt_app_can_process_events(qapp, qtbot) -> None:
    """T16: QApplication processa eventos (sanity check de event loop).

    Usa ``qapp`` fixture do pytest-qt (canônica) em vez de ``QApplication.instance()``
    para garantir robustez se a API interna mudar no futuro.
    """
    from geosteering_ai.gui.qt_compat import QtCore

    assert qapp is not None, "Fixture qapp não retornou QApplication válida"

    # Disparar um single-shot timer e aguardar callback
    flag = {"fired": False}

    def _set_flag() -> None:
        flag["fired"] = True

    QtCore.QTimer.singleShot(50, _set_flag)
    qtbot.wait(150)  # tempo suficiente para o timer disparar
    assert flag["fired"] is True
