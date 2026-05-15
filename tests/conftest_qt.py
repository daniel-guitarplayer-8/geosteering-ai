# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/conftest_qt.py                                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Fixtures pytest-qt para suite GUI                          ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Test infrastructure (Sprint v2.33)                         ║
# ║  Versão      : v2.33                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-15                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest-qt + PyQt6/PySide6 (via sm_qt_compat)               ║
# ║  Dependências: pytest, pytest-qt, sm_qt_compat                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Fornece fixtures compartilhadas pelos testes pytest-qt da GUI:         ║
# ║                                                                           ║
# ║      • ``qt_app``: QApplication singleton com headless support            ║
# ║      • ``mock_simulation_thread``: stub de ``SimulationThread`` que       ║
# ║         emite sinais sintéticos via ``QTimer.singleShot``                 ║
# ║      • ``mock_sim_request``: builder de ``SimRequest`` mínimo p/ tests    ║
# ║                                                                           ║
# ║    Os testes usam ``qtbot`` (fixture do plugin pytest-qt) para interagir  ║
# ║    com a UI e aguardar sinais. As fixtures aqui não substituem ``qtbot`` —║
# ║    complementam com objetos específicos do Simulation Manager.            ║
# ║                                                                           ║
# ║  HEADLESS                                                                 ║
# ║    Em Linux CI, ``QT_QPA_PLATFORM=offscreen`` é setado automaticamente    ║
# ║    se ``DISPLAY`` não estiver presente. macOS/Windows dev rodam com       ║
# ║    backend nativo (cocoa/windows).                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Fixtures pytest-qt para suite GUI do Simulation Manager (Sprint v2.33)."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

# Configurar Qt platform ANTES de qualquer import Qt — necessário em CI Linux
# headless. ``DISPLAY`` indica X11 disponível (xvfb-run define DISPLAY=:99 etc).
if "DISPLAY" not in os.environ and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Sincronizar binding entre pytest-qt e sm_qt_compat. Sem este setup, pytest-qt
# pode escolher PySide6 enquanto sm_qt_compat já carregou PyQt6 (ou vice-versa),
# causando ``TypeError: Need to pass a QWidget to addWidget`` por isinstance
# cross-binding falso. ``QT_API`` é honrado pelo pytest-qt (e por ``qtpy``).
# Estratégia: tentar PyQt6 primeiro (preferido por sm_qt_compat). Se ausente,
# cair para PySide6.
if "QT_API" not in os.environ:
    try:
        import PyQt6.QtCore  # noqa: F401

        os.environ["QT_API"] = "pyqt6"
    except ImportError:
        try:
            import PySide6.QtCore  # noqa: F401

            os.environ["QT_API"] = "pyside6"
        except ImportError:
            # Nenhum binding — testes serão skipped pelo conftest da suite GUI
            pass


@pytest.fixture(scope="session")
def qt_binding() -> str:
    """Retorna o binding Qt em uso (``"PyQt6"`` ou ``"PySide6"``).

    Útil para testes que precisam ramificar lógica entre bindings (raro,
    pois ``sm_qt_compat`` cobre 99% das diferenças).

    Returns:
        String com o nome do binding ativo.

    Raises:
        RuntimeError: se nenhum binding Qt6 estiver disponível (testes GUI
            devem ser skipados antes via marker; esta fixture sinaliza
            ambiente inválido com mensagem clara).

    Example:
        >>> def test_signal_type(qt_binding):
        ...     if qt_binding == "PyQt6":
        ...         assert ...
    """
    from geosteering_ai.simulation.tests.sm_qt_compat import QT_BINDING

    if QT_BINDING is None:
        raise RuntimeError(
            "Nenhum binding Qt6 disponível (PyQt6 ou PySide6 ausente). "
            "Testes GUI exigem um binding instalado."
        )
    return QT_BINDING


@pytest.fixture
def mock_sim_request() -> Any:
    """Constrói um ``SimRequest`` mínimo válido para testes da GUI.

    Returns:
        Objeto ``SimRequest`` com parâmetros default (10 posições em -5..5 m,
        1 frequência 20 kHz, 1 TR 1.0 m, 1 dip 0°). Suficiente para validar
        que a UI aceita o request sem erros, sem rodar simulação real.

    Example:
        >>> def test_thread_accepts_request(qtbot, mock_sim_request):
        ...     thread = SimulationThread(mock_sim_request, [])
        ...     assert thread._req is mock_sim_request
    """
    # Lazy import — só quando o teste é chamado
    import numpy as np

    from geosteering_ai.simulation.tests.sm_workers import SimRequest

    return SimRequest(
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        positions_z=np.linspace(-5.0, 5.0, 10).astype(np.float64),
        backend="numba",
        n_workers=1,
        n_threads=1,
    )


@pytest.fixture
def mock_simulation_thread() -> MagicMock:
    """Substitui ``SimulationThread`` por ``MagicMock`` que emite sinais.

    O mock expõe os mesmos sinais (``progress_update``, ``finished_all``,
    ``error``, ``paused``, ``resumed``, ``cancelled``) e métodos
    (``start``, ``request_pause``, ``request_resume``, ``request_stop``)
    do ``SimulationThread`` real, mas SEM rodar simulação.

    Útil para validar que:
    - A UI conecta-se corretamente aos sinais (testar sinais emitidos)
    - Botões pause/cancel chamam métodos corretos do thread
    - A UI reage a ``error.emit(...)`` exibindo a mensagem

    Returns:
        MagicMock configurado com sinais Qt-like (objetos com ``.emit()``,
        ``.connect()``, ``.disconnect()``). Cada chamada a ``.emit()`` é
        rastreável via ``mock.signal.emit.call_args_list``.

    Example:
        >>> def test_progress_update(qtbot, mock_simulation_thread):
        ...     mock_simulation_thread.progress_update.emit(50, 100, 1234.5)
        ...     assert mock_simulation_thread.progress_update.emit.called
    """
    thread = MagicMock()

    # Mock dos sinais — cada um com ``emit``, ``connect``, ``disconnect``
    for signal_name in (
        "progress_update",
        "finished_all",
        "error",
        "paused",
        "resumed",
        "cancelled",
        "log",
    ):
        signal_mock = MagicMock()
        signal_mock.emit = MagicMock()
        signal_mock.connect = MagicMock()
        signal_mock.disconnect = MagicMock()
        setattr(thread, signal_name, signal_mock)

    # Mock dos métodos de controle cooperativo (v2.11)
    thread.start = MagicMock()
    thread.quit = MagicMock()
    thread.wait = MagicMock(return_value=True)
    thread.request_pause = MagicMock()
    thread.request_resume = MagicMock()
    thread.request_stop = MagicMock()
    thread.is_paused = MagicMock(return_value=False)
    thread.is_cancelled = MagicMock(return_value=False)
    thread.isRunning = MagicMock(return_value=False)
    thread.isFinished = MagicMock(return_value=True)

    return thread
