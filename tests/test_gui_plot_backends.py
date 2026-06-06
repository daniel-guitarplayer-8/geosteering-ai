# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_gui_plot_backends.py                                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0006-gui-plot-backends                                     ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — backends de plotagem (PlotCanvas ABC + 4 backends)   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre os ACs da spec 0006 (extração Strangler-Fig dos backends de      ║
# ║    plotagem p/ geosteering_ai/gui/plot_backends/). API + smoke offscreen  ║
# ║    (matplotlib/pyqtgraph) + retrocompat do shim (identidade + submódulo)  ║
# ║    + fix de portabilidade (plotly via qt_compat) + fronteira de import.   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0006 — backends de plotagem (PlotCanvas ABC · 4 backends · shim)."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Qt headless — mesmo padrão de tests/test_gui_mvvm_base.py. Setado no IMPORT do
# módulo (antes da coleta) → o QApplication dos smoke-tests sobe offscreen.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ════════════════════════════════════════════════════════════════════════════
# RF-1 — API do pacote gui/plot_backends (puro)
# ════════════════════════════════════════════════════════════════════════════
def test_plot_backends_api_imports():
    """AC-1.1 — a API canônica importa de geosteering_ai.gui.plot_backends."""
    from geosteering_ai.gui.plot_backends import (  # noqa: F401
        AxisConfig,
        PlotBackend,
        PlotCanvas,
        SubplotHandle,
        available_backends,
        make_canvas,
    )


def test_plotcanvas_is_abstract():
    """AC-1.2 — PlotCanvas é ABC; instanciá-la diretamente levanta TypeError."""
    from geosteering_ai.gui.plot_backends import PlotCanvas

    with pytest.raises(TypeError):
        PlotCanvas()  # métodos abstratos não implementados


def test_available_backends_always_includes_matplotlib():
    """AC-1.3 — MATPLOTLIB sempre disponível; opcionais só se instalados."""
    from geosteering_ai.gui.plot_backends import PlotBackend, available_backends

    backends = available_backends()
    assert PlotBackend.MATPLOTLIB in backends
    # backends opcionais aparecem se-e-somente-se importáveis
    import importlib.util

    for name, member in [
        ("pyqtgraph", PlotBackend.PYQTGRAPH),
        ("plotly", PlotBackend.PLOTLY),
        ("vispy", PlotBackend.VISPY),
    ]:
        present = importlib.util.find_spec(name) is not None
        assert (member in backends) == present


def test_make_canvas_unknown_backend_raises():
    """make_canvas com backend desconhecido levanta ValueError (edge-case da factory)."""
    from geosteering_ai.gui.plot_backends import make_canvas

    with pytest.raises(ValueError):
        make_canvas(object())  # type: ignore[arg-type]  # não é um PlotBackend


# ════════════════════════════════════════════════════════════════════════════
# RF-1 — smoke offscreen (Qt headless)
# ════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def offscreen_app():
    """QApplication headless (offscreen) — sem necessidade de DISPLAY/xvfb."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from geosteering_ai.gui.qt_compat import QT_AVAILABLE, QtWidgets

    if not QT_AVAILABLE:
        pytest.skip("nenhum binding Qt6 no ambiente")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.gui
def test_matplotlib_canvas_smoke(offscreen_app):
    """AC-1.4 — MatplotlibCanvas: grid + line + hline + axis + draw + dark, sem erro."""
    import numpy as np

    from geosteering_ai.gui.plot_backends import AxisConfig, PlotBackend, make_canvas

    canvas = make_canvas(PlotBackend.MATPLOTLIB)
    assert canvas.widget() is not None
    grid = canvas.add_subplot_grid(1, 1)
    ax = grid[0][0]
    canvas.plot_line(
        ax, np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]), label="t"
    )
    canvas.add_hline(ax, 1.0)
    canvas.set_axis_config(ax, AxisConfig(title="t", xlabel="A/m", ylabel="z (m)"))
    canvas.draw()
    canvas.set_dark_mode(True)
    canvas.set_dark_mode(False)
    canvas.clear()


@pytest.mark.gui
def test_pyqtgraph_canvas_smoke(offscreen_app):
    """AC-1.4 (PyQtGraph) — backend de alta performance plota sem erro (skip se ausente)."""
    pytest.importorskip("pyqtgraph")
    import numpy as np

    from geosteering_ai.gui.plot_backends import PlotBackend, make_canvas

    canvas = make_canvas(PlotBackend.PYQTGRAPH)
    assert canvas.widget() is not None
    grid = canvas.add_subplot_grid(1, 1)
    canvas.plot_line(grid[0][0], np.array([0.0, 1.0]), np.array([0.0, 1.0]), label="t")
    canvas.draw()
    canvas.set_dark_mode(True)


# ════════════════════════════════════════════════════════════════════════════
# De-shim (spec 0011 Fase 0) — o shim de plot_backends foi REMOVIDO
# ════════════════════════════════════════════════════════════════════════════
def test_plot_backends_shim_removed():
    """Spec 0011 Fase 0 — o shim ``sm_plot_backends`` foi removido (consumidores em gui/)."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("geosteering_ai.simulation.tests.sm_plot_backends")


# ════════════════════════════════════════════════════════════════════════════
# RF-3/RF-4 — fix de portabilidade (plotly via qt_compat; pyqtgraph via qt_compat)
# ════════════════════════════════════════════════════════════════════════════
def test_plotly_canvas_has_no_hardcoded_pyqt6():
    """AC-3.1 — guard de regressão: plotly NÃO importa PyQt6.QtWebEngine hardcoded."""
    from geosteering_ai.gui.plot_backends import plotly_canvas

    src = inspect.getsource(plotly_canvas)
    assert "from PyQt6.QtWebEngineWidgets" not in src
    assert "load_qwebengineview" in src  # usa o helper portátil


def test_pyqtgraph_canvas_uses_qt_compat():
    """AC-3.3 — pyqtgraph importa Qt via gui.qt_compat (não sm_qt_compat/PyQt6 direto)."""
    from geosteering_ai.gui.plot_backends import pyqtgraph_canvas

    src = inspect.getsource(pyqtgraph_canvas)
    assert "sm_qt_compat" not in src
    assert "from geosteering_ai.gui.qt_compat import" in src


def test_load_qwebengineview_returns_class():
    """AC-3.2 — load_qwebengineview() resolve QWebEngineView do binding ativo.

    Roda em SUBPROCESSO fresco (sem QApplication prévia): o PyQt6 exige que
    QtWebEngine seja importado ANTES de um ``QCoreApplication`` — daí o processo
    isolado. Skipa se o módulo QtWebEngine não estiver instalado.
    """
    code = (
        "from geosteering_ai.gui.qt_compat import load_qwebengineview\n"
        "try:\n"
        "    v = load_qwebengineview()\n"
        "except ImportError:\n"
        "    print('SKIP_NO_WEBENGINE')\n"
        "    raise SystemExit(0)\n"
        "assert v.__name__ == 'QWebEngineView', v\n"
        "print('WEBENGINE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if "SKIP_NO_WEBENGINE" in proc.stdout:
        pytest.skip("Qt WebEngine ausente no ambiente")
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "WEBENGINE_OK" in proc.stdout


# ════════════════════════════════════════════════════════════════════════════
# RF-5 — fronteira de import (lazy: ABC sem Qt/backends pesados)
# ════════════════════════════════════════════════════════════════════════════
def test_plot_backends_import_adds_no_heavy_libs():
    """AC-5.1 — importar gui.plot_backends NÃO adiciona Qt/pyqtgraph/plotly/vispy (lazy).

    Usa delta sobre o baseline ``import geosteering_ai`` (cujo __init__ já carrega
    matplotlib via data/models) — assim o teste mede só o que o pacote ADICIONA.
    """
    code = (
        "import sys\n"
        "import geosteering_ai\n"  # baseline (parent pode puxar matplotlib)
        "before = set(sys.modules)\n"
        "import geosteering_ai.gui.plot_backends  # noqa\n"
        "added = set(sys.modules) - before\n"
        "heavy = {'PyQt6', 'PySide6', 'pyqtgraph', 'plotly', 'vispy', 'matplotlib'}\n"
        "bad = sorted(m for m in added if m.split('.')[0] in heavy)\n"
        "assert not bad, f'plot_backends puxou libs pesadas no import: {bad}'\n"
        "print('BOUNDARY_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "BOUNDARY_OK" in proc.stdout


def test_core_does_not_import_gui_plot_backends():
    """AC-5.2 — o core (simulation/…) não importa gui.plot_backends (fronteira)."""
    code = (
        "import sys\n"
        "import geosteering_ai.simulation.dispatch  # noqa\n"
        "import geosteering_ai.simulation.config  # noqa\n"
        "bad = 'geosteering_ai.gui.plot_backends' in sys.modules\n"
        "assert not bad, 'core puxou gui.plot_backends'\n"
        "print('CORE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "CORE_OK" in proc.stdout
