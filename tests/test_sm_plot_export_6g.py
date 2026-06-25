# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_plot_export_6g.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Fatia 6g — export da figura da galeria (PNG/PDF/SVG/EPS + DPI)            ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-25                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre o 6g (estreito): a ResultsView ganha um botão "Exportar figura…" ║
# ║    + DPI que salvam a figura CORRENTE via um MatplotlibCanvas OFF-SCREEN   ║
# ║    (replay do mesmo _draw_current_mode), honrando DPI/PDF/EPS independente ║
# ║    do backend on-screen. 100% apresentação — NÃO re-simula (sem física).   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do 6g — export da figura da galeria (PNG/PDF/SVG/EPS + DPI), sem re-simular."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apps.sim_manager.perspectives.simulation.results_viewmodel import (  # noqa: E402
    ResultsViewModel,
)


def _result(n_models: int = 4, n_pos: int = 10) -> dict:
    """Resultado sintético (H6 + positions_z + geologia) — mesmo shape do 6d."""
    rng = np.random.default_rng(7)
    shape = (n_models, 1, 1, n_pos, 1, 9)
    h6 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    geology = [
        {
            "rho_h": np.array([1.0, 10.0, 100.0]) * (m + 1),
            "rho_v": np.array([2.0, 20.0, 200.0]) * (m + 1),
            "thicknesses": np.array([4.0]),
        }
        for m in range(n_models)
    ]
    return {
        "H6": h6,
        "positions_z": np.linspace(0.0, 9.0, n_pos),
        "geology": geology,
        "n_models": n_models,
    }


def _view(result: dict | None = None):
    from apps.sim_manager.perspectives.simulation.results_view import ResultsView

    rvm = ResultsViewModel(page_size=4)
    if result is not None:
        rvm.set_result(result)
    return ResultsView(rvm)


# ════════════════════════════════════════════════════════════════════════════
# Toolbar: DPI + botão; gating por has_result
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_export_controls_present_and_gated(qtbot):
    """DPI (default 150, range 50-600) + botão; botão só habilita COM resultado."""
    view = _view()  # sem resultado
    qtbot.addWidget(view)
    assert view._dpi.value() == 150
    assert view._dpi.minimum() == 50 and view._dpi.maximum() == 600
    assert not view._export_btn.isEnabled()  # sem resultado → desabilitado
    view._vm.set_result(_result())
    assert view._export_btn.isEnabled()  # com resultado → habilitado


# ════════════════════════════════════════════════════════════════════════════
# Salvamento off-screen (matplotlib) — formatos + restauração do canvas
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
@pytest.mark.parametrize("ext", ["png", "pdf", "svg", "eps"])
def test_save_current_figure_writes_file(qtbot, tmp_path, ext):
    """Escreve um arquivo NÃO-vazio (matplotlib honra todos) e RESTAURA o canvas on-screen."""
    view = _view(_result())
    qtbot.addWidget(view)
    on_screen = view._canvas
    out = tmp_path / f"fig.{ext}"
    view._save_current_figure(str(out), dpi=120)
    assert out.exists() and out.stat().st_size > 0, f"{ext}: arquivo vazio/ausente"
    assert view._canvas is on_screen  # canvas on-screen RESTAURADO (finally)


@pytest.mark.gui
@pytest.mark.parametrize("mode", ["curve", "rho", "lambda", "heatmap"])
def test_export_all_modes(qtbot, tmp_path, mode):
    """Exporta em TODOS os modos (curva/ρ/λ/heatmap) — caminho único _draw_current_mode."""
    view = _view(_result())
    qtbot.addWidget(view)
    view._vm.plot_mode = mode
    out = tmp_path / f"fig_{mode}.png"
    view._save_current_figure(str(out), dpi=100)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.gui
def test_export_does_not_resimulate(qtbot, tmp_path):
    """Export NÃO mexe no resultado (mesmo objeto H6) nem re-simula — pura apresentação."""
    res = _result()
    view = _view(res)
    qtbot.addWidget(view)
    before = view._vm.curve_for(0).copy()
    view._save_current_figure(str(tmp_path / "x.png"), dpi=100)
    assert view._vm.has_result
    assert np.array_equal(view._vm.curve_for(0), before)  # dados intactos


# ════════════════════════════════════════════════════════════════════════════
# Fluxo via QFileDialog (mockado) + no-op sem resultado
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_export_figure_via_dialog(qtbot, tmp_path, monkeypatch):
    """_export_figure usa QFileDialog (mockado) → salva no path retornado."""
    from geosteering_ai.gui.qt_compat import QtWidgets

    view = _view(_result())
    qtbot.addWidget(view)
    out = tmp_path / "dialog.png"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(out), "PNG (*.png)"),
    )
    view._export_figure()
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.gui
def test_export_cancel_dialog_is_noop(qtbot, monkeypatch):
    """Cancelar o diálogo (path vazio) → no-op, sem exceção."""
    from geosteering_ai.gui.qt_compat import QtWidgets

    view = _view(_result())
    qtbot.addWidget(view)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: ("", "")
    )
    view._export_figure()  # não deve levantar


@pytest.mark.gui
def test_export_no_result_skips_dialog(qtbot, monkeypatch):
    """Sem resultado, _export_figure é no-op e NÃO abre o diálogo (guard has_result)."""
    from geosteering_ai.gui.qt_compat import QtWidgets

    view = _view()  # sem resultado
    qtbot.addWidget(view)
    called = {"dialog": False}

    def _fake(*a, **k):
        called["dialog"] = True
        return ("", "")

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", _fake)
    view._export_figure()
    assert called["dialog"] is False


# ════════════════════════════════════════════════════════════════════════════
# Revisão Turn 7 — normalização de extensão + gate alinhado ao guard
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_export_appends_extension_from_filter(qtbot, tmp_path, monkeypatch):
    """Path SEM extensão + filtro PDF → anexa .pdf (não salva PNG silencioso)."""
    from geosteering_ai.gui.qt_compat import QtWidgets

    view = _view(_result())
    qtbot.addWidget(view)
    base = tmp_path / "relatorio"  # SEM extensão
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(base), "PDF (*.pdf)"),
    )
    view._export_figure()
    assert (tmp_path / "relatorio.pdf").exists()
    assert not (tmp_path / "relatorio.png").exists()  # NÃO caiu em PNG


@pytest.mark.gui
def test_export_button_disabled_without_depth(qtbot):
    """H6 válido mas SEM positions_z → botão DESABILITADO (gate == guard de export)."""
    res = _result()
    res.pop("positions_z")  # H6 ok, depth None
    view = _view()
    qtbot.addWidget(view)
    view._vm.set_result(res)
    assert view._vm.has_result  # H6 presente
    assert view._vm.depth is None  # sem positions_z
    assert not view._export_btn.isEnabled()  # gate alinhado ao guard (não vira no-op)
