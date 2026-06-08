# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_parallelism_output.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Lote 1 — paridade SimulatorView ↔ monólito                               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM MVVM — paralelismo (workers/threads) + saída + contadores║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-08                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre o Lote 1: estado workers/threads (clamp + sessão), wire REAL de    ║
# ║    num_threads (NÃO muda resultados — paridade), escrita .dat/.out, e os     ║
# ║    widgets da SimulatorView (paralelismo/aviso/contadores) + overflow +     ║
# ║    status bar.                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do Lote 1 — paralelismo (workers/threads) + saída Fortran + contadores."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from geosteering_ai.gui.services.sim_request import (  # noqa: E402
    SimRequest,
    _apply_thread_count,
    _run_simulation,
)


def _make_sim_vm():
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    class _Stub:
        def __init__(self) -> None:
            self.finished = VMSignal()
            self.error = VMSignal()
            self.progress = VMSignal()
            self.requests: list = []

        def run(self, request) -> None:  # noqa: ANN001
            self.requests.append(request)

    return SimulationViewModel(service=_Stub())


# ════════════════════════════════════════════════════════════════════════════
# VM — estado workers/threads + saída (clamp + roundtrip .session)
# ════════════════════════════════════════════════════════════════════════════
def test_vm_parallelism_clamp_and_session():
    vm = _make_sim_vm()
    vm.n_workers = 999
    vm.threads_per_worker = 0
    assert vm.n_workers == 256 and vm.threads_per_worker == 1  # clamp [1, 256]
    vm.n_workers = 8
    vm.threads_per_worker = 2
    vm.output_dir = "/tmp/sm_out"
    vm.save_fortran_artifacts = True
    d = vm.to_session_dict()
    for k in (
        "n_workers",
        "threads_per_worker",
        "output_dir",
        "save_fortran_artifacts",
    ):
        assert k in d
    vm2 = _make_sim_vm()
    vm2.load_session_dict(d)
    assert vm2.n_workers == 8 and vm2.threads_per_worker == 2
    assert vm2.output_dir == "/tmp/sm_out" and vm2.save_fortran_artifacts is True


def test_vm_run_passes_parallelism_to_request():
    vm = _make_sim_vm()
    vm.n_workers = 6
    vm.threads_per_worker = 3
    vm.output_dir = "/tmp/x"
    vm.save_fortran_artifacts = True
    vm.run()
    req = vm._service.requests[-1]
    assert req.n_workers == 6 and req.threads_per_worker == 3
    assert req.output_dir == "/tmp/x" and req.save_fortran_artifacts is True


# ════════════════════════════════════════════════════════════════════════════
# num_threads — EFEITO REAL mas resultados BIT-IDÊNTICOS (fidelidade)
# ════════════════════════════════════════════════════════════════════════════
def test_apply_thread_count_noop_and_real():
    # ≤0 = no-op; valores válidos não levantam (best-effort).
    _apply_thread_count(0)
    _apply_thread_count(-5)
    _apply_thread_count(2)  # aplica (ou no-op se numba ausente) — nunca levanta


def test_num_threads_does_not_change_results():
    """set_num_threads é knob de threading: 1 vs N threads → H6 BIT-IDÊNTICO."""
    base = dict(geology_mode="fixed", n_models=2, frequencies_hz=(20000.0,), tj=10.0)
    h_1 = _run_simulation(SimRequest(threads_per_worker=1, **base))["H6"]
    h_n = _run_simulation(SimRequest(threads_per_worker=4, **base))["H6"]
    assert np.array_equal(h_1, h_n), "threading NÃO pode alterar resultados (paridade)"


# ════════════════════════════════════════════════════════════════════════════
# Saída — escrita .dat (22-col) + .out (ASCII), best-effort
# ════════════════════════════════════════════════════════════════════════════
def test_run_simulation_writes_artifacts(tmp_path):
    out = _run_simulation(
        SimRequest(
            geology_mode="fixed",
            n_models=2,
            frequencies_hz=(20000.0,),
            tj=10.0,
            output_dir=str(tmp_path),
            save_fortran_artifacts=True,
        )
    )
    assert out["artifacts_error"] is None
    assert out["artifacts_path"] == str(tmp_path / "sm_output.dat")
    assert (tmp_path / "sm_output.dat").is_file()
    assert (tmp_path / "sm_output.out").is_file()
    assert (tmp_path / "sm_output.dat").stat().st_size > 0


def test_run_simulation_artifacts_off_by_default(tmp_path):
    out = _run_simulation(
        SimRequest(geology_mode="fixed", n_models=1, tj=10.0, output_dir=str(tmp_path))
    )
    # save_fortran_artifacts=False → não grava, sem erro.
    assert out["artifacts_path"] is None and out["artifacts_error"] is None
    assert not (tmp_path / "sm_output.dat").exists()


def test_run_simulation_artifacts_error_does_not_crash():
    """Diretório inválido → artifacts_error setado, mas o H6 É retornado (sim não falha)."""
    out = _run_simulation(
        SimRequest(
            geology_mode="fixed",
            n_models=1,
            tj=10.0,
            output_dir="/proc/forbidden_xyz/cannot_write",
            save_fortran_artifacts=True,
        )
    )
    assert out["H6"] is not None  # resultado preservado
    assert out["artifacts_path"] is None and out["artifacts_error"]  # erro registrado


# ════════════════════════════════════════════════════════════════════════════
# SimulatorView — paralelismo + aviso + contadores (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_simulator_view_parallelism_and_counters(qtbot):
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_sim_vm()
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    # widgets de paralelismo presentes
    assert view._n_workers.minimum() == 1 and view._n_workers.maximum() == 256
    assert view._threads.minimum() == 1
    assert "CPU:" in view._cpu_info.text()
    # contadores derivados dos CSV
    view._freqs.setText("20000, 40000, 60000")
    view._dips.setText("0")
    view._trs.setText("1, 2")
    assert view._nf_label.text() == "3"
    assert view._ntheta_label.text() == "1"
    assert view._ntr_label.text() == "2"
    # CSV inválido → "—"
    view._freqs.setText("20000, abc")
    assert view._nf_label.text() == "—"


@pytest.mark.gui
def test_simulator_view_oversubscription_warning(qtbot):
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_sim_vm()
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    phys = view._physical_cores or 4
    view._n_workers.setValue(min(256, phys * 4))
    view._threads.setValue(4)
    assert "Oversubscrição" in view._parallel_warn.text()
    view._n_workers.setValue(1)
    view._threads.setValue(1)
    assert view._parallel_warn.text() == ""


# ════════════════════════════════════════════════════════════════════════════
# Overflow — recentes elididos mas emit do caminho COMPLETO
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_experiments_panel_recents_no_overflow(qtbot):
    from apps.sim_manager.perspectives.simulation.experiments_view import (
        ExperimentsPanel,
    )
    from geosteering_ai.gui.qt_compat import Qt

    panel = ExperimentsPanel()
    qtbot.addWidget(panel)
    long = "/home/user/" + "muito_longo/" * 8 + "Experimento.exp.json"
    panel.set_recents([long])
    # sem scrollbar horizontal + elide no meio
    assert (
        panel._recents.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert panel._recents.textElideMode() == Qt.TextElideMode.ElideMiddle
    item = panel._recents.item(0)
    assert item.text() == long and item.toolTip() == long  # texto/tooltip íntegros
    emitted: list = []
    panel.recent_activated.connect(emitted.append)
    panel.recent_activated.emit(item.text())
    assert emitted == [long]  # emit do caminho COMPLETO (não truncado)


# ════════════════════════════════════════════════════════════════════════════
# Status bar — Cache / Plot / Binding (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_status_bar_fields(qtbot):
    from apps.sim_manager.main_window import SM_MainWindow
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    win = SM_MainWindow(AppContext(app_name="SM"))
    qtbot.addWidget(win)
    win.add_perspective(SimulationPerspective())
    assert "Binding:" in win._sb_binding.text()
    assert win._sb_plot.text().startswith("Plot:") and "—" not in win._sb_plot.text()
    assert win._sb_cache.text().startswith("Cache:")
