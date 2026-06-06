# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sim_app_skeleton.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0011a-sm-app-skeleton                                      ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app MVVM — walking skeleton                             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre os ACs da spec 0011a: ViewModel PURO (com service stub, sem Qt),  ║
# ║    fidelidade (SimulationService → simulate_batch real → shape H6),        ║
# ║    threading (Worker off-thread), end-to-end real (perspectiva → run →     ║
# ║    result_ready) e a fronteira de import (VM sem Qt; core sem gui/apps).   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0011a — walking skeleton do SM MVVM (VM puro · fidelidade · e2e)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Qt headless — mesmo padrão das outras suites GUI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ════════════════════════════════════════════════════════════════════════════
# RF-4 — SimulationViewModel PURO (sem Qt; service stub)
# ════════════════════════════════════════════════════════════════════════════
def _make_stub_service():
    """Service stub PURO (VMSignals + run que registra a requisição)."""
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    class StubService:
        def __init__(self) -> None:
            self.finished = VMSignal()
            self.error = VMSignal()
            self.requests: list = []

        def run(self, request) -> None:  # noqa: ANN001
            self.requests.append(request)

    return StubService()


def test_vm_validate_run_and_result_with_stub():
    """AC-1 — VM puro: validar → run chama service.run; finished → result_ready + done."""
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    svc = _make_stub_service()
    vm = SimulationViewModel(service=svc)
    rec: list = []
    vm.result_ready.connect(rec.append)

    vm.frequency_hz = 20000.0
    vm.dip_deg = 0.0
    vm.n_models = 3
    vm.run()
    assert len(svc.requests) == 1  # validou e delegou
    assert svc.requests[0].n_models == 3
    assert vm.status == "running"

    # o service "termina" → VM atualiza estado e re-emite
    svc.finished.emit({"H6": "fake", "positions_z": "z"})
    assert vm.status == "done"
    assert rec == [{"H6": "fake", "positions_z": "z"}]
    assert vm.last_result is not None


def test_vm_rejects_invalid_params_without_calling_service():
    """AC-1 — validação reprova params fora da errata; NÃO chama o service."""
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    svc = _make_stub_service()
    vm = SimulationViewModel(service=svc)
    vm.frequency_hz = 50.0  # < 100 Hz → inválido
    vm.run()
    assert svc.requests == []  # NÃO delegou
    assert vm.status == "error"
    assert vm.last_result is not None and vm.last_result["errors"]


def test_vm_error_signal_sets_error_status():
    """AC-1 — erro do service → status 'error' + last_result['error']."""
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    svc = _make_stub_service()
    vm = SimulationViewModel(service=svc)
    vm.run()
    svc.error.emit("falha na física")
    assert vm.status == "error"
    assert vm.last_result == {"error": "falha na física"}


def test_vm_rejects_dip_above_90():
    """AC-1 (fidelidade) — dip > 90° é REJEITADO no VM (o core valida [0, 90]°)."""
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    svc = _make_stub_service()
    vm = SimulationViewModel(service=svc)
    vm.dip_deg = 100.0  # o simulate_batch recusaria (paridade Fortran [0,90])
    vm.run()
    assert svc.requests == []  # NÃO delegou
    assert vm.status == "error"
    assert vm.last_result is not None
    assert any("Dip" in e for e in vm.last_result["errors"])
    # 90° é o limite aceito
    vm2 = SimulationViewModel(service=_make_stub_service())
    vm2.dip_deg = 90.0
    assert vm2.validate() == []


# ════════════════════════════════════════════════════════════════════════════
# RF-3 — fidelidade: SimulationService → simulate_batch real → shape H6
# ════════════════════════════════════════════════════════════════════════════
def test_simulation_run_produces_correct_h6_shape():
    """AC-2 — _run_simulation chama simulate_batch real; H6 com shape/dtype corretos.

    Lento (inclui JIT warmup do Numba). Física INTOCADA — só verifica shape/dtype/finitude.
    """
    import numpy as np

    from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

    req = SimRequest(frequencies_hz=(20000.0,), n_models=2, backend="numba")
    result = _run_simulation(req)
    h6 = result["H6"]
    assert h6.shape == (2, 1, 1, 50, 1, 9)  # (n_models, nTR, nAng, n_pos, nf, 9)
    assert np.iscomplexobj(h6)
    assert np.all(np.isfinite(h6.view(np.float64)))
    assert result["backend"] == "numba"


def test_build_batch_outputs_valid_shapes():
    """AC-2 — _build_batch produz shapes/ranges físicos válidos (puro, rápido)."""
    import numpy as np

    from geosteering_ai.gui.services.sim_request import SimRequest, _build_batch

    rho_h, rho_v, esp, positions_z = _build_batch(SimRequest(n_models=3))
    assert rho_h.shape == (3, 3) and rho_v.shape == (3, 3)  # (n_models, n_layers)
    assert esp.shape == (3, 1)  # n_layers - 2 internas
    assert positions_z.shape == (50,)
    assert rho_h.dtype == np.float64
    # ranges físicos: ρ ∈ [0.01, 1e6]; λ = √(ρᵥ/ρₕ) ∈ [1, 5]
    assert np.all((rho_h >= 0.01) & (rho_h <= 1.0e6))
    lam = np.sqrt(rho_v / rho_h)
    assert np.all((lam >= 1.0) & (lam <= 5.0))


# ════════════════════════════════════════════════════════════════════════════
# RF-1 — threading: Worker roda callable off-thread (qtbot)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_worker_runs_callable_off_thread(qtbot):
    """AC-3 — Worker executa o callable noutra thread e emite finished(result)."""
    from geosteering_ai.gui.threading import Worker, run_in_thread

    worker = Worker(lambda a, b: a * b, 6, 7)
    with qtbot.waitSignal(worker.signals.finished, timeout=5000) as blocker:
        thread = run_in_thread(worker)
    assert blocker.args == [42]
    qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)


@pytest.mark.gui
def test_worker_emits_error_on_exception(qtbot):
    """AC-3 — exceção no callable vira error(str) (nada escapa da thread)."""
    from geosteering_ai.gui.threading import Worker, run_in_thread

    def boom():
        raise ValueError("falha proposital")

    worker = Worker(boom)
    with qtbot.waitSignal(worker.signals.error, timeout=5000) as blocker:
        thread = run_in_thread(worker)
    assert "falha proposital" in blocker.args[0]
    # JOIN obrigatório: uma QThread destruída enquanto roda aborta o processo.
    qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)


@pytest.mark.gui
def test_service_is_busy_during_run_then_idle(qtbot):
    """AC-3 — BaseService.is_busy() é True durante o trabalho e False após o join."""
    import threading

    from geosteering_ai.gui.services.base import BaseService

    gate = threading.Event()
    service = BaseService()
    service._run_async(gate.wait)  # worker bloqueia até liberarmos o gate
    qtbot.waitUntil(service.is_busy, timeout=3000)  # True durante
    assert service.is_busy()
    gate.set()  # libera o worker → finished → thread.quit
    qtbot.waitUntil(lambda: not service.is_busy(), timeout=3000)  # False após


# ════════════════════════════════════════════════════════════════════════════
# RF-5/RF-6 — end-to-end real (perspectiva → run → result_ready) (qtbot)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_perspective_builds_view_under_main_window(qtbot):
    """AC-4 — a SimulationPerspective constrói View+VM sob SM_MainWindow (lazy)."""
    from apps.sim_manager.main_window import SM_MainWindow
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    win = SM_MainWindow(AppContext(app_name="teste 0011a"))
    qtbot.addWidget(win)
    win.add_perspective(SimulationPerspective())
    assert win.perspective_count == 1
    assert win.windowTitle() == "teste 0011a"


@pytest.mark.gui
def test_end_to_end_real_simulation(qtbot):
    """AC-4 — VM real + Service real: run() → (async) → result_ready com H6 correto.

    O caminho COMPLETO da pilha MVVM: VM puro → SimulationService → Worker (thread)
    → simulate_batch → finished (VMSignal na main thread) → result_ready. Lento
    (Numba), mas é a PROVA do walking skeleton.
    """
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
    from geosteering_ai.gui.services import SimulationService

    # Service explícito (mesmo que a perspectiva injeta) → ref p/ JOIN no teardown.
    service = SimulationService()
    vm = SimulationViewModel(service=service)
    rec: list = []
    vm.result_ready.connect(rec.append)
    vm.frequency_hz = 20000.0
    vm.n_models = 2
    vm.run()
    # spin o event loop até concluir (marshaling worker→main via QueuedConnection).
    # 60s cobre o JIT warmup do Numba + simulate_batch no batch pequeno (skeleton).
    qtbot.waitUntil(lambda: vm.status in ("done", "error"), timeout=60000)
    assert vm.status == "done", f"sim não concluiu: {vm.last_result}"
    assert rec and rec[0]["H6"].shape == (2, 1, 1, 50, 1, 9)
    # JOIN da worker thread antes do teardown (evita 'QThread destroyed while running').
    qtbot.waitUntil(lambda: not service.is_busy(), timeout=10000)


# ════════════════════════════════════════════════════════════════════════════
# RNF-2/RNF-4 — fronteira de import
# ════════════════════════════════════════════════════════════════════════════
def test_viewmodel_importable_without_qt():
    """AC-6 — importar o ViewModel NÃO importa PyQt6/PySide6 (Princípio X)."""
    code = (
        "import sys\n"
        "import apps.sim_manager.perspectives.simulation.viewmodel  # noqa\n"
        "import geosteering_ai.gui.services.sim_request  # noqa (puro)\n"
        "bad = [m for m in ('PyQt6', 'PySide6') if m in sys.modules]\n"
        "assert not bad, f'VM/sim_request puxou Qt: {bad}'\n"
        "print('PURE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "PURE_OK" in proc.stdout


def test_core_does_not_import_gui_or_apps():
    """AC-6 — o core (simulation/…) não importa gui/apps (fronteira)."""
    code = (
        "import sys\n"
        "import geosteering_ai.simulation.dispatch  # noqa\n"
        "bad = [m for m in sys.modules if m.startswith(('geosteering_ai.gui', 'apps'))]\n"
        "assert not bad, f'core puxou gui/apps: {bad}'\n"
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
