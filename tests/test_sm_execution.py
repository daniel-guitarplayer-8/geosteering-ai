# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_execution.py                                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0014-sm-execution-feedback (Fatia 6a)                      ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM MVVM — execução & feedback (progresso/cancel/pause)     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-06                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre a Fatia 6a: controle cooperativo (_await_resume_or_cancel),       ║
# ║    progresso por-grupo + cancelamento de _run_simulation (PURO, sem Qt),   ║
# ║    o ViewModel (progresso/status_display/cancel/log) e a injeção de         ║
# ║    progresso pelo Worker + roteamento do Service (gated em Qt).            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da Fatia 6a — execução & feedback (progresso, cancelamento, pause, status)."""

from __future__ import annotations

import os
import threading
from typing import Any, List, Tuple

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from geosteering_ai.gui.qt_compat import QT_AVAILABLE
from geosteering_ai.gui.services.sim_request import (
    SimRequest,
    _await_resume_or_cancel,
    _run_simulation,
)
from geosteering_ai.gui.viewmodels.signal import VMSignal


# ════════════════════════════════════════════════════════════════════════════
# Controle cooperativo (_await_resume_or_cancel) — PURO
# ════════════════════════════════════════════════════════════════════════════
def test_await_cancel_returns_true_when_set():
    """RF-1 — cancel setado → True imediatamente."""
    cancel = threading.Event()
    cancel.set()
    assert _await_resume_or_cancel(cancel, None) is True


def test_await_running_returns_false():
    """RF-1 — sem cancel, pause setado (rodando) → False (prossegue)."""
    pause = threading.Event()
    pause.set()  # rodando
    assert _await_resume_or_cancel(threading.Event(), pause) is False


def test_await_none_events_returns_false():
    """RF-1 — sem eventos (None) → False (caminho jax/subprocesso)."""
    assert _await_resume_or_cancel(None, None) is False


# ════════════════════════════════════════════════════════════════════════════
# _run_simulation — progresso + cancelamento (PURO; numba real)
# ════════════════════════════════════════════════════════════════════════════
def test_run_simulation_cancelled_before_physics():
    """AC-2 — cancel pré-setado → {'cancelled': True} SEM rodar simulate_batch."""
    cancel = threading.Event()
    cancel.set()
    req = SimRequest(geology_mode="fixed", n_models=2, frequencies_hz=(20000.0,))
    out = _run_simulation(req, cancel_event=cancel)
    assert out.get("cancelled") is True
    assert "H6" not in out  # resultado parcial descartado (não corrompido)


def test_run_simulation_progress_stochastic():
    """AC-1 — progresso por-grupo: inicial (0,n) + (n,n) no fim (1 grupo fixo)."""
    calls: List[Tuple[int, int]] = []
    req = SimRequest(
        geology_mode="stochastic",
        n_models=3,
        n_layers_fixed=4,  # 1 grupo → progresso determinístico
        frequencies_hz=(20000.0,),
        tj=10.0,
        rng_seed=42,
    )
    out = _run_simulation(req, progress_callback=lambda d, t: calls.append((d, t)))
    assert "H6" in out and out["H6"].shape[0] == 3
    assert calls[0] == (0, 3)  # estado inicial
    assert calls[-1] == (3, 3)  # 100% ao fim


# ════════════════════════════════════════════════════════════════════════════
# ViewModel — progresso / status_display / cancel / log (PURO via VMSignal stub)
# ════════════════════════════════════════════════════════════════════════════
class _StubService:
    """Service stub PURO (VMSignal) com progress + request_cancel/pause/resume."""

    def __init__(self) -> None:
        self.finished = VMSignal()
        self.error = VMSignal()
        self.progress = VMSignal()
        self.requests: list = []
        self.cancels = 0
        self.pauses = 0
        self.resumes = 0

    def run(self, request: Any) -> None:
        self.requests.append(request)

    def request_cancel(self) -> None:
        self.cancels += 1

    def request_pause(self) -> None:
        self.pauses += 1

    def request_resume(self) -> None:
        self.resumes += 1


def _make_vm():
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    return SimulationViewModel(service=_StubService())


def test_vm_progress_and_status_display():
    """AC-1/AC-4 — progress emitido pelo service atualiza done/total + status_display."""
    vm = _make_vm()
    vm.run()  # stub só registra (não conclui) → status "running"
    assert vm.status == "running"
    vm._service.progress.emit(2, 4)
    assert vm.progress_done == 2 and vm.progress_total == 4
    disp = vm.status_display
    assert "executando" in disp["state"]


def test_vm_cancelled_result_sets_status_and_skips_results():
    """AC-2 — resultado {'cancelled': True} → status 'cancelled'; galeria NÃO alimentada."""
    vm = _make_vm()
    logs: List[str] = []
    vm.log_entry.connect(logs.append)
    vm.run()
    vm._service.finished.emit({"cancelled": True, "backend": "numba"})
    assert vm.status == "cancelled"
    assert (vm.last_result or {}).get("cancelled") is True
    assert vm.results.has_result is False  # galeria intocada
    assert any("cancelad" in m.lower() for m in logs)


def test_vm_request_cancel_delegates_and_guards():
    """AC-4 — request_cancel delega ao service só enquanto 'running' (guard)."""
    vm = _make_vm()
    vm.run()
    vm.request_cancel()
    assert vm._service.cancels == 1
    vm._service.finished.emit({"cancelled": True, "backend": "numba"})  # → cancelled
    vm.request_cancel()  # status != running → no-op
    assert vm._service.cancels == 1


def test_vm_normal_finish_feeds_results_and_logs():
    """AC-6 — resultado normal: status 'done', galeria alimentada, log de conclusão."""
    import numpy as np

    vm = _make_vm()
    logs: List[str] = []
    vm.log_entry.connect(logs.append)
    vm.run()
    h6 = np.zeros((2, 1, 1, 10, 1, 9), dtype=np.complex128)
    vm._service.finished.emit(
        {"H6": h6, "positions_z": np.zeros(10), "backend": "numba"}
    )
    assert vm.status == "done"
    assert vm.results.has_result is True
    assert any("Conclu" in m for m in logs)


# ════════════════════════════════════════════════════════════════════════════
# Worker (injeção de progresso) + Service (roteamento) — gated em Qt
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not QT_AVAILABLE, reason="requer binding Qt6")
@pytest.mark.gui
def test_worker_injects_progress_callback(qapp):
    """RF-2 — Worker(report_progress=True) injeta progress_callback ao chamar fn."""
    from geosteering_ai.gui.threading.worker import Worker

    seen = {}

    def fn(progress_callback=None):
        seen["has_cb"] = progress_callback is not None
        return "ok"

    w = Worker(fn, report_progress=True)
    w.run()  # síncrono (sem QThread) — exercita só a injeção
    assert seen["has_cb"] is True

    w2 = Worker(lambda **k: seen.update(no_cb="progress_callback" not in k) or "ok")
    w2.run()
    assert seen["no_cb"] is True  # sem report_progress → NÃO injeta


@pytest.mark.skipif(not QT_AVAILABLE, reason="requer binding Qt6")
@pytest.mark.gui
def test_service_routes_numba_with_progress_and_events(qapp, monkeypatch):
    """RF-3 — numba: _run_async com report_progress + cancel_event/pause_event."""
    from geosteering_ai.gui.services.simulation_service import SimulationService

    svc = SimulationService()
    captured = {}
    monkeypatch.setattr(
        svc, "_run_async", lambda fn, *a, **k: captured.update(k, called=True)
    )
    svc.run(SimRequest(backend="numba"))
    assert captured.get("called") is True
    assert captured.get("report_progress") is True
    assert captured.get("cancel_event") is svc._cancel_event
    assert captured.get("pause_event") is svc._pause_event


@pytest.mark.skipif(not QT_AVAILABLE, reason="requer binding Qt6")
@pytest.mark.gui
def test_service_reentrancy_guard(qapp, monkeypatch):
    """Revisão #2/#4 — run() ignora 2ª chamada se já ocupado (não reseta eventos)."""
    from geosteering_ai.gui.services.simulation_service import SimulationService

    svc = SimulationService()
    calls: list = []
    monkeypatch.setattr(svc, "is_busy", lambda: True)  # simula worker em voo
    monkeypatch.setattr(svc, "_run_async", lambda *a, **k: calls.append("async"))
    monkeypatch.setattr(svc, "_run_in_subprocess", lambda *a, **k: calls.append("sub"))
    svc.run(SimRequest(backend="numba"))
    assert calls == []  # guard bloqueou (sem despacho, sem reset de eventos)


@pytest.mark.skipif(not QT_AVAILABLE, reason="requer binding Qt6")
@pytest.mark.gui
def test_service_wait_unblocks_paused_worker(qapp):
    """Revisão #1 — wait() seta cancel+resume ANTES de bloquear (evita deadlock)."""
    from geosteering_ai.gui.services.simulation_service import SimulationService

    svc = SimulationService()
    svc.request_pause()  # _pause_event limpo (pausado)
    svc.wait(timeout_ms=1)  # sem threads → retorna rápido; deve destravar/cancelar
    assert svc._cancel_event.is_set()
    assert svc._pause_event.is_set()


@pytest.mark.skipif(not QT_AVAILABLE, reason="requer binding Qt6")
@pytest.mark.gui
def test_service_request_cancel_sets_event(qapp):
    """RF-3 — request_cancel seta o cancel_event (e despausa)."""
    from geosteering_ai.gui.services.simulation_service import SimulationService

    svc = SimulationService()
    svc.request_pause()
    assert not svc._pause_event.is_set()
    svc.request_cancel()
    assert svc._cancel_event.is_set()
    assert svc._pause_event.is_set()  # despausado p/ o loop ver o cancel
