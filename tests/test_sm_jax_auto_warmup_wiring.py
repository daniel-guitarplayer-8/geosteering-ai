# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_jax_auto_warmup_wiring.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Wiring do warmup JAX config-aware ON-SELECT (Fase 2) — gatilho na View     ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garante o cabeamento (sem jax real): ao selecionar backend jax/auto, a   ║
# ║    SimulatorView dispara ``warmup_config(specs)`` no service compartilhado  ║
# ║    com os specs SHAPE-MATCHING da config corrente — respeitando a pref      ║
# ║    ``jax_auto_warmup``, o guard ``is_busy``, e no-op p/ numba/sem-service.  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de wiring do warmup config-aware on-select (Fase 2) — mock service, jax-free."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _MockWarmup:
    """Mock do JaxWarmupService — registra chamadas de warmup_config sem tocar jax."""

    def __init__(self) -> None:
        self.config_calls: list = []
        self._busy = False

    def is_busy(self) -> bool:
        return self._busy

    def warmup_config(self, specs) -> None:  # noqa: ANN001
        self.config_calls.append(specs)

    def warmup(self) -> None:
        pass


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


def _pref(monkeypatch, on: bool) -> None:
    """Força a pref jax_auto_warmup (carrega defaults determinísticos)."""
    from apps.sim_manager.perspectives.preferences.service import PreferencesService

    monkeypatch.setattr(
        PreferencesService, "load", lambda self: {"jax_auto_warmup": on}, raising=True
    )


def _view_fixed_jax(qtbot, mock):
    """Cria a SimulatorView com geologia FIXA (n_layers=15, 1000 modelos) + mock service."""
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_sim_vm()
    view = SimulatorView(vm, jax_warmup_service=mock)
    qtbot.addWidget(view)
    view._geo_nlf_check.setChecked(True)
    view._geo_nlf.setValue(15)
    view._n_models.setValue(1000)
    return view


# ════════════════════════════════════════════════════════════════════════════
# build_sim_request — espelha o estado do VM (fonte única reusada pelo warmup)
# ════════════════════════════════════════════════════════════════════════════
def test_vm_build_sim_request_reflects_state():
    """``build_sim_request`` reflete o estado corrente do VM (a MESMA req que run usa)."""
    vm = _make_sim_vm()
    vm.n_models = 500
    vm.backend = "jax"
    vm.n_layers_fixed = 12
    req = vm.build_sim_request()
    assert req.n_models == 500
    assert req.backend == "jax"
    assert req.n_layers_fixed == 12


# ════════════════════════════════════════════════════════════════════════════
# Gatilho on-select — dispara warmup_config com specs shape-matching
# ════════════════════════════════════════════════════════════════════════════
def test_select_jax_triggers_warmup_config(qtbot, monkeypatch):
    """Selecionar 'jax' (pref ON) → warmup_config com specs da config (n_layers=15)."""
    _pref(monkeypatch, on=True)
    mock = _MockWarmup()
    view = _view_fixed_jax(qtbot, mock)
    view._sim_backend.setCurrentText("jax")  # dispara _on_sim_backend_changed
    assert len(mock.config_calls) == 1
    specs = mock.config_calls[0]
    assert specs and all(s["n_layers"] == 15 for s in specs)
    assert all(s["complex_dtype"] == "complex128" for s in specs)


def test_select_auto_also_triggers(qtbot, monkeypatch):
    """'auto' também dispara o warmup (o dispatcher pode escolher JAX)."""
    _pref(monkeypatch, on=True)
    mock = _MockWarmup()
    view = _view_fixed_jax(qtbot, mock)
    view._sim_backend.setCurrentText("auto")
    assert len(mock.config_calls) == 1


def test_pref_off_no_warmup(qtbot, monkeypatch):
    """pref jax_auto_warmup=False → nenhum warmup ao selecionar JAX."""
    _pref(monkeypatch, on=False)
    mock = _MockWarmup()
    view = _view_fixed_jax(qtbot, mock)
    view._sim_backend.setCurrentText("jax")
    assert mock.config_calls == []


def test_numba_backend_no_warmup(qtbot, monkeypatch):
    """Selecionar 'numba' → não dispara warmup (handler retorna cedo)."""
    _pref(monkeypatch, on=True)
    mock = _MockWarmup()
    view = _view_fixed_jax(qtbot, mock)
    view._sim_backend.setCurrentText("numba")
    assert mock.config_calls == []


def test_is_busy_guard_no_double_warmup(qtbot, monkeypatch):
    """Guard is_busy (warmup já em voo) → não empilha (debounce natural)."""
    _pref(monkeypatch, on=True)
    mock = _MockWarmup()
    mock._busy = True
    view = _view_fixed_jax(qtbot, mock)
    view._sim_backend.setCurrentText("jax")
    assert mock.config_calls == []


def test_no_service_no_crash(qtbot, monkeypatch):
    """Sem service (None — CI/sem-jax/Studio) → no-op gracioso, sem crash."""
    _pref(monkeypatch, on=True)
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_sim_vm()
    view = SimulatorView(vm, jax_warmup_service=None)
    qtbot.addWidget(view)
    view._geo_nlf_check.setChecked(True)
    view._sim_backend.setCurrentText("jax")  # não deve levantar
