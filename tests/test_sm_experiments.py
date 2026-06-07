# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_experiments.py                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0016-sm-experiments-history (Fatia 6c)                     ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM MVVM — experimentos & histórico                         ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-07                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre a Fatia 6c (PURO): roundtrip .exp.json (forward-compat),          ║
# ║    ExperimentsViewModel (new/open/add_snapshot/cache ●/○/select/recents)   ║
# ║    via stub VMSignal — sem Qt.                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da Fatia 6c — experimentos & histórico (PURO, stub VMSignal)."""

from __future__ import annotations

from typing import Any, Dict, List

from apps.sim_manager.perspectives.simulation.experiment_state import (
    ExperimentState,
    SimulationSnapshot,
)
from apps.sim_manager.perspectives.simulation.experiments_viewmodel import (
    ExperimentsViewModel,
)
from geosteering_ai.gui.viewmodels.signal import VMSignal


# ════════════════════════════════════════════════════════════════════════════
# ExperimentState / SimulationSnapshot — roundtrip + forward-compat
# ════════════════════════════════════════════════════════════════════════════
def test_experiment_roundtrip_and_forward_compat():
    exp = ExperimentState(
        name="Exp1",
        description="teste",
        output_dir="out",
        params={"n_models": 4},
        snapshots=[SimulationSnapshot("s1", "2026-01-01", "#1", "numba", 4, 1.5, {})],
    )
    d = exp.to_dict()
    d["chave_futura_desconhecida"] = 123  # forward-compat: deve ser ignorada
    back = ExperimentState.from_dict(d)
    assert back.name == "Exp1" and back.params == {"n_models": 4}
    assert len(back.snapshots) == 1 and back.snapshots[0].snapshot_id == "s1"
    assert back.snapshots[0].n_models == 4


def test_experiment_snapshot_ops():
    exp = ExperimentState(name="E")
    exp.append_snapshot(SimulationSnapshot("a", "t", "l"))
    exp.append_snapshot(SimulationSnapshot("b", "t", "l"))
    assert len(exp.snapshots) == 2
    exp.remove_snapshot("a")
    assert [s.snapshot_id for s in exp.snapshots] == ["b"]
    exp.clear_snapshots()
    assert exp.snapshots == []


# ════════════════════════════════════════════════════════════════════════════
# ExperimentsViewModel (PURO) — stub service
# ════════════════════════════════════════════════════════════════════════════
class _StubExpService:
    def __init__(self) -> None:
        self.cache_updated = VMSignal()
        self.error = VMSignal()
        self.saved = VMSignal()
        self._cache: Dict[str, Any] = {}
        self._recents: List[str] = []
        self.saved_calls = 0

    def create_experiment(self, name: str, desc: str, out: str) -> ExperimentState:
        return ExperimentState(
            name=name, description=desc, output_dir=out, file_path=f"{out}/x.exp.json"
        )

    def load_experiment(self, path: str) -> ExperimentState:
        if "bad" in path:
            raise OSError("arquivo inexistente")
        return ExperimentState(
            name="carregado",
            file_path=path,
            snapshots=[SimulationSnapshot("s1", "t", "#1")],
        )

    def save_experiment_async(self, exp: ExperimentState) -> None:
        self.saved_calls += 1

    def push_recent(self, path: str) -> List[str]:
        self._recents = [path] + [p for p in self._recents if p != path]
        return self._recents

    def load_recents(self) -> List[str]:
        return list(self._recents)

    def cache_put(self, sid: str, bundle: Any) -> List[str]:
        self._cache[sid] = bundle
        self.cache_updated.emit(sid, True)
        return []

    def cache_get(self, sid: str) -> Any:
        return self._cache.get(sid)

    def cache_contains(self, sid: str) -> bool:
        return sid in self._cache


def _make_vm() -> ExperimentsViewModel:
    return ExperimentsViewModel(_StubExpService())


def test_vm_new_experiment_and_add_snapshot():
    vm = _make_vm()
    added: List[Any] = []
    vm.snapshot_added.connect(lambda sid, lbl, ic: added.append((sid, ic)))
    assert vm.new_experiment("E", "", "out") is True
    assert vm.experiment is not None and vm.experiment.name == "E"
    vm.add_snapshot(SimulationSnapshot("s1", "t", "#1"), in_cache=True)
    assert len(vm.snapshots) == 1 and vm.is_in_cache("s1")
    assert added == [("s1", True)]


def test_vm_new_experiment_empty_name_rejected():
    vm = _make_vm()
    errors: List[str] = []
    vm.error.connect(errors.append)
    assert vm.new_experiment("  ", "", "out") is False
    assert errors and vm.experiment is None


def test_vm_open_experiment_ok_and_invalid():
    vm = _make_vm()
    assert vm.open_experiment("good.exp.json") is True
    assert vm.experiment is not None and len(vm.snapshots) == 1
    assert vm.recents == ["good.exp.json"]
    errors: List[str] = []
    vm.error.connect(errors.append)
    assert vm.open_experiment("bad.exp.json") is False
    assert errors


def test_vm_cache_status_changed():
    vm = _make_vm()
    events: List[Any] = []
    vm.cache_status_changed.connect(lambda sid, ic: events.append((sid, ic)))
    vm._on_cache_updated("s1", True)
    assert vm.is_in_cache("s1")
    vm.mark_out_of_cache("s1")
    assert not vm.is_in_cache("s1")
    assert ("s1", True) in events and ("s1", False) in events


def test_vm_select_and_clear():
    vm = _make_vm()
    vm.new_experiment("E", "", "out")
    snap = SimulationSnapshot("s9", "t", "#9")
    vm.add_snapshot(snap, in_cache=False)
    assert vm.select_snapshot("s9") is snap
    assert vm.select_snapshot("nope") is None
    vm.clear_snapshots()
    assert vm.snapshots == [] and not vm.is_in_cache("s9")


def test_vm_save_delegates_and_updates_params():
    vm = _make_vm()
    vm.new_experiment("E", "", "out")
    vm.save_experiment(params={"n_models": 7})
    assert vm.experiment is not None and vm.experiment.params == {"n_models": 7}
    assert vm._service.saved_calls == 1
