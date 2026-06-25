# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_jax_coldstart.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Fix JAX cold-start — SM MVVM: messaging honesto + knob de Geometrias (K)  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-17                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Investigação 2026-06-17: o "JAX 6× mais lento que o Numba" no SM é um   ║
# ║    cold-start de compilação XLA (uma vez por geometria/tamanho), NÃO uma   ║
# ║    regressão nem fallback p/ CPU — o cache de disco já faz as execuções     ║
# ║    seguintes baterem o Numba (~29s < 36.8s). Estes testes fixam as duas     ║
# ║    correções de baixo risco (sem tocar física/dispatch):                   ║
# ║      (1) MENSAGEM honesta: run() emite a nota de cold-start p/ jax/auto.    ║
# ║      (2) KNOB de Geometrias (K): expõe SimRequest.n_geometries na UI        ║
# ║          (0=auto), permitindo trocar diversidade por cold-start (K menor ⇒  ║
# ║          menos compilações; medido K=1 ~108s vs K=4 ~173s; warm ~29s).      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do fix de cold-start do JAX no SM (mensagem + knob de Geometrias K)."""

from __future__ import annotations

import os
from typing import Any, List

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from geosteering_ai.gui.viewmodels.signal import VMSignal  # noqa: E402


class _StubService:
    """Service stub PURO (VMSignal) — só registra a requisição (não conclui)."""

    def __init__(self) -> None:
        self.finished = VMSignal()
        self.error = VMSignal()
        self.progress = VMSignal()
        self.requests: list = []

    def run(self, request: Any) -> None:
        self.requests.append(request)


def _make_vm():
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    return SimulationViewModel(service=_StubService())


# ════════════════════════════════════════════════════════════════════════════
# (1) Mensagem honesta de cold-start (PURO — sem Qt)
# ════════════════════════════════════════════════════════════════════════════
# Backends VÁLIDOS no VM (_BACKENDS = numba|jax|auto); "jax_gpu" é aceito pelo
# dispatch/sim_request mas o VM não o oferece, então não é exercido aqui.
@pytest.mark.parametrize("backend", ["jax", "auto"])
def test_run_emits_jax_coldstart_note(backend):
    """backend jax/auto → run() emite a nota de cold-start XLA (1 vez, cache de disco)."""
    vm = _make_vm()
    vm.backend = backend
    logs: List[str] = []
    vm.log_entry.connect(logs.append)
    vm.run()
    joined = " ".join(logs).lower()
    assert "xla" in joined and "cache" in joined  # explica compile + cache de disco
    assert "geometrias (k)" in joined  # aponta o knob p/ encurtar o cold-start


def test_run_numba_has_no_jax_note():
    """backend numba → NÃO emite a nota de XLA (não há cold-start de GPU no Numba)."""
    vm = _make_vm()
    vm.backend = "numba"
    logs: List[str] = []
    vm.log_entry.connect(logs.append)
    vm.run()
    assert not any("xla" in m.lower() for m in logs)


def test_run_jax_per_model_has_no_note():
    """backend jax + diversidade per_model → geometria não-agrupável → cai p/ Numba
    (sem compile XLA), então a nota de cold-start NÃO se aplica (espelha
    _templates_active)."""
    vm = _make_vm()
    vm.backend = "jax"
    vm.geometry_diversity = "per_model"
    logs: List[str] = []
    vm.log_entry.connect(logs.append)
    vm.run()
    assert not any("xla" in m.lower() for m in logs)


# ════════════════════════════════════════════════════════════════════════════
# (2) Knob de Geometrias (K) — round-trip de sessão (PURO)
# ════════════════════════════════════════════════════════════════════════════
def test_n_geometries_session_roundtrip():
    """n_geometries persiste no dict de sessão e recarrega — tanto int QUANTO None."""
    # int
    vm = _make_vm()
    vm.n_geometries = 3
    vm2 = _make_vm()
    vm2.load_session_dict(vm.to_session_dict())
    assert vm2.n_geometries == 3
    # None (auto) — também deve sobreviver ao round-trip
    vm3 = _make_vm()
    vm3.n_geometries = None
    vm4 = _make_vm()
    vm4.load_session_dict(vm3.to_session_dict())
    assert vm4.n_geometries is None


def test_n_geometries_setter_clamps_both_bounds():
    """setter: None fica None; piso 1 (0→1); teto _N_GEOMETRIES_MAX (sincroniza c/ o
    spinbox — evita o desync VM/UI de um .session de versão futura com K grande)."""
    from apps.sim_manager.perspectives.simulation.viewmodel import _N_GEOMETRIES_MAX

    vm = _make_vm()
    vm.n_geometries = None
    assert vm.n_geometries is None
    vm.n_geometries = 0  # piso → 1 (0 não é um nº de geometrias válido)
    assert vm.n_geometries == 1
    vm.n_geometries = 10_000  # teto → _N_GEOMETRIES_MAX (== range máx do spinbox)
    assert vm.n_geometries == _N_GEOMETRIES_MAX


def test_n_geometries_session_clamps_out_of_range():
    """.session de versão futura com K > teto → clampado no load (sem desync VM/UI)."""
    from apps.sim_manager.perspectives.simulation.viewmodel import _N_GEOMETRIES_MAX

    vm = _make_vm()
    d = vm.to_session_dict()
    d["n_geometries"] = 999  # fora do range do spinbox
    vm.load_session_dict(d)
    assert vm.n_geometries == _N_GEOMETRIES_MAX  # clampado pelo setter (via setattr)


# ════════════════════════════════════════════════════════════════════════════
# (2) Knob de Geometrias (K) — wiring da View (GUI)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_view_exposes_geometry_count_spinbox(qtbot):
    """A View expõe o spinbox Geometrias (K) com 'auto' (0) como valor especial."""
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_vm()
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    assert hasattr(view, "_geom_k")
    assert view._geom_k.minimum() == 0  # 0 = auto
    assert view._geom_k.specialValueText() == "auto"


@pytest.mark.gui
def test_view_pushes_geometry_count_to_vm(qtbot):
    """_push_inputs_to_vm: K>0 → vm.n_geometries=K; K=0 → None (auto)."""
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_vm()
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    view._geom_k.setValue(2)
    view._push_inputs_to_vm()
    assert vm.n_geometries == 2  # K explícito propagado
    view._geom_k.setValue(0)
    view._push_inputs_to_vm()
    assert vm.n_geometries is None  # 0 → auto (teto ≤4)


@pytest.mark.gui
def test_view_syncs_geometry_count_from_vm(qtbot):
    """_sync_inputs_from_vm: None → 0 (auto); int → o próprio valor."""
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_vm()
    vm.n_geometries = 5
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    view._sync_inputs_from_vm()
    assert view._geom_k.value() == 5
    vm.n_geometries = None
    view._sync_inputs_from_vm()
    assert view._geom_k.value() == 0  # auto
