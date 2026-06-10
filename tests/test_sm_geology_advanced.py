# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_geology_advanced.py                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0015-sm-geology-advanced (Fatia 6b)                        ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM MVVM — geologia avançada (manual/canônico/Hankel)       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-06                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre a Fatia 6b: ManualLayersModel (validação/from_canonical),         ║
# ║    geologia manual em _run_simulation, filtro de Hankel selecionável, o     ║
# ║    ViewModel (apply_canonical_profile + sessão), e a PARIDADE <1e-12        ║
# ║    (manual numba vs jax, gated GPU). Tudo PURO exceto o gated.              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da Fatia 6b — geologia avançada (manual, perfis canônicos, filtro Hankel)."""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.gui.services.manual_geology import ManualLayersModel
from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation
from geosteering_ai.gui.viewmodels.signal import VMSignal


def _gpu_available() -> bool:
    try:
        from geosteering_ai.simulation.dispatch import _jax_gpu_available

        return bool(_jax_gpu_available())
    except Exception:
        return False


_needs_gpu = pytest.mark.skipif(not _gpu_available(), reason="requer GPU JAX")

_MANUAL = dict(
    geology_mode="manual",
    manual_n_layers=3,
    manual_thicknesses=(8.0,),
    manual_rho_h=(1.0, 10.0, 100.0),
    manual_rho_v=(2.0, 20.0, 200.0),  # ρᵥ ≥ ρₕ (TIV)
)


# ════════════════════════════════════════════════════════════════════════════
# ManualLayersModel — validação + from_canonical (PURO)
# ════════════════════════════════════════════════════════════════════════════
def test_manual_layers_valid():
    m = ManualLayersModel(3, (8.0,), (1.0, 10.0, 100.0), (2.0, 20.0, 200.0))
    assert m.validate() == []
    d = m.to_model_dict()
    assert d["n_layers"] == 3 and d["thicknesses"] == [8.0]


def test_manual_layers_invalid_sizes_and_tiv():
    # ρᵥ < ρₕ (viola TIV λ≥1) + thicknesses com tamanho errado.
    bad = ManualLayersModel(3, (8.0, 1.0), (1.0, 10.0, 100.0), (0.5, 20.0, 200.0))
    errors = bad.validate()
    assert any("espessura" in e for e in errors)
    assert any("ρᵥ" in e for e in errors)


def test_manual_from_canonical():
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    cm = get_canonical_model("oklahoma_3")
    m = ManualLayersModel.from_canonical(cm)
    assert m.n_layers == cm.n_layers == 3
    assert len(m.rho_h) == 3 and len(m.thicknesses) == 1
    assert m.validate() == []


# ════════════════════════════════════════════════════════════════════════════
# _run_simulation — geologia manual + filtro de Hankel (PURO; numba real)
# ════════════════════════════════════════════════════════════════════════════
def test_run_simulation_manual_geology():
    req = SimRequest(n_models=2, frequencies_hz=(20000.0,), tj=10.0, **_MANUAL)
    out = _run_simulation(req)
    assert out["H6"].shape[0] == 2
    assert np.all(np.isfinite(out["H6"].view(np.float64)))


def test_run_simulation_hankel_filter_flows():
    """O filtro de Hankel selecionado flui a simulate_batch (resultados diferem)."""
    base = dict(geology_mode="fixed", n_models=2, frequencies_hz=(20000.0,), tj=10.0)
    h_wer = _run_simulation(SimRequest(hankel_filter="werthmuller_201pt", **base))["H6"]
    h_kong = _run_simulation(SimRequest(hankel_filter="kong_61pt", **base))["H6"]
    assert h_wer.shape == h_kong.shape
    assert np.all(np.isfinite(h_kong.view(np.float64)))
    # Filtros distintos ⇒ resultados distintos (prova que o parâmetro flui).
    assert not np.array_equal(h_wer, h_kong)


# ════════════════════════════════════════════════════════════════════════════
# ViewModel — apply_canonical_profile + sessão (PURO via VMSignal stub)
# ════════════════════════════════════════════════════════════════════════════
class _StubService:
    def __init__(self) -> None:
        self.finished = VMSignal()
        self.error = VMSignal()
        self.progress = VMSignal()

    def run(self, request):  # noqa: ANN001
        pass


def _make_vm():
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    return SimulationViewModel(service=_StubService())


def test_vm_apply_canonical_profile():
    """Lote 2: aplica perfil canônico → manual + n_layers_fixed + tj/h1 CANÔNICOS.

    A geometria (tj/h1) reusa as funções do monólito (``sm_canonical_profiles``) —
    janela global +20 m simétrica — em vez da antiga fórmula ad-hoc 0.1·Σesp. Usa
    oklahoma_28 (28 camadas) para provar a correção do contador (Task 4: deixava de
    refletir o perfil e ficava preso em 5).
    """
    from geosteering_ai.simulation.tests.sm_canonical_profiles import (
        compute_canonical_h1,
        compute_canonical_reference_tj,
    )
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    vm = _make_vm()
    vm.apply_canonical_profile("oklahoma_28")  # auto_tj/auto_h1 default True
    cm = get_canonical_model("oklahoma_28")
    sesp = float(sum(float(x) for x in cm.esp))
    tj_ref = compute_canonical_reference_tj(current_esp_sum=sesp)
    assert vm.geology_mode == "manual"
    assert vm.manual_layers is not None and vm.manual_layers.n_layers == 28
    assert vm.n_layers_fixed == 28  # Task 4: contador reflete o perfil (não trava em 5)
    assert vm.tj == tj_ref  # paridade c/ o monólito (janela global +20 m)
    assert vm.h1 == compute_canonical_h1(
        tj_ref, sesp
    )  # centralização simétrica (tj−Σesp)/2
    assert vm.validate() == []


def test_vm_manual_mode_without_layers_invalid():
    vm = _make_vm()
    vm.geology_mode = "manual"
    vm.manual_layers = None
    assert any("manual" in e.lower() for e in vm.validate())


def test_vm_session_roundtrip_hankel_and_manual():
    import json

    vm = _make_vm()
    vm.hankel_filter = "kong_61pt"
    vm.apply_canonical_profile("oklahoma_3", auto_tj=False, auto_h1=False)
    blob = json.dumps(vm.to_session_dict())
    vm2 = _make_vm()
    vm2.load_session_dict(json.loads(blob))
    assert vm2.hankel_filter == "kong_61pt"
    assert vm2.manual_layers is not None and vm2.manual_layers.n_layers == 3
    assert vm2.geology_mode == "manual"


# ════════════════════════════════════════════════════════════════════════════
# Revisão adversarial — guards (Hankel inválido + coerção de sessão corrompida)
# ════════════════════════════════════════════════════════════════════════════
def test_hankel_filters_constant_matches_catalog():
    """O _HANKEL_FILTERS do VM casa com o catálogo real (evita staleness)."""
    from apps.sim_manager.perspectives.simulation.viewmodel import _HANKEL_FILTERS
    from geosteering_ai.simulation.filters.loader import FilterLoader

    assert set(_HANKEL_FILTERS) == set(FilterLoader().available())


def test_vm_invalid_hankel_rejected_and_sanitized():
    """Filtro inválido: validate() reprova; load de .session corrompido saneia."""
    vm = _make_vm()
    vm.hankel_filter = "bogus"
    assert any("Hankel" in e for e in vm.validate())
    vm2 = _make_vm()
    vm2.load_session_dict({"hankel_filter": "bogus"})
    assert vm2.hankel_filter == "werthmuller_201pt"  # saneado p/ default


def test_vm_load_manual_layers_string_coerced():
    """`.session` com strings nas camadas manuais é coerido p/ float (não corrompe)."""
    vm = _make_vm()
    vm.load_session_dict(
        {
            "geology_mode": "manual",
            "manual_layers": {
                "n_layers": 3,
                "thicknesses": ["8.0"],
                "rho_h": ["1.0", "10.0", "100.0"],
                "rho_v": ["2.0", "20.0", "200.0"],
            },
        }
    )
    assert vm.manual_layers is not None
    assert isinstance(vm.manual_layers.thicknesses[0], float)
    assert vm.manual_layers.validate() == []  # floats coeridos → válido


# ════════════════════════════════════════════════════════════════════════════
# Paridade <1e-12 — geologia manual numba vs jax (gated GPU)
# ════════════════════════════════════════════════════════════════════════════
@_needs_gpu
def test_parity_manual_numba_vs_jax():
    base = dict(n_models=2, frequencies_hz=(20000.0,), tj=10.0, **_MANUAL)
    h_n = _run_simulation(SimRequest(backend="numba", **base))["H6"]
    h_j = _run_simulation(SimRequest(backend="jax", **base))["H6"]
    assert h_n.shape == h_j.shape
    assert np.max(np.abs(h_n - h_j)) < 1e-12
