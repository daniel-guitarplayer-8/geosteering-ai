# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_pr1_fixes.py                                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  PR-1 — SM MVVM: diversidade de geometria (#4) + rótulo paginação (#5) +   ║
# ║         fidelidade de termos (#6) + fontes (#7c)                           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-16                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Gate das correções do PR-1: (#4) o colapso de geometria a K templates   ║
# ║    torna o batch agrupável (JAX GPU) sem perder modelos nem violar         ║
# ║    Σesp=total_depth, e só ativa p/ jax/auto; round-trip do estado no VM;   ║
# ║    (#5) o rótulo de paginação mostra o intervalo + total; (#7c) fontes.    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do PR-1 — geometria/paginação/termos/fontes do SM MVVM."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from geosteering_ai.gui.services.sim_request import (  # noqa: E402
    SimRequest,
    _collapse_geometry_to_templates,
    _generate_stochastic_models,
    _templates_active,
)


# ════════════════════════════════════════════════════════════════════════════
# #4 — diversidade de geometria (batchabilidade no JAX GPU)
# ════════════════════════════════════════════════════════════════════════════
def test_templates_active_by_backend_and_mode():
    assert _templates_active(SimRequest(backend="jax", geometry_diversity="auto"))
    assert _templates_active(SimRequest(backend="auto", geometry_diversity="auto"))
    assert not _templates_active(SimRequest(backend="numba", geometry_diversity="auto"))
    assert _templates_active(
        SimRequest(backend="numba", geometry_diversity="templates")
    )
    assert not _templates_active(
        SimRequest(backend="jax", geometry_diversity="per_model")
    )


def _distinct_esp_per_nlayers(models):
    from collections import defaultdict

    by_nl = defaultdict(set)
    for m in models:
        by_nl[m["n_layers"]].add(tuple(m["thicknesses"]))
    return {k: len(v) for k, v in by_nl.items()}


def test_collapse_geometry_makes_groupable_preserving_count():
    """N geometrias únicas → ≤K por n_layers (agrupável), sem perder modelos."""
    req = SimRequest(
        geology_mode="stochastic",
        n_models=600,
        backend="jax",
        rng_seed=7,
        n_layers_fixed=5,
    )
    models = _generate_stochastic_models(req)
    assert len(models) == 600
    assert _distinct_esp_per_nlayers(models)[5] == 600  # únicas antes
    collapsed = _collapse_geometry_to_templates([dict(m) for m in models], None)
    assert len(collapsed) == 600  # contagem preservada
    distinct = _distinct_esp_per_nlayers(collapsed)[5]
    assert distinct <= 4  # cap default
    assert distinct <= 600 // 2  # garante agrupável (n_grupos ≤ 0.5·g)


def test_collapse_preserves_sum_thickness():
    """Cada espessura templada é uma cópia de uma REAL → Σesp == total_depth preservado."""
    req = SimRequest(
        geology_mode="stochastic",
        n_models=200,
        backend="jax",
        rng_seed=11,
        n_layers_fixed=6,
        tj=30.0,
    )
    models = _generate_stochastic_models(req)
    sums_before = {round(float(np.sum(m["thicknesses"])), 6) for m in models}
    collapsed = _collapse_geometry_to_templates([dict(m) for m in models], None)
    # todas as somas pós-colapso devem ser um subconjunto das somas reais (cópias)
    sums_after = {round(float(np.sum(m["thicknesses"])), 6) for m in collapsed}
    assert sums_after.issubset(sums_before)
    # e cada modelo soma ~total_depth (a geologia estocástica garante Σesp=total_depth)
    for m in collapsed:
        assert abs(float(np.sum(m["thicknesses"])) - 30.0) < 1e-6


def test_collapse_respects_explicit_k():
    req = SimRequest(
        geology_mode="stochastic", n_models=400, rng_seed=3, n_layers_fixed=5
    )
    models = _generate_stochastic_models(req)
    collapsed = _collapse_geometry_to_templates(
        [dict(m) for m in models], n_geometries=2
    )
    assert _distinct_esp_per_nlayers(collapsed)[5] == 2


def _groups_by_nlayers(models):
    from collections import defaultdict

    g = defaultdict(list)
    for i, m in enumerate(models):
        g[int(m["n_layers"])].append(i)
    return g


def test_collapse_resolves_dispatch_groupable_per_group():
    """Cada grupo de n_layers (a granularidade REAL do dispatch) fica AGRUPÁVEL.

    ``_simulate_grouped`` chama ``simulate_batch`` POR grupo de n_layers, então o
    critério de agrupabilidade (``n_grupos ≤ 0.5·g``) vale POR grupo — não no batch
    inteiro. Valida o invariante na granularidade real (fixo E ragged).
    """
    from geosteering_ai.simulation._jax.multi_forward import group_by_geometry

    for fixed in (5, None):  # fixo E ragged (default)
        req = SimRequest(
            geology_mode="stochastic", n_models=600, rng_seed=5, n_layers_fixed=fixed
        )
        models = _collapse_geometry_to_templates(_generate_stochastic_models(req), None)
        for _nl, idxs in _groups_by_nlayers(models).items():
            g = len(idxs)
            esp = np.array([models[i]["thicknesses"] for i in idxs], dtype=np.float64)
            n_groups = len(group_by_geometry(esp))
            assert n_groups <= max(1, 0.5 * g)  # agrupável por grupo (dispatch.py:127)


def test_collapse_ragged_preserves_count_and_groups():
    """Caminho RAGGED (default n_layers_fixed=None): vários grupos, contagem preservada.

    O modo real do usuário é ragged. Cada grupo de n_layers colapsa a ≤K esp distintas;
    grupos com ≥32 modelos batelam na GPU, grupos menores caem p/ Numba (ocupação por
    grupo — documentado). Aqui garantimos a corretude do colapso ragged (não a perf).
    """
    req = SimRequest(geology_mode="stochastic", n_models=1000, rng_seed=9)  # ragged
    models = _collapse_geometry_to_templates(_generate_stochastic_models(req), None)
    assert len(models) == 1000  # contagem preservada
    groups = _groups_by_nlayers(models)
    assert len(groups) >= 2  # ragged ⇒ múltiplos grupos de n_layers
    distinct = _distinct_esp_per_nlayers(models)
    assert all(v <= 4 for v in distinct.values())  # ≤K por grupo
    # N=1000 / faixa 3-11 ⇒ a maioria dos grupos tem ≥32 modelos (GPU-elegível).
    big = [len(idxs) for idxs in groups.values() if len(idxs) >= 32]
    assert sum(len(groups[k]) for k in groups) == 1000
    assert big, "esperado ≥1 grupo grande (≥32) com N=1000 ragged (GPU-elegível)"


def test_collapse_odd_group_stays_groupable():
    """g ímpar (fronteira do floor g//2) permanece agrupável — guarda contra ceil."""
    # n_models ímpar com n_layers fixo → 1 grupo de tamanho ímpar.
    req = SimRequest(
        geology_mode="stochastic", n_models=101, rng_seed=1, n_layers_fixed=5
    )
    models = _collapse_geometry_to_templates(_generate_stochastic_models(req), None)
    distinct = _distinct_esp_per_nlayers(models)[5]
    assert distinct <= 101 // 2  # 50 → agrupável (n_grupos ≤ 0.5·101=50.5)


def test_vm_geometry_diversity_session_roundtrip():
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    class _Stub:
        def __init__(self):
            self.finished = VMSignal()
            self.error = VMSignal()
            self.progress = VMSignal()

        def run(self, request):  # noqa: ANN001
            pass

    vm = SimulationViewModel(service=_Stub())
    assert vm.geometry_diversity == "auto"  # default
    assert vm.n_geometries is None  # default (auto, cap 4)
    vm.geometry_diversity = "templates"
    vm.n_geometries = 3
    d = vm.to_session_dict()
    assert d["geometry_diversity"] == "templates" and d["n_geometries"] == 3
    vm2 = SimulationViewModel(service=_Stub())
    vm2.load_session_dict(d)
    assert vm2.geometry_diversity == "templates" and vm2.n_geometries == 3
    # caso None (auto, cap 4): muta vm2 p/ não-None antes do load → pega regressão
    vm2.n_geometries = 9
    vm2.load_session_dict({"geometry_diversity": "auto", "n_geometries": None})
    assert vm2.geometry_diversity == "auto" and vm2.n_geometries is None


def test_vm_sanitizes_invalid_geometry_diversity():
    """.session com geometry_diversity inválido → normalizado p/ 'auto' (whitelist)."""
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    class _Stub:
        def __init__(self):
            self.finished = VMSignal()
            self.error = VMSignal()
            self.progress = VMSignal()

        def run(self, request):  # noqa: ANN001
            pass

    vm = SimulationViewModel(service=_Stub())
    vm.load_session_dict({"geometry_diversity": "garbage"})
    assert vm.geometry_diversity == "auto"  # saneado (espelha o guard de backend)


def test_numba_auto_default_leaves_ensemble_uncollapsed():
    """#4 — numba+auto (DEFAULT) NÃO colapsa: o ensemble fica per-model (sem regressão).

    Prova end-to-end (via _run_simulation) que o caminho default preserva a diversidade
    total de geometria — só jax/auto colapsam. N=120 fixo p/ 1 grupo grande de esp únicas.
    """
    from geosteering_ai.gui.services.sim_request import _run_simulation

    out = _run_simulation(
        SimRequest(
            geology_mode="stochastic",
            n_models=120,
            backend="numba",
            geometry_diversity="auto",
            rng_seed=4,
            n_layers_fixed=5,
        )
    )
    geo = out["geology"]
    assert len(geo) == 120
    distinct = len({tuple(np.asarray(g["thicknesses"]).round(9)) for g in geo})
    assert distinct == 120  # per-model preservado (NÃO colapsado no default numba)


# ════════════════════════════════════════════════════════════════════════════
# #7c — fontes levemente maiores
# ════════════════════════════════════════════════════════════════════════════
def test_font_sizes_bumped():
    from geosteering_ai.gui.theme.tokens import ANTIGRAVITY_DARK

    assert ANTIGRAVITY_DARK.font_size_sm == 12
    assert ANTIGRAVITY_DARK.font_size_base == 14
    assert ANTIGRAVITY_DARK.font_size_lg == 18


# ════════════════════════════════════════════════════════════════════════════
# #5 — rótulo de paginação desambiguado (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_pagination_label_shows_range_and_total(qtbot):
    from apps.sim_manager.perspectives.simulation.results_view import ResultsView
    from apps.sim_manager.perspectives.simulation.results_viewmodel import (
        ResultsViewModel,
    )

    vm = ResultsViewModel(page_size=12)
    # 100 modelos → 9 páginas de 12; o rótulo NÃO pode ser só "1/9".
    h6 = np.zeros((100, 1, 1, 8, 1, 9), dtype=np.complex128)
    vm.set_result({"H6": h6, "positions_z": np.linspace(-1, 7, 8)})
    view = ResultsView(vm)
    qtbot.addWidget(view)
    txt = view._page_lbl.text()
    assert "de 100" in txt  # total de modelos explícito
    assert "modelos" in txt and "pág" in txt  # desambiguado (não é "84 modelos")


@pytest.mark.gui
def test_pagination_label_heatmap_mode(qtbot):
    """No modo heatmap (mostra TODOS de uma vez), o rótulo NÃO usa 'pág X/N'."""
    from apps.sim_manager.perspectives.simulation.results_view import ResultsView
    from apps.sim_manager.perspectives.simulation.results_viewmodel import (
        ResultsViewModel,
    )

    vm = ResultsViewModel(page_size=12)
    h6 = np.zeros((100, 1, 1, 8, 1, 9), dtype=np.complex128)
    vm.set_result({"H6": h6, "positions_z": np.linspace(-1, 7, 8)})
    vm.plot_mode = "heatmap"  # ensemble image — todos os modelos de uma vez
    view = ResultsView(vm)
    qtbot.addWidget(view)
    txt = view._page_lbl.text()
    assert "ensemble" in txt and "100 modelos" in txt
    assert "pág" not in txt  # heatmap não pagina
