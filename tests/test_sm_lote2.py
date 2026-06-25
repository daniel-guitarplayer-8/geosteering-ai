# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_lote2.py                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Lote 2 — paridade SM MVVM ↔ monólito (geometria canônica + reorg)        ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM MVVM — física tj/h1 canônica + 3 categorias + Fortran   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-09                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre o Lote 2: (2b) geometria canônica tj/h1 BYTE-IDÊNTICA ao         ║
# ║    monólito (reuso de sm_canonical_profiles); (2a) Task 4 n_layers do     ║
# ║    perfil; (2c) reorg em 3 categorias + 2 radios; (2e) paralelismo        ║
# ║    Fortran (estado/UI); (2d) BottomBar com 7 campos.                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do Lote 2 — geometria canônica + reorg 3-categorias + Fortran + BottomBar."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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
# 2b — GEOMETRIA CANÔNICA: VM == monólito (sm_canonical_profiles) p/ os 7 perfis
# ════════════════════════════════════════════════════════════════════════════
def test_canonical_geometry_matches_monolith_all_profiles():
    """GATE de fidelidade: tj/h1 do VM == funções canônicas do monólito (todos os 7).

    Antes (Lote 1) o VM usava ad-hoc h1=0.1·Σesp / tj=Σesp+h1 (assimétrico, por-modelo).
    Agora reusa compute_canonical_* → janela GLOBAL (max Σesp batch +20 m) + h1 simétrico.
    """
    from geosteering_ai.simulation.tests.sm_canonical_profiles import (
        compute_canonical_h1,
        compute_canonical_reference_tj,
    )
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
        list_canonical_models,
    )

    for name in list_canonical_models():
        vm = _make_sim_vm()
        cm = get_canonical_model(name)
        sesp = float(sum(float(x) for x in cm.esp))
        vm.apply_canonical_profile(name)  # auto_tj/auto_h1 default True
        tj_ref = compute_canonical_reference_tj(current_esp_sum=sesp)
        assert vm.tj == tj_ref, f"{name}: tj {vm.tj} != monólito {tj_ref}"
        assert vm.h1 == compute_canonical_h1(tj_ref, sesp), f"{name}: h1 != monólito"
        assert vm.n_layers_fixed == cm.n_layers, f"{name}: n_layers_fixed != perfil"


def test_canonical_auto_flags_independent():
    """auto_tj/auto_h1 independentes: tj manual + h1 auto usa o tj ATUAL."""
    from geosteering_ai.simulation.tests.sm_canonical_profiles import (
        compute_canonical_h1,
    )
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    vm = _make_sim_vm()
    vm.tj = 100.0
    vm.apply_canonical_profile("oklahoma_5", auto_tj=False, auto_h1=True)
    sesp = float(sum(float(x) for x in get_canonical_model("oklahoma_5").esp))
    assert vm.tj == 100.0  # auto_tj=False → tj NÃO mexido
    assert vm.h1 == compute_canonical_h1(100.0, sesp)  # h1 usa o tj atual (100)


def test_canonical_no_auto_keeps_geometry_but_sets_layers():
    """Sem auto: geometria intocada, MAS n_layers_fixed AINDA é setado (Task 4)."""
    vm = _make_sim_vm()
    vm.tj = 42.0
    vm.h1 = 3.0
    vm.apply_canonical_profile("oklahoma_3", auto_tj=False, auto_h1=False)
    assert vm.tj == 42.0 and vm.h1 == 3.0  # geometria intocada
    assert vm.n_layers_fixed == 3  # Task 4: nº de camadas reflete o perfil


# ════════════════════════════════════════════════════════════════════════════
# Turn 7 item 1 — aplicar perfil espelha os campos DERIVADOS do estocástico
# (ρ/λ/min_thickness/anisotropia/n_models/range) — fecha "só Oklahoma 3 importa"
# ════════════════════════════════════════════════════════════════════════════
def test_canonical_profile_mirrors_stochastic_fields():
    """Item 1: aplicar um perfil grava os mesmos campos derivados do monólito
    (simulation_manager.py:1974-2057). Valida 3 perfis representativos."""
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    for name in ("oklahoma_28", "oklahoma_15", "hou_7"):
        vm = _make_sim_vm()
        cm = get_canonical_model(name)
        rho_h = [float(x) for x in cm.rho_h]
        rho_v = [float(x) for x in cm.rho_v]
        esp = [float(x) for x in cm.esp]
        n = int(cm.n_layers)
        lambdas = [
            float(max(rv / rh, 1.0)) ** 0.5 for rh, rv in zip(rho_h, rho_v) if rh > 0.0
        ]
        aniso = any(abs(rv - rh) > 1e-9 * max(rh, 1.0) for rh, rv in zip(rho_h, rho_v))

        vm.apply_canonical_profile(name)

        assert vm.rho_h_min == min(rho_h), name
        assert vm.rho_h_max == max(rho_h), name
        assert vm.min_thickness == max(1e-3, min(esp)), name
        assert vm.anisotropic is aniso, name
        assert vm.lambda_min == max(1.0, min(lambdas)), name
        assert vm.lambda_max == max(vm.lambda_min, max(lambdas)), name
        assert vm.n_models == 1, name  # perfil determinístico (monólito spin_nmodels=1)
        assert vm.n_layers_min == n, name
        assert vm.n_layers_max == n + 1, name  # exclusivo → colapsa em {n}


def test_canonical_isotropic_profile_disables_anisotropy():
    """oklahoma_15/devine_8 são isotrópicos (ρᵥ==ρₕ) → anisotropic=False, λ=[1,1]."""
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    for name in ("oklahoma_15", "devine_8"):
        vm = _make_sim_vm()
        cm = get_canonical_model(name)
        assert all(rv == rh for rh, rv in zip(cm.rho_h, cm.rho_v)), f"{name} isotrópico"
        vm.apply_canonical_profile(name)
        assert vm.anisotropic is False, name
        assert vm.lambda_min == 1.0 and vm.lambda_max == 1.0, name


# ════════════════════════════════════════════════════════════════════════════
# Turn 7 item 3 — defaults do VM == página de produção do SM monólito
# ════════════════════════════════════════════════════════════════════════════
def test_mvvm_defaults_match_monolith_production():
    """Item 3: GATE de regressão dos defaults alinhados ao monólito. n_pos é
    DERIVADO (600 = SEQUENCE_LENGTH da errata). Errata freq/TR/dip já coincidia."""
    vm = _make_sim_vm()
    assert vm.h1 == 10.0
    assert vm.tj == 120.0
    assert vm.p_med == 0.2
    assert vm.n_pos == 600  # ceil(120/0.2), dip0=0 → SEQUENCE_LENGTH
    assert vm.n_models == 2000
    assert vm.min_thickness == 0.5
    assert vm.rho_h_max == 1800.0
    assert vm.n_layers_max == 32  # exclusivo (monólito spinbox 31 +1)
    assert vm.n_layers_fixed is None  # "n camadas fixo" OFF
    assert vm.save_fortran_artifacts is True
    assert vm.h1_auto is False and vm.tj_auto is False
    assert vm.validate() == []  # default permanece válido


def test_auto_geometry_flags_persist_in_session():
    """Achado da revisão: h1_auto/tj_auto agora persistem na sessão (eram View-only)."""
    import json

    vm = _make_sim_vm()
    assert (
        vm.h1_auto is False and vm.tj_auto is False
    )  # default OFF (paridade produção, item 3)
    vm.h1_auto = False
    vm.tj_auto = False
    d = json.loads(json.dumps(vm.to_session_dict()))
    assert d["h1_auto"] is False and d["tj_auto"] is False
    vm2 = _make_sim_vm()
    vm2.load_session_dict(d)
    assert vm2.h1_auto is False and vm2.tj_auto is False


# ════════════════════════════════════════════════════════════════════════════
# (PR-2) Paralelismo Fortran REMOVIDO — o SM MVVM usa só Numba JIT + JAX GPU.
# Os testes de n_workers_fortran/threads_fortran foram retirados junto com os campos.
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# 2c — REORG em 3 categorias + 2 radios (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_simulator_view_three_categories(qtbot):
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_sim_vm()
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    # Categoria 1: "Perfil Pré-configurado" (top-level)
    assert view._profile_box.title() == "Perfil Pré-configurado"
    # h1/tj-auto separados (2 checkboxes, não o antigo único)
    assert view._h1_auto is not None and view._tj_auto is not None
    assert not hasattr(view, "_auto_geo_check")
    # 2 radios (sem combo de modo)
    assert view._radio_random.text().startswith("Aleatória")
    assert view._radio_manual.text().startswith("Manual")
    assert not hasattr(view, "_geo_mode")
    # PR-2: par ÚNICO workers/threads (Fortran removido — só Numba JIT + JAX GPU)
    assert view._n_workers.minimum() == 1 and view._n_workers.maximum() == 256
    assert view._threads.minimum() == 1
    assert not hasattr(view, "_n_workers_fortran")
    assert not hasattr(view, "_threads_fortran")


@pytest.mark.gui
def test_simulator_view_has_vertical_scroll(qtbot):
    """Fix Lote 2: a coluna de config tem QScrollArea vertical (conteúdo > janela).

    O conteúdo cresceu (3 categorias + paralelismo Numba/Fortran + saída + galeria)
    e ultrapassa ~2000px — sem scroll, os widgets de baixo ficavam inacessíveis.
    """
    from apps.sim_manager.perspectives.simulation.view import SimulatorView
    from geosteering_ai.gui.qt_compat import Qt, QtWidgets

    vm = _make_sim_vm()
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    sa = view.findChild(QtWidgets.QScrollArea, "SimulatorScroll")
    assert sa is not None and sa.widgetResizable()
    # h-scroll OFF (sem vazamento horizontal); v-scroll quando necessário.
    assert sa.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    # conteúdo bem mais alto que uma janela típica → scroll é necessário.
    assert sa.widget().sizeHint().height() > 800
    # widgets seguem acessíveis por atributo (binding intacto). PR-2 (#1): a galeria
    # (_results) saiu desta aba p/ a perspectiva Resultados.
    assert view._profile_box is not None and view._output_box is not None
    assert not hasattr(view, "_results")


@pytest.mark.gui
def test_history_panel_wrapped_in_scroll(qtbot):
    """Fix Lote 2: o painel de histórico (coluna direita) é envolvido em QScrollArea."""
    from geosteering_ai.gui.qt_compat import QtWidgets
    from geosteering_ai.gui.shell.widgets.secondary_sidebar import SecondarySidebar

    sb = SecondarySidebar()
    qtbot.addWidget(sb)
    panel = QtWidgets.QWidget()
    sb.set_history_panel(panel)
    scroll = sb._tabs.widget(0)
    assert isinstance(scroll, QtWidgets.QScrollArea)
    assert scroll.widget() is panel
    assert sb._history is panel  # API pública (self._history) inalterada


@pytest.mark.gui
def test_view_apply_canonical_updates_layer_count(qtbot):
    """Task 4 end-to-end: aplicar Oklahoma 28 na View atualiza o spinbox p/ 28 (não 5)."""
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_sim_vm()
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    assert view._geo_nlf.value() == 5  # default antes de aplicar
    idx = view._canonical_combo.findData("oklahoma_28")
    view._canonical_combo.setCurrentIndex(idx)
    view._on_apply_canonical()
    assert view._geo_nlf.value() == 28  # Task 4: contador atualizado
    assert view._radio_manual.isChecked()  # comutou p/ Manual
    assert "28 camadas" in view._manual_info.text()
    assert round(view._tj.value(), 2) == 67.85  # física canônica refletida na UI
    assert round(view._h1.value(), 2) == 10.00


# ════════════════════════════════════════════════════════════════════════════
# 2d — BottomBar com 7 campos (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_bottombar_full_fields(qtbot):
    from apps.sim_manager.main_window import SM_MainWindow
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    win = SM_MainWindow(AppContext(app_name="SM"))
    qtbot.addWidget(win)
    win.add_perspective(SimulationPerspective())
    assert win._sb_exp.text().startswith("Exp:")
    assert "ocioso" in win._sb_state.text()
    assert win._sb_elapsed.text().startswith("Elapsed:")
    assert win._sb_throughput.text().startswith("Throughput:")
    assert win._sb_cache.text().startswith("Cache:")
    assert win._sb_binding.text().startswith("Binding:")
