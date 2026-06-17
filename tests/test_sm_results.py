# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_results.py                                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0011d-sm-results                                           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app MVVM — galeria do ensemble (Fatia 4)                ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-06                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre os ACs da spec 0011d: ResultsViewModel PURO (curvas Re/Im/Mag/    ║
# ║    Phase AC-1, clamp AC-2, paginação AC-3, cache LRU AC-4), galeria render  ║
# ║    (AC-5, gui), .session roundtrip (AC-6) e a fronteira/wiring (AC-7).      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0011d — galeria do ensemble (ResultsViewModel · galeria · .session)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def _synthetic_h6(n_models=5, n_tr=2, n_ang=3, n_pos=8, n_f=2):
    """H6 sintético complexo (n_models, nTR, nAng, n_pos, nf, 9), determinístico."""
    rng = np.random.default_rng(0)
    shape = (n_models, n_tr, n_ang, n_pos, n_f, 9)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _result(**kw):
    h6 = _synthetic_h6(**kw)
    return {"H6": h6, "positions_z": np.linspace(-1.0, 9.0, h6.shape[3]), "info": {}}


def _make_results_vm(page_size=12, cache=None):
    from apps.sim_manager.perspectives.simulation.results_viewmodel import (
        ResultsViewModel,
    )

    return ResultsViewModel(page_size=page_size, cache=cache)


def _make_sim_vm():
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    class StubService:
        def __init__(self) -> None:
            self.finished = VMSignal()
            self.error = VMSignal()
            self.requests: list = []

        def run(self, request) -> None:  # noqa: ANN001
            self.requests.append(request)

    return SimulationViewModel(service=StubService())


# ════════════════════════════════════════════════════════════════════════════
# AC-7 — ResultsViewModel PURO (sem Qt)
# ════════════════════════════════════════════════════════════════════════════
def test_results_vm_importable_without_qt():
    """AC-7 — importar o ResultsViewModel NÃO puxa Qt (Princípio X)."""
    code = (
        "import sys\n"
        "import apps.sim_manager.perspectives.simulation.results_viewmodel  # noqa\n"
        "bad = [m for m in ('PyQt6', 'PySide6') if m in sys.modules]\n"
        "assert not bad, f'ResultsViewModel puxou Qt: {bad}'\n"
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


# ════════════════════════════════════════════════════════════════════════════
# AC-1 — curva == transform(H6[m, tr, dip, :, freq, comp])
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "kind,fn",
    [
        ("re", np.real),
        ("im", np.imag),
        ("mag", np.abs),
        ("phase", lambda h: np.degrees(np.angle(h))),
    ],
)
def test_curve_for_matches_transform(kind, fn):
    """AC-1 — curve_for == {re/im/mag/phase}(componente) p/ a config selecionada."""
    rvm = _make_results_vm()
    res = _result()
    rvm.set_result(res)
    rvm.component_index = 8  # Hzz
    rvm.plot_kind = kind
    rvm.tr_index = 1
    rvm.dip_index = 2
    rvm.freq_index = 1
    h6 = res["H6"]
    for m in range(h6.shape[0]):
        curve = rvm.curve_for(m)
        expected = np.asarray(fn(h6[m, 1, 2, :, 1, 8]), dtype=np.float64)
        assert curve.shape == (h6.shape[3],)
        assert np.allclose(curve, expected)
        assert np.all(np.isfinite(curve))


# ════════════════════════════════════════════════════════════════════════════
# AC-2 — clamp de seletores
# ════════════════════════════════════════════════════════════════════════════
def test_set_result_clamps_selectors():
    """AC-2 — set_result clampa índices fora de range aos novos dims."""
    rvm = _make_results_vm()
    rvm.set_result(_result(n_tr=3, n_ang=4, n_f=2))
    rvm.tr_index = 2
    rvm.dip_index = 3
    rvm.freq_index = 1
    rvm.component_index = 5
    # novo resultado MENOR → índices re-clampados
    rvm.set_result(_result(n_tr=1, n_ang=1, n_f=1))
    assert rvm.tr_index == 0
    assert rvm.dip_index == 0
    assert rvm.freq_index == 0
    assert 0 <= rvm.component_index <= 8


def test_setters_clamp_to_dims():
    """AC-2 — setar índice fora de range é clampado, sem crash."""
    rvm = _make_results_vm()
    rvm.set_result(_result(n_tr=2, n_ang=2, n_f=2))
    rvm.tr_index = 99
    rvm.dip_index = -5
    rvm.freq_index = 99
    rvm.component_index = 99
    assert rvm.tr_index == 1
    assert rvm.dip_index == 0
    assert rvm.freq_index == 1
    assert rvm.component_index == 8
    rvm.plot_kind = "weird"  # inválido → "re"
    assert rvm.plot_kind == "re"


def test_curve_for_raises_out_of_range():
    """AC-2 — curve_for sem resultado ou índice inválido levanta IndexError."""
    rvm = _make_results_vm()
    with pytest.raises(IndexError):
        rvm.curve_for(0)  # sem resultado
    rvm.set_result(_result(n_models=3))
    with pytest.raises(IndexError):
        rvm.curve_for(3)


def test_set_result_rejects_malformed_h6():
    """AC-2 (robustez) — H6 sem 9 componentes (ou não-6D) é rejeitado na ENTRADA."""
    rvm = _make_results_vm()
    rvm.set_result(_result(n_models=3))
    assert rvm.has_result  # 9 componentes → OK
    # eixo −1 com 7 componentes (malformado) → rejeitado (has_result False, sem crash)
    bad = np.zeros((3, 1, 1, 8, 1, 7), dtype=np.complex128)
    rvm.set_result({"H6": bad, "positions_z": np.linspace(-1.0, 9.0, 8)})
    assert not rvm.has_result
    assert rvm.dims is None
    # H6 não-6D também rejeitado
    rvm.set_result(
        {"H6": np.zeros((3, 8, 9), dtype=np.complex128), "positions_z": None}
    )
    assert not rvm.has_result


# ════════════════════════════════════════════════════════════════════════════
# AC-3 — paginação
# ════════════════════════════════════════════════════════════════════════════
def test_pagination():
    """AC-3 — n_pages/page_models corretos; última página parcial; page clampada."""
    rvm = _make_results_vm(page_size=3)
    rvm.set_result(_result(n_models=7))
    assert rvm.n_pages == 3  # ceil(7/3)
    assert rvm.page_models() == [0, 1, 2]
    rvm.page = 2
    assert rvm.page_models() == [6]  # última, parcial
    rvm.page = 99
    assert rvm.page == 2  # clampada


def test_pagination_empty_without_result():
    """AC-3 — sem resultado: n_pages=0, page_models vazio."""
    rvm = _make_results_vm()
    assert rvm.n_pages == 0
    assert rvm.page_models() == []
    assert not rvm.has_result


# ════════════════════════════════════════════════════════════════════════════
# AC-4 — cache LRU de curvas
# ════════════════════════════════════════════════════════════════════════════
def test_cache_hit_and_eviction():
    """AC-4 — 2ª chamada vem do cache (mesmo objeto); cache bounded (LRU evicta)."""
    from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache

    rvm = _make_results_vm(cache=LRUPlotCache(maxlen=2, max_bytes=1.0e9))
    rvm.set_result(_result(n_models=5))
    a1 = rvm.curve_for(0)
    a2 = rvm.curve_for(0)
    assert a1 is a2  # cache hit → MESMO objeto
    rvm.curve_for(1)
    rvm.curve_for(2)  # maxlen=2 → modelo 0 evictado
    a3 = rvm.curve_for(0)
    assert a1 is not a3  # recomputou (não veio do cache)
    assert len(rvm._cache) <= 2  # bounded


def test_set_result_clears_cache():
    """AC-4 — um novo resultado limpa o cache de curvas (não vaza do anterior)."""
    rvm = _make_results_vm()
    rvm.set_result(_result(n_models=3))
    rvm.curve_for(0)
    assert len(rvm._cache) >= 1
    rvm.set_result(_result(n_models=3))
    assert len(rvm._cache) == 0


# ════════════════════════════════════════════════════════════════════════════
# AC-7 (wiring) — SimulationViewModel alimenta a galeria ao concluir
# ════════════════════════════════════════════════════════════════════════════
def test_simulation_vm_feeds_results_on_finished():
    """AC-7 — finished → results.set_result + result_ready ainda emitido."""
    vm = _make_sim_vm()
    rec: list = []
    vm.result_ready.connect(rec.append)
    res = _result(n_models=4)
    vm._service.finished.emit(res)
    assert vm.status == "done"
    assert rec and rec[0] is res  # result_ready preservado
    assert vm.results.has_result  # galeria alimentada
    assert vm.results.n_models == 4


# ════════════════════════════════════════════════════════════════════════════
# AC-6 — .session roundtrip (params; resultado reproduzível pela seed)
# ════════════════════════════════════════════════════════════════════════════
def test_session_roundtrip_params():
    """AC-6 — to_session_dict → JSON → load_session_dict reconstrói os params."""
    vm = _make_sim_vm()
    vm.frequencies = (20000.0, 40000.0)
    vm.dips = (0.0, 30.0)
    vm.generator = "halton"
    vm.geology_mode = "stochastic"
    vm.n_layers_fixed = None
    vm.n_layers_max = 12
    vm.anisotropic = False
    vm.rng_seed = 123
    blob = json.dumps(vm.to_session_dict())  # JSON-serializável (AC-6)

    vm2 = _make_sim_vm()
    vm2.load_session_dict(json.loads(blob))
    assert vm2.frequencies == (20000.0, 40000.0)
    assert vm2.dips == (0.0, 30.0)
    assert vm2.generator == "halton"
    assert vm2.n_layers_fixed is None
    assert vm2.n_layers_max == 12
    assert vm2.anisotropic is False
    assert vm2.rng_seed == 123


def test_session_file_roundtrip(tmp_path):
    """AC-6 — SessionDocument.save/load (.session JSON, sem pickle) round-trip."""
    from geosteering_ai.gui.persistence.session import SessionDocument

    vm = _make_sim_vm()
    vm.frequencies = (15000.0,)
    vm.rng_seed = 7
    path = tmp_path / "exp.session"
    SessionDocument(data=vm.to_session_dict()).save(str(path))
    assert "pickle" not in path.read_text()  # JSON puro, sem pickle
    doc = SessionDocument.load(str(path))
    vm2 = _make_sim_vm()
    vm2.load_session_dict(doc.data)
    assert vm2.frequencies == (15000.0,)
    assert vm2.rng_seed == 7


# ════════════════════════════════════════════════════════════════════════════
# AC-5 — galeria render (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_results_view_renders_gallery(qtbot):
    """AC-5 — ResultsView monta a grade após set_result; trocar seletor re-renderiza."""
    from apps.sim_manager.perspectives.simulation.results_view import ResultsView

    rvm = _make_results_vm(page_size=4)
    view = ResultsView(rvm)
    qtbot.addWidget(view)
    rvm.set_result(_result(n_models=5, n_tr=2, n_ang=2, n_f=2))
    # paginação: 5 modelos / 4 = 2 páginas
    assert "1/2" in view._page_lbl.text()
    # config spinners habilitados (dims > 1)
    assert view._tr.isEnabled() and view._freq.isEnabled()
    # trocar componente/plot-kind/página re-renderiza sem crash
    view._on_cmp_changed(8)  # Hzz
    view._on_kind_changed(2)  # |H|
    view._on_next()  # página 2
    assert "2/2" in view._page_lbl.text()
    assert rvm.page == 1 and rvm.page_models() == [4]  # última página parcial


@pytest.mark.gui
def test_simulator_view_has_no_embedded_gallery(qtbot):
    """PR-2 (#1) — a galeria SAIU da aba Simulação (vive em Resultados); inputs intactos."""
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_sim_vm()
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    assert not hasattr(view, "_results")  # galeria NÃO está mais embutida
    # o ResultsViewModel continua no SimulationViewModel (publicado em ctx.extras)
    vm.results.set_result(_result(n_models=3))
    assert vm.results.has_result
    # sync de inputs após "carregar sessão" segue funcionando
    vm.load_session_dict(
        {"frequencies": [12345.0], "n_models": 7, "generator": "halton"}
    )
    view._sync_inputs_from_vm()
    assert "12345" in view._freqs.text()
    assert view._n_models.value() == 7
    assert view._geo_generator.currentText() == "halton"


@pytest.mark.gui
def test_results_perspective_reuses_simulation_vm(qtbot):
    """PR-2 (#1) — a ResultsPerspective liga a galeria ao MESMO ResultsViewModel da Simulação."""
    from apps.sim_manager.perspectives.results.perspective import ResultsPerspective
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    ctx = AppContext(app_name="t")
    # Simulação builda primeiro → publica ctx.extras["results_vm"].
    sim_view = SimulationPerspective().build_view(ctx)
    qtbot.addWidget(sim_view)
    assert "results_vm" in ctx.extras
    # Resultados reusa o MESMO VM (1 VM, 2 Views).
    res_persp = ResultsPerspective()
    assert res_persp.build_viewmodel(ctx) is ctx.extras["results_vm"]
    res_view = res_persp.build_view(ctx)
    qtbot.addWidget(res_view)
    # um resultado simulado aparece na galeria de Resultados (VM compartilhado).
    ctx.extras["results_vm"].set_result(_result(n_models=3))
    assert res_persp._vm.has_result


def test_results_perspective_fallback_without_extras():
    """Sem ctx.extras['results_vm'] → cria um VM vazio (galeria 'sem resultado', sem crash)."""
    from apps.sim_manager.perspectives.results.perspective import ResultsPerspective
    from apps.sim_manager.perspectives.simulation.results_viewmodel import (
        ResultsViewModel,
    )
    from geosteering_ai.gui.shell.context import AppContext

    vm = ResultsPerspective().build_viewmodel(AppContext(app_name="t"))
    assert isinstance(vm, ResultsViewModel) and not vm.has_result


def test_results_perspective_fallback_with_wrong_type_extras():
    """ctx.extras['results_vm'] de TIPO ERRADO → fallback p/ VM vazio (guard isinstance)."""
    from apps.sim_manager.perspectives.results.perspective import ResultsPerspective
    from apps.sim_manager.perspectives.simulation.results_viewmodel import (
        ResultsViewModel,
    )
    from geosteering_ai.gui.shell.context import AppContext

    ctx = AppContext(app_name="t")
    ctx.extras["results_vm"] = "não é um ResultsViewModel"  # poluição
    vm = ResultsPerspective().build_viewmodel(ctx)
    assert isinstance(vm, ResultsViewModel) and not vm.has_result


@pytest.mark.gui
def test_results_binds_shared_vm_via_real_shell_boot(qtbot):
    """Contrato de boot (#1): registrando AMBAS as perspectivas na SM_MainWindow na
    ordem de produção, ativar SÓ a aba Resultados ainda liga a galeria ao MESMO VM da
    Simulação (Simulação order=0 builda no boot e publica results_vm). Protege a fiação
    contra reordenação acidental em app.py.
    """
    from apps.sim_manager.main_window import SM_MainWindow
    from apps.sim_manager.perspectives.results.perspective import ResultsPerspective
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    win = SM_MainWindow(AppContext(app_name="boot"))
    qtbot.addWidget(win)
    win.add_perspective(SimulationPerspective())  # order=0 → builda no boot
    win.add_perspective(ResultsPerspective())  # order=1 → lazy
    # ativa a aba Resultados pela rail (build lazy).
    idx = next(
        i
        for i in range(win._host_count())
        if win._by_container.get(id(win._host_widget(i))).id == "results"
    )
    win._on_rail_selected(idx)
    shared = win.ctx.extras.get("results_vm")
    assert shared is not None
    # alimenta via o VM compartilhado → a galeria de Resultados reflete (mesmo objeto).
    shared.set_result(_result(n_models=3))
    assert shared.has_result


# ════════════════════════════════════════════════════════════════════════════
# Toggle matplotlib ↔ PyQtGraph (quick win — escolha de backend de plot)
# ════════════════════════════════════════════════════════════════════════════
def test_results_vm_plot_backend_default_and_setter():
    """O ResultsViewModel default p/ PYQTGRAPH (Fatia 6d); o setter aceita enum E str."""
    from geosteering_ai.gui.plot_backends.base import PlotBackend

    rvm = _make_results_vm()
    assert rvm.plot_backend == PlotBackend.PYQTGRAPH  # default 6d (interativo)
    rec: list = []
    rvm.changed.connect(lambda name, value: rec.append(name))
    rvm.plot_backend = "matplotlib"  # str → enum
    assert rvm.plot_backend == PlotBackend.MATPLOTLIB
    assert "_plot_backend" in rec  # emitiu changed (a View re-renderiza)
    rvm.plot_backend = PlotBackend.PYQTGRAPH  # enum
    assert rvm.plot_backend == PlotBackend.PYQTGRAPH


def test_session_roundtrip_includes_plot_backend():
    """O .session persiste o backend de plot (str do enum) e o restaura."""
    from geosteering_ai.gui.plot_backends.base import PlotBackend

    vm = _make_sim_vm()
    vm.results.plot_backend = PlotBackend.PYQTGRAPH
    d = vm.to_session_dict()
    assert d["plot_backend"] == "pyqtgraph"
    vm2 = _make_sim_vm()
    vm2.load_session_dict(json.loads(json.dumps(d)))
    assert vm2.results.plot_backend == PlotBackend.PYQTGRAPH


@pytest.mark.gui
def test_results_view_backend_toggle_rebuilds_canvas(qtbot):
    """Trocar o backend recria o canvas (pyqtgraph default → matplotlib → pyqtgraph)."""
    pytest.importorskip("pyqtgraph")
    from apps.sim_manager.perspectives.simulation.results_view import ResultsView
    from geosteering_ai.gui.plot_backends.base import PlotBackend

    rvm = _make_results_vm(page_size=4)
    view = ResultsView(rvm)
    qtbot.addWidget(view)
    rvm.set_result(_result(n_models=3))
    assert view._active_backend == PlotBackend.PYQTGRAPH  # default 6d
    first_widget = view._canvas.widget()

    # via combo (caminho do usuário) → recria o canvas p/ matplotlib
    view._on_backend_changed("matplotlib")
    assert rvm.plot_backend == PlotBackend.MATPLOTLIB
    assert view._active_backend == PlotBackend.MATPLOTLIB
    assert view._canvas.widget() is not first_widget  # canvas recriado

    # volta p/ pyqtgraph
    view._on_backend_changed("pyqtgraph")
    assert view._active_backend == PlotBackend.PYQTGRAPH


@pytest.mark.gui
def test_results_view_falls_back_when_backend_unavailable(qtbot, monkeypatch):
    """Revisão quick-wins — backend sem deps (.session) → fallback p/ matplotlib, sem crash.

    Um ``.session`` pode carregar um backend enum-válido mas com deps ausentes
    (ex.: "plotly" sem WebEngine). ``make_canvas`` levanta ImportError; a View
    DEVE cair p/ matplotlib (sem quebrar a galeria) e reconciliar o VM.
    """
    from apps.sim_manager.perspectives.simulation import results_view as rv_mod
    from apps.sim_manager.perspectives.simulation.results_view import ResultsView
    from geosteering_ai.gui.plot_backends.base import PlotBackend

    orig_make = rv_mod.make_canvas

    def fake_make(backend, **kw):
        if backend != PlotBackend.MATPLOTLIB:
            raise ImportError(
                f"dep ausente p/ {backend}"
            )  # simula backend indisponível
        return orig_make(backend, **kw)

    monkeypatch.setattr(rv_mod, "make_canvas", fake_make)

    # (1) construção com backend "indisponível" → fallback no __init__ (sem crash)
    rvm = _make_results_vm()
    rvm.plot_backend = PlotBackend.PYQTGRAPH  # "indisponível" via fake
    view = ResultsView(rvm)
    qtbot.addWidget(view)
    assert view._active_backend == PlotBackend.MATPLOTLIB
    assert rvm.plot_backend == PlotBackend.MATPLOTLIB  # VM reconciliado

    # (2) toggle runtime p/ backend indisponível → fallback no _rebuild (sem crash)
    rvm.set_result(_result(n_models=2))
    view._on_backend_changed("pyqtgraph")  # fake faz ImportError → fallback
    assert view._active_backend == PlotBackend.MATPLOTLIB
    assert rvm.plot_backend == PlotBackend.MATPLOTLIB
