# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sim_app_skeleton.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0011a-sm-app-skeleton + 0011b-sm-params                    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app MVVM — walking skeleton + params completos          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre os ACs das specs 0011a (skeleton) + 0011b (params completos):     ║
# ║    ViewModel PURO multi-valor (service stub, sem Qt), validação POR         ║
# ║    ELEMENTO, ``n_pos`` derivado, fidelidade ``positions_z`` à fórmula       ║
# ║    Fortran, fidelidade física (simulate_batch real → shape H6) inclusive    ║
# ║    multi-config, threading (Worker off-thread), end-to-end real e a         ║
# ║    fronteira de import (VM sem Qt; core sem gui/apps).                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes das specs 0011a/0011b — SM MVVM (VM puro multi-valor · fidelidade · e2e)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Qt headless — mesmo padrão das outras suites GUI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
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


def _make_vm():
    """ViewModel real com service stub (puro, sem Qt)."""
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    return SimulationViewModel(service=_make_stub_service())


# ════════════════════════════════════════════════════════════════════════════
# RF-3 (0011b) — SimulationViewModel PURO multi-valor (sem Qt; service stub)
# ════════════════════════════════════════════════════════════════════════════
def test_vm_validate_run_and_result_with_stub():
    """AC-2 — VM puro multi-valor: run monta SimRequest completo; finished → done."""
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    svc = _make_stub_service()
    vm = SimulationViewModel(service=svc)
    rec: list = []
    vm.result_ready.connect(rec.append)

    vm.frequencies = (20000.0, 40000.0)
    vm.dips = (0.0, 30.0)
    vm.tr_spacings = (1.0,)
    vm.n_models = 3
    vm.run()
    assert len(svc.requests) == 1  # validou e delegou
    req = svc.requests[0]
    # o SimRequest carrega TODOS os params multi-valor + geometria
    assert req.frequencies_hz == (20000.0, 40000.0)
    assert req.dip_degs == (0.0, 30.0)
    assert req.tr_spacings_m == (1.0,)
    assert req.n_models == 3
    assert (req.h1, req.tj, req.p_med) == (
        10.0,
        120.0,
        0.2,
    )  # defaults == monólito (item 3)
    assert vm.status == "running"

    # o service "termina" → VM atualiza estado e re-emite
    svc.finished.emit({"H6": "fake", "positions_z": "z"})
    assert vm.status == "done"
    assert rec == [{"H6": "fake", "positions_z": "z"}]
    assert vm.last_result is not None


def test_vm_rejects_invalid_params_without_calling_service():
    """AC-2 — validação reprova params fora da errata; NÃO chama o service."""
    vm = _make_vm()
    vm._service.requests.clear()
    vm.frequencies = (5.0,)  # < 10 Hz (limite do simulador) → inválido
    vm.run()
    assert vm._service.requests == []  # NÃO delegou
    assert vm.status == "error"
    assert vm.last_result is not None and vm.last_result["errors"]


def test_vm_validates_per_element():
    """AC-2 — errata POR ELEMENTO: 1 valor ruim na lista já reprova (freq/dip/TR)."""
    # freq: 1 boa + 1 ruim
    vm = _make_vm()
    vm.frequencies = (20000.0, 5.0)  # 5 Hz < 100 → inválido
    errs = vm.validate()
    assert any("Frequência" in e and "5" in e for e in errs)

    # dip: 1 bom + 1 ruim (> 90°, o core rejeitaria — paridade Fortran)
    vm = _make_vm()
    vm.dips = (0.0, 100.0)
    errs = vm.validate()
    assert any("Dip" in e for e in errs)

    # TR: 1 bom + 1 ruim (> 50 m — limite do simulador, multi_forward.py:617)
    vm = _make_vm()
    vm.tr_spacings = (1.0, 60.0)
    errs = vm.validate()
    assert any("TR" in e for e in errs)


def test_vm_rejects_nonpositive_geometry_and_empty_lists():
    """AC-2 — h1/tj/p_med > 0; listas não-vazias; n_models ≥ 1."""
    vm = _make_vm()
    vm.h1 = 0.0
    vm.tj = 0.0
    vm.p_med = 0.0
    vm.frequencies = ()
    vm.dips = ()
    vm.tr_spacings = ()
    vm.n_models = 0
    errs = vm.validate()
    blob = " ".join(errs)
    assert "h1" in blob and "tj" in blob and "p_med" in blob
    assert "frequência" in blob and "dip" in blob and "TR" in blob
    assert "modelos" in blob


def test_vm_error_signal_sets_error_status():
    """AC-2 — erro do service → status 'error' + last_result['error']."""
    vm = _make_vm()
    vm.run()
    vm._service.error.emit("falha na física")
    assert vm.status == "error"
    assert vm.last_result == {"error": "falha na física"}


def test_vm_rejects_dip_above_90():
    """AC-2 (fidelidade) — dip > 90° é REJEITADO no VM (o core valida [0, 90]°)."""
    vm = _make_vm()
    vm.dips = (100.0,)  # o simulate_batch recusaria (paridade Fortran [0,90])
    vm.run()
    assert vm._service.requests == []  # NÃO delegou
    assert vm.status == "error"
    assert vm.last_result is not None
    assert any("Dip" in e for e in vm.last_result["errors"])
    # 89° (dentro do range, geometria sã) é aceito
    vm2 = _make_vm()
    vm2.dips = (89.0,)
    assert vm2.validate() == []


def test_vm_rejects_degenerate_n_pos():
    """AC-2 (robustez) — dip≈90° (n_pos explode) é rejeitado ANTES de despachar (anti-OOM).

    dip=90° está DENTRO de [0,90]° mas é geometria degenerada (cos→0 ⇒ n_pos≈1e7 ⇒
    tensor H6 de dezenas de GB). O VM barra isso com erro claro; nada chega ao batch.
    """
    vm = _make_vm()
    vm.dips = (90.0,)  # default tj=10, p_med=1 ⇒ n_pos ≈ 1e7 > _N_POS_MAX
    vm.run()
    assert vm._service.requests == []  # NÃO despachou (evita OOM)
    assert vm.status == "error"
    assert any("degenerada" in e or "n_pos" in e for e in vm.last_result["errors"])


def test_vm_accepts_widened_simulator_ranges():
    """AC-2 (fidelidade) — VM aceita o que simulate_batch aceita: TR≤50 m, freq∈[10,2e6].

    A errata do VM espelha o SIMULADOR (multi_forward._validate_multi_inputs +
    SimulationConfig), não o pipeline DL ([100,1e6]/[0.1,10]) — senão rejeitaria
    configs físicas válidas (ex.: TR=25 m, deep-reading PeriScope).
    """
    vm = _make_vm()
    vm.frequencies = (10.0, 2.0e6)  # limites do simulador (config.py:158)
    vm.dips = (0.0,)
    vm.tr_spacings = (0.1, 25.0, 50.0)  # 25 m = deep-reading (rejeitado por [0.1,10])
    assert vm.validate() == []


# ════════════════════════════════════════════════════════════════════════════
# AC-4 (0011b) — n_pos derivado (read-only) na convenção Fortran
# ════════════════════════════════════════════════════════════════════════════
def test_vm_n_pos_derived_matches_formula():
    """AC-4 — vm.n_pos == max(1, ceil(tj/(p_med·cos(dip0)))) (convenção Fortran)."""
    vm = _make_vm()
    vm.h1, vm.tj, vm.p_med = 2.0, 17.0, 0.5
    vm.dips = (30.0, 60.0)  # só dip0 conta p/ n_pos
    cos_d = max(1e-6, math.cos(math.radians(30.0)))
    expected = max(1, int(math.ceil(17.0 / (0.5 * cos_d))))
    assert vm.n_pos == expected


def test_vm_n_pos_zero_when_unset():
    """AC-4 — n_pos derivado é 0 (exibição) se dips vazio ou p_med/tj ≤ 0 (sem crash)."""
    vm = _make_vm()
    vm.dips = ()
    assert vm.n_pos == 0
    vm.dips = (0.0,)
    vm.p_med = 0.0
    assert vm.n_pos == 0


# ════════════════════════════════════════════════════════════════════════════
# AC-1 (0011b) — fidelidade positions_z à fórmula EXATA do monólito
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "h1,tj,p_med,dip0",
    [
        (1.0, 10.0, 1.0, 0.0),  # dip 0 → cos=1 → n_pos=ceil(tj/p_med)
        (2.0, 17.0, 0.5, 30.0),  # geometria/dip não-triviais
        (1.0, 10.0, 1.0, 60.0),  # cos(60°)=0.5 → dobra n_pos
        (0.5, 8.0, 2.0, 89.0),  # dip extremo → muitos pontos (cos→0)
        (1.0, 10.0, 1.0, 90.0),  # guard 1e-6 evita ÷0 em 90°
    ],
)
def test_compute_positions_z_matches_monolith_formula(h1, tj, p_med, dip0):
    """AC-1 — _compute_positions_z replica byte-a-byte a fórmula Fortran do monólito."""
    import numpy as np

    from geosteering_ai.gui.services.sim_request import SimRequest, _compute_positions_z

    got = _compute_positions_z(SimRequest(h1=h1, tj=tj, p_med=p_med, dip_degs=(dip0,)))
    # fórmula canônica do monólito (simulation_manager.py:~8221-8225)
    cos_d = max(1e-6, math.cos(math.radians(abs(dip0))))
    n_pos = max(1, int(math.ceil(tj / (p_med * cos_d))))
    expected = np.linspace(-h1, tj - h1, n_pos, dtype=np.float64)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)  # igualdade EXATA (mesma fórmula)


def test_compute_n_pos_single_source_of_truth():
    """AC-1/AC-4 — compute_n_pos é a fonte única (positions_z usa o mesmo n_pos)."""
    from geosteering_ai.gui.services.sim_request import (
        SimRequest,
        _compute_positions_z,
        compute_n_pos,
    )

    req = SimRequest(h1=1.0, tj=13.0, p_med=0.7, dip_degs=(45.0,))
    assert len(_compute_positions_z(req)) == compute_n_pos(13.0, 0.7, 45.0)


def test_compute_n_pos_rejects_nonpositive_pmed():
    """AC-1 (robustez) — compute_n_pos levanta ValueError se p_med ≤ 0 (API pública)."""
    from geosteering_ai.gui.services.sim_request import compute_n_pos

    with pytest.raises(ValueError, match="p_med"):
        compute_n_pos(10.0, 0.0, 45.0)


# ════════════════════════════════════════════════════════════════════════════
# RF-2 (0011b) — fidelidade: SimulationService → simulate_batch real → shape H6
# ════════════════════════════════════════════════════════════════════════════
def test_simulation_run_produces_correct_h6_shape():
    """AC-3 — _run_simulation chama simulate_batch real; H6 com shape/dtype corretos.

    Lento (inclui JIT warmup do Numba). Física INTOCADA — só verifica shape/dtype/finitude.
    n_pos é DERIVADO (default h1=1, tj=10, p_med=1, dip0=0 → n_pos=10).
    """
    import numpy as np

    from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

    req = SimRequest(frequencies_hz=(20000.0,), n_models=2, backend="numba")
    result = _run_simulation(req)
    h6 = result["H6"]
    # (n_models, nTR, nAng, n_pos, nf, 9); n_pos=10 (dip0=0, tj/p_med=10)
    assert h6.shape == (2, 1, 1, 10, 1, 9)
    assert np.iscomplexobj(h6)
    assert np.all(np.isfinite(h6.view(np.float64)))
    assert result["backend"] == "numba"


def test_run_simulation_multi_config_shape():
    """AC-3 — multi-config: 2 freqs × 2 dips × 1 TR → H6 (n, 1, 2, n_pos, 2, 9) finito.

    Prova que os eixos nf/nAng vêm das listas. Lento (Numba, cache compartilhado).
    """
    import numpy as np

    from geosteering_ai.gui.services.sim_request import (
        SimRequest,
        _run_simulation,
        compute_n_pos,
    )

    req = SimRequest(
        frequencies_hz=(20000.0, 40000.0),
        dip_degs=(0.0, 30.0),
        tr_spacings_m=(1.0,),
        n_models=2,
        backend="numba",
    )
    result = _run_simulation(req)
    h6 = result["H6"]
    n_pos = compute_n_pos(req.tj, req.p_med, req.dip_degs[0])  # dip0=0 → 10
    assert h6.shape == (2, 1, 2, n_pos, 2, 9)  # nAng=2 (dips), nf=2 (freqs)
    assert np.all(np.isfinite(h6.view(np.float64)))


def test_build_batch_outputs_valid_shapes():
    """AC-3 — _build_batch produz shapes/ranges físicos válidos (puro, rápido).

    positions_z agora vem da convenção Fortran (n_pos derivado), não mais (50,).
    """
    import numpy as np

    from geosteering_ai.gui.services.sim_request import (
        SimRequest,
        _build_batch,
        compute_n_pos,
    )

    req = SimRequest(n_models=3)
    rho_h, rho_v, esp, positions_z = _build_batch(req)
    assert rho_h.shape == (3, 3) and rho_v.shape == (3, 3)  # (n_models, n_layers)
    assert esp.shape == (3, 1)  # n_layers - 2 internas
    assert positions_z.shape == (compute_n_pos(req.tj, req.p_med, req.dip_degs[0]),)
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


@pytest.mark.gui
def test_service_threads_bounded_across_runs(qtbot):
    """Revisão 0011a #10 — BaseService NÃO acumula refs de threads mortas.

    Pruning em ``_run_async`` (antes de adicionar) + nos slots ``_on_worker_*``
    (após a entrega) mantém ``_threads`` enxuto através de N simulações sequenciais.
    is_busy() é predicado PURO (não muta _threads) — pruning ali soltaria um
    worker com ``finished`` em-voo (resultado perdido).
    """
    from geosteering_ai.gui.services.base import BaseService

    service = BaseService()
    for _ in range(6):
        service._run_async(lambda: 1)
        qtbot.waitUntil(lambda: not service.is_busy(), timeout=5000)
    # Não acumulou: ≤ 2 (a última + uma possível pendente de prune), NÃO 6.
    qtbot.waitUntil(lambda: len(service._threads) <= 2, timeout=5000)
    assert len(service._threads) <= 2


# ════════════════════════════════════════════════════════════════════════════
# RF-5 — end-to-end real (perspectiva → run → result_ready) (qtbot)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_perspective_builds_view_under_main_window(qtbot):
    """AC-5 — a SimulationPerspective constrói View+VM sob SM_MainWindow (lazy)."""
    from apps.sim_manager.main_window import SM_MainWindow
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    win = SM_MainWindow(AppContext(app_name="teste 0011b"))
    qtbot.addWidget(win)
    win.add_perspective(SimulationPerspective())
    assert win.perspective_count == 1
    assert win.windowTitle() == "teste 0011b"


@pytest.mark.gui
def test_end_to_end_real_simulation(qtbot):
    """AC-3/AC-5 — VM real + Service real: run() → (async) → result_ready com H6.

    O caminho COMPLETO da pilha MVVM: VM puro → SimulationService → Worker (thread)
    → simulate_batch → finished (VMSignal na main thread) → result_ready. Lento
    (Numba), mas é a PROVA do walking skeleton. n_pos derivado (dip0=0 → 10).
    """
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
    from geosteering_ai.gui.services import SimulationService

    # Service explícito (mesmo que a perspectiva injeta) → ref p/ JOIN no teardown.
    service = SimulationService()
    vm = SimulationViewModel(service=service)
    rec: list = []
    vm.result_ready.connect(rec.append)
    vm.frequencies = (20000.0,)
    vm.n_models = 2
    # Params PEQUENOS explícitos (não depende dos defaults pesados de produção —
    # item 3 mudou tj/p_med p/ 120/0.2 ⇒ n_pos=600, e n_layers até 31): tj=10/
    # p_med=1 ⇒ n_pos=10; n_layers_fixed=3 ⇒ viável (3 cam.) e rápido.
    vm.tj = 10.0
    vm.p_med = 1.0
    vm.n_layers_fixed = 3
    vm.run()
    # spin o event loop até concluir (marshaling worker→main via QueuedConnection).
    # 60s cobre o JIT warmup do Numba + simulate_batch no batch pequeno (skeleton).
    qtbot.waitUntil(lambda: vm.status in ("done", "error"), timeout=60000)
    assert vm.status == "done", f"sim não concluiu: {vm.last_result}"
    assert rec and rec[0]["H6"].shape == (2, 1, 1, 10, 1, 9)
    # JOIN da worker thread antes do teardown (evita 'QThread destroyed while running').
    qtbot.waitUntil(lambda: not service.is_busy(), timeout=10000)


# ════════════════════════════════════════════════════════════════════════════
# RNF-2/RNF-4 — fronteira de import
# ════════════════════════════════════════════════════════════════════════════
def test_viewmodel_importable_without_qt():
    """AC-5 — importar o ViewModel NÃO importa PyQt6/PySide6 (Princípio X)."""
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
    """AC-5 — o core (simulation/…) não importa gui/apps (fronteira)."""
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
