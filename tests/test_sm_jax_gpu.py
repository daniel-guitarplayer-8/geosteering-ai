# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_jax_gpu.py                                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0012-sm-jax-gpu                                            ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app MVVM — JAX GPU em subprocesso (TLS-safe)            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-06                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre a spec 0012: roteamento de backend (numba in-thread / jax-auto    ║
# ║    subprocesso), VM backend (validação), paridade JAX-GPU vs Numba <1e-12  ║
# ║    (gated GPU), e a ISOLAÇÃO TLS-safe (processo da GUI não importa JAX).    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0012 — JAX GPU no SM MVVM (subprocesso TLS-safe + seletor backend)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _gpu_available() -> bool:
    """``True`` se há GPU JAX (gate dos testes que rodam JAX de fato)."""
    try:
        from geosteering_ai.simulation.dispatch import _jax_gpu_available

        return bool(_jax_gpu_available())
    except Exception:
        return False


_GPU = _gpu_available()
_needs_gpu = pytest.mark.skipif(not _GPU, reason="requer GPU JAX (RTX A6000 local)")


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


def _suicide() -> None:
    """Mata o subprocesso SEM levantar exceção Python (simula OOM/segfault/driver).

    ``os._exit`` encerra o intérprete imediatamente — o pool não recebe nem
    resultado nem exceção marshalada → ``BrokenProcessPool``. Módulo-nível p/ ser
    picklável pelo spawn.
    """
    os._exit(1)


# ════════════════════════════════════════════════════════════════════════════
# AC-5 — VM backend (validação + run monta SimRequest) — PURO, sem GPU
# ════════════════════════════════════════════════════════════════════════════
def test_vm_backend_default_and_run():
    """AC-5 — default 'numba'; run() passa o backend ao SimRequest."""
    vm = _make_sim_vm()
    assert vm.backend == "numba"
    vm.backend = "jax"
    vm.run()
    assert vm._service.requests[-1].backend == "jax"


def test_vm_backend_invalid_rejected():
    """AC-5 — backend fora de {numba,jax,auto} é reprovado por validate()."""
    vm = _make_sim_vm()
    vm.backend = "cuda"  # inválido
    vm.run()
    assert vm._service.requests == []  # NÃO despachou
    assert vm.status == "error"
    assert any("Backend" in e for e in vm.last_result["errors"])
    for ok in ("numba", "jax", "auto"):
        vm2 = _make_sim_vm()
        vm2.backend = ok
        assert vm2.validate() == []


def test_session_roundtrip_includes_backend():
    """AC-5 — o backend persiste no .session."""
    import json

    vm = _make_sim_vm()
    vm.backend = "auto"
    blob = json.dumps(vm.to_session_dict())
    vm2 = _make_sim_vm()
    vm2.load_session_dict(json.loads(blob))
    assert vm2.backend == "auto"


def test_load_session_clamps_invalid_backend():
    """Review 0012 #3 — .session corrompido c/ backend inválido cai p/ 'numba' no load.

    Um .session editado à mão pode trazer ``backend="cuda"``. Carregá-lo às cegas
    deixaria o combo (só {numba,jax,auto}) dessincronizado do VM e só estouraria no
    run(). O load deve sanear → "numba" (estado válido + combo sincronizado).
    """
    vm = _make_sim_vm()
    vm.load_session_dict({"backend": "cuda", "n_models": 2})
    assert vm.backend == "numba"  # saneado no load (não "cuda")
    assert vm.validate() == []  # estado válido (não reprovado no run)


# ════════════════════════════════════════════════════════════════════════════
# AC-2 — roteamento: numba in-thread; jax/auto subprocesso
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_service_routes_backend(qtbot, monkeypatch):
    """AC-2 — numba → _run_async (in-thread); jax/auto → _run_in_subprocess."""
    from geosteering_ai.gui.services.sim_request import SimRequest
    from geosteering_ai.gui.services.simulation_service import SimulationService

    svc = SimulationService()
    calls: list = []
    sub_kwargs: list = []
    monkeypatch.setattr(svc, "_run_async", lambda fn, *a, **k: calls.append("async"))
    monkeypatch.setattr(
        svc,
        "_run_in_subprocess",
        lambda fn, *a, **k: (calls.append("sub"), sub_kwargs.append(k)),
    )
    svc.run(SimRequest(backend="numba"))
    svc.run(SimRequest(backend="jax"))
    svc.run(SimRequest(backend="auto"))
    assert calls == ["async", "sub", "sub"]
    # PR worker persistente: jax/auto rodam no subprocesso PERSISTENTE (persistent=True).
    assert [k.get("persistent") for k in sub_kwargs] == [True, True]


# ════════════════════════════════════════════════════════════════════════════
# AC-4 — paridade JAX-GPU vs Numba <1e-12 (gated GPU)
# ════════════════════════════════════════════════════════════════════════════
@_needs_gpu
def test_parity_jax_gpu_vs_numba():
    """AC-4 — mesmos modelos: |H6_jax − H6_numba| < 1e-12 (física idêntica)."""
    import numpy as np

    from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

    base = dict(geology_mode="fixed", n_models=2, frequencies_hz=(20000.0,), tj=10.0)
    h_n = _run_simulation(SimRequest(backend="numba", **base))["H6"]
    h_j = _run_simulation(SimRequest(backend="jax", **base))["H6"]
    assert h_j.shape == h_n.shape
    assert np.max(np.abs(h_n - h_j)) < 1e-12


@_needs_gpu
def test_pool_run_marshals_jax_in_subprocess():
    """AC-1/AC-3 — _pool_run roda _run_simulation(jax) num subprocesso spawn e retorna H6."""
    import numpy as np

    from geosteering_ai.gui.services.base import _pool_run
    from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

    req = SimRequest(
        backend="jax", geology_mode="fixed", n_models=2, frequencies_hz=(20000.0,)
    )
    result = _pool_run(_run_simulation, (req,), {})
    assert result["H6"].shape == (2, 1, 1, 10, 1, 9)
    assert np.all(np.isfinite(result["H6"].view(np.float64)))


# ════════════════════════════════════════════════════════════════════════════
# AC-1 — a simulação roda num SUBPROCESSO (CUDA isolado; QThread não inita CUDA)
# ════════════════════════════════════════════════════════════════════════════
def test_pool_run_executes_in_subprocess():
    """AC-1 — _pool_run roda o callable num PROCESSO separado (pid ≠ pid da GUI).

    É o cerne da isolação TLS-safe: a computação JAX (que inicializa CUDA) corre
    num subprocesso spawn, NÃO na QThread da GUI (onde o init de CUDA estouraria o
    TLS / ``_dl_allocate_tls_init``). Não precisa de GPU (testa só o mecanismo).
    """
    import os

    from geosteering_ai.gui.services.base import _pool_run

    child_pid = _pool_run(os.getpid, (), {})
    assert isinstance(child_pid, int) and child_pid > 0
    assert child_pid != os.getpid()  # rodou noutro processo (spawn)


def test_pool_run_wraps_broken_pool_with_clear_message():
    """Review 0012 #2 — morte abrupta do subprocesso → RuntimeError acionável.

    Se o worker morre sem devolver exceção (OOM killer/GPU OOM/crash do driver
    CUDA/segfault), o stdlib levanta ``BrokenProcessPool`` com msg críptica
    ("terminated abruptly"). ``_pool_run`` deve re-levantar como ``RuntimeError``
    com causa acionável (mencionando memória/CUDA + sugestão de mitigação), que o
    Worker capta → error VMSignal → UI mostra algo útil.
    """
    from geosteering_ai.gui.services.base import _pool_run

    with pytest.raises(RuntimeError) as exc_info:
        _pool_run(_suicide, (), {})
    msg = str(exc_info.value).lower()
    assert "subprocesso" in msg
    assert "numba" in msg  # sugere a mitigação (cair p/ numba)


@_needs_gpu
def test_jax_sim_e2e_via_service_no_tls_crash():
    """AC-1/AC-3 — Service.run(jax) → (subprocesso) → finished com H6 correto, SEM crash.

    Subprocesso LIMPO (QApplication headless): se o JAX rodasse na QThread, o init
    de CUDA estouraria o TLS (crash). Concluir com H6 correto PROVA que a sim correu
    no subprocesso (a isolação funciona) e que o marshaling Worker→pool→VMSignal
    entrega o resultado na main thread. (jax-MÓDULO pode estar em sys.modules via
    opt_einsum/TF — irrelevante; o que importa é não INICIAR CUDA na QThread.)
    """
    code = (
        "import os, time\n"
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        "from geosteering_ai.gui.qt_compat import QtWidgets\n"
        "app = QtWidgets.QApplication([])\n"
        "from geosteering_ai.gui.services.simulation_service import SimulationService\n"
        "from geosteering_ai.gui.services.sim_request import SimRequest\n"
        "svc = SimulationService()\n"
        "done = {}\n"
        "svc.finished.connect(lambda r: done.setdefault('r', r))\n"
        "svc.error.connect(lambda m: done.setdefault('err', m))\n"
        "svc.run(SimRequest(backend='jax', geology_mode='fixed', n_models=2,\n"
        "                   frequencies_hz=(20000.0,)))\n"
        "t0 = time.time()\n"
        "while not done and time.time() - t0 < 240:\n"
        "    app.processEvents(); time.sleep(0.05)\n"
        "assert 'r' in done, f'sim não concluiu (erro? {done})'\n"
        "assert done['r']['H6'].shape == (2, 1, 1, 10, 1, 9), done['r']['H6'].shape\n"
        "print('TLS_SAFE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2500:]
    assert "TLS_SAFE_OK" in proc.stdout
