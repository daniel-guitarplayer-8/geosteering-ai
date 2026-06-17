# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_jax_persistent.py                                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  PR — Worker JAX PERSISTENTE no SM (fecha a disparidade CLI×SM)            ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-17                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    O SM rodava cada sim jax/auto num subprocesso EFÊMERO (criado/destruído ║
# ║    por run) → re-pagava init CUDA + reload do cache XLA (~17 s/run). O CLI  ║
# ║    é rápido por rodar JAX in-process. O pool PERSISTENTE em gui.services.   ║
# ║    base mantém UM subprocesso spawn vivo entre runs (2º+ run ~12 s vs ~29). ║
# ║    Estes testes fixam o ciclo de vida (reuso, self-heal, idempotência,     ║
# ║    teardown) — todos CPU-safe (os.getpid no filho; sem GPU/JAX).           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do worker JAX persistente do SM (pool singleton + self-heal + teardown)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import geosteering_ai.gui.services.base as base_mod  # noqa: E402
from geosteering_ai.gui.services.base import (  # noqa: E402
    _acquire_jax_pool,
    _pool_run_persistent,
    release_jax_pool,
)

_BASE_PY = Path(base_mod.__file__)


def _suicide() -> None:
    """Mata o filho SEM exceção Python (OOM/segfault/driver) → BrokenProcessPool.

    Módulo-nível p/ ser picklável pelo spawn. ``os._exit`` não dá chance ao pool de
    marshalar resultado/exceção → o ``future.result()`` levanta ``BrokenProcessPool``.
    """
    os._exit(1)


@pytest.fixture(autouse=True)
def _clean_jax_pool():
    """Garante pool limpo antes/depois de cada teste (sem leak nem ordem-dependência)."""
    release_jax_pool()
    yield
    release_jax_pool()


# ════════════════════════════════════════════════════════════════════════════
# Reuso do subprocesso persistente (o ganho central)
# ════════════════════════════════════════════════════════════════════════════
def test_pool_run_persistent_reuses_same_child_pid():
    """2 chamadas reusam o MESMO subprocesso (pid idêntico ≠ pai) — prova a persistência."""
    pid1 = _pool_run_persistent(os.getpid, (), {})
    pid2 = _pool_run_persistent(os.getpid, (), {})
    assert pid1 == pid2  # MESMO filho reutilizado (não recriado por run)
    assert pid1 != os.getpid()  # rodou noutro processo (spawn)


def test_jax_pool_max_workers_one():
    """O pool persistente tem exatamente 1 worker (sim serial; 1 GPU)."""
    pool = _acquire_jax_pool()
    assert pool._max_workers == 1


# ════════════════════════════════════════════════════════════════════════════
# Self-heal: um crash do filho NÃO pode envenenar todos os runs futuros
# ════════════════════════════════════════════════════════════════════════════
def test_pool_run_persistent_self_heals_after_broken_pool():
    """Filho morto → RuntimeError acionável; o PRÓXIMO run recria o pool (pid FRESCO)."""
    pid0 = _pool_run_persistent(os.getpid, (), {})
    with pytest.raises(RuntimeError) as exc:
        _pool_run_persistent(
            _suicide, (), {}
        )  # mata o filho 2× (self-heal re-tenta 1×)
    msg = str(exc.value).lower()
    assert "subprocesso" in msg and "numba" in msg  # mensagem acionável preservada
    # O pool envenenado foi descartado → o próximo run cria um filho NOVO e funciona.
    pid_new = _pool_run_persistent(os.getpid, (), {})
    assert pid_new != os.getpid() and pid_new != pid0  # processo fresco (recriado)


def test_pool_run_persistent_propagates_fn_exception_without_retry():
    """Exceção do PRÓPRIO fn (não morte do filho) propaga — NÃO é mascarada/re-tentada.

    Usa ``_domain_error`` (callable módulo-nível, picklável). Só ``BrokenProcessPool``/
    ``CancelledError`` (falhas de infra) são capturados; uma exceção normal de ``fn``
    sobe via ``result()``.
    """
    with pytest.raises(ValueError, match="dom"):
        _pool_run_persistent(_domain_error, (), {})


# ── Revisão adversarial: release externo (encerramento) vs crash do filho ─────
# Pools FALSOS (determinísticos, sem GPU) exercitam o roteamento exato de exceções.
class _FakeFuture:
    def __init__(self, exc):
        self._exc = exc

    def result(self):
        raise self._exc


class _FakePool:
    """Pool falso cujo future.result() levanta ``exc``; conta submits + shutdowns."""

    def __init__(self, exc):
        self._exc = exc
        self.submits = 0
        self.shutdowns = 0

    def submit(self, fn, *a, **k):
        self.submits += 1
        return _FakeFuture(self._exc)

    def shutdown(self, **k):
        self.shutdowns += 1


def test_pool_run_persistent_external_release_aborts_clean(monkeypatch):
    """release externo (global ≠ nosso pool) → CancelledError → aborta limpo (sem retry).

    Mensagem clara (_SHUTDOWN_CANCEL_MSG), NÃO o erro vazio que escapava antes nem o
    _SUBPROCESS_DIED_MSG. E NÃO re-roda a sim durante o encerramento (1 submit só).
    """
    from concurrent.futures import CancelledError

    fake = _FakePool(CancelledError())
    monkeypatch.setattr(base_mod, "_acquire_jax_pool", lambda: fake)
    monkeypatch.setattr(base_mod, "_JAX_POOL", None)  # global ≠ fake → release externo
    with pytest.raises(RuntimeError, match="cancelad"):
        _pool_run_persistent(os.getpid, (), {})
    assert fake.submits == 1  # abortou — NÃO re-tentou/recriou pool no shutdown


def test_pool_run_persistent_submit_after_shutdown_aborts_clean(monkeypatch):
    """release externo entre acquire e submit (RuntimeError no submit) → aborta limpo."""

    class _ShutdownPool:
        def __init__(self):
            self.submits = 0

        def submit(self, fn, *a, **k):
            self.submits += 1
            raise RuntimeError("cannot schedule new futures after shutdown")

        def shutdown(self, **k):
            pass

    fake = _ShutdownPool()
    monkeypatch.setattr(base_mod, "_acquire_jax_pool", lambda: fake)
    monkeypatch.setattr(base_mod, "_JAX_POOL", None)  # global ≠ fake → release externo
    with pytest.raises(RuntimeError, match="cancelad"):
        _pool_run_persistent(os.getpid, (), {})
    assert fake.submits == 1  # NÃO recriou pool durante o shutdown


def test_pool_run_persistent_crash_retries_then_dies(monkeypatch):
    """crash do NOSSO pool (global == pool) → self-heal re-tenta 1×, depois _SUBPROCESS_DIED."""
    from concurrent.futures.process import BrokenProcessPool

    made: list = []

    def _fake_acquire():
        p = _FakePool(BrokenProcessPool("dead"))
        base_mod._JAX_POOL = p  # global É este pool → crash nosso → external=False
        made.append(p)
        return p

    monkeypatch.setattr(base_mod, "_acquire_jax_pool", _fake_acquire)
    with pytest.raises(RuntimeError, match="subprocesso"):
        _pool_run_persistent(os.getpid, (), {})
    assert len(made) == 2  # re-tentou exatamente 1× (2 pools criados)
    assert all(p.shutdowns >= 1 for p in made)  # cada pool morto foi encerrado


# ════════════════════════════════════════════════════════════════════════════
# Teardown / lifecycle
# ════════════════════════════════════════════════════════════════════════════
def test_release_jax_pool_idempotent_and_recreate():
    """release sem pool = no-op; após acquire+release o singleton zera; recria depois."""
    release_jax_pool()  # no-op seguro (sem pool ativo)
    assert base_mod._JAX_POOL is None
    _acquire_jax_pool()
    assert base_mod._JAX_POOL is not None
    release_jax_pool()
    assert base_mod._JAX_POOL is None  # zerado
    pid = _pool_run_persistent(os.getpid, (), {})  # recria sob demanda
    assert pid != os.getpid()


def test_atexit_registered_once(monkeypatch):
    """atexit.register(release_jax_pool) é chamado UMA vez (na 1ª criação), não por run."""
    import atexit as _atexit

    calls: list = []
    monkeypatch.setattr(_atexit, "register", lambda fn, *a, **k: calls.append(fn))
    # Reset do guard p/ exercitar o caminho de 1ª criação de forma determinística.
    monkeypatch.setattr(base_mod, "_JAX_POOL_ATEXIT_DONE", False)
    release_jax_pool()
    _acquire_jax_pool()  # cria → registra
    _acquire_jax_pool()  # reusa → NÃO registra de novo
    assert calls.count(release_jax_pool) == 1


# ════════════════════════════════════════════════════════════════════════════
# TLS-safety: base.py NUNCA importa jax em escopo de módulo (só o filho importa)
# ════════════════════════════════════════════════════════════════════════════
def test_base_module_has_no_top_level_jax_import():
    """Guarda estática: ``base.py`` não tem ``import jax`` / ``from jax`` top-level.

    O invariante TLS é que o PROCESSO da GUI nunca inicializa CUDA — só o subprocesso
    filho importa JAX (via fn=_run_simulation). Asserir 'jax' ∉ sys.modules seria
    frágil (o TF transitivamente importa o módulo jax). A guarda real é: este módulo,
    importado pela GUI, não puxa jax por si — checado na fonte (top-level, col 0).
    """
    src = _BASE_PY.read_text(encoding="utf-8")
    offenders = [
        ln
        for ln in src.splitlines()
        if re.match(r"^(import jax|from jax)\b", ln)  # col 0 = top-level
    ]
    assert offenders == [], f"import jax top-level em base.py: {offenders}"


def _domain_error() -> None:
    """Callable módulo-nível (picklável) que levanta ValueError — p/ o teste de propagação."""
    raise ValueError("erro de domínio do fn")


# ════════════════════════════════════════════════════════════════════════════
# Teardown fiado na janela / app (não vaza o subprocesso GPU)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_sm_closeEvent_calls_release(qtbot, monkeypatch):
    """Fechar a SM_MainWindow chama release_jax_pool (libera o worker GPU persistente)."""
    from apps.sim_manager.main_window import SM_MainWindow
    from geosteering_ai.gui.shell.context import AppContext

    called: list = []
    monkeypatch.setattr(base_mod, "release_jax_pool", lambda: called.append(True))
    win = SM_MainWindow(AppContext(app_name="t"))
    qtbot.addWidget(win)
    win.close()  # dispara closeEvent → release (import local pega o spy)
    assert called == [True]


def test_app_wires_aboutToQuit_release():
    """Guarda estática: app.py fia release_jax_pool no aboutToQuit (defense-in-depth).

    Testar o disparo real exigiria rodar ``main()`` (que bloqueia em ``exec()``); a
    guarda de fonte impede a remoção acidental do wiring de teardown do quit.
    """
    import apps.sim_manager.app as app_mod

    src = Path(app_mod.__file__).read_text(encoding="utf-8")
    assert "aboutToQuit.connect" in src
    assert "release_jax_pool" in src
