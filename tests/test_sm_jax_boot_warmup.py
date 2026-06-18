# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_jax_boot_warmup.py                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  PR — Boot warmup do JAX (op 3) + preferência jax_boot_warmup             ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-17                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Pré-aquecer (opt-in) o worker JAX persistente no boot, submetendo a     ║
# ║    forma canônica ao MESMO pool de base.py. Estes testes fixam: opt-in     ║
# ║    default False; no-op sem jax; guard via find_spec (sem importar jax na   ║
# ║    GUI — checado estaticamente); submissão ao pool persistente; falha       ║
# ║    silenciosa; round-trip da preferência.                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do boot warmup do JAX no SM (opt-in) + preferência jax_boot_warmup."""

from __future__ import annotations

import importlib.util
import os
import re
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import apps.sim_manager.boot_warmup as boot_mod  # noqa: E402
import geosteering_ai.gui.services.base as base_mod  # noqa: E402
import geosteering_ai.gui.services.jax_warmup_service as warmup_mod  # noqa: E402
from apps.sim_manager.boot_warmup import schedule_boot_warmup  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_jax_pool():
    base_mod.release_jax_pool()
    yield
    base_mod.release_jax_pool()


class _Ctx:
    """AppContext mínimo (só ``extras``) p/ estacionar o service."""

    def __init__(self) -> None:
        self.extras: dict = {}


class _Window:
    """Janela falsa com statusBar().showMessage (registra as mensagens)."""

    def __init__(self) -> None:
        self.msgs: list = []

    def statusBar(self):  # noqa: N802 — espelha a API do Qt
        win = self

        class _Bar:
            def showMessage(self, m):
                win.msgs.append(m)

        return _Bar()


# ════════════════════════════════════════════════════════════════════════════
# TLS-safety: nem boot_warmup nem jax_warmup_service importam jax top-level
# ════════════════════════════════════════════════════════════════════════════
def test_no_top_level_jax_import_in_warmup_modules():
    """Guarda estática: o JAX só é importado DENTRO do filho (não na GUI)."""
    for mod in (boot_mod, warmup_mod):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        offenders = [
            ln for ln in src.splitlines() if re.match(r"^(import jax|from jax)\b", ln)
        ]
        assert offenders == [], f"import jax top-level em {mod.__name__}: {offenders}"


# ════════════════════════════════════════════════════════════════════════════
# schedule_boot_warmup — opt-in + guards
# ════════════════════════════════════════════════════════════════════════════
def test_schedule_disabled_is_noop():
    """enabled=False → no-op (None), nada estacionado em ctx.extras."""
    ctx, win = _Ctx(), _Window()
    assert schedule_boot_warmup(ctx, win, enabled=False) is None
    assert "jax_warmup_service" not in ctx.extras
    assert win.msgs == []


def test_schedule_no_jax_is_noop(monkeypatch):
    """JAX ausente (find_spec→None) → no-op, mesmo com enabled=True (TLS-safe)."""
    monkeypatch.setattr(boot_mod.importlib.util, "find_spec", lambda name: None)
    ctx, win = _Ctx(), _Window()
    assert schedule_boot_warmup(ctx, win, enabled=True) is None
    assert "jax_warmup_service" not in ctx.extras


# ════════════════════════════════════════════════════════════════════════════
# JaxWarmupService — guard sem jax + submissão ao pool PERSISTENTE
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_warmup_service_skips_without_jax(qtbot, monkeypatch):
    """warmup() sem jax → finished({'skipped': True}) SEM criar worker/importar jax."""
    import importlib.util as _ilu

    from geosteering_ai.gui.services.jax_warmup_service import JaxWarmupService

    # warmup() faz `import importlib.util` LOCAL → patch o find_spec REAL do módulo.
    monkeypatch.setattr(_ilu, "find_spec", lambda name: None)
    svc = JaxWarmupService()
    got: list = []
    svc.finished.connect(got.append)
    svc.warmup()
    assert got and got[0].get("skipped") is True


@pytest.mark.gui
def test_warmup_service_submits_canonical_to_persistent_pool(qtbot, monkeypatch):
    """warmup() submete _warmup_in_pool ao pool PERSISTENTE de base.py (não efêmero)."""
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax ausente neste ambiente")
    from geosteering_ai.gui.services.jax_warmup_service import (
        JaxWarmupService,
        _warmup_in_pool,
    )

    submitted: list = []

    class _FakeFuture:
        def result(self):
            return {"skipped": False, "buckets_warmed": 3}

    class _FakePool:
        def submit(self, fn, *a, **k):
            submitted.append(fn)
            return _FakeFuture()

    monkeypatch.setattr(base_mod, "_acquire_jax_pool", lambda: _FakePool())
    svc = JaxWarmupService()
    got: list = []
    svc.finished.connect(got.append)
    svc.warmup()
    t0 = time.time()
    while not got and time.time() - t0 < 10:
        qtbot.wait(20)  # processa eventos até o VMSignal cruzar p/ a main thread
    assert got and got[0].get("buckets_warmed") == 3
    assert submitted == [
        _warmup_in_pool
    ]  # submeteu a forma canônica AO pool persistente


@pytest.mark.gui
def test_warmup_failure_does_not_crash(qtbot, monkeypatch):
    """Falha no warmup → error VMSignal tratado (status limpo); app não cai."""
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax ausente neste ambiente")

    class _BoomPool:
        def submit(self, fn, *a, **k):
            raise RuntimeError("boom no warmup")

    monkeypatch.setattr(base_mod, "_acquire_jax_pool", lambda: _BoomPool())
    ctx, win = _Ctx(), _Window()
    svc = schedule_boot_warmup(ctx, win, enabled=True)
    assert svc is not None  # agendou (jax presente)
    t0 = time.time()
    while win.msgs[-1:] != [""] and time.time() - t0 < 10:
        qtbot.wait(20)  # espera o handler de erro limpar o status
    assert win.msgs[0].startswith("Aquecendo")  # mostrou o início
    assert win.msgs[-1] == ""  # limpou em silêncio (erro não alarma)


# ════════════════════════════════════════════════════════════════════════════
# Preferência jax_boot_warmup — default False + round-trip
# ════════════════════════════════════════════════════════════════════════════
def test_pref_default_jax_boot_warmup_false():
    """O default da preferência é False (opt-in)."""
    from apps.sim_manager.perspectives.preferences.service import (
        DEFAULT_PREFERENCES,
        PreferencesService,
    )

    assert DEFAULT_PREFERENCES["jax_boot_warmup"] is False
    assert PreferencesService().defaults()["jax_boot_warmup"] is False


def test_pref_vm_roundtrip_jax_boot_warmup():
    """O VM de Preferências persiste/recarrega jax_boot_warmup (True sobrevive)."""
    from apps.sim_manager.perspectives.preferences.service import PreferencesService
    from apps.sim_manager.perspectives.preferences.viewmodel import (
        PreferencesViewModel,
    )

    vm = PreferencesViewModel(service=PreferencesService())
    assert vm.jax_boot_warmup is False  # default
    vm.jax_boot_warmup = True
    d = vm.to_session_dict()
    assert d["jax_boot_warmup"] is True
    vm2 = PreferencesViewModel(service=PreferencesService())
    vm2.load_session_dict(d)
    assert vm2.jax_boot_warmup is True


def test_app_wires_boot_warmup():
    """Guarda estática: app.py chama schedule_boot_warmup (lê a pref jax_boot_warmup)."""
    import apps.sim_manager.app as app_mod

    src = Path(app_mod.__file__).read_text(encoding="utf-8")
    assert "schedule_boot_warmup" in src and "jax_boot_warmup" in src


# ── Revisão adversarial: coerção de bool robusta + guard anti-double-schedule ──
def test_pref_load_string_false_does_not_enable():
    """Revisão #3 (CONFIRMED): '.session' com "false" (string) NÃO liga o warmup.

    ``bool("false")`` seria ``True`` (string não-vazia) — _safe_bool corrige. Tipo
    inesperado cai p/ o valor corrente (não corrompe).
    """
    from apps.sim_manager.perspectives.preferences.service import PreferencesService
    from apps.sim_manager.perspectives.preferences.viewmodel import (
        PreferencesViewModel,
    )

    vm = PreferencesViewModel(service=PreferencesService())
    vm.load_session_dict({"jax_boot_warmup": "false"})
    assert vm.jax_boot_warmup is False  # "false" NÃO liga (bug do bool() cru evitado)
    vm.load_session_dict({"jax_boot_warmup": "true"})
    assert vm.jax_boot_warmup is True
    vm.load_session_dict({"jax_boot_warmup": {"x": 1}})  # tipo inesperado
    assert vm.jax_boot_warmup is True  # manteve o último válido (não corrompeu)


def test_schedule_reuses_busy_existing_service(monkeypatch):
    """Revisão #2/#4: 2ª chamada com warmup em voo REUSA o service (não sobrescreve)."""
    monkeypatch.setattr(boot_mod.importlib.util, "find_spec", lambda name: object())

    class _BusySvc:
        def is_busy(self):
            return True

    busy = _BusySvc()
    ctx, win = _Ctx(), _Window()
    ctx.extras["jax_warmup_service"] = busy
    result = schedule_boot_warmup(ctx, win, enabled=True)
    assert result is busy  # reusou o existente em voo
    assert (
        ctx.extras["jax_warmup_service"] is busy
    )  # NÃO dropou a ref (sem GC da QThread)
