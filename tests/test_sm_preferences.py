# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_preferences.py                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Fatia 6e — perspectiva Preferências (SM MVVM)                            ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM MVVM — Preferências (tema, paths, backend, cache LRU)    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-15                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre a Fatia 6e: serviço de persistência atômica (defaults, merge,     ║
# ║    degradação graciosa), ViewModel PURO (sinais, clamps, round-trip de      ║
# ║    sessão, save/load/restore) e a View Qt (widgets, sync, save→disco,       ║
# ║    restore) + a perspectiva plugável (id/order + build_view/viewmodel).    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da Fatia 6e — Preferências (serviço + ViewModel PURO + View Qt)."""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apps.sim_manager.perspectives.preferences.service import (  # noqa: E402
    DEFAULT_PREFERENCES,
    PreferencesService,
)
from apps.sim_manager.perspectives.preferences.viewmodel import (  # noqa: E402
    PreferencesViewModel,
)


# ════════════════════════════════════════════════════════════════════════════
# PreferencesService — persistência atômica + degradação graciosa
# ════════════════════════════════════════════════════════════════════════════
def test_service_defaults_are_deep_copied():
    svc = PreferencesService(path="/tmp/none.json")
    a = svc.defaults()
    a["paths"]["output_dir"] = "MUTATED"
    b = svc.defaults()
    assert b["paths"]["output_dir"] == ""  # cópia profunda — não vaza mutação
    assert DEFAULT_PREFERENCES["paths"]["output_dir"] == ""


def test_service_load_missing_file_returns_defaults(tmp_path):
    svc = PreferencesService(path=tmp_path / "nao_existe.json")
    assert svc.load() == DEFAULT_PREFERENCES  # 1º boot → defaults


def test_service_load_corrupt_file_returns_defaults(tmp_path):
    p = tmp_path / "p.json"
    p.write_text("{ isto não é json", encoding="utf-8")
    svc = PreferencesService(path=p)
    assert svc.load() == DEFAULT_PREFERENCES
    assert svc.path == p  # sanity


def test_service_load_non_dict_returns_defaults(tmp_path):
    p = tmp_path / "p.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert PreferencesService(path=p).load() == DEFAULT_PREFERENCES


def test_service_round_trip_and_partial_merge(tmp_path):
    p = tmp_path / "p.json"
    svc = PreferencesService(path=p)
    svc.save({"cache_max_mb": 512, "paths": {"output_dir": "/tmp/out"}})
    assert p.is_file()
    loaded = svc.load()
    assert loaded["cache_max_mb"] == 512
    assert loaded["paths"]["output_dir"] == "/tmp/out"
    # paths ausentes mantêm o default (merge num nível adicional)
    assert loaded["paths"]["tatu_binary"] == ""
    # chaves não fornecidas caem para o default
    assert loaded["theme"] == DEFAULT_PREFERENCES["theme"]


def test_service_preserves_unknown_keys(tmp_path):
    """Forward-compat: chaves extras desconhecidas sobrevivem ao round-trip."""
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"future_flag": True}), encoding="utf-8")
    assert PreferencesService(path=p).load().get("future_flag") is True


# ════════════════════════════════════════════════════════════════════════════
# PreferencesViewModel — sinais, clamps, round-trip, save/load/restore (PURO)
# ════════════════════════════════════════════════════════════════════════════
def _vm(tmp_path):
    return PreferencesViewModel(service=PreferencesService(path=tmp_path / "p.json"))


def test_vm_signals_fire_on_change(tmp_path):
    vm = _vm(tmp_path)
    themes, backends, caches, changes = [], [], [], []
    vm.theme_changed.connect(themes.append)
    vm.plot_backend_changed.connect(backends.append)
    vm.cache_changed.connect(lambda mb, n: caches.append((mb, n)))
    vm.changed.connect(lambda name, value: changes.append(name))
    vm.theme = "antigravity_light"  # != default → dispara theme_changed
    vm.plot_backend = "pyqtgraph"
    vm.cache_max_mb = 512
    vm.cache_max_snapshots = 20
    vm.set_path("output_dir", "/tmp/out")
    assert themes == ["antigravity_light"]
    assert backends == ["pyqtgraph"]
    assert caches == [(512, 12), (512, 20)]
    assert "_paths" in changes  # set_path emite changed
    # idempotência: setar o MESMO valor NÃO re-emite (dedupe via _set/_equal)
    vm.theme = "antigravity_light"
    vm.plot_backend = "pyqtgraph"
    vm.cache_max_mb = 512
    vm.cache_max_snapshots = 20
    vm.set_path("output_dir", "/tmp/out")
    assert themes == ["antigravity_light"]
    assert backends == ["pyqtgraph"]
    assert caches == [(512, 12), (512, 20)]  # nenhum re-emit de cache


def test_vm_clamps_cache_lower_bounds(tmp_path):
    vm = _vm(tmp_path)
    vm.cache_max_mb = 1  # < 32 → clamp 32
    vm.cache_max_snapshots = 0  # < 1 → clamp 1
    assert vm.cache_max_mb == 32
    assert vm.cache_max_snapshots == 1


def test_vm_set_path_validates_key(tmp_path):
    vm = _vm(tmp_path)
    vm.set_path("output_dir", "/tmp/x")
    assert vm.get_path("output_dir") == "/tmp/x"
    with pytest.raises(ValueError):
        vm.set_path("chave_invalida", "/tmp/y")


def test_vm_session_round_trip(tmp_path):
    vm = _vm(tmp_path)
    vm.plot_backend = "plotly"
    vm.cache_max_mb = 1024
    vm.cache_max_snapshots = 30
    vm.set_path("tatu_binary", "/opt/tatu.x")
    d = vm.to_session_dict()
    vm2 = _vm(tmp_path)
    vm2.load_session_dict(d)
    assert vm2.plot_backend == "plotly"
    assert vm2.cache_max_mb == 1024
    assert vm2.cache_max_snapshots == 30
    assert vm2.get_path("tatu_binary") == "/opt/tatu.x"


def test_vm_load_session_dict_partial_keeps_current(tmp_path):
    vm = _vm(tmp_path)
    vm.cache_max_mb = 999
    vm.load_session_dict({"plot_backend": "vispy"})  # sem cache → mantém
    assert vm.plot_backend == "vispy"
    assert vm.cache_max_mb == 999


def test_vm_load_session_dict_partial_paths_merges_per_key(tmp_path):
    """paths dict parcial: a chave dada muda; as omitidas são preservadas."""
    vm = _vm(tmp_path)
    vm.set_path("output_dir", "/keep/out")
    vm.set_path("tatu_binary", "/keep/tatu")
    vm.load_session_dict({"paths": {"tatu_binary": "/novo/tatu"}})  # só 1 chave
    assert vm.get_path("tatu_binary") == "/novo/tatu"  # atualizada
    assert vm.get_path("output_dir") == "/keep/out"  # preservada (não some)


def test_vm_load_wrong_typed_numeric_degrades_gracefully(tmp_path):
    """JSON válido com numérico de tipo errado NÃO crasha o boot (degrada p/ default).

    Regressão (revisão adversarial 6e): ``int("512MB")``/``int(None)`` levantava
    e quebrava o build da perspectiva (= boot do SM ao abrir a aba Preferências).
    """
    p = tmp_path / "p.json"
    p.write_text(
        json.dumps({"cache_max_mb": "512MB", "cache_max_snapshots": None}),
        encoding="utf-8",
    )
    svc = PreferencesService(path=p)
    vm = PreferencesViewModel(service=svc)
    vm.load()  # não pode levantar
    assert vm.cache_max_mb == DEFAULT_PREFERENCES["cache_max_mb"]  # 256
    assert vm.cache_max_snapshots == DEFAULT_PREFERENCES["cache_max_snapshots"]  # 12


def test_vm_restore_defaults_emits_single_consistent_cache_pair(tmp_path):
    """restore_defaults emite cache_changed UMA vez, com o par final consistente."""
    vm = _vm(tmp_path)
    vm.cache_max_mb = 4096
    vm.cache_max_snapshots = 99
    caches = []
    vm.cache_changed.connect(lambda mb, n: caches.append((mb, n)))
    vm.restore_defaults()
    assert caches == [
        (
            DEFAULT_PREFERENCES["cache_max_mb"],
            DEFAULT_PREFERENCES["cache_max_snapshots"],
        )
    ]


def test_vm_from_dict_raises_clear_error(tmp_path):
    """from_dict herdado é explicitamente NÃO suportado (exige service)."""
    with pytest.raises(NotImplementedError):
        PreferencesViewModel.from_dict({"theme": "x"})


def test_vm_save_then_reload_round_trips_via_disk(tmp_path):
    svc = PreferencesService(path=tmp_path / "p.json")
    vm = PreferencesViewModel(service=svc)
    vm.load()
    vm.cache_max_mb = 512
    vm.set_path("output_dir", "/tmp/out")
    saved = []
    vm.saved.connect(lambda: saved.append(True))
    vm.save()
    assert saved == [True]
    vm2 = PreferencesViewModel(service=svc)
    vm2.load()
    assert vm2.cache_max_mb == 512
    assert vm2.get_path("output_dir") == "/tmp/out"


def test_vm_restore_defaults(tmp_path):
    vm = _vm(tmp_path)
    vm.cache_max_mb = 4096
    vm.cache_max_snapshots = 99
    vm.set_path("output_dir", "/tmp/x")
    vm.restore_defaults()
    assert vm.cache_max_mb == DEFAULT_PREFERENCES["cache_max_mb"]
    assert vm.cache_max_snapshots == DEFAULT_PREFERENCES["cache_max_snapshots"]
    assert vm.get_path("output_dir") == ""


# ════════════════════════════════════════════════════════════════════════════
# PreferencesPanel — View Qt (widgets, sync, save→disco, restore) (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_panel_builds_and_syncs_from_vm(qtbot, tmp_path):
    from apps.sim_manager.perspectives.preferences.view import PreferencesPanel

    # pré-grava prefs não-default → o VM/Panel devem refleti-las.
    svc = PreferencesService(path=tmp_path / "p.json")
    svc.save(
        {"cache_max_mb": 777, "cache_max_snapshots": 7, "plot_backend": "matplotlib"}
    )
    vm = PreferencesViewModel(service=svc)
    vm.load()
    panel = PreferencesPanel(vm)
    qtbot.addWidget(panel)
    assert panel._theme.count() >= 1
    assert panel._plot_backend.count() >= 1  # ao menos matplotlib
    assert panel._cache_mb.value() == 777
    assert panel._cache_snaps.value() == 7
    assert set(panel._path_edits.keys()) == {
        "output_dir",
        "tatu_binary",
        "python_binary",
        "geosteering_ai",
    }


@pytest.mark.gui
def test_panel_save_persists_to_disk(qtbot, tmp_path):
    from apps.sim_manager.perspectives.preferences.view import PreferencesPanel

    svc = PreferencesService(path=tmp_path / "p.json")
    vm = PreferencesViewModel(service=svc)
    vm.load()
    panel = PreferencesPanel(vm)
    qtbot.addWidget(panel)
    panel._cache_mb.setValue(640)
    panel._path_edits["output_dir"].setText("/tmp/zzz")
    panel._on_save_clicked()
    assert (tmp_path / "p.json").is_file()
    reload = svc.load()
    assert reload["cache_max_mb"] == 640
    assert reload["paths"]["output_dir"] == "/tmp/zzz"
    assert "salvas" in panel._status.text()


@pytest.mark.gui
def test_panel_save_shows_error_on_oserror(qtbot, tmp_path, monkeypatch):
    """Ramo de ERRO do Salvar: OSError do service vira status (sem crash)."""
    from apps.sim_manager.perspectives.preferences.view import PreferencesPanel

    svc = PreferencesService(path=tmp_path / "p.json")
    vm = PreferencesViewModel(service=svc)
    vm.load()
    panel = PreferencesPanel(vm)
    qtbot.addWidget(panel)

    def _boom(_data):
        raise OSError("disco cheio")

    monkeypatch.setattr(svc, "save", _boom)
    panel._on_save_clicked()  # não pode crashar
    assert "falha ao salvar" in panel._status.text()


@pytest.mark.gui
def test_panel_restore_resets_widgets(qtbot, tmp_path):
    from apps.sim_manager.perspectives.preferences.view import PreferencesPanel

    vm = _vm(tmp_path)
    panel = PreferencesPanel(vm)
    qtbot.addWidget(panel)
    panel._cache_snaps.setValue(99)
    panel._on_restore_clicked()
    assert panel._cache_snaps.value() == DEFAULT_PREFERENCES["cache_max_snapshots"]
    assert "restaurados" in panel._status.text()


@pytest.mark.gui
def test_panel_set_combo_preserves_unknown_persisted_value(qtbot, tmp_path):
    """Backend persistido fora do catálogo atual não é perdido ao reabrir."""
    from apps.sim_manager.perspectives.preferences.view import PreferencesPanel

    svc = PreferencesService(path=tmp_path / "p.json")
    svc.save({"plot_backend": "backend_inexistente"})
    vm = PreferencesViewModel(service=svc)
    vm.load()
    panel = PreferencesPanel(vm)
    qtbot.addWidget(panel)
    assert panel._plot_backend.currentText() == "backend_inexistente"


# ════════════════════════════════════════════════════════════════════════════
# PreferencesPerspective — contrato plugável (id/order + build)
# ════════════════════════════════════════════════════════════════════════════
def test_perspective_metadata():
    from apps.sim_manager.perspectives.preferences.perspective import (
        PreferencesPerspective,
    )

    p = PreferencesPerspective()
    assert p.id == "preferences"
    assert p.title == "Preferências"
    assert p.order == 4
    assert p.icon_glyph == "⚙"


def test_perspective_build_viewmodel_loads(monkeypatch, tmp_path):
    """build_viewmodel cria um VM já carregado (HOME isolado p/ não tocar config real)."""
    from apps.sim_manager.perspectives.preferences.perspective import (
        PreferencesPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    monkeypatch.setenv("HOME", str(tmp_path))  # PreferencesService usa ~/.config
    vm = PreferencesPerspective().build_viewmodel(AppContext(app_name="t"))
    assert isinstance(vm, PreferencesViewModel)
    assert vm.theme == DEFAULT_PREFERENCES["theme"]  # 1º boot → defaults


@pytest.mark.gui
def test_perspective_build_view_returns_widget(qtbot, monkeypatch, tmp_path):
    from apps.sim_manager.perspectives.preferences.perspective import (
        PreferencesPerspective,
    )
    from geosteering_ai.gui.qt_compat import QtWidgets
    from geosteering_ai.gui.shell.context import AppContext

    monkeypatch.setenv("HOME", str(tmp_path))
    view = PreferencesPerspective().build_view(AppContext(app_name="t"))
    qtbot.addWidget(view)
    assert isinstance(view, QtWidgets.QWidget)
