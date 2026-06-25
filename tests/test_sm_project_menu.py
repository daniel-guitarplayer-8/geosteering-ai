# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_project_menu.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  PR-3 — SM MVVM: diálogo de projeto no startup (#7a) + barra de menu (#7b) ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-17                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Gate do PR-3: (#7a) resolve_project (PURO) abre/cria projeto a partir de ║
# ║    uma pasta; o diálogo popula recentes; main(show_startup=False) NÃO        ║
# ║    bloqueia (headless). (#7b) a SM_MainWindow tem o menu "Arquivo" com       ║
# ║    atalhos e roteia as ações via ctx.extras (late-binding None-safe), e a    ║
# ║    SimulationPerspective publica file_actions/session_actions.              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do PR-3 — projeto no startup (#7a) + barra de menu (#7b)."""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apps.sim_manager.perspectives.simulation.experiment_state import (  # noqa: E402
    ExperimentState,
)
from apps.sim_manager.startup_dialog import resolve_project  # noqa: E402


class _StubService:
    """Service stub mínimo (load_experiment/load_recents) p/ resolve_project + diálogo."""

    def __init__(self, recents=None):
        self._recents = list(recents or [])

    def load_experiment(self, path):
        return ExperimentState(
            name="carregado", output_dir=os.path.dirname(path), file_path=path
        )

    def load_recents(self):
        return list(self._recents)

    def push_recent(self, path):
        self._recents.insert(0, path)
        return self._recents


# ════════════════════════════════════════════════════════════════════════════
# #7a — resolve_project (PURO)
# ════════════════════════════════════════════════════════════════════════════
def test_resolve_project_loads_existing_exp_json(tmp_path):
    p = tmp_path / "proj.exp.json"
    p.write_text(
        json.dumps({"name": "X", "output_dir": str(tmp_path)}), encoding="utf-8"
    )
    state = resolve_project(_StubService(), str(tmp_path))
    assert state.file_path == str(p)  # carregou o .exp.json existente


def test_resolve_project_creates_new_for_empty_dir(tmp_path):
    d = tmp_path / "novo_projeto"
    d.mkdir()
    state = resolve_project(_StubService(), str(d))
    assert isinstance(state, ExperimentState)
    assert state.name == "novo_projeto" and state.output_dir == str(d)
    assert not state.file_path  # em RAM (materializa ao salvar)


def test_resolve_project_picks_first_sorted_when_multiple(tmp_path):
    """Vários .exp.json na pasta → escolhe o primeiro em ordem (determinístico)."""
    for nm in ("b_proj.exp.json", "a_proj.exp.json"):
        (tmp_path / nm).write_text("{}", encoding="utf-8")
    state = resolve_project(_StubService(), str(tmp_path))
    assert state.file_path == str(tmp_path / "a_proj.exp.json")  # 1º sorted


# ════════════════════════════════════════════════════════════════════════════
# #7a — main(show_startup=False) NÃO bloqueia (headless/teste)
# ════════════════════════════════════════════════════════════════════════════
def test_main_signature_has_show_startup_default_false():
    import inspect

    from apps.sim_manager.app import main

    sig = inspect.signature(main)
    assert sig.parameters["show_startup"].default is False


def test_app_import_still_tf_free():
    """O import de app.py segue sem TF (startup_dialog/service só dentro de main)."""
    import subprocess
    import sys

    code = (
        "import sys; import apps.sim_manager.app; "
        "assert 'tensorflow' not in sys.modules; print('OK')"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]


# ════════════════════════════════════════════════════════════════════════════
# #7a — ProjectStartupDialog (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_startup_dialog_populates_recents(qtbot):
    from apps.sim_manager.startup_dialog import ProjectStartupDialog

    svc = _StubService(recents=["/a/p1.exp.json", "/b/p2.exp.json"])
    dlg = ProjectStartupDialog(svc)
    qtbot.addWidget(dlg)
    assert dlg._recents.count() == 2
    assert dlg.result_state is None  # nada escolhido ainda


@pytest.mark.gui
def test_startup_dialog_recent_activation_sets_state(qtbot):
    from apps.sim_manager.startup_dialog import ProjectStartupDialog
    from geosteering_ai.gui.qt_compat import Qt

    svc = _StubService(recents=["/a/p1.exp.json"])
    dlg = ProjectStartupDialog(svc)
    qtbot.addWidget(dlg)
    item = dlg._recents.item(0)
    assert item.data(Qt.ItemDataRole.UserRole) == "/a/p1.exp.json"
    dlg._on_recent_activated(item)  # simula duplo-clique
    assert dlg.result_state is not None
    assert dlg.result_state.file_path == "/a/p1.exp.json"


# ════════════════════════════════════════════════════════════════════════════
# #7b — barra de menu (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_main_window_has_file_menu_with_shortcuts(qtbot):
    from apps.sim_manager.main_window import SM_MainWindow
    from geosteering_ai.gui.qt_compat import QtGui, QtWidgets
    from geosteering_ai.gui.shell.context import AppContext

    win = SM_MainWindow(AppContext(app_name="t"))
    qtbot.addWidget(win)
    menus = [m.title() for m in win.menuBar().findChildren(QtWidgets.QMenu)]
    assert any("Arquivo" in t for t in menus)
    # ações + atalhos (texto com & de mnemônico)
    shortcuts = {
        a.text(): a.shortcut().toString() for a in win.findChildren(QtGui.QAction)
    }
    labels = "".join(shortcuts.keys())
    assert "Novo" in labels and "Abrir" in labels and "Salvar" in labels
    # atalho-chave presente (Ctrl+N no "Novo")
    assert any("Ctrl+N" in sc for txt, sc in shortcuts.items() if "Novo" in txt)


@pytest.mark.gui
def test_menu_routes_to_ctx_extras_actions(qtbot):
    from apps.sim_manager.main_window import SM_MainWindow
    from geosteering_ai.gui.shell.context import AppContext

    ctx = AppContext(app_name="t")
    win = SM_MainWindow(ctx)
    qtbot.addWidget(win)
    calls = []
    ctx.extras["file_actions"] = {"new": lambda: calls.append("new")}
    win._invoke("file_actions", "new")
    assert calls == ["new"]
    # no-op se a ação não existe (perspectiva não buildou) — sem crash
    win._invoke("file_actions", "inexistente")
    win._invoke("session_actions", "save")
    assert calls == ["new"]


@pytest.mark.gui
def test_simulation_perspective_publishes_menu_actions(qtbot):
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    # secondary_sidebar stub mínimo (a publicação ocorre após a wiring do painel).
    class _Sidebar:
        def append_log(self, *_):
            pass

        def set_history_panel(self, *_):
            pass

    ctx = AppContext(app_name="t")
    ctx.extras["secondary_sidebar"] = _Sidebar()
    view = SimulationPerspective().build_view(ctx)
    qtbot.addWidget(view)
    fa = ctx.extras.get("file_actions")
    sa = ctx.extras.get("session_actions")
    assert fa and set(fa) == {"new", "open", "save", "save_as", "close"}
    assert sa and set(sa) == {"save", "open"}
    assert all(callable(f) for f in fa.values())


@pytest.mark.gui
def test_menu_action_fires_handler_via_real_shell(qtbot):
    """Integração end-to-end (#7b): a SM_MainWindow REAL (com a sidebar Antigravity)
    builda a Simulação no boot → publica file_actions → a QAction do menu dispara o
    handler real. Cobre o caminho de produção (com sidebar), não só o stub.
    """
    from apps.sim_manager.main_window import SM_MainWindow
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    win = SM_MainWindow(AppContext(app_name="t"))
    qtbot.addWidget(win)
    win.add_perspective(
        SimulationPerspective()
    )  # order=0 → builda no boot (com sidebar)
    # a Simulação publicou os comandos do menu via a sidebar REAL do shell.
    assert "file_actions" in win.ctx.extras
    assert set(win.ctx.extras["file_actions"]) == {
        "new",
        "open",
        "save",
        "save_as",
        "close",
    }
    # disparar "save" pelo roteador do menu chama o handler real (sem crash).
    win._invoke("file_actions", "save")  # _on_save_experiment → _exp_vm.save_experiment


# ════════════════════════════════════════════════════════════════════════════
# #7a — adoção do projeto na Simulação (output_dir)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_simulation_adopts_project_output_dir(qtbot, tmp_path):
    from apps.sim_manager.perspectives.simulation.perspective import (
        SimulationPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    class _Sidebar:
        def append_log(self, *_):
            pass

        def set_history_panel(self, *_):
            pass

    ctx = AppContext(app_name="t")
    ctx.extras["secondary_sidebar"] = _Sidebar()
    ctx.extras["project"] = ExperimentState(name="Proj", output_dir=str(tmp_path))
    ctx.extras["project_dir"] = str(tmp_path)
    persp = SimulationPerspective()
    persp.build_view(ctx)
    qtbot.addWidget(persp._view)
    assert persp._sim_vm.output_dir == str(tmp_path)  # pré-preencheu de project_dir
