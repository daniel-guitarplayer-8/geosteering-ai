# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/main_window.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SM_MainWindow — casca QMainWindow do Simulation Manager    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — casca (spec 0011a)                         ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — walking skeleton                                ║
# ║  Dependências: gui.shell.MainWindowBase                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Janela principal do SM MVVM: subclasse de ``MainWindowBase`` que        ║
# ║    hospeda as perspectivas como abas (lazy). No walking skeleton é uma     ║
# ║    casca mínima; o menu Novo/Abrir/Salvar ``.session`` entra numa fatia    ║
# ║    futura (reusa ``gui.persistence.SessionDocument``).                     ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SM_MainWindow                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SM_MainWindow`` — casca QMainWindow do Simulation Manager MVVM (0011a/0013)."""

from __future__ import annotations

from typing import Any, Optional

from geosteering_ai.gui.qt_compat import QtGui
from geosteering_ai.gui.shell.antigravity_window import AntigravityMainWindow
from geosteering_ai.gui.shell.context import AppContext

__all__ = ["SM_MainWindow"]


class SM_MainWindow(AntigravityMainWindow):
    """Janela principal do Simulation Manager (MVVM) — shell Antigravity + barra de menu.

    Note:
        Spec 0013: herda ``AntigravityMainWindow`` (activity rail + perspectivas
        empilhadas + secondary sidebar Histórico/Log/Artifacts + status bar com
        accent). PR-3 (#7b): adiciona a barra de menu "Arquivo" (Novo/Abrir/Salvar/
        Salvar Como/Fechar/Sair) + submenu "Sessão". As ações ROTEIAM via
        ``ctx.extras["file_actions"]``/``["session_actions"]`` (callables publicados
        pela SimulationPerspective) — late-binding None-safe, sem duplicar lógica.
    """

    def __init__(self, ctx: AppContext, *, parent: Optional[Any] = None) -> None:
        super().__init__(ctx, parent=parent)
        self._build_menubar()

    # ── Barra de menu (PR-3 #7b) ────────────────────────────────────────────
    def _build_menubar(self) -> None:
        """Monta o menu "Arquivo" (+ submenu "Sessão") com atalhos padrão."""
        file_menu = self.menuBar().addMenu("&Arquivo")
        # (rótulo, atalho, chave do callable em ctx.extras["file_actions"])
        for label, shortcut, key in (
            ("&Novo projeto", "Ctrl+N", "new"),
            ("&Abrir projeto…", "Ctrl+O", "open"),
            ("&Salvar", "Ctrl+S", "save"),
            ("Salvar &como…", "Ctrl+Shift+S", "save_as"),
            ("&Fechar", "Ctrl+W", "close"),
        ):
            act = QtGui.QAction(label, self)
            act.setShortcut(QtGui.QKeySequence(shortcut))
            act.triggered.connect(
                lambda _checked=False, k=key: self._invoke("file_actions", k)
            )
            file_menu.addAction(act)
        # Submenu "Sessão" (.session — params reproduzíveis).
        sess = file_menu.addMenu("Sess&ão")
        for label, key in (("Salvar sessão…", "save"), ("Abrir sessão…", "open")):
            act = QtGui.QAction(label, self)
            act.triggered.connect(
                lambda _checked=False, k=key: self._invoke("session_actions", k)
            )
            sess.addAction(act)
        file_menu.addSeparator()
        quit_act = QtGui.QAction("Sai&r", self)
        quit_act.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)  # ação de janela (não precisa de extras)
        file_menu.addAction(quit_act)

    def _invoke(self, group: str, key: str) -> None:
        """Dispara o callable ``ctx.extras[group][key]`` (late-binding None-safe).

        O menu é criado no ``__init__`` (antes das perspectivas buildarem, lazy); ler
        os callables NO CLIQUE evita acoplamento de ordem de inicialização. Se a
        perspectiva ainda não publicou o grupo (ou foi destruída), a ação é no-op.
        """
        actions = self.ctx.extras.get(group)
        if isinstance(actions, dict):
            fn = actions.get(key)
            if callable(fn):
                fn()
