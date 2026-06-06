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

from geosteering_ai.gui.shell.antigravity_window import AntigravityMainWindow

__all__ = ["SM_MainWindow"]


class SM_MainWindow(AntigravityMainWindow):
    """Janela principal do Simulation Manager (MVVM) — shell Antigravity.

    Note:
        Spec 0013: herda ``AntigravityMainWindow`` (activity rail + perspectivas
        empilhadas + secondary sidebar Histórico/Log/Artifacts + status bar com
        accent). Hospeda a perspectiva Simulação + scaffolds das próximas fatias.
        Menu ``.session`` e toolbar próprios são fatias futuras.
    """
