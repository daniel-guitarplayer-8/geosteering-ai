# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/__init__.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Shell MVVM — contratos da casca de aplicação              ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell MVVM (spec 0005)                              ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Camada de casca (shell) do MVVM: contexto de app, contrato de         ║
# ║    perspectiva e a janela base.                                          ║
# ║                                                                           ║
# ║  FRONTEIRA DE IMPORT (Princípio X / RNF-3)                               ║
# ║    Este ``__init__`` re-exporta APENAS o que é Qt-FREE — ``AppContext`` e ║
# ║    ``Perspective`` (importáveis e testáveis sem PyQt6/PySide6). A casca   ║
# ║    ``MainWindowBase`` IMPORTA Qt e NÃO é re-exportada aqui; acesse-a via  ║
# ║    ``from geosteering_ai.gui.shell.main_window_base import MainWindowBase`` ║
# ║    para que ``import geosteering_ai.gui.shell`` permaneça Qt-free.        ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    AppContext · Perspective   (MainWindowBase fica no submódulo)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Camada de casca (shell) do MVVM — contratos Qt-free + janela base (submódulo).

Re-exporta apenas o que é Qt-FREE (``AppContext``, ``Perspective``). A
``MainWindowBase`` (que importa Qt) vive em ``main_window_base.py`` e NÃO é
re-exportada aqui, preservando ``import geosteering_ai.gui.shell`` sem Qt
(Princípio X). Ver ``docs/architecture/04_ui_ux_mvvm.md``.
"""

from __future__ import annotations

from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.shell.perspective import Perspective

__all__ = ["AppContext", "Perspective"]
