# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/viewmodels/__init__.py                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ViewModels — camada MVVM PURA (sem Qt)                     ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — base MVVM (spec 0005)                                ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : stdlib PURO (NÃO importa Qt) — Princípio X                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Camada de ViewModels do MVVM. Tudo aqui é Python PURO (sem PyQt6/      ║
# ║    PySide6) e testável sem ``pytest-qt``. A View (Qt) adapta os sinais.   ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    VMSignal (signal.py) · BaseViewModel (base.py)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Camada de ViewModels (MVVM) — Python PURO, testável sem Qt (Princípio X)."""

from __future__ import annotations

from geosteering_ai.gui.viewmodels.base import BaseViewModel
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["BaseViewModel", "VMSignal"]
