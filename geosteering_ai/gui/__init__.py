# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/__init__.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Fundação GUI — pacote de 1ª classe (infra Qt compartilhada) ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI (Simulation Manager + Geosteering AI Studio)           ║
# ║  Versão      : v0.1 (keystone — spec 0004)                                ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : PyQt6 (preferido) / PySide6 (fallback) via qt_compat        ║
# ║  Dependências: extra pip [gui]; NÃO importado por core (models/losses/…)   ║
# ║  Padrão      : MVVM (Princípio X) — fundação compartilhada SM + Studio      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Pacote de 1ª classe que abriga a INFRAESTRUTURA Qt extraída de         ║
# ║    ``simulation/tests/sm_*.py`` (Strangler Fig — ADR-S01 do blueprint).   ║
# ║    Tanto o Simulation Manager (SM) quanto o futuro Geosteering AI Studio  ║
# ║    importam desta fundação. A física NUNCA vive aqui — converge sempre    ║
# ║    em ``simulation/multi_forward.py`` (Princípio XI).                     ║
# ║                                                                           ║
# ║  CONTEÚDO (incremental — Fase 0)                                          ║
# ║    • qt_compat        binding PyQt6/PySide6 + locale C + dark-mode (0004) ║
# ║    • [futuro 0005]    MVVM base (VMSignal, MainWindowBase, Perspective)   ║
# ║    • [futuro 0006]    plotting/backends (PlotCanvas ABC + 4 backends)     ║
# ║    • [futuro 0007]    persistence (.session atômico)                      ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    (nenhum no nível do pacote — submódulos importados explicitamente,     ║
# ║     ex.: ``from geosteering_ai.gui.qt_compat import QtCore``)             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Fundação GUI do Geosteering AI — infraestrutura Qt compartilhada (SM + Studio).

Este pacote é a casa canônica da infraestrutura de interface, extraída de
``geosteering_ai/simulation/tests/sm_*.py`` via *Strangler Fig* (spec 0004 — a
keystone da Fase 0). O ``core`` do pacote (``models``, ``losses``, ``training``,
``inference``, ``data``, ``simulation``) **nunca** importa ``gui`` — a dependência
de Qt é opcional (extra pip ``[gui]``).

Uso::

    from geosteering_ai.gui.qt_compat import QtCore, QtWidgets, Signal, QT_BINDING

Ver: ``docs/architecture/04_ui_ux_mvvm.md`` (arquitetura MVVM) e a reconciliação
SM↔Studio em ``docs/reports/geosteering_ai_portfolio_4_produtos_status_2026-06-04.md``.
"""

from __future__ import annotations

# Submódulos são importados explicitamente pelos consumidores (qt_compat, etc.).
# Não re-exportamos no nível do pacote para manter os imports rastreáveis (D8).
__all__: list[str] = []
