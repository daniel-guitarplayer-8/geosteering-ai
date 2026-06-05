# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_qt_compat.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Compatibilidade Qt (shim retrocompat) ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18 · 2026-06-05 (relocado p/ gui/, spec 0004)       ║
# ║  Status      : Produção (shim de retrocompatibilidade)                    ║
# ║  Dependências: geosteering_ai.gui.qt_compat                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Shim de retrocompatibilidade. A camada de compatibilidade Qt           ║
# ║    (PyQt6/PySide6 + locale C + dark-mode) foi RELOCADA na spec 0004 para  ║
# ║    o pacote de PRODUÇÃO de 1ª classe ``geosteering_ai/gui/qt_compat.py``  ║
# ║    (para que SM e Studio compartilhem a fundação Qt, e a GUI deixe de      ║
# ║    viver dentro de ``tests/``). Este módulo re-exporta dali — os 16        ║
# ║    importadores existentes (``sm_*.py``, ``simulation_manager.py``,        ║
# ║    ``tests/conftest*.py``) continuam funcionando SEM alteração.           ║
# ║                                                                           ║
# ║  PADRÃO                                                                   ║
# ║    Idêntico ao precedente ``sm_io.py`` (v2.53): impl em produção +        ║
# ║    shim de re-export no caminho legado (Strangler Fig — ADR-S01).         ║
# ║                                                                           ║
# ║  EXPORTS (re-export de gui.qt_compat — objetos IDÊNTICOS)                 ║
# ║    QtCore, QtGui, QtWidgets, Qt, Signal, Slot, QThread, QObject,          ║
# ║    QT_AVAILABLE, QT_BINDING, QT_IMPORT_ERROR, check_qt_available,         ║
# ║    detect_os_dark_mode, enforce_c_locale, format_float, make_double_spin  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Shim de retrocompat — re-exporta a camada de compatibilidade Qt.

A implementação canônica vive em :mod:`geosteering_ai.gui.qt_compat` (relocada na
spec 0004 — keystone da Fase 0). Mantido para não quebrar os importadores legados
que faziam ``from geosteering_ai.simulation.tests.sm_qt_compat import ...``
(imports relativos ``from .sm_qt_compat import ...`` nos módulos ``sm_*.py``).

Os objetos re-exportados são os MESMOS (mesma identidade) do módulo de produção —
a detecção de binding roda uma única vez no import de ``gui.qt_compat``.

TODO (Fase 0B — specs 0005+): este shim é TRANSITÓRIO. Quando os demais módulos
``sm_*.py`` forem relocados para ``geosteering_ai/gui/`` e seus imports relativos
atualizados para ``geosteering_ai.gui.*``, este shim (e a inversão de camada
"código em ``tests/`` importando ``gui/``") deve ser REMOVIDO.
"""

from __future__ import annotations

from geosteering_ai.gui.qt_compat import (
    QT_AVAILABLE,
    QT_BINDING,
    QT_IMPORT_ERROR,
    QObject,
    Qt,
    QtCore,
    QtGui,
    QThread,
    QtWidgets,
    Signal,
    Slot,
    check_qt_available,
    detect_os_dark_mode,
    enforce_c_locale,
    format_float,
    make_double_spin,
)

__all__ = [
    "QObject",
    "QT_AVAILABLE",
    "QT_BINDING",
    "QT_IMPORT_ERROR",
    "QThread",
    "Qt",
    "QtCore",
    "QtGui",
    "QtWidgets",
    "Signal",
    "Slot",
    "check_qt_available",
    "detect_os_dark_mode",
    "enforce_c_locale",
    "format_float",
    "make_double_spin",
]
