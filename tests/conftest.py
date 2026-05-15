# -*- coding: utf-8 -*-
"""Root conftest — setup global executado antes de qualquer plugin pytest.

Função única: alinhar ``QT_API`` (lido por pytest-qt) com o binding que
``sm_qt_compat`` carrega (PyQt6 preferido, PySide6 fallback). Necessário
porque pytest-qt detecta o binding em ``pytest_configure`` (early), antes
de qualquer ``pytest_plugins`` declarado em arquivos de teste.

Sem este conftest, pytest-qt pode escolher PySide6 enquanto a aplicação
real (Simulation Manager) usa PyQt6 — provocando ``TypeError`` em
``qtbot.addWidget`` por cross-binding isinstance falso.

Sprint v2.33 — Suite pytest-qt para GUI.
"""

from __future__ import annotations

import os

# Setar QT_API ANTES de qualquer import Qt ou pytest-qt. Idempotente: respeita
# override do usuário via env var.
if "QT_API" not in os.environ:
    try:
        import PyQt6.QtCore  # noqa: F401

        os.environ["QT_API"] = "pyqt6"
    except ImportError:
        try:
            import PySide6.QtCore  # noqa: F401

            os.environ["QT_API"] = "pyside6"
        except ImportError:
            # Nenhum binding Qt6 — testes GUI serão skipped no módulo
            pass

# Mesmo setup para Qt platform headless (CI Linux). Espelha a lógica de
# ``tests/conftest_qt.py`` mas executado mais cedo no ciclo de carga.
if "DISPLAY" not in os.environ and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
