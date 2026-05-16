# -*- coding: utf-8 -*-
"""Root conftest — setup global executado antes de qualquer plugin pytest.

Funções:

1. Alinhar ``QT_API`` (lido por pytest-qt) com o binding que
   ``sm_qt_compat`` carrega (PyQt6 preferido, PySide6 fallback). Necessário
   porque pytest-qt detecta o binding em ``pytest_configure`` (early), antes
   de qualquer ``pytest_plugins`` declarado em arquivos de teste.

   Sem este conftest, pytest-qt pode escolher PySide6 enquanto a aplicação
   real (Simulation Manager) usa PyQt6 — provocando ``TypeError`` em
   ``qtbot.addWidget`` por cross-binding isinstance falso.

   Sprint v2.33 — Suite pytest-qt para GUI.

2. Configurar ``NUMBA_CACHE_DIR`` em tmpfs ANTES de qualquer import Numba
   (v2.36 D3, finalizado v2.38). Reduz I/O do carregamento de ``.nbc``
   entre testes paralelos sem poluir ``geosteering_ai/__pycache__/_numba/``.
   Respeita override do usuário (``NUMBA_CACHE_DIR`` já setado).

   Antes deste setup, ``cli/_main.py`` definia a env var em tempo de
   import — mas só quando o CLI era importado. Testes que importam
   diretamente ``simulation/_numba/*`` não acionavam o setup e geravam
   ``.nbc`` em locais inconsistentes entre runs.
"""

from __future__ import annotations

import os
import tempfile

# ── NUMBA_CACHE_DIR em tmpfs (v2.36 D3) ──────────────────────────────────
# DEVE executar antes de qualquer ``import numba`` ou
# ``from geosteering_ai.simulation._numba import ...``. Idempotente:
# preserva override do usuário (CI, debug).
if "NUMBA_CACHE_DIR" not in os.environ:
    _cache_dir = os.path.join(tempfile.gettempdir(), "geosteering_numba_cache_test")
    try:
        os.makedirs(_cache_dir, mode=0o700, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = _cache_dir
    except OSError:
        # Filesystem read-only ou permissão negada: deixa Numba
        # usar o default (__pycache__/_numba/). Não-fatal.
        pass

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
