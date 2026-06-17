# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_qt_binding_boundary.py                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Subsistema  : GUI — binding Qt (PyQt6 primário / PySide6 fallback)       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-06                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garante que TODO acesso a Qt em gui/ + apps/ passa por ``qt_compat`` —  ║
# ║    o ÚNICO ponto de resolução do binding (tenta PyQt6, cai p/ PySide6).    ║
# ║    Importar ``PyQt6``/``PySide6`` direto (mesmo em ``TYPE_CHECKING``)      ║
# ║    quebraria o fallback agnóstico ao binding. Guard de regressão.          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Fronteira de binding Qt — gui/ + apps/ só acessam Qt via ``qt_compat``."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Captura ``import PyQt6...`` / ``from PyQt6...`` (idem PySide6) em qualquer linha.
_DIRECT_QT_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:PyQt6|PySide6)\b", re.MULTILINE
)

# Arquivo autorizado a tocar os bindings diretamente na FUNDAÇÃO gui/ (resolve o fallback).
_QT_COMPAT_ONLY = {"qt_compat.py"}

# No app MVVM (apps/), o entry-point app.py também é autorizado: ele cria o QApplication
# ANTES de qualquer import de geosteering_ai. Importar qt_compat (que vive sob
# geosteering_ai.gui) dispararia geosteering_ai/__init__ → noise → training → TensorFlow,
# e TF carregado ANTES do QApplication causa SEGFAULT no plugin xcb do Qt (regressão real
# corrigida em 2026-06-16; ver app.py "ORDEM DE BOOT CRÍTICA" + tests/test_sm_app_boot.py).
# O bootstrap direto AINDA PRESERVA o fallback agnóstico (tenta PyQt6, cai p/ PySide6).
_APP_BOOTSTRAP_ALLOWED = _QT_COMPAT_ONLY | {"app.py"}


def _offenders(root: Path, allowed: set[str]) -> list[str]:
    """Arquivos .py sob ``root`` que importam PyQt6/PySide6 direto (exceto ``allowed``)."""
    found: list[str] = []
    for py in sorted(root.rglob("*.py")):
        if py.name in allowed:
            continue
        if _DIRECT_QT_IMPORT.search(py.read_text(encoding="utf-8")):
            found.append(str(py.relative_to(PROJECT_ROOT)))
    return found


def test_gui_imports_qt_only_via_qt_compat():
    """A fundação gui/ acessa Qt SÓ via qt_compat (PyQt6 primário / PySide6 fallback)."""
    offenders = _offenders(PROJECT_ROOT / "geosteering_ai" / "gui", _QT_COMPAT_ONLY)
    assert not offenders, (
        "gui/ deve importar Qt só via geosteering_ai.gui.qt_compat "
        f"(inclusive em TYPE_CHECKING). Violações: {offenders}"
    )


def test_app_imports_qt_only_via_qt_compat():
    """O app MVVM (apps/) acessa Qt SÓ via qt_compat — exceto o bootstrap de app.py.

    ``app.py`` cria o QApplication antes de importar geosteering_ai (ordem
    TF-após-QApplication obrigatória — senão segfault xcb), por isso importa o binding
    direto (preservando o fallback PyQt6→PySide6). Demais módulos de apps/ continuam
    restritos ao qt_compat.
    """
    offenders = _offenders(PROJECT_ROOT / "apps", _APP_BOOTSTRAP_ALLOWED)
    assert not offenders, (
        "apps/ deve importar Qt só via geosteering_ai.gui.qt_compat "
        "(exceto o bootstrap de QApplication em app.py). "
        f"Violações: {offenders}"
    )


def test_qt_compat_resolves_a_binding():
    """qt_compat resolve um binding (PyQt6 ou PySide6) e expõe os nomes unificados."""
    from geosteering_ai.gui import qt_compat

    assert qt_compat.QT_AVAILABLE, qt_compat.QT_IMPORT_ERROR
    assert qt_compat.QT_BINDING in ("PyQt6", "PySide6")
    # nomes unificados (Signal/Slot/QtWidgets) disponíveis independentemente do binding
    assert qt_compat.QtWidgets is not None
    assert qt_compat.Signal is not None
