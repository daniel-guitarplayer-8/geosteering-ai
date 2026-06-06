# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_gui_foundation.py                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0004-gui-foundation                                        ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — fundação geosteering_ai/gui/ (keystone Fase 0)        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Verifica a extração de ``sm_qt_compat`` → ``gui/qt_compat`` (Strangler ║
# ║    Fig): o caminho NOVO importa, o shim no caminho ANTIGO re-exporta os    ║
# ║    MESMOS objetos (identidade), e o pacote ``gui`` é de 1ª classe.        ║
# ║                                                                           ║
# ║  MAPA AC → TESTE                                                          ║
# ║    AC-1.1 novo caminho importa ........ test_gui_qt_compat_imports        ║
# ║    AC-1.2 pacote gui importável ....... test_gui_package_importable       ║
# ║    AC-1.3 __all__ completo ............ test_gui_qt_compat_all_complete   ║
# ║    Spec 0011 F0: shim REMOVIDO ........ test_legacy_shim_removed          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0004 — fundação ``geosteering_ai/gui/`` (extração de qt_compat).

Não exigem ``DISPLAY``/``QApplication`` (apenas import + identidade de símbolos),
logo rodam fora do caminho ``pytest-qt``/``xvfb``. A regressão da GUI completa é
coberta por ``tests/test_simulation_manager_gui.py`` (rodado sob xvfb no GATE-V).
"""

from __future__ import annotations

import importlib

import pytest

# Símbolos públicos do contrato de ``qt_compat`` (spec 0004 §3 + ``load_qwebengineview``
# adicionado no code-review de 0006/0007).
EXPECTED_SYMBOLS = {
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
    "load_qwebengineview",
    "make_double_spin",
}

_NEW = "geosteering_ai.gui.qt_compat"
_OLD = "geosteering_ai.simulation.tests.sm_qt_compat"


# ── AC-1.x — caminho NOVO (gui/) ─────────────────────────────────────────────
def test_gui_package_importable():
    """AC-1.2 — o pacote ``geosteering_ai.gui`` é de 1ª classe e importável."""
    mod = importlib.import_module("geosteering_ai.gui")
    assert mod is not None
    assert mod.__all__ == []  # submódulos importados explicitamente (D8)


def test_gui_qt_compat_imports():
    """AC-1.1 — o caminho NOVO importa os símbolos-chave."""
    from geosteering_ai.gui.qt_compat import (  # noqa: F401
        QT_AVAILABLE,
        QT_BINDING,
        QtCore,
        QThread,
        QtWidgets,
        Signal,
    )


def test_gui_qt_compat_all_complete():
    """AC-1.3 — ``__all__`` tem todos os nomes esperados E são atributos reais."""
    qc = importlib.import_module(_NEW)
    assert set(qc.__all__) == EXPECTED_SYMBOLS
    for name in qc.__all__:
        assert hasattr(qc, name), f"{name} em __all__ mas ausente no módulo"


# ── De-shim (spec 0011 Fase 0) — o caminho ANTIGO foi REMOVIDO ───────────────
def test_legacy_shim_removed():
    """Spec 0011 Fase 0 — o shim antigo (``sm_qt_compat``) foi REMOVIDO.

    Os consumidores migraram para ``gui.qt_compat`` diretamente; o caminho antigo
    não existe mais (importá-lo levanta ``ModuleNotFoundError``).
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(_OLD)


def test_binding_resolved_when_qt_available():
    """Se há binding Qt6 no ambiente, ele é resolvido.

    ``QT_BINDING`` ∈ {PyQt6, PySide6} e ``QtWidgets`` não é ``None``.
    """
    from geosteering_ai.gui.qt_compat import QT_AVAILABLE, QT_BINDING, QtWidgets

    if not QT_AVAILABLE:
        pytest.skip("nenhum binding Qt6 (PyQt6/PySide6) no ambiente")
    assert QT_BINDING in {"PyQt6", "PySide6"}
    assert QtWidgets is not None
