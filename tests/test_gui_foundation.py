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
# ║    AC-1.3 __all__ completo (16) ....... test_gui_qt_compat_all_complete   ║
# ║    AC-2.1 shim antigo importa ......... test_legacy_shim_imports          ║
# ║    AC-2.2 identidade re-exportada ..... test_shim_reexports_same_objects  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0004 — fundação ``geosteering_ai/gui/`` (extração de qt_compat).

Não exigem ``DISPLAY``/``QApplication`` (apenas import + identidade de símbolos),
logo rodam fora do caminho ``pytest-qt``/``xvfb``. A regressão da GUI completa é
coberta por ``tests/test_simulation_manager_gui.py`` (rodado sob xvfb no GATE-V).
"""

from __future__ import annotations

import importlib

import pytest

# Os 16 símbolos públicos do contrato de ``qt_compat`` (spec 0004 §3).
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
    """AC-1.3 — ``__all__`` tem os 16 nomes E todos são atributos reais."""
    qc = importlib.import_module(_NEW)
    assert set(qc.__all__) == EXPECTED_SYMBOLS
    for name in qc.__all__:
        assert hasattr(qc, name), f"{name} em __all__ mas ausente no módulo"


# ── AC-2.x — caminho ANTIGO (shim de retrocompat) ────────────────────────────
def test_legacy_shim_imports():
    """AC-2.1 — o caminho ANTIGO (shim) ainda importa os símbolos."""
    from geosteering_ai.simulation.tests.sm_qt_compat import (  # noqa: F401
        QT_BINDING,
        QtCore,
        QThread,
        QtWidgets,
        Signal,
    )


def test_shim_reexports_same_objects():
    """AC-2.2 — o shim re-exporta os MESMOS objetos (identidade) do módulo canônico.

    ``gui.qt_compat.X is sm_qt_compat.X`` para os 16 símbolos (referência, não cópia).
    """
    new = importlib.import_module(_NEW)
    old = importlib.import_module(_OLD)
    # __all__ idêntico nos dois caminhos.
    assert set(old.__all__) == set(new.__all__) == EXPECTED_SYMBOLS
    # Identidade de CADA símbolo re-exportado (mesma referência, não cópia).
    for name in EXPECTED_SYMBOLS:
        assert getattr(old, name) is getattr(
            new, name
        ), f"{name}: identidade quebrada entre gui.qt_compat e o shim"


def test_shim_is_distinct_module():
    """O shim é um MÓDULO distinto (re-export por valor, não alias sys.modules)."""
    new = importlib.import_module(_NEW)
    old = importlib.import_module(_OLD)
    assert old is not new  # módulos diferentes; símbolos idênticos (test acima)


def test_binding_resolved_when_qt_available():
    """Se há binding Qt6 no ambiente, ele é resolvido.

    ``QT_BINDING`` ∈ {PyQt6, PySide6} e ``QtWidgets`` não é ``None``.
    """
    from geosteering_ai.gui.qt_compat import QT_AVAILABLE, QT_BINDING, QtWidgets

    if not QT_AVAILABLE:
        pytest.skip("nenhum binding Qt6 (PyQt6/PySide6) no ambiente")
    assert QT_BINDING in {"PyQt6", "PySide6"}
    assert QtWidgets is not None
