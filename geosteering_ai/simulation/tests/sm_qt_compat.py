# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_qt_compat.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Camada de Compatibilidade Qt          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Atualizado  : 2026-04-18 (locale C + helpers tipográficos)               ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: PyQt6 (preferido) | PySide6 | PyQt5                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Exporta símbolos Qt (QtWidgets, QtCore, QtGui, QThread, Signal, Slot)  ║
# ║    de forma neutra entre os bindings Python mais usados, garantindo que   ║
# ║    o Simulation Manager rode no maior número de ambientes Python sem     ║
# ║    exigir uma instalação específica de binding Qt.                       ║
# ║                                                                           ║
# ║  ORDEM DE RESOLUÇÃO                                                       ║
# ║    1. PyQt6      (moderno, API Qt 6, requer Python ≥ 3.9)                 ║
# ║    2. PySide6    (oficial Nokia/Qt Company, API idêntica a PyQt6)         ║
# ║    3. PyQt5      (legado, Qt 5 — uso com WARN por obsolescência)          ║
# ║                                                                           ║
# ║  DIFERENÇAS TRATADAS                                                      ║
# ║    • pyqtSignal (PyQt) vs Signal (PySide)   → exporta como ``Signal``     ║
# ║    • pyqtSlot   (PyQt) vs Slot   (PySide)   → exporta como ``Slot``       ║
# ║    • QAction em QtWidgets (PyQt5) vs QtGui  → reexport uniforme           ║
# ║    • Enums Qt6 (Qt.AlignmentFlag.AlignLeft) vs Qt5 (Qt.AlignLeft)         ║
# ║      → constantes auxiliares ``ALIGN_*`` / ``ORIENT_*`` isentas de break  ║
# ║                                                                           ║
# ║  NOVOS HELPERS (2026-04-18)                                               ║
# ║    • enforce_c_locale(app)     → força ponto como separador decimal      ║
# ║    • make_double_spin(..)       → QDoubleSpinBox com locale C             ║
# ║    • format_float(value, prec)  → formatação universal com ponto         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Camada de compatibilidade entre PyQt6, PySide6 e PyQt5.

Este módulo provê imports *unified* para permitir que todo o restante do
Simulation Manager use uma única convenção (``Signal``, ``Slot``,
``QT_BINDING``) independente do binding Qt disponível no ambiente.

Também fornece helpers para garantir que a apresentação de números
decimais seja **sempre** com ponto (``.``) — jamais vírgula (``,``) —
independente do locale do sistema (pt_BR usa vírgula por padrão em
``QDoubleSpinBox`` Qt6, o que é incorreto para dados científicos).

Example:
    Uso típico em módulos da aplicação::

        >>> from geosteering_ai.simulation.tests.sm_qt_compat import (
        ...     QtCore, QtGui, QtWidgets, Signal, Slot, Qt, QT_BINDING
        ... )
        >>> class MyWidget(QtWidgets.QWidget):
        ...     my_sig = Signal(int)

Note:
    Se nenhum binding estiver instalado, ``import sm_qt_compat`` ainda
    funciona, mas o atributo ``QT_AVAILABLE`` será ``False``. O programa
    principal detecta isso e apresenta uma mensagem de instalação clara.
"""
from __future__ import annotations

import sys
from typing import Any, Optional

QT_BINDING: Optional[str] = None
QT_AVAILABLE: bool = False
QT_IMPORT_ERROR: Optional[str] = None

QtCore: Any = None
QtGui: Any = None
QtWidgets: Any = None
Qt: Any = None
Signal: Any = None
Slot: Any = None
QThread: Any = None
QObject: Any = None


def _try_pyqt6() -> bool:
    """Tenta importar PyQt6. Retorna True se bem-sucedido."""
    global QtCore, QtGui, QtWidgets, Qt, Signal, Slot, QThread, QObject
    global QT_BINDING, QT_AVAILABLE
    try:
        from PyQt6 import QtCore as _QtCore
        from PyQt6 import QtGui as _QtGui
        from PyQt6 import QtWidgets as _QtWidgets

        QtCore = _QtCore
        QtGui = _QtGui
        QtWidgets = _QtWidgets
        Qt = _QtCore.Qt
        Signal = _QtCore.pyqtSignal
        Slot = _QtCore.pyqtSlot
        QThread = _QtCore.QThread
        QObject = _QtCore.QObject
        QT_BINDING = "PyQt6"
        QT_AVAILABLE = True
        return True
    except ImportError:
        return False


def _try_pyside6() -> bool:
    """Tenta importar PySide6. Retorna True se bem-sucedido."""
    global QtCore, QtGui, QtWidgets, Qt, Signal, Slot, QThread, QObject
    global QT_BINDING, QT_AVAILABLE
    try:
        from PySide6 import QtCore as _QtCore
        from PySide6 import QtGui as _QtGui
        from PySide6 import QtWidgets as _QtWidgets

        QtCore = _QtCore
        QtGui = _QtGui
        QtWidgets = _QtWidgets
        Qt = _QtCore.Qt
        Signal = _QtCore.Signal
        Slot = _QtCore.Slot
        QThread = _QtCore.QThread
        QObject = _QtCore.QObject
        QT_BINDING = "PySide6"
        QT_AVAILABLE = True
        return True
    except ImportError:
        return False


def _try_pyqt5() -> bool:
    """Tenta importar PyQt5 (legado). Retorna True se bem-sucedido."""
    global QtCore, QtGui, QtWidgets, Qt, Signal, Slot, QThread, QObject
    global QT_BINDING, QT_AVAILABLE
    try:
        from PyQt5 import QtCore as _QtCore
        from PyQt5 import QtGui as _QtGui
        from PyQt5 import QtWidgets as _QtWidgets

        QtCore = _QtCore
        QtGui = _QtGui
        QtWidgets = _QtWidgets
        Qt = _QtCore.Qt
        Signal = _QtCore.pyqtSignal
        Slot = _QtCore.pyqtSlot
        QThread = _QtCore.QThread
        QObject = _QtCore.QObject
        QT_BINDING = "PyQt5"
        QT_AVAILABLE = True
        return True
    except ImportError:
        return False


# Resolve o binding na importação deste módulo (uma única vez por processo).
if not _try_pyqt6():
    if not _try_pyside6():
        if not _try_pyqt5():
            QT_IMPORT_ERROR = (
                "Nenhum binding Qt disponível. Instale um dos seguintes:\n"
                "  pip install PyQt6           (recomendado — Qt 6)\n"
                "  pip install PySide6         (alternativa oficial Qt Company)\n"
                "  pip install PyQt5           (legado Qt 5)\n"
            )


# ──────────────────────────────────────────────────────────────────────────
# Helpers de enum — Qt6 usa IntEnum sub-namespaced; Qt5 exporta flat.
# Fornecemos constantes estáveis para evitar "if QT_BINDING == ..." em
# toda a aplicação. Quando ``Qt`` não existir (sem binding instalado),
# retornamos ``None`` para evitar AttributeError na import time.
# ──────────────────────────────────────────────────────────────────────────
def _get_attr_chain(root: Any, *names: str) -> Any:
    """Retorna ``root.a.b.c`` para a primeira cadeia existente.

    Em Qt6, ``Qt.AlignmentFlag.AlignLeft`` é o caminho canônico.
    Em Qt5 / PyQt5 com API v1, ``Qt.AlignLeft`` é o caminho flat.
    """
    if root is None:
        return None
    for name in names:
        root = getattr(root, name, None)
        if root is None:
            return None
    return root


ALIGN_LEFT = _get_attr_chain(Qt, "AlignmentFlag", "AlignLeft") or _get_attr_chain(
    Qt, "AlignLeft"
)
ALIGN_CENTER = _get_attr_chain(Qt, "AlignmentFlag", "AlignCenter") or _get_attr_chain(
    Qt, "AlignCenter"
)
ALIGN_RIGHT = _get_attr_chain(Qt, "AlignmentFlag", "AlignRight") or _get_attr_chain(
    Qt, "AlignRight"
)
ALIGN_VCENTER = _get_attr_chain(Qt, "AlignmentFlag", "AlignVCenter") or _get_attr_chain(
    Qt, "AlignVCenter"
)
ORIENT_H = _get_attr_chain(Qt, "Orientation", "Horizontal") or _get_attr_chain(
    Qt, "Horizontal"
)
ORIENT_V = _get_attr_chain(Qt, "Orientation", "Vertical") or _get_attr_chain(
    Qt, "Vertical"
)


# ──────────────────────────────────────────────────────────────────────────
# Locale — ponto como separador decimal (jamais vírgula)
# ──────────────────────────────────────────────────────────────────────────


def enforce_c_locale(app: Optional[Any] = None) -> None:
    """Força o locale C (ponto como separador decimal) em toda a aplicação.

    pt_BR (locale comum em sistemas macOS/Linux no Brasil) usa vírgula
    para separar inteiros e decimais em ``QDoubleSpinBox``. Dados
    científicos devem usar ponto. Este helper:

    1. Altera ``QLocale`` default para ``QLocale::C`` (ponto, sem grupos).
    2. Quando ``app`` é fornecido, também ajusta o ``QApplication`` e
       ``QStyle`` para respeitar o novo default.

    Args:
        app: Instância ``QApplication`` opcional — se ausente, apenas
            ajusta a default do QLocale.
    """
    if QtCore is None:
        return
    # Default classe — respeita por qualquer widget criado após este call
    c_locale = (
        QtCore.QLocale(QtCore.QLocale.Language.C)
        if _get_attr_chain(QtCore.QLocale, "Language", "C")
        else QtCore.QLocale.c() if hasattr(QtCore.QLocale, "c") else QtCore.QLocale()
    )
    try:
        # Remove separador de grupo (1,000,000 → 1000000) em spinboxes.
        try:
            c_locale.setNumberOptions(QtCore.QLocale.NumberOption.OmitGroupSeparator)
        except Exception:
            try:
                c_locale.setNumberOptions(QtCore.QLocale.OmitGroupSeparator)
            except Exception:
                pass
        QtCore.QLocale.setDefault(c_locale)
    except Exception:
        pass
    if app is not None:
        try:
            # PyQt5/6: QApplication herda o QLocale.default automaticamente
            # quando criado após o setDefault. Nenhum setter é requerido.
            pass
        except Exception:
            pass


def make_double_spin(
    default: float,
    lo: float,
    hi: float,
    step: float = 0.1,
    decimals: int = 3,
    suffix: str = "",
) -> Any:
    """``QDoubleSpinBox`` com locale C (ponto) aplicado explicitamente.

    Args:
        default: Valor inicial.
        lo/hi: Limites inferior/superior.
        step: Passo do botão incremento.
        decimals: Número de casas decimais visíveis.
        suffix: Texto anexo (ex.: " m", " Hz").
    """
    if QtWidgets is None:
        return None
    w = QtWidgets.QDoubleSpinBox()
    try:
        c_locale = (
            QtCore.QLocale(QtCore.QLocale.Language.C)
            if _get_attr_chain(QtCore.QLocale, "Language", "C")
            else QtCore.QLocale.c() if hasattr(QtCore.QLocale, "c") else QtCore.QLocale()
        )
        try:
            c_locale.setNumberOptions(QtCore.QLocale.NumberOption.OmitGroupSeparator)
        except Exception:
            try:
                c_locale.setNumberOptions(QtCore.QLocale.OmitGroupSeparator)
            except Exception:
                pass
        w.setLocale(c_locale)
    except Exception:
        pass
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(default)
    if suffix:
        w.setSuffix(suffix)
    return w


def format_float(value: float, precision: int = 4, thousands: bool = False) -> str:
    """Formata float com ponto (nunca vírgula) — universal para labels/tabelas.

    Args:
        value: Valor a formatar.
        precision: Casas decimais.
        thousands: Se ``True``, separa milhares com ``','`` (padrão inglês).

    Returns:
        String com ponto como separador decimal.
    """
    if thousands:
        return f"{value:,.{precision}f}"
    return f"{value:.{precision}f}"


def check_qt_available() -> None:
    """Verifica disponibilidade de binding Qt — aborta com mensagem se ausente.

    Raises:
        SystemExit: Se nenhum binding Qt estiver instalado. A mensagem de
            erro inclui comandos ``pip install`` para cada binding suportado.
    """
    if not QT_AVAILABLE:
        sys.stderr.write("\n[Simulation Manager] ERRO: " + (QT_IMPORT_ERROR or "") + "\n")
        sys.exit(1)


__all__ = [
    "ALIGN_CENTER",
    "ALIGN_LEFT",
    "ALIGN_RIGHT",
    "ALIGN_VCENTER",
    "ORIENT_H",
    "ORIENT_V",
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
    "enforce_c_locale",
    "format_float",
    "make_double_spin",
]
