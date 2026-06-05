# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/qt_compat.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — Camada de Compatibilidade Qt (fundação compartilhada) ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Atualizado  : 2026-04-27 (v2.7a PyQt6+PySide6) · 2026-06-05 (spec 0004 — ║
# ║                relocado de simulation/tests/sm_qt_compat.py p/ gui/, via   ║
# ║                git mv; shim de retrocompat mantido no caminho antigo)      ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: PyQt6 (preferido) | PySide6 (fallback oficial Qt Company)  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Exporta símbolos Qt (QtWidgets, QtCore, QtGui, QThread, Signal, Slot)  ║
# ║    de forma neutra entre PyQt6 e PySide6 — fundação Qt compartilhada       ║
# ║    pelo Simulation Manager E pelo Geosteering AI Studio (sem exigir um     ║
# ║    binding específico).                                                    ║
# ║                                                                           ║
# ║  ORDEM DE RESOLUÇÃO                                                       ║
# ║    1. PyQt6      (recomendado, GPL v3, Python ≥ 3.9)                      ║
# ║    2. PySide6    (fallback, LGPL, Qt Company oficial)                     ║
# ║    Nota: PyQt5 (Qt 5) foi removido em v2.7a — incompatível com Python     ║
# ║          3.13+ e com EOL do ramo open-source em 2023.                     ║
# ║                                                                           ║
# ║  DIFERENÇAS TRATADAS                                                      ║
# ║    • pyqtSignal (PyQt6) vs Signal (PySide6) → exporta como ``Signal``     ║
# ║    • pyqtSlot   (PyQt6) vs Slot   (PySide6) → exporta como ``Slot``       ║
# ║    • Enums Qt6 tipados (Qt.AlignmentFlag.AlignLeft) — use diretamente;    ║
# ║      constantes ALIGN_*/ORIENT_* removidas em v2.7a.                      ║
# ║                                                                           ║
# ║  HELPERS DISPONÍVEIS                                                      ║
# ║    • enforce_c_locale(app)       → força ponto como separador decimal     ║
# ║    • make_double_spin(..)        → QDoubleSpinBox com locale C            ║
# ║    • format_float(value, prec)   → formatação universal com ponto         ║
# ║    • detect_os_dark_mode()       → detecta dark mode do SO via QPalette   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Camada de compatibilidade entre PyQt6 e PySide6 (fundação GUI compartilhada).

Provê imports *unified* para que o Simulation Manager e o Geosteering AI Studio
usem uma única convenção (``Signal``, ``Slot``, ``QT_BINDING``) independente do
binding Qt6 disponível no ambiente.

Helpers para locale C (ponto decimal em dados científicos) e detecção
nativa de dark mode do sistema operacional via ``QPalette`` (Qt6-native).

Example:
    Uso típico em módulos da aplicação::

        >>> from geosteering_ai.gui.qt_compat import (
        ...     QtCore, QtGui, QtWidgets, Signal, Slot, Qt, QT_BINDING
        ... )
        >>> class MyWidget(QtWidgets.QWidget):
        ...     my_sig = Signal(int)
        >>>
        >>> # Enums Qt6 — usar diretamente (não mais constantes ALIGN_*):
        >>> lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

Note:
    Se nenhum binding Qt6 estiver instalado, ``import qt_compat`` ainda
    funciona, mas ``QT_AVAILABLE`` será ``False``. O programa principal
    detecta isso (``check_qt_available``) e apresenta mensagem de instalação clara.
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
    """Tenta importar PySide6 (fallback oficial Qt Company).

    Returns:
        bool: ``True`` se o import foi bem-sucedido.
    """
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


# ── Resolução do binding (uma única vez por processo) ─────────────────────
# Ordem: PyQt6 (preferido) → PySide6 (fallback). PyQt5 removido em v2.7a.
if not _try_pyqt6():
    if not _try_pyside6():
        QT_IMPORT_ERROR = (
            "Nenhum binding Qt6 disponível. Instale um dos seguintes:\n"
            "  pip install PyQt6           (recomendado — GPL v3)\n"
            "  pip install PySide6         (alternativa oficial Qt Company — LGPL)\n"
        )


# ── QtWebEngine — loader sob demanda (módulo Qt PESADO e OPCIONAL) ─────────


def load_qwebengineview() -> Any:
    """Importa ``QWebEngineView`` do binding Qt ativo (PyQt6 ou PySide6).

    ``QtWebEngine`` é um módulo Qt **pesado** (~150 MB, Chromium embutido) e
    **opcional** — por isso NÃO é importado no carregamento de ``qt_compat``.
    Esta função o resolve sob demanda (lazy), respeitando o binding detectado
    em :data:`QT_BINDING`. Único ponto de decisão de binding para QtWebEngine
    (evita ``from PyQt6.QtWebEngineWidgets import …`` hardcoded espalhado).

    Usado pelo backend Plotly (``gui/plot_backends/plotly_canvas.py``), que
    renderiza HTML interativo num ``QWebEngineView``.

    Returns:
        A classe ``QWebEngineView`` do binding Qt ativo.

    Raises:
        ImportError: se nenhum binding Qt6 foi detectado, ou se o módulo
            QtWebEngine não está instalado (ex.: falta ``PyQt6-WebEngine``).

    Note:
        PyQt6 exige o pacote separado ``PyQt6-WebEngine``; PySide6 já o inclui.
    """
    if QT_BINDING not in ("PyQt6", "PySide6"):
        raise ImportError(
            "QtWebEngine indisponível: nenhum binding Qt6 detectado "
            f"(QT_BINDING={QT_BINDING!r})."
        )
    # importlib evita o dual ``from PyQt6/PySide6 import QWebEngineView`` (cujos
    # tipos divergem p/ mypy); ModuleNotFoundError (subclasse de ImportError) é
    # levantado se o módulo QtWebEngine não estiver instalado.
    import importlib

    module = importlib.import_module(f"{QT_BINDING}.QtWebEngineWidgets")
    return module.QWebEngineView


# ── Locale — ponto como separador decimal (jamais vírgula) ────────────────


def _make_c_locale() -> Any:
    """Cria QLocale C (ponto decimal, sem separador de grupos).

    Qt6 garante ``QLocale.Language.C`` como enum tipado — acesso direto
    sem necessidade de fallbacks Qt5.
    """
    if QtCore is None:
        return None
    locale = QtCore.QLocale(QtCore.QLocale.Language.C)
    try:
        locale.setNumberOptions(QtCore.QLocale.NumberOption.OmitGroupSeparator)
    except Exception:
        pass
    return locale


def enforce_c_locale(app: Optional[Any] = None) -> None:  # noqa: ARG001
    """Força o locale C (ponto como separador decimal) em toda a aplicação.

    pt_BR (locale comum em sistemas macOS/Linux no Brasil) usa vírgula
    para separar inteiros e decimais em ``QDoubleSpinBox``. Dados
    científicos devem usar ponto. Este helper:

    1. Altera ``QLocale`` default para ``QLocale::C`` (ponto, sem grupos).
    2. Qualquer widget criado após este call herda o locale automaticamente.

    Args:
        app: Instância ``QApplication`` opcional — aceita para assinatura
            compatível, mas Qt6 aplica o default globalmente sem precisar
            ajustar o QApplication individualmente.
    """
    if QtCore is None:
        return
    locale = _make_c_locale()
    if locale is not None:
        try:
            QtCore.QLocale.setDefault(locale)
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
    """``QDoubleSpinBox`` com locale C (ponto decimal) aplicado explicitamente.

    Args:
        default: Valor inicial.
        lo: Limite inferior.
        hi: Limite superior.
        step: Passo do botão de incremento.
        decimals: Número de casas decimais visíveis.
        suffix: Texto anexo ao valor (ex.: ``" m"``, ``" Hz"``).

    Returns:
        ``QDoubleSpinBox`` configurado, ou ``None`` se QtWidgets não estiver
        disponível.
    """
    if QtWidgets is None:
        return None
    w = QtWidgets.QDoubleSpinBox()
    locale = _make_c_locale()
    if locale is not None:
        try:
            w.setLocale(locale)
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
    """Formata float com ponto (nunca vírgula) — universal para labels e tabelas.

    Args:
        value: Valor a formatar.
        precision: Casas decimais.
        thousands: Se ``True``, separa milhares com ``','`` (padrão inglês).

    Returns:
        String com ponto como separador decimal, independente do locale do SO.
    """
    if thousands:
        return f"{value:,.{precision}f}"
    return f"{value:.{precision}f}"


def detect_os_dark_mode() -> bool:
    """Detecta dark mode do sistema operacional via QPalette (Qt6-native).

    Usa a cor de fundo da janela do sistema (``QPalette.ColorRole.Window``)
    para determinar se o SO está em modo escuro. Luminância < 128 indica
    tema escuro ativo.

    Returns:
        ``True`` se o SO estiver em dark mode, ``False`` caso contrário ou
        se não houver ``QApplication`` ativa.

    Note:
        Requer que ``QApplication`` já tenha sido criada. Retorna ``False``
        se chamado antes da criação da aplicação.
    """
    if QtWidgets is None or QtGui is None:
        return False
    app = QtWidgets.QApplication.instance()
    if app is None:
        return False
    try:
        palette = app.palette()
        window_color = palette.color(QtGui.QPalette.ColorRole.Window)
        # Luminância < 128 → fundo escuro → dark mode ativo no SO.
        # bool(): Qt é não-tipado (lightness() é Any); explicita o retorno bool (mypy).
        return bool(window_color.lightness() < 128)
    except Exception:
        return False


def check_qt_available() -> None:
    """Verifica disponibilidade de binding Qt6 — aborta com mensagem se ausente.

    Raises:
        SystemExit: Se nenhum binding Qt6 estiver instalado. A mensagem de
            erro inclui comandos ``pip install`` para PyQt6 e PySide6.
    """
    if not QT_AVAILABLE:
        # Módulo COMPARTILHADO (SM + Studio) → prefixo genérico (spec 0004).
        sys.stderr.write(
            "\n[Geosteering AI GUI] ERRO: " + (QT_IMPORT_ERROR or "") + "\n"
        )
        sys.exit(1)


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
