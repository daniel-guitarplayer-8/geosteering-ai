# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_toast.py                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Toast Notifications                   ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Status      : Produção (v2.6)                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Notificações não-bloqueantes para eventos não-críticos. Substitui      ║
# ║    QMessageBox modal (que bloqueia UI thread) para casos como:            ║
# ║    "Simulação concluída", "Cache evictado", "Figura salva".               ║
# ║                                                                           ║
# ║    QMessageBox CONTINUA em uso para:                                      ║
# ║      • Erros críticos (parâmetros inválidos, falha I/O)                   ║
# ║      • Confirmações destrutivas (limpar histórico, sobrescrita)           ║
# ║                                                                           ║
# ║  COMPORTAMENTO                                                            ║
# ║    Posiciona automaticamente no canto inferior direito do parent.         ║
# ║    Fade-in (300ms) → display (X ms) → fade-out (300ms) → auto-destroy.    ║
# ║    Múltiplos toasts empilham verticalmente (max 3 visíveis).              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""ToastNotification não-bloqueante para v2.6."""
from __future__ import annotations

from typing import Dict, List, Optional

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    _QT = "PyQt6"
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

        _QT = "PyQt5"
    except ImportError:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

        _QT = "PySide6"

__all__ = ["ToastNotification", "ToastManager"]

# Cores VSCode-style por nível
_LEVEL_COLORS: Dict[str, str] = {
    "info": "#007acc",  # azul
    "success": "#4ec9b0",  # verde
    "warning": "#dcdcaa",  # amarelo
    "error": "#f48771",  # vermelho
}


class ToastNotification(QtWidgets.QLabel):
    """Notificação flutuante não-bloqueante com fade in/out.

    Attributes:
        level: ``"info"``, ``"success"``, ``"warning"`` ou ``"error"``.
        duration_ms: tempo total de display (excluindo fade).
    """

    closed = QtCore.pyqtSignal() if _QT == "PyQt6" or _QT == "PyQt5" else QtCore.Signal()

    def __init__(
        self,
        message: str,
        level: str = "info",
        duration_ms: int = 3000,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._level = level
        self._duration_ms = duration_ms
        self.setText(message)
        self.setWordWrap(True)
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)
        self._apply_style()
        # Window flags para flutuar acima do parent sem bordas
        flags = (
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

    def _apply_style(self) -> None:
        bg = _LEVEL_COLORS.get(self._level, _LEVEL_COLORS["info"])
        # Texto escuro em fundos claros, texto claro em fundos escuros
        fg = "#1e1e1e" if self._level in ("success", "warning") else "#ffffff"
        self.setStyleSheet(
            f"QLabel {{"
            f"  background: {bg};"
            f"  color: {fg};"
            f"  padding: 12px 16px;"
            f"  border-radius: 6px;"
            f"  font-size: 12px;"
            f"  font-weight: 500;"
            f"}}"
        )

    def show_at_corner(self, parent: QtWidgets.QWidget, offset_y: int = 0) -> None:
        """Posiciona no canto inferior direito do parent + offset vertical."""
        self.adjustSize()
        if parent is not None:
            geo = parent.geometry()
            x = parent.x() + geo.width() - self.width() - 24
            y = parent.y() + geo.height() - self.height() - 48 - offset_y
            self.move(x, y)
        # Fade-in via animação de opacity
        self._fade_in()

    def _fade_in(self) -> None:
        eff = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(eff)
        anim = QtCore.QPropertyAnimation(eff, b"opacity", self)
        anim.setDuration(300)
        anim.setStartValue(0.0)
        anim.setEndValue(0.95)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.start()
        self._anim_in = anim
        self.show()
        QtCore.QTimer.singleShot(self._duration_ms, self._fade_out)

    def _fade_out(self) -> None:
        eff = self.graphicsEffect()
        if eff is None:
            self.hide()
            self.deleteLater()
            self.closed.emit()
            return
        anim = QtCore.QPropertyAnimation(eff, b"opacity", self)
        anim.setDuration(300)
        anim.setStartValue(0.95)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        anim.finished.connect(self._on_faded)
        anim.start()
        self._anim_out = anim

    def _on_faded(self) -> None:
        self.hide()
        self.closed.emit()
        self.deleteLater()


class ToastManager:
    """Gerenciador de fila de toasts (max 3 visíveis simultaneamente).

    Empilha toasts verticalmente — o mais recente fica no topo.
    Toasts adicionais ficam em fila e aparecem conforme os antigos somem.
    """

    MAX_VISIBLE = 3

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        self._parent = parent
        self._visible: List[ToastNotification] = []
        self._queue: List[tuple] = []  # (message, level, duration_ms)

    def show(
        self,
        message: str,
        level: str = "info",
        duration_ms: int = 3000,
    ) -> None:
        """Exibe um toast. Se houver MAX_VISIBLE ativos, enfileira."""
        if len(self._visible) >= self.MAX_VISIBLE:
            self._queue.append((message, level, duration_ms))
            return
        toast = ToastNotification(message, level, duration_ms, parent=self._parent)
        toast.closed.connect(lambda: self._on_closed(toast))
        self._visible.append(toast)
        self._reposition()

    def _reposition(self) -> None:
        for i, toast in enumerate(self._visible):
            offset_y = i * (toast.sizeHint().height() + 8)
            toast.show_at_corner(self._parent, offset_y)

    def _on_closed(self, toast: ToastNotification) -> None:
        if toast in self._visible:
            self._visible.remove(toast)
        self._reposition()
        if self._queue:
            msg, lvl, dur = self._queue.pop(0)
            self.show(msg, lvl, dur)

    def clear(self) -> None:
        for toast in list(self._visible):
            toast.hide()
            toast.deleteLater()
        self._visible.clear()
        self._queue.clear()
