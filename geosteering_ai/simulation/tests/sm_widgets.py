# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_widgets.py                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Widgets reutilizáveis                 ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Status      : Produção (v2.6b)                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Widgets Qt reutilizáveis usados em múltiplos pontos do                 ║
# ║    Simulation Manager. Hoje contém:                                       ║
# ║      • CollapsibleGroupBox — Wrapper que toggleia visibilidade do         ║
# ║        conteúdo via QToolButton (▼/▶) com animação de altura.            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Widgets Qt reutilizáveis (v2.6b)."""
from __future__ import annotations

from typing import Optional

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

__all__ = ["CollapsibleGroupBox"]


class CollapsibleGroupBox(QtWidgets.QWidget):
    """QGroupBox-like com toggle expand/collapse via QToolButton.

    Uso:
        box = CollapsibleGroupBox("Filtros avançados", collapsed=True)
        box.setContentLayout(my_form_layout)  # OU
        box.addWidget(my_widget)
        parent_layout.addWidget(box)

    Attributes:
        toggled: Signal emitido quando o estado expand/collapse muda.
            Emite ``True`` para expanded, ``False`` para collapsed.

    Note:
        Usa ``setVisible(False)`` no conteúdo (sem animação por padrão para
        evitar reflow assíncrono em formulários complexos). A animação pode
        ser adicionada via ``QPropertyAnimation`` em ``maximumHeight`` no
        método ``_on_toggled``.
    """

    if _QT in ("PyQt6", "PyQt5"):
        toggled = QtCore.pyqtSignal(bool)  # type: ignore[attr-defined]
    else:
        toggled = QtCore.Signal(bool)  # type: ignore[attr-defined]

    def __init__(
        self,
        title: str,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        collapsed: bool = False,
    ) -> None:
        super().__init__(parent)
        self._title = title

        # Toggle button — actua como header
        self._toggle = QtWidgets.QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setStyleSheet(
            "QToolButton {"
            "  border: none;"
            "  text-align: left;"
            "  font-weight: bold;"
            "  padding: 4px 6px;"
            "  background: transparent;"
            "}"
            "QToolButton:hover { background: rgba(127,127,127,0.08); }"
        )
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._update_toggle_text()
        self._toggle.toggled.connect(self._on_toggled)

        # Container do conteúdo
        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 4, 6, 4)
        self._content_layout.setSpacing(6)
        self._content.setVisible(not collapsed)

        # Frame com leve borda esquerda para separação visual
        self._content.setStyleSheet(
            "QWidget {" "  border-left: 2px solid rgba(127,127,127,0.25);" "}"
        )

        # Layout raiz (vertical)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._toggle)
        v.addWidget(self._content)

    # ── API pública ──────────────────────────────────────────────────────

    def setContentLayout(self, layout: "QtWidgets.QLayout") -> None:
        """Define um layout para o conteúdo (substitui qualquer existente)."""
        # Remove layout anterior
        old = self._content.layout()
        if old is not None:
            QtWidgets.QWidget().setLayout(old)  # transfere para órfão (será GC)
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(layout)
        self._content_layout.addWidget(wrapper)

    def addWidget(self, w: "QtWidgets.QWidget") -> None:
        """Adiciona widget ao conteúdo colapsável."""
        self._content_layout.addWidget(w)

    def addLayout(self, layout: "QtWidgets.QLayout") -> None:
        """Adiciona um layout ao conteúdo colapsável."""
        self._content_layout.addLayout(layout)

    def setCollapsed(self, collapsed: bool) -> None:
        """Define explicitamente o estado collapsed/expanded."""
        self._toggle.setChecked(not collapsed)

    def isCollapsed(self) -> bool:
        return not self._toggle.isChecked()

    def title(self) -> str:
        return self._title

    def setTitle(self, title: str) -> None:
        self._title = title
        self._update_toggle_text()

    # ── Internos ─────────────────────────────────────────────────────────

    def _update_toggle_text(self) -> None:
        marker = "▼" if self._toggle.isChecked() else "▶"
        self._toggle.setText(f"{marker} {self._title}")

    def _on_toggled(self, checked: bool) -> None:
        self._update_toggle_text()
        self._content.setVisible(checked)
        try:
            self.toggled.emit(checked)
        except Exception:
            pass
