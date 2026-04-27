# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_widgets.py                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Widgets reutilizáveis                 ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Atualizado  : 2026-04-27 (v2.7a — sm_qt_compat + QPropertyAnimation)    ║
# ║  Status      : Produção (v2.7a)                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Widgets Qt reutilizáveis usados em múltiplos pontos do                 ║
# ║    Simulation Manager. Hoje contém:                                       ║
# ║      • CollapsibleGroupBox — Wrapper que toggleia visibilidade do         ║
# ║        conteúdo via QToolButton (▼/▶) com animação suave de altura       ║
# ║        (QPropertyAnimation, 180ms, easing OutCubic).                     ║
# ║                                                                           ║
# ║  MUDANÇAS v2.7a                                                           ║
# ║    • Imports migrados para sm_qt_compat (suporta PyQt6 e PySide6)        ║
# ║    • QPropertyAnimation em _on_toggled() — animação de 180ms             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Widgets Qt reutilizáveis (v2.7a)."""
from __future__ import annotations

from typing import Optional

from .sm_qt_compat import QtCore, QtWidgets, Signal

__all__ = ["CollapsibleGroupBox"]

# ── Signal compatível com PyQt6 e PySide6 ────────────────────────────────
# sm_qt_compat exporta Signal = pyqtSignal (PyQt6) ou Signal (PySide6).
# Usado como atributo de classe na definição abaixo.
_Signal = Signal  # alias para clareza dentro da definição da classe


class CollapsibleGroupBox(QtWidgets.QWidget):
    """QGroupBox-like com toggle expand/collapse via QToolButton.

    Expande/colapsa o conteúdo com animação suave (``QPropertyAnimation``
    sobre ``maximumHeight``, 180ms, easing ``OutCubic``).

    Uso::

        box = CollapsibleGroupBox("Filtros avançados", collapsed=True)
        box.setContentLayout(my_form_layout)  # OU
        box.addWidget(my_widget)
        parent_layout.addWidget(box)

    Attributes:
        toggled: Signal emitido quando o estado expand/collapse muda.
            Emite ``True`` para expanded, ``False`` para collapsed.

    Note:
        A animação usa ``maximumHeight`` — o widget precisa ter um
        ``sizeHint()`` definido pelo conteúdo para calcular a altura alvo.
        Em formulários complexos, ``sizeHint().height()`` pode ser 0 antes
        do primeiro paint; neste caso a animação degenera para um setVisible
        imediato (fallback seguro).
    """

    toggled = _Signal(bool)

    def __init__(
        self,
        title: str,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        collapsed: bool = False,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._anim: Optional[QtCore.QPropertyAnimation] = None  # referência anti-GC

        # Toggle button — actua como header clicável
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

        # Container do conteúdo colapsável
        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 4, 6, 4)
        self._content_layout.setSpacing(6)
        self._content.setVisible(not collapsed)

        # Leve borda esquerda para separação visual
        self._content.setStyleSheet(
            "QWidget { border-left: 2px solid rgba(127,127,127,0.25); }"
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
        old = self._content.layout()
        if old is not None:
            QtWidgets.QWidget().setLayout(old)  # transfere para órfão → GC
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
        """Expande ou colapsa com animação QPropertyAnimation (180ms)."""
        self._update_toggle_text()

        # Calcular altura alvo antes de mostrar o widget
        target_h = self._content.sizeHint().height()

        if checked:
            # Expandir: tornar visível antes de animar (sizeHint precisa do widget)
            self._content.setVisible(True)
            self._content.setMaximumHeight(0)
            target_h = max(target_h, self._content.sizeHint().height())
        else:
            target_h = 0

        # Iniciar animação — cancelar eventual animação anterior
        if self._anim is not None:
            self._anim.stop()

        if target_h > 0 or not checked:
            anim = QtCore.QPropertyAnimation(self._content, b"maximumHeight")
            anim.setDuration(180)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            anim.setStartValue(
                self._content.maximumHeight() if checked else self._content.height()
            )
            anim.setEndValue(target_h if checked else 0)
            if not checked:
                # Ocultar após a animação terminar
                anim.finished.connect(lambda: self._content.setVisible(False))
                anim.finished.connect(lambda: self._content.setMaximumHeight(16777215))
            else:
                # Remover limitação de altura ao final da expansão
                anim.finished.connect(lambda: self._content.setMaximumHeight(16777215))
            anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
            self._anim = anim  # manter referência contra GC prematuro
        else:
            # Fallback sem animação (sizeHint=0 antes do primeiro paint)
            self._content.setVisible(checked)
            self._content.setMaximumHeight(16777215)

        try:
            self.toggled.emit(checked)
        except Exception:
            pass
