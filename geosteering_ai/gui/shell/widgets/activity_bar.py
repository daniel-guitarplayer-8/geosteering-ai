# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/widgets/activity_bar.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ActivityBar — nav rail vertical (estilo Antigravity)       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell widgets (spec 0013)                            ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de shell                               ║
# ║  Framework   : Qt6 via gui.qt_compat (QToolButton autoExclusive)           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Barra de navegação vertical (icons-only) à esquerda — o "activity bar"  ║
# ║    do Google Antigravity IDE. Cada perspectiva é um ``QToolButton``        ║
# ║    checkable+autoExclusive; clicar emite ``selected(index)``. Itens        ║
# ║    desabilitados (scaffold "em breve") aparecem cinza e não selecionáveis.  ║
# ║    A estética (fundo, accent na seleção) vem do QSS ``#ActivityBar``.       ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ActivityBar                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ActivityBar`` — nav rail vertical por ícones (estilo Antigravity, spec 0013)."""

from __future__ import annotations

from typing import Any, List, Optional

from geosteering_ai.gui.qt_compat import Qt, QtGui, QtWidgets, Signal

__all__ = ["ActivityBar"]


class ActivityBar(QtWidgets.QWidget):  # type: ignore[misc] # QtWidgets é Any (qt_compat)
    """Barra de navegação vertical (activity rail) — emite ``selected(index)``.

    Cada item é um ``QToolButton`` checkable + autoExclusive (rádio visual). A
    ordem dos itens segue a ordem de inserção (alinhada ao ``QStackedWidget`` do
    host). Itens desabilitados (``enabled=False``) são scaffolds não-clicáveis.

    Signals:
        selected(int): emitido ao clicar num item habilitado (índice lógico).

    Note:
        ``objectName == "ActivityBar"`` — o QSS global o estiliza (fundo escuro,
        accent na borda esquerda do item selecionado). Sem lógica de domínio.
    """

    selected = Signal(int)

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ActivityBar")
        self._buttons: List[Any] = []
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 6, 0, 6)
        self._layout.setSpacing(2)
        self._layout.addStretch(1)  # empurra os itens para o topo

    def insert_item(
        self, index: int, glyph: str, tooltip: str, *, enabled: bool = True
    ) -> None:
        """Insere um item de navegação na posição ``index`` (mantém alinhamento).

        Args:
            index: posição lógica (igual ao índice no ``QStackedWidget`` do host).
            glyph: caractere/ícone exibido (ex.: ``"🧪"``).
            tooltip: rótulo da perspectiva (ex.: ``"Simulação"``).
            enabled: ``False`` → scaffold cinza, não-clicável ("em breve").
        """
        btn = QtWidgets.QToolButton(self)
        btn.setText(glyph)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setAutoExclusive(True)
        btn.setEnabled(enabled)
        btn.setFixedSize(48, 44)
        font = btn.font()
        font.setPointSize(15)
        btn.setFont(font)
        btn.setCursor(QtGui.QCursor(Qt.CursorShape.PointingHandCursor))
        # ``clicked`` recomputa o índice ATUAL do botão (estável: sem remoção) →
        # imune a deslocamento por inserções fora de ordem.
        btn.clicked.connect(lambda _checked=False, b=btn: self._on_clicked(b))
        # Insere ANTES do stretch (última posição do layout).
        self._layout.insertWidget(index, btn)
        self._buttons.insert(index, btn)

    def set_current(self, index: int) -> None:
        """Marca o item ``index`` como selecionado (sem emitir ``selected``)."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)

    def _on_clicked(self, btn: Any) -> None:
        """Slot interno — emite ``selected`` com o índice atual do botão clicado."""
        # Guard defensivo: se o botão não está mais na lista (limpeza futura/GC),
        # ignora em vez de estourar ValueError num slot Qt (crasharia a UI).
        if btn in self._buttons:
            self.selected.emit(self._buttons.index(btn))
