# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/placeholder.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : PlaceholderPerspective — scaffold "em breve" na rail       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell MVVM (spec 0013)                               ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de shell                               ║
# ║  Framework   : Qt6 via gui.qt_compat (build_view) + Perspective ABC        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Perspectiva-scaffold (``enabled = False``) que ocupa um item CINZA na    ║
# ║    activity rail — torna a ORGANIZAÇÃO de todos os recursos do SM visível    ║
# ║    antes de cada um ser implementado (Resultados/Benchmark/Análise/         ║
# ║    Preferências → Fatias 6c-6i). Importa Qt (build_view) → NÃO re-exportada ║
# ║    por ``gui.shell`` (Qt-free).                                            ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    PlaceholderPerspective                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``PlaceholderPerspective`` — scaffold "em breve" na activity rail (spec 0013)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.shell.perspective import Perspective
from geosteering_ai.gui.viewmodels.base import BaseViewModel

if TYPE_CHECKING:  # pragma: no cover — só type-checking
    from geosteering_ai.gui.qt_compat import QtWidgets

__all__ = ["PlaceholderPerspective"]


class PlaceholderPerspective(Perspective):
    """Perspectiva-scaffold desabilitada (item cinza na rail; view "em breve").

    Args:
        id: chave estável da perspectiva (ex.: ``"results"``).
        title: rótulo (tooltip da rail; ex.: ``"Resultados"``).
        icon_glyph: caractere exibido na rail (ex.: ``"📊"``).
        order: posição relativa na rail.
        roadmap: rótulo da fatia que a implementará (ex.: ``"Fatia 6f"``).

    Note:
        ``enabled = False`` → o host (``AntigravityMainWindow``) cria um item de
        rail NÃO-clicável. Como nunca é ativada, sua View "em breve" raramente é
        construída; ainda assim é leve (um rótulo centralizado).
    """

    enabled = False

    def __init__(
        self, *, id: str, title: str, icon_glyph: str, order: int, roadmap: str = ""
    ) -> None:
        self.id = id
        self.title = title
        self.icon_glyph = icon_glyph
        self.order = order
        self._roadmap = roadmap

    def build_viewmodel(self, ctx: AppContext) -> BaseViewModel:
        """ViewModel trivial (vazio) — o scaffold não tem estado."""
        return BaseViewModel()

    def build_view(self, ctx: AppContext) -> "QtWidgets.QWidget":
        """View "em breve" — um rótulo centralizado (sem lógica)."""
        from geosteering_ai.gui.qt_compat import Qt, QtWidgets

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sufixo = f"\n({self._roadmap})" if self._roadmap else ""
        label = QtWidgets.QLabel(
            f"{self.icon_glyph}  {self.title}\n\nEm breve.{sufixo}"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setProperty("role", "hint")
        layout.addWidget(label)
        return page
