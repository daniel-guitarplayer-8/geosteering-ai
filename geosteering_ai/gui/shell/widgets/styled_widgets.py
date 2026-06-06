# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/widgets/styled_widgets.py                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : helpers de estilo Antigravity (cards, pill, sombra)        ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell widgets (spec 0013)                            ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de shell                               ║
# ║  Framework   : Qt6 via gui.qt_compat (QtWidgets + QtGui)                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Helpers para a estética Antigravity: ``make_card`` (QGroupBox elevado   ║
# ║    com sombra), ``make_pill_button`` (botão com propriedade ``role`` →      ║
# ║    QSS primary/danger/ghost) e ``apply_shadow`` (QGraphicsDropShadowEffect).║
# ║    A cor/raio vêm do QSS global (tokens) — aqui só estrutura + sombra.      ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    make_card · make_pill_button · apply_shadow                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Helpers de estilo Antigravity — cards, pill buttons e sombra (spec 0013)."""

from __future__ import annotations

from typing import Any, Optional

from geosteering_ai.gui.qt_compat import QtGui, QtWidgets

__all__ = ["make_card", "make_pill_button", "apply_shadow"]


def apply_shadow(widget: Any, *, blur: int = 18, dy: int = 3, alpha: int = 120) -> Any:
    """Aplica uma sombra suave (``QGraphicsDropShadowEffect``) — cara de card elevado.

    QtWidgets/QSS não suportam ``box-shadow``; este é o workaround nativo Qt para
    o efeito de elevação do estilo Antigravity (sombra projetada para baixo).

    Args:
        widget: o widget a receber a sombra.
        blur: raio de desfoque da sombra (px).
        dy: deslocamento vertical (px) — sombra projetada para baixo.
        alpha: opacidade da sombra (0-255).

    Returns:
        O próprio ``widget`` (encadeável).
    """
    effect = QtWidgets.QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setColor(QtGui.QColor(0, 0, 0, alpha))
    effect.setOffset(0, dy)
    widget.setGraphicsEffect(effect)
    return widget


def make_card(title: str = "", parent: Optional[Any] = None) -> Any:
    """Cria um ``QGroupBox`` estilo card Antigravity (superfície elevada + sombra).

    A cor/borda/raio vêm do QSS global (tokens); aqui só montamos o container e
    aplicamos a sombra de elevação.

    Args:
        title: título do card (rótulo do QGroupBox).
        parent: widget pai opcional.

    Returns:
        O ``QGroupBox`` configurado (adicione um layout com o conteúdo).
    """
    box = QtWidgets.QGroupBox(title, parent)
    apply_shadow(box)
    return box


def make_pill_button(text: str, *, role: str = "", parent: Optional[Any] = None) -> Any:
    """Cria um ``QPushButton`` pill com a propriedade dinâmica ``role`` (p/ QSS).

    O QSS global estiliza ``QPushButton[role="primary"|"danger"|"ghost"]``. A
    propriedade é definida ANTES do ``show`` (o polish do estilo a aplica).

    Args:
        text: rótulo do botão.
        role: ``"primary"`` (ação principal, accent), ``"danger"`` (destrutivo),
            ``"ghost"`` (texto/transparente) ou ``""`` (padrão).
        parent: widget pai opcional.

    Returns:
        O ``QPushButton`` configurado.
    """
    btn = QtWidgets.QPushButton(text, parent)
    if role:
        btn.setProperty("role", role)
    return btn
