# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/theme/manager.py                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : apply_theme — aplica o QSS do tema na QApplication         ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — theme (estética profissional)                        ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de tema                                ║
# ║  Framework   : PURO (app duck-typed — só usa setStyleSheet; sem importar Qt)║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Aplica a folha de estilo (QSS) global na ``QApplication`` a partir dos  ║
# ║    tokens. O ``app`` é DUCK-TYPED (só precisa de ``setStyleSheet``) — o     ║
# ║    módulo permanece Qt-free (testável com um stub).                        ║
# ║                                                                           ║
# ║  Note                                                                     ║
# ║    Os canvases de plot (matplotlib/pyqtgraph) NÃO são QWidget alcançáveis  ║
# ║    pela árvore — o fundo escuro deles é ajustado por ``set_dark_mode(True)``║
# ║    na própria View ao criar o canvas (ver ResultsView).                    ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    apply_theme                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``apply_theme(app, tokens)`` — aplica o QSS do tema na QApplication (duck-typed)."""

from __future__ import annotations

from typing import Any

from geosteering_ai.gui.theme.stylesheet import generate_qss
from geosteering_ai.gui.theme.tokens import ANTIGRAVITY_DARK, ThemeTokens

__all__ = ["apply_theme"]


def apply_theme(app: Any, tokens: ThemeTokens = ANTIGRAVITY_DARK) -> None:
    """Aplica o tema (QSS global) na ``QApplication``.

    Args:
        app: a ``QApplication`` (duck-typed: precisa de ``setStyleSheet(str)``);
            ``None`` é no-op (ex.: sem binding Qt).
        tokens: paleta a usar (default :data:`ANTIGRAVITY_DARK`).
    """
    if app is None:
        return
    app.setStyleSheet(generate_qss(tokens))
