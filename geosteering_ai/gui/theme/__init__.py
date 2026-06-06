# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/theme/__init__.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Pacote      : gui.theme — tema visual centralizado (PURO, sem Qt)        ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — estética profissional (tipo Google Antigravity)      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Tema centralizado: tokens (paleta/tipografia) → QSS gerado → aplicado   ║
# ║    na QApplication. Tudo PURO (Qt-free) exceto o ``app`` duck-typed.        ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ThemeTokens · ANTIGRAVITY_DARK · generate_qss · apply_theme            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``gui.theme`` — tema visual centralizado (tokens → QSS → apply_theme)."""

from __future__ import annotations

from geosteering_ai.gui.theme.manager import apply_theme
from geosteering_ai.gui.theme.stylesheet import generate_qss
from geosteering_ai.gui.theme.tokens import ANTIGRAVITY_DARK, ThemeTokens

__all__ = ["ThemeTokens", "ANTIGRAVITY_DARK", "generate_qss", "apply_theme"]
