# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/theme/tokens.py                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ThemeTokens — paleta/tipografia do tema (PURO, sem Qt)     ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — theme (estética profissional)                        ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de tema                                ║
# ║  Framework   : stdlib PURO — NÃO importa Qt (testável sem binding)         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Tokens de design (cores, tipografia, raios) do tema — a fonte única    ║
# ║    de verdade da paleta. O QSS (stylesheet.py) é GERADO a partir destes    ║
# ║    tokens, permitindo trocar a paleta (dark/light/custom) sem reescrever   ║
# ║    o CSS. Paleta padrão inspirada no Google Antigravity (dark pro).        ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ThemeTokens · ANTIGRAVITY_DARK                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ThemeTokens`` — paleta/tipografia (PURO) p/ gerar o QSS do tema (Antigravity)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ThemeTokens", "ANTIGRAVITY_DARK"]


@dataclass(frozen=True)
class ThemeTokens:
    """Tokens de design do tema (cores ``#rrggbb``, tipografia, raios).

    Attributes:
        bg_primary: fundo da janela/diálogos (mais escuro).
        bg_secondary: superfícies elevadas (inputs, painéis, botões).
        bg_tertiary: linhas alternadas / superfície sutil.
        border: bordas e separadores.
        text_primary: texto base.
        text_secondary: labels/hints.
        text_muted: texto desabilitado/placeholder.
        accent: acento primário (foco, seleção, botão primário).
        accent_hover: acento em hover.
        accent_pressed: acento pressionado.
        success/warning/error: cores semânticas de estado.
        selection_bg: fundo de seleção de texto/itens.
        font_family: pilha de fontes (Segoe UI → -apple-system → sans-serif).
        font_size_base: tamanho base (px).
        radius/radius_sm: raios de canto (px).

    Note:
        PURO (sem Qt) — testável isoladamente. A paleta padrão
        :data:`ANTIGRAVITY_DARK` segue o estilo do Google Antigravity.
    """

    bg_primary: str = "#1a1a1a"
    bg_secondary: str = "#2d2d30"
    bg_tertiary: str = "#252526"
    border: str = "#404040"
    text_primary: str = "#e0e0e0"
    text_secondary: str = "#a8a8a8"
    text_muted: str = "#6b6b6b"
    accent: str = "#4f46e5"
    accent_hover: str = "#6366f1"
    accent_pressed: str = "#4338ca"
    success: str = "#10b981"
    warning: str = "#f59e0b"
    error: str = "#ef4444"
    selection_bg: str = "#3730a3"
    font_family: str = '"Segoe UI", -apple-system, "Helvetica Neue", Arial, sans-serif'
    font_size_base: int = 13
    radius: int = 6
    radius_sm: int = 4


# Paleta padrão — dark profissional (Google Antigravity).
ANTIGRAVITY_DARK = ThemeTokens()
