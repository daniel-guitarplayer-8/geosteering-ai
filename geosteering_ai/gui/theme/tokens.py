# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/theme/tokens.py                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ThemeTokens — paleta/tipografia do tema (PURO, sem Qt)     ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — theme (estética profissional Google Antigravity)     ║
# ║  Versão      : v0.2 (spec 0013 — escala de espaçamento/raio + Antigravity) ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de tema                                ║
# ║  Framework   : stdlib PURO — NÃO importa Qt (testável sem binding)         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Tokens de design (cores, tipografia, espaçamento, raios) — fonte única ║
# ║    de verdade da paleta. O QSS (stylesheet.py) é GERADO destes tokens,     ║
# ║    permitindo trocar a paleta sem reescrever o CSS. Paleta padrão          ║
# ║    :data:`ANTIGRAVITY_DARK` inspirada no Google Antigravity IDE (dark      ║
# ║    midnight + accent índigo/roxo, cards elevados, botões pill).            ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ThemeTokens · ANTIGRAVITY_DARK                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ThemeTokens`` — paleta/tipografia/espaçamento (PURO) p/ gerar o QSS Antigravity."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ThemeTokens", "ANTIGRAVITY_DARK"]


@dataclass(frozen=True)
class ThemeTokens:
    """Tokens de design do tema (cores ``#rrggbb``, tipografia, espaçamento, raios).

    Inspiração: Google Antigravity IDE (dark midnight + accent índigo/roxo). As
    superfícies sobem em luminância (``bg_primary`` < ``bg_secondary`` <
    ``bg_tertiary``) para dar a sensação de cards elevados.

    Attributes:
        bg_primary: fundo da janela/diálogos (midnight, mais escuro).
        bg_secondary: superfícies elevadas (cards/inputs/painéis).
        bg_tertiary: hover/linha alternada (sutil acima do secondary).
        bg_rail: fundo da activity rail / barras de navegação.
        border: bordas e separadores (sutis, levemente índigo).
        text_primary/secondary/muted: hierarquia de texto (contraste WCAG-aware).
        accent/accent_hover/accent_pressed: índigo primário (foco, seleção, botão).
        accent_light: superfície índigo profunda (botão secundário/realce sutil).
        success/warning/error: cores semânticas de estado.
        selection_bg: fundo de seleção de texto/itens.
        font_family/font_family_mono: pilhas de fonte (UI / monoespaçada p/ log).
        font_size_sm/base/lg: tamanhos (px) — hints / corpo / títulos.
        spacing_xs..xxl: escala de espaçamento (px) — 2/4/8/12/16/24.
        radius_sm/radius/radius_md/radius_lg: raios de canto (px) — 4/6/8/12.

    Note:
        PURO (sem Qt) — testável isoladamente. Campos legados (``radius``,
        ``font_size_base``) preservados p/ compat; novos campos são aditivos.
    """

    # ── Cores base (Antigravity dark midnight) ────────────────────────────
    bg_primary: str = "#131217"
    bg_secondary: str = "#1c1b24"
    bg_tertiary: str = "#26242f"
    bg_rail: str = "#0f0e13"
    border: str = "#2e2c3a"
    text_primary: str = "#e6e5ec"
    text_secondary: str = "#aeacbd"
    text_muted: str = "#7c7a8c"
    # ── Accent índigo/roxo (identidade Antigravity) ───────────────────────
    accent: str = "#4f46e5"
    accent_hover: str = "#6366f1"
    accent_pressed: str = "#4338ca"
    accent_light: str = "#312e81"
    # ── Semânticas ────────────────────────────────────────────────────────
    success: str = "#10b981"
    warning: str = "#f59e0b"
    error: str = "#ef4444"
    selection_bg: str = "#3730a3"
    # ── Tipografia ──────────────────────────────────────────────────────
    font_family: str = '"Segoe UI", -apple-system, "Helvetica Neue", Arial, sans-serif'
    font_family_mono: str = '"JetBrains Mono", "Menlo", "Consolas", "Monaco", monospace'
    font_size_sm: int = 11
    font_size_base: int = 13
    font_size_lg: int = 16
    # ── Espaçamento (escala) ──────────────────────────────────────────────
    spacing_xs: int = 2
    spacing_sm: int = 4
    spacing_md: int = 8
    spacing_lg: int = 12
    spacing_xl: int = 16
    spacing_xxl: int = 24
    # ── Raios de canto ────────────────────────────────────────────────────
    radius_sm: int = 4
    radius: int = 6
    radius_md: int = 8
    radius_lg: int = 12


# Paleta padrão — dark profissional inspirada no Google Antigravity IDE.
ANTIGRAVITY_DARK = ThemeTokens()
