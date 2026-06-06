# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/theme/stylesheet.py                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : generate_qss — QSS gerado a partir dos tokens (PURO)       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — theme (estética profissional)                        ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de tema                                ║
# ║  Framework   : stdlib PURO — gera string QSS (sem Qt; testável)            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Gera a folha de estilo Qt (QSS) a partir de :class:`ThemeTokens`. Uma   ║
# ║    f-string parametrizada cobre os widgets do app (janela, inputs, botões, ║
# ║    combos+popup, grupos, abas, barra de progresso, scrollbars, tooltip).   ║
# ║    Trocar a paleta = trocar os tokens; o QSS re-gera automaticamente.       ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    generate_qss                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``generate_qss(tokens)`` — QSS do tema gerado a partir dos tokens (PURO, sem Qt)."""

from __future__ import annotations

from geosteering_ai.gui.theme.tokens import ThemeTokens

__all__ = ["generate_qss"]


def generate_qss(tokens: ThemeTokens) -> str:
    """Gera a folha de estilo (QSS) global do app a partir dos ``tokens``.

    Args:
        tokens: paleta/tipografia (ver :class:`ThemeTokens`).

    Returns:
        String QSS pronta para ``QApplication.setStyleSheet`` (cobre janela,
        QLabel, QLineEdit, QSpinBox/QDoubleSpinBox, QComboBox(+popup),
        QPushButton(+estados), QGroupBox, QTabWidget/QTabBar, QProgressBar,
        QPlainTextEdit/QTextEdit, QCheckBox, QScrollBar e QToolTip).
    """
    t = tokens
    return f"""
/* ── Base ───────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {t.bg_primary};
    color: {t.text_primary};
    font-family: {t.font_family};
    font-size: {t.font_size_base}px;
}}
QMainWindow, QDialog {{ background-color: {t.bg_primary}; }}
QLabel {{ background: transparent; color: {t.text_primary}; }}
QLabel:disabled {{ color: {t.text_muted}; }}

/* ── Inputs (line edit, spin boxes, text edit) ──────────────────────── */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    padding: 4px 6px;
    selection-background-color: {t.selection_bg};
    selection-color: {t.text_primary};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {t.accent}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {t.text_muted};
    background-color: {t.bg_tertiary};
}}
QLineEdit::placeholder {{ color: {t.text_muted}; }}

/* ── ComboBox + popup ───────────────────────────────────────────────── */
QComboBox {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    padding: 4px 8px;
}}
QComboBox:focus {{ border: 1px solid {t.accent}; }}
QComboBox:disabled {{ color: {t.text_muted}; background-color: {t.bg_tertiary}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    selection-background-color: {t.accent};
    selection-color: #ffffff;
    outline: none;
}}

/* ── Botões ─────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius}px;
    padding: 6px 14px;
}}
QPushButton:hover {{ border: 1px solid {t.accent}; }}
QPushButton:pressed {{ background-color: {t.accent_pressed}; color: #ffffff; }}
QPushButton:default {{ background-color: {t.accent}; color: #ffffff; border: none; }}
QPushButton:default:hover {{ background-color: {t.accent_hover}; }}
QPushButton:disabled {{ color: {t.text_muted}; background-color: {t.bg_tertiary}; }}

/* ── GroupBox ───────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {t.bg_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius}px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {t.text_secondary};
}}

/* ── Abas ───────────────────────────────────────────────────────────── */
QTabWidget::pane {{ border: 1px solid {t.border}; border-radius: {t.radius_sm}px; }}
QTabBar::tab {{
    background-color: {t.bg_primary};
    color: {t.text_secondary};
    padding: 7px 16px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {t.text_primary}; border-bottom: 2px solid {t.accent}; }}
QTabBar::tab:hover {{ color: {t.text_primary}; }}

/* ── Barra de progresso ─────────────────────────────────────────────── */
QProgressBar {{
    background-color: {t.bg_secondary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    text-align: center;
    color: {t.text_primary};
}}
QProgressBar::chunk {{ background-color: {t.accent}; border-radius: {t.radius_sm}px; }}

/* ── CheckBox ───────────────────────────────────────────────────────── */
QCheckBox {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    background-color: {t.bg_secondary};
}}
QCheckBox::indicator:checked {{ background-color: {t.accent}; border: 1px solid {t.accent}; }}

/* ── ScrollBars ─────────────────────────────────────────────────────── */
QScrollBar:vertical {{ background: {t.bg_primary}; width: 12px; margin: 0; }}
QScrollBar:horizontal {{ background: {t.bg_primary}; height: 12px; margin: 0; }}
QScrollBar::handle {{ background: {t.border}; border-radius: {t.radius_sm}px; min-height: 24px; }}
QScrollBar::handle:hover {{ background: {t.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ── ToolTip ────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.accent};
    border-radius: {t.radius_sm}px;
    padding: 4px 6px;
}}
"""
