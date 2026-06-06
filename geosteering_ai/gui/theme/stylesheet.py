# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/theme/stylesheet.py                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : generate_qss — QSS Antigravity gerado dos tokens (PURO)    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — theme (estética profissional Google Antigravity)     ║
# ║  Versão      : v0.2 (spec 0013 — cards, pill roles, rail, sidebar)         ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de tema                                ║
# ║  Framework   : stdlib PURO — gera string QSS (sem Qt; testável)            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Gera a folha de estilo Qt (QSS) Antigravity a partir de ``ThemeTokens``.║
# ║    Cobre: base, inputs (focus ring), combos+popup, botões pill com         ║
# ║    ``[role]`` (primary/danger/ghost), cards (QGroupBox elevado), abas,      ║
# ║    progressbar, checkbox/radio, scrollbars finas, listas, tabela, tooltip,  ║
# ║    status bar com accent, activity rail (#ActivityBar) e secondary sidebar. ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    generate_qss                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``generate_qss(tokens)`` — QSS Antigravity gerado dos tokens (PURO, sem Qt)."""

from __future__ import annotations

from geosteering_ai.gui.theme.tokens import ThemeTokens

__all__ = ["generate_qss"]


def generate_qss(tokens: ThemeTokens) -> str:
    """Gera a folha de estilo (QSS) global Antigravity a partir dos ``tokens``.

    Args:
        tokens: paleta/tipografia/espaçamento (ver :class:`ThemeTokens`).

    Returns:
        String QSS pronta para ``QApplication.setStyleSheet``. Cobre janela,
        labels (+roles), inputs (+focus), combos (+popup), botões (+roles pill),
        cards (QGroupBox), abas, progressbar, checkbox/radio, scrollbars, listas,
        tabela, tooltip, status bar (accent), activity rail e secondary sidebar.

    Note:
        Estética inspirada no Google Antigravity IDE (dark midnight + índigo).
        Botões pill usam a propriedade dinâmica ``role`` (primary/danger/ghost);
        defina-a antes do ``show`` (ver ``shell/widgets/styled_widgets.py``).
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
QLabel[role="heading"] {{ font-size: {t.font_size_lg}px; font-weight: 600; color: #ffffff; padding: 2px 0 6px 0; }}
QLabel[role="section"] {{ font-size: {t.font_size_sm}px; font-weight: 600; color: {t.text_secondary}; }}
QLabel[role="hint"] {{ font-size: {t.font_size_sm}px; color: {t.text_muted}; }}

/* ── Inputs (line edit, spin boxes, text edit) — focus ring índigo ──── */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    padding: {t.spacing_sm}px {t.spacing_md}px;
    selection-background-color: {t.selection_bg};
    selection-color: #ffffff;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {t.accent}; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border: 1px solid {t.accent_hover}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QPlainTextEdit:disabled, QTextEdit:disabled {{
    color: {t.text_muted};
    background-color: {t.bg_tertiary};
}}
QLineEdit::placeholder {{ color: {t.text_muted}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {t.bg_tertiary}; border: none; width: 16px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {t.accent_light}; }}

/* ── ComboBox + popup ───────────────────────────────────────────────── */
QComboBox {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    padding: {t.spacing_sm}px {t.spacing_md}px;
    min-height: 22px;
}}
QComboBox:focus, QComboBox:on {{ border: 1px solid {t.accent}; }}
QComboBox:disabled {{ color: {t.text_muted}; background-color: {t.bg_tertiary}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {t.text_secondary}; margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    selection-background-color: {t.accent};
    selection-color: #ffffff;
    outline: none;
    padding: 2px;
}}
QComboBox QAbstractItemView::item {{ padding: {t.spacing_sm}px {t.spacing_md}px; min-height: 24px; }}

/* ── Botões (pill) — default + roles primary/danger/ghost ───────────── */
QPushButton {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_md}px;
    padding: {t.spacing_md}px {t.spacing_xl}px;
    font-weight: 500;
    min-height: 16px;
}}
QPushButton:hover {{ background-color: {t.bg_tertiary}; border: 1px solid {t.accent_hover}; }}
QPushButton:pressed {{ background-color: {t.accent_pressed}; color: #ffffff; }}
QPushButton:disabled {{ color: {t.text_muted}; background-color: {t.bg_tertiary}; border-color: {t.border}; }}
QPushButton:default, QPushButton[role="primary"] {{
    background-color: {t.accent}; color: #ffffff; border: 1px solid {t.accent_hover}; font-weight: 600;
}}
QPushButton:default:hover, QPushButton[role="primary"]:hover {{ background-color: {t.accent_hover}; }}
QPushButton[role="primary"]:pressed {{ background-color: {t.accent_pressed}; }}
QPushButton[role="primary"]:disabled {{ background-color: {t.accent_light}; color: {t.text_secondary}; }}
QPushButton[role="danger"] {{ background-color: {t.error}; color: #ffffff; border: 1px solid {t.error}; font-weight: 600; }}
QPushButton[role="danger"]:hover {{ background-color: #f87171; }}
QPushButton[role="danger"]:disabled {{ background-color: {t.bg_tertiary}; color: {t.text_muted}; border-color: {t.border}; }}
QPushButton[role="ghost"] {{ background: transparent; color: {t.text_secondary}; border: 1px solid transparent; }}
QPushButton[role="ghost"]:hover {{ background-color: {t.bg_secondary}; color: {t.text_primary}; border: 1px solid {t.border}; }}

/* ── Cards (QGroupBox elevado) ──────────────────────────────────────── */
QGroupBox {{
    background-color: {t.bg_secondary};
    border: 1px solid {t.border};
    border-radius: {t.radius_md}px;
    margin-top: {t.spacing_lg}px;
    padding: {t.spacing_lg}px {t.spacing_md}px {t.spacing_md}px {t.spacing_md}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {t.spacing_lg}px;
    padding: 0 {t.spacing_sm}px;
    color: {t.text_secondary};
}}

/* ── Abas ───────────────────────────────────────────────────────────── */
QTabWidget::pane {{ border: 1px solid {t.border}; border-radius: {t.radius_sm}px; top: -1px; }}
QTabBar::tab {{
    background-color: {t.bg_primary};
    color: {t.text_secondary};
    padding: {t.spacing_md}px {t.spacing_xl}px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {t.text_primary}; border-bottom: 2px solid {t.accent}; }}
QTabBar::tab:hover:!selected {{ color: {t.text_primary}; background-color: {t.bg_secondary}; }}

/* ── Barra de progresso ─────────────────────────────────────────────── */
QProgressBar {{
    background-color: {t.bg_secondary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    text-align: center;
    color: {t.text_primary};
    min-height: 16px;
}}
QProgressBar::chunk {{ background-color: {t.accent}; border-radius: {t.radius_sm}px; }}

/* ── CheckBox / RadioButton ─────────────────────────────────────────── */
QCheckBox, QRadioButton {{ background: transparent; spacing: 6px; color: {t.text_primary}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    background-color: {t.bg_secondary};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border: 1px solid {t.accent_hover}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {t.accent}; border: 1px solid {t.accent};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ background-color: {t.bg_tertiary}; border-color: {t.border}; }}

/* ── ScrollBars (finas) ─────────────────────────────────────────────── */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {t.border}; border-radius: {t.radius_sm}px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {t.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ── Listas / Árvores / Tabela ──────────────────────────────────────── */
QListWidget, QTreeWidget, QTreeView, QListView {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    outline: none;
    alternate-background-color: {t.bg_tertiary};
}}
QListWidget::item, QTreeWidget::item {{ padding: {t.spacing_sm}px {t.spacing_md}px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background-color: {t.accent}; color: #ffffff; }}
QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected {{ background-color: {t.bg_tertiary}; }}
QTableWidget, QTableView {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
    gridline-color: {t.border};
    alternate-background-color: {t.bg_tertiary};
    outline: none;
}}
QTableWidget::item:selected, QTableView::item:selected {{ background-color: {t.accent}; color: #ffffff; }}
QHeaderView::section {{
    background-color: {t.bg_tertiary};
    color: {t.text_secondary};
    padding: {t.spacing_sm}px {t.spacing_md}px;
    border: none;
    border-bottom: 1px solid {t.border};
    font-weight: 600;
}}

/* ── ToolTip ────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {t.bg_secondary};
    color: {t.text_primary};
    border: 1px solid {t.accent};
    border-radius: {t.radius_sm}px;
    padding: {t.spacing_sm}px {t.spacing_md}px;
}}

/* ── Status bar (accent Antigravity) ────────────────────────────────── */
QStatusBar {{
    background-color: {t.accent};
    color: #ffffff;
    border-top: 1px solid {t.accent_pressed};
}}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: #ffffff; background: transparent; }}

/* ── Splitter ───────────────────────────────────────────────────────── */
QSplitter::handle {{ background-color: {t.bg_primary}; }}
QSplitter::handle:hover {{ background-color: {t.accent_light}; }}

/* ── Activity rail (nav vertical por ícones) ────────────────────────── */
QWidget#ActivityBar {{ background-color: {t.bg_rail}; border-right: 1px solid {t.border}; }}
QWidget#ActivityBar QToolButton {{
    background: transparent; color: {t.text_secondary};
    border: none; border-left: 2px solid transparent;
    border-radius: 0px; padding: {t.spacing_md}px;
}}
QWidget#ActivityBar QToolButton:hover {{ color: {t.text_primary}; background-color: {t.bg_secondary}; }}
QWidget#ActivityBar QToolButton:checked {{
    color: #ffffff; background-color: {t.bg_secondary}; border-left: 2px solid {t.accent};
}}
QWidget#ActivityBar QToolButton:disabled {{ color: {t.text_muted}; }}

/* ── Secondary sidebar (Histórico/Log/Artifacts) ────────────────────── */
QWidget#SecondarySidebar {{ background-color: {t.bg_secondary}; border-left: 1px solid {t.border}; }}
QPlainTextEdit#LogView {{
    background-color: {t.bg_primary};
    color: {t.text_secondary};
    font-family: {t.font_family_mono};
    font-size: {t.font_size_sm}px;
    border: 1px solid {t.border};
    border-radius: {t.radius_sm}px;
}}

/* ── Dock ───────────────────────────────────────────────────────────── */
QDockWidget {{ color: {t.text_primary}; titlebar-close-icon: none; }}
QDockWidget::title {{ background-color: {t.bg_tertiary}; padding: {t.spacing_sm}px {t.spacing_md}px; }}
"""
