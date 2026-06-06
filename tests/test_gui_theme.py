# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_gui_theme.py                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Subsistema  : GUI — theme (estética profissional Antigravity)            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-06                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida o tema PURO: tokens (hex válido), QSS gerado (cobre widgets +    ║
# ║    usa os tokens), apply_theme (duck-typed, sem Qt) e a fronteira          ║
# ║    (gui.theme importável sem binding Qt).                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do ``gui.theme`` — tokens · QSS · apply_theme (PURO, sem Qt)."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class _StubApp:
    """QApplication duck-typed (só ``setStyleSheet``) — testa apply_theme sem Qt."""

    def __init__(self) -> None:
        self.qss: str | None = None

    def setStyleSheet(self, qss: str) -> None:  # noqa: N802 — API Qt
        self.qss = qss


def test_theme_importable_without_qt():
    """Fronteira — importar gui.theme + gerar QSS NÃO puxa Qt (Princípio X)."""
    code = (
        "import sys\n"
        "from geosteering_ai.gui.theme import generate_qss, ANTIGRAVITY_DARK, apply_theme\n"
        "qss = generate_qss(ANTIGRAVITY_DARK)\n"
        "assert 'QPushButton' in qss\n"
        "bad = [m for m in ('PyQt6', 'PySide6') if m in sys.modules]\n"
        "assert not bad, f'theme puxou Qt: {bad}'\n"
        "print('PURE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "PURE_OK" in proc.stdout


def test_tokens_are_valid_hex():
    """Todos os tokens de COR são ``#rrggbb`` válidos."""
    from geosteering_ai.gui.theme import ANTIGRAVITY_DARK

    color_fields = [
        f.name
        for f in fields(ANTIGRAVITY_DARK)
        if f.name not in ("font_family", "font_size_base", "radius", "radius_sm")
    ]
    assert color_fields  # sanidade
    for name in color_fields:
        value = getattr(ANTIGRAVITY_DARK, name)
        assert _HEX_RE.match(value), f"token {name}={value!r} não é #rrggbb"


def test_generate_qss_covers_widgets():
    """O QSS cobre os widgets principais do app."""
    from geosteering_ai.gui.theme import ANTIGRAVITY_DARK, generate_qss

    qss = generate_qss(ANTIGRAVITY_DARK)
    for selector in (
        "QMainWindow",
        "QLineEdit",
        "QDoubleSpinBox",
        "QComboBox QAbstractItemView",  # popup do combo (crítico)
        "QPushButton",
        "QGroupBox",
        "QTabBar::tab",
        "QProgressBar::chunk",
        "QCheckBox::indicator",
        "QScrollBar",
        "QToolTip",
    ):
        assert selector in qss, f"QSS não cobre {selector!r}"


def test_generate_qss_uses_tokens():
    """O QSS é DERIVADO dos tokens (trocar o token muda o QSS)."""
    from geosteering_ai.gui.theme import ThemeTokens, generate_qss

    custom = ThemeTokens(bg_primary="#abcdef", accent="#123456")
    qss = generate_qss(custom)
    assert "#abcdef" in qss  # bg_primary aplicado
    assert "#123456" in qss  # accent aplicado


def test_apply_theme_sets_stylesheet():
    """apply_theme aplica o QSS (duck-typed, sem Qt)."""
    from geosteering_ai.gui.theme import ANTIGRAVITY_DARK, apply_theme

    app = _StubApp()
    apply_theme(app)
    assert app.qss is not None
    assert "QPushButton" in app.qss
    assert ANTIGRAVITY_DARK.bg_primary in app.qss


def test_apply_theme_none_is_noop():
    """apply_theme(None) é no-op (ex.: sem binding Qt) — sem crash."""
    from geosteering_ai.gui.theme import apply_theme

    apply_theme(None)  # não levanta
