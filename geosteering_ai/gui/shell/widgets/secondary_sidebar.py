# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/widgets/secondary_sidebar.py                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SecondarySidebar — painel direito (Histórico/Log/Artifacts)║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell widgets (spec 0013)                            ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação de shell                               ║
# ║  Framework   : Qt6 via gui.qt_compat (QTabWidget)                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Painel lateral direito do shell Antigravity (a "Manager surface"):      ║
# ║    abas Histórico (lista de simulações) · Log (saída monoespaçada) ·        ║
# ║    Artifacts (resultados/plots — placeholder). A Fatia 6a alimenta o Log    ║
# ║    (``append_log``) e o Histórico (``add_history_item``). Sem lógica de     ║
# ║    domínio — só apresentação. QSS via ``#SecondarySidebar``/``#LogView``.   ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SecondarySidebar                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SecondarySidebar`` — painel direito Histórico/Log/Artifacts (spec 0013)."""

from __future__ import annotations

from typing import Any, Optional

from geosteering_ai.gui.qt_compat import Qt, QtWidgets

__all__ = ["SecondarySidebar"]

_LOG_MAX_BLOCKS = 5000  # rotaciona o log (evita crescimento ilimitado)


class SecondarySidebar(QtWidgets.QWidget):  # type: ignore[misc] # QtWidgets é Any
    """Painel lateral direito (abas Histórico/Log/Artifacts) do shell Antigravity.

    Note:
        ``objectName == "SecondarySidebar"`` (QSS). O ``QPlainTextEdit`` do log usa
        ``objectName == "LogView"`` (fonte monoespaçada via QSS). A Fatia 6a usa
        ``append_log``/``add_history_item`` para feedback de execução.
    """

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SecondarySidebar")

        self._tabs = QtWidgets.QTabWidget(self)

        # ── Histórico ─────────────────────────────────────────────────────
        self._history = QtWidgets.QListWidget()
        self._history.setToolTip("Snapshots de simulações desta sessão.")
        self._tabs.addTab(self._history, "Histórico")

        # ── Log ───────────────────────────────────────────────────────────
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setObjectName("LogView")
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(_LOG_MAX_BLOCKS)
        self._log.setPlaceholderText("Log da simulação…")
        self._tabs.addTab(self._log, "Log")

        # ── Artifacts (placeholder — Fatia 6d) ────────────────────────────
        artifacts = QtWidgets.QLabel(
            "Sem artefatos ainda.\n(plots/relatórios — Fatia 6d)"
        )
        artifacts.setAlignment(Qt.AlignmentFlag.AlignCenter)
        artifacts.setWordWrap(True)
        artifacts.setProperty("role", "hint")
        self._tabs.addTab(artifacts, "Artifacts")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)

    # ── API para a perspectiva ativa (Fatia 6a) ───────────────────────────
    def append_log(self, text: str) -> None:
        """Anexa uma linha ao painel de Log (rotaciona em ``_LOG_MAX_BLOCKS``)."""
        self._log.appendPlainText(text)

    def clear_log(self) -> None:
        """Limpa o painel de Log."""
        self._log.clear()

    def add_history_item(self, text: str) -> None:
        """Adiciona um item ao topo do Histórico de simulações."""
        self._history.insertItem(0, text)

    def focus_log(self) -> None:
        """Traz a aba de Log para frente (ex.: ao iniciar uma simulação)."""
        self._tabs.setCurrentWidget(self._log)
