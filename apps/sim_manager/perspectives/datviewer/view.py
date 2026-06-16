# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/datviewer/view.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : DatViewerPanel — View Qt do visualizador .dat              ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Visualizador .dat (Fatia 6h)    ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — paridade c/ sm_dat_viewer.py do monólito          ║
# ║  Framework   : Qt6 via gui.qt_compat                                       ║
# ║  Dependências: gui.qt_compat, .viewmodel, .service (column_labels)          ║
# ║  Padrão      : View (MVVM) — sem lógica; binding aos VMSignals do VM        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    "Abrir .dat…" → tabela 22-col (rótulos físicos canônicos) + painel de    ║
# ║    metadados do .out paralelo. SOMENTE LEITURA (zero física). A tabela       ║
# ║    exibe no máx. DISPLAY_ROW_CAP linhas (truncamento explícito no header —   ║
# ║    sem cap silencioso); o resultado completo permanece no ViewModel.        ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    DatViewerPanel                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``DatViewerPanel`` — View Qt do visualizador ``.dat``/``.out`` (Fatia 6h)."""

from __future__ import annotations

import math
from typing import Any

from apps.sim_manager.perspectives.datviewer.service import (
    DatLoadResult,
    column_labels,
)
from apps.sim_manager.perspectives.datviewer.viewmodel import DatViewerViewModel
from geosteering_ai.gui.qt_compat import Qt, QtWidgets

__all__ = ["DatViewerPanel"]

# Teto de linhas RENDERIZADAS na tabela (a matriz completa fica no ViewModel).
# Um .dat pode ter milhões de registros — popular o QTableWidget inteiro
# travaria a UI. O header avisa explicitamente quando há truncamento.
DISPLAY_ROW_CAP = 1000


def _heading_title(text: str) -> Any:
    """``QLabel`` de título de seção (``role="heading"`` — estilizado pelo QSS)."""
    lbl = QtWidgets.QLabel(text)
    lbl.setProperty("role", "heading")
    return lbl


def _hint(text: str) -> Any:
    """``QLabel`` de subtítulo descritivo (``role="hint"`` — estilizado pelo QSS)."""
    lbl = QtWidgets.QLabel(text)
    lbl.setProperty("role", "hint")
    lbl.setWordWrap(True)
    return lbl


class DatViewerPanel(QtWidgets.QWidget):  # type: ignore[misc] # QtWidgets é Any → mypy
    """View do visualizador ``.dat``: abrir → tabela + metadados ``.out``.

    Args:
        vm: o :class:`DatViewerViewModel` (a View se liga aos seus VMSignals).
        parent: widget pai opcional.

    Note:
        A View não lê nem interpreta física — só dispara ``vm.open(path)`` e
        renderiza o :class:`DatLoadResult` que o VM emite (``loaded``/``load_error``).
    """

    def __init__(self, vm: DatViewerViewModel, parent: Any = None) -> None:
        super().__init__(parent)
        self._vm = vm

        # ── Barra de ações ───────────────────────────────────────────────────
        self._open_btn = QtWidgets.QPushButton("Abrir .dat…")
        self._open_btn.setProperty("role", "primary")
        self._open_btn.setToolTip(
            "Abre um .dat (binário 22-col ou ASCII) + o .out paralelo (metadados), "
            "sem re-simular. Formato gravado pelo grupo 'Saída' da Simulação."
        )
        self._summary = QtWidgets.QLabel("Nenhum arquivo carregado.")
        self._summary.setProperty("role", "hint")
        self._summary.setWordWrap(True)

        # ── Tabela (matriz) + metadados (.out) num splitter vertical ─────────
        self._table = QtWidgets.QTableWidget(0, 0)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setAlternatingRowColors(True)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.horizontalHeader().setStretchLastSection(True)

        self._metadata = QtWidgets.QPlainTextEdit()
        self._metadata.setReadOnly(True)
        self._metadata.setPlaceholderText("(metadados do .out aparecerão aqui)")

        splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._table)
        meta_box = QtWidgets.QGroupBox("Metadados (.out)")
        meta_layout = QtWidgets.QVBoxLayout(meta_box)
        meta_layout.addWidget(self._metadata)
        splitter.addWidget(meta_box)
        splitter.setStretchFactor(0, 3)  # tabela domina
        splitter.setStretchFactor(1, 1)

        # ── Layout ────────────────────────────────────────────────────────────
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(self._open_btn)
        action_row.addStretch(1)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(_heading_title("Visualizador .dat / .out"))
        root.addWidget(
            _hint(
                "Inspeção somente-leitura de artefatos Fortran-compat (.dat binário "
                "22-col + .out ASCII) sem re-simular. Layout 22-col: meds · zobs · "
                "res_h · res_v · Re/Im dos 9 componentes do tensor."
            )
        )
        root.addLayout(action_row)
        root.addWidget(self._summary)
        root.addWidget(splitter, stretch=1)

        # ── Binding ──────────────────────────────────────────────────────────
        self._open_btn.clicked.connect(self._on_open_clicked)
        self._vm.loaded.connect(self._on_loaded)
        self._vm.load_error.connect(self._on_error)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_open_clicked(self) -> None:
        """Abre o seletor de arquivo e dispara a leitura no VM."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir .dat", "", "Artefato Fortran (*.dat);;Todos os arquivos (*)"
        )
        if path:
            self._vm.open(path)

    def _on_loaded(self, result: DatLoadResult) -> None:
        """Popula a tabela (capada) + metadados a partir de um resultado válido."""
        self._summary.setText(self._summary_text(result))
        self._metadata.setPlainText(
            result.out_metadata or "(arquivo .out não encontrado)"
        )
        self._populate_table(result)

    def _on_error(self, msg: str) -> None:
        """Exibe a falha de leitura (sem crash) e limpa a tabela."""
        self._summary.setText(f"⚠ {msg}")
        self._table.clearContents()
        self._table.setRowCount(0)
        self._metadata.setPlainText("")

    # ── Helpers ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _summary_text(result: DatLoadResult) -> str:
        """Linha de resumo (inclui aviso de truncamento se aplicável)."""
        out = "com .out" if result.out_metadata else "sem .out"
        base = (
            f"{result.dat_path} · {result.n_rows} linhas × {result.n_cols} col · "
            f"{result.fmt} · {result.size_mb:.1f} MB · {out}"
        )
        if result.n_rows > DISPLAY_ROW_CAP:
            base += f"  —  exibindo as primeiras {DISPLAY_ROW_CAP} de {result.n_rows}"
        return base

    def _populate_table(self, result: DatLoadResult) -> None:
        """Renderiza até :data:`DISPLAY_ROW_CAP` linhas da matriz na tabela."""
        data = result.data
        if data is None or result.n_rows == 0 or result.n_cols == 0:
            self._table.clearContents()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return
        n_show = min(result.n_rows, DISPLAY_ROW_CAP)
        self._table.setColumnCount(result.n_cols)
        self._table.setHorizontalHeaderLabels(column_labels(result.n_cols))
        self._table.setRowCount(n_show)
        for r in range(n_show):
            for c in range(result.n_cols):
                item = QtWidgets.QTableWidgetItem(self._fmt_cell(data[r, c], c))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._table.setItem(r, c, item)

    @staticmethod
    def _fmt_cell(value: float, col: int) -> str:
        """Formata uma célula: col0 (model_id) como int só se finito e inteiro.

        No fallback ASCII a col0 pode ser não-inteira ou não-finita (NaN/Inf de dump
        corrompido). ``int(NaN)`` levantava ``ValueError`` — engolido pelo VMSignal,
        deixando a tabela meio-renderizada sem aviso. Aqui valores não-inteiros/não-
        finitos caem no formatador float (``"nan"``/``"inf"``/valor), nunca crasham.
        """
        if col == 0 and math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
