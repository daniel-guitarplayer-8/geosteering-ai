# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_dat_viewer.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Visualizador de .dat / .out          ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Status      : Produção (v2.6) — implementação inicial                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Carrega arquivos `.dat` (formato 22-col Fortran) e `.out` (metadados)  ║
# ║    para visualização sem re-executar a simulação. Reaproveita os plots    ║
# ║    de sm_plots.py e oferece tabela numérica para inspeção.                ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    from sm_dat_viewer import DatViewerDialog                              ║
# ║    dlg = DatViewerDialog(parent, dat_path)                                ║
# ║    dlg.exec()                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""DatViewerDialog — visualizador de arquivos .dat/.out (v2.6 P3)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    _QT = "PyQt6"
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

        _QT = "PyQt5"
    except ImportError:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

        _QT = "PySide6"

__all__ = ["DatViewerDialog", "load_dat_file"]


def load_dat_file(dat_path: Path) -> dict:
    """Carrega arquivo .dat 22-col + .out paralelo (se existir).

    Args:
        dat_path: caminho para .dat (ASCII, formato Fortran).

    Returns:
        Dict com chaves: ``"data"`` (ndarray 2D), ``"n_rows"``, ``"n_cols"``,
        ``"out_metadata"`` (str se .out existe, None caso contrário).

    Raises:
        FileNotFoundError: se .dat não existe.
        ValueError: se arquivo malformado.
    """
    dat_path = Path(dat_path)
    if not dat_path.exists():
        raise FileNotFoundError(f"Arquivo .dat não encontrado: {dat_path}")

    # Aviso se arquivo > 500 MB (não trava — apenas log)
    size_mb = dat_path.stat().st_size / 1e6
    if size_mb > 500:
        import logging

        logging.warning(
            "load_dat_file: arquivo grande (%.0f MB) — carregamento pode "
            "demorar. Considere chunked load para .dat > 1 GB.",
            size_mb,
        )

    try:
        data = np.loadtxt(dat_path, comments="#")
    except Exception as exc:
        raise ValueError(f"Falha ao parsear {dat_path}: {exc}") from exc

    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_rows, n_cols = data.shape

    # Procura .out paralelo (mesmo basename, extensão .out)
    out_path = dat_path.with_suffix(".out")
    out_metadata: Optional[str] = None
    if out_path.exists():
        try:
            out_metadata = out_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            out_metadata = None

    return {
        "data": data,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "out_metadata": out_metadata,
        "dat_path": str(dat_path),
        "size_mb": size_mb,
    }


class DatViewerDialog(QtWidgets.QDialog):
    """Dialog para visualizar arquivos .dat/.out exportados pelo SM.

    UI:
      • Header: caminho do arquivo + tamanho + nº de linhas
      • Splitter horizontal (top/bottom):
          - Top: QTextEdit com metadados do .out
          - Bottom: QTableView com primeiras 1000 linhas do tensor
      • Buttons: Fechar | Adicionar ao histórico (futuro v2.6.x)

    Note:
        Implementação inicial v2.6 — focada em load + display tabular.
        Plot integrado com `sm_plots.plot_em_profile` está em backlog.
    """

    def __init__(self, parent: QtWidgets.QWidget, dat_path: Path) -> None:
        super().__init__(parent)
        self.setWindowTitle("Visualizador .dat / .out — v2.6")
        self.setModal(False)
        self.resize(900, 600)

        try:
            self._info = load_dat_file(Path(dat_path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Erro ao carregar", f"<code>{exc}</code>"
            )
            self._info = None
            return

        root = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            f"<b>Arquivo:</b> <code>{self._info['dat_path']}</code><br/>"
            f"<b>Linhas:</b> {self._info['n_rows']} · "
            f"<b>Colunas:</b> {self._info['n_cols']} · "
            f"<b>Tamanho:</b> {self._info['size_mb']:.1f} MB"
        )
        header.setWordWrap(True)
        header.setStyleSheet("padding:8px; background:#252526; color:#d4d4d4;")
        root.addWidget(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Metadata .out
        self.text_meta = QtWidgets.QPlainTextEdit()
        self.text_meta.setReadOnly(True)
        meta = self._info.get("out_metadata") or "(arquivo .out não encontrado)"
        self.text_meta.setPlainText(meta)
        splitter.addWidget(self.text_meta)

        # Tabela com primeiras N linhas
        MAX_ROWS = 1000
        data = self._info["data"]
        view_data = data[:MAX_ROWS]
        self.table = QtWidgets.QTableWidget(view_data.shape[0], view_data.shape[1])
        for r in range(view_data.shape[0]):
            for c in range(view_data.shape[1]):
                item = QtWidgets.QTableWidgetItem(f"{view_data[r, c]:.6e}")
                self.table.setItem(r, c, item)
        if data.shape[0] > MAX_ROWS:
            self.table.setHorizontalHeaderLabels(
                [f"col{i}" for i in range(view_data.shape[1])]
            )
            note = QtWidgets.QLabel(
                f"⚠ Mostrando primeiras {MAX_ROWS} linhas de {data.shape[0]}."
            )
            note.setStyleSheet("color:#dcdcaa;")
            splitter.addWidget(note)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        bottom = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        bottom.rejected.connect(self.reject)
        root.addWidget(bottom)
