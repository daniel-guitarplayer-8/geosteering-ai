# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/layers_dialog.py               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : LayersDialog — editor manual de camadas (Qt)               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0015, 6b)      ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — geologia avançada                               ║
# ║  Framework   : Qt6 via gui.qt_compat (QDialog + QTableWidget)              ║
# ║  Padrão      : View (Qt) — edita um ManualLayersModel (PURO); sem física   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Dialog de edição manual de camadas (espelha sm_layers_dialog do         ║
# ║    monólito): nº de camadas + tabela ρₕ/ρᵥ/espessura. Devolve um           ║
# ║    ``ManualLayersModel`` (PURO) via :meth:`get_model`. A espessura só é      ║
# ║    editável nas camadas INTERNAS (os 2 semi-espaços não têm espessura).     ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    LayersDialog                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``LayersDialog`` — editor manual de camadas (ρₕ/ρᵥ/espessura) → ManualLayersModel."""

from __future__ import annotations

from typing import Any, Optional

from geosteering_ai.gui.qt_compat import Qt, QtWidgets
from geosteering_ai.gui.services.manual_geology import ManualLayersModel

__all__ = ["LayersDialog"]

_MAX_LAYERS = 50


class LayersDialog(QtWidgets.QDialog):  # type: ignore[misc] # QtWidgets é Any
    """Dialog modal de edição manual de camadas → :class:`ManualLayersModel`.

    Args:
        initial: modelo a pré-carregar (ex.: perfil canônico) ou ``None`` (default 3 cam.).
        parent: widget pai opcional.

    Note:
        A tabela tem ``n_layers`` linhas × 3 colunas (ρₕ, ρᵥ, espessura). A espessura
        é editável só nas linhas INTERNAS (1 … n−2); os 2 semi-espaços (1ª/última)
        têm a célula de espessura desabilitada (convenção do simulador).
    """

    def __init__(
        self, initial: Optional[ManualLayersModel] = None, parent: Any = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Editar camadas (geologia manual)")
        self.resize(440, 360)
        self._initial = initial

        self._n_spin = QtWidgets.QSpinBox()
        self._n_spin.setRange(2, _MAX_LAYERS)
        self._n_spin.setValue(initial.n_layers if initial else 3)
        self._n_spin.setToolTip("Nº de camadas (inclui 2 semi-espaços).")

        self._table = QtWidgets.QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["ρₕ (Ω·m)", "ρᵥ (Ω·m)", "espessura (m)"])
        self._table.horizontalHeader().setStretchLastSection(True)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Nº de camadas:"))
        top.addWidget(self._n_spin)
        top.addStretch(1)
        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self._table, stretch=1)
        root.addWidget(buttons)

        self._n_spin.valueChanged.connect(self._rebuild_rows)
        self._rebuild_rows(self._n_spin.value())

    def _rebuild_rows(self, n_layers: int) -> None:
        """(Re)constrói as linhas da tabela p/ ``n_layers`` (preserva o que couber)."""
        prev = self._read_rows()
        self._table.setRowCount(n_layers)
        for row in range(n_layers):
            is_internal = 0 < row < n_layers - 1
            defaults = prev[row] if row < len(prev) else self._default_row(row)
            self._set_cell(row, 0, defaults[0])
            self._set_cell(row, 1, defaults[1])
            # Espessura só nas camadas internas; semi-espaços = "—" desabilitado.
            esp_item = QtWidgets.QTableWidgetItem(
                f"{defaults[2]:g}" if is_internal else "—"
            )
            if not is_internal:
                esp_item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # não-editável
                esp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 2, esp_item)
            self._table.setVerticalHeaderItem(
                row, QtWidgets.QTableWidgetItem(f"camada {row + 1}")
            )

    @staticmethod
    def _default_row(row: int) -> tuple:
        """Valores default por linha (ρₕ, ρᵥ, espessura) — TIV brando."""
        rho = 1.0 if row == 0 else (100.0 if row == 1 else 10.0)
        return (rho, rho, 8.0)

    def _set_cell(self, row: int, col: int, value: float) -> None:
        self._table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{value:g}"))

    def _read_rows(self) -> list:
        """Lê a tabela como lista de tuplas (ρₕ, ρᵥ, espessura) — não-numérico → 0."""
        out = []
        for row in range(self._table.rowCount()):
            vals = []
            for col in range(3):
                item = self._table.item(row, col)
                try:
                    vals.append(float(item.text()) if item else 0.0)
                except ValueError:
                    vals.append(0.0)
            out.append(tuple(vals))
        return out

    def get_model(self) -> ManualLayersModel:
        """Devolve o :class:`ManualLayersModel` editado (validar com ``.validate()``)."""
        rows = self._read_rows()
        n = len(rows)
        rho_h = tuple(r[0] for r in rows)
        rho_v = tuple(r[1] for r in rows)
        # Espessura só das camadas internas (linhas 1 … n−2).
        thick = tuple(rows[i][2] for i in range(1, n - 1)) if n >= 2 else ()
        return ManualLayersModel(
            n_layers=n, thicknesses=thick, rho_h=rho_h, rho_v=rho_v
        )
