# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_layers_dialog.py                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Entrada Manual de Camadas (v2.4)      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-20                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Caixa de diálogo modal para o usuário inserir manualmente as camadas   ║
# ║    de um modelo geológico TIV, como alternativa à geração aleatória       ║
# ║    (QMC/PRNG). Reproduz a convenção de ``buildValidamodels.py`` (alturas  ║
# ║    fixas) e permite reproduzir os 8 perfis canônicos bit-a-bit após       ║
# ║    aplicar um perfil e (opcionalmente) editar valores.                    ║
# ║                                                                           ║
# ║  LAYOUT DA TABELA                                                         ║
# ║    ┌──────────────────┬──────────────┬──────────────┬──────────────────┐  ║
# ║    │  h [m]           │  ρₕ [Ω·m]    │  λ = √(ρᵥ/ρₕ)│  ρᵥ [Ω·m] (auto) │  ║
# ║    ├──────────────────┼──────────────┼──────────────┼──────────────────┤  ║
# ║    │  8.0             │  20.0        │  1.414       │  40.0            │  ║
# ║    │  …               │  …           │  …           │  …               │  ║
# ║    └──────────────────┴──────────────┴──────────────┴──────────────────┘  ║
# ║                                                                           ║
# ║  CONVENÇÃO DE CAMADAS                                                     ║
# ║    - N linhas = N camadas totais (inclui semi-espaços superior/inferior)  ║
# ║    - espessuras: N-2 valores internos + 2 camadas de semi-espaço cujos    ║
# ║      "h" são interpretados pelo simulador como infinitos                  ║
# ║    - Convenção idêntica a buildValidamodels.py:                           ║
# ║        ncam = 3; esp = [8*pe]     →  3 camadas, 1 espessura interna       ║
# ║        ncam = 5; esp = [17,8,4]   →  5 camadas, 3 espessuras internas     ║
# ║      No dialog exibimos TODAS as N camadas mas marcamos a 1ª e última     ║
# ║      como "Semi-espaço" (h read-only = ∞).                                ║
# ║                                                                           ║
# ║  VALIDAÇÃO                                                                ║
# ║    h > 0, ρₕ ∈ [0.01, 1e6], λ ∈ [1.0, 5.0]. Erros destacam a célula em    ║
# ║    vermelho e impedem o accept().                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Caixa de diálogo modal para inserção manual de camadas TIV."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .sm_qt_compat import QT_AVAILABLE, QtCore, QtGui, QtWidgets

__all__ = ["LayersManualDialog"]


# Tamanho da fonte padrão nas células da tabela (um pouco maior que o
# default do QTableWidget para melhor legibilidade dos subscritos ₕ/ᵥ).
_CELL_FONT_SIZE = 12

# Cor de destaque para células com erro de validação (vermelho suave
# compatível com o tema escuro VSCode do Simulation Manager).
_ERROR_COLOR = "#5a2a2a"


class LayersManualDialog(QtWidgets.QDialog):
    """Editor manual de camadas para geração determinística de modelos.

    Attributes:
        table: ``QTableWidget`` com 4 colunas (h, ρₕ, λ, ρᵥ).
        _result: Dict retornado por :meth:`get_layers` após ``accept()``.

    Example:
        >>> dlg = LayersManualDialog(parent, initial={
        ...     "n_layers": 3,
        ...     "thicknesses": [8.0],      # 1 camada interna
        ...     "rho_h": [1.0, 10.0, 100.0],
        ...     "rho_v": [2.0, 20.0, 200.0],
        ... })
        >>> if dlg.exec():
        ...     layers = dlg.get_layers()
    """

    COL_H = 0
    COL_RHO_H = 1
    COL_LAMBDA = 2
    COL_RHO_V = 3

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        initial: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Entrada Manual de Camadas — Simulation Manager")
        self.setModal(True)
        self.resize(780, 520)

        self._result: Optional[Dict[str, Any]] = None
        self._updating_cells: bool = False  # evita recursão em cellChanged

        # ═══ Instruções ══════════════════════════════════════════════════
        lbl_help = QtWidgets.QLabel(
            "<b>Defina as camadas do modelo geológico.</b><br/>"
            "• Coluna <b>h</b>: espessura em metros (primeira e última linhas "
            "são semi-espaços — h é ignorado pelo simulador).<br/>"
            "• Coluna <b>ρₕ</b>: resistividade horizontal (Ω·m).<br/>"
            "• Coluna <b>λ</b>: fator de anisotropia TIV, λ = √(ρᵥ/ρₕ) ≥ 1.<br/>"
            "• Coluna <b>ρᵥ = ρₕ·λ²</b> é computada automaticamente "
            "(read-only)."
        )
        lbl_help.setWordWrap(True)
        lbl_help.setStyleSheet("color: #d4d4d4; padding: 4px;")

        # ═══ Controles de contagem de camadas ════════════════════════════
        self.spin_n_layers = QtWidgets.QSpinBox()
        self.spin_n_layers.setRange(2, 200)
        self.spin_n_layers.setValue(3)
        self.spin_n_layers.setSuffix(" camadas")
        self.spin_n_layers.valueChanged.connect(self._on_n_layers_changed)

        self.btn_add_row = QtWidgets.QPushButton("+ Linha")
        self.btn_del_row = QtWidgets.QPushButton("− Linha")
        self.btn_add_row.clicked.connect(self._on_add_row)
        self.btn_del_row.clicked.connect(self._on_del_row)

        row_count = QtWidgets.QHBoxLayout()
        row_count.addWidget(QtWidgets.QLabel("Nº de camadas:"))
        row_count.addWidget(self.spin_n_layers)
        row_count.addWidget(self.btn_add_row)
        row_count.addWidget(self.btn_del_row)
        row_count.addStretch(1)

        # ═══ Tabela ══════════════════════════════════════════════════════
        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(
            ["h [m]", "ρₕ [Ω·m]", "λ = √(ρᵥ/ρₕ)", "ρᵥ [Ω·m] (auto)"]
        )
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        for c in range(4):
            header.setSectionResizeMode(
                c,
                (
                    QtWidgets.QHeaderView.ResizeMode.Stretch
                    if hasattr(QtWidgets.QHeaderView, "ResizeMode")
                    else QtWidgets.QHeaderView.Stretch
                ),
            )
        self.table.verticalHeader().setDefaultSectionSize(28)
        font = QtGui.QFont()
        font.setPointSize(_CELL_FONT_SIZE)
        self.table.setFont(font)
        self.table.cellChanged.connect(self._on_cell_changed)

        # ═══ Rodapé (Aplicar / Cancelar) ═════════════════════════════════
        self.btn_apply = QtWidgets.QPushButton("Aplicar")
        self.btn_cancel = QtWidgets.QPushButton("Cancelar")
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel.clicked.connect(self.reject)

        row_btn = QtWidgets.QHBoxLayout()
        row_btn.addStretch(1)
        row_btn.addWidget(self.btn_cancel)
        row_btn.addWidget(self.btn_apply)

        # ═══ Layout raiz ═════════════════════════════════════════════════
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)
        root.addWidget(lbl_help)
        root.addLayout(row_count)
        root.addWidget(self.table, 1)
        root.addLayout(row_btn)

        # Popula com estado inicial (ou default 3 camadas)
        if initial:
            self._populate_from_initial(initial)
        else:
            self.spin_n_layers.setValue(3)
            self._populate_default_rows(3)

    # ── Popular tabela ────────────────────────────────────────────────────
    def _populate_default_rows(self, n: int) -> None:
        """Preenche N linhas com valores placeholder (1 Ω·m, isotrópico)."""
        self._updating_cells = True
        try:
            self.table.setRowCount(n)
            for r in range(n):
                self._set_row(r, h=1.0, rho_h=1.0, lam=1.0)
        finally:
            self._updating_cells = False

    def _populate_from_initial(self, initial: Dict[str, Any]) -> None:
        """Popula a tabela com valores de um CanonicalModel ou snapshot.

        Aceita chaves: ``n_layers``, ``thicknesses`` (N-2 valores internos
        OU N valores — preenche 0 nos semi-espaços), ``rho_h`` (N), ``rho_v``
        (N, opcional). Se ``rho_v`` for omitido, assume isotrópico.
        """
        rho_h = [float(x) for x in (initial.get("rho_h") or [])]
        rho_v = [float(x) for x in (initial.get("rho_v") or [])]
        n_layers = int(initial.get("n_layers") or len(rho_h) or 3)
        if n_layers < 2:
            n_layers = 2
        thicknesses_raw = [float(x) for x in (initial.get("thicknesses") or [])]

        # thicknesses pode vir como N-2 (só internas) ou N (inclui semi-espaços)
        if len(thicknesses_raw) == n_layers - 2:
            thicknesses = [0.0] + thicknesses_raw + [0.0]
        elif len(thicknesses_raw) == n_layers:
            thicknesses = thicknesses_raw
        else:
            thicknesses = [0.0] * n_layers
            for i, v in enumerate(thicknesses_raw):
                if i + 1 < len(thicknesses):
                    thicknesses[i + 1] = v  # preserva internas

        if not rho_v:
            rho_v = list(rho_h)  # isotrópico default
        # Alinha tamanhos com n_layers (pad com último valor)
        while len(rho_h) < n_layers:
            rho_h.append(rho_h[-1] if rho_h else 1.0)
        while len(rho_v) < n_layers:
            rho_v.append(rho_v[-1] if rho_v else 1.0)

        self._updating_cells = True
        try:
            blocker = QtCore.QSignalBlocker(self.spin_n_layers)
            self.spin_n_layers.setValue(n_layers)
            del blocker
            self.table.setRowCount(n_layers)
            for r in range(n_layers):
                rh = float(rho_h[r])
                rv = float(rho_v[r])
                lam = (rv / rh) ** 0.5 if rh > 0 else 1.0
                h_val = float(thicknesses[r]) if r < len(thicknesses) else 0.0
                self._set_row(r, h=h_val, rho_h=rh, lam=lam)
        finally:
            self._updating_cells = False

    def _set_row(self, r: int, *, h: float, rho_h: float, lam: float) -> None:
        """Define os 4 valores de uma linha (ρᵥ auto)."""
        is_boundary = r == 0 or r == self.table.rowCount() - 1
        # h
        item_h = QtWidgets.QTableWidgetItem()
        if is_boundary:
            item_h.setText("∞ (semi-espaço)")
            item_h.setFlags(item_h.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_h.setForeground(QtGui.QColor("#888888"))
        else:
            item_h.setText(f"{h:.4f}")
        self.table.setItem(r, self.COL_H, item_h)
        # rho_h
        item_rh = QtWidgets.QTableWidgetItem(f"{rho_h:.4f}")
        self.table.setItem(r, self.COL_RHO_H, item_rh)
        # lambda
        item_l = QtWidgets.QTableWidgetItem(f"{lam:.4f}")
        self.table.setItem(r, self.COL_LAMBDA, item_l)
        # rho_v (read-only derivado)
        rv = rho_h * (lam**2)
        item_rv = QtWidgets.QTableWidgetItem(f"{rv:.4f}")
        item_rv.setFlags(item_rv.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item_rv.setForeground(QtGui.QColor("#a5a5a5"))
        self.table.setItem(r, self.COL_RHO_V, item_rv)

    # ── Signals ───────────────────────────────────────────────────────────
    def _on_n_layers_changed(self, n: int) -> None:
        """Spinbox alterou o número de camadas — ajusta rowCount."""
        if self._updating_cells:
            return
        current = self.table.rowCount()
        if n == current:
            return
        self._updating_cells = True
        try:
            if n > current:
                for r in range(current, n):
                    self.table.insertRow(r)
                    self._set_row(r, h=1.0, rho_h=1.0, lam=1.0)
            else:
                while self.table.rowCount() > n:
                    self.table.removeRow(self.table.rowCount() - 1)
            # Re-marca linhas de borda (primeira e última) como semi-espaço
            self._refresh_boundaries()
        finally:
            self._updating_cells = False

    def _refresh_boundaries(self) -> None:
        """Atualiza semi-espaço de 1ª/última linha (após insert/remove)."""
        n = self.table.rowCount()
        if n < 2:
            return
        for r in (0, n - 1):
            item_h = self.table.item(r, self.COL_H)
            if item_h is None:
                continue
            item_h.setText("∞ (semi-espaço)")
            item_h.setFlags(item_h.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_h.setForeground(QtGui.QColor("#888888"))
        # Linhas do meio voltam a ser editáveis (caso tenham sido marcadas)
        for r in range(1, n - 1):
            item_h = self.table.item(r, self.COL_H)
            if item_h is None:
                continue
            txt = item_h.text()
            if txt.startswith("∞"):
                item_h.setText("1.0000")
            item_h.setFlags(item_h.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item_h.setForeground(QtGui.QColor("#d4d4d4"))

    def _on_add_row(self) -> None:
        self.spin_n_layers.setValue(self.spin_n_layers.value() + 1)

    def _on_del_row(self) -> None:
        if self.spin_n_layers.value() > 2:
            self.spin_n_layers.setValue(self.spin_n_layers.value() - 1)

    def _on_cell_changed(self, row: int, col: int) -> None:
        """Recalcula ρᵥ quando ρₕ ou λ mudam. Limpa highlight de erro."""
        if self._updating_cells:
            return
        if col not in (self.COL_RHO_H, self.COL_LAMBDA):
            return
        try:
            rh_item = self.table.item(row, self.COL_RHO_H)
            l_item = self.table.item(row, self.COL_LAMBDA)
            if rh_item is None or l_item is None:
                return
            rh = float(rh_item.text().replace(",", "."))
            lam = float(l_item.text().replace(",", "."))
            rv = rh * (lam**2)
            self._updating_cells = True
            try:
                rv_item = QtWidgets.QTableWidgetItem(f"{rv:.4f}")
                rv_item.setFlags(rv_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                rv_item.setForeground(QtGui.QColor("#a5a5a5"))
                self.table.setItem(row, self.COL_RHO_V, rv_item)
            finally:
                self._updating_cells = False
            # Remove highlight de erro (se havia)
            for c in (self.COL_RHO_H, self.COL_LAMBDA):
                item = self.table.item(row, c)
                if item is not None:
                    item.setBackground(QtGui.QColor("transparent"))
        except ValueError:
            # entrada inválida — destaca em vermelho
            item = self.table.item(row, col)
            if item is not None:
                item.setBackground(QtGui.QColor(_ERROR_COLOR))

    # ── Aplicar / Validar ─────────────────────────────────────────────────
    def _on_apply(self) -> None:
        """Valida e aceita. Se houver erro, mostra QMessageBox e mantém aberto."""
        layers = self._validate_and_collect()
        if layers is None:
            return
        self._result = layers
        self.accept()

    def _validate_and_collect(self) -> Optional[Dict[str, Any]]:
        """Lê a tabela; valida h/ρₕ/λ; retorna dict ou None se houver erro."""
        n = self.table.rowCount()
        if n < 2:
            QtWidgets.QMessageBox.warning(
                self, "Modelo inválido", "Mínimo 2 camadas (2 semi-espaços)."
            )
            return None

        thicknesses_internal: List[float] = []
        rho_h: List[float] = []
        rho_v: List[float] = []
        errors: List[str] = []

        for r in range(n):
            is_boundary = r == 0 or r == n - 1
            # h
            if not is_boundary:
                item_h = self.table.item(r, self.COL_H)
                try:
                    h = float((item_h.text() if item_h else "0").replace(",", "."))
                    if h <= 0:
                        errors.append(f"Linha {r+1}: h deve ser > 0 (obtido {h}).")
                    thicknesses_internal.append(h)
                except (ValueError, AttributeError):
                    errors.append(f"Linha {r+1}: h inválido.")
                    thicknesses_internal.append(0.0)
            # rho_h
            item_rh = self.table.item(r, self.COL_RHO_H)
            try:
                rh = float((item_rh.text() if item_rh else "0").replace(",", "."))
                if rh < 0.01 or rh > 1e6:
                    errors.append(
                        f"Linha {r+1}: ρₕ fora de [0.01, 1e6] Ω·m (obtido {rh})."
                    )
                rho_h.append(rh)
            except (ValueError, AttributeError):
                errors.append(f"Linha {r+1}: ρₕ inválido.")
                rho_h.append(1.0)
            # lambda → rho_v
            item_l = self.table.item(r, self.COL_LAMBDA)
            try:
                lam = float((item_l.text() if item_l else "1").replace(",", "."))
                if lam < 1.0 or lam > 5.0:
                    errors.append(f"Linha {r+1}: λ fora de [1.0, 5.0] (obtido {lam}).")
                rho_v.append(rho_h[-1] * lam * lam)
            except (ValueError, AttributeError):
                errors.append(f"Linha {r+1}: λ inválido.")
                rho_v.append(rho_h[-1])

        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Valores inválidos",
                "Corrija os erros abaixo:<br/><br/>"
                + "<br/>".join(f"• {e}" for e in errors[:15])
                + ("<br/>(...)" if len(errors) > 15 else ""),
            )
            return None

        return {
            "n_layers": n,
            "thicknesses": thicknesses_internal,  # N-2 internas (convenção SM)
            "rho_h": rho_h,
            "rho_v": rho_v,
        }

    def get_layers(self) -> Optional[Dict[str, Any]]:
        """Retorna o dict validado após ``accept()``, ou ``None``."""
        return self._result
