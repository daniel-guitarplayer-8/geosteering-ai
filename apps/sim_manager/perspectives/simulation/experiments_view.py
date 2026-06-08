# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/experiments_view.py            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ExperimentsPanel + NewExperimentDialog (Qt)               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — experimentos (spec 0016, Fatia 6c)        ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — experimentos & histórico                        ║
# ║  Framework   : Qt6 via gui.qt_compat                                       ║
# ║  Padrão      : View (MVVM) — emite sinais; sem lógica de domínio           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Painel da aba "Histórico" (secondary sidebar): toolbar Novo/Abrir/      ║
# ║    Salvar/Fechar + busca + lista de snapshots (● em cache / ○ fora) +      ║
# ║    info + recentes. Double-click recarrega um snapshot. Sem lógica — só    ║
# ║    sinais que a perspectiva liga ao ViewModel.                             ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ExperimentsPanel · NewExperimentDialog                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ExperimentsPanel`` + ``NewExperimentDialog`` — histórico/experimentos (0016)."""

from __future__ import annotations

from typing import Any, List, Optional

from geosteering_ai.gui.qt_compat import Qt, QtWidgets, Signal

__all__ = ["ExperimentsPanel", "NewExperimentDialog"]

_SNAP_ROLE = int(Qt.ItemDataRole.UserRole)


def _no_hscroll(widget: Any, elide: "Qt.TextElideMode") -> None:
    """Configura um ``QListWidget`` p/ NÃO vazar a coluna (elide + sem h-scroll).

    Elide é só de pintura — ``item.text()`` permanece íntegro (emits corretos).
    ``setUniformItemSizes`` acelera o layout de listas grandes.
    """
    widget.setWordWrap(False)
    widget.setUniformItemSizes(True)
    widget.setTextElideMode(elide)
    widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)


class NewExperimentDialog(QtWidgets.QDialog):  # type: ignore[misc] # QtWidgets é Any
    """Dialog de criação de experimento (nome + descrição + diretório de saída)."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Novo experimento")
        self.resize(420, 220)
        self._name = QtWidgets.QLineEdit("Experimento")
        self._desc = QtWidgets.QPlainTextEdit()
        self._desc.setMaximumHeight(80)
        self._dir = QtWidgets.QLineEdit("sm_experiments")
        browse = QtWidgets.QPushButton("…")
        browse.setMaximumWidth(32)
        browse.clicked.connect(self._browse)

        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(self._dir, stretch=1)
        dir_row.addWidget(browse)
        form = QtWidgets.QFormLayout()
        form.addRow("Nome:", self._name)
        form.addRow("Descrição:", self._desc)
        form.addRow("Diretório:", dir_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(buttons)

    def _browse(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Diretório de saída")
        if path:
            self._dir.setText(path)

    def values(self) -> tuple:
        """Retorna ``(name, description, output_dir)``."""
        return (
            self._name.text().strip(),
            self._desc.toPlainText().strip(),
            self._dir.text().strip() or "sm_experiments",
        )


class ExperimentsPanel(QtWidgets.QWidget):  # type: ignore[misc] # QtWidgets é Any
    """Painel de experimentos & histórico (aba Histórico da secondary sidebar)."""

    request_new = Signal()
    request_open = Signal()
    request_save = Signal()
    request_close = Signal()
    request_clear = Signal()
    snapshot_activated = Signal(str)  # double-click (snap_id) → recarrega
    snapshot_selected = Signal(str)  # clique (snap_id) → mostra info
    recent_activated = Signal(str)  # double-click num recente (path)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)

        # ── Toolbar ───────────────────────────────────────────────────────
        bar = QtWidgets.QHBoxLayout()
        for text, sig in (
            ("Novo", self.request_new),
            ("Abrir", self.request_open),
            ("Salvar", self.request_save),
            ("Fechar", self.request_close),
        ):
            btn = QtWidgets.QPushButton(text)
            btn.setProperty("role", "ghost")
            btn.clicked.connect(sig.emit)
            bar.addWidget(btn)
        bar.addStretch(1)

        self._exp_label = QtWidgets.QLabel("(sem experimento)")
        self._exp_label.setWordWrap(True)
        self._exp_label.setProperty("role", "section")

        # ── Busca + lista de snapshots ────────────────────────────────────
        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText("Filtrar histórico…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(lambda _t: self._apply_filter())

        self._list = QtWidgets.QListWidget()
        # Anti-overflow (sem vazar a coluna): elide o texto + NUNCA scrollbar
        # horizontal. ``item.text()`` permanece o texto COMPLETO (elide é só
        # pintura), então double-click/emit continuam corretos.
        _no_hscroll(self._list, Qt.TextElideMode.ElideRight)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.currentItemChanged.connect(self._on_current_changed)

        self._info = QtWidgets.QPlainTextEdit()
        self._info.setReadOnly(True)
        self._info.setMaximumHeight(110)
        self._info.setPlaceholderText("Selecione um snapshot…")

        clear_btn = QtWidgets.QPushButton("Limpar histórico")
        clear_btn.setProperty("role", "danger")
        clear_btn.clicked.connect(self.request_clear.emit)

        # ── Recentes ──────────────────────────────────────────────────────
        self._recents = QtWidgets.QListWidget()
        self._recents.setMaximumHeight(90)
        # Caminhos absolutos longos: elide no MEIO (preserva o basename) + sem
        # scrollbar horizontal. ``item.text()`` segue completo → emit do caminho
        # íntegro (não truncado). Tooltip por item mostra o caminho completo.
        _no_hscroll(self._recents, Qt.TextElideMode.ElideMiddle)
        self._recents.itemDoubleClicked.connect(
            lambda it: self.recent_activated.emit(it.text())
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addLayout(bar)
        root.addWidget(self._exp_label)
        root.addWidget(self._search)
        root.addWidget(self._list, stretch=1)
        root.addWidget(self._info)
        root.addWidget(clear_btn)
        root.addWidget(QtWidgets.QLabel("Recentes:"))
        root.addWidget(self._recents)

    # ── API (chamada pela perspectiva) ─────────────────────────────────────
    def set_experiment_label(self, name: Optional[str], path: str = "") -> None:
        """Atualiza o rótulo do experimento (nome no rótulo; caminho no tooltip).

        Exibir o caminho absoluto INLINE vaza a coluna (um path sem espaços não
        quebra com ``wordWrap``). Mostramos só o nome; o caminho completo fica no
        tooltip + no campo Exp da status bar.
        """
        if name:
            self._exp_label.setText(name)
            self._exp_label.setToolTip(f"{name}  ·  {path}" if path else name)
        else:
            self._exp_label.setText("(sem experimento)")
            self._exp_label.setToolTip("")

    def add_snapshot(
        self, snap_id: str, label: str, *, in_cache: bool, info: str = ""
    ) -> None:
        """Insere um snapshot no TOPO da lista (● em cache / ○ fora)."""
        item = QtWidgets.QListWidgetItem(("● " if in_cache else "○ ") + label)
        item.setData(_SNAP_ROLE, snap_id)
        if info:
            item.setToolTip(info)
        self._list.insertItem(0, item)
        self._apply_filter()

    def mark_out_of_cache(self, snap_id: str) -> None:
        """Troca ● → ○ no item do snapshot (evicção do cache)."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(_SNAP_ROLE) == snap_id and item.text().startswith("● "):
                item.setText("○ " + item.text()[2:])
                return

    def set_snapshot_info(self, text: str) -> None:
        """Atualiza o painel de info do snapshot selecionado."""
        self._info.setPlainText(text)

    def set_recents(self, paths: List[str]) -> None:
        """Popula a lista de recentes (tooltip = caminho completo; texto elidido)."""
        self._recents.clear()
        for path in paths:
            item = QtWidgets.QListWidgetItem(path)
            item.setToolTip(path)  # caminho completo no hover (texto pintado elide)
            self._recents.addItem(item)

    def clear_history(self) -> None:
        """Limpa a lista + info."""
        self._list.clear()
        self._info.clear()

    # ── Slots internos ──────────────────────────────────────────────────────
    def _apply_filter(self) -> None:
        needle = self._search.text().lower().strip()
        for i in range(self._list.count()):
            item = self._list.item(i)
            hay = (item.text() + "\n" + (item.toolTip() or "")).lower()
            item.setHidden(bool(needle) and needle not in hay)

    def _on_double_click(self, item: Any) -> None:
        snap_id = item.data(_SNAP_ROLE)
        if snap_id:
            self.snapshot_activated.emit(str(snap_id))

    def _on_current_changed(self, current: Any, _prev: Any) -> None:
        if current is not None:
            snap_id = current.data(_SNAP_ROLE)
            if snap_id:
                self.snapshot_selected.emit(str(snap_id))
