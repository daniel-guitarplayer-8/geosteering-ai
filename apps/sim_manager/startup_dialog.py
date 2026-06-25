# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/startup_dialog.py                                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ProjectStartupDialog — abrir/criar projeto no boot         ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — startup (PR-3 #7a)                          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : Qt6 via gui.qt_compat                                       ║
# ║  Dependências: gui.qt_compat, .experiments_service, .experiment_state      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Diálogo de boas-vindas (espelha o WelcomeWidget do SM monólito): ao      ║
# ║    iniciar, o usuário ABRE ou CRIA um PROJETO. Um projeto = uma PASTA com    ║
# ║    os artefatos (.exp.json + .dat/.out gerados). Reusa ExperimentsService/  ║
# ║    ExperimentState/NewExperimentDialog (não inventa modelo de persistência).║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ProjectStartupDialog · resolve_project                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ProjectStartupDialog`` — abrir/criar projeto no startup do SM (PR-3 #7a)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from apps.sim_manager.perspectives.simulation.experiment_state import ExperimentState
from geosteering_ai.gui.qt_compat import QtWidgets

__all__ = ["ProjectStartupDialog", "resolve_project"]

logger = logging.getLogger(__name__)


def resolve_project(service: Any, directory: str) -> ExperimentState:
    """Resolve uma PASTA de projeto num ``ExperimentState`` (lógica PURA, sem Qt).

    Se a pasta contém um ``.exp.json``, carrega-o (projeto existente). Senão, cria
    um ``ExperimentState`` novo em RAM apontando para a pasta (materializado em disco
    ao salvar). Helper PURO (testável com um service stub) — espelha o "abrir pasta"
    do WelcomeWidget do monólito.

    Args:
        service: objeto com ``load_experiment(path) -> ExperimentState`` (duck-typed).
        directory: caminho da pasta do projeto.

    Returns:
        O ``ExperimentState`` carregado (se havia ``.exp.json``) ou um novo.
    """
    d = Path(directory)
    exp_files = sorted(d.glob("*.exp.json"))
    if exp_files:
        return service.load_experiment(str(exp_files[0]))
    return ExperimentState(name=d.name or "Projeto", output_dir=str(d))


class ProjectStartupDialog(QtWidgets.QDialog):  # type: ignore[misc] # QtWidgets é Any
    """Diálogo de boas-vindas: ABRIR ou CRIAR um projeto (pasta com artefatos).

    Args:
        service: o :class:`ExperimentsService` (injetado — testável).
        parent: widget pai opcional.

    Note:
        Ao concluir, :attr:`result_state` guarda o ``ExperimentState`` escolhido
        (``None`` se cancelado). ``exec()`` retorna ``Accepted`` quando um projeto
        foi criado/aberto. Mostra os RECENTES (``service.load_recents()``).
    """

    def __init__(self, service: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self._service = service
        self.result_state: Optional[ExperimentState] = None
        self.setWindowTitle("Geosteering AI — Simulation Manager")
        self.resize(560, 420)

        title = QtWidgets.QLabel("Simulation Manager")
        title.setProperty("role", "heading")
        subtitle = QtWidgets.QLabel(
            "Abra um projeto existente ou crie um novo. Um projeto é uma pasta com os "
            "artefatos do experimento (.exp.json, .dat/.out)."
        )
        subtitle.setProperty("role", "hint")
        subtitle.setWordWrap(True)

        self._new_btn = QtWidgets.QPushButton("Novo projeto…")
        self._new_btn.setProperty("role", "primary")
        self._open_btn = QtWidgets.QPushButton("Abrir projeto…")
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self._new_btn)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch(1)

        self._recents = QtWidgets.QListWidget()
        self._recents.setToolTip("Projetos recentes (duplo-clique para abrir).")
        self._populate_recents()

        quit_btn = QtWidgets.QPushButton("Sair")
        quit_btn.setProperty("role", "ghost")
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(quit_btn)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(title)
        root.addWidget(subtitle)
        root.addLayout(btn_row)
        root.addWidget(QtWidgets.QLabel("Recentes:"))
        root.addWidget(self._recents, stretch=1)
        root.addLayout(bottom)

        self._new_btn.clicked.connect(self._on_new)
        self._open_btn.clicked.connect(self._on_open)
        self._recents.itemDoubleClicked.connect(self._on_recent_activated)
        quit_btn.clicked.connect(self.reject)

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def _populate_recents(self) -> None:
        """Preenche a lista com os recentes (caminho COMPLETO no UserRole)."""
        from geosteering_ai.gui.qt_compat import Qt

        self._recents.clear()
        for path in self._service.load_recents():
            item = QtWidgets.QListWidgetItem(path)
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._recents.addItem(item)

    def _accept_with(self, state: ExperimentState) -> None:
        """Registra o projeto escolhido, empilha nos recentes e aceita o diálogo."""
        self.result_state = state
        if state.file_path:
            self._service.push_recent(state.file_path)
        self.accept()

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_new(self) -> None:
        """Cria um projeto novo (reusa NewExperimentDialog + create_experiment)."""
        from apps.sim_manager.perspectives.simulation.experiments_view import (
            NewExperimentDialog,
        )

        dlg = NewExperimentDialog(parent=self)
        if not dlg.exec():
            return
        name, desc, out_dir = dlg.values()
        try:
            os.makedirs(
                out_dir, exist_ok=True
            )  # paridade com o monólito (cria a pasta)
            exp = self._service.create_experiment(name, desc, out_dir)
            self._service.save_experiment_async(exp)  # materializa o .exp.json
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Falha ao criar projeto", f"Não foi possível criar:\n{exc}"
            )
            return
        self._accept_with(exp)

    def _on_open(self) -> None:
        """Abre uma PASTA de projeto (resolve o .exp.json ou cria em RAM)."""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Abrir pasta do projeto"
        )
        if not path:
            return
        try:
            state = resolve_project(self._service, path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Falha ao abrir projeto", f"Não foi possível abrir:\n{exc}"
            )
            return
        self._accept_with(state)

    def _on_recent_activated(self, item: Any) -> None:
        """Abre um projeto recente (caminho .exp.json no UserRole)."""
        from geosteering_ai.gui.qt_compat import Qt

        path = item.data(Qt.ItemDataRole.UserRole) or item.text()
        try:
            state = self._service.load_experiment(path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Falha ao abrir recente", f"Não foi possível abrir:\n{exc}"
            )
            return
        self._accept_with(state)
