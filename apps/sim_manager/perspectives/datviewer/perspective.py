# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/datviewer/perspective.py                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : DatViewerPerspective — plugin de aba (MVVM)                ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app — perspectiva Visualizador .dat (Fatia 6h)           ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: gui.shell.Perspective, .viewmodel, .service, .view          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementa o contrato ``Perspective``: cria o ViewModel PURO (com o      ║
# ║    DatViewerService injetado) e a View Qt ligada a ele. ``build_view`` usa   ║
# ║    ``build_viewmodel`` internamente — testes exercitam o VM sem Qt.         ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    DatViewerPerspective                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``DatViewerPerspective`` — contrato Perspective do visualizador ``.dat`` (6h)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apps.sim_manager.perspectives.datviewer.service import DatViewerService
from apps.sim_manager.perspectives.datviewer.viewmodel import DatViewerViewModel
from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.shell.perspective import Perspective

if TYPE_CHECKING:  # pragma: no cover — só type-checking (via qt_compat, não PyQt6)
    from geosteering_ai.gui.qt_compat import QtWidgets

__all__ = ["DatViewerPerspective"]


class DatViewerPerspective(Perspective):
    """Perspectiva Visualizador .dat — cria DatViewerVM (Service injetado) + Panel."""

    id = "datviewer"
    title = "Visualizador .dat"
    icon = "table"
    icon_glyph = "📄"  # ícone na activity rail Antigravity (spec 0013)
    order = 5

    def build_viewmodel(self, ctx: AppContext) -> DatViewerViewModel:
        """Cria o ViewModel PURO com o :class:`DatViewerService` real injetado.

        Args:
            ctx: contexto da aplicação (não usado nesta fatia).

        Returns:
            Um :class:`DatViewerViewModel` pronto (sem arquivo carregado ainda).
        """
        return DatViewerViewModel(service=DatViewerService())

    def build_view(self, ctx: AppContext) -> "QtWidgets.QWidget":
        """Cria a View Qt ligada a um ViewModel novo.

        Args:
            ctx: contexto da aplicação.

        Returns:
            O :class:`DatViewerPanel` (``QWidget``) raiz da perspectiva.
        """
        # Import local (Views importam Qt) — mantém este módulo leve/testável.
        from apps.sim_manager.perspectives.datviewer.view import DatViewerPanel

        vm = self.build_viewmodel(ctx)
        view = DatViewerPanel(vm)
        self._vm = vm  # ref viva (evita GC do VM enquanto a View existe)
        self._view = view

        # ── Observabilidade (opcional) — loga leituras na secondary sidebar ──
        sidebar = ctx.extras.get("secondary_sidebar")
        if sidebar is not None and hasattr(sidebar, "append_log"):
            vm.loaded.connect(
                lambda res: sidebar.append_log(
                    f"📄 .dat carregado: {res.n_rows}×{res.n_cols} ({res.fmt})."
                )
            )
            vm.load_error.connect(lambda msg: sidebar.append_log(f"📄 Erro: {msg}"))
        return view
