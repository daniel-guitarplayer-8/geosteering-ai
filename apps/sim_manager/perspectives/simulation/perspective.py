# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/perspective.py                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimulationPerspective — plugin de aba (MVVM)               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app — perspectiva Simulação (spec 0011a)                ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — walking skeleton                                ║
# ║  Dependências: gui.shell.Perspective, gui.services, .viewmodel, .view      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementa o contrato ``Perspective``: cria o ViewModel PURO (com o     ║
# ║    Service real injetado) e a View Qt ligada a ele. ``build_view`` usa     ║
# ║    ``build_viewmodel`` internamente — testes podem chamar este último      ║
# ║    direto para exercitar o VM sem Qt.                                      ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimulationPerspective                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimulationPerspective`` — contrato Perspective da Simulação (0011a)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
from geosteering_ai.gui.services import SimulationService
from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.shell.perspective import Perspective

if TYPE_CHECKING:  # pragma: no cover — só type-checking (via qt_compat, não PyQt6)
    from geosteering_ai.gui.qt_compat import QtWidgets

__all__ = ["SimulationPerspective"]


def _history_label(result: dict) -> str:
    """Rótulo curto de um resultado p/ o Histórico da sidebar (Fatia 6a)."""
    h6 = result.get("H6")
    n = getattr(h6, "shape", (None,))[0] if h6 is not None else "?"
    return f"sim · backend={result.get('backend', '?')} · {n} modelo(s)"


class SimulationPerspective(Perspective):
    """Perspectiva Simulação — cria SimulationVM (Service injetado) + SimulatorView."""

    id = "simulation"
    title = "Simulação"
    icon = "flask"
    icon_glyph = "🧪"  # ícone na activity rail Antigravity (spec 0013)
    order = 0

    def build_viewmodel(self, ctx: AppContext) -> SimulationViewModel:
        """Cria o ViewModel PURO com o :class:`SimulationService` real injetado.

        Args:
            ctx: contexto da aplicação (não usado nesta fatia).

        Returns:
            Um :class:`SimulationViewModel` pronto.
        """
        return SimulationViewModel(service=SimulationService())

    def build_view(self, ctx: AppContext) -> "QtWidgets.QWidget":
        """Cria a View Qt ligada a um ViewModel novo (via :meth:`build_viewmodel`).

        Args:
            ctx: contexto da aplicação.

        Returns:
            A :class:`SimulatorView` (``QWidget``) raiz da perspectiva.
        """
        # Import local (View importa Qt) — mantém este módulo leve até a UI subir.
        from apps.sim_manager.perspectives.simulation.view import SimulatorView

        viewmodel = self.build_viewmodel(ctx)
        # Liga o feedback (Fatia 6a) à secondary sidebar do shell (Histórico/Log),
        # via ctx.extras — sem acoplar o ViewModel (puro) ao shell.
        sidebar = ctx.extras.get("secondary_sidebar")
        if sidebar is not None:
            viewmodel.log_entry.connect(sidebar.append_log)
            viewmodel.result_ready.connect(
                lambda result: sidebar.add_history_item(_history_label(result))
            )
        return SimulatorView(viewmodel)
