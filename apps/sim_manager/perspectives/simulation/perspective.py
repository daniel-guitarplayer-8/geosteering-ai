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


class SimulationPerspective(Perspective):
    """Perspectiva Simulação — cria SimulationVM (Service injetado) + SimulatorView."""

    id = "simulation"
    title = "Simulação"
    icon = "flask"
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
        return SimulatorView(viewmodel)
