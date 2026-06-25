# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/results/perspective.py                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ResultsPerspective — plugin de aba (MVVM)                  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app — perspectiva Resultados (Fatia 6i / PR-2)          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: gui.shell.Perspective, results_view, results_viewmodel      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Hospeda a galeria do ensemble (antes embutida na aba Simulação). Reusa  ║
# ║    o ``ResultsViewModel`` JÁ EXISTENTE da Simulação (1 VM, 2 Views): a      ║
# ║    SimulationPerspective publica ``ctx.extras["results_vm"]`` e esta liga   ║
# ║    a ``ResultsView`` ao MESMO VM — o resultado simulado aparece aqui. Se     ║
# ║    nenhuma simulação rodou ainda (ou ordem de registro mudou), cria um VM   ║
# ║    vazio (galeria "sem resultado", sem crash).                             ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ResultsPerspective                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ResultsPerspective`` — galeria do ensemble numa perspectiva dedicada (6i)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apps.sim_manager.perspectives.simulation.results_viewmodel import ResultsViewModel
from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.shell.perspective import Perspective

if TYPE_CHECKING:  # pragma: no cover — só type-checking (via qt_compat, não PyQt6)
    from geosteering_ai.gui.qt_compat import QtWidgets

logger = logging.getLogger(__name__)

__all__ = ["ResultsPerspective"]


class ResultsPerspective(Perspective):
    """Perspectiva Resultados — galeria do ensemble (reusa o ResultsViewModel da Simulação)."""

    id = "results"
    title = "Resultados"
    icon = "chart"
    icon_glyph = "📊"  # ícone na activity rail Antigravity (spec 0013)
    order = 1

    def build_viewmodel(self, ctx: AppContext) -> ResultsViewModel:
        """Reusa o ``ResultsViewModel`` da Simulação (via ctx.extras) ou cria um vazio.

        Args:
            ctx: contexto da aplicação. Espera ``ctx.extras["results_vm"]`` publicado
                pela SimulationPerspective (Simulação tem order=0 → builda no boot,
                antes de Resultados ser ativada).

        Returns:
            O ``ResultsViewModel`` compartilhado da Simulação, ou um novo vazio
            (fallback defensivo — galeria "sem resultado" em vez de crash).
        """
        vm = ctx.extras.get("results_vm")
        if isinstance(vm, ResultsViewModel):
            return vm
        # Fallback defensivo: sem o VM compartilhado, a galeria fica vazia (não
        # crasha). Loga p/ TORNAR VISÍVEL uma regressão de fiação (ex.: ordem de
        # registro alterada em app.py — Simulação order=0 deve buildar 1º e publicar
        # results_vm antes de Resultados ativar). Hoje é inalcançável (ordem garante).
        logger.warning(
            "ResultsPerspective: ctx.extras['results_vm'] ausente/inválido — usando VM "
            "detached (galeria ficará vazia). Verifique a ordem de registro em app.py "
            "(SimulationPerspective order=0 deve buildar antes de Resultados)."
        )
        return ResultsViewModel()

    def build_view(self, ctx: AppContext) -> "QtWidgets.QWidget":
        """Cria a ``ResultsView`` ligada ao ResultsViewModel compartilhado.

        Args:
            ctx: contexto da aplicação.

        Returns:
            A :class:`ResultsView` (``QWidget``) raiz da perspectiva.
        """
        # Import local (Views importam Qt) — mantém este módulo leve/testável.
        from apps.sim_manager.perspectives.results.view import ResultsView

        vm = self.build_viewmodel(ctx)
        view = ResultsView(vm)
        self._vm = vm  # ref viva (evita GC do VM enquanto a View existe)
        self._view = view
        return view
