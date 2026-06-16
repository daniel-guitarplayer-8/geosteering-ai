# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/preferences/perspective.py                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : PreferencesPerspective — plugin de aba (MVVM)              ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app — perspectiva Preferências (Fatia 6e)               ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: gui.shell.Perspective, .viewmodel, .service, .view          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementa o contrato ``Perspective``: cria o ViewModel PURO (com o     ║
# ║    PreferencesService real injetado, já carregado do disco) e a View Qt    ║
# ║    ligada a ele. ``build_view`` usa ``build_viewmodel`` internamente —      ║
# ║    testes podem chamar este último direto para exercitar o VM sem Qt.      ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    PreferencesPerspective                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``PreferencesPerspective`` — contrato Perspective das Preferências (Fatia 6e)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apps.sim_manager.perspectives.preferences.service import PreferencesService
from apps.sim_manager.perspectives.preferences.viewmodel import PreferencesViewModel
from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.shell.perspective import Perspective

if TYPE_CHECKING:  # pragma: no cover — só type-checking (via qt_compat, não PyQt6)
    from geosteering_ai.gui.qt_compat import QtWidgets

__all__ = ["PreferencesPerspective"]


class PreferencesPerspective(Perspective):
    """Perspectiva Preferências — cria PreferencesVM (Service injetado) + Panel."""

    id = "preferences"
    title = "Preferências"
    icon = "gear"
    icon_glyph = "⚙"  # ícone na activity rail Antigravity (spec 0013)
    order = 4

    def build_viewmodel(self, ctx: AppContext) -> PreferencesViewModel:
        """Cria o ViewModel PURO com o :class:`PreferencesService` real, já carregado.

        Args:
            ctx: contexto da aplicação (não usado nesta fatia).

        Returns:
            Um :class:`PreferencesViewModel` com as preferências do disco aplicadas
            (1º boot → defaults, via degradação graciosa do serviço).
        """
        vm = PreferencesViewModel(service=PreferencesService())
        vm.load()  # reflete o preferences.json persistido (ou defaults no 1º boot)
        return vm

    def build_view(self, ctx: AppContext) -> "QtWidgets.QWidget":
        """Cria a View Qt ligada a um ViewModel novo (já carregado do disco).

        Args:
            ctx: contexto da aplicação.

        Returns:
            O :class:`PreferencesPanel` (``QWidget``) raiz da perspectiva.
        """
        # Import local (Views importam Qt) — mantém este módulo leve/testável.
        from apps.sim_manager.perspectives.preferences.view import PreferencesPanel

        vm = self.build_viewmodel(ctx)
        view = PreferencesPanel(vm)
        self._vm = vm  # ref viva (evita GC do VM enquanto a View existe)
        self._view = view

        # ── Observabilidade (opcional) — loga a PERSISTÊNCIA na secondary sidebar ──
        # Liga-se a ``saved`` (evento de persistência), NÃO aos sinais por-campo:
        # estes disparam também em "Restaurar padrões" (que NÃO persiste), o que
        # produziria um log enganoso ("efetivo no próximo boot") sem nada salvo.
        # A aplicação ao vivo às demais perspectivas é fatia futura.
        sidebar = ctx.extras.get("secondary_sidebar")
        if sidebar is not None and hasattr(sidebar, "append_log"):
            vm.saved.connect(
                lambda: sidebar.append_log(
                    f"⚙ Preferências salvas (tema={vm.theme}, "
                    f"plot={vm.plot_backend}) — efetivo no próximo boot."
                )
            )
        return view
