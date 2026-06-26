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

import os
from typing import TYPE_CHECKING

from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
from geosteering_ai.gui.services import SimulationService
from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.shell.perspective import Perspective

if TYPE_CHECKING:  # pragma: no cover — só type-checking (via qt_compat, não PyQt6)
    from geosteering_ai.gui.qt_compat import QtWidgets

__all__ = ["SimulationPerspective"]


def _snapshot_info(snap: object) -> str:
    """Texto de info de um snapshot p/ o painel de histórico (Fatia 6c)."""
    return (
        f"{getattr(snap, 'label', '')}\n"
        f"timestamp: {getattr(snap, 'timestamp', '—')}\n"
        f"backend: {getattr(snap, 'backend', '—')}\n"
        f"nº modelos: {getattr(snap, 'n_models', '—')}\n"
        f"tempo: {getattr(snap, 'elapsed_s', 0.0):.1f} s"
    )


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
        """Cria a View Qt ligada a um ViewModel novo + experimentos/histórico (6c).

        Args:
            ctx: contexto da aplicação.

        Returns:
            A :class:`SimulatorView` (``QWidget``) raiz da perspectiva.
        """
        # Imports locais (Views importam Qt) — mantém este módulo leve.
        from apps.sim_manager.perspectives.simulation.experiments_service import (
            ExperimentsService,
        )
        from apps.sim_manager.perspectives.simulation.experiments_view import (
            ExperimentsPanel,
        )
        from apps.sim_manager.perspectives.simulation.experiments_viewmodel import (
            ExperimentsViewModel,
        )
        from apps.sim_manager.perspectives.simulation.view import SimulatorView

        # PR-3 (#7a): adota o projeto do startup (pasta de saída default), se houver.
        project = ctx.extras.get("project")
        project_dir = ctx.extras.get("project_dir") or ""
        sim_vm = self.build_viewmodel(ctx)
        if project_dir:
            sim_vm.output_dir = project_dir  # pré-preenche o campo "Saída"
        # Injeta o JaxWarmupService COMPARTILHADO (criado no boot, em ctx.extras) p/ o
        # warmup config-aware ao selecionar jax/auto. None no CI/sem-jax → gatilho no-op.
        view = SimulatorView(
            sim_vm, jax_warmup_service=ctx.extras.get("jax_warmup_service")
        )
        self._sim_vm = sim_vm  # ref viva

        # PR-2 (#1): publica o ResultsViewModel p/ a perspectiva "Resultados" (1 VM,
        # 2 Views). A Simulação tem order=0 → builda no boot, antes de Resultados ser
        # ativada (order=1), então o extra existe quando a galeria é construída.
        ctx.extras["results_vm"] = sim_vm.results

        sidebar = ctx.extras.get("secondary_sidebar")
        if sidebar is None:
            return view
        sim_vm.log_entry.connect(sidebar.append_log)

        # ── Fatia 6c — experimentos & histórico (na secondary sidebar) ────────
        # Reusa o ExperimentsService do startup (PR-3 #7a) se publicado — evita 2
        # services (recentes/cache divergentes); senão cria um (boot sem diálogo).
        self._exp_service = (
            ctx.extras.get("experiments_service") or ExperimentsService()
        )
        self._exp_vm = ExperimentsViewModel(self._exp_service)
        self._exp_panel = ExperimentsPanel()
        self._view = view
        sidebar.set_history_panel(self._exp_panel)

        # VM → painel
        self._exp_vm.experiment_changed.connect(self._on_experiment_changed)
        self._exp_vm.snapshot_added.connect(
            lambda sid, lbl, ic: self._exp_panel.add_snapshot(sid, lbl, in_cache=ic)
        )
        self._exp_vm.cache_status_changed.connect(
            lambda sid, ic: None if ic else self._exp_panel.mark_out_of_cache(sid)
        )
        self._exp_vm.recents_changed.connect(self._exp_panel.set_recents)
        self._exp_vm.error.connect(sidebar.append_log)

        # painel → ações
        self._exp_panel.request_new.connect(self._on_new_experiment)
        self._exp_panel.request_open.connect(self._on_open_experiment)
        self._exp_panel.request_save.connect(self._on_save_experiment)
        self._exp_panel.request_close.connect(self._exp_vm.close_experiment)
        self._exp_panel.request_clear.connect(self._on_clear_history)
        self._exp_panel.snapshot_selected.connect(self._on_snapshot_selected)
        self._exp_panel.snapshot_activated.connect(self._on_snapshot_reload)
        self._exp_panel.recent_activated.connect(self._exp_vm.open_experiment)

        # PR-3 (#7b): publica os comandos p/ a barra de menu (Arquivo/Sessão) — os
        # MESMOS handlers do painel de experimentos e da View (sem duplicar lógica).
        ctx.extras["file_actions"] = {
            "new": self._on_new_experiment,
            "open": self._on_open_experiment,
            "save": self._on_save_experiment,
            "save_as": self._on_save_as_experiment,
            "close": self._exp_vm.close_experiment,
        }
        ctx.extras["session_actions"] = {
            "save": view._on_save_session,
            "open": view._on_open_session,
        }

        # resultado de simulação → snapshot + cache (reabrível por double-click)
        sim_vm.result_ready.connect(self._on_sim_result)

        # ── Status bar (Lote 1) — Plot backend + Cache (se o shell os expõe) ──
        # Setters sempre existem como atributos (None se o shell não os expõe) —
        # contrato explícito; os handlers guardam contra teardown do widget Qt.
        self._sb_set_plot = None
        self._sb_set_cache = None
        self._sb_set_state = None
        self._sb_set_elapsed = None
        self._sb_set_throughput = None
        self._sb_set_exp = None
        status_bar = ctx.extras.get("status_bar")
        if isinstance(status_bar, dict):
            self._sb_set_plot = status_bar.get("set_plot")
            self._sb_set_cache = status_bar.get("set_cache")
            self._sb_set_state = status_bar.get("set_state")
            self._sb_set_elapsed = status_bar.get("set_elapsed")
            self._sb_set_throughput = status_bar.get("set_throughput")
            self._sb_set_exp = status_bar.get("set_exp")
            if self._sb_set_plot is not None:
                self._sb_set_plot(sim_vm.results.plot_backend.value)
                sim_vm.results.changed.connect(self._on_results_changed_status)
            if self._sb_set_cache is not None:
                self._exp_vm.cache_status_changed.connect(self._on_cache_status_changed)
            # Estado · Elapsed · Throughput (Lote 2) — da SimulationViewModel.
            if (
                self._sb_set_state is not None
                or self._sb_set_elapsed is not None
                or self._sb_set_throughput is not None
            ):
                sim_vm.changed.connect(self._on_sim_status_changed)
                self._on_sim_status_changed("_status", None)  # estado inicial

        # PR-3 (#7a): adota o PROJETO do startup, se houver; senão default em RAM.
        if project is not None and getattr(project, "name", ""):
            if project.file_path and os.path.exists(project.file_path):
                self._exp_vm.open_experiment(project.file_path)  # projeto existente
            else:
                self._exp_vm.new_experiment(
                    project.name, project.description, project.output_dir
                )
        else:
            self._exp_vm.new_experiment(
                "Sessão", "", "sm_experiments"
            )  # default em RAM
        self._exp_vm.load_recents()
        return view

    def _on_results_changed_status(self, name: str, _value: object) -> None:
        """Atualiza o campo Plot da status bar quando o backend de plot muda."""
        if name == "_plot_backend" and self._sb_set_plot is not None:
            try:
                self._sb_set_plot(self._sim_vm.results.plot_backend.value)
            except RuntimeError:
                pass  # widget Qt já destruído (ordem de teardown) — não-fatal

    def _on_cache_status_changed(self, *_: object) -> None:
        """Atualiza o campo Cache da status bar (guardado contra teardown do widget)."""
        if self._sb_set_cache is None:
            return
        try:
            self._sb_set_cache(self._exp_vm.cache_count)
        except RuntimeError:
            pass  # widget Qt já destruído (ordem de teardown) — não-fatal

    def _on_sim_status_changed(self, name: str, _value: object = None) -> None:
        """Atualiza Estado · Elapsed · Throughput na BottomBar (Lote 2).

        Reage a ``_status``/``_progress`` da SimulationViewModel, lendo o snapshot
        ATÔMICO ``status_display`` (evita race entre properties). Cada setter é
        guardado contra ``RuntimeError`` (teardown do widget Qt).
        """
        if name not in ("_status", "_progress"):
            return
        info = self._sim_vm.status_display
        for setter, key in (
            (self._sb_set_state, "state"),
            (self._sb_set_elapsed, "elapsed"),
            (self._sb_set_throughput, "throughput"),
        ):
            if setter is not None:
                try:
                    setter(info[key])
                except (RuntimeError, KeyError):
                    pass  # widget destruído OU status_display sem a chave — não-fatal

    # ── Handlers de experimentos & histórico (Fatia 6c) ─────────────────────
    def _on_experiment_changed(self, exp: object) -> None:
        name = getattr(exp, "name", None)
        path = getattr(exp, "file_path", "") if exp else ""
        self._exp_panel.set_experiment_label(name, path)
        # BottomBar (Lote 2): nome do experimento ativo.
        if self._sb_set_exp is not None:
            try:
                self._sb_set_exp(name or "—")
            except RuntimeError:
                pass  # widget Qt já destruído (ordem de teardown) — não-fatal
        self._exp_panel.clear_history()
        for snap in getattr(exp, "snapshots", []):
            self._exp_panel.add_snapshot(
                snap.snapshot_id,
                snap.label,
                in_cache=self._exp_vm.is_in_cache(snap.snapshot_id),
                info=_snapshot_info(snap),
            )

    def _on_new_experiment(self) -> None:
        from apps.sim_manager.perspectives.simulation.experiments_view import (
            NewExperimentDialog,
        )

        dialog = NewExperimentDialog(parent=self._view)
        if dialog.exec():
            name, desc, out_dir = dialog.values()
            self._exp_vm.new_experiment(name, desc, out_dir)

    def _on_open_experiment(self) -> None:
        from geosteering_ai.gui.qt_compat import QtWidgets

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._view, "Abrir experimento", "", "Experimento (*.exp.json)"
        )
        if path:
            self._exp_vm.open_experiment(path)

    def _on_save_experiment(self) -> None:
        self._exp_vm.save_experiment(params=self._sim_vm.to_session_dict())

    def _on_save_as_experiment(self) -> None:
        """Salvar Como… (PR-3 #7b) — escolhe novo .exp.json e reusa save_experiment."""
        from geosteering_ai.gui.qt_compat import QtWidgets

        exp = self._exp_vm.experiment
        if exp is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._view,
            "Salvar experimento como",
            exp.file_path or "",
            "Experimento (*.exp.json)",
        )
        if not path:
            return
        if not path.endswith(".exp.json"):
            path += ".exp.json"
        exp.file_path = path  # redireciona o destino; save_experiment grava nele
        self._exp_vm.save_experiment(params=self._sim_vm.to_session_dict())

    def _on_clear_history(self) -> None:
        from geosteering_ai.gui.qt_compat import QtWidgets

        resp = QtWidgets.QMessageBox.question(
            self._view,
            "Limpar histórico",
            "Apagar todos os snapshots desta sessão? Não pode ser desfeito.",
        )
        if resp == QtWidgets.QMessageBox.StandardButton.Yes:
            self._exp_vm.clear_snapshots()
            self._exp_service.cache_clear()
            self._exp_panel.clear_history()

    def _on_snapshot_selected(self, snap_id: str) -> None:
        snap = self._exp_vm.select_snapshot(snap_id)
        if snap is not None:
            self._exp_panel.set_snapshot_info(_snapshot_info(snap))

    def _on_snapshot_reload(self, snap_id: str) -> None:
        bundle = self._exp_service.cache_get(snap_id)
        if bundle is not None:
            self._sim_vm.results.set_result(bundle)  # reabre na galeria
        else:
            msg = (
                "tensor grande demais para o cache"
                if self._exp_service.cache_was_too_big(snap_id)
                else "fora do cache"
            )
            self._sim_vm.log_entry.emit(f"↺ Reload indisponível ({msg}) — re-execute.")

    def _on_sim_result(self, result: dict) -> None:
        h6 = result.get("H6")
        n_models = int(result.get("n_models") or (h6.shape[0] if h6 is not None else 0))
        count = len(self._exp_vm.snapshots) + 1
        backend = str(result.get("backend", "?"))
        elapsed = float(getattr(self._sim_vm, "elapsed_s", 0.0))
        label = f"#{count} · {backend} · {n_models} mod · {elapsed:.1f}s"
        snap = self._exp_service.make_snapshot(
            label=label,
            backend=backend,
            n_models=n_models,
            elapsed_s=elapsed,
            params=self._sim_vm.to_session_dict(),
        )
        # cache ANTES de add (evicções marcam itens antigos; o novo item já entra
        # com o estado de cache final). Bundle reabrível = o próprio result.
        self._exp_service.cache_put(snap.snapshot_id, dict(result))
        self._exp_vm.add_snapshot(
            snap, in_cache=self._exp_service.cache_contains(snap.snapshot_id)
        )
