# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/experiments_viewmodel.py       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ExperimentsViewModel — experimentos & histórico (PURO)    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — experimentos (spec 0016, Fatia 6c)        ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — experimentos & histórico                        ║
# ║  Framework   : Python PURO — NÃO importa Qt (Princípio X; testável)        ║
# ║  Padrão      : ViewModel (MVVM) — Service INJETADO (duck-typed)           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Estado de UI dos experimentos: experimento atual + histórico de         ║
# ║    snapshots + chaves em cache + recentes. Delega persistência/cache ao    ║
# ║    Service injetado. PURO (testável sem Qt). Emite VMSignals que a View    ║
# ║    (painel na secondary sidebar) consome.                                  ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ExperimentsViewModel                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ExperimentsViewModel`` — experimentos & histórico (PURO, service injetado) (0016)."""

from __future__ import annotations

from typing import Any, List, Optional, Set

from apps.sim_manager.perspectives.simulation.experiment_state import (
    ExperimentState,
    SimulationSnapshot,
)
from geosteering_ai.gui.viewmodels.base import BaseViewModel
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["ExperimentsViewModel"]


class ExperimentsViewModel(BaseViewModel):
    """ViewModel PURO de experimentos & histórico (``service`` injetado).

    Signals (VMSignal):
        experiment_changed(ExperimentState | None): experimento atual mudou (abrir/novo/fechar).
        snapshot_added(snap_id, label, in_cache): novo snapshot no histórico.
        cache_status_changed(snap_id, in_cache): bundle entrou/saiu do cache.
        recents_changed(list[str]): lista de recentes atualizada.
        error(str): falha (persistência etc.).

    Note:
        PURO — o ``service`` é duck-typed (precisa de create/load/save_async/cache_*/
        recents + VMSignals ``saved``/``error``/``cache_updated``). Testável com stub.
    """

    def __init__(self, service: Any) -> None:
        super().__init__()
        self._service = service
        self._experiment: Optional[ExperimentState] = None
        self._cache_keys: Set[str] = set()
        self._recents: List[str] = []

        self.experiment_changed: VMSignal = VMSignal()
        self.snapshot_added: VMSignal = VMSignal()
        self.cache_status_changed: VMSignal = VMSignal()
        self.recents_changed: VMSignal = VMSignal()
        self.error: VMSignal = VMSignal()

        if hasattr(service, "cache_updated"):
            service.cache_updated.connect(self._on_cache_updated)
        if hasattr(service, "error"):
            service.error.connect(self.error.emit)

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def experiment(self) -> Optional[ExperimentState]:
        """Experimento atual (ou ``None``)."""
        return self._experiment

    @property
    def snapshots(self) -> List[SimulationSnapshot]:
        """Snapshots do experimento atual (cópia)."""
        return list(self._experiment.snapshots) if self._experiment else []

    @property
    def recents(self) -> List[str]:
        """Caminhos recentes (cópia)."""
        return list(self._recents)

    def is_in_cache(self, snap_id: str) -> bool:
        """``True`` se o bundle do snapshot está em cache (ícone ●)."""
        return snap_id in self._cache_keys

    @property
    def cache_count(self) -> int:
        """Nº de bundles atualmente em cache (campo Cache da status bar)."""
        return len(self._cache_keys)

    # ── Ações ──────────────────────────────────────────────────────────────
    def new_experiment(self, name: str, description: str, output_dir: str) -> bool:
        """Cria um experimento novo (vira o atual). ``False`` se nome vazio."""
        if not name or not name.strip():
            self.error.emit("Nome do experimento vazio.")
            return False
        self._experiment = self._service.create_experiment(
            name, description, output_dir
        )
        self._cache_keys.clear()
        self.experiment_changed.emit(self._experiment)
        self.changed.emit("_experiment", self._experiment)
        return True

    def open_experiment(self, path: str) -> bool:
        """Abre um ``.exp.json`` (vira o atual + repovoa histórico). ``False`` em erro."""
        try:
            exp = self._service.load_experiment(path)
        except (OSError, ValueError) as exc:
            self.error.emit(f"Falha ao abrir: {exc}")
            return False
        self._experiment = exp
        self._cache_keys.clear()  # cache é efêmero (RAM) — vazio ao reabrir
        self._recents = self._service.push_recent(path)
        self.experiment_changed.emit(exp)
        self.recents_changed.emit(self._recents)
        self.changed.emit("_experiment", exp)
        return True

    def save_experiment(self, params: Optional[dict] = None) -> None:
        """Salva o experimento atual (async). Atualiza ``params`` se fornecido."""
        if self._experiment is None:
            self.error.emit("Sem experimento para salvar.")
            return
        if params is not None:
            self._experiment.params = dict(params)
        self._service.save_experiment_async(self._experiment)
        if self._experiment.file_path:
            self._recents = self._service.push_recent(self._experiment.file_path)
            self.recents_changed.emit(self._recents)

    def close_experiment(self) -> None:
        """Fecha o experimento atual (volta ao estado sem experimento)."""
        self._experiment = None
        self._cache_keys.clear()
        self.experiment_changed.emit(None)
        self.changed.emit("_experiment", None)

    def add_snapshot(self, snap: SimulationSnapshot, *, in_cache: bool) -> None:
        """Adiciona um snapshot ao histórico do experimento atual."""
        if self._experiment is None:
            return
        self._experiment.append_snapshot(snap)
        if in_cache:
            self._cache_keys.add(snap.snapshot_id)
        self.snapshot_added.emit(snap.snapshot_id, snap.label, in_cache)
        self.changed.emit("_snapshots", snap.snapshot_id)

    def clear_snapshots(self) -> None:
        """Limpa o histórico do experimento atual."""
        if self._experiment is None:
            return
        self._experiment.clear_snapshots()
        self._cache_keys.clear()
        self.changed.emit("_snapshots", None)

    def mark_out_of_cache(self, snap_id: str) -> None:
        """Marca um snapshot como fora-de-cache (○)."""
        self._cache_keys.discard(snap_id)
        self.cache_status_changed.emit(snap_id, False)

    def select_snapshot(self, snap_id: str) -> Optional[SimulationSnapshot]:
        """Retorna o snapshot por id (ou ``None``)."""
        if self._experiment is None:
            return None
        for snap in self._experiment.snapshots:
            if snap.snapshot_id == snap_id:
                return snap
        return None

    def load_recents(self) -> List[str]:
        """Carrega os recentes do Service (QSettings)."""
        self._recents = self._service.load_recents()
        self.recents_changed.emit(self._recents)
        return self._recents

    # ── Callbacks do service ─────────────────────────────────────────────────
    def _on_cache_updated(self, snap_id: str, in_cache: bool) -> None:
        if in_cache:
            self._cache_keys.add(snap_id)
        else:
            self._cache_keys.discard(snap_id)
        self.cache_status_changed.emit(snap_id, in_cache)
