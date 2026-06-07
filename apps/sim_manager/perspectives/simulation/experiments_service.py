# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/experiments_service.py         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ExperimentsService — persistência + cache LRU (Qt)        ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — experimentos (spec 0016, Fatia 6c)        ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — experimentos & histórico                        ║
# ║  Framework   : Qt6 via gui.qt_compat (QObject + QSettings + QThread save)  ║
# ║  Padrão      : Service (MVVM L2) — emite VMSignal (pub/sub PURO)           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Persiste ``ExperimentState`` em ``.exp.json`` (escrita ATÔMICA async    ║
# ║    via SnapshotPersistThread), mantém um ``LRUPlotCache`` de bundles de     ║
# ║    resultado (p/ reload do histórico) e a lista de recentes (QSettings).    ║
# ║    Emite ``saved``/``error``/``cache_updated`` (VMSignal) — o ViewModel     ║
# ║    PURO conecta-se sem importar Qt.                                         ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    ExperimentsService                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ExperimentsService`` — persistência .exp.json + cache LRU + recentes (0016)."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from apps.sim_manager.perspectives.simulation.experiment_state import (
    ExperimentState,
    SimulationSnapshot,
)
from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache, default_max_bytes
from geosteering_ai.gui.persistence.snapshot import SnapshotPersistThread
from geosteering_ai.gui.qt_compat import QObject, QtCore
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["ExperimentsService"]

_RECENTS_KEY = "experiment/recents"
_RECENTS_MAX = 10


def _now_iso() -> str:
    """ISO-8601 do instante atual (segundos)."""
    return datetime.now().isoformat(timespec="seconds")


class ExperimentsService(QObject):  # type: ignore[misc] # QObject é Any (qt_compat)
    """Service de experimentos: persistência atômica async + cache LRU + recentes.

    Attributes:
        saved: ``VMSignal`` — emitido (str path) ao concluir um save assíncrono.
        error: ``VMSignal`` — emitido (str msg) em falha de I/O.
        cache_updated: ``VMSignal`` — emitido (snap_id, in_cache: bool) ao mudar o cache.

    Note:
        ``saved``/``error``/``cache_updated`` são ``VMSignal`` (pub/sub PURO). É um
        QObject só para a ponte do ``SnapshotPersistThread`` (QThread→main).
    """

    def __init__(self, *, parent: Any = None) -> None:
        super().__init__(parent)
        self.saved: VMSignal = VMSignal()
        self.error: VMSignal = VMSignal()
        self.cache_updated: VMSignal = VMSignal()
        self._cache = LRUPlotCache(maxlen=3, max_bytes=default_max_bytes())
        self._save_thread: Optional[Any] = None  # mantém ref viva da QThread

    # ── Experimento (criar/carregar/salvar) ───────────────────────────────
    def create_experiment(
        self, name: str, description: str, output_dir: str
    ) -> ExperimentState:
        """Cria um ``ExperimentState`` novo + define ``file_path`` (slug.exp.json)."""
        slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in name).strip("_")
        slug = slug or "experimento"
        path = os.path.join(output_dir or ".", f"{slug}.exp.json")
        now = _now_iso()
        return ExperimentState(
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            output_dir=output_dir,
            file_path=path,
        )

    def load_experiment(self, path: str) -> ExperimentState:
        """Carrega um ``ExperimentState`` de ``.exp.json`` (síncrono)."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        exp = ExperimentState.from_dict(data)
        exp.file_path = path
        return exp

    def save_experiment_async(
        self, exp: ExperimentState, path: Optional[str] = None
    ) -> None:
        """Salva o experimento em ``.exp.json`` de forma ASSÍNCRONA (QThread atômica)."""
        target = path or exp.file_path
        if not target:
            self.error.emit("Caminho do experimento não definido.")
            return
        exp.file_path = target
        exp.updated_at = _now_iso()
        json_text = json.dumps(exp.to_dict(), indent=2, ensure_ascii=False)
        thread = SnapshotPersistThread(json_text, target)
        thread.finished_ok.connect(self._on_saved)
        thread.error.connect(self._on_save_error)
        # finaliza+solta a thread ao terminar (sem deleteLater — segue worker.py).
        thread.finished_ok.connect(thread.quit)
        thread.error.connect(thread.quit)
        self._save_thread = thread  # ref viva (senão o GC aborta a QThread)
        thread.start()

    def make_snapshot(
        self,
        *,
        label: str,
        backend: str,
        n_models: int,
        elapsed_s: float,
        params: Dict[str, Any],
    ) -> SimulationSnapshot:
        """Cria um ``SimulationSnapshot`` (gera UUID4 + timestamp)."""
        return SimulationSnapshot(
            snapshot_id=uuid.uuid4().hex,
            timestamp=_now_iso(),
            label=label,
            backend=backend,
            n_models=int(n_models),
            elapsed_s=float(elapsed_s),
            params=dict(params),
        )

    # ── Cache LRU de bundles (p/ reload do histórico) ──────────────────────
    def cache_put(self, snap_id: str, bundle: Dict[str, Any]) -> List[str]:
        """Insere um bundle de resultado no cache LRU. Emite ``cache_updated`` por mudança."""
        evicted = self._cache.put(snap_id, bundle)
        for old_id in evicted:
            if old_id != snap_id:
                self.cache_updated.emit(old_id, False)
        # Estado FINAL (o próprio snap_id pode ter sido auto-evictado por tamanho).
        self.cache_updated.emit(snap_id, snap_id in self._cache)
        return evicted

    def cache_get(self, snap_id: str) -> Optional[Dict[str, Any]]:
        """Recupera um bundle do cache (MRU) ou ``None``."""
        return self._cache.get(snap_id)

    def cache_contains(self, snap_id: str) -> bool:
        """``True`` se o snapshot está no cache."""
        return snap_id in self._cache

    def cache_was_too_big(self, snap_id: str) -> bool:
        """``True`` se o snapshot foi auto-evictado por exceder ``max_bytes``."""
        return self._cache.was_too_big(snap_id)

    def cache_clear(self) -> None:
        """Limpa o cache de bundles."""
        self._cache.clear()

    # ── Recentes (QSettings) ───────────────────────────────────────────────
    def load_recents(self) -> List[str]:
        """Lê a lista de experimentos recentes (QSettings; forward-compat)."""
        raw = self._settings().value(_RECENTS_KEY, [])
        if isinstance(raw, str):
            return [raw] if raw else []
        try:
            return [str(x) for x in raw if x]
        except TypeError:
            return []

    def push_recent(self, path: str) -> List[str]:
        """Move/insere ``path`` no topo dos recentes (dedup; top-10). Retorna a lista."""
        recents = [p for p in self.load_recents() if p != path]
        recents.insert(0, path)
        recents = recents[:_RECENTS_MAX]
        self._settings().setValue(_RECENTS_KEY, recents)
        return recents

    # ── Internos ───────────────────────────────────────────────────────────
    @staticmethod
    def _settings() -> Any:
        """QSettings com escopo explícito (paridade c/ o monólito)."""
        return QtCore.QSettings("Geosteering AI", "Simulation Manager")

    def _on_saved(self, path: str) -> None:
        self.saved.emit(path)

    def _on_save_error(self, message: str) -> None:
        self.error.emit(message)
