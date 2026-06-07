# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/experiment_state.py            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : ExperimentState + SimulationSnapshot (PURO)               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — experimentos (spec 0016, Fatia 6c)        ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — experimentos & histórico                        ║
# ║  Framework   : stdlib PURO — NÃO importa Qt (Princípio X; testável)        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Modelo PURO de experimento (.exp.json) + snapshot de simulação. NÃO     ║
# ║    importa do monólito (simulation_manager.py) — definição própria do SM   ║
# ║    MVVM, evitando acoplamento ao código que está sendo estrangulado.       ║
# ║    Serializável (to_dict/from_dict) com forward-compat (ignora chaves      ║
# ║    desconhecidas). Timestamps/IDs são INJETADOS (o Service gera) p/ manter ║
# ║    o modelo testável sem efeitos colaterais.                               ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimulationSnapshot · ExperimentState                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``ExperimentState`` + ``SimulationSnapshot`` — modelo PURO de experimento (0016)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = ["SimulationSnapshot", "ExperimentState"]


@dataclass
class SimulationSnapshot:
    """Metadados de UMA simulação no histórico de um experimento (PURO/serializável).

    Attributes:
        snapshot_id: identificador estável (UUID4 — gerado pelo Service).
        timestamp: ISO-8601 da execução (gerado pelo Service).
        label: rótulo legível (ex.: ``"#3 · numba · 1000 mod · 12.4 s"``).
        backend: backend efetivo (``numba``/``jax``/...).
        n_models: nº de modelos do ensemble.
        elapsed_s: tempo de execução (s).
        params: dict de params da simulação (``to_session_dict`` do VM) — reprodutível.
    """

    snapshot_id: str
    timestamp: str
    label: str
    backend: str = "numba"
    n_models: int = 0
    elapsed_s: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dict JSON."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "label": self.label,
            "backend": self.backend,
            "n_models": int(self.n_models),
            "elapsed_s": float(self.elapsed_s),
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationSnapshot":
        """Reconstrói de um dict (forward-compat: só lê as chaves conhecidas)."""
        return cls(
            snapshot_id=str(data.get("snapshot_id", "")),
            timestamp=str(data.get("timestamp", "")),
            label=str(data.get("label", "")),
            backend=str(data.get("backend", "numba")),
            n_models=int(data.get("n_models", 0)),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            params=dict(data.get("params", {})),
        )


@dataclass
class ExperimentState:
    """Estado de um experimento (.exp.json): metadados + params + histórico (PURO).

    Attributes:
        name: nome legível do experimento.
        description: descrição livre.
        created_at/updated_at: ISO-8601 (gerados/atualizados pelo Service).
        output_dir: diretório de saída (.dat/.out futuros).
        file_path: caminho do ``.exp.json`` (preenchido ao salvar).
        params: params atuais do experimento (``to_session_dict`` do VM).
        snapshots: histórico de :class:`SimulationSnapshot`.
    """

    name: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    output_dir: str = ""
    file_path: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    snapshots: List[SimulationSnapshot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dict JSON (snapshots aninhados)."""
        return {
            "schema_version": 1,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "output_dir": self.output_dir,
            "file_path": self.file_path,
            "params": dict(self.params),
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentState":
        """Reconstrói de um dict (forward-compat: ignora chaves desconhecidas)."""
        snaps = [
            SimulationSnapshot.from_dict(s)
            for s in data.get("snapshots", [])
            if isinstance(s, dict)
        ]
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            output_dir=str(data.get("output_dir", "")),
            file_path=str(data.get("file_path", "")),
            params=dict(data.get("params", {})),
            snapshots=snaps,
        )

    def append_snapshot(self, snap: SimulationSnapshot) -> None:
        """Adiciona um snapshot ao histórico."""
        self.snapshots.append(snap)

    def remove_snapshot(self, snapshot_id: str) -> None:
        """Remove o snapshot com o ``snapshot_id`` dado (no-op se ausente)."""
        self.snapshots = [s for s in self.snapshots if s.snapshot_id != snapshot_id]

    def clear_snapshots(self) -> None:
        """Limpa todo o histórico de snapshots."""
        self.snapshots = []
