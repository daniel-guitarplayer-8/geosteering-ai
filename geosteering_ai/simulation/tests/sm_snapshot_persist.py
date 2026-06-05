# -*- coding: utf-8 -*-
"""Shim de retrocompat — re-exporta ``gui.persistence.snapshot`` (spec 0007).

TRANSITÓRIO: a implementação canônica vive em
:mod:`geosteering_ai.gui.persistence.snapshot` (relocada na spec 0007 via git mv;
escrita ENDURECIDA p/ atômica). Mantido para não quebrar ``from .sm_snapshot_persist
import SnapshotPersistThread`` (monólito). Removido quando o SM migrar (0011).
"""

from __future__ import annotations

from geosteering_ai.gui.persistence.snapshot import SnapshotPersistThread

__all__ = ["SnapshotPersistThread"]
