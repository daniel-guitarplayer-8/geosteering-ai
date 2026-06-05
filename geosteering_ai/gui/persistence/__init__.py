# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/persistence/__init__.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Pacote      : gui.persistence — persistência de estado de UI            ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — persistência (spec 0007)                            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FRONTEIRA DE IMPORT                                                      ║
# ║    Re-exporta SÓ a API Qt-FREE (``atomic_write_text``, ``SessionDocument``, ║
# ║    ``LRUPlotCache``, ``default_max_bytes``) — para que                     ║
# ║    ``import geosteering_ai.gui.persistence`` NÃO importe Qt (Princípio X:  ║
# ║    SessionDocument testável sem ``pytest-qt``). O                         ║
# ║    ``SnapshotPersistThread`` (``QThread``) vive em                         ║
# ║    ``gui.persistence.snapshot`` (submódulo) — acesso explícito.           ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    atomic_write_text · SessionDocument · LRUPlotCache · default_max_bytes ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Persistência de estado de UI (``.session`` atômico, cache LRU) — fundação gui/.

Re-exporta a API **Qt-free** (atômico + SessionDocument + cache). A camada async
``SnapshotPersistThread`` (``QThread``) fica no submódulo ``snapshot`` para manter
o pacote importável sem Qt.
"""

from __future__ import annotations

from geosteering_ai.gui.persistence.atomic import atomic_write_text
from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache, default_max_bytes
from geosteering_ai.gui.persistence.session import SessionDocument

__all__ = [
    "LRUPlotCache",
    "SessionDocument",
    "atomic_write_text",
    "default_max_bytes",
]
