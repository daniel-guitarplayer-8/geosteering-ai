# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/services/__init__.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Pacote      : gui.services — camada de orquestração L2 (MVVM)            ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — services (spec 0011a)                                ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FRONTEIRA DE IMPORT (lazy)                                               ║
# ║    Re-exporta ``BaseService``/``SimRequest``/``SimulationService`` por      ║
# ║    ``__getattr__`` (PEP 562) — assim importar o PACOTE (ou o submódulo PURO║
# ║    ``sim_request``) NÃO puxa Qt. Só ao acessar ``BaseService``/             ║
# ║    ``SimulationService`` é que o módulo Qt é importado. Mantém o ViewModel  ║
# ║    (que importa ``gui.services.sim_request``) PURO (Princípio X).          ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    BaseService (Qt) · SimulationService (Qt) · SimRequest (puro)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Camada de Service (MVVM L2) — re-export LAZY (Qt-free no import do pacote) (0011a)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["BaseService", "SimRequest", "SimulationService"]

if TYPE_CHECKING:  # pragma: no cover — só p/ type-checkers (não roda em runtime)
    from geosteering_ai.gui.services.base import BaseService as BaseService
    from geosteering_ai.gui.services.sim_request import SimRequest as SimRequest
    from geosteering_ai.gui.services.simulation_service import (
        SimulationService as SimulationService,
    )


def __getattr__(name: str) -> Any:
    """Importa sob demanda (PEP 562) — só puxa Qt quando o Service é acessado."""
    if name == "SimRequest":
        from geosteering_ai.gui.services.sim_request import SimRequest

        return SimRequest
    if name == "BaseService":
        from geosteering_ai.gui.services.base import BaseService

        return BaseService
    if name == "SimulationService":
        from geosteering_ai.gui.services.simulation_service import SimulationService

        return SimulationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
