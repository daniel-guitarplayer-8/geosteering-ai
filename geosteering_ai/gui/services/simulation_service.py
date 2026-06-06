# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/services/simulation_service.py                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimulationService — roda simulate_batch off-thread          ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — services (spec 0011a)                                ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação (walking skeleton)                     ║
# ║  Dependências: gui.services.base (Qt), gui.services.sim_request (puro)     ║
# ║  Padrão      : Service (MVVM L2) — orquestra; NÃO toca a física            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Dispara ``_run_simulation`` (que chama ``simulate_batch``) numa thread  ║
# ║    separada e emite o resultado via ``VMSignal``. A parte PURA (SimRequest ║
# ║    + batch + chamada) vive em ``sim_request.py`` (sem Qt) para o ViewModel ║
# ║    importar ``SimRequest`` sem puxar Qt; aqui fica só a casca Qt-touching.  ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimulationService · SimRequest (re-export por conveniência)            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimulationService`` — roda ``simulate_batch`` off-thread, emite via VMSignal (0011a)."""

from __future__ import annotations

from geosteering_ai.gui.services.base import BaseService
from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

__all__ = ["SimRequest", "SimulationService"]


class SimulationService(BaseService):
    """Service que roda ``simulate_batch`` off-thread e emite o resultado via VMSignal.

    Example:
        >>> svc = SimulationService()
        >>> svc.finished.connect(vm._on_sim_finished)   # VMSignal (puro)
        >>> svc.run(SimRequest(frequencies_hz=(20000.0,), n_models=2))

    Note:
        ``run`` retorna IMEDIATAMENTE (não-bloqueante); o resultado chega depois
        via ``self.finished`` na MAIN thread. Erros (ex.: parâmetro inválido na
        física) chegam via ``self.error``. A física é INTOCADA — só orquestração.
    """

    def run(self, request: SimRequest) -> None:
        """Dispara a simulação off-thread (não-bloqueante).

        Args:
            request: a requisição (já validada pelo ViewModel).
        """
        self._run_async(_run_simulation, request)
