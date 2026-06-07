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

import threading
from typing import Any

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

    Roteamento de backend (spec 0012):
      ┌──────────────┬──────────────────────────────────────────────────────────┐
      │ "numba"      │ in-thread (QThread) — rápido, sem spawn (caminho da Fatia 2)│
      │ "jax"/"auto" │ SUBPROCESSO spawn — isola JAX/CUDA (TLS-safe; QThread       │
      │              │ crasharia ao init CUDA). "auto" pode resolver p/ jax.       │
      └──────────────┴──────────────────────────────────────────────────────────┘
    """

    # Backends que (podem) inicializar JAX → exigem subprocesso (TLS-safe).
    _SUBPROCESS_BACKENDS = ("jax", "auto")

    def __init__(self, *, parent: Any = None) -> None:
        super().__init__(parent=parent)
        # Eventos cooperativos (Fatia 6a) — só atuam no caminho numba in-thread
        # (memória compartilhada com a QThread). cancel: set=cancelar. pause:
        # set=rodando / clear=pausado. No subprocesso (jax) NÃO são usados (v1).
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # default: rodando (não pausado)

    def run(self, request: SimRequest) -> None:
        """Dispara a simulação off-thread (não-bloqueante); roteia por backend.

        Args:
            request: a requisição (já validada pelo ViewModel).
        """
        # Guard de reentrância NO SERVICE (defesa, além do guard do ViewModel):
        # resetar os eventos com um worker ainda em voo seria uma corrida (o worker
        # lê cancel/pause enquanto o run() os reseta). 1 simulação por vez.
        if self.is_busy():
            return
        # Reseta o controle cooperativo para esta run (limpa cancel; despausa).
        self._cancel_event.clear()
        self._pause_event.set()
        if request.backend in self._SUBPROCESS_BACKENDS:
            # jax/auto podem init CUDA → subprocesso isolado (a QThread crasharia).
            # Progresso/cancel intra-run NÃO cruzam o processo (v1) — só finished/error.
            self._run_in_subprocess(_run_simulation, request)
        else:
            # numba in-thread (rápido, sem spawn): progresso por-grupo + cancel/pause
            # cooperativos via os eventos (memória compartilhada com a QThread).
            self._run_async(
                _run_simulation,
                request,
                report_progress=True,
                cancel_event=self._cancel_event,
                pause_event=self._pause_event,
            )

    def request_cancel(self) -> None:
        """Solicita cancelamento cooperativo (numba: aborta entre grupos)."""
        self._cancel_event.set()
        self._pause_event.set()  # despausa p/ o loop de espera ver o cancel e sair

    def request_pause(self) -> None:
        """Pausa cooperativa (numba: bloqueia entre grupos). No-op p/ jax (v1)."""
        self._pause_event.clear()

    def request_resume(self) -> None:
        """Retoma de uma pausa cooperativa."""
        self._pause_event.set()

    def wait(self, timeout_ms: int = 30000) -> None:
        """Teardown seguro: cancela + despausa ANTES de bloquear (evita deadlock).

        Um worker PAUSADO fica num sleep-loop cooperativo (``_await_resume_or_cancel``)
        até ``pause_event`` ser setado. Se ``wait()`` bloqueasse a main thread sem
        antes despausar, o worker nunca sairia do loop → deadlock até o timeout. Aqui
        setamos cancel (sai do loop) + resume (destrava) e então delegamos ao base.
        """
        self._cancel_event.set()
        self._pause_event.set()
        super().wait(timeout_ms)
