# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/threading/__init__.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Pacote      : gui.threading — execução off-thread (Worker)               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — threading (spec 0011a)                               ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Camada de threading da fundação ``gui/``: roda callables pesados fora  ║
# ║    da UI thread. IMPORTA Qt (é infra de GUI) — o ``core`` nunca a importa. ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    Worker · WorkerSignals · run_in_thread                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Threading da fundação ``gui/`` — ``Worker`` off-thread (spec 0011a)."""

from __future__ import annotations

from geosteering_ai.gui.threading.worker import Worker, WorkerSignals, run_in_thread

__all__ = ["Worker", "WorkerSignals", "run_in_thread"]
