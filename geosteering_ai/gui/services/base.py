# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/services/base.py                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : BaseService — orquestração L2 (backend off-thread → VMSignal)║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — services (spec 0011a)                                ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : Qt6 via gui.qt_compat + gui.threading (orquestra threads)   ║
# ║  Dependências: gui.qt_compat, gui.threading, gui.viewmodels.signal         ║
# ║  Padrão      : Service (MVVM L2) — recebe config, roda off-thread, emite   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Base da camada de Service: roda trabalho PESADO num ``Worker`` e        ║
# ║    re-emite o resultado via ``VMSignal`` (pub/sub PURO) — a interface que  ║
# ║    o ViewModel consome SEM importar Qt.                                    ║
# ║                                                                           ║
# ║  PONTE DE THREAD (por que BaseService é um QObject)                       ║
# ║    Os sinais Qt do Worker (worker thread) são conectados a SLOTS deste     ║
# ║    QObject (afinidade = MAIN thread) → a conexão AUTO vira QueuedConnection║
# ║    → ``_on_worker_finished`` roda na MAIN thread → ``self.finished.emit``  ║
# ║    (VMSignal) e, portanto, os callbacks da View rodam na MAIN thread.     ║
# ║    (Conectar o sinal do Worker a um callable Python "solto" rodaria na     ║
# ║    worker thread — bug de Qt fora da main thread.)                        ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    BaseService                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``BaseService`` — base MVVM L2: roda backend off-thread e emite via ``VMSignal``."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

from geosteering_ai.gui.qt_compat import QObject, Qt
from geosteering_ai.gui.threading import Worker, run_in_thread
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["BaseService"]


class BaseService(QObject):  # type: ignore[misc] # QObject é Any (qt_compat) → mypy
    """Base de Service: orquestra um ``Worker`` e re-emite via ``VMSignal`` na main thread.

    Attributes:
        finished: ``VMSignal`` — emitido (na MAIN thread) com o resultado do worker.
        error: ``VMSignal`` — emitido com a mensagem de erro (str).
        progress: ``VMSignal`` — emitido com ``(done, total)``.

    Note:
        ``finished``/``error``/``progress`` são ``VMSignal`` (pub/sub PURO) — o
        ViewModel conecta-se a eles SEM importar Qt (Princípio X). É um QObject só
        para que a ponte worker→main thread use QueuedConnection (ver header).
    """

    def __init__(self, *, parent: Any = None) -> None:
        super().__init__(parent)
        self.finished: VMSignal = VMSignal()
        self.error: VMSignal = VMSignal()
        self.progress: VMSignal = VMSignal()
        # Mantém (thread, worker) vivos — sem isto o GC coletaria e a worker
        # thread seria abortada no meio do trabalho.
        self._threads: List[Tuple[Any, Worker]] = []

    def _run_async(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Roda ``fn(*args, **kwargs)`` num ``Worker`` off-thread.

        Os sinais do Worker são conectados a slots DESTE QObject (main thread) →
        os ``VMSignal`` são emitidos na main thread (ver header).

        Args:
            fn: callable pesado (módulo-nível, ex.: ``_run_simulation``).
            *args/**kwargs: argumentos repassados a ``fn``.
        """
        worker = Worker(fn, *args, **kwargs)
        # QueuedConnection EXPLÍCITA: o worker emite na worker thread; o slot DEVE
        # rodar na main thread (afinidade deste QObject). AUTO já resolveria assim,
        # mas explícito blinda contra regressão se a afinidade mudar num refactor.
        queued = Qt.ConnectionType.QueuedConnection
        worker.signals.finished.connect(self._on_worker_finished, queued)
        worker.signals.error.connect(self._on_worker_error, queued)
        worker.signals.progress.connect(self._on_worker_progress, queued)
        thread = run_in_thread(worker)
        # Descarta pares cuja thread já terminou (evita acúmulo) + guarda o novo.
        self._threads = [(t, w) for (t, w) in self._threads if t.isRunning()]
        self._threads.append((thread, worker))

    def is_busy(self) -> bool:
        """``True`` se alguma worker thread ainda está rodando.

        Útil para a UI (ex.: bloquear "fechar" durante o trabalho) e para
        teardown limpo de teste (uma ``QThread`` destruída enquanto roda aborta).
        """
        return any(thread.isRunning() for (thread, _w) in self._threads)

    def wait(self, timeout_ms: int = 30000) -> None:
        """Bloqueia até as worker threads terminarem (teardown limpo).

        Args:
            timeout_ms: tempo máximo de espera POR thread, em ms.
        """
        for thread, _worker in self._threads:
            if thread.isRunning():
                thread.wait(timeout_ms)

    # ── Slots (rodam na MAIN thread via QueuedConnection) ────────────────────
    def _on_worker_finished(self, result: Any) -> None:
        self.finished.emit(result)

    def _on_worker_error(self, message: str) -> None:
        self.error.emit(message)

    def _on_worker_progress(self, done: int, total: int) -> None:
        self.progress.emit(done, total)
