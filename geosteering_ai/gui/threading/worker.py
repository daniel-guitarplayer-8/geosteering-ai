# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/threading/worker.py                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Worker genérico (executa callable off-thread)              ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — threading (spec 0011a)                               ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : Qt6 via gui.qt_compat (QObject + Signal + QThread)          ║
# ║  Dependências: gui.qt_compat                                              ║
# ║  Padrão      : Worker(QObject) + moveToThread (preferido a subclasse QThread)║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Roda um ``callable`` PESADO (ex.: ``simulate_batch``) numa thread       ║
# ║    separada, sem congelar a UI. Emite ``finished(result)``/``error(msg)``  ║
# ║    via ``WorkerSignals`` (QObject SEPARADO) — o resultado é marshalado de  ║
# ║    volta à MAIN thread pela conexão Qt (QueuedConnection automática).      ║
# ║                                                                           ║
# ║  INVARIANTES DE THREAD (CRÍTICO)                                          ║
# ║    • ``WorkerSignals`` é um QObject SEPARADO do ``Worker`` (sobrevive ao   ║
# ║      ``moveToThread``; conexões feitas antes permanecem válidas).         ║
# ║    • O ``Worker`` SÓ emite sinais — NUNCA cria ``QWidget``/``QObject`` de  ║
# ║      UI (Qt fora da main thread = comportamento indefinido).              ║
# ║    • Quem RECEBE deve ser um slot de um QObject com afinidade à MAIN      ║
# ║      thread → a conexão AUTO vira QueuedConnection (slot roda na main).   ║
# ║      Conectar a um callable Python "solto" rodaria na worker thread.     ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    Worker · WorkerSignals · run_in_thread                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``Worker`` genérico — executa um callable off-thread e emite ``finished``/``error``.

A camada de threading mínima da fundação ``gui/`` (spec 0011a). Usada pela camada
de Service (``gui/services``) para rodar simulações sem congelar a UI, com o
resultado marshalado à main thread via ``WorkerSignals`` (Qt).
"""

from __future__ import annotations

from typing import Any, Callable

from geosteering_ai.gui.qt_compat import QObject, QThread, Signal

__all__ = ["Worker", "WorkerSignals", "run_in_thread"]


class WorkerSignals(QObject):  # type: ignore[misc] # QObject é Any (qt_compat) → mypy
    """Sinais Qt do :class:`Worker` (QObject SEPARADO — sobrevive ao ``moveToThread``).

    Attributes:
        finished: ``Signal(object)`` — resultado do callable (sucesso).
        error: ``Signal(str)`` — mensagem de erro (qualquer exceção do callable).
        progress: ``Signal(int, int)`` — (concluído, total) para barras de progresso.

    Note:
        ``Signal(object)`` numa conexão *queued* in-process passa o objeto POR
        REFERÊNCIA (PyQt6 NÃO faz pickle/deep-copy in-process). É thread-safe aqui
        porque o worker PRODUZ-E-SOLTA o resultado (nenhuma mutação concorrente
        após o ``emit``). Se um produtor RETIVESSE e mutasse o payload, seria
        necessário ``copy.deepcopy`` antes de emitir.
    """

    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, int)


class Worker(QObject):  # type: ignore[misc] # QObject é Any (qt_compat) → mypy
    """Executa ``fn(*args, **kwargs)`` numa thread separada (via :func:`run_in_thread`).

    Attributes:
        signals: o :class:`WorkerSignals` (conecte-se a ele ANTES de iniciar).

    Example:
        >>> w = Worker(simulate_batch, rho_h, rho_v, esp, z, backend="numba")
        >>> w.signals.finished.connect(self._on_done)   # slot de QObject na main thread
        >>> thread = run_in_thread(w)

    Note:
        ``run`` é o slot disparado por ``QThread.started`` — roda na worker thread.
        Qualquer exceção do callable é capturada e re-emitida como ``error(str)``
        (a thread nunca propaga exceção não-tratada).
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Captura o callable + argumentos (não inicia nada).

        Args:
            fn: o callable PESADO a rodar off-thread.
            *args: posicionais repassados a ``fn``.
            **kwargs: nomeados repassados a ``fn``.
        """
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:  # noqa: D401 — slot Qt
        """Roda o callable (na worker thread); emite ``finished`` ou ``error``.

        NÃO chame diretamente — conecte a ``QThread.started`` (ver
        :func:`run_in_thread`). Captura QUALQUER exceção (guard de topo da thread).
        """
        try:
            result = self._fn(*self._args, **self._kwargs)
        except BaseException as exc:  # noqa: BLE001 — guard de topo: nada escapa da thread
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(result)


def run_in_thread(worker: Worker) -> Any:
    """Move ``worker`` para uma ``QThread`` nova e a inicia.

    Args:
        worker: o :class:`Worker` (conecte seus ``signals`` ANTES de chamar).

    Returns:
        A ``QThread`` criada (o chamador DEVE manter uma referência viva ao
        ``thread`` E ao ``worker`` — senão o GC os coleta e a worker thread é
        abortada). ``BaseService`` guarda ambos em ``_threads``.

    Note:
        A thread encerra (``quit``) automaticamente ao ``finished``/``error``.
        NÃO usamos ``worker.deleteLater``: a worker thread já saiu do event loop
        ao ``finished``, então o ``deleteLater`` (agendado para a thread do
        worker) nunca seria processado — e o evento pendente corrompe testes
        subsequentes. O ``worker`` é coletado pelo GC quando o chamador o solta
        (após a thread já ter terminado — seguro).
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    # Encerra a thread ao concluir (sucesso OU erro).
    worker.signals.finished.connect(thread.quit)
    worker.signals.error.connect(thread.quit)
    thread.start()
    return thread
