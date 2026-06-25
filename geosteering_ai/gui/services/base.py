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

import atexit
import threading
from typing import Any, Callable, List, Optional, Tuple

from geosteering_ai.gui.qt_compat import QObject, Qt
from geosteering_ai.gui.threading import Worker, run_in_thread
from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["BaseService", "release_jax_pool"]

# Mensagem acionável (PT-BR) quando o subprocesso de simulação morre sem devolver
# exceção Python (OOM killer / GPU OOM / crash do driver CUDA / segfault). Usada
# tanto pelo pool efêmero (``_pool_run``) quanto pelo persistente (``_pool_run_persistent``).
_SUBPROCESS_DIED_MSG = (
    "O subprocesso de simulação encerrou abruptamente — provável falta de "
    "memória (GPU/CPU), crash do driver CUDA ou segfault. Tente um batch "
    "menor (nº de modelos) ou o backend 'numba'."
)

# Mensagem quando o worker persistente foi liberado por um ``release_jax_pool`` EXTERNO
# (encerramento: closeEvent/aboutToQuit/atexit) enquanto a sim estava em voo. NÃO é
# crash — é cancelamento deliberado; o ``_pool_run_persistent`` aborta limpo (sem
# re-rodar/recriar pool durante o shutdown) e mostra esta msg (em vez de erro vazio).
_SHUTDOWN_CANCEL_MSG = (
    "Simulação JAX cancelada — o worker GPU foi encerrado (app fechando ou "
    "pool liberado). Reabra/re-execute para rodar novamente."
)


def _pool_run(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """Roda ``fn(*args, **kwargs)`` num SUBPROCESSO (ProcessPool ``spawn``, 1 worker).

    Chamado pelo ``Worker`` (na QThread) — BLOQUEIA no ``future.result()``. O JAX/
    CUDA inicializa num subprocesso ISOLADO (``spawn`` = processo Python limpo, sem o
    hazard fork+Qt+CUDA); o processo da GUI NUNCA importa JAX → o TLS estático fica
    íntegro p/ a ``libgomp``/Numba (evita o crash ``_dl_allocate_tls_init``).

    Args:
        fn: callable módulo-nível PICKLABLE (ex.: ``_run_simulation``).
        args: posicionais PICKLABLE repassados a ``fn`` no subprocesso.
        kwargs: nomeados PICKLABLE repassados a ``fn`` no subprocesso.

    Returns:
        O retorno de ``fn`` (PICKLABLE — ex.: dict com ``H6`` ndarray).

    Note:
        Pool EFÊMERO (1 worker, criado/destruído por chamada): simples e correto —
        o ``spawn`` + init CUDA custa ~s por run (amortizado pela GPU em batch
        grande). Pool persistente + warmup = otimização futura (ver spec 0012 OUT).
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            return pool.submit(fn, *args, **kwargs).result()
    except BrokenProcessPool as exc:
        # Worker morreu sem devolver exceção Python (OOM killer/GPU OOM/crash do
        # driver CUDA/segfault). A msg padrão ("terminated abruptly") não ajuda —
        # re-levanta com causa acionável (o Worker capta → error VMSignal → UI).
        raise RuntimeError(
            "O subprocesso de simulação encerrou abruptamente — provável falta de "
            "memória (GPU/CPU), crash do driver CUDA ou segfault. Tente um batch "
            "menor (nº de modelos) ou o backend 'numba'."
        ) from exc


# ──────────────────────────────────────────────────────────────────────────────
# Pool JAX PERSISTENTE (singleton) — fecha a disparidade CLI×SM
# ──────────────────────────────────────────────────────────────────────────────
# O ``_pool_run`` acima é EFÊMERO (cria/destrói o subprocesso por chamada): simples,
# mas re-paga init CUDA + reload do cache XLA de disco + import a CADA run (~17 s
# medido). O CLI é rápido porque roda o JAX IN-PROCESS (CUDA + cache JIT quentes entre
# runs). Para o SM ter o MESMO comportamento sem reintroduzir o hazard fork+Qt+CUDA,
# mantemos UM subprocesso ``spawn`` VIVO e reutilizado entre runs (espelha o pool
# persistente do Numba em ``simulation/_workers.py``). Resultado: 2º+ run cai de ~29 s
# p/ ~12 s (medido A6000). Invariantes:
#   • ``base.py`` NUNCA importa ``jax`` em escopo de módulo — só o subprocesso filho
#     importa JAX (TLS-safe: o processo da GUI fica íntegro p/ libgomp/Numba).
#   • Teardown é responsabilidade do chamador: ``release_jax_pool()`` em
#     ``atexit`` (registrado aqui) + ``closeEvent`` da janela + ``aboutToQuit``.
#   • ``_pool_run`` (efêmero) permanece INTOCADO — é o default seguro e o contrato dos
#     testes diretos ``test_pool_run_*``. O persistente é opt-in (``persistent=True``).
_JAX_POOL: Optional[Any] = None
_JAX_POOL_LOCK = threading.Lock()
_JAX_POOL_ATEXIT_DONE = False


def _acquire_jax_pool() -> Any:
    """Retorna (ou cria) o ``ProcessPoolExecutor`` ``spawn`` de 1 worker PERSISTENTE.

    Singleton protegido por ``_JAX_POOL_LOCK``. Na 1ª criação registra
    ``atexit.register(release_jax_pool)`` (uma única vez) p/ não vazar o subprocesso
    GPU na saída do interpretador. NÃO importa ``jax`` — só faz ``spawn`` de um Python
    limpo; o JAX é importado DENTRO do filho (via ``fn=_run_simulation``). O ``spawn``
    roda ``_jax_pool_initializer`` ANTES de qualquer trabalho (ordem do env de VRAM).

    O initializer vive em ``sim_request`` (módulo Qt-FREE, JÁ importado pelo filho via
    ``_run_simulation``) — NÃO aqui — para que o filho resolva a referência ``spawn``
    sem importar este ``base.py``, que puxa Qt (mantém o filho enxuto/sem Qt no cold-start).

    Returns:
        O ``ProcessPoolExecutor`` persistente (1 worker, contexto ``spawn``).
    """
    global _JAX_POOL, _JAX_POOL_ATEXIT_DONE
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from geosteering_ai.gui.services.sim_request import _jax_pool_initializer

    with _JAX_POOL_LOCK:
        if _JAX_POOL is None:
            ctx = multiprocessing.get_context("spawn")
            _JAX_POOL = ProcessPoolExecutor(
                max_workers=1, mp_context=ctx, initializer=_jax_pool_initializer
            )
            if not _JAX_POOL_ATEXIT_DONE:
                atexit.register(release_jax_pool)
                _JAX_POOL_ATEXIT_DONE = True
        return _JAX_POOL


def _release_jax_pool_unlocked() -> None:
    """Encerra e zera o pool JAX persistente (SEM lock — uso interno).

    ``wait=False`` (não bloqueia o fechamento da UI num run longo) +
    ``cancel_futures=True`` (aborta tarefas pendentes; Python 3.9+). Best-effort —
    qualquer falha de shutdown é engolida (o objetivo é só liberar o recurso).
    """
    global _JAX_POOL
    if _JAX_POOL is not None:
        try:
            _JAX_POOL.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover — Python < 3.9 (sem cancel_futures)
            _JAX_POOL.shutdown(wait=False)
        except Exception:  # noqa: BLE001 — teardown best-effort; nunca propaga
            pass
        _JAX_POOL = None


def release_jax_pool() -> None:
    """Encerra o pool JAX persistente (público; idempotente; thread-safe).

    Chamar em ``closeEvent`` da janela, ``aboutToQuit`` e ``atexit`` (defense-in-depth)
    p/ não vazar o subprocesso GPU (contexto CUDA + VRAM). No-op seguro se não há pool.
    Após o release, a próxima chamada a :func:`_acquire_jax_pool` cria um pool novo.
    """
    with _JAX_POOL_LOCK:
        _release_jax_pool_unlocked()


def _discard_failed_pool(pool: Any) -> bool:
    """Descarta o pool que FALHOU e diz se a falha veio de um release EXTERNO (teardown).

    Distingue (revisão adversarial PR-worker-persistente) dois cenários de falha:
      • **release externo** (``release_jax_pool`` de encerramento — closeEvent/
        aboutToQuit/atexit — trocou/nulou o singleton): retorna ``True`` → o chamador
        ABORTA limpo (NÃO recria pool nem re-roda a sim durante o shutdown).
      • **crash do nosso pool** (o filho morreu sozinho; o singleton ainda é ``pool``):
        zera o singleton e retorna ``False`` → o chamador RE-TENTA com um pool fresco.

    Faz ``shutdown`` do pool ESPECÍFICO que falhou (``pool``, não o global) — assim, se
    um runner concorrente já recriou um pool novo, este NÃO é derrubado por engano. O
    ``shutdown`` roda FORA do lock (não bloqueia outras threads).

    Args:
        pool: o ``ProcessPoolExecutor`` que acabou de falhar (submit/result).

    Returns:
        ``True`` se a falha decorreu de um release externo (abortar); ``False`` se foi
        crash do próprio ``pool`` (re-tentar).
    """
    global _JAX_POOL
    with _JAX_POOL_LOCK:
        external = _JAX_POOL is not pool  # teardown trocou/nulou o singleton?
        if not external:
            _JAX_POOL = None  # nosso pool morreu → zera p/ o próximo acquire recriar
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except TypeError:  # pragma: no cover — Python < 3.9 (sem cancel_futures)
        pool.shutdown(wait=False)
    except Exception:  # noqa: BLE001 — best-effort; nunca propaga
        pass
    return external


def _pool_run_persistent(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """Como :func:`_pool_run`, mas REUSA o subprocesso PERSISTENTE (CUDA/JIT quentes).

    Trata 3 desfechos da chamada ao pool, distinguindo-os com cuidado:
      • **exceção do PRÓPRIO ``fn``** (ValueError/RuntimeError da simulação): propaga de
        ``result()`` SEM mascaramento nem retry (só ``BrokenProcessPool``/``CancelledError``
        são capturados ali — ambos são falhas de INFRAESTRUTURA, não do ``fn``).
      • **release externo** (``release_jax_pool`` de encerramento cancelou um future
        PENDENTE → ``CancelledError``; ou desligou o pool → ``RuntimeError`` no submit):
        ABORTA limpo com :data:`_SHUTDOWN_CANCEL_MSG` (sem recriar pool/re-rodar no
        shutdown; evita o erro VAZIO e o spawn de filho durante o fechamento).
      • **crash do filho** (CUDA/OOM/segfault → ``BrokenProcessPool``): self-heal —
        descarta o pool e re-tenta 1× com um filho fresco; 2ª falha → :data:`_SUBPROCESS_DIED_MSG`.

    Args:
        fn: callable módulo-nível PICKLABLE (ex.: ``_run_simulation``).
        args/kwargs: argumentos PICKLABLE repassados a ``fn`` no subprocesso.

    Returns:
        O retorno de ``fn`` (PICKLABLE).

    Raises:
        RuntimeError: crash do subprocesso 2× (``_SUBPROCESS_DIED_MSG``); release externo
            durante a sim (``_SHUTDOWN_CANCEL_MSG``); ou exceção do próprio ``fn``.
    """
    from concurrent.futures import CancelledError
    from concurrent.futures.process import BrokenProcessPool

    for attempt in (0, 1):
        pool = _acquire_jax_pool()
        try:
            future = pool.submit(fn, *args, **kwargs)
        except (BrokenProcessPool, RuntimeError) as exc:
            # submit recusado: pool quebrado OU desligado (shutdown). _discard_failed_pool
            # distingue release externo (→ aborta) de crash do nosso pool (→ re-tenta).
            if _discard_failed_pool(pool):
                raise RuntimeError(_SHUTDOWN_CANCEL_MSG) from exc
            if attempt == 1:
                raise RuntimeError(_SUBPROCESS_DIED_MSG) from exc
            continue
        try:
            return future.result()
        except (BrokenProcessPool, CancelledError) as exc:
            # Filho morreu (BrokenProcessPool) OU future cancelado por release externo
            # (CancelledError — cancel_futures=True num future ainda PENDENTE). Exceções
            # do PRÓPRIO fn NÃO caem aqui (não são esses tipos) → propagam intactas.
            if _discard_failed_pool(pool):
                raise RuntimeError(_SHUTDOWN_CANCEL_MSG) from exc
            if attempt == 1:
                raise RuntimeError(_SUBPROCESS_DIED_MSG) from exc
    raise AssertionError(
        "unreachable"
    )  # pragma: no cover — o laço sempre retorna/levanta


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

    def _run_async(
        self,
        fn: Callable[..., Any],
        *args: Any,
        report_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Roda ``fn(*args, **kwargs)`` num ``Worker`` off-thread.

        Os sinais do Worker são conectados a slots DESTE QObject (main thread) →
        os ``VMSignal`` são emitidos na main thread (ver header).

        Args:
            fn: callable pesado (módulo-nível, ex.: ``_run_simulation``).
            report_progress: se ``True``, o Worker injeta ``progress_callback=`` em
                ``fn`` (feedback de progresso → ``progress`` VMSignal; Fatia 6a). Só
                no caminho in-thread (numba) — o subprocesso (jax) não usa.
            *args/**kwargs: argumentos repassados a ``fn``.
        """
        worker = Worker(fn, *args, report_progress=report_progress, **kwargs)
        # QueuedConnection EXPLÍCITA: o worker emite na worker thread; o slot DEVE
        # rodar na main thread (afinidade deste QObject). AUTO já resolveria assim,
        # mas explícito blinda contra regressão se a afinidade mudar num refactor.
        queued = Qt.ConnectionType.QueuedConnection
        worker.signals.finished.connect(self._on_worker_finished, queued)
        worker.signals.error.connect(self._on_worker_error, queued)
        worker.signals.progress.connect(self._on_worker_progress, queued)
        thread = run_in_thread(worker)
        self._prune_threads()  # solta pares já terminados antes de guardar o novo
        self._threads.append((thread, worker))

    def _run_in_subprocess(
        self,
        fn: Callable[..., Any],
        *args: Any,
        persistent: bool = False,
        **kwargs: Any,
    ) -> None:
        """Como :meth:`_run_async`, mas roda ``fn`` num SUBPROCESSO (spawn) — TLS-safe.

        Necessário p/ backends que inicializam JAX/CUDA: numa ``QThread`` o init do
        CUDA estoura o TLS (``_dl_allocate_tls_init``). Aqui o Worker/QThread só
        BLOQUEIA no resultado do subprocesso (via :func:`_pool_run`); o processo da
        GUI não importa JAX. Resultado/erro chegam pelos mesmos ``VMSignal``
        (``finished``/``error``) na MAIN thread (idêntico ao caminho in-thread).

        Args:
            fn: callable módulo-nível PICKLABLE (rodará no subprocesso).
            persistent: se ``True``, usa o subprocesso PERSISTENTE
                (:func:`_pool_run_persistent` — CUDA/JIT quentes entre runs, ~12 s no 2º+
                run vs ~29 s do efêmero). Default ``False`` (pool efêmero, contrato dos
                testes diretos). O caminho ``jax``/``auto`` do SM passa ``persistent=True``.
            *args/**kwargs: argumentos PICKLABLE repassados a ``fn``.
        """
        # O Worker roda runner(fn, args, kwargs) na QThread → submete fn ao ProcessPool
        # spawn (efêmero ou persistente) e aguarda. fn + args são picklados p/ o filho.
        runner = _pool_run_persistent if persistent else _pool_run
        self._run_async(runner, fn, args, kwargs)

    def _prune_threads(self) -> None:
        """Solta os pares ``(thread, worker)`` cujas threads já terminaram.

        Chamado em :meth:`_run_async` (antes de adicionar) e nos slots
        ``_on_worker_finished``/``_on_worker_error`` (na MAIN thread, APÓS o
        ``emit`` — o resultado já foi entregue, então soltar a ref do worker não
        descarta um sinal em-voo). Mantém ``_threads`` enxuto sem o "leak" de
        refs mortas que acumularia até a próxima simulação.

        Note:
            NÃO chamar em :meth:`is_busy` (polled pela UI): pruning ali poderia
            soltar um ``worker`` cujo ``finished`` ainda está na fila de eventos
            da main thread → Qt descartaria o ``QMetaCallEvent`` pendente (sender
            destruído) → RESULTADO PERDIDO. O pruning só é seguro DEPOIS da entrega.
        """
        self._threads = [(t, w) for (t, w) in self._threads if t.isRunning()]

    def is_busy(self) -> bool:
        """``True`` se alguma worker thread ainda está rodando.

        Útil para a UI (ex.: bloquear "fechar" durante o trabalho) e para
        teardown limpo de teste (uma ``QThread`` destruída enquanto roda aborta).
        Predicado PURO (não muta ``_threads`` — ver :meth:`_prune_threads`).
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
        self.finished.emit(result)  # entrega ANTES do prune (sinal já consumido)
        self._prune_threads()

    def _on_worker_error(self, message: str) -> None:
        self.error.emit(message)
        self._prune_threads()

    def _on_worker_progress(self, done: int, total: int) -> None:
        self.progress.emit(done, total)
