# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/viewmodels/signal.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : VMSignal — sinal pub/sub puro-Python (MVVM)                ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — base MVVM (spec 0005)                                ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : stdlib PURO (NÃO importa Qt) — Princípio X (VM testável)    ║
# ║  Dependências: logging                                                    ║
# ║  Padrão      : Observer/pub-sub (binding View↔ViewModel sem acoplamento)   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Permite que um ViewModel PURO (sem import de Qt) notifique a View de    ║
# ║    mudanças de estado. A View registra um slot Qt como callback; o teste  ║
# ║    registra uma lista/função comum — SEM QApplication, SEM pytest-qt.     ║
# ║                                                                           ║
# ║  BINDING (regra de ouro — blueprint §2.2)                                 ║
# ║    ViewModel:  self.progress = VMSignal()                                 ║
# ║    View:       vm.progress.connect(self._on_progress)   # adapter Qt       ║
# ║    Teste:      vm.progress.connect(rec.append)          # puro             ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    VMSignal                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``VMSignal`` — sinal pub/sub puro-Python para o binding MVVM (View ↔ ViewModel).

Uma lista de callbacks com ``connect``/``disconnect``/``emit``/``clear``. NÃO importa
Qt: o ViewModel permanece Python puro e testável sem ``pytest-qt`` (Princípio X).

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  ViewModel (puro)            VMSignal              View (Qt) / Teste   │
    │     self.changed  ── emit(x) ──▶ [cb1, cb2] ──▶ cb1(x)  (slot Qt)     │
    │                                              └─▶ cb2(x)  (rec.append) │
    └──────────────────────────────────────────────────────────────────────┘

Note:
    **Thread-safety**: ``emit`` é SÍNCRONO — chama os callbacks na thread que
    chamou ``emit``. Se um Service emite de uma worker thread, o *marshaling* para
    a thread da GUI é responsabilidade do **adapter da View** (que conecta um slot
    Qt com ``Qt.ConnectionType.QueuedConnection``) ou da camada de Service (spec
    futura — 0011). ``VMSignal`` em si NÃO toca Qt nem threads.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = ["VMSignal"]

_Callback = Callable[..., None]


class VMSignal:
    """Sinal pub/sub puro-Python (lista de callbacks) para o binding MVVM.

    Attributes:
        (privado) ``_callbacks``: lista ordenada de callbacks conectados.

    Example:
        >>> s = VMSignal()
        >>> recebidos = []
        >>> s.connect(recebidos.append)
        >>> s.emit(42)
        >>> recebidos
        [42]

    Note:
        ``connect`` é IDEMPOTENTE (o mesmo callback não é adicionado duas vezes).
        ``emit`` ISOLA exceções de cada callback (uma falha não impede os demais;
        a exceção é logada). Ver thread-safety no docstring do módulo.
    """

    __slots__ = ("_callbacks",)

    def __init__(self) -> None:
        self._callbacks: list[_Callback] = []

    def connect(self, callback: _Callback) -> None:
        """Conecta ``callback`` (idempotente — não duplica se já conectado).

        Args:
            callback: chamável invocado a cada ``emit`` com os mesmos ``*args``/
                ``**kwargs``.
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def disconnect(self, callback: _Callback) -> None:
        """Desconecta ``callback`` (no-op se não estiver conectado).

        Args:
            callback: o chamável previamente conectado.
        """
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass  # não conectado — no-op silencioso (idempotente)

    def emit(self, *args: Any, **kwargs: Any) -> None:
        """Notifica todos os callbacks conectados, isolando exceções.

        Itera sobre uma CÓPIA da lista (um callback pode ``disconnect`` a si mesmo
        durante a emissão sem corromper a iteração). Exceção em um callback é
        capturada e LOGADA — os demais callbacks ainda recebem a notificação.

        Args:
            *args: posicionais repassados a cada callback.
            **kwargs: nomeados repassados a cada callback.
        """
        for callback in list(self._callbacks):
            try:
                callback(*args, **kwargs)
            except Exception:  # noqa: BLE001 — isolamento: 1 cb ruim não derruba os demais
                logger.exception(
                    "VMSignal: callback %r levantou exceção (isolado)", callback
                )

    def clear(self) -> None:
        """Remove TODOS os callbacks (útil ao destruir a View/ViewModel)."""
        self._callbacks.clear()

    def __len__(self) -> int:
        """Número de callbacks conectados (útil em testes/diagnóstico)."""
        return len(self._callbacks)
