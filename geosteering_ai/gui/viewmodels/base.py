# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/viewmodels/base.py                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : BaseViewModel — base de ViewModel PURO (MVVM)              ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — base MVVM (spec 0005)                                ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : stdlib PURO (NÃO importa Qt) — Princípio X                  ║
# ║  Dependências: gui.viewmodels.signal (VMSignal)                           ║
# ║  Padrão      : MVVM — ViewModel testável sem pytest-qt; estado serializável║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Base para ViewModels do SM e do Studio. Mantém estado de UI OBSERVÁVEL ║
# ║    (notifica a View via ``changed`` VMSignal) e SERIALIZÁVEL (to_dict/     ║
# ║    from_dict — fundação para ``.session``/``.gsproj``, specs 0007/0018).  ║
# ║    NÃO importa Qt → 80% da lógica de UI roda como pytest comum.           ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    BaseViewModel                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``BaseViewModel`` — base de ViewModel puro-Python para o MVVM da GUI.

O estado de UI vive no ViewModel (não na View) — por isso é OBSERVÁVEL (via o
``VMSignal`` ``changed``) e SERIALIZÁVEL (``to_dict``/``from_dict``). Em MVVM, o
estado da UI É o ViewModel; salvá-lo = persistir a sessão (specs 0007/0018).

Example:
    >>> class CounterVM(BaseViewModel):
    ...     _STATE_FIELDS = ("count",)
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.count = 0
    ...     def increment(self) -> None:
    ...         self._set("count", self.count + 1)
    >>> vm = CounterVM()
    >>> recebidos = []
    >>> vm.changed.connect(lambda name, value: recebidos.append((name, value)))
    >>> vm.increment()
    >>> recebidos
    [('count', 1)]
    >>> vm.to_dict()
    {'count': 1}
"""

from __future__ import annotations

from typing import Any

from geosteering_ai.gui.viewmodels.signal import VMSignal

__all__ = ["BaseViewModel"]

_UNSET = object()  # sentinela: atributo ainda não definido (1ª atribuição sempre emite)


class BaseViewModel:
    """Base de ViewModel PURO (sem Qt) com estado observável e serializável.

    Attributes:
        changed: ``VMSignal`` emitido como ``changed(name, value)`` a cada mudança
            de estado feita via :meth:`_set`.

    Class Attributes:
        _STATE_FIELDS: nomes dos atributos serializados por :meth:`to_dict`/
            :meth:`from_dict`. Subclasses declaram (default vazio).

    Note:
        :meth:`_set` compara com ``==`` (estado escalar). ViewModels com estado
        de array (NumPy) devem sobrescrever :meth:`_set` para a comparação correta.
        Não importa Qt — a View adapta ``changed`` a um slot Qt (ver ``VMSignal``).
    """

    _STATE_FIELDS: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.changed: VMSignal = VMSignal()

    def _set(self, name: str, value: Any) -> bool:
        """Atribui ``self.<name> = value`` e emite ``changed`` se o valor mudou.

        A comparação usa :meth:`_equal` (segura para tipos cujo ``==`` não reduz a
        ``bool`` — ex.: arrays NumPy). Subclasses com estado de array podem
        sobrescrever :meth:`_equal` para deduplicação precisa (ex.: ``np.array_equal``).

        Args:
            name: nome do atributo de estado.
            value: novo valor.

        Returns:
            bool: ``True`` se o valor mudou (e ``changed`` foi emitido); ``False``
            se igual ao atual (no-op).
        """
        old = getattr(self, name, _UNSET)
        if old is not _UNSET and self._equal(old, value):
            return False
        setattr(self, name, value)
        self.changed.emit(name, value)
        return True

    @staticmethod
    def _equal(old: Any, value: Any) -> bool:
        """Igualdade SEGURA p/ a deduplicação de :meth:`_set` (default: ``==`` escalar).

        Tipos cujo ``==`` NÃO reduz a ``bool`` (arrays NumPy levantam
        ``ValueError: truth value of an array is ambiguous``) caem para
        "não-igual" → :meth:`_set` RE-EMITE (conservador: nunca PERDE uma
        notificação, no pior caso emite a mais).

        Args:
            old: valor atual.
            value: valor novo.

        Returns:
            bool: ``True`` se considerados iguais (sem emitir).

        Note:
            ViewModels com estado de array devem sobrescrever para precisão::

                @staticmethod
                def _equal(old, value):
                    import numpy as np
                    return np.array_equal(old, value)
        """
        try:
            return bool(old == value)
        except (ValueError, TypeError):  # ex.: array NumPy → '==' não reduz a bool
            return False

    def to_dict(self) -> dict[str, Any]:
        """Serializa o estado declarado em :attr:`_STATE_FIELDS`.

        Returns:
            dict mapeando cada campo de estado ao seu valor atual.
        """
        return {field: getattr(self, field) for field in self._STATE_FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseViewModel":
        """Reconstrói um ViewModel a partir de ``to_dict`` (round-trip).

        Args:
            data: dict produzido por :meth:`to_dict` (campos em :attr:`_STATE_FIELDS`).

        Returns:
            Nova instância com o estado restaurado. Campos ausentes em ``data``
            mantêm o default da subclasse (forward-compat com snapshots antigos).

        Note:
            CONTRATO: a subclasse DEVE ser construtível sem argumentos (``cls()``).
            ViewModels que exijam dependências no ``__init__`` devem ou (a) aceitá-las
            com default, ou (b) sobrescrever ``from_dict``. Espelha ``to_dict``, que
            só serializa :attr:`_STATE_FIELDS` (não dependências de construção).
        """
        obj = cls()
        for field in cls._STATE_FIELDS:
            if field in data:
                setattr(obj, field, data[field])
        return obj
