# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/main_window_base.py                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : MainWindowBase — casca QMainWindow host de perspectivas    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell MVVM (spec 0005)                               ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : Qt6 via gui.qt_compat (requer extra [gui])                  ║
# ║  Dependências: gui.qt_compat, gui.shell.context, gui.shell.perspective    ║
# ║  Padrão      : View (MVVM) — casca compartilhada SM + Studio              ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Casca base (``QMainWindow``) que hospeda ``Perspective``s como abas,   ║
# ║    com construção LAZY (cada perspectiva só constrói View/ViewModel ao    ║
# ║    ser ativada). Reusa ``gui/qt_compat`` (binding PyQt6/PySide6). SM e    ║
# ║    Studio subclassam para menu/toolbar/.session próprios.                 ║
# ║                                                                           ║
# ║  NOTA DE IMPORT                                                           ║
# ║    Este módulo IMPORTA Qt (é a View). NÃO é re-exportado por              ║
# ║    ``gui/shell/__init__.py`` — para que ``import geosteering_ai.gui.shell`` ║
# ║    (Perspective/AppContext) permaneça Qt-free. Acesse via este submódulo. ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    MainWindowBase                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``MainWindowBase`` — casca ``QMainWindow`` que hospeda perspectivas (abas, lazy).

A View base compartilhada pelo Simulation Manager e pelo Geosteering AI Studio.
Cada ``Perspective`` vira uma aba; a View/ViewModel da perspectiva só é construída
quando a aba é ATIVADA (lazy) — startup rápido. Reusa ``gui/qt_compat``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from geosteering_ai.gui.qt_compat import QtWidgets
from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.shell.perspective import Perspective

__all__ = ["MainWindowBase"]


class MainWindowBase(QtWidgets.QMainWindow):  # type: ignore[misc] # QtWidgets é Any (qt_compat)
    """Casca base ``QMainWindow`` que hospeda ``Perspective``s como abas (lazy).

    Attributes:
        ctx: o :class:`AppContext` compartilhado (título, serviços futuros).

    Example:
        >>> # (sob QApplication)
        >>> mw = MainWindowBase(AppContext(app_name="Geosteering AI Studio"))
        >>> mw.add_perspective(minha_perspectiva)   # vira aba; build no on_activate

    Note:
        Construção LAZY: ``build_view``/``build_viewmodel`` de uma perspectiva só
        rodam quando a aba é ativada (``currentChanged``). O rastreio de "já
        construída" usa a IDENTIDADE do container (``id(widget)``), imune ao
        deslocamento de índices ao inserir abas fora de ordem.

    Warning:
        A REMOÇÃO de abas (``removeTab``) NÃO é suportada nesta fundação. As chaves
        de ``_by_container``/``_built`` são ``id(widget)`` de containers vivos; o
        widget removido só seria coletado pelo GC após o C++-side do Qt liberar, e
        ``id`` PODE ser reciclado por um container futuro → falso-positivo de "já
        construída". Adicionar remoção exige limpar ambos os mapas no descarte do
        container (a ser tratado quando uma spec introduzir abas fecháveis).
    """

    def __init__(self, ctx: AppContext, *, parent: Optional[Any] = None) -> None:
        """Inicializa a casca com um ``QTabWidget`` central + statusbar.

        Args:
            ctx: contexto da aplicação (define o título da janela).
            parent: widget pai opcional (default ``None``).
        """
        super().__init__(parent)
        self.ctx: AppContext = ctx
        self.setWindowTitle(ctx.app_name)

        # Mapas keyed pela IDENTIDADE do container (estável; imune a shift de índice).
        self._by_container: Dict[int, Perspective] = {}
        self._built: set[int] = set()

        # Host de perspectivas — hook overridável (default: abas; Antigravity: rail+stack).
        self._init_host()
        self.statusBar().showMessage("Pronto")
        self.apply_theme()

    # ── Host de perspectivas (hooks overridáveis pela subclasse) ───────────
    def _init_host(self) -> None:
        """Cria o host de perspectivas. Default: ``QTabWidget`` central (abas).

        Subclasses (ex.: ``AntigravityMainWindow``) sobrescrevem para trocar o
        host (activity rail + ``QStackedWidget``), mantendo a lógica de
        registro/lazy-build da base via os accessors ``_host_*``.
        """
        self._tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self._tabs)
        self._tabs.currentChanged.connect(self._on_host_index_changed)

    def _host_count(self) -> int:
        """Número de perspectivas no host (default: nº de abas)."""
        return int(self._tabs.count())  # int(): QtWidgets é Any (qt_compat) → mypy

    def _host_widget(self, index: int) -> Any:
        """Container da perspectiva no ``index`` (default: aba)."""
        return self._tabs.widget(index)

    def _host_current_index(self) -> int:
        """Índice da perspectiva ativa (default: aba atual)."""
        return int(self._tabs.currentIndex())  # int(): QtWidgets é Any → mypy

    def _host_insert(
        self, index: int, container: Any, perspective: Perspective
    ) -> None:
        """Insere o ``container`` da perspectiva no host (default: ``insertTab``)."""
        self._tabs.insertTab(index, container, perspective.title)

    # ── API pública ──────────────────────────────────────────────────────
    @property
    def perspective_count(self) -> int:
        """Número de perspectivas registradas (abas ou itens da rail+stack)."""
        return self._host_count()

    def add_perspective(self, perspective: Perspective) -> None:
        """Registra uma perspectiva no host (ordenada por ``perspective.order``).

        A View/ViewModel é construída de forma LAZY (só ao ativar). Se a inserida
        for a atual (ex.: a primeira), constrói imediatamente.

        Args:
            perspective: instância concreta de :class:`Perspective`.
        """
        index = self._insertion_index(perspective.order)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Registra ANTES de inserir: a inserção pode emitir ``currentChanged``.
        self._by_container[id(container)] = perspective
        self._host_insert(index, container, perspective)

        # Garante a construção da perspectiva ATUAL (idempotente).
        self._ensure_built(self._host_current_index())

    def apply_theme(self) -> None:
        """Aplica o tema da casca. Default: no-op (subclasses sobrescrevem)."""
        return None

    # ── Internos ─────────────────────────────────────────────────────────
    def _insertion_index(self, order: int) -> int:
        """Posição de inserção que mantém as abas ordenadas por ``order``.

        Args:
            order: o ``perspective.order`` da aba a inserir.

        Returns:
            int: índice ANTES da primeira aba existente com ``order`` maior; ou o
            total de abas (append) se nenhuma for maior. Empates preservam a ordem
            de inserção (estável — o ``>`` estrito não desloca iguais).
        """
        for i in range(self._host_count()):
            container = self._host_widget(i)
            existing = self._by_container.get(id(container))
            if existing is not None and existing.order > order:
                return i
        return self._host_count()

    def _on_host_index_changed(self, index: int) -> None:
        """Slot de mudança de índice do host — constrói a perspectiva ativada (lazy)."""
        self._ensure_built(index)

    def _ensure_built(self, index: int) -> None:
        """Constrói a View da perspectiva no índice (uma única vez).

        Idempotente: já-construídas (por identidade do container) são ignoradas.
        """
        if index < 0:
            return
        container = self._host_widget(index)
        if container is None or id(container) in self._built:
            return
        perspective = self._by_container.get(id(container))
        if (
            perspective is None
        ):  # pragma: no cover — container sem perspectiva (defensivo)
            return
        perspective.on_activate()
        view = perspective.build_view(self.ctx)
        layout = container.layout()
        if layout is not None and view is not None:
            layout.addWidget(view)
        self._built.add(id(container))
