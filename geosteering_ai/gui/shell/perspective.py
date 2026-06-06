# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/perspective.py                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Perspective ABC — contrato de plugin de aba (MVVM)         ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell MVVM (spec 0005)                               ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação                                        ║
# ║  Framework   : abc + typing (Qt SÓ em TYPE_CHECKING → testável sem Qt)     ║
# ║  Dependências: gui.shell.context, gui.viewmodels.base                     ║
# ║  Padrão      : ABC (espelha o PlotCanvas ABC dos plot_backends)           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Contrato de PLUGIN para cada aba/perspectiva do SM e do Studio. Uma    ║
# ║    perspectiva produz (a) um ViewModel PURO e (b) uma View Qt que faz o   ║
# ║    binding aos sinais do VM. O ``MainWindowBase`` hospeda perspectivas;   ║
# ║    o Studio as descobre por ``entry_points`` (spec futura — 0013).        ║
# ║                                                                           ║
# ║  IMPORTÁVEL SEM Qt                                                        ║
# ║    ``QWidget`` aparece só em ``TYPE_CHECKING`` (+ ``from __future__``) →   ║
# ║    importar este módulo NÃO importa PyQt6/PySide6 (Princípio X / RNF-3).  ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    Perspective                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``Perspective`` ABC — contrato de plugin de aba/perspectiva (MVVM).

Cada perspectiva (Simulação, Treinamento, Inferência, Geonavegação Realtime,
Model Registry) implementa este contrato. Espelha o padrão Strategy/ABC dos
``plot_backends``. Importável SEM Qt (``QWidget`` só em ``TYPE_CHECKING``) — o
contrato e os ViewModels permanecem testáveis sem ``pytest-qt`` (Princípio X).

.. code-block:: text

    MainWindowBase (host)  ──add_perspective(p)──▶  cria aba
        on_activate (lazy) ─▶ p.build_viewmodel(ctx)  [VM PURO]
                            └▶ p.build_view(ctx)       [View Qt faz binding ao VM]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from geosteering_ai.gui.shell.context import AppContext
from geosteering_ai.gui.viewmodels.base import BaseViewModel

if (
    TYPE_CHECKING
):  # pragma: no cover — só para type-checking; nunca importado em runtime
    # Via qt_compat (NÃO PyQt6 direto): acesso Qt agnóstico ao binding (PyQt6
    # primário / PySide6 fallback). Em runtime o bloco nunca executa.
    from geosteering_ai.gui.qt_compat import QtWidgets

__all__ = ["Perspective"]


class Perspective(ABC):
    """Contrato de uma perspectiva (aba) plugável do SM/Studio.

    Attributes:
        id: identificador estável (ex.: ``"simulation"``) — chave do plugin.
        title: rótulo exibido na aba (ex.: ``"Simulação & Datasets"``).
        icon: nome/recurso do ícone (ex.: ``"flask"``); ``""`` se nenhum.
        order: posição relativa entre as abas (menor = mais à esquerda).

    Note:
        ``build_view`` e ``build_viewmodel`` são ABSTRATOS — uma subclasse que não
        os implemente NÃO pode ser instanciada (``TypeError``). ``on_activate``/
        ``on_close`` têm default não-abstrato (no-op / sem veto). A View criada por
        ``build_view`` deve conectar os ``VMSignal``s do ViewModel aos seus slots Qt
        (binding) — o ViewModel permanece sem importar Qt.

    Warning:
        ``id``/``title``/``icon``/``order`` são atributos de CLASSE com defaults
        vazios (``""``/``0``). Toda subclasse concreta DEVE declarar ao menos
        ``id`` (não-vazio, estável — chave do plugin) e ``title`` (rótulo da aba);
        herdar os defaults vazios produz abas anônimas e colisão de chave entre
        perspectivas distintas. Os defaults existem só para o type-checker — não
        são valores válidos de runtime. (São tipos imutáveis: não há risco de
        estado mutável compartilhado entre instâncias.)
    """

    id: str = ""
    title: str = ""
    icon: str = ""
    order: int = 0

    @abstractmethod
    def build_view(self, ctx: AppContext) -> "QtWidgets.QWidget":
        """Cria a View (``QWidget`` Qt) e faz o binding aos sinais do ViewModel.

        Args:
            ctx: contexto compartilhado da aplicação.

        Returns:
            O ``QWidget`` raiz da perspectiva (hospedado como aba).
        """
        raise NotImplementedError

    @abstractmethod
    def build_viewmodel(self, ctx: AppContext) -> BaseViewModel:
        """Cria o ViewModel PURO (sem Qt) da perspectiva.

        Args:
            ctx: contexto compartilhado da aplicação.

        Returns:
            O ``BaseViewModel`` (ou subclasse) que carrega o estado de UI.
        """
        raise NotImplementedError

    def on_activate(self) -> None:
        """Chamado quando a aba é aberta (lazy build). Default: no-op."""
        return None

    def on_close(self) -> bool:
        """Chamado ao tentar fechar a aba.

        Returns:
            bool: ``True`` libera o fechamento; ``False`` VETA (ex.: undo sujo).
            Default: ``True`` (sem veto).
        """
        return True
