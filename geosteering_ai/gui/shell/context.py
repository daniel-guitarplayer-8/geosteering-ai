# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/context.py                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : AppContext — contexto compartilhado da aplicação GUI       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell MVVM (spec 0005)                               ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção — fundação (mínimo, extensível)                   ║
# ║  Framework   : stdlib (dataclass) — sem Qt                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Objeto de contexto passado às ``Perspective``s (e Views/ViewModels).   ║
# ║    Carrega estado/serviços compartilhados da app. MÍNIMO nesta spec —     ║
# ║    cresce em specs futuras (services 0011, project/.gsproj 0018,          ║
# ║    model registry 0024) SEM quebrar o contrato ``Perspective``.           ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    AppContext                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``AppContext`` — contexto compartilhado passado às perspectivas (MVVM shell).

Mínimo nesta spec (0005); campos crescem incrementalmente (serviços, projeto,
registry) conforme as specs futuras, sem alterar a assinatura de ``Perspective``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppContext:
    """Contexto compartilhado da aplicação GUI (SM ou Studio).

    Attributes:
        app_name: nome exibido da aplicação (ex.: ``"Geosteering AI Studio"`` ou
            ``"Simulation Manager"``).
        extras: espaço de extensão para serviços/estado compartilhado que specs
            futuras (0011 services, 0018 project) injetarão — evita quebrar a
            assinatura de ``Perspective.build_view(ctx)`` a cada adição.

    Note:
        Deliberadamente MÍNIMO (spec 0005). Não importa Qt nem a biblioteca de
        domínio — é um *container* de contexto neutro.
    """

    app_name: str = "Geosteering AI"
    extras: dict[str, Any] = field(default_factory=dict)


__all__ = ["AppContext"]
