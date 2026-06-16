# -*- coding: utf-8 -*-
"""Perspectiva **Visualizador .dat/.out** do SM MVVM (Fatia 6h).

Carrega artefatos Fortran-compat (``.dat`` binário 22-col + ``.out`` ASCII)
gravados pelo próprio SM (grupo "Saída" da Simulação) ou pelo ``tatu.x``, e os
inspeciona SEM re-simular (somente leitura — zero física).

Exporta a perspectiva plugável (``DatViewerPerspective``), o ViewModel PURO
(``DatViewerViewModel``) e o serviço de leitura (``DatViewerService``). A View Qt
(``DatViewerPanel``) é importada localmente em ``build_view`` para manter o
ViewModel testável sem Qt (Princípio X).
"""

from __future__ import annotations

from apps.sim_manager.perspectives.datviewer.perspective import DatViewerPerspective
from apps.sim_manager.perspectives.datviewer.service import (
    DatLoadResult,
    DatViewerService,
)
from apps.sim_manager.perspectives.datviewer.viewmodel import DatViewerViewModel

__all__ = [
    "DatViewerPerspective",
    "DatViewerViewModel",
    "DatViewerService",
    "DatLoadResult",
]
