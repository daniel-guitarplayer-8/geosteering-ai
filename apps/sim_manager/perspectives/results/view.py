# -*- coding: utf-8 -*-
"""Re-export da ``ResultsView`` para a perspectiva Resultados (Fatia 6i / PR-2).

A View da galeria já existe em ``simulation/results_view.py`` (ligada a um
``ResultsViewModel`` puro). Para honrar a estrutura ``perspectives/results/`` sem
duplicar código Qt nem mover o arquivo (o que quebraria imports/testes da Simulação),
este módulo apenas re-exporta a ``ResultsView`` existente.
"""

from __future__ import annotations

from apps.sim_manager.perspectives.simulation.results_view import ResultsView

__all__ = ["ResultsView"]
