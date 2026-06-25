# -*- coding: utf-8 -*-
"""Perspectiva **Resultados** do SM MVVM (Fatia 6i / PR-2).

A galeria de plot do ensemble (curvas/perfis ρ-λ/heatmap) — antes embutida na aba
Simulação — vive aqui, numa perspectiva dedicada. Reusa a ``ResultsView`` e o
``ResultsViewModel`` da Simulação (padrão 1 VM, N Views): a Simulação publica o seu
``ResultsViewModel`` em ``ctx.extras["results_vm"]`` e esta perspectiva liga uma
2ª View ao MESMO VM, então o resultado simulado aparece aqui automaticamente.
"""

from __future__ import annotations

from apps.sim_manager.perspectives.results.perspective import ResultsPerspective

__all__ = ["ResultsPerspective"]
