# -*- coding: utf-8 -*-
"""Perspectiva Simulação (MVVM) — VM puro + View Qt + Perspective (spec 0011a).

**Qt-free no nível do pacote** (Princípio X): este ``__init__`` NÃO importa Qt,
para que ``…simulation.viewmodel`` (PURO) seja importável sem puxar PyQt6/PySide6.
``SimulationPerspective`` (que importa ``gui.services`` Qt) vive em
``perspective.py`` — acesse explicitamente
(``from apps.sim_manager.perspectives.simulation.perspective import SimulationPerspective``).
"""

from __future__ import annotations

__all__: list[str] = []
