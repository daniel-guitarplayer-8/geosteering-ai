# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/postprocess/__init__.py                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Pós-processamento (F6, F7)             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 2.2)                                   ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x                                                  ║
# ║  Dependências: numpy                                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Módulos de pós-processamento que operam sobre o tensor H complexo   ║
# ║    produzido pelo forward (Sprints 2.2+) para simular features         ║
# ║    avançadas do Fortran v10.0:                                          ║
# ║                                                                           ║
# ║      • F6 (compensation.py) — Compensação Midpoint Schlumberger CDR    ║
# ║      • F7 (tilted.py)       — Antenas Inclinadas (tilted beams)        ║
# ║                                                                           ║
# ║    Ambos são **opt-in** via flags em `SimulationConfig`:                ║
# ║      • `use_compensation: bool = False`                                 ║
# ║      • `use_tilted_antennas: bool = False`                              ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/PerfilaAnisoOmp.f08:714-730 (F7 tilted)           ║
# ║    • Fortran_Gerador/PerfilaAnisoOmp.f08:804-868,1320-1419 (F6 comp)  ║
# ║    • Schlumberger CDR literature (compensated dual resistivity)        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Pós-processamento de tensores H forward (Sprint 2.2)."""
from __future__ import annotations

from geosteering_ai.simulation.postprocess.compensation import apply_compensation
from geosteering_ai.simulation.postprocess.tilted import apply_tilted_antennas

__all__ = [
    "apply_compensation",
    "apply_tilted_antennas",
]
