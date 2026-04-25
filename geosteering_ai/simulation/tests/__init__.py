# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/__init__.py                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Aplicação GUI PyQt (testes)          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Agrega o programa Simulation Manager — interface gráfica PyQt6 (com   ║
# ║    fallback automático para PySide6 / PyQt5) para orquestração do        ║
# ║    simulador Python Numba JIT otimizado e do simulador Fortran tatu.x,  ║
# ║    incluindo geração estocástica de modelos TIV, benchmarking e         ║
# ║    visualização de campos eletromagnéticos.                              ║
# ║                                                                           ║
# ║  MÓDULOS INTERNOS                                                         ║
# ║    ┌──────────────────────────┬───────────────────────────────────────┐  ║
# ║    │  sm_qt_compat.py         │  Shim PyQt6/PySide6/PyQt5 + locale C  │  ║
# ║    │  sm_model_gen.py         │  Geradores aleatórios (Sobol, etc.)   │  ║
# ║    │  sm_workers.py           │  QThread + ProcessPool sandbox        │  ║
# ║    │  sm_benchmark.py         │  Config A/B + 30k experiment          │  ║
# ║    │  sm_plots.py             │  Plots EM + PlotStyle + tensor 3×6    │  ║
# ║    │  sm_io.py                │  Export .dat/.out/CSV/MD              │  ║
# ║    │  simulation_manager.py   │  MainWindow macOS/VSCode + entrypoint │  ║
# ║    └──────────────────────────┴───────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  EXECUÇÃO                                                                 ║
# ║    python -m geosteering_ai.simulation.tests.simulation_manager          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Subpacote de testes e ferramentas GUI do simulador Python Numba JIT.

Este subpacote contém o aplicativo Simulation Manager, uma interface gráfica
baseada em PyQt que orquestra os simuladores Python (Numba JIT otimizado) e
Fortran (``tatu.x``) sob o paradigma Central Master-Plan / Parallel Execution
(Worker Sandboxes), oferecendo geração estocástica de modelos geológicos,
benchmarks comparativos e visualização de campos EM.

Note:
    O arquivo ``simulation_manager.py`` é o ponto de entrada único da
    aplicação. Os demais módulos (``sm_*.py``) são submódulos internos
    agregadores de responsabilidades bem definidas.
"""
from __future__ import annotations

__all__: list[str] = []
