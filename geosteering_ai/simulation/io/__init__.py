# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/io/__init__.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Exportadores Fortran-compatíveis       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 2.2)                                   ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : stdlib + numpy                                             ║
# ║  Dependências: numpy (obrigatório)                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Exportadores opt-in que escrevem resultados do simulador Python em   ║
# ║    formatos idênticos aos gerados por `tatu.x` (simulador Fortran       ║
# ║    v10.0), para:                                                         ║
# ║                                                                           ║
# ║      • Reproduzir o modelo no simulador Fortran (validação cruzada)    ║
# ║      • Salvar amostras para uso externo em análises ou treino offline  ║
# ║      • Auditar paridade numérica entre os dois backends                 ║
# ║                                                                           ║
# ║  MÓDULOS                                                                  ║
# ║    • model_in.py   — escreve arquivos `model.in` (texto ASCII)         ║
# ║    • binary_dat.py — escreve `.dat` 22-col binário + `.out` metadata   ║
# ║                                                                           ║
# ║  OPT-IN                                                                   ║
# ║    Os exportadores são ativados via flags em `SimulationConfig`:        ║
# ║      • `export_model_in: bool = False`                                  ║
# ║      • `export_binary_dat: bool = False`                                ║
# ║      • `output_dir: str = "."`                                          ║
# ║      • `output_filename: str = "simulation"`                            ║
# ║                                                                           ║
# ║    Nenhum exportador executa se as flags estiverem desativadas.         ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/PerfilaAnisoOmp.f08 (writes_files, linha 1279)    ║
# ║    • Fortran_Gerador/fifthBuildTIVModels.py (escrita model.in)          ║
# ║    • .claude/commands/geosteering-simulator-python.md §6.1              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Exportadores Fortran-compatíveis do simulador Python (Sprint 2.2).

Subpacote com dois módulos:

- :mod:`model_in`   — escreve `model.in` ASCII idêntico ao formato v10.0.
- :mod:`binary_dat` — escreve `.dat` 22-col binário (stream, native endian)
  + `info{filename}.out` metadata em texto.

Ambos são **opt-in** via flags em `SimulationConfig`. Por default, o
simulador não toca no sistema de arquivos.

Example:
    Fluxo típico com exportação ativada::

        >>> from geosteering_ai.simulation import SimulationConfig
        >>> from geosteering_ai.simulation.io import export_model_in
        >>> cfg = SimulationConfig(
        ...     export_model_in=True,
        ...     output_dir="/tmp/sim",
        ...     output_filename="test",
        ... )
        >>> import numpy as np
        >>> rho_h = np.array([1.0, 10.0, 1.0])
        >>> rho_v = np.array([1.0, 10.0, 1.0])
        >>> thicknesses = np.array([1.52])
        >>> path = export_model_in(cfg, rho_h, rho_v, thicknesses)
"""
from __future__ import annotations

from geosteering_ai.simulation.io.binary_dat import (
    DTYPE_22COL,
    export_binary_dat,
    export_out_metadata,
)
from geosteering_ai.simulation.io.model_in import export_model_in

__all__ = [
    "DTYPE_22COL",
    "export_binary_dat",
    "export_model_in",
    "export_out_metadata",
]
