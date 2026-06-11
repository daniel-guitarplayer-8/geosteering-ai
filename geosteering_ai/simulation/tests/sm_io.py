# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_io.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Export .dat / .out (Fortran-compat)   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Status      : Produção — SHIM de retrocompatibilidade                    ║
# ║  Dependências: geosteering_ai.simulation.io.tensor_dat (produção)         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    SHIM fino: re-exporta os escritores .dat/.out da implementação de     ║
# ║    PRODUÇÃO ``geosteering_ai.simulation.io.tensor_dat``. Recuperado do    ║
# ║    WIP v2.53 (item 4 da triagem): elimina ~170 linhas DUPLICADAS — a     ║
# ║    lógica idêntica vivia aqui E em ``io/tensor_dat.py``.                  ║
# ║                                                                           ║
# ║    Retrocompat 100% VERIFICADA antes do shim: ``compute_nmeds_per_angle``║
# ║    e ``write_out_file`` têm lógica BYTE-IDÊNTICA nas duas cópias;         ║
# ║    ``write_dat_from_tensor`` difere só por coerções ``np.asarray(...)``   ║
# ║    defensivas em ``tensor_dat`` (no-op em ndarray → mesma saída p/ todo   ║
# ║    input que esta cópia já aceitava; mais robusto p/ listas).             ║
# ║                                                                           ║
# ║    Consumidores preservados: ``simulation_manager.py`` (monólito,         ║
# ║    ``compute_nmeds_per_angle``), ``sm_workers.py`` e ``test_sm_workers_   ║
# ║    io.py`` (``write_dat_from_tensor``/``write_out_file``).                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Shim de retrocompatibilidade — re-exporta os escritores .dat/.out.

A implementação canônica vive em
``geosteering_ai.simulation.io.tensor_dat`` (módulo de produção). Este
módulo permanece como ponto de import estável para os consumidores
históricos (monólito do Simulation Manager, ``sm_workers``, testes), evitando
a duplicação de ~170 linhas que existia antes do shim.

Note:
    Mantido em ``simulation/tests/`` apenas por compatibilidade de import.
    Código novo deve importar diretamente de
    ``geosteering_ai.simulation.io.tensor_dat``.
"""

from __future__ import annotations

# Re-export da implementação de produção (ponto único de verdade).
from ..io.tensor_dat import (
    compute_nmeds_per_angle,
    write_dat_from_tensor,
    write_out_file,
)

__all__ = ["compute_nmeds_per_angle", "write_dat_from_tensor", "write_out_file"]
