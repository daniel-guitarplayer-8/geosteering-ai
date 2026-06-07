# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_io.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Export .dat / .out (Fortran-compat)   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18 · 2026-06-02 (relocado p/ produção)             ║
# ║  Status      : Produção (shim de retrocompat)                             ║
# ║  Dependências: geosteering_ai.simulation.io.tensor_dat                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Shim de retrocompatibilidade. As implementações canônicas de          ║
# ║    ``write_dat_from_tensor`` / ``write_out_file`` / ``compute_nmeds_      ║
# ║    per_angle`` foram RELOCADAS na Sprint v2.53 para o módulo de           ║
# ║    PRODUÇÃO ``geosteering_ai/simulation/io/tensor_dat.py`` (para que a    ║
# ║    CLI e outros módulos de produção não importem de um subpacote          ║
# ║    ``tests``). Este módulo re-exporta dali — importadores existentes      ║
# ║    (``sm_workers.py``, ``simulation_manager.py``, ``tests/``) continuam   ║
# ║    funcionando sem alteração.                                            ║
# ║                                                                           ║
# ║  EXPORTS (re-export de io.tensor_dat)                                     ║
# ║    write_dat_from_tensor, write_out_file, compute_nmeds_per_angle         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Shim de retrocompat — re-exporta os escritores `.dat`/`.out`.

As implementações vivem em
:mod:`geosteering_ai.simulation.io.tensor_dat` (relocadas na Sprint v2.53).
Mantido para não quebrar os importadores legados que faziam
``from geosteering_ai.simulation.tests.sm_io import write_dat_from_tensor``.
"""

from __future__ import annotations

from geosteering_ai.simulation.io.tensor_dat import (
    compute_nmeds_per_angle,
    write_dat_from_tensor,
    write_out_file,
)

__all__ = ["compute_nmeds_per_angle", "write_dat_from_tensor", "write_out_file"]
