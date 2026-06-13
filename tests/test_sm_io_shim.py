# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_io_shim.py                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — sm_io shim de retrocompat             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-10                                                 ║
# ║  Status      : Produção (guarda anti-drift)                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Guarda do shim ``simulation/tests/sm_io.py`` (item 4 da triagem).

``sm_io`` virou um SHIM que re-exporta os escritores .dat/.out da implementação
de produção ``simulation/io/tensor_dat.py`` (elimina ~170 linhas duplicadas).
Estes testes garantem que o shim permanece um shim — i.e. que os símbolos
expostos por ``sm_io`` SÃO os de ``tensor_dat`` (impede a reintrodução de uma
cópia divergente que regrediria o monólito, que importa
``compute_nmeds_per_angle`` de ``sm_io``).
"""

from __future__ import annotations

_NAMES = ("compute_nmeds_per_angle", "write_dat_from_tensor", "write_out_file")


def test_sm_io_reexports_are_canonical_tensor_dat():
    """Cada símbolo de ``sm_io`` É o objeto de ``io.tensor_dat`` (identidade)."""
    from geosteering_ai.simulation.io import tensor_dat
    from geosteering_ai.simulation.tests import sm_io

    for name in _NAMES:
        assert getattr(sm_io, name) is getattr(tensor_dat, name), (
            f"sm_io.{name} divergiu de io.tensor_dat.{name} — o shim foi quebrado "
            "(uma cópia local reapareceu → risco de regressão no monólito)."
        )


def test_sm_io_all_matches_canonical():
    """``sm_io.__all__`` cobre exatamente os 3 escritores re-exportados."""
    from geosteering_ai.simulation.tests import sm_io

    assert sorted(sm_io.__all__) == sorted(_NAMES)


def test_monolith_consumer_import_path():
    """O caminho de import do monólito (``compute_nmeds_per_angle``) resolve."""
    from geosteering_ai.simulation.tests.sm_io import compute_nmeds_per_angle

    assert compute_nmeds_per_angle.__module__.endswith("io.tensor_dat")
