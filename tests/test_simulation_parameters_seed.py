# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_parameters_seed.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — UI Parameters (Seed PRNG)            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-11 (v2.29 — realocado de test_simulation_pool_warmup) ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest + Qt (offscreen)                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garante que a v2.19 (Sprint 19.1) NÃO regrediu o controle de seed     ║
# ║    PRNG na UI. Este teste foi originalmente em                            ║
# ║    `test_simulation_pool_warmup.py` (deletado em v2.29 com a remoção     ║
# ║    de PoolWarmupThread e infraestrutura de warmup persistente).          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Teste de não-regressão do controle de seed PRNG na UI (Sprint v2.19)."""

from __future__ import annotations

import os

# Habilita Qt headless ANTES de importar PyQt — evita "could not connect
# to X server" em CI/sistemas sem display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_parameters_page_has_seed_widgets() -> None:
    """``ParametersPage`` deve ter ``chk_random_seed`` + ``spn_fixed_seed``.

    Sprint 19.1 (v2.19): UI de semente PRNG. Removê-los regride o bug
    funcional (modelos sempre idênticos) que esta versão corrigiu.
    """
    from geosteering_ai.simulation.tests.simulation_manager import ParametersPage
    from geosteering_ai.gui.qt_compat import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    page = ParametersPage()
    assert hasattr(page, "chk_random_seed")
    assert hasattr(page, "spn_fixed_seed")
    assert hasattr(page, "get_rng_seed")
    # Default v2.19 = aleatório
    assert page.chk_random_seed.isChecked() is True
    assert page.get_rng_seed() is None
