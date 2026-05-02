# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_pool_warmup.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager v2.18+v2.19 (não-regressão)             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-02                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest + Qt (offscreen)                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garante que a v2.19 (Sprints 19.1/19.2) NÃO regrediu o fix de v2.18:  ║
# ║    medição correta do throughput pós-pool-warm e pré-aquecimento de pool ║
# ║    em background. Estes testes complementam os smoke T29-T32 da v2.18.   ║
# ║                                                                           ║
# ║  COBERTURA DOS TESTES                                                     ║
# ║    1. PoolWarmupThread tem signals warmup_done + warmup_error            ║
# ║    2. PoolWarmupThread aceita argumentos n_workers, n_threads, filter    ║
# ║    3. SimulationPage cria lbl_warmup_status + _warmup_thread em init     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de não-regressão do PoolWarmupThread e t0_sim (v2.18 → v2.19)."""
from __future__ import annotations

import os

import pytest

# Habilita Qt headless ANTES de importar PyQt — evita "could not connect
# to X server" em CI/sistemas sem display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ──────────────────────────────────────────────────────────────────────────
# 1. PoolWarmupThread expõe API esperada
# ──────────────────────────────────────────────────────────────────────────
def test_pool_warmup_thread_has_signals() -> None:
    """``PoolWarmupThread`` deve expor ``warmup_done`` e ``warmup_error``.

    Esses sinais são consumidos por ``SimulationPage`` para atualizar o
    label ``lbl_warmup_status`` (cinza→amarelo→verde). Remover ou renomear
    quebra o ciclo de vida visual da v2.18.
    """
    from geosteering_ai.simulation.tests.sm_workers import PoolWarmupThread

    assert hasattr(
        PoolWarmupThread, "warmup_done"
    ), "PoolWarmupThread.warmup_done ausente — GUI v2.18 não atualiza label"
    assert hasattr(PoolWarmupThread, "warmup_error"), (
        "PoolWarmupThread.warmup_error ausente — falhas no pre-warm "
        "ficariam silenciosas"
    )


# ──────────────────────────────────────────────────────────────────────────
# 2. PoolWarmupThread aceita parâmetros de configuração
# ──────────────────────────────────────────────────────────────────────────
def test_pool_warmup_thread_accepts_config_args() -> None:
    """``PoolWarmupThread.__init__`` deve aceitar ``(n_workers, n_threads, hankel_filter)``.

    Mudanças futuras na assinatura quebram o ``_start_background_warmup``
    da ``SimulationPage`` — este teste detecta antes que chegue à GUI.
    """
    # Necessita um QApplication ativo para construir QThread.
    from geosteering_ai.simulation.tests.sm_qt_compat import QtWidgets
    from geosteering_ai.simulation.tests.sm_workers import PoolWarmupThread

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None  # silencia ruff/mypy

    thread = PoolWarmupThread(n_workers=2, n_threads=2, hankel_filter="werthmuller_201pt")
    assert thread._n_workers == 2
    assert thread._n_threads == 2
    assert thread._hankel_filter == "werthmuller_201pt"


# ──────────────────────────────────────────────────────────────────────────
# 3. SimulationPage instancia label de warmup + thread holder
# ──────────────────────────────────────────────────────────────────────────
def test_simulator_page_has_warmup_attributes() -> None:
    """``SimulatorPage`` deve ter ``lbl_warmup_status`` e ``_warmup_thread``.

    Estes atributos foram introduzidos em v2.18 e são consumidos pelos
    smoke tests T31/T32. Removê-los regride a UX de pre-warm.
    """
    from geosteering_ai.simulation.tests.simulation_manager import SimulatorPage
    from geosteering_ai.simulation.tests.sm_qt_compat import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    page = SimulatorPage()
    assert hasattr(
        page, "lbl_warmup_status"
    ), "SimulatorPage.lbl_warmup_status ausente — label de warmup removido"
    assert hasattr(
        page, "_warmup_thread"
    ), "SimulatorPage._warmup_thread ausente — atributo holder removido"


# ──────────────────────────────────────────────────────────────────────────
# 4. Verificação de t0_sim no SimulationThread (v2.18)
# ──────────────────────────────────────────────────────────────────────────
def test_simulation_thread_uses_t0_sim_pattern() -> None:
    """A função ``SimulationThread.run`` deve referenciar ``t0_sim`` (v2.18).

    Garantia simples por inspeção textual de que o fix de medição
    (timer pós-pool-warm) não foi acidentalmente revertido.
    """
    import inspect

    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    src = inspect.getsource(SimulationThread.run)
    assert "t0_sim" in src, (
        "SimulationThread.run não menciona 't0_sim' — fix de medição da "
        "v2.18 pode ter sido revertido. Throughput voltará a reportar "
        "~38k mod/h em cold start."
    )


# ──────────────────────────────────────────────────────────────────────────
# 5. ParametersPage expõe controle de seed (v2.19) — proteção complementar
# ──────────────────────────────────────────────────────────────────────────
def test_parameters_page_has_seed_widgets() -> None:
    """``ParametersPage`` deve ter ``chk_random_seed`` + ``spn_fixed_seed``.

    Sprint 19.1 (v2.19): UI de semente PRNG. Removê-los regride o bug
    funcional (modelos sempre idênticos) que esta versão corrigiu.
    """
    from geosteering_ai.simulation.tests.simulation_manager import ParametersPage
    from geosteering_ai.simulation.tests.sm_qt_compat import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    page = ParametersPage()
    assert hasattr(page, "chk_random_seed")
    assert hasattr(page, "spn_fixed_seed")
    assert hasattr(page, "get_rng_seed")
    # Default v2.19 = aleatório
    assert page.chk_random_seed.isChecked() is True
    assert page.get_rng_seed() is None
