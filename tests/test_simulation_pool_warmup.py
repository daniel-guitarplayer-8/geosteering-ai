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

    thread = PoolWarmupThread(
        n_workers=2, n_threads=2, hankel_filter="werthmuller_201pt"
    )
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


# ──────────────────────────────────────────────────────────────────────────
# 6. Regressão: _run_numba_warmup_task usa shapes corretos (v2.26 fix)
# ──────────────────────────────────────────────────────────────────────────
def test_run_numba_warmup_task_has_correct_shapes() -> None:
    """``_run_numba_warmup_task`` deve usar shapes que passem em ``_validate_multi_inputs``.

    Bug v2.10–v2.24: warmup usava ``esp=[5.0, 5.0]`` com ``rho_h`` de 3
    camadas. ``_validate_multi_inputs`` exige ``esp.shape[0]==n-2==1``, então
    a validação levantava ``ValueError``, capturado por ``except Exception:
    pass``. ``_WORKER_INITIALIZED`` virava True mesmo assim, e
    ``run_numba_chunk`` também pulava seu warmup secundário. Resultado:
    primeira simulação real pagava o JIT cold-start completo (50–70k vs.
    150–175k mod/h steady-state).

    v2.25: fix de shape no inicializador — mas warmup no inicializador
    causava travamento ao fechar o app (não cancelável, 35–38 s JIT frio).

    v2.26: warmup movido para ``_run_numba_warmup_task`` (Future cancelável).
    Este teste garante que a função mantém shapes corretos na nova localização.
    """
    import inspect

    from geosteering_ai.simulation.tests.sm_workers import _run_numba_warmup_task

    src = inspect.getsource(_run_numba_warmup_task)
    # Extrai apenas o corpo do código (após a docstring) para evitar
    # falsos positivos em menções históricas dentro da docstring.
    _first = src.index('"""')
    _close = src.index('"""', _first + 3)
    code = src[_close + 3 :]
    # rho com 10 camadas → esp deve ter 8 elementos (n-2).
    assert "_rho" in code and "_esp" in code, (
        "_run_numba_warmup_task não usa variáveis _rho/_esp — fix de shape "
        "v2.26 pode ter sido revertido. Warmup voltará a falhar silenciosamente "
        "via ValueError em _validate_multi_inputs."
    )
    assert "np.full(8" in code or "_np.full(8" in code, (
        "_run_numba_warmup_task não cria esp com 8 elementos (n-2=8 para 10 "
        "camadas). Shape de esp deve casar com rho — caso contrário "
        "_validate_multi_inputs levanta ValueError e warmup é perdido."
    )
    # Posições suficientes para ativar JIT (qualquer tamanho ≥ 1).
    assert "linspace" in code, (
        "_run_numba_warmup_task não usa np.linspace para positions_z — "
        "verificar que o warmup ainda cria posições representativas."
    )
    # Confirma que o código NÃO usa 600 posições (overhead desnecessário v2.25).
    assert "600" not in code, (
        "_run_numba_warmup_task ainda usa 600 posições (overhead v2.25). "
        "v2.26 usa 50 posições — JIT ativa na 1ª chamada, tamanho não importa."
    )


def test_run_numba_warmup_task_smoke() -> None:
    """Chama ``_run_numba_warmup_task`` direto no processo principal — não deve raise.

    Smoke test em-process (sem ProcessPoolExecutor): garante que os shapes
    do warmup passam por ``_validate_multi_inputs`` e que o ciclo completo
    de ``simulate_multi`` executa. Como ``_WORKER_INITIALIZED`` é global,
    resetamos antes para forçar nova execução, e restauramos depois para
    não interferir com outros testes.
    """
    from geosteering_ai.simulation.tests import sm_workers

    prev = sm_workers._WORKER_INITIALIZED
    sm_workers._WORKER_INITIALIZED = False
    try:
        result = sm_workers._run_numba_warmup_task(
            n_threads=1, hankel_filter="werthmuller_201pt"
        )
        assert result is True, (
            "_run_numba_warmup_task retornou False — warmup levantou exceção "
            "capturada pelo except. Verifique shapes (esp.shape[0]==n-2) e "
            "disponibilidade de Numba/filtro Hankel."
        )
        assert sm_workers._WORKER_INITIALIZED, (
            "_run_numba_warmup_task retornou True mas _WORKER_INITIALIZED ficou "
            "False — inconsistência interna na função."
        )
    finally:
        sm_workers._WORKER_INITIALIZED = prev


def test_numba_init_worker_is_lightweight() -> None:
    """``_numba_init_worker`` v2.26 NÃO deve conter ``simulate_multi``.

    v2.26: warmup movido do inicializador para ``_run_numba_warmup_task``
    (Future). O inicializador deve ser leve (< 100 ms) para evitar que o
    handler ``atexit`` do ``ProcessPoolExecutor`` trave ao fechar o app quando
    workers estão em compilação JIT Numba (35–38 s no cold-start).
    """
    import inspect

    from geosteering_ai.simulation.tests.sm_workers import _numba_init_worker

    src = inspect.getsource(_numba_init_worker)
    # Extrai apenas o corpo do código (após a docstring) para não confundir
    # menções históricas na docstring com chamadas reais.
    _first = src.index('"""')
    _close = src.index('"""', _first + 3)
    code = src[_close + 3 :]
    assert "simulate_multi" not in code, (
        "_numba_init_worker ainda chama simulate_multi — warmup pesado no "
        "inicializador causa travamento ao fechar o app (handler atexit do "
        "ProcessPoolExecutor chama shutdown(wait=True) em todos os pools, "
        "esperando indefinidamente por workers presos em JIT de 35+ s)."
    )
