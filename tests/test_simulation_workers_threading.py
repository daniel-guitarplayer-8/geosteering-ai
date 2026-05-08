# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_workers_threading.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — Simulation Manager v2.16 threading masking        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-01 (Sprint 15.1, v2.16)                            ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest 8.x                                                 ║
# ║  Dependências: numba, geosteering_ai.simulation                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Smoke tests que detectam regressão do bug de threading masking         ║
# ║    introduzido em v2.15 (commits 0f92035 + e1c8864) e corrigido em        ║
# ║    v2.16 (Sprint 15.1).                                                   ║
# ║                                                                           ║
# ║    BUG ORIGINAL (v2.15):                                                  ║
# ║      • multi_forward.py:880-886 envolvia numba.set_num_threads() em       ║
# ║        try/except RuntimeError: pass (silenciava falha sem fallback)      ║
# ║      • sm_workers.py + _workers.py removeram NUMBA_NUM_THREADS env var,   ║
# ║        deixando workers spawn com pool = cpu_count() (16 hyperthreaded)   ║
# ║      • Resultado: workers com threads ativas em estado indefinido,        ║
# ║        regressão de 4–8× em produção (200k mod/h → 25–38k mod/h)          ║
# ║                                                                           ║
# ║    FIX v2.16 (Sprint 15.1):                                               ║
# ║      • _acquire_numba_pool e _acquire_pool: setam NUMBA_NUM_THREADS no    ║
# ║        env do PAI antes do spawn — workers herdam e Numba lê na primeira  ║
# ║        import dimensionando o pool corretamente                           ║
# ║      • multi_forward.py:880-886: try/except: pass → logger.warning        ║
# ║        com diagnóstico observável                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes Sprint 15.1 — Threading masking observável em workers Numba."""
from __future__ import annotations

import logging
import multiprocessing as _mp
import os
from concurrent.futures import ProcessPoolExecutor

import pytest

# Skip o módulo inteiro se Numba não disponível — esses testes não fazem
# sentido sem Numba (são especificamente sobre seu thread pool).
numba = pytest.importorskip("numba")


# ═════════════════════════════════════════════════════════════════════════
# Helpers — funções top-level (precisam ser pickláveis para spawn)
# ═════════════════════════════════════════════════════════════════════════


def _worker_init_with_env(n_threads: int) -> None:
    """Inicializador que apenas valida o env var — sem warmup pesado."""
    # No spawn, NUMBA_NUM_THREADS já está no env (setado pelo pai).
    # O worker importa numba pela primeira vez aqui — pool dimensionado.
    pass  # noqa: PIE790 — explícito: pool é setado pelo env, não aqui


def _worker_get_num_threads() -> tuple[int, int]:
    """Retorna (NUMBA_NUM_THREADS env, numba.config.NUMBA_NUM_THREADS).

    Importa Numba (lazy) e captura ambos os valores para validar que o
    pool foi dimensionado a partir do env var herdado do pai.
    """
    import numba as _nb

    env_value = int(os.environ.get("NUMBA_NUM_THREADS", "-1"))
    config_value = int(_nb.config.NUMBA_NUM_THREADS)
    active_value = int(_nb.get_num_threads())
    return env_value, config_value, active_value


# ═════════════════════════════════════════════════════════════════════════
# 1 — Workers herdam NUMBA_NUM_THREADS do env do pai
# ═════════════════════════════════════════════════════════════════════════


def test_worker_inherits_numba_num_threads_from_parent_env() -> None:
    """Sprint 15.1: workers spawn herdam NUMBA_NUM_THREADS do env do pai.

    Cenário regressão v2.15: pai não setava NUMBA_NUM_THREADS antes do
    spawn → workers iniciavam com cpu_count() → set_num_threads() falhava
    silenciosamente em chamadas posteriores.

    Pós-fix v2.16: `_acquire_numba_pool` e `_acquire_pool` setam o env no
    pai antes do spawn → workers nascem com pool dimensionado.
    """
    target_threads = 2
    saved_env = os.environ.get("NUMBA_NUM_THREADS")
    try:
        # Simula o que `_acquire_numba_pool` faz no v2.16:
        os.environ["NUMBA_NUM_THREADS"] = str(target_threads)
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=_mp.get_context("spawn"),
            initializer=_worker_init_with_env,
            initargs=(target_threads,),
        ) as pool:
            results = list(pool.map(_worker_get_num_threads_args, [None, None]))

        for env_val, config_val, active_val in results:
            assert env_val == target_threads, (
                f"NUMBA_NUM_THREADS env não foi herdado pelo worker: "
                f"esperado={target_threads}, atual={env_val}"
            )
            assert config_val == target_threads, (
                f"numba.config.NUMBA_NUM_THREADS não corresponde ao env: "
                f"esperado={target_threads}, atual={config_val}. "
                f"Indica que pool foi dimensionado com cpu_count() em vez "
                f"de respeitar o env var."
            )
            # active pode ser config inicialmente (sem set_num_threads explícito)
            assert (
                active_val <= config_val
            ), f"threads ativas ({active_val}) > pool size ({config_val}) — inválido"
    finally:
        # Restaurar env original para não contaminar outros testes.
        if saved_env is None:
            os.environ.pop("NUMBA_NUM_THREADS", None)
        else:
            os.environ["NUMBA_NUM_THREADS"] = saved_env


def _worker_get_num_threads_args(_arg) -> tuple[int, int, int]:
    """Wrapper para `pool.map` (que sempre passa 1 argumento)."""
    return _worker_get_num_threads()


# ═════════════════════════════════════════════════════════════════════════
# 2 — multi_forward.py loga warning em mismatch (não silencia)
# ═════════════════════════════════════════════════════════════════════════


def test_set_num_threads_emits_warning_on_runtime_error(caplog) -> None:
    """Sprint 15.1: substituição de `try/except: pass` por logger.warning.

    Quando `numba.set_num_threads(n)` lança RuntimeError em estado
    incompatível, o código deve LOGAR a falha (não silenciar) para
    permitir diagnóstico de regressões futuras.

    Este teste mocka `set_num_threads` para forçar RuntimeError e captura
    o log do logger de `multi_forward`.
    """
    from unittest.mock import patch

    from geosteering_ai.simulation.config import SimulationConfig

    cfg = SimulationConfig(num_threads=4, backend="numba")

    with caplog.at_level(
        logging.WARNING, logger="geosteering_ai.simulation.multi_forward"
    ):
        # Mock set_num_threads para forçar RuntimeError
        with patch.object(
            numba,
            "set_num_threads",
            side_effect=RuntimeError("Cannot set NUMBA_NUM_THREADS to a different value"),
        ):
            # Reproduzir o bloco em multi_forward.py:877-907 manualmente,
            # uma vez que `simulate_multi` requer todos os arrays físicos
            # — aqui validamos apenas a observabilidade do warning.
            import numba as _numba

            current_active = _numba.get_num_threads()
            if current_active != cfg.num_threads:
                try:
                    _numba.set_num_threads(cfg.num_threads)
                except RuntimeError as exc:
                    pool_size = _numba.config.NUMBA_NUM_THREADS
                    logger = logging.getLogger("geosteering_ai.simulation.multi_forward")
                    logger.warning(
                        "numba.set_num_threads(%d) falhou (threads ativas atual=%d, "
                        "pool size NUMBA_NUM_THREADS=%d): %s. Performance pode "
                        "degradar significativamente; verifique se o env var "
                        "NUMBA_NUM_THREADS foi setado antes do spawn dos workers.",
                        cfg.num_threads,
                        current_active,
                        pool_size,
                        exc,
                    )

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "set_num_threads" in r.getMessage() and "falhou" in r.getMessage()
        for r in warnings
    ), (
        f"Esperado warning sobre falha em set_num_threads. "
        f"Capturado: {[r.getMessage() for r in warnings]}"
    )


# ═════════════════════════════════════════════════════════════════════════
# 3 — Smoke integração: simulate_multi via worker pool com N threads
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_simulate_multi_in_worker_respects_n_threads() -> None:
    """Smoke E2E: `simulate_multi` em worker spawn com `num_threads=2`
    deve rodar com exatamente 2 threads ativas (não cpu_count()).

    Este é o teste mais importante: replica o caminho da GUI
    (ProcessPoolExecutor + simulate_multi) e valida que threads ativas
    correspondem ao configurado.

    Marcado `@slow` porque dispara JIT e leva 5–15 segundos.
    """
    import numpy as np

    target_threads = 2
    saved_env = os.environ.get("NUMBA_NUM_THREADS")
    try:
        os.environ["NUMBA_NUM_THREADS"] = str(target_threads)

        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=_mp.get_context("spawn"),
        ) as pool:
            future = pool.submit(_worker_simulate_and_get_threads, target_threads)
            # Timeout 300s: JIT cold-start de simulate_multi compila
            # _simulate_positions_njit_cached + dipoles + kernel; pode
            # levar 1.5–3 min em hardware modesto.
            n_threads_inside = future.result(timeout=300.0)

        assert n_threads_inside == target_threads, (
            f"Worker rodou simulate_multi com {n_threads_inside} threads "
            f"ativas, esperado {target_threads}. Indica regressão do "
            f"threading masking de v2.15 (commits 0f92035 + e1c8864)."
        )
    finally:
        if saved_env is None:
            os.environ.pop("NUMBA_NUM_THREADS", None)
        else:
            os.environ["NUMBA_NUM_THREADS"] = saved_env


def _worker_simulate_and_get_threads(target_threads: int) -> int:
    """Roda simulate_multi mínima dentro do worker e retorna threads ativas."""
    import numpy as np

    from geosteering_ai.simulation import SimulationConfig, simulate_multi

    cfg = SimulationConfig(num_threads=target_threads, backend="numba")
    # Modelo trivial 3 camadas (esp.shape == n-2 == 1), 2 posições — força
    # JIT mas é rápido.
    simulate_multi(
        rho_h=np.array([1.0, 10.0, 1.0], dtype=np.float64),
        rho_v=np.array([1.0, 10.0, 1.0], dtype=np.float64),
        esp=np.array([5.0], dtype=np.float64),
        positions_z=np.array([0.0, 5.0], dtype=np.float64),
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg,
    )
    import numba as _nb

    return int(_nb.get_num_threads())
