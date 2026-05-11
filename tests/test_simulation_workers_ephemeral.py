# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_workers_ephemeral.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager v2.29 (Back to Basics)                  ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-11                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest                                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garante que a v2.29 (Sprint Back to Basics) NÃO regrediu para a       ║
# ║    arquitetura problemática v2.18–v2.28 (pool persistente +              ║
# ║    PoolWarmupThread + _WORKER_INITIALIZED + t0_sim + NOOPs).             ║
# ║                                                                           ║
# ║    A arquitetura v2.29 (modelo `old_geosteering_ai/`):                   ║
# ║      • Pool EFÊMERO (`with ProcessPoolExecutor`)                         ║
# ║      • Warmup INLINE em `run_numba_chunk` com dados reais               ║
# ║      • `t0 = time.perf_counter()` único, antes do pool                  ║
# ║      • Sem inicializador, sem warmup background, sem flag global        ║
# ║                                                                           ║
# ║  COBERTURA                                                                ║
# ║    1. sm_workers.py NÃO contém símbolos da arquitetura persistente      ║
# ║    2. SimulationThread.run() usa `with ProcessPoolExecutor` (ephemeral) ║
# ║    3. run_numba_chunk tem warmup inline incondicional                    ║
# ║    4. SimulationThread NÃO usa `t0_sim` (medição volta a `t0` único)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de não-regressão da arquitetura ephemeral do Simulation Manager v2.29."""

from __future__ import annotations

import inspect
import os

# Habilita Qt headless ANTES de importar PyQt — evita "could not connect
# to X server" em CI/sistemas sem display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ──────────────────────────────────────────────────────────────────────────
# 1. sm_workers.py NÃO contém símbolos da arquitetura persistente removida
# ──────────────────────────────────────────────────────────────────────────
def test_sm_workers_has_no_persistent_pool_symbols() -> None:
    """``sm_workers.py`` v2.29 NÃO deve conter símbolos da infra v2.18–v2.28.

    Removidos em v2.29 (Back to Basics) porque causavam regressão de
    throughput (75–107k vs >150k esperado), warmup visível ~33s, e hang
    do Python ao fechar a GUI antes da simulação terminar.

    Símbolos obsoletos:
      • ``_PERSISTENT_POOL`` (pool persistente global)
      • ``_PERSISTENT_POOL_CONFIG`` (config rastreada)
      • ``_acquire_numba_pool`` (factory do pool persistente)
      • ``release_numba_pool`` (cleanup explícito)
      • ``PoolWarmupThread`` (warmup background na GUI)
      • ``_WORKER_INITIALIZED`` (flag global de warmup secundário)
      • ``_run_numba_warmup_task`` (task de warmup sintético)
      • ``_numba_init_worker`` (inicializador custom)
      • ``_noop`` (função de sincronização NOOP)
    """
    from geosteering_ai.simulation.tests import sm_workers

    # Verifica DEFINIÇÕES (classes/funções/globais), não menções em comentários
    # — comentários históricos como "PoolWarmupThread removido em v2.29" são OK.
    forbidden_defs = [
        ("class PoolWarmupThread", "PoolWarmupThread"),
        ("def _acquire_numba_pool", "_acquire_numba_pool"),
        ("def release_numba_pool", "release_numba_pool"),
        ("def _run_numba_warmup_task", "_run_numba_warmup_task"),
        ("def _numba_init_worker", "_numba_init_worker"),
        ("def _noop", "_noop"),
        ("_PERSISTENT_POOL:", "_PERSISTENT_POOL"),
        ("_PERSISTENT_POOL_CONFIG:", "_PERSISTENT_POOL_CONFIG"),
        ("_WORKER_INITIALIZED:", "_WORKER_INITIALIZED"),
    ]
    src = inspect.getsource(sm_workers)
    found = [name for marker, name in forbidden_defs if marker in src]
    assert not found, (
        f"sm_workers.py DEFINE símbolos obsoletos da arquitetura v2.18–v2.28: "
        f"{found}. v2.29 (Back to Basics) removeu toda essa infraestrutura."
    )

    # Também verifica que esses símbolos NÃO estão em __all__
    forbidden_exports = {
        "PoolWarmupThread",
        "_acquire_numba_pool",
        "release_numba_pool",
        "_run_numba_warmup_task",
        "_numba_init_worker",
        "_noop",
        "_PERSISTENT_POOL",
        "_WORKER_INITIALIZED",
    }
    exports = set(getattr(sm_workers, "__all__", []))
    leaked = forbidden_exports & exports
    assert not leaked, f"sm_workers.__all__ exporta símbolos obsoletos: {leaked}"


# ──────────────────────────────────────────────────────────────────────────
# 2. SimulationThread.run() usa `with ProcessPoolExecutor` (ephemeral)
# ──────────────────────────────────────────────────────────────────────────
def test_simulation_thread_uses_context_manager_pool() -> None:
    """``SimulationThread.run()`` v2.29 deve usar pool efêmero via ``with``.

    Bug v2.18–v2.28: pool persistente global causava race conditions, pool
    recriado se config diferisse entre warmup e simulação, e hang do Python
    no shutdown porque workers em JIT compilation não respondiam a
    ``cancel_futures=True``.

    v2.29: ``with ProcessPoolExecutor(max_workers=n_workers) as pool:``
    garante shutdown limpo via context manager `__exit__`.
    """
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    src = inspect.getsource(SimulationThread.run)
    assert "with ProcessPoolExecutor" in src, (
        "SimulationThread.run() NÃO usa `with ProcessPoolExecutor` — "
        "pode estar usando pool persistente (regressão v2.18–v2.28)."
    )
    assert "_acquire_numba_pool" not in src, (
        "SimulationThread.run() ainda referencia `_acquire_numba_pool` — "
        "deveria usar `with ProcessPoolExecutor` ephemeral."
    )


# ──────────────────────────────────────────────────────────────────────────
# 3. run_numba_chunk tem warmup inline incondicional
# ──────────────────────────────────────────────────────────────────────────
def test_run_numba_chunk_has_inline_warmup() -> None:
    """``run_numba_chunk`` v2.29 deve ter warmup INLINE com dados reais.

    Bug v2.18–v2.28: warmup secundário era CONDICIONAL via
    ``if chunk and not _WORKER_INITIALIZED``. Flag ``_WORKER_INITIALIZED``
    podia ficar inconsistente entre pool persistente e workers, causando
    compilação JIT inline APÓS ``t0_sim`` (drag de throughput).

    v2.29: warmup é INCONDICIONAL (`if chunk:`) e usa dados REAIS de
    `chunk[0]` (cobre TODOS os paths Numba — anisotropia, dip≠0, etc.).
    Tempo de warmup é EXCLUÍDO do `elapsed` retornado (t0 é setado APÓS).
    """
    from geosteering_ai.simulation.tests.sm_workers import run_numba_chunk

    src = inspect.getsource(run_numba_chunk)
    assert "warmup" in src.lower(), (
        "run_numba_chunk NÃO tem comentário/código relacionado a warmup — "
        "warmup inline (modelo OLD) pode ter sido removido."
    )
    assert "_WORKER_INITIALIZED" not in src, (
        "run_numba_chunk ainda referencia _WORKER_INITIALIZED — flag global "
        "obsoleta da arquitetura v2.18–v2.28 deve ter sido removida em v2.29."
    )
    assert "chunk[0]" in src, (
        "run_numba_chunk não usa `chunk[0]` para warmup — fix v2.29 requer "
        "dados REAIS do primeiro modelo do chunk para cobrir todos paths JIT."
    )


# ──────────────────────────────────────────────────────────────────────────
# 4. SimulationThread NÃO usa `t0_sim` (medição volta a `t0` único)
# ──────────────────────────────────────────────────────────────────────────
def test_simulation_thread_no_t0_sim() -> None:
    """``SimulationThread.run()`` v2.29 NÃO deve usar ``t0_sim``.

    Bug v2.18–v2.28: ``t0_sim`` era setado APÓS NOOPs sincronizarem o pool,
    visando isolar tempo de warmup do tempo de simulação. Mas se o warmup
    em Workers 1..N ocorria APÓS ``t0_sim`` (por flag mal sincronizada),
    o tempo de JIT era contado como simulação → throughput baixo.

    v2.29: volta a ``t0`` único setado ANTES do pool/warmup. Tempo total
    INCLUI warmup honestamente. Em simulações grandes (N>=1000), o overhead
    dilui-se. Mensagem ao usuário sobre warmup esperado é explícita.
    """
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    src = inspect.getsource(SimulationThread.run)
    assert "t0_sim" not in src, (
        "SimulationThread.run() ainda referencia `t0_sim` — esquema de "
        "sincronização via NOOPs (v2.18–v2.28) deve ter sido removido em v2.29."
    )
    # Garante que t0 ainda existe (medição não foi removida totalmente)
    assert "t0 = time.perf_counter()" in src, (
        "SimulationThread.run() não tem `t0 = time.perf_counter()` — "
        "medição de tempo total foi removida acidentalmente."
    )


# ──────────────────────────────────────────────────────────────────────────
# 5. NumbaPrimer permanece disponível (cache em disco no startup)
# ──────────────────────────────────────────────────────────────────────────
def test_numba_primer_available() -> None:
    """``NumbaPrimer`` v2.29 deve permanecer disponível.

    NumbaPrimer popula o cache JIT em disco (.nbi/.nbc) no startup da GUI
    rodando ``simulate_multi`` 1× no processo principal. Workers spawned
    pelo pool efêmero leem esse cache em ~1–2 s (vs ~30 s cold compile).

    Esta é a única infra de "warmup" que sobreviveu à v2.29 — porque ela
    NÃO interage com o pool persistente (que foi removido).
    """
    from geosteering_ai.simulation.tests.sm_workers import NumbaPrimer

    assert hasattr(
        NumbaPrimer, "primer_done"
    ), "NumbaPrimer.primer_done ausente — signal de conclusão removido."
    assert hasattr(
        NumbaPrimer, "primer_failed"
    ), "NumbaPrimer.primer_failed ausente — signal de erro removido."


# ──────────────────────────────────────────────────────────────────────────
# 6. SimulationThread preserva pause/cancel cooperativo (v2.11)
# ──────────────────────────────────────────────────────────────────────────
def test_simulation_thread_has_pause_cancel() -> None:
    """``SimulationThread`` v2.29 deve preservar API de pause/cancel (v2.11).

    O refactor v2.29 removeu infra de warmup mas preservou recursos não-
    conflitantes: pause/resume/cancel cooperativo.
    """
    from geosteering_ai.simulation.tests.sm_workers import SimulationThread

    for method in ("request_pause", "request_resume", "request_cancel", "request_stop"):
        assert hasattr(SimulationThread, method), (
            f"SimulationThread.{method} ausente — v2.29 deveria preservar "
            f"a API de controle cooperativo da Sprint v2.11."
        )

    for signal in ("paused", "resumed", "cancelled"):
        assert hasattr(SimulationThread, signal), (
            f"SimulationThread.{signal} ausente — v2.29 deveria preservar "
            f"os signals Qt de controle cooperativo da Sprint v2.11."
        )
