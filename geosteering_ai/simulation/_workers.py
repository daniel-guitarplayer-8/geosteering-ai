# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_workers.py                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Workers Nativos (v2.12 / Sprint 12.1)  ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-30 (Sprint 12.1)                                   ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : stdlib (concurrent.futures, multiprocessing, threading)   ║
# ║  Dependências: numpy, geosteering_ai.simulation.multi_forward (lazy)     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Núcleo de paralelismo inter-modelo do simulador Python. Migra para    ║
# ║    o core a infraestrutura de pool de workers que antes existia apenas   ║
# ║    na camada UI (`tests/sm_workers.py`). Após v2.12 a API pública        ║
# ║    `simulate_multi(models=[...], n_workers=N)` despacha N modelos para  ║
# ║    um ProcessPoolExecutor com warmup JIT cooperativo, agregando         ║
# ║    `H_stack` (n_models, nTR, nAngles, n_pos, nf, 9) preservando         ║
# ║    a ordem original.                                                    ║
# ║                                                                           ║
# ║  4 MODOS DE EXECUÇÃO (relatório §3, 2026-04-30)                          ║
# ║    ┌────────┬──────────┬────────┬─────────────────────────────────────┐  ║
# ║    │  Modo  │ workers  │threads │  Quando usar                        │  ║
# ║    ├────────┼──────────┼────────┼─────────────────────────────────────┤  ║
# ║    │   A    │    1     │   1    │  Debug / 1 modelo, n_pos < 50      │  ║
# ║    │   B    │    1     │   N    │  1 modelo, n_pos ≥ 100             │  ║
# ║    │   C    │    M     │   1    │  Batch grande, n_pos baixo         │  ║
# ║    │   D    │    M     │   K    │  ★ DEFAULT PRODUÇÃO (30k modelos) │  ║
# ║    └────────┴──────────┴────────┴─────────────────────────────────────┘  ║
# ║    Anti-oversubscription: quando `threads_per_worker is None`, usa      ║
# ║    `eff_threads = max(1, cpu_count // n_workers)` — garante              ║
# ║    `n_workers × eff_threads <= cpu_count`.                              ║
# ║                                                                           ║
# ║  ARQUITETURA                                                              ║
# ║    ┌──────────────────────────────────────────────────────────────────┐ ║
# ║    │  simulate_multi(models=[m1, m2, ..., mN], n_workers=4, ...)     │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  _resolve_effective_threads(4, None)  → 2  (em CPU 8-core)     │ ║
# ║    │  _detect_mode(4, 2)                    → "D"                   │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  _split_models_uniform(models, 4)                              │ ║
# ║    │     → [[(0,m0),(1,m1)...], [(N/4,m...)...], ...]              │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  _acquire_pool(4, 2, 'werthmuller_201pt')                      │ ║
# ║    │     → ProcessPoolExecutor singleton                            │ ║
# ║    │     → cada worker chama _simulate_worker_init no spawn         │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  futures = [pool.submit(_run_simulate_chunk, chunk, kwargs)    │ ║
# ║    │             for chunk in chunks]                               │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  for fut in as_completed(futures):                              │ ║
# ║    │     idxs, H_chunk = fut.result()                                │ ║
# ║    │     # Reagrega preservando ordem original                       │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  MultiSimulationResultBatch                                     │ ║
# ║    │    .H_stack: (n_models, nTR, nAngles, n_pos, nf, 9)            │ ║
# ║    │    .mode: "D"                                                   │ ║
# ║    │    .throughput_mod_per_h: ~960                                  │ ║
# ║    └──────────────────────────────────────────────────────────────────┘ ║
# ║                                                                           ║
# ║  THREAD-SAFETY                                                            ║
# ║    `_acquire_pool` usa `threading.Lock` para proteger o singleton        ║
# ║    `_PERSISTENT_POOL`. A função pode ser chamada concorrentemente de    ║
# ║    diferentes threads (típico em UI Qt) sem race conditions.            ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • docs/reports/v2.11_simulador_python_analise_paralelismo_2026-04-30.md║
# ║      (§3 — Workers Nativos; §6 — Plano de Implementação v2.12)         ║
# ║    • geosteering_ai/simulation/tests/sm_workers.py:73-139               ║
# ║      (origem do warmup JIT + pool persistente, agora migrado)          ║
# ║    • docs/reports/v2.12_workers_nativos_2026-04-30.md (relatório v2.12)║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Workers nativos para `simulate_multi` (Sprint 12.1, v2.12).

Este módulo expõe o núcleo de paralelismo inter-modelo do simulador Python
otimizado. A estratégia segue rigorosamente o relatório técnico
`docs/reports/v2.11_simulador_python_analise_paralelismo_2026-04-30.md`,
adicionando 3 kwargs em `simulate_multi`:

  • `models: Optional[List[dict]]` — batch de modelos para paralelizar.
  • `n_workers: Optional[int]`     — número de processos do pool.
  • `threads_per_worker: Optional[int]` — threads Numba por worker (auto).

Quando `models is None`, `simulate_multi` mantém o comportamento atual
(single-modelo in-place). Quando `models` é fornecido, este módulo
orquestra um `ProcessPoolExecutor` com warmup JIT compartilhado.

Example:
    Batch 30k modelos no Modo D (4 workers × 2 threads em CPU 8-core)::

        >>> from geosteering_ai.simulation import simulate_multi
        >>> models = [{"rho_h": ..., "rho_v": ..., "esp": ...} for _ in range(30000)]
        >>> result = simulate_multi(
        ...     models=models,
        ...     positions_z=np.linspace(-2, 7, 600),
        ...     tr_spacings_m=[1.0],
        ...     dip_degs=[0.0],
        ...     frequencies_hz=[20000.0],
        ...     n_workers=4,
        ...     threads_per_worker=2,  # ou None para auto
        ... )
        >>> result.H_stack.shape  # (n_models, nTR, nAngles, n_pos, nf, 9)
        (30000, 1, 1, 600, 1, 9)
        >>> result.mode
        'D'
"""
from __future__ import annotations

import multiprocessing as _mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Singleton: pool persistente reutilizado entre chamadas consecutivas
# ──────────────────────────────────────────────────────────────────────────────
# Estratégia idêntica a `tests/sm_workers.py` mas no core: o ProcessPoolExecutor
# é caro de criar (3-10s para spawn de N processos + import Numba + warmup JIT).
# Reusá-lo entre chamadas com a MESMA configuração `(n_workers, n_threads,
# hankel_filter)` reduz o custo de chamadas subsequentes a ~0.
#
# `_LOCK` protege o singleton contra race conditions quando `_acquire_pool`
# é invocado de múltiplas threads (típico em UI Qt onde `closeEvent` corre na
# main thread enquanto `SimulationThread` corre em paralelo).
_PERSISTENT_POOL: Optional[ProcessPoolExecutor] = None
_PERSISTENT_POOL_CONFIG: Optional[Tuple[int, int, str]] = None
_LOCK = threading.Lock()

# Flag interna: True dentro de um worker spawned após o inicializador rodar.
# False no processo principal — impede warmup redundante.
_WORKER_INITIALIZED: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Detecção de topologia de CPU (v2.17, Sprint 17.1)
# ──────────────────────────────────────────────────────────────────────────────
# Em CPUs com Hyperthreading (HT) ou SMT, ``os.cpu_count()`` retorna o número
# de threads lógicas (cores físicos × 2 em HT/SMT). Calcular paralelismo com
# base em threads lógicas resulta em **oversubscription** quando
# ``n_workers × threads_per_worker = cpu_count()``: cada worker tenta usar
# todas as threads lógicas disponíveis, gerando contenção severa por:
#
#   1. Cache L1/L2 compartilhado entre hyperthreads do mesmo core físico;
#   2. FPU/ALU compartilhada entre hyperthreads;
#   3. Context switch overhead em scheduler do SO;
#   4. Numba TBB/OMP × N processos = 64+ threads competindo por 8 cores.
#
# Em workloads CPU-bound puros (como Numba JIT prange), ter 1 thread por
# core físico é tipicamente **30-50% mais rápido** que 1 thread por hyperthread.
# Portanto a recomendação correta é ``n_workers × threads_per_worker ≤
# physical_cores``, não ``≤ logical_cores``.

# Cache para evitar invocações repetidas do subprocess (lento em alguns SOs).
_CPU_TOPOLOGY_CACHE: Optional[Tuple[int, int, bool]] = None


def detect_cpu_topology() -> Tuple[int, int, bool]:
    """Detecta a topologia da CPU (lógicas × físicas × hyperthreading).

    Estratégia em camadas (primeira que funcionar):
      1. ``psutil.cpu_count(logical=False)`` — método portável e preciso
         se psutil estiver instalado (não é dependência obrigatória).
      2. macOS: ``sysctl -n hw.physicalcpu``
      3. Linux: contagem de ``physical id``/``core id`` únicos em
         ``/proc/cpuinfo``
      4. Windows: ``wmic cpu get NumberOfCores``
      5. Fallback heurístico: assume HT/SMT se ``cpu_count() >= 4`` →
         ``physical = logical // 2``. Em hardware sem HT/SMT (ex.: Apple
         Silicon performance cores, alguns Xeon Cascade Lake) este
         fallback subestima — o usuário pode override via spinbox.

    O resultado é cacheado em ``_CPU_TOPOLOGY_CACHE`` (módulo-global) já
    que a topologia da CPU não muda durante a execução do processo.

    Returns:
        Tupla ``(logical_cores, physical_cores, has_hyperthreading)``:
          - ``logical_cores``: ``os.cpu_count()`` (threads visíveis ao SO)
          - ``physical_cores``: cores físicos efetivos (>= 1)
          - ``has_hyperthreading``: ``True`` se ``logical > physical``

    Example:
        Em Mac Intel 8C/16T (Hyperthreading ativo)::

            >>> logical, phys, ht = detect_cpu_topology()
            >>> (logical, phys, ht)
            (16, 8, True)

        Em Apple Silicon M1 8-core (sem HT)::

            >>> detect_cpu_topology()
            (8, 8, False)

    Note:
        Esta função NUNCA falha — em caso de erro em todas as estratégias,
        retorna ``(logical, logical, False)`` (assume sem HT, mais
        conservador para evitar oversubscription assumida indevidamente).
    """
    global _CPU_TOPOLOGY_CACHE
    if _CPU_TOPOLOGY_CACHE is not None:
        return _CPU_TOPOLOGY_CACHE

    import platform
    import subprocess

    logical = os.cpu_count() or 1
    physical: Optional[int] = None

    # Estratégia 1: psutil (portável, preciso)
    try:
        import psutil  # type: ignore[import-not-found]

        phys = psutil.cpu_count(logical=False)
        if phys is not None and phys > 0:
            physical = int(phys)
    except ImportError:
        pass
    except Exception:
        pass

    # Estratégia 2: macOS sysctl
    if physical is None and platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"],
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            phys_int = int(out.decode("utf-8").strip())
            if phys_int > 0:
                physical = phys_int
        except Exception:
            pass

    # Estratégia 3: Linux /proc/cpuinfo
    if physical is None and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                cores: set = set()
                phys_id = core_id = None
                for line in fh:
                    if line.startswith("physical id"):
                        phys_id = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":")[1].strip()
                    elif line.strip() == "" and phys_id is not None:
                        cores.add((phys_id, core_id))
                        phys_id = core_id = None
                if cores:
                    physical = len(cores)
        except Exception:
            pass

    # Estratégia 4: Windows wmic
    if physical is None and platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "NumberOfCores"],
                stderr=subprocess.DEVNULL,
                timeout=3.0,
            )
            for line in out.decode("utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.isdigit() and int(line) > 0:
                    physical = (physical or 0) + int(line)
        except Exception:
            pass

    # Estratégia 5: Heurística fallback
    if physical is None or physical < 1:
        # Se logical >= 4, é provável que haja HT/SMT — assume metade.
        # Para logical < 4, provavelmente sem HT (CPUs antigas/ARM single-thread).
        physical = max(1, logical // 2) if logical >= 4 else logical

    physical = min(physical, logical)  # Sanidade: physical nunca > logical
    has_ht = logical > physical
    _CPU_TOPOLOGY_CACHE = (logical, physical, has_ht)
    return _CPU_TOPOLOGY_CACHE


def recommend_default_parallelism(
    n_models_hint: Optional[int] = None,
) -> Tuple[int, int]:
    """Recomenda ``(n_workers, threads_per_worker)`` ótimos para o hardware.

    Estratégia confirmada empiricamente em v2.17 + v2.20:
      • Para batch grande (``n_models_hint >= 10`` ou ``None``): preferir
        múltiplos workers com 2 threads cada (Modo D híbrido). Isso
        amortiza spawn overhead e maximiza throughput em CPU 8C+.
      • Para single/poucos modelos (``n_models_hint < 10``): preferir
        1 worker com N threads (Modo B multi-thread). Spawn de pool
        seria desperdício para 1-9 modelos.
      • Em ambos os casos: ``n_workers × threads_per_worker ≤ physical_cores``
        — nunca oversubscriba sobre cores físicos para workloads CPU-bound.

    Análise empírica v2.20 (Sprint 20.1, descoberta+reversão):
    Tentou-se v2.20.1 retornar `(phys, logical/phys)` em CPUs com HT/SMT
    sob hipótese de que o kernel `hmd_tiv` recursivo seria memory-bound e
    HT esconderia cache miss latency. Medição rigorosa em Mac 8C/16T HT
    com 5 runs consecutivos por config refutou a hipótese:

      | Config | Mediana E (600 pts) | Comportamento |
      |:------:|:-------------------:|:--------------|
      | 4w × 2t (8 threads = phys)    | **46k mod/h** | melhor |
      | 4w × 4t (16 threads = logical)| 38k mod/h     | -20-25% |

    O kernel é compute-bound suficiente para HT degradar (context switch
    + cache thrashing entre hyperthreads). A heurística v2.17 está
    confirmada como correta. Resultado: v2.20 reverteu a mudança e
    documentou a descoberta para evitar tentativas futuras.

    Args:
        n_models_hint: Estimativa do número de modelos a simular. Quando
            ``None`` (default), assume batch grande.

    Returns:
        Tupla ``(n_workers, threads_per_worker)`` recomendada.

    Example:
        Em Mac 8C/16T HT, batch grande (default GUI)::

            >>> recommend_default_parallelism()
            (4, 2)  # 4 workers × 2 threads = 8 = phys_cores ✓

        Em Mac 8C/16T HT, single model::

            >>> recommend_default_parallelism(n_models_hint=1)
            (1, 8)  # 1 worker × 8 threads = phys_cores ✓

        Em Linux 32C/64T HT, batch grande::

            >>> recommend_default_parallelism()
            (16, 2)  # 16 workers × 2 threads = 32 = phys_cores ✓

        Em Apple Silicon M1 8-core (sem HT)::

            >>> recommend_default_parallelism()
            (4, 2)  # 4 × 2 = 8 = cores ✓

    Note:
        Esta função é o ponto único de verdade para defaults da GUI e
        do benchmark CLI. Mudanças aqui propagam automaticamente para
        ``simulation_manager.py:spin_workers/spin_threads`` e para o
        argparse de ``bench_v214_numba.py``. A confirmação empírica
        v2.20 está documentada em ``docs/reports/v2.20_2026-05-02.md``.
    """
    _, phys, _ = detect_cpu_topology()
    if n_models_hint is not None and n_models_hint < 10:
        # Single/poucos modelos: 1 worker com N threads (Modo B)
        return (1, max(1, phys))
    # Batch grande: workers = phys // 2, threads = 2 (Modo D)
    # Para phys=2: workers=1, threads=2 (cai em Modo B)
    # Para phys=4: workers=2, threads=2
    # Para phys=8: workers=4, threads=2
    # Para phys=16: workers=8, threads=2
    n_workers = max(1, phys // 2)
    threads_per_worker = max(1, phys // n_workers)
    return (n_workers, threads_per_worker)


# ──────────────────────────────────────────────────────────────────────────────
# Anti-oversubscription + detecção de modo
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_effective_threads(
    n_workers: int,
    threads_per_worker: Optional[int],
    cpu_count: Optional[int] = None,
) -> int:
    """Calcula threads-por-worker com lógica anti-oversubscription.

    Quando `threads_per_worker` é ``None``, distribui CPUs disponíveis
    igualmente entre workers, garantindo que `n_workers × eff_threads <=
    cpu_count`. Isso evita o cenário típico de **oversubscription** onde
    cada worker tenta usar `os.cpu_count()` threads simultaneamente,
    levando a contenção severa (Numba TBB/OMP × N processos = 64+ threads
    em CPU 8-core → trashing).

    Args:
        n_workers: Número de workers no pool. Deve ser ``>= 1``.
        threads_per_worker: Override explícito do usuário. Se ``None``,
            ativa o cálculo auto. Se inteiro, valida e retorna.
        cpu_count: Override de CPUs disponíveis (útil para testes).
            Se ``None``, usa ``os.cpu_count() or 1``.

    Returns:
        Número efetivo de threads por worker (sempre ``>= 1``).

    Raises:
        ValueError: Se `threads_per_worker < 1` ou `n_workers < 1`.

    Example:
        Em CPU 8-core, distribuição automática para 4 workers::

            >>> _resolve_effective_threads(4, None, cpu_count=8)
            2

        Override explícito do usuário::

            >>> _resolve_effective_threads(4, 3, cpu_count=8)
            3

    Note:
        Para `n_workers > cpu_count` em modo auto, retorna 1 (1 thread por
        worker). Isso significa oversubscription **proposital** quando o
        usuário escolhe explicitamente mais workers que CPUs — caso de
        uso: I/O-bound chunks ou workload com muitos waits.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers deve ser >= 1 (got {n_workers})")
    if threads_per_worker is not None:
        if threads_per_worker < 1:
            raise ValueError(
                f"threads_per_worker deve ser >= 1 (got {threads_per_worker})"
            )
        return int(threads_per_worker)
    cpu = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, cpu // max(1, n_workers))


def _detect_mode(n_workers: int, eff_threads: int) -> str:
    """Identifica o modo de execução (A/B/C/D) baseado em workers × threads.

    Mapeamento:
        ``(1, 1)`` → ``"A"`` (Single — debug/pequeno)
        ``(1, N)`` → ``"B"`` (Multi-Thread — 1 simulação grande)
        ``(M, 1)`` → ``"C"`` (Single+Workers — batch n_pos baixo)
        ``(M, K)`` → ``"D"`` (Hybrid — DEFAULT PRODUÇÃO 30k modelos)

    Args:
        n_workers: Número de workers no pool.
        eff_threads: Threads efetivas por worker (após resolução).

    Returns:
        ``"A"``, ``"B"``, ``"C"`` ou ``"D"``.
    """
    if n_workers == 1 and eff_threads == 1:
        return "A"
    if n_workers == 1 and eff_threads > 1:
        return "B"
    if n_workers > 1 and eff_threads == 1:
        return "C"
    return "D"


# ──────────────────────────────────────────────────────────────────────────────
# Worker initializer + warmup JIT
# ──────────────────────────────────────────────────────────────────────────────
def _simulate_worker_init(n_threads: int, hankel_filter: str) -> None:
    """Inicializador picklable executado UMA vez por worker spawned.

    Aquece o cache JIT Numba com modelos triviais (single e multi-combo).
    Após isso, futuras chamadas em ``_run_simulate_chunk`` rodam JIT-puro.

    Args:
        n_threads: Threads Numba ativas por worker (mascaramento via
            ``numba.set_num_threads``). NÃO seta ``NUMBA_NUM_THREADS`` em
            env var — ver nota abaixo.
        hankel_filter: Nome do filtro Hankel para warmup.

    Note:
        NÃO setamos ``NUMBA_NUM_THREADS`` aqui. Quando este inicializador
        é chamado, ``geosteering_ai.simulation`` já foi importado como
        efeito colateral do unpickle do próprio worker — Python importa o
        pacote pai para resolver o módulo desta função. Nesse ponto Numba
        já foi inicializado com ``NUMBA_NUM_THREADS = os.cpu_count()``.

        Mudar ``os.environ["NUMBA_NUM_THREADS"]`` APÓS o import de Numba
        faz ``reload_config()`` (chamado pelo compilador JIT) detectar
        discrepância env vs config e lançar ``RuntimeError: Cannot set
        NUMBA_NUM_THREADS to a different value``.

        Threads ativas são controladas via ``SimulationConfig.num_threads``
        que chama ``numba.set_num_threads(n)`` — mascaramento correto.

        Marcado idempotente via ``_WORKER_INITIALIZED``.
    """
    global _WORKER_INITIALIZED
    if _WORKER_INITIALIZED:
        return
    # OMP_NUM_THREADS limita BLAS/MKL (seguro após import de Numba).
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "2")
    os.environ.setdefault("KMP_WARNINGS", "FALSE")
    try:
        from geosteering_ai.simulation.config import SimulationConfig
        from geosteering_ai.simulation.multi_forward import simulate_multi as _sm

        _cfg = SimulationConfig(
            backend="numba",
            num_threads=n_threads,
            hankel_filter=hankel_filter,
        )
        # Warmup 1: single-combo → compila _simulate_positions_njit_cached
        _sm(
            rho_h=np.array([1.0, 10.0, 1.0], dtype=np.float64),
            rho_v=np.array([1.0, 10.0, 1.0], dtype=np.float64),
            esp=np.array([5.0, 5.0], dtype=np.float64),
            positions_z=np.array([0.0, 5.0], dtype=np.float64),
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cfg=_cfg,
        )
        # Warmup 2: multi-combo → pré-compila _simulate_combined_prange
        _sm(
            rho_h=np.array([1.0, 10.0, 1.0], dtype=np.float64),
            rho_v=np.array([1.0, 10.0, 1.0], dtype=np.float64),
            esp=np.array([5.0, 5.0], dtype=np.float64),
            positions_z=np.array([0.0, 5.0], dtype=np.float64),
            frequencies_hz=[20000.0, 40000.0],
            tr_spacings_m=[1.0, 2.0],
            dip_degs=[0.0],
            cfg=_cfg,
        )
    except Exception:
        pass
    _WORKER_INITIALIZED = True


# ──────────────────────────────────────────────────────────────────────────────
# Pool persistente + lifecycle
# ──────────────────────────────────────────────────────────────────────────────
def _acquire_pool(
    n_workers: int, n_threads: int, hankel_filter: str
) -> ProcessPoolExecutor:
    """Retorna (ou cria) o `ProcessPoolExecutor` persistente do core.

    Singleton com chave ``(n_workers, n_threads, hankel_filter)``. Se a
    chave atual diverge do pool ativo, encerra o antigo e cria novo
    com a configuração desejada. Protegido por ``_LOCK`` para
    thread-safety.

    Args:
        n_workers: Número de processos no pool.
        n_threads: Threads Numba por worker (passado a `_simulate_worker_init`).
        hankel_filter: Nome do filtro Hankel para warmup JIT.

    Returns:
        ProcessPoolExecutor pronto para `submit`.

    Note:
        Usa contexto **spawn** (não fork) para isolamento total dos
        workers — evita herança de estado Python parcialmente
        inicializado, que causava bugs sutis em macOS/Windows.

        O cleanup é responsabilidade do chamador via `release_pool()`,
        tipicamente em `closeEvent` da UI ou em `atexit`.
    """
    global _PERSISTENT_POOL, _PERSISTENT_POOL_CONFIG
    cfg_key = (int(n_workers), int(n_threads), str(hankel_filter))
    with _LOCK:
        if _PERSISTENT_POOL is None or _PERSISTENT_POOL_CONFIG != cfg_key:
            _release_pool_unlocked()
            # ── FIX v2.16 (Sprint 15.1): dimensionar pool de threads Numba ──
            # Sete `NUMBA_NUM_THREADS` no env do PAI ANTES do spawn dos
            # workers. Os workers spawn herdam o env, e Numba lê o valor
            # durante a primeira import (que ocorre dentro do
            # `_simulate_worker_init`). Resultado: cada worker nasce com
            # pool Numba dimensionado em `n_threads`, evitando o RuntimeError
            # de `set_num_threads` em chamadas posteriores.
            #
            # Histórico v2.15: este passo foi removido (commit e1c8864) sob
            # a premissa de que `numba.set_num_threads()` em `simulate_multi`
            # bastaria. Mas o pool nascia com `cpu_count()` (16 hyperthreaded)
            # e `set_num_threads` falhava silenciosamente em estados
            # internos específicos (try/except: pass em multi_forward.py:880),
            # provocando regressão de 4–8× em produção.
            os.environ["NUMBA_NUM_THREADS"] = str(cfg_key[1])
            os.environ["OMP_NUM_THREADS"] = str(cfg_key[1])
            os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "2")
            os.environ.setdefault("KMP_WARNINGS", "FALSE")
            # ──────────────────────────────────────────────────────────────
            _PERSISTENT_POOL = ProcessPoolExecutor(
                max_workers=cfg_key[0],
                mp_context=_mp.get_context("spawn"),
                initializer=_simulate_worker_init,
                initargs=(cfg_key[1], cfg_key[2]),
            )
            _PERSISTENT_POOL_CONFIG = cfg_key
        return _PERSISTENT_POOL


def _release_pool_unlocked() -> None:
    """Versão sem lock de `release_pool` — chamada interna."""
    global _PERSISTENT_POOL, _PERSISTENT_POOL_CONFIG
    if _PERSISTENT_POOL is not None:
        try:
            # `cancel_futures=True` aborta tarefas pendentes (Python 3.9+).
            _PERSISTENT_POOL.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python < 3.9 não suporta cancel_futures.
            _PERSISTENT_POOL.shutdown(wait=False)
        except Exception:
            pass
        _PERSISTENT_POOL = None
        _PERSISTENT_POOL_CONFIG = None


def release_pool() -> None:
    """Encerra o pool persistente do core (público).

    Chamar em `closeEvent` da UI ou via `atexit` para liberar
    recursos. Após o release, a próxima chamada a `_acquire_pool`
    cria pool novo.

    Note:
        É segura para chamar mesmo quando não há pool ativo (no-op).
        Não bloqueia (`wait=False`) — workers em execução são
        cancelados via `cancel_futures=True`.
    """
    with _LOCK:
        _release_pool_unlocked()


# ──────────────────────────────────────────────────────────────────────────────
# Divisão de trabalho — split uniforme com remainder distribuído
# ──────────────────────────────────────────────────────────────────────────────
def _split_models_uniform(
    models: List[Dict[str, Any]], n_workers: int
) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """Divide `models` em `n_workers` chunks uniformes preservando índices.

    Usa o algoritmo clássico de distribuição de remainder: os primeiros
    ``n_total % n_workers`` chunks recebem 1 item extra, os restantes
    recebem ``n_total // n_workers``. Isto garante:

      • diferença máxima de 1 item entre chunks (balanced load);
      • cobertura completa: ``sum(len(c) for c in chunks) == n_total``;
      • ordem preservada: o índice original de cada modelo é mantido
        como tupla ``(orig_idx, model)`` para permitir reagregação
        ordenada na main thread.

    Args:
        models: Lista de modelos a distribuir.
        n_workers: Número de workers (chunks). Será limitado a
            ``min(n_workers, len(models))`` para evitar chunks vazios.

    Returns:
        Lista de chunks, cada chunk = ``[(orig_idx, model), ...]``.

    Example:
        ``10 modelos ÷ 3 workers``::

            >>> chunks = _split_models_uniform([{"id": i} for i in range(10)], 3)
            >>> [len(c) for c in chunks]
            [4, 3, 3]
            >>> [idx for c in chunks for idx, _ in c]
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    n_total = len(models)
    if n_total == 0:
        return []
    n_workers = max(1, min(n_workers, n_total))
    base = n_total // n_workers
    remainder = n_total % n_workers
    chunks: List[List[Tuple[int, Dict[str, Any]]]] = []
    pos = 0
    for w in range(n_workers):
        size = base + (1 if w < remainder else 0)
        chunks.append([(pos + j, models[pos + j]) for j in range(size)])
        pos += size
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Worker function (picklable)
# ──────────────────────────────────────────────────────────────────────────────
def _run_simulate_chunk(
    chunk: List[Tuple[int, Dict[str, Any]]],
    sim_kwargs: Dict[str, Any],
) -> Tuple[List[int], np.ndarray]:
    """Executa um chunk de modelos dentro de um worker spawned.

    **Picklable** — não captura closures Python complexas. Recebe lista
    de tuplas ``(orig_idx, model)`` e dict com kwargs comuns para todos
    os modelos do chunk.

    Itera sequencialmente dentro do worker (sem nested pool — evita
    explosão de processos). Cada modelo invoca
    :func:`simulate_multi` no caminho single-modelo, retornando um
    `MultiSimulationResult`. Os tensores `H_tensor` são empilhados
    em um array 6-D ``(n_chunk, nTR, nAngles, n_pos, nf, 9)``.

    Args:
        chunk: Lista de pares ``(orig_idx, model_dict)``. Cada
            ``model_dict`` deve conter chaves ``rho_h``, ``rho_v``, ``esp``.
        sim_kwargs: Kwargs comuns repassados a `simulate_multi`,
            incluindo ``positions_z``, ``frequencies_hz``,
            ``tr_spacings_m``, ``dip_degs``, ``cfg``, ``hankel_filter``,
            etc.

    Returns:
        Tupla ``(indices, H_chunk)``:
          • indices: lista de índices originais (preservada para
            reagregação ordenada).
          • H_chunk: array 6-D shape ``(n_chunk, nTR, nAngles, n_pos,
            nf, 9)`` complex128.

    Note:
        Usa import lazy de `simulate_multi` para garantir que o módulo
        seja resolvido **dentro** do worker (necessário em contexto spawn).
    """
    # Import lazy: o worker spawned precisa importar simulate_multi
    # dentro do próprio processo. Import no topo do módulo seria
    # circular (multi_forward → _workers → multi_forward).
    from geosteering_ai.simulation.multi_forward import simulate_multi

    if not chunk:
        return [], np.empty((0,), dtype=np.complex128)

    indices: List[int] = []
    H_parts: List[np.ndarray] = []
    for orig_idx, model in chunk:
        result = simulate_multi(
            rho_h=np.asarray(model["rho_h"], dtype=np.float64),
            rho_v=np.asarray(model["rho_v"], dtype=np.float64),
            esp=np.asarray(model["esp"], dtype=np.float64),
            **sim_kwargs,
        )
        indices.append(orig_idx)
        # Adiciona axis-0 para empilhamento posterior.
        H_parts.append(result.H_tensor[None, ...])
    H_chunk = np.concatenate(H_parts, axis=0)
    return indices, H_chunk


# ──────────────────────────────────────────────────────────────────────────────
# Container de saída — MultiSimulationResultBatch
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MultiSimulationResultBatch:
    """Resultado de :func:`simulate_multi` chamado com `models=[...]`.

    Diferente de :class:`MultiSimulationResult` (single-modelo, axis 0 = nTR),
    este container empilha múltiplos modelos no axis 0:

      • ``H_stack[i]`` = `H_tensor` do modelo ``i`` (mesma shape de
        `MultiSimulationResult.H_tensor`).
      • Inclui métricas de execução (tempo, throughput, modo).

    Attributes:
        H_stack: Tensor empilhado.
            Shape ``(n_models, nTR, nAngles, n_pos, nf, 9)`` complex128.
        z_obs: Profundidades do ponto-médio. Shape ``(nAngles, n_pos)``
            float64. Compartilhado entre todos os modelos do batch
            (depende apenas de ``positions_z``, ``dip_degs``, ``tr_spacings_m``).
        elapsed_s: Tempo total de execução (segundos).
        throughput_mod_per_h: Modelos por hora (n_models / elapsed × 3600).
        backend: Backend usado (``"numba"``, ``"fortran_f2py"``, etc).
        n_workers: Número de workers do pool (1 = in-process).
        n_threads: Threads efetivas por worker.
        mode: Modo de execução (``"A"`` / ``"B"`` / ``"C"`` / ``"D"``).

    Example:
        Acesso ao tensor do 5º modelo::

            >>> result = simulate_multi(models=models, n_workers=4)
            >>> H_5 = result.H_stack[5]      # shape (nTR, nAngles, n_pos, nf, 9)
            >>> result.mode
            'D'

    Note:
        Frozen dataclass para evitar mutação acidental de tensores
        compartilhados entre workers.
    """

    H_stack: np.ndarray
    z_obs: np.ndarray
    elapsed_s: float
    throughput_mod_per_h: float
    backend: str
    n_workers: int
    n_threads: int
    mode: str

    def to_list_of_results(self) -> List["np.ndarray"]:
        """Retorna lista de tensores ``H[i]`` (axis-0 split).

        Útil para integração com APIs upstream que esperam iteráveis
        de tensores 5-D em vez de tensor 6-D empilhado.

        Returns:
            Lista de N arrays 5-D, cada um shape
            ``(nTR, nAngles, n_pos, nf, 9)``.
        """
        return [self.H_stack[i] for i in range(self.H_stack.shape[0])]


# ──────────────────────────────────────────────────────────────────────────────
# Orquestrador principal — chamado por simulate_multi(models=[...])
# ──────────────────────────────────────────────────────────────────────────────
def run_batch(
    models: List[Dict[str, Any]],
    sim_kwargs: Dict[str, Any],
    n_workers: int,
    threads_per_worker: Optional[int],
    backend: str,
    hankel_filter: str,
) -> MultiSimulationResultBatch:
    """Executa batch multi-modelo com workers nativos (4 modos A/B/C/D).

    Função de alto nível chamada pelo dispatcher de :func:`simulate_multi`
    quando o usuário fornece ``models=[...]``. Delegada para evitar
    poluir `multi_forward.py` com lógica de paralelismo inter-processo.

    Args:
        models: Lista não-vazia de dicionários ``{rho_h, rho_v, esp}``.
        sim_kwargs: Kwargs comuns para `simulate_multi` (frequencies_hz,
            tr_spacings_m, dip_degs, positions_z, cfg, etc.).
            **Não deve conter** ``rho_h``, ``rho_v``, ``esp`` (vêm de `models`).
        n_workers: Número de workers do pool (>= 1).
        threads_per_worker: Threads Numba por worker. ``None`` ativa
            anti-oversubscription auto.
        backend: String de backend para campo da dataclass de saída.
        hankel_filter: Nome do filtro Hankel para warmup JIT do pool.

    Returns:
        :class:`MultiSimulationResultBatch` com ``H_stack`` agregado.

    Raises:
        ValueError: Se ``models`` é vazia ou ``n_workers < 1``.
    """
    if not models:
        raise ValueError("simulate_multi(models=[...]): lista vazia.")
    if n_workers < 1:
        raise ValueError(f"n_workers deve ser >= 1 (got {n_workers})")

    n_models = len(models)
    eff_threads = _resolve_effective_threads(n_workers, threads_per_worker)
    mode = _detect_mode(n_workers, eff_threads)

    t0 = time.perf_counter()

    # ── Modos A/B (n_workers == 1): execução in-process ───────────────
    # Sem custo de spawn/IPC. O paralelismo aqui é apenas intra-modelo
    # via `prange` Numba (controlado por num_threads no cfg).
    if n_workers == 1:
        H_parts, z_obs_first = _run_inproc(models, sim_kwargs, eff_threads)
        H_stack = np.stack(H_parts, axis=0)
    else:
        # ── Modos C/D (n_workers > 1): ProcessPool ────────────────────
        # Code-review fix P0 #1 (race condition shutdown × submit): se
        # outra thread chama `release_pool()` entre `_acquire_pool()` e
        # `pool.submit()`, o `submit` levanta RuntimeError. Capturamos e
        # re-emitimos com mensagem informativa indicando cancelamento.
        chunks = _split_models_uniform(models, n_workers)
        try:
            pool = _acquire_pool(n_workers, eff_threads, hankel_filter)
            futures = [
                pool.submit(_run_simulate_chunk, chunk, sim_kwargs)
                for chunk in chunks
                if chunk
            ]
        except RuntimeError as e:
            # Pool foi shutdown por release_pool() concorrente — tipicamente
            # um closeEvent durante simulação ativa. Sinalizamos cancelamento.
            raise RuntimeError(
                "Pool de workers foi encerrado (release_pool) durante o "
                f"submit do batch. Causa provável: closeEvent ou shutdown "
                f"concorrente. Detalhe: {e}"
            ) from e

        H_by_idx: Dict[int, np.ndarray] = {}
        try:
            for fut in as_completed(futures):
                indices, H_chunk = fut.result()
                for k, idx in enumerate(indices):
                    H_by_idx[idx] = H_chunk[k]
        except RuntimeError as e:
            # Idem: workers cancelados via cancel_futures=True durante shutdown.
            raise RuntimeError(
                "Workers cancelados durante a coleta de resultados (provável "
                f"closeEvent concorrente). Detalhe: {e}"
            ) from e

        if len(H_by_idx) != n_models:
            raise RuntimeError(
                f"Inconsistência na agregação: esperados {n_models} resultados, "
                f"obtidos {len(H_by_idx)}."
            )
        H_stack = np.stack([H_by_idx[i] for i in range(n_models)], axis=0)

        # `z_obs` é determinístico em função de positions_z/dip — derivar
        # a partir de uma simulação rápida do primeiro modelo seria custoso.
        # Em vez disso, computamos in-place com o primeiro modelo (custo de 1
        # chamada extra, mas evita ambiguidade em workers).
        z_obs_first = _compute_z_obs_first(models[0], sim_kwargs)

    elapsed = max(time.perf_counter() - t0, 1e-12)
    throughput = float(n_models) / elapsed * 3600.0

    return MultiSimulationResultBatch(
        H_stack=H_stack,
        z_obs=z_obs_first,
        elapsed_s=float(elapsed),
        throughput_mod_per_h=float(throughput),
        backend=str(backend),
        n_workers=int(n_workers),
        n_threads=int(eff_threads),
        mode=str(mode),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers privados de `run_batch`
# ──────────────────────────────────────────────────────────────────────────────
def _run_inproc(
    models: List[Dict[str, Any]],
    sim_kwargs: Dict[str, Any],
    eff_threads: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Executa modelos sequencialmente no main process (Modos A/B).

    Configura `numba.set_num_threads(eff_threads)` quando possível,
    invoca `simulate_multi` por modelo e coleta `H_tensor` + `z_obs`.

    Returns:
        Tupla ``(H_parts, z_obs)`` onde ``H_parts`` é lista de N tensores
        5-D e ``z_obs`` é o array do primeiro modelo (compartilhado entre
        todos quando `positions_z`/`dip_degs`/`tr_spacings_m` são iguais).
    """
    from geosteering_ai.simulation.multi_forward import simulate_multi

    # Modo B: configura threads Numba se disponível. No Modo A (eff=1)
    # também aplicamos para garantir que prange use 1 thread (paridade
    # serial determinística).
    try:
        import numba

        numba.set_num_threads(int(eff_threads))
    except Exception:  # pragma: no cover
        # Numba não instalado ou sem suporte a set_num_threads — segue.
        pass

    H_parts: List[np.ndarray] = []
    z_obs_first: Optional[np.ndarray] = None
    for model in models:
        result = simulate_multi(
            rho_h=np.asarray(model["rho_h"], dtype=np.float64),
            rho_v=np.asarray(model["rho_v"], dtype=np.float64),
            esp=np.asarray(model["esp"], dtype=np.float64),
            **sim_kwargs,
        )
        H_parts.append(result.H_tensor)
        if z_obs_first is None:
            z_obs_first = np.array(result.z_obs)
    assert z_obs_first is not None  # sempre populado se models não-vazio
    return H_parts, z_obs_first


def _compute_z_obs_first(
    first_model: Dict[str, Any], sim_kwargs: Dict[str, Any]
) -> np.ndarray:
    """Computa `z_obs` do primeiro modelo (Modos C/D).

    `z_obs` depende apenas de ``positions_z``, ``dip_degs`` e
    ``tr_spacings_m`` — é idêntico para todos os modelos do batch
    quando esses parâmetros são compartilhados (caso comum).

    Como custo de 1 chamada in-process é desprezível (~0.01s para
    n_pos=600), prefere-se computar localmente em vez de marshalar
    `z_obs` de um worker (que adicionaria complexidade ao protocolo
    de retorno).

    Args:
        first_model: Primeiro modelo do batch (para rho_h/rho_v/esp).
        sim_kwargs: Kwargs comuns repassados a `simulate_multi`.

    Returns:
        Array float64 shape ``(nAngles, n_pos)``.
    """
    from geosteering_ai.simulation.multi_forward import simulate_multi

    result = simulate_multi(
        rho_h=np.asarray(first_model["rho_h"], dtype=np.float64),
        rho_v=np.asarray(first_model["rho_v"], dtype=np.float64),
        esp=np.asarray(first_model["esp"], dtype=np.float64),
        **sim_kwargs,
    )
    return np.array(result.z_obs)


# ──────────────────────────────────────────────────────────────────────────────
# Inventário de exports
# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    # Container público
    "MultiSimulationResultBatch",
    # API pública
    "detect_cpu_topology",
    "recommend_default_parallelism",
    "release_pool",
    "run_batch",
    # Helpers expostos para testes
    "_detect_mode",
    "_resolve_effective_threads",
    "_split_models_uniform",
]
