#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MCP SERVER: Numba Profiler — Geosteering AI v2.0                           ║
# ║  Fase 1 do roadmap multi-agente §22.1 (I1.10): handlers REAIS                ║
# ║                                                                              ║
# ║  Propósito:                                                                  ║
# ║    Expor profiling/benchmark do simulador Numba como capacidades MCP        ║
# ║    nativas para o agente geosteering-perf-reviewer (Haiku 4.5).             ║
# ║                                                                              ║
# ║  6 Tools expostas via MCP:                                                   ║
# ║    • run_scenario_benchmark    — simulate_multi mediana 5 runs               ║
# ║    • compare_branches          — diff de throughput main vs feature branch  ║
# ║    • check_cpu_topology        — phys/logical/HT detection (real)           ║
# ║    • check_oversubscription    — workers × threads vs phys_cores            ║
# ║    • profile_kernel            — cProfile + pstats top-N hotspots           ║
# ║    • analyze_jit_cache         — get_jit_cache_info (RAM + signatures + disk║
# ║                                                                              ║
# ║  Reutilização (REAL, não scaffold):                                          ║
# ║    • simulate_multi (multi_forward.py)                                       ║
# ║    • detect_cpu_topology (_workers.py:169)                                   ║
# ║    • recommend_default_parallelism (_workers.py:292)                         ║
# ║    • get_jit_cache_info (multi_forward.py — novo, I1.10 prep)                ║
# ║                                                                              ║
# ║  Dependências: mcp>=1.0,<2.0, numpy, geosteering_ai (editable)              ║
# ║  Status: PRODUÇÃO (Fase 1 I1.10 — 2026-05-09)                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""MCP Server para profiling/benchmark do simulador Geosteering AI v2.0.

Expõe ao Claude Code (via Model Context Protocol) 6 capacidades de
profiling/benchmark normalmente executadas via
``benchmarks/bench_v22_flat_prange.py`` e
``geosteering_ai.simulation._workers``.

Vantagens vs CLI manual:
    • Resposta estruturada (JSON) com p25/p50/p75
    • Cache de detect_cpu_topology (1× por process)
    • Validação automática de oversubscription antes de cada bench
    • Integração direta com agente ``geosteering-perf-reviewer``

Note:
    Cenários "leves" (A, E quick) rodam em <1s; cenários pesados (J completo)
    podem levar minutos. Default ``runs=2, n_pos=30`` torna handlers
    smoke-friendly.

Ref: docs/reports/arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md §22.1
"""

from __future__ import annotations

import cProfile
import io
import json
import logging
import pstats
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Permitir invocação direta (`python tools/numba-profiler-mcp/server.py`)
# garantindo que `geosteering_ai` seja importável a partir da raiz do repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Cenários Catalogados (referência: docs/reference/analise_cenarios_otimizacao_simulador_numba.md)
# ──────────────────────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict[str, Any]] = {
    "A": {
        "n_pos": 30, "nf": 1, "nTR": 1, "nAng": 1,
        "baseline_v2_21_mod_h": 1_392_371,
        "desc": "Baseline (referência histórica)",
    },
    "B": {
        "n_pos": 200, "nf": 1, "nTR": 3, "nAng": 4,
        "baseline_v2_21_mod_h": 303_452,
        "desc": "Multi-array LWD",
    },
    "E": {
        "n_pos": 600, "nf": 1, "nTR": 1, "nAng": 1,
        "baseline_v2_21_mod_h": 121_957,
        "desc": "Produção LWD (meta histórica >120k)",
    },
    "F": {
        "n_pos": 600, "nf": 4, "nTR": 1, "nAng": 1,
        "baseline_v2_21_mod_h": 100_000,
        "desc": "ARC multi-frequência",
    },
    "J": {
        "n_pos": 600, "nf": 4, "nTR": 4, "nAng": 8,
        "baseline_v2_21_mod_h": None,
        "desc": "Periscope-15 completo (todos eixos)",
    },
}


# ──────────────────────────────────────────────────────────────────────────
# Handlers (implementação real)
# ──────────────────────────────────────────────────────────────────────────


def _build_scenario_inputs(s: dict[str, Any]) -> dict[str, Any]:
    """Constrói arrays canônicos (Oklahoma 3-camadas) para um cenário.

    Retorna kwargs prontos para ``simulate_multi``.
    """
    import numpy as np

    n_pos = s["n_pos"]
    nf = s["nf"]
    nTR = s["nTR"]
    nAng = s["nAng"]

    rho_h = np.array([1.0, 20.0, 1.0], dtype=np.float64)
    rho_v = np.array([1.0, 40.0, 1.0], dtype=np.float64)
    esp = np.array([5.0], dtype=np.float64)
    positions_z = np.linspace(-2.0, 7.0, n_pos)
    tr_spacings_m = list(np.linspace(0.5, 1.5, nTR))
    dip_degs = list(np.linspace(0.0, 60.0, nAng))
    frequencies_hz = list(np.linspace(20000.0, 200000.0, nf))

    return {
        "rho_h": rho_h,
        "rho_v": rho_v,
        "esp": esp,
        "positions_z": positions_z,
        "tr_spacings_m": tr_spacings_m,
        "dip_degs": dip_degs,
        "frequencies_hz": frequencies_hz,
    }


def _impl_run_scenario_benchmark(
    scenario_id: str = "E",
    runs: int = 2,
    use_flat_prange: bool = True,
) -> dict[str, Any]:
    """Executa um cenário catalogado e retorna mediana/p25/p75 de N runs.

    Delegação: ``geosteering_ai.simulation.simulate_multi``.

    Args:
        scenario_id: ID do cenário (A, B, E, F, J).
        runs: Número de runs para mediana (default 2 para smoke-test).
              Recomendado ≥5 para validade estatística.
        use_flat_prange: Ativa Sprint v2.22 FLAT prange (default True
            desde v2.22.4).

    Returns:
        Dict com:

        - ``scenario_id`` / ``scenario_desc``
        - ``config`` (n_pos, nf, nTR, nAng)
        - ``runs`` (int)
        - ``median_s`` (float)
        - ``stdev_s`` (float)
        - ``p25_s`` / ``p75_s`` (float)
        - ``mod_h`` (float): throughput em modelos/hora
        - ``baseline_v2_21_mod_h`` (float | None)
        - ``speedup_vs_baseline`` (float | None)
    """
    if scenario_id not in SCENARIOS:
        return {
            "error": f"Unknown scenario: {scenario_id}",
            "available": list(SCENARIOS.keys()),
        }

    try:
        from geosteering_ai.simulation import simulate_multi
        from geosteering_ai.simulation.config import SimulationConfig
    except ImportError as exc:
        return {
            "scenario_id": scenario_id,
            "error": f"ImportError: {exc}",
        }

    s = SCENARIOS[scenario_id]
    inputs = _build_scenario_inputs(s)

    # Config do simulador honra use_flat_prange
    cfg = SimulationConfig(
        frequency_hz=inputs["frequencies_hz"][0],
        tr_spacing_m=inputs["tr_spacings_m"][0],
        backend="numba",
        use_flat_prange=use_flat_prange,
    )

    times: list[float] = []
    n_total_models = s["nTR"] * s["nAng"] * s["nf"] * s["n_pos"]

    # Warmup: 1 chamada para JIT compile (não cronometra)
    simulate_multi(**inputs, cfg=cfg)

    for _ in range(max(runs, 1)):
        t0 = time.perf_counter()
        simulate_multi(**inputs, cfg=cfg)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    median_s = statistics.median(times)
    stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
    p25_s = statistics.quantiles(times, n=4)[0] if len(times) >= 4 else min(times)
    p75_s = statistics.quantiles(times, n=4)[2] if len(times) >= 4 else max(times)
    mod_h = (n_total_models / median_s) * 3600.0 if median_s > 0 else 0.0

    baseline = s.get("baseline_v2_21_mod_h")
    speedup = mod_h / baseline if baseline else None

    return {
        "scenario_id": scenario_id,
        "scenario_desc": s["desc"],
        "config": {k: v for k, v in s.items() if k not in ("desc", "baseline_v2_21_mod_h")},
        "use_flat_prange": use_flat_prange,
        "runs": runs,
        "median_s": median_s,
        "stdev_s": stdev_s,
        "p25_s": p25_s,
        "p75_s": p75_s,
        "n_models_per_run": n_total_models,
        "mod_h": mod_h,
        "baseline_v2_21_mod_h": baseline,
        "speedup_vs_baseline": speedup,
    }


def _impl_compare_branches(
    scenario_id: str = "E",
    branch_a: str = "main",
    branch_b: str = "HEAD",
    runs: int = 3,
) -> dict[str, Any]:
    """Compara throughput entre 2 branches (mediana N runs cada).

    Para evitar trabalho perdido + state corruption, requer working tree
    LIMPO (sem mudanças unstaged). Retorna ``{error: "dirty tree"}``
    caso contrário.

    Args:
        scenario_id: Cenário a executar (default ``E``, smoke-friendly).
        branch_a: Branch baseline (default ``main``).
        branch_b: Branch sob teste (default ``HEAD``).
        runs: Runs por branch.

    Returns:
        Dict com ``mod_h_a``, ``mod_h_b``, ``speedup``, ``regression``.

    Note:
        ATUALMENTE só executa benchmark no branch ATUAL (``HEAD``) e
        reporta como ``branch_b``. Para checkout multi-branch real,
        usar ``git worktree add`` em sessão dedicada — comportamento
        pleno em sprints futuros (Etapa 2 §22.2).
    """
    # Validação: working tree limpo
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_REPO_ROOT,
            check=False,
        )
        if result.stdout.strip():
            return {
                "scenario_id": scenario_id,
                "branch_a": branch_a,
                "branch_b": branch_b,
                "error": "dirty tree",
                "details": "Working tree não está limpo; commit/stash antes de compare_branches.",
                "git_status": result.stdout[:500],
            }
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {
            "scenario_id": scenario_id,
            "error": f"git status falhou: {exc}",
        }

    # Branch atual
    try:
        current_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_REPO_ROOT,
            check=False,
        ).stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        current_branch = "unknown"

    # Por enquanto, executa apenas no branch atual (limitação documentada)
    bench_b = _impl_run_scenario_benchmark(scenario_id=scenario_id, runs=runs)

    return {
        "scenario_id": scenario_id,
        "branch_a": branch_a,
        "branch_b": branch_b,
        "current_branch": current_branch,
        "bench_b": bench_b,
        "note": (
            "compare_branches roda apenas no branch atual nesta versão. "
            "Para diff real, executar em sessão dedicada com git worktree."
        ),
        "limitation": "single-branch",
    }


def _impl_check_cpu_topology() -> dict[str, Any]:
    """Detecta topologia da CPU via geosteering_ai._workers.

    Delegação: ``detect_cpu_topology() -> (logical, physical, has_ht)``.

    Returns:
        Dict com ``logical_cores``, ``physical_cores``, ``has_hyperthreading``,
        ``ratio_logical_phys`` (float).
    """
    try:
        from geosteering_ai.simulation._workers import detect_cpu_topology
    except ImportError as exc:
        return {"error": f"ImportError: {exc}"}

    logical, physical, has_ht = detect_cpu_topology()
    ratio = logical / physical if physical > 0 else 0.0

    return {
        "logical_cores": logical,
        "physical_cores": physical,
        "has_hyperthreading": has_ht,
        "ratio_logical_phys": ratio,
    }


def _impl_check_oversubscription(
    n_workers: int = 4,
    threads_per_worker: int = 2,
) -> dict[str, Any]:
    """Verifica se config workers × threads excede phys_cores.

    Delegação: ``detect_cpu_topology`` + ``recommend_default_parallelism``.

    Args:
        n_workers: Número de workers paralelos.
        threads_per_worker: Threads Numba por worker.

    Returns:
        Dict com ``oversubscribed`` (bool), ``recommendation`` (str).
    """
    try:
        from geosteering_ai.simulation._workers import (
            detect_cpu_topology,
            recommend_default_parallelism,
        )
    except ImportError as exc:
        return {"error": f"ImportError: {exc}"}

    logical, physical, has_ht = detect_cpu_topology()
    total = n_workers * threads_per_worker
    oversubscribed = total > physical

    rec_w, rec_t = recommend_default_parallelism()

    if oversubscribed:
        recommendation = (
            f"OVERSUBSCRIBED: {total} threads > {physical} phys cores. "
            f"Recomendado: {rec_w} workers × {rec_t} threads (={rec_w * rec_t})."
        )
    else:
        recommendation = "OK"

    return {
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "total_threads": total,
        "physical_cores": physical,
        "logical_cores": logical,
        "has_hyperthreading": has_ht,
        "oversubscribed": oversubscribed,
        "ratio": total / physical if physical > 0 else 0.0,
        "recommended_workers": rec_w,
        "recommended_threads_per_worker": rec_t,
        "recommendation": recommendation,
    }


def _impl_profile_kernel(
    scenario_id: str = "A",
    top_n: int = 15,
) -> dict[str, Any]:
    """Profile de uma execução via cProfile, retorna top-N hotspots.

    Roda ``simulate_multi`` com cenário leve sob ``cProfile.Profile()``,
    extrai estatísticas com ``pstats`` ordenadas por ``cumulative``.

    Args:
        scenario_id: Cenário a profile (default ``A``, mais rápido).
        top_n: Número de hotspots a reportar (default 15).

    Returns:
        Dict com:

        - ``scenario_id``
        - ``elapsed_s`` (float)
        - ``hotspots_text`` (str): output formatado de pstats
        - ``n_function_calls`` (int)
    """
    if scenario_id not in SCENARIOS:
        return {
            "error": f"Unknown scenario: {scenario_id}",
            "available": list(SCENARIOS.keys()),
        }

    try:
        from geosteering_ai.simulation import simulate_multi
        from geosteering_ai.simulation.config import SimulationConfig
    except ImportError as exc:
        return {"error": f"ImportError: {exc}"}

    s = SCENARIOS[scenario_id]
    inputs = _build_scenario_inputs(s)
    cfg = SimulationConfig(
        frequency_hz=inputs["frequencies_hz"][0],
        tr_spacing_m=inputs["tr_spacings_m"][0],
        backend="numba",
    )

    # Warmup JIT
    simulate_multi(**inputs, cfg=cfg)

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    simulate_multi(**inputs, cfg=cfg)
    profiler.disable()
    elapsed = time.perf_counter() - t0

    # Capture pstats output
    sio = io.StringIO()
    stats = pstats.Stats(profiler, stream=sio).sort_stats("cumulative")
    stats.print_stats(top_n)
    hotspots_text = sio.getvalue()

    return {
        "scenario_id": scenario_id,
        "elapsed_s": elapsed,
        "hotspots_text": hotspots_text,
        "n_function_calls": getattr(stats, "total_calls", -1),
        "top_n": top_n,
    }


def _impl_analyze_jit_cache() -> dict[str, Any]:
    """Analisa cache JIT Numba (RAM + signatures + disk).

    Delegação: ``geosteering_ai.simulation.get_jit_cache_info`` (I1.10 prep).

    Returns:
        Dict com:

        - ``n_entries`` (int): entradas em RAM
        - ``approx_bytes`` (int)
        - ``keys_summary`` (list[str])
        - ``dispatcher_signatures`` (dict[str, int])
        - ``cache_dir_disk_bytes`` (int)
    """
    try:
        from geosteering_ai.simulation import get_jit_cache_info
    except ImportError as exc:
        return {"error": f"ImportError: {exc}"}

    info = get_jit_cache_info()
    info["check_name"] = "jit_cache_info"
    return info


# ──────────────────────────────────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "run_scenario_benchmark": _impl_run_scenario_benchmark,
    "compare_branches": _impl_compare_branches,
    "check_cpu_topology": _impl_check_cpu_topology,
    "check_oversubscription": _impl_check_oversubscription,
    "profile_kernel": _impl_profile_kernel,
    "analyze_jit_cache": _impl_analyze_jit_cache,
}


def _build_tool_definitions() -> list[dict[str, Any]]:
    """Retorna definições de tools com inputSchema JSONSchema (MCP-compatível)."""
    return [
        {
            "name": "run_scenario_benchmark",
            "description": (
                "Executa simulate_multi para cenário catalogado (A/B/E/F/J) "
                "e retorna mediana/p25/p75 de N runs."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_id": {
                        "type": "string",
                        "enum": ["A", "B", "E", "F", "J"],
                        "default": "E",
                    },
                    "runs": {"type": "integer", "default": 2, "minimum": 1},
                    "use_flat_prange": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "compare_branches",
            "description": (
                "Compara throughput entre 2 branches. Requer working tree limpo. "
                "Versão atual: roda apenas no branch atual (limitação documentada)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string", "default": "E"},
                    "branch_a": {"type": "string", "default": "main"},
                    "branch_b": {"type": "string", "default": "HEAD"},
                    "runs": {"type": "integer", "default": 3},
                },
            },
        },
        {
            "name": "check_cpu_topology",
            "description": (
                "Detecta logical/physical cores e hyperthreading via "
                "geosteering_ai._workers.detect_cpu_topology."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "check_oversubscription",
            "description": (
                "Verifica se workers × threads_per_worker excede physical_cores. "
                "Sugere recommend_default_parallelism canônica."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "n_workers": {"type": "integer", "default": 4, "minimum": 1},
                    "threads_per_worker": {"type": "integer", "default": 2, "minimum": 1},
                },
            },
        },
        {
            "name": "profile_kernel",
            "description": (
                "cProfile + pstats sobre simulate_multi em cenário catalogado. "
                "Retorna top-N hotspots por cumulative time."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string", "default": "A"},
                    "top_n": {"type": "integer", "default": 15, "minimum": 1},
                },
            },
        },
        {
            "name": "analyze_jit_cache",
            "description": (
                "Diagnóstico do cache JIT Numba: entradas em RAM, "
                "dispatcher signatures, tamanho on-disk de __pycache__/_numba/."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


# ──────────────────────────────────────────────────────────────────────────
# MCP Server boilerplate (runtime — requer mcp instalado)
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entrypoint do MCP server (stdio transport).

    Inicializa um ``mcp.server.Server`` e registra as 6 tools.
    Cada handler é despachado em ``asyncio.to_thread`` (CPU-bound).

    Raises:
        SystemExit: Se ``mcp`` não estiver instalado (com fallback JSON).
    """
    import asyncio

    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError as exc:
        logger.error(
            "numba-profiler MCP requer 'mcp>=1.0,<2.0'. "
            "Instale com: pip install -r tools/numba-profiler-mcp/requirements.txt"
        )
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"mcp package not installed: {exc}",
                    "tools": [t["name"] for t in _build_tool_definitions()],
                }
            )
        )
        sys.exit(1)

    server: Server = Server("numba-profiler")

    @server.list_tools()  # type: ignore[no-untyped-call,misc]
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=defn["name"],
                description=defn["description"],
                inputSchema=defn["inputSchema"],
            )
            for defn in _build_tool_definitions()
        ]

    @server.call_tool()  # type: ignore[no-untyped-call,misc]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")
        result = await asyncio.to_thread(fn, **(arguments or {}))
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    logger.info(
        "numba-profiler MCP server iniciando (6 tools, %d handlers reais)",
        len(TOOL_REGISTRY),
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
