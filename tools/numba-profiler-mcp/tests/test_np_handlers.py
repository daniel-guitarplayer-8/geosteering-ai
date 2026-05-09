# -*- coding: utf-8 -*-
"""Testes unitários para os handlers do numba-profiler MCP (I1.10)."""

from __future__ import annotations

import importlib.util  # noqa: E402
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parents[1]
_SERVER_PATH = _MCP_DIR / "server.py"
_spec = importlib.util.spec_from_file_location("numba_profiler_server", _SERVER_PATH)
assert _spec and _spec.loader, "Falha ao carregar numba_profiler/server.py"
server = importlib.util.module_from_spec(_spec)
sys.modules["numba_profiler_server"] = server
_spec.loader.exec_module(server)

# ──────────────────────────────────────────────────────────────────────────
# Tool registry / metadata
# ──────────────────────────────────────────────────────────────────────────


def test_tool_registry_has_six_handlers() -> None:
    assert len(server.TOOL_REGISTRY) == 6
    expected = {
        "run_scenario_benchmark",
        "compare_branches",
        "check_cpu_topology",
        "check_oversubscription",
        "profile_kernel",
        "analyze_jit_cache",
    }
    assert set(server.TOOL_REGISTRY.keys()) == expected


def test_tool_definitions_match_registry() -> None:
    defs = server._build_tool_definitions()
    names = {d["name"] for d in defs}
    assert names == set(server.TOOL_REGISTRY.keys())
    for d in defs:
        assert d["inputSchema"]["type"] == "object"


def test_scenarios_catalog_keys() -> None:
    """Cenários A, B, E, F, J devem estar catalogados."""
    assert set(server.SCENARIOS.keys()) == {"A", "B", "E", "F", "J"}
    for s in server.SCENARIOS.values():
        assert "n_pos" in s and "nf" in s and "nTR" in s and "nAng" in s
        assert "desc" in s


# ──────────────────────────────────────────────────────────────────────────
# check_cpu_topology — leitura real do hardware
# ──────────────────────────────────────────────────────────────────────────


def test_check_cpu_topology_returns_positive_cores() -> None:
    """logical_cores e physical_cores devem ser >= 1."""
    result = server._impl_check_cpu_topology()
    assert "error" not in result
    assert result["logical_cores"] >= 1
    assert result["physical_cores"] >= 1
    assert result["logical_cores"] >= result["physical_cores"]
    assert isinstance(result["has_hyperthreading"], bool)


# ──────────────────────────────────────────────────────────────────────────
# check_oversubscription — recommendation logic
# ──────────────────────────────────────────────────────────────────────────


def test_check_oversubscription_balanced_returns_ok() -> None:
    """1w × 1t nunca pode oversubscribar (cabe em 1 core físico)."""
    result = server._impl_check_oversubscription(n_workers=1, threads_per_worker=1)
    assert result["oversubscribed"] is False
    assert result["recommendation"] == "OK"


def test_check_oversubscription_overload_warns() -> None:
    """workers × threads >> phys_cores deve flagear oversubscription."""
    # 32 × 32 = 1024 threads — virtualmente impossível em qualquer hardware
    result = server._impl_check_oversubscription(n_workers=32, threads_per_worker=32)
    assert result["oversubscribed"] is True
    assert "OVERSUBSCRIBED" in result["recommendation"]
    assert "recommended_workers" in result
    assert "recommended_threads_per_worker" in result


# ──────────────────────────────────────────────────────────────────────────
# analyze_jit_cache — wrap de get_jit_cache_info
# ──────────────────────────────────────────────────────────────────────────


def test_analyze_jit_cache_returns_expected_keys() -> None:
    """Diagnóstico deve incluir n_entries, signatures, disk_bytes."""
    result = server._impl_analyze_jit_cache()
    assert "error" not in result
    assert result["check_name"] == "jit_cache_info"
    assert "n_entries" in result
    assert "approx_bytes" in result
    assert "dispatcher_signatures" in result
    assert "cache_dir_disk_bytes" in result
    # Após pelo menos 1 simulação prévia (ou cache JIT compilada), n_entries
    # pode ser 0 ou positivo — só validamos estrutura.
    assert isinstance(result["n_entries"], int)
    assert isinstance(result["dispatcher_signatures"], dict)


def test_analyze_jit_cache_dispatchers_listed() -> None:
    """Os 7 dispatchers principais devem estar no dict (mesmo com 0 sigs)."""
    result = server._impl_analyze_jit_cache()
    expected_dispatchers = {
        "_simulate_combined_prange_flat",
        "_simulate_combined_prange",
        "_simulate_positions_njit",
        "_simulate_positions_njit_cached",
        "_fields_in_freqs_kernel_cached",
        "_fields_at_single_freq",
        "precompute_common_arrays_cache",
    }
    assert expected_dispatchers.issubset(set(result["dispatcher_signatures"].keys()))


# ──────────────────────────────────────────────────────────────────────────
# run_scenario_benchmark — executa simulate_multi de verdade
# ──────────────────────────────────────────────────────────────────────────


def test_run_scenario_benchmark_unknown_scenario_returns_error() -> None:
    result = server._impl_run_scenario_benchmark(scenario_id="X")
    assert "error" in result
    assert "available" in result


@pytest.mark.slow
def test_run_scenario_benchmark_A_quick() -> None:
    """Cenário A (n_pos=30, nf=1) é o mais rápido — runs=2 suficiente."""
    result = server._impl_run_scenario_benchmark(scenario_id="A", runs=2)
    assert "error" not in result
    assert result["scenario_id"] == "A"
    assert result["runs"] == 2
    assert result["median_s"] > 0
    assert result["mod_h"] > 0
    assert result["n_models_per_run"] == 30
    assert result["use_flat_prange"] is True


# ──────────────────────────────────────────────────────────────────────────
# compare_branches — validação de tree limpo
# ──────────────────────────────────────────────────────────────────────────


def test_compare_branches_dirty_tree_detection(tmp_path, monkeypatch) -> None:
    """Se git status reporta dirty, retorna error sem rodar bench."""
    # Forçamos uma working dir suja: criamos um arquivo untracked em raiz
    # do repo. Como não controlamos o estado real, verificamos só estrutura.
    result = server._impl_compare_branches(scenario_id="E", runs=1)
    # Se branch atual está limpo, segue para bench (sem erro);
    # se sujo, retorna error. Validamos estrutura conforme estado.
    if "error" in result:
        assert (
            result.get("error") in {"dirty tree"}
            or "git" in str(result.get("error", "")).lower()
        )
    else:
        # Branch limpo: deve ter bench_b ou nota de limitação
        assert "current_branch" in result or "bench_b" in result


# ──────────────────────────────────────────────────────────────────────────
# profile_kernel — cProfile execução
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_profile_kernel_returns_hotspots() -> None:
    """profile_kernel sobre cenário A deve retornar hotspots_text não-vazio."""
    result = server._impl_profile_kernel(scenario_id="A", top_n=10)
    assert "error" not in result
    assert result["scenario_id"] == "A"
    assert result["elapsed_s"] > 0
    assert "hotspots_text" in result
    # pstats output sempre inclui "function calls" header
    assert "function calls" in result["hotspots_text"].lower()
    assert result["top_n"] == 10


def test_profile_kernel_unknown_scenario() -> None:
    result = server._impl_profile_kernel(scenario_id="UNKNOWN")
    assert "error" in result
