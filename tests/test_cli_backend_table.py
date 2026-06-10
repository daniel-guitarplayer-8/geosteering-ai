# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_cli_backend_table.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP — backend resolver + tabela ASCII (Sprint v2.53)   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-02                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Testa (rápido, sem JIT):                                               ║
# ║      • ``_backend.resolve_backend`` — numba/jax + fallback gracioso        ║
# ║      • ``_table.render_kv_table`` — Unicode + fallback ASCII + truncamento ║
# ║      • ``_table.build_result_rows`` — linhas numba vs jax (GPU/strategy)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do resolver de backend + renderizador de tabela ASCII (v2.53)."""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.cli._backend import resolve_backend
from geosteering_ai.cli._table import build_result_rows, render_kv_table

# ════════════════════════════════════════════════════════════════════════
# resolve_backend
# ════════════════════════════════════════════════════════════════════════


def test_resolve_numba_is_cpu() -> None:
    """numba → (numba, cpu) sempre, sem sondar GPU."""
    assert resolve_backend("numba") == ("numba", "cpu")


def test_resolve_jax_with_gpu(monkeypatch) -> None:
    """jax + GPU JAX visível (mock) → (jax, gpu)."""
    import geosteering_ai.simulation.dispatch as disp

    monkeypatch.setattr(disp, "_jax_gpu_available", lambda: True)
    assert resolve_backend("jax") == ("jax", "gpu")


def test_resolve_jax_without_gpu_falls_back(monkeypatch) -> None:
    """jax SEM GPU JAX (mock) → fallback gracioso (numba, cpu)."""
    import geosteering_ai.simulation.dispatch as disp

    monkeypatch.setattr(disp, "_jax_gpu_available", lambda: False)
    assert resolve_backend("jax", quiet=True) == ("numba", "cpu")


def test_resolve_invalid_raises() -> None:
    """Backend desconhecido → ValueError."""
    with pytest.raises(ValueError, match="backend inválido"):
        resolve_backend("cuda")


# ════════════════════════════════════════════════════════════════════════
# render_kv_table
# ════════════════════════════════════════════════════════════════════════


def test_render_unicode_has_box_and_content() -> None:
    """Tabela Unicode contém bordas box-drawing + título + chave/valor."""
    out = render_kv_table("TÍTULO", [("backend", "numba"), ("device", "cpu")])
    assert "┌" in out and "┐" in out and "└" in out and "┘" in out
    assert "TÍTULO" in out
    assert "backend" in out and "numba" in out
    assert "device" in out and "cpu" in out


def test_render_ascii_fallback(monkeypatch) -> None:
    """``GEOSTEERING_ASCII_TABLE`` força bordas ASCII (+ - |) sem Unicode."""
    monkeypatch.setenv("GEOSTEERING_ASCII_TABLE", "1")
    out = render_kv_table("T", [("k", "v")])
    assert "+" in out and "-" in out and "|" in out
    assert "┌" not in out and "│" not in out


def test_render_truncates_long_value() -> None:
    """Valor maior que ``max_value_width`` é truncado com reticências."""
    long_val = "x" * 200
    out = render_kv_table("T", [("k", long_val)], max_value_width=20)
    assert "…" in out
    # Nenhuma linha excede o esperado (20 + bordas/padding) — sem o valor cru.
    assert long_val not in out


def test_render_empty_rows_is_title_only() -> None:
    """Lista vazia → tabela só com a faixa de título (não levanta)."""
    out = render_kv_table("SÓ TÍTULO", [])
    assert "SÓ TÍTULO" in out
    assert out.count("\n") >= 3  # topo + título + separador + base


# ════════════════════════════════════════════════════════════════════════
# build_result_rows
# ════════════════════════════════════════════════════════════════════════

_HW = {
    "cpu_model": "AMD Ryzen Threadripper",
    "cpu_physical": 32,
    "cpu_logical": 64,
    "ram_gb": 251.2,
    "numba_threads": 64,
    "gpu_name": "NVIDIA RTX A6000",
    "gpu_vram_gb": 48.0,
}


def _rows_dict(stats: dict) -> dict:
    """Helper: build_result_rows → dict chave→valor para asserts fáceis."""
    return dict(build_result_rows(stats, _HW))


def test_build_rows_numba_has_workers_not_jax() -> None:
    """Backend numba: linha workers×threads presente; SEM jax_strategy/GPU."""
    d = _rows_dict(
        {
            "backend": "numba",
            "device": "cpu",
            "throughput_mod_h": 234665.0,
            "elapsed_s": 1.5,
            "n_models": 100,
            "n_pos": 600,
            "n_freqs": 1,
            "n_dips": 1,
            "n_trs": 1,
            "workers": None,
            "threads": None,
            "dtype": "complex128",
            "nan_count": 0,
            "inf_count": 0,
            "all_finite": True,
        }
    )
    assert d["backend"] == "numba"
    assert "workers × threads" in d
    assert "throughput" in d and "mod/h" in d["throughput"]
    assert "jax_strategy" not in d
    assert "GPU" not in d
    assert d["finitude OK"].startswith("sim")


def test_build_rows_jax_has_strategy_and_gpu() -> None:
    """Backend jax: linhas jax_strategy + GPU + VRAM presentes."""
    d = _rows_dict(
        {
            "backend": "jax",
            "device": "gpu",
            "throughput_mod_h": 500000.0,
            "elapsed_s": 0.7,
            "n_models": 256,
            "n_pos": 600,
            "n_freqs": 1,
            "n_dips": 1,
            "n_trs": 1,
            "jax_strategy": "bucketed",
            "n_geometry_groups": 3,
            "dtype": "complex64",
            "nan_count": 0,
            "inf_count": 0,
            "all_finite": True,
        }
    )
    assert d["backend"] == "jax"
    assert d["jax_strategy"] == "bucketed"
    assert d["GPU"] == "NVIDIA RTX A6000"
    assert "VRAM" in d
    assert d["n_geometry_groups"] == "3"
    assert "workers × threads" not in d


def test_build_rows_finitude_failure_flagged() -> None:
    """all_finite False → linha 'finitude OK' marca falha (✗)."""
    d = _rows_dict(
        {
            "backend": "numba",
            "n_models": 1,
            "n_pos": 1,
            "n_freqs": 1,
            "n_dips": 1,
            "n_trs": 1,
            "nan_count": 2,
            "inf_count": 1,
            "all_finite": False,
        }
    )
    assert d["NaNs"] == "2"
    assert d["Infs"] == "1"
    assert "✗" in d["finitude OK"]


def test_build_rows_shows_fallback_reason() -> None:
    """stats['reason'] presente → linha 'motivo (fallback)' aparece na tabela."""
    d = _rows_dict(
        {
            "backend": "numba",
            "device": "cpu",
            "reason": "jax→numba: geometria heterogênea (500/500)",
            "n_models": 500,
            "n_pos": 600,
            "n_freqs": 1,
            "n_dips": 1,
            "n_trs": 1,
            "nan_count": 0,
            "inf_count": 0,
            "all_finite": True,
        }
    )
    assert "motivo (fallback)" in d
    assert "heterogênea" in d["motivo (fallback)"]


# ════════════════════════════════════════════════════════════════════════
# Sprint v2.54 — fallback de agrupabilidade do dispatcher (correção do hang)
# ════════════════════════════════════════════════════════════════════════


def _distinct_esp(n: int) -> np.ndarray:
    """n geometrias DISTINTAS (esp aleatório contínuo) → n grupos."""
    return np.random.default_rng(0).uniform(2.0, 10.0, size=(n, 3))


def test_resolve_jax_nongroupable_falls_back_to_numba() -> None:
    """jax forçado + geometria não-agrupável + numba_fallback=True → Numba.

    Esta é a CORREÇÃO do hang: com 64 geometrias distintas (não-agrupável,
    64 > 0.5·64), o dispatcher cai p/ Numba em vez do JAX-grouped degenerado.
    """
    from geosteering_ai.simulation.dispatch import _resolve_backend

    eff, reason, n_groups = _resolve_backend(
        "jax", 64, _distinct_esp(64), numba_fallback=True, n_models_gpu_threshold=32
    )
    assert eff == "numba"
    assert n_groups == 64
    assert "agrup" in reason.lower()  # motivo menciona agrupabilidade


def test_resolve_jax_nongroupable_forces_jax_when_no_fallback() -> None:
    """jax forçado + numba_fallback=False → JAX mesmo não-agrupável.

    Preserva ``--compare-backends`` (DEVE forçar JAX p/ medir paridade real).
    """
    from geosteering_ai.simulation.dispatch import _resolve_backend

    eff, _reason, _n = _resolve_backend(
        "jax", 64, _distinct_esp(64), numba_fallback=False, n_models_gpu_threshold=32
    )
    assert eff == "jax"


def test_compare_backends_forces_jax_on_both(monkeypatch) -> None:
    """FURO 1: run_compare_backends passa numba_fallback=False a run_once.

    Sem isso, o compare mediria Numba×Numba (speedup≈1.0× espúrio, paridade≈0).
    Espia ``run_once`` e confirma ``numba_fallback=False`` em AMBOS os backends.
    """
    import geosteering_ai.cli._exec as ex

    seen: dict[str, object] = {}

    def _spy(be, models, positions_z, **kw):
        seen[be] = kw.get("numba_fallback")
        H = np.ones((len(models), 1, 1, 4, 1, 9), dtype=np.complex128)
        return H, 0.01, (1 if be == "jax" else None), be, None

    monkeypatch.setattr(ex, "run_once", _spy)
    rc = ex.run_compare_backends(
        models=[{"rho_h": np.ones(5), "rho_v": np.ones(5), "esp": np.ones(3)}],
        positions_z=np.linspace(-5.0, 5.0, 4),
        frequencies_hz=[20000.0],
        dip_degs=[0.0],
        tr_spacings_m=[1.0],
        n_pos=4,
        workers=None,
        threads=None,
        dtype="complex128",
        jax_strategy="bucketed",
        warmup=False,
        as_json=False,
        quiet=True,
        title="t",
    )
    assert rc == 0
    assert seen.get("numba") is False  # compare NÃO deixa cair p/ numba
    assert seen.get("jax") is False
