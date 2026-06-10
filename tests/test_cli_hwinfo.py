# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_cli_hwinfo.py                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP — coleta de hardware (Sprint v2.53)                ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-02                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Garante que ``collect_hardware_info`` é defensivo: nunca levanta,       ║
# ║    sempre retorna as chaves esperadas (CPU/RAM/threads) e adiciona as      ║
# ║    chaves de GPU somente quando ``want_gpu=True``.                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da coleta defensiva de specs de hardware (v2.53)."""

from __future__ import annotations

from geosteering_ai.cli._hwinfo import collect_hardware_info

_BASE_KEYS = {"cpu_model", "cpu_physical", "cpu_logical", "ram_gb", "numba_threads"}
_GPU_KEYS = {"gpu_name", "gpu_vram_gb", "jax_devices"}


def test_collect_base_keys_present() -> None:
    """want_gpu=False → chaves base presentes, chaves GPU ausentes."""
    hw = collect_hardware_info(want_gpu=False)
    assert _BASE_KEYS.issubset(hw)
    assert not (_GPU_KEYS & set(hw))


def test_collect_want_gpu_adds_gpu_keys() -> None:
    """want_gpu=True → chaves GPU presentes (valores podem ser None)."""
    hw = collect_hardware_info(want_gpu=True)
    assert _GPU_KEYS.issubset(hw)


def test_cpu_model_is_str() -> None:
    """``cpu_model`` é sempre str (nunca None) — fallback 'desconhecido'."""
    hw = collect_hardware_info()
    assert isinstance(hw["cpu_model"], str)
    assert hw["cpu_model"]


def test_cpu_logical_positive_or_none() -> None:
    """``cpu_logical`` é int positivo ou None — nunca negativo."""
    hw = collect_hardware_info()
    val = hw["cpu_logical"]
    assert val is None or (isinstance(val, int) and val >= 1)


def test_ram_gb_float_or_none() -> None:
    """``ram_gb`` é float (>0) ou None."""
    hw = collect_hardware_info()
    val = hw["ram_gb"]
    assert val is None or (isinstance(val, float) and val > 0.0)


def test_never_raises_even_if_probes_fail(monkeypatch) -> None:
    """Mesmo com /proc e subprocess quebrados, retorna dict sem levantar."""
    import geosteering_ai.cli._hwinfo as hwmod

    def _boom(*_a, **_k):
        raise OSError("probe indisponível")

    # Sabota /proc + subprocess + topologia de CPU.
    monkeypatch.setattr("builtins.open", _boom)
    monkeypatch.setattr(hwmod.subprocess, "run", _boom)
    hw = hwmod.collect_hardware_info(want_gpu=True)
    assert isinstance(hw, dict)
    assert _BASE_KEYS.issubset(hw)
    # cpu_model cai para 'desconhecido' (platform.processor pode até responder).
    assert isinstance(hw["cpu_model"], str)
