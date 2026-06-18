# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_cli_warmup_jax.py                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  PR — geosteering-warmup --jax/--jax-only/--jax-auto (op 4)               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-18                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Flags JAX ADITIVAS do geosteering-warmup: aquecem a forma canônica do   ║
# ║    SM no cache XLA (compartilhado com o worker persistente + o CLI). Testes ║
# ║    diretos via main() com spies (sem warmup real): default = só Numba       ║
# ║    (no-regression); --jax = ambos; --jax-only = só JAX; --jax-auto = JAX só  ║
# ║    com GPU; mutuamente exclusivos; shapes canônicos corretos.               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes das flags JAX do ``geosteering-warmup`` (op 4)."""

from __future__ import annotations

import pytest

import geosteering_ai.cli._main as cli_main
import geosteering_ai.cli.warmup as w


@pytest.fixture
def spies(monkeypatch):
    """Substitui os warmups reais por spies (registram a ordem de chamada)."""
    calls: list = []
    monkeypatch.setattr(
        cli_main, "_warmup_numba_tier2_sync", lambda **k: calls.append("numba")
    )
    monkeypatch.setattr(
        w,
        "_warmup_jax_canonical_sm",
        lambda **k: (calls.append("jax"), {"skipped": False})[1],
    )
    return calls


# ════════════════════════════════════════════════════════════════════════════
# Parser — flags presentes + mutuamente exclusivas
# ════════════════════════════════════════════════════════════════════════════
def test_help_lists_jax_flags():
    """--help mostra as 3 flags JAX."""
    h = w.build_parser().format_help()
    assert "--jax" in h and "--jax-only" in h and "--jax-auto" in h


@pytest.mark.parametrize(
    "argv",
    [["--jax", "--jax-only"], ["--jax", "--jax-auto"], ["--jax-only", "--jax-auto"]],
)
def test_jax_flags_mutually_exclusive(argv):
    """Qualquer par de flags JAX → argparse sai com código 2."""
    with pytest.raises(SystemExit) as exc:
        w.build_parser().parse_args(argv)
    assert exc.value.code == 2


# ════════════════════════════════════════════════════════════════════════════
# main() — seleção de backends (no-regression + flags)
# ════════════════════════════════════════════════════════════════════════════
def test_default_warms_numba_not_jax(spies):
    """Sem flags → SÓ Numba (byte-for-byte com o comportamento legado v2.44)."""
    assert w.main([]) == 0
    assert spies == ["numba"]  # JAX NÃO é aquecido por default


def test_jax_warms_both(spies):
    """--jax → Numba E JAX (Numba primeiro)."""
    assert w.main(["--jax"]) == 0
    assert spies == ["numba", "jax"]


def test_jax_only_skips_numba(spies):
    """--jax-only → SÓ JAX (pula o Numba)."""
    assert w.main(["--jax-only"]) == 0
    assert spies == ["jax"]


def test_jax_auto_noop_without_gpu(spies, monkeypatch):
    """--jax-auto sem GPU visível → SÓ Numba (JAX pulado, exit 0 — ideal p/ CI CPU)."""
    monkeypatch.setattr(w, "_gpu_visible", lambda: False)
    assert w.main(["--jax-auto"]) == 0
    assert spies == ["numba"]  # JAX pulado (sem GPU)


def test_jax_auto_warms_with_gpu(spies, monkeypatch):
    """--jax-auto com GPU visível → Numba + JAX."""
    monkeypatch.setattr(w, "_gpu_visible", lambda: True)
    assert w.main(["--jax-auto"]) == 0
    assert spies == ["numba", "jax"]


# ════════════════════════════════════════════════════════════════════════════
# main() — shapes canônicos passados a warmup_jax_simulator (guarda contra drift)
# ════════════════════════════════════════════════════════════════════════════
def test_jax_canonical_shapes(monkeypatch):
    """--jax-only chama warmup_jax_simulator com a forma CANÔNICA do SM."""
    try:
        import geosteering_ai.simulation._jax.warmup as wj
    except Exception:  # noqa: BLE001 — jax ausente neste ambiente
        pytest.skip("módulo _jax indisponível (jax ausente)")

    captured: dict = {}
    monkeypatch.setattr(
        wj,
        "warmup_jax_simulator",
        lambda **k: (captured.update(k), {"skipped": False})[1],
    )
    monkeypatch.setattr(cli_main, "_warmup_numba_tier2_sync", lambda **k: None)
    assert w.main(["--jax-only"]) == 0
    assert captured["n_layers"] == 20
    assert captured["n_positions"] == 600
    assert captured["n_models"] == 64
    assert captured["hankel_filter"] == "werthmuller_201pt"
    assert captured["complex_dtype"] == "complex128"
    assert captured["jax_strategy"] == "bucketed"


def test_jax_skipped_runtime_returns_zero(spies, monkeypatch):
    """warmup_jax_simulator retornando {'skipped': True} (HAS_JAX False) → exit 0 (no-op)."""
    monkeypatch.setattr(
        w,
        "_warmup_jax_canonical_sm",
        lambda **k: (spies.append("jax"), {"skipped": True, "reason": "jax_absent"})[1],
    )
    assert w.main(["--jax-only"]) == 0  # observabilidade, não gating
