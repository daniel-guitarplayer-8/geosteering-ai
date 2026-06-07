# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_cli_warmup_entry_point.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP — entry point `geosteering-warmup` (Sprint v2.32)  ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-13                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida o entry point standalone `geosteering-warmup` introduzido na    ║
# ║    Sprint v2.32:                                                          ║
# ║                                                                           ║
# ║    1. `--help` retorna 0 e exibe descrição esperada.                      ║
# ║    2. `--version` imprime a versão correta (v2.32) no stdout.             ║
# ║    3. Execução completa (warmup real) retorna 0 dentro de timeout.       ║
# ║       Marcado `@pytest.mark.slow` porque envolve compilação JIT cold.     ║
# ║    4. `pyproject.toml` declara o entry point com path correto.           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do entry point ``geosteering-warmup`` (Sprint v2.32)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


def test_warmup_help_does_not_error() -> None:
    """`python -m geosteering_ai.cli.warmup --help` retorna 0 sem erros."""
    proc = subprocess.run(
        [sys.executable, "-m", "geosteering_ai.cli.warmup", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, f"--help retornou {proc.returncode}: {proc.stderr}"
    assert "geosteering-warmup" in proc.stdout
    assert "--verbose" in proc.stdout
    assert "--version" in proc.stdout


def test_warmup_version_prints_current_version() -> None:
    """`geosteering-warmup --version` imprime versão atual (≥ v2.32) no stdout."""
    proc = subprocess.run(
        [sys.executable, "-m", "geosteering_ai.cli.warmup", "--version"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, f"--version retornou {proc.returncode}"
    # Versão evoluiu de v2.32 → v2.33/34/35 ao longo do bundle Sprint v2.33-v2.35.
    # Verificação robusta: prefixo de versão presente, não string específica.
    assert "Geosteering AI CLI v" in proc.stdout, (
        f"Esperado prefixo 'Geosteering AI CLI v' em stdout; "
        f"recebido: {proc.stdout!r}"
    )


@pytest.mark.slow
def test_warmup_runs_and_succeeds() -> None:
    """`geosteering-warmup` (sem flags) executa warmup completo e retorna 0.

    Marcado slow porque envolve compilação JIT cold (~10-60s em ambiente
    fresco). Timeout de 300s é conservador para CI/macOS sem cache aquecido.

    O teste verifica:
      - Exit code 0 (warmup completou sem propagar exceção).
      - Mensagem "OK (X.Xs)" no stdout.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "geosteering_ai.cli.warmup"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    assert (
        proc.returncode == 0
    ), f"warmup retornou {proc.returncode}: stderr={proc.stderr[:500]}"
    assert "Warming up Geosteering AI" in proc.stdout
    assert (
        "OK (" in proc.stdout
    ), f"Esperado 'OK (...)' em stdout; recebido: {proc.stdout!r}"


def test_warmup_entry_point_declared_in_pyproject() -> None:
    """pyproject.toml declara o entry point `geosteering-warmup`."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'geosteering-warmup = "geosteering_ai.cli.warmup:main"' in pyproject
    ), "entry point geosteering-warmup ausente em pyproject.toml [project.scripts]"


# ════════════════════════════════════════════════════════════════════════
# Sprint v2.51 — flags --jax/--auto + lógica de lift de JAX_PLATFORMS
# (testa a parte de MAIOR RISCO: o lift do force-CPU p/ GPU; sem jax real)
# ════════════════════════════════════════════════════════════════════════

import argparse  # noqa: E402

from geosteering_ai.cli import warmup as _warmup_mod  # noqa: E402


def test_warmup_parser_has_jax_flags() -> None:
    """build_parser expõe as flags v2.51 (--jax/--gpu/--auto + overrides)."""
    parser = _warmup_mod.build_parser()
    args = parser.parse_args(["--jax", "--jax-n-pos", "300", "--jax-dtype", "complex64"])
    assert args.jax is True
    assert args.jax_n_pos == 300
    assert args.jax_dtype == "complex64"
    # --gpu é alias de --jax (mesmo dest).
    assert parser.parse_args(["--gpu"]).jax is True
    assert parser.parse_args(["--auto"]).auto is True
    # Sem flags → tudo False/default.
    base = parser.parse_args([])
    assert base.jax is False and base.auto is False


def _ns(jax=False, auto=False) -> argparse.Namespace:
    return argparse.Namespace(jax=jax, auto=auto)


def test_resolve_lift_jax_when_user_unset_and_cpu() -> None:
    """--jax + usuário NÃO setou + default cpu → liberar a GPU (lift=True)."""
    want, lift = _warmup_mod._resolve_jax_warmup(
        _ns(jax=True), user_jax_platforms=None, env={"JAX_PLATFORMS": "cpu"}
    )
    assert want is True and lift is True


def test_resolve_no_lift_when_user_set_platforms() -> None:
    """--jax mas usuário setou JAX_PLATFORMS=cpu explicitamente → NÃO lifta."""
    want, lift = _warmup_mod._resolve_jax_warmup(
        _ns(jax=True), user_jax_platforms="cpu", env={"JAX_PLATFORMS": "cpu"}
    )
    assert want is True and lift is False


def test_resolve_no_lift_when_env_not_cpu() -> None:
    """--jax + env já em GPU (não 'cpu') → nada a liftar."""
    want, lift = _warmup_mod._resolve_jax_warmup(
        _ns(jax=True), user_jax_platforms=None, env={"JAX_PLATFORMS": "cuda"}
    )
    assert want is True and lift is False


def test_resolve_auto_without_gpu(monkeypatch) -> None:
    """--auto sem GPU (_gpu_available→False) → want_jax False (skip JAX warmup)."""
    monkeypatch.setattr(_warmup_mod, "_gpu_available", lambda: False)
    want, lift = _warmup_mod._resolve_jax_warmup(
        _ns(auto=True), user_jax_platforms=None, env={"JAX_PLATFORMS": "cpu"}
    )
    assert want is False and lift is False


def test_resolve_auto_with_gpu(monkeypatch) -> None:
    """--auto + GPU presente (mock) → habilita want_jax + lift."""
    monkeypatch.setattr(_warmup_mod, "_gpu_available", lambda: True)
    want, lift = _warmup_mod._resolve_jax_warmup(
        _ns(auto=True), user_jax_platforms=None, env={"JAX_PLATFORMS": "cpu"}
    )
    assert want is True and lift is True


def test_resolve_neither_flag() -> None:
    """Sem --jax nem --auto → want_jax False (comportamento legado Numba-only)."""
    want, lift = _warmup_mod._resolve_jax_warmup(
        _ns(), user_jax_platforms=None, env={"JAX_PLATFORMS": "cpu"}
    )
    assert want is False and lift is False


def test_gpu_available_returns_bool() -> None:
    """_gpu_available retorna bool (sem levantar), via nvidia-smi gracioso."""
    assert isinstance(_warmup_mod._gpu_available(), bool)


# ════════════════════════════════════════════════════════════════════════
# Sprint v2.52 — flags --numba do warmup Numba CPU de cobertura completa
# ════════════════════════════════════════════════════════════════════════


def test_warmup_parser_has_numba_flags() -> None:
    """build_parser expõe as flags v2.52 (--numba/--no-numba + overrides)."""
    parser = _warmup_mod.build_parser()
    # Default: --numba ligado.
    base = parser.parse_args([])
    assert base.numba is True
    assert base.numba_n_pos == 200
    assert base.numba_n_layers == 5
    assert base.numba_threads is None
    # Overrides + --no-numba.
    args = parser.parse_args(
        ["--no-numba", "--numba-n-pos", "300", "--numba-n-layers", "8"]
    )
    assert args.numba is False
    assert args.numba_n_pos == 300
    assert args.numba_n_layers == 8
    assert parser.parse_args(["--numba-threads", "4"]).numba_threads == 4
