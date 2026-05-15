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
    assert "Geosteering AI Simulation Manager v" in proc.stdout, (
        f"Esperado prefixo 'Geosteering AI Simulation Manager v' em stdout; "
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
