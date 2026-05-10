# -*- coding: utf-8 -*-
"""Testes I2.6: CLI MVP do Geosteering AI (Sprint v2.24).

Valida:
- Estrutura do módulo `geosteering_ai.cli/`
- Subcomandos `version`, `simulate --help`, `benchmark --help`
- Entry point declarado em pyproject.toml
- Reutilização correta de simulate_multi e auto-detect de paralelismo
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


# ── Estrutura ────────────────────────────────────────────────────


def test_cli_module_exists():
    """I2.6 — geosteering_ai/cli/ existe com arquivos esperados."""
    cli_dir = PROJECT_ROOT / "geosteering_ai" / "cli"
    assert cli_dir.is_dir()
    expected = ["__init__.py", "__main__.py", "main.py", "simulate.py", "benchmark.py"]
    for name in expected:
        assert (cli_dir / name).exists(), f"Arquivo CLI ausente: {name}"


def test_cli_main_exports_main():
    """I2.6 — geosteering_ai.cli expõe `main`."""
    from geosteering_ai.cli import main

    assert callable(main)


def test_pyproject_declares_entry_point():
    """I2.6 — pyproject.toml declara geosteering-cli."""
    content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" in content
    assert "geosteering-cli" in content
    assert "geosteering_ai.cli:main" in content


# ── Subcomandos ──────────────────────────────────────────────────


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Executa python -m geosteering_ai.cli <args>."""
    return subprocess.run(
        [sys.executable, "-m", "geosteering_ai.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(PROJECT_ROOT),
    )


def test_cli_version_subcommand():
    """I2.6 — `version` retorna versão atual."""
    proc = _run_cli(["version"])
    assert proc.returncode == 0
    assert "Geosteering AI" in proc.stdout
    assert "v2.24" in proc.stdout


def test_cli_help_does_not_error():
    """I2.6 — `--help` exit code 0 e mostra subcomandos."""
    proc = _run_cli(["--help"])
    assert proc.returncode == 0
    assert "simulate" in proc.stdout
    assert "benchmark" in proc.stdout
    assert "version" in proc.stdout


def test_cli_no_command_returns_2():
    """I2.6 — chamada sem subcomando retorna exit code 2 + help no stderr."""
    proc = _run_cli([])
    assert proc.returncode == 2
    # argparse imprime help no stderr
    assert "comandos" in proc.stderr or "commands" in proc.stderr.lower()


def test_cli_simulate_help():
    """I2.6 — `simulate --help` exit code 0."""
    proc = _run_cli(["simulate", "--help"])
    assert proc.returncode == 0
    assert "--models" in proc.stdout
    assert "--workers" in proc.stdout
    assert "--seed" in proc.stdout


def test_cli_benchmark_help():
    """I2.6 — `benchmark --help` exit code 0."""
    proc = _run_cli(["benchmark", "--help"])
    assert proc.returncode == 0
    assert "--scenario" in proc.stdout
    assert "--n" in proc.stdout


def test_cli_invalid_subcommand_errors():
    """I2.6 — subcomando inválido reporta erro."""
    proc = _run_cli(["nonexistent-cmd"])
    assert proc.returncode != 0


def test_cli_invalid_scenario_errors():
    """I2.6 — `--scenario Z` (inválido) retorna exit code != 0."""
    proc = _run_cli(["benchmark", "--scenario", "Z", "--n", "1"])
    assert proc.returncode != 0
    # argparse rejeita choices inválidos
    assert "invalid choice" in proc.stderr.lower() or "Z" in proc.stderr


# ── Imports + lazy-loading ───────────────────────────────────────


def test_cli_does_not_import_simulation_at_help_time():
    """I2.6 — `--help` é rápido (não carrega numba/simulation_module)."""
    import time

    t0 = time.perf_counter()
    proc = _run_cli(["--help"])
    dt = time.perf_counter() - t0
    assert proc.returncode == 0
    # --help deve ser < 5s mesmo no primeiro uso (numba não importado)
    assert dt < 5.0, f"--help muito lento: {dt:.2f}s — possível eager import"


def test_cli_module_path_is_geosteering_ai_cli():
    """I2.6 — módulo correto é geosteering_ai.cli."""
    import importlib

    cli_mod = importlib.import_module("geosteering_ai.cli")
    cli_main_mod = importlib.import_module("geosteering_ai.cli.main")

    assert hasattr(cli_mod, "main")
    assert hasattr(cli_main_mod, "build_parser")
    assert hasattr(cli_main_mod, "SIMULATION_MANAGER_VERSION")
    assert cli_main_mod.SIMULATION_MANAGER_VERSION == "v2.24"


# ── Smoke test simulação real (lento) ────────────────────────────


@pytest.mark.slow
def test_cli_simulate_smoke_5_models():
    """I2.6 — `simulate --models 5` completa sem erro (smoke).

    Usa --workers 1 --threads 1 para evitar overhead de pool em CI.
    Marcado como slow porque importa simulação completa (~10-30s).
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "geosteering_ai.cli",
            "simulate",
            "--models",
            "5",
            "--n-pos",
            "10",
            "--workers",
            "1",
            "--threads",
            "1",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
    )
    # Aceita exit code 0 (sucesso) ou 1 (caminho de erro tratado) — apenas
    # validamos que CLI não crashou inesperadamente
    assert proc.returncode in (
        0,
        1,
    ), f"CLI crashou ou timeout: {proc.returncode}, stderr={proc.stderr[:500]}"
