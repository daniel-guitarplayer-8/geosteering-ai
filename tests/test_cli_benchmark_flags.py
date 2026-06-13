# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_cli_benchmark_flags.py                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI benchmark — flags ricas (paridade c/ simulate)         ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-10                                                 ║
# ║  Status      : Produção (guarda de superfície CLI)                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Guarda das flags ricas recuperadas no ``benchmark`` (item 3 da triagem).

Fecha o gap simulate↔benchmark: ``--geometry``/``--n-geometries``/``--dtype``/
``--jax-strategy``/``--jax-chunk-size``/``--repeat`` + ``--list-scenarios``,
SEM o rename de versão (a constante ``SIMULATION_MANAGER_VERSION`` permanece
``v2.37``). Testes RÁPIDOS — exercitam a superfície (parser + ``--list-scenarios``
+ ``_build_models`` geometria) em-processo, sem disparar o JIT pesado da simulação.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


def test_list_scenarios_in_process(capsys):
    """``_list_scenarios`` imprime os 8 cenários (A..H) + dimensões e retorna 0."""
    from geosteering_ai.cli.benchmark import _list_scenarios

    rc = _list_scenarios()
    out = capsys.readouterr().out
    assert rc == 0
    for sid in ("A", "B", "C", "D", "E", "F", "G", "H"):
        assert f"  {sid} " in out or f"  {sid}  " in out, f"cenário {sid} ausente"
    assert "combos/pos" in out
    assert "512" in out  # Cenário H = 8×8×8


def test_build_models_geometry_templates_groups_esp():
    """``--geometry templates`` compartilha a ``esp`` (agrupável p/ JAX).

    ``per-model`` gera ``esp`` única por modelo (n_models geometrias distintas);
    ``templates`` replica K geometrias → muito menos geometrias distintas. Guarda
    que o wiring de ``geometry`` em ``_build_models`` realmente muda a amostragem.
    """
    from geosteering_ai.cli.benchmark import _build_models

    n = 16
    per_model = _build_models(n, geometry="per-model")
    templates = _build_models(n, geometry="templates", n_geometries=2)
    assert len(per_model) == n and len(templates) == n

    def _n_unique_esp(models: list[dict]) -> int:
        return len({np.asarray(m["esp"]).tobytes() for m in models})

    # per-model → ~n geometrias distintas; templates(K=2) → no máx. 2.
    assert _n_unique_esp(templates) <= 2
    assert _n_unique_esp(per_model) > _n_unique_esp(templates)


def test_benchmark_help_shows_rich_flags():
    """``benchmark --help`` expõe as flags ricas recuperadas (subprocesso)."""
    proc = subprocess.run(
        [sys.executable, "-m", "geosteering_ai.cli", "benchmark", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0
    for flag in (
        "--geometry",
        "--n-geometries",
        "--dtype",
        "--jax-strategy",
        "--jax-chunk-size",
        "--repeat",
        "--list-scenarios",
    ):
        assert flag in proc.stdout, f"flag {flag} ausente no benchmark --help"


def test_benchmark_list_scenarios_subprocess():
    """``benchmark --list-scenarios`` roda rápido (sem simular) e sai 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "geosteering_ai.cli", "benchmark", "--list-scenarios"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0
    assert "Cenários de benchmark" in proc.stdout
    assert "H" in proc.stdout and "512" in proc.stdout
