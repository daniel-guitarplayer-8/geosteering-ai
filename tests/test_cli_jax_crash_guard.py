# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_cli_jax_crash_guard.py                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP — guarda anti-crash TLS do JAX (Sprint v2.55)      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-02                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida as correções do crash `_dl_allocate_tls_init` (libgomp após     ║
# ║    CUDA) + ruído de log:                                                  ║
# ║      • `_count_geometry_groups` ≡ `group_by_geometry` (jax-free);         ║
# ║      • `resolve_backend_preflight`: jax+não-agrupável → Numba SEM chamar   ║
# ║        `_jax_gpu_available` (sem init CUDA → sem crash TLS);              ║
# ║      • `cli/_main` seta NUMBA_NUM_THREADS/OMP/OPENBLAS (mitigação TLS);    ║
# ║      • o auto-detect de paralelismo loga em DEBUG (não INFO).            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da guarda anti-crash TLS do backend JAX (Sprint v2.55)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from geosteering_ai.cli._exec import (
    _count_geometry_groups,
    resolve_backend_preflight,
)
from geosteering_ai.cli.simulate import _build_random_models

PROJECT_ROOT = Path(__file__).parent.parent

# ════════════════════════════════════════════════════════════════════════
# _count_geometry_groups — equivalência jax-free com group_by_geometry
# ════════════════════════════════════════════════════════════════════════


def test_count_groups_matches_group_by_geometry() -> None:
    """`_count_geometry_groups` (NumPy puro) == len(group_by_geometry) (jax)."""
    from geosteering_ai.simulation._jax.multi_forward import group_by_geometry

    rng = np.random.default_rng(0)
    cases = [
        rng.uniform(2, 10, (64, 3)),  # per-model → 64 grupos
        rng.uniform(2, 10, (500, 3)),  # per-model → 500 grupos
        np.tile([3.0, 4.0, 5.0], (200, 1)),  # compartilhado → 1 grupo
        np.zeros((10, 0)),  # n_esp=0 → 1 grupo trivial
    ]
    for esp in cases:
        assert _count_geometry_groups(esp) == len(group_by_geometry(esp))


def test_count_groups_per_model_is_n() -> None:
    """per-model real (gerador da CLI) → n grupos distintos."""
    _, _, esp = _batch(_build_random_models(48, 42))
    assert _count_geometry_groups(esp) == 48


def test_count_groups_templates_is_k() -> None:
    """templates K=3 → 3 grupos."""
    _, _, esp = _batch(
        _build_random_models(96, 42, geometry="templates", n_geometries=3)
    )
    assert _count_geometry_groups(esp) == 3


def _batch(models):
    from geosteering_ai.cli._exec import models_to_batch

    return models_to_batch(models)


# ════════════════════════════════════════════════════════════════════════
# resolve_backend_preflight — NÃO inicializa CUDA quando vai rodar Numba
# ════════════════════════════════════════════════════════════════════════


def test_preflight_jax_nongroupable_avoids_cuda_init(monkeypatch) -> None:
    """jax + geometria não-agrupável → ('numba','cpu') SEM chamar
    `_jax_gpu_available` (= sem `jax.devices()` = sem init CUDA = sem crash TLS).

    Esta é a GUARDA central do crash `_dl_allocate_tls_init`.
    """
    import geosteering_ai.simulation.dispatch as disp

    called = {"v": False}

    def _spy():
        called["v"] = True
        return True

    monkeypatch.setattr(disp, "_jax_gpu_available", _spy)

    models = _build_random_models(64, 42)  # per-model (default) → não-agrupável
    backend, device, reason = resolve_backend_preflight("jax", models, quiet=True)

    assert backend == "numba"
    assert device == "cpu"
    assert reason is not None and "não-agrupável" in reason
    assert called["v"] is False, "init CUDA NÃO deve ser tocado p/ o caminho Numba"


def test_preflight_jax_groupable_probes_gpu(monkeypatch) -> None:
    """jax + geometria agrupável → sonda a GPU (`_jax_gpu_available`) → jax/gpu."""
    import geosteering_ai.simulation.dispatch as disp

    called = {"v": False}

    def _spy():
        called["v"] = True
        return True

    monkeypatch.setattr(disp, "_jax_gpu_available", _spy)

    models = _build_random_models(64, 42, geometry="templates")  # agrupável
    backend, device, reason = resolve_backend_preflight("jax", models, quiet=True)

    assert called["v"] is True  # sondou a GPU (groupable → segue p/ jax)
    assert backend == "jax"
    assert device == "gpu"
    assert reason is None


def test_preflight_numba_never_probes_gpu(monkeypatch) -> None:
    """backend numba → ('numba','cpu',None) sem sondar GPU."""
    import geosteering_ai.simulation.dispatch as disp

    called = {"v": False}
    monkeypatch.setattr(
        disp, "_jax_gpu_available", lambda: called.__setitem__("v", True) or True
    )
    backend, device, reason = resolve_backend_preflight(
        "numba", _build_random_models(8, 42), quiet=True
    )
    assert (backend, device, reason) == ("numba", "cpu", None)
    assert called["v"] is False


# ════════════════════════════════════════════════════════════════════════
# Mitigação de threads (env) + log em DEBUG
# ════════════════════════════════════════════════════════════════════════


def _run_main_env(extra_env: dict) -> dict:
    """Importa cli/_main num subprocesso e devolve as 3 env vars de thread."""
    env = dict(os.environ)
    for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.pop(k, None)
    env.update(extra_env)
    code = (
        "import os; import geosteering_ai.cli._main as m; "
        "print(os.environ.get('NUMBA_NUM_THREADS'), "
        "os.environ.get('OMP_NUM_THREADS'), os.environ.get('OPENBLAS_NUM_THREADS'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    vals = out.stdout.strip().split()
    return {"numba": vals[0], "omp": vals[1], "openblas": vals[2]}


def test_main_sets_thread_env_defaults() -> None:
    """Importar `cli/_main` seta NUMBA_NUM_THREADS (≥1) + OMP/OPENBLAS=1."""
    v = _run_main_env({})
    assert v["numba"].isdigit() and int(v["numba"]) >= 1
    assert v["omp"] == "1"
    assert v["openblas"] == "1"


def test_main_preserves_thread_env_override() -> None:
    """`setdefault` preserva override do usuário (NUMBA_NUM_THREADS=7)."""
    v = _run_main_env({"NUMBA_NUM_THREADS": "7", "OMP_NUM_THREADS": "9"})
    assert v["numba"] == "7"
    assert v["omp"] == "9"


def test_autodetect_log_is_debug_not_info(caplog) -> None:
    """O auto-detect de paralelismo loga em DEBUG, não INFO (silêncio no JAX)."""
    from geosteering_ai.simulation.config import SimulationConfig

    with caplog.at_level(logging.INFO, logger="geosteering_ai.simulation.config"):
        SimulationConfig()  # sem workers → dispara o auto-detect

    info_autodetect = [
        r
        for r in caplog.records
        if r.levelno >= logging.INFO and "auto-detect" in r.getMessage()
    ]
    assert not info_autodetect, "auto-detect NÃO deve logar em INFO (deve ser DEBUG)"
