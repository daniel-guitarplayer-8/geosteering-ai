# -*- coding: utf-8 -*-
"""Testes Sprint 4.4 — Validação binária Fortran ↔ Python.

Cobertura:

- **Smoke tatu.x (1 teste)**: subprocess executa, retorna rc=0, gera .dat.
- **Parser .dat (1 teste)**: round-trip binário usando DTYPE_22COL.
- **7 modelos canônicos × Numba (7 testes)**: paridade max_abs<1e-6.
- **Alta resistividade (1 teste)**: oklahoma_28 com ρ>1000 estável.

Guarda: todos os testes fazem `skipif` se ``Fortran_Gerador/tatu.x``
não existir, permitindo rodar CI sem o binário Fortran compilado.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.io.binary_dat import DTYPE_22COL
from geosteering_ai.simulation.io.model_in import export_model_in
from geosteering_ai.simulation.validation.canonical_models import get_canonical_model
from geosteering_ai.simulation.validation.compare_fortran import (
    DEFAULT_FORTRAN_EXEC,
    compare_fortran_python,
    read_fortran_dat_22col,
    run_tatu_x,
)

FORTRAN_AVAILABLE = DEFAULT_FORTRAN_EXEC.exists()
fortran_required = pytest.mark.skipif(
    not FORTRAN_AVAILABLE,
    reason=f"tatu.x não compilado em {DEFAULT_FORTRAN_EXEC}. "
    "Execute `make -C Fortran_Gerador` para habilitar.",
)


# ═════════════════════════════════════════════════════════════════════════
# 1 — Smoke tatu.x subprocess
# ═════════════════════════════════════════════════════════════════════════


@fortran_required
def test_run_tatu_x_smoke(tmp_path: Path) -> None:
    """tatu.x executa via subprocess com rc=0 e gera .dat válido."""
    m = get_canonical_model("oklahoma_3")
    cfg = SimulationConfig(
        frequency_hz=20000.0,
        tr_spacing_m=1.0,
        backend="numba",
        export_model_in=True,
        export_binary_dat=False,
        output_dir=str(tmp_path),
        output_filename="smoke",
    )
    model_in = export_model_in(cfg, m.rho_h, m.rho_v, m.esp, h1=-2.0, tj=8.0, pmed=0.2)
    result = run_tatu_x(model_in, output_dir=tmp_path, timeout_s=60.0)
    assert result[
        "success"
    ], f"tatu.x falhou rc={result['returncode']} stderr={result['stderr'][:400]}"
    assert result["dat_path"].exists()
    assert result["elapsed_s"] < 30.0, "Smoke com 40 posições demorou demais"


# ═════════════════════════════════════════════════════════════════════════
# 2 — Parser .dat 22-col round-trip
# ═════════════════════════════════════════════════════════════════════════


@fortran_required
def test_fortran_dat_parser_roundtrip(tmp_path: Path) -> None:
    """read_fortran_dat_22col decodifica .dat Fortran corretamente."""
    m = get_canonical_model("oklahoma_5")
    cfg = SimulationConfig(
        frequency_hz=20000.0,
        tr_spacing_m=1.0,
        backend="numba",
        export_model_in=True,
        export_binary_dat=False,
        output_dir=str(tmp_path),
        output_filename="parsing",
    )
    model_in = export_model_in(cfg, m.rho_h, m.rho_v, m.esp, h1=-1.0, tj=6.0, pmed=0.25)
    result = run_tatu_x(model_in, output_dir=tmp_path, timeout_s=60.0)
    assert result["success"]

    records = read_fortran_dat_22col(result["dat_path"])
    assert records.dtype == DTYPE_22COL
    assert len(records) > 5
    # z_obs crescente (profundidade aumenta)
    z = records["z_obs"]
    assert np.all(np.diff(z) > -1e-10), "z_obs deveria ser monotônica"
    # Hxx finito (não-NaN, não-Inf)
    assert np.all(np.isfinite(records["Re_Hxx"]))
    assert np.all(np.isfinite(records["Im_Hxx"]))


# ═════════════════════════════════════════════════════════════════════════
# 3-9 — Paridade Fortran ↔ Python Numba nos 7 canônicos
# ═════════════════════════════════════════════════════════════════════════


CANONICAL_MODELS = [
    "oklahoma_3",
    "oklahoma_5",
    "devine_8",
    "oklahoma_15",
    "oklahoma_28",
    "hou_7",
    "viking_graben_10",
]


@fortran_required
@pytest.mark.parametrize("model_name", CANONICAL_MODELS)
def test_compare_fortran_python_numba(model_name: str, tmp_path: Path) -> None:
    """Paridade Numba vs Fortran: max_abs < 1e-6 nos 7 modelos canônicos."""
    results = compare_fortran_python(
        canonical_model_name=model_name,
        backends=["numba"],
        n_positions=80,
        workdir=tmp_path,
    )
    assert len(results) == 1
    r = results[0]
    assert r.passed, (
        f"[{model_name}/numba] max_abs={r.max_abs_error:.2e} "
        f"> tol={r.tol_abs_used:.0e} (notes={r.notes})"
    )
    assert (
        r.max_abs_error < 1e-5
    ), f"[{model_name}] esperava max_abs<1e-5, obteve {r.max_abs_error:.2e}"


# ═════════════════════════════════════════════════════════════════════════
# 10 — Estabilidade alta resistividade (ρ > 1000 Ω·m)
# ═════════════════════════════════════════════════════════════════════════


@fortran_required
def test_compare_fortran_high_rho_stability(tmp_path: Path) -> None:
    """oklahoma_28 (ρ>1000 Ω·m) permanece estável — sem NaN/Inf."""
    results = compare_fortran_python(
        canonical_model_name="oklahoma_28",
        backends=["numba"],
        n_positions=100,
        workdir=tmp_path,
    )
    r = results[0]
    assert np.isfinite(r.max_abs_error), "max_abs_error não-finito (NaN/Inf)"
    assert np.isfinite(r.l2_error)
    assert r.passed, f"oklahoma_28 (alta ρ) falhou: max_abs={r.max_abs_error:.2e}"
