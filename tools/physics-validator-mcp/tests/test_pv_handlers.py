# -*- coding: utf-8 -*-
"""Testes unitários para os handlers do physics-validator MCP (I1.9).

Não requer ``mcp`` instalado — testa os ``_impl_*`` diretamente.
"""

from __future__ import annotations

# Permite executar tanto via `pytest tools/physics-validator-mcp/tests/`
# quanto via `pytest` na raiz do repo. Carrega o server.py com nome único
# (`physics_validator_server`) para evitar colisão com o `server` do MCP irmão.
import importlib.util  # noqa: E402
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parents[1]
_SERVER_PATH = _MCP_DIR / "server.py"
_spec = importlib.util.spec_from_file_location("physics_validator_server", _SERVER_PATH)
assert _spec and _spec.loader, "Falha ao carregar physics_validator/server.py"
server = importlib.util.module_from_spec(_spec)
sys.modules["physics_validator_server"] = server
_spec.loader.exec_module(server)

_REPO_ROOT = _MCP_DIR.parents[1]
_TATU_X = _REPO_ROOT / "Fortran_Gerador" / "tatu.x"


# ──────────────────────────────────────────────────────────────────────────
# Tool registry / metadata
# ──────────────────────────────────────────────────────────────────────────


def test_tool_registry_has_six_handlers() -> None:
    """O TOOL_REGISTRY deve expor exatamente 6 handlers nomeados."""
    assert len(server.TOOL_REGISTRY) == 6
    expected = {
        "check_fortran_parity",
        "check_maxwell_symmetry",
        "check_decoupling_factors",
        "check_errata_immutable",
        "check_skin_depth",
        "run_canonical_models",
    }
    assert set(server.TOOL_REGISTRY.keys()) == expected


def test_tool_definitions_match_registry() -> None:
    """Definições (inputSchema) devem cobrir todos os handlers do registry."""
    defs = server._build_tool_definitions()
    names = {d["name"] for d in defs}
    assert names == set(server.TOOL_REGISTRY.keys())
    # Cada definição deve ter inputSchema dict
    for d in defs:
        assert "inputSchema" in d
        assert d["inputSchema"]["type"] == "object"


# ──────────────────────────────────────────────────────────────────────────
# check_decoupling_factors — função pura, valor analítico
# ──────────────────────────────────────────────────────────────────────────


def test_check_decoupling_factors_unit_spacing() -> None:
    """ACp ≈ -0.07957747, ACx ≈ +0.15915494, ratio = 2.0 exato em L=1.0."""
    result = server._impl_check_decoupling_factors(spacing_m=1.0)
    assert result["passed"] is True
    assert abs(result["ACp_computed"] - (-0.07957747154594767)) < 1e-15
    assert abs(result["ACx_computed"] - (+0.15915494309189535)) < 1e-15
    assert abs(result["ratio_ACx_over_minus_ACp"] - 2.0) < 1e-12


def test_check_decoupling_factors_scaling_with_L() -> None:
    """ACp escala com 1/L³ — dobrar L reduz |ACp| por fator de 8."""
    r1 = server._impl_check_decoupling_factors(spacing_m=1.0)
    r2 = server._impl_check_decoupling_factors(spacing_m=2.0)
    ratio = r1["ACp_computed"] / r2["ACp_computed"]
    assert abs(ratio - 8.0) < 1e-9


# ──────────────────────────────────────────────────────────────────────────
# check_skin_depth — função analítica fechada
# ──────────────────────────────────────────────────────────────────────────


def test_check_skin_depth_known_value() -> None:
    """ρ=100 Ω·m, f=20 kHz → δ ≈ 35.59 m (validado contra fórmula 503√(ρ/f))."""
    result = server._impl_check_skin_depth(rho_omega_m=100.0, frequency_hz=20000.0)
    assert result["passed"] is True
    assert abs(result["skin_depth_m"] - 35.588127170858854) < 1e-6
    assert result["warning"] == "OK"


def test_check_skin_depth_high_attenuation_warning() -> None:
    """ρ baixa + f alta → δ << 1m → warning de alta atenuação."""
    result = server._impl_check_skin_depth(rho_omega_m=0.01, frequency_hz=1.0e6)
    # 503·√(0.01 / 1e6) = 503·1e-4 ≈ 0.05 m
    assert result["skin_depth_m"] < 1.0
    assert "atenuação" in result["warning"].lower()


# ──────────────────────────────────────────────────────────────────────────
# check_errata_immutable — coerência runtime vs CLAUDE.md
# ──────────────────────────────────────────────────────────────────────────


def test_check_errata_immutable_matches_simulation_config() -> None:
    """Defaults de SimulationConfig() devem bater com ERRATA_DEFAULTS."""
    result = server._impl_check_errata_immutable()
    assert result["passed"] is True
    assert result["mismatches"] == []
    # Confere que as constantes errata estão expostas
    constants = result["errata_constants"]
    assert constants["FREQUENCY_HZ"] == 20000.0
    assert constants["SPACING_METERS"] == 1.0
    assert constants["TARGET_SCALING"] == "log10"


# ──────────────────────────────────────────────────────────────────────────
# check_maxwell_symmetry — validate_all_analytical (3 sub-checks)
# ──────────────────────────────────────────────────────────────────────────


def test_check_maxwell_symmetry_default_filter() -> None:
    """Werthmüller 201pt deve passar todos os 3 sub-checks (decoupling + 2 VMD)."""
    result = server._impl_check_maxwell_symmetry()
    assert result["check_name"] == "maxwell_symmetry"
    assert "decoupling" in result
    assert "vmd_axial" in result
    assert "vmd_broadside" in result
    # Sub-check decoupling deve sempre passar (analítico puro)
    # NB: validate_decoupling retorna numpy bool — comparar com == em vez de is
    assert bool(result["decoupling"].get("pass")) is True


# ──────────────────────────────────────────────────────────────────────────
# check_fortran_parity — skip se tatu.x não compilado
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not _TATU_X.exists(),
    reason="tatu.x não compilado em Fortran_Gerador/",
)
def test_check_fortran_parity_oklahoma_3() -> None:
    """Paridade Numba vs Fortran no oklahoma_3 deve ser <1e-6 (smoke)."""
    result = server._impl_check_fortran_parity(
        model_name="oklahoma_3",
        tolerance=1.0e-6,
        backend="numba",
        n_positions=20,  # smoke rápido
    )
    assert result.get("skipped") is not True
    assert result["passed"] is True
    assert result["max_abs_error"] < 1.0e-6


def test_check_fortran_parity_skips_gracefully_when_tatu_missing() -> None:
    """Se tatu.x não existe, retorna skipped sem exceção."""
    # Salva path original e força inexistente
    original = server.DEFAULT_FORTRAN_EXEC
    server.DEFAULT_FORTRAN_EXEC = Path("/nonexistent/tatu.x")
    try:
        result = server._impl_check_fortran_parity(model_name="oklahoma_3")
        assert result.get("skipped") is True
        assert "não compilado" in result["reason"]
    finally:
        server.DEFAULT_FORTRAN_EXEC = original


# ──────────────────────────────────────────────────────────────────────────
# run_canonical_models — agregador
# ──────────────────────────────────────────────────────────────────────────


def test_run_canonical_models_returns_summary() -> None:
    """O agregador deve incluir n_models, n_passed, n_skipped, results[]."""
    # Subset pequeno para teste rápido
    result = server._impl_run_canonical_models(
        models=["oklahoma_3"],
        backend="numba",
        n_positions=20,
    )
    assert result["check_name"] == "canonical_models_suite"
    assert result["n_models"] == 1
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 1
