# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_cli_geometry.py                                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP — amostragem de geometria p/ JAX (Sprint v2.54)    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-02                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida `sample_geometry` + os geradores `_build_models`/               ║
# ║    `_build_random_models` com `--geometry`: per-model → n grupos          ║
# ║    (degenerado p/ JAX), templates → K grupos batcháveis, quantized →      ║
# ║    parcial. Garante que `per-model` (default) preserva o stream rng       ║
# ║    legado (reprodutibilidade) e que templates torna a geometria           ║
# ║    AGRUPÁVEL (n_grupos ≤ 0.5·n_models → JAX não cai p/ Numba).            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da amostragem de geometria da CLI (--geometry) — Sprint v2.54."""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.cli._exec import models_to_batch, sample_geometry
from geosteering_ai.cli.benchmark import _build_models
from geosteering_ai.cli.simulate import _build_random_models
from geosteering_ai.simulation._jax.multi_forward import group_by_geometry

# ════════════════════════════════════════════════════════════════════════
# sample_geometry — modos
# ════════════════════════════════════════════════════════════════════════


def test_per_model_is_unique_per_row() -> None:
    """per-model → cada modelo tem esp distinto (n grupos = n_models)."""
    rng = np.random.default_rng(42)
    esp = sample_geometry(rng, 100, 3, mode="per-model")
    assert esp.shape == (100, 3)
    assert len(group_by_geometry(esp)) == 100


def test_templates_makes_few_groups_by_default() -> None:
    """templates → POUCOS grupos por default (v2.56: max(1, min(n//256, 4)))."""
    rng = np.random.default_rng(42)
    # n=64 → 64//256=0 → max(1,0)=1 grupo; n=1000 → 1000//256=3 grupos.
    esp64 = sample_geometry(rng, 64, 3, mode="templates")
    assert esp64.shape == (64, 3)
    assert len(group_by_geometry(esp64)) == 1
    esp1000 = sample_geometry(np.random.default_rng(1), 1000, 3, mode="templates")
    assert len(group_by_geometry(esp1000)) == 3
    # cap em 4 mesmo p/ n enorme.
    esp_big = sample_geometry(np.random.default_rng(2), 100000, 3, mode="templates")
    assert len(group_by_geometry(esp_big)) == 4


def test_templates_explicit_n_geometries() -> None:
    """templates com n_geometries=4 → exatamente 4 grupos."""
    rng = np.random.default_rng(7)
    esp = sample_geometry(rng, 100, 3, mode="templates", n_geometries=4)
    assert len(group_by_geometry(esp)) == 4


def test_templates_is_groupable_for_jax() -> None:
    """templates torna a geometria AGRUPÁVEL (n_grupos ≤ 0.5·n_models)."""
    rng = np.random.default_rng(1)
    n = 128
    esp = sample_geometry(rng, n, 3, mode="templates")
    n_groups = len(group_by_geometry(esp))
    assert n_groups <= 0.5 * n  # passa o gate _GROUPABLE_RATIO_MAX do dispatcher


def test_quantized_reduces_groups_with_coarse_step() -> None:
    """quantized com passo grosso → bem menos grupos que per-model."""
    rng = np.random.default_rng(3)
    esp = sample_geometry(rng, 200, 3, mode="quantized", quantize_step=4.0)
    # esp ∈ [2,10] arredondado a 4 → {4, 8} (+ clamp) → poucas combinações.
    assert len(group_by_geometry(esp)) < 50


def test_quantized_step_must_be_positive() -> None:
    """quantize_step <= 0 → ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="quantize_step"):
        sample_geometry(rng, 10, 3, mode="quantized", quantize_step=0.0)


def test_invalid_mode_raises() -> None:
    """mode inválido → ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="geometry inválido"):
        sample_geometry(rng, 10, 3, mode="shared")


def test_n_esp_zero_returns_empty() -> None:
    """n_esp=0 (≤2 camadas) → array (n_models, 0) trivial."""
    rng = np.random.default_rng(0)
    esp = sample_geometry(rng, 5, 0, mode="templates")
    assert esp.shape == (5, 0)


# ════════════════════════════════════════════════════════════════════════
# Geradores da CLI (_build_models / _build_random_models)
# ════════════════════════════════════════════════════════════════════════


def test_build_models_per_model_preserves_legacy_stream() -> None:
    """per-model (default) preserva o stream rng legado (reprodutibilidade).

    O esp do default deve bater BIT-A-BIT com o gerador legado (esp sorteado
    no loop após rho_h/rho_v) — garante zero surpresa p/ seeds existentes.
    """
    # Reconstrói o stream legado manualmente.
    rng_ref = np.random.default_rng(42)
    legacy_esp = []
    for _ in range(4):
        _rho_h = rng_ref.uniform(1.0, 100.0, size=5)
        _rho_v = _rho_h * rng_ref.uniform(1.0, 3.0, size=5)
        legacy_esp.append(rng_ref.uniform(2.0, 10.0, size=3))

    models = _build_models(4, seed=42)  # default geometry="per-model"
    for m, exp in zip(models, legacy_esp):
        np.testing.assert_array_equal(m["esp"], exp)


def test_build_models_templates_groupable() -> None:
    """_build_models(geometry=templates) → geometria agrupável (poucos grupos)."""
    models = _build_models(1000, geometry="templates")
    _, _, esp = models_to_batch(models)
    n_groups = len(group_by_geometry(esp))
    assert n_groups == 3  # v2.56: 1000//256=3 (era 31)
    assert n_groups <= 0.5 * 1000  # agrupável (gate do dispatcher)


def test_build_random_models_templates_groupable() -> None:
    """_build_random_models(geometry=templates) → geometria agrupável."""
    models = _build_random_models(96, 42, geometry="templates", n_geometries=3)
    _, _, esp = models_to_batch(models)
    assert len(group_by_geometry(esp)) == 3


def test_build_models_per_model_is_degenerate() -> None:
    """_build_models default (per-model) → n grupos (degenerado p/ JAX)."""
    models = _build_models(50)
    _, _, esp = models_to_batch(models)
    assert len(group_by_geometry(esp)) == 50


def test_templates_shares_geometry_but_varies_rho() -> None:
    """templates: modelos i e i+K compartilham esp mas têm rho DIFERENTE.

    Confirma o cenário fisicamente realista (formação fixa, resistividades
    variadas) que satura o vmap do JAX.
    """
    models = _build_models(8, geometry="templates", n_geometries=2)
    # Modelos 0 e 2 compartilham o template 0 (round-robin % 2).
    np.testing.assert_array_equal(models[0]["esp"], models[2]["esp"])
    # mas o rho difere (sorteado por modelo no loop).
    assert not np.array_equal(models[0]["rho_h"], models[2]["rho_h"])
