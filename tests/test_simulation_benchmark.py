# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_benchmark.py                                       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Testes Benchmark (Sprint 2.7)          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do benchmark forward (Sprint 2.7).

Baterias:
- TestBenchmarkSmoke (4): run_benchmark funciona, shapes corretas, no NaN.
- TestBenchmarkResult (3): to_markdown, pass_target, platform_info.
- TestModelFactories (3): 3 tamanhos geram modelos válidos.
"""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation.benchmarks.bench_forward import (
    BenchmarkResult,
    run_benchmark,
)


class TestBenchmarkSmoke:
    """Smoke tests: benchmark roda e retorna resultados válidos."""

    def test_small_runs(self):
        """Benchmark small executa sem erros."""
        result = run_benchmark(size="small", n_iter=2, n_warmup=1)
        assert isinstance(result, BenchmarkResult)
        assert result.n_iter == 2
        assert result.n_warmup == 1

    def test_throughput_positive(self):
        """Throughput é positivo."""
        result = run_benchmark(size="small", n_iter=2, n_warmup=1)
        assert result.throughput_models_per_hour > 0

    def test_times_list_correct_length(self):
        """Lista de tempos tem n_iter elementos."""
        result = run_benchmark(size="small", n_iter=3, n_warmup=1)
        assert len(result.times_seconds) == 3
        assert all(t > 0 for t in result.times_seconds)

    def test_medium_runs(self):
        """Benchmark medium executa sem erros."""
        result = run_benchmark(size="medium", n_iter=2, n_warmup=1)
        assert result.throughput_models_per_hour > 0
        assert result.n_layers == 7
        assert result.n_positions == 300


class TestBenchmarkResult:
    """Testes do container BenchmarkResult."""

    def test_to_markdown_contains_throughput(self):
        """Relatório markdown contém throughput."""
        result = run_benchmark(size="small", n_iter=2, n_warmup=1)
        md = result.to_markdown()
        assert "mod/h" in md
        assert "Benchmark Forward" in md

    def test_pass_target_flag(self):
        """Flag pass_target reflete a meta de 40k mod/h."""
        result = run_benchmark(size="small", n_iter=2, n_warmup=1)
        if result.throughput_models_per_hour >= 40_000:
            assert result.pass_target is True
        else:
            assert result.pass_target is False

    def test_platform_info_non_empty(self):
        """Informações de plataforma estão preenchidas."""
        result = run_benchmark(size="small", n_iter=2, n_warmup=1)
        assert len(result.platform_info) > 10
        assert "Python" in result.platform_info


class TestModelFactories:
    """Testes das fábricas de modelos."""

    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_model_factory_produces_valid(self, size):
        """Cada fábrica produz modelo com shapes consistentes."""
        from geosteering_ai.simulation.benchmarks.bench_forward import _MODEL_FACTORIES

        model = _MODEL_FACTORIES[size]()
        n = model["n_layers"]
        assert model["rho_h"].shape == (n,)
        assert model["rho_v"].shape == (n,)
        if n > 2:
            assert model["esp"].shape == (n - 2,)
        assert model["positions_z"].shape == (model["n_positions"],)

    def test_invalid_size_raises(self):
        """Tamanho inválido levanta ValueError."""
        with pytest.raises(ValueError, match="inválido"):
            run_benchmark(size="huge", n_iter=1)
