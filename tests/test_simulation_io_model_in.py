# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_io_model_in.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — Leitor model.in + simulate_from_model_in         ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest + numpy                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de ``read_model_in`` e ``simulate_from_model_in`` (Sprint 3.3.4+)."""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation import SimulationConfig
from geosteering_ai.simulation.io import (
    export_model_in,
    read_model_in,
    simulate_from_model_in,
)


class TestReadModelInRoundtrip:
    """Roundtrip: export_model_in → read_model_in → campos iguais."""

    def test_basic_roundtrip(self, tmp_path) -> None:
        cfg = SimulationConfig(export_model_in=True, output_dir=str(tmp_path))
        rho_h = np.array([1.0, 100.0, 1.0])
        rho_v = np.array([1.0, 200.0, 1.0])
        esp = np.array([5.0])
        path = export_model_in(cfg, rho_h, rho_v, esp)
        params = read_model_in(path)
        assert params["n_layers"] == 3
        np.testing.assert_array_almost_equal(params["rho_h"], rho_h)
        np.testing.assert_array_almost_equal(params["rho_v"], rho_v)
        np.testing.assert_array_almost_equal(params["esp"], esp)

    def test_frequencies_preserved(self, tmp_path) -> None:
        cfg = SimulationConfig(
            export_model_in=True,
            output_dir=str(tmp_path),
            frequency_hz=20000.0,
        )
        rho_h = np.array([10.0, 100.0, 10.0])
        rho_v = np.array([10.0, 100.0, 10.0])
        esp = np.array([5.0])
        path = export_model_in(cfg, rho_h, rho_v, esp)
        params = read_model_in(path)
        assert len(params["frequencies_hz"]) >= 1
        assert abs(params["frequencies_hz"][0] - 20000.0) < 1.0


class TestReadModelInErrors:
    """Erros de leitura devem levantar exceções claras."""

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="não encontrado"):
            read_model_in("/tmp/inexistente_model_xyz.in")

    def test_malformed_file(self, tmp_path) -> None:
        bad = tmp_path / "bad_model.in"
        bad.write_text("isto não é um model.in válido\n")
        with pytest.raises(ValueError, match="inválido"):
            read_model_in(bad)


class TestSimulateFromModelIn:
    """simulate_from_model_in executa e exporta .dat/.out."""

    def test_produces_result(self, tmp_path) -> None:
        # Primeiro, gera um model.in válido
        cfg = SimulationConfig(export_model_in=True, output_dir=str(tmp_path))
        rho_h = np.array([1.0, 100.0, 1.0])
        rho_v = np.array([1.0, 100.0, 1.0])
        esp = np.array([5.0])
        model_path = export_model_in(cfg, rho_h, rho_v, esp)

        # Executa simulação a partir do model.in
        result = simulate_from_model_in(
            model_path,
            output_dir=str(tmp_path / "output"),
            export_dat=True,
            export_out=True,
            positions_z=np.linspace(0.0, 5.0, 20),
        )
        assert result.H_tensor.shape[0] == 20
        assert result.H_tensor.shape[2] == 9

    def test_produces_dat_file(self, tmp_path) -> None:
        cfg = SimulationConfig(export_model_in=True, output_dir=str(tmp_path))
        rho_h = np.array([1.0, 100.0, 1.0])
        esp = np.array([5.0])
        model_path = export_model_in(cfg, rho_h, rho_h, esp)
        out_dir = tmp_path / "out_dat"
        simulate_from_model_in(
            model_path,
            output_dir=str(out_dir),
            export_dat=True,
            export_out=False,
            positions_z=np.array([2.5]),
        )
        dat_files = list(out_dir.glob("*.dat"))
        assert len(dat_files) >= 1, "Nenhum .dat gerado"
