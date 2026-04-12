# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_io.py                                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Testes Exportadores (Sprint 2.2)      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest 7.x + numpy 2.x                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida os exportadores Fortran-compatíveis da Sprint 2.2:            ║
# ║                                                                           ║
# ║      1. export_model_in — layout, flags, round-trip texto               ║
# ║      2. export_binary_dat — layout byte, round-trip via np.fromfile    ║
# ║      3. export_out_metadata — conteúdo .out texto                       ║
# ║                                                                           ║
# ║    Além do gating opt-in (falha se flag=False).                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes dos exportadores Fortran-compatíveis (Sprint 2.2).

Testes implementados:

- **TestModelInBasic** (5 testes): opt-in gating, layout base, filter_type.
- **TestModelInFlags** (4 testes): F5/F6/F7 flags, multi-TR, multi-freq.
- **TestBinaryDatRoundTrip** (5 testes): layout byte, round-trip NumPy.
- **TestOutMetadata** (2 testes): conteúdo .out mínimo e F7.

Total: 16 testes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geosteering_ai.simulation import SimulationConfig
from geosteering_ai.simulation.io import (
    DTYPE_22COL,
    export_binary_dat,
    export_model_in,
    export_out_metadata,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures compartilhadas
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Diretório temporário único por teste."""
    out = tmp_path / "sim_out"
    out.mkdir(exist_ok=True)
    return out


@pytest.fixture
def cfg_model_in(tmp_output_dir: Path) -> SimulationConfig:
    """SimulationConfig com export_model_in ativo."""
    return SimulationConfig(
        frequency_hz=20000.0,
        tr_spacing_m=1.0,
        export_model_in=True,
        output_dir=str(tmp_output_dir),
        output_filename="validacao",
    )


@pytest.fixture
def cfg_binary(tmp_output_dir: Path) -> SimulationConfig:
    """SimulationConfig com export_binary_dat ativo."""
    return SimulationConfig(
        frequency_hz=20000.0,
        tr_spacing_m=1.0,
        export_binary_dat=True,
        output_dir=str(tmp_output_dir),
        output_filename="validacao",
    )


@pytest.fixture
def rho_profile_7layers():
    """Perfil 7 camadas paridade com exemplo model.in."""
    rho_h = np.array([1.0, 80.0, 1.0, 10.0, 1.0, 0.3, 1.0], dtype=np.float64)
    rho_v = np.array([10.0, 80.0, 10.0, 10.0, 10.0, 0.3, 10.0], dtype=np.float64)
    thicknesses = np.array([1.52, 2.35, 2.10, 1.88, 0.92], dtype=np.float64)
    return rho_h, rho_v, thicknesses


# ──────────────────────────────────────────────────────────────────────────────
# TestModelInBasic — 5 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestModelInBasic:
    """Testes básicos do exportador model.in."""

    def test_opt_in_gating(self, tmp_output_dir: Path):
        """Chamar com flag desativada levanta RuntimeError."""
        cfg = SimulationConfig(output_dir=str(tmp_output_dir))
        assert cfg.export_model_in is False
        with pytest.raises(RuntimeError, match="cfg.export_model_in=True"):
            export_model_in(
                cfg,
                rho_h=np.ones(3),
                rho_v=np.ones(3),
                thicknesses=np.ones(1),
            )

    def test_basic_layout_produces_file(
        self, cfg_model_in: SimulationConfig, rho_profile_7layers
    ):
        """Layout base gera arquivo .model.in no diretório correto."""
        rho_h, rho_v, thicknesses = rho_profile_7layers
        path = export_model_in(cfg_model_in, rho_h, rho_v, thicknesses)
        assert path.exists()
        assert path.name == "validacao.model.in"
        content = path.read_text(encoding="utf-8")
        # Deve conter pelo menos 7 camadas de resistividades
        assert content.count("!resistividades hor/vert") == 7

    def test_layout_includes_frequency_and_tr(
        self, cfg_model_in: SimulationConfig, rho_profile_7layers
    ):
        """Layout inclui frequência e espaçamento TR."""
        rho_h, rho_v, thicknesses = rho_profile_7layers
        path = export_model_in(cfg_model_in, rho_h, rho_v, thicknesses)
        content = path.read_text(encoding="utf-8")
        assert "20000.000000" in content
        assert "distância T-R" in content
        assert "1.000000" in content  # 1.0 m default

    def test_filter_type_werthmuller_default(
        self, cfg_model_in: SimulationConfig, rho_profile_7layers
    ):
        """hankel_filter=werthmuller_201pt → filter_type=0."""
        rho_h, rho_v, thicknesses = rho_profile_7layers
        path = export_model_in(cfg_model_in, rho_h, rho_v, thicknesses)
        content = path.read_text(encoding="utf-8")
        assert "0                 !Filtro: 0=Werthmuller" in content

    def test_filter_type_kong(self, tmp_output_dir: Path, rho_profile_7layers):
        """hankel_filter=kong_61pt → filter_type=1."""
        cfg = SimulationConfig(
            hankel_filter="kong_61pt",
            export_model_in=True,
            output_dir=str(tmp_output_dir),
        )
        rho_h, rho_v, thicknesses = rho_profile_7layers
        path = export_model_in(cfg, rho_h, rho_v, thicknesses)
        content = path.read_text(encoding="utf-8")
        assert "1                 !Filtro: 1=Kong" in content


# ──────────────────────────────────────────────────────────────────────────────
# TestModelInFlags — 4 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestModelInFlags:
    """Testes das flags opcionais (F5, F6, F7) no model.in."""

    def test_multi_freq_enables_f5(self, tmp_output_dir: Path, rho_profile_7layers):
        """frequencies_hz com 2+ valores ativa F5=1."""
        cfg = SimulationConfig(
            frequencies_hz=[20000.0, 100000.0, 400000.0],
            export_model_in=True,
            output_dir=str(tmp_output_dir),
        )
        rho_h, rho_v, thicknesses = rho_profile_7layers
        path = export_model_in(cfg, rho_h, rho_v, thicknesses)
        content = path.read_text(encoding="utf-8")
        assert "3                 !número de frequências" in content
        assert "1                 !F5: use_arbitrary_freq" in content

    def test_multi_tr(self, tmp_output_dir: Path, rho_profile_7layers):
        """tr_spacings_m com 3 valores → nTR=3 + 3 distâncias."""
        cfg = SimulationConfig(
            tr_spacings_m=[0.5, 1.0, 2.0],
            export_model_in=True,
            output_dir=str(tmp_output_dir),
        )
        rho_h, rho_v, thicknesses = rho_profile_7layers
        path = export_model_in(cfg, rho_h, rho_v, thicknesses)
        content = path.read_text(encoding="utf-8")
        assert "3                 !número de pares T-R" in content
        # 3 linhas de distância
        assert content.count("!distância T-R") == 3

    def test_f6_compensation_enabled(self, tmp_output_dir: Path, rho_profile_7layers):
        """use_compensation=True escreve flag F6=1 + pares."""
        cfg = SimulationConfig(
            tr_spacings_m=[0.5, 1.0, 2.0],
            use_compensation=True,
            comp_pairs=((0, 2),),
            export_model_in=True,
            output_dir=str(tmp_output_dir),
        )
        rho_h, rho_v, thicknesses = rho_profile_7layers
        path = export_model_in(cfg, rho_h, rho_v, thicknesses)
        content = path.read_text(encoding="utf-8")
        assert "1                 !F6: use_compensation" in content
        # Fortran usa 1-based: (0,2) → (1,3)
        assert "1 3" in content

    def test_f7_tilted_antennas_enabled(self, tmp_output_dir: Path, rho_profile_7layers):
        """use_tilted_antennas=True escreve F7=1 + betas/phis."""
        cfg = SimulationConfig(
            use_tilted_antennas=True,
            tilted_configs=((45.0, 0.0), (45.0, 90.0)),
            export_model_in=True,
            output_dir=str(tmp_output_dir),
        )
        rho_h, rho_v, thicknesses = rho_profile_7layers
        path = export_model_in(cfg, rho_h, rho_v, thicknesses)
        content = path.read_text(encoding="utf-8")
        assert "1                 !F7: use_tilted_antennas" in content
        assert "2                 !n_tilted" in content
        assert "45.0000 45.0000" in content  # betas
        assert "0.0000 90.0000" in content  # phis


# ──────────────────────────────────────────────────────────────────────────────
# TestBinaryDatRoundTrip — 5 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestBinaryDatRoundTrip:
    """Testes de round-trip binário .dat."""

    def test_opt_in_gating(self, tmp_output_dir: Path):
        """Chamar com export_binary_dat=False levanta RuntimeError."""
        cfg = SimulationConfig(output_dir=str(tmp_output_dir))
        with pytest.raises(RuntimeError, match="cfg.export_binary_dat=True"):
            export_binary_dat(
                cfg,
                H_tensor=np.zeros((1, 5, 1, 9), dtype=np.complex128),
                z_obs=np.zeros(5),
                rho_h_profile=np.zeros(5),
                rho_v_profile=np.zeros(5),
            )

    def test_dtype_is_172_bytes(self):
        """DTYPE_22COL tem exatamente 172 bytes (4 + 21*8)."""
        assert DTYPE_22COL.itemsize == 172

    def test_roundtrip_byte_exact(self, cfg_binary: SimulationConfig):
        """Escrever + ler + comparar byte-exato."""
        n_meds = 20
        # Tensor com valores distintos por componente para verificar ordem
        H = np.arange(n_meds * 9, dtype=np.float64).reshape(1, n_meds, 1, 9) * (
            1.0 + 0.5j
        )
        H = H.astype(np.complex128)
        z_obs = np.linspace(0.0, 19.0, n_meds)
        rho_h = np.full(n_meds, 100.0)
        rho_v = np.full(n_meds, 120.0)

        path = export_binary_dat(cfg_binary, H, z_obs, rho_h, rho_v)
        assert path.exists()
        assert path.stat().st_size == n_meds * 172

        data = np.fromfile(path, dtype=DTYPE_22COL)
        assert data.shape == (n_meds,)
        # Verificar índice 1-based
        assert data["i"][0] == 1
        assert data["i"][-1] == n_meds
        # Verificar z_obs
        np.testing.assert_allclose(data["z_obs"], z_obs)
        # Verificar rho_h e rho_v
        np.testing.assert_allclose(data["rho_h"], rho_h)
        np.testing.assert_allclose(data["rho_v"], rho_v)
        # Verificar Hxx (primeira coluna do tensor)
        np.testing.assert_allclose(data["Re_Hxx"], H[0, :, 0, 0].real)
        np.testing.assert_allclose(data["Im_Hxx"], H[0, :, 0, 0].imag)
        # Verificar Hzz (última coluna do tensor)
        np.testing.assert_allclose(data["Re_Hzz"], H[0, :, 0, 8].real)
        np.testing.assert_allclose(data["Im_Hzz"], H[0, :, 0, 8].imag)

    def test_append_mode_duplicates_records(self, cfg_binary: SimulationConfig):
        """append=True adiciona registros ao arquivo existente."""
        n_meds = 5
        H = np.zeros((1, n_meds, 1, 9), dtype=np.complex128)
        z_obs = np.arange(n_meds, dtype=np.float64)
        rho = np.ones(n_meds, dtype=np.float64)

        path = export_binary_dat(cfg_binary, H, z_obs, rho, rho, append=False)
        assert np.fromfile(path, dtype=DTYPE_22COL).shape == (n_meds,)

        # Append — adiciona mais n_meds registros
        export_binary_dat(cfg_binary, H, z_obs, rho, rho, append=True)
        data = np.fromfile(path, dtype=DTYPE_22COL)
        assert data.shape == (2 * n_meds,)

    def test_shape_2d_auto_adapts(self, cfg_binary: SimulationConfig):
        """Shape (nmeds, 9) é adaptado para (1, nmeds, 1, 9)."""
        n_meds = 5
        H = np.ones((n_meds, 9), dtype=np.complex128)
        z_obs = np.arange(n_meds, dtype=np.float64)
        rho = np.ones(n_meds, dtype=np.float64)
        path = export_binary_dat(cfg_binary, H, z_obs, rho, rho)
        data = np.fromfile(path, dtype=DTYPE_22COL)
        assert data.shape == (n_meds,)


# ──────────────────────────────────────────────────────────────────────────────
# TestOutMetadata — 2 testes
# ──────────────────────────────────────────────────────────────────────────────
class TestOutMetadata:
    """Testes de metadata .out."""

    def test_basic_out_content(self, cfg_binary: SimulationConfig):
        """.out contém linhas mínimas de nt/nf/nmodels/angulos/freqs/nmeds."""
        path = export_out_metadata(
            cfg_binary,
            n_models=1,
            angulos_deg=np.array([0.0]),
            freqs_hz=np.array([20000.0]),
            nmeds_per_theta=np.array([600]),
        )
        assert path.exists()
        assert path.name == "infovalidacao.out"
        content = path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # Layout esperado (linha 0 = "1 1 1", linha 1 = ángulos, linha 2 = freqs, linha 3 = nmeds, linha 4 = "0 0" F7 off)
        assert lines[0].strip().split() == ["1", "1", "1"]
        assert "0.0" in lines[1]
        assert "20000.0" in lines[2]
        assert "600" in lines[3]
        assert lines[4].strip() == "0 0"  # F7 off

    def test_f7_adds_tilted_info(self, tmp_output_dir: Path):
        """Com F7 ativo, .out inclui 3 linhas extras (flag + betas + phis)."""
        cfg = SimulationConfig(
            use_tilted_antennas=True,
            tilted_configs=((45.0, 0.0), (45.0, 90.0)),
            export_binary_dat=True,
            output_dir=str(tmp_output_dir),
        )
        path = export_out_metadata(
            cfg,
            n_models=1,
            angulos_deg=np.array([0.0]),
            freqs_hz=np.array([20000.0]),
            nmeds_per_theta=np.array([600]),
        )
        content = path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # Linha 4 agora tem "1 2" (F7 on, 2 tilted)
        assert lines[4].strip() == "1 2"
        # Linha 5 tem betas, linha 6 tem phis
        assert "45.0000" in lines[5]
        assert "90.0000" in lines[6]
