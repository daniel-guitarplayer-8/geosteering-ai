"""Testes unitarios para o modulo data/ (Bloco 2).

Cobre: loading, splitting, feature_views, geosignals, scaling, pipeline.
Usa dados sinteticos (sem dependencia de .dat/.out reais).
"""

import math
import os
import struct
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.feature_views import EPS, VALID_VIEWS, apply_feature_view
from geosteering_ai.data.geosignals import (
    FAMILY_DEPS,
    compute_expanded_features,
    compute_geosignals,
)
from geosteering_ai.data.loading import (
    EM_COMPONENTS,
    AngleGroup,
    OutMetadata,
    apply_decoupling,
    load_binary_dat,
    parse_out_metadata,
    segregate_by_angle,
)
from geosteering_ai.data.scaling import (
    apply_target_scaling,
    create_scaler,
    fit_scaler,
    inverse_target_scaling,
    transform_features,
)
from geosteering_ai.data.splitting import DataSplits, apply_split, split_model_ids

# ════════════════════════════════════════════════════════════════════════
# FIXTURES — Dados sinteticos
# ════════════════════════════════════════════════════════════════════════


def _make_synthetic_out(tmp_path, n_models=10, n_angles=1, n_freqs=1, nmeds=600):
    """Cria arquivo .out sintetico."""
    out_path = str(tmp_path / "test.out")
    with open(out_path, "w") as f:
        f.write(f"{n_angles} {n_freqs} {n_models}\n")
        f.write("0.0\n")  # theta=0
        f.write("20000.0\n")  # freq=20kHz
        f.write(f"{nmeds}\n")  # nmeds para theta=0
    return out_path


def _make_synthetic_dat(tmp_path, n_rows, n_columns=22):
    """Cria arquivo .dat binario sintetico no formato 22-col."""
    dat_path = str(tmp_path / "test.dat")
    rng = np.random.default_rng(42)

    with open(dat_path, "wb") as f:
        for i in range(n_rows):
            # Col 0: meds (int32)
            f.write(struct.pack("<i", i % 600))
            # Col 1: zobs (float64) — profundidade em metros
            z = (i % 600) * 0.25  # 0 a 150 m
            f.write(struct.pack("<d", z))
            # Col 2-3: res_h, res_v (float64)
            f.write(struct.pack("<d", 10.0 ** rng.uniform(0, 3)))
            f.write(struct.pack("<d", 10.0 ** rng.uniform(0, 3)))
            # Col 4-21: componentes EM (float64)
            for _ in range(18):
                f.write(struct.pack("<d", rng.normal(0, 1e-3)))

    return dat_path


def _make_synthetic_angle_group(n_seq=20, nmeds=600, n_feat=5, n_tgt=2):
    """Cria AngleGroup sintetico para testes de split."""
    rng = np.random.default_rng(42)
    return AngleGroup(
        theta=0.0,
        x=rng.standard_normal((n_seq, nmeds, n_feat)).astype(np.float64),
        y=np.abs(rng.standard_normal((n_seq, nmeds, n_tgt))).astype(np.float64) + 0.1,
        z_meters=np.tile(np.linspace(0, 150, nmeds), (n_seq, 1)),
        model_ids=np.arange(n_seq, dtype=np.int32),
        nmeds=nmeds,
    )


# ════════════════════════════════════════════════════════════════════════
# TESTS — Loading
# ════════════════════════════════════════════════════════════════════════


class TestParseOut:
    """parse_out_metadata: parsing de arquivo .out."""

    def test_basic_parsing(self, tmp_path):
        out_path = _make_synthetic_out(tmp_path)
        meta = parse_out_metadata(out_path)
        assert meta.n_angles == 1
        assert meta.n_freqs == 1
        assert meta.n_models == 10
        assert meta.theta_list == [0.0]
        assert meta.freq_list == [20000.0]
        assert meta.nmeds_list == [600]
        assert meta.total_rows == 10 * 600

    def test_multi_angle(self, tmp_path):
        out_path = str(tmp_path / "multi.out")
        with open(out_path, "w") as f:
            f.write("2 1 5\n")
            f.write("0.0 30.0\n")
            f.write("20000.0\n")
            f.write("600 622\n")
        meta = parse_out_metadata(out_path)
        assert meta.n_angles == 2
        assert meta.nmeds_list == [600, 622]
        assert meta.rows_per_model == 600 + 622

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_out_metadata("/nonexistent/path.out")


class TestLoadDat:
    """load_binary_dat: carregamento de .dat binario."""

    def test_load_22col(self, tmp_path):
        n_rows = 6000  # 10 models * 600 nmeds
        dat_path = _make_synthetic_dat(tmp_path, n_rows)
        data = load_binary_dat(dat_path, n_columns=22)
        assert data.shape == (n_rows, 22)
        assert data.dtype == np.float64

    def test_expected_rows_validation(self, tmp_path):
        dat_path = _make_synthetic_dat(tmp_path, 6000)
        with pytest.raises(ValueError, match="Esperadas"):
            load_binary_dat(dat_path, expected_rows=5000)

    def test_zobs_column(self, tmp_path):
        dat_path = _make_synthetic_dat(tmp_path, 600)
        data = load_binary_dat(dat_path)
        # zobs esta na coluna 1 e deve variar de 0 a ~150
        z = data[:, 1]
        assert z.min() >= 0.0
        assert z.max() <= 150.0


class TestDecoupling:
    """apply_decoupling: remocao de acoplamento direto."""

    def test_decoupling_values(self):
        config = PipelineConfig.baseline()
        data = np.zeros((10, 22), dtype=np.float64)
        data[:, 4] = 1.0  # Re{Hxx}
        data[:, 12] = 1.0  # Re{Hyy}
        data[:, 20] = 1.0  # Re{Hzz}

        result = apply_decoupling(data, config)

        L = 1.0
        ACp = -1.0 / (4.0 * math.pi * L**3)
        ACx = +1.0 / (2.0 * math.pi * L**3)

        np.testing.assert_allclose(result[:, 4], 1.0 - ACp, rtol=1e-10)
        np.testing.assert_allclose(result[:, 12], 1.0 - ACp, rtol=1e-10)
        np.testing.assert_allclose(result[:, 20], 1.0 - ACx, rtol=1e-10)

    def test_decoupling_preserves_other_columns(self):
        config = PipelineConfig.baseline()
        data = np.ones((5, 22), dtype=np.float64)
        result = apply_decoupling(data, config)
        # Coluna 1 (zobs) inalterada
        np.testing.assert_array_equal(result[:, 1], data[:, 1])


# ════════════════════════════════════════════════════════════════════════
# TESTS — Splitting
# ════════════════════════════════════════════════════════════════════════


class TestSplitting:
    """split_model_ids + apply_split: particionamento [P1]."""

    def test_split_no_overlap(self):
        train, val, test = split_model_ids(100)
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()
        assert len(train) + len(val) + len(test) == 100

    def test_split_ratios(self):
        train, val, test = split_model_ids(1000, 0.7, 0.15, 0.15)
        assert len(train) == 700
        assert len(val) == 150

    def test_split_deterministic(self):
        t1, v1, _ = split_model_ids(50, seed=42)
        t2, v2, _ = split_model_ids(50, seed=42)
        assert t1 == t2
        assert v1 == v2

    def test_apply_split_shapes(self):
        group = _make_synthetic_angle_group(n_seq=20)
        train, val, test = split_model_ids(20, 0.7, 0.15, 0.15)
        splits = apply_split(group, train, val, test)

        total = splits.x_train.shape[0] + splits.x_val.shape[0] + splits.x_test.shape[0]
        assert total == 20
        assert splits.z_train.shape[0] == splits.x_train.shape[0]
        assert splits.z_val.shape[0] == splits.x_val.shape[0]
        assert splits.z_test.shape[0] == splits.x_test.shape[0]

    def test_z_meters_preserved(self):
        group = _make_synthetic_angle_group(n_seq=10)
        train, val, test = split_model_ids(10, 0.7, 0.15, 0.15)
        splits = apply_split(group, train, val, test)

        # z_meters deve estar em metros (0 a 150), nunca normalizado
        if splits.z_train.size > 0:
            assert splits.z_train.min() >= 0.0
            assert splits.z_train.max() <= 150.0


# ════════════════════════════════════════════════════════════════════════
# TESTS — Feature Views
# ════════════════════════════════════════════════════════════════════════


class TestFeatureViews:
    """6 Feature Views com numpy."""

    @pytest.fixture
    def x_3d(self):
        """Dados sinteticos: (n_seq, seq_len, 5) = [z, Re_H1, Im_H1, Re_H2, Im_H2]."""
        rng = np.random.default_rng(42)
        return rng.uniform(0.001, 1.0, size=(4, 100, 5))

    def test_identity_passthrough(self, x_3d):
        result = apply_feature_view(x_3d, "identity")
        np.testing.assert_array_equal(result, x_3d)

    def test_raw_passthrough(self, x_3d):
        result = apply_feature_view(x_3d, "raw")
        np.testing.assert_array_equal(result, x_3d)

    def test_shape_preserved_all_views(self, x_3d):
        for view in VALID_VIEWS:
            result = apply_feature_view(x_3d, view)
            assert result.shape == x_3d.shape, f"Shape changed for view '{view}'"

    def test_z_preserved(self, x_3d):
        """z (coluna 0) nunca e modificado por nenhuma FV."""
        for view in VALID_VIEWS:
            result = apply_feature_view(x_3d, view)
            np.testing.assert_array_equal(
                result[:, :, 0],
                x_3d[:, :, 0],
                err_msg=f"z modificado em view '{view}'",
            )

    def test_invalid_view_raises(self, x_3d):
        with pytest.raises(ValueError, match="invalida"):
            apply_feature_view(x_3d, "nonexistent_view")

    def test_H1_logH2_transforms_h2(self, x_3d):
        result = apply_feature_view(x_3d, "H1_logH2")
        # Re(H1) e Im(H1) devem estar inalterados
        np.testing.assert_array_equal(result[:, :, 1], x_3d[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], x_3d[:, :, 2])
        # Col 3 deve ser log10|H2| (diferente do original)
        assert not np.allclose(result[:, :, 3], x_3d[:, :, 3])

    def test_logH1_logH2_all_transformed(self, x_3d):
        result = apply_feature_view(x_3d, "logH1_logH2")
        # Nenhum canal EM deve ser igual ao original
        for i in range(1, 5):
            assert not np.allclose(result[:, :, i], x_3d[:, :, i])

    def test_2d_input(self):
        """Feature Views devem funcionar com entrada 2D."""
        x_2d = np.random.uniform(0.001, 1.0, size=(100, 5))
        for view in VALID_VIEWS:
            result = apply_feature_view(x_2d, view)
            assert result.shape == x_2d.shape


# ════════════════════════════════════════════════════════════════════════
# TESTS — Feature Views com features EM expandidas
# ════════════════════════════════════════════════════════════════════════


class TestFeatureViewsExpanded:
    """Feature Views com features EM extras (alem de Hxx/Hzz baseline)."""

    @pytest.fixture
    def x_7feat(self):
        """Dados 3D com 7 features: [z, Re(Hxx), Im(Hxx), Re(Hxy), Im(Hxy), Re(Hzz), Im(Hzz)]."""
        rng = np.random.default_rng(42)
        return rng.uniform(0.001, 1.0, size=(4, 100, 7))

    @pytest.fixture
    def x_19feat(self):
        """Dados 3D com 19 features: [z, 18 EM (tensor 3x3 completo)]."""
        rng = np.random.default_rng(42)
        return rng.uniform(0.001, 1.0, size=(4, 100, 19))

    def test_identity_preserves_all_7(self, x_7feat):
        """identity/raw preserva todas as 7 colunas sem alteracao."""
        result = apply_feature_view(x_7feat, "identity")
        np.testing.assert_array_equal(result, x_7feat)

    def test_shape_preserved_all_views_7feat(self, x_7feat):
        """Shape (4, 100, 7) preservada para todas as FVs com h1_cols/h2_cols."""
        for view in VALID_VIEWS:
            result = apply_feature_view(x_7feat, view, h1_cols=(1, 2), h2_cols=(5, 6))
            assert result.shape == x_7feat.shape, f"Shape changed for view '{view}'"

    def test_z_preserved_expanded(self, x_7feat):
        """z (coluna 0) nunca e modificada mesmo com features extras."""
        for view in VALID_VIEWS:
            result = apply_feature_view(x_7feat, view, h1_cols=(1, 2), h2_cols=(5, 6))
            np.testing.assert_array_equal(
                result[:, :, 0],
                x_7feat[:, :, 0],
                err_msg=f"z modificado em view '{view}'",
            )

    def test_extra_em_columns_preserved(self, x_7feat):
        """Colunas EM extras (Hxy, cols 3-4) NAO sao modificadas pelas FVs."""
        for view in VALID_VIEWS:
            result = apply_feature_view(x_7feat, view, h1_cols=(1, 2), h2_cols=(5, 6))
            # Hxy (cols 3 e 4) devem permanecer intactas
            np.testing.assert_array_equal(
                result[:, :, 3],
                x_7feat[:, :, 3],
                err_msg=f"Hxy Re modificado em view '{view}'",
            )
            np.testing.assert_array_equal(
                result[:, :, 4],
                x_7feat[:, :, 4],
                err_msg=f"Hxy Im modificado em view '{view}'",
            )

    def test_H1_logH2_with_explicit_cols(self, x_7feat):
        """H1_logH2 transforma Hzz (cols 5-6) mas preserva Hxx (cols 1-2) e Hxy (cols 3-4)."""
        result = apply_feature_view(x_7feat, "H1_logH2", h1_cols=(1, 2), h2_cols=(5, 6))
        # Re(Hxx) e Im(Hxx) preservados
        np.testing.assert_array_equal(result[:, :, 1], x_7feat[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], x_7feat[:, :, 2])
        # Re(Hxy) e Im(Hxy) preservados (extra)
        np.testing.assert_array_equal(result[:, :, 3], x_7feat[:, :, 3])
        np.testing.assert_array_equal(result[:, :, 4], x_7feat[:, :, 4])
        # Hzz (cols 5-6) transformadas
        assert not np.allclose(result[:, :, 5], x_7feat[:, :, 5])

    def test_logH1_logH2_with_explicit_cols(self, x_7feat):
        """logH1_logH2 transforma Hxx (1-2) e Hzz (5-6) mas preserva Hxy (3-4)."""
        result = apply_feature_view(
            x_7feat, "logH1_logH2", h1_cols=(1, 2), h2_cols=(5, 6)
        )
        # Hxx transformado
        assert not np.allclose(result[:, :, 1], x_7feat[:, :, 1])
        assert not np.allclose(result[:, :, 2], x_7feat[:, :, 2])
        # Hxy preservado
        np.testing.assert_array_equal(result[:, :, 3], x_7feat[:, :, 3])
        np.testing.assert_array_equal(result[:, :, 4], x_7feat[:, :, 4])
        # Hzz transformado
        assert not np.allclose(result[:, :, 5], x_7feat[:, :, 5])
        assert not np.allclose(result[:, :, 6], x_7feat[:, :, 6])

    def test_full_tensor_19feat(self, x_19feat):
        """Tensor completo 19 features — FV opera apenas em H1/H2 especificados."""
        # H1=Hxx (cols 1,2), H2=Hzz (cols 17,18) no array de 19 colunas
        # [z, Re(Hxx), Im(Hxx), Re(Hxy)..., Re(Hzz), Im(Hzz)]
        result = apply_feature_view(
            x_19feat, "logH1_logH2", h1_cols=(1, 2), h2_cols=(17, 18)
        )
        assert result.shape == x_19feat.shape
        # z preservado
        np.testing.assert_array_equal(result[:, :, 0], x_19feat[:, :, 0])
        # Colunas intermediarias (3-16) preservadas
        for i in range(3, 17):
            np.testing.assert_array_equal(
                result[:, :, i],
                x_19feat[:, :, i],
                err_msg=f"Coluna {i} modificada indevidamente",
            )

    def test_backward_compat_default_cols(self, x_7feat):
        """Sem h1_cols/h2_cols, FV assume layout legado [z, H1Re, H1Im, H2Re, H2Im, ...]."""
        # Com 7 features e sem h1/h2 explicito, o default é
        # h1_cols=(1,2), h2_cols=(3,4) — posicional legado
        x_5feat = x_7feat[:, :, :5]  # layout [z, Re(H1), Im(H1), Re(H2), Im(H2)]
        result = apply_feature_view(x_5feat, "H1_logH2")
        # Re(H1) e Im(H1) preservados
        np.testing.assert_array_equal(result[:, :, 1], x_5feat[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], x_5feat[:, :, 2])
        # H2 transformado
        assert not np.allclose(result[:, :, 3], x_5feat[:, :, 3])

    def test_2d_expanded(self):
        """Feature Views devem funcionar com entrada 2D expandida."""
        x_2d = np.random.uniform(0.001, 1.0, size=(100, 7))
        result = apply_feature_view(x_2d, "logH1_logH2", h1_cols=(1, 2), h2_cols=(5, 6))
        assert result.shape == x_2d.shape


# ════════════════════════════════════════════════════════════════════════
# TESTS — Geosignals
# ════════════════════════════════════════════════════════════════════════


class TestGeosignals:
    """5 familias de geosinais."""

    @pytest.fixture
    def raw_data_22(self):
        """Dados sinteticos: (1000, 22) com componentes EM."""
        rng = np.random.default_rng(42)
        data = np.zeros((1000, 22), dtype=np.float64)
        data[:, 1] = np.linspace(0, 150, 1000)  # zobs
        data[:, 2] = rng.uniform(1, 100, 1000)  # res_h
        data[:, 3] = rng.uniform(1, 100, 1000)  # res_v
        for i in range(4, 22):
            data[:, i] = rng.normal(0, 1e-3, 1000)
        return data

    def test_expanded_features_usd_uhr(self):
        result = compute_expanded_features([1, 4, 5, 20, 21], ["USD", "UHR"])
        # USD precisa ZZ(20,21), XZ(8,9), ZX(16,17)
        # UHR precisa ZZ(20,21), XX(4,5), YY(12,13)
        assert 8 in result and 9 in result  # XZ
        assert 12 in result and 13 in result  # YY
        assert 16 in result and 17 in result  # ZX
        assert result == sorted(result)  # deve estar ordenado

    def test_expanded_features_all_base_preserved(self):
        base = [1, 4, 5, 20, 21]
        result = compute_expanded_features(base, ["USD"])
        for col in base:
            assert col in result

    def test_compute_gs_shape(self, raw_data_22):
        gs = compute_geosignals(raw_data_22, ["USD", "UHR"])
        assert gs.shape == (1000, 4)  # 2 familias × 2 canais

    def test_compute_gs_all_families(self, raw_data_22):
        all_fams = list(FAMILY_DEPS.keys())
        gs = compute_geosignals(raw_data_22, all_fams)
        assert gs.shape == (1000, 2 * len(all_fams))

    def test_compute_gs_attenuation_range(self, raw_data_22):
        gs = compute_geosignals(raw_data_22, ["UHR"])
        att = gs[:, 0]
        assert att.min() >= -100.0
        assert att.max() <= 100.0

    def test_unknown_family_raises(self, raw_data_22):
        with pytest.raises(ValueError, match="desconhecida"):
            compute_geosignals(raw_data_22, ["INVALID"])


# ════════════════════════════════════════════════════════════════════════
# TESTS — Scaling
# ════════════════════════════════════════════════════════════════════════


class TestTargetScaling:
    """apply_target_scaling + inverse_target_scaling: roundtrip."""

    def test_log10_roundtrip(self):
        y = np.array([1.0, 10.0, 100.0, 1000.0])
        scaled = apply_target_scaling(y, "log10")
        recovered = inverse_target_scaling(scaled, "log10")
        np.testing.assert_allclose(recovered, y, rtol=1e-10)

    def test_sqrt_roundtrip(self):
        y = np.array([4.0, 16.0, 25.0])
        scaled = apply_target_scaling(y, "sqrt")
        recovered = inverse_target_scaling(scaled, "sqrt")
        np.testing.assert_allclose(recovered, y, rtol=1e-10)

    def test_none_identity(self):
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(apply_target_scaling(y, "none"), y)


class TestScalers:
    """create_scaler + fit_scaler + transform_features."""

    def test_all_scaler_types(self):
        for st in ["standard", "minmax", "robust", "maxabs", "none"]:
            scaler = create_scaler(st)
            if st == "none":
                assert scaler is None
            else:
                assert scaler is not None

    def test_fit_and_transform_3d(self):
        x = np.random.randn(10, 100, 5).astype(np.float64)
        scaler = fit_scaler(x, "standard")
        x_scaled = transform_features(x, scaler)
        assert x_scaled.shape == x.shape
        # Mean should be ~0 after standard scaling
        flat = x_scaled.reshape(-1, 5)
        np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=1e-10)

    def test_none_scaler_is_copy(self):
        x = np.random.randn(5, 100, 3)
        result = transform_features(x, None)
        np.testing.assert_array_equal(result, x)
        # Must be a copy, not same object
        assert result is not x

    def test_invalid_scaler_raises(self):
        with pytest.raises(ValueError, match="invalido"):
            create_scaler("nonexistent")


# ════════════════════════════════════════════════════════════════════════
# TESTS — Pipeline Integration (sem .dat/.out reais)
# ════════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """DataPipeline: testes de integracao com dados sinteticos."""

    def test_pipeline_init_baseline(self):
        config = PipelineConfig.baseline()
        from geosteering_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(config)
        assert pipeline.is_onthefly is False

    def test_pipeline_init_geosinais(self):
        config = PipelineConfig.geosinais_p4()
        from geosteering_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(config)
        assert pipeline.is_onthefly is True

    def test_prepare_with_synthetic_data(self, tmp_path):
        """Teste end-to-end: .out + .dat sinteticos → PreparedData."""
        n_models = 10
        nmeds = 600
        n_rows = n_models * nmeds

        out_path = _make_synthetic_out(tmp_path, n_models=n_models, nmeds=nmeds)
        dat_path = _make_synthetic_dat(tmp_path, n_rows)

        config = PipelineConfig.baseline()
        from geosteering_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(config)
        data = pipeline.prepare(dat_path, out_path)

        # Shapes basicos
        assert data.x_train.ndim == 3
        assert data.y_train.ndim == 3
        assert data.z_train.ndim == 2
        assert data.x_train.shape[-1] == config.n_base_features
        assert data.y_train.shape[-1] == len(config.output_targets)

        # z_meters em metros (0 a 150), nunca normalizado
        assert data.z_train.min() >= 0.0
        assert data.z_train.max() <= 150.0
        assert data.z_test.min() >= 0.0

        # Total sequences = n_models
        total = data.x_train.shape[0] + data.x_val.shape[0] + data.x_test.shape[0]
        assert total == n_models

    def test_z_meters_not_scaled(self, tmp_path):
        """Bug fix: z_meters NUNCA deve ser normalizado."""
        out_path = _make_synthetic_out(tmp_path, n_models=10, nmeds=600)
        dat_path = _make_synthetic_dat(tmp_path, 6000)

        config = PipelineConfig.baseline()
        from geosteering_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(config)
        data = pipeline.prepare(dat_path, out_path)

        # z_meters deve estar no range fisico [0, 150] metros
        all_z = np.concatenate(
            [data.z_train.ravel(), data.z_val.ravel(), data.z_test.ravel()]
        )
        assert all_z.min() >= 0.0, "z_meters tem valores negativos (normalizado?)"
        assert all_z.max() <= 150.0, "z_meters excede 150m (dados corrompidos?)"
        # Se fosse normalizado (StandardScaler), teria media ~0 e desvio ~1
        assert all_z.mean() > 10.0, "z_meters parece normalizado (media muito baixa)"


# ════════════════════════════════════════════════════════════════════════
# TESTS — Pipeline h1_cols/h2_cols Wiring (features EM expandidas)
# ════════════════════════════════════════════════════════════════════════


class TestPipelineH1H2Wiring:
    """Verifica que DataPipeline computa e propaga h1_cols/h2_cols."""

    def test_h1h2_cols_baseline_default(self):
        """Com baseline [1,4,5,20,21], h1_cols=(1,2) h2_cols=(3,4) posicional."""
        from geosteering_ai.data.pipeline import DataPipeline

        config = PipelineConfig.baseline()
        pipeline = DataPipeline(config)
        # Default: H1=Hxx em posicoes 1,2 e H2=Hzz em posicoes 3,4
        # (indices no array EXTRAIDO, nao no .dat)
        assert pipeline._h1_cols == (1, 2)
        assert pipeline._h2_cols == (3, 4)

    def test_h1h2_cols_expanded_hxy(self):
        """Com Hxy adicionado, Hzz desloca para posicoes 5,6."""
        from geosteering_ai.data.pipeline import DataPipeline

        config = PipelineConfig(
            input_features=[1, 4, 5, 6, 7, 20, 21],
        )
        pipeline = DataPipeline(config)
        # Array extraido: [z(0), ReHxx(1), ImHxx(2), ReHxy(3), ImHxy(4), ReHzz(5), ImHzz(6)]
        assert pipeline._h1_cols == (1, 2)
        assert pipeline._h2_cols == (5, 6)

    def test_h1h2_cols_full_tensor(self):
        """Tensor 3x3 completo: Hzz nas ultimas posicoes."""
        from geosteering_ai.data.pipeline import DataPipeline

        full = [1] + list(range(4, 22))  # z + 18 EM
        config = PipelineConfig(input_features=full)
        pipeline = DataPipeline(config)
        # Re(Hxx)=col4 → pos 1, Im(Hxx)=col5 → pos 2
        # Re(Hzz)=col20 → pos 17, Im(Hzz)=col21 → pos 18
        assert pipeline._h1_cols == (1, 2)
        assert pipeline._h2_cols == (17, 18)

    def test_apply_fv_gs_uses_h1h2(self):
        """_apply_fv_gs() usa h1_cols/h2_cols em FV nao-identity."""
        from geosteering_ai.data.pipeline import DataPipeline

        config = PipelineConfig(
            input_features=[1, 4, 5, 6, 7, 20, 21],
            feature_view="logH1_logH2",
        )
        pipeline = DataPipeline(config)

        rng = np.random.default_rng(42)
        x = rng.uniform(0.001, 1.0, size=(3, 100, 7)).astype(np.float32)

        result = pipeline._apply_fv_gs(x)

        # Hxy (cols 3,4 no array) devem estar intactas
        np.testing.assert_array_equal(result[:, :, 3], x[:, :, 3])
        np.testing.assert_array_equal(result[:, :, 4], x[:, :, 4])
        # Hxx (cols 1,2) deve estar transformado (log10+phase)
        assert not np.allclose(result[:, :, 1], x[:, :, 1])
        # Hzz (cols 5,6) deve estar transformado (log10+phase)
        assert not np.allclose(result[:, :, 5], x[:, :, 5])

    def test_prepare_offline_expanded_features(self, tmp_path):
        """Prepare offline com features expandidas + FV nao-identity."""
        from geosteering_ai.data.pipeline import DataPipeline

        n_models = 10
        nmeds = 600
        n_rows = n_models * nmeds

        out_path = _make_synthetic_out(tmp_path, n_models=n_models, nmeds=nmeds)
        dat_path = _make_synthetic_dat(tmp_path, n_rows)

        config = PipelineConfig(
            input_features=[1, 4, 5, 6, 7, 20, 21],
            feature_view="logH1_logH2",
        )
        pipeline = DataPipeline(config)
        data = pipeline.prepare(dat_path, out_path)

        # Shape: 7 features base
        assert data.x_train.shape[-1] == 7
        assert data.x_val.shape[-1] == 7


# ════════════════════════════════════════════════════════════════════════
# TESTS — P2 (theta) e P3 (freq) como features na cadeia de dados
# ════════════════════════════════════════════════════════════════════════


class TestThetaFreqInjection:
    """Injecao de theta e frequencia como colunas prefixo."""

    def test_theta_injection_shape(self, tmp_path):
        """P2: x_seq ganha 1 coluna extra (theta_norm)."""
        n_models = 10
        nmeds = 600
        out_path = _make_synthetic_out(tmp_path, n_models=n_models, nmeds=nmeds)
        dat_path = _make_synthetic_dat(tmp_path, n_models * nmeds)

        config = PipelineConfig(use_theta_as_feature=True)
        from geosteering_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(config)
        data = pipeline.prepare(dat_path, out_path)
        # 5 base + 1 theta = 6 features
        assert data.x_train.shape[-1] == 6

    def test_freq_injection_shape(self, tmp_path):
        """P3: x_seq ganha 1 coluna extra (f_norm)."""
        n_models = 10
        nmeds = 600
        out_path = _make_synthetic_out(tmp_path, n_models=n_models, nmeds=nmeds)
        dat_path = _make_synthetic_dat(tmp_path, n_models * nmeds)

        config = PipelineConfig(use_freq_as_feature=True)
        from geosteering_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(config)
        data = pipeline.prepare(dat_path, out_path)
        assert data.x_train.shape[-1] == 6

    def test_theta_freq_combined_shape(self, tmp_path):
        """P2+P3: x_seq ganha 2 colunas extra (theta_norm + f_norm)."""
        n_models = 10
        nmeds = 600
        out_path = _make_synthetic_out(tmp_path, n_models=n_models, nmeds=nmeds)
        dat_path = _make_synthetic_dat(tmp_path, n_models * nmeds)

        config = PipelineConfig(
            use_theta_as_feature=True,
            use_freq_as_feature=True,
        )
        from geosteering_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(config)
        data = pipeline.prepare(dat_path, out_path)
        assert data.x_train.shape[-1] == 7

    def test_theta_constant_per_sequence(self, tmp_path):
        """theta_norm deve ser constante ao longo de toda a sequencia."""
        n_models = 10
        nmeds = 600
        out_path = _make_synthetic_out(tmp_path, n_models=n_models, nmeds=nmeds)
        dat_path = _make_synthetic_dat(tmp_path, n_models * nmeds)

        config = PipelineConfig(use_theta_as_feature=True)
        from geosteering_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(config)
        data = pipeline.prepare(dat_path, out_path)
        # Col 0 = theta_norm, deve ser constante ao longo de seq_len (axis=1)
        for i in range(data.x_train.shape[0]):
            theta_seq = data.x_train[i, :, 0]
            assert np.all(
                theta_seq == theta_seq[0]
            ), f"theta_norm nao constante na sequencia {i}"

    def test_theta_value_normalized(self, tmp_path):
        """theta_norm deve estar em [0, 1] (theta/90)."""
        n_models = 10
        nmeds = 600
        out_path = _make_synthetic_out(tmp_path, n_models=n_models, nmeds=nmeds)
        dat_path = _make_synthetic_dat(tmp_path, n_models * nmeds)

        config = PipelineConfig(use_theta_as_feature=True)
        from geosteering_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(config)
        data = pipeline.prepare(dat_path, out_path)
        theta_col = data.x_train[:, 0, 0]  # primeiro ponto de cada sequencia
        assert np.all(theta_col >= 0.0)
        assert np.all(theta_col <= 1.0)

    def test_noise_preserves_theta_freq(self):
        """Noise NAO deve afetar colunas prefixo (theta, freq) nem z_obs."""
        from geosteering_ai.noise.functions import apply_raw_em_noise

        rng = np.random.default_rng(42)
        # Layout P2+P3: [theta, freq, z, ReHxx, ImHxx, ReHzz, ImHzz]
        x = rng.uniform(0.1, 1.0, size=(3, 100, 7)).astype(np.float32)
        x_noisy = apply_raw_em_noise(x, noise_level=0.1, n_protected=3)

        # theta (col 0), freq (col 1), z (col 2) devem estar intactos
        np.testing.assert_array_equal(x_noisy[:, :, 0], x[:, :, 0])
        np.testing.assert_array_equal(x_noisy[:, :, 1], x[:, :, 1])
        np.testing.assert_array_equal(x_noisy[:, :, 2], x[:, :, 2])
        # EM (cols 3-6) devem ser diferentes (ruidosos)
        assert not np.allclose(x_noisy[:, :, 3], x[:, :, 3])
