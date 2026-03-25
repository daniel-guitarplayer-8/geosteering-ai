"""Testes unitarios para PipelineConfig.

Valida errata, mutual exclusivity, presets, YAML roundtrip,
e propriedades derivadas.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geosteering_ai.config import PipelineConfig


class TestErrata:
    """Valores fisicos criticos — Errata v4.4.5 + v5.0.15."""

    def test_frequency_must_be_20000(self):
        with pytest.raises(AssertionError, match="20000.0"):
            PipelineConfig(frequency_hz=2.0)

    def test_spacing_must_be_1(self):
        with pytest.raises(AssertionError, match="1.0"):
            PipelineConfig(spacing_meters=1000.0)

    def test_sequence_length_must_be_600(self):
        with pytest.raises(AssertionError, match="600"):
            PipelineConfig(sequence_length=601)

    def test_target_scaling_must_be_log10(self):
        with pytest.raises(AssertionError, match="log10"):
            PipelineConfig(target_scaling="log")

    def test_input_features_22col(self):
        with pytest.raises(AssertionError):
            PipelineConfig(input_features=[0, 3, 4, 7, 8])

    def test_output_targets_22col(self):
        with pytest.raises(AssertionError):
            PipelineConfig(output_targets=[1, 2])


class TestMutualExclusivity:
    """N-Stage e Curriculum sao mutuamente exclusivos."""

    def test_nstage_and_curriculum_raises(self):
        with pytest.raises(AssertionError, match="mutuamente exclusivos"):
            PipelineConfig(use_nstage=True, use_curriculum=True)

    def test_nstage_without_curriculum_ok(self):
        config = PipelineConfig(use_nstage=True, use_curriculum=False)
        assert config.use_nstage is True
        assert config.use_curriculum is False

    def test_curriculum_without_nstage_ok(self):
        config = PipelineConfig(use_nstage=False, use_curriculum=True)
        assert config.use_curriculum is True


class TestPresets:
    """Presets de classe produzem configs validas."""

    def test_baseline(self):
        config = PipelineConfig.baseline()
        assert config.use_noise is False
        assert config.use_curriculum is False
        assert config.model_type == "ResNet_18"

    def test_robusto(self):
        config = PipelineConfig.robusto()
        assert config.noise_level_max == 0.08
        assert config.learning_rate == 1e-4
        assert config.use_curriculum is True
        assert config.epochs_no_noise == 10
        assert config.noise_ramp_epochs == 80

    def test_nstage_n2(self):
        config = PipelineConfig.nstage(n=2)
        assert config.use_nstage is True
        assert config.n_training_stages == 2
        assert config.use_curriculum is False

    def test_nstage_n3(self):
        config = PipelineConfig.nstage(n=3)
        assert config.n_training_stages == 3

    def test_geosinais_p4(self):
        config = PipelineConfig.geosinais_p4()
        assert config.use_geosignal_features is True
        assert config.geosignal_set == "usd_uhr"

    def test_realtime(self):
        config = PipelineConfig.realtime()
        assert config.inference_mode == "realtime"
        assert config.use_causal_mode is True
        assert config.model_type == "WaveNet"

    def test_realtime_custom_model(self):
        config = PipelineConfig.realtime(model_type="Causal_Transformer")
        assert config.model_type == "Causal_Transformer"


class TestDerivedProperties:
    """Propriedades derivadas calculadas corretamente."""

    def test_n_base_features(self):
        config = PipelineConfig.baseline()
        assert config.n_base_features == 5

    def test_n_features_without_gs(self):
        config = PipelineConfig.baseline()
        assert config.n_features == 5

    def test_n_features_with_gs_usd_uhr(self):
        config = PipelineConfig.geosinais_p4()
        assert config.n_geosignal_channels == 4  # 2 familias × 2 canais
        assert config.n_features == 9  # 5 + 4

    def test_needs_onthefly_noise_and_gs(self):
        config = PipelineConfig.geosinais_p4()
        assert config.needs_onthefly_fv_gs is True

    def test_not_needs_onthefly_baseline(self):
        config = PipelineConfig.baseline()
        assert config.needs_onthefly_fv_gs is False

    def test_not_needs_onthefly_noise_but_identity_fv(self):
        config = PipelineConfig.robusto()  # noise=True, FV=identity, GS=off
        assert config.needs_onthefly_fv_gs is False

    def test_realtime_auto_causal(self):
        config = PipelineConfig(inference_mode="realtime", use_causal_mode=False)
        assert config.use_causal_mode is True  # auto-derivado

    def test_resolve_families_usd_uhr(self):
        config = PipelineConfig.geosinais_p4()
        assert config.resolve_families() == ["USD", "UHR"]

    def test_resolve_families_full_3d(self):
        config = PipelineConfig.geosinais_p4(geosignal_set="full_3d")
        assert config.resolve_families() == ["USD", "UAD", "UHR", "UHA", "U3DF"]


class TestSerialization:
    """Serializacao YAML e dict."""

    def test_to_dict(self):
        config = PipelineConfig.robusto()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["learning_rate"] == 1e-4
        assert d["noise_level_max"] == 0.08

    def test_copy_with_override(self):
        config = PipelineConfig.robusto()
        config2 = config.copy(learning_rate=3e-4)
        assert config2.learning_rate == 3e-4
        assert config.learning_rate == 1e-4  # original inalterado

    def test_yaml_roundtrip(self, tmp_path):
        config = PipelineConfig.robusto()
        path = str(tmp_path / "test_config.yaml")
        config.to_yaml(path)
        loaded = PipelineConfig.from_yaml(path)
        assert config.to_dict() == loaded.to_dict()

    def test_from_yaml_preset(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "robusto.yaml"
        )
        if os.path.exists(config_path):
            config = PipelineConfig.from_yaml(config_path)
            assert config.noise_level_max == 0.08
            assert config.use_curriculum is True


class TestValidation:
    """Validacoes de ranges e consistencia."""

    def test_noise_level_range(self):
        with pytest.raises(AssertionError):
            PipelineConfig(noise_level_max=1.5)

    def test_batch_size_positive(self):
        with pytest.raises(AssertionError):
            PipelineConfig(batch_size=0)

    def test_noise_types_weights_length(self):
        with pytest.raises(AssertionError):
            PipelineConfig(noise_types=["gaussian", "multiplicative"], noise_weights=[1.0])

    def test_feature_view_valid(self):
        with pytest.raises(AssertionError, match="invalido"):
            PipelineConfig(feature_view="invalid_view")

    def test_output_channels_valid(self):
        with pytest.raises(AssertionError):
            PipelineConfig(output_channels=3)

    def test_inference_mode_valid(self):
        with pytest.raises(AssertionError, match="invalido"):
            PipelineConfig(inference_mode="streaming")

    def test_scaler_type_valid(self):
        with pytest.raises(AssertionError, match="invalido"):
            PipelineConfig(scaler_type="invalid_scaler")

    def test_optimizer_valid(self):
        with pytest.raises(AssertionError, match="invalido"):
            PipelineConfig(optimizer="invalid_opt")

    def test_smoothing_type_valid(self):
        with pytest.raises(AssertionError, match="invalido"):
            PipelineConfig(smoothing_type="cubic_spline")
