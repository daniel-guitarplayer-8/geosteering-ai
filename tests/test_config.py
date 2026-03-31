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

    def test_frequency_hz_valid_range(self):
        """frequency_hz aceita range [100, 1e6] Hz."""
        # Valores validos (LWD comercial)
        config_2k = PipelineConfig(frequency_hz=2000.0)
        assert config_2k.frequency_hz == 2000.0
        config_400k = PipelineConfig(frequency_hz=400000.0)
        assert config_400k.frequency_hz == 400000.0
        # Default 20 kHz
        config_default = PipelineConfig()
        assert config_default.frequency_hz == 20000.0
        # Boundary values (extremos do range)
        config_min = PipelineConfig(frequency_hz=100.0)
        assert config_min.frequency_hz == 100.0
        config_max = PipelineConfig(frequency_hz=1e6)
        assert config_max.frequency_hz == 1e6

    def test_frequency_hz_rejects_out_of_range(self):
        """frequency_hz rejeita valores fora de [100, 1e6]."""
        with pytest.raises(AssertionError, match="frequency_hz"):
            PipelineConfig(frequency_hz=50.0)  # Abaixo do minimo
        with pytest.raises(AssertionError, match="frequency_hz"):
            PipelineConfig(frequency_hz=2e6)  # Acima do maximo

    def test_spacing_meters_valid_range(self):
        """spacing_meters aceita range [0.1, 10.0] m."""
        config_near = PipelineConfig(spacing_meters=0.25)
        assert config_near.spacing_meters == 0.25
        config_deep = PipelineConfig(spacing_meters=5.0)
        assert config_deep.spacing_meters == 5.0
        # Default 1.0 m
        config_default = PipelineConfig()
        assert config_default.spacing_meters == 1.0
        # Boundary values (extremos do range)
        config_min = PipelineConfig(spacing_meters=0.1)
        assert config_min.spacing_meters == 0.1
        config_max = PipelineConfig(spacing_meters=10.0)
        assert config_max.spacing_meters == 10.0

    def test_spacing_meters_rejects_out_of_range(self):
        """spacing_meters rejeita valores fora de [0.1, 10.0]."""
        with pytest.raises(AssertionError, match="spacing_meters"):
            PipelineConfig(spacing_meters=0.01)  # Abaixo
        with pytest.raises(AssertionError, match="spacing_meters"):
            PipelineConfig(spacing_meters=1000.0)  # Muito acima

    def test_sequence_length_valid_range(self):
        """sequence_length aceita range [10, 100000]."""
        config_300 = PipelineConfig(sequence_length=300)
        assert config_300.sequence_length == 300
        config_1200 = PipelineConfig(sequence_length=1200)
        assert config_1200.sequence_length == 1200
        # Default 600
        config_default = PipelineConfig()
        assert config_default.sequence_length == 600
        # Boundary values (extremos do range)
        config_min = PipelineConfig(sequence_length=10)
        assert config_min.sequence_length == 10
        config_max = PipelineConfig(sequence_length=100000)
        assert config_max.sequence_length == 100000

    def test_sequence_length_rejects_out_of_range(self):
        """sequence_length rejeita valores fora de [10, 100000]."""
        with pytest.raises(AssertionError, match="sequence_length"):
            PipelineConfig(sequence_length=5)  # Abaixo
        with pytest.raises(AssertionError, match="sequence_length"):
            PipelineConfig(sequence_length=200000)  # Acima

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
            PipelineConfig(
                noise_types=["gaussian", "multiplicative"], noise_weights=[1.0]
            )

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

    def test_eps_tf_too_small(self):
        """eps_tf=1e-30 proibido pela Errata v5.0.15 (float32 unsafe)."""
        with pytest.raises(AssertionError, match="1e-15"):
            PipelineConfig(eps_tf=1e-30)

    def test_eps_tf_valid(self):
        config = PipelineConfig(eps_tf=1e-12)
        assert config.eps_tf == 1e-12

    def test_train_ratio_zero_raises(self):
        with pytest.raises(AssertionError, match="train_ratio"):
            PipelineConfig(train_ratio=0.0)

    def test_val_ratio_zero_raises(self):
        with pytest.raises(AssertionError, match="val_ratio"):
            PipelineConfig(val_ratio=0.0)


class TestFrozenImmutability:
    """PipelineConfig eh frozen (imutavel apos construcao)."""

    def test_cannot_mutate_after_init(self):
        config = PipelineConfig.baseline()
        with pytest.raises(AttributeError):
            config.learning_rate = 0.999

    def test_errata_fields_immutable(self):
        config = PipelineConfig()
        with pytest.raises(AttributeError):
            config.frequency_hz = 2.0

    def test_copy_creates_new_instance(self):
        config = PipelineConfig.robusto()
        config2 = config.copy(learning_rate=3e-4)
        assert config2.learning_rate == 3e-4
        assert config.learning_rate == 1e-4


class TestLossConfigFields:
    """Verifica que os 22 campos de loss avancada existem com defaults corretos."""

    def test_geophysical_thresholds_exist(self):
        config = PipelineConfig()
        assert config.penalty_warmup_epochs == 10
        assert config.interface_threshold == 0.5
        assert config.high_rho_threshold == 300.0
        assert config.low_rho_threshold == 50.0

    def test_gangorra_fields_exist(self):
        config = PipelineConfig()
        assert config.gangorra_beta_min == 0.1
        assert config.gangorra_beta_max == 0.5
        assert config.gangorra_max_noise == 0.1

    def test_robust_fields_exist(self):
        config = PipelineConfig()
        assert config.robust_alpha == 0.15
        assert config.robust_beta == 0.1
        assert config.robust_gamma == 0.15
        assert config.robust_delta_smooth == 0.05

    def test_advanced_loss_fields_exist(self):
        config = PipelineConfig()
        assert config.look_ahead_decay_rate == 10.0
        assert config.dilate_alpha == 0.5
        assert config.dilate_gamma == 0.01
        assert config.dilate_downsample == 10
        assert config.enc_decoder_recon_weight == 0.1
        assert config.sobolev_lambda == 0.1
        assert config.cross_gradient_lambda == 0.1
        assert config.spectral_lambda == 0.5

    def test_morales_fields_exist(self):
        config = PipelineConfig()
        assert config.use_adaptive_omega is False
        assert config.morales_omega_initial == 0.15
        assert config.morales_ramp_epochs == 50

    def test_yaml_roundtrip_new_fields(self):
        """Novos campos devem sobreviver YAML roundtrip."""
        import os
        import tempfile

        config = PipelineConfig(
            penalty_warmup_epochs=20,
            high_rho_threshold=500.0,
            gangorra_beta_max=0.7,
            sobolev_lambda=0.05,
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            config.to_yaml(path)
            loaded = PipelineConfig.from_yaml(path)
            assert loaded.penalty_warmup_epochs == 20
            assert loaded.high_rho_threshold == 500.0
            assert loaded.gangorra_beta_max == 0.7
            assert loaded.sobolev_lambda == 0.05
        finally:
            os.unlink(path)


# ════════════════════════════════════════════════════════════════════════
# TESTS — Flexible Input Features (suporte a features EM expandidas)
# ════════════════════════════════════════════════════════════════════════


class TestFlexibleInputFeatures:
    """Validacao semantica de input_features — baseline obrigatorio, extensoes permitidas."""

    def test_default_baseline_accepted(self):
        """Default [1,4,5,20,21] deve continuar funcionando."""
        config = PipelineConfig()
        assert config.input_features == [1, 4, 5, 20, 21]

    def test_old_format_9col_rejected(self):
        """Formato antigo [0,3,4,7,8] do legado 9-col DEVE ser rejeitado.

        Col 0 (meds) eh metadata proibida. Col 3 overlap com targets.
        Baseline {1,4,5,20,21} faltando. Multiplas validacoes pegam.
        """
        with pytest.raises(AssertionError, match="metadata"):
            PipelineConfig(input_features=[0, 3, 4, 7, 8])

    def test_old_format_partial_rejected(self):
        """Mesmo 1 indice de metadata (col 0) misturado DEVE ser rejeitado."""
        with pytest.raises(AssertionError, match="metadata"):
            PipelineConfig(input_features=[0, 1, 4, 5, 20, 21])

    def test_baseline_subset_missing_hxx_rejected(self):
        """Remover Hxx (cols 4,5) do baseline DEVE ser rejeitado."""
        with pytest.raises(AssertionError, match="baseline"):
            PipelineConfig(input_features=[1, 20, 21])

    def test_baseline_subset_missing_hzz_rejected(self):
        """Remover Hzz (cols 20,21) do baseline DEVE ser rejeitado."""
        with pytest.raises(AssertionError, match="baseline"):
            PipelineConfig(input_features=[1, 4, 5])

    def test_baseline_plus_hxy_accepted(self):
        """Baseline + Hxy (cols 6,7) deve ser aceito."""
        config = PipelineConfig(input_features=[1, 4, 5, 6, 7, 20, 21])
        assert config.input_features == [1, 4, 5, 6, 7, 20, 21]
        assert config.n_base_features == 7

    def test_baseline_plus_hxz_accepted(self):
        """Baseline + Hxz (cols 8,9) deve ser aceito."""
        config = PipelineConfig(input_features=[1, 4, 5, 8, 9, 20, 21])
        assert config.input_features == [1, 4, 5, 8, 9, 20, 21]
        assert config.n_base_features == 7

    def test_full_tensor_accepted(self):
        """Tensor completo 3x3 (z + 18 EM) deve ser aceito."""
        full = [1] + list(range(4, 22))
        config = PipelineConfig(input_features=full)
        assert config.n_base_features == 19

    def test_out_of_range_index_rejected(self):
        """Indices fora de [0, n_columns) devem ser rejeitados."""
        with pytest.raises(AssertionError, match="range"):
            PipelineConfig(input_features=[1, 4, 5, 20, 21, 25])

    def test_negative_index_rejected(self):
        """Indices negativos devem ser rejeitados."""
        with pytest.raises(AssertionError, match="range"):
            PipelineConfig(input_features=[-1, 1, 4, 5, 20, 21])

    def test_overlap_with_targets_rejected(self):
        """Features que incluem targets (cols 2,3) devem ser rejeitadas."""
        with pytest.raises(AssertionError, match="overlap"):
            PipelineConfig(input_features=[1, 2, 4, 5, 20, 21])

    def test_n_features_dynamic_with_extra_em(self):
        """n_features deve refletir features extras."""
        config = PipelineConfig(input_features=[1, 4, 5, 6, 7, 20, 21])
        assert config.n_features == 7  # sem GS

    def test_n_features_dynamic_with_extra_em_and_gs(self):
        """n_features deve somar features extras + geosignais."""
        config = PipelineConfig(
            input_features=[1, 4, 5, 6, 7, 20, 21],
            use_geosignal_features=True,
            geosignal_set="usd_uhr",
        )
        # 7 base + 4 GS (2 familias x 2 canais)
        assert config.n_features == 11

    def test_yaml_roundtrip_extended_features(self, tmp_path):
        """Features estendidas devem sobreviver YAML roundtrip."""
        config = PipelineConfig(input_features=[1, 4, 5, 6, 7, 20, 21])
        path = str(tmp_path / "flex_config.yaml")
        config.to_yaml(path)
        loaded = PipelineConfig.from_yaml(path)
        assert loaded.input_features == [1, 4, 5, 6, 7, 20, 21]

    def test_copy_preserves_extended_features(self):
        """copy() deve preservar features estendidas."""
        config = PipelineConfig(input_features=[1, 4, 5, 8, 9, 20, 21])
        config2 = config.copy(learning_rate=3e-4)
        assert config2.input_features == [1, 4, 5, 8, 9, 20, 21]


# ════════════════════════════════════════════════════════════════════════
# TESTS — Perspectivas P2 (theta) e P3 (frequencia) como features
# ════════════════════════════════════════════════════════════════════════


class TestThetaFreqFeatures:
    """Perspectivas P2 (theta) e P3 (freq) como features de entrada."""

    def test_p1_n_prefix_zero(self):
        """P1 baseline: n_prefix = 0 (sem theta, sem freq)."""
        config = PipelineConfig()
        assert config.n_prefix == 0

    def test_p2_n_prefix_one(self):
        """P2 theta: n_prefix = 1."""
        config = PipelineConfig(use_theta_as_feature=True)
        assert config.n_prefix == 1

    def test_p3_n_prefix_one(self):
        """P3 freq: n_prefix = 1."""
        config = PipelineConfig(use_freq_as_feature=True)
        assert config.n_prefix == 1

    def test_p2p3_n_prefix_two(self):
        """P2+P3: n_prefix = 2."""
        config = PipelineConfig(
            use_theta_as_feature=True,
            use_freq_as_feature=True,
        )
        assert config.n_prefix == 2

    def test_n_features_with_theta(self):
        """n_features com theta: n_base + 1 (theta) + 0 (GS)."""
        config = PipelineConfig(use_theta_as_feature=True)
        assert config.n_features == 6  # 5 base + 1 theta

    def test_n_features_with_freq(self):
        """n_features com freq: n_base + 1 (freq) + 0 (GS)."""
        config = PipelineConfig(use_freq_as_feature=True)
        assert config.n_features == 6  # 5 base + 1 freq

    def test_n_features_with_theta_freq(self):
        """n_features com theta+freq: n_base + 2."""
        config = PipelineConfig(
            use_theta_as_feature=True,
            use_freq_as_feature=True,
        )
        assert config.n_features == 7  # 5 base + 2

    def test_n_features_with_theta_freq_gs(self):
        """n_features com theta+freq+GS: n_base + 2 + 4 GS."""
        config = PipelineConfig(
            use_theta_as_feature=True,
            use_freq_as_feature=True,
            use_geosignal_features=True,
            geosignal_set="usd_uhr",
        )
        assert config.n_features == 11  # 5 base + 2 prefix + 4 GS

    def test_freq_normalization_valid(self):
        """freq_normalization deve aceitar log10, khz, raw."""
        for norm in ("log10", "khz", "raw"):
            config = PipelineConfig(
                use_freq_as_feature=True,
                freq_normalization=norm,
            )
            assert config.freq_normalization == norm

    def test_freq_normalization_invalid(self):
        """freq_normalization invalido deve falhar."""
        with pytest.raises(AssertionError, match="freq_normalization"):
            PipelineConfig(
                use_freq_as_feature=True,
                freq_normalization="invalid",
            )

    def test_yaml_roundtrip_p2p3(self, tmp_path):
        """P2+P3 devem sobreviver YAML roundtrip."""
        config = PipelineConfig(
            use_theta_as_feature=True,
            use_freq_as_feature=True,
            freq_normalization="log10",
        )
        path = str(tmp_path / "p2p3.yaml")
        config.to_yaml(path)
        loaded = PipelineConfig.from_yaml(path)
        assert loaded.use_theta_as_feature is True
        assert loaded.use_freq_as_feature is True
        assert loaded.freq_normalization == "log10"
        assert loaded.n_prefix == 2


# ════════════════════════════════════════════════════════════════════════
# TESTS — static_injection_mode (Abordagens A/B/C)
# ════════════════════════════════════════════════════════════════════════


class TestStaticInjectionMode:
    """Abordagens A (broadcast), B (dual_input), C (film)."""

    def test_default_is_broadcast(self):
        """Default: Abordagem A (broadcast)."""
        config = PipelineConfig()
        assert config.static_injection_mode == "broadcast"

    def test_dual_input_accepted(self):
        """Abordagem B: dual_input."""
        config = PipelineConfig(
            use_theta_as_feature=True,
            static_injection_mode="dual_input",
        )
        assert config.static_injection_mode == "dual_input"

    def test_film_accepted(self):
        """Abordagem C: film."""
        config = PipelineConfig(
            use_theta_as_feature=True,
            static_injection_mode="film",
        )
        assert config.static_injection_mode == "film"

    def test_invalid_mode_rejected(self):
        """Modo invalido deve falhar."""
        with pytest.raises(AssertionError, match="static_injection_mode"):
            PipelineConfig(static_injection_mode="invalid")

    def test_film_restricted_to_compatible_archs(self):
        """FiLM so deve ser permitido para arquiteturas compativeis."""
        # ResNet_18 eh compativel com FiLM
        config = PipelineConfig(
            use_theta_as_feature=True,
            static_injection_mode="film",
            model_type="ResNet_18",
        )
        assert config.static_injection_mode == "film"

    def test_film_incompatible_arch_rejected(self):
        """FiLM com arquitetura incompativel deve falhar."""
        with pytest.raises(AssertionError, match="film"):
            PipelineConfig(
                use_theta_as_feature=True,
                static_injection_mode="film",
                model_type="N_BEATS",
            )

    def test_dual_input_all_archs_accepted(self):
        """dual_input deve funcionar com qualquer arquitetura."""
        for mt in ("ResNet_18", "LSTM", "TFT", "N_BEATS", "FNO", "WaveNet"):
            config = PipelineConfig(
                use_theta_as_feature=True,
                static_injection_mode="dual_input",
                model_type=mt,
            )
            assert config.static_injection_mode == "dual_input"

    def test_broadcast_without_theta_freq_ok(self):
        """broadcast sem theta/freq ativo: sem prefixo."""
        config = PipelineConfig(static_injection_mode="broadcast")
        assert config.n_prefix == 0

    def test_dual_input_n_prefix_zero(self):
        """dual_input: n_prefix=0 (escalares separados, nao no array)."""
        config = PipelineConfig(
            use_theta_as_feature=True,
            static_injection_mode="dual_input",
        )
        # Em dual_input, theta/freq NAO sao injetados como prefixo
        assert config.n_prefix == 0
