"""Testes para Estrategias B e C — Oversampling, Curriculum Rho, Second-Order Features.

Valida:
  - Oversampling de alta resistividade (data/sampling.py)
  - Curriculum de rho (training/callbacks.py: RhoCurriculumCallback)
  - Features de 2o grau (data/second_order.py)
  - Feature View "second_order" (data/feature_views.py)
  - Config fields e validacoes

Referencia: Estrategias B e C dos relatorios de alta resistividade.
"""

import numpy as np
import pytest

from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.feature_views import VALID_VIEWS, apply_feature_view
from geosteering_ai.data.sampling import (
    compute_rho_max_per_sequence,
    filter_by_rho_max,
    oversample_high_rho,
)
from geosteering_ai.data.second_order import compute_second_order_features

# ════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mixed_rho_dataset():
    """Dataset com mix de baixa e alta resistividade.

    10 sequencias, 50 pontos cada, 5 features.
    Targets: rho_h varia de 5 a 5000 Ohm.m entre sequencias.
    """
    n_seq, seq_len, n_feat = 10, 50, 5
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n_seq, seq_len, n_feat)).astype(np.float32)
    z = np.tile(np.arange(seq_len, dtype=np.float64)[np.newaxis, :], (n_seq, 1))

    # rho_h variando: 5 sequencias com rho baixo, 5 com rho alto
    rho_values = [5, 10, 20, 50, 80, 150, 500, 1000, 3000, 5000]
    y = np.zeros((n_seq, seq_len, 2), dtype=np.float32)
    for i, rho in enumerate(rho_values):
        y[i, :, 0] = rho  # rho_h
        y[i, :, 1] = rho * 1.5  # rho_v

    return x, y, z


@pytest.fixture
def em_data_3d():
    """Array 3D com layout EM: [z, Re(H1), Im(H1), Re(H2), Im(H2)]."""
    n_seq, seq_len = 5, 100
    rng = np.random.default_rng(123)
    x = rng.standard_normal((n_seq, seq_len, 5)).astype(np.float32)
    # Garantir que Im nao seja zero (para Re/Im ratio)
    x[:, :, 2] = np.clip(x[:, :, 2], 0.1, None)
    x[:, :, 4] = np.clip(x[:, :, 4], 0.1, None)
    return x


# ════════════════════════════════════════════════════════════════════════════
# TESTES: compute_rho_max_per_sequence
# ════════════════════════════════════════════════════════════════════════════


class TestComputeRhoMax:
    """Testes para compute_rho_max_per_sequence."""

    def test_output_shape(self, mixed_rho_dataset):
        """Output shape deve ser (n_seq,)."""
        _, y, _ = mixed_rho_dataset
        rho_max = compute_rho_max_per_sequence(y)
        assert rho_max.shape == (10,)

    def test_values_correct(self, mixed_rho_dataset):
        """Valores devem corresponder ao max de rho_h por sequencia."""
        _, y, _ = mixed_rho_dataset
        rho_max = compute_rho_max_per_sequence(y)
        expected = [5, 10, 20, 50, 80, 150, 500, 1000, 3000, 5000]
        np.testing.assert_allclose(rho_max, expected)

    def test_uses_channel_0(self):
        """Deve usar canal 0 (rho_h), nao canal 1 (rho_v)."""
        y = np.zeros((3, 10, 2))
        y[0, :, 0] = 100  # rho_h
        y[0, :, 1] = 200  # rho_v (maior, mas deve ser ignorado)
        rho_max = compute_rho_max_per_sequence(y)
        assert rho_max[0] == 100.0


# ════════════════════════════════════════════════════════════════════════════
# TESTES: oversample_high_rho
# ════════════════════════════════════════════════════════════════════════════


class TestOversampleHighRho:
    """Testes para oversampling de alta resistividade."""

    def test_dataset_expands(self, mixed_rho_dataset):
        """Dataset deve crescer com oversampling."""
        x, y, z = mixed_rho_dataset
        config = PipelineConfig(
            use_rho_oversampling=True,
            rho_oversampling_threshold=100.0,
            rho_oversampling_factor=3,
        )
        x_new, y_new, z_new = oversample_high_rho(x, y, z, config)
        # 5 sequencias com rho > 100, fator 3 → +10 extras
        assert x_new.shape[0] == 10 + 5 * 2  # original + 2 extras cada

    def test_low_rho_unchanged(self, mixed_rho_dataset):
        """Sequencias de baixa rho nao devem ser duplicadas."""
        x, y, z = mixed_rho_dataset
        config = PipelineConfig(
            use_rho_oversampling=True,
            rho_oversampling_threshold=100.0,
            rho_oversampling_factor=2,
        )
        _, y_new, _ = oversample_high_rho(x, y, z, config)
        # Contar sequencias com rho_max <= 100
        rho_max = compute_rho_max_per_sequence(y_new)
        n_low = (rho_max <= 100).sum()
        assert n_low == 5  # nao duplicadas

    def test_no_high_rho_returns_original(self):
        """Se nenhuma sequencia supera threshold, retorna original."""
        n_seq, seq_len = 5, 50
        x = np.ones((n_seq, seq_len, 5))
        y = np.full((n_seq, seq_len, 2), 10.0)  # tudo rho=10
        z = np.ones((n_seq, seq_len))
        config = PipelineConfig(
            use_rho_oversampling=True,
            rho_oversampling_threshold=100.0,
            rho_oversampling_factor=3,
        )
        x_new, y_new, z_new = oversample_high_rho(x, y, z, config)
        assert x_new.shape[0] == n_seq  # inalterado

    def test_shapes_consistent(self, mixed_rho_dataset):
        """x, y, z devem ter mesmo n_seq apos oversampling."""
        x, y, z = mixed_rho_dataset
        config = PipelineConfig(
            use_rho_oversampling=True,
            rho_oversampling_threshold=100.0,
            rho_oversampling_factor=3,
        )
        x_new, y_new, z_new = oversample_high_rho(x, y, z, config)
        assert x_new.shape[0] == y_new.shape[0] == z_new.shape[0]


# ════════════════════════════════════════════════════════════════════════════
# TESTES: filter_by_rho_max
# ════════════════════════════════════════════════════════════════════════════


class TestFilterByRhoMax:
    """Testes para filtro de curriculum de rho."""

    def test_basic_filter(self):
        """Filtra corretamente por threshold."""
        rho_max = np.array([10, 50, 200, 5000])
        mask = filter_by_rho_max(rho_max, 100.0)
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_all_below_threshold(self):
        """Tudo abaixo: mascara toda True."""
        rho_max = np.array([10, 20, 30])
        mask = filter_by_rho_max(rho_max, 100.0)
        assert mask.all()

    def test_all_above_threshold(self):
        """Tudo acima: mascara toda False."""
        rho_max = np.array([200, 300, 500])
        mask = filter_by_rho_max(rho_max, 100.0)
        assert not mask.any()

    def test_boundary_value(self):
        """Valor exatamente no threshold: incluido (<=)."""
        rho_max = np.array([100.0])
        mask = filter_by_rho_max(rho_max, 100.0)
        assert mask[0]


# ════════════════════════════════════════════════════════════════════════════
# TESTES: RhoCurriculumCallback
# ════════════════════════════════════════════════════════════════════════════


class TestRhoCurriculumCallback:
    """Testes para callback de curriculum de rho."""

    @pytest.fixture
    def curriculum_config(self):
        """Config para curriculum de rho."""
        return PipelineConfig(
            use_rho_curriculum=True,
            rho_curriculum_epochs_easy=10,
            rho_curriculum_epochs_ramp=40,
            rho_curriculum_rho_max_start=100.0,
            rho_curriculum_rho_max_end=10000.0,
        )

    def test_phase_easy(self, curriculum_config):
        """Fase easy: threshold = rho_max_start."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow nao disponivel")
        from geosteering_ai.training.callbacks import RhoCurriculumCallback

        var = tf.Variable(0.0, dtype=tf.float32)
        cb = RhoCurriculumCallback(var, curriculum_config)
        cb.on_epoch_begin(0)
        assert var.numpy() == pytest.approx(100.0)
        cb.on_epoch_begin(9)
        assert var.numpy() == pytest.approx(100.0)

    def test_phase_ramp(self, curriculum_config):
        """Fase ramp: threshold cresce linearmente."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow nao disponivel")
        from geosteering_ai.training.callbacks import RhoCurriculumCallback

        var = tf.Variable(0.0, dtype=tf.float32)
        cb = RhoCurriculumCallback(var, curriculum_config)
        cb.on_epoch_begin(30)  # midpoint of ramp (10 + 20/40)
        # Progress = (30-10)/40 = 0.5 → threshold = 100 + 0.5*9900 = 5050
        assert var.numpy() == pytest.approx(5050.0)

    def test_phase_full(self, curriculum_config):
        """Fase full: threshold = rho_max_end."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow nao disponivel")
        from geosteering_ai.training.callbacks import RhoCurriculumCallback

        var = tf.Variable(0.0, dtype=tf.float32)
        cb = RhoCurriculumCallback(var, curriculum_config)
        cb.on_epoch_begin(100)  # well past ramp
        assert var.numpy() == pytest.approx(10000.0)


# ════════════════════════════════════════════════════════════════════════════
# TESTES: compute_second_order_features
# ════════════════════════════════════════════════════════════════════════════


class TestSecondOrderFeatures:
    """Testes para features de 2o grau."""

    def test_output_shape_3d(self, em_data_3d):
        """Output 3D shape (n_seq, seq_len, 6)."""
        so = compute_second_order_features(em_data_3d, h1_cols=(1, 2), h2_cols=(3, 4))
        assert so.shape == (5, 100, 6)

    def test_output_shape_2d(self):
        """Output 2D shape (seq_len, 6)."""
        x = np.random.randn(100, 5).astype(np.float32)
        x[:, 2] = np.clip(x[:, 2], 0.1, None)
        x[:, 4] = np.clip(x[:, 4], 0.1, None)
        so = compute_second_order_features(x, h1_cols=(1, 2), h2_cols=(3, 4))
        assert so.shape == (100, 6)

    def test_power_non_negative(self, em_data_3d):
        """Potencia |H|^2 deve ser >= 0."""
        so = compute_second_order_features(em_data_3d, h1_cols=(1, 2), h2_cols=(3, 4))
        assert so[:, :, 0].min() >= 0  # |H1|^2
        assert so[:, :, 1].min() >= 0  # |H2|^2

    def test_ratio_clipped(self, em_data_3d):
        """Razao Re/Im deve estar em [-100, 100]."""
        so = compute_second_order_features(em_data_3d, h1_cols=(1, 2), h2_cols=(3, 4))
        assert so[:, :, 4].min() >= -100.0
        assert so[:, :, 4].max() <= 100.0
        assert so[:, :, 5].min() >= -100.0
        assert so[:, :, 5].max() <= 100.0

    def test_gradient_first_element(self, em_data_3d):
        """Gradiente do 1o elemento: diff com prepend → valor = 0 ou proximo."""
        so = compute_second_order_features(em_data_3d, h1_cols=(1, 2), h2_cols=(3, 4))
        # O primeiro valor do gradiente eh 0 (diff com prepend do proprio valor)
        np.testing.assert_allclose(so[:, 0, 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(so[:, 0, 3], 0.0, atol=1e-6)

    def test_dtype_float32(self, em_data_3d):
        """Output deve ser float32."""
        so = compute_second_order_features(em_data_3d, h1_cols=(1, 2), h2_cols=(3, 4))
        assert so.dtype == np.float32


# ════════════════════════════════════════════════════════════════════════════
# TESTES: Feature View "second_order"
# ════════════════════════════════════════════════════════════════════════════


class TestSecondOrderFeatureView:
    """Testes para Feature View 'second_order'."""

    def test_in_valid_views(self):
        """'second_order' deve estar em VALID_VIEWS."""
        assert "second_order" in VALID_VIEWS

    def test_feature_view_changes_channels(self, em_data_3d):
        """FV second_order: 5 features EM → 1+6=7 (z + 6 SO)."""
        result = apply_feature_view(em_data_3d, "second_order")
        # z_obs (1) + 6 second-order features = 7
        assert result.shape[-1] == 7  # 5 - 4 EM + 6 SO = 7

    def test_feature_view_preserves_z(self, em_data_3d):
        """FV second_order: z_obs deve ser preservado."""
        result = apply_feature_view(em_data_3d, "second_order")
        # z_obs eh a primeira coluna nao-EM
        # Como removemos 4 EM e adicionamos 6 SO no final,
        # z_obs (col 0) deve ficar na posicao 0
        np.testing.assert_allclose(result[:, :, 0], em_data_3d[:, :, 0])


# ════════════════════════════════════════════════════════════════════════════
# TESTES: Config
# ════════════════════════════════════════════════════════════════════════════


class TestConfigStrategiesBC:
    """Testes de validacao de config para estrategias B e C."""

    def test_oversampling_defaults_off(self):
        """Oversampling desativado por padrao."""
        config = PipelineConfig()
        assert config.use_rho_oversampling is False
        assert config.use_rho_curriculum is False
        assert config.use_second_order_features is False

    def test_oversampling_and_curriculum_exclusive(self):
        """Oversampling e curriculum sao mutuamente exclusivos."""
        with pytest.raises(AssertionError, match="mutuamente exclusivos"):
            PipelineConfig(use_rho_oversampling=True, use_rho_curriculum=True)

    def test_oversampling_threshold_positive(self):
        """Threshold deve ser positivo."""
        with pytest.raises(AssertionError, match="rho_oversampling_threshold"):
            PipelineConfig(use_rho_oversampling=True, rho_oversampling_threshold=-1.0)

    def test_oversampling_factor_min_2(self):
        """Factor deve ser >= 2."""
        with pytest.raises(AssertionError, match="rho_oversampling_factor"):
            PipelineConfig(use_rho_oversampling=True, rho_oversampling_factor=1)

    def test_curriculum_rho_max_order(self):
        """rho_max_end > rho_max_start."""
        with pytest.raises(AssertionError, match="rho_max_end"):
            PipelineConfig(
                use_rho_curriculum=True,
                rho_curriculum_rho_max_start=1000.0,
                rho_curriculum_rho_max_end=100.0,
            )

    def test_second_order_mode_valid(self):
        """Modo invalido deve falhar."""
        with pytest.raises(AssertionError, match="second_order_mode"):
            PipelineConfig(
                use_second_order_features=True,
                second_order_mode="invalid",
            )

    def test_second_order_requires_identity_fv(self):
        """Second-order requer feature_view=identity ou raw."""
        with pytest.raises(AssertionError, match="feature_view='identity'"):
            PipelineConfig(
                use_second_order_features=True,
                second_order_mode="feature_view",
                feature_view="H1_logH2",
            )

    def test_second_order_postprocess_rejects_non_identity(self):
        """Postprocess tambem requer identity/raw."""
        with pytest.raises(AssertionError, match="feature_view='identity'"):
            PipelineConfig(
                use_second_order_features=True,
                second_order_mode="postprocess",
                feature_view="logH1_logH2",
            )

    def test_second_order_fv_mode_with_identity_ok(self):
        """mode=feature_view com identity deve funcionar."""
        config = PipelineConfig(
            use_second_order_features=True,
            second_order_mode="feature_view",
            feature_view="identity",
        )
        assert config.second_order_mode == "feature_view"

    def test_n_features_with_postprocess(self):
        """n_features deve incluir +6 para postprocess."""
        config = PipelineConfig(
            use_second_order_features=True,
            second_order_mode="postprocess",
        )
        assert config.n_features == 5 + 6  # baseline + 6 SO

    def test_n_features_without_second_order(self):
        """n_features sem second_order deve ser 5."""
        config = PipelineConfig()
        assert config.n_features == 5

    def test_n_second_order_channels_postprocess(self):
        """n_second_order_channels = 6 para postprocess."""
        config = PipelineConfig(
            use_second_order_features=True,
            second_order_mode="postprocess",
        )
        assert config.n_second_order_channels == 6

    def test_n_second_order_channels_fv(self):
        """n_second_order_channels = 0 para feature_view mode."""
        config = PipelineConfig(
            use_second_order_features=True,
            second_order_mode="feature_view",
        )
        assert config.n_second_order_channels == 0

    @pytest.mark.parametrize("mode", ["feature_view", "postprocess"])
    def test_valid_modes_accepted(self, mode):
        """Modos validos devem ser aceitos."""
        config = PipelineConfig(
            use_second_order_features=True,
            second_order_mode=mode,
        )
        assert config.second_order_mode == mode
