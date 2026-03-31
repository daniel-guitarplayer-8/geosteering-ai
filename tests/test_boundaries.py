"""Testes para data/boundaries.py — DTB (Distance to Boundary) P5.

Valida deteccao de fronteiras, computacao de DTB labels,
scaling DTB, targets estendidos, e pipeline completo.

Referencia: docs/physics/perspectivas.md secao P5.
"""

import numpy as np
import pytest

from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.boundaries import (
    apply_dtb_scaling,
    build_extended_targets,
    compute_dtb_for_dataset,
    compute_dtb_labels,
    detect_boundaries,
    inverse_dtb_scaling,
)

# ════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def three_layer_profile():
    """Perfil 3 camadas: rho=10 (0-49), rho=100 (50-99), rho=5 (100-149).

    Fronteiras em indices ~49 e ~99 (transicoes abruptas).
    z_obs linear de 0 a 149 metros.
    """
    rho = np.concatenate(
        [
            np.full(50, 10.0),
            np.full(50, 100.0),
            np.full(50, 5.0),
        ]
    )
    z = np.arange(150, dtype=np.float64)
    return rho, z


@pytest.fixture
def uniform_profile():
    """Perfil uniforme: rho=50 constante, 100 pontos."""
    rho = np.full(100, 50.0)
    z = np.arange(100, dtype=np.float64)
    return rho, z


@pytest.fixture
def dtb_p5_config():
    """PipelineConfig preset DTB P5."""
    return PipelineConfig.dtb_p5()


# ════════════════════════════════════════════════════════════════════════════
# TESTES: detect_boundaries()
# ════════════════════════════════════════════════════════════════════════════


class TestDetectBoundaries:
    """Testes para deteccao de fronteiras geologicas."""

    def test_three_layer_detects_two_boundaries(self, three_layer_profile):
        """Perfil 3 camadas deve detectar 2 fronteiras."""
        rho, z = three_layer_profile
        boundaries = detect_boundaries(rho, z)
        assert len(boundaries) == 2

    def test_boundary_positions_near_transitions(self, three_layer_profile):
        """Fronteiras devem estar proximas aos pontos de transicao."""
        rho, z = three_layer_profile
        boundaries = detect_boundaries(rho, z)
        # Transicoes em indice ~49 e ~99
        assert abs(boundaries[0] - 49) <= 2
        assert abs(boundaries[1] - 99) <= 2

    def test_uniform_profile_no_boundaries(self, uniform_profile):
        """Perfil uniforme nao deve ter fronteiras."""
        rho, z = uniform_profile
        boundaries = detect_boundaries(rho, z)
        assert len(boundaries) == 0

    def test_returns_sorted_indices(self, three_layer_profile):
        """Indices devem estar em ordem crescente."""
        rho, z = three_layer_profile
        boundaries = detect_boundaries(rho, z)
        assert np.all(np.diff(boundaries) > 0)

    def test_min_separation_respected(self):
        """Fronteiras consecutivas devem respeitar min_separation."""
        # Perfil com transicoes muito proximas
        rho = np.concatenate(
            [
                np.full(10, 10.0),
                np.full(5, 100.0),  # camada fina
                np.full(10, 10.0),
            ]
        )
        z = np.arange(25, dtype=np.float64)
        boundaries = detect_boundaries(rho, z, min_separation=3)
        if len(boundaries) > 1:
            for i in range(1, len(boundaries)):
                assert boundaries[i] - boundaries[i - 1] >= 3

    def test_threshold_sensitivity(self, three_layer_profile):
        """Threshold alto deve detectar menos fronteiras."""
        rho, z = three_layer_profile
        # Threshold muito alto: nenhuma deteccao
        boundaries_high = detect_boundaries(rho, z, threshold=5.0)
        # Threshold normal: 2 deteccoes
        boundaries_normal = detect_boundaries(rho, z, threshold=0.3)
        assert len(boundaries_high) <= len(boundaries_normal)

    def test_shape_mismatch_raises(self):
        """Shapes incompativeis devem gerar ValueError."""
        rho = np.ones(100)
        z = np.ones(50)
        with pytest.raises(ValueError, match="shape"):
            detect_boundaries(rho, z)

    def test_non_1d_raises(self):
        """Array 2D deve gerar ValueError."""
        rho = np.ones((10, 2))
        z = np.ones((10, 2))
        with pytest.raises(ValueError, match="1D"):
            detect_boundaries(rho, z)

    def test_single_point_returns_empty(self):
        """Array com 1 ponto retorna vazio."""
        rho = np.array([10.0])
        z = np.array([0.0])
        boundaries = detect_boundaries(rho, z)
        assert len(boundaries) == 0

    def test_returns_int64_dtype(self, three_layer_profile):
        """Indices devem ser int64."""
        rho, z = three_layer_profile
        boundaries = detect_boundaries(rho, z)
        assert boundaries.dtype == np.int64


# ════════════════════════════════════════════════════════════════════════════
# TESTES: compute_dtb_labels()
# ════════════════════════════════════════════════════════════════════════════


class TestComputeDTBLabels:
    """Testes para computacao de DTB labels."""

    def test_no_boundaries_all_max(self):
        """Sem fronteiras, todos DTB = dtb_max."""
        z = np.arange(100, dtype=np.float64)
        dtb_up, dtb_down = compute_dtb_labels(z, np.array([]), dtb_max=3.0)
        np.testing.assert_array_equal(dtb_up, 3.0)
        np.testing.assert_array_equal(dtb_down, 3.0)

    def test_boundary_point_has_zero_dtb(self):
        """Ponto na fronteira deve ter DTB = 0."""
        z = np.arange(100, dtype=np.float64)
        boundaries = np.array([50])
        dtb_up, dtb_down = compute_dtb_labels(z, boundaries, dtb_max=5.0)
        # Ponto 50 esta NA fronteira → DTB_down = 0.0
        assert dtb_down[50] == pytest.approx(0.0, abs=1e-6)

    def test_dtb_increases_with_distance(self):
        """DTB deve aumentar conforme distancia da fronteira."""
        z = np.arange(100, dtype=np.float64)
        boundaries = np.array([50])
        dtb_up, dtb_down = compute_dtb_labels(z, boundaries, dtb_max=10.0)
        # Pontos antes da fronteira: DTB_down decresce conforme se aproxima
        assert dtb_down[45] > dtb_down[48]

    def test_dtb_clipped_at_max(self):
        """DTB nao deve exceder dtb_max."""
        z = np.arange(100, dtype=np.float64)
        boundaries = np.array([50])
        dtb_max = 3.0
        dtb_up, dtb_down = compute_dtb_labels(z, boundaries, dtb_max=dtb_max)
        assert dtb_up.max() <= dtb_max + 1e-6
        assert dtb_down.max() <= dtb_max + 1e-6

    def test_dtb_range_valid(self):
        """DTB deve estar em [0, dtb_max]."""
        z = np.arange(100, dtype=np.float64)
        boundaries = np.array([30, 70])
        dtb_max = 5.0
        dtb_up, dtb_down = compute_dtb_labels(z, boundaries, dtb_max=dtb_max)
        assert dtb_up.min() >= 0.0
        assert dtb_down.min() >= 0.0
        assert dtb_up.max() <= dtb_max + 1e-6
        assert dtb_down.max() <= dtb_max + 1e-6

    def test_two_boundaries(self):
        """Dois boundaries: DTB_up e DTB_down corretos entre eles."""
        z = np.arange(100, dtype=np.float64)
        boundaries = np.array([30, 70])
        dtb_max = 50.0  # Grande o suficiente para nao clippar
        dtb_up, dtb_down = compute_dtb_labels(z, boundaries, dtb_max=dtb_max)
        # Ponto 50: equidistante de ambas fronteiras
        assert dtb_up[50] == pytest.approx(20.0, abs=1.0)
        assert dtb_down[50] == pytest.approx(20.0, abs=1.0)

    def test_output_shape_matches_input(self):
        """Output shape deve igualar input shape."""
        z = np.arange(200, dtype=np.float64)
        boundaries = np.array([50, 100, 150])
        dtb_up, dtb_down = compute_dtb_labels(z, boundaries)
        assert dtb_up.shape == (200,)
        assert dtb_down.shape == (200,)

    def test_output_dtype_float32(self):
        """Output deve ser float32."""
        z = np.arange(50, dtype=np.float64)
        dtb_up, dtb_down = compute_dtb_labels(z, np.array([25]))
        assert dtb_up.dtype == np.float32
        assert dtb_down.dtype == np.float32


# ════════════════════════════════════════════════════════════════════════════
# TESTES: apply_dtb_scaling() / inverse_dtb_scaling()
# ════════════════════════════════════════════════════════════════════════════


class TestDTBScaling:
    """Testes para scaling e inverse scaling de DTB."""

    @pytest.mark.parametrize("method", ["linear", "log", "normalized"])
    def test_roundtrip(self, method):
        """Roundtrip: inverse(forward(x)) == x."""
        dtb = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        dtb_max = 3.0
        scaled = apply_dtb_scaling(dtb, method, dtb_max)
        restored = inverse_dtb_scaling(scaled, method, dtb_max)
        np.testing.assert_allclose(restored, dtb, atol=1e-6)

    def test_linear_is_identity(self):
        """Linear scaling: output == input."""
        dtb = np.array([0.0, 1.0, 2.0, 3.0])
        scaled = apply_dtb_scaling(dtb, "linear")
        np.testing.assert_array_equal(scaled, dtb)

    def test_normalized_range_01(self):
        """Normalized: output em [0, 1]."""
        dtb = np.array([0.0, 1.5, 3.0])
        scaled = apply_dtb_scaling(dtb, "normalized", dtb_max=3.0)
        np.testing.assert_allclose(scaled, [0.0, 0.5, 1.0])

    def test_log_zero_is_zero(self):
        """Log scaling: log1p(0) = 0."""
        dtb = np.array([0.0])
        scaled = apply_dtb_scaling(dtb, "log")
        assert scaled[0] == pytest.approx(0.0)

    def test_log_positive(self):
        """Log scaling: log1p(x) > 0 para x > 0."""
        dtb = np.array([1.0, 2.0, 3.0])
        scaled = apply_dtb_scaling(dtb, "log")
        assert np.all(scaled > 0)

    def test_invalid_method_raises(self):
        """Metodo invalido deve gerar ValueError."""
        dtb = np.array([1.0])
        with pytest.raises(ValueError, match="invalido"):
            apply_dtb_scaling(dtb, "invalid_method")

    def test_inverse_invalid_raises(self):
        """Metodo invalido na inversa deve gerar ValueError."""
        dtb = np.array([1.0])
        with pytest.raises(ValueError, match="nao implementada"):
            inverse_dtb_scaling(dtb, "invalid_method")


# ════════════════════════════════════════════════════════════════════════════
# TESTES: build_extended_targets()
# ════════════════════════════════════════════════════════════════════════════


class TestBuildExtendedTargets:
    """Testes para construcao de targets 6-canais."""

    def test_output_shape(self):
        """Output shape deve ser (seq_len, 6)."""
        n = 100
        y = build_extended_targets(
            np.ones(n),
            np.ones(n) * 2,
            np.ones(n) * 0.5,
            np.ones(n) * 1.0,
            np.ones(n) * 10,
            np.ones(n) * 50,
        )
        assert y.shape == (n, 6)

    def test_channel_order(self):
        """Canais devem estar na ordem correta."""
        n = 10
        rho_h = np.full(n, 1.0)
        rho_v = np.full(n, 2.0)
        dtb_up = np.full(n, 3.0)
        dtb_down = np.full(n, 4.0)
        rho_up = np.full(n, 5.0)
        rho_down = np.full(n, 6.0)
        y = build_extended_targets(rho_h, rho_v, dtb_up, dtb_down, rho_up, rho_down)
        np.testing.assert_array_equal(y[:, 0], 1.0)  # rho_h
        np.testing.assert_array_equal(y[:, 1], 2.0)  # rho_v
        np.testing.assert_array_equal(y[:, 2], 3.0)  # DTB_up
        np.testing.assert_array_equal(y[:, 3], 4.0)  # DTB_down
        np.testing.assert_array_equal(y[:, 4], 5.0)  # rho_up
        np.testing.assert_array_equal(y[:, 5], 6.0)  # rho_down

    def test_output_dtype_float32(self):
        """Output deve ser float32."""
        n = 10
        y = build_extended_targets(
            np.ones(n, dtype=np.float64),
            np.ones(n, dtype=np.float64),
            np.ones(n),
            np.ones(n),
            np.ones(n),
            np.ones(n),
        )
        assert y.dtype == np.float32


# ════════════════════════════════════════════════════════════════════════════
# TESTES: compute_dtb_for_dataset()
# ════════════════════════════════════════════════════════════════════════════


class TestComputeDTBForDataset:
    """Testes para pipeline completo DTB."""

    def test_output_shape_6_channels(self, dtb_p5_config):
        """Output deve ter 6 canais quando DTB ativo."""
        n_seq, seq_len = 5, 100
        # 3 camadas por sequencia
        rho_h = np.concatenate([np.full(33, 10.0), np.full(34, 100.0), np.full(33, 5.0)])
        rho_v = rho_h * 1.5
        y = np.stack([rho_h, rho_v], axis=-1)  # (100, 2)
        y_3d = np.tile(y[np.newaxis, :, :], (n_seq, 1, 1))  # (5, 100, 2)
        z = np.tile(np.arange(seq_len, dtype=np.float64)[np.newaxis, :], (n_seq, 1))

        y_ext = compute_dtb_for_dataset(y_3d, z, dtb_p5_config)
        assert y_ext.shape == (n_seq, seq_len, 6)

    def test_rho_channels_preserved(self, dtb_p5_config):
        """Canais rho_h e rho_v devem ser preservados."""
        n_seq, seq_len = 2, 50
        rho_h = np.full(seq_len, 10.0)
        rho_v = np.full(seq_len, 20.0)
        y = np.stack([rho_h, rho_v], axis=-1)
        y_3d = np.tile(y[np.newaxis, :, :], (n_seq, 1, 1))
        z = np.tile(np.arange(seq_len, dtype=np.float64)[np.newaxis, :], (n_seq, 1))

        y_ext = compute_dtb_for_dataset(y_3d, z, dtb_p5_config)
        np.testing.assert_allclose(y_ext[0, :, 0], 10.0)  # rho_h
        np.testing.assert_allclose(y_ext[0, :, 1], 20.0)  # rho_v

    def test_dtb_channels_range(self, dtb_p5_config):
        """DTB channels devem estar em range valido."""
        n_seq, seq_len = 3, 100
        rho_h = np.concatenate([np.full(50, 10.0), np.full(50, 100.0)])
        rho_v = rho_h * 1.5
        y = np.stack([rho_h, rho_v], axis=-1)
        y_3d = np.tile(y[np.newaxis, :, :], (n_seq, 1, 1))
        z = np.tile(np.arange(seq_len, dtype=np.float64)[np.newaxis, :], (n_seq, 1))

        y_ext = compute_dtb_for_dataset(y_3d, z, dtb_p5_config)
        # DTB channels (2, 3) devem ser >= 0
        assert y_ext[:, :, 2].min() >= 0.0
        assert y_ext[:, :, 3].min() >= 0.0

    def test_wrong_input_shape_raises(self, dtb_p5_config):
        """Input shape errada deve gerar ValueError."""
        y = np.ones((5, 100, 3))  # 3 canais, esperado 2
        z = np.ones((5, 100))
        with pytest.raises(ValueError, match="shape"):
            compute_dtb_for_dataset(y, z, dtb_p5_config)

    def test_output_dtype_float32(self, dtb_p5_config):
        """Output deve ser float32."""
        n_seq, seq_len = 2, 50
        y = np.ones((n_seq, seq_len, 2))
        z = np.tile(np.arange(seq_len, dtype=np.float64)[np.newaxis, :], (n_seq, 1))
        y_ext = compute_dtb_for_dataset(y, z, dtb_p5_config)
        assert y_ext.dtype == np.float32


# ════════════════════════════════════════════════════════════════════════════
# TESTES: PipelineConfig DTB
# ════════════════════════════════════════════════════════════════════════════


class TestConfigDTB:
    """Testes para campos DTB no PipelineConfig."""

    def test_dtb_p5_preset(self):
        """Preset dtb_p5 deve ter configuracao correta."""
        config = PipelineConfig.dtb_p5()
        assert config.output_channels == 6
        assert config.use_dtb_as_target is True
        assert config.use_dtb_loss is True
        assert config.dtb_max_from_picasso == 3.0
        assert config.dtb_scaling == "linear"

    def test_dtb_p5_custom_scaling(self):
        """Preset dtb_p5 aceita dtb_scaling customizado."""
        config = PipelineConfig.dtb_p5(dtb_scaling="normalized")
        assert config.dtb_scaling == "normalized"

    def test_dtb_scaling_validation(self):
        """dtb_scaling invalido deve gerar AssertionError."""
        with pytest.raises(AssertionError, match="dtb_scaling"):
            PipelineConfig(dtb_scaling="invalid")

    def test_dtb_max_must_be_positive(self):
        """dtb_max_from_picasso deve ser > 0."""
        with pytest.raises(AssertionError, match="dtb_max_from_picasso"):
            PipelineConfig(dtb_max_from_picasso=-1.0)

    def test_dtb_target_requires_output_channels(self):
        """use_dtb_as_target=True com output_channels=2 deve falhar."""
        with pytest.raises(AssertionError, match="output_channels"):
            PipelineConfig(use_dtb_as_target=True, output_channels=2)

    def test_dtb_loss_requires_dtb_as_target(self):
        """use_dtb_loss=True sem use_dtb_as_target deve falhar."""
        with pytest.raises(AssertionError, match="use_dtb_as_target"):
            PipelineConfig(use_dtb_loss=True, use_dtb_as_target=False)

    def test_dtb_loss_without_target_even_with_6ch(self):
        """use_dtb_loss=True, use_dtb_as_target=False, output_channels=6 deve falhar."""
        with pytest.raises(AssertionError, match="use_dtb_as_target"):
            PipelineConfig(use_dtb_loss=True, use_dtb_as_target=False, output_channels=6)

    def test_dtb_p5_with_kwargs(self):
        """Preset dtb_p5 deve aceitar kwargs adicionais."""
        config = PipelineConfig.dtb_p5(learning_rate=5e-5)
        assert config.learning_rate == 5e-5
        assert config.output_channels == 6

    @pytest.mark.parametrize("scaling", ["linear", "log", "normalized"])
    def test_valid_dtb_scaling_accepted(self, scaling):
        """Todos os metodos validos devem ser aceitos."""
        config = PipelineConfig(dtb_scaling=scaling)
        assert config.dtb_scaling == scaling

    def test_default_dtb_fields(self):
        """Defaults devem ser conservadores (DTB off)."""
        config = PipelineConfig()
        assert config.use_dtb_as_target is False
        assert config.use_dtb_loss is False
        assert config.dtb_max_from_picasso == 3.0
        assert config.dtb_scaling == "linear"
