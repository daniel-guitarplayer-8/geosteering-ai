"""Testes para geosteering_ai.noise — Bloco 2c.

Cobre os 2 submodulos de noise/:
    - functions: NOISE_FN_MAP, apply_noise_tf, apply_raw_em_noise,
                 create_noise_level_var
    - curriculum: CurriculumSchedule, compute_noise_level,
                  UpdateNoiseLevelCallback
"""

import numpy as np
import pytest

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

requires_tf = pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")


# ═══════════════════════════════════════════════════════════════════════════
# NOISE FUNCTIONS — NUMPY (OFFLINE)
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyRawEmNoise:
    """Testa apply_raw_em_noise (versao numpy)."""

    def _make_data(self, n_seq=5, seq_len=600, n_feat=5):
        rng = np.random.RandomState(42)
        return rng.randn(n_seq, seq_len, n_feat).astype(np.float32)

    def test_z_preserved(self):
        """z_obs (col 0) NUNCA recebe noise."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        x_noisy = apply_raw_em_noise(x, noise_level=0.1, seed=0)
        np.testing.assert_array_equal(x_noisy[:, :, 0], x[:, :, 0])

    def test_em_channels_modified(self):
        """Colunas EM (1:) recebem noise."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        x_noisy = apply_raw_em_noise(x, noise_level=0.1, seed=0)
        assert not np.allclose(x_noisy[:, :, 1:], x[:, :, 1:])

    def test_shape_preserved(self):
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        x_noisy = apply_raw_em_noise(x, noise_level=0.05, seed=0)
        assert x_noisy.shape == x.shape

    def test_original_unchanged(self):
        """apply_raw_em_noise retorna copia, original intacto."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        x_copy = x.copy()
        _ = apply_raw_em_noise(x, noise_level=0.1, seed=0)
        np.testing.assert_array_equal(x, x_copy)

    def test_reproducible_with_seed(self):
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        a = apply_raw_em_noise(x, noise_level=0.05, seed=123)
        b = apply_raw_em_noise(x, noise_level=0.05, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_zero_noise_level(self):
        """noise_level=0.0 retorna dados identicos ao original."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        x_noisy = apply_raw_em_noise(x, noise_level=0.0, seed=0)
        np.testing.assert_array_almost_equal(x_noisy, x, decimal=10)

    def test_gaussian_type(self):
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=["gaussian"],
            seed=0,
        )
        assert result.shape == x.shape

    def test_multiplicative_type(self):
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=["multiplicative"],
            seed=0,
        )
        assert result.shape == x.shape

    def test_uniform_type(self):
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=["uniform"],
            seed=0,
        )
        assert result.shape == x.shape

    def test_dropout_type(self):
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=["dropout"],
            seed=0,
        )
        assert result.shape == x.shape

    def test_mixed_types(self):
        """Mix de gaussian + multiplicative com pesos."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=["gaussian", "multiplicative"],
            noise_weights=[0.8, 0.2],
            seed=0,
        )
        assert result.shape == x.shape
        np.testing.assert_array_equal(result[:, :, 0], x[:, :, 0])

    def test_invalid_type_raises(self):
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        with pytest.raises(ValueError, match="desconhecido"):
            apply_raw_em_noise(x, noise_types=["invalid_noise"])

    def test_2d_raises(self):
        from geosteering_ai.noise import apply_raw_em_noise

        x = np.zeros((600, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="3D"):
            apply_raw_em_noise(x)


# ═══════════════════════════════════════════════════════════════════════════
# NOISE FUNCTIONS — TENSORFLOW (ON-THE-FLY)
# ═══════════════════════════════════════════════════════════════════════════


@requires_tf
class TestNoiseFnMap:
    """Testa NOISE_FN_MAP e funcoes TF individuais."""

    def test_original_four_types_registered(self):
        from geosteering_ai.noise import NOISE_FN_MAP

        assert "gaussian" in NOISE_FN_MAP
        assert "multiplicative" in NOISE_FN_MAP
        assert "uniform" in NOISE_FN_MAP
        assert "dropout" in NOISE_FN_MAP

    def test_valid_noise_types_frozenset(self):
        from geosteering_ai.noise import VALID_NOISE_TYPES

        assert isinstance(VALID_NOISE_TYPES, frozenset)
        assert len(VALID_NOISE_TYPES) >= 4


@requires_tf
class TestApplyNoiseTf:
    """Testa apply_noise_tf (dispatcher TF)."""

    def _make_tf_data(self, batch=4, seq_len=100, n_feat=5):
        return tf.random.normal((batch, seq_len, n_feat), seed=42)

    def test_gaussian_preserves_z(self):
        """z_obs (col 0) preservado no noise TF."""
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.1)
        x_noisy = apply_noise_tf(x, nv, ["gaussian"], [1.0])
        np.testing.assert_array_equal(
            x[:, :, :1].numpy(),
            x_noisy[:, :, :1].numpy(),
        )

    def test_shape_preserved(self):
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.05)
        x_noisy = apply_noise_tf(x, nv, ["gaussian"], [1.0])
        assert x_noisy.shape == x.shape

    def test_em_modified(self):
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.1)
        x_noisy = apply_noise_tf(x, nv, ["gaussian"], [1.0])
        assert not np.allclose(
            x[:, :, 1:].numpy(),
            x_noisy[:, :, 1:].numpy(),
        )

    def test_multiplicative_tf(self):
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.05)
        result = apply_noise_tf(x, nv, ["multiplicative"], [1.0])
        assert result.shape == x.shape

    def test_uniform_tf(self):
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.05)
        result = apply_noise_tf(x, nv, ["uniform"], [1.0])
        assert result.shape == x.shape

    def test_dropout_tf(self):
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.1)
        result = apply_noise_tf(x, nv, ["dropout"], [1.0])
        assert result.shape == x.shape

    def test_mixed_tf(self):
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.05)
        result = apply_noise_tf(
            x,
            nv,
            ["gaussian", "multiplicative"],
            [0.7, 0.3],
        )
        assert result.shape == x.shape

    def test_invalid_type_raises(self):
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.05)
        with pytest.raises(ValueError, match="desconhecido"):
            apply_noise_tf(x, nv, ["bogus"], [1.0])


@requires_tf
class TestCreateNoiseLevelVar:
    """Testa create_noise_level_var."""

    def test_creates_tf_variable(self):
        from geosteering_ai.noise import create_noise_level_var

        nv = create_noise_level_var(0.0)
        assert isinstance(nv, tf.Variable)
        assert nv.numpy() == pytest.approx(0.0)

    def test_not_trainable(self):
        from geosteering_ai.noise import create_noise_level_var

        nv = create_noise_level_var(0.05)
        assert nv.trainable is False

    def test_assignable(self):
        from geosteering_ai.noise import create_noise_level_var

        nv = create_noise_level_var(0.0)
        nv.assign(0.08)
        assert nv.numpy() == pytest.approx(0.08)


# ═══════════════════════════════════════════════════════════════════════════
# CURRICULUM SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════


class TestCurriculumSchedule:
    """Testa CurriculumSchedule (3 fases)."""

    def _make_schedule(self):
        from geosteering_ai.noise.curriculum import CurriculumSchedule

        return CurriculumSchedule(
            noise_level_max=0.08,
            epochs_no_noise=10,
            noise_ramp_epochs=80,
        )

    def test_phase_1_clean(self):
        sched = self._make_schedule()
        for ep in range(10):
            assert sched.get_level(ep) == 0.0
            assert sched.get_phase(ep) == "clean"

    def test_phase_2_ramp_start(self):
        sched = self._make_schedule()
        level = sched.get_level(10)
        assert level == pytest.approx(0.0)
        assert sched.get_phase(10) == "ramp"

    def test_phase_2_ramp_midpoint(self):
        sched = self._make_schedule()
        # Epoch 50: (50-10)/80 * 0.08 = 0.04
        level = sched.get_level(50)
        assert level == pytest.approx(0.04)
        assert sched.get_phase(50) == "ramp"

    def test_phase_2_ramp_end(self):
        sched = self._make_schedule()
        # Epoch 89: (89-10)/80 * 0.08 = 0.079
        level = sched.get_level(89)
        assert level == pytest.approx(0.08 * 79 / 80)
        assert sched.get_phase(89) == "ramp"

    def test_phase_3_stable(self):
        sched = self._make_schedule()
        for ep in [90, 100, 200, 500]:
            assert sched.get_level(ep) == pytest.approx(0.08)
            assert sched.get_phase(ep) == "stable"

    def test_end_ramp_epoch(self):
        sched = self._make_schedule()
        assert sched.end_ramp_epoch == 90

    def test_monotonic_increase(self):
        """Noise level eh monotonicamente nao-decrescente."""
        sched = self._make_schedule()
        levels = [sched.get_level(ep) for ep in range(200)]
        for i in range(1, len(levels)):
            assert levels[i] >= levels[i - 1]

    def test_from_config(self):
        from geosteering_ai.config import PipelineConfig
        from geosteering_ai.noise.curriculum import CurriculumSchedule

        config = PipelineConfig.robusto()
        sched = CurriculumSchedule.from_config(config)
        assert sched.noise_level_max == 0.08
        assert sched.epochs_no_noise == 10
        assert sched.noise_ramp_epochs == 80


class TestComputeNoiseLevel:
    """Testa compute_noise_level (funcao pura)."""

    def test_clean_phase(self):
        from geosteering_ai.noise import compute_noise_level

        assert compute_noise_level(0, 0.08, 10, 80) == 0.0

    def test_ramp_midpoint(self):
        from geosteering_ai.noise import compute_noise_level

        level = compute_noise_level(50, 0.08, 10, 80)
        assert level == pytest.approx(0.04)

    def test_stable_phase(self):
        from geosteering_ai.noise import compute_noise_level

        assert compute_noise_level(100, 0.08, 10, 80) == pytest.approx(0.08)


# ═══════════════════════════════════════════════════════════════════════════
# UPDATE NOISE LEVEL CALLBACK
# ═══════════════════════════════════════════════════════════════════════════


@requires_tf
class TestUpdateNoiseLevelCallback:
    """Testa UpdateNoiseLevelCallback."""

    def test_curriculum_updates_var(self):
        """Testa que o callback atualiza noise_level_var seguindo 3 fases."""
        from geosteering_ai.config import PipelineConfig
        from geosteering_ai.noise import create_noise_level_var
        from geosteering_ai.noise.curriculum import UpdateNoiseLevelCallback

        nv = create_noise_level_var(0.0)
        # Fix CR#1: API correta — (noise_level_var, config: PipelineConfig)
        config = PipelineConfig(
            noise_level_max=0.08,
            epochs_no_noise=10,
            noise_ramp_epochs=80,
        )
        cb = UpdateNoiseLevelCallback(nv, config)

        # Fase 1 (clean): epoch 0 — noise = 0.0
        cb.on_epoch_begin(0)
        assert nv.numpy() == pytest.approx(0.0)

        # Fase 2 (ramp midpoint): epoch 50 — noise = 0.04
        cb.on_epoch_begin(50)
        assert nv.numpy() == pytest.approx(0.04)

        # Fase 3 (stable): epoch 100 — noise = 0.08
        cb.on_epoch_begin(100)
        assert nv.numpy() == pytest.approx(0.08)

    def test_no_curriculum_constant(self):
        """Sem curriculum (epochs_no_noise=0, ramp=0), noise constante."""
        from geosteering_ai.config import PipelineConfig
        from geosteering_ai.noise import create_noise_level_var
        from geosteering_ai.noise.curriculum import UpdateNoiseLevelCallback

        nv = create_noise_level_var(0.0)
        # Fix CR#1: sem curriculum — epochs_no_noise=0 e ramp=0 dá noise
        # constante desde epoch 0 (pula direto para Fase 3).
        config = PipelineConfig(
            noise_level_max=0.08,
            epochs_no_noise=0,
            noise_ramp_epochs=0,
        )
        cb = UpdateNoiseLevelCallback(nv, config)

        cb.on_epoch_begin(0)
        assert nv.numpy() == pytest.approx(0.08)

        cb.on_epoch_begin(5)
        assert nv.numpy() == pytest.approx(0.08)


# ═══════════════════════════════════════════════════════════════════════════
# INIT RE-EXPORTS
# ═══════════════════════════════════════════════════════════════════════════


@requires_tf
class TestInitReExports:
    """Verifica que noise/__init__.py re-exporta todos os simbolos."""

    def test_all_exports_accessible(self):
        import geosteering_ai.noise as noise

        assert hasattr(noise, "NOISE_FN_MAP")
        assert hasattr(noise, "VALID_NOISE_TYPES")
        assert hasattr(noise, "create_noise_level_var")
        assert hasattr(noise, "apply_noise_tf")
        assert hasattr(noise, "apply_raw_em_noise")
        assert hasattr(noise, "CurriculumSchedule")
        assert hasattr(noise, "compute_noise_level")
        assert hasattr(noise, "UpdateNoiseLevelCallback")

    def test_all_list_matches_exports(self):
        import geosteering_ai.noise as noise

        for name in noise.__all__:
            assert hasattr(noise, name), f"Missing export: {name}"


# ═══════════════════════════════════════════════════════════════════════════
# FASE II: 11 NOVOS TIPOS DE NOISE (5 CORE + 6 LWD)
# ═══════════════════════════════════════════════════════════════════════════
# Cada tipo de noise DEVE:
#   1. Estar registrado em NOISE_FN_MAP
#   2. Preservar z_obs (col 0) intacto
#   3. Preservar shape do tensor de entrada
#   4. Funcionar via apply_raw_em_noise (numpy)
#   5. Funcionar via apply_noise_tf (TF, se HAS_TF)
#
# Os 11 tipos:
#   CORE:    drift, depth_dependent, spikes, pink, saturation
#   LWD:     shoulder_bed, borehole_effect, mud_invasion,
#            anisotropy_misalignment, formation_heterogeneity, telemetry
# ──────────────────────────────────────────────────────────────────────────

# Lista dos 11 novos tipos para parametrizacao
_NEW_NOISE_TYPES = [
    # ── 5 CORE ──────────────────────────────────────────────────────────
    "drift",
    "depth_dependent",
    "spikes",
    "pink",
    "saturation",
    # ── 6 LWD Geofisicos (R1-R6) ────────────────────────────────────────
    "shoulder_bed",
    "borehole_effect",
    "mud_invasion",
    "anisotropy_misalignment",
    "formation_heterogeneity",
    "telemetry",
]


class TestNewNoiseTypesRegistered:
    """Verifica que os 11 novos tipos estao registrados em NOISE_FN_MAP."""

    def test_noise_fn_map_has_15_types(self):
        """NOISE_FN_MAP deve ter 15 tipos (4 originais + 11 novos)."""
        from geosteering_ai.noise import NOISE_FN_MAP

        assert len(NOISE_FN_MAP) == 15, (
            f"Esperado 15 tipos, encontrado {len(NOISE_FN_MAP)}: "
            f"{sorted(NOISE_FN_MAP.keys())}"
        )

    def test_valid_noise_types_has_15(self):
        """VALID_NOISE_TYPES deve ter 15 tipos."""
        from geosteering_ai.noise import VALID_NOISE_TYPES

        assert len(VALID_NOISE_TYPES) == 15

    @pytest.mark.parametrize("noise_type", _NEW_NOISE_TYPES)
    def test_type_in_noise_fn_map(self, noise_type):
        """Cada novo tipo esta registrado em NOISE_FN_MAP."""
        from geosteering_ai.noise import NOISE_FN_MAP

        assert (
            noise_type in NOISE_FN_MAP
        ), f"'{noise_type}' nao encontrado em NOISE_FN_MAP"

    @pytest.mark.parametrize("noise_type", _NEW_NOISE_TYPES)
    def test_type_is_callable(self, noise_type):
        """Cada novo tipo eh callable."""
        from geosteering_ai.noise import NOISE_FN_MAP

        assert callable(NOISE_FN_MAP[noise_type])


class TestNewNoiseTypesNumpy:
    """Testa os 11 novos tipos via apply_raw_em_noise (numpy)."""

    def _make_data(self, n_seq=5, seq_len=100, n_feat=5):
        rng = np.random.RandomState(42)
        return rng.randn(n_seq, seq_len, n_feat).astype(np.float32)

    @pytest.mark.parametrize("noise_type", _NEW_NOISE_TYPES)
    def test_shape_preserved(self, noise_type):
        """Shape preservado para cada novo tipo."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=[noise_type],
            seed=42,
        )
        assert result.shape == x.shape

    @pytest.mark.parametrize("noise_type", _NEW_NOISE_TYPES)
    def test_z_obs_preserved(self, noise_type):
        """z_obs (col 0) NUNCA recebe noise para cada novo tipo."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=[noise_type],
            seed=42,
        )
        np.testing.assert_array_equal(result[:, :, 0], x[:, :, 0])

    @pytest.mark.parametrize("noise_type", _NEW_NOISE_TYPES)
    def test_original_unchanged(self, noise_type):
        """apply_raw_em_noise retorna copia, original intacto."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        x_copy = x.copy()
        _ = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=[noise_type],
            seed=42,
        )
        np.testing.assert_array_equal(x, x_copy)

    @pytest.mark.parametrize("noise_type", _NEW_NOISE_TYPES)
    def test_reproducible_with_seed(self, noise_type):
        """Resultados reproduziveis com mesma seed."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = self._make_data()
        a = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=[noise_type],
            seed=123,
        )
        b = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=[noise_type],
            seed=123,
        )
        np.testing.assert_array_equal(a, b)


@requires_tf
class TestNewNoiseTypesTf:
    """Testa os 11 novos tipos via apply_noise_tf (TF on-the-fly)."""

    def _make_tf_data(self, batch=4, seq_len=100, n_feat=5):
        return tf.random.normal((batch, seq_len, n_feat), seed=42)

    @pytest.mark.parametrize("noise_type", _NEW_NOISE_TYPES)
    def test_shape_preserved_tf(self, noise_type):
        """Shape preservado na versao TF."""
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.05)
        result = apply_noise_tf(x, nv, [noise_type], [1.0])
        assert result.shape == x.shape

    @pytest.mark.parametrize("noise_type", _NEW_NOISE_TYPES)
    def test_z_obs_preserved_tf(self, noise_type):
        """z_obs (col 0) preservado na versao TF."""
        from geosteering_ai.noise import apply_noise_tf, create_noise_level_var

        x = self._make_tf_data()
        nv = create_noise_level_var(0.05)
        result = apply_noise_tf(x, nv, [noise_type], [1.0])
        np.testing.assert_array_equal(
            x[:, :, :1].numpy(),
            result[:, :, :1].numpy(),
        )


class TestDriftNoisePhysics:
    """Testes especificos de drift: ruido acumulado temporalmente."""

    def test_drift_accumulates(self):
        """Drift deve ter autocorrelacao temporal (acumulado)."""
        from geosteering_ai.noise import apply_raw_em_noise

        rng = np.random.RandomState(42)
        x = np.zeros((1, 200, 5), dtype=np.float32)
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=["drift"],
            seed=0,
        )
        # Drift eh cumsum de noise — valores consecutivos devem ser
        # correlacionados (diferenca < sigma em media)
        em = result[0, :, 1]
        diffs = np.abs(np.diff(em))
        assert np.mean(diffs) < np.std(em), "Drift deve ter autocorrelacao temporal"


class TestSaturationNoisePhysics:
    """Testes especificos de saturation: clipping em percentil."""

    def test_saturation_clips(self):
        """Saturation deve clipar valores extremos."""
        from geosteering_ai.noise import apply_raw_em_noise

        rng = np.random.RandomState(42)
        # Dados com valores extremos
        x = rng.randn(2, 100, 5).astype(np.float32) * 10
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=["saturation"],
            seed=0,
        )
        # Resultado deve ter range menor ou igual ao original nas colunas EM
        for col in range(1, 5):
            assert np.max(np.abs(result[:, :, col])) <= np.max(np.abs(x[:, :, col]))


class TestFormationHeterogeneityPhysics:
    """Testes especificos de formation_heterogeneity (R5) — unico tipo que
    perturba TANTO features quanto targets quando aplicado ao array completo."""

    def test_modifies_em_channels(self):
        """Formation heterogeneity modifica colunas EM."""
        from geosteering_ai.noise import apply_raw_em_noise

        x = np.random.RandomState(42).randn(3, 100, 5).astype(np.float32)
        result = apply_raw_em_noise(
            x,
            noise_level=0.05,
            noise_types=["formation_heterogeneity"],
            seed=0,
        )
        assert not np.allclose(result[:, :, 1:], x[:, :, 1:])
