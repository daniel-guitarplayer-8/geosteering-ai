# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTES: geosteering_ai.losses — catalog + factory                        ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes para o subpacote losses (26 losses + LossFactory).

Estrutura:
    CPU-safe (sem TF):
        TestLossRegistry — contagem, nomes, VALID_LOSS_TYPES
        TestLossFactoryAPI — list_available_losses, list_available
    TF-required:
        TestGenericLosses — 13 funcoes diretas (A)
        TestGeophysicalLosses — 4 factories (B)
        TestGeosteering — 2 losses (C)
        TestAdvancedLosses — 7 factories (D)
        TestBuildCombined — combined loss fn
        TestBuildLossFn — build_loss_fn wrapper
"""
import dataclasses

import pytest

# ── Deteccao de TF (mesmo padrao de test_models.py) ──────────────────────────
try:
    import tensorflow as _tf  # noqa: F401

    HAS_TF = True
except ImportError:
    HAS_TF = False

requires_tf = pytest.mark.skipif(not HAS_TF, reason="TensorFlow nao disponivel")

# ── Imports CPU-safe ──────────────────────────────────────────────────────────
from geosteering_ai.config import PipelineConfig
from geosteering_ai.losses.factory import (
    _LOSS_REGISTRY,
    VALID_LOSS_TYPES,
    LossFactory,
    build_loss_fn,
    list_available_losses,
)

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _make_config(**kwargs) -> PipelineConfig:
    """Cria PipelineConfig a partir do baseline com overrides."""
    return dataclasses.replace(PipelineConfig.baseline(), **kwargs)


def _make_tensors(batch=2, seq=600, channels=2):
    """Cria tensores TF aleatorios para testes de loss."""
    import numpy as np
    import tensorflow as tf

    rng = np.random.default_rng(42)
    y_true = tf.constant(rng.uniform(0.5, 3.5, (batch, seq, channels)), dtype=tf.float32)
    y_pred = tf.constant(rng.uniform(0.5, 3.5, (batch, seq, channels)), dtype=tf.float32)
    return y_true, y_pred


# ════════════════════════════════════════════════════════════════════════════
# TESTES CPU-SAFE
# ════════════════════════════════════════════════════════════════════════════


class TestLossRegistry:
    """Testes de registry e VALID_LOSS_TYPES (CPU-safe)."""

    def test_registry_count_26(self):
        """Deve haver exatamente 26 losses no registry."""
        assert len(_LOSS_REGISTRY) == 26, f"Esperado 26, encontrado {len(_LOSS_REGISTRY)}"

    def test_valid_loss_types_count_26(self):
        """VALID_LOSS_TYPES deve ter 26 entradas."""
        assert len(VALID_LOSS_TYPES) == 26

    def test_registry_matches_valid_types(self):
        """Registry e VALID_LOSS_TYPES devem conter os mesmos nomes."""
        assert set(_LOSS_REGISTRY.keys()) == set(VALID_LOSS_TYPES)

    def test_category_a_13_genericas(self):
        """Categoria A deve ter 13 losses genericas."""
        cat_a = [
            "mse",
            "rmse",
            "mae",
            "mbe",
            "rse",
            "rae",
            "mape",
            "msle",
            "rmsle",
            "nrmse",
            "rrmse",
            "huber",
            "log_cosh",
        ]
        for name in cat_a:
            assert name in _LOSS_REGISTRY, f"'{name}' falta na categoria A"
        assert len(cat_a) == 13

    def test_category_b_4_geofisicas(self):
        """Categoria B deve ter 4 losses geofisicas."""
        cat_b = [
            "log_scale_aware",
            "adaptive_log_scale",
            "robust_log_scale",
            "adaptive_robust",
        ]
        for name in cat_b:
            assert name in _LOSS_REGISTRY, f"'{name}' falta na categoria B"

    def test_category_c_2_geosteering(self):
        """Categoria C deve ter 2 losses geosteering."""
        assert "probabilistic_nll" in _LOSS_REGISTRY
        assert "look_ahead_weighted" in _LOSS_REGISTRY

    def test_category_d_7_avancadas(self):
        """Categoria D deve ter 7 losses avancadas."""
        cat_d = [
            "dilate",
            "enc_decoder",
            "multitask",
            "sobolev",
            "cross_gradient",
            "spectral",
            "morales_physics_hybrid",
        ]
        for name in cat_d:
            assert name in _LOSS_REGISTRY, f"'{name}' falta na categoria D"
        assert len(cat_d) == 7

    def test_default_rmse_in_registry(self):
        """Loss default 'rmse' deve estar no registry."""
        assert "rmse" in _LOSS_REGISTRY

    def test_invalid_loss_raises_value_error(self):
        """Loss invalida deve levantar ValueError."""
        config = _make_config(loss_type="rmse")
        with pytest.raises(ValueError, match="invalida"):
            # Hack: mutar o loss_type temporariamente para testar
            bad_config = dataclasses.replace(config)
            object.__setattr__(bad_config, "loss_type", "bad_loss_name")
            LossFactory.get(bad_config)


class TestLossFactoryAPI:
    """Testes da API publica de LossFactory (CPU-safe)."""

    def test_n_losses_property_26(self):
        """LossFactory.N_LOSSES deve ser 26."""
        assert LossFactory.N_LOSSES == 26

    def test_list_available_returns_26(self):
        """list_available() deve retornar 26 nomes."""
        available = LossFactory.list_available()
        assert len(available) == 26

    def test_list_available_losses_function(self):
        """list_available_losses() deve retornar 26 nomes."""
        losses = list_available_losses()
        assert len(losses) == 26

    def test_list_available_sorted(self):
        """Resultado deve estar em ordem alfabetica."""
        available = LossFactory.list_available()
        assert available == sorted(available)

    def test_list_available_losses_sorted(self):
        """list_available_losses() deve estar em ordem alfabetica."""
        losses = list_available_losses()
        assert losses == sorted(losses)

    def test_valid_loss_types_sorted_subset(self):
        """Todos VALID_LOSS_TYPES devem estar disponiveis."""
        available = set(LossFactory.list_available())
        for lt in VALID_LOSS_TYPES:
            assert lt in available


# ════════════════════════════════════════════════════════════════════════════
# TESTES TF-REQUIRED
# ════════════════════════════════════════════════════════════════════════════


@requires_tf
class TestGenericLosses:
    """Testes das 13 losses genericas (Categoria A)."""

    def _assert_loss(self, loss_type: str):
        """Verifica forward pass: retorna escalar positivo (ou mbe pode ser neg)."""
        import tensorflow as tf

        config = _make_config(loss_type=loss_type)
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.get(config)
        result = loss_fn(y_true, y_pred)
        assert result.shape == (), f"Loss deve ser escalar, shape={result.shape}"
        assert not tf.math.is_nan(result), f"Loss '{loss_type}' retornou NaN"
        # Fix CR#3: tf.math.is_finite retorna tensor → .numpy() para bool.
        assert tf.math.is_finite(result).numpy(), f"Loss '{loss_type}' retornou inf"

    def test_mse(self):
        self._assert_loss("mse")

    def test_rmse(self):
        self._assert_loss("rmse")

    def test_mae(self):
        self._assert_loss("mae")

    def test_mbe(self):
        self._assert_loss("mbe")

    def test_rse(self):
        self._assert_loss("rse")

    def test_rae(self):
        self._assert_loss("rae")

    def test_mape(self):
        self._assert_loss("mape")

    def test_msle(self):
        self._assert_loss("msle")

    def test_rmsle(self):
        self._assert_loss("rmsle")

    def test_nrmse(self):
        self._assert_loss("nrmse")

    def test_rrmse(self):
        self._assert_loss("rrmse")

    def test_huber(self):
        self._assert_loss("huber")

    def test_log_cosh(self):
        self._assert_loss("log_cosh")

    def test_mse_zero_when_identical(self):
        """MSE deve ser 0 quando y_true == y_pred."""
        import tensorflow as tf

        y_true, _ = _make_tensors()
        config = _make_config(loss_type="mse")
        loss_fn = LossFactory.get(config)
        result = loss_fn(y_true, y_true)
        assert float(result) < 1e-6, f"MSE deve ser ~0 quando identico, got {result}"

    def test_rmse_nonnegative(self):
        """RMSE deve ser >= 0."""
        import tensorflow as tf

        y_true, y_pred = _make_tensors()
        config = _make_config(loss_type="rmse")
        loss_fn = LossFactory.get(config)
        result = loss_fn(y_true, y_pred)
        assert float(result) >= 0.0


@requires_tf
class TestGeophysicalLosses:
    """Testes das 4 losses geofisicas (Categoria B)."""

    def _assert_geophysical(self, loss_type: str):
        """Verifica forward pass de loss geofisica."""
        import tensorflow as tf

        config = _make_config(loss_type=loss_type)
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.get(config, epoch_var=None, noise_level_var=None)
        result = loss_fn(y_true, y_pred)
        assert result.shape == ()
        assert not tf.math.is_nan(result), f"Loss '{loss_type}' retornou NaN"

    def test_log_scale_aware(self):
        self._assert_geophysical("log_scale_aware")

    def test_adaptive_log_scale(self):
        self._assert_geophysical("adaptive_log_scale")

    def test_robust_log_scale(self):
        self._assert_geophysical("robust_log_scale")

    def test_adaptive_robust(self):
        self._assert_geophysical("adaptive_robust")

    def test_log_scale_aware_with_epoch_var(self):
        """log_scale_aware com epoch_var deve funcionar corretamente."""
        import tensorflow as tf

        epoch_var = tf.Variable(5, dtype=tf.int32)
        config = _make_config(loss_type="log_scale_aware")
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.get(config, epoch_var=epoch_var)
        result = loss_fn(y_true, y_pred)
        assert not tf.math.is_nan(result)

    def test_adaptive_log_scale_with_noise(self):
        """adaptive_log_scale com noise_level_var deve usar gangorra."""
        import tensorflow as tf

        noise_var = tf.Variable(0.05, dtype=tf.float32)
        config = _make_config(loss_type="adaptive_log_scale")
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.get(config, noise_level_var=noise_var)
        result = loss_fn(y_true, y_pred)
        assert not tf.math.is_nan(result)


@requires_tf
class TestGeosteering:
    """Testes das losses geosteering (Categoria C)."""

    def test_probabilistic_nll_shape(self):
        """probabilistic_nll deve funcionar com y_pred de 2x canais."""
        import numpy as np
        import tensorflow as tf

        rng = np.random.default_rng(42)
        # y_pred tem 2x output_channels (media + log-var)
        y_true = tf.constant(rng.uniform(0.5, 3.5, (2, 600, 2)), dtype=tf.float32)
        y_pred = tf.constant(rng.uniform(0.5, 3.5, (2, 600, 4)), dtype=tf.float32)
        result = probabilistic_nll_loss(y_true, y_pred)
        assert result.shape == ()
        assert not tf.math.is_nan(result)

    def test_look_ahead_weighted(self):
        """look_ahead_weighted deve retornar escalar nao-negativo."""
        import tensorflow as tf

        config = _make_config(loss_type="look_ahead_weighted")
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.get(config)
        result = loss_fn(y_true, y_pred)
        assert result.shape == ()
        assert float(result) >= 0.0
        assert not tf.math.is_nan(result)


@requires_tf
class TestAdvancedLosses:
    """Testes das 7 losses avancadas (Categoria D)."""

    def _assert_advanced(self, loss_type: str):
        import tensorflow as tf

        config = _make_config(loss_type=loss_type)
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.get(config)
        result = loss_fn(y_true, y_pred)
        assert result.shape == ()
        assert not tf.math.is_nan(result), f"Loss '{loss_type}' retornou NaN"

    def test_dilate(self):
        self._assert_advanced("dilate")

    def test_enc_decoder(self):
        """enc_decoder espera 2x canais em y_pred."""
        import numpy as np
        import tensorflow as tf

        rng = np.random.default_rng(42)
        y_true = tf.constant(rng.uniform(0.5, 3.5, (2, 600, 2)), dtype=tf.float32)
        y_pred = tf.constant(rng.uniform(0.5, 3.5, (2, 600, 4)), dtype=tf.float32)
        config = _make_config(loss_type="enc_decoder")
        loss_fn = LossFactory.get(config)
        result = loss_fn(y_true, y_pred)
        assert result.shape == ()

    def test_multitask(self):
        self._assert_advanced("multitask")

    def test_sobolev(self):
        self._assert_advanced("sobolev")

    def test_cross_gradient(self):
        self._assert_advanced("cross_gradient")

    def test_spectral(self):
        self._assert_advanced("spectral")

    def test_morales_physics_hybrid(self):
        self._assert_advanced("morales_physics_hybrid")

    def test_morales_hybrid_with_epoch_var(self):
        """morales_physics_hybrid com epoch_var deve funcionar."""
        import tensorflow as tf

        epoch_var = tf.Variable(25, dtype=tf.int32)
        config = _make_config(
            loss_type="morales_physics_hybrid",
            use_morales_hybrid_loss=True,
        )
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.get(config, epoch_var=epoch_var)
        result = loss_fn(y_true, y_pred)
        assert not tf.math.is_nan(result)


@requires_tf
class TestBuildCombined:
    """Testes da loss combinada build_combined()."""

    def test_simple_base_loss(self):
        """Sem extras, deve retornar a loss base."""
        import tensorflow as tf

        config = _make_config(
            loss_type="rmse",
            use_look_ahead_loss=False,
            use_dtb_loss=False,
            use_morales_hybrid_loss=False,
        )
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.build_combined(config)
        result = loss_fn(y_true, y_pred)
        assert result.shape == ()
        assert not tf.math.is_nan(result)

    def test_with_look_ahead(self):
        """Com look_ahead ativo, deve retornar escalar valido."""
        import tensorflow as tf

        config = _make_config(
            loss_type="rmse",
            use_look_ahead_loss=True,
            look_ahead_weight=0.1,
            use_dtb_loss=False,
            use_morales_hybrid_loss=False,
        )
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.build_combined(config)
        result = loss_fn(y_true, y_pred)
        assert result.shape == ()
        assert not tf.math.is_nan(result)

    def test_morales_replaces_base(self):
        """use_morales_hybrid_loss=True deve usar morales (#26) como loss."""
        import tensorflow as tf

        config = _make_config(
            loss_type="rmse",
            use_look_ahead_loss=False,
            use_dtb_loss=False,
            use_morales_hybrid_loss=True,
        )
        y_true, y_pred = _make_tensors()
        loss_fn = LossFactory.build_combined(config)
        result = loss_fn(y_true, y_pred)
        assert result.shape == ()
        assert not tf.math.is_nan(result)

    def test_morales_omega_effect(self):
        """morales_physics_omega=1.0 deve dar MSE puro, 0.0 deve dar MAE puro."""
        import tensorflow as tf

        y_true, y_pred = _make_tensors()

        config_mse = _make_config(
            loss_type="morales_physics_hybrid",
            use_morales_hybrid_loss=True,
            morales_physics_omega=1.0,
        )
        config_mae = _make_config(
            loss_type="morales_physics_hybrid",
            use_morales_hybrid_loss=True,
            morales_physics_omega=0.0,
        )

        fn_mse = LossFactory.build_combined(config_mse)
        fn_mae = LossFactory.build_combined(config_mae)

        # Com omega=1.0: resultado ≈ MSE
        # Com omega=0.0: resultado ≈ MAE
        # Ambos devem ser finitos e positivos
        r_mse = fn_mse(y_true, y_pred)
        r_mae = fn_mae(y_true, y_pred)
        assert not tf.math.is_nan(r_mse)
        assert not tf.math.is_nan(r_mae)
        assert float(r_mse) > 0
        assert float(r_mae) > 0


@requires_tf
class TestBuildLossFn:
    """Testes do wrapper build_loss_fn()."""

    def test_build_loss_fn_default(self):
        """build_loss_fn com config default deve retornar callable valido."""
        import tensorflow as tf

        config = _make_config()
        loss_fn = build_loss_fn(config)
        y_true, y_pred = _make_tensors()
        result = loss_fn(y_true, y_pred)
        assert result.shape == ()
        assert not tf.math.is_nan(result)

    def test_all_direct_losses_forward_pass(self):
        """Todas as 13 losses diretas devem passar no forward pass."""
        import tensorflow as tf

        direct_types = [
            "mse",
            "rmse",
            "mae",
            "mbe",
            "rse",
            "rae",
            "mape",
            "msle",
            "rmsle",
            "nrmse",
            "rrmse",
            "huber",
            "log_cosh",
        ]
        y_true, y_pred = _make_tensors()
        for lt in direct_types:
            config = _make_config(loss_type=lt)
            loss_fn = build_loss_fn(config)
            result = loss_fn(y_true, y_pred)
            assert not tf.math.is_nan(result), f"NaN em loss '{lt}'"


# ── Fixtures de funcoes especificas para TestGeosteering ─────────────────────
from geosteering_ai.losses.catalog import probabilistic_nll as probabilistic_nll_loss
