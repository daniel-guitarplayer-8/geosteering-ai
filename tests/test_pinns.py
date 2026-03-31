"""Testes para losses/pinns.py — PINNs, TIV constraint, lambda schedule.

Cobertura:
    - TestLambdaSchedule: 4 schedules × 3 fases (warmup/ramp/hold)
    - TestOraclePhysicsLoss: 3 normas (l1/l2/huber), DTB compat
    - TestSurrogatePhysicsLoss: placeholder (retorna 0.0)
    - TestMaxwellPhysicsLoss: curvatura, normalizacao por condutividade
    - TestTIVConstraintLoss: valido (rho_v>=rho_h), invalido, misto
    - TestBuildPINNsLoss: integracao com epoch_var e schedule
    - TestTIVConstraintLayer: hard constraint em models/blocks.py
    - TestConfigPINNs: validacao dos campos PINN em PipelineConfig
    - TestFactoryIntegration: build_combined com PINNs ativadas
"""

import math

import numpy as np
import pytest

from geosteering_ai.config import PipelineConfig
from geosteering_ai.losses.pinns import (
    VALID_LAMBDA_SCHEDULES,
    VALID_PINNS_SCENARIOS,
    build_pinns_loss,
    compute_lambda_schedule,
    make_maxwell_physics_loss,
    make_oracle_physics_loss,
    make_surrogate_physics_loss,
    make_tiv_constraint_loss,
)

# ── TF disponivel? ──────────────────────────────────────────────────
try:
    import tensorflow as tf

    _HAS_TF = True
except ImportError:
    _HAS_TF = False

skip_no_tf = pytest.mark.skipif(not _HAS_TF, reason="TensorFlow nao disponivel")

# ── Helpers ──────────────────────────────────────────────────────────
B, N, C = 4, 100, 2  # batch, seq_len, output_channels


def _make_tensors(rho_h_range=(0.5, 2.5), rho_v_offset=0.2):
    """Cria y_true e y_pred sinteticos em log10 scale."""
    rng = np.random.default_rng(42)
    rho_h = rng.uniform(*rho_h_range, size=(B, N, 1)).astype(np.float32)
    rho_v = rho_h + rho_v_offset  # rho_v > rho_h (TIV valido)
    y_true = np.concatenate([rho_h, rho_v], axis=-1)
    # y_pred com pequeno erro
    y_pred = y_true + rng.normal(0, 0.05, size=y_true.shape).astype(np.float32)
    return tf.constant(y_true), tf.constant(y_pred)


def _make_config(**kwargs):
    """Cria PipelineConfig com defaults para PINNs."""
    defaults = dict(
        use_pinns=True,
        pinns_scenario="oracle",
        pinns_lambda=0.01,
        pinns_warmup_epochs=5,
        pinns_ramp_epochs=10,
        pinns_lambda_schedule="linear",
        pinns_physics_norm="l2",
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


# ════════════════════════════════════════════════════════════════════════
# TESTES: LAMBDA SCHEDULE
# ════════════════════════════════════════════════════════════════════════


class TestLambdaSchedule:
    """Testes para compute_lambda_schedule()."""

    def test_warmup_returns_zero(self):
        """Durante warmup, lambda deve ser 0."""
        for s in VALID_LAMBDA_SCHEDULES:
            val = compute_lambda_schedule(
                epoch=3,
                warmup_epochs=10,
                ramp_epochs=20,
                lambda_target=0.01,
                schedule=s,
            )
            assert val == 0.0, f"schedule={s}, epoch=3: expected 0.0, got {val}"

    def test_hold_returns_target(self):
        """Apos warmup+ramp, lambda deve ser target."""
        for s in VALID_LAMBDA_SCHEDULES:
            val = compute_lambda_schedule(
                epoch=50,
                warmup_epochs=10,
                ramp_epochs=20,
                lambda_target=0.05,
                schedule=s,
            )
            assert val == pytest.approx(0.05), f"schedule={s}: {val}"

    def test_linear_midpoint(self):
        """Linear: no meio do ramp, lambda = target/2."""
        val = compute_lambda_schedule(
            epoch=20,
            warmup_epochs=10,
            ramp_epochs=20,
            lambda_target=0.1,
            schedule="linear",
        )
        assert val == pytest.approx(0.05)

    def test_cosine_midpoint(self):
        """Cosine: no meio do ramp, lambda = target/2 (ponto de inflexao)."""
        val = compute_lambda_schedule(
            epoch=20,
            warmup_epochs=10,
            ramp_epochs=20,
            lambda_target=0.1,
            schedule="cosine",
        )
        expected = 0.1 * 0.5 * (1 - math.cos(math.pi * 0.5))
        assert val == pytest.approx(expected)

    def test_step_before_midpoint(self):
        """Step: antes do meio do ramp, lambda = 0."""
        val = compute_lambda_schedule(
            epoch=12,
            warmup_epochs=10,
            ramp_epochs=20,
            lambda_target=0.1,
            schedule="step",
        )
        assert val == 0.0

    def test_step_after_midpoint(self):
        """Step: apos o meio do ramp, lambda = target."""
        val = compute_lambda_schedule(
            epoch=22,
            warmup_epochs=10,
            ramp_epochs=20,
            lambda_target=0.1,
            schedule="step",
        )
        assert val == pytest.approx(0.1)

    def test_fixed_during_ramp(self):
        """Fixed: qualquer epoca apos warmup = target."""
        val = compute_lambda_schedule(
            epoch=11,
            warmup_epochs=10,
            ramp_epochs=20,
            lambda_target=0.1,
            schedule="fixed",
        )
        assert val == pytest.approx(0.1)

    def test_invalid_schedule_raises(self):
        """Schedule invalido deve levantar ValueError."""
        with pytest.raises(ValueError, match="invalido"):
            compute_lambda_schedule(
                epoch=15,
                warmup_epochs=10,
                ramp_epochs=20,
                lambda_target=0.1,
                schedule="exponential",
            )

    def test_zero_ramp_returns_target(self):
        """ramp_epochs=0: lambda = target imediatamente apos warmup."""
        val = compute_lambda_schedule(
            epoch=10,
            warmup_epochs=10,
            ramp_epochs=0,
            lambda_target=0.1,
            schedule="linear",
        )
        assert val == pytest.approx(0.1)

    def test_monotonic_linear(self):
        """Linear: lambda cresce monotonicamente durante ramp."""
        values = [
            compute_lambda_schedule(
                epoch=e,
                warmup_epochs=5,
                ramp_epochs=20,
                lambda_target=0.1,
                schedule="linear",
            )
            for e in range(5, 26)
        ]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


# ════════════════════════════════════════════════════════════════════════
# TESTES: ORACLE PHYSICS LOSS
# ════════════════════════════════════════════════════════════════════════


@skip_no_tf
class TestOraclePhysicsLoss:
    """Testes para make_oracle_physics_loss()."""

    def test_l2_norm_returns_scalar(self):
        """L2 norm: retorna scalar float32 >= 0."""
        config = _make_config(pinns_physics_norm="l2")
        fn = make_oracle_physics_loss(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.shape == ()
        assert loss.dtype == tf.float32
        assert loss.numpy() >= 0

    def test_l1_norm(self):
        """L1 norm: retorna scalar >= 0."""
        config = _make_config(pinns_physics_norm="l1")
        fn = make_oracle_physics_loss(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.numpy() >= 0

    def test_huber_norm(self):
        """Huber norm: retorna scalar >= 0."""
        config = _make_config(pinns_physics_norm="huber")
        fn = make_oracle_physics_loss(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.numpy() >= 0

    def test_perfect_prediction_zero_loss(self):
        """Predicao perfeita → loss = 0."""
        config = _make_config(pinns_physics_norm="l2")
        fn = make_oracle_physics_loss(config)
        y = tf.constant(np.random.randn(B, N, C).astype(np.float32))
        loss = fn(y, y)
        assert loss.numpy() == pytest.approx(0.0, abs=1e-6)

    def test_operates_on_first_two_channels(self):
        """Com DTB (6 canais), opera apenas em [0:2]."""
        config = _make_config(pinns_physics_norm="l2")
        fn = make_oracle_physics_loss(config)
        rng = np.random.default_rng(42)
        y_true = tf.constant(rng.normal(size=(B, N, 6)).astype(np.float32))
        y_pred = tf.constant(rng.normal(size=(B, N, 6)).astype(np.float32))
        loss = fn(y_true, y_pred)
        # Deve operar sem erro com 6 canais
        assert loss.numpy() >= 0


# ════════════════════════════════════════════════════════════════════════
# TESTES: SURROGATE PHYSICS LOSS
# ════════════════════════════════════════════════════════════════════════


@skip_no_tf
class TestSurrogatePhysicsLoss:
    """Testes para make_surrogate_physics_loss() — placeholder."""

    def test_returns_zero_without_model(self):
        """Sem surrogate model, retorna 0.0."""
        config = _make_config(
            pinns_scenario="surrogate",
            pinns_use_forward_surrogate=False,
        )
        fn = make_surrogate_physics_loss(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.numpy() == pytest.approx(0.0)

    def test_returns_zero_with_empty_path(self):
        """Com path vazio, retorna 0.0."""
        config = _make_config(
            pinns_scenario="surrogate",
            pinns_use_forward_surrogate=True,
            surrogate_model_path="",
        )
        fn = make_surrogate_physics_loss(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.numpy() == pytest.approx(0.0)


# ════════════════════════════════════════════════════════════════════════
# TESTES: MAXWELL PHYSICS LOSS
# ════════════════════════════════════════════════════════════════════════


@skip_no_tf
class TestMaxwellPhysicsLoss:
    """Testes para make_maxwell_physics_loss()."""

    def test_returns_scalar(self):
        """Retorna scalar float32 >= 0."""
        config = _make_config(pinns_scenario="maxwell")
        fn = make_maxwell_physics_loss(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.shape == ()
        assert loss.dtype == tf.float32
        assert loss.numpy() >= 0

    def test_flat_profile_near_zero(self):
        """Perfil plano (sem curvatura) → loss ≈ 0."""
        config = _make_config(pinns_scenario="maxwell")
        fn = make_maxwell_physics_loss(config)
        # Perfil constante: rho_h=1.0, rho_v=1.5 (log10 scale)
        flat = np.full((B, N, 2), [1.0, 1.5], dtype=np.float32)
        y_true = tf.constant(flat)
        y_pred = tf.constant(flat)
        loss = fn(y_true, y_pred)
        assert loss.numpy() < 1e-6

    def test_high_curvature_penalized(self):
        """Perfil com curvatura alta → loss > 0."""
        config = _make_config(pinns_scenario="maxwell")
        fn = make_maxwell_physics_loss(config)
        rng = np.random.default_rng(42)
        y_pred = tf.constant(rng.normal(size=(B, N, 2)).astype(np.float32))
        y_true = tf.constant(rng.normal(size=(B, N, 2)).astype(np.float32))
        loss = fn(y_true, y_pred)
        assert loss.numpy() > 0

    def test_conductive_allows_more_curvature(self):
        """Meio condutivo (alta sigma) permite mais curvatura que resistivo."""
        config = _make_config(pinns_scenario="maxwell")
        fn = make_maxwell_physics_loss(config)
        # Mesmo perfil com curvatura, mas resistividades diferentes
        rng = np.random.default_rng(42)
        curvature = rng.normal(0, 0.5, size=(B, N, 2)).astype(np.float32)

        # Condutivo: rho = 1 Ohm.m (log10 = 0)
        conductive = curvature.copy()
        conductive += 0.0  # baseline log10(rho)=0 → sigma=1
        loss_cond = fn(tf.zeros_like(tf.constant(conductive)), tf.constant(conductive))

        # Resistivo: rho = 1000 Ohm.m (log10 = 3)
        resistive = curvature.copy()
        resistive += 3.0  # baseline log10(rho)=3 → sigma=0.001
        loss_res = fn(tf.zeros_like(tf.constant(resistive)), tf.constant(resistive))

        # Resistivo deve ter loss MAIOR (menos curvatura permitida)
        assert loss_res.numpy() > loss_cond.numpy()


# ════════════════════════════════════════════════════════════════════════
# TESTES: TIV CONSTRAINT LOSS
# ════════════════════════════════════════════════════════════════════════


@skip_no_tf
class TestTIVConstraintLoss:
    """Testes para make_tiv_constraint_loss()."""

    def test_valid_tiv_zero_loss(self):
        """rho_v >= rho_h em todos os pontos → loss = 0."""
        config = _make_config()
        fn = make_tiv_constraint_loss(config)
        # rho_h < rho_v em log10 scale
        y_true, y_pred = _make_tensors(rho_v_offset=0.5)
        loss = fn(y_true, y_pred)
        assert loss.numpy() == pytest.approx(0.0, abs=1e-6)

    def test_invalid_tiv_positive_loss(self):
        """rho_v < rho_h → loss > 0."""
        config = _make_config()
        fn = make_tiv_constraint_loss(config)
        # Inversao: rho_h > rho_v
        rho_h = np.full((B, N, 1), 2.0, dtype=np.float32)
        rho_v = np.full((B, N, 1), 1.5, dtype=np.float32)
        y_pred = tf.constant(np.concatenate([rho_h, rho_v], axis=-1))
        y_true = tf.constant(np.zeros_like(y_pred.numpy()))
        loss = fn(y_true, y_pred)
        expected = (2.0 - 1.5) ** 2  # 0.25
        assert loss.numpy() == pytest.approx(expected, abs=1e-5)

    def test_equal_rho_zero_loss(self):
        """rho_v == rho_h (arenito limpo) → loss = 0."""
        config = _make_config()
        fn = make_tiv_constraint_loss(config)
        equal = np.full((B, N, 2), 1.5, dtype=np.float32)
        y_pred = tf.constant(equal)
        y_true = tf.constant(np.zeros_like(equal))
        loss = fn(y_true, y_pred)
        assert loss.numpy() == pytest.approx(0.0, abs=1e-6)

    def test_mixed_valid_invalid(self):
        """Misto: alguns pontos validos, outros invalidos."""
        config = _make_config()
        fn = make_tiv_constraint_loss(config)
        rho_h = np.array([[1.0, 2.0, 1.5, 3.0]], dtype=np.float32).reshape(1, 4, 1)
        rho_v = np.array([[1.5, 1.5, 2.0, 2.5]], dtype=np.float32).reshape(1, 4, 1)
        y_pred = tf.constant(np.concatenate([rho_h, rho_v], axis=-1))
        y_true = tf.constant(np.zeros_like(y_pred.numpy()))
        loss = fn(y_true, y_pred)
        # Violacoes: ponto 1 (2.0-1.5=0.5), ponto 3 (3.0-2.5=0.5)
        # mean(0^2 + 0.5^2 + 0^2 + 0.5^2) / 4 = 0.5/4 = 0.125
        assert loss.numpy() == pytest.approx(0.125, abs=1e-5)


# ════════════════════════════════════════════════════════════════════════
# TESTES: BUILD PINNS LOSS
# ════════════════════════════════════════════════════════════════════════


@skip_no_tf
class TestBuildPINNsLoss:
    """Testes para build_pinns_loss() — factory central."""

    def test_disabled_returns_zero(self):
        """use_pinns=False → loss = 0.0."""
        config = PipelineConfig(use_pinns=False)
        fn = build_pinns_loss(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.numpy() == pytest.approx(0.0)

    def test_oracle_returns_scalar(self):
        """Oracle scenario com epoch_var retorna scalar."""
        config = _make_config(pinns_scenario="oracle")
        epoch_var = tf.Variable(15, dtype=tf.int32)
        fn = build_pinns_loss(config, epoch_var=epoch_var)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.shape == ()
        assert loss.numpy() >= 0

    def test_warmup_zero_loss(self):
        """Com lambda_var=0.0 (warmup), loss PINNs = 0."""
        config = _make_config(
            use_tiv_constraint=False,
        )
        # Simula warmup: lambda=0.0 (callback atualizaria)
        lam_var = tf.Variable(0.0, dtype=tf.float32)
        fn = build_pinns_loss(config, pinns_lambda_var=lam_var)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.numpy() == pytest.approx(0.0, abs=1e-6)

    def test_with_tiv_constraint(self):
        """PINNs + TIV constraint ativadas juntas."""
        config = _make_config(
            use_tiv_constraint=True,
            tiv_constraint_weight=0.1,
        )
        fn = build_pinns_loss(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.numpy() >= 0

    def test_maxwell_scenario(self):
        """Maxwell scenario funciona."""
        config = _make_config(pinns_scenario="maxwell")
        fn = build_pinns_loss(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.numpy() >= 0

    def test_tiv_only_without_pinns(self):
        """TIV constraint ativa sem cenario PINN (use_pinns=False)."""
        config = PipelineConfig(
            use_pinns=False,
            use_tiv_constraint=True,
            tiv_constraint_weight=0.1,
        )
        fn = build_pinns_loss(config)
        # Cria dados com violacao TIV (rho_h > rho_v)
        rho_h = np.full((B, N, 1), 2.0, dtype=np.float32)
        rho_v = np.full((B, N, 1), 1.5, dtype=np.float32)
        y_pred = tf.constant(np.concatenate([rho_h, rho_v], axis=-1))
        y_true = tf.constant(np.zeros_like(y_pred.numpy()))
        loss = fn(y_true, y_pred)
        # TIV loss > 0 quando rho_h > rho_v
        assert loss.numpy() > 0


# ════════════════════════════════════════════════════════════════════════
# TESTES: TIV CONSTRAINT LAYER (HARD CONSTRAINT)
# ════════════════════════════════════════════════════════════════════════


@skip_no_tf
class TestTIVConstraintLayer:
    """Testes para models/blocks.py: tiv_constraint_layer()."""

    def test_valid_tiv_preserved(self):
        """Quando rho_v > rho_h, output ≈ input."""
        from geosteering_ai.models.blocks import tiv_constraint_layer

        # rho_h=1.0, rho_v=2.0 → delta = softplus(1.0) ≈ 1.3133
        # rho_v_out = 1.0 + 1.3133 = 2.3133 (ligeiramente maior que 2.0)
        x = tf.constant([[[1.0, 2.0]]], dtype=tf.float32)
        out = tiv_constraint_layer(x)
        assert out.shape == (1, 1, 2)
        # rho_h preservado
        assert out[0, 0, 0].numpy() == pytest.approx(1.0)
        # rho_v >= rho_h
        assert out[0, 0, 1].numpy() >= out[0, 0, 0].numpy()

    def test_invalid_tiv_corrected(self):
        """Quando rho_v < rho_h, corrige para rho_v ≈ rho_h."""
        from geosteering_ai.models.blocks import tiv_constraint_layer

        # rho_h=2.0, rho_v=1.0 → delta = softplus(-1.0) ≈ 0.3133
        # rho_v_out = 2.0 + 0.3133 = 2.3133 (agora >= rho_h)
        x = tf.constant([[[2.0, 1.0]]], dtype=tf.float32)
        out = tiv_constraint_layer(x)
        assert out[0, 0, 1].numpy() >= out[0, 0, 0].numpy()

    def test_preserves_extra_channels(self):
        """Com output_channels > 2, canais extras preservados."""
        from geosteering_ai.models.blocks import tiv_constraint_layer

        x = tf.constant([[[1.0, 2.0, 3.0, 4.0]]], dtype=tf.float32)
        out = tiv_constraint_layer(x)
        assert out.shape == (1, 1, 4)
        # Canais 2,3 preservados
        assert out[0, 0, 2].numpy() == pytest.approx(3.0)
        assert out[0, 0, 3].numpy() == pytest.approx(4.0)

    def test_batch_and_sequence(self):
        """Funciona com batch e sequence dimensions."""
        from geosteering_ai.models.blocks import tiv_constraint_layer

        rng = np.random.default_rng(42)
        x = tf.constant(rng.normal(size=(B, N, 2)).astype(np.float32))
        out = tiv_constraint_layer(x)
        assert out.shape == (B, N, 2)
        # rho_v >= rho_h em TODOS os pontos
        rho_h = out[..., 0].numpy()
        rho_v = out[..., 1].numpy()
        assert np.all(rho_v >= rho_h - 1e-6)

    def test_gradient_flows(self):
        """Gradientes fluem pela camada (diferenciavel)."""
        from geosteering_ai.models.blocks import tiv_constraint_layer

        x = tf.Variable(np.random.randn(2, 10, 2).astype(np.float32))
        with tf.GradientTape() as tape:
            out = tiv_constraint_layer(x)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert grads.shape == x.shape


# ════════════════════════════════════════════════════════════════════════
# TESTES: CONFIG PINNS VALIDATION
# ════════════════════════════════════════════════════════════════════════


class TestConfigPINNs:
    """Testes para campos PINN em PipelineConfig."""

    def test_default_pinns_disabled(self):
        """Por default, PINNs desativadas."""
        config = PipelineConfig()
        assert config.use_pinns is False
        assert config.pinns_scenario == "oracle"
        assert config.pinns_lambda == 0.01

    def test_valid_scenarios(self):
        """Todos os cenarios validos sao aceitos."""
        for scenario in VALID_PINNS_SCENARIOS:
            config = PipelineConfig(pinns_scenario=scenario)
            assert config.pinns_scenario == scenario

    def test_invalid_scenario_raises(self):
        """Cenario invalido levanta AssertionError."""
        with pytest.raises(AssertionError, match="pinns_scenario"):
            PipelineConfig(pinns_scenario="invalid")

    def test_valid_schedules(self):
        """Todos os schedules validos sao aceitos."""
        for schedule in VALID_LAMBDA_SCHEDULES:
            config = PipelineConfig(pinns_lambda_schedule=schedule)
            assert config.pinns_lambda_schedule == schedule

    def test_invalid_schedule_raises(self):
        """Schedule invalido levanta AssertionError."""
        with pytest.raises(AssertionError, match="pinns_lambda_schedule"):
            PipelineConfig(pinns_lambda_schedule="exponential")

    def test_invalid_norm_raises(self):
        """Norma invalida levanta AssertionError."""
        with pytest.raises(AssertionError, match="pinns_physics_norm"):
            PipelineConfig(pinns_physics_norm="mse")

    def test_negative_lambda_raises(self):
        """Lambda negativo levanta AssertionError."""
        with pytest.raises(AssertionError, match="pinns_lambda"):
            PipelineConfig(pinns_lambda=-0.01)

    def test_tiv_weight_zero_raises(self):
        """TIV weight <= 0 com use_tiv_constraint=True levanta AssertionError."""
        with pytest.raises(AssertionError, match="tiv_constraint_weight"):
            PipelineConfig(use_tiv_constraint=True, tiv_constraint_weight=0.0)

    def test_tiv_constraint_defaults(self):
        """Defaults de TIV constraint."""
        config = PipelineConfig()
        assert config.use_tiv_constraint is False
        assert config.tiv_constraint_weight == 0.1


# ════════════════════════════════════════════════════════════════════════
# TESTES: FACTORY INTEGRATION
# ════════════════════════════════════════════════════════════════════════


@skip_no_tf
class TestFactoryIntegration:
    """Testes de integracao: build_combined com PINNs."""

    def test_build_combined_with_pinns(self):
        """build_combined integra PINNs na loss combinada."""
        from geosteering_ai.losses.factory import LossFactory

        config = _make_config(loss_type="rmse")
        fn = LossFactory.build_combined(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.shape == ()
        assert loss.numpy() > 0

    def test_build_combined_with_tiv_only(self):
        """TIV constraint sem cenario PINN (use_pinns=False)."""
        from geosteering_ai.losses.factory import LossFactory

        config = PipelineConfig(
            use_pinns=False,
            use_tiv_constraint=True,
            tiv_constraint_weight=0.5,
        )
        fn = LossFactory.build_combined(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.shape == ()
        assert loss.numpy() > 0

    def test_build_combined_pinns_plus_look_ahead(self):
        """PINNs + look_ahead combinadas."""
        from geosteering_ai.losses.factory import LossFactory

        config = _make_config(
            use_look_ahead_loss=True,
            look_ahead_weight=0.1,
        )
        fn = LossFactory.build_combined(config)
        y_true, y_pred = _make_tensors()
        loss = fn(y_true, y_pred)
        assert loss.shape == ()
        assert loss.numpy() > 0
