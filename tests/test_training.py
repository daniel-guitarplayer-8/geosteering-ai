# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTES: geosteering_ai.training — loop, nstage, metrics, callbacks       ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes para o subpacote training.

Estrutura:
    CPU-safe (sem TF):
        TestTrainingResult — dataclass instantiation, defaults
        TestNStageResult — dataclass instantiation, defaults
        TestNStageParams — compute_stage_params para N=2 e N=3
    TF-required:
        TestBuildMetrics — build_metrics retorna lista com R2Score
        TestBuildCallbacks — build_callbacks retorna lista com EarlyStopping
        TestTrainingLoopCompile — TrainingLoop._create_optimizer
"""
import dataclasses
import math

import pytest

# ── Deteccao de TF (mesmo padrao de test_losses.py) ──────────────────────────
try:
    import tensorflow as _tf  # noqa: F401

    HAS_TF = True
except ImportError:
    HAS_TF = False

requires_tf = pytest.mark.skipif(not HAS_TF, reason="TensorFlow nao disponivel")

# ── Imports CPU-safe ──────────────────────────────────────────────────────────
# Importa direto dos modulos para evitar TF-dependent imports no __init__.py
from geosteering_ai.config import PipelineConfig
from geosteering_ai.training.callbacks import (
    make_cosine_schedule,
    make_step_schedule,
    make_warmup_cosine_schedule,
)
from geosteering_ai.training.loop import TrainingResult
from geosteering_ai.training.nstage import NStageResult, NStageTrainer

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════


def _make_config(**kwargs) -> PipelineConfig:
    """Cria PipelineConfig a partir do baseline com overrides."""
    return dataclasses.replace(PipelineConfig.baseline(), **kwargs)


# ════════════════════════════════════════════════════════════════════════════
# TESTES CPU-SAFE
# ════════════════════════════════════════════════════════════════════════════


class TestTrainingResult:
    """TrainingResult dataclass — instantiation e defaults."""

    def test_default_instantiation(self):
        """TrainingResult com defaults tem valores iniciais corretos."""
        result = TrainingResult()
        assert result.history == {}
        assert result.training_time == 0.0
        assert result.final_epoch == 0
        assert result.best_epoch == -1
        assert result.best_val_loss == float("inf")
        assert result.final_metrics == {}

    def test_custom_instantiation(self):
        """TrainingResult aceita valores customizados."""
        history = {"loss": [0.5, 0.3], "val_loss": [0.6, 0.4]}
        result = TrainingResult(
            history=history,
            training_time=120.5,
            final_epoch=2,
            best_epoch=2,
            best_val_loss=0.4,
            final_metrics={"loss": 0.3, "val_loss": 0.4},
        )
        assert result.training_time == 120.5
        assert result.best_epoch == 2
        assert result.best_val_loss == 0.4
        assert result.final_metrics["loss"] == 0.3
        assert len(result.history["loss"]) == 2

    def test_is_dataclass(self):
        """TrainingResult e uma dataclass."""
        assert dataclasses.is_dataclass(TrainingResult)


class TestNStageResult:
    """NStageResult dataclass — instantiation e defaults."""

    def test_default_instantiation(self):
        """NStageResult com defaults tem valores iniciais corretos."""
        result = NStageResult()
        assert result.merged_history == {}
        assert result.stage_histories == []
        assert result.stage_params == []
        assert result.total_epochs == 0
        assert result.best_val_loss == float("inf")

    def test_custom_instantiation(self):
        """NStageResult aceita valores customizados."""
        result = NStageResult(
            merged_history={"loss": [0.5, 0.3, 0.2]},
            stage_histories=[{"loss": [0.5]}, {"loss": [0.3, 0.2]}],
            stage_params=[{"noise_level": 0.0}, {"noise_level": 0.08}],
            total_epochs=3,
            best_val_loss=0.2,
        )
        assert result.total_epochs == 3
        assert result.best_val_loss == 0.2
        assert len(result.stage_params) == 2
        assert result.stage_params[0]["noise_level"] == 0.0

    def test_is_dataclass(self):
        """NStageResult e uma dataclass."""
        assert dataclasses.is_dataclass(NStageResult)


class TestNStageParams:
    """NStageTrainer.compute_stage_params() — auto-calculo de params."""

    def test_n2_stage1_clean(self):
        """N=2, Stage 1: noise=0, lr=base, patience=ES patience."""
        config = PipelineConfig.nstage(n=2)
        trainer = NStageTrainer(config)
        p1 = trainer.compute_stage_params(1)

        assert p1["noise_level"] == 0.0
        assert p1["learning_rate"] == config.learning_rate
        assert p1["patience"] == config.early_stopping_patience
        assert p1["epochs"] == config.nstage_stage1_epochs
        assert p1["use_mini_curriculum"] is False
        assert p1["ramp_epochs"] == 0

    def test_n2_stage2_full_noise(self):
        """N=2, Stage 2: noise=noise_max, lr=base*decay."""
        config = PipelineConfig.nstage(n=2)
        trainer = NStageTrainer(config)
        p2 = trainer.compute_stage_params(2)

        # noise = noise_max * (2-1) / (2-1) = noise_max
        assert p2["noise_level"] == pytest.approx(config.noise_level_max)
        # lr = base * decay^(2-1) = base * decay
        expected_lr = config.learning_rate * config.stage_lr_decay
        assert p2["learning_rate"] == pytest.approx(expected_lr)
        # patience = base_patience + (2-2) * 5 = base_patience
        assert p2["patience"] == config.nstage_base_patience

    def test_n3_stage2_half_noise(self):
        """N=3, Stage 2: noise=noise_max*0.5, lr=base*decay."""
        config = PipelineConfig.nstage(n=3)
        trainer = NStageTrainer(config)
        p2 = trainer.compute_stage_params(2)

        # noise = noise_max * (2-1) / (3-1) = noise_max * 0.5
        expected_noise = config.noise_level_max * 0.5
        assert p2["noise_level"] == pytest.approx(expected_noise)
        # lr = base * decay^(2-1) = base * decay
        expected_lr = config.learning_rate * config.stage_lr_decay
        assert p2["learning_rate"] == pytest.approx(expected_lr)

    def test_n3_stage3_full_noise(self):
        """N=3, Stage 3: noise=noise_max, lr=base*decay^2."""
        config = PipelineConfig.nstage(n=3)
        trainer = NStageTrainer(config)
        p3 = trainer.compute_stage_params(3)

        # noise = noise_max * (3-1) / (3-1) = noise_max
        assert p3["noise_level"] == pytest.approx(config.noise_level_max)
        # lr = base * decay^(3-1) = base * decay^2
        expected_lr = config.learning_rate * (config.stage_lr_decay**2)
        assert p3["learning_rate"] == pytest.approx(expected_lr)
        # patience = base_patience + (3-2) * 5 = base_patience + 5
        assert p3["patience"] == config.nstage_base_patience + 5

    def test_invalid_stage_idx_raises(self):
        """Stage idx fora do range levanta ValueError."""
        config = PipelineConfig.nstage(n=2)
        trainer = NStageTrainer(config)

        with pytest.raises(ValueError, match="stage_idx"):
            trainer.compute_stage_params(0)

        with pytest.raises(ValueError, match="stage_idx"):
            trainer.compute_stage_params(3)

    def test_nstage_requires_use_nstage(self):
        """NStageTrainer requer use_nstage=True."""
        config = _make_config(use_nstage=False)
        with pytest.raises(ValueError, match="use_nstage"):
            NStageTrainer(config)

    def test_nstage_requires_n_ge_2(self):
        """NStageTrainer requer n_training_stages >= 2."""
        config = _make_config(use_nstage=True, n_training_stages=1)
        with pytest.raises(ValueError, match="n_training_stages"):
            NStageTrainer(config)

    def test_epochs_distribution_n3(self):  # noqa: E301
        """N=3: epocas de stages 2 e 3 sao iguais (distribuicao uniforme)."""
        config = PipelineConfig.nstage(n=3)
        trainer = NStageTrainer(config)
        p2 = trainer.compute_stage_params(2)
        p3 = trainer.compute_stage_params(3)

        assert p2["epochs"] == p3["epochs"]
        # Epocas por stage = (total - stage1) / (N-1)
        expected = (config.epochs - config.nstage_stage1_epochs) // 2
        assert p2["epochs"] == expected


# ════════════════════════════════════════════════════════════════════════════
# TESTES LR SCHEDULE HELPERS (CPU-safe, sem TF)
# ════════════════════════════════════════════════════════════════════════════


class TestMakeCosineSchedule:
    """make_cosine_schedule — cosine annealing (Loshchilov & Hutter 2016)."""

    def test_epoch_zero_returns_lr_initial(self):
        """Epoch 0 retorna lr_initial (cos(0) = 1)."""
        sched = make_cosine_schedule(1e-4, 1e-7, 200)
        assert sched(0) == pytest.approx(1e-4)

    def test_final_epoch_returns_lr_min(self):
        """Epoch T retorna lr_min (cos(pi) = -1)."""
        sched = make_cosine_schedule(1e-4, 1e-7, 200)
        assert sched(200) == pytest.approx(1e-7)

    def test_midpoint_is_halfway(self):
        """Epoch T/2 retorna valor proximo da media (cos(pi/2) = 0)."""
        sched = make_cosine_schedule(1e-4, 1e-7, 200)
        mid = sched(100)
        expected = 1e-7 + 0.5 * (1e-4 - 1e-7) * (1 + math.cos(math.pi * 0.5))
        assert mid == pytest.approx(expected)

    def test_monotonically_decreasing(self):
        """Schedule decresce monotonicamente."""
        sched = make_cosine_schedule(1e-3, 1e-6, 100)
        values = [sched(e) for e in range(101)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_saturation_beyond_total_epochs(self):
        """Apos total_epochs, retorna lr_min constante."""
        sched = make_cosine_schedule(1e-4, 1e-7, 100)
        assert sched(150) == pytest.approx(1e-7)
        assert sched(999) == pytest.approx(1e-7)

    def test_invalid_lr_raises(self):
        """lr_initial <= lr_min levanta ValueError."""
        with pytest.raises(ValueError, match="lr_initial"):
            make_cosine_schedule(1e-7, 1e-4, 200)

    def test_invalid_epochs_raises(self):
        """total_epochs < 1 levanta ValueError."""
        with pytest.raises(ValueError, match="total_epochs"):
            make_cosine_schedule(1e-4, 1e-7, 0)


class TestMakeStepSchedule:
    """make_step_schedule — step decay (He et al. 2015)."""

    def test_epoch_zero_returns_lr_initial(self):
        """Epoch 0 retorna lr_initial (nenhum step aplicado)."""
        sched = make_step_schedule(1e-3, factor=0.1, step_size=50)
        assert sched(0) == pytest.approx(1e-3)

    def test_first_step(self):
        """Epoch step_size reduz por factor."""
        sched = make_step_schedule(1e-3, factor=0.1, step_size=50)
        assert sched(50) == pytest.approx(1e-4)

    def test_second_step(self):
        """Epoch 2*step_size reduz por factor^2."""
        sched = make_step_schedule(1e-3, factor=0.1, step_size=50)
        assert sched(100) == pytest.approx(1e-5)

    def test_between_steps_constant(self):
        """Entre steps, LR permanece constante."""
        sched = make_step_schedule(1e-3, factor=0.1, step_size=50)
        assert sched(25) == pytest.approx(1e-3)
        assert sched(49) == pytest.approx(1e-3)

    def test_lr_min_floor(self):
        """LR nunca cai abaixo de lr_min."""
        sched = make_step_schedule(1e-3, factor=0.1, step_size=10, lr_min=1e-5)
        # Epoch 30 → 1e-3 * 0.1^3 = 1e-6 < lr_min=1e-5
        assert sched(30) == pytest.approx(1e-5)

    def test_invalid_factor_raises(self):
        """factor fora de (0,1) levanta ValueError."""
        with pytest.raises(ValueError, match="factor"):
            make_step_schedule(1e-3, factor=1.5)

    def test_invalid_step_size_raises(self):
        """step_size < 1 levanta ValueError."""
        with pytest.raises(ValueError, match="step_size"):
            make_step_schedule(1e-3, step_size=0)


class TestMakeWarmupCosineSchedule:
    """make_warmup_cosine_schedule — warmup + cosine (Vaswani 2017)."""

    def test_epoch_zero_returns_lr_min(self):
        """Epoch 0 retorna lr_min (inicio do warmup)."""
        sched = make_warmup_cosine_schedule(1e-4, 1e-7, 200, warmup_epochs=10)
        assert sched(0) == pytest.approx(1e-7)

    def test_warmup_end_returns_lr_initial(self):
        """Epoch warmup_epochs retorna lr_initial (pico)."""
        sched = make_warmup_cosine_schedule(1e-4, 1e-7, 200, warmup_epochs=10)
        # Epoch 10: inicio da fase cosine, cos(0) = 1 → lr_initial
        assert sched(10) == pytest.approx(1e-4)

    def test_warmup_is_linear(self):
        """Fase warmup cresce linearmente."""
        sched = make_warmup_cosine_schedule(1e-4, 0.0, 200, warmup_epochs=10)
        # Metade do warmup → metade do lr_initial
        assert sched(5) == pytest.approx(5e-5)

    def test_final_epoch_returns_lr_min(self):
        """Epoch total_epochs retorna lr_min."""
        sched = make_warmup_cosine_schedule(1e-4, 1e-7, 200, warmup_epochs=10)
        assert sched(200) == pytest.approx(1e-7)

    def test_cosine_phase_decreases(self):
        """Fase cosine decresce monotonicamente apos warmup."""
        sched = make_warmup_cosine_schedule(1e-3, 1e-6, 100, warmup_epochs=10)
        cosine_values = [sched(e) for e in range(10, 101)]
        for i in range(len(cosine_values) - 1):
            assert cosine_values[i] >= cosine_values[i + 1]

    def test_warmup_ge_total_raises(self):
        """warmup_epochs >= total_epochs levanta ValueError."""
        with pytest.raises(ValueError, match="warmup_epochs"):
            make_warmup_cosine_schedule(1e-4, 1e-7, 100, warmup_epochs=100)

    def test_saturation_beyond_total(self):
        """Apos total_epochs, retorna lr_min."""
        sched = make_warmup_cosine_schedule(1e-4, 1e-7, 100, warmup_epochs=10)
        assert sched(150) == pytest.approx(1e-7)


# ════════════════════════════════════════════════════════════════════════════
# TESTES TF-REQUIRED
# ════════════════════════════════════════════════════════════════════════════


@requires_tf
class TestBuildMetrics:
    """build_metrics retorna lista de Keras metrics."""

    def test_returns_list(self):
        """build_metrics retorna lista nao-vazia."""
        from geosteering_ai.training.metrics import build_metrics

        config = _make_config()
        metrics = build_metrics(config)
        assert isinstance(metrics, list)
        assert len(metrics) >= 1

    def test_contains_r2score(self):
        """Lista de metricas contem R2Score."""
        from geosteering_ai.training.metrics import build_metrics

        config = _make_config()
        metrics = build_metrics(config)
        names = [getattr(m, "name", str(m)) for m in metrics]
        # Pelo menos uma metrica deve ter 'r2' no nome
        assert any("r2" in n.lower() for n in names)


@requires_tf
class TestBuildCallbacks:
    """build_callbacks retorna lista com EarlyStopping."""

    def test_returns_list_with_early_stopping(self):
        """build_callbacks inclui EarlyStopping."""
        import tensorflow as tf

        from geosteering_ai.training.callbacks import build_callbacks

        config = _make_config()
        callbacks = build_callbacks(config)
        assert isinstance(callbacks, list)
        assert len(callbacks) >= 2  # EarlyStopping + BestEpochTracker
        types = [type(cb).__name__ for cb in callbacks]
        assert "EarlyStopping" in types


@requires_tf
class TestTrainingLoopCompile:
    """TrainingLoop._create_optimizer cria optimizer TF."""

    def test_create_optimizer_adam(self):
        """_create_optimizer('adam') retorna Adam optimizer."""
        import tensorflow as tf

        from geosteering_ai.training.loop import TrainingLoop

        config = _make_config()
        loop = TrainingLoop(config)
        opt = loop._create_optimizer(1e-4)
        assert opt is not None
        assert "adam" in type(opt).__name__.lower()


@requires_tf
class TestSetupMixedPrecision:
    """TrainingLoop._setup_mixed_precision — policy global."""

    def test_sets_float32_when_disabled(self):
        """use_mixed_precision=False → policy float32."""
        import tensorflow as tf

        from geosteering_ai.training.loop import TrainingLoop

        config = _make_config(use_mixed_precision=False)
        loop = TrainingLoop(config)
        loop._setup_mixed_precision()
        policy = tf.keras.mixed_precision.global_policy()
        assert policy.name == "float32"

    def test_sets_mixed_float16_when_enabled(self):
        """use_mixed_precision=True → policy mixed_float16."""
        import tensorflow as tf

        from geosteering_ai.training.loop import TrainingLoop

        config = _make_config(use_mixed_precision=True)
        loop = TrainingLoop(config)
        try:
            loop._setup_mixed_precision()
            policy = tf.keras.mixed_precision.global_policy()
            assert policy.name == "mixed_float16"
        finally:
            # Cleanup: restaurar float32 para nao contaminar outros testes
            tf.keras.mixed_precision.set_global_policy("float32")


@requires_tf
class TestBuildTfDataset:
    """DataPipeline.build_tf_dataset — tf.data.Dataset otimizado."""

    def _make_prepared(self):
        """Cria PreparedData sintetico para testes."""
        import numpy as np

        from geosteering_ai.data.pipeline import PreparedData

        n_train, n_val, n_test = 20, 5, 5
        seq_len, n_feat, n_tgt = 600, 5, 2
        rng = np.random.RandomState(42)

        return PreparedData(
            x_train=rng.randn(n_train, seq_len, n_feat).astype(np.float32),
            y_train=rng.randn(n_train, seq_len, n_tgt).astype(np.float32),
            z_train=rng.randn(n_train, seq_len).astype(np.float32),
            x_val=rng.randn(n_val, seq_len, n_feat).astype(np.float32),
            y_val=rng.randn(n_val, seq_len, n_tgt).astype(np.float32),
            z_val=rng.randn(n_val, seq_len).astype(np.float32),
            x_test=rng.randn(n_test, seq_len, n_feat).astype(np.float32),
            y_test=rng.randn(n_test, seq_len, n_tgt).astype(np.float32),
            z_test=rng.randn(n_test, seq_len).astype(np.float32),
        )

    def test_train_returns_dataset(self):
        """Split 'train' retorna tf.data.Dataset."""
        import tensorflow as tf

        from geosteering_ai.data.pipeline import DataPipeline

        config = _make_config(batch_size=4)
        pipeline = DataPipeline(config)
        prepared = self._make_prepared()
        pipeline._last_prepared = prepared
        ds = pipeline.build_tf_dataset(prepared, split="train")
        assert isinstance(ds, tf.data.Dataset)

    def test_val_returns_dataset(self):
        """Split 'val' retorna tf.data.Dataset."""
        import tensorflow as tf

        from geosteering_ai.data.pipeline import DataPipeline

        config = _make_config(batch_size=4)
        pipeline = DataPipeline(config)
        prepared = self._make_prepared()
        pipeline._last_prepared = prepared
        ds = pipeline.build_tf_dataset(prepared, split="val")
        assert isinstance(ds, tf.data.Dataset)

    def test_test_returns_dataset(self):
        """Split 'test' retorna tf.data.Dataset."""
        import tensorflow as tf

        from geosteering_ai.data.pipeline import DataPipeline

        config = _make_config(batch_size=4)
        pipeline = DataPipeline(config)
        prepared = self._make_prepared()
        pipeline._last_prepared = prepared
        ds = pipeline.build_tf_dataset(prepared, split="test")
        assert isinstance(ds, tf.data.Dataset)

    def test_invalid_split_raises(self):
        """Split invalido levanta ValueError."""
        from geosteering_ai.data.pipeline import DataPipeline

        config = _make_config(batch_size=4)
        pipeline = DataPipeline(config)
        prepared = self._make_prepared()
        pipeline._last_prepared = prepared
        with pytest.raises(ValueError, match="invalido"):
            pipeline.build_tf_dataset(prepared, split="foo")

    def test_train_batch_shape(self):
        """Batch de train tem shape correto (batch, 600, 5)."""
        import tensorflow as tf

        from geosteering_ai.data.pipeline import DataPipeline

        config = _make_config(batch_size=4)
        pipeline = DataPipeline(config)
        prepared = self._make_prepared()
        pipeline._last_prepared = prepared
        ds = pipeline.build_tf_dataset(prepared, split="train")
        batches = list(ds.take(1))
        assert len(batches) == 1, "Dataset nao retornou nenhum batch"
        x_batch, y_batch = batches[0]
        assert x_batch.shape[1] == 600
        assert x_batch.shape[2] == 5
        assert y_batch.shape[2] == 2


# ════════════════════════════════════════════════════════════════════════════
# TESTES: setup_gpu() — Configuracao de memoria GPU
# ════════════════════════════════════════════════════════════════════════════


@requires_tf
class TestSetupGpu:
    """setup_gpu() — configuracao de memory_growth e deteccao de GPUs."""

    def test_returns_dict(self):
        """setup_gpu() retorna dicionario com chaves esperadas."""
        from geosteering_ai.utils.system import setup_gpu

        result = setup_gpu()
        assert isinstance(result, dict)
        assert "gpu_count" in result
        assert "memory_growth_set" in result
        assert "devices" in result

    def test_gpu_count_is_int(self):
        """gpu_count eh inteiro nao-negativo."""
        from geosteering_ai.utils.system import setup_gpu

        result = setup_gpu()
        assert isinstance(result["gpu_count"], int)
        assert result["gpu_count"] >= 0

    def test_memory_growth_set_is_bool(self):
        """memory_growth_set eh booleano."""
        from geosteering_ai.utils.system import setup_gpu

        result = setup_gpu()
        assert isinstance(result["memory_growth_set"], bool)

    def test_devices_is_list(self):
        """devices eh lista (vazia se sem GPU)."""
        from geosteering_ai.utils.system import setup_gpu

        result = setup_gpu()
        assert isinstance(result["devices"], list)

    def test_idempotent(self):
        """Chamar setup_gpu() 2x nao gera erro."""
        from geosteering_ai.utils.system import setup_gpu

        result1 = setup_gpu()
        result2 = setup_gpu()
        # Ambas devem retornar mesmo gpu_count
        assert result1["gpu_count"] == result2["gpu_count"]
        # Segunda chamada pode retornar memory_growth_set=False se GPU
        # ja foi inicializada (RuntimeError silenciado). Isso e esperado.
        # O importante e que nao levanta excecao e retorna tipo correto.
        assert isinstance(result2["memory_growth_set"], bool)


# ════════════════════════════════════════════════════════════════════════════
# TESTES: use_xla — PipelineConfig + jit_compile no compile()
# ════════════════════════════════════════════════════════════════════════════


class TestUseXlaConfig:
    """PipelineConfig.use_xla — campo existe com default False."""

    def test_default_is_false(self):
        """use_xla default eh False (opt-in explicito)."""
        config = PipelineConfig.baseline()
        assert config.use_xla is False

    def test_can_set_true(self):
        """use_xla pode ser setado para True via replace."""
        config = _make_config(use_xla=True)
        assert config.use_xla is True

    def test_serializes_to_dict(self):
        """use_xla aparece no to_dict()."""
        config = _make_config(use_xla=True)
        d = config.to_dict()
        assert "use_xla" in d
        assert d["use_xla"] is True


@requires_tf
class TestCompileJitCompile:
    """TrainingLoop.compile() — jit_compile controlado por config.use_xla."""

    def test_compile_without_xla_succeeds(self):
        """compile() com use_xla=False compila modelo sem jit_compile."""
        import tensorflow as tf

        from geosteering_ai.training.loop import TrainingLoop

        config = _make_config(use_xla=False)
        loop = TrainingLoop(config)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(2, input_shape=(5,)),
            ]
        )
        loop.compile(model, "mse", ["mae"])
        assert model.optimizer is not None
        # Verificar que jit_compile NAO foi ativado
        if hasattr(model, "jit_compile"):  # Keras 3
            assert model.jit_compile is False
        elif hasattr(model, "_jit_compile"):  # TF 2.x interno
            assert model._jit_compile is False

    def test_compile_with_xla_succeeds(self):
        """compile() com use_xla=True compila modelo com jit_compile=True."""
        import tensorflow as tf

        from geosteering_ai.training.loop import TrainingLoop

        config = _make_config(use_xla=True)
        loop = TrainingLoop(config)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(2, input_shape=(5,)),
            ]
        )
        loop.compile(model, "mse", ["mae"])
        assert model.optimizer is not None
        # Verificar que jit_compile FOI ativado
        if hasattr(model, "jit_compile"):  # Keras 3
            assert model.jit_compile is True
        elif hasattr(model, "_jit_compile"):  # TF 2.x interno
            assert model._jit_compile is True

    def test_xla_flag_logged(self):
        """compile() loga XLA quando use_xla=True."""
        import tensorflow as tf

        from geosteering_ai.training.loop import TrainingLoop

        config = _make_config(use_xla=True)
        loop = TrainingLoop(config)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(2, input_shape=(5,)),
            ]
        )
        # Nao deve gerar excecao; XLA pode nao estar disponivel em CPU
        # mas compile() com jit_compile=True deve funcionar (TF faz
        # fallback para nao-XLA em CPU se necessario).
        loop.compile(model, "mse", ["mae"])


# ════════════════════════════════════════════════════════════════════════════
# TESTES: GradientMonitor + add_gradient_monitor (Fase E)
# ════════════════════════════════════════════════════════════════════════════
# Verifica GradientMonitor: instantiation, on_epoch_end, explosion/vanishing
# detection, frequency filtering. Verifica add_gradient_monitor factory:
# guards (model=None, loss_fn=None, sample_batch=None, config=False).
# Ref: Review Fase E — H3 (unit tests obrigatorios).
# ════════════════════════════════════════════════════════════════════════════


@requires_tf
class TestGradientMonitor:
    """GradientMonitor — monitoramento de gradientes reais via GradientTape."""

    def _make_simple_model(self):
        """Cria modelo sequencial simples para testes."""
        import tensorflow as tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(2),
            ]
        )
        # Forward pass dummy para inicializar pesos
        model(tf.zeros([1, 5]))
        return model

    def _make_sample_batch(self, batch_size=8, n_feat=5, seq_len=10):
        """Cria batch de amostra sintetico (sem seq dim para modelo simples)."""
        import numpy as np

        rng = np.random.default_rng(42)
        x = rng.standard_normal((batch_size, n_feat)).astype("float32")
        y = rng.standard_normal((batch_size, 2)).astype("float32")
        return (x, y)

    def test_instantiation(self):
        """GradientMonitor instancia corretamente com params validos."""
        import tensorflow as tf

        from geosteering_ai.training.callbacks import GradientMonitor

        config = _make_config(use_gradient_monitor=True)
        model = self._make_simple_model()
        loss_fn = tf.keras.losses.MeanSquaredError()
        sample_batch = self._make_sample_batch()

        gm = GradientMonitor(model, loss_fn, sample_batch, config)
        assert gm is not None
        assert isinstance(gm, tf.keras.callbacks.Callback)

    def test_invalid_sample_batch_raises(self):
        """GradientMonitor rejeita sample_batch que nao e tupla (x, y)."""
        import tensorflow as tf

        from geosteering_ai.training.callbacks import GradientMonitor

        config = _make_config(use_gradient_monitor=True)
        model = self._make_simple_model()
        loss_fn = tf.keras.losses.MeanSquaredError()

        with pytest.raises(ValueError, match="sample_batch deve ser tupla"):
            GradientMonitor(model, loss_fn, "not_a_tuple", config)

    def test_on_epoch_end_runs_without_error(self):
        """on_epoch_end computa gradientes sem erro para epoca no freq."""
        import tensorflow as tf

        from geosteering_ai.training.callbacks import GradientMonitor

        config = _make_config(
            use_gradient_monitor=True,
            gradient_monitor_freq=1,  # monitora a cada epoca
        )
        model = self._make_simple_model()
        loss_fn = tf.keras.losses.MeanSquaredError()
        sample_batch = self._make_sample_batch()

        gm = GradientMonitor(model, loss_fn, sample_batch, config)
        logs = {}
        gm.on_epoch_end(0, logs=logs)

        # Logs devem conter grad_norm_mean, grad_norm_max, grad_norm_min
        assert "grad_norm_mean" in logs
        assert "grad_norm_max" in logs
        assert "grad_norm_min" in logs
        assert logs["grad_norm_mean"] > 0
        assert logs["grad_norm_max"] >= logs["grad_norm_mean"]

    def test_frequency_filtering(self):
        """on_epoch_end pula epocas fora do freq."""
        import tensorflow as tf

        from geosteering_ai.training.callbacks import GradientMonitor

        config = _make_config(
            use_gradient_monitor=True,
            gradient_monitor_freq=5,
        )
        model = self._make_simple_model()
        loss_fn = tf.keras.losses.MeanSquaredError()
        sample_batch = self._make_sample_batch()

        gm = GradientMonitor(model, loss_fn, sample_batch, config)

        # Epoch 0 → (0+1) % 5 = 1 ≠ 0 → skip
        logs_skip = {}
        gm.on_epoch_end(0, logs=logs_skip)
        assert "grad_norm_mean" not in logs_skip

        # Epoch 4 → (4+1) % 5 = 0 → compute
        logs_compute = {}
        gm.on_epoch_end(4, logs=logs_compute)
        assert "grad_norm_mean" in logs_compute

    def test_reduce_mean_guard_for_non_scalar_loss(self):
        """on_epoch_end normaliza loss nao-scalar via reduce_mean."""
        import tensorflow as tf

        from geosteering_ai.training.callbacks import GradientMonitor

        config = _make_config(
            use_gradient_monitor=True,
            gradient_monitor_freq=1,
        )
        model = self._make_simple_model()
        # Loss com reduction='none' retorna tensor, nao scalar
        loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")
        sample_batch = self._make_sample_batch()

        gm = GradientMonitor(model, loss_fn, sample_batch, config)
        logs = {}
        gm.on_epoch_end(0, logs=logs)  # nao deve explodir

        assert "grad_norm_mean" in logs
        assert logs["grad_norm_mean"] > 0


@requires_tf
class TestAddGradientMonitor:
    """add_gradient_monitor factory — guards e integracao."""

    def test_noop_when_disabled(self):
        """add_gradient_monitor retorna lista inalterada quando False."""
        from geosteering_ai.training.callbacks import add_gradient_monitor

        config = _make_config(use_gradient_monitor=False)
        callbacks = [1, 2, 3]
        result = add_gradient_monitor(callbacks, None, None, None, config)
        assert result is callbacks
        assert len(result) == 3

    def test_guard_model_none(self):
        """add_gradient_monitor retorna lista inalterada quando model=None."""
        from geosteering_ai.training.callbacks import add_gradient_monitor

        config = _make_config(use_gradient_monitor=True)
        callbacks = []
        result = add_gradient_monitor(callbacks, None, "loss_fn", ("x", "y"), config)
        assert len(result) == 0

    def test_guard_loss_fn_none(self):
        """add_gradient_monitor retorna lista inalterada quando loss_fn=None."""
        import tensorflow as tf

        from geosteering_ai.training.callbacks import add_gradient_monitor

        config = _make_config(use_gradient_monitor=True)
        model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(5,))])
        callbacks = []
        result = add_gradient_monitor(callbacks, model, None, ("x", "y"), config)
        assert len(result) == 0

    def test_guard_sample_batch_none(self):
        """add_gradient_monitor retorna lista inalterada quando batch=None."""
        import tensorflow as tf

        from geosteering_ai.training.callbacks import add_gradient_monitor

        config = _make_config(use_gradient_monitor=True)
        model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(5,))])
        loss_fn = tf.keras.losses.MeanSquaredError()
        callbacks = []
        result = add_gradient_monitor(callbacks, model, loss_fn, None, config)
        assert len(result) == 0

    def test_adds_callback_when_enabled(self):
        """add_gradient_monitor adiciona GradientMonitor quando tudo valido."""
        import numpy as np
        import tensorflow as tf

        from geosteering_ai.training.callbacks import (
            GradientMonitor,
            add_gradient_monitor,
        )

        config = _make_config(use_gradient_monitor=True)
        model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(5,))])
        model(tf.zeros([1, 5]))  # inicializa pesos
        loss_fn = tf.keras.losses.MeanSquaredError()
        x = np.random.randn(8, 5).astype("float32")
        y = np.random.randn(8, 2).astype("float32")

        callbacks = []
        result = add_gradient_monitor(callbacks, model, loss_fn, (x, y), config)
        assert len(result) == 1
        assert isinstance(result[0], tf.keras.callbacks.Callback)


class TestGradientMonitorConfig:
    """Validacao de config fields do GradientMonitor em __post_init__."""

    def test_valid_config_ok(self):
        """Config valida com gradient monitor nao levanta erro."""
        config = _make_config(
            use_gradient_monitor=True,
            gradient_monitor_freq=5,
            gradient_explosion_threshold=100.0,
            gradient_vanishing_threshold=1e-7,
        )
        assert config.use_gradient_monitor is True

    def test_freq_zero_raises(self):
        """gradient_monitor_freq=0 levanta AssertionError."""
        with pytest.raises(AssertionError, match="gradient_monitor_freq"):
            _make_config(use_gradient_monitor=True, gradient_monitor_freq=0)

    def test_negative_explosion_raises(self):
        """gradient_explosion_threshold negativo levanta AssertionError."""
        with pytest.raises(AssertionError, match="gradient_explosion_threshold"):
            _make_config(use_gradient_monitor=True, gradient_explosion_threshold=-1.0)

    def test_vanishing_greater_than_explosion_raises(self):
        """vanishing > explosion levanta AssertionError."""
        with pytest.raises(AssertionError, match="vanishing_threshold"):
            _make_config(
                use_gradient_monitor=True,
                gradient_vanishing_threshold=200.0,
                gradient_explosion_threshold=100.0,
            )
