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
        expected_lr = config.learning_rate * (config.stage_lr_decay ** 2)
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

    def test_epochs_distribution_n3(self):
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
