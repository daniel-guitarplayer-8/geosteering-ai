# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTES: geosteering_ai.inference — realtime, pipeline, export            ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes para o subpacote inference.

Estrutura:
    CPU-safe (sem TF):
        TestRealtimeInference — buffer operations sem modelo
    TF-required:
        TestExportFunctions — export functions existem e sao callable
        TestInferencePipeline — InferencePipeline instantiation
"""
import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Deteccao de TF (mesmo padrao de test_losses.py) ──────────────────────────
try:
    import tensorflow as _tf  # noqa: F401
    HAS_TF = True
except ImportError:
    HAS_TF = False

requires_tf = pytest.mark.skipif(not HAS_TF, reason="TensorFlow nao disponivel")

# ── Imports CPU-safe ──────────────────────────────────────────────────────────
from geosteering_ai.config import PipelineConfig
from geosteering_ai.inference.realtime import RealtimeInference


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _make_config(**kwargs) -> PipelineConfig:
    """Cria PipelineConfig a partir do baseline com overrides."""
    return dataclasses.replace(PipelineConfig.baseline(), **kwargs)


def _mock_pipeline(window_size: int = 600, n_columns: int = 22):
    """Cria mock de InferencePipeline para testes do RealtimeInference."""
    pipeline = MagicMock()
    pipeline.config = MagicMock()
    pipeline.config.n_columns = n_columns
    pipeline.config.model_type = "ResNet_18"
    pipeline.predict.return_value = np.zeros((1, window_size, 2))
    return pipeline


# ════════════════════════════════════════════════════════════════════════════
# TESTES CPU-SAFE
# ════════════════════════════════════════════════════════════════════════════


class TestRealtimeInference:
    """RealtimeInference — buffer operations sem modelo real."""

    def test_default_window_size_600(self):
        """Window size default e 600 (SEQUENCE_LENGTH)."""
        pipeline = _mock_pipeline()
        rt = RealtimeInference(pipeline, window_size=600)
        assert rt.window_size == 600

    def test_is_ready_false_when_empty(self):
        """Buffer vazio: is_ready retorna False."""
        pipeline = _mock_pipeline()
        rt = RealtimeInference(pipeline, window_size=600)
        assert rt.is_ready is False

    def test_buffer_fill_empty(self):
        """Buffer vazio: buffer_fill retorna 0.0."""
        pipeline = _mock_pipeline()
        rt = RealtimeInference(pipeline, window_size=600)
        assert rt.buffer_fill == pytest.approx(0.0)

    def test_buffer_fill_partial(self):
        """Buffer parcialmente cheio retorna fracao correta."""
        pipeline = _mock_pipeline(window_size=10, n_columns=22)
        rt = RealtimeInference(pipeline, window_size=10)

        # Adiciona 3 medicoes de 10
        for _ in range(3):
            rt.update(np.zeros(22, dtype=np.float32))

        assert rt.buffer_fill == pytest.approx(0.3)
        assert rt.is_ready is False

    def test_is_ready_true_when_full(self):
        """Buffer cheio: is_ready retorna True."""
        pipeline = _mock_pipeline(window_size=5, n_columns=22)
        rt = RealtimeInference(pipeline, window_size=5)

        for _ in range(5):
            rt.update(np.zeros(22, dtype=np.float32))

        assert rt.is_ready is True
        assert rt.buffer_fill == pytest.approx(1.0)

    def test_update_returns_none_before_full(self):
        """update() retorna None enquanto buffer nao esta cheio."""
        pipeline = _mock_pipeline(window_size=5, n_columns=22)
        rt = RealtimeInference(pipeline, window_size=5)

        result = rt.update(np.zeros(22, dtype=np.float32))
        assert result is None

    def test_reset_clears_buffer(self):
        """reset() limpa buffer e contador."""
        pipeline = _mock_pipeline(window_size=5, n_columns=22)
        rt = RealtimeInference(pipeline, window_size=5)

        for _ in range(3):
            rt.update(np.zeros(22, dtype=np.float32))

        rt.reset()
        assert rt.is_ready is False
        assert rt.buffer_fill == pytest.approx(0.0)
        assert rt.n_updates == 0

    def test_pipeline_none_raises(self):
        """Pipeline None levanta ValueError."""
        with pytest.raises(ValueError, match="pipeline"):
            RealtimeInference(None, window_size=600)

    def test_window_size_zero_raises(self):
        """Window size <= 0 levanta ValueError."""
        pipeline = _mock_pipeline()
        with pytest.raises(ValueError, match="window_size"):
            RealtimeInference(pipeline, window_size=0)


# ════════════════════════════════════════════════════════════════════════════
# TESTES TF-REQUIRED
# ════════════════════════════════════════════════════════════════════════════


@requires_tf
class TestExportFunctions:
    """Funcoes de export existem e sao callable."""

    def test_export_functions_importable(self):
        """export_saved_model, export_tflite, export_onnx sao importaveis."""
        from geosteering_ai.inference.export import (
            export_saved_model,
            export_tflite,
            export_onnx,
        )
        assert callable(export_saved_model)
        assert callable(export_tflite)
        assert callable(export_onnx)


@requires_tf
class TestInferencePipeline:
    """InferencePipeline — instantiation basica."""

    def test_importable(self):
        """InferencePipeline e importavel."""
        from geosteering_ai.inference.pipeline import InferencePipeline
        assert InferencePipeline is not None
