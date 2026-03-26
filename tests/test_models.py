# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: tests/test_models.py                                              ║
# ║  Bloco: 3 (testes) — Verificacao das 44 arquiteturas + Registry           ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes para geosteering_ai.models (blocks + registry + arquiteturas).

Estrategia de testes:
    - Registry e metadados: CPU-only (sem TF) → sempre executam
    - Blocos e modelos: requerem TF → skipados se nao disponivel
    - Forward pass: (batch=2, 600, n_feat) → (2, 600, 2)

Execucao:
    pytest tests/test_models.py -v --tb=short
    pytest tests/test_models.py -v -k "TestRegistry"  # apenas CPU-safe
"""

from __future__ import annotations

import sys
import importlib
import pytest
from dataclasses import replace

# ── Deteccao de TensorFlow ────────────────────────────────────────────────
try:
    import tensorflow as tf  # noqa: F401
    HAS_TF = True
except ImportError:  # pragma: no cover
    HAS_TF = False

requires_tf = pytest.mark.skipif(not HAS_TF, reason="TensorFlow nao disponivel (CPU-only dev)")

# ── PipelineConfig (sempre disponivel) ────────────────────────────────────
from geosteering_ai.config import PipelineConfig
from geosteering_ai.models.registry import (
    build_model,
    get_model_info,
    is_causal_compatible,
    list_available_models,
    ModelRegistry,
    _CAUSAL_INCOMPATIBLE,
    _FAMILIES,
)


# ════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def baseline_config() -> PipelineConfig:
    """Config baseline: ResNet-18, sem noise, sem GS."""
    return PipelineConfig.baseline()


@pytest.fixture
def realtime_config() -> PipelineConfig:
    """Config realtime: WaveNet causal."""
    return PipelineConfig.realtime()


# ════════════════════════════════════════════════════════════════════════════
# SECAO: REGISTRY — TESTES CPU-SAFE (sem TF)
# ════════════════════════════════════════════════════════════════════════════


class TestRegistryMetadata:
    """Testes de metadados do registry (CPU-safe, sem TF)."""

    def test_total_architectures_44(self):
        """Registry deve ter exatamente 44 arquiteturas."""
        models = list_available_models()
        assert len(models) == 44, (
            f"Esperado 44 arquiteturas, encontrado {len(models)}: {models}"
        )

    def test_cnn_family_7(self):
        """Familia CNN deve ter 7 arquiteturas."""
        cnn = list_available_models("CNN")
        assert len(cnn) == 7, f"Esperado 7 CNN, encontrado {len(cnn)}: {cnn}"

    def test_unet_family_14(self):
        """Familia UNet deve ter 14 variantes."""
        unet = list_available_models("UNet")
        assert len(unet) == 14, f"Esperado 14 UNet, encontrado {len(unet)}: {unet}"

    def test_geosteering_family_5(self):
        """Familia Geosteering deve ter 5 arquiteturas causais."""
        gs = list_available_models("Geosteering")
        assert len(gs) == 5, f"Esperado 5 Geosteering, encontrado {len(gs)}: {gs}"

    def test_transformer_family_6(self):
        """Familia Transformer deve ter 6 variantes."""
        tr = list_available_models("Transformer")
        assert len(tr) == 6, f"Esperado 6 Transformer, encontrado {len(tr)}: {tr}"

    def test_decomposition_family_2(self):
        """Familia Decomposition deve ter 2 arquiteturas."""
        d = list_available_models("Decomposition")
        assert len(d) == 2

    def test_default_resnet18_in_registry(self):
        """ResNet_18 deve estar no registry (default ★)."""
        assert "ResNet_18" in list_available_models()

    def test_all_families_covered(self):
        """Todas as 9 familias esperadas devem existir."""
        expected = {"CNN", "TCN", "RNN", "Hybrid", "UNet",
                    "Transformer", "Decomposition", "Advanced", "Geosteering"}
        actual = set(_FAMILIES.values())
        assert expected == actual, f"Familias inesperadas: {actual - expected}"

    def test_invalid_model_type_raises(self):
        """model_type invalido deve levantar ValueError."""
        with pytest.raises(ValueError, match="nao encontrado"):
            from geosteering_ai.models.registry import _get_build_fn
            _get_build_fn("ModeloInexistente")

    def test_invalid_model_info_raises(self):
        """get_model_info para model invalido deve levantar ValueError."""
        with pytest.raises(ValueError):
            get_model_info("ArquiteturaQueNaoExiste")


class TestCausalCompatibility:
    """Testes de compatibilidade causal (CPU-safe)."""

    def test_resnet18_is_causal_compatible(self):
        """ResNet-18 deve ser causal-compatible."""
        assert is_causal_compatible("ResNet_18")

    def test_bilstm_is_causal_incompatible(self):
        """BiLSTM e CAUSAL_INCOMPATIBLE."""
        assert not is_causal_compatible("BiLSTM")

    def test_wavenet_is_causal_compatible(self):
        """WaveNet e causal nativo."""
        assert is_causal_compatible("WaveNet")

    def test_all_unet_causal_incompatible(self):
        """Todos os U-Nets devem ser CAUSAL_INCOMPATIBLE."""
        for name in list_available_models("UNet"):
            assert not is_causal_compatible(name), f"UNet {name} deveria ser incompativel"

    def test_all_geosteering_causal_compatible(self):
        """Todas as arquiteturas Geosteering devem ser causal-compatible."""
        for name in list_available_models("Geosteering"):
            assert is_causal_compatible(name), f"Geosteering {name} deveria ser compativel"

    def test_causal_incompatible_count_18(self):
        """Deve haver exatamente 18 arquiteturas CAUSAL_INCOMPATIBLE.

        14 UNets + BiLSTM + CNN_BiLSTM_ED + N_BEATS + N_HiTS = 18.
        """
        assert len(_CAUSAL_INCOMPATIBLE) == 18, (
            f"Esperado 18 incompativeis, encontrado {len(_CAUSAL_INCOMPATIBLE)}"
        )

    def test_26_causal_compatible(self):
        """Deve haver 26 arquiteturas causal-compatible (44 - 18)."""
        all_models = list_available_models()
        compat = [m for m in all_models if is_causal_compatible(m)]
        assert len(compat) == 26, f"Esperado 26 compat, encontrado {len(compat)}"


class TestGetModelInfo:
    """Testes de metadados por arquitetura (CPU-safe)."""

    def test_resnet18_info(self):
        """ResNet_18 deve ter tier=1 e familia=CNN."""
        info = get_model_info("ResNet_18")
        assert info["tier"] == 1
        assert info["family"] == "CNN"
        assert info["causal_compatible"] is True
        assert "ResNet" in info["description"]

    def test_wavenet_info(self):
        """WaveNet deve ter familia=Geosteering e causal."""
        info = get_model_info("WaveNet")
        assert info["family"] == "Geosteering"
        assert info["causal_compatible"] is True

    def test_all_models_have_complete_info(self):
        """Todos os 44 modelos devem ter info completa (4 campos)."""
        for name in list_available_models():
            info = get_model_info(name)
            assert "name" in info
            assert "family" in info
            assert "tier" in info
            assert "causal_compatible" in info


class TestModelRegistryClass:
    """Testes da fachada OO ModelRegistry (CPU-safe)."""

    def test_count_property_44(self):
        """registry.count deve retornar 44."""
        registry = ModelRegistry()
        assert registry.count == 44

    def test_available_method(self):
        """registry.available() deve listar todos os modelos."""
        registry = ModelRegistry()
        assert len(registry.available()) == 44

    def test_available_by_family(self):
        """registry.available(family='CNN') deve retornar 7."""
        registry = ModelRegistry()
        assert len(registry.available("CNN")) == 7

    def test_causal_compatible_method(self):
        """registry.causal_compatible() deve delegar para is_causal_compatible()."""
        registry = ModelRegistry()
        assert registry.causal_compatible("TCN") is True
        assert registry.causal_compatible("BiLSTM") is False


# ════════════════════════════════════════════════════════════════════════════
# SECAO: BLOCOS KERAS — REQUEREM TF
# ════════════════════════════════════════════════════════════════════════════


@requires_tf
class TestBlocksBasic:
    """Testes de blocos individuais (forward pass shape)."""

    @pytest.fixture(autouse=True)
    def _make_tensor(self):
        """Tensor de entrada padrao: (2, 600, 5)."""
        import tensorflow as tf
        self.x = tf.random.normal((2, 600, 5))
        self.x_ch32 = tf.random.normal((2, 600, 32))

    def test_residual_block_1d_shape(self):
        """residual_block_1d deve preservar batch e seq_len."""
        from geosteering_ai.models.blocks import residual_block_1d
        y = residual_block_1d(self.x, filters=32)
        assert y.shape == (2, 600, 32)

    def test_residual_block_1d_causal(self):
        """residual_block_1d causal deve ter mesma shape."""
        from geosteering_ai.models.blocks import residual_block_1d
        y = residual_block_1d(self.x, filters=32, causal=True)
        assert y.shape == (2, 600, 32)

    def test_bottleneck_block_1d_shape(self):
        """bottleneck_block_1d output = filters * expansion."""
        from geosteering_ai.models.blocks import bottleneck_block_1d
        y = bottleneck_block_1d(self.x, filters=16, expansion=4)
        assert y.shape == (2, 600, 64)

    def test_conv_next_block_shape(self):
        """conv_next_block deve retornar (2, 600, filters)."""
        from geosteering_ai.models.blocks import conv_next_block
        y = conv_next_block(self.x, filters=32)
        assert y.shape == (2, 600, 32)

    def test_se_block_shape_preserved(self):
        """se_block deve preservar shape."""
        from geosteering_ai.models.blocks import se_block
        y = se_block(self.x_ch32, reduction=8)
        assert y.shape == self.x_ch32.shape

    def test_dilated_causal_block_shape(self):
        """dilated_causal_block deve retornar (2, 600, filters)."""
        from geosteering_ai.models.blocks import dilated_causal_block
        y = dilated_causal_block(self.x, filters=32, dilation_rate=2)
        assert y.shape == (2, 600, 32)

    def test_inception_module_output_4x(self):
        """inception_module deve retornar 4 * filters canais."""
        from geosteering_ai.models.blocks import inception_module
        y = inception_module(self.x, filters=16)
        assert y.shape == (2, 600, 64)  # 4 * 16

    def test_series_decomp_shapes(self):
        """series_decomp_block deve retornar dois tensores de mesma shape."""
        from geosteering_ai.models.blocks import series_decomp_block
        seasonal, trend = series_decomp_block(self.x_ch32)
        assert seasonal.shape == self.x_ch32.shape
        assert trend.shape == self.x_ch32.shape

    def test_output_projection_shape(self):
        """output_projection deve retornar (2, 600, out_ch)."""
        from geosteering_ai.models.blocks import output_projection
        y = output_projection(self.x_ch32, output_channels=2)
        assert y.shape == (2, 600, 2)

    def test_feedforward_block_shape(self):
        """feedforward_block deve preservar ultima dimensao."""
        from geosteering_ai.models.blocks import feedforward_block
        import tensorflow as tf
        x = tf.random.normal((2, 600, 64))
        y = feedforward_block(x, ff_dim=128)
        assert y.shape == (2, 600, 64)

    def test_inception_time_block_shape(self):
        """inception_time_block deve retornar (2, 600, 4*filters)."""
        from geosteering_ai.models.blocks import inception_time_block
        y = inception_time_block(self.x, filters=16)
        assert y.shape == (2, 600, 64)

    def test_transformer_encoder_block_shape(self):
        """transformer_encoder_block deve preservar shape."""
        from geosteering_ai.models.blocks import transformer_encoder_block
        import tensorflow as tf
        x = tf.random.normal((2, 600, 64))
        y = transformer_encoder_block(x, num_heads=4, key_dim=16, ff_dim=128)
        assert y.shape == x.shape

    def test_gated_activation_block_shape(self):
        """gated_activation_block deve retornar (2, 600, filters)."""
        from geosteering_ai.models.blocks import gated_activation_block
        y = gated_activation_block(self.x_ch32, filters=16)
        assert y.shape == (2, 600, 16)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: ARQUITETURAS — FORWARD PASS (requerem TF)
# ════════════════════════════════════════════════════════════════════════════


def _forward_pass(config: PipelineConfig) -> None:
    """Helper: constroi modelo e executa forward pass."""
    import tensorflow as tf
    import numpy as np

    model = build_model(config)
    x = np.random.randn(2, config.sequence_length, config.n_features).astype("float32")
    y = model(x, training=False)
    assert y.shape == (2, config.sequence_length, config.output_channels), (
        f"Shape esperada (2, {config.sequence_length}, {config.output_channels}), "
        f"obtida {y.shape}"
    )


def _make_config(model_type: str, **kwargs) -> PipelineConfig:
    """Config minima para testar um model_type."""
    base = PipelineConfig.baseline()
    # Para frozen dataclass, usar dataclasses.replace
    import dataclasses
    return dataclasses.replace(base, model_type=model_type, **kwargs)


@requires_tf
class TestCNNForward:
    """Forward pass das 7 arquiteturas CNN."""

    def test_resnet18(self):
        _forward_pass(_make_config("ResNet_18"))

    def test_resnet34(self):
        _forward_pass(_make_config("ResNet_34"))

    def test_resnet50(self):
        _forward_pass(_make_config("ResNet_50"))

    def test_convnext(self):
        _forward_pass(_make_config("ConvNeXt"))

    def test_inceptionnet(self):
        _forward_pass(_make_config("InceptionNet"))

    def test_inceptiontime(self):
        _forward_pass(_make_config("InceptionTime"))

    def test_cnn1d(self):
        _forward_pass(_make_config("CNN_1D"))


@requires_tf
class TestTCNForward:
    """Forward pass das 2 arquiteturas TCN."""

    def test_tcn(self):
        _forward_pass(_make_config("TCN"))

    def test_tcn_advanced(self):
        _forward_pass(_make_config("TCN_Advanced"))


@requires_tf
class TestRNNForward:
    """Forward pass das 2 arquiteturas RNN."""

    def test_lstm(self):
        _forward_pass(_make_config("LSTM"))

    def test_bilstm(self):
        _forward_pass(_make_config("BiLSTM"))


@requires_tf
class TestHybridForward:
    """Forward pass das 2 arquiteturas Hybrid."""

    def test_cnn_lstm(self):
        _forward_pass(_make_config("CNN_LSTM"))

    def test_cnn_bilstm_ed(self):
        _forward_pass(_make_config("CNN_BiLSTM_ED"))


@requires_tf
class TestUNetForward:
    """Forward pass das 14 variantes U-Net."""

    def test_unet_base(self):
        _forward_pass(_make_config("UNet_Base"))

    def test_unet_attention(self):
        _forward_pass(_make_config("UNet_Attention"))

    def test_unet_resnet18(self):
        _forward_pass(_make_config("UNet_ResNet18"))

    def test_unet_attention_resnet18(self):
        _forward_pass(_make_config("UNet_Attention_ResNet18"))

    def test_unet_resnet34(self):
        _forward_pass(_make_config("UNet_ResNet34"))

    def test_unet_convnext(self):
        _forward_pass(_make_config("UNet_ConvNeXt"))

    def test_unet_efficientnet(self):
        _forward_pass(_make_config("UNet_EfficientNet"))


@requires_tf
class TestTransformerForward:
    """Forward pass das 6 arquiteturas Transformer."""

    def test_transformer(self):
        _forward_pass(_make_config("Transformer"))

    def test_simple_tft(self):
        _forward_pass(_make_config("Simple_TFT"))

    def test_tft(self):
        _forward_pass(_make_config("TFT"))

    def test_patchtst(self):
        _forward_pass(_make_config("PatchTST"))

    def test_autoformer(self):
        _forward_pass(_make_config("Autoformer"))

    def test_itransformer(self):
        _forward_pass(_make_config("iTransformer"))


@requires_tf
class TestDecompositionForward:
    """Forward pass das 2 arquiteturas de decomposicao."""

    def test_nbeats(self):
        _forward_pass(_make_config("N_BEATS"))

    def test_nhits(self):
        _forward_pass(_make_config("N_HiTS"))


@requires_tf
class TestAdvancedForward:
    """Forward pass das 4 arquiteturas avancadas."""

    def test_dnn(self):
        _forward_pass(_make_config("DNN"))

    def test_geophysical_attention(self):
        _forward_pass(_make_config("Geophysical_Attention"))

    def test_deeponet(self):
        _forward_pass(_make_config("DeepONet"))

    def test_fno(self):
        _forward_pass(_make_config("FNO"))


@requires_tf
class TestGeosteering:
    """Forward pass das 5 arquiteturas geosteering causais."""

    def test_wavenet(self):
        _forward_pass(_make_config("WaveNet"))

    def test_causal_transformer(self):
        _forward_pass(_make_config("Causal_Transformer", inference_mode="realtime"))

    def test_informer(self):
        _forward_pass(_make_config("Informer"))

    def test_mamba_s4(self):
        _forward_pass(_make_config("Mamba_S4"))

    def test_encoder_forecaster(self):
        _forward_pass(_make_config("Encoder_Forecaster"))


@requires_tf
class TestBuildModelValidation:
    """Testes de validacao no build_model."""

    def test_invalid_model_type_raises(self):
        """model_type invalido deve levantar ValueError."""
        import dataclasses
        cfg = dataclasses.replace(PipelineConfig.baseline(), model_type="XYZ_Invalid")
        # Precisa passar pela validacao do config (que nao valida model_type)
        # build_model valida internamente
        with pytest.raises(ValueError):
            build_model(cfg)

    def test_registry_class_build(self):
        """ModelRegistry.build() deve construir ResNet-18."""
        cfg = PipelineConfig.baseline()
        registry = ModelRegistry()
        model = registry.build(cfg)
        assert model.name == "ResNet_18"

    def test_resnet18_output_shape_default(self):
        """ResNet-18 deve produzir (2, 600, 2) por default."""
        import numpy as np
        cfg = PipelineConfig.baseline()
        model = build_model(cfg)
        x = np.random.randn(2, 600, cfg.n_features).astype("float32")
        y = model(x, training=False)
        assert y.shape == (2, 600, 2)

    def test_output_channels_4(self):
        """output_channels=4 deve produzir (2, 600, 4)."""
        import numpy as np
        import dataclasses
        cfg = dataclasses.replace(PipelineConfig.baseline(), output_channels=4)
        model = build_model(cfg)
        x = np.random.randn(2, 600, cfg.n_features).astype("float32")
        y = model(x, training=False)
        assert y.shape == (2, 600, 4)


@requires_tf
class TestInitReExports:
    """Verifica que __init__.py re-exporta corretamente."""

    def test_build_model_importable(self):
        """build_model deve ser importavel de geosteering_ai.models."""
        from geosteering_ai.models import build_model as bm
        assert callable(bm)

    def test_registry_importable(self):
        """ModelRegistry deve ser importavel de geosteering_ai.models."""
        from geosteering_ai.models import ModelRegistry as MR
        assert MR().count == 44

    def test_blocks_importable(self):
        """Blocos utilitarios devem ser importaveis diretamente."""
        from geosteering_ai.models import residual_block_1d, output_projection
        assert callable(residual_block_1d)
        assert callable(output_projection)
