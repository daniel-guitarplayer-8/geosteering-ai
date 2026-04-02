"""Testes para data/surrogate_data.py e models/surrogate.py.

Cobertura:
    - TestGetComponentColumnIndices: mapeamento componente → indices
    - TestExtractSurrogatePairs: extracao de pares, shapes, decoupling
    - TestComputeComponentWeights: pesos diagonal vs cruzada
    - TestSurrogateDataset: dataclass, metadados, channel_names
    - TestConfigSurrogateComponents: validacao de componentes no config
    - TestBuildSurrogate: construcao do modelo TCN, shapes, parametros
"""

import numpy as np
import pytest

from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.surrogate_data import (
    SurrogateDataset,
    compute_component_weights,
    extract_surrogate_pairs,
    get_component_column_indices,
)

# ── TF disponivel? ──────────────────────────────────────────────────
try:
    import tensorflow as tf

    _HAS_TF = True
except ImportError:
    _HAS_TF = False

skip_no_tf = pytest.mark.skipif(not _HAS_TF, reason="TensorFlow nao disponivel")

# ── Helpers ──────────────────────────────────────────────────────────
B, N = 5, 50  # batch (modelos), seq_len


def _make_synthetic_dat(n_models=B, seq_len=N):
    """Cria array sintetico 22-colunas simulando .dat Fortran."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_models, seq_len, 22)).astype(np.float32)
    # Coluna 0: meds (indices)
    data[:, :, 0] = np.arange(seq_len)[None, :]
    # Coluna 1: zobs (profundidade)
    data[:, :, 1] = np.linspace(0, 100, seq_len)[None, :]
    # Colunas 2, 3: rho_h, rho_v (positivos, Ohm.m)
    data[:, :, 2] = rng.uniform(1.0, 1000.0, (n_models, seq_len)).astype(np.float32)
    data[:, :, 3] = data[:, :, 2] * rng.uniform(1.0, 5.0, (n_models, seq_len)).astype(
        np.float32
    )
    return data


# ════════════════════════════════════════════════════════════════════════
# TESTES: GET_COMPONENT_COLUMN_INDICES
# ════════════════════════════════════════════════════════════════════════


class TestGetComponentColumnIndices:
    """Testes para get_component_column_indices()."""

    def test_modo_a_default(self):
        """Modo A: XX + ZZ → cols 4,5,20,21."""
        indices = get_component_column_indices(["XX", "ZZ"])
        assert indices == [4, 5, 20, 21]

    def test_modo_b_geosteering(self):
        """Modo B: XX + ZZ + XZ + ZX → 8 indices."""
        indices = get_component_column_indices(["XX", "ZZ", "XZ", "ZX"])
        assert indices == [4, 5, 20, 21, 8, 9, 16, 17]
        assert len(indices) == 8

    def test_modo_c_tensor_completo(self):
        """Modo C: 9 componentes → 18 indices."""
        all_comps = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
        indices = get_component_column_indices(all_comps)
        assert len(indices) == 18

    def test_componente_invalida_raises(self):
        """Componente invalida gera ValueError."""
        with pytest.raises(ValueError, match="invalida"):
            get_component_column_indices(["XX", "INVALID"])

    def test_single_component(self):
        """Uma unica componente → 2 indices (Re + Im)."""
        indices = get_component_column_indices(["XZ"])
        assert indices == [8, 9]

    def test_order_preserved(self):
        """Ordem das componentes na entrada eh preservada nos indices."""
        indices_ab = get_component_column_indices(["ZZ", "XX"])
        assert indices_ab == [20, 21, 4, 5]  # ZZ primeiro, depois XX


# ════════════════════════════════════════════════════════════════════════
# TESTES: EXTRACT_SURROGATE_PAIRS
# ════════════════════════════════════════════════════════════════════════


class TestExtractSurrogatePairs:
    """Testes para extract_surrogate_pairs()."""

    def test_shape_modo_a(self):
        """Modo A: x_rho (B,N,2), y_em (B,N,4)."""
        data = _make_synthetic_dat()
        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        ds = extract_surrogate_pairs(data, config)
        assert ds.x_rho.shape == (B, N, 2)
        assert ds.y_em.shape == (B, N, 4)
        assert ds.n_channels == 4

    def test_shape_modo_b(self):
        """Modo B: x_rho (B,N,2), y_em (B,N,8)."""
        data = _make_synthetic_dat()
        config = PipelineConfig(surrogate_output_components=["XX", "ZZ", "XZ", "ZX"])
        ds = extract_surrogate_pairs(data, config)
        assert ds.x_rho.shape == (B, N, 2)
        assert ds.y_em.shape == (B, N, 8)
        assert ds.n_channels == 8

    def test_rho_in_log10_scale(self):
        """x_rho deve estar em escala log10, nao linear."""
        data = _make_synthetic_dat()
        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        ds = extract_surrogate_pairs(data, config)
        # rho original em [1, 5000] → log10 em [0, ~3.7]
        assert ds.x_rho.min() >= -2.0  # clamp inferior
        assert ds.x_rho.max() <= 5.0  # clamp superior

    def test_channel_names(self):
        """channel_names devem corresponder as componentes."""
        data = _make_synthetic_dat()
        config = PipelineConfig(surrogate_output_components=["XX", "ZZ", "XZ"])
        ds = extract_surrogate_pairs(data, config)
        expected = [
            "Re(Hxx)",
            "Im(Hxx)",
            "Re(Hzz)",
            "Im(Hzz)",
            "Re(Hxz)",
            "Im(Hxz)",
        ]
        assert ds.channel_names == expected

    def test_decoupling_applied(self):
        """Com decoupling, Re(Hxx) e Re(Hzz) devem ser diferentes do raw."""
        data = _make_synthetic_dat()
        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        ds_decoup = extract_surrogate_pairs(data, config, apply_decoup=True)
        ds_raw = extract_surrogate_pairs(data, config, apply_decoup=False)
        # Re(Hxx) deve diferir (decoupling subtrai ACp)
        assert not np.allclose(ds_decoup.y_em[:, :, 0], ds_raw.y_em[:, :, 0])
        # Im(Hxx) nao muda (decoupling so afeta Re)
        assert np.allclose(ds_decoup.y_em[:, :, 1], ds_raw.y_em[:, :, 1])

    def test_cross_components_no_decoupling(self):
        """Componentes cruzadas (XZ) nao recebem decoupling."""
        data = _make_synthetic_dat()
        config = PipelineConfig(surrogate_output_components=["XZ"])
        ds_decoup = extract_surrogate_pairs(data, config, apply_decoup=True)
        ds_raw = extract_surrogate_pairs(data, config, apply_decoup=False)
        # XZ nao tem campo primario — decoupling nao altera
        assert np.allclose(ds_decoup.y_em, ds_raw.y_em)

    def test_invalid_ndim_raises(self):
        """Array 2D deve gerar ValueError."""
        data_2d = np.random.randn(100, 22).astype(np.float32)
        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        with pytest.raises(ValueError, match="3D"):
            extract_surrogate_pairs(data_2d, config)

    def test_insufficient_columns_raises(self):
        """Array com < 22 colunas deve gerar ValueError."""
        data_short = np.random.randn(5, 50, 10).astype(np.float32)
        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        with pytest.raises(ValueError, match="22 colunas"):
            extract_surrogate_pairs(data_short, config)


# ════════════════════════════════════════════════════════════════════════
# TESTES: COMPUTE_COMPONENT_WEIGHTS
# ════════════════════════════════════════════════════════════════════════


class TestComputeComponentWeights:
    """Testes para compute_component_weights()."""

    def test_modo_a_all_diagonal(self):
        """Modo A (XX, ZZ): todos diagonais → peso = w_diagonal."""
        config = PipelineConfig(
            surrogate_output_components=["XX", "ZZ"],
            surrogate_weight_diagonal=1.0,
            surrogate_weight_cross=5.0,
        )
        w = compute_component_weights(config)
        assert w.shape == (4,)
        assert np.all(w == 1.0)  # XX e ZZ sao diagonais

    def test_modo_b_mixed(self):
        """Modo B (XX, ZZ, XZ, ZX): diagonais + cruzadas."""
        config = PipelineConfig(
            surrogate_output_components=["XX", "ZZ", "XZ", "ZX"],
            surrogate_weight_diagonal=1.0,
            surrogate_weight_cross=5.0,
        )
        w = compute_component_weights(config)
        assert w.shape == (8,)
        # XX: w=1.0, ZZ: w=1.0, XZ: w=5.0, ZX: w=5.0
        assert w[0] == 1.0  # Re(Hxx)
        assert w[1] == 1.0  # Im(Hxx)
        assert w[2] == 1.0  # Re(Hzz)
        assert w[3] == 1.0  # Im(Hzz)
        assert w[4] == 5.0  # Re(Hxz) — cruzada
        assert w[5] == 5.0  # Im(Hxz)
        assert w[6] == 5.0  # Re(Hzx) — cruzada
        assert w[7] == 5.0  # Im(Hzx)

    def test_all_cross(self):
        """Apenas cruzadas → todos com peso w_cross."""
        config = PipelineConfig(
            surrogate_output_components=["XZ", "ZX"],
            surrogate_weight_diagonal=1.0,
            surrogate_weight_cross=3.0,
        )
        w = compute_component_weights(config)
        assert np.all(w == 3.0)


# ════════════════════════════════════════════════════════════════════════
# TESTES: CONFIG — SURROGATE OUTPUT COMPONENTS VALIDATION
# ════════════════════════════════════════════════════════════════════════


class TestConfigSurrogateComponents:
    """Testes para validacao de surrogate_output_components no config."""

    def test_default_is_xx_zz(self):
        """Default eh ['XX', 'ZZ'] (Modo A)."""
        config = PipelineConfig()
        assert config.surrogate_output_components == ["XX", "ZZ"]

    def test_modo_b_accepted(self):
        """Modo B com 4 componentes eh aceito."""
        config = PipelineConfig(surrogate_output_components=["XX", "ZZ", "XZ", "ZX"])
        assert len(config.surrogate_output_components) == 4

    def test_all_9_components_accepted(self):
        """Modo C com 9 componentes eh aceito."""
        all_comps = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
        config = PipelineConfig(surrogate_output_components=all_comps)
        assert len(config.surrogate_output_components) == 9

    def test_invalid_component_raises(self):
        """Componente invalida gera AssertionError."""
        with pytest.raises(AssertionError, match="surrogate_output_components"):
            PipelineConfig(surrogate_output_components=["XX", "INVALID"])

    def test_empty_list_raises(self):
        """Lista vazia gera AssertionError."""
        with pytest.raises(AssertionError, match="pelo menos 1"):
            PipelineConfig(surrogate_output_components=[])

    def test_weight_positive_required(self):
        """Pesos devem ser positivos."""
        with pytest.raises(AssertionError, match="surrogate_weight_diagonal"):
            PipelineConfig(surrogate_weight_diagonal=-1.0)

    def test_duplicate_components_rejected(self):
        """Componentes duplicadas geram AssertionError."""
        with pytest.raises(AssertionError, match="duplicatas"):
            PipelineConfig(surrogate_output_components=["XX", "XX"])


# ════════════════════════════════════════════════════════════════════════
# TESTES: BUILD_SURROGATE (modelo TCN)
# ════════════════════════════════════════════════════════════════════════


@skip_no_tf
class TestBuildSurrogate:
    """Testes para build_surrogate() — construcao do SurrogateNet TCN."""

    def test_output_shape_modo_a(self):
        """Modo A: saida (B, N, 4) para input (B, N, 2)."""
        from geosteering_ai.models.surrogate import build_surrogate

        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        model = build_surrogate(config)
        assert model.output_shape == (None, None, 4)

    def test_output_shape_modo_b(self):
        """Modo B: saida (B, N, 8) para input (B, N, 2)."""
        from geosteering_ai.models.surrogate import build_surrogate

        config = PipelineConfig(surrogate_output_components=["XX", "ZZ", "XZ", "ZX"])
        model = build_surrogate(config)
        assert model.output_shape == (None, None, 8)

    def test_forward_pass(self):
        """Forward pass funciona sem erro com dados sinteticos."""
        from geosteering_ai.models.surrogate import build_surrogate

        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        model = build_surrogate(config, dropout_rate=0.0)
        x = tf.random.normal((2, 50, 2))
        y = model(x, training=False)
        assert y.shape == (2, 50, 4)

    def test_input_shape(self):
        """Input shape deve ser (None, None, 2)."""
        from geosteering_ai.models.surrogate import build_surrogate

        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        model = build_surrogate(config)
        assert model.input_shape == (None, None, 2)

    def test_has_parameters(self):
        """Modelo deve ter parametros treinaveis > 0."""
        from geosteering_ai.models.surrogate import build_surrogate

        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        model = build_surrogate(config)
        assert model.count_params() > 0

    def test_gradient_flows(self):
        """Gradientes fluem de loss ate input via GradientTape."""
        from geosteering_ai.models.surrogate import build_surrogate

        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        model = build_surrogate(config, dropout_rate=0.0)
        x = tf.random.normal((2, 50, 2))
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x, training=True)
            loss = tf.reduce_mean(y**2)
        grad = tape.gradient(loss, x)
        assert grad is not None
        assert not tf.reduce_all(grad == 0).numpy()

    def test_variable_sequence_length(self):
        """Modelo aceita sequencias de comprimento variavel."""
        from geosteering_ai.models.surrogate import build_surrogate

        config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        model = build_surrogate(config, dropout_rate=0.0)
        for seq_len in [10, 50, 200, 600]:
            x = tf.random.normal((1, seq_len, 2))
            y = model(x, training=False)
            assert y.shape == (1, seq_len, 4)
