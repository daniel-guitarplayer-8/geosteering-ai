# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_visualization.py                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — Visualização do Simulador Python (Sprint 2.8)    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest + matplotlib (Agg backend)                          ║
# ║  Dependências: pytest, matplotlib, numpy                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Testes de fumaça (smoke tests) para o subpacote `visualization`.     ║
# ║    Não validam a estética do gráfico, mas garantem que:                 ║
# ║      1. As funções executam sem erro com inputs típicos                 ║
# ║      2. O Figure retornado tem a estrutura esperada (número de axes,   ║
# ║         títulos, y-axis invertido)                                      ║
# ║      3. Conversões de dados (real/imag) não perdem informação          ║
# ║                                                                           ║
# ║  BACKEND DO MATPLOTLIB                                                    ║
# ║    Agg (não interativo) — selecionado automaticamente pelo pytest       ║
# ║    via `matplotlib.use('Agg')` no início dos testes para evitar          ║
# ║    abrir janelas em CI e ambientes headless.                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de fumaça para `geosteering_ai.simulation.visualization`."""
from __future__ import annotations

# Força backend não-interativo ANTES de importar pyplot (evita janelas).
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from geosteering_ai.simulation import SimulationConfig, simulate  # noqa: E402
from geosteering_ai.simulation.visualization import (  # noqa: E402
    plot_benchmark_comparison,
    plot_resistivity_profile,
    plot_tensor_profile,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def simple_result():
    """Resultado de simulação com 3 camadas TIV (reusado entre testes)."""
    cfg = SimulationConfig(parallel=False)
    return simulate(
        rho_h=np.array([1.0, 100.0, 1.0]),
        rho_v=np.array([1.0, 200.0, 1.0]),
        esp=np.array([5.0]),
        positions_z=np.linspace(-2.0, 7.0, 30),
        cfg=cfg,
    )


# ──────────────────────────────────────────────────────────────────────────────
# plot_resistivity_profile
# ──────────────────────────────────────────────────────────────────────────────


class TestResistivityProfile:
    """Testa o painel standalone de perfil de resistividade."""

    def test_returns_fig_and_ax(self, simple_result) -> None:
        fig, ax = plot_resistivity_profile(simple_result)
        assert fig is not None
        assert ax is not None

    def test_y_axis_inverted(self, simple_result) -> None:
        """Convenção geológica: profundidade cresce para baixo."""
        fig, ax = plot_resistivity_profile(simple_result)
        ymin, ymax = ax.get_ylim()
        assert ymin > ymax, "Eixo y deve estar invertido (inverted_yaxis)"

    def test_has_two_curves(self, simple_result) -> None:
        """Deve ter 2 linhas: ρₕ (steelblue) e ρᵥ (darkorange tracejada)."""
        fig, ax = plot_resistivity_profile(simple_result)
        lines = ax.get_lines()
        assert len(lines) >= 2
        # A ordem é: ρₕ primeiro (sólida), ρᵥ depois (tracejada)
        assert lines[0].get_linestyle() == "-"
        assert lines[1].get_linestyle() == "--"

    def test_uses_existing_ax(self, simple_result) -> None:
        """Quando `ax` é fornecido, fig retornado é None."""
        import matplotlib.pyplot as plt

        fig_outer, ax_outer = plt.subplots()
        fig_inner, ax_inner = plot_resistivity_profile(simple_result, ax=ax_outer)
        assert fig_inner is None
        assert ax_inner is ax_outer


# ──────────────────────────────────────────────────────────────────────────────
# plot_tensor_profile
# ──────────────────────────────────────────────────────────────────────────────


class TestTensorProfile:
    """Testa a figura completa GridSpec(3, 7)."""

    def test_returns_figure(self, simple_result) -> None:
        fig = plot_tensor_profile(simple_result)
        assert fig is not None
        # Fechamos explicitamente para liberar memória do backend Agg
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_expected_number_of_axes(self, simple_result) -> None:
        """Layout deve ter 1 (ρ panel) + 6*3 (tensor Re/Im) = 19 axes."""
        fig = plot_tensor_profile(simple_result)
        n_axes = len(fig.axes)
        # 1 coluna ρ + 6 colunas × 3 linhas = 19 axes
        assert n_axes == 19, f"Esperado 19 axes; obtido {n_axes}"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_suptitle_present(self, simple_result) -> None:
        fig = plot_tensor_profile(simple_result, title="Meu Teste")
        suptitle_text = fig._suptitle.get_text()
        assert "Meu Teste" in suptitle_text
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_freq_idx_out_of_range_raises(self, simple_result) -> None:
        """freq_idx inválido deve levantar ValueError explicativo."""
        nf = simple_result.H_tensor.shape[1]
        with pytest.raises(ValueError, match="freq_idx"):
            plot_tensor_profile(simple_result, freq_idx=nf + 10)


# ──────────────────────────────────────────────────────────────────────────────
# plot_benchmark_comparison
# ──────────────────────────────────────────────────────────────────────────────


class TestBenchmarkComparison:
    """Testa o plot comparativo de múltiplos BenchmarkResult."""

    def _mock_result(self, name: str, tp: float, t_mean: float = 0.1) -> object:
        """Constrói um BenchmarkResult mínimo mockado (duck typing)."""

        class _MockResult:
            profile_name = name
            throughput_models_per_hour = tp
            mean_time_seconds = t_mean
            time_std_seconds = 0.01

        return _MockResult()

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="vazio"):
            plot_benchmark_comparison([])

    def test_single_result_works(self) -> None:
        results = [self._mock_result("small", 50000.0)]
        fig = plot_benchmark_comparison(results)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multiple_results_bar_chart(self) -> None:
        """Com 3 benchmarks, deve haver 3 barras no painel 1."""
        results = [
            self._mock_result("small", 66000.0),
            self._mock_result("medium", 15000.0),
            self._mock_result("large", 3600.0),
        ]
        fig = plot_benchmark_comparison(results)
        axes = fig.axes
        # 2 painéis (throughput + speedup)
        assert len(axes) == 2
        ax_tp = axes[0]
        # O 3o container de patches é o bar plot (rects)
        rects = [p for p in ax_tp.patches if hasattr(p, "get_height")]
        assert len(rects) >= 3
        import matplotlib.pyplot as plt

        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 3.3.2 — 4 novos plots LWD/PINN industriais
# ──────────────────────────────────────────────────────────────────────────────

from geosteering_ai.simulation.visualization import (  # noqa: E402
    plot_anisotropy_ratio_sensitivity,
    plot_apparent_resistivity_curves,
    plot_geosignal_response_vs_dip,
    plot_pinn_loss_decomposition,
)


class TestApparentResistivityCurves:
    """Smoke tests — plot_apparent_resistivity_curves (padrão LWD industrial)."""

    def test_returns_fig_with_2_panels(self, simple_result) -> None:
        fig = plot_apparent_resistivity_curves(simple_result)
        assert fig is not None
        # 2 painéis (perfil verdadeiro + ρ_a aparente)
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_y_axis_inverted_geology_convention(self, simple_result) -> None:
        """Eixo y deve estar invertido (profundidade cresce para baixo)."""
        fig = plot_apparent_resistivity_curves(simple_result)
        ymin, ymax = fig.axes[0].get_ylim()
        assert ymin > ymax
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_title_pt_br_accented(self, simple_result) -> None:
        """Título deve estar em PT-BR com acentuação."""
        fig = plot_apparent_resistivity_curves(simple_result)
        title_text = fig._suptitle.get_text()
        assert "aparente" in title_text
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_no_nan_in_output(self, simple_result) -> None:
        """Nenhum NaN deve ser produzido nos dados plotados."""
        fig = plot_apparent_resistivity_curves(
            simple_result, components=("Hzz",), freq_indices=[0]
        )
        ax_app = fig.axes[1]
        for line in ax_app.get_lines():
            xdata = line.get_xdata()
            assert not np.any(np.isnan(xdata))
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestGeosignalResponseVsDip:
    """Smoke tests — plot_geosignal_response_vs_dip (LWD directional)."""

    @pytest.fixture
    def results_by_dip(self):
        """Gera 5 simulações em dips distintos (reusa layer simples)."""
        cfg = SimulationConfig(parallel=False)
        dips_deg = [0.0, 30.0, 45.0, 60.0, 90.0]
        out = {}
        for dip in dips_deg:
            res = simulate(
                rho_h=np.array([1.0, 100.0, 1.0]),
                rho_v=np.array([1.0, 200.0, 1.0]),
                esp=np.array([5.0]),
                positions_z=np.linspace(-2.0, 7.0, 20),
                dip_deg=dip,
                cfg=cfg,
            )
            out[dip] = res
        return out

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="vazio"):
            plot_geosignal_response_vs_dip({})

    def test_returns_2x2_panels(self, results_by_dip) -> None:
        fig = plot_geosignal_response_vs_dip(results_by_dip)
        assert fig is not None
        # 2×2 painéis (USD/UAD/UHR/UHA)
        assert len(fig.axes) == 4
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_titles_contain_gs_names(self, results_by_dip) -> None:
        fig = plot_geosignal_response_vs_dip(results_by_dip)
        titles = [ax.get_title() for ax in fig.axes]
        assert any("USD" in t for t in titles)
        assert any("UAD" in t for t in titles)
        assert any("UHR" in t for t in titles)
        assert any("UHA" in t for t in titles)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestAnisotropyRatioSensitivity:
    """Smoke tests — plot_anisotropy_ratio_sensitivity (∂H/∂λ)."""

    # Perfil geológico do `simple_result` fixture — caller deve fornecê-lo
    # explicitamente (Sprint 3.3.2: não pode ser reconstruído de forma
    # robusta a partir de rho_h_at_obs).
    _RHO_H = np.array([1.0, 100.0, 1.0])
    _RHO_V = np.array([1.0, 200.0, 1.0])
    _ESP = np.array([5.0])

    def test_returns_fig_with_2_panels(self, simple_result) -> None:
        """Com 5 lambdas sweep, figura deve ter painel ρ_h + heatmap."""
        lambdas = np.array([0.7, 0.9, 1.0, 1.3, 1.6])
        fig = plot_anisotropy_ratio_sensitivity(
            simple_result,
            self._RHO_H,
            self._RHO_V,
            self._ESP,
            lambdas=lambdas,
            component="Hzz",
        )
        assert fig is not None
        # 2 painéis + colorbar axes
        assert len(fig.axes) >= 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_title_mentions_lambda(self, simple_result) -> None:
        lambdas = np.array([0.8, 1.0, 1.2])
        fig = plot_anisotropy_ratio_sensitivity(
            simple_result, self._RHO_H, self._RHO_V, self._ESP, lambdas=lambdas
        )
        title_text = fig._suptitle.get_text()
        assert "lambda" in title_text.lower() or "λ" in title_text
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_shape_mismatch_raises(self, simple_result) -> None:
        """rho_h.shape != rho_v.shape → ValueError."""
        with pytest.raises(ValueError, match="shape"):
            plot_anisotropy_ratio_sensitivity(
                simple_result,
                np.array([1.0, 100.0]),  # shape mismatch
                np.array([1.0, 200.0, 1.0]),
                np.array([5.0]),
                lambdas=np.array([1.0, 1.2]),
            )


class TestPinnLossDecomposition:
    """Smoke tests — plot_pinn_loss_decomposition."""

    @pytest.fixture
    def synthetic_loss_history(self):
        epochs = np.arange(50)
        return {
            "epochs": epochs,
            "loss_data": np.exp(-0.08 * epochs) + 1e-4,
            "loss_physics": 0.5 * np.exp(-0.05 * epochs) + 1e-5,
            "loss_continuity": 0.15 * np.exp(-0.03 * epochs) + 1e-6,
        }

    def test_returns_fig_with_2_panels(self, synthetic_loss_history) -> None:
        fig = plot_pinn_loss_decomposition(synthetic_loss_history)
        assert fig is not None
        # 2 painéis (curvas + razões relativas)
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_missing_epochs_raises(self) -> None:
        with pytest.raises(KeyError, match="epochs"):
            plot_pinn_loss_decomposition({"loss_data": [1.0]})

    def test_missing_component_raises(self, synthetic_loss_history) -> None:
        history = {k: v for k, v in synthetic_loss_history.items() if k != "loss_physics"}
        with pytest.raises(KeyError, match="loss_physics"):
            plot_pinn_loss_decomposition(history)

    def test_ratio_panel_values_bounded(self, synthetic_loss_history) -> None:
        """Razões L_i/L_total devem estar em [0, 1]."""
        fig = plot_pinn_loss_decomposition(synthetic_loss_history)
        ax_ratio = fig.axes[1]
        ymin, ymax = ax_ratio.get_ylim()
        assert ymin >= -0.01 and ymax <= 1.1
        import matplotlib.pyplot as plt

        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 3.3.3+ — 6 plots curados (a/b/c/d)
# ──────────────────────────────────────────────────────────────────────────────

from geosteering_ai.simulation.visualization import (  # noqa: E402
    plot_backend_comparison_heatmap,
    plot_geometric_factor_sensitivity,
    plot_induction_number_heatmap,
    plot_inference_latency_distribution,
    plot_memory_usage_vs_profile_size,
    plot_multi_frequency_hodograph,
)


class TestInductionNumberHeatmap:
    """Categoria (a) — física: heatmap do número de indução B = ωμ₀σL²."""

    def test_returns_figure(self) -> None:
        fig = plot_induction_number_heatmap(spacing_m=1.0)
        assert fig is not None
        assert len(fig.axes) >= 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_log_scales(self) -> None:
        fig = plot_induction_number_heatmap()
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestMultiFrequencyHodograph:
    """Categoria (c) — geofísica: hodógrafo Re×Im multi-freq."""

    def test_returns_figure(self, simple_result) -> None:
        fig = plot_multi_frequency_hodograph(simple_result, component="Hzz")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_unknown_component_raises(self, simple_result) -> None:
        with pytest.raises(ValueError, match="Componente desconhecido"):
            plot_multi_frequency_hodograph(simple_result, component="Hbogus")


class TestGeometricFactorSensitivity:
    """Categoria (c) — geofísica: G(z) = |dH/dz|."""

    def test_returns_2_panels(self, simple_result) -> None:
        fig = plot_geometric_factor_sensitivity(simple_result, component="Hzz")
        assert fig is not None
        # 2 painéis: |H(z)| + G(z) normalizado
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_y_axis_inverted(self, simple_result) -> None:
        fig = plot_geometric_factor_sensitivity(simple_result)
        ax_h = fig.axes[0]
        ymin, ymax = ax_h.get_ylim()
        assert ymin > ymax  # profundidade cresce para baixo
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_freq_idx_out_of_range_raises(self, simple_result) -> None:
        nf = simple_result.H_tensor.shape[1]
        with pytest.raises(ValueError, match="freq_idx"):
            plot_geometric_factor_sensitivity(simple_result, freq_idx=nf + 99)


class TestMemoryUsageVsProfileSize:
    """Categoria (b) — diagnóstico: pico de RAM (MB) vs n_pontos."""

    def test_returns_figure_single_curve(self) -> None:
        sizes = np.array([10, 50, 100, 500])
        mem = np.array([12.0, 45.0, 90.0, 410.0])
        fig = plot_memory_usage_vs_profile_size(sizes, mem)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multiple_curves(self) -> None:
        sizes = np.array([10, 50, 100, 500])
        mem = np.array(
            [
                [12.0, 45.0, 90.0, 410.0],
                [25.0, 80.0, 160.0, 720.0],
            ]
        )
        fig = plot_memory_usage_vs_profile_size(sizes, mem, labels=["Numba", "JAX"])
        ax = fig.axes[0]
        assert len(ax.get_lines()) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_label_size_mismatch_raises(self) -> None:
        sizes = np.array([10, 50])
        mem = np.array([[12.0, 45.0], [25.0, 80.0]])
        with pytest.raises(ValueError, match="len.labels"):
            plot_memory_usage_vs_profile_size(sizes, mem, labels=["only_one"])


class TestBackendComparisonHeatmap:
    """Categoria (b) — diagnóstico: heatmap tempo (ms) backend × n_freq."""

    def test_returns_figure(self) -> None:
        times = np.array(
            [
                [1.2, 5.8, 28.0, 140.0],
                [3.5, 8.0, 30.0, 110.0],
            ]
        )
        fig = plot_backend_comparison_heatmap(
            times, backends=["Numba", "JAX-hybrid"], n_freqs=[1, 4, 16, 64]
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_shape_validation_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            plot_backend_comparison_heatmap(np.array([1.0, 2.0]))


class TestInferenceLatencyDistribution:
    """Categoria (d) — ML/DL: histograma + boxplot latência por batch."""

    def test_dict_input_returns_figure(self) -> None:
        rng = np.random.default_rng(0)
        data = {
            1: rng.normal(15.0, 3.0, 80),
            8: rng.normal(35.0, 6.0, 80),
            32: rng.normal(85.0, 12.0, 80),
        }
        fig = plot_inference_latency_distribution(data)
        assert fig is not None
        # 2 painéis: histograma + boxplot
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_array_input_with_batch_sizes(self) -> None:
        rng = np.random.default_rng(1)
        arr = rng.normal(20.0, 4.0, (3, 50))
        fig = plot_inference_latency_distribution(
            arr, batch_sizes=[1, 8, 32], realtime_target_ms=50.0
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_array_input_missing_batch_sizes_raises(self) -> None:
        arr = np.random.default_rng(2).normal(20.0, 4.0, (3, 50))
        with pytest.raises(ValueError, match="batch_sizes"):
            plot_inference_latency_distribution(arr)
