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
