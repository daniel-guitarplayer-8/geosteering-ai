# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTES: geosteering_ai.visualization — holdout, picasso, eda, realtime   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes para o subpacote visualization.

Estrutura (ALL CPU-safe — nenhum teste requer TF):
    TestImports — simbolos publicos importaveis (original + 5 novos EDA)
    TestPlotHoldoutSamplesNoDisplay — Agg backend, show=False, sem excecao
    TestPlotFeatureDistributions — histogramas + KDE, 3 splits
    TestPlotCorrelationHeatmap — correlacao com threshold
    TestPlotSampleProfiles — componentes EM vs profundidade
    TestPlotTrainValTestComparison — boxplot splits, deteccao leakage
    TestPlotSensitivityHeatmap — variancia features x profundidade
"""
import numpy as np
import pytest

# ── Imports CPU-safe (visualization e NumPy+matplotlib only) ──────────────────
from geosteering_ai.visualization import (
    RealtimeMonitor,
    plot_eda_summary,
    plot_holdout_samples,
    plot_picasso_dod,
)
from geosteering_ai.visualization.eda import (
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_sample_profiles,
    plot_sensitivity_heatmap,
    plot_train_val_test_comparison,
)

# ════════════════════════════════════════════════════════════════════════════
# TESTES CPU-SAFE
# ════════════════════════════════════════════════════════════════════════════


class TestImports:
    """Simbolos publicos do subpacote visualization."""

    def test_plot_holdout_samples_importable(self):
        """plot_holdout_samples e importavel e callable."""
        assert callable(plot_holdout_samples)

    def test_plot_picasso_dod_importable(self):
        """plot_picasso_dod e importavel e callable."""
        assert callable(plot_picasso_dod)

    def test_plot_eda_summary_importable(self):
        """plot_eda_summary e importavel e callable."""
        assert callable(plot_eda_summary)

    def test_realtime_monitor_importable(self):
        """RealtimeMonitor e importavel."""
        assert RealtimeMonitor is not None


class TestPlotHoldoutSamplesNoDisplay:
    """plot_holdout_samples com Agg backend — sem display."""

    def test_basic_plot_no_exception(self, tmp_path):
        """Plotagem basica sem excecao com show=False."""
        import matplotlib

        matplotlib.use("Agg")

        rng = np.random.default_rng(42)
        y_true = rng.uniform(0.5, 3.0, (3, 100, 2)).astype(np.float32)
        y_pred = rng.uniform(0.5, 3.0, (3, 100, 2)).astype(np.float32)

        # Deve executar sem excecao
        plot_holdout_samples(
            y_true,
            y_pred,
            n_samples=1,
            show=False,
        )

    def test_save_to_dir(self, tmp_path):
        """Plotagem com salvamento em diretorio temporario."""
        import matplotlib

        matplotlib.use("Agg")

        rng = np.random.default_rng(42)
        y_true = rng.uniform(0.5, 3.0, (3, 100, 2)).astype(np.float32)
        y_pred = rng.uniform(0.5, 3.0, (3, 100, 2)).astype(np.float32)

        save_dir = str(tmp_path / "holdout_plots")
        plot_holdout_samples(
            y_true,
            y_pred,
            n_samples=1,
            save_dir=save_dir,
            show=False,
        )

        # Verifica que o diretorio foi criado
        import os

        assert os.path.isdir(save_dir)

    def test_shape_mismatch_raises(self):
        """Shapes diferentes levantam ValueError."""
        y_true = np.random.randn(3, 100, 2).astype(np.float32)
        y_pred = np.random.randn(5, 100, 2).astype(np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            plot_holdout_samples(y_true, y_pred, show=False)


# ════════════════════════════════════════════════════════════════════════════
# TESTES: 5 FUNCOES EDA AVANCADO (Fase IV)
# ════════════════════════════════════════════════════════════════════════════
# Cada classe testa uma funcao nova do eda.py expandido.
# Todos usam Agg backend (headless) e show=False.
# Dados sinteticos simulam o layout do pipeline: (n_models, 600, 5).
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def eda_data_3d():
    """Dados sinteticos 3D simulando pipeline P1 baseline.

    Shape: (10, 600, 5) — 10 modelos, 600 medidas, 5 features.
    Features: [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)].
    """
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 600, 5)).astype(np.float32)


@pytest.fixture
def eda_feature_names():
    """Nomes canonicos das 5 features baseline P1."""
    return ["z_obs", "Re(Hxx)", "Im(Hxx)", "Re(Hzz)", "Im(Hzz)"]


@pytest.fixture
def eda_splits(eda_data_3d):
    """Dicionario train/val/test a partir de eda_data_3d."""
    return {
        "train": eda_data_3d[:6],  # 6 modelos
        "val": eda_data_3d[6:8],  # 2 modelos
        "test": eda_data_3d[8:],  # 2 modelos
    }


class TestPlotFeatureDistributions:
    """plot_feature_distributions: histogramas com KDE, 3 splits sobrepostos."""

    def test_importable(self):
        """Funcao e importavel e callable."""
        assert callable(plot_feature_distributions)

    def test_single_array_runs(self, eda_data_3d, eda_feature_names):
        """Executa sem excecao com um unico array 3D."""
        import matplotlib

        matplotlib.use("Agg")
        plot_feature_distributions(
            eda_data_3d,
            feature_names=eda_feature_names,
            show=False,
        )

    def test_splits_dict_runs(self, eda_splits, eda_feature_names):
        """Executa sem excecao com dicionario de splits."""
        import matplotlib

        matplotlib.use("Agg")
        plot_feature_distributions(
            eda_splits,
            feature_names=eda_feature_names,
            show=False,
        )

    def test_save_creates_file(self, eda_data_3d, tmp_path):
        """Salvamento cria arquivo PNG no diretorio."""
        import matplotlib

        matplotlib.use("Agg")
        plot_feature_distributions(
            eda_data_3d,
            save_dir=str(tmp_path),
            show=False,
        )
        assert (tmp_path / "eda_feature_distributions.png").exists()

    def test_invalid_ndim_raises(self):
        """Array 1D levanta ValueError."""
        with pytest.raises(ValueError):
            plot_feature_distributions(np.zeros(10), show=False)

    def test_2d_input_accepted(self):
        """Array 2D (n_samples, n_features) aceito sem excecao."""
        import matplotlib

        matplotlib.use("Agg")
        data_2d = np.random.default_rng(0).standard_normal((100, 5)).astype(np.float32)
        plot_feature_distributions(data_2d, show=False)


class TestPlotCorrelationHeatmap:
    """plot_correlation_heatmap: correlacao com anotacoes e threshold."""

    def test_importable(self):
        """Funcao e importavel e callable."""
        assert callable(plot_correlation_heatmap)

    def test_basic_run(self, eda_data_3d, eda_feature_names):
        """Executa sem excecao com dados 3D e nomes."""
        import matplotlib

        matplotlib.use("Agg")
        plot_correlation_heatmap(
            eda_data_3d,
            feature_names=eda_feature_names,
            show=False,
        )

    def test_threshold_filtering(self, eda_data_3d):
        """Threshold filtra correlacoes fracas na anotacao."""
        import matplotlib

        matplotlib.use("Agg")
        plot_correlation_heatmap(
            eda_data_3d,
            threshold=0.5,
            show=False,
        )

    def test_save_creates_file(self, eda_data_3d, tmp_path):
        """Salvamento cria arquivo PNG."""
        import matplotlib

        matplotlib.use("Agg")
        plot_correlation_heatmap(
            eda_data_3d,
            save_dir=str(tmp_path),
            show=False,
        )
        assert (tmp_path / "eda_correlation_heatmap.png").exists()

    def test_method_spearman(self, eda_data_3d):
        """Metodo Spearman funciona sem excecao."""
        import matplotlib

        matplotlib.use("Agg")
        plot_correlation_heatmap(
            eda_data_3d,
            method="spearman",
            show=False,
        )

    def test_invalid_method_raises(self, eda_data_3d):
        """Metodo invalido levanta ValueError."""
        with pytest.raises(ValueError, match="method"):
            plot_correlation_heatmap(
                eda_data_3d,
                method="invalid_method",
                show=False,
            )


class TestPlotSampleProfiles:
    """plot_sample_profiles: componentes EM vs profundidade (z_obs)."""

    def test_importable(self):
        """Funcao e importavel e callable."""
        assert callable(plot_sample_profiles)

    def test_basic_run(self, eda_data_3d, eda_feature_names):
        """Executa sem excecao com dados 3D."""
        import matplotlib

        matplotlib.use("Agg")
        plot_sample_profiles(
            eda_data_3d,
            feature_names=eda_feature_names,
            show=False,
        )

    def test_n_samples_limit(self, eda_data_3d):
        """Limitar n_samples funciona sem excecao."""
        import matplotlib

        matplotlib.use("Agg")
        plot_sample_profiles(
            eda_data_3d,
            n_samples=2,
            show=False,
        )

    def test_save_creates_file(self, eda_data_3d, tmp_path):
        """Salvamento cria arquivo PNG."""
        import matplotlib

        matplotlib.use("Agg")
        plot_sample_profiles(
            eda_data_3d,
            save_dir=str(tmp_path),
            show=False,
        )
        assert (tmp_path / "eda_sample_profiles.png").exists()

    def test_requires_3d(self):
        """Array 2D levanta ValueError (precisa de sequencia temporal)."""
        with pytest.raises(ValueError, match="3D"):
            plot_sample_profiles(
                np.random.randn(100, 5).astype(np.float32),
                show=False,
            )


class TestPlotTrainValTestComparison:
    """plot_train_val_test_comparison: boxplots splits para detectar leakage."""

    def test_importable(self):
        """Funcao e importavel e callable."""
        assert callable(plot_train_val_test_comparison)

    def test_basic_run(self, eda_splits, eda_feature_names):
        """Executa sem excecao com dicionario de splits."""
        import matplotlib

        matplotlib.use("Agg")
        plot_train_val_test_comparison(
            eda_splits,
            feature_names=eda_feature_names,
            show=False,
        )

    def test_save_creates_file(self, eda_splits, tmp_path):
        """Salvamento cria arquivo PNG."""
        import matplotlib

        matplotlib.use("Agg")
        plot_train_val_test_comparison(
            eda_splits,
            save_dir=str(tmp_path),
            show=False,
        )
        assert (tmp_path / "eda_train_val_test_comparison.png").exists()

    def test_missing_split_warns(self, eda_data_3d, caplog):
        """Dicionario sem 'val' loga warning mas nao falha."""
        import matplotlib

        matplotlib.use("Agg")
        import logging

        with caplog.at_level(logging.WARNING, logger="geosteering_ai.visualization.eda"):
            plot_train_val_test_comparison(
                {"train": eda_data_3d[:6], "test": eda_data_3d[6:]},
                show=False,
            )
        assert any(
            "val" in rec.message.lower() for rec in caplog.records
        ), "Esperado warning sobre split 'val' faltando"

    def test_invalid_type_raises(self):
        """Argumento que nao e dict levanta TypeError."""
        with pytest.raises(TypeError):
            plot_train_val_test_comparison(
                np.zeros((10, 600, 5)),
                show=False,
            )


class TestPlotSensitivityHeatmap:
    """plot_sensitivity_heatmap: variancia features x profundidade."""

    def test_importable(self):
        """Funcao e importavel e callable."""
        assert callable(plot_sensitivity_heatmap)

    def test_basic_run(self, eda_data_3d, eda_feature_names):
        """Executa sem excecao com dados 3D."""
        import matplotlib

        matplotlib.use("Agg")
        plot_sensitivity_heatmap(
            eda_data_3d,
            feature_names=eda_feature_names,
            show=False,
        )

    def test_save_creates_file(self, eda_data_3d, tmp_path):
        """Salvamento cria arquivo PNG."""
        import matplotlib

        matplotlib.use("Agg")
        plot_sensitivity_heatmap(
            eda_data_3d,
            save_dir=str(tmp_path),
            show=False,
        )
        assert (tmp_path / "eda_sensitivity_heatmap.png").exists()

    def test_requires_3d(self):
        """Array 2D levanta ValueError (precisa de eixo temporal)."""
        with pytest.raises(ValueError, match="3D"):
            plot_sensitivity_heatmap(
                np.random.randn(100, 5).astype(np.float32),
                show=False,
            )

    def test_metric_gradient(self, eda_data_3d):
        """Metrica 'gradient' funciona sem excecao."""
        import matplotlib

        matplotlib.use("Agg")
        plot_sensitivity_heatmap(
            eda_data_3d,
            metric="gradient",
            show=False,
        )
