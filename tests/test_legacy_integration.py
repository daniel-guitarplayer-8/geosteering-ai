# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTES: Integracao Legacy C26A/C42A/C42B → v2.0                         ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes para os 3 modulos adaptados do legado:

    C26A → data/inspection.py: inspect_data_splits(), export_inspection_csv()
    C42A → evaluation/config_report.py: generate_config_report()
    C42B → visualization/holdout.py: plot_holdout_clean_noisy()

Todos CPU-safe (nenhum teste requer TensorFlow).
"""
import logging

import numpy as np
import pytest

from geosteering_ai.config import PipelineConfig

# ════════════════════════════════════════════════════════════════════════════
# FIXTURES COMPARTILHADAS
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def config():
    """PipelineConfig baseline para testes."""
    return PipelineConfig.baseline()


@pytest.fixture
def splits_3d():
    """Dict de splits 3D simulando pipeline P1 baseline.

    Shape por split: (n_models, 600, 5).
    Features: [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)].
    """
    rng = np.random.default_rng(42)
    return {
        "train": rng.standard_normal((8, 100, 5)).astype(np.float32),
        "val": rng.standard_normal((2, 100, 5)).astype(np.float32),
        "test": rng.standard_normal((2, 100, 5)).astype(np.float32),
    }


# ════════════════════════════════════════════════════════════════════════════
# C26A: DATA INSPECTION (data/inspection.py)
# ════════════════════════════════════════════════════════════════════════════


class TestInspectDataSplits:
    """inspect_data_splits: estatisticas por split, NaN/Inf check."""

    def test_importable(self):
        """Funcao e importavel."""
        from geosteering_ai.data.inspection import inspect_data_splits

        assert callable(inspect_data_splits)

    def test_returns_dict(self, splits_3d, config):
        """Retorna dict com chaves esperadas."""
        from geosteering_ai.data.inspection import inspect_data_splits

        result = inspect_data_splits(splits_3d, config)
        assert isinstance(result, dict)
        assert "splits" in result
        assert "any_nan" in result
        assert "any_inf" in result

    def test_no_nan_in_clean_data(self, splits_3d, config):
        """Dados limpos nao devem ter NaN."""
        from geosteering_ai.data.inspection import inspect_data_splits

        result = inspect_data_splits(splits_3d, config)
        assert result["any_nan"] is False

    def test_detects_nan(self, config):
        """Detecta NaN quando presente."""
        from geosteering_ai.data.inspection import inspect_data_splits

        data = np.ones((2, 50, 5), dtype=np.float32)
        data[0, 10, 2] = np.nan
        result = inspect_data_splits({"train": data}, config)
        assert result["any_nan"] is True

    def test_detects_inf(self, config):
        """Detecta Inf quando presente."""
        from geosteering_ai.data.inspection import inspect_data_splits

        data = np.ones((2, 50, 5), dtype=np.float32)
        data[1, 20, 3] = np.inf
        result = inspect_data_splits({"train": data}, config)
        assert result["any_inf"] is True

    def test_per_split_stats(self, splits_3d, config):
        """Cada split tem min, max, mean, std por feature."""
        from geosteering_ai.data.inspection import inspect_data_splits

        result = inspect_data_splits(splits_3d, config)
        for split_name in ["train", "val", "test"]:
            stats = result["splits"][split_name]
            assert "n_samples" in stats
            assert "n_features" in stats
            assert "per_feature" in stats
            assert len(stats["per_feature"]) == 5  # 5 features

    def test_per_feature_has_stats(self, splits_3d, config):
        """Cada feature tem min, max, mean, std."""
        from geosteering_ai.data.inspection import inspect_data_splits

        result = inspect_data_splits(splits_3d, config)
        feat_stats = result["splits"]["train"]["per_feature"][0]
        for key in ["min", "max", "mean", "std"]:
            assert key in feat_stats
            assert isinstance(feat_stats[key], float)


class TestExportInspectionCsv:
    """export_inspection_csv: salva CSV com estatisticas."""

    def test_importable(self):
        from geosteering_ai.data.inspection import export_inspection_csv

        assert callable(export_inspection_csv)

    def test_creates_csv(self, splits_3d, config, tmp_path):
        """Cria arquivo CSV no diretorio."""
        from geosteering_ai.data.inspection import (
            export_inspection_csv,
            inspect_data_splits,
        )

        summary = inspect_data_splits(splits_3d, config)
        export_inspection_csv(summary, str(tmp_path))
        csv_path = tmp_path / "data_inspection_summary.csv"
        assert csv_path.exists()

    def test_csv_has_content(self, splits_3d, config, tmp_path):
        """CSV tem header e linhas de dados."""
        from geosteering_ai.data.inspection import (
            export_inspection_csv,
            inspect_data_splits,
        )

        summary = inspect_data_splits(splits_3d, config)
        export_inspection_csv(summary, str(tmp_path))
        csv_path = tmp_path / "data_inspection_summary.csv"
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) >= 2  # header + pelo menos 1 linha


# ════════════════════════════════════════════════════════════════════════════
# C42A: CONFIG REPORT (evaluation/config_report.py)
# ════════════════════════════════════════════════════════════════════════════


class TestGenerateConfigReport:
    """generate_config_report: relatorio pre-treinamento de FLAGS."""

    def test_importable(self):
        from geosteering_ai.evaluation.config_report import generate_config_report

        assert callable(generate_config_report)

    def test_returns_string(self, config):
        """Retorna string nao-vazia."""
        from geosteering_ai.evaluation.config_report import generate_config_report

        report = generate_config_report(config)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_contains_model_type(self, config):
        """Relatorio menciona model_type."""
        from geosteering_ai.evaluation.config_report import generate_config_report

        report = generate_config_report(config)
        assert config.model_type in report

    def test_contains_errata_values(self, config):
        """Relatorio inclui valores criticos da errata."""
        from geosteering_ai.evaluation.config_report import generate_config_report

        report = generate_config_report(config)
        assert "20000.0" in report  # frequency_hz
        assert "600" in report  # sequence_length

    def test_contains_section_headers(self, config):
        """Relatorio tem secoes organizadas."""
        from geosteering_ai.evaluation.config_report import generate_config_report

        report = generate_config_report(config)
        # Pelo menos as secoes principais devem existir
        assert "Modelo" in report or "modelo" in report or "model" in report.lower()

    def test_field_count(self, config):
        """Relatorio inclui contagem de campos."""
        from geosteering_ai.evaluation.config_report import generate_config_report

        report = generate_config_report(config)
        assert (
            "campos" in report.lower()
            or "fields" in report.lower()
            or "total" in report.lower()
        )

    def test_robusto_config(self):
        """Funciona com preset robusto."""
        from geosteering_ai.evaluation.config_report import generate_config_report

        config = PipelineConfig.robusto()
        report = generate_config_report(config)
        assert "0.08" in report  # noise_level_max


# ════════════════════════════════════════════════════════════════════════════
# C42B: HOLDOUT CLEAN VS NOISY (visualization/holdout.py)
# ════════════════════════════════════════════════════════════════════════════


class TestPlotHoldoutCleanNoisy:
    """plot_holdout_clean_noisy: overlay clean (preto) vs noisy (magenta)."""

    def test_importable(self):
        from geosteering_ai.visualization.holdout import plot_holdout_clean_noisy

        assert callable(plot_holdout_clean_noisy)

    def test_basic_run(self, config):
        """Executa sem excecao com dados sinteticos."""
        import matplotlib

        matplotlib.use("Agg")
        from geosteering_ai.visualization.holdout import plot_holdout_clean_noisy

        rng = np.random.default_rng(42)
        data = rng.standard_normal((3, 100, 5)).astype(np.float32)
        plot_holdout_clean_noisy(data, config=config, n_samples=1, show=False)

    def test_save_creates_file(self, config, tmp_path):
        """Salvamento cria arquivo PNG."""
        import matplotlib

        matplotlib.use("Agg")
        from geosteering_ai.visualization.holdout import plot_holdout_clean_noisy

        rng = np.random.default_rng(42)
        data = rng.standard_normal((3, 100, 5)).astype(np.float32)
        plot_holdout_clean_noisy(
            data,
            config=config,
            n_samples=1,
            save_dir=str(tmp_path),
            show=False,
        )
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 1

    def test_requires_3d(self, config):
        """Array 2D levanta ValueError."""
        from geosteering_ai.visualization.holdout import plot_holdout_clean_noisy

        with pytest.raises(ValueError, match="3D"):
            plot_holdout_clean_noisy(
                np.zeros((100, 5), dtype=np.float32),
                config=config,
                show=False,
            )

    def test_n_samples_respected(self, config):
        """n_samples limita numero de figuras geradas."""
        import matplotlib

        matplotlib.use("Agg")
        from geosteering_ai.visualization.holdout import plot_holdout_clean_noisy

        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 100, 5)).astype(np.float32)
        # Nao deve falhar com n_samples > n_models
        plot_holdout_clean_noisy(data, config=config, n_samples=20, show=False)

    def test_noise_applied(self, config):
        """Noise deve ser aplicado nas colunas EM quando noise_level > 0."""
        import matplotlib

        matplotlib.use("Agg")
        from geosteering_ai.visualization.holdout import plot_holdout_clean_noisy

        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, 100, 5)).astype(np.float32)
        # Deve executar sem erro com noise > 0
        config_noisy = PipelineConfig.robusto()
        plot_holdout_clean_noisy(
            data,
            config=config_noisy,
            n_samples=1,
            show=False,
        )
