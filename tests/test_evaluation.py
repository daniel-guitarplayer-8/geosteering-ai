# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTES: geosteering_ai.evaluation — metrics, comparison                  ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes para o subpacote evaluation.

Estrutura (ALL CPU-safe — nenhum teste requer TF):
    TestComputeR2 — R2 para arrays identicos e diferentes
    TestComputeRMSE — RMSE para arrays identicos e diferentes
    TestComputeMAE — MAE para arrays identicos
    TestComputeMBE — MBE para arrays identicos, positivo quando y_pred > y_true
    TestComputeMAPE — MAPE para arrays identicos
    TestMetricsReport — dataclass, to_dict(), summary()
    TestComputeAllMetrics — shape validation, basic computation
    TestCompareModels — ranking por r2 (desc), por rmse (asc)
"""
import numpy as np
import pytest

# ── Imports CPU-safe (evaluation e NumPy-only) ───────────────────────────────
from geosteering_ai.evaluation.metrics import (
    MetricsReport,
    compute_all_metrics,
    compute_r2,
    compute_rmse,
    compute_mae,
    compute_mbe,
    compute_mape,
)
from geosteering_ai.evaluation.comparison import (
    ComparisonResult,
    compare_models,
)


# ════════════════════════════════════════════════════════════════════════════
# TESTES: METRICAS INDIVIDUAIS
# ════════════════════════════════════════════════════════════════════════════


class TestComputeR2:
    """compute_r2 — coeficiente de determinacao."""

    def test_identical_arrays_r2_1(self):
        """R2 = 1.0 para arrays identicos."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_r2(y, y) == pytest.approx(1.0)

    def test_different_arrays_r2_lt_1(self):
        """R2 < 1.0 para arrays diferentes."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.3])
        r2 = compute_r2(y_true, y_pred)
        assert r2 < 1.0
        assert r2 > 0.0  # predicao ainda razoavel

    def test_constant_target_r2_zero(self):
        """R2 = 0.0 para target constante (ss_tot=0)."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        assert compute_r2(y_true, y_pred) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        """Shapes diferentes levantam ValueError."""
        with pytest.raises(ValueError, match="Shapes"):
            compute_r2(np.array([1, 2, 3]), np.array([1, 2]))


class TestComputeRMSE:
    """compute_rmse — Root Mean Squared Error."""

    def test_identical_arrays_rmse_zero(self):
        """RMSE = 0 para arrays identicos."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == pytest.approx(0.0)

    def test_different_arrays_rmse_positive(self):
        """RMSE > 0 para arrays diferentes."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        rmse = compute_rmse(y_true, y_pred)
        assert rmse > 0
        assert rmse == pytest.approx(0.1, abs=1e-6)

    def test_shape_mismatch_raises(self):
        """Shapes diferentes levantam ValueError."""
        with pytest.raises(ValueError, match="Shapes"):
            compute_rmse(np.array([1, 2]), np.array([1, 2, 3]))


class TestComputeMAE:
    """compute_mae — Mean Absolute Error."""

    def test_identical_arrays_mae_zero(self):
        """MAE = 0 para arrays identicos."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        """MAE calculado corretamente para valores conhecidos."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert compute_mae(y_true, y_pred) == pytest.approx(0.5)


class TestComputeMBE:
    """compute_mbe — Mean Bias Error."""

    def test_identical_arrays_mbe_zero(self):
        """MBE = 0 para arrays identicos."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mbe(y, y) == pytest.approx(0.0)

    def test_positive_when_pred_gt_true(self):
        """MBE > 0 quando y_pred > y_true (superestimativa)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        mbe = compute_mbe(y_true, y_pred)
        assert mbe > 0
        assert mbe == pytest.approx(1.0)

    def test_negative_when_pred_lt_true(self):
        """MBE < 0 quando y_pred < y_true (subestimativa)."""
        y_true = np.array([2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        mbe = compute_mbe(y_true, y_pred)
        assert mbe < 0
        assert mbe == pytest.approx(-1.0)


class TestComputeMAPE:
    """compute_mape — Mean Absolute Percentage Error."""

    def test_identical_arrays_mape_zero(self):
        """MAPE = 0 para arrays identicos."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mape(y, y) == pytest.approx(0.0)

    def test_different_arrays_mape_positive(self):
        """MAPE > 0 para arrays diferentes."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 22.0, 33.0])
        mape = compute_mape(y_true, y_pred)
        assert mape > 0
        # Erro de 10% em todos os pontos → MAPE ~ 10%
        assert mape == pytest.approx(10.0, abs=0.01)


# ════════════════════════════════════════════════════════════════════════════
# TESTES: METRICS REPORT
# ════════════════════════════════════════════════════════════════════════════


class TestMetricsReport:
    """MetricsReport dataclass — to_dict, summary."""

    def _make_report(self, **kwargs):
        defaults = dict(
            r2=0.98, r2_rh=0.985, r2_rv=0.975,
            rmse=0.02, rmse_rh=0.018, rmse_rv=0.022,
            mae=0.015, mbe=-0.001, mape=1.5,
        )
        defaults.update(kwargs)
        return MetricsReport(**defaults)

    def test_instantiation(self):
        """MetricsReport instancia com todos os campos."""
        report = self._make_report()
        assert report.r2 == 0.98
        assert report.rmse == 0.02

    def test_to_dict_keys(self):
        """to_dict retorna dict com 9 chaves."""
        report = self._make_report()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 9
        expected_keys = {"r2", "r2_rh", "r2_rv", "rmse", "rmse_rh",
                         "rmse_rv", "mae", "mbe", "mape"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        """to_dict preserva valores corretos."""
        report = self._make_report()
        d = report.to_dict()
        assert d["r2"] == 0.98
        assert d["mbe"] == -0.001

    def test_summary_returns_string(self):
        """summary() retorna string multi-linha."""
        report = self._make_report()
        s = report.summary()
        assert isinstance(s, str)
        assert "R2" in s
        assert "RMSE" in s
        assert "log10" in s


# ════════════════════════════════════════════════════════════════════════════
# TESTES: COMPUTE ALL METRICS
# ════════════════════════════════════════════════════════════════════════════


class TestComputeAllMetrics:
    """compute_all_metrics — shape validation e basic computation."""

    def test_identical_arrays_perfect_r2(self):
        """Arrays identicos: R2 = 1.0, RMSE = 0."""
        rng = np.random.default_rng(42)
        y = rng.uniform(0.5, 3.0, (5, 100, 2)).astype(np.float32)
        report = compute_all_metrics(y, y)
        assert report.r2 == pytest.approx(1.0)
        assert report.rmse == pytest.approx(0.0, abs=1e-10)
        assert report.mae == pytest.approx(0.0, abs=1e-10)
        assert report.mbe == pytest.approx(0.0, abs=1e-10)

    def test_shape_2d_raises(self):
        """Shape 2D levanta ValueError (esperado 3D)."""
        y = np.random.randn(10, 600).astype(np.float32)
        with pytest.raises(ValueError, match="Shape esperado"):
            compute_all_metrics(y, y)

    def test_wrong_channels_raises(self):
        """Ultima dim != 2 levanta ValueError."""
        y = np.random.randn(5, 100, 3).astype(np.float32)
        with pytest.raises(ValueError, match="Shape esperado"):
            compute_all_metrics(y, y)

    def test_shape_mismatch_raises(self):
        """Shapes diferentes levantam ValueError."""
        y1 = np.random.randn(5, 100, 2).astype(np.float32)
        y2 = np.random.randn(3, 100, 2).astype(np.float32)
        with pytest.raises(ValueError, match="Shapes"):
            compute_all_metrics(y1, y2)

    def test_returns_metrics_report(self):
        """compute_all_metrics retorna MetricsReport."""
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0.5, 3.0, (5, 100, 2))
        y_pred = y_true + rng.normal(0, 0.01, y_true.shape)
        report = compute_all_metrics(y_true, y_pred)
        assert isinstance(report, MetricsReport)
        assert report.r2 > 0.99  # pequeno ruido → R2 alto


# ════════════════════════════════════════════════════════════════════════════
# TESTES: COMPARE MODELS
# ════════════════════════════════════════════════════════════════════════════


class TestCompareModels:
    """compare_models — ranking por metricas."""

    def _make_reports(self):
        """Cria 2 MetricsReport: ResNet melhor que UNet."""
        r_resnet = MetricsReport(
            r2=0.98, r2_rh=0.985, r2_rv=0.975,
            rmse=0.02, rmse_rh=0.018, rmse_rv=0.022,
            mae=0.015, mbe=-0.001, mape=1.5,
        )
        r_unet = MetricsReport(
            r2=0.95, r2_rh=0.96, r2_rv=0.94,
            rmse=0.04, rmse_rh=0.035, rmse_rv=0.045,
            mae=0.03, mbe=0.005, mape=3.0,
        )
        return {"ResNet_18": r_resnet, "UNet": r_unet}

    def test_ranking_by_r2_descending(self):
        """Ranking por R2: ResNet (0.98) antes de UNet (0.95)."""
        results = self._make_reports()
        comp = compare_models(results, metric="r2")
        assert comp.best_model == "ResNet_18"
        assert comp.ranking[0] == "ResNet_18"
        assert comp.ranking[1] == "UNet"

    def test_ranking_by_rmse_ascending(self):
        """Ranking por RMSE: ResNet (0.02) antes de UNet (0.04)."""
        results = self._make_reports()
        comp = compare_models(results, metric="rmse")
        assert comp.best_model == "ResNet_18"
        assert comp.ranking[0] == "ResNet_18"

    def test_returns_comparison_result(self):
        """compare_models retorna ComparisonResult."""
        results = self._make_reports()
        comp = compare_models(results, metric="r2")
        assert isinstance(comp, ComparisonResult)
        assert comp.ranking_metric == "r2"
        assert set(comp.model_names) == {"ResNet_18", "UNet"}

    def test_empty_results_raises(self):
        """Dict vazio levanta ValueError."""
        with pytest.raises(ValueError, match="vazio"):
            compare_models({}, metric="r2")

    def test_invalid_metric_raises(self):
        """Metrica invalida levanta ValueError."""
        results = self._make_reports()
        with pytest.raises(ValueError, match="invalida"):
            compare_models(results, metric="invalid_metric")

    def test_summary_returns_string(self):
        """ComparisonResult.summary() retorna string com ranking."""
        results = self._make_reports()
        comp = compare_models(results, metric="r2")
        s = comp.summary()
        assert isinstance(s, str)
        assert "ResNet_18" in s
        assert "UNet" in s
        assert "RANKING" in s
