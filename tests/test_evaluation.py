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

from geosteering_ai.evaluation.comparison import ComparisonResult, compare_models

# ── Imports CPU-safe (evaluation e NumPy-only) ───────────────────────────────
from geosteering_ai.evaluation.metrics import (
    MetricsReport,
    compute_all_metrics,
    compute_mae,
    compute_mape,
    compute_mbe,
    compute_r2,
    compute_rmse,
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
            r2=0.98,
            r2_rh=0.985,
            r2_rv=0.975,
            rmse=0.02,
            rmse_rh=0.018,
            rmse_rv=0.022,
            mae=0.015,
            mbe=-0.001,
            mape=1.5,
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
        expected_keys = {
            "r2",
            "r2_rh",
            "r2_rv",
            "rmse",
            "rmse_rh",
            "rmse_rv",
            "mae",
            "mbe",
            "mape",
        }
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
            r2=0.98,
            r2_rh=0.985,
            r2_rv=0.975,
            rmse=0.02,
            rmse_rh=0.018,
            rmse_rv=0.022,
            mae=0.015,
            mbe=-0.001,
            mape=1.5,
        )
        r_unet = MetricsReport(
            r2=0.95,
            r2_rh=0.96,
            r2_rv=0.94,
            rmse=0.04,
            rmse_rh=0.035,
            rmse_rv=0.045,
            mae=0.03,
            mbe=0.005,
            mape=3.0,
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


# ════════════════════════════════════════════════════════════════════════════
# TESTES: DOD (Depth of Detection) — Picasso analitico
# ════════════════════════════════════════════════════════════════════════════

from geosteering_ai.evaluation.dod import (
    DODResult,
    compute_dod_anisotropy,
    compute_dod_contrast,
    compute_dod_dip,
    compute_dod_frequency,
    compute_dod_map,
    compute_dod_snr,
    compute_dod_standard,
)


class TestDOD:
    """Testes para o modulo DOD (Depth of Detection) analitico."""

    # ── Ranges de resistividade para testes ───────────────────────────
    RT1 = np.logspace(-1, 3, 20)  # 0.1 a 1000 Ohm.m
    RT2 = np.logspace(-1, 3, 20)

    def test_dod_standard_shape(self):
        """compute_dod_standard via meshgrid produz shape (n_rt1, n_rt2)."""
        RT1_grid, RT2_grid = np.meshgrid(self.RT1, self.RT2, indexing="ij")
        dod = compute_dod_standard(RT1_grid, RT2_grid)
        assert dod.shape == (20, 20)
        # Todos os valores devem ser >= 0 (DOD fisicamente positivo)
        assert np.all(dod >= 0.0)

    def test_dod_diagonal_zero(self):
        """DOD ~ 0 na diagonal (rt1 == rt2, sem contraste)."""
        rt_values = np.logspace(-1, 3, 50)
        dod = compute_dod_standard(rt_values, rt_values)
        # Na diagonal, rt1 == rt2 → contraste = 0 → DOD = 0
        assert np.allclose(dod, 0.0, atol=1e-10)

    def test_dod_contrast_monotonic(self):
        """Maior contraste de resistividade → maior DOD."""
        rt1_fixed = np.array([10.0])
        # Contraste crescente: 10 vs 20, 10 vs 100, 10 vs 1000
        rt2_values = np.array([20.0, 100.0, 1000.0])
        dod_values = []
        for rt2 in rt2_values:
            dod = compute_dod_contrast(rt1_fixed, np.array([rt2]))
            dod_values.append(float(dod[0]))
        # DOD deve ser monotonicamente crescente com contraste
        assert dod_values[0] < dod_values[1] < dod_values[2]

    def test_dod_frequency_inverse(self):
        """Frequencia menor → DOD maior (penetracao mais profunda)."""
        rt1 = np.array([10.0])
        rt2 = np.array([100.0])
        dod_low_freq = compute_dod_standard(rt1, rt2, frequency_hz=2_000.0)
        dod_high_freq = compute_dod_standard(rt1, rt2, frequency_hz=200_000.0)
        # Frequencia 100x menor → DOD ~10x maior (escala sqrt)
        assert float(dod_low_freq.item()) > float(dod_high_freq.item())

    def test_dod_result_dataclass(self):
        """DODResult contem todos os campos esperados."""
        result = compute_dod_map(
            rt1_range=np.logspace(-1, 2, 10),
            rt2_range=np.logspace(-1, 2, 10),
            method="standard",
        )
        assert isinstance(result, DODResult)
        assert result.dod_map.shape == (10, 10)
        assert result.method == "standard"
        assert result.frequency_hz == 20_000.0
        assert result.spacing_m == 1.0
        assert isinstance(result.metadata, dict)
        assert result.rt1_range.shape == (10,)
        assert result.rt2_range.shape == (10,)

    def test_compute_dod_map_dispatch(self):
        """compute_dod_map despacha corretamente para todos os 6 metodos."""
        rt1 = np.logspace(0, 2, 5)
        rt2 = np.logspace(0, 2, 5)

        for method in ["standard", "contrast", "snr", "anisotropy", "dip"]:
            result = compute_dod_map(rt1, rt2, method=method)
            assert isinstance(result, DODResult), f"Falha para method={method}"
            assert result.method == method
            assert result.dod_map.shape == (
                5,
                5,
            ), f"Shape incorreto para method={method}: {result.dod_map.shape}"

        # Frequency retorna 3D (5, 5, 3) com default 3 frequencias
        result_freq = compute_dod_map(rt1, rt2, method="frequency")
        assert result_freq.dod_map.shape == (5, 5, 3)

    def test_compute_dod_map_invalid_method_raises(self):
        """Metodo invalido levanta ValueError."""
        with pytest.raises(ValueError, match="invalido"):
            compute_dod_map(
                np.logspace(0, 2, 5),
                np.logspace(0, 2, 5),
                method="invalid",
            )

    def test_dod_snr_noise_effect(self):
        """Ruído maior → DOD menor (degradação por SNR)."""
        # Contraste baixo (10 vs 12 Ωm, signal≈0.17) para que o noise
        # realmente degrade o SNR abaixo do threshold.
        rt1 = np.array([10.0])
        rt2 = np.array([12.0])
        dod_low_noise = compute_dod_snr(rt1, rt2, noise_level=0.01)
        dod_high_noise = compute_dod_snr(rt1, rt2, noise_level=0.5)
        assert float(dod_low_noise.item()) > float(dod_high_noise.item())

    def test_dod_anisotropy_isotropic(self):
        """Meio isotropico (rho_h == rho_v) → DOD = skin depth."""
        rho = np.array([10.0])
        dod = compute_dod_anisotropy(rho, rho, frequency_hz=20_000.0)
        # DOD deve ser igual ao skin depth para meio isotropico
        from geosteering_ai.evaluation.dod import _compute_skin_depth

        delta = _compute_skin_depth(rho, 20_000.0)
        assert np.allclose(dod, delta)

    def test_dod_dip_reduces_dod(self):
        """Mergulho > 0 reduz DOD (correcao coseno)."""
        rt1 = np.array([10.0])
        rt2 = np.array([100.0])
        dod_0 = compute_dod_dip(rt1, rt2, dip_deg=0.0)
        dod_45 = compute_dod_dip(rt1, rt2, dip_deg=45.0)
        dod_90 = compute_dod_dip(rt1, rt2, dip_deg=90.0)
        # dip=0 → maximo, dip=90 → zero
        assert float(dod_0.item()) > float(dod_45.item())
        assert float(dod_90.item()) == pytest.approx(0.0, abs=1e-10)

    def test_dod_frequency_sweep_shape(self):
        """compute_dod_frequency retorna shape correto para sweep."""
        rt1 = np.array([10.0, 100.0])
        rt2 = np.array([10.0, 100.0])
        freqs = [2_000.0, 20_000.0, 200_000.0, 2_000_000.0]
        dod_3d = compute_dod_frequency(rt1, rt2, frequencies=freqs)
        assert dod_3d.shape == (2, 4)  # (n_rt, n_freq)
