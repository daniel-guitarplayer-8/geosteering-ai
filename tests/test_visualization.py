# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTES: geosteering_ai.visualization — holdout, picasso, eda, realtime   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes para o subpacote visualization.

Estrutura (ALL CPU-safe — nenhum teste requer TF):
    TestImports — 4 simbolos publicos importaveis
    TestPlotHoldoutSamplesNoDisplay — Agg backend, show=False, sem excecao
"""
import numpy as np
import pytest

# ── Imports CPU-safe (visualization e NumPy+matplotlib only) ──────────────────
from geosteering_ai.visualization import (
    plot_holdout_samples,
    plot_picasso_dod,
    plot_eda_summary,
    RealtimeMonitor,
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
            y_true, y_pred,
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
            y_true, y_pred,
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
