# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_canonical_models.py                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — Modelos Canônicos (Sprint 2.9)                   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest                                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Testa o catálogo de 7 modelos canônicos + wrappers de plotagem.       ║
# ║    Valida: ranges físicos, shapes dos arrays, nomes/títulos, invariantes║
# ║    geométricos (interfaces crescentes), simulação forward funcional,    ║
# ║    wrapper plot_canonical_model sem erros.                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do catálogo de modelos canônicos."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend não-interativo antes de pyplot

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from geosteering_ai.simulation.validation.canonical_models import (  # noqa: E402
    CanonicalModel,
    get_all_canonical_models,
    get_canonical_model,
    list_canonical_models,
)

# Lista de todos os modelos para parametrização
_ALL_MODEL_NAMES = list_canonical_models()


class TestCatalog:
    """Testa o catálogo de modelos."""

    def test_list_has_seven_models(self) -> None:
        assert len(_ALL_MODEL_NAMES) == 7

    def test_expected_names(self) -> None:
        expected = {
            "oklahoma_3",
            "oklahoma_5",
            "devine_8",
            "oklahoma_15",
            "oklahoma_28",
            "hou_7",
            "viking_graben_10",
        }
        assert set(_ALL_MODEL_NAMES) == expected

    def test_get_all_returns_sorted_by_layers(self) -> None:
        models = get_all_canonical_models()
        layer_counts = [m.n_layers for m in models]
        assert layer_counts == sorted(layer_counts)

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="desconhecido"):
            get_canonical_model("model_not_existing")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", _ALL_MODEL_NAMES)
class TestIndividualModels:
    """Testes aplicados a cada um dos 7 modelos."""

    def test_is_canonical_model_instance(self, name: str) -> None:
        m = get_canonical_model(name)  # type: ignore[arg-type]
        assert isinstance(m, CanonicalModel)
        assert m.name == name

    def test_arrays_have_correct_shapes(self, name: str) -> None:
        m = get_canonical_model(name)  # type: ignore[arg-type]
        assert m.rho_h.shape == (m.n_layers,)
        assert m.rho_v.shape == (m.n_layers,)
        assert m.esp.shape == (m.n_layers - 2,)

    def test_rho_h_positive(self, name: str) -> None:
        m = get_canonical_model(name)  # type: ignore[arg-type]
        assert np.all(m.rho_h > 0)

    def test_rho_v_positive(self, name: str) -> None:
        m = get_canonical_model(name)  # type: ignore[arg-type]
        assert np.all(m.rho_v > 0)

    def test_esp_positive(self, name: str) -> None:
        m = get_canonical_model(name)  # type: ignore[arg-type]
        if m.esp.size > 0:
            assert np.all(m.esp > 0)

    def test_interfaces_monotonic(self, name: str) -> None:
        """Interfaces devem ser estritamente crescentes."""
        m = get_canonical_model(name)  # type: ignore[arg-type]
        interfaces = m.interfaces
        if interfaces.size > 1:
            diffs = np.diff(interfaces)
            assert np.all(diffs > 0)

    def test_depth_range_ok(self, name: str) -> None:
        m = get_canonical_model(name)  # type: ignore[arg-type]
        assert m.min_depth >= 0.0
        assert m.max_depth > m.min_depth

    def test_anisotropy_type_valid(self, name: str) -> None:
        m = get_canonical_model(name)  # type: ignore[arg-type]
        assert m.anisotropy_type in {"isotropic", "tiv", "tiv_strong"}


@pytest.mark.parametrize("name", _ALL_MODEL_NAMES)
class TestSimulationIntegration:
    """Valida que cada modelo é simulável sem erros."""

    def test_simulate_returns_finite(self, name: str) -> None:
        from geosteering_ai.simulation import SimulationConfig, simulate

        m = get_canonical_model(name)  # type: ignore[arg-type]
        # Usa apenas 30 posições para rapidez
        positions_z = np.linspace(m.min_depth - 1.0, m.max_depth + 1.0, 30)

        cfg = SimulationConfig(parallel=False)
        result = simulate(
            rho_h=m.rho_h,
            rho_v=m.rho_v,
            esp=m.esp,
            positions_z=positions_z,
            cfg=cfg,
        )

        assert result.H_tensor.shape == (30, 1, 9)
        assert np.all(np.isfinite(result.H_tensor))


class TestPlotCanonicalModel:
    """Testa o wrapper plot_canonical_model."""

    def test_plots_oklahoma_3(self) -> None:
        from geosteering_ai.simulation.visualization import plot_canonical_model

        fig = plot_canonical_model("oklahoma_3", n_positions=50)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plots_viking_graben(self) -> None:
        from geosteering_ai.simulation.visualization import plot_canonical_model

        fig = plot_canonical_model("viking_graben_10", n_positions=50, dip_deg=45.0)
        suptitle_text = fig._suptitle.get_text()
        assert "Viking" in suptitle_text
        assert "45" in suptitle_text  # dip incluído no título
        import matplotlib.pyplot as plt

        plt.close(fig)
