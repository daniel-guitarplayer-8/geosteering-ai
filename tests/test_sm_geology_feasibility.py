# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_geology_feasibility.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Q3 fix — viabilidade geométrica: tj vs n_layers·min_thickness            ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-18                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Quando tj ≤ (n_layers−2)·min_thickness, o gerador degenera as           ║
# ║    espessuras (partição igual, < min_thickness, IDÊNTICAS → diversidade    ║
# ║    zero) — modelos válidos, mas min_thickness violado em silêncio. Fix:    ║
# ║    (a) fail-fast no validate() do VM; (b) warning não-levantante em         ║
# ║    _build_n_layer_choices (chokepoint comum a generate_models E ao           ║
# ║    monólito ModelGenerationThread → cobre CLI/SM-MVVM/monólito/Studio).      ║
# ║    Estes testes fixam ambos + não-regressão (configs viáveis seguem OK).     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do fix de viabilidade geométrica (Q3) — validate() do VM + warning lib."""

from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from geosteering_ai.gui.viewmodels.signal import VMSignal  # noqa: E402


class _StubService:
    def __init__(self) -> None:
        self.finished = VMSignal()
        self.error = VMSignal()
        self.progress = VMSignal()

    def run(self, request: Any) -> None:  # noqa: ANN001
        pass


def _vm():
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel

    return SimulationViewModel(service=_StubService())


def _feasibility_errors(vm):
    return [e for e in vm.validate() if "insuficiente" in e and "tj" in e]


# ════════════════════════════════════════════════════════════════════════════
# (a) validate() do VM — fail-fast
# ════════════════════════════════════════════════════════════════════════════
def test_validate_rejects_infeasible_fixed():
    """n_layers fixo grande + tj pequeno → erro de viabilidade (degeneração)."""
    vm = _vm()
    vm.geology_mode = "stochastic"
    vm.n_layers_fixed = 20  # 18 internas
    vm.tj = 10.0
    vm.min_thickness = 1.0  # 18·1 = 18 > tj=10 → inviável
    errs = _feasibility_errors(vm)
    assert errs, f"esperava erro de viabilidade; validate()={vm.validate()}"


def test_validate_accepts_feasible_fixed():
    """n_layers fixo pequeno (default) → SEM erro de viabilidade (config válida)."""
    vm = _vm()
    vm.geology_mode = "stochastic"
    vm.n_layers_fixed = 5  # 3 internas · 1 = 3 < tj=10
    vm.tj = 10.0
    vm.min_thickness = 1.0
    assert _feasibility_errors(vm) == []
    assert vm.validate() == []  # default totalmente válido


def test_validate_rejects_infeasible_variable_uses_max():
    """n_layers variável usa o PIOR caso (máx amostrável = max−1) p/ a viabilidade."""
    vm = _vm()
    vm.geology_mode = "stochastic"
    vm.n_layers_fixed = None
    vm.n_layers_min = 3
    vm.n_layers_max = 21  # máx amostrável = 20 → 18 internas
    vm.tj = 10.0
    vm.min_thickness = 1.0
    assert _feasibility_errors(vm), "deep models (20 camadas) degeneram → erro"


def test_validate_accepts_feasible_variable_default():
    """n_layers variável default [3,11) (máx 10 → 8 internas) + tj=10 → OK."""
    vm = _vm()
    vm.geology_mode = "stochastic"
    vm.n_layers_fixed = None
    vm.n_layers_min = 3
    vm.n_layers_max = 11
    vm.tj = 10.0
    vm.min_thickness = 1.0
    assert _feasibility_errors(vm) == []


def test_validate_borderline_equal_is_rejected():
    """tj EXATAMENTE == n_internas·min_thickness → rejeitado (≤): partição igual,
    diversidade nula (e o solver iterativo ainda fica levemente abaixo do piso)."""
    vm = _vm()
    vm.geology_mode = "stochastic"
    vm.n_layers_fixed = 12  # 10 internas
    vm.tj = 10.0
    vm.min_thickness = 1.0  # 10·1 = 10 == tj
    assert _feasibility_errors(vm)


def test_validate_skips_feasibility_for_manual_mode():
    """Modo manual NÃO dispara o check de viabilidade estocástica."""
    vm = _vm()
    vm.geology_mode = "manual"  # _validate_geology não é chamado
    vm.n_layers_fixed = 20
    vm.tj = 10.0
    vm.min_thickness = 1.0
    assert _feasibility_errors(vm) == []  # nenhum erro de viabilidade estocástica


# ════════════════════════════════════════════════════════════════════════════
# (b) generate_models — warning não-levantante (lib-level)
# ════════════════════════════════════════════════════════════════════════════
def test_generate_models_warns_when_infeasible(caplog):
    """Config inviável → logger.warning (NÃO levanta; modelos ainda gerados)."""
    from geosteering_ai.gui.services.stochastic_geology import (
        GenConfig,
        generate_models,
    )

    cfg = GenConfig(total_depth=10.0, n_layers_fixed=20, min_thickness=1.0)
    with caplog.at_level(logging.WARNING, logger="geosteering_ai.gui.services"):
        models = generate_models(cfg, 3, rng_seed=1)
    assert len(models) == 3  # NÃO levantou — modelos gerados
    assert any(
        "degeneram" in r.message or "insuficiente" in r.message for r in caplog.records
    ), f"esperava warning de degeneração; logs={[r.message for r in caplog.records]}"


def test_generate_models_no_warn_when_feasible(caplog):
    """Config viável (tj grande) → SEM warning de degeneração."""
    from geosteering_ai.gui.services.stochastic_geology import (
        GenConfig,
        generate_models,
    )

    cfg = GenConfig(total_depth=120.0, n_layers_fixed=20, min_thickness=1.0)
    with caplog.at_level(logging.WARNING, logger="geosteering_ai.gui.services"):
        generate_models(cfg, 3, rng_seed=1)
    assert not any(
        "degeneram" in r.message or "insuficiente" in r.message for r in caplog.records
    )


def test_warning_covers_monolith_path_via_build_n_layer_choices(caplog):
    """Cobertura do MONÓLITO (achado CONFIRMED da revisão): o warning vive em
    _build_n_layer_choices (chokepoint comum), então o ModelGenerationThread — que o
    chama DIRETO, sem generate_models — também é coberto."""
    from geosteering_ai.gui.services.stochastic_geology import (
        GenConfig,
        _build_n_layer_choices,
    )

    cfg = GenConfig(total_depth=10.0, n_layers_fixed=20, min_thickness=1.0)
    with caplog.at_level(logging.WARNING, logger="geosteering_ai.gui.services"):
        _build_n_layer_choices(cfg)  # caminho direto do monólito
    assert any("degeneram" in r.message for r in caplog.records)
