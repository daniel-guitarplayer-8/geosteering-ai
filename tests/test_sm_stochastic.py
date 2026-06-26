# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_stochastic.py                                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0011c-sm-stochastic                                        ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app MVVM — geração estocástica (Fatia 3)                ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-06                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre os ACs da spec 0011c: extração pura do gerador (sem Qt) + re-      ║
# ║    export do monólito (AC-1/AC-2), geração estocástica via simulate_batch  ║
# ║    (AC-3), agrupamento por n_layers + reassembly de ordem (AC-4), VM puro   ║
# ║    com validação de geologia (AC-5) e a fronteira (AC-6).                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0011c — geração estocástica do SM MVVM (extração · agrupamento · VM)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def _make_vm():
    """ViewModel real com service stub (puro, sem Qt)."""
    from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
    from geosteering_ai.gui.viewmodels.signal import VMSignal

    class StubService:
        def __init__(self) -> None:
            self.finished = VMSignal()
            self.error = VMSignal()
            self.requests: list = []

        def run(self, request) -> None:  # noqa: ANN001
            self.requests.append(request)

    return SimulationViewModel(service=StubService())


# ════════════════════════════════════════════════════════════════════════════
# AC-1 / AC-6 — extração PURA (gerador sem Qt) + determinismo dos 7 geradores
# ════════════════════════════════════════════════════════════════════════════
def test_stochastic_geology_importable_without_qt():
    """AC-6 — importar o gerador NÃO puxa Qt (Princípio X)."""
    code = (
        "import sys\n"
        "import geosteering_ai.gui.services.stochastic_geology  # noqa\n"
        "bad = [m for m in ('PyQt6', 'PySide6') if m in sys.modules]\n"
        "assert not bad, f'gerador puxou Qt: {bad}'\n"
        "print('PURE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "PURE_OK" in proc.stdout


@pytest.mark.parametrize(
    "generator",
    [
        "sobol",
        "halton",
        "niederreiter",
        "mersenne_twister",
        "uniform",
        "normal",
        "box_muller",
    ],
)
def test_generate_models_deterministic_and_valid(generator):
    """AC-1 — cada gerador é determinístico (mesma seed) e produz modelos TIV válidos."""
    from geosteering_ai.gui.services.stochastic_geology import (
        MODEL_KEYS,
        GenConfig,
        generate_models,
    )

    cfg = GenConfig(n_layers_fixed=5, generator=generator, total_depth=20.0)
    a = generate_models(cfg, n_models=8, rng_seed=42)
    b = generate_models(cfg, n_models=8, rng_seed=42)
    assert a == b  # determinístico
    m0 = a[0]
    assert tuple(m0.keys()) == MODEL_KEYS
    assert len(m0["rho_h"]) == 5 and len(m0["thicknesses"]) == 3  # n_layers-2
    assert all(0.99 <= r <= 1801.0 for r in m0["rho_h"])  # range ρₕ default
    assert all(1.0 <= L <= 1.4143 for L in m0["lambda"])  # λ ∈ [1, √2]
    assert abs(sum(m0["thicknesses"]) - 20.0) < 1e-6  # Σ esp == total_depth


# ════════════════════════════════════════════════════════════════════════════
# AC-2 — re-export do monólito + DRY (mesmo objeto, não duplicado)
# ════════════════════════════════════════════════════════════════════════════
def test_sm_model_gen_reexports_core_dry():
    """AC-2 — sm_model_gen re-exporta o core (mesmo objeto = DRY, não duplicação)."""
    from geosteering_ai.gui.services import stochastic_geology as core
    from geosteering_ai.simulation.tests import sm_model_gen as mono

    # mesmos OBJETOS (re-import, não cópia)
    assert mono.GenConfig is core.GenConfig
    assert mono.generate_models is core.generate_models
    assert mono.GENERATORS_AVAILABLE is core.GENERATORS_AVAILABLE
    assert mono._resolve_rng_seed is core._resolve_rng_seed
    # casca Qt fica no monólito
    assert mono.DEFAULT_GEN_CHUNK_SIZE == 500
    assert hasattr(mono.ModelGenerationThread, "seed_used")


def test_generate_models_bit_parity_both_entrypoints():
    """AC-2 — mesma seed → modelos IDÊNTICOS pelos dois caminhos de import."""
    from geosteering_ai.gui.services.stochastic_geology import GenConfig as GC_core
    from geosteering_ai.gui.services.stochastic_geology import (
        generate_models as gm_core,
    )
    from geosteering_ai.simulation.tests.sm_model_gen import generate_models as gm_mono

    cfg = GC_core(n_layers_min=3, n_layers_max=8, generator="sobol")
    assert gm_core(cfg, 12, rng_seed=99) == gm_mono(cfg, 12, rng_seed=99)


# ════════════════════════════════════════════════════════════════════════════
# AC-3 — geração estocástica via simulate_batch (shape + determinismo)  [Numba]
# ════════════════════════════════════════════════════════════════════════════
def test_run_simulation_stochastic_fixed_layers():
    """AC-3 — estocástico n_layers fixo: H6 correto, finito, determinístico (seed)."""
    import numpy as np

    from geosteering_ai.gui.services.sim_request import SimRequest, _run_simulation

    req = SimRequest(
        geology_mode="stochastic",
        n_layers_fixed=5,
        n_models=8,
        rng_seed=42,
        frequencies_hz=(20000.0,),
        backend="numba",
    )
    r1 = _run_simulation(req)
    r2 = _run_simulation(req)
    h6 = r1["H6"]
    assert h6.shape == (8, 1, 1, 10, 1, 9)  # (n_models, nTR, nAng, n_pos, nf, 9)
    assert np.all(np.isfinite(h6.view(np.float64)))
    assert np.array_equal(r1["H6"], r2["H6"])  # mesma seed → mesmo H6
    assert r1["info"]["n_groups"] == 1


# ════════════════════════════════════════════════════════════════════════════
# AC-4 — agrupamento por n_layers (variável) + reassembly de ordem  [Numba]
# ════════════════════════════════════════════════════════════════════════════
def test_run_simulation_stochastic_variable_layers_grouping():
    """AC-4 — n_layers variável → múltiplos grupos → H6 multi-config finito."""
    import numpy as np

    from geosteering_ai.gui.services.sim_request import (
        SimRequest,
        _generate_stochastic_models,
        _run_simulation,
    )

    req = SimRequest(
        geology_mode="stochastic",
        n_layers_min=3,
        n_layers_max=7,
        n_models=12,
        rng_seed=7,
        frequencies_hz=(20000.0, 40000.0),
        dip_degs=(0.0, 30.0),
        backend="numba",
    )
    models = _generate_stochastic_models(req)
    n_distinct = len({m["n_layers"] for m in models})
    r = _run_simulation(req)
    h6 = r["H6"]
    # nTR=1, nAng=2 (dips), nf=2 (freqs), n_pos=10 (dip0=0)
    assert h6.shape == (12, 1, 2, 10, 2, 9)
    assert np.all(np.isfinite(h6.view(np.float64)))
    assert r["info"]["n_groups"] == n_distinct  # 1 grupo por n_layers distinto


def test_simulate_grouped_preserves_original_order():
    """AC-4 — a linha i do batch == simulação do modelo i isolado (ordem preservada)."""
    import collections

    import numpy as np

    from geosteering_ai.gui.services.sim_request import (
        SimRequest,
        _compute_positions_z,
        _generate_stochastic_models,
        _simulate_grouped,
    )

    req = SimRequest(
        geology_mode="stochastic",
        n_layers_min=3,
        n_layers_max=7,
        n_models=12,
        rng_seed=7,
        frequencies_hz=(20000.0,),
        backend="numba",
    )
    models = _generate_stochastic_models(req)
    positions_z = _compute_positions_z(req)
    h6, _ = _simulate_grouped(models, positions_z, req)
    # escolhe um modelo de cada n_layers distinto e compara com a sim isolada
    by_layers = collections.defaultdict(list)
    for i, m in enumerate(models):
        by_layers[m["n_layers"]].append(i)
    for _n_layers, idxs in by_layers.items():
        idx = idxs[-1]
        single, _ = _simulate_grouped([models[idx]], positions_z, req)
        assert np.array_equal(h6[idx], single[0]), f"ordem quebrada no índice {idx}"


# ════════════════════════════════════════════════════════════════════════════
# AC-5 — VM puro: validação de geologia (espelha GenConfig.validate)
# ════════════════════════════════════════════════════════════════════════════
def test_vm_geology_defaults_valid():
    """AC-5 — os defaults de geologia do VM passam na validação."""
    vm = _make_vm()
    assert vm.validate() == []


def test_vm_geology_rejects_bad_params():
    """AC-5 — ρ inválido, gerador/distribuição desconhecidos, λ<1, n_layers<3 reprovam."""
    # ρ_min ≥ ρ_max
    vm = _make_vm()
    vm.rho_h_min, vm.rho_h_max = 100.0, 10.0
    assert any("ρ" in e for e in vm.validate())
    # gerador desconhecido
    vm = _make_vm()
    vm.generator = "inexistente"
    assert any("Gerador" in e for e in vm.validate())
    # distribuição inválida
    vm = _make_vm()
    vm.rho_h_distribution = "weibull"
    assert any("Distribuição" in e for e in vm.validate())
    # λ < 1 (anisotrópico)
    vm = _make_vm()
    vm.lambda_min = 0.5
    assert any("λ" in e for e in vm.validate())
    # n_layers fixo < 3
    vm = _make_vm()
    vm.n_layers_fixed = 2
    assert any("n_layers" in e for e in vm.validate())
    # n_layers range invertido
    vm = _make_vm()
    vm.n_layers_fixed = None
    vm.n_layers_min, vm.n_layers_max = 10, 5
    assert any("n_layers" in e for e in vm.validate())


def test_vm_fixed_mode_skips_geology_validation():
    """AC-5 — modo "fixed" ignora params de geologia (não valida o que não usa)."""
    vm = _make_vm()
    vm.geology_mode = "fixed"
    vm.generator = "lixo"  # seria inválido em stochastic
    vm.rho_h_min, vm.rho_h_max = 100.0, 1.0  # idem
    assert vm.validate() == []


def test_vm_run_builds_stochastic_request():
    """AC-5 — run() monta um SimRequest com TODOS os campos de geologia."""
    vm = _make_vm()
    vm.geology_mode = "stochastic"
    vm.generator = "halton"
    vm.rho_h_distribution = "uniform"
    vm.n_layers_fixed = None
    vm.n_layers_min, vm.n_layers_max = 4, 9
    vm.anisotropic = False
    vm.rng_seed = 123
    vm.run()
    req = vm._service.requests[-1]
    assert req.geology_mode == "stochastic"
    assert req.generator == "halton"
    assert req.rho_h_distribution == "uniform"
    assert req.n_layers_fixed is None
    assert (req.n_layers_min, req.n_layers_max) == (4, 9)
    assert req.anisotropic is False
    assert req.rng_seed == 123
    assert vm.status == "running"


def test_vm_defaults_to_random_seed():
    """AC-5 (KB-018 / v2.19) — o VM MVVM default p/ semente ALEATÓRIA (None).

    Paridade com o monólito (``test_simulation_parameters_seed`` assevera
    ``get_rng_seed() is None`` por default). Protege a camada MVVM contra a
    regressão KB-018 (``rng_seed=42`` hardcoded → ensembles idênticos a cada
    sessão) — cujo guard regex só varre o monólito, não ``apps/``.
    """
    vm = _make_vm()
    assert vm.rng_seed is None, "VM deve nascer com semente aleatória (None), não fixa."


# ════════════════════════════════════════════════════════════════════════════
# Correções de paridade/bugs (revisão SM MVVM ↔ monólito)
# ════════════════════════════════════════════════════════════════════════════
def test_vm_run_guards_reentrancy():
    """BUG B2 — run() é guardado contra re-entrância (1 simulação por vez).

    Com status="running" (1ª run despachada, sem finished), uma 2ª run() não
    despacha um SEGUNDO worker — o invariante vive na camada VM, não na View.
    """
    vm = _make_vm()
    vm.run()  # status → running; stub registra 1
    assert len(vm._service.requests) == 1
    assert vm.status == "running"
    vm.run()  # guard: status running → no-op
    assert len(vm._service.requests) == 1  # NÃO despachou de novo


def test_parse_csv_accepts_comma_and_semicolon():
    """G14 — _parse_csv_floats aceita ',' e ';' (paridade _parse_float_list)."""
    from apps.sim_manager.perspectives.simulation.view import _parse_csv_floats

    assert _parse_csv_floats("20000, 40000") == (20000.0, 40000.0)
    assert _parse_csv_floats("20000; 40000") == (20000.0, 40000.0)
    assert _parse_csv_floats("1, 2; 3") == (1.0, 2.0, 3.0)
    assert _parse_csv_floats("  ") == ()  # vazio → tupla vazia (VM valida depois)
    import pytest as _pytest

    with _pytest.raises(ValueError, match="não numérico"):
        _parse_csv_floats("1, abc")


@pytest.mark.gui
def test_view_n_layers_max_inclusive_conversion(qtbot):
    """BUG B1 — a View trata n_layers_max INCLUSIVE (+1 p/ o exclusive do VM/monólito).

    Init: VM exclusive=32 → spin mostra 31 (inclusive). Run: spin=11 (inclusive)
    → VM/SimRequest recebe 12 (exclusive), igual ao build_gen_config do monólito.
    """
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = (
        _make_vm()
    )  # _n_layers_max default = 32 (exclusive) — paridade monólito (item 3)
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    # init: spin inclusive = exclusive − 1
    assert view._geo_nl_max.value() == vm.n_layers_max - 1 == 31
    # run com spin inclusive = 11 + modo sampled → VM exclusive = 12
    view._radio_random.setChecked(
        True
    )  # Lote 2: radio Aleatória (era combo "stochastic")
    view._geo_nlf_check.setChecked(False)  # amostra [min, max)
    view._geo_nl_min.setValue(3)
    view._geo_nl_max.setValue(11)  # INCLUSIVE
    view._on_run_clicked()
    req = vm._service.requests[-1]
    assert (
        req.n_layers_max == 12
    ), "spin inclusive 11 → exclusive 12 (paridade monólito)"


@pytest.mark.gui
def test_view_geology_enable_by_radio(qtbot):
    """Lote 2 (Task 3) — 2 radios Aleatória/Manual; ambos funcionais (sem trava).

    No modo Manual os samplers ficam desabilitados e o editor de camadas habilitado;
    no modo Aleatória, o inverso. Substitui o antigo combo de 3 modos (cujo 'fixed'
    travava TODOS os campos — o bug reportado). O perfil canônico fica SEMPRE acessível.
    """
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_vm()
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    # Modo Manual: samplers OFF, editor + perfil canônico ON (nada trava).
    view._radio_manual.setChecked(True)
    assert not view._geo_generator.isEnabled()
    assert not view._geo_rho_min.isEnabled()
    assert not view._geo_seed.isEnabled()
    assert view._edit_layers_btn.isEnabled()
    assert view._apply_canonical_btn.isEnabled()  # perfil canônico SEMPRE acessível
    # Modo Aleatória: samplers ON, editor OFF.
    view._radio_random.setChecked(True)
    assert view._geo_generator.isEnabled()
    assert view._geo_rho_min.isEnabled()
    assert not view._edit_layers_btn.isEnabled()


@pytest.mark.gui
def test_view_backend_jax_ux_guard(qtbot):
    """Turn 8 — UX guard: escolher jax/auto com ensemble ragged mostra aviso honesto
    (1ª execução compila; Numba é mais rápido p/ ragged); n camadas fixo mostra a nota
    de geometria fixa; numba não alarma."""
    from apps.sim_manager.perspectives.simulation.view import SimulatorView

    vm = _make_vm()
    vm.n_layers_fixed = None  # ragged
    view = SimulatorView(vm)
    qtbot.addWidget(view)
    view._sim_backend.setCurrentText("jax")
    assert "ragged" in view._status.text().lower()
    # n camadas fixo → nota de geometria fixa (sem "ragged")
    vm.n_layers_fixed = 20
    view._sim_backend.setCurrentText("numba")  # reset
    view._sim_backend.setCurrentText("auto")
    assert (
        "fixo" in view._status.text().lower()
        and "ragged" not in view._status.text().lower()
    )
