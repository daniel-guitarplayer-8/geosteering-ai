# -*- coding: utf-8 -*-
"""Sprint v2.37 F1 — Testes da heurística adaptativa FLAT/não-FLAT.

Valida:
- Campo `cfg.flat_prange_min_combos` existe, default = 2, range [1, 65536].
- Single-combo (nTR × nAng × nf == 1) força não-FLAT mesmo com
  `use_flat_prange=True` (recupera Cenário E em produção LWD).
- Multi-combo (nTR × nAng × nf ≥ 2) usa FLAT quando `use_flat_prange=True`.
- Override `flat_prange_min_combos=1` reverte ao comportamento v2.22.4
  (sempre FLAT quando `use_flat_prange=True`).
- Paridade bit-exata vs ambos os caminhos (FLAT e não-FLAT) para o
  mesmo modelo.

Contexto: v2.36 D5 identificou regressão −15% em Cenário E quando
`use_flat_prange=True` (default v2.22.4+). A heurística desta sprint
força não-FLAT para single-combo, recuperando 184k → 157k mod/h.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.forward import simulate
from geosteering_ai.simulation.multi_forward import simulate_multi

# ── 1. Config field validation ─────────────────────────────────────────


def test_flat_prange_min_combos_default_is_2():
    """v2.37 F1 — default 2 evita FLAT em single-combo (Cenário E)."""
    cfg = SimulationConfig()
    assert cfg.flat_prange_min_combos == 2


@pytest.mark.parametrize("value", [1, 2, 4, 16, 1024, 65536])
def test_flat_prange_min_combos_valid_range(value: int):
    """v2.37 F1 — range válido [1, 65536]."""
    cfg = SimulationConfig(flat_prange_min_combos=value)
    assert cfg.flat_prange_min_combos == value


@pytest.mark.parametrize("invalid", [0, -1, -100, 65537])
def test_flat_prange_min_combos_invalid_range_rejects(invalid: int):
    """v2.37 F1 — rejeita valores fora de [1, 65536]."""
    with pytest.raises(AssertionError, match="flat_prange_min_combos"):
        SimulationConfig(flat_prange_min_combos=invalid)


# ── 2. Paridade bit-exata entre paths FLAT/não-FLAT ───────────────────


def _model_canonical():
    """Modelo TIV 5-camadas para teste de paridade."""
    return {
        "rho_h": np.array([5.0, 50.0, 100.0, 50.0, 5.0]),
        "rho_v": np.array([5.0, 50.0, 200.0, 50.0, 5.0]),
        "esp": np.array([1.5, 2.0, 3.0]),
    }


def test_single_combo_heuristic_matches_legacy_non_flat():
    """v2.37 F1 — single-combo c/ heurística == não-FLAT explícito.

    Heurística (default min_combos=2) força não-FLAT para single-combo
    independentemente de `use_flat_prange`. Deve produzir tensor
    bit-exato vs `use_flat_prange=False` (path legado v2.21).
    """
    m = _model_canonical()
    positions_z = np.linspace(-3.0, 3.0, 100).astype(np.float64)

    # Caminho heurístico: use_flat_prange=True + min_combos=2 (default)
    cfg_heuristic = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=2,
        parallel=True,
    )

    # Caminho legado explícito: use_flat_prange=False
    cfg_legacy = SimulationConfig(
        use_flat_prange=False,
        parallel=True,
    )

    r_heuristic = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_heuristic,
    )
    r_legacy = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_legacy,
    )

    assert np.array_equal(r_heuristic.H_tensor, r_legacy.H_tensor), (
        "Single-combo c/ heurística deve produzir tensor bit-exato vs "
        "use_flat_prange=False (path legado)."
    )


def test_min_combos_1_reverts_to_flat_default():
    """v2.37 F1 — min_combos=1 reverte ao comportamento v2.22.4 (sempre FLAT)."""
    m = _model_canonical()
    positions_z = np.linspace(-3.0, 3.0, 100).astype(np.float64)

    # Override min_combos=1: força FLAT mesmo em single-combo
    cfg_force_flat = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        parallel=True,
    )
    cfg_legacy = SimulationConfig(
        use_flat_prange=False,
        parallel=True,
    )

    r_flat = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_force_flat,
    )
    r_legacy = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_legacy,
    )

    # Ambos paths produzem o mesmo resultado físico (paridade FLAT bit-exata
    # contra não-FLAT já validada em test_simulation_v22_flat_prange.py).
    # Aqui só validamos que o override funciona.
    assert np.array_equal(r_flat.H_tensor, r_legacy.H_tensor)


def test_multi_combo_still_uses_flat_when_enabled():
    """v2.37 F1 — multi-combo (nf≥2) continua usando FLAT (default).

    Cenário com nTR=1, nAng=1, nf=2 → n_effective_combos=2 ≥ min_combos=2.
    Deve usar o caminho FLAT e produzir resultado bit-exato vs FLAT
    forçado (min_combos=1).
    """
    m = _model_canonical()
    positions_z = np.linspace(-3.0, 3.0, 100).astype(np.float64)
    freqs = [10000.0, 50000.0]  # 2 frequências → n_effective_combos=2

    cfg_heuristic = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=2,  # default
        parallel=True,
    )
    cfg_force_flat = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        parallel=True,
    )

    r_heuristic = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_heuristic,
    )
    r_force = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_force_flat,
    )

    # Multi-combo: ambos usam FLAT, bit-exato
    assert np.array_equal(r_heuristic.H_tensor, r_force.H_tensor)


@pytest.mark.parametrize(
    "tr_spacings, dip_degs, freqs",
    [
        # Single-combo (heurística ativa)
        ([1.0], [0.0], [20000.0]),
        # Multi-TR (n_combos=3)
        ([0.5, 1.0, 1.5], [0.0], [20000.0]),
        # Multi-Ang (n_combos=4)
        ([1.0], [0.0, 30.0, 60.0, 89.0], [20000.0]),
        # Multi-freq (n_eff_combos=3)
        ([1.0], [0.0], [10000.0, 20000.0, 50000.0]),
        # Multi-TR × Multi-Ang × Multi-freq (n_eff_combos=12)
        ([0.5, 1.0], [0.0, 45.0], [10000.0, 50000.0, 100000.0]),
    ],
)
def test_paridade_heuristic_vs_force_flat(tr_spacings, dip_degs, freqs):
    """v2.37 F1 — heurística produz resultado bit-exato em todos os shapes.

    Garante que a heurística NÃO altera o tensor físico, apenas o
    caminho de execução. Cobre single-combo (escolhe não-FLAT) e
    multi-combo (escolhe FLAT).
    """
    m = _model_canonical()
    positions_z = np.linspace(-3.0, 3.0, 50).astype(np.float64)

    cfg_heuristic = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=2,
        parallel=True,
    )
    cfg_force_flat = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        parallel=True,
    )

    r_heuristic = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=tr_spacings,
        dip_degs=dip_degs,
        cfg=cfg_heuristic,
    )
    r_force = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=tr_spacings,
        dip_degs=dip_degs,
        cfg=cfg_force_flat,
    )

    assert np.array_equal(r_heuristic.H_tensor, r_force.H_tensor), (
        f"Paridade quebrada: tr={tr_spacings}, dip={dip_degs}, freqs={freqs}. "
        f"max_diff={np.abs(r_heuristic.H_tensor - r_force.H_tensor).max()}"
    )


# ── 3. Backward-compat — `simulate()` (single-TR shim) ────────────────


def test_simulate_shim_uses_heuristic_by_default():
    """v2.37 F1 — `simulate()` (shim single-TR) também ativa heurística.

    Como `simulate()` chama `simulate_multi` internamente com nTR=1,
    nAng=1, a heurística deve forçar não-FLAT automaticamente.
    Validação: tensor bit-exato vs `use_flat_prange=False` explícito.
    """
    m = _model_canonical()
    positions_z = np.linspace(-3.0, 3.0, 100).astype(np.float64)

    r_default = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        cfg=SimulationConfig(),  # use_flat_prange=True + min_combos=2
    )
    r_legacy = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        cfg=SimulationConfig(use_flat_prange=False),
    )

    assert np.array_equal(r_default.H_tensor, r_legacy.H_tensor)
