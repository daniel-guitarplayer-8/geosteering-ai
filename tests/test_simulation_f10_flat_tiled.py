# -*- coding: utf-8 -*-
"""Sprint F10 (v2.39) — Testes de paridade para _simulate_combined_prange_flat_tiled.

Valida:
- Paridade bit-exata: FLAT-tiled == FLAT não-tiled para tile_size ∈ {2,4,8,16}.
- Paridade bit-exata: FLAT-tiled == não-FLAT legado (transitividade Fortran <1e-12).
- Comportamento correto quando n_pos não é múltiplo de tile_size (última tile parcial).
- Cobertura de múltiplos cenários: single-freq, multi-freq, multi-TR, multi-dip.
- Configuração via `use_tiled_positions=True` + `use_flat_prange=True`.
- Fallback para FLAT não-tiled quando n_pos == 1 (guard against tile overhead).

Contexto: F10 implementa tile/block no hot path FLAT de produção
(`_simulate_combined_prange_flat`). Cada tarefa prange processa `tile_size`
posições consecutivas com as mesmas slices de cache quentes em L2,
reduzindo fetch de main memory ~tile_size× para os ~280 kB de cache por (ci, i_f).
"""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.multi_forward import simulate_multi

# ── Fixtures de modelos ────────────────────────────────────────────────────────


def _model_3layer():
    """Modelo TIV 3-camadas — Cenário E (single-TR, single-freq)."""
    return {
        "rho_h": np.array([2.0, 100.0, 2.0]),
        "rho_v": np.array([2.0, 200.0, 2.0]),
        "esp": np.array([2.0]),
    }


def _model_5layer():
    """Modelo TIV 5-camadas — Cenário B (multi-TR, single-freq)."""
    return {
        "rho_h": np.array([5.0, 50.0, 100.0, 50.0, 5.0]),
        "rho_v": np.array([5.0, 50.0, 200.0, 50.0, 5.0]),
        "esp": np.array([1.5, 2.0, 3.0]),
    }


def _model_7layer():
    """Modelo TIV 7-camadas — Cenário multi-dip/multi-freq."""
    return {
        "rho_h": np.array([1.0, 10.0, 100.0, 500.0, 100.0, 10.0, 1.0]),
        "rho_v": np.array([1.0, 10.0, 200.0, 1000.0, 200.0, 10.0, 1.0]),
        "esp": np.array([1.0, 1.5, 2.0, 1.5, 1.0]),
    }


# ── 1. Paridade FLAT-tiled vs FLAT não-tiled ──────────────────────────────────


@pytest.mark.parametrize("tile_size", [2, 4, 8, 16])
def test_flat_tiled_matches_flat_single_freq(tile_size: int):
    """F10 — paridade bit-exata FLAT-tiled vs FLAT para single-freq."""
    m = _model_5layer()
    positions_z = np.linspace(-5.0, 5.0, 120, dtype=np.float64)

    cfg_tiled = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,  # garante FLAT mesmo single-combo
        use_tiled_positions=True,
        tile_size=tile_size,
        tile_size_auto=False,
        parallel=True,
    )
    cfg_flat = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=False,
        parallel=True,
    )

    r_tiled = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_tiled,
    )
    r_flat = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_flat,
    )

    assert np.array_equal(
        r_tiled.H_tensor, r_flat.H_tensor
    ), f"tile_size={tile_size}: FLAT-tiled deve ser bit-exato vs FLAT não-tiled."


@pytest.mark.parametrize("tile_size", [2, 4, 8, 16])
def test_flat_tiled_matches_flat_multi_freq(tile_size: int):
    """F10 — paridade bit-exata FLAT-tiled vs FLAT para multi-freq (nf=3)."""
    m = _model_5layer()
    positions_z = np.linspace(-4.0, 4.0, 80, dtype=np.float64)

    freqs = [5000.0, 20000.0, 100000.0]

    cfg_tiled = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=True,
        tile_size=tile_size,
        tile_size_auto=False,
        parallel=True,
    )
    cfg_flat = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=False,
        parallel=True,
    )

    r_tiled = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_tiled,
    )
    r_flat = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=freqs,
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_flat,
    )

    assert np.array_equal(
        r_tiled.H_tensor, r_flat.H_tensor
    ), f"tile_size={tile_size}, multi-freq: FLAT-tiled deve ser bit-exato vs FLAT."


@pytest.mark.parametrize("tile_size", [4, 8])
def test_flat_tiled_matches_flat_multi_tr_multi_dip(tile_size: int):
    """F10 — paridade bit-exata FLAT-tiled vs FLAT para multi-TR × multi-dip."""
    m = _model_7layer()
    positions_z = np.linspace(-6.0, 6.0, 100, dtype=np.float64)

    cfg_tiled = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=True,
        tile_size=tile_size,
        tile_size_auto=False,
        parallel=True,
    )
    cfg_flat = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=False,
        parallel=True,
    )

    kwargs = dict(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[0.5, 1.0, 2.0],
        dip_degs=[0.0, 30.0, 60.0],
    )

    r_tiled = simulate_multi(**kwargs, cfg=cfg_tiled)
    r_flat = simulate_multi(**kwargs, cfg=cfg_flat)

    assert np.array_equal(
        r_tiled.H_tensor, r_flat.H_tensor
    ), f"tile_size={tile_size}, multi-TR×dip: FLAT-tiled deve ser bit-exato vs FLAT."


# ── 2. Paridade FLAT-tiled vs não-FLAT legado (transitividade <1e-12) ─────────


def test_flat_tiled_matches_legacy_non_flat():
    """F10 — FLAT-tiled bit-exato vs não-FLAT legado (Sprint 13.3).

    Por transitividade, garante paridade com Fortran <1e-12
    (FLAT-tiled == FLAT == não-FLAT == Fortran).
    """
    m = _model_5layer()
    positions_z = np.linspace(-5.0, 5.0, 90, dtype=np.float64)

    cfg_tiled = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=True,
        tile_size=8,
        tile_size_auto=False,
        parallel=True,
    )
    cfg_legacy = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=False,
        parallel=True,
    )

    r_tiled = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_tiled,
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

    assert np.array_equal(
        r_tiled.H_tensor, r_legacy.H_tensor
    ), "FLAT-tiled (F10) deve ser bit-exato vs não-FLAT legado (Sprint 13.3)."


# ── 3. Borda: n_pos não múltiplo de tile_size ─────────────────────────────────


@pytest.mark.parametrize(
    "n_pos,tile_size",
    [
        (7, 4),  # 7 = 1×4 + 3 (última tile parcial de 3)
        (9, 4),  # 9 = 2×4 + 1 (última tile parcial de 1)
        (13, 8),  # 13 = 1×8 + 5 (última tile parcial de 5)
        (1, 4),  # n_pos=1: tile inteiro é 1 posição
    ],
)
def test_partial_tile_at_boundary(n_pos: int, tile_size: int):
    """F10 — última tile parcial (n_pos % tile_size != 0) produz tensor correto."""
    m = _model_3layer()
    positions_z = np.linspace(-2.0, 2.0, n_pos, dtype=np.float64)

    cfg_tiled = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=True,
        tile_size=tile_size,
        tile_size_auto=False,
        parallel=True,
    )
    cfg_flat = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=False,
        parallel=True,
    )

    r_tiled = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_tiled,
    )
    r_flat = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_flat,
    )

    assert np.array_equal(
        r_tiled.H_tensor, r_flat.H_tensor
    ), f"n_pos={n_pos}, tile_size={tile_size}: tile parcial deve ser bit-exato."


# ── 4. tile_size_auto seleciona tile correto pelo recommend_tile_size ─────────


def test_tile_size_auto_activates_in_flat_path():
    """F10 — tile_size_auto=True usa recommend_tile_size() no path FLAT."""
    m = _model_5layer()
    positions_z = np.linspace(-5.0, 5.0, 120, dtype=np.float64)

    cfg_auto = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=True,
        tile_size_auto=True,  # heurística: n=120 > 64 → tile=8
        parallel=True,
    )
    cfg_flat = SimulationConfig(
        use_flat_prange=True,
        flat_prange_min_combos=1,
        use_tiled_positions=False,
        parallel=True,
    )

    r_auto = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_auto,
    )
    r_flat = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_flat,
    )

    assert np.array_equal(
        r_auto.H_tensor, r_flat.H_tensor
    ), "tile_size_auto=True no path FLAT deve produzir tensor bit-exato vs FLAT padrão."


# ── 5. Guard: use_flat_prange=False ignora use_tiled_positions no FLAT path ───


def test_non_flat_path_unaffected_by_flat_tiled():
    """F10 — use_flat_prange=False ignora use_tiled_positions (path não-FLAT inalterado)."""
    m = _model_5layer()
    positions_z = np.linspace(-4.0, 4.0, 60, dtype=np.float64)

    # use_tiled_positions=True no path não-FLAT usa _simulate_positions_njit_cached_tiled
    cfg_non_flat_tiled = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=True,
        tile_size=8,
        tile_size_auto=False,
        parallel=True,
    )
    cfg_non_flat = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=False,
        parallel=True,
    )

    r_tiled = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_non_flat_tiled,
    )
    r_non_flat = simulate_multi(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg_non_flat,
    )

    assert np.array_equal(
        r_tiled.H_tensor, r_non_flat.H_tensor
    ), "Caminho não-FLAT com tile deve ser bit-exato vs não-FLAT sem tile (v2.36 O2)."
