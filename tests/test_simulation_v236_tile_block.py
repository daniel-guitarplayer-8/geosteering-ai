# -*- coding: utf-8 -*-
"""Sprint v2.36 O2 — Testes paridade tile/block vs legacy.

Valida:
- ``_simulate_positions_njit_cached_tiled`` produz tensor bit-exato vs
  ``_simulate_positions_njit_cached`` para múltiplos tile_size.
- Config field ``tile_size`` valida range [1, 64].
- Config field ``use_tiled_positions`` é bool default False.
- Dispatcher em ``multi_forward.simulate_multi`` honra ``cfg.use_tiled_positions``
  + ``cfg.tile_size`` quando ``use_flat_prange=False``.

Contexto: O2 é prova de conceito do tiling — só ativa no path single-TR
legado (não-FLAT). FLAT prange (default desde v2.22.4) permanece intocado.
Critério de aceite: ``np.array_equal`` entre tensor tiled e tensor não-tiled
para 4 tile_sizes × 3 perfis = 12 combinações.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.forward import simulate

# ── 1. Config field validation ─────────────────────────────────────────


def test_use_tiled_positions_default_false():
    """v2.36 O2 — `use_tiled_positions` default = False (opt-in)."""
    cfg = SimulationConfig()
    assert cfg.use_tiled_positions is False


def test_tile_size_default_4():
    """v2.36 O2 — `tile_size` default = 4 (cobre L1 típico)."""
    cfg = SimulationConfig()
    assert cfg.tile_size == 4


@pytest.mark.parametrize("tile_size", [1, 2, 4, 8, 16, 32, 64])
def test_tile_size_valid_range(tile_size: int):
    """v2.36 O2 — tile_size aceita range [1, 64]."""
    cfg = SimulationConfig(use_tiled_positions=True, tile_size=tile_size)
    assert cfg.tile_size == tile_size


@pytest.mark.parametrize("invalid_tile_size", [0, -1, 65, 1000])
def test_tile_size_invalid_range_rejects(invalid_tile_size: int):
    """v2.36 O2 — tile_size fora de [1, 64] rejeita via assert."""
    with pytest.raises(AssertionError, match="tile_size"):
        SimulationConfig(use_tiled_positions=True, tile_size=invalid_tile_size)


# ── 2. Paridade tile vs legacy (gate de aceitação O2) ────────────────


def _model_simple_3layer():
    """Perfil canônico simples para teste de paridade."""
    return {
        "rho_h": np.array([10.0, 50.0, 10.0]),
        "rho_v": np.array([10.0, 50.0, 10.0]),
        "esp": np.array([5.0]),  # 1 camada interna (n=3 totais)
        "positions_z": np.linspace(-4.0, 4.0, 32),
    }


def _model_inv0dip_600pts():
    """Perfil Inv0Dip estendido (cenário E baseline)."""
    return {
        "rho_h": np.array([10.0, 100.0, 5.0]),
        "rho_v": np.array([10.0, 100.0, 5.0]),
        "esp": np.array([2.0]),
        "positions_z": np.linspace(-5.0, 5.0, 100),
    }


def _model_oklahoma_like():
    """Perfil oklahoma-like (5 camadas, anisotrópico)."""
    return {
        "rho_h": np.array([10.0, 30.0, 100.0, 30.0, 10.0]),
        "rho_v": np.array([15.0, 50.0, 100.0, 50.0, 15.0]),
        "esp": np.array([2.0, 3.0, 2.0]),
        "positions_z": np.linspace(-6.0, 6.0, 64),
    }


_MODELS = {
    "simple_3layer": _model_simple_3layer,
    "inv0dip_600pts": _model_inv0dip_600pts,
    "oklahoma_like": _model_oklahoma_like,
}


@pytest.mark.parametrize("tile_size", [1, 2, 4, 8])
@pytest.mark.parametrize("model_name", list(_MODELS.keys()))
def test_tile_block_parity_vs_legacy(tile_size: int, model_name: str):
    """v2.36 O2 — tensor tiled == tensor legacy bit-exato.

    Critério crítico da Sprint v2.36 O2: se a paridade quebrar para
    qualquer (tile_size, modelo), o tile/block NÃO pode ser promovido.
    """
    m = _MODELS[model_name]()

    # Path single-TR legado (use_flat_prange=False) — onde tile/block atua
    cfg_legacy = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=False,
        parallel=True,
    )
    cfg_tiled = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=True,
        tile_size=tile_size,
        parallel=True,
    )

    r_legacy = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg_legacy,
    )
    r_tiled = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg_tiled,
    )

    # Paridade bit-exata: tensor 9-comp idêntico
    assert np.array_equal(r_legacy.H_tensor, r_tiled.H_tensor), (
        f"Paridade tile/block QUEBRADA em {model_name}, tile_size={tile_size}. "
        f"Tensor tiled difere do legado — diff_max={np.abs(r_legacy.H_tensor - r_tiled.H_tensor).max()}."
    )
    assert np.array_equal(r_legacy.z_obs, r_tiled.z_obs)
    assert np.array_equal(r_legacy.rho_h_at_obs, r_tiled.rho_h_at_obs)
    assert np.array_equal(r_legacy.rho_v_at_obs, r_tiled.rho_v_at_obs)


def test_tile_size_equals_n_positions_no_remainder():
    """v2.36 O2 — tile_size = n_pos processa tudo em 1 tile (caso limite)."""
    m = _model_simple_3layer()
    n_pos = len(m["positions_z"])  # 32

    cfg_legacy = SimulationConfig(use_flat_prange=False, use_tiled_positions=False)
    cfg_tiled = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=True,
        tile_size=min(n_pos, 64),
    )

    r_legacy = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg_legacy,
    )
    r_tiled = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg_tiled,
    )

    assert np.array_equal(r_legacy.H_tensor, r_tiled.H_tensor)


def test_tile_size_with_remainder_handles_correctly():
    """v2.36 O2 — n_pos não divisível por tile_size processa resto."""
    m = _model_simple_3layer()
    n_pos = len(m["positions_z"])  # 32 — não divisível por 5
    tile_size = 5

    cfg_legacy = SimulationConfig(use_flat_prange=False, use_tiled_positions=False)
    cfg_tiled = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=True,
        tile_size=tile_size,
    )

    r_legacy = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg_legacy,
    )
    r_tiled = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=m["positions_z"],
        cfg=cfg_tiled,
    )

    assert (
        n_pos % tile_size != 0
    ), "Setup inválido: n_pos deve não ser múltiplo de tile_size"
    assert np.array_equal(r_legacy.H_tensor, r_tiled.H_tensor), (
        f"Tile com resto (n_pos={n_pos}, tile_size={tile_size}) "
        f"não preserva paridade."
    )
