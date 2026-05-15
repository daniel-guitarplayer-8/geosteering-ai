# -*- coding: utf-8 -*-
"""Sprint v2.37 F2 — Testes auto-detect tile_size.

Valida:
- `recommend_tile_size(n_pos)` retorna valores empiricamente otimizados.
- Campo `cfg.tile_size_auto` default True (ativado quando
  `use_tiled_positions=True`).
- Quando `tile_size_auto=True`, `cfg.tile_size` é IGNORADO.
- Paridade bit-exata: auto-tile == manual-tile com mesmo valor.
- Paridade bit-exata: auto-tile == legacy (não-tiled) em todos os
  perfis canônicos.

Contexto: v2.36 O2 validou que tile/block tem ganho profile-dependent.
v2.37 F2 introduz heurística que escolhe o tile ideal automaticamente,
permitindo promover `use_tiled_positions=True` como default em v2.38+.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation.config import SimulationConfig, recommend_tile_size
from geosteering_ai.simulation.forward import simulate

# ── 1. Heurística recommend_tile_size ─────────────────────────────────


@pytest.mark.parametrize(
    "n_pos, expected_tile",
    [
        (0, 2),  # caso degenerado: 0 ou negativo retorna mínimo
        (-10, 2),  # negativo (defensivo)
        (1, 2),  # 1 posição: tile=2 (mínimo)
        (16, 2),  # 16 posições: tile=2
        (64, 2),  # 64 posições: tile=2 (limite small)
        (65, 8),  # 65 posições: tile=8
        (100, 8),  # 100 posições: tile=8 (small canônico)
        (200, 8),  # 200 posições: tile=8 (medium canônico)
        (256, 8),  # 256 posições: tile=8
        (257, 8),  # 257 posições: tile=8 (heurística conservadora v2.37)
        (600, 8),  # 600 posições: tile=8 (large canônico, validado Intel i9)
        (10_000, 8),  # extra-large: tile=8 (cap)
    ],
)
def test_recommend_tile_size_thresholds(n_pos: int, expected_tile: int):
    """v2.37 F2 — heurística empírica `n_pos → tile_size`.

    Heurística conservadora v2.37: tile=8 para qualquer n_pos > 64
    (validado em Intel Core i9-9980HK + cross-referenciado com bench
    v2.36). tile=16 mostrou regressão −5% a −30% em vários perfis.
    """
    assert recommend_tile_size(n_pos) == expected_tile


def test_recommend_tile_size_within_valid_range():
    """v2.37 F2 — retorno sempre dentro do range aceito pelo validador."""
    for n in [0, 1, 50, 100, 200, 600, 10_000]:
        ts = recommend_tile_size(n)
        assert 1 <= ts <= 64, f"recommend_tile_size({n}) = {ts} fora de [1, 64]"


# ── 2. Config field validation ─────────────────────────────────────────


def test_tile_size_auto_default_true():
    """v2.37 F2 — `tile_size_auto` default True (heurística sempre ativa)."""
    cfg = SimulationConfig()
    assert cfg.tile_size_auto is True


def test_tile_size_auto_accepts_false():
    """v2.37 F2 — opt-out: setar False mantém tile_size manual."""
    cfg = SimulationConfig(tile_size_auto=False, tile_size=4)
    assert cfg.tile_size_auto is False
    assert cfg.tile_size == 4


# ── 3. Paridade bit-exata: auto-tile vs manual ─────────────────────────


def _model_canonical():
    """Modelo TIV 5-camadas para teste de paridade."""
    return {
        "rho_h": np.array([5.0, 50.0, 100.0, 50.0, 5.0]),
        "rho_v": np.array([5.0, 50.0, 200.0, 50.0, 5.0]),
        "esp": np.array([1.5, 2.0, 3.0]),
    }


@pytest.mark.parametrize("n_pos", [50, 100, 200, 400, 600])
def test_auto_tile_matches_manual_with_same_size(n_pos: int):
    """v2.37 F2 — auto-tile == manual com mesmo tile_size (paridade)."""
    m = _model_canonical()
    positions_z = np.linspace(-3.0, 3.0, n_pos).astype(np.float64)
    expected_tile = recommend_tile_size(n_pos)

    cfg_auto = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=True,
        tile_size_auto=True,
        parallel=True,
    )
    cfg_manual = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=True,
        tile_size_auto=False,
        tile_size=expected_tile,
        parallel=True,
    )

    r_auto = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        cfg=cfg_auto,
    )
    r_manual = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        cfg=cfg_manual,
    )

    assert np.array_equal(r_auto.H_tensor, r_manual.H_tensor), (
        f"Auto-tile (n_pos={n_pos}, recomendado tile={expected_tile}) "
        f"deve produzir tensor bit-exato vs manual com mesmo tile."
    )


# ── 4. Paridade auto-tile vs legacy não-tiled ─────────────────────────


@pytest.mark.parametrize("n_pos", [50, 100, 200, 400, 600])
def test_auto_tile_matches_non_tiled_legacy(n_pos: int):
    """v2.37 F2 — auto-tile preserva paridade vs caminho não-tiled.

    Crítico para promover `use_tiled_positions=True` como default em
    sprints futuras: a heurística NÃO PODE quebrar a paridade física.
    """
    m = _model_canonical()
    positions_z = np.linspace(-3.0, 3.0, n_pos).astype(np.float64)

    cfg_auto = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=True,
        tile_size_auto=True,
        parallel=True,
    )
    cfg_legacy = SimulationConfig(
        use_flat_prange=False,
        use_tiled_positions=False,
        parallel=True,
    )

    r_auto = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        cfg=cfg_auto,
    )
    r_legacy = simulate(
        rho_h=m["rho_h"],
        rho_v=m["rho_v"],
        esp=m["esp"],
        positions_z=positions_z,
        cfg=cfg_legacy,
    )

    assert np.array_equal(r_auto.H_tensor, r_legacy.H_tensor), (
        f"Paridade quebrada: auto-tile (n_pos={n_pos}) difere de legacy "
        f"não-tiled. Max diff = {np.abs(r_auto.H_tensor - r_legacy.H_tensor).max()}"
    )
