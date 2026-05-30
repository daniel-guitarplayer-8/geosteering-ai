# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_dataset_export.py                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Sprint v2.45 — geração de dataset .dat no fluxo batched    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-29                                                 ║
# ║  Status      : Produção (gate de merge)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes Sprint v2.45 — `export_per_model` em `simulate_multi_jax_batched`.

Valida a geração de dataset .dat (formato Fortran 22-col) a partir do fluxo
batched JAX: `cfg.export_per_model=True` → 1 conjunto de .dat por modelo do
batch, reusando o writer validado `export_multi_tr_dat`. Cobre o campo
`H_tilted=None` (compat com `export_info_out`) e a validação de config.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX não instalado")

from geosteering_ai.simulation import SimulationConfig  # noqa: E402
from geosteering_ai.simulation._jax.multi_forward import (  # noqa: E402
    MultiSimulationResultBatchedJAX,
    MultiSimulationResultJAX,
    simulate_multi_jax_batched,
)


def _shared_geom_batch(n_models: int, n: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    rho_h = rng.uniform(1.0, 100.0, size=(n_models, n))
    rho_v = rho_h.copy()
    esp = np.tile(np.array([6.0]), (n_models, 1))
    return rho_h, rho_v, esp


def test_h_tilted_field_present_on_jax_results():
    """Ambos os dataclasses JAX expõem H_tilted=None (compat writers .dat)."""
    rho_h, rho_v, esp = _shared_geom_batch(2)
    res = simulate_multi_jax_batched(
        rho_h,
        rho_v,
        esp,
        np.linspace(-5, 5, 10),
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=SimulationConfig(backend="jax", jax_strategy="bucketed"),
    )
    assert isinstance(res, MultiSimulationResultBatchedJAX)
    assert res.H_tilted is None
    single = res.get_model(0)
    assert isinstance(single, MultiSimulationResultJAX)
    assert single.H_tilted is None


def test_export_per_model_config_validation():
    """export_per_model exige output_filename não-vazio (como export_binary_dat)."""
    SimulationConfig(backend="jax", export_per_model=True, output_filename="ds")
    with pytest.raises(AssertionError):
        SimulationConfig(backend="jax", export_per_model=True, output_filename="")


def test_export_per_model_generates_dat(tmp_path):
    """batched + export_per_model=True → N conjuntos de .dat + round-trip 22-col."""
    from geosteering_ai.simulation.validation.compare_fortran import (
        read_fortran_dat_22col,
    )

    n_models, n_pos = 3, 20
    rho_h, rho_v, esp = _shared_geom_batch(n_models, seed=1)
    cfg = SimulationConfig(
        backend="jax",
        jax_strategy="bucketed",
        export_per_model=True,
        output_dir=str(tmp_path),
        output_filename="dataset",
    )
    res = simulate_multi_jax_batched(
        rho_h,
        rho_v,
        esp,
        np.linspace(-5, 5, n_pos),
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg,
    )
    assert res.n_models == n_models

    # 1 .dat por modelo (nTR=1 → sem sufixo _TR). info{...}.out também é gerado.
    dats = sorted(p for p in os.listdir(tmp_path) if p.endswith(".dat"))
    assert len(dats) == n_models, f"esperado {n_models} .dat, obtido {dats}"
    assert any("model000000" in d for d in dats)

    # Round-trip: cada .dat tem nAng×nf×n_pos = 1×1×20 = 20 registros (172 bytes).
    for d in dats:
        full = os.path.join(tmp_path, d)
        assert os.path.getsize(full) == n_pos * 172
        rec = read_fortran_dat_22col(full)
        assert rec.shape[0] == n_pos
        assert "Re_Hxx" in rec.dtype.names


def test_export_per_model_multi_tr(tmp_path):
    """Multi-TR: cada modelo gera nTR arquivos _TR{j}.dat."""
    n_models, n_pos, n_tr = 2, 15, 2
    rho_h, rho_v, esp = _shared_geom_batch(n_models, seed=2)
    cfg = SimulationConfig(
        backend="jax",
        jax_strategy="bucketed",
        export_per_model=True,
        output_dir=str(tmp_path),
        output_filename="ds",
    )
    simulate_multi_jax_batched(
        rho_h,
        rho_v,
        esp,
        np.linspace(-5, 5, n_pos),
        frequencies_hz=[20000.0],
        tr_spacings_m=[0.5, 1.0],
        dip_degs=[0.0],
        cfg=cfg,
    )
    dats = [p for p in os.listdir(tmp_path) if p.endswith(".dat")]
    # n_models × nTR arquivos (_TR1, _TR2 por modelo).
    assert len(dats) == n_models * n_tr, f"esperado {n_models * n_tr}, obtido {dats}"


def test_export_per_model_default_off(tmp_path):
    """Default export_per_model=False → nenhum .dat (sem I/O)."""
    rho_h, rho_v, esp = _shared_geom_batch(2)
    simulate_multi_jax_batched(
        rho_h,
        rho_v,
        esp,
        np.linspace(-5, 5, 10),
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=SimulationConfig(backend="jax", output_dir=str(tmp_path)),
    )
    assert [p for p in os.listdir(tmp_path) if p.endswith(".dat")] == []
