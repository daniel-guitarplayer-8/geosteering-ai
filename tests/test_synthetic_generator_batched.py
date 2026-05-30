# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_synthetic_generator_batched.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Geração de dataset no caminho BATCHED JAX (correção        ║
# ║                da lacuna: loop por-modelo → batched + agrupamento geom)    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-30                                                 ║
# ║  Status      : Produção (gate de merge)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes — `simulate_multi_jax_batched_grouped` + `SyntheticDataGenerator` batched.

Cobre a correção da lacuna de produtização do treino:
  (1) helper de agrupamento por geometria — paridade vs serial/direto (<1e-13);
  (2) generate_batch usa o caminho batched (não loop por-modelo);
  (3) multi-config (freq×dip×TR);
  (4) n_geometries → geometria compartilhada (batchável);
  (5) paridade cross-backend JAX vs Numba (<1e-9).
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

jax_only = pytest.mark.skipif(not HAS_JAX, reason="JAX não instalado")


# ─────────────────────────────────────────────────────────────────────────────
# (1) Helper de agrupamento por geometria — paridade
# ─────────────────────────────────────────────────────────────────────────────
@jax_only
def test_grouped_homogeneous_matches_direct():
    """esp idêntico → 1 grupo → bit-exato vs simulate_multi_jax_batched direto."""
    from geosteering_ai.simulation import (
        SimulationConfig,
        simulate_multi_jax_batched,
        simulate_multi_jax_batched_grouped,
    )

    rng = np.random.default_rng(0)
    n, npos = 5, 30
    pz = np.linspace(-2, 12, npos)
    rho_h = rng.uniform(1, 100, (6, n))
    rho_v = rho_h.copy()
    esp = np.tile(np.array([5.0, 3.0, 2.0]), (6, 1))
    cfg = SimulationConfig(backend="jax", jax_strategy="bucketed", dtype="complex128")
    kw = dict(frequencies_hz=[20000.0, 50000.0], tr_spacings_m=[1.0], dip_degs=[0.0])

    H_direct = simulate_multi_jax_batched(rho_h, rho_v, esp, pz, cfg=cfg, **kw).H_tensor
    H_grp, info = simulate_multi_jax_batched_grouped(
        rho_h, rho_v, esp, pz, cfg=cfg, **kw
    )

    assert info["n_groups"] == 1
    assert np.abs(np.asarray(H_direct) - H_grp).max() < 1e-13


@jax_only
def test_grouped_heterogeneous_matches_per_model():
    """esp variável → cada modelo bate com simulate_multi_jax (1 modelo) — <1e-13."""
    from geosteering_ai.simulation import (
        SimulationConfig,
        simulate_multi_jax,
        simulate_multi_jax_batched_grouped,
    )

    rng = np.random.default_rng(1)
    n, npos = 5, 24
    pz = np.linspace(-2, 12, npos)
    rho_h = rng.uniform(1, 100, (5, n))
    rho_v = rho_h.copy()
    esp = np.array([5.0, 3.0, 2.0])[None, :] * rng.uniform(0.8, 1.2, (5, 3))
    cfg = SimulationConfig(backend="jax", jax_strategy="bucketed", dtype="complex128")
    kw = dict(frequencies_hz=[20000.0], tr_spacings_m=[0.5, 1.0], dip_degs=[0.0, 30.0])

    H_grp, info = simulate_multi_jax_batched_grouped(
        rho_h, rho_v, esp, pz, cfg=cfg, **kw
    )
    assert info["n_groups"] == 5  # todas distintas
    assert H_grp.shape == (5, 2, 2, npos, 1, 9)
    worst = 0.0
    for i in range(5):
        r = simulate_multi_jax(rho_h[i], rho_v[i], esp[i], pz, cfg=cfg, **kw)
        worst = max(worst, float(np.abs(np.asarray(r.H_tensor) - H_grp[i]).max()))
    assert worst < 1e-13, f"max|Δ|={worst:.2e}"


@jax_only
def test_grouped_preserves_model_order():
    """Modelos com geometrias intercaladas voltam na ORDEM ORIGINAL."""
    from geosteering_ai.simulation import (
        SimulationConfig,
        simulate_multi_jax,
        simulate_multi_jax_batched_grouped,
    )

    rng = np.random.default_rng(2)
    n, npos = 4, 20
    pz = np.linspace(-1, 10, npos)
    rho_h = rng.uniform(1, 100, (6, n))
    rho_v = rho_h.copy()
    # 2 geometrias intercaladas: A,B,A,B,A,B
    espA = np.array([4.0, 2.0])
    espB = np.array([6.0, 3.0])
    esp = np.stack([espA, espB, espA, espB, espA, espB])
    cfg = SimulationConfig(backend="jax", jax_strategy="bucketed", dtype="complex128")
    kw = dict(frequencies_hz=[20000.0], tr_spacings_m=[1.0], dip_degs=[0.0])

    H_grp, info = simulate_multi_jax_batched_grouped(
        rho_h, rho_v, esp, pz, cfg=cfg, **kw
    )
    assert info["n_groups"] == 2
    assert sorted(info["group_sizes"]) == [3, 3]
    for i in range(6):
        r = simulate_multi_jax(rho_h[i], rho_v[i], esp[i], pz, cfg=cfg, **kw)
        assert np.abs(np.asarray(r.H_tensor) - H_grp[i]).max() < 1e-13


# ─────────────────────────────────────────────────────────────────────────────
# (2-4) generate_batch — batched, multi-config, n_geometries
# ─────────────────────────────────────────────────────────────────────────────
@jax_only
def test_generate_batch_jax_backward_compat_shape():
    """Single-config JAX → (n_models, n_pos, nf, 9) + foi pelo caminho batched."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(PipelineConfig(simulator_backend="jax"))
    b = gen.generate_batch(n_models=4, n_positions=20, n_layers=3, seed=7)
    assert b.H_tensor.shape == (4, 20, 1, 9)
    assert b.dat_22col.shape == (4 * 20,)
    assert b.metadata["backend"] == "jax"
    # Prova que passou pelo caminho batched/agrupado (não loop por-modelo).
    assert b.metadata["n_geometry_groups"] is not None
    assert b.metadata["n_configs"] == 1


@jax_only
def test_generate_batch_multiconfig_shape():
    """Multi-config JAX → (n_models, nTR, nAng, n_pos, nf, 9)."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(PipelineConfig(simulator_backend="jax"))
    b = gen.generate_batch(
        n_models=3,
        n_positions=15,
        n_layers=3,
        seed=7,
        frequencies_hz=[20000.0, 50000.0, 100000.0],
        dip_degs=[0.0, 30.0, 60.0],
        tr_spacings_m=[0.5, 1.0],
    )
    assert b.H_tensor.shape == (
        3,
        2,
        3,
        15,
        3,
        9,
    )  # (n_models, nTR, nAng, n_pos, nf, 9)
    assert b.metadata["n_configs"] == 3 * 3 * 2
    assert b.metadata["config_shape"] == {"nf": 3, "nTR": 2, "nAng": 3}


@jax_only
def test_generate_batch_n_geometries_groups():
    """n_geometries=K compartilha geometria → exatamente K grupos batcháveis."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(PipelineConfig(simulator_backend="jax"))
    b = gen.generate_batch(
        n_models=12, n_positions=15, n_layers=4, n_geometries=3, seed=9
    )
    assert b.metadata["n_geometry_groups"] == 3  # 12 modelos, 3 geometrias
    # Confirma que esp realmente se repete (3 distintos entre 12).
    uniq = {b.esp[i].tobytes() for i in range(12)}
    assert len(uniq) == 3


@jax_only
def test_generate_batch_jax_chunked_matches_unchunked():
    """jax_chunk_size_models só limita VRAM — valores idênticos ao não-chunked."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(PipelineConfig(simulator_backend="jax"))
    kw = dict(n_models=6, n_positions=15, n_layers=3, n_geometries=1, seed=11)
    b_full = gen.generate_batch(**kw)
    b_chunk = gen.generate_batch(**kw, jax_chunk_size_models=2)
    assert b_full.metadata["n_geometry_groups"] == 1  # 1 geometria → 1 grupo grande
    assert np.abs(b_full.H_tensor - b_chunk.H_tensor).max() < 1e-13


def test_generate_batch_grid_covers_thickest_model():
    """Grid positions_z cobre o modelo MAIS espesso do batch (não só o modelo 0)."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    # Numba (sem GPU) — testa só a lógica de grid, independe do backend.
    gen = SyntheticDataGenerator(PipelineConfig(simulator_backend="numba"))
    b = gen.generate_batch(
        n_models=8, n_positions=30, n_layers=4, thickness_range=(1.0, 20.0), seed=5
    )
    max_total_thick = float(b.esp.sum(axis=1).max())
    # O topo do grid deve cobrir a espessura total do modelo mais espesso.
    assert b.positions_z.max() >= max_total_thick


@jax_only
def test_generate_batch_default_is_templates():
    """Sprint A: default geometry_mode='templates' → grupos < n_models (batchável)."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(PipelineConfig(simulator_backend="jax"))
    b = gen.generate_batch(n_models=64, n_positions=8, n_layers=4, seed=3)
    assert b.metadata["geometry_mode"] == "templates"
    # K = max(1, 64//32) = 2 → 2 geometrias distintas (muito < 64).
    assert b.metadata["n_geometry_groups"] == 2
    assert b.metadata["backend"] == "jax"  # batchável → fica no JAX


@jax_only
def test_geometry_mode_quantize_groups():
    """geometry_mode='quantize' → esp arredondado compartilha → < n_models grupos."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(PipelineConfig(simulator_backend="jax"))
    b = gen.generate_batch(
        n_models=40,
        n_positions=8,
        n_layers=4,
        geometry_mode="quantize",
        quantize_step=3.0,
        numba_fallback=False,
        seed=5,
    )
    assert b.metadata["geometry_mode"] == "quantize"
    assert b.metadata["n_geometry_groups"] < 40  # quantização agrupa


@jax_only
def test_per_model_triggers_numba_fallback():
    """geometry_mode='per_model' + jax → não-agrupável → fallback automático Numba."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(PipelineConfig(simulator_backend="jax"))
    b = gen.generate_batch(
        n_models=6, n_positions=10, n_layers=4, geometry_mode="per_model", seed=3
    )
    assert b.metadata["n_geometry_groups"] == 6  # cada modelo único
    assert b.metadata["backend"] == "numba"  # caiu p/ Numba (não-agrupável)
    assert b.metadata["fallback"] is not None


@jax_only
def test_numba_fallback_disabled_stays_jax():
    """numba_fallback=False força JAX-grouped mesmo degenerado (per_model)."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator(PipelineConfig(simulator_backend="jax"))
    b = gen.generate_batch(
        n_models=4,
        n_positions=8,
        n_layers=4,
        geometry_mode="per_model",
        numba_fallback=False,
        seed=3,
    )
    assert b.metadata["backend"] == "jax"
    assert b.metadata["fallback"] is None


def test_group_by_geometry_helper():
    """group_by_geometry: partição por esp.tobytes(), ordem 1ª ocorrência, cobertura."""
    import numpy as np

    from geosteering_ai.simulation import group_by_geometry

    esp = np.array([[5.0, 3.0], [6.0, 2.0], [5.0, 3.0], [7.0, 1.0]])  # A,B,A,C
    g = group_by_geometry(esp)
    assert [list(x) for x in g] == [[0, 2], [1], [3]]
    # esp vazio (n<=2) → 1 grupo trivial
    assert len(group_by_geometry(np.zeros((3, 0)))) == 1
    # cobertura sem repetição
    assert sorted(np.concatenate(g).tolist()) == [0, 1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# (5) Paridade cross-backend JAX vs Numba (invariante <1e-9 do projeto)
# ─────────────────────────────────────────────────────────────────────────────
@jax_only
def test_generate_batch_jax_vs_numba_parity():
    """Mesmos modelos (mesmo seed) → H idêntico JAX vs Numba (<1e-9)."""
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    kw = dict(
        n_models=4,
        n_positions=20,
        n_layers=3,
        seed=123,
        frequencies_hz=[20000.0, 50000.0],
        dip_degs=[0.0, 30.0],
        tr_spacings_m=[1.0],
    )
    b_jax = SyntheticDataGenerator(
        PipelineConfig(simulator_backend="jax")
    ).generate_batch(**kw)
    b_num = SyntheticDataGenerator(
        PipelineConfig(simulator_backend="numba")
    ).generate_batch(**kw)

    # Mesmo seed → mesmos modelos amostrados.
    assert np.allclose(b_jax.rho_h, b_num.rho_h)
    assert np.allclose(b_jax.esp, b_num.esp)
    # Paridade física do tensor H entre backends (invariante do projeto <1e-10).
    diff = np.abs(b_jax.H_tensor - b_num.H_tensor).max()
    assert diff < 1e-10, f"JAX vs Numba max|Δ|={diff:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# (D4) to_feature_dataset — extração de features compacta (FV/GS)
# ─────────────────────────────────────────────────────────────────────────────
def _numba_gen():
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    return SyntheticDataGenerator(PipelineConfig(simulator_backend="numba"))


def test_to_feature_dataset_raw_compact():
    """Raw: H_tensor (…,9 complexas) → X=[z,Re/Im Hxx,Re/Im Hzz] (5) + y (2)."""
    gen = _numba_gen()
    b = gen.generate_batch(n_models=4, n_positions=12, n_layers=3, seed=1)
    fd = gen.to_feature_dataset(b)  # apply_transforms=False (default)
    assert fd.X.shape == (4, 12, 5) and fd.y.shape == (4, 12, 2)
    assert fd.X.dtype == np.float32
    assert fd.metadata["feature_view"] == "raw"
    # col0 = z_obs (positions_z); col1 = Re(Hxx) = H[...,0].real.
    assert np.allclose(fd.X[0, :, 0], b.positions_z.astype(np.float32))
    assert np.allclose(
        fd.X[:, :, 1], b.H_tensor[:, :, 0, 0].real.astype(np.float32), atol=1e-4
    )


def test_to_feature_dataset_transforms_and_gs():
    """apply_transforms → decouple+FV; geosignal_families → +2 colunas por família."""
    gen = _numba_gen()
    b = gen.generate_batch(n_models=3, n_positions=10, n_layers=3, seed=2)
    fd = gen.to_feature_dataset(b, apply_transforms=True, geosignal_families=["UHR"])
    assert fd.metadata["decoupled"] is True
    assert fd.X.shape[-1] == 5 + 2  # 5 base + UHR(att, phase)
    assert fd.feature_names[-2:] == ["UHR_att", "UHR_phase"]


def test_to_feature_dataset_multiconfig_shape():
    """Multi-config → X (n_models, nTR*nAng*nf, n_pos, n_feat)."""
    gen = _numba_gen()
    b = gen.generate_batch(
        n_models=3,
        n_positions=8,
        n_layers=3,
        frequencies_hz=[2e4, 5e4],
        dip_degs=[0.0, 30.0],
        tr_spacings_m=[1.0],
        seed=2,
    )
    fd = gen.to_feature_dataset(b)
    assert fd.X.shape == (3, 1 * 2 * 2, 8, 5)  # 4 configs
    assert fd.metadata["n_config"] == 4
