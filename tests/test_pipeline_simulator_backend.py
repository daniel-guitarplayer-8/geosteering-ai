# -*- coding: utf-8 -*-
"""Testes Sprint 6.1 + 6.2 — PipelineConfig.simulator_backend + SyntheticDataGenerator."""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.synthetic_generator import (
    GeneratedBatch,
    SyntheticDataGenerator,
)

# ═════════════════════════════════════════════════════════════════════════
# Sprint 6.1 — PipelineConfig simulator_* fields
# ═════════════════════════════════════════════════════════════════════════


def test_pipeline_config_simulator_defaults() -> None:
    """Defaults conservadores: fortran_f2py + complex128 + cpu + native."""
    cfg = PipelineConfig()
    assert cfg.simulator_backend == "fortran_f2py"
    assert cfg.simulator_precision == "complex128"
    assert cfg.simulator_device == "cpu"
    assert cfg.simulator_jax_mode == "native"
    assert cfg.simulator_cache_common_arrays is True


def test_pipeline_config_simulator_numba_valid() -> None:
    """Numba + CPU + complex128 é válido."""
    cfg = PipelineConfig(
        simulator_backend="numba",
        simulator_device="cpu",
    )
    assert cfg.simulator_backend == "numba"


def test_pipeline_config_simulator_jax_gpu_valid() -> None:
    """JAX + GPU + complex64 é válido."""
    cfg = PipelineConfig(
        simulator_backend="jax",
        simulator_device="gpu",
        simulator_precision="complex64",
    )
    assert cfg.simulator_device == "gpu"


def test_pipeline_config_simulator_fortran_gpu_raises() -> None:
    """fortran_f2py + gpu levanta AssertionError."""
    with pytest.raises(AssertionError, match="simulator_device='gpu' requer"):
        PipelineConfig(
            simulator_backend="fortran_f2py",
            simulator_device="gpu",
        )


def test_pipeline_config_simulator_numba_gpu_raises() -> None:
    """numba + gpu levanta AssertionError."""
    with pytest.raises(AssertionError, match="simulator_device='gpu' requer"):
        PipelineConfig(
            simulator_backend="numba",
            simulator_device="gpu",
        )


def test_pipeline_config_simulator_invalid_backend_raises() -> None:
    """Backend fora da lista válida levanta AssertionError."""
    with pytest.raises(AssertionError, match="simulator_backend"):
        PipelineConfig(simulator_backend="invalid")


# ═════════════════════════════════════════════════════════════════════════
# Sprint 6.2 — SyntheticDataGenerator
# ═════════════════════════════════════════════════════════════════════════


def test_synthetic_generator_numba_shape() -> None:
    """GeneratedBatch tem shapes corretos para Numba."""
    cfg = PipelineConfig(simulator_backend="numba")
    gen = SyntheticDataGenerator(cfg)
    batch = gen.generate_batch(n_models=3, n_positions=20, n_layers=3, seed=42)
    assert batch.H_tensor.shape == (3, 20, 1, 9)
    assert batch.rho_h.shape == (3, 3)
    assert batch.rho_v.shape == (3, 3)
    assert batch.esp.shape == (3, 1)
    assert batch.positions_z.shape == (20,)
    assert batch.dat_22col.shape == (3 * 20,)
    assert batch.metadata["backend"] == "numba"
    assert batch.n_models == 3
    assert batch.n_layers == 3


def test_synthetic_generator_reproducible_seed() -> None:
    """Mesmo seed → mesmo batch."""
    cfg = PipelineConfig(simulator_backend="numba")
    gen1 = SyntheticDataGenerator(cfg)
    gen2 = SyntheticDataGenerator(cfg)
    b1 = gen1.generate_batch(n_models=2, n_positions=10, seed=123)
    b2 = gen2.generate_batch(n_models=2, n_positions=10, seed=123)
    np.testing.assert_array_equal(b1.rho_h, b2.rho_h)
    np.testing.assert_allclose(b1.H_tensor, b2.H_tensor, atol=1e-14)


def test_synthetic_generator_dat_22col_format() -> None:
    """dat_22col é compatível com DTYPE_22COL e escrevível em .dat."""
    import os
    import tempfile

    from geosteering_ai.simulation.io.binary_dat import DTYPE_22COL

    cfg = PipelineConfig(simulator_backend="numba")
    gen = SyntheticDataGenerator(cfg)
    batch = gen.generate_batch(n_models=2, n_positions=10, seed=42)
    assert batch.dat_22col.dtype == DTYPE_22COL

    # Teste round-trip: escrever e reler
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        path = f.name
    try:
        batch.dat_22col.tofile(path)
        reread = np.fromfile(path, dtype=DTYPE_22COL)
        assert len(reread) == len(batch.dat_22col)
        np.testing.assert_array_equal(reread["Re_Hxx"], batch.dat_22col["Re_Hxx"])
    finally:
        os.unlink(path)


def test_synthetic_generator_fortran_backend_raises() -> None:
    """simulator_backend='fortran_f2py' levanta ValueError."""
    cfg = PipelineConfig(simulator_backend="fortran_f2py")
    with pytest.raises(ValueError, match="não suporta simulator_backend='fortran_f2py'"):
        SyntheticDataGenerator(cfg)


def test_synthetic_generator_high_rho_stability() -> None:
    """Modelos com ρ > 1000 permanecem estáveis (finite)."""
    cfg = PipelineConfig(simulator_backend="numba")
    gen = SyntheticDataGenerator(cfg)
    batch = gen.generate_batch(
        n_models=2,
        n_positions=10,
        n_layers=3,
        rho_h_range=(10.0, 5000.0),
        rho_v_range=(10.0, 10000.0),
        seed=7,
    )
    assert np.all(np.isfinite(batch.H_tensor.real))
    assert np.all(np.isfinite(batch.H_tensor.imag))


def test_synthetic_generator_throughput_reported() -> None:
    """Metadata inclui throughput em mod/h."""
    cfg = PipelineConfig(simulator_backend="numba")
    gen = SyntheticDataGenerator(cfg)
    batch = gen.generate_batch(n_models=3, n_positions=20, seed=42)
    assert "throughput_mod_h" in batch.metadata
    assert batch.metadata["throughput_mod_h"] > 0
    assert batch.metadata["elapsed_s"] > 0
