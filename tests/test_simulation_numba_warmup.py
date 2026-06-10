"""Testes do warmup Numba-nativo do simulador CPU (v2.52).

Cobre `geosteering_ai/simulation/_numba/warmup.py`:

    - T1 test_warmup_compiles_parallel_kernels: aquece flat/cached/precompute
    - T2 test_zero_recompile_after_warmup:      real-sim pós-warmup → 0 novos compiles
    - T3 test_warmup_preserves_numerics:        cold == warm bit-exato (paridade)
    - T4 test_warmup_skips_without_numba:        HAS_NUMBA False → skip gracioso
    - T5 test_warmup_rejects_invalid_args:       validação eager

GAP FECHADO (v2.52): o warmup legado (callback JAX, 3-pos) NÃO aquece os kernels
`parallel=True`/prange que dominam produção. Este warmup os pré-compila via o
caminho REAL `simulate_multi(backend="numba")` — JAX-independente.

T1-T3 compilam Numba (lentos a frio) → marcados `slow`. Métrica de cobertura usa
os kernels Python-dispatcháveis (flat/cached/precompute); os dipolos são inlinados
njit-a-njit (signatures standalone = 0; código compila dentro de flat/cached).
"""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation._numba import HAS_NUMBA

pytestmark = pytest.mark.skipif(
    not HAS_NUMBA, reason="Numba não instalado — warmup Numba requer numba"
)

if HAS_NUMBA:
    from geosteering_ai.simulation._numba.kernel import precompute_common_arrays_cache
    from geosteering_ai.simulation._numba.warmup import warmup_numba_simulator
    from geosteering_ai.simulation.config import SimulationConfig
    from geosteering_ai.simulation.forward import (
        _simulate_combined_prange_flat,
        _simulate_positions_njit_cached,
    )
    from geosteering_ai.simulation.multi_forward import simulate_multi


_PARALLEL_KERNELS = (
    "_simulate_combined_prange_flat",
    "_simulate_positions_njit_cached",
    "precompute_common_arrays_cache",
)


def _signatures() -> dict:
    return {
        "_simulate_combined_prange_flat": len(
            _simulate_combined_prange_flat.signatures
        ),
        "_simulate_positions_njit_cached": len(
            _simulate_positions_njit_cached.signatures
        ),
        "precompute_common_arrays_cache": len(
            precompute_common_arrays_cache.signatures
        ),
    }


def _numba_cfg(n_positions: int) -> "SimulationConfig":
    return SimulationConfig(
        backend="numba",
        dtype="complex128",
        hankel_filter="werthmuller_201pt",
        n_positions=n_positions,
    )


def _matching_inputs(n_layers=3, n_positions=20):
    """Inputs reais que casam com o grid default do warmup (rho=10, esp=5)."""
    n_esp = max(n_layers - 2, 0)
    total = 5.0 * n_esp if n_esp > 0 else 5.0
    pos = np.linspace(-1.0, total + 1.0, n_positions)
    rho = np.full(n_layers, 10.0)
    esp = np.full(n_esp, 5.0)
    return rho, rho.copy(), esp, pos


# ════════════════════════════════════════════════════════════════════════
# T1-T3 — compilam Numba (slow)
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_warmup_compiles_parallel_kernels():
    """warmup aquece os kernels paralelos de produção (flat/cached/precompute)."""
    info = warmup_numba_simulator(n_layers=3, n_positions=20)
    assert info["skipped"] is False
    for kernel in _PARALLEL_KERNELS:
        assert info["n_signatures"][kernel] >= 1, f"{kernel} não foi aquecido"
        assert kernel in info["functions_warmed"]
    assert info["elapsed_s"] > 0.0
    assert info["threads"] >= 1


@pytest.mark.slow
def test_zero_recompile_after_warmup():
    """Real-sim pós-warmup (MESMA config) → 0 novos compiles (cache hit)."""
    warmup_numba_simulator(n_layers=3, n_positions=20)
    sig_before = _signatures()
    assert all(v >= 1 for v in sig_before.values())

    rho_h, rho_v, esp, pos = _matching_inputs(n_layers=3, n_positions=20)
    cfg = _numba_cfg(20)
    # Cobre ambos os branches (multi-combo + single-combo) como o warmup.
    simulate_multi(
        rho_h,
        rho_v,
        esp,
        pos,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0, 2.0],
        dip_degs=[0.0],
        cfg=cfg,
    )
    simulate_multi(
        rho_h,
        rho_v,
        esp,
        pos,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
        cfg=cfg,
    )
    sig_after = _signatures()
    assert sig_after == sig_before, (
        f"warmup ineficaz: novas assinaturas compiladas no real-sim "
        f"(before={sig_before} after={sig_after})"
    )


@pytest.mark.slow
def test_warmup_preserves_numerics_bit_exact():
    """Warmup NÃO altera numérica: simula == simula-pós-warmup bit-exato.

    Guarda a paridade Fortran <1e-6 na fronteira do warmup: como o warmup só
    PRÉ-COMPILA o mesmo kernel, o resultado de simulate_multi é IDÊNTICO.
    """
    rho_h, rho_v, esp, pos = _matching_inputs(n_layers=3, n_positions=20)
    cfg = _numba_cfg(20)
    kw = dict(
        frequencies_hz=[20000.0], tr_spacings_m=[1.0, 2.0], dip_degs=[0.0], cfg=cfg
    )

    h_ref = np.asarray(simulate_multi(rho_h, rho_v, esp, pos, **kw).H_tensor)
    warmup_numba_simulator(n_layers=3, n_positions=20)
    h_warm = np.asarray(simulate_multi(rho_h, rho_v, esp, pos, **kw).H_tensor)

    np.testing.assert_array_equal(h_ref, h_warm)


# ════════════════════════════════════════════════════════════════════════
# T4-T5 — instantâneos
# ════════════════════════════════════════════════════════════════════════


def test_warmup_skips_without_numba(monkeypatch):
    """HAS_NUMBA False → warmup retorna skip gracioso (não levanta)."""
    import geosteering_ai.simulation._numba as nmod

    monkeypatch.setattr(nmod, "HAS_NUMBA", False)
    out = warmup_numba_simulator(n_layers=3, n_positions=20)
    assert out["skipped"] is True
    assert out["reason"] == "numba_absent"
    assert out["functions_warmed"] == []


def test_warmup_rejects_invalid_args():
    """Validação eager de n_layers/n_positions/n_models."""
    with pytest.raises(ValueError, match="n_layers"):
        warmup_numba_simulator(n_layers=1)
    with pytest.raises(ValueError, match="n_positions"):
        warmup_numba_simulator(n_positions=1)
    with pytest.raises(ValueError, match="n_models"):
        warmup_numba_simulator(n_models=0)


@pytest.mark.slow
def test_warmup_from_config_smoke():
    """warmup_numba_simulator_from_config deriva kwargs e roda (smoke, slow)."""
    from geosteering_ai.simulation._numba.warmup import (
        warmup_numba_simulator_from_config,
    )

    cfg = SimulationConfig(backend="numba", n_positions=12)
    info = warmup_numba_simulator_from_config(cfg, n_layers=3)
    assert info["skipped"] is False
    assert info["n_positions"] == 12
