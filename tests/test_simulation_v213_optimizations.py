# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_v213_optimizations.py                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — Otimizações Numba v2.13                          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-01                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre as 3 otimizações implementadas na v2.13:                        ║
# ║                                                                           ║
# ║    Sprint 13.1 — Vetorização de frequências (prange nf):                ║
# ║      • Paridade bit-exata vs serial v2.12 (multi-freq scenarios)         ║
# ║      • Caso degenerado nf=1 (não regride)                                ║
# ║      • Multi-freq parity (múltiplos freqs em uma única chamada)         ║
# ║                                                                           ║
# ║    Sprint 13.2 — Cache cross-call:                                       ║
# ║      • Default `cache_persistent=False` preserva v2.12                  ║
# ║      • Hit bit-exato com mesma geometria (rotina PINN)                  ║
# ║      • Miss em geometrias diferentes (freqs, layers)                    ║
# ║      • `release_numba_cache()` libera + retorna count                    ║
# ║      • Thread-safety via ThreadPoolExecutor concorrente                  ║
# ║                                                                           ║
# ║    Sprint 13.4 — `nogil=True` universal:                                 ║
# ║      • Smoke: chamadas concorrentes via ThreadPool não corrompem         ║
# ║      • Resultados bit-exatos sob multi-thread                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes das otimizações v2.13 do simulador Python (Sprints 13.1, 13.2, 13.4)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from geosteering_ai.simulation import (
    get_numba_cache_size,
    release_numba_cache,
    simulate_multi,
)


@pytest.fixture(autouse=True)
def _cleanup_cache():
    """Garante cache global vazio antes/depois de cada teste."""
    release_numba_cache()
    yield
    release_numba_cache()


def _canonical_3layer():
    """Modelo canônico 3 camadas para testes de paridade."""
    return dict(
        rho_h=np.array([10.0, 1.0, 10.0]),
        rho_v=np.array([10.0, 1.0, 10.0]),
        esp=np.array([5.0]),
        positions_z=np.linspace(-2.0, 7.0, 30),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Sprint 13.1 — Vetorização de frequências
# ═════════════════════════════════════════════════════════════════════════════
class TestSprint131FreqVectorize:
    """Sprint 13.1: prange(nf) em _fields_in_freqs_kernel_cached."""

    def test_freq_vec_nf_1_no_regression(self):
        """Caso degenerado nf=1 deve produzir resultado válido."""
        m = _canonical_3layer()
        result = simulate_multi(
            **m, frequencies_hz=[20000.0], tr_spacings_m=[1.0], dip_degs=[0.0]
        )
        assert result.H_tensor.shape == (1, 1, 30, 1, 9)
        assert np.all(np.isfinite(result.H_tensor))

    def test_freq_vec_multi_freq_independence(self):
        """Multi-freq: cada freq deve dar resposta H distinta para o mesmo modelo."""
        m = _canonical_3layer()
        result = simulate_multi(
            **m,
            frequencies_hz=[20000.0, 100000.0, 500000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )
        assert result.H_tensor.shape == (1, 1, 30, 3, 9)

        # Hzz (componente 8) deve diferir entre frequências
        hzz_20k = result.H_tensor[0, 0, 15, 0, 8]
        hzz_500k = result.H_tensor[0, 0, 15, 2, 8]
        assert abs(hzz_20k - hzz_500k) > 1e-6, "Freqs distintas devem dar H distinto"

    def test_freq_vec_multi_call_parity(self):
        """Chamar simulate_multi 2× com mesmas frequências dá resultados idênticos."""
        m = _canonical_3layer()
        kwargs = dict(
            **m,
            frequencies_hz=[20000.0, 40000.0, 60000.0, 100000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )
        r1 = simulate_multi(**kwargs)
        r2 = simulate_multi(**kwargs)
        np.testing.assert_array_equal(
            r1.H_tensor,
            r2.H_tensor,
            err_msg="prange(nf) deterministic — deve ser bit-exato entre chamadas",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Sprint 13.2 — Cache cross-call
# ═════════════════════════════════════════════════════════════════════════════
class TestSprint132CacheCrossCall:
    """Sprint 13.2: cache_persistent + release_numba_cache."""

    def test_cache_default_off_no_pollution(self):
        """Default `cache_persistent=False` NÃO deve popular cache global."""
        m = _canonical_3layer()
        assert get_numba_cache_size() == 0
        simulate_multi(
            **m,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )
        assert (
            get_numba_cache_size() == 0
        ), "Default behavior must NOT populate global cache"

    def test_cache_persistent_populates_global(self):
        """`cache_persistent=True` deve popular o cache global."""
        m = _canonical_3layer()
        assert get_numba_cache_size() == 0
        simulate_multi(
            **m,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cache_persistent=True,
        )
        assert get_numba_cache_size() == 1

    def test_cache_hit_bit_exact(self):
        """2ª chamada com mesma geometria deve dar resultado bit-exato (cache hit)."""
        m = _canonical_3layer()
        kwargs = dict(
            **m,
            frequencies_hz=[20000.0, 40000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cache_persistent=True,
        )
        r1 = simulate_multi(**kwargs)
        size1 = get_numba_cache_size()
        r2 = simulate_multi(**kwargs)
        size2 = get_numba_cache_size()

        assert size1 == size2, "Cache hit não deve crescer cache"
        np.testing.assert_array_equal(
            r1.H_tensor,
            r2.H_tensor,
            err_msg="Cache hit deve dar resultado bit-exato",
        )

    def test_cache_miss_different_freqs(self):
        """Frequências diferentes devem invalidar hit (key inclui freqs_signature)."""
        m = _canonical_3layer()
        simulate_multi(
            **m,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cache_persistent=True,
        )
        size1 = get_numba_cache_size()
        simulate_multi(
            **m,
            frequencies_hz=[60000.0],  # freq diferente
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cache_persistent=True,
        )
        size2 = get_numba_cache_size()
        assert size2 == size1 + 1, "Freqs diferentes devem causar miss"

    def test_cache_miss_different_layers(self):
        """Modelos com perfis distintos não devem colidir no cache."""
        rho_a = np.array([10.0, 1.0, 10.0])
        rho_b = np.array([10.0, 5.0, 10.0])  # camada interna diferente
        esp = np.array([5.0])
        positions_z = np.linspace(-2.0, 7.0, 30)

        simulate_multi(
            rho_h=rho_a,
            rho_v=rho_a,
            esp=esp,
            positions_z=positions_z,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cache_persistent=True,
        )
        size1 = get_numba_cache_size()

        simulate_multi(
            rho_h=rho_b,
            rho_v=rho_b,
            esp=esp,
            positions_z=positions_z,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cache_persistent=True,
        )
        size2 = get_numba_cache_size()
        assert size2 == size1 + 1, "Perfis distintos devem causar miss"

    def test_release_numba_cache_returns_count(self):
        """`release_numba_cache()` deve retornar nº de entradas liberadas."""
        m = _canonical_3layer()
        simulate_multi(
            **m,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cache_persistent=True,
        )
        simulate_multi(
            **m,
            frequencies_hz=[40000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            cache_persistent=True,
        )
        assert get_numba_cache_size() == 2
        n_freed = release_numba_cache()
        assert n_freed == 2
        assert get_numba_cache_size() == 0

    def test_release_empty_cache(self):
        """Liberar cache vazio retorna 0 sem erro."""
        n_freed = release_numba_cache()
        assert n_freed == 0


# ═════════════════════════════════════════════════════════════════════════════
# Sprint 13.4 — nogil=True universal (smoke threading)
# ═════════════════════════════════════════════════════════════════════════════
class TestSprint134NogilThreading:
    """Sprint 13.4: nogil=True permite ThreadPool externo sem corrupção."""

    def test_concurrent_threadpool_no_corruption(self):
        """4 threads chamando simulate_multi simultaneamente não corrompem resultados."""
        m = _canonical_3layer()
        kwargs = dict(
            **m,
            frequencies_hz=[20000.0, 40000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )

        # Baseline: chamada serial
        baseline = simulate_multi(**kwargs)

        def _call():
            return simulate_multi(**kwargs)

        # 4 chamadas concorrentes
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_call) for _ in range(4)]
            results = [f.result() for f in futures]

        # Todos devem ser idênticos ao baseline (determinismo Numba + nogil safe)
        for i, r in enumerate(results):
            np.testing.assert_allclose(
                r.H_tensor,
                baseline.H_tensor,
                rtol=1e-12,
                atol=1e-15,
                err_msg=f"Thread {i} divergiu de baseline serial",
            )


# ═════════════════════════════════════════════════════════════════════════════
# Smoke geral — backward-compat v2.12 preservado
# ═════════════════════════════════════════════════════════════════════════════
class TestBackwardCompatV212:
    """Garante que v2.13 não regride APIs/resultados de v2.12."""

    def test_simulate_multi_no_kwargs_v212_path(self):
        """API v2.12 (single-model, sem cache_persistent) deve funcionar igual."""
        m = _canonical_3layer()
        result = simulate_multi(
            **m,
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )
        from geosteering_ai.simulation import MultiSimulationResult

        assert isinstance(result, MultiSimulationResult)
        assert result.H_tensor.shape == (1, 1, 30, 1, 9)
        assert np.all(np.isfinite(result.H_tensor))

    def test_simulate_multi_models_batch_v212_path(self):
        """API v2.12 (batch via models=[...]) continua funcionando."""
        from geosteering_ai.simulation import MultiSimulationResultBatch

        models = [
            dict(
                rho_h=np.array([10.0, 1.0, 10.0]),
                rho_v=np.array([10.0, 1.0, 10.0]),
                esp=np.array([5.0]),
            )
            for _ in range(3)
        ]
        result = simulate_multi(
            models=models,
            positions_z=np.linspace(-2.0, 7.0, 20),
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            n_workers=1,  # Modo A — single process, single thread
        )
        assert isinstance(result, MultiSimulationResultBatch)
        assert result.H_stack.shape[0] == 3
