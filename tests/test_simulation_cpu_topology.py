# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_cpu_topology.py                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager v2.17 (Sprint 17.1)                    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-02                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest                                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida `detect_cpu_topology()` e `recommend_default_parallelism()`,   ║
# ║    funções introduzidas na v2.17 para corrigir a regressão persistente   ║
# ║    de 3× em produção GUI causada por oversubscrição em CPUs com          ║
# ║    Hyperthreading (HT) ou SMT.                                            ║
# ║                                                                           ║
# ║  COBERTURA DOS TESTES                                                     ║
# ║    1. detect_cpu_topology retorna valores razoáveis no hardware atual    ║
# ║    2. cache funciona (segunda chamada não invoca subprocess)             ║
# ║    3. recommend_default_parallelism para batch grande respeita phys     ║
# ║    4. recommend_default_parallelism para single-model usa Modo B        ║
# ║    5. recomendação NUNCA causa oversubscrição                           ║
# ║    6. fallback heurístico em caso de erro                                 ║
# ║                                                                           ║
# ║  REGRESSÃO ALVO                                                           ║
# ║    GUI v2.16 produzia 38k mod/h em 8C/16T HT (vs 123k esperado).        ║
# ║    Causa: defaults (4w × 4t = 16) excediam cores físicos (8) →           ║
# ║    oversubscrição 2× em HT. Fix v2.17: defaults (4w × 2t = 8) =        ║
# ║    cores físicos exatos.                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de detecção de topologia CPU e recomendação de paralelismo (v2.17)."""
from __future__ import annotations

import os
from unittest import mock

import pytest

from geosteering_ai.simulation import _workers as _workers_mod
from geosteering_ai.simulation import (
    detect_cpu_topology,
    recommend_default_parallelism,
)


@pytest.fixture(autouse=True)
def _reset_topology_cache():
    """Limpa o cache global de topologia entre testes para evitar contaminação."""
    _workers_mod._CPU_TOPOLOGY_CACHE = None
    yield
    _workers_mod._CPU_TOPOLOGY_CACHE = None


class TestDetectCpuTopology:
    """Validação da função `detect_cpu_topology()`."""

    def test_returns_tuple_of_three_ints_and_bool(self) -> None:
        """detect_cpu_topology retorna (logical, physical, has_ht) bem-formado."""
        result = detect_cpu_topology()
        assert isinstance(result, tuple)
        assert len(result) == 3
        logical, physical, has_ht = result
        assert isinstance(logical, int)
        assert isinstance(physical, int)
        assert isinstance(has_ht, bool)

    def test_values_are_sane(self) -> None:
        """Valores detectados são consistentes com o hardware real."""
        logical, physical, has_ht = detect_cpu_topology()
        # Limites de sanidade — qualquer máquina razoável satisfaz:
        assert logical >= 1
        assert physical >= 1
        assert physical <= logical  # físico nunca > lógico
        # Coerência has_ht
        assert has_ht == (logical > physical)

    def test_matches_os_cpu_count(self) -> None:
        """logical_cores corresponde a os.cpu_count() (truth source)."""
        logical, _, _ = detect_cpu_topology()
        expected_logical = os.cpu_count() or 1
        assert logical == expected_logical

    def test_cache_returns_same_instance(self) -> None:
        """Cache funciona — segunda chamada retorna mesmo valor sem re-detectar."""
        first = detect_cpu_topology()
        second = detect_cpu_topology()
        assert first == second
        # Verificar que o cache foi populado (não None)
        assert _workers_mod._CPU_TOPOLOGY_CACHE is not None

    def test_fallback_heuristic_when_psutil_and_subprocess_fail(self) -> None:
        """Fallback heurístico ativa quando todas as estratégias falham."""
        with mock.patch.dict("sys.modules", {"psutil": None}):
            with mock.patch(
                "subprocess.check_output",
                side_effect=FileNotFoundError("não disponível"),
            ):
                with mock.patch("builtins.open", side_effect=OSError("denied")):
                    # Reset cache para forçar nova detecção
                    _workers_mod._CPU_TOPOLOGY_CACHE = None
                    logical, physical, has_ht = detect_cpu_topology()
                    # Heurística: se logical >= 4, physical = logical // 2
                    if logical >= 4:
                        assert physical == logical // 2
                        assert has_ht is True
                    else:
                        assert physical == logical
                        assert has_ht is False


class TestRecommendDefaultParallelism:
    """Validação da função `recommend_default_parallelism()`."""

    def test_returns_tuple_of_two_positive_ints(self) -> None:
        """Retorna (n_workers, threads_per_worker) com valores >= 1."""
        n_workers, threads = recommend_default_parallelism()
        assert isinstance(n_workers, int)
        assert isinstance(threads, int)
        assert n_workers >= 1
        assert threads >= 1

    def test_does_not_oversubscribe_physical_cores_for_batch(self) -> None:
        """Invariante v2.17 (confirmado por v2.20): workers × threads ≤ cores físicos.

        Confirmação empírica v2.20: medição com 5 runs consecutivos em
        Mac 8C/16T HT mostrou que oversubscription com HT degrada
        Cenário E em 20-25% (4w × 4t = 38k mediana vs 4w × 2t = 46k
        mediana). HT NÃO ajuda neste kernel — context switch entre
        hyperthreads + cache thrashing dominam. Esta invariante DEVE ser
        mantida.
        """
        _, phys, _ = detect_cpu_topology()
        n_workers, threads = recommend_default_parallelism()
        total_threads = n_workers * threads
        assert total_threads <= phys, (
            f"Oversubscrição: {n_workers} × {threads} = {total_threads} threads "
            f"> {phys} cores físicos. Análise empírica v2.20 confirmou que HT "
            f"degrada throughput em 20-25% no kernel hmd_tiv."
        )

    def test_single_model_hint_returns_mode_b(self) -> None:
        """Para single-model (n_models < 10), retorna Modo B (1 worker, N threads)."""
        n_workers, threads = recommend_default_parallelism(n_models_hint=1)
        _, phys, _ = detect_cpu_topology()
        assert n_workers == 1
        assert threads == phys  # 1 worker × phys threads = phys cores físicos

    def test_small_batch_returns_mode_b(self) -> None:
        """Batch pequeno (1-9 modelos) também usa Modo B (spawn overhead)."""
        for n in [1, 2, 5, 9]:
            n_workers, threads = recommend_default_parallelism(n_models_hint=n)
            assert n_workers == 1
            _, phys, _ = detect_cpu_topology()
            assert threads == phys

    def test_large_batch_returns_mode_d_or_b(self) -> None:
        """Batch grande (>= 10) retorna Modo D (M workers × K threads)."""
        n_workers, threads = recommend_default_parallelism(n_models_hint=100)
        _, phys, _ = detect_cpu_topology()
        if phys >= 4:
            # Modo D: workers >= 2
            assert n_workers >= 2
            assert n_workers * threads <= phys
        else:
            # phys == 2: cai em Modo B (1 worker × 2 threads)
            assert n_workers * threads <= phys

    def test_default_no_hint_assumes_batch(self) -> None:
        """Sem hint, assume batch grande (default da GUI)."""
        n_workers, threads = recommend_default_parallelism()
        n_workers_with_hint, threads_with_hint = recommend_default_parallelism(
            n_models_hint=100
        )
        assert (n_workers, threads) == (n_workers_with_hint, threads_with_hint)


class TestSimulatedHardware:
    """Validação com topologias mockadas — cobre cenários de produção."""

    @pytest.fixture
    def mock_topology(self):
        """Helper: mock _CPU_TOPOLOGY_CACHE com valores específicos."""

        def _set(logical: int, physical: int) -> None:
            _workers_mod._CPU_TOPOLOGY_CACHE = (logical, physical, logical > physical)

        return _set

    def test_mac_intel_8c_16t_ht(self, mock_topology) -> None:
        """Mac Intel 8C/16T HT — confirmado empiricamente em v2.20.

        Medição 5× consecutiva em Mac 8C/16T HT (Cenário E 600 pts):
        - 4w × 2t (8 threads = phys): mediana 46k mod/h
        - 4w × 4t (16 threads = HT):  mediana 38k mod/h (-25%)

        HT degrada por context switch entre hyperthreads + cache trashing.
        """
        mock_topology(logical=16, physical=8)
        n_workers, threads = recommend_default_parallelism()
        assert n_workers == 4
        assert threads == 2
        assert n_workers * threads == 8  # = phys, sem oversubscrição ✓

    def test_apple_silicon_m1_8core_no_ht(self, mock_topology) -> None:
        """Apple Silicon M1 8-core (sem HT)."""
        mock_topology(logical=8, physical=8)
        n_workers, threads = recommend_default_parallelism()
        assert n_workers == 4
        assert threads == 2
        assert n_workers * threads == 8

    def test_linux_xeon_32c_64t_ht(self, mock_topology) -> None:
        """Linux Xeon 32C/64T HT (servidor produção)."""
        mock_topology(logical=64, physical=32)
        n_workers, threads = recommend_default_parallelism()
        assert n_workers == 16
        assert threads == 2
        assert n_workers * threads == 32  # = phys ✓

    def test_low_end_4c_8t_ht(self, mock_topology) -> None:
        """CPU low-end 4C/8T HT (laptop antigo)."""
        mock_topology(logical=8, physical=4)
        n_workers, threads = recommend_default_parallelism()
        assert n_workers == 2
        assert threads == 2
        assert n_workers * threads == 4

    def test_dual_core_no_ht(self, mock_topology) -> None:
        """CPU dual-core sem HT (caso degenerado)."""
        mock_topology(logical=2, physical=2)
        n_workers, threads = recommend_default_parallelism()
        # Para phys=2: workers=1 (phys // 2 = 1), threads = 2
        assert n_workers == 1
        assert threads == 2
        assert n_workers * threads == 2  # = phys ✓

    def test_single_core(self, mock_topology) -> None:
        """CPU single-core (caso extremo)."""
        mock_topology(logical=1, physical=1)
        n_workers, threads = recommend_default_parallelism()
        assert n_workers == 1
        assert threads == 1

    def test_apple_m_pro_10c_10t_no_ht(self, mock_topology) -> None:
        """Apple Silicon M2/M3 Pro 10-core (sem HT, P+E unificados)."""
        mock_topology(logical=10, physical=10)
        n_workers, threads = recommend_default_parallelism()
        # Sem HT: target = phys = 10; workers = 5, threads = 2
        assert n_workers == 5
        assert threads == 2
        assert n_workers * threads == 10  # = phys ✓


class TestNoRegression:
    """Garantias de não-regressão para uso em produção."""

    def test_recommendation_is_deterministic(self) -> None:
        """Múltiplas chamadas retornam o mesmo valor (sem aleatoriedade)."""
        first = recommend_default_parallelism()
        for _ in range(5):
            assert recommend_default_parallelism() == first

    def test_topology_cache_persists_across_calls(self) -> None:
        """Cache evita re-detecção custosa (subprocess) em chamadas repetidas."""
        # Primeira chamada popula o cache
        first = detect_cpu_topology()
        cached = _workers_mod._CPU_TOPOLOGY_CACHE
        assert cached == first
        # Mock sysctl/cpuinfo: se cache funciona, não chamamos subprocess
        with mock.patch(
            "subprocess.check_output", side_effect=AssertionError("não deve ser chamado")
        ):
            second = detect_cpu_topology()
            assert second == first  # vem do cache, não da chamada
