# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_sprint13.py                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — JIT cache observability v2.15                    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-01                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre Sprint 14.3 (v2.15) — observability do triple-cache JIT JAX:    ║
# ║                                                                           ║
# ║    • get_jit_cache_info() retorna 0 após clear_jit_cache + clear_unified ║
# ║    • Após simulate_multi_jax: bucketed_size > 0 ou unified_size > 0      ║
# ║    • strategy="unified" popula APENAS _UNIFIED_JIT_CACHE                ║
# ║    • estimated_vram_mb > 0 quando há entradas; 0 quando vazio           ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • forward_pure.py:_BUCKET_JIT_CACHE, _UNIFIED_JIT_CACHE,              ║
# ║      _UNIFIED_CHUNKED_JIT_CACHE                                          ║
# ║    • Sprints 10-12 (v1.5.0/v1.6.0): triple-cache estruturado            ║
# ║    • Sprint 14.3 (v2.15): observability via get_jit_cache_info()        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do JIT cache observability v2.15 (Sprint 14.3)."""
from __future__ import annotations

import numpy as np
import pytest

# Skip global se JAX não disponível (CI sem dependência opcional)
jax = pytest.importorskip("jax")
from geosteering_ai.simulation._jax.forward_pure import (  # noqa: E402
    clear_jit_cache,
    clear_unified_jit_cache,
    get_jit_cache_info,
)


def _canonical_3layer():
    """Modelo canônico 3 camadas para os testes."""
    return dict(
        rho_h=np.array([10.0, 1.0, 10.0]),
        rho_v=np.array([10.0, 1.0, 10.0]),
        esp=np.array([5.0]),
        positions_z=np.linspace(-2.0, 7.0, 30),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Sprint 14.3 — JIT cache observability
# ═════════════════════════════════════════════════════════════════════════════
class TestSprint143JITCacheInfo:
    """Sprint 14.3: ``get_jit_cache_info()`` reporta 3 caches + VRAM."""

    def test_jit_cache_info_empty(self):
        """Após clear_jit_cache + clear_unified: total_xla_programs == 0."""
        clear_jit_cache()
        clear_unified_jit_cache()
        info = get_jit_cache_info()

        # Validação dos 3 caches zerados
        assert info["bucketed_size"] == 0
        assert info["unified_size"] == 0
        assert info["chunked_size"] == 0
        assert info["total_xla_programs"] == 0

        # VRAM estimate igual a 0 quando todos os caches estão vazios
        assert info["estimated_vram_mb"] == 0.0

        # Strategy distribution corresponde aos sizes
        dist = info["strategy_distribution"]
        assert dist == {"bucketed": 0, "unified": 0, "chunked": 0}

        # Backward-compat (chave herdada de v1.5.0)
        assert info["n_entries"] == 0
        assert info["keys"] == []
        assert info["maxsize"] >= 1  # default 64

    def test_jit_cache_info_after_simulate_unified(self):
        """Após simulate_multi_jax(strategy='unified'): unified_size>=1."""
        from geosteering_ai.simulation import SimulationConfig
        from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax

        clear_jit_cache()
        clear_unified_jit_cache()

        m = _canonical_3layer()
        cfg = SimulationConfig(
            frequency_hz=20000.0,
            tr_spacing_m=1.0,
            backend="jax",
            jax_strategy="unified",
        )
        _ = simulate_multi_jax(
            cfg=cfg,
            rho_h=m["rho_h"],
            rho_v=m["rho_v"],
            esp=m["esp"],
            positions_z=m["positions_z"],
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )

        info = get_jit_cache_info()
        # Estratégia unified deve ter populado APENAS _UNIFIED_JIT_CACHE
        assert info["unified_size"] >= 1, (
            f"Strategy 'unified' deveria popular _UNIFIED_JIT_CACHE, "
            f"got unified_size={info['unified_size']}"
        )
        assert (
            info["bucketed_size"] == 0
        ), "Strategy 'unified' NÃO deve popular _BUCKET_JIT_CACHE"

    def test_jit_cache_info_vram_estimate(self):
        """VRAM estimate > 0 com cache populado, 0 sem."""
        clear_jit_cache()
        clear_unified_jit_cache()

        # Cache vazio → VRAM = 0
        info_empty = get_jit_cache_info()
        assert info_empty["estimated_vram_mb"] == 0.0

        # Popular cache via simulação simples
        from geosteering_ai.simulation import SimulationConfig
        from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax

        m = _canonical_3layer()
        cfg = SimulationConfig(
            frequency_hz=20000.0,
            tr_spacing_m=1.0,
            backend="jax",
            jax_strategy="unified",
        )
        _ = simulate_multi_jax(
            cfg=cfg,
            rho_h=m["rho_h"],
            rho_v=m["rho_v"],
            esp=m["esp"],
            positions_z=m["positions_z"],
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )

        info_populated = get_jit_cache_info()
        # VRAM heurística > 0 quando há ao menos 1 entrada
        assert info_populated["estimated_vram_mb"] > 0.0
        # Sanity check: heurística é razoavelmente conservadora
        # (3 × n × npt × 16 / 1024² ≈ ordem de KB-MB para casos típicos)
        assert (
            info_populated["estimated_vram_mb"] < 1000.0
        ), "Heurística VRAM excessiva — investigar"

    def test_jit_cache_info_idempotent_readonly(self):
        """get_jit_cache_info é read-only — chamar 3× não modifica state."""
        from geosteering_ai.simulation import SimulationConfig
        from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax

        clear_jit_cache()
        clear_unified_jit_cache()
        m = _canonical_3layer()
        cfg = SimulationConfig(
            frequency_hz=20000.0,
            tr_spacing_m=1.0,
            backend="jax",
            jax_strategy="unified",
        )
        _ = simulate_multi_jax(
            cfg=cfg,
            rho_h=m["rho_h"],
            rho_v=m["rho_v"],
            esp=m["esp"],
            positions_z=m["positions_z"],
            frequencies_hz=[20000.0],
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
        )

        info1 = get_jit_cache_info()
        info2 = get_jit_cache_info()
        info3 = get_jit_cache_info()

        # As 3 chamadas devem retornar exatamente o mesmo state
        assert info1["total_xla_programs"] == info2["total_xla_programs"]
        assert info2["total_xla_programs"] == info3["total_xla_programs"]
        assert info1["estimated_vram_mb"] == info2["estimated_vram_mb"]
        assert info2["unified_size"] == info3["unified_size"]
