# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Arquivo  : tests/test_simulation_jax_ctx_cache.py                          ║
# ║  Sprint   : O3p1 (v2.43)                                                    ║
# ║  Data     : 2026-05-27                                                       ║
# ║  Autor    : Daniel Leal                                                      ║
# ║  Camada   : 8 — tests (DataPipeline)                                         ║
# ║  Padrão   : D1, D5, D9 (Google docstrings, pytest, no print)                ║
# ║  Encoding : UTF-8                                                            ║
# ║  Status   : ACTIVE                                                           ║
# ║                                                                              ║
# ║  Sumário                                                                     ║
# ║    Valida `_CTX_CACHE` (LRU) de `ForwardPureContext` introduzido em         ║
# ║    Sprint O3p1: hit/miss básico, miss por geometria diferente, miss por     ║
# ║    `strategy`/`complex_dtype` diferente, eviction LRU, paridade vs build    ║
# ║    direto, e API pública (clear/info/maxsize).                              ║
# ║                                                                              ║
# ║  Dependências                                                                ║
# ║    pytest 8.x, JAX 0.7.x, numpy 2.x.                                         ║
# ║                                                                              ║
# ║  Físico                                                                      ║
# ║    Cache LRU é puramente algorítmico; testes usam geometrias arbitrárias    ║
# ║    sem necessidade de paridade Fortran. T6/T8 garantem forward_pure_jax     ║
# ║    produz mesmo H usando ctx cached vs direto.                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes para ``_CTX_CACHE`` — Sprint O3p1 (v2.43).

Cobre:
  • T1: hit/miss básico (2 builds idênticos → 1 build real, 1 hit)
  • T2: miss por esp diferente
  • T3: miss por strategy diferente
  • T4: miss por complex_dtype diferente
  • T5: LRU eviction (maxsize=2, 3 geometrias distintas)
  • T6: paridade ctx_cached == ctx direto (mesmo H_out)
  • T7: clear_ctx_cache + get_ctx_cache_info funcionais
  • T8: forward_pure_jax produz mesmo H usando ctx_cached vs ctx direto
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# JAX_ENABLE_X64 deve ser setado ANTES de importar JAX
os.environ.setdefault("JAX_ENABLE_X64", "True")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — geometria de teste padrão (5 camadas)
# ──────────────────────────────────────────────────────────────────────────────


def _geom_default() -> dict:
    """Geometria de referência (5 camadas, 10 posições, 1 freq)."""
    return dict(
        rho_h=np.array([1.0, 100.0, 50.0, 200.0, 5.0]),
        rho_v=np.array([1.5, 150.0, 75.0, 300.0, 7.5]),
        esp=np.array([5.0, 3.0, 4.0]),
        positions_z=np.linspace(-5.0, 5.0, 10),
        freqs_hz=np.array([20000.0]),
        tr_spacing_m=1.0,
        dip_deg=0.0,
        hankel_filter="werthmuller_201pt",
        strategy="bucketed",
        chunk_size=None,
        complex_dtype="complex128",
    )


def _geom_diff_esp() -> dict:
    """Mesma geometria mas com `esp` diferente (mudança de espessura)."""
    geom = _geom_default()
    geom["esp"] = np.array([6.0, 2.5, 4.0])  # primeira espessura diferente
    return geom


@pytest.fixture(autouse=True)
def _reset_ctx_cache():
    """Reseta `_CTX_CACHE` antes de cada teste (isolamento)."""
    from geosteering_ai.simulation._jax.forward_pure import (
        clear_ctx_cache,
        set_ctx_cache_maxsize,
    )

    clear_ctx_cache()
    set_ctx_cache_maxsize(32)  # default
    yield
    clear_ctx_cache()
    set_ctx_cache_maxsize(32)


# ──────────────────────────────────────────────────────────────────────────────
# T1: hit/miss básico
# ──────────────────────────────────────────────────────────────────────────────


def test_t1_hit_miss_basico():
    """Dois builds idênticos → 1 entry no cache, segundo é hit."""
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context_cached,
        get_ctx_cache_info,
    )

    geom = _geom_default()
    assert get_ctx_cache_info()["n_entries"] == 0

    # Build 1 — miss
    ctx1 = build_static_context_cached(**geom)
    assert get_ctx_cache_info()["n_entries"] == 1

    # Build 2 — hit (mesma geom)
    ctx2 = build_static_context_cached(**geom)
    assert get_ctx_cache_info()["n_entries"] == 1
    # Hit retorna a MESMA instância
    assert ctx1 is ctx2


# ──────────────────────────────────────────────────────────────────────────────
# T2: miss por esp diferente
# ──────────────────────────────────────────────────────────────────────────────


def test_t2_miss_por_esp_diferente():
    """Build com `esp` diferente cria nova entry."""
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context_cached,
        get_ctx_cache_info,
    )

    geom1 = _geom_default()
    geom2 = _geom_diff_esp()

    ctx1 = build_static_context_cached(**geom1)
    ctx2 = build_static_context_cached(**geom2)
    assert get_ctx_cache_info()["n_entries"] == 2
    assert ctx1 is not ctx2


# ──────────────────────────────────────────────────────────────────────────────
# T3: miss por strategy diferente
# ──────────────────────────────────────────────────────────────────────────────


def test_t3_miss_por_strategy_diferente():
    """`strategy="bucketed"` vs `"unified"` → entries distintas."""
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context_cached,
        get_ctx_cache_info,
    )

    geom = _geom_default()
    ctx_b = build_static_context_cached(**geom)
    geom["strategy"] = "unified"
    ctx_u = build_static_context_cached(**geom)

    assert get_ctx_cache_info()["n_entries"] == 2
    assert ctx_b.strategy == "bucketed"
    assert ctx_u.strategy == "unified"


# ──────────────────────────────────────────────────────────────────────────────
# T4: miss por complex_dtype diferente
# ──────────────────────────────────────────────────────────────────────────────


def test_t4_miss_por_complex_dtype_diferente():
    """`complex_dtype="complex128"` vs `"complex64"` → entries distintas."""
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context_cached,
        get_ctx_cache_info,
    )

    geom = _geom_default()
    ctx_128 = build_static_context_cached(**geom)
    geom["complex_dtype"] = "complex64"
    ctx_64 = build_static_context_cached(**geom)

    assert get_ctx_cache_info()["n_entries"] == 2
    assert ctx_128.complex_dtype == "complex128"
    assert ctx_64.complex_dtype == "complex64"


# ──────────────────────────────────────────────────────────────────────────────
# T5: LRU eviction (maxsize=2, 3 geometrias distintas)
# ──────────────────────────────────────────────────────────────────────────────


def test_t5_lru_eviction():
    """3 builds distintos com maxsize=2 → cache mantém só os 2 últimos."""
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context_cached,
        get_ctx_cache_info,
        set_ctx_cache_maxsize,
    )

    set_ctx_cache_maxsize(2)

    # Build A (TR=1.0)
    geom_a = _geom_default()
    geom_a["tr_spacing_m"] = 1.0
    build_static_context_cached(**geom_a)
    assert get_ctx_cache_info()["n_entries"] == 1

    # Build B (TR=1.5)
    geom_b = _geom_default()
    geom_b["tr_spacing_m"] = 1.5
    build_static_context_cached(**geom_b)
    assert get_ctx_cache_info()["n_entries"] == 2

    # Build C (TR=2.0) — evicta A (mais antigo)
    geom_c = _geom_default()
    geom_c["tr_spacing_m"] = 2.0
    build_static_context_cached(**geom_c)
    assert get_ctx_cache_info()["n_entries"] == 2

    # Refazer A → miss (foi evictado)
    ctx_a_old_info = get_ctx_cache_info()["n_entries"]
    build_static_context_cached(**geom_a)
    # Cache ainda em 2 (A entra, B é evictado agora)
    assert get_ctx_cache_info()["n_entries"] == 2
    assert ctx_a_old_info == 2


# ──────────────────────────────────────────────────────────────────────────────
# T6: paridade ctx_cached == ctx direto (estrutura)
# ──────────────────────────────────────────────────────────────────────────────


def test_t6_paridade_ctx_cached_vs_direto():
    """`build_static_context_cached` retorna estrutura idêntica ao build direto."""
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context,
        build_static_context_cached,
    )

    geom = _geom_default()
    ctx_direct = build_static_context(**geom)
    ctx_cached = build_static_context_cached(**geom)

    # Mesma estrutura
    assert ctx_direct.n == ctx_cached.n
    assert ctx_direct.npt == ctx_cached.npt
    assert ctx_direct.dip_rad == ctx_cached.dip_rad
    assert ctx_direct.dz_half == ctx_cached.dz_half
    assert ctx_direct.r_half == ctx_cached.r_half
    assert ctx_direct.strategy == ctx_cached.strategy
    assert ctx_direct.complex_dtype == ctx_cached.complex_dtype
    np.testing.assert_array_equal(ctx_direct.camad_t_array, ctx_cached.camad_t_array)
    np.testing.assert_array_equal(ctx_direct.camad_r_array, ctx_cached.camad_r_array)


# ──────────────────────────────────────────────────────────────────────────────
# T7: clear_ctx_cache + get_ctx_cache_info funcionais
# ──────────────────────────────────────────────────────────────────────────────


def test_t7_clear_e_info_funcionais():
    """`clear_ctx_cache` esvazia + retorna contagem; `get_ctx_cache_info` reflete."""
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context_cached,
        clear_ctx_cache,
        get_ctx_cache_info,
    )

    # Build 3 entries distintas (variando TR)
    for L in (1.0, 1.5, 2.0):
        geom = _geom_default()
        geom["tr_spacing_m"] = L
        build_static_context_cached(**geom)

    info = get_ctx_cache_info()
    assert info["n_entries"] == 3
    assert info["maxsize"] == 32

    n_removed = clear_ctx_cache()
    assert n_removed == 3
    assert get_ctx_cache_info()["n_entries"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# T8: forward_pure_jax produz mesmo H usando ctx_cached vs ctx direto
# ──────────────────────────────────────────────────────────────────────────────


def test_t8_forward_pure_jax_paridade_cached_vs_direto():
    """`forward_pure_jax(ctx_cached)` produz mesmo H que `forward_pure_jax(ctx_direct)`."""
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context,
        build_static_context_cached,
        forward_pure_jax,
    )

    geom = _geom_default()
    rho_h_jnp = jnp.asarray(geom["rho_h"], dtype=jnp.float64)
    rho_v_jnp = jnp.asarray(geom["rho_v"], dtype=jnp.float64)

    ctx_direct = build_static_context(**geom)
    ctx_cached = build_static_context_cached(**geom)

    H_direct = forward_pure_jax(rho_h_jnp, rho_v_jnp, ctx_direct)
    H_cached = forward_pure_jax(rho_h_jnp, rho_v_jnp, ctx_cached)

    # Paridade bit-a-bit (mesmo path, mesma física)
    np.testing.assert_array_equal(np.asarray(H_direct), np.asarray(H_cached))
    assert H_direct.dtype == H_cached.dtype


# ──────────────────────────────────────────────────────────────────────────────
# Extra: set_ctx_cache_maxsize validação de input
# ──────────────────────────────────────────────────────────────────────────────


def test_extra_set_maxsize_negativo_levanta_erro():
    """`set_ctx_cache_maxsize(-1)` levanta ValueError."""
    from geosteering_ai.simulation._jax.forward_pure import set_ctx_cache_maxsize

    with pytest.raises(ValueError, match="maxsize deve ser >= 1"):
        set_ctx_cache_maxsize(-1)


def test_extra_set_maxsize_zero_levanta_erro():
    """`set_ctx_cache_maxsize(0)` levanta ValueError (alinhado ao sibling JIT).

    Regressão (review O2/O3, finding P1): ``maxsize=0`` era aceito mas
    tornava a eviction LRU degenerada — ``build_static_context_cached``
    chamava ``popitem`` em cache vazio (``0 >= 0`` → ``KeyError``). Agora
    o setter rejeita ``< 1`` e a eviction usa ``while … max(MAXSIZE, 1)``.
    """
    from geosteering_ai.simulation._jax.forward_pure import set_ctx_cache_maxsize

    with pytest.raises(ValueError, match="maxsize deve ser >= 1"):
        set_ctx_cache_maxsize(0)
    # Restaura default para não vazar estado entre testes.
    set_ctx_cache_maxsize(32)


def test_extra_ctx_cache_no_collision_n1_n2():
    """Regressão O4: chave NÃO colide entre n=1 e n=2 (esp vazio em ambos).

    Bug (exposto pelo path batched-bucketed): ``_hash_ctx_key`` omitia o nº de
    camadas ``n``. Para n=1 e n=2 o ``esp`` é vazio ``(0,)`` em AMBOS, e se
    positions/freqs/TR/dip forem iguais, a chave colidia → o ctx de n=1
    (``h_arr`` shape (1,)) era retornado para uma chamada n=2 → mismatch
    (1,) vs (2,2) sob vmap. Fix: ``n`` entra na chave.
    """
    from geosteering_ai.simulation._jax.forward_pure import (
        build_static_context_cached,
        clear_ctx_cache,
    )

    clear_ctx_cache()
    pos = np.linspace(-5.0, 5.0, 12)
    freqs = np.array([20000.0])

    # n=1 PRIMEIRO (popula o cache) — esp vazio.
    ctx1 = build_static_context_cached(
        rho_h=np.array([10.0]),
        rho_v=np.array([10.0]),
        esp=np.empty(0, dtype=np.float64),
        positions_z=pos,
        freqs_hz=freqs,
        tr_spacing_m=1.0,
        dip_deg=0.0,
        strategy="bucketed",
        complex_dtype="complex128",
    )
    # n=2 com MESMA geometria-de-chave (esp vazio, mesmas pos/freqs/TR/dip).
    ctx2 = build_static_context_cached(
        rho_h=np.array([10.0, 20.0]),
        rho_v=np.array([10.0, 20.0]),
        esp=np.empty(0, dtype=np.float64),
        positions_z=pos,
        freqs_hz=freqs,
        tr_spacing_m=1.0,
        dip_deg=0.0,
        strategy="bucketed",
        complex_dtype="complex128",
    )

    assert ctx1.n == 1
    assert ctx2.n == 2  # se colidisse, retornaria ctx1 (n=1)
    assert tuple(ctx1.h_arr_jnp.shape) == (1,)
    assert tuple(ctx2.h_arr_jnp.shape) == (2,)  # NÃO (1,) — prova ausência de colisão
