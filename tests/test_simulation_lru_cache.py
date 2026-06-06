# -*- coding: utf-8 -*-
"""Testes do cache LRU configurável (Sprint v2.29.2).

Valida:
- ``default_max_bytes()`` retorna 10% RAM com piso 500 MB e teto 4 GB
- Fallback para 500 MB quando ``psutil`` indisponível
- ``LRUPlotCache(max_bytes=None)`` ativa auto-detect
- Retrocompatibilidade: ``LRUPlotCache(max_bytes=500e6)`` mantém comportamento histórico
- Política LRU + eviction funciona com novos limites
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest


def test_default_max_bytes_in_expected_range() -> None:
    """``default_max_bytes()`` retorna valor entre 500 MB e 4 GB (piso/teto)."""
    from geosteering_ai.gui.persistence.plot_cache import default_max_bytes

    val = default_max_bytes()
    assert 500e6 <= val <= 4e9, (
        f"default_max_bytes()={val:.2e} fora do range [500 MB, 4 GB] "
        f"esperado (piso 500 MB para RAM <5 GB, teto 4 GB para RAM >40 GB)."
    )


def test_default_max_bytes_fallback_without_psutil(monkeypatch) -> None:
    """Fallback retorna exatamente 500e6 quando ``psutil`` indisponível.

    Implementação: ``default_max_bytes`` faz ``import psutil`` lazy dentro
    do try/except. ``monkeypatch.setattr`` injeta um ``__import__`` falso
    que rejeita "psutil"; ao final do teste, o fixture do pytest restaura
    automaticamente o estado original (sem necessidade de finally manual
    nem ``importlib.reload``).
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil simulado indisponível")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from geosteering_ai.gui.persistence.plot_cache import default_max_bytes

    assert default_max_bytes() == 500e6, (
        "Fallback sem psutil deve retornar EXATAMENTE 500 MB "
        "(preserva comportamento histórico v2.4c–v2.29.1)."
    )


def test_lru_cache_accepts_none_max_bytes() -> None:
    """``LRUPlotCache(max_bytes=None)`` deve usar ``default_max_bytes()``."""
    from geosteering_ai.gui.persistence.plot_cache import (
        LRUPlotCache,
        default_max_bytes,
    )

    cache = LRUPlotCache(maxlen=3, max_bytes=None)
    assert cache.max_bytes == default_max_bytes(), (
        f"max_bytes=None deve resolver para default_max_bytes()="
        f"{default_max_bytes():.2e}, mas cache.max_bytes={cache.max_bytes:.2e}"
    )


def test_lru_cache_backward_compat_500mb() -> None:
    """Retrocompatibilidade: ``max_bytes=500e6`` mantém o comportamento histórico."""
    from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache

    cache = LRUPlotCache(maxlen=3, max_bytes=500e6)
    assert cache.max_bytes == 500e6


def test_lru_cache_custom_max_bytes_evicts_correctly() -> None:
    """Política LRU eviction funciona com novos limites custom — testa
    eviction por maxlen E por bytes.
    """
    from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache

    # ── Cenário A: eviction por maxlen ────────────────────────────────
    small = {"H_stack": np.zeros((100, 100), dtype=np.complex128)}  # ~160 KB
    cache_maxlen = LRUPlotCache(maxlen=2, max_bytes=2 * 1024 * 1024 * 1024)
    cache_maxlen.put("a", small)
    cache_maxlen.put("b", small)
    evicted = cache_maxlen.put("c", small)
    assert evicted == ["a"], f"Esperado evict ['a'] por maxlen, got {evicted}"
    assert sorted(cache_maxlen.keys()) == ["b", "c"]

    # ── Cenário B: eviction por bytes ────────────────────────────────
    # Cria 2 tensores de ~600 MB cada (complex128 = 16 B/elem).
    # Cache com max_bytes=1 GB → primeiro put cabe; segundo put força eviction
    # do primeiro por exceder o limite de bytes combinado (600 + 600 > 1024).
    target_mb = 600.0
    n = int(np.sqrt(target_mb * 1024 * 1024 / 16))
    big1 = {"H_stack": np.zeros((n, n), dtype=np.complex128)}
    big2 = {"H_stack": np.zeros((n, n), dtype=np.complex128)}
    bytes_each = big1["H_stack"].nbytes
    assert bytes_each > 500e6, f"Setup inválido: {bytes_each / 1e6:.0f} MB < 500"

    cache_bytes = LRUPlotCache(maxlen=10, max_bytes=1024 * 1024 * 1024)
    cache_bytes.put("big1", big1)
    assert "big1" in cache_bytes.keys()
    evicted_bytes = cache_bytes.put("big2", big2)
    assert "big1" in evicted_bytes, (
        f"Esperado evict de 'big1' por bytes (limite 1 GB, dois tensores "
        f"de {bytes_each / 1e6:.0f} MB excedem), got evicted={evicted_bytes}"
    )
    assert "big2" in cache_bytes.keys()


def test_lru_cache_was_too_big_with_higher_limit() -> None:
    """Tensor que excederia 500 MB cabe se ``max_bytes`` aumentado para 2 GB."""
    from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache

    # Cria tensor falso de ~600 MB (não cabe em 500 MB, cabe em 2 GB)
    n = int(np.sqrt(600 * 1024 * 1024 / 16))  # complex128 = 16 bytes/elem
    big_bundle = {"H_stack": np.zeros((n, n), dtype=np.complex128)}
    bundle_mb = big_bundle["H_stack"].nbytes / 1e6
    assert (
        bundle_mb > 500
    ), f"Setup inválido: bundle tem {bundle_mb:.0f} MB, esperado >500"

    # Limite 500 MB — bundle excede, é auto-evictado
    cache_500 = LRUPlotCache(maxlen=3, max_bytes=500e6)
    cache_500.put("big", big_bundle)
    assert cache_500.was_too_big(
        "big"
    ), "Bundle >500 MB deveria ser marcado como too_big em cache de 500 MB"

    # Limite 2 GB — bundle cabe, NÃO é too_big
    cache_2gb = LRUPlotCache(maxlen=3, max_bytes=2 * 1024 * 1024 * 1024)
    cache_2gb.put("big", big_bundle)
    assert not cache_2gb.was_too_big("big"), (
        f"Bundle de {bundle_mb:.0f} MB deveria caber em cache de 2 GB "
        f"(was_too_big={cache_2gb.was_too_big('big')})"
    )
    assert cache_2gb.get("big") is not None


def test_lru_cache_validates_negative_max_bytes() -> None:
    """``max_bytes < 0`` ainda dispara ``ValueError`` (validação preservada)."""
    from geosteering_ai.gui.persistence.plot_cache import LRUPlotCache

    with pytest.raises(ValueError, match="max_bytes must be >= 0"):
        LRUPlotCache(maxlen=3, max_bytes=-1.0)
