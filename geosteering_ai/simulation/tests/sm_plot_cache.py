# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_plot_cache.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Cache LRU de plot_bundles             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-23                                                 ║
# ║  Status      : Produção (v2.4c)                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Gestão de memória para tensores H de simulações históricas no          ║
# ║    Simulation Manager. Mantém apenas os N snapshots mais recentes em      ║
# ║    RAM, evitando travamento da GUI quando o usuário executa múltiplas     ║
# ║    simulações de 1000+ modelos (cada H_stack pode chegar a 860 MB).      ║
# ║                                                                           ║
# ║  ESTRUTURA                                                                ║
# ║    ┌──────────────────────────────────────────────────────────────────┐   ║
# ║    │  LRUPlotCache(maxlen=3, max_bytes=500e6)                         │   ║
# ║    │    ├─ put(key, bundle)   → List[evicted_keys]                    │   ║
# ║    │    ├─ get(key)           → bundle (move-to-end MRU)              │   ║
# ║    │    ├─ clear()            → bytes freed                           │   ║
# ║    │    ├─ total_bytes()      → soma de H_stack.nbytes                │   ║
# ║    │    ├─ __len__/__contains__/keys                                  │   ║
# ║    │    └─ evicta quando len > maxlen OU bytes > max_bytes            │   ║
# ║    └──────────────────────────────────────────────────────────────────┘   ║
# ║                                                                           ║
# ║  INVARIANTES                                                              ║
# ║    • Nunca toca tensores JAX / Numba / Fortran (plotagem NumPy apenas).  ║
# ║    • Backend-agnostic: armazena dicts opacos.                            ║
# ║    • Thread-safety: NÃO thread-safe. Uso exclusivo no main thread Qt.   ║
# ║    • Preserva `self._current_sim` — este cache só gerencia histórico.   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Cache LRU de plot_bundles por snapshot_id, com limite duplo (N, bytes)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["LRUPlotCache"]


class LRUPlotCache:
    """Cache LRU (Least Recently Used) de plot_bundles por ``snapshot_id``.

    Mantém no máximo ``maxlen`` items ou até ``max_bytes`` de consumo de RAM
    (estimado via ``H_stack.nbytes`` + ``z_obs.nbytes``). Quando qualquer
    limite é ultrapassado, remove os itens menos recentemente acessados em
    ordem FIFO (primeiro inserido, primeiro removido).

    Attributes:
        maxlen: Número máximo de bundles em cache. Default 3.
        max_bytes: Limite superior de bytes totais. Default 500 MB.

    Example:
        >>> cache = LRUPlotCache(maxlen=2)
        >>> cache.put("a", {"H_stack": np.zeros((10, 10), dtype=complex)})
        []
        >>> cache.put("b", {"H_stack": np.zeros((10, 10), dtype=complex)})
        []
        >>> cache.put("c", {"H_stack": np.zeros((10, 10), dtype=complex)})
        ['a']
        >>> sorted(cache.keys())
        ['b', 'c']

    Note:
        Este cache NÃO toca backends de simulação (JAX/Numba/Fortran).
        Operações são todas O(1) amortizado via OrderedDict.
    """

    def __init__(self, maxlen: int = 3, max_bytes: float = 500e6) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        if max_bytes < 0:
            raise ValueError(f"max_bytes must be >= 0, got {max_bytes}")
        self.maxlen: int = int(maxlen)
        self.max_bytes: float = float(max_bytes)
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._sizes: Dict[str, int] = {}

    # ── API pública ─────────────────────────────────────────────────────

    def put(self, key: str, bundle: Dict[str, Any]) -> List[str]:
        """Insere ou atualiza um bundle e aplica política LRU.

        Args:
            key: Identificador único (tipicamente ``snapshot_id`` UUID4).
            bundle: Dict com ao menos ``H_stack`` (ndarray) — opaco.

        Returns:
            Lista de keys que foram evictadas (FIFO) para respeitar os
            limites. Lista vazia se nada foi removido.

        Note:
            Se ``key`` já existe, é atualizado e movido para o fim (MRU).
            Contagem de bytes é recalculada.
        """
        if key in self._store:
            # Atualização — remove entrada antiga (libera contagem de bytes)
            del self._store[key]
            self._sizes.pop(key, None)

        self._store[key] = bundle
        self._sizes[key] = self.estimate_bytes(bundle)

        # Aplica política LRU: evicta enquanto passar de qualquer limite
        evicted: List[str] = []
        while len(self._store) > self.maxlen or self.total_bytes() > self.max_bytes:
            if not self._store:
                break
            old_key, _ = self._store.popitem(last=False)  # FIFO (oldest)
            self._sizes.pop(old_key, None)
            evicted.append(old_key)
            # Se o cache ficou vazio mas ainda acima de max_bytes (item gigante),
            # paramos aqui — o item atual fica mesmo acima do limite.
            if not self._store:
                break
        return evicted

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Recupera um bundle e o marca como mais recentemente acessado.

        Args:
            key: Identificador do bundle.

        Returns:
            Bundle armazenado ou ``None`` se chave não existe.
        """
        if key not in self._store:
            return None
        # Move para o fim (MRU) — O(1) em OrderedDict
        self._store.move_to_end(key, last=True)
        return self._store[key]

    def clear(self) -> int:
        """Remove todos os bundles do cache.

        Returns:
            Número de snapshots removidos.
        """
        n = len(self._store)
        self._store.clear()
        self._sizes.clear()
        return n

    def total_bytes(self) -> int:
        """Soma bytes estimados de todos os bundles atualmente em cache."""
        return int(sum(self._sizes.values()))

    def keys(self) -> List[str]:
        """Lista ordenada (LRU→MRU) das chaves em cache."""
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def estimate_bytes(bundle: Dict[str, Any]) -> int:
        """Estima RAM consumida por um bundle somando ``.nbytes`` dos arrays.

        Considera as chaves pesadas conhecidas: ``H_stack``, ``H_n``, ``H_f``,
        ``z_obs``. Ignora chaves sem atributo ``.nbytes``.

        Args:
            bundle: Dict opaco com possíveis ndarrays.

        Returns:
            Bytes estimados (inteiro). Zero se nenhum array for reconhecido.
        """
        total = 0
        heavy_keys = ("H_stack", "H_n", "H_f", "z_obs")
        for k in heavy_keys:
            arr = bundle.get(k)
            if arr is None:
                continue
            nbytes = getattr(arr, "nbytes", None)
            if nbytes is not None:
                total += int(nbytes)
                continue
            # Fallback — numpy conversion (não deveria acontecer em produção)
            try:
                total += int(np.asarray(arr).nbytes)
            except (TypeError, ValueError):
                pass
        return total
