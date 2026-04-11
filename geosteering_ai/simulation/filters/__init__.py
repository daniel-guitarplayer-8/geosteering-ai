# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/filters/__init__.py                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Filtros Hankel Digitais                 ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 1.1)                                   ║
# ║  Última Rev. : 2026-04-11 (pós-revisão: D1 completo)                     ║
# ║  Status      : Produção                                                  ║
# ║  Framework   : NumPy 2.x                                                  ║
# ║  Dependências: numpy (runtime), pytest (testes)                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Pacote de acesso aos filtros Hankel digitais (Werthmüller 201pt,      ║
# ║    Kong 61pt, Anderson 801pt) usados pelo simulador Python otimizado.   ║
# ║    Este módulo re-exporta `FilterLoader` e `HankelFilter` para permitir  ║
# ║    importação direta via `from geosteering_ai.simulation.filters import`.║
# ║                                                                           ║
# ║  ARQUITETURA                                                              ║
# ║    ┌─────────────────────────────────────────────────────────────────┐  ║
# ║    │  Caller (Numba/JAX kernel)                                      │  ║
# ║    │     │                                                           │  ║
# ║    │     │  from ...filters import FilterLoader, HankelFilter        │  ║
# ║    │     ▼                                                           │  ║
# ║    │  __init__.py  (este arquivo)                                    │  ║
# ║    │     │                                                           │  ║
# ║    │     │  re-export                                                │  ║
# ║    │     ▼                                                           │  ║
# ║    │  loader.py    (implementação)                                  │  ║
# ║    │     │                                                           │  ║
# ║    │     ▼                                                           │  ║
# ║    │  *.npz        (artefatos binários)                             │  ║
# ║    └─────────────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  API PÚBLICA                                                              ║
# ║    FilterLoader   — fábrica com cache classe-level thread-safe          ║
# ║    HankelFilter   — dataclass imutável (frozen, arrays read-only)       ║
# ║                                                                           ║
# ║  ARTEFATOS INSTALADOS                                                     ║
# ║    • werthmuller_201pt.npz  (★ default, filter_type=0)                  ║
# ║    • kong_61pt.npz          (filter_type=1, ~3.3× mais rápido)          ║
# ║    • anderson_801pt.npz     (filter_type=2, máxima precisão)            ║
# ║                                                                           ║
# ║  ESTADO ATUAL                                                             ║
# ║    Sprint 1.1 concluída: 3 filtros extraídos, FilterLoader implementado,║
# ║    45 testes passando. Pronto para consumo por kernels Numba/JAX nas    ║
# ║    Fases 2-3.                                                            ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • geosteering_ai/simulation/filters/README.md                        ║
# ║    • geosteering_ai/simulation/filters/loader.py                        ║
# ║    • Fortran_Gerador/filtersv2.f08 (fonte primária, 5559 LOC)          ║
# ║    • scripts/extract_hankel_weights.py                                  ║
# ║    • docs/reference/plano_simulador_python_jax_numba.md                ║
# ║                                                                           ║
# ║  NOTAS DE IMPLEMENTAÇÃO                                                  ║
# ║    Este é um módulo de re-export puro: não contém lógica além dos       ║
# ║    imports. Toda a funcionalidade está em `loader.py`. A separação     ║
# ║    serve para expor uma API pública estável (este __init__) mesmo se   ║
# ║    a organização interna de `loader.py` for refatorada no futuro.      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Subsistema de filtros Hankel digitais — fachada de re-export.

Re-exporta `FilterLoader` e `HankelFilter` para permitir importação direta::

    from geosteering_ai.simulation.filters import FilterLoader, HankelFilter

    loader = FilterLoader()
    filt = loader.load("werthmuller_201pt")
    print(filt.npt)  # 201

Example:
    Uso típico em um kernel Numba (Fase 2, pendente)::

        from geosteering_ai.simulation.filters import FilterLoader

        _filt = FilterLoader().load("werthmuller_201pt")
        _abscissas = _filt.abscissas   # read-only
        _weights_j0 = _filt.weights_j0 # read-only
        _weights_j1 = _filt.weights_j1 # read-only

        @njit(parallel=True, fastmath=True, cache=True)
        def hankel_quadrature(kr_fn, r):
            # uses _abscissas, _weights_j0 from closure
            ...
"""
from __future__ import annotations

from geosteering_ai.simulation.filters.loader import FilterLoader, HankelFilter

__all__ = [
    "FilterLoader",
    "HankelFilter",
]
