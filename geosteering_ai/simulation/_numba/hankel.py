# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_numba/hankel.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Quadratura Hankel Digital (Sprint 2.3)  ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x + Numba 0.60+ (dual-mode, opcional)              ║
# ║  Dependências: numpy (obrigatório), numba (opcional, speedup JIT)         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Helpers para a quadratura de Hankel digital (integração discreta      ║
# ║    ∫ f(kr) Jν(kr·r) dkr ≈ (1/r) · Σₖ f(kr_k/r) · wJν_k) usada em vários  ║
# ║    módulos do simulador (dipoles.py já faz a integração inline,          ║
# ║    mas este módulo expõe helpers reutilizáveis + o método `integrate`    ║
# ║    para consumidores externos).                                          ║
# ║                                                                           ║
# ║    NOTE: a integração propriamente dita (Σ f·wJν) já está embebida       ║
# ║    nos kernels `hmd_tiv` e `vmd` da Sprint 2.2 — este módulo expõe      ║
# ║    os helpers abaixo como API de conveniência e documentação.           ║
# ║                                                                           ║
# ║  CAMADA DE ABSTRAÇÃO                                                      ║
# ║    ┌──────────────────────────────────────────────────────────────┐     ║
# ║    │ filters/loader.py (Sprint 1.1)                               │     ║
# ║    │   └── FilterLoader.load(name) → HankelFilter                 │     ║
# ║    │         (abscissas, weights_j0, weights_j1, fortran_type)    │     ║
# ║    │            │                                                 │     ║
# ║    │            ▼                                                 │     ║
# ║    │ _numba/hankel.py (Sprint 2.3)      ← este arquivo           │     ║
# ║    │   ├── prepare_kr(npt, krJ0J1, hordist) → kr                  │     ║
# ║    │   ├── integrate_j0(f_kr, wJ0, r) → complex                   │     ║
# ║    │   ├── integrate_j1(f_kr, wJ1, r) → complex                   │     ║
# ║    │   └── integrate_j0_j1(f_kr, wJ0, wJ1, r) → (complex, complex)║     ║
# ║    │            │                                                 │     ║
# ║    │            ▼                                                 │     ║
# ║    │ _numba/dipoles.py (Sprint 2.2) consome diretamente           │     ║
# ║    │   (por enquanto inline, migração opcional futura)            │     ║
# ║    └──────────────────────────────────────────────────────────────┘     ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Anderson, W.L. (1989). A hybrid fast Hankel transform algorithm    ║
# ║      for EM modeling. Geophysics 54, 263-266.                           ║
# ║    • Kong, F.N. (2007). Hankel transform filters for dipole antenna     ║
# ║      radiation in a conductive medium. Geophysical Prospecting 55.     ║
# ║    • Werthmüller, D. (2015). empymod Documentation — digital filter    ║
# ║      tables.                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Helpers para quadratura Hankel digital (Sprint 2.3).

Funções auxiliares que encapsulam operações frequentes em integrais de
Hankel digitais aplicadas no simulador. A maioria dos kernels da
Sprint 2.2 (`hmd_tiv`, `vmd`) já faz a integração inline para
performance máxima — este módulo expõe o mesmo algoritmo como funções
reutilizáveis para consumidores externos, testes e documentação.

Example:
    Cálculo canônico ``H = Σ f(kr) · w_J0``::

        >>> import numpy as np
        >>> from geosteering_ai.simulation._numba.hankel import (
        ...     prepare_kr, integrate_j0,
        ... )
        >>> from geosteering_ai.simulation.filters import FilterLoader
        >>> filt = FilterLoader().load("werthmuller_201pt")
        >>> hordist = 1.0  # 1 metro
        >>> kr = prepare_kr(hordist, filt.abscissas)
        >>> f_kr = np.ones(filt.abscissas.shape[0], dtype=np.complex128)
        >>> integral = integrate_j0(f_kr, filt.weights_j0, hordist)
        >>> # Resultado: ∫₀^∞ J₀(kr·r)·1·dkr = δ(r) (distribuição)
        >>> # Numericamente: soma truncada dos pesos ÷ r
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from geosteering_ai.simulation._numba.propagation import njit

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
# Threshold para distância horizontal singular (paridade dipoles.py).
_HORDIST_SINGULARITY_EPS: float = 1.0e-9
_R_GUARD: float = 1.0e-2


@njit
def prepare_kr(hordist: float, krJ0J1: np.ndarray) -> np.ndarray:
    """Escala as abscissas do filtro Hankel pela distância horizontal.

    Transforma as abscissas adimensionais do filtro digital em
    wavenumbers físicos `kr = krJ0J1 / r` usados pelos kernels do
    forward. Aplica o guard de singularidade quando `r → 0`.

    Args:
        hordist: Distância horizontal transmissor-receptor em metros.
            Se < 1e-9, usa `r = 0.01 m` como guard (paridade Fortran).
        krJ0J1: Array `(npt,)` float64 com abscissas adimensionais do
            filtro (lidas de `FilterLoader.load(...).abscissas`).

    Returns:
        Array `(npt,)` float64 com `kr = krJ0J1 / r` em unidades 1/m.

    Example:
        >>> import numpy as np
        >>> abs_filter = np.array([0.1, 1.0, 10.0])
        >>> kr = prepare_kr(2.0, abs_filter)
        >>> kr
        array([0.05, 0.5 , 5.  ])
    """
    if hordist < _HORDIST_SINGULARITY_EPS:
        r = _R_GUARD
    else:
        r = hordist
    return krJ0J1 / r


@njit
def integrate_j0(f_kr: np.ndarray, weights_j0: np.ndarray, hordist: float) -> complex:
    """Calcula a integral de Hankel digital do tipo J₀.

    Aproxima a integral contínua via quadratura digital:

        ∫₀^∞ f(kr) · J₀(kr · r) · dkr ≈ (1/r) · Σₖ f(kr_k / r) · wJ0_k

    Args:
        f_kr: Array `(npt,)` complex128 com `f(kr)` já avaliada nas
            abscissas escaladas `kr = krJ0J1 / r` (use
            :func:`prepare_kr` para obter as abscissas).
        weights_j0: Array `(npt,)` float64 com pesos J₀ do filtro.
        hordist: Distância horizontal em metros (usada para dividir
            no final, replicando o `1/r` externo do filtro).

    Returns:
        Integral aproximada como um escalar complex128.

    Note:
        A convenção de multiplicar por `1/r` no final é a utilizada
        por Werthmüller, Kong e Anderson — os pesos `wJ0_k` são
        adimensionais e devem ser multiplicados por `1/r` pós-soma.
    """
    if hordist < _HORDIST_SINGULARITY_EPS:
        r = _R_GUARD
    else:
        r = hordist
    # Soma discreta
    total = complex(0.0, 0.0)
    for k in range(f_kr.shape[0]):
        total = total + f_kr[k] * weights_j0[k]
    return total / r


@njit
def integrate_j1(f_kr: np.ndarray, weights_j1: np.ndarray, hordist: float) -> complex:
    """Calcula a integral de Hankel digital do tipo J₁.

    Aproxima:

        ∫₀^∞ f(kr) · J₁(kr · r) · dkr ≈ (1/r) · Σₖ f(kr_k / r) · wJ1_k

    Args:
        f_kr: Array `(npt,)` complex128 com `f(kr)` já avaliada.
        weights_j1: Array `(npt,)` float64 com pesos J₁ do filtro.
        hordist: Distância horizontal em metros.

    Returns:
        Integral aproximada como escalar complex128.
    """
    if hordist < _HORDIST_SINGULARITY_EPS:
        r = _R_GUARD
    else:
        r = hordist
    total = complex(0.0, 0.0)
    for k in range(f_kr.shape[0]):
        total = total + f_kr[k] * weights_j1[k]
    return total / r


@njit
def integrate_j0_j1(
    f_kr: np.ndarray,
    weights_j0: np.ndarray,
    weights_j1: np.ndarray,
    hordist: float,
) -> Tuple[complex, complex]:
    """Calcula as integrais J₀ e J₁ simultaneamente.

    Otimização sobre chamar :func:`integrate_j0` e :func:`integrate_j1`
    separadamente: percorre o array `f_kr` uma única vez.

    Args:
        f_kr: Array `(npt,)` complex128.
        weights_j0: Array `(npt,)` float64 — pesos J₀.
        weights_j1: Array `(npt,)` float64 — pesos J₁.
        hordist: Distância horizontal em metros.

    Returns:
        Tupla `(integral_j0, integral_j1)` de dois `complex`.
    """
    if hordist < _HORDIST_SINGULARITY_EPS:
        r = _R_GUARD
    else:
        r = hordist
    tot_j0 = complex(0.0, 0.0)
    tot_j1 = complex(0.0, 0.0)
    for k in range(f_kr.shape[0]):
        tot_j0 = tot_j0 + f_kr[k] * weights_j0[k]
        tot_j1 = tot_j1 + f_kr[k] * weights_j1[k]
    return tot_j0 / r, tot_j1 / r


__all__ = [
    "prepare_kr",
    "integrate_j0",
    "integrate_j1",
    "integrate_j0_j1",
]
