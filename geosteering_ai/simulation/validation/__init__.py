# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/validation/__init__.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Casos de Validação Analítica            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 1.3)                                   ║
# ║  Status      : Produção                                                  ║
# ║  Framework   : NumPy 2.x                                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Provê soluções analíticas fechadas para casos canônicos de EM 1D    ║
# ║    que servem como "ground truth" independente para validação dos      ║
# ║    backends numéricos (Numba na Fase 2, JAX na Fase 3).                ║
# ║                                                                           ║
# ║    A motivação é: **um teste que compara o solver consigo mesmo não    ║
# ║    detecta bugs**. Precisamos de uma fonte de verdade externa,         ║
# ║    matematicamente confiável. Os casos analíticos aqui são:            ║
# ║      1. Limite estático (decoupling factors ACp, ACx)                  ║
# ║      2. Skin depth de plano homogêneo (closed-form)                    ║
# ║      3. VMD em full-space isotrópico (Ward-Hohmann 1988)               ║
# ║                                                                           ║
# ║  EXPORTS PÚBLICOS                                                         ║
# ║    • static_decoupling_factors(L)  → (ACp, ACx)                         ║
# ║    • skin_depth(f, rho)            → δ em metros                        ║
# ║    • wavenumber_quasi_static(f, rho) → k complexo                      ║
# ║    • vmd_fullspace_axial(L, f, rho, m) → H complexo                    ║
# ║    • vmd_fullspace_broadside(L, f, rho, m) → H complexo                ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Ward, S.H. & Hohmann, G.W. (1988). "Electromagnetic theory for    ║
# ║      geophysical applications". SEG Investigations in Geophysics 3.    ║
# ║    • Moran, J.H. & Gianzero, S. (1979). "Effects of formation           ║
# ║      anisotropy on resistivity-logging measurements." Geophysics 44.  ║
# ║    • Nabighian, M.N. (1988). "Electromagnetic methods in applied       ║
# ║      geophysics — Theory, Vol. 1." SEG.                                ║
# ║                                                                           ║
# ║  CONVENÇÕES                                                               ║
# ║    • Convenção temporal: e^(-iωt) (geofísica padrão, Moran-Gianzero)  ║
# ║    • k² = iωμ₀σ (quasi-static) → k com parte imaginária positiva     ║
# ║    • Factor de atenuação: e^(ikr) com Im(k) > 0 → atenuação com r    ║
# ║    • μ₀ = 4π × 10⁻⁷ H/m (permeabilidade do vácuo)                     ║
# ║    • σ = 1/ρ (condutividade, S/m)                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Soluções analíticas de referência para validação dos backends numéricos.

Este subpacote fornece funções puras em NumPy que computam soluções
analíticas fechadas para cenários canônicos de EM 1D. Estas funções são
usadas como ground-truth independente para validar os backends Numba e
JAX a serem implementados nas Fases 2 e 3.

Example:
    Uso típico em um teste de regressão::

        >>> from geosteering_ai.simulation.validation import (
        ...     static_decoupling_factors,
        ...     skin_depth,
        ...     vmd_fullspace_axial,
        ... )
        >>> ACp, ACx = static_decoupling_factors(L=1.0)
        >>> round(ACp, 6)
        -0.079577
        >>> round(ACx, 6)
        0.159155

Note:
    Todas as funções são puras (sem estado global) e thread-safe. Usam
    apenas NumPy, sem dependências externas. Podem ser chamadas em
    contextos Numba `@njit` (sem JIT, pois usam apenas operações
    primitivas) ou JAX `@jit` (após conversão trivial para `jnp`).
"""
from __future__ import annotations

from geosteering_ai.simulation.validation.half_space import (
    MU_0,
    skin_depth,
    static_decoupling_factors,
    vmd_fullspace_axial,
    vmd_fullspace_broadside,
    wavenumber_quasi_static,
)

__all__ = [
    "MU_0",
    "skin_depth",
    "static_decoupling_factors",
    "vmd_fullspace_axial",
    "vmd_fullspace_broadside",
    "wavenumber_quasi_static",
]
