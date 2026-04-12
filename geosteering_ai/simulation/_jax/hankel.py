# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/hankel.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend JAX Hankel (Sprint 3.1)        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : JAX 0.4.30+                                                ║
# ║  Dependências: jax, jaxlib                                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Quadratura digital de Hankel via JAX. Port funcional do módulo       ║
# ║    `_numba/hankel.py`, usando `jnp.einsum` para paralelização            ║
# ║    automática sobre o eixo npt.                                          ║
# ║                                                                           ║
# ║  FÓRMULA                                                                  ║
# ║    Quadratura digital do filtro J₀/J₁:                                   ║
# ║                                                                           ║
# ║        F(r) = ∫₀^∞ f(kr) · Jν(kr · r) · dkr                            ║
# ║             ≈ (1/r) · Σᵢ f(kᵢ/r) · wᵢ(ν)                                ║
# ║                                                                           ║
# ║    Onde:                                                                  ║
# ║      kᵢ = filter.abscissas                                               ║
# ║      wᵢ(0) = filter.weights_j0                                          ║
# ║      wᵢ(1) = filter.weights_j1                                          ║
# ║      f(kr) = valores da função (kernel) nas abscissas                   ║
# ║                                                                           ║
# ║  DESIGN JAX                                                               ║
# ║    • `@jax.jit` em cada função — compilação XLA automática               ║
# ║    • `jnp.einsum` para dot product — XLA gera BLAS (cblas_zgemv)        ║
# ║    • Arrays de entrada são `jnp.ndarray` (ou convertidos via            ║
# ║      `jnp.asarray`); caller é responsável por não ficar alternando       ║
# ║      entre dispositivos CPU/GPU                                           ║
# ║                                                                           ║
# ║  PARIDADE COM NUMBA                                                       ║
# ║    Tolerância esperada: < 1e-13 em float64. Diferenças residuais         ║
# ║    vêm do reordenamento de somas pelo XLA (associativity) — dentro       ║
# ║    do erro de máquina para complex128.                                    ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • _numba/hankel.py — implementação de referência                     ║
# ║    • _numba/dipoles.py — consumidores principais                        ║
# ║    • Werthmüller (2017) Geophysics 82(6) WB9 — filtro 201pt             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Quadratura de Hankel via JAX (Sprint 3.1).

Funções puras + JIT-compiláveis que implementam ∫ f(kr) · Jν(kr·r) dkr
via filtros digitais pré-computados (FilterLoader). Port funcional
equivalente ao :mod:`geosteering_ai.simulation._numba.hankel`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Garantia redundante de float64 (o `__init__.py` já chamou config.update,
# mas módulos podem ser importados em ordens distintas).
jax.config.update("jax_enable_x64", True)


# ──────────────────────────────────────────────────────────────────────────────
# integrate_j0, integrate_j1 — integrais individuais
# ──────────────────────────────────────────────────────────────────────────────


@jax.jit
def integrate_j0(values: jax.Array, weights_j0: jax.Array) -> jax.Array:
    """Calcula ∫ f(kr) · J₀(kr·r) dkr pelo filtro digital.

    Args:
        values: Valores da função f(kr) avaliada nas abscissas do
            filtro. Shape ``(npt,)`` complex128 (tipicamente).
        weights_j0: Pesos do filtro para J₀. Shape ``(npt,)`` float64.

    Returns:
        Escalar complex128 (ou float64 se ``values`` for real) — o
        valor da integral. A divisão por ``r`` (inerente à quadratura
        digital) é responsabilidade do caller.

    Note:
        Equivalente a ``jnp.dot(weights_j0, values)``. O ``einsum`` é
        usado para clareza semântica ("soma sobre i de w_i · v_i").
    """
    return jnp.einsum("i,i->", weights_j0, values)


@jax.jit
def integrate_j1(values: jax.Array, weights_j1: jax.Array) -> jax.Array:
    """Calcula ∫ f(kr) · J₁(kr·r) dkr pelo filtro digital.

    Args:
        values: Valores de f(kr) nas abscissas. Shape ``(npt,)``.
        weights_j1: Pesos do filtro para J₁. Shape ``(npt,)`` float64.

    Returns:
        Escalar — valor da integral (caller divide por r).
    """
    return jnp.einsum("i,i->", weights_j1, values)


# ──────────────────────────────────────────────────────────────────────────────
# integrate_j0_j1 — versão batch que retorna ambas integrais
# ──────────────────────────────────────────────────────────────────────────────


@jax.jit
def integrate_j0_j1(
    values: jax.Array,
    weights_j0: jax.Array,
    weights_j1: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Calcula J₀ e J₁ simultaneamente, aproveitando fusion XLA.

    Args:
        values: Valores de f(kr) nas abscissas. Shape ``(npt,)``.
        weights_j0: Pesos J₀. Shape ``(npt,)`` float64.
        weights_j1: Pesos J₁. Shape ``(npt,)`` float64.

    Returns:
        Tupla ``(integral_j0, integral_j1)`` — ambos escalares.

    Note:
        A execução sob ``@jax.jit`` permite que o XLA **funde** as duas
        operações em uma única passada pela memória de ``values``, o que
        é mais eficiente que chamar ``integrate_j0`` + ``integrate_j1``
        separadamente quando a função é chamada muitas vezes dentro de
        ``vmap`` (Sprint 3.3).
    """
    I0 = jnp.einsum("i,i->", weights_j0, values)
    I1 = jnp.einsum("i,i->", weights_j1, values)
    return I0, I1


__all__ = ["integrate_j0", "integrate_j1", "integrate_j0_j1"]
