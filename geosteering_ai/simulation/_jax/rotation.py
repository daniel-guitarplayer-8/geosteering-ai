# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/rotation.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend JAX Rotação (Sprint 3.1)       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : JAX 0.4.30+                                                ║
# ║  Dependências: jax, jaxlib                                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Rotação de tensor magnético H (3×3) via ângulos de Euler (α, β, γ)  ║
# ║    usando JAX. Port funcional diferenciável do `_numba/rotation.py`,    ║
# ║    compatível com `jax.grad` e `jax.jacfwd` — fundação para autodiff   ║
# ║    de PINNs na Sprint 3.3+.                                              ║
# ║                                                                           ║
# ║  FÓRMULA                                                                  ║
# ║    H' = Rᵀ · H · R                                                       ║
# ║                                                                           ║
# ║    onde R(α, β, γ) é a matriz de rotação de Euler construída por        ║
# ║    :func:`build_rotation_matrix`. Convenção idêntica ao Fortran         ║
# ║    `RtHR` (utils.f08:321-355, Liu 2017 eq. 4.80).                        ║
# ║                                                                           ║
# ║  DIFERENCIABILIDADE                                                       ║
# ║    Todas as operações usam `jnp.sin`, `jnp.cos`, `@` (matmul), que      ║
# ║    são primitivas JAX diferenciáveis. Logo:                              ║
# ║                                                                           ║
# ║        >>> grad_fn = jax.grad(lambda a: rotate_tensor(a, 0, 0, H)[0, 0]) ║
# ║                                                                           ║
# ║    computa ∂H'_{0,0} / ∂α via autodiff exato.                           ║
# ║                                                                           ║
# ║  PARIDADE COM NUMBA                                                       ║
# ║    Tolerância esperada: < 1e-14 (limite ULP float64). O uso de           ║
# ║    `@` (matmul) pelo JAX pode gerar ordem de somas ligeiramente         ║
# ║    diferente do loop explícito do Fortran, mas dentro do erro de        ║
# ║    máquina.                                                               ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • _numba/rotation.py — implementação de referência                   ║
# ║    • Fortran utils.f08:321-355 — `RtHR`                                  ║
# ║    • Liu, Y. (2017) — "Electromagnetic logging..." eq. 4.80             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Rotação de tensor via JAX (Sprint 3.1).

Construção da matriz R(α, β, γ) e aplicação H' = Rᵀ·H·R usando
primitivas JAX (diferenciáveis via ``jax.grad``). Paridade numérica
< 1e-14 com :mod:`geosteering_ai.simulation._numba.rotation`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Garantia de float64 (ver nota no _jax/__init__.py)
jax.config.update("jax_enable_x64", True)


# ──────────────────────────────────────────────────────────────────────────────
# build_rotation_matrix — matriz R(α, β, γ)
# ──────────────────────────────────────────────────────────────────────────────


@jax.jit
def build_rotation_matrix(
    alpha: float,
    beta: float,
    gamma: float,
) -> jax.Array:
    """Constrói matriz de rotação 3×3 a partir de ângulos de Euler.

    Port JAX das linhas 141-155 de `_numba/rotation.py` (equivalente a
    Fortran ``utils.f08:349-351``). Usa ``jnp.stack`` para montar a
    matriz linha-a-linha, permitindo traçado via ``jax.jit`` sem
    problemas de alocação inplace.

    Args:
        alpha: Ângulo α em radianos (dip da ferramenta).
            0 = vertical, π/2 = horizontal.
        beta: Ângulo β em radianos (azimute no plano xy).
        gamma: Ângulo γ em radianos (rotação em torno do eixo da
            ferramenta).

    Returns:
        Matriz R shape ``(3, 3)`` float64 com a rotação.

    Note:
        Diferenciável: ``jax.grad(lambda a: build_rotation_matrix(a, 0, 0)[0, 0])``
        retorna ∂R₀₀/∂α = -sin(α)·cos(β)·cos(γ) - cos(β)·... (correto por
        autodiff simbólico do XLA).

    Example:
        Identidade (α=β=γ=0)::

            >>> import jax.numpy as jnp
            >>> R = build_rotation_matrix(0.0, 0.0, 0.0)
            >>> jnp.allclose(R, jnp.eye(3))
            Array(True, dtype=bool)
    """
    sena = jnp.sin(alpha)
    cosa = jnp.cos(alpha)
    senb = jnp.sin(beta)
    cosb = jnp.cos(beta)
    seng = jnp.sin(gamma)
    cosg = jnp.cos(gamma)

    # Construção linha-a-linha via jnp.stack (mais traço-friendly que
    # alocação inplace tipo `R[i, j] = ...`).
    row0 = jnp.stack(
        [
            cosa * cosb * cosg - senb * seng,
            -cosa * cosb * seng - senb * cosg,
            sena * cosb,
        ]
    )
    row1 = jnp.stack(
        [
            cosa * senb * cosg + cosb * seng,
            -cosa * senb * seng + cosb * cosg,
            sena * senb,
        ]
    )
    row2 = jnp.stack(
        [
            -sena * cosg,
            sena * seng,
            cosa,
        ]
    )

    return jnp.stack([row0, row1, row2])


# ──────────────────────────────────────────────────────────────────────────────
# rotate_tensor — H' = Rᵀ · H · R
# ──────────────────────────────────────────────────────────────────────────────


@jax.jit
def rotate_tensor(
    alpha: float,
    beta: float,
    gamma: float,
    H: jax.Array,
) -> jax.Array:
    """Rotaciona tensor 3×3 por ângulos de Euler: H' = Rᵀ · H · R.

    Port JAX de :func:`geosteering_ai.simulation._numba.rotation.rotate_tensor`.

    Args:
        alpha: Ângulo α (dip) em radianos.
        beta: Ângulo β (azimute) em radianos.
        gamma: Ângulo γ em radianos.
        H: Tensor complex128 shape ``(3, 3)``.

    Returns:
        Tensor rotacionado H' shape ``(3, 3)`` complex128.

    Note:
        Caso típico no orquestrador: ``rotate_tensor(dip, 0, 0, H)``
        (β = γ = 0), equivalente ao Fortran ``RtHR(ang, 0, 0, matH)``
        em `PerfilaAnisoOmp.f08:989`.

    Example:
        Rotação identidade preserva tensor::

            >>> import jax.numpy as jnp
            >>> H = jnp.array(
            ...     [[1, 2j, 3], [4, 5, 6j], [7j, 8, 9]],
            ...     dtype=jnp.complex128,
            ... )
            >>> H_rot = rotate_tensor(0.0, 0.0, 0.0, H)
            >>> jnp.allclose(H_rot, H)
            Array(True, dtype=bool)
    """
    R = build_rotation_matrix(alpha, beta, gamma)
    # Rᵀ · H · R — R é real (float64), H é complex128. XLA gera matmul
    # complex automaticamente pela promoção de tipos.
    R_c = R.astype(H.dtype)
    return R_c.T @ H @ R_c


__all__ = ["build_rotation_matrix", "rotate_tensor"]
