# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_numba/rotation.py                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Rotação de Tensor (Sprint 2.3)          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x + Numba 0.60+ (dual-mode, opcional)              ║
# ║  Dependências: numpy (obrigatório), numba (opcional, speedup JIT)         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Porta para Python+Numba a rotação de tensor magnético usada pelo      ║
# ║    simulador Fortran para transformar o tensor H 3×3 do sistema do      ║
# ║    poço (x, y, z) para o sistema da ferramenta triaxial com orientação  ║
# ║    arbitrária (α, β, γ).                                                 ║
# ║                                                                           ║
# ║    Equivalente Fortran: `RtHR` em `utils.f08:321-355`.                  ║
# ║                                                                           ║
# ║  REFERÊNCIA BIBLIOGRÁFICA                                                 ║
# ║    Liu, Q-H. (2017). "Theory of Electromagnetic Well Logging".          ║
# ║    Página 148, equação 4.80 e figura 4.17.                             ║
# ║                                                                           ║
# ║  FÓRMULA MATEMÁTICA                                                       ║
# ║    Dado H(3,3) no sistema do poço e ângulos de Euler (α, β, γ):         ║
# ║                                                                           ║
# ║      R = matriz de rotação 3×3 que leva (x,y,z) → (x',y',z')            ║
# ║      Rt = transposta de R                                               ║
# ║      H' = Rᵀ · H · R                                                    ║
# ║                                                                           ║
# ║    Os três ângulos são:                                                 ║
# ║      α: ângulo que o segmento (observação → origem) faz com OZ         ║
# ║         (dip da ferramenta, 0°=vertical, 90°=horizontal)               ║
# ║      β: ângulo que a projeção no plano XOY faz com OX (azimute)        ║
# ║      γ: ângulo de rotação em torno do eixo da ferramenta               ║
# ║                                                                           ║
# ║    Matriz R (fortran utils.f08:349-351):                                ║
# ║      R[0,:] = [ cα·cβ·cγ − sβ·sγ,  −cα·cβ·sγ − sβ·cγ,  sα·cβ ]         ║
# ║      R[1,:] = [ cα·sβ·cγ + cβ·sγ,  −cα·sβ·sγ + cβ·cγ,  sα·sβ ]         ║
# ║      R[2,:] = [ −sα·cγ,             sα·sγ,              cα    ]         ║
# ║    Onde c = cos, s = sin.                                               ║
# ║                                                                           ║
# ║  CASOS ESPECIAIS                                                          ║
# ║    • α=0, β=0, γ=0 → R=I → H' = H (identidade)                         ║
# ║    • α=0, β=0, γ=γ → rotação simples em torno de OZ                    ║
# ║    • α=π/2, β=0, γ=0 → ferramenta horizontal apontando em x            ║
# ║    • No uso típico do simulador Fortran (PerfilaAnisoOmp.f08:989):      ║
# ║      `RtHR(ang, 0.d0, 0.d0, matH)` — só dip, sem azimute nem           ║
# ║      rotação axial.                                                     ║
# ║                                                                           ║
# ║  CONVENÇÕES                                                              ║
# ║    • Ângulos em **radianos** (API Python). A conversão de graus         ║
# ║      deve ser feita pelo chamador via `np.deg2rad()`.                   ║
# ║    • Entrada `H` e saída são `complex128(3,3)`.                        ║
# ║    • Row-major NumPy (C-contiguous). Fortran é column-major mas         ║
# ║      a matemática da rotação é idêntica em ambos.                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Rotação de tensor magnético via ângulos de Euler (Sprint 2.3).

Módulo com duas funções:

- :func:`rotate_tensor` — calcula `H' = Rᵀ · H · R` com `R` construída
  a partir dos ângulos de Euler (α, β, γ).
- :func:`build_rotation_matrix` — retorna só a matriz `R` (útil para
  debug e testes).

Example:
    Rotação nula (identidade)::

        >>> import numpy as np
        >>> from geosteering_ai.simulation._numba.rotation import rotate_tensor
        >>> H = np.eye(3, dtype=np.complex128)
        >>> H_rot = rotate_tensor(0.0, 0.0, 0.0, H)
        >>> np.allclose(H_rot, H)
        True

    Rotação de 90° em torno de OZ (γ = π/2)::

        >>> H = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.complex128)
        >>> H_rot = rotate_tensor(0.0, 0.0, np.pi / 2, H)
        >>> # Após rotação, Hxx e Hyy trocam de lugar
        >>> np.allclose(H_rot[0, 0], 2.0)  # dentro de tolerância
        True

Note:
    Paridade bit-exata com Fortran `RtHR` (utils.f08:321-355) não é
    possível por ordem de operações diferentes em `matmul(matmul(Rt,H),R)`.
    Tolerância real esperada: < 1e-14 (ULP float64).
"""
from __future__ import annotations

import numpy as np

from geosteering_ai.simulation._numba.propagation import njit


# ──────────────────────────────────────────────────────────────────────────────
# BUILD_ROTATION_MATRIX — só a matriz R (útil em testes)
# ──────────────────────────────────────────────────────────────────────────────
@njit
def build_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Constrói a matriz de rotação 3×3 a partir de ângulos de Euler.

    Port Python+Numba das linhas 349-351 de `RtHR` no Fortran
    (`utils.f08`). Retorna apenas a matriz `R` sem aplicar ao tensor.

    Args:
        alpha: Ângulo α em radianos. Segmento (observação → origem)
            com OZ. Também chamado "dip" — 0°=vertical, π/2=horizontal.
        beta: Ângulo β em radianos. Projeção no plano XOY com OX
            (azimute). 0=eixo x, π/2=eixo y.
        gamma: Ângulo γ em radianos. Rotação em torno do eixo da
            ferramenta.

    Returns:
        Array `(3, 3)` float64 com a matriz de rotação `R`.

    Note:
        Observa-se que no caso típico do simulador Fortran
        (`RtHR(ang, 0, 0, H)`), apenas α é não-nulo — β=γ=0. Isso
        reduz a matriz para uma rotação simples em torno do eixo
        y efetivo (apenas dip).

    Example:
        Identidade (α=β=γ=0)::

            >>> R = build_rotation_matrix(0.0, 0.0, 0.0)
            >>> np.allclose(R, np.eye(3))
            True
    """
    sena = np.sin(alpha)
    cosa = np.cos(alpha)
    senb = np.sin(beta)
    cosb = np.cos(beta)
    seng = np.sin(gamma)
    cosg = np.cos(gamma)

    R = np.empty((3, 3), dtype=np.float64)

    # Fortran utils.f08:349
    R[0, 0] = cosa * cosb * cosg - senb * seng
    R[0, 1] = -cosa * cosb * seng - senb * cosg
    R[0, 2] = sena * cosb

    # Fortran utils.f08:350
    R[1, 0] = cosa * senb * cosg + cosb * seng
    R[1, 1] = -cosa * senb * seng + cosb * cosg
    R[1, 2] = sena * senb

    # Fortran utils.f08:351
    R[2, 0] = -sena * cosg
    R[2, 1] = sena * seng
    R[2, 2] = cosa

    return R


# ──────────────────────────────────────────────────────────────────────────────
# ROTATE_TENSOR — H' = Rᵀ · H · R
# ──────────────────────────────────────────────────────────────────────────────
@njit
def rotate_tensor(
    alpha: float,
    beta: float,
    gamma: float,
    H: np.ndarray,
) -> np.ndarray:
    """Rotaciona um tensor magnético 3×3 por ângulos de Euler (α, β, γ).

    Port Python+Numba de `RtHR` (Fortran `utils.f08:321-355`). Calcula:

        H' = Rᵀ · H · R

    onde `R = R(α, β, γ)` é a matriz de rotação construída por
    :func:`build_rotation_matrix`.

    Args:
        alpha: Ângulo α em radianos (dip da ferramenta).
        beta: Ângulo β em radianos (azimute no plano xy).
        gamma: Ângulo γ em radianos (rotação em torno do eixo da ferramenta).
        H: Tensor magnético complex128 shape `(3, 3)`.
            Cada linha é o campo gerado por um dipolo; cada coluna é
            uma componente no sistema do poço:
                H[0,:] = [Hxx, Hxy, Hxz]  ← dipolo x
                H[1,:] = [Hyx, Hyy, Hyz]  ← dipolo y
                H[2,:] = [Hzx, Hzy, Hzz]  ← dipolo z

    Returns:
        Array `(3, 3)` complex128 com o tensor rotacionado no sistema
        da ferramenta triaxial.

    Note:
        **Caso de uso típico no orquestrador**: `ang` é o dip do poço
        (graus convertidos para rad), `beta = gamma = 0`. Isso é o que
        o Fortran faz em `PerfilaAnisoOmp.f08:989`:

            tH = RtHR(ang, 0.d0, 0.d0, matH)

    Example:
        Identidade (α=β=γ=0)::

            >>> import numpy as np
            >>> H = np.array([[1, 2j, 3],
            ...               [4, 5, 6j],
            ...               [7j, 8, 9]], dtype=np.complex128)
            >>> H_rot = rotate_tensor(0.0, 0.0, 0.0, H)
            >>> np.allclose(H_rot, H)
            True
    """
    # Constrói R
    R = build_rotation_matrix(alpha, beta, gamma)
    # Transposta: para rotação pura (sem reflexão), Rᵀ = R⁻¹
    Rt = R.T

    # Produto Rᵀ · H · R
    # Numba: usar @ ou np.dot — ambos suportados
    tmp = np.dot(Rt.astype(np.complex128), H)
    H_rot = np.dot(tmp, R.astype(np.complex128))
    return H_rot


__all__ = [
    "build_rotation_matrix",
    "rotate_tensor",
]
