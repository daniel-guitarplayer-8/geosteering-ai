# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/postprocess/compensation.py                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Feature F6 Compensação Midpoint       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 2.2)                                   ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x                                                  ║
# ║  Dependências: numpy                                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementação da feature F6 (Compensação Midpoint / Schlumberger    ║
# ║    Compensated Dual Resistivity — CDR) como pós-processamento do        ║
# ║    tensor H de uma simulação multi-TR.                                  ║
# ║                                                                           ║
# ║    Cancela efeitos simétricos de 1ª ordem em torno do ponto médio      ║
# ║    entre dois transmissores `T_near` e `T_far`: rugosidade do poço,   ║
# ║    excentricidade da ferramenta, vibração BHA, etc.                    ║
# ║                                                                           ║
# ║  FÓRMULA (paridade Fortran PerfilaAnisoOmp.f08:804-868)                 ║
# ║    Para cada par `(near_idx, far_idx)` e componente `ic ∈ [0, 8]`:     ║
# ║                                                                           ║
# ║      H_comp[ic]   = 0.5 · (H_near[ic] + H_far[ic])                     ║
# ║      Δφ[°]        = 180/π · (arg(H_near[ic]) − arg(H_far[ic]))         ║
# ║      Δα[dB]       = 20 · log10(|H_near[ic]| / |H_far[ic]|)              ║
# ║                                                                           ║
# ║  PRÉ-REQUISITO                                                            ║
# ║    len(tr_spacings_m) ≥ 2 — a feature F6 é inerentemente multi-TR.    ║
# ║    Validado em SimulationConfig.__post_init__ (Grupo 8).               ║
# ║                                                                           ║
# ║  CUSTO                                                                    ║
# ║    ~162 μs para configuração típica (nTR=3, ntheta=1, nmmax=600, nf=2). ║
# ║    Função puramente NumPy (sem Numba) — não está no hot path.          ║
# ║                                                                           ║
# ║  OPT-IN                                                                   ║
# ║    Ativado por `cfg.use_compensation=True` + `cfg.comp_pairs=tuple`.    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""F6 Compensação Midpoint — pós-processamento de tensores H.

Implementação puro NumPy da Schlumberger Compensated Dual Resistivity
(CDR) como pós-processamento do tensor H forward. Combina pares
`(T_near, T_far)` para cancelar efeitos simétricos de 1ª ordem.

Example:
    Uso com um par de compensação::

        >>> import numpy as np
        >>> from geosteering_ai.simulation.postprocess import apply_compensation
        >>> n_tr, ntheta, nmeds, nf, ncomp = 3, 1, 10, 1, 9
        >>> H = np.random.randn(n_tr, ntheta, nmeds, nf, ncomp).astype(
        ...     np.complex128
        ... )
        >>> H_comp, phase_diff, atten = apply_compensation(H, ((0, 2),))
        >>> H_comp.shape
        (1, 1, 10, 1, 9)

Note:
    Paridade com Fortran PerfilaAnisoOmp.f08 linhas 804-868
    (`compute_compensation_tensor_hanning_transforms`).
"""
from __future__ import annotations

import math

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
# Fator graus: 180/π. Pré-computado para evitar divisões no hot path.
_RAD_TO_DEG: float = 180.0 / math.pi


def apply_compensation(
    H_tensors_per_tr: np.ndarray,
    comp_pairs: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica F6 Compensação Midpoint a N pares T_near/T_far.

    Para cada par `(near_idx, far_idx)` em `comp_pairs`:

        H_comp[ic]   = 0.5 · (H_near[ic] + H_far[ic])
        Δφ[°]        = 180/π · (arg(H_near[ic]) − arg(H_far[ic]))
        Δα[dB]       = 20 · log10(|H_near[ic]| / |H_far[ic]|)

    Args:
        H_tensors_per_tr: Array complex128 shape `(n_tr, ntheta, nmeds,
            nf, 9)` contendo o tensor H de cada TR na dimensão inicial.
        comp_pairs: Tupla de pares `(near_idx, far_idx)` 0-based. Os
            índices devem estar no range `[0, n_tr)` e devem ser
            distintos dentro de cada par. Validado antes de chamar
            esta função.

    Returns:
        Tupla `(H_comp, phase_diff_deg, atten_db)`:

        - ``H_comp``          : `(n_pairs, ntheta, nmeds, nf, 9)` complex128
        - ``phase_diff_deg``  : `(n_pairs, ntheta, nmeds, nf, 9)` float64
        - ``atten_db``        : `(n_pairs, ntheta, nmeds, nf, 9)` float64

    Raises:
        ValueError: Se shape de H_tensors_per_tr for inválido ou se
            algum par contiver índice fora do range.

    Note:
        Esta função NÃO verifica `cfg.use_compensation` — assume que
        o chamador já validou via `SimulationConfig.__post_init__`.
        Se chamada sem pares válidos, levanta ValueError explícito.

        **Convenção de fase**: a diferença `arg(H_near) - arg(H_far)`
        pode ser ambígua em `±π` quando as fases se aproximam do
        branch cut. O np.angle retorna em `(-π, π]` mas a diferença
        não é normalizada — documentação explícita para o chamador.

    Example:
        Caso trivial (H_near == H_far → H_comp == H_near)::

            >>> H = np.ones((2, 1, 5, 1, 9), dtype=np.complex128)
            >>> H_comp, dphi, dalpha = apply_compensation(H, ((0, 1),))
            >>> np.allclose(H_comp, H[0:1])
            True
            >>> np.allclose(dphi, 0.0)
            True
            >>> np.allclose(dalpha, 0.0)
            True
    """
    # ── Validação básica ──────────────────────────────────────────
    H_tensors_per_tr = np.asarray(H_tensors_per_tr, dtype=np.complex128)
    if H_tensors_per_tr.ndim != 5:
        raise ValueError(
            f"H_tensors_per_tr.ndim={H_tensors_per_tr.ndim} inválido. "
            f"Esperado 5: (n_tr, ntheta, nmeds, nf, 9)."
        )
    if H_tensors_per_tr.shape[-1] != 9:
        raise ValueError(
            f"H_tensors_per_tr última dim={H_tensors_per_tr.shape[-1]}, "
            f"esperado 9 componentes (Hxx..Hzz)."
        )
    n_tr, ntheta, nmeds, nf, ncomp = H_tensors_per_tr.shape

    if len(comp_pairs) == 0:
        raise ValueError("comp_pairs vazio — nada para compensar.")

    # Validação dos pares (fail-fast)
    for i, pair in enumerate(comp_pairs):
        if len(pair) != 2:
            raise ValueError(f"comp_pairs[{i}]={pair} deve ter 2 elementos.")
        near, far = pair
        if not (0 <= near < n_tr and 0 <= far < n_tr):
            raise ValueError(f"comp_pairs[{i}]=({near},{far}) fora do range [0, {n_tr}).")
        if near == far:
            raise ValueError(f"comp_pairs[{i}]=({near},{far}): near == far é degenerado.")

    # ── Pré-alocação dos arrays de saída ──────────────────────────
    n_pairs = len(comp_pairs)
    output_shape = (n_pairs, ntheta, nmeds, nf, ncomp)
    H_comp = np.empty(output_shape, dtype=np.complex128)
    phase_diff_deg = np.empty(output_shape, dtype=np.float64)
    atten_db = np.empty(output_shape, dtype=np.float64)

    # ── Loop sobre pares ──────────────────────────────────────────
    for i, (near_idx, far_idx) in enumerate(comp_pairs):
        H_near = H_tensors_per_tr[near_idx]  # (ntheta, nmeds, nf, 9)
        H_far = H_tensors_per_tr[far_idx]

        # Compensação midpoint clássica CDR
        H_comp[i] = 0.5 * (H_near + H_far)

        # Diferença de fase (graus)
        phase_diff_deg[i] = _RAD_TO_DEG * (np.angle(H_near) - np.angle(H_far))

        # Atenuação relativa (dB) — cuidado com divisão por zero
        # em componentes nulas (típico Hxy=0 em meios 1D rotacionados)
        mag_near = np.abs(H_near)
        mag_far = np.abs(H_far)
        # Guard: se |H_far| == 0, o atten fica infinito — usamos NaN para
        # sinalizar "indefinido" em vez de Inf (mais fácil de filtrar).
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mag_far > 0.0, mag_near / mag_far, np.nan)
            atten_db[i] = 20.0 * np.log10(ratio)

    return H_comp, phase_diff_deg, atten_db


__all__ = ["apply_compensation"]
