# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_correlation.py                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Análise de correlação e ensemble     ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Status      : Produção (v2.6)                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Funções e dialogs para análise estatística do tensor H:                ║
# ║      • Matriz de correlação Pearson/Spearman/Kendall entre componentes    ║
# ║      • Envelope P5/P95 do ensemble de modelos                             ║
# ║      • Detecção de outliers via z-score                                   ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    from sm_correlation import compute_correlation_matrix                  ║
# ║    matrix, labels = compute_correlation_matrix(H_stack, method='kendall')║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Análise de correlação e ensemble para v2.6 (P3)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "compute_correlation_matrix",
    "compute_ensemble_envelope",
    "detect_outliers_zscore",
    "EM_COMPONENT_LABELS",
]

EM_COMPONENT_LABELS: List[str] = [
    "Hxx",
    "Hxy",
    "Hxz",
    "Hyx",
    "Hyy",
    "Hyz",
    "Hzx",
    "Hzy",
    "Hzz",
]
GEOSIGNAL_LABELS: List[str] = ["USD", "UAD", "UHR", "UHA", "U3DF"]


def compute_correlation_matrix(
    H_stack: np.ndarray,
    method: str = "pearson",
    component_indices: Optional[List[int]] = None,
    include_geosignals: bool = False,
    use_real_part: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Computa matriz NxN de correlação entre componentes EM ao longo de z_obs.

    Args:
        H_stack: Tensor com shape ``(nTR, nAng, n_pos, nf, 9)`` (sem ensemble)
            ou ``(n_models, nTR, nAng, n_pos, nf, 9)`` (ensemble agrega).
        method: ``"pearson"`` (default), ``"spearman"`` ou ``"kendall"``.
        component_indices: subset de componentes (índices 0..8). None = todos.
        include_geosignals: se True, anexa USD/UAD/UHR/UHA aos componentes EM
            (currently no-op stub — reservado para v2.6.x).
        use_real_part: se True usa Re(H), senão usa |H|. Default True.

    Returns:
        ``(matrix, labels)`` onde:
          - ``matrix`` é (N, N) com valores em [-1, 1]
          - ``labels`` lista de N nomes (e.g., ['Hxx', 'Hzz', ...])

    Raises:
        ValueError: se method não reconhecido ou tensor com shape inválido.

    Note:
        Para ensemble (6D), agrega via mean sobre n_models antes de correlacionar.
        Cada combinação (iTR, iAng, ifq) gera uma série temporal ao longo de
        n_pos; correlações são calculadas e médias entre combinações.
    """
    valid_methods = {"pearson", "spearman", "kendall"}
    if method not in valid_methods:
        raise ValueError(f"method deve ser um de {valid_methods}, recebeu: {method!r}")
    H = np.asarray(H_stack)
    if H.ndim == 6:
        # Ensemble: agrega via média antes de correlacionar
        H = H.mean(axis=0)
    if H.ndim != 5:
        raise ValueError(
            f"H_stack deve ter shape 5D (nTR,nAng,n_pos,nf,9) ou 6D com "
            f"ensemble, recebeu shape {H.shape}"
        )
    nTR, nAng, n_pos, nf, n_comp = H.shape
    if n_comp != 9:
        raise ValueError(f"Última dimensão deve ser 9 (componentes EM), recebeu {n_comp}")

    if component_indices is None:
        component_indices = list(range(9))
    indices = list(component_indices)
    n = len(indices)
    labels = [EM_COMPONENT_LABELS[i] for i in indices]

    # Extrai parte real ou magnitude
    if use_real_part:
        data = H.real
    else:
        data = np.abs(H)

    # Para cada combinação (itr, iang, ifq), calcula matriz NxN, depois média
    accum = np.zeros((n, n), dtype=np.float64)
    counts = 0
    for itr in range(nTR):
        for iang in range(nAng):
            for ifq in range(nf):
                series_matrix = data[itr, iang, :, ifq, :][:, indices]  # (n_pos, n)
                m = _corrmat(series_matrix, method)
                accum += m
                counts += 1
    if counts == 0:
        return np.eye(n), labels
    matrix = accum / counts
    return matrix, labels


def _corrmat(data: np.ndarray, method: str) -> np.ndarray:
    """Helper interno: matriz NxN para um único array (n_obs, n_features)."""
    n_features = data.shape[1]
    if method == "pearson":
        # NumPy nativo é vetorizado e rápido
        if data.shape[0] < 2:
            return np.eye(n_features)
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.corrcoef(data, rowvar=False)
        return np.nan_to_num(mat, nan=0.0)
    # Spearman e Kendall via scipy
    try:
        from scipy.stats import kendalltau, spearmanr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_correlation_matrix(method='spearman'|'kendall') requer scipy"
        ) from exc
    mat = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            xi = data[:, i]
            xj = data[:, j]
            if method == "spearman":
                r, _ = spearmanr(xi, xj)
            else:  # kendall
                r, _ = kendalltau(xi, xj)
            r = float(r) if np.isfinite(r) else 0.0
            mat[i, j] = r
            mat[j, i] = r
    return mat


def compute_ensemble_envelope(
    H_stack: np.ndarray,
    percentiles: Tuple[float, float] = (5.0, 95.0),
) -> Dict[str, np.ndarray]:
    """Mediana + envelope percentil do ensemble.

    Args:
        H_stack: Tensor 6D com shape ``(n_models, nTR, nAng, n_pos, nf, 9)``.
        percentiles: tuple ``(p_low, p_high)`` em porcentos. Default (5, 95).

    Returns:
        Dict com chaves ``"median"``, ``"p_low"``, ``"p_high"``, todas com
        shape ``(nTR, nAng, n_pos, nf, 9)``.

    Raises:
        ValueError: se H_stack não for 6D.
    """
    H = np.asarray(H_stack)
    if H.ndim != 6:
        raise ValueError(
            f"compute_ensemble_envelope requer shape 6D (n_models, ...), "
            f"recebeu {H.shape}"
        )
    p_low, p_high = float(percentiles[0]), float(percentiles[1])
    return {
        "median": np.median(H, axis=0),
        "p_low": np.percentile(H, p_low, axis=0),
        "p_high": np.percentile(H, p_high, axis=0),
    }


def detect_outliers_zscore(
    H_stack: np.ndarray,
    threshold: float = 3.0,
    component_index: int = 8,  # Hzz default
) -> np.ndarray:
    """Detecta modelos outliers via z-score por posição.

    Args:
        H_stack: Tensor 6D ``(n_models, nTR, nAng, n_pos, nf, 9)``.
        threshold: z-score absoluto considerado outlier (default 3.0).
        component_index: componente EM a inspecionar (default 8 = Hzz).

    Returns:
        Array bool de shape ``(n_models,)`` — True para modelos outliers
        (z-score > threshold em mais de 10% das posições).
    """
    H = np.asarray(H_stack)
    if H.ndim != 6:
        raise ValueError(f"Requer shape 6D, recebeu {H.shape}")
    # Extrai componente, agrega TR/ang/freq via mean
    comp = np.abs(H[..., component_index]).mean(axis=(1, 2, 4))  # (n_models, n_pos)
    mu = comp.mean(axis=0)
    sigma = comp.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1e-12)
    z = np.abs(comp - mu[None, :]) / sigma[None, :]
    frac_above = (z > threshold).mean(axis=1)
    return frac_above > 0.1
