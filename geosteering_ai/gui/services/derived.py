# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/services/derived.py                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Grandezas derivadas — geosinais + perfis ρ/λ (PURO)       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — services (spec 0017, Fatia 6d)                      ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — plots completos                                 ║
# ║  Framework   : numpy PURO — NÃO importa Qt (Princípio X; testável)         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Funções PURAS para os plots científicos da Fatia 6d:                    ║
# ║    (1) GEOSINAIS — derivados do tensor H6 (9 componentes), BYTE-FIÉIS ao    ║
# ║        monólito (sm_plots.py::_compute_geosignal:1406-1446). Índices do      ║
# ║        eixo −1: 0=Hxx 1=Hxy 2=Hxz 3=Hyx 4=Hyy 5=Hyz 6=Hzx 7=Hzy 8=Hzz       ║
# ║        (idêntico a COMPONENT_NAMES da galeria).                             ║
# ║    (2) PERFIS ρ/λ — step-function da geologia (ρ por camada + espessuras)   ║
# ║        vs profundidade (convenção Fortran: interfaces em cumsum(esp)).      ║
# ║                                                                           ║
# ║  FIDELIDADE                                                               ║
# ║    Geosinal é função determinística do H6 — se a paridade do H6 é <1e-12   ║
# ║    (Fortran/JAX/Numba), o geosinal preserva a paridade. ε=1e-20 nas razões. ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    GEOSIGNALS · compute_geosignal · rho_profile · lambda_profile          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Grandezas derivadas PURAS — geosinais (byte-fiéis) + perfis ρ/λ (spec 0017)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["GEOSIGNALS", "compute_geosignal", "rho_profile", "lambda_profile"]

# Geosinais suportados (ordem de exibição). Derivados das 9 componentes do H6.
GEOSIGNALS: Tuple[str, ...] = ("USD", "UAD", "UHR", "UHA", "U3DF")

# Guard de divisão por zero (idêntico ao monólito; efeito < 1e-20 ≪ 1e-12).
_EPS = 1e-20

# Índices canônicos do eixo −1 do H6 (== COMPONENT_NAMES).
_HXX, _HXY, _HXZ, _HYX, _HYY, _HYZ, _HZX, _HZY, _HZZ = range(9)


def compute_geosignal(name: str, h9: np.ndarray) -> np.ndarray:
    """Calcula um geosinal complexo a partir das 9 componentes do tensor (BYTE-FIEL).

    Fórmulas (monólito ``sm_plots.py:1434-1445``):
      ┌──────┬───────────────────────────────────────┐
      │ USD  │ Hxx / Hyy            (razão planar TIV) │
      │ UAD  │ Hxx − Hyy            (diferença planar) │
      │ UHR  │ Hxz / Hzz       (razão axial/planar)    │
      │ UHA  │ Hxz − Hzz       (diferença axial)       │
      │ U3DF │ (Hxy−Hyx)/(Hxy+Hyx+ε)  (fator 3D)       │
      └──────┴───────────────────────────────────────┘

    Args:
        name: um de :data:`GEOSIGNALS`.
        h9: array complexo com o eixo −1 = 9 componentes (Hxx..Hzz). Demais eixos
            (ex.: profundidade) são preservados — operação vetorizada.

    Returns:
        Array complexo com shape ``h9.shape[:-1]`` (o geosinal por posição).

    Raises:
        ValueError: se ``name`` não está em :data:`GEOSIGNALS` ou ``h9`` não tem 9
            componentes no eixo −1.
    """
    if h9.shape[-1] != 9:
        raise ValueError(f"h9 deve ter 9 componentes no eixo -1 (got {h9.shape[-1]}).")
    hxx = h9[..., _HXX]
    hxy = h9[..., _HXY]
    hxz = h9[..., _HXZ]
    hyx = h9[..., _HYX]
    hyy = h9[..., _HYY]
    hzz = h9[..., _HZZ]
    if name == "USD":
        return hxx / (hyy + _EPS)  # type: ignore[no-any-return]  # numpy → Any
    if name == "UAD":
        return hxx - hyy  # type: ignore[no-any-return]
    if name == "UHR":
        return hxz / (hzz + _EPS)  # type: ignore[no-any-return]
    if name == "UHA":
        return hxz - hzz  # type: ignore[no-any-return]
    if name == "U3DF":
        return (hxy - hyx) / (hxy + hyx + _EPS)  # type: ignore[no-any-return]
    raise ValueError(f"geosinal {name!r} desconhecido (use {list(GEOSIGNALS)}).")


def _layer_index_at(
    z: np.ndarray, thicknesses: np.ndarray, n_layers: int
) -> np.ndarray:
    """Índice da camada (0..n_layers−1) em cada profundidade ``z`` (convenção Fortran).

    Interfaces internas em ``boundaries = [0, cumsum(esp)]`` (n_layers−1 valores);
    ``searchsorted(..., side="right")`` mapeia z → índice da camada; clip a [0, n−1].
    """
    cum = np.cumsum(np.asarray(thicknesses, dtype=float))
    boundaries = np.concatenate((np.zeros(1, dtype=float), cum))
    idx = np.searchsorted(boundaries, np.asarray(z, dtype=float), side="right")
    return np.clip(idx, 0, n_layers - 1)


def rho_profile(
    z: np.ndarray, rho_per_layer: np.ndarray, thicknesses: np.ndarray
) -> np.ndarray:
    """Perfil ρ(z) step a partir de ρ por-camada + espessuras internas (convenção Fortran).

    Args:
        z: profundidades de medição ``positions_z`` (n_pos,).
        rho_per_layer: ρ por camada (n_layers,).
        thicknesses: espessuras internas (n_layers−2,).

    Returns:
        ρ avaliado em cada ``z`` (step-function), shape ``z.shape``.
    """
    rho = np.asarray(rho_per_layer, float)
    idx = _layer_index_at(z, thicknesses, len(rho))
    return rho[idx]  # type: ignore[no-any-return]  # numpy fancy-index → Any


def lambda_profile(
    z: np.ndarray,
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    thicknesses: np.ndarray,
) -> np.ndarray:
    """Perfil de anisotropia ``λ(z) = √(ρᵥ/ρₕ)`` step (≥1, TIV — monólito ``:1734``).

    Args:
        z: profundidades ``positions_z`` (n_pos,).
        rho_h/rho_v: ρₕ/ρᵥ por camada (n_layers,).
        thicknesses: espessuras internas (n_layers−2,).

    Returns:
        λ avaliado em cada ``z`` (step), shape ``z.shape``; clip [1, ∞) (TIV física).
    """
    rh = np.clip(np.asarray(rho_h, float), 1e-12, None)
    rv = np.asarray(rho_v, float)
    lam = np.sqrt(np.clip(rv / rh, 1.0, None))  # λ = √(ρᵥ/ρₕ), λ≥1
    idx = _layer_index_at(z, thicknesses, len(lam))
    return lam[idx]  # type: ignore[no-any-return]  # numpy fancy-index → Any
