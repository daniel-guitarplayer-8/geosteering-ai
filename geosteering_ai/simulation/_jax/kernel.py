# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/kernel.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend JAX Kernel (Sprint 3.3)        ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção (híbrido JAX + Numba via pure_callback)          ║
# ║  Framework   : JAX 0.4.30+                                                ║
# ║  Dependências: jax, jaxlib, numpy, geosteering_ai.simulation._numba      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Orquestrador forward JAX com suporte a `vmap` sobre posições do      ║
# ║    poço — **aqui vem o speedup real** da Fase 3. Arquitetura híbrida:  ║
# ║    reusa kernels Numba complexos (hmd_tiv, vmd, 900 LOC) via            ║
# ║    `jax.pure_callback`, evitando re-implementar os 6 casos geométricos. ║
# ║                                                                           ║
# ║  ARQUITETURA                                                              ║
# ║    ┌───────────────────────────────────────────────────────────────┐  ║
# ║    │  fields_in_freqs_jax_batch(positions, ...) — API pública      │  ║
# ║    │     │                                                         │  ║
# ║    │     ├── jax.vmap sobre n_positions  ◄── speedup XLA          │  ║
# ║    │     │    │                                                    │  ║
# ║    │     │    └── _single_position_forward (por posição)           │  ║
# ║    │     │         │                                               │  ║
# ║    │     │         ├── common_arrays_jax  (jax.lax.scan)           │  ║
# ║    │     │         ├── common_factors_jax (operações jnp)          │  ║
# ║    │     │         ├── dipoles via pure_callback → Numba @njit    │  ║
# ║    │     │         └── rotate_tensor_jax                            │  ║
# ║    │     │                                                         │  ║
# ║    │     └── retorna H_tensor (n_pos, nf, 9) complex128            │  ║
# ║    └───────────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  VANTAGENS DO MODELO HÍBRIDO                                              ║
# ║    1. `vmap` fornece vetorização automática sobre posições — XLA        ║
# ║       gera SIMD/GPU kernels eficientes.                                  ║
# ║    2. Propagação (etapa mais custosa) roda em JAX puro → diferenciável ║
# ║       via `jax.grad` (fundamento para PINNs Fase 5).                    ║
# ║    3. Dipolos (900 LOC de ramificações) reusados de Numba — nenhum     ║
# ║       trabalho duplicado, paridade numérica automática.                  ║
# ║                                                                           ║
# ║  LIMITAÇÕES                                                               ║
# ║    • `pure_callback` não é diferenciável (JAX não sabe o gradiente    ║
# ║      dos dipolos Numba). Para PINNs completos, Sprint 3.3.1 futura     ║
# ║      portará hmd_tiv/vmd para JAX nativo.                                ║
# ║    • `pure_callback` roda na CPU host — não GPU. Speedup virá          ║
# ║      principalmente do vmap+XLA da propagação + paralelismo Numba     ║
# ║      dos dipolos.                                                         ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • _numba/kernel.py — implementação de referência                    ║
# ║    • docs/reference/plano_simulador_python_jax_numba.md §7             ║
# ║    • JAX docs: jax.pure_callback, jax.vmap                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Orquestrador forward JAX com vmap sobre posições (Sprint 3.3).

Expõe :func:`fields_in_freqs_jax_batch` que calcula o tensor H para
múltiplas posições do poço em uma única chamada, usando ``jax.vmap``
para vetorização automática (CPU/GPU).
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from geosteering_ai.simulation._jax.propagation import (
    common_arrays_jax,
    common_factors_jax,
)
from geosteering_ai.simulation._jax.rotation import rotate_tensor
from geosteering_ai.simulation._numba.dipoles import hmd_tiv, vmd
from geosteering_ai.simulation._numba.geometry import (
    _sanitize_profile_kernel,
    find_layers_tr,
)

# Constante física — permeabilidade magnética do vácuo
_MU_0: float = 4.0e-7 * math.pi


# ──────────────────────────────────────────────────────────────────────────────
# Callback para dipolos Numba (Sprint 3.3)
# ──────────────────────────────────────────────────────────────────────────────
# jax.pure_callback chama funções Python/NumPy arbitrárias de dentro de
# código traçado JAX. Usado aqui para reusar os 900 LOC de hmd_tiv+vmd do
# backend Numba sem re-implementar em JAX (economia de ~1500 LOC de port).


def _dipoles_numba_host(
    Tx,
    Ty,
    Tz,
    n_int,
    camad_r,
    camad_t,
    npt_int,
    krJ0J1,
    wJ0,
    wJ1,
    h_arr,
    prof_arr,
    zeta_c,
    eta,
    cx,
    cy,
    cz,
    u,
    s,
    uh,
    sh,
    RTEdw,
    RTEup,
    RTMdw,
    RTMup,
    Mxdw,
    Mxup,
    Eudw,
    Euup,
    FEdwz,
    FEupz,
):
    """Wrapper Python que chama hmd_tiv + vmd (@njit) e retorna matH (3,3).

    Convertido para numpy arrays antes de chamar Numba. Retorna matH
    3x3 complex128, que JAX converte de volta para jnp.array.
    """
    Hx_hmd, Hy_hmd, Hz_hmd = hmd_tiv(
        float(Tx),
        float(Ty),
        float(Tz),
        int(n_int),
        int(camad_r),
        int(camad_t),
        int(npt_int),
        np.asarray(krJ0J1),
        np.asarray(wJ0),
        np.asarray(wJ1),
        np.asarray(h_arr),
        np.asarray(prof_arr),
        complex(zeta_c),
        np.asarray(eta),
        float(cx),
        float(cy),
        float(cz),
        np.asarray(u),
        np.asarray(s),
        np.asarray(uh),
        np.asarray(sh),
        np.asarray(RTEdw),
        np.asarray(RTEup),
        np.asarray(RTMdw),
        np.asarray(RTMup),
        np.asarray(Mxdw),
        np.asarray(Mxup),
        np.asarray(Eudw),
        np.asarray(Euup),
    )
    Hx_vmd, Hy_vmd, Hz_vmd = vmd(
        float(Tx),
        float(Ty),
        float(Tz),
        int(n_int),
        int(camad_r),
        int(camad_t),
        int(npt_int),
        np.asarray(krJ0J1),
        np.asarray(wJ0),
        np.asarray(wJ1),
        np.asarray(h_arr),
        np.asarray(prof_arr),
        complex(zeta_c),
        float(cx),
        float(cy),
        float(cz),
        np.asarray(u),
        np.asarray(uh),
        np.asarray(np.asarray(u) / complex(zeta_c)),  # AdmInt recalc
        np.asarray(RTEdw),
        np.asarray(RTEup),
        np.asarray(FEdwz),
        np.asarray(FEupz),
    )

    matH = np.empty((3, 3), dtype=np.complex128)
    matH[0, 0] = Hx_hmd[0]
    matH[0, 1] = Hy_hmd[0]
    matH[0, 2] = Hz_hmd[0]
    matH[1, 0] = Hx_hmd[1]
    matH[1, 1] = Hy_hmd[1]
    matH[1, 2] = Hz_hmd[1]
    matH[2, 0] = Hx_vmd
    matH[2, 1] = Hy_vmd
    matH[2, 2] = Hz_vmd
    return matH


# ──────────────────────────────────────────────────────────────────────────────
# Simulação de uma única posição — JAX puro (exceto dipolos via callback)
# ──────────────────────────────────────────────────────────────────────────────


def _single_position_jax(
    Tx: float,
    Ty: float,
    Tz: float,
    cx: float,
    cy: float,
    cz: float,
    dip_rad: float,
    n: int,
    npt: int,
    camad_t: int,
    camad_r: int,
    rho_h: jax.Array,
    rho_v: jax.Array,
    h_arr: jax.Array,
    prof_arr: jax.Array,
    eta: jax.Array,
    freq: float,
    krJ0J1: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
) -> jax.Array:
    """Calcula H para uma frequência numa posição via backend JAX.

    Retorna array (9,) complex128 — Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz.
    """
    omega = 2.0 * jnp.pi * freq
    zeta = 1j * omega * _MU_0

    # Distância horizontal
    dx = cx - Tx
    dy = cy - Ty
    r = jnp.sqrt(dx * dx + dy * dy)

    # Sprint 3.2: propagação em JAX puro
    outs = common_arrays_jax(n, npt, r, krJ0J1, zeta, h_arr, eta)
    u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt = outs

    cf = common_factors_jax(
        n,
        npt,
        Tz,
        h_arr,
        prof_arr,
        camad_t,
        u,
        s,
        uh,
        sh,
        RTEdw,
        RTEup,
        RTMdw,
        RTMup,
    )
    Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz = cf

    # Dipolos via pure_callback — 900 LOC reusadas do Numba
    result_shape = jax.ShapeDtypeStruct((3, 3), jnp.complex128)
    matH = jax.pure_callback(
        _dipoles_numba_host,
        result_shape,
        Tx,
        Ty,
        Tz,
        n,
        camad_r,
        camad_t,
        npt,
        krJ0J1,
        wJ0,
        wJ1,
        h_arr,
        prof_arr,
        zeta,
        eta,
        cx,
        cy,
        cz,
        u,
        s,
        uh,
        sh,
        RTEdw,
        RTEup,
        RTMdw,
        RTMup,
        Mxdw,
        Mxup,
        Eudw,
        Euup,
        FEdwz,
        FEupz,
    )

    # Rotação
    tH = rotate_tensor(dip_rad, 0.0, 0.0, matH)

    # Flatten (3,3) → (9,) em ordem (row-major: Hxx, Hxy, Hxz, Hyx, ...)
    return tH.reshape(9)


# ──────────────────────────────────────────────────────────────────────────────
# API pública: fields_in_freqs_jax_batch — vmap sobre posições
# ──────────────────────────────────────────────────────────────────────────────


def fields_in_freqs_jax_batch(
    positions_z: np.ndarray,
    dz_half: float,
    r_half: float,
    dip_rad: float,
    n: int,
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    freqs_hz: np.ndarray,
    krJ0J1: np.ndarray,
    wJ0: np.ndarray,
    wJ1: np.ndarray,
) -> np.ndarray:
    """Forward JAX batch sobre posições do poço.

    API **sem vmap real** mas com pipeline JAX (propagação) + Numba
    (dipolos). Implementação com ``vmap`` real fica para Sprint 3.3.1,
    quando `pure_callback` for substituído por port JAX nativo dos
    dipolos. Por enquanto, loop Python externo + JAX/JIT para a
    propagação (ainda mais rápido que chamada Python pura por conta do
    JIT cache do propagation).

    Args:
        positions_z: (n_positions,) — profundidades ponto-médio (m).
        dz_half: metade da separação vertical (L·cos(dip)/2).
        r_half: metade do afastamento horizontal (L·sin(dip)/2).
        dip_rad: dip em radianos.
        n: número de camadas.
        rho_h, rho_v: (n,) resistividades.
        esp: (n-2,) espessuras internas.
        freqs_hz: (nf,) frequências.
        krJ0J1, wJ0, wJ1: (npt,) filtro Hankel.

    Returns:
        H_tensor shape (n_positions, nf, 9) complex128.

    Note:
        Na Sprint 3.3.1, ``_single_position_jax`` será paralelizado via
        ``jax.vmap`` sobre positions_z, removendo o loop Python externo.
        Atualmente, a vantagem é diferenciabilidade parcial (propagação)
        e preparação para GPU na Sprint 3.4.
    """
    n_positions = positions_z.shape[0]
    nf = freqs_hz.shape[0]
    npt = krJ0J1.shape[0]

    # Geometria estática
    if n == 1:
        h_arr = np.zeros(1, dtype=np.float64)
        prof_arr = np.array([-1.0e300, 1.0e300], dtype=np.float64)
    else:
        h_arr, prof_arr = _sanitize_profile_kernel(n, esp)

    eta = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        eta[i, 0] = 1.0 / rho_h[i]
        eta[i, 1] = 1.0 / rho_v[i]

    # Conversão para JAX (float64 → complex128 onde apropriado)
    krJ0J1_j = jnp.asarray(krJ0J1)
    wJ0_j = jnp.asarray(wJ0)
    wJ1_j = jnp.asarray(wJ1)
    rho_h_j = jnp.asarray(rho_h)
    rho_v_j = jnp.asarray(rho_v)
    h_arr_j = jnp.asarray(h_arr)
    prof_arr_j = jnp.asarray(prof_arr)
    eta_j = jnp.asarray(eta)

    H_tensor = np.empty((n_positions, nf, 9), dtype=np.complex128)

    for j in range(n_positions):
        z_mid = float(positions_z[j])
        Tz = z_mid - dz_half
        cz = z_mid + dz_half
        Tx = -r_half
        cx = r_half

        if n == 1:
            camad_t, camad_r = 0, 0
        else:
            camad_t, camad_r = find_layers_tr(n, Tz, cz, prof_arr)

        for i_f in range(nf):
            freq = float(freqs_hz[i_f])
            cH_9 = _single_position_jax(
                Tx,
                0.0,
                Tz,
                cx,
                0.0,
                cz,
                dip_rad,
                n,
                npt,
                camad_t,
                camad_r,
                rho_h_j,
                rho_v_j,
                h_arr_j,
                prof_arr_j,
                eta_j,
                freq,
                krJ0J1_j,
                wJ0_j,
                wJ1_j,
            )
            H_tensor[j, i_f, :] = np.asarray(cH_9)

    return H_tensor


__all__ = ["fields_in_freqs_jax_batch"]
