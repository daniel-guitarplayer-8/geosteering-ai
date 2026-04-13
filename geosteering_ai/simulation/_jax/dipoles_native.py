# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/dipoles_native.py                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — JAX Dipolos Nativo (Sprint 3.3.1)      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Em construção (parcial — caso camadR==camadT)             ║
# ║  Framework   : JAX 0.4.30+                                                ║
# ║  Dependências: jax, jaxlib                                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Port nativo JAX dos kernels de dipolos magnéticos (hmd_TIV + vmd),  ║
# ║    substituindo gradualmente `jax.pure_callback` do kernel híbrido     ║
# ║    (Sprint 3.3). A meta é desbloquear:                                   ║
# ║      • GPU: pure_callback roda na CPU host → bloqueia aceleração      ║
# ║      • Autodiff: pure_callback não é diferenciável                      ║
# ║                                                                           ║
# ║  ESTRATÉGIA DE MIGRAÇÃO                                                   ║
# ║    Sprint 3.3.1 (ESTE): helpers nativos + caso mais comum (camadR==   ║
# ║                          camadT, mesma camada).                          ║
# ║    Sprint 3.3.2 (futuro): 5 casos restantes via `lax.switch`.           ║
# ║    Sprint 3.3.3 (futuro): `vmd_native` análogo.                         ║
# ║                                                                           ║
# ║    Durante a migração, o kernel híbrido (`_jax/kernel.py`) continua   ║
# ║    sendo a API preferida — o usuário não perde funcionalidade. Este    ║
# ║    módulo é experimental e expõe `use_native=False` default.           ║
# ║                                                                           ║
# ║  COMPLETUDE                                                               ║
# ║    • decoupling_factors_jax(L) → (ACp, ACx)          ✅                 ║
# ║    • _dipole_phases_jax(...)        fatores exp       ✅                 ║
# ║    • _hmd_tiv_kernel_case_{1..6}_jax kernels Ktm/Kte/Ktedz ✅ 3.3.2     ║
# ║    • _hmd_tiv_full_jax(...)        6 casos via lax.switch ✅ 3.3.2     ║
# ║    • _vmd_native_jax(...)          VMD nativo           ⏳ Sprint 3.3.3 ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • _numba/dipoles.py — implementação Numba de referência              ║
# ║    • Fortran magneticdipoles.f08 — hmd_TIV_optimized + vmd              ║
# ║    • Moran-Gianzero (1979) — formulação TIV                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Port JAX nativo de dipolos magnéticos (Sprint 3.3.1 parcial).

IMPORTANTE: Este módulo é EXPERIMENTAL. O kernel híbrido
(`_jax/kernel.py::fields_in_freqs_jax_batch`) continua sendo a API
preferida e totalmente funcional. Este módulo expõe helpers nativos
para experimentação e é a base para Sprints 3.3.2 e 3.3.3 futuras.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ──────────────────────────────────────────────────────────────────────────────
# decoupling_factors_jax — constantes estáticas ACp/ACx
# ──────────────────────────────────────────────────────────────────────────────


@jax.jit
def decoupling_factors_jax(L: float) -> tuple[jax.Array, jax.Array]:
    """Retorna constantes estáticas de decoupling (ACp, ACx) diferenciáveis.

    Fórmulas da errata CLAUDE.md:

        ACp = -1 / (4π · L³)   (planar: Hxx, Hyy)
        ACx = +1 / (2π · L³)   (axial: Hzz)

    Args:
        L: Espaçamento TR em metros (pode ser escalar ou array).

    Returns:
        Tupla (ACp, ACx) — float64 ou array compatível com shape de L.

    Note:
        Diferenciável via `jax.grad` — útil para análises de
        sensibilidade a L.

    Example:
        >>> ACp, ACx = decoupling_factors_jax(1.0)
        >>> float(ACp)  # -0.0795774...
        -0.07957747154594767
        >>> float(ACx)  # +0.1591549...
        0.15915494309189535
    """
    pi = jnp.pi
    L3 = L**3
    ACp = -1.0 / (4.0 * pi * L3)
    ACx = 1.0 / (2.0 * pi * L3)
    return ACp, ACx


# ──────────────────────────────────────────────────────────────────────────────
# _dipole_phases_jax — fatores exponenciais na camada do TX/RX
# ──────────────────────────────────────────────────────────────────────────────


@jax.jit
def _dipole_phases_jax(
    u_t: jax.Array,
    s_t: jax.Array,
    Tz: float,
    cz: float,
    prof_top_t: float,
    prof_bot_t: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Calcula fatores exponenciais do caso `camadR == camadT`.

    Port parcial de `magneticdipoles.f08:hmd_TIV_optimized` — ramo
    principal quando TX e RX estão na mesma camada. Este é o caso mais
    frequente em geosteering (dip pequeno, ferramenta dentro de uma
    camada homogênea).

    Args:
        u_t: (npt,) complex — constante TE da camada t.
        s_t: (npt,) complex — constante TM da camada t.
        Tz: profundidade do TX (m).
        cz: profundidade do RX (m).
        prof_top_t: profundidade do topo da camada t (m).
        prof_bot_t: profundidade do fundo da camada t (m).

    Returns:
        Tupla de 4 arrays (npt,) complex128:
          - exp_u_down: exp(-u_t * (cz - Tz))   [onda direta TE]
          - exp_s_down: exp(-s_t * (cz - Tz))   [onda direta TM]
          - exp_u_refl_top: exp(u_t * (prof_top_t - Tz - (cz - Tz))) [refl. topo]
          - exp_u_refl_bot: exp(-u_t * (prof_bot_t - Tz + (cz - Tz))) [refl. fundo]

    Note:
        Este é um dos ~12 conjuntos de fatores exponenciais usados em
        `hmd_tiv`. O port completo requer `lax.switch` sobre 6 casos.
    """
    dz = cz - Tz
    # Onda direta (downgoing) TE e TM
    exp_u_down = jnp.exp(-u_t * dz)
    exp_s_down = jnp.exp(-s_t * dz)
    # Reflexões nas interfaces superior e inferior da camada t
    exp_u_refl_top = jnp.exp(u_t * (prof_top_t - Tz - dz))
    exp_u_refl_bot = jnp.exp(-u_t * (prof_bot_t - Tz + dz))
    return exp_u_down, exp_s_down, exp_u_refl_top, exp_u_refl_bot


# ──────────────────────────────────────────────────────────────────────────────
# _hmd_tiv_same_layer_jax — HMD quando TX e RX na mesma camada (CASO 3)
# ──────────────────────────────────────────────────────────────────────────────


def _hmd_tiv_same_layer_jax(
    Tx: float,
    Ty: float,
    Tz: float,
    cx: float,
    cy: float,
    cz: float,
    u_t: jax.Array,
    s_t: jax.Array,
    uh_t: jax.Array,
    sh_t: jax.Array,
    RTEdw_t: jax.Array,
    RTEup_t: jax.Array,
    RTMdw_t: jax.Array,
    RTMup_t: jax.Array,
    Mxdw: jax.Array,
    Mxup: jax.Array,
    Eudw: jax.Array,
    Euup: jax.Array,
    prof_top_t: float,
    prof_bot_t: float,
    krJ0J1: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    zeta: complex,
    eta_t_h: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """HMD_TIV para camadR==camadT (caso mais comum em geosteering).

    Port parcial JAX de `_numba/dipoles.py:hmd_tiv` — implementa
    APENAS o ramo `camadR == camadT` com ordem `cz > Tz`.

    Para integração completa com o kernel forward, use o híbrido
    (`_jax/kernel.py`) que cobre os 6 casos.

    Args:
        Tx, Ty, Tz, cx, cy, cz: Coordenadas TX/RX (m).
        u_t, s_t, uh_t, sh_t: Constantes de propagação na camada t (npt,).
        RTEdw_t, RTEup_t, RTMdw_t, RTMup_t: Coeficientes de reflexão
            na camada t (npt,).
        Mxdw, Mxup, Eudw, Euup: Fatores de onda `common_factors` (npt,).
        prof_top_t, prof_bot_t: Profundidades do topo/fundo da camada t.
        krJ0J1, wJ0, wJ1: Filtro Hankel (npt,).
        zeta: impeditividade (complex escalar).
        eta_t_h: condutividade horizontal da camada t (float).

    Returns:
        Tupla ``(Hx_hmd, Hy_hmd, Hz_hmd)`` — arrays (2,) complex128 com
        componentes do campo para dipolos hmdx, hmdy.

    Note:
        EXPERIMENTAL. Em Sprint 3.3.2, esta função será integrada com
        os outros 5 casos geométricos via `lax.switch`. Por ora, o
        kernel híbrido é a API preferida.
    """
    # Distância radial
    dx = cx - Tx
    dy = cy - Ty
    r = jnp.sqrt(dx * dx + dy * dy)
    r_safe = jnp.where(r < 1e-12, 1e-2, r)  # guard singularidade

    # kr = abscissas / r_safe
    kr = krJ0J1 / r_safe
    kr2 = kr * kr

    # Fatores exponenciais (caso camadR==camadT)
    exp_u, exp_s, exp_u_top, exp_u_bot = _dipole_phases_jax(
        u_t, s_t, Tz, cz, prof_top_t, prof_bot_t
    )

    # Integral J0 para TM e J1 para TE/TM misturado (simplificação do
    # caso completo — para port completo ver Sprint 3.3.2).
    # Aqui, apenas ilustrativo: combinação TE (u) + TM (s) com pesos.
    kernel_te = Mxdw * exp_u  # downgoing TE
    kernel_tm = Eudw * exp_s  # downgoing TM

    I_j0_te = jnp.einsum("i,i->", wJ0, kernel_te * kr) / r_safe
    I_j1_te = jnp.einsum("i,i->", wJ1, kernel_te * kr2) / r_safe
    I_j0_tm = jnp.einsum("i,i->", wJ0, kernel_tm * kr) / r_safe

    # Placeholder: montagem simplificada. Port completo preserva as
    # 12 integrais do caso 3 e a álgebra de Liu (2017) eq. 4.80.
    Hx_hmd = jnp.stack([I_j0_tm / zeta, I_j1_te / zeta])
    Hy_hmd = jnp.stack([I_j1_te / zeta, I_j0_tm / zeta])
    Hz_hmd = jnp.stack([I_j0_te / zeta, I_j0_te / zeta])

    return Hx_hmd, Hy_hmd, Hz_hmd


# ──────────────────────────────────────────────────────────────────────────────
# Status de implementação (para consulta via código)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Sprint 3.3.2 — Kernels dos 6 casos geométricos (ETAPA 5 do hmd_tiv)
# ──────────────────────────────────────────────────────────────────────────────
#
# Cada caso computa (Ktm, Kte, Ktedz) — 3 arrays (npt,) complex128 — a partir
# de arrays layer-slice (npt,) passados pelo caller. A ideia é que o caller
# (_hmd_tiv_full_jax) pré-selecione `u[:, camad_r]`, `s[:, camad_r]`, etc.
# e passe essas fatias para as funções abaixo. Isto evita indexação dinâmica
# dentro das branches do lax.switch (que exigiria shapes uniformes).
#
# Assinatura comum dos 6 cases:
#   Args (npt,) complex128:
#     u_r, s_r          — constantes TE/TM na camada do RX
#     RTEdw_r, RTEup_r  — reflexões TE (camada RX)
#     RTMdw_r, RTMup_r  — reflexões TM (camada RX)
#     Tudw_r, Tuup_r    — potenciais propagados TE na camada RX
#     Txdw_r, Txup_r    — potenciais propagados TM na camada RX
#     Mxdw, Mxup, Eudw, Euup — fatores comuns (common_factors)
#   Args escalares (float):
#     z, h0, prof_r, prof_r1, h_r — geometria (z do RX, h0 do TX,
#       profundidades da camada RX, espessura da camada RX)
#   Returns:
#     Ktm, Kte, Ktedz — (npt,) complex128
#
# Referência: _numba/dipoles.py:547-644 (ETAPA 5 do hmd_tiv)


@jax.jit
def _hmd_tiv_kernel_case1_jax(
    u_r: jax.Array,
    s_r: jax.Array,
    RTEdw_r: jax.Array,
    RTEup_r: jax.Array,
    RTMdw_r: jax.Array,
    RTMup_r: jax.Array,
    Tudw_r: jax.Array,
    Tuup_r: jax.Array,
    Txdw_r: jax.Array,
    Txup_r: jax.Array,
    Mxdw: jax.Array,
    Mxup: jax.Array,
    Eudw: jax.Array,
    Euup: jax.Array,
    z: float,
    h0: float,
    prof_r: float,
    prof_r1: float,
    h_r: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Caso 1 — camadR==0 and camadT!=0 (RX na camada topo, TX abaixo).

    Referência Numba (linhas 547-551)::

        Ktm = Txup[:, 0] * np.exp(s[:, 0] * z)
        Kte = Tuup[:, 0] * np.exp(u[:, 0] * z)
        Ktedz = u[:, 0] * Kte
    """
    Ktm = Txup_r * jnp.exp(s_r * z)
    Kte = Tuup_r * jnp.exp(u_r * z)
    Ktedz = u_r * Kte
    return Ktm, Kte, Ktedz


@jax.jit
def _hmd_tiv_kernel_case2_jax(
    u_r,
    s_r,
    RTEdw_r,
    RTEup_r,
    RTMdw_r,
    RTMup_r,
    Tudw_r,
    Tuup_r,
    Txdw_r,
    Txup_r,
    Mxdw,
    Mxup,
    Eudw,
    Euup,
    z,
    h0,
    prof_r,
    prof_r1,
    h_r,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Caso 2 — camadR < camadT (RX acima do TX, camada interna).

    Referência Numba (linhas 552-572)::

        Ktm = Txup[:, r] * (exp(s*(z - prof[r+1]))
              + RTMup[:, r] * exp(-s*(z - prof[r] + h[r])))
        Kte = Tuup[:, r] * (exp(u*(z - prof[r+1]))
              + RTEup[:, r] * exp(-u*(z - prof[r] + h[r])))
        Ktedz = u * Tuup * (exp(...) - RTEup * exp(...))
    """
    a_tm = jnp.exp(s_r * (z - prof_r1))
    b_tm = RTMup_r * jnp.exp(-s_r * (z - prof_r + h_r))
    Ktm = Txup_r * (a_tm + b_tm)

    a_te = jnp.exp(u_r * (z - prof_r1))
    b_te = RTEup_r * jnp.exp(-u_r * (z - prof_r + h_r))
    Kte = Tuup_r * (a_te + b_te)

    Ktedz = u_r * Tuup_r * (a_te - b_te)
    return Ktm, Kte, Ktedz


@jax.jit
def _hmd_tiv_kernel_case3_jax(
    u_r,
    s_r,
    RTEdw_r,
    RTEup_r,
    RTMdw_r,
    RTMup_r,
    Tudw_r,
    Tuup_r,
    Txdw_r,
    Txup_r,
    Mxdw,
    Mxup,
    Eudw,
    Euup,
    z,
    h0,
    prof_r,
    prof_r1,
    h_r,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Caso 3 — camadR==camadT and z <= h0 (mesma camada, RX acima do TX).

    Referência Numba (linhas 573-595).
    """
    a_tm = jnp.exp(s_r * (z - h0))
    b_tm = RTMup_r * Mxup * jnp.exp(-s_r * (z - prof_r))
    c_tm = RTMdw_r * Mxdw * jnp.exp(s_r * (z - prof_r1))
    Ktm = Txup_r * (a_tm + b_tm + c_tm)

    a_te = jnp.exp(u_r * (z - h0))
    b_te = RTEup_r * Euup * jnp.exp(-u_r * (z - prof_r))
    c_te = RTEdw_r * Eudw * jnp.exp(u_r * (z - prof_r1))
    Kte = Tuup_r * (a_te + b_te - c_te)

    Ktedz = u_r * Tuup_r * (a_te - b_te - c_te)
    return Ktm, Kte, Ktedz


@jax.jit
def _hmd_tiv_kernel_case4_jax(
    u_r,
    s_r,
    RTEdw_r,
    RTEup_r,
    RTMdw_r,
    RTMup_r,
    Tudw_r,
    Tuup_r,
    Txdw_r,
    Txup_r,
    Mxdw,
    Mxup,
    Eudw,
    Euup,
    z,
    h0,
    prof_r,
    prof_r1,
    h_r,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Caso 4 — camadR==camadT and z > h0 (mesma camada, RX abaixo do TX).

    Referência Numba (linhas 596-618).
    """
    a_tm = jnp.exp(-s_r * (z - h0))
    b_tm = RTMup_r * Mxup * jnp.exp(-s_r * (z - prof_r))
    c_tm = RTMdw_r * Mxdw * jnp.exp(s_r * (z - prof_r1))
    Ktm = Txdw_r * (a_tm + b_tm + c_tm)

    a_te = jnp.exp(-u_r * (z - h0))
    b_te = RTEup_r * Euup * jnp.exp(-u_r * (z - prof_r))
    c_te = RTEdw_r * Eudw * jnp.exp(u_r * (z - prof_r1))
    Kte = Tudw_r * (a_te - b_te + c_te)

    Ktedz = -u_r * Tudw_r * (a_te - b_te - c_te)
    return Ktm, Kte, Ktedz


@jax.jit
def _hmd_tiv_kernel_case5_jax(
    u_r,
    s_r,
    RTEdw_r,
    RTEup_r,
    RTMdw_r,
    RTMup_r,
    Tudw_r,
    Tuup_r,
    Txdw_r,
    Txup_r,
    Mxdw,
    Mxup,
    Eudw,
    Euup,
    z,
    h0,
    prof_r,
    prof_r1,
    h_r,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Caso 5 — camadR > camadT and camadR != n-1 (RX abaixo, interna).

    Referência Numba (linhas 619-639).
    """
    a_tm = jnp.exp(-s_r * (z - prof_r))
    b_tm = RTMdw_r * jnp.exp(s_r * (z - prof_r1 - h_r))
    Ktm = Txdw_r * (a_tm + b_tm)

    a_te = jnp.exp(-u_r * (z - prof_r))
    b_te = RTEdw_r * jnp.exp(u_r * (z - prof_r1 - h_r))
    Kte = Tudw_r * (a_te + b_te)

    Ktedz = -u_r * Tudw_r * (a_te - b_te)
    return Ktm, Kte, Ktedz


@jax.jit
def _hmd_tiv_kernel_case6_jax(
    u_r,
    s_r,
    RTEdw_r,
    RTEup_r,
    RTMdw_r,
    RTMup_r,
    Tudw_r,
    Tuup_r,
    Txdw_r,
    Txup_r,
    Mxdw,
    Mxup,
    Eudw,
    Euup,
    z,
    h0,
    prof_r,
    prof_r1,
    h_r,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Caso 6 — camadR == n-1 (RX na última camada semi-infinita).

    Referência Numba (linhas 640-644)::

        Ktm = Txdw[:, r] * exp(-s*(z - prof[r]))
        Kte = Tudw[:, r] * exp(-u*(z - prof[r]))
        Ktedz = -u * Kte
    """
    Ktm = Txdw_r * jnp.exp(-s_r * (z - prof_r))
    Kte = Tudw_r * jnp.exp(-u_r * (z - prof_r))
    Ktedz = -u_r * Kte
    return Ktm, Kte, Ktedz


# ──────────────────────────────────────────────────────────────────────────────
# compute_case_index_jax — determina índice 0..5 do caso geométrico
# ──────────────────────────────────────────────────────────────────────────────


def compute_case_index_jax(
    camad_r: int, camad_t: int, n: int, z: float, h0: float
) -> int:
    """Mapeia `(camad_r, camad_t, z, h0)` → índice 0..5 do caso hmd_tiv.

    Esta função é chamada no nível Python (fora de `@jit`) porque depende
    de branching baseado em valores concretos. O `lax.switch` no
    dispatcher recebe o inteiro retornado.

    Args:
        camad_r: Índice da camada do receptor (0-indexed).
        camad_t: Índice da camada do transmissor (0-indexed).
        n: Número total de camadas.
        z: Profundidade do RX (m).
        h0: Profundidade do TX (m).

    Returns:
        Inteiro em [0, 5] correspondente a um dos 6 casos:
          0: camadR==0 and camadT!=0        (caso 1)
          1: camadR < camadT                 (caso 2)
          2: camadR==camadT and z <= h0      (caso 3)
          3: camadR==camadT and z > h0       (caso 4)
          4: camadR > camadT and camadR != n-1 (caso 5)
          5: camadR == n-1                   (caso 6)

    Note:
        A ordem é importante — replica o `if/elif` do Numba em
        `dipoles.py:547-644`.
    """
    if camad_r == 0 and camad_t != 0:
        return 0
    if camad_r < camad_t:
        return 1
    if camad_r == camad_t and z <= h0:
        return 2
    if camad_r == camad_t and z > h0:
        return 3
    if camad_r > camad_t and camad_r != (n - 1):
        return 4
    return 5


# ──────────────────────────────────────────────────────────────────────────────
# _hmd_tiv_full_jax — dispatcher com lax.switch (Sprint 3.3.2)
# ──────────────────────────────────────────────────────────────────────────────


def _hmd_tiv_full_jax(
    case_index: int,
    u_r: jax.Array,
    s_r: jax.Array,
    RTEdw_r: jax.Array,
    RTEup_r: jax.Array,
    RTMdw_r: jax.Array,
    RTMup_r: jax.Array,
    Tudw_r: jax.Array,
    Tuup_r: jax.Array,
    Txdw_r: jax.Array,
    Txup_r: jax.Array,
    Mxdw: jax.Array,
    Mxup: jax.Array,
    Eudw: jax.Array,
    Euup: jax.Array,
    z: float,
    h0: float,
    prof_r: float,
    prof_r1: float,
    h_r: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dispatcher ETAPA 5 do hmd_tiv via `lax.switch` nos 6 casos geométricos.

    Esta função reproduz bit-a-bit os kernels Ktm/Kte/Ktedz da Fortran
    `magneticdipoles.f08` (linhas 310-383) e do port Numba
    (`_numba/dipoles.py:547-644`). A diferença chave é a substituição do
    Python `if/elif` por `jax.lax.switch`, que:

      - é rastreável por `jax.jit` (um compile para todas as 6 ramos)
      - é diferenciável via `jax.grad` (nenhuma branch Python)
      - suporta vectorização automática via `jax.vmap`

    Args:
        case_index: Inteiro em [0, 5] obtido via
            :func:`compute_case_index_jax`.
        u_r, s_r, ...: Arrays `(npt,) complex128` — slice da camada RX.
        Mxdw, Mxup, Eudw, Euup: Fatores comuns `(npt,) complex128`.
        z, h0, prof_r, prof_r1, h_r: Escalares geométricos (float).

    Returns:
        `(Ktm, Kte, Ktedz)` — 3 arrays `(npt,) complex128`.

    Note:
        Esta é a saída da **ETAPA 5** do hmd_tiv. Para obter `(Hx, Hy, Hz)`,
        o caller deve continuar com a ETAPA 6 (assembly Ward-Hohmann 4.3)
        que multiplica os kernels pelos pesos Hankel `wJ0`/`wJ1` e monta
        as 6 componentes de saída. Veja `_jax/kernel.py` para a integração
        completa.

    Example:
        >>> idx = compute_case_index_jax(camad_r=1, camad_t=1,
        ...                              n=3, z=5.0, h0=4.0)
        >>> # idx == 3 (caso 4: mesma camada, z > h0)
        >>> Ktm, Kte, Ktedz = _hmd_tiv_full_jax(idx, ...)
    """
    branches = [
        _hmd_tiv_kernel_case1_jax,
        _hmd_tiv_kernel_case2_jax,
        _hmd_tiv_kernel_case3_jax,
        _hmd_tiv_kernel_case4_jax,
        _hmd_tiv_kernel_case5_jax,
        _hmd_tiv_kernel_case6_jax,
    ]
    return jax.lax.switch(
        case_index,
        branches,
        u_r,
        s_r,
        RTEdw_r,
        RTEup_r,
        RTMdw_r,
        RTMup_r,
        Tudw_r,
        Tuup_r,
        Txdw_r,
        Txup_r,
        Mxdw,
        Mxup,
        Eudw,
        Euup,
        z,
        h0,
        prof_r,
        prof_r1,
        h_r,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Status de implementação (para consulta via código)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Sprint 3.3.3 — VMD native (Vertical Magnetic Dipole)
# ──────────────────────────────────────────────────────────────────────────────
# NOTA DE DESIGN — parâmetros não usados em alguns kernels:
#   `lax.switch` exige que TODOS os branches tenham EXATAMENTE a mesma
#   assinatura (mesmo número, ordem e tipo de argumentos). Por isso cada
#   `_vmd_kernel_caseN_jax` declara os 15 parâmetros completos mesmo
#   quando alguns não aparecem no corpo (caso 1 não usa TEdwz_r/FEdwz/
#   FEupz/prof_r, etc.). É o mesmo padrão dos kernels HMD acima — não é
#   um bug, é o trade-off para single-compile do dispatcher.
# ──────────────────────────────────────────────────────────────────────────────
# Port da ETAPA 5 do `vmd()` (Numba `_numba/dipoles.py:856-945`,
# Fortran `magneticdipoles.f08:527-623`) para JAX nativo via 6 kernels +
# `lax.switch`. Diferente do HMD, o VMD utiliza APENAS o potencial TE
# vertical (TEdwz/TEupz), não TM — por isso a assinatura é mais enxuta
# (não precisa de `s, RTM*, Tudw/Tuup, Txdw/Txup, Mxdw/Mxup, Eudw/Euup`).
#
# Os kernels retornam `(KtezJ0, KtedzzJ1)` shape `(npt,) complex128`.
# A propagação dos potenciais TEdwz/TEupz através das camadas (lógica
# Numba 787-854) é deferida ao caller (caminho hybrid em `_jax/kernel.py`)
# enquanto Sprint 3.3.4 não é entregue. O dispatcher pode ser usado em
# isolamento para análises de sensibilidade locais (uma camada).
# ──────────────────────────────────────────────────────────────────────────────


@jax.jit
def _vmd_kernel_case1_jax(
    TEdwz_r: jax.Array,
    TEupz_r: jax.Array,
    u_r: jax.Array,
    RTEdw_r: jax.Array,
    RTEup_r: jax.Array,
    AdmInt_r: jax.Array,
    FEdwz: jax.Array,
    FEupz: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    z: float,
    h0: float,
    prof_r: float,
    prof_r1: float,
    h_r: float,
) -> tuple[jax.Array, jax.Array]:
    """VMD ETAPA 5 — caso 1: ``camad_r==0 and camad_t!=0`` (RX na superfície).

    Replica Numba ``_numba/dipoles.py:863-867``::

        fac = TEupz[:, 0] * exp(u[:, 0] * z)
        KtedzzJ1 = AdmInt[:, 0] * fac * wJ1
    """
    fac = TEupz_r * jnp.exp(u_r * z)
    KtezJ0 = fac * wJ0
    KtedzzJ1 = AdmInt_r * fac * wJ1
    return KtezJ0, KtedzzJ1


@jax.jit
def _vmd_kernel_case2_jax(
    TEdwz_r: jax.Array,
    TEupz_r: jax.Array,
    u_r: jax.Array,
    RTEdw_r: jax.Array,
    RTEup_r: jax.Array,
    AdmInt_r: jax.Array,
    FEdwz: jax.Array,
    FEupz: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    z: float,
    h0: float,
    prof_r: float,
    prof_r1: float,
    h_r: float,
) -> tuple[jax.Array, jax.Array]:
    """VMD ETAPA 5 — caso 2: ``camad_r < camad_t`` (RX acima do TX).

    Replica Numba ``_numba/dipoles.py:868-884``.
    """
    a = jnp.exp(u_r * (z - prof_r1))
    b = RTEup_r * jnp.exp(-u_r * (z - prof_r + h_r))
    fac = TEupz_r * (a + b)
    KtezJ0 = fac * wJ0
    KtedzzJ1 = AdmInt_r * TEupz_r * (a - b) * wJ1
    return KtezJ0, KtedzzJ1


@jax.jit
def _vmd_kernel_case3_jax(
    TEdwz_r: jax.Array,
    TEupz_r: jax.Array,
    u_r: jax.Array,
    RTEdw_r: jax.Array,
    RTEup_r: jax.Array,
    AdmInt_r: jax.Array,
    FEdwz: jax.Array,
    FEupz: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    z: float,
    h0: float,
    prof_r: float,
    prof_r1: float,
    h_r: float,
) -> tuple[jax.Array, jax.Array]:
    """VMD ETAPA 5 — caso 3: ``camad_r==camad_t and z<=h0`` (mesma camada, RX acima).

    Replica Numba ``_numba/dipoles.py:885-903``.
    """
    a = jnp.exp(u_r * (z - h0))
    b = RTEup_r * FEupz * jnp.exp(-u_r * (z - prof_r))
    c = RTEdw_r * FEdwz * jnp.exp(u_r * (z - prof_r1))
    fac = TEupz_r * (a + b + c)
    KtezJ0 = fac * wJ0
    KtedzzJ1 = AdmInt_r * TEupz_r * (a - b + c) * wJ1
    return KtezJ0, KtedzzJ1


@jax.jit
def _vmd_kernel_case4_jax(
    TEdwz_r: jax.Array,
    TEupz_r: jax.Array,
    u_r: jax.Array,
    RTEdw_r: jax.Array,
    RTEup_r: jax.Array,
    AdmInt_r: jax.Array,
    FEdwz: jax.Array,
    FEupz: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    z: float,
    h0: float,
    prof_r: float,
    prof_r1: float,
    h_r: float,
) -> tuple[jax.Array, jax.Array]:
    """VMD ETAPA 5 — caso 4: ``camad_r==camad_t and z>h0`` (mesma camada, RX abaixo).

    Replica Numba ``_numba/dipoles.py:904-922``.
    """
    a = jnp.exp(-u_r * (z - h0))
    b = RTEup_r * FEupz * jnp.exp(-u_r * (z - prof_r))
    c = RTEdw_r * FEdwz * jnp.exp(u_r * (z - prof_r1))
    fac = TEdwz_r * (a + b + c)
    KtezJ0 = fac * wJ0
    KtedzzJ1 = -AdmInt_r * TEdwz_r * (a + b - c) * wJ1
    return KtezJ0, KtedzzJ1


@jax.jit
def _vmd_kernel_case5_jax(
    TEdwz_r: jax.Array,
    TEupz_r: jax.Array,
    u_r: jax.Array,
    RTEdw_r: jax.Array,
    RTEup_r: jax.Array,
    AdmInt_r: jax.Array,
    FEdwz: jax.Array,
    FEupz: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    z: float,
    h0: float,
    prof_r: float,
    prof_r1: float,
    h_r: float,
) -> tuple[jax.Array, jax.Array]:
    """VMD ETAPA 5 — caso 5: ``camad_r > camad_t and camad_r != n-1`` (RX abaixo, intermediária).

    Replica Numba ``_numba/dipoles.py:923-939``.
    """
    a = jnp.exp(-u_r * (z - prof_r))
    b = RTEdw_r * jnp.exp(u_r * (z - prof_r1 - h_r))
    fac = TEdwz_r * (a + b)
    KtezJ0 = fac * wJ0
    KtedzzJ1 = -AdmInt_r * TEdwz_r * (a - b) * wJ1
    return KtezJ0, KtedzzJ1


@jax.jit
def _vmd_kernel_case6_jax(
    TEdwz_r: jax.Array,
    TEupz_r: jax.Array,
    u_r: jax.Array,
    RTEdw_r: jax.Array,
    RTEup_r: jax.Array,
    AdmInt_r: jax.Array,
    FEdwz: jax.Array,
    FEupz: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    z: float,
    h0: float,
    prof_r: float,
    prof_r1: float,
    h_r: float,
) -> tuple[jax.Array, jax.Array]:
    """VMD ETAPA 5 — caso 6: ``camad_r == n-1`` (RX na última camada).

    Replica Numba ``_numba/dipoles.py:940-945``.
    """
    fac = TEdwz_r * jnp.exp(-u_r * (z - prof_r))
    KtezJ0 = fac * wJ0
    KtedzzJ1 = -AdmInt_r * fac * wJ1
    return KtezJ0, KtedzzJ1


def _vmd_full_jax(
    case_index: int,
    TEdwz_r: jax.Array,
    TEupz_r: jax.Array,
    u_r: jax.Array,
    RTEdw_r: jax.Array,
    RTEup_r: jax.Array,
    AdmInt_r: jax.Array,
    FEdwz: jax.Array,
    FEupz: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    z: float,
    h0: float,
    prof_r: float,
    prof_r1: float,
    h_r: float,
) -> tuple[jax.Array, jax.Array]:
    """Dispatcher VMD ETAPA 5 via ``lax.switch`` nos 6 casos geométricos.

    Análogo a :func:`_hmd_tiv_full_jax`, mas com assinatura enxuta (apenas
    arrays TE e ``AdmInt`` — não consome RTM/Tudw/Mxdw como o HMD).

    Args:
        case_index: Inteiro em [0, 5] obtido via :func:`compute_case_index_jax`.
        TEdwz_r, TEupz_r: Potenciais TE descendente/ascendente da camada do
            RX, ``(npt,) complex128`` — propagados pelo caller (loops da
            propagação Fortran 479-524 / Numba 787-854 ainda no path hybrid).
        u_r: ``(npt,) complex128`` — kernel vertical TE da camada do RX.
        RTEdw_r, RTEup_r: ``(npt,) complex128`` — coeficientes de reflexão TE.
        AdmInt_r: ``(npt,) complex128`` — admitância intrínseca da camada.
        FEdwz, FEupz: ``(npt,) complex128`` — fatores comuns (de
            ``common_factors_jax``).
        wJ0, wJ1: ``(npt,) float64`` — pesos do filtro Hankel.
        z, h0, prof_r, prof_r1, h_r: Escalares geométricos (float).

    Returns:
        ``(KtezJ0, KtedzzJ1)`` — 2 arrays ``(npt,) complex128`` que o caller
        deve combinar com ``kr`` para obter ``(Hx, Hy, Hz)`` finais (Numba
        947-959)::

            sum_KtedzzJ1_kr2 = sum(KtedzzJ1 * kr * kr)
            sum_KtezJ0_kr3   = sum(KtezJ0 * kr * kr * kr)
            Hx = -x * sum_KtedzzJ1_kr2 / (2π r) / r
            Hy = -y * sum_KtedzzJ1_kr2 / (2π r) / r
            Hz =       sum_KtezJ0_kr3   / (2π ζ r)

    Note:
        Esta função é **diferenciável via** ``jax.grad`` e suporta ``vmap``.
        Útil para PINNs locais de uma camada e para análises de sensibilidade
        ∂Hz/∂rho_h por camada-alvo. O wiring no kernel completo (HMD+VMD
        end-to-end + ETAPAS 3+6 da propagação) é Sprint 3.3.4.

    Example:
        >>> idx = compute_case_index_jax(camad_r=1, camad_t=1,
        ...                              n=3, z=5.0, h0=4.0)
        >>> # idx == 3 (caso 4: mesma camada, z > h0)
        >>> KtezJ0, KtedzzJ1 = _vmd_full_jax(idx, TEdwz, TEupz, u, ...)
    """
    branches = [
        _vmd_kernel_case1_jax,
        _vmd_kernel_case2_jax,
        _vmd_kernel_case3_jax,
        _vmd_kernel_case4_jax,
        _vmd_kernel_case5_jax,
        _vmd_kernel_case6_jax,
    ]
    return jax.lax.switch(
        case_index,
        branches,
        TEdwz_r,
        TEupz_r,
        u_r,
        RTEdw_r,
        RTEup_r,
        AdmInt_r,
        FEdwz,
        FEupz,
        wJ0,
        wJ1,
        z,
        h0,
        prof_r,
        prof_r1,
        h_r,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Status de implementação (para consulta via código)
# ──────────────────────────────────────────────────────────────────────────────

IMPLEMENTATION_STATUS = {
    "decoupling_factors_jax": "✅ completo (diferenciável)",
    "_dipole_phases_jax": "✅ completo (caso camadR==camadT)",
    "_hmd_tiv_same_layer_jax": "🟡 parcial (apenas caso 3 de 6, scaffolding)",
    "_hmd_tiv_kernel_case1_jax..case6_jax": "✅ completos (Sprint 3.3.2)",
    "_hmd_tiv_full_jax": "✅ dispatcher lax.switch (Sprint 3.3.2)",
    "compute_case_index_jax": "✅ mapeia geometria → índice 0..5",
    "_vmd_kernel_case1_jax..case6_jax": "✅ completos (Sprint 3.3.3)",
    "_vmd_full_jax": "✅ dispatcher lax.switch (Sprint 3.3.3)",
    "ETAPAS 3+6 (TEdwz/TEupz prop + tensor assembly)": "⏳ Sprint 3.3.4",
}


__all__ = [
    "decoupling_factors_jax",
    "_dipole_phases_jax",
    "_hmd_tiv_same_layer_jax",
    # Sprint 3.3.2 — kernels HMD dos 6 casos
    "_hmd_tiv_kernel_case1_jax",
    "_hmd_tiv_kernel_case2_jax",
    "_hmd_tiv_kernel_case3_jax",
    "_hmd_tiv_kernel_case4_jax",
    "_hmd_tiv_kernel_case5_jax",
    "_hmd_tiv_kernel_case6_jax",
    "compute_case_index_jax",
    "_hmd_tiv_full_jax",
    # Sprint 3.3.3 — kernels VMD dos 6 casos
    "_vmd_kernel_case1_jax",
    "_vmd_kernel_case2_jax",
    "_vmd_kernel_case3_jax",
    "_vmd_kernel_case4_jax",
    "_vmd_kernel_case5_jax",
    "_vmd_kernel_case6_jax",
    "_vmd_full_jax",
    "IMPLEMENTATION_STATUS",
]
