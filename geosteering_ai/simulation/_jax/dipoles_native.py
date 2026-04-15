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
    # Sprint 7.x (PR #14d): fast-path quando camad_r/camad_t/n são Python ints
    # concretos (bucketed). Sprint 10 Phase 2 (PR #24-part2): tracer-path
    # quando camad_r/camad_t são JAX tracers (unified, vmap sobre posições).
    if isinstance(camad_r, int) and isinstance(camad_t, int) and isinstance(n, int):
        # Caminho fast-path — todos os argumentos inteiros são concretos.
        # Usa ``jnp.where`` apenas quando ``z``/``h0`` são tracers.
        if camad_r == 0 and camad_t != 0:
            return 0
        if camad_r < camad_t:
            return 1
        if camad_r == camad_t:
            # z vs h0 pode ser tracer → resolve via jnp.where.
            return jnp.where(z <= h0, 2, 3)
        if camad_r > camad_t and camad_r != (n - 1):
            return 4
        return 5
    # Caminho tracer (Sprint 10 Phase 2): camad_r/camad_t são tracers int32.
    # Toda a árvore de decisão usa ``jnp.where`` encadeado. Ordem equivalente
    # ao if/elif do fast-path — o primeiro branch verdadeiro vence.
    case = jnp.where(
        (camad_r == 0) & (camad_t != 0),
        0,
        jnp.where(
            camad_r < camad_t,
            1,
            jnp.where(
                (camad_r == camad_t) & (z <= h0),
                2,
                jnp.where(
                    camad_r == camad_t,  # z > h0 implícito (já filtrou z<=h0)
                    3,
                    jnp.where(
                        (camad_r > camad_t) & (camad_r != (n - 1)),
                        4,
                        5,  # fallback: camad_r == n-1 (caso 6)
                    ),
                ),
            ),
        ),
    )
    return case


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
# Sprint 3.3.4 — ETAPAS 3+6 Nativas JAX (propagação + assembly)
# ──────────────────────────────────────────────────────────────────────────────
# Port completo end-to-end dos dipolos HMD e VMD em JAX puro, eliminando
# a dependência de `jax.pure_callback` → Numba host. Permite:
#   • `jax.grad` end-to-end sobre rho_h, rho_v, esp
#   • Execução 100% em GPU (sem callback para CPU host)
#   • `jax.vmap` e `jax.pmap` sobre posições/frequências
#
# ESTRATÉGIA: n, camad_r, camad_t são Python ints (concretos no trace time),
# então usamos Python loops/if que JAX unroll durante tracing. Cada
# combinação (n, camad_r, camad_t) gera um XLA program especializado.
# Para n típico (3-22 camadas), o overhead de compilação é aceitável
# (< 120s, regression guard em test_native_compile_time_budget).
#
# REFERÊNCIAS:
#   • _numba/dipoles.py:340-522 (HMD ETAPA 3 propagação)
#   • _numba/dipoles.py:524-535 (ETAPA 4 fatores geométricos)
#   • _numba/dipoles.py:646-697 (HMD ETAPA 6 assembly Ward-Hohmann)
#   • _numba/dipoles.py:787-854 (VMD ETAPA 3 propagação)
#   • _numba/dipoles.py:947-960 (VMD ETAPA 6 assembly)
# ──────────────────────────────────────────────────────────────────────────────

_MX = 1.0  # Momento magnético unitário (A·m²)
_MZ = 1.0
_PI_NATIVE = jnp.pi
_TWO_PI_NATIVE = 2.0 * jnp.pi
_EPS_SINGULARITY = 1e-12
_R_GUARD = 0.01  # Guard para r → 0 (paridade Fortran utils.f08:195)


def _hmd_tiv_native_jax(
    Tx: float,
    Ty: float,
    h0: float,
    n: int,
    camad_r: int,
    camad_t: int,
    npt: int,
    krJ0J1: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    h: jax.Array,
    prof: jax.Array,
    zeta: complex,
    eta: jax.Array,
    cx: float,
    cy: float,
    z: float,
    u: jax.Array,
    s: jax.Array,
    uh: jax.Array,
    sh: jax.Array,
    RTEdw: jax.Array,
    RTEup: jax.Array,
    RTMdw: jax.Array,
    RTMup: jax.Array,
    Mxdw: jax.Array,
    Mxup: jax.Array,
    Eudw: jax.Array,
    Euup: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """HMD TIV end-to-end em JAX nativo (ETAPAS 3+4+5+6).

    Port line-for-line de ``_numba/dipoles.py:hmd_tiv`` (1300 LOC).
    Retorna ``(Hx_p, Hy_p, Hz_p)`` — cada array ``(2,) complex128``
    onde índice 0 = hmdx, índice 1 = hmdy.

    Args:
        Mesmos que ``hmd_tiv`` Numba. Arrays ``u, s, uh, sh, RTE*, RTM*``
        são ``(npt, n) complex128`` de ``common_arrays_jax``.
        ``Mxdw, Mxup, Eudw, Euup`` são ``(npt,) complex128`` de
        ``common_factors_jax``.
    """
    # ── Geometria local ─────────────────────────────────────────────
    dx = cx - Tx
    dy = cy - Ty
    x = jnp.where(jnp.abs(dx) < _EPS_SINGULARITY, 0.0, dx)
    y = jnp.where(jnp.abs(dy) < _EPS_SINGULARITY, 0.0, dy)
    r = jnp.sqrt(x * x + y * y)
    r = jnp.where(r < _EPS_SINGULARITY, _R_GUARD, r)
    kr = krJ0J1 / r

    # ── ETAPA 3: Propagação dos potenciais TM/TE entre camadas ──────
    Txdw = jnp.zeros((npt, n), dtype=jnp.complex128)
    Tudw = jnp.zeros((npt, n), dtype=jnp.complex128)
    Txup = jnp.zeros((npt, n), dtype=jnp.complex128)
    Tuup = jnp.zeros((npt, n), dtype=jnp.complex128)

    if camad_r > camad_t:
        # Caso A: RX abaixo do TX — propagação descendente
        for j in range(camad_t, camad_r + 1):
            if j == camad_t:
                Txdw = Txdw.at[:, j].set(_MX / (2.0 * s[:, camad_t]))
                Tudw = Tudw.at[:, j].set(-_MX / 2.0)
            elif j == (camad_t + 1) and j == (n - 1):
                if n > 1:
                    Txdw = Txdw.at[:, j].set(
                        s[:, j - 1]
                        * Txdw[:, j - 1]
                        * (
                            jnp.exp(-s[:, j - 1] * (prof[camad_t + 1] - h0))
                            + RTMup[:, j - 1] * Mxup * jnp.exp(-sh[:, j - 1])
                            - RTMdw[:, j - 1] * Mxdw
                        )
                        / s[:, j]
                    )
                    Tudw = Tudw.at[:, j].set(
                        u[:, j - 1]
                        * Tudw[:, j - 1]
                        * (
                            jnp.exp(-u[:, j - 1] * (prof[camad_t + 1] - h0))
                            - RTEup[:, j - 1] * Euup * jnp.exp(-uh[:, j - 1])
                            - RTEdw[:, j - 1] * Eudw
                        )
                        / u[:, j]
                    )
                else:
                    Txdw = Txdw.at[:, j].set(
                        s[:, j - 1]
                        * Txdw[:, j - 1]
                        * (jnp.exp(s[:, j - 1] * h0) - RTMdw[:, j - 1] * Mxdw)
                        / s[:, j]
                    )
                    Tudw = Tudw.at[:, j].set(
                        u[:, j - 1]
                        * Tudw[:, j - 1]
                        * (jnp.exp(u[:, j - 1] * h0) - RTEdw[:, j - 1] * Eudw)
                        / u[:, j]
                    )
            elif j == (camad_t + 1) and j != (n - 1):
                Txdw = Txdw.at[:, j].set(
                    s[:, j - 1]
                    * Txdw[:, j - 1]
                    * (
                        jnp.exp(-s[:, j - 1] * (prof[camad_t + 1] - h0))
                        + RTMup[:, j - 1] * Mxup * jnp.exp(-sh[:, j - 1])
                        - RTMdw[:, j - 1] * Mxdw
                    )
                    / ((1.0 - RTMdw[:, j] * jnp.exp(-2.0 * sh[:, j])) * s[:, j])
                )
                Tudw = Tudw.at[:, j].set(
                    u[:, j - 1]
                    * Tudw[:, j - 1]
                    * (
                        jnp.exp(-u[:, j - 1] * (prof[camad_t + 1] - h0))
                        - RTEup[:, j - 1] * Euup * jnp.exp(-uh[:, j - 1])
                        - RTEdw[:, j - 1] * Eudw
                    )
                    / ((1.0 - RTEdw[:, j] * jnp.exp(-2.0 * uh[:, j])) * u[:, j])
                )
            elif j != (n - 1):
                Txdw = Txdw.at[:, j].set(
                    s[:, j - 1]
                    * Txdw[:, j - 1]
                    * jnp.exp(-sh[:, j - 1])
                    * (1.0 - RTMdw[:, j - 1])
                    / ((1.0 - RTMdw[:, j] * jnp.exp(-2.0 * sh[:, j])) * s[:, j])
                )
                Tudw = Tudw.at[:, j].set(
                    u[:, j - 1]
                    * Tudw[:, j - 1]
                    * jnp.exp(-uh[:, j - 1])
                    * (1.0 - RTEdw[:, j - 1])
                    / ((1.0 - RTEdw[:, j] * jnp.exp(-2.0 * uh[:, j])) * u[:, j])
                )
            else:  # j == n - 1
                Txdw = Txdw.at[:, j].set(
                    s[:, j - 1]
                    * Txdw[:, j - 1]
                    * jnp.exp(-sh[:, j - 1])
                    * (1.0 - RTMdw[:, j - 1])
                    / s[:, j]
                )
                Tudw = Tudw.at[:, j].set(
                    u[:, j - 1]
                    * Tudw[:, j - 1]
                    * jnp.exp(-uh[:, j - 1])
                    * (1.0 - RTEdw[:, j - 1])
                    / u[:, j]
                )
    elif camad_r < camad_t:
        # Caso B: RX acima do TX — propagação ascendente
        for j in range(camad_t, camad_r - 1, -1):
            if j == camad_t:
                Txup = Txup.at[:, j].set(_MX / (2.0 * s[:, camad_t]))
                Tuup = Tuup.at[:, j].set(_MX / 2.0)
            elif j == (camad_t - 1) and j == 0:
                Txup = Txup.at[:, j].set(
                    s[:, j + 1]
                    * Txup[:, j + 1]
                    * (
                        jnp.exp(-s[:, j + 1] * h0)
                        - RTMup[:, j + 1] * Mxup
                        + RTMdw[:, j + 1] * Mxdw * jnp.exp(-sh[:, j + 1])
                    )
                    / s[:, j]
                )
                Tuup = Tuup.at[:, j].set(
                    u[:, j + 1]
                    * Tuup[:, j + 1]
                    * (
                        jnp.exp(-u[:, j + 1] * h0)
                        - RTEup[:, j + 1] * Euup
                        - RTEdw[:, j + 1] * Eudw * jnp.exp(-uh[:, j + 1])
                    )
                    / u[:, j]
                )
            elif j == (camad_t - 1) and j != 0:
                Txup = Txup.at[:, j].set(
                    s[:, j + 1]
                    * Txup[:, j + 1]
                    * (
                        jnp.exp(s[:, j + 1] * (prof[j + 1] - h0))
                        + RTMdw[:, j + 1] * Mxdw * jnp.exp(-sh[:, j + 1])
                        - RTMup[:, j + 1] * Mxup
                    )
                    / ((1.0 - RTMup[:, j] * jnp.exp(-2.0 * sh[:, j])) * s[:, j])
                )
                Tuup = Tuup.at[:, j].set(
                    u[:, j + 1]
                    * Tuup[:, j + 1]
                    * (
                        jnp.exp(u[:, j + 1] * (prof[camad_t] - h0))
                        - RTEup[:, j + 1] * Euup
                        - RTEdw[:, j + 1] * Eudw * jnp.exp(-uh[:, j + 1])
                    )
                    / ((1.0 - RTEup[:, j] * jnp.exp(-2.0 * uh[:, j])) * u[:, j])
                )
            elif j != 0:
                Txup = Txup.at[:, j].set(
                    s[:, j + 1]
                    * Txup[:, j + 1]
                    * jnp.exp(-sh[:, j + 1])
                    * (1.0 - RTMup[:, j + 1])
                    / ((1.0 - RTMup[:, j] * jnp.exp(-2.0 * sh[:, j])) * s[:, j])
                )
                Tuup = Tuup.at[:, j].set(
                    u[:, j + 1]
                    * Tuup[:, j + 1]
                    * jnp.exp(-uh[:, j + 1])
                    * (1.0 - RTEup[:, j + 1])
                    / ((1.0 - RTEup[:, j] * jnp.exp(-2.0 * uh[:, j])) * u[:, j])
                )
            else:  # j == 0
                Txup = Txup.at[:, j].set(
                    s[:, j + 1]
                    * Txup[:, j + 1]
                    * jnp.exp(-sh[:, j + 1])
                    * (1.0 - RTMup[:, j + 1])
                    / s[:, j]
                )
                Tuup = Tuup.at[:, j].set(
                    u[:, j + 1]
                    * Tuup[:, j + 1]
                    * jnp.exp(-uh[:, j + 1])
                    * (1.0 - RTEup[:, j + 1])
                    / u[:, j]
                )
    else:
        # Caso C: mesma camada
        Tudw = Tudw.at[:, camad_t].set(-_MX / 2.0)
        Tuup = Tuup.at[:, camad_t].set(_MX / 2.0)
        Txdw = Txdw.at[:, camad_t].set(_MX / (2.0 * s[:, camad_t]))
        Txup = Txup.at[:, camad_t].set(Txdw[:, camad_t])

    # ── ETAPA 4: Fatores geométricos ────────────────────────────────
    x2_r2 = x * x / (r * r)
    y2_r2 = y * y / (r * r)
    xy_r2 = x * y / (r * r)
    twox2_r2m1 = 2.0 * x2_r2 - 1.0
    twoy2_r2m1 = 2.0 * y2_r2 - 1.0
    twopir = _TWO_PI_NATIVE * r
    kh2_r = -zeta * eta[camad_r, 0]

    # ── ETAPA 5: Kernels via dispatcher existente ───────────────────
    case_idx = compute_case_index_jax(camad_r, camad_t, n, z, h0)
    Ktm, Kte, Ktedz = _hmd_tiv_full_jax(
        case_idx,
        u[:, camad_r],
        s[:, camad_r],
        RTEdw[:, camad_r],
        RTEup[:, camad_r],
        RTMdw[:, camad_r],
        RTMup[:, camad_r],
        Tudw[:, camad_r],
        Tuup[:, camad_r],
        Txdw[:, camad_r],
        Txup[:, camad_r],
        Mxdw,
        Mxup,
        Eudw,
        Euup,
        z,
        h0,
        prof[camad_r],
        prof[camad_r + 1],
        h[camad_r],
    )

    # ── ETAPA 6: Assembly Ward-Hohmann (hmdx + hmdy) ────────────────
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1
    Kte_J1 = Kte * wJ1
    Ktedz_J0 = Ktedz * wJ0
    Ktedz_J1 = Ktedz * wJ1

    sum_Ktedz_J1 = jnp.sum(Ktedz_J1)
    sum_Ktm_J1 = jnp.sum(Ktm_J1)
    sum_Ktedz_J0_kr = jnp.sum(Ktedz_J0 * kr)
    sum_Ktm_J0_kr = jnp.sum(Ktm_J0 * kr)
    sum_add_J1 = jnp.sum(Ktedz_J1 + kh2_r * Ktm_J1)
    sum_add_J0_kr = jnp.sum((Ktedz_J0 + kh2_r * Ktm_J0) * kr)
    sum_Kte_J1_kr2 = jnp.sum(Kte_J1 * kr * kr)

    # HMDX (índice 0)
    kernelHxJ1 = (twox2_r2m1 * sum_Ktedz_J1 - kh2_r * twoy2_r2m1 * sum_Ktm_J1) / r
    kernelHxJ0 = x2_r2 * sum_Ktedz_J0_kr - kh2_r * y2_r2 * sum_Ktm_J0_kr
    Hx0 = (kernelHxJ1 - kernelHxJ0) / twopir
    kernelHyJ1 = sum_add_J1 / r
    kernelHyJ0 = sum_add_J0_kr / 2.0
    Hy0 = xy_r2 * (kernelHyJ1 - kernelHyJ0) / _PI_NATIVE / r
    kernelHzJ1 = x * sum_Kte_J1_kr2 / r
    Hz0 = -kernelHzJ1 / twopir

    # HMDY (índice 1)
    Hx1 = Hy0  # simetria: Hx_hmdy = Hy_hmdx
    kernelHyJ1_y = (twoy2_r2m1 * sum_Ktedz_J1 - kh2_r * twox2_r2m1 * sum_Ktm_J1) / r
    kernelHyJ0_y = y2_r2 * sum_Ktedz_J0_kr - kh2_r * x2_r2 * sum_Ktm_J0_kr
    Hy1 = (kernelHyJ1_y - kernelHyJ0_y) / twopir
    kernelHzJ1_y = y * sum_Kte_J1_kr2 / r
    Hz1 = -kernelHzJ1_y / twopir

    Hx_p = jnp.array([Hx0, Hx1])
    Hy_p = jnp.array([Hy0, Hy1])
    Hz_p = jnp.array([Hz0, Hz1])
    return Hx_p, Hy_p, Hz_p


def _vmd_native_jax(
    Tx: float,
    Ty: float,
    h0: float,
    n: int,
    camad_r: int,
    camad_t: int,
    npt: int,
    krJ0J1: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    h: jax.Array,
    prof: jax.Array,
    zeta: complex,
    cx: float,
    cy: float,
    z: float,
    u: jax.Array,
    uh: jax.Array,
    AdmInt: jax.Array,
    RTEdw: jax.Array,
    RTEup: jax.Array,
    FEdwz: jax.Array,
    FEupz: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """VMD end-to-end em JAX nativo (ETAPAS 3+5+6).

    Port line-for-line de ``_numba/dipoles.py:vmd`` (260 LOC).
    Retorna ``(Hx_p, Hy_p, Hz_p)`` — escalares complex128.
    """
    # ── Geometria local ─────────────────────────────────────────────
    dx = cx - Tx
    dy = cy - Ty
    x = jnp.where(jnp.abs(dx) < _EPS_SINGULARITY, 0.0, dx)
    y = jnp.where(jnp.abs(dy) < _EPS_SINGULARITY, 0.0, dy)
    r = jnp.sqrt(x * x + y * y)
    r = jnp.where(r < _EPS_SINGULARITY, _R_GUARD, r)
    kr = krJ0J1 / r

    # ── ETAPA 3: Propagação TEdwz / TEupz ───────────────────────────
    TEdwz = jnp.zeros((npt, n), dtype=jnp.complex128)
    TEupz = jnp.zeros((npt, n), dtype=jnp.complex128)

    if camad_r > camad_t:
        for j in range(camad_t, camad_r + 1):
            if j == camad_t:
                TEdwz = TEdwz.at[:, j].set(zeta * _MZ / (2.0 * u[:, j]))
            elif j == (camad_t + 1) and j == (n - 1):
                TEdwz = TEdwz.at[:, j].set(
                    TEdwz[:, j - 1]
                    * (
                        jnp.exp(-u[:, camad_t] * (prof[camad_t + 1] - h0))
                        + RTEup[:, camad_t] * FEupz * jnp.exp(-uh[:, camad_t])
                        + RTEdw[:, camad_t] * FEdwz
                    )
                )
            elif j == (camad_t + 1) and j != (n - 1):
                TEdwz = TEdwz.at[:, j].set(
                    TEdwz[:, j - 1]
                    * (
                        jnp.exp(-u[:, camad_t] * (prof[camad_t + 1] - h0))
                        + RTEup[:, camad_t] * FEupz * jnp.exp(-uh[:, camad_t])
                        + RTEdw[:, camad_t] * FEdwz
                    )
                    / (1.0 + RTEdw[:, j] * jnp.exp(-2.0 * uh[:, j]))
                )
            elif j != (n - 1):
                TEdwz = TEdwz.at[:, j].set(
                    TEdwz[:, j - 1]
                    * (1.0 + RTEdw[:, j - 1])
                    * jnp.exp(-uh[:, j - 1])
                    / (1.0 + RTEdw[:, j] * jnp.exp(-2.0 * uh[:, j]))
                )
            else:  # j == n - 1
                TEdwz = TEdwz.at[:, j].set(
                    TEdwz[:, j - 1] * (1.0 + RTEdw[:, j - 1]) * jnp.exp(-uh[:, j - 1])
                )
    elif camad_r < camad_t:
        for j in range(camad_t, camad_r - 1, -1):
            if j == camad_t:
                TEupz = TEupz.at[:, j].set(zeta * _MZ / (2.0 * u[:, j]))
            elif j == (camad_t - 1) and j == 0:
                TEupz = TEupz.at[:, j].set(
                    TEupz[:, j + 1]
                    * (
                        jnp.exp(-u[:, camad_t] * h0)
                        + RTEup[:, camad_t] * FEupz
                        + RTEdw[:, camad_t] * FEdwz * jnp.exp(-uh[:, camad_t])
                    )
                )
            elif j == (camad_t - 1) and j != 0:
                TEupz = TEupz.at[:, j].set(
                    TEupz[:, j + 1]
                    * (
                        jnp.exp(u[:, camad_t] * (prof[camad_t] - h0))
                        + RTEup[:, camad_t] * FEupz
                        + RTEdw[:, camad_t] * FEdwz * jnp.exp(-uh[:, camad_t])
                    )
                    / (1.0 + RTEup[:, j] * jnp.exp(-2.0 * uh[:, j]))
                )
            elif j != 0:
                TEupz = TEupz.at[:, j].set(
                    TEupz[:, j + 1]
                    * (1.0 + RTEup[:, j + 1])
                    * jnp.exp(-uh[:, j + 1])
                    / (1.0 + RTEup[:, j] * jnp.exp(-2.0 * uh[:, j]))
                )
            else:  # j == 0
                TEupz = TEupz.at[:, j].set(
                    TEupz[:, j + 1] * (1.0 + RTEup[:, j + 1]) * jnp.exp(-uh[:, j + 1])
                )
    else:
        # Mesma camada
        TEdwz = TEdwz.at[:, camad_r].set(zeta * _MZ / (2.0 * u[:, camad_t]))
        TEupz = TEupz.at[:, camad_r].set(TEdwz[:, camad_r])

    # ── ETAPA 5: Kernels via dispatcher existente ───────────────────
    case_idx = compute_case_index_jax(camad_r, camad_t, n, z, h0)
    KtezJ0, KtedzzJ1 = _vmd_full_jax(
        case_idx,
        TEdwz[:, camad_r],
        TEupz[:, camad_r],
        u[:, camad_r],
        RTEdw[:, camad_r],
        RTEup[:, camad_r],
        AdmInt[:, camad_r],
        FEdwz,
        FEupz,
        wJ0,
        wJ1,
        z,
        h0,
        prof[camad_r],
        prof[camad_r + 1],
        h[camad_r],
    )

    # ── ETAPA 6: Assembly Hx, Hy, Hz ───────────────────────────────
    twopir = _TWO_PI_NATIVE * r
    sum_KtedzzJ1_kr2 = jnp.sum(KtedzzJ1 * kr * kr)
    sum_KtezJ0_kr3 = jnp.sum(KtezJ0 * kr * kr * kr)
    Hx_p = -x * sum_KtedzzJ1_kr2 / twopir / r
    Hy_p = -y * sum_KtedzzJ1_kr2 / twopir / r
    Hz_p = sum_KtezJ0_kr3 / 2.0 / _PI_NATIVE / zeta / r
    return Hx_p, Hy_p, Hz_p


def native_dipoles_full_jax(
    Tx: float,
    Ty: float,
    Tz: float,
    n: int,
    camad_r: int,
    camad_t: int,
    npt: int,
    krJ0J1: jax.Array,
    wJ0: jax.Array,
    wJ1: jax.Array,
    h_arr: jax.Array,
    prof_arr: jax.Array,
    zeta: complex,
    eta: jax.Array,
    cx: float,
    cy: float,
    cz: float,
    u: jax.Array,
    s: jax.Array,
    uh: jax.Array,
    sh: jax.Array,
    RTEdw: jax.Array,
    RTEup: jax.Array,
    RTMdw: jax.Array,
    RTMup: jax.Array,
    Mxdw: jax.Array,
    Mxup: jax.Array,
    Eudw: jax.Array,
    Euup: jax.Array,
    FEdwz: jax.Array,
    FEupz: jax.Array,
) -> jax.Array:
    """Orquestrador end-to-end: HMD (hmdx+hmdy) + VMD → matH (3,3).

    Substitui ``_dipoles_numba_host`` (``pure_callback``) no kernel JAX
    quando ``use_native_dipoles=True``. Mesma interface de saída:
    array ``(3,3) complex128`` com::

        matH[0,:] = [Hxx, Hxy, Hxz]   (hmdx)
        matH[1,:] = [Hyx, Hyy, Hyz]   (hmdy)
        matH[2,:] = [Hzx, Hzy, Hzz]   (vmd)

    **Diferenciável** via ``jax.grad`` sobre qualquer input JAX.
    """
    h0 = Tz  # profundidade do TX (convenção Numba: h0 = Tz)

    # HMD: retorna (Hx[2], Hy[2], Hz[2])
    Hx_hmd, Hy_hmd, Hz_hmd = _hmd_tiv_native_jax(
        Tx,
        Ty,
        h0,
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
    )

    # VMD: AdmInt = u / zeta (recalc como em _dipoles_numba_host)
    AdmInt = u / zeta
    Hx_vmd, Hy_vmd, Hz_vmd = _vmd_native_jax(
        Tx,
        Ty,
        h0,
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
        cx,
        cy,
        cz,
        u,
        uh,
        AdmInt,
        RTEdw,
        RTEup,
        FEdwz,
        FEupz,
    )

    # Assembly (3,3) — mesma ordem que _dipoles_numba_host
    matH = jnp.array(
        [
            [Hx_hmd[0], Hy_hmd[0], Hz_hmd[0]],
            [Hx_hmd[1], Hy_hmd[1], Hz_hmd[1]],
            [Hx_vmd, Hy_vmd, Hz_vmd],
        ]
    )
    return matH


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 10 Phase 2 — Wrappers unified (PR #24-part2)
# ──────────────────────────────────────────────────────────────────────────────
# Diferença dos wrappers legados:
#   • ETAPA 3 (propagação TE/TM) usa `_hmd_tiv_propagation_unified` /
#     `_vmd_propagation_unified` (jax.lax.fori_loop + jnp.where encadeado) em
#     vez de `for j in range(camad_t, camad_r+1)` Python → `camad_t`/`camad_r`
#     viram tracers int32.
#   • ETAPAS 4/5/6 permanecem inalteradas (já tracer-compat via
#     `compute_case_index_jax`, `_hmd_tiv_full_jax` e `_vmd_full_jax`).
#   • Consequência: 44 programas XLA → 1 por (n, npt) → VRAM T4 ~11 GB → 250 MB.
#
# Usados apenas quando `cfg.jax_strategy="unified"` (opt-in). O caminho legacy
# (`_hmd_tiv_native_jax`, `_vmd_native_jax`, `native_dipoles_full_jax`)
# permanece default e intocado (backward-compat com testes LRU).
#
# Referências:
#   • `dipoles_unified._hmd_tiv_propagation_unified:L451-594`
#   • `dipoles_unified._vmd_propagation_unified:L788-911`
# ──────────────────────────────────────────────────────────────────────────────
from geosteering_ai.simulation._jax.dipoles_unified import (  # noqa: E402
    _hmd_tiv_propagation_unified,
    _vmd_propagation_unified,
)


def _hmd_tiv_native_jax_unified(
    Tx: float,
    Ty: float,
    h0: float,
    n: int,
    camad_r,  # ← tracer int32 (Sprint 10 Phase 2)
    camad_t,  # ← tracer int32 (Sprint 10 Phase 2)
    npt: int,
    krJ0J1: "jax.Array",
    wJ0: "jax.Array",
    wJ1: "jax.Array",
    h: "jax.Array",
    prof: "jax.Array",
    zeta: complex,
    eta: "jax.Array",
    cx: float,
    cy: float,
    z: float,
    u: "jax.Array",
    s: "jax.Array",
    uh: "jax.Array",
    sh: "jax.Array",
    RTEdw: "jax.Array",
    RTEup: "jax.Array",
    RTMdw: "jax.Array",
    RTMup: "jax.Array",
    Mxdw: "jax.Array",
    Mxup: "jax.Array",
    Eudw: "jax.Array",
    Euup: "jax.Array",
):
    """HMD TIV unified (Sprint 10 Phase 2) — espelho de ``_hmd_tiv_native_jax``.

    Substitui a ETAPA 3 (propagação TE/TM) por
    :func:`_hmd_tiv_propagation_unified`, permitindo que ``camad_t``/``camad_r``
    sejam tracers int32 (não mais Python ints). ETAPAS 4/5/6 permanecem
    idênticas ao legacy — reutilizam ``compute_case_index_jax`` e
    ``_hmd_tiv_full_jax`` (já tracer-compat).

    Returns:
        ``(Hx_p, Hy_p, Hz_p)`` — cada array ``(2,) complex128`` (idx 0 = hmdx,
        idx 1 = hmdy). Semântica idêntica a :func:`_hmd_tiv_native_jax`.
    """
    # ── Geometria local (idem legacy) ───────────────────────────────
    dx = cx - Tx
    dy = cy - Ty
    x = jnp.where(jnp.abs(dx) < _EPS_SINGULARITY, 0.0, dx)
    y = jnp.where(jnp.abs(dy) < _EPS_SINGULARITY, 0.0, dy)
    r = jnp.sqrt(x * x + y * y)
    r = jnp.where(r < _EPS_SINGULARITY, _R_GUARD, r)
    kr = krJ0J1 / r

    # ── ETAPA 3 UNIFIED: propagação via fori_loop ───────────────────
    Txdw, Tudw, Txup, Tuup = _hmd_tiv_propagation_unified(
        camad_t,
        camad_r,
        n,
        npt,
        s,
        u,
        sh,
        uh,
        RTMdw,
        RTMup,
        RTEdw,
        RTEup,
        Mxdw,
        Mxup,
        Eudw,
        Euup,
        prof,
        h0,
    )

    # ── ETAPA 4: Fatores geométricos (idem legacy) ──────────────────
    x2_r2 = x * x / (r * r)
    y2_r2 = y * y / (r * r)
    xy_r2 = x * y / (r * r)
    twox2_r2m1 = 2.0 * x2_r2 - 1.0
    twoy2_r2m1 = 2.0 * y2_r2 - 1.0
    twopir = _TWO_PI_NATIVE * r
    kh2_r = -zeta * eta[camad_r, 0]  # dynamic_slice (tracer-safe)

    # ── ETAPA 5: Kernels via lax.switch (tracer-safe) ───────────────
    case_idx = compute_case_index_jax(camad_r, camad_t, n, z, h0)
    Ktm, Kte, Ktedz = _hmd_tiv_full_jax(
        case_idx,
        u[:, camad_r],
        s[:, camad_r],
        RTEdw[:, camad_r],
        RTEup[:, camad_r],
        RTMdw[:, camad_r],
        RTMup[:, camad_r],
        Tudw[:, camad_r],
        Tuup[:, camad_r],
        Txdw[:, camad_r],
        Txup[:, camad_r],
        Mxdw,
        Mxup,
        Eudw,
        Euup,
        z,
        h0,
        prof[camad_r],
        prof[camad_r + 1],
        h[camad_r],
    )

    # ── ETAPA 6: Assembly Ward-Hohmann (idem legacy) ────────────────
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1
    Kte_J1 = Kte * wJ1
    Ktedz_J0 = Ktedz * wJ0
    Ktedz_J1 = Ktedz * wJ1

    sum_Ktedz_J1 = jnp.sum(Ktedz_J1)
    sum_Ktm_J1 = jnp.sum(Ktm_J1)
    sum_Ktedz_J0_kr = jnp.sum(Ktedz_J0 * kr)
    sum_Ktm_J0_kr = jnp.sum(Ktm_J0 * kr)
    sum_add_J1 = jnp.sum(Ktedz_J1 + kh2_r * Ktm_J1)
    sum_add_J0_kr = jnp.sum((Ktedz_J0 + kh2_r * Ktm_J0) * kr)
    sum_Kte_J1_kr2 = jnp.sum(Kte_J1 * kr * kr)

    # HMDX (índice 0)
    kernelHxJ1 = (twox2_r2m1 * sum_Ktedz_J1 - kh2_r * twoy2_r2m1 * sum_Ktm_J1) / r
    kernelHxJ0 = x2_r2 * sum_Ktedz_J0_kr - kh2_r * y2_r2 * sum_Ktm_J0_kr
    Hx0 = (kernelHxJ1 - kernelHxJ0) / twopir
    kernelHyJ1 = sum_add_J1 / r
    kernelHyJ0 = sum_add_J0_kr / 2.0
    Hy0 = xy_r2 * (kernelHyJ1 - kernelHyJ0) / _PI_NATIVE / r
    kernelHzJ1 = x * sum_Kte_J1_kr2 / r
    Hz0 = -kernelHzJ1 / twopir

    # HMDY (índice 1)
    Hx1 = Hy0
    kernelHyJ1_y = (twoy2_r2m1 * sum_Ktedz_J1 - kh2_r * twox2_r2m1 * sum_Ktm_J1) / r
    kernelHyJ0_y = y2_r2 * sum_Ktedz_J0_kr - kh2_r * x2_r2 * sum_Ktm_J0_kr
    Hy1 = (kernelHyJ1_y - kernelHyJ0_y) / twopir
    kernelHzJ1_y = y * sum_Kte_J1_kr2 / r
    Hz1 = -kernelHzJ1_y / twopir

    return jnp.array([Hx0, Hx1]), jnp.array([Hy0, Hy1]), jnp.array([Hz0, Hz1])


def _vmd_native_jax_unified(
    Tx: float,
    Ty: float,
    h0: float,
    n: int,
    camad_r,  # ← tracer int32
    camad_t,  # ← tracer int32
    npt: int,
    krJ0J1: "jax.Array",
    wJ0: "jax.Array",
    wJ1: "jax.Array",
    h: "jax.Array",
    prof: "jax.Array",
    zeta: complex,
    cx: float,
    cy: float,
    z: float,
    u: "jax.Array",
    uh: "jax.Array",
    AdmInt: "jax.Array",
    RTEdw: "jax.Array",
    RTEup: "jax.Array",
    FEdwz: "jax.Array",
    FEupz: "jax.Array",
):
    """VMD unified (Sprint 10 Phase 2) — espelho de ``_vmd_native_jax``.

    Substitui a ETAPA 3 (propagação TE) por
    :func:`_vmd_propagation_unified`; ETAPAS 5/6 permanecem idênticas ao legacy.
    Retorna ``(Hx_p, Hy_p, Hz_p)`` — escalares complex128.
    """
    # ── Geometria local (idem legacy) ───────────────────────────────
    dx = cx - Tx
    dy = cy - Ty
    x = jnp.where(jnp.abs(dx) < _EPS_SINGULARITY, 0.0, dx)
    y = jnp.where(jnp.abs(dy) < _EPS_SINGULARITY, 0.0, dy)
    r = jnp.sqrt(x * x + y * y)
    r = jnp.where(r < _EPS_SINGULARITY, _R_GUARD, r)
    kr = krJ0J1 / r

    # ── ETAPA 3 UNIFIED: propagação TE via fori_loop ────────────────
    TEdwz, TEupz = _vmd_propagation_unified(
        camad_t,
        camad_r,
        n,
        npt,
        u,
        uh,
        RTEdw,
        RTEup,
        FEdwz,
        FEupz,
        prof,
        h0,
        zeta,
    )

    # ── ETAPA 5: Kernels via lax.switch (tracer-safe) ───────────────
    case_idx = compute_case_index_jax(camad_r, camad_t, n, z, h0)
    KtezJ0, KtedzzJ1 = _vmd_full_jax(
        case_idx,
        TEdwz[:, camad_r],
        TEupz[:, camad_r],
        u[:, camad_r],
        RTEdw[:, camad_r],
        RTEup[:, camad_r],
        AdmInt[:, camad_r],
        FEdwz,
        FEupz,
        wJ0,
        wJ1,
        z,
        h0,
        prof[camad_r],
        prof[camad_r + 1],
        h[camad_r],
    )

    # ── ETAPA 6: Assembly (idem legacy) ─────────────────────────────
    twopir = _TWO_PI_NATIVE * r
    sum_KtedzzJ1_kr2 = jnp.sum(KtedzzJ1 * kr * kr)
    sum_KtezJ0_kr3 = jnp.sum(KtezJ0 * kr * kr * kr)
    Hx_p = -x * sum_KtedzzJ1_kr2 / twopir / r
    Hy_p = -y * sum_KtedzzJ1_kr2 / twopir / r
    Hz_p = sum_KtezJ0_kr3 / 2.0 / _PI_NATIVE / zeta / r
    return Hx_p, Hy_p, Hz_p


def native_dipoles_full_jax_unified(
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
):
    """Orquestrador unified: HMD + VMD → matH (3,3). Paridade 1:1 com
    :func:`native_dipoles_full_jax` (legacy) mas aceita ``camad_t``/``camad_r``
    como tracers. **Diferenciável** via ``jax.grad``/``jax.jacfwd``.
    """
    h0 = Tz

    Hx_hmd, Hy_hmd, Hz_hmd = _hmd_tiv_native_jax_unified(
        Tx,
        Ty,
        h0,
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
    )

    AdmInt = u / zeta
    Hx_vmd, Hy_vmd, Hz_vmd = _vmd_native_jax_unified(
        Tx,
        Ty,
        h0,
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
        cx,
        cy,
        cz,
        u,
        uh,
        AdmInt,
        RTEdw,
        RTEup,
        FEdwz,
        FEupz,
    )

    return jnp.array(
        [
            [Hx_hmd[0], Hy_hmd[0], Hz_hmd[0]],
            [Hx_hmd[1], Hy_hmd[1], Hz_hmd[1]],
            [Hx_vmd, Hy_vmd, Hz_vmd],
        ]
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
    "_hmd_tiv_native_jax": "✅ HMD end-to-end ETAPAS 3+4+5+6 (Sprint 3.3.4)",
    "_vmd_native_jax": "✅ VMD end-to-end ETAPAS 3+5+6 (Sprint 3.3.4)",
    "native_dipoles_full_jax": "✅ orquestrador HMD+VMD → matH (3,3) (Sprint 3.3.4)",
    "_hmd_tiv_native_jax_unified": "✅ HMD unified — camad tracers (Sprint 10 Phase 2)",
    "_vmd_native_jax_unified": "✅ VMD unified — camad tracers (Sprint 10 Phase 2)",
    "native_dipoles_full_jax_unified": "✅ orquestrador unified (Sprint 10 Phase 2)",
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
    # Sprint 3.3.4 — ETAPAS 3+6 end-to-end
    "_hmd_tiv_native_jax",
    "_vmd_native_jax",
    "native_dipoles_full_jax",
    # Sprint 10 Phase 2 — wrappers unified (PR #24-part2)
    "_hmd_tiv_native_jax_unified",
    "_vmd_native_jax_unified",
    "native_dipoles_full_jax_unified",
    "IMPLEMENTATION_STATUS",
]
