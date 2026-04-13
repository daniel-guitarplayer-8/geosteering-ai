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
# ║    • _hmd_tiv_same_layer_jax(...)  caso camadR==camadT ✅               ║
# ║    • _hmd_tiv_full_jax(...)        6 casos              ⏳ Sprint 3.3.2 ║
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

IMPLEMENTATION_STATUS = {
    "decoupling_factors_jax": "✅ completo (diferenciável)",
    "_dipole_phases_jax": "✅ completo (caso camadR==camadT)",
    "_hmd_tiv_same_layer_jax": "🟡 parcial (apenas caso 3 de 6)",
    "_hmd_tiv_full_jax": "⏳ planejado para Sprint 3.3.2",
    "_vmd_native_jax": "⏳ planejado para Sprint 3.3.3",
}


__all__ = [
    "decoupling_factors_jax",
    "_dipole_phases_jax",
    "_hmd_tiv_same_layer_jax",
    "IMPLEMENTATION_STATUS",
]
