# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/propagation.py                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend JAX Propagação (Sprint 3.2)    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : JAX 0.4.30+                                                ║
# ║  Dependências: jax, jaxlib                                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Port JAX das funções `common_arrays` e `common_factors` do           ║
# ║    subpacote `_numba/propagation.py`. Usa `jax.lax.scan` para as        ║
# ║    recursões TE/TM bottom-up e top-down, permitindo compilação XLA     ║
# ║    sem perda de performance em loops Python.                             ║
# ║                                                                           ║
# ║  ESTRATÉGIA DE PORT                                                       ║
# ║    • Recursões TE/TM → `jax.lax.scan` (loop compilado + carry complexo)║
# ║    • Constantes por camada → `jax.vmap` sobre eixo de camadas          ║
# ║    • Operações primitivas → `jnp.sqrt`, `jnp.exp`, divisões elemento   ║
# ║    • Arrays complex128 → respeitar `jax.config jax_enable_x64=True`    ║
# ║                                                                           ║
# ║  DIFERENCIABILIDADE                                                       ║
# ║    Todas as operações são primitivas diferenciáveis. Permite uso       ║
# ║    de `jax.grad`/`jax.jacfwd` para gradientes ∂(common_arrays)/∂ρ       ║
# ║    diretamente, fundação para PINNs na Fase 5.                           ║
# ║                                                                           ║
# ║  PARIDADE COM NUMBA                                                       ║
# ║    Tolerância esperada: < 1e-12 (float64) nas 9 saídas. Diferenças    ║
# ║    residuais vêm de reordenamento XLA das somas em matmul implícitos. ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • _numba/propagation.py — implementação de referência                ║
# ║    • Fortran utils.f08:158-297 — commonarraysMD + commonfactorsMD      ║
# ║    • JAX docs: jax.lax.scan (https://jax.readthedocs.io)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Port JAX de common_arrays + common_factors (Sprint 3.2).

Reusa a lógica de :mod:`geosteering_ai.simulation._numba.propagation`,
substituindo loops Python por :func:`jax.lax.scan` para permitir
compilação XLA sem overhead interpretado.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Garantia de float64 (ver nota no _jax/__init__.py)
jax.config.update("jax_enable_x64", True)

# Constantes idênticas ao backend Numba
_HORDIST_SINGULARITY_EPS: float = 1.0e-12
_HORDIST_SINGULARITY_R: float = 1.0e-2


# ──────────────────────────────────────────────────────────────────────────────
# common_arrays_jax — port de _numba.common_arrays
# ──────────────────────────────────────────────────────────────────────────────


def common_arrays_jax(
    n: int,
    npt: int,
    hordist: float,
    krJ0J1: jax.Array,
    zeta: complex,
    h: jax.Array,
    eta: jax.Array,
) -> tuple[jax.Array, ...]:
    """Port JAX de :func:`_numba.propagation.common_arrays`.

    Retorna os 9 arrays complexos necessários pelos dipolos, usando
    ``jax.lax.scan`` para as recursões TE/TM.

    Args:
        n: Número de camadas (int estático para JIT).
        npt: Número de pontos do filtro Hankel.
        hordist: Distância horizontal (m). Se < 1e-12, usa 1e-2 m
            (paridade Fortran).
        krJ0J1: Array ``(npt,)`` float64 — abscissas do filtro.
        zeta: Impeditividade complexa ``i·ω·μ₀`` (scalar complex128).
        h: Array ``(n,)`` float64 — espessuras por camada.
        eta: Array ``(n, 2)`` float64 — condutividades ``[σh, σv]``.

    Returns:
        Tupla de 9 arrays ``(npt, n)`` complex128:
        ``(u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt)``.

    Note:
        ``n`` e ``npt`` devem ser conhecidos estaticamente para usar
        ``@jax.jit(static_argnames=('n', 'npt'))`` upstream. A função
        em si NÃO é decorada com jit — deixa escolha ao caller (útil
        dentro de ``vmap`` onde jit é aplicado ao nível externo).
    """
    # Guard de singularidade — jnp.where para branch-free
    r = jnp.where(hordist < _HORDIST_SINGULARITY_EPS, _HORDIST_SINGULARITY_R, hordist)

    kr = krJ0J1 / r  # (npt,)
    kr_squared = (kr * kr).astype(jnp.complex128)  # promovido para complex

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 1: Constantes de propagação por camada (vmap sobre camadas)
    # ──────────────────────────────────────────────────────────────────
    # Para cada camada i, calcula u[:, i], s[:, i], uh, sh, AdmInt, ImpInt
    # usando operações primitivas. Usa vmap over i para eficiência XLA.

    def _per_layer(h_i, eta_i):
        """Retorna (u_i, s_i, uh_i, sh_i, AdmInt_i, ImpInt_i, tghuh_i, tghsh_i)."""
        sigma_h = eta_i[0]
        sigma_v = eta_i[1]
        kh2 = -zeta * sigma_h
        kv2 = -zeta * sigma_v
        lamb2 = sigma_h / sigma_v

        u_i = jnp.sqrt(kr_squared - kh2)  # (npt,) complex
        v_i = jnp.sqrt(kr_squared - kv2)
        s_i = jnp.sqrt(lamb2) * v_i

        AdmInt_i = u_i / zeta
        ImpInt_i = s_i / sigma_h

        uh_i = u_i * h_i
        sh_i = s_i * h_i

        # Forma explícita de tanh (bit-match com Fortran utils.f08:212)
        exp_m2uh = jnp.exp(-2.0 * uh_i)
        exp_m2sh = jnp.exp(-2.0 * sh_i)
        tghuh_i = (1.0 - exp_m2uh) / (1.0 + exp_m2uh)
        tghsh_i = (1.0 - exp_m2sh) / (1.0 + exp_m2sh)

        return u_i, s_i, uh_i, sh_i, AdmInt_i, ImpInt_i, tghuh_i, tghsh_i

    # vmap sobre camadas → shape final (n, npt, ...) — precisamos transpor
    # para (npt, n) para consistência com backend Numba.
    u_all, s_all, uh_all, sh_all, AdmInt_all, ImpInt_all, tghuh_all, tghsh_all = jax.vmap(
        _per_layer, in_axes=(0, 0)
    )(h, eta)
    # Atual: shape (n, npt). Transpor para (npt, n).
    u = u_all.T
    s = s_all.T
    uh = uh_all.T
    sh = sh_all.T
    AdmInt = AdmInt_all.T
    ImpInt = ImpInt_all.T
    tghuh = tghuh_all.T
    tghsh = tghsh_all.T

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 2: Recursão bottom-up (camada n-1 → 0) via jax.lax.scan
    # ──────────────────────────────────────────────────────────────────
    # Terminal: camada n-1 (meio infinito inferior) — RT = 0, Adm = AdmInt
    # Para cada i de n-2 até 0:
    #   AdmApdw[:, i]  = AdmInt[:,i] * (AdmApdw[:,i+1] + AdmInt[:,i]*tghuh[:,i])
    #                  / (AdmInt[:,i] + AdmApdw[:,i+1]*tghuh[:,i])
    #   ImpApdw[:, i]  = ... (análogo com Imp/tghsh)
    #   RTEdw[:, i]    = (AdmInt[:,i] - AdmApdw[:,i+1]) / (AdmInt[:,i] + AdmApdw[:,i+1])
    #   RTMdw[:, i]    = ... (análogo Imp)

    def _bottom_up_step(carry, i):
        """Recursão bottom-up: computa (AdmApdw[i], ImpApdw[i], RTEdw[i], RTMdw[i])."""
        AdmApdw_next, ImpApdw_next = carry  # (npt,) — camada i+1
        AdmInt_i = AdmInt[:, i]
        ImpInt_i = ImpInt[:, i]
        tghuh_i = tghuh[:, i]
        tghsh_i = tghsh[:, i]

        AdmApdw_i = (
            AdmInt_i
            * (AdmApdw_next + AdmInt_i * tghuh_i)
            / (AdmInt_i + AdmApdw_next * tghuh_i)
        )
        ImpApdw_i = (
            ImpInt_i
            * (ImpApdw_next + ImpInt_i * tghsh_i)
            / (ImpInt_i + ImpApdw_next * tghsh_i)
        )
        RTEdw_i = (AdmInt_i - AdmApdw_next) / (AdmInt_i + AdmApdw_next)
        RTMdw_i = (ImpInt_i - ImpApdw_next) / (ImpInt_i + ImpApdw_next)

        return (AdmApdw_i, ImpApdw_i), (AdmApdw_i, ImpApdw_i, RTEdw_i, RTMdw_i)

    # Condições terminais na camada n-1
    AdmApdw_terminal = AdmInt[:, n - 1]
    ImpApdw_terminal = ImpInt[:, n - 1]

    if n > 1:
        indices_bottom = jnp.arange(n - 2, -1, -1)
        _, (AdmApdw_stacked, ImpApdw_stacked, RTEdw_stacked, RTMdw_stacked) = (
            jax.lax.scan(
                _bottom_up_step,
                (AdmApdw_terminal, ImpApdw_terminal),
                indices_bottom,
            )
        )
        # scan retorna em ordem de iteração (n-2, n-3, ..., 0) — precisamos
        # reverter para que o eixo i esteja alinhado com (0, 1, ..., n-2, n-1).
        RTEdw_inner = RTEdw_stacked[::-1]  # shape (n-1, npt)
        RTMdw_inner = RTMdw_stacked[::-1]

        # Monta saídas finais (npt, n): inserir terminal em [:, n-1]
        RTEdw = jnp.concatenate(
            [RTEdw_inner.T, jnp.zeros((npt, 1), dtype=jnp.complex128)], axis=1
        )
        RTMdw = jnp.concatenate(
            [RTMdw_inner.T, jnp.zeros((npt, 1), dtype=jnp.complex128)], axis=1
        )
    else:
        # n == 1: só terminais
        RTEdw = jnp.zeros((npt, 1), dtype=jnp.complex128)
        RTMdw = jnp.zeros((npt, 1), dtype=jnp.complex128)

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 3: Recursão top-down (camada 0 → n-1) via jax.lax.scan
    # ──────────────────────────────────────────────────────────────────
    def _top_down_step(carry, i):
        AdmApup_prev, ImpApup_prev = carry  # (npt,) — camada i-1
        AdmInt_i = AdmInt[:, i]
        ImpInt_i = ImpInt[:, i]
        tghuh_i = tghuh[:, i]
        tghsh_i = tghsh[:, i]

        AdmApup_i = (
            AdmInt_i
            * (AdmApup_prev + AdmInt_i * tghuh_i)
            / (AdmInt_i + AdmApup_prev * tghuh_i)
        )
        ImpApup_i = (
            ImpInt_i
            * (ImpApup_prev + ImpInt_i * tghsh_i)
            / (ImpInt_i + ImpApup_prev * tghsh_i)
        )
        RTEup_i = (AdmInt_i - AdmApup_prev) / (AdmInt_i + AdmApup_prev)
        RTMup_i = (ImpInt_i - ImpApup_prev) / (ImpInt_i + ImpApup_prev)

        return (AdmApup_i, ImpApup_i), (AdmApup_i, ImpApup_i, RTEup_i, RTMup_i)

    AdmApup_terminal = AdmInt[:, 0]
    ImpApup_terminal = ImpInt[:, 0]

    if n > 1:
        indices_top = jnp.arange(1, n)
        _, (_, _, RTEup_stacked, RTMup_stacked) = jax.lax.scan(
            _top_down_step,
            (AdmApup_terminal, ImpApup_terminal),
            indices_top,
        )
        # scan retorna na ordem (1, 2, ..., n-1) — ordem direta
        RTEup_inner = RTEup_stacked  # shape (n-1, npt)
        RTMup_inner = RTMup_stacked

        RTEup = jnp.concatenate(
            [jnp.zeros((npt, 1), dtype=jnp.complex128), RTEup_inner.T], axis=1
        )
        RTMup = jnp.concatenate(
            [jnp.zeros((npt, 1), dtype=jnp.complex128), RTMup_inner.T], axis=1
        )
    else:
        RTEup = jnp.zeros((npt, 1), dtype=jnp.complex128)
        RTMup = jnp.zeros((npt, 1), dtype=jnp.complex128)

    return (u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt)


# ──────────────────────────────────────────────────────────────────────────────
# common_factors_jax — port de _numba.common_factors
# ──────────────────────────────────────────────────────────────────────────────


def common_factors_jax(
    n: int,
    npt: int,
    h0: float,
    h: jax.Array,
    prof: jax.Array,
    camad_t: int,
    u: jax.Array,
    s: jax.Array,
    uh: jax.Array,
    sh: jax.Array,
    RTEdw: jax.Array,
    RTEup: jax.Array,
    RTMdw: jax.Array,
    RTMup: jax.Array,
) -> tuple[jax.Array, ...]:
    """Port JAX de :func:`_numba.propagation.common_factors`.

    Calcula os 6 fatores de onda (Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)
    usando as constantes pré-calculadas por :func:`common_arrays_jax`.

    Args:
        n, npt: Dimensões (estáticas para JIT).
        h0: Profundidade do transmissor (m).
        h: (n,) espessuras.
        prof: (n+1,) profundidades das interfaces (com sentinels ±1e300).
        camad_t: Índice 0-based da camada do transmissor.
        u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup: (npt, n) complex128 —
            saídas de :func:`common_arrays_jax`.

    Returns:
        Tupla ``(Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)`` cada um (npt,)
        complex128.

    Note:
        Implementação direta (sem recursão), port linha-a-linha do
        Fortran `utils.f08:243-297`. Ops XLA fundem com a chamada de
        :func:`common_arrays_jax` quando JIT aplicado upstream.
    """
    # Port line-for-line do _numba/propagation.py:555-609.
    top_t = prof[camad_t]
    bot_t = prof[camad_t + 1]
    h_t = h[camad_t]

    u_t = u[:, camad_t]
    s_t = s[:, camad_t]
    uh_t = uh[:, camad_t]
    sh_t = sh[:, camad_t]
    RTEdw_t = RTEdw[:, camad_t]
    RTEup_t = RTEup[:, camad_t]
    RTMdw_t = RTMdw[:, camad_t]
    RTMup_t = RTMup[:, camad_t]

    # BLOCO TM (Mxdw, Mxup) — usa s_t (constante TM)
    den_TM = 1.0 - RTMdw_t * RTMup_t * jnp.exp(-2.0 * sh_t)

    Mxdw = (
        jnp.exp(-s_t * (bot_t - h0)) + RTMup_t * jnp.exp(s_t * (top_t - h0 - h_t))
    ) / den_TM
    Mxup = (
        jnp.exp(s_t * (top_t - h0)) + RTMdw_t * jnp.exp(-s_t * (bot_t - h0 + h_t))
    ) / den_TM

    # BLOCO TE (Eudw, Euup) — usa u_t (constante TE), sinal - na 2ª parcela
    den_TE = 1.0 - RTEdw_t * RTEup_t * jnp.exp(-2.0 * uh_t)

    Eudw = (
        jnp.exp(-u_t * (bot_t - h0)) - RTEup_t * jnp.exp(u_t * (top_t - h0 - h_t))
    ) / den_TE
    Euup = (
        jnp.exp(u_t * (top_t - h0)) - RTEdw_t * jnp.exp(-u_t * (bot_t - h0 + h_t))
    ) / den_TE

    # BLOCO FE (FEdwz, FEupz) — sinal + nas 2ª parcelas (vs bloco E)
    FEdwz = (
        jnp.exp(-u_t * (bot_t - h0)) + RTEup_t * jnp.exp(u_t * (top_t - h_t - h0))
    ) / den_TE
    FEupz = (
        jnp.exp(u_t * (top_t - h0)) + RTEdw_t * jnp.exp(-u_t * (bot_t + h_t - h0))
    ) / den_TE

    return (Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)


__all__ = ["common_arrays_jax", "common_factors_jax"]
