# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/forward_pure.py                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Forward JAX-puro diferenciável         ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13 (Sprint 5.1b — PR #14b)                        ║
# ║  Status      : Produção (experimental, CPU + GPU)                         ║
# ║  Framework   : JAX 0.4.38+                                                ║
# ║  Dependências: jax, jax.numpy, _jax/kernel.py, _jax/dipoles_native.py   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Forward JAX 100% puro (sem ``pure_callback``, sem ``np.asarray``)     ║
# ║    que aceita ``rho_h`` / ``rho_v`` como ``jnp.ndarray`` traceable e     ║
# ║    permite ``jax.jacfwd`` / ``jax.grad`` end-to-end nativo.              ║
# ║                                                                           ║
# ║    É construído em cima de ``_single_position_jax`` com                 ║
# ║    ``use_native_dipoles=True`` (Sprint 3.3.4) vetorizando sobre:        ║
# ║      • posições ``positions_z`` (n_positions,) via ``jax.vmap``        ║
# ║      • frequências ``freqs_hz`` (nf,) via ``jax.vmap`` aninhado        ║
# ║                                                                           ║
# ║  DIFERENCIAÇÃO                                                            ║
# ║    ``jax.jacfwd(forward_pure_jax, argnums=(0, 1))(rho_h, rho_v)``      ║
# ║    retorna ``(dH/drho_h, dH/drho_v)`` com shapes                        ║
# ║    ``(n_positions, nf, 9, n_layers)`` em ``complex128``.                ║
# ║                                                                           ║
# ║  DIFERENÇA VS ``fields_in_freqs_jax_batch``                              ║
# ║    ┌──────────────────────────┬──────────────┬──────────────────────┐  ║
# ║    │  Característica          │  JAX batch   │  forward_pure_jax    │  ║
# ║    ├──────────────────────────┼──────────────┼──────────────────────┤  ║
# ║    │  Loop Python sobre pos   │  sim         │  não (vmap)          │  ║
# ║    │  ``np.empty`` do output  │  sim         │  não (jnp.stack)     │  ║
# ║    │  ``rho_h`` traceable     │  não         │  sim                 │  ║
# ║    │  ``jax.jacfwd`` nativo   │  não         │  sim                 │  ║
# ║    │  GPU XLA fusion          │  parcial     │  total               │  ║
# ║    │  Default em produção     │  sim (CPU)   │  quando jacfwd ativo │  ║
# ║    └──────────────────────────┴──────────────┴──────────────────────┘  ║
# ║                                                                           ║
# ║  COMPATIBILIDADE                                                          ║
# ║    • CPU local (macOS Intel i9, Linux): funcional, rápido                ║
# ║    • GPU Colab (T4/A100): funcional via XLA (testado em notebook)        ║
# ║    • TPU: não testado (deveria funcionar via XLA também)                 ║
# ║                                                                           ║
# ║  RESTRIÇÃO IMPORTANTE                                                     ║
# ║    O caminho híbrido (``use_native_dipoles=False`` via                  ║
# ║    ``pure_callback``) **permanece intocado** em                         ║
# ║    ``fields_in_freqs_jax_batch``. Este módulo é **complementar**,       ║
# ║    não substituto. A seleção entre forward_pure_jax e o hybrid          ║
# ║    é decidida no dispatcher `compute_jacobian_jax` (Sprint 5.1).        ║
# ║                                                                           ║
# ║  CORRELAÇÃO COM CLAUDE.md                                                 ║
# ║    • Restrição 3 (JAX híbrido não removido): atendida — este módulo    ║
# ║      coexiste com o caminho híbrido.                                   ║
# ║    • Restrição 4 (alta ρ): testado com ρ=1500 Ω·m em               ║
# ║      ``test_simulation_jax_jacfwd_native.py``.                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Forward JAX puro diferenciável para ``jax.jacfwd`` end-to-end.

Este módulo expõe :func:`forward_pure_jax`, uma versão do forward 1D EM TIV
que usa apenas operações JAX (``jnp``, ``jax.vmap``, ``lax.switch``) sem
``pure_callback`` nem ``np.asarray`` dentro do trace. Isto permite aplicar
``jax.jacfwd`` / ``jax.grad`` diretamente em ``rho_h`` / ``rho_v``.

Example:
    Cálculo do Jacobiano ∂H/∂ρ via autodiff nativo::

        >>> import jax
        >>> import jax.numpy as jnp
        >>> jax.config.update("jax_enable_x64", True)
        >>> from geosteering_ai.simulation._jax.forward_pure import (
        ...     forward_pure_jax, build_static_context,
        ... )
        >>> import numpy as np
        >>> ctx = build_static_context(
        ...     rho_h=np.array([10.0, 100.0, 10.0]),
        ...     rho_v=np.array([10.0, 100.0, 10.0]),
        ...     esp=np.array([5.0]),
        ...     positions_z=np.linspace(-2.0, 7.0, 20),
        ...     freqs_hz=np.array([20000.0]),
        ...     tr_spacing_m=1.0,
        ...     dip_deg=0.0,
        ...     hankel_filter="werthmuller_201pt",
        ... )
        >>> H = forward_pure_jax(ctx.rho_h_jnp, ctx.rho_v_jnp, ctx)
        >>> H.shape  # (n_positions, nf, 9)
        (20, 1, 9)
        >>> J_h, J_v = jax.jacfwd(
        ...     lambda rh, rv: forward_pure_jax(rh, rv, ctx), argnums=(0, 1)
        ... )(ctx.rho_h_jnp, ctx.rho_v_jnp)
        >>> J_h.shape  # (n_pos, nf, 9, n_layers)
        (20, 1, 9, 3)

Note:
    O custo de vmap × vmap é dominado por ``native_dipoles_full_jax``
    (Sprint 3.3.4). Em CPU macOS Intel i9, forward_pure_jax é ~3-5× mais
    lento que o caminho Numba/JAX híbrido — mas **habilita jacfwd nativo
    e GPU XLA fusion**.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False
    jax = None  # type: ignore
    jnp = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# StaticContext — container de arrays "estáticos" (não diferenciados)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ForwardPureContext:
    """Contexto estático para ``forward_pure_jax``.

    Agrupa todos os arrays que são **constantes** em relação ao cálculo do
    Jacobiano (``rho_h``, ``rho_v`` são os únicos diferenciados). Preparar
    o contexto fora do trace JAX evita recomputações caras dentro do
    ``vmap`` (especialmente o filtro Hankel).

    Attributes:
        rho_h_jnp: (n,) float64 — resistividade horizontal inicial (jnp).
        rho_v_jnp: (n,) float64 — resistividade vertical inicial (jnp).
        esp_np: (n-2,) float64 — espessuras (NumPy, não diferenciável).
        positions_z_jnp: (n_pos,) float64.
        freqs_hz_jnp: (nf,) float64.
        dz_half: L·cos(dip)/2 (m).
        r_half: L·sin(dip)/2 (m).
        dip_rad: dip em radianos.
        n: número de camadas (inteiro estático).
        npt: número de pontos do filtro Hankel.
        krJ0J1, wJ0, wJ1: filtro Hankel como jnp.ndarray.
        h_arr_jnp: (n,) — profundidades acumuladas (paridade Fortran).
        prof_arr_jnp: (n+1,) — interfaces incluindo ±1e300 sentinelas.
        camad_t_array: (n_pos,) int — camada do transmissor por posição.
        camad_r_array: (n_pos,) int — camada do receptor por posição.

    Note:
        Todos os campos jnp são **estáticos** no trace de ``jacfwd``;
        apenas ``rho_h`` / ``rho_v`` passados ao ``forward_pure_jax``
        são efetivamente diferenciados.
    """

    rho_h_jnp: "jax.Array"
    rho_v_jnp: "jax.Array"
    esp_np: np.ndarray
    positions_z_jnp: "jax.Array"
    freqs_hz_jnp: "jax.Array"
    dz_half: float
    r_half: float
    dip_rad: float
    n: int
    npt: int
    krJ0J1: "jax.Array"
    wJ0: "jax.Array"
    wJ1: "jax.Array"
    h_arr_jnp: "jax.Array"
    prof_arr_jnp: "jax.Array"
    camad_t_array: np.ndarray
    camad_r_array: np.ndarray


def build_static_context(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
    positions_z: np.ndarray,
    freqs_hz: np.ndarray,
    tr_spacing_m: float,
    dip_deg: float,
    hankel_filter: str = "werthmuller_201pt",
) -> ForwardPureContext:
    """Pré-computa arrays estáticos para ``forward_pure_jax``.

    Args:
        rho_h: (n,) Ω·m — resistividade horizontal.
        rho_v: (n,) Ω·m — resistividade vertical.
        esp: (n-2,) m — espessuras internas.
        positions_z: (n_pos,) m — profundidades ponto-médio.
        freqs_hz: (nf,) Hz.
        tr_spacing_m: distância transmissor-receptor (m).
        dip_deg: ângulo de inclinação do poço (graus).
        hankel_filter: nome do filtro Hankel
            (``"werthmuller_201pt"``, ``"kong_61pt"``, ``"anderson_801pt"``).

    Returns:
        :class:`ForwardPureContext` imutável pronto para consumo por
        :func:`forward_pure_jax`.

    Raises:
        ImportError: Se JAX não estiver instalado.
        ValueError: Se ``rho_h.shape != rho_v.shape`` ou
            ``esp.shape[0] != n - 2``.
    """
    if not HAS_JAX:
        raise ImportError("forward_pure_jax requer JAX (pip install 'jax[cpu]').")
    from geosteering_ai.simulation._jax.kernel import (  # noqa: WPS433
        _sanitize_profile_kernel,
    )
    from geosteering_ai.simulation._numba.geometry import (  # noqa: WPS433
        find_layers_tr,
    )
    from geosteering_ai.simulation.filters import FilterLoader  # noqa: WPS433

    rho_h = np.asarray(rho_h, dtype=np.float64)
    rho_v = np.asarray(rho_v, dtype=np.float64)
    esp = np.asarray(esp, dtype=np.float64)
    positions_z = np.asarray(positions_z, dtype=np.float64)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

    n = rho_h.shape[0]
    if rho_v.shape[0] != n:
        raise ValueError(f"rho_v.shape={rho_v.shape} != rho_h.shape={rho_h.shape}")
    if n >= 2 and esp.shape[0] != n - 2:
        raise ValueError(f"esp.shape[0]={esp.shape[0]} != n-2={n-2}")

    # Filtro Hankel
    filt = FilterLoader().load(hankel_filter)
    krJ0J1 = np.asarray(filt.abscissas, dtype=np.float64)
    wJ0 = np.asarray(filt.weights_j0, dtype=np.float64)
    wJ1 = np.asarray(filt.weights_j1, dtype=np.float64)
    npt = krJ0J1.shape[0]

    # Perfil geométrico
    if n == 1:
        h_arr = np.zeros(1, dtype=np.float64)
        prof_arr = np.array([-1.0e300, 1.0e300], dtype=np.float64)
    else:
        h_arr, prof_arr = _sanitize_profile_kernel(n, esp)

    # Geometria TR
    dip_rad = float(np.deg2rad(dip_deg))
    L = float(tr_spacing_m)
    dz_half = 0.5 * L * np.cos(dip_rad)
    r_half = 0.5 * L * np.sin(dip_rad)

    # Determinação de camad_t/camad_r por posição (determinístico — não
    # diferenciável em relação a esp/rho). Isto evita ``jnp.searchsorted``
    # no trace e mantém ``forward_pure_jax`` livre de ramos sobre rho.
    n_pos = positions_z.shape[0]
    camad_t_array = np.empty(n_pos, dtype=np.int32)
    camad_r_array = np.empty(n_pos, dtype=np.int32)
    if n == 1:
        camad_t_array[:] = 0
        camad_r_array[:] = 0
    else:
        for i, z_mid in enumerate(positions_z):
            Tz = float(z_mid) - dz_half
            cz = float(z_mid) + dz_half
            ct, cr = find_layers_tr(n, Tz, cz, prof_arr)
            camad_t_array[i] = ct
            camad_r_array[i] = cr

    return ForwardPureContext(
        rho_h_jnp=jnp.asarray(rho_h, dtype=jnp.float64),
        rho_v_jnp=jnp.asarray(rho_v, dtype=jnp.float64),
        esp_np=esp,
        positions_z_jnp=jnp.asarray(positions_z, dtype=jnp.float64),
        freqs_hz_jnp=jnp.asarray(freqs_hz, dtype=jnp.float64),
        dz_half=dz_half,
        r_half=r_half,
        dip_rad=dip_rad,
        n=n,
        npt=npt,
        krJ0J1=jnp.asarray(krJ0J1),
        wJ0=jnp.asarray(wJ0),
        wJ1=jnp.asarray(wJ1),
        h_arr_jnp=jnp.asarray(h_arr),
        prof_arr_jnp=jnp.asarray(prof_arr),
        camad_t_array=camad_t_array,
        camad_r_array=camad_r_array,
    )


# ──────────────────────────────────────────────────────────────────────────────
# forward_pure_jax — principal API diferenciável
# ──────────────────────────────────────────────────────────────────────────────


def forward_pure_jax(
    rho_h: "jax.Array",
    rho_v: "jax.Array",
    ctx: ForwardPureContext,
) -> "jax.Array":
    """Forward JAX puro — ``jax.jacfwd(...)(rho_h, rho_v)`` funciona.

    Args:
        rho_h: (n,) float64 — resistividades horizontais (traceable).
        rho_v: (n,) float64 — resistividades verticais (traceable).
        ctx: Contexto estático pré-computado via :func:`build_static_context`.

    Returns:
        H_tensor shape ``(n_positions, nf, 9)`` complex128.

    Note:
        A função **reconstrói ``eta``** internamente via ``jnp.stack`` em
        vez de ``np.empty``, preservando a dependência traceable sobre
        ``rho_h``/``rho_v``. Isto habilita ``jax.jacfwd`` end-to-end.

        Performance em CPU macOS Intel i9: ~3-5× mais lento que o
        caminho JAX híbrido (``fields_in_freqs_jax_batch`` via
        ``pure_callback``). Em GPU Colab T4 espera-se ≥5× speedup em
        batch grandes (n_pos≥300, n_layers≥7).
    """
    # ──────────────────────────────────────────────────────────────────────
    # Sprint 7.x (PR #14d): orquestrador com bucketing por (camad_t, camad_r)
    # ──────────────────────────────────────────────────────────────────────
    # Estratégia:
    #   1. Agrupa posições em buckets geométricos (camad_t, camad_r) únicos —
    #      feito em NumPy fora do trace JAX.
    #   2. Para cada bucket, invoca `_forward_bucket_jit(ct, cr)` que faz
    #      `jax.vmap(_single_pos_one_freq, in_axes=(0, None))` sobre as
    #      posições do bucket, depois `vmap` sobre frequências.
    #   3. `_forward_bucket_jit` usa `functools.lru_cache` por `(ct, cr, n, npt)`
    #      para evitar retrace; cada compilação XLA é reutilizada enquanto a
    #      geometria macro permanecer a mesma.
    #
    # Ganhos:
    #   • JIT cache persistente → 30-100× vs baseline (sem @jit externo).
    #   • vmap fusa 20-600 posições em um único kernel XLA.
    #   • Bucketing mantém `camad_t, camad_r` estáticos dentro do JIT (evita
    #     problemas com tracers em branches que usam `for j in range(ct, cr+1)`
    #     dentro de `_hmd_tiv_full_jax` / `_vmd_full_jax`).
    #
    # Paridade: bit-a-bit vs versão anterior (mesma função `_single_position_jax`).
    return _forward_pure_jax_bucketed_impl(rho_h, rho_v, ctx)


# ──────────────────────────────────────────────────────────────────────────────
# Implementação interna — bucketing + jit + vmap (Sprint 7.x — PR #14d)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Cache LRU bounded — Sprint 7.x+ (PR #14e) — controla VRAM no GPU
# ──────────────────────────────────────────────────────────────────────────────
#
# MOTIVAÇÃO
#   Oklahoma_28 gera 44 buckets distintos (camad_t, camad_r), cada um
#   compilando um XLA program separado. Sem bound, o cache cresce até vazar
#   VRAM (12 GB em T4, 60 GB em A100 — observados em execução real 2026-04-14).
#
# SOLUÇÃO
#   • `OrderedDict` com `maxsize` → evict LRU quando cheio.
#   • `clear_jit_cache()` pública → usuário limpa antes de novo batch de modelos.
#   • `get_jit_cache_info()` → monitoramento diagnóstico.
#
# OBSERVAÇÃO
#   Para um único modelo geológico, cada bucket é compilado 1× e reutilizado
#   em todas as chamadas subsequentes (`forward_pure_jax` ou `jax.jacfwd`).
#   Para N modelos distintos em sequência, N×n_buckets compilações acumulam.
#   O LRU limita ao `maxsize` mais recentes — suficiente para um único modelo
#   com até ~28 camadas, e evita OOM no treino on-the-fly.
from collections import OrderedDict

_BUCKET_JIT_CACHE_MAXSIZE: int = 64
"""Tamanho máximo do cache LRU por processo. 64 cobre modelos até ~28 camadas.

Configurável via :func:`set_jit_cache_maxsize`. Valores maiores reduzem
recompilações em sequências longas; menores liberam VRAM agressivamente.
"""

_BUCKET_JIT_CACHE: "OrderedDict[tuple, object]" = OrderedDict()


def set_jit_cache_maxsize(maxsize: int) -> None:
    """Ajusta o tamanho máximo do cache LRU de JIT-compilações por bucket.

    Args:
        maxsize: Número máximo de XLA programs compilados simultaneamente.
            Cada entrada consome ~10-100 MB de VRAM em GPU T4. Para A100 com
            80 GB, valor pode ser 128+. Para T4 com 16 GB, recomendado ≤ 32.

    Note:
        A chamada NÃO invalida entradas existentes. Use :func:`clear_jit_cache`
        para invalidação explícita. Em caso de `maxsize < len(cache)`,
        evicções LRU acontecerão na próxima inserção.
    """
    global _BUCKET_JIT_CACHE_MAXSIZE
    if maxsize < 1:
        raise ValueError(f"maxsize deve ser >= 1 (recebido {maxsize})")
    _BUCKET_JIT_CACHE_MAXSIZE = int(maxsize)


def clear_jit_cache() -> int:
    """Limpa o cache de JIT-compilações por bucket.

    Retorna o número de entradas removidas. Chame antes de mudar para um
    novo modelo geológico para liberar VRAM.

    Example:
        >>> H = forward_pure_jax(rh, rv, ctx1)          # popula cache
        >>> n_removed = clear_jit_cache()               # libera VRAM
        >>> H = forward_pure_jax(rh, rv, ctx2)          # recompila para ctx2
    """
    n = len(_BUCKET_JIT_CACHE)
    _BUCKET_JIT_CACHE.clear()
    return n


def get_jit_cache_info() -> dict:
    """Retorna dicionário com estatísticas do cache JIT.

    Returns:
        dict com chaves:

        - ``n_entries`` (int): buckets compilados atualmente.
        - ``maxsize`` (int): limite antes de eviction.
        - ``keys`` (list): chaves ``(ct, cr, n, npt)`` das entradas presentes.
    """
    return {
        "n_entries": len(_BUCKET_JIT_CACHE),
        "maxsize": _BUCKET_JIT_CACHE_MAXSIZE,
        "keys": list(_BUCKET_JIT_CACHE.keys()),
    }


def _get_bucket_jit(ct: int, cr: int, n: int, npt: int):
    """Retorna (e memoriza) função jitada para bucket (ct, cr, n, npt).

    A função aceita ``(rho_h, rho_v, z_bucket, freqs, ctx_arrays)`` e retorna
    ``H_bucket`` com shape ``(n_pos_bucket, nf, 9)``.

    O cache é ``OrderedDict`` com LRU bounded em
    :data:`_BUCKET_JIT_CACHE_MAXSIZE` — previne vazamento de VRAM em GPU.
    """
    from geosteering_ai.simulation._jax.kernel import _single_position_jax

    key = (int(ct), int(cr), int(n), int(npt))
    if key in _BUCKET_JIT_CACHE:
        # Move to end (LRU tracking: este é o MRU agora)
        _BUCKET_JIT_CACHE.move_to_end(key)
        return _BUCKET_JIT_CACHE[key]

    def _forward_bucket(
        rho_h: "jax.Array",
        rho_v: "jax.Array",
        z_bucket: "jax.Array",
        freqs: "jax.Array",
        dz_half: float,
        r_half: float,
        dip_rad: float,
        h_arr: "jax.Array",
        prof_arr: "jax.Array",
        krJ0J1: "jax.Array",
        wJ0: "jax.Array",
        wJ1: "jax.Array",
    ) -> "jax.Array":
        # Reconstrói eta traceable (diferenciável em rho_h/rho_v).
        eta = jnp.stack([1.0 / rho_h, 1.0 / rho_v], axis=-1)

        def _one_pos_one_freq(z_mid, freq):
            Tz = z_mid - dz_half
            cz = z_mid + dz_half
            return _single_position_jax(
                -r_half,
                0.0,
                Tz,
                r_half,
                0.0,
                cz,
                dip_rad,
                n,
                npt,
                ct,
                cr,  # estáticos — fecha sobre este bucket
                rho_h,
                rho_v,
                h_arr,
                prof_arr,
                eta,
                freq,
                krJ0J1,
                wJ0,
                wJ1,
                use_native_dipoles=True,
            )

        # vmap sobre frequências (in_axes=(None, 0)) → shape (nf, 9)
        vmap_freq = jax.vmap(_one_pos_one_freq, in_axes=(None, 0))
        # vmap sobre posições (in_axes=(0, None)) → shape (n_pos_b, nf, 9)
        vmap_both = jax.vmap(vmap_freq, in_axes=(0, None))
        return vmap_both(z_bucket, freqs)

    # Nota: `donate_argnums` foi considerado para reuso de buffer, mas
    # quebra chamadas repetidas com os mesmos `rho_h`/`rho_v` (e também
    # o fluxo de `jax.jacfwd`). Deixamos sem doação para manter
    # compatibilidade com o padrão de uso (warmup + múltiplas chamadas).
    jitted = jax.jit(_forward_bucket, static_argnames=())

    # LRU eviction — mantém cache limitado para evitar OOM em GPU.
    if len(_BUCKET_JIT_CACHE) >= _BUCKET_JIT_CACHE_MAXSIZE:
        _BUCKET_JIT_CACHE.popitem(last=False)  # evict oldest (LRU)
    _BUCKET_JIT_CACHE[key] = jitted
    return jitted


def _forward_pure_jax_bucketed_impl(
    rho_h: "jax.Array",
    rho_v: "jax.Array",
    ctx: ForwardPureContext,
) -> "jax.Array":
    """Implementação bucketed — agrupa posições por (camad_t, camad_r) única.

    Para modelos reais com 3–22 camadas e 100–600 posições, tipicamente há
    3–10 buckets distintos. Cada bucket é compilado uma vez via JAX JIT e
    reexecutado em vmap sobre posições e frequências.

    Args:
        rho_h, rho_v: tracers JAX (diferenciáveis).
        ctx: contexto estático pré-computado.

    Returns:
        Tensor ``(n_positions, nf, 9)`` em ordem original de ``positions_z``.
    """
    n_pos = ctx.positions_z_jnp.shape[0]
    nf = ctx.freqs_hz_jnp.shape[0]

    # Agrupa posições em buckets — feito em NumPy puro (fora do trace).
    ct_arr = np.asarray(ctx.camad_t_array, dtype=np.int32)
    cr_arr = np.asarray(ctx.camad_r_array, dtype=np.int32)
    key_arr = ct_arr.astype(np.int64) * 10_000 + cr_arr.astype(np.int64)
    unique_keys, inverse = np.unique(key_arr, return_inverse=True)

    # Shape de saída final (em ordem original) — usa jnp.empty + scatter.
    H_out = jnp.zeros((n_pos, nf, 9), dtype=jnp.complex128)
    for bucket_idx, key in enumerate(unique_keys):
        mask = inverse == bucket_idx
        indices = np.nonzero(mask)[0]
        ct = int(ct_arr[indices[0]])
        cr = int(cr_arr[indices[0]])
        z_bucket_np = ctx.positions_z_jnp[indices]  # jnp slice; OK fora do trace

        jitted_fn = _get_bucket_jit(ct, cr, ctx.n, ctx.npt)
        H_bucket = jitted_fn(
            rho_h,
            rho_v,
            z_bucket_np,
            ctx.freqs_hz_jnp,
            ctx.dz_half,
            ctx.r_half,
            ctx.dip_rad,
            ctx.h_arr_jnp,
            ctx.prof_arr_jnp,
            ctx.krJ0J1,
            ctx.wJ0,
            ctx.wJ1,
        )  # (n_pos_bucket, nf, 9)

        # Scatter dentro da saída na ordem original.
        H_out = H_out.at[jnp.asarray(indices)].set(H_bucket)

    return H_out


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 8 — warmup coletivo + chunking de posições (PR #14f)
# ──────────────────────────────────────────────────────────────────────────────
#
# OBJETIVO
#   Reduzir pico de VRAM em GPU (T4 ~11 GB, A100 ~60 GB observados) mesmo
#   quando `maxsize` do cache LRU está no valor padrão.
#
# ESTRATÉGIAS
#   1. `warmup_all_buckets(ctx, rho_h_ref, rho_v_ref)` — força compilação de
#      todos os buckets distintos **com arrays pequenos** (n_pos_bucket = 1),
#      amortizando o custo de JIT e estabilizando o cache XLA antes da
#      primeira chamada "real".
#
#   2. `forward_pure_jax_chunked(rho_h, rho_v, ctx, chunk_size=32)` —
#      processa `positions_z` em chunks de tamanho fixo, reduzindo o pico
#      de VRAM durante `vmap` (que materializa tensores `(n_pos_bucket,
#      nf, 9, ...)` em paralelo). Para oklahoma_28 com 100 posições em T4,
#      chunk_size=32 reduz pico em ~3×.
#
# TRADE-OFFS
#   • `warmup` aumenta tempo inicial (primeira chamada) mas TODAS as
#     subsequentes são rápidas. Ideal para treino on-the-fly em Colab.
#   • `chunked` é um pouco mais lento que o vmap monolítico em CPU (Numba)
#     mas evita OOM em GPU — especialmente T4 (16 GB) com modelos grandes.


def warmup_all_buckets(
    ctx: ForwardPureContext,
    rho_h_ref: Optional["jax.Array"] = None,
    rho_v_ref: Optional["jax.Array"] = None,
) -> int:
    """Força compilação XLA de todos os buckets `(camad_t, camad_r)` de ``ctx``.

    Útil em notebooks Colab para amortizar o custo de JIT **antes** da primeira
    medição cronometrada — a função retorna apenas após todas as compilações
    estarem prontas e todos os valores bloqueados em GPU.

    Args:
        ctx: Contexto estático pré-computado via :func:`build_static_context`.
        rho_h_ref: Array float64 de referência para a compilação (usa
            ``ctx.rho_h_jnp`` se ``None``). Os valores não afetam o resultado
            da compilação — somente shape e dtype.
        rho_v_ref: análogo para resistividade vertical.

    Returns:
        Número de buckets compilados (i.e., `len(unique_keys)` do ctx).

    Note:
        Como **cada bucket é um JIT cachado separado**, esta função usa
        internamente o mesmo ``_get_bucket_jit`` que ``forward_pure_jax``,
        garantindo que a compilação realizada aqui será reutilizada depois.
    """
    if rho_h_ref is None:
        rho_h_ref = ctx.rho_h_jnp
    if rho_v_ref is None:
        rho_v_ref = ctx.rho_v_jnp

    ct_arr = np.asarray(ctx.camad_t_array, dtype=np.int32)
    cr_arr = np.asarray(ctx.camad_r_array, dtype=np.int32)
    key_arr = ct_arr.astype(np.int64) * 10_000 + cr_arr.astype(np.int64)
    unique_keys, inverse = np.unique(key_arr, return_inverse=True)

    for bucket_idx in range(len(unique_keys)):
        mask = inverse == bucket_idx
        indices = np.nonzero(mask)[0]
        ct = int(ct_arr[indices[0]])
        cr = int(cr_arr[indices[0]])
        # IMPORTANTE: usa o shape EXATO que será usado em produção (todas as
        # posições deste bucket) — JAX compila por shape, então um warmup
        # com shape diferente não beneficia a chamada real.
        z_bucket = ctx.positions_z_jnp[indices]
        jitted_fn = _get_bucket_jit(ct, cr, ctx.n, ctx.npt)
        H_bucket_warm = jitted_fn(
            rho_h_ref,
            rho_v_ref,
            z_bucket,
            ctx.freqs_hz_jnp,
            ctx.dz_half,
            ctx.r_half,
            ctx.dip_rad,
            ctx.h_arr_jnp,
            ctx.prof_arr_jnp,
            ctx.krJ0J1,
            ctx.wJ0,
            ctx.wJ1,
        )
        H_bucket_warm.block_until_ready()
    return len(unique_keys)


def forward_pure_jax_chunked(
    rho_h: "jax.Array",
    rho_v: "jax.Array",
    ctx: ForwardPureContext,
    chunk_size: int = 32,
) -> "jax.Array":
    """Versão com chunking de posições — reduz pico de VRAM em GPU.

    Processa ``ctx.positions_z_jnp`` em lotes de tamanho ``chunk_size``,
    reaproveitando os mesmos JIT-compilados por bucket. O custo total de
    computação é equivalente, mas o **pico de VRAM** diminui em ~n_chunks ×.

    Args:
        rho_h, rho_v: tracers JAX (diferenciáveis).
        ctx: contexto estático pré-computado.
        chunk_size: número de posições por chunk. Default 32 (equilíbrio
            entre overhead de Python e pico de memória). Para GPU T4 com
            modelos 28+ camadas, recomenda-se ``chunk_size=16``.

    Returns:
        Tensor ``(n_positions, nf, 9)`` idêntico ao ``forward_pure_jax``.

    Note:
        Quando ``n_positions <= chunk_size``, a função é equivalente a
        ``forward_pure_jax`` (1 único chunk).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size deve ser >= 1 (recebido {chunk_size})")

    n_pos = ctx.positions_z_jnp.shape[0]
    nf = ctx.freqs_hz_jnp.shape[0]

    if n_pos <= chunk_size:
        return _forward_pure_jax_bucketed_impl(rho_h, rho_v, ctx)

    # Divide posições em chunks e aplica o bucketing em cada — buffer
    # acumula o resultado final na ordem original.
    H_out = jnp.zeros((n_pos, nf, 9), dtype=jnp.complex128)
    ct_full = np.asarray(ctx.camad_t_array, dtype=np.int32)
    cr_full = np.asarray(ctx.camad_r_array, dtype=np.int32)
    for start in range(0, n_pos, chunk_size):
        end = min(start + chunk_size, n_pos)
        # Constrói sub-contexto com fatias dos arrays por-posição (os
        # arrays estáticos do filtro Hankel são compartilhados).
        sub_ctx = ForwardPureContext(
            rho_h_jnp=ctx.rho_h_jnp,
            rho_v_jnp=ctx.rho_v_jnp,
            esp_np=ctx.esp_np,
            positions_z_jnp=ctx.positions_z_jnp[start:end],
            freqs_hz_jnp=ctx.freqs_hz_jnp,
            dz_half=ctx.dz_half,
            r_half=ctx.r_half,
            dip_rad=ctx.dip_rad,
            n=ctx.n,
            npt=ctx.npt,
            krJ0J1=ctx.krJ0J1,
            wJ0=ctx.wJ0,
            wJ1=ctx.wJ1,
            h_arr_jnp=ctx.h_arr_jnp,
            prof_arr_jnp=ctx.prof_arr_jnp,
            camad_t_array=ct_full[start:end],
            camad_r_array=cr_full[start:end],
        )
        H_chunk = _forward_pure_jax_bucketed_impl(rho_h, rho_v, sub_ctx)
        H_out = H_out.at[start:end].set(H_chunk)
    return H_out


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 9 — pmap multi-GPU para batches de modelos (PR #14f)
# ──────────────────────────────────────────────────────────────────────────────


def forward_pure_jax_pmap(
    rho_h_batch: "jax.Array",
    rho_v_batch: "jax.Array",
    ctx: ForwardPureContext,
) -> "jax.Array":
    """Executa ``forward_pure_jax`` sobre um batch de modelos via ``jax.pmap``.

    Distribui ``n_models`` modelos sobre os dispositivos disponíveis (``GPU``
    ou ``CPU`` lógico). Requer ``rho_h_batch.shape[0] == n_devices``.

    Args:
        rho_h_batch: Array shape ``(n_devices, n_layers)``.
        rho_v_batch: Array shape ``(n_devices, n_layers)``.
        ctx: contexto estático compartilhado (mesma geometria de poço).

    Returns:
        Tensor ``(n_devices, n_positions, nf, 9)`` — um forward por modelo.

    Raises:
        ValueError: Se ``rho_h_batch.shape[0]`` != número de dispositivos.
        ImportError: Se JAX não instalado.

    Note:
        Para executar em hardware com múltiplas GPUs (A100 × 4 em Colab
        Pro+), defina o env var ``JAX_PLATFORMS=gpu`` e aloque o runtime
        apropriado. Em hardware mono-GPU ou CPU, ``n_devices=1`` e a
        chamada reduz-se a ``jax.vmap`` sobre batch.

        Os modelos compartilham o mesmo ``ctx`` (geometria do poço);
        variações de resistividade são o único parâmetro diferenciado.
    """
    if not HAS_JAX:
        raise ImportError("forward_pure_jax_pmap requer JAX instalado")

    n_devices = jax.local_device_count()
    if rho_h_batch.shape[0] != n_devices:
        raise ValueError(
            f"rho_h_batch.shape[0]={rho_h_batch.shape[0]} != "
            f"n_devices={n_devices}. Use rho_h_batch com shape "
            f"({n_devices}, n_layers) ou ajuste via "
            "XLA_FLAGS='--xla_force_host_platform_device_count={N}'."
        )

    def _single_model(rh, rv):
        return forward_pure_jax(rh, rv, ctx)

    pmapped = jax.pmap(_single_model, axis_name="devices")
    return pmapped(rho_h_batch, rho_v_batch)


__all__ = [
    "HAS_JAX",
    "ForwardPureContext",
    "build_static_context",
    "clear_jit_cache",
    "forward_pure_jax",
    "forward_pure_jax_chunked",
    "forward_pure_jax_pmap",
    "get_jit_cache_info",
    "set_jit_cache_maxsize",
    "warmup_all_buckets",
]
