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

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Imports pós-config: jax.config.update precisa preceder importações JAX/Numba
# para que enable_x64 seja honrado. ruff E402 silenciado intencionalmente.
from geosteering_ai.simulation._jax.propagation import (  # noqa: E402
    common_arrays_jax,
    common_factors_jax,
)
from geosteering_ai.simulation._jax.rotation import rotate_tensor  # noqa: E402
from geosteering_ai.simulation._numba.dipoles import hmd_tiv, vmd  # noqa: E402
from geosteering_ai.simulation._numba.geometry import (  # noqa: E402
    _sanitize_profile_kernel,
    find_layers_tr,
)

# Constante física — permeabilidade magnética do vácuo
_MU_0: float = 4.0e-7 * math.pi


def _to_writeable(arr) -> np.ndarray:
    """Converte para ``np.ndarray`` contíguo e writeable (Sprint v2.31, refinado v2.38).

    Arrays vindos de ``jax.pure_callback`` chegam com ``writeable=False``
    (JAX trata buffers como imutáveis). O Numba trata ``mutable=False``
    como tipo distinto de ``mutable=True``, gerando **especialização
    duplicada** dos kernels ``hmd_tiv``/``vmd`` no cache em disco
    (``.1.nbc`` + ``.2.nbc``), inflando o cold-start em 20-40 s.

    Esta helper força ``mutable=True`` via cópia explícita quando
    necessário, garantindo que apenas 1 especialização seja compilada
    independente do caminho de entrada (Numba puro vs JAX callback).

    **Uso seletivo (v2.38)**: aplicar APENAS em arrays que chegam writeable
    no caminho Numba direto (h_arr, prof_arr, eta, u, s, ...). Filtros
    de Hankel (``krJ0J1``, ``wJ0``, ``wJ1``) são ``setflags(write=False)``
    pelo ``FilterLoader`` — passe-os via :func:`_to_readonly_contig` para
    preservar ``readonly`` e convergir com a especialização Numba direta.

    Args:
        arr: Array-like (numpy, jax, ou qualquer convertível via
            ``np.asarray``).

    Returns:
        ``np.ndarray`` contíguo em C-order com ``flags.writeable=True``.
    """
    out = np.ascontiguousarray(arr)
    if not out.flags.writeable:
        out = out.copy()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Cache process-wide de filtros Hankel já transferidos para o device JAX
# ──────────────────────────────────────────────────────────────────────────────
# Sprint O1 (v2.43) — otimização #6 Quick Wins JAX GPU
#
# Motivação:
#   `FilterLoader._class_cache` (loader.py:305) já garante que o `.npz`
#   é lido do disco UMA VEZ por processo (cache classe-level com lock).
#   Porém, os callers JAX (`_jax/multi_forward.py`, `_jax/forward_pure.py`)
#   chamam `FilterLoader().load(...)` + `jnp.asarray(...)` em todo
#   `simulate_multi_jax`. O `np.asarray()` é praticamente no-op (filtros
#   já vêm contíguos float64), mas o `jnp.asarray()` força um host→device
#   transfer (10-100µs em GPU, ~5µs em CPU) e produz um novo `jax.Array`
#   handle a cada chamada, impedindo o XLA de reusar buffers entre invocations.
#
# Solução:
#   Cache LRU process-wide (`functools.lru_cache`) chaveado pelo nome
#   canônico do filtro. Retorna a tupla `(krJ0J1, wJ0, wJ1)` já como
#   `jax.Array` no device default. Subsequent calls com mesmo `filter_name`
#   retornam exatamente os mesmos handles, permitindo XLA bufferreuse.
#
# Thread-safety:
#   `functools.lru_cache` é thread-safe sob GIL do CPython (insert/get
#   atômico). `FilterLoader.load()` interno usa double-checked locking.
#   Em modo spawn (workers Numba), cada processo tem seu próprio cache —
#   correto e desejado (não há shared memory inter-processo para handles JAX).
#
# Cache invalidation:
#   Cache chaveado por nome canônico (str). Se caller passar
#   `'werthmuller_201pt'` vs `'kong_61pt'` vs `'anderson_801pt'`, cada
#   um terá entrada distinta no LRU (maxsize=8 > 3 filtros catalogados).
#   Aliases (e.g. `'wer'`, `'0'`) são resolvidos para canônico ANTES de
#   chavear, evitando entradas duplicadas no LRU para o mesmo filtro.


@functools.lru_cache(maxsize=8)
def _get_hankel_filter_cached(
    filter_name: str = "werthmuller_201pt",
) -> tuple["jax.Array", "jax.Array", "jax.Array"]:
    """Retorna ``(krJ0J1, wJ0, wJ1)`` como ``jax.Array`` via cache LRU.

    Carrega o filtro Hankel UMA VEZ por processo (e por nome canônico)
    e transfere os arrays para o device JAX default. Subsequent calls
    retornam os mesmos handles — permite ao XLA reusar buffers entre
    invocações de :func:`fields_in_freqs_jax_batch`.

    Args:
        filter_name: Nome canônico do filtro, alias amigável ou string
            numérica equivalente ao ``filter_type`` do Fortran.
            Aceita: ``'werthmuller_201pt'`` (default, ``filter_type=0``),
            ``'kong_61pt'`` (``=1``), ``'anderson_801pt'`` (``=2``), além
            de aliases como ``'wer'``, ``'kong'``, ``'and'`` e strings
            numéricas ``'0'``, ``'1'``, ``'2'``.

    Returns:
        Tupla ``(krJ0J1, wJ0, wJ1)`` — abscissas e pesos J0/J1 já como
        ``jax.Array`` float64 no device JAX default (CPU/GPU/TPU).

    Note:
        Helper introduzida no Sprint O1 v2.43 (Quick Wins JAX GPU,
        otimização #6). Use-a em hot paths para evitar repetir o
        ``jnp.asarray(...)`` em cada chamada do simulador. Os arrays
        retornados são imutáveis (``HankelFilter`` marca readonly no
        lado NumPy; ``jax.Array`` é semanticamente imutável).

        Cache invalidation: chaveado por nome canônico após
        :meth:`FilterLoader.resolve_name`. Se caller passar dois aliases
        do mesmo filtro, ambos compartilham a mesma entrada (resolvidos
        para o canônico antes de cachear).

    See Also:
        :class:`geosteering_ai.simulation.filters.FilterLoader`:
            cache classe-level de :class:`HankelFilter` (lado NumPy);
            esta helper adiciona a camada JAX-device por cima.
    """
    # Import local: evita custo de importar `filters` no import-time de
    # `_jax/kernel`, e quebra ciclo potencial caso filters.py importe
    # algo de _jax no futuro.
    from geosteering_ai.simulation.filters import FilterLoader

    loader = FilterLoader()
    # Resolve alias/numérico → canônico ANTES de o LRU chavear.
    # Isso garante que `load('wer')` e `load('werthmuller_201pt')`
    # compartilhem a mesma entrada do LRU (1 entrada, 1 transferência
    # para device — em vez de 2 entradas idênticas em conteúdo).
    # Nota: como `lru_cache` chaveia pelo argumento bruto, a resolução
    # já aconteceu antes de chamarmos esta helper se o caller passar
    # apenas nomes canônicos. Para garantir consistência, expomos um
    # wrapper público (`get_hankel_filter_jax`) que normaliza primeiro.
    filt = loader.load(filter_name)

    # `jnp.asarray` (não `jnp.array`): evita cópia desnecessária quando
    # o backend já consegue reusar o buffer NumPy (CPU). Em GPU, faz
    # host→device uma única vez por (processo, nome canônico).
    kr_jax = jnp.asarray(filt.abscissas, dtype=jnp.float64)
    w0_jax = jnp.asarray(filt.weights_j0, dtype=jnp.float64)
    w1_jax = jnp.asarray(filt.weights_j1, dtype=jnp.float64)
    return kr_jax, w0_jax, w1_jax


def get_hankel_filter_jax(
    filter_name: str = "werthmuller_201pt",
) -> tuple["jax.Array", "jax.Array", "jax.Array"]:
    """API pública: filtro Hankel como ``jax.Array`` (cache process-wide).

    Wrapper sobre :func:`_get_hankel_filter_cached` que normaliza aliases
    para o nome canônico ANTES de consultar o LRU. Isso garante que
    chamadas com aliases distintos do mesmo filtro (e.g. ``'wer'`` vs
    ``'werthmuller_201pt'`` vs ``'0'``) compartilhem a mesma entrada
    no cache LRU, evitando entradas duplicadas e transferências
    host→device redundantes em GPU.

    Args:
        filter_name: Nome canônico, alias ou string numérica do filtro.

    Returns:
        Tupla ``(krJ0J1, wJ0, wJ1)`` como ``jax.Array`` float64 no
        device JAX default. Mesmos handles em chamadas subsequentes
        com o mesmo nome canônico (após resolução de alias).

    Example:
        >>> kr, w0, w1 = get_hankel_filter_jax()                # default 201pt
        >>> kr2, w0_2, w1_2 = get_hankel_filter_jax("wer")     # alias
        >>> kr is kr2                                            # True — mesmo handle
        True
        >>> kr_k, _, _ = get_hankel_filter_jax("kong_61pt")    # filter_type=1
        >>> kr_k is kr                                           # False — filtro diferente
        False
    """
    from geosteering_ai.simulation.filters import FilterLoader

    canonical = FilterLoader().resolve_name(filter_name)
    return _get_hankel_filter_cached(canonical)


def _to_readonly_contig(arr) -> np.ndarray:
    """Converte para ``np.ndarray`` contíguo preservando ``readonly`` (v2.38).

    Complementa :func:`_to_writeable`. Usado para os filtros de Hankel
    (``krJ0J1``, ``wJ0``, ``wJ1``) que vêm do ``FilterLoader`` com
    ``setflags(write=False)`` no caminho Numba direto. Forçá-los para
    writeable no caminho JAX criaria uma 2ª especialização Numba
    (``readonly array`` vs ``array``) — exatamente o sintoma que o fix
    v2.31 pretendia eliminar.

    Args:
        arr: Array-like (numpy, jax, ou convertível).

    Returns:
        ``np.ndarray`` contíguo em C-order. ``writeable=False`` se o
        input já era readonly (ou após cópia, marcado readonly para
        casar com FilterLoader).
    """
    out = np.ascontiguousarray(arr)
    if out.flags.writeable:
        out = out.copy()
        out.setflags(write=False)
    return out


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
    # Sprint v2.31 (fix v2.38): convergir UMA especialização Numba.
    # Numba trata `readonly array` e `array` como tipos distintos. O caminho
    # Numba direto recebe filtros readonly (FilterLoader.setflags(write=False))
    # e demais arrays writeable (construídos localmente). O caminho JAX chega
    # com tudo readonly via jax.pure_callback. Para convergir, espelhamos
    # exatamente as mutabilidades do caminho direto:
    #   - filtros (krJ0J1, wJ0, wJ1): _to_readonly_contig (preserva readonly)
    #   - demais arrays: _to_writeable (força mutable=True)
    kr_ro = _to_readonly_contig(krJ0J1)
    w0_ro = _to_readonly_contig(wJ0)
    w1_ro = _to_readonly_contig(wJ1)
    u_np = _to_writeable(u)
    Hx_hmd, Hy_hmd, Hz_hmd = hmd_tiv(
        float(Tx),
        float(Ty),
        float(Tz),
        int(n_int),
        int(camad_r),
        int(camad_t),
        int(npt_int),
        kr_ro,
        w0_ro,
        w1_ro,
        _to_writeable(h_arr),
        _to_writeable(prof_arr),
        complex(zeta_c),
        _to_writeable(eta),
        float(cx),
        float(cy),
        float(cz),
        u_np,
        _to_writeable(s),
        _to_writeable(uh),
        _to_writeable(sh),
        _to_writeable(RTEdw),
        _to_writeable(RTEup),
        _to_writeable(RTMdw),
        _to_writeable(RTMup),
        _to_writeable(Mxdw),
        _to_writeable(Mxup),
        _to_writeable(Eudw),
        _to_writeable(Euup),
    )
    Hx_vmd, Hy_vmd, Hz_vmd = vmd(
        float(Tx),
        float(Ty),
        float(Tz),
        int(n_int),
        int(camad_r),
        int(camad_t),
        int(npt_int),
        kr_ro,
        w0_ro,
        w1_ro,
        _to_writeable(h_arr),
        _to_writeable(prof_arr),
        complex(zeta_c),
        float(cx),
        float(cy),
        float(cz),
        u_np,
        _to_writeable(uh),
        _to_writeable(u_np / complex(zeta_c)),  # AdmInt recalc
        _to_writeable(RTEdw),
        _to_writeable(RTEup),
        _to_writeable(FEdwz),
        _to_writeable(FEupz),
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
    use_native_dipoles: bool = False,
    use_unified: bool = False,
    complex_dtype=None,
) -> jax.Array:
    """Calcula H para uma frequência numa posição via backend JAX.

    Args:
        use_native_dipoles: Se True, usa ``native_dipoles_full_jax``
            (JAX puro end-to-end, diferenciável, GPU-ready). Se False
            (default), usa ``pure_callback`` → Numba host (bit-exato,
            mais rápido em CPU).
        use_unified: Sprint 10 Phase 2 (PR #24-part2). Se True, e
            ``use_native_dipoles=True``, usa ``native_dipoles_full_jax_unified``
            — propagação TE/TM via ``jax.lax.fori_loop`` aceitando
            ``camad_t``/``camad_r`` como tracers int32. Consolida 44 buckets
            XLA em 1 único JIT por ``(n, npt)``. Ignorado quando
            ``use_native_dipoles=False``.
        complex_dtype: Sprint O2 (v2.43) — dtype complexo. ``None`` →
            ``jnp.complex128`` (paridade Fortran sagrada). ``jnp.complex64``
            opt-in (2× menor footprint, GPU/PINN). Path híbrido (``pure_callback``)
            SEMPRE chama Numba em complex128 e faz cast pós-callback.

    Retorna array (9,) no ``complex_dtype`` solicitado — Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz.
    """
    # Sprint O2 (v2.43): default complex128. Override só altera os arrays
    # propagadores e o tensor H final.
    if complex_dtype is None:
        complex_dtype = jnp.complex128
    omega = 2.0 * jnp.pi * freq
    zeta = 1j * omega * _MU_0

    # Distância horizontal
    dx = cx - Tx
    dy = cy - Ty
    r = jnp.sqrt(dx * dx + dy * dy)

    # Sprint 3.2: propagação em JAX puro
    # Sprint O2: complex_dtype propagado para common_arrays_jax → arrays
    # u/s/uh/sh/RTE*/RTM*/AdmInt no dtype solicitado.
    outs = common_arrays_jax(
        n, npt, r, krJ0J1, zeta, h_arr, eta, complex_dtype=complex_dtype
    )
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

    # Sprint 3.3.4: escolhe entre JAX nativo end-to-end (diferenciável,
    # GPU-ready) ou callback Numba host (bit-exato, rápido em CPU).
    if use_native_dipoles:
        if use_unified:
            # Sprint 10 Phase 2: camad_t/camad_r como tracers → 1 JIT global.
            from geosteering_ai.simulation._jax.dipoles_native import (
                native_dipoles_full_jax_unified as _dipoles_impl,
            )
        else:
            # Sprint 3.3.4 legacy: camad_t/camad_r como Python ints (44 buckets).
            from geosteering_ai.simulation._jax.dipoles_native import (
                native_dipoles_full_jax as _dipoles_impl,
            )

        matH = _dipoles_impl(
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
    else:
        # Caminho híbrido (default) — callback para Numba @njit host
        # Sprint O2 (v2.43): Numba host SEMPRE retorna complex128 (paridade
        # Fortran sagrada). Cast pós-callback para complex_dtype se opt-in.
        # Notavelmente, as `u`, `s`, ... passadas para o callback foram
        # downcast para complex_dtype, mas o callback faz `_to_writeable` que
        # converte via `np.ascontiguousarray` → preserva dtype incoming.
        # Para mantermos paridade exata no path híbrido, deveríamos passar
        # versões complex128 ao callback. Solução escolhida: o caminho híbrido
        # com complex64 não é caso de uso suportado (use_native_dipoles=True
        # é o path complex64 esperado). Mantemos paridade c128 inalterada
        # neste branch, apenas castamos resultado a complex_dtype.
        result_shape = jax.ShapeDtypeStruct((3, 3), jnp.complex128)
        matH_c128 = jax.pure_callback(
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
            eta.astype(jnp.float64),
            cx,
            cy,
            cz,
            u.astype(jnp.complex128),
            s.astype(jnp.complex128),
            uh.astype(jnp.complex128),
            sh.astype(jnp.complex128),
            RTEdw.astype(jnp.complex128),
            RTEup.astype(jnp.complex128),
            RTMdw.astype(jnp.complex128),
            RTMup.astype(jnp.complex128),
            Mxdw.astype(jnp.complex128),
            Mxup.astype(jnp.complex128),
            Eudw.astype(jnp.complex128),
            Euup.astype(jnp.complex128),
            FEdwz.astype(jnp.complex128),
            FEupz.astype(jnp.complex128),
        )
        # Cast pós-callback (no-op se complex_dtype=complex128).
        matH = matH_c128.astype(complex_dtype)

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
    use_native_dipoles: bool = False,
) -> np.ndarray:
    """Forward JAX batch sobre posições do poço.

    API híbrida: pipeline JAX (propagação via ``common_arrays_jax`` +
    ``common_factors_jax``) + dipolos Numba via ``jax.pure_callback``.
    Estratégia default (``use_native_dipoles=False``) preserva bit-
    exatness vs Fortran em CPU e é a mais rápida para a maioria dos
    casos — o ``pure_callback`` reutiliza os ~900 LOC validados do port
    Numba.

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
        use_native_dipoles: Se True, usa ``native_dipoles_full_jax``
            (Sprint 3.3.4) — JAX puro end-to-end, diferenciável via
            ``jax.grad``, executável em GPU T4/A100 sem callback para
            CPU host. Se False (default), usa ``pure_callback`` →
            Numba @njit host (bit-exato, mais rápido em CPU puro).

    Returns:
        H_tensor shape (n_positions, nf, 9) complex128.

    Note:
        Com ``use_native_dipoles=True`` (Sprint 3.3.4 — PR #12):
          1. ETAPA 3: propagação TM/TE via Python loops (unrolled por JAX)
          2. ETAPA 5: kernels via ``lax.switch`` (Sprints 3.3.2/3.3.3)
          3. ETAPA 6: assembly Ward-Hohmann com ``jnp.sum``
          4. ``jax.grad`` / ``jax.jacfwd`` sobre ``rho_h`` / ``rho_v``
          5. GPU T4/A100 via XLA kernel fusion (sem callback CPU)

        O hybrid path (``use_native_dipoles=False``) permanece default
        e recomendado para produção em CPU — é mais rápido por evitar
        overhead de compilação JAX.
    """
    if use_native_dipoles:
        import logging

        _log = logging.getLogger(__name__)
        _log.info(
            "use_native_dipoles=True — usando JAX nativo end-to-end "
            "(Sprint 3.3.4). ETAPAS 3+5+6 sem pure_callback."
        )
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
        # Convenção Fortran (PerfilaAnisoOmp.f08:677-679): T abaixo, R acima.
        # Transmissor em +x/+z relativo ao ponto-médio; receptor em −x/−z.
        # Corrigido em v1.5.0 (PR #21) para bater com Numba path.
        Tz = z_mid + dz_half
        cz = z_mid - dz_half
        Tx = r_half
        cx = -r_half

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
                use_native_dipoles=use_native_dipoles,
            )
            H_tensor[j, i_f, :] = np.asarray(cH_9)

    return H_tensor


__all__ = ["fields_in_freqs_jax_batch", "get_hankel_filter_jax"]
