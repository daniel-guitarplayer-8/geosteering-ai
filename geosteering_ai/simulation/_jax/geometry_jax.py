# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/geometry_jax.py                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend JAX — Geometria tracer-safe     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-16 (Sprint 12 Phase 1 — PR #25 / v1.6.0)          ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : JAX 0.4.30+                                                ║
# ║  Dependências: jax, jax.numpy                                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Porta tracer-safe das funções geométricas NumPy/Numba que localizam    ║
# ║    camadas de transmissor/receptor em perfis 1D estratificados. A        ║
# ║    variante JAX (:func:`find_layers_tr_jax`) é requisito para:           ║
# ║      - `vmap` real sobre `(iTR, iAng)` em `simulate_multi_jax`          ║
# ║      - PINNs on-esp (Sprint 13+) onde `esp` vira parâmetro treinável   ║
# ║    Implementa-se via `jnp.searchsorted`, que é O(log n), diferenciável  ║
# ║    (não-suave em boundaries, mas definido via convenção jax) e vmappable. ║
# ║                                                                           ║
# ║  CONVENÇÃO DE IGUALDADE NA FRONTEIRA (paridade Fortran `utils.f08:63`)   ║
# ║    - Receptor: `z >= prof[i]` (inclusivo) → `searchsorted(side="right")`  ║
# ║      Se `z == prof[i]` exatamente, o receptor fica na camada ABAIXO.      ║
# ║    - Transmissor: `h0 > prof[j]` (estrito) → `searchsorted(side="left")`  ║
# ║      Se `h0 == prof[j]` exatamente, o transmissor fica na camada ACIMA.   ║
# ║    Esta assimetria é proposital e idêntica ao Fortran/Numba.              ║
# ║                                                                           ║
# ║  PARIDADE BIT-EXATA                                                        ║
# ║    Sweep de 1000+ pares `(h0, z)` contra `_numba/geometry.py::find_layers_tr` ║
# ║    retorna `diff == 0` (inteiros são idênticos, não apenas <1e-12).       ║
# ║    Ver `tests/test_simulation_jax_sprint12.py::test_find_layers_tr_jax_*`║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                               ║
# ║    • Plano Sprint 12: `/Users/daniel/.claude/plans/cosmic-riding-garden.md`║
# ║    • Numba equivalente: `_numba/geometry.py:211-294`                      ║
# ║    • Fortran origem: `Fortran_Gerador/utils.f08:45-87`                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Geometria tracer-safe em JAX — port de `_numba/geometry.py`.

Este módulo implementa :func:`find_layers_tr_jax`, substituta de
:func:`~geosteering_ai.simulation._numba.geometry.find_layers_tr` que aceita
argumentos como tracers JAX (necessário para `vmap` sobre `(iTR, iAng)` em
:func:`~geosteering_ai.simulation._jax.multi_forward._simulate_multi_jax_vmap_real`).

Example:
    Um único par (h0, z) com tracers::

        >>> import jax, jax.numpy as jnp
        >>> from geosteering_ai.simulation._jax.geometry_jax import find_layers_tr_jax
        >>> prof = jnp.array([-1e300, 0.0, 5.0, 1e300])
        >>> camad_t, camad_r = find_layers_tr_jax(2.5, 2.5, prof, n=3)
        >>> int(camad_t), int(camad_r)
        (1, 1)

    Batch over arrays de posições::

        >>> Tz = jnp.array([-1.0, 2.5, 7.0])
        >>> Rz = jnp.array([2.5, 2.5, 2.5])
        >>> ct, cr = find_layers_tr_jax_vmap(Tz, Rz, prof, 3)
        >>> ct.tolist(), cr.tolist()
        ([0, 1, 2], [1, 1, 1])
"""
from __future__ import annotations

from typing import Final, Tuple

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX: Final[bool] = True
except ImportError:
    _HAS_JAX: Final[bool] = False  # type: ignore[no-redef]


# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas e sentinelas
# ──────────────────────────────────────────────────────────────────────────────
# _INFINITY_PROF é a sentinela usada em `_numba.geometry.sanitize_profile`
# para prof[0] = -INF (topo) e prof[n] = +INF (fundo). Mantemos o mesmo valor
# 1e300 para paridade bit-exata em contextos onde `prof` já foi construído
# externamente (ex.: via `_numba.geometry.sanitize_profile`).
_INFINITY_PROF: Final[float] = 1.0e300


# ──────────────────────────────────────────────────────────────────────────────
# Função principal: find_layers_tr_jax
# ──────────────────────────────────────────────────────────────────────────────
def find_layers_tr_jax(
    h0,
    z,
    prof_array,
    n: int,
):
    """Localiza as camadas do transmissor e do receptor — tracer-safe.

    Port JAX tracer-compatível de
    :func:`geosteering_ai.simulation._numba.geometry.find_layers_tr`. Aceita
    `h0`, `z` e `prof_array` como tracers JAX, permitindo `vmap` e `jacfwd`.

    Args:
        h0: Profundidade do transmissor em metros. Scalar float ou tracer 0-D.
            Convenção: positivo para baixo; negativo indica TX no ar
            (semi-espaço superior).
        z: Profundidade do receptor em metros. Scalar float ou tracer 0-D.
        prof_array: Array `(n+1,)` float64 com boundaries acumuladas
            estritamente crescentes, produzido por
            :func:`_numba.geometry.sanitize_profile`. Tem sentinelas
            ``prof_array[0] = -1e300`` (topo) e ``prof_array[n] = +1e300``
            (fundo).
        n: Número total de camadas (incluindo semi-espaços). **Estático em
            JAX** (usado em `jnp.clip(..., 0, n-1)`).

    Returns:
        Tupla ``(camad_t, camad_r)``:

        - ``camad_t``: int32 tracer 0-D em ``[0, n-1]`` — camada do TX.
        - ``camad_r``: int32 tracer 0-D em ``[0, n-1]`` — camada do RX.

    Note:
        **Convenção assimétrica de fronteira** (paridade Fortran/Numba):

        - **Receptor**: ``z >= prof[i]`` inclusivo. Implementado via
          ``searchsorted(side="right")``, que coloca ``z`` exatamente
          igual a ``prof[i]`` na camada ``i`` (ABAIXO da fronteira).
        - **Transmissor**: ``h0 > prof[j]`` estrito. Implementado via
          ``searchsorted(side="left")``, que coloca ``h0`` exatamente
          igual a ``prof[j]`` na camada ``j-1`` (ACIMA da fronteira).

        Esta assimetria evita imprecisão em componentes não-tangenciais do
        tensor EM quando o RX ou TX está exatamente em uma interface (caso
        raro mas crítico para paridade bit-exata com Fortran).

        Complexidade: O(log n) por chamada — `searchsorted` é binário.
        Em T4 GPU com n=28, searchsorted custa ~10 ns por invocação, tornando
        o overhead desprezível vs o forward EM completo (~ms).

    Example:
        Perfil 3-camadas (topo + meio 5m + base)::

            >>> import jax.numpy as jnp
            >>> prof = jnp.array([-1e300, 0.0, 5.0, 1e300])
            >>> find_layers_tr_jax(-1.0, 2.5, prof, n=3)
            (Array(0, dtype=int32), Array(1, dtype=int32))
            >>> find_layers_tr_jax(2.5, 2.5, prof, n=3)
            (Array(1, dtype=int32), Array(1, dtype=int32))

        Batch via `find_layers_tr_jax_vmap`::

            >>> Tz = jnp.array([-1.0, 2.5, 7.0])
            >>> Rz = jnp.array([ 2.5, 2.5, 2.5])
            >>> ct, cr = find_layers_tr_jax_vmap(Tz, Rz, prof, 3)

    See Also:
        - :func:`geosteering_ai.simulation._numba.geometry.find_layers_tr`
          (implementação NumPy/Numba original)
        - :func:`find_layers_tr_jax_vmap` (versão vetorizada sobre posições)
    """
    if not _HAS_JAX:  # pragma: no cover
        raise ImportError(
            "find_layers_tr_jax requer JAX instalado. Use find_layers_tr do "
            "backend Numba (_numba.geometry) quando JAX não estiver disponível."
        )

    # ── Receptor: convenção inclusiva (z >= prof[i]) ─────────────────────
    # searchsorted(side="right") retorna a posição onde `z` seria inserido
    # *após* valores iguais, mantendo a ordem. Para prof = [-INF, 0, 5, INF]
    # e z=0, retorna 2 (depois do 0 em prof[1]). Subtraindo 1 → camada 1,
    # que corresponde a `z >= prof[1]` (0 >= 0) ✓.
    idx_r = jnp.clip(jnp.searchsorted(prof_array, z, side="right") - 1, 0, n - 1).astype(
        jnp.int32
    )

    # ── Transmissor: convenção estrita (h0 > prof[j]) ─────────────────────
    # searchsorted(side="left") retorna a posição onde `h0` seria inserido
    # *antes* de valores iguais. Para prof = [-INF, 0, 5, INF] e h0=0,
    # retorna 1 (antes do 0 em prof[1]). Subtraindo 1 → camada 0, que
    # corresponde ao fato de que 0 > 0 é falso (TX permanece na camada 0).
    idx_t = jnp.clip(jnp.searchsorted(prof_array, h0, side="left") - 1, 0, n - 1).astype(
        jnp.int32
    )

    return idx_t, idx_r


# ──────────────────────────────────────────────────────────────────────────────
# Versão vetorizada: find_layers_tr_jax_vmap
# ──────────────────────────────────────────────────────────────────────────────
if _HAS_JAX:
    # vmap sobre eixo 0 de h0 e z (posições), mantendo prof_array e n escalares.
    # Assinatura batch:
    #   find_layers_tr_jax_vmap(h0_arr, z_arr, prof_array, n)
    #     → (camad_t_arr, camad_r_arr)
    # com h0_arr.shape == z_arr.shape == (n_pos,)
    find_layers_tr_jax_vmap = jax.vmap(find_layers_tr_jax, in_axes=(0, 0, None, None))
else:  # pragma: no cover

    def find_layers_tr_jax_vmap(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("find_layers_tr_jax_vmap requer JAX instalado.")


# ──────────────────────────────────────────────────────────────────────────────
# Inventário público
# ──────────────────────────────────────────────────────────────────────────────
__all__: Final[list[str]] = [
    "find_layers_tr_jax",
    "find_layers_tr_jax_vmap",
]
