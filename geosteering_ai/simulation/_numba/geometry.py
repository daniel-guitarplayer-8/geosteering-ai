# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_numba/geometry.py                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Geometria de Camadas (Sprint 2.3)       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x + Numba 0.60+ (dual-mode, opcional)              ║
# ║  Dependências: numpy (obrigatório), numba (opcional, speedup JIT)         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Porta para Python+Numba as rotinas Fortran de determinação da         ║
# ║    geometria de camadas a partir do perfil geológico + posição           ║
# ║    transmissor/receptor:                                                  ║
# ║                                                                           ║
# ║      • sanitize_profile    — constrói arrays `h` (espessuras) e          ║
# ║                              `prof` (interfaces) a partir de `esp`.      ║
# ║      • find_layers_tr      — dada uma posição `(h0, z)`, retorna os     ║
# ║                              índices das camadas onde estão TX e RX.    ║
# ║      • layer_at_depth      — dada profundidade `z`, retorna camada.     ║
# ║                                                                           ║
# ║    Equivalentes Fortran:                                                 ║
# ║      • `sanitize_hprof_well` (utils.f08:5-43)                           ║
# ║      • `findlayersTR2well`   (utils.f08:45-87)                          ║
# ║      • `layer2z_inwell`      (utils.f08:299-319)                        ║
# ║                                                                           ║
# ║  ESQUEMA DE INDEXAÇÃO 0-BASED                                             ║
# ║    Python usa índices 0-based; o Fortran usa 1-based. A tradução é       ║
# ║    literal via subtração de 1 nos limites dos loops.                    ║
# ║                                                                           ║
# ║    prof[0] = topo da camada 0 (geralmente 0.0)                           ║
# ║    prof[i+1] = fundo da camada i = topo da camada i+1                   ║
# ║    prof[n] = 1e300 (infinito, semi-espaço inferior)                      ║
# ║                                                                           ║
# ║  DIAGRAMA VISUAL                                                          ║
# ║    ┌────────────────────────────────────────────────┐                   ║
# ║    │  Camada 0 (semi-espaço superior)              │                   ║
# ║    │  interface 0 -------- prof[0] = 0.0           │                   ║
# ║    │  Camada 1                h = esp[0]           │                   ║
# ║    │  interface 1 -------- prof[1] = h[1]          │                   ║
# ║    │  ...                                          │                   ║
# ║    │  Camada n-2              h = esp[n-3]         │                   ║
# ║    │  interface n-2 ------ prof[n-1]               │                   ║
# ║    │  Camada n-1 (semi-espaço inferior)            │                   ║
# ║    │  prof[n] = 1e300                              │                   ║
# ║    └────────────────────────────────────────────────┘                   ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/utils.f08 (findlayersTR2well, layer2z_inwell)      ║
# ║    • Fortran_Gerador/PerfilaAnisoOmp.f08 (uso em fieldsinfreqs)         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Geometria de camadas do simulador Python (Sprint 2.3).

Módulo com 3 funções utilitárias para determinação da posição do
transmissor/receptor no perfil estratificado:

- :func:`sanitize_profile` — monta `h` (espessuras com padding nos
  semi-espaços) e `prof` (profundidades das interfaces, com sentinels).
- :func:`find_layers_tr` — dada `(h0, z)`, retorna `(camad_t, camad_r)`
  em indexação 0-based.
- :func:`layer_at_depth` — dada profundidade `z`, retorna índice da
  camada (0-based).

Example:
    Perfil 3-camadas (topo + meio + base)::

        >>> import numpy as np
        >>> from geosteering_ai.simulation._numba.geometry import (
        ...     sanitize_profile, find_layers_tr,
        ... )
        >>> esp = np.array([5.0])  # 1 camada interna de 5 m
        >>> h, prof = sanitize_profile(n=3, esp=esp)
        >>> h   # shape (3,)
        array([0., 5., 0.])
        >>> prof  # shape (4,), prof[3]=1e300
        array([0.e+00, 0.e+00, 5.e+00, 1.e+300])
        >>> find_layers_tr(n=3, h0=2.5, z=2.5, prof=prof)
        (1, 1)

Note:
    A convenção 1-based do Fortran é substituída por 0-based em Python.
    Comparações como `z >= prof(n-1)` (Fortran) tornam-se
    `z >= prof[n-1]` (Python, usando profs sem sentinel final), onde
    `prof` aqui tem shape `(n,)` no estilo Fortran sem `prof(0)=0`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from geosteering_ai.simulation._numba.propagation import njit

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
# Valor "infinito" usado para representar o semi-espaço inferior.
# Paridade com Fortran `1.d300` (utils.f08:41). Este valor é grande o
# suficiente para qualquer profundidade geológica realista mas ainda
# finito para cálculos de exponencial.
_INFINITY_PROF: float = 1.0e300


def sanitize_profile(
    n: int,
    esp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constrói arrays `h` e `prof` a partir de espessuras de camadas.

    Port Python de `sanitize_hprof_well` (Fortran `utils.f08:5-43`).
    Monta os arrays que descrevem a geometria do perfil estratificado
    no formato esperado pelos kernels do simulador:

    - ``h[i]``      — espessura da camada `i` em metros. Camadas de
      semi-espaço (topo `i=0` e base `i=n-1`) têm `h=0.0` (convencional).
    - ``prof[i]``   — profundidade da interface `i` (fundo da camada
      `i-1` e topo da camada `i`). `prof[0] = 0.0` (topo do modelo)
      e `prof[n] = 1e300` (infinito, fundo do semi-espaço).

    Args:
        n: Número total de camadas (≥ 2). Inclui os dois semi-espaços
            infinitos (topo e base).
        esp: Array 1D `float64` com as espessuras das camadas internas.
            Pode ter tamanho `n-2` (apenas internas) ou `n` (inclui
            dummies nos semi-espaços — paridade com caller Fortran).

    Returns:
        Tupla `(h, prof)`:

        - ``h``: `ndarray(n,) float64` — espessuras. `h[0]=h[n-1]=0.0`.
        - ``prof``: `ndarray(n+1,) float64` — profundidades das
          interfaces. `prof[0]=0.0`, `prof[n]=1e300`.

    Raises:
        ValueError: Se `n < 2` ou se `esp` tiver shape inválido.

    Note:
        Esta função **não é decorada com @njit** porque trabalha com
        shapes dinâmicos de `esp` que variam entre chamadas. Overhead
        é desprezível (O(n) operações puras em n ≤ ~30 camadas típicas).

    Example:
        Perfil de 5 camadas com 3 camadas internas::

            >>> esp = np.array([1.5, 2.0, 1.0])  # 3 internas
            >>> h, prof = sanitize_profile(n=5, esp=esp)
            >>> h
            array([0. , 1.5, 2. , 1. , 0. ])
            >>> prof
            array([0.0e+000, 0.0e+000, 1.5e+000, 3.5e+000, 4.5e+000, 1.0e+300])
    """
    if n < 2:
        raise ValueError(
            f"n={n} inválido. Deve haver pelo menos 2 camadas "
            f"(topo e base — semi-espaços infinitos)."
        )
    esp = np.asarray(esp, dtype=np.float64)
    if esp.ndim != 1:
        raise ValueError(f"esp deve ser 1D, recebido shape {esp.shape}.")

    # Validação de shape (específica em Python por f-string contextual).
    if esp.shape[0] != n - 2 and esp.shape[0] != n:
        raise ValueError(
            f"esp.shape={esp.shape} inválido. Esperado ({n - 2},) ou " f"({n},)."
        )

    # Sprint 2.9: delega para kernel @njit puro.
    return _sanitize_profile_kernel(n, esp)


# ──────────────────────────────────────────────────────────────────────────────
# _sanitize_profile_kernel — versão @njit para chamada de kernels (Sprint 2.9)
# ──────────────────────────────────────────────────────────────────────────────
# Idêntico a sanitize_profile, mas decorado com @njit e sem raise de
# validações com f-strings. Assume shape já validado pelo wrapper.


@njit(cache=True)
def _sanitize_profile_kernel(
    n: int,
    esp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kernel @njit de sanitize_profile.

    Idêntico a :func:`sanitize_profile` sem validação. Assume
    ``esp.shape == (n-2,)`` ou ``(n,)`` já verificado pelo wrapper.
    """
    h = np.zeros(n, dtype=np.float64)
    # Espessuras internas vão em h[1:n-1]
    if esp.shape[0] == n - 2:
        if n > 2:
            h[1 : n - 1] = esp
    else:  # esp.shape[0] == n
        h[1 : n - 1] = esp[1 : n - 1]

    # Constrói prof: shape (n+1,) — sentinel topo/fundo + acumulado
    prof = np.empty(n + 1, dtype=np.float64)
    prof[0] = -_INFINITY_PROF
    cumulative = 0.0
    for i in range(n):
        cumulative += h[i]
        prof[i + 1] = cumulative
    prof[n] = _INFINITY_PROF

    return h, prof


@njit
def find_layers_tr(
    n: int,
    h0: float,
    z: float,
    prof: np.ndarray,
) -> Tuple[int, int]:
    """Determina as camadas do transmissor e do receptor.

    Port Python+Numba de `findlayersTR2well` (Fortran `utils.f08:45-87`).
    Dado o perfil estratificado e as profundidades do TX (`h0`) e RX (`z`),
    retorna os índices 0-based das camadas onde eles estão.

    Args:
        n: Número total de camadas (incluindo semi-espaços).
        h0: Profundidade do transmissor em metros. Convenção: positivo
            para baixo; negativo indica TX no ar (semi-espaço superior).
        z: Profundidade do receptor em metros.
        prof: Array `(n+1,)` float64 produzido por :func:`sanitize_profile`.
            `prof[0]=0.0` (topo), `prof[n]=1e300` (fundo).

    Returns:
        Tupla `(camad_t, camad_r)`:

        - ``camad_t``: índice 0-based `[0, n-1]` da camada do transmissor.
        - ``camad_r``: índice 0-based `[0, n-1]` da camada do receptor.

    Note:
        **Convenção de igualdade** (paridade Fortran `utils.f08:63`):

        - Para o **receptor**, se `z == prof[i]`, o receptor é colocado
          na camada **acima** (`camad_r = i-1`). Isso evita imprecisão em
          componentes não-tangenciais do tensor.
        - Para o **transmissor**, usa `>` estrito (`utils.f08:80`), de
          modo que `h0 == prof[i]` coloca o TX na camada `i-1`.

        Esta assimetria é proposital e idêntica ao Fortran.

    Example:
        Perfil 3-camadas (topo + meio 5m + base)::

            >>> import numpy as np
            >>> prof = np.array([0.0, 0.0, 5.0, 1e300])
            >>> find_layers_tr(n=3, h0=-1.0, z=2.5, prof=prof)
            (0, 1)
            >>> find_layers_tr(n=3, h0=2.5, z=2.5, prof=prof)
            (1, 1)
            >>> find_layers_tr(n=3, h0=2.5, z=7.0, prof=prof)
            (1, 2)
    """
    # ── Localização do receptor ────────────────────────────────────
    # Fortran `utils.f08:62-72`. No Fortran 1-based:
    #   camad = 1  (default, receptor na 1ª camada)
    #   if z >= prof(n-1) then camad = n
    #   else: do i=n-1,2,-1 ; if z >= prof(i-1) then camad=i; exit ; end if ; end do
    #
    # Tradução 0-based Python (usando prof[i+1] = topo da camada i+1):
    #   prof[n-1] (em 0-based Python) é o topo da camada n-1 (semi-espaço inferior)
    #   Se z >= prof[n-1]: receptor no semi-espaço inferior (camada n-1)
    #   Else: percorre camadas de n-2 para baixo (exclui topo camada 0)
    camad_r = 0
    if z >= prof[n - 1]:
        camad_r = n - 1
    else:
        # Equivalente Fortran: do i=n-1,2,-1; testa z >= prof(i-1)
        # 0-based: varrer i de n-2 até 1 (inclusive); testar z >= prof[i]
        # Atribui camad_r = i e quebra
        for i in range(n - 2, 0, -1):
            if z >= prof[i]:
                camad_r = i
                break

    # ── Localização do transmissor ─────────────────────────────────
    # Fortran `utils.f08:75-85`. Diferença sutil: usa `>` estrito em vez
    # de `>=` (para evitar TX exatamente na interface — paridade Fortran).
    camad_t = 0
    if h0 > prof[n - 1]:
        camad_t = n - 1
    else:
        for j in range(n - 2, 0, -1):
            if h0 > prof[j]:
                camad_t = j
                break

    return camad_t, camad_r


@njit
def layer_at_depth(n: int, z: float, prof: np.ndarray) -> int:
    """Determina a camada 0-based em que a profundidade `z` se encontra.

    Port Python+Numba de `layer2z_inwell` (Fortran `utils.f08:299-319`).
    Usado para calcular o `layerObs` (camada do ponto-médio T-R) no
    orquestrador forward.

    Args:
        n: Número total de camadas.
        z: Profundidade em metros.
        prof: Array `(n+1,)` float64 de :func:`sanitize_profile`.

    Returns:
        Índice 0-based `[0, n-1]` da camada.

    Note:
        Mesma convenção de `find_layers_tr`: usa `>=` (camada *abaixo*
        da interface), exceto no semi-espaço superior onde `z < prof[1]`
        coloca na camada 0.

    Example:
        >>> import numpy as np
        >>> prof = np.array([0.0, 0.0, 5.0, 10.0, 1e300])
        >>> layer_at_depth(n=4, z=-1.0, prof=prof)  # semi-espaço topo
        0
        >>> layer_at_depth(n=4, z=2.5, prof=prof)   # camada 1
        1
        >>> layer_at_depth(n=4, z=7.5, prof=prof)   # camada 2
        2
        >>> layer_at_depth(n=4, z=100.0, prof=prof) # semi-espaço base
        3
    """
    layer = 0
    if z >= prof[n - 1]:
        layer = n - 1
    else:
        for i in range(n - 2, 0, -1):
            if z >= prof[i]:
                layer = i
                break
    return layer


__all__ = [
    "sanitize_profile",
    "find_layers_tr",
    "layer_at_depth",
]
