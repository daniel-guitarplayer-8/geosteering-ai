# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_numba/kernel.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Orquestrador Forward (Sprint 2.4)       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x + Numba 0.60+ (dual-mode, opcional)              ║
# ║  Dependências: numpy (obrigatório), numba (opcional, speedup JIT)         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Orquestrador forward que amarra toda a cadeia de cálculo do tensor   ║
# ║    H em uma única função `fields_in_freqs`:                             ║
# ║                                                                           ║
# ║      1. Determina camadas do TX e RX (find_layers_tr)                   ║
# ║      2. Calcula zeta, eta a partir de frequência e resistividades       ║
# ║      3. Chama common_arrays + common_factors (Sprint 2.1)              ║
# ║      4. Chama hmd_tiv + vmd (Sprint 2.2)                                ║
# ║      5. Monta matH 3×3 a partir dos retornos dos dipolos                ║
# ║      6. Aplica rotate_tensor (RtHR) para o sistema da ferramenta        ║
# ║      7. Retorna cH (9 componentes flat) por frequência                  ║
# ║                                                                           ║
# ║    Porta para Python+Numba a subrotina Fortran `fieldsinfreqs`         ║
# ║    (PerfilaAnisoOmp.f08:937-993).                                       ║
# ║                                                                           ║
# ║  FLUXO DE DADOS                                                           ║
# ║    ┌──────────────────────────────────────────────────────────────────┐ ║
# ║    │  Posição (Tx, Ty, Tz, cx, cy, cz) + perfil (n, rho_h, rho_v, h) │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │   find_layers_tr → (camad_t, camad_r)                           │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │   LOOP por frequência f:                                        │ ║
# ║    │     zeta = i·ω·μ₀                                               │ ║
# ║    │     common_arrays → (u, s, uh, sh, RTEdw, RTEup,                │ ║
# ║    │                      RTMdw, RTMup, AdmInt)                      │ ║
# ║    │     common_factors → (Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)    │ ║
# ║    │     hmd_tiv → (Hx_hmd, Hy_hmd, Hz_hmd)  [shape (2,)]            │ ║
# ║    │     vmd → (Hx_vmd, Hy_vmd, Hz_vmd)  [complex scalars]           │ ║
# ║    │     matH[0,:] = [Hx_hmd[0], Hy_hmd[0], Hz_hmd[0]]  (dipolo x)  │ ║
# ║    │     matH[1,:] = [Hx_hmd[1], Hy_hmd[1], Hz_hmd[1]]  (dipolo y)  │ ║
# ║    │     matH[2,:] = [Hx_vmd,    Hy_vmd,    Hz_vmd   ]  (dipolo z)  │ ║
# ║    │     tH = rotate_tensor(dip_rad, 0, 0, matH)                     │ ║
# ║    │     cH[f, :] = [tH[0,0], tH[0,1], ..., tH[2,2]]  (9 comps)     │ ║
# ║    │            │                                                    │ ║
# ║    │            ▼                                                    │ ║
# ║    │  Retorno: cH shape (nf, 9) complex128                          │ ║
# ║    └──────────────────────────────────────────────────────────────────┘ ║
# ║                                                                           ║
# ║  ALTA PERFORMANCE                                                         ║
# ║    • Loop interno (sobre frequências) NÃO paralelizado porque a         ║
# ║      paralelização será feita no loop externo (posições do poço) na    ║
# ║      Sprint 2.5 via `prange` do Numba.                                 ║
# ║    • Arrays `u`, `s`, etc são pré-alocados no stack do thread           ║
# ║      (Numba otimiza `np.empty` com shape conhecido).                    ║
# ║                                                                           ║
# ║  PRÉ-REQUISITOS                                                           ║
# ║    • Sprint 2.1 — common_arrays, common_factors                         ║
# ║    • Sprint 2.2 — hmd_tiv, vmd                                          ║
# ║    • Sprint 2.3 — find_layers_tr, rotate_tensor                         ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Fortran_Gerador/PerfilaAnisoOmp.f08 (fieldsinfreqs, linhas 937-993)║
# ║    • Fortran_Gerador/utils.f08 (RtHR, findlayersTR2well)                ║
# ║    • Moran & Gianzero (1979) — convenção temporal e^(-iωt)              ║
# ║    • Ward & Hohmann (1988) §4.3 — dipolos magnéticos em meios 1D       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Orquestrador forward do simulador Python (Sprint 2.4).

Função única :func:`fields_in_freqs` que encapsula toda a cadeia
`common_arrays → common_factors → hmd_tiv → vmd → rotate_tensor`
para uma posição fixa TX/RX e uma ou mais frequências.

Example:
    Forward simples (1 camada, 1 frequência, TX/RX broadside)::

        >>> import numpy as np
        >>> from geosteering_ai.simulation.filters import FilterLoader
        >>> from geosteering_ai.simulation._numba.kernel import fields_in_freqs
        >>> filt = FilterLoader().load("werthmuller_201pt")
        >>> n = 1
        >>> rho_h = np.array([100.0])
        >>> rho_v = np.array([100.0])
        >>> freqs = np.array([20000.0])
        >>> cH = fields_in_freqs(
        ...     Tx=0.0, Ty=0.0, Tz=0.0,
        ...     cx=1.0, cy=0.0, cz=0.0,
        ...     dip_rad=0.0,  # ferramenta vertical
        ...     n=n,
        ...     rho_h=rho_h, rho_v=rho_v,
        ...     esp=np.zeros(0, dtype=np.float64),  # sem camadas internas
        ...     freqs_hz=freqs,
        ...     krJ0J1=filt.abscissas,
        ...     wJ0=filt.weights_j0,
        ...     wJ1=filt.weights_j1,
        ... )
        >>> cH.shape
        (1, 9)

Note:
    Esta é a API **interna** da Sprint 2.4. A API pública de alto nível
    `simulate(cfg)` será criada na Sprint 2.5 em `forward.py`.
"""
from __future__ import annotations

import math

import numpy as np

from geosteering_ai.simulation._numba.dipoles import hmd_tiv, vmd
from geosteering_ai.simulation._numba.geometry import (
    find_layers_tr,
    layer_at_depth,
    sanitize_profile,
)
from geosteering_ai.simulation._numba.propagation import (
    common_arrays,
    common_factors,
    njit,
)
from geosteering_ai.simulation._numba.rotation import rotate_tensor

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas
# ──────────────────────────────────────────────────────────────────────────────
# Permeabilidade magnética do vácuo μ₀ = 4π × 10⁻⁷ H/m.
# Paridade com Fortran `parameters.f08:8` (mu = 4.d-7 * pi).
_MU_0: float = 4.0e-7 * math.pi


def fields_in_freqs(
    Tx: float,
    Ty: float,
    Tz: float,
    cx: float,
    cy: float,
    cz: float,
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
    """Calcula o tensor magnético H para uma posição TR e múltiplas frequências.

    Orquestrador forward que amarra toda a cadeia da Fase 2 em uma única
    chamada. Port Python da subrotina Fortran `fieldsinfreqs`
    (`PerfilaAnisoOmp.f08:937-993`).

    Args:
        Tx: Abscissa do transmissor (m).
        Ty: Ordenada do transmissor (m).
        Tz: Profundidade do transmissor (m). Positivo=para baixo.
        cx: Abscissa do receptor (m).
        cy: Ordenada do receptor (m).
        cz: Profundidade do receptor (m).
        dip_rad: Dip da ferramenta em **radianos** (α em Liu 2017).
            0 = ferramenta vertical (eixo z); π/2 = horizontal (eixo x).
        n: Número total de camadas (incluindo 2 semi-espaços).
        rho_h: Array `(n,)` float64 com resistividades horizontais (Ω·m).
        rho_v: Array `(n,)` float64 com resistividades verticais (Ω·m).
        esp: Array 1D float64 com espessuras das camadas internas (pode
            ter shape `(n-2,)` ou `(n,)` — ver :func:`sanitize_profile`).
            Vazio (shape `(0,)`) permitido quando `n == 2` (dois
            semi-espaços apenas).
        freqs_hz: Array `(nf,)` float64 com frequências em Hz.
        krJ0J1: Array `(npt,)` float64 com abscissas do filtro Hankel
            (de `FilterLoader.load(...).abscissas`).
        wJ0: Array `(npt,)` float64 com pesos J₀.
        wJ1: Array `(npt,)` float64 com pesos J₁.

    Returns:
        Array `(nf, 9)` complex128 com o tensor H rotacionado para o
        sistema da ferramenta, uma linha por frequência. Ordem das
        9 componentes flat:

            [Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz]

    Raises:
        ValueError: Se shapes forem inconsistentes ou se `n < 1`.

    Note:
        **Paridade com Fortran**: a função segue linha-a-linha o
        `fieldsinfreqs` do `PerfilaAnisoOmp.f08:937-993`, incluindo:

        - Uso de `find_layers_tr` (equivalente `findlayersTR2well`)
        - Uso de `common_arrays` e `common_factors` (Sprint 2.1)
        - Uso de `hmd_tiv` com convenção `hmdxy` (retorna 2
          polarizações)
        - Uso de `vmd` (retorna 3 escalares)
        - Montagem `matH[0,:] = (hmd_tiv Hx[0], Hy[0], Hz[0])` ...
        - Rotação final via `rotate_tensor(dip_rad, 0, 0, matH)`

        **Não-paridade**: este orquestrador retorna **apenas** `cH`
        (shape `(nf, 9)`), enquanto o Fortran também retorna `zrho`
        (profundidade + resistividades verdadeiras na camada do
        ponto-médio). Essa informação é trivial de calcular
        externamente via :func:`~geometry.layer_at_depth`.

    Example:
        Forward em meio homogêneo isotrópico ρ=100 Ω·m::

            >>> import numpy as np
            >>> from geosteering_ai.simulation.filters import FilterLoader
            >>> from geosteering_ai.simulation._numba.kernel import (
            ...     fields_in_freqs,
            ... )
            >>> filt = FilterLoader().load("werthmuller_201pt")
            >>> cH = fields_in_freqs(
            ...     Tx=0.0, Ty=0.0, Tz=0.0,
            ...     cx=1.0, cy=0.0, cz=0.0,
            ...     dip_rad=0.0,
            ...     n=1,
            ...     rho_h=np.array([100.0]),
            ...     rho_v=np.array([100.0]),
            ...     esp=np.zeros(0, dtype=np.float64),
            ...     freqs_hz=np.array([20000.0]),
            ...     krJ0J1=filt.abscissas,
            ...     wJ0=filt.weights_j0, wJ1=filt.weights_j1,
            ... )
            >>> cH.shape
            (1, 9)
            >>> # Hzz (última coluna) deve ser ~ACp = -1/(4π) no limite estático
            >>> abs(cH[0, 8].real + 1 / (4 * np.pi)) < 1e-3
            True
    """
    # ── Validação de entrada ───────────────────────────────────────
    rho_h = np.ascontiguousarray(rho_h, dtype=np.float64)
    rho_v = np.ascontiguousarray(rho_v, dtype=np.float64)
    esp = np.ascontiguousarray(esp, dtype=np.float64)
    freqs_hz = np.ascontiguousarray(freqs_hz, dtype=np.float64)
    krJ0J1 = np.ascontiguousarray(krJ0J1, dtype=np.float64)
    wJ0 = np.ascontiguousarray(wJ0, dtype=np.float64)
    wJ1 = np.ascontiguousarray(wJ1, dtype=np.float64)

    if n < 1:
        raise ValueError(f"n={n} inválido. Deve ser >= 1.")
    if rho_h.shape != (n,):
        raise ValueError(f"rho_h.shape={rho_h.shape}, esperado ({n},).")
    if rho_v.shape != (n,):
        raise ValueError(f"rho_v.shape={rho_v.shape}, esperado ({n},).")
    if krJ0J1.shape != wJ0.shape or wJ0.shape != wJ1.shape:
        raise ValueError(
            f"Filtro Hankel inconsistente: krJ0J1.shape={krJ0J1.shape}, "
            f"wJ0.shape={wJ0.shape}, wJ1.shape={wJ1.shape}."
        )

    npt = krJ0J1.shape[0]
    nf = freqs_hz.shape[0]

    # ── Geometria: sanitize + localização TX/RX ────────────────────
    # Casos especiais:
    #   n == 1: apenas 1 camada (full-space). Não existem "interfaces"
    #           internas; TX e RX sempre na camada 0. Criamos h=[0] e
    #           prof=[0, 1e300] manualmente para evitar esp de tamanho
    #           negativo em sanitize_profile.
    if n == 1:
        h_arr = np.zeros(1, dtype=np.float64)
        # Sentinels: prof[0] = -1e300 (topo ∞), prof[1] = +1e300 (fundo ∞)
        # Paridade com Fortran sanitize_hprof_well: prof(0) = -1.d300.
        # CRÍTICO: sem o sentinel negativo, exp(s * (prof[0] - h0)) overflow
        # quando h0 é negativo (TX acima de z=0).
        prof_arr = np.array([-1.0e300, 1.0e300], dtype=np.float64)
        camad_t = 0
        camad_r = 0
    else:
        h_arr, prof_arr = sanitize_profile(n, esp)
        camad_t, camad_r = find_layers_tr(n, Tz, cz, prof_arr)

    # ── Admitividade eta[i, 0]=σh, eta[i, 1]=σv ───────────────────
    eta = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        eta[i, 0] = 1.0 / rho_h[i]
        eta[i, 1] = 1.0 / rho_v[i]

    # ── Distância horizontal r (usada em common_arrays e dipolos) ─
    dx = cx - Tx
    dy = cy - Ty
    r = math.sqrt(dx * dx + dy * dy)

    # ── Output ─────────────────────────────────────────────────────
    cH = np.empty((nf, 9), dtype=np.complex128)

    # ──────────────────────────────────────────────────────────────
    # Loop sobre frequências (Fortran: `do i = 1, nf`)
    # ──────────────────────────────────────────────────────────────
    # Não paralelizamos aqui — a paralelização entra no loop externo
    # de posições do poço via `prange` na Sprint 2.5.
    for i_f in range(nf):
        freq = freqs_hz[i_f]
        omega = 2.0 * math.pi * freq
        zeta = 1j * omega * _MU_0

        # ── Sprint 2.1: common_arrays + common_factors ────────────
        u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt = common_arrays(
            n, npt, r, krJ0J1, zeta, h_arr, eta
        )
        Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz = common_factors(
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

        # ── Sprint 2.2: dipolos magnéticos ────────────────────────
        Hx_hmd, Hy_hmd, Hz_hmd = hmd_tiv(
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
        )
        Hx_vmd, Hy_vmd, Hz_vmd = vmd(
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

        # ── Montagem matH 3×3 (paridade Fortran linha 986-988) ────
        # matH[linha, coluna]:
        #   linha = dipolo (0=x, 1=y, 2=z)
        #   coluna = componente do campo observado (0=x, 1=y, 2=z)
        matH = np.empty((3, 3), dtype=np.complex128)
        # Dipolo x (hmdx): retorno do hmd_tiv índice [0]
        matH[0, 0] = Hx_hmd[0]
        matH[0, 1] = Hy_hmd[0]
        matH[0, 2] = Hz_hmd[0]
        # Dipolo y (hmdy): retorno do hmd_tiv índice [1]
        matH[1, 0] = Hx_hmd[1]
        matH[1, 1] = Hy_hmd[1]
        matH[1, 2] = Hz_hmd[1]
        # Dipolo z (vmd): escalares complex
        matH[2, 0] = Hx_vmd
        matH[2, 1] = Hy_vmd
        matH[2, 2] = Hz_vmd

        # ── Sprint 2.3: rotação para o sistema da ferramenta ──────
        # Fortran: `tH = RtHR(ang, 0.d0, 0.d0, matH)` — só dip
        tH = rotate_tensor(dip_rad, 0.0, 0.0, matH)

        # ── Flatten para o layout 9-col (Hxx, Hxy, ..., Hzz) ──────
        cH[i_f, 0] = tH[0, 0]
        cH[i_f, 1] = tH[0, 1]
        cH[i_f, 2] = tH[0, 2]
        cH[i_f, 3] = tH[1, 0]
        cH[i_f, 4] = tH[1, 1]
        cH[i_f, 5] = tH[1, 2]
        cH[i_f, 6] = tH[2, 0]
        cH[i_f, 7] = tH[2, 1]
        cH[i_f, 8] = tH[2, 2]

    return cH


def compute_zrho(
    Tz: float,
    cz: float,
    n: int,
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    esp: np.ndarray,
) -> tuple:
    """Calcula a profundidade do ponto-médio TR e as resistividades ali.

    Helper de conveniência que reproduz o cálculo Fortran `zrho(i,:)`
    (`PerfilaAnisoOmp.f08:972`):

        zobs = (Tz + cz) / 2
        layerObs = layer2z_inwell(n, zobs, prof[1:n-1])
        zrho = (zobs, rho_h[layerObs], rho_v[layerObs])

    Args:
        Tz: Profundidade do transmissor em metros.
        cz: Profundidade do receptor em metros.
        n: Número total de camadas.
        rho_h: Array `(n,)` float64 — resistividades horizontais.
        rho_v: Array `(n,)` float64 — resistividades verticais.
        esp: Array 1D float64 com espessuras (ver :func:`sanitize_profile`).

    Returns:
        Tupla `(zobs, rho_h_mid, rho_v_mid)` de 3 floats.

    Example:
        >>> import numpy as np
        >>> zobs, rh, rv = compute_zrho(
        ...     Tz=0.0, cz=10.0, n=3,
        ...     rho_h=np.array([1.0, 100.0, 1.0]),
        ...     rho_v=np.array([1.0, 100.0, 1.0]),
        ...     esp=np.array([20.0]),
        ... )
        >>> zobs
        5.0
        >>> rh, rv   # meio da camada 1 (100 Ω·m)
        (100.0, 100.0)
    """
    rho_h = np.asarray(rho_h, dtype=np.float64)
    rho_v = np.asarray(rho_v, dtype=np.float64)
    esp = np.asarray(esp, dtype=np.float64)

    zobs = 0.5 * (Tz + cz)
    if n == 1:
        # Full-space: só existe 1 camada
        return zobs, float(rho_h[0]), float(rho_v[0])

    _, prof_arr = sanitize_profile(n, esp)
    layer_obs = layer_at_depth(n, zobs, prof_arr)
    return zobs, float(rho_h[layer_obs]), float(rho_v[layer_obs])


__all__ = [
    "fields_in_freqs",
    "compute_zrho",
]
