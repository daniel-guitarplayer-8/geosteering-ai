# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_numba/dipoles.py                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Dipolos Magnéticos TIV (Sprint 2.2)    ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x + Numba 0.60+ (dual-mode, opcional)             ║
# ║  Dependências: numpy (obrigatório), numba (opcional, speedup JIT)        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Porta para Python+Numba as duas subrotinas Fortran que calculam o    ║
# ║    campo magnético primário de dipolos magnéticos em meios TIV:         ║
# ║                                                                           ║
# ║      • hmd_tiv — Horizontal Magnetic Dipole (2 polarizações: x e y)    ║
# ║                  Port de `hmd_TIV_optimized` em                         ║
# ║                  Fortran_Gerador/magneticdipoles.f08:91–440            ║
# ║                  Produz componentes: Hxx, Hxy, Hxz, Hyx, Hyy, Hyz.     ║
# ║                                                                           ║
# ║      • vmd     — Vertical Magnetic Dipole (1 polarização: z)           ║
# ║                  Port de `vmd_optimized` em                             ║
# ║                  Fortran_Gerador/magneticdipoles.f08:442–624           ║
# ║                  Produz componentes: Hzx, Hzy, Hzz.                     ║
# ║                                                                           ║
# ║    Ambas consomem os 9 arrays de `common_arrays` e 6 fatores de         ║
# ║    `common_factors` (Sprint 2.1) e adicionam a integração de Hankel     ║
# ║    (∫ f(kr)·Jν(kr·r) dkr) via pesos `wJ0`/`wJ1` do filtro selecionado. ║
# ║                                                                           ║
# ║  DIAGRAMA DO FLUXO DE DADOS                                              ║
# ║    ┌────────────────────────────────────────────────────────────────┐  ║
# ║    │  common_arrays + common_factors (Sprint 2.1)                   │  ║
# ║    │            │                                                   │  ║
# ║    │            ▼                                                   │  ║
# ║    │  {u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt}           │  ║
# ║    │  {Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz}                       │  ║
# ║    │            │                                                   │  ║
# ║    │       ┌────┴─────┐                                            │  ║
# ║    │       ▼          ▼                                            │  ║
# ║    │  ┌─────────┐  ┌─────────┐                                    │  ║
# ║    │  │ hmd_tiv │  │   vmd   │                                    │  ║
# ║    │  └────┬────┘  └────┬────┘                                    │  ║
# ║    │       │            │                                          │  ║
# ║    │       ▼            ▼                                          │  ║
# ║    │ (Hxx,Hxy,Hxz,  (Hzx,Hzy,Hzz)                                 │  ║
# ║    │  Hyx,Hyy,Hyz)                                                 │  ║
# ║    │       │            │                                          │  ║
# ║    │       └──────┬─────┘                                          │  ║
# ║    │              ▼                                                │  ║
# ║    │        Tensor H (3×3) complex128                              │  ║
# ║    └────────────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  INTEGRAÇÃO HANKEL DIGITAL                                                ║
# ║    Todos os campos são calculados como integrais de Hankel discretas:   ║
# ║                                                                           ║
# ║        ∫₀^∞ f(kr)·Jν(kr·r) dkr ≈ (1/r) · Σₖ f(kr_k/r)·wJν_k            ║
# ║                                                                           ║
# ║    Onde (kr_k, wJ0_k, wJ1_k) são os npt pontos do filtro Werthmüller, ║
# ║    Kong, ou Anderson carregados via `FilterLoader`. O fator 1/r é      ║
# ║    absorvido pela escala `kr = krJ0J1 / r` feita nas subrotinas.       ║
# ║                                                                           ║
# ║  CONVENÇÕES                                                              ║
# ║    • mx = my = mz = 1.0 A·m² (momento dipolar unitário, linha 16-18    ║
# ║      de parameters.f08)                                                 ║
# ║    • eps = 1e-9 m (threshold de singularidade r → 0)                   ║
# ║    • r_guard = 0.01 m (valor usado quando r < eps)                     ║
# ║    • zeta = i·ω·μ₀ (convenção e^(-iωt), Moran-Gianzero 1979)          ║
# ║    • Indexação: 0-based em Python (Fortran é 1-based)                  ║
# ║    • prof shape: (n+1,) — prof[i] = topo da camada i;                  ║
# ║      prof[i+1] = fundo da camada i                                      ║
# ║                                                                           ║
# ║  DUAL-MODE NUMBA                                                          ║
# ║    Herda o padrão de `_numba/propagation.py`: decorator `@njit` com     ║
# ║    fallback no-op. `cache=True, fastmath=False, error_model="numpy"`.   ║
# ║                                                                           ║
# ║  REFERÊNCIAS BIBLIOGRÁFICAS                                              ║
# ║    • Moran, J.H. & Gianzero, S. (1979). Effects of formation           ║
# ║      anisotropy on resistivity-logging measurements. Geophysics 44.    ║
# ║    • Ward, S.H. & Hohmann, G.W. (1988). Electromagnetic theory for     ║
# ║      geophysical applications. SEG IG-3, §4.3 (dipole sources).        ║
# ║    • Chew, W.C. (1995). Waves and Fields in Inhomogeneous Media. §2.2 ║
# ║    • Kong, F.N. (2007). Hankel transform filters for dipole antenna    ║
# ║      radiation in a conductive medium. Geophysical Prospecting 55.    ║
# ║                                                                           ║
# ║  NOTAS DE IMPLEMENTAÇÃO                                                   ║
# ║    1. As subrotinas Fortran fazem alocação dinâmica de `Tudw, Txdw,    ║
# ║       Tuup, Txup, TEdwz, TEupz` no hot path. No Python+Numba fazemos   ║
# ║       a mesma coisa — Numba otimiza alocações pequenas via stack-like  ║
# ║       fast path quando dtype e shape são conhecidos em tempo de JIT.   ║
# ║    2. A função trata 3 casos geométricos: camadR > camadT (receptor   ║
# ║       abaixo do transmissor), camadR < camadT (acima), e camadR ==    ║
# ║       camadT (mesma camada — sub-caso z ≤ h0 ou z > h0).              ║
# ║    3. Os potenciais primários mx, mz = 1.0 A·m² (momento unitário).   ║
# ║    4. O retorno de `hmd_tiv` é 3 arrays `(2,)` complex128 para os     ║
# ║       dois dipolos (x e y) simultaneamente — padrão `hmdxy` Fortran.  ║
# ║    5. O retorno de `vmd` é 3 escalares complex128 (Hx, Hy, Hz).      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Campos magnéticos primários de dipolos horizontais e verticais em meios TIV.

Este módulo implementa dois kernels Numba:

- :func:`hmd_tiv` — HMD (Horizontal Magnetic Dipole), polarizações x e y.
- :func:`vmd`    — VMD (Vertical Magnetic Dipole).

Ambos consomem os 9 arrays invariantes produzidos por
:func:`geosteering_ai.simulation._numba.propagation.common_arrays` e os 6
fatores de :func:`~geosteering_ai.simulation._numba.propagation.common_factors`,
e adicionam a integração de Hankel final via pesos `wJ0`/`wJ1` do filtro.

Example:
    Uso em teste unitário (half-space isotrópico)::

        >>> import numpy as np
        >>> from geosteering_ai.simulation._numba.propagation import (
        ...     common_arrays, common_factors,
        ... )
        >>> from geosteering_ai.simulation._numba.dipoles import vmd
        >>> from geosteering_ai.simulation.filters import FilterLoader
        >>> filt = FilterLoader().load("werthmuller_201pt")
        >>> n, npt = 1, filt.abscissas.shape[0]
        >>> h = np.array([0.0], dtype=np.float64)
        >>> eta = np.array([[0.01, 0.01]], dtype=np.float64)  # 100 Ω·m
        >>> prof = np.array([0.0, 1e300], dtype=np.float64)
        >>> zeta = 1j * 2 * np.pi * 20000.0 * 4e-7 * np.pi
        >>> u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt = common_arrays(
        ...     n, npt, 1.0, filt.abscissas, zeta, h, eta
        ... )
        >>> Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz = common_factors(
        ...     n, npt, 0.0, h, prof, 0,
        ...     u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup,
        ... )
        >>> Hx, Hy, Hz = vmd(
        ...     0.0, 0.0, 0.0, n, 0, 0, npt, filt.abscissas,
        ...     filt.weights_j0, filt.weights_j1, h, prof, zeta,
        ...     1.0, 0.0, 0.0, u, uh, AdmInt, RTEdw, RTEup, FEdwz, FEupz,
        ... )
        >>> # Hz em full-space isotrópico com L=1, f=20 kHz, ρ=100 Ω·m
        >>> abs(Hz) > 0
        True

Note:
    Esta API é **interna** ao simulador Python. Consumidores devem usar
    `geosteering_ai.simulation.simulate(cfg)` (Sprint 2.5 em diante).
"""
from __future__ import annotations

from typing import Final

import numpy as np

from geosteering_ai.simulation._numba.propagation import HAS_NUMBA, njit

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES PRIVADAS (paridade com parameters.f08)
# ──────────────────────────────────────────────────────────────────────────────
# Threshold abaixo do qual r = sqrt((cx-Tx)² + (cy-Ty)²) é considerado
# singular. Paridade com `eps` em parameters.f08:10 (`eps = 1.d-9`).
_EPS_SINGULARITY: Final[float] = 1.0e-9

# Valor usado como `r` quando distância horizontal é singular. Paridade
# com linha 170 (hmd_TIV) e 473 (vmd) de magneticdipoles.f08 (`r = 1.d-2`).
_R_GUARD: Final[float] = 1.0e-2

# Momentos dipolares unitários (parameters.f08:16-18).
# Fatoráveis mais tarde como argumentos opcionais quando o simulador
# suportar dipolos de momento arbitrário (atualmente normalizado a 1).
_MX: Final[complex] = complex(1.0, 0.0)
_MZ: Final[complex] = complex(1.0, 0.0)

# Aliases para π e 2π (reduz divisões repetidas no hot path).
_PI: Final[float] = np.pi
_TWO_PI: Final[float] = 2.0 * np.pi


# ──────────────────────────────────────────────────────────────────────────────
# HMD_TIV — Horizontal Magnetic Dipole (2 polarizações: x, y)
# ──────────────────────────────────────────────────────────────────────────────
@njit
def hmd_tiv(
    Tx: float,
    Ty: float,
    h0: float,
    n: int,
    camad_r: int,
    camad_t: int,
    npt: int,
    krJ0J1: np.ndarray,
    wJ0: np.ndarray,
    wJ1: np.ndarray,
    h: np.ndarray,
    prof: np.ndarray,
    zeta: complex,
    eta: np.ndarray,
    cx: float,
    cy: float,
    z: float,
    u: np.ndarray,
    s: np.ndarray,
    uh: np.ndarray,
    sh: np.ndarray,
    RTEdw: np.ndarray,
    RTEup: np.ndarray,
    RTMdw: np.ndarray,
    RTMup: np.ndarray,
    Mxdw: np.ndarray,
    Mxup: np.ndarray,
    Eudw: np.ndarray,
    Euup: np.ndarray,
):
    """Campo magnético primário de um HMD (Horizontal Magnetic Dipole) em TIV.

    Port Python+Numba de `hmd_TIV_optimized` (Fortran
    `magneticdipoles.f08:91-440`). Produz simultaneamente o campo de
    **dois dipolos HMD**: um alinhado ao eixo x (`hmdx`) e outro ao
    eixo y (`hmdy`), aproveitando a rotação de 90° anti-horária entre
    os dois (economia: ~30% de instruções vs chamar 2 vezes).

    O campo é obtido pela soma de potenciais TE e TM com decaimento
    exponencial nas camadas, integrados via quadratura de Hankel
    digital (`wJ0`, `wJ1`).

    Args:
        Tx: Abscissa x do transmissor em metros.
        Ty: Ordenada y do transmissor em metros.
        h0: Profundidade (z) do transmissor em metros. Convenção:
            positivo para baixo, negativo indica transmissor no ar.
        n: Número de camadas geológicas.
        camad_r: Índice 0-based da camada do receptor.
        camad_t: Índice 0-based da camada do transmissor.
        npt: Número de pontos do filtro Hankel (61, 201 ou 801).
        krJ0J1: Array `(npt,)` float64 com abscissas do filtro Hankel
            (de `FilterLoader.load(...).abscissas`).
        wJ0: Array `(npt,)` float64 com pesos da bessel J₀
            (de `FilterLoader.load(...).weights_j0`).
        wJ1: Array `(npt,)` float64 com pesos da bessel J₁
            (de `FilterLoader.load(...).weights_j1`).
        h: Array `(n,)` float64 com espessuras das camadas em metros.
        prof: Array `(n+1,)` float64 com profundidades das interfaces.
            `prof[i]` = topo da camada i; `prof[i+1]` = fundo da camada i.
        zeta: Impeditividade complexa `i·ω·μ₀`.
        eta: Array `(n, 2)` float64 com `eta[i, 0]` = σₕ e
            `eta[i, 1]` = σᵥ da camada i.
        cx: Abscissa x do receptor em metros.
        cy: Ordenada y do receptor em metros.
        z: Profundidade (z) do receptor em metros.
        u, s, uh, sh: Arrays `(npt, n)` complex128 produzidos por
            :func:`common_arrays` — constantes de propagação.
        RTEdw, RTEup, RTMdw, RTMup: Arrays `(npt, n)` complex128
            produzidos por :func:`common_arrays` — coefs. de reflexão.
        Mxdw, Mxup, Eudw, Euup: Arrays `(npt,)` complex128 produzidos
            por :func:`common_factors` — fatores de onda refletida TM/TE.

    Returns:
        Tupla de 3 `np.ndarray` shape `(2,)` complex128:

        - ``Hx_p`` — `[Hx_hmdx, Hx_hmdy]` (campo Hx do dipolo x e y)
        - ``Hy_p`` — `[Hy_hmdx, Hy_hmdy]`
        - ``Hz_p`` — `[Hz_hmdx, Hz_hmdy]`

        Indexando como tensor 3×3: `H[dipolo_row, componente_col]`
        fornece as 6 componentes não-triviais (Hxx, Hxy, Hxz, Hyx, Hyy,
        Hyz). Hzx, Hzy, Hzz vêm de :func:`vmd`.

    Note:
        A rotação entre hmdx e hmdy segue a relação:

            hmdx → hmdy:  x → y,  y → -x,  Hx → Hy,  Hy → -Hx

        O código espelha o case `hmdxy` do Fortran (linha 413-435),
        que calcula ambos os dipolos em uma única chamada.

        **Casos geométricos tratados**:

        1. `camad_r == 0 and camad_t != 0` — receptor na camada topo
        2. `camad_r < camad_t`              — receptor acima do TX
        3. `camad_r == camad_t and z ≤ h0`  — mesma camada, acima
        4. `camad_r == camad_t and z > h0`  — mesma camada, abaixo
        5. `camad_r > camad_t and camad_r != n-1` — abaixo em camada interna
        6. `camad_r == n-1`                  — receptor na camada final

    Example:
        Teste single-layer (full-space isotrópico)::

            >>> # veja test_simulation_numba_dipoles.py
    """
    # ──────────────────────────────────────────────────────────────────
    # ETAPA 1 — Geometria local (x, y, r, kr)
    # ──────────────────────────────────────────────────────────────────
    # Fortran linhas 159-170: aplica threshold de igualdade em eps antes
    # de calcular r. Se r < eps, usa r = 1e-2 m como guard.
    if abs(cx - Tx) < _EPS_SINGULARITY:
        x = 0.0
    else:
        x = cx - Tx
    if abs(cy - Ty) < _EPS_SINGULARITY:
        y = 0.0
    else:
        y = cy - Ty
    r = np.sqrt(x * x + y * y)
    if r < _EPS_SINGULARITY:
        r = _R_GUARD

    # Abscissas escaladas — kr[k] = krJ0J1[k] / r (vetor de npt pontos).
    kr = krJ0J1 / r

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 2 — Alocação dos potenciais Tudw/Txdw/Tuup/Txup
    # ──────────────────────────────────────────────────────────────────
    # Usamos shape (npt, n) — maior do que o intervalo [camadT, camadR]
    # estritamente necessário. Isso evita reallocs por caso geométrico
    # e mantém Numba feliz com tipos estáticos. Os slots fora do
    # intervalo relevante ficam com lixo (nunca são lidos).
    Tudw = np.zeros((npt, n), dtype=np.complex128)
    Txdw = np.zeros((npt, n), dtype=np.complex128)
    Tuup = np.zeros((npt, n), dtype=np.complex128)
    Txup = np.zeros((npt, n), dtype=np.complex128)

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 3 — Propagação dos potenciais entre camadas
    # ──────────────────────────────────────────────────────────────────
    # Três casos geométricos: camad_r > camad_t (descida),
    # camad_r < camad_t (subida), camad_r == camad_t (mesma camada).
    # A estrutura replica linhas 178-301 do Fortran linha-a-linha.
    if camad_r > camad_t:
        # ── CASO A: receptor abaixo do transmissor — propagação descendente
        for j in range(camad_t, camad_r + 1):
            if j == camad_t:
                Txdw[:, j] = _MX / (2.0 * s[:, camad_t])
                Tudw[:, j] = -_MX / 2.0
            elif j == (camad_t + 1) and j == (n - 1):
                # Próxima camada é a última — trata semi-espaço inferior
                if n > 1:
                    Txdw[:, j] = (
                        s[:, j - 1]
                        * Txdw[:, j - 1]
                        * (
                            np.exp(-s[:, j - 1] * (prof[camad_t + 1] - h0))
                            + RTMup[:, j - 1] * Mxup * np.exp(-sh[:, j - 1])
                            - RTMdw[:, j - 1] * Mxdw
                        )
                        / s[:, j]
                    )
                    Tudw[:, j] = (
                        u[:, j - 1]
                        * Tudw[:, j - 1]
                        * (
                            np.exp(-u[:, j - 1] * (prof[camad_t + 1] - h0))
                            - RTEup[:, j - 1] * Euup * np.exp(-uh[:, j - 1])
                            - RTEdw[:, j - 1] * Eudw
                        )
                        / u[:, j]
                    )
                else:
                    # Caso degenerado (dois semi-espaços apenas)
                    Txdw[:, j] = (
                        s[:, j - 1]
                        * Txdw[:, j - 1]
                        * (np.exp(s[:, j - 1] * h0) - RTMdw[:, j - 1] * Mxdw)
                        / s[:, j]
                    )
                    Tudw[:, j] = (
                        u[:, j - 1]
                        * Tudw[:, j - 1]
                        * (np.exp(u[:, j - 1] * h0) - RTEdw[:, j - 1] * Eudw)
                        / u[:, j]
                    )
            elif j == (camad_t + 1) and j != (n - 1):
                # Primeira camada depois do TX, mas não é a última
                Txdw[:, j] = (
                    s[:, j - 1]
                    * Txdw[:, j - 1]
                    * (
                        np.exp(-s[:, j - 1] * (prof[camad_t + 1] - h0))
                        + RTMup[:, j - 1] * Mxup * np.exp(-sh[:, j - 1])
                        - RTMdw[:, j - 1] * Mxdw
                    )
                    / ((1.0 - RTMdw[:, j] * np.exp(-2.0 * sh[:, j])) * s[:, j])
                )
                Tudw[:, j] = (
                    u[:, j - 1]
                    * Tudw[:, j - 1]
                    * (
                        np.exp(-u[:, j - 1] * (prof[camad_t + 1] - h0))
                        - RTEup[:, j - 1] * Euup * np.exp(-uh[:, j - 1])
                        - RTEdw[:, j - 1] * Eudw
                    )
                    / ((1.0 - RTEdw[:, j] * np.exp(-2.0 * uh[:, j])) * u[:, j])
                )
            elif j != (n - 1):
                # Camada intermediária (não primeira após TX, não última)
                Txdw[:, j] = (
                    s[:, j - 1]
                    * Txdw[:, j - 1]
                    * np.exp(-sh[:, j - 1])
                    * (1.0 - RTMdw[:, j - 1])
                    / ((1.0 - RTMdw[:, j] * np.exp(-2.0 * sh[:, j])) * s[:, j])
                )
                Tudw[:, j] = (
                    u[:, j - 1]
                    * Tudw[:, j - 1]
                    * np.exp(-uh[:, j - 1])
                    * (1.0 - RTEdw[:, j - 1])
                    / ((1.0 - RTEdw[:, j] * np.exp(-2.0 * uh[:, j])) * u[:, j])
                )
            else:  # j == n - 1 (última camada, semi-espaço inferior)
                Txdw[:, j] = (
                    s[:, j - 1]
                    * Txdw[:, j - 1]
                    * np.exp(-sh[:, j - 1])
                    * (1.0 - RTMdw[:, j - 1])
                    / s[:, j]
                )
                Tudw[:, j] = (
                    u[:, j - 1]
                    * Tudw[:, j - 1]
                    * np.exp(-uh[:, j - 1])
                    * (1.0 - RTEdw[:, j - 1])
                    / u[:, j]
                )
    elif camad_r < camad_t:
        # ── CASO B: receptor acima do transmissor — propagação ascendente
        for j in range(camad_t, camad_r - 1, -1):
            if j == camad_t:
                Txup[:, j] = _MX / (2.0 * s[:, camad_t])
                Tuup[:, j] = _MX / 2.0
            elif j == (camad_t - 1) and j == 0:
                # Camada imediatamente acima do TX é a primeira
                if n > 1:
                    Txup[:, j] = (
                        s[:, j + 1]
                        * Txup[:, j + 1]
                        * (
                            np.exp(-s[:, j + 1] * h0)
                            - RTMup[:, j + 1] * Mxup
                            + RTMdw[:, j + 1] * Mxdw * np.exp(-sh[:, j + 1])
                        )
                        / s[:, j]
                    )
                    Tuup[:, j] = (
                        u[:, j + 1]
                        * Tuup[:, j + 1]
                        * (
                            np.exp(-u[:, j + 1] * h0)
                            - RTEup[:, j + 1] * Euup
                            - RTEdw[:, j + 1] * Eudw * np.exp(-uh[:, j + 1])
                        )
                        / u[:, j]
                    )
                else:
                    # Dois semi-espaços apenas
                    Txup[:, j] = (
                        s[:, j + 1]
                        * Txup[:, j + 1]
                        * (np.exp(-s[:, j + 1] * h0) - RTMup[:, j + 1] * Mxup)
                        / s[:, j]
                    )
                    Tuup[:, j] = (
                        u[:, j + 1]
                        * Tuup[:, j + 1]
                        * (np.exp(-u[:, j + 1] * h0) - RTEup[:, j + 1] * Euup)
                        / u[:, j]
                    )
            elif j == (camad_t - 1) and j != 0:
                # Primeira camada acima do TX, não é a topo
                Txup[:, j] = (
                    s[:, j + 1]
                    * Txup[:, j + 1]
                    * (
                        np.exp(s[:, j + 1] * (prof[j + 1] - h0))
                        + RTMdw[:, j + 1] * Mxdw * np.exp(-sh[:, j + 1])
                        - RTMup[:, j + 1] * Mxup
                    )
                    / ((1.0 - RTMup[:, j] * np.exp(-2.0 * sh[:, j])) * s[:, j])
                )
                Tuup[:, j] = (
                    u[:, j + 1]
                    * Tuup[:, j + 1]
                    * (
                        np.exp(u[:, j + 1] * (prof[camad_t] - h0))
                        - RTEup[:, j + 1] * Euup
                        - RTEdw[:, j + 1] * Eudw * np.exp(-uh[:, j + 1])
                    )
                    / ((1.0 - RTEup[:, j] * np.exp(-2.0 * uh[:, j])) * u[:, j])
                )
            elif j != 0:
                # Camada intermediária (acima)
                Txup[:, j] = (
                    s[:, j + 1]
                    * Txup[:, j + 1]
                    * np.exp(-sh[:, j + 1])
                    * (1.0 - RTMup[:, j + 1])
                    / ((1.0 - RTMup[:, j] * np.exp(-2.0 * sh[:, j])) * s[:, j])
                )
                Tuup[:, j] = (
                    u[:, j + 1]
                    * Tuup[:, j + 1]
                    * np.exp(-uh[:, j + 1])
                    * (1.0 - RTEup[:, j + 1])
                    / ((1.0 - RTEup[:, j] * np.exp(-2.0 * uh[:, j])) * u[:, j])
                )
            else:  # j == 0 (topo)
                Txup[:, j] = (
                    s[:, j + 1]
                    * Txup[:, j + 1]
                    * np.exp(-sh[:, j + 1])
                    * (1.0 - RTMup[:, j + 1])
                    / s[:, j]
                )
                Tuup[:, j] = (
                    u[:, j + 1]
                    * Tuup[:, j + 1]
                    * np.exp(-uh[:, j + 1])
                    * (1.0 - RTEup[:, j + 1])
                    / u[:, j]
                )
    else:
        # ── CASO C: mesma camada (camad_r == camad_t)
        Tudw[:, camad_t] = -_MX / 2.0
        Tuup[:, camad_t] = _MX / 2.0
        Txdw[:, camad_t] = _MX / (2.0 * s[:, camad_t])
        Txup[:, camad_t] = Txdw[:, camad_t]

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 4 — Fatores geométricos auxiliares
    # ──────────────────────────────────────────────────────────────────
    x2_r2 = x * x / (r * r)
    y2_r2 = y * y / (r * r)
    xy_r2 = x * y / (r * r)
    twox2_r2m1 = 2.0 * x2_r2 - 1.0
    twoy2_r2m1 = 2.0 * y2_r2 - 1.0
    twopir = _TWO_PI * r

    # kh² da camada do receptor (usado nos kernels).
    kh2_r = -zeta * eta[camad_r, 0]

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 5 — Kernels Ktm, Kte, Ktedz (dependem do caso geométrico)
    # ──────────────────────────────────────────────────────────────────
    # Fortran linhas 310-386: 6 casos (camad_r == 0 with camad_t != 0,
    # camad_r < camad_t, same layer above/below, above interior layer,
    # and last layer).
    Ktm = np.zeros(npt, dtype=np.complex128)
    Kte = np.zeros(npt, dtype=np.complex128)
    Ktedz = np.zeros(npt, dtype=np.complex128)

    if camad_r == 0 and camad_t != 0:
        # Receptor na camada topo, TX abaixo (linha 310 Fortran)
        Ktm = Txup[:, 0] * np.exp(s[:, 0] * z)
        Kte = Tuup[:, 0] * np.exp(u[:, 0] * z)
        Ktedz = u[:, 0] * Kte
    elif camad_r < camad_t:
        # Receptor acima do TX, camada interna (linha 319)
        Ktm = Txup[:, camad_r] * (
            np.exp(s[:, camad_r] * (z - prof[camad_r + 1]))
            + RTMup[:, camad_r]
            * np.exp(-s[:, camad_r] * (z - prof[camad_r] + h[camad_r]))
        )
        Kte = Tuup[:, camad_r] * (
            np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
            + RTEup[:, camad_r]
            * np.exp(-u[:, camad_r] * (z - prof[camad_r] + h[camad_r]))
        )
        Ktedz = (
            u[:, camad_r]
            * Tuup[:, camad_r]
            * (
                np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
                - RTEup[:, camad_r]
                * np.exp(-u[:, camad_r] * (z - prof[camad_r] + h[camad_r]))
            )
        )
    elif camad_r == camad_t and z <= h0:
        # Mesma camada, receptor acima do TX (linha 332)
        Ktm = Txup[:, camad_r] * (
            np.exp(s[:, camad_r] * (z - h0))
            + RTMup[:, camad_r] * Mxup * np.exp(-s[:, camad_r] * (z - prof[camad_r]))
            + RTMdw[:, camad_r] * Mxdw * np.exp(s[:, camad_r] * (z - prof[camad_r + 1]))
        )
        Kte = Tuup[:, camad_r] * (
            np.exp(u[:, camad_r] * (z - h0))
            + RTEup[:, camad_r] * Euup * np.exp(-u[:, camad_r] * (z - prof[camad_r]))
            - RTEdw[:, camad_r] * Eudw * np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
        )
        Ktedz = (
            u[:, camad_r]
            * Tuup[:, camad_r]
            * (
                np.exp(u[:, camad_r] * (z - h0))
                - RTEup[:, camad_r] * Euup * np.exp(-u[:, camad_r] * (z - prof[camad_r]))
                - RTEdw[:, camad_r]
                * Eudw
                * np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
            )
        )
    elif camad_r == camad_t and z > h0:
        # Mesma camada, receptor abaixo do TX (linha 348)
        Ktm = Txdw[:, camad_r] * (
            np.exp(-s[:, camad_r] * (z - h0))
            + RTMup[:, camad_r] * Mxup * np.exp(-s[:, camad_r] * (z - prof[camad_r]))
            + RTMdw[:, camad_r] * Mxdw * np.exp(s[:, camad_r] * (z - prof[camad_r + 1]))
        )
        Kte = Tudw[:, camad_r] * (
            np.exp(-u[:, camad_r] * (z - h0))
            - RTEup[:, camad_r] * Euup * np.exp(-u[:, camad_r] * (z - prof[camad_r]))
            + RTEdw[:, camad_r] * Eudw * np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
        )
        Ktedz = (
            -u[:, camad_r]
            * Tudw[:, camad_r]
            * (
                np.exp(-u[:, camad_r] * (z - h0))
                - RTEup[:, camad_r] * Euup * np.exp(-u[:, camad_r] * (z - prof[camad_r]))
                - RTEdw[:, camad_r]
                * Eudw
                * np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
            )
        )
    elif camad_r > camad_t and camad_r != (n - 1):
        # Receptor abaixo do TX, camada interna (linha 364)
        Ktm = Txdw[:, camad_r] * (
            np.exp(-s[:, camad_r] * (z - prof[camad_r]))
            + RTMdw[:, camad_r]
            * np.exp(s[:, camad_r] * (z - prof[camad_r + 1] - h[camad_r]))
        )
        Kte = Tudw[:, camad_r] * (
            np.exp(-u[:, camad_r] * (z - prof[camad_r]))
            + RTEdw[:, camad_r]
            * np.exp(u[:, camad_r] * (z - prof[camad_r + 1] - h[camad_r]))
        )
        Ktedz = (
            -u[:, camad_r]
            * Tudw[:, camad_r]
            * (
                np.exp(-u[:, camad_r] * (z - prof[camad_r]))
                - RTEdw[:, camad_r]
                * np.exp(u[:, camad_r] * (z - prof[camad_r + 1] - h[camad_r]))
            )
        )
    else:
        # Receptor na última camada (linha 377)
        Ktm = Txdw[:, camad_r] * np.exp(-s[:, camad_r] * (z - prof[camad_r]))
        Kte = Tudw[:, camad_r] * np.exp(-u[:, camad_r] * (z - prof[camad_r]))
        Ktedz = -u[:, camad_r] * Kte

    # Multiplicação pelos pesos Hankel (J0, J1).
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1
    Kte_J1 = Kte * wJ1
    Ktedz_J0 = Ktedz * wJ0
    Ktedz_J1 = Ktedz * wJ1

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 6 — Assembly das 6 componentes H (hmdx + hmdy simultâneos)
    # ──────────────────────────────────────────────────────────────────
    # Fortran linhas 413-435 (case 'hmdxy'). Os kernels abaixo vêm
    # diretamente das equações de Ward-Hohmann 1988 §4.3.

    # Pré-computar somas dos kernels (evita recálculos entre kx e ky).
    sum_Ktedz_J1 = np.sum(Ktedz_J1)
    sum_Ktm_J1 = np.sum(Ktm_J1)
    sum_Ktedz_J0_kr = np.sum(Ktedz_J0 * kr)
    sum_Ktm_J0_kr = np.sum(Ktm_J0 * kr)
    sum_add_J1 = np.sum(Ktedz_J1 + kh2_r * Ktm_J1)
    sum_add_J0_kr = np.sum((Ktedz_J0 + kh2_r * Ktm_J0) * kr)
    sum_Kte_J1_kr2 = np.sum(Kte_J1 * kr * kr)

    # Hx_p, Hy_p, Hz_p são arrays (2,) — [0] para hmdx, [1] para hmdy
    Hx_p = np.zeros(2, dtype=np.complex128)
    Hy_p = np.zeros(2, dtype=np.complex128)
    Hz_p = np.zeros(2, dtype=np.complex128)

    # ── HMDX (dipolo alinhado ao x) ────────────────────────────────
    kernelHxJ1 = (twox2_r2m1 * sum_Ktedz_J1 - kh2_r * twoy2_r2m1 * sum_Ktm_J1) / r
    kernelHxJ0 = x2_r2 * sum_Ktedz_J0_kr - kh2_r * y2_r2 * sum_Ktm_J0_kr
    Hx_p[0] = (kernelHxJ1 - kernelHxJ0) / twopir

    kernelHyJ1 = sum_add_J1 / r
    kernelHyJ0 = sum_add_J0_kr / 2.0
    Hy_p[0] = xy_r2 * (kernelHyJ1 - kernelHyJ0) / _PI / r

    kernelHzJ1 = x * sum_Kte_J1_kr2 / r
    Hz_p[0] = -kernelHzJ1 / twopir

    # ── HMDY (dipolo alinhado ao y, rotação 90°) ───────────────────
    # Por reaproveitamento da álgebra Fortran `hmdxy`:
    # Hx_hmdy = Hy_hmdx (linha 428)
    Hx_p[1] = Hy_p[0]

    kernelHyJ1_y = (twoy2_r2m1 * sum_Ktedz_J1 - kh2_r * twox2_r2m1 * sum_Ktm_J1) / r
    kernelHyJ0_y = y2_r2 * sum_Ktedz_J0_kr - kh2_r * x2_r2 * sum_Ktm_J0_kr
    Hy_p[1] = (kernelHyJ1_y - kernelHyJ0_y) / twopir

    kernelHzJ1_y = y * sum_Kte_J1_kr2 / r
    Hz_p[1] = -kernelHzJ1_y / twopir

    return (Hx_p, Hy_p, Hz_p)


# ──────────────────────────────────────────────────────────────────────────────
# VMD — Vertical Magnetic Dipole (polarização z)
# ──────────────────────────────────────────────────────────────────────────────
@njit
def vmd(
    Tx: float,
    Ty: float,
    h0: float,
    n: int,
    camad_r: int,
    camad_t: int,
    npt: int,
    krJ0J1: np.ndarray,
    wJ0: np.ndarray,
    wJ1: np.ndarray,
    h: np.ndarray,
    prof: np.ndarray,
    zeta: complex,
    cx: float,
    cy: float,
    z: float,
    u: np.ndarray,
    uh: np.ndarray,
    AdmInt: np.ndarray,
    RTEdw: np.ndarray,
    RTEup: np.ndarray,
    FEdwz: np.ndarray,
    FEupz: np.ndarray,
):
    """Campo magnético primário de um VMD (Vertical Magnetic Dipole).

    Port Python+Numba de `vmd_optimized` (Fortran
    `magneticdipoles.f08:442-624`). Calcula as componentes Hx, Hy, Hz
    no receptor (cx, cy, z) de um dipolo magnético vertical no
    transmissor (Tx, Ty, h0), integrando potenciais TE via quadratura
    de Hankel.

    Args:
        Tx, Ty, h0: Posição do transmissor (m).
        n: Número de camadas.
        camad_r: Índice 0-based da camada do receptor.
        camad_t: Índice 0-based da camada do transmissor.
        npt: Número de pontos do filtro Hankel.
        krJ0J1: Array `(npt,)` — abscissas do filtro.
        wJ0, wJ1: Arrays `(npt,)` — pesos J₀ e J₁.
        h: Array `(n,)` — espessuras das camadas.
        prof: Array `(n+1,)` — profundidades das interfaces.
        zeta: Impeditividade `i·ω·μ₀`.
        cx, cy, z: Posição do receptor (m).
        u, uh, AdmInt, RTEdw, RTEup: Arrays `(npt, n)` de `common_arrays`.
        FEdwz, FEupz: Arrays `(npt,)` de `common_factors`.

    Returns:
        Tupla `(Hx_p, Hy_p, Hz_p)` de 3 escalares complex128:

        - ``Hx_p`` — componente Hzx (linha 3, col 1 do tensor H)
        - ``Hy_p`` — componente Hzy (linha 3, col 2)
        - ``Hz_p`` — componente Hzz (linha 3, col 3) — ★ componente axial

    Note:
        Fortran linhas 479-524 tratam 3 casos para propagação TE
        vertical (TEdwz, TEupz); linhas 527-623 tratam 6 casos
        geométricos para o kernel final. Port line-for-line.

        Componente `Hz_p` é a mais importante para LWD axial — é o que
        os modelos DL consomem como input axial (Re/Im Hzz, colunas
        20-21 do schema 22-col).
    """
    # ── Geometria local ─────────────────────────────────────────────
    if abs(cx - Tx) < _EPS_SINGULARITY:
        x = 0.0
    else:
        x = cx - Tx
    if abs(cy - Ty) < _EPS_SINGULARITY:
        y = 0.0
    else:
        y = cy - Ty
    r = np.sqrt(x * x + y * y)
    if r < _EPS_SINGULARITY:
        r = _R_GUARD

    kr = krJ0J1 / r

    # ── Propagação dos potenciais verticais TEdwz, TEupz ───────────
    TEdwz = np.zeros((npt, n), dtype=np.complex128)
    TEupz = np.zeros((npt, n), dtype=np.complex128)

    if camad_r > camad_t:
        # Receptor abaixo do TX — propaga TEdwz descendente
        for j in range(camad_t, camad_r + 1):
            if j == camad_t:
                TEdwz[:, j] = zeta * _MZ / (2.0 * u[:, j])
            elif j == (camad_t + 1) and j == (n - 1):
                TEdwz[:, j] = TEdwz[:, j - 1] * (
                    np.exp(-u[:, camad_t] * (prof[camad_t + 1] - h0))
                    + RTEup[:, camad_t] * FEupz * np.exp(-uh[:, camad_t])
                    + RTEdw[:, camad_t] * FEdwz
                )
            elif j == (camad_t + 1) and j != (n - 1):
                TEdwz[:, j] = (
                    TEdwz[:, j - 1]
                    * (
                        np.exp(-u[:, camad_t] * (prof[camad_t + 1] - h0))
                        + RTEup[:, camad_t] * FEupz * np.exp(-uh[:, camad_t])
                        + RTEdw[:, camad_t] * FEdwz
                    )
                    / (1.0 + RTEdw[:, j] * np.exp(-2.0 * uh[:, j]))
                )
            elif j != (n - 1):
                TEdwz[:, j] = (
                    TEdwz[:, j - 1]
                    * (1.0 + RTEdw[:, j - 1])
                    * np.exp(-uh[:, j - 1])
                    / (1.0 + RTEdw[:, j] * np.exp(-2.0 * uh[:, j]))
                )
            else:  # j == n - 1
                TEdwz[:, j] = (
                    TEdwz[:, j - 1] * (1.0 + RTEdw[:, j - 1]) * np.exp(-uh[:, j - 1])
                )
    elif camad_r < camad_t:
        # Receptor acima do TX — propaga TEupz ascendente
        for j in range(camad_t, camad_r - 1, -1):
            if j == camad_t:
                TEupz[:, j] = zeta * _MZ / (2.0 * u[:, j])
            elif j == (camad_t - 1) and j == 0:
                TEupz[:, j] = TEupz[:, j + 1] * (
                    np.exp(-u[:, camad_t] * h0)
                    + RTEup[:, camad_t] * FEupz
                    + RTEdw[:, camad_t] * FEdwz * np.exp(-uh[:, camad_t])
                )
            elif j == (camad_t - 1) and j != 0:
                TEupz[:, j] = (
                    TEupz[:, j + 1]
                    * (
                        np.exp(u[:, camad_t] * (prof[camad_t] - h0))
                        + RTEup[:, camad_t] * FEupz
                        + RTEdw[:, camad_t] * FEdwz * np.exp(-uh[:, camad_t])
                    )
                    / (1.0 + RTEup[:, j] * np.exp(-2.0 * uh[:, j]))
                )
            elif j != 0:
                TEupz[:, j] = (
                    TEupz[:, j + 1]
                    * (1.0 + RTEup[:, j + 1])
                    * np.exp(-uh[:, j + 1])
                    / (1.0 + RTEup[:, j] * np.exp(-2.0 * uh[:, j]))
                )
            else:  # j == 0
                TEupz[:, j] = (
                    TEupz[:, j + 1] * (1.0 + RTEup[:, j + 1]) * np.exp(-uh[:, j + 1])
                )
    else:
        # Mesma camada (camad_r == camad_t)
        TEdwz[:, camad_r] = zeta * _MZ / (2.0 * u[:, camad_t])
        TEupz[:, camad_r] = TEdwz[:, camad_r]

    # ──────────────────────────────────────────────────────────────────
    # Kernels fac, Ktz, Ktedzz (6 casos geométricos)
    # ──────────────────────────────────────────────────────────────────
    twopir = _TWO_PI * r
    fac = np.zeros(npt, dtype=np.complex128)
    KtedzzJ1 = np.zeros(npt, dtype=np.complex128)

    if camad_r == 0 and camad_t != 0:
        fac = TEupz[:, 0] * np.exp(u[:, 0] * z)
        KtezJ0 = fac * wJ0
        KtezJ1 = fac * wJ1
        KtedzzJ1 = AdmInt[:, 0] * KtezJ1
    elif camad_r < camad_t:
        fac = TEupz[:, camad_r] * (
            np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
            + RTEup[:, camad_r]
            * np.exp(-u[:, camad_r] * (z - prof[camad_r] + h[camad_r]))
        )
        KtezJ0 = fac * wJ0
        KtezJ1 = fac * wJ1
        KtedzzJ1 = (
            AdmInt[:, camad_r]
            * TEupz[:, camad_r]
            * (
                np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
                - RTEup[:, camad_r]
                * np.exp(-u[:, camad_r] * (z - prof[camad_r] + h[camad_r]))
            )
        ) * wJ1
    elif camad_r == camad_t and z <= h0:
        fac = TEupz[:, camad_r] * (
            np.exp(u[:, camad_r] * (z - h0))
            + RTEup[:, camad_r] * FEupz * np.exp(-u[:, camad_r] * (z - prof[camad_r]))
            + RTEdw[:, camad_r] * FEdwz * np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
        )
        KtezJ0 = fac * wJ0
        KtezJ1 = fac * wJ1
        KtedzzJ1 = (
            AdmInt[:, camad_r]
            * TEupz[:, camad_r]
            * (
                np.exp(u[:, camad_r] * (z - h0))
                - RTEup[:, camad_r] * FEupz * np.exp(-u[:, camad_r] * (z - prof[camad_r]))
                + RTEdw[:, camad_r]
                * FEdwz
                * np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
            )
        ) * wJ1
    elif camad_r == camad_t and z > h0:
        fac = TEdwz[:, camad_r] * (
            np.exp(-u[:, camad_r] * (z - h0))
            + RTEup[:, camad_r] * FEupz * np.exp(-u[:, camad_r] * (z - prof[camad_r]))
            + RTEdw[:, camad_r] * FEdwz * np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
        )
        KtezJ0 = fac * wJ0
        KtezJ1 = fac * wJ1
        KtedzzJ1 = (
            -AdmInt[:, camad_r]
            * TEdwz[:, camad_r]
            * (
                np.exp(-u[:, camad_r] * (z - h0))
                + RTEup[:, camad_r] * FEupz * np.exp(-u[:, camad_r] * (z - prof[camad_r]))
                - RTEdw[:, camad_r]
                * FEdwz
                * np.exp(u[:, camad_r] * (z - prof[camad_r + 1]))
            )
        ) * wJ1
    elif camad_r > camad_t and camad_r != (n - 1):
        fac = TEdwz[:, camad_r] * (
            np.exp(-u[:, camad_r] * (z - prof[camad_r]))
            + RTEdw[:, camad_r]
            * np.exp(u[:, camad_r] * (z - prof[camad_r + 1] - h[camad_r]))
        )
        KtezJ0 = fac * wJ0
        KtezJ1 = fac * wJ1
        KtedzzJ1 = (
            -AdmInt[:, camad_r]
            * TEdwz[:, camad_r]
            * (
                np.exp(-u[:, camad_r] * (z - prof[camad_r]))
                - RTEdw[:, camad_r]
                * np.exp(u[:, camad_r] * (z - prof[camad_r + 1] - h[camad_r]))
            )
        ) * wJ1
    else:
        # Última camada (n-1)
        fac = TEdwz[:, n - 1] * np.exp(-u[:, n - 1] * (z - prof[n - 1]))
        KtezJ0 = fac * wJ0
        KtezJ1 = fac * wJ1
        KtedzzJ1 = -AdmInt[:, n - 1] * fac * wJ1

    # ── Assembly final Hx, Hy, Hz ──────────────────────────────────
    sum_KtedzzJ1_kr2 = np.sum(KtedzzJ1 * kr * kr)
    sum_KtezJ0_kr3 = np.sum(KtezJ0 * kr * kr * kr)

    kernelHx = -x * sum_KtedzzJ1_kr2 / twopir
    Hx_p = kernelHx / r

    kernelHy = -y * sum_KtedzzJ1_kr2 / twopir
    Hy_p = kernelHy / r

    kernelHz = sum_KtezJ0_kr3 / 2.0 / _PI / zeta
    Hz_p = kernelHz / r

    return (Hx_p, Hy_p, Hz_p)


__all__ = [
    "HAS_NUMBA",
    "hmd_tiv",
    "vmd",
]
