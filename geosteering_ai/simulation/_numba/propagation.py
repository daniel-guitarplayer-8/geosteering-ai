# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_numba/propagation.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Propagação EM 1D TIV (Sprint 2.1)       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11                                                ║
# ║  Status      : Produção                                                  ║
# ║  Framework   : NumPy 2.x + Numba 0.60+ (dual-mode, opcional)             ║
# ║  Dependências: numpy (obrigatório), numba (opcional, speedup JIT)        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Porta para Python+Numba as duas subrotinas Fortran que respondem     ║
# ║    por ~75% do custo computacional do forward EM 1D TIV no simulador   ║
# ║    `tatu.x` (módulo `DManisoTIV`):                                      ║
# ║                                                                           ║
# ║      • commonarraysMD   — pré-calcula 9 arrays invariantes no ponto    ║
# ║                           de medida (constantes de propagação u, s,    ║
# ║                           produtos uh, sh, admitância intrínseca       ║
# ║                           AdmInt, coeficientes de reflexão RTEdw,      ║
# ║                           RTEup, RTMdw, RTMup).                         ║
# ║                                                                           ║
# ║      • commonfactorsMD  — calcula 6 fatores de onda refletida na       ║
# ║                           camada onde está o transmissor (Mxdw,        ║
# ║                           Mxup, Eudw, Euup, FEdwz, FEupz), usados pelos║
# ║                           kernels hmd_TIV e vmd da Sprint 2.2.         ║
# ║                                                                           ║
# ║  EQUIVALÊNCIA MATEMÁTICA COM FORTRAN                                      ║
# ║    Fontes primárias:                                                    ║
# ║      • Fortran_Gerador/utils.f08, linhas 158–241 (commonarraysMD)      ║
# ║      • Fortran_Gerador/utils.f08, linhas 243–297 (commonfactorsMD)    ║
# ║    As fórmulas abaixo são port line-for-line do Fortran, com           ║
# ║    indexação convertida de 1-based para 0-based e escolhas de Python   ║
# ║    idiomáticas (slicing em vez de loops explícitos sobre npt).        ║
# ║                                                                           ║
# ║  CONVENÇÕES                                                              ║
# ║    • Convenção temporal: e^(-iωt) (padrão geofísica Moran-Gianzero)   ║
# ║    • zeta = i·ω·μ₀ = impedância do vácuo escalada                      ║
# ║    • eta[i, 0] = σh (condutividade horizontal, S/m)                    ║
# ║    • eta[i, 1] = σv (condutividade vertical, S/m)                      ║
# ║    • h[i]      = espessura da camada i (m)                             ║
# ║    • prof[i]   = profundidade da interface i (m, crescente)            ║
# ║    • camad_t   = índice 0-based da camada onde está o transmissor      ║
# ║                                                                           ║
# ║  DIAGRAMA DO PIPELINE                                                     ║
# ║    ┌────────────────────────────────────────────────────────────────┐  ║
# ║    │  (frequency_hz, eta, h)                                        │  ║
# ║    │            │                                                   │  ║
# ║    │            ▼                                                   │  ║
# ║    │    zeta = i·ω·μ₀                                              │  ║
# ║    │            │                                                   │  ║
# ║    │   (hordist, krJ0J1 from FilterLoader, zeta, h, eta)           │  ║
# ║    │            │                                                   │  ║
# ║    │            ▼                                                   │  ║
# ║    │   ┌───────────────────┐                                       │  ║
# ║    │   │  common_arrays    │                                       │  ║
# ║    │   │  (Sprint 2.1 ★)    │                                       │  ║
# ║    │   └─────────┬─────────┘                                       │  ║
# ║    │             │                                                  │  ║
# ║    │             ▼                                                  │  ║
# ║    │    9 arrays (npt × n) complex128:                              │  ║
# ║    │    u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt           │  ║
# ║    │             │                                                  │  ║
# ║    │  + (h0, prof, camad_t)                                         │  ║
# ║    │             │                                                  │  ║
# ║    │             ▼                                                  │  ║
# ║    │   ┌───────────────────┐                                       │  ║
# ║    │   │  common_factors   │                                       │  ║
# ║    │   │  (Sprint 2.1 ★)    │                                       │  ║
# ║    │   └─────────┬─────────┘                                       │  ║
# ║    │             │                                                  │  ║
# ║    │             ▼                                                  │  ║
# ║    │    6 arrays (npt,) complex128:                                 │  ║
# ║    │    Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz                       │  ║
# ║    │             │                                                  │  ║
# ║    │             ▼                                                  │  ║
# ║    │   ┌───────────────────┐                                       │  ║
# ║    │   │  hmd_TIV, vmd     │  (Sprint 2.2)                         │  ║
# ║    │   └─────────┬─────────┘                                       │  ║
# ║    │             │                                                  │  ║
# ║    │             ▼                                                  │  ║
# ║    │         H tensor                                               │  ║
# ║    └────────────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  DUAL-MODE NUMBA                                                          ║
# ║    O decorador `@njit` é resolvido em tempo de import:                 ║
# ║      • Numba disponível → `@njit(cache=True, fastmath=False,           ║
# ║        error_model="numpy")` compila as funções via LLVM.              ║
# ║      • Numba ausente → `@njit` vira no-op, e as funções rodam em       ║
# ║        NumPy puro (correção preservada, sem speedup).                  ║
# ║    Este design permite rodar testes e debugging em qualquer ambiente, ║
# ║    incluindo CI mínimos e máquinas com conflitos de versão.           ║
# ║                                                                           ║
# ║  REFERÊNCIAS BIBLIOGRÁFICAS                                              ║
# ║    • Moran, J.H. & Gianzero, S. (1979). Effects of formation          ║
# ║      anisotropy on resistivity-logging measurements. Geophysics 44.  ║
# ║    • Ward, S.H. & Hohmann, G.W. (1988). Electromagnetic theory for    ║
# ║      geophysical applications. SEG IG-3.                              ║
# ║    • Chew, W.C. (1995). Waves and Fields in Inhomogeneous Media. IEEE ║
# ║                                                                           ║
# ║  NOTAS DE IMPLEMENTAÇÃO                                                  ║
# ║    1. Sinal de `zeta`: Fortran usa `kh² = -zeta·σh`. Esta convenção    ║
# ║       implica `zeta = i·ω·μ₀` (com sinal positivo na frente do i).    ║
# ║       Compatível com convenção temporal e^(-iωt).                     ║
# ║    2. Guard de singularidade: quando `hordist < 1e-12`, usa           ║
# ║       `r = 0.01 m` (replicando o comportamento Fortran linha 195).   ║
# ║    3. Sem alocação dinâmica: todos os arrays de saída são              ║
# ║       pré-alocados com `np.empty((npt, n), dtype=complex128)`.        ║
# ║    4. Recursões são serializadas (não paralelas) porque cada iteração║
# ║       depende do resultado anterior — paralelização vai para o loop  ║
# ║       externo de posições na Sprint 2.4.                              ║
# ║    5. A função `tanh` é calculada via forma explícita `(1-e^-2x)/    ║
# ║       (1+e^-2x)` para casamento bit-exato com Fortran (linha 212).   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Funções de propagação EM 1D TIV com decoradores Numba.

Este módulo implementa duas funções:

- :func:`common_arrays` — pré-calcula 9 arrays invariantes no ponto de
  medida (equivalente a `commonarraysMD` em `utils.f08`).
- :func:`common_factors` — calcula 6 fatores de onda refletida na camada
  do transmissor (equivalente a `commonfactorsMD` em `utils.f08`).

Ambas são decoradas com `@njit` de forma **dual-mode**: funcionam tanto
com Numba instalado (speedup JIT) quanto sem (NumPy puro).

Example:
    Uso típico em teste unitário::

        >>> import numpy as np
        >>> from geosteering_ai.simulation._numba.propagation import (
        ...     common_arrays, common_factors, HAS_NUMBA,
        ... )
        >>> # Meio homogêneo isotrópico: 1 camada, σ = 1 S/m
        >>> n, npt = 1, 201
        >>> h = np.array([0.0], dtype=np.float64)
        >>> eta = np.array([[1.0, 1.0]], dtype=np.float64)
        >>> krJ0J1 = np.linspace(0.001, 50.0, npt)
        >>> zeta = 1j * 2 * np.pi * 20000.0 * 4e-7 * np.pi
        >>> u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt = common_arrays(
        ...     n, npt, 1.0, krJ0J1, zeta, h, eta
        ... )
        >>> RTEdw[0, 0] == 0.0  # meio infinito inferior
        True

Note:
    A API desta função é **interna** ao simulador Python. Consumidores
    devem usar `geosteering_ai.simulation.simulate()` (Sprint 2.5).
"""
from __future__ import annotations

import logging
from typing import Final

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# DUAL-MODE NUMBA (fallback gracioso se não instalado)
# ──────────────────────────────────────────────────────────────────────────────
# Se Numba estiver instalado e compatível, o decorador `@njit` compila as
# funções via LLVM. Caso contrário, `njit` vira no-op e as funções rodam
# em NumPy puro — correção é preservada, speedup não.
#
# O sinalizador `HAS_NUMBA` é exposto para que os testes possam
# diferenciar os dois modos (ex.: mensagens de skip informativas).
try:
    from numba import njit as _numba_njit

    HAS_NUMBA: Final[bool] = True

    def njit(*args, **kwargs):
        """Wrapper de `numba.njit` com defaults do Sprint 2.1.

        Aplica `cache=True`, `fastmath=False` e
        `error_model='numpy'` se o chamador não os especificar. A
        paridade bit-exata com Fortran `real(dp)` requer aritmética
        IEEE 754 estrita (sem FMA reordering), por isso fastmath fica
        desabilitado por default nesta Sprint. Uma variante `_fast`
        (fastmath=True) pode ser avaliada na Sprint 2.7 junto com o
        benchmark.
        """
        # Se usado como `@njit` (sem parênteses), args[0] é a função.
        if len(args) == 1 and callable(args[0]):
            return _numba_njit(
                cache=True,
                fastmath=False,
                error_model="numpy",
            )(args[0])
        # Uso como `@njit(...)` — aplica os defaults se não sobrescritos.
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", False)
        kwargs.setdefault("error_model", "numpy")
        return _numba_njit(*args, **kwargs)

except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op fallback quando Numba não está disponível.

        Retorna a função original inalterada, permitindo que o módulo
        seja importado e executado em ambientes sem Numba (ex.: CI
        mínimo, máquinas com conflito NumPy×Numba). Performance é
        reduzida mas correção é preservada.
        """
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def wrapper(f):
            return f

        return wrapper


logger = logging.getLogger(__name__)
if not HAS_NUMBA:
    logger.warning(
        "Numba não disponível — propagation.py rodará em NumPy puro "
        "(speedup JIT desabilitado). Instale numba para performance "
        "completa: pip install numba"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES PRIVADAS
# ──────────────────────────────────────────────────────────────────────────────
# Threshold abaixo do qual `hordist` é considerado singular (T e R
# coincidentes). Valor ajustado do Fortran (`eps` em parameters.f08).
_HORDIST_SINGULARITY_EPS: Final[float] = 1.0e-12

# Valor usado como `r` quando hordist é singular. Replica Fortran
# `utils.f08:195` (`r = 1.d-2` → 1 cm). Evita divisão por zero em
# `kr = krJ0J1 / r` sem introduzir erro físico significativo no
# regime quasi-estático (onde kr·r ≪ 1 para frequências LWD).
_HORDIST_SINGULARITY_R: Final[float] = 1.0e-2


# ──────────────────────────────────────────────────────────────────────────────
# COMMON_ARRAYS — pré-calcula 9 arrays invariantes em j
# ──────────────────────────────────────────────────────────────────────────────
@njit
def common_arrays(
    n: int,
    npt: int,
    hordist: float,
    krJ0J1: np.ndarray,
    zeta: complex,
    h: np.ndarray,
    eta: np.ndarray,
):
    """Pré-calcula 9 arrays invariantes no ponto de medida.

    Port Python+Numba de `commonarraysMD` (Fortran `utils.f08:158-241`).
    Produz as quantidades invariantes em `j` (posição do receptor)
    que são reutilizadas pelos kernels `hmd_TIV` e `vmd` na Sprint 2.2.

    Args:
        n: Número de camadas geológicas (int ≥ 1).
        npt: Número de pontos do filtro Hankel (61, 201, 801 ou outro).
        hordist: Distância horizontal entre transmissor e receptor em
            metros. Se menor que ``1e-12``, usa-se
            ``r = 1e-2`` m (replicando Fortran `utils.f08:195`).
        krJ0J1: Array 1D `(npt,)` float64 com abscissas do filtro
            Hankel (lidas de `FilterLoader.load(...).abscissas`).
            Read-only OK.
        zeta: Impeditividade complexa ``i·ω·μ₀`` (convenção temporal
            e^(-iωt)). Para `f=20 kHz`, vale aproximadamente
            ``2π · 20000 · 4π × 1e-7 · j``.
        h: Array 1D `(n,)` float64 com espessuras das camadas em metros.
            Para a última camada (meio infinito inferior), o valor
            geralmente é 0.0 — a recursão não usa ``h[n-1]``.
        eta: Array 2D `(n, 2)` float64 com condutividades: `eta[i, 0]`
            = σₕ (horizontal) e `eta[i, 1]` = σᵥ (vertical) da camada
            i. Para meio isotrópico, σₕ = σᵥ.

    Returns:
        Tupla de 9 `np.ndarray` complex128, shape `(npt, n)`:

        - ``u``      — constante de propagação horizontal (TE),
          `u[:, i] = sqrt(kr² − kh²_i)` com `kh² = −zeta·σh`.
        - ``s``      — constante de propagação vertical escalada
          (TM), `s[:, i] = sqrt(λ_i)·sqrt(kr² − kv²_i)` com
          `kv² = −zeta·σv` e `λ_i = σh_i / σv_i`.
        - ``uh``     — `u[:, i] · h[i]` (produto adimensional).
        - ``sh``     — `s[:, i] · h[i]`.
        - ``RTEdw``  — reflexão TE interfaces inferiores (recursão
          bottom-up). `RTEdw[:, n-1] = 0`.
        - ``RTEup``  — reflexão TE interfaces superiores (recursão
          top-down). `RTEup[:, 0] = 0`.
        - ``RTMdw``  — reflexão TM interfaces inferiores.
        - ``RTMup``  — reflexão TM interfaces superiores.
        - ``AdmInt`` — admitância intrínseca TE, `u / zeta`.

    Raises:
        ValueError: Via asserts em Python puro (Numba descarta asserts
            em modo `@njit`; validação é feita no wrapper externo).

    Note:
        Port line-for-line de `utils.f08:158-241`. Diferenças de
        indexação:

        - Fortran `1:n` → Python `0:n` (0-based)
        - Fortran `eta(i, 1)` → Python `eta[i, 0]` (σh)
        - Fortran `eta(i, 2)` → Python `eta[i, 1]` (σv)

        Os arrays intermediários `v`, `tghuh`, `tghsh`, `AdmApdw`,
        `ImpInt`, `ImpApdw`, `AdmApup`, `ImpApup` são calculados
        internamente mas NÃO retornados. Apenas os 9 arrays usados
        pelos dipolos (`hmd_TIV`, `vmd`) saem da função.

        A forma explícita de `tanh` — `(1 - exp(-2x))/(1 + exp(-2x))` —
        é usada em vez de `np.tanh` para casamento bit-exato com
        Fortran (linha 212).

    Example:
        Meio homogêneo isotrópico (1 camada, σ = 1 S/m)::

            >>> import numpy as np
            >>> n, npt = 1, 201
            >>> h = np.array([0.0], dtype=np.float64)
            >>> eta = np.array([[1.0, 1.0]], dtype=np.float64)
            >>> krJ0J1 = np.linspace(0.001, 50.0, npt)
            >>> zeta = 1j * 2 * np.pi * 20000.0 * 4e-7 * np.pi
            >>> outs = common_arrays(n, npt, 1.0, krJ0J1, zeta, h, eta)
            >>> u, s = outs[0], outs[1]
            >>> np.allclose(s, u)  # isotrópico → s == u
            True
    """
    # ── Guard de singularidade ───────────────────────────────────────
    # Fortran `utils.f08:194-198`: quando r → 0 (T e R coincidentes),
    # kr → ∞ e o forward tem singularidade. O Fortran usa r = 1e-2 m
    # para contornar — replicamos bit-a-bit para preservar paridade.
    if hordist < _HORDIST_SINGULARITY_EPS:
        r = _HORDIST_SINGULARITY_R
    else:
        r = hordist

    # ── Abscissas reescaladas ────────────────────────────────────────
    # kr tem unidades de 1/m (número de onda radial). O filtro Hankel
    # fornece valores adimensionais, divididos por r para obter kr.
    kr = krJ0J1 / r
    kr_squared = kr * kr  # (npt,) float64 — reutilizado em u e s

    # ── Pré-alocação dos 9 arrays de saída ───────────────────────────
    # `np.empty` em vez de `np.zeros` porque todas as células são
    # preenchidas no loop seguinte (performance marginal, mas
    # consistente com o estilo Fortran de `allocate` + assign).
    u = np.empty((npt, n), dtype=np.complex128)
    s = np.empty((npt, n), dtype=np.complex128)
    uh = np.empty((npt, n), dtype=np.complex128)
    sh = np.empty((npt, n), dtype=np.complex128)
    RTEdw = np.empty((npt, n), dtype=np.complex128)
    RTEup = np.empty((npt, n), dtype=np.complex128)
    RTMdw = np.empty((npt, n), dtype=np.complex128)
    RTMup = np.empty((npt, n), dtype=np.complex128)
    AdmInt = np.empty((npt, n), dtype=np.complex128)

    # ── Pré-alocação de intermediários ───────────────────────────────
    # Estes arrays não saem da função, mas são necessários para a
    # recursão de reflexão/transmissão nas etapas 2 e 3.
    ImpInt = np.empty((npt, n), dtype=np.complex128)
    tghuh = np.empty((npt, n), dtype=np.complex128)
    tghsh = np.empty((npt, n), dtype=np.complex128)
    AdmApdw = np.empty((npt, n), dtype=np.complex128)
    ImpApdw = np.empty((npt, n), dtype=np.complex128)
    AdmApup = np.empty((npt, n), dtype=np.complex128)
    ImpApup = np.empty((npt, n), dtype=np.complex128)

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 1: Constantes de propagação por camada (não-recursiva)
    # ──────────────────────────────────────────────────────────────────
    # Para cada camada i, calcula:
    #   u[:, i] = sqrt(kr² + zeta·σh_i)    [constante TE]
    #   v[:, i] = sqrt(kr² + zeta·σv_i)    [vertical intermediário]
    #   s[:, i] = sqrt(λ_i) · v[:, i]       [constante TM escalada]
    #
    # No Fortran, `kh² = -zeta · σh` e `u = sqrt(kr² - kh²) =
    # sqrt(kr² + zeta·σh)`. Esta é a forma que implementamos aqui
    # para preservar bit-exactness (evita `-(-x) = x` com arredondamento).
    for i in range(n):
        # Numbers de onda ao quadrado (complexos, pois zeta é complexo)
        kh2 = -zeta * eta[i, 0]  # -i·ω·μ₀·σh
        kv2 = -zeta * eta[i, 1]  # -i·ω·μ₀·σv
        lamb2 = eta[i, 0] / eta[i, 1]  # σh / σv (lambda²)

        # u e v: raízes principais (Re ≥ 0, Im ≥ 0 garantidos pelo
        # argumento positivo em kr² quasi-estático)
        u[:, i] = np.sqrt(kr_squared - kh2)
        v_col = np.sqrt(kr_squared - kv2)
        s[:, i] = np.sqrt(lamb2) * v_col

        # Admitância e impedância intrínsecas
        AdmInt[:, i] = u[:, i] / zeta
        ImpInt[:, i] = s[:, i] / eta[i, 0]

        # Produtos escalados pela espessura
        uh[:, i] = u[:, i] * h[i]
        sh[:, i] = s[:, i] * h[i]

        # Tangentes hiperbólicas via forma explícita (casamento
        # bit-exato com Fortran line 212).
        exp_m2uh = np.exp(-2.0 * uh[:, i])
        exp_m2sh = np.exp(-2.0 * sh[:, i])
        tghuh[:, i] = (1.0 - exp_m2uh) / (1.0 + exp_m2uh)
        tghsh[:, i] = (1.0 - exp_m2sh) / (1.0 + exp_m2sh)

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 2: Recursão bottom-up para RTEdw, RTMdw, AdmApdw, ImpApdw
    # ──────────────────────────────────────────────────────────────────
    # Terminal: meio infinito inferior (camada n-1) não tem reflexão
    # para baixo — `RTEdw = RTMdw = 0` e `AdmApdw = AdmInt`.
    AdmApdw[:, n - 1] = AdmInt[:, n - 1]
    ImpApdw[:, n - 1] = ImpInt[:, n - 1]
    RTEdw[:, n - 1] = 0.0 + 0.0j
    RTMdw[:, n - 1] = 0.0 + 0.0j

    # Recursão de n-2 até 0, propagando admitância/impedância aparente
    # das camadas inferiores para cima.
    for i in range(n - 2, -1, -1):
        AdmApdw[:, i] = (
            AdmInt[:, i]
            * (AdmApdw[:, i + 1] + AdmInt[:, i] * tghuh[:, i])
            / (AdmInt[:, i] + AdmApdw[:, i + 1] * tghuh[:, i])
        )
        ImpApdw[:, i] = (
            ImpInt[:, i]
            * (ImpApdw[:, i + 1] + ImpInt[:, i] * tghsh[:, i])
            / (ImpInt[:, i] + ImpApdw[:, i + 1] * tghsh[:, i])
        )
        RTEdw[:, i] = (AdmInt[:, i] - AdmApdw[:, i + 1]) / (
            AdmInt[:, i] + AdmApdw[:, i + 1]
        )
        RTMdw[:, i] = (ImpInt[:, i] - ImpApdw[:, i + 1]) / (
            ImpInt[:, i] + ImpApdw[:, i + 1]
        )

    # ──────────────────────────────────────────────────────────────────
    # ETAPA 3: Recursão top-down para RTEup, RTMup, AdmApup, ImpApup
    # ──────────────────────────────────────────────────────────────────
    # Terminal: meio infinito superior (camada 0, topo do modelo) não
    # tem reflexão para cima — `RTEup = RTMup = 0` e `AdmApup = AdmInt`.
    AdmApup[:, 0] = AdmInt[:, 0]
    ImpApup[:, 0] = ImpInt[:, 0]
    RTEup[:, 0] = 0.0 + 0.0j
    RTMup[:, 0] = 0.0 + 0.0j

    # Recursão de 1 até n-1, propagando admitância aparente das
    # camadas superiores para baixo.
    for i in range(1, n):
        AdmApup[:, i] = (
            AdmInt[:, i]
            * (AdmApup[:, i - 1] + AdmInt[:, i] * tghuh[:, i])
            / (AdmInt[:, i] + AdmApup[:, i - 1] * tghuh[:, i])
        )
        ImpApup[:, i] = (
            ImpInt[:, i]
            * (ImpApup[:, i - 1] + ImpInt[:, i] * tghsh[:, i])
            / (ImpInt[:, i] + ImpApup[:, i - 1] * tghsh[:, i])
        )
        RTEup[:, i] = (AdmInt[:, i] - AdmApup[:, i - 1]) / (
            AdmInt[:, i] + AdmApup[:, i - 1]
        )
        RTMup[:, i] = (ImpInt[:, i] - ImpApup[:, i - 1]) / (
            ImpInt[:, i] + ImpApup[:, i - 1]
        )

    return (u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt)


# ──────────────────────────────────────────────────────────────────────────────
# COMMON_FACTORS — calcula 6 fatores de onda refletida
# ──────────────────────────────────────────────────────────────────────────────
@njit
def common_factors(
    n: int,
    npt: int,
    h0: float,
    h: np.ndarray,
    prof: np.ndarray,
    camad_t: int,
    u: np.ndarray,
    s: np.ndarray,
    uh: np.ndarray,
    sh: np.ndarray,
    RTEdw: np.ndarray,
    RTEup: np.ndarray,
    RTMdw: np.ndarray,
    RTMup: np.ndarray,
):
    """Calcula 6 fatores de onda refletida na camada do transmissor.

    Port Python+Numba de `commonfactorsMD` (Fortran `utils.f08:243-297`).
    Produz os 6 fatores de onda (Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)
    que são consumidos pelos dipolos magnéticos na Sprint 2.2.

    Args:
        n: Número total de camadas.
        npt: Número de pontos do filtro Hankel.
        h0: Profundidade do transmissor em metros (0 = topo do modelo,
            positivo para baixo). Se negativa, indica transmissor no
            ar (caso raro em LWD).
        h: Array 1D `(n,)` float64 com espessuras das camadas.
        prof: Array 1D `(n+1,)` float64 com profundidades das
            interfaces. `prof[0]` é o topo da camada 0 (geralmente 0.0)
            e `prof[i+1]` é o fundo da camada i (= topo da camada i+1).
        camad_t: Índice 0-based da camada onde o transmissor está
            localizado. Deve estar em `[0, n-1]`.
        u, s, uh, sh: Arrays `(npt, n)` complex128 produzidos por
            :func:`common_arrays`.
        RTEdw, RTEup, RTMdw, RTMup: Coeficientes de reflexão TE/TM
            produzidos por :func:`common_arrays`.

    Returns:
        Tupla de 6 `np.ndarray` complex128, shape `(npt,)`:

        - ``Mxdw``  — fator TM da onda refletida pelas camadas inferiores
          (potencial πₓ).
        - ``Mxup``  — fator TM da onda refletida pelas camadas superiores.
        - ``Eudw``  — fator TE da onda refletida pelas camadas inferiores
          (potencial πᵤ).
        - ``Euup``  — fator TE da onda refletida pelas camadas superiores.
        - ``FEdwz`` — fator TE adicional usado na modelagem com DMV
          (dipolo magnético vertical).
        - ``FEupz`` — fator TE complementar para DMV.

    Note:
        Port line-for-line de `utils.f08:243-297`. Diferenças de
        indexação:

        - Fortran `camadT` (1-based) → Python `camad_t` (0-based)
        - Fortran `prof(camadT-1)` → Python `prof[camad_t]` (topo da
          camada ``camad_t``)
        - Fortran `prof(camadT)` → Python `prof[camad_t + 1]` (fundo
          da camada ``camad_t``)
        - Fortran `h(camadT)` → Python `h[camad_t]` (espessura da
          camada do transmissor)

        Os denominadores `den_TM` e `den_TE` devem ser não-nulos
        para todo kr, o que é garantido pela condição física
        ``|RTM*|·|RT*| < 1`` (soma de reflexões convergente).
    """
    # ── Extrai colunas da camada do transmissor ─────────────────────
    # Acesso frequente a estas colunas — cacheamos em variáveis locais
    # para reduzir indexação redundante (ajuda o compilador Numba).
    u_t = u[:, camad_t]  # (npt,) complex128
    s_t = s[:, camad_t]
    uh_t = uh[:, camad_t]  # não usado diretamente, mas sh_t sim
    sh_t = sh[:, camad_t]
    RTEdw_t = RTEdw[:, camad_t]
    RTEup_t = RTEup[:, camad_t]
    RTMdw_t = RTMdw[:, camad_t]
    RTMup_t = RTMup[:, camad_t]

    # ── Distâncias relativas à camada do transmissor ────────────────
    # `top_t` é a profundidade do topo da camada do transmissor;
    # `bot_t` é o fundo. O offset `h0` é a profundidade do próprio
    # transmissor. `h_t` é a espessura da camada.
    top_t = prof[camad_t]  # prof(camadT-1) no Fortran
    bot_t = prof[camad_t + 1]  # prof(camadT)   no Fortran
    h_t = h[camad_t]  # h(camadT)      no Fortran

    # ──────────────────────────────────────────────────────────────────
    # BLOCO TM (Mxdw, Mxup)
    # ──────────────────────────────────────────────────────────────────
    # `den_TM = 1 - RTMdw·RTMup·exp(-2·sh)` é o denominador da soma
    # geométrica infinita de reflexões entre as interfaces superior e
    # inferior da camada do transmissor. Para |RTM·RTM·exp| < 1,
    # converge.
    den_TM = 1.0 - RTMdw_t * RTMup_t * np.exp(-2.0 * sh_t)

    Mxdw = (
        np.exp(-s_t * (bot_t - h0)) + RTMup_t * np.exp(s_t * (top_t - h0 - h_t))
    ) / den_TM

    Mxup = (
        np.exp(s_t * (top_t - h0)) + RTMdw_t * np.exp(-s_t * (bot_t - h0 + h_t))
    ) / den_TM

    # ──────────────────────────────────────────────────────────────────
    # BLOCO TE (Eudw, Euup) — note que o sinal da 2ª parcela é NEGATIVO
    # ──────────────────────────────────────────────────────────────────
    # A diferença de sinal (subtração em vez de adição) no bloco TE
    # vem da convenção do potencial πᵤ vs πₓ. Ver Moran-Gianzero 1979
    # §III para derivação.
    den_TE = 1.0 - RTEdw_t * RTEup_t * np.exp(-2.0 * uh_t)

    Eudw = (
        np.exp(-u_t * (bot_t - h0)) - RTEup_t * np.exp(u_t * (top_t - h0 - h_t))
    ) / den_TE

    Euup = (
        np.exp(u_t * (top_t - h0)) - RTEdw_t * np.exp(-u_t * (bot_t - h0 + h_t))
    ) / den_TE

    # ──────────────────────────────────────────────────────────────────
    # BLOCO FE (FEdwz, FEupz) — usados apenas na modelagem com DMV
    # ──────────────────────────────────────────────────────────────────
    # A diferença frente ao bloco E é o sinal + em vez de -. São
    # fatores adicionais para o dipolo magnético vertical (VMD).
    FEdwz = (
        np.exp(-u_t * (bot_t - h0)) + RTEup_t * np.exp(u_t * (top_t - h_t - h0))
    ) / den_TE

    FEupz = (
        np.exp(u_t * (top_t - h0)) + RTEdw_t * np.exp(-u_t * (bot_t + h_t - h0))
    ) / den_TE

    return (Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)


__all__ = [
    "HAS_NUMBA",
    "common_arrays",
    "common_factors",
]
