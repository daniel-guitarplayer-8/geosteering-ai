# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/validation/half_space.py                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Soluções Analíticas de Referência       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 1.3)                                   ║
# ║  Status      : Produção                                                  ║
# ║  Framework   : NumPy 2.x                                                  ║
# ║  Dependências: numpy                                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementa soluções analíticas fechadas (closed-form) para 5 casos  ║
# ║    canônicos de EM 1D, que servem como ground-truth para validar os    ║
# ║    backends numéricos (Numba na Fase 2, JAX na Fase 3) do simulador.  ║
# ║                                                                           ║
# ║  CASOS IMPLEMENTADOS                                                      ║
# ║    ┌────────────────────────────┬─────────────────────────────────────┐  ║
# ║    │  Caso                       │  Referência bibliográfica          │  ║
# ║    ├────────────────────────────┼─────────────────────────────────────┤  ║
# ║    │  1. Decoupling estático     │  CLAUDE.md errata + textbook       │  ║
# ║    │  2. Skin depth homogêneo    │  Nabighian (1988) eq. 1.4          │  ║
# ║    │  3. Número de onda quasi-   │  Ward & Hohmann (1988) eq. 1.17    │  ║
# ║    │     estático                │                                     │  ║
# ║    │  4. VMD full-space axial    │  Ward & Hohmann (1988) eq. 2.56    │  ║
# ║    │  5. VMD full-space          │  Ward & Hohmann (1988) eq. 2.57    │  ║
# ║    │     broadside               │                                     │  ║
# ║    └────────────────────────────┴─────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  CONVENÇÕES (críticas para reproducibilidade)                            ║
# ║    • Convenção temporal: e^(-iωt) — padrão em geofísica e em           ║
# ║      Moran-Gianzero (1979), que é a referência do simulador Fortran. ║
# ║    • k² = iωμ₀σ  (quasi-estática, válida para ω ≪ σ/ε₀)               ║
# ║    • k = √(iωμ₀σ) com parte imaginária positiva                       ║
# ║    • Fator de propagação e^(ikr): com Im(k) > 0 → atenuação com r    ║
# ║      (campo decai exponencialmente para dentro do conductor)          ║
# ║    • μ₀ = 4π × 10⁻⁷ H/m (permeabilidade magnética do vácuo)           ║
# ║    • Resistividade ρ [Ω·m] → condutividade σ [S/m] = 1/ρ              ║
# ║                                                                           ║
# ║  USO TÍPICO EM TESTES                                                     ║
# ║    Os casos 1-5 são invocados pelos testes em                           ║
# ║    `tests/test_simulation_half_space.py` para verificar propriedades   ║
# ║    físicas fundamentais (sinal, decaimento, limite estático, etc.).   ║
# ║    Nas Fases 2-3, os backends Numba/JAX serão comparados contra      ║
# ║    estes valores com tolerância < 1e-10 (float64).                    ║
# ║                                                                           ║
# ║  DESIGN                                                                   ║
# ║    • Todas as funções são **puras** (sem estado global).                ║
# ║    • Entradas aceitam escalar ou `np.ndarray` (broadcast automático).  ║
# ║    • Saídas em `complex128` para os campos, `float64` para escalares. ║
# ║    • Validação de argumentos via `assert` (fail-fast).                  ║
# ║                                                                           ║
# ║  LIMITAÇÕES CONHECIDAS                                                   ║
# ║    Apenas casos ISOTRÓPICOS implementados nesta Sprint 1.3. O full-   ║
# ║    space TIV (anisotrópico) requer decomposição em modos TE/TM e      ║
# ║    será adicionado na Sprint 4.x junto com os testes de validação     ║
# ║    cruzada Numba ↔ JAX ↔ Fortran ↔ empymod.                           ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Ward, S.H. & Hohmann, G.W. (1988). "Electromagnetic theory for    ║
# ║      geophysical applications." In: Electromagnetic Methods in         ║
# ║      Applied Geophysics, Vol. 1, SEG.                                  ║
# ║    • Moran, J.H. & Gianzero, S. (1979). "Effects of formation          ║
# ║      anisotropy on resistivity-logging measurements." Geophysics 44. ║
# ║    • Nabighian, M.N. (1988). "Electromagnetic Methods in Applied      ║
# ║      Geophysics — Theory", Vol. 1, SEG.                                ║
# ║    • CLAUDE.md (errata imutável com valores ACp e ACx).                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Soluções analíticas fechadas para casos canônicos de EM 1D.

Este módulo expõe cinco funções puras em NumPy que computam soluções
analíticas fechadas para cenários canônicos de propagação EM 1D em
meios homogêneos isotrópicos:

- :func:`static_decoupling_factors` — Constantes `ACp` e `ACx` (limite
  estático, ω → 0).
- :func:`skin_depth` — Profundidade de penetração δ em meio homogêneo.
- :func:`wavenumber_quasi_static` — Número de onda complexo k no limite
  quasi-estático.
- :func:`vmd_fullspace_axial` — Campo H de VMD em full-space isotrópico,
  observado no eixo do dipolo (geometria axial).
- :func:`vmd_fullspace_broadside` — Campo H de VMD em full-space
  isotrópico, observado perpendicular ao eixo (geometria broadside).

As funções são usadas como **ground-truth independente** para validar
os backends numéricos do simulador Python (Numba na Fase 2, JAX na
Fase 3). Um teste que compara o solver consigo mesmo não detecta bugs;
uma comparação contra formulação analítica externa sim.

Example:
    Verificação das constantes de decoupling com L=1.0 m::

        >>> import numpy as np
        >>> from geosteering_ai.simulation.validation import (
        ...     static_decoupling_factors,
        ... )
        >>> ACp, ACx = static_decoupling_factors(L=1.0)
        >>> np.isclose(ACp, -0.07957747154594767)
        True
        >>> np.isclose(ACx, 0.15915494309189535)
        True

Note:
    Todas as fórmulas usam a convenção temporal **e^(-iωt)** (padrão
    geofísica, Moran-Gianzero 1979). Para converter para e^(+iωt)
    (padrão Ward-Hohmann / engenharia), substitua ``i → -i`` em todas
    as expressões complexas.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS
# ──────────────────────────────────────────────────────────────────────────────
# Permeabilidade magnética do vácuo. Usada em todas as fórmulas EM para
# conversão entre condutividade σ e número de onda complexo k. Valor
# exato definido pelo SI: μ₀ = 4π × 10⁻⁷ H/m. Não é aproximado.
MU_0: float = 4.0 * np.pi * 1.0e-7

# Tipo para entradas que aceitam escalar ou array (broadcast-friendly).
_ArrayLike = float | NDArray[np.float64]


# ──────────────────────────────────────────────────────────────────────────────
# CASO 1: DECOUPLING FACTORS (LIMITE ESTÁTICO σ → 0)
# ──────────────────────────────────────────────────────────────────────────────
def static_decoupling_factors(
    L: float,
) -> tuple[float, float]:
    """Retorna as constantes de decoupling magnético no limite estático.

    No limite de frequência zero (ou condutividade zero), o campo
    magnético de um dipolo em vácuo (ou meio isolante) reduz-se a uma
    função puramente geométrica de 1/L³. Os coeficientes que
    multiplicam esse fator são chamados de "decoupling factors" no
    contexto de ferramentas LWD indução:

    **ACp** (planar/broadside, aplicável a Hxx e Hyy):

    .. math::
        \\text{ACp} = -\\frac{1}{4\\pi L^3}

    **ACx** (axial, aplicável a Hzz):

    .. math::
        \\text{ACx} = +\\frac{1}{2\\pi L^3}

    Para L = 1 m, os valores numéricos são aproximadamente
    ACp ≈ -0.079577 e ACx ≈ +0.159155.

    Estes valores servem como normalização das medidas brutas no
    simulador Fortran (`tatu.x`), sendo subtraídos (ou divididos)
    para isolar o sinal geológico sobre o background da ferramenta.
    Estão registrados como "Errata Imutável" no ``CLAUDE.md`` do
    projeto.

    Args:
        L: Espaçamento transmissor-receptor em metros. Deve ser > 0.

    Returns:
        Tupla ``(ACp, ACx)`` com os dois fatores em unidades SI
        (1/m³ × número-puro).

    Raises:
        AssertionError: Se ``L <= 0``.

    Example:
        >>> ACp, ACx = static_decoupling_factors(L=1.0)
        >>> round(ACp, 6), round(ACx, 6)
        (-0.079577, 0.159155)

        >>> # Conferência: a razão ACx / (-ACp) sempre é 2.
        >>> round(ACx / (-ACp), 6)
        2.0

    Note:
        Sinal negativo de ACp é convenção **geofísica** (campo de dipolo
        broadside aponta em direção oposta ao vetor-momento). Não é um
        erro — reflete a convenção adotada pelo simulador Fortran e
        pelos modelos do pipeline v2.0.
    """
    assert L > 0, f"L={L} m inválido; espaçamento TR deve ser > 0."
    L3 = L**3
    ACp = -1.0 / (4.0 * np.pi * L3)
    ACx = +1.0 / (2.0 * np.pi * L3)
    return ACp, ACx


# ──────────────────────────────────────────────────────────────────────────────
# CASO 2: SKIN DEPTH (PROFUNDIDADE DE PENETRAÇÃO)
# ──────────────────────────────────────────────────────────────────────────────
def skin_depth(
    frequency_hz: _ArrayLike,
    resistivity_ohm_m: _ArrayLike,
) -> _ArrayLike:
    """Profundidade de penetração (skin depth) em meio homogêneo.

    Em um meio condutor homogêneo isotrópico, uma onda EM plana
    atenua-se exponencialmente com a distância. A profundidade na qual
    a amplitude cai para 1/e é chamada "skin depth":

    .. math::
        \\delta = \\sqrt{\\frac{2}{\\omega \\mu_0 \\sigma}}
               = \\sqrt{\\frac{\\rho}{\\pi f \\mu_0}}

    onde ω = 2πf, σ = 1/ρ.

    Aproximação prática (geofísica):

    .. math::
        \\delta \\approx 503.3 \\cdot \\sqrt{\\rho / f}  \\quad [m]

    Args:
        frequency_hz: Frequência da onda EM em Hertz. Escalar ou array.
            Deve ser > 0.
        resistivity_ohm_m: Resistividade do meio em Ω·m. Escalar ou
            array. Deve ser > 0.

    Returns:
        Profundidade de penetração δ em metros. Mesmo shape que a
        entrada (broadcast automático).

    Raises:
        AssertionError: Se qualquer frequência ou resistividade for ≤ 0.

    Example:
        >>> round(skin_depth(frequency_hz=20000.0, resistivity_ohm_m=1.0), 2)
        3.56
        >>> round(skin_depth(frequency_hz=20000.0, resistivity_ohm_m=100.0), 2)
        35.59

    Note:
        Válido somente no limite quasi-estático (ωε₀ ≪ σ), que para
        ρ < 1000 Ω·m é satisfeito até frequências da ordem de 1 MHz.
        Acima disso, efeitos de deslocamento (displacement current)
        tornam-se importantes e esta fórmula precisa de correção.
        Ref: Nabighian (1988), eq. 1.14-1.15.
    """
    freq = np.asarray(frequency_hz, dtype=np.float64)
    rho = np.asarray(resistivity_ohm_m, dtype=np.float64)
    assert np.all(freq > 0), f"frequency_hz deve ser > 0, recebeu {freq}"
    assert np.all(rho > 0), f"resistivity_ohm_m deve ser > 0, recebeu {rho}"

    # δ = √(ρ / (π f μ₀))
    delta = np.sqrt(rho / (np.pi * freq * MU_0))

    # Preserva o tipo escalar na saída se ambas as entradas forem
    # escalares (ergonomia — evita retornar array 0-d).
    if delta.ndim == 0:
        return float(delta)
    return delta


# ──────────────────────────────────────────────────────────────────────────────
# CASO 3: NÚMERO DE ONDA QUASI-ESTÁTICO
# ──────────────────────────────────────────────────────────────────────────────
def wavenumber_quasi_static(
    frequency_hz: float,
    resistivity_ohm_m: float,
) -> complex:
    """Número de onda complexo k no limite quasi-estático.

    No limite ω ≪ σ/ε₀ (corrente de deslocamento desprezível), o
    número de onda EM reduz-se a:

    .. math::
        k^2 = i \\omega \\mu_0 \\sigma = \\frac{i \\omega \\mu_0}{\\rho}

    com a **convenção temporal e^(-iωt)** (padrão geofísica). Extraindo
    a raiz principal (parte imaginária positiva):

    .. math::
        k = (1 + i) / (\\delta \\sqrt{2}) \\cdot \\sqrt{2}
          = \\frac{1 + i}{\\delta}

    onde δ é o skin depth. Note que ``Im(k) > 0`` — isso garante que
    ``e^(ikr)`` decai com r crescente (atenuação para dentro do
    condutor), que é o comportamento físico esperado.

    Args:
        frequency_hz: Frequência em Hz (escalar > 0).
        resistivity_ohm_m: Resistividade em Ω·m (escalar > 0).

    Returns:
        Número de onda k como número complexo Python (complex128).

    Raises:
        AssertionError: Se qualquer argumento for ≤ 0.

    Example:
        >>> k = wavenumber_quasi_static(20000.0, 1.0)
        >>> round(k.real, 4), round(k.imag, 4)
        (0.2812, 0.2812)
        >>> k.imag > 0  # garante decaimento com r
        True

    Note:
        A escolha da raiz principal (Im(k) > 0 em vez de < 0) determina
        o sinal do decaimento. Em convenção e^(+iωt) (Ward-Hohmann), a
        raiz principal seria Im(k) < 0, e o fator seria e^(-ikr).
        Ambas convenções dão o mesmo campo físico — só mudam os sinais.
    """
    assert frequency_hz > 0, f"frequency_hz={frequency_hz} deve ser > 0"
    assert resistivity_ohm_m > 0, f"resistivity_ohm_m={resistivity_ohm_m} deve ser > 0"

    omega = 2.0 * np.pi * frequency_hz
    sigma = 1.0 / resistivity_ohm_m

    # k² = iωμ₀σ (convenção e^(-iωt))
    k_squared = 1j * omega * MU_0 * sigma
    # Raiz principal com Im(k) > 0 (numpy.sqrt de complexo já seleciona
    # a raiz com argumento em [-π/2, π/2], o que para k² = i·valor_real
    # positivo dá k no primeiro quadrante: Re(k) > 0 e Im(k) > 0).
    k = np.sqrt(k_squared)
    return complex(k)


# ──────────────────────────────────────────────────────────────────────────────
# CASO 4: VMD FULL-SPACE AXIAL (θ = 0)
# ──────────────────────────────────────────────────────────────────────────────
def vmd_fullspace_axial(
    L: float,
    frequency_hz: float,
    resistivity_ohm_m: float,
    moment_Am2: float = 1.0,
) -> complex:
    """Campo H de VMD em full-space isotrópico, observação axial.

    Dipolo magnético vertical (VMD) de momento ``m`` posicionado na
    origem, alinhado com o eixo z. Observação em ``(0, 0, L)`` com
    ``L > 0``. A componente Hz neste ponto, em meio isotrópico
    homogêneo de resistividade ``ρ``, é dada por (Ward & Hohmann
    1988, eq. 2.56, convertida para e^(-iωt)):

    .. math::
        H_z(L) = \\frac{m}{2\\pi L^3} \\cdot (1 - i k L) \\cdot e^{i k L}

    onde ``k`` é o número de onda quasi-estático. No limite ``k → 0``
    (σ → 0 ou ω → 0), esta expressão reduz-se ao fator de decoupling
    ``ACx·m = m/(2π L³)``, que é exatamente o valor computado por
    :func:`static_decoupling_factors`.

    Args:
        L: Distância transmissor-receptor em metros (L > 0, axial).
        frequency_hz: Frequência em Hz (> 0).
        resistivity_ohm_m: Resistividade em Ω·m (> 0).
        moment_Am2: Momento magnético do dipolo em A·m². Default 1.0.

    Returns:
        Campo Hz complexo em A/m (moment-normalized se m=1).

    Raises:
        AssertionError: Se qualquer argumento (exceto momento) for ≤ 0.

    Example:
        >>> H = vmd_fullspace_axial(L=1.0, frequency_hz=20000.0,
        ...                         resistivity_ohm_m=1.0)
        >>> abs(H) > 0
        True

        >>> # Limite estático: frequência muito baixa → ACx × m
        >>> H_static = vmd_fullspace_axial(
        ...     L=1.0, frequency_hz=1e-6, resistivity_ohm_m=1.0)
        >>> _, ACx = static_decoupling_factors(L=1.0)
        >>> np.isclose(H_static.real, ACx, atol=1e-8)
        True

    Note:
        A fórmula pressupõe meio **isotrópico** (ρh = ρv). Para meio
        TIV (ρh ≠ ρv), a expressão contém termos adicionais e será
        implementada na Sprint 4.x junto com os testes de validação
        cruzada contra empymod e Fortran.
    """
    assert L > 0, f"L={L} m inválido; axial exige L > 0."
    assert frequency_hz > 0, f"frequency_hz={frequency_hz} deve ser > 0"
    assert resistivity_ohm_m > 0, f"resistivity_ohm_m={resistivity_ohm_m} deve ser > 0"

    k = wavenumber_quasi_static(frequency_hz, resistivity_ohm_m)
    kL = k * L
    prefactor = moment_Am2 / (2.0 * np.pi * L**3)
    # H_z = (m / 2πL³) · (1 - ikL) · e^(ikL)
    H = prefactor * (1.0 - 1j * kL) * np.exp(1j * kL)
    return complex(H)


# ──────────────────────────────────────────────────────────────────────────────
# CASO 5: VMD FULL-SPACE BROADSIDE (θ = π/2)
# ──────────────────────────────────────────────────────────────────────────────
def vmd_fullspace_broadside(
    L: float,
    frequency_hz: float,
    resistivity_ohm_m: float,
    moment_Am2: float = 1.0,
) -> complex:
    """Campo H de VMD em full-space isotrópico, observação broadside.

    Dipolo magnético vertical (VMD) de momento ``m`` na origem, eixo
    z. Observação em ``(L, 0, 0)`` (perpendicular ao eixo do dipolo).
    A componente que aponta na direção z neste ponto é (Ward &
    Hohmann 1988, eq. 2.57 com e^(-iωt)):

    .. math::
        H_z^{\\text{broad}}(L) = -\\frac{m}{4\\pi L^3}
            \\cdot (1 - ikL - (kL)^2) \\cdot e^{ikL}

    No limite ``k → 0``, esta expressão reduz-se a ``ACp·m = -m/(4π L³)``,
    que é exatamente o valor computado por
    :func:`static_decoupling_factors` (primeiro elemento do par).

    Args:
        L: Distância do receptor ao eixo do dipolo em metros (L > 0).
        frequency_hz: Frequência em Hz (> 0).
        resistivity_ohm_m: Resistividade em Ω·m (> 0).
        moment_Am2: Momento magnético em A·m². Default 1.0.

    Returns:
        Campo Hz complexo em A/m.

    Raises:
        AssertionError: Se L, frequency_hz ou resistivity forem ≤ 0.

    Example:
        >>> H = vmd_fullspace_broadside(
        ...     L=1.0, frequency_hz=20000.0, resistivity_ohm_m=1.0)
        >>> abs(H) > 0
        True

        >>> # Limite estático: frequência muito baixa → ACp × m
        >>> H_static = vmd_fullspace_broadside(
        ...     L=1.0, frequency_hz=1e-6, resistivity_ohm_m=1.0)
        >>> ACp, _ = static_decoupling_factors(L=1.0)
        >>> np.isclose(H_static.real, ACp, atol=1e-8)
        True

    Note:
        O fator ``(1 - ikL - (kL)²)`` tem 3 termos (diferente do axial
        que tem 2): o termo ``(kL)²`` domina em altas frequências e é
        responsável pela dependência ~ω² do sinal secundário em
        ferramentas indução LWD. A razão ``broadside / axial`` é
        um dos observáveis mais úteis para inversão anisotrópica.
    """
    assert L > 0, f"L={L} m inválido; broadside exige L > 0."
    assert frequency_hz > 0, f"frequency_hz={frequency_hz} deve ser > 0"
    assert resistivity_ohm_m > 0, f"resistivity_ohm_m={resistivity_ohm_m} deve ser > 0"

    k = wavenumber_quasi_static(frequency_hz, resistivity_ohm_m)
    kL = k * L
    prefactor = -moment_Am2 / (4.0 * np.pi * L**3)
    # H_z^broad = -(m / 4πL³) · (1 - ikL - (kL)²) · e^(ikL)
    H = prefactor * (1.0 - 1j * kL - kL**2) * np.exp(1j * kL)
    return complex(H)


__all__ = [
    "MU_0",
    "skin_depth",
    "static_decoupling_factors",
    "vmd_fullspace_axial",
    "vmd_fullspace_broadside",
    "wavenumber_quasi_static",
]
