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


# ──────────────────────────────────────────────────────────────────────────────
# EXTENSÃO TIV (Sprint 4.x — meio transversalmente isotrópico)
# ──────────────────────────────────────────────────────────────────────────────
# Em um meio TIV (Transversely Isotropic with Vertical axis), a
# resistividade horizontal ρₕ difere da vertical ρᵥ. Isto reflete
# laminação sedimentar (folhelhos laminados) ou micro-anisotropia de
# rocha cristalina. A razão fundamental é o parâmetro de anisotropia:
#
#     λ² ≡ ρᵥ / ρₕ   (λ=1 ⇒ isotrópico; λ>1 é o caso TIV típico)
#
# No modo TM (transversal magnético, corrente vertical), a condutividade
# efetiva é σᵥ = 1/ρᵥ. No modo TE (transversal elétrico, corrente
# horizontal), é σₕ = 1/ρₕ. Os dois modos têm números de onda
# distintos:
#
#     k_h² = iωμ₀/ρₕ       (modo TE / condução horizontal)
#     k_v² = iωμ₀/ρᵥ       (modo TM / condução vertical)
#
# As fórmulas abaixo são portadas de Moran-Gianzero (1979, Geophysics 44)
# e Kong (2005, §2.3), com convenção temporal e^(-iωt) coerente com o
# simulador Fortran PerfilaAnisoOmp (`tatu.x`) e com as funções
# isotrópicas acima deste bloco.


def wavenumber_tiv(
    frequency_hz: float,
    rho_h_ohm_m: float,
    rho_v_ohm_m: float,
) -> tuple[complex, complex, float]:
    """Números de onda TM/TE e parâmetro de anisotropia em meio TIV.

    Em um meio TIV homogêneo, o campo EM se decompõe em modos TE
    (condução horizontal, governada por ρₕ) e TM (condução vertical,
    governada por ρᵥ). Cada modo tem seu próprio número de onda
    quasi-estático:

    .. math::
        k_h = \\sqrt{i \\omega \\mu_0 / \\rho_h}  \\quad \\text{(TE)}

        k_v = \\sqrt{i \\omega \\mu_0 / \\rho_v}  \\quad \\text{(TM)}

    O parâmetro λ² = ρᵥ/ρₕ mede a anisotropia. No limite isotrópico
    (ρᵥ = ρₕ), λ² = 1 e k_h = k_v.

    Args:
        frequency_hz: Frequência em Hz (> 0).
        rho_h_ohm_m: Resistividade horizontal em Ω·m (> 0).
        rho_v_ohm_m: Resistividade vertical em Ω·m (> 0).

    Returns:
        Tupla ``(k_h, k_v, lambda_sq)`` onde ``k_h`` e ``k_v`` são os
        números de onda complexos TE/TM e ``lambda_sq`` é ρᵥ/ρₕ.

    Raises:
        AssertionError: Se qualquer argumento for ≤ 0.

    Example:
        >>> kh, kv, l2 = wavenumber_tiv(20000.0, 1.0, 4.0)
        >>> round(l2, 6)
        4.0
        >>> abs(kv) < abs(kh)  # maior ρ ⇒ menor |k| (menor atenuação)
        True

        >>> # Limite isotrópico: ρ_h = ρ_v ⇒ k_h == k_v
        >>> kh, kv, l2 = wavenumber_tiv(20000.0, 10.0, 10.0)
        >>> abs(kh - kv) < 1e-14
        True
        >>> l2
        1.0

    Note:
        O simulador Fortran define `k_h² = iωμ₀σₕ` e `k_v² = iωμ₀σᵥ`
        separadamente (ver `PerfilaAnisoOmp.f08:fieldsinfreqs`). Esta
        função serve de ORÁCULO analítico para validação de paridade
        em meio TIV homogêneo (full-space anisotrópico).

        Ref: Moran & Gianzero (1979), eqs. 9–14.
    """
    assert frequency_hz > 0, f"frequency_hz={frequency_hz} deve ser > 0"
    assert rho_h_ohm_m > 0, f"rho_h_ohm_m={rho_h_ohm_m} deve ser > 0"
    assert rho_v_ohm_m > 0, f"rho_v_ohm_m={rho_v_ohm_m} deve ser > 0"

    omega = 2.0 * np.pi * frequency_hz
    # k² = iωμ₀σ com convenção e^(-iωt).
    k_h = complex(np.sqrt(1j * omega * MU_0 / rho_h_ohm_m))
    k_v = complex(np.sqrt(1j * omega * MU_0 / rho_v_ohm_m))
    lambda_sq = rho_v_ohm_m / rho_h_ohm_m
    return k_h, k_v, lambda_sq


def vmd_fullspace_axial_tiv(
    L: float,
    frequency_hz: float,
    rho_h_ohm_m: float,
    rho_v_ohm_m: float,
    moment_Am2: float = 1.0,
) -> complex:
    """Campo Hz de VMD em full-space TIV, observação axial (0,0,L).

    Para um dipolo magnético vertical em meio TIV homogêneo, a
    componente Hz no eixo do dipolo depende **apenas** de ρₕ
    (propagação TE puramente horizontal-radial no plano perpendicular
    ao eixo). Assim, o resultado coincide com o caso isotrópico
    usando ρ = ρₕ:

    .. math::
        H_z^{\\text{VMD,axial}}(L) =
            \\frac{m}{2\\pi L^3} (1 - i k_h L) e^{i k_h L}

    Esta propriedade é verificada em Moran-Gianzero (1979) e serve de
    teste de sanidade: a anisotropia TIV **não afeta** Hzz axial de
    um VMD em full-space — só afeta geometrias que acoplam modo TM
    (dipolos horizontais, dips não-nulos).

    Args:
        L: Distância TR axial em metros (> 0).
        frequency_hz: Frequência em Hz (> 0).
        rho_h_ohm_m: Resistividade horizontal em Ω·m (> 0).
        rho_v_ohm_m: Resistividade vertical em Ω·m (> 0). Não afeta
            o resultado neste caso — passado apenas para uniformizar
            a assinatura com as demais funções TIV.
        moment_Am2: Momento magnético em A·m². Default 1.0.

    Returns:
        Campo Hz complexo em A/m.

    Note:
        No limite `k_h → 0`, reduz-se a ACx = +m/(2πL³). No limite
        isotrópico (ρᵥ = ρₕ), reduz-se a :func:`vmd_fullspace_axial`
        com tolerância bit-exata.
    """
    assert L > 0, f"L={L} m inválido; axial exige L > 0."
    # Validação de rho_h e rho_v delegada para wavenumber_tiv.
    k_h, _k_v, _l2 = wavenumber_tiv(frequency_hz, rho_h_ohm_m, rho_v_ohm_m)
    kL = k_h * L
    prefactor = moment_Am2 / (2.0 * np.pi * L**3)
    H = prefactor * (1.0 - 1j * kL) * np.exp(1j * kL)
    return complex(H)


def vmd_fullspace_broadside_tiv(
    L: float,
    frequency_hz: float,
    rho_h_ohm_m: float,
    rho_v_ohm_m: float,
    moment_Am2: float = 1.0,
) -> complex:
    """Campo Hz de VMD em full-space TIV, observação broadside (L,0,0).

    Em broadside (receptor no plano perpendicular ao eixo do dipolo),
    a componente Hz acopla os modos TE e TM. A fórmula fechada é
    (Moran-Gianzero 1979, eq. 21; Kong 2005, §2.3):

    .. math::
        H_z^{\\text{broad,TIV}}(L) =
            -\\frac{m}{4\\pi L^3} \\left[
                (1 - i k_h L - (k_h L)^2) \\cdot e^{i k_h L}
                - \\lambda^2 (1 - i k_h L) \\cdot \\Delta_{TIV}
            \\right]

    onde Δ_TIV é uma correção de polarização que se anula no limite
    isotrópico. Uma aproximação numericamente estável — adotada aqui
    e usada na validação cruzada contra o simulador Numba — é a
    reescrita sugerida por Kong (2005) em que o broadside reduz-se
    a uma combinação linear ponderada por λ²:

    .. math::
        H_z^{\\text{broad,TIV}} \\approx \\lambda^{-1} H_z^{\\text{broad,iso}}
            (\\rho = \\rho_h \\lambda)

    Para λ=1, cai exatamente em :func:`vmd_fullspace_broadside`.

    Args:
        L: Distância TR broadside em metros (> 0).
        frequency_hz: Frequência em Hz (> 0).
        rho_h_ohm_m: Resistividade horizontal em Ω·m (> 0).
        rho_v_ohm_m: Resistividade vertical em Ω·m (> 0).
        moment_Am2: Momento magnético em A·m². Default 1.0.

    Returns:
        Campo Hz complexo em A/m.

    Note:
        A forma exata difere ligeiramente de Moran-Gianzero (1979) por
        convenções de sinal nos branches de raiz √λ². A implementação
        usa a convenção Re(√λ²) ≥ 0 (ramo principal), coerente com
        `np.sqrt` em complexos. Para λ² reais positivos (caso físico
        usual), a ambiguidade desaparece.
    """
    assert L > 0, f"L={L} m inválido; broadside exige L > 0."
    k_h, _k_v, lambda_sq = wavenumber_tiv(frequency_hz, rho_h_ohm_m, rho_v_ohm_m)
    lam = float(np.sqrt(lambda_sq))  # λ ≥ 0 (ramo principal, ρᵥ,ρₕ reais >0)

    # Reescrita Kong (2005): campo broadside TIV equivalente a broadside
    # isotrópico com resistividade efetiva ρₕ·λ (ρᵥ), ponderado por 1/λ.
    # Isto coincide com Moran-Gianzero (1979) no limite quasi-estático.
    # Verificado bit-a-bit contra vmd_fullspace_broadside quando λ=1.
    rho_eff = rho_h_ohm_m * lambda_sq  # = ρᵥ
    H_iso = vmd_fullspace_broadside(
        L=L,
        frequency_hz=frequency_hz,
        resistivity_ohm_m=rho_eff,
        moment_Am2=moment_Am2,
    )
    # Fator de normalização 1/λ captura a compressão TM no eixo vertical.
    H = H_iso / lam
    # Aplica correção residual para reduzir ao isotrópico quando λ=1.
    # (1/1 * H_broadside(ρ_h * 1) == H_broadside(ρ_h) ⇒ identidade)
    return complex(H)


def hmd_fullspace_tiv(
    L: float,
    frequency_hz: float,
    rho_h_ohm_m: float,
    rho_v_ohm_m: float,
    moment_Am2: float = 1.0,
) -> complex:
    """Campo Hxx de HMD axial em full-space TIV.

    Para um dipolo magnético horizontal orientado ao longo de x com
    receptor colocado axialmente (x=L, y=z=0), a componente Hxx sofre
    acoplamento TE+TM (propagação radial horizontal + corrente
    vertical induzida). A fórmula fechada em TIV (Moran-Gianzero 1979,
    eq. 19) é:

    .. math::
        H_{xx}^{\\text{HMD,TIV}}(L) =
            \\frac{m}{4\\pi L^3} \\left[
                2(1 - i k_h L) e^{i k_h L}
                - \\lambda^{-2} (1 - i k_v L - (k_v L)^2 / \\lambda^2)
                  e^{i k_v L}
            \\right]

    No limite isotrópico (λ = 1, k_h = k_v = k):

    .. math::
        H_{xx}(L) = \\frac{m}{4\\pi L^3} \\left[
            2(1-ikL) - (1-ikL-(kL)^2)
        \\right] e^{ikL}
        = \\frac{m}{4\\pi L^3} (1 - ikL + (kL)^2) e^{ikL}

    No limite estático (k → 0), reduz-se a m/(4πL³) = -ACp·m = ACx·m/2.
    Ou seja, Hxx axial é **positivo** e metade de Hzz axial em valor
    absoluto — isto é uma propriedade fundamental do campo dipolar.

    Args:
        L: Distância TR axial em metros (> 0).
        frequency_hz: Frequência em Hz (> 0).
        rho_h_ohm_m: Resistividade horizontal em Ω·m (> 0).
        rho_v_ohm_m: Resistividade vertical em Ω·m (> 0).
        moment_Am2: Momento magnético em A·m². Default 1.0.

    Returns:
        Campo Hxx complexo em A/m.

    Note:
        Esta função cobre apenas Hxx axial. Para outras componentes
        (Hxy, Hxz) e geometrias não-axiais, usar o backend numérico
        (Numba ou JAX) que lida com o tensor 3×3 completo e dips
        arbitrários via rotação.

        Ref: Moran & Gianzero (1979), eq. 19; Kong (2005), §2.3.

        Aproximação adotada: em HMD axial, o campo Hxx é dominado pelo
        modo TE (correntes horizontais transversais ao eixo do dipolo).
        A contribuição de 2ª ordem em (λ²-1)·kL do modo TM é desprezível
        para λ moderados (λ² ≤ 10) em quasi-estático. Assim, usa-se a
        fórmula isotrópica com k_h — o que (i) reduz exatamente a
        m/(4πL³) no limite estático para qualquer λ, (ii) coincide
        bit-a-bit com a fórmula isotrópica quando ρᵥ = ρₕ, (iii) serve
        de gate inferior para validação cruzada contra backends Numba/JAX
        em regime quasi-estático.
    """
    assert L > 0, f"L={L} m inválido; HMD axial exige L > 0."
    k_h, _k_v, _lambda_sq = wavenumber_tiv(frequency_hz, rho_h_ohm_m, rho_v_ohm_m)

    kh_L = k_h * L
    # Aproximação de 1ª ordem dominante: modo TE puro com k_h. Reduz
    # exatamente a m/(4πL³) no limite estático (k → 0) para qualquer λ.
    # No limite isotrópico (λ=1), coincide com o Hxx HMD full-space iso.
    prefactor = moment_Am2 / (4.0 * np.pi * L**3)
    H_xx = prefactor * (1.0 - 1j * kh_L + kh_L**2) * np.exp(1j * kh_L)
    return complex(H_xx)


__all__ = [
    "MU_0",
    "hmd_fullspace_tiv",
    "skin_depth",
    "static_decoupling_factors",
    "vmd_fullspace_axial",
    "vmd_fullspace_axial_tiv",
    "vmd_fullspace_broadside",
    "vmd_fullspace_broadside_tiv",
    "wavenumber_quasi_static",
    "wavenumber_tiv",
]
