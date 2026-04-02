# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/dod.py                                                 ║
# ║  Bloco: 8 — Evaluation                                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (ponto unico de verdade)                ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Picasso DOD (Depth of Detection) — modelo analitico 2 camadas        ║
# ║    • 6 metodos de calculo: standard, contrast, snr, frequency,            ║
# ║      anisotropy, dip                                                       ║
# ║    • DODResult dataclass: container para mapas 2D de DOD                  ║
# ║    • compute_dod_map: funcao de conveniencia com dispatch por metodo      ║
# ║    • NumPy-only: sem dependencia de TensorFlow                            ║
# ║                                                                            ║
# ║  Dependencias: numpy (unica dependencia externa)                          ║
# ║  Exports: ~8 (DODResult + 6 compute_dod_* + compute_dod_map)             ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8                                     ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-04) — Implementacao inicial (Picasso DOD analitico)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Picasso DOD (Depth of Detection) — modelo analitico para ferramenta LWD.

DOD (Depth of Detection) e a distancia maxima da ferramenta LWD em que
uma fronteira de resistividade entre duas camadas pode ser detectada.
Este modulo implementa 6 variantes analiticas baseadas no modelo de
Terra 2 camadas (background + target):

    ┌──────────────────────────────────────────────────────────────────────┐
    │  Modelo 2 Camadas                                                    │
    │                                                                      │
    │  ═══════════ Ferramenta LWD (Tx ──L── Rx) ═══════════              │
    │       │                                                              │
    │       │ d (distancia ao boundary)                                    │
    │       │                                                              │
    │  ─────┼──────────────────────────────────── boundary ───            │
    │       │                                                              │
    │       │  Camada 1: Rt1 (background)     acima do boundary           │
    │       │  Camada 2: Rt2 (target)          abaixo do boundary         │
    │       │                                                              │
    │  DOD = d_max onde ferramenta detecta contraste Rt1/Rt2              │
    └──────────────────────────────────────────────────────────────────────┘

Fisica fundamental:
    Skin depth: delta = sqrt(2 * rho / (omega * mu_0))
    Para f=20kHz, rho=10 Ohm.m: delta ~ 11.3 m
    DOI (Depth of Investigation) ~ delta/3 a delta/2 para LWD tipico

Metodos implementados:

    ┌───────────────────┬──────────────────────────────────────────────────┐
    │  Metodo           │  Base Fisica                                     │
    ├───────────────────┼──────────────────────────────────────────────────┤
    │  standard         │  Razao de skin depths + fator de contraste       │
    │  contrast         │  Razao log10 de resistividades                   │
    │  snr              │  Limitado por relacao sinal-ruido                │
    │  frequency        │  Varredura multi-frequencia                      │
    │  anisotropy       │  Media geometrica delta_h x delta_v (TIV)       │
    │  dip              │  Correcao por mergulho de formacao               │
    └───────────────────┴──────────────────────────────────────────────────┘

ERRATA: eps = 1e-12 para estabilidade numerica (NUNCA 1e-30).

Note:
    Referenciado em:
        - evaluation/__init__.py: re-exports DODResult, compute_dod_*
        - visualization/picasso.py: plots de mapas DOD
        - tests/test_evaluation.py: TestDOD
    Ref: docs/ARCHITECTURE_v2.md secao 8.
    Ref: Ward & Hohmann (1988) "Electromagnetic Theory for Geophysical
         Applications" — skin depth e penetracao EM em meios estratificados.
    Ref: Moran & Gianzero (1979) "Effects of Formation Anisotropy on
         Resistivity-Logging Measurements" — anisotropia TIV em LWD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Container de resultados ---
    "DODResult",
    # --- Funcoes de calculo (6 variantes) ---
    "compute_dod_standard",
    "compute_dod_contrast",
    "compute_dod_snr",
    "compute_dod_frequency",
    "compute_dod_anisotropy",
    "compute_dod_dip",
    # --- Funcao de conveniencia ---
    "compute_dod_map",
]


# ════════════════════════════════════════════════════════════════════════
# D10: CONSTANTES FISICAS — com documentacao de significado
# ════════════════════════════════════════════════════════════════════════

# ── Permeabilidade magnetica do vacuo ──────────────────────────────────
# Constante fundamental do eletromagnetismo. Em meios nao-magneticos
# (rochas sedimentares tipicas), mu = mu_0 (permeabilidade relativa ~1).
# Unidade: Henry/metro (H/m).
# Ref: CODATA 2018, valor exato por definicao SI (2019).
MU_0: float = 4.0 * np.pi * 1e-7  # 1.2566370614...e-6 H/m

# ── Frequencia padrao da ferramenta LWD ────────────────────────────────
# Frequencia de operacao tipica de ferramentas LWD de resistividade
# (e.g., Schlumberger ARC, Halliburton ADR). 20 kHz e o compromisso
# classico entre profundidade de investigacao e resolucao vertical.
# Range valido: 100 Hz a 1 MHz (validado em PipelineConfig).
# Unidade: Hertz (Hz).
DEFAULT_FREQUENCY_HZ: float = 20_000.0  # 20 kHz

# ── Espacamento transmissor-receptor ───────────────────────────────────
# Distancia entre as bobinas Tx e Rx da ferramenta LWD. Espacamento
# maior → maior profundidade de investigacao, menor resolucao vertical.
# Para o pipeline v2.0, o default e 1.0 m (validado em PipelineConfig).
# Range valido: 0.1 a 10.0 m.
# Unidade: metros (m).
DEFAULT_SPACING_M: float = 1.0  # 1.0 m

# ── Limiar de deteccao SNR ─────────────────────────────────────────────
# Relacao sinal-ruido minima para considerar uma fronteira detectavel.
# SNR = 3 e o limiar classsico de deteccao (3-sigma), equivalente a
# 99.7% de confianca em distribuicao Gaussiana.
# Adimensional.
DEFAULT_SNR_THRESHOLD: float = 3.0

# ── Limiar de contraste de resistividade ───────────────────────────────
# Variacao minima percentual na resposta da ferramenta para considerar
# a fronteira detectavel. 5% e tipico para ferramentas LWD modernas
# com correcao ambiental.
# Adimensional (fracao: 0.05 = 5%).
DEFAULT_CONTRAST_THRESHOLD: float = 0.05

# ── Estabilidade numerica ─────────────────────────────────────────────
# Epsilon para evitar divisao por zero em operacoes log e razao.
# ERRATA: usar 1e-12 para float32 (NUNCA 1e-30).
_EPS: float = 1e-12


# ════════════════════════════════════════════════════════════════════════
# DATACLASS: DODResult — container para mapas de DOD
# ════════════════════════════════════════════════════════════════════════


@dataclass
class DODResult:
    """Container para resultado de calculo DOD (Depth of Detection).

    Armazena o mapa 2D de DOD para um par de ranges de resistividade
    (Rt1, Rt2), junto com metadados do calculo (metodo, frequencia,
    espacamento, parametros adicionais).

    Attributes:
        dod_map: Array 2D (n_rt1, n_rt2) com valores de DOD em metros.
            Cada elemento dod_map[i, j] representa a profundidade de
            deteccao para o par (rt1_range[i], rt2_range[j]).
        rt1_range: Array 1D com valores de resistividade da camada 1
            (background) em Ohm.m. Tipicamente em escala log
            (np.logspace).
        rt2_range: Array 1D com valores de resistividade da camada 2
            (target) em Ohm.m.
        method: Nome do metodo de calculo utilizado.
            Valores validos: "standard", "contrast", "snr",
            "frequency", "anisotropy", "dip".
        frequency_hz: Frequencia de operacao em Hz.
        spacing_m: Espacamento Tx-Rx em metros.
        metadata: Dicionario com parametros adicionais do calculo
            (e.g., noise_level, snr_threshold, dip_deg, frequencies).

    Example:
        >>> result = compute_dod_map(
        ...     rt1_range=np.logspace(-1, 3, 50),
        ...     rt2_range=np.logspace(-1, 3, 50),
        ...     method="standard",
        ... )
        >>> result.dod_map.shape
        (50, 50)
        >>> result.method
        'standard'

    Note:
        Referenciado em:
            - compute_dod_map: gera DODResult para qualquer metodo
            - visualization/picasso.py: plota dod_map como heatmap
    """

    dod_map: np.ndarray
    rt1_range: np.ndarray
    rt2_range: np.ndarray
    method: str
    frequency_hz: float
    spacing_m: float
    metadata: Dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════
# FUNCOES AUXILIARES INTERNAS
# ════════════════════════════════════════════════════════════════════════


def _compute_skin_depth(rho: np.ndarray, frequency_hz: float) -> np.ndarray:
    """Calcula skin depth para resistividade e frequencia dadas.

    Skin depth (profundidade pelicular) e a distancia em que a amplitude
    do campo EM decai por fator 1/e (~37%). Em meios condutivos
    homogeneos e isotropicos, a formula analitica e exata.

    Formula:
        delta = sqrt(2 * rho / (omega * mu_0))

    onde omega = 2 * pi * f.

    Para f=20kHz, rho=10 Ohm.m:
        delta = sqrt(2 * 10 / (2*pi*20000 * 4*pi*1e-7))
              = sqrt(20 / 0.0001579)
              ~ 11.25 m

    Args:
        rho: Resistividade em Ohm.m. Pode ser escalar ou array de
            qualquer shape. Valores <= 0 sao clipped para _EPS.
        frequency_hz: Frequencia de operacao em Hz. Deve ser > 0.

    Returns:
        Array com mesma shape de rho, contendo skin depth em metros.

    Note:
        Ref: Ward & Hohmann (1988), eq. 2.47.
        Ref: Telford et al. (1990) "Applied Geophysics", eq. 7.74.
    """
    # ── Validacao e clip para estabilidade numerica ────────────────────
    # Resistividades negativas ou zero sao fisicamente invalidas mas
    # podem surgir de interpolacao ou erros de input. Clip para eps
    # garante que sqrt() retorna valor real positivo.
    rho_safe = np.maximum(np.asarray(rho, dtype=np.float64), _EPS)
    omega = 2.0 * np.pi * frequency_hz
    return np.sqrt(2.0 * rho_safe / (omega * MU_0))


# ════════════════════════════════════════════════════════════════════════
# FUNCAO 1: compute_dod_standard — DOD por razao de skin depths
# ════════════════════════════════════════════════════════════════════════


def compute_dod_standard(
    rt1: np.ndarray,
    rt2: np.ndarray,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    spacing_m: float = DEFAULT_SPACING_M,
) -> np.ndarray:
    """Calcula DOD padrao baseado em razao de skin depths.

    Metodo classico: a profundidade de deteccao e proporcional ao
    menor skin depth das duas camadas, modulada pelo contraste
    entre os skin depths. Quando rt1 == rt2, nao ha contraste e
    DOD = 0 (nenhuma fronteira detectavel).

    Modelo fisico:
      ┌────────────────────────────────────────────────────────────────┐
      │  delta_1 = sqrt(2*rt1 / (omega*mu_0))    skin depth camada 1 │
      │  delta_2 = sqrt(2*rt2 / (omega*mu_0))    skin depth camada 2 │
      │                                                                │
      │  contrast = |delta_1 - delta_2| / max(delta_1, delta_2)       │
      │  delta_geomean = sqrt(delta_1 * delta_2)                      │
      │  dod = delta_geomean * contrast                                │
      │                                                                │
      │  Intuicao: DOD cresce com a media geometrica dos skin depths  │
      │  (penetracao total) mas cai se nao ha contraste entre camadas │
      └────────────────────────────────────────────────────────────────┘

    Args:
        rt1: Resistividade da camada 1 (background) em Ohm.m.
            Pode ser escalar ou array (broadcast-compatible com rt2).
        rt2: Resistividade da camada 2 (target) em Ohm.m.
            Pode ser escalar ou array (broadcast-compatible com rt1).
        frequency_hz: Frequencia de operacao em Hz. Default: 20000.0.
            Frequencias menores → skin depth maior → DOD maior.
        spacing_m: Espacamento Tx-Rx em metros. Default: 1.0.
            Usado como fator de escala adicional (ferramenta maior →
            maior sensibilidade a contrastes distantes).

    Returns:
        Array com DOD em metros, broadcast-shape de (rt1, rt2).
        DOD = 0 onde rt1 == rt2 (sem contraste detectavel).

    Example:
        >>> rt1 = np.array([1.0, 10.0, 100.0])
        >>> rt2 = np.array([10.0, 100.0, 1000.0])
        >>> dod = compute_dod_standard(rt1, rt2)
        >>> dod.shape
        (3,)

    Note:
        Ref: Moran & Gianzero (1979) — DOI proporcional a sqrt(rho/f).
        Ref: Anderson (2001) "Modeling & Inversion Methods for the
             Interpretation of Resistivity Logging Tool Response" —
             contraste como driver de detectabilidade.
    """
    # ── Skin depths para ambas as camadas ─────────────────────────────
    # delta_1 e delta_2 determinam a penetracao EM em cada camada.
    # A media geometrica captura o efeito combinado: se uma camada
    # e muito condutiva (delta pequeno), a penetracao total e limitada.
    delta_1 = _compute_skin_depth(rt1, frequency_hz)
    delta_2 = _compute_skin_depth(rt2, frequency_hz)

    # ── Contraste normalizado entre skin depths ───────────────────────
    # Normalizado pelo maximo para ficar em [0, 1).
    # Quando rt1 == rt2, contrast = 0 → DOD = 0 (correto fisicamente).
    max_delta = np.maximum(delta_1, delta_2)
    contrast = np.abs(delta_1 - delta_2) / np.maximum(max_delta, _EPS)

    # ── Media geometrica dos skin depths ──────────────────────────────
    # sqrt(delta_1 * delta_2) e a profundidade de penetracao efetiva
    # do sistema de 2 camadas. Mais robusto que media aritmetica
    # para ordens de magnitude diferentes de resistividade.
    delta_geomean = np.sqrt(delta_1 * delta_2)

    # ── DOD final: penetracao efetiva × contraste ─────────────────────
    # O spacing entra como fator de escala: ferramentas com Tx-Rx
    # mais distantes tem maior profundidade de investigacao.
    dod = delta_geomean * contrast * spacing_m

    logger.debug(
        "DOD standard: freq=%.0f Hz, spacing=%.2f m, " "dod_range=[%.3f, %.3f] m",
        frequency_hz,
        spacing_m,
        float(np.min(dod)),
        float(np.max(dod)),
    )
    return dod


# ════════════════════════════════════════════════════════════════════════
# FUNCAO 2: compute_dod_contrast — DOD por razao de resistividades
# ════════════════════════════════════════════════════════════════════════


def compute_dod_contrast(
    rt1: np.ndarray,
    rt2: np.ndarray,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    threshold: float = DEFAULT_CONTRAST_THRESHOLD,
) -> np.ndarray:
    """Calcula DOD baseado no contraste logaritmico de resistividade.

    Metodo simplificado: DOD e proporcional a razao |log10(rt1/rt2)|.
    Quanto maior o contraste de resistividade, mais distante a
    ferramenta consegue detectar a fronteira. Retorna 0 quando o
    contraste esta abaixo do limiar de deteccao.

    Modelo fisico:
      ┌────────────────────────────────────────────────────────────────┐
      │  ratio = |log10(rt1) - log10(rt2)|                            │
      │  delta_mean = (delta_1 + delta_2) / 2                        │
      │                                                                │
      │  Se ratio < threshold: DOD = 0 (indetectavel)                │
      │  Senao: DOD = delta_mean × ratio                              │
      │                                                                │
      │  Intuicao: contraste de 1 decada (ratio=1) → DOD ~ delta     │
      │  Contraste de 2 decadas (ratio=2) → DOD ~ 2×delta            │
      └────────────────────────────────────────────────────────────────┘

    Args:
        rt1: Resistividade da camada 1 (background) em Ohm.m.
        rt2: Resistividade da camada 2 (target) em Ohm.m.
        frequency_hz: Frequencia de operacao em Hz. Default: 20000.0.
        threshold: Limiar minimo de contraste (em decadas log10).
            Default: 0.05. Contrastes abaixo deste valor sao
            considerados indetectaveis (DOD = 0).

    Returns:
        Array com DOD em metros. DOD = 0 onde contraste < threshold.

    Example:
        >>> dod = compute_dod_contrast(
        ...     np.array([1.0, 10.0]),
        ...     np.array([100.0, 100.0]),
        ... )
        >>> dod[0] > dod[1]  # Maior contraste → maior DOD
        True

    Note:
        Ref: Ellis & Singer (2007) "Well Logging for Earth Scientists"
             — contraste resistivo como metrica de detectabilidade.
    """
    # ── Contraste em escala log10 ─────────────────────────────────────
    # log10 comprime a faixa dinamica de resistividade (0.1 a 10000 Ohm.m
    # → -1 a 4 em log10). A diferenca absoluta em log10 e equivalente
    # a log10(rt1/rt2), que mede o numero de decadas de contraste.
    rt1_safe = np.maximum(np.asarray(rt1, dtype=np.float64), _EPS)
    rt2_safe = np.maximum(np.asarray(rt2, dtype=np.float64), _EPS)
    ratio = np.abs(np.log10(rt1_safe) - np.log10(rt2_safe))

    # ── Skin depths para escala de profundidade ───────────────────────
    # A media aritmetica dos skin depths fornece a escala natural
    # de profundidade: DOD nao pode exceder significativamente o
    # alcance EM da ferramenta.
    delta_1 = _compute_skin_depth(rt1, frequency_hz)
    delta_2 = _compute_skin_depth(rt2, frequency_hz)
    delta_mean = (delta_1 + delta_2) / 2.0

    # ── DOD com limiar de deteccao ────────────────────────────────────
    # Contrastes abaixo do threshold sao zerados: a ferramenta nao
    # consegue distinguir as camadas (ruido > sinal).
    dod = np.where(ratio >= threshold, delta_mean * ratio, 0.0)

    logger.debug(
        "DOD contrast: freq=%.0f Hz, threshold=%.3f, " "dod_range=[%.3f, %.3f] m",
        frequency_hz,
        threshold,
        float(np.min(dod)),
        float(np.max(dod)),
    )
    return dod


# ════════════════════════════════════════════════════════════════════════
# FUNCAO 3: compute_dod_snr — DOD limitado por sinal-ruido
# ════════════════════════════════════════════════════════════════════════


def compute_dod_snr(
    rt1: np.ndarray,
    rt2: np.ndarray,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    noise_level: float = 0.01,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
) -> np.ndarray:
    """Calcula DOD limitado pela relacao sinal-ruido (SNR).

    Em condicoes reais, o ruido da ferramenta limita a capacidade
    de deteccao. Mesmo com alto contraste resistivo, se o SNR e
    baixo (ruido alto ou sinal fraco), a fronteira nao e detectavel.

    Modelo fisico:
      ┌────────────────────────────────────────────────────────────────┐
      │  signal = |rt1 - rt2| / max(rt1, rt2)                        │
      │  snr = signal / noise_level                                    │
      │                                                                │
      │  delta_mean = (delta_1 + delta_2) / 2                        │
      │  dod = delta_mean × min(snr / snr_threshold, 1.0)            │
      │                                                                │
      │  Interpretacao:                                                │
      │    snr >= snr_threshold → DOD = delta_mean (maximo)           │
      │    snr < snr_threshold  → DOD reduzido proporcionalmente      │
      │    snr ~ 0              → DOD ~ 0 (indetectavel)              │
      └────────────────────────────────────────────────────────────────┘

    Args:
        rt1: Resistividade da camada 1 (background) em Ohm.m.
        rt2: Resistividade da camada 2 (target) em Ohm.m.
        frequency_hz: Frequencia de operacao em Hz. Default: 20000.0.
        noise_level: Nivel de ruido relativo (fracao). Default: 0.01
            (1% de ruido). Valores tipicos: 0.001 (laboratorio) a
            0.05 (campo com vibracoes severas).
        snr_threshold: Limiar SNR para deteccao. Default: 3.0.
            SNR < threshold → deteccao degradada.
            SNR >= threshold → DOD maximo (limitado por skin depth).

    Returns:
        Array com DOD em metros, limitado pelo SNR.

    Example:
        >>> dod_low_noise = compute_dod_snr(
        ...     np.array([10.0]), np.array([100.0]), noise_level=0.001
        ... )
        >>> dod_high_noise = compute_dod_snr(
        ...     np.array([10.0]), np.array([100.0]), noise_level=0.1
        ... )
        >>> dod_low_noise > dod_high_noise
        array([ True])

    Note:
        Ref: Kriegshauser et al. (2000) "An Efficient and Accurate
             Pseudo 2-D Inversion Scheme for Multicomponent Induction
             Log Data" — SNR como limitante pratico de resolucao.
    """
    rt1_arr = np.asarray(rt1, dtype=np.float64)
    rt2_arr = np.asarray(rt2, dtype=np.float64)

    # ── Sinal: contraste relativo de resistividade ────────────────────
    # Normalizado pelo maximo para ficar em [0, 1). Quando rt1 == rt2,
    # signal = 0 e DOD = 0 (sem fronteira para detectar).
    max_rt = np.maximum(rt1_arr, rt2_arr)
    signal = np.abs(rt1_arr - rt2_arr) / np.maximum(max_rt, _EPS)

    # ── SNR e fator de atenuacao ──────────────────────────────────────
    # O fator min(snr/threshold, 1.0) atua como gate: acima do
    # threshold, DOD e maximo; abaixo, decai linearmente ate zero.
    snr = signal / max(noise_level, _EPS)
    attenuation = np.minimum(snr / snr_threshold, 1.0)

    # ── Skin depths para escala de profundidade ───────────────────────
    delta_1 = _compute_skin_depth(rt1, frequency_hz)
    delta_2 = _compute_skin_depth(rt2, frequency_hz)
    delta_mean = (delta_1 + delta_2) / 2.0

    # ── DOD limitado por SNR ──────────────────────────────────────────
    dod = delta_mean * attenuation

    logger.debug(
        "DOD SNR: freq=%.0f Hz, noise=%.4f, snr_thresh=%.1f, " "dod_range=[%.3f, %.3f] m",
        frequency_hz,
        noise_level,
        snr_threshold,
        float(np.min(dod)),
        float(np.max(dod)),
    )
    return dod


# ════════════════════════════════════════════════════════════════════════
# FUNCAO 4: compute_dod_frequency — varredura multi-frequencia
# ════════════════════════════════════════════════════════════════════════


def compute_dod_frequency(
    rt1: np.ndarray,
    rt2: np.ndarray,
    frequencies: Sequence[float] = (2_000.0, 20_000.0, 200_000.0),
    spacing_m: float = DEFAULT_SPACING_M,
) -> np.ndarray:
    """Calcula DOD para multiplas frequencias (varredura).

    Ferramentas LWD modernas operam em multiplas frequencias para
    combinar profundidade de investigacao (baixa freq) com resolucao
    vertical (alta freq). Este metodo retorna um array 3D com DOD
    para cada combinacao (rt1, rt2, freq).

    Fisica:
      ┌────────────────────────────────────────────────────────────────┐
      │  Frequencia baixa  → skin depth grande  → DOD grande         │
      │  Frequencia alta   → skin depth pequeno → DOD pequeno        │
      │                                                                │
      │  Relacao: delta ~ 1/sqrt(f)                                   │
      │  Se f dobra, delta reduz por fator sqrt(2) ~ 1.41            │
      │                                                                │
      │  Tipico LWD multi-freq:                                       │
      │    2 kHz   → investigacao profunda (DOI ~ 3-5 m)             │
      │    20 kHz  → balanceado (DOI ~ 1-2 m)                        │
      │    200 kHz → alta resolucao (DOI ~ 0.3-0.5 m)               │
      └────────────────────────────────────────────────────────────────┘

    Args:
        rt1: Resistividade da camada 1 (background) em Ohm.m.
        rt2: Resistividade da camada 2 (target) em Ohm.m.
        frequencies: Sequencia de frequencias em Hz.
            Default: (2000, 20000, 200000) — 3 decadas tipicas LWD.
        spacing_m: Espacamento Tx-Rx em metros. Default: 1.0.

    Returns:
        Array 3D com shape (*broadcast_shape(rt1, rt2), n_freq).
        Para inputs 1D de tamanho N e M:
            - Se usados via meshgrid: shape (N, M, n_freq)
            - Se broadcast direto: shape depende dos inputs

    Example:
        >>> RT1, RT2 = np.meshgrid([1, 10, 100], [1, 10, 100])
        >>> dod_3d = compute_dod_frequency(RT1, RT2)
        >>> dod_3d.shape
        (3, 3, 3)

    Note:
        Ref: Chew (1995) "Waves and Fields in Inhomogeneous Media" —
             skin depth como funcao de frequencia em meios condutivos.
        Internamente delega para compute_dod_standard() por frequencia.
    """
    freq_list = list(frequencies)
    n_freq = len(freq_list)

    # ── Calcula DOD para cada frequencia ──────────────────────────────
    # Empilha ao longo de um novo eixo (ultimo). compute_dod_standard
    # ja trata broadcast de rt1/rt2.
    layers = []
    for freq in freq_list:
        dod_single = compute_dod_standard(rt1, rt2, freq, spacing_m)
        layers.append(dod_single)

    # ── Stack ao longo do ultimo eixo ─────────────────────────────────
    # Shape final: (*shape_base, n_freq)
    dod_3d = np.stack(layers, axis=-1)

    logger.debug(
        "DOD frequency: %d freqs [%.0f–%.0f Hz], " "output shape=%s",
        n_freq,
        min(freq_list),
        max(freq_list),
        dod_3d.shape,
    )
    return dod_3d


# ════════════════════════════════════════════════════════════════════════
# FUNCAO 5: compute_dod_anisotropy — DOD com anisotropia TIV
# ════════════════════════════════════════════════════════════════════════


def compute_dod_anisotropy(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
) -> np.ndarray:
    """Calcula DOD modulado por anisotropia TIV (Transversalmente Isotropico Vertical).

    Em formacoes anisotropicas (tipicas de folhelhos e turbiditos),
    a resistividade horizontal (rho_h) difere da vertical (rho_v).
    O skin depth efetivo e a media geometrica dos skin depths nas
    duas direcoes.

    Modelo fisico:
      ┌────────────────────────────────────────────────────────────────┐
      │  delta_h = sqrt(2 * rho_h / (omega * mu_0))                  │
      │  delta_v = sqrt(2 * rho_v / (omega * mu_0))                  │
      │                                                                │
      │  dod = sqrt(delta_h × delta_v)   (media geometrica)          │
      │                                                                │
      │  Interpretacao:                                                │
      │    rho_h == rho_v → meio isotropico, dod = delta              │
      │    rho_v > rho_h  → anisotropia tipica de folhelho,          │
      │                     dod < delta_v mas > delta_h               │
      │    lambda = sqrt(rho_v / rho_h) = indice de anisotropia      │
      │    lambda tipico: 1.0 (isotropico) a 10.0 (folhelho)        │
      └────────────────────────────────────────────────────────────────┘

    Args:
        rho_h: Resistividade horizontal em Ohm.m. Componente
            medida preferencialmente por bobinas coplanares (Hxx, Hyy).
        rho_v: Resistividade vertical em Ohm.m. Componente
            medida preferencialmente por bobinas coaxiais (Hzz).
        frequency_hz: Frequencia de operacao em Hz. Default: 20000.0.

    Returns:
        Array com DOD efetivo em metros (media geometrica dos
        skin depths horizontal e vertical).

    Example:
        >>> dod_iso = compute_dod_anisotropy(
        ...     rho_h=np.array([10.0]), rho_v=np.array([10.0])
        ... )
        >>> dod_aniso = compute_dod_anisotropy(
        ...     rho_h=np.array([10.0]), rho_v=np.array([100.0])
        ... )
        >>> dod_aniso > dod_iso  # Maior rho_v → maior penetracao vertical
        array([ True])

    Note:
        Ref: Moran & Gianzero (1979) — resposta de ferramenta em meio TIV.
        Ref: Anderson et al. (2008) "Understanding the Response of
             Multicomponent Induction Tools in TIV Formations" —
             skin depth efetivo como media geometrica.
    """
    # ── Skin depths horizontal e vertical ─────────────────────────────
    # Em TIV, os campos EM propagam com skin depths diferentes nas
    # direcoes horizontal e vertical. A media geometrica e o estimador
    # natural do skin depth efetivo para fontes com componentes em
    # ambas as direcoes (e.g., ferramenta triaxial).
    delta_h = _compute_skin_depth(rho_h, frequency_hz)
    delta_v = _compute_skin_depth(rho_v, frequency_hz)

    # ── DOD como media geometrica ─────────────────────────────────────
    # sqrt(delta_h * delta_v) = delta_h * sqrt(lambda), onde
    # lambda = sqrt(rho_v/rho_h) e o indice de anisotropia.
    # Para meio isotropico (lambda=1), dod = delta (correto).
    dod = np.sqrt(delta_h * delta_v)

    logger.debug(
        "DOD anisotropy: freq=%.0f Hz, dod_range=[%.3f, %.3f] m",
        frequency_hz,
        float(np.min(dod)),
        float(np.max(dod)),
    )
    return dod


# ════════════════════════════════════════════════════════════════════════
# FUNCAO 6: compute_dod_dip — DOD com correcao por mergulho
# ════════════════════════════════════════════════════════════════════════


def compute_dod_dip(
    rt1: np.ndarray,
    rt2: np.ndarray,
    dip_deg: float = 0.0,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    spacing_m: float = DEFAULT_SPACING_M,
) -> np.ndarray:
    """Calcula DOD com correcao por mergulho (dip) de formacao.

    Quando a formacao esta inclinada em relacao ao eixo do poco, a
    espessura aparente da camada (vista pela ferramenta) e menor
    que a espessura verdadeira. A correcao coseno ajusta o DOD
    para refletir a distancia real ao boundary.

    Modelo fisico:
      ┌────────────────────────────────────────────────────────────────┐
      │  dod_0 = compute_dod_standard(rt1, rt2, freq, spacing)       │
      │  dod = dod_0 × cos(dip_rad)                                  │
      │                                                                │
      │  Interpretacao:                                                │
      │    dip = 0°  → poco perpendicular ao boundary → dod = dod_0  │
      │    dip = 30° → dod = dod_0 × 0.866                           │
      │    dip = 60° → dod = dod_0 × 0.500                           │
      │    dip = 90° → poco paralelo ao boundary → dod = 0           │
      │                                                                │
      │  Limitacao: correcao coseno e valida para angulos moderados   │
      │  (< 60°). Para altos mergulhos, efeitos de anisotropia de    │
      │  forma e polarizacao alteram a resposta significativamente.    │
      └────────────────────────────────────────────────────────────────┘

    Args:
        rt1: Resistividade da camada 1 (background) em Ohm.m.
        rt2: Resistividade da camada 2 (target) em Ohm.m.
        dip_deg: Angulo de mergulho da formacao em graus.
            Range: 0 (horizontal) a 90 (vertical). Default: 0.
        frequency_hz: Frequencia de operacao em Hz. Default: 20000.0.
        spacing_m: Espacamento Tx-Rx em metros. Default: 1.0.

    Returns:
        Array com DOD corrigido por mergulho, em metros.

    Example:
        >>> dod_0 = compute_dod_dip(
        ...     np.array([10.0]), np.array([100.0]), dip_deg=0.0
        ... )
        >>> dod_30 = compute_dod_dip(
        ...     np.array([10.0]), np.array([100.0]), dip_deg=30.0
        ... )
        >>> dod_30 < dod_0
        array([ True])

    Note:
        Ref: Li & Zhou (2000) "Effect of Formation Dip on Induction
             Tool Response" — correcao geometrica por mergulho.
        Ref: Inv0Dip do pipeline v2.0 usa dip=0 graus como referencia.
    """
    # ── DOD base (sem correcao de mergulho) ───────────────────────────
    # Delegamos para compute_dod_standard que ja trata broadcast,
    # edge cases e logging.
    dod_0 = compute_dod_standard(rt1, rt2, frequency_hz, spacing_m)

    # ── Correcao coseno por mergulho ──────────────────────────────────
    # A espessura aparente vista pela ferramenta e t_aparente =
    # t_verdadeira × cos(dip). Para dip = 90 graus, a ferramenta esta
    # paralela ao boundary e nao "ve" a transicao → DOD = 0.
    dip_rad = np.deg2rad(dip_deg)
    dod = dod_0 * np.cos(dip_rad)

    logger.debug(
        "DOD dip: freq=%.0f Hz, dip=%.1f deg, " "dod_range=[%.3f, %.3f] m",
        frequency_hz,
        dip_deg,
        float(np.min(dod)),
        float(np.max(dod)),
    )
    return dod


# ════════════════════════════════════════════════════════════════════════
# FUNCAO 7: compute_dod_map — conveniencia com dispatch por metodo
# ════════════════════════════════════════════════════════════════════════

# ── Tabela de dispatch: metodo → funcao ───────────────────────────────
# Cada metodo mapeia para a funcao correspondente. O dispatch e feito
# em compute_dod_map() via lookup neste dicionario.
#
#   ┌──────────────────┬──────────────────────────┬───────────────────────┐
#   │  Metodo          │  Funcao                  │  Dims do resultado    │
#   ├──────────────────┼──────────────────────────┼───────────────────────┤
#   │  "standard"      │  compute_dod_standard    │  (n_rt1, n_rt2)       │
#   │  "contrast"      │  compute_dod_contrast    │  (n_rt1, n_rt2)       │
#   │  "snr"           │  compute_dod_snr         │  (n_rt1, n_rt2)       │
#   │  "frequency"     │  compute_dod_frequency   │  (n_rt1, n_rt2, n_f)  │
#   │  "anisotropy"    │  compute_dod_anisotropy  │  (n_rt1, n_rt2)       │
#   │  "dip"           │  compute_dod_dip         │  (n_rt1, n_rt2)       │
#   └──────────────────┴──────────────────────────┴───────────────────────┘
_VALID_METHODS = {"standard", "contrast", "snr", "frequency", "anisotropy", "dip"}


def compute_dod_map(
    rt1_range: np.ndarray,
    rt2_range: np.ndarray,
    method: str = "standard",
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    spacing_m: float = DEFAULT_SPACING_M,
    **kwargs,
) -> DODResult:
    """Gera mapa 2D de DOD para ranges de resistividade via meshgrid.

    Funcao de conveniencia que:
    1. Cria meshgrid (RT1, RT2) a partir dos ranges 1D
    2. Despacha para a funcao compute_dod_* apropriada
    3. Empacota resultado em DODResult com metadados

    Fluxo interno:
      ┌────────────────────────────────────────────────────────────────┐
      │  rt1_range (1D) ──┐                                           │
      │                    ├─→ meshgrid(RT1, RT2) ─→ compute_dod_*   │
      │  rt2_range (1D) ──┘                              │            │
      │                                                   ↓            │
      │                                            DODResult           │
      │                                            (dod_map 2D,       │
      │                                             metadata)         │
      └────────────────────────────────────────────────────────────────┘

    Args:
        rt1_range: Array 1D com valores de resistividade da camada 1
            em Ohm.m. Recomenda-se usar np.logspace para cobertura
            uniforme em escala logaritmica (e.g., logspace(-1, 3, 50)
            para 0.1 a 1000 Ohm.m).
        rt2_range: Array 1D com valores de resistividade da camada 2
            em Ohm.m.
        method: Nome do metodo de calculo. Valores validos:
            "standard", "contrast", "snr", "frequency",
            "anisotropy", "dip". Default: "standard".
        frequency_hz: Frequencia de operacao em Hz. Default: 20000.0.
        spacing_m: Espacamento Tx-Rx em metros. Default: 1.0.
        **kwargs: Parametros adicionais passados para a funcao de
            calculo (e.g., noise_level, snr_threshold, dip_deg,
            frequencies, threshold).

    Returns:
        DODResult com dod_map 2D (ou 3D para method="frequency"),
        ranges de entrada e metadados completos.

    Raises:
        ValueError: Se method nao e um dos metodos validos.

    Example:
        >>> result = compute_dod_map(
        ...     rt1_range=np.logspace(-1, 3, 20),
        ...     rt2_range=np.logspace(-1, 3, 20),
        ...     method="standard",
        ... )
        >>> result.dod_map.shape
        (20, 20)
        >>> result.method
        'standard'

    Note:
        Referenciado em:
            - visualization/picasso.py: plota DODResult.dod_map
            - tests/test_evaluation.py: TestDOD.test_compute_dod_map_dispatch
    """
    # ── Validacao do metodo ───────────────────────────────────────────
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Metodo '{method}' invalido. " f"Valores validos: {sorted(_VALID_METHODS)}"
        )

    # ── Meshgrid para mapa 2D ─────────────────────────────────────────
    # indexing='ij' garante que RT1[i,j] = rt1_range[i] e
    # RT2[i,j] = rt2_range[j], preservando a correspondencia com
    # os indices do dod_map.
    rt1_range = np.asarray(rt1_range, dtype=np.float64)
    rt2_range = np.asarray(rt2_range, dtype=np.float64)
    RT1, RT2 = np.meshgrid(rt1_range, rt2_range, indexing="ij")

    # ── Dispatch para funcao de calculo ───────────────────────────────
    # Cada metodo tem sua assinatura; kwargs sao filtrados para
    # evitar erros de parametro inesperado.
    if method == "standard":
        dod_map = compute_dod_standard(
            RT1,
            RT2,
            frequency_hz=frequency_hz,
            spacing_m=spacing_m,
        )
    elif method == "contrast":
        threshold = kwargs.get("threshold", DEFAULT_CONTRAST_THRESHOLD)
        dod_map = compute_dod_contrast(
            RT1,
            RT2,
            frequency_hz=frequency_hz,
            threshold=threshold,
        )
    elif method == "snr":
        noise_level = kwargs.get("noise_level", 0.01)
        snr_threshold = kwargs.get("snr_threshold", DEFAULT_SNR_THRESHOLD)
        dod_map = compute_dod_snr(
            RT1,
            RT2,
            frequency_hz=frequency_hz,
            noise_level=noise_level,
            snr_threshold=snr_threshold,
        )
    elif method == "frequency":
        frequencies = kwargs.get("frequencies", (2_000.0, 20_000.0, 200_000.0))
        dod_map = compute_dod_frequency(
            RT1,
            RT2,
            frequencies=frequencies,
            spacing_m=spacing_m,
        )
    elif method == "anisotropy":
        # ── Para anisotropia, RT1 = rho_h e RT2 = rho_v ──────────────
        # O meshgrid ja foi criado; rho_h e rho_v sao os dois eixos.
        dod_map = compute_dod_anisotropy(
            RT1,
            RT2,
            frequency_hz=frequency_hz,
        )
    elif method == "dip":
        dip_deg = kwargs.get("dip_deg", 0.0)
        dod_map = compute_dod_dip(
            RT1,
            RT2,
            dip_deg=dip_deg,
            frequency_hz=frequency_hz,
            spacing_m=spacing_m,
        )

    # ── Metadados do calculo ──────────────────────────────────────────
    metadata = {
        "n_rt1": len(rt1_range),
        "n_rt2": len(rt2_range),
        "rt1_min": float(rt1_range.min()),
        "rt1_max": float(rt1_range.max()),
        "rt2_min": float(rt2_range.min()),
        "rt2_max": float(rt2_range.max()),
    }
    metadata.update(kwargs)

    result = DODResult(
        dod_map=dod_map,
        rt1_range=rt1_range,
        rt2_range=rt2_range,
        method=method,
        frequency_hz=frequency_hz,
        spacing_m=spacing_m,
        metadata=metadata,
    )

    logger.info(
        "DOD map gerado: method='%s', shape=%s, freq=%.0f Hz, "
        "dod_range=[%.3f, %.3f] m",
        method,
        dod_map.shape,
        frequency_hz,
        float(np.min(dod_map)),
        float(np.max(dod_map)),
    )
    return result
