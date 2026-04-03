# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/advanced.py                                           ║
# ║  Bloco: 8 — Evaluation                                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • interface_metrics: deteccao de interfaces + analise de sharpness     ║
# ║    • error_by_resistivity_band: RMSE estratificado por banda log10        ║
# ║    • error_by_anisotropy: RMSE por razao de anisotropia rho_v/rho_h      ║
# ║    • spatial_error_profile: RMSE(z) ao longo da sequencia                ║
# ║    • physical_coherence_check: verificacao TIV (rho_v >= rho_h)          ║
# ║    • stability_analysis: analise de variancia sob perturbacao             ║
# ║    • NumPy-only (exceto stability_analysis que faz lazy import de TF)    ║
# ║                                                                            ║
# ║  Dependencias: numpy (principal), tensorflow (lazy, apenas stability)     ║
# ║  Exports: ~10 (6 funcoes + 4 dataclasses)                                ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8.3                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C50-C55)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Metricas avancadas de avaliacao para inversao de resistividade.

Complementa o modulo metrics.py (metricas basicas R2/RMSE/MAE/MBE/MAPE)
com analises especializadas para o dominio geofisico:

    ┌──────────────────────────────┬──────────────────────────────────────────┐
    │ Funcao                       │ Proposito                                │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ interface_metrics            │ Deteccao de interfaces + sharpness (C50) │
    │ error_by_resistivity_band   │ RMSE por banda de resistividade (C51)    │
    │ error_by_anisotropy         │ RMSE por razao lambda = rho_v/rho_h (C52)│
    │ spatial_error_profile       │ RMSE(z) ao longo da sequencia (C53)      │
    │ physical_coherence_check    │ Violacoes TIV rho_v >= rho_h (C54)       │
    │ stability_analysis          │ Variancia sob perturbacao gaussiana (C55) │
    └──────────────────────────────┴──────────────────────────────────────────┘

Todas as funcoes operam em dominio log10 (TARGET_SCALING = "log10"), exceto
error_by_anisotropy que converte internamente para dominio linear (Ohm.m)
para calcular a razao de anisotropia.

ERRATA: eps = 1e-12 para float32 (NUNCA 1e-30).

Note:
    Referenciado em:
        - evaluation/__init__.py: re-exports todas as funcoes e dataclasses
        - tests/test_evaluation.py: testes unitarios
    Ref: docs/ARCHITECTURE_v2.md secao 8.3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Dataclasses de resultado ---
    "InterfaceReport",
    "CoherenceReport",
    "StabilityReport",
    # --- Funcoes de analise avancada ---
    "interface_metrics",
    "error_by_resistivity_band",
    "error_by_anisotropy",
    "spatial_error_profile",
    "physical_coherence_check",
    "stability_analysis",
]


# ════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════
# Epsilon seguro para float32 — protege contra divisao por zero.
# NUNCA usar 1e-30 (causa subnormais em float32, gradientes explodidos).
# Ref: Errata v5.0.15, IEEE 754 float32 min normal ~ 1.175e-38.
# ────────────────────────────────────────────────────────────────────────

EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════
# DATACLASSES DE RESULTADO
#
# Containers imutaveis para resultados de analises avancadas. Cada
# dataclass documenta o significado fisico de cada campo.
#
# Ref: docs/reference/metricas_avancadas.md
# ════════════════════════════════════════════════════════════════════════


@dataclass
class InterfaceReport:
    """Relatorio de deteccao de interfaces geologicas.

    Interfaces sao transicoes abruptas de resistividade entre camadas
    adjacentes no modelo geologico 1D. Detectadas onde |Dy_true| > threshold.

    Attributes:
        n_interfaces_true: Total de interfaces detectadas em y_true.
        n_interfaces_pred: Total de interfaces detectadas em y_pred.
        detection_rate: Fracao de interfaces true corretamente detectadas
            em y_pred (dentro de +-tolerance pontos). Range [0, 1].
        sharpness_ratio_mean: Media de max|Dy_pred| / max|Dy_true| nas
            interfaces detectadas. Valor ~1.0 indica sharpness preservada;
            <1.0 indica suavizacao (smoothing); >1.0 indica overshoot.
        false_positive_rate: Fracao de interfaces pred que nao correspondem
            a nenhuma interface true. Range [0, 1].

    Example:
        >>> report = interface_metrics(y_true, y_pred, threshold=0.5)
        >>> report.detection_rate
        0.85

    Note:
        Referenciado em:
            - evaluation/advanced.py: interface_metrics retorna InterfaceReport
            - tests/test_evaluation.py: testes de deteccao de interfaces
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """

    n_interfaces_true: int
    n_interfaces_pred: int
    detection_rate: float
    sharpness_ratio_mean: float
    false_positive_rate: float


@dataclass
class CoherenceReport:
    """Relatorio de coerencia fisica TIV (rho_v >= rho_h).

    Em meios Transversalmente Isotropicos Verticais (TIV), a resistividade
    vertical (rho_v) deve ser maior ou igual a horizontal (rho_h). Violacoes
    indicam predicoes fisicamente inconsistentes.

    Attributes:
        total_points: Total de pontos avaliados (N * seq_len).
        violations: Numero de pontos onde rho_v < rho_h.
        violation_rate: violations / total_points. Range [0, 1].
        mean_violation_magnitude: Media de (rho_h - rho_v) nos pontos
            violados, em dominio log10. Valor 0.0 se nao ha violacoes.

    Example:
        >>> report = physical_coherence_check(y_pred)
        >>> report.violation_rate
        0.02

    Note:
        Referenciado em:
            - evaluation/advanced.py: physical_coherence_check retorna CoherenceReport
            - tests/test_evaluation.py: testes de violacao TIV
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """

    total_points: int
    violations: int
    violation_rate: float
    mean_violation_magnitude: float


@dataclass
class StabilityReport:
    """Relatorio de estabilidade sob perturbacao gaussiana.

    Avalia a robustez do modelo adicionando ruido gaussiano(sigma) ao
    input e medindo a variancia das predicoes. Alta variancia indica
    sensibilidade excessiva a perturbacoes no input.

    Attributes:
        mean_variance: Variancia media das predicoes sobre todas as
            saidas e todas as perturbacoes.
        max_variance: Variancia maxima em qualquer ponto individual.
        sensitivity_profile: Array (seq_len,) com variancia media por
            indice de profundidade. Util para detectar regioes sensiveis.

    Example:
        >>> report = stability_analysis(model, x_test, sigma=0.01)
        >>> report.mean_variance
        1.5e-05

    Note:
        Referenciado em:
            - evaluation/advanced.py: stability_analysis retorna StabilityReport
            - tests/test_evaluation.py: testes de perturbacao
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """

    mean_variance: float
    max_variance: float
    sensitivity_profile: np.ndarray


# ════════════════════════════════════════════════════════════════════════
# FUNCOES AUXILIARES (internas)
#
# Funcoes utilitarias compartilhadas pelas metricas avancadas.
# Nao exportadas no __all__ — uso interno apenas.
# ════════════════════════════════════════════════════════════════════════


def _validate_3d_paired(
    y_true: np.ndarray, y_pred: np.ndarray, *, name: str = "y"
) -> Tuple[np.ndarray, np.ndarray]:
    """Valida e converte par de arrays para (N, seq, 2) float64.

    Args:
        y_true: Array com valores reais. Deve ser 3D com ultima dim == 2.
        y_pred: Array com predicoes. Mesmo shape de y_true.
        name: Nome do par para mensagens de erro.

    Returns:
        Tupla (y_true, y_pred) como float64.

    Raises:
        ValueError: Se shapes sao invalidos ou incompativeis.

    Note:
        Referenciado em:
            - evaluation/advanced.py: todas as funcoes de analise avancada
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 3 or y_true.shape[-1] != 2:
        raise ValueError(
            f"Shape esperado (N, seq, 2) para {name}_true, " f"recebido {y_true.shape}"
        )
    if y_pred.ndim != 3 or y_pred.shape[-1] != 2:
        raise ValueError(
            f"Shape esperado (N, seq, 2) para {name}_pred, " f"recebido {y_pred.shape}"
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes incompativeis: {name}_true={y_true.shape}, "
            f"{name}_pred={y_pred.shape}"
        )
    return y_true, y_pred


def _validate_3d_single(y: np.ndarray, *, name: str = "y") -> np.ndarray:
    """Valida e converte array unico para (N, seq, 2) float64.

    Args:
        y: Array a validar. Deve ser 3D com ultima dim == 2.
        name: Nome do array para mensagens de erro.

    Returns:
        Array como float64.

    Raises:
        ValueError: Se shape e invalido.

    Note:
        Referenciado em:
            - evaluation/advanced.py: physical_coherence_check
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 3 or y.shape[-1] != 2:
        raise ValueError(f"Shape esperado (N, seq, 2) para {name}, recebido {y.shape}")
    return y


def _detect_interfaces(y: np.ndarray, *, threshold: float) -> List[List[int]]:
    """Detecta indices de interfaces em cada amostra.

    Interface = ponto onde |y[i+1] - y[i]| > threshold em qualquer
    componente (rho_h ou rho_v). Retorna lista de listas de indices.

    Args:
        y: Array (N, seq, 2) em dominio log10.
        threshold: Limiar de gradiente para deteccao.

    Returns:
        Lista de N listas, cada uma com indices de interfaces detectadas.

    Note:
        Referenciado em:
            - evaluation/advanced.py: interface_metrics (deteccao interna)
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """
    n_samples = y.shape[0]
    all_interfaces: List[List[int]] = []

    for i in range(n_samples):
        # Gradiente ao longo da sequencia para ambas componentes
        # D7: gradiente absoluto maximo entre rho_h e rho_v
        diff = np.abs(np.diff(y[i], axis=0))  # (seq-1, 2)
        max_diff = np.max(diff, axis=1)  # (seq-1,) — max entre componentes
        indices = np.where(max_diff > threshold)[0].tolist()
        all_interfaces.append(indices)

    return all_interfaces


# ════════════════════════════════════════════════════════════════════════
# C50: INTERFACE METRICS — Deteccao de interfaces + sharpness
#
# Detecta transicoes abruptas de resistividade (interfaces entre camadas
# geologicas) comparando gradientes discretos contra um threshold. Avalia
# se o modelo preserva a nitidez (sharpness) das transicoes ou suaviza
# as fronteiras, o que e critico para geosteering onde a posicao exata
# da interface determina decisoes de perfuracao.
#
# Ref: docs/reference/metricas_avancadas.md secao 1
# ════════════════════════════════════════════════════════════════════════


def interface_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
    tolerance: int = 3,
) -> InterfaceReport:
    """Deteccao de interfaces + analise de sharpness (C50).

    Detecta interfaces onde |y_true[i+1] - y_true[i]| > threshold.
    Verifica se y_pred possui transicao correspondente dentro de
    +-tolerance pontos. Calcula sharpness = max|Dy_pred| / max|Dy_true|
    na vizinhanca de cada interface.

    Args:
        y_true: Array (N, seq, 2) com valores reais em dominio log10.
        y_pred: Array (N, seq, 2) com predicoes em dominio log10.
        threshold: Limiar de gradiente para deteccao de interface.
            Valor 0.5 em log10 corresponde a ~3.16x de contraste
            de resistividade entre camadas adjacentes.
        tolerance: Numero de pontos de tolerancia para matching.
            Uma interface pred e considerada correta se esta dentro
            de +-tolerance pontos da interface true.

    Returns:
        InterfaceReport com metricas de deteccao e sharpness.

    Raises:
        ValueError: Se shapes sao invalidos ou incompativeis.

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-exports interface_metrics
            - tests/test_evaluation.py: testes de deteccao de interfaces
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """
    y_true, y_pred = _validate_3d_paired(y_true, y_pred)
    n_samples, seq_len, _ = y_true.shape

    # D7: detecta interfaces em true e pred usando mesmo threshold
    true_interfaces = _detect_interfaces(y_true, threshold=threshold)
    pred_interfaces = _detect_interfaces(y_pred, threshold=threshold)

    total_true = sum(len(ifs) for ifs in true_interfaces)
    total_pred = sum(len(ifs) for ifs in pred_interfaces)

    detected = 0
    sharpness_ratios: List[float] = []
    matched_pred_global: int = 0

    for sample_idx in range(n_samples):
        t_ifs = true_interfaces[sample_idx]
        p_ifs = pred_interfaces[sample_idx]
        matched_pred_in_sample: set = set()

        for t_idx in t_ifs:
            # D7: busca interface pred dentro de +-tolerance
            best_match = None
            best_dist = tolerance + 1

            for p_idx in p_ifs:
                dist = abs(p_idx - t_idx)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_match = p_idx

            if best_match is not None:
                detected += 1
                matched_pred_in_sample.add(best_match)

                # D7: sharpness ratio — max|Dy_pred| / max|Dy_true| na vizinhanca
                lo = max(0, t_idx - tolerance)
                hi = min(seq_len - 1, t_idx + tolerance + 1)

                # Gradientes true e pred na vizinhanca da interface
                true_grad = np.abs(np.diff(y_true[sample_idx, lo:hi], axis=0))
                pred_grad = np.abs(np.diff(y_pred[sample_idx, lo:hi], axis=0))

                max_true_grad = np.max(true_grad) if true_grad.size > 0 else EPS
                max_pred_grad = np.max(pred_grad) if pred_grad.size > 0 else 0.0

                # Protecao contra divisao por zero
                ratio = max_pred_grad / max(max_true_grad, EPS)
                sharpness_ratios.append(float(ratio))

            # endif best_match
        # endfor t_idx

        matched_pred_global += len(matched_pred_in_sample)
    # endfor sample_idx

    # D7: taxas de deteccao e falso positivo
    detection_rate = detected / max(total_true, 1)
    sharpness_mean = float(np.mean(sharpness_ratios)) if sharpness_ratios else 0.0

    # Falsos positivos: interfaces pred que nao correspondem a nenhuma true
    false_positives = total_pred - matched_pred_global
    false_positive_rate = false_positives / max(total_pred, 1)

    report = InterfaceReport(
        n_interfaces_true=total_true,
        n_interfaces_pred=total_pred,
        detection_rate=float(detection_rate),
        sharpness_ratio_mean=float(sharpness_mean),
        false_positive_rate=float(false_positive_rate),
    )

    logger.info(
        "interface_metrics: true=%d, pred=%d, det_rate=%.3f, "
        "sharpness=%.3f, fp_rate=%.3f",
        report.n_interfaces_true,
        report.n_interfaces_pred,
        report.detection_rate,
        report.sharpness_ratio_mean,
        report.false_positive_rate,
    )
    return report


# ════════════════════════════════════════════════════════════════════════
# C51: ERROR BY RESISTIVITY BAND — RMSE estratificado por banda
#
# Segmenta o erro por faixas de resistividade em dominio log10, permitindo
# identificar se o modelo performa melhor em resistividades baixas (folhelhos)
# ou altas (carbonatos/arenitos limpos). Bandas default cobrem 5 ordens de
# grandeza (0.1 a 10000 Ohm.m), representativas de cenarios de geosteering.
#
# Ref: docs/reference/metricas_avancadas.md secao 2
# ════════════════════════════════════════════════════════════════════════


def error_by_resistivity_band(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bands: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """RMSE estratificado por banda de resistividade em log10 Ohm.m (C51).

    Segmenta pontos de y_true em bandas de resistividade e calcula RMSE
    separado para cada banda. Permite identificar faixas de resistividade
    onde o modelo tem melhor/pior desempenho.

    Bandas default (log10 Ohm.m):
        (-1, 0)  → 0.1-1 Ohm.m     (folhelhos condutivos)
        (0, 1)   → 1-10 Ohm.m      (folhelhos/siltitos)
        (1, 2)   → 10-100 Ohm.m    (arenitos)
        (2, 3)   → 100-1000 Ohm.m  (carbonatos)
        (3, 4)   → 1000-10000 Ohm.m (evaporitos/carbonatos limpos)

    Args:
        y_true: Array (N, seq, 2) com valores reais em dominio log10.
        y_pred: Array (N, seq, 2) com predicoes em dominio log10.
        bands: Lista de tuplas (lo, hi) definindo bandas em log10 Ohm.m.
            Default: [(-1,0), (0,1), (1,2), (2,3), (3,4)].

    Returns:
        Dict mapeando label da banda (ex: "[0.0, 1.0)") para dict com:
            "rmse" (float): RMSE na banda.
            "count" (int): Numero de pontos na banda.
            "pct" (float): Percentual de pontos na banda.

    Raises:
        ValueError: Se shapes sao invalidos ou incompativeis.

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-exports error_by_resistivity_band
            - tests/test_evaluation.py: testes de estratificacao por banda
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """
    y_true, y_pred = _validate_3d_paired(y_true, y_pred)

    if bands is None:
        bands = [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4)]

    # D7: flatten para facilitar indexacao — preserva pareamento
    y_true_flat = y_true.reshape(-1)  # (N*seq*2,)
    y_pred_flat = y_pred.reshape(-1)  # (N*seq*2,)
    total_points = len(y_true_flat)

    result: Dict[str, Dict[str, float]] = {}

    for lo, hi in bands:
        label = f"[{lo:.1f}, {hi:.1f})"

        # D7: mascara para pontos dentro da banda
        mask = (y_true_flat >= lo) & (y_true_flat < hi)
        count = int(np.sum(mask))

        if count == 0:
            result[label] = {
                "rmse": 0.0,
                "count": 0,
                "pct": 0.0,
            }
            logger.debug("error_by_resistivity_band: banda %s sem pontos", label)
        else:
            errors = y_true_flat[mask] - y_pred_flat[mask]
            rmse = float(np.sqrt(np.mean(errors**2)))
            pct = 100.0 * count / max(total_points, 1)
            result[label] = {
                "rmse": rmse,
                "count": count,
                "pct": pct,
            }
        # endif count
    # endfor bands

    logger.info(
        "error_by_resistivity_band: %d bandas avaliadas, total=%d pontos",
        len(bands),
        total_points,
    )
    return result


# ════════════════════════════════════════════════════════════════════════
# C52: ERROR BY ANISOTROPY — RMSE por razao lambda = rho_v/rho_h
#
# Classifica pontos por grau de anisotropia (lambda = rho_v/rho_h) e
# calcula RMSE para cada categoria. Anisotropia extrema e mais dificil
# de inverter pois as duas componentes divergem significativamente.
# Calculo em dominio linear (10^y) para razao fisicamente significativa.
#
# Ref: docs/reference/metricas_avancadas.md secao 3
# ════════════════════════════════════════════════════════════════════════


def error_by_anisotropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """RMSE por razao de anisotropia lambda = rho_v / rho_h (C52).

    Converte y_true para dominio linear (Ohm.m) para calcular a razao
    de anisotropia lambda = rho_v / rho_h. Classifica pontos em 4 bins
    e calcula RMSE (em dominio log10) para cada bin.

    Bins de anisotropia (lambda = rho_v/rho_h):
        quasi-isotropic: [1.0, 1.5)  — camadas homogeneas
        moderate:        [1.5, 3.0)  — anisotropia moderada
        high:            [3.0, 5.0)  — anisotropia alta
        extreme:         [5.0, inf)  — anisotropia extrema

    Args:
        y_true: Array (N, seq, 2) com valores reais em dominio log10.
            Canal 0 = rho_h (horizontal), canal 1 = rho_v (vertical).
        y_pred: Array (N, seq, 2) com predicoes em dominio log10.

    Returns:
        Dict mapeando nome do bin para dict com:
            "rmse" (float): RMSE (em log10) para pontos no bin.
            "count" (int): Numero de pontos no bin.
            "pct" (float): Percentual de pontos no bin.

    Raises:
        ValueError: Se shapes sao invalidos ou incompativeis.

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-exports error_by_anisotropy
            - tests/test_evaluation.py: testes de segmentacao por anisotropia
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """
    y_true, y_pred = _validate_3d_paired(y_true, y_pred)

    # D7: converte para dominio linear (Ohm.m) para calcular razao
    rho_h_true = np.power(10.0, y_true[..., 0])  # (N, seq)
    rho_v_true = np.power(10.0, y_true[..., 1])  # (N, seq)

    # D7: razao de anisotropia — protege divisao por zero
    lambda_ratio = rho_v_true / np.maximum(rho_h_true, EPS)  # (N, seq)

    # Bins de anisotropia com limites e labels
    bins = [
        ("quasi-isotropic", 1.0, 1.5),
        ("moderate", 1.5, 3.0),
        ("high", 3.0, 5.0),
        ("extreme", 5.0, np.inf),
    ]

    # D7: erro total (ambas componentes) por ponto
    error_sq = (y_true - y_pred) ** 2  # (N, seq, 2)
    mse_per_point = np.mean(error_sq, axis=-1)  # (N, seq) — media rho_h + rho_v

    total_points = lambda_ratio.size
    result: Dict[str, Dict[str, float]] = {}

    for label, lo, hi in bins:
        mask = (lambda_ratio >= lo) & (lambda_ratio < hi)
        count = int(np.sum(mask))

        if count == 0:
            result[label] = {"rmse": 0.0, "count": 0, "pct": 0.0}
            logger.debug("error_by_anisotropy: bin '%s' sem pontos", label)
        else:
            rmse = float(np.sqrt(np.mean(mse_per_point[mask])))
            pct = 100.0 * count / max(total_points, 1)
            result[label] = {"rmse": rmse, "count": count, "pct": pct}
        # endif count
    # endfor bins

    logger.info(
        "error_by_anisotropy: %d bins avaliados, total=%d pontos",
        len(bins),
        total_points,
    )
    return result


# ════════════════════════════════════════════════════════════════════════
# C53: SPATIAL ERROR PROFILE — RMSE(z) ao longo da sequencia
#
# Calcula RMSE para cada índice de profundidade (z) ao longo dos seq_len
# pontos de medição. Útil para detectar efeitos de borda no início e
# fim da sequencia, onde redes convolucionais/causais tendem a ter
# erros maiores por falta de contexto vizinho.
#
# Ref: docs/reference/metricas_avancadas.md secao 4
# ════════════════════════════════════════════════════════════════════════


def spatial_error_profile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """RMSE(z) ao longo dos pontos de medicao da sequencia (C53).

    Calcula RMSE para cada indice de profundidade, mediando sobre todas
    as amostras e componentes. Retorna perfil 1D de tamanho seq_len.

    Args:
        y_true: Array (N, seq, 2) com valores reais em dominio log10.
        y_pred: Array (N, seq, 2) com predicoes em dominio log10.

    Returns:
        Array 1D (seq_len,) com RMSE por indice de profundidade.
        Valores maiores nas bordas indicam efeitos de borda.

    Raises:
        ValueError: Se shapes sao invalidos ou incompativeis.

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-exports spatial_error_profile
            - tests/test_evaluation.py: testes de perfil espacial
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """
    y_true, y_pred = _validate_3d_paired(y_true, y_pred)
    seq_len = y_true.shape[1]

    # D7: MSE por indice de profundidade, mediado sobre amostras e componentes
    # error_sq shape: (N, seq, 2) → mean sobre (N, 2) → (seq,)
    error_sq = (y_true - y_pred) ** 2  # (N, seq, 2)
    mse_per_z = np.mean(error_sq, axis=(0, 2))  # (seq,)
    rmse_per_z = np.sqrt(mse_per_z)  # (seq,)

    logger.info(
        "spatial_error_profile: seq_len=%d, RMSE min=%.4f, max=%.4f, mean=%.4f",
        seq_len,
        float(np.min(rmse_per_z)),
        float(np.max(rmse_per_z)),
        float(np.mean(rmse_per_z)),
    )
    return rmse_per_z


# ════════════════════════════════════════════════════════════════════════
# C54: PHYSICAL COHERENCE CHECK — Verificacao TIV (rho_v >= rho_h)
#
# Em meios TIV (Transversalmente Isotropicos Verticais), a resistividade
# vertical e sempre maior ou igual a horizontal. Predicoes que violam
# esta restricao sao fisicamente inconsistentes e indicam que o modelo
# nao aprendeu a restricao fisica fundamental.
#
# Operacao em dominio log10: y_pred[..., 1] >= y_pred[..., 0]
# equivale a rho_v >= rho_h no dominio linear.
#
# Ref: docs/reference/metricas_avancadas.md secao 5
# ════════════════════════════════════════════════════════════════════════


def physical_coherence_check(
    y_pred: np.ndarray,
) -> CoherenceReport:
    """Verificacao de coerencia fisica TIV: rho_v >= rho_h (C54).

    No dominio log10: y_pred[..., 1] >= y_pred[..., 0] para todo ponto.
    Conta e reporta violacoes onde rho_v < rho_h.

    Args:
        y_pred: Array (N, seq, 2) com predicoes em dominio log10.
            Canal 0 = log10(rho_h), canal 1 = log10(rho_v).

    Returns:
        CoherenceReport com contagem e magnitude das violacoes.

    Raises:
        ValueError: Se shape e invalido.

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-exports physical_coherence_check
            - tests/test_evaluation.py: testes de coerencia TIV
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """
    y_pred = _validate_3d_single(y_pred, name="y_pred")

    # D7: log10(rho_h) = canal 0, log10(rho_v) = canal 1
    rho_h_log = y_pred[..., 0]  # (N, seq)
    rho_v_log = y_pred[..., 1]  # (N, seq)

    total_points = rho_h_log.size

    # D7: violacao onde rho_v < rho_h (em log10: canal 1 < canal 0)
    violation_mask = rho_v_log < rho_h_log
    violations = int(np.sum(violation_mask))
    violation_rate = violations / max(total_points, 1)

    # D7: magnitude media da violacao (rho_h - rho_v, positivo onde violado)
    if violations > 0:
        violation_magnitudes = rho_h_log[violation_mask] - rho_v_log[violation_mask]
        mean_magnitude = float(np.mean(violation_magnitudes))
    else:
        mean_magnitude = 0.0

    report = CoherenceReport(
        total_points=total_points,
        violations=violations,
        violation_rate=float(violation_rate),
        mean_violation_magnitude=mean_magnitude,
    )

    if violations > 0:
        logger.warning(
            "physical_coherence_check: %d violacoes TIV (%.2f%%) "
            "com magnitude media %.4f log10",
            violations,
            violation_rate * 100.0,
            mean_magnitude,
        )
    else:
        logger.info(
            "physical_coherence_check: 0 violacoes TIV em %d pontos",
            total_points,
        )
    return report


# ════════════════════════════════════════════════════════════════════════
# C55: STABILITY ANALYSIS — Perturbacao → variancia de predicoes
#
# Avalia a robustez do modelo injetando ruido gaussiano(sigma) no input
# n_perturbations vezes e medindo a variancia das predicoes. Modelos
# estaveis devem ter baixa variancia mesmo sob perturbacoes moderadas.
#
# NOTA: Esta funcao requer TensorFlow (lazy import) para model.predict().
# Todas as outras funcoes deste modulo sao NumPy-only.
#
# Ref: docs/reference/metricas_avancadas.md secao 6
# ════════════════════════════════════════════════════════════════════════


def stability_analysis(
    model,
    x_test: np.ndarray,
    *,
    config: Optional["PipelineConfig"] = None,
    n_perturbations: int = 10,
    sigma: float = 0.01,
) -> StabilityReport:
    """Analise de estabilidade via perturbacao gaussiana (C55).

    Adiciona ruido gaussiano(sigma) ao x_test n_perturbations vezes,
    gera predicoes para cada versao perturbada, e mede a variancia
    das predicoes. Alta variancia indica sensibilidade excessiva.

    Args:
        model: Modelo Keras treinado (tf.keras.Model). Deve aceitar
            x_test como input e retornar predicoes (N, seq, 2).
        x_test: Array (N, seq, n_features) com dados de teste.
        config: PipelineConfig (opcional, para metadados de logging).
        n_perturbations: Numero de perturbacoes a aplicar. Mais
            perturbacoes dao estimativa mais estavel da variancia.
        sigma: Desvio padrao do ruido gaussiano aditivo.
            Valor 0.01 corresponde a ~1% de perturbacao relativa.

    Returns:
        StabilityReport com variancia media, maxima, e perfil por
        profundidade.

    Raises:
        ImportError: Se TensorFlow nao esta instalado.
        ValueError: Se x_test nao e 3D.

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-exports stability_analysis
            - tests/test_evaluation.py: testes de perturbacao
        Ref: docs/ARCHITECTURE_v2.md secao 8.3.
    """
    # D7: lazy import de TensorFlow — nao requerido no nivel do modulo
    try:
        import tensorflow as tf  # noqa: F811
    except ImportError as e:
        raise ImportError(
            "stability_analysis requer TensorFlow. " "Instale com: pip install tensorflow"
        ) from e

    x_test = np.asarray(x_test, dtype=np.float32)
    if x_test.ndim != 3:
        raise ValueError(
            f"Shape esperado (N, seq, n_features) para x_test, "
            f"recebido shape={x_test.shape} (ndim={x_test.ndim})"
        )

    n_samples, seq_len, n_features = x_test.shape
    rng = np.random.default_rng()

    logger.info(
        "stability_analysis: %d perturbacoes, sigma=%.4f, " "x_test shape=%s",
        n_perturbations,
        sigma,
        x_test.shape,
    )

    # D7: coleta predicoes para cada perturbacao
    all_predictions: List[np.ndarray] = []

    for i in range(n_perturbations):
        noise = rng.normal(0.0, sigma, size=x_test.shape).astype(np.float32)
        x_perturbed = x_test + noise
        # D7: model.predict retorna (N, seq, 2)
        y_pred = model.predict(x_perturbed, verbose=0)
        all_predictions.append(np.asarray(y_pred, dtype=np.float64))

    # D7: stack para (n_perturbations, N, seq, 2)
    preds_stack = np.stack(all_predictions, axis=0)

    # D7: variancia ao longo do eixo de perturbacoes (eixo 0)
    variance = np.var(preds_stack, axis=0)  # (N, seq, 2)

    mean_var = float(np.mean(variance))
    max_var = float(np.max(variance))

    # D7: perfil de sensibilidade por profundidade — media sobre (N, 2)
    sensitivity = np.mean(variance, axis=(0, 2))  # (seq,)

    report = StabilityReport(
        mean_variance=mean_var,
        max_variance=max_var,
        sensitivity_profile=sensitivity,
    )

    logger.info(
        "stability_analysis: mean_var=%.6f, max_var=%.6f",
        mean_var,
        max_var,
    )
    return report
