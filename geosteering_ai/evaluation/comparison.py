# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/comparison.py                                          ║
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
# ║    • ComparisonResult dataclass: ranking multi-modelo por metrica         ║
# ║    • compare_models: recebe dict {nome: MetricsReport}, retorna ranking  ║
# ║    • Suporta ranking por R2 (desc), RMSE (asc), MAE (asc)               ║
# ║    • Logging estruturado do ranking final                                 ║
# ║                                                                            ║
# ║  Dependencias: evaluation/metrics.py (MetricsReport)                      ║
# ║  Exports: ~2 (ComparisonResult, compare_models)                           ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8.2                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Comparacao e ranking de modelos por metricas de avaliacao.

Recebe um dicionario {nome_modelo: MetricsReport} e produz um ranking
ordenado pela metrica escolhida (R2, RMSE ou MAE). Util para comparar
diferentes arquiteturas (ex: ResNet_18 vs UNet vs Transformer) ou
diferentes configuracoes de hiperparametros.

Metricas de ranking suportadas:

    ┌──────────┬──────────────────────────────────────────────────────────┐
    │ Metrica  │ Ordenacao                                                │
    ├──────────┼──────────────────────────────────────────────────────────┤
    │ r2       │ Descendente (maior R2 = melhor)                          │
    │ r2_rh    │ Descendente (maior R2 rho_h = melhor)                    │
    │ r2_rv    │ Descendente (maior R2 rho_v = melhor)                    │
    │ rmse     │ Ascendente (menor RMSE = melhor)                         │
    │ rmse_rh  │ Ascendente (menor RMSE rho_h = melhor)                   │
    │ rmse_rv  │ Ascendente (menor RMSE rho_v = melhor)                   │
    │ mae      │ Ascendente (menor MAE = melhor)                          │
    │ mbe      │ Ascendente por |MBE| (menor bias = melhor)               │
    │ mape     │ Ascendente (menor MAPE = melhor)                         │
    └──────────┴──────────────────────────────────────────────────────────┘

Note:
    Referenciado em:
        - evaluation/__init__.py: re-exports ComparisonResult, compare_models
        - tests/test_evaluation.py: testes de ranking e best_model
    Ref: docs/ARCHITECTURE_v2.md secao 8.2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from geosteering_ai.evaluation.metrics import MetricsReport

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Container de comparacao ---
    "ComparisonResult",
    # --- Funcao principal ---
    "compare_models",
]


# ════════════════════════════════════════════════════════════════════════
# CONSTANTES — Configuracao de ranking por metrica
#
# Metricas "higher is better" (R2): ordenacao descendente.
# Metricas "lower is better" (RMSE, MAE, MAPE): ordenacao ascendente.
# MBE: ordena por |MBE| (menor bias absoluto = melhor).
#
# Ref: docs/reference/metricas.md
# ════════════════════════════════════════════════════════════════════════

# Metricas onde MAIOR valor indica melhor modelo
_HIGHER_IS_BETTER = {"r2", "r2_rh", "r2_rv"}

# Metricas validas para ranking (todas as metricas do MetricsReport)
_VALID_METRICS = {"r2", "r2_rh", "r2_rv", "rmse", "rmse_rh", "rmse_rv",
                  "mae", "mbe", "mape"}


# ════════════════════════════════════════════════════════════════════════
# COMPARISON RESULT — Container dataclass para resultado de comparacao
#
# Armazena o ranking completo de modelos, incluindo o melhor modelo
# e o dicionario de MetricsReport para consulta posterior.
#
# Ref: docs/ARCHITECTURE_v2.md secao 8.2
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    """Resultado da comparacao entre multiplos modelos.

    Contem o ranking ordenado pela metrica escolhida, o nome do melhor
    modelo, e acesso a todas as MetricsReport individuais.

    Attributes:
        model_names: Lista de nomes de modelos (ordem original).
        metrics: Dicionario {nome_modelo: MetricsReport} para consulta.
        best_model: Nome do modelo com melhor desempenho na metrica.
        ranking: Lista de nomes ordenados do melhor ao pior.
        ranking_metric: Nome da metrica usada para ordenacao.

    Example:
        >>> from geosteering_ai.evaluation.metrics import MetricsReport
        >>> results = {
        ...     "ResNet_18": MetricsReport(
        ...         r2=0.98, r2_rh=0.985, r2_rv=0.975,
        ...         rmse=0.02, rmse_rh=0.018, rmse_rv=0.022,
        ...         mae=0.015, mbe=-0.001, mape=1.5,
        ...     ),
        ...     "UNet": MetricsReport(
        ...         r2=0.95, r2_rh=0.96, r2_rv=0.94,
        ...         rmse=0.04, rmse_rh=0.035, rmse_rv=0.045,
        ...         mae=0.03, mbe=0.005, mape=3.0,
        ...     ),
        ... }
        >>> comparison = compare_models(results, metric="r2")
        >>> comparison.best_model
        'ResNet_18'
        >>> comparison.ranking
        ['ResNet_18', 'UNet']

    Note:
        Referenciado em:
            - evaluation/comparison.py: compare_models (retorno)
            - evaluation/__init__.py: re-export
            - tests/test_evaluation.py: testes de ranking
        Ref: docs/ARCHITECTURE_v2.md secao 8.2.
    """
    model_names: List[str]
    metrics: Dict[str, MetricsReport]
    best_model: str
    ranking: List[str]
    ranking_metric: str = "r2"

    def summary(self) -> str:
        """String formatada com ranking de modelos.

        Gera tabela de ranking com a metrica usada para ordenacao
        e a posicao de cada modelo. Inclui marcador para o melhor.

        Returns:
            String multi-linha com ranking formatado.

        Note:
            Referenciado em:
                - evaluation/comparison.py: compare_models (logging)
            Ref: docs/ARCHITECTURE_v2.md secao 8.2.
        """
        lines = [
            "=" * 65,
            f"  RANKING DE MODELOS (por {self.ranking_metric})",
            "=" * 65,
        ]

        for i, name in enumerate(self.ranking, start=1):
            report = self.metrics[name]
            metric_val = report.to_dict()[self.ranking_metric]

            # MBE: exibir valor absoluto entre parenteses
            if self.ranking_metric == "mbe":
                marker = " <-- BEST" if name == self.best_model else ""
                lines.append(
                    f"  #{i}  {name:<30s}  "
                    f"{self.ranking_metric}={metric_val:>+10.6f}  "
                    f"(|MBE|={abs(metric_val):.6f}){marker}"
                )
            else:
                marker = " <-- BEST" if name == self.best_model else ""
                lines.append(
                    f"  #{i}  {name:<30s}  "
                    f"{self.ranking_metric}={metric_val:>10.6f}{marker}"
                )

        lines.append("=" * 65)
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# COMPARE MODELS — Funcao principal de comparacao
#
# Recebe dicionario {nome: MetricsReport}, ordena pela metrica
# especificada e retorna ComparisonResult com ranking.
#
# Direcao de ordenacao:
#   - R2, R2_rh, R2_rv: descendente (higher is better)
#   - RMSE, MAE, MAPE: ascendente (lower is better)
#   - MBE: ascendente por |MBE| (menor bias absoluto)
#
# Ref: docs/ARCHITECTURE_v2.md secao 8.2
# ════════════════════════════════════════════════════════════════════════

def compare_models(
    results: Dict[str, MetricsReport],
    *,
    metric: str = "r2",
) -> ComparisonResult:
    """Compara multiplos modelos por metricas de avaliacao.

    Recebe dicionario mapeando nomes de modelos a seus MetricsReport
    e retorna ranking ordenado pela metrica escolhida. O melhor modelo
    e identificado automaticamente com base na direcao da metrica.

    Args:
        results: Dicionario {nome_modelo: MetricsReport}. Cada entrada
            contem as metricas de avaliacao de um modelo treinado.
            Deve conter ao menos 1 modelo.
        metric: Metrica para ranking. Opcoes: "r2", "r2_rh", "r2_rv",
            "rmse", "rmse_rh", "rmse_rv", "mae", "mbe", "mape".
            Default: "r2" (coeficiente de determinacao global).

    Returns:
        ComparisonResult com ranking, best_model, e acesso a metricas.

    Raises:
        ValueError: Se results esta vazio, ou se metric nao e valida.

    Example:
        >>> from geosteering_ai.evaluation.metrics import MetricsReport
        >>> r1 = MetricsReport(
        ...     r2=0.98, r2_rh=0.985, r2_rv=0.975,
        ...     rmse=0.02, rmse_rh=0.018, rmse_rv=0.022,
        ...     mae=0.015, mbe=-0.001, mape=1.5,
        ... )
        >>> r2 = MetricsReport(
        ...     r2=0.95, r2_rh=0.96, r2_rv=0.94,
        ...     rmse=0.04, rmse_rh=0.035, rmse_rv=0.045,
        ...     mae=0.03, mbe=0.005, mape=3.0,
        ... )
        >>> comp = compare_models({"ResNet_18": r1, "UNet": r2})
        >>> comp.best_model
        'ResNet_18'

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-export
            - tests/test_evaluation.py: testes de ranking e edge cases
        Ref: docs/ARCHITECTURE_v2.md secao 8.2.
    """
    # ── Validacao de entrada ──
    if not results:
        raise ValueError(
            "results esta vazio. Fornecer ao menos 1 modelo para comparacao."
        )

    if metric not in _VALID_METRICS:
        raise ValueError(
            f"Metrica invalida: '{metric}'. "
            f"Opcoes validas: {sorted(_VALID_METRICS)}"
        )

    logger.info(
        "Comparando %d modelos pela metrica '%s'...",
        len(results),
        metric,
    )

    # ── Extrair valor da metrica para cada modelo ──
    model_names = list(results.keys())
    metric_values = {}
    for name, report in results.items():
        metric_values[name] = report.to_dict()[metric]

    # ── Ordenar pela metrica ──
    # MBE: ordena por valor absoluto (menor |bias| = melhor)
    if metric == "mbe":
        sorted_names = sorted(
            model_names,
            key=lambda n: abs(metric_values[n]),
        )
    elif metric in _HIGHER_IS_BETTER:
        # R2: descendente (maior = melhor)
        sorted_names = sorted(
            model_names,
            key=lambda n: metric_values[n],
            reverse=True,
        )
    else:
        # RMSE, MAE, MAPE: ascendente (menor = melhor)
        sorted_names = sorted(
            model_names,
            key=lambda n: metric_values[n],
        )

    best_model = sorted_names[0]

    comparison = ComparisonResult(
        model_names=model_names,
        metrics=results,
        best_model=best_model,
        ranking=sorted_names,
        ranking_metric=metric,
    )

    # ── Logging do ranking ──
    logger.info(
        "Ranking de modelos:\n%s",
        comparison.summary(),
    )

    return comparison
