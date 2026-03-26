# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/geosteering_report.py                                 ║
# ║  Bloco: 8 — Evaluation (C73)                                            ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • generate_geosteering_report: relatorio Markdown especifico para     ║
# ║      operacoes de geosteering (6 secoes)                                  ║
# ║    • Well Summary, Geosteering Metrics, Offline vs Realtime,             ║
# ║      Uncertainty Analysis, Figures, Steering Recommendations             ║
# ║    • NumPy-only: sem dependencia de TensorFlow                           ║
# ║                                                                            ║
# ║  Dependencias: json, datetime, os (stdlib apenas)                         ║
# ║  Exports: ~1 (generate_geosteering_report)                              ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8.9                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C73)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Geracao de relatorio Markdown especifico para geosteering.

Produz um relatorio Markdown com 6 secoes especializadas para
operacoes de geosteering em tempo real, complementando o relatorio
generico de ``evaluation/report.py`` (C65).

Estrutura do relatorio:

    ┌────────────────────────────────────────────────────────────────────┐
    │  Relatorio de Geosteering                                         │
    ├────────────────────────────────────────────────────────────────────┤
    │  1. Well Summary       (model_type, causal mode, latencia)        │
    │  2. Geosteering Metrics (DTB error, look-ahead accuracy)         │
    │  3. Offline vs Realtime (delta R2, delta RMSE, latencia)         │
    │  4. Uncertainty Analysis (resumo de metricas UQ)                  │
    │  5. Figures             (imagens linkadas)                        │
    │  6. Steering Recs.     (recomendacoes baseadas em metricas)      │
    └────────────────────────────────────────────────────────────────────┘

Cada secao e None-safe: omitida ou com placeholder se dados
nao fornecidos.

Note:
    Referenciado em:
        - evaluation/__init__.py: re-export generate_geosteering_report
        - evaluation/realtime_comparison.py: ModeComparisonResult alimenta secao 3
        - evaluation/geosteering_metrics.py: GeoMetrics alimenta secao 2
        - tests/test_evaluation.py: testes de geracao de relatorio geosteering
    Ref: docs/ARCHITECTURE_v2.md secao 8.9.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.evaluation.geosteering_metrics import GeoMetrics
    from geosteering_ai.evaluation.realtime_comparison import (
        ModeComparisonResult,
    )

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Geracao de relatorio geosteering ---
    "generate_geosteering_report",
]


# ════════════════════════════════════════════════════════════════════════
# CONSTANTES — Limiares para recomendacoes de steering
#
# Limiares empiricos para classificar desempenho do modelo em
# cenario de geosteering. Baseados em experiencia operacional.
# ════════════════════════════════════════════════════════════════════════

# Degradacao maxima aceitavel de R2 (offline → realtime)
_DELTA_R2_ACCEPTABLE = 0.05

# Degradacao maxima aceitavel de RMSE (em log10 decades)
_DELTA_RMSE_ACCEPTABLE = 0.10

# Acuracia minima aceitavel de look-ahead
_LOOK_AHEAD_MIN_ACCEPTABLE = 0.70

# Latencia maxima aceitavel para geosteering realtime (ms)
_LATENCY_MAX_ACCEPTABLE_MS = 50.0

# Epsilon para estabilidade numerica (float32)
_EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════
# FUNCAO PRINCIPAL: generate_geosteering_report
#
# Gera relatorio Markdown com 6 secoes especializadas para geosteering.
# Cada secao e construida por funcao auxiliar _build_section_*.
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │ Fluxo de geracao                                                   │
# │                                                                     │
# │  config ──────────────→ Secao 1 (Well Summary)                    │
# │  geo_metrics ─────────→ Secao 2 (Geosteering Metrics)            │
# │  mode_comparison ─────→ Secao 3 (Offline vs Realtime)            │
# │  (metricas internas) ─→ Secao 4 (Uncertainty Analysis)           │
# │  figure_paths ────────→ Secao 5 (Figures)                        │
# │  (metricas + limiares)→ Secao 6 (Steering Recommendations)       │
# │                                                                     │
# │  Resultado: string Markdown + salvar opcional em arquivo           │
# └──────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def generate_geosteering_report(
    config: PipelineConfig,
    *,
    geo_metrics: Optional[GeoMetrics] = None,
    mode_comparison: Optional[ModeComparisonResult] = None,
    figure_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Gera relatorio Markdown especifico para operacoes de geosteering.

    Produz um documento Markdown com 6 secoes focadas em metricas
    operacionais de geosteering: sumario do poco, metricas DTB e
    look-ahead, comparacao offline vs realtime, analise de incerteza,
    figuras, e recomendacoes de steering.

    Cada secao e None-safe: se os dados correspondentes nao forem
    fornecidos, a secao exibe um placeholder informativo.

    Args:
        config: PipelineConfig do experimento. Obrigatorio. Usado
            para secao 1 (Well Summary) com model_type, causal mode,
            e parametros do pipeline.
        geo_metrics: GeoMetrics com metricas de geosteering (DTB error,
            look-ahead accuracy). Se fornecido, alimenta secoes 2
            (metricas) e 6 (recomendacoes). Opcional.
        mode_comparison: ModeComparisonResult com metricas comparativas
            offline vs realtime. Se fornecido, alimenta secao 3
            (comparacao) e 6 (recomendacoes). Opcional.
        figure_paths: Lista de caminhos para figuras geradas
            (curtain plots, DTB profiles, dashboards). Cada caminho
            e incluido como link Markdown na secao 5. Opcional.
        output_path: Caminho para salvar o relatorio Markdown.
            Se fornecido, cria diretorios intermediarios e salva.
            Opcional (None = apenas retorna string).

    Returns:
        String Markdown completa do relatorio de geosteering.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig.realtime(model_type="WaveNet")
        >>> md = generate_geosteering_report(config)
        >>> "Well Summary" in md
        True

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-export
            - tests/test_evaluation.py: test_generate_geosteering_report
        Ref: docs/ARCHITECTURE_v2.md secao 8.9.
    """
    # ── Import da versao do pacote (lazy, evita circular) ──
    from geosteering_ai import __version__

    # ── Timestamp de geracao ──
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Construir secoes do relatorio ──
    sections: List[str] = []

    # Titulo principal
    sections.append(
        f"# Relatorio de Geosteering — Geosteering AI v{__version__}\n\n"
        f"Gerado em: {timestamp}\n"
    )

    # Secao 1: Well Summary
    sections.append(_build_section_well_summary(config, mode_comparison))

    # Secao 2: Geosteering Metrics
    sections.append(_build_section_geo_metrics(geo_metrics))

    # Secao 3: Offline vs Realtime
    sections.append(_build_section_mode_comparison(mode_comparison))

    # Secao 4: Uncertainty Analysis
    sections.append(_build_section_uncertainty(geo_metrics, mode_comparison))

    # Secao 5: Figures
    sections.append(_build_section_figures(figure_paths))

    # Secao 6: Steering Recommendations
    sections.append(
        _build_section_recommendations(geo_metrics, mode_comparison)
    )

    # ── Montar documento final ──
    report = "\n---\n\n".join(sections)

    logger.info(
        "Relatorio de geosteering gerado: %d secoes, %d caracteres.",
        len(sections),
        len(report),
    )

    # ── Salvar em arquivo se output_path fornecido ──
    if output_path is not None:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Relatorio de geosteering salvo em: %s", output_path)

    return report


# ════════════════════════════════════════════════════════════════════════
# FUNCOES AUXILIARES: _build_section_*
#
# Cada funcao gera o Markdown de uma secao do relatorio.
# Todas sao None-safe: retornam placeholder se dados ausentes.
#
# Ref: docs/ARCHITECTURE_v2.md secao 8.9
# ════════════════════════════════════════════════════════════════════════

def _build_section_well_summary(
    config: PipelineConfig,
    mode_comparison: Optional[ModeComparisonResult] = None,
) -> str:
    """Secao 1: Well Summary.

    Exibe modelo, modo causal, parametros de pipeline e latencia.

    Args:
        config: PipelineConfig com model_type e flags de causal.
        mode_comparison: ModeComparisonResult para exibir latencia.

    Returns:
        String Markdown da secao.

    Note:
        Referenciado em:
            - evaluation/geosteering_report.py: generate_geosteering_report
        Ref: docs/ARCHITECTURE_v2.md secao 8.9.
    """
    lines = ["## 1. Well Summary\n"]

    config_dict = config.to_dict()

    lines.append(f"- **Arquitetura:** {config.model_type}")
    lines.append(f"- **Loss:** {config.loss_type}")

    # Modo causal
    use_causal = config_dict.get("use_causal_mode", False)
    causal_label = "Causal (realtime)" if use_causal else "Acausal (offline)"
    lines.append(f"- **Modo de Inferencia:** {causal_label}")

    # Frequencia e spacing
    freq_hz = config_dict.get("frequency_hz", 20000.0)
    spacing_m = config_dict.get("spacing_meters", 1.0)
    lines.append(f"- **Frequencia:** {freq_hz:g} Hz")
    lines.append(f"- **Spacing:** {spacing_m:g} m")

    # Sequence length
    seq_len = config_dict.get("sequence_length", 600)
    lines.append(f"- **Sequence Length:** {seq_len}")

    # Feature view
    fv = config_dict.get("feature_view", "N/A")
    lines.append(f"- **Feature View:** {fv}")

    # Latencia (se disponivel via mode_comparison)
    if mode_comparison is not None and mode_comparison.latency_ms >= 0:
        lines.append(f"- **Latencia Realtime:** {mode_comparison.latency_ms:.1f} ms")
    else:
        lines.append("- **Latencia Realtime:** *nao medida*")

    return "\n".join(lines)


def _build_section_geo_metrics(
    geo_metrics: Optional[GeoMetrics] = None,
) -> str:
    """Secao 2: Geosteering Metrics.

    Exibe DTB error e look-ahead accuracy em tabela.

    Args:
        geo_metrics: GeoMetrics com metricas de geosteering.

    Returns:
        String Markdown da secao.

    Note:
        Referenciado em:
            - evaluation/geosteering_report.py: generate_geosteering_report
        Ref: docs/ARCHITECTURE_v2.md secao 8.9.
    """
    lines = ["## 2. Geosteering Metrics\n"]

    if geo_metrics is None:
        lines.append("*Metricas de geosteering nao disponiveis.*")
        return "\n".join(lines)

    gm = geo_metrics.to_dict()

    lines.append("| Metrica | Valor |")
    lines.append("|:--------|------:|")

    # DTB Error
    dtb_mean = gm.get("dtb_error_mean")
    dtb_std = gm.get("dtb_error_std")
    if dtb_mean is not None and not _is_nan(dtb_mean):
        lines.append(f"| DTB Error Mean | {dtb_mean:.6f} |")
    else:
        lines.append("| DTB Error Mean | *N/A (DTB nao fornecido)* |")
    if dtb_std is not None and not _is_nan(dtb_std):
        lines.append(f"| DTB Error Std | {dtb_std:.6f} |")

    # Look-ahead
    la = gm.get("look_ahead_accuracy")
    if la is not None:
        lines.append(f"| Look-Ahead Accuracy | {la:.4f} ({la:.1%}) |")

    n_det = gm.get("n_interfaces_detected")
    n_tot = gm.get("n_interfaces_total")
    if n_det is not None and n_tot is not None:
        lines.append(f"| Interfaces Detectadas | {n_det}/{n_tot} |")

    return "\n".join(lines)


def _build_section_mode_comparison(
    mode_comparison: Optional[ModeComparisonResult] = None,
) -> str:
    """Secao 3: Offline vs Realtime.

    Exibe metricas comparativas e deltas de degradacao.

    Args:
        mode_comparison: ModeComparisonResult com deltas.

    Returns:
        String Markdown da secao.

    Note:
        Referenciado em:
            - evaluation/geosteering_report.py: generate_geosteering_report
        Ref: docs/ARCHITECTURE_v2.md secao 8.9.
    """
    lines = ["## 3. Offline vs Realtime\n"]

    if mode_comparison is None:
        lines.append("*Comparacao offline vs realtime nao disponivel.*")
        return "\n".join(lines)

    mc = mode_comparison.to_dict()

    lines.append("| Metrica | Offline | Realtime | Delta |")
    lines.append("|:--------|-------:|--------:|------:|")

    # R2
    lines.append(
        f"| R2 | {mc['r2_offline']:.6f} | {mc['r2_realtime']:.6f} | "
        f"{mc['delta_r2']:+.6f} |"
    )

    # RMSE
    lines.append(
        f"| RMSE | {mc['rmse_offline']:.6f} | {mc['rmse_realtime']:.6f} | "
        f"{mc['delta_rmse']:+.6f} |"
    )

    # Latencia
    latency = mc.get("latency_ms", -1.0)
    if latency >= 0:
        lines.append(f"\n- **Latencia media realtime:** {latency:.1f} ms")

    # Diagnostico de degradacao
    delta_r2 = mc["delta_r2"]
    delta_rmse = mc["delta_rmse"]
    if delta_r2 > _DELTA_R2_ACCEPTABLE:
        lines.append(
            f"\n> **ATENCAO:** Degradacao de R2 ({delta_r2:.4f}) "
            f"excede limiar aceitavel ({_DELTA_R2_ACCEPTABLE})."
        )
    if delta_rmse > _DELTA_RMSE_ACCEPTABLE:
        lines.append(
            f"\n> **ATENCAO:** Degradacao de RMSE ({delta_rmse:.4f} decades) "
            f"excede limiar aceitavel ({_DELTA_RMSE_ACCEPTABLE})."
        )

    return "\n".join(lines)


def _build_section_uncertainty(
    geo_metrics: Optional[GeoMetrics] = None,
    mode_comparison: Optional[ModeComparisonResult] = None,
) -> str:
    """Secao 4: Uncertainty Analysis.

    Resumo qualitativo de incerteza baseado nas metricas disponiveis.

    Args:
        geo_metrics: GeoMetrics para avaliar confianca.
        mode_comparison: ModeComparisonResult para avaliar estabilidade.

    Returns:
        String Markdown da secao.

    Note:
        Referenciado em:
            - evaluation/geosteering_report.py: generate_geosteering_report
        Ref: docs/ARCHITECTURE_v2.md secao 8.9.
    """
    lines = ["## 4. Uncertainty Analysis\n"]

    if geo_metrics is None and mode_comparison is None:
        lines.append("*Dados de incerteza nao disponiveis.*")
        return "\n".join(lines)

    lines.append(
        "Avaliacao qualitativa da confianca do modelo para operacoes "
        "de geosteering:\n"
    )

    # ── Confianca baseada em look-ahead accuracy ──
    if geo_metrics is not None:
        la = geo_metrics.look_ahead_accuracy
        if la >= 0.90:
            conf_label = "ALTA"
            conf_note = "Modelo antecipa >90% das interfaces."
        elif la >= _LOOK_AHEAD_MIN_ACCEPTABLE:
            conf_label = "MODERADA"
            conf_note = (
                f"Modelo antecipa {la:.0%} das interfaces "
                f"(limiar minimo: {_LOOK_AHEAD_MIN_ACCEPTABLE:.0%})."
            )
        else:
            conf_label = "BAIXA"
            conf_note = (
                f"Modelo antecipa apenas {la:.0%} das interfaces. "
                "Considerar re-treinamento ou ajuste de janela causal."
            )
        lines.append(f"- **Confianca Look-Ahead:** {conf_label} — {conf_note}")

        # DTB confidence
        dtb_mean = geo_metrics.dtb_error_mean
        if not _is_nan(dtb_mean):
            if dtb_mean < 0.05:
                lines.append("- **Confianca DTB:** ALTA — erro medio < 0.05")
            elif dtb_mean < 0.15:
                lines.append(
                    f"- **Confianca DTB:** MODERADA — erro medio = {dtb_mean:.4f}"
                )
            else:
                lines.append(
                    f"- **Confianca DTB:** BAIXA — erro medio = {dtb_mean:.4f} "
                    "(considerar dados de treinamento adicionais)"
                )

    # ── Estabilidade baseada em delta R2/RMSE ──
    if mode_comparison is not None:
        delta_r2 = mode_comparison.delta_r2
        if delta_r2 < 0.02:
            lines.append(
                "- **Estabilidade Offline→Realtime:** ALTA — "
                f"delta R2 = {delta_r2:.4f} (< 0.02)"
            )
        elif delta_r2 < _DELTA_R2_ACCEPTABLE:
            lines.append(
                "- **Estabilidade Offline→Realtime:** MODERADA — "
                f"delta R2 = {delta_r2:.4f}"
            )
        else:
            lines.append(
                "- **Estabilidade Offline→Realtime:** BAIXA — "
                f"delta R2 = {delta_r2:.4f} (> {_DELTA_R2_ACCEPTABLE})"
            )

    return "\n".join(lines)


def _build_section_figures(
    figure_paths: Optional[List[str]] = None,
) -> str:
    """Secao 5: Figures.

    Lista figuras como links Markdown para imagens.

    Args:
        figure_paths: Lista de caminhos de figuras.

    Returns:
        String Markdown da secao.

    Note:
        Referenciado em:
            - evaluation/geosteering_report.py: generate_geosteering_report
        Ref: docs/ARCHITECTURE_v2.md secao 8.9.
    """
    lines = ["## 5. Figuras\n"]

    if not figure_paths:
        lines.append("*Nenhuma figura gerada.*")
        return "\n".join(lines)

    for i, path in enumerate(figure_paths, start=1):
        filename = os.path.basename(path)
        lines.append(f"### Figura {i}: {filename}\n")
        lines.append(f"![{filename}]({path})\n")

    return "\n".join(lines)


def _build_section_recommendations(
    geo_metrics: Optional[GeoMetrics] = None,
    mode_comparison: Optional[ModeComparisonResult] = None,
) -> str:
    """Secao 6: Steering Recommendations.

    Gera recomendacoes automatizadas baseadas nos limiares definidos.

    Args:
        geo_metrics: GeoMetrics para avaliar look-ahead e DTB.
        mode_comparison: ModeComparisonResult para avaliar degradacao.

    Returns:
        String Markdown da secao com recomendacoes.

    Note:
        Referenciado em:
            - evaluation/geosteering_report.py: generate_geosteering_report
        Ref: docs/ARCHITECTURE_v2.md secao 8.9.
    """
    lines = ["## 6. Steering Recommendations\n"]

    recommendations: List[str] = []
    warnings: List[str] = []

    if geo_metrics is None and mode_comparison is None:
        lines.append(
            "*Recomendacoes nao disponiveis — "
            "fornecer geo_metrics e/ou mode_comparison.*"
        )
        return "\n".join(lines)

    # ── Avaliar look-ahead accuracy ──────────────────────────────────
    if geo_metrics is not None:
        la = geo_metrics.look_ahead_accuracy

        if la >= 0.90:
            recommendations.append(
                "Look-ahead accuracy excelente (>90%). "
                "Modelo adequado para geosteering autonomo."
            )
        elif la >= _LOOK_AHEAD_MIN_ACCEPTABLE:
            recommendations.append(
                f"Look-ahead accuracy moderada ({la:.0%}). "
                "Considerar supervisao humana para decisoes criticas."
            )
        else:
            warnings.append(
                f"Look-ahead accuracy insuficiente ({la:.0%} < "
                f"{_LOOK_AHEAD_MIN_ACCEPTABLE:.0%}). "
                "NAO recomendado para geosteering autonomo. "
                "Acoes: (1) aumentar look_ahead_window, "
                "(2) re-treinar com mais dados de interface, "
                "(3) considerar modelo causal-compatible."
            )

        # ── DTB ──
        dtb_mean = geo_metrics.dtb_error_mean
        if not _is_nan(dtb_mean) and dtb_mean > 0.15:
            warnings.append(
                f"DTB error elevado (mean={dtb_mean:.4f}). "
                "Considerar: (1) incluir DTB como target adicional (P5), "
                "(2) aumentar peso de DTB na loss."
            )

    # ── Avaliar degradacao offline → realtime ────────────────────────
    if mode_comparison is not None:
        delta_r2 = mode_comparison.delta_r2
        delta_rmse = mode_comparison.delta_rmse
        latency = mode_comparison.latency_ms

        if delta_r2 > _DELTA_R2_ACCEPTABLE:
            warnings.append(
                f"Degradacao R2 significativa (delta={delta_r2:.4f}). "
                "Acoes: (1) re-treinar com flag causal, "
                "(2) ajustar sliding window, "
                "(3) considerar arquitetura geosteering (WaveNet, GeoResNet)."
            )

        if delta_rmse > _DELTA_RMSE_ACCEPTABLE:
            warnings.append(
                f"Degradacao RMSE significativa (delta={delta_rmse:.4f} decades). "
                "Considerar fine-tuning causal com dados realtime."
            )

        if 0 <= latency <= _LATENCY_MAX_ACCEPTABLE_MS:
            recommendations.append(
                f"Latencia aceitavel ({latency:.1f} ms < "
                f"{_LATENCY_MAX_ACCEPTABLE_MS:.0f} ms). "
                "Modelo viavel para inferencia em tempo real."
            )
        elif latency > _LATENCY_MAX_ACCEPTABLE_MS:
            warnings.append(
                f"Latencia elevada ({latency:.1f} ms > "
                f"{_LATENCY_MAX_ACCEPTABLE_MS:.0f} ms). "
                "Acoes: (1) exportar para TFLite, "
                "(2) reduzir modelo (pruning), "
                "(3) reduzir sequence_length."
            )

        if delta_r2 <= 0.02 and delta_rmse <= 0.05:
            recommendations.append(
                "Transicao offline→realtime com degradacao minima. "
                "Modelo estavel para deploy."
            )

    # ── Montar secao ─────────────────────────────────────────────────
    if recommendations:
        lines.append("### Recomendacoes\n")
        for rec in recommendations:
            lines.append(f"- {rec}")

    if warnings:
        lines.append("\n### Alertas\n")
        for warn_msg in warnings:
            lines.append(f"- **ALERTA:** {warn_msg}")

    if not recommendations and not warnings:
        lines.append("*Sem recomendacoes — metricas dentro dos limiares.*")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# FUNCAO AUXILIAR INTERNA: _is_nan
#
# Verifica se um valor numerico e NaN de forma segura.
# Necessario porque float("nan") != float("nan").
# ════════════════════════════════════════════════════════════════════════

def _is_nan(value: Any) -> bool:
    """Verifica se valor e NaN de forma segura.

    Args:
        value: Valor a verificar. Pode ser float, int, ou outro tipo.

    Returns:
        True se valor e float NaN, False caso contrario.

    Note:
        Referenciado em:
            - evaluation/geosteering_report.py: _build_section_*
        Ref: docs/ARCHITECTURE_v2.md secao 8.9.
    """
    try:
        return float(value) != float(value)
    except (TypeError, ValueError):
        return False
