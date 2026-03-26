# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/report.py                                             ║
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
# ║    • generate_report: gera relatorio Markdown automatizado do experimento ║
# ║    • 6 secoes: sumario, config, treinamento, metricas, figuras, reprod.  ║
# ║    • Retorna string Markdown e opcionalmente salva em arquivo            ║
# ║    • NumPy-only: sem dependencia de TensorFlow                           ║
# ║                                                                            ║
# ║  Dependencias: json, datetime, os (stdlib apenas)                         ║
# ║  Exports: ~1 (generate_report)                                           ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8.6                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C65)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Geracao automatizada de relatorio Markdown pos-treinamento.

Produz um relatorio Markdown completo com 6 secoes que documentam
o experimento de inversao de resistividade: sumario executivo,
configuracao, resultados de treinamento, metricas de avaliacao,
figuras geradas, e informacoes de reprodutibilidade.

Estrutura do relatorio:

    ┌────────────────────────────────────────────────────────────────────┐
    │  Relatorio Markdown                                                │
    ├────────────────────────────────────────────────────────────────────┤
    │  1. Sumario Executivo   (model_type, best R2, RMSE)               │
    │  2. Configuracao        (campos-chave em tabela)                  │
    │  3. Resultado Treino    (epochs, tempo, convergencia)             │
    │  4. Metricas            (tabela completa MetricsReport)           │
    │  5. Figuras             (imagens linkadas, se fornecidas)         │
    │  6. Reprodutibilidade   (config YAML, git info, versao)          │
    └────────────────────────────────────────────────────────────────────┘

O relatorio e gerado como string Markdown e opcionalmente salvo em
arquivo. Todas as secoes sao None-safe (omitidas ou com placeholder
se dados nao fornecidos).

Note:
    Referenciado em:
        - evaluation/__init__.py: re-export generate_report
        - evaluation/manifest.py: manifesto pode alimentar o relatorio
        - tests/test_evaluation.py: testes de geracao de relatorio
    Ref: docs/ARCHITECTURE_v2.md secao 8.6.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Geracao de relatorio ---
    "generate_report",
]


# ════════════════════════════════════════════════════════════════════════
# CAMPOS-CHAVE PARA TABELA DE CONFIGURACAO
#
# Subconjunto dos 121+ campos do PipelineConfig exibido na secao 2
# do relatorio. Selecionados por relevancia para analise rapida.
# Cada tupla: (nome_campo, label_legivel, unidade_ou_nota).
#
# Ref: docs/ARCHITECTURE_v2.md secao 3.1
# ════════════════════════════════════════════════════════════════════════

_KEY_CONFIG_FIELDS = [
    ("model_type", "Arquitetura", ""),
    ("loss_type", "Loss", ""),
    ("learning_rate", "Learning Rate", ""),
    ("epochs", "Epochs Maximo", ""),
    ("batch_size", "Batch Size", ""),
    ("sequence_length", "Sequence Length", "pontos"),
    ("feature_view", "Feature View", ""),
    ("target_scaling", "Target Scaling", ""),
    ("noise_level_max", "Noise Level Max", ""),
    ("use_noise_augmentation", "Noise Augmentation", ""),
    ("use_curriculum_noise", "Curriculum Noise", ""),
    ("use_dual_validation", "Dual Validation", ""),
    ("early_stopping_patience", "Early Stopping Patience", "epochs"),
    ("use_nstage_training", "N-Stage Training", ""),
    ("frequency_hz", "Frequencia", "Hz"),
    ("spacing_meters", "Spacing", "m"),
]


# ════════════════════════════════════════════════════════════════════════
# FUNCAO PRINCIPAL: generate_report
#
# Gera relatorio Markdown completo a partir de PipelineConfig e
# dados opcionais (metricas, manifesto, figuras).
# Cada secao e construida por funcao auxiliar _build_section_*.
#
# Ref: docs/ARCHITECTURE_v2.md secao 8.6
# ════════════════════════════════════════════════════════════════════════

def generate_report(
    config: PipelineConfig,
    *,
    metrics_report: Any = None,
    manifest: Optional[Dict[str, Any]] = None,
    figure_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Gera relatorio Markdown automatizado do experimento.

    Produz um documento Markdown com 6 secoes documentando completamente
    o experimento: sumario executivo, configuracao, treinamento,
    metricas, figuras, e reprodutibilidade. Secoes sem dados disponíveis
    incluem placeholder informativo.

    Args:
        config: PipelineConfig do experimento. Obrigatorio. Usado
            para secoes 1 (sumario), 2 (configuracao) e 6 (reprod.).
        metrics_report: MetricsReport com metricas de avaliacao. Se
            fornecido, deve ter metodo to_dict(). Usado nas secoes
            1 (sumario) e 4 (metricas). Opcional.
        manifest: Dicionario do manifesto (como retornado por
            create_manifest). Se fornecido, enriquece secoes 3
            (treinamento) e 6 (reprodutibilidade). Opcional.
        figure_paths: Lista de caminhos para figuras geradas.
            Cada caminho e incluido como link Markdown na secao 5.
            Opcional (secao omitida se vazio ou None).
        output_path: Caminho para salvar o relatorio Markdown.
            Se fornecido, cria diretorios intermediarios e salva.
            Opcional (None = apenas retorna string).

    Returns:
        String Markdown completa do relatorio.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig.baseline()
        >>> md = generate_report(config)
        >>> "# Relatorio" in md
        True

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-export
            - tests/test_evaluation.py: test_generate_report
        Ref: docs/ARCHITECTURE_v2.md secao 8.6.
    """
    # ── Import da versao do pacote (lazy, evita circular) ──
    from geosteering_ai import __version__

    # ── Timestamp de geracao ──
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Construir secoes do relatorio ──
    sections: List[str] = []

    # Titulo principal
    sections.append(
        f"# Relatorio de Experimento — Geosteering AI v{__version__}\n\n"
        f"Gerado em: {timestamp}\n"
    )

    # Secao 1: Sumario Executivo
    sections.append(_build_section_summary(config, metrics_report))

    # Secao 2: Configuracao
    sections.append(_build_section_config(config))

    # Secao 3: Resultados de Treinamento
    sections.append(_build_section_training(manifest))

    # Secao 4: Metricas de Avaliacao
    sections.append(_build_section_metrics(metrics_report))

    # Secao 5: Figuras
    sections.append(_build_section_figures(figure_paths))

    # Secao 6: Reprodutibilidade
    sections.append(_build_section_reproducibility(config, manifest, __version__))

    # ── Montar documento final ──
    report = "\n---\n\n".join(sections)

    logger.info(
        "Relatorio gerado: %d secoes, %d caracteres.",
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
        logger.info("Relatorio salvo em: %s", output_path)

    return report


# ════════════════════════════════════════════════════════════════════════
# FUNCOES AUXILIARES: _build_section_*
#
# Cada funcao gera o Markdown de uma secao do relatorio.
# Todas sao None-safe: retornam placeholder se dados ausentes.
#
# Ref: docs/ARCHITECTURE_v2.md secao 8.6
# ════════════════════════════════════════════════════════════════════════

def _build_section_summary(
    config: PipelineConfig,
    metrics_report: Any = None,
) -> str:
    """Secao 1: Sumario Executivo.

    Exibe modelo, e opcionalmente R2 e RMSE globais.

    Args:
        config: PipelineConfig com model_type.
        metrics_report: MetricsReport opcional para exibir R2/RMSE.

    Returns:
        String Markdown da secao.

    Note:
        Referenciado em:
            - evaluation/report.py: generate_report (secao 1)
        Ref: docs/ARCHITECTURE_v2.md secao 8.6.
    """
    lines = ["## 1. Sumario Executivo\n"]
    lines.append(f"- **Arquitetura:** {config.model_type}")
    lines.append(f"- **Loss:** {config.loss_type}")

    if metrics_report is not None and hasattr(metrics_report, "to_dict"):
        metrics = metrics_report.to_dict()
        # R2 global — quanto mais proximo de 1.0, melhor
        r2 = metrics.get("r2")
        if r2 is not None:
            lines.append(f"- **R2 Global:** {r2:.6f}")
        # RMSE global — em dominio log10 (decadas)
        rmse = metrics.get("rmse")
        if rmse is not None:
            lines.append(f"- **RMSE Global:** {rmse:.6f}")
        # MAE global
        mae = metrics.get("mae")
        if mae is not None:
            lines.append(f"- **MAE Global:** {mae:.6f}")
    else:
        lines.append("- *Metricas nao disponíveis.*")

    return "\n".join(lines)


def _build_section_config(config: PipelineConfig) -> str:
    """Secao 2: Configuracao.

    Exibe campos-chave do PipelineConfig em tabela Markdown.

    Args:
        config: PipelineConfig do experimento.

    Returns:
        String Markdown da secao com tabela de configuracao.

    Note:
        Referenciado em:
            - evaluation/report.py: generate_report (secao 2)
        Ref: docs/ARCHITECTURE_v2.md secao 8.6.
    """
    config_dict = config.to_dict()

    lines = ["## 2. Configuracao\n"]
    lines.append("| Campo | Valor | Nota |")
    lines.append("|:------|:------|:-----|")

    for field_name, label, unit in _KEY_CONFIG_FIELDS:
        value = config_dict.get(field_name, "N/A")
        # Formatar valor para exibicao
        if isinstance(value, float):
            display_value = f"{value:g}"
        elif isinstance(value, bool):
            display_value = "Sim" if value else "Nao"
        else:
            display_value = str(value)
        lines.append(f"| {label} | {display_value} | {unit} |")

    return "\n".join(lines)


def _build_section_training(
    manifest: Optional[Dict[str, Any]] = None,
) -> str:
    """Secao 3: Resultados de Treinamento.

    Exibe epochs, tempo, best epoch e best val_loss se disponíveis
    no manifesto.

    Args:
        manifest: Dicionario do manifesto com chave "training" opcional.

    Returns:
        String Markdown da secao.

    Note:
        Referenciado em:
            - evaluation/report.py: generate_report (secao 3)
        Ref: docs/ARCHITECTURE_v2.md secao 8.6.
    """
    lines = ["## 3. Resultados de Treinamento\n"]

    training = None
    if manifest is not None:
        training = manifest.get("training")

    if training is None:
        lines.append("*Dados de treinamento nao disponíveis.*")
        return "\n".join(lines)

    # ── Exibir campos do treinamento em lista ──
    epochs = training.get("epochs")
    if epochs is not None:
        lines.append(f"- **Epochs treinados:** {epochs}")

    time_s = training.get("time_s")
    if time_s is not None:
        # Converter para formato legivel (minutos se > 60s)
        if time_s >= 3600:
            lines.append(f"- **Tempo de treinamento:** {time_s / 3600:.1f} h")
        elif time_s >= 60:
            lines.append(f"- **Tempo de treinamento:** {time_s / 60:.1f} min")
        else:
            lines.append(f"- **Tempo de treinamento:** {time_s:.1f} s")

    best_epoch = training.get("best_epoch")
    if best_epoch is not None:
        lines.append(f"- **Melhor epoch:** {best_epoch}")

    best_val_loss = training.get("best_val_loss")
    if best_val_loss is not None:
        lines.append(f"- **Melhor val_loss:** {best_val_loss:.6f}")

    # ── Diagnostico de convergencia ──
    if epochs is not None and best_epoch is not None:
        # Quantos epochs apos o melhor (indica overfitting potencial)
        epochs_after_best = epochs - best_epoch
        if epochs_after_best > 0:
            lines.append(
                f"- **Epochs apos melhor:** {epochs_after_best} "
                f"(early stopping ou patience)"
            )

    return "\n".join(lines)


def _build_section_metrics(metrics_report: Any = None) -> str:
    """Secao 4: Metricas de Avaliacao.

    Exibe tabela completa de metricas do MetricsReport.

    Args:
        metrics_report: MetricsReport com metodo to_dict(). Opcional.

    Returns:
        String Markdown da secao com tabela de metricas.

    Note:
        Referenciado em:
            - evaluation/report.py: generate_report (secao 4)
        Ref: docs/ARCHITECTURE_v2.md secao 8.6.
    """
    lines = ["## 4. Metricas de Avaliacao\n"]

    if metrics_report is None or not hasattr(metrics_report, "to_dict"):
        lines.append("*Metricas nao disponíveis.*")
        return "\n".join(lines)

    metrics = metrics_report.to_dict()

    lines.append("| Metrica | Valor |")
    lines.append("|:--------|------:|")

    # ── Ordem semantica: global primeiro, depois por componente ──
    _METRIC_LABELS = {
        "r2": "R2 Global",
        "r2_rh": "R2 rho_h",
        "r2_rv": "R2 rho_v",
        "rmse": "RMSE Global",
        "rmse_rh": "RMSE rho_h",
        "rmse_rv": "RMSE rho_v",
        "mae": "MAE Global",
        "mbe": "MBE Global",
        "mape": "MAPE Global (%)",
    }

    for key, label in _METRIC_LABELS.items():
        value = metrics.get(key)
        if value is not None:
            lines.append(f"| {label} | {value:.6f} |")

    return "\n".join(lines)


def _build_section_figures(
    figure_paths: Optional[List[str]] = None,
) -> str:
    """Secao 5: Figuras.

    Lista figuras como links Markdown para imagens.

    Args:
        figure_paths: Lista de caminhos de figuras. Opcional.

    Returns:
        String Markdown da secao.

    Note:
        Referenciado em:
            - evaluation/report.py: generate_report (secao 5)
        Ref: docs/ARCHITECTURE_v2.md secao 8.6.
    """
    lines = ["## 5. Figuras\n"]

    if not figure_paths:
        lines.append("*Nenhuma figura gerada.*")
        return "\n".join(lines)

    for i, path in enumerate(figure_paths, start=1):
        # Extrair nome do arquivo para caption
        filename = os.path.basename(path)
        lines.append(f"### Figura {i}: {filename}\n")
        lines.append(f"![{filename}]({path})\n")

    return "\n".join(lines)


def _build_section_reproducibility(
    config: PipelineConfig,
    manifest: Optional[Dict[str, Any]] = None,
    version: str = "2.0.0",
) -> str:
    """Secao 6: Reprodutibilidade.

    Informacoes para reproduzir o experimento: versao, config YAML,
    git commit (se disponível no manifesto).

    Args:
        config: PipelineConfig do experimento.
        manifest: Dicionario do manifesto com metadados extras. Opcional.
        version: Versao do Geosteering AI.

    Returns:
        String Markdown da secao.

    Note:
        Referenciado em:
            - evaluation/report.py: generate_report (secao 6)
        Ref: docs/ARCHITECTURE_v2.md secao 8.6.
    """
    lines = ["## 6. Reprodutibilidade\n"]

    lines.append(f"- **Geosteering AI:** v{version}")
    lines.append("- **Framework:** TensorFlow 2.x / Keras")

    # ── Config YAML path (se disponível no manifesto extra) ──
    extra = None
    if manifest is not None:
        extra = manifest.get("extra")

    if extra is not None and isinstance(extra, dict):
        yaml_path = extra.get("config_yaml_path")
        if yaml_path:
            lines.append(f"- **Config YAML:** `{yaml_path}`")

        git_commit = extra.get("git_commit")
        if git_commit:
            lines.append(f"- **Git Commit:** `{git_commit}`")

        git_branch = extra.get("git_branch")
        if git_branch:
            lines.append(f"- **Git Branch:** `{git_branch}`")

        dataset_path = extra.get("dataset_path")
        if dataset_path:
            lines.append(f"- **Dataset:** `{dataset_path}`")

    # ── Seed para reproducao ──
    config_dict = config.to_dict()
    seed = config_dict.get("random_seed")
    if seed is not None:
        lines.append(f"- **Random Seed:** {seed}")

    # ── Instrucoes de reproducao ──
    lines.append("\n### Como Reproduzir\n")
    lines.append("```bash")
    lines.append("pip install git+https://github.com/daniel-leal/geosteering-ai.git@<tag>")
    lines.append("python -m geosteering_ai.train --config <config.yaml>")
    lines.append("```")

    return "\n".join(lines)
