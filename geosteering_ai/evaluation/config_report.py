# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/config_report.py                                      ║
# ║  Bloco: 8 — Evaluation                                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals)                          ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Relatorio pre-treinamento de configuracao (FLAGS audit)              ║
# ║    • Introspecao automatica de todos os campos do PipelineConfig          ║
# ║    • Secoes: fisicas, modelo, dados, noise, treinamento                  ║
# ║    • Validacao de errata (20000.0, 1.0, 600, log10, etc.)               ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), dataclasses                   ║
# ║  Exports: ~1 funcao — ver __all__                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8, legado C42A                      ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Adaptado de C42A_Relatorio_FLAGS_ARCH_PARAMS.py  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Relatorio pre-treinamento de configuracao (FLAGS audit).

Gera um relatorio textual completo de todos os campos do PipelineConfig,
organizado por secoes semanticas. Substitui o legado C42A (1458 linhas)
por introspecao automatica do dataclass (~200 linhas).

   ┌──────────────────────────────────────────────────────────────────────────┐
   │  Secoes do Relatorio:                                                    │
   │                                                                          │
   │  1. Errata Fisica — valores criticos validados                          │
   │  2. Modelo — arquitetura, causal, skip connections                      │
   │  3. Dados — features, targets, split, scaling                           │
   │  4. Noise — tipos, curriculum, N-Stage                                  │
   │  5. Treinamento — LR, epochs, callbacks, losses                        │
   │  6. Resumo — contagem total/bool/numericos                              │
   └──────────────────────────────────────────────────────────────────────────┘

Example:
    >>> from geosteering_ai.evaluation.config_report import generate_config_report
    >>> from geosteering_ai.config import PipelineConfig
    >>> config = PipelineConfig.robusto()
    >>> report = generate_config_report(config)
    >>> print(report)

Note:
    Adaptado de C42A_Relatorio_FLAGS_ARCH_PARAMS.py (legado v5.0.15).
    No v2.0, PipelineConfig centraliza todos os campos — introspecao via
    dataclasses.fields() substitui 19 PARTEs de resolucao de FLAGS.
    Ref: docs/ARCHITECTURE_v2.md secao 8.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# SECAO: CONSTANTES — AGRUPAMENTO SEMANTICO DE CAMPOS
# ════════════════════════════════════════════════════════════════════════════
# Campos do PipelineConfig agrupados por secao para o relatorio.
# Se um campo nao esta em nenhum grupo, vai para "Outros".
# ──────────────────────────────────────────────────────────────────────────

_ERRATA_FIELDS = {
    "frequency_hz",
    "spacing_meters",
    "sequence_length",
    "target_scaling",
    "input_features",
    "output_targets",
    "eps_tf",
}

_MODEL_FIELDS = {
    "model_type",
    "use_causal_mode",
    "use_skip_connections",
    "use_se_block",
    "se_reduction",
    "use_physical_constraint_layer",
}

_DATA_FIELDS = {
    "feature_view",
    "use_geosignal_features",
    "geosignal_families",
    "split_by_model",
    "train_ratio",  # split ratios sao 3 campos separados
    "val_ratio",
    "test_ratio",
    "scaler_type",
    "smoothing_type",
    # n_features eh @property (nao dataclass field) — vai para Outros
}

_NOISE_FIELDS = {
    "noise_types",
    "noise_weights",
    "noise_level_max",
    "use_noise",  # campo real (nao use_noise_augmentation)
    "use_curriculum",  # campo real (nao use_curriculum_noise)
    "epochs_no_noise",
    "noise_ramp_epochs",
    "use_nstage",  # campo real (nao use_nstage_training)
    "n_training_stages",
    "nstage_stage1_epochs",
    "stage_lr_decay",
}

_TRAINING_FIELDS = {
    "learning_rate",
    "epochs",
    "batch_size",
    "early_stopping_patience",
    "use_restore_best_weights",
    # use_lr_scheduler e lr_scheduler_type NAO existem em PipelineConfig
    "use_mixed_precision",
    "use_xla",
    "loss_type",
    "use_gradient_clipping",
    "gradient_clip_norm",
}

_HOLDOUT_FIELDS = {
    "use_holdout_plots",
    "holdout_plots_max_samples",
    "holdout_plots_dpi",
}


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCAO PRINCIPAL — generate_config_report
# ════════════════════════════════════════════════════════════════════════════
# Introspecta PipelineConfig via dataclasses.fields() e gera relatorio
# textual organizado por secoes semanticas. Cada campo eh formatado
# com nome, tipo, e valor atual.
#
# No legado C42A, isso exigia 19 PARTEs e 1458 linhas para resolver
# FLAGS via globals. No v2.0, PipelineConfig eh um dataclass imutavel
# que pode ser introspecionado automaticamente.
#
# Ref: Legado C42A PARTEs 4-17.
# ──────────────────────────────────────────────────────────────────────────


def generate_config_report(config: "PipelineConfig") -> str:
    """Gera relatorio textual completo da configuracao do pipeline.

    Introspecta todos os campos do PipelineConfig e organiza em secoes
    semanticas. Inclui contagem de campos (total, bool, numericos) e
    validacao visual dos valores criticos da errata.

    Args:
        config: PipelineConfig configurado para o experimento.

    Returns:
        str: Relatorio formatado (multi-linha, ~200-400 linhas).

    Example:
        >>> from geosteering_ai.evaluation.config_report import generate_config_report
        >>> report = generate_config_report(PipelineConfig.baseline())
        >>> "ResNet_18" in report
        True

    Note:
        Referenciado em:
            - tests/test_legacy_integration.py: TestGenerateConfigReport
            - training/loop.py: diagnostico pre-treinamento
        Ref: Legado C42A, v2.0 adaptado via dataclasses.fields().
        No legado, 1458 linhas. No v2.0, ~200 linhas (introspecao).
    """
    lines = []
    _sep = "=" * 72

    # ── Header ───────────────────────────────────────────────────────
    lines.append(_sep)
    lines.append("  RELATORIO DE CONFIGURACAO — PRE-TREINAMENTO")
    lines.append("  Geosteering AI v2.0 — PipelineConfig Audit")
    lines.append(_sep)
    lines.append("")

    # ── Extrair todos os campos ──────────────────────────────────────
    fields = dataclasses.fields(config)
    field_dict = {f.name: getattr(config, f.name) for f in fields}

    # ── Contadores ───────────────────────────────────────────────────
    total = len(fields)
    n_bool = sum(1 for v in field_dict.values() if isinstance(v, bool))
    n_true = sum(1 for v in field_dict.values() if v is True)
    n_false = sum(1 for v in field_dict.values() if v is False)
    n_numeric = sum(
        1
        for v in field_dict.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    )

    # ── Secao 1: Errata Fisica ───────────────────────────────────────
    lines.append("  SECAO 1: Errata Fisica (valores criticos)")
    lines.append("-" * 72)
    _render_group(lines, config, field_dict, _ERRATA_FIELDS)
    lines.append("")

    # ── Secao 2: Modelo ──────────────────────────────────────────────
    lines.append("  SECAO 2: Modelo e Arquitetura")
    lines.append("-" * 72)
    _render_group(lines, config, field_dict, _MODEL_FIELDS)
    lines.append("")

    # ── Secao 3: Dados ───────────────────────────────────────────────
    lines.append("  SECAO 3: Dados e Features")
    lines.append("-" * 72)
    _render_group(lines, config, field_dict, _DATA_FIELDS)
    lines.append("")

    # ── Secao 4: Noise ───────────────────────────────────────────────
    lines.append("  SECAO 4: Noise e Curriculum")
    lines.append("-" * 72)
    _render_group(lines, config, field_dict, _NOISE_FIELDS)
    lines.append("")

    # ── Secao 5: Treinamento ─────────────────────────────────────────
    lines.append("  SECAO 5: Treinamento")
    lines.append("-" * 72)
    _render_group(lines, config, field_dict, _TRAINING_FIELDS)
    lines.append("")

    # ── Secao 6: Holdout / Visualizacao ──────────────────────────────
    lines.append("  SECAO 6: Holdout / Visualizacao")
    lines.append("-" * 72)
    _render_group(lines, config, field_dict, _HOLDOUT_FIELDS)
    lines.append("")

    # ── Secao 7: Outros campos ───────────────────────────────────────
    known = (
        _ERRATA_FIELDS
        | _MODEL_FIELDS
        | _DATA_FIELDS
        | _NOISE_FIELDS
        | _TRAINING_FIELDS
        | _HOLDOUT_FIELDS
    )
    others = {k: v for k, v in field_dict.items() if k not in known}
    if others:
        lines.append("  SECAO 7: Outros campos")
        lines.append("-" * 72)
        for name, value in sorted(others.items()):
            lines.append(f"    {name:<40s} = {_fmt(value)}")
        lines.append("")

    # ── Resumo ───────────────────────────────────────────────────────
    lines.append(_sep)
    lines.append(f"  RESUMO: {total} campos total")
    lines.append(f"    Bool: {n_bool} ({n_true} True, {n_false} False)")
    lines.append(f"    Numericos: {n_numeric}")
    lines.append(f"    Outros: {total - n_bool - n_numeric}")
    lines.append(_sep)

    report = "\n".join(lines)
    logger.info("Config report gerado: %d campos, %d linhas", total, len(lines))
    return report


# ════════════════════════════════════════════════════════════════════════════
# SECAO: HELPERS DE FORMATACAO
# ════════════════════════════════════════════════════════════════════════════
# Formatadores internos para renderizar valores e grupos de campos.
# ──────────────────────────────────────────────────────────────────────────


def _fmt(value) -> str:
    """Formata valor para exibicao no relatorio."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value}"
    if isinstance(value, (list, tuple)):
        return str(value)
    return str(value)


def _render_group(lines, config, field_dict, field_names):
    """Renderiza um grupo de campos no relatorio."""
    for name in sorted(field_names):
        if name in field_dict:
            value = field_dict[name]
            lines.append(f"    {name:<40s} = {_fmt(value)}")


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
__all__ = [
    "generate_config_report",
]
