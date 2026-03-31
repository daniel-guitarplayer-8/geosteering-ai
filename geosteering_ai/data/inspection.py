# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/inspection.py                                               ║
# ║  Bloco: 2 — Preparacao de Dados                                          ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals)                          ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Inspecao tabular de dados pre-treinamento (NaN, Inf, estatisticas)   ║
# ║    • Estatisticas por split (train/val/test) e por feature                ║
# ║    • Export CSV para EDA_PLOTS_DIR                                        ║
# ║    • Diagnostico rapido: detecta problemas ANTES de treinar               ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), numpy                         ║
# ║  Exports: ~2 funcoes — ver __all__                                        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4 (data), legado C26A               ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Adaptado de C26A_Inspecao_Dados.py (legado)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Inspecao de dados pre-treinamento — NaN/Inf check, estatisticas por split.

Diagnostico rapido dos dados do pipeline ANTES do treinamento. Verifica
integridade numerica (NaN, Inf) e calcula estatisticas descritivas
(min, max, mean, std) por feature e por split (train/val/test).

   ┌──────────────────────────────────────────────────────────────────────────┐
   │  Fluxo de Inspecao:                                                      │
   │                                                                          │
   │  splits dict → inspect_data_splits(splits, config)                      │
   │       ↓                                                                  │
   │  Para cada split (train/val/test):                                       │
   │    1. Contar NaN e Inf                                                   │
   │    2. Calcular min, max, mean, std por feature                          │
   │    3. Registrar n_samples, n_features, seq_len                          │
   │       ↓                                                                  │
   │  Retorna dict consolidado → export_inspection_csv(summary, save_dir)    │
   └──────────────────────────────────────────────────────────────────────────┘

Example:
    >>> from geosteering_ai.data.inspection import inspect_data_splits
    >>> from geosteering_ai.config import PipelineConfig
    >>> config = PipelineConfig.baseline()
    >>> splits = {"train": x_train, "val": x_val, "test": x_test}
    >>> summary = inspect_data_splits(splits, config)
    >>> summary["any_nan"]  # False se dados limpos

Note:
    Adaptado de C26A_Inspecao_Dados.py (legado v5.0.15).
    Diferenca principal: usa config: PipelineConfig como parametro.
    Referenciado em:
        - data/pipeline.py: diagnostico pos-split
        - training/loop.py: validacao pre-treinamento
    Ref: docs/ARCHITECTURE_v2.md secao 4.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCAO PRINCIPAL — inspect_data_splits
# ════════════════════════════════════════════════════════════════════════════
# Itera sobre splits (train/val/test), calcula estatisticas por feature,
# verifica NaN/Inf. Retorna dict consolidado com metadata completo.
# Equivalente v2.0 do legado C26A._inspect_group(), mas com PipelineConfig.
# Ref: Legado C26A PARTE 2 + PARTE 3.
# ──────────────────────────────────────────────────────────────────────────


def inspect_data_splits(
    splits: Dict[str, np.ndarray],
    config: "PipelineConfig",
    *,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """Inspeciona splits de dados: NaN/Inf check e estatisticas por feature.

    Calcula estatisticas descritivas (min, max, mean, std) para cada
    feature em cada split do pipeline. Detecta problemas de integridade
    numerica (NaN, Inf) que causariam falha silenciosa no treinamento.

    Args:
        splits: Dicionario com chaves "train", "val", "test".
            Cada valor eh np.ndarray 2D (n, n_feat) ou 3D (n, seq, n_feat).
        config: PipelineConfig para metadados (feature_view, etc.).
        feature_names: Nomes das features (None = genericos).

    Returns:
        dict com estrutura:
            {
                "any_nan": bool,
                "any_inf": bool,
                "n_splits": int,
                "splits": {split_name: {n_samples, n_features, ...}, ...}
            }

    Example:
        >>> from geosteering_ai.data.inspection import inspect_data_splits
        >>> splits = {"train": np.random.randn(10, 600, 5).astype(np.float32)}
        >>> s = inspect_data_splits(splits, PipelineConfig.baseline())
        >>> s["any_nan"]
        False

    Note:
        Referenciado em:
            - tests/test_legacy_integration.py: TestInspectDataSplits
        Ref: Legado C26A._inspect_group(), v2.0 adaptado.
        NaN/Inf em dados de treinamento causam loss=NaN silencioso.
    """
    global_nan = False
    global_inf = False
    splits_result = {}

    for split_name, arr in splits.items():
        # ── Achatar se 3D para estatisticas por feature ──────────────
        if arr.ndim == 3:
            n_samples, seq_len, n_features = arr.shape
            flat = arr.reshape(-1, n_features)
        elif arr.ndim == 2:
            n_samples, n_features = arr.shape
            seq_len = None
            flat = arr
        else:
            logger.warning(
                "Split '%s' tem ndim=%d (esperado 2 ou 3). Ignorando.",
                split_name,
                arr.ndim,
            )
            continue

        # ── NaN/Inf check ────────────────────────────────────────────
        nan_count = int(np.isnan(flat).sum())
        inf_count = int(np.isinf(flat).sum())
        has_nan = nan_count > 0
        has_inf = inf_count > 0
        if has_nan:
            global_nan = True
            logger.warning(
                "ATENCAO: %d NaN detectados no split '%s'",
                nan_count,
                split_name,
            )
        if has_inf:
            global_inf = True
            logger.warning(
                "ATENCAO: %d Inf detectados no split '%s'",
                inf_count,
                split_name,
            )

        # ── Nomes de features ────────────────────────────────────────
        if feature_names and len(feature_names) == n_features:
            names = list(feature_names)
        else:
            names = [f"feat_{i}" for i in range(n_features)]

        # ── Estatisticas por feature ─────────────────────────────────
        per_feature = []
        for i in range(n_features):
            col = flat[:, i]
            per_feature.append(
                {
                    "name": names[i],
                    "min": float(np.nanmin(col)),
                    "max": float(np.nanmax(col)),
                    "mean": float(np.nanmean(col)),
                    "std": float(np.nanstd(col)),
                }
            )

        splits_result[split_name] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "seq_len": seq_len,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "per_feature": per_feature,
        }

        logger.info(
            "Inspecao '%s': %d amostras, %d features, NaN=%d, Inf=%d",
            split_name,
            n_samples,
            n_features,
            nan_count,
            inf_count,
        )

    return {
        "any_nan": global_nan,
        "any_inf": global_inf,
        "n_splits": len(splits_result),
        "splits": splits_result,
    }


# ════════════════════════════════════════════════════════════════════════════
# SECAO: EXPORT CSV — export_inspection_csv
# ════════════════════════════════════════════════════════════════════════════
# Salva resultado de inspect_data_splits() como CSV para referencia.
# Formato: split, feature, min, max, mean, std, has_nan, has_inf.
# Ref: Legado C26A PARTE 4 (CSV export).
# ──────────────────────────────────────────────────────────────────────────


def _safe_fmt(val: float, fmt: str = ".6f") -> str:
    """Formata valor numerico para CSV, tratando NaN/Inf como 'N/A'.

    Valores NaN ou Inf ocorrem quando uma coluna inteira contem apenas
    NaN (np.nanmin retorna NaN) ou quando dados corrompidos geram Inf.
    Parsers CSV downstream podem falhar com strings "nan"/"inf", entao
    exportamos como "N/A" para compatibilidade.

    Args:
        val: Valor numerico (float) a formatar.
        fmt: Formato f-string (default: ".6f" — 6 casas decimais).

    Returns:
        str: Valor formatado ou "N/A" se NaN/Inf.

    Note:
        Ref: CR-7 LOW — export_inspection_csv nao tratava NaN/Inf.
    """
    if np.isnan(val) or np.isinf(val):
        return "N/A"
    return f"{val:{fmt}}"


def export_inspection_csv(
    summary: Dict,
    save_dir: str,
    filename: str = "data_inspection_summary.csv",
) -> Path:
    """Exporta resumo de inspecao como CSV.

    Args:
        summary: Dict retornado por inspect_data_splits().
        save_dir: Diretorio para salvar o CSV.
        filename: Nome do arquivo (default: data_inspection_summary.csv).

    Returns:
        Path: Caminho absoluto do CSV criado.

    Note:
        Valores NaN/Inf nas estatisticas sao exportados como "N/A"
        para compatibilidade com parsers CSV downstream.
        Referenciado em: tests/test_legacy_integration.py.
        Ref: Legado C26A PARTE 4.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = save_path / filename

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "feature",
                "min",
                "max",
                "mean",
                "std",
                "has_nan",
                "has_inf",
            ]
        )
        for split_name, stats in summary["splits"].items():
            for feat in stats["per_feature"]:
                writer.writerow(
                    [
                        split_name,
                        feat["name"],
                        _safe_fmt(feat["min"]),
                        _safe_fmt(feat["max"]),
                        _safe_fmt(feat["mean"]),
                        _safe_fmt(feat["std"]),
                        stats["has_nan"],
                        stats["has_inf"],
                    ]
                )

    logger.info("CSV de inspecao salvo em: %s", csv_path)
    return csv_path


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
__all__ = [
    "inspect_data_splits",
    "export_inspection_csv",
]
