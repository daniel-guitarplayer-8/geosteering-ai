# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: evaluation/manifest.py                                           ║
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
# ║    • create_manifest: gera manifesto completo do experimento (dict)       ║
# ║    • save_manifest: persiste manifesto em JSON formatado                  ║
# ║    • load_manifest: carrega manifesto de arquivo JSON                     ║
# ║    • Registro de reprodutibilidade: config, metricas, training, metadata  ║
# ║    • NumPy-only: sem dependencia de TensorFlow                           ║
# ║                                                                            ║
# ║  Dependencias: json, datetime, os, dataclasses (stdlib apenas)            ║
# ║  Exports: ~3 (create_manifest, save_manifest, load_manifest)             ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 8.5                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (C64)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Manifesto de experimento para reprodutibilidade e rastreabilidade.

Gera, salva e carrega manifestos JSON contendo todas as informacoes
necessarias para reproduzir um experimento de inversao geofisica:
configuracao completa (121+ campos), metricas de avaliacao, resultados
de treinamento, e metadados adicionais do usuario.

Estrutura do manifesto:

    ┌────────────────────────────────────────────────────────────────────┐
    │  manifesto (dict)                                                  │
    ├────────────────────────────────────────────────────────────────────┤
    │  timestamp         ISO 8601 (UTC)                                  │
    │  geosteering_ai_version  "2.0.0"                                  │
    │  config            PipelineConfig.to_dict() (121+ campos)         │
    │  metrics           MetricsReport.to_dict() (9 metricas)           │
    │  training          {epochs, time_s, best_epoch, best_val_loss}    │
    │  extra             metadados adicionais do usuario                │
    └────────────────────────────────────────────────────────────────────┘

Formato de saida: JSON com indentacao de 2 espacos, chaves ordenadas
por secao (nao alfabeticamente), compativel com jq e Python json.

ERRATA: Versao sempre "2.0.0" (lida de geosteering_ai.__version__).

Note:
    Referenciado em:
        - evaluation/__init__.py: re-exports create_manifest, save_manifest,
          load_manifest
        - evaluation/report.py: generate_report consome manifesto
        - tests/test_evaluation.py: testes de serializacao JSON
    Ref: docs/ARCHITECTURE_v2.md secao 8.5.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Criacao de manifesto ---
    "create_manifest",
    # --- Persistencia JSON ---
    "save_manifest",
    "load_manifest",
]


# ════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════════════
# Versao do formato de manifesto — permite detectar incompatibilidades
# entre versoes do Geosteering AI na leitura de manifestos antigos.
# ────────────────────────────────────────────────────────────────────────

_MANIFEST_FORMAT_VERSION = "1.0"


# ════════════════════════════════════════════════════════════════════════
# FUNCAO PRINCIPAL: create_manifest
#
# Gera o dict completo do manifesto a partir de PipelineConfig e
# resultados opcionais (metricas, treinamento, metadados extras).
# Todas as secoes opcionais sao None-safe (omitidas se nao fornecidas).
#
# Ref: docs/ARCHITECTURE_v2.md secao 8.5
# ════════════════════════════════════════════════════════════════════════

def create_manifest(
    config: PipelineConfig,
    *,
    metrics_report: Any = None,
    training_result: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cria manifesto completo do experimento como dicionario.

    Consolida todas as informacoes do experimento em um unico dict
    serializavel para JSON. Inclui timestamp UTC, versao do pacote,
    configuracao completa, e opcionalmente metricas, resultado de
    treinamento, e metadados extras do usuario.

    O manifesto serve como registro de reprodutibilidade: dado o mesmo
    config YAML + tag git + seed, o resultado deve ser identico.

    Args:
        config: PipelineConfig do experimento. Obrigatorio. Todos os
            121+ campos sao incluidos via config.to_dict().
        metrics_report: MetricsReport com metricas de avaliacao. Se
            fornecido, deve ter metodo to_dict() retornando
            Dict[str, float]. Opcional (None = secao omitida).
        training_result: Dicionario com resultados de treinamento.
            Chaves esperadas: "epochs" (int), "time_s" (float),
            "best_epoch" (int), "best_val_loss" (float). Chaves
            extras sao preservadas. Opcional (None = secao omitida).
        extra_metadata: Dicionario com metadados adicionais do usuario.
            Qualquer dado serializavel para JSON. Exemplos: notas,
            hardware, dataset_path, git_commit. Opcional.

    Returns:
        Dicionario com o manifesto completo. Chaves de topo:
        "timestamp", "geosteering_ai_version", "manifest_format_version",
        "config", e opcionalmente "metrics", "training", "extra".

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig.baseline()
        >>> manifest = create_manifest(config)
        >>> manifest["geosteering_ai_version"]
        '2.0.0'
        >>> "config" in manifest
        True

    Note:
        Referenciado em:
            - evaluation/report.py: generate_report consome manifesto
            - evaluation/__init__.py: re-export
            - tests/test_evaluation.py: test_create_manifest
        Ref: docs/ARCHITECTURE_v2.md secao 8.5.
    """
    # ── Import da versao do pacote (lazy, evita circular) ──
    from geosteering_ai import __version__

    # ── Timestamp UTC em formato ISO 8601 ──
    timestamp = datetime.now(timezone.utc).isoformat()

    manifest: Dict[str, Any] = {
        "timestamp": timestamp,
        "geosteering_ai_version": __version__,
        "manifest_format_version": _MANIFEST_FORMAT_VERSION,
    }

    # ── Secao config: todos os campos do PipelineConfig ──
    # config.to_dict() usa dataclasses.asdict() para conversao recursiva.
    manifest["config"] = config.to_dict()
    logger.info(
        "Manifesto: config com %d campos incluidos.",
        len(manifest["config"]),
    )

    # ── Secao metrics: metricas de avaliacao (opcional) ──
    # MetricsReport.to_dict() retorna Dict[str, float] com 9 metricas.
    if metrics_report is not None:
        if hasattr(metrics_report, "to_dict"):
            manifest["metrics"] = metrics_report.to_dict()
            logger.info(
                "Manifesto: %d metricas incluidas.",
                len(manifest["metrics"]),
            )
        else:
            logger.warning(
                "metrics_report fornecido nao possui metodo to_dict(). "
                "Secao 'metrics' omitida do manifesto."
            )

    # ── Secao training: resultado do treinamento (opcional) ──
    # Chaves esperadas: epochs, time_s, best_epoch, best_val_loss.
    if training_result is not None:
        if not isinstance(training_result, dict):
            logger.warning(
                "training_result deve ser dict, recebido %s. "
                "Secao 'training' omitida.",
                type(training_result).__name__,
            )
        else:
            manifest["training"] = dict(training_result)
            logger.info(
                "Manifesto: training com %d chaves incluido.",
                len(manifest["training"]),
            )

    # ── Secao extra: metadados adicionais do usuario (opcional) ──
    if extra_metadata is not None:
        if not isinstance(extra_metadata, dict):
            logger.warning(
                "extra_metadata deve ser dict, recebido %s. "
                "Secao 'extra' omitida.",
                type(extra_metadata).__name__,
            )
        else:
            manifest["extra"] = dict(extra_metadata)
            logger.info(
                "Manifesto: extra com %d chaves incluido.",
                len(manifest["extra"]),
            )

    logger.info(
        "Manifesto criado: %d secoes de topo, timestamp=%s.",
        len(manifest),
        timestamp,
    )

    return manifest


# ════════════════════════════════════════════════════════════════════════
# PERSISTENCIA: save_manifest / load_manifest
#
# Salva e carrega manifestos em formato JSON com indentacao de 2
# espacos. Cria diretorios intermediarios automaticamente.
#
# Ref: docs/ARCHITECTURE_v2.md secao 8.5
# ════════════════════════════════════════════════════════════════════════

def save_manifest(manifest: Dict[str, Any], path: str) -> None:
    """Salva manifesto como arquivo JSON formatado.

    Persiste o dicionario do manifesto em um arquivo JSON com
    indentacao de 2 espacos e codificacao UTF-8. Cria diretorios
    intermediarios se nao existirem.

    Args:
        manifest: Dicionario do manifesto (como retornado por
            create_manifest). Deve ser serializavel para JSON.
        path: Caminho do arquivo de saida (ex: "outputs/manifest.json").
            Extensao .json recomendada mas nao obrigatoria.

    Raises:
        TypeError: Se manifest contiver objetos nao serializaveis
            para JSON (ex: numpy arrays — converter para list antes).
        OSError: Se nao for possivel criar diretorios ou escrever
            o arquivo.

    Example:
        >>> manifest = {"timestamp": "2026-03-26T12:00:00+00:00"}
        >>> save_manifest(manifest, "/tmp/test_manifest.json")

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-export
            - tests/test_evaluation.py: test_save_load_manifest
        Ref: docs/ARCHITECTURE_v2.md secao 8.5.
    """
    # ── Cria diretorios intermediarios se necessario ──
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Manifesto salvo em: %s", path)


def load_manifest(path: str) -> Dict[str, Any]:
    """Carrega manifesto de arquivo JSON.

    Le e desserializa um manifesto previamente salvo por
    save_manifest. Valida que o resultado e um dicionario.

    Args:
        path: Caminho do arquivo JSON do manifesto.

    Returns:
        Dicionario com o manifesto carregado. Estrutura identica
        a retornada por create_manifest.

    Raises:
        FileNotFoundError: Se o arquivo nao existir.
        json.JSONDecodeError: Se o arquivo nao for JSON valido.
        TypeError: Se o conteudo JSON nao for um dicionario de topo.

    Example:
        >>> manifest = load_manifest("/tmp/test_manifest.json")
        >>> manifest["geosteering_ai_version"]
        '2.0.0'

    Note:
        Referenciado em:
            - evaluation/__init__.py: re-export
            - evaluation/report.py: pode consumir manifesto carregado
            - tests/test_evaluation.py: test_save_load_manifest
        Ref: docs/ARCHITECTURE_v2.md secao 8.5.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"Manifesto deve ser um dicionario JSON, "
            f"recebido {type(data).__name__}."
        )

    logger.info(
        "Manifesto carregado de: %s (%d secoes de topo).",
        path,
        len(data),
    )

    return data
