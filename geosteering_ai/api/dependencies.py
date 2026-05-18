# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MÓDULO: api/dependencies.py                                               ║
# ║  Bloco: 11 — API REST (NOVO em v2.39)                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning      ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: FastAPI Depends + thread-safe singleton                        ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Docker CPU             ║
# ║  Pacote: geosteering_ai.api                                                ║
# ║  Config: variáveis de ambiente (GEOSTEERING_MODEL_PATH, *_CORS_ORIGINS)    ║
# ║                                                                            ║
# ║  Propósito:                                                                ║
# ║    • Singleton lazy do InferencePipeline (carrega no 1º /predict)          ║
# ║    • get_pipeline() para Depends() do FastAPI                              ║
# ║    • get_settings() encapsula leitura de env vars                          ║
# ║    • Permite override em testes via dependency_overrides do FastAPI        ║
# ║                                                                            ║
# ║  Dependências: tensorflow + joblib (lazy, só ao carregar modelo)           ║
# ║  Exports: get_pipeline, get_settings, Settings, reset_pipeline_cache       ║
# ║  Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1        ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v2.39 (2026-05-18) — Implementação inicial (Sprint I2.7)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Dependências FastAPI — Singleton lazy do InferencePipeline + Settings.

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  FLUXO DE CARREGAMENTO LAZY                                         │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  [App startup]                                                       │
    │     └─→ /health                ─→  responde 200 (sem TF, ms)         │
    │                                                                      │
    │  [Primeira POST /predict]                                            │
    │     └─→ get_pipeline()         ─→  Lock acquire                      │
    │                                ─→  InferencePipeline.load(PATH)      │
    │                                ─→  TF carrega (~3-5s no 1º load)    │
    │                                ─→  Lock release                      │
    │                                ─→  cached para próximas requests     │
    │                                                                      │
    │  [Próximas POST /predict]                                            │
    │     └─→ get_pipeline()         ─→  retorna instância cached (≪1ms)   │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

Note:
    Singleton implementado via módulo-level + threading.Lock para
    thread-safety em workers Uvicorn com múltiplas threads.
    Testes podem injetar pipelines mock via
    `app.dependency_overrides[get_pipeline] = lambda: mock_pipeline`.
    Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.inference import InferencePipeline

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports públicos
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    "get_pipeline",
    "get_settings",
    "Settings",
    "reset_pipeline_cache",
    "ModelNotLoadedError",
    "ModelLoadFailedError",
]


# ────────────────────────────────────────────────────────────────────────
# D10: Exceções tipadas — facilitam tratamento no app.py handler
# ────────────────────────────────────────────────────────────────────────


class ModelNotLoadedError(RuntimeError):
    """GEOSTEERING_MODEL_PATH não está setada — modelo nunca foi configurado.

    Caso esperado em containers recém-iniciados sem volume montado.
    Resolve-se setando a env var ou montando volume com pipeline serializado.
    """


class ModelLoadFailedError(RuntimeError):
    """Tentativa de carregar pipeline falhou (path inválido, arquivo corrompido).

    Indica problema operacional: caminho aponta para diretório errado,
    pipeline incompleto (faltando model.keras/scalers.joblib/config.yaml),
    ou config.yaml não-parseable.
    """


# ────────────────────────────────────────────────────────────────────────
# D10: Variáveis de ambiente — nomes canônicos
# ────────────────────────────────────────────────────────────────────────
# Prefixo GEOSTEERING_ evita colisão com env vars de outras aplicações.
# Default seguro: CORS=* (MVP), payload=16 MiB.
ENV_MODEL_PATH = "GEOSTEERING_MODEL_PATH"
ENV_CORS_ORIGINS = "GEOSTEERING_API_CORS_ORIGINS"
ENV_MAX_BODY_BYTES = "GEOSTEERING_API_MAX_BODY_BYTES"
ENV_DOCS_ENABLED = "GEOSTEERING_API_DOCS_ENABLED"


# ════════════════════════════════════════════════════════════════════════
# SETTINGS — Encapsula variáveis de ambiente da API
#
# Usar dataclass (não Pydantic) para evitar custo de import em /health.
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Settings:
    """Configurações da API REST lidas de variáveis de ambiente.

    Attributes:
        model_path: Caminho do diretório do `InferencePipeline` serializado
            (deve conter model.keras + scalers.joblib + config.yaml).
            None se a env var não estiver setada — `/predict` retornará 503.
        cors_origins: Lista de origens permitidas para CORS. ``["*"]`` (default)
            é apropriado para MVP; em produção usar lista explícita.
        max_body_bytes: Limite do corpo da requisição em bytes (default 16 MiB).
            Acima disso, FastAPI retorna 413 Request Entity Too Large.
        docs_enabled: Se True (default), expõe `/docs` e `/openapi.json`.
            Setar para False em produção para esconder estrutura interna.

    Note:
        Instância obtida via `get_settings()` (cached com lru_cache).
        Para forçar releitura em testes: `get_settings.cache_clear()`.
    """

    model_path: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_body_bytes: int = 16 * 1024 * 1024  # 16 MiB
    docs_enabled: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Lê variáveis de ambiente e retorna `Settings` (cached singleton).

    Returns:
        Settings imutável construído a partir do ambiente atual.

    Note:
        Cache via `functools.lru_cache(maxsize=1)`. Em testes, chamar
        `get_settings.cache_clear()` após `monkeypatch.setenv(...)` para
        forçar releitura.
        Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1.
    """
    # ── model_path: opcional, usado em get_pipeline() ─────────────
    model_path = os.environ.get(ENV_MODEL_PATH) or None

    # ── cors_origins: CSV separado por vírgula, default ["*"] ─────
    cors_raw = os.environ.get(ENV_CORS_ORIGINS, "*")
    cors_origins = [origin.strip() for origin in cors_raw.split(",") if origin.strip()]
    if not cors_origins:
        cors_origins = ["*"]

    # ── max_body_bytes: numérico, default 16 MiB ──────────────────
    max_body_raw = os.environ.get(ENV_MAX_BODY_BYTES)
    try:
        max_body_bytes = int(max_body_raw) if max_body_raw else 16 * 1024 * 1024
    except ValueError:
        logger.warning(
            "Valor inválido para %s=%r; usando default 16 MiB.",
            ENV_MAX_BODY_BYTES,
            max_body_raw,
        )
        max_body_bytes = 16 * 1024 * 1024

    # ── docs_enabled: aceita "0"/"false"/"no" como desabilitar ────
    docs_raw = os.environ.get(ENV_DOCS_ENABLED, "1").strip().lower()
    docs_enabled = docs_raw not in {"0", "false", "no", "off"}

    settings = Settings(
        model_path=model_path,
        cors_origins=cors_origins,
        max_body_bytes=max_body_bytes,
        docs_enabled=docs_enabled,
    )
    logger.debug(
        "Settings carregadas — model_path=%s, cors_origins=%s, "
        "max_body_bytes=%d, docs_enabled=%s",
        settings.model_path,
        settings.cors_origins,
        settings.max_body_bytes,
        settings.docs_enabled,
    )
    return settings


# ════════════════════════════════════════════════════════════════════════
# PIPELINE SINGLETON — Lazy load do InferencePipeline
#
# Variável módulo-level + Lock garantem thread-safety em workers Uvicorn.
# Não usar lru_cache aqui: queremos controle explícito do erro (RuntimeError
# vira 503 no FastAPI; lru_cache caching de exceções é inconsistente).
# ════════════════════════════════════════════════════════════════════════

_pipeline_lock = threading.Lock()
_pipeline_instance: "Optional[InferencePipeline]" = None
_pipeline_loaded_path: Optional[str] = None


def get_pipeline() -> "InferencePipeline":
    """Retorna o InferencePipeline singleton — carrega no primeiro acesso.

    Returns:
        InferencePipeline carregado a partir de `GEOSTEERING_MODEL_PATH`.

    Raises:
        RuntimeError: Se `GEOSTEERING_MODEL_PATH` não estiver setada ou
            apontar para diretório inválido. O FastAPI converte para 503.

    Note:
        Thread-safe via threading.Lock — múltiplas requisições concorrentes
        ao primeiro carregamento só disparam um único load.
        Em testes, sobrescrever via:
            ``app.dependency_overrides[get_pipeline] = lambda: mock``
        Ou limpar cache via `reset_pipeline_cache()`.
        Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1.
    """
    global _pipeline_instance, _pipeline_loaded_path

    # Fast path: já carregado, retorna sem lock
    if _pipeline_instance is not None:
        return _pipeline_instance

    settings = get_settings()
    if not settings.model_path:
        # Mensagem genérica ao cliente (não vaza nome de env nem detalhe interno).
        # O nome da env var fica em log INFO (operadores precisam saber).
        logger.info("Pipeline não carregado: %s não configurada.", ENV_MODEL_PATH)
        raise ModelNotLoadedError(
            "Modelo de inferência não está disponível. "
            "Configure o serviço com um pipeline válido."
        )

    with _pipeline_lock:
        # Double-checked locking: outra thread pode ter carregado durante o wait
        if _pipeline_instance is not None:
            return _pipeline_instance

        logger.info(
            "Carregando InferencePipeline de %s (primeira requisição)...",
            settings.model_path,
        )

        # ── Lazy import — TensorFlow só carrega aqui ──────────────
        from geosteering_ai.inference import InferencePipeline

        try:
            _pipeline_instance = InferencePipeline.load(settings.model_path)
            _pipeline_loaded_path = settings.model_path
        except FileNotFoundError as exc:
            # Path interno em log; mensagem ao cliente genérica.
            logger.warning(
                "Pipeline não encontrado em %s: %s", settings.model_path, exc
            )
            raise ModelLoadFailedError(
                "Pipeline configurado não encontrado no servidor."
            ) from exc
        except Exception as exc:
            logger.exception(
                "Falha ao carregar InferencePipeline de %s", settings.model_path
            )
            raise ModelLoadFailedError(
                "Falha ao carregar o pipeline. Consulte os logs do servidor."
            ) from exc

        logger.info(
            "InferencePipeline carregado — model_type=%s, feature_view=%s",
            _pipeline_instance.config.model_type,
            _pipeline_instance.config.feature_view,
        )
        return _pipeline_instance


def is_pipeline_loaded() -> bool:
    """Retorna True se o pipeline singleton já está em memória.

    Usado por `/health` para reportar `model_loaded` sem carregar nada.

    Note:
        Não adquire o lock — leitura atômica de variável módulo-level
        é segura em Python (GIL).
    """
    return _pipeline_instance is not None


def get_loaded_model_path() -> Optional[str]:
    """Retorna o caminho do pipeline carregado, ou None se ainda não."""
    return _pipeline_loaded_path


def reset_pipeline_cache() -> None:
    """Limpa o singleton — usado por testes entre runs.

    Note:
        NÃO chamar em produção. Pode causar erro em requisições em
        andamento se outra thread estiver no meio de uma predição
        (o `model` referenciado fica órfão mas continua válido).
    """
    global _pipeline_instance, _pipeline_loaded_path
    with _pipeline_lock:
        _pipeline_instance = None
        _pipeline_loaded_path = None
    get_settings.cache_clear()
    logger.debug("Cache do pipeline e das settings limpos.")
