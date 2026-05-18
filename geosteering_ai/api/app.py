# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MÓDULO: api/app.py                                                        ║
# ║  Bloco: 11 — API REST (NOVO em v2.39)                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning      ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: FastAPI + CORS + exception handlers                            ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Docker CPU             ║
# ║  Pacote: geosteering_ai.api                                                ║
# ║                                                                            ║
# ║  Propósito:                                                                ║
# ║    • Cria instância FastAPI com lifespan, CORS, exception handlers         ║
# ║    • Monta roteadores /health e /predict                                   ║
# ║    • Padroniza erros (RuntimeError→503, exceções→500) via ErrorResponse    ║
# ║    • Middleware: logging de request_id, latency, status                    ║
# ║                                                                            ║
# ║  Dependências: fastapi, starlette (vem com FastAPI), uvicorn (runtime)     ║
# ║  Exports: app (FastAPI), create_app() factory                              ║
# ║  Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1        ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v2.39 (2026-05-18) — Implementação inicial (Sprint I2.7, commit 5/9)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""App FastAPI principal — montagem de rotas, CORS, lifespan, exception handlers.

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  PIPELINE DE REQUISIÇÃO                                             │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  HTTP Request                                                       │
    │     ↓                                                                │
    │  CORS middleware (Starlette)                                        │
    │     ↓                                                                │
    │  Logging middleware (request_id, latency)                           │
    │     ↓                                                                │
    │  Route handler (/health ou /predict)                                │
    │     ↓                                                                │
    │  Exception handlers (RuntimeError→503, ValueError→422, Exc→500)     │
    │     ↓                                                                │
    │  Response (JSON)                                                    │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

Note:
    Use `create_app()` para testes (permite injetar settings custom).
    `app` é a instância padrão usada pelo entry point `geosteering-api`.
    Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from geosteering_ai.api import __version__
from geosteering_ai.api.dependencies import (
    ModelLoadFailedError,
    ModelNotLoadedError,
    get_settings,
)
from geosteering_ai.api.routes.health import router as health_router
from geosteering_ai.api.routes.predict import router as predict_router
from geosteering_ai.api.schemas import ErrorResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

logger = logging.getLogger(__name__)

__all__ = ["app", "create_app"]


# ════════════════════════════════════════════════════════════════════════
# LIFESPAN — Hooks de startup/shutdown
#
# Não carrega TF aqui: respeitar a decisão D2 do plano (lazy load no
# primeiro /predict). Logamos apenas configuração efetiva.
# ════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> "AsyncGenerator[None, None]":
    """Lifespan async — startup/shutdown da aplicação.

    Note:
        Mantemos o startup leve: apenas log de configuração. TensorFlow
        e InferencePipeline são carregados sob demanda (lazy) no
        primeiro POST /predict. Decisão D2 do plano v2.39.
    """
    settings = get_settings()
    logger.info(
        "API REST iniciando — version=%s, model_path=%s, cors_origins=%s, "
        "docs_enabled=%s",
        __version__,
        settings.model_path,
        settings.cors_origins,
        settings.docs_enabled,
    )
    yield
    logger.info("API REST encerrando.")


# ════════════════════════════════════════════════════════════════════════
# FACTORY — Permite criar instâncias com config custom (útil em testes)
# ════════════════════════════════════════════════════════════════════════


def create_app() -> FastAPI:
    """Cria e configura uma nova instância FastAPI.

    Returns:
        FastAPI configurado com CORS, lifespan, rotas e exception handlers.

    Note:
        Usado pelo módulo (variable `app` abaixo) e por testes que
        precisam de uma instância isolada com `dependency_overrides`.
        Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1.
    """
    settings = get_settings()

    fastapi_app = FastAPI(
        title="Geosteering AI — API REST",
        description=(
            "API REST para inferência 1D de resistividade via Deep Learning. "
            "Expõe `InferencePipeline` (TensorFlow/Keras) via HTTP."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )

    # ── CORS — configurado via env GEOSTEERING_API_CORS_ORIGINS ──
    # Fix A2/ALTO-1 audit v2.39: spec CORS proíbe allow_credentials=True
    # combinado com allow_origins=["*"]. Forçar False quando wildcard
    # para evitar header silenciosamente removido pelo Starlette.
    cors_is_wildcard = settings.cors_origins == ["*"]
    if cors_is_wildcard:
        logger.warning(
            "CORS configurado com '*' (wildcard) — allow_credentials forçado "
            "a False. Em produção, configure %s com lista explícita.",
            "GEOSTEERING_API_CORS_ORIGINS",
        )
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=not cors_is_wildcard,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Middleware de body size (MED-1 audit v2.39 — DoS protection) ──
    # Checa Content-Length ANTES de Pydantic alocar o JSON em memória.
    # Cap default 16 MiB, configurável via GEOSTEERING_API_MAX_BODY_BYTES.
    max_body_bytes = settings.max_body_bytes

    @fastapi_app.middleware("http")
    async def enforce_body_size(
        request: Request,
        call_next: "Callable[[Request], Awaitable[Response]]",
    ) -> Response:
        """Rejeita payloads acima do limite com 413 antes de qualquer alocação."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
            except ValueError:
                size = 0
            if size > max_body_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content=ErrorResponse(
                        detail=(
                            f"Payload de {size} bytes excede o limite "
                            f"({max_body_bytes} bytes)."
                        ),
                        type="payload_too_large",
                    ).model_dump(),
                )
        return await call_next(request)

    # ── Middleware de logging — request_id + latency ─────────────
    # M3 audit v2.39: persistir request_id em request.state para que os
    # exception handlers também consigam refletir no header de erro.
    @fastapi_app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: "Callable[[Request], Awaitable[Response]]",
    ) -> Response:
        """Adiciona request_id, mede latência e loga resultado."""
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        t_start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.exception(
                "Erro não-capturado — request_id=%s, path=%s, elapsed=%.2fms",
                request_id,
                request.url.path,
                elapsed_ms,
            )
            raise
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%d elapsed=%.2fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ── Exception handlers — padronizam corpo via ErrorResponse ──
    # A1/MED-3 audit v2.39: handlers tipados separam model_not_loaded vs
    # model_load_failed (cliente pode tomar ações diferentes).
    def _build_error_response(
        request: Request, status_code: int, detail: str, type_: str
    ) -> JSONResponse:
        """Constrói JSONResponse padronizado preservando request_id."""
        resp = JSONResponse(
            status_code=status_code,
            content=ErrorResponse(detail=detail, type=type_).model_dump(),
        )
        rid = getattr(request.state, "request_id", None)
        if rid:
            resp.headers["x-request-id"] = rid
        return resp

    @fastapi_app.exception_handler(ModelNotLoadedError)
    async def _model_not_loaded_handler(
        request: Request, exc: ModelNotLoadedError
    ) -> JSONResponse:
        """ModelNotLoadedError (env var não setada) → 503."""
        logger.info("ModelNotLoaded em %s: %s", request.url.path, exc)
        return _build_error_response(
            request,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            str(exc),
            "model_not_loaded",
        )

    @fastapi_app.exception_handler(ModelLoadFailedError)
    async def _model_load_failed_handler(
        request: Request, exc: ModelLoadFailedError
    ) -> JSONResponse:
        """ModelLoadFailedError (path inválido / corrompido) → 503."""
        logger.warning("ModelLoadFailed em %s: %s", request.url.path, exc)
        return _build_error_response(
            request,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            str(exc),
            "model_load_failed",
        )

    @fastapi_app.exception_handler(RuntimeError)
    async def _runtime_error_handler(
        request: Request, exc: RuntimeError
    ) -> JSONResponse:
        """RuntimeError genérico (não-tipado) → 500.

        Handler de fallback para RuntimeErrors que escapam dos handlers
        mais específicos (ModelNotLoadedError, ModelLoadFailedError).
        """
        logger.exception("RuntimeError não-tipado em %s: %s", request.url.path, exc)
        return _build_error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Erro interno do servidor.",
            "internal",
        )

    @fastapi_app.exception_handler(Exception)
    async def _internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Exceção não-categorizada → 500 com log de stacktrace.

        Note:
            HTTPException, RequestValidationError e StarletteHTTPException
            têm handlers próprios do FastAPI/Starlette com precedência
            maior. Este handler só captura exceções que NÃO subclassam
            HTTPException (ex: OSError, KeyError). Validado por
            tests/test_api_predict.py::TestPredictErrors (422 do Pydantic
            continua retornando 422, não 500).
        """
        logger.exception("Erro interno em %s: %s", request.url.path, exc)
        return _build_error_response(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Erro interno do servidor.",
            "internal",
        )

    # ── Montagem das rotas ────────────────────────────────────────
    fastapi_app.include_router(health_router)
    fastapi_app.include_router(predict_router)

    return fastapi_app


# ────────────────────────────────────────────────────────────────────────
# INSTÂNCIA PADRÃO — Usada pelo entry point geosteering-api e por uvicorn
# ────────────────────────────────────────────────────────────────────────
app = create_app()
