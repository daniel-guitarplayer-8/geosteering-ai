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
from fastapi.responses import JSONResponse

from geosteering_ai.api import __version__
from geosteering_ai.api.dependencies import get_settings
from geosteering_ai.api.routes.health import router as health_router
from geosteering_ai.api.routes.predict import router as predict_router
from geosteering_ai.api.schemas import ErrorResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

logger = logging.getLogger(__name__)

__all__ = ["app", "create_app"]


# ════════════════════════════════════════════════════════════════════════
# LIFESPAN — Hooks de startup/shutdown
#
# Não carrega TF aqui: respeitar a decisão D2 do plano (lazy load no
# primeiro /predict). Logamos apenas configuração efetiva.
# ════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> "AsyncIterator[None]":
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
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Middleware de logging — request_id + latency ─────────────
    @fastapi_app.middleware("http")
    async def log_requests(
        request: Request,
        call_next: "Callable[[Request], Awaitable]",
    ):
        """Adiciona request_id, mede latência e loga resultado."""
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
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
    @fastapi_app.exception_handler(RuntimeError)
    async def _runtime_error_handler(
        request: Request, exc: RuntimeError
    ) -> JSONResponse:
        """RuntimeError (modelo não carregado, falha de load) → 503."""
        logger.warning("RuntimeError em %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                detail=str(exc),
                type="model_not_loaded",
            ).model_dump(),
        )

    @fastapi_app.exception_handler(Exception)
    async def _internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Exceção genérica → 500 com log de stacktrace."""
        logger.exception("Erro interno em %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="Erro interno do servidor.",
                type="internal",
            ).model_dump(),
        )

    # ── Montagem das rotas ────────────────────────────────────────
    fastapi_app.include_router(health_router)
    fastapi_app.include_router(predict_router)

    return fastapi_app


# ────────────────────────────────────────────────────────────────────────
# INSTÂNCIA PADRÃO — Usada pelo entry point geosteering-api e por uvicorn
# ────────────────────────────────────────────────────────────────────────
app = create_app()
