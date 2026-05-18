# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MÓDULO: api/routes/health.py                                              ║
# ║  Bloco: 11 — API REST (NOVO em v2.39)                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning      ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: FastAPI APIRouter                                              ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Docker CPU             ║
# ║  Pacote: geosteering_ai.api.routes                                         ║
# ║                                                                            ║
# ║  Propósito:                                                                ║
# ║    • GET /health — status do serviço (liveness/readiness probe)            ║
# ║    • Sem custo de TensorFlow — responde em <1 ms                           ║
# ║    • Reporta estado do singleton InferencePipeline (model_loaded)          ║
# ║                                                                            ║
# ║  Dependências: fastapi.APIRouter                                           ║
# ║  Exports: router (APIRouter)                                               ║
# ║  Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1        ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v2.39 (2026-05-18) — Implementação inicial (Sprint I2.7, commit 3/9)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Roteador `/health` — Health check sem custo de TensorFlow.

Adequado para Kubernetes liveness/readiness probes (custo ≪1 ms).
NÃO carrega TensorFlow; reporta apenas o estado do singleton em
`geosteering_ai.api.dependencies`.

Note:
    Para um endpoint que valida que o modelo está realmente operacional,
    use `POST /predict` com payload mínimo (forward pass real).
    Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from geosteering_ai.api import __version__
from geosteering_ai.api.dependencies import (
    get_loaded_model_path,
    is_pipeline_loaded,
)
from geosteering_ai.api.schemas import HealthResponse

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Roteador exportado para montagem em app.py
# ────────────────────────────────────────────────────────────────────────
router = APIRouter(tags=["health"])

__all__ = ["router"]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check do serviço",
    description=(
        "Retorna 'ok' se o serviço está respondendo, com informação "
        "do modelo carregado em memória. Custo: <1 ms (sem TensorFlow)."
    ),
)
async def health() -> HealthResponse:
    """Endpoint GET /health — retorna status do serviço.

    Returns:
        HealthResponse com status, versão, e estado do pipeline.

    Note:
        Este endpoint propositalmente NÃO chama `get_pipeline()` para
        evitar carregar TensorFlow em probes frequentes. O campo
        `model_loaded` reflete o estado atual do singleton.
    """
    loaded = is_pipeline_loaded()
    path = get_loaded_model_path()

    logger.debug("Health check — model_loaded=%s, version=%s", loaded, __version__)

    return HealthResponse(
        status="ok",
        version=__version__,
        model_loaded=loaded,
        model_path=path,
    )
