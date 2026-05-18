# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MÓDULO: api/routes/predict.py                                             ║
# ║  Bloco: 11 — API REST (NOVO em v2.39)                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning      ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: FastAPI APIRouter + InferencePipeline                          ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Docker CPU             ║
# ║  Pacote: geosteering_ai.api.routes                                         ║
# ║                                                                            ║
# ║  Propósito:                                                                ║
# ║    • POST /predict — inferência sobre raw_data 22-col                      ║
# ║    • Integra InferencePipeline.predict() via Depends(get_pipeline)         ║
# ║    • Suporta MC Dropout para UQ (return_uncertainty=true)                  ║
# ║    • Mede latência fim-a-fim (latency_ms)                                  ║
# ║                                                                            ║
# ║  Dependências: fastapi, numpy, InferencePipeline (lazy via Depends)        ║
# ║  Exports: router (APIRouter)                                               ║
# ║  Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1        ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v2.39 (2026-05-18) — Implementação inicial (Sprint I2.7, commit 4/9)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Roteador `/predict` — Inferência HTTP sobre raw_data 22-colunas.

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  FLUXO DA REQUISIÇÃO POST /predict                                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Request JSON                                                       │
    │     ↓                                                                │
    │  Pydantic (schemas.PredictRequest)                                  │
    │     ↓ validação shape + ranges                                      │
    │  Depends(get_pipeline)                                              │
    │     ↓ singleton InferencePipeline (lazy load 1x)                    │
    │  np.asarray(raw_data, dtype=float32)                                │
    │     ↓                                                                │
    │  pipeline.predict(raw_data, theta=, freq=, return_uncertainty=)     │
    │     ↓ FV + GS + scale + model.predict + inverse_scaling             │
    │  PredictResponse (predicoes em Ohm.m + latency_ms)                  │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

Note:
    Conversão list-→ndarray usa dtype=float32 para alinhar com o
    InferencePipeline (que opera em float32 internamente).
    Latência inclui serialização JSON de saída (overhead ~5-10% para
    payloads grandes; aceitável no MVP).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, status

from geosteering_ai.api.dependencies import get_pipeline
from geosteering_ai.api.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Roteador exportado para montagem em app.py
# ────────────────────────────────────────────────────────────────────────
router = APIRouter(tags=["predict"])

__all__ = ["router"]


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Inferência de resistividade (raw_data 22-col → ρ_h, ρ_v)",
    description=(
        "Executa o `InferencePipeline` carregado em memória sobre o tensor "
        "`raw_data` no formato canônico 22-colunas. Suporta MC Dropout para "
        "estimativa de incerteza (`return_uncertainty=true`).\n\n"
        "**Respostas de erro**:\n"
        "- 422: shape inválido (Pydantic)\n"
        "- 503: modelo não carregado (`GEOSTEERING_MODEL_PATH` ausente/inválido)\n"
        "- 500: erro interno (log com stacktrace no servidor)"
    ),
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Endpoint POST /predict — invoca InferencePipeline.predict().

    Args:
        request: Schema validado contendo raw_data + theta/freq/UQ opcionais.

    Returns:
        PredictResponse com predições em Ω·m, shape e latency_ms.

    Raises:
        HTTPException(422): Se conversão para ndarray falhar (jagged array)
            ou se `pipeline.predict()` rejeitar shape (defesa em profundidade
            — Pydantic já valida último eixo == 22).
        HTTPException(503): Propagado de `get_pipeline()` via exception handler
            em `app.py` quando modelo não está carregado.

    Note:
        `get_pipeline()` é chamado INTERNAMENTE (não via Depends) para
        garantir que a validação Pydantic do body aconteça PRIMEIRO. Caso
        contrário, requisições malformadas sem modelo carregado retornariam
        503 em vez do 422 esperado.
        Conversão list → ndarray usa dtype=float32 (mesmo dtype interno do
        InferencePipeline) para evitar cópia adicional no Passo 1 do pipeline.
        Ref: geosteering_ai/inference/pipeline.py:264
    """
    t_start = time.perf_counter()

    # ── Passo 1: Converter list aninhada → np.ndarray (float32) ──
    # InferencePipeline.predict() valida shape 3D + n_columns interno;
    # aqui só convertemos. ValueError de np.asarray (ex: jagged array)
    # vira HTTPException 422 via try/except.
    try:
        raw = np.asarray(request.raw_data, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        logger.warning("Falha ao converter raw_data para ndarray: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"raw_data inválido: não foi possível converter para tensor "
                f"3D float32. Detalhe: {exc}"
            ),
        ) from exc

    # ── Passo 2: Resolver pipeline APÓS validação do body ────────
    # Importante: chamada manual (não Depends) para garantir ordem
    # Pydantic-then-Pipeline. RuntimeError → 503 via app.py handler.
    pipeline = get_pipeline()

    # ── Passo 3: Executar pipeline.predict() ─────────────────────
    # ValueError do pipeline (shape errado) → 422
    try:
        result = pipeline.predict(
            raw,
            theta=request.theta,
            freq=request.freq,
            return_uncertainty=request.return_uncertainty,
            mc_samples=request.mc_samples,
        )
    except ValueError as exc:
        logger.warning("Pipeline rejeitou shape: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    # ── Passo 4: Desempacotar resposta (com ou sem incerteza) ────
    # Narrowing manual para satisfazer mypy: a assinatura de predict()
    # retorna np.ndarray | tuple[np.ndarray, np.ndarray].
    y_pred: np.ndarray
    uncertainty_list: Optional[list] = None
    if request.return_uncertainty:
        assert isinstance(
            result, tuple
        ), "predict() deve retornar tuple quando return_uncertainty=True"
        y_pred, y_std = result
        uncertainty_list = y_std.tolist()
    else:
        assert isinstance(
            result, np.ndarray
        ), "predict() deve retornar ndarray quando return_uncertainty=False"
        y_pred = result

    # ── Passo 5: Construir PredictResponse ───────────────────────
    latency_ms = (time.perf_counter() - t_start) * 1000.0
    shape = list(y_pred.shape)

    logger.info(
        "Predição concluída — shape=%s, latency=%.2fms, uq=%s, model=%s",
        shape,
        latency_ms,
        request.return_uncertainty,
        pipeline.config.model_type,
    )

    return PredictResponse(
        predictions=y_pred.tolist(),
        uncertainty=uncertainty_list,
        shape=shape,
        latency_ms=latency_ms,
        model_type=pipeline.config.model_type,
        mc_samples=request.mc_samples if request.return_uncertainty else None,
    )
