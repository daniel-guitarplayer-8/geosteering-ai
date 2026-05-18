# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MÓDULO: api/schemas.py                                                    ║
# ║  Bloco: 11 — API REST (NOVO em v2.39)                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning      ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: Pydantic v2 (validação + serialização JSON)                    ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Docker CPU             ║
# ║  Pacote: geosteering_ai.api                                                ║
# ║  Config: contratos JSON imutáveis, validados via Pydantic v2               ║
# ║                                                                            ║
# ║  Propósito:                                                                ║
# ║    • Define schemas Pydantic v2 para todos os endpoints                    ║
# ║    • Contrato alinhado ao formato canônico 22-colunas (config.n_columns)   ║
# ║    • Validação fail-fast de shape e ranges físicos                         ║
# ║    • Serialização JSON via .model_dump()                                   ║
# ║                                                                            ║
# ║  Dependências: pydantic v2 (extra [api])                                   ║
# ║  Exports: PredictRequest, PredictResponse, HealthResponse, ErrorResponse   ║
# ║  Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1        ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v2.39 (2026-05-18) — Implementação inicial (Sprint I2.7)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Schemas Pydantic v2 — Contratos JSON da API REST.

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  HIERARQUIA DE SCHEMAS                                              │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  PredictRequest   ─→  /predict (input)                              │
    │  PredictResponse  ←─  /predict (output, sem incerteza)              │
    │  HealthResponse   ←─  /health  (status do serviço)                  │
    │  ErrorResponse    ←─  qualquer endpoint (formato uniforme de erro)  │
    │                                                                      │
    │  Constantes do projeto preservadas:                                 │
    │    N_COLUMNS=22 (formato canônico raw_data)                         │
    │    SEQUENCE_LENGTH=600 (default, configurável via PipelineConfig)   │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

Note:
    Pydantic v2 é OBRIGATÓRIO (v1 não suportado).
    Referenciado em:
        - api/routes/predict.py: PredictRequest, PredictResponse
        - api/routes/health.py: HealthResponse
        - api/app.py: ErrorResponse (handlers de exceção)
    Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ────────────────────────────────────────────────────────────────────────
# D10: Constantes físicas espelhadas do PipelineConfig (errata imutável)
# ────────────────────────────────────────────────────────────────────────
# N_COLUMNS=22 é o formato canônico do projeto desde C25/Sprint v2.0.
# Validado em PipelineConfig.__post_init__(): config.n_columns == 22.
N_COLUMNS: int = 22

# Ranges físicos válidos para theta e frequência (errata CLAUDE.md).
THETA_MIN_DEG: float = 0.0
THETA_MAX_DEG: float = 90.0
FREQ_MIN_HZ: float = 100.0
FREQ_MAX_HZ: float = 1_000_000.0

# Limites de MC Dropout (UQ — incerteza via amostragem).
MC_SAMPLES_MIN: int = 2
MC_SAMPLES_MAX: int = 200


# ════════════════════════════════════════════════════════════════════════
# PREDICT REQUEST — Entrada do endpoint POST /predict
#
# Contrato alinhado ao InferencePipeline.predict() existente:
#   raw_data: shape (N, sequence_length, 22) — formato canônico 22-col
#   theta:    obrigatório se config.use_theta_as_feature=True
#   freq:     obrigatório se config.use_freq_as_feature=True
#   return_uncertainty + mc_samples: MC Dropout para UQ
# ════════════════════════════════════════════════════════════════════════


class PredictRequest(BaseModel):
    """Requisição de predição — dados brutos 22-colunas + parâmetros opcionais.

    Corresponde diretamente à assinatura de
    `InferencePipeline.predict(raw_data, theta, freq, return_uncertainty, mc_samples)`.

    Attributes:
        raw_data: Tensor 3D no formato (n_samples, sequence_length, 22).
            Cada linha contém uma medição LWD no formato canônico
            22-colunas (ver `geosteering_ai/data/loading.py` para o layout).
        theta: Ângulo de inclinação em graus, range [0, 90]. Obrigatório
            se o modelo treinado usar `use_theta_as_feature=True`.
        freq: Frequência EM em Hz, range [100, 1e6]. Obrigatório se o
            modelo treinado usar `use_freq_as_feature=True`.
        return_uncertainty: Se True, ativa MC Dropout — retorna média e
            desvio-padrão das `mc_samples` forward passes.
        mc_samples: Número de amostras MC Dropout (ignorado se
            `return_uncertainty=False`). Range [2, 200], default 30.

    Example:
        >>> req = PredictRequest(
        ...     raw_data=[[[0.0]*22]*600],  # (1 amostra, 600 medições)
        ...     theta=15.0, freq=20000.0,
        ... )
        >>> req.model_dump()  # serializa para JSON

    Note:
        Validação de shape acontece em dois níveis: Pydantic (este schema)
        verifica que o último eixo tem 22 colunas; o `InferencePipeline.predict()`
        revalida shape completo (3D) antes do forward pass.
        Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "raw_data": [[[0.0] * N_COLUMNS] * 4],
                "theta": 0.0,
                "freq": 20000.0,
                "return_uncertainty": False,
                "mc_samples": 30,
            }
        },
    )

    raw_data: List[List[List[float]]] = Field(
        ...,
        description=(
            "Tensor 3D (n_samples, sequence_length, 22) no formato "
            "canônico 22-colunas. Lista aninhada para serialização JSON."
        ),
    )
    theta: Optional[float] = Field(
        default=None,
        ge=THETA_MIN_DEG,
        le=THETA_MAX_DEG,
        description="Ângulo de inclinação em graus (0–90).",
    )
    freq: Optional[float] = Field(
        default=None,
        ge=FREQ_MIN_HZ,
        le=FREQ_MAX_HZ,
        description="Frequência EM em Hz (100–1e6).",
    )
    return_uncertainty: bool = Field(
        default=False,
        description="Se True, retorna média e desvio-padrão via MC Dropout.",
    )
    mc_samples: int = Field(
        default=30,
        ge=MC_SAMPLES_MIN,
        le=MC_SAMPLES_MAX,
        description="Número de amostras MC Dropout (ignorado se return_uncertainty=False).",
    )

    @field_validator("raw_data")
    @classmethod
    def _validate_raw_data_shape(
        cls, v: List[List[List[float]]]
    ) -> List[List[List[float]]]:
        """Valida que raw_data tem shape 3D não-vazio com último eixo = 22 colunas.

        Args:
            v: Valor recebido para raw_data.

        Returns:
            O mesmo valor, se válido.

        Raises:
            ValueError: Se raw_data for vazio, não-3D ou tiver eixo final ≠ 22.

        Note:
            Validação rápida sem materializar np.ndarray (preserva latência).
            Shape completo é revalidado em `InferencePipeline.predict()`.
        """
        if not v:
            raise ValueError("raw_data não pode ser vazio (≥1 amostra).")

        # ── Verifica 1º eixo (n_samples ≥ 1) ──────────────────────
        n_samples = len(v)
        if n_samples == 0:
            raise ValueError("raw_data deve ter ≥1 amostra.")

        # ── Verifica 2º eixo (sequence_length ≥ 1) e ──────────────
        # ── 3º eixo (n_columns == 22) em todas as amostras ────────
        for i, sample in enumerate(v):
            if not sample:
                raise ValueError(f"raw_data[{i}] tem sequence_length=0; mínimo é 1.")
            for j, row in enumerate(sample):
                if len(row) != N_COLUMNS:
                    raise ValueError(
                        f"raw_data[{i}][{j}] tem {len(row)} colunas; "
                        f"esperado {N_COLUMNS} (formato canônico 22-col)."
                    )

        return v


# ════════════════════════════════════════════════════════════════════════
# PREDICT RESPONSE — Saída do endpoint POST /predict
#
# Retorna predicoes em Ohm.m (dominio original, ja com inverse_target_scaling).
# Campo `uncertainty` so eh preenchido quando return_uncertainty=True.
# ════════════════════════════════════════════════════════════════════════


class PredictResponse(BaseModel):
    """Resposta de predição — resistividades preditas no domínio físico (Ω·m).

    Attributes:
        predictions: Tensor 3D (n_samples, sequence_length, n_targets),
            geralmente n_targets=2 para [rho_h, rho_v]. Valores em Ω·m
            (já com inverse_target_scaling aplicado — não em log10).
        uncertainty: Desvio-padrão das predições via MC Dropout (mesma
            shape que predictions). None se return_uncertainty=False.
            **Atenção**: valores em unidades log10 (decadas), NÃO em Ω·m.
            Aplicar 10**std seria fisicamente incorreto para desvio-padrão.
        shape: Lista [n_samples, sequence_length, n_targets] para validação
            rápida no cliente.
        latency_ms: Latência total do endpoint em milissegundos (inclui
            pré-processamento, forward pass e pós-processamento).
        model_type: String identificando a arquitetura (ex: "ResNet_18").
            Útil para clientes monitoram drift de modelo.
        mc_samples: Número de amostras MC Dropout efetivamente usadas
            (apenas presente se return_uncertainty=True).

    Example:
        >>> resp = PredictResponse(
        ...     predictions=[[[1.5, 2.0]]],
        ...     shape=[1, 1, 2],
        ...     latency_ms=12.4,
        ...     model_type="ResNet_18",
        ... )

    Note:
        Predições retornadas em domínio físico (Ω·m); o cliente NÃO deve
        aplicar 10** novamente. Para incerteza, valores ficam em log10
        (decadas) — esta é uma escolha de física, não bug. Ref: doc
        `inverse_target_scaling` em data/scaling.py.
    """

    model_config = ConfigDict(extra="forbid")

    predictions: List[List[List[float]]] = Field(
        ...,
        description="Predições de resistividade (Ω·m), shape (N, seq, n_targets).",
    )
    uncertainty: Optional[List[List[List[float]]]] = Field(
        default=None,
        description=(
            "Desvio-padrão via MC Dropout (em log10 decadas, não Ω·m). "
            "Presente apenas se return_uncertainty=True."
        ),
    )
    shape: List[int] = Field(
        ...,
        description="Shape [n_samples, sequence_length, n_targets].",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Latência total do endpoint em milissegundos.",
    )
    model_type: str = Field(
        ...,
        description="Arquitetura do modelo (ex: ResNet_18, ModernTCN).",
    )
    mc_samples: Optional[int] = Field(
        default=None,
        description="Número de amostras MC Dropout (se UQ ativada).",
    )


# ════════════════════════════════════════════════════════════════════════
# HEALTH RESPONSE — Saída do endpoint GET /health
#
# Sem autenticação, sem custo de TF (ideal para liveness/readiness probes).
# ════════════════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    """Resposta de health check — status do serviço e modelo carregado.

    Attributes:
        status: String literal "ok" quando o serviço está respondendo.
            Em versões futuras pode incluir "degraded" para warm-up parcial.
        version: Versão do pacote `geosteering_ai.api` (ex: "2.39.0").
        model_loaded: True se o `InferencePipeline` já foi carregado em
            memória (após a primeira `/predict` ou via env var no startup).
        model_path: Caminho absoluto do diretório do pipeline carregado,
            ou None se ainda não carregado.

    Example:
        >>> HealthResponse(status="ok", version="2.39.0", model_loaded=False)

    Note:
        `/health` NÃO carrega TensorFlow — o campo `model_loaded` reflete
        o estado atual do singleton em `dependencies.py`. Adequado para
        Kubernetes liveness probes (custo ≪1 ms).
    """

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description='Literal "ok" quando saudável.')
    version: str = Field(..., description="Versão da API (geosteering_ai.api).")
    model_loaded: bool = Field(
        ..., description="True se InferencePipeline já carregado em memória."
    )
    model_path: Optional[str] = Field(
        default=None, description="Caminho do pipeline carregado (None se não)."
    )


# ════════════════════════════════════════════════════════════════════════
# ERROR RESPONSE — Formato uniforme para todos os erros 4xx/5xx
#
# Padroniza o corpo de erro retornado pelos exception handlers em app.py.
# ════════════════════════════════════════════════════════════════════════


class ErrorResponse(BaseModel):
    """Resposta de erro padronizada para todos os endpoints.

    Attributes:
        detail: Mensagem de erro legível em PT-BR.
        type: Categoria de erro para tratamento programático no cliente.
            Valores conhecidos:
                - "model_not_loaded" (503): GEOSTEERING_MODEL_PATH não setado
                  ou pipeline inválido.
                - "validation_error" (422): Shape ou tipo de dado inválido.
                - "internal" (500): Erro não-categorizado (logado com stacktrace).

    Example:
        >>> ErrorResponse(detail="Modelo não carregado.", type="model_not_loaded")
    """

    model_config = ConfigDict(extra="forbid")

    detail: str = Field(..., description="Mensagem de erro em PT-BR.")
    type: str = Field(
        ..., description="Categoria do erro (model_not_loaded, validation_error, ...)."
    )


__all__ = [
    "PredictRequest",
    "PredictResponse",
    "HealthResponse",
    "ErrorResponse",
    # Constantes exportadas para uso em tests/dependencies
    "N_COLUMNS",
    "THETA_MIN_DEG",
    "THETA_MAX_DEG",
    "FREQ_MIN_HZ",
    "FREQ_MAX_HZ",
    "MC_SAMPLES_MIN",
    "MC_SAMPLES_MAX",
]
