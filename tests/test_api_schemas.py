# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTE: tests/test_api_schemas.py                                          ║
# ║  Bloco: 11 — API REST                                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Sprint v2.39 (I2.7)                                ║
# ║  Cobertura: schemas Pydantic v2 — validação de shape e ranges físicos      ║
# ║  Custo: CPU-only, sem TensorFlow (apenas pydantic)                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes dos schemas Pydantic da API REST (CPU-only, sem TF)."""

from __future__ import annotations

import pytest

# pytest.importorskip antes de qualquer import do pacote api (gate de [api])
pydantic = pytest.importorskip("pydantic", minversion="2.0")

from geosteering_ai.api.schemas import (  # noqa: E402
    N_COLUMNS,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


class TestPredictRequest:
    """Valida o schema PredictRequest (shape + ranges físicos)."""

    def test_minimal_valid_request(self) -> None:
        """Request mínima válida: apenas raw_data com 1 amostra × 1 medição × 22 col."""
        req = PredictRequest(raw_data=[[[0.0] * N_COLUMNS]])
        assert req.theta is None
        assert req.freq is None
        assert req.return_uncertainty is False
        assert req.mc_samples == 30  # default

    def test_full_request_with_all_optional(self) -> None:
        """Request completa com todos os campos opcionais."""
        req = PredictRequest(
            raw_data=[[[0.1] * N_COLUMNS] * 5],
            theta=15.0,
            freq=20000.0,
            return_uncertainty=True,
            mc_samples=50,
        )
        assert req.theta == 15.0
        assert req.freq == 20000.0
        assert req.return_uncertainty is True
        assert req.mc_samples == 50

    def test_wrong_n_columns_raises(self) -> None:
        """raw_data com último eixo != 22 deve falhar."""
        with pytest.raises(pydantic.ValidationError) as exc_info:
            PredictRequest(raw_data=[[[0.0] * 21]])  # 21 colunas
        assert "21 colunas" in str(exc_info.value)
        assert "22" in str(exc_info.value)

    def test_empty_raw_data_raises(self) -> None:
        """raw_data vazio deve falhar."""
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(raw_data=[])

    def test_empty_sequence_raises(self) -> None:
        """raw_data com sequence_length=0 deve falhar."""
        with pytest.raises(pydantic.ValidationError) as exc_info:
            PredictRequest(raw_data=[[]])
        assert "sequence_length=0" in str(exc_info.value)

    def test_theta_out_of_range_raises(self) -> None:
        """theta fora do range [0, 90] deve falhar."""
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(raw_data=[[[0.0] * N_COLUMNS]], theta=-1.0)
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(raw_data=[[[0.0] * N_COLUMNS]], theta=91.0)

    def test_freq_out_of_range_raises(self) -> None:
        """freq fora do range [100, 1e6] deve falhar."""
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(raw_data=[[[0.0] * N_COLUMNS]], freq=50.0)
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(raw_data=[[[0.0] * N_COLUMNS]], freq=2_000_000.0)

    def test_mc_samples_out_of_range_raises(self) -> None:
        """mc_samples fora do range [2, 200] deve falhar."""
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(raw_data=[[[0.0] * N_COLUMNS]], mc_samples=1)
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(raw_data=[[[0.0] * N_COLUMNS]], mc_samples=999)

    def test_extra_field_raises(self) -> None:
        """Campos extras devem ser rejeitados (extra=forbid)."""
        with pytest.raises(pydantic.ValidationError):
            PredictRequest(raw_data=[[[0.0] * N_COLUMNS]], unknown_field="x")


class TestPredictResponse:
    """Valida o schema PredictResponse."""

    def test_minimal_valid_response(self) -> None:
        """Resposta sem incerteza: apenas predictions + shape + latency + model_type."""
        resp = PredictResponse(
            predictions=[[[1.0, 2.0]]],
            shape=[1, 1, 2],
            latency_ms=12.4,
            model_type="ResNet_18",
        )
        assert resp.uncertainty is None
        assert resp.mc_samples is None
        assert resp.latency_ms == 12.4

    def test_response_with_uncertainty(self) -> None:
        """Resposta com MC Dropout: uncertainty + mc_samples preenchidos."""
        resp = PredictResponse(
            predictions=[[[1.0, 2.0]]],
            uncertainty=[[[0.1, 0.2]]],
            shape=[1, 1, 2],
            latency_ms=22.4,
            model_type="ResNet_18",
            mc_samples=30,
        )
        assert resp.uncertainty == [[[0.1, 0.2]]]
        assert resp.mc_samples == 30

    def test_negative_latency_raises(self) -> None:
        """latency_ms < 0 deve falhar (ge=0.0)."""
        with pytest.raises(pydantic.ValidationError):
            PredictResponse(
                predictions=[[[1.0, 2.0]]],
                shape=[1, 1, 2],
                latency_ms=-1.0,
                model_type="ResNet_18",
            )


class TestHealthResponse:
    """Valida o schema HealthResponse."""

    def test_minimal_health(self) -> None:
        """Health sem modelo carregado."""
        resp = HealthResponse(
            status="ok",
            version="2.39.0",
            model_loaded=False,
        )
        assert resp.model_path is None

    def test_health_with_loaded_model(self) -> None:
        """Health com modelo carregado e path informado."""
        resp = HealthResponse(
            status="ok",
            version="2.39.0",
            model_loaded=True,
            model_path="/models/resnet18",
        )
        assert resp.model_loaded is True
        assert resp.model_path == "/models/resnet18"


class TestErrorResponse:
    """Valida o schema ErrorResponse."""

    def test_minimal_error(self) -> None:
        """Erro com detail + type."""
        err = ErrorResponse(detail="Modelo não carregado.", type="model_not_loaded")
        assert err.detail == "Modelo não carregado."
        assert err.type == "model_not_loaded"
