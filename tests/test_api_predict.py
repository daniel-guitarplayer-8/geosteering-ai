# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTE: tests/test_api_predict.py                                          ║
# ║  Bloco: 11 — API REST                                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Sprint v2.39 (I2.7)                                ║
# ║  Cobertura: POST /predict — happy path, errors, MC Dropout                 ║
# ║  Custo: CPU-only via mock do InferencePipeline (sem TensorFlow)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes do endpoint POST /predict com mock do InferencePipeline.

Usa `app.dependency_overrides` para injetar um pipeline mock sem
carregar TensorFlow — testes rodam em <1s mesmo em CI sem GPU.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", minversion="0.110")
httpx = pytest.importorskip("httpx", minversion="0.27")

from fastapi.testclient import TestClient  # noqa: E402

from geosteering_ai.api import dependencies  # noqa: E402
from geosteering_ai.api.app import app  # noqa: E402
from geosteering_ai.api.dependencies import reset_pipeline_cache  # noqa: E402
from geosteering_ai.api.schemas import N_COLUMNS  # noqa: E402

# ────────────────────────────────────────────────────────────────────────
# Helpers / Fixtures
# ────────────────────────────────────────────────────────────────────────


def _make_mock_pipeline(
    n_samples: int = 1,
    seq_len: int = 4,
    n_targets: int = 2,
    model_type: str = "ResNet_18_Mock",
    return_uncertainty: bool = False,
) -> MagicMock:
    """Constrói um mock de InferencePipeline com .predict() determinístico.

    Args:
        n_samples, seq_len, n_targets: shape esperado da saída.
        model_type: string retornada via mock.config.model_type.
        return_uncertainty: se True, .predict() retorna tuple (mean, std).
    """
    mock = MagicMock()
    mock.config.model_type = model_type
    mock.config.n_columns = N_COLUMNS

    mean = np.ones((n_samples, seq_len, n_targets), dtype=np.float32) * 1.5
    std = np.ones((n_samples, seq_len, n_targets), dtype=np.float32) * 0.1

    def _predict(raw_data: Any, **kwargs: Any) -> Any:
        if kwargs.get("return_uncertainty", False):
            return (mean, std)
        return mean

    mock.predict.side_effect = _predict
    return mock


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limpa cache de pipeline + env vars + dependency_overrides entre testes."""
    monkeypatch.delenv("GEOSTEERING_MODEL_PATH", raising=False)
    reset_pipeline_cache()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    reset_pipeline_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_pipeline_loaded(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Injeta um mock no singleton de dependencies (monkeypatch direto).

    Como o handler chama `get_pipeline()` diretamente (não via Depends),
    monkeypatchamos o módulo-level `_pipeline_instance` em dependencies.
    """
    mock = _make_mock_pipeline()
    monkeypatch.setattr(dependencies, "_pipeline_instance", mock)
    monkeypatch.setattr(dependencies, "_pipeline_loaded_path", "/mock/path")
    return mock


# ────────────────────────────────────────────────────────────────────────
# Testes — Happy path
# ────────────────────────────────────────────────────────────────────────


class TestPredictHappyPath:
    """Testes do POST /predict com mock pipeline carregado."""

    def test_predict_returns_200(
        self, client: TestClient, mock_pipeline_loaded: MagicMock
    ) -> None:
        """Request válida deve retornar 200."""
        r = client.post(
            "/predict",
            json={"raw_data": [[[0.0] * N_COLUMNS] * 4]},
        )
        assert r.status_code == 200, r.text

    def test_predict_response_shape(
        self, client: TestClient, mock_pipeline_loaded: MagicMock
    ) -> None:
        """Resposta deve ter predictions (1, 4, 2) e campos esperados."""
        body = client.post(
            "/predict",
            json={"raw_data": [[[0.0] * N_COLUMNS] * 4]},
        ).json()
        assert body["shape"] == [1, 4, 2]
        assert len(body["predictions"]) == 1
        assert len(body["predictions"][0]) == 4
        assert len(body["predictions"][0][0]) == 2
        assert body["model_type"] == "ResNet_18_Mock"
        assert body["latency_ms"] >= 0.0
        assert body["uncertainty"] is None
        assert body["mc_samples"] is None

    def test_predict_passes_theta_freq_to_pipeline(
        self, client: TestClient, mock_pipeline_loaded: MagicMock
    ) -> None:
        """theta e freq devem ser repassados ao pipeline.predict()."""
        client.post(
            "/predict",
            json={
                "raw_data": [[[0.0] * N_COLUMNS] * 4],
                "theta": 30.0,
                "freq": 50000.0,
            },
        )
        call_kwargs = mock_pipeline_loaded.predict.call_args.kwargs
        assert call_kwargs["theta"] == 30.0
        assert call_kwargs["freq"] == 50000.0

    def test_predict_with_uncertainty(
        self, client: TestClient, mock_pipeline_loaded: MagicMock
    ) -> None:
        """return_uncertainty=True deve preencher uncertainty + mc_samples."""
        body = client.post(
            "/predict",
            json={
                "raw_data": [[[0.0] * N_COLUMNS] * 4],
                "return_uncertainty": True,
                "mc_samples": 50,
            },
        ).json()
        assert body["uncertainty"] is not None
        assert len(body["uncertainty"]) == 1
        assert body["mc_samples"] == 50


# ────────────────────────────────────────────────────────────────────────
# Testes — Erros
# ────────────────────────────────────────────────────────────────────────


class TestPredictErrors:
    """Casos de erro do POST /predict."""

    def test_predict_503_when_model_not_loaded(self, client: TestClient) -> None:
        """Sem GEOSTEERING_MODEL_PATH, deve retornar 503 model_not_loaded."""
        r = client.post("/predict", json={"raw_data": [[[0.0] * N_COLUMNS]]})
        assert r.status_code == 503
        body = r.json()
        assert body["type"] == "model_not_loaded"
        assert "GEOSTEERING_MODEL_PATH" in body["detail"]

    def test_predict_422_wrong_n_columns(self, client: TestClient) -> None:
        """Shape com último eixo != 22 deve dar 422 (Pydantic)."""
        r = client.post("/predict", json={"raw_data": [[[0.0] * 21]]})
        assert r.status_code == 422

    def test_predict_422_theta_out_of_range(self, client: TestClient) -> None:
        """theta fora de [0, 90] deve dar 422."""
        r = client.post(
            "/predict",
            json={"raw_data": [[[0.0] * N_COLUMNS]], "theta": 999.0},
        )
        assert r.status_code == 422

    def test_predict_422_extra_field(self, client: TestClient) -> None:
        """Campo desconhecido deve dar 422 (extra=forbid)."""
        r = client.post(
            "/predict",
            json={"raw_data": [[[0.0] * N_COLUMNS]], "rogue": True},
        )
        assert r.status_code == 422

    def test_predict_422_pipeline_value_error(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pipeline levantando ValueError deve virar 422."""
        mock = _make_mock_pipeline()
        mock.predict.side_effect = ValueError("shape interno inválido (mock)")
        monkeypatch.setattr(dependencies, "_pipeline_instance", mock)
        monkeypatch.setattr(dependencies, "_pipeline_loaded_path", "/mock")
        r = client.post(
            "/predict",
            json={"raw_data": [[[0.0] * N_COLUMNS]]},
        )
        assert r.status_code == 422
        assert "shape interno inválido" in r.text

    def test_predict_500_unexpected_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exceção genérica do pipeline deve virar 500 internal.

        Usa raise_server_exceptions=False para que o TestClient deixe o
        handler global capturar a exceção em vez de re-levantá-la.
        """
        mock = _make_mock_pipeline()
        mock.predict.side_effect = OSError("disco cheio (mock)")
        monkeypatch.setattr(dependencies, "_pipeline_instance", mock)
        monkeypatch.setattr(dependencies, "_pipeline_loaded_path", "/mock")
        client_no_reraise = TestClient(app, raise_server_exceptions=False)
        r = client_no_reraise.post(
            "/predict",
            json={"raw_data": [[[0.0] * N_COLUMNS]]},
        )
        assert r.status_code == 500
        assert r.json()["type"] == "internal"


# ────────────────────────────────────────────────────────────────────────
# Testes — /health após carregar mock
# ────────────────────────────────────────────────────────────────────────


class TestHealthAfterMockLoad:
    """Valida que /health reflete o estado do singleton após mock."""

    def test_health_reports_loaded_when_mock_set(
        self, client: TestClient, mock_pipeline_loaded: MagicMock
    ) -> None:
        """Com mock injetado, /health deve reportar model_loaded=True."""
        body = client.get("/health").json()
        assert body["model_loaded"] is True
        assert body["model_path"] == "/mock/path"
