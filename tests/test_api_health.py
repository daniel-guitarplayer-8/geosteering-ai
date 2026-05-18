# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TESTE: tests/test_api_health.py                                           ║
# ║  Bloco: 11 — API REST                                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Sprint v2.39 (I2.7)                                ║
# ║  Cobertura: GET /health (status + version + model_loaded + model_path)     ║
# ║  Custo: CPU-only, sem TensorFlow (apenas fastapi + httpx)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Testes do endpoint GET /health (CPU-only, sem TF)."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", minversion="0.110")
httpx = pytest.importorskip("httpx", minversion="0.27")

from fastapi.testclient import TestClient  # noqa: E402

from geosteering_ai.api import __version__  # noqa: E402
from geosteering_ai.api.app import app  # noqa: E402
from geosteering_ai.api.dependencies import reset_pipeline_cache  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garante estado limpo entre testes (cache de pipeline + env vars)."""
    monkeypatch.delenv("GEOSTEERING_MODEL_PATH", raising=False)
    monkeypatch.delenv("GEOSTEERING_API_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("GEOSTEERING_API_DOCS_ENABLED", raising=False)
    reset_pipeline_cache()
    yield
    reset_pipeline_cache()


@pytest.fixture
def client() -> TestClient:
    """TestClient FastAPI compartilhado entre testes do módulo."""
    return TestClient(app)


class TestHealthEndpoint:
    """Testes do endpoint GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health deve sempre retornar 200, mesmo sem modelo."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_payload_structure(self, client: TestClient) -> None:
        """Payload deve ter exatamente: status, version, model_loaded, model_path."""
        body = client.get("/health").json()
        assert set(body.keys()) == {"status", "version", "model_loaded", "model_path"}

    def test_health_status_ok(self, client: TestClient) -> None:
        """status deve ser literalmente 'ok'."""
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_version_matches_package(self, client: TestClient) -> None:
        """version deve corresponder a geosteering_ai.api.__version__."""
        body = client.get("/health").json()
        assert body["version"] == __version__

    def test_health_model_not_loaded_by_default(self, client: TestClient) -> None:
        """Sem GEOSTEERING_MODEL_PATH, model_loaded deve ser False."""
        body = client.get("/health").json()
        assert body["model_loaded"] is False
        assert body["model_path"] is None

    def test_health_propagates_request_id(self, client: TestClient) -> None:
        """Middleware deve refletir x-request-id no response header."""
        r = client.get("/health", headers={"x-request-id": "test-12345"})
        assert r.headers.get("x-request-id") == "test-12345"

    def test_health_generates_request_id_when_absent(self, client: TestClient) -> None:
        """Middleware deve gerar x-request-id quando ausente na request."""
        r = client.get("/health")
        rid = r.headers.get("x-request-id")
        assert rid is not None
        assert len(rid) > 0


class TestOpenAPIExposure:
    """Testes da exposição condicional do OpenAPI/docs."""

    def test_openapi_json_exposed_by_default(self, client: TestClient) -> None:
        """OpenAPI deve estar disponível por default."""
        r = client.get("/openapi.json")
        assert r.status_code == 200
        info = r.json()["info"]
        assert info["title"] == "Geosteering AI — API REST"
        assert info["version"] == __version__

    def test_docs_endpoint_available(self, client: TestClient) -> None:
        """Swagger UI deve responder em /docs."""
        r = client.get("/docs")
        assert r.status_code == 200
        assert "swagger" in r.text.lower() or "openapi" in r.text.lower()
