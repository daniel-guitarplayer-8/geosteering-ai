# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MÓDULO: api/__init__.py                                                   ║
# ║  Bloco: 11 — API REST (NOVO em v2.39)                                      ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning      ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: FastAPI + Pydantic v2 + Uvicorn (extra opcional [api])         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Docker CPU             ║
# ║  Pacote: geosteering_ai.api (pip install -e ".[api]")                      ║
# ║  Config: variáveis de ambiente (GEOSTEERING_MODEL_PATH, *_CORS_ORIGINS)    ║
# ║                                                                            ║
# ║  Propósito:                                                                ║
# ║    • Expõe `InferencePipeline` via HTTP (FastAPI)                          ║
# ║    • Endpoints MVP: GET /health + POST /predict                            ║
# ║    • Sem autenticação (MVP); CORS configurável via env                     ║
# ║    • Lazy load do TensorFlow — /health funciona sem modelo                 ║
# ║                                                                            ║
# ║  Dependências: fastapi, pydantic v2, uvicorn (extra [api])                 ║
# ║  Exports: __version__ (string)                                             ║
# ║  Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1        ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v2.39 (2026-05-18) — Implementação inicial (Sprint I2.7)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Pacote API REST — Exposição HTTP do InferencePipeline.

Este pacote é **opcional**: requer instalação via `pip install -e ".[api]"`.
Caso as dependências FastAPI/Pydantic/Uvicorn não estejam disponíveis,
o restante do pacote `geosteering_ai` permanece funcional.

Exemplo de uso (linha de comando):

.. code-block:: bash

    pip install -e ".[api]"
    geosteering-api --host 0.0.0.0 --port 8000
    curl http://localhost:8000/health

Endpoints disponíveis:

============== ===== ==========================================
Método         Path  Descrição
============== ===== ==========================================
GET            /health   Status do serviço + modelo carregado
POST           /predict  Inferência sobre raw_data 22-col
============== ===== ==========================================

Note:
    O carregamento do `InferencePipeline` é **lazy**: só ocorre na
    primeira requisição a `/predict`. Isso permite que `/health`
    responda em milissegundos mesmo sem TensorFlow disponível.
    Ref: docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md §2.1.
"""

from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────
# D8: Versão do pacote API — espelha geosteering_ai.__version__
# ────────────────────────────────────────────────────────────────────────
# Mantida como string literal para evitar import circular com
# o pacote pai (que pode importar tensorflow em alguns subpacotes).
__version__ = "2.39.0"

__all__ = [
    "__version__",
]
