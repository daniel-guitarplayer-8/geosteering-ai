# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MÓDULO: api/routes/__init__.py                                            ║
# ║  Bloco: 11 — API REST (NOVO em v2.39)                                      ║
# ║                                                                            ║
# ║  Marca o diretório `routes/` como subpacote Python.                        ║
# ║  Os submódulos `health.py` e `predict.py` são importados diretamente       ║
# ║  por `geosteering_ai.api.app` (sem re-export aqui para evitar imports      ║
# ║  desnecessários quando apenas um dos roteadores é usado).                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote `routes` — Define APIRouters montados em app.py."""
