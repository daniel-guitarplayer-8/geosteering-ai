# -*- coding: utf-8 -*-
"""Testes do MCP server stdio handshake (I1.9).

Estes testes garantem que o boilerplate MCP (Server + stdio_server) está
corretamente configurado — sem efetivamente subir processo. Para teste
end-to-end via JSONRPC subprocess, ver runtime via Claude Code.

Como mcp pode não estar instalado no env de teste, validamos:
1. Import do server.py funciona sem mcp (try/except em main())
2. _build_tool_definitions retorna 6 entries com inputSchema válido
3. TOOL_REGISTRY exporta callables compatíveis com asyncio.to_thread
"""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parents[1]
_SERVER_PATH = _MCP_DIR / "server.py"
_spec = importlib.util.spec_from_file_location("physics_validator_server", _SERVER_PATH)
assert _spec and _spec.loader, "Falha ao carregar physics_validator/server.py"
server = importlib.util.module_from_spec(_spec)
sys.modules["physics_validator_server"] = server
_spec.loader.exec_module(server)


def test_server_imports_without_mcp_installed() -> None:
    """O módulo server.py deve importar mesmo sem `mcp` no env."""
    # Se chegamos aqui, o import já passou. Sanity additional.
    assert hasattr(server, "TOOL_REGISTRY")
    assert hasattr(server, "_build_tool_definitions")
    assert hasattr(server, "main")


def test_tool_definitions_jsonschema_compliance() -> None:
    """Cada inputSchema deve seguir JSONSchema básico (type=object + properties)."""
    defs = server._build_tool_definitions()
    assert len(defs) == 6
    for d in defs:
        schema = d["inputSchema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        # Cada property deve ter "type"
        for prop_name, prop_schema in schema["properties"].items():
            assert (
                "type" in prop_schema
            ), f"Tool {d['name']} property {prop_name} missing type"


def test_tool_registry_callables_work_via_asyncio_to_thread() -> None:
    """asyncio.to_thread deve poder despachar handlers CPU-bound."""

    async def call_one() -> dict:
        fn = server.TOOL_REGISTRY["check_decoupling_factors"]
        return await asyncio.to_thread(fn, spacing_m=1.0)

    result = asyncio.run(call_one())
    assert result["passed"] is True
    assert abs(result["ratio_ACx_over_minus_ACp"] - 2.0) < 1e-12


def test_main_emits_error_when_mcp_missing(monkeypatch, capsys) -> None:
    """Se ``mcp`` não pode ser importado, ``main()`` emite JSON e exit(1).

    Mock determinístico via ``__import__``: força ImportError em ``mcp`` e
    submódulos, independente do ambiente de teste ter mcp instalado.
    """
    # Remove qualquer "mcp" cacheado em sys.modules
    for mod_name in list(sys.modules):
        if mod_name == "mcp" or mod_name.startswith("mcp."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    # Força ImportError quando alguém tentar `import mcp` ou submódulo
    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError(f"No module named '{name}' (test mock)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    with pytest.raises(SystemExit) as exc_info:
        server.main()
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    # JSON com lista de tools
    assert "tools" in captured.out
    assert "check_fortran_parity" in captured.out
