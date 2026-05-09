# -*- coding: utf-8 -*-
"""Testes do MCP server stdio handshake para numba-profiler (I1.10)."""

from __future__ import annotations

import asyncio
import importlib.util  # noqa: E402
import sys
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parents[1]
_SERVER_PATH = _MCP_DIR / "server.py"
_spec = importlib.util.spec_from_file_location("numba_profiler_server", _SERVER_PATH)
assert _spec and _spec.loader, "Falha ao carregar numba_profiler/server.py"
server = importlib.util.module_from_spec(_spec)
sys.modules["numba_profiler_server"] = server
_spec.loader.exec_module(server)


def test_server_imports_without_mcp_installed() -> None:
    assert hasattr(server, "TOOL_REGISTRY")
    assert hasattr(server, "_build_tool_definitions")
    assert hasattr(server, "main")
    assert hasattr(server, "SCENARIOS")


def test_tool_definitions_jsonschema_compliance() -> None:
    defs = server._build_tool_definitions()
    assert len(defs) == 6
    for d in defs:
        schema = d["inputSchema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        for prop_name, prop_schema in schema["properties"].items():
            assert (
                "type" in prop_schema
            ), f"Tool {d['name']} property {prop_name} missing type"


def test_tool_registry_callables_work_via_asyncio_to_thread() -> None:
    async def call_one() -> dict:
        fn = server.TOOL_REGISTRY["check_cpu_topology"]
        return await asyncio.to_thread(fn)

    result = asyncio.run(call_one())
    assert "error" not in result
    assert result["physical_cores"] >= 1


def test_main_emits_error_when_mcp_missing(monkeypatch, capsys) -> None:
    """Se mcp não pode ser importado, main() emite JSON e exit(1)."""
    try:
        import mcp  # noqa: F401
        import pytest

        pytest.skip("mcp já instalado; branch de erro só testável sem mcp")
    except ImportError:
        pass

    for mod_name in list(sys.modules):
        if mod_name == "mcp" or mod_name.startswith("mcp."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        server.main()
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "tools" in captured.out
    assert "run_scenario_benchmark" in captured.out
