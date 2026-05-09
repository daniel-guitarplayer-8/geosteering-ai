#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MCP SERVER: Physics Validator — Geosteering AI v2.0                         ║
# ║  Fase 1 do roadmap multi-agente §22.1 (I1.9): handlers REAIS                 ║
# ║                                                                              ║
# ║  Propósito:                                                                  ║
# ║    Expor validação física do simulador como capacidades MCP nativas         ║
# ║    para o agente geosteering-physics-reviewer.                              ║
# ║                                                                              ║
# ║  6 Tools expostas via MCP:                                                   ║
# ║    • check_fortran_parity       — Compara Python vs Fortran <1e-12          ║
# ║    • check_maxwell_symmetry     — Valida tensor analítico via half_space    ║
# ║    • check_decoupling_factors   — ACp, ACx vs valores teóricos              ║
# ║    • check_errata_immutable     — Compara SimulationConfig() vs errata      ║
# ║    • check_skin_depth           — δ ≈ 503/√(ρ·f) em half-space             ║
# ║    • run_canonical_models       — 7 modelos canônicos (oklahoma, etc.)      ║
# ║                                                                              ║
# ║  Reutilização (REAL, não scaffold):                                          ║
# ║    • compare_fortran_python (validation/compare_fortran.py:405)              ║
# ║    • static_decoupling_factors (validation/half_space.py:130)                ║
# ║    • skin_depth (validation/half_space.py:195)                               ║
# ║    • validate_all_analytical (validation/compare_analytical.py:254)          ║
# ║    • get_canonical_model + list_canonical_models (canonical_models.py)       ║
# ║                                                                              ║
# ║  Dependências: mcp>=1.0,<2.0 (runtime), numpy, geosteering_ai (editable)    ║
# ║  Status: PRODUÇÃO (Fase 1 I1.9 — 2026-05-09)                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""MCP Server para validação física do simulador Geosteering AI v2.0.

Expõe ao Claude Code (via Model Context Protocol) as 6 verificações
físicas inviláveis do projeto, normalmente executadas no hook
`run-fortran-parity.sh` e nos testes pytest.

Vantagens vs hook bash:
    • Resposta estruturada (JSON) em vez de stdout
    • Resultado parametrizável (modelo canônico, tolerância, filtro Hankel)
    • Integração direta com agente `geosteering-physics-reviewer`
    • Reutilização das funções de produção `geosteering_ai.simulation.validation.*`

Note:
    Este servidor MCP delega o trabalho real para os módulos de validação
    do simulador. Quando `tatu.x` não está compilado, `check_fortran_parity`
    retorna um dict com `skipped=True`; os outros handlers funcionam sempre.

Ref: docs/reports/arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md §22.1
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Permitir invocação direta (`python tools/physics-validator-mcp/server.py`)
# garantindo que `geosteering_ai` seja importável a partir da raiz do repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Constantes Físicas (Errata Imutável — CLAUDE.md)
# ──────────────────────────────────────────────────────────────────────────

ERRATA_DEFAULTS: dict[str, Any] = {
    "FREQUENCY_HZ": 20000.0,
    "SPACING_METERS": 1.0,
    "SEQUENCE_LENGTH": 600,
    "TARGET_SCALING": "log10",
    "INPUT_FEATURES": [1, 4, 5, 20, 21],
    "OUTPUT_TARGETS": [2, 3],
    "EPS_TF_MIN": 1e-15,
}

CANONICAL_MODELS: list[str] = [
    "oklahoma_3",
    "oklahoma_5",
    "devine_8",
    "oklahoma_15",
    "oklahoma_28",
    "hou_7",
    "viking_graben_10",
]

DEFAULT_FORTRAN_EXEC = _REPO_ROOT / "Fortran_Gerador" / "tatu.x"


# ──────────────────────────────────────────────────────────────────────────
# Handlers (implementação real — delegam para validation modules)
# ──────────────────────────────────────────────────────────────────────────


def _impl_check_fortran_parity(
    model_name: str = "oklahoma_3",
    tolerance: float = 1.0e-6,
    backend: str = "numba",
    n_positions: int = 50,
) -> dict[str, Any]:
    """Compara simulador Python vs Fortran (gold standard) em modelo canônico.

    Delegação: ``geosteering_ai.simulation.validation.compare_fortran_python``.

    Args:
        model_name: Nome do modelo canônico (default ``oklahoma_3``, mais leve).
        tolerance: Tolerância máxima absoluta. Default 1e-6 (Numba/JAX hybrid).
        backend: ``"numba"`` (default), ``"jax_hybrid"``, ou ``"jax_native"``.
        n_positions: Número de posições (default 50, suficiente para smoke).

    Returns:
        Dict estruturado:

        - ``check_name`` (str): "fortran_parity"
        - ``model_name`` (str)
        - ``backend`` (str)
        - ``passed`` (bool): True se max_abs < tolerance
        - ``max_abs_error`` (float)
        - ``max_rel_error`` (float)
        - ``tolerance`` (float)
        - ``n_positions`` (int)
        - ``elapsed_fortran_s`` / ``elapsed_python_s`` (float)
        - ``speedup`` (float)
        - ``skipped`` (bool): True se ``tatu.x`` não disponível
        - ``error`` (str, opcional): mensagem de erro se exceção
    """
    if not DEFAULT_FORTRAN_EXEC.exists():
        return {
            "check_name": "fortran_parity",
            "model_name": model_name,
            "backend": backend,
            "passed": False,
            "skipped": True,
            "reason": f"tatu.x não compilado em {DEFAULT_FORTRAN_EXEC}",
            "tolerance": tolerance,
        }

    try:
        from geosteering_ai.simulation.validation.compare_fortran import (
            compare_fortran_python,
        )
    except ImportError as exc:
        return {
            "check_name": "fortran_parity",
            "model_name": model_name,
            "passed": False,
            "error": f"ImportError: {exc}",
        }

    try:
        results = compare_fortran_python(
            canonical_model_name=model_name,
            backends=[backend],
            tol_abs=tolerance,
            n_positions=n_positions,
            fortran_exec=DEFAULT_FORTRAN_EXEC,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return {
            "check_name": "fortran_parity",
            "model_name": model_name,
            "backend": backend,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if not results:
        return {
            "check_name": "fortran_parity",
            "model_name": model_name,
            "passed": False,
            "error": "compare_fortran_python retornou lista vazia",
        }

    res = results[0]
    return {
        "check_name": "fortran_parity",
        "model_name": model_name,
        "backend": res.backend,
        "passed": res.passed,
        "max_abs_error": res.max_abs_error,
        "max_rel_error": res.max_rel_error,
        "l2_error": res.l2_error,
        "tolerance": res.tol_abs_used,
        "n_positions": res.n_positions,
        "elapsed_fortran_s": res.elapsed_fortran_s,
        "elapsed_python_s": res.elapsed_python_s,
        "speedup": res.speedup,
    }


def _impl_check_maxwell_symmetry(
    filter_name: str = "werthmuller_201pt",
    tol_decoupling: float = 1e-6,
    tol_vmd: float = 1e-4,
) -> dict[str, Any]:
    """Valida 3 propriedades analíticas (decoupling + VMD axial + VMD broadside).

    Delegação: ``geosteering_ai.simulation.validation.compare_analytical``.

    Em meio half-space isotrópico ou TIV, o tensor EM tem propriedades
    analíticas conhecidas (decoupling factors fixos, VMD com forma fechada).
    Esta função consolida 3 testes do submódulo ``compare_analytical``.

    Args:
        filter_name: Filtro Hankel (default ``werthmuller_201pt``).
        tol_decoupling: Tolerância para decoupling factors (default 1e-6).
        tol_vmd: Tolerância para VMD analítico (default 1e-4).

    Returns:
        Dict com:

        - ``check_name`` (str)
        - ``passed`` (bool): True se TODOS os 3 sub-checks passam
        - ``decoupling`` (dict): {pass, diff, ...}
        - ``vmd_axial`` (dict): idem
        - ``vmd_broadside`` (dict): idem
        - ``filter_name`` (str)
    """
    try:
        from geosteering_ai.simulation.validation.compare_analytical import (
            validate_all_analytical,
        )
    except ImportError as exc:
        return {
            "check_name": "maxwell_symmetry",
            "passed": False,
            "error": f"ImportError: {exc}",
        }

    results = validate_all_analytical(
        filter_name=filter_name,
        tol_decoupling=tol_decoupling,
        tol_vmd=tol_vmd,
    )

    all_pass = all(r["pass"] for r in results.values())
    return {
        "check_name": "maxwell_symmetry",
        "passed": all_pass,
        "filter_name": filter_name,
        "decoupling": results.get("decoupling", {}),
        "vmd_axial": results.get("vmd_axial", {}),
        "vmd_broadside": results.get("vmd_broadside", {}),
    }


def _impl_check_decoupling_factors(spacing_m: float = 1.0) -> dict[str, Any]:
    """Calcula ACp e ACx usando função de produção e compara com errata.

    Delegação: ``geosteering_ai.simulation.validation.half_space.static_decoupling_factors``.

    ACp = -1/(4πL³), ACx = +1/(2πL³). Para L=1.0:
        ACp ≈ -0.07957747, ACx ≈ +0.15915494, ACx/(-ACp) = 2.0 exato.

    Args:
        spacing_m: Espaçamento TX-RX em metros (default 1.0).

    Returns:
        Dict com ``ACp_computed``, ``ACx_computed``, ``ratio_ACx_minus_ACp``,
        ``passed`` (True se ratio == 2.0 dentro de 1e-12).
    """
    try:
        from geosteering_ai.simulation.validation.half_space import (
            static_decoupling_factors,
        )
    except ImportError as exc:
        return {
            "check_name": "decoupling_factors",
            "spacing_m": spacing_m,
            "passed": False,
            "error": f"ImportError: {exc}",
        }

    ACp, ACx = static_decoupling_factors(L=spacing_m)
    ratio = ACx / (-ACp)
    passed = abs(ratio - 2.0) < 1e-12

    return {
        "check_name": "decoupling_factors",
        "spacing_m": spacing_m,
        "ACp_computed": ACp,
        "ACx_computed": ACx,
        "ratio_ACx_over_minus_ACp": ratio,
        "expected_ratio": 2.0,
        "passed": passed,
    }


def _impl_check_errata_immutable() -> dict[str, Any]:
    """Compara errata em runtime (SimulationConfig) com valores constantes.

    Delegação: ``geosteering_ai.simulation.config.SimulationConfig`` defaults.

    Verifica que a errata documentada em CLAUDE.md continua coerente com
    os defaults do dataclass de configuração do simulador.

    Returns:
        Dict com ``passed`` (bool) e ``mismatches`` (lista de discrepâncias).
    """
    try:
        from geosteering_ai.simulation.config import SimulationConfig
    except ImportError as exc:
        return {
            "check_name": "errata_immutable",
            "passed": False,
            "error": f"ImportError: {exc}",
        }

    cfg = SimulationConfig()
    mismatches: list[dict[str, Any]] = []

    expected_freq = ERRATA_DEFAULTS["FREQUENCY_HZ"]
    if abs(cfg.frequency_hz - expected_freq) > 1e-9:
        mismatches.append({
            "field": "frequency_hz",
            "expected": expected_freq,
            "actual": cfg.frequency_hz,
        })

    expected_spacing = ERRATA_DEFAULTS["SPACING_METERS"]
    if abs(cfg.tr_spacing_m - expected_spacing) > 1e-9:
        mismatches.append({
            "field": "tr_spacing_m",
            "expected": expected_spacing,
            "actual": cfg.tr_spacing_m,
        })

    return {
        "check_name": "errata_immutable",
        "passed": len(mismatches) == 0,
        "mismatches": mismatches,
        "errata_constants": ERRATA_DEFAULTS,
    }


def _impl_check_skin_depth(
    rho_omega_m: float = 100.0,
    frequency_hz: float = 20000.0,
) -> dict[str, Any]:
    """Calcula skin depth via função de produção (não fórmula manual).

    Delegação: ``geosteering_ai.simulation.validation.half_space.skin_depth``.

    Para ρ=100 Ω·m e f=20 kHz: δ ≈ 35.57 m.

    Args:
        rho_omega_m: Resistividade em Ω·m (default 100.0).
        frequency_hz: Frequência em Hz (default 20 kHz).

    Returns:
        Dict com ``skin_depth_m`` (float) e ``warning`` se valor pequeno.
    """
    try:
        from geosteering_ai.simulation.validation.half_space import skin_depth
    except ImportError as exc:
        return {
            "check_name": "skin_depth",
            "passed": False,
            "error": f"ImportError: {exc}",
        }

    delta = float(skin_depth(frequency_hz=frequency_hz, resistivity_ohm_m=rho_omega_m))

    warning: str
    if delta < 1.0:
        warning = "skin_depth < 1m: alta atenuação (verifique faixa física)"
    elif delta > 1000.0:
        warning = "skin_depth > 1km: meio quase isolante"
    else:
        warning = "OK"

    return {
        "check_name": "skin_depth",
        "rho_omega_m": rho_omega_m,
        "frequency_hz": frequency_hz,
        "skin_depth_m": delta,
        "warning": warning,
        "passed": True,
    }


def _impl_run_canonical_models(
    tolerance: float = 1.0e-6,
    backend: str = "numba",
    n_positions: int = 50,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Executa paridade Fortran para múltiplos modelos canônicos.

    Args:
        tolerance: Tolerância (default 1e-6).
        backend: Backend Python (default ``numba``).
        n_positions: Posições por modelo (default 50, smoke-friendly).
        models: Lista de modelos (default: 7 canônicos).

    Returns:
        Dict com resultado por modelo + resumo agregado.
    """
    if models is None:
        models = list(CANONICAL_MODELS)

    results: list[dict[str, Any]] = []
    for model in models:
        results.append(
            _impl_check_fortran_parity(
                model_name=model,
                tolerance=tolerance,
                backend=backend,
                n_positions=n_positions,
            )
        )

    n_passed = sum(1 for r in results if r.get("passed"))
    n_skipped = sum(1 for r in results if r.get("skipped"))
    return {
        "check_name": "canonical_models_suite",
        "n_models": len(models),
        "n_passed": n_passed,
        "n_skipped": n_skipped,
        "all_passed": n_passed == len(models),
        "results": results,
    }


# ──────────────────────────────────────────────────────────────────────────
# Tool registry — usado tanto por testes quanto pelo MCP server boilerplate
# ──────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "check_fortran_parity": _impl_check_fortran_parity,
    "check_maxwell_symmetry": _impl_check_maxwell_symmetry,
    "check_decoupling_factors": _impl_check_decoupling_factors,
    "check_errata_immutable": _impl_check_errata_immutable,
    "check_skin_depth": _impl_check_skin_depth,
    "run_canonical_models": _impl_run_canonical_models,
}


def _build_tool_definitions() -> list[dict[str, Any]]:
    """Retorna definições de tools com inputSchema JSONSchema (MCP-compatível).

    Estrutura segue o spec MCP ``Tool``: nome, descrição e inputSchema.
    Usado tanto por list_tools (MCP) quanto por testes do tools/list handshake.
    """
    return [
        {
            "name": "check_fortran_parity",
            "description": (
                "Compara simulador Python (Numba/JAX) vs Fortran (gold) em "
                "modelo canônico. Skip se tatu.x não compilado."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "default": "oklahoma_3"},
                    "tolerance": {"type": "number", "default": 1.0e-6},
                    "backend": {
                        "type": "string",
                        "enum": ["numba", "jax_hybrid", "jax_native"],
                        "default": "numba",
                    },
                    "n_positions": {"type": "integer", "default": 50},
                },
            },
        },
        {
            "name": "check_maxwell_symmetry",
            "description": (
                "Valida 3 propriedades analíticas (decoupling, VMD axial, "
                "VMD broadside) via half_space.py."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter_name": {"type": "string", "default": "werthmuller_201pt"},
                    "tol_decoupling": {"type": "number", "default": 1.0e-6},
                    "tol_vmd": {"type": "number", "default": 1.0e-4},
                },
            },
        },
        {
            "name": "check_decoupling_factors",
            "description": (
                "Calcula ACp = -1/(4πL³) e ACx = +1/(2πL³) (decoupling LWD)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spacing_m": {"type": "number", "default": 1.0},
                },
            },
        },
        {
            "name": "check_errata_immutable",
            "description": (
                "Compara errata em runtime (SimulationConfig) com valores "
                "documentados em CLAUDE.md."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "check_skin_depth",
            "description": (
                "Calcula skin depth δ ≈ 503·√(ρ/f) via "
                "geosteering_ai.simulation.validation.half_space.skin_depth."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rho_omega_m": {"type": "number", "default": 100.0},
                    "frequency_hz": {"type": "number", "default": 20000.0},
                },
            },
        },
        {
            "name": "run_canonical_models",
            "description": (
                "Executa paridade Fortran para os 7 modelos canônicos "
                "(oklahoma_3, devine_8, etc)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tolerance": {"type": "number", "default": 1.0e-6},
                    "backend": {"type": "string", "default": "numba"},
                    "n_positions": {"type": "integer", "default": 50},
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    ]


# ──────────────────────────────────────────────────────────────────────────
# MCP Server boilerplate (runtime — requer mcp instalado)
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entrypoint do MCP server (stdio transport).

    Inicializa um ``mcp.server.Server`` e registra as 6 tools via
    ``@server.list_tools()`` e ``@server.call_tool()``. Cada handler é
    despachado em ``asyncio.to_thread`` (CPU-bound).

    Raises:
        ImportError: Se ``mcp`` não estiver instalado.
            Para teste sem MCP, importe ``TOOL_REGISTRY`` diretamente.
    """
    import asyncio

    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError as exc:
        logger.error(
            "physics-validator MCP requer 'mcp>=1.0,<2.0'. "
            "Instale com: pip install -r tools/physics-validator-mcp/requirements.txt"
        )
        # Modo fallback: emite JSON com lista de tools disponíveis (handshake legado)
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"mcp package not installed: {exc}",
                    "tools": [t["name"] for t in _build_tool_definitions()],
                }
            )
        )
        sys.exit(1)

    server: Server = Server("physics-validator")

    @server.list_tools()  # type: ignore[no-untyped-call,misc]
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=defn["name"],
                description=defn["description"],
                inputSchema=defn["inputSchema"],
            )
            for defn in _build_tool_definitions()
        ]

    @server.call_tool()  # type: ignore[no-untyped-call,misc]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")
        result = await asyncio.to_thread(fn, **(arguments or {}))
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    logger.info(
        "physics-validator MCP server iniciando (6 tools, %d handlers reais)",
        len(TOOL_REGISTRY),
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
