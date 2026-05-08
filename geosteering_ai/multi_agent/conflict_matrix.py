"""Re-export de ``.claude/parallelism_rules.py`` como módulo Python testável.

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — Conflict Matrix Bridge
═══════════════════════════════════════════════════════════════════════════

A configuração de conflitos vive em ``.claude/parallelism_rules.py`` (fora
de ``geosteering_ai/``) por se tratar de config do harness Claude Code.
Este módulo carrega-a dinamicamente via ``importlib.util`` para uso em
testes e código Python.

Razão para o split:
  - ``.claude/`` é a fronteira do harness: hooks, skills, settings.
  - ``geosteering_ai/`` é a biblioteca instalável (pip).
  - Multi-agent rules pertencem ao harness, mas testes pertencem à
    biblioteca → bridge necessária.

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List


def _load_rules_module() -> Any:
    """Carrega ``.claude/parallelism_rules.py`` como módulo dinâmico.

    Returns:
        Módulo Python com atributos ``CONFLITO``, ``PODEM_PARALELIZAR``,
        ``LIMITES``, ``can_run_together``, ``check_limits``, ``get_conflicts``.

    Raises:
        FileNotFoundError: se ``parallelism_rules.py`` não existir.
        ImportError: se módulo falhar ao carregar.
    """
    env_root = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_root:
        candidates = [Path(env_root) / ".claude" / "parallelism_rules.py"]
    else:
        # Fallback: procurar em pais de __file__
        here = Path(__file__).resolve()
        candidates = [
            here.parent.parent.parent / ".claude" / "parallelism_rules.py",
            Path.cwd() / ".claude" / "parallelism_rules.py",
        ]

    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "_claude_parallelism_rules", str(path)
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["_claude_parallelism_rules"] = module
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        f"parallelism_rules.py não encontrado. Tentativas: {candidates}"
    )


# Lazy load para evitar erros em imports circulares ou ambientes sem .claude/
# Lock protege inicialização em ambientes multi-thread (watchdog, daemons).
# Double-checked locking: check externo sem lock (fast path) + check interno
# com lock (slow path) — seguro porque atribuição a referência é atômica no CPython
# e no Python 3.13 nogil (PEP 703) via referência counted.
_RULES_MODULE: Any = None
_RULES_LOCK: threading.Lock = threading.Lock()


def _rules() -> Any:
    """Retorna módulo cacheado (lazy load, thread-safe)."""
    global _RULES_MODULE
    if _RULES_MODULE is None:
        with _RULES_LOCK:
            if _RULES_MODULE is None:  # double-checked locking
                _RULES_MODULE = _load_rules_module()
    return _RULES_MODULE


def can_run_together(agent_a: str, agent_b: str) -> bool:
    """Re-export de ``parallelism_rules.can_run_together``."""
    return bool(_rules().can_run_together(agent_a, agent_b))


def get_conflicts(agent: str) -> List[str]:
    """Re-export de ``parallelism_rules.get_conflicts``."""
    return list(_rules().get_conflicts(agent))


def check_limits(active_agents: List[Dict[str, str]]) -> List[str]:
    """Re-export de ``parallelism_rules.check_limits``."""
    return list(_rules().check_limits(active_agents))


__all__ = [
    "can_run_together",
    "get_conflicts",
    "check_limits",
]
