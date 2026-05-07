"""Pacote ``geosteering_ai.multi_agent`` — Etapa 1 Quality Mesh + Multi-Agent.

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — Multi-Agent Hardening (§40, §41 documento-base)
═══════════════════════════════════════════════════════════════════════════

Infraestrutura para múltiplos agentes Claude Code rodarem em paralelo sem
corromper o repositório:

- ``LockManager``: lock-file por arquivo, TTL, detecção de stale (TTL OR
  PID morto). API pythônica testável usada pelos hooks bash.
- ``AgentConflictError``: exceção quando lock conflict detectado dentro do TTL.
- ``LockInfo``: dataclass do payload JSON do lock-file.
- ``can_run_together()``: re-export de ``.claude/parallelism_rules`` para
  uso em testes e código Python.

Lock-file format::

    .claude/locks/<sha256(file_path)[:16]>.lock

Conteúdo (JSON)::

    {
      "agent_id": "numba-jit-engineer",
      "acquired_at": "2026-05-07T16:30:00Z",
      "ttl_sec": 300,
      "pid": 12345,
      "file_path": "/abs/path/to/file.py"
    }

═══════════════════════════════════════════════════════════════════════════
"""

from geosteering_ai.multi_agent.conflict_matrix import (
    can_run_together,
    check_limits,
    get_conflicts,
)
from geosteering_ai.multi_agent.lock_manager import (
    AgentConflictError,
    LockInfo,
    LockManager,
)

__all__ = [
    "LockManager",
    "LockInfo",
    "AgentConflictError",
    "can_run_together",
    "check_limits",
    "get_conflicts",
]

__version__ = "1.0.0"
