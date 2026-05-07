"""Testes do sistema de locks multi-agente (Etapa 1).

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — Multi-Agent Lock Tests
═══════════════════════════════════════════════════════════════════════════

5 testes obrigatórios da Etapa 1 (Sprint 1.8):

1. test_lock_acquire_release_basic       — A acquire+release, B acquire OK
2. test_lock_blocks_concurrent_agent     — A acquire, B → AgentConflictError
3. test_stale_lock_auto_cleanup          — TTL expirado é removido
4. test_dead_pid_lock_cleanup            — PID morto detectado (opt-in)
5. test_conflict_matrix_blocks_pair      — can_run_together(numba, validator)=False

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from geosteering_ai.multi_agent import (
    AgentConflictError,
    LockManager,
    can_run_together,
    check_limits,
    get_conflicts,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_lock_manager(tmp_path: Path) -> LockManager:
    """LockManager isolado em tmp_path para evitar contaminação."""
    return LockManager(project_root=tmp_path)


@pytest.fixture
def tmp_critical_file(tmp_path: Path) -> Path:
    """Arquivo dummy crítico para testar locks."""
    f = tmp_path / "fake_kernel.py"
    f.write_text("# fake kernel for testing", encoding="utf-8")
    return f


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Acquire + release básico
# ═══════════════════════════════════════════════════════════════════════════


def test_lock_acquire_release_basic(
    tmp_lock_manager: LockManager, tmp_critical_file: Path
) -> None:
    """A adquire, libera; B adquire mesma path → OK."""
    mgr = tmp_lock_manager

    # A adquire
    assert mgr.acquire(str(tmp_critical_file), "agent-A") is True
    info = mgr.is_locked(str(tmp_critical_file))
    assert info is not None
    assert info.agent_id == "agent-A"

    # A libera
    assert mgr.release(str(tmp_critical_file)) is True
    assert mgr.is_locked(str(tmp_critical_file)) is None

    # B adquire (lock liberado, sem conflict)
    assert mgr.acquire(str(tmp_critical_file), "agent-B") is True
    info_b = mgr.is_locked(str(tmp_critical_file))
    assert info_b is not None
    assert info_b.agent_id == "agent-B"


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Lock bloqueia agente concorrente
# ═══════════════════════════════════════════════════════════════════════════


def test_lock_blocks_concurrent_agent(
    tmp_lock_manager: LockManager, tmp_critical_file: Path
) -> None:
    """A adquire; B tenta adquirir mesma path → AgentConflictError."""
    mgr = tmp_lock_manager

    mgr.acquire(str(tmp_critical_file), "agent-A", ttl=60)

    with pytest.raises(AgentConflictError) as excinfo:
        mgr.acquire(str(tmp_critical_file), "agent-B")

    err = excinfo.value
    assert err.owner == "agent-A"
    assert err.ttl_sec == 60
    assert err.age_sec >= 0

    # A pode re-acquire (mesmo agente)
    assert mgr.acquire(str(tmp_critical_file), "agent-A") is True


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Lock stale (TTL expirado) é auto-removido
# ═══════════════════════════════════════════════════════════════════════════


def test_stale_lock_auto_cleanup(
    tmp_lock_manager: LockManager, tmp_critical_file: Path
) -> None:
    """Lock com TTL expirado é considerado stale e removido em próximo acquire."""
    mgr = tmp_lock_manager

    # A adquire com TTL=1s
    mgr.acquire(str(tmp_critical_file), "agent-A", ttl=1)

    # Forçar idade > TTL fazendo sleep
    time.sleep(1.5)

    info = mgr.is_locked(str(tmp_critical_file))
    assert info is None, "Lock stale deveria ser removido por is_locked()"

    # B adquire sem conflito (lock stale foi removido)
    assert mgr.acquire(str(tmp_critical_file), "agent-B") is True


def test_stale_lock_via_manual_timestamp(
    tmp_lock_manager: LockManager, tmp_critical_file: Path
) -> None:
    """Lock com acquired_at no passado (manipulação direta) é stale."""
    mgr = tmp_lock_manager
    mgr.acquire(str(tmp_critical_file), "agent-A", ttl=60)

    # Manipular timestamp do lock-file para 10 min atrás
    lock_path = mgr._lock_file(str(tmp_critical_file))
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    data["acquired_at"] = (
        (datetime.now(timezone.utc) - timedelta(minutes=10))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    lock_path.write_text(json.dumps(data), encoding="utf-8")

    # Verificar stale + cleanup
    cleaned = mgr.cleanup_stale()
    assert cleaned == 1
    assert mgr.is_locked(str(tmp_critical_file)) is None


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: PID morto detectado (opt-in via is_stale(check_pid=True))
# ═══════════════════════════════════════════════════════════════════════════


def test_dead_pid_check(tmp_lock_manager: LockManager, tmp_critical_file: Path) -> None:
    """Lock cujo PID não existe é detectável via LockInfo.is_pid_dead()."""
    mgr = tmp_lock_manager
    mgr.acquire(str(tmp_critical_file), "agent-A")

    # Manipular PID para um valor improvável (> 4 milhões)
    lock_path = mgr._lock_file(str(tmp_critical_file))
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    data["pid"] = 4_000_001  # PID muito improvável de existir
    lock_path.write_text(json.dumps(data), encoding="utf-8")

    # Re-ler info e verificar PID dead
    from geosteering_ai.multi_agent import LockInfo

    info = LockInfo(**data)
    assert info.is_pid_dead() is True

    # is_stale(check_pid=True) detecta
    assert info.is_stale(check_pid=True) is True

    # is_stale() default (TTL apenas) NÃO detecta (lock recém-criado)
    assert info.is_stale() is False


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Conflict matrix bloqueia pares conflitantes
# ═══════════════════════════════════════════════════════════════════════════


def test_conflict_matrix_blocks_pair() -> None:
    """can_run_together() reflete catálogo CONFLITO de parallelism_rules."""
    # Conflitos diretos
    assert can_run_together("numba-jit-engineer", "numba-validator") is False
    assert can_run_together("numba-validator", "numba-jit-engineer") is False  # simétrico
    assert can_run_together("jax-engineer", "jax-validator") is False
    assert can_run_together("docs-writer", "repo-housekeeper") is False

    # Pares paralelizáveis
    assert can_run_together("numba-jit-engineer", "dl-training-engineer") is True
    assert can_run_together("docs-writer", "physics-validator-mcp") is True

    # Mesmo agente — não duplica
    assert can_run_together("numba-jit-engineer", "numba-jit-engineer") is False

    # get_conflicts retorna lista
    conflicts = get_conflicts("numba-jit-engineer")
    assert "numba-validator" in conflicts
    assert "fortran-parity-validator" in conflicts


def test_check_limits_caps_models() -> None:
    """check_limits() detecta violações de cap por modelo."""
    # Sem violações
    assert check_limits([{"name": "a", "model": "sonnet"}]) == []

    # Excede max_opus_concurrent (cap 1)
    violations = check_limits(
        [
            {"name": "a", "model": "opus"},
            {"name": "b", "model": "opus"},
        ]
    )
    assert len(violations) >= 1
    assert any("Opus" in v for v in violations)

    # Excede max_total_simultaneous (cap 5)
    many = [{"name": f"a{i}", "model": "haiku"} for i in range(6)]
    violations = check_limits(many)
    assert any("Total" in v or "Haiku" in v for v in violations)


# ═══════════════════════════════════════════════════════════════════════════
# Test extra: list_active + release_by_agent
# ═══════════════════════════════════════════════════════════════════════════


def test_release_by_agent_releases_all_locks(
    tmp_lock_manager: LockManager, tmp_path: Path
) -> None:
    """release_by_agent libera TODOS os locks do agente (Stop hook)."""
    mgr = tmp_lock_manager

    f1 = tmp_path / "file1.py"
    f2 = tmp_path / "file2.py"
    f3 = tmp_path / "file3.py"
    for f in (f1, f2, f3):
        f.write_text("", encoding="utf-8")

    mgr.acquire(str(f1), "agent-A")
    mgr.acquire(str(f2), "agent-A")
    mgr.acquire(str(f3), "agent-B")

    assert len(mgr.list_active()) == 3

    count = mgr.release_by_agent("agent-A")
    assert count == 2

    remaining = mgr.list_active()
    assert len(remaining) == 1
    assert remaining[0].agent_id == "agent-B"
