"""Lock manager para edits concorrentes de múltiplos agentes Claude Code.

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — Multi-Agent LockManager (§40.4–40.5 documento-base)
═══════════════════════════════════════════════════════════════════════════

Implementa lock advisory com TTL e detecção de stale via PID check. Cada
arquivo crítico (``_numba/*``, ``forward.py``, etc.) recebe lock individual
quando um agente vai editá-lo, prevenindo race conditions onde dois agentes
modificam o mesmo arquivo simultaneamente.

Lock-file format
────────────────
Path: ``<project>/.claude/locks/<sha256(file_path)[:16]>.lock``
Conteúdo: JSON com ``agent_id``, ``acquired_at``, ``ttl_sec``, ``pid``,
``file_path``.

Stale detection (2 critérios independentes)
────────────────────────────────────────────
1. **TTL expirado**: ``(now - acquired_at) > ttl_sec`` (default 300s).
2. **PID morto**: ``os.kill(pid, 0)`` lança ``ProcessLookupError`` ou
   ``OSError(EPERM)`` (processo não existe ou pertence a outro user).

Quando lock estale, é removido silenciosamente em próximo ``acquire()``.

API
───
- ``acquire(file_path, agent_id, ttl=300) -> bool``
- ``release(file_path)``
- ``is_locked(file_path) -> Optional[LockInfo]``
- ``cleanup_stale() -> int`` (manutenção)

CLI (chamado pelos hooks bash)
──────────────────────────────
::

    python -m geosteering_ai.multi_agent.lock_manager acquire <path> [agent_id]
    python -m geosteering_ai.multi_agent.lock_manager release <path>
    python -m geosteering_ai.multi_agent.lock_manager status [path]
    python -m geosteering_ai.multi_agent.lock_manager cleanup

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_TTL_SEC = 300  # 5 minutos
DEFAULT_LOCK_DIR_NAME = ".claude/locks"


# ═══════════════════════════════════════════════════════════════════════════
# Exceções e dataclasses
# ═══════════════════════════════════════════════════════════════════════════


class AgentConflictError(Exception):
    """Erro quando lock conflict detectado dentro do TTL.

    Atributos:
        file_path: caminho do arquivo que outro agente está editando.
        owner: ``agent_id`` do dono do lock atual.
        age_sec: idade do lock em segundos.
        ttl_sec: TTL configurado do lock.
    """

    def __init__(self, file_path: str, owner: str, age_sec: int, ttl_sec: int) -> None:
        self.file_path = file_path
        self.owner = owner
        self.age_sec = age_sec
        self.ttl_sec = ttl_sec
        super().__init__(
            f"Lock conflict: {file_path} held by '{owner}' "
            f"({age_sec}s old, ttl={ttl_sec}s)"
        )


@dataclass(frozen=True)
class LockInfo:
    """Payload imutável do lock-file."""

    agent_id: str
    acquired_at: str  # ISO-8601 UTC
    ttl_sec: int
    pid: int
    file_path: str

    def age_seconds(self) -> int:
        """Idade do lock em segundos (now - acquired_at)."""
        acquired_dt = datetime.fromisoformat(self.acquired_at.replace("Z", "+00:00"))
        return int((datetime.now(timezone.utc) - acquired_dt).total_seconds())

    def is_expired(self) -> bool:
        """``True`` se o lock excedeu o TTL."""
        return self.age_seconds() > self.ttl_sec

    def is_pid_dead(self) -> bool:
        """``True`` se o PID que criou o lock não existe mais."""
        try:
            # signal 0 não envia sinal, apenas checa existência
            os.kill(self.pid, 0)
            return False
        except ProcessLookupError:
            return True
        except OSError as exc:
            # EPERM = processo existe mas pertence a outro user (raro mas válido)
            if exc.errno == errno.EPERM:
                return False
            return True

    def is_stale(self, check_pid: bool = False) -> bool:
        """``True`` se TTL expirou (ou PID morto, se ``check_pid=True``).

        Para CLI uso ephemeral (``acquire`` em subshells distintos), o PID
        do criador do lock geralmente já morreu antes do próximo acquire,
        levando a falsos positivos. TTL é o critério primário de segurança.

        ``check_pid=True`` é útil em código de longa duração (ex.: daemon)
        onde a Python process que criou o lock continua viva.
        """
        if self.is_expired():
            return True
        if check_pid and self.is_pid_dead():
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# LockManager
# ═══════════════════════════════════════════════════════════════════════════


class LockManager:
    """Gerencia locks por arquivo com TTL e detecção de stale.

    Args:
        project_root: raiz do projeto. Default lê ``CLAUDE_PROJECT_DIR`` env
            ou auto-detecta via ``Path.cwd()``.
        lock_dir: diretório dos lock-files (relativo ao project_root).
            Default ``.claude/locks``.

    Example:
        >>> mgr = LockManager()
        >>> mgr.acquire("/path/to/file.py", "agent-A")
        True
        >>> mgr.acquire("/path/to/file.py", "agent-B")  # raises
        Traceback (most recent call last):
            ...
        AgentConflictError: Lock conflict: ...
        >>> mgr.release("/path/to/file.py")
        >>> mgr.is_locked("/path/to/file.py") is None
        True
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        lock_dir: str = DEFAULT_LOCK_DIR_NAME,
    ) -> None:
        if project_root is None:
            env_root = os.environ.get("CLAUDE_PROJECT_DIR")
            project_root = Path(env_root) if env_root else Path.cwd()
        self.project_root = Path(project_root).resolve()
        self.lock_dir = self.project_root / lock_dir
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers internos ────────────────────────────────────────────────────

    @staticmethod
    def _hash_path(file_path: str) -> str:
        """SHA-256 truncado (16 chars) do path absoluto — chave do lock-file."""
        abs_path = str(Path(file_path).resolve())
        return hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:16]

    def _lock_file(self, file_path: str) -> Path:
        return self.lock_dir / f"{self._hash_path(file_path)}.lock"

    def _read_lock(self, lock_file: Path) -> Optional[LockInfo]:
        try:
            data = json.loads(lock_file.read_text(encoding="utf-8"))
            info = LockInfo(**data)
            # Smoke-test acquired_at parseável; corrupção retorna None
            info.age_seconds()
            return info
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError, OSError):
            return None

    # ── API pública ─────────────────────────────────────────────────────────

    def acquire(
        self,
        file_path: str,
        agent_id: str,
        ttl: int = DEFAULT_TTL_SEC,
        force: bool = False,
    ) -> bool:
        """Adquire lock para ``file_path`` em nome de ``agent_id``.

        Args:
            file_path: caminho do arquivo a ser locked.
            agent_id: identificador do agente requisitante.
            ttl: TTL do lock em segundos (default 300s).
            force: se ``True``, sobrescreve lock existente sem checar conflict.

        Returns:
            ``True`` se lock adquirido com sucesso.

        Raises:
            AgentConflictError: se já existe lock fresco (não-stale) com
                outro ``agent_id``.
        """
        lock_file = self._lock_file(file_path)
        info = LockInfo(
            agent_id=agent_id,
            acquired_at=datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            ttl_sec=ttl,
            pid=os.getpid(),
            file_path=str(Path(file_path).resolve()),
        )
        payload = json.dumps(asdict(info), indent=2)

        if force:
            # Sobrescreve sem checar conflict (uso interno apenas)
            lock_file.write_text(payload, encoding="utf-8")
            return True

        # ── Tentativa atômica via O_CREAT|O_EXCL (POSIX) ─────────────────────
        # Este syscall falha (FileExistsError) se o lock-file já existe,
        # eliminando a race TOCTOU entre check e write.
        try:
            fd = os.open(
                str(lock_file),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            return True
        except FileExistsError:
            pass  # Outro processo criou primeiro — investiga abaixo

        # ── Lock-file existe; verificar se é stale ou conflict ───────────────
        existing = self._read_lock(lock_file)
        if existing is None:
            # JSON inválido ou file deletado mid-flight: trata como stale
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass
            return self.acquire(file_path, agent_id, ttl=ttl, force=False)

        if existing.is_stale():
            # Stale: remove e tenta novamente (recursão limitada por O_EXCL)
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass
            return self.acquire(file_path, agent_id, ttl=ttl, force=False)

        if existing.agent_id != agent_id:
            raise AgentConflictError(
                file_path=file_path,
                owner=existing.agent_id,
                age_sec=existing.age_seconds(),
                ttl_sec=existing.ttl_sec,
            )

        # Same agent re-acquiring — sobrescreve atomicamente via temp + rename
        # (rename é atômico em POSIX dentro do mesmo filesystem)
        tmp = lock_file.with_suffix(f".lock.{os.getpid()}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(lock_file)  # Path.replace = os.replace = atomic rename
        return True

    def release(self, file_path: str) -> bool:
        """Libera lock de ``file_path``. Idempotente.

        Returns:
            ``True`` se lock existia e foi removido; ``False`` se não havia lock.
        """
        lock_file = self._lock_file(file_path)
        if lock_file.exists():
            lock_file.unlink()
            return True
        return False

    def release_by_agent(self, agent_id: str) -> int:
        """Libera TODOS os locks de ``agent_id``. Útil em Stop hook."""
        count = 0
        for lf in self.lock_dir.glob("*.lock"):
            info = self._read_lock(lf)
            if info is not None and info.agent_id == agent_id:
                lf.unlink()
                count += 1
        return count

    def is_locked(self, file_path: str) -> Optional[LockInfo]:
        """Retorna ``LockInfo`` se locked (e fresh); ``None`` se livre/stale.

        Locks stale são automaticamente removidos.
        """
        lock_file = self._lock_file(file_path)
        if not lock_file.exists():
            return None
        info = self._read_lock(lock_file)
        if info is None or info.is_stale():
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass
            return None
        return info

    def cleanup_stale(self) -> int:
        """Remove todos os locks stale (TTL expirado ou PID morto).

        Returns:
            Número de locks removidos.
        """
        count = 0
        for lf in list(self.lock_dir.glob("*.lock")):
            info = self._read_lock(lf)
            if info is None or info.is_stale():
                try:
                    lf.unlink()
                    count += 1
                except FileNotFoundError:
                    pass
        return count

    def list_active(self) -> list[LockInfo]:
        """Lista todos os locks atualmente válidos (não-stale)."""
        active = []
        for lf in self.lock_dir.glob("*.lock"):
            info = self._read_lock(lf)
            if info is not None and not info.is_stale():
                active.append(info)
        return active


# ═══════════════════════════════════════════════════════════════════════════
# CLI — chamado pelos hooks bash
# ═══════════════════════════════════════════════════════════════════════════


def _cli_main(argv: Optional[list[str]] = None) -> int:
    """Entry-point CLI usado pelos hooks acquire-lock.sh / release-lock.sh."""
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2

    cmd = args[0]
    mgr = LockManager()

    try:
        if cmd == "acquire":
            if len(args) < 2:
                print("Usage: acquire <file_path> [agent_id] [ttl]", file=sys.stderr)
                return 2
            path = args[1]
            agent_id = (
                args[2]
                if len(args) > 2
                else os.environ.get("CLAUDE_AGENT_ID", "default-agent")
            )
            ttl = int(args[3]) if len(args) > 3 else DEFAULT_TTL_SEC
            try:
                mgr.acquire(path, agent_id, ttl=ttl)
                print(f"[lock] ✓ acquired by '{agent_id}': {path}", file=sys.stderr)
                return 0
            except AgentConflictError as e:
                print(f"[lock] ✗ {e}", file=sys.stderr)
                return 1

        elif cmd == "release":
            if len(args) < 2:
                print("Usage: release <file_path|--all>", file=sys.stderr)
                return 2
            target = args[1]
            if target == "--all":
                agent_id = os.environ.get("CLAUDE_AGENT_ID", "default-agent")
                count = mgr.release_by_agent(agent_id)
                print(f"[lock] released {count} locks for '{agent_id}'", file=sys.stderr)
            else:
                released = mgr.release(target)
                if released:
                    print(f"[lock] released: {target}", file=sys.stderr)
            return 0

        elif cmd == "status":
            if len(args) >= 2:
                info = mgr.is_locked(args[1])
                if info is None:
                    print(f"[lock] FREE: {args[1]}")
                    return 0
                print(
                    f"[lock] LOCKED: {args[1]} by '{info.agent_id}' "
                    f"(age={info.age_seconds()}s, ttl={info.ttl_sec}s, pid={info.pid})"
                )
                return 0
            else:
                active = mgr.list_active()
                print(f"[lock] {len(active)} active locks:")
                for info in active:
                    print(f"  {info.file_path}")
                    print(
                        f"    agent={info.agent_id} pid={info.pid} "
                        f"age={info.age_seconds()}s/{info.ttl_sec}s"
                    )
                return 0

        elif cmd == "cleanup":
            count = mgr.cleanup_stale()
            print(f"[lock] cleaned {count} stale locks", file=sys.stderr)
            return 0

        else:
            print(f"Unknown command: {cmd}", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[lock] ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(_cli_main())
