"""File-watcher daemon — Quality Mesh Layer 7 (§41.3 documento-base).

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — File Watcher Daemon
═══════════════════════════════════════════════════════════════════════════

Monitora arquivos críticos e dispara validações fora-de-banda quando
modificações são detectadas FORA do Claude Code (ex.: edição manual, IDE
externa, git operations).

Protege contra:
  - Edits manuais que podem quebrar paridade Fortran
  - Anti-patterns introduzidos via copy/paste manual
  - Modificações em arquivos de errata física (config.py)

Diferenças vs PostToolUse hook:
  - Hook só dispara em Edit/Write do Claude Code
  - Daemon dispara em qualquer modificação no filesystem (incluindo
    IDE externa, git checkout, etc.)

Dependências:
  - ``watchdog`` (já listado em pyproject.toml dev-extras)

Uso (standalone, não auto-start):
  $ python tools/file_watcher_daemon.py

Auto-start (futuro, Etapa 1.5+):
  - macOS: ``~/Library/LaunchAgents/com.geosteering.watcher.plist``
  - Linux: systemd user unit

Logs: ``.claude/watcher.log`` (rotação manual; gitignored)
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Set

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print(
        "[watcher] ERROR: watchdog não instalado. Instale com:\n"
        "    pip install watchdog\n"
        "ou:\n"
        "    pip install -e '.[dev]'",
        file=sys.stderr,
    )
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Configuração
# ═══════════════════════════════════════════════════════════════════════════


PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd())).resolve()
WATCHER_LOG = PROJECT_DIR / ".claude" / "watcher.log"

# Paths críticos a monitorar (relativos ao PROJECT_DIR)
CRITICAL_PATHS = [
    "geosteering_ai/simulation/_numba",
    "geosteering_ai/simulation/_jax",
    "geosteering_ai/simulation/forward.py",
    "geosteering_ai/simulation/multi_forward.py",
    "geosteering_ai/config.py",
    "Fortran_Gerador",
    "CLAUDE.md",
]

# Extensões a observar (filtro fino dentro dos paths)
WATCHED_EXTENSIONS = {".py", ".f08", ".f90", ".md", ".yaml", ".yml", ".toml"}

# Cooldown entre validações para evitar spam (segundos)
VALIDATION_COOLDOWN_SEC = 30


# ═══════════════════════════════════════════════════════════════════════════
# Setup logging
# ═══════════════════════════════════════════════════════════════════════════


def setup_logger() -> logging.Logger:
    WATCHER_LOG.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("geosteering.watcher")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter("[watcher] %(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(console)

    # File rotativo (max 10MB, 3 backups)
    file_h = RotatingFileHandler(
        WATCHER_LOG, maxBytes=10 * 1024 * 1024, backupCount=3
    )
    file_h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_h)

    return logger


# ═══════════════════════════════════════════════════════════════════════════
# Event Handler
# ═══════════════════════════════════════════════════════════════════════════


class GeosteeringWatcher(FileSystemEventHandler):
    """Handler de eventos para arquivos críticos."""

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.logger = logger
        self._last_validation: dict[str, float] = {}

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if not self._is_watched(path):
            return
        self._maybe_validate(path)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if not self._is_watched(path):
            return
        self.logger.info(f"created: {path.relative_to(PROJECT_DIR)}")

    def _is_watched(self, path: Path) -> bool:
        if path.suffix not in WATCHED_EXTENSIONS:
            return False
        # Path tem que estar dentro de algum CRITICAL_PATHS
        try:
            rel = path.resolve().relative_to(PROJECT_DIR)
        except ValueError:
            return False
        rel_str = str(rel)
        for crit in CRITICAL_PATHS:
            if rel_str == crit or rel_str.startswith(f"{crit}/") or rel_str.startswith(f"{crit}\\"):
                return True
        return False

    def _maybe_validate(self, path: Path) -> None:
        """Dispara validação respeitando cooldown."""
        rel_str = str(path.resolve().relative_to(PROJECT_DIR))
        now = time.time()
        last = self._last_validation.get(rel_str, 0.0)
        if now - last < VALIDATION_COOLDOWN_SEC:
            return
        self._last_validation[rel_str] = now

        self.logger.info(f"modified: {rel_str} (validação acionada)")
        # Validações async ficam para Etapa 1.5+ (apenas log neste ponto):
        #   - se path em _numba/ ou forward.py: rodar paridade Fortran
        #   - se path em config.py: rodar validate-physics
        #   - sempre: rodar check-anti-patterns precommit


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def collect_watch_dirs() -> Set[Path]:
    """Resolve CRITICAL_PATHS para diretórios efetivos a observar."""
    dirs: Set[Path] = set()
    for crit in CRITICAL_PATHS:
        full = PROJECT_DIR / crit
        if not full.exists():
            continue
        if full.is_dir():
            dirs.add(full)
        elif full.is_file():
            dirs.add(full.parent)
    return dirs


def main(argv: List[str] | None = None) -> int:
    logger = setup_logger()
    watch_dirs = collect_watch_dirs()

    if not watch_dirs:
        logger.error(f"Nenhum path crítico encontrado em {PROJECT_DIR}")
        return 1

    logger.info(f"daemon iniciando — projeto: {PROJECT_DIR}")
    logger.info(f"observando {len(watch_dirs)} diretórios:")
    for d in sorted(watch_dirs):
        logger.info(f"  {d.relative_to(PROJECT_DIR)}")

    handler = GeosteeringWatcher(logger)
    observer = Observer()
    for d in watch_dirs:
        observer.schedule(handler, str(d), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("daemon interrompido (SIGINT)")
        observer.stop()
    observer.join()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
