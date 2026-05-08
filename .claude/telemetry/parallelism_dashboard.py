"""Dashboard CLI de paralelismo de agentes Claude Code (Etapa 1).

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — Parallelism Dashboard (§40.7 documento-base)
═══════════════════════════════════════════════════════════════════════════

CLI que mostra status atual de agentes em paralelo, locks ativos,
e telemetria de conflitos.

Lê:
- ``.claude/state.json`` (registro de agentes ativos, mantido por orquestrador)
- ``.claude/locks/*.lock`` (locks ativos, mantidos por LockManager)

Mantém histórico em:
- ``.claude/telemetry/history.csv`` (timestamp, agents, locks, conflicts)

CLI
───
::

    python -m claude.telemetry.parallelism_dashboard       # snapshot único
    python -m claude.telemetry.parallelism_dashboard --watch [--interval 5]

Exemplo de output:
::

    ╔══════════════════════════════════════════════════════════════════╗
    ║  GEOSTEERING AI — PARALLELISM DASHBOARD                         ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Agente                    Modelo    Idade    Locks  Status     ║
    ║  ─────────────────────────────────────────────────────────────── ║
    ║  numba-jit-engineer        Sonnet    14m32s    2     RUN         ║
    ║  fortran-parity-validator  Haiku      0m34s    0     RUN         ║
    ║  ─────────────────────────────────────────────────────────────── ║
    ║  TOTAL: 2 agentes, 2 locks ativos                               ║
    ╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _project_dir() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    return Path(env).resolve() if env else Path.cwd().resolve()


def _read_state(project_dir: Path) -> Dict[str, Any]:
    """Carrega .claude/state.json. Retorna estrutura vazia se ausente."""
    state_file = project_dir / ".claude" / "state.json"
    if not state_file.exists():
        # Tentar template
        tmpl = project_dir / ".claude" / "state.json.template"
        if tmpl.exists():
            try:
                return json.loads(tmpl.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"agents": [], "telemetry": {}}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"agents": [], "telemetry": {}}


def _read_locks(project_dir: Path) -> List[Dict[str, Any]]:
    """Lista todos os locks ativos lendo arquivos .lock."""
    lock_dir = project_dir / ".claude" / "locks"
    if not lock_dir.exists():
        return []
    locks = []
    for lf in lock_dir.glob("*.lock"):
        try:
            data = json.loads(lf.read_text(encoding="utf-8"))
            locks.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return locks


def _format_age(iso_timestamp: str) -> str:
    """Formata idade desde ISO-8601 timestamp como '14m32s'."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return "—"
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"


def render_snapshot(project_dir: Optional[Path] = None) -> str:
    """Renderiza snapshot ASCII do dashboard."""
    pd = project_dir or _project_dir()
    state = _read_state(pd)
    locks = _read_locks(pd)
    agents = state.get("agents", [])

    # Mapear locks por agente
    locks_by_agent: Dict[str, int] = {}
    for lk in locks:
        aid = lk.get("agent_id", "?")
        locks_by_agent[aid] = locks_by_agent.get(aid, 0) + 1

    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════════╗")
    lines.append("║  GEOSTEERING AI — PARALLELISM DASHBOARD                         ║")
    lines.append("╠══════════════════════════════════════════════════════════════════╣")
    lines.append("║  Agente                    Modelo    Idade    Locks  Status     ║")
    lines.append("║  ─────────────────────────────────────────────────────────────── ║")

    if not agents:
        lines.append("║  (nenhum agente ativo registrado em state.json)                  ║")
    else:
        for ag in agents:
            name = (ag.get("name") or "?")[:24].ljust(24)
            model = (ag.get("model") or "?")[:8].ljust(8)
            age = _format_age(ag.get("started_at", "")).ljust(7)
            locks_count = str(locks_by_agent.get(ag.get("name", ""), 0)).ljust(5)
            status = (ag.get("status") or "?")[:8].ljust(8)
            lines.append(f"║  {name}  {model}  {age}  {locks_count}  {status} ║")

    lines.append("║  ─────────────────────────────────────────────────────────────── ║")
    total_msg = f"TOTAL: {len(agents)} agentes, {len(locks)} locks ativos"
    lines.append(f"║  {total_msg.ljust(64)}║")

    telem = state.get("telemetry", {})
    if telem:
        conflicts = telem.get("total_conflicts_avoided", 0)
        spawned = telem.get("agents_spawned", 0)
        lines.append(
            f"║  Sessão: {spawned} spawned, {conflicts} conflicts avoided".ljust(67)
            + "║"
        )

    lines.append("╚══════════════════════════════════════════════════════════════════╝")

    return "\n".join(lines)


def append_history(project_dir: Optional[Path] = None) -> None:
    """Appenda linha CSV em .claude/telemetry/history.csv (snapshot atual)."""
    pd = project_dir or _project_dir()
    csv_file = pd / ".claude" / "telemetry" / "history.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    state = _read_state(pd)
    locks = _read_locks(pd)
    telem = state.get("telemetry", {})

    is_new = not csv_file.exists()
    with csv_file.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(
                ["timestamp", "agents_count", "locks_count", "conflicts_avoided"]
            )
        w.writerow(
            [
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                len(state.get("agents", [])),
                len(locks),
                telem.get("total_conflicts_avoided", 0),
            ]
        )


def main(argv: Optional[List[str]] = None) -> int:
    """Entry-point CLI."""
    parser = argparse.ArgumentParser(
        prog="parallelism-dashboard",
        description="Dashboard de paralelismo de agentes Claude Code",
    )
    parser.add_argument(
        "--watch", action="store_true", help="Modo watch (atualiza periodicamente)"
    )
    parser.add_argument(
        "--interval", type=int, default=5, help="Intervalo em segundos (default 5)"
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Não appendar em history.csv",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Override de CLAUDE_PROJECT_DIR",
    )
    args = parser.parse_args(argv)

    pd = (args.project_dir or _project_dir()).resolve()

    def render_once() -> None:
        print(render_snapshot(pd))
        if not args.no_history:
            try:
                append_history(pd)
            except OSError as exc:
                print(f"[telemetry] aviso: falha ao gravar history.csv: {exc}",
                      file=sys.stderr)

    if not args.watch:
        render_once()
        return 0

    try:
        while True:
            # Limpa terminal entre frames
            print("\033[H\033[J", end="")
            render_once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[dashboard] interrompido", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
