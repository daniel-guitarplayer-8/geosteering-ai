#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: acquire-lock.sh                                             ║
# ║  Evento: PreToolUse (Edit|Write)                                   ║
# ║  Projeto: Geosteering AI v2.0 — Etapa 1 Multi-Agent                ║
# ║                                                                    ║
# ║  Adquire lock no arquivo crítico antes de Edit/Write.              ║
# ║  Bloqueia (exit 1) se outro agente já possui lock fresco (<5min).  ║
# ║                                                                    ║
# ║  Filtro: apenas arquivos críticos (_numba/, forward.py, etc.).     ║
# ║  Edits em tests/, docs/, *.md são ignorados (não locked).          ║
# ║                                                                    ║
# ║  Lock-files vivem em .claude/locks/, gerenciados pelo LockManager. ║
# ║                                                                    ║
# ║  Env vars:                                                         ║
# ║    CLAUDE_AGENT_ID  — identificador do agente (default: orchestrator) ║
# ║    CLAUDE_PROJECT_DIR — raiz do projeto                            ║
# ║    LOCK_TTL_SEC     — TTL custom (default 300s)                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
[ -z "$FILE_PATH" ] && exit 0

# Filtro: apenas arquivos críticos ao kernel JIT/forward
case "$FILE_PATH" in
    *_numba/* | *_jax/* | *forward.py | *multi_forward.py | *config.py | *kernel.py)
        ;;
    *)
        exit 0  # arquivos não críticos não precisam de lock
        ;;
esac

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
AGENT_ID="${CLAUDE_AGENT_ID:-orchestrator}"
TTL="${LOCK_TTL_SEC:-300}"

# Resolver Python (preferir venv)
if [ -x "$HOME/Geosteering_AI_venv/bin/python" ]; then
    PYTHON="$HOME/Geosteering_AI_venv/bin/python"
else
    PYTHON="python3"
fi

cd "$PROJECT_DIR"

# Tenta adquirir lock via CLI do LockManager
if "$PYTHON" -m geosteering_ai.multi_agent.lock_manager acquire \
        "$FILE_PATH" "$AGENT_ID" "$TTL"; then
    exit 0
else
    EXIT_CODE=$?
    echo "" >&2
    echo "============================================================" >&2
    echo "[acquire-lock] Edit BLOQUEADO em: $FILE_PATH" >&2
    echo "  Agente requisitante: $AGENT_ID" >&2
    echo "  Outro agente possui lock fresco." >&2
    echo "  Tente novamente após o lock expirar (TTL=${TTL}s)" >&2
    echo "  ou use 'python -m geosteering_ai.multi_agent.lock_manager status' para ver donos." >&2
    echo "============================================================" >&2
    exit $EXIT_CODE
fi
