#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: release-lock.sh                                             ║
# ║  Eventos:                                                          ║
# ║    PostToolUse (Edit|Write)  — libera lock do arquivo específico   ║
# ║    Stop                       — libera TODOS os locks do agente    ║
# ║                                                                    ║
# ║  Modo:                                                             ║
# ║    sem args    — lê file_path do JSON stdin (PostToolUse)          ║
# ║    --all       — libera todos os locks de CLAUDE_AGENT_ID (Stop)   ║
# ║                                                                    ║
# ║  Sempre exit 0 (libertar lock não deve falhar a operação).         ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
AGENT_ID="${CLAUDE_AGENT_ID:-orchestrator}"

if [ -x "$HOME/Geosteering_AI_venv/bin/python" ]; then
    PYTHON="$HOME/Geosteering_AI_venv/bin/python"
else
    PYTHON="python3"
fi

cd "$PROJECT_DIR"

# Modo --all: libera todos os locks do agente
if [ "${1:-}" = "--all" ]; then
    "$PYTHON" -m geosteering_ai.multi_agent.lock_manager release --all 2>/dev/null || true
    exit 0
fi

# Modo PostToolUse: lê JSON do stdin
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
[ -z "$FILE_PATH" ] && exit 0

# Filtro: só desbloqueia se foi arquivo crítico (mesmo glob do acquire)
case "$FILE_PATH" in
    *_numba/* | *_jax/* | *forward.py | *multi_forward.py | *config.py | *kernel.py)
        "$PYTHON" -m geosteering_ai.multi_agent.lock_manager release "$FILE_PATH" 2>/dev/null || true
        ;;
esac

exit 0
