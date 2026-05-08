#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: run-fortran-parity.sh                                       ║
# ║  Evento: PostToolUse (Edit|Write) / Pre-commit (via .pre-commit)   ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Modo QUICK (padrao PostToolUse): apenas oklahoma_3 — ~8s         ║
# ║  Modo FULL  (pre-commit):         7 modelos canonicos — ~146s     ║
# ║                                                                    ║
# ║  Controle via variavel de ambiente:                                ║
# ║    FORTRAN_PARITY_MODE=quick   → oklahoma_3 apenas (default)      ║
# ║    FORTRAN_PARITY_MODE=full    → suite completa                    ║
# ║    CLAUDE_BYPASS_FORTRAN_PARITY=1 → desabilita completamente      ║
# ║                                                                    ║
# ║  Exit codes:                                                       ║
# ║    0 — paridade preservada (ou bypass ativo)                       ║
# ║    1 — FALHA: paridade Fortran quebrada                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

# Bypass global
[ "${CLAUDE_BYPASS_FORTRAN_PARITY:-0}" = "1" ] && exit 0

MODE="${FORTRAN_PARITY_MODE:-quick}"

# Modo FULL (pre-commit): nao le stdin — roda diretamente sem filtro de path
if [ "$MODE" = "full" ]; then
    FILE_PATH="(pre-commit: arquivos criticos)"
else
    # Modo PostToolUse: le file_path do JSON stdin injetado pelo Claude Code
    INPUT=$(cat)
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
    [ -z "$FILE_PATH" ] && exit 0

    # Filtro: apenas arquivos criticos ao kernel JIT
    case "$FILE_PATH" in
        *_numba/* | *forward.py | *multi_forward.py)
            ;;
        *)
            exit 0
            ;;
    esac
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$PROJECT_DIR"

# Ativar venv (silencioso se nao existir)
if [ -f ~/Geosteering_AI_venv/bin/activate ]; then
    # shellcheck disable=SC1090
    source ~/Geosteering_AI_venv/bin/activate
fi

# PYTHONPATH necessario para imports relativos (tests/_fortran_helpers)
# pyproject.toml define pythonpath=["."] mas exportar garante compatibilidade
# quando pytest e invocado fora do rootdir.
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Selecionar filtro e descricao do modo
if [ "$MODE" = "full" ]; then
    PYTEST_FILTER="fortran_python_numba"
    MODE_DESC="suite completa (7 modelos)"
else
    PYTEST_FILTER="fortran_python_numba and oklahoma_3"
    MODE_DESC="quick (oklahoma_3)"
fi

echo "[fortran-parity] verificando paridade <1e-12 — modo ${MODE_DESC}..." >&2

if pytest tests/test_simulation_compare_fortran.py \
         -k "$PYTEST_FILTER" \
         --tb=short \
         -q 2>&1 | tail -10 >&2; then
    echo "[fortran-parity] OK — paridade preservada (<1e-12) [${MODE_DESC}]" >&2
    exit 0
else
    echo "" >&2
    echo "============================================================" >&2
    echo "[fortran-parity] FALHA — PARIDADE FORTRAN QUEBRADA!" >&2
    echo "============================================================" >&2
    echo "Arquivo modificado: $FILE_PATH" >&2
    echo "Modo: ${MODE_DESC}" >&2
    echo "Restaure com: .claude/recovery.sh $FILE_PATH latest" >&2
    echo "ou: git checkout HEAD -- $FILE_PATH" >&2
    echo "============================================================" >&2
    exit 1
fi
