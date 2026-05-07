#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: check-anti-patterns-precommit.sh                            ║
# ║  Framework: pre-commit (passa paths via argv)                      ║
# ║                                                                    ║
# ║  Versão pre-commit do check-anti-patterns: lê arquivos do disco    ║
# ║  (não do JSON Claude Code) e itera contra .claude/anti-patterns.   ║
# ║  txt. Diferenças vs PreToolUse:                                    ║
# ║    - Recebe paths como $1, $2, ... (não JSON via stdin)            ║
# ║    - Lê arquivo do disco ao invés de tool_input.new_string         ║
# ║    - Saída em formato amigável para git output                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
PATTERNS_FILE="$PROJECT_DIR/.claude/anti-patterns.txt"
[ ! -f "$PATTERNS_FILE" ] && exit 0
[ $# -eq 0 ] && exit 0

EXIT_CODE=0
TOTAL_BLOCKS=0
TOTAL_WARNS=0

for FILE_PATH in "$@"; do
    [ ! -f "$FILE_PATH" ] && continue

    while IFS=$'\t' read -r kb_id pattern severity path_glob; do
        [ -z "${kb_id:-}" ] && continue
        case "$kb_id" in '#'*) continue ;; esac

        # Compatibilidade com formato 3 colunas (legacy)
        if [ -z "${path_glob:-}" ]; then
            path_glob="$severity"
            severity="BLOCK"
        fi

        # shellcheck disable=SC2053
        [[ "$FILE_PATH" == $path_glob ]] || continue

        if grep -qE "$pattern" "$FILE_PATH"; then
            case "$severity" in
                BLOCK)
                    echo "[BLOCK] $kb_id em $FILE_PATH (pattern: $pattern)" >&2
                    EXIT_CODE=1
                    TOTAL_BLOCKS=$((TOTAL_BLOCKS + 1))
                    ;;
                WARN)
                    echo "[WARN]  $kb_id em $FILE_PATH (pattern: $pattern)" >&2
                    TOTAL_WARNS=$((TOTAL_WARNS + 1))
                    ;;
            esac
        fi
    done < "$PATTERNS_FILE"
done

if [ $TOTAL_BLOCKS -gt 0 ] || [ $TOTAL_WARNS -gt 0 ]; then
    echo "" >&2
    echo "Total: $TOTAL_BLOCKS BLOCKs, $TOTAL_WARNS WARNs" >&2
    [ $TOTAL_BLOCKS -gt 0 ] && echo "Consulte docs/known_bugs.md para fix-canônico." >&2
fi

exit $EXIT_CODE
