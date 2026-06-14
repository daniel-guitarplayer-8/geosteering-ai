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

# Bypass global: CLAUDE_BYPASS_ANTI_PATTERNS=1 desabilita todas as verificacoes.
[ "${CLAUDE_BYPASS_ANTI_PATTERNS:-0}" = "1" ] && exit 0

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
PATTERNS_FILE="$PROJECT_DIR/.claude/anti-patterns.txt"
[ ! -f "$PATTERNS_FILE" ] && exit 0
[ $# -eq 0 ] && exit 0

EXIT_CODE=0
TOTAL_BLOCKS=0
TOTAL_WARNS=0

STRIP_HELPER="$PROJECT_DIR/.claude/hooks/strip_code_comments.py"

for FILE_PATH in "$@"; do
    [ ! -f "$FILE_PATH" ] && continue

    # ── Pré-processa o arquivo UMA vez: remove COMENTÁRIOS + DOCSTRINGS/strings ──
    # Os padrões devem casar só CÓDIGO REAL — nunca o mesmo texto num comentário
    # ou docstring (que dava falso-positivo: o header D1 "NUNCA globals().get()"
    # disparava KB-GLB; um print() de exemplo numa docstring dispararia KB-PRT).
    #   - .py  → tokenize (branqueia comentários E strings/docstrings; robusto);
    #   - demais (.sh/.yaml/.f08) → filtro por-linha de comentário (#/!);
    #   - fallback gracioso se python3 falhar (degrada p/ filtro por-linha).
    STRIPPED_FILE="$(mktemp)"
    case "$FILE_PATH" in
        *.py)
            if ! python3 "$STRIP_HELPER" "$FILE_PATH" > "$STRIPPED_FILE" 2>/dev/null; then
                grep -vE '^[[:space:]]*[#!]' "$FILE_PATH" > "$STRIPPED_FILE" 2>/dev/null || true
            fi
            ;;
        *)
            grep -vE '^[[:space:]]*[#!]' "$FILE_PATH" > "$STRIPPED_FILE" 2>/dev/null || true
            ;;
    esac

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

        if grep -qE -- "$pattern" "$STRIPPED_FILE"; then
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

    rm -f "$STRIPPED_FILE"
done

if [ $TOTAL_BLOCKS -gt 0 ] || [ $TOTAL_WARNS -gt 0 ]; then
    echo "" >&2
    echo "Total: $TOTAL_BLOCKS BLOCKs, $TOTAL_WARNS WARNs" >&2
    [ $TOTAL_BLOCKS -gt 0 ] && echo "Consulte docs/known_bugs.md para fix-canônico." >&2
fi

exit $EXIT_CODE
