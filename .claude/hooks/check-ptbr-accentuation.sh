#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: check-ptbr-accentuation.sh (I2.5 — Sprint v2.24)            ║
# ║  Evento: PostToolUse (Edit|Write)                                  ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Detecta palavras em PT-BR escritas SEM acentuação no conteúdo    ║
# ║  recém-editado. Alerta o usuário (severidade WARN) sem bloquear   ║
# ║  a operação.                                                       ║
# ║                                                                    ║
# ║  Catálogo: .claude/ptbr-words.txt (TSV: sem_acento → com_acento)  ║
# ║                                                                    ║
# ║  Aplicabilidade:                                                   ║
# ║    - Arquivos .md, .py, .sh                                        ║
# ║    - SKIP em legacy/, old_geosteering_ai/, .backups/, node_modules ║
# ║                                                                    ║
# ║  Bypass: CLAUDE_BYPASS_PTBR=1                                      ║
# ║                                                                    ║
# ║  Exit codes:                                                       ║
# ║    0 — sempre (não bloqueia, apenas alerta no stderr)             ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -uo pipefail

# Bypass global
[ "${CLAUDE_BYPASS_PTBR:-0}" = "1" ] && exit 0

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
NEW=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')

[ -z "$FILE_PATH" ] && exit 0
[ -z "$NEW" ] && exit 0

# Aplicar apenas a arquivos textuais relevantes
case "$FILE_PATH" in
    *.md|*.py|*.sh|*.f08|*.f90) ;;
    *) exit 0 ;;
esac

# Whitelist de paths a ignorar (código legado/externos/backups)
case "$FILE_PATH" in
    */legacy/*|*/old_geosteering_ai/*|*/.backups/*|*/node_modules/*|*/__pycache__/*) exit 0 ;;
esac

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || echo "/Users/daniel/Geosteering_AI")}"
WORDS_FILE="$PROJECT_DIR/.claude/ptbr-words.txt"
[ ! -f "$WORDS_FILE" ] && exit 0

WARNS=()

# Lê catálogo TSV
while IFS=$'\t' read -r unaccented accented; do
    # Pular linhas vazias e comentários
    [ -z "${unaccented:-}" ] && continue
    case "$unaccented" in '#'*) continue ;; esac
    [ -z "${accented:-}" ] && continue

    # grep -wE = word-boundary, evita match em "naomi", "configuracaoX", etc.
    # -i = case-insensitive (cobre "Nao", "NAO", etc.)
    if echo "$NEW" | grep -wiE "$unaccented" > /dev/null 2>&1; then
        WARNS+=("'$unaccented' → deveria ser '$accented'")
    fi
done < "$WORDS_FILE"

# Emitir alertas (sem bloquear)
if [ ${#WARNS[@]} -gt 0 ]; then
    echo "" >&2
    echo "[ptbr-accentuation] Palavras sem acento detectadas em: $FILE_PATH" >&2
    for w in "${WARNS[@]}"; do
        echo "  ⚠ $w" >&2
    done
    echo "  Regra inviolável CLAUDE.md: documentos PT-BR DEVEM ter acentuação." >&2
    echo "" >&2
fi

exit 0
