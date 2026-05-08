#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: check-anti-patterns.sh                                      ║
# ║  Evento: PreToolUse (Edit|Write)                                   ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Bloqueia ou alerta sobre padrões catalogados em .claude/anti-     ║
# ║  patterns.txt como erros comprovados — KB-013, KB-018, KB-019,    ║
# ║  KB-PYT (PyTorch), KB-PRT (print()), KB-GLB (globals), etc.       ║
# ║                                                                    ║
# ║  Formato anti-patterns.txt (TSV 4 colunas):                        ║
# ║    KB-XXX<TAB>regex<TAB>severity<TAB>path_glob                     ║
# ║                                                                    ║
# ║  Severity:                                                         ║
# ║    BLOCK — exit 1 (edição BLOQUEADA)                              ║
# ║    WARN  — exit 0 + mensagem stderr (alerta, não bloqueia)        ║
# ║                                                                    ║
# ║  Exit codes:                                                       ║
# ║    0 — sem violações BLOCK (WARNs podem aparecer no stderr)       ║
# ║    1 — pelo menos uma violação BLOCK detectada                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

# Bypass global: CLAUDE_BYPASS_ANTI_PATTERNS=1 desabilita todas as verificacoes.
# Usar apenas em contextos legitimos: debugging de anti-patterns, escrita de
# testes *sobre* padroes proibidos, ou geracoes intermediarias temporarias.
[ "${CLAUDE_BYPASS_ANTI_PATTERNS:-0}" = "1" ] && exit 0

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
NEW=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')

[ -z "$FILE_PATH" ] && exit 0
[ -z "$NEW" ] && exit 0

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || echo "/Users/daniel/Geosteering_AI")}"
PATTERNS_FILE="$PROJECT_DIR/.claude/anti-patterns.txt"
[ ! -f "$PATTERNS_FILE" ] && exit 0

BLOCKS=()
WARNS=()

# Lê catálogo TSV: kb_id <TAB> regex <TAB> severity <TAB> path_glob
while IFS=$'\t' read -r kb_id pattern severity path_glob; do
    # Pular linhas vazias e comentários (#)
    [ -z "${kb_id:-}" ] && continue
    case "$kb_id" in '#'*) continue ;; esac

    # Linhas legacy de 3 colunas: severity vira "BLOCK" (default seguro)
    if [ -z "${path_glob:-}" ]; then
        path_glob="$severity"
        severity="BLOCK"
    fi

    # Casar path do arquivo contra o glob específico do KB
    # shellcheck disable=SC2053
    [[ "$FILE_PATH" == $path_glob ]] || continue

    # Procurar regex no novo conteúdo (-- evita word-split em padrões com espaços)
    if echo "$NEW" | grep -qE -- "$pattern"; then
        case "$severity" in
            BLOCK) BLOCKS+=("$kb_id — pattern: $pattern") ;;
            WARN)  WARNS+=("$kb_id — pattern: $pattern") ;;
            *)     BLOCKS+=("$kb_id (severity desconhecida '$severity' tratada como BLOCK)") ;;
        esac
    fi
done < "$PATTERNS_FILE"

# Emite WARNs (não bloqueiam)
if [ ${#WARNS[@]} -gt 0 ]; then
    echo "" >&2
    echo "[anti-patterns] AVISOS em: $FILE_PATH" >&2
    for w in "${WARNS[@]}"; do
        echo "  ⚠ $w" >&2
    done
fi

# Emite BLOCKs (bloqueiam)
if [ ${#BLOCKS[@]} -gt 0 ]; then
    echo "" >&2
    echo "============================================================" >&2
    echo "[anti-patterns] EDIT BLOQUEADO em: $FILE_PATH" >&2
    echo "============================================================" >&2
    for v in "${BLOCKS[@]}"; do
        echo "  ✗ $v" >&2
    done
    echo "" >&2
    echo "Consulte docs/known_bugs.md para causa-raiz e workaround." >&2
    echo "============================================================" >&2
    exit 1
fi
exit 0
