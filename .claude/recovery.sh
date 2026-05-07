#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  RECOVERY: restauração rápida de arquivo a partir de .backups/     ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Uso:                                                              ║
# ║    .claude/recovery.sh <arquivo>           — lista versões         ║
# ║    .claude/recovery.sh <arquivo> <HHMMSS>  — restaura essa versão  ║
# ║    .claude/recovery.sh <arquivo> latest    — restaura mais recente ║
# ║                                                                    ║
# ║  Examples:                                                         ║
# ║    .claude/recovery.sh CLAUDE.md                                   ║
# ║    .claude/recovery.sh CLAUDE.md latest                            ║
# ║    .claude/recovery.sh CLAUDE.md 162544                            ║
# ║    .claude/recovery.sh geosteering_ai/config.py latest 2026-05-06  ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
cd "$PROJECT_DIR"

usage() {
    grep -E '^# ║' "$0" | sed 's/^# ║ *//; s/ *║$//'
    exit 1
}

[ $# -eq 0 ] && usage

FILE_PATH="$1"
TIMESTAMP="${2:-list}"
DATE="${3:-$(date +%Y-%m-%d)}"

# Path relativo (remove prefixo do projeto se presente)
REL_PATH="${FILE_PATH#$PROJECT_DIR/}"
REL_PATH="${REL_PATH#./}"

BACKUP_BASE="$PROJECT_DIR/.backups/$DATE"

if [ ! -d "$BACKUP_BASE" ]; then
    echo "[recovery] Sem backups para o dia $DATE em $PROJECT_DIR/.backups/" >&2
    echo "Dias disponíveis:" >&2
    ls -1 "$PROJECT_DIR/.backups/" 2>/dev/null | head -5 >&2
    exit 1
fi

# Buscar todas as versões do arquivo
mapfile -t VERSIONS < <(find "$BACKUP_BASE" -name "$(basename "$REL_PATH").*.bak" -type f 2>/dev/null | sort -r)

if [ ${#VERSIONS[@]} -eq 0 ]; then
    echo "[recovery] Nenhum backup encontrado para $REL_PATH em $DATE" >&2
    exit 1
fi

# Modo LIST: exibir versões disponíveis
if [ "$TIMESTAMP" = "list" ]; then
    echo "Versões disponíveis para $REL_PATH em $DATE:"
    for v in "${VERSIONS[@]}"; do
        TS=$(basename "$v" | sed 's/^.*\.\([0-9]\{6\}\)\.bak$/\1/')
        SIZE=$(stat -f%z "$v" 2>/dev/null || stat -c%s "$v")
        printf "  %s  (%s bytes)\n" "$TS" "$SIZE"
    done
    echo ""
    echo "Para restaurar: $0 $REL_PATH <HHMMSS> [$DATE]"
    echo "Para restaurar a mais recente: $0 $REL_PATH latest [$DATE]"
    exit 0
fi

# Modo RESTORE
if [ "$TIMESTAMP" = "latest" ]; then
    BACKUP_FILE="${VERSIONS[0]}"
    TIMESTAMP=$(basename "$BACKUP_FILE" | sed 's/^.*\.\([0-9]\{6\}\)\.bak$/\1/')
else
    BACKUP_FILE="$BACKUP_BASE/${REL_PATH}.${TIMESTAMP}.bak"
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "[recovery] Versão $TIMESTAMP não encontrada para $REL_PATH" >&2
        echo "Versões disponíveis:" >&2
        for v in "${VERSIONS[@]}"; do
            echo "  $(basename "$v")" >&2
        done
        exit 1
    fi
fi

# Backup do arquivo atual antes de sobrescrever (proteção dupla)
if [ -f "$FILE_PATH" ]; then
    SAFETY="$BACKUP_BASE/${REL_PATH}.pre-recovery-$(date +%H%M%S).bak"
    mkdir -p "$(dirname "$SAFETY")"
    cp -p "$FILE_PATH" "$SAFETY"
    echo "[recovery] Salvo estado atual em: $(basename "$SAFETY")"
fi

# Restaurar
cp -p "$BACKUP_FILE" "$FILE_PATH"
echo "[recovery] ✓ Restaurado: $REL_PATH ← $DATE/$TIMESTAMP"
echo "[recovery]   Origem: $BACKUP_FILE"
