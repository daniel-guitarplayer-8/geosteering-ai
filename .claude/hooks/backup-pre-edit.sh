#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: backup-pre-edit.sh                                          ║
# ║  Evento: PreToolUse (Edit|Write)                                   ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Cria cópia de segurança do arquivo ANTES de qualquer Edit/Write.  ║
# ║  Preserva versões múltiplas no mesmo dia via sufixo HH:MM:SS.bak.  ║
# ║  Compatibilidade: macOS (BSD) + Linux (GNU).                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0
[ ! -f "$FILE_PATH" ] && exit 0  # criação de arquivo novo — sem backup necessário

# Apenas extensões versionáveis (não backup de binários ou logs)
case "$FILE_PATH" in
    *.py | *.f08 | *.yaml | *.yml | *.json | *.md | *.sh | *.toml | *.tex | *.bib)
        ;;
    *)
        exit 0
        ;;
esac

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
BACKUP_DIR="$PROJECT_DIR/.backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Path relativo para preservar estrutura de diretórios
REL_PATH="${FILE_PATH#$PROJECT_DIR/}"

# Sufixo com timestamp HH:MM:SS para múltiplas versões no mesmo dia
TIMESTAMP=$(date +%H%M%S)
DEST_FILE="$BACKUP_DIR/${REL_PATH}.${TIMESTAMP}.bak"
mkdir -p "$(dirname "$DEST_FILE")"
cp -p "$FILE_PATH" "$DEST_FILE"

echo "[backup] ${REL_PATH} → .backups/$(date +%Y-%m-%d)/${REL_PATH}.${TIMESTAMP}.bak" >&2
exit 0  # nunca bloqueia
