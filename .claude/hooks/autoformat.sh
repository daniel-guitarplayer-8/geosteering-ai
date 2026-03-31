#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK B2: Garantia de Qualidade de Codigo — Autoformat             ║
# ║  Evento: PostToolUse (Edit|Write)                                  ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Auto-formata arquivos Python apos cada edicao:                    ║
# ║    1. isort: ordena imports (perfil black)                         ║
# ║    2. black: formatacao PEP8 (line-length 90)                      ║
# ║                                                                    ║
# ║  Silencia erros para nao bloquear o fluxo de trabalho.             ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -uo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0
[[ "$FILE_PATH" != *.py ]] && exit 0
[ ! -f "$FILE_PATH" ] && exit 0

# Apenas modulos do pacote e testes
if [[ "$FILE_PATH" != *geosteering_ai/*.py ]] && \
   [[ "$FILE_PATH" != *geosteering_ai/**/*.py ]] && \
   [[ "$FILE_PATH" != *tests/*.py ]]; then
    exit 0
fi

# ── isort (ordenar imports) ──────────────────────────────────────────
if command -v isort > /dev/null 2>&1; then
    isort --profile black --line-length 90 --quiet "$FILE_PATH" 2>/dev/null || true
fi

# ── black (formatacao) ───────────────────────────────────────────────
if command -v black > /dev/null 2>&1; then
    black --line-length 90 --quiet "$FILE_PATH" 2>/dev/null || true
fi

exit 0