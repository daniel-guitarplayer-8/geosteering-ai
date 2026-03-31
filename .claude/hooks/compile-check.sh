#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK C1: Ciclo Teste — Verificacao de Compilacao                  ║
# ║  Evento: PostToolUse (Edit|Write)                                  ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Executa py_compile imediatamente apos cada edicao de .py.         ║
# ║  SyntaxError eh capturado na hora, nao minutos depois.             ║
# ║  Bloqueia (exit 2) se o arquivo nao compilar.                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -uo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0
[[ "$FILE_PATH" != *.py ]] && exit 0
[ ! -f "$FILE_PATH" ] && exit 0

# Verificar compilacao
# Usar python (conda) em vez de python3 (sistema) para compatibilidade
PYTHON_CMD="${PYTHON:-python}"
RESULT=$($PYTHON_CMD -m py_compile "$FILE_PATH" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    {
        echo "Hook C1 — Erro de compilacao em $(basename "$FILE_PATH"):"
        echo "$RESULT"
        echo "Corrija o SyntaxError antes de continuar."
    } >&2
    exit 2
fi

exit 0