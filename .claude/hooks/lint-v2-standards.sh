#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK B1: Garantia de Qualidade de Codigo — Lint v2.0              ║
# ║  Evento: PostToolUse (Edit|Write)                                  ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Verifica padroes obrigatorios apos cada edicao:                   ║
# ║    1. print() proibido — usar logging                              ║
# ║    2. globals().get() proibido em geosteering_ai/                  ║
# ║    3. import torch proibido                                        ║
# ║    4. Mega-header D1 presente                                      ║
# ║                                                                    ║
# ║  Nao bloqueia (exit 0), apenas injeta feedback para correcao.      ║
# ║  Compatibilidade: macOS (BSD grep -E) + Linux (GNU grep -E).      ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0
[[ "$FILE_PATH" != *.py ]] && exit 0

# Apenas modulos do pacote v2.0
[[ "$FILE_PATH" != *geosteering_ai/*.py ]] && [[ "$FILE_PATH" != *geosteering_ai/**/*.py ]] && exit 0

# Verificar se o arquivo existe
[ ! -f "$FILE_PATH" ] && exit 0

WARNINGS=()

# ── 1. print() proibido ─────────────────────────────────────────────
if grep -nE '^[[:space:]]*print\(' "$FILE_PATH" > /dev/null 2>&1; then
    LINES=$(grep -nE '^[[:space:]]*print\(' "$FILE_PATH" | head -3 | cut -d: -f1 | tr '\n' ',' | sed 's/,$//')
    WARNINGS+=("print() detectado (linhas $LINES). Usar logger.info() ou logger.debug().")
fi

# ── 2. globals().get() proibido ──────────────────────────────────────
if grep -nE 'globals\(\)[[:space:]]*\.get\(' "$FILE_PATH" > /dev/null 2>&1; then
    COUNT=$(grep -cE 'globals\(\)[[:space:]]*\.get\(' "$FILE_PATH")
    WARNINGS+=("globals().get() detectado ($COUNT ocorrencias). Usar config.param via PipelineConfig.")
fi

# ── 3. import torch proibido ────────────────────────────────────────
if grep -nE '^[[:space:]]*(import torch|from torch)' "$FILE_PATH" > /dev/null 2>&1; then
    WARNINGS+=("import torch detectado. Framework exclusivo: TensorFlow/Keras.")
fi

# ── 4. Mega-header D1 presente ──────────────────────────────────────
if ! head -30 "$FILE_PATH" | grep -q 'Geosteering AI v2.0' 2>/dev/null; then
    WARNINGS+=("Mega-header D1 ausente ou incompleto (primeiras 30 linhas nao contem 'Geosteering AI v2.0').")
fi

# ── Resultado (feedback, nao bloqueio) ───────────────────────────────
if [ ${#WARNINGS[@]} -gt 0 ]; then
    {
        echo "Hook B1 — Lint v2.0 ($(basename "$FILE_PATH")):"
        for w in "${WARNINGS[@]}"; do
            echo "  - $w"
        done
    } >&2
fi

exit 0