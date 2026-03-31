#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK A: Protecao da Integridade das Validacoes Fisicas            ║
# ║  Evento: PreToolUse (Edit|Write)                                   ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Protege contra:                                                   ║
# ║    1. Enfraquecimento das validacoes de errata em config.py        ║
# ║    2. Valores perigosos em YAML configs (antes do runtime)         ║
# ║    3. Import de PyTorch (proibido no projeto)                      ║
# ║    4. eps < 1e-15 (float32 unsafe)                                 ║
# ║    5. globals().get() em modulos v2.0                              ║
# ║                                                                    ║
# ║  NOTA: frequency_hz, spacing_meters, etc. sao defaults validados   ║
# ║  pelo __post_init__ em runtime. Este hook protege a LOGICA de      ║
# ║  validacao contra enfraquecimento e captura erros em YAML antes    ║
# ║  de from_yaml() ser chamado.                                       ║
# ║                                                                    ║
# ║  Compatibilidade: macOS (BSD grep -E) + Linux (GNU grep -E).      ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
NEW_STRING=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')

[ -z "$FILE_PATH" ] && exit 0
[ -z "$NEW_STRING" ] && exit 0

VIOLATIONS=()

# ── 1. PyTorch proibido (qualquer .py) ───────────────────────────────
if [[ "$FILE_PATH" == *.py ]]; then
    if echo "$NEW_STRING" | grep -qE '^[[:space:]]*(import torch|from torch)'; then
        VIOLATIONS+=("PROIBIDO: PyTorch nao pode ser importado. Framework exclusivo: TensorFlow/Keras.")
    fi
fi

# ── 2. eps perigoso para float32 (qualquer .py) ─────────────────────
#    Detecta eps = 1e-20, 1e-25, 1e-30 etc. (expoentes -16 a -99)
if [[ "$FILE_PATH" == *.py ]]; then
    if echo "$NEW_STRING" | grep -qE 'eps[_a-z]*[[:space:]]*[=:][[:space:]]*1e-(1[6-9]|[2-9][0-9])'; then
        VIOLATIONS+=("PERIGOSO: eps < 1e-15 detectado. Constraint: eps_tf >= 1e-15. Recomendado: 1e-12.")
    fi
fi

# ── 3. Protecao da logica de validacao em config.py ──────────────────
if [[ "$FILE_PATH" == *config.py ]]; then
    # frequency_hz, spacing_meters, sequence_length agora usam range validation
    # (100-1e6 Hz, 0.1-10.0 m, 10-100000) — nao mais equality assert.
    # Proteger apenas contra comentar asserts dos campos IMUTAVEIS.
    if echo "$NEW_STRING" | grep -qE '#[[:space:]]*assert[[:space:]]+self\.(target_scaling|input_features|output_targets)'; then
        VIOLATIONS+=("config.py: Assertiva de errata comentada. Validacoes de errata nao devem ser desativadas.")
    fi
fi

# ── 4. Valores perigosos em YAML configs ─────────────────────────────
if [[ "$FILE_PATH" == *.yaml ]] || [[ "$FILE_PATH" == *.yml ]]; then
    if echo "$NEW_STRING" | grep -qE 'frequency_hz:[[:space:]]*2\.0[[:space:]]*$'; then
        VIOLATIONS+=("YAML: frequency_hz: 2.0 provavelmente errado. Default: 20000.0 Hz. Range valido: [100, 1e6].")
    fi
    if echo "$NEW_STRING" | grep -qE 'spacing_meters:[[:space:]]*1000'; then
        VIOLATIONS+=("YAML: spacing_meters: 1000 provavelmente errado. Default: 1.0 m. Range valido: [0.1, 10.0].")
    fi
    if echo "$NEW_STRING" | grep -qE "target_scaling:[[:space:]]*['\"]log['\"]"; then
        VIOLATIONS+=("YAML: target_scaling: 'log' provavelmente errado. Default validado: 'log10'.")
    fi
    if echo "$NEW_STRING" | grep -qE 'eps_tf:[[:space:]]*1e-(1[6-9]|[2-9][0-9])'; then
        VIOLATIONS+=("YAML: eps_tf abaixo de 1e-15 viola constraint float32.")
    fi
fi

# ── 5. globals().get() proibido em modulos v2.0 ─────────────────────
if [[ "$FILE_PATH" == *geosteering_ai/*.py ]] || [[ "$FILE_PATH" == *geosteering_ai/**/*.py ]]; then
    if echo "$NEW_STRING" | grep -qE 'globals\(\)[[:space:]]*\.get\('; then
        VIOLATIONS+=("Padrao v2.0: globals().get() proibido em geosteering_ai/. Usar config.param via PipelineConfig.")
    fi
fi

# ── Resultado ────────────────────────────────────────────────────────
if [ ${#VIOLATIONS[@]} -gt 0 ]; then
    {
        echo "Hook A — Protecao da Integridade das Validacoes Fisicas:"
        echo "Arquivo: $FILE_PATH"
        for v in "${VIOLATIONS[@]}"; do
            echo "  - $v"
        done
    } >&2
    exit 2
fi

exit 0