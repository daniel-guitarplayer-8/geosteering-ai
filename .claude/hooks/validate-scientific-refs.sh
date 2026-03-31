#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HOOK F: Validação de Referências Científicas                          ║
# ║  Evento: PostToolUse (Edit|Write)                                      ║
# ║  Projeto: Geosteering AI v2.0                                         ║
# ║  Fase: C — Automação Científica                                        ║
# ║                                                                        ║
# ║  Propósito:                                                            ║
# ║    1. Detectar constantes físicas novas/modificadas em código Python   ║
# ║    2. Alertar quando artigos são referenciados sem formatação padrão   ║
# ║    3. Sugerir uso do /consensus-search para validação                  ║
# ║    4. Verificar consistência de referências bibliográficas             ║
# ║                                                                        ║
# ║  NOTA: Este hook é informativo (exit 0) — não bloqueia edições.       ║
# ║  Emite avisos via stderr que aparecem como sugestões no Claude Code.   ║
# ║                                                                        ║
# ║  Compatibilidade: macOS (BSD grep -E) + Linux (GNU grep -E).          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
set -uo pipefail
# Nota: 'set -e' omitido intencionalmente — hook informativo NUNCA deve bloquear.
# Se qualquer comando falhar (ex: jq ausente), o trap garante exit 0.
trap 'exit 0' ERR

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
NEW_STRING=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')

# ── Sair silenciosamente se não houver contexto ──────────────────────
[ -z "$FILE_PATH" ] && exit 0
[ -z "$NEW_STRING" ] && exit 0

SUGGESTIONS=()

# ════════════════════════════════════════════════════════════════════════
# REGRA 1: Constantes físicas novas ou modificadas
# ════════════════════════════════════════════════════════════════════════
# Detecta definições de constantes físicas que podem precisar de
# validação contra a literatura científica.
# ──────────────────────────────────────────────────────────────────────
if [[ "$FILE_PATH" == *.py ]]; then
    # Detectar constantes numéricas em contexto físico
    if echo "$NEW_STRING" | grep -qE '(FREQUENCY|SPACING|SKIN_DEPTH|MU_0|SIGMA|OMEGA|PERMEABILITY|CONDUCTIVITY|RESISTIVITY)[_A-Z]*[[:space:]]*=[[:space:]]*[0-9]'; then
        SUGGESTIONS+=("Constante física detectada. Considere validar via /consensus-search com query relevante.")
    fi

    # Detectar novas equações ou fórmulas
    if echo "$NEW_STRING" | grep -qE '(delta|skin_depth|k_squared|wavenumber|helmholtz)[[:space:]]*=[[:space:]]*'; then
        SUGGESTIONS+=("Equação física detectada. Referência bibliográfica recomendada (padrão D12).")
    fi
fi

# ════════════════════════════════════════════════════════════════════════
# REGRA 2: Referências bibliográficas sem formatação padrão
# ════════════════════════════════════════════════════════════════════════
# O padrão do projeto é: "Autor et al. (YYYY)" ou "Ref: Autor et al.
# (YYYY) — Título/contribuição".
# ──────────────────────────────────────────────────────────────────────
if [[ "$FILE_PATH" == *.py ]]; then
    # Detectar referências "et al." sem ano (YYYY) na mesma linha
    # Nota: grep -q suprime saída, então não pode ser piped.
    # Solução: verificar presença de "et al." E ausência de "(YYYY)".
    if echo "$NEW_STRING" | grep -qE 'based on [A-Z][a-z]+ et al\.'; then
        if ! echo "$NEW_STRING" | grep -qE 'et al\.[^)]*\([12][0-9]{3}\)'; then
            SUGGESTIONS+=("Referência bibliográfica sem ano detectada. Formato recomendado: 'Autor et al. (YYYY) — contribuição'.")
        fi
    fi
fi

# ════════════════════════════════════════════════════════════════════════
# REGRA 3: Novos módulos de loss ou arquitetura
# ════════════════════════════════════════════════════════════════════════
# Quando uma nova loss function ou arquitetura é adicionada, sugerir
# busca por literatura relacionada para fundamentação.
# ──────────────────────────────────────────────────────────────────────
if [[ "$FILE_PATH" == *losses/*.py ]] || [[ "$FILE_PATH" == *models/*.py ]]; then
    # Nova função de loss detectada
    if echo "$NEW_STRING" | grep -qE '^def make_[a-z_]+\('; then
        SUGGESTIONS+=("Nova factory de loss/modelo detectada. Considere /consensus-search para fundamentação bibliográfica.")
    fi
    # Nova classe de modelo
    if echo "$NEW_STRING" | grep -qE '^def build_[a-z_]+\('; then
        SUGGESTIONS+=("Nova função build_ detectada. Considere /arxiv-search para estado-da-arte da arquitetura.")
    fi
fi

# ════════════════════════════════════════════════════════════════════════
# REGRA 4: Documentação com referências
# ════════════════════════════════════════════════════════════════════════
# Verificar se documentação MD menciona papers sem links
# ──────────────────────────────────────────────────────────────────────
if [[ "$FILE_PATH" == *.md ]]; then
    if echo "$NEW_STRING" | grep -qE 'et al\.\s*\([12][0-9]{3}\)' && ! echo "$NEW_STRING" | grep -qE '(doi\.org|arxiv\.org|scholar\.google)'; then
        SUGGESTIONS+=("Referência a artigo sem link detectada. Considere adicionar DOI ou ArXiv URL.")
    fi
fi

# ── Resultado (informativo — não bloqueia) ───────────────────────────
if [ ${#SUGGESTIONS[@]} -gt 0 ]; then
    {
        echo "Hook F — Validação de Referências Científicas (informativo):"
        echo "Arquivo: $FILE_PATH"
        for s in "${SUGGESTIONS[@]}"; do
            echo "  [INFO] $s"
        done
    } >&2
fi

# Sempre exit 0 — hook informativo, não bloqueia edições
exit 0