#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HOOK: colab-token-refresh.sh                                              ║
# ║  Tipo: PreToolUse (matcher Bash com regex \b(colab|gcloud)\b)              ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Sprint v2.40 (I2.2 MCP colab-bridge)                ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Propósito:                                                                ║
# ║    Avisa proativamente quando o token GCP/Colab está prestes a expirar    ║
# ║    (mais de 50 min desde a última renovação). Não bloqueia execução;      ║
# ║    apenas imprime warning para o usuário considerar `gcloud auth login`.  ║
# ║                                                                            ║
# ║  Comportamento:                                                            ║
# ║    1. Lê modtime de ~/.config/gcloud/access_tokens.db                     ║
# ║    2. Calcula idade em segundos                                            ║
# ║    3. Se >3000s (50min), emite warning (stderr) + exit 0 (não-bloqueante) ║
# ║    4. Nunca dispara `gcloud auth print-access-token` (pode abrir UI prompt)║
# ║                                                                            ║
# ║  Limites:                                                                  ║
# ║    - Não verifica validade real do token (apenas idade do arquivo)        ║
# ║    - Tokens podem ser válidos por mais tempo se OAuth refresh automático  ║
# ║    - Tokens podem expirar antes de 50min em casos raros (revogação)       ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v2.40 (2026-05-18) — Implementação inicial (Sprint I2.2, commit C02)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────
# Configuração: threshold de idade do token (segundos)
# ────────────────────────────────────────────────────────────────────────
readonly TOKEN_AGE_WARNING_THRESHOLD=3000  # 50 minutos
readonly TOKEN_DB="${HOME}/.config/gcloud/access_tokens.db"

# ────────────────────────────────────────────────────────────────────────
# Função: emite warning sem bloquear
# ────────────────────────────────────────────────────────────────────────
emit_warning() {
    local age_min="$1"
    cat >&2 <<EOF

⚠️  [colab-token-refresh] Token GCP/Colab pode estar prestes a expirar.
    Idade aproximada: ${age_min} min (limite 60 min antes de expirar).

    Para refresh proativo (antes de operação longa em Colab):
        gcloud auth login
        gcloud auth print-access-token  # confirma novo token

    Este aviso é informativo; execução do comando NÃO foi bloqueada.

EOF
}

# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

# Saída defensiva: se gcloud não instalado, nada a fazer
if ! command -v gcloud &> /dev/null; then
    exit 0
fi

# Saída defensiva: se DB nunca foi criada, usuário ainda não fez login — nada a fazer
if [[ ! -f "${TOKEN_DB}" ]]; then
    exit 0
fi

# Early-exit: só roda lógica completa se comando é colab/gcloud-relacionado.
# Hardening v2.40 (review security #1): matcher Bash em settings.json é amplo;
# filtrar aqui evita overhead em comandos não relacionados (ls, pytest, etc).
TOOL_INPUT="${CLAUDE_TOOL_INPUT:-}"
if [[ -n "${TOOL_INPUT}" ]]; then
    if ! echo "${TOOL_INPUT}" | grep -qE '\b(colab|gcloud)\b'; then
        exit 0
    fi
fi

# Calcular idade do arquivo de tokens (em segundos)
# Compatível com macOS (BSD stat) e Linux (GNU stat)
if [[ "$(uname)" == "Darwin" ]]; then
    file_mtime=$(stat -f %m "${TOKEN_DB}" 2>/dev/null || echo "0")
else
    file_mtime=$(stat -c %Y "${TOKEN_DB}" 2>/dev/null || echo "0")
fi

# Hardening v2.40 (review security #2): fallback defensivo
# se stat retornar string vazia (não esperado, mas defesa em profundidade).
file_mtime=${file_mtime:-0}

now=$(date +%s)
age=$((now - file_mtime))

if [[ "${age}" -gt "${TOKEN_AGE_WARNING_THRESHOLD}" ]]; then
    age_min=$((age / 60))
    emit_warning "${age_min}"
fi

exit 0
