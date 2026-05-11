#!/usr/bin/env bash
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  check-perf-regression.sh                                                 ║
# ║  Sprint v2.29.3                                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Hook anti-regressão de throughput do simulador Python Numba JIT.         ║
# ║                                                                           ║
# ║  Roda 1 benchmark mínimo (Cenário E, n=200 warm cache) e compara          ║
# ║  contra baseline registrado em .claude/perf_baseline.json. Reporta        ║
# ║  WARN (não bloqueia) se throughput < 90% do baseline.                     ║
# ║                                                                           ║
# ║  USO MANUAL                                                               ║
# ║    bash .claude/hooks/check-perf-regression.sh                            ║
# ║                                                                           ║
# ║  TRIGGER AUTOMÁTICO                                                       ║
# ║    Configurar em settings.json como PostToolUse após 'git commit' que     ║
# ║    toque geosteering_ai/simulation/. Não bloqueia (apenas alerta).        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
set -euo pipefail

BASELINE_FILE="${CLAUDE_PROJECT_DIR:-.}/.claude/perf_baseline.json"
SCENARIO="${SCENARIO:-E}"
N_MODELS="${N_MODELS:-200}"
THRESHOLD_PCT="${THRESHOLD_PCT:-90}"  # alerta se < 90% do baseline
VERSION="${VERSION:-v2.29.3}"  # versão do projeto (override via env var)

# Ativar venv se não estiver ativo
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${HOME}/Geosteering_AI_venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/Geosteering_AI_venv/bin/activate"
fi

if ! command -v python &>/dev/null; then
    echo "[check-perf-regression] WARN: python não encontrado, pulando." >&2
    exit 0
fi

# Verifica se geosteering_ai está disponível
if ! python -c "import geosteering_ai" 2>/dev/null; then
    echo "[check-perf-regression] WARN: pacote geosteering_ai não importável, pulando." >&2
    exit 0
fi

# Executa benchmark e captura mod/h (último número antes de 'mod/h')
OUTPUT=$(python -m geosteering_ai.cli benchmark --scenario "${SCENARIO}" --n "${N_MODELS}" 2>&1 | tail -3)
THROUGHPUT=$(echo "${OUTPUT}" | grep -oE '[0-9,]+\s*mod/h' | tr -d ',' | grep -oE '[0-9]+' | head -1)

if [ -z "${THROUGHPUT}" ]; then
    echo "[check-perf-regression] WARN: throughput não capturado. Output:" >&2
    echo "${OUTPUT}" >&2
    exit 0
fi

echo "[check-perf-regression] Cenário ${SCENARIO} n=${N_MODELS}: ${THROUGHPUT} mod/h"

# Verifica baseline
if [ ! -f "${BASELINE_FILE}" ]; then
    echo "[check-perf-regression] Baseline ausente. Criando: ${BASELINE_FILE}"
    mkdir -p "$(dirname "${BASELINE_FILE}")"
    cat > "${BASELINE_FILE}" <<EOF
{
  "scenarios": {
    "${SCENARIO}_n${N_MODELS}": {
      "throughput_mod_h": ${THROUGHPUT},
      "measured_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
      "version": "${VERSION}"
    }
  },
  "threshold_pct": ${THRESHOLD_PCT},
  "notes": "Atualize manualmente após otimizações intencionais (>+5%)."
}
EOF
    exit 0
fi

# Lê baseline
BASELINE=$(python -c "
import json, sys
with open('${BASELINE_FILE}') as f:
    data = json.load(f)
key = '${SCENARIO}_n${N_MODELS}'
sc = data.get('scenarios', {}).get(key)
if sc is None:
    print('NO_KEY')
else:
    print(int(sc.get('throughput_mod_h', 0)))
")

if [ "${BASELINE}" = "NO_KEY" ]; then
    echo "[check-perf-regression] Baseline para ${SCENARIO}_n${N_MODELS} ausente, adicionando."
    python -c "
import json
with open('${BASELINE_FILE}', 'r+') as f:
    data = json.load(f)
    data.setdefault('scenarios', {})['${SCENARIO}_n${N_MODELS}'] = {
        'throughput_mod_h': ${THROUGHPUT},
        'measured_at': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
        'version': '${VERSION}',
    }
    f.seek(0); f.truncate()
    json.dump(data, f, indent=2)
"
    exit 0
fi

# Guard divisão por zero
if [ "${BASELINE}" -eq 0 ]; then
    echo "[check-perf-regression] WARN: baseline registrado é 0 — impossível calcular percentual." >&2
    exit 0
fi

# Calcula percentual atual vs baseline
PCT=$(python -c "print(int(${THROUGHPUT} * 100 / ${BASELINE}))")
THRESHOLD_LOW=$((BASELINE * THRESHOLD_PCT / 100))

echo "[check-perf-regression] Baseline: ${BASELINE} mod/h · Atual: ${THROUGHPUT} mod/h · ${PCT}% baseline"

if [ "${THROUGHPUT}" -lt "${THRESHOLD_LOW}" ]; then
    cat >&2 <<EOF
[check-perf-regression] ⚠️  ALERTA: REGRESSÃO DE THROUGHPUT DETECTADA

    Cenário ${SCENARIO} n=${N_MODELS}:
      Baseline:      ${BASELINE} mod/h
      Atual:         ${THROUGHPUT} mod/h
      Percentual:    ${PCT}% (limite: ${THRESHOLD_PCT}%)
      Threshold:     ${THRESHOLD_LOW} mod/h

    Investigue se a mudança recente introduziu regressão. Se intencional,
    atualize o baseline em ${BASELINE_FILE}.
EOF
    # Não bloqueia (exit 0) — apenas alerta
    exit 0
fi

echo "[check-perf-regression] ✓ PASS (${PCT}% do baseline)"
exit 0
