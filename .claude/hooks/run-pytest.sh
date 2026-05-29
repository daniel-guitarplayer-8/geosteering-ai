#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK C2: Ciclo Teste — pytest Automatico no Stop                  ║
# ║  Evento: Stop                                                      ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Quando Claude termina uma tarefa, executa pytest automaticamente. ║
# ║  Se testes falharem, bloqueia a conclusao e injeta as falhas       ║
# ║  como proxima instrucao para Claude corrigir.                      ║
# ║                                                                    ║
# ║  Protecao anti-loop: se stop_hook_active=true, permite parar.     ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -uo pipefail

INPUT=$(cat)

# ── Protecao anti-loop infinito ──────────────────────────────────────
STOP_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
if [ "$STOP_ACTIVE" = "true" ]; then
    exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
TEST_DIR="$PROJECT_DIR/tests"

# Se nao existem testes, permitir parar
[ ! -d "$TEST_DIR" ] && exit 0

# Verificar se ha arquivos de teste
TESTS_EXIST=$(find "$TEST_DIR" -name 'test_*.py' -type f 2>/dev/null | head -1)
[ -z "$TESTS_EXIST" ] && exit 0

# ── Resolver Python: $PYTHON → conda env Linux → venv macOS → python ─
# Hooks rodam sem ~/.bashrc; precisamos resolver explicitamente.
if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
    PY="${PYTHON}"
elif [ -x "$HOME/anaconda3/envs/Geosteering_AI/bin/python" ]; then
    PY="$HOME/anaconda3/envs/Geosteering_AI/bin/python"
elif [ -x "$HOME/Geosteering_AI_venv/bin/python" ]; then
    PY="$HOME/Geosteering_AI_venv/bin/python"
else
    PY="python"
fi

# ── Opção A — Threshold de bloqueio do warmup CLI em hardware lento ─
# test_warmup_thread_initializes_without_blocking usa env var como sentinel;
# 2.0s default era calibrado para Mac M2 Pro. Em Threadripper 7970X o pool
# de 16 workers leva ~2.3s para inicializar. 5.0s é folga confortável.
export WARMUP_BLOCKING_THRESHOLD_S="${WARMUP_BLOCKING_THRESHOLD_S:-5.0}"

# ── Opção C — Escopo smoke (suite total > 120s timeout do Stop hook) ─
# Apenas sentinels rápidos (~30s total) que cobrem config + CLI + API.
SMOKE_PATHS=(
    "tests/test_config.py"
    "tests/test_cli_mvp.py"
    "tests/test_api_health.py"
    "tests/test_api_schemas.py"
)

# ── Executar pytest (rapido, sem verbose) ────────────────────────────
cd "$PROJECT_DIR"
RESULT=$("$PY" -m pytest "${SMOKE_PATHS[@]}" -x --tb=short -q 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    # Extrair as ultimas 25 linhas com as falhas
    TAIL=$(echo "$RESULT" | tail -25)
    cat << EOF
{
  "decision": "block",
  "reason": "pytest falhou. Corrija os testes antes de finalizar:\n$TAIL"
}
EOF
    exit 0
fi

# Testes passam — permitir parar
exit 0
