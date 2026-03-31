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

# ── Executar pytest (rapido, sem verbose) ────────────────────────────
cd "$PROJECT_DIR"
RESULT=$(${PYTHON:-python} -m pytest tests/ -x --tb=short -q 2>&1)
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