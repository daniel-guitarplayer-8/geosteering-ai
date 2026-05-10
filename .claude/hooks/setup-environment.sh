#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK E1: Injecao de Contexto — Setup de Sessao                    ║
# ║  Evento: SessionStart (startup)                                    ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  No inicio de cada sessao nova:                                    ║
# ║    1. Informa branch e ultimo commit                               ║
# ║    2. Executa pytest rapido e mostra resultado                     ║
# ║    3. Conta arquivos e linhas do pacote                            ║
# ║    4. Mostra versao do Python                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -uo pipefail

INPUT=$(cat)
SOURCE=$(echo "$INPUT" | jq -r '.source // "startup"')

# Apenas em startup (nao resume/compact)
[ "$SOURCE" != "startup" ] && exit 0

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
cd "$PROJECT_DIR" 2>/dev/null || exit 0

# ── Coletar informacoes ──────────────────────────────────────────────
GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "desconhecido")
LAST_COMMIT=$(git log --oneline -1 2>/dev/null || echo "sem commits")
MODIFIED=$(git diff --name-only 2>/dev/null | wc -l | tr -d ' ')
UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')
PY_VERSION=$(${PYTHON:-python} --version 2>/dev/null || echo "Python nao encontrado")

# Contar arquivos do pacote
PKG_FILES=$(find "$PROJECT_DIR/geosteering_ai" -name '*.py' -type f 2>/dev/null | wc -l | tr -d ' ')
TEST_FILES=$(find "$PROJECT_DIR/tests" -name 'test_*.py' -type f 2>/dev/null | wc -l | tr -d ' ')

# Pytest rapido (silencioso)
if [ -d "$PROJECT_DIR/tests" ]; then
    TEST_RESULT=$(cd "$PROJECT_DIR" && ${PYTHON:-python} -m pytest tests/ -q --tb=no 2>&1 | tail -1)
else
    TEST_RESULT="Sem diretorio tests/"
fi

# ── Sprint v2.23 — CPU topology + recomendacao de paralelismo ─────────
# Mostra topologia detectada e defaults adaptativos (n_workers x threads).
# Tolerante a falhas: se modulo nao importavel, omite linha.
CPU_INFO=$(${PYTHON:-python} -c "
try:
    from geosteering_ai.simulation._workers import detect_cpu_topology, recommend_default_parallelism
    logical, physical, has_ht = detect_cpu_topology()
    nw, npt = recommend_default_parallelism()
    ht = 'sim' if has_ht else 'nao'
    print(f'CPU: {physical}P/{logical}L (HT={ht}) - default v2.23: {nw}w x {npt}t')
except Exception:
    pass
" 2>/dev/null)

# ── Injetar contexto ────────────────────────────────────────────────
cat << EOF
=== Geosteering AI v2.0 — Sessao Iniciada ===
Branch: $GIT_BRANCH | Commit: $LAST_COMMIT
Git: $MODIFIED modificado(s), $UNTRACKED nao-rastreado(s)
Pacote: $PKG_FILES .py em geosteering_ai/ | $TEST_FILES testes
Testes: $TEST_RESULT
$PY_VERSION
${CPU_INFO:+$CPU_INFO}
EOF

exit 0
