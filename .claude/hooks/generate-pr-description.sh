#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: generate-pr-description.sh (I2.5 — Sprint v2.24)            ║
# ║  Evento: Manual (chamado via slash command, comando shell ou pelo  ║
# ║           usuário antes de `gh pr create`)                          ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Gera descrição formatada para Pull Request a partir de:           ║
# ║    - git log <base>..HEAD                                          ║
# ║    - git diff --stat <base>..HEAD                                  ║
# ║    - template em .claude/templates/pr_description_template.md      ║
# ║                                                                    ║
# ║  Output: stdout (Markdown pronto para colar em `gh pr create`)    ║
# ║                                                                    ║
# ║  Uso:                                                              ║
# ║    bash .claude/hooks/generate-pr-description.sh                   ║
# ║    bash .claude/hooks/generate-pr-description.sh main              ║
# ║                                                                    ║
# ║  Exit codes:                                                       ║
# ║    0 — sucesso                                                     ║
# ║    1 — não está em git repo, ou template não encontrado            ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -uo pipefail

BASE_BRANCH="${1:-main}"

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || echo "")}"
if [ -z "$PROJECT_DIR" ]; then
    echo "[generate-pr-description] erro: não está em um repositório git" >&2
    exit 1
fi

TEMPLATE="$PROJECT_DIR/.claude/templates/pr_description_template.md"
if [ ! -f "$TEMPLATE" ]; then
    echo "[generate-pr-description] erro: template não encontrado em $TEMPLATE" >&2
    exit 1
fi

cd "$PROJECT_DIR"

# Branch atual
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "desconhecida")

# Verificar se base existe. Fallback para o ref remoto `origin/<base>` quando
# o ref local não existe — caso da CI (checkout de PR não cria branch local
# `main`, só `origin/main` após fetch-depth:0). Um base genuinamente inválido
# (não-local E não-remoto) ainda falha → preserva o gate de "base inválida".
if ! git rev-parse --verify "$BASE_BRANCH" >/dev/null 2>&1; then
    if git rev-parse --verify "origin/$BASE_BRANCH" >/dev/null 2>&1; then
        BASE_BRANCH="origin/$BASE_BRANCH"
    else
        echo "[generate-pr-description] erro: branch base '$BASE_BRANCH' não existe" >&2
        exit 1
    fi
fi

# Contagens
COMMIT_COUNT=$(git log --oneline "$BASE_BRANCH..HEAD" 2>/dev/null | wc -l | tr -d ' ')
FILE_COUNT=$(git diff --name-only "$BASE_BRANCH..HEAD" 2>/dev/null | wc -l | tr -d ' ')

# Lista de commits (formatada como bullet list)
COMMITS_LIST=$(
    git log "$BASE_BRANCH..HEAD" --pretty=format:"- %h %s" 2>/dev/null \
    || echo "- (sem commits)"
)

# Resumo automático: extrai a descrição do primeiro commit (assume convenção
# tipo `feat(escopo): descrição` ou similar)
FIRST_COMMIT_SUBJECT=$(
    git log "$BASE_BRANCH..HEAD" --reverse --pretty=format:"%s" 2>/dev/null \
    | head -1
)
SUMMARY="${FIRST_COMMIT_SUBJECT:-Branch $CURRENT_BRANCH}"

# Mudanças: lista de arquivos modificados (com tipo de mudança)
CHANGES=$(
    git diff --name-status "$BASE_BRANCH..HEAD" 2>/dev/null \
    | awk '{
        prefix = "modificado"
        if ($1 == "A") prefix = "adicionado"
        else if ($1 == "D") prefix = "removido"
        else if ($1 == "R" || $1 ~ /^R/) prefix = "renomeado"
        printf "- %s `%s`\n", prefix, $NF
    }' \
    | head -30
)
if [ -z "$CHANGES" ]; then
    CHANGES="- (sem mudanças vs $BASE_BRANCH)"
fi

# Substituição via Python (suporta valores multi-linha sem escape complexo).
# Python 3 é dependência garantida do projeto (CLAUDE.md fixa Python 3.13).
# Variáveis bash são exportadas para que `os.environ` as veja.
export SUMMARY CHANGES CURRENT_BRANCH BASE_BRANCH COMMITS_LIST COMMIT_COUNT FILE_COUNT

python3 - "$TEMPLATE" <<'PYEOF'
import os
import sys

template_path = sys.argv[1]
with open(template_path, "r", encoding="utf-8") as fh:
    content = fh.read()

mapping = {
    "{SUMMARY}": os.environ.get("SUMMARY", ""),
    "{CHANGES}": os.environ.get("CHANGES", ""),
    "{BRANCH}": os.environ.get("CURRENT_BRANCH", ""),
    "{BASE}": os.environ.get("BASE_BRANCH", ""),
    "{COMMITS_LIST}": os.environ.get("COMMITS_LIST", ""),
    "{COMMIT_COUNT}": os.environ.get("COMMIT_COUNT", "0"),
    "{FILE_COUNT}": os.environ.get("FILE_COUNT", "0"),
}
for placeholder, value in mapping.items():
    content = content.replace(placeholder, value)

sys.stdout.write(content)
PYEOF
exit $?
