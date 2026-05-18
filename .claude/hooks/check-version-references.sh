#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HOOK: check-version-references.sh                                         ║
# ║  Tipo: PreCommit (matcher Edit|Write em *.md|*.py)                          ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Sprint v2.40.2 (ADR-0001 enforcer)                  ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Propósito:                                                                ║
# ║    Enforça regra dura R5 do ADR-0001: nenhum arquivo .md/.py do projeto   ║
# ║    pode cunhar definição de versão futura (vX.Y > current+1) sem          ║
# ║    referenciar explicitamente docs/ROADMAP.md.                            ║
# ║                                                                            ║
# ║  Lógica:                                                                   ║
# ║    1. Lê versão atual de pyproject.toml (campo version="X.Y.Z")           ║
# ║    2. Para cada arquivo .md ou .py modificado:                            ║
# ║       a. Procura padrões vX.Y onde X.Y > current+1                        ║
# ║       b. Se encontrar, exige menção a "docs/ROADMAP.md" ou "ROADMAP.md"   ║
# ║          no mesmo arquivo (qualquer linha)                                ║
# ║       c. Se NÃO houver menção, BLOQUEIA o commit                          ║
# ║                                                                            ║
# ║  Exceções (não bloqueia):                                                  ║
# ║    - Arquivos em docs/ROADMAP.md, docs/sprints/, docs/decisions/          ║
# ║      (são os ÚNICOS lugares onde versões futuras podem ser definidas)     ║
# ║    - Arquivos em docs/CHANGELOG.md (apenas histórico, refs OK)            ║
# ║    - Arquivos em docs/reports/ (relatórios long-form podem citar)         ║
# ║    - Arquivos em docs/sprints/archive/ (snapshots históricos)             ║
# ║    - Versão exata atual ou +1 (e.g., v2.40 ou v2.41 se current=v2.40)     ║
# ║                                                                            ║
# ║  Bypass: CLAUDE_BYPASS_VERSION_CHECK=1 (apenas em casos legítimos)        ║
# ║                                                                            ║
# ║  Histórico:                                                                ║
# ║    v2.40.2 (2026-05-18) — Implementação inicial (ADR-0001 enforcer)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────
# Bypass
# ────────────────────────────────────────────────────────────────────────
if [[ "${CLAUDE_BYPASS_VERSION_CHECK:-0}" == "1" ]]; then
    exit 0
fi

# ────────────────────────────────────────────────────────────────────────
# Detectar versão atual: PRIORIDADE git tags > pyproject.toml
# Projeto usa git tags como fonte canônica (pyproject.toml pode ficar
# stale entre releases — campo version="X.Y.Z" não é atualizado a cada
# minor). Git tags vX.Y.Z são a SSoT de versão real publicada.
# ────────────────────────────────────────────────────────────────────────
CURRENT_VERSION=""

# Tenta git tags primeiro (formato vX.Y.Z ou vX.Y)
if git -C "$(git rev-parse --show-toplevel 2>/dev/null)" rev-parse --git-dir &>/dev/null; then
    LATEST_TAG=$(git -C "$(git rev-parse --show-toplevel)" tag --list 'v[0-9]*' --sort=-v:refname 2>/dev/null | head -1)
    if [[ -n "$LATEST_TAG" ]]; then
        CURRENT_VERSION=$(echo "$LATEST_TAG" | sed -E 's/^v([0-9]+)\.([0-9]+).*/\1.\2/')
    fi
fi

# Fallback: pyproject.toml (apenas se git tags indisponíveis)
if [[ -z "$CURRENT_VERSION" ]]; then
    PYPROJECT="$(git rev-parse --show-toplevel 2>/dev/null)/pyproject.toml"
    if [[ -f "$PYPROJECT" ]]; then
        CURRENT_VERSION=$(grep -m1 -E '^version\s*=' "$PYPROJECT" | sed -E 's/.*"([0-9]+)\.([0-9]+)\..*/\1.\2/')
    fi
fi

if [[ -z "$CURRENT_VERSION" ]]; then
    # Não conseguimos determinar versão atual — não bloqueia
    exit 0
fi

CURRENT_MAJOR=$(echo "$CURRENT_VERSION" | cut -d. -f1)
CURRENT_MINOR=$(echo "$CURRENT_VERSION" | cut -d. -f2)
NEXT_MINOR=$((CURRENT_MINOR + 1))
# Limite tolerável: até current+1 sem ref a ROADMAP.md
MAX_TOLERATED_MINOR=$NEXT_MINOR

# ────────────────────────────────────────────────────────────────────────
# Determinar arquivos a checar
# ────────────────────────────────────────────────────────────────────────
# Em PreCommit hook, $CLAUDE_TOOL_INPUT contém o path do arquivo editado.
# Fallback: pegar arquivos staged.
FILES_TO_CHECK=""
if [[ -n "${CLAUDE_TOOL_INPUT:-}" ]]; then
    # Tentar extrair file_path do JSON do tool input
    FILE_FROM_TOOL=$(echo "${CLAUDE_TOOL_INPUT}" | grep -oE '"file_path"\s*:\s*"[^"]+"' | sed 's/.*"file_path"\s*:\s*"\([^"]*\)".*/\1/' | head -1)
    if [[ -n "$FILE_FROM_TOOL" ]]; then
        FILES_TO_CHECK="$FILE_FROM_TOOL"
    fi
fi
if [[ -z "$FILES_TO_CHECK" ]]; then
    FILES_TO_CHECK=$(git diff --cached --name-only --diff-filter=AM 2>/dev/null | grep -E '\.(md|py)$' || true)
fi
if [[ -z "$FILES_TO_CHECK" ]]; then
    exit 0
fi

# ────────────────────────────────────────────────────────────────────────
# Função: verifica se arquivo é exempt (não precisa check)
# ────────────────────────────────────────────────────────────────────────
is_exempt() {
    local f="$1"
    # Caminhos exempt (lista canônica)
    case "$f" in
        docs/ROADMAP.md|*/docs/ROADMAP.md) return 0 ;;
        docs/CHANGELOG.md|*/docs/CHANGELOG.md) return 0 ;;
        docs/sprints/*|*/docs/sprints/*) return 0 ;;
        docs/decisions/*|*/docs/decisions/*) return 0 ;;
        docs/reports/*|*/docs/reports/*) return 0 ;;
        *.claude/plans/*) return 0 ;;
    esac
    return 1
}

# ────────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────────
BLOCKED=0
for file in $FILES_TO_CHECK; do
    [[ ! -f "$file" ]] && continue
    if is_exempt "$file"; then
        continue
    fi

    # Buscar padrões vX.Y (sem patch) ou vX.Y.Z (com patch)
    # Captura apenas X e Y, descarta Z.
    VERSIONS_IN_FILE=$(grep -oE 'v[0-9]+\.[0-9]+' "$file" 2>/dev/null | sort -u || true)
    [[ -z "$VERSIONS_IN_FILE" ]] && continue

    # Detectar versões > current+1
    OFFENDING_VERSIONS=""
    for v in $VERSIONS_IN_FILE; do
        major=$(echo "$v" | sed -E 's/v([0-9]+)\..*/\1/')
        minor=$(echo "$v" | sed -E 's/v[0-9]+\.([0-9]+).*/\1/')
        # Comparar: bloqueia se major > current_major OU (major == current AND minor > max_tolerated)
        if [[ "$major" -gt "$CURRENT_MAJOR" ]]; then
            OFFENDING_VERSIONS="$OFFENDING_VERSIONS $v"
        elif [[ "$major" -eq "$CURRENT_MAJOR" && "$minor" -gt "$MAX_TOLERATED_MINOR" ]]; then
            OFFENDING_VERSIONS="$OFFENDING_VERSIONS $v"
        fi
    done

    [[ -z "$OFFENDING_VERSIONS" ]] && continue

    # Há versões futuras — exige referência a ROADMAP.md
    if ! grep -qE '(docs/)?ROADMAP\.md|backlog em ROADMAP|ver ROADMAP|see ROADMAP' "$file" 2>/dev/null; then
        cat >&2 <<EOF

❌ [check-version-references] BLOCKED em $file

    Versões futuras detectadas (>$CURRENT_MAJOR.$MAX_TOLERATED_MINOR):
       $OFFENDING_VERSIONS

    Versão atual do projeto (git tag mais recente): v$CURRENT_VERSION

    REGRA-DURA R5 (ADR-0001): qualquer arquivo .md/.py que mencione versão
    futura DEVE referenciar docs/ROADMAP.md (única fonte canônica de roadmap).

    Para corrigir, adicione no arquivo uma linha como:
       "Ver docs/ROADMAP.md para o backlog priorizado canônico."

    Ou, se a menção é histórica/contextual (e.g., changelog comentário),
    mova-a para um dos paths exempt:
       docs/{ROADMAP,CHANGELOG}.md | docs/{sprints,decisions,reports}/

    Bypass (apenas em casos legítimos):
       CLAUDE_BYPASS_VERSION_CHECK=1 git commit ...

EOF
        BLOCKED=1
    fi
done

exit $BLOCKED
