#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  worktree-create.sh — Cria worktree com sparse-checkout            ║
# ║  Uso: .claude/scripts/worktree-create.sh <branch> [<dir>]         ║
# ║                                                                    ║
# ║  Lê .worktreeinclude para configurar sparse-checkout.              ║
# ║  Cria symlink para Geosteering_AI_venv (não recria — I1.7).       ║
# ║  Copia .env se existir no diretório principal.                     ║
# ║                                                                    ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║  Contexto: §18.3 do documento de aprofundamento + I1.7 (Fase 1)   ║
# ║  Compatibilidade: macOS (BSD) + Linux (GNU)                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

BRANCH="${1:-}"
if [[ -z "$BRANCH" ]]; then
    echo "Uso: $(basename "$0") <nome-da-branch> [<diretório-destino>]" >&2
    echo ""
    echo "Exemplos:" >&2
    echo "  $(basename "$0") feat/simulator-v2.23" >&2
    echo "  $(basename "$0") feat/exp-pytorch-adapter .claude/worktrees/pytorch" >&2
    exit 1
fi

# Resolver raiz do repositório a partir da localização do script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

WORKTREE_NAME="$(basename "$BRANCH" | tr '/' '-')"
WORKTREE_DIR="${2:-$REPO_ROOT/.claude/worktrees/$WORKTREE_NAME}"
WORKTREEINCLUDE="$REPO_ROOT/.worktreeinclude"
VENV_PATH="$HOME/Geosteering_AI_venv"

# ── Verificações pré-condição ────────────────────────────────────────
if [[ ! -f "$WORKTREEINCLUDE" ]]; then
    echo "ERRO: .worktreeinclude não encontrado em $REPO_ROOT" >&2
    exit 2
fi

if [[ -d "$WORKTREE_DIR" ]]; then
    echo "ERRO: Diretório de worktree já existe: $WORKTREE_DIR" >&2
    echo "Use 'git worktree list' para ver worktrees ativas." >&2
    exit 2
fi

# ── Criar worktree ────────────────────────────────────────────────────
echo "[worktree-create] Criando worktree: $WORKTREE_DIR (branch: $BRANCH)"
# Tenta criar nova branch; se já existir, usa a existente
if git -C "$REPO_ROOT" branch --list "$BRANCH" | grep -q .; then
    git -C "$REPO_ROOT" worktree add "$WORKTREE_DIR" "$BRANCH"
else
    git -C "$REPO_ROOT" worktree add "$WORKTREE_DIR" -b "$BRANCH"
fi

# ── Configurar sparse-checkout com padrões do .worktreeinclude ───────
echo "[worktree-create] Configurando sparse-checkout..."
git -C "$WORKTREE_DIR" sparse-checkout init --cone

# Ler padrões do .worktreeinclude (ignorar comentários e linhas vazias)
PATTERNS=()
while IFS= read -r line; do
    # Remover comentários inline e espaços
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"  # ltrim
    line="${line%"${line##*[![:space:]]}"}"  # rtrim
    [[ -z "$line" ]] && continue
    PATTERNS+=("$line")
done < "$WORKTREEINCLUDE"

if [[ ${#PATTERNS[@]} -gt 0 ]]; then
    git -C "$WORKTREE_DIR" sparse-checkout set "${PATTERNS[@]}"
else
    echo "[worktree-create] AVISO: .worktreeinclude sem padrões — sparse-checkout não configurado."
fi

# ── Symlink para venv (NÃO recria — satisfaz critério I1.7) ──────────
if [[ -d "$VENV_PATH" ]]; then
    echo "[worktree-create] Criando symlink para venv: $VENV_PATH"
    ln -sf "$VENV_PATH" "$WORKTREE_DIR/Geosteering_AI_venv"
else
    echo "[worktree-create] AVISO: $VENV_PATH não encontrado." >&2
    echo "[worktree-create] Crie o venv com: python3 -m venv ~/Geosteering_AI_venv" >&2
fi

# ── Copiar .env se existir (não versionado, necessário para tokens) ──
if [[ -f "$REPO_ROOT/.env" ]]; then
    echo "[worktree-create] Copiando .env..."
    cp "$REPO_ROOT/.env" "$WORKTREE_DIR/.env"
fi

# ── Sumário ──────────────────────────────────────────────────────────
VENV_LINK="$(readlink "$WORKTREE_DIR/Geosteering_AI_venv" 2>/dev/null || echo 'não configurado')"
echo ""
echo "[worktree-create] ✅ Worktree criada com sucesso:"
echo "  Localização : $WORKTREE_DIR"
echo "  Branch      : $BRANCH"
echo "  Venv        : $VENV_LINK"
echo "  Padrões SC  : ${#PATTERNS[@]} entradas de .worktreeinclude"
echo ""
echo "Para usar:"
echo "  cd $WORKTREE_DIR"
echo "  source Geosteering_AI_venv/bin/activate"
echo "  claude  # abre Claude Code nesta worktree"
echo ""
echo "Para remover quando concluído:"
echo "  git worktree remove $WORKTREE_DIR"
echo "  git branch -d $BRANCH"
