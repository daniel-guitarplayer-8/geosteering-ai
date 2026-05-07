#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: run-fortran-parity.sh                                       ║
# ║  Evento: PostToolUse (Edit|Write)                                  ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Após edição em _numba/*, forward.py ou multi_forward.py, valida   ║
# ║  paridade Fortran <1e-12 nos modelos canônicos. Se quebrar, alerta ║
# ║  o usuário com instruções de rollback via .backups/.               ║
# ║                                                                    ║
# ║  Filtragem por path evita executar pytest em qualquer edit.        ║
# ║  Timeout do hook: 120s (suficiente para -k fortran_python_numba).  ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
[ -z "$FILE_PATH" ] && exit 0

# Filtro: apenas arquivos críticos ao kernel JIT
case "$FILE_PATH" in
    *_numba/* | *forward.py | *multi_forward.py)
        ;;
    *)
        exit 0
        ;;
esac

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
cd "$PROJECT_DIR"

# Ativar venv (silencioso se não existir)
if [ -f ~/Geosteering_AI_venv/bin/activate ]; then
    # shellcheck disable=SC1090
    source ~/Geosteering_AI_venv/bin/activate
fi

echo "[fortran-parity] verificando paridade <1e-12 em modelos canônicos…" >&2

# PYTHONPATH=. necessário para imports relativos (tests/_fortran_helpers)
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Rodar paridade Fortran (modelos canônicos críticos)
# Nota: --timeout removido (pytest-timeout não é dep obrigatória).
# Hook timeout de 120s no settings.json é suficiente.
if pytest tests/test_simulation_compare_fortran.py \
         -k fortran_python_numba \
         --tb=short \
         -q 2>&1 | tail -10 >&2; then
    echo "[fortran-parity] OK — paridade preservada (<1e-12)" >&2
    exit 0
else
    echo "" >&2
    echo "============================================================" >&2
    echo "[fortran-parity] FALHA — PARIDADE FORTRAN QUEBRADA!" >&2
    echo "============================================================" >&2
    echo "Arquivo modificado: $FILE_PATH" >&2
    echo "Restaure com: cp .backups/$(date +%Y-%m-%d)/<rel>.<HHMMSS>.bak <arquivo>" >&2
    echo "ou git checkout HEAD -- $FILE_PATH" >&2
    echo "============================================================" >&2
    exit 1
fi
