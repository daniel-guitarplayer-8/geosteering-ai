#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK G1: Garantia tatu.x Linux ELF                                ║
# ║  Evento: SessionStart (startup) + invocável manualmente            ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Garante que Fortran_Gerador/tatu.x seja SEMPRE um binário ELF     ║
# ║  Linux executável neste ambiente. O binário é git-tracked e o      ║
# ║  histórico contém uma versão Mach-O (macOS) — operações git        ║
# ║  (checkout, reset, worktree de commit antigo) podem restaurá-la,   ║
# ║  quebrando a paridade Fortran. Este hook detecta Mach-O / ausência ║
# ║  / corrupção e reconstrói via `make` (gfortran).                   ║
# ║                                                                    ║
# ║  Estratégia defense-in-depth:                                      ║
# ║    1. Binário Linux commitado na branch de trabalho (tracked OK)   ║
# ║    2. Este hook reconstrói se detectar não-ELF (rede de segurança) ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -uo pipefail

# Aceita JSON do protocolo de hooks (SessionStart) ou exec direto.
INPUT=$(cat 2>/dev/null || true)

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FORTRAN_DIR="$PROJECT_DIR/Fortran_Gerador"
TATU="$FORTRAN_DIR/tatu.x"

# Se não há diretório Fortran, nada a fazer (permite parar/seguir).
[ ! -d "$FORTRAN_DIR" ] && exit 0

# ── Detecção: tatu.x é ELF Linux executável? ─────────────────────────
needs_rebuild=0
reason=""

if [ ! -f "$TATU" ]; then
    needs_rebuild=1
    reason="ausente"
elif ! file "$TATU" 2>/dev/null | grep -q "ELF"; then
    needs_rebuild=1
    reason="não-ELF ($(file -b "$TATU" 2>/dev/null | cut -c1-30))"
elif [ ! -x "$TATU" ]; then
    # ELF mas sem bit de execução — corrige sem recompilar.
    chmod +x "$TATU" 2>/dev/null || true
fi

if [ "$needs_rebuild" -eq 0 ]; then
    # tatu.x já é ELF Linux válido — silencioso, permite seguir.
    exit 0
fi

# ── Reconstrução via make (gfortran) ─────────────────────────────────
if ! command -v gfortran >/dev/null 2>&1; then
    echo "=== [ensure-tatu-linux] AVISO: tatu.x $reason e gfortran indisponível ===" >&2
    echo "    Instale gfortran (apt install gfortran) ou recompile manualmente." >&2
    # Não bloqueia: testes de paridade farão skip se tatu.x não rodar.
    exit 0
fi

echo "=== [ensure-tatu-linux] tatu.x $reason — reconstruindo via make ===" >&2
if make -C "$FORTRAN_DIR" clean >/dev/null 2>&1 && make -C "$FORTRAN_DIR" >/dev/null 2>&1; then
    if file "$TATU" 2>/dev/null | grep -q "ELF"; then
        echo "    OK: tatu.x reconstruído como ELF Linux." >&2
    else
        echo "    AVISO: make concluiu mas tatu.x ainda não é ELF." >&2
    fi
else
    echo "    ERRO: make falhou. Recompile manualmente em $FORTRAN_DIR." >&2
fi

# SessionStart hooks nunca bloqueiam (exit 0).
exit 0
