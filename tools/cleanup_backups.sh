#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TOOL: cleanup_backups.sh                                          ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Politica de retencao de backups (.backups/):                      ║
# ║    - Ultimas 24h: TODOS os backups preservados                     ║
# ║    - 1 a 7 dias: 1 backup por arquivo por dia (mais recente)       ║
# ║    - Mais de 7 dias: removidos (git e o backup permanente)         ║
# ║                                                                    ║
# ║  Uso:                                                              ║
# ║    tools/cleanup_backups.sh [--dry-run] [--backups-dir DIR]        ║
# ║                                                                    ║
# ║  Flags:                                                            ║
# ║    --dry-run     Lista arquivos que seriam removidos (nao remove)  ║
# ║    --backups-dir Diretorio de backups (default: .backups/)         ║
# ║                                                                    ║
# ║  Uso recomendado: executar semanalmente via cron ou manualmente.   ║
# ║  Nao configurar como hook automatico (pode ser lento em repos      ║
# ║  grandes com muitos backups acumulados).                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

# ── Parseamento de argumentos ────────────────────────────────────────────────
DRY_RUN=0
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
BACKUPS_DIR="$PROJECT_DIR/.backups"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --backups-dir) BACKUPS_DIR="$2"; shift ;;
        *) echo "Uso: $0 [--dry-run] [--backups-dir DIR]" >&2; exit 1 ;;
    esac
    shift
done

if [ ! -d "$BACKUPS_DIR" ]; then
    echo "[cleanup_backups] Diretorio de backups nao encontrado: $BACKUPS_DIR" >&2
    exit 0
fi

# ── Calcular timestamps de corte ────────────────────────────────────────────
# macOS usa BSD date; Linux usa GNU date — compatibilidade com ambos
if date -v-1d +%Y-%m-%d >/dev/null 2>&1; then
    # BSD (macOS)
    CUTOFF_24H=$(date -v-1d +%Y-%m-%d)
    CUTOFF_7D=$(date -v-7d +%Y-%m-%d)
else
    # GNU (Linux)
    CUTOFF_24H=$(date -d "1 day ago" +%Y-%m-%d)
    CUTOFF_7D=$(date -d "7 days ago" +%Y-%m-%d)
fi

TODAY=$(date +%Y-%m-%d)

echo "[cleanup_backups] Iniciando limpeza em: $BACKUPS_DIR"
echo "[cleanup_backups] Politica:"
echo "  - < 24h (hoje/$CUTOFF_24H): preservar todos"
echo "  - 1-7 dias ($CUTOFF_24H a $CUTOFF_7D): manter 1 por arquivo/dia"
echo "  - > 7 dias (antes de $CUTOFF_7D): remover tudo"
[ "$DRY_RUN" = "1" ] && echo "[cleanup_backups] MODO DRY-RUN: nao remove nada"
echo ""

REMOVED=0
KEPT=0
TOTAL=0

# ── Processar cada diretorio de data ─────────────────────────────────────────
for DAY_DIR in "$BACKUPS_DIR"/*/; do
    [ -d "$DAY_DIR" ] || continue
    DAY=$(basename "$DAY_DIR")

    # Validar formato YYYY-MM-DD
    if ! echo "$DAY" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'; then
        continue
    fi

    # Ultimas 24h (hoje ou ontem): preservar TODOS os backups.
    # Nota: comparacao por data-calendario — politica real e "2 dias completos"
    # (hoje + ontem), nao exatamente 24h. Backups de CUTOFF_24H (ontem) tambem
    # sao preservados (>= em vez de >).
    if [ "$DAY" = "$TODAY" ] || [ "$DAY" \>= "$CUTOFF_24H" ]; then
        COUNT=$(find "$DAY_DIR" -name "*.bak" -type f 2>/dev/null | wc -l | tr -d ' ')
        KEPT=$((KEPT + COUNT))
        TOTAL=$((TOTAL + COUNT))
        continue
    fi

    # Mais de 7 dias: remover TUDO
    if [ "$DAY" \< "$CUTOFF_7D" ]; then
        while IFS= read -r BAK; do
            TOTAL=$((TOTAL + 1))
            REMOVED=$((REMOVED + 1))
            if [ "$DRY_RUN" = "1" ]; then
                echo "  [DRY] remover: $BAK" >&2
            else
                rm -f "$BAK"
                echo "  remover: $BAK" >&2
            fi
        done < <(find "$DAY_DIR" -name "*.bak" -type f 2>/dev/null)

        # Remover diretorio vazio
        if [ "$DRY_RUN" = "0" ]; then
            find "$DAY_DIR" -type d -empty -delete 2>/dev/null || true
        fi
        continue
    fi

    # 1-7 dias: manter apenas 1 backup por arquivo (mais recente)
    # Agrupar por nome-base do arquivo (.bak removido, timestamp removido)
    # Exemplo: geosteering_ai/simulation/forward.py.142233.bak
    #          → base = geosteering_ai/simulation/forward.py

    declare -A SEEN_FILES
    while IFS= read -r BAK; do
        TOTAL=$((TOTAL + 1))
        BASENAME=$(basename "$BAK" | sed 's/\.[0-9]\{6\}\.bak$//')
        RELDIR="${BAK#$DAY_DIR}"
        RELDIR=$(dirname "$RELDIR")
        FILE_KEY="${RELDIR}/${BASENAME}"

        if [ -z "${SEEN_FILES[$FILE_KEY]+x}" ]; then
            # Primeiro encontrado = mais recente (find -name sort nao garante ordem;
            # usar ls -t para garantir)
            SEEN_FILES[$FILE_KEY]="$BAK"
        fi
    done < <(find "$DAY_DIR" -name "*.bak" -type f 2>/dev/null | sort -r)

    # Segunda passagem: remover duplicatas (nao o mais recente)
    while IFS= read -r BAK; do
        BASENAME=$(basename "$BAK" | sed 's/\.[0-9]\{6\}\.bak$//')
        RELDIR="${BAK#$DAY_DIR}"
        RELDIR=$(dirname "$RELDIR")
        FILE_KEY="${RELDIR}/${BASENAME}"

        if [ "${SEEN_FILES[$FILE_KEY]}" = "$BAK" ]; then
            KEPT=$((KEPT + 1))
        else
            REMOVED=$((REMOVED + 1))
            if [ "$DRY_RUN" = "1" ]; then
                echo "  [DRY] deduplicar: $BAK"
            else
                rm -f "$BAK"
            fi
        fi
    done < <(find "$DAY_DIR" -name "*.bak" -type f 2>/dev/null | sort -r)

    unset SEEN_FILES
    declare -A SEEN_FILES
done

echo ""
echo "[cleanup_backups] Resumo: $TOTAL arquivos analisados | $KEPT mantidos | $REMOVED removidos"
if [ "$DRY_RUN" = "1" ]; then
    echo "[cleanup_backups] DRY-RUN: execute sem --dry-run para aplicar."
fi
