#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK D: Protecao de Arquivos Criticos                             ║
# ║  Evento: PreToolUse (Edit|Write)                                   ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Dois niveis de protecao:                                          ║
# ║    BLOQUEIO TOTAL: modelos treinados (.keras, .h5), .env,          ║
# ║      __pycache__, node_modules                                     ║
# ║    AUDITORIA: config.py, pyproject.toml, CI workflows,             ║
# ║      baseline.yaml — permite edicao mas registra em log            ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0

# ── Arquivos completamente protegidos (BLOQUEIO) ────────────────────
case "$FILE_PATH" in
    *.keras|*.h5|*.hdf5)
        echo "PROTEGIDO: Modelos treinados ($FILE_PATH) nao podem ser editados." >&2
        exit 2
        ;;
    *.env|*.env.local|*.env.production)
        echo "PROTEGIDO: Arquivos de credenciais ($FILE_PATH) nao podem ser editados." >&2
        exit 2
        ;;
    *__pycache__/*|*.pyc)
        echo "PROTEGIDO: Cache Python ($FILE_PATH) nao deve ser editado." >&2
        exit 2
        ;;
    *.pytest_cache/*|*.mypy_cache/*)
        echo "PROTEGIDO: Cache de ferramentas ($FILE_PATH) nao deve ser editado." >&2
        exit 2
        ;;
    *node_modules/*)
        echo "PROTEGIDO: Dependencias externas ($FILE_PATH) nao devem ser editadas." >&2
        exit 2
        ;;
esac

# ── Arquivos criticos (AUDITORIA — permite, mas loga) ───────────────
AUDIT_LOG="${HOME}/.claude/critical-edits.log"

case "$FILE_PATH" in
    *geosteering_ai/config.py)
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') EDIT config.py (PipelineConfig)" >> "$AUDIT_LOG"
        echo "AUDITORIA: config.py eh o ponto unico de verdade do pipeline. Edicao registrada." >&2
        ;;
    *pyproject.toml)
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') EDIT pyproject.toml (dependencias)" >> "$AUDIT_LOG"
        echo "AUDITORIA: pyproject.toml controla dependencias do pacote. Edicao registrada." >&2
        ;;
    *configs/baseline.yaml)
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') EDIT configs/baseline.yaml (preset canonico)" >> "$AUDIT_LOG"
        echo "AUDITORIA: baseline.yaml eh o preset canonico. Edicao registrada." >&2
        ;;
    *.github/workflows/*)
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') EDIT CI workflow: $(basename "$FILE_PATH")" >> "$AUDIT_LOG"
        echo "AUDITORIA: Workflow de CI/CD editado. Edicao registrada." >&2
        ;;
    *tests/test_config.py)
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') EDIT tests/test_config.py (testes de errata)" >> "$AUDIT_LOG"
        echo "AUDITORIA: Testes de errata editados. Edicao registrada." >&2
        ;;
esac

exit 0