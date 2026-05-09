#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK: Política refinada de PyTorch — PreToolUse (Edit|Write)     ║
# ║  Projeto: Geosteering AI v2.0+                                     ║
# ║                                                                    ║
# ║  Origem: §75 do documento de aprofundamento + pré-mortem inaugural ║
# ║  2026-05-09. A regra evoluiu de "PROIBIDO em qualquer parte do     ║
# ║  pipeline" para "PROIBIDO em production paths; PERMITIDO via       ║
# ║  adapter isolado e research/".                                     ║
# ║                                                                    ║
# ║  Bloqueia: `import torch` ou `from torch ...` em arquivos .py      ║
# ║  dentro de production paths abaixo.                                ║
# ║                                                                    ║
# ║  Production paths (BLOQUEIO):                                      ║
# ║    geosteering_ai/{models, losses, training, inference,            ║
# ║                   evaluation, data, simulation, visualization,    ║
# ║                   utils}/                                          ║
# ║                                                                    ║
# ║  Permite (sem bloqueio):                                           ║
# ║    geosteering_ai/adapters/      (adapter PyTorch — Sprint v2.30)  ║
# ║    geosteering_ai/research/      (módulos de pesquisa)             ║
# ║    tests/, benchmarks/, docs/, scripts/, notebooks/                ║
# ║    Arquivos não-Python (.md, .json, etc.)                          ║
# ║                                                                    ║
# ║  Exit codes:                                                       ║
# ║    0 — OK (sem violação OU path permitido OU não-Python)           ║
# ║    2 — BLOQUEIO (import torch direto em production path)           ║
# ║                                                                    ║
# ║  Compatibilidade: macOS (BSD grep -E) + Linux (GNU grep -E).      ║
# ╚══════════════════════════════════════════════════════════════════════╝

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
NEW_STRING=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')

# Pular se nenhum arquivo passado
[ -z "$FILE_PATH" ] && exit 0
[ -z "$NEW_STRING" ] && exit 0

# Pular se não for arquivo Python
[[ "$FILE_PATH" != *.py ]] && exit 0

# Production paths — qualquer arquivo dentro destes é bloqueado se importar torch
PROD_PATHS=(
    "geosteering_ai/models/"
    "geosteering_ai/losses/"
    "geosteering_ai/training/"
    "geosteering_ai/inference/"
    "geosteering_ai/evaluation/"
    "geosteering_ai/data/"
    "geosteering_ai/simulation/"
    "geosteering_ai/visualization/"
    "geosteering_ai/utils/"
)

# Verificar se arquivo está em path de produção
in_prod=false
for p in "${PROD_PATHS[@]}"; do
    if [[ "$FILE_PATH" == *"$p"* ]]; then
        in_prod=true
        break
    fi
done

# Não-produção: permitido sem checagem (adapters/, research/, tests/, etc.)
if [[ "$in_prod" == "false" ]]; then
    exit 0
fi

# Production path: detectar imports diretos de torch no NEW_STRING
# Regex: linha de código (não comentário) com `import torch` ou `from torch ...`
if echo "$NEW_STRING" | grep -qE '^[[:space:]]*(import torch([[:space:]]|$|\.)|from torch[[:space:]]+import)'; then
    cat >&2 <<EOF
[BLOQUEIO] Tentativa de importar PyTorch em production path:
  Arquivo: $FILE_PATH

[INFO] Production paths não podem importar torch diretamente.
[INFO] Use o adapter pattern (Sprint v2.30):
       from geosteering_ai.adapters import get_adapter
       torch_adapter = get_adapter("pytorch")

[INFO] Documentação: §75 do documento de aprofundamento.
[INFO] Política refinada em CLAUDE.md (Proibições Absolutas).
[INFO] Para pesquisa exploratória, use geosteering_ai/research/ (não bloqueado).
EOF
    exit 2
fi

# OK
exit 0
