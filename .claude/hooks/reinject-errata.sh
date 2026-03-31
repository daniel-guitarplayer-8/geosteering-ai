#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HOOK E2: Injecao de Contexto — Reinjecao apos Compactacao        ║
# ║  Evento: SessionStart (compact)                                    ║
# ║  Projeto: Geosteering AI v2.0                                      ║
# ║                                                                    ║
# ║  Apos compactacao de contexto (sessoes longas), reinjetar:         ║
# ║    1. Restricoes criticas da Errata (defaults validados)           ║
# ║    2. Proibicoes absolutas do projeto                              ║
# ║    3. Padroes obrigatorios v2.0                                    ║
# ║    4. Ultimos 5 commits (estado recente)                           ║
# ║                                                                    ║
# ║  Impede que Claude "esqueca" restricoes apos compactacao.          ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -uo pipefail

INPUT=$(cat)
SOURCE=$(echo "$INPUT" | jq -r '.source // ""')

# Apenas apos compactacao
[ "$SOURCE" != "compact" ] && exit 0

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"

# ── Reinjetar restricoes criticas ────────────────────────────────────
cat << 'CONTEXT'
=== GEOSTEERING AI v2.0 — CONTEXTO RESTAURADO APOS COMPACTACAO ===

DEFAULTS VALIDADOS (ranges aceitos pelo __post_init__):
  frequency_hz = 20000.0   (default 20 kHz — range [100, 1e6] Hz)
  spacing_meters = 1.0     (default — range [0.1, 10.0] m)
  sequence_length = 600    (default Inv0Dip 0 graus — range [10, 100000])
  target_scaling = "log10" (escala targets — NUNCA "log")
  input_features = [1,4,5,20,21]  (22-col — NUNCA [0,3,4,7,8])
  output_targets = [2,3]   (22-col — NUNCA [1,2])
  eps_tf >= 1e-15           (float32 safe — default 1e-12, NUNCA 1e-30)

PROIBICOES ABSOLUTAS:
  - PyTorch (import torch) — Framework exclusivo: TensorFlow/Keras
  - globals().get() em geosteering_ai/ — Usar config.param via PipelineConfig
  - print() em geosteering_ai/ — Usar logging (logger.info/debug/warning)
  - Split por amostra — SEMPRE split por modelo geologico (P1)
  - Scaler fit em dados ruidosos — SEMPRE fit em dados LIMPOS
  - Noise offline com GS — on-the-fly eh o UNICO path fisicamente correto

PADROES OBRIGATORIOS:
  - Toda funcao recebe config: PipelineConfig (NUNCA globals)
  - Factory/Registry para componentes (ModelRegistry, LossFactory)
  - Docstrings Google-style (D5/D6) e mega-header D1 em todo modulo
  - pytest tests/ -v --tb=short antes de qualquer commit

CONTEXT

# Estado recente do git
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "Commits recentes:"
    cd "$PROJECT_DIR" && git log --oneline -5 2>/dev/null
    echo ""
    echo "Arquivos modificados:"
    cd "$PROJECT_DIR" && git diff --name-only 2>/dev/null | head -10
fi

exit 0