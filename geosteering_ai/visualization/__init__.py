# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.visualization                                  ║
# ║  Bloco: 9 — Visualization                                                ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Modulos: holdout.py, picasso.py, eda.py, realtime.py                    ║
# ║  Proposito: Visualizacoes para inversao geofisica — holdout, DOD, EDA,   ║
# ║             monitoramento realtime                                         ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (4 modulos)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de visualizacao — holdout, Picasso DOD, EDA, realtime.

Modulos:
    holdout     Comparacao true vs predicted em amostras holdout
    picasso     Mapas Picasso DOD (Depth of Detection) por contraste
    eda         Analise exploratoria: distribuicoes, correlacoes, boxplots
    realtime    Monitoramento ao vivo para inferencia realtime

Todas as funcoes recebem ``config: PipelineConfig`` como parametro opcional.
Matplotlib e importado de forma lazy (dentro de cada funcao) para ambientes
sem display grafico.

Referencia: docs/ARCHITECTURE_v2.md secao 9.
"""

# ──────────────────────────────────────────────────────────────────────
# Imports: holdout.py — comparacao true vs predicted
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.holdout import plot_holdout_samples

# ──────────────────────────────────────────────────────────────────────
# Imports: picasso.py — mapas Picasso DOD
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.picasso import plot_picasso_dod

# ──────────────────────────────────────────────────────────────────────
# Imports: eda.py — analise exploratoria de dados
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.eda import plot_eda_summary

# ──────────────────────────────────────────────────────────────────────
# Imports: realtime.py — monitoramento ao vivo
# ──────────────────────────────────────────────────────────────────────
from geosteering_ai.visualization.realtime import RealtimeMonitor

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente por modulo
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- holdout.py: comparacao true vs predicted ---
    "plot_holdout_samples",
    # --- picasso.py: mapas Picasso DOD ---
    "plot_picasso_dod",
    # --- eda.py: analise exploratoria ---
    "plot_eda_summary",
    # --- realtime.py: monitoramento ao vivo ---
    "RealtimeMonitor",
]
