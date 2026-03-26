# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUBPACOTE: geosteering_ai.noise                                           ║
# ║  Bloco: 2c — Noise On-The-Fly                                             ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║                                                                            ║
# ║  Modulos: functions.py, curriculum.py                                     ║
# ║  Cadeia: raw EM → noise(σ) → FV → GS → scale → modelo                   ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4.5 (noise)                          ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Subpacote de noise — funcoes on-the-fly e curriculum schedule.

Implementa a injecao de ruido no pipeline de dados para robustez
do modelo a medidas LWD reais com ruido eletronico e ambiental.

Modulos:
    functions     4 tipos de noise TF (gaussian, multiplicative, uniform,
                  dropout) + dispatcher multi-tipo + versao numpy offline
    curriculum    CurriculumSchedule (3 fases: clean→ramp→stable),
                  UpdateNoiseLevelCallback (Keras callback)

Principios:
    - Noise aplicado ANTES de FV e GS (fidelidade fisica LWD)
    - z_obs (col 0) NUNCA recebe noise
    - tf.Variable compartilhado entre callback e tf.data.map
    - Curriculum: clean → ramp linear → stable no maximo

Referencia: docs/ARCHITECTURE_v2.md secao 4.5, skill secao 7.3.
"""

# ── Noise functions (functions.py) ────────────────────────────────────────
from geosteering_ai.noise.functions import (
    NOISE_FN_MAP,
    VALID_NOISE_TYPES,
    apply_noise_tf,
    apply_raw_em_noise,
    create_noise_level_var,
)

# ── Curriculum schedule (curriculum.py) ───────────────────────────────────
from geosteering_ai.noise.curriculum import (
    CurriculumSchedule,
    UpdateNoiseLevelCallback,
    compute_noise_level,
)

# ── D8: Exports publicos ─────────────────────────────────────────────────
__all__ = [
    # --- functions.py ---
    "NOISE_FN_MAP",
    "VALID_NOISE_TYPES",
    "create_noise_level_var",
    "apply_noise_tf",
    "apply_raw_em_noise",
    # --- curriculum.py ---
    "CurriculumSchedule",
    "compute_noise_level",
    "UpdateNoiseLevelCallback",
]
