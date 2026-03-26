# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: utils/__init__.py                                                 ║
# ║  Bloco: 5 — Utilitarios                                                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Re-exportar simbolos publicos dos 6 submodulos de utils/             ║
# ║    • API unificada: from geosteering_ai.utils import get_logger, ...      ║
# ║    • Agrupar semanticamente: logging, tempo, validacao, formatacao,       ║
# ║      sistema, I/O                                                          ║
# ║                                                                            ║
# ║  Dependencias: utils/logger, utils/timer, utils/validation,               ║
# ║                utils/formatting, utils/system, utils/io                    ║
# ║  Exports: ~30 simbolos — ver __all__                                      ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9 (utils/)                           ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ════════════════════════════════════════════════════════════════════════════
# SECAO: RE-EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Importacoes centralizadas de todos os submodulos de utils/.
# Permite uso direto: from geosteering_ai.utils import get_logger
# Agrupados semanticamente por submodulo de origem.
# ──────────────────────────────────────────────────────────────────────────

# ── Logging (utils/logger.py) ─────────────────────────────────────────────
from geosteering_ai.utils.logger import (
    C,
    ColoredFormatter,
    get_logger,
    setup_logger,
)

# ── Tempo (utils/timer.py) ────────────────────────────────────────────────
from geosteering_ai.utils.timer import (
    ProgressTracker,
    elapsed_since,
    format_time,
    timer_decorator,
)

# ── Validacao (utils/validation.py) ───────────────────────────────────────
from geosteering_ai.utils.validation import (
    ValidationTracker,
    validate_shape,
)

# ── Formatacao (utils/formatting.py) ──────────────────────────────────────
from geosteering_ai.utils.formatting import (
    colorize_flag_value,
    format_bytes,
    format_compact,
    format_number,
    log_flag_block,
    log_header,
    log_section,
)

# ── Sistema (utils/system.py) ─────────────────────────────────────────────
from geosteering_ai.utils.system import (
    GLOBAL_SEED,
    clear_memory,
    detect_environment,
    ensure_dirs,
    get_environment_info,
    gpu_memory_info,
    has_gpu,
    is_colab,
    is_jupyter,
    is_kaggle,
    memory_usage,
    safe_mkdir,
    set_all_seeds,
)

# ── I/O (utils/io.py) ────────────────────────────────────────────────────
from geosteering_ai.utils.io import (
    NumpyEncoder,
    safe_json_dump,
    safe_json_load,
)


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos re-exportados.
# Agrupados semanticamente por submodulo de origem.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Logging ───────────────────────────────────────────────────────
    "C",
    "ColoredFormatter",
    "setup_logger",
    "get_logger",
    # ── Tempo ─────────────────────────────────────────────────────────
    "format_time",
    "elapsed_since",
    "timer_decorator",
    "ProgressTracker",
    # ── Validacao ─────────────────────────────────────────────────────
    "ValidationTracker",
    "validate_shape",
    # ── Formatacao ────────────────────────────────────────────────────
    "format_number",
    "format_compact",
    "format_bytes",
    "log_header",
    "log_section",
    "colorize_flag_value",
    "log_flag_block",
    # ── Sistema ───────────────────────────────────────────────────────
    "GLOBAL_SEED",
    "is_colab",
    "is_kaggle",
    "is_jupyter",
    "has_gpu",
    "detect_environment",
    "get_environment_info",
    "safe_mkdir",
    "ensure_dirs",
    "memory_usage",
    "gpu_memory_info",
    "clear_memory",
    "set_all_seeds",
    # ── I/O ───────────────────────────────────────────────────────────
    "NumpyEncoder",
    "safe_json_dump",
    "safe_json_load",
]
