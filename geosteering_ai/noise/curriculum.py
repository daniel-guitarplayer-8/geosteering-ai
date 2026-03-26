# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: noise/curriculum.py                                               ║
# ║  Bloco: 2c — Noise On-The-Fly                                             ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • CurriculumSchedule: encapsula logica de 3 fases do curriculum        ║
# ║    • compute_noise_level(): funcao pura epoch→noise_level                 ║
# ║    • UpdateNoiseLevelCallback: Keras callback para atualizar tf.Variable  ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), noise/functions.py             ║
# ║  Exports: ~3 simbolos — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4.5 (noise), skill secao 7.3         ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CURRICULUM SCHEDULE (3 FASES)
# ════════════════════════════════════════════════════════════════════════════
# Encapsula a logica do curriculum learning em 3 fases:
#   Fase 1 (Clean):  epoch < epochs_no_noise       → noise = 0.0
#   Fase 2 (Ramp):   epochs_no_noise ≤ epoch < end → noise linear 0→max
#   Fase 3 (Stable): epoch ≥ end                   → noise = max
#
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │  Curriculum Noise 3-Phase Schedule                                   │
#   │                                                                      │
#   │  noise_level                                                         │
#   │  ▲                                                                   │
#   │  │            ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾  Fase 3: Stable                  │
#   │  max ........╱                      (noise_level_max)               │
#   │  │         ╱                                                         │
#   │  │       ╱   Fase 2: Ramp                                           │
#   │  │     ╱     (linear 0 → max)                                       │
#   │  │   ╱                                                               │
#   │  0 ──┘                                                               │
#   │  │ Fase 1: Clean                                                     │
#   │  │ (noise = 0.0)                                                     │
#   │  └──┬────────┬────────────────────┬──────────────── epoch            │
#   │     0   epochs_no_noise     end_ramp                                 │
#   │              (10)          (10+80=90)                                 │
#   └──────────────────────────────────────────────────────────────────────┘
#
# Ref: Skill geosteering-v2 secao 7.3, legado C40 UpdateNoiseLevelCallback.
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CurriculumSchedule:
    """Schedule de curriculum noise em 3 fases.

    Encapsula os parametros do curriculum e fornece o metodo
    ``get_level(epoch)`` para calcular o noise level em qualquer
    epoca. Imutavel (frozen) para seguranca.

    Fase 1 (Clean): epoch < epochs_no_noise → level = 0.0
    Fase 2 (Ramp):  linear 0 → noise_level_max
    Fase 3 (Stable): level = noise_level_max

    Attributes:
        noise_level_max (float): Nivel maximo de noise (sigma).
            Range: [0, 1]. Default do config: 0.08 (E-Robusto).
        epochs_no_noise (int): Duracao da fase clean em epocas.
            Default do config: 10.
        noise_ramp_epochs (int): Duracao da fase ramp em epocas.
            Default do config: 80.

    Example:
        >>> from geosteering_ai.noise.curriculum import CurriculumSchedule
        >>> sched = CurriculumSchedule(noise_level_max=0.08, epochs_no_noise=10, noise_ramp_epochs=80)
        >>> sched.get_level(0)    # 0.0 (clean)
        >>> sched.get_level(50)   # 0.04 (ramp)
        >>> sched.get_level(100)  # 0.08 (stable)

    Note:
        Referenciado em:
            - noise/curriculum.py: UpdateNoiseLevelCallback
            - training/nstage.py: N-Stage curriculum per stage
            - tests/test_noise.py: TestCurriculum
        Ref: docs/ARCHITECTURE_v2.md secao 4.5.
        Construido a partir de PipelineConfig via .from_config().
    """

    noise_level_max: float
    epochs_no_noise: int
    noise_ramp_epochs: int

    @classmethod
    def from_config(cls, config: PipelineConfig) -> CurriculumSchedule:
        """Constroi schedule a partir de PipelineConfig.

        Args:
            config: Configuracao do pipeline. Atributos usados:
                - config.noise_level_max
                - config.epochs_no_noise
                - config.noise_ramp_epochs

        Returns:
            CurriculumSchedule configurado.

        Note:
            Referenciado em: training/loop.py (setup do curriculum).
        """
        return cls(
            noise_level_max=config.noise_level_max,
            epochs_no_noise=config.epochs_no_noise,
            noise_ramp_epochs=config.noise_ramp_epochs,
        )

    @property
    def end_ramp_epoch(self) -> int:
        """Epoca onde a fase ramp termina e stable comeca."""
        return self.epochs_no_noise + self.noise_ramp_epochs

    def get_level(self, epoch: int) -> float:
        """Calcula noise level para uma epoca.

        Funcao pura (sem side effects). Retorna o noise level
        que deve ser usado na epoca informada, seguindo o
        schedule de 3 fases.

        Args:
            epoch: Numero da epoca (0-indexed).

        Returns:
            float: Noise level (sigma) para a epoca.
                Range: [0.0, noise_level_max].
        """
        if epoch < self.epochs_no_noise:
            # ── Fase 1 (Clean): modelo aprende mapeamento limpo ───────
            return 0.0
        elif epoch < self.end_ramp_epoch:
            # ── Fase 2 (Ramp): adaptacao gradual ao ruido ────────────
            progress = (epoch - self.epochs_no_noise) / max(self.noise_ramp_epochs, 1)
            return self.noise_level_max * progress
        else:
            # ── Fase 3 (Stable): treinamento no nivel maximo ─────────
            return self.noise_level_max

    def get_phase(self, epoch: int) -> str:
        """Retorna nome da fase para uma epoca.

        Args:
            epoch: Numero da epoca (0-indexed).

        Returns:
            str: "clean", "ramp", ou "stable".
        """
        if epoch < self.epochs_no_noise:
            return "clean"
        elif epoch < self.end_ramp_epoch:
            return "ramp"
        else:
            return "stable"


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCAO PURA compute_noise_level
# ════════════════════════════════════════════════════════════════════════════
# Funcao standalone para calcular noise level sem instanciar
# CurriculumSchedule. Util em contextos simples (testes, scripts).
# ──────────────────────────────────────────────────────────────────────────


def compute_noise_level(
    epoch: int,
    noise_level_max: float,
    epochs_no_noise: int,
    noise_ramp_epochs: int,
) -> float:
    """Calcula noise level para uma epoca (funcao pura).

    Implementacao standalone da logica de 3 fases. Para uso
    repetido, prefira CurriculumSchedule.get_level().

    Args:
        epoch: Numero da epoca (0-indexed).
        noise_level_max: Nivel maximo (sigma).
        epochs_no_noise: Duracao da fase clean.
        noise_ramp_epochs: Duracao da fase ramp.

    Returns:
        float: Noise level para a epoca. Range: [0.0, noise_level_max].

    Example:
        >>> from geosteering_ai.noise.curriculum import compute_noise_level
        >>> compute_noise_level(epoch=50, noise_level_max=0.08,
        ...                    epochs_no_noise=10, noise_ramp_epochs=80)
        0.04

    Note:
        Referenciado em: testes, scripts de analise.
        Equivalente funcional a CurriculumSchedule.get_level().
    """
    schedule = CurriculumSchedule(
        noise_level_max=noise_level_max,
        epochs_no_noise=epochs_no_noise,
        noise_ramp_epochs=noise_ramp_epochs,
    )
    return schedule.get_level(epoch)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: KERAS CALLBACK
# ════════════════════════════════════════════════════════════════════════════
# UpdateNoiseLevelCallback atualiza o tf.Variable compartilhado
# no inicio de cada epoca, seguindo o CurriculumSchedule.
# Quando use_curriculum=False, mantem noise constante em noise_level_max.
# Ref: Legado C40 UpdateNoiseLevelCallback, adaptado para v2.0.
# ──────────────────────────────────────────────────────────────────────────


class UpdateNoiseLevelCallback:
    """Keras callback que atualiza noise level por epoca.

    Atualiza o tf.Variable compartilhado no inicio de cada epoca
    seguindo o CurriculumSchedule (3 fases). Quando curriculum
    esta desativado (use_curriculum=False), mantem noise constante
    no valor noise_level_max desde a epoca 0.

    Herda de tf.keras.callbacks.Callback (importacao lazy para
    evitar dependencia de TF em testes CPU-only).

    Attributes:
        noise_var (tf.Variable): Variable compartilhado com pipeline.
        schedule (CurriculumSchedule): Schedule de 3 fases.
        use_curriculum (bool): Se False, noise constante.

    Example:
        >>> from geosteering_ai.noise import create_noise_level_var
        >>> from geosteering_ai.noise.curriculum import (
        ...     CurriculumSchedule, UpdateNoiseLevelCallback)
        >>> noise_var = create_noise_level_var(0.0)
        >>> sched = CurriculumSchedule(0.08, 10, 80)
        >>> cb = UpdateNoiseLevelCallback(noise_var, sched, use_curriculum=True)

    Note:
        Referenciado em:
            - training/callbacks.py: build_callbacks() inclui este callback
            - training/loop.py: model.fit(callbacks=[...])
        Ref: Legado C40 UpdateNoiseLevelCallback.
        Quando use_curriculum=False, noise_var.assign(noise_level_max)
        desde a epoca 0 (noise constante, sem rampa).
    """

    def __init__(
        self,
        noise_var: "tf.Variable",
        schedule: CurriculumSchedule,
        use_curriculum: bool = True,
        log: Optional[logging.Logger] = None,
    ) -> None:
        """Inicializa callback.

        Args:
            noise_var: tf.Variable escalar compartilhado.
            schedule: CurriculumSchedule com parametros das 3 fases.
            use_curriculum: Se True, aplica schedule 3-fases.
                Se False, noise constante em schedule.noise_level_max.
            log: Logger opcional.
        """
        self.noise_var = noise_var
        self.schedule = schedule
        self.use_curriculum = use_curriculum
        self._log = log or logger

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Atualiza noise level no inicio da epoca.

        Args:
            epoch: Numero da epoca (0-indexed).
            logs: Dict de logs do Keras (ignorado).
        """
        if self.use_curriculum:
            level = self.schedule.get_level(epoch)
            phase = self.schedule.get_phase(epoch)
        else:
            # ── Noise constante (sem curriculum) ──────────────────────
            level = self.schedule.noise_level_max
            phase = "constant"

        self.noise_var.assign(level)
        self._log.debug(
            "Epoch %d: noise_level=%.6f (phase=%s)", epoch, level, phase,
        )


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Schedule ──────────────────────────────────────────────────────
    "CurriculumSchedule",
    "compute_noise_level",
    # ── Callback ──────────────────────────────────────────────────────
    "UpdateNoiseLevelCallback",
]
