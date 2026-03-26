# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: inference/realtime.py                                             ║
# ║  Bloco: 7 — Inference                                                    ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • RealtimeInference: sliding window para geosteering em tempo real     ║
# ║    • Buffer circular de medicoes com tamanho = SEQUENCE_LENGTH (600)      ║
# ║    • Inferencia a cada nova medicao (on-arrival)                          ║
# ║    • Compoe InferencePipeline para cadeia FV+GS+scale+predict            ║
# ║                                                                            ║
# ║  Dependencias: inference/pipeline.py (InferencePipeline)                  ║
# ║  Exports: ~1 classe (RealtimeInference)                                   ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6.2                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""RealtimeInference — Sliding window para geosteering em tempo real.

Mantem buffer circular de medicoes e executa inferencia a cada nova
chegada. Projetado para cenarios LWD onde medicoes chegam
sequencialmente durante perfuracao.

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────┐
    │  SLIDING WINDOW — Inferencia Realtime                               │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Medicoes LWD chegam sequencialmente (1 por ponto de medicao):      │
    │                                                                      │
    │    t=0   t=1   t=2   ...   t=599  t=600  t=601  ...                │
    │    [m0]  [m1]  [m2]  ...   [m599]                                   │
    │    └─────────── buffer (600) ──────────────┘                        │
    │                              → predict()                             │
    │                                                                      │
    │    Apos t=600 (buffer cheio):                                       │
    │    [m1]  [m2]  [m3]  ...   [m600]                                   │
    │    └─────────── buffer (600) ──────────────┘                        │
    │                              → predict()                             │
    │                                                                      │
    │  Buffer circular: FIFO, descarta medicao mais antiga ao encher.     │
    │  Predicao retornada para CADA update apos buffer cheio.             │
    └──────────────────────────────────────────────────────────────────────┘

Referencia: docs/ARCHITECTURE_v2.md secao 6.2.

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Referenciado em:
        - inference/__init__.py: re-exportado como API publica
        - tests/test_inference.py: TestRealtimeInference
    Ref: docs/ARCHITECTURE_v2.md secao 6.2 (RealtimeInference).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# D8: Exports publicos deste modulo
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    "RealtimeInference",
]


# ════════════════════════════════════════════════════════════════════════
# REALTIME INFERENCE — Sliding window para geosteering
#
# Utiliza buffer circular (collections.deque com maxlen) para manter
# as ultimas `window_size` medicoes. Quando o buffer atinge capacidade
# completa, cada nova medicao desloca a mais antiga (FIFO) e dispara
# uma inferencia via InferencePipeline.predict().
#
# window_size = SEQUENCE_LENGTH = 600 (Errata v4.4.5 — NUNCA 601)
#
# Cenario tipico: ferramenta LWD (Logging While Drilling) envia
# medicoes EM a cada ponto de profundidade durante perfuracao.
# O geosteerer recebe predicoes de resistividade em tempo real
# para tomar decisoes de trajetoria.
# ════════════════════════════════════════════════════════════════════════

class RealtimeInference:
    """Sliding window para inferencia em tempo real de geosteering.

    Mantem buffer circular das ultimas `window_size` medicoes e executa
    inferencia a cada nova medicao via InferencePipeline. Projetado para
    cenarios LWD onde dados chegam sequencialmente.

    Attributes:
        pipeline: InferencePipeline com modelo treinado e scalers.
        window_size: Tamanho da janela deslizante (default 600 = SEQUENCE_LENGTH).
        buffer: deque circular com capacidade maxima window_size.
        n_updates: Contador de medicoes recebidas desde ultimo reset.

    Example:
        >>> from geosteering_ai.inference import InferencePipeline, RealtimeInference
        >>>
        >>> pipeline = InferencePipeline.load("/path/to/trained_pipeline")
        >>> rt = RealtimeInference(pipeline, window_size=600)
        >>>
        >>> for measurement in stream_of_measurements:
        ...     result = rt.update(measurement)
        ...     if result is not None:
        ...         print(f"Predicao: rho_h={result[0, -1, 0]:.2f} Ohm.m")

    Note:
        Referenciado em:
            - inference/__init__.py: re-exportado como API publica
            - tests/test_inference.py: TestRealtimeInference
        Ref: docs/ARCHITECTURE_v2.md secao 6.2 (RealtimeInference).
        window_size DEVE ser 600 (SEQUENCE_LENGTH, Errata v4.4.5).
        Buffer implementado via collections.deque(maxlen=window_size).
        Retorna None ate buffer estar cheio (preenchimento inicial).
    """

    # ────────────────────────────────────────────────────────────────
    # D2: Inicializacao — configura buffer circular e pipeline
    # ────────────────────────────────────────────────────────────────

    def __init__(
        self,
        pipeline: InferencePipeline,
        window_size: int = 600,
    ) -> None:
        """Inicializa RealtimeInference com pipeline e tamanho de janela.

        Args:
            pipeline: InferencePipeline com modelo e scalers carregados.
                O modelo deve estar pronto para predicao (model != None).
            window_size: Tamanho da janela deslizante em numero de medicoes.
                Default: 600 (= SEQUENCE_LENGTH). Cada medicao e um vetor
                de 22 valores (formato 22-colunas padrao do pipeline).

        Raises:
            ValueError: Se window_size <= 0 ou pipeline for None.

        Note:
            Referenciado em:
                - RealtimeInference.update(): usa self.buffer e self.pipeline
                - RealtimeInference.reset(): reinicializa self.buffer
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
        """
        if pipeline is None:
            raise ValueError("pipeline nao pode ser None")
        if window_size <= 0:
            raise ValueError(
                f"window_size deve ser > 0, recebido: {window_size}"
            )

        self.pipeline = pipeline
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)
        self.n_updates: int = 0

        logger.info(
            "RealtimeInference inicializado — window_size=%d, "
            "model_type=%s",
            window_size,
            pipeline.config.model_type,
        )

    # ────────────────────────────────────────────────────────────────
    # D2: Update — adiciona medicao e executa inferencia
    # ────────────────────────────────────────────────────────────────

    def update(
        self,
        measurement: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Adiciona nova medicao ao buffer e executa inferencia.

        Cada medicao e um vetor 1D de 22 valores (formato 22-colunas).
        Quando o buffer atinge window_size, cada chamada a update()
        dispara uma inferencia e retorna a predicao.

        Args:
            measurement: np.ndarray de shape (22,) com uma unica medicao
                no formato 22-colunas padrao. Os 22 valores correspondem
                ao mapeamento COL_MAP_22 (z, Re/Im EM, targets).

        Returns:
            np.ndarray de shape (1, window_size, n_targets) com predicoes
            em Ohm.m se buffer estiver cheio. None se buffer ainda nao
            atingiu window_size (fase de preenchimento inicial).

        Raises:
            ValueError: Se measurement nao tiver shape (22,) ou
                shape (n_columns,) conforme config.n_columns.

        Note:
            Referenciado em:
                - tests/test_inference.py: TestRealtimeInference.test_update_returns_none,
                  TestRealtimeInference.test_update_returns_prediction
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Internamente: acumula no buffer, monta array 3D, chama
            pipeline.predict() quando len(buffer) == window_size.
        """
        expected_cols = self.pipeline.config.n_columns
        measurement = np.asarray(measurement, dtype=np.float32)

        if measurement.ndim != 1 or measurement.shape[0] != expected_cols:
            raise ValueError(
                f"measurement deve ter shape ({expected_cols},), "
                f"recebido: {measurement.shape}"
            )

        # Adiciona ao buffer circular (FIFO — descarta o mais antigo se cheio)
        self.buffer.append(measurement)
        self.n_updates += 1

        # Buffer ainda nao cheio — fase de preenchimento
        if len(self.buffer) < self.window_size:
            logger.debug(
                "Buffer %d/%d — aguardando preenchimento",
                len(self.buffer), self.window_size,
            )
            return None

        # Buffer cheio — montar array 3D e executar inferencia
        # Shape: (1, window_size, 22) — batch unitario
        window_array = np.stack(list(self.buffer), axis=0)
        raw_data = window_array[np.newaxis, :, :]  # (1, window_size, 22)

        prediction = self.pipeline.predict(raw_data)

        logger.debug(
            "Inferencia realtime — update #%d, shape=%s",
            self.n_updates, prediction.shape,
        )
        return prediction

    # ────────────────────────────────────────────────────────────────
    # D2: Reset — limpa buffer e contador
    # ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Limpa o buffer circular e reinicia o contador.

        Deve ser chamado quando se inicia uma nova secao de poco ou
        quando o fluxo de medicoes e interrompido e reiniciado.

        Note:
            Referenciado em:
                - tests/test_inference.py: TestRealtimeInference.test_reset
            Ref: docs/ARCHITECTURE_v2.md secao 6.2.
            Apos reset, as proximas window_size medicoes serao fase
            de preenchimento (update retorna None).
        """
        self.buffer.clear()
        self.n_updates = 0
        logger.info("RealtimeInference buffer limpo — reset completo")

    # ────────────────────────────────────────────────────────────────
    # D2: Propriedades — status do buffer
    # ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True se buffer esta cheio e pronto para inferencia.

        Note:
            Referenciado em:
                - tests/test_inference.py: TestRealtimeInference.test_is_ready
        """
        return len(self.buffer) >= self.window_size

    @property
    def buffer_fill(self) -> float:
        """Fracao de preenchimento do buffer (0.0 a 1.0).

        Returns:
            Float entre 0.0 (vazio) e 1.0 (cheio).

        Note:
            Util para exibir progresso de preenchimento na UI.
        """
        return len(self.buffer) / self.window_size

    def __repr__(self) -> str:
        """Representacao concisa para logging.

        Note:
            Exibe window_size, preenchimento do buffer e status.
        """
        return (
            f"RealtimeInference("
            f"window_size={self.window_size}, "
            f"buffer={len(self.buffer)}/{self.window_size}, "
            f"ready={self.is_ready})"
        )
