# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: visualization/realtime.py                                         ║
# ║  Bloco: 9 — Visualization                                                ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Monitoramento ao vivo para inferencia realtime de geosteering        ║
# ║    • Plot continuamente atualizado: rho_h, rho_v com janela deslizante   ║
# ║    • Intervalo de confianca (incerteza) quando disponivel                 ║
# ║    • Suporte a ambientes interativos e headless                           ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), numpy, matplotlib (lazy)      ║
# ║  Exports: ~1 (RealtimeMonitor)                                            ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9.4                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (novo modulo)                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Monitoramento ao vivo para inferencia realtime de geosteering.

Classe RealtimeMonitor cria um display continuamente atualizado mostrando:
    - Predicao atual de resistividade (rho_h, rho_v)
    - Intervalo de confianca (se incerteza disponivel via UQ ensemble)
    - Janela deslizante das medicoes recentes

Projetado para uso em operacoes de geosteering realtime onde a equipe
precisa acompanhar a inversao em tempo real durante a perfuracao.

Example:
    >>> from geosteering_ai.visualization import RealtimeMonitor
    >>> monitor = RealtimeMonitor(window_size=50)
    >>> for pred in predictions:
    ...     monitor.update(pred)
    >>> monitor.close()

Note:
    Matplotlib importado de forma lazy (suporta ambientes headless).
    Referenciado em:
        - inference/realtime.py: loop de inferencia realtime
        - inference/pipeline.py: InferencePipeline com display
    Ref: docs/ARCHITECTURE_v2.md secao 9.4.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Classe principal ---
    "RealtimeMonitor",
]

# ──────────────────────────────────────────────────────────────────────
# Logger do modulo (D9: NUNCA print)
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# D10: Constantes de visualizacao
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_WINDOW_SIZE = 100
_FIGSIZE = (14, 6)
_UPDATE_INTERVAL_MS = 100   # intervalo de atualizacao em milissegundos
_CONFIDENCE_ALPHA = 0.25    # transparencia do intervalo de confianca

# Cores para consistencia visual com holdout.py
_COLOR_RHO_H = "#1f77b4"   # azul matplotlib default
_COLOR_RHO_V = "#d62728"   # vermelho matplotlib default
_COLOR_CI = "#aec7e8"      # azul claro para intervalo de confianca


# ──────────────────────────────────────────────────────────────────────
# D2: Classe principal — monitoramento realtime
# ──────────────────────────────────────────────────────────────────────
class RealtimeMonitor:
    """Display ao vivo para monitoramento de inferencia realtime.

    Cria uma figura matplotlib com 2 subplots (rho_h e rho_v)
    atualizados a cada chamada de ``update()``. Mantém uma janela
    deslizante de tamanho ``window_size`` com as predicoes mais
    recentes.

    Attributes:
        config: PipelineConfig opcional com metadados do pipeline.
        window_size: Tamanho da janela deslizante (default 100).
        _rho_h_buffer: Deque com historico de predicoes rho_h.
        _rho_v_buffer: Deque com historico de predicoes rho_v.
        _unc_h_buffer: Deque com incerteza rho_h (se disponivel).
        _unc_v_buffer: Deque com incerteza rho_v (se disponivel).
        _timestamps: Deque com timestamps das predicoes.
        _fig: Figura matplotlib (None ate primeira chamada a update).
        _ax_h: Axes para rho_h.
        _ax_v: Axes para rho_v.
        _step: Contador de passos de atualizacao.

    Example:
        >>> monitor = RealtimeMonitor(window_size=50)
        >>> pred = np.array([1.5, 2.3])  # [rho_h, rho_v] em log10
        >>> monitor.update(pred)
        >>> monitor.update(pred, uncertainty=np.array([0.1, 0.15]))
        >>> monitor.close()

    Note:
        Usa ``plt.ion()`` para modo interativo; ``plt.pause()`` para
        atualizacao nao-bloqueante. Em ambientes headless, as atualizacoes
        sao apenas logadas sem exibicao grafica.
        Ref: docs/ARCHITECTURE_v2.md secao 9.4.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> None:
        """Inicializa o monitor realtime.

        Args:
            config: PipelineConfig opcional para titulo e metadados.
                Se fornecido, model_type e incluido no titulo da figura.
            window_size: Tamanho maximo da janela deslizante (default 100).
                Predicoes mais antigas sao descartadas automaticamente.

        Note:
            A figura matplotlib so e criada na primeira chamada a update().
            Ref: docs/ARCHITECTURE_v2.md secao 9.4.
        """
        self.config = config
        self.window_size = window_size

        # Buffers circulares para janela deslizante
        self._rho_h_buffer: deque = deque(maxlen=window_size)
        self._rho_v_buffer: deque = deque(maxlen=window_size)
        self._unc_h_buffer: deque = deque(maxlen=window_size)
        self._unc_v_buffer: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)

        # Estado matplotlib (lazy init)
        self._fig = None
        self._ax_h = None
        self._ax_v = None
        self._step: int = 0
        self._interactive: bool = False

        logger.info(
            "RealtimeMonitor inicializado: window_size=%d", window_size
        )

    def _init_figure(self) -> None:
        """Cria a figura matplotlib em modo interativo.

        Chamado internamente na primeira invocacao de update().
        Usa plt.ion() para modo nao-bloqueante.

        Note:
            Se matplotlib nao estiver disponivel ou o backend nao
            suportar display, a flag _interactive e setada para False
            e atualizacoes sao apenas logadas.
        """
        try:
            import matplotlib.pyplot as plt
            plt.ion()  # modo interativo
            self._interactive = True
        except ImportError:
            logger.warning(
                "matplotlib nao instalado. Monitor operara em modo log-only."
            )
            self._interactive = False
            return
        except Exception:
            logger.warning(
                "Backend matplotlib sem suporte a display. Modo log-only."
            )
            self._interactive = False
            return

        self._fig, (self._ax_h, self._ax_v) = plt.subplots(
            1, 2, figsize=_FIGSIZE,
        )

        # Titulo da figura
        title = "Realtime Monitor"
        if self.config is not None:
            title = f"Realtime — {self.config.model_type}"
        self._fig.suptitle(title, fontsize=12, fontweight="bold")

        # Configuracao inicial dos axes
        self._ax_h.set_xlabel("Passo")
        self._ax_h.set_ylabel(r"$\log_{10}(\rho_h)$ [$\Omega\cdot$m]")
        self._ax_h.set_title(r"$\rho_h$ (horizontal)")
        self._ax_h.grid(True, alpha=0.3)

        self._ax_v.set_xlabel("Passo")
        self._ax_v.set_ylabel(r"$\log_{10}(\rho_v)$ [$\Omega\cdot$m]")
        self._ax_v.set_title(r"$\rho_v$ (vertical)")
        self._ax_v.grid(True, alpha=0.3)

        self._fig.tight_layout()
        logger.info("Figura realtime criada com sucesso")

    def update(
        self,
        prediction: np.ndarray,
        *,
        uncertainty: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Atualiza o display com uma nova predicao.

        Adiciona a predicao aos buffers circulares e redesenha a figura.
        Se incerteza for fornecida, plota intervalo de confianca como
        banda sombreada ao redor da predicao.

        Args:
            prediction: Array (2,) com [rho_h, rho_v] em log10 (ohm.m).
                Canal 0 = rho_h (horizontal), canal 1 = rho_v (vertical).
            uncertainty: Array (2,) opcional com [unc_h, unc_v] em log10.
                Desvio padrao da predicao (ex.: ensemble UQ). Se None,
                intervalo de confianca nao e plotado.
            timestamp: Timestamp opcional (float, ex.: epoch time).
                Se None, usa contador interno sequencial.

        Raises:
            ValueError: Se prediction nao tiver shape (2,).

        Note:
            A primeira chamada inicializa a figura via _init_figure().
            Ref: docs/ARCHITECTURE_v2.md secao 9.4.
        """
        prediction = np.asarray(prediction).ravel()
        if prediction.shape[0] < 2:
            msg = f"Esperado prediction com >= 2 valores, recebido shape={prediction.shape}."
            raise ValueError(msg)

        # Lazy init da figura
        if self._fig is None:
            self._init_figure()

        # Atualizar buffers
        self._rho_h_buffer.append(float(prediction[0]))
        self._rho_v_buffer.append(float(prediction[1]))

        if uncertainty is not None:
            uncertainty = np.asarray(uncertainty).ravel()
            self._unc_h_buffer.append(float(uncertainty[0]))
            self._unc_v_buffer.append(float(uncertainty[1]))
        else:
            self._unc_h_buffer.append(None)
            self._unc_v_buffer.append(None)

        if timestamp is not None:
            self._timestamps.append(timestamp)
        else:
            self._timestamps.append(self._step)

        self._step += 1

        # Redesenhar se display ativo
        if self._interactive:
            self._redraw()
        else:
            # Modo log-only para ambientes headless
            if self._step % 50 == 0:
                logger.info(
                    "Step %d: rho_h=%.4f, rho_v=%.4f",
                    self._step, prediction[0], prediction[1],
                )

    def _redraw(self) -> None:
        """Redesenha ambos os subplots com dados atuais dos buffers.

        Limpa os axes e replota todas as series na janela deslizante.
        Inclui banda de confianca se incerteza estiver disponivel.

        Note:
            Chamado internamente por update(). Nao deve ser invocado
            diretamente pelo usuario.
        """
        import matplotlib.pyplot as plt

        x = np.array(self._timestamps)
        rho_h = np.array(self._rho_h_buffer)
        rho_v = np.array(self._rho_v_buffer)

        # --- Subplot rho_h ---
        self._ax_h.cla()
        self._ax_h.plot(x, rho_h, color=_COLOR_RHO_H, linewidth=1.2, label=r"$\rho_h$")

        # Banda de confianca para rho_h
        unc_h_vals = list(self._unc_h_buffer)
        if any(v is not None for v in unc_h_vals):
            unc_h = np.array([v if v is not None else 0.0 for v in unc_h_vals])
            self._ax_h.fill_between(
                x, rho_h - unc_h, rho_h + unc_h,
                color=_COLOR_CI, alpha=_CONFIDENCE_ALPHA, label="IC",
            )

        self._ax_h.set_xlabel("Passo")
        self._ax_h.set_ylabel(r"$\log_{10}(\rho_h)$ [$\Omega\cdot$m]")
        self._ax_h.set_title(r"$\rho_h$ (horizontal)")
        self._ax_h.legend(loc="upper left", fontsize=8)
        self._ax_h.grid(True, alpha=0.3)

        # --- Subplot rho_v ---
        self._ax_v.cla()
        self._ax_v.plot(x, rho_v, color=_COLOR_RHO_V, linewidth=1.2, label=r"$\rho_v$")

        # Banda de confianca para rho_v
        unc_v_vals = list(self._unc_v_buffer)
        if any(v is not None for v in unc_v_vals):
            unc_v = np.array([v if v is not None else 0.0 for v in unc_v_vals])
            self._ax_v.fill_between(
                x, rho_v - unc_v, rho_v + unc_v,
                color=_COLOR_CI, alpha=_CONFIDENCE_ALPHA, label="IC",
            )

        self._ax_v.set_xlabel("Passo")
        self._ax_v.set_ylabel(r"$\log_{10}(\rho_v)$ [$\Omega\cdot$m]")
        self._ax_v.set_title(r"$\rho_v$ (vertical)")
        self._ax_v.legend(loc="upper left", fontsize=8)
        self._ax_v.grid(True, alpha=0.3)

        # --- Flush display ---
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        plt.pause(0.001)  # pequena pausa para backend processar eventos

    def close(self) -> None:
        """Fecha o display de monitoramento e libera recursos.

        Desativa modo interativo do matplotlib e fecha a figura.
        Seguro para chamar multiplas vezes (idempotente).

        Note:
            Ref: docs/ARCHITECTURE_v2.md secao 9.4.
        """
        if self._interactive and self._fig is not None:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close(self._fig)
            logger.info(
                "RealtimeMonitor fechado apos %d passos", self._step
            )

        self._fig = None
        self._ax_h = None
        self._ax_v = None
        self._interactive = False
