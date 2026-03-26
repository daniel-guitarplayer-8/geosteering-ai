# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: utils/timer.py                                                    ║
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
# ║    • Medicao de tempo com format_time() e elapsed_since()                 ║
# ║    • Decorator timer_decorator para profiling de funcoes                   ║
# ║    • ProgressTracker para progresso com ETA e barra ASCII                 ║
# ║                                                                            ║
# ║  Dependencias: time (stdlib), functools (stdlib), logging (stdlib)         ║
# ║  Exports: ~4 simbolos — ver __all__                                       ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 9 (utils/)                           ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FORMATACAO DE TEMPO
# ════════════════════════════════════════════════════════════════════════════
# Converte duracao em segundos para formato humano com unidade adaptativa.
# Escala automatica: ms (<1s), s (<60s), min (<3600s), h (>=3600s).
# Usado por timer_decorator, ProgressTracker, e training loop.
# ──────────────────────────────────────────────────────────────────────────


def format_time(seconds: float) -> str:
    """Formata duracao em segundos para string legivel com unidade adaptativa.

    Seleciona automaticamente a unidade mais apropriada para a duracao:
    milissegundos para operacoes rapidas, segundos para operacoes curtas,
    minutos para treinamento tipico, horas para treinos longos.

    Args:
        seconds: Duracao em segundos (float, pode ser fracao).

    Returns:
        str: Duracao formatada com 2 casas decimais e unidade.
            Exemplos: "45.23 ms", "1.50 s", "12.34 min", "2.10 h".

    Example:
        >>> from geosteering_ai.utils.timer import format_time
        >>> format_time(0.045)
        '45.00 ms'
        >>> format_time(125.5)
        '2.09 min'

    Note:
        Referenciado em:
            - utils/timer.py (elapsed_since, timer_decorator, ProgressTracker)
            - training/loop.py (duracao de epocas e treinamento total)
        Ref: Padrao D9 — logging de performance.
    """
    if seconds < 1.0:
        # ── Milissegundos: operacoes sub-segundo (forward pass, I/O) ──
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60.0:
        # ── Segundos: operacoes curtas (load dataset, compile model) ───
        return f"{seconds:.2f} s"
    elif seconds < 3600.0:
        # ── Minutos: treinamento tipico (10-60 epocas) ────────────────
        return f"{seconds / 60:.2f} min"
    else:
        # ── Horas: treinamento longo (200+ epocas, HPO) ──────────────
        return f"{seconds / 3600:.2f} h"


def elapsed_since(t0: float, label: str = "Tempo") -> str:
    """Calcula e formata tempo decorrido desde t0.

    Wrapper conveniente sobre format_time() usando time.perf_counter().
    Padrao de uso: ``t0 = time.perf_counter()`` no inicio, depois
    ``elapsed_since(t0, "Treinamento")`` ao final.

    Args:
        t0: Timestamp de inicio (de time.perf_counter()).
        label: Prefixo descritivo para a mensagem.

    Returns:
        str: String formatada "{label}: {duracao}".
            Exemplo: "Treinamento: 12.34 min".

    Example:
        >>> import time
        >>> from geosteering_ai.utils.timer import elapsed_since
        >>> t0 = time.perf_counter()
        >>> # ... operacao ...
        >>> msg = elapsed_since(t0, "Operacao")

    Note:
        Referenciado em: todos os modulos que medem tempo de execucao.
    """
    return f"{label}: {format_time(time.perf_counter() - t0)}"


# ════════════════════════════════════════════════════════════════════════════
# SECAO: TIMER DECORATOR
# ════════════════════════════════════════════════════════════════════════════
# Decorator para profiling automatico de funcoes.
# Loga duracao via logger.info() apos cada chamada.
# Preserva metadata da funcao original via functools.wraps.
# ──────────────────────────────────────────────────────────────────────────


def timer_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator que mede e loga o tempo de execucao de uma funcao.

    Usa time.perf_counter() para alta resolucao. Loga resultado via
    logger.info() com nome qualificado da funcao e duracao formatada.
    Preserva metadata (__name__, __doc__) via functools.wraps.

    Args:
        func: Funcao a ser decorada.

    Returns:
        Callable: Funcao wrapper com medicao de tempo.

    Example:
        >>> from geosteering_ai.utils.timer import timer_decorator
        >>> @timer_decorator
        ... def treinar_modelo(config):
        ...     pass  # treinamento

    Note:
        Referenciado em: training/loop.py, inference/pipeline.py.
        Ref: Padrao D9 — profiling via logging.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        duration = format_time(time.perf_counter() - t0)
        logger.info("%s: %s", func.__qualname__, duration)
        return result

    return wrapper


# ════════════════════════════════════════════════════════════════════════════
# SECAO: PROGRESS TRACKER
# ════════════════════════════════════════════════════════════════════════════
# Tracker de progresso com barra ASCII e estimativa de tempo restante.
# Ideal para loops longos (loading de dados, processamento batch).
# Usa logger ao inves de print (D9 obrigatorio).
# ──────────────────────────────────────────────────────────────────────────


class ProgressTracker:
    """Tracker de progresso com barra ASCII e ETA.

    Exibe progresso de operacoes longas (loading de dados, processamento
    batch, validacoes) como barra ASCII com percentual e estimativa de
    tempo restante. Toda saida via logging (NUNCA print — D9).

    Formato da barra:
        ``[========..........] 40.0% — ETA: 1.23 s``

    Attributes:
        total (int): Numero total de passos.
        description (str): Label descritivo para o log.
        current (int): Passo atual (0-indexed, incrementado por update).
        start_time (float): Timestamp de inicio (perf_counter).

    Example:
        >>> from geosteering_ai.utils.timer import ProgressTracker
        >>> tracker = ProgressTracker(total=100, description="Loading")
        >>> for i in range(100):
        ...     tracker.update()
        >>> tracker.finish()

    Note:
        Referenciado em:
            - data/loading.py (carregamento de modelos geologicos)
            - training/loop.py (iteracao sobre epocas)
        Ref: Padrao D9 — progresso via logging estruturado.
    """

    # Largura da barra ASCII em caracteres.
    _BAR_WIDTH: int = 30

    def __init__(
        self,
        total: int,
        description: str = "Progresso",
        log: Optional[logging.Logger] = None,
    ) -> None:
        """Inicializa tracker de progresso.

        Args:
            total: Numero total de passos esperados.
            description: Label descritivo para mensagens de log.
            log: Logger opcional. Se None, usa logger do modulo.
        """
        if total <= 0:
            raise ValueError(
                f"ProgressTracker: total deve ser > 0, recebido {total}"
            )
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.perf_counter()
        self._log = log or logger

    def update(self, step: Optional[int] = None) -> None:
        """Atualiza progresso e loga barra ASCII com ETA.

        Args:
            step: Passo atual (1-indexed). Se None, incrementa current+1.
        """
        if step is not None:
            self.current = step
        else:
            self.current += 1

        if self.total <= 0:
            return

        # ── Calculo de progresso e ETA ────────────────────────────────
        fraction = min(self.current / self.total, 1.0)
        filled = int(self._BAR_WIDTH * fraction)
        bar = "=" * filled + "." * (self._BAR_WIDTH - filled)
        pct = fraction * 100.0

        elapsed = time.perf_counter() - self.start_time
        if fraction > 0:
            eta = elapsed * (1.0 - fraction) / fraction
            eta_str = format_time(eta)
        else:
            eta_str = "?"

        self._log.info(
            "%s: [%s] %.1f%% — ETA: %s",
            self.description, bar, pct, eta_str,
        )

    def finish(self) -> None:
        """Loga conclusao com tempo total decorrido."""
        elapsed = time.perf_counter() - self.start_time
        self._log.info(
            "%s: concluido em %s",
            self.description, format_time(elapsed),
        )


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Formatacao de tempo ───────────────────────────────────────────
    "format_time",
    "elapsed_since",
    # ── Decorator ─────────────────────────────────────────────────────
    "timer_decorator",
    # ── Tracker ───────────────────────────────────────────────────────
    "ProgressTracker",
]
