# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_phase_timer.py                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Cronologia Instrumentada de Fases     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-29 (v2.11)                                         ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : Qt6 (Signal/Slot) + time.perf_counter (monotônico)         ║
# ║  Dependências: sm_qt_compat (camada PyQt6/PySide6)                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Mede e expõe via Qt signals o tempo gasto em cada fase do ciclo de    ║
# ║    simulação do Simulation Manager. Substitui prints ad-hoc por um       ║
# ║    rastreamento estruturado, persistente e exibível na UI (painel        ║
# ║    "Cronologia da Simulação").                                            ║
# ║                                                                           ║
# ║  FASES RASTREADAS (8 marcos)                                              ║
# ║    ┌──────────────────────────────────────────────────────────────────┐  ║
# ║    │  validation         → coleta + validação de parâmetros UI        │  ║
# ║    │  generation         → ModelGenerationThread (geração de modelos) │  ║
# ║    │  pool_warmup        → spawn de workers + JIT warmup Numba        │  ║
# ║    │  simulation         → loop principal as_completed(futures)       │  ║
# ║    │  ipc_concat         → IPC pickle + np.concatenate de H_stack     │  ║
# ║    │  set_current_sim    → população de combos + cache de plot bundle │  ║
# ║    │  save_dat           → SaveArtifactsThread (escrita binária .dat) │  ║
# ║    │  snapshot_persist   → SnapshotPersistThread (JSON do experimento)│  ║
# ║    └──────────────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  POR QUE INSTRUMENTAR                                                     ║
# ║    • Diagnóstico de freezing GUI: identifica a fase responsável por      ║
# ║      cada gap detectado pelo MainThreadHeartbeat (sm_heartbeat.py).      ║
# ║    • Comparação baseline vs pós-fix: dois snapshots do mesmo cenário     ║
# ║      revelam onde a otimização teve efeito (e onde não).                ║
# ║    • Apresentação ao usuário: o painel "Cronologia" mostra tempos       ║
# ║      exatos de cada fase, dando feedback de produção em tempo real.    ║
# ║                                                                           ║
# ║  USO TÍPICO                                                               ║
# ║    timer = PhaseTimer()                                                   ║
# ║    timer.begin("generation")                                              ║
# ║    # ... loop de geração ...                                              ║
# ║    timer.end("generation")                                                ║
# ║    print(timer.format_summary())                                          ║
# ║    # → "validation: 0.01s | generation: 4.21s | pool_warmup: 0.83s | …"  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Cronologia instrumentada das fases de simulação.

Provê a classe :class:`PhaseTimer` que mede tempos de fase com
``time.perf_counter`` (monotônico, sub-microssegundo) e os expõe como
sinais Qt para integração com a UI.

A instrumentação é leve (custo ~1µs por begin/end), permanente (não é
debug-only) e thread-safe para emissão a partir de QThreads filhos
(`generation`, `simulation`, `save_dat`, etc.) — usa lock interno para
proteger o dicionário de estados.

Example:
    Uso típico em ``MainWindow._start_simulation``::

        >>> from .sm_phase_timer import PhaseTimer
        >>> self.phase_timer = PhaseTimer(self)
        >>> self.phase_timer.phase_completed.connect(self._on_phase_completed)
        >>> self.phase_timer.begin("validation")
        >>> # ... validar entradas ...
        >>> self.phase_timer.end("validation")

Note:
    O timer NÃO encerra automaticamente uma fase ao iniciar outra.
    Fases podem se sobrepor (ex.: ``simulation`` enquanto ``save_dat``
    ainda roda em outra thread). Use ``end(name)`` explicitamente.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from .sm_qt_compat import QObject, Signal

# ──────────────────────────────────────────────────────────────────────────
# Constantes — fases canônicas do ciclo de simulação
# ──────────────────────────────────────────────────────────────────────────

# Lista ordenada — define a sequência típica exibida no painel de Cronologia.
# A ordem aqui é a ordem em que as fases ocorrem no fluxo normal de simulação.
PHASE_ORDER: List[str] = [
    "validation",
    "generation",
    "pool_warmup",
    "simulation",
    "ipc_concat",
    "set_current_sim",
    "save_dat",
    "snapshot_persist",
]

# Rótulos amigáveis em PT-BR para apresentação ao usuário no painel Cronologia.
# Sincronizados com PHASE_ORDER — qualquer fase nova deve ter rótulo aqui.
PHASE_LABELS: Dict[str, str] = {
    "validation": "Validação de parâmetros",
    "generation": "Geração de modelos",
    "pool_warmup": "Warmup do pool",
    "simulation": "Simulação",
    "ipc_concat": "Concatenação IPC",
    "set_current_sim": "População de UI",
    "save_dat": "Salvando .dat",
    "snapshot_persist": "Persistindo snapshot",
}


# ──────────────────────────────────────────────────────────────────────────
# Classe principal
# ──────────────────────────────────────────────────────────────────────────


class PhaseTimer(QObject):
    """Mede tempo de cada fase do ciclo de simulação e emite via sinais Qt.

    Internamente usa ``time.perf_counter()`` (monotônico, alta resolução)
    para evitar saltos negativos quando o relógio do sistema é ajustado.
    Estados de fases ativas são protegidos por ``threading.Lock`` para
    permitir chamadas seguras a partir de QThreads filhos.

    Attributes:
        phase_started: Sinal Qt ``Signal(str)`` — emitido quando ``begin``
            é chamado. Argumento: nome da fase.
        phase_completed: Sinal Qt ``Signal(str, float)`` — emitido quando
            ``end`` é chamado. Argumentos: nome, segundos decorridos.
        timings: Mapa final ``{phase: elapsed_sec}`` (somente leitura).
        active: Mapa interno ``{phase: t_start}`` de fases em execução.

    Example:
        Integração com ``ModelGenerationThread``::

            >>> timer = PhaseTimer()
            >>> def on_models_ready(models):
            ...     timer.end("generation")
            >>> gen_thread.finished_models.connect(on_models_ready)
            >>> timer.begin("generation")
            >>> gen_thread.start()
    """

    # Sinais Qt — emitidos para a UI (main thread) atualizar painéis.
    # phase_started: notifica que uma fase entrou em execução.
    # phase_completed: notifica conclusão com tempo decorrido em segundos.
    phase_started = Signal(str)
    phase_completed = Signal(str, float)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Inicializa o timer com dicionários vazios e lock thread-safe.

        Args:
            parent: ``QObject`` pai opcional para gerenciamento de
                ciclo de vida Qt. Default ``None`` (sem parent).
        """
        super().__init__(parent)
        # Dicionário de fases ativas: phase_name → t_start (perf_counter).
        # Acesso protegido por _lock — emissões podem vir de QThreads.
        self._active: Dict[str, float] = {}
        # Resultados consolidados após end(): phase_name → elapsed_sec.
        # Esta é a forma persistente que sobrevive ao ciclo da simulação.
        self._timings: Dict[str, float] = {}
        # Lock para proteger _active e _timings em chamadas multi-thread.
        # phase_started/phase_completed são thread-safe via Qt signal queue.
        self._lock = threading.Lock()

    # ── API pública ──────────────────────────────────────────────────────

    def begin(self, phase: str) -> None:
        """Marca início de uma fase e emite ``phase_started``.

        Args:
            phase: Nome canônico da fase (deve estar em ``PHASE_ORDER``,
                mas fases customizadas são permitidas para extensões).

        Note:
            Chamar ``begin`` para uma fase já ativa sobrescreve o tempo
            de início — útil para reiniciar contadores em casos de retry.
        """
        with self._lock:
            self._active[phase] = time.perf_counter()
        # Emit fora do lock para evitar deadlock com slots conectados
        # que possam chamar de volta a outra API do PhaseTimer.
        self.phase_started.emit(phase)

    def end(self, phase: str) -> Optional[float]:
        """Encerra uma fase ativa e emite ``phase_completed``.

        Args:
            phase: Nome da fase a encerrar. Se a fase não estava ativa,
                retorna ``None`` silenciosamente (no-op seguro).

        Returns:
            Tempo decorrido em segundos, ou ``None`` se a fase não
            estava em ``_active``.
        """
        with self._lock:
            t_start = self._active.pop(phase, None)
            if t_start is None:
                return None
            elapsed = time.perf_counter() - t_start
            self._timings[phase] = elapsed
        self.phase_completed.emit(phase, elapsed)
        return elapsed

    def reset(self) -> None:
        """Limpa todos os timings e fases ativas.

        Útil ao iniciar uma nova simulação para evitar acumulação de
        dados da execução anterior no painel de cronologia.
        """
        with self._lock:
            self._active.clear()
            self._timings.clear()

    # ── Inspecção (somente leitura) ──────────────────────────────────────

    def get_summary(self) -> Dict[str, float]:
        """Retorna cópia imutável do mapa de tempos consolidados.

        Returns:
            Dicionário ``{phase_name: elapsed_sec}`` ordenado pela
            ordem em que as fases foram completadas (insertion order).
        """
        with self._lock:
            return dict(self._timings)

    def get_active_phases(self) -> Dict[str, float]:
        """Retorna fases atualmente em execução com tempo decorrido até agora.

        Útil para o painel de cronologia mostrar barras "em curso" com
        tempo parcial atualizado em tempo real (ex.: a cada 500ms via
        ``QTimer``).

        Returns:
            Dicionário ``{phase_name: elapsed_so_far_sec}``.
        """
        now = time.perf_counter()
        with self._lock:
            return {phase: now - t0 for phase, t0 in self._active.items()}

    def total_elapsed(self) -> float:
        """Soma os tempos de TODAS as fases completadas.

        Note:
            Soma simples — NÃO leva em conta sobreposição de fases (ex.:
            ``save_dat`` rodando paralelo a ``snapshot_persist``). Para
            wall-clock real do ciclo, capture timestamps à parte.
        """
        with self._lock:
            return sum(self._timings.values())

    # ── Apresentação ─────────────────────────────────────────────────────

    def format_summary(self, separator: str = " | ") -> str:
        """Formata sumário em string compacta para log e tooltip.

        Args:
            separator: String entre fases. Default ``" | "``.

        Returns:
            String tipo ``"validation: 0.01s | generation: 4.21s | ..."``.
            Ordem segue ``PHASE_ORDER`` quando possível, senão insertion.
        """
        with self._lock:
            timings = dict(self._timings)
        # Ordena: primeiro fases canônicas em PHASE_ORDER, depois extras.
        ordered_keys: List[str] = [p for p in PHASE_ORDER if p in timings]
        extras: List[str] = [p for p in timings if p not in PHASE_ORDER]
        ordered_keys.extend(extras)
        parts = [f"{p}: {timings[p]:.2f}s" for p in ordered_keys]
        return separator.join(parts)

    def format_summary_pt(self, separator: str = " | ") -> str:
        """Versão PT-BR com rótulos amigáveis (``PHASE_LABELS``).

        Args:
            separator: String entre fases.

        Returns:
            String tipo ``"Validação: 0.01s | Geração de modelos: 4.21s | …"``.
        """
        with self._lock:
            timings = dict(self._timings)
        ordered_keys = [p for p in PHASE_ORDER if p in timings]
        extras = [p for p in timings if p not in PHASE_ORDER]
        ordered_keys.extend(extras)
        parts = [f"{PHASE_LABELS.get(p, p)}: {timings[p]:.2f}s" for p in ordered_keys]
        return separator.join(parts)


__all__ = ["PHASE_LABELS", "PHASE_ORDER", "PhaseTimer"]
