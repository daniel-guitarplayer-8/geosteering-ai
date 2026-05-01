# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_heartbeat.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Sentinel de Travamento na Main Thread║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-29 (v2.11)                                         ║
# ║  Status      : Produção (modo Debug — opt-in via env var ou menu)        ║
# ║  Framework   : QTimer (Qt event loop) + time.perf_counter (monotônico)   ║
# ║  Dependências: sm_qt_compat (camada PyQt6/PySide6)                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Detecta gaps no event loop da main thread Qt (= GUI travada). Um      ║
# ║    `QTimer` configurado para 16ms (60 FPS, paint budget) periodicamente  ║
# ║    registra o tempo entre disparos. Se o tempo exceder o threshold      ║
# ║    (default 50ms), o gap é gravado em log e contabilizado.              ║
# ║                                                                           ║
# ║  POR QUE FUNCIONA                                                         ║
# ║    O event loop Qt processa eventos em ordem na main thread. Se uma     ║
# ║    operação síncrona (loop pesado, I/O bloqueante, lock contention)    ║
# ║    consome a CPU da main thread, o `QTimer.timeout` é adiado até a     ║
# ║    operação terminar. O gap medido é precisamente o tempo que a GUI    ║
# ║    ficou irresponsiva — i.e., quanto o usuário percebeu como freeze.   ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    Modo automático (env var):                                            ║
# ║      $ SM_HEARTBEAT=1 python -m geosteering_ai.simulation.tests....      ║
# ║                                                                           ║
# ║    Modo programático:                                                    ║
# ║      hb = MainThreadHeartbeat(threshold_ms=50.0)                         ║
# ║      hb.start()                                                          ║
# ║      # ... simulação ocorre ...                                          ║
# ║      hb.stop()                                                           ║
# ║      report = hb.report()                                                ║
# ║                                                                           ║
# ║  CRITÉRIOS DE APROVAÇÃO v2.11                                             ║
# ║    • max_gap_ms < 50 ms para qualquer N (100, 1k, 10k, 30k modelos)     ║
# ║    • sum_gap_ms < 200 ms total durante a simulação                       ║
# ║    • total_gaps < 5 (gaps acima do threshold)                             ║
# ║                                                                           ║
# ║  CUSTO DA INSTRUMENTAÇÃO                                                  ║
# ║    QTimer 16ms ≈ 60 disparos/seg → ~0.1% CPU em I/O baixo. Custo de    ║
# ║    cada tick é dominado por ``time.perf_counter`` (~50ns) + acesso a   ║
# ║    list (amortizado). Negligível mesmo em sessões longas.              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Detector de gaps na main thread Qt — sentinel de freezing da GUI.

Provê :class:`MainThreadHeartbeat` que mede a responsividade do event
loop Qt via ``QTimer`` periódico. Cada tick atrasado além do threshold
é registrado como ``HeartbeatGap``, formando um histórico empírico do
travamento percebido pelo usuário.

Diferente de :class:`PhaseTimer` (que mede tempo de fases lógicas), o
heartbeat mede o tempo REAL em que a GUI ficou paralisada — incluindo
gaps causados por trabalho na main thread, contention de GIL ou I/O
bloqueante de C extensions.

Example:
    Validação empírica do fix de v2.11::

        >>> hb = MainThreadHeartbeat(threshold_ms=50.0)
        >>> hb.start()
        >>> _start_simulation()  # 30k modelos
        >>> wait_for_completion()
        >>> hb.stop()
        >>> report = hb.report()
        >>> assert report.max_gap_ms < 50.0, f"Freeze detectado: {report}"

Note:
    O heartbeat é leve mas NÃO substitui ``PhaseTimer`` — ambos são
    complementares. O heartbeat detecta SE houve freeze; o phase timer
    detecta QUAL fase causou. Use os dois juntos para diagnóstico
    completo no relatório de profiling.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .sm_qt_compat import QObject, QtCore, Signal

# ──────────────────────────────────────────────────────────────────────────
# Defaults — calibrados para 60 FPS (16ms paint budget Qt6)
# ──────────────────────────────────────────────────────────────────────────

# Intervalo entre ticks. 16ms = 60 FPS = budget de paint frame Qt6.
# Valores menores (ex.: 8ms) aumentam resolução mas custam CPU.
DEFAULT_TICK_MS: int = 16

# Threshold para considerar um tick "atrasado" (gap). 50ms corresponde ao
# critério visual humano de "lag perceptível" (~3 frames perdidos a 60 FPS).
DEFAULT_THRESHOLD_MS: float = 50.0

# Variável de ambiente para ativação automática no startup do MainWindow.
# Quando definida (qualquer valor truthy), heartbeat inicia em showEvent.
ENV_FLAG: str = "SM_HEARTBEAT"


# ──────────────────────────────────────────────────────────────────────────
# Estruturas de dados
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class HeartbeatGap:
    """Registro de um gap individual detectado na main thread.

    Attributes:
        timestamp: Momento da detecção (``time.perf_counter()`` em
            segundos desde o início do processo).
        gap_ms: Duração do gap em milissegundos (``> threshold_ms``).
        wall_time: Wall-clock time (``time.time()``) para correlação
            cross-machine. Útil em logs persistidos.
    """

    timestamp: float
    gap_ms: float
    wall_time: float


@dataclass
class HeartbeatReport:
    """Sumário consolidado dos gaps detectados em uma sessão.

    Attributes:
        total_ticks: Número total de ticks recebidos (incluindo on-time).
        total_gaps: Número de ticks acima do threshold.
        max_gap_ms: Maior gap individual registrado (em ms).
        sum_gap_ms: Soma de todos os gaps (em ms) — wall-clock total
            de tempo em que a GUI ficou bloqueada.
        avg_gap_ms: Média dos gaps acima do threshold (em ms).
        threshold_ms: Threshold usado na captura (info de contexto).
        gaps: Lista completa dos eventos individuais (até MAX_GAPS).
    """

    total_ticks: int = 0
    total_gaps: int = 0
    max_gap_ms: float = 0.0
    sum_gap_ms: float = 0.0
    avg_gap_ms: float = 0.0
    threshold_ms: float = DEFAULT_THRESHOLD_MS
    gaps: List[HeartbeatGap] = field(default_factory=list)

    def passes_v211_criteria(self) -> bool:
        """Retorna True se a sessão passa nos critérios de aprovação v2.11.

        Critérios (definidos no plano de implementação):
        - ``max_gap_ms < 50ms``
        - ``sum_gap_ms < 200ms``
        - ``total_gaps < 5``
        """
        return self.max_gap_ms < 50.0 and self.sum_gap_ms < 200.0 and self.total_gaps < 5

    def format_summary(self) -> str:
        """Formata sumário em string single-line para log/console.

        Returns:
            Ex.: ``"ticks=1864 gaps=2 max=72.3ms sum=139.5ms avg=69.8ms"``.
        """
        return (
            f"ticks={self.total_ticks} gaps={self.total_gaps} "
            f"max={self.max_gap_ms:.1f}ms sum={self.sum_gap_ms:.1f}ms "
            f"avg={self.avg_gap_ms:.1f}ms threshold={self.threshold_ms:.0f}ms"
        )

    def to_dict(self) -> dict:
        """Serializa para dict (JSON-compatível).

        Útil para escrita em ``v2.11_baseline_profiling.md`` e relatórios
        comparativos baseline vs pós-fix.
        """
        return {
            "total_ticks": self.total_ticks,
            "total_gaps": self.total_gaps,
            "max_gap_ms": self.max_gap_ms,
            "sum_gap_ms": self.sum_gap_ms,
            "avg_gap_ms": self.avg_gap_ms,
            "threshold_ms": self.threshold_ms,
            "passes_v211": self.passes_v211_criteria(),
            "gaps": [
                {
                    "timestamp": g.timestamp,
                    "gap_ms": g.gap_ms,
                    "wall_time": g.wall_time,
                }
                for g in self.gaps
            ],
        }


# ──────────────────────────────────────────────────────────────────────────
# Sentinel — QTimer-based gap detector
# ──────────────────────────────────────────────────────────────────────────

# Limite máximo de gaps armazenados em memória — evita crescimento ilimitado
# em sessões muito longas. Após este limite, gaps mais antigos são descartados
# (ring-buffer simples). 1000 gaps × ~50 bytes = 50KB — desprezível.
MAX_GAPS: int = 1000


class MainThreadHeartbeat(QObject):
    """Detector de gaps na main thread Qt via ``QTimer`` periódico.

    O timer dispara a cada ``tick_ms`` milissegundos. Em cada tick, mede
    o intervalo desde o tick anterior; se exceder ``threshold_ms``,
    registra um :class:`HeartbeatGap` e emite o sinal ``gap_detected``.

    Attributes:
        gap_detected: Sinal Qt ``Signal(float)`` — emitido quando um gap
            é detectado. Argumento: gap em ms. Útil para alertas em UI
            (ex.: piscar ícone vermelho na status bar).
        threshold_ms: Threshold acima do qual ticks são considerados gaps.
        tick_ms: Intervalo nominal entre ticks do QTimer.

    Example:
        Modo programático com captura completa::

            >>> hb = MainThreadHeartbeat(threshold_ms=50.0)
            >>> hb.start()
            >>> # ... ações que podem travar GUI ...
            >>> hb.stop()
            >>> report = hb.report()
            >>> if not report.passes_v211_criteria():
            ...     print(f"⚠️ Freeze detectado: {report.format_summary()}")

    Note:
        Sentinel é apenas observador — NÃO consome eventos do loop. O
        custo é o próprio ``QTimer.timeout`` (~50µs por tick incluindo
        ``perf_counter()`` + list append). A 60 FPS isso é < 1% CPU.
    """

    # Sinal emitido em cada gap > threshold. Argumento: ms decorridos.
    # Útil para integração com UI debug (status bar, indicadores, toasts).
    gap_detected = Signal(float)

    def __init__(
        self,
        threshold_ms: float = DEFAULT_THRESHOLD_MS,
        tick_ms: int = DEFAULT_TICK_MS,
        parent: Optional[QObject] = None,
    ) -> None:
        """Inicializa o sentinel sem startá-lo (chame ``start()``).

        Args:
            threshold_ms: Limite acima do qual gaps são registrados.
                Default ``50.0`` ms (3 frames de paint budget perdidos).
            tick_ms: Intervalo nominal do QTimer. Default ``16`` ms (60 FPS).
            parent: ``QObject`` pai opcional (default ``None``).
        """
        super().__init__(parent)
        # Configuração imutável após construção (mudar em runtime exige stop+start).
        self.threshold_ms = float(threshold_ms)
        self.tick_ms = int(tick_ms)
        # Estado mutável — protegido implicitamente pelo event loop Qt
        # (ticks executam serialmente na main thread, sem race possível).
        self._timer: Optional[QtCore.QTimer] = None
        self._last_tick: float = 0.0
        self._gaps: List[HeartbeatGap] = []
        self._total_ticks: int = 0
        self._is_running: bool = False

    # ── Controle de ciclo de vida ────────────────────────────────────────

    def start(self) -> None:
        """Inicia o ``QTimer`` e começa a registrar ticks.

        Idempotente: chamar ``start`` quando já rodando é no-op seguro.
        Reseta o histórico de gaps a cada novo ``start()`` — para preservar
        múltiplas sessões, chame ``report()`` antes.
        """
        if self._is_running:
            return
        if QtCore is None:  # Qt não disponível (ambiente headless raro)
            return
        # Reset do estado para a nova sessão.
        self._gaps.clear()
        self._total_ticks = 0
        self._last_tick = time.perf_counter()
        # QTimer associado a este QObject — herda thread (main) automaticamente.
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.tick_ms)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()
        self._is_running = True

    def stop(self) -> None:
        """Para o ``QTimer`` e congela o estado para inspeção via ``report()``.

        Idempotente: chamar ``stop`` quando não rodando é no-op seguro.
        """
        if not self._is_running:
            return
        if self._timer is not None:
            try:
                self._timer.stop()
                self._timer.timeout.disconnect(self._on_tick)
            except Exception:
                # Disconnect pode falhar se Qt está finalizando — silenciamos.
                pass
            self._timer = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Retorna ``True`` se o ``QTimer`` está ativo e medindo ticks."""
        return self._is_running

    # ── Slot interno do QTimer (main thread) ─────────────────────────────

    def _on_tick(self) -> None:
        """Slot conectado ao ``QTimer.timeout``. Roda na main thread.

        Calcula o intervalo desde o tick anterior e — se exceder threshold —
        registra um gap. Mantém ring-buffer de tamanho ``MAX_GAPS``.
        """
        now = time.perf_counter()
        # gap_ms = tempo entre ticks (deveria ser ≈ tick_ms se main thread livre)
        gap_ms = (now - self._last_tick) * 1000.0
        self._last_tick = now
        self._total_ticks += 1
        # Threshold check — só registra ticks atrasados (gaps reais).
        if gap_ms > self.threshold_ms:
            entry = HeartbeatGap(
                timestamp=now,
                gap_ms=gap_ms,
                wall_time=time.time(),
            )
            self._gaps.append(entry)
            # Ring buffer: descarta gaps antigos se exceder limite.
            if len(self._gaps) > MAX_GAPS:
                self._gaps = self._gaps[-MAX_GAPS:]
            # Emit fora da seção crítica (não há crítica — main thread serial).
            self.gap_detected.emit(gap_ms)

    # ── Inspecção e relatórios ──────────────────────────────────────────

    def report(self) -> HeartbeatReport:
        """Snapshot consolidado dos gaps detectados até este momento.

        Pode ser chamado durante a execução (live) ou após ``stop()``.

        Returns:
            :class:`HeartbeatReport` imutável (cópia dos gaps internos).
        """
        gaps_snapshot = list(self._gaps)
        n_gaps = len(gaps_snapshot)
        if n_gaps == 0:
            return HeartbeatReport(
                total_ticks=self._total_ticks,
                threshold_ms=self.threshold_ms,
            )
        gap_values = [g.gap_ms for g in gaps_snapshot]
        return HeartbeatReport(
            total_ticks=self._total_ticks,
            total_gaps=n_gaps,
            max_gap_ms=max(gap_values),
            sum_gap_ms=sum(gap_values),
            avg_gap_ms=sum(gap_values) / n_gaps,
            threshold_ms=self.threshold_ms,
            gaps=gaps_snapshot,
        )

    def reset(self) -> None:
        """Limpa o histórico de gaps mantendo o timer ativo.

        Útil para isolar uma sub-fase de medição (ex.: medir apenas
        a fase ``simulation`` sem incluir ``generation``).
        """
        self._gaps.clear()
        self._total_ticks = 0
        self._last_tick = time.perf_counter()


# ──────────────────────────────────────────────────────────────────────────
# Helpers de ativação (env var, programático)
# ──────────────────────────────────────────────────────────────────────────


def is_enabled_via_env() -> bool:
    """Retorna ``True`` se variável de ambiente ``SM_HEARTBEAT`` é truthy.

    Aceita qualquer valor não-vazio diferente de ``"0"``, ``"false"``,
    ``"no"`` (case-insensitive). Use em ``MainWindow.showEvent`` para
    auto-iniciar o sentinel sem mudança de código.

    Example:
        ``$ SM_HEARTBEAT=1 python -m geosteering_ai.simulation.tests…``
    """
    val = os.environ.get(ENV_FLAG, "").strip().lower()
    if not val:
        return False
    return val not in ("0", "false", "no", "off")


__all__ = [
    "DEFAULT_THRESHOLD_MS",
    "DEFAULT_TICK_MS",
    "ENV_FLAG",
    "HeartbeatGap",
    "HeartbeatReport",
    "MAX_GAPS",
    "MainThreadHeartbeat",
    "is_enabled_via_env",
]
