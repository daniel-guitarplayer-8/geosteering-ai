# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/shell/widgets/animation_bar.py                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : AnimationBar — playback de frames (slider + play + speed)   ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — shell widgets (spec 0017, Fatia 6d)                  ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — plots dinâmicos                                  ║
# ║  Framework   : Qt6 via gui.qt_compat (QSlider + QTimer)                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Barra reutilizável de animação: ``◀ ▶/⏸ ▶`` + slider + velocidade        ║
# ║    (fps) + rótulo de frame. Um ``QTimer`` avança o frame e emite           ║
# ║    ``frame_changed(int)`` — a galeria usa isso p/ varrer o ensemble        ║
# ║    (single-model playback, alavancando o PyQtGraph dinâmico — Fatia 6d).    ║
# ║    Sem lógica de domínio: só emite o índice do frame atual.                 ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    AnimationBar                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``AnimationBar`` — playback de frames (slider + play/pause + velocidade) (0017)."""

from __future__ import annotations

from typing import Any

from geosteering_ai.gui.qt_compat import Qt, QtCore, QtWidgets, Signal

__all__ = ["AnimationBar"]

# Velocidades de playback (rótulo → frames por segundo). O intervalo do QTimer
# é ``1000 / fps`` ms. Default 5 fps (legível em ensembles típicos).
_SPEEDS: tuple[tuple[str, int], ...] = (
    ("1 fps", 1),
    ("5 fps", 5),
    ("10 fps", 10),
    ("20 fps", 20),
)
_DEFAULT_SPEED_INDEX = 1  # 5 fps


class AnimationBar(QtWidgets.QWidget):  # type: ignore[misc] # QtWidgets é Any → mypy
    """Barra de animação: play/pause + slider + velocidade, dirigindo um índice de frame.

    Attributes:
        frame_changed: ``Signal(int)`` — emitido a cada novo frame (slider OU timer).

    Example:
        >>> bar = AnimationBar()
        >>> bar.frame_changed.connect(lambda i: vm.__setattr__("focus_model", i))
        >>> bar.set_frame_count(20)   # ensemble de 20 modelos
        >>> bar.set_frame(0)

    Note:
        O ``QTimer`` faz wrap-around (``(frame + 1) % count``) — playback em loop.
        ``set_frame`` bloqueia sinais do slider p/ evitar realimentação (View↔VM).
    """

    frame_changed = Signal(int)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setObjectName("AnimationBar")
        self._count = 0  # nº de frames (== n_models do ensemble)

        # ── Botão play/pause ──────────────────────────────────────────────────
        self._play = QtWidgets.QPushButton("▶")
        self._play.setCheckable(True)
        self._play.setMaximumWidth(40)
        self._play.setToolTip("Play/Pause (varre os modelos do ensemble).")
        self._play.toggled.connect(self._on_play_toggled)

        # ── Slider de frame ───────────────────────────────────────────────────
        self._slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)

        # ── Velocidade (fps) ──────────────────────────────────────────────────
        self._speed = QtWidgets.QComboBox()
        self._speed.addItems([label for label, _fps in _SPEEDS])
        self._speed.setCurrentIndex(_DEFAULT_SPEED_INDEX)
        self._speed.setToolTip("Velocidade do playback (frames por segundo).")
        self._speed.currentIndexChanged.connect(self._on_speed_changed)

        # ── Rótulo "i/n" ──────────────────────────────────────────────────────
        self._label = QtWidgets.QLabel("—")
        self._label.setMinimumWidth(54)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # ── Timer de playback ─────────────────────────────────────────────────
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self._interval_ms())
        self._timer.timeout.connect(self._advance)

        # ── Layout ────────────────────────────────────────────────────────────
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(6, 2, 6, 2)
        row.addWidget(self._play)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(self._label)
        row.addWidget(self._speed)

    # ── API pública ──────────────────────────────────────────────────────────
    def set_frame_count(self, count: int) -> None:
        """Define o nº de frames (range do slider [0, count−1]); pausa se ``count≤1``."""
        self._count = max(0, int(count))
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._count - 1))
        self._slider.blockSignals(False)
        if self._count <= 1:
            self._stop()
        self._refresh_label()

    def set_frame(self, index: int) -> None:
        """Posiciona o slider em ``index`` (sem emitir — evita realimentação View↔VM)."""
        if self._count <= 0:
            self._refresh_label()
            return
        idx = max(0, min(int(index), self._count - 1))
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._refresh_label()

    def current_frame(self) -> int:
        """Índice do frame atual (valor do slider)."""
        return int(self._slider.value())

    def is_playing(self) -> bool:
        """``True`` se o playback (timer) está ativo."""
        return bool(self._timer.isActive())

    # ── Slots internos ─────────────────────────────────────────────────────────
    def _interval_ms(self) -> int:
        """Intervalo do timer (ms) a partir da velocidade (fps) selecionada."""
        _label, fps = _SPEEDS[max(0, min(self._speed.currentIndex(), len(_SPEEDS) - 1))]
        return max(1, int(1000 / fps))

    def _on_play_toggled(self, playing: bool) -> None:
        if playing and self._count > 1:
            self._play.setText("⏸")
            self._timer.start(self._interval_ms())
        else:
            self._stop()

    def _on_speed_changed(self, _idx: int) -> None:
        self._timer.setInterval(self._interval_ms())

    def _on_slider_changed(self, value: int) -> None:
        self._refresh_label()
        self.frame_changed.emit(int(value))

    def _advance(self) -> None:
        """Avança um frame com wrap-around; emite via ``setValue`` (slider → sinal)."""
        if self._count <= 1:
            self._stop()
            return
        nxt = (self.current_frame() + 1) % self._count
        self._slider.setValue(nxt)  # dispara _on_slider_changed → frame_changed

    def _stop(self) -> None:
        """Pausa o playback e reseta o botão (sem mexer no frame atual)."""
        self._timer.stop()
        if self._play.isChecked():
            self._play.blockSignals(True)
            self._play.setChecked(False)
            self._play.blockSignals(False)
        self._play.setText("▶")

    def _refresh_label(self) -> None:
        if self._count <= 0:
            self._label.setText("—")
        else:
            self._label.setText(f"{self.current_frame() + 1}/{self._count}")
