# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_animation_bar.py                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — EnsembleAnimationBar                  ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Status      : Produção (v2.6b)                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Widget de barra de animação para navegar pelos n_models de um          ║
# ║    ensemble simulado. Substitui clique manual em spin_model_idx por       ║
# ║    slider + play/pause + speed control (frames per second).               ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    bar = EnsembleAnimationBar(parent)                                     ║
# ║    bar.setMaximum(n_models - 1)                                           ║
# ║    bar.valueChanged.connect(spin_model_idx.setValue)                      ║
# ║    layout.addWidget(bar)                                                  ║
# ║                                                                           ║
# ║  CONTROLES                                                                ║
# ║    [◀◀ ◀ ▶ ▶ ▶▶]   [───slider───]   [speed fps]   [Modelo: 5 / 100]      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""EnsembleAnimationBar — slider + play/pause para ensemble (v2.6b)."""
from __future__ import annotations

from typing import Optional

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    _QT = "PyQt6"
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

        _QT = "PyQt5"
    except ImportError:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

        _QT = "PySide6"


# Escolhe Signal compatível com binding (PyQt usa pyqtSignal; PySide usa Signal)
if _QT in ("PyQt6", "PyQt5"):
    _Signal = QtCore.pyqtSignal  # type: ignore[attr-defined]
else:
    _Signal = QtCore.Signal  # type: ignore[attr-defined]


__all__ = ["EnsembleAnimationBar"]


class EnsembleAnimationBar(QtWidgets.QWidget):
    """Barra de animação para navegar pelos modelos de um ensemble.

    Attributes:
        valueChanged: Signal emitido quando o índice do modelo muda.
            Conectado tipicamente a ``spin_model_idx.setValue``.

    Note:
        Quando ``setMaximum(0)`` (ensemble com apenas 1 modelo), a barra
        é automaticamente escondida via ``setVisible(False)``.
    """

    valueChanged = _Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.btn_first = QtWidgets.QToolButton()
        self.btn_first.setText("⏮")
        self.btn_first.setToolTip("Primeiro modelo (0)")

        self.btn_prev = QtWidgets.QToolButton()
        self.btn_prev.setText("◀")
        self.btn_prev.setToolTip("Modelo anterior")

        self.btn_play = QtWidgets.QToolButton()
        self.btn_play.setText("▶")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip("Play / Pause animação (loop nos n modelos)")

        self.btn_next = QtWidgets.QToolButton()
        self.btn_next.setText("▶")
        self.btn_next.setToolTip("Próximo modelo")

        self.btn_last = QtWidgets.QToolButton()
        self.btn_last.setText("⏭")
        self.btn_last.setToolTip("Último modelo")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)

        self.spin_speed = QtWidgets.QDoubleSpinBox()
        self.spin_speed.setMinimum(0.5)
        self.spin_speed.setMaximum(60.0)
        self.spin_speed.setValue(5.0)
        self.spin_speed.setSuffix(" fps")
        self.spin_speed.setDecimals(1)
        self.spin_speed.setToolTip(
            "Velocidade da animação em frames por segundo (0.5 – 60)."
        )

        self.lbl_pos = QtWidgets.QLabel("Modelo: 0 / 0")
        self.lbl_pos.setMinimumWidth(120)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(False)
        self._timer.timeout.connect(self._tick)

        # Layout horizontal: [btns play] [slider stretch] [speed] [label]
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(6)
        for btn in (
            self.btn_first,
            self.btn_prev,
            self.btn_play,
            self.btn_next,
            self.btn_last,
        ):
            row.addWidget(btn)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin_speed)
        row.addWidget(self.lbl_pos)

        # Conexões
        self.btn_first.clicked.connect(lambda: self.setValue(0))
        self.btn_prev.clicked.connect(lambda: self.setValue(self.value() - 1))
        self.btn_next.clicked.connect(lambda: self.setValue(self.value() + 1))
        self.btn_last.clicked.connect(lambda: self.setValue(self.maximum()))
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin_speed.valueChanged.connect(self._on_speed_changed)

        self.setMaximum(0)  # esconde inicialmente

    # ── API pública ───────────────────────────────────────────────────────

    def setMaximum(self, n: int) -> None:
        """Define o índice máximo (n_models - 1). 0 esconde a barra."""
        n = max(0, int(n))
        self.slider.setMaximum(n)
        self._update_label()
        # Esconde se ensemble tem apenas 1 modelo
        self.setVisible(n > 0)
        # Para a animação se índice atual ficou fora do range
        if self.value() > n:
            self.setValue(n)

    def maximum(self) -> int:
        return int(self.slider.maximum())

    def setValue(self, i: int) -> None:
        """Atualiza o slider e dispara ``valueChanged``."""
        i = max(0, min(int(i), self.maximum()))
        if i != self.slider.value():
            self.slider.setValue(i)
        else:
            # Mesmo valor — força emissão do sinal e label update
            self._update_label()
            self.valueChanged.emit(i)

    def value(self) -> int:
        return int(self.slider.value())

    def stop(self) -> None:
        """Para a animação se estiver rodando."""
        if self.btn_play.isChecked():
            self.btn_play.setChecked(False)

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_slider_changed(self, i: int) -> None:
        self._update_label()
        self.valueChanged.emit(int(i))

    def _on_play_toggled(self, checked: bool) -> None:
        self.btn_play.setText("⏸" if checked else "▶")
        if checked:
            interval_ms = max(16, int(1000.0 / max(0.5, self.spin_speed.value())))
            self._timer.start(interval_ms)
        else:
            self._timer.stop()

    def _on_speed_changed(self, fps: float) -> None:
        if self._timer.isActive():
            self._timer.setInterval(max(16, int(1000.0 / max(0.5, fps))))

    def _tick(self) -> None:
        i = self.value() + 1
        if i > self.maximum():
            i = 0  # loop
        self.slider.setValue(i)

    def _update_label(self) -> None:
        self.lbl_pos.setText(f"Modelo: {self.value()} / {self.maximum()}")

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._timer.stop()
        super().closeEvent(event)
