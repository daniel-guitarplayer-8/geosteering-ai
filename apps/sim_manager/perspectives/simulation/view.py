# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/view.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimulatorView — View Qt da perspectiva Simulação           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0011a)         ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — walking skeleton                                ║
# ║  Framework   : Qt6 via gui.qt_compat + gui.plot_backends                   ║
# ║  Dependências: gui.qt_compat, gui.plot_backends, .viewmodel                ║
# ║  Padrão      : View (MVVM) — sem lógica; faz binding aos VMSignals do VM   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Inputs (frequência, dip, nº modelos) + botão Run + status + um          ║
# ║    PlotCanvas. Ao Run: copia os inputs para o ViewModel e chama vm.run().  ║
# ║    Liga-se a ``vm.changed`` (atualiza status/botão) e ``vm.result_ready``  ║
# ║    (plota a resposta EM do modelo 0). Sem lógica de domínio.              ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimulatorView                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimulatorView`` — View Qt (inputs + Run + plot) ligada ao ViewModel (0011a)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
from geosteering_ai.gui.plot_backends import AxisConfig, PlotBackend, make_canvas
from geosteering_ai.gui.qt_compat import QtWidgets

__all__ = ["SimulatorView"]


class SimulatorView(QtWidgets.QWidget):  # type: ignore[misc] # QtWidgets é Any → mypy
    """View da Simulação: inputs + Run + status + 1 plot, ligada a um ViewModel.

    Args:
        vm: o :class:`SimulationViewModel` (a View se liga aos seus VMSignals).
        parent: widget pai opcional.

    Note:
        A View não valida nem simula — só coleta inputs, chama ``vm.run()`` e
        renderiza o que o VM emite (``changed``/``result_ready``).
    """

    def __init__(self, vm: SimulationViewModel, parent: Any = None) -> None:
        super().__init__(parent)
        self._vm = vm

        # ── Inputs ───────────────────────────────────────────────────────────
        self._freq = QtWidgets.QDoubleSpinBox()
        self._freq.setRange(100.0, 1.0e6)
        self._freq.setDecimals(0)
        self._freq.setValue(vm.frequency_hz)
        self._freq.setSuffix(" Hz")

        self._dip = QtWidgets.QDoubleSpinBox()
        self._dip.setRange(0.0, 105.0)
        self._dip.setDecimals(1)
        self._dip.setValue(vm.dip_deg)
        self._dip.setSuffix(" °")

        self._n_models = QtWidgets.QSpinBox()
        self._n_models.setRange(1, 100)
        self._n_models.setValue(vm.n_models)

        self._run_btn = QtWidgets.QPushButton("Run")
        self._status = QtWidgets.QLabel("idle")

        # ── Canvas (matplotlib via plot_backends) ────────────────────────────
        self._canvas = make_canvas(PlotBackend.MATPLOTLIB, parent=self)

        # ── Layout ────────────────────────────────────────────────────────────
        form = QtWidgets.QFormLayout()
        form.addRow("Frequência:", self._freq)
        form.addRow("Dip:", self._dip)
        form.addRow("Nº modelos:", self._n_models)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._run_btn)
        controls.addWidget(self._status)
        controls.addStretch(1)
        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(controls)
        root.addWidget(self._canvas.widget(), stretch=1)

        # ── Binding ──────────────────────────────────────────────────────────
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._vm.changed.connect(self._on_vm_changed)
        self._vm.result_ready.connect(self._on_result_ready)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_run_clicked(self) -> None:
        """Copia os inputs para o VM e dispara a simulação."""
        self._vm.frequency_hz = self._freq.value()
        self._vm.dip_deg = self._dip.value()
        self._vm.n_models = self._n_models.value()
        self._vm.run()

    def _on_vm_changed(self, name: str, value: Any) -> None:
        """Reflete mudanças de estado do VM (status/botão)."""
        if name == "_status":
            self._status.setText(str(value))
            self._run_btn.setEnabled(value != "running")

    def _on_result_ready(self, result: Dict[str, Any]) -> None:
        """Plota a resposta EM (Re do 1º canal, modelo 0) vs. profundidade."""
        # Guard defensivo: o contrato do Service garante o shape, mas a View não
        # confia cegamente (resultado malformado → status, sem crash).
        h6 = result.get("H6")
        positions_z = result.get("positions_z")
        if h6 is None or positions_z is None or getattr(h6, "ndim", 0) != 6:
            self._status.setText("resultado inválido")
            return
        # H6: (n_models, nTR, nAng, n_pos, nf, 9) — modelo 0, TR0, dip0, freq0, canal 0.
        curve = np.real(h6[0, 0, 0, :, 0, 0])

        self._canvas.clear()
        grid = self._canvas.add_subplot_grid(1, 1)
        ax = grid[0][0]
        self._canvas.plot_line(ax, curve, positions_z, label="Re(H₀)")
        self._canvas.set_axis_config(
            ax,
            AxisConfig(
                title="Resposta EM — modelo 0",
                xlabel="Re(H₀)",
                ylabel="Profundidade z (m)",
                invert_y=True,
            ),
        )
        self._canvas.draw()
