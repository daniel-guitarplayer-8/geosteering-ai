# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/view.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimulatorView — View Qt da perspectiva Simulação           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0011b)         ║
# ║  Versão      : v0.2                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — params completos (Fatia 2)                       ║
# ║  Framework   : Qt6 via gui.qt_compat + gui.plot_backends                   ║
# ║  Dependências: gui.qt_compat, gui.plot_backends, .viewmodel                ║
# ║  Padrão      : View (MVVM) — sem lógica; faz binding aos VMSignals do VM   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Inputs multi-config (freqs/dips/TRs via CSV) + geometria (h1/tj/p_med/  ║
# ║    nº modelos via spinbox) + label de ``n_pos`` derivado + Run + status +  ║
# ║    um PlotCanvas. Ao Run: copia os inputs para o ViewModel e chama          ║
# ║    vm.run(). Liga-se a ``vm.changed`` (status/botão) e ``vm.result_ready``  ║
# ║    (plota a resposta EM do modelo 0). Sem lógica de domínio.              ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimulatorView                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimulatorView`` — View Qt (inputs multi-config + Run + plot) ligada ao VM (0011b)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
from geosteering_ai.gui.plot_backends import AxisConfig, PlotBackend, make_canvas
from geosteering_ai.gui.qt_compat import QtWidgets
from geosteering_ai.gui.services.sim_request import compute_n_pos

__all__ = ["SimulatorView"]


def _parse_csv_floats(text: str) -> Tuple[float, ...]:
    """Converte CSV (``"20000, 40000"``) em tupla de floats (tokens vazios ignorados).

    Args:
        text: texto CSV separado por vírgula.

    Returns:
        Tupla de floats na ordem do texto (vazia se o texto for vazio/só vírgulas).

    Raises:
        ValueError: se algum token não-vazio não for numérico (mensagem clara em
            PT-BR; o chamador exibe no status, sem crash).
    """
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            raise ValueError(f"valor não numérico: '{tok}'") from None
    return tuple(out)


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

        # ── Inputs de listas (CSV multi-config) ──────────────────────────────
        self._freqs = QtWidgets.QLineEdit(self._fmt_csv(vm.frequencies))
        self._freqs.setPlaceholderText("ex.: 20000, 40000")
        self._dips = QtWidgets.QLineEdit(self._fmt_csv(vm.dips))
        self._dips.setPlaceholderText("ex.: 0, 30")
        self._trs = QtWidgets.QLineEdit(self._fmt_csv(vm.tr_spacings))
        self._trs.setPlaceholderText("ex.: 1.0, 2.0")

        # ── Geometria Fortran (spinboxes) + nº modelos ───────────────────────
        self._h1 = QtWidgets.QDoubleSpinBox()
        self._h1.setRange(0.01, 1000.0)
        self._h1.setDecimals(2)
        self._h1.setValue(vm.h1)
        self._h1.setSuffix(" m")

        self._tj = QtWidgets.QDoubleSpinBox()
        self._tj.setRange(0.1, 10000.0)
        self._tj.setDecimals(2)
        self._tj.setValue(vm.tj)
        self._tj.setSuffix(" m")

        self._p_med = QtWidgets.QDoubleSpinBox()
        self._p_med.setRange(0.01, 100.0)
        self._p_med.setDecimals(2)
        self._p_med.setValue(vm.p_med)
        self._p_med.setSuffix(" m")

        self._n_models = QtWidgets.QSpinBox()
        self._n_models.setRange(1, 100)
        self._n_models.setValue(vm.n_models)

        self._n_pos_label = QtWidgets.QLabel("n_pos: —")
        self._run_btn = QtWidgets.QPushButton("Run")
        self._status = QtWidgets.QLabel("idle")

        # ── Canvas (matplotlib via plot_backends) ────────────────────────────
        self._canvas = make_canvas(PlotBackend.MATPLOTLIB, parent=self)

        # ── Layout ────────────────────────────────────────────────────────────
        form = QtWidgets.QFormLayout()
        form.addRow("Frequências (Hz):", self._freqs)
        form.addRow("Dips (°):", self._dips)
        form.addRow("Espaçamentos TR (m):", self._trs)
        form.addRow("h1:", self._h1)
        form.addRow("tj:", self._tj)
        form.addRow("p_med:", self._p_med)
        form.addRow("Nº modelos:", self._n_models)
        form.addRow("Posições derivadas:", self._n_pos_label)
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
        # n_pos derivado: recalcula ao editar dips/tj/p_med (convenção Fortran).
        self._dips.textChanged.connect(self._refresh_n_pos)
        self._tj.valueChanged.connect(self._refresh_n_pos)
        self._p_med.valueChanged.connect(self._refresh_n_pos)
        self._refresh_n_pos()

    # ── Helpers ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _fmt_csv(values: Tuple[float, ...]) -> str:
        """Formata uma tupla de floats em CSV (``(20000.0,) → "20000"``)."""
        return ", ".join(f"{v:g}" for v in values)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_run_clicked(self) -> None:
        """Parseia os inputs, copia para o VM e dispara a simulação.

        CSV inválido (token não-numérico) → status de erro, SEM chamar o VM e
        SEM crash. A validação de ranges (errata) é responsabilidade do VM.
        """
        try:
            freqs = _parse_csv_floats(self._freqs.text())
            dips = _parse_csv_floats(self._dips.text())
            trs = _parse_csv_floats(self._trs.text())
        except ValueError as exc:
            self._status.setText(f"entrada inválida: {exc}")
            return
        self._vm.frequencies = freqs
        self._vm.dips = dips
        self._vm.tr_spacings = trs
        self._vm.h1 = self._h1.value()
        self._vm.tj = self._tj.value()
        self._vm.p_med = self._p_med.value()
        self._vm.n_models = self._n_models.value()
        self._vm.run()

    def _refresh_n_pos(self, *_: Any) -> None:
        """Atualiza o label ``n_pos`` (convenção Fortran) a partir de dips/tj/p_med.

        CSV de dips inválido/vazio ou ``p_med``/``tj`` ≤ 0 → exibe ``—`` (a
        validação real é no VM; aqui é só feedback visual, sem crash).
        """
        try:
            dips = _parse_csv_floats(self._dips.text())
        except ValueError:
            self._n_pos_label.setText("n_pos: —")
            return
        tj, p_med = self._tj.value(), self._p_med.value()
        if not dips or p_med <= 0.0 or tj <= 0.0:
            self._n_pos_label.setText("n_pos: —")
            return
        self._n_pos_label.setText(f"n_pos: {compute_n_pos(tj, p_med, dips[0])}")

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
