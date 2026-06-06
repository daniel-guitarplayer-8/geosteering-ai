# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/simulation/view.py                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimulatorView — View Qt da perspectiva Simulação           ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Simulação (spec 0011c)         ║
# ║  Versão      : v0.3                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — geração estocástica (Fatia 3)                    ║
# ║  Framework   : Qt6 via gui.qt_compat + gui.plot_backends                   ║
# ║  Dependências: gui.qt_compat, gui.plot_backends, gui.services, .viewmodel   ║
# ║  Padrão      : View (MVVM) — sem lógica; faz binding aos VMSignals do VM   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Inputs multi-config (freqs/dips/TRs via CSV) + geometria (h1/tj/p_med/  ║
# ║    nº modelos) + grupo de GEOLOGIA estocástica (gerador, ranges ρₕ/λ,       ║
# ║    distribuição, n_layers, espessura, seed) + label de ``n_pos`` + Run +    ║
# ║    status + PlotCanvas. Ao Run: copia os inputs para o ViewModel e chama    ║
# ║    vm.run(). Liga-se a ``vm.changed`` (status/botão) e ``vm.result_ready``  ║
# ║    (plota a resposta EM do modelo 0). Sem lógica de domínio.              ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimulatorView                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimulatorView`` — View Qt (inputs + geologia + Run + galeria + .session), VM (0011d)."""

from __future__ import annotations

from typing import Any, Tuple

from apps.sim_manager.perspectives.simulation.results_view import ResultsView
from apps.sim_manager.perspectives.simulation.viewmodel import SimulationViewModel
from geosteering_ai.gui.persistence.session import SessionDocument
from geosteering_ai.gui.qt_compat import QtWidgets
from geosteering_ai.gui.services.sim_request import compute_n_pos
from geosteering_ai.gui.services.stochastic_geology import GENERATORS_AVAILABLE

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
    # ``;`` é aceito como separador (paridade c/ o monólito ``_parse_float_list``).
    for tok in text.replace(";", ",").split(","):
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
        self._freqs.setToolTip(
            "Frequências (Hz), separadas por vírgula. Range [10, 2e6]."
        )
        self._dips = QtWidgets.QLineEdit(self._fmt_csv(vm.dips))
        self._dips.setPlaceholderText("ex.: 0, 30")
        self._dips.setToolTip(
            "Ângulos de mergulho (°), separados por vírgula. Range [0, 90]."
        )
        self._trs = QtWidgets.QLineEdit(self._fmt_csv(vm.tr_spacings))
        self._trs.setPlaceholderText("ex.: 1.0, 2.0")
        self._trs.setToolTip(
            "Espaçamentos T-R (m), separados por vírgula. Range [0.1, 50]."
        )

        # ── Geometria Fortran (spinboxes) + nº modelos ───────────────────────
        # Ranges/steps espelham a ParametersPage do monólito (simulation_manager.py
        # :1112-1114) p/ guardrail visual consistente.
        self._h1 = QtWidgets.QDoubleSpinBox()
        self._h1.setRange(0.1, 500.0)
        self._h1.setDecimals(3)
        self._h1.setSingleStep(0.5)
        self._h1.setValue(vm.h1)
        self._h1.setSuffix(" m")
        self._h1.setToolTip(
            "Altura do 1º ponto-médio T-R acima da 1ª interface (convenção Fortran). "
            "z_obs começa em −h1 (interfaces em z=0)."
        )

        self._tj = QtWidgets.QDoubleSpinBox()
        self._tj.setRange(1.0, 5000.0)
        self._tj.setDecimals(2)
        self._tj.setSingleStep(1.0)
        self._tj.setValue(vm.tj)
        self._tj.setSuffix(" m")
        self._tj.setToolTip(
            "Janela de investigação (m). Também é o total_depth da geologia "
            "estocástica (Σ espessuras = tj). n_pos = ceil(tj/(p_med·cos dip₀))."
        )

        self._p_med = QtWidgets.QDoubleSpinBox()
        self._p_med.setRange(0.01, 10.0)
        self._p_med.setDecimals(3)
        self._p_med.setSingleStep(0.01)
        self._p_med.setValue(vm.p_med)
        self._p_med.setSuffix(" m")
        self._p_med.setToolTip("Passo entre medidas (m). Menor passo ⇒ mais posições.")

        self._n_models = QtWidgets.QSpinBox()
        # Paridade com o monólito (spin_nmodels: 1–10M, passo 100). O cap antigo
        # de 100 impedia ensembles grandes. NOTA: sem ProcessPool (Fatia 5), N
        # grande roda lento (off-thread, UI responsiva) — sem barra de progresso ainda.
        self._n_models.setRange(1, 10_000_000)
        self._n_models.setSingleStep(100)
        self._n_models.setValue(vm.n_models)
        self._n_models.setToolTip(
            "Nº de perfis geológicos a gerar/simular (1–10M). Sem ProcessPool "
            "(Fatia 5), N grande roda lento — a UI fica responsiva (off-thread)."
        )

        # Backend de simulação (spec 0012): numba (in-thread) | jax | auto (subprocesso).
        self._sim_backend = QtWidgets.QComboBox()
        self._sim_backend.addItems(["numba", "jax", "auto"])
        self._sim_backend.setCurrentText(vm.backend)
        self._sim_backend.setToolTip(
            "Motor de simulação: numba (CPU, in-thread) · jax (GPU, subprocesso TLS-safe) "
            "· auto (decide GPU/CPU por tamanho/geometria). jax/auto rodam em subprocesso."
        )

        # ── Geologia estocástica (Fatia 3) ───────────────────────────────────
        self._geology_box = self._build_geology_group(vm)

        self._n_pos_label = QtWidgets.QLabel("n_pos: —")
        # ── Execução & feedback (Fatia 6a) ───────────────────────────────────
        self._run_btn = QtWidgets.QPushButton("▶  Iniciar")
        self._run_btn.setProperty("role", "primary")  # pill accent (QSS)
        self._pause_btn = QtWidgets.QPushButton("⏸  Pausar")
        self._pause_btn.setCheckable(True)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setToolTip(
            "Pausa/retoma entre grupos (cooperativo; backend numba). No-op p/ jax."
        )
        self._cancel_btn = QtWidgets.QPushButton("⏹  Cancelar")
        self._cancel_btn.setProperty("role", "danger")  # pill vermelho (QSS)
        self._cancel_btn.setEnabled(False)
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._save_btn = QtWidgets.QPushButton("Salvar sessão…")
        self._open_btn = QtWidgets.QPushButton("Abrir sessão…")
        self._status = QtWidgets.QLabel("● ocioso")

        # ── Galeria do ensemble (Fatia 4) — substitui o plot único ───────────
        self._results = ResultsView(vm.results, parent=self)

        # ── Layout ────────────────────────────────────────────────────────────
        form = QtWidgets.QFormLayout()
        form.addRow("Frequências (Hz):", self._freqs)
        form.addRow("Dips (°):", self._dips)
        form.addRow("Espaçamentos TR (m):", self._trs)
        form.addRow("h1:", self._h1)
        form.addRow("tj:", self._tj)
        form.addRow("p_med:", self._p_med)
        form.addRow("Nº modelos:", self._n_models)
        form.addRow("Backend:", self._sim_backend)
        form.addRow("Posições derivadas:", self._n_pos_label)
        # Linha 1: execução (Iniciar/Pausar/Cancelar) + progresso.
        exec_row = QtWidgets.QHBoxLayout()
        exec_row.addWidget(self._run_btn)
        exec_row.addWidget(self._pause_btn)
        exec_row.addWidget(self._cancel_btn)
        exec_row.addWidget(self._progress_bar, stretch=1)
        # Linha 2: sessão + status (estado · elapsed · throughput).
        session_row = QtWidgets.QHBoxLayout()
        session_row.addWidget(self._save_btn)
        session_row.addWidget(self._open_btn)
        session_row.addWidget(self._status)
        session_row.addStretch(1)
        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self._geology_box)
        root.addLayout(exec_row)
        root.addLayout(session_row)
        root.addWidget(self._results, stretch=1)

        # ── Binding ──────────────────────────────────────────────────────────
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._save_btn.clicked.connect(self._on_save_session)
        self._open_btn.clicked.connect(self._on_open_session)
        self._vm.changed.connect(self._on_vm_changed)
        # n_pos derivado: recalcula ao editar dips/tj/p_med (convenção Fortran).
        self._dips.textChanged.connect(self._refresh_n_pos)
        self._tj.valueChanged.connect(self._refresh_n_pos)
        self._p_med.valueChanged.connect(self._refresh_n_pos)
        self._refresh_n_pos()
        # Geologia: habilita os campos só no modo "stochastic" (UX reativa —
        # no modo "fixed" eles são ignorados, então ficam desabilitados/cinza).
        self._geo_mode.currentTextChanged.connect(self._on_geology_mode_changed)
        self._on_geology_mode_changed(self._geo_mode.currentText())

    # ── Helpers ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _fmt_csv(values: Tuple[float, ...]) -> str:
        """Formata uma tupla de floats em CSV (``(20000.0,) → "20000"``)."""
        return ", ".join(f"{v:g}" for v in values)

    def _build_geology_group(self, vm: SimulationViewModel) -> Any:
        """Constrói o QGroupBox de geologia estocástica (widgets + valores iniciais).

        Espelha os recursos do SM monolítico (combo gerador, ranges ρₕ/λ,
        distribuição, n_layers fixo/range, espessura mínima, seed). A validação
        de ranges é do VM; aqui só coletamos os valores.
        """
        # combos
        self._geo_mode = QtWidgets.QComboBox()
        self._geo_mode.addItems(["stochastic", "fixed"])
        self._geo_mode.setCurrentText(vm.geology_mode)
        self._geo_generator = QtWidgets.QComboBox()
        self._geo_generator.addItems(list(GENERATORS_AVAILABLE))
        self._geo_generator.setCurrentText(vm.generator)
        self._geo_distr = QtWidgets.QComboBox()
        self._geo_distr.addItems(["loguni", "uniform"])
        self._geo_distr.setCurrentText(vm.rho_h_distribution)
        # ρₕ range
        self._geo_rho_min = QtWidgets.QDoubleSpinBox()
        self._geo_rho_min.setRange(0.001, 1.0e6)
        self._geo_rho_min.setDecimals(3)
        self._geo_rho_min.setValue(vm.rho_h_min)
        self._geo_rho_min.setSuffix(" Ω·m")
        self._geo_rho_max = QtWidgets.QDoubleSpinBox()
        self._geo_rho_max.setRange(0.001, 1.0e6)
        self._geo_rho_max.setDecimals(3)
        self._geo_rho_max.setValue(vm.rho_h_max)
        self._geo_rho_max.setSuffix(" Ω·m")
        # anisotropia λ
        self._geo_aniso = QtWidgets.QCheckBox("Anisotrópico (ρᵥ=λ²·ρₕ)")
        self._geo_aniso.setChecked(vm.anisotropic)
        self._geo_lambda_min = QtWidgets.QDoubleSpinBox()
        self._geo_lambda_min.setRange(1.0, 5.0)
        self._geo_lambda_min.setDecimals(3)
        self._geo_lambda_min.setSingleStep(0.05)  # passo do monólito
        self._geo_lambda_min.setValue(vm.lambda_min)
        self._geo_lambda_max = QtWidgets.QDoubleSpinBox()
        self._geo_lambda_max.setRange(1.0, 5.0)
        self._geo_lambda_max.setDecimals(3)
        self._geo_lambda_max.setSingleStep(0.05)
        self._geo_lambda_max.setValue(vm.lambda_max)
        # n_layers (fixo OU range)
        self._geo_nlf_check = QtWidgets.QCheckBox("n_layers fixo")
        self._geo_nlf_check.setChecked(vm.n_layers_fixed is not None)
        self._geo_nlf = QtWidgets.QSpinBox()
        self._geo_nlf.setRange(3, 200)
        self._geo_nlf.setValue(
            vm.n_layers_fixed if vm.n_layers_fixed is not None else 5
        )
        self._geo_nl_min = QtWidgets.QSpinBox()
        self._geo_nl_min.setRange(3, 200)
        self._geo_nl_min.setValue(vm.n_layers_min)
        self._geo_nl_min.setToolTip("Nº mínimo de camadas (inclui 2 semi-espaços).")
        # n_layers máx é INCLUSIVE na UI (paridade c/ o monólito, que faz value()+1
        # ao montar GenConfig). O VM guarda EXCLUSIVE; convertemos aqui (init −1,
        # copy-back +1) — assim "máx = N" significa "até N camadas", como no monólito.
        self._geo_nl_max = QtWidgets.QSpinBox()
        self._geo_nl_max.setRange(3, 200)
        self._geo_nl_max.setValue(vm.n_layers_max - 1)
        self._geo_nl_max.setToolTip(
            "Nº máximo de camadas (INCLUSIVE — 'até N'). Internamente vira o bound "
            "exclusive do gerador (N+1)."
        )
        # espessura mínima
        self._geo_min_thick = QtWidgets.QDoubleSpinBox()
        self._geo_min_thick.setRange(0.01, 100.0)
        self._geo_min_thick.setDecimals(2)
        self._geo_min_thick.setValue(vm.min_thickness)
        self._geo_min_thick.setSuffix(" m")
        # seed
        self._geo_seed_random = QtWidgets.QCheckBox("Semente aleatória")
        self._geo_seed_random.setChecked(vm.rng_seed is None)
        self._geo_seed = QtWidgets.QSpinBox()
        self._geo_seed.setRange(0, 2_147_483_647)
        self._geo_seed.setValue(vm.rng_seed if vm.rng_seed is not None else 42)

        # layout em form
        gform = QtWidgets.QFormLayout()
        gform.addRow("Modo:", self._geo_mode)
        gform.addRow("Gerador:", self._geo_generator)
        gform.addRow("Distribuição ρₕ:", self._geo_distr)
        gform.addRow("ρₕ mín:", self._geo_rho_min)
        gform.addRow("ρₕ máx:", self._geo_rho_max)
        gform.addRow("", self._geo_aniso)
        gform.addRow("λ mín:", self._geo_lambda_min)
        gform.addRow("λ máx:", self._geo_lambda_max)
        gform.addRow("", self._geo_nlf_check)
        gform.addRow("n_layers (fixo):", self._geo_nlf)
        gform.addRow("n_layers mín:", self._geo_nl_min)
        gform.addRow("n_layers máx:", self._geo_nl_max)
        gform.addRow("Espessura mín:", self._geo_min_thick)
        gform.addRow("", self._geo_seed_random)
        gform.addRow("Semente (fixa):", self._geo_seed)
        box = QtWidgets.QGroupBox("Geologia estocástica")
        box.setLayout(gform)
        return box

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
        self._vm.backend = self._sim_backend.currentText()
        # ── Geologia estocástica (Fatia 3) ───────────────────────────────────
        self._vm.geology_mode = self._geo_mode.currentText()
        self._vm.generator = self._geo_generator.currentText()
        self._vm.rho_h_distribution = self._geo_distr.currentText()
        self._vm.rho_h_min = self._geo_rho_min.value()
        self._vm.rho_h_max = self._geo_rho_max.value()
        self._vm.anisotropic = self._geo_aniso.isChecked()
        self._vm.lambda_min = self._geo_lambda_min.value()
        self._vm.lambda_max = self._geo_lambda_max.value()
        self._vm.min_thickness = self._geo_min_thick.value()
        self._vm.n_layers_fixed = (
            self._geo_nlf.value() if self._geo_nlf_check.isChecked() else None
        )
        self._vm.n_layers_min = self._geo_nl_min.value()
        # spin INCLUSIVE → bound EXCLUSIVE do gerador (paridade monólito build_gen_config).
        self._vm.n_layers_max = self._geo_nl_max.value() + 1
        self._vm.rng_seed = (
            None if self._geo_seed_random.isChecked() else self._geo_seed.value()
        )
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

    def _on_geology_mode_changed(self, text: str) -> None:
        """Habilita os campos de geologia só no modo ``"stochastic"`` (UX reativa)."""
        enabled = text == "stochastic"
        for w in (
            self._geo_generator,
            self._geo_distr,
            self._geo_rho_min,
            self._geo_rho_max,
            self._geo_aniso,
            self._geo_lambda_min,
            self._geo_lambda_max,
            self._geo_nlf_check,
            self._geo_nlf,
            self._geo_nl_min,
            self._geo_nl_max,
            self._geo_min_thick,
            self._geo_seed_random,
            self._geo_seed,
        ):
            w.setEnabled(enabled)

    def _on_pause_clicked(self) -> None:
        """Toggle Pausar/Retomar (cooperativo; numba). Delega ao ViewModel."""
        if self._pause_btn.isChecked():
            self._vm.request_pause()
            self._pause_btn.setText("▶  Retomar")
        else:
            self._vm.request_resume()
            self._pause_btn.setText("⏸  Pausar")

    def _on_cancel_clicked(self) -> None:
        """Solicita o cancelamento cooperativo da simulação em voo."""
        self._vm.request_cancel()

    def _on_vm_changed(self, name: str, value: Any) -> None:
        """Reflete o estado do VM: botões, barra de progresso e status (Fatia 6a)."""
        if name not in ("_status", "_progress"):
            return
        state = self._vm.status
        running = state == "running"
        # Botões por estado (1 simulação por vez; o guard real vive no VM).
        self._run_btn.setEnabled(not running)
        self._pause_btn.setEnabled(running)
        self._cancel_btn.setEnabled(running)
        if not running and self._pause_btn.isChecked():
            self._pause_btn.setChecked(False)
            self._pause_btn.setText("⏸  Pausar")
        # Barra de progresso (0–100%).
        total = max(1, self._vm.progress_total)
        self._progress_bar.setValue(int(100 * self._vm.progress_done / total))
        # Status: estado · elapsed · throughput (ou mensagem de erro).
        color = {
            "error": "#ef4444",
            "running": "#6366f1",
            "done": "#10b981",
            "cancelled": "#f59e0b",
        }.get(state, "")
        self._status.setStyleSheet(f"color: {color};" if color else "")
        if state == "error":
            lr = self._vm.last_result or {}
            msg = lr.get("error") or "; ".join(lr.get("errors", [])) or "erro"
            self._status.setText(f"[ERRO] {msg}")
        else:
            info = self._vm.status_display
            self._status.setText(
                f"{info['state']} · {info['elapsed']} · {info['throughput']}"
            )

    # ── Persistência .session (params; resultado reproduzível pela seed) ─────
    def _on_save_session(self) -> None:
        """Salva os params atuais num ``.session`` (JSON, sem pickle)."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Salvar sessão", "", "Sessão (*.session)"
        )
        if not path:
            return
        if not path.endswith(".session"):
            path += ".session"
        # coleta os inputs ATUAIS para o VM (sem disparar run) antes de serializar.
        self._push_inputs_to_vm()
        try:
            SessionDocument(data=self._vm.to_session_dict()).save(path)
            self._status.setText(f"sessão salva: {path}")
        except OSError as exc:
            self._status.setText(f"falha ao salvar: {exc}")

    def _on_open_session(self) -> None:
        """Abre um ``.session`` e repopula os params (+ os widgets de input)."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir sessão", "", "Sessão (*.session)"
        )
        if not path:
            return
        try:
            doc = SessionDocument.load(path)
        except (OSError, ValueError) as exc:
            self._status.setText(f"falha ao abrir: {exc}")
            return
        self._vm.load_session_dict(doc.data)
        self._sync_inputs_from_vm()
        self._status.setText(f"sessão carregada: {path}")

    def _push_inputs_to_vm(self) -> None:
        """Copia os widgets de input para o VM (sem disparar run) — pré-serialização."""
        try:
            self._vm.frequencies = _parse_csv_floats(self._freqs.text())
            self._vm.dips = _parse_csv_floats(self._dips.text())
            self._vm.tr_spacings = _parse_csv_floats(self._trs.text())
        except ValueError:
            pass  # mantém o último válido; a validação real é no Run
        self._vm.h1 = self._h1.value()
        self._vm.tj = self._tj.value()
        self._vm.p_med = self._p_med.value()
        self._vm.n_models = self._n_models.value()
        self._vm.backend = self._sim_backend.currentText()
        self._vm.geology_mode = self._geo_mode.currentText()
        self._vm.generator = self._geo_generator.currentText()
        self._vm.rho_h_distribution = self._geo_distr.currentText()
        self._vm.rho_h_min = self._geo_rho_min.value()
        self._vm.rho_h_max = self._geo_rho_max.value()
        self._vm.anisotropic = self._geo_aniso.isChecked()
        self._vm.lambda_min = self._geo_lambda_min.value()
        self._vm.lambda_max = self._geo_lambda_max.value()
        self._vm.min_thickness = self._geo_min_thick.value()
        self._vm.n_layers_fixed = (
            self._geo_nlf.value() if self._geo_nlf_check.isChecked() else None
        )
        self._vm.n_layers_min = self._geo_nl_min.value()
        self._vm.n_layers_max = self._geo_nl_max.value() + 1
        self._vm.rng_seed = (
            None if self._geo_seed_random.isChecked() else self._geo_seed.value()
        )

    def _sync_inputs_from_vm(self) -> None:
        """Reflete o estado do VM nos widgets de input (após abrir uma sessão)."""
        self._freqs.setText(self._fmt_csv(self._vm.frequencies))
        self._dips.setText(self._fmt_csv(self._vm.dips))
        self._trs.setText(self._fmt_csv(self._vm.tr_spacings))
        self._h1.setValue(self._vm.h1)
        self._tj.setValue(self._vm.tj)
        self._p_med.setValue(self._vm.p_med)
        self._n_models.setValue(self._vm.n_models)
        self._sim_backend.setCurrentText(self._vm.backend)
        self._geo_mode.setCurrentText(self._vm.geology_mode)
        self._geo_generator.setCurrentText(self._vm.generator)
        self._geo_distr.setCurrentText(self._vm.rho_h_distribution)
        self._geo_rho_min.setValue(self._vm.rho_h_min)
        self._geo_rho_max.setValue(self._vm.rho_h_max)
        self._geo_aniso.setChecked(self._vm.anisotropic)
        self._geo_lambda_min.setValue(self._vm.lambda_min)
        self._geo_lambda_max.setValue(self._vm.lambda_max)
        self._geo_min_thick.setValue(self._vm.min_thickness)
        self._geo_nlf_check.setChecked(self._vm.n_layers_fixed is not None)
        if self._vm.n_layers_fixed is not None:
            self._geo_nlf.setValue(self._vm.n_layers_fixed)
        self._geo_nl_min.setValue(self._vm.n_layers_min)
        self._geo_nl_max.setValue(self._vm.n_layers_max - 1)  # exclusive → inclusive
        self._geo_seed_random.setChecked(self._vm.rng_seed is None)
        if self._vm.rng_seed is not None:
            self._geo_seed.setValue(self._vm.rng_seed)
        self._refresh_n_pos()
