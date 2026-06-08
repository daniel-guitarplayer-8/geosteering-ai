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


def _heading_title(text: str) -> Any:
    """``QLabel`` de título de seção (``role="heading"`` — estilizado pelo QSS)."""
    lbl = QtWidgets.QLabel(text)
    lbl.setProperty("role", "heading")
    return lbl


def _hint(text: str) -> Any:
    """``QLabel`` de subtítulo descritivo (``role="hint"`` — estilizado pelo QSS)."""
    lbl = QtWidgets.QLabel(text)
    lbl.setProperty("role", "hint")
    lbl.setWordWrap(True)
    return lbl


def _group(title: str, hint: str, inner: Any) -> Any:
    """``QGroupBox`` (card) com subtítulo ``role="hint"`` + um layout/widget interno."""
    box = QtWidgets.QGroupBox(title)
    vbox = QtWidgets.QVBoxLayout(box)
    if hint:
        vbox.addWidget(_hint(hint))
    if isinstance(inner, QtWidgets.QLayout):
        vbox.addLayout(inner)
    else:
        vbox.addWidget(inner)
    return box


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

        # ── Contadores derivados (nf/ntheta/nº pares T-R) ────────────────────
        # Read-only: derivados dos CSV de freqs/dips/TRs (atualizados em textChanged).
        self._nf_label = QtWidgets.QLabel("—")
        self._ntheta_label = QtWidgets.QLabel("—")
        self._ntr_label = QtWidgets.QLabel("—")

        # ── Geologia estocástica (Fatia 3) ───────────────────────────────────
        self._geology_box = self._build_geology_group(vm)
        # ── Paralelismo (Lote 1) + Saída ─────────────────────────────────────
        self._parallel_box = self._build_parallelism_group(vm)
        self._output_box = self._build_output_group(vm)

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
        # Geometria + aquisição (com contadores derivados no topo, espelhando o
        # monólito: "Nº de frequências (nf)", "Nº de ângulos (ntheta)", "Nº de
        # pares T-R" como rótulos read-only acima dos CSV que os definem).
        form = QtWidgets.QFormLayout()
        form.addRow("Nº de frequências (nf):", self._nf_label)
        form.addRow("Nº de ângulos (ntheta):", self._ntheta_label)
        form.addRow("Nº de pares T-R:", self._ntr_label)
        form.addRow("Frequências (Hz):", self._freqs)
        form.addRow("Dips (°):", self._dips)
        form.addRow("Espaçamentos TR (m):", self._trs)
        form.addRow("h1:", self._h1)
        form.addRow("tj:", self._tj)
        form.addRow("p_med:", self._p_med)
        form.addRow("Nº modelos:", self._n_models)
        form.addRow("Backend:", self._sim_backend)
        form.addRow("Posições derivadas:", self._n_pos_label)
        geometry_box = _group(
            "Geometria da ferramenta e aquisição",
            "Configure a geometria da ferramenta LWD e a aquisição. Passe o mouse "
            "sobre cada campo para uma explicação detalhada.",
            form,
        )
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
        root.addWidget(_heading_title("Parâmetros da simulação"))
        root.addWidget(geometry_box)
        root.addWidget(self._geology_box)
        root.addWidget(self._parallel_box)
        root.addWidget(self._output_box)
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
        # Contadores nf/ntheta/nTR derivados dos CSV (atualizam ao digitar).
        self._freqs.textChanged.connect(self._refresh_counts)
        self._dips.textChanged.connect(self._refresh_counts)
        self._trs.textChanged.connect(self._refresh_counts)
        self._refresh_counts()
        # Geologia: habilita os campos só no modo "stochastic" (UX reativa —
        # no modo "fixed" eles são ignorados, então ficam desabilitados/cinza).
        self._geo_mode.currentTextChanged.connect(self._on_geology_mode_changed)
        # Fatia 6b — perfil canônico + editor manual de camadas.
        self._apply_canonical_btn.clicked.connect(self._on_apply_canonical)
        self._edit_layers_btn.clicked.connect(self._on_edit_layers)
        self._refresh_manual_info()
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
        self._geo_mode.addItems(["stochastic", "fixed", "manual"])  # +manual (Fatia 6b)
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

        # ── Fatia 6b — filtro de Hankel + perfil canônico + editor manual ────
        from geosteering_ai.simulation.filters.loader import FilterLoader
        from geosteering_ai.simulation.validation.canonical_models import (
            get_all_canonical_models,
        )

        self._hankel_combo = QtWidgets.QComboBox()
        self._hankel_combo.addItems(list(FilterLoader().available()))
        self._hankel_combo.setCurrentText(vm.hankel_filter)
        self._hankel_combo.setToolTip(
            "Filtro de Hankel (transformada quasi-estática). Paridade <1e-12 preservada."
        )
        self._canonical_combo = QtWidgets.QComboBox()
        self._canonical_combo.addItem("—", userData=None)
        for cm in get_all_canonical_models():
            self._canonical_combo.addItem(cm.title, userData=cm.name)
        self._apply_canonical_btn = QtWidgets.QPushButton("Aplicar perfil")
        self._auto_geo_check = QtWidgets.QCheckBox("Auto-geometria (tj/h1)")
        self._auto_geo_check.setChecked(True)
        self._edit_layers_btn = QtWidgets.QPushButton("Editar camadas…")
        self._edit_layers_btn.setToolTip(
            "Editor manual de camadas (ρₕ/ρᵥ/espessura) → modo 'manual'."
        )
        self._manual_info = QtWidgets.QLabel("manual: —")
        self._manual_info.setProperty("role", "hint")

        # layout em form
        gform = QtWidgets.QFormLayout()
        gform.addRow("Modo:", self._geo_mode)
        gform.addRow("Filtro Hankel:", self._hankel_combo)
        canon_row = QtWidgets.QHBoxLayout()
        canon_row.addWidget(self._canonical_combo, stretch=1)
        canon_row.addWidget(self._apply_canonical_btn)
        gform.addRow("Perfil canônico:", canon_row)
        gform.addRow("", self._auto_geo_check)
        manual_row = QtWidgets.QHBoxLayout()
        manual_row.addWidget(self._edit_layers_btn)
        manual_row.addWidget(self._manual_info, stretch=1)
        gform.addRow("Geologia manual:", manual_row)
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

    # ── Paralelismo (Lote 1) — workers + threads + CPU + aviso ───────────────
    def _build_parallelism_group(self, vm: SimulationViewModel) -> Any:
        """Grupo "Paralelismo": workers + threads (efeito real) + info de CPU + aviso.

        Defaults vêm do VM (``recommend_default_parallelism``); a linha de CPU usa
        ``detect_cpu_topology`` (import LAZY+guardado, mantém a View leve). O aviso de
        oversubscrição (``W×T ≤ cores físicos``) atualiza em ``valueChanged``.
        """
        self._n_workers = QtWidgets.QSpinBox()
        self._n_workers.setRange(1, 256)
        self._n_workers.setValue(vm.n_workers)
        self._n_workers.setToolTip(
            "Nº de workers sandbox (processos). ESTADO/UI — o ProcessPoolExecutor "
            "real é a Fatia 5; hoje o numba roda in-process com N threads."
        )
        self._threads = QtWidgets.QSpinBox()
        self._threads.setRange(1, 256)
        self._threads.setValue(vm.threads_per_worker)
        self._threads.setToolTip(
            "Nº de threads por worker. EFEITO REAL: numba.set_num_threads(...) antes "
            "do simulate_batch (resultados bit-idênticos; clampado a NUMBA_NUM_THREADS)."
        )
        # Topologia da CPU (best-effort) — espelha a linha do monólito.
        try:
            from geosteering_ai.simulation._workers import detect_cpu_topology

            _logical, _physical, _ht = detect_cpu_topology()
            self._physical_cores = int(_physical)
            ht = " (HT/SMT ativo)" if _ht else ""
            cpu_txt = (
                f"CPU: {_physical} cores físicos · {_logical} threads lógicas{ht}"
            )
        except Exception:  # noqa: BLE001 — detecção best-effort
            self._physical_cores = 0
            cpu_txt = "CPU: topologia indisponível"
        self._cpu_info = QtWidgets.QLabel(cpu_txt)
        self._cpu_info.setProperty("role", "hint")
        self._parallel_warn = QtWidgets.QLabel("")
        self._parallel_warn.setWordWrap(True)
        self._n_workers.valueChanged.connect(self._refresh_parallel_warn)
        self._threads.valueChanged.connect(self._refresh_parallel_warn)

        pform = QtWidgets.QFormLayout()
        pform.addRow("Nº de workers sandbox:", self._n_workers)
        pform.addRow("Nº de threads por worker:", self._threads)
        pform.addRow(self._cpu_info)
        pform.addRow(self._parallel_warn)
        box = _group(
            "Paralelismo",
            "Defina o paralelismo por workers sandbox (threads têm efeito real no "
            "numba; o pool de processos é a Fatia 5).",
            pform,
        )
        self._refresh_parallel_warn()
        return box

    # ── Saída (Lote 1) — diretório + artefatos Fortran-compat ────────────────
    def _build_output_group(self, vm: SimulationViewModel) -> Any:
        """Grupo "Saída": diretório + "Procurar…" + checkbox de artefatos .dat/.out."""
        self._output_dir = QtWidgets.QLineEdit(vm.output_dir)
        self._output_dir.setPlaceholderText("Diretório de saída (vazio = não grava)…")
        self._browse_out = QtWidgets.QPushButton("Procurar…")
        self._browse_out.setProperty("role", "ghost")
        self._browse_out.clicked.connect(self._on_browse_output)
        self._save_artifacts = QtWidgets.QCheckBox(
            "Salvar artefatos Fortran-compat (.dat binário 22-col + .out ASCII)"
        )
        self._save_artifacts.setChecked(vm.save_fortran_artifacts)
        self._save_artifacts.setToolTip(
            "Após a simulação, grava o tensor H em .dat (22-col) + metadados .out "
            "(idênticos ao tatu.x). col2/col3 = ρₕ/ρᵥ no ponto de observação."
        )
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(self._output_dir, stretch=1)
        dir_row.addWidget(self._browse_out)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(dir_row)
        vbox.addWidget(self._save_artifacts)
        return _group(
            "Saída",
            "Diretório e artefatos Fortran-compatíveis (.dat/.out) gravados ao fim "
            "da simulação.",
            vbox,
        )

    def _refresh_counts(self, *_: Any) -> None:
        """Atualiza os contadores nf/ntheta/nº pares T-R (derivados dos CSV)."""
        for widget, label in (
            (self._freqs, self._nf_label),
            (self._dips, self._ntheta_label),
            (self._trs, self._ntr_label),
        ):
            try:
                label.setText(str(len(_parse_csv_floats(widget.text()))))
            except ValueError:
                label.setText("—")

    def _refresh_parallel_warn(self, *_: Any) -> None:
        """Aviso de oversubscrição: ``W×T ≤ cores físicos`` (senão ⚠, igual ao monólito)."""
        w = self._n_workers.value()
        t = self._threads.value()
        total = w * t
        phys = self._physical_cores
        if phys and total > phys:
            self._parallel_warn.setText(
                f"⚠ Oversubscrição: {w}×{t} = {total} threads em {phys} cores "
                f"físicos. Ideal: workers × threads ≤ {phys}."
            )
            self._parallel_warn.setStyleSheet("color: #f59e0b;")  # âmbar
        else:
            self._parallel_warn.setText("")
            self._parallel_warn.setStyleSheet("")

    def _on_browse_output(self) -> None:
        """Abre um seletor de diretório → preenche o campo de saída."""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Diretório de saída", self._output_dir.text() or ""
        )
        if path:
            self._output_dir.setText(path)

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
        self._vm.hankel_filter = self._hankel_combo.currentText()  # Fatia 6b
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
        # ── Paralelismo (Lote 1) + Saída ─────────────────────────────────────
        self._vm.n_workers = self._n_workers.value()
        self._vm.threads_per_worker = self._threads.value()
        self._vm.output_dir = self._output_dir.text().strip()
        self._vm.save_fortran_artifacts = self._save_artifacts.isChecked()
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
        """Habilita campos por modo: estocástico (samplers) × manual (editor)."""
        stochastic = text == "stochastic"
        manual = text == "manual"
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
            w.setEnabled(stochastic)
        # Editor manual / perfil canônico só fazem sentido no modo "manual".
        self._edit_layers_btn.setEnabled(manual)
        self._apply_canonical_btn.setEnabled(manual)
        self._canonical_combo.setEnabled(manual)
        self._auto_geo_check.setEnabled(manual)

    def _on_apply_canonical(self) -> None:
        """Aplica o perfil canônico selecionado → geologia manual (+ auto-geometria)."""
        name = self._canonical_combo.currentData()
        if not name:
            self._status.setText("selecione um perfil canônico")
            return
        self._vm.apply_canonical_profile(
            str(name), auto_geometry=self._auto_geo_check.isChecked()
        )
        self._geo_mode.setCurrentText("manual")  # apply_canonical já setou no VM
        self._sync_inputs_from_vm()  # reflete tj/h1 (auto-geo) + manual info

    def _on_edit_layers(self) -> None:
        """Abre o editor manual de camadas; ao aceitar, fixa a geologia manual no VM."""
        from apps.sim_manager.perspectives.simulation.layers_dialog import LayersDialog

        dialog = LayersDialog(initial=self._vm.manual_layers, parent=self)
        if dialog.exec():
            model = dialog.get_model()
            errors = model.validate()
            if errors:
                self._status.setText(f"camadas inválidas: {errors[0]}")
                return
            self._vm.manual_layers = model
            self._vm.geology_mode = "manual"
            self._geo_mode.setCurrentText("manual")
            self._refresh_manual_info()

    def _refresh_manual_info(self) -> None:
        """Atualiza o rótulo-resumo da geologia manual atual."""
        ml = self._vm.manual_layers
        self._manual_info.setText(
            "manual: —" if ml is None else f"manual: {ml.n_layers} camadas"
        )

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
        # Pause/cancel cooperativos só atuam no backend numba (in-thread); no
        # jax/auto (subprocesso) seriam no-op (v1) → manter desabilitados (UX honesta).
        supports_ctrl = running and self._vm.backend == "numba"
        # Botões por estado (1 simulação por vez; o guard real vive no VM).
        self._run_btn.setEnabled(not running)
        self._pause_btn.setEnabled(supports_ctrl)
        self._cancel_btn.setEnabled(supports_ctrl)
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
        self._vm.hankel_filter = self._hankel_combo.currentText()  # Fatia 6b
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
        # ── Paralelismo (Lote 1) + Saída ─────────────────────────────────────
        self._vm.n_workers = self._n_workers.value()
        self._vm.threads_per_worker = self._threads.value()
        self._vm.output_dir = self._output_dir.text().strip()
        self._vm.save_fortran_artifacts = self._save_artifacts.isChecked()

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
        self._hankel_combo.setCurrentText(self._vm.hankel_filter)  # Fatia 6b
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
        # ── Paralelismo (Lote 1) + Saída ─────────────────────────────────────
        self._n_workers.setValue(self._vm.n_workers)
        self._threads.setValue(self._vm.threads_per_worker)
        self._output_dir.setText(self._vm.output_dir)
        self._save_artifacts.setChecked(self._vm.save_fortran_artifacts)
        self._refresh_n_pos()
        self._refresh_counts()
