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
# ║  Framework   : Qt6 via gui.qt_compat                                       ║
# ║  Dependências: gui.qt_compat, gui.services, .viewmodel                      ║
# ║  Padrão      : View (MVVM) — sem lógica; faz binding aos VMSignals do VM   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Inputs multi-config (freqs/dips/TRs via CSV) + geometria (h1/tj/p_med/  ║
# ║    nº modelos) + grupo de GEOLOGIA estocástica (gerador, ranges ρₕ/λ,       ║
# ║    distribuição, n_layers, espessura, seed) + label de ``n_pos`` + Run +    ║
# ║    status + sessão. Ao Run: copia os inputs para o ViewModel e chama        ║
# ║    vm.run(). Liga-se a ``vm.changed`` (status/botão). A PLOTAGEM vive na     ║
# ║    perspectiva Resultados (PR-2 #1) — esta aba NÃO tem galeria.            ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimulatorView                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimulatorView`` — View Qt (inputs + geologia + Run + sessão; sem galeria), VM."""

from __future__ import annotations

from typing import Any, Tuple

from apps.sim_manager.perspectives.simulation.viewmodel import (
    _N_GEOMETRIES_MAX,
    SimulationViewModel,
)
from geosteering_ai.gui.persistence.session import SessionDocument
from geosteering_ai.gui.qt_compat import Qt, QtWidgets
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
    """View da Simulação: inputs + Run + status (SEM galeria), ligada a um ViewModel.

    Args:
        vm: o :class:`SimulationViewModel` (a View se liga aos seus VMSignals).
        parent: widget pai opcional.

    Note:
        A View não valida nem simula — só coleta inputs, chama ``vm.run()`` e
        reflete o estado que o VM emite (``changed``). A plotagem do ensemble vive
        na perspectiva Resultados (PR-2 #1), ligada ao mesmo ``vm.results``.
    """

    def __init__(
        self,
        vm: SimulationViewModel,
        parent: Any = None,
        *,
        jax_warmup_service: Any = None,
    ) -> None:
        super().__init__(parent)
        self._vm = vm
        # Service de warmup JAX COMPARTILHADO (de ctx.extras; None no CI/sem-jax/Studio).
        # Usado p/ o warmup config-aware ao selecionar jax/auto (:meth:`_maybe_warmup_config`).
        self._jax_warmup_service = jax_warmup_service

        # ── Inputs de listas (CSV multi-config) ──────────────────────────────
        self._freqs = QtWidgets.QLineEdit(self._fmt_csv(vm.frequencies))
        self._freqs.setPlaceholderText("ex.: 20000, 40000")
        self._freqs.setToolTip(
            "Frequências (Hz), separadas por vírgula. Range [10, 2e6]."
        )
        self._dips = QtWidgets.QLineEdit(self._fmt_csv(vm.dips))
        self._dips.setPlaceholderText("ex.: 0, 30")
        self._dips.setToolTip(
            "Ângulos de mergulho (°), separados por vírgula. Range [0, 90].\n\n"
            "NOTA: o nº de pontos de medição (n_pos) e a grade TVD são derivados SÓ do "
            "1º dip (n_pos=ceil(tj/(p_med·cos(dip0)))) e COMPARTILHADOS por todos os "
            "ângulos (grade única; o dip muda a resposta EM, não a amostragem). Logo a "
            "ORDEM importa: '0,15' usa a grade do 0° e '15,0' usa a do 15° — sem perda "
            "de precisão (cada ponto é exato), só densidade de amostragem diferente."
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

        # ── Auto-geometria canônica (Lote 2) — h1/tj derivados do perfil ─────
        # Dois checkboxes (espelham o monólito): ao aplicar um perfil canônico,
        # derivam tj/h1 das funções canônicas (paridade). Ver _on_apply_canonical.
        self._tj_auto = QtWidgets.QCheckBox("tj automático")
        self._tj_auto.setChecked(vm.tj_auto)
        self._tj_auto.setToolTip(
            "Ao aplicar um perfil canônico, deriva tj da janela de referência "
            "global: max(Σesp do batch, Σesp atual) + 20 m (paridade c/ o monólito)."
        )
        self._h1_auto = QtWidgets.QCheckBox("h1 automático")
        self._h1_auto.setChecked(vm.h1_auto)
        self._h1_auto.setToolTip(
            "Ao aplicar um perfil canônico, deriva h1 da centralização simétrica "
            "h1 = (tj − Σesp)/2 (margem simétrica do perfil — paridade c/ o monólito)."
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
            "· auto (decide GPU/CPU por tamanho/geometria). jax/auto rodam em subprocesso.\n\n"
            "DESEMPENHO: JAX é rápido com 'n camadas FIXO' (1 forma XLA — a 1ª execução "
            "compila e as seguintes reusam o cache). Para ENSEMBLES ragged (n camadas "
            "variável, ex.: o default de 2000 modelos), a 1ª execução JAX compila MUITAS "
            "formas (minutos, uma vez); o Numba é mais rápido p/ ragged (sem compilação)."
        )
        self._sim_backend.currentTextChanged.connect(self._on_sim_backend_changed)

        # ── Contadores derivados (nf/ntheta/nº pares T-R) ────────────────────
        # Read-only: derivados dos CSV de freqs/dips/TRs (atualizados em textChanged).
        self._nf_label = QtWidgets.QLabel("—")
        self._ntheta_label = QtWidgets.QLabel("—")
        self._ntr_label = QtWidgets.QLabel("—")

        # ── Categoria 3: "Geração de Modelos" (+Hankel) — cria também os widgets
        #    do perfil canônico, reaproveitados pela Categoria 1 logo abaixo.
        self._geology_box = self._build_geology_group(vm)
        # ── Categoria 1: "Perfil Pré-configurado" (perfil canônico + Aplicar) ──
        self._profile_box = self._build_profile_group()
        # ── Paralelismo (Lote 1/2) + Saída ───────────────────────────────────
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

        # PR-2 (#1): a galeria de plot foi MOVIDA p/ a perspectiva "Resultados"
        # (a aba Simulação não tem mais plotagem). O ResultsViewModel vive em
        # ``vm.results`` e é publicado em ``ctx.extras["results_vm"]`` pela
        # SimulationPerspective; a ResultsPerspective liga uma 2ª View ao MESMO VM.

        # ── Layout ────────────────────────────────────────────────────────────
        # Geometria + aquisição (com contadores derivados no topo, espelhando o
        # monólito: "Nº de frequências (nf)", "Nº de ângulos (ntheta)", "Nº de
        # pares T-R" como rótulos read-only acima dos CSV que os definem).
        # Categoria 2 (Lote 2): "Configurações da Ferramenta" (renomeada de
        # "Geometria da ferramenta") — geometria + aquisição + auto-geometria.
        # n_models → "Geração de Modelos"; backend → "Paralelismo" (gate Numba/Fortran).
        form = QtWidgets.QFormLayout()
        form.addRow("Nº de frequências (nf):", self._nf_label)
        form.addRow("Nº de ângulos (ntheta):", self._ntheta_label)
        form.addRow("Nº de pares T-R:", self._ntr_label)
        form.addRow("Frequências (Hz):", self._freqs)
        form.addRow("Ângulos de dip (graus):", self._dips)
        form.addRow("Espaçamentos T-R (m):", self._trs)
        form.addRow("Altura h1:", self._h1)
        form.addRow("", self._h1_auto)
        form.addRow("Janela de investigação tj:", self._tj)
        form.addRow("", self._tj_auto)
        form.addRow("Passo entre medidas p_med:", self._p_med)
        form.addRow("Posições derivadas:", self._n_pos_label)
        geometry_box = _group(
            "Configurações da Ferramenta",
            "Geometria da ferramenta LWD e aquisição (frequências, dips, "
            "espaçamentos T-R, janela de investigação). Passe o mouse sobre cada "
            "campo para uma explicação detalhada.",
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
        # ── Conteúdo vertical (config + execução + galeria) ───────────────────
        # Montado num widget interno e envolvido por uma QScrollArea VERTICAL: o
        # conteúdo cresceu (3 categorias + paralelismo Numba/Fortran + saída +
        # galeria) e ultrapassa a altura da janela — sem scroll, os widgets de
        # baixo (Saída, Iniciar, galeria) ficavam INACESSÍVEIS. Fix do Lote 2.
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(_heading_title("Parâmetros da simulação"))
        content_layout.addWidget(
            _hint(
                "Configure a geometria da ferramenta LWD, o filtro de Hankel e os "
                "parâmetros de geração estocástica de perfis TIV. Passe o mouse sobre "
                "cada campo para uma explicação detalhada."
            )
        )
        # ── 3 categorias (Lote 2): Perfil Pré-configurado · Configurações da
        #    Ferramenta · Geração de Modelos (+Hankel) — espelham o monólito.
        content_layout.addWidget(self._profile_box)
        content_layout.addWidget(geometry_box)
        content_layout.addWidget(self._geology_box)
        content_layout.addWidget(self._parallel_box)
        content_layout.addWidget(self._output_box)
        content_layout.addLayout(exec_row)
        content_layout.addLayout(session_row)
        # PR-2 (#1): galeria removida desta aba (vive em "Resultados"). O stretch
        # final empurra o conteúdo p/ o topo dentro da QScrollArea.
        content_layout.addStretch(1)

        # QScrollArea: h-scroll OFF (widgetResizable acompanha a largura do
        # viewport — sem vazamento horizontal); v-scroll AS-NEEDED.
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("SimulatorScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

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
        # Geologia (Lote 2): 2 radios Aleatória/Manual habilitam os campos do modo
        # ativo (samplers no modo Aleatória; editor de camadas no modo Manual).
        self._radio_random.toggled.connect(self._on_geology_mode_changed)
        self._radio_manual.toggled.connect(self._on_geology_mode_changed)
        # Fatia 6b — perfil canônico + editor manual de camadas.
        self._apply_canonical_btn.clicked.connect(self._on_apply_canonical)
        self._edit_layers_btn.clicked.connect(self._on_edit_layers)
        self._refresh_manual_info()
        self._on_geology_mode_changed()

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
        # Modo de geração (Lote 2): 2 radios (Aleatória/Manual) — substitui o combo
        # de 3 modos. O "fixed" legado deixa de ser exposto na UI (o VM ainda o
        # aceita p/ retrocompat/testes); "n_layers fixo" cobre o caso fixo dentro
        # do modo Aleatória. Resolve o bug "modo fixed travava tudo" (Task 3).
        self._radio_random = QtWidgets.QRadioButton("Aleatória (QMC / PRNG)")
        self._radio_manual = QtWidgets.QRadioButton("Manual (tabela de camadas)")
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.addButton(self._radio_random)
        self._mode_group.addButton(self._radio_manual)
        # Estado inicial do VM: manual ⟺ "manual"; senão Aleatória (inclui "fixed").
        if vm.geology_mode == "manual":
            self._radio_manual.setChecked(True)
        else:
            self._radio_random.setChecked(True)
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
        self._geo_lambda_min.setDecimals(4)  # 4 casas = paridade monólito (λ=√2→1.4142)
        self._geo_lambda_min.setSingleStep(0.05)  # passo do monólito
        self._geo_lambda_min.setValue(vm.lambda_min)
        self._geo_lambda_max = QtWidgets.QDoubleSpinBox()
        self._geo_lambda_max.setRange(1.0, 5.0)
        self._geo_lambda_max.setDecimals(4)  # 4 casas = paridade monólito (λ=√2→1.4142)
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

        # ── Tooltips físicos/geofísicos (#6 — fidelidade com o SM monólito) ──
        # Contexto e faixas típicas dos parâmetros geológicos/petrofísicos.
        self._geo_generator.setToolTip(
            "Gerador quasi-aleatório (QMC). 'sobol' (default): sequência de baixa "
            "discrepância — cobertura uniforme do espaço de parâmetros, ideal p/ ML. "
            "'halton': QMC base-prima. Demais: PRNG clássico."
        )
        self._geo_distr.setToolTip(
            "Distribuição de amostragem de ρₕ. 'loguni' (default): log-uniforme — cobre "
            "várias ordens de magnitude (≈0.3–5000 Ω·m). 'uniform': linear (concentra nos altos)."
        )
        self._geo_rho_min.setToolTip(
            "Resistividade horizontal MÍNIMA (Ω·m). Típicos: ~0.3 (água salgada), ~1 "
            "(água doce), 50–200 (folhelho), 1000+ (hidrocarboneto/carbonato resistivo)."
        )
        self._geo_rho_max.setToolTip(
            "Resistividade horizontal MÁXIMA (Ω·m). Máximo usual ~5000 (tight gas / "
            "carbonato muito resistivo)."
        )
        self._geo_aniso.setToolTip(
            "Anisotropia TIV: ρᵥ = λ²·ρₕ (transversal isotrópico vertical, eixo vertical). "
            "Desmarque p/ meio isotrópico (λ=1 ⇒ ρᵥ=ρₕ)."
        )
        self._geo_lambda_min.setToolTip(
            "Coeficiente de anisotropia MÍNIMO (λ = √(ρᵥ/ρₕ) ≥ 1). Em meios laminados "
            "ρᵥ ≥ ρₕ (λ ≥ 1)."
        )
        self._geo_lambda_max.setToolTip(
            "Coeficiente de anisotropia MÁXIMO. Típicos: ~1.4 (fraca), ~1.7 (moderada), "
            "~2.2 (forte)."
        )
        self._geo_min_thick.setToolTip(
            "Espessura MÍNIMA por camada (m). Piso do stick-breaking — evita camadas "
            "finas demais (≪ resolução do filtro de Hankel / comprimento de onda)."
        )
        self._geo_seed_random.setToolTip(
            "Semente aleatória (recomendado): cada execução gera um ensemble diferente. "
            "Desmarque p/ semente FIXA (reprodutibilidade bit-a-bit)."
        )
        self._geo_seed.setToolTip(
            "Semente fixa do RNG. Mesma semente + mesmos parâmetros ⇒ ensemble idêntico. "
            "Ativa apenas com 'Semente aleatória' desmarcada."
        )

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
        self._edit_layers_btn = QtWidgets.QPushButton("Editar camadas…")
        self._edit_layers_btn.setToolTip(
            "Editor manual de camadas (ρₕ/ρᵥ/espessura) → modo 'manual'."
        )
        self._manual_info = QtWidgets.QLabel("manual: —")
        self._manual_info.setProperty("role", "hint")

        # ── Categoria 3 (Lote 2): "Geração de Modelos" (+Filtros de Hankel) ──
        # Ordem espelha as imagens de referência: radios → editor manual →
        # quantidade → gerador → anisotropia → ρₕ → distribuição → espessura →
        # n_layers → semente → filtro de Hankel. O combo de perfil canônico NÃO
        # entra aqui (vai p/ a Categoria 1 "Perfil Pré-configurado").
        gform = QtWidgets.QFormLayout()
        gform.addRow(self._radio_random)
        gform.addRow(self._radio_manual)
        manual_row = QtWidgets.QHBoxLayout()
        manual_row.addWidget(self._edit_layers_btn)
        manual_row.addWidget(self._manual_info, stretch=1)
        gform.addRow(manual_row)
        gform.addRow("Quantidade de modelos:", self._n_models)
        gform.addRow("Gerador aleatório:", self._geo_generator)
        gform.addRow("", self._geo_aniso)
        gform.addRow("λ mínimo (TIV):", self._geo_lambda_min)
        gform.addRow("λ máximo (TIV):", self._geo_lambda_max)
        gform.addRow("ρₕ mínimo:", self._geo_rho_min)
        gform.addRow("ρₕ máximo:", self._geo_rho_max)
        gform.addRow("Distribuição de ρₕ:", self._geo_distr)
        gform.addRow("Espessura mínima:", self._geo_min_thick)
        gform.addRow("Nº camadas mínimo:", self._geo_nl_min)
        gform.addRow("Nº camadas máximo:", self._geo_nl_max)
        gform.addRow("", self._geo_nlf_check)
        gform.addRow("Nº camadas fixo:", self._geo_nlf)
        gform.addRow("", self._geo_seed_random)
        gform.addRow("Semente fixa:", self._geo_seed)
        gform.addRow("Filtro Hankel:", self._hankel_combo)
        return _group(
            "Geração de Modelos",
            "Defina como os perfis geológicos são gerados (Aleatória QMC/PRNG ou "
            "Manual via tabela de camadas) e o filtro de Hankel da transformada.",
            gform,
        )

    # ── Categoria 1 (Lote 2) — "Perfil Pré-configurado" ─────────────────────
    def _build_profile_group(self) -> Any:
        """Grupo top-level "Perfil Pré-configurado": perfil canônico + Aplicar.

        Reaproveita ``_canonical_combo``/``_apply_canonical_btn`` (criados em
        :meth:`_build_geology_group`) — aqui só os promove a uma seção própria
        (espelha o monólito, onde o perfil canônico é a 1ª seção). Aplicar um
        perfil comuta para o modo Manual e (com h1/tj automático) deriva a
        geometria canônica (ver :meth:`_on_apply_canonical`).
        """
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._canonical_combo, stretch=1)
        row.addWidget(self._apply_canonical_btn)
        return _group(
            "Perfil Pré-configurado",
            "Selecione um perfil canônico da literatura (Oklahoma, Devine, …) e "
            "clique em Aplicar perfil para preencher as camadas.",
            row,
        )

    # ── Paralelismo (Lote 1/2) — backend + workers/threads Numba & Fortran ───
    def _build_parallelism_group(self, vm: SimulationViewModel) -> Any:
        """Grupo "Paralelismo": workers + threads (efeito real) + info de CPU + aviso.

        Defaults vêm do VM (``recommend_default_parallelism``); a linha de CPU usa
        ``detect_cpu_topology`` (import LAZY+guardado, mantém a View leve). O aviso de
        oversubscrição (``W×T ≤ cores físicos``) atualiza em ``valueChanged``.
        """
        # ── Diversidade de geometria (PR-1) — batchabilidade no JAX GPU ──────
        self._geom_diversity = QtWidgets.QComboBox()
        self._geom_diversity.addItems(["auto", "templates", "per_model"])
        self._geom_diversity.setCurrentText(vm.geometry_diversity)
        self._geom_diversity.setToolTip(
            "Diversidade de geometria (espessuras) p/ o JAX GPU. No modo estocástico, "
            "espessuras únicas por modelo impedem o batch na GPU (cada modelo vira 1 "
            "grupo → cai p/ Numba). 'templates' colapsa a K geometrias por nº de "
            "camadas (round-robin) → o JAX-grouped satura a GPU; ρₕ/ρᵥ/λ continuam "
            "variando por modelo. 'auto' = templates p/ jax/auto, por-modelo p/ numba "
            "(Numba é indiferente). 'per_model' preserva a diversidade total (JAX lento).\n\n"
            "DICA de ocupação: a GPU bateleia por grupo de nº de camadas e exige ≥32 "
            "modelos/grupo. Para batelar TODO o ensemble: use N grande (ex.: 1000 "
            "modelos ⇒ ~125/grupo) OU marque 'n_layers fixo' (1 grupo único). Com poucos "
            "modelos e faixa de camadas larga, grupos pequenos ainda rodam em Numba.\n\n"
            "COLD-START (JAX GPU): a 1ª execução de cada geometria/tamanho COMPILA os "
            "kernels XLA (dezenas de s por geometria distinta, custo ÚNICO); execuções "
            "seguintes reusam o cache de disco (~30 s, mais rápido que o Numba). Use "
            "'Geometrias (K)' p/ controlar o trade-off diversidade × cold-start."
        )
        # ── Geometrias distintas (K) — trade-off diversidade × cold-start XLA ─────
        # Expõe SimRequest.n_geometries (já existia no VM, faltava na UI): cada K vira
        # 1 programa XLA compilado na 1ª execução JAX. K=1 ⇒ 1 compilação ⇒ cold-start
        # ≈ Numba (todas as geometrias iguais; ρₕ/ρᵥ/λ ainda variam por modelo). 0=auto
        # (teto ≤4 por nº de camadas). Não afeta o Numba nem execuções já em cache.
        self._geom_k = QtWidgets.QSpinBox()
        self._geom_k.setRange(0, _N_GEOMETRIES_MAX)  # 0=auto; teto == clamp do VM
        self._geom_k.setValue(vm.n_geometries or 0)
        self._geom_k.setSpecialValueText("auto")  # 0 → "auto" (teto ≤4)
        self._geom_k.setToolTip(
            "Nº de geometrias distintas (K) nos modos 'templates'/'auto'. 'auto' (0) usa "
            "o teto (≤4) por nº de camadas. K controla o trade-off DIVERSIDADE × "
            "COLD-START do JAX GPU: cada K vira 1 programa XLA compilado na 1ª execução "
            "(dezenas de s/compilação, cresce com o tamanho do grupo). K menor ⇒ menos "
            "compilações ⇒ cold-start mais curto (medido: K=1 ~108 s vs K=4 ~173 s p/ "
            "1000 modelos), mas todos os modelos compartilham a MESMA geometria (ρₕ/ρᵥ/λ "
            "ainda variam). K maior ⇒ mais diversidade geológica, cold-start mais longo. "
            "Sem efeito no Numba; execuções já em cache rodam ~30 s (batem o Numba) p/ "
            "qualquer K."
        )
        # ── Workers + threads (par ÚNICO) ────────────────────────────────────
        # Os únicos simuladores do SM MVVM são Numba JIT e JAX GPU (sem Fortran).
        self._n_workers = QtWidgets.QSpinBox()
        self._n_workers.setRange(1, 256)
        self._n_workers.setValue(vm.n_workers)
        self._n_workers.setToolTip(
            "Nº de workers (processos sandbox). ESTADO/UI — o ProcessPoolExecutor real "
            "é a Fatia 5; hoje o numba roda in-process com N threads."
        )
        self._threads = QtWidgets.QSpinBox()
        self._threads.setRange(1, 256)
        self._threads.setValue(vm.threads_per_worker)
        self._threads.setToolTip(
            "Nº de threads por worker. EFEITO REAL no Numba: numba.set_num_threads(...) "
            "antes do simulate_batch (bit-idêntico; clampado a NUMBA_NUM_THREADS). No JAX "
            "(subprocesso GPU) é no-op — a GPU paraleliza internamente."
        )
        # Topologia da CPU (best-effort) — espelha a linha do monólito.
        try:
            from geosteering_ai.simulation._workers import detect_cpu_topology

            _logical, _physical, _ht = detect_cpu_topology()
            self._physical_cores = int(_physical)
            ht = " (HT/SMT ativo)" if _ht else ""
            cpu_txt = f"CPU: {_physical} cores físicos · {_logical} threads lógicas{ht}"
        except Exception:  # noqa: BLE001 — detecção best-effort
            self._physical_cores = 0
            cpu_txt = "CPU: topologia indisponível"
        self._cpu_info = QtWidgets.QLabel(cpu_txt)
        self._cpu_info.setProperty("role", "hint")
        self._parallel_warn = QtWidgets.QLabel("")
        self._parallel_warn.setWordWrap(True)
        # Aviso de oversubscrição usa o par do backend ATIVO (Numba).
        self._n_workers.valueChanged.connect(self._refresh_parallel_warn)
        self._threads.valueChanged.connect(self._refresh_parallel_warn)

        pform = QtWidgets.QFormLayout()
        pform.addRow("Backend:", self._sim_backend)
        pform.addRow("Diversidade de geometria (JAX):", self._geom_diversity)
        pform.addRow("Geometrias (K):", self._geom_k)
        pform.addRow("Nº de workers:", self._n_workers)
        pform.addRow("Nº de threads:", self._threads)
        pform.addRow(self._cpu_info)
        pform.addRow(self._parallel_warn)
        box = _group(
            "Paralelismo",
            "Simuladores: Numba JIT (CPU, in-thread) e JAX GPU (subprocesso). Threads "
            "têm efeito real no Numba (numba.set_num_threads); workers (pool de "
            "processos) = Fatia 5. Nenhuma execução Fortran no SM.",
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
            "Salvar artefatos .dat/.out (22-col)"
        )
        self._save_artifacts.setChecked(vm.save_fortran_artifacts)
        self._save_artifacts.setToolTip(
            "Após a simulação, grava o tensor H em .dat (22-col binário) + metadados "
            ".out ASCII — formato do tatu.x, LIDO pelo Visualizador .dat. col2/col3 = "
            "ρₕ/ρᵥ no ponto de observação. (Formato de arquivo; não executa Fortran.)"
        )
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(self._output_dir, stretch=1)
        dir_row.addWidget(self._browse_out)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(dir_row)
        vbox.addWidget(self._save_artifacts)
        return _group(
            "Saída",
            "Diretório e artefatos .dat/.out (22-col, formato tatu.x) gravados ao fim "
            "da simulação — reabríveis no Visualizador .dat.",
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
    def _on_sim_backend_changed(self, text: str) -> None:
        """Aviso honesto de UX ao escolher jax/auto (Turn 8 — fix da regressão JAX).

        JAX é rápido p/ geometria com 'n camadas FIXO' (1 forma XLA). Para ENSEMBLES
        ragged (n camadas variável — ex.: o default de 2000 modelos), a 1ª execução JAX
        compila MUITAS formas (minutos, uma vez/máquina; o cache de disco acelera as
        seguintes) — o Numba é mais rápido. Apenas ORIENTA via status; NÃO bloqueia nem
        muda o backend (o VM lê o combo ao rodar). Não muta o VM (evita recursão de sync).
        """
        if text not in ("jax", "auto"):
            return
        if self._vm.n_layers_fixed is None:
            self._status.setText(
                "● JAX + ensemble ragged: 1ª execução compila (min, uma vez/máquina; "
                "cache acelera as seguintes). Numba é mais rápido p/ ragged."
            )
        else:
            self._status.setText(
                "● JAX (n camadas fixo): 1ª execução compila (~min); seguintes reusam o cache."
            )
        # Dispara o warmup config-aware em background (pref jax_auto_warmup, default ON) —
        # pré-compila as formas EXATAS da config p/ a 1ª sim ficar cache-hit (~12 s).
        self._maybe_warmup_config()

    def _maybe_warmup_config(self) -> None:
        """Aquece o JAX com as formas da config corrente, em background (on-select).

        Gateado por: presença do ``JaxWarmupService`` (None no CI/sem-jax), pref
        ``jax_auto_warmup`` (default ON), e guard ``is_busy`` (não empilha — debounce
        natural). Sincroniza o VM com os widgets (silencioso), monta a ``SimRequest``
        corrente e submete os specs shape-matching ao worker persistente. Best-effort:
        NUNCA quebra a UI nem alarma — uma falha de warmup é apenas otimização perdida.
        """
        svc = self._jax_warmup_service
        if svc is None or getattr(svc, "is_busy", lambda: False)():
            return
        try:
            from apps.sim_manager.perspectives.preferences.service import (
                PreferencesService,
            )

            if not bool(PreferencesService().load().get("jax_auto_warmup", True)):
                return
            if not self._sync_vm_from_widgets(show_errors=False):
                return
            from geosteering_ai.gui.services.jax_warmup_spec import build_warmup_specs

            specs = build_warmup_specs(self._vm.build_sim_request())
            if not specs:
                return  # numba / grupos < limiar GPU → nada a aquecer
            self._status.setText(
                "● Aquecendo JAX GPU em background — a 1ª simulação ficará rápida "
                "(~12 s). A UI segue livre; o Numba continua disponível."
            )
            svc.warmup_config(specs)
        except Exception:  # noqa: BLE001 — warmup é otimização; UI nunca quebra
            pass

    def _on_run_clicked(self) -> None:
        """Parseia os inputs, copia para o VM e dispara a simulação.

        CSV inválido (token não-numérico) → status de erro, SEM chamar o VM e
        SEM crash. A validação de ranges (errata) é responsabilidade do VM.
        """
        if self._sync_vm_from_widgets():
            self._vm.run()

    def _sync_vm_from_widgets(self, *, show_errors: bool = True) -> bool:
        """Parseia os inputs e copia p/ o VM (fonte única do estado corrente).

        Retorna ``False`` se algum CSV for inválido (com ``show_errors`` seta o status de
        erro; o warmup usa ``show_errors=False`` p/ falhar silenciosamente). Extraído de
        :meth:`_on_run_clicked` p/ reuso pelo gatilho de warmup config-aware
        (:meth:`_maybe_warmup_config`) — assim o warmup monta a MESMA ``SimRequest`` que a
        simulação real, garantindo o shape-matching (cache-hit).

        Mutar o VM aqui é SEGURO mesmo a partir do handler do combo: ``_on_vm_changed``
        só reage a ``_status``/``_progress`` (não aos campos de input) → sem recursão.
        """
        try:
            freqs = _parse_csv_floats(self._freqs.text())
            dips = _parse_csv_floats(self._dips.text())
            trs = _parse_csv_floats(self._trs.text())
        except ValueError as exc:
            if show_errors:
                self._status.setText(f"entrada inválida: {exc}")
            return False
        self._vm.frequencies = freqs
        self._vm.dips = dips
        self._vm.tr_spacings = trs
        self._vm.h1 = self._h1.value()
        self._vm.tj = self._tj.value()
        self._vm.p_med = self._p_med.value()
        self._vm.n_models = self._n_models.value()
        self._vm.backend = self._sim_backend.currentText()
        self._vm.geometry_diversity = self._geom_diversity.currentText()  # PR-1 (#4)
        self._vm.n_geometries = self._geom_k.value() or None  # 0 → auto (teto ≤4)
        self._vm.hankel_filter = self._hankel_combo.currentText()  # Fatia 6b
        # ── Geologia estocástica (Fatia 3) ───────────────────────────────────
        self._vm.geology_mode = (
            "manual" if self._radio_manual.isChecked() else "stochastic"
        )
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
        self._vm.h1_auto = self._h1_auto.isChecked()
        self._vm.tj_auto = self._tj_auto.isChecked()
        return True

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

    def _on_geology_mode_changed(self, *_: Any) -> None:
        """Habilita campos por modo (Lote 2): Aleatória (samplers) × Manual (editor).

        Resolve o bug do modo "fixed" travado (Task 3): agora só há 2 estados
        (Aleatória/Manual), ambos funcionais. O perfil canônico (Categoria 1
        "Perfil Pré-configurado") fica SEMPRE acessível — aplicá-lo comuta para Manual.
        """
        manual = self._radio_manual.isChecked()
        stochastic = not manual
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
        # Editor manual só faz sentido no modo Manual. O perfil canônico (combo +
        # Aplicar) e os checkboxes h1/tj-auto ficam SEMPRE habilitados (seções próprias).
        self._edit_layers_btn.setEnabled(manual)

    def _on_apply_canonical(self) -> None:
        """Aplica o perfil canônico selecionado → geologia manual (+ auto-geometria)."""
        name = self._canonical_combo.currentData()
        if not name:
            self._status.setText("selecione um perfil canônico")
            return
        # Espelha o monólito (simulation_manager.py:2014-2015): aplicar um perfil
        # FORÇA a auto-geometria ON (h1/tj derivados da Σesp), mesmo que os
        # checkboxes estivessem desligados — que é o default de produção (item 3).
        # _sync_inputs_from_vm (abaixo) re-marca os checkboxes a partir do VM.
        self._vm.h1_auto = True
        self._vm.tj_auto = True
        self._vm.apply_canonical_profile(str(name), auto_tj=True, auto_h1=True)
        self._radio_manual.setChecked(True)  # apply_canonical já setou no VM
        self._sync_inputs_from_vm()  # reflete tj/h1 (auto-geo) + n_layers
        self._refresh_manual_info()  # Task 4: rótulo "N camadas" do perfil aplicado

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
            self._radio_manual.setChecked(True)
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
        self._vm.geometry_diversity = self._geom_diversity.currentText()  # PR-1 (#4)
        self._vm.n_geometries = self._geom_k.value() or None  # 0 → auto (teto ≤4)
        self._vm.hankel_filter = self._hankel_combo.currentText()  # Fatia 6b
        self._vm.geology_mode = (
            "manual" if self._radio_manual.isChecked() else "stochastic"
        )
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
        self._vm.h1_auto = self._h1_auto.isChecked()
        self._vm.tj_auto = self._tj_auto.isChecked()

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
        self._geom_diversity.setCurrentText(self._vm.geometry_diversity)  # PR-1 (#4)
        self._geom_k.setValue(self._vm.n_geometries or 0)  # None → auto (0)
        self._hankel_combo.setCurrentText(self._vm.hankel_filter)  # Fatia 6b
        # Radio (Lote 2): manual ⟺ "manual"; senão Aleatória (inclui "fixed" legado).
        if self._vm.geology_mode == "manual":
            self._radio_manual.setChecked(True)
        else:
            self._radio_random.setChecked(True)
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
        self._h1_auto.setChecked(self._vm.h1_auto)
        self._tj_auto.setChecked(self._vm.tj_auto)
        self._refresh_n_pos()
        self._refresh_counts()
