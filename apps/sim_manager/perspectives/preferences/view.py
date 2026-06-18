# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/preferences/view.py                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : PreferencesPanel — View Qt da perspectiva Preferências     ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app (MVVM) — perspectiva Preferências (Fatia 6e)         ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — paridade c/ PreferencesPage do monólito          ║
# ║  Framework   : Qt6 via gui.qt_compat + gui.theme + gui.plot_backends        ║
# ║  Dependências: gui.qt_compat, gui.theme.manager, gui.plot_backends, .vm     ║
# ║  Padrão      : View (MVVM) — push-on-action; sem lógica de domínio          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Formulário de preferências do SM: tema · 4 paths (com "Procurar…") ·    ║
# ║    backend de plot default · limites do cache LRU (MB + nº de snapshots) +  ║
# ║    botões Salvar/Restaurar. Espelha o subconjunto MVVM da PreferencesPage   ║
# ║    do monólito. Modelo push-on-action (igual à SimulatorView): widgets só   ║
# ║    são copiados ao VM em "Salvar"; "Restaurar" repovoa os widgets do VM.    ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    PreferencesPanel                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``PreferencesPanel`` — View Qt das preferências do SM (Fatia 6e)."""

from __future__ import annotations

from typing import Any, Tuple

from apps.sim_manager.perspectives.preferences.viewmodel import PreferencesViewModel
from geosteering_ai.gui.qt_compat import Qt, QtWidgets

__all__ = ["PreferencesPanel"]

# Rótulo legível + tooltip de cada path (paridade com a PreferencesPage do monólito).
# (key, rótulo, modo de seleção: "dir" | "file", tooltip)
_PATH_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "output_dir",
        "Diretório de saída",
        "dir",
        "Diretório default para artefatos (.dat/.out) e experimentos salvos.",
    ),
    (
        "tatu_binary",
        "Binário tatu.x (Fortran)",
        "file",
        "Caminho do executável Fortran tatu.x (backend de referência; vazio = auto).",
    ),
    (
        "python_binary",
        "Interpretador Python",
        "file",
        "Python usado p/ os subprocessos JAX-GPU (vazio = sys.executable).",
    ),
    (
        "geosteering_ai",
        "Raiz do pacote geosteering_ai",
        "dir",
        "Raiz do repositório/pacote (vazio = autodetecção pelo import instalado).",
    ),
)


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


class PreferencesPanel(QtWidgets.QWidget):  # type: ignore[misc] # QtWidgets é Any → mypy
    """View das preferências: tema + paths + backend + cache LRU + Salvar/Restaurar.

    Args:
        vm: o :class:`PreferencesViewModel` (a View lê/escreve seu estado).
        parent: widget pai opcional.

    Note:
        Modelo push-on-action (igual à ``SimulatorView``): os widgets só são
        copiados ao VM em "Salvar" — assim NÃO há laço View↔VM nem necessidade de
        flag de guarda. "Restaurar padrões" repovoa os widgets a partir do VM.
        A aplicação AO VIVO de tema/backend/cache às demais perspectivas é uma
        fatia futura — aqui as preferências são apenas PERSISTIDAS (UX honesta).
    """

    def __init__(self, vm: PreferencesViewModel, parent: Any = None) -> None:
        super().__init__(parent)
        self._vm = vm

        # ── Aparência: tema ──────────────────────────────────────────────────
        # available_themes() é puro (sem QApplication) — hoje só "antigravity_dark"
        # (spec 0013); novos temas plugam aqui sem mudar a View.
        self._theme = QtWidgets.QComboBox()
        self._theme.addItems(self._available_themes())
        self._theme.setToolTip(
            "Tema visual do SM. Hoje só 'antigravity_dark' (spec 0013); novos temas "
            "serão adicionados em fatias futuras."
        )

        # ── Backend de plot default ──────────────────────────────────────────
        self._plot_backend = QtWidgets.QComboBox()
        self._plot_backend.addItems(self._available_backends())
        self._plot_backend.setToolTip(
            "Backend de renderização default da galeria: matplotlib (sempre "
            "disponível) · pyqtgraph (rápido) · plotly (interativo) · vispy (GPU)."
        )

        # ── Cache LRU de plots (limites) ─────────────────────────────────────
        self._cache_mb = QtWidgets.QSpinBox()
        self._cache_mb.setRange(32, 32_768)  # mín. 32 MB (guardrail do VM)
        self._cache_mb.setSingleStep(64)
        self._cache_mb.setSuffix(" MB")
        self._cache_mb.setToolTip(
            "Teto de memória do cache LRU de snapshots de plot. Ao exceder, os "
            "snapshots menos recentes são despejados."
        )
        self._cache_snaps = QtWidgets.QSpinBox()
        self._cache_snaps.setRange(1, 512)  # mín. 1 (guardrail do VM)
        self._cache_snaps.setToolTip(
            "Nº máximo de snapshots mantidos no cache LRU (maxlen), independente do "
            "teto de MB. O que estourar primeiro (nº ou MB) rege o despejo."
        )

        # ── Desempenho — boot warmup do JAX GPU (opt-in) ─────────────────────
        self._jax_boot_warmup = QtWidgets.QCheckBox(
            "Aquecer o worker JAX GPU no boot (tira o cold-start da 1ª simulação)"
        )
        self._jax_boot_warmup.setToolTip(
            "OPT-IN (desligado por padrão). Ao abrir o SM, aquece em background o worker "
            "JAX GPU persistente (init CUDA + compila a forma canônica), de modo que a 1ª "
            "simulação jax/auto já reuse o worker quente (~12 s) em vez de pagar o "
            "cold-start. Gasta GPU/VRAM no boot — deixe DESLIGADO se usa só o Numba. Sem "
            "efeito se o JAX não estiver instalado. Ver docs/reference/"
            "sm_jax_persistent_worker.md."
        )

        # ── Paths (4 linhas: QLineEdit + "Procurar…") ────────────────────────
        self._path_edits: dict[str, Any] = {}
        paths_form = QtWidgets.QFormLayout()
        for key, label, mode, tip in _PATH_SPECS:
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("(vazio = autodetecção)")
            edit.setToolTip(tip)
            browse = QtWidgets.QPushButton("Procurar…")
            browse.setProperty("role", "ghost")
            # Liga o botão ao seletor certo (dir/arquivo) p/ esta chave específica.
            browse.clicked.connect(
                lambda _checked=False, k=key, m=mode, lb=label: self._on_browse(
                    k, m, lb
                )
            )
            row = QtWidgets.QHBoxLayout()
            row.addWidget(edit, stretch=1)
            row.addWidget(browse)
            paths_form.addRow(f"{label}:", row)
            self._path_edits[key] = edit

        # ── Botões + status ──────────────────────────────────────────────────
        self._save_btn = QtWidgets.QPushButton("Salvar preferências")
        self._save_btn.setProperty("role", "primary")
        self._restore_btn = QtWidgets.QPushButton("Restaurar padrões")
        self._restore_btn.setProperty("role", "ghost")
        self._status = QtWidgets.QLabel("")
        self._status.setProperty("role", "hint")

        # ── Layout (cards numa QScrollArea vertical) ─────────────────────────
        appearance_form = QtWidgets.QFormLayout()
        appearance_form.addRow("Tema:", self._theme)
        appearance_form.addRow("Backend de plot:", self._plot_backend)

        cache_form = QtWidgets.QFormLayout()
        cache_form.addRow("Limite de memória:", self._cache_mb)
        cache_form.addRow("Máx. de snapshots:", self._cache_snaps)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._restore_btn)
        btn_row.addWidget(self._status, stretch=1)
        btn_row.addStretch(0)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(_heading_title("Preferências"))
        content_layout.addWidget(
            _hint(
                "Configurações do SM persistidas em "
                "~/.config/geosteering_ai/preferences.json. A aplicação ao vivo de "
                "tema/backend/cache às demais perspectivas será integrada numa fatia "
                "futura — aqui as preferências são salvas para o próximo arranque."
            )
        )
        content_layout.addWidget(
            _group(
                "Aparência",
                "Tema visual e backend de renderização default da galeria de plots.",
                appearance_form,
            )
        )
        content_layout.addWidget(
            _group(
                "Cache de plots (LRU)",
                "Limites do cache de snapshots de plot — teto de memória e nº máximo "
                "de entradas.",
                cache_form,
            )
        )
        perf_form = QtWidgets.QFormLayout()
        perf_form.addRow(self._jax_boot_warmup)
        content_layout.addWidget(
            _group(
                "Desempenho (JAX GPU)",
                "Pré-aquecimento opcional do worker JAX GPU persistente no boot — tira o "
                "cold-start XLA da 1ª simulação (opt-in; gasta GPU no arranque).",
                perf_form,
            )
        )
        content_layout.addWidget(
            _group(
                "Caminhos",
                "Localizações de binários e diretórios (vazio = autodetecção). "
                "Passe o mouse sobre cada campo para detalhes.",
                paths_form,
            )
        )
        content_layout.addLayout(btn_row)
        content_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("PreferencesScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

        # ── Binding ──────────────────────────────────────────────────────────
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._restore_btn.clicked.connect(self._on_restore_clicked)
        self._vm.saved.connect(self._on_saved)

        # Popula os widgets com o estado atual do VM (já carregado do disco).
        self._sync_from_vm()

    # ── Helpers de catálogo (puros; degradam gracioso se o módulo faltar) ────
    @staticmethod
    def _available_themes() -> list[str]:
        """Nomes de tema disponíveis (hoje só ``antigravity_dark`` — spec 0013)."""
        return ["antigravity_dark"]

    @staticmethod
    def _available_backends() -> list[str]:
        """Backends de plot disponíveis no ambiente (de ``available_backends()``).

        Degrada gracioso (``["matplotlib"]``) se o módulo de backends não importar
        — assim a View nunca crasha por dependência opcional ausente.
        """
        try:
            from geosteering_ai.gui.plot_backends.base import available_backends

            return [b.value for b in available_backends()] or ["matplotlib"]
        except Exception:  # noqa: BLE001 — dep opcional ausente → fallback seguro
            return ["matplotlib"]

    # ── Sync VM → widgets (após init e após "Restaurar padrões") ─────────────
    def _sync_from_vm(self) -> None:
        """Reflete o estado do VM nos widgets (não dispara push de volta ao VM)."""
        self._set_combo(self._theme, self._vm.theme)
        self._set_combo(self._plot_backend, self._vm.plot_backend)
        self._cache_mb.setValue(int(self._vm.cache_max_mb))
        self._cache_snaps.setValue(int(self._vm.cache_max_snapshots))
        self._jax_boot_warmup.setChecked(self._vm.jax_boot_warmup)
        for key, edit in self._path_edits.items():
            edit.setText(self._vm.get_path(key))

    @staticmethod
    def _set_combo(combo: Any, value: str) -> None:
        """Seleciona ``value`` no combo; se ausente, adiciona-o (preserva o salvo).

        Garante que um valor persistido fora do catálogo atual (ex.: backend não
        mais disponível) não seja silenciosamente perdido ao reabrir a tela.
        """
        idx = combo.findText(value)
        if idx < 0 and value:
            combo.addItem(value)
            idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ── Push widgets → VM (só em "Salvar") ───────────────────────────────────
    def _push_to_vm(self) -> None:
        """Copia os widgets para o VM (via setters/props) — pré-persistência."""
        self._vm.theme = self._theme.currentText()
        self._vm.plot_backend = self._plot_backend.currentText()
        self._vm.cache_max_mb = self._cache_mb.value()
        self._vm.cache_max_snapshots = self._cache_snaps.value()
        self._vm.jax_boot_warmup = self._jax_boot_warmup.isChecked()
        for key, edit in self._path_edits.items():
            self._vm.set_path(key, edit.text().strip())

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_browse(self, key: str, mode: str, label: str) -> None:
        """Abre o seletor (dir/arquivo) e preenche o campo do path ``key``."""
        edit = self._path_edits[key]
        current = edit.text().strip()
        if mode == "dir":
            path = QtWidgets.QFileDialog.getExistingDirectory(self, label, current)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, label, current)
        if path:
            edit.setText(path)

    def _on_save_clicked(self) -> None:
        """Copia os widgets ao VM e persiste (status reflete sucesso/erro)."""
        self._push_to_vm()
        try:
            self._vm.save()
        except OSError as exc:
            self._status.setText(f"falha ao salvar: {exc}")

    def _on_restore_clicked(self) -> None:
        """Restaura os valores-padrão no VM e repovoa os widgets (sem persistir)."""
        self._vm.restore_defaults()
        self._sync_from_vm()
        self._status.setText("padrões restaurados (não salvos)")

    def _on_saved(self) -> None:
        """Feedback de persistência bem-sucedida."""
        self._status.setText("✓ preferências salvas")
