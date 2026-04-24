# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/simulation_manager.py                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — GUI PyQt (entrypoint)                 ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Atualizado  : 2026-04-18 (dark theme · tabs · experiment dialog)        ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : PyQt6 (fallback PySide6/PyQt5) + matplotlib                ║
# ║  Dependências: numpy, scipy, matplotlib, geosteering_ai.simulation        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Janela principal do Simulation Manager com tema dark (VSCode-         ║
# ║    inspired) e navegação por abas superiores. Ao inicializar, o         ║
# ║    usuário obrigatoriamente cria ou abre um "experimento" — arquivo    ║
# ║    ``*.exp.json`` contendo todos os parâmetros (geometria, geração,     ║
# ║    backend, benchmark, preferências). Após o experimento estar         ║
# ║    carregado, as abas ficam habilitadas.                                ║
# ║                                                                           ║
# ║  ABAS (QTabWidget no topo)                                                ║
# ║    (1) Parâmetros — geometria + filtro Hankel + geração estocástica    ║
# ║    (2) Simulador  — backend, workers sandbox, execução, log            ║
# ║    (3) Benchmark  — Config A/B/C + experimento 30k                      ║
# ║    (4) Resultados — plots + seletor de modelo (simulação ou benchmark) ║
# ║    (5) Preferências — caminhos manuais + plot style + LaTeX             ║
# ║                                                                           ║
# ║  CONFIG C (modelos canônicos)                                             ║
# ║    Cada modelo canônico tem parâmetros próprios (freq/TR/dip) lidos   ║
# ║    de CANONICAL_MODEL_CONFIGS no sm_benchmark, baseados nos scripts     ║
# ║    de validação (bench_numba_vs_fortran_local.py, buildValidamodels.py). ║
# ║    O usuário pode habilitar modelos individualmente e customizar seus   ║
# ║    parâmetros via diálogo dedicado.                                     ║
# ║                                                                           ║
# ║  EXECUÇÃO                                                                 ║
# ║    python -m geosteering_ai.simulation.tests.simulation_manager          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Simulation Manager — entrypoint GUI PyQt (tema escuro VSCode + tabs).

Programa executável único com tema escuro VSCode-style, navegação por abas
superiores, e sistema de experimentos persistidos em ``.exp.json``.

Example:
    Execução direta::

        $ python -m geosteering_ai.simulation.tests.simulation_manager

Note:
    Requer que pelo menos um binding Qt esteja instalado. Se ausente, o
    programa emite mensagem clara de instalação e termina com exit code 1.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sm_qt_compat import (
    ALIGN_CENTER,
    ALIGN_RIGHT,
    ALIGN_VCENTER,
    ORIENT_H,
    QT_AVAILABLE,
    QT_BINDING,
    QtCore,
    QtGui,
    QtWidgets,
    Signal,
    check_qt_available,
    enforce_c_locale,
    format_float,
    make_double_spin,
)

if QT_AVAILABLE:
    from .sm_benchmark import (
        CANONICAL_BENCHMARK_MODELS,
        BenchmarkSettings,
        BenchmarkThread,
        BenchRecord,
        default_canonical_config,
    )
    from .sm_io import compute_nmeds_per_angle
    from .sm_model_gen import GENERATORS_AVAILABLE, GenConfig, generate_models
    from .sm_plots import (
        COMPONENT_NAMES,
        PLOT_KINDS,
        EMCanvas,
        PlotStyle,
        apply_style,
        plot_anisotropy,
        plot_benchmark_compare,
        plot_em_profile,
        plot_geosignals,
        plot_resistivity_profile,
        plot_tensor_full,
    )
    from .sm_workers import SaveArtifactsThread, SimRequest, SimulationThread


# ══════════════════════════════════════════════════════════════════════════
# Detecção automática de paths
# ══════════════════════════════════════════════════════════════════════════


def find_geosteering_ai_root(start: Optional[Path] = None) -> Optional[Path]:
    """Busca ascendente por ``geosteering_ai/__init__.py``."""
    start = (start or Path.cwd()).resolve()
    for parent in [start, *start.parents]:
        candidate = parent / "geosteering_ai" / "__init__.py"
        if candidate.is_file():
            return parent / "geosteering_ai"
        if (parent / "__init__.py").is_file() and parent.name == "geosteering_ai":
            return parent
    return None


def find_tatu_binary(start: Optional[Path] = None) -> Optional[Path]:
    """Tenta localizar ``tatu.x`` na árvore do projeto."""
    start = (start or Path.cwd()).resolve()
    for parent in [start, *start.parents]:
        cand = parent / "Fortran_Gerador" / "tatu.x"
        if cand.is_file():
            return cand
    return None


# ══════════════════════════════════════════════════════════════════════════
# QSettings + paths
# ══════════════════════════════════════════════════════════════════════════


def _qsettings() -> Any:
    return QtCore.QSettings("Geosteering AI", "Simulation Manager")


def _bool(v: Any, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def load_plot_style() -> PlotStyle:
    s = _qsettings()
    default = PlotStyle()
    try:
        return PlotStyle(
            dpi=int(s.value("plot/dpi", default.dpi)),
            font_family=str(s.value("plot/font_family", default.font_family)),
            font_size=int(s.value("plot/font_size", default.font_size)),
            line_width=float(s.value("plot/line_width", default.line_width)),
            grid=_bool(s.value("plot/grid", default.grid)),
            grid_alpha=float(s.value("plot/grid_alpha", default.grid_alpha)),
            color_real=str(s.value("plot/color_real", default.color_real)),
            color_imag=str(s.value("plot/color_imag", default.color_imag)),
            color_mag=str(s.value("plot/color_mag", default.color_mag)),
            color_phase=str(s.value("plot/color_phase", default.color_phase)),
            color_rho_h=str(s.value("plot/color_rho_h", default.color_rho_h)),
            color_rho_v=str(s.value("plot/color_rho_v", default.color_rho_v)),
            color_numba=str(s.value("plot/color_numba", default.color_numba)),
            color_fortran=str(s.value("plot/color_fortran", default.color_fortran)),
            color_layer_boundary=str(
                s.value("plot/color_layer_boundary", default.color_layer_boundary)
            ),
            axis_precision=int(s.value("plot/axis_precision", default.axis_precision)),
            show_layer_boundaries=_bool(
                s.value("plot/show_layer_boundaries", default.show_layer_boundaries)
            ),
            palette=str(s.value("plot/palette", default.palette)),
            background=str(s.value("plot/background", default.background)),
            tight_layout=_bool(s.value("plot/tight_layout", default.tight_layout)),
            use_latex=_bool(s.value("plot/use_latex", default.use_latex), default=False),
            use_mathtext=_bool(s.value("plot/use_mathtext", default.use_mathtext)),
            legend_location=str(s.value("plot/legend_location", default.legend_location)),
            title_location=str(s.value("plot/title_location", default.title_location)),
            minor_ticks=_bool(
                s.value("plot/minor_ticks", default.minor_ticks), default=False
            ),
            spine_width=float(s.value("plot/spine_width", default.spine_width)),
            line_style=str(s.value("plot/line_style", default.line_style)),
            marker_size=float(s.value("plot/marker_size", default.marker_size)),
            marker_style=str(s.value("plot/marker_style", default.marker_style)),
        )
    except Exception:
        return default


def save_plot_style(style: PlotStyle) -> None:
    s = _qsettings()
    for k, v in asdict(style).items():
        s.setValue(f"plot/{k}", v)


def load_paths() -> Dict[str, str]:
    s = _qsettings()
    detected_pkg = find_geosteering_ai_root()
    detected_tatu = find_tatu_binary()
    return {
        "geosteering_ai": str(
            s.value("paths/geosteering_ai", str(detected_pkg) if detected_pkg else "")
        ),
        "tatu_binary": str(
            s.value("paths/tatu_binary", str(detected_tatu) if detected_tatu else "")
        ),
        "python_binary": str(s.value("paths/python_binary", sys.executable)),
        "output_dir": str(s.value("paths/output_dir", str(Path.cwd() / "sm_output"))),
    }


def save_paths(paths: Dict[str, str]) -> None:
    s = _qsettings()
    for k, v in paths.items():
        s.setValue(f"paths/{k}", v)


# ══════════════════════════════════════════════════════════════════════════
# Experimento (.exp.json)
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class SimulationSnapshot:
    """Registro persistível de uma execução de simulação no experimento.

    Armazena apenas **metadados e parâmetros** — o tensor H (pesado) fica
    em cache paralelo em memória (``MainWindow._sim_history_cache``) e é
    perdido ao fechar o programa. Ao reabrir um ``.exp.json``, snapshots
    mostram parâmetros e permitem reexecução com os mesmos settings.

    Attributes:
        snapshot_id: UUID4 (identificador único estável).
        timestamp: ISO-8601 local do instante de finalização.
        label: Texto exibido no QListWidget. Ex.:
            ``"#3 — 2026-04-20 15:42 — numba, 1000 mod, 4w"``.
        backend: ``"numba"`` | ``"fortran"``.
        n_models: Quantidade de modelos simulados.
        n_workers: Workers paralelos (ProcessPoolExecutor).
        n_threads: Threads por worker.
        params_snapshot: ``ParametersPage.to_dict()`` (geometria + geração
            + filtro). Permite reconstruir a aba Simulador.
        exec_params: Parâmetros derivados usados na execução (freqs, trs,
            dips, h1, tj, p_med, n_pos, positions_z range).
        elapsed_s: Duração total da simulação em segundos.
        artifacts: ``{"dat": path, "out": path}`` se artefatos foram
            exportados; ``None`` caso contrário.
    """

    snapshot_id: str = ""
    timestamp: str = ""
    label: str = ""
    backend: str = "numba"
    n_models: int = 0
    n_workers: int = 1
    n_threads: int = 1
    params_snapshot: Dict[str, Any] = field(default_factory=dict)
    exec_params: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    artifacts: Optional[Dict[str, str]] = None

    def format_info(self) -> str:
        """Formata snapshot como texto multi-linha para o painel lateral.

        Returns:
            String Markdown-like pronta para ``QTextEdit.setText``.
        """
        lines: List[str] = []
        lines.append(f"<b>{self.label}</b>")
        lines.append(f"<i>snapshot_id</i>: <code>{self.snapshot_id[:8]}…</code>")
        lines.append(f"<i>timestamp</i>: {self.timestamp}")
        lines.append(f"<i>backend</i>: <b>{self.backend}</b>")
        lines.append(
            f"<i>n_models</i>: {self.n_models} · "
            f"<i>workers</i>: {self.n_workers} · "
            f"<i>threads/worker</i>: {self.n_threads}"
        )
        lines.append(f"<i>elapsed</i>: {self.elapsed_s:.2f} s")
        if self.artifacts:
            lines.append(f"<i>artifacts</i>: {self.artifacts.get('dat', '—')}")
        lines.append("<hr/>")
        lines.append("<b>Parâmetros de execução</b>")
        exec_p = self.exec_params or {}
        for k, v in exec_p.items():
            lines.append(f"  • <i>{k}</i>: {v}")
        lines.append("<hr/>")
        lines.append("<b>Snapshot da aba Parâmetros</b>")
        ps = self.params_snapshot or {}
        for k, v in ps.items():
            # Truncar listas longas (freqs, tr, dips, rho arrays)
            sval = str(v)
            if len(sval) > 120:
                sval = sval[:117] + "…"
            lines.append(f"  • <i>{k}</i>: {sval}")
        return "<br/>".join(lines)


@dataclass
class ExperimentState:
    """Estado persistível de um experimento Simulation Manager.

    Attributes:
        name: Nome human-readable do experimento.
        description: Descrição livre.
        created_at: ISO-8601 do instante de criação.
        updated_at: ISO-8601 da última edição.
        file_path: Caminho absoluto do arquivo ``.exp.json``. Pode ser
            vazio se o experimento ainda não foi salvo.
        output_dir: Diretório de saída (artefatos .dat/.out/plots).
        parameters: dict com todos os parâmetros (geometria, geração,
            filtro, benchmark, plot_style). Formato JSON serializável.
        simulations: Lista de :class:`SimulationSnapshot` persistidos (v2.4+).
            Backward-compat: arquivos .exp.json anteriores não possuem este
            campo e recebem ``[]`` ao carregar.
    """

    name: str = "Experimento sem título"
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    file_path: str = ""
    output_dir: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    simulations: List[SimulationSnapshot] = field(default_factory=list)

    def to_json(self) -> str:
        """Serializa em JSON indentado (preserva ordem de chaves)."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str) -> "ExperimentState":
        """Carrega estado de um arquivo ``.exp.json``.

        Backward-compatível com arquivos pré-v2.4 (sem campo ``simulations``):
        chaves desconhecidas são descartadas; ``simulations`` é convertido
        de ``List[Dict]`` em ``List[SimulationSnapshot]``.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Filtra chaves desconhecidas (forward-compat se .exp.json tiver
        # campos extra de versões futuras) e converte simulations.
        known = {f.name for f in fields(cls)}
        raw_sims = data.get("simulations", []) or []
        data = {k: v for k, v in data.items() if k in known}
        sims: List[SimulationSnapshot] = []
        for s in raw_sims:
            if isinstance(s, dict):
                snap_known = {f.name for f in fields(SimulationSnapshot)}
                filtered = {k: v for k, v in s.items() if k in snap_known}
                sims.append(SimulationSnapshot(**filtered))
        data["simulations"] = sims
        return cls(**data)

    def save(self, path: Optional[str] = None) -> str:
        """Salva em disco. Retorna o caminho final utilizado."""
        import datetime

        target = path or self.file_path
        if not target:
            raise ValueError("Caminho do experimento não definido.")
        self.file_path = target
        now = datetime.datetime.now().isoformat(timespec="seconds")
        if not self.created_at:
            self.created_at = now
        self.updated_at = now
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return target

    # ── Hist\u00f3rico de simula\u00e7\u00f5es (v2.4+) ───────────────────────────────
    def append_simulation(self, snapshot: SimulationSnapshot) -> None:
        """Adiciona um snapshot ao histórico. Atualiza ``updated_at``."""
        import datetime

        self.simulations.append(snapshot)
        self.updated_at = datetime.datetime.now().isoformat(timespec="seconds")

    def clear_simulations(self) -> None:
        """Remove todos os snapshots do histórico."""
        import datetime

        self.simulations.clear()
        self.updated_at = datetime.datetime.now().isoformat(timespec="seconds")

    def remove_simulation(self, snapshot_id: str) -> bool:
        """Remove um snapshot por ID. Retorna ``True`` se encontrado."""
        for i, s in enumerate(self.simulations):
            if s.snapshot_id == snapshot_id:
                del self.simulations[i]
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════
# Stylesheet — Dark theme (VSCode-inspired)
# ══════════════════════════════════════════════════════════════════════════

APP_STYLESHEET = """
/* ═══════════════════════════════════════════════════════════════════
   Dark theme — VSCode-inspired
   Background principal:  #1e1e1e   (editor bg)
   Superficies elevadas:  #252526   (sidebar/panel bg)
   Bordas / separadores:  #3c3c3c
   Texto base:            #d4d4d4
   Texto muted:           #a5a5a5
   Texto apagado:         #858585
   Acento azul:           #007acc   (VSCode blue)
   Acento hover:          #1e8fe6
   Alerta/stop vermelho:  #e5414e
═══════════════════════════════════════════════════════════════════ */

QMainWindow, QDialog, QWidget {
    background: #1e1e1e;
    color: #d4d4d4;
    font-size: 13px;
    selection-background-color: #094771;
    selection-color: #ffffff;
}

QLabel { color: #d4d4d4; background: transparent; }
QLabel[role="heading"] {
    font-size: 16px; font-weight: 600; color: #ffffff;
    padding: 2px 0 6px 0;
}
QLabel[role="section"] {
    font-size: 11px; color: #a5a5a5; letter-spacing: 0.4px;
    text-transform: uppercase; padding: 6px 0 2px 0;
}
QLabel[role="hint"] {
    color: #858585; font-size: 11px; padding: 2px 0;
}

/* ─── Painéis e grupos ─────────────────────────────────────────── */
QWidget#Panel {
    background: #252526; border-radius: 6px;
    border: 1px solid #3c3c3c;
}
QGroupBox {
    background: #252526; border: 1px solid #3c3c3c;
    border-radius: 6px; margin-top: 16px; padding: 10px;
    font-weight: 600; color: #d4d4d4;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: #a5a5a5;
}

/* ─── Campos de entrada ─────────────────────────────────────────── */
QLineEdit, QDoubleSpinBox, QSpinBox, QPlainTextEdit, QTextEdit {
    background: #1e1e1e; border: 1px solid #3c3c3c;
    border-radius: 4px; padding: 4px 8px; color: #d4d4d4;
    selection-background-color: #264f78; selection-color: #ffffff;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border: 1px solid #007acc;
}
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled,
QComboBox:disabled { color: #6a6a6a; background: #2a2a2a; }

QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {
    background: #2d2d2d; border: 1px solid #3c3c3c;
    border-radius: 2px;
}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid #d4d4d4;
}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid #d4d4d4;
}

/* ─── ComboBox — garantir contraste e sem clipping de texto ──────── */
QComboBox {
    background: #1e1e1e; border: 1px solid #3c3c3c;
    border-radius: 4px; padding: 5px 30px 5px 10px;     /* padding-right reserva espaço para a seta */
    color: #d4d4d4; min-height: 26px; min-width: 120px;
}
QComboBox:hover { border: 1px solid #007acc; }
QComboBox:on { /* quando a popup está aberta */
    border: 1px solid #007acc; background: #2a2a2a;
}
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: top right;
    width: 22px; border: none; background: transparent;
}
QComboBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid #d4d4d4;
    margin-right: 6px;
}
QComboBox::down-arrow:on { border-top-color: #ffffff; }

/* ── CRÍTICO: a lista popup do ComboBox usa QListView internamente.
      Estilizar ambos explicitamente garante legibilidade em Qt5/Qt6. ── */
QComboBox QAbstractItemView,
QComboBox QListView {
    background: #252526; color: #d4d4d4;
    border: 1px solid #3c3c3c; outline: 0;
    selection-background-color: #094771;
    selection-color: #ffffff;
    padding: 2px;
    min-width: 160px;
}
QComboBox QAbstractItemView::item,
QComboBox QListView::item {
    padding: 6px 14px; min-height: 26px;
    color: #d4d4d4; background: transparent;
    border: none;
}
QComboBox QAbstractItemView::item:selected,
QComboBox QListView::item:selected {
    background: #094771; color: #ffffff;
}
QComboBox QAbstractItemView::item:hover,
QComboBox QListView::item:hover {
    background: #37373d; color: #ffffff;
}

/* ─── Botões ─────────────────────────────────────────────────────── */
QPushButton {
    background: #2d2d30; border: 1px solid #3c3c3c;
    border-radius: 4px; padding: 6px 14px; min-height: 22px;
    color: #d4d4d4;
}
QPushButton:hover { background: #37373d; border: 1px solid #007acc; }
QPushButton:pressed { background: #1e1e1e; }
QPushButton:disabled { color: #6a6a6a; background: #262626; border-color: #333; }
QPushButton[role="primary"] {
    background: #007acc; color: #ffffff; border: 1px solid #1c97ea;
    font-weight: 600;
}
QPushButton[role="primary"]:hover { background: #1e8fe6; }
QPushButton[role="primary"]:pressed { background: #005a9e; }
QPushButton[role="primary"]:disabled {
    background: #30475c; color: #8ab9d8; border-color: #2a3f50;
}
QPushButton[role="danger"] {
    background: #e5414e; color: #ffffff; border: 1px solid #b8333e; font-weight: 600;
}
QPushButton[role="danger"]:hover  { background: #f05560; }
QPushButton[role="danger"]:disabled { background: #5a2a2f; color: #b47478; border-color: #3a1d21; }
QPushButton[role="ghost"] {
    background: transparent; color: #a5a5a5; border: 1px solid transparent;
}
QPushButton[role="ghost"]:hover { color: #ffffff; background: #2d2d30; }

/* ─── CheckBox / RadioButton ─────────────────────────────────────── */
QCheckBox, QRadioButton { color: #d4d4d4; background: transparent; spacing: 6px; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px; height: 14px;
    background: #1e1e1e; border: 1px solid #666; border-radius: 2px;
}
QCheckBox::indicator:checked {
    background: #007acc; border: 1px solid #007acc;
    image: none;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border: 1px solid #007acc; }
QRadioButton::indicator { border-radius: 7px; }
QRadioButton::indicator:checked { background: #007acc; border: 1px solid #007acc; }

/* ─── Tabs ───────────────────────────────────────────────────────── */
QTabWidget::pane {
    background: #1e1e1e; border: 1px solid #3c3c3c; border-top: 0;
    top: -1px;
}
QTabBar::tab {
    background: #2d2d30; color: #a5a5a5;
    border: 1px solid #3c3c3c; border-bottom: none;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
    padding: 8px 18px; margin-right: 2px; min-width: 120px;
}
QTabBar::tab:selected {
    background: #1e1e1e; color: #ffffff;
    border-bottom: 2px solid #007acc;
}
QTabBar::tab:hover:!selected {
    background: #37373d; color: #d4d4d4;
}
QTabBar::tab:disabled { color: #555555; background: #222222; }

/* ─── Progress & Status ─────────────────────────────────────────── */
QProgressBar {
    border: 1px solid #3c3c3c; border-radius: 4px; height: 14px;
    text-align: center; background: #252526; color: #d4d4d4;
}
QProgressBar::chunk { background: #007acc; border-radius: 3px; }
QStatusBar { background: #007acc; color: #ffffff; border-top: 1px solid #0a6cb4; }
QStatusBar QLabel { color: #ffffff; }

/* ─── Tabela ─────────────────────────────────────────────────────── */
QTableWidget {
    background: #1e1e1e; alternate-background-color: #252526;
    border: 1px solid #3c3c3c; border-radius: 4px;
    gridline-color: #3c3c3c; color: #d4d4d4;
}
QTableWidget::item:selected { background: #094771; color: #ffffff; }
QHeaderView::section {
    background: #2d2d30; color: #d4d4d4; padding: 6px 10px;
    border: none; border-bottom: 1px solid #3c3c3c; font-weight: 600;
}

/* ─── Log ────────────────────────────────────────────────────────── */
QPlainTextEdit#LogView {
    background: #0b0b0b; color: #e6e6e6;
    font-family: "Menlo", "Consolas", "Monaco", monospace;
    font-size: 12px; border-radius: 4px; padding: 8px;
    border: 1px solid #3c3c3c;
}

/* ─── Scroll ─────────────────────────────────────────────────────── */
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical {
    background: #252526; width: 12px; margin: 4px 2px;
}
QScrollBar::handle:vertical {
    background: #3e3e42; border-radius: 4px; min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: #4e4e52; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal {
    background: #252526; height: 12px; margin: 2px 4px;
}
QScrollBar::handle:horizontal {
    background: #3e3e42; border-radius: 4px; min-width: 28px;
}

/* ─── ToolBar ────────────────────────────────────────────────────── */
QToolBar {
    background: #252526; border-bottom: 1px solid #3c3c3c; spacing: 6px;
    padding: 6px;
}
QToolBar QToolButton {
    padding: 6px 10px; border-radius: 4px; color: #d4d4d4;
}
QToolBar QToolButton:hover { background: #37373d; }

/* ─── ListWidget ─────────────────────────────────────────────────── */
QListWidget {
    background: #1e1e1e; border: 1px solid #3c3c3c; border-radius: 4px;
    color: #d4d4d4; outline: 0;
}
QListWidget::item { padding: 4px 8px; }
QListWidget::item:selected { background: #094771; color: #ffffff; }
QListWidget::item:hover:!selected { background: #2a2d2e; }

/* ─── Splitter ───────────────────────────────────────────────────── */
QSplitter::handle { background: #3c3c3c; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical   { height: 1px; }

/* ─── Tooltip ────────────────────────────────────────────────────── */
QToolTip {
    background: #252526; color: #d4d4d4;
    border: 1px solid #3c3c3c; padding: 6px 8px;
    border-radius: 4px;
}

/* ─── MessageBox buttons ─────────────────────────────────────────── */
QDialogButtonBox QPushButton { min-width: 80px; }
"""


# ══════════════════════════════════════════════════════════════════════════
# Helpers de UI
# ══════════════════════════════════════════════════════════════════════════


def _qline(default: str = "", placeholder: str = "") -> "QtWidgets.QLineEdit":
    w = QtWidgets.QLineEdit()
    w.setText(default)
    if placeholder:
        w.setPlaceholderText(placeholder)
    return w


def _spin_int(default: int, lo: int, hi: int, step: int = 1) -> "QtWidgets.QSpinBox":
    w = QtWidgets.QSpinBox()
    try:
        c_locale = (
            QtCore.QLocale(QtCore.QLocale.Language.C)
            if hasattr(QtCore.QLocale, "Language")
            else QtCore.QLocale.c()
        )
        try:
            c_locale.setNumberOptions(QtCore.QLocale.NumberOption.OmitGroupSeparator)
        except Exception:
            pass
        w.setLocale(c_locale)
    except Exception:
        pass
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(default)
    return w


def _spin_float(
    default: float,
    lo: float,
    hi: float,
    step: float = 0.1,
    decimals: int = 3,
    suffix: str = "",
) -> "QtWidgets.QDoubleSpinBox":
    return make_double_spin(default, lo, hi, step, decimals, suffix)


def _combo(items: List[str], default: Optional[str] = None) -> "QtWidgets.QComboBox":
    """Cria ``QComboBox`` com size adjust policy adequado para evitar clipping.

    - ``AdjustToContents`` força o combo a ter largura suficiente para o maior item.
    - ``setMinimumContentsLength`` garante largura mínima mesmo em listas curtas.
    - View popup herda o stylesheet global (dark com seleção azul).
    """
    w = QtWidgets.QComboBox()
    try:
        policy = getattr(QtWidgets.QComboBox.SizeAdjustPolicy, "AdjustToContents", None)
        if policy is None:
            policy = QtWidgets.QComboBox.AdjustToContents
        w.setSizeAdjustPolicy(policy)
    except Exception:
        pass
    w.setMinimumContentsLength(max(8, max((len(s) for s in items), default=8)))
    for it in items:
        w.addItem(it)
    if default is not None:
        idx = w.findText(default)
        if idx >= 0:
            w.setCurrentIndex(idx)
    return w


def _parse_float_list(text: str, default: List[float]) -> List[float]:
    """Interpreta "1.0, 2.0, 3.0" → [1.0, 2.0, 3.0]; devolve default se vazio."""
    if not text or not text.strip():
        return list(default)
    try:
        raw = text.replace(";", ",")
        parts = [t.strip() for t in raw.split(",") if t.strip()]
        return [float(p) for p in parts]
    except ValueError:
        return list(default)


def _frame_shape(name: str) -> Any:
    shape_enum = getattr(QtWidgets.QFrame, "Shape", None)
    if shape_enum is not None:
        return getattr(shape_enum, name)
    return getattr(QtWidgets.QFrame, name)


def _frame_shadow(name: str) -> Any:
    shadow_enum = getattr(QtWidgets.QFrame, "Shadow", None)
    if shadow_enum is not None:
        return getattr(shadow_enum, name)
    return getattr(QtWidgets.QFrame, name)


def _hsep() -> "QtWidgets.QFrame":
    sep = QtWidgets.QFrame()
    sep.setFrameShape(_frame_shape("HLine"))
    sep.setFrameShadow(_frame_shadow("Sunken"))
    sep.setStyleSheet("color:#3c3c3c;")
    return sep


def _heading(text: str, role: str = "heading") -> "QtWidgets.QLabel":
    lbl = QtWidgets.QLabel(text)
    lbl.setProperty("role", role)
    lbl.setWordWrap(True)
    return lbl


def _centered_form() -> Tuple["QtWidgets.QFormLayout", "QtWidgets.QHBoxLayout"]:
    """Cria um ``QFormLayout`` centralizado horizontalmente.

    Retorna ``(form, outer_hbox)`` — o ``outer_hbox`` deve ser adicionado
    ao layout pai; o ``form`` já vive dentro dele centralizado.
    """
    form = QtWidgets.QFormLayout()
    form.setLabelAlignment(ALIGN_RIGHT)
    form.setFormAlignment(ALIGN_CENTER | ALIGN_VCENTER if ALIGN_CENTER else 0x4)
    form.setHorizontalSpacing(16)
    form.setVerticalSpacing(8)
    outer = QtWidgets.QHBoxLayout()
    outer.addStretch(1)
    wrap = QtWidgets.QWidget()
    wrap.setLayout(form)
    wrap.setMaximumWidth(720)
    outer.addWidget(wrap)
    outer.addStretch(1)
    return form, outer


def _color_for_bg(hex_color: str) -> str:
    try:
        c = hex_color.lstrip("#")
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000000" if lum > 150 else "#ffffff"
    except Exception:
        return "#ffffff"


def _tooltip(widget: Any, text: str) -> None:
    """Aplica tooltip de forma compatível (normaliza wrap em html)."""
    widget.setToolTip(f"<div style='max-width:420px; padding:4px;'>{text}</div>")


# ══════════════════════════════════════════════════════════════════════════
# Dialog — Novo experimento / Abrir
# ══════════════════════════════════════════════════════════════════════════


class NewExperimentDialog(QtWidgets.QDialog):
    """Diálogo para criar um novo experimento ``.exp.json``."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Novo experimento")
        self.setMinimumWidth(560)
        self._result: Optional[ExperimentState] = None

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(ALIGN_RIGHT)
        form.setHorizontalSpacing(12)

        self.edit_name = _qline(
            "Experimento " + QtCore.QDate.currentDate().toString("yyyy-MM-dd")
        )
        _tooltip(
            self.edit_name,
            (
                "<b>Nome do experimento</b><br/>"
                "Título human-readable; também usado para compor o nome do arquivo "
                "<code>.exp.json</code> (caracteres não-alfanuméricos viram '_')."
            ),
        )
        self.edit_desc = QtWidgets.QPlainTextEdit()
        self.edit_desc.setPlaceholderText(
            "Descrição livre do experimento — objetivo, hipóteses, observações…"
        )
        self.edit_desc.setMaximumHeight(80)
        _tooltip(
            self.edit_desc,
            (
                "<b>Descrição livre</b><br/>"
                "Notas sobre motivação, hipóteses e observações. Salva no .exp.json "
                "e exibida ao reabrir. Não afeta a execução."
            ),
        )

        default_dir = str(Path.cwd() / "sm_experiments")
        self.edit_dir = _qline(default_dir, "Pasta do experimento")
        _tooltip(
            self.edit_dir,
            (
                "<b>Diretório do experimento</b><br/>"
                "Pasta onde serão salvos o <code>.exp.json</code> e os artefatos "
                "(.dat/.out/plots) gerados. Criada automaticamente se não existir."
            ),
        )
        btn_dir = QtWidgets.QPushButton("Procurar…")
        btn_dir.clicked.connect(self._browse_dir)
        _tooltip(btn_dir, "Abrir seletor de pastas do sistema.")
        row_dir = QtWidgets.QHBoxLayout()
        row_dir.addWidget(self.edit_dir, 1)
        row_dir.addWidget(btn_dir)
        wrap_dir = QtWidgets.QWidget()
        wrap_dir.setLayout(row_dir)

        form.addRow("Nome:", self.edit_name)
        form.addRow("Descrição:", self.edit_desc)
        form.addRow("Diretório de saída:", wrap_dir)

        btnbox = QtWidgets.QDialogButtonBox()
        btn_ok = btnbox.addButton(
            "Criar experimento", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        btn_cancel = btnbox.addButton(
            "Cancelar", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole
        )
        btn_ok.setProperty("role", "primary")
        btn_ok.clicked.connect(self._on_accept)
        btn_cancel.clicked.connect(self.reject)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(_heading("Criar novo experimento"))
        root.addWidget(
            _heading(
                "Um experimento agrupa todos os parâmetros (geometria, filtro, geração, "
                "benchmark, preferências) em um único arquivo .exp.json reabrível.",
                role="section",
            )
        )
        root.addLayout(form)
        root.addWidget(btnbox)

    def _browse_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Selecionar pasta de experimentos",
            self.edit_dir.text() or str(Path.cwd()),
        )
        if path:
            self.edit_dir.setText(path)

    def _on_accept(self) -> None:
        name = self.edit_name.text().strip() or "experimento_sem_nome"
        dir_path = self.edit_dir.text().strip() or str(Path.cwd() / "sm_experiments")
        os.makedirs(dir_path, exist_ok=True)
        # Slug do nome para o arquivo
        slug = "".join(c if c.isalnum() or c in "_-" else "_" for c in name).strip("_")
        file_path = os.path.join(dir_path, f"{slug or 'experimento'}.exp.json")
        exp = ExperimentState(
            name=name,
            description=self.edit_desc.toPlainText().strip(),
            file_path=file_path,
            output_dir=dir_path,
        )
        try:
            exp.save(file_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Falha ao salvar", str(e))
            return
        self._result = exp
        self.accept()

    def result_state(self) -> Optional[ExperimentState]:
        return self._result


# ══════════════════════════════════════════════════════════════════════════
# Welcome — tela inicial (bloqueia acesso até experimento carregado)
# ══════════════════════════════════════════════════════════════════════════


class WelcomeWidget(QtWidgets.QWidget):
    """Tela de boas-vindas exibida antes do experimento ser criado/aberto."""

    request_new = Signal()
    request_open = Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        lbl_title = QtWidgets.QLabel("Simulation Manager")
        lbl_title.setStyleSheet(
            "font-size: 28px; font-weight: 600; color: #ffffff; padding: 8px 0;"
        )
        lbl_subtitle = QtWidgets.QLabel(
            "Geosteering AI v2.0 · Python Numba JIT + Fortran tatu.x"
        )
        lbl_subtitle.setStyleSheet("color: #a5a5a5; font-size: 14px;")
        lbl_info = QtWidgets.QLabel(
            "Para começar, crie um novo experimento ou abra um existente. Cada experimento "
            "salva todos os parâmetros (geometria, filtro, geração, benchmark, preferências) "
            "em um arquivo <code>.exp.json</code> reabrível e versionável."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #d4d4d4; font-size: 13px; padding: 10px 0;")

        btn_new = QtWidgets.QPushButton("➕   Novo experimento")
        btn_new.setProperty("role", "primary")
        btn_new.setMinimumHeight(48)
        btn_new.setStyleSheet("font-size: 14px;")
        btn_new.clicked.connect(self.request_new.emit)

        btn_open = QtWidgets.QPushButton("📂   Abrir experimento…")
        btn_open.setMinimumHeight(48)
        btn_open.setStyleSheet("font-size: 14px;")
        btn_open.clicked.connect(self.request_open.emit)

        # Recentes (se houver)
        recents = self._load_recents()
        box_recents = QtWidgets.QGroupBox("Experimentos recentes")
        rec_layout = QtWidgets.QVBoxLayout(box_recents)
        if recents:
            for path in recents[:8]:
                btn = QtWidgets.QPushButton(f"🗂  {Path(path).name}   —   {path}")
                btn.setProperty("role", "ghost")
                btn.setStyleSheet("text-align:left; color:#d4d4d4;")
                btn.clicked.connect(lambda _=False, p=path: self._open_path(p))
                rec_layout.addWidget(btn)
        else:
            rec_layout.addWidget(QtWidgets.QLabel("(nenhum experimento recente)"))

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(60, 50, 60, 50)
        root.setSpacing(10)
        root.addWidget(lbl_title)
        root.addWidget(lbl_subtitle)
        root.addWidget(lbl_info)
        row_btns = QtWidgets.QHBoxLayout()
        row_btns.addWidget(btn_new)
        row_btns.addWidget(btn_open)
        row_btns.addStretch(1)
        root.addLayout(row_btns)
        root.addSpacing(20)
        root.addWidget(box_recents)
        root.addStretch(1)

    def _load_recents(self) -> List[str]:
        s = _qsettings()
        raw = s.value("experiment/recents", [])
        if isinstance(raw, str):
            return [raw] if raw else []
        try:
            return [str(x) for x in raw if x]
        except Exception:
            return []

    def _open_path(self, path: str) -> None:
        # emitido via attribute hack — parent ouvirá
        self.property_selected_path = path
        self.request_open.emit()


def _push_recent(path: str) -> None:
    s = _qsettings()
    raw = s.value("experiment/recents", [])
    if isinstance(raw, str):
        recents = [raw] if raw else []
    else:
        try:
            recents = [str(x) for x in raw if x]
        except Exception:
            recents = []
    recents = [p for p in recents if p != path]
    recents.insert(0, path)
    s.setValue("experiment/recents", recents[:10])


# ══════════════════════════════════════════════════════════════════════════
# Aba 1 — Parâmetros (centralizada + tooltips)
# ══════════════════════════════════════════════════════════════════════════


class ParametersPage(QtWidgets.QWidget):
    """Página de parâmetros da simulação e geração estocástica (centralizada)."""

    # Emitido sempre que um parâmetro que afeta o cálculo de ``n_pos``
    # (pontos de medição por modelo) muda: tj, p_med, ângulos de dip.
    # Consumido por SimulatorPage para atualizar ``lbl_npos`` em tempo real.
    parametersChanged = Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # Perfil canônico atualmente aplicado (None = geração estocástica livre).
        # Quando != None, ``_start_simulation`` usa este perfil fixo em vez de
        # gerar modelos estocásticos.
        self._canonical_override: Optional[Any] = None

        # Perfil manual (v2.4+). Dict com chaves ``thicknesses``, ``rho_h``,
        # ``rho_v``, ``n_layers`` preenchido via ``LayersManualDialog``. Quando
        # != None e ``radio_manual`` está ativo, ``_start_simulation`` replica
        # este modelo em vez de gerar estocasticamente ou usar canônico.
        self._manual_layers: Optional[Dict[str, Any]] = None

        # ═══ Grupo 1 — Geometria da Ferramenta ═══════════════════════════
        grp_geom = QtWidgets.QGroupBox("Geometria da Ferramenta")
        form_geom, wrap_geom = _centered_form()

        self.spin_nf = _spin_int(1, 1, 32)
        self.spin_nang = _spin_int(1, 1, 32)
        self.spin_ntr = _spin_int(1, 1, 32)
        self.edit_freqs = _qline("20000", "Ex.: 20000, 40000")
        self.edit_angles = _qline("0", "Ex.: 0, 30, 60")
        self.edit_trs = _qline("1.0", "Ex.: 0.5, 1.0, 1.5")
        self.spin_h1 = _spin_float(10.0, 0.1, 500.0, 0.5, 3, " m")
        self.spin_tj = _spin_float(120.0, 1.0, 5000.0, 1.0, 2, " m")
        self.spin_pmed = _spin_float(0.2, 0.01, 10.0, 0.01, 3, " m")

        # ── Auto-geometria (fidelidade buildValidamodels.py) ─────────────
        # Checkboxes que derivam h1 e tj da geometria do modelo canônico:
        #   tj = Σesp + 2·h1      (margem simétrica cobre toda a estratigrafia)
        #   h1 = (tj − Σesp) / 2  (centraliza as medições)
        # Quando ambas ativas e um perfil canônico estiver aplicado, h1 é
        # derivado como margem proporcional (max(5.0, 0.20·Σesp)) e tj segue
        # da fórmula. Os spinboxes ficam desabilitados enquanto o checkbox
        # correspondente está ativo (valor imutável pelo usuário).
        # v2.4: texto do QCheckBox mantido curto; legenda explicativa vai em
        # um QLabel abaixo (word-wrap ativado) para não vazar das margens.
        self.check_auto_h1 = QtWidgets.QCheckBox("h1 automático")
        self.check_auto_tj = QtWidgets.QCheckBox("tj automático")
        self.check_auto_h1.setChecked(False)
        self.check_auto_tj.setChecked(False)

        # Legendas word-wrap abaixo de cada checkbox (antes estavam entre
        # parênteses no texto do próprio QCheckBox, mas ficavam truncadas em
        # telas < 1280 px ou quando o tema aumentava a fonte).
        self._lbl_auto_h1_caption = QtWidgets.QLabel(
            "<i>(margem simétrica do perfil canônico)</i>"
        )
        self._lbl_auto_tj_caption = QtWidgets.QLabel(
            "<i>(Σesp + 2·h1, cobre toda a estratigrafia)</i>"
        )
        for _lbl in (self._lbl_auto_h1_caption, self._lbl_auto_tj_caption):
            _lbl.setWordWrap(True)
            _lbl.setStyleSheet("color:#a5a5a5; padding-left: 22px;")
        _tooltip(
            self.check_auto_h1,
            (
                "<b>h1 automático — centralização canônica</b><br/>"
                "Quando ativo e um perfil canônico estiver aplicado:<br/>"
                "&nbsp;&nbsp;<code>h1 = (tj − Σesp) / 2</code><br/>"
                "Centraliza a estratigrafia dentro da janela <i>tj</i> atual. "
                "O campo <i>h1</i> fica desabilitado. Reproduz a convenção de "
                "<code>buildValidamodels.py</code>."
            ),
        )
        _tooltip(
            self.check_auto_tj,
            (
                "<b>tj automático — janela global canônica</b><br/>"
                "Quando ativo e um perfil canônico estiver aplicado:<br/>"
                "&nbsp;&nbsp;<code>tj = max(Σesp across validation batch) + 20 m</code><br/>"
                "Batch de referência: Oklahoma 3/5/15/28, Devine 8, Hou 7 "
                "(max Σesp = 47.85 m em Oklahoma 28) → <b>tj ≈ 67.85 m</b>. "
                "Para modelos fora do batch (ex.: Viking Graben 10, Σesp=233 m), "
                "a janela é expandida para <code>Σesp + 20 m</code>. Reproduz "
                "fielmente <code>buildValidamodels.py</code>."
            ),
        )

        _tooltip(
            self.spin_nf,
            (
                "<b>Número de frequências (nf)</b><br/>"
                "Quantidade de frequências a serem propagadas no mesmo modelo. "
                "Deve corresponder ao comprimento da lista em 'Frequências (Hz)'. "
                "Usado para simulações multi-espectrais e compensação de fase."
            ),
        )
        _tooltip(
            self.spin_nang,
            (
                "<b>Número de ângulos (ntheta)</b><br/>"
                "Quantidade de ângulos de dip (inclinação relativa do poço em relação "
                "às camadas). Cada ângulo multiplica o custo da simulação."
            ),
        )
        _tooltip(
            self.spin_ntr,
            (
                "<b>Número de pares Transmissor–Receptor (TR)</b><br/>"
                "Quantidade de configurações geométricas T-R distintas a serem "
                "calculadas. Reside no primeiro eixo do tensor H de saída."
            ),
        )
        _tooltip(
            self.edit_freqs,
            (
                "<b>Frequências operacionais do LWD (Hz)</b><br/>"
                "Lista separada por vírgulas. Exemplos típicos da indústria: "
                "400 Hz, 2 kHz, 6 kHz, 20 kHz, 40 kHz, 100 kHz, 400 kHz, 2 MHz. "
                "Maior frequência → maior sensibilidade a camadas finas, menor profundidade."
            ),
        )
        _tooltip(
            self.edit_angles,
            (
                "<b>Ângulos de dip (graus)</b><br/>"
                "Inclinação do poço relativa à normal das camadas. 0° = perfil vertical (atravessa perpendicular); "
                "90° = perfil horizontal (paralelo às camadas). Lista separada por vírgulas."
            ),
        )
        _tooltip(
            self.edit_trs,
            (
                "<b>Espaçamentos Transmissor–Receptor (m)</b><br/>"
                "Lista separada por vírgulas. Valores típicos: 0.5 m (short), 1 m, 1.5 m, "
                "8.19 m, 20.43 m (deep). TR maior → maior profundidade de investigação, "
                "menor resolução vertical."
            ),
        )
        _tooltip(
            self.spin_h1,
            (
                "<b>Altura h1 (m) — topo do eixo de medição</b><br/>"
                "Distância do primeiro ponto-médio T-R até a primeira interface "
                "geológica. Em buildValidamodels.py é computada como "
                "(tj − Σespessuras)/2 para centralizar as medidas."
            ),
        )
        _tooltip(
            self.spin_tj,
            (
                "<b>Janela de investigação tj (m)</b><br/>"
                "Extensão total do eixo de medição ao longo do poço. Combinada com "
                "p_med, define n_pos = ⌈tj / (p_med · cos(dip))⌉ posições."
            ),
        )
        _tooltip(
            self.spin_pmed,
            (
                "<b>Passo entre medidas p_med (m)</b><br/>"
                "Distância entre posições consecutivas do ponto-médio T-R. "
                "Amostragem típica: 0.1 m (fina), 0.2 m (padrão), 0.5 m (rápida)."
            ),
        )

        form_geom.addRow("Nº de frequências (nf):", self.spin_nf)
        form_geom.addRow("Nº de ângulos (ntheta):", self.spin_nang)
        form_geom.addRow("Nº de pares T-R:", self.spin_ntr)
        form_geom.addRow("Frequências (Hz):", self.edit_freqs)
        form_geom.addRow("Ângulos de dip (graus):", self.edit_angles)
        form_geom.addRow("Espaçamentos T-R (m):", self.edit_trs)
        form_geom.addRow("Altura h1:", self.spin_h1)
        # v2.4: empacota checkbox + legenda word-wrap num QWidget para 2 linhas
        _auto_h1_container = QtWidgets.QWidget()
        _auto_h1_vbox = QtWidgets.QVBoxLayout(_auto_h1_container)
        _auto_h1_vbox.setContentsMargins(0, 0, 0, 0)
        _auto_h1_vbox.setSpacing(2)
        _auto_h1_vbox.addWidget(self.check_auto_h1)
        _auto_h1_vbox.addWidget(self._lbl_auto_h1_caption)
        form_geom.addRow("", _auto_h1_container)
        form_geom.addRow("Janela de investigação tj:", self.spin_tj)
        _auto_tj_container = QtWidgets.QWidget()
        _auto_tj_vbox = QtWidgets.QVBoxLayout(_auto_tj_container)
        _auto_tj_vbox.setContentsMargins(0, 0, 0, 0)
        _auto_tj_vbox.setSpacing(2)
        _auto_tj_vbox.addWidget(self.check_auto_tj)
        _auto_tj_vbox.addWidget(self._lbl_auto_tj_caption)
        form_geom.addRow("", _auto_tj_container)
        form_geom.addRow("Passo entre medidas p_med:", self.spin_pmed)

        layout_geom = QtWidgets.QVBoxLayout(grp_geom)
        layout_geom.addLayout(wrap_geom)

        # Sinais que invalidam n_pos → propagam para SimulatorPage
        self.spin_tj.valueChanged.connect(lambda _v: self.parametersChanged.emit())
        self.spin_pmed.valueChanged.connect(lambda _v: self.parametersChanged.emit())
        self.edit_angles.textChanged.connect(lambda _t: self.parametersChanged.emit())

        # Auto-geometria: recalcula h1/tj e atualiza estado de habilitação.
        self.check_auto_h1.toggled.connect(self._on_auto_geometry_toggled)
        self.check_auto_tj.toggled.connect(self._on_auto_geometry_toggled)

        # ═══ Grupo 2 — Perfil Pré-configurado ════════════════════════════
        # Droplist de modelos canônicos (Oklahoma 3/5/28, Devine 8, Viking
        # Graben 10, Padrão). Ao clicar "Aplicar perfil", a configuração
        # geológica do modelo canônico é aplicada, preservando a geometria
        # da ferramenta (h1, tj, p_med, freqs, TR, dip).
        grp_canonical = QtWidgets.QGroupBox("Perfil Pré-configurado")
        canonical_inner = QtWidgets.QVBoxLayout()

        from .sm_canonical_profiles import CANONICAL_PROFILE_LABELS

        self.combo_canonical = QtWidgets.QComboBox()
        for label, _key in CANONICAL_PROFILE_LABELS:
            self.combo_canonical.addItem(label)
        self.btn_apply_canonical = QtWidgets.QPushButton("Aplicar perfil")
        self.lbl_profile_info = QtWidgets.QLabel(
            "Selecione um perfil canônico da literatura e clique em "
            "<b>Aplicar perfil</b> para preencher as camadas."
        )
        self.lbl_profile_info.setWordWrap(True)
        self.lbl_profile_info.setStyleSheet("color: #888888; font-size: 11px;")

        _tooltip(
            self.combo_canonical,
            (
                "<b>Perfis canônicos da literatura</b><br/>"
                "Lista de modelos geológicos de referência validados contra Fortran:<br/>"
                "• <b>Padrão</b>: mantém a configuração atual (sem sobrescrever).<br/>"
                "• <b>Oklahoma 3/5/28</b>: benchmarks TIV clássicos (Anderson 2001).<br/>"
                "• <b>Devine 8</b>: modelo isotrópico (Kriegshäuser 2000).<br/>"
                "• <b>Viking Graben 10</b>: reservatório N. Sea (Eidesmo 2002).<br/><br/>"
                "Após selecionar, clique em <b>Aplicar perfil</b> para preencher "
                "as camadas (ρₕ, ρᵥ, espessuras). Parâmetros de ferramenta "
                "(h1, tj, p_med, freqs, TR, dip) <b>não</b> são sobrescritos."
            ),
        )
        _tooltip(
            self.btn_apply_canonical,
            (
                "<b>Aplicar perfil canônico selecionado</b><br/>"
                "Preenche o modelo geológico com os valores do perfil escolhido, "
                "desliga 'Geração de Modelos' e fixa <i>n_models</i> = 1. "
                "Os parâmetros da ferramenta (frequências, TR, dip, h1, tj, p_med) "
                "permanecem inalterados."
            ),
        )

        row_canon = QtWidgets.QHBoxLayout()
        row_canon.addWidget(self.combo_canonical, 1)
        row_canon.addWidget(self.btn_apply_canonical, 0)
        canonical_inner.addLayout(row_canon)
        canonical_inner.addWidget(self.lbl_profile_info)

        layout_canonical = QtWidgets.QVBoxLayout(grp_canonical)
        layout_canonical.addLayout(canonical_inner)

        # Atualiza o preview quando o usuário troca o item (NÃO aplica ainda)
        self.combo_canonical.currentTextChanged.connect(self._on_canonical_selected)
        self.btn_apply_canonical.clicked.connect(self._apply_canonical_profile)

        # ═══ Grupo 3 — Geração de Modelos (Aleatória | Manual) ═══════════
        # v2.4: radiobotões no topo permitem ao usuário escolher entre
        # geração aleatória (QMC/PRNG, fluxo atual) e entrada manual via
        # QTableWidget (LayersManualDialog em sm_layers_dialog.py).
        grp_gen = QtWidgets.QGroupBox("Geração de Modelos")
        form_gen, wrap_gen = _centered_form()

        # ── Radiobotões modo de geração (topo do grupo) ───────────────────
        self.radio_random = QtWidgets.QRadioButton("Aleatória (QMC / PRNG)")
        self.radio_manual = QtWidgets.QRadioButton("Manual (tabela de camadas)")
        self.radio_random.setChecked(True)
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.addButton(self.radio_random, 0)
        self._mode_group.addButton(self.radio_manual, 1)
        _tooltip(
            self.radio_random,
            (
                "<b>Geração aleatória</b><br/>"
                "Cada modelo tem alturas, ρₕ e λ sorteados segundo os parâmetros "
                "abaixo (gerador QMC/PRNG). Use para datasets estatísticos e ML."
            ),
        )
        _tooltip(
            self.radio_manual,
            (
                "<b>Geração manual</b><br/>"
                "Usuário define alturas, ρₕ e λ de cada camada em tabela "
                "editável. Ideal para reproduzir modelos canônicos "
                "(buildValidamodels.py) ou estudar casos específicos."
            ),
        )
        self.btn_edit_layers = QtWidgets.QPushButton("Editar tabela de camadas…")
        self.btn_edit_layers.setEnabled(False)  # habilitado quando Manual ativo
        _tooltip(
            self.btn_edit_layers,
            (
                "<b>Abrir tabela manual</b><br/>"
                "Abre uma janela com QTableWidget de 4 colunas (h, ρₕ, λ, ρᵥ=ρₕ·λ²). "
                "Quando um perfil canônico é aplicado, a tabela é pré-preenchida com "
                "os valores bit-exatos de buildValidamodels.py."
            ),
        )
        self.lbl_manual_status = QtWidgets.QLabel("<i>Nenhum perfil manual definido.</i>")
        self.lbl_manual_status.setStyleSheet("color: #a5a5a5;")

        row_mode = QtWidgets.QHBoxLayout()
        row_mode.addWidget(self.radio_random)
        row_mode.addWidget(self.radio_manual)
        row_mode.addStretch(1)

        row_manual_btn = QtWidgets.QHBoxLayout()
        row_manual_btn.addWidget(self.btn_edit_layers)
        row_manual_btn.addWidget(self.lbl_manual_status, 1)

        # Conectar signals (implementados mais abaixo na classe)
        self.radio_random.toggled.connect(self._on_generation_mode_toggled)
        self.btn_edit_layers.clicked.connect(self._on_edit_manual_layers)

        self.spin_nmodels = _spin_int(2000, 1, 10_000_000, 100)
        self.combo_generator = QtWidgets.QComboBox()
        for g in GENERATORS_AVAILABLE:
            self.combo_generator.addItem(g)
        self.combo_generator.setCurrentText("sobol")
        self.check_aniso = QtWidgets.QCheckBox("Perfil anisotrópico (TIV)")
        self.check_aniso.setChecked(True)
        self.spin_lambda_min = _spin_float(1.0, 1.0, 5.0, 0.05, 3)
        self.spin_lambda_max = _spin_float(1.4142, 1.0, 5.0, 0.05, 4)
        self.spin_rho_min = _spin_float(1.0, 0.001, 1e6, 1.0, 3, " Ω·m")
        self.spin_rho_max = _spin_float(1800.0, 0.001, 1e6, 10.0, 3, " Ω·m")
        self.combo_rho_distr = QtWidgets.QComboBox()
        self.combo_rho_distr.addItems(["loguni", "uniform"])
        self.spin_min_thick = _spin_float(0.5, 0.01, 100.0, 0.1, 2, " m")
        self.spin_nlayers_min = _spin_int(3, 3, 200)
        self.spin_nlayers_max = _spin_int(31, 3, 200)
        self.spin_nlayers_fixed = _spin_int(0, 0, 200)

        _tooltip(
            self.spin_nmodels,
            (
                "<b>Quantidade de modelos</b><br/>"
                "Total de perfis geológicos estocásticos a gerar. Cada perfil será "
                "simulado independentemente. Valores típicos: 100–30 000."
            ),
        )
        _tooltip(
            self.combo_generator,
            (
                "<b>Gerador de números aleatórios</b><br/>"
                "• <b>sobol</b> (★ padrão): QMC de baixa discrepância (scipy.qmc.Sobol), "
                "melhor cobertura espacial de perfis. Recomendado para ML.<br/>"
                "• <b>halton</b>: QMC base-prima.<br/>"
                "• <b>niederreiter</b>: aproximado via Halton.<br/>"
                "• <b>mersenne_twister</b>: PRNG clássico NumPy.<br/>"
                "• <b>uniform</b>: alias PRNG U[0,1].<br/>"
                "• <b>normal</b>: PRNG log-normal mapeado via CDF.<br/>"
                "• <b>box_muller</b>: transformação analítica U→N."
            ),
        )
        _tooltip(
            self.check_aniso,
            (
                "<b>Perfil anisotrópico (TIV)</b><br/>"
                "Se ativo, aplica ρᵥ = λ²·ρₕ com λ sorteado em [λ_min, λ_max]. "
                "Se inativo, força ρᵥ = ρₕ (perfil isotrópico)."
            ),
        )
        _tooltip(
            self.spin_lambda_min,
            (
                "<b>Fator de anisotropia λ mínimo</b><br/>"
                "Limite inferior de λ = √(ρᵥ/ρₕ). Deve ser ≥ 1.0 para respeitar "
                "a física TIV (ρᵥ ≥ ρₕ em camadas laminadas)."
            ),
        )
        _tooltip(
            self.spin_lambda_max,
            (
                "<b>Fator de anisotropia λ máximo</b><br/>"
                "Limite superior de λ. Valores típicos: 1.4 (fraca), 1.7 (moderada), "
                "2.0+ (forte). √2 ≈ 1.4142 é o padrão de benchmarks."
            ),
        )
        _tooltip(
            self.spin_rho_min,
            (
                "<b>Resistividade horizontal ρₕ mínima (Ω·m)</b><br/>"
                "Limite inferior da distribuição. Valores típicos: 0.3 Ω·m (água salgada), "
                "1–10 (folhelhos), 10–100 (arenitos saturados)."
            ),
        )
        _tooltip(
            self.spin_rho_max,
            (
                "<b>Resistividade horizontal ρₕ máxima (Ω·m)</b><br/>"
                "Limite superior. Valores típicos: 1 000 (HC convencional), 10 000+ (tight gas)."
            ),
        )
        _tooltip(
            self.combo_rho_distr,
            (
                "<b>Distribuição de ρₕ</b><br/>"
                "• <b>loguni</b> (padrão): log-uniforme em [min, max] — cobre várias "
                "ordens de magnitude igualmente.<br/>"
                "• <b>uniform</b>: uniforme linear — concentra massa nos valores altos."
            ),
        )
        _tooltip(
            self.spin_min_thick,
            (
                "<b>Espessura mínima por camada (m)</b><br/>"
                "Piso mínimo para cada camada interna, aplicado após o stick-breaking. "
                "Evita camadas degeneradas (<< λ/4 do filtro Hankel)."
            ),
        )
        _tooltip(
            self.spin_nlayers_min,
            (
                "<b>Número mínimo de camadas (inclui os 2 semi-espaços)</b><br/>"
                "Sorteio uniforme de n_camadas em [min, max] quando 'Nº camadas fixo' = 0."
            ),
        )
        _tooltip(
            self.spin_nlayers_max,
            (
                "<b>Número máximo de camadas</b><br/>"
                "Limite superior da amostragem uniforme de n_camadas."
            ),
        )
        _tooltip(
            self.spin_nlayers_fixed,
            (
                "<b>Nº de camadas fixo (0 = automático)</b><br/>"
                "Se ≥ 3, força todos os perfis a terem exatamente esse número de camadas. "
                "Se 0, sorteia uniformemente em [min, max]."
            ),
        )

        form_gen.addRow("Quantidade de modelos:", self.spin_nmodels)
        form_gen.addRow("Gerador aleatório:", self.combo_generator)
        form_gen.addRow("", self.check_aniso)
        form_gen.addRow("λ mínimo (TIV):", self.spin_lambda_min)
        form_gen.addRow("λ máximo (TIV):", self.spin_lambda_max)
        form_gen.addRow("ρₕ mínimo:", self.spin_rho_min)
        form_gen.addRow("ρₕ máximo:", self.spin_rho_max)
        form_gen.addRow("Distribuição de ρₕ:", self.combo_rho_distr)
        form_gen.addRow("Espessura mínima:", self.spin_min_thick)
        form_gen.addRow("Nº camadas mínimo:", self.spin_nlayers_min)
        form_gen.addRow("Nº camadas máximo:", self.spin_nlayers_max)
        form_gen.addRow("Nº camadas fixo:", self.spin_nlayers_fixed)

        layout_gen = QtWidgets.QVBoxLayout(grp_gen)
        layout_gen.addLayout(row_mode)
        layout_gen.addLayout(row_manual_btn)
        layout_gen.addWidget(_hsep())
        layout_gen.addLayout(wrap_gen)

        # Guarda widgets do modo "aleatório" para enable/disable em lote quando
        # o usuário chavear para "Manual" (E2).
        self._random_mode_widgets = [
            self.spin_nmodels,
            self.combo_generator,
            self.check_aniso,
            self.spin_lambda_min,
            self.spin_lambda_max,
            self.spin_rho_min,
            self.spin_rho_max,
            self.combo_rho_distr,
            self.spin_min_thick,
            self.spin_nlayers_min,
            self.spin_nlayers_max,
            self.spin_nlayers_fixed,
        ]

        # ═══ Grupo 3 — Filtro Hankel ═════════════════════════════════════
        grp_filt = QtWidgets.QGroupBox("Filtros de Hankel")
        form_filt, wrap_filt = _centered_form()
        self.combo_filter = QtWidgets.QComboBox()
        self.combo_filter.addItems(["werthmuller_201pt", "kong_61pt", "anderson_801pt"])
        _tooltip(
            self.combo_filter,
            (
                "<b>Filtro Hankel (transformada inversa J₀/J₁)</b><br/>"
                "Converte kernel espectral ↔ domínio espacial via quadratura log-espaçada.<br/>"
                "• <b>werthmuller_201pt</b> (★ padrão): precisão alta, 201 pontos.<br/>"
                "• <b>kong_61pt</b>: 3.3× mais rápido, precisão moderada (~1e-6).<br/>"
                "• <b>anderson_801pt</b>: precisão máxima (~1e-12), 4× mais lento."
            ),
        )
        form_filt.addRow("Filtro Hankel:", self.combo_filter)
        layout_filt = QtWidgets.QVBoxLayout(grp_filt)
        layout_filt.addLayout(wrap_filt)

        # ═══ Layout raiz ═════════════════════════════════════════════════
        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)
        root.addWidget(_heading("Parâmetros da Simulação"))
        root.addWidget(
            _heading(
                "Configure a geometria da ferramenta LWD, o filtro Hankel e os "
                "parâmetros de geração estocástica de perfis TIV. Passe o mouse "
                "sobre cada campo para ver uma explicação detalhada.",
                role="section",
            )
        )
        # Ordem (top → bottom) escolhida para que a "Perfil Pré-configurado"
        # apareça como PRIMEIRO passo — é o atalho mais comum; depois o
        # usuário ajusta Geometria da ferramenta e finamente a Geração
        # Estocástica, terminando na escolha do Filtro Hankel.
        root.addWidget(grp_canonical)
        root.addWidget(grp_geom)
        root.addWidget(grp_gen)
        root.addWidget(grp_filt)
        root.addStretch(1)

        container = QtWidgets.QWidget()
        container.setLayout(root)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(_frame_shape("NoFrame"))
        scroll.setWidget(container)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Perfis canônicos ──────────────────────────────────────────────────
    def _on_canonical_selected(self, label: str) -> None:
        """Atualiza apenas o preview informativo — NÃO aplica o perfil."""
        from .sm_canonical_profiles import format_profile_info, get_profile_by_label

        cm = get_profile_by_label(label)
        if cm is None:
            self.lbl_profile_info.setText(
                "Selecione um perfil canônico da literatura e clique em "
                "<b>Aplicar perfil</b> para preencher as camadas."
            )
        else:
            self.lbl_profile_info.setText(format_profile_info(cm))

    # ── Auto-geometria (fidelidade buildValidamodels.py) ──────────────────
    def _on_auto_geometry_toggled(self, _checked: bool = False) -> None:
        """Callback acionado quando check_auto_h1 ou check_auto_tj mudam.

        Desabilita os spinboxes correspondentes (h1/tj viram read-only quando
        auto-calculados) e recalcula os valores a partir do perfil canônico
        ativo. Se não houver perfil canônico aplicado, apenas atualiza o
        estado de habilitação — os valores atuais permanecem.
        """
        auto_h1 = bool(self.check_auto_h1.isChecked())
        auto_tj = bool(self.check_auto_tj.isChecked())

        # Spinbox desabilitado quando o valor é auto-calculado.
        self.spin_h1.setEnabled(not auto_h1)
        self.spin_tj.setEnabled(not auto_tj)

        # Recalcula imediatamente se houver perfil canônico aplicado.
        self._recompute_auto_geometry()

    def _recompute_auto_geometry(self) -> None:
        """Recalcula h1 e/ou tj conforme ``buildValidamodels.py``.

        Convenção (idêntica ao script de validação Fortran):

          ``tj_ref = max(Σesp de BUILDVALID_REFERENCE_KEYS) + 20 m``
                   = max(47.85, …) + 20 = 67.85 m  (janela global)

          ``h1 = (tj − Σesp_modelo) / 2``           (centraliza a estratigrafia)

        Se o modelo atual tem Σesp > max do batch de referência (ex.: Viking
        Graben 10 com 233 m), a janela expande para ``Σesp_current + margem``.

        Regras por combinação de checkboxes:

        * **Ambas ativas**: ``tj = tj_ref`` (global compartilhado entre
          modelos), ``h1 = (tj − Σesp) / 2``. Fidelidade máxima a
          ``buildValidamodels.py``.
        * **Só auto_tj**: ``tj = tj_ref`` (ignora h1 do usuário — a janela é
          uma decisão de batch, não do usuário).
        * **Só auto_h1**: ``h1 = (tj_atual − Σesp) / 2`` (usa o tj manual
          do usuário — útil quando o usuário define uma janela diferente).
        * **Nenhuma ativa**: no-op.
        * **Sem perfil canônico**: no-op (não adulterar valores manuais).

        Emite ``parametersChanged`` uma única vez no final para atualizar
        ``lbl_npos`` da SimulatorPage.
        """
        cm = self._canonical_override
        if cm is None:
            return

        auto_h1 = bool(self.check_auto_h1.isChecked())
        auto_tj = bool(self.check_auto_tj.isChecked())
        if not (auto_h1 or auto_tj):
            return

        try:
            sum_esp = sum(float(x) for x in cm.esp) if cm.esp is not None else 0.0
        except Exception:
            return
        if sum_esp <= 0.0:
            return

        from .sm_canonical_profiles import (
            compute_canonical_h1,
            compute_canonical_reference_tj,
        )

        # Bloqueia sinais durante a escrita para evitar feedback (spinbox →
        # parametersChanged → recompute). Emissão única ao final.
        blocker_h1 = QtCore.QSignalBlocker(self.spin_h1)
        blocker_tj = QtCore.QSignalBlocker(self.spin_tj)
        try:
            if auto_tj:
                # tj global (buildValidamodels.py): max(batch_max, Σesp_current)
                # + 20 m. Para modelos dentro do batch clássico, tj ≈ 67.85 m.
                tj_val = compute_canonical_reference_tj(current_esp_sum=sum_esp)
                self.spin_tj.setValue(float(tj_val))
            if auto_h1:
                # Lê o tj efetivo (pode ter acabado de ser setado acima).
                tj_effective = float(self.spin_tj.value())
                h1_val = compute_canonical_h1(tj_effective, sum_esp)
                self.spin_h1.setValue(float(h1_val))
        finally:
            del blocker_h1, blocker_tj

        try:
            self.parametersChanged.emit()
        except Exception:
            pass

    def _apply_canonical_profile(self) -> None:
        """Aplica o perfil canônico selecionado ao estado da página.

        Ações:
          1. Armazena o ``CanonicalModel`` em ``_canonical_override`` para
             que ``_start_simulation`` use o perfil fixo em vez da geração
             estocástica.
          2. **Populates campos de Geometria da Ferramenta** que dependem
             do perfil (apenas ``tj`` — ajusta se necessário para cobrir
             toda a estratigrafia + margem).
          3. **Populates campos de Geração Estocástica** com valores
             derivados do perfil canônico:
               * ``n_layers_fixed = cm.n_layers``
               * ``rho_min / rho_max`` = range real de ρₕ
               * ``min_thickness`` = menor espessura real
               * ``anisotropic`` = True se qualquer ρᵥ ≠ ρₕ
               * ``lambda_min / lambda_max`` = range real de √(ρᵥ/ρₕ)
               * ``n_models = 1`` (perfil fixo)
          4. **Preserva**: freqs, dips, TR, nf, nang, ntr, h1, p_med,
             filtro Hankel, gerador — são configurações da ferramenta,
             não do modelo geológico.
        """
        from .sm_canonical_profiles import get_profile_by_label

        label = self.combo_canonical.currentText()
        cm = get_profile_by_label(label)
        if cm is None:
            # "Padrão" — remove override, restaura geração estocástica livre.
            # Também desliga auto-geometria: sem Σesp canônico, os campos
            # h1/tj voltam ao controle manual do usuário. v2.4: também
            # remove override manual e volta radio para "Aleatória".
            self._canonical_override = None
            self.set_manual_layers(None)
            blocker_radio = QtCore.QSignalBlocker(self.radio_random)
            self.radio_random.setChecked(True)
            del blocker_radio
            self._on_generation_mode_toggled(False)
            try:
                self.check_auto_h1.setChecked(False)
                self.check_auto_tj.setChecked(False)
                self._on_auto_geometry_toggled()  # re-habilita spinboxes
            except Exception:
                pass
            self.lbl_profile_info.setText(
                "Configuração atual preservada (geração estocástica habilitada)."
            )
            return

        self._canonical_override = cm

        # v2.4: Pré-preenche o modo Manual com os valores bit-exatos de
        # buildValidamodels.py e chaveia o radiobutton para "Manual". O
        # usuário pode aceitar como está (rodar Numba/Fortran) ou editar
        # valores abrindo a tabela. Se preferir voltar ao modo aleatório,
        # basta clicar em "Aleatória" — nesse caso _manual_layers é zerado.
        manual_seed = {
            "n_layers": int(cm.n_layers),
            "thicknesses": [float(x) for x in cm.esp],
            "rho_h": [float(x) for x in cm.rho_h],
            "rho_v": [float(x) for x in cm.rho_v],
            "title": getattr(cm, "title", ""),
        }
        self.set_manual_layers(manual_seed)
        blocker_radio = QtCore.QSignalBlocker(self.radio_manual)
        self.radio_manual.setChecked(True)
        del blocker_radio
        # Força a atualização do estado enable/disable dos widgets QMC
        self._on_generation_mode_toggled(True)

        # Coleta estatísticas derivadas do perfil geológico.
        rho_h = [float(x) for x in cm.rho_h]
        rho_v = [float(x) for x in cm.rho_v]
        esp = [float(x) for x in cm.esp]
        sum_esp = sum(esp) if esp else 0.0

        rho_min_val = min(rho_h) if rho_h else 1.0
        rho_max_val = max(rho_h) if rho_h else 1000.0

        # Menor espessura interna — evita zero/negativo.
        thick_min_val = max(1e-3, min(esp)) if esp else 0.5

        # Anisotropia: ativa se qualquer ρᵥ difere de ρₕ (tolerância 1e-9 em razão)
        aniso_active = False
        lambda_values: List[float] = []
        for rh, rv in zip(rho_h, rho_v):
            if rh > 0.0:
                ratio = max(rv / rh, 1.0)  # clamp para respeitar TIV ρᵥ≥ρₕ
                lam = float(ratio) ** 0.5
                lambda_values.append(lam)
                if abs(rv - rh) > 1e-9 * max(rh, 1.0):
                    aniso_active = True
        if lambda_values:
            lam_min_val = max(1.0, min(lambda_values))
            lam_max_val = max(lam_min_val, max(lambda_values))
        else:
            lam_min_val, lam_max_val = 1.0, 1.0

        # ── Geometria da Ferramenta — Auto (fidelidade buildValidamodels.py) ──
        # Ao aplicar um perfil canônico, ativamos AMBOS os checkboxes de
        # auto-geometria (h1 e tj derivados da Σesp do modelo). Isso garante
        # que o Simulator Manager reproduza exatamente a convenção de
        # ``buildValidamodels.py``:
        #   h1 = max(5.0, 0.20·Σesp)   (margem simétrica proporcional)
        #   tj = Σesp + 2·h1           (janela cobre toda a estratigrafia)
        # O usuário pode desligar qualquer checkbox depois, re-habilitando
        # o campo manual correspondente.
        try:
            blocker_h1 = QtCore.QSignalBlocker(self.check_auto_h1)
            blocker_tj = QtCore.QSignalBlocker(self.check_auto_tj)
            try:
                self.check_auto_h1.setChecked(True)
                self.check_auto_tj.setChecked(True)
            finally:
                del blocker_h1, blocker_tj
            # Atualiza estado de habilitação (spinboxes desabilitados quando
            # auto). Também dispara o recálculo via _recompute_auto_geometry.
            self._on_auto_geometry_toggled()
        except Exception:
            # Fallback: se algo falhar na cadeia Qt, aplica manualmente a
            # fórmula canônica preservando o comportamento esperado.
            h1_val = max(5.0, round(sum_esp * 0.20, 3))
            tj_val = round(sum_esp + 2.0 * h1_val, 3)
            try:
                self.spin_h1.setValue(float(h1_val))
                self.spin_tj.setValue(float(tj_val))
            except Exception:
                pass

        # ── Geração Estocástica ──────────────────────────────────────────
        _setters = [
            (self.spin_nlayers_fixed, int(cm.n_layers)),
            (
                self.spin_nlayers_min,
                min(int(cm.n_layers), self.spin_nlayers_min.maximum()),
            ),
            (self.spin_nlayers_max, max(int(cm.n_layers), self.spin_nlayers_min.value())),
            (self.spin_rho_min, float(rho_min_val)),
            (self.spin_rho_max, float(rho_max_val)),
            (self.spin_min_thick, float(thick_min_val)),
            (self.spin_lambda_min, float(lam_min_val)),
            (self.spin_lambda_max, float(lam_max_val)),
            (self.spin_nmodels, 1),
        ]
        for widget, value in _setters:
            try:
                widget.setValue(value)
            except Exception:
                pass

        try:
            self.check_aniso.setChecked(bool(aniso_active))
        except Exception:
            pass

        # Emite parametersChanged para que SimulatorPage.lbl_npos recalcule
        # (tj pode ter mudado acima).
        try:
            self.parametersChanged.emit()
        except Exception:
            pass

        self.lbl_profile_info.setText(
            f"<b>Aplicado:</b> {cm.title} — {int(cm.n_layers)} camadas. "
            f"ρₕ ∈ [{rho_min_val:g}, {rho_max_val:g}] Ω·m · "
            f"λ ∈ [{lam_min_val:.2f}, {lam_max_val:.2f}] · "
            f"min_thick={thick_min_val:g} m. "
            f"Geração estocástica fixada em 1 modelo (perfil determinístico)."
        )

    def has_canonical_override(self) -> bool:
        """True se um perfil canônico está ativo (não é o 'Padrão')."""
        return self._canonical_override is not None

    def get_canonical_override(self):  # -> Optional[CanonicalModel]
        """Retorna o CanonicalModel atualmente aplicado, ou None."""
        return self._canonical_override

    def clear_canonical_override(self) -> None:
        """Remove o perfil canônico ativo (útil ao carregar um .exp.json)."""
        self._canonical_override = None

    # ── Geração de Modelos — modo aleatório × manual (v2.4) ───────────────
    def _on_generation_mode_toggled(self, _checked: bool = False) -> None:
        """Callback disparado ao trocar entre Aleatória e Manual.

        Modo **Aleatória**: habilita todos os widgets QMC/PRNG, desabilita
        o botão "Editar tabela…", zera ``self._manual_layers``.
        Modo **Manual**: desabilita widgets QMC, habilita "Editar tabela…".
        O ``spin_nmodels`` permanece habilitado em ambos os modos (define
        quantas réplicas do modelo manual/canônico rodar em batch).
        """
        manual = bool(self.radio_manual.isChecked())
        # spin_nmodels fica habilitado em ambos os modos
        for w in self._random_mode_widgets:
            if w is self.spin_nmodels:
                continue
            w.setEnabled(not manual)
        self.btn_edit_layers.setEnabled(manual)
        if not manual:
            # Voltou para Aleatória: descartar override manual
            self._manual_layers = None
            self.lbl_manual_status.setText("<i>Nenhum perfil manual definido.</i>")
        else:
            if self._manual_layers is None:
                self.lbl_manual_status.setText(
                    "<i>Clique em 'Editar tabela…' para definir o modelo.</i>"
                )
            else:
                n = int(self._manual_layers.get("n_layers", 0))
                self.lbl_manual_status.setText(
                    f"<b>Perfil manual definido</b> — {n} camadas."
                )

    def _on_edit_manual_layers(self) -> None:
        """Abre o ``LayersManualDialog`` para edição da tabela de camadas.

        Se houver override canônico ativo ou um ``self._manual_layers``
        prévio, o dialog é pré-preenchido com esses valores. Ao aceitar, o
        dict resultante é gravado em ``self._manual_layers`` e o status é
        atualizado.
        """
        from .sm_layers_dialog import LayersManualDialog

        initial: Optional[Dict[str, Any]] = None
        if self._manual_layers is not None:
            initial = dict(self._manual_layers)
        elif self._canonical_override is not None:
            cm = self._canonical_override
            initial = {
                "n_layers": int(cm.n_layers),
                "thicknesses": [float(x) for x in cm.esp],
                "rho_h": [float(x) for x in cm.rho_h],
                "rho_v": [float(x) for x in cm.rho_v],
                "title": getattr(cm, "title", ""),
            }

        dlg = LayersManualDialog(self, initial=initial)
        if (
            dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted
            if hasattr(QtWidgets.QDialog, "DialogCode")
            else dlg.exec()
        ):
            layers = dlg.get_layers()
            if layers is None:
                return
            self._manual_layers = layers
            n = int(layers.get("n_layers", 0))
            self.lbl_manual_status.setText(
                f"<b>Perfil manual definido</b> — {n} camadas."
            )
            # Sincroniza spin_nlayers_fixed para refletir a escolha manual
            blocker = QtCore.QSignalBlocker(self.spin_nlayers_fixed)
            self.spin_nlayers_fixed.setValue(n)
            del blocker

    def has_manual_override(self) -> bool:
        """True se ``radio_manual`` está ativo E ``_manual_layers`` preenchido."""
        return bool(self.radio_manual.isChecked()) and self._manual_layers is not None

    def get_manual_override(self) -> Optional[Dict[str, Any]]:
        """Retorna o dict ``{n_layers, thicknesses, rho_h, rho_v}`` ou None."""
        if not self.has_manual_override():
            return None
        return dict(self._manual_layers or {})

    def set_manual_layers(self, layers: Optional[Dict[str, Any]]) -> None:
        """Setter programático (usado em ``_apply_canonical_profile`` e restore)."""
        self._manual_layers = {k: v for k, v in layers.items()} if layers else None
        if layers is None:
            self.lbl_manual_status.setText("<i>Nenhum perfil manual definido.</i>")
        else:
            n = int(layers.get("n_layers", 0))
            self.lbl_manual_status.setText(
                f"<b>Perfil manual definido</b> — {n} camadas."
            )

    # API pública
    def collect_freqs(self) -> List[float]:
        return _parse_float_list(self.edit_freqs.text(), [20000.0])[
            : self.spin_nf.value()
        ]

    def collect_angles(self) -> List[float]:
        return _parse_float_list(self.edit_angles.text(), [0.0])[: self.spin_nang.value()]

    def collect_trs(self) -> List[float]:
        return _parse_float_list(self.edit_trs.text(), [1.0])[: self.spin_ntr.value()]

    def build_gen_config(self) -> GenConfig:
        fixed = self.spin_nlayers_fixed.value()
        return GenConfig(
            total_depth=float(self.spin_tj.value()),
            n_layers_min=int(self.spin_nlayers_min.value()),
            n_layers_max=int(self.spin_nlayers_max.value()) + 1,
            n_layers_fixed=int(fixed) if fixed >= 3 else None,
            rho_h_min=float(self.spin_rho_min.value()),
            rho_h_max=float(self.spin_rho_max.value()),
            rho_h_distribution=self.combo_rho_distr.currentText(),
            anisotropic=bool(self.check_aniso.isChecked()),
            lambda_min=float(self.spin_lambda_min.value()),
            lambda_max=float(self.spin_lambda_max.value()),
            min_thickness=float(self.spin_min_thick.value()),
            generator=self.combo_generator.currentText(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nf": self.spin_nf.value(),
            "nang": self.spin_nang.value(),
            "ntr": self.spin_ntr.value(),
            "freqs": self.edit_freqs.text(),
            "angles": self.edit_angles.text(),
            "trs": self.edit_trs.text(),
            "h1": self.spin_h1.value(),
            "tj": self.spin_tj.value(),
            "auto_h1": self.check_auto_h1.isChecked(),
            "auto_tj": self.check_auto_tj.isChecked(),
            "p_med": self.spin_pmed.value(),
            "n_models": self.spin_nmodels.value(),
            "generator": self.combo_generator.currentText(),
            "aniso": self.check_aniso.isChecked(),
            "lambda_min": self.spin_lambda_min.value(),
            "lambda_max": self.spin_lambda_max.value(),
            "rho_min": self.spin_rho_min.value(),
            "rho_max": self.spin_rho_max.value(),
            "rho_distr": self.combo_rho_distr.currentText(),
            "min_thick": self.spin_min_thick.value(),
            "nlayers_min": self.spin_nlayers_min.value(),
            "nlayers_max": self.spin_nlayers_max.value(),
            "nlayers_fixed": self.spin_nlayers_fixed.value(),
            "filter": self.combo_filter.currentText(),
        }

    def from_dict(self, d: Dict[str, Any]) -> None:
        if not d:
            return
        try:
            self.spin_nf.setValue(int(d.get("nf", 1)))
            self.spin_nang.setValue(int(d.get("nang", 1)))
            self.spin_ntr.setValue(int(d.get("ntr", 1)))
            self.edit_freqs.setText(str(d.get("freqs", "20000")))
            self.edit_angles.setText(str(d.get("angles", "0")))
            self.edit_trs.setText(str(d.get("trs", "1.0")))
            self.spin_h1.setValue(float(d.get("h1", 10.0)))
            self.spin_tj.setValue(float(d.get("tj", 120.0)))
            self.spin_pmed.setValue(float(d.get("p_med", 0.2)))
            # Estado das checkboxes de auto-geometria; o setChecked dispara
            # o toggled → _on_auto_geometry_toggled → atualiza habilitação.
            self.check_auto_h1.setChecked(bool(d.get("auto_h1", False)))
            self.check_auto_tj.setChecked(bool(d.get("auto_tj", False)))
            self.spin_nmodels.setValue(int(d.get("n_models", 2000)))
            self.combo_generator.setCurrentText(d.get("generator", "sobol"))
            self.check_aniso.setChecked(bool(d.get("aniso", True)))
            self.spin_lambda_min.setValue(float(d.get("lambda_min", 1.0)))
            self.spin_lambda_max.setValue(float(d.get("lambda_max", 1.4142)))
            self.spin_rho_min.setValue(float(d.get("rho_min", 1.0)))
            self.spin_rho_max.setValue(float(d.get("rho_max", 1800.0)))
            self.combo_rho_distr.setCurrentText(d.get("rho_distr", "loguni"))
            self.spin_min_thick.setValue(float(d.get("min_thick", 0.5)))
            self.spin_nlayers_min.setValue(int(d.get("nlayers_min", 3)))
            self.spin_nlayers_max.setValue(int(d.get("nlayers_max", 31)))
            self.spin_nlayers_fixed.setValue(int(d.get("nlayers_fixed", 0)))
            self.combo_filter.setCurrentText(d.get("filter", "werthmuller_201pt"))
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# Aba 2 — Simulador (merged save checkbox + fix stop/start)
# ══════════════════════════════════════════════════════════════════════════


class SimulatorPage(QtWidgets.QWidget):
    """Página de escolha do backend, paralelismo e execução."""

    request_start = Signal()
    request_stop = Signal()
    request_clear_simulations = Signal()  # v2.4 — botão "Limpar Simulações"
    request_history_select = Signal(str)  # v2.4 — snapshot_id (clique simples)
    request_history_open = Signal(str)  # v2.4 — snapshot_id (duplo-clique)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        ncpu = os.cpu_count() or 8

        grp_backend = QtWidgets.QGroupBox("Backend")
        form_bk = QtWidgets.QFormLayout(grp_backend)
        form_bk.setLabelAlignment(ALIGN_RIGHT)
        self.combo_backend = QtWidgets.QComboBox()
        self.combo_backend.addItems(["numba", "fortran"])
        _tooltip(
            self.combo_backend,
            (
                "<b>Backend do simulador</b><br/>"
                "• <b>numba</b>: Python JIT-compilado com AOT cache. Mais rápido em lote grande "
                "(warmup único por worker). Exporta .dat/.out.<br/>"
                "• <b>fortran</b>: tatu.x legado OpenMP. Cada modelo reescreve model.in + subprocess."
            ),
        )
        self.lbl_fortran_path = QtWidgets.QLabel("(definido em Preferências)")
        self.lbl_python_path = QtWidgets.QLabel(sys.executable)
        for lbl in (self.lbl_fortran_path, self.lbl_python_path):
            lbl.setStyleSheet("color:#a5a5a5; font-family:Menlo, Consolas, monospace;")
        _tooltip(
            self.lbl_fortran_path,
            (
                "<b>Binário Fortran tatu.x</b><br/>"
                "Caminho absoluto do executável ``tatu.x`` usado quando o backend é "
                "'fortran'. Se o campo mostrar '(indefinido)', configure em Preferências → "
                "'tatu.x (Fortran)'."
            ),
        )
        _tooltip(
            self.lbl_python_path,
            (
                "<b>Interpretador Python</b><br/>"
                "Executável Python usado pelos subprocessos do ProcessPoolExecutor. "
                "Deve ter <code>geosteering_ai</code>, <code>numba</code> e <code>numpy</code> "
                "instalados no mesmo ambiente."
            ),
        )
        form_bk.addRow("Simulador selecionado:", self.combo_backend)
        form_bk.addRow("Binário Fortran (tatu.x):", self.lbl_fortran_path)
        form_bk.addRow("Interpretador Python:", self.lbl_python_path)

        grp_par = QtWidgets.QGroupBox(
            "Paralelismo — Central Master-Plan · Parallel Execution"
        )
        par_form = QtWidgets.QFormLayout(grp_par)
        par_form.setLabelAlignment(ALIGN_RIGHT)
        self.spin_workers = _spin_int(max(1, ncpu // 4), 1, 256)
        self.spin_threads = _spin_int(max(2, ncpu // max(1, ncpu // 4)), 1, 256)
        _tooltip(
            self.spin_workers,
            (
                "<b>Nº de workers sandbox</b><br/>"
                "Processos isolados executados em paralelo (ProcessPoolExecutor). Cada worker "
                "tem seu próprio warmup JIT (Numba) ou cópia do tatu.x (Fortran)."
            ),
        )
        _tooltip(
            self.spin_threads,
            (
                "<b>Nº de threads por worker</b><br/>"
                "Define NUMBA_NUM_THREADS/OMP_NUM_THREADS dentro do subprocesso. "
                "Workers × Threads ≤ nº de CPU lógicas recomendado."
            ),
        )
        self.lbl_ncpu = QtWidgets.QLabel(
            f"CPU cores disponíveis: {ncpu}   |   Sistema: {platform.system()} {platform.release()}"
        )
        _tooltip(
            self.lbl_ncpu,
            (
                "<b>Recursos do sistema</b><br/>"
                "CPUs lógicas detectadas via <code>os.cpu_count()</code> + sistema operacional. "
                "Use como referência para dimensionar Workers × Threads."
            ),
        )
        par_form.addRow("Nº de workers sandbox:", self.spin_workers)
        par_form.addRow("Nº de threads por worker:", self.spin_threads)
        par_form.addRow("", self.lbl_ncpu)

        grp_out = QtWidgets.QGroupBox("Saída")
        out_form = QtWidgets.QFormLayout(grp_out)
        out_form.setLabelAlignment(ALIGN_RIGHT)
        self.edit_output_dir = _qline("", "Diretório de saída")
        _tooltip(
            self.edit_output_dir,
            (
                "<b>Diretório de saída</b><br/>"
                "Pasta onde serão salvos artefatos (.dat, .out, CSV de benchmark, figuras). "
                "Criada automaticamente se não existir."
            ),
        )
        self.btn_browse_out = QtWidgets.QPushButton("Procurar…")
        self.btn_browse_out.clicked.connect(self._browse_out)
        _tooltip(
            self.btn_browse_out,
            "Abrir seletor de pastas do sistema para definir o diretório de saída.",
        )
        row_out = QtWidgets.QHBoxLayout()
        row_out.addWidget(self.edit_output_dir, 1)
        row_out.addWidget(self.btn_browse_out)
        wrap_out = QtWidgets.QWidget()
        wrap_out.setLayout(row_out)

        # ── Checkbox única (.dat + .out) ────────────────────────────────
        self.check_save_artifacts = QtWidgets.QCheckBox(
            "Salvar artefatos Fortran-compat (.dat binário 22-col + .out ASCII)"
        )
        self.check_save_artifacts.setChecked(True)
        _tooltip(
            self.check_save_artifacts,
            (
                "<b>Salvar .dat + .out</b><br/>"
                "Por convenção do Geosteering AI, o .dat (binário 22 colunas) só é "
                "útil quando acompanhado do .out (metadados ASCII homônimo). Esta "
                "opção salva os dois arquivos juntos. Ativada por padrão."
            ),
        )
        out_form.addRow("Diretório:", wrap_out)
        out_form.addRow("", self.check_save_artifacts)

        grp_exec = QtWidgets.QGroupBox("Execução")
        exec_row = QtWidgets.QHBoxLayout(grp_exec)
        self.btn_start = QtWidgets.QPushButton("▶  Iniciar Simulação")
        self.btn_start.setProperty("role", "primary")
        _tooltip(
            self.btn_start,
            (
                "<b>Iniciar simulação</b><br/>"
                "Gera os modelos estocásticos (conforme 'Geração Estocástica') e despacha "
                "para os workers sandbox. Progresso exibido em tempo real."
            ),
        )
        self.btn_stop = QtWidgets.QPushButton("■  Parar")
        self.btn_stop.setProperty("role", "danger")
        self.btn_stop.setEnabled(False)
        _tooltip(
            self.btn_stop,
            (
                "<b>Parar simulação</b><br/>"
                "Solicita parada cooperativa entre chunks. O worker atual termina sua batch "
                "antes de liberar o processo. Após parar, 'Iniciar Simulação' fica disponível."
            ),
        )
        # v2.4: bot\u00e3o "Limpar Simula\u00e7\u00f5es" \u2014 apaga hist\u00f3rico do experimento
        # com confirma\u00e7\u00e3o via QMessageBox (handler no MainWindow).
        self.btn_clear_sims = QtWidgets.QPushButton("🗑  Limpar Simulações")
        self.btn_clear_sims.setProperty("role", "warning")
        _tooltip(
            self.btn_clear_sims,
            (
                "<b>Limpar simulações do experimento</b><br/>"
                "Remove todos os snapshots do histórico (e os tensores H em "
                "cache de sessão). A ação pede confirmação antes de executar "
                "e não pode ser desfeita."
            ),
        )
        self.btn_clear_sims.clicked.connect(self.request_clear_simulations.emit)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        _tooltip(
            self.progress, "Progresso da simulação em % (modelos concluídos / total)."
        )
        self.lbl_throughput = QtWidgets.QLabel("Throughput: — mod/h")
        self.lbl_throughput.setStyleSheet("font-weight:600; color:#ffffff;")
        _tooltip(
            self.lbl_throughput,
            (
                "<b>Throughput atual</b><br/>"
                "Modelos simulados por hora, calculado como (done / elapsed_s) × 3600. "
                "Valores típicos: 60k–1M mod/h conforme backend e paralelismo."
            ),
        )
        exec_row.addWidget(self.btn_start)
        exec_row.addWidget(self.btn_stop)
        exec_row.addWidget(self.btn_clear_sims)
        exec_row.addWidget(self.progress, 1)
        exec_row.addWidget(self.lbl_throughput)
        self.btn_start.clicked.connect(self.request_start.emit)
        self.btn_stop.clicked.connect(self.request_stop.emit)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Log de execução aparecerá aqui.")
        self.log_view.setMaximumBlockCount(5000)
        _tooltip(
            self.log_view,
            (
                "<b>Log de execução</b><br/>"
                "Mensagens informativas, avisos e erros da simulação + workers. "
                "Limite de ~5000 linhas (rotação automática)."
            ),
        )

        # ── Resumo derivado (n_pos por modelo) ──────────────────────────
        # Mostra a quantidade de pontos de medição por amostra geológica,
        # calculada como ⌈tj / (p_med · cos(dip₀))⌉. Atualizada em tempo
        # real por MainWindow via connect(parametersChanged, update_npos).
        self.lbl_npos = QtWidgets.QLabel("Pontos de medição por modelo: —")
        self.lbl_npos.setStyleSheet(
            "font-weight: 600; color: #4ec9b0; "
            "padding: 6px 10px; background: #252526; "
            "border-left: 3px solid #007acc; border-radius: 2px;"
        )
        _tooltip(
            self.lbl_npos,
            (
                "<b>Pontos de medição por modelo (n_pos)</b><br/>"
                "Calculado como ⌈tj / (p_med · cos(dip₀))⌉ onde:<br/>"
                "• <b>tj</b>: janela de investigação (m)<br/>"
                "• <b>p_med</b>: passo entre medidas (m)<br/>"
                "• <b>dip₀</b>: primeiro ângulo de dip da lista<br/><br/>"
                "Este é o tamanho do eixo de posições (n_pos) do tensor H de saída. "
                "Alterações em tj, p_med ou ângulos de dip na aba Parâmetros "
                "atualizam este valor em tempo real."
            ),
        )

        # ── v2.4: Histórico de simulações (lista + painel de parâmetros) ──
        # Cada "Iniciar Simulação" registra um SimulationSnapshot no
        # ExperimentState e aparece aqui como item clicável. Clique simples
        # mostra os parâmetros no painel lateral; duplo-clique recarrega o
        # tensor H (se ainda em cache) em ResultsPage.
        grp_history = QtWidgets.QGroupBox("Simulações Realizadas")
        hist_layout = QtWidgets.QVBoxLayout(grp_history)
        hist_layout.setContentsMargins(8, 8, 8, 8)
        self.sim_history_list = QtWidgets.QListWidget()
        self.sim_history_list.setAlternatingRowColors(True)
        self.sim_history_list.setMinimumHeight(120)
        _tooltip(
            self.sim_history_list,
            (
                "<b>Histórico de simulações do experimento</b><br/>"
                "Cada execução de 'Iniciar Simulação' adiciona uma entrada "
                "aqui. Clique para ver parâmetros ao lado; duplo-clique "
                "para reabrir o tensor H na aba Resultados."
            ),
        )
        self.sim_history_info = QtWidgets.QTextEdit()
        self.sim_history_info.setReadOnly(True)
        self.sim_history_info.setPlaceholderText(
            "Selecione uma simulação acima para ver seus parâmetros."
        )
        self.sim_history_info.setMinimumHeight(120)
        _tooltip(
            self.sim_history_info,
            (
                "<b>Parâmetros da simulação selecionada</b><br/>"
                "Snapshot completo de backend, workers, threads, geometria "
                "(h1, tj, p_med), frequências, TR, dip, Σesp e artefatos "
                "exportados (.dat/.out)."
            ),
        )
        hist_splitter = QtWidgets.QSplitter(ORIENT_H)
        hist_splitter.addWidget(self.sim_history_list)
        hist_splitter.addWidget(self.sim_history_info)
        hist_splitter.setStretchFactor(0, 2)
        hist_splitter.setStretchFactor(1, 5)
        hist_layout.addWidget(hist_splitter)
        # Bridge signals → MainWindow handlers
        self.sim_history_list.currentItemChanged.connect(self._on_history_item_changed)
        self.sim_history_list.itemDoubleClicked.connect(
            self._on_history_item_doubleclicked
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)
        root.addWidget(_heading("Simulador — Execução Paralela"))
        root.addWidget(
            _heading(
                "Selecione o backend, defina o paralelismo por workers sandbox e "
                "acompanhe o throughput em tempo real.",
                role="section",
            )
        )
        root.addWidget(self.lbl_npos)
        root.addWidget(grp_backend)
        root.addWidget(grp_par)
        root.addWidget(grp_out)
        root.addWidget(grp_exec)
        # v2.4b: ``grp_history`` permanece criado aqui (mantém signals/m\u00e9todos)
        # mas o SimulatorTab pode extra\u00ed-lo para uma 3\u00aa coluna dedicada via
        # ``detach_history_group()`` — nesse caso, o grupo N\u00c3O \u00e9 adicionado
        # ao layout principal desta p\u00e1gina.
        self.grp_history = grp_history
        self._history_in_root = False  # marcado True se adicionado abaixo
        # Log view sempre permanece nesta coluna (meio):
        root.addWidget(self.log_view, 1)

    # ── v2.4b: API para SimulatorTab reparentar o grupo de hist\u00f3rico ──
    def take_history_group(self) -> QtWidgets.QGroupBox:
        """Retorna o ``grp_history`` (remove do layout atual se estiver).

        Usado por :class:`SimulatorTab` para mover o grupo para a 3\u00aa coluna
        do splitter horizontal (layout de 3 colunas v2.4b). O widget
        continua sendo de propriedade desta p\u00e1gina (signals/m\u00e9todos
        permanecem acess\u00edveis via ``self.grp_history``, ``self.sim_history_list``
        e ``self.sim_history_info``).
        """
        if self.grp_history.parent() is not None:
            self.grp_history.setParent(None)
        self._history_in_root = False
        return self.grp_history

    # ── v2.4: Hist\u00f3rico de simula\u00e7\u00f5es ───────────────────────────────────
    def _on_history_item_changed(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        _previous: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        """Clique simples — emite snapshot_id para o MainWindow."""
        if current is None:
            return
        snap_id = (
            current.data(QtCore.Qt.ItemDataRole.UserRole)
            if hasattr(QtCore.Qt, "ItemDataRole")
            else current.data(QtCore.Qt.UserRole)
        )
        if snap_id:
            self.request_history_select.emit(str(snap_id))

    def _on_history_item_doubleclicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Duplo-clique — emite snapshot_id para reabrir em Resultados."""
        snap_id = (
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            if hasattr(QtCore.Qt, "ItemDataRole")
            else item.data(QtCore.Qt.UserRole)
        )
        if snap_id:
            self.request_history_open.emit(str(snap_id))

    def add_history_entry(self, snapshot_id: str, label: str, *, in_cache: bool) -> None:
        """Adiciona (ou atualiza) item no QListWidget de histórico."""
        item = QtWidgets.QListWidgetItem(("● " if in_cache else "○ ") + label)
        role = (
            QtCore.Qt.ItemDataRole.UserRole
            if hasattr(QtCore.Qt, "ItemDataRole")
            else QtCore.Qt.UserRole
        )
        item.setData(role, snapshot_id)
        # Tooltip com indicativo de cache
        item.setToolTip(
            f"snapshot_id: {snapshot_id}\n"
            + (
                "Tensor H disponível em cache (duplo-clique reabre em Resultados)."
                if in_cache
                else "Tensor não em cache — duplo-clique mostrará aviso de re-execução."
            )
        )
        self.sim_history_list.addItem(item)

    def clear_history_list(self) -> None:
        """Limpa a lista do histórico na UI (sem tocar no ExperimentState)."""
        self.sim_history_list.clear()
        self.sim_history_info.clear()

    def set_history_info(self, html: str) -> None:
        """Atualiza o painel lateral de parâmetros do snapshot selecionado."""
        self.sim_history_info.setHtml(html)

    def _browse_out(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Selecionar diretório de saída",
            self.edit_output_dir.text() or str(Path.cwd()),
        )
        if path:
            self.edit_output_dir.setText(path)

    def set_fortran_path(self, path: str) -> None:
        self.lbl_fortran_path.setText(path or "(indefinido)")

    def set_python_path(self, path: str) -> None:
        self.lbl_python_path.setText(path or sys.executable)

    def set_output_dir(self, path: str) -> None:
        if path and not self.edit_output_dir.text():
            self.edit_output_dir.setText(path)

    def append_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def set_running(self, running: bool) -> None:
        """Atualiza o estado dos botões Start/Stop.

        CORREÇÃO: ``running=False`` SEMPRE reabilita Start. Garantia de que
        após Stop (ou erro) o usuário possa reiniciar sem travar a UI.
        """
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def update_npos_from_params(self, params_page: "ParametersPage") -> None:
        """Recalcula e atualiza o label ``lbl_npos`` a partir dos parâmetros.

        Fórmula: ``n_pos = ⌈tj / (p_med · cos(dip₀))⌉``. Invocado pela
        ``MainWindow`` ao conectar ``ParametersPage.parametersChanged``.
        """
        import math

        try:
            tj = float(params_page.spin_tj.value())
            p_med = float(params_page.spin_pmed.value())
            dips = params_page.collect_angles() or [0.0]
            dip0 = float(dips[0])
            cos_d = max(1e-6, math.cos(math.radians(abs(dip0))))
            n_pos = max(1, int(math.ceil(tj / (p_med * cos_d))))
            self.lbl_npos.setText(
                f"Pontos de medição por modelo: <b>{n_pos}</b>  "
                f"<span style='color:#888'>(tj={tj:g} m, p_med={p_med:g} m, "
                f"dip₀={dip0:g}° → ⌈{tj:g} / ({p_med:g}·cos({dip0:g}°))⌉)</span>"
            )
        except Exception:
            self.lbl_npos.setText("Pontos de medição por modelo: —")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.combo_backend.currentText(),
            "workers": self.spin_workers.value(),
            "threads": self.spin_threads.value(),
            "output_dir": self.edit_output_dir.text(),
            "save_artifacts": self.check_save_artifacts.isChecked(),
        }

    def from_dict(self, d: Dict[str, Any]) -> None:
        if not d:
            return
        try:
            self.combo_backend.setCurrentText(d.get("backend", "numba"))
            self.spin_workers.setValue(int(d.get("workers", 2)))
            self.spin_threads.setValue(int(d.get("threads", 4)))
            self.edit_output_dir.setText(str(d.get("output_dir", "")))
            self.check_save_artifacts.setChecked(bool(d.get("save_artifacts", True)))
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# Aba 3 — Benchmark (+ Config C uniforme + Config D per-model dialog)
# ══════════════════════════════════════════════════════════════════════════


class ConfigDRowEditor(QtWidgets.QDialog):
    """Editor de parâmetros canônicos per-model (uma linha de Config D)."""

    def __init__(
        self,
        model_name: str,
        current: Dict[str, Any],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Parâmetros canônicos — {model_name}")
        self.setMinimumWidth(520)
        self._model = model_name
        self._result: Optional[Dict[str, Any]] = None

        default = default_canonical_config(model_name)
        eff = {**default, **(current or {})}

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(ALIGN_RIGHT)
        form.setHorizontalSpacing(12)

        self.edit_freqs = _qline(
            ", ".join(f"{f:g}" for f in eff["freqs_hz"]),
            "Ex.: 20000, 40000",
        )
        self.edit_trs = _qline(
            ", ".join(f"{t:g}" for t in eff["tr_list_m"]),
            "Ex.: 1.0, 8.19, 20.43",
        )
        self.edit_dips = _qline(
            ", ".join(f"{d:g}" for d in eff["dip_list_deg"]),
            "Ex.: 0, 30",
        )
        self.spin_h1 = _spin_float(float(eff["h1"]), 0.01, 500.0, 0.5, 4, " m")
        self.spin_pmed = _spin_float(float(eff["p_med"]), 0.01, 10.0, 0.01, 3, " m")

        for w, tip in [
            (self.edit_freqs, "Frequências (Hz) específicas do modelo canônico."),
            (self.edit_trs, "Espaçamentos T-R (m) específicos do modelo canônico."),
            (self.edit_dips, "Ângulos de dip (°) específicos do modelo canônico."),
            (self.spin_h1, "Altura h1 do modelo (m)."),
            (self.spin_pmed, "Passo entre medidas p_med (m)."),
        ]:
            _tooltip(w, tip)

        form.addRow("Frequências (Hz):", self.edit_freqs)
        form.addRow("Espaçamentos TR (m):", self.edit_trs)
        form.addRow("Ângulos de dip (graus):", self.edit_dips)
        form.addRow("Altura h1:", self.spin_h1)
        form.addRow("Passo p_med:", self.spin_pmed)

        lbl_desc = QtWidgets.QLabel(default.get("description", ""))
        lbl_desc.setStyleSheet("color:#a5a5a5; padding:6px 0;")
        lbl_desc.setWordWrap(True)

        btnbox = QtWidgets.QDialogButtonBox()
        btn_reset = btnbox.addButton(
            "↻  Restaurar padrão", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
        )
        btn_ok = btnbox.addButton(
            "Salvar", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        btn_cancel = btnbox.addButton(
            "Cancelar", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole
        )
        btn_ok.setProperty("role", "primary")
        btn_ok.clicked.connect(self._on_accept)
        btn_cancel.clicked.connect(self.reject)
        btn_reset.clicked.connect(self._on_reset)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(_heading(f"Config D — {model_name}"))
        root.addWidget(lbl_desc)
        root.addLayout(form)
        root.addWidget(btnbox)

    def _on_reset(self) -> None:
        default = default_canonical_config(self._model)
        self.edit_freqs.setText(", ".join(f"{f:g}" for f in default["freqs_hz"]))
        self.edit_trs.setText(", ".join(f"{t:g}" for t in default["tr_list_m"]))
        self.edit_dips.setText(", ".join(f"{d:g}" for d in default["dip_list_deg"]))
        self.spin_h1.setValue(float(default["h1"]))
        self.spin_pmed.setValue(float(default["p_med"]))

    def _on_accept(self) -> None:
        freqs = _parse_float_list(self.edit_freqs.text(), [20000.0])
        trs = _parse_float_list(self.edit_trs.text(), [1.0])
        dips = _parse_float_list(self.edit_dips.text(), [0.0])
        if not freqs or not trs or not dips:
            QtWidgets.QMessageBox.warning(
                self,
                "Valores inválidos",
                "Frequências, TR e dips não podem ser vazios.",
            )
            return
        self._result = {
            "freqs_hz": freqs,
            "tr_list_m": trs,
            "dip_list_deg": dips,
            "h1": float(self.spin_h1.value()),
            "p_med": float(self.spin_pmed.value()),
        }
        self.accept()

    def result_override(self) -> Optional[Dict[str, Any]]:
        return self._result


class ConfigDDialog(QtWidgets.QDialog):
    """Janela dedicada — tabela per-model de parâmetros canônicos (Config D).

    Exibe a tabela com colunas ``Habilitar | Modelo | Freqs | TR · Dip | Ação``.
    O botão "Customizar…" de cada linha abre um :class:`ConfigDRowEditor`.
    """

    def __init__(
        self,
        selected_models: List[str],
        overrides: Dict[str, Dict[str, Any]],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Config D — parâmetros canônicos per-model")
        self.setMinimumSize(880, 440)
        self._overrides: Dict[str, Dict[str, Any]] = dict(overrides or {})
        self._checks: Dict[str, QtWidgets.QCheckBox] = {}

        lbl_info = QtWidgets.QLabel(
            "Cada modelo canônico pode ser habilitado individualmente e "
            "rodado com seus próprios parâmetros (freq/TR/dip/h1/p_med). "
            "Os valores padrão vêm dos scripts <i>bench_numba_vs_fortran_local.py</i> "
            "e <i>buildValidamodels.py</i>. Clique em 'Customizar…' para editar."
        )
        lbl_info.setStyleSheet("color:#a5a5a5; padding-bottom:6px;")
        lbl_info.setWordWrap(True)

        self.table = QtWidgets.QTableWidget(len(CANONICAL_BENCHMARK_MODELS), 5)
        self.table.setHorizontalHeaderLabels(
            ["Habilitar", "Modelo", "Freqs (Hz)", "TR (m) · Dip (°)", "Ação"]
        )
        _tooltip(
            self.table,
            (
                "<b>Tabela de modelos canônicos (Config D)</b><br/>"
                "• <b>Habilitar</b>: marca o modelo para rodar.<br/>"
                "• <b>Modelo</b>: nome canônico.<br/>"
                "• <b>Freqs / TR · Dip</b>: parâmetros efetivos (default + override).<br/>"
                "• <b>Ação</b>: 'Customizar…' abre editor de linha individual."
            ),
        )
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(260)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, m in enumerate(CANONICAL_BENCHMARK_MODELS):
            cb = QtWidgets.QCheckBox()
            cb.setChecked(m in selected_models)
            self._checks[m] = cb
            wrap_cb = QtWidgets.QWidget()
            layout_cb = QtWidgets.QHBoxLayout(wrap_cb)
            layout_cb.setContentsMargins(8, 2, 8, 2)
            layout_cb.addWidget(cb)
            layout_cb.addStretch(1)
            self.table.setCellWidget(row, 0, wrap_cb)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(m))
            eff = {**default_canonical_config(m), **self._overrides.get(m, {})}
            self._refresh_row(row, eff)
            btn_edit = QtWidgets.QPushButton("Customizar…")
            btn_edit.clicked.connect(
                lambda _=False, mname=m, r=row: self._edit_row(mname, r)
            )
            self.table.setCellWidget(row, 4, btn_edit)

        btnbox = QtWidgets.QDialogButtonBox()
        btn_ok = btnbox.addButton(
            "Aplicar", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )
        btn_cancel = btnbox.addButton(
            "Cancelar", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole
        )
        btn_ok.setProperty("role", "primary")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(_heading("Config D — parâmetros canônicos per-model"))
        root.addWidget(lbl_info)
        root.addWidget(self.table, 1)
        root.addWidget(btnbox)

    def _refresh_row(self, row: int, params: Dict[str, Any]) -> None:
        freqs = params.get("freqs_hz", [])
        trs = params.get("tr_list_m", [])
        dips = params.get("dip_list_deg", [])
        freqs_txt = ", ".join(f"{f/1000:g} kHz" for f in freqs)
        tr_dip = (
            "TR=["
            + ", ".join(f"{t:g}" for t in trs)
            + "] · θ=["
            + ", ".join(f"{d:g}" for d in dips)
            + "]"
        )
        item_f = QtWidgets.QTableWidgetItem(freqs_txt)
        item_d = QtWidgets.QTableWidgetItem(tr_dip)
        self.table.setItem(row, 2, item_f)
        self.table.setItem(row, 3, item_d)

    def _edit_row(self, model_name: str, row: int) -> None:
        current = self._overrides.get(model_name, {})
        dlg = ConfigDRowEditor(model_name, current, parent=self)
        if dlg.exec():
            override = dlg.result_override()
            if override:
                self._overrides[model_name] = override
                eff = {**default_canonical_config(model_name), **override}
                self._refresh_row(row, eff)
                self._checks[model_name].setChecked(True)

    def selected_models(self) -> List[str]:
        return [m for m, cb in self._checks.items() if cb.isChecked()]

    def overrides(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._overrides)


class BenchmarkPage(QtWidgets.QWidget):
    """Página de benchmarks — Config A/B/C/D + 30k."""

    request_start = Signal()
    request_stop = Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        ncpu = os.cpu_count() or 8
        # Estado interno da Config D (persistido mesmo com diálogo fechado)
        self._config_d_models: List[str] = []
        self._config_d_overrides: Dict[str, Dict[str, Any]] = {}

        # ── Experimentos A/B/C/D/30k ──────────────────────────────────────
        grp_exp = QtWidgets.QGroupBox("Experimentos fixos (A / B / C / D / 30k)")
        exp_layout = QtWidgets.QVBoxLayout(grp_exp)
        self.check_cfg_a = QtWidgets.QCheckBox(
            "Config A — 1 frequência (20 kHz) · 1 TR (1 m) · 1 dip (0°) · 600 posições"
        )
        self.check_cfg_b = QtWidgets.QCheckBox(
            "Config B — 2 frequências (20/40 kHz) · 1 TR (1 m) · 2 dips (0°/30°) · 600 posições"
        )
        self.check_cfg_c = QtWidgets.QCheckBox(
            "Config C — 2 freqs (2 kHz, 6 kHz) · 2 TRs (8.19 m, 20.43 m) · "
            "dip 0° · h1 = 10.000 m · p_med = 0.200 m (uniforme para todos "
            "os modelos canônicos selecionados)"
        )
        self.check_cfg_d = QtWidgets.QCheckBox(
            "Config D — parâmetros canônicos per-model customizáveis "
            "(abre janela dedicada)"
        )
        self.check_30k = QtWidgets.QCheckBox(
            "Experimento 30k modelos aleatórios (20 camadas, ρₕ alta)"
        )
        self.check_cfg_a.setChecked(True)
        self.check_cfg_b.setChecked(True)

        _tooltip(
            self.check_cfg_a,
            (
                "<b>Config A — sweep uniforme baseline</b><br/>"
                "1 frequência (20 kHz), 1 TR (1 m), 1 dip (0°), 600 posições. "
                "Geometria fixa aplicada a todos os modelos canônicos selecionados. "
                "Extraída de <i>bench_numba_vs_fortran_local.py</i>."
            ),
        )
        _tooltip(
            self.check_cfg_b,
            (
                "<b>Config B — sweep uniforme multi-freq/multi-dip</b><br/>"
                "2 frequências (20/40 kHz), 1 TR, 2 dips (0°/30°), 600 posições. "
                "Útil para validar paridade em configurações de produção."
            ),
        )
        _tooltip(
            self.check_30k,
            (
                "<b>Experimento 30k</b><br/>"
                "Gera 30 000 perfis aleatórios (20 camadas, ρₕ alta) e roda Numba full. "
                "Para Fortran usa um subset e extrapola linearmente (ver 'Subset Fortran')."
            ),
        )
        _tooltip(
            self.check_cfg_c,
            (
                "<b>Config C — uniforme</b><br/>"
                "Aplica uma única configuração a todos os modelos canônicos selecionados "
                "no grupo 'Modelos canônicos (Config A/B/C)':<br/>"
                "• freqs = [2 kHz, 6 kHz]<br/>"
                "• TR = [8.19 m, 20.43 m]<br/>"
                "• dip = [0°]<br/>"
                "• h1 = 10.000 m, p_med = 0.200 m<br/>"
                "Valores extraídos de <i>buildValidamodels.py</i>."
            ),
        )
        _tooltip(
            self.check_cfg_d,
            (
                "<b>Config D — per-model customizável</b><br/>"
                "Ao marcar, abre uma janela com a tabela de modelos canônicos. "
                "Cada modelo pode ser habilitado individualmente e ter seus parâmetros "
                "(freq/TR/dip/h1/p_med) customizados via 'Customizar…'."
            ),
        )

        exp_layout.addWidget(self.check_cfg_a)
        exp_layout.addWidget(self.check_cfg_b)
        exp_layout.addWidget(self.check_cfg_c)

        # Linha Config D: checkbox + botão para reabrir janela
        row_d = QtWidgets.QHBoxLayout()
        row_d.setContentsMargins(0, 0, 0, 0)
        row_d.addWidget(self.check_cfg_d, 1)
        self.btn_open_cfg_d = QtWidgets.QPushButton("Abrir janela Config D…")
        self.btn_open_cfg_d.setEnabled(False)
        self.btn_open_cfg_d.clicked.connect(self._open_config_d_dialog)
        _tooltip(
            self.btn_open_cfg_d,
            (
                "Reabrir a janela de Config D para editar a seleção de modelos e seus "
                "overrides (freq/TR/dip/h1/p_med) sem perder o estado atual."
            ),
        )
        row_d.addWidget(self.btn_open_cfg_d)
        wrap_d = QtWidgets.QWidget()
        wrap_d.setLayout(row_d)
        exp_layout.addWidget(wrap_d)

        self.lbl_cfg_d_status = QtWidgets.QLabel("Config D: nenhum modelo habilitado")
        self.lbl_cfg_d_status.setStyleSheet(
            "color:#a5a5a5; font-size:11px; padding-left:22px;"
        )
        exp_layout.addWidget(self.lbl_cfg_d_status)

        exp_layout.addWidget(self.check_30k)

        # Config D: ao marcar, abre diálogo imediatamente (se ainda não houve)
        self.check_cfg_d.toggled.connect(self._on_cfg_d_toggled)

        grp_models_ab = QtWidgets.QGroupBox("Modelos canônicos (Config A/B/C)")
        mdl_layout = QtWidgets.QGridLayout(grp_models_ab)
        self.model_checks: Dict[str, QtWidgets.QCheckBox] = {}
        _model_tips = {
            "oklahoma_3": "3 camadas TIV simples (Tech. Report 32/2011).",
            "oklahoma_5": "5 camadas TIV gradual (Tech. Report 32/2011).",
            "oklahoma_28": "28 camadas TIV forte, ρᵥ=2·ρₕ (Tech. Report 32/2011 p.58).",
            "devine_8": "8 camadas isotrópico (Tech. Report 32/2011).",
            "hou_7": "7 camadas TIV (Hou, Mallan & Verdin 2006).",
            "viking_graben_10": "10 camadas reservatório Mar do Norte (Eidesmo et al. 2002, adapt.).",
        }
        for i, m in enumerate(CANONICAL_BENCHMARK_MODELS):
            cb = QtWidgets.QCheckBox(m)
            cb.setChecked(True)
            _tooltip(cb, f"<b>{m}</b><br/>{_model_tips.get(m, 'Modelo canônico.')}")
            self.model_checks[m] = cb
            mdl_layout.addWidget(cb, i // 2, i % 2)

        # ── Backends ────────────────────────────────────────────────────
        grp_bk = QtWidgets.QGroupBox("Backends a comparar")
        bk_layout = QtWidgets.QHBoxLayout(grp_bk)
        self.check_numba = QtWidgets.QCheckBox("Numba JIT (Python otimizado)")
        self.check_numba.setChecked(True)
        _tooltip(
            self.check_numba,
            (
                "<b>Numba JIT</b> — Python otimizado com warmup único por worker. "
                "Exporta .dat/.out e expõe o tensor H para plots. Throughput típico: 150k–1M mod/h."
            ),
        )
        self.check_fortran = QtWidgets.QCheckBox("Fortran tatu.x (OpenMP)")
        self.check_fortran.setChecked(True)
        _tooltip(
            self.check_fortran,
            (
                "<b>Fortran tatu.x</b> — código legado OpenMP. Cada modelo reescreve model.in + "
                "subprocess. Requer binário em Preferências → tatu.x."
            ),
        )
        bk_layout.addWidget(self.check_numba)
        bk_layout.addWidget(self.check_fortran)
        bk_layout.addStretch(1)

        # ── Paralelismo ─────────────────────────────────────────────────
        grp_par = QtWidgets.QGroupBox("Paralelismo")
        par_form = QtWidgets.QFormLayout(grp_par)
        par_form.setLabelAlignment(ALIGN_RIGHT)
        self.spin_workers_numba = _spin_int(max(1, ncpu // 4), 1, 256)
        self.spin_threads_numba = _spin_int(4, 1, 256)
        self.spin_workers_fortran = _spin_int(max(1, ncpu // 2), 1, 256)
        self.spin_threads_fortran = _spin_int(2, 1, 256)
        self.spin_n_iter = _spin_int(3, 1, 100)
        self.spin_fort_subset = _spin_int(300, 1, 30000)
        _tooltip(
            self.spin_workers_numba, "Nº de processos-sandbox Numba rodando em paralelo."
        )
        _tooltip(
            self.spin_threads_numba, "NUMBA_NUM_THREADS dentro de cada subprocesso Numba."
        )
        _tooltip(
            self.spin_workers_fortran, "Nº de cópias independentes de tatu.x em paralelo."
        )
        _tooltip(self.spin_threads_fortran, "OMP_NUM_THREADS por invocação de tatu.x.")
        _tooltip(
            self.spin_n_iter,
            (
                "Nº de repetições do mesmo modelo para medição estatística em Config A/B/C/D. "
                "Valores maiores reduzem ruído; 3 iterações é o mínimo aceitável."
            ),
        )
        _tooltip(
            self.spin_fort_subset,
            (
                "Quantidade de modelos do lote 30k rodados pelo Fortran. O tempo total "
                "é extrapolado linearmente para os 30 000 modelos completos."
            ),
        )
        par_form.addRow("Workers — Numba:", self.spin_workers_numba)
        par_form.addRow("Threads por worker — Numba:", self.spin_threads_numba)
        par_form.addRow("Workers — Fortran:", self.spin_workers_fortran)
        par_form.addRow("Threads por worker — Fortran:", self.spin_threads_fortran)
        par_form.addRow("Nº iterações (Config A/B/C):", self.spin_n_iter)
        par_form.addRow("Subset Fortran (30k):", self.spin_fort_subset)

        # ── Execução ────────────────────────────────────────────────────
        grp_exec = QtWidgets.QGroupBox("Execução do Benchmark")
        exec_row = QtWidgets.QHBoxLayout(grp_exec)
        self.btn_start = QtWidgets.QPushButton("▶  Iniciar Benchmark")
        self.btn_start.setProperty("role", "primary")
        _tooltip(
            self.btn_start,
            (
                "<b>Iniciar benchmark</b><br/>"
                "Executa todas as configurações marcadas (A/B/C/D/30k) sequencialmente "
                "para cada modelo canônico habilitado."
            ),
        )
        self.btn_stop = QtWidgets.QPushButton("■  Parar")
        self.btn_stop.setProperty("role", "danger")
        self.btn_stop.setEnabled(False)
        _tooltip(
            self.btn_stop,
            (
                "Sinaliza parada entre células (modelo × config). A célula atual "
                "termina antes de liberar o thread."
            ),
        )
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        _tooltip(self.progress, "Células concluídas / total de células do benchmark.")
        self.lbl_current = QtWidgets.QLabel("(aguardando)")
        _tooltip(self.lbl_current, "Modelo × config atualmente sendo executado.")
        exec_row.addWidget(self.btn_start)
        exec_row.addWidget(self.btn_stop)
        exec_row.addWidget(self.progress, 1)
        exec_row.addWidget(self.lbl_current)
        self.btn_start.clicked.connect(self.request_start.emit)
        self.btn_stop.clicked.connect(self.request_stop.emit)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Log de benchmark aparecerá aqui.")
        self.log_view.setMaximumBlockCount(5000)
        _tooltip(
            self.log_view,
            (
                "Log estruturado do benchmark: cada linha mostra [idx/total] modelo · "
                "Config · throughput parcial."
            ),
        )

        self.table = QtWidgets.QTableWidget(0, 8)
        _tooltip(
            self.table,
            (
                "<b>Tabela de resultados</b><br/>"
                "Uma linha por célula (modelo × config). Colunas: ms/modelo e modelos/hora "
                "para cada backend + speedup Fortran/Numba."
            ),
        )
        self.table.setHorizontalHeaderLabels(
            [
                "Modelo",
                "Config",
                "Nº modelos",
                "Numba (ms/mod)",
                "Fortran (ms/mod)",
                "Numba (mod/h)",
                "Fortran (mod/h)",
                "Speedup",
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)

        grid = QtWidgets.QVBoxLayout(self)
        grid.setContentsMargins(20, 16, 20, 16)
        grid.setSpacing(12)
        grid.addWidget(_heading("Benchmark — Numba vs Fortran"))
        grid.addWidget(
            _heading(
                "Compare throughput e paridade numérica entre os dois simuladores. "
                "Config A/B/C aplicam parâmetros uniformes; Config D permite "
                "customização per-model numa janela dedicada.",
                role="section",
            )
        )
        top = QtWidgets.QHBoxLayout()
        col_left = QtWidgets.QVBoxLayout()
        col_left.addWidget(grp_exp)
        col_left.addWidget(grp_models_ab)
        col_right = QtWidgets.QVBoxLayout()
        col_right.addWidget(grp_bk)
        col_right.addWidget(grp_par)
        top.addLayout(col_left, 1)
        top.addLayout(col_right, 1)
        grid.addLayout(top)
        grid.addWidget(grp_exec)
        grid.addWidget(self.log_view, 1)
        grid.addWidget(self.table, 1)

    # ── Config D dialog management ───────────────────────────────────────
    def _on_cfg_d_toggled(self, checked: bool) -> None:
        """Quando a checkbox Config D é marcada, abre o diálogo da tabela.

        Se desmarcada, apenas desativa o botão 'Abrir janela…'. A seleção
        interna de modelos é preservada (para re-habilitação sem re-configurar).
        """
        self.btn_open_cfg_d.setEnabled(checked)
        if checked and not self._config_d_models:
            # Primeira vez: abre automaticamente para o usuário configurar
            self._open_config_d_dialog()
            if not self._config_d_models:
                # Usuário cancelou; desfaz a checagem silenciosamente
                self.check_cfg_d.blockSignals(True)
                self.check_cfg_d.setChecked(False)
                self.btn_open_cfg_d.setEnabled(False)
                self.check_cfg_d.blockSignals(False)

    def _open_config_d_dialog(self) -> None:
        dlg = ConfigDDialog(
            selected_models=self._config_d_models,
            overrides=self._config_d_overrides,
            parent=self,
        )
        if dlg.exec():
            self._config_d_models = dlg.selected_models()
            self._config_d_overrides = dlg.overrides()
            self._update_cfg_d_status()

    def _update_cfg_d_status(self) -> None:
        if self._config_d_models:
            txt = (
                f"Config D: {len(self._config_d_models)} modelo(s) habilitado(s) — "
                + ", ".join(self._config_d_models)
            )
        else:
            txt = "Config D: nenhum modelo habilitado"
        self.lbl_cfg_d_status.setText(txt)

    # ── API pública ──────────────────────────────────────────────────────
    def selected_models(self) -> List[str]:
        return [m for m, cb in self.model_checks.items() if cb.isChecked()]

    def selected_config_d_models(self) -> List[str]:
        return list(self._config_d_models)

    def config_d_overrides(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._config_d_overrides)

    def append_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def populate_table(self, records: List[BenchRecord]) -> None:
        self.table.setRowCount(0)
        for r in records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            vals = [
                r.model_name,
                r.config,
                str(r.n_models),
                format_float(r.numba_ms_per_model, 2),
                format_float(r.fortran_ms_per_model, 2),
                format_float(r.numba_mod_h, 0, thousands=True),
                format_float(r.fortran_mod_h, 0, thousands=True),
                f"{format_float(r.speedup, 2)}×",
            ]
            for col, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)
                item.setTextAlignment(ALIGN_CENTER if ALIGN_CENTER else 0x4)
                self.table.setItem(row, col, item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_a": self.check_cfg_a.isChecked(),
            "run_b": self.check_cfg_b.isChecked(),
            "run_c": self.check_cfg_c.isChecked(),
            "run_d": self.check_cfg_d.isChecked(),
            "run_d_models": list(self._config_d_models),
            "run_d_overrides": self._config_d_overrides,
            "run_30k": self.check_30k.isChecked(),
            "models_ab": self.selected_models(),
            "bk_numba": self.check_numba.isChecked(),
            "bk_fortran": self.check_fortran.isChecked(),
            "workers_numba": self.spin_workers_numba.value(),
            "threads_numba": self.spin_threads_numba.value(),
            "workers_fortran": self.spin_workers_fortran.value(),
            "threads_fortran": self.spin_threads_fortran.value(),
            "n_iter": self.spin_n_iter.value(),
            "fort_subset": self.spin_fort_subset.value(),
        }

    def from_dict(self, d: Dict[str, Any]) -> None:
        if not d:
            return
        try:
            self.check_cfg_a.setChecked(bool(d.get("run_a", True)))
            self.check_cfg_b.setChecked(bool(d.get("run_b", True)))
            self.check_cfg_c.setChecked(bool(d.get("run_c", False)))
            self.check_30k.setChecked(bool(d.get("run_30k", False)))
            self.check_numba.setChecked(bool(d.get("bk_numba", True)))
            self.check_fortran.setChecked(bool(d.get("bk_fortran", True)))
            self.spin_workers_numba.setValue(int(d.get("workers_numba", 2)))
            self.spin_threads_numba.setValue(int(d.get("threads_numba", 4)))
            self.spin_workers_fortran.setValue(int(d.get("workers_fortran", 4)))
            self.spin_threads_fortran.setValue(int(d.get("threads_fortran", 2)))
            self.spin_n_iter.setValue(int(d.get("n_iter", 3)))
            self.spin_fort_subset.setValue(int(d.get("fort_subset", 300)))
            models_ab = d.get("models_ab", [])
            for m, cb in self.model_checks.items():
                cb.setChecked(m in models_ab or not models_ab)
            # Config D state
            self._config_d_overrides = dict(d.get("run_d_overrides", {}) or {})
            self._config_d_models = list(d.get("run_d_models", []) or [])
            # Toggle checkbox D without firing the dialog
            self.check_cfg_d.blockSignals(True)
            self.check_cfg_d.setChecked(bool(d.get("run_d", False)))
            self.btn_open_cfg_d.setEnabled(self.check_cfg_d.isChecked())
            self.check_cfg_d.blockSignals(False)
            self._update_cfg_d_status()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════
# Aba 4 — Resultados (+ model selector)
# ══════════════════════════════════════════════════════════════════════════


class ResultsPage(QtWidgets.QWidget):
    """Página de plotagem interativa — simulação + benchmark + seletor de modelo."""

    def __init__(
        self,
        style: PlotStyle,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._style = style
        self._current_sim: Optional[Dict[str, Any]] = None
        self._benchmark_plot_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._sim_models: List[Dict[str, Any]] = []  # cache dos modelos gerados

        # Painel esquerdo — controles
        controls = QtWidgets.QGroupBox("Controles de Plot")
        ctrl_layout = QtWidgets.QFormLayout(controls)
        ctrl_layout.setLabelAlignment(ALIGN_RIGHT)
        ctrl_layout.setHorizontalSpacing(12)

        self.combo_source = QtWidgets.QComboBox()
        self.combo_source.addItems(["Simulação atual", "Benchmark (Numba vs Fortran)"])
        _tooltip(self.combo_source, "Fonte dos dados a plotar.")

        self.combo_model_key = QtWidgets.QComboBox()
        _tooltip(
            self.combo_model_key,
            (
                "Célula específica (modelo × config) a plotar. Em 'Simulação atual', "
                "se múltiplos modelos foram gerados, lista os índices 0..N−1."
            ),
        )

        self.list_components = QtWidgets.QListWidget()
        self.list_components.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
            if hasattr(QtWidgets.QAbstractItemView, "SelectionMode")
            else QtWidgets.QAbstractItemView.MultiSelection
        )
        self.list_components.setMaximumHeight(150)
        for c in COMPONENT_NAMES:
            item = QtWidgets.QListWidgetItem(c)
            self.list_components.addItem(item)
            if c in ("Hxx", "Hzz"):
                item.setSelected(True)
        _tooltip(
            self.list_components,
            (
                "<b>Componentes EM a plotar</b><br/>"
                "Seleção múltipla das 9 componentes do tensor H (Hxx..Hzz). Relevantes:<br/>"
                "• <b>Hxx / Hyy</b> — diagonais XY (dominantes em dip baixo).<br/>"
                "• <b>Hzz</b> — eixo Z (sensibilidade ao limite de camada).<br/>"
                "• <b>Hxz / Hzx</b> — off-diagonais (aparecem apenas em dip ≠ 0)."
            ),
        )

        self.combo_plot_kind = QtWidgets.QComboBox()
        self.combo_plot_kind.addItems(
            [
                "Tensor H completo (Re/Im 3×6)",
                "Componentes EM",
                "Perfil de Resistividade",
                "Geosinais (USD/UAD/UHR/UHA)",
                "Anisotropia λ = √(ρᵥ/ρₕ)",
                "Benchmark compare (Numba vs Fortran)",
            ]
        )
        _tooltip(
            self.combo_plot_kind,
            (
                "<b>Tipo de plot</b><br/>"
                "• <b>Tensor H completo</b>: grade 3×6 Re/Im para visão panorâmica.<br/>"
                "• <b>Componentes EM</b>: usa o 'Modo' selecionado abaixo.<br/>"
                "• <b>Perfil de Resistividade</b>: ρₕ/ρᵥ vs profundidade.<br/>"
                "• <b>Geosinais</b>: USD/UAD/UHR/UHA derivados.<br/>"
                "• <b>Anisotropia</b>: λ = √(ρᵥ/ρₕ) vs profundidade.<br/>"
                "• <b>Benchmark compare</b>: Numba (linha) vs Fortran (pontos)."
            ),
        )

        self.combo_kind_mode = QtWidgets.QComboBox()
        self.combo_kind_mode.addItems(PLOT_KINDS)
        # v2.4: default "Re + Im" conforme requisito (antes era Magnitude+Fase)
        idx_reim = self.combo_kind_mode.findText("Re + Im")
        if idx_reim >= 0:
            self.combo_kind_mode.setCurrentIndex(idx_reim)
        _tooltip(
            self.combo_kind_mode,
            (
                "<b>Modo de renderização (para 'Componentes EM')</b><br/>"
                "Magnitude + Fase, Re + Im (★ padrão), Magnitude (dB) ou um "
                "dos 'Só …' (painel único). Ignorado pelos outros tipos de plot."
            ),
        )

        # ── v2.4: Filtros TR / Ângulo / Frequência (QListWidget checkável) ──
        # Permitem ao usuário escolher quais combinações plotar em vez de
        # sempre a primeira (comportamento antigo: itr=0, iang=0, ifq=0).
        self.list_tr_filter = self._make_check_list("Espaçamentos T-R")
        self.list_ang_filter = self._make_check_list("Ângulos de dip")
        self.list_freq_filter = self._make_check_list("Frequências")
        # ── v2.4: Filtro de geosinais (5 derivados) ─────────────────────────
        self.list_geo_filter = self._make_check_list("Geosinais")
        for name in ("USD", "UAD", "UHR", "UHA", "U3DF"):
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.list_geo_filter.addItem(item)

        _tooltip(
            self.list_tr_filter,
            "Combinações de TR a plotar (padrão: todas marcadas).",
        )
        _tooltip(
            self.list_ang_filter,
            "Ângulos de dip a plotar (padrão: todos marcados).",
        )
        _tooltip(
            self.list_freq_filter,
            "Frequências a plotar (padrão: todas marcadas).",
        )
        _tooltip(
            self.list_geo_filter,
            (
                "<b>Geosinais a plotar</b><br/>"
                "• <b>USD</b>: log₁₀|Hxx/Hyy| (Up-Side-Detection).<br/>"
                "• <b>UAD</b>: ∠Hxx − ∠Hyy (°).<br/>"
                "• <b>UHR</b>: log₁₀|Hxz/Hzz| (harmonic resistivity).<br/>"
                "• <b>UHA</b>: ∠Hxz − ∠Hzz (°).<br/>"
                "• <b>U3DF</b>: (Hxy − Hyx) / (Hxy + Hyx) — fator 3D assimétrico."
            ),
        )

        self.btn_plot = QtWidgets.QPushButton("Plotar")
        self.btn_plot.setProperty("role", "primary")
        _tooltip(self.btn_plot, "Renderiza o plot com os dados e configurações atuais.")
        self.btn_save = QtWidgets.QPushButton("Salvar figura…")
        _tooltip(self.btn_save, "Exporta a figura atual como PNG, PDF ou SVG no disco.")

        ctrl_layout.addRow("Fonte:", self.combo_source)
        ctrl_layout.addRow("Modelo / célula:", self.combo_model_key)
        ctrl_layout.addRow("Componentes EM:", self.list_components)
        ctrl_layout.addRow("Tipo de plot:", self.combo_plot_kind)
        ctrl_layout.addRow("Modo (Componentes EM):", self.combo_kind_mode)
        ctrl_layout.addRow("Filtro — TR:", self.list_tr_filter)
        ctrl_layout.addRow("Filtro — Ângulo:", self.list_ang_filter)
        ctrl_layout.addRow("Filtro — Frequência:", self.list_freq_filter)
        ctrl_layout.addRow("Filtro — Geosinais:", self.list_geo_filter)
        ctrl_layout.addRow(self.btn_plot)
        ctrl_layout.addRow(self.btn_save)

        # Painel direito — canvas
        self.canvas = EMCanvas(self, figsize=(14, 9), style=self._style)

        splitter = QtWidgets.QSplitter(ORIENT_H if ORIENT_H else 0x1)
        left_wrap = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_wrap)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(controls)
        left_layout.addStretch(1)
        splitter.addWidget(left_wrap)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1100])

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.addWidget(_heading("Resultados — Visualização Interativa"))
        root.addWidget(
            _heading(
                "Plots de componentes EM, perfil ρ, geosinais e comparação Numba vs Fortran. "
                "Selecione o modelo em 'Modelo / célula' quando múltiplos perfis foram gerados.",
                role="section",
            )
        )
        root.addWidget(splitter, 1)

        self.btn_plot.clicked.connect(self._on_plot)
        self.btn_save.clicked.connect(self._on_save)
        self.combo_source.currentIndexChanged.connect(self._refresh_keys)

    # ── v2.4: helpers de filtro ───────────────────────────────────────────
    def _make_check_list(self, label: str) -> QtWidgets.QListWidget:
        """Cria um QListWidget vazio com itens checkáveis e altura limitada."""
        lst = QtWidgets.QListWidget()
        lst.setMaximumHeight(90)
        lst.setAlternatingRowColors(True)
        lst.setToolTip(f"<b>Filtro — {label}</b>")
        return lst

    def _populate_filter(
        self,
        widget: QtWidgets.QListWidget,
        values: Sequence[Any],
        fmt: str = "{}",
    ) -> None:
        """Popula o QListWidget com itens checáveis (todos marcados)."""
        widget.blockSignals(True)
        try:
            widget.clear()
            for v in values:
                item = QtWidgets.QListWidgetItem(fmt.format(v))
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                widget.addItem(item)
        finally:
            widget.blockSignals(False)

    def _filter_mask(self, widget: QtWidgets.QListWidget) -> List[bool]:
        """Retorna máscara booleana de itens marcados (todos True se vazio)."""
        n = widget.count()
        if n == 0:
            return []
        return [
            widget.item(i).checkState() == QtCore.Qt.CheckState.Checked for i in range(n)
        ]

    def _geosignal_mask(self) -> List[bool]:
        """Máscara dos 5 geosinais na ordem [USD, UAD, UHR, UHA, U3DF]."""
        return self._filter_mask(self.list_geo_filter)

    def set_style(self, style: PlotStyle) -> None:
        self._style = style
        if self.canvas is not None:
            self.canvas.set_style(style)

    def set_current_simulation(
        self, result: Dict[str, Any], models: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        self._current_sim = result
        self._sim_models = list(models or [])
        self._refresh_keys()
        # v2.4: popula filtros TR / ang / freq com os valores deste sim
        self._populate_filter(self.list_tr_filter, result.get("trs", []), "{} m")
        self._populate_filter(self.list_ang_filter, result.get("dips", []), "{}°")
        self._populate_filter(self.list_freq_filter, result.get("freqs", []), "{:g} Hz")

    def set_benchmark_plot_data(
        self, data: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> None:
        self._benchmark_plot_data = data
        self._refresh_keys()
        # Popula filtros a partir da primeira entrada (todos os modelos usam
        # o mesmo conjunto de freqs/TRs/dips dentro de um benchmark).
        if data:
            first = next(iter(data.values()))
            self._populate_filter(self.list_tr_filter, first.get("trs", []), "{} m")
            self._populate_filter(self.list_ang_filter, first.get("dips", []), "{}°")
            self._populate_filter(
                self.list_freq_filter, first.get("freqs", []), "{:g} Hz"
            )

    def _refresh_keys(self) -> None:
        self.combo_model_key.clear()
        if self.combo_source.currentIndex() == 0:
            if self._current_sim is not None:
                H = self._current_sim.get("H_stack")
                n_models = int(H.shape[0]) if (H is not None and H.ndim == 6) else 1
                if n_models <= 1:
                    self.combo_model_key.addItem("(simulação atual — modelo único)")
                else:
                    for i in range(n_models):
                        title = ""
                        if i < len(self._sim_models):
                            md = self._sim_models[i]
                            title = (
                                md.get("title", "")
                                or f'{md.get("n_layers", "?")} camadas'
                            )
                        self.combo_model_key.addItem(
                            f"Modelo #{i + 1}/{n_models} — {title}".strip(" —")
                        )
        else:
            for m, c in self._benchmark_plot_data.keys():
                self.combo_model_key.addItem(f"{m} / Config {c}")

    def _selected_components(self) -> List[str]:
        items = self.list_components.selectedItems()
        return [it.text() for it in items] if items else ["Hxx", "Hzz"]

    def _resolve_current_sim_H(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Retorna (H0 — tensor do modelo selecionado, model_dict) para sim atual."""
        sim = self._current_sim or {}
        H = sim.get("H_stack")
        if H is None or not hasattr(H, "size") or H.size == 0:
            return None, {}
        idx = max(0, self.combo_model_key.currentIndex())
        # Quando só há 1 modelo, combo_model_key.currentIndex() pode ser 0
        if H.ndim == 6:
            idx = min(idx, H.shape[0] - 1)
            H0 = H[idx]
            md = self._sim_models[idx] if idx < len(self._sim_models) else {}
        else:
            H0 = H
            md = sim.get("model", {}) or {}
        return H0, md

    def _on_plot(self) -> None:
        kind = self.combo_plot_kind.currentText()
        kind_mode = self.combo_kind_mode.currentText()
        comps = self._selected_components()
        # v2.4: m\u00e1scaras de combina\u00e7\u00f5es selecionadas pelo usu\u00e1rio
        tr_mask = self._filter_mask(self.list_tr_filter)
        ang_mask = self._filter_mask(self.list_ang_filter)
        freq_mask = self._filter_mask(self.list_freq_filter)
        geo_mask = self._geosignal_mask()

        if self.combo_source.currentIndex() == 0:
            sim = self._current_sim
            if not sim:
                return
            H0, md = self._resolve_current_sim_H()
            if H0 is None:
                return
            z_obs = sim.get("z_obs")
            freqs = sim.get("freqs", [20000.0])
            trs = sim.get("trs", [1.0])
            dips = sim.get("dips", [0.0])
            rho_h = md.get("rho_h", [])
            rho_v = md.get("rho_v", rho_h)
            thicknesses = md.get("thicknesses", [])
            title = md.get("title", "") or f"Modelo {self.combo_model_key.currentText()}"

            if kind == "Tensor H completo (Re/Im 3×6)":
                plot_tensor_full(
                    self.canvas,
                    H0,
                    z_obs,
                    freqs,
                    trs,
                    dips,
                    title=title,
                    thicknesses=thicknesses,
                    rho_h=rho_h,
                    rho_v=rho_v,
                    style=self._style,
                    tr_mask=tr_mask or None,
                    ang_mask=ang_mask or None,
                    freq_mask=freq_mask or None,
                )
            elif kind == "Componentes EM":
                plot_em_profile(
                    self.canvas,
                    H0,
                    z_obs,
                    freqs,
                    trs,
                    dips,
                    comps,
                    title=f"{title} — {', '.join(comps)}",
                    kind=kind_mode,
                    style=self._style,
                    thicknesses=thicknesses,
                )
            elif kind == "Geosinais (USD/UAD/UHR/UHA)":
                plot_geosignals(
                    self.canvas,
                    H0,
                    z_obs,
                    freqs,
                    trs,
                    dips,
                    title=f"Geosinais — {title}",
                    style=self._style,
                    thicknesses=thicknesses,
                    geosignal_mask=geo_mask or None,
                    tr_mask=tr_mask or None,
                    ang_mask=ang_mask or None,
                    freq_mask=freq_mask or None,
                )
            elif kind == "Perfil de Resistividade":
                plot_resistivity_profile(
                    self.canvas,
                    rho_h,
                    rho_v,
                    thicknesses,
                    title=title,
                    style=self._style,
                    z_obs=z_obs,
                )
            elif kind == "Anisotropia λ = √(ρᵥ/ρₕ)":
                plot_anisotropy(
                    self.canvas,
                    rho_h,
                    rho_v,
                    thicknesses,
                    title=f"Anisotropia — {title}",
                    style=self._style,
                )
            else:
                plot_benchmark_compare(
                    self.canvas,
                    H0,
                    None,
                    z_obs,
                    freqs,
                    trs,
                    dips,
                    comps,
                    title="Somente Numba (sem benchmark ativo)",
                    style=self._style,
                    thicknesses=thicknesses,
                )
        else:
            key_text = self.combo_model_key.currentText()
            if not key_text or "/" not in key_text:
                return
            parts = key_text.split("/")
            model_name = parts[0].strip()
            config = parts[1].replace("Config", "").strip()
            entry = self._benchmark_plot_data.get((model_name, config))
            if entry is None:
                return
            freqs = entry.get("freqs", [20000.0])
            trs = entry.get("trs", [1.0])
            dips = entry.get("dips", [0.0])
            H_n = entry.get("H_numba")
            H_f = entry.get("H_fortran")
            md = entry.get("model", {}) or {}
            thicknesses = md.get("thicknesses", [])
            rho_h = md.get("rho_h", [])
            rho_v = md.get("rho_v", rho_h)
            positions_z = entry.get("positions_z")
            z_obs_2d = None
            if positions_z is not None:
                z_obs_2d = np.broadcast_to(positions_z, (len(dips), positions_z.shape[0]))

            # v2.4/E16: remove guarda "and H_n is not None" \u2014 aceita Fortran
            # quando Numba ausente. Usa Numba se dispon\u00edvel, sen\u00e3o Fortran.
            if kind == "Tensor H completo (Re/Im 3×6)":
                H_plot = H_n if H_n is not None else H_f
                backend_tag = (
                    "Numba+Fortran (sobrepostos)"
                    if H_n is not None and H_f is not None
                    else "Numba" if H_n is not None else "Fortran"
                )
                if H_plot is None:
                    return
                plot_tensor_full(
                    self.canvas,
                    H_plot,
                    z_obs_2d,
                    freqs,
                    trs,
                    dips,
                    title=(
                        f"{md.get('title', model_name)} — Config {config} "
                        f"[{backend_tag}]"
                    ),
                    thicknesses=thicknesses,
                    rho_h=rho_h,
                    rho_v=rho_v,
                    style=self._style,
                    tr_mask=tr_mask or None,
                    ang_mask=ang_mask or None,
                    freq_mask=freq_mask or None,
                )
            elif kind == "Perfil de Resistividade":
                plot_resistivity_profile(
                    self.canvas,
                    rho_h,
                    rho_v,
                    thicknesses,
                    title=f"{md.get('title', model_name)} — Config {config}",
                    style=self._style,
                    z_obs=z_obs_2d,
                )
            elif kind == "Anisotropia λ = √(ρᵥ/ρₕ)":
                plot_anisotropy(
                    self.canvas,
                    rho_h,
                    rho_v,
                    thicknesses,
                    title=f"{md.get('title', model_name)} — Config {config}",
                    style=self._style,
                )
            elif kind == "Componentes EM":
                H_plot = H_n if H_n is not None else H_f
                if H_plot is None:
                    return
                backend_tag = (
                    "Numba"
                    if H_n is not None and H_f is None
                    else "Fortran" if H_f is not None and H_n is None else "Numba+Fortran"
                )
                plot_em_profile(
                    self.canvas,
                    H_plot,
                    z_obs_2d,
                    freqs,
                    trs,
                    dips,
                    comps,
                    title=f"{model_name} — Config {config} ({backend_tag})",
                    kind=kind_mode,
                    style=self._style,
                    thicknesses=thicknesses,
                )
            elif kind == "Geosinais (USD/UAD/UHR/UHA)":
                H_plot = H_n if H_n is not None else H_f
                if H_plot is None:
                    return
                plot_geosignals(
                    self.canvas,
                    H_plot,
                    z_obs_2d,
                    freqs,
                    trs,
                    dips,
                    title=f"{model_name} — Config {config}",
                    style=self._style,
                    thicknesses=thicknesses,
                    geosignal_mask=geo_mask or None,
                    tr_mask=tr_mask or None,
                    ang_mask=ang_mask or None,
                    freq_mask=freq_mask or None,
                )
            else:
                plot_benchmark_compare(
                    self.canvas,
                    H_n,
                    H_f,
                    z_obs_2d,
                    freqs,
                    trs,
                    dips,
                    comps,
                    title=f"Benchmark — {model_name} · Config {config}",
                    style=self._style,
                    thicknesses=thicknesses,
                )

    def _on_save(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Salvar figura",
            "plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if path:
            self.canvas.save(path)


# ══════════════════════════════════════════════════════════════════════════
# Aba 5 — Preferências (+ LaTeX + extras)
# ══════════════════════════════════════════════════════════════════════════


class PreferencesPage(QtWidgets.QWidget):
    """Página de preferências: caminhos manuais + plot style + LaTeX."""

    paths_changed = Signal(dict)
    style_changed = Signal(object)

    def __init__(
        self,
        paths: Dict[str, str],
        style: PlotStyle,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._paths = dict(paths)
        self._style = style

        # ─── Paths ───────────────────────────────────────────────────────
        grp_paths = QtWidgets.QGroupBox("Caminhos (segurança · override manual)")
        grp_paths.setToolTip(
            "Caso a detecção automática falhe, defina aqui o caminho absoluto "
            "para geosteering_ai, tatu.x e o interpretador Python."
        )
        paths_form = QtWidgets.QFormLayout(grp_paths)
        paths_form.setLabelAlignment(ALIGN_RIGHT)

        self.edit_pkg = _qline(self._paths.get("geosteering_ai", ""), "…/geosteering_ai")
        _tooltip(
            self.edit_pkg,
            (
                "<b>Pacote geosteering_ai</b><br/>"
                "Caminho absoluto da pasta do pacote Python (contém <code>__init__.py</code>). "
                "Usado para <code>sys.path.insert</code> e importar os módulos de simulação."
            ),
        )
        self.edit_tatu = _qline(
            self._paths.get("tatu_binary", ""), "…/Fortran_Gerador/tatu.x"
        )
        _tooltip(
            self.edit_tatu,
            (
                "<b>Binário Fortran tatu.x</b><br/>"
                "Executável compilado do simulador legado (OpenMP). Deve ter permissão de execução. "
                "Verificado com <code>os.access(path, os.X_OK)</code>."
            ),
        )
        self.edit_python = _qline(self._paths.get("python_binary", ""), "…/bin/python")
        _tooltip(
            self.edit_python,
            (
                "<b>Interpretador Python</b><br/>"
                "Executável Python usado pelos subprocessos (ProcessPoolExecutor). "
                "Por padrão, <code>sys.executable</code> do processo pai."
            ),
        )
        self.edit_output = _qline(self._paths.get("output_dir", ""), "diretório de saída")
        _tooltip(
            self.edit_output,
            (
                "<b>Diretório de saída padrão</b><br/>"
                "Usado quando o experimento não define output_dir próprio. "
                "Artefatos .dat/.out/CSV/plots são gravados nesta pasta."
            ),
        )

        self.lbl_pkg_status = QtWidgets.QLabel("")
        self.lbl_tatu_status = QtWidgets.QLabel("")
        self.lbl_python_status = QtWidgets.QLabel("")
        for lbl in (self.lbl_pkg_status, self.lbl_tatu_status, self.lbl_python_status):
            lbl.setStyleSheet("color:#a5a5a5; font-size:11px;")
        _tooltip(self.lbl_pkg_status, "Status de validação do pacote (em tempo real).")
        _tooltip(
            self.lbl_tatu_status,
            "Status de validação do binário tatu.x (existe + executável).",
        )
        _tooltip(self.lbl_python_status, "Status de validação do interpretador Python.")

        btn_pkg = QtWidgets.QPushButton("Procurar pasta…")
        btn_pkg.clicked.connect(lambda: self._browse_dir(self.edit_pkg))
        _tooltip(
            btn_pkg, "Abrir seletor de pastas para localizar a pasta geosteering_ai."
        )
        btn_tatu = QtWidgets.QPushButton("Procurar binário…")
        btn_tatu.clicked.connect(lambda: self._browse_file(self.edit_tatu))
        _tooltip(btn_tatu, "Abrir seletor de arquivos para localizar o binário tatu.x.")
        btn_python = QtWidgets.QPushButton("Procurar intérprete…")
        btn_python.clicked.connect(lambda: self._browse_file(self.edit_python))
        _tooltip(btn_python, "Abrir seletor para localizar o interpretador Python.")
        btn_output = QtWidgets.QPushButton("Procurar pasta…")
        btn_output.clicked.connect(lambda: self._browse_dir(self.edit_output))
        _tooltip(btn_output, "Abrir seletor de pastas para o diretório de saída padrão.")
        btn_auto = QtWidgets.QPushButton("↻  Redetectar automaticamente")
        btn_auto.clicked.connect(self._auto_detect)
        _tooltip(
            btn_auto,
            (
                "Revarredura ascendente por 'geosteering_ai/__init__.py' e 'Fortran_Gerador/tatu.x' "
                "a partir do diretório atual. Preenche os campos acima automaticamente."
            ),
        )

        def _row(
            edit: "QtWidgets.QLineEdit", btn: "QtWidgets.QPushButton"
        ) -> "QtWidgets.QWidget":
            row = QtWidgets.QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(btn)
            w = QtWidgets.QWidget()
            w.setLayout(row)
            return w

        paths_form.addRow("geosteering_ai:", _row(self.edit_pkg, btn_pkg))
        paths_form.addRow("", self.lbl_pkg_status)
        paths_form.addRow("tatu.x (Fortran):", _row(self.edit_tatu, btn_tatu))
        paths_form.addRow("", self.lbl_tatu_status)
        paths_form.addRow("Python (interpretador):", _row(self.edit_python, btn_python))
        paths_form.addRow("", self.lbl_python_status)
        paths_form.addRow("Diretório de saída:", _row(self.edit_output, btn_output))
        paths_form.addRow("", btn_auto)

        # ─── Plot style ──────────────────────────────────────────────────
        grp_style = QtWidgets.QGroupBox("Estilo de Plot")
        style_form = QtWidgets.QFormLayout(grp_style)
        style_form.setLabelAlignment(ALIGN_RIGHT)

        self.spin_dpi = _spin_int(style.dpi, 50, 600)
        _tooltip(
            self.spin_dpi,
            (
                "<b>Resolução (DPI)</b><br/>"
                "Pontos por polegada da figura. 100 = tela; 150–200 = apresentação; "
                "300+ = publicação. Afeta também savefig.dpi."
            ),
        )
        self.spin_font = _spin_int(style.font_size, 6, 32)
        _tooltip(
            self.spin_font,
            (
                "<b>Tamanho base da fonte (pt)</b><br/>"
                "Aplicado a labels, títulos e legendas (com variações por rótulo). "
                "Padrão matplotlib = 10."
            ),
        )
        self.combo_font_family = QtWidgets.QComboBox()
        self.combo_font_family.setEditable(True)
        self.combo_font_family.addItems(
            [
                "DejaVu Sans",
                "Helvetica",
                "Arial",
                "SF Pro Display",
                "Inter",
                "Roboto",
                "CMU Serif",
                "Times New Roman",
            ]
        )
        self.combo_font_family.setCurrentText(style.font_family)
        self.spin_lw = _spin_float(style.line_width, 0.3, 6.0, 0.1, 2)
        self.spin_axis_prec = _spin_int(style.axis_precision, 0, 8)
        self.spin_spine_w = _spin_float(style.spine_width, 0.2, 4.0, 0.1, 2)
        self.spin_marker_size = _spin_float(style.marker_size, 1.0, 20.0, 0.5, 2)
        self.check_grid = QtWidgets.QCheckBox("Exibir grid de fundo")
        self.check_grid.setChecked(style.grid)
        self.check_layer_boundary = QtWidgets.QCheckBox("Desenhar interfaces das camadas")
        self.check_layer_boundary.setChecked(style.show_layer_boundaries)
        self.check_minor_ticks = QtWidgets.QCheckBox("Mostrar ticks menores nos eixos")
        self.check_minor_ticks.setChecked(style.minor_ticks)
        self.check_use_latex = QtWidgets.QCheckBox(
            "Renderizar textos e fórmulas com LaTeX (requer TeX instalado)"
        )
        self.check_use_latex.setChecked(style.use_latex)
        _tooltip(
            self.check_use_latex,
            (
                "<b>LaTeX em plots</b><br/>"
                "Ativa <code>rcParams['text.usetex']=True</code>. Requer um ambiente "
                "LaTeX completo no sistema (<code>pdflatex</code>, <code>dvipng</code>). "
                "Se ausente, matplotlib emite warnings e cai em mathtext nativo.<br/>"
                "macOS: <code>brew install --cask mactex-no-gui</code><br/>"
                "Linux: <code>sudo apt install texlive-latex-extra dvipng</code>"
            ),
        )
        self.check_use_mathtext = QtWidgets.QCheckBox(
            "Usar mathtext nativo (Computer Modern, sem dependência externa)"
        )
        self.check_use_mathtext.setChecked(style.use_mathtext)
        self.combo_palette = QtWidgets.QComboBox()
        self.combo_palette.addItems(
            [
                "tab10",
                "tab20",
                "Set1",
                "Set2",
                "viridis",
                "cividis",
                "plasma",
                "magma",
                "coolwarm",
            ]
        )
        self.combo_palette.setCurrentText(style.palette)
        self.combo_legend_loc = QtWidgets.QComboBox()
        self.combo_legend_loc.addItems(
            [
                "best",
                "upper right",
                "upper left",
                "lower right",
                "lower left",
                "center left",
                "center right",
                "upper center",
                "lower center",
                "center",
            ]
        )
        self.combo_legend_loc.setCurrentText(style.legend_location)
        self.combo_title_loc = QtWidgets.QComboBox()
        self.combo_title_loc.addItems(["center", "left", "right"])
        self.combo_title_loc.setCurrentText(style.title_location)
        self.combo_line_style = QtWidgets.QComboBox()
        self.combo_line_style.addItems(["-", "--", "-.", ":"])
        self.combo_line_style.setCurrentText(style.line_style)
        self.combo_marker_style = QtWidgets.QComboBox()
        self.combo_marker_style.addItems(["o", "s", "^", "v", "D", "x", "+", "*"])
        self.combo_marker_style.setCurrentText(style.marker_style)

        self.btn_color_real = self._color_btn(style.color_real)
        self.btn_color_imag = self._color_btn(style.color_imag)
        self.btn_color_rho_h = self._color_btn(style.color_rho_h)
        self.btn_color_rho_v = self._color_btn(style.color_rho_v)
        self.btn_color_numba = self._color_btn(style.color_numba)
        self.btn_color_fortran = self._color_btn(style.color_fortran)
        self.btn_color_layer = self._color_btn(style.color_layer_boundary)
        self.btn_color_bg = self._color_btn(style.background)

        # ── Tooltips didáticos para todos os widgets de estilo ──────────
        _tooltip(
            self.combo_font_family,
            (
                "<b>Tipografia das figuras</b><br/>"
                "Família tipográfica usada em títulos/labels. Cross-platform: 'DejaVu Sans'. "
                "macOS: 'Helvetica' / 'SF Pro'. Publicações LaTeX-like: 'CMU Serif'."
            ),
        )
        _tooltip(
            self.spin_lw,
            (
                "<b>Espessura da linha (pt)</b><br/>"
                "Largura das curvas de plot (magnitude, Re, Im, ρ). 1.0 = padrão matplotlib; "
                "1.6–2.0 = bom para apresentações."
            ),
        )
        _tooltip(
            self.spin_axis_prec,
            (
                "<b>Precisão dos ticks dos eixos</b><br/>"
                "Casas decimais dos rótulos de ticks (usa FormatStrFormatter). "
                "Ex.: 4 → '-0.1635'."
            ),
        )
        _tooltip(
            self.spin_spine_w,
            (
                "<b>Espessura das bordas do axes (spines)</b><br/>"
                "Largura das linhas de contorno do painel de plot. Padrão = 1.0."
            ),
        )
        _tooltip(
            self.spin_marker_size,
            (
                "<b>Tamanho dos marcadores</b><br/>"
                "Usado em scatter e comparações benchmark (Fortran = pontos). "
                "Default 4 pt; 6–8 para apresentações."
            ),
        )
        _tooltip(
            self.check_grid,
            (
                "<b>Grid de fundo</b><br/>"
                "Linhas pontilhadas horizontais/verticais auxiliando leitura. "
                "Transparência em grid_alpha = 0.35."
            ),
        )
        _tooltip(
            self.check_layer_boundary,
            (
                "<b>Interfaces das camadas</b><br/>"
                "Linhas horizontais tracejadas nos topos das interfaces do modelo geológico. "
                "Úteis para correlacionar H com a estratigrafia."
            ),
        )
        _tooltip(
            self.check_minor_ticks,
            (
                "<b>Ticks menores</b><br/>"
                "Ativa <code>xtick.minor.visible</code>/<code>ytick.minor.visible</code>. "
                "Leitura mais fina dos eixos sem sobrecarregar labels."
            ),
        )
        _tooltip(
            self.check_use_mathtext,
            (
                "<b>Mathtext nativo do matplotlib</b><br/>"
                "Renderiza $…$ com fonte pseudo-LaTeX (Computer Modern) sem dependência externa. "
                "Ativa por padrão — desative apenas se houver conflito com use_latex=True."
            ),
        )
        _tooltip(
            self.combo_palette,
            (
                "<b>Paleta para multi-curvas</b><br/>"
                "Paleta matplotlib usada quando há várias curvas (multi-freq × multi-TR × multi-dip). "
                "Qualitativas: tab10, Set1, Set2. Sequenciais: viridis, cividis, plasma."
            ),
        )
        _tooltip(
            self.combo_legend_loc,
            (
                "<b>Localização da legenda</b><br/>"
                "Posicionamento padrão das legendas nos plots. 'best' = matplotlib escolhe "
                "automaticamente o canto com menor sobreposição."
            ),
        )
        _tooltip(
            self.combo_title_loc,
            (
                "<b>Alinhamento horizontal do título</b><br/>"
                "center (padrão), left, right. Afeta todas as figuras."
            ),
        )
        _tooltip(
            self.combo_line_style,
            (
                "<b>Estilo da linha padrão</b><br/>"
                "'-' sólido, '--' tracejado, '-.' traço-ponto, ':' pontilhado."
            ),
        )
        _tooltip(
            self.combo_marker_style,
            (
                "<b>Estilo do marcador padrão</b><br/>"
                "o = círculo, s = quadrado, ^/v = triângulos, D = losango, x/+ = cruzes, * = estrela."
            ),
        )
        for btn, lbl in [
            (self.btn_color_real, "Re(H) — curvas de parte real do tensor EM"),
            (self.btn_color_imag, "Im(H) — curvas de parte imaginária do tensor EM"),
            (self.btn_color_rho_h, "ρₕ — resistividade horizontal nos perfis"),
            (self.btn_color_rho_v, "ρᵥ — resistividade vertical (tracejada)"),
            (self.btn_color_numba, "Numba — backend Python JIT em comparações"),
            (
                self.btn_color_fortran,
                "Fortran — backend tatu.x (marcadores) em comparações",
            ),
            (self.btn_color_layer, "Cor das linhas horizontais de interface de camada"),
            (self.btn_color_bg, "Cor de fundo do painel de plot (axes + figure)"),
        ]:
            _tooltip(btn, f"<b>Cor · {lbl}</b><br/>Clique para abrir o seletor de cores.")

        style_form.addRow("Resolução (DPI):", self.spin_dpi)
        style_form.addRow("Tipografia:", self.combo_font_family)
        style_form.addRow("Tamanho da fonte (pt):", self.spin_font)
        style_form.addRow("Espessura da linha:", self.spin_lw)
        style_form.addRow("Espessura dos eixos:", self.spin_spine_w)
        style_form.addRow("Tamanho do marcador:", self.spin_marker_size)
        style_form.addRow("Estilo da linha:", self.combo_line_style)
        style_form.addRow("Estilo do marcador:", self.combo_marker_style)
        style_form.addRow("Paleta multi-curva:", self.combo_palette)
        style_form.addRow("Localização da legenda:", self.combo_legend_loc)
        style_form.addRow("Alinhamento do título:", self.combo_title_loc)
        style_form.addRow("Casas decimais nos eixos:", self.spin_axis_prec)
        style_form.addRow("", self.check_grid)
        style_form.addRow("", self.check_layer_boundary)
        style_form.addRow("", self.check_minor_ticks)
        style_form.addRow("", self.check_use_latex)
        style_form.addRow("", self.check_use_mathtext)
        style_form.addRow("Cor Re(H):", self.btn_color_real)
        style_form.addRow("Cor Im(H):", self.btn_color_imag)
        style_form.addRow("Cor ρₕ:", self.btn_color_rho_h)
        style_form.addRow("Cor ρᵥ:", self.btn_color_rho_v)
        style_form.addRow("Cor Numba (comparação):", self.btn_color_numba)
        style_form.addRow("Cor Fortran (comparação):", self.btn_color_fortran)
        style_form.addRow("Cor das interfaces:", self.btn_color_layer)
        style_form.addRow("Cor de fundo do plot:", self.btn_color_bg)

        self.btn_save = QtWidgets.QPushButton("Salvar preferências")
        self.btn_save.setProperty("role", "primary")
        self.btn_save.clicked.connect(self._on_save)
        _tooltip(
            self.btn_save,
            (
                "Persiste caminhos + estilo em QSettings (escopo 'Geosteering AI / Simulation Manager'). "
                "Aplica imediatamente aos plots da aba Resultados."
            ),
        )
        self.btn_reset = QtWidgets.QPushButton("Restaurar padrões")
        self.btn_reset.clicked.connect(self._on_reset)
        _tooltip(
            self.btn_reset,
            (
                "Restaura o estilo de plot para os valores-padrão do dataclass PlotStyle "
                "(não altera caminhos). É necessário clicar em 'Salvar' para persistir."
            ),
        )

        row_btn = QtWidgets.QHBoxLayout()
        row_btn.addWidget(self.btn_save)
        row_btn.addWidget(self.btn_reset)
        row_btn.addStretch(1)

        root = QtWidgets.QVBoxLayout()
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)
        root.addWidget(_heading("Preferências"))
        root.addWidget(
            _heading(
                "Override manual dos caminhos (se a detecção automática falhar), "
                "personalização completa do estilo dos plots e opção de render LaTeX.",
                role="section",
            )
        )
        root.addWidget(grp_paths)
        root.addWidget(grp_style)
        root.addLayout(row_btn)
        root.addStretch(1)

        container = QtWidgets.QWidget()
        container.setLayout(root)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(_frame_shape("NoFrame"))
        scroll.setWidget(container)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.edit_pkg.textChanged.connect(self._validate_paths)
        self.edit_tatu.textChanged.connect(self._validate_paths)
        self.edit_python.textChanged.connect(self._validate_paths)
        self._validate_paths()

    def _color_btn(self, color: str) -> "QtWidgets.QPushButton":
        btn = QtWidgets.QPushButton(color)
        btn.setStyleSheet(
            f"background:{color}; color:{_color_for_bg(color)}; "
            "border:1px solid #3c3c3c; border-radius:4px; padding:4px 10px;"
        )
        btn.clicked.connect(lambda: self._pick_color(btn))
        return btn

    def _pick_color(self, btn: "QtWidgets.QPushButton") -> None:
        dlg = QtWidgets.QColorDialog(self)
        try:
            dlg.setCurrentColor(QtGui.QColor(btn.text()))
        except Exception:
            pass
        if dlg.exec():
            c = dlg.currentColor().name()
            btn.setText(c)
            btn.setStyleSheet(
                f"background:{c}; color:{_color_for_bg(c)}; "
                "border:1px solid #3c3c3c; border-radius:4px; padding:4px 10px;"
            )

    def _browse_file(self, edit: "QtWidgets.QLineEdit") -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Selecionar arquivo", edit.text() or str(Path.cwd())
        )
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: "QtWidgets.QLineEdit") -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Selecionar pasta", edit.text() or str(Path.cwd())
        )
        if path:
            edit.setText(path)

    def _auto_detect(self) -> None:
        pkg = find_geosteering_ai_root()
        tatu = find_tatu_binary()
        if pkg:
            self.edit_pkg.setText(str(pkg))
        if tatu:
            self.edit_tatu.setText(str(tatu))
        self.edit_python.setText(sys.executable)

    def _validate_paths(self) -> None:
        def _status(ok: bool, detail: str) -> str:
            color = "#3ecf5f" if ok else "#e5414e"
            mark = "✓" if ok else "✗"
            return f'<span style="color:{color};">{mark} {detail}</span>'

        pkg = Path(self.edit_pkg.text()) if self.edit_pkg.text() else None
        ok_pkg = bool(pkg and (pkg / "__init__.py").is_file())
        self.lbl_pkg_status.setText(
            _status(ok_pkg, "Pacote válido." if ok_pkg else "Pacote não encontrado.")
        )
        tatu = Path(self.edit_tatu.text()) if self.edit_tatu.text() else None
        ok_tatu = bool(tatu and tatu.is_file() and os.access(tatu, os.X_OK))
        self.lbl_tatu_status.setText(
            _status(
                ok_tatu,
                "Binário executável." if ok_tatu else "Binário ausente ou sem permissão.",
            )
        )
        py = Path(self.edit_python.text()) if self.edit_python.text() else None
        ok_py = bool(py and py.is_file())
        self.lbl_python_status.setText(
            _status(ok_py, "Interpretador válido." if ok_py else "Interpretador ausente.")
        )

    def _collect_style(self) -> PlotStyle:
        return PlotStyle(
            dpi=int(self.spin_dpi.value()),
            font_family=self.combo_font_family.currentText().strip() or "DejaVu Sans",
            font_size=int(self.spin_font.value()),
            line_width=float(self.spin_lw.value()),
            grid=self.check_grid.isChecked(),
            grid_alpha=0.35,
            color_real=self.btn_color_real.text(),
            color_imag=self.btn_color_imag.text(),
            color_mag=self.btn_color_real.text(),
            color_phase=self.btn_color_imag.text(),
            color_rho_h=self.btn_color_rho_h.text(),
            color_rho_v=self.btn_color_rho_v.text(),
            color_numba=self.btn_color_numba.text(),
            color_fortran=self.btn_color_fortran.text(),
            color_layer_boundary=self.btn_color_layer.text(),
            axis_precision=int(self.spin_axis_prec.value()),
            show_layer_boundaries=self.check_layer_boundary.isChecked(),
            palette=self.combo_palette.currentText(),
            background=self.btn_color_bg.text(),
            tight_layout=True,
            use_latex=self.check_use_latex.isChecked(),
            use_mathtext=self.check_use_mathtext.isChecked(),
            legend_location=self.combo_legend_loc.currentText(),
            title_location=self.combo_title_loc.currentText(),
            minor_ticks=self.check_minor_ticks.isChecked(),
            spine_width=float(self.spin_spine_w.value()),
            line_style=self.combo_line_style.currentText(),
            marker_size=float(self.spin_marker_size.value()),
            marker_style=self.combo_marker_style.currentText(),
        )

    def _collect_paths(self) -> Dict[str, str]:
        return {
            "geosteering_ai": self.edit_pkg.text().strip(),
            "tatu_binary": self.edit_tatu.text().strip(),
            "python_binary": self.edit_python.text().strip(),
            "output_dir": self.edit_output.text().strip(),
        }

    def _on_save(self) -> None:
        paths = self._collect_paths()
        style = self._collect_style()
        save_paths(paths)
        save_plot_style(style)
        self._paths = paths
        self._style = style
        apply_style(style)
        self.paths_changed.emit(paths)
        self.style_changed.emit(style)
        QtWidgets.QMessageBox.information(
            self, "Preferências", "Preferências salvas com sucesso."
        )

    def _on_reset(self) -> None:
        self._style = PlotStyle()
        self.spin_dpi.setValue(self._style.dpi)
        self.combo_font_family.setCurrentText(self._style.font_family)
        self.spin_font.setValue(self._style.font_size)
        self.spin_lw.setValue(self._style.line_width)
        self.spin_spine_w.setValue(self._style.spine_width)
        self.spin_marker_size.setValue(self._style.marker_size)
        self.spin_axis_prec.setValue(self._style.axis_precision)
        self.check_grid.setChecked(self._style.grid)
        self.check_layer_boundary.setChecked(self._style.show_layer_boundaries)
        self.check_minor_ticks.setChecked(self._style.minor_ticks)
        self.check_use_latex.setChecked(self._style.use_latex)
        self.check_use_mathtext.setChecked(self._style.use_mathtext)
        self.combo_palette.setCurrentText(self._style.palette)
        self.combo_legend_loc.setCurrentText(self._style.legend_location)
        self.combo_title_loc.setCurrentText(self._style.title_location)
        self.combo_line_style.setCurrentText(self._style.line_style)
        self.combo_marker_style.setCurrentText(self._style.marker_style)


# ══════════════════════════════════════════════════════════════════════════
# Aba unificada — Simulador (agrega Parâmetros na coluna esquerda)
# ══════════════════════════════════════════════════════════════════════════


class SimulatorTab(QtWidgets.QWidget):
    """Aba unificada com 3 colunas (v2.4b).

    Layout horizontal via ``QSplitter``:

    ┌──────────────────────┬─────────────────────────┬─────────────────────┐
    │ 1. Parâmetros da     │ 2. Simulador —          │ 3. Simulações        │
    │    Simulação         │    Execução Paralela    │    Realizadas        │
    │   (ParametersPage)   │   (SimulatorPage sem    │  (grp_history         │
    │                      │   grp_history)          │  reparentado aqui)   │
    └──────────────────────┴─────────────────────────┴─────────────────────┘

    Expõe ``self.params`` (ParametersPage) e ``self.sim`` (SimulatorPage)
    para acesso direto. O grupo de histórico é propriedade de ``self.sim``
    (signals/métodos permanecem), mas é visualmente colocado na 3ª coluna
    via ``SimulatorPage.take_history_group()``.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.params = ParametersPage()
        self.sim = SimulatorPage()

        # Coluna 3 — container para o histórico de simulações
        self._history_column = QtWidgets.QWidget()
        col3_layout = QtWidgets.QVBoxLayout(self._history_column)
        col3_layout.setContentsMargins(8, 8, 8, 8)
        col3_layout.setSpacing(8)
        col3_layout.addWidget(_heading("Simulações Realizadas"))
        col3_layout.addWidget(
            _heading(
                "Cada execução de 'Iniciar Simulação' é registrada aqui. "
                "Clique para ver parâmetros; duplo-clique reabre o tensor "
                "na aba Resultados.",
                role="section",
            )
        )
        # Move grp_history da SimulatorPage para esta 3ª coluna.
        history_group = self.sim.take_history_group()
        col3_layout.addWidget(history_group, 1)

        splitter = QtWidgets.QSplitter(ORIENT_H if ORIENT_H else 0x1)
        splitter.addWidget(self.params)
        splitter.addWidget(self.sim)
        splitter.addWidget(self._history_column)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setChildrenCollapsible(False)
        # Tamanhos iniciais proporcionais (Parâmetros : Execução : Histórico)
        splitter.setSizes([560, 680, 500])
        _tooltip(
            splitter,
            (
                "Arraste as divisórias para ajustar o espaço entre: "
                "Parâmetros da Simulação (esquerda) · Simulador Execução "
                "Paralela (centro) · Simulações Realizadas (direita)."
            ),
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter)

        # Atualização em tempo real de ``lbl_npos`` quando tj, p_med ou os
        # ângulos de dip mudam em ParametersPage.
        self.params.parametersChanged.connect(
            lambda: self.sim.update_npos_from_params(self.params)
        )
        # Primeira sincronização logo após o layout ser instanciado.
        self.sim.update_npos_from_params(self.params)

    # Proxies para manter compatibilidade com a API antiga do MainWindow
    # (``request_start``/``request_stop`` emitidos a partir da SimulatorPage
    # continuam funcionando naturalmente — basta que o MainWindow conecte à
    # ``self.page_sim.sim.request_start`` / ``.request_stop``).


# ══════════════════════════════════════════════════════════════════════════
# Janela principal
# ══════════════════════════════════════════════════════════════════════════


class MainWindow(QtWidgets.QMainWindow):
    """Janela principal com QTabWidget no topo + gating por experimento."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Simulation Manager — Geosteering AI v2.0 ({QT_BINDING})")
        self.setMinimumSize(1280, 820)

        self._paths = load_paths()
        self._plot_style = load_plot_style()
        apply_style(self._plot_style)
        self._experiment: Optional[ExperimentState] = None

        # Páginas — a aba "Simulador" une Parâmetros + Simulador em colunas.
        self.page_simulator = SimulatorTab()
        # Aliases retrocompatíveis: toda a lógica de negócio (save/restore,
        # start/stop sim, tooltips, dialogs) usa os nomes antigos.
        self.page_params = self.page_simulator.params
        self.page_sim = self.page_simulator.sim
        self.page_bench = BenchmarkPage()
        self.page_results = ResultsPage(self._plot_style)
        self.page_prefs = PreferencesPage(self._paths, self._plot_style)

        # Welcome widget (mostrado enquanto _experiment is None)
        self.welcome = WelcomeWidget()
        self.welcome.request_new.connect(self._on_new_experiment)
        self.welcome.request_open.connect(self._on_open_experiment)

        # QStackedWidget: index 0 = welcome, index 1 = tabs
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.welcome)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.page_simulator, "▶  Simulador")
        self.tabs.addTab(self.page_bench, "📊  Benchmark")
        self.tabs.addTab(self.page_results, "🖼  Resultados")
        self.tabs.addTab(self.page_prefs, "⚙  Preferências")
        self.stack.addWidget(self.tabs)

        self.setCentralWidget(self.stack)

        self._build_toolbar()

        # Status bar
        self.status = self.statusBar()
        self.lbl_status_exp = QtWidgets.QLabel("Sem experimento")
        self.lbl_status_throughput = QtWidgets.QLabel("Throughput: —")
        self.lbl_status_elapsed = QtWidgets.QLabel("Elapsed: —")
        self.lbl_status_backend = QtWidgets.QLabel(f"Binding: {QT_BINDING}")
        for lbl in (
            self.lbl_status_exp,
            self.lbl_status_throughput,
            self.lbl_status_elapsed,
            self.lbl_status_backend,
        ):
            lbl.setStyleSheet("padding:0 10px;")
        self.status.addPermanentWidget(self.lbl_status_exp)
        self.status.addPermanentWidget(self.lbl_status_throughput)
        self.status.addPermanentWidget(self.lbl_status_elapsed)
        self.status.addPermanentWidget(self.lbl_status_backend)
        self.status.showMessage("Crie ou abra um experimento para começar.")

        self._apply_paths(self._paths)

        self._sim_thread: Optional[SimulationThread] = None
        self._bench_thread: Optional[BenchmarkThread] = None
        # SaveArtifactsThread sobrevive ao retorno de _on_sim_finished —
        # armazenado como atributo para GC não destruir prematuramente.
        self._save_thread: Optional[SaveArtifactsThread] = None

        # v2.4: Cache em mem\u00f3ria de tensores H por snapshot_id. Permite
        # duplo-clique no hist\u00f3rico → recarregar plot em Resultados sem
        # re-executar. Perdido ao fechar o programa (apenas metadados v\u00e3o
        # para o .exp.json via ExperimentState.simulations).
        self._sim_history_cache: Dict[str, Dict[str, Any]] = {}

        self.page_sim.request_start.connect(self._start_simulation)
        self.page_sim.request_stop.connect(self._stop_simulation)
        self.page_sim.request_clear_simulations.connect(
            self._on_clear_simulations_requested
        )
        self.page_sim.request_history_select.connect(self._on_history_snapshot_selected)
        self.page_sim.request_history_open.connect(self._on_history_snapshot_open)
        self.page_bench.request_start.connect(self._start_benchmark)
        self.page_bench.request_stop.connect(self._stop_benchmark)
        self.page_prefs.paths_changed.connect(self._apply_paths)
        self.page_prefs.style_changed.connect(self._apply_style)

    def _build_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("main")
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(16, 16))
        QAction = getattr(QtGui, "QAction", None) or QtWidgets.QAction
        self.act_new = QAction("➕  Novo", self)
        self.act_new.triggered.connect(self._on_new_experiment)
        self.act_new.setToolTip(
            "Criar um novo experimento (.exp.json) — abre diálogo com nome, "
            "descrição e diretório."
        )
        self.act_open = QAction("📂  Abrir…", self)
        self.act_open.triggered.connect(self._on_open_experiment)
        self.act_open.setToolTip("Abrir um experimento existente (.exp.json).")
        self.act_save = QAction("💾  Salvar", self)
        self.act_save.triggered.connect(self._on_save_experiment)
        self.act_save.setEnabled(False)
        self.act_save.setToolTip(
            "Persistir o estado atual (parâmetros + simulador + benchmark) no "
            "arquivo do experimento aberto."
        )
        self.act_save_as = QAction("💾  Salvar como…", self)
        self.act_save_as.triggered.connect(self._on_save_experiment_as)
        self.act_save_as.setEnabled(False)
        self.act_save_as.setToolTip(
            "Salvar o experimento atual em um novo arquivo .exp.json."
        )
        self.act_close = QAction("✕  Fechar experimento", self)
        self.act_close.triggered.connect(self._on_close_experiment)
        self.act_close.setEnabled(False)
        self.act_close.setToolTip(
            "Fechar o experimento atual e voltar para a tela de boas-vindas. "
            "Oferece a opção de salvar antes."
        )
        toolbar.addAction(self.act_new)
        toolbar.addAction(self.act_open)
        toolbar.addSeparator()
        toolbar.addAction(self.act_save)
        toolbar.addAction(self.act_save_as)
        toolbar.addSeparator()
        toolbar.addAction(self.act_close)
        self.addToolBar(toolbar)

    # ── Experimento ──────────────────────────────────────────────────────

    def _unlock_tabs(self, exp: ExperimentState) -> None:
        self._experiment = exp
        self.stack.setCurrentIndex(1)
        self.act_save.setEnabled(True)
        self.act_save_as.setEnabled(True)
        self.act_close.setEnabled(True)
        self.lbl_status_exp.setText(f"Exp: {exp.name}")
        self.status.showMessage(f"Experimento carregado: {exp.file_path}")
        self.setWindowTitle(f"Simulation Manager — {exp.name} ({QT_BINDING})")
        _push_recent(exp.file_path)

        # v2.4: Popula a lista de hist\u00f3rico a partir do ExperimentState
        # (snapshots persistidos). Cache come\u00e7a vazio \u2014 duplo-clique avisa
        # o usu\u00e1rio que o tensor n\u00e3o est\u00e1 em mem\u00f3ria.
        self._sim_history_cache.clear()
        self.page_sim.clear_history_list()
        for snap in exp.simulations:
            self.page_sim.add_history_entry(snap.snapshot_id, snap.label, in_cache=False)

    def _on_new_experiment(self) -> None:
        dlg = NewExperimentDialog(self)
        if dlg.exec():
            exp = dlg.result_state()
            if exp is not None:
                # Estado inicial: coleta valores padrão de cada aba
                self._capture_parameters_into(exp)
                exp.save()
                self._unlock_tabs(exp)

    def _on_open_experiment(self) -> None:
        # Permitir clique em "Recentes" pela welcome
        preselected = getattr(self.welcome, "property_selected_path", None)
        if preselected:
            path = preselected
            self.welcome.property_selected_path = None
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Abrir experimento",
                str(Path.cwd()),
                "Experimento (*.exp.json);;JSON (*.json)",
            )
        if not path:
            return
        try:
            exp = ExperimentState.from_file(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Falha ao abrir", str(e))
            return
        self._restore_parameters_from(exp)
        self._unlock_tabs(exp)

    def _on_save_experiment(self) -> None:
        if self._experiment is None:
            return
        self._capture_parameters_into(self._experiment)
        try:
            self._experiment.save()
            self.status.showMessage(f"Salvo em {self._experiment.file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Falha ao salvar", str(e))

    def _on_save_experiment_as(self) -> None:
        if self._experiment is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Salvar experimento como",
            self._experiment.file_path or "experiment.exp.json",
            "Experimento (*.exp.json);;JSON (*.json)",
        )
        if not path:
            return
        self._capture_parameters_into(self._experiment)
        try:
            self._experiment.save(path)
            self.status.showMessage(f"Salvo em {path}")
            self.setWindowTitle(
                f"Simulation Manager — {self._experiment.name} ({QT_BINDING})"
            )
            _push_recent(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Falha ao salvar", str(e))

    def _on_close_experiment(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Fechar experimento",
            "Deseja salvar alterações antes de fechar?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
            return
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._on_save_experiment()
        self._experiment = None
        # v2.4: limpa cache de tens\u00f5es e UI do hist\u00f3rico ao fechar experimento
        self._sim_history_cache.clear()
        self.page_sim.clear_history_list()
        self.stack.setCurrentIndex(0)
        self.act_save.setEnabled(False)
        self.act_save_as.setEnabled(False)
        self.act_close.setEnabled(False)
        self.lbl_status_exp.setText("Sem experimento")
        self.setWindowTitle(f"Simulation Manager — Geosteering AI v2.0 ({QT_BINDING})")

    def _capture_parameters_into(self, exp: ExperimentState) -> None:
        exp.parameters = {
            "parameters": self.page_params.to_dict(),
            "simulator": self.page_sim.to_dict(),
            "benchmark": self.page_bench.to_dict(),
        }
        if not exp.output_dir and self.page_sim.edit_output_dir.text():
            exp.output_dir = self.page_sim.edit_output_dir.text()

    def _restore_parameters_from(self, exp: ExperimentState) -> None:
        params = exp.parameters or {}
        self.page_params.from_dict(params.get("parameters", {}))
        self.page_sim.from_dict(params.get("simulator", {}))
        self.page_bench.from_dict(params.get("benchmark", {}))
        if exp.output_dir:
            self.page_sim.edit_output_dir.setText(exp.output_dir)

    # ── Paths / Style ─────────────────────────────────────────────────────

    def _apply_paths(self, paths: Dict[str, str]) -> None:
        self._paths = paths
        self.page_sim.set_fortran_path(paths.get("tatu_binary", ""))
        self.page_sim.set_python_path(paths.get("python_binary", ""))
        self.page_sim.set_output_dir(paths.get("output_dir", ""))
        pkg_path = paths.get("geosteering_ai", "")
        if pkg_path and os.path.isdir(pkg_path):
            parent = str(Path(pkg_path).parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)

    def _apply_style(self, style: PlotStyle) -> None:
        self._plot_style = style
        self.page_results.set_style(style)

    def _fortran_binary(self) -> str:
        return self._paths.get("tatu_binary", "")

    # ── Simulação ─────────────────────────────────────────────────────────

    def _start_simulation(self) -> None:
        if self._sim_thread and self._sim_thread.isRunning():
            QtWidgets.QMessageBox.warning(
                self,
                "Simulação em curso",
                "Aguarde o término antes de iniciar outra.",
            )
            return
        # R2: proteção contra SaveArtifactsThread ainda em curso da última
        # simulação — iniciar nova agora corromperia o .dat em escrita.
        save_thread = getattr(self, "_save_thread", None)
        if save_thread is not None and save_thread.isRunning():
            QtWidgets.QMessageBox.warning(
                self,
                "Exportação em curso",
                "Aguarde a exportação dos artefatos da simulação anterior "
                "terminar antes de iniciar uma nova.",
            )
            return
        try:
            params = self.page_params
            sim = self.page_sim
            freqs = params.collect_freqs() or [20000.0]
            trs = params.collect_trs() or [1.0]
            dips = params.collect_angles() or [0.0]
            import math

            cos_d = max(1e-6, math.cos(math.radians(abs(dips[0]))))
            h1_val = float(params.spin_h1.value())
            tj_val = float(params.spin_tj.value())
            p_med_val = float(params.spin_pmed.value())
            n_pos = max(1, int(math.ceil(tj_val / (p_med_val * cos_d))))
            # Convenção Fortran / buildValidamodels.py: z_obs começa em −h1
            # (acima da 1ª interface) e termina em tj − h1 (abaixo da última
            # interface). Interfaces ficam em z ∈ [0, Σesp]. É A MESMA convenção
            # interna do Fortran (PerfilaAnisoOmp.f08:677) — o backend Numba
            # (simulate_multi) usa ``positions_z`` como posição do ponto-médio
            # T-R no referencial do modelo, então passar linspace(−h1, tj−h1)
            # alinha Numba ↔ Fortran ↔ buildValidamodels.py bit-a-bit.
            positions_z = np.linspace(-h1_val, tj_val - h1_val, n_pos, dtype=np.float64)
            sim.append_log(
                f"Pontos de medição por modelo: {n_pos} "
                f"(z ∈ [{-h1_val:.2f}, {tj_val - h1_val:.2f}] m, interfaces em z=0)"
            )

            # Hierarquia de geração (v2.4): Manual → Canônico → Estocástico.
            # Manual tem prioridade porque o usuário pode ter editado a tabela
            # (inclusive partindo de um perfil canônico pré-preenchido).
            n_models = max(1, int(params.spin_nmodels.value()))
            if params.has_manual_override():
                manual = params.get_manual_override() or {}
                models = [
                    {
                        "n_layers": int(manual.get("n_layers", 3)),
                        "thicknesses": list(manual.get("thicknesses", [])),
                        "rho_h": list(manual.get("rho_h", [])),
                        "rho_v": list(manual.get("rho_v", [])),
                    }
                    for _ in range(n_models)
                ]
                sim.append_log(
                    f"Perfil manual ativo — {int(manual.get('n_layers', 0))} "
                    f"camadas · {n_models} cópia(s) idêntica(s)."
                )
            elif params.has_canonical_override():
                # Se houver perfil canônico aplicado (e o usuário não está em
                # Manual), replica o canônico — preservado para compat com
                # fluxos que não passam pela tabela manual.
                from .sm_canonical_profiles import build_single_model_dict

                cm = params.get_canonical_override()
                single = build_single_model_dict(cm)
                models = [dict(single) for _ in range(n_models)]
                sim.append_log(
                    f"Perfil canônico ativo: {cm.title} — "
                    f"{n_models} cópia(s) idêntica(s)."
                )
            else:
                gen_cfg = params.build_gen_config()
                n_models = int(params.spin_nmodels.value())
                sim.append_log(
                    f"Gerando {n_models} modelos (generator={gen_cfg.generator})…"
                )
                models = generate_models(gen_cfg, n_models=n_models, rng_seed=42)
                sim.append_log(f"  → {len(models)} perfis prontos.")

            output_dir = sim.edit_output_dir.text().strip() or str(
                Path.cwd() / "sm_output"
            )
            os.makedirs(output_dir, exist_ok=True)
            fortran_bin = self._fortran_binary()
            if sim.combo_backend.currentText() == "fortran" and not os.path.isfile(
                fortran_bin
            ):
                QtWidgets.QMessageBox.critical(
                    self,
                    "tatu.x não encontrado",
                    f"O binário Fortran não foi encontrado em {fortran_bin!r}.\n"
                    "Defina o caminho em Preferências.",
                )
                return

            save_flag = sim.check_save_artifacts.isChecked()
            req = SimRequest(
                frequencies_hz=freqs,
                tr_spacings_m=trs,
                dip_degs=dips,
                positions_z=positions_z,
                backend=sim.combo_backend.currentText(),
                n_workers=int(sim.spin_workers.value()),
                n_threads=int(sim.spin_threads.value()),
                hankel_filter=params.combo_filter.currentText(),
                h1=float(params.spin_h1.value()),
                tj=float(params.spin_tj.value()),
                p_med=float(params.spin_pmed.value()),
                fortran_binary=fortran_bin,
                save_raw=save_flag,
                output_dir=output_dir,
                output_filename="sim_output",
            )

            self._sim_thread = SimulationThread(req, models, parent=self)
            self._sim_thread.log.connect(sim.append_log)
            self._sim_thread.progress_update.connect(self._on_sim_progress)
            self._sim_thread.finished_all.connect(
                lambda d: self._on_sim_finished(d, req, models)
            )
            self._sim_thread.error.connect(self._on_sim_error)
            # Reabilita o Start QUANDO o thread terminar (qualquer desfecho),
            # garantindo que parar + iniciar de novo funcione sempre.
            self._sim_thread.finished.connect(lambda: sim.set_running(False))
            sim.set_running(True)
            sim.progress.setValue(0)
            sim.append_log("Iniciando simulação…")
            self._sim_thread.start()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Erro", str(exc))
            self.page_sim.set_running(False)

    def _stop_simulation(self) -> None:
        if self._sim_thread and self._sim_thread.isRunning():
            self._sim_thread.request_stop()
            self.page_sim.append_log(
                "Solicitação de parada enviada (parada ocorrerá entre chunks)."
            )

    def _on_sim_progress(self, done: int, total: int, mod_h: float) -> None:
        pct = int(100.0 * done / max(total, 1))
        self.page_sim.progress.setValue(pct)
        self.page_sim.lbl_throughput.setText(
            f"Throughput: {format_float(mod_h, 0, thousands=True)} mod/h"
        )
        self.lbl_status_throughput.setText(
            f"Throughput: {format_float(mod_h, 0, thousands=True)} mod/h"
        )
        self.status.showMessage(
            f"{done}/{total} modelos · {format_float(mod_h, 0, thousands=True)} mod/h"
        )
        if done % 100 == 0 or done == total:
            self.page_sim.append_log(
                f"  Progresso: {done}/{total} modelos · "
                f"{format_float(mod_h, 0, thousands=True)} mod/h"
            )

    def _on_sim_finished(
        self,
        result: Dict[str, Any],
        req: "SimRequest",
        models: List[dict],
    ) -> None:
        self.page_sim.progress.setValue(100)
        elapsed = float(result.get("elapsed", 0.0))
        mod_h = float(result.get("throughput_mod_h", 0.0))
        self.page_sim.append_log(
            f"Simulação concluída em {format_float(elapsed, 1)}s — "
            f"{format_float(mod_h, 0, thousands=True)} mod/h"
        )
        self.lbl_status_elapsed.setText(f"Elapsed: {format_float(elapsed, 1)}s")

        # ── Exportação .dat/.out em SaveArtifactsThread (não bloqueia GUI) ──
        # ANTES: write_dat_from_tensor rodava aqui na main thread, causando
        # freeze de 15-60s para n_models=3000+. AGORA: delega para QThread
        # dedicada, main thread retorna imediatamente. Usuário pode navegar
        # entre abas e visualizar resultados enquanto I/O termina.
        try:
            if req.backend == "numba" and self.page_sim.check_save_artifacts.isChecked():
                H_stack = result.get("H_stack")
                z_obs = result.get("z_obs")
                if H_stack is not None and H_stack.size and z_obs is not None:
                    dat_path = os.path.join(req.output_dir, f"{req.output_filename}.dat")
                    out_path = os.path.join(req.output_dir, f"{req.output_filename}.out")
                    nmeds = compute_nmeds_per_angle(req.tj, req.p_med, req.dip_degs)
                    self._save_thread = SaveArtifactsThread(
                        dat_path=dat_path,
                        out_path=out_path,
                        H_stack=H_stack,
                        z_obs=z_obs,
                        n_dips=len(req.dip_degs),
                        n_freqs=len(req.frequencies_hz),
                        nmaxmodel=len(models),
                        angles=list(req.dip_degs),
                        freqs_hz=list(req.frequencies_hz),
                        nmeds_per_angle=list(nmeds),
                        parent=self,
                    )
                    self._save_thread.saved.connect(self._on_artifacts_saved)
                    self._save_thread.error.connect(self._on_artifacts_error)
                    self.page_sim.append_log(
                        "Exportando .dat/.out em background (GUI permanece responsiva)…"
                    )
                    self.status.showMessage("Exportando artefatos em background…")
                    self._save_thread.start()
                else:
                    self.page_sim.append_log(
                        "  [AVISO] Tensor vazio — .dat/.out não serão salvos."
                    )
        except Exception as exc:
            self.page_sim.append_log(f"  [AVISO] Falha ao iniciar exportação: {exc}")

        plot_bundle = {
            "backend": req.backend,
            "H_stack": result.get("H_stack"),
            "z_obs": result.get("z_obs"),
            "freqs": req.frequencies_hz,
            "trs": req.tr_spacings_m,
            "dips": req.dip_degs,
            "model": models[0] if models else {},
        }
        self.page_results.set_current_simulation(plot_bundle, models=models)
        self.tabs.setCurrentWidget(self.page_results)

        # ── v2.4: Registra snapshot no hist\u00f3rico do experimento ───────
        # O snapshot cont\u00e9m par\u00e2metros (leve, serializ\u00e1vel) e \u00e9 persistido
        # no .exp.json. O tensor H (pesado) fica apenas em cache de sess\u00e3o
        # via self._sim_history_cache[snapshot_id].
        try:
            self._append_simulation_snapshot(req, result, models, plot_bundle)
        except Exception as exc:
            self.page_sim.append_log(
                f"  [AVISO] Falha ao registrar snapshot no hist\u00f3rico: {exc}"
            )

    def _append_simulation_snapshot(
        self,
        req: Any,
        result: Dict[str, Any],
        models: List[dict],
        plot_bundle: Dict[str, Any],
    ) -> None:
        """Cria e adiciona um :class:`SimulationSnapshot` ao experimento."""
        import datetime
        import uuid as _uuid

        if self._experiment is None:
            # Sem experimento ativo (welcome screen) — n\u00e3o registra hist\u00f3rico
            return

        snap_id = str(_uuid.uuid4())
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Label curto para o QListWidget
        n_hist = len(self._experiment.simulations) + 1
        n_models = len(models)
        n_workers = int(
            getattr(req, "n_workers", 0) or int(self.page_sim.spin_workers.value())
        )
        n_threads = int(
            getattr(req, "n_threads", 0) or int(self.page_sim.spin_threads.value())
        )
        label = (
            f"#{n_hist} — {now} — {req.backend}, {n_models} mod, "
            f"{n_workers}w × {n_threads}t"
        )

        # params_snapshot = to_dict() da ParametersPage (leve, JSON-safe)
        try:
            params_snapshot = self.page_params.to_dict()
        except Exception:
            params_snapshot = {}

        # exec_params: par\u00e2metros derivados efetivamente usados
        exec_params: Dict[str, Any] = {
            "freqs_hz": list(req.frequencies_hz),
            "tr_spacings_m": list(req.tr_spacings_m),
            "dip_degs": list(req.dip_degs),
            "h1": float(self.page_params.spin_h1.value()),
            "tj": float(self.page_params.spin_tj.value()),
            "p_med": float(self.page_params.spin_pmed.value()),
            "n_pos": int(
                plot_bundle.get("z_obs").shape[0]
                if plot_bundle.get("z_obs") is not None
                else 0
            ),
            "throughput_mod_h": float(result.get("throughput_mod_h", 0.0)),
        }
        # Range da janela de medi\u00e7\u00e3o (somente m\u00edn/m\u00e1x, n\u00e3o todo o array)
        z_obs = plot_bundle.get("z_obs")
        if z_obs is not None and hasattr(z_obs, "shape") and z_obs.size:
            exec_params["z_range_m"] = [float(z_obs.min()), float(z_obs.max())]

        snap = SimulationSnapshot(
            snapshot_id=snap_id,
            timestamp=now,
            label=label,
            backend=req.backend,
            n_models=n_models,
            n_workers=n_workers,
            n_threads=n_threads,
            params_snapshot=params_snapshot,
            exec_params=exec_params,
            elapsed_s=float(result.get("elapsed", 0.0)),
            artifacts=None,  # preenchido pelo _on_artifacts_saved se aplic\u00e1vel
        )
        self._experiment.append_simulation(snap)
        # Cache do tensor H (sobrevive at\u00e9 o fim da sess\u00e3o)
        self._sim_history_cache[snap_id] = dict(plot_bundle)
        self.page_sim.add_history_entry(snap_id, snap.label, in_cache=True)

    # ── v2.4: Handlers de hist\u00f3rico ────────────────────────────────────
    def _on_clear_simulations_requested(self) -> None:
        """Handler do botão 'Limpar Simulações' — QMessageBox de confirmação."""
        if self._experiment is None:
            return
        n = len(self._experiment.simulations)
        if n == 0:
            QtWidgets.QMessageBox.information(
                self,
                "Sem simula\u00e7\u00f5es",
                "N\u00e3o h\u00e1 simula\u00e7\u00f5es registradas no experimento.",
            )
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Limpar simula\u00e7\u00f5es",
            f"Deseja apagar <b>{n}</b> simula\u00e7\u00e3o(\u00f5es) do experimento?<br/><br/>"
            "Esta a\u00e7\u00e3o remove o hist\u00f3rico persistido no .exp.json e o cache de "
            "tensores da sess\u00e3o atual. N\u00e3o pode ser desfeita.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._experiment.clear_simulations()
            self._sim_history_cache.clear()
            self.page_sim.clear_history_list()
            self.status.showMessage(
                f"Hist\u00f3rico de simula\u00e7\u00f5es limpo ({n} entrada(s) removida(s)).",
                5000,
            )

    def _on_history_snapshot_selected(self, snapshot_id: str) -> None:
        """Clique simples na lista — mostra parâmetros no painel lateral."""
        if self._experiment is None:
            return
        for snap in self._experiment.simulations:
            if snap.snapshot_id == snapshot_id:
                self.page_sim.set_history_info(snap.format_info())
                return

    def _on_history_snapshot_open(self, snapshot_id: str) -> None:
        """Duplo-clique \u2014 reabre tensor H em Resultados se em cache."""
        bundle = self._sim_history_cache.get(snapshot_id)
        if bundle is None:
            QtWidgets.QMessageBox.information(
                self,
                "Tensor n\u00e3o dispon\u00edvel",
                "Esta simula\u00e7\u00e3o foi carregada do .exp.json, mas o tensor H "
                "n\u00e3o est\u00e1 em mem\u00f3ria (cache perdido ao fechar o programa).<br/>"
                "Para visualiz\u00e1-la, re-execute com os mesmos par\u00e2metros.",
            )
            return
        self.page_results.set_current_simulation(bundle, models=[bundle.get("model", {})])
        self.tabs.setCurrentWidget(self.page_results)

    def _on_sim_error(self, msg: str) -> None:
        self.page_sim.set_running(False)
        self.page_sim.append_log(f"[ERRO] {msg}")
        QtWidgets.QMessageBox.critical(self, "Erro na simulação", msg[:800])

    def _on_artifacts_saved(self, info: Dict[str, Any]) -> None:
        """Handler do sinal SaveArtifactsThread.saved — executa na main thread."""
        elapsed = float(info.get("elapsed_s", 0.0))
        dat_path = info.get("dat", "")
        out_path = info.get("out", "")
        self.page_sim.append_log(
            f"  .dat salvo em {dat_path}  (I/O: {format_float(elapsed, 1)}s)"
        )
        self.page_sim.append_log(f"  .out salvo em {out_path}")
        self.status.showMessage("Exportação concluída.", 5000)

    def _on_artifacts_error(self, msg: str) -> None:
        """Handler do sinal SaveArtifactsThread.error."""
        self.page_sim.append_log(f"  [AVISO] Falha ao exportar artefatos: {msg[:400]}")
        self.status.showMessage("Falha na exportação de artefatos.", 8000)

    # ── Benchmark ─────────────────────────────────────────────────────────

    def _start_benchmark(self) -> None:
        if self._bench_thread and self._bench_thread.isRunning():
            QtWidgets.QMessageBox.warning(
                self,
                "Benchmark em curso",
                "Aguarde o término.",
            )
            return
        try:
            bt = self.page_bench
            params = self.page_params
            fortran_bin = self._fortran_binary()
            if bt.check_fortran.isChecked() and not os.path.isfile(fortran_bin):
                QtWidgets.QMessageBox.critical(
                    self,
                    "tatu.x não encontrado",
                    f"Binário Fortran não encontrado em {fortran_bin!r}.\n"
                    "Defina o caminho em Preferências.",
                )
                return
            output_dir = self.page_sim.edit_output_dir.text().strip() or str(
                Path.cwd() / "sm_output"
            )
            os.makedirs(output_dir, exist_ok=True)

            config_d_models = bt.selected_config_d_models()
            run_d = bt.check_cfg_d.isChecked() and len(config_d_models) > 0

            settings = BenchmarkSettings(
                run_config_a=bt.check_cfg_a.isChecked(),
                run_config_b=bt.check_cfg_b.isChecked(),
                run_config_c=bt.check_cfg_c.isChecked(),
                run_config_d=run_d,
                run_30k=bt.check_30k.isChecked(),
                models=bt.selected_models() or CANONICAL_BENCHMARK_MODELS,
                config_d_models=config_d_models,
                config_d_overrides=bt.config_d_overrides(),
                n_iter=int(bt.spin_n_iter.value()),
                fortran_subset=int(bt.spin_fort_subset.value()),
                n_workers_numba=int(bt.spin_workers_numba.value()),
                n_threads_numba=int(bt.spin_threads_numba.value()),
                n_workers_fortran=int(bt.spin_workers_fortran.value()),
                n_threads_fortran=int(bt.spin_threads_fortran.value()),
                fortran_binary=fortran_bin,
                run_numba=bt.check_numba.isChecked(),
                run_fortran=bt.check_fortran.isChecked(),
                hankel_filter=params.combo_filter.currentText(),
                h1=float(params.spin_h1.value()),
                tj=float(params.spin_tj.value()),
                p_med=float(params.spin_pmed.value()),
                output_dir=output_dir,
            )

            self._bench_thread = BenchmarkThread(settings, parent=self)
            self._bench_thread.log.connect(bt.append_log)
            self._bench_thread.progress.connect(self._on_bench_progress)
            self._bench_thread.finished_all.connect(self._on_bench_finished)
            self._bench_thread.error.connect(self._on_bench_error)
            self._bench_thread.finished.connect(lambda: bt.set_running(False))
            bt.set_running(True)
            bt.progress.setValue(0)
            bt.lbl_current.setText("iniciando…")
            self._bench_thread.start()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Erro", str(exc))
            self.page_bench.set_running(False)

    def _stop_benchmark(self) -> None:
        if self._bench_thread and self._bench_thread.isRunning():
            self.page_bench.append_log(
                "Parada solicitada — aguardando célula atual terminar…"
            )

    def _on_bench_progress(self, i: int, n: int, label: str) -> None:
        pct = int(100.0 * i / max(n, 1))
        self.page_bench.progress.setValue(pct)
        self.page_bench.lbl_current.setText(label)
        self.status.showMessage(f"Benchmark {i}/{n} — {label}")

    def _on_bench_finished(self, records: list, plot_data: dict) -> None:
        self.page_bench.progress.setValue(100)
        self.page_bench.append_log(f"Benchmark concluído — {len(records)} células.")
        self.page_bench.populate_table(records)
        self.page_results.set_benchmark_plot_data(plot_data)

    def _on_bench_error(self, msg: str) -> None:
        self.page_bench.set_running(False)
        self.page_bench.append_log(f"[ERRO] {msg}")
        QtWidgets.QMessageBox.critical(self, "Erro no benchmark", msg[:800])


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════


def _run_smoke_test() -> int:
    """Smoke test offscreen: valida estrutura da UI sem mostrar a janela.

    Executado via ``--smoke-test`` (ou env ``QT_QPA_PLATFORM=offscreen``).
    Não abre diálogo de novo experimento; apenas checa:

      * 4 abas presentes (Simulador / Benchmark / Resultados / Preferências)
      * ``SimulatorTab`` expõe aliases ``self.params`` e ``self.sim``
      * ``combo_canonical`` tem 6 itens
      * ``lbl_npos`` é atualizado ao mudar ``spin_pmed``
      * ``SaveArtifactsThread`` é instanciável (sem .start())

    Retorna 0 em sucesso, 1 em falha. Imprime relatório no stdout.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    check_qt_available()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Simulation Manager (smoke)")
    app.setStyleSheet(APP_STYLESHEET)
    enforce_c_locale(app)

    failures: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if cond:
            print(f"  [OK]   {msg}")
        else:
            print(f"  [FAIL] {msg}")
            failures.append(msg)

    print("=== Simulation Manager — Smoke Test ===")
    window = MainWindow()

    # 4 abas presentes
    tab_texts = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    check(
        len(tab_texts) == 4,
        f"4 abas presentes (got {len(tab_texts)}): {tab_texts}",
    )
    expected_tabs = ["Simulador", "Benchmark", "Resultados", "Preferências"]
    for exp in expected_tabs:
        check(
            any(exp in t for t in tab_texts),
            f"Aba '{exp}' está presente",
        )

    # Aliases retrocompatíveis
    check(
        hasattr(window, "page_params")
        and window.page_params is window.page_simulator.params,
        "Alias window.page_params aponta para SimulatorTab.params",
    )
    check(
        hasattr(window, "page_sim") and window.page_sim is window.page_simulator.sim,
        "Alias window.page_sim aponta para SimulatorTab.sim",
    )

    # combo_canonical com 8 itens (Padrão + 6 de buildValidamodels.py + Viking Graben 10)
    combo = window.page_params.combo_canonical
    check(
        combo.count() == 8,
        f"combo_canonical tem 8 itens (got {combo.count()})",
    )

    # Botão Aplicar perfil presente
    check(
        hasattr(window.page_params, "btn_apply_canonical"),
        "Botão 'Aplicar perfil' está presente",
    )

    # lbl_npos inicial e reatividade
    window.page_sim.update_npos_from_params(window.page_params)
    initial_npos = window.page_sim.lbl_npos.text()
    check(
        "Pontos de medição" in initial_npos,
        f"lbl_npos contém texto inicial: {initial_npos[:60]}…",
    )
    window.page_params.spin_pmed.setValue(0.4)
    # Signal parametersChanged deve ter disparado update_npos_from_params
    updated_npos = window.page_sim.lbl_npos.text()
    check(
        updated_npos != initial_npos,
        "lbl_npos atualiza ao mudar p_med (reatividade)",
    )

    # SaveArtifactsThread instanciável
    try:
        import numpy as _np

        from .sm_workers import SaveArtifactsThread as _SAT

        _sat = _SAT(
            dat_path="/tmp/_smoke.dat",
            out_path="/tmp/_smoke.out",
            H_stack=_np.empty((0,), dtype=_np.complex128),
            z_obs=_np.empty((0,), dtype=_np.float64),
            n_dips=1,
            n_freqs=1,
            nmaxmodel=1,
            angles=[0.0],
            freqs_hz=[20000.0],
            nmeds_per_angle=[600],
        )
        check(True, "SaveArtifactsThread instancia sem erro")
        del _sat
    except Exception as exc:
        check(False, f"SaveArtifactsThread instancia sem erro ({exc})")

    # Aplicar perfil canônico
    try:
        window.page_params.combo_canonical.setCurrentIndex(1)  # Oklahoma 3
        window.page_params._apply_canonical_profile()
        check(
            window.page_params.has_canonical_override(),
            "Aplicar Oklahoma 3 define canonical_override",
        )
        override = window.page_params.get_canonical_override()
        check(
            override is not None and int(override.n_layers) == 3,
            f"canonical_override.n_layers == 3 (got {int(override.n_layers)})",
        )
    except Exception as exc:
        check(False, f"Aplicar perfil canônico funciona ({exc})")

    # ── v2.4 checks ─────────────────────────────────────────────────────
    # radio Aleatória / Manual
    try:
        check(
            hasattr(window.page_params, "radio_random")
            and hasattr(window.page_params, "radio_manual"),
            "radio_random / radio_manual presentes (v2.4)",
        )
        check(
            hasattr(window.page_params, "_manual_layers"),
            "_manual_layers attr presente (v2.4)",
        )
        # Após aplicar canônico (acima), deve ter virado 'Manual' com tabela populada
        check(
            window.page_params.radio_manual.isChecked(),
            "radio_manual auto-ativa ao aplicar canônico (v2.4)",
        )
        check(
            window.page_params.has_manual_override(),
            "has_manual_override() == True após canônico (v2.4)",
        )
    except Exception as exc:
        check(False, f"radio/manual override (v2.4) ({exc})")

    # LayersManualDialog instanciável
    try:
        from .sm_layers_dialog import LayersManualDialog as _LMD

        _dlg = _LMD(
            parent=window,
            initial={
                "n_layers": 3,
                "thicknesses": [8.0],
                "rho_h": [1.0, 10.0, 100.0],
                "rho_v": [2.0, 20.0, 200.0],
            },
        )
        check(True, "LayersManualDialog instancia com 3 camadas (v2.4)")
        _dlg.close()
    except Exception as exc:
        check(False, f"LayersManualDialog instancia ({exc})")

    # Checkboxes h1/tj automático em 2 linhas (legenda em QLabel separado)
    try:
        check(
            hasattr(window.page_params, "_lbl_auto_h1_caption")
            and hasattr(window.page_params, "_lbl_auto_tj_caption"),
            "Legendas word-wrap dos checkboxes h1/tj presentes (v2.4)",
        )
        check(
            window.page_params.check_auto_h1.text() == "h1 automático",
            f"QCheckBox h1 texto curto: '{window.page_params.check_auto_h1.text()}'",
        )
    except Exception as exc:
        check(False, f"Checkbox h1/tj word-wrap ({exc})")

    # Botão "Limpar Simulações" presente
    try:
        check(
            hasattr(window.page_sim, "btn_clear_sims"),
            "btn_clear_sims presente no SimulatorPage (v2.4)",
        )
        check(
            hasattr(window.page_sim, "sim_history_list")
            and hasattr(window.page_sim, "sim_history_info"),
            "sim_history_list / sim_history_info presentes (v2.4)",
        )
    except Exception as exc:
        check(False, f"Histórico UI ({exc})")

    # SimulationSnapshot + ExperimentState.simulations
    try:
        from .simulation_manager import SimulationSnapshot as _Snap

        _snap = _Snap(
            snapshot_id="test-id",
            timestamp="2026-04-20 10:00:00",
            label="#1 — test",
            backend="numba",
            n_models=100,
            n_workers=1,
            n_threads=1,
        )
        info = _snap.format_info()
        check(
            "test-id"[:8] in info and "numba" in info,
            "SimulationSnapshot.format_info() renderiza corretamente (v2.4)",
        )
    except Exception as exc:
        check(False, f"SimulationSnapshot ({exc})")

    # ResultsPage — default Re+Im
    try:
        check(
            window.page_results.combo_kind_mode.currentText() == "Re + Im",
            f"combo_kind_mode default == 'Re + Im' "
            f"(got {window.page_results.combo_kind_mode.currentText()!r}) (v2.4)",
        )
        # Filtros TR/ang/freq/geosinais
        check(
            hasattr(window.page_results, "list_tr_filter")
            and hasattr(window.page_results, "list_ang_filter")
            and hasattr(window.page_results, "list_freq_filter")
            and hasattr(window.page_results, "list_geo_filter"),
            "ResultsPage tem 4 filtros: TR / Ângulo / Frequência / Geosinais (v2.4)",
        )
        # Geosinais devem conter USD, UAD, UHR, UHA, U3DF
        geo_items = [
            window.page_results.list_geo_filter.item(i).text()
            for i in range(window.page_results.list_geo_filter.count())
        ]
        check(
            geo_items == ["USD", "UAD", "UHR", "UHA", "U3DF"],
            f"Geosinais disponíveis == USD/UAD/UHR/UHA/U3DF (got {geo_items}) (v2.4)",
        )
    except Exception as exc:
        check(False, f"ResultsPage v2.4 ({exc})")

    # U3DF helper em sm_plots
    try:
        import numpy as _np

        from .sm_plots import GEOSIGNAL_NAMES as _GN
        from .sm_plots import _compute_geosignal as _cg

        check(
            tuple(_GN) == ("USD", "UAD", "UHR", "UHA", "U3DF"),
            f"GEOSIGNAL_NAMES == 5 esperados (got {_GN}) (v2.4)",
        )
        H_dummy = _np.ones((10, 9), dtype=_np.complex128)
        u3df = _cg("U3DF", H_dummy)
        check(
            _np.all(_np.isfinite(u3df)),
            "U3DF compute retorna array finito (v2.4)",
        )
    except Exception as exc:
        check(False, f"U3DF helper ({exc})")

    # ── v2.4b checks ─────────────────────────────────────────────────────
    # _rho_per_z — bit-exato a buildValidamodels.py (Oklahoma 3)
    try:
        import numpy as _np

        from .sm_plots import _rho_per_z as _rpz

        # Oklahoma 3: esp=[2.4384], rho_h=[1, 20, 1]
        z = _np.array([-10.0, -0.01, 0.0, 1.22, 2.4384, 2.5, 10.0])
        rho = _rpz(z, [1.0, 20.0, 1.0], [2.4384])
        expected = _np.array([1.0, 1.0, 20.0, 20.0, 1.0, 1.0, 1.0])
        check(
            _np.allclose(rho, expected),
            f"_rho_per_z Oklahoma 3: z=[{list(z)}] → ρ_h={list(rho)} (expected {list(expected)})",
        )
    except Exception as exc:
        check(False, f"_rho_per_z Oklahoma 3 ({exc})")

    # Layout 3-colunas — histórico na 3ª coluna do SimulatorTab
    try:
        sim_tab = window.page_simulator
        check(
            hasattr(sim_tab, "_history_column"),
            "SimulatorTab tem 3ª coluna _history_column (v2.4b)",
        )
        # grp_history deve ter como ancestral o _history_column, NÃO o SimulatorPage root
        grp = window.page_sim.grp_history
        check(
            grp.parent() is not None,
            "grp_history está reparentado (v2.4b)",
        )
        # Valida que o splitter de SimulatorTab tem 3 widgets
        splitter = sim_tab.findChild(QtWidgets.QSplitter)
        check(
            splitter is not None and splitter.count() >= 3,
            f"QSplitter da SimulatorTab tem 3+ widgets (got {splitter.count() if splitter else 0}) (v2.4b)",
        )
    except Exception as exc:
        check(False, f"Layout 3-colunas ({exc})")

    print(f"\n=== Resultado: {len(failures)} falha(s) ===")
    window.close()
    return 1 if failures else 0


def main() -> int:
    if "--smoke-test" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--smoke-test"]
        return _run_smoke_test()

    check_qt_available()
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    if QtCore is not None:
        enforce_c_locale(None)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Simulation Manager")
    app.setOrganizationName("Geosteering AI")
    app.setStyleSheet(APP_STYLESHEET)
    enforce_c_locale(app)

    window = MainWindow()
    window.showMaximized()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    sys.exit(main())
