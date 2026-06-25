# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_gui_ui_files.py                                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — arquivos .ui do Qt Designer (camada View)            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-06                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Guarda os arquivos .ui (Qt Designer/Qt Creator) editáveis: garante que ║
# ║    (1) ``qt_compat.load_ui`` carrega o .ui no binding ativo e (2) o        ║
# ║    CONTRATO de binding (objectNames que a View fina consome) está intacto. ║
# ║    Editar o .ui no Designer e renomear/remover um objectName quebra a View ║
# ║    silenciosamente — este teste falha cedo nesse caso.                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes dos arquivos .ui (Qt Designer) — carregamento via load_ui + contrato de objectNames."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from geosteering_ai.gui.qt_compat import QT_AVAILABLE, QtWidgets, load_ui

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_UI = (
    PROJECT_ROOT
    / "apps"
    / "sim_manager"
    / "perspectives"
    / "simulation"
    / "simulator.ui"
)

# objectNames que a View fina usará como contrato de binding (ler valores/conectar
# sinais). Renomear/remover algum no Designer quebra a futura View → teste falha.
_BINDING_CONTRACT = (
    # geometria & aquisição
    "freqsEdit",
    "dipsEdit",
    "trsEdit",
    "h1Spin",
    "tjSpin",
    "pMedSpin",
    "nModelsSpin",
    "nPosLabel",
    "freqFixedRadio",
    "hankelFactorRadio",
    # backend
    "backendCombo",
    # geologia estocástica
    "geoModeCombo",
    "geoGeneratorCombo",
    "geoDistrCombo",
    "geoRhoMinSpin",
    "geoRhoMaxSpin",
    "geoAnisoCheck",
    "geoLambdaMinSpin",
    "geoLambdaMaxSpin",
    "geoNlfCheck",
    "geoNlfSpin",
    "geoNlMinSpin",
    "geoNlMaxSpin",
    "geoMinThickSpin",
    "geoSeedRandomCheck",
    "geoSeedSpin",
    # design-alvo (fatias 6a-6h) — já no .ui p/ desenhar a tela completa
    "canonicalCombo",
    "applyCanonicalButton",
    "genRandomRadio",
    "genManualRadio",
    "editLayersButton",
    "hankelFilterCombo",
    "hankelAutoRadio",
    "hankelOrderSpin",
    "hankelPathEdit",
    # auto-geometria canônica (Lote 2) — h1/tj separados (substitui autoGeoCheck)
    "h1AutoCheck",
    "tjAutoCheck",
    # paralelismo (Lote 1) — par ÚNICO workers/threads (PR-2: Fortran removido) + saída
    "workersSpin",
    "threadsSpin",
    "cpuInfoLabel",
    "parallelWarnLabel",
    "saveArtifactsCheck",
    # contadores derivados (Lote 1)
    "nfLabel",
    "nthetaLabel",
    "ntrPairsLabel",
    "runButton",
    "pauseButton",
    "resumeButton",
    "cancelButton",
    "saveButton",
    "openButton",
    "statusLabel",
    "progressBar",
    "outputDirEdit",
    "browseOutButton",
    "logEdit",
    # experimentos & histórico (Fatia 6c)
    "expNewButton",
    "expOpenButton",
    "expSaveButton",
    "expCloseButton",
    "expClearButton",
    "historyList",
    "historySearchEdit",
    # PR-2 (#1): a galeria de resultados (plotBackendCombo/plotModeCombo/channelCombo/
    # plotKindCombo/animPlayButton/animSlider/animSpeedCombo) SAIU do simulador para a
    # perspectiva Resultados (ResultsView), então não é mais contrato do simulator.ui.
)

pytestmark = pytest.mark.skipif(
    not QT_AVAILABLE, reason="requer binding Qt6 (PyQt6/PySide6)"
)


@pytest.mark.gui
def test_simulator_ui_file_exists():
    """O arquivo simulator.ui existe e é XML não-vazio."""
    assert SIMULATOR_UI.is_file(), f"não encontrado: {SIMULATOR_UI}"
    text = SIMULATOR_UI.read_text(encoding="utf-8")
    assert text.lstrip().startswith("<?xml"), "deve começar com declaração XML"
    assert "<class>SimulatorForm</class>" in text


@pytest.mark.gui
def test_simulator_ui_loads_via_load_ui(qapp):
    """``load_ui`` carrega simulator.ui e devolve a widget de topo (QWidget)."""
    ui = load_ui(SIMULATOR_UI)
    assert ui is not None
    assert isinstance(ui, QtWidgets.QWidget)


@pytest.mark.gui
def test_simulator_ui_binding_contract_intact(qapp):
    """Todos os objectNames do contrato existem no .ui (Designer não quebrou a View)."""
    ui = load_ui(SIMULATOR_UI)
    missing = [
        name
        for name in _BINDING_CONTRACT
        if ui.findChild(QtWidgets.QWidget, name) is None
    ]
    assert missing == [], f"objectNames ausentes no simulator.ui: {missing}"


@pytest.mark.gui
def test_simulator_ui_backend_combo_options(qapp):
    """O combo de backend oferece numba/jax/auto (fortran é design-alvo)."""
    ui = load_ui(SIMULATOR_UI)
    combo = ui.findChild(QtWidgets.QComboBox, "backendCombo")
    assert combo is not None
    items = [combo.itemText(i) for i in range(combo.count())]
    for backend in ("numba", "jax", "auto"):
        assert backend in items, f"backend {backend!r} ausente em {items}"
