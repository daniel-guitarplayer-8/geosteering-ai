# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_datviewer.py                                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Fatia 6h — perspectiva Visualizador .dat/.out (SM MVVM)                  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM MVVM — leitura somente de artefatos Fortran-compat       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-16                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre a Fatia 6h: serviço de leitura (.dat binário 22-col + fallback     ║
# ║    ASCII + .out paralelo + degradação graciosa), legenda 22-col, ViewModel  ║
# ║    PURO (open/loaded/load_error/summary) e a View Qt (tabela, truncamento,  ║
# ║    metadados, erro) + a perspectiva plugável. SOMENTE LEITURA — zero física.║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da Fatia 6h — Visualizador .dat/.out (serviço + ViewModel + View Qt)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apps.sim_manager.perspectives.datviewer.service import (  # noqa: E402
    DatViewerService,
    column_labels,
)
from apps.sim_manager.perspectives.datviewer.viewmodel import (  # noqa: E402
    DatViewerViewModel,
)


def _write_binary_dat(path, n_rows: int = 5) -> None:
    """Escreve um .dat binário 22-col (1 int32 + 21 float64 = 172 B/registro)."""
    dtype = np.dtype(
        [("col0", np.int32)] + [(f"col{i}", np.float64) for i in range(1, 22)]
    )
    buf = np.zeros(n_rows, dtype=dtype)
    buf["col0"] = np.arange(1, n_rows + 1, dtype=np.int32)
    for i in range(1, 22):
        buf[f"col{i}"] = np.arange(n_rows, dtype=np.float64) + i * 0.5
    buf.tofile(str(path))


# ════════════════════════════════════════════════════════════════════════════
# column_labels — legenda 22-col canônica vs genérica
# ════════════════════════════════════════════════════════════════════════════
def test_column_labels_22col_canonical():
    labels = column_labels(22)
    assert len(labels) == 22
    assert labels[0] == "meds"
    assert labels[1] == "zobs (m)"
    assert labels[2] == "res_h (Ω·m)"
    assert labels[3] == "res_v (Ω·m)"
    assert labels[4] == "Re(Hxx)" and labels[5] == "Im(Hxx)"
    assert labels[20] == "Re(Hzz)" and labels[21] == "Im(Hzz)"


def test_column_labels_generic_for_other_widths():
    labels = column_labels(3)
    assert labels == ["col0", "col1", "col2"]


# ════════════════════════════════════════════════════════════════════════════
# DatViewerService — leitura binária/ASCII + .out + degradação graciosa
# ════════════════════════════════════════════════════════════════════════════
def test_service_loads_binary_22col(tmp_path):
    p = tmp_path / "sm_output.dat"
    _write_binary_dat(p, n_rows=5)
    res = DatViewerService().load(p)
    assert res.error is None
    assert res.fmt == "binário 22-col"
    assert res.data.shape == (5, 22)
    assert list(res.data[:, 0].astype(int)) == [1, 2, 3, 4, 5]  # col0 = model_id
    assert res.out_metadata is None  # sem .out paralelo


def test_service_reads_out_sibling(tmp_path):
    p = tmp_path / "sm_output.dat"
    _write_binary_dat(p, n_rows=3)
    (tmp_path / "sm_output.out").write_text("nmaxmodel=3\nnf=1\n", encoding="utf-8")
    res = DatViewerService().load(p)
    assert res.out_metadata is not None
    assert "nmaxmodel=3" in res.out_metadata


def test_service_missing_file_returns_error(tmp_path):
    res = DatViewerService().load(tmp_path / "nao_existe.dat")
    assert res.data is None
    assert res.error and "não encontrado" in res.error  # não levanta


def test_service_ascii_fallback(tmp_path):
    p = tmp_path / "ascii.dat"
    np.savetxt(str(p), np.arange(15, dtype=float).reshape(5, 3))
    res = DatViewerService().load(p)
    assert res.error is None
    assert res.fmt == "ASCII"
    assert res.n_cols == 3 and res.n_rows == 5


def test_service_unreadable_returns_error(tmp_path):
    """Lixo não-binário-22 e não-ASCII → error setado (sem crash)."""
    p = tmp_path / "garbage.dat"
    p.write_bytes(b"\x00\x01\x02 not a matrix \xff")
    res = DatViewerService().load(p)
    # binário falha (não múltiplo de 172) E ASCII falha (não numérico) → erro
    assert res.data is None and res.error


def test_service_single_scalar_ascii_no_crash(tmp_path):
    """Regressão (revisão 6h): .dat ASCII com UM escalar → (1,1), NÃO crasha.

    np.loadtxt devolvia array 0-D; o load() levantava IndexError (violando o
    contrato "nunca levanta"). ndmin=2 corrige → (1,1).
    """
    p = tmp_path / "scalar.dat"
    p.write_text("42.0\n", encoding="utf-8")
    res = DatViewerService().load(p)
    assert res.error is None
    assert res.fmt == "ASCII"
    assert res.data.shape == (1, 1)


def test_service_single_column_ascii_keeps_orientation(tmp_path):
    """ASCII de 1 coluna × N linhas → (N,1) (não (1,N)). ndmin=2 preserva orientação."""
    p = tmp_path / "col.dat"
    p.write_text("1\n2\n3\n", encoding="utf-8")
    res = DatViewerService().load(p)
    assert res.error is None
    assert res.n_rows == 3 and res.n_cols == 1


def test_service_empty_dat_is_error(tmp_path):
    """Regressão (revisão 6h): .dat 0-byte → erro explícito (não 'sucesso' 0×22)."""
    p = tmp_path / "empty.dat"
    p.write_bytes(b"")
    res = DatViewerService().load(p)
    assert res.data is None and res.error and "sem registros" in res.error


def test_service_whitespace_only_is_error(tmp_path):
    """ASCII só com whitespace → erro (não linha-fantasma)."""
    p = tmp_path / "ws.dat"
    p.write_text("   \n  \n", encoding="utf-8")
    res = DatViewerService().load(p)
    assert res.data is None and res.error


def test_looks_binary_discriminates_content(tmp_path):
    """_looks_binary distingue por CONTEÚDO (resolve colisão 'tamanho múltiplo 172').

    Um .dat ASCII cujo tamanho casualmente fosse múltiplo de 172 era lido como
    binário 22-col (lixo silencioso). A detecção por NUL/não-imprimível corrige.
    """
    svc = DatViewerService()
    bin_p = tmp_path / "b.dat"
    _write_binary_dat(bin_p, n_rows=5)
    assert svc._looks_binary(bin_p) is True  # int32+float64 tem NUL

    # ASCII numérico com tamanho EXATAMENTE múltiplo de 172 bytes (preenchido).
    ascii_p = tmp_path / "a.dat"
    body = "1 2 3\n"
    pad = 172 - (len(body.encode()) % 172)
    ascii_p.write_text(body + " " * (pad - 1) + "\n", encoding="utf-8")
    assert ascii_p.stat().st_size % 172 == 0  # casaria com a detecção-por-tamanho
    assert svc._looks_binary(ascii_p) is False  # mas o conteúdo é ASCII
    res = svc.load(ascii_p)
    assert res.fmt == "ASCII"  # lido como ASCII, NÃO como binário-lixo


# ════════════════════════════════════════════════════════════════════════════
# DatViewerViewModel — open/loaded/load_error/summary (PURO)
# ════════════════════════════════════════════════════════════════════════════
def test_vm_open_success_emits_loaded(tmp_path):
    p = tmp_path / "sm_output.dat"
    _write_binary_dat(p, n_rows=4)
    vm = DatViewerViewModel(service=DatViewerService())
    loaded, errors = [], []
    vm.loaded.connect(loaded.append)
    vm.load_error.connect(errors.append)
    vm.open(str(p))
    assert errors == []
    assert len(loaded) == 1 and loaded[0].n_rows == 4
    assert vm.result is not None and vm.result.n_cols == 22
    assert vm.dat_path == str(p)


def test_vm_open_missing_emits_error(tmp_path):
    vm = DatViewerViewModel(service=DatViewerService())
    loaded, errors = [], []
    vm.loaded.connect(loaded.append)
    vm.load_error.connect(errors.append)
    vm.open(str(tmp_path / "nope.dat"))
    assert loaded == []
    assert len(errors) == 1
    assert vm.result is not None and vm.result.error


def test_vm_summary(tmp_path):
    vm = DatViewerViewModel(service=DatViewerService())
    assert "Nenhum arquivo" in vm.summary()
    p = tmp_path / "sm_output.dat"
    _write_binary_dat(p, n_rows=2)
    vm.open(str(p))
    s = vm.summary()
    assert "2 linhas" in s and "22 col" in s and "binário 22-col" in s


# ════════════════════════════════════════════════════════════════════════════
# DatViewerPanel — View Qt (tabela, truncamento, metadados, erro) (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_panel_populates_table_on_load(qtbot, tmp_path):
    from apps.sim_manager.perspectives.datviewer.view import DatViewerPanel

    p = tmp_path / "sm_output.dat"
    _write_binary_dat(p, n_rows=6)
    (tmp_path / "sm_output.out").write_text("meta=ok", encoding="utf-8")
    vm = DatViewerViewModel(service=DatViewerService())
    panel = DatViewerPanel(vm)
    qtbot.addWidget(panel)
    vm.open(str(p))  # dispara _on_loaded via signal
    assert panel._table.rowCount() == 6
    assert panel._table.columnCount() == 22
    assert panel._table.horizontalHeaderItem(0).text() == "meds"
    assert panel._table.item(0, 0).text() == "1"  # model_id da 1ª linha
    assert "meta=ok" in panel._metadata.toPlainText()


@pytest.mark.gui
def test_panel_truncates_large_table(qtbot, tmp_path):
    from apps.sim_manager.perspectives.datviewer.view import (
        DISPLAY_ROW_CAP,
        DatViewerPanel,
    )

    p = tmp_path / "big.dat"
    _write_binary_dat(p, n_rows=DISPLAY_ROW_CAP + 5)
    vm = DatViewerViewModel(service=DatViewerService())
    panel = DatViewerPanel(vm)
    qtbot.addWidget(panel)
    vm.open(str(p))
    assert panel._table.rowCount() == DISPLAY_ROW_CAP  # capado
    assert "exibindo as primeiras" in panel._summary.text()  # truncamento explícito


@pytest.mark.gui
def test_panel_truncation_boundary_exact_cap(qtbot, tmp_path):
    """Fronteira: n_rows == CAP → sem aviso; n_rows == CAP+1 → com aviso."""
    from apps.sim_manager.perspectives.datviewer.view import (
        DISPLAY_ROW_CAP,
        DatViewerPanel,
    )

    vm = DatViewerViewModel(service=DatViewerService())
    panel = DatViewerPanel(vm)
    qtbot.addWidget(panel)

    at_cap = tmp_path / "at_cap.dat"
    _write_binary_dat(at_cap, n_rows=DISPLAY_ROW_CAP)
    vm.open(str(at_cap))
    assert panel._table.rowCount() == DISPLAY_ROW_CAP
    assert "exibindo as primeiras" not in panel._summary.text()  # sem truncamento

    over = tmp_path / "over.dat"
    _write_binary_dat(over, n_rows=DISPLAY_ROW_CAP + 1)
    vm.open(str(over))
    assert panel._table.rowCount() == DISPLAY_ROW_CAP
    assert "exibindo as primeiras" in panel._summary.text()  # com truncamento


@pytest.mark.gui
def test_panel_renders_nan_col0_without_crash(qtbot, tmp_path):
    """Regressão (revisão 6h): NaN/Inf na col0 (ASCII) → renderiza 'nan', sem crash.

    int(NaN) levantava ValueError engolido pelo VMSignal → tabela meio-renderizada
    sem aviso. _fmt_cell agora cai no formatador float p/ não-finitos/não-inteiros.
    """
    from apps.sim_manager.perspectives.datviewer.view import DatViewerPanel

    p = tmp_path / "nan.dat"
    p.write_text("1 2 3\nnan 5 6\n7 8 9\n", encoding="utf-8")
    vm = DatViewerViewModel(service=DatViewerService())
    panel = DatViewerPanel(vm)
    qtbot.addWidget(panel)
    vm.open(str(p))  # não pode crashar
    assert (
        panel._table.rowCount() == 3
    )  # TODAS as linhas presentes (não meio-renderiza)
    assert panel._table.item(1, 0).text() == "nan"  # col0 NaN → "nan"
    assert panel._table.item(2, 0).text() == "7"  # linha após o NaN é renderizada


@pytest.mark.gui
def test_panel_shows_error_without_crash(qtbot, tmp_path):
    from apps.sim_manager.perspectives.datviewer.view import DatViewerPanel

    vm = DatViewerViewModel(service=DatViewerService())
    panel = DatViewerPanel(vm)
    qtbot.addWidget(panel)
    vm.open(str(tmp_path / "missing.dat"))
    assert "⚠" in panel._summary.text()
    assert panel._table.rowCount() == 0


# ════════════════════════════════════════════════════════════════════════════
# DatViewerPerspective — contrato plugável (id/order + build)
# ════════════════════════════════════════════════════════════════════════════
def test_perspective_metadata():
    from apps.sim_manager.perspectives.datviewer.perspective import (
        DatViewerPerspective,
    )

    p = DatViewerPerspective()
    assert p.id == "datviewer"
    assert p.order == 5
    assert p.icon_glyph == "📄"


def test_perspective_build_viewmodel():
    from apps.sim_manager.perspectives.datviewer.perspective import (
        DatViewerPerspective,
    )
    from geosteering_ai.gui.shell.context import AppContext

    vm = DatViewerPerspective().build_viewmodel(AppContext(app_name="t"))
    assert isinstance(vm, DatViewerViewModel)
    assert vm.result is None  # nada carregado ainda


@pytest.mark.gui
def test_perspective_build_view_returns_widget(qtbot):
    from apps.sim_manager.perspectives.datviewer.perspective import (
        DatViewerPerspective,
    )
    from geosteering_ai.gui.qt_compat import QtWidgets
    from geosteering_ai.gui.shell.context import AppContext

    view = DatViewerPerspective().build_view(AppContext(app_name="t"))
    qtbot.addWidget(view)
    assert isinstance(view, QtWidgets.QWidget)
