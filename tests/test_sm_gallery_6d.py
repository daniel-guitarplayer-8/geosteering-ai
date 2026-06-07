# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_gallery_6d.py                                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0017-sm-plots-complete (Fatia 6d)                          ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM MVVM — galeria 6d (canais/geosinais/perfis/heatmap/anim) ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-07                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre a galeria estendida da Fatia 6d: canais (9 componentes + 5        ║
# ║    geosinais), modos (curve/rho/lambda/heatmap), perfis ρ/λ da geologia,    ║
# ║    heatmap de ensemble, foco/animation bar e o render da View (4 modos).    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da Fatia 6d — galeria estendida (canais/geosinais/perfis/heatmap/animação)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from apps.sim_manager.perspectives.simulation.results_viewmodel import (  # noqa: E402
    CHANNELS,
    COMPONENT_NAMES,
    PLOT_MODES,
    ResultsViewModel,
    _kind_transform,
)
from geosteering_ai.gui.services.derived import (  # noqa: E402
    GEOSIGNALS,
    compute_geosignal,
)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def _h6(n_models=4, n_pos=10):
    """H6 sintético (n_models, 1, 1, n_pos, 1, 9) determinístico."""
    rng = np.random.default_rng(7)
    shape = (n_models, 1, 1, n_pos, 1, 9)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _result_with_geology(n_models=4, n_pos=10):
    """Resultado com H6 + positions_z + geologia (3 camadas) por modelo."""
    h6 = _h6(n_models, n_pos)
    geology = [
        {
            "rho_h": np.array([1.0, 10.0, 100.0]) * (m + 1),
            "rho_v": np.array([2.0, 20.0, 200.0]) * (m + 1),
            "thicknesses": np.array([4.0]),  # 3 camadas → 1 interna
        }
        for m in range(n_models)
    ]
    return {
        "H6": h6,
        "positions_z": np.linspace(0.0, 9.0, n_pos),
        "geology": geology,
        "n_models": n_models,
    }


def _vm(result=None):
    rvm = ResultsViewModel(page_size=4)
    if result is not None:
        rvm.set_result(result)
    return rvm


# ════════════════════════════════════════════════════════════════════════════
# Canais: 9 componentes + 5 geosinais
# ════════════════════════════════════════════════════════════════════════════
def test_channels_layout():
    assert CHANNELS == COMPONENT_NAMES + GEOSIGNALS
    assert len(CHANNELS) == 14
    assert CHANNELS[9:] == GEOSIGNALS


def test_channel_index_clamp_and_name():
    rvm = _vm(_result_with_geology())
    rvm.channel_index = 99
    assert rvm.channel_index == 13 and rvm.channel_name == "U3DF"
    assert rvm.is_geosignal
    rvm.channel_index = 8
    assert rvm.channel_name == "Hzz" and not rvm.is_geosignal
    # compat: component_index continua clampando a 0..8.
    rvm.component_index = 99
    assert rvm.component_index == 8


def test_curve_for_geosignal_matches_derived():
    """O canal geosinal deve bater com compute_geosignal sobre as 9 componentes."""
    res = _result_with_geology()
    rvm = _vm(res)
    h6 = res["H6"]
    for gi, name in enumerate(GEOSIGNALS):
        rvm.channel_index = len(COMPONENT_NAMES) + gi
        rvm.plot_kind = "re"
        for m in range(h6.shape[0]):
            h9 = h6[m, 0, 0, :, 0, :]
            expected = _kind_transform("re", compute_geosignal(name, h9))
            assert np.allclose(rvm.curve_for(m), expected), name


def test_curve_for_component_unchanged():
    """Canal componente (0..8) preserva o caminho cru H6[...,ch]."""
    res = _result_with_geology()
    rvm = _vm(res)
    rvm.channel_index = 4  # Hyy
    rvm.plot_kind = "mag"
    h6 = res["H6"]
    assert np.allclose(rvm.curve_for(2), np.abs(h6[2, 0, 0, :, 0, 4]))


# ════════════════════════════════════════════════════════════════════════════
# Modos da galeria
# ════════════════════════════════════════════════════════════════════════════
def test_plot_mode_setter():
    rvm = _vm()
    assert rvm.plot_mode == "curve"
    for m in PLOT_MODES:
        rvm.plot_mode = m
        assert rvm.plot_mode == m
    rvm.plot_mode = "bogus"
    assert rvm.plot_mode == "curve"


# ════════════════════════════════════════════════════════════════════════════
# Perfis ρ/λ (geologia)
# ════════════════════════════════════════════════════════════════════════════
def test_rho_curves_from_geology():
    res = _result_with_geology(n_models=4, n_pos=10)
    rvm = _vm(res)
    out = rvm.rho_curves_for(1)
    assert out is not None
    z, rho_h, rho_v = out
    assert z.shape == (10,) and rho_h.shape == (10,) and rho_v.shape == (10,)
    # modelo 1: rho_h=[2,20,200]; interfaces=[0,4]. Convenção monólito (sm_plots
    # :500-533): z=0 cai na PRIMEIRA interface → camada 1 (camada 0 = topo, z<0).
    assert rho_h[0] == 20.0  # z=0 → camada 1 (z=0 é a interface 0→1)
    assert rho_h[-1] == 200.0  # z=9 → camada 2
    assert np.all(rho_v >= rho_h)  # ρv = 2·ρh no fixture


def test_lambda_curve_from_geology():
    res = _result_with_geology()
    rvm = _vm(res)
    out = rvm.lambda_curve_for(0)
    assert out is not None
    z, lam = out
    assert lam.shape == z.shape
    assert np.all(lam >= 1.0)  # TIV: λ ≥ 1
    # ρv=2·ρh → λ=√2 em todas as camadas
    assert np.allclose(lam, np.sqrt(2.0))


def test_profiles_none_without_geology():
    """Sem geologia no resultado → perfis retornam None (a View mostra 'sem geologia')."""
    h6 = _h6()
    rvm = _vm({"H6": h6, "positions_z": np.linspace(0, 9, h6.shape[3])})
    assert rvm.rho_curves_for(0) is None
    assert rvm.lambda_curve_for(0) is None
    assert rvm.geology_for(0) is None


def test_geology_for_out_of_range():
    rvm = _vm(_result_with_geology(n_models=3))
    assert rvm.geology_for(0) is not None
    assert rvm.geology_for(99) is None
    assert rvm.geology_for(-1) is None


# ════════════════════════════════════════════════════════════════════════════
# Heatmap de ensemble (imagem multidimensional)
# ════════════════════════════════════════════════════════════════════════════
def test_ensemble_image_shape_and_extent():
    res = _result_with_geology(n_models=4, n_pos=10)
    rvm = _vm(res)
    rvm.channel_index = 8  # Hzz
    rvm.plot_kind = "mag"
    out = rvm.ensemble_image()
    assert out is not None
    img, extent, label = out
    assert img.shape == (4, 10)  # n_models × n_pos
    # linha m == curve_for(m)
    assert np.allclose(img[2], rvm.curve_for(2))
    assert extent[0] == 0.0 and extent[1] == 9.0  # z range
    assert "Hzz" in label


def test_ensemble_image_none_without_result():
    assert _vm().ensemble_image() is None


# ════════════════════════════════════════════════════════════════════════════
# Foco / animação (estado PURO)
# ════════════════════════════════════════════════════════════════════════════
def test_focus_model_clamp():
    rvm = _vm(_result_with_geology(n_models=4))
    rvm.focus_model = 99
    assert rvm.focus_model == 3
    rvm.focus_model = -5
    assert rvm.focus_model == 0


# ════════════════════════════════════════════════════════════════════════════
# AnimationBar (widget Qt)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
def test_animation_bar_set_frame_no_emit_slider_emits(qtbot):
    from geosteering_ai.gui.shell.widgets.animation_bar import AnimationBar

    bar = AnimationBar()
    qtbot.addWidget(bar)
    rec: list = []
    bar.frame_changed.connect(rec.append)
    bar.set_frame_count(5)
    bar.set_frame(2)  # sync externo → NÃO emite (evita realimentação View↔VM)
    assert rec == [] and bar.current_frame() == 2
    bar._slider.setValue(3)  # ação do usuário → emite
    assert rec == [3] and bar.current_frame() == 3


@pytest.mark.gui
def test_animation_bar_play_pause(qtbot):
    from geosteering_ai.gui.shell.widgets.animation_bar import AnimationBar

    bar = AnimationBar()
    qtbot.addWidget(bar)
    bar.set_frame_count(5)
    bar._play.setChecked(True)
    assert bar.is_playing()
    bar._play.setChecked(False)
    assert not bar.is_playing()
    # ≤1 frame não inicia playback (nada a varrer).
    bar.set_frame_count(1)
    bar._play.setChecked(True)
    assert not bar.is_playing() and not bar._play.isChecked()


@pytest.mark.gui
def test_animation_bar_advance_wraps(qtbot):
    from geosteering_ai.gui.shell.widgets.animation_bar import AnimationBar

    bar = AnimationBar()
    qtbot.addWidget(bar)
    rec: list = []
    bar.frame_changed.connect(rec.append)
    bar.set_frame_count(3)
    bar.set_frame(2)
    bar._advance()  # 2 → 0 (wrap), emite via slider
    assert bar.current_frame() == 0 and rec == [0]


# ════════════════════════════════════════════════════════════════════════════
# ResultsView — render dos 4 modos (gui)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.gui
@pytest.mark.parametrize("mode", list(PLOT_MODES))
def test_results_view_renders_each_mode(qtbot, mode):
    from apps.sim_manager.perspectives.simulation.results_view import ResultsView

    rvm = _vm()
    view = ResultsView(rvm)
    qtbot.addWidget(view)
    rvm.set_result(_result_with_geology(n_models=5, n_pos=10))
    rvm.plot_mode = mode  # re-render sem crash
    # perfis desabilitam canal/kind; heatmap desabilita paginação/anim.
    if mode in ("rho", "lambda"):
        assert not view._cmp.isEnabled() and not view._kind.isEnabled()
    if mode == "heatmap":
        assert not view._anim.isEnabled()


@pytest.mark.gui
def test_results_view_geosignal_and_animation(qtbot):
    from apps.sim_manager.perspectives.simulation.results_view import ResultsView

    rvm = _vm()
    view = ResultsView(rvm)
    qtbot.addWidget(view)
    rvm.set_result(_result_with_geology(n_models=8, n_pos=10))
    # canal geosinal renderiza
    view._cmp.setCurrentIndex(len(COMPONENT_NAMES))  # USD
    assert rvm.channel_name == "USD"
    # animação: mover o slider foca o modelo E leva a página até ele
    view._anim._slider.setValue(6)
    assert rvm.focus_model == 6
    assert rvm.page == 6 // rvm.page_size
