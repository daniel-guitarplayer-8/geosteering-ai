# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_cli_dat_save.py                                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP — gravação .dat/.out + helpers _exec (Sprint v2.53)║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-02                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    1. Helpers puros de ``_exec`` (rápidos, sem JIT): models_to_batch,     ║
# ║       rho_at_obs_from_batch, finitude_stats, parity_max_abs_diff.         ║
# ║    2. CONFORMIDADE FÍSICA do .dat com geosteering-physics.md §4 (col1=    ║
# ║       zobs, col2/3=res_h/res_v REAIS na camada, col4-21=Re/Im Hxx..Hzz).  ║
# ║    3. FIX do bug .npz (grava H_stack real, não repr(result)).             ║
# ║    4. Não-regressão de roteamento: caminho Numba NÃO chama simulate_batch.║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de gravação .dat/.out (conformidade §4) + helpers _exec (v2.53)."""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.cli._exec import (
    finitude_stats,
    models_to_batch,
    parity_max_abs_diff,
    rho_at_obs_from_batch,
    run_once,
)
from geosteering_ai.cli._main import build_parser
from geosteering_ai.cli.simulate import _build_random_models, handle_simulate

# ════════════════════════════════════════════════════════════════════════
# Helpers puros de _exec (rápidos, sem compilação JIT)
# ════════════════════════════════════════════════════════════════════════


def test_models_to_batch_shapes() -> None:
    """Empilha N modelos em (N,5)/(N,5)/(N,3)."""
    rho_h, rho_v, esp = models_to_batch(_build_random_models(3, 42))
    assert rho_h.shape == (3, 5)
    assert rho_v.shape == (3, 5)
    assert esp.shape == (3, 3)


def test_models_to_batch_empty_raises() -> None:
    """Lista vazia → ValueError (nada a empilhar)."""
    with pytest.raises(ValueError, match="vazio"):
        models_to_batch([])


def test_rho_at_obs_matches_scalar_layer_lookup() -> None:
    """res_h/res_v no z_obs == resistividade da camada (twin escalar)."""
    from geosteering_ai.data.synthetic_generator import _find_layer_for_z

    rho_h = np.array([[1.0, 10.0, 100.0, 5.0, 50.0]])  # (1, 5)
    rho_v = rho_h * 2.0
    esp = np.array([[2.0, 3.0, 4.0]])  # fronteiras cumsum em 2, 5, 9
    pos = np.array([-1.0, 0.5, 3.0, 6.0, 20.0])

    rh_obs, rv_obs = rho_at_obs_from_batch(pos, rho_h, rho_v, esp)
    assert rh_obs.shape == (1, 5)
    for j, z in enumerate(pos):
        layer = _find_layer_for_z(float(z), esp[0], 5)
        assert rh_obs[0, j] == rho_h[0, layer]
        assert rv_obs[0, j] == rho_v[0, layer]


def test_finitude_stats_clean() -> None:
    """Tensor finito → 0 NaN/Inf, all_finite True."""
    assert finitude_stats(np.ones((2, 3), dtype=np.complex128)) == {
        "nan_count": 0,
        "inf_count": 0,
        "all_finite": True,
    }


def test_finitude_stats_detects_nan_inf() -> None:
    """NaN e Inf injetados são contados; all_finite False."""
    H = np.ones((2, 3), dtype=np.complex128)
    H[0, 0] = np.nan
    H[1, 2] = complex(np.inf, 0.0)
    s = finitude_stats(H)
    assert s["nan_count"] >= 1
    assert s["inf_count"] >= 1
    assert s["all_finite"] is False


def test_parity_identical_is_zero() -> None:
    """Tensores idênticos → max|Δ| == 0."""
    a = np.arange(12, dtype=np.complex128).reshape(3, 4)
    assert parity_max_abs_diff(a, a) == 0.0


def test_parity_known_diff() -> None:
    """Diferença conhecida 3+4j → |Δ| = 5."""
    a = np.zeros((2, 2), dtype=np.complex128)
    b = a.copy()
    b[0, 0] = complex(3.0, 4.0)
    assert parity_max_abs_diff(a, b) == pytest.approx(5.0)


def test_parity_shape_mismatch_raises() -> None:
    """Shapes incompatíveis → ValueError."""
    with pytest.raises(ValueError, match="incompatíveis"):
        parity_max_abs_diff(np.zeros((2, 2)), np.zeros((2, 3)))


# ════════════════════════════════════════════════════════════════════════
# Conformidade física do .dat (geosteering-physics.md §4) — sim real (JIT)
# ════════════════════════════════════════════════════════════════════════


def test_dat_save_physics_22col_compliance(tmp_path) -> None:
    """O .dat gerado segue o layout 22-col de geosteering-physics.md §4.

    Verifica:
      - shape (n_models·n_pos, 22) [1 freq × 1 dip × 1 TR];
      - col1 == zobs (grid de posições);
      - col2/col3 == res_h/res_v REAIS na camada de cada z_obs (NÃO zeros);
      - col4/5 == Re/Im(Hxx); col20/21 == Re/Im(Hzz) (ordem canônica);
      - .out parseável (nt nf nmaxmodel na 1ª linha).
    """
    from geosteering_ai.cli._exec import models_to_batch as _to_batch
    from geosteering_ai.data.loading import load_binary_dat
    from geosteering_ai.data.synthetic_generator import _layer_at_batch

    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "simulate",
            "--models",
            "3",
            "--n-pos",
            "16",
            "--seed",
            "42",
            "--out",
            str(out_dir),
            "--format",
            "dat",
            "--quiet",
        ]
    )
    assert handle_simulate(args) == 0

    dat = out_dir / "simulate_results.dat"
    out = out_dir / "simulate_results.out"
    assert dat.exists() and out.exists()

    arr = load_binary_dat(str(dat), 22)
    assert arr.shape == (3 * 16, 22)

    # Reconstrói o esperado (mesma seed) — res_h/res_v na camada de cada z.
    models = _build_random_models(3, 42)
    pos = np.linspace(-5.0, 5.0, 16)
    rho_h_b, rho_v_b, esp_b = _to_batch(models)
    layer = _layer_at_batch(pos, esp_b, 5)
    rho_h_obs = np.take_along_axis(rho_h_b, layer, axis=1)
    rho_v_obs = np.take_along_axis(rho_v_b, layer, axis=1)

    # H6 para checar a ordem canônica das componentes (col4=Hxx … col20=Hzz).
    H6, _, _, _eff, _reason = run_once(
        "numba",
        models,
        pos,
        frequencies_hz=[20000.0],
        dip_degs=[0.0],
        tr_spacings_m=[1.0],
    )

    # Registros em ordem Fortran (m, itr, kt, fi, jm) → r = m*16 + pos.
    for r in range(arr.shape[0]):
        m, j = divmod(r, 16)
        assert arr[r, 1] == pytest.approx(pos[j])  # col1 zobs §4
        assert arr[r, 2] == pytest.approx(rho_h_obs[m, j])  # col2 res_h §4
        assert arr[r, 3] == pytest.approx(rho_v_obs[m, j])  # col3 res_v §4
        assert arr[r, 4] == pytest.approx(H6[m, 0, 0, j, 0, 0].real)  # Re Hxx
        assert arr[r, 5] == pytest.approx(H6[m, 0, 0, j, 0, 0].imag)  # Im Hxx
        assert arr[r, 20] == pytest.approx(H6[m, 0, 0, j, 0, 8].real)  # Re Hzz
        assert arr[r, 21] == pytest.approx(H6[m, 0, 0, j, 0, 8].imag)  # Im Hzz

    # A correção-chave: col2/col3 NÃO são zeros (resistividade real).
    assert (arr[:, 2] != 0.0).all()
    assert (arr[:, 3] != 0.0).all()

    # .out parseável: 1ª linha = nt nf nmaxmodel.
    head = out.read_text(encoding="utf-8").splitlines()[0].split()
    assert len(head) == 3
    assert int(head[0]) == 1  # nt (1 dip)
    assert int(head[1]) == 1  # nf (1 freq)
    assert int(head[2]) == 3  # nmaxmodel (3 modelos)


def test_npz_save_uses_real_tensor(tmp_path) -> None:
    """FIX v2.53: --format npz grava o tensor H REAL (não repr(result)).

    O bug legado lia ``getattr(result, "H", None)`` (sempre None — o atributo
    é ``H_stack``) e salvava ``repr(result)`` em vez do tensor.
    """
    out_dir = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "simulate",
            "--models",
            "2",
            "--n-pos",
            "8",
            "--seed",
            "42",
            "--out",
            str(out_dir),
            "--format",
            "npz",
            "--quiet",
        ]
    )
    assert handle_simulate(args) == 0

    npz = out_dir / "simulate_results.npz"
    assert npz.exists()
    data = np.load(npz)
    assert "H" in data.files  # tensor real
    assert "repr" not in data.files  # NÃO o fallback quebrado
    assert data["H"].shape == (2, 1, 1, 8, 1, 9)
    assert np.isfinite(data["H"]).all()


def test_run_once_numba_does_not_route_through_dispatcher(monkeypatch) -> None:
    """NÃO-REGRESSÃO: o caminho Numba usa simulate_multi(models=…), NÃO o
    dispatcher (preserva o pool de workers — throughput intocado)."""
    import geosteering_ai.simulation.dispatch as disp

    def _must_not_call(*_a, **_k):
        raise AssertionError("caminho Numba não deve chamar simulate_batch")

    monkeypatch.setattr(disp, "simulate_batch", _must_not_call)

    H6, elapsed, groups, eff, reason = run_once(
        "numba",
        _build_random_models(2, 42),
        np.linspace(-5.0, 5.0, 8),
        frequencies_hz=[20000.0],
        dip_degs=[0.0],
        tr_spacings_m=[1.0],
    )
    assert H6.shape == (2, 1, 1, 8, 1, 9)
    assert groups is None  # numba não reporta grupos de geometria
    assert elapsed > 0.0
    assert eff == "numba"  # backend efetivo
    assert reason is None  # numba não tem motivo de fallback
