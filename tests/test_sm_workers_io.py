# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_sm_workers_io.py                                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — Simulation Manager v2.16 I/O vetorizado           ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-01 (Sprint 15.4, v2.16)                            ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest 8.x                                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida bit-exatness de ``write_dat_from_tensor`` (vetorizada em        ║
# ║    Sprint 15.4) vs implementação de referência (5 loops Python).          ║
# ║                                                                           ║
# ║    Sprint 15.4 substituiu 5 loops aninhados (~1.8M iterações para         ║
# ║    600 modelos × 600 pos × 1 freq) por broadcast + reshape NumPy,         ║
# ║    ganhando 5–30× em tempo de I/O. Este teste garante que o byte          ║
# ║    layout do .dat permanece idêntico ao Fortran-compatible original.      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes Sprint 15.4 — Bit-exatness write_dat_from_tensor vetorizada."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from geosteering_ai.simulation.tests.sm_io import write_dat_from_tensor


def _write_dat_reference_loop(
    path: str,
    H_stack: np.ndarray,
    z_obs: np.ndarray,
    rho_h_at_obs=None,
    rho_v_at_obs=None,
    model_id_start: int = 1,
) -> None:
    """Implementação de referência (5 loops Python) — só usada para teste.

    Reproduz a versão pré-Sprint 15.4 para comparação bit-a-bit. Não deve
    ser usada em produção (15× mais lenta que a vetorizada).
    """
    H = np.asarray(H_stack)
    n_models, nTR, nAng, n_pos, nf, n_comp = H.shape
    fields = [("col0", np.int32)] + [(f"col{i}", np.float64) for i in range(1, 22)]
    rec_dtype = np.dtype(fields)
    total_records = n_models * nTR * nAng * nf * n_pos
    buf = np.zeros(total_records, dtype=rec_dtype)

    idx = 0
    for m in range(n_models):
        model_id = model_id_start + m
        for itr in range(nTR):
            for kt in range(nAng):
                for fi in range(nf):
                    for jm in range(n_pos):
                        rec = buf[idx]
                        rec["col0"] = model_id
                        if z_obs.ndim == 2:
                            rec["col1"] = float(z_obs[kt, jm])
                        else:
                            rec["col1"] = float(z_obs[jm])
                        rec["col2"] = (
                            float(rho_h_at_obs[m, kt, jm])
                            if rho_h_at_obs is not None
                            else 0.0
                        )
                        rec["col3"] = (
                            float(rho_v_at_obs[m, kt, jm])
                            if rho_v_at_obs is not None
                            else 0.0
                        )
                        for ic in range(9):
                            val = H[m, itr, kt, jm, fi, ic]
                            rec[f"col{4 + 2 * ic}"] = float(val.real)
                            rec[f"col{5 + 2 * ic}"] = float(val.imag)
                        idx += 1

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buf.tofile(path)


def _make_synthetic_tensors(
    n_models: int = 5,
    nTR: int = 2,
    nAng: int = 3,
    n_pos: int = 7,
    nf: int = 2,
    seed: int = 42,
) -> tuple:
    """Gera tensores sintéticos com padrão único por (m, itr, kt, jm, fi, ic).

    Usa valores fracionários distintos para detectar erros de
    ordenação/transposição via bit-exatness.
    """
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n_models, nTR, nAng, n_pos, nf, 9)) + 1j * rng.normal(
        size=(n_models, nTR, nAng, n_pos, nf, 9)
    )
    H = H.astype(np.complex128)
    z_obs_2d = np.linspace(-5.0, 25.0, nAng * n_pos).reshape(nAng, n_pos)
    rho_h = rng.uniform(0.5, 100.0, size=(n_models, nAng, n_pos))
    rho_v = rng.uniform(0.5, 100.0, size=(n_models, nAng, n_pos))
    return H, z_obs_2d, rho_h, rho_v


# ═════════════════════════════════════════════════════════════════════════
# 1 — Bit-exatness: vetorizada vs loop em modelo pequeno
# ═════════════════════════════════════════════════════════════════════════


def test_write_dat_vectorized_matches_loop_small(tmp_path: Path) -> None:
    """Sprint 15.4: vetorizada produz .dat idêntico bit-a-bit ao loop."""
    H, z_obs, rho_h, rho_v = _make_synthetic_tensors(
        n_models=5, nTR=2, nAng=3, n_pos=7, nf=2
    )

    path_vec = tmp_path / "vec.dat"
    path_ref = tmp_path / "ref.dat"

    write_dat_from_tensor(
        str(path_vec),
        H,
        z_obs,
        rho_h_at_obs=rho_h,
        rho_v_at_obs=rho_v,
        model_id_start=1,
    )
    _write_dat_reference_loop(
        str(path_ref),
        H,
        z_obs,
        rho_h_at_obs=rho_h,
        rho_v_at_obs=rho_v,
        model_id_start=1,
    )

    bytes_vec = path_vec.read_bytes()
    bytes_ref = path_ref.read_bytes()
    assert len(bytes_vec) == len(
        bytes_ref
    ), f"Tamanho .dat divergente: vec={len(bytes_vec)}, ref={len(bytes_ref)}"
    assert bytes_vec == bytes_ref, (
        "Conteúdo .dat divergente entre vetorizada e loop. "
        "Pode indicar erro de transposição (axis order) ou broadcast."
    )


# ═════════════════════════════════════════════════════════════════════════
# 2 — Bit-exatness: z_obs 1D (alternativa de schema)
# ═════════════════════════════════════════════════════════════════════════


def test_write_dat_vectorized_z_obs_1d(tmp_path: Path) -> None:
    """``z_obs`` shape (n_pos,) também deve produzir bit-exatness."""
    H, _, rho_h, rho_v = _make_synthetic_tensors(n_models=3, nTR=1, nAng=2, n_pos=5, nf=2)
    z_obs_1d = np.linspace(-2.0, 8.0, 5)

    path_vec = tmp_path / "vec1d.dat"
    path_ref = tmp_path / "ref1d.dat"

    write_dat_from_tensor(
        str(path_vec),
        H,
        z_obs_1d,
        rho_h_at_obs=rho_h,
        rho_v_at_obs=rho_v,
    )
    _write_dat_reference_loop(
        str(path_ref),
        H,
        z_obs_1d,
        rho_h_at_obs=rho_h,
        rho_v_at_obs=rho_v,
    )

    assert path_vec.read_bytes() == path_ref.read_bytes()


# ═════════════════════════════════════════════════════════════════════════
# 3 — Bit-exatness: rho_h/v_at_obs None (zera col2/col3)
# ═════════════════════════════════════════════════════════════════════════


def test_write_dat_vectorized_no_rho(tmp_path: Path) -> None:
    """Quando ``rho_h/v_at_obs=None``, col2/col3 devem ser zero."""
    H, z_obs, _, _ = _make_synthetic_tensors(n_models=2, n_pos=4, nf=1)

    path_vec = tmp_path / "vec_nor.dat"
    path_ref = tmp_path / "ref_nor.dat"

    write_dat_from_tensor(str(path_vec), H, z_obs)
    _write_dat_reference_loop(str(path_ref), H, z_obs)

    assert path_vec.read_bytes() == path_ref.read_bytes()


# ═════════════════════════════════════════════════════════════════════════
# 4 — Performance: vetorizada é ≥3× mais rápida que loop em escala média
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_write_dat_vectorized_is_faster(tmp_path: Path) -> None:
    """Sprint 15.4 é ≥3× mais rápida que o loop original em escala média."""
    import time

    H, z_obs, rho_h, rho_v = _make_synthetic_tensors(
        n_models=20, nTR=2, nAng=3, n_pos=200, nf=2
    )

    path_vec = tmp_path / "perf_vec.dat"
    path_ref = tmp_path / "perf_ref.dat"

    t0 = time.perf_counter()
    write_dat_from_tensor(
        str(path_vec),
        H,
        z_obs,
        rho_h_at_obs=rho_h,
        rho_v_at_obs=rho_v,
    )
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    _write_dat_reference_loop(
        str(path_ref),
        H,
        z_obs,
        rho_h_at_obs=rho_h,
        rho_v_at_obs=rho_v,
    )
    t_ref = time.perf_counter() - t0

    speedup = t_ref / t_vec
    assert speedup >= 3.0, (
        f"Vetorizada deveria ser ≥3× mais rápida; speedup={speedup:.2f}x "
        f"(t_vec={t_vec:.3f}s, t_ref={t_ref:.3f}s)"
    )
    # Sanity: bit-exatness mesmo em escala maior
    assert path_vec.read_bytes() == path_ref.read_bytes()
