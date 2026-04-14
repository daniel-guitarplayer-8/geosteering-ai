# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_multi.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes Sprint 11 — multi-TR/multi-ângulo Numba             ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-14 (PR #15)                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  ESCOPO                                                                   ║
# ║    10 testes que validam simulate_multi() + MultiSimulationResult +      ║
# ║    export_multi_tr_dat + F6/F7 wiring + shim simulate():                 ║
# ║                                                                           ║
# ║    1. single_tr_single_angle_parity        (shim ≡ simulate_multi unwrap)║
# ║    2. multi_tr_matches_single_calls        (bit-exato vs N chamadas)    ║
# ║    3. multi_angle_matches_single_calls     (bit-exato vs N chamadas)    ║
# ║    4. mixed_multi_tr_multi_angle           (nTR>1 AND nAngles>1 bit-ex.)║
# ║    5. high_rho_multi                       (ρ>1000 oklahoma_28 finito)  ║
# ║    6. f6_wiring                             (H_comp ≡ apply_compensation)║
# ║    7. f7_wiring                             (H_tilted ≡ apply_tilted)   ║
# ║    8. cache_dedup_vertical                  (1 cache para dip=0°)       ║
# ║    8b. cache_dedup_collision_distinct_results (mesmo hordist, H difere) ║
# ║    9. fortran_numerical_parity_dat         (@fortran_required)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes Sprint 11 — multi-TR/multi-ângulo nativos no simulador Numba.

Validam que :func:`simulate_multi` reproduz fielmente o comportamento do
Fortran v10.0 e é bit-exato vs. chamadas independentes a :func:`simulate`.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from geosteering_ai.simulation import (
    MultiSimulationResult,
    SimulationConfig,
    SimulationResult,
    simulate,
    simulate_multi,
)
from geosteering_ai.simulation.io.binary_dat_multi import (
    export_multi_tr_dat,
    read_multi_tr_dat,
)
from geosteering_ai.simulation.postprocess.compensation import apply_compensation
from geosteering_ai.simulation.postprocess.tilted import apply_tilted_antennas
from geosteering_ai.simulation.validation.canonical_models import get_canonical_model

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_FORTRAN_EXEC = Path(__file__).parent.parent / "Fortran_Gerador" / "tatu.x"
FORTRAN_AVAILABLE = DEFAULT_FORTRAN_EXEC.exists()

fortran_required = pytest.mark.skipif(
    not FORTRAN_AVAILABLE,
    reason=f"Fortran executable not found: {DEFAULT_FORTRAN_EXEC}",
)


@pytest.fixture
def small_model():
    """Modelo oklahoma_3 — 3 camadas TIV simples."""
    m = get_canonical_model("oklahoma_3")
    return {
        "rho_h": np.asarray(m.rho_h, dtype=np.float64),
        "rho_v": np.asarray(m.rho_v, dtype=np.float64),
        "esp": np.asarray(m.esp, dtype=np.float64),
        "positions_z": np.linspace(m.min_depth - 2, m.max_depth + 2, 30),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Teste 1 — single-TR single-angle parity (invariante estrutural via shim)
# ──────────────────────────────────────────────────────────────────────────────
def test_single_tr_single_angle_parity(small_model):
    """`simulate()` é literalmente shim de `simulate_multi()` → bit-exato."""
    # simulate() via shim
    res_single = simulate(
        rho_h=small_model["rho_h"],
        rho_v=small_model["rho_v"],
        esp=small_model["esp"],
        positions_z=small_model["positions_z"],
        frequency_hz=20000.0,
        tr_spacing_m=1.0,
        dip_deg=0.0,
    )

    # simulate_multi direto com nTR=1, nAngles=1
    res_multi = simulate_multi(
        rho_h=small_model["rho_h"],
        rho_v=small_model["rho_v"],
        esp=small_model["esp"],
        positions_z=small_model["positions_z"],
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0],
    )

    assert isinstance(res_single, SimulationResult)
    assert isinstance(res_multi, MultiSimulationResult)
    assert res_multi.H_tensor.shape == (1, 1, 30, 1, 9)
    # Bit-exato: são literalmente os mesmos dados
    np.testing.assert_array_equal(res_single.H_tensor, res_multi.H_tensor[0, 0])


# ──────────────────────────────────────────────────────────────────────────────
# Teste 2 — multi-TR bit-exato vs N chamadas independentes
# ──────────────────────────────────────────────────────────────────────────────
def test_multi_tr_matches_single_calls(small_model):
    """simulate_multi(tr=[0.5,1,2]).H[i] ≡ simulate(tr=L_i) para cada i."""
    tr_list = [0.5, 1.0, 2.0]

    res_multi = simulate_multi(
        **small_model,
        frequencies_hz=[20000.0],
        tr_spacings_m=tr_list,
        dip_degs=[0.0],
    )

    for i, L in enumerate(tr_list):
        res_single = simulate(**small_model, tr_spacing_m=L, dip_deg=0.0)
        np.testing.assert_array_equal(
            res_multi.H_tensor[i, 0],
            res_single.H_tensor,
            err_msg=f"H_tensor mismatch at TR index {i} (L={L})",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Teste 3 — multi-ângulo bit-exato vs N chamadas independentes
# ──────────────────────────────────────────────────────────────────────────────
def test_multi_angle_matches_single_calls(small_model):
    """simulate_multi(dip=[0,30,60]).H[j] ≡ simulate(dip=θ_j) para cada j."""
    dip_list = [0.0, 30.0, 60.0]

    res_multi = simulate_multi(
        **small_model,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=dip_list,
    )

    for j, theta in enumerate(dip_list):
        res_single = simulate(**small_model, tr_spacing_m=1.0, dip_deg=theta)
        np.testing.assert_array_equal(
            res_multi.H_tensor[0, j],
            res_single.H_tensor,
            err_msg=f"H_tensor mismatch at angle index {j} (θ={theta}°)",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Teste 4 — mixed multi-TR × multi-ângulo (Plan agent Test 3b)
# ──────────────────────────────────────────────────────────────────────────────
def test_mixed_multi_tr_multi_angle(small_model):
    """H[i,j] ≡ simulate(L_i, θ_j) para todos pares (i, j)."""
    tr_list = [0.5, 1.5]
    dip_list = [0.0, 45.0]

    res_multi = simulate_multi(
        **small_model,
        frequencies_hz=[20000.0],
        tr_spacings_m=tr_list,
        dip_degs=dip_list,
    )
    assert res_multi.H_tensor.shape == (2, 2, 30, 1, 9)

    for i, L in enumerate(tr_list):
        for j, theta in enumerate(dip_list):
            res_single = simulate(**small_model, tr_spacing_m=L, dip_deg=theta)
            np.testing.assert_array_equal(
                res_multi.H_tensor[i, j],
                res_single.H_tensor,
                err_msg=f"Mismatch at (TR={L}, θ={theta}°)",
            )


# ──────────────────────────────────────────────────────────────────────────────
# Teste 5 — alta resistividade multi-TR × multi-angle (oklahoma_28, ρ>1000 Ω·m)
# ──────────────────────────────────────────────────────────────────────────────
def test_high_rho_multi():
    """oklahoma_28 com ρ > 1000 Ω·m × nTR=3 × nAngles=3 → todos finitos."""
    m = get_canonical_model("oklahoma_28")
    res = simulate_multi(
        rho_h=np.asarray(m.rho_h),
        rho_v=np.asarray(m.rho_v),
        esp=np.asarray(m.esp),
        positions_z=np.linspace(m.min_depth - 1, m.max_depth + 1, 40),
        frequencies_hz=[20000.0],
        tr_spacings_m=[0.5, 1.0, 2.0],
        dip_degs=[0.0, 30.0, 60.0],
    )
    assert res.H_tensor.shape == (3, 3, 40, 1, 9)
    assert np.all(
        np.isfinite(res.H_tensor)
    ), "H_tensor contém NaN/Inf em oklahoma_28 alta-ρ"
    # Magnitude physical-sanity: tensor em LWD tipicamente 1e-6 a 1e-3 A/m²
    # (não validação estrita, apenas guard contra explosão numérica)
    mag_max = np.abs(res.H_tensor).max()
    assert mag_max < 1e3, f"Magnitude H_tensor = {mag_max} excessiva (overflow?)"


# ──────────────────────────────────────────────────────────────────────────────
# Teste 6 — F6 wiring (compensação CDR) é idêntica a apply_compensation direto
# ──────────────────────────────────────────────────────────────────────────────
def test_f6_wiring(small_model):
    """use_compensation=True produz H_comp idêntico à chamada direta de apply_compensation."""
    res = simulate_multi(
        **small_model,
        frequencies_hz=[20000.0],
        tr_spacings_m=[0.5, 1.0, 1.5],
        dip_degs=[0.0],
        use_compensation=True,
        comp_pairs=((0, 2), (1, 2)),
    )

    assert res.H_comp is not None
    assert res.phase_diff_deg is not None
    assert res.atten_db is not None
    # Shape esperada: (n_pairs, nAngles, n_pos, nf, 9)
    assert res.H_comp.shape == (2, 1, 30, 1, 9)

    # Reproduz via apply_compensation direto
    H_comp_ref, phase_ref, atten_ref = apply_compensation(
        H_tensors_per_tr=res.H_tensor,
        comp_pairs=((0, 2), (1, 2)),
    )
    np.testing.assert_array_equal(res.H_comp, H_comp_ref)
    np.testing.assert_array_equal(res.phase_diff_deg, phase_ref)
    # atten_db pode conter NaN (por design — proteção div-por-zero); usa equal_nan
    np.testing.assert_array_equal(res.atten_db, atten_ref)


# ──────────────────────────────────────────────────────────────────────────────
# Teste 7 — F7 wiring (tilted antennas) idêntico a apply_tilted_antennas direto
# ──────────────────────────────────────────────────────────────────────────────
def test_f7_wiring(small_model):
    """use_tilted=True produz H_tilted idêntico à chamada direta de apply_tilted_antennas."""
    tilted = ((0.0, 0.0), (45.0, 0.0), (45.0, 90.0))
    res = simulate_multi(
        **small_model,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0],
        dip_degs=[0.0, 30.0],
        use_tilted=True,
        tilted_configs=tilted,
    )

    assert res.H_tilted is not None
    # Shape: (n_tilted, nTR, nAngles, n_pos, nf)
    assert res.H_tilted.shape == (3, 1, 2, 30, 1)

    # Reproduz via apply_tilted_antennas direto
    H_tilted_ref = apply_tilted_antennas(
        H_tensor=res.H_tensor,
        tilted_configs=tilted,
    )
    np.testing.assert_array_equal(res.H_tilted, H_tilted_ref)


# ──────────────────────────────────────────────────────────────────────────────
# Teste 8 — cache dedup: poço vertical com nAngles=5 usa 1 cache
# ──────────────────────────────────────────────────────────────────────────────
def test_cache_dedup_vertical(small_model):
    """Para dip=0° em todos os ângulos (unique hordist=0): 1 cache compartilhado."""
    res = simulate_multi(
        **small_model,
        frequencies_hz=[20000.0],
        tr_spacings_m=[0.5, 1.0, 1.5, 2.0],
        dip_degs=[0.0] * 5,
    )
    # hordist = L·|sin(0)| = 0 para TODAS as 4×5=20 combinações → 1 único cache
    assert res.unique_hordist_count == 1, (
        f"Esperado 1 cache único (todos dip=0°), " f"got {res.unique_hordist_count}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Teste 8b — cache dedup: colisão L·|sin(θ)| idêntica mas H distinto
# ──────────────────────────────────────────────────────────────────────────────
def test_cache_dedup_collision_distinct_results(small_model):
    """L=2,θ=30° e L=1,θ=90° compartilham cache (hordist=1) mas H_tensor difere.

    Isso valida que a dedup por hordist é correta: o cache de common_arrays
    é mesmo (pois depende apenas de hordist); mas o output final difere
    porque dz_half = L·cos(θ)/2 diverge (1·cos(30°)/2 = 0.433 vs
    2·cos(90°)/2 = 0.0).
    """
    # Simula (L=1, θ=90°) e (L=2, θ=30°) em batch
    res = simulate_multi(
        **small_model,
        frequencies_hz=[20000.0],
        tr_spacings_m=[1.0, 2.0],
        dip_degs=[90.0, 30.0],
    )

    # hordist para ambos os pares:
    #   (L=1, θ=90°) → 1·sin(90°) = 1.0
    #   (L=2, θ=30°) → 2·sin(30°) = 1.0
    #   (L=1, θ=30°) → 1·sin(30°) = 0.5
    #   (L=2, θ=90°) → 2·sin(90°) = 2.0
    # => 3 caches únicos: {0.5, 1.0, 2.0}
    assert res.unique_hordist_count == 3, (
        f"Esperado 3 caches (para hordists {{0.5, 1.0, 2.0}}), "
        f"got {res.unique_hordist_count}"
    )

    # Os dois pares (L=1, θ=90°) e (L=2, θ=30°) compartilham cache mas
    # devem produzir H_tensor DIFERENTES (dz_half diverge).
    #   (L=1, θ=90°): dz_half = 0.5·1·cos(90°) = 0.0
    #   (L=2, θ=30°): dz_half = 0.5·2·cos(30°) = 0.866
    H_A = res.H_tensor[0, 0]  # (L=1, θ=90°)
    H_B = res.H_tensor[1, 1]  # (L=2, θ=30°)
    # Garantia: resultados NÃO são idênticos (dz_half diferente)
    assert not np.array_equal(H_A, H_B), (
        "H_tensor idênticos — colisão de cache incorreta "
        "(deveria divergir por dz_half distinto)"
    )
    # Mas ambos finitos
    assert np.all(np.isfinite(H_A))
    assert np.all(np.isfinite(H_B))


# ──────────────────────────────────────────────────────────────────────────────
# Teste 9 — Paridade numérica com Fortran (byte-file exato não é viável
# entre compiladores; paridade numérica < 1e-12 é o gate)
# ──────────────────────────────────────────────────────────────────────────────
@fortran_required
def test_fortran_numerical_parity_dat(tmp_path):
    """Python `.dat` tem paridade numérica < 1e-12 vs Fortran `.dat`.

    Executa `tatu.x` com modelo trivial (iso 3 camadas, dip=0°, nTR=2),
    executa `simulate_multi` + `export_multi_tr_dat` com mesmos params,
    carrega ambos via `read_multi_tr_dat` e compara campo a campo.

    Note:
        Diferenças no último ULP (~1e-15) entre gfortran e Numba LLVM
        são esperadas — teste valida paridade NUMÉRICA, não BYTE-EXACT.
        Tolerância `1e-12` é 6 ordens de magnitude melhor que o gate
        `1e-6` padrão do pacote (compare_fortran_python).
    """
    # ── Setup: executa tatu.x ────────────────────────────────────
    fort_workdir = tmp_path / "fort"
    fort_workdir.mkdir()
    (fort_workdir / "model.in").write_text(
        "1\n20000.0\n"  # nf, freq
        "1\n0.0\n"  # ntheta, theta
        "29.5418\n4.0\n0.2\n"  # h1, tj, p_med
        "2\n0.5\n1.0\n"  # nTR, dTR
        "trivial\n"  # filename
        "3\n10.0 10.0\n1.0 1.0\n10.0 10.0\n"  # n, rho_h/v per layer
        "1.5\n"  # esp (n-2=1 camada interna)
        "1 1\n",  # modelm, nmaxmodel
    )
    # Copia executável para workdir (tatu.x lê ./model.in)
    shutil.copy(DEFAULT_FORTRAN_EXEC, fort_workdir / "tatu.x")
    result = subprocess.run(
        ["./tatu.x"],
        cwd=fort_workdir,
        capture_output=True,
        timeout=60,
        text=True,
    )
    assert result.returncode == 0, f"tatu.x falhou: {result.stderr}"

    # ── Python: simulate_multi com mesmos parâmetros ──────────────
    rho_h = np.array([10.0, 1.0, 10.0])
    rho_v = np.array([10.0, 1.0, 10.0])
    esp = np.array([1.5])
    h1, pz, nmeds = 29.5418, 0.2, 20
    positions_z = -h1 + np.arange(nmeds) * pz

    res = simulate_multi(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=positions_z,
        frequencies_hz=[20000.0],
        tr_spacings_m=[0.5, 1.0],
        dip_degs=[0.0],
    )

    py_workdir = tmp_path / "py"
    py_workdir.mkdir()
    export_multi_tr_dat(res, "trivial", py_workdir)

    # ── Compara por registro (struct 22-col) ──────────────────────
    n_records = nmeds  # nAngles × nf × n_pos = 1×1×20 = 20
    max_err_global = 0.0
    for iTR in (1, 2):
        fort_dat = fort_workdir / f"trivial_TR{iTR}.dat"
        py_dat = py_workdir / f"trivial_TR{iTR}.dat"
        assert fort_dat.exists(), f"tatu.x não gerou {fort_dat.name}"
        assert py_dat.exists(), f"Python não gerou {py_dat.name}"

        fort_data = read_multi_tr_dat(fort_dat, n_records)
        py_data = read_multi_tr_dat(py_dat, n_records)

        # Índices devem ser bit-exatos
        np.testing.assert_array_equal(fort_data["i"], py_data["i"])

        # Campos float: tolerância 1e-12 (6 ordens melhor que 1e-6 padrão)
        for field in fort_data.dtype.names:
            if field == "i":
                continue
            fort_vals = fort_data[field]
            py_vals = py_data[field]
            # Ignora NaN (caso degenerado de atten_db; aqui campos principais)
            diff = np.abs(fort_vals - py_vals)
            max_diff = float(np.nanmax(diff))
            max_err_global = max(max_err_global, max_diff)
            assert (
                max_diff < 1e-12
            ), f"TR{iTR}, field {field}: max_abs_diff = {max_diff} > 1e-12"
    # Sanity: ficou realmente baixo?
    assert max_err_global < 1e-12


# ──────────────────────────────────────────────────────────────────────────────
# Teste 10 (bônus) — validação de entradas (fail-fast ValueError)
# ──────────────────────────────────────────────────────────────────────────────
class TestInputValidation:
    """Validações fail-fast em simulate_multi."""

    def test_empty_tr_spacings_raises(self, small_model):
        with pytest.raises(ValueError, match="tr_spacings_m vazio"):
            simulate_multi(**small_model, tr_spacings_m=[])

    def test_empty_dip_degs_raises(self, small_model):
        with pytest.raises(ValueError, match="dip_degs vazio"):
            simulate_multi(**small_model, dip_degs=[])

    def test_tr_out_of_range_raises(self, small_model):
        with pytest.raises(ValueError, match="fora do range"):
            simulate_multi(**small_model, tr_spacings_m=[100.0])

    def test_dip_out_of_range_raises(self, small_model):
        with pytest.raises(ValueError, match="fora do range"):
            simulate_multi(**small_model, dip_degs=[95.0])

    def test_f6_requires_two_trs(self, small_model):
        with pytest.raises(ValueError, match="use_compensation=True requer"):
            simulate_multi(
                **small_model,
                tr_spacings_m=[1.0],  # apenas 1 TR
                use_compensation=True,
                comp_pairs=((0, 0),),
            )

    def test_f6_invalid_pair_raises(self, small_model):
        with pytest.raises(ValueError, match="fora do range|degenerado"):
            simulate_multi(
                **small_model,
                tr_spacings_m=[0.5, 1.0],
                use_compensation=True,
                comp_pairs=((0, 99),),
            )

    def test_f7_empty_configs_raises(self, small_model):
        with pytest.raises(ValueError, match="tilted_configs não-vazio"):
            simulate_multi(**small_model, use_tilted=True, tilted_configs=())


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
