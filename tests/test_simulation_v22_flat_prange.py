"""Sprint v2.22 — Testes de paridade FLAT prange (4D) vs caminho legado.

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — Sprint v2.22 (FLAT prange)
═══════════════════════════════════════════════════════════════════════════

Valida que ``_simulate_combined_prange_flat`` em ``forward.py`` produz
resultados **bit-exatos** ao caminho legado ``_simulate_combined_prange``
(Sprint 13.3, v2.21) para todos os cenários de simulação suportados.

A nova função colapsa 4 dimensões de paralelismo em um único ``prange``:
    nTR × nAng × n_pos × nf  →  prange(n_total)

A versão legacy mantém ``range(nf)`` serial dentro de
``_fields_in_freqs_kernel_cached``. Bit-exatness é garantida porque a
física é idêntica e a ordem de operações por (i_tr, i_ang, j, i_f) é
preservada — apenas o LOOP outer foi colapsado.

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation.config import SimulationConfig
from geosteering_ai.simulation.multi_forward import simulate_multi

# ──────────────────────────────────────────────────────────────────────────
# Modelos sintéticos para teste rápido (CI-friendly)
# ──────────────────────────────────────────────────────────────────────────


def _model_3layer():
    """3 camadas isotrópicas (smoke test)."""
    return dict(
        rho_h=np.array([10.0, 100.0, 10.0]),
        rho_v=np.array([10.0, 100.0, 10.0]),
        esp=np.array([2.0]),
    )


def _model_5layer_tiv():
    """5 camadas TIV (anisotropia transversa)."""
    return dict(
        rho_h=np.array([5.0, 50.0, 100.0, 50.0, 5.0]),
        rho_v=np.array([5.0, 50.0, 200.0, 50.0, 5.0]),
        esp=np.array([1.5, 2.0, 3.0]),
    )


def _model_high_rho():
    """Modelo alta resistividade (carbonato seco, ρ ~5000 Ω·m)."""
    return dict(
        rho_h=np.array([10.0, 5000.0, 10.0]),
        rho_v=np.array([10.0, 5000.0, 10.0]),
        esp=np.array([3.0]),
    )


# ──────────────────────────────────────────────────────────────────────────
# Cenários do roadmap §4 (analise_cenarios_otimizacao_simulador_numba.md)
# ──────────────────────────────────────────────────────────────────────────


CENARIOS = [
    pytest.param(
        dict(tr_spacings_m=[1.0], dip_degs=[0.0], frequencies_hz=[20000.0]),
        id="A_nf1_1TR_1ang",
    ),
    pytest.param(
        dict(
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            frequencies_hz=[2000.0, 20000.0, 100000.0, 400000.0],
        ),
        id="F_nf4_1TR_1ang",
    ),
    pytest.param(
        dict(
            tr_spacings_m=[0.5, 1.0, 1.5],
            dip_degs=[0.0, 30.0, 60.0, 89.0],
            frequencies_hz=[20000.0],
        ),
        id="B_nf1_3TR_4ang",
    ),
    pytest.param(
        dict(
            tr_spacings_m=[0.5, 1.0, 1.5],
            dip_degs=[0.0, 30.0, 60.0, 89.0],
            frequencies_hz=[2000.0, 20000.0, 100000.0, 400000.0],
        ),
        id="J_nf4_3TR_4ang_full",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Paridade bit-exata FLAT vs legacy
# ═══════════════════════════════════════════════════════════════════════════


class TestFlatVsLegacyBitExact:
    """Sprint v2.22.1 — FLAT prange produz output BIT-EXATO ao legacy."""

    @pytest.mark.parametrize("scenario", CENARIOS)
    def test_3layer_isotropic(self, scenario):
        """3 camadas isotrópicas — paridade FLAT vs legacy."""
        model = _model_3layer()
        positions_z = np.linspace(-1.0, 4.0, 30)

        cfg_legacy = SimulationConfig(use_flat_prange=False, parallel=True)
        cfg_flat = SimulationConfig(use_flat_prange=True, parallel=True)

        res_legacy = simulate_multi(
            **model, positions_z=positions_z, cfg=cfg_legacy, **scenario
        )
        res_flat = simulate_multi(
            **model, positions_z=positions_z, cfg=cfg_flat, **scenario
        )

        np.testing.assert_array_equal(
            res_flat.H_tensor,
            res_legacy.H_tensor,
            err_msg=f"FLAT vs legacy diferem em 3layer isotropic / {scenario}",
        )

    @pytest.mark.parametrize("scenario", CENARIOS)
    def test_5layer_tiv(self, scenario):
        """5 camadas TIV — paridade FLAT vs legacy."""
        model = _model_5layer_tiv()
        positions_z = np.linspace(-2.0, 8.0, 30)

        cfg_legacy = SimulationConfig(use_flat_prange=False, parallel=True)
        cfg_flat = SimulationConfig(use_flat_prange=True, parallel=True)

        res_legacy = simulate_multi(
            **model, positions_z=positions_z, cfg=cfg_legacy, **scenario
        )
        res_flat = simulate_multi(
            **model, positions_z=positions_z, cfg=cfg_flat, **scenario
        )

        np.testing.assert_array_equal(
            res_flat.H_tensor,
            res_legacy.H_tensor,
            err_msg=f"FLAT vs legacy diferem em 5layer TIV / {scenario}",
        )

    @pytest.mark.parametrize("scenario", CENARIOS)
    def test_high_rho_carbonate(self, scenario):
        """Alta resistividade (ρ=5000 Ω·m) — paridade FLAT vs legacy."""
        model = _model_high_rho()
        positions_z = np.linspace(-1.0, 5.0, 30)

        cfg_legacy = SimulationConfig(use_flat_prange=False, parallel=True)
        cfg_flat = SimulationConfig(use_flat_prange=True, parallel=True)

        res_legacy = simulate_multi(
            **model, positions_z=positions_z, cfg=cfg_legacy, **scenario
        )
        res_flat = simulate_multi(
            **model, positions_z=positions_z, cfg=cfg_flat, **scenario
        )

        np.testing.assert_array_equal(
            res_flat.H_tensor,
            res_legacy.H_tensor,
            err_msg=f"FLAT vs legacy diferem em alta-rho / {scenario}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Cobertura específica de multi-frequência (caso onde FLAT mais ganha)
# ═══════════════════════════════════════════════════════════════════════════


class TestFlatMultiFreqParity:
    """Cobertura granular: FLAT == legacy para diferentes nf."""

    @pytest.mark.parametrize("nf", [1, 2, 4, 8, 10])
    def test_multi_freq_parity(self, nf):
        """nf ∈ {1, 2, 4, 8, 10} — FLAT bit-exato com legacy."""
        model = _model_5layer_tiv()
        positions_z = np.linspace(-2.0, 8.0, 50)
        # Distribui frequências em escala log
        frequencies_hz = list(np.logspace(3.0, 5.5, nf))

        cfg_legacy = SimulationConfig(use_flat_prange=False, parallel=True)
        cfg_flat = SimulationConfig(use_flat_prange=True, parallel=True)

        res_legacy = simulate_multi(
            **model,
            positions_z=positions_z,
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            frequencies_hz=frequencies_hz,
            cfg=cfg_legacy,
        )
        res_flat = simulate_multi(
            **model,
            positions_z=positions_z,
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            frequencies_hz=frequencies_hz,
            cfg=cfg_flat,
        )

        diff = np.max(np.abs(res_flat.H_tensor - res_legacy.H_tensor))
        assert diff == 0.0, (
            f"FLAT vs legacy quebrou bit-exatness em nf={nf}: "
            f"max|diff|={diff} (esperado 0.0)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Decomposição de índice flat (k → i_tr, i_ang, j, i_f)
# ═══════════════════════════════════════════════════════════════════════════


class TestFlatIndexDecomposition:
    """Valida que a decomposição flat cobre exatamente n_total tarefas."""

    @pytest.mark.parametrize(
        "nTR,nAng,n_pos,nf",
        [
            (1, 1, 30, 1),
            (1, 1, 30, 4),
            (3, 4, 30, 1),
            (3, 4, 30, 4),
            (2, 2, 100, 8),
        ],
    )
    def test_index_coverage(self, nTR, nAng, n_pos, nf):
        """Decomposição flat cobre todos (i_tr, i_ang, j, i_f) sem duplicatas."""
        nAngles = nAng
        n_combos = nTR * nAng
        n_combo_pos_f = n_pos * nf
        n_total = n_combos * n_combo_pos_f

        seen = set()
        for k in range(n_total):
            i_combo = k // n_combo_pos_f
            rem = k % n_combo_pos_f
            j = rem // nf
            i_f = rem % nf
            i_tr = i_combo // nAngles
            i_ang = i_combo % nAngles

            tup = (i_tr, i_ang, j, i_f)
            assert tup not in seen, f"Duplicata em k={k}: {tup}"
            seen.add(tup)

        # Cobertura completa: produto cartesiano exato
        assert (
            len(seen) == nTR * nAng * n_pos * nf
        ), f"Cobertura incompleta: {len(seen)} != {nTR*nAng*n_pos*nf}"


# ═══════════════════════════════════════════════════════════════════════════
# Regressão Cenário E (nf=1) — meta: sem regressão vs v2.21
# ═══════════════════════════════════════════════════════════════════════════


class TestFlatNoRegressionCenarioE:
    """Cenário E (n_pos=600, nf=1, 1TR, 1ang): meta v2.21 ≥120k mod/h."""

    def test_cenario_e_smoke_correctness(self):
        """Smoke test correção em Cenário E (sem benchmark)."""
        model = _model_5layer_tiv()
        positions_z = np.linspace(-2.0, 8.0, 600)

        cfg_legacy = SimulationConfig(use_flat_prange=False, parallel=True)
        cfg_flat = SimulationConfig(use_flat_prange=True, parallel=True)

        res_legacy = simulate_multi(
            **model,
            positions_z=positions_z,
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            frequencies_hz=[20000.0],
            cfg=cfg_legacy,
        )
        res_flat = simulate_multi(
            **model,
            positions_z=positions_z,
            tr_spacings_m=[1.0],
            dip_degs=[0.0],
            frequencies_hz=[20000.0],
            cfg=cfg_flat,
        )
        np.testing.assert_array_equal(res_flat.H_tensor, res_legacy.H_tensor)


# ═══════════════════════════════════════════════════════════════════════════
# Paridade Fortran <1e-12 — gate inviolável
# ═══════════════════════════════════════════════════════════════════════════


class TestFlatFortranParity:
    """Sprint v2.22.3 — FLAT preserva paridade Fortran <1e-12.

    Estratégia: como FLAT é bit-exato com legacy (validado em
    TestFlatVsLegacyBitExact), e legacy já tem paridade Fortran <1e-12
    (validado em test_simulation_compare_fortran.py), por transitividade
    FLAT também tem paridade <1e-12. Aqui validamos que o caminho FLAT
    produz valores físicos sensatos (não NaN, não Inf, não zero).
    """

    @pytest.mark.parametrize("scenario", CENARIOS)
    def test_no_nan_inf_in_flat_output(self, scenario):
        """FLAT output não contém NaN/Inf em modelos canônicos."""
        model = _model_5layer_tiv()
        positions_z = np.linspace(-2.0, 8.0, 30)

        cfg_flat = SimulationConfig(use_flat_prange=True, parallel=True)
        res = simulate_multi(**model, positions_z=positions_z, cfg=cfg_flat, **scenario)

        assert np.all(
            np.isfinite(res.H_tensor.real)
        ), f"NaN/Inf detectado em FLAT real / {scenario}"
        assert np.all(
            np.isfinite(res.H_tensor.imag)
        ), f"NaN/Inf detectado em FLAT imag / {scenario}"
        # Sanidade: campo magnético não deve ser zero em todas as posições
        assert np.any(
            np.abs(res.H_tensor) > 1e-30
        ), f"FLAT produziu tensor zero / {scenario}"
