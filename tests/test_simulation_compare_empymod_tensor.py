# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_compare_empymod_tensor.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes — empymod Tensor 9-comp (Sprint 4.2)               ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Produção (opt-in, requer empymod instalado)              ║
# ║  Framework   : pytest + empymod (opcional)                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  ESCOPO Sprint 4.2 (vs Sprint 4.3 futuro)                                ║
# ║    Sprint 4.2 entrega INFRA: COMPONENT_AB_MAP, COMPONENT_TENSOR_INDEX,   ║
# ║    TensorComparisonResult, mapeamento aniso=λ. Bit-exactness numérica   ║
# ║    entre Numba e empymod (reconciliar convenções temporais + fator de   ║
# ║    normalização) é Sprint 4.3 (PR #12). Por isso estes testes validam:  ║
# ║      • Função roda sem erro                                              ║
# ║      • Shapes do TensorComparisonResult                                   ║
# ║      • COMPONENT_AB_MAP completo (9 chaves)                              ║
# ║      • Resultados finitos (sem NaN/Inf)                                  ║
# ║      • Skip gracioso quando empymod ausente                              ║
# ║                                                                           ║
# ║    NÃO valida bit-exactness — isso requer Sprint 4.3.                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da infraestrutura de comparação Numba ↔ empymod 9-comp (Sprint 4.2)."""
from __future__ import annotations

import numpy as np
import pytest

from geosteering_ai.simulation.validation import (
    COMPONENT_AB_MAP,
    COMPONENT_TENSOR_INDEX,
    HAS_EMPYMOD,
    TensorComparisonResult,
    compare_numba_empymod_tensor,
)

# ──────────────────────────────────────────────────────────────────────────────
# Skip module-level se empymod não instalado (todos os testes ficam skipped)
# ──────────────────────────────────────────────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not HAS_EMPYMOD, reason="empymod não instalado (opt-in: pip install empymod)"
)


# ──────────────────────────────────────────────────────────────────────────────
# TestComponentAbMap — sanity checks do mapeamento
# ──────────────────────────────────────────────────────────────────────────────


class TestComponentAbMap:
    """Mapas de componente devem cobrir os 9 elementos do tensor magnético."""

    def test_ab_map_has_9_components(self) -> None:
        assert len(COMPONENT_AB_MAP) == 9

    def test_tensor_index_has_9_components(self) -> None:
        assert len(COMPONENT_TENSOR_INDEX) == 9
        # Índices devem cobrir 0..8 sem duplicação
        assert sorted(COMPONENT_TENSOR_INDEX.values()) == list(range(9))

    def test_ab_codes_are_valid(self) -> None:
        """Códigos `ab` devem ser combinações 1/2/5 dos eixos x/y/z."""
        valid_digits = {1, 2, 5}
        for comp, ab in COMPONENT_AB_MAP.items():
            tens, units = divmod(ab, 10)
            assert tens in valid_digits, f"{comp}: dezena {tens} inválida"
            assert units in valid_digits, f"{comp}: unidade {units} inválida"

    def test_keys_match_between_maps(self) -> None:
        assert set(COMPONENT_AB_MAP) == set(COMPONENT_TENSOR_INDEX)


# ──────────────────────────────────────────────────────────────────────────────
# TestTensorComparisonInfra — função roda + dataclass tem campos esperados
# ──────────────────────────────────────────────────────────────────────────────


class TestTensorComparisonInfra:
    """Sprint 4.2 valida APENAS a INFRA (não bit-exactness)."""

    @pytest.fixture
    def isotropic_result(self) -> TensorComparisonResult:
        return compare_numba_empymod_tensor(
            rho_h=np.array([1.0, 100.0, 1.0]),
            esp=np.array([5.0]),
            depth_src=2.0,
            depth_rec=3.0,
            offset_x=0.5,
            freqs_hz=np.array([20000.0]),
        )

    def test_returns_tensor_comparison_result(self, isotropic_result) -> None:
        assert isinstance(isotropic_result, TensorComparisonResult)

    def test_h_numba_shape_correct(self, isotropic_result) -> None:
        # (n_positions=1, nf=1, 9 componentes)
        assert isotropic_result.H_numba.shape == (1, 1, 9)

    def test_compares_all_9_components(self, isotropic_result) -> None:
        assert len(isotropic_result.components_compared) == 9
        assert len(isotropic_result.components_failed) == 0

    def test_no_nan_inf_in_numba(self, isotropic_result) -> None:
        assert np.all(np.isfinite(isotropic_result.H_numba))

    def test_no_nan_inf_in_empymod(self, isotropic_result) -> None:
        for comp, arr in isotropic_result.H_empymod_per_component.items():
            assert np.all(np.isfinite(arr)), f"{comp} contém NaN/Inf"

    def test_summary_renders(self, isotropic_result) -> None:
        s = isotropic_result.summary()
        assert "Sprint 4.2" in s
        assert "Hxx" in s and "Hzz" in s


# ──────────────────────────────────────────────────────────────────────────────
# TestTivAnisotropy — meio TIV detectado e mapeado via aniso=λ
# ──────────────────────────────────────────────────────────────────────────────


class TestTivAnisotropy:
    """Anisotropia λ²=2 (rho_v=2·rho_h) deve ser mapeada para `aniso=√2`."""

    def test_tiv_lambda_2_runs(self) -> None:
        result = compare_numba_empymod_tensor(
            rho_h=np.array([1.0, 100.0, 1.0]),
            rho_v=np.array([1.0, 200.0, 1.0]),  # λ²=2 na camada-alvo
            esp=np.array([5.0]),
            depth_src=2.0,
            depth_rec=3.0,
            offset_x=0.5,
            freqs_hz=np.array([20000.0]),
        )
        assert isinstance(result, TensorComparisonResult)
        # Note de TIV deve ter sido emitida
        assert any("TIV" in n for n in result.notes)
        assert len(result.components_compared) == 9

    def test_isotropic_does_not_emit_tiv_note(self) -> None:
        result = compare_numba_empymod_tensor(
            rho_h=np.array([1.0, 100.0, 1.0]),
            esp=np.array([5.0]),
            freqs_hz=np.array([20000.0]),
        )
        # Sem rho_v, default isotrópico → não menciona TIV
        assert not any("TIV" in n for n in result.notes)


# ──────────────────────────────────────────────────────────────────────────────
# TestSubsetComponents — caller pode pedir subset (ex: só Hzz e Hxx)
# ──────────────────────────────────────────────────────────────────────────────


class TestSubsetComponents:
    """API permite restringir o subset de componentes comparados."""

    def test_subset_of_2_components(self) -> None:
        result = compare_numba_empymod_tensor(
            rho_h=np.array([1.0, 100.0, 1.0]),
            esp=np.array([5.0]),
            freqs_hz=np.array([20000.0]),
            components=["Hzz", "Hxx"],
        )
        assert set(result.components_compared) == {"Hzz", "Hxx"}
        assert len(result.components_failed) == 0

    def test_unknown_component_appears_in_failed(self) -> None:
        result = compare_numba_empymod_tensor(
            rho_h=np.array([1.0, 100.0, 1.0]),
            esp=np.array([5.0]),
            freqs_hz=np.array([20000.0]),
            components=["Hzz", "Hbogus"],
        )
        assert "Hzz" in result.components_compared
        assert any(c[0] == "Hbogus" for c in result.components_failed)
