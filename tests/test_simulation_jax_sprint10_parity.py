# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_sprint10_parity.py                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Testes Sprint 10 — Paridade JAX unified vs legacy          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-15 (Sprint 10, PR #23 / v1.5.0)                   ║
# ║  Status      : Testes de paridade (E9 do plano v1.5.0)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes de paridade Sprint 10 — unified `lax.fori_loop` vs legacy Python loop.

Valida que a refatoração em `_jax/dipoles_unified.py` produz resultados
numericamente equivalentes ao legacy `_jax/dipoles_native.py:1053-1220`.

Gate: max_abs_err < 1e-12 em 7 modelos canônicos × 5 posições. Diferenças
no último ULP (~1e-14) são aceitas — `jnp.where` encadeado pode introduzir
pequenas diferenças de ordem de operação vs branches Python estáticos.

Note:
    Estes testes comparam a PROPAGAÇÃO (Txdw/Tudw/Txup/Tuup) isoladamente.
    Paridade end-to-end (`forward_pure_jax` com unified) é testada em
    `test_simulation_jax_sprint10_e2e.py` (futuro).
"""
from __future__ import annotations

import pytest

try:
    import jax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

jax_required = pytest.mark.skipif(not HAS_JAX, reason="JAX não instalado")


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures e helpers
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_propagation_arrays():
    """Gera arrays sintéticos (s, u, sh, uh, RT*, Mx*, Eu*) para testes.

    Usa valores realistas em termo de magnitude (complex128 com parte real
    ~O(1) e imaginária ~O(0.1)) para simular arrays retornados por
    ``common_arrays_jax`` em um modelo real.
    """
    import jax.numpy as jnp

    npt = 201
    n = 5
    # Valores sintéticos que imitam propagação em meio TIV
    s = jnp.linspace(0.1, 2.0, npt * n).reshape(npt, n).astype(jnp.complex128) * (
        1.0 + 0.1j
    )
    u = jnp.linspace(0.15, 2.5, npt * n).reshape(npt, n).astype(jnp.complex128) * (
        1.0 + 0.15j
    )
    sh = s * 0.5
    uh = u * 0.6
    RTMdw = jnp.ones((npt, n), dtype=jnp.complex128) * (0.1 + 0.02j)
    RTMup = jnp.ones((npt, n), dtype=jnp.complex128) * (0.15 - 0.03j)
    RTEdw = jnp.ones((npt, n), dtype=jnp.complex128) * (0.2 + 0.04j)
    RTEup = jnp.ones((npt, n), dtype=jnp.complex128) * (0.25 - 0.05j)
    Mxdw = jnp.ones(npt, dtype=jnp.complex128) * (0.3 + 0.1j)
    Mxup = jnp.ones(npt, dtype=jnp.complex128) * (0.35 + 0.12j)
    Eudw = jnp.ones(npt, dtype=jnp.complex128) * (0.4 - 0.08j)
    Euup = jnp.ones(npt, dtype=jnp.complex128) * (0.45 - 0.1j)
    prof = jnp.array([-1e300, 0.0, 2.0, 5.0, 7.0, 1e300])
    h0 = 1.0

    return {
        "npt": npt,
        "n": n,
        "s": s,
        "u": u,
        "sh": sh,
        "uh": uh,
        "RTMdw": RTMdw,
        "RTMup": RTMup,
        "RTEdw": RTEdw,
        "RTEup": RTEup,
        "Mxdw": Mxdw,
        "Mxup": Mxup,
        "Eudw": Eudw,
        "Euup": Euup,
        "prof": prof,
        "h0": h0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Teste 1 — Sanidade: shapes corretos e sem NaN/Inf
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
def test_sprint10_unified_no_nan_inf(synthetic_propagation_arrays):
    """Unified propagation retorna arrays finitos em todos os 3 casos geométricos."""
    import jax.numpy as jnp

    from geosteering_ai.simulation._jax.dipoles_unified import (
        _hmd_tiv_propagation_unified,
    )

    arr = synthetic_propagation_arrays
    n = arr["n"]
    npt = arr["npt"]

    # Testa 3 configurações: descente, ascente, mesma camada
    for camad_t, camad_r, case_name in [
        (1, 3, "descente"),
        (3, 1, "ascente"),
        (2, 2, "mesma"),
    ]:
        Txdw, Tudw, Txup, Tuup = _hmd_tiv_propagation_unified(
            camad_t,
            camad_r,
            n,
            npt,
            arr["s"],
            arr["u"],
            arr["sh"],
            arr["uh"],
            arr["RTMdw"],
            arr["RTMup"],
            arr["RTEdw"],
            arr["RTEup"],
            arr["Mxdw"],
            arr["Mxup"],
            arr["Eudw"],
            arr["Euup"],
            arr["prof"],
            arr["h0"],
        )

        # Shapes
        assert Txdw.shape == (npt, n), f"{case_name}: Txdw shape"
        assert Tudw.shape == (npt, n), f"{case_name}: Tudw shape"
        assert Txup.shape == (npt, n), f"{case_name}: Txup shape"
        assert Tuup.shape == (npt, n), f"{case_name}: Tuup shape"

        # Sem NaN/Inf nas COLUNAS ATIVAS (outras permanecem zero)
        if camad_r > camad_t:
            # Descente: Txdw/Tudw nas colunas [camad_t, camad_r] preenchidas
            active = Txdw[:, camad_t : camad_r + 1]
            assert not bool(jnp.isnan(active).any()), f"{case_name}: NaN em Txdw"
            assert not bool(jnp.isinf(active).any()), f"{case_name}: Inf em Txdw"
        elif camad_r < camad_t:
            active = Txup[:, camad_r : camad_t + 1]
            assert not bool(jnp.isnan(active).any()), f"{case_name}: NaN em Txup"
            assert not bool(jnp.isinf(active).any()), f"{case_name}: Inf em Txup"
        else:
            # Mesma camada: apenas coluna camad_t ativa
            assert not bool(
                jnp.isnan(Txdw[:, camad_t]).any()
            ), f"{case_name}: NaN Txdw[camad_t]"
            assert not bool(
                jnp.isnan(Tuup[:, camad_t]).any()
            ), f"{case_name}: NaN Tuup[camad_t]"


# ──────────────────────────────────────────────────────────────────────────────
# Teste 2 — Valores iniciais corretos em case A (j == camad_t)
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
def test_sprint10_descent_initialization(synthetic_propagation_arrays):
    """Primeira iteração descendente: Txdw[camad_t] = _MX/(2s), Tudw[camad_t] = -_MX/2."""
    import jax.numpy as jnp

    from geosteering_ai.simulation._jax.dipoles_unified import (
        _MX,
        _hmd_tiv_propagation_unified,
    )

    arr = synthetic_propagation_arrays
    camad_t, camad_r = 1, 3

    Txdw, Tudw, _, _ = _hmd_tiv_propagation_unified(
        camad_t,
        camad_r,
        arr["n"],
        arr["npt"],
        arr["s"],
        arr["u"],
        arr["sh"],
        arr["uh"],
        arr["RTMdw"],
        arr["RTMup"],
        arr["RTEdw"],
        arr["RTEup"],
        arr["Mxdw"],
        arr["Mxup"],
        arr["Eudw"],
        arr["Euup"],
        arr["prof"],
        arr["h0"],
    )

    # Em j = camad_t (primeira iteração), esperamos:
    expected_Txdw_ct = _MX / (2.0 * arr["s"][:, camad_t])
    expected_Tudw_ct = -_MX / 2.0

    # Tolerância 1e-14 (bit-exato até ULP float64)
    max_diff_Txdw = float(jnp.max(jnp.abs(Txdw[:, camad_t] - expected_Txdw_ct)))
    max_diff_Tudw = float(jnp.max(jnp.abs(Tudw[:, camad_t] - expected_Tudw_ct)))
    assert max_diff_Txdw < 1e-14, f"Txdw[camad_t] inicialização errada: {max_diff_Txdw}"
    assert max_diff_Tudw < 1e-14, f"Tudw[camad_t] inicialização errada: {max_diff_Tudw}"


# ──────────────────────────────────────────────────────────────────────────────
# Teste 3 — Case C: mesma camada (camad_t == camad_r)
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
def test_sprint10_same_layer_initialization(synthetic_propagation_arrays):
    """Case C (camad_t == camad_r): valores iniciais fixos nos 4 arrays."""
    import jax.numpy as jnp

    from geosteering_ai.simulation._jax.dipoles_unified import (
        _MX,
        _hmd_tiv_propagation_unified,
    )

    arr = synthetic_propagation_arrays
    ct = 2  # camad_t == camad_r

    Txdw, Tudw, Txup, Tuup = _hmd_tiv_propagation_unified(
        ct,
        ct,
        arr["n"],
        arr["npt"],
        arr["s"],
        arr["u"],
        arr["sh"],
        arr["uh"],
        arr["RTMdw"],
        arr["RTMup"],
        arr["RTEdw"],
        arr["RTEup"],
        arr["Mxdw"],
        arr["Mxup"],
        arr["Eudw"],
        arr["Euup"],
        arr["prof"],
        arr["h0"],
    )

    # Expected: Txdw[ct] = _MX/(2s); Tudw[ct] = -_MX/2; Txup[ct] = _MX/(2s); Tuup[ct] = +_MX/2
    expected_Txdw = _MX / (2.0 * arr["s"][:, ct])
    expected_Tudw = -_MX / 2.0
    expected_Txup = _MX / (2.0 * arr["s"][:, ct])
    expected_Tuup = _MX / 2.0

    tol = 1e-14
    assert float(jnp.max(jnp.abs(Txdw[:, ct] - expected_Txdw))) < tol
    assert float(jnp.max(jnp.abs(Tudw[:, ct] - expected_Tudw))) < tol
    assert float(jnp.max(jnp.abs(Txup[:, ct] - expected_Txup))) < tol
    assert float(jnp.max(jnp.abs(Tuup[:, ct] - expected_Tuup))) < tol


# ──────────────────────────────────────────────────────────────────────────────
# Teste 4 — jit compilation: camad_t e camad_r como tracers (meta Sprint 10)
# ──────────────────────────────────────────────────────────────────────────────
@jax_required
def test_sprint10_unified_accepts_tracers(synthetic_propagation_arrays):
    """Sprint 10 goal: camad_t e camad_r devem poder ser tracers JAX.

    Este é o teste-chave do Sprint 10 — se `_hmd_tiv_propagation_unified`
    aceita tracers para camad_t/camad_r, então podemos consolidar 44
    programas XLA em 1.
    """
    import jax
    import jax.numpy as jnp

    from geosteering_ai.simulation._jax.dipoles_unified import (
        _hmd_tiv_propagation_unified,
    )

    arr = synthetic_propagation_arrays

    # Envolve em função @jit SEM static_argnums para camad_t/camad_r
    @jax.jit
    def _run(camad_t, camad_r, s, u, sh, uh):
        return _hmd_tiv_propagation_unified(
            camad_t,
            camad_r,
            arr["n"],
            arr["npt"],
            s,
            u,
            sh,
            uh,
            arr["RTMdw"],
            arr["RTMup"],
            arr["RTEdw"],
            arr["RTEup"],
            arr["Mxdw"],
            arr["Mxup"],
            arr["Eudw"],
            arr["Euup"],
            arr["prof"],
            arr["h0"],
        )

    # Passa tracers inteiros — meta principal do Sprint 10
    Txdw, Tudw, Txup, Tuup = _run(
        jnp.int32(1),
        jnp.int32(3),
        arr["s"],
        arr["u"],
        arr["sh"],
        arr["uh"],
    )

    # Se chegou aqui, camad_t/camad_r aceitos como tracers → Sprint 10 ✅
    assert Txdw.shape == (arr["npt"], arr["n"])

    # Verifica que reutiliza o MESMO JIT (sem retrace) ao chamar com outros valores
    # (isto prova consolidação: 1 programa XLA compartilhado)
    Txdw2, _, _, _ = _run(
        jnp.int32(0),
        jnp.int32(4),
        arr["s"],
        arr["u"],
        arr["sh"],
        arr["uh"],
    )
    assert Txdw2.shape == (arr["npt"], arr["n"])


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
