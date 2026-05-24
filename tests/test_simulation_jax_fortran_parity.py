# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_fortran_parity.py                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Sprint O0 (T1.1) — paridade JAX-bucketed vs Fortran tatu.x ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-24 (Sprint O0 do plano de otimização JAX GPU)      ║
# ║  Status      : Produção (gate Tier 1 anti-regressão)                      ║
# ║  Framework   : pytest + JAX + Fortran tatu.x (via compare_fortran_python) ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Teste T1.1 — paridade direta JAX (estratégia ``bucketed``) vs Fortran ``tatu.x``.

**Motivação**: até esta sprint, toda a paridade do simulador JAX era validada
de forma **transitiva**: ``JAX vs Numba`` (testes existentes) + ``Numba vs
Fortran`` (testes existentes). Se ambos ``Numba`` e ``JAX`` regredirem no
mesmo commit, a regressão é invisível. Este teste fecha o triângulo,
exercitando diretamente ``forward_pure_jax`` com ``cfg.jax_strategy="bucketed"``
contra a saída do binário Fortran ``tatu.x``.

**Cobertura**: 3 modelos canônicos com complexidade crescente:

  - ``oklahoma_3``  — 3 camadas TIV (baseline mínimo)
  - ``devine_8``    — 8 camadas isotrópico (multi-interface)
  - ``hou_7``       — 7 camadas TIV (referência Hou et al. 2006)

**Tolerância**: `<1e-10` (paridade Fortran de fato). O `compare_fortran_python`
usa por default ``DEFAULT_TOL_ABS = 1e-6``, mas o caminho ``jax_hybrid``
encadeia as mesmas operações Hankel → propagação que o Numba, portanto
esperamos atingir `1e-10` ou melhor.

**Skip gracioso**:
  - JAX não instalado
  - ``tatu.x`` não executável no OS atual (e.g., binário macOS em CI Linux)
"""

from __future__ import annotations

import pytest

# ── Skip global se JAX ou tatu.x ausentes (Sprint v2.40 D9) ────────────────────
from geosteering_ai.simulation._jax import HAS_JAX
from tests._fortran_helpers import _tatu_runnable

pytestmark = [
    pytest.mark.skipif(
        not HAS_JAX,
        reason="JAX não instalado — T1.1 requer `pip install jax[cpu]` ou `jax[cuda12]`",
    ),
    pytest.mark.skipif(
        not _tatu_runnable(),
        reason="tatu.x não executável neste OS (ex: binário macOS em CI Linux)",
    ),
    pytest.mark.gpu,
]

if HAS_JAX:
    import jax

    jax.config.update("jax_enable_x64", True)


# ══════════════════════════════════════════════════════════════════════════════
# T1.1 — Paridade JAX bucketed vs Fortran tatu.x em 3 modelos canônicos
# ══════════════════════════════════════════════════════════════════════════════

CANONICAL_MODELS_TO_TEST = ["oklahoma_3", "devine_8", "hou_7"]
"""Lista de modelos canônicos a serem testados no T1.1.

Selecionados para cobrir gradiente de complexidade:
  - oklahoma_3 (3 camadas TIV)
  - devine_8 (8 camadas isotrópico)
  - hou_7 (7 camadas TIV, referência Hou et al. 2006)

Note:
    Sprint O0 NÃO inclui oklahoma_28 (28 camadas) — modelo maior fica para
    Tier 2 (T2.x) para manter tempo de coleta CI < 5 min em modo full.
"""


@pytest.mark.parametrize("model_name", CANONICAL_MODELS_TO_TEST)
def test_forward_pure_bucketed_fortran_parity_canonical(model_name: str) -> None:
    """Valida paridade direta JAX-bucketed vs Fortran ``tatu.x`` em modelo canônico.

    Este teste reusa a infraestrutura existente em
    :func:`geosteering_ai.simulation.validation.compare_fortran.compare_fortran_python`,
    invocando o backend ``"jax_hybrid"`` (que internamente usa
    ``cfg.jax_strategy="bucketed"`` por default).

    Args:
        model_name: Nome do modelo canônico (``oklahoma_3``, ``devine_8``, ``hou_7``).

    Asserts:
        - ``result.success_jax_hybrid`` is True (paridade dentro da tolerância)
        - ``result.max_abs_error_jax_hybrid < 1e-10`` (paridade científica)
        - Sem NaN/Inf em qualquer componente

    Note:
        Falha indica regressão direta no caminho ``forward_pure_jax`` bucketed.
        Esta é uma proteção anti-regressão crítica antes de Sprint O1+.
    """
    from geosteering_ai.simulation.validation.compare_fortran import (
        compare_fortran_python,
    )

    # ── Invoca pipeline padrão JAX-hybrid vs Fortran ──────────────────────────
    # ``compare_fortran_python`` resolve internamente: monta model.in, executa
    # tatu.x, parseia .dat, roda forward_pure_jax (bucketed), compara campo a
    # campo via max(|JAX - Fortran|).
    result = compare_fortran_python(
        model_name=model_name,
        backends=["jax_hybrid"],
        tol_abs=1e-10,  # tolerância estrita para gate Tier 1
    )

    # ── Validações de gate ────────────────────────────────────────────────────
    assert (
        result.get("success_jax_hybrid", False) is True
    ), f"T1.1 paridade JAX-bucketed vs Fortran FALHOU em {model_name}: {result}"

    max_err = result.get("max_abs_error_jax_hybrid", float("inf"))
    assert max_err < 1e-10, (
        f"T1.1 paridade JAX-bucketed vs Fortran {model_name}: "
        f"max|diff|={max_err:.3e} excede tolerância 1e-10"
    )


def test_t11_fortran_parity_smoke() -> None:
    """Smoke test mínimo: pipeline JAX-Fortran roda sem exceção em oklahoma_3.

    Útil para detectar quebras de infraestrutura (ImportError, FileNotFoundError,
    OSError) cedo, antes de rodar os 3 modelos parametricos completos.
    """
    from geosteering_ai.simulation.validation.compare_fortran import (
        compare_fortran_python,
    )

    result = compare_fortran_python(
        model_name="oklahoma_3",
        backends=["jax_hybrid"],
        tol_abs=1e-6,  # smoke: aceitar qualquer paridade básica
    )
    assert result is not None, "T1.1 smoke: compare_fortran_python retornou None"
    assert "jax_hybrid" in str(result), "T1.1 smoke: backend jax_hybrid não consumido"
