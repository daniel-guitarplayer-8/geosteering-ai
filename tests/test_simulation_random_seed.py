# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_random_seed.py                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager v2.19 (Sprint 19.1)                     ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-02                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : pytest                                                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Valida o controle de semente PRNG no gerador estocástico de modelos   ║
# ║    TIV (Sprint 19.1). Antes da v2.19, ``rng_seed=42`` era hardcoded em   ║
# ║    ``simulation_manager.py:8088``, gerando sempre os mesmos modelos a   ║
# ║    cada execução — bug funcional que impedia ensembles diversos.         ║
# ║                                                                           ║
# ║  COBERTURA DOS TESTES                                                     ║
# ║    1. ``rng_seed=None`` (default v2.19) produz modelos distintos         ║
# ║    2. Semente fixa reproduz modelos bit-a-bit                            ║
# ║    3. ``return_seed=True`` retorna a semente realmente usada             ║
# ║    4. Smoke test seed=42 ainda funciona (compat)                          ║
# ║                                                                           ║
# ║  REGRESSÃO ALVO                                                           ║
# ║    Bug funcional v2.18: cada "Iniciar Simulação" gerava o mesmo lote    ║
# ║    de N modelos (mesma sequência ρₕ/ρᵥ/esp). Fix v2.19: padrão é        ║
# ║    aleatório; reprodutibilidade fica disponível via UI (checkbox).       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes do gerador de seed PRNG do Simulation Manager (v2.19 Sprint 19.1)."""
from __future__ import annotations

import pytest

from geosteering_ai.simulation.tests.sm_model_gen import (
    GenConfig,
    _resolve_rng_seed,
    generate_models,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures comuns
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture
def small_cfg() -> GenConfig:
    """Configuração mínima e rápida para testes (Sobol determinístico)."""
    return GenConfig(generator="sobol")


# ──────────────────────────────────────────────────────────────────────────
# 1. rng_seed=None gera modelos distintos a cada chamada
# ──────────────────────────────────────────────────────────────────────────
def test_random_seed_produces_different_models(small_cfg: GenConfig) -> None:
    """``rng_seed=None`` deve gerar sequências de modelos distintas.

    A regressão alvo é o bug v2.18 onde o seed era hardcoded em 42, fazendo
    com que duas execuções consecutivas produzissem modelos idênticos. Com
    o default v2.19 (None), cada chamada usa uma semente aleatória de 63
    bits via :func:`secrets.randbits`, garantindo diversidade estatística.
    """
    models_1 = generate_models(small_cfg, n_models=5, rng_seed=None)
    models_2 = generate_models(small_cfg, n_models=5, rng_seed=None)

    # Pelo menos um modelo do primeiro lote deve diferir do correspondente
    # no segundo. Comparamos rho_h (sempre presente em qualquer perfil).
    different = any(models_1[i]["rho_h"] != models_2[i]["rho_h"] for i in range(5))
    assert different, (
        "rng_seed=None deveria gerar modelos distintos a cada chamada — "
        "se este teste falhar, a semente está sendo fixada em algum lugar."
    )


# ──────────────────────────────────────────────────────────────────────────
# 2. Semente fixa reproduz modelos bit-a-bit
# ──────────────────────────────────────────────────────────────────────────
def test_fixed_seed_reproduces_models(small_cfg: GenConfig) -> None:
    """Semente fixa deve gerar a mesma sequência em chamadas repetidas.

    Reprodutibilidade é importante para experimentos científicos — usuário
    desliga o checkbox "Semente aleatória" e fixa um inteiro para obter
    modelos idênticos entre execuções.
    """
    models_a = generate_models(small_cfg, n_models=5, rng_seed=123)
    models_b = generate_models(small_cfg, n_models=5, rng_seed=123)

    for i in range(5):
        assert (
            models_a[i]["rho_h"] == models_b[i]["rho_h"]
        ), f"Modelo {i} difere entre chamadas com mesmo seed=123"
        assert models_a[i]["rho_v"] == models_b[i]["rho_v"]
        assert models_a[i]["thicknesses"] == models_b[i]["thicknesses"]


# ──────────────────────────────────────────────────────────────────────────
# 3. return_seed=True retorna semente realmente usada
# ──────────────────────────────────────────────────────────────────────────
def test_return_seed_yields_actual_seed(small_cfg: GenConfig) -> None:
    """``return_seed=True`` deve retornar a tupla (models, actual_seed).

    Permite à GUI/logs registrar qual semente foi usada mesmo no modo
    aleatório, viabilizando reprodutibilidade post-hoc.
    """
    result = generate_models(small_cfg, n_models=2, rng_seed=None, return_seed=True)
    assert isinstance(result, tuple) and len(result) == 2
    models, actual_seed = result
    assert isinstance(models, list) and len(models) == 2
    assert isinstance(actual_seed, int)
    assert actual_seed >= 0, "Semente não pode ser negativa"
    # 63 bits = max 2**63 - 1 (cabe em int64 signed)
    assert actual_seed < (1 << 63), "Semente deve caber em 63 bits"


# ──────────────────────────────────────────────────────────────────────────
# 4. Compatibilidade legada (smoke test seed=42)
# ──────────────────────────────────────────────────────────────────────────
def test_legacy_seed_42_still_works(small_cfg: GenConfig) -> None:
    """``rng_seed=42`` continua produzindo a mesma sequência (smoke test).

    O smoke test ``simulation_manager.py:_run_smoke_test`` (linha 10269)
    intencionalmente usa seed=42 para determinismo. Este teste garante
    que essa API legada não quebrou na v2.19.
    """
    models_x = generate_models(small_cfg, n_models=3, rng_seed=42)
    models_y = generate_models(small_cfg, n_models=3, rng_seed=42)
    assert all(
        models_x[i]["rho_h"] == models_y[i]["rho_h"] for i in range(3)
    ), "seed=42 não está reproduzindo modelos — quebra de smoke test"


# ──────────────────────────────────────────────────────────────────────────
# 5. _resolve_rng_seed cobertura de None vs int
# ──────────────────────────────────────────────────────────────────────────
def test_resolve_rng_seed_with_none_returns_random_int() -> None:
    """``_resolve_rng_seed(None)`` deve retornar inteiro de 63 bits aleatório."""
    s1 = _resolve_rng_seed(None)
    s2 = _resolve_rng_seed(None)
    assert isinstance(s1, int) and isinstance(s2, int)
    # Probabilisticamente, dois sorteios de 63 bits têm chance ~1/2^63 de
    # coincidir — colisão aqui significa bug determinístico.
    assert s1 != s2, "Duas resoluções consecutivas geraram a mesma semente"
    assert 0 <= s1 < (1 << 63)
    assert 0 <= s2 < (1 << 63)


def test_resolve_rng_seed_with_int_passes_through() -> None:
    """``_resolve_rng_seed(int)`` deve retornar o inteiro inalterado."""
    assert _resolve_rng_seed(42) == 42
    assert _resolve_rng_seed(0) == 0
    assert _resolve_rng_seed(2**31 - 1) == 2**31 - 1


# ──────────────────────────────────────────────────────────────────────────
# 6. ModelGenerationThread.seed_used signal (Qt-aware)
# ──────────────────────────────────────────────────────────────────────────
def test_model_generation_thread_seed_used_signal_present() -> None:
    """``ModelGenerationThread.seed_used`` deve estar definido (v2.19)."""
    from geosteering_ai.simulation.tests.sm_model_gen import (
        ModelGenerationThread,
    )

    assert hasattr(ModelGenerationThread, "seed_used"), (
        "ModelGenerationThread.seed_used (Sprint 19.1) ausente — "
        "GUI não conseguirá logar a semente usada"
    )
