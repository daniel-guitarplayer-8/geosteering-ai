"""Testes da Sprint v2.23: Fastmath Dual-Mode + Threads Adaptativos.

Valida:
- A.1 Fastmath: campo `use_fastmath` em SimulationConfig + decoradores
  @njit(fastmath=True) aplicados em geometry.py e rotation.py
- A.2 Threads adaptativos: auto-detect via recommend_default_parallelism()
  quando n_workers e threads_per_worker são None
- Backward-compat: defaults explícitos não disparam auto-detect

Contexto: Sprint v2.23 da §22.1+ (pos-Fase 1) do projeto.
"""

import logging

from geosteering_ai.simulation._numba import geometry, rotation
from geosteering_ai.simulation._workers import recommend_default_parallelism
from geosteering_ai.simulation.config import SimulationConfig

# ── A.1 Fastmath ───────────────────────────────────────────────────


def test_geometry_fastmath_decorator_applied():
    """Sprint v2.23 A.1 — find_layers_tr e layer_at_depth têm fastmath=True."""
    for fname in ("find_layers_tr", "layer_at_depth"):
        fn = getattr(geometry, fname)
        opts = getattr(fn, "targetoptions", {})
        assert opts.get("fastmath") is True, (
            f"{fname} deveria ter fastmath=True (Sprint v2.23 A.1). "
            f"targetoptions={opts}"
        )


def test_rotation_fastmath_decorator_applied():
    """Sprint v2.23 A.1 — build_rotation_matrix e rotate_tensor têm fastmath=True."""
    for fname in ("build_rotation_matrix", "rotate_tensor"):
        fn = getattr(rotation, fname)
        opts = getattr(fn, "targetoptions", {})
        assert opts.get("fastmath") is True, (
            f"{fname} deveria ter fastmath=True (Sprint v2.23 A.1). "
            f"targetoptions={opts}"
        )


# ── A.2 Threads adaptativos ────────────────────────────────────────


def test_simulation_config_auto_detect_threads_when_none():
    """Sprint v2.23 A.2 — quando ambos None, auto-detect via recommend_default_parallelism."""
    cfg = SimulationConfig(n_workers=None, threads_per_worker=None)
    rec_workers, rec_threads = recommend_default_parallelism()
    assert (
        cfg.n_workers == rec_workers
    ), f"n_workers esperado {rec_workers}, obtido {cfg.n_workers}"
    assert (
        cfg.threads_per_worker == rec_threads
    ), f"threads_per_worker esperado {rec_threads}, obtido {cfg.threads_per_worker}"


def test_simulation_config_explicit_n_workers_preserved():
    """Sprint v2.23 A.2 — n_workers explícito (não-None) NÃO dispara auto-detect."""
    cfg = SimulationConfig(n_workers=1, threads_per_worker=None)
    assert cfg.n_workers == 1, "n_workers explícito foi sobrescrito pelo auto-detect"


def test_simulation_config_explicit_threads_preserved():
    """Sprint v2.23 A.2 — threads_per_worker explícito NÃO dispara auto-detect.

    Edge case: quando o usuário passa apenas threads_per_worker (não None) e
    deixa n_workers=None, o auto-detect NÃO dispara (requer ambos None).
    Resultado esperado: threads_per_worker preservado E n_workers permanece None
    (downstream em simulate_multi decide single-process ou multi-worker).
    """
    cfg = SimulationConfig(n_workers=None, threads_per_worker=4)
    assert (
        cfg.threads_per_worker == 4
    ), "threads_per_worker explícito foi sobrescrito pelo auto-detect"
    assert cfg.n_workers is None, (
        "n_workers deveria permanecer None (auto-detect só dispara se AMBOS None). "
        f"Valor obtido: {cfg.n_workers}"
    )


def test_threads_adaptive_logged(caplog):
    """Sprint v2.23 A.2 — auto-detect emite mensagem de log de diagnóstico.

    Nota (v2.55): o log do auto-detect foi rebaixado de ``INFO`` → ``DEBUG`` em
    ``config.py`` (redução de ruído "n_workers=16" no caminho JAX). A mensagem
    CONTINUA sendo emitida (diagnóstico preservado), só mudou de nível — por isso
    o ``caplog`` captura em ``DEBUG``. Antes desta correção o teste capturava em
    ``INFO`` e falhava (CI vermelho pré-existente em ``main``).
    """
    with caplog.at_level(logging.DEBUG, logger="geosteering_ai.simulation.config"):
        SimulationConfig(n_workers=None, threads_per_worker=None)
    msgs = [r.message for r in caplog.records]
    assert any(
        "v2.23" in m.lower() or "auto-detect" in m.lower() for m in msgs
    ), f"Mensagem de log v2.23 ausente. Logs capturados: {msgs}"
