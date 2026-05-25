# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_simulation_jax_perf_baseline.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Sprint O0 (T1.6) — gate automático throughput JAX GPU      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-24 (Sprint O0 do plano de otimização JAX GPU)      ║
# ║  Status      : Produção (gate Tier 1 anti-regressão de performance)       ║
# ║  Framework   : pytest + JAX + .claude/perf_baseline.json                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Teste T1.6 — gate automático de throughput JAX GPU vs baseline da GPU em uso.

**Motivação**: sem um gate automático de performance, qualquer otimização que
acidentalmente DEGRADE throughput pode passar por CI sem alarme. Baselines
oficiais v2.43 estão registradas em ``.claude/perf_baseline.json`` em duas
seções: ``jax_gpu_t4`` e ``jax_gpu_a100``. O gate detecta a GPU em uso e
compara contra a seção apropriada (evita falso positivo de regressão quando
T4 roda contra threshold A100, que é ~1.32× maior em A).

**Cenários cobertos** (subset do gate oficial A/B/E):

  - **A**: n_models=50, n_pos=1, nf=1, nTR=1, nAng=1
  - **B**: n_models=50, n_pos=100, nf=1, nTR=1, nAng=1
  - **E**: n_models=50, n_pos=600, nf=1, nTR=1, nAng=1

Thresholds variam por hardware — consulte ``perf_baseline.json``.

**Metodologia**:
  - 1 warmup (descartado)
  - 3 runs hot, mediana usada para comparação
  - SKIP gracioso se rodando em CPU
  - SKIP gracioso se baseline JSON ou seção do hardware ausente

**Markers**: ``@pytest.mark.gpu`` + ``@pytest.mark.slow`` (não roda em CI
rápido — apenas em pre-release Colab T4/A100).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Skip global se JAX não instalado (Sprint v2.40 D9) + marker gpu+slow
from geosteering_ai.simulation._jax import HAS_JAX

pytestmark = [
    pytest.mark.skipif(
        not HAS_JAX,
        reason="JAX não instalado — T1.6 requer `pip install jax[cuda12]`",
    ),
    pytest.mark.gpu,
    pytest.mark.slow,
]

if HAS_JAX:
    import jax

    jax.config.update("jax_enable_x64", True)


# ══════════════════════════════════════════════════════════════════════════════
# Localização da baseline
# ══════════════════════════════════════════════════════════════════════════════

_BASELINE_PATH = Path(__file__).resolve().parents[1] / ".claude" / "perf_baseline.json"
"""Caminho absoluto para ``.claude/perf_baseline.json`` na raiz do projeto."""


def _detect_gpu_baseline_key() -> str:
    """Detecta GPU em uso e retorna a chave da baseline correspondente.

    Mapeia ``device.device_kind`` para a seção apropriada em
    ``perf_baseline.json``. Sem este mapeamento, o gate compararia T4
    contra threshold A100 (1.32× maior em cenário A), gerando falsos
    positivos de regressão.

    Returns:
        ``"jax_gpu_a100"`` se A100 detectada, ``"jax_gpu_t4"`` se T4,
        ``"jax_gpu_a100"`` como fallback conservador (oficial canônica).
    """
    if not HAS_JAX:
        return "jax_gpu_a100"
    try:
        device = jax.devices()[0]
        kind = (getattr(device, "device_kind", "") or "").lower()
        # Fallback: alguns backends expõem só via str(device)
        if not kind:
            kind = str(device).lower()
        if "a100" in kind:
            return "jax_gpu_a100"
        if "t4" in kind:
            return "jax_gpu_t4"
    except Exception:
        pass
    return "jax_gpu_a100"


def _load_baseline(section: str | None = None) -> dict[str, Any] | None:
    """Carrega seção da baseline GPU do ``perf_baseline.json``.

    Args:
        section: Chave da seção (``"jax_gpu_t4"`` ou ``"jax_gpu_a100"``).
            Se ``None``, detecta hardware via :func:`_detect_gpu_baseline_key`.

    Returns:
        Dict com cenários ``{A,B,E}_hot`` + thresholds, ou ``None`` se
        arquivo não existir ou seção ausente.
    """
    if not _BASELINE_PATH.exists():
        return None
    try:
        with _BASELINE_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if section is None:
        section = _detect_gpu_baseline_key()
    return data.get(section)


def _is_running_on_gpu() -> bool:
    """Detecta se JAX está rodando em GPU CUDA (não CPU).

    Returns:
        ``True`` se ``jax.default_backend() in ("gpu", "cuda")``,
        ``False`` caso contrário.
    """
    if not HAS_JAX:
        return False
    try:
        backend = jax.default_backend()
        return backend in ("gpu", "cuda") or "cuda" in str(backend).lower()
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Helper de medição
# ══════════════════════════════════════════════════════════════════════════════


def _measure_throughput_mod_h(
    n_models: int,
    n_pos: int,
    n_freqs: int = 1,
    n_tr: int = 1,
    n_ang: int = 1,
    n_runs_hot: int = 3,
) -> float:
    """Mede throughput em mod/h via ``simulate_multi_jax_batched``.

    Args:
        n_models: Número de modelos batch.
        n_pos: Número de posições.
        n_freqs, n_tr, n_ang: Dimensões adicionais.
        n_runs_hot: Quantidade de runs hot (após warmup).

    Returns:
        Mediana de throughput em modelos/hora.
    """
    import gc

    from geosteering_ai.simulation._jax.forward_pure import (
        clear_jit_cache,
        clear_unified_jit_cache,
    )
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax_batched

    # Mitigação OOM T4/A100 (Sprint O1 unroll regression): após 170+ testes
    # GPU, os caches project-level retêm dezenas de programas XLA compilados,
    # cada um pinando buffers persistentes na VRAM. ``jax.clear_caches()``
    # limpa SOMENTE o cache interno do JAX — os caches do projeto
    # (``_BUCKET_JIT_CACHE``, ``_UNIFIED_JIT_CACHE``,
    # ``_UNIFIED_CHUNKED_JIT_CACHE``) precisam ser limpos explicitamente.
    # Cenário E aloca ~5.6 GiB num único op; sem cleanup completo falha em
    # T4 15 GB (3 GB livres após estado acumulado) e até em A100 40 GB.
    clear_jit_cache()
    clear_unified_jit_cache()
    jax.clear_caches()
    gc.collect()

    # ── Modelo sintético reproduzível (oklahoma_3-like) ───────────────────────
    rng = np.random.default_rng(42)
    rho_h_batch = rng.uniform(1.0, 100.0, size=(n_models, 3)).astype(np.float64)
    rho_v_batch = rho_h_batch.copy()  # isotrópico para simplicidade
    esp_batch = np.full((n_models, 1), 5.0, dtype=np.float64)
    positions_z = np.linspace(-5.0, 5.0, n_pos).astype(np.float64)
    freqs = np.linspace(1e4, 1e5, n_freqs).astype(np.float64).tolist()
    trs = [1.0 * (i + 1) for i in range(n_tr)]
    dips = [0.0 + 10.0 * i for i in range(n_ang)]

    kwargs = dict(
        frequencies_hz=freqs,
        tr_spacings_m=trs,
        dip_degs=dips,
    )

    # ── Warmup (descartado) ───────────────────────────────────────────────────
    res = simulate_multi_jax_batched(
        rho_h_batch, rho_v_batch, esp_batch, positions_z, **kwargs
    )
    _ = res.H_tensor.shape  # força sync

    # ── Runs hot (mediana) ────────────────────────────────────────────────────
    throughputs: list[float] = []
    for _ in range(n_runs_hot):
        t0 = time.perf_counter()
        res = simulate_multi_jax_batched(
            rho_h_batch, rho_v_batch, esp_batch, positions_z, **kwargs
        )
        _ = res.H_tensor.shape
        elapsed = time.perf_counter() - t0
        throughputs.append(n_models / elapsed * 3600.0)

    return statistics.median(throughputs)


# ══════════════════════════════════════════════════════════════════════════════
# T1.6 — Gate de regressão em A, B, E
# ══════════════════════════════════════════════════════════════════════════════

_SCENARIOS_GATE = {
    "A": {"n_models": 50, "n_pos": 1, "n_freqs": 1, "n_tr": 1, "n_ang": 1},
    "B": {"n_models": 50, "n_pos": 100, "n_freqs": 1, "n_tr": 1, "n_ang": 1},
    "E": {"n_models": 50, "n_pos": 600, "n_freqs": 1, "n_tr": 1, "n_ang": 1},
}


@pytest.mark.parametrize("scenario", ["A", "B", "E"])
def test_throughput_gpu_regression_gate(scenario: str) -> None:
    """Gate de regressão: throughput atual ≥ 90% da baseline da GPU em uso.

    Detecta T4 vs A100 e compara contra a seção apropriada de
    ``perf_baseline.json``. Falha se otimização degradar performance
    abaixo do limiar estabelecido para o hardware atual.

    Args:
        scenario: Identificador do cenário gate (``A``, ``B``, ``E``).
    """
    # ── Skips graciosos ───────────────────────────────────────────────────────
    if not _is_running_on_gpu():
        pytest.skip("T1.6 requer GPU CUDA — rodando em CPU, sem sentido medir")

    baseline_key = _detect_gpu_baseline_key()
    baseline = _load_baseline(baseline_key)
    if baseline is None:
        pytest.skip(
            f"Baseline {baseline_key} ausente em .claude/perf_baseline.json — "
            f"rode `validate_jax_gpu_v240.ipynb` no hardware correspondente primeiro"
        )

    scen_key = f"{scenario}_hot"
    if scen_key not in baseline:
        pytest.skip(f"Cenário {scen_key} ausente em baseline {baseline_key}")

    # ── Threshold oficial (90% do baseline) ───────────────────────────────────
    threshold = baseline[scen_key].get("threshold_90pct")
    if threshold is None:
        pytest.skip(f"threshold_90pct ausente em {baseline_key}[{scen_key}]")

    # ── Skip arquitetural: T4 + cenário E ─────────────────────────────────────
    # Cenário E aloca ~5.6 GiB num único op (50 modelos × 600 pos × n_freq×TR).
    # T4 tem 15 GB total; após 170+ testes prévios, ~12 GB ficam retidos
    # FORA do pool JAX (cuDNN handles, fragmentação do contexto CUDA). O
    # ``cuda_async`` allocator e ``memory_stats()`` enxergam apenas o pool
    # JAX (~884 MB), não os 12 GB externos — qualquer pre-flight check
    # baseado em ``memory_stats()`` falha em detectar a real escassez.
    #
    # Skip incondicional em T4 é a única solução robusta: gate continua
    # válido em A100 (40 GB folga) e em sessão isolada T4 sem estado
    # acumulado (pytest com --forked ou execução standalone).
    if baseline_key == "jax_gpu_t4" and scenario == "E":
        pytest.skip(
            "Cenário E em T4: limite arquitetural — 5.6 GiB allocation + "
            "~12 GiB de estado CUDA acumulado de testes prévios > 15 GiB VRAM "
            "T4. memory_stats() não enxerga memória externa ao pool JAX. "
            "Para gate E em T4, rode em sessão isolada "
            "(`pytest --forked tests/test_simulation_jax_perf_baseline.py -k E`)"
            " ou use A100 (40 GiB)."
        )

    # ── Medição atual ─────────────────────────────────────────────────────────
    cfg = _SCENARIOS_GATE[scenario]
    measured = _measure_throughput_mod_h(**cfg)

    # ── Gate ──────────────────────────────────────────────────────────────────
    assert measured >= threshold, (
        f"T1.6 REGRESSÃO de throughput em cenário {scenario} "
        f"(baseline={baseline_key}): "
        f"medido={measured:,.0f} mod/h < threshold={threshold:,.0f} mod/h "
        f"(baseline={baseline[scen_key].get('throughput_mod_h'):,.0f} mod/h, "
        f"90% gate)"
    )


def test_baseline_file_structure_is_valid() -> None:
    """Sanity check: baseline JSON existe E tem estrutura mínima esperada.

    Valida ambas as seções (T4 e A100) — falha indica corrupção do arquivo
    de baseline (não regressão de performance). Roda sempre, mesmo em CPU.
    """
    if not _BASELINE_PATH.exists():
        pytest.skip(f".claude/perf_baseline.json ausente em {_BASELINE_PATH}")

    with _BASELINE_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    for section in ("jax_gpu_t4", "jax_gpu_a100"):
        assert section in data, f"Seção {section!r} faltando na baseline"
        block = data[section]
        assert "_meta" in block, f"Subsection '_meta' faltando em {section}"
        for scen in ("A_hot", "B_hot", "E_hot"):
            assert scen in block, f"Cenário {scen} faltando em {section}"
            assert (
                "threshold_90pct" in block[scen]
            ), f"threshold_90pct faltando em {section}[{scen}]"
            assert (
                "throughput_mod_h" in block[scen]
            ), f"throughput_mod_h faltando em {section}[{scen}]"
