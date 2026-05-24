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
"""Teste T1.6 — gate automático de throughput JAX GPU vs baseline A100.

**Motivação**: sem um gate automático de performance, qualquer otimização que
acidentalmente DEGRADE throughput pode passar por CI sem alarme. A baseline
A100 oficial v2.43 está registrada em ``.claude/perf_baseline.json`` (seção
``jax_gpu_a100``). Este teste mede throughput atual via
``simulate_multi_jax_batched`` e falha se cair abaixo do ``threshold_90pct``.

**Cenários cobertos** (subset do gate oficial A/B/E):

  - **A**: n_models=50, n_pos=1, nf=1, nTR=1, nAng=1 — threshold 8.215.977 mod/h
  - **B**: n_models=50, n_pos=100, nf=1, nTR=1, nAng=1 — threshold 256.323 mod/h
  - **E**: n_models=50, n_pos=600, nf=1, nTR=1, nAng=1 — threshold 43.031 mod/h

**Metodologia**:
  - 1 warmup (descartado)
  - 3 runs hot, mediana usada para comparação
  - SKIP gracioso se rodando em CPU (não faz sentido medir A100 em CPU)
  - SKIP gracioso se baseline JSON não encontrado

**Markers**: ``@pytest.mark.gpu`` + ``@pytest.mark.slow`` (não roda em CI
rápido — apenas em pre-release Colab A100).
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


def _load_baseline() -> dict[str, Any] | None:
    """Carrega seção ``jax_gpu_a100`` do ``perf_baseline.json``.

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
    return data.get("jax_gpu_a100")


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
    from geosteering_ai.simulation._jax.multi_forward import simulate_multi_jax_batched

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
    """Gate de regressão: throughput atual ≥ 90% do baseline A100 (v2.43).

    Falha se otimização degradar performance abaixo do limiar estabelecido.

    Args:
        scenario: Identificador do cenário gate (``A``, ``B``, ``E``).
    """
    # ── Skips graciosos ───────────────────────────────────────────────────────
    if not _is_running_on_gpu():
        pytest.skip("T1.6 requer GPU CUDA — rodando em CPU, sem sentido medir")

    baseline = _load_baseline()
    if baseline is None:
        pytest.skip(
            "Baseline jax_gpu_a100 ausente em .claude/perf_baseline.json — "
            "rode `validate_jax_gpu_v240.ipynb` em A100 primeiro"
        )

    scen_key = f"{scenario}_hot"
    if scen_key not in baseline:
        pytest.skip(f"Cenário {scen_key} ausente na baseline")

    # ── Threshold oficial (90% do baseline) ───────────────────────────────────
    threshold = baseline[scen_key].get("threshold_90pct")
    if threshold is None:
        pytest.skip(f"threshold_90pct ausente em baseline[{scen_key}]")

    # ── Medição atual ─────────────────────────────────────────────────────────
    cfg = _SCENARIOS_GATE[scenario]
    measured = _measure_throughput_mod_h(**cfg)

    # ── Gate ──────────────────────────────────────────────────────────────────
    assert measured >= threshold, (
        f"T1.6 REGRESSÃO de throughput em cenário {scenario}: "
        f"medido={measured:,.0f} mod/h < threshold={threshold:,.0f} mod/h "
        f"(baseline={baseline[scen_key].get('throughput_mod_h'):,.0f} mod/h, "
        f"90% gate)"
    )


def test_baseline_file_structure_is_valid() -> None:
    """Sanity check: baseline JSON existe E tem estrutura mínima esperada.

    Falha indica corrupção do arquivo de baseline (não regressão de performance).
    Roda sempre, mesmo em CPU.
    """
    if not _BASELINE_PATH.exists():
        pytest.skip(f".claude/perf_baseline.json ausente em {_BASELINE_PATH}")

    with _BASELINE_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    assert "jax_gpu_a100" in data, "Seção 'jax_gpu_a100' faltando na baseline"
    a100 = data["jax_gpu_a100"]
    assert "_meta" in a100, "Subsection '_meta' faltando em jax_gpu_a100"
    for scen in ("A_hot", "B_hot", "E_hot"):
        assert scen in a100, f"Cenário {scen} faltando em jax_gpu_a100"
        assert "threshold_90pct" in a100[scen], f"threshold_90pct faltando em {scen}"
        assert "throughput_mod_h" in a100[scen], f"throughput_mod_h faltando em {scen}"
