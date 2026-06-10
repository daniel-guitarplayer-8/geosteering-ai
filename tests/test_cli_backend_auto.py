# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/test_cli_backend_auto.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Spec        : 0003-cli-backend-auto                                      ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI — seleção de backend (`--backend auto`)                ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-05                                                 ║
# ║  Status      : Produção                                                   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Cobre os critérios de aceite da spec 0003: `--backend auto`, o         ║
# ║    DeprecationWarning do default implícito, a resolução TLS-safe da       ║
# ║    árvore do dispatcher e o reporting do backend efetivo (nunca "auto").  ║
# ║                                                                           ║
# ║  MAPA AC → TESTE                                                          ║
# ║    AC-1.1 simulate --backend auto exit 0 .... test_cli_simulate_auto_*    ║
# ║    AC-1.2 benchmark --backend auto exit 0 ... test_cli_benchmark_auto_*   ║
# ║    AC-1.3 --backend inválido exit 2 ......... test_cli_backend_invalid_*  ║
# ║    AC-2.1 sem GPU → numba ................... test_auto_no_gpu_*          ║
# ║    AC-2.2 consistente c/ dispatcher ........ test_auto_consistent_*       ║
# ║    AC-3.1 default implícito → warning ....... test_default_implicit_*     ║
# ║    AC-3.2 escolha explícita → sem warning ... test_explicit_*            ║
# ║    AC-4.1 JSON reporta backend concreto ..... test_cli_auto_json_*        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Testes da spec 0003 — CLI ``--backend auto`` (seleção automática numba/jax).

Mistura testes UNITÁRIOS rápidos (resolução de backend, deprecação) e testes
E2E por subprocess (exit codes, JSON) marcados ``slow``. A resolução ``auto`` é
verificada como CONSISTENTE com ``dispatch._resolve_backend`` (mesma decisão) e
TLS-SAFE (a GPU nunca é sondada para batches pequenos ou geometria não-agrupável).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def _run_cli(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    """Executa ``python -m geosteering_ai.cli <args>`` e captura stdout/stderr."""
    return subprocess.run(
        [sys.executable, "-m", "geosteering_ai.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


def _models(n: int, *, n_esp: int = 3, distinct: bool = False) -> list[dict]:
    """Constrói ``n`` modelos ``{rho_h, rho_v, esp}`` para o pré-voo de backend.

    ``distinct=False`` → todas as geometrias IGUAIS (1 grupo → agrupável).
    ``distinct=True``  → cada modelo com ``esp`` ÚNICA (n grupos → não-agrupável).
    """
    out: list[dict] = []
    n_layers = n_esp + 2
    for i in range(n):
        if distinct:
            esp = (np.arange(n_esp, dtype=np.float64) + 2.0) + float(i)
        else:
            esp = np.full(n_esp, 5.0, dtype=np.float64)
        out.append(
            {
                "rho_h": np.ones(n_layers, dtype=np.float64),
                "rho_v": np.full(n_layers, 2.0, dtype=np.float64),
                "esp": esp,
            }
        )
    return out


def _extract_json(stdout: str) -> Optional[dict]:
    """Extrai o primeiro objeto JSON de ``stdout`` (logs vão p/ stderr)."""
    start, end = stdout.find("{"), stdout.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return json.loads(stdout[start : end + 1])


def _collect_backend_values(obj) -> list[str]:
    """Coleta recursivamente todos os valores de chaves ``"backend"`` no JSON."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "backend" and isinstance(v, str):
                found.append(v)
            else:
                found.extend(_collect_backend_values(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_collect_backend_values(v))
    return found


# ════════════════════════════════════════════════════════════════════════════
# RF-3 — resolve_requested_backend + DeprecationWarning (AC-3.1, AC-3.2)
# ════════════════════════════════════════════════════════════════════════════
def test_default_implicit_emits_deprecation_warning():
    """AC-3.1 — sem --backend (None) → DeprecationWarning citando 'auto', e numba."""
    from geosteering_ai.cli._exec import resolve_requested_backend

    args = argparse.Namespace(backend=None)
    with pytest.warns(DeprecationWarning) as record:
        result = resolve_requested_backend(args)
    assert result == "numba"
    # Mensagem deve ser acionável: cita a mudança (auto), a versão-alvo e o atual.
    msg = str(record[0].message)
    assert "auto" in msg and "v2.57.0" in msg and "numba" in msg


def test_explicit_numba_no_warning():
    """AC-3.2 — --backend numba (explícito) NÃO emite DeprecationWarning."""
    from geosteering_ai.cli._exec import resolve_requested_backend

    args = argparse.Namespace(backend="numba")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # qualquer warning → erro
        assert resolve_requested_backend(args) == "numba"


def test_explicit_auto_passthrough_no_warning():
    """AC-3.2 (extensão) — --backend auto explícito passa adiante, sem warning."""
    from geosteering_ai.cli._exec import resolve_requested_backend

    args = argparse.Namespace(backend="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_requested_backend(args) == "auto"


# ════════════════════════════════════════════════════════════════════════════
# RF-2 — resolução TLS-safe da árvore auto (AC-2.1) + propriedade anti-crash-TLS
# ════════════════════════════════════════════════════════════════════════════
def test_auto_no_gpu_resolves_numba(monkeypatch):
    """AC-2.1 — sem GPU, auto (n≥32, agrupável) resolve numba/cpu (nunca 'auto')."""
    monkeypatch.setattr(
        "geosteering_ai.simulation.dispatch._jax_gpu_available", lambda: False
    )
    from geosteering_ai.cli._exec import resolve_backend_preflight

    backend, device, reason = resolve_backend_preflight("auto", _models(64), quiet=True)
    assert backend == "numba"
    assert device == "cpu"
    assert reason and "auto" in reason


def test_auto_small_batch_resolves_numba_without_probing_gpu(monkeypatch):
    """TLS-safe — batch pequeno (n<32) resolve numba SEM jamais sondar a GPU.

    Garante que o disqualificador jax-free roda PRIMEIRO: se a GPU fosse sondada,
    importaria/inicializaria o JAX e arriscaria o crash de TLS (v2.55).
    """
    probed = {"gpu": False}

    def _spy_probe() -> bool:
        probed["gpu"] = True
        return True

    monkeypatch.setattr(
        "geosteering_ai.simulation.dispatch._jax_gpu_available", _spy_probe
    )
    from geosteering_ai.cli._exec import resolve_backend_preflight

    backend, _, _ = resolve_backend_preflight("auto", _models(4), quiet=True)
    assert backend == "numba"
    assert probed["gpu"] is False, "GPU foi sondada para batch pequeno (risco TLS)"


def test_auto_non_groupable_resolves_numba_without_probing_gpu(monkeypatch):
    """TLS-safe — geometria não-agrupável resolve numba SEM sondar a GPU."""
    probed = {"gpu": False}

    def _spy_probe() -> bool:
        probed["gpu"] = True
        return True

    monkeypatch.setattr(
        "geosteering_ai.simulation.dispatch._jax_gpu_available", _spy_probe
    )
    from geosteering_ai.cli._exec import resolve_backend_preflight

    backend, _, reason = resolve_backend_preflight(
        "auto", _models(64, distinct=True), quiet=True
    )
    assert backend == "numba"
    assert probed["gpu"] is False
    assert reason and "agrup" in reason.lower()


def test_auto_gpu_groupable_large_resolves_jax(monkeypatch):
    """auto com GPU + n≥32 + agrupável → jax/gpu."""
    monkeypatch.setattr(
        "geosteering_ai.simulation.dispatch._jax_gpu_available", lambda: True
    )
    from geosteering_ai.cli._exec import resolve_backend_preflight

    backend, device, reason = resolve_backend_preflight("auto", _models(64), quiet=True)
    assert backend == "jax"
    assert device == "gpu"
    assert reason and "JAX" in reason


def test_explicit_numba_preflight_unchanged():
    """Regressão — requested='numba' continua resolvendo numba/cpu (sem tocar jax)."""
    from geosteering_ai.cli._exec import resolve_backend_preflight

    backend, device, reason = resolve_backend_preflight(
        "numba", _models(64), quiet=True
    )
    assert backend == "numba"
    assert device == "cpu"
    assert reason is None


# ════════════════════════════════════════════════════════════════════════════
# AC-2.2 — consistência com dispatch._resolve_backend (mesma decisão)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "n,gpu,distinct",
    [
        (64, True, False),  # GPU + grande + agrupável → jax
        (64, False, False),  # sem GPU → numba
        (4, True, False),  # pequeno → numba
        (64, True, True),  # não-agrupável → numba
    ],
)
def test_auto_consistent_with_dispatcher(monkeypatch, n, gpu, distinct):
    """A decisão da CLI (``auto``) é IDÊNTICA à de ``dispatch._resolve_backend``."""
    monkeypatch.setattr(
        "geosteering_ai.simulation.dispatch._jax_gpu_available", lambda: gpu
    )
    try:
        from geosteering_ai.simulation.dispatch import (
            _N_MODELS_GPU_THRESHOLD,
            _resolve_backend,
        )
    except Exception:  # pragma: no cover — dispatch indisponível
        pytest.skip("geosteering_ai.simulation.dispatch indisponível")

    from geosteering_ai.cli._exec import models_to_batch, resolve_backend_preflight

    models = _models(n, distinct=distinct)
    cli_backend, _, _ = resolve_backend_preflight("auto", models, quiet=True)

    _rho_h, _rho_v, esp_batch = models_to_batch(models)
    try:
        disp_backend, _, _ = _resolve_backend(
            "auto",
            n,
            esp_batch,
            numba_fallback=True,
            n_models_gpu_threshold=_N_MODELS_GPU_THRESHOLD,
        )
    except ImportError:  # pragma: no cover — _jax.multi_forward indisponível
        pytest.skip("jax indisponível para dispatch._resolve_backend")

    assert cli_backend == disp_backend


def test_auto_constants_mirror_dispatcher():
    """Guard de DRIFT — as constantes LOCAIS espelham as do dispatcher.

    O ramo ``auto`` usa cópias locais (``_AUTO_*``) para manter o caminho Numba
    jax-free (importar ``dispatch`` carregaria o jax). Este teste garante que os
    espelhos NÃO divergem da fonte — se o dispatcher mudar a árvore, este teste
    falha e força a atualização do espelho.
    """
    try:
        from geosteering_ai.simulation.dispatch import (
            _GROUPABLE_RATIO_MAX,
            _N_MODELS_GPU_THRESHOLD,
        )
    except Exception:  # pragma: no cover — dispatch indisponível
        pytest.skip("geosteering_ai.simulation.dispatch indisponível")

    from geosteering_ai.cli import _exec

    assert _exec._AUTO_GROUPABLE_RATIO_MAX == _GROUPABLE_RATIO_MAX
    assert _exec._AUTO_N_MODELS_GPU_THRESHOLD == _N_MODELS_GPU_THRESHOLD


def _models_k_groups(n: int, k: int, *, n_esp: int = 3) -> list[dict]:
    """``n`` modelos com EXATAMENTE ``k`` geometrias distintas (round-robin)."""
    out: list[dict] = []
    n_layers = n_esp + 2
    for i in range(n):
        esp = np.full(n_esp, 5.0 + float(i % k), dtype=np.float64)
        out.append(
            {
                "rho_h": np.ones(n_layers, dtype=np.float64),
                "rho_v": np.full(n_layers, 2.0, dtype=np.float64),
                "esp": esp,
            }
        )
    return out


@pytest.mark.parametrize(
    "n,k,gpu",
    [
        (32, 1, True),  # n == limiar exato (32 não é < 32) + 1 grupo → jax
        (31, 1, True),  # n == limiar-1 (< 32) → numba (fronteira do limiar)
        (32, 16, True),  # k == 0.5*n exato (16 ≤ 16) → agrupável → jax
        (32, 17, True),  # k > 0.5*n (17 > 16) → não-agrupável → numba
    ],
)
def test_auto_boundary_consistent_with_dispatcher(monkeypatch, n, k, gpu):
    """Fronteiras (n==32, k==0.5n) — decisão da CLI == dispatcher (off-by-one guard)."""
    monkeypatch.setattr(
        "geosteering_ai.simulation.dispatch._jax_gpu_available", lambda: gpu
    )
    try:
        from geosteering_ai.simulation.dispatch import (
            _N_MODELS_GPU_THRESHOLD,
            _resolve_backend,
        )
    except Exception:  # pragma: no cover
        pytest.skip("geosteering_ai.simulation.dispatch indisponível")

    from geosteering_ai.cli._exec import models_to_batch, resolve_backend_preflight

    models = _models_k_groups(n, k)
    cli_backend, _, _ = resolve_backend_preflight("auto", models, quiet=True)

    _rho_h, _rho_v, esp_batch = models_to_batch(models)
    try:
        disp_backend, _, _ = _resolve_backend(
            "auto",
            n,
            esp_batch,
            numba_fallback=True,
            n_models_gpu_threshold=_N_MODELS_GPU_THRESHOLD,
        )
    except ImportError:  # pragma: no cover
        pytest.skip("jax indisponível para dispatch._resolve_backend")

    assert cli_backend == disp_backend


# ════════════════════════════════════════════════════════════════════════════
# RF-1 / RF-4 — E2E por subprocess (AC-1.1, AC-1.2, AC-1.3, AC-4.1)
# ════════════════════════════════════════════════════════════════════════════
def test_cli_backend_invalid_exit2():
    """AC-1.3 — --backend inválido → exit 2 (argparse rejeita via choices)."""
    proc = _run_cli(["simulate", "--models", "2", "--backend", "invalido"], timeout=60)
    assert proc.returncode == 2


@pytest.mark.slow
def test_cli_simulate_auto_exit0():
    """AC-1.1 — simulate --backend auto (CPU → numba) retorna exit 0."""
    proc = _run_cli(
        ["simulate", "--models", "4", "--n-pos", "20", "--backend", "auto", "--quiet"]
    )
    assert proc.returncode == 0, proc.stderr[-800:]


@pytest.mark.slow
def test_cli_benchmark_auto_exit0():
    """AC-1.2 — benchmark --scenario A --backend auto retorna exit 0."""
    proc = _run_cli(
        ["benchmark", "--scenario", "A", "--n", "8", "--backend", "auto", "--quiet"]
    )
    assert proc.returncode == 0, proc.stderr[-800:]


@pytest.mark.slow
def test_cli_auto_json_reports_concrete_backend():
    """AC-4.1 — simulate --backend auto --json reporta backend ∈ {numba,jax}, não 'auto'."""
    proc = _run_cli(
        ["simulate", "--models", "4", "--n-pos", "20", "--backend", "auto", "--json"]
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    payload = _extract_json(proc.stdout)
    assert payload is not None, f"JSON ausente no stdout: {proc.stdout[:400]!r}"
    backends = _collect_backend_values(payload)
    assert backends, f"campo 'backend' ausente no JSON: {payload}"
    assert all(b in {"numba", "jax"} for b in backends), backends
    assert "auto" not in backends
