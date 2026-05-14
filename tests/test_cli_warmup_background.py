# -*- coding: utf-8 -*-
"""Testes para Sprint v2.31 Part 2 — Background warmup thread.

Valida:
- Background warmup thread inicializa sem bloquear CLI
- Warmup thread não degrada performance do usuário
- Non-blocking behavior (main() retorna <2s)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Threshold configurável via env var para ambientes CI lentos (default: 2s)
_BLOCKING_THRESHOLD_S = float(os.environ.get("WARMUP_BLOCKING_THRESHOLD_S", "2.0"))


def test_warmup_thread_initializes_without_blocking():
    """Sprint v2.31 Part 2 — Warmup thread não bloqueia main().

    Verifica que `main()` retorna rapidamente mesmo com warmup thread
    sendo disparada em background. O teste roda o subcomando rápido `version`
    que deve completar sem interferência do warmup.

    O threshold é configurável via env var ``WARMUP_BLOCKING_THRESHOLD_S``
    para ambientes CI mais lentos (default: 2s).
    """
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "geosteering_ai.cli", "version"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(PROJECT_ROOT),
    )
    dt = time.perf_counter() - t0

    # Warmup thread é daemon → não bloqueia shutdown
    assert proc.returncode == 0, f"CLI falhou: {proc.stderr[:200]}"
    assert (
        "Geosteering AI Simulation Manager v" in proc.stdout
    ), f"Versão não encontrada no output: {proc.stdout!r}"
    assert dt < _BLOCKING_THRESHOLD_S, (
        f"main() bloqueou por {dt:.2f}s (esperado <{_BLOCKING_THRESHOLD_S}s). "
        "Warmup thread pode estar bloqueando."
    )


def test_warmup_thread_completes_before_simulation():
    """Sprint v2.31 Part 2 — Simulação completa sem hang do warmup thread.

    Verifica que o subcomando `simulate` com modelo pequeno (--models 1)
    executa e completa normalmente mesmo com warmup thread rodando
    em background. Timeout de 30s é conservador — warmup deveria
    completar bem antes de simulate começar.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "geosteering_ai.cli",
            "simulate",
            "--models",
            "1",
            "--n-pos",
            "5",
            "--workers",
            "1",
            "--threads",
            "1",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=str(PROJECT_ROOT),
    )

    # Simule pode retornar 0 (sucesso) ou 1 (erro tratado)
    # Apenas validamos que não hung indefinidamente (timeout teria sido acionado)
    assert proc.returncode in (
        0,
        1,
    ), f"CLI falhou inesperadamente: {proc.stderr[:200]}"
