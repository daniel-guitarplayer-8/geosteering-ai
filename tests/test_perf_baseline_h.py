# -*- coding: utf-8 -*-
"""Sprint v2.36 D2 — Guard tests para baseline Cenário H.

Valida:
- Entrada `H_n2_stress` existe em `.claude/perf_baseline.json` com
  `throughput_mod_h > 0` (não-null).
- Cenário H em `cli/benchmark.py` respeita range de dips [0, 90]°
  (paridade Fortran, multi_forward.py linha 254).
- Combinatória 8×8×8 = 512 preservada.

Contexto: a v2.35 introduziu Cenário H mas (1) deixou baseline ausente
e (2) tinha dip 105° fora do range válido, disparando ValueError no
worker. v2.36 D2 corrige ambos.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_FILE = PROJECT_ROOT / ".claude" / "perf_baseline.json"


def test_baseline_h_n2_stress_present():
    """v2.36 D2 — H_n2_stress baseline existe e tem throughput > 0."""
    data = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    assert "scenarios" in data
    assert (
        "H_n2_stress" in data["scenarios"]
    ), "Sprint v2.36 D2 — entrada H_n2_stress ausente em perf_baseline.json"
    entry = data["scenarios"]["H_n2_stress"]
    assert (
        entry.get("throughput_mod_h") is not None
    ), "H_n2_stress.throughput_mod_h não pode ser null (Sprint v2.36 D2)"
    assert isinstance(entry["throughput_mod_h"], (int, float))
    assert entry["throughput_mod_h"] > 0
    assert entry.get("type") == "warm"


def test_scenario_h_dips_within_fortran_range():
    """v2.36 D2 — Cenário H dips ∈ [0, 90]° (paridade Fortran)."""
    from geosteering_ai.cli.benchmark import SCENARIOS

    assert "H" in SCENARIOS
    dips = SCENARIOS["H"]["dips"]
    for d in dips:
        assert 0.0 <= d <= 90.0, (
            f"Cenário H dip={d}° fora do range [0, 90]° "
            f"(paridade Fortran — multi_forward.py linha 254)"
        )


def test_scenario_h_dimensions_preserved():
    """v2.36 D2 — Cenário H mantém 8×8×8 = 512 combos."""
    from geosteering_ai.cli.benchmark import SCENARIOS

    sc = SCENARIOS["H"]
    assert len(sc["freqs"]) == 8
    assert len(sc["trs"]) == 8
    assert len(sc["dips"]) == 8
    assert len(sc["freqs"]) * len(sc["trs"]) * len(sc["dips"]) == 512
