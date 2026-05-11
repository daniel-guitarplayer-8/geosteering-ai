#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sprint v2.24 — Script de contagem automática de testes pytest.

Executa a suite de testes completa e extrai a contagem de PASS/SKIP/FAIL
no formato canônico do projeto Geosteering AI:

    "1624 PASS / 295 SKIP / 0 FAIL"

Esse formato é o usado em CLAUDE.md (linha 16), CHANGELOG.md, ROADMAP.md
e nos relatórios de sprint em docs/reports/.

Origem do débito (M2 da revisão final v2.23):
    Antes desta automação, a contagem era atualizada manualmente após cada
    sprint, levando a divergências (ex: CLAUDE.md "1604+" vs suite real
    "1624"). Este script elimina o débito.

Modos de operação:
    - Default: roda `pytest tests/ -q --tb=no` e imprime a linha formatada
    - --json: emite JSON estruturado (para automação CI/CD)
    - --from-file PATH: lê output do pytest do arquivo (em vez de executar)

Exemplo de uso:
    $ python scripts/count_pytest_pass.py
    1624 PASS / 295 SKIP / 0 FAIL

    $ python scripts/count_pytest_pass.py --json
    {"passed": 1624, "skipped": 295, "failed": 0, "xfailed": 0, "errors": 0}
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict


# ── Regex para extrair contagens da última linha do pytest -q ─────────
# Exemplos de linhas que pytest emite no fim:
#   "1624 passed, 295 skipped in 911.45s"
#   "1624 passed, 295 skipped, 1 failed in 911.45s"
#   "1 failed, 1623 passed, 295 skipped, 2 xfailed in 911s"
_RX_PASSED = re.compile(r"(\d+)\s+passed")
_RX_SKIPPED = re.compile(r"(\d+)\s+skipped")
_RX_FAILED = re.compile(r"(\d+)\s+failed")
_RX_XFAILED = re.compile(r"(\d+)\s+xfailed")
_RX_ERRORS = re.compile(r"(\d+)\s+error")


def parse_pytest_summary(text: str) -> Dict[str, int]:
    """Extrai contagens da saída do pytest.

    Aceita output de `pytest -q --tb=no` ou `pytest -v` — busca padrões
    `\\d+ passed`, `\\d+ skipped`, `\\d+ failed`, `\\d+ xfailed`,
    `\\d+ error` em qualquer parte do texto. Quando o padrão não aparece,
    retorna 0.

    Args:
        text: Saída completa do pytest (stdout + stderr).

    Returns:
        Dict com chaves passed, skipped, failed, xfailed, errors.
    """
    def find(rx: re.Pattern[str]) -> int:
        m = rx.search(text)
        return int(m.group(1)) if m else 0

    return {
        "passed": find(_RX_PASSED),
        "skipped": find(_RX_SKIPPED),
        "failed": find(_RX_FAILED),
        "xfailed": find(_RX_XFAILED),
        "errors": find(_RX_ERRORS),
    }


def run_pytest(test_dir: str = "tests/", timeout_s: int = 1800) -> str:
    """Executa pytest e retorna stdout+stderr.

    Args:
        test_dir: Diretório dos testes. Default `tests/`.
        timeout_s: Timeout total em segundos. Default 30 min.

    Returns:
        Output completo do pytest.

    Raises:
        subprocess.TimeoutExpired: se pytest exceder timeout_s.
    """
    cmd = [sys.executable, "-m", "pytest", test_dir, "-q", "--tb=no"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return proc.stdout + proc.stderr


def format_canonical(counts: Dict[str, int]) -> str:
    """Formata contagens no padrão canônico do projeto.

    Padrão: ``"X PASS / Y SKIP / Z FAIL"`` (em maiúsculas, separador
    com espaços ao redor da barra). Esse formato é o usado em CLAUDE.md,
    CHANGELOG.md, ROADMAP.md e relatórios.
    """
    return (
        f"{counts['passed']} PASS / "
        f"{counts['skipped']} SKIP / "
        f"{counts['failed']} FAIL"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sprint v2.24 — contagem automática de testes pytest.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emite JSON estruturado em vez do formato canônico",
    )
    parser.add_argument(
        "--test-dir",
        default="tests/",
        help="diretório de testes (default: tests/)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="timeout em segundos (default: 1800)",
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="lê output do pytest deste arquivo (em vez de executar)",
    )
    args = parser.parse_args(argv)

    if args.from_file is not None:
        text = args.from_file.read_text(encoding="utf-8")
    else:
        text = run_pytest(args.test_dir, args.timeout)

    counts = parse_pytest_summary(text)

    if args.json:
        print(json.dumps(counts, indent=2))
    else:
        print(format_canonical(counts))

    # Exit code reflete a saúde da suite — 0 só se 0 fail e 0 errors
    return 0 if (counts["failed"] == 0 and counts["errors"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
