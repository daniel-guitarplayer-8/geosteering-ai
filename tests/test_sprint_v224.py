# -*- coding: utf-8 -*-
"""Testes da Sprint v2.24: Débitos Técnicos (Sprint v2.23 follow-up).

Valida:
- 1.1 Script count_pytest_pass.py existe + executável + parser correto
- 1.2 setup-environment.sh tem acentuação PT-BR
- 1.3 Todos @njit em _numba/ têm cache=True explícito
- 1.4 use_fastmath em config.py está documentado como "status documental"

Contexto: Sprint v2.24 fecha 4 débitos identificados na revisão final
da Sprint v2.23 (M1, M2, N2, N6) e ainda preserva paridade <1e-12.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ── Frente 1.1 — count_pytest_pass.py ────────────────────────────


def test_count_pytest_pass_script_exists():
    """v2.24 1.1 — script existe em scripts/."""
    script = PROJECT_ROOT / "scripts" / "count_pytest_pass.py"
    assert script.exists(), f"Script ausente em {script}"


def test_count_pytest_pass_script_executable():
    """v2.24 1.1 — script tem permissão de execução."""
    script = PROJECT_ROOT / "scripts" / "count_pytest_pass.py"

    mode = script.stat().st_mode
    assert mode & 0o100, f"Script não é executável: {oct(mode)}"


def _load_count_pytest_pass():
    """Carrega scripts/count_pytest_pass.py como módulo isolado.

    Usa importlib.util para evitar manipulação de sys.path (W7 do
    code-review v2.24): cleanup limpo, sem efeitos colaterais
    em testes paralelos.
    """
    import importlib.util

    script = PROJECT_ROOT / "scripts" / "count_pytest_pass.py"
    spec = importlib.util.spec_from_file_location("count_pytest_pass", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_count_pytest_pass_parses_canonical_output():
    """v2.24 1.1 — extrai PASS/SKIP/FAIL do output canônico do pytest."""
    mod = _load_count_pytest_pass()
    text = "1624 passed, 295 skipped in 911.45s"
    counts = mod.parse_pytest_summary(text)
    assert counts["passed"] == 1624
    assert counts["skipped"] == 295
    assert counts["failed"] == 0


def test_count_pytest_pass_format_canonical():
    """v2.24 1.1 — formata como 'X PASS / Y SKIP / Z FAIL'."""
    mod = _load_count_pytest_pass()
    result = mod.format_canonical(
        {"passed": 1624, "skipped": 295, "failed": 0, "xfailed": 0, "errors": 0}
    )
    assert result == "1624 PASS / 295 SKIP / 0 FAIL"


def test_count_pytest_pass_json_mode():
    """v2.24 1.1 — modo --json emite JSON estruturado."""
    script = PROJECT_ROOT / "scripts" / "count_pytest_pass.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json", "--from-file", "/dev/stdin"],
        input="1624 passed, 295 skipped in 911s",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["passed"] == 1624
    assert data["skipped"] == 295
    assert data["failed"] == 0


# ── Frente 1.2 — setup-environment.sh PT-BR ──────────────────────


def test_setup_env_no_unaccented_pt_words():
    """v2.24 1.2 — setup-environment.sh não tem palavras PT sem acento."""
    hook = PROJECT_ROOT / ".claude" / "hooks" / "setup-environment.sh"
    content = hook.read_text(encoding="utf-8")

    forbidden = [
        r"\bnao\b",
        r"\bsessao\b",
        r"\binicio\b",
        r"\bultimo\b",
        r"\brapido\b",
        r"\bversao\b",
        r"\binformacoes\b",
        r"\bdiretorio\b",
        r"\brecomendacao\b",
        r"\bmodulo\b",
        r"\bimportavel\b",
    ]
    # Permite usos legítimos em strings de comparação como `if has_ht else 'não'`
    # após a correção. Verifica que NENHUM destes padrões ocorre.
    for pattern in forbidden:
        matches = re.findall(pattern, content, flags=re.IGNORECASE)
        assert not matches, (
            f"Palavra PT sem acento detectada em setup-environment.sh: "
            f"pattern={pattern!r}, matches={matches}"
        )


def test_setup_env_has_accented_pt_words():
    """v2.24 1.2 — setup-environment.sh contém palavras PT com acento."""
    hook = PROJECT_ROOT / ".claude" / "hooks" / "setup-environment.sh"
    content = hook.read_text(encoding="utf-8")

    expected = ["não", "Sessão", "Injeção", "informações"]
    for word in expected:
        assert word in content, f"Palavra acentuada ausente: {word!r}"


def test_setup_env_bash_syntax_valid():
    """v2.24 1.2 — sintaxe bash do hook é válida."""
    hook = PROJECT_ROOT / ".claude" / "hooks" / "setup-environment.sh"
    proc = subprocess.run(
        ["bash", "-n", str(hook)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode == 0, f"sintaxe bash inválida: {proc.stderr}"


# ── Frente 1.3 — cache=True em todos @njit ───────────────────────


def test_all_njit_decorators_have_cache_true():
    """v2.24 1.3 — TODO decorador @njit em _numba/ tem cache=True.

    Procura recursivamente por `@njit` (sem parênteses) ou `@njit(...)` que
    não contenha `cache=True`. Se encontrar, falha listando os arquivos:linhas.
    """
    numba_dir = PROJECT_ROOT / "geosteering_ai" / "simulation" / "_numba"
    violations = []
    for py_file in numba_dir.glob("*.py"):
        for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
            stripped = line.strip()
            # Match @njit, @njit(), @njit(fastmath=True), etc — exclui anotações em strings
            if not stripped.startswith("@njit"):
                continue
            # Aceita qualquer decorador que mencione cache=True
            if "cache=True" not in stripped:
                # Trata caso especial: @njit é equivalente a @njit() sem cache
                violations.append(f"{py_file.name}:{lineno}: {stripped}")
    assert not violations, (
        "Decoradores @njit sem cache=True (Sprint v2.24 1.3):\n" + "\n".join(violations)
    )


# ── Frente 1.4 — use_fastmath removido em v2.36 D4 ───────────────
# Testes obsoletos removidos em v2.36 D4 (campo era puramente documental
# com 0 leituras dinâmicas — removido para limpar a API pública).
