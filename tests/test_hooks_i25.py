# -*- coding: utf-8 -*-
"""Testes I2.5: Hooks de Qualidade (Sprint v2.24).

Valida:
- check-ptbr-accentuation.sh: detecta palavras sem acento, ignora paths legacy
- generate-pr-description.sh: gera markdown estruturado a partir de git
- Catálogo .claude/ptbr-words.txt está bem-formado (TSV 2-col)
- Settings.json registra o hook PT-BR
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
PTBR_HOOK = PROJECT_ROOT / ".claude" / "hooks" / "check-ptbr-accentuation.sh"
PR_HOOK = PROJECT_ROOT / ".claude" / "hooks" / "generate-pr-description.sh"
PTBR_WORDS = PROJECT_ROOT / ".claude" / "ptbr-words.txt"
PR_TEMPLATE = PROJECT_ROOT / ".claude" / "templates" / "pr_description_template.md"
SETTINGS = PROJECT_ROOT / ".claude" / "settings.json"


# ── Existência e sintaxe ─────────────────────────────────────────


def test_ptbr_hook_exists_and_executable():
    """I2.5 — hook PT-BR existe + executável."""
    assert PTBR_HOOK.exists()
    assert PTBR_HOOK.stat().st_mode & 0o100


def test_pr_description_hook_exists_and_executable():
    """I2.5 — hook PR description existe + executável."""
    assert PR_HOOK.exists()
    assert PR_HOOK.stat().st_mode & 0o100


def test_ptbr_words_catalog_exists():
    """I2.5 — catálogo de palavras PT-BR existe."""
    assert PTBR_WORDS.exists()


def test_pr_description_template_exists():
    """I2.5 — template de descrição de PR existe."""
    assert PR_TEMPLATE.exists()


def test_ptbr_hook_bash_syntax_valid():
    """I2.5 — sintaxe bash do hook PT-BR é válida."""
    proc = subprocess.run(
        ["bash", "-n", str(PTBR_HOOK)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode == 0, f"sintaxe inválida: {proc.stderr}"


def test_pr_description_hook_bash_syntax_valid():
    """I2.5 — sintaxe bash do hook PR description é válida."""
    proc = subprocess.run(
        ["bash", "-n", str(PR_HOOK)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert proc.returncode == 0, f"sintaxe inválida: {proc.stderr}"


# ── Catálogo PT-BR ──────────────────────────────────────────────


def test_ptbr_words_catalog_tsv_format():
    """I2.5 — catálogo é TSV 2-col com palavra→acentuada."""
    content = PTBR_WORDS.read_text(encoding="utf-8")
    pairs = 0
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        assert len(parts) == 2, f"Linha mal-formada (espera 2 colunas TSV): {line!r}"
        unaccented, accented = parts
        assert unaccented and accented, f"Coluna vazia em: {line!r}"
        pairs += 1
    assert pairs >= 30, f"Catálogo PT-BR muito pequeno: {pairs} pares"


def test_ptbr_words_catalog_includes_essentials():
    """I2.5 — catálogo inclui palavras-chave do projeto."""
    content = PTBR_WORDS.read_text(encoding="utf-8")
    essentials = [
        "configuracao\tconfiguração",
        "nao\tnão",
        "execucao\texecução",
        "funcao\tfunção",
        "implementacao\timplementação",
    ]
    for pair in essentials:
        assert pair in content, f"Par essencial ausente: {pair!r}"


# ── Settings.json registra o hook ────────────────────────────────


def test_settings_json_registers_ptbr_hook():
    """I2.5 — .claude/settings.json registra check-ptbr-accentuation.sh."""
    data = json.loads(SETTINGS.read_text(encoding="utf-8"))
    hooks_text = json.dumps(data)
    assert "check-ptbr-accentuation.sh" in hooks_text


# ── Comportamento do hook PT-BR ──────────────────────────────────


def _run_ptbr_hook(file_path: str, content: str) -> tuple[int, str]:
    """Executa hook PT-BR com input simulado e retorna (exit_code, stderr)."""
    payload = json.dumps(
        {"tool_input": {"file_path": file_path, "new_string": content}}
    )
    proc = subprocess.run(
        ["bash", str(PTBR_HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return proc.returncode, proc.stderr


def test_ptbr_hook_warns_on_unaccented_text():
    """I2.5 — hook alerta sobre palavras PT sem acento em .md."""
    code, err = _run_ptbr_hook(
        "/tmp/foo.md", "Esta e uma configuracao de teste sem acentuacao."
    )
    assert code == 0, "Hook nunca bloqueia (apenas alerta)"
    assert "configuracao" in err
    assert "→" in err  # seta presente no formato


def test_ptbr_hook_silent_on_accented_text():
    """I2.5 — hook não alerta quando texto está acentuado."""
    code, err = _run_ptbr_hook(
        "/tmp/foo.md", "Esta é uma configuração com acentuação correta."
    )
    assert code == 0
    assert "configuracao" not in err
    # Não deve haver alerta sobre PT-BR
    assert "ptbr-accentuation" not in err


def test_ptbr_hook_skips_legacy_paths():
    """I2.5 — hook ignora paths em legacy/, old_geosteering_ai/, .backups/."""
    for legacy_path in [
        "/Users/foo/legacy/old.md",
        "/Users/foo/old_geosteering_ai/x.py",
        "/Users/foo/.backups/2026-05-10/file.md",
    ]:
        code, err = _run_ptbr_hook(legacy_path, "configuracao sem acento")
        assert code == 0
        assert (
            "ptbr-accentuation" not in err
        ), f"Hook não deveria alertar em path legacy: {legacy_path}"


def test_ptbr_hook_skips_non_text_files():
    """I2.5 — hook ignora arquivos não-textuais (.json, .npz, etc)."""
    for non_text in ["/tmp/x.json", "/tmp/y.npz", "/tmp/z.png"]:
        code, err = _run_ptbr_hook(non_text, "configuracao sem acento")
        assert code == 0
        assert "ptbr-accentuation" not in err


def test_ptbr_hook_bypass_via_env():
    """I2.5 — CLAUDE_BYPASS_PTBR=1 desabilita verificação."""
    payload = json.dumps(
        {"tool_input": {"file_path": "/tmp/foo.md", "new_string": "configuracao"}}
    )
    proc = subprocess.run(
        ["bash", str(PTBR_HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=5,
        env={"CLAUDE_BYPASS_PTBR": "1", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0
    assert "ptbr-accentuation" not in proc.stderr


# ── Hook PR description ──────────────────────────────────────────


def test_pr_description_template_has_required_sections():
    """I2.5 — template de PR tem seções Resumo/Mudanças/Test Plan."""
    content = PR_TEMPLATE.read_text(encoding="utf-8")
    assert "## Resumo" in content
    assert "## Mudanças" in content
    assert "## Test Plan" in content
    assert "{SUMMARY}" in content
    assert "{CHANGES}" in content
    assert "{BRANCH}" in content


def test_pr_description_hook_outputs_markdown():
    """I2.5 — hook gera Markdown válido com substituições."""
    proc = subprocess.run(
        ["bash", str(PR_HOOK), "main"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    output = proc.stdout

    # Verificar substituições básicas
    assert "## Resumo" in output
    assert "## Mudanças" in output
    assert "## Test Plan" in output
    # Não deve haver placeholders não-substituídos
    for placeholder in ["{SUMMARY}", "{CHANGES}", "{BRANCH}", "{BASE}"]:
        assert placeholder not in output, f"Placeholder não-substituído: {placeholder}"


def test_pr_description_hook_errors_on_invalid_base():
    """I2.5 — hook reporta erro quando base inválida."""
    proc = subprocess.run(
        ["bash", str(PR_HOOK), "branch-que-nao-existe-zzz"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode != 0
    assert "não existe" in proc.stderr or "not exist" in proc.stderr.lower()
