"""Testes de validação do .worktreeinclude e infraestrutura de worktrees.

Valida que:
- O arquivo .worktreeinclude existe e lista apenas paths que existem no repo
- Diretórios pesados estão corretamente excluídos (não listados)
- O venv NÃO está na lista de inclusão (é configurado via symlink pelo script)
- O script worktree-create.sh existe, é executável e tem sintaxe válida

Estes testes são rápidos (<1s) e não criam worktrees reais (sem side-effects).
Contexto: I1.7 da Fase 1 §22.1 — Fundação Multi-Agente.
"""

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
WORKTREEINCLUDE = REPO_ROOT / ".worktreeinclude"
WORKTREE_SCRIPT = REPO_ROOT / ".claude" / "scripts" / "worktree-create.sh"

# Diretórios pesados que NUNCA devem aparecer como entries de inclusão
HEAVY_DIRS = [
    "Relatorio_2025",
    "Relatorio_2026",
    "Resultados_Relatorio_2026",
    ".backups",
    "old_geosteering_ai",
    "Prompts_Geosteering_AI",
    "latex-document-skill-main.zip",
]


def _read_included_paths() -> list[str]:
    """Retorna lista de paths incluídos (sem comentários e vazios)."""
    lines = []
    for line in WORKTREEINCLUDE.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            lines.append(line)
    return lines


# ── Testes do .worktreeinclude ───────────────────────────────────────


def test_worktreeinclude_exists():
    """Arquivo .worktreeinclude deve existir na raiz do repositório."""
    assert WORKTREEINCLUDE.exists(), f".worktreeinclude não encontrado em {REPO_ROOT}"
    assert WORKTREEINCLUDE.is_file()


def test_worktreeinclude_has_entries():
    """Arquivo deve conter ao menos 5 paths de inclusão."""
    paths = _read_included_paths()
    assert len(paths) >= 5, f"Esperado ≥5 entradas, encontrado {len(paths)}: {paths}"


def test_all_listed_paths_exist():
    """Cada path listado no .worktreeinclude deve existir no repositório.

    Garante que o arquivo não referencia diretórios inexistentes — o que
    causaria erros silenciosos ao configurar sparse-checkout.
    """
    paths = _read_included_paths()
    missing = []
    for p in paths:
        target = REPO_ROOT / p.rstrip("/")
        if not target.exists():
            missing.append(p)
    assert not missing, (
        f".worktreeinclude lista {len(missing)} path(s) inexistente(s): {missing}\n"
        "Remova ou corrija os paths inválidos."
    )


def test_heavy_dirs_not_listed():
    """Diretórios pesados (Relatorio_*, .backups, etc.) não devem ser incluídos.

    Esses diretórios podem ter dezenas de GB; incluí-los em worktrees
    desperdiçaria disco e tornaria a criação muito lenta.
    """
    content = WORKTREEINCLUDE.read_text(encoding="utf-8")
    included = _read_included_paths()
    for heavy in HEAVY_DIRS:
        assert heavy not in included, (
            f"Diretório pesado '{heavy}' está listado em .worktreeinclude. "
            "Deve ser excluído — worktrees não devem copiar dados pesados."
        )
    # Verificação secundária: não deve aparecer nem como entry não-comentada
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for heavy in HEAVY_DIRS:
            assert (
                heavy not in stripped
            ), f"Diretório pesado '{heavy}' encontrado em linha não-comentada: {line!r}"


def test_venv_not_listed_as_included():
    """Geosteering_AI_venv NÃO deve estar na lista de inclusão.

    O venv é configurado via symlink pelo worktree-create.sh, não copiado.
    Copiar ~2 GB do venv para cada worktree seria contraproducente (I1.7).
    """
    paths = _read_included_paths()
    venv_entries = [p for p in paths if "Geosteering_AI_venv" in p]
    assert not venv_entries, (
        f"Geosteering_AI_venv está na lista de inclusão: {venv_entries}\n"
        "O venv deve ser symlinked pelo worktree-create.sh, não copiado."
    )


def test_essential_dirs_present():
    """Diretórios essenciais para desenvolvimento devem estar incluídos."""
    paths = _read_included_paths()
    essential = ["geosteering_ai/", "tests/", "docs/", ".claude/", "tools/"]
    for required in essential:
        assert required in paths, (
            f"Diretório essencial '{required}' não está em .worktreeinclude. "
            "Worktrees criadas sem ele não terão acesso ao código necessário."
        )


# ── Testes do script worktree-create.sh ─────────────────────────────


def test_worktree_create_script_exists():
    """Script .claude/scripts/worktree-create.sh deve existir."""
    assert WORKTREE_SCRIPT.exists(), (
        f"worktree-create.sh não encontrado em {WORKTREE_SCRIPT}\n"
        "Crie o script conforme especificação I1.7 (§18.3 do doc de aprofundamento)."
    )


def test_worktree_create_script_is_executable():
    """Script deve ter permissão de execução."""
    assert os.access(WORKTREE_SCRIPT, os.X_OK), (
        f"worktree-create.sh não é executável: {WORKTREE_SCRIPT}\n"
        "Execute: chmod +x .claude/scripts/worktree-create.sh"
    )


def test_worktree_create_script_syntax():
    """Script deve passar verificação de sintaxe bash sem erros."""
    result = subprocess.run(
        ["bash", "-n", str(WORKTREE_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Erro de sintaxe em worktree-create.sh:\n{result.stderr}"
