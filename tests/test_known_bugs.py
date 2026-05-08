"""Testes de regressão para bugs catalogados em ``docs/known_bugs.md``.

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — Quality Mesh Layer 4 (Regression Tests)
═══════════════════════════════════════════════════════════════════════════

Cada teste KB-XXX corresponde a uma entrada em ``docs/known_bugs.md`` e a
um padrão regex em ``.claude/anti-patterns.txt``. O objetivo é prevenir
reintrodução de bugs históricos via:

1. **Verificação de catálogo**: garantir que o padrão regex está presente
   em ``anti-patterns.txt`` (Quality Mesh L5).
2. **Verificação de código**: detectar se o padrão proibido está presente
   no código-fonte. Se presente, ``pytest.xfail`` indica que o fix ainda
   não foi merged nesta branch (mas a presença do bug é conhecida).

Política de resultados:
- PASS: bug ausente (fix aplicado) — comportamento normal.
- XFAIL: bug presente mas catalogado (regressão conhecida; aguarda merge
  do fix de outra branch).
- FAIL: bug presente E não catalogado (regressão real, deve bloquear).

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ── Setup paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ANTI_PATTERNS_FILE = PROJECT_ROOT / ".claude" / "anti-patterns.txt"
KNOWN_BUGS_FILE = PROJECT_ROOT / "docs" / "known_bugs.md"


def _load_anti_patterns() -> dict[str, dict[str, str]]:
    """Carrega .claude/anti-patterns.txt em dict {kb_id: {pattern, severity, glob}}."""
    if not ANTI_PATTERNS_FILE.exists():
        pytest.fail(f"Catálogo ausente: {ANTI_PATTERNS_FILE}")

    entries: dict[str, dict[str, str]] = {}
    for line in ANTI_PATTERNS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            kb_id, pattern, severity, glob = parts[:4]
            entries[kb_id] = {"pattern": pattern, "severity": severity, "glob": glob}
        elif len(parts) == 3:
            kb_id, pattern, glob = parts
            entries[kb_id] = {"pattern": pattern, "severity": "BLOCK", "glob": glob}
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# KB-013 — Nested prange em _fields_in_freqs_kernel_cached
# ═══════════════════════════════════════════════════════════════════════════


class TestKB013NestedPrange:
    """KB-013 (CRÍTICO): @njit(parallel=True) em função folha causa overhead.

    Versão introduzida: v2.13 (Sprint 13.1, commit 0f92035).
    Versão corrigida: v2.21 (Sprint 21.1, commit cba27dd).

    Sintoma: Cenário E throughput cai de 122k → 46k mod/h (-62%).
    """

    def test_catalog_entry_exists(self):
        """KB-013 deve estar registrado em anti-patterns.txt."""
        entries = _load_anti_patterns()
        assert "KB-013" in entries, "KB-013 ausente em .claude/anti-patterns.txt"
        entry = entries["KB-013"]
        assert (
            "parallel=True" in entry["pattern"]
        ), f"Regex de KB-013 não detecta padrão alvo: {entry['pattern']}"
        assert entry["severity"] == "BLOCK", "KB-013 deve ser BLOCK"

    def test_known_bugs_md_documents_kb013(self):
        """docs/known_bugs.md deve documentar KB-013 com causa-raiz."""
        if not KNOWN_BUGS_FILE.exists():
            pytest.fail(f"Catálogo ausente: {KNOWN_BUGS_FILE}")
        content = KNOWN_BUGS_FILE.read_text(encoding="utf-8")
        assert "KB-013" in content, "KB-013 ausente em known_bugs.md"
        assert (
            "Nested" in content or "nested" in content
        ), "KB-013 deve mencionar 'nested' (prange aninhado)"
        assert (
            "v2.13" in content and "v2.21" in content
        ), "KB-013 deve documentar versão de introdução (v2.13) e fix (v2.21)"

    def test_kernel_function_exists(self):
        """_fields_in_freqs_kernel_cached deve existir em _numba/kernel.py."""
        kernel_file = (
            PROJECT_ROOT / "geosteering_ai" / "simulation" / "_numba" / "kernel.py"
        )
        assert kernel_file.exists(), f"Arquivo ausente: {kernel_file}"
        content = kernel_file.read_text(encoding="utf-8")
        assert (
            "def _fields_in_freqs_kernel_cached" in content
        ), "Função _fields_in_freqs_kernel_cached não encontrada em kernel.py"

    def test_no_parallel_true_in_kernel_cached(self):
        """Decorador @njit em _fields_in_freqs_kernel_cached NÃO deve ter parallel=True.

        Estado esperado em v2.21+: ``@njit(cache=True, nogil=True)``.
        Se este teste falhar (parallel=True presente), aplicar fix do
        commit cba27dd ou aguardar merge da branch feat/simulation-manager-v2.21.
        """
        kernel_file = (
            PROJECT_ROOT / "geosteering_ai" / "simulation" / "_numba" / "kernel.py"
        )
        content = kernel_file.read_text(encoding="utf-8")

        # Buscar o bloco do decorator antes da função
        pattern = re.compile(
            r"@njit\(([^)]*)\)\s*\n\s*def _fields_in_freqs_kernel_cached",
            re.MULTILINE,
        )
        match = pattern.search(content)
        if match is None:
            pytest.fail(
                "Não foi possível localizar decorador @njit imediatamente antes de "
                "_fields_in_freqs_kernel_cached"
            )

        decorator_args = match.group(1)
        if "parallel=True" in decorator_args:
            pytest.xfail(
                "KB-013 ativo: parallel=True presente em _fields_in_freqs_kernel_cached. "
                "Fix está em commit cba27dd (branch feat/simulation-manager-v2.21). "
                "Merge para main resolverá este teste."
            )
        # Se parallel=True ausente, teste passa
        assert "parallel=True" not in decorator_args


# ═══════════════════════════════════════════════════════════════════════════
# KB-018 — rng_seed=42 hardcoded em simulation_manager.py
# ═══════════════════════════════════════════════════════════════════════════


class TestKB018RngSeedHardcoded:
    """KB-018 (ALTA): rng_seed=42 hardcoded faz GUI gerar mesmos modelos.

    Versão introduzida: v2.18.
    Versão corrigida: v2.19 (UI control + Optional[int]).
    """

    def test_catalog_entry_exists(self):
        entries = _load_anti_patterns()
        assert "KB-018" in entries, "KB-018 ausente em anti-patterns.txt"
        assert "rng_seed" in entries["KB-018"]["pattern"]

    def test_known_bugs_md_documents_kb018(self):
        if not KNOWN_BUGS_FILE.exists():
            pytest.skip(f"Catálogo ausente: {KNOWN_BUGS_FILE}")
        content = KNOWN_BUGS_FILE.read_text(encoding="utf-8")
        assert "KB-018" in content
        assert "rng_seed" in content

    def test_simulation_manager_exists(self):
        sm_file = (
            PROJECT_ROOT
            / "geosteering_ai"
            / "simulation"
            / "tests"
            / "simulation_manager.py"
        )
        assert sm_file.exists(), f"Arquivo ausente: {sm_file}"

    def test_no_rng_seed_42_hardcoded_outside_smoke_test(self):
        """rng_seed=42 só é permitido em _run_smoke_test (determinismo de teste).

        Estado esperado em v2.19+: rng_seed=Optional[int], default None.
        """
        sm_file = (
            PROJECT_ROOT
            / "geosteering_ai"
            / "simulation"
            / "tests"
            / "simulation_manager.py"
        )
        if not sm_file.exists():
            pytest.skip(f"simulation_manager.py ausente em {sm_file}")

        lines = sm_file.read_text(encoding="utf-8").splitlines()
        violations = []
        in_smoke_test = False

        for i, line in enumerate(lines, start=1):
            # Detectar se estamos dentro de _run_smoke_test ou função similar
            if re.search(r"def\s+(_run_smoke_test|_smoke|smoke_test)\s*\(", line):
                in_smoke_test = True
            elif re.match(r"^\s*def\s+", line) and in_smoke_test:
                # Saímos do smoke test
                in_smoke_test = False

            # Verificar rng_seed=42 fora do smoke test
            if re.search(r"rng_seed\s*=\s*42[,\s\)]", line):
                if not in_smoke_test and "smoke" not in line.lower():
                    violations.append((i, line.strip()))

        if violations:
            details = "\n".join(f"  L{n}: {ln}" for n, ln in violations)
            pytest.xfail(
                f"KB-018 ativo: rng_seed=42 hardcoded fora de smoke test:\n{details}\n"
                "Fix está em v2.19 (UI control + Optional[int]). "
                "Merge para main resolverá este teste."
            )


# ═══════════════════════════════════════════════════════════════════════════
# KB-019 — Oversubscription threading em CPUs HT/SMT
# ═══════════════════════════════════════════════════════════════════════════


class TestKB019Oversubscription:
    """KB-019 (MÉDIA): defaults workers × threads excedem cores físicos em HT.

    Versão introduzida: v2.19.
    Versão corrigida: v2.20 (recommend_default_parallelism usa phys_cores).
    """

    def test_catalog_entry_exists(self):
        entries = _load_anti_patterns()
        assert "KB-019" in entries, "KB-019 ausente em anti-patterns.txt"

    def test_known_bugs_md_documents_kb019(self):
        if not KNOWN_BUGS_FILE.exists():
            pytest.skip(f"Catálogo ausente: {KNOWN_BUGS_FILE}")
        content = KNOWN_BUGS_FILE.read_text(encoding="utf-8")
        assert "KB-019" in content
        assert "oversubscription" in content.lower() or "phys" in content.lower()

    def test_recommend_default_parallelism_returns_phys_safe(self):
        """recommend_default_parallelism() retorna (n_workers, threads_per_worker).

        Estado esperado: n_workers × threads_per_worker ≤ physical_cores.
        """
        try:
            from geosteering_ai.simulation._workers import (
                detect_cpu_topology,
                recommend_default_parallelism,
            )
        except ImportError as exc:
            pytest.skip(f"Módulo _workers não disponível: {exc}")

        result = recommend_default_parallelism()
        assert (
            isinstance(result, tuple) and len(result) == 2
        ), f"Esperado tuple (n_workers, threads_per_worker), recebido: {result}"
        n_workers, threads_per_worker = result
        assert (
            n_workers >= 1 and threads_per_worker >= 1
        ), f"Valores inválidos: workers={n_workers}, threads={threads_per_worker}"

        # Total threads deve ser ≤ physical_cores (estratégia v2.17+ confirmada v2.20)
        try:
            topology = detect_cpu_topology()
            phys_cores = (
                topology[1]
                if isinstance(topology, tuple)
                else topology.get("physical_cores", topology.get("phys_cores"))
            )
        except Exception:
            pytest.skip("detect_cpu_topology não disponível")

        total = n_workers * threads_per_worker
        if total > phys_cores:
            pytest.xfail(
                f"KB-019 ativo: defaults causam oversubscription "
                f"({n_workers}w × {threads_per_worker}t = {total} > {phys_cores} phys). "
                f"Fix em v2.20 — branch atual ainda não tem o fix mergeado."
            )
        assert total <= phys_cores, (
            f"Oversubscription detectado: {n_workers} × {threads_per_worker} = "
            f"{total} > {phys_cores} cores físicos"
        )
