"""Regras de paralelismo entre agentes Claude Code (Etapa 1 Multi-Agent).

═══════════════════════════════════════════════════════════════════════════
GEOSTEERING AI v2.0 — Multi-Agent Parallelism Rules (§40 documento-base)
═══════════════════════════════════════════════════════════════════════════

Define quais agentes podem ou NÃO podem rodar simultaneamente, com base em:

1. **CONFLITO**: agentes que tocam o MESMO subsistema NÃO podem paralelizar
   (ex.: numba-jit-engineer + numba-validator vão competir em ``_numba/*``).

2. **PODEM_PARALELIZAR**: tuplas de domínios independentes onde paralelismo
   é seguro (ex.: numba + dl-training operam em diretórios distintos).

3. **LIMITES**: caps globais por modelo + profundidade + total simultâneo,
   alinhados com tier de custo Anthropic API.

Lido por:
- ``geosteering_ai.multi_agent.conflict_matrix`` (Python testável)
- ``.claude/hooks/acquire-lock.sh`` (validação em tempo real)

Mantido em ``.claude/`` (fora de geosteering_ai/) por se tratar de config
do harness Claude Code, não de código de produção. Carregado dinamicamente
via ``importlib.util.spec_from_file_location`` quando necessário.

═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONFLITO — agentes que NÃO podem rodar simultaneamente
# ═══════════════════════════════════════════════════════════════════════════
#
# Estrutura: dict[str, list[str]]
#   chave   = agente A
#   valor   = lista de agentes que conflitam com A (não-comutativo: precisa
#             estar em ambos os sentidos para captura simétrica)
#
# Critérios de conflito:
#   - Editam o MESMO subsistema (ex.: ambos tocam _numba/*)
#   - Um valida o que outro escreve (ex.: jax-engineer vs jax-validator)
#   - Sequenciam append a arquivo compartilhado (CHANGELOG, ROADMAP)
#
CONFLITO: Dict[str, List[str]] = {
    # Subsistema Numba JIT — só um editor por vez
    "numba-jit-engineer": [
        "numba-validator",
        "fortran-parity-validator",
        "validate-physics-hook",
    ],
    "numba-validator": [
        "numba-jit-engineer",
    ],
    "fortran-parity-validator": [
        "numba-jit-engineer",
        "jax-engineer",
    ],
    # Subsistema JAX — análogo
    "jax-engineer": [
        "jax-validator",
        "fortran-parity-validator",
    ],
    "jax-validator": [
        "jax-engineer",
    ],
    # Subsistema DL — engineer e architecture-engineer não tocam ao mesmo tempo
    "dl-training-engineer": [
        "dl-architecture-engineer",
    ],
    "dl-architecture-engineer": [
        "dl-training-engineer",
    ],
    # Subsistema docs — sequencializam para evitar race em CHANGELOG/ROADMAP
    "docs-writer": [
        "repo-housekeeper",
        "upgrade-scout",
    ],
    "repo-housekeeper": [
        "docs-writer",
    ],
    "upgrade-scout": [
        "docs-writer",
    ],
    # Hooks críticos não paralelizam com edits no _numba
    "validate-physics-hook": [
        "numba-jit-engineer",
        "jax-engineer",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. PODEM_PARALELIZAR — domínios independentes onde paralelismo é seguro
# ═══════════════════════════════════════════════════════════════════════════
#
# Estrutura: set[tuple[str, str]] (ordenado lexicograficamente para
# busca determinística)
#
# Wildcard "any-other-agent" indica compatível com qualquer agente
# (apenas para agentes read-only).
#
PODEM_PARALELIZAR: Set[Tuple[str, str]] = {
    ("dl-training-engineer", "numba-jit-engineer"),
    ("docs-writer", "physics-validator-mcp"),
    ("jax-engineer", "literature-research-agent"),
    ("any-other-agent", "upgrade-scout"),  # upgrade-scout é read-only
    ("any-other-agent", "literature-research-agent"),  # read-only
}


# ═══════════════════════════════════════════════════════════════════════════
# 3. LIMITES globais por modelo + total
# ═══════════════════════════════════════════════════════════════════════════
#
# Alinhados com:
#   - Custo médio mensal Anthropic API (€200/mês objetivo §40.8)
#   - Limites práticos de paralelismo cognitivo (humano coordena ≤5 agentes)
#   - Profundidade max 2: orchestrator → specialist (sem sub-sub-agents)
#
LIMITES: Dict[str, int] = {
    "max_sonnet_concurrent": 4,   # Sonnet 4.5/4.6 — workhorse
    "max_haiku_concurrent": 2,    # Haiku 4.5 — validações rápidas
    "max_opus_concurrent": 1,     # Opus 4.7 — orquestrador único
    "max_depth": 2,               # orchestrator → specialist → tool
    "max_total_simultaneous": 5,  # cap absoluto
    "lock_ttl_sec": 300,          # locks expiram em 5 min sem renew
    "stale_pid_check": 1,         # 1 = checar PID via os.kill(p, 0)
}


# ═══════════════════════════════════════════════════════════════════════════
# 4. API utilitária
# ═══════════════════════════════════════════════════════════════════════════


def can_run_together(agent_a: str, agent_b: str) -> bool:
    """Retorna ``True`` se os dois agentes podem rodar em paralelo, ``False``
    caso haja conflito catalogado.

    Política:
      1. Mesmos agentes (a == b): ``False`` (não duplica).
      2. Conflito explícito em ``CONFLITO``: ``False``.
      3. Par em ``PODEM_PARALELIZAR``: ``True``.
      4. Wildcard ``any-other-agent``: ``True`` se um dos lados é read-only.
      5. Default: ``True`` (whitelist conservadora — explícito > implícito).

    Examples:
        >>> can_run_together("numba-jit-engineer", "numba-validator")
        False
        >>> can_run_together("numba-jit-engineer", "dl-training-engineer")
        True
        >>> can_run_together("docs-writer", "docs-writer")
        False
    """
    if agent_a == agent_b:
        return False

    # Conflito explícito (bidirecional)
    conflicts_a = CONFLITO.get(agent_a, [])
    conflicts_b = CONFLITO.get(agent_b, [])
    if agent_b in conflicts_a or agent_a in conflicts_b:
        return False

    # Par explícito autorizado
    pair = tuple(sorted([agent_a, agent_b]))
    if pair in PODEM_PARALELIZAR:
        return True

    # Wildcard read-only
    for cat_pair in PODEM_PARALELIZAR:
        if "any-other-agent" in cat_pair:
            other = cat_pair[0] if cat_pair[1] == "any-other-agent" else cat_pair[1]
            if agent_a == other or agent_b == other:
                return True

    # Default: permite (whitelist conservadora)
    return True


def get_conflicts(agent: str) -> List[str]:
    """Retorna lista de agentes em conflito com ``agent``."""
    return CONFLITO.get(agent, [])


def check_limits(active_agents: List[Dict[str, str]]) -> List[str]:
    """Valida limites globais de modelo. Retorna lista de violações.

    Args:
        active_agents: lista de dicts ``{"name": str, "model": "sonnet|haiku|opus"}``.

    Returns:
        Lista de mensagens de violação (vazia se tudo ok).
    """
    violations: List[str] = []
    counts = {"sonnet": 0, "haiku": 0, "opus": 0, "total": len(active_agents)}

    for ag in active_agents:
        model = ag.get("model", "sonnet").lower()
        if model in counts:
            counts[model] += 1

    if counts["sonnet"] > LIMITES["max_sonnet_concurrent"]:
        violations.append(
            f"Sonnet: {counts['sonnet']} ativos > limite {LIMITES['max_sonnet_concurrent']}"
        )
    if counts["haiku"] > LIMITES["max_haiku_concurrent"]:
        violations.append(
            f"Haiku: {counts['haiku']} ativos > limite {LIMITES['max_haiku_concurrent']}"
        )
    if counts["opus"] > LIMITES["max_opus_concurrent"]:
        violations.append(
            f"Opus: {counts['opus']} ativos > limite {LIMITES['max_opus_concurrent']}"
        )
    if counts["total"] > LIMITES["max_total_simultaneous"]:
        violations.append(
            f"Total: {counts['total']} agentes > cap {LIMITES['max_total_simultaneous']}"
        )

    return violations


__all__ = [
    "CONFLITO",
    "PODEM_PARALELIZAR",
    "LIMITES",
    "can_run_together",
    "get_conflicts",
    "check_limits",
]


if __name__ == "__main__":
    # CLI smoke test
    import sys

    print("[parallelism-rules] Conflitos catalogados:", len(CONFLITO))
    print("[parallelism-rules] Pares paralelizáveis :", len(PODEM_PARALELIZAR))
    print("[parallelism-rules] Limites              :", LIMITES)
    print()

    if len(sys.argv) >= 3:
        a, b = sys.argv[1], sys.argv[2]
        ok = can_run_together(a, b)
        print(f"  {a} ↔ {b}: {'OK' if ok else 'CONFLITO'}")
        sys.exit(0 if ok else 1)
