---
Spec: 0003-cli-backend-auto
Plano-de: spec.md
Status: planejado
Data: 2026-06-05
---

# Plano 0003 — CLI `--backend auto` (HOW)

> Pré-requisito: `spec.md` passou GATE-S (0 `[NEEDS CLARIFICATION]`).

## 1. Gate de Constituição

| Princípio | Aplicável? | Viola? | Como o plano cumpre |
|:--|:--:|:--:|:--|
| I Paridade (3 regimes) | não | não | **não toca cálculo EM**; `auto` só escolhe backend. `Converge-Em: n/a` |
| II Errata imutável | não | não | nenhuma constante física tocada |
| III TF/Keras exclusivo | sim | não | sem torch |
| IV Python 3.13 | sim | não | — |
| V Config-parâmetro | sim | não | sem globals; usa `args`/funções puras |
| VI Logging | sim | não | `logger` em `_backend`/`_exec`; `print` só no contrato stdout da CLI |
| VII D1–D14 | sim | não | docstrings D5/D7 atualizadas nos pontos tocados |
| VIII PT-BR | sim | não | acentuação correta |
| IX SSoT | sim | não | `Backlog-Code: E-cli-auto` (a promover em ROADMAP §0) |
| X MVVM | n/a | — | não é GUI |
| XI Fundação/física | sim | não | reusa `dispatch._resolve_backend`; **não reimplementa heurística** |
| XII Gates | sim | não | testes + code-reviewer (Trilha E) |

**GATE-P: sem violação sem-ADR.** Esta spec **não gera ADR** (mudança MINOR, reversível).

## 2. Arquitetura Técnica

```
┌────────────────────────── geosteering-cli simulate|benchmark ──────────────────────────┐
│                                                                                          │
│  argparse (_main.py)                                                                      │
│   --backend {numba, jax, AUTO}   default = None  ◄── (era "numba"; vira sentinela)        │
│        │                                                                                  │
│        ▼  handle_simulate / handle_benchmark                                              │
│  resolve_requested_backend(args)  [NOVO, _exec.py]                                        │
│   • args.backend is None → DeprecationWarning("default → auto em v2.57.0") + "numba"      │
│   • senão → args.backend  (numba|jax|auto)                                                │
│        │ requested ∈ {numba, jax, auto}                                                   │
│        ▼  run_once / preparação                                                           │
│  resolve_backend_preflight(requested, esp_batch, n_models, quiet)  [_exec.py:217 — ESTENDER]│
│   ┌────────────────────────────────────────────────────────────────────────────────┐    │
│   │ requested == "auto":                                                             │    │
│   │   n_groups = _count_geometry_groups(esp_batch)     # NumPy puro (TLS-safe)        │    │
│   │   backend, reason, _ = dispatch._resolve_backend(n_models, esp_batch, ...)         │    │
│   │       └─ árvore MEDIDA + guard anti-OOM 80GB (dispatch.py:94)  ◄── ÚNICA FONTE     │    │
│   │   device = "gpu" if backend=="jax" else "cpu"                                      │    │
│   │ requested ∈ {numba, jax}:  (caminho atual, inalterado)                            │    │
│   └────────────────────────────────────────────────────────────────────────────────┘    │
│        │ (backend_efetivo ∈ {numba, jax}, device, reason)                                 │
│        ▼                                                                                   │
│  warmup_backend / run_once / tabela+JSON  → reportam backend EFETIVO (nunca "auto")       │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Decisão-chave:** `auto` é resolvido para um backend **concreto** (`numba`|`jax`) no pré-voo, ANTES
do compute — assim warmup, device, reporting, `--compare-backends` e `--json` funcionam sem mudança.
A heurística vem **exclusivamente** de `dispatch._resolve_backend` (AC-2.2); a CLI não a duplica.

## 3. Contratos / APIs

```python
# geosteering_ai/cli/_exec.py  (NOVO)
def resolve_requested_backend(args: argparse.Namespace) -> str:
    """Normaliza args.backend e emite DeprecationWarning quando ausente.

    Returns:
        "numba" (default implícito, com aviso) | "numba" | "jax" | "auto".
    Note:
        Quando `args.backend is None` (usuário não passou --backend), emite
        DeprecationWarning citando a mudança de default para "auto" em v2.57.0
        e retorna "numba" (comportamento compatível com hoje).
    """

# geosteering_ai/cli/_exec.py  (ESTENDER resolve_backend_preflight — _exec.py:217)
# Assinatura REAL (confirmada em T00): recebe `models` (lista de dicts), não esp_batch.
def resolve_backend_preflight(
    requested: str,            # agora aceita "auto" além de "numba"/"jax"
    models: Sequence[dict],
    *, quiet: bool = False,
) -> tuple[str, str, str | None]:
    """... + ramo 'auto' TLS-safe: reusa as CONSTANTES do dispatcher
    (_GROUPABLE_RATIO_MAX, _N_MODELS_GPU_THRESHOLD, _jax_gpu_available) e
    _count_geometry_groups (NumPy) — NÃO chama dispatch._resolve_backend
    (importa _jax.multi_forward + sonda GPU antes da agrupabilidade → risco TLS).
    Ordem: n_models<limiar? → não-agrupável? → (só então) sonda GPU.
    Retorna SEMPRE backend concreto ('numba'|'jax'); nunca 'auto'."""
```

| Símbolo | Mudança |
|:--|:--|
| `_main.py` `--backend` | `choices=["numba","jax","auto"]`, `default=None` (sentinela) |
| `_main.py` help | descreve `auto` (árvore medida) + aviso de futura mudança de default |
| `_exec.py::resolve_requested_backend` | NOVO |
| `_exec.py::resolve_backend_preflight` | ramo `"auto"` → `dispatch._resolve_backend` |

## 4. Estrutura de Arquivos

| Arquivo | Ação | Conteúdo |
|:--|:--|:--|
| `geosteering_ai/cli/_main.py` | modificar | `choices`+`default=None`+help; chamar `resolve_requested_backend` em `handle_*` (via `_exec`) |
| `geosteering_ai/cli/_exec.py` | modificar | `resolve_requested_backend` (novo) + ramo `auto` em `resolve_backend_preflight` |
| `geosteering_ai/cli/simulate.py` | modificar (mín.) | usar `resolve_requested_backend(args)` antes do preflight |
| `geosteering_ai/cli/benchmark.py` | modificar (mín.) | idem |
| `tests/test_cli_backend_auto.py` | criar | AC-1.1..AC-4.1 |

## 5. Decisões de Design / ADRs

| Decisão | Opções | Escolha | Justificativa | ADR? |
|:--|:--|:--|:--|:--:|
| detectar "default implícito" | sentinela `default=None` vs `argparse` action custom | `default=None` | simples, não quebra `choices`; distingue ausência de escolha explícita | não |
| fonte da heurística `auto` | chamar `_resolve_backend` vs **reusar constantes+primitivas TLS-safe** | reusar constantes+primitivas | T00: `_resolve_backend` importa `_jax` + sonda GPU 1º → não-TLS-safe; reusar `_GROUPABLE_RATIO_MAX`/`_N_MODELS_GPU_THRESHOLD`/`_jax_gpu_available` evita drift (Princ. XI) sem o risco | não |
| mudar default agora | sim vs **não (1 minor)** | **não** | C6: compat de scripts/CI; PEP 387 DeprecationWarning → muda em v2.57.0 | não |

## 6. Riscos Técnicos e Mitigações

| Risco | Prob. | Impacto | Mitigação |
|:--|:--:|:--:|:--|
| crash TLS no ramo `auto` (CUDA init antes do pool Numba) | ~~média~~ **mitigado (T00)** | alto | **RESOLVIDO**: a CLI NÃO chama `_resolve_backend`; replica a árvore com `_count_geometry_groups` (NumPy) + sonda GPU **por último**. O caminho Numba nunca importa jax (defesa v2.55 preservada) |
| `default=None` quebra leitura de `args.backend` em algum ponto | baixa | médio | normalizar cedo via `resolve_requested_backend`; grep por usos de `args.backend` |
| Warning aparece em massa no CI (que não passa `--backend`) | média | baixo | CI usa `simulate`/`benchmark` — aceitável (é o sinal pretendido); pode-se silenciar com `--backend numba` explícito nos workflows |

## 7. Estratégia de Teste

| Camada | O quê | Onde |
|:--|:--|:--|
| unit | `resolve_requested_backend`: None→warn+numba; "numba"→sem warn | `tests/test_cli_backend_auto.py` |
| unit | `resolve_backend_preflight("auto", ...)` chama `dispatch._resolve_backend` (monkeypatch) e retorna concreto | idem (AC-2.1, AC-2.2) |
| e2e (subprocess) | `simulate --backend auto` exit 0; `--backend invalido` exit 2; `--json` backend∈{numba,jax} | idem (AC-1.1..1.3, AC-4.1) |
| warn | `pytest.warns(DeprecationWarning)` sem `--backend`; ausência com `--backend numba` | idem (AC-3.1, AC-3.2) |
| paridade | **N/A** (não toca EM) — declarado | — |
| regressão | suíte `tests/test_cli_*.py` verde (comportamento default inalterado) | CI |

## 8. Critério de Pronto do Plano (GATE-P)
- [x] Tabela de constituição sem violação sem-ADR
- [x] Contratos com assinaturas exatas
- [x] Nenhum ADR necessário (MINOR reversível)
- [x] Estratégia de teste cobre AC-1..AC-4 + RNF-2 (compat)
