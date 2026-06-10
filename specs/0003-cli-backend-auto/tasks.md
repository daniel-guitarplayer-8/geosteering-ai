---
Spec: 0003-cli-backend-auto
Tarefas-de: plan.md
Status: planejado
Data: 2026-06-05
---

# Tarefas 0003 — CLI `--backend auto`

> Cada tarefa: ATÔMICA (1 commit), TESTÁVEL, com ordem/dependências. Mapeia para RF/AC do spec.

## Grafo de Dependências
```
T01 ─► T02 ─► T03 ─► T05 ─► T07 (verify)
  │            └► T04 ──────┘
  └► T06 (docs/help, paraleliza)
T00 (investigação TLS) ─► T03   (bloqueia o ramo auto)
```

## Lista de Tarefas
| # | Tarefa | Cobre | Arquivos | Dep | Commit (mensagem) | Done quando |
|:--|:--|:--|:--|:--:|:--|:--|
| **T00 ✓** | **CONCLUÍDA**: `dispatch._resolve_backend` sonda GPU (`_jax_gpu_available`, l.158) E importa `_jax.multi_forward` (l.121) independente da agrupabilidade → **não-TLS-safe p/ a CLI**. Decisão: ramo `auto` replica a árvore via `_count_geometry_groups` + constantes do dispatcher, GPU por último. | Risco-TLS | (análise) | — | (sem commit) | ✓ rota TLS-safe definida; AC-2.2 revisado |
| **T01** | `resolve_requested_backend(args)` em `_exec.py`: `None→warn(DeprecationWarning, "default → auto em v2.57.0")+"numba"`; senão devolve `args.backend`. | RF-3/AC-3.1,3.2 | `cli/_exec.py` | T00 | `feat(cli): resolve_requested_backend + DeprecationWarning de default` | AC-3.1, AC-3.2 (unit) |
| **T02** | `--backend`: `choices=["numba","jax","auto"]`, `default=None`; help descreve `auto` + aviso. Normalizar via `resolve_requested_backend` em `handle_simulate`/`handle_benchmark`. | RF-1/AC-1.3, RF-3 | `cli/_main.py`, `cli/simulate.py`, `cli/benchmark.py` | T01 | `feat(cli): --backend auto (choice) mantendo default numba` | AC-1.3 (exit 2 p/ inválido); default inalterado |
| **T03** | Ramo `"auto"` em `resolve_backend_preflight`: **replica** a árvore via `_count_geometry_groups` + `_GROUPABLE_RATIO_MAX` + `_N_MODELS_GPU_THRESHOLD` + `_jax_gpu_available` (TLS-safe, GPU por último); retorna backend concreto. | RF-2/AC-2.1,2.2 | `cli/_exec.py` | T02 | `feat(cli): backend=auto (árvore do dispatcher, TLS-safe)` | AC-2.1, AC-2.2 (consistência) |
| **T04** | Reporting: tabela + `--json` mostram backend EFETIVO quando `auto`. | RF-4/AC-4.1 | `cli/_exec.py`, `cli/_table.py`, `cli/simulate.py` | T02 | `feat(cli): reporta backend resolvido (nunca 'auto')` | AC-4.1 (JSON ∈ {numba,jax}) |
| **T05** | Testes `tests/test_cli_backend_auto.py`: AC-1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2, 4.1. | todos AC | `tests/test_cli_backend_auto.py` | T03,T04 | `test(cli): cobre --backend auto + deprecação + reporting` | pytest verde |
| **T06** | Docs D5/D7: docstrings de `_add_common_backend_args`, `resolve_backend_preflight`; nota no help; atualizar `cli/__init__.py` (menção a `auto`). | RNF-3 | `cli/_main.py`, `cli/_exec.py`, `cli/__init__.py` | — | `docs(cli): documenta --backend auto (D1-D14)` | hook PT-BR verde |
| **T07** | Verify: suíte `tests/test_cli_*.py` verde (compat default) + code-reviewer (Trilha E). Confirmar que CI sem `--backend` só emite warning (não falha). | RNF-2,5, GATE-V | — | T05,T06 | (gate, sem commit) | GATE-V verde |

## Ordem de Execução (waves)
1. **Wave 0** (bloqueante): T00 (investigação TLS — decide a rota do ramo auto)
2. **Wave 1** (serial): T01 → T02
3. **Wave 2** (paralela): T03, T04, T06
4. **Wave 3**: T05
5. **Wave 4** (gate): T07 → abre merge

## Regras de Commit
- 1 tarefa = 1 commit atômico; Conventional Commits + escopo `cli`.
- `ruff-format` → re-add + re-commit (two-step), nunca `--amend` (memória `commit-workflow-precommit`).
- Conda env `Geosteering_AI` no PATH antes de commitar.
- Footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

## Checklist de Pronto (GATE-T)
- [x] Toda tarefa mapeia a ≥1 AC/RNF
- [x] Toda tarefa de código é 1 commit atômico (T00/T07 são análise/gate, sem commit)
- [x] Grafo sem ciclo
- [x] Última tarefa é o gate de verify
- [x] T00 (risco TLS) precede o ramo `auto` (T03) — risco crítico endereçado primeiro

## Estimativa
~0,5 dia (1 produto, sem física, capacidade já existente na lib). Valida o ciclo SDD end-to-end.
