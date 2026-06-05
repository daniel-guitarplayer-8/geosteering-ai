---
Spec: NNNN-slug
Tarefas-de: plan.md
Status: planejado
Data: YYYY-MM-DD
---

# Tarefas NNNN — <Título>

> Cada tarefa: **ATÔMICA** (1 commit), **TESTÁVEL** (≥1 AC verificável), com ordem e
> dependências explícitas. Mapeia para RF/AC da spec e arquivos do plan §4.

## Grafo de Dependências
```
T01 ─► T02 ─► T04 ─► T06 (verify)
        └──► T03 ─────┘
T05 (independente, paraleliza com T02)
```

## Lista de Tarefas
| # | Tarefa | Cobre | Arquivos | Dep | Commit (mensagem) | Done quando |
|:--|:--|:--|:--|:--:|:--|:--|
| T01 | criar pacote com `__init__` (D8) | — | `ingest/__init__.py` | — | `feat(ingest): esqueleto do pacote` | importável, `__all__` ok |
| T02 | `LASReader.read` → 22-col | RF-1/AC-1.1 | `ingest/las.py` | T01 | `feat(ingest): LASReader 22-col` | AC-1.1 |
| T03 | teste `test_22col_shape` | AC-1.2 | `tests/test_ingest_las.py` | T02 | `test(ingest): shape 22-col` | pytest verde |
| T04 | wiring em `data/loading.py` | RF-2 | `data/loading.py` | T02 | `feat(data): aceita fonte LAS` | sem regressão |
| T05 | docstrings D5/D6 + mega-header D1 | RNF-3 | todos | — | `docs(ingest): D1-D14` | hook PT-BR verde |
| T06 | verify: suite + parity + reviewers | todos AC | — | T03,T04,T05 | (gate, sem commit) | GATE-V verde |

## Ordem de Execução (waves)
1. **Wave 1** (serial): T01
2. **Wave 2** (paralela): T02, T05
3. **Wave 3**: T03, T04
4. **Wave 4** (gate): T06 → abre merge

## Regras de Commit (alinhadas ao workflow do projeto)
- 1 tarefa = 1 commit atômico; mensagem **Conventional Commits** + escopo do módulo.
- Pré-commit roda hooks; se `ruff-format` mexer, **re-add + re-commit** (two-step) — NÃO usar
  `--amend` como fallback (memória `commit-workflow-precommit`).
- Conda env `Geosteering_AI` no PATH antes de commitar.
- Footer obrigatório nos commits gerados por IA:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

## Checklist de Pronto (GATE-T → libera Implement)
- [ ] Toda tarefa mapeia a ≥1 AC ou RNF
- [ ] Toda tarefa é 1 commit atômico
- [ ] Grafo de dependências sem ciclo
- [ ] Última tarefa é o gate de verify (GATE-V)
