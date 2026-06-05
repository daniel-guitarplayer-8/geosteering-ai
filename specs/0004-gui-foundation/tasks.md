---
Spec: 0004-gui-foundation
Tarefas-de: plan.md
Status: planejado
Data: 2026-06-05
---

# Tarefas 0004 — Fundação `geosteering_ai/gui/`

> Cada tarefa: ATÔMICA (1 commit), TESTÁVEL, com ordem/dependências.

## Grafo de Dependências
```
T01 ─► T02 ─► T03 ─► T05 ─► T06 (verify)
                └► T04 ──────┘
```

## Lista de Tarefas
| # | Tarefa | Cobre | Arquivos | Dep | Commit | Done quando |
|:--|:--|:--|:--|:--:|:--|:--|
| **T01** | Criar pacote `gui/` (`__init__.py` D1) | RF-1/AC-1.2 | `geosteering_ai/gui/__init__.py` | — | `feat(gui): cria pacote geosteering_ai/gui (fundação Qt)` | pacote importável |
| **T02** | `git mv sm_qt_compat.py → gui/qt_compat.py` + atualizar mega-header (lar canônico) | RF-1/AC-1.1,1.3 | `gui/qt_compat.py` | T01 | `feat(gui): reloca qt_compat p/ gui/ (git mv, Strangler Fig)` | `from geosteering_ai.gui.qt_compat import QtCore` ok |
| **T03** | Shim em `simulation/tests/sm_qt_compat.py` (re-export explícito dos 16 + `__all__`) | RF-2/RF-3/AC-2.1,2.2 | `simulation/tests/sm_qt_compat.py` | T02 | `feat(gui): sm_qt_compat vira shim de retrocompat (re-exporta gui.qt_compat)` | identidade `is` preservada |
| **T04** | Extra `[gui]` (PyQt6) em `pyproject.toml` | RF-4/AC-4.1 | `pyproject.toml` | T01 | `build(gui): extra pip [gui] (PyQt6)` | extra presente |
| **T05** | Testes `tests/test_gui_foundation.py` (AC-1.x, AC-2.1, AC-2.2) | AC-1.x/2.1/2.2 | `tests/test_gui_foundation.py` | T03 | `test(gui): fundação gui/ + retrocompat sm_qt_compat (identidade)` | pytest verde |
| **T06** | Verify: GUI suite (xvfb) + smoke import SM + ruff/mypy + reviewers | AC-2.3/2.4 | — | T05,T04 | (gate) | GATE-V verde |

## Ordem de Execução (waves)
1. **Wave 1**: T01
2. **Wave 2** (paralela): T02 → (T03), T04
3. **Wave 3**: T05
4. **Wave 4** (gate): T06

## Regras de Commit
- 1 tarefa = 1 commit atômico; Conventional Commits + escopo `gui`.
- `ruff-format` → re-add + re-commit (two-step); nunca `--amend` (memória `commit-workflow-precommit`).
- Conda env `Geosteering_AI` no PATH antes de commitar.
- Footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

## Checklist de Pronto (GATE-T)
- [x] Toda tarefa mapeia a ≥1 AC
- [x] Toda tarefa de código é 1 commit atômico
- [x] Grafo sem ciclo
- [x] Última tarefa é o gate de verify (inclui GUI suite sob xvfb — AC-2.3)

## Estimativa
~0,5 dia (relocação mecânica + shim + testes). Risco controlado (precedente `sm_io.py`).
