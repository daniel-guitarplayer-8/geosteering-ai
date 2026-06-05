---
Spec: 0005-gui-mvvm-base
Tarefas-de: plan.md
Status: planejado
Data: 2026-06-05
---

# Tarefas 0005 — Base MVVM

## Grafo de Dependências
```
T01(VMSignal) ─► T02(BaseViewModel) ─► T05(testes puros) ─► T07(verify)
T03(AppContext) ─► T04(Perspective ABC) ─────────────────┘
T06(MainWindowBase) ─► T07
```

## Lista de Tarefas
| # | Tarefa | Cobre | Arquivos | Dep | Commit | Done quando |
|:--|:--|:--|:--|:--:|:--|:--|
| **T01** | `VMSignal` pub/sub puro (connect/disconnect/emit/clear; isola exceções+loga) | RF-1/AC-1.x | `gui/viewmodels/{__init__,signal}.py` | — | `feat(gui): VMSignal pub/sub puro-Python (MVVM)` | AC-1.1..1.5 |
| **T02** | `BaseViewModel` puro (changed VMSignal; `_set`; to_dict/from_dict) | RF-2/AC-2.x | `gui/viewmodels/base.py` | T01 | `feat(gui): BaseViewModel puro (estado observável + serializável)` | AC-2.x |
| **T03** | `AppContext` mínimo (dataclass extensível) | RF-4 | `gui/shell/{__init__,context}.py` | — | `feat(gui): AppContext (contexto de app)` | importável |
| **T04** | `Perspective` ABC (sem Qt runtime; TYPE_CHECKING p/ QWidget) | RF-3/AC-3.x | `gui/shell/perspective.py` | T02,T03 | `feat(gui): Perspective ABC (contrato de plugin MVVM)` | AC-3.x |
| **T05** | Testes puros (VMSignal/BaseViewModel/Perspective) | AC-1.x/2.x/3.x | `tests/test_gui_mvvm_base.py` | T04 | `test(gui): base MVVM pura (sem pytest-qt)` | pytest verde |
| **T06** | `MainWindowBase`(QMainWindow): add_perspective + statusbar + tema | RF-5/AC-5.x | `gui/shell/main_window_base.py` | T03,T04 | `feat(gui): MainWindowBase (host de perspectivas, Qt)` | AC-5.1/5.2 (xvfb) |
| **T07** | Verify: testes puros + xvfb(AC-5) + GUI suite 16/16 + ruff/mypy + reviewers | AC-5.3 | — | T05,T06 | (gate) | GATE-V verde |

## Ordem de Execução (waves)
1. **Wave 1** (paralela): T01, T03
2. **Wave 2** (paralela): T02, T06(depende T03; pode após T04) — serializa T04 antes de T06
3. **Wave 3**: T04 → T06
4. **Wave 4**: T05
5. **Wave 5** (gate): T07

## Regras de Commit
- 1 tarefa = 1 commit atômico; Conventional Commits + escopo `gui`.
- `ruff-format` → re-add + re-commit (two-step); nunca `--amend`.
- Conda env `Geosteering_AI` no PATH; footer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Checklist de Pronto (GATE-T)
- [x] Toda tarefa mapeia a ≥1 AC
- [x] 1 commit atômico por tarefa de código
- [x] Grafo sem ciclo
- [x] Última tarefa é o gate de verify (inclui GUI suite + import-boundary)

## Estimativa
~0,5–1 dia (abstrações novas pequenas, bem-testáveis; risco baixo).
