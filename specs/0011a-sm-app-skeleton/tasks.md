---
Spec: 0011a-sm-app-skeleton
Tarefas-de: plan.md
Status: planejado
Data: 2026-06-05
---

# Tarefas 0011a — Walking skeleton

## Grafo
```
T01(gui/threading) ─► T02(gui/services base+SimulationService) ─► T04(View) ─► T05(testes) ─► T06(verify)
T03(SimulationVM puro) ───────────────────────────────────────┘
T04 ─► T04b(perspectiva + main_window + app)
```

## Lista
| # | Tarefa | Cobre | Arquivos | Dep |
|:--|:--|:--|:--|:--:|
| **T01** | `Worker`+`WorkerSignals`+`run_in_thread` | RF-1/AC-3 | `gui/threading/{__init__,worker}.py` | — |
| **T02** | `BaseService` + `SimulationService` + `SimRequest` (`_build_batch`/`_run_simulation`→`simulate_batch`) | RF-2/RF-3/AC-2/AC-5 | `gui/services/{__init__,base,simulation_service}.py` | T01 |
| **T03** | `SimulationViewModel` PURO (params+validate+run; service injetado) | RF-4/AC-1 | `apps/sim_manager/perspectives/simulation/viewmodel.py` | — |
| **T04** | `SimulatorView` (QWidget: inputs+Run+status+PlotCanvas; binding) | RF-5/AC-4 | `…/simulation/view.py` | T02,T03 |
| **T04b** | `SimulationPerspective` + `SM_MainWindow` + `app.py` (+ `apps/__init__`) | RF-6/AC-4 | `apps/sim_manager/*` | T04 |
| **T05** | Suíte `tests/test_sim_app_skeleton.py` | RF-7/AC-1..6 | `tests/test_sim_app_skeleton.py` | T04b |
| **T06** | Verify: ruff/mypy + suíte + regressão SM 16/16 + fronteira | AC-6 | — | T05 |

## Ordem (waves)
1. T01 → 2. T02 ∥ T03 → 3. T04 → 4. T04b → 5. T05 → 6. T06

## Commit
- **Sem commitar** até revisão do diff. Pre-commit: pré-aplicar auto-fixers; conda env no PATH; footer Co-Authored-By.

## GATE-T
- [x] Toda tarefa → ≥1 AC; grafo sem ciclo; última = verify (regressão SM + fronteira); fidelidade via `simulate_batch` real (AC-2).

## Estimativa
~1 sessão (skeleton mínimo; threading é o ponto técnico).
