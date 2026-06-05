---
Spec: 0006-gui-plot-backends
Tarefas-de: plan.md
Status: planejado
Data: 2026-06-05
---

# Tarefas 0006 — Extração dos backends de plotagem

## Grafo de Dependências
```
T01(git mv pacote) ─► T02(fix imports/headers) ─► T03(helper qt_compat) ─► T04(shim) ─► T05(testes) ─► T06(verify)
                                                          └────────────────────────────┘
```

## Lista de Tarefas
| # | Tarefa | Cobre | Arquivos | Dep | Done quando |
|:--|:--|:--|:--|:--:|:--|
| **T01** | `git mv simulation/tests/sm_plot_backends/` → `geosteering_ai/gui/plot_backends/` (6 arquivos) | RF-1 | pacote movido | — | dir em `gui/`; `git status` mostra rename |
| **T02** | Atualizar headers D1 (caminho) + fix `pyqtgraph_canvas` (`..sm_qt_compat`→`gui.qt_compat`) | RF-4/AC-3.3, RNF-5 | `gui/plot_backends/*.py` | T01 | imports válidos; `import gui.plot_backends` ok |
| **T03** | Helper `load_qwebengineview()` em `qt_compat` + fix `plotly_canvas` (PyQt6→helper) | RF-3/AC-3.1/3.2 | `gui/qt_compat.py`, `gui/plot_backends/plotly_canvas.py` | T01 | sem `from PyQt6.QtWebEngineWidgets`; helper retorna classe |
| **T04** | Shim-pacote retrocompat em `simulation/tests/sm_plot_backends/` (`__init__` + 5 submódulos re-export) | RF-2/AC-2.x | `simulation/tests/sm_plot_backends/*.py` | T02,T03 | imports legados + submódulo + identidade ok |
| **T05** | Suíte `tests/test_gui_plot_backends.py` (API, smoke offscreen, shim, fronteira, guard plotly) | RF-6/AC-1.x/2.x/3.x/5.x | `tests/test_gui_plot_backends.py` | T04 | pytest verde (skip vispy) |
| **T06** | Verify: ruff/mypy + suíte 0006 + regressão GUI SM 16/16 + fronteira `import gui.plot_backends` sem Qt/mpl | AC-2.4/5.1/5.2 | — | T05 | GATE-V verde |

## Ordem de Execução (waves)
1. **Wave 1**: T01 (move atômico)
2. **Wave 2** (paralela): T02, T03 (imports/headers ⊥ helper+plotly)
3. **Wave 3**: T04 (shim — depende dos imports finais)
4. **Wave 4**: T05 (testes)
5. **Wave 5** (gate): T06

## Regras de Commit
- **Sem commitar** até revisão do diff pelo usuário (instrução vigente). Quando autorizado: 1 tarefa = 1 commit atômico, escopo `gui`.
- Pre-commit: pré-aplicar auto-fixers (eof/whitespace/ruff-format) antes de `git add` (evita o loop de stash); conda env `Geosteering_AI` no PATH; footer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Checklist de Pronto (GATE-T)
- [x] Toda tarefa mapeia a ≥1 AC
- [x] Grafo sem ciclo
- [x] Última tarefa é o gate de verify (inclui regressão SM + fronteira)
- [x] Riscos do `git mv` (rename/submódulo) endereçados (T01/T04)

## Estimativa
~0,5 dia (extração + 2 fixes pontuais + shim + testes; risco baixo — código já maduro/testado indiretamente).
