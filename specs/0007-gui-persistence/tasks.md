---
Spec: 0007-gui-persistence
Tarefas-de: plan.md
Status: planejado
Data: 2026-06-05
---

# Tarefas 0007 — Persistência gui/ (atômica + SessionDocument)

## Grafo de Dependências
```
T01(atomic.py) ─► T02(session.py) ─────────────► T05(testes) ─► T06(verify)
T03(git mv plot_cache+snapshot) ─► T04(shims) ──┘
T01 ─► T03(snapshot usa atomic) ────────────────┘
```

## Lista de Tarefas
| # | Tarefa | Cobre | Arquivos | Dep | Done quando |
|:--|:--|:--|:--|:--:|:--|
| **T01** | `atomic_write_text` (tmp mesmo-dir + fsync + os.replace; limpa tmp em falha) | RF-1/AC-1.x | `gui/persistence/{__init__,atomic}.py` | — | grava + crash-safe |
| **T02** | `SessionDocument` (dataclass pura; to/from_json forward-compat; save/load atômico; sem pickle) | RF-2/AC-2.x | `gui/persistence/session.py` | T01 | round-trip + forward-compat |
| **T03** | `git mv` sm_plot_cache→plot_cache, sm_snapshot_persist→snapshot; fix import qt_compat; snapshot usa `atomic_write_text` | RF-3/RF-4/AC-3.3 | `gui/persistence/{plot_cache,snapshot}.py` | T01 | imports válidos; snapshot atômico |
| **T04** | Shims (módulo) retrocompat | RF-3/RF-4/AC-3.2 | `simulation/tests/sm_{plot_cache,snapshot_persist}.py` | T03 | identidade + legados ok |
| **T05** | Suíte `tests/test_gui_persistence.py` | RF-6/AC-1.x/2.x/3.x/5.x | `tests/test_gui_persistence.py` | T02,T04 | pytest verde |
| **T06** | Verify: ruff/mypy + suíte 0007 + regressão (lru_cache + GUI SM) + fronteira | AC-3.4/5.x | — | T05 | GATE-V verde |

## Ordem de Execução (waves)
1. **Wave 1**: T01 (atomic — base de tudo)
2. **Wave 2** (paralela): T02 (session), T03 (mv + hardening)
3. **Wave 3**: T04 (shims)
4. **Wave 4**: T05 (testes)
5. **Wave 5** (gate): T06

## Regras de Commit
- **Sem commitar** até revisão do diff (instrução vigente). Quando autorizado: commits atômicos, escopo `gui`.
- Pre-commit: pré-aplicar auto-fixers + full-stage dos arquivos movidos (evita loop de stash); conda env no PATH; footer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Checklist de Pronto (GATE-T)
- [x] Toda tarefa mapeia a ≥1 AC
- [x] Grafo sem ciclo
- [x] Última tarefa é o gate de verify (inclui regressão lru_cache + GUI SM + fronteira)
- [x] Hardening (atômico) tem teste de crash-resistance (AC-1.2)

## Estimativa
~0,5 dia (2 moves + 2 módulos novos pequenos + shims + testes; hardening é o foco técnico).
