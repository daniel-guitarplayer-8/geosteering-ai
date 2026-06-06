---
Spec: 0011d-sm-results
Tarefas-de: spec.md
Status: planejado
Data: 2026-06-06
---

# Tarefas 0011d — ResultsView (galeria + cache + .session)

## Constituição (gate)
Sem física nova (só re-exibe H6). VM PURO (Princípio X). DRY: reusa LRUPlotCache/SessionDocument/
add_subplot_grid. .session JSON sem pickle. Monólito intocado. Sem ADR.

## Contratos (assinaturas)
```python
# apps/.../results_viewmodel.py  (PURO, sem Qt)
COMPONENT_NAMES = ("Hxx","Hxy","Hxz","Hyx","Hyy","Hyz","Hzx","Hzy","Hzz")  # axis −1 do H6
PLOT_KINDS = ("re","im","mag","phase")
class ResultsViewModel(BaseViewModel):
    def set_result(result: dict) -> None        # clampa seletores; emite changed + results_changed
    @property component_index/plot_kind/tr_index/dip_index/freq_index/page (+ setters via _set, clamp)
    @property has_result/dims/n_pages/depth/component_name -> ...
    def page_models() -> list[int]              # índices de modelo da página atual
    def curve_for(model_index) -> np.ndarray    # transform(H6[m,tr,dip,:,freq,comp]); LRU cache
    results_changed: VMSignal

# apps/.../results_view.py  (Qt)
class ResultsView(QWidget):                     # galeria add_subplot_grid + seletores + paginação
    def __init__(vm: ResultsViewModel, parent=None)

# viewmodel.py: SimulationViewModel.__init__ cria self.results=ResultsViewModel();
#   _on_sim_finished → self.results.set_result(result) (mantém result_ready)
#   to_session_dict()/load_session_dict(d) p/ .session (params do SimRequest)
# view.py: embute ResultsView; botões Salvar/Abrir .session (SessionDocument)
```

## Lista
| # | Tarefa | Cobre | Arquivos | Dep |
|:--|:--|:--|:--|:--:|
| **T01** | `ResultsViewModel` PURO (seletores+clamp, curve_for Re/Im/Mag/Phase, paginação, LRU cache) | RF-1/AC-1..4 | `apps/.../results_viewmodel.py` | — |
| **T02** | `ResultsView` (galeria grade + barra seletores + paginação) | RF-2/AC-5 | `apps/.../results_view.py` | T01 |
| **T03** | Wire: SimulationVM compõe `results` + feed; SimulatorView embute galeria; .session save/load | RF-3/RF-4/AC-6 | `apps/.../viewmodel.py`, `view.py` | T02 |
| **T04** | Testes (VM puro: curvas/clamp/paginação/cache; galeria gui; session roundtrip; fronteira) + GATE-V + revisão + commit | RF-5/AC-1..7 | `tests/test_sm_results.py` | T03 |

## Commit
Sem commitar até revisão; pré-aplicar auto-fixers (ruff-format two-step); footer Co-Authored-By.

## GATE-T
- [x] Toda tarefa → ≥1 AC; última = verify+review; curvas (AC-1) e cache (AC-4) e .session (AC-6) testados;
  VM puro (AC-7); galeria gui (AC-5).
