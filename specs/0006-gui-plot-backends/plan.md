---
Spec: 0006-gui-plot-backends
Plano-de: spec.md
Status: planejado
Data: 2026-06-05
---

# Plano 0006 — Extração dos backends de plotagem (HOW)

> Pré-requisito: `spec.md` passou GATE-S.

## 1. Gate de Constituição

| Princípio | Aplicável? | Viola? | Como o plano cumpre |
|:--|:--:|:--:|:--|
| I Paridade (3 regimes) | não | não | plotagem não toca cálculo EM (`Converge-Em: n/a`) |
| II Errata imutável | não | não | — |
| III TF/Keras exclusivo | sim | não | sem torch (canvas só plotam) |
| IV Python 3.13 | sim | não | — |
| V Config-parâmetro | sim | não | canvas recebem `style`/`figsize` por parâmetro; sem `globals()` |
| VI Logging | sim | não | sem `print()` (já conforme no código existente) |
| VII D1–D14 | sim | não | mega-headers atualizados ao novo caminho; docstrings preservadas |
| VIII PT-BR | sim | não | — |
| IX SSoT | sim | não | `Backlog-Code: F-mvc-split` |
| X MVVM | sim | não | canvas desacopla plotagem da View; ABC sem Qt |
| XI Fundação | sim | não | `gui/plot_backends` compartilhado SM+Studio; `core` não importa `gui` |
| XII Gates | sim | não | reviewers (Trilha F) + GUI suite xvfb + fronteira |

**GATE-P: sem violação.** Não gera ADR (extração reversível, padrão Strangler-Fig já ratificado em 0004).

## 2. Arquitetura Técnica (antes → depois)

```
ANTES                                          DEPOIS (0006)
simulation/tests/sm_plot_backends/   ──move──▶ geosteering_ai/gui/plot_backends/
  base.py  (PlotCanvas ABC+factory)              base.py            (idêntico)
  mpl_canvas.py                                  mpl_canvas.py      (idêntico)
  pyqtgraph_canvas.py  ──fix import──▶           pyqtgraph_canvas.py (qt_compat)
  plotly_canvas.py     ──fix bug PyQt6──▶        plotly_canvas.py   (qt_compat helper)
  vispy_canvas.py                                vispy_canvas.py    (idêntico)
  __init__.py                                    __init__.py        (idêntico)

simulation/tests/sm_plot_backends/  ◄──SHIM──── re-exporta de gui.plot_backends
  __init__.py + base/mpl/pyqtgraph/plotly/vispy   (cada um: from gui.plot_backends.X import *)
  → monólito (`from .sm_plot_backends[...]`) intacto; migração real = 0011
```

Encaixe MVVM: a **View** cria o canvas via `make_canvas(backend)`; o **ViewModel** puro chama
`canvas.plot_line()/draw()` sem importar Qt. `gui/plot_backends/base.py` **não importa Qt** —
cada backend importa suas deps (matplotlib/pyqtgraph/plotly/vispy + Qt) só ao instanciar (lazy).

## 3. Contratos / APIs (o que muda)

```python
# geosteering_ai/gui/qt_compat.py  — NOVO helper (única fonte de binding)
def load_qwebengineview() -> Any:
    """Retorna a classe QWebEngineView do binding Qt ativo (PyQt6/PySide6).
    QtWebEngine é módulo Qt PESADO e OPCIONAL — resolvido sob demanda (não no import).
    Raises ImportError se ausente ou sem binding."""
    # ramifica por QT_BINDING ∈ {"PyQt6","PySide6"}; importa .QtWebEngineWidgets

# geosteering_ai/gui/plot_backends/  — API PRESERVADA (move + 2 imports corrigidos)
class PlotCanvas(ABC): ...            # widget/clear/draw/save/add_subplot_grid/plot_line/
                                      #   add_hline/set_axis_config/set_dark_mode
def make_canvas(backend, parent=None, figsize=(14,9), style=None) -> PlotCanvas: ...
def available_backends() -> list[PlotBackend]: ...   # MATPLOTLIB sempre; outros se instalados

# plotly_canvas.py  ── ANTES (bug):  from PyQt6.QtWebEngineWidgets import QWebEngineView
#                   ── DEPOIS:       from geosteering_ai.gui.qt_compat import load_qwebengineview
#                                    QWebEngineView = load_qwebengineview()
# pyqtgraph_canvas.py ── ANTES: from ..sm_qt_compat import QtCore, QtGui
#                     ── DEPOIS: from geosteering_ai.gui.qt_compat import QtCore, QtGui
```

## 4. Estrutura de Arquivos

| Arquivo | Ação | Conteúdo |
|:--|:--|:--|
| `gui/plot_backends/{__init__,base,mpl_canvas,pyqtgraph_canvas,plotly_canvas,vispy_canvas}.py` | `git mv` | pacote movido (1.242 LOC) |
| `gui/plot_backends/plotly_canvas.py` | editar | usar `load_qwebengineview()` + header novo |
| `gui/plot_backends/pyqtgraph_canvas.py` | editar | import `gui.qt_compat` + header novo |
| `gui/plot_backends/*.py` (headers) | editar | D1 mega-header atualizado ao caminho `gui/plot_backends/` |
| `gui/qt_compat.py` | editar | + `load_qwebengineview()` |
| `simulation/tests/sm_plot_backends/__init__.py` | criar (shim) | re-export de `gui.plot_backends` |
| `simulation/tests/sm_plot_backends/{base,mpl_canvas,pyqtgraph_canvas,plotly_canvas,vispy_canvas}.py` | criar (shim) | re-export do submódulo gui correspondente |
| `tests/test_gui_plot_backends.py` | criar | AC-1.x/2.x/3.x/5.x |

## 5. Decisões de Design

| Decisão | Opções | Escolha | Justificativa | ADR? |
|:--|:--|:--|:--|:--:|
| Quantos backends mover | ABC+mpl vs **4** | 4 | pacote atômico; factory referencia os 4; decisão do usuário | não |
| Tipo de shim | módulo plano vs **pacote** | pacote | monólito acessa submódulo `pyqtgraph_canvas` (linha 10169) | não |
| Fix QWebEngine | branch inline no plotly vs **helper em qt_compat** | helper | qt_compat = única fonte de decisão de binding (DRY) | não |
| Extrair `PlotStyle` agora | sim vs **defer 0011** | defer | exige tocar `sm_plots.py` (1.772 LOC); `set_dark_mode` já existe por canvas | não |

## 6. Riscos Técnicos e Mitigações

| Risco | Prob. | Impacto | Mitigação |
|:--|:--:|:--:|:--|
| quebrar `from .sm_plot_backends.pyqtgraph_canvas import …` (monólito) | média | alto | shim-pacote com submódulo re-export; AC-2.3 + AC-2.4 (suíte SM) |
| `git mv` perder histórico/rename | baixa | baixo | `git mv` preserva; verificar `git status` antes do commit |
| fix plotly mudar comportamento | baixa | médio | só troca a ORIGEM do `QWebEngineView`; AC-3.1 guard + smoke real (PyQt6+WebEngine no env) |
| `vispy` ausente derrubar testes | média | baixo | testes de vispy com `pytest.importorskip`/skip-graceful (RNF-4) |
| ABC puxar Qt/matplotlib no import | baixa | médio | base.py só stdlib+numpy; AC-5.1 subprocess valida |

## 7. Estratégia de Teste

| Camada | O quê | Onde |
|:--|:--|:--|
| API (puro) | imports + `PlotCanvas` ABC + `available_backends()` | `tests/test_gui_plot_backends.py` (AC-1.1/1.2/1.3) |
| smoke (xvfb) | `make_canvas(MATPLOTLIB)` + grid+line+draw+dark | idem (AC-1.4) |
| smoke (xvfb) | `make_canvas(PYQTGRAPH)` (pyqtgraph no env) | idem — skip se ausente |
| fix (puro) | guard: plotly sem `from PyQt6.QtWebEngineWidgets`; `load_qwebengineview()` ok | idem (AC-3.1/3.2) |
| shim | imports legados + identidade de objeto + submódulo | idem (AC-2.1/2.2/2.3) |
| fronteira (subprocess) | `import gui.plot_backends` sem Qt/mpl/backends em `sys.modules` | idem (AC-5.1) |
| regressão | suíte GUI do SM 16/16 | `tests/test_simulation_manager_gui.py` (AC-2.4) |

## 8. Critério de Pronto do Plano (GATE-P)
- [x] Constituição sem violação
- [x] Contratos com assinaturas exatas (helper + imports corrigidos)
- [x] Nenhum ADR bloqueante (Strangler-Fig já ratificado)
- [x] Estratégia de teste cobre AC-1.x..AC-5.x + fronteira + regressão SM
