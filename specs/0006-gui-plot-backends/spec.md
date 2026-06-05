---
Spec: 0006-gui-plot-backends
Titulo: Extração dos backends de plotagem para gui/plot_backends/ (PlotCanvas ABC + 4 backends, Strangler Fig)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: n/a   # abstração de plotagem; não toca cálculo EM
Status: planejado
Released-As:
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-05
---

# Spec 0006 — `gui/plot_backends` (PlotCanvas ABC + 4 backends)

## 1. Contexto e Problema

A spec 0004 estabeleceu `geosteering_ai/gui/` (com `qt_compat`); a 0005 a base MVVM. Falta
**desacoplar a plotagem da View**: o Simulation Manager (spec 0011) e o Studio (0013) precisam
de UMA abstração de canvas compartilhada, em vez de cada app falar com matplotlib direto.

**Descoberta-chave:** a abstração **já existe e está bem-formada** — `PlotCanvas` ABC +
factory + **4 backends concretos** (1.242 LOC) vivem hoje, por acidente histórico, em
`geosteering_ai/simulation/tests/sm_plot_backends/`. Esta spec é uma **extração Strangler-Fig**
(igual à 0004 com `qt_compat`): mover para um pacote de 1ª classe, **corrigir 1 bug de
portabilidade**, e deixar um shim de retrocompat — **não** é desenvolvimento do zero.

| Estado | Onde | Evidência |
|:--|:--|:--|
| pronto (a mover) | `simulation/tests/sm_plot_backends/` | `base.py` (ABC+factory) + 4 canvas, 1.242 LOC |
| 🐛 a corrigir | `…/sm_plot_backends/plotly_canvas.py:54,60` | `from PyQt6.QtWebEngineWidgets import …` **hardcoded** (quebra PySide6) |
| a fixar (relativo) | `…/sm_plot_backends/pyqtgraph_canvas.py:32` | `from ..sm_qt_compat import …` (inválido após o move) |
| ausente | `geosteering_ai/gui/plot_backends/` | diretório não existe |

## 2. User Stories

| ID | Como… | Quero… | Para… | Prioridade |
|:--|:--|:--|:--|:--:|
| US-1 | Dev de app (SM/Studio) | obter um canvas via `make_canvas(backend)` em `gui/` | plotar sem acoplar a View a matplotlib/pyqtgraph | Must |
| US-2 | Dev de ViewModel | chamar `canvas.plot_line()/draw()` sem importar Qt | manter o VM puro/testável (Princípio X) | Must |
| US-3 | Mantenedor do SM | imports legados (`from .sm_plot_backends import …`) intactos | não quebrar o monólito (migração é 0011) | Must |
| US-4 | Dev em PySide6 | o backend Plotly funcionar no binding ativo | portabilidade multi-binding (não só PyQt6) | Should |

## 3. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | Mover `sm_plot_backends/` → `geosteering_ai/gui/plot_backends/` (PlotCanvas ABC, `AxisConfig`, `SubplotHandle`, `PlotBackend`, `make_canvas`, `available_backends`, 4 canvas) preservando API e comportamento | Must | MOVE (`git mv`) |
| RF-2 | Shim de retrocompat **pacote** em `simulation/tests/sm_plot_backends/` re-exportando de `gui.plot_backends`, **incluindo o submódulo `pyqtgraph_canvas`** (acesso direto no monólito:10169) | Must | NOVO (shim) |
| RF-3 | Corrigir `plotly_canvas` (PyQt6 hardcoded) → `qt_compat`; adicionar helper `load_qwebengineview()` em `gui/qt_compat.py` (resolve `QWebEngineView` do binding ativo) | Must | FIX |
| RF-4 | Ajustar `pyqtgraph_canvas` import `..sm_qt_compat` → `gui.qt_compat` (após o move) | Must | FIX |
| RF-5 | Importar `geosteering_ai.gui.plot_backends` **não** importa Qt/matplotlib/pyqtgraph/plotly/vispy (lazy) | Must | INVARIANTE |
| RF-6 | Suíte nova `tests/test_gui_plot_backends.py` (API, fronteira, shim, smoke offscreen, guard do fix plotly) | Must | NOVO |

### RF-1 — Critérios de Aceite (extração)
- [ ] **AC-1.1**: `from geosteering_ai.gui.plot_backends import PlotCanvas, PlotBackend, make_canvas, available_backends, AxisConfig, SubplotHandle` funciona.
- [ ] **AC-1.2**: `PlotCanvas` é ABC — instanciá-la diretamente levanta `TypeError`.
- [ ] **AC-1.3**: `available_backends()` SEMPRE inclui `MATPLOTLIB`; inclui `PYQTGRAPH`/`PLOTLY`/`VISPY` apenas se instalados.
- [ ] **AC-1.4** (smoke, offscreen): `make_canvas(PlotBackend.MATPLOTLIB)` retorna canvas com `widget()`; `add_subplot_grid(1,1)` + `plot_line` + `draw()` + `set_dark_mode(True)` sem erro.

### RF-2 — Critérios de Aceite (shim / zero-regressão do SM)
- [ ] **AC-2.1**: `from geosteering_ai.simulation.tests.sm_plot_backends import make_canvas, PlotCanvas, PlotBackend, available_backends` funciona.
- [ ] **AC-2.2**: identidade de objeto — `sm_plot_backends.make_canvas is gui.plot_backends.make_canvas` (shim re-exporta, não redefine).
- [ ] **AC-2.3**: submódulo — `from geosteering_ai.simulation.tests.sm_plot_backends.pyqtgraph_canvas import PyQtGraphCanvas` funciona e é o **mesmo** objeto de `gui.plot_backends.pyqtgraph_canvas`.
- [ ] **AC-2.4**: a suíte GUI do SM (`tests/test_simulation_manager_gui.py`) continua 16/16 (sem regressão).

### RF-3/RF-4 — Critérios de Aceite (fix de portabilidade)
- [ ] **AC-3.1**: `gui/plot_backends/plotly_canvas.py` **não** contém `from PyQt6.QtWebEngineWidgets` (guard de regressão do bug).
- [ ] **AC-3.2**: `load_qwebengineview()` em `qt_compat` retorna a classe `QWebEngineView` do binding ativo, ou levanta `ImportError` claro se ausente.
- [ ] **AC-3.3**: `gui/plot_backends/pyqtgraph_canvas.py` importa Qt via `geosteering_ai.gui.qt_compat` (não `sm_qt_compat`, não PyQt6 direto).

### RF-5 — Critérios de Aceite (fronteira/lazy)
- [ ] **AC-5.1** (subprocess): após `import geosteering_ai.gui.plot_backends`, **nenhum** de `{PyQt6, PySide6, matplotlib, pyqtgraph, plotly, vispy}` está em `sys.modules`.
- [ ] **AC-5.2**: `core` (simulation/models/…) não importa `gui` (mantém o hook de fronteira 0004/0005).

## 4. Requisitos Não-Funcionais (RNF)

| ID | Categoria | Requisito | Métrica/Limite |
|:--|:--|:--|:--|
| RNF-1 | Paridade física | **N/A** — plotagem não toca cálculo EM (paridade Fortran <1e-12 intocada) | declarado |
| RNF-2 | Comportamento | extração bit-comportamental: a lógica dos canvas não muda (só localização + 2 imports) | diff só move/import |
| RNF-3 | Fronteira/lazy | ABC importável sem nenhum backend; cada backend importa deps só ao instanciar | AC-5.1 |
| RNF-4 | Deps opcionais | `pyqtgraph`/`plotly`/`vispy` opcionais (`[gui-plot]`); testes **skipam** quando ausentes (ex.: vispy) | skip-graceful |
| RNF-5 | Doc | D1–D14 (headers atualizados ao novo caminho, docstrings preservadas) | conformes |

## 5. Escopo

### IN
- `git mv simulation/tests/sm_plot_backends/` → `geosteering_ai/gui/plot_backends/` (6 arquivos).
- Fix `plotly_canvas` (PyQt6→qt_compat) + helper `load_qwebengineview()` em `qt_compat`.
- Fix `pyqtgraph_canvas` import (`..sm_qt_compat`→`gui.qt_compat`).
- Shim **pacote** de retrocompat em `simulation/tests/sm_plot_backends/` (+ submódulos re-export).
- `tests/test_gui_plot_backends.py`.

### OUT (próximas specs)
- **Refator de `sm_plots.py`** (1.772 LOC) para chamar `canvas.add_subplot_grid()` nas ~30 funções `plot_*` → migração do SM (**0011** / v2.6c). `EMCanvas` legado e `plot_*` permanecem.
- **Extração de `PlotStyle`/`apply_style`** para `gui/plotting/style.py` (tokens de tema desacoplados): o INDEX cita "tokens de tema", mas cada canvas já provê `set_dark_mode(dark)`; mover `PlotStyle` exige tocar `sm_plots.py` (escopo 0011). **Diferença consciente vs INDEX** — registrada aqui.
- Remoção do shim → quando o SM migrar (**0011**).

## 6. [NEEDS CLARIFICATION]
- [x] ~~Mover 4 backends ou só ABC+Matplotlib?~~ → **RESOLVIDO** (decisão do usuário): mover os **4** (pacote atômico/lazy; a factory `make_canvas` referencia os 4; meia-extração quebraria a factory).
- [x] ~~Shim flat-module ou pacote?~~ → **RESOLVIDO**: **pacote** — o monólito acessa o submódulo `sm_plot_backends.pyqtgraph_canvas` (linha 10169); um módulo plano não satisfaria isso.
- [x] ~~Como corrigir o `QWebEngineView` de forma multi-binding?~~ → **RESOLVIDO**: helper `load_qwebengineview()` em `qt_compat` (única fonte de decisão de binding), ramificando por `QT_BINDING`.

**GATE-S: PASSOU** — 0 marcadores abertos.

## 7. Dependências e Riscos de Escopo

| Tipo | Item | Impacto |
|:--|:--|:--|
| Dep | spec 0004 (`gui/qt_compat`) | canvases reusam Qt via `qt_compat`; helper novo lá |
| Dep | spec 0005 (gui/ base) | mesma fronteira de import / padrão de teste offscreen |
| Risco | quebrar import do monólito (submódulo) | mitigado: shim-pacote com re-export de submódulos; AC-2.3/2.4 |
| Risco | `vispy` ausente no ambiente | mitigado: backends opcionais skip-graceful (RNF-4) |
| Risco | fix do plotly alterar comportamento | baixo: só troca a ORIGEM do `QWebEngineView`; AC-3.1/3.2 |

## 8. Critério de Pronto da Spec (GATE-S)
- [x] 0 marcadores `[NEEDS CLARIFICATION]`
- [x] Todo RF tem ≥1 AC testável
- [x] Escopo IN/OUT explícito (refator `sm_plots.py` e `PlotStyle` sequenciados p/ 0011)
- [x] `Produtos` e `Converge-Em` declarados
- [x] `Backlog-Code` (`F-mvc-split`) no ROADMAP §0
- [x] Nenhum princípio da CONSTITUTION violado (plotagem desacoplada; fronteira preservada)
