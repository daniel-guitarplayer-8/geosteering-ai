# Investigação — SM MVVM × Monólito, Design de GUI, Studio, SDD e Premortem

| Atributo | Valor |
|:--|:--|
| **Data** | 2026-06-06 |
| **Autor** | Daniel Leal (investigação conduzida por Claude Code, 5 agentes paralelos) |
| **Tipo** | Relatório de investigação/decisão (não segue o template de versão; gerado a pedido explícito) |
| **Escopo** | Q1 acesso ao monólito · Q2 paridade SM MVVM × monólito · Q3 design de GUI · Q4 status do Studio · Q5 SDD · Q6 premortem |
| **Branches** | `feat/gui-*` (empilhadas, **não mergeadas à `main`**) |
| **Método** | Inventário read-only de ~17.9k LOC do monólito + `apps/sim_manager/` + `geosteering_ai/gui/` + `specs/` + git + pesquisa web |

---

## 0. Sumário Executivo

1. **Sim, temos o SM Monolítico** — `geosteering_ai/simulation/tests/simulation_manager.py` (**10.762 LOC**) + **14 módulos `sm_*.py`** (~7.1k LOC). Roda com `python -m geosteering_ai.simulation.tests.simulation_manager`. **Não preciso que você me envie nada** — o código está no repo e foi 100% inventariado.
2. **Sim, estamos portando o monólito → MVVM**, via **Strangler Fig** (incremental, nunca big-bang). Já foram entregues **5 fatias** (skeleton, params, geologia estocástica, galeria de resultados, JAX-GPU). Falta a **maior parte** dos recursos — o que você observou ("GUI muito diferente, recursos ausentes") **é esperado**: o MVVM é hoje **1 perspectiva** vs **4 abas + 9 diálogos** do monólito.
3. **Gap de paridade: ~20 áreas de recurso** (benchmark, preferências, editor de camadas, análise de correlação/ensemble, compositor de plots, gestão de experimentos, menus/atalhos, status bar, etc.). Há um **plano de 9 fatias** proposto abaixo (§4).
4. **Design de GUI sem prompt:** **Qt Designer (`.ui`)** é o caminho — e é **MVVM-safe por construção**. **"Claude Design" existe** mas exporta web (não Qt). **Figma→PyQt6 não existe** (só Figma→QML). O **maior ganho** é um **loop de screenshot offscreen** (`QWidget.grab()` → PNG → o agente lê e ajusta) — sua infra de CI **já suporta** isso.
5. **O Studio NÃO foi iniciado em código.** Zero `apps/studio/`, zero commits. O que existe é a **fundação MVVM compartilhada** (`geosteering_ai/gui/`, specs 0004-0007) que o Studio **vai reaproveitar**. O trabalho atual no SM **é** o que constrói essa fundação. (Esclarecimento detalhado em §7.)
6. **O SDD está maduro e sendo seguido com rigor**: CONSTITUTION (12 princípios), 6 gates, 22 hooks, 3 templates, ADR-0001, rastreabilidade RF→AC→teste. 10/11 specs implementadas. (§8)
7. **Premortem (§9):** os 3 maiores riscos são (a) **branches não-mergeadas** (trabalho invisível/perecível), (b) **gap de paridade nunca fecha** (MVVM vira abandonware), (c) **GUI construída às cegas** (sem loop visual). Soluções documentadas em §9.

---

## 1. (Q1) Acesso ao Simulation Manager Monolítico

### 1.1 Existe e está íntegro

| Item | Valor |
|:--|:--|
| Arquivo principal | `geosteering_ai/simulation/tests/simulation_manager.py` |
| Tamanho | **10.762 linhas** |
| Status git | Tracked, íntegro (último toque: `95c09ae` de-shim spec 0011 Fase 0) |
| Módulos auxiliares | **14 × `sm_*.py`** (~7.1k LOC) no mesmo diretório |

**14 módulos `sm_*.py`** (o "resto" do monólito):

| Módulo | LOC | Propósito |
|:--|--:|:--|
| `sm_plots.py` | 1.796 | Motor de plotagem matplotlib (7 tipos, 9 componentes, geosinais) |
| `sm_workers.py` | 1.103 | Threads (simulação, benchmark, geração, salvamento de artefatos) |
| `sm_correlation.py` | 911 | Diálogos de correlação (Pearson/Spearman/Kendall) + ensemble |
| `sm_benchmark.py` | 830 | Config A/B/C/D + thread + modelos canônicos |
| `sm_layers_dialog.py` | 425 | **Editor manual de camadas** (QTableWidget: n_layers, esp., ρh, ρv) |
| `sm_heartbeat.py` | 410 | Monitor de travamento da main thread (opt-in `SM_HEARTBEAT=1`) |
| `sm_phase_timer.py` | 294 | Cronometragem de fases (profiling) |
| `sm_model_gen.py` | 279 | Wrapper Qt do gerador estocástico (já compartilhado com o MVVM) |
| `sm_canonical_profiles.py` | 239 | 6 perfis canônicos pré-configurados |
| `sm_animation_bar.py` | 208 | Slider de navegação do ensemble |
| `sm_toast.py` | 197 | Notificações não-bloqueantes |
| `sm_widgets.py` | 196 | Widgets reutilizáveis |
| `sm_dat_viewer.py` | 180 | Visualizador de `.dat`/`.out` |
| `sm_io.py` | 41 | Utilitários I/O |

### 1.2 Como executar via linha de comando

**Não há entry-point em `pyproject.toml`** (`[project.scripts]` só tem `geosteering-cli`, `geosteering-warmup`, `geosteering-api`). O monólito é lançado como **módulo Python**:

```bash
# 1) Ativar o ambiente (conda com PyQt6)
conda activate Geosteering_AI            # ou: source ~/Geosteering_AI_venv/bin/activate

# 2) Lançar o SM monolítico (abre a GUI)
python -m geosteering_ai.simulation.tests.simulation_manager

# Variante: suite de smoke-test integral (sem abrir GUI interativa)
python -m geosteering_ai.simulation.tests.simulation_manager --smoke-test
```

- `main()` (linha 10723) inicializa `multiprocessing` com **`spawn`**, cria `QApplication` com stylesheet dark, instancia `MainWindow` maximizada.
- `if __name__ == "__main__": sys.exit(main())` (linha 10761).
- Dependências de runtime: **PyQt6 ≥ 6.6** (fallback PySide6 via `gui/qt_compat.py`), matplotlib, numpy, scipy.

### 1.3 Sobre enviar os arquivos

**Não é necessário.** O monólito inteiro (principal + 14 `sm_*`) está versionado no repo e foi integralmente inventariado nesta investigação. Tenho acesso direto a todos os recursos para adaptá-los/otimizá-los no MVVM.

---

## 2. (Q2a) Estamos portando o monólito para MVVM? Tudo?

**Sim — e a intenção é paridade total de recursos, incrementalmente (Strangler Fig).**

A estratégia **Strangler Fig** (Martin Fowler) significa: construir a nova app **ao lado** da antiga, migrando recurso a recurso, com a antiga **intocada e funcional** até a nova a "estrangular" (substituir). Nunca um big-bang.

```
┌────────────────────────────────────────────────────────────────────┐
│  MONÓLITO (intocado, 100% funcional)         MVVM (cresce por fatia) │
│  simulation_manager.py + 14 sm_*.py    →→→   apps/sim_manager/       │
│  ~17.9k LOC · 4 abas · 9 diálogos            geosteering_ai/gui/     │
│                                                                      │
│  Fatias entregues: ███░░░░░░░░░░░░░░░░░  ~25% da paridade           │
└────────────────────────────────────────────────────────────────────┘
```

**O que JÁ foi portado (5 fatias / specs):**

| Spec/Fatia | Recurso | Status |
|:--|:--|:--:|
| 0011a (Fatia 1) | Walking skeleton (Worker, Service, ViewModel, View, Perspective) | ✅ |
| 0011b (Fatia 2) | Params multi-config (freqs/dips/TRs) + geometria + `n_pos` Fortran | ✅ |
| 0011c (Fatia 3) | Geração estocástica (7 geradores) + batch ragged | ✅ |
| 0011d (Fatia 4) | Galeria de resultados (componentes, plot-kinds, seletores, paginação, cache LRU) | ✅ |
| 0012 (Fatia 5) | **JAX-GPU em subprocesso TLS-safe** (numba/jax/auto) | ✅ |

> **Curiosidade técnica:** o MVVM **já tem um recurso que o monólito não tem** — o **backend JAX-GPU**. O monólito só oferece **numba/fortran**. Em compensação, o monólito tem ~20 áreas de recurso que o MVVM ainda não tem (§4).

---

## 3. (Q2c) Vantagens do MVVM sobre o monólito

| Dimensão | Monólito | MVVM | Ganho |
|:--|:--|:--|:--|
| **Testabilidade** | GUI testável só com `pytest-qt` (lento, precisa de QApplication); lógica acoplada a widgets | **ViewModel é Python puro** (sem Qt) → testável com `pytest` comum, sem display. Validação da errata, `compute_n_pos`, guardrail OOM testados isoladamente | **Alto** — testes rápidos, determinísticos, sem flakiness de GUI |
| **Modularidade** | 1 arquivo de 10.7k LOC + 14 satélites; tudo na `MainWindow` | **Perspectivas plugin** (`Perspective` ABC); cada tela é um trio View/VM/Service independente, descobrível | **Alto** — telas adicionadas sem tocar o núcleo |
| **Manutenção** | Mudar a lógica exige navegar 10.7k linhas; risco de efeito colateral | Separação de responsabilidades (View só renderiza; VM decide; Service orquestra) | **Alto** — mudança localizada |
| **Reuso (Studio)** | Nada reaproveitável sem refactor | `geosteering_ai/gui/` é **fundação compartilhada** — o Studio herda `MainWindowBase`, `Perspective`, `VMSignal`, plot backends, persistência | **Crítico** — o Studio só é viável por causa disto |
| **Multi-binding** | Acoplado (de-shim recente mitigou) | `qt_compat` agnóstico (PyQt6 ↔ PySide6) | Médio |
| **Fidelidade física** | `simulate_batch` chamado de dentro da GUI | Física isolada em `sim_request.py` (puro); GUI nunca toca o cálculo | **Alto** — paridade <1e-12 protegida por construção |
| **Troca de backend de plot** | `swap_backend` matplotlib/pyqtgraph/plotly (já existe!) | `make_canvas` + `PlotBackend` enum (mesma capacidade, melhor estruturada) | Paridade |

**Resumo:** o MVVM troca **velocidade inicial** (mais arquivos, mais cerimônia) por **velocidade sustentável** (testes rápidos, telas plugáveis, fundação reutilizável pelo Studio, fidelidade blindada). É um investimento que só compensa porque (a) o projeto terá MUITAS telas (Studio) e (b) a física não pode regredir.

---

## 4. (Q2b) Recursos ausentes no MVVM + plano de portabilidade

### 4.1 Gap de paridade (o que falta)

Cruzando os dois inventários, o MVVM ainda **não tem** (~20 áreas):

| # | Recurso ausente | Onde está no monólito | Complexidade |
|:-:|:--|:--|:--:|
| 1 | **Controles de execução** (Pause/Resume/Stop/Cancel) + **barra de progresso** + **painel de log** | `SimulatorPage` 2503-2630 | Média |
| 2 | **Status bar** (estado, throughput, elapsed, cache, binding) | 7416-7443 | Baixa |
| 3 | **Gestão de experimentos** (`.exp.json`: Novo/Abrir/Salvar/Fechar, recentes) | 857-978, 7970-8049 | Média-Alta |
| 4 | **Histórico de snapshots** (lista, indicador de cache ●/○, busca, reload) | 2654-2872 | Média |
| 5 | **Editor manual de camadas** (tabela ρh/ρv/espessura) | `sm_layers_dialog.py` (425) | Média |
| 6 | **Perfis canônicos** (6 presets) + auto-geometria | 1287-1338, `sm_canonical_profiles.py` | Baixa |
| 7 | **Config de filtros de Hankel** (auto vs manual) | 1596-1642 | Baixa |
| 8 | **Config de paralelismo** (n_workers, threads/worker, python exec) | 2356-2475 | Média |
| 9 | **Aba Benchmark** (Config A/B/C/D, tabela de resultados, export CSV) | `BenchmarkPage` 3260-3722, `sm_benchmark.py` | **Alta** |
| 10 | **Aba Preferências** (paths, estilo de plot, cache LRU UI) | `PreferencesPage` 6393-7283 | Média-Alta |
| 11 | **Plot composer** (compositor rico de figuras + export) | 3953-4387 | Alta |
| 12 | **Análise de correlação** (heatmap Pearson/Spearman/Kendall + CSV) | `sm_correlation.py` (911) | Alta |
| 13 | **Análise de ensemble** (mediana + P5/P95 + outliers) | `sm_correlation.py` | Média-Alta |
| 14 | **Animation bar** (slider do ensemble) | `sm_animation_bar.py` (208) | Baixa |
| 15 | **DatViewer** (`.dat`/`.out` sem re-simular) | `sm_dat_viewer.py` (180) | Baixa |
| 16 | **Plots especializados**: perfil ρ, anisotropia λ, **geosinais (5)**, tensor completo, comparação Numba×Fortran, testes Jacobiano/forward | `sm_plots.py` | **Alta** |
| 17 | **Exportação** (`.dat`/`.out`, PNG/PDF com DPI) — UI | `SaveArtifactsThread`, PlotComposer | Média |
| 18 | **Menus + toolbars + atalhos** (Ctrl+R/B/S/O, F5, Ctrl+1-4, Esc…) | 7679-7913 | Baixa |
| 19 | **Toasts** (notificações) | `sm_toast.py` (197) | Baixa |
| 20 | **Backend Fortran na GUI** (MVVM só tem numba/jax/auto) | `SimulatorPage` 2316-2354 | Média |
| — | Observabilidade (phase timer, heartbeat) | `sm_phase_timer.py`, `sm_heartbeat.py` | Baixa (opcional) |

> O MVVM hoje tem **4 plot-kinds** (Re/Im/Mag/Phase de componentes); o monólito tem **8 categorias** de plot incluindo perfis físicos e geosinais. Item 16 é o de maior valor científico.

### 4.2 Plano de portabilidade proposto (as etapas)

Sequência sugerida (cada fatia = 1 spec SDD, com revisão adversarial e paridade preservada). Ordenada por **valor × dependência**:

| Fatia | Tema | Recursos | Por que nesta ordem |
|:--|:--|:--|:--|
| **6a** ⭐ | **Execução & feedback** | #1 progresso/log/cancel/pause, #2 status bar | Pré-requisito de UX para sims longas (GPU); destrava o cancel deferido da 0012 |
| **6b** | **Geologia avançada** | #5 editor de camadas, #6 perfis canônicos + auto-geo, #7 Hankel | Completa a entrada de modelo (modo manual hoje ausente) |
| **6c** | **Experimentos & histórico** | #3 `.exp.json`/`.gsproj`, #4 histórico+cache, recentes | Durabilidade do trabalho do usuário |
| **6d** | **Plots completos** | #16 perfis ρ/λ/geosinais/tensor, #11 plot composer, #14 animation bar | Maior valor científico; o "olho" do geofísico |
| **6e** | **Exportação** | #17 `.dat`/`.out`/PNG/PDF | Fecha o ciclo de saída |
| **6f** | **Análise estatística** | #12 correlação, #13 ensemble, #15 DatViewer | Recursos de análise pós-simulação |
| **6g** | **Benchmark** | #9 aba benchmark completa | Independente; pode paralelizar |
| **6h** | **Preferências & backends** | #10 preferências, #20 Fortran na GUI, #8 paralelismo | Configuração avançada |
| **6i** | **Chrome & polish** | #18 menus/toolbars/atalhos, #19 toasts, observabilidade | Acabamento; melhor por último |
| **Cutover** | **Congelar & migrar** | Freeze do monólito → MVVM vira o SM oficial | Só após paridade + validação visual |

**Quando?** Depende do ritmo de sprints. Estimativa grosseira: **~9 fatias**, cada uma 1 sprint curto. Não há datas no SSoT (ADR-0001 R2: versão atribuída no 1º commit da sprint). A próxima recomendada é a **6a**.

---

## 5. (Q3) Como desenhar a GUI sem ser só por prompt

### 5.1 Achados (pesquisa web, 2026)

| Ferramenta | Gera PyQt6/QtWidgets? | Veredito |
|:--|:--:|:--|
| **Qt Designer (`.ui` XML)** | ✅ Sim, nativo | **Caminho recomendado.** `.ui` descreve só a View → **MVVM-safe por construção** |
| **"Claude Design"** (Anthropic, abr/2026) | ❌ Exporta HTML/PDF/PPTX/Canva | Só **ideação visual** (mockup); não gera Qt |
| **Figma → Qt** (plugin oficial / Qt Bridge / FigmaQML) | ❌ Gera **QML**, nunca QtWidgets | **Figma→PyQt6 não existe.** A própria Qt confirma: "no direct solution for Qt Widgets" |
| **Figma MCP** (`get_design_context`) | ❌ Retorna React+Tailwind (web) | Útil só como **referência visual/tokens**; agente re-escreve à mão |
| **Mockup PNG → código (visão)** | ⚠️ Parcial | Bom **input de intenção**; espere iteração, não one-shot |
| **QML / Qt Design Studio** | (outra stack) | **Não migrar** — seu domínio é data/forms desktop; QML atrita com VM-Python-puro |

### 5.2 O ponto central: `.ui` é MVVM-safe

Um `.ui` é XML declarativo — **não importa nada**, logo **não pode ferir a pureza do ViewModel**. A separação se mantém:

```
form.ui (XML, só widgets)  ──uic.loadUi──▶  SimulatorView (camada fina, à mão)
                                                 │  conecta sinais Qt + VMSignal
                                                 ▼
                                      SimulationViewModel (Python PURO, INTOCADO)
```

O `viewmodel.py` (600+ linhas: errata, `compute_n_pos`, OOM guard) **não muda uma vírgula** — só os widgets passam a ser montados pelo `.ui` em vez de instanciados à mão.

**Dois fluxos:**
- **`pyuic6`** compila `.ui`→`.py` (tipos no IDE; mas `.py` gerado **não se edita à mão**, é regenerado).
- **`uic.loadUi`** carrega em runtime (`.ui` como fonte única, sem `.py` no repo). **Recomendado** para este projeto.

**Pré-requisito de implementação:** o shim `qt_compat.py` hoje **não re-exporta `uic`**. Como o carregamento de `.ui` é binding-específico (`PyQt6.uic.loadUi` vs `PySide6.QtUiTools.QUiLoader`), seria preciso adicionar um helper `load_ui(path, baseinstance)` que despacha por `QT_BINDING` — senão `.ui` cravaria PyQt6 e quebraria o fallback PySide6.

### 5.3 O maior ganho: loop de screenshot offscreen

**O agente CONSEGUE ver a GUI que gerou** — e sua infra **já suporta** (`QT_QPA_PLATFORM=offscreen` + `xvfb` na CI; `pytest-qt>=4.4`):

```python
# preview_view.py — rodar via: QT_QPA_PLATFORM=offscreen python preview_view.py
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from geosteering_ai.gui.qt_compat import QtWidgets
app = QtWidgets.QApplication([])
view = SimulatorView(vm); view.resize(900, 700)
view.grab().save("/tmp/preview.png")   # QWidget.grab() → PNG (offscreen-friendly)
```

O agente então **lê `/tmp/preview.png`** (a tool Read renderiza imagens) e **literalmente vê** alinhamento/espaçamento/widgets faltando → corrige. Transforma "gerar UI às cegas" em **iteração visual fechada** (e vira teste de regressão visual para o Studio).

### 5.4 Recomendação final (design de GUI)

1. **ViewModels: 100% à mão, sempre puros** (nunca tocados por Designer/Figma/Claude Design).
2. **Views — híbrido por tipo de tela:**
   - **Telas grandes/estáticas** (futuro Studio, abas densas) → **Qt Designer (`.ui`) via `uic.loadUi`**.
   - **Telas dinâmicas/compostas** (galeria, grupos parametrizados) → **Python à mão** (como hoje).
   - **SM atual:** não reescrever o que funciona; adotar `.ui` só em telas novas.
3. **Estender o shim** com `load_ui` antes de adotar `.ui`.
4. **Adotar o loop de screenshot offscreen já** (ROI imediato, custo quase zero).
5. **Ideação:** mockup low-fi (Excalidraw/Balsamiq/Claude Design) → PNG → primeiro rascunho.
6. **Não fazer:** Figma→PyQt6 (inexistente), migrar para QML, editar `.py` do `pyuic6` à mão.

---

## 6. (Q4) Você iniciou o Geosteering AI Studio?

### Resposta direta: **NÃO. Zero linhas de código de aplicação do Studio.**

| Evidência | Resultado |
|:--|:--|
| `find . -iname "*studio*"` | Só 2 **documentos** de planejamento (nenhum código) |
| `ls apps/` | Só `sim_manager/` — **não existe `apps/studio/`** |
| `git log --all --grep="studio"` | **Zero commits** |
| Perspectivas plugin (Treino/Inferência/Realtime/Registry) | **Zero implementadas** |

### O que está acontecendo (esclarecimento)

Você pode ter associado o trabalho recente ao Studio porque ele cria **`geosteering_ai/gui/`** — mas essa é a **fundação MVVM compartilhada** (specs 0004-0007), não o Studio. A relação:

```
            ┌─────────────────────────────────────────────────┐
            │   geosteering_ai/gui/  (FUNDAÇÃO COMPARTILHADA)  │
            │   MainWindowBase · Perspective · VMSignal ·      │
            │   plot_backends · persistence · services        │
            └───────────────┬─────────────────┬───────────────┘
                            │                 │
              ┌─────────────▼──────┐   ┌──────▼──────────────────┐
              │  SM MVVM (HOJE)    │   │  Studio (FUTURO, 0%)    │
              │  apps/sim_manager/ │   │  apps/studio/ (não existe)│
              └────────────────────┘   └─────────────────────────┘
```

- **O que você está construindo agora:** o **SM MVVM** + a **fundação `gui/`** que o Studio herdará.
- **Studio:** permanece em **planejamento/requisitos** (relatórios de 2026-06-04: 47 RF, 4 pilares, decisões abertas Q1-Q8). Bloqueado por **decisões de negócio**, não técnicas.
- **Quando começar o Studio:** após (a) decisões Q1-Q8 e (b) maturação da fundação via o SM. O SM é o "campo de provas" da fundação que o Studio vai reusar.

**Conclusão:** você não está perdido — o trabalho está **na trilha certa** (fundação + SM). O Studio é um produto **separado** (provável repo próprio, trilha comercial), ainda não iniciado.

---

## 7. (Q5) Como o SDD está contribuindo

### 7.1 Maturidade comprovada

| Artefato | Evidência | Maturidade |
|:--|:--|:--:|
| **CONSTITUTION.md** | 12 princípios invioláveis, v1.0 | Alta |
| **6 Gates** | GATE-S/P/T/I/V com checklist | Alta |
| **22 hooks pre-commit** | Paridade Fortran, errata, anti-pytorch, PT-BR, anti-patterns | Alta |
| **3 templates** | spec/plan/tasks canônicos | Alta |
| **ADR-0001** | SSoT do planejamento (INDEX→ROADMAP→sprints→CHANGELOG→ADR) | Alta |
| **Rastreabilidade** | RF→AC→teste em 8 arquivos (docstrings citam AC-1, RF-2) | Alta |
| **Commits** | 19% dos últimos 100 referenciam specs | Alta |

**Specs:** 11 criadas, **10 implementadas com testes verdes** (90.9%). Front-matter rico (`Backlog-Code`, `Trilha`, `Produtos`, `Converge-Em`, `Status`, `Constituicao`).

### 7.2 Contribuição concreta ao projeto

1. **Fidelidade blindada:** o princípio "Paridade Física Sagrada" + hook `run-fortran-parity.sh` impede que qualquer spec quebre <1e-12. **Nenhuma regressão física** em 10 specs.
2. **Escopo controlado:** seções IN/OUT explícitas + `[NEEDS CLARIFICATION]` + GATE-S (0 marcadores) evitam scope creep. (Ex.: a 0012 deferiu timeout/cancel explicitamente — e a revisão adversarial confirmou que isso era OUT.)
3. **Decomposição executável:** o épico do SM virou fatias atômicas (0011a-d, 0012) → commits pequenos, revisáveis, com rollback fácil.
4. **Rastreabilidade:** cada AC tem teste; cada commit referencia spec. Auditável.
5. **Revisão adversarial integrada:** specs passam pelos gates **e** por revisão adversarial (a 0012 teve 24 achados → 2 genuínos corrigidos). O SDD não elimina revisão — aumenta a confiança antes dela.

### 7.3 Dívidas do processo (oportunidades)

| Dívida | Evidência | Correção sugerida |
|:--|:--|:--|
| `Status: planejado` desatualizado | Todas as 11 specs dizem "planejado" apesar do código pronto | Atualizar para "implementado" no merge |
| `Released-As` vazio | Nenhuma spec mergeada com versão | **Mergear as branches `feat/gui-*` à `main`** (ver premortem) |
| ADRs pendentes | Decisões abertas (D-A a D-H) sem ADR | Ratificar antes da próxima onda |

---

## 8. (Q6) Premortem do SM MVVM e do processo

> **Premortem** = imaginar que o projeto **já fracassou** daqui a 6-12 meses e trabalhar de trás para frente para descobrir as causas — antes que aconteçam.

### 8.1 Cenários de fracasso (ordenados por risco = probabilidade × impacto)

| # | Cenário de fracasso | Prob. | Impacto | Sinais hoje |
|:-:|:--|:--:|:--:|:--|
| **P1** | **Branches `feat/gui-*` nunca mergeiam à `main`** → trabalho invisível, `main` diverge, conflitos acumulam, risco de perda (não-pushadas) | **Alta** | **Alto** | `Released-As` vazio; memória confirma branches empilhadas e **não-pushadas** |
| **P2** | **Gap de paridade nunca fecha** → MVVM trava em "skeleton+params+results"; usuários ficam no monólito; MVVM vira abandonware | **Média-Alta** | **Alto** | ~20 áreas faltando; ~25% de paridade após 5 fatias |
| **P3** | **GUI construída às cegas** → sem loop visual, a UI gerada destoa do monólito (você já notou "GUI muito diferente"); retrabalho estético | **Média** | **Médio** | Loop de screenshot **ainda não adotado** |
| **P4** | **Dois SMs divergem** → monólito continua evoluindo (teve v2.29.x) enquanto o MVVM persegue alvo móvel; correções duplicadas | **Média** | **Médio** | Monólito ainda é o oficial; sem freeze formal |
| **P5** | **Pureza MVVM erode** sob pressão de features complexas (plot composer, correlação) → lógica vaza para a View; perde-se a testabilidade | **Média** | **Alto** | Features complexas (16/12/13) ainda não portadas — risco futuro |
| **P6** | **Regressão de fidelidade** → o reassembly/grouping novo (`sim_request`) introduz Δ numérico | **Baixa** | **Crítico** | Mitigado por testes de paridade, mas código novo |
| **P7** | **Spawn de subprocesso domina UX** → JAX efêmero (~s de spawn+CUDA por run) torna sims interativas pequenas lentas → usuário volta ao monólito | **Média** | **Médio** | Pool efêmero (documentado como v1; OUT na 0012) |
| **P8** | **Over-engineering da fundação** para um Studio que é 0% e gated em Q1-Q8 → esforço em abstração especulativa | **Baixa-Média** | **Médio** | Fundação já existe; risco se crescer sem consumidor real |

### 8.2 Soluções documentadas

| Risco | Solução | Quando |
|:--|:--|:--|
| **P1** ⭐ | **Mergear/pushar as `feat/gui-*` à `main` AGORA.** Consolidar a pilha, push, atualizar `Status`/`Released-As`. Trabalho não-pushado é trabalho perecível. | **Imediato** (antes da Fatia 6a) |
| **P2** | Seguir o **plano de 9 fatias** (§4.2) com cadência fixa; medir paridade a cada fatia (checklist §4.1). Priorizar por valor (6a→6d primeiro). | Contínuo |
| **P3** | **Adotar o loop de screenshot offscreen** (§5.3) já na Fatia 6a; criar `preview_*.py` + teste de regressão visual. | Fatia 6a |
| **P4** | **Declarar freeze do monólito** (só correções críticas; sem features novas) assim que a paridade chegar a ~70%. Documentar a data de cutover. | Após ~70% paridade |
| **P5** | Manter o **GATE de pureza**: teste que falha se o VM importar Qt; revisão adversarial foca vazamento View↔VM em cada fatia de feature complexa. | Cada fatia |
| **P6** | **Nunca reimplementar física** — só `simulate_batch`. Teste de paridade <1e-12 obrigatório por fatia que toca dados. (Já é princípio I.) | Sempre |
| **P7** | Implementar **pool persistente + warmup** (já mapeado como OUT/futuro na 0012) quando a Fatia 6a integrar progresso/cancel; ou preflight numba-vs-jax para batches pequenos. | Fatia 6a/6h |
| **P8** | **YAGNI na fundação**: só abstrair o que o SM (consumidor real) exige. Adiar abstrações Studio-only até o Studio começar. | Contínuo |

### 8.3 Premortem do processo (SDD/desenvolvimento)

| Risco do processo | Solução |
|:--|:--|
| Specs `planejado` eternamente → status perde sentido | Hook/checklist que atualiza `Status`/`Released-As` no merge |
| Revisão adversarial vira teatro (achados sempre "refutados") | Manter taxa de achados genuínos auditável; variar ângulos de revisão |
| Documentação cresce mais que o código (relatórios em excesso) | Consolidar relatórios; INDEX/ROADMAP como SSoT (não reports dispersos) |
| INDEX/ROADMAP stale (memória confirma) | Atualizar na mesma PR da feature |

---

## 9. Próximos passos recomendados

1. ⭐ **Resolver P1**: consolidar e **mergear as branches `feat/gui-*` à `main`** (+ atualizar `Status`/`Released-As`, INDEX/ROADMAP). Higiene crítica.
2. **Fatia 6a** (execução & feedback): progresso/log/cancel/pause + status bar — **e adotar o loop de screenshot offscreen** (mata P3 e destrava o cancel da 0012).
3. **Estender `qt_compat`** com `load_ui` e fazer um **piloto de `.ui`** numa tela nova (validar o fluxo Designer→MVVM).
4. Seguir o **plano de 9 fatias** (§4.2) até ~70% de paridade → **freeze do monólito** → cutover.

---

*Relatório gerado por investigação read-only (5 agentes paralelos). Nenhum código foi modificado. Fontes web citadas na investigação de design de GUI (PythonGUIs, Qt docs, Anthropic, Figma, arXiv).*
