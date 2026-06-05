> **Blueprint de Arquitetura do Geosteering AI — UI/UX + MVVM.** Índice: [README.md](README.md) · Constituição SDD: [../../specs/CONSTITUTION.md](../../specs/CONSTITUTION.md) · Roadmap: [../../specs/ROADMAP.md](../../specs/ROADMAP.md). Gerado 2026-06-05 (workflow multi-agente + revisão crítica).

# Arquitetura UI/UX (Qt6 · MVVM) — Geosteering AI

> Escopo: fundação Qt compartilhada (`geosteering_ai/gui/`), os **dois** apps que a consomem
> (Simulation Manager e Studio flagship), regras de threading/undo/persistência/temas,
> e estratégia de testabilidade. Toda decisão é ancorada na infra Qt **já existente** em
> `geosteering_ai/simulation/tests/sm_*.py` (citada como `arquivo:linha`).

## 0. Reconciliação SM ↔ Studio (regra fundadora)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  FUNDAÇÃO COMPARTILHADA  →  geosteering_ai/gui/   (infra Qt, SEM domínio de produto)│
│    qt_compat · plotting/backends · services/threading · persistence · widgets ·    │
│    shell (MainWindowBase + Perspective ABC) · viewmodels (base puro)               │
└───────────────┬───────────────────────────────────────────────┬──────────────────┘
                │                                                 │
   ┌────────────▼──────────────┐                    ┌─────────────▼──────────────────┐
   │  APP Simulation Manager   │                    │  APP Studio (FLAGSHIP)          │
   │  gui/ + simulation/ APENAS│                    │  gui/ + BIBLIOTECA INTEIRA      │
   │  1 perspectiva: Simulação │                    │  5 perspectivas (plugins)       │
   │  casca própria (sm_app/)  │                    │  casca própria MVVM (studio/)   │
   └───────────────────────────┘                    └─────────────────────────────────┘
                │                                                 │
                └───────────────► geosteering_ai/simulation/multi_forward.py ◄──────────┘
                          FÍSICA NUNCA DUPLICADA — convergência única (paridade Fortran <1e-12)
```

| Princípio | Regra concreta | Status |
|:----------|:---------------|:-------|
| Fundação única | `geosteering_ai/gui/` extraído de `sm_*.py`; nenhum app importa de `simulation/tests/` | planejado |
| Dois produtos, não um | SM e Studio são `apps/` independentes; ambos `import geosteering_ai.gui` | planejado |
| Casca própria | Cada app tem seu `MainWindow`, registra suas perspectivas; o monólito `simulation_manager.py` NÃO é esqueleto | planejado |
| Física única | SM (Numba/JAX/Fortran) e Studio convergem em `multi_forward.py` / `dispatch.py:233` | implementado |
| Backend sem Qt | `geosteering_ai/{models,losses,training,inference,...}` permanecem importáveis sem Qt | implementado |

**Divergência registrada vs. digest `synth_architecture.md:72`**: o digest propunha um pacote
top-level `geosteering_studio/`. Esta tarefa exige explicitamente a fundação **in-tree**
`geosteering_ai/gui/`. Adotamos `geosteering_ai/gui/` (fundação) + `apps/sim_manager/` +
`apps/studio/` (cascas). Trade-off: o pip-lib `geosteering_ai` passa a ter um subpacote que
*pode* importar Qt — mitigado por Qt como `extra` opcional (`pip install geosteering-ai[gui]`)
e por `gui/` jamais ser importado pelo core. Ver Open Questions Q1.

---

## 1. Por que MVVM (e não MVC) para apps Qt complexos

| Critério | MVC clássico | MVP | **MVVM (escolhido)** |
|:---------|:-------------|:----|:---------------------|
| Acoplamento View↔lógica | Controller conhece widgets | Presenter chama View por interface | ViewModel **não conhece** a View (binding por sinais) |
| Testabilidade sem GUI | Baixa (Controller toca Qt) | Média (mock de View) | **Alta** — ViewModel é Python puro, testável sem `pytest-qt` |
| Fit com Qt | Qt não tem Controller natural | Verboso (1 interface/tela) | **Nativo**: sinais/slots = data binding; `QUndoStack`/`QAbstractItemModel` casam com VM |
| Estado de UI complexo | Espalhado | No Presenter | **Centralizado e serializável** no ViewModel (`.session`/`.gsproj`) |
| Múltiplas Views/1 estado | Difícil | Difícil | **Trivial** (curtain 2D + tabela + 3D leem o mesmo VM) |

**Decisão**: MVVM. Três razões concretas para *este* projeto:

1. **Meta de cobertura ≥80% sem `pytest-qt`** — hoje a lógica vive dentro de `QWidget`
   (`SimulatorPage`, `ResultsPage` no monólito), forçando teste headless via `xvfb`.
   Movendo a lógica para ViewModels puros, 80% da suíte roda como pytest comum, rápido e
   determinístico. A cobertura atual (~25%) é limitada justamente por isso.
2. **Reuso multi-View** — o mesmo `SimulationResultVM` alimenta 4 backends de plot
   (`sm_plot_backends/`) + tabela + export, sem duplicar estado. MVC exigiria um Controller
   por combinação.
3. **Serialização de estado = persistência** — em MVVM o estado da UI É o ViewModel; salvar
   `.gsproj` é serializar VMs (evolução natural de `ExperimentState` em
   `simulation_manager.py:331`).

> **MVC falharia** porque o Qt não oferece um "Controller" idiomático: sinais/slots empurram
> a lógica para dentro do widget (View-Controller fundido), exatamente o anti-padrão do
> monólito de 10,7k LOC. MVVM inverte isso com data-binding explícito.

---

## 2. Pacote compartilhado `geosteering_ai/gui/`

### 2.1 Estrutura de diretórios (origem de cada item em `sm_*.py`)

```
geosteering_ai/gui/                       ← FUNDAÇÃO Qt (extra opcional [gui])
├── __init__.py                           ← exporta API estável + versão de infra
├── qt_compat.py                          ← 100% de sm_qt_compat.py (PyQt6→PySide6, locale C, dark detect)
├── shell/                                ← infra de janela reutilizável (SEM lógica de produto)
│   ├── main_window_base.py               ← MainWindowBase: dock host + menu/toolbar + statusbar + tema
│   ├── perspective.py                    ← Perspective ABC (espelha PlotCanvas ABC, base.py:107)
│   ├── perspective_host.py               ← QTabWidget/QDockWidget host + lazy-load por entry_point
│   ├── theme.py                          ← ThemeManager (Light/Dark/System) ← detect_os_dark_mode()
│   ├── tokens.py                         ← design tokens (cores/spacing/tipografia) — fonte única
│   ├── toast.py                          ← de sm_toast.py (ToastManager + ToastNotification)
│   └── status_bar.py                     ← StatusBarController (cache/throughput/binding/backend)
├── viewmodels/
│   ├── base.py                           ← BaseViewModel (PURO; emite via callback, NÃO importa Qt)
│   ├── signal.py                         ← VMSignal: pub/sub puro-Python (adaptado a Qt no bind)
│   └── command.py                        ← CommandVM + suporte a QUndoStack (lado VM, sem Qt)
├── services/
│   ├── base.py                           ← BaseService (orquestra backend; emite via VMSignal)
│   └── threading/
│       ├── worker.py                     ← Worker(QObject) + moveToThread (I/O-bound)
│       ├── process_pool.py               ← EphemeralProcessRunner (de sm_workers.py:470, anti-hang v2.29)
│       └── signals.py                    ← WorkerSignals(QObject): progress/log/finished/error/paused
├── plotting/
│   ├── style.py                          ← PlotStyle (tema → cores de canvas)
│   └── backends/                         ← 100% de sm_plot_backends/ (Strategy+Factory)
│       ├── base.py                       ← PlotCanvas ABC + make_canvas() + available_backends()
│       ├── mpl_canvas.py · pyqtgraph_canvas.py · plotly_canvas.py · vispy_canvas.py
├── persistence/
│   ├── snapshot.py                       ← de sm_snapshot_persist.py (SnapshotPersistThread → genérico)
│   ├── plot_cache.py                     ← de sm_plot_cache.py (LRUPlotCache, auto 10% RAM)
│   ├── session.py                        ← SessionDocument (.session — estado volátil de UI)
│   └── project.py                        ← ProjectDocument (.gsproj — projeto persistente)
├── widgets/                              ← widgets atômicos do design system (sem domínio)
│   ├── collapsible.py                    ← de sm_widgets.py (CollapsibleGroupBox)
│   ├── animation_bar.py                  ← de sm_animation_bar.py (EnsembleAnimationBar)
│   ├── spin.py                           ← make_double_spin() wrappers (locale C garantido)
│   └── log_console.py                    ← console estruturado (consome Signal(str))
└── diagnostics/
    └── heartbeat.py                      ← de sm_heartbeat.py (MainThreadHeartbeat, opt-in env)
```

| Origem (`simulation/tests/`) | Destino (`gui/`) | Transformação | Status |
|:-----------------------------|:-----------------|:--------------|:-------|
| `sm_qt_compat.py` | `gui/qt_compat.py` | move literal (reuso ~100%) | implementado→reubicar |
| `sm_plot_backends/` | `gui/plotting/backends/` | move literal (ABC já genérica) | implementado→reubicar |
| `sm_snapshot_persist.py:63` | `gui/persistence/snapshot.py` | generalizar (texto JSON opaco já é genérico) | implementado→generalizar |
| `sm_plot_cache.py` | `gui/persistence/plot_cache.py` | move literal (já backend-agnóstico) | implementado→reubicar |
| `sm_workers.py:470` `SimulationThread` | `gui/services/threading/process_pool.py` | QThread→`EphemeralProcessRunner` + Worker | parcial (migrar) |
| `sm_toast/widgets/animation_bar/heartbeat` | `gui/{shell,widgets,diagnostics}/` | move literal | implementado→reubicar |
| Lógica em `*Page` (monólito) | `apps/*/viewmodels/` | **extrair** (MVVM) | ausente→extrair |

### 2.2 Diagrama de camadas MVVM (binding por sinal; ViewModel sem Qt)

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ VIEW (QWidget/QML)                       [importa Qt; ZERO lógica de domínio]        │
│   • lê tokens de gui/shell/tokens.py     • conecta sinais ↔ slots do VM             │
│   • SimulatorView, CurtainView, TrainingView ...                                    │
└───────────────┬───────────────────────────────────────────▲──────────────────────-─┘
   binding ↓ (user action → VM method)        notify ↑ (VMSignal → slot atualiza View)
┌───────────────▼───────────────────────────────────────────┴──────────────────────-─┐
│ VIEWMODEL  (Python PURO — NÃO importa PyQt6/PySide6)        [TESTÁVEL sem pytest-qt]  │
│   • estado de UI observável (props + VMSignal)  • valida entrada  • comandos undo    │
│   • SimulationVM, TrainingVM, RealtimeVM ...   • serializável → .session/.gsproj     │
└───────────────┬───────────────────────────────────────────▲──────────────────────-─┘
        chama ↓ (config validada)                    resultado ↑ (deepcopy em sinal)
┌───────────────▼───────────────────────────────────────────┴──────────────────────-─┐
│ MODEL = DOMÍNIO            PipelineConfig · SimulationConfig (frozen) · Project ·     │
│                            WellSession · ModelVersion · Run         [DDD value obj]   │
└───────────────┬───────────────────────────────────────────▲──────────────────────-─┘
        invoca ↓ (assíncrono, threaded)               sinais ↑ (progress/log/finished)
┌───────────────▼───────────────────────────────────────────┴──────────────────────-─┐
│ SERVICE (L2)   SimulationService · TrainingService · InferenceService · IngestSvc    │
│   orquestra threading (Worker+moveToThread | EphemeralProcessRunner)                  │
└───────────────┬─────────────────────────────────────────────────────────────────────┘
                ▼  import direto (sem Qt)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ BACKEND (L1)  geosteering_ai/  simulation(dispatch.py:233) · models · losses ·        │
│               training · inference(realtime.py:170) · evaluation · visualization      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Regra de ouro do binding** (como o VM puro fala com a View Qt sem importar Qt):

```
ViewModel:  self.progress = VMSignal()            # pub/sub puro Python
View (bind): vm.progress.connect(self._on_progress)  # adapter Qt no construtor da View
```
`VMSignal` (`gui/viewmodels/signal.py`) é uma lista de callbacks; a View registra um slot Qt.
Assim o teste faz `vm.progress.connect(rec.append)` — **sem QApplication, sem pytest-qt**.

---

## 3. App Simulation Manager (`apps/sim_manager/`)

### 3.1 Composição (= `gui/` + `simulation/` APENAS)

```
apps/sim_manager/
├── app.py                         ← bootstrap QApplication; registra 1 perspectiva; SM_MainWindow
├── main_window.py                 ← SM_MainWindow(MainWindowBase): menu Novo/Abrir/Salvar .session
├── perspectives/
│   └── simulation/
│       ├── view.py                ← SimulatorView (QWidget) — de SimulatorPage:2297 (só widgets)
│       ├── viewmodel.py           ← SimulationVM (PURO) — extrai lógica de ParametersPage:1077
│       ├── results_view.py        ← de ResultsPage:4385 (galeria + 4 backends plot)
│       └── benchmark_view.py      ← de aba Benchmark (sm_benchmark.py:830)
└── services/
    ├── simulation_service.py      ← consome SimulationService de gui/ + dispatch.py:233
    └── benchmark_service.py       ← de sm_benchmark.py (cenários A–H)
```

| Característica | Decisão | Status |
|:--------------|:--------|:-------|
| Escopo | SÓ simulação: forward Numba/JAX/Fortran + benchmark + geração estocástica de modelos | implementado (lógica em monólito) |
| Backend selector | `auto`/`numba`/`jax`/`fortran` via `dispatch.py:233` — **fecha gap** do SM atual (hardcode `backend="numba"` em `sm_workers.py:178,185`) | parcial→wire |
| Persistência | `.session` (estado de UI: params + snapshots) via `gui/persistence/session.py` | planejado (evolui `.exp.json`) |
| Threading | `EphemeralProcessRunner` (Numba/Fortran) + `Worker+moveToThread` (JAX warmup/IO) | parcial (migrar de `QThread`) |
| Dependência | importa `geosteering_ai.gui` + `geosteering_ai.simulation` — **nada mais** | planejado |

### 3.2 Fluxo (1 perspectiva)

```
SimulatorView ──(run)──▶ SimulationVM.run(SimRequest validado)
                              │  copy.deepcopy(req) antes de cruzar thread
                              ▼
                      SimulationService ──▶ EphemeralProcessRunner
                              │                 with ProcessPoolExecutor:  (anti-hang v2.29)
                              │                   warmup INLINE chunk[0] → loop → stack H
                              ▼  progress_update / log / finished_all (deepcopy do dict)
                      SimulationVM ──VMSignal──▶ ResultsView (PlotCanvas ×4) + StatusBar
```

---

## 4. App Studio (flagship — MVVM dia 1)

### 4.1 Composição (= `gui/` + biblioteca inteira; 5 perspectivas como plugins)

```
apps/studio/
├── app.py                         ← bootstrap; descobre perspectivas via entry_points
├── main_window.py                 ← Studio_MainWindow(MainWindowBase) + QUndoStack global
├── domain/                        ← L3 DDD (de ExperimentState:331 → Project/Well/Run/ModelVersion)
│   ├── workspace.py · project.py · well.py · run.py · model_version.py
├── perspectives/                  ← cada uma = plugin (Perspective ABC + entry_point)
│   ├── simulation/   {view,viewmodel}.py     ← Sim & Datasets
│   ├── training/     {view,viewmodel}.py     ← Treinamento
│   ├── inference/    {view,viewmodel}.py     ← Inferência/Análise
│   ├── realtime/     {view,viewmodel}.py     ← Geonavegação Realtime (núcleo)
│   └── registry/     {view,viewmodel}.py     ← Model Registry
├── services/
│   ├── simulation_service.py · training_service.py · inference_service.py
│   ├── ingest_service.py          ← WITS/ETP/LAS/DLIS (gap — ver §7)
│   └── registry_service.py        ← MLflow on-prem A6000 (gap)
└── viewmodels/  (VMs puros por perspectiva — testáveis sem Qt)
```

### 4.2 As 5 perspectivas (plugin = `Perspective ABC` + `entry_points`)

```python
# geosteering_ai/gui/shell/perspective.py
class Perspective(ABC):                       # espelha PlotCanvas ABC (base.py:107)
    id: str; title: str; icon: str; order: int
    @abstractmethod
    def build_view(self, ctx: AppContext) -> QWidget: ...   # cria View+VM, faz binding
    @abstractmethod
    def build_viewmodel(self, ctx: AppContext) -> BaseViewModel: ...  # PURO
    def on_activate(self) -> None: ...        # lazy: só constrói ao abrir a aba
    def on_close(self) -> bool: ...           # veto se houver undo dirty
```
```toml
# pyproject.toml  — descoberta automática (igual a sm_plot_backends Strategy validado)
[project.entry-points."geosteering_studio.perspectives"]
simulation = "geosteering_ai.apps.studio.perspectives.simulation:SimulationPerspective"
training   = "geosteering_ai.apps.studio.perspectives.training:TrainingPerspective"
inference  = "geosteering_ai.apps.studio.perspectives.inference:InferencePerspective"
realtime   = "geosteering_ai.apps.studio.perspectives.realtime:RealtimePerspective"
registry   = "geosteering_ai.apps.studio.perspectives.registry:RegistryPerspective"
```

| # | Perspectiva | ViewModel (puro) | Service / Backend consumido | Origem | Status |
|:-:|:------------|:-----------------|:----------------------------|:-------|:-------|
| 1 | Simulação & Datasets | `SimulationVM`, `DatasetGenVM` | `dispatch.py:233` · `data/synthetic_generator.py:196` · `validate_numeric_parity` | `SimulatorPage:2297`+`sm_model_gen` | parcial |
| 2 | Treinamento | `TrainingVM`, `HpoVM` | `ModelRegistry` · `LossFactory` · `training/loop.py:289` · `nstage.py:195` | novo (consome lib) | planejado |
| 3 | Inferência/Análise | `InferenceVM`, `EdaVM` | `InferencePipeline` · `evaluation/` · `visualization/` (Picasso/DOD) | `ResultsPage:4385`+`sm_plots` | parcial |
| 4 | Geonavegação Realtime | `RealtimeVM`, `SteeringVM` | `inference/realtime.py:170` (deque) · `UncertaintyEstimator` · `IngestService` | novo (núcleo) | planejado |
| 5 | Model Registry | `RegistryVM`, `CompareVM` | `evaluation/comparison` · MLflow (`RegistryService`) | novo | planejado |

### 4.3 Casca própria + undo global

- O Studio **não** herda o `MainWindow` do monólito; instancia `MainWindowBase` (de `gui/`)
  e injeta sua própria barra de perspectivas + `QUndoStack` global.
- Cada perspectiva é construída **lazy** (`on_activate`) — o app abre instantâneo mesmo com
  TensorFlow não-importado até a aba Treinamento ser aberta (import pesado adiado).

```
Studio_MainWindow
├── PerspectiveHost (QTabWidget)  ──discover──▶ entry_points("geosteering_studio.perspectives")
│     [Sim&Datasets] [Treino] [Inferência] [Realtime] [Registry]   ← ordenadas por .order
├── QUndoStack (global)  ──setClean()──▶ ligado a Project salvo (.gsproj)
├── DockManager (painéis: Log, Histórico/Cache LRU, Propriedades)
└── StatusBarController (backend ativo, GPU A6000, throughput, dirty *)
```

---

## 5. Regras de threading, undo/redo, persistência, temas, design system

### 5.1 Threading — matriz de decisão (origem: anti-hang v2.29, `sm_workers.py`)

```
┌── Natureza da carga ──────────────┬── Mecanismo ───────────────────┬── Por quê ──────────────┐
│ I/O-bound (save, ingest WITS,     │ Worker(QObject) + moveToThread │ libera GIL no I/O;       │
│ MLflow, leitura LAS/DLIS)         │ (gui/services/threading/       │ event loop por thread;   │
│                                   │  worker.py)                    │ preferível a QThread sub │
├───────────────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ CPU-bound Numba JIT (forward,     │ EphemeralProcessRunner:        │ GIL + Numba; pool POR    │
│ geração dataset)                  │ with ProcessPoolExecutor(...): │ simulação, fechado limpo │
│                                   │ warmup INLINE chunk[0]         │ (anti-hang v2.29;        │
│                                   │ (de sm_workers.py:470,699)     │ 150k+ mod/h)             │
├───────────────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ GPU JAX (batched dispatch)        │ Worker+moveToThread (1 device  │ XLA serializa; processo  │
│                                   │ por processo; NÃO ProcessPool) │ único evita disputa CUDA │
├───────────────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ TensorFlow training (GPU-bound)   │ subprocesso dedicado +         │ TF aloca a GPU inteira;  │
│                                   │ moveToThread p/ supervisão     │ isolar do event loop Qt  │
├───────────────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ Realtime inference on-arrival     │ callback pub/sub (NÃO loop);   │ gargalo=taxa de chegada  │
│                                   │ deque maxlen=seq_len           │ (mud-pulse ≫ inferência) │
└───────────────────────────────────┴────────────────────────────────┴──────────────────────────┘
```

**Invariantes de sinal (obrigatórios):**
1. **`copy.deepcopy` em payloads mutáveis** ao emitir `Signal(dict)`/`Signal(np.ndarray)` —
   o sinal carrega *referência*; sem deepcopy há corrida entre thread produtora e a UI.
   (Aplica-se a `finished_all(dict)` em `sm_workers.py:505`.)
2. **`WorkerSignals(QObject)` separado** do worker (padrão moveToThread) com:
   `progress(int,int,float)`, `log(str)`, `finished(object)`, `error(str)`, `paused()`,
   `resumed()`, `cancelled()` — espelha os 7 sinais de `SimulationThread` (`:503-509`).
3. **Cancelamento cooperativo** entre chunks via `threading.Event` (de `_wait_if_paused`,
   `sm_workers.py:566`) — nunca `terminate()`.
4. **Sem objeto Qt criado fora da main thread** (regra Qt); Workers só emitem sinais.

### 5.2 Undo/Redo (QUndoStack)

```
Ação do usuário (View) ──▶ VM cria QUndoCommand ──push──▶ QUndoStack (no MainWindow)
                                  │  redo()/undo() chamam métodos do VM (puro)
                                  ▼
   ParameterEditCommand · AddLayerCommand · DeleteSnapshotCommand · RenameRunCommand
```
| Regra | Decisão |
|:------|:--------|
| Granularidade | 1 `QUndoCommand` por edição atômica de domínio (camada, parâmetro, snapshot) |
| Lado do VM | `redo()/undo()` chamam métodos puros do VM; o Command não toca widgets |
| `mergeWith` | edições contínuas do mesmo spinbox fundem-se (1 undo por gesto) |
| Dirty tracking | `QUndoStack.cleanChanged` → marca `Project` sujo → título com `*` + bloqueia close |
| Escopo | Studio: stack global; SM: stack por `.session` |

### 5.3 Persistência — `.session` vs `.gsproj`

```
┌── .session (volátil/UI) ─────────────────┐   ┌── .gsproj (projeto durável) ──────────────┐
│ JSON; estado de ViewModels:              │   │ container (zip/HDF5): manifest + refs:     │
│  perspectiva ativa, params, layout dock, │   │  Project/Well/Run/ModelVersion (DDD),      │
│  snapshots em cache, backend selecionado │   │  datasets (paths/hash), lineage git+seed,  │
│ escopo: 1 sessão de trabalho             │   │  modelos registrados, .session embutida    │
│ migra de .exp.json (sim_mgr.py:331)      │   │ escopo: projeto reabrível, versionável     │
└──────────────────────────────────────────┘   └────────────────────────────────────────────┘
        ▲ salvo via SnapshotPersistThread (JSON pré-serializado fora da UI thread, sm_snapshot_persist.py:63)
```
| Regra | Decisão | Status |
|:------|:--------|:-------|
| I/O fora da UI | serializar JSON na main thread (rápido), gravar em thread (`snapshot.py`) | implementado |
| Forward-compat | chaves desconhecidas preservadas no load (já em `ExperimentState`) | implementado |
| Escrita atômica | write-temp → `os.replace` (endurecer `sm_snapshot_persist.py:130` que faz write direto) | parcial |
| `.gsproj` | container com manifest + lineage; SM usa só `.session` | planejado |
| Clean state | salvar → `QUndoStack.setClean()` | planejado |

### 5.4 Temas + design system

```
gui/shell/tokens.py  (FONTE ÚNICA)
├── cores:    --bg, --surface, --accent, --text, --grid, --uq-band, --interface-line
├── spacing:  4·8·12·16·24 (escala 4px)
├── tipografia: monospace p/ números científicos (locale C), sans p/ UI
└── 3 temas:  Light · Dark (VSCode-like) · System (detect_os_dark_mode(), qt_compat.py:240)
        │
        ├─▶ ThemeManager.apply(app)  → QPalette + QSS
        ├─▶ PlotStyle  → set_dark_mode() em cada PlotCanvas (base.py:190)
        └─▶ tokens consumidos por TODO widget (sem cor hardcoded)
```
| Componente do design system | Origem | Status |
|:----------------------------|:-------|:-------|
| `CollapsibleGroupBox` | `sm_widgets.py:37` | implementado |
| `ToastManager`/`ToastNotification` | `sm_toast.py:54,150` | implementado |
| `EnsembleAnimationBar` (scrubber UQ) | `sm_animation_bar.py:54` | implementado |
| `make_double_spin` (locale C) | `qt_compat.py:184` | implementado |
| `LogConsole` (estruturado) | extrair do monólito | parcial |
| `PlotCanvas` ×4 backends | `sm_plot_backends/` | implementado |

---

## 6. Testabilidade

```
┌── CAMADA ──────────┬── Tipo de teste ─────────────┬── Ferramenta ──────┬── Meta ──┐
│ ViewModel (puro)   │ unitário (estado, validação,  │ pytest (SEM Qt)    │ ≥90%     │
│                    │ comandos undo, serialização)  │                    │          │
│ Service            │ integração c/ backend mockado │ pytest + fakes     │ ≥80%     │
│ Threading          │ contrato de sinais (ordem,    │ pytest + QSignalSpy│ smoke    │
│                    │ deepcopy, cancel cooperativo) │ (headless xvfb)    │          │
│ View (widgets)     │ smoke (constrói, binda, fecha)│ pytest-qt + xvfb   │ smoke    │
│ Regressão visual   │ snapshot PNG por PlotCanvas   │ pytest + img-diff  │ baseline │
└────────────────────┴───────────────────────────────┴────────────────────┴──────────┘
        Cobertura global alvo ≥80% — viável porque ~80% da lógica está em VMs puros
```

| Garantia | Como | Enforcement |
|:---------|:-----|:------------|
| VM nunca importa Qt | teste de import-guard: `assert "PyQt6" not in sys.modules` após `import viewmodels` | hook CI + teste dedicado |
| Sinais sem corrida | `QSignalSpy` valida deepcopy (mutar payload original não afeta o recebido) | pytest |
| Regressão visual | render headless de cada `PlotCanvas` → comparar com baseline (tolerância px) | CI (artefato em falha) |
| Sem `print` | VMs/Services usam `logging` (regra D9); CLI é a única exceção stdout | hook existente |
| Headless CI | `xvfb-run -a pytest` (já no CI, `conftest.py` alinha `QT_API`) | implementado |

**Esqueleto de teste de VM (sem `pytest-qt`):**
```python
def test_simulation_vm_validates_frequency():
    vm = SimulationVM(SimulationService(fake_backend))
    errors = []
    vm.validation_error.connect(errors.append)   # VMSignal puro
    vm.set_frequency_hz(2.0)                        # viola errata (default 20000.0)
    assert errors and "FREQUENCY_HZ" in errors[0]   # roda sem QApplication
```

---

## 7. Gaps que a arquitetura precisa absorver

| Gap | Camada que resolve | Status |
|:----|:-------------------|:-------|
| `geosteering_ai/gui/` inexistente | extração de `sm_*.py` (§2) | ausente→planejado |
| SM sem JAX (hardcode `backend="numba"`, `sm_workers.py:178,185`) | `SimulationService` via `dispatch.py:233` | parcial |
| `ingest/` (WITS/ETP/LAS/DLIS/PWLS) | `IngestService` (Studio) | ausente |
| MLflow/registry | `RegistryService` (Studio) | ausente |
| UQ calibrada P10/P50/P90 + CRPS | `RealtimeVM`/`InferenceVM` + `UncertaintyEstimator` | parcial |
| Lógica acoplada a `QWidget` (cobertura ~25%) | extração para VMs puros (§6) | parcial |

---

## 8. Resumo de status

| Item entregue | Status |
|:--------------|:-------|
| Infra Qt base (qt_compat, plot backends, persistência, threading, widgets) | **implementado** (em `simulation/tests/`, a reubicar) |
| Fundação `geosteering_ai/gui/` (pacote nomeado) | **planejado** |
| MVVM (ViewModels puros) | **ausente** (lógica no monólito → extrair) |
| App SM como casca própria (gui/+simulation/) | **parcial** (monólito funcional; refatorar) |
| App Studio (5 perspectivas plugin) | **planejado** (0% código) |
| Undo/Redo · `.gsproj` · Registry | **planejado** |
