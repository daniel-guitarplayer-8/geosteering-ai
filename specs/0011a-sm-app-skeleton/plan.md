---
Spec: 0011a-sm-app-skeleton
Plano-de: spec.md
Status: planejado
Data: 2026-06-05
---

# Plano 0011a — Walking skeleton (HOW)

## 1. Gate de Constituição

| Princípio | Aplicável? | Viola? | Como cumpre |
|:--|:--:|:--:|:--|
| I Paridade (3 regimes) | sim | não | Service só chama `simulate_batch` (física <1e-12 intocada) |
| III TF/Keras exclusivo | sim | não | sem torch |
| V Config-parâmetro | sim | não | `SimRequest`/VM por parâmetro; service injetado; sem globals |
| VI Logging | sim | não | `logging` (não print) |
| X MVVM | sim | não | **núcleo**: VM puro testável; View⊥VM⊥Service; binding por VMSignal |
| XI Fundação | sim | não | `core` não importa `gui`/`apps`; VM não importa Qt |
| XII Gates | sim | não | reviewers + suíte + fronteira + regressão SM |

**GATE-P: sem violação.** Sem ADR (compõe abstrações já ratificadas).

## 2. Contratos / APIs (assinaturas exatas)

```python
# geosteering_ai/gui/threading/worker.py  (Qt)
class WorkerSignals(QObject):
    finished = Signal(object)      # resultado
    error = Signal(str)
    progress = Signal(int, int)    # done, total

class Worker(QObject):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs) -> None: ...
    def run(self) -> None:         # slot — roda na worker thread; emite finished/error

def run_in_thread(worker: Worker) -> QThread:  # moveToThread + start; quit no fim

# geosteering_ai/gui/services/base.py  (Qt — orquestração L2)
class BaseService:
    finished: VMSignal             # re-emite o resultado na MAIN thread
    error: VMSignal
    progress: VMSignal
    def _run_async(self, fn, *a, **k) -> None: ...   # Worker+thread; conecta sinais → VMSignal

# geosteering_ai/gui/services/simulation_service.py
@dataclass(frozen=True)
class SimRequest:
    frequencies_hz: tuple[float, ...] = (20000.0,)
    tr_spacings_m: tuple[float, ...] = (1.0,)
    dip_degs: tuple[float, ...] = (0.0,)
    n_models: int = 2
    backend: str = "numba"

class SimulationService(BaseService):
    def run(self, request: SimRequest) -> None: ...   # _run_async(_run_simulation, request)
# _build_batch(req) -> (rho_h, rho_v, esp, positions_z)   # batch fixo pequeno (3 camadas TIV)
# _run_simulation(req) -> dict{H6, positions_z, info, backend}   # chama simulate_batch (PURO p/ teste)

# apps/sim_manager/perspectives/simulation/viewmodel.py  (PURO, sem Qt)
class SimulationViewModel(BaseViewModel):
    _STATE_FIELDS = ("_freq_hz", "_dip_deg", "_n_models", "_status")
    def __init__(self, service) -> None: ...          # service INJETADO (duck-typed)
    # properties: frequency_hz, dip_deg, n_models, status, last_result
    result_ready: VMSignal
    def validate(self) -> list[str]: ...              # lista de erros (vazia = ok)
    def run(self) -> None: ...                         # valida → SimRequest → service.run

# apps/sim_manager/perspectives/simulation/view.py  (Qt)
class SimulatorView(QtWidgets.QWidget):
    def __init__(self, vm: SimulationViewModel, parent=None) -> None: ...  # binda VMSignals

# apps/sim_manager/perspectives/simulation/__init__.py
class SimulationPerspective(Perspective):
    id, title, order = "simulation", "Simulação", 0
    def build_viewmodel(self, ctx) -> SimulationViewModel: ...  # SimulationVM(SimulationService())
    def build_view(self, ctx) -> QWidget: ...                   # vm=build_viewmodel; SimulatorView(vm)

# apps/sim_manager/main_window.py
class SM_MainWindow(MainWindowBase): ...   # casca; menu .session = fatia futura

# apps/sim_manager/app.py
def main() -> int: ...   # QApplication → SM_MainWindow(AppContext) → add_perspective → exec
```

## 3. Estrutura de Arquivos

| Arquivo | Ação |
|:--|:--|
| `gui/threading/{__init__,worker}.py` | criar |
| `gui/services/{__init__,base,simulation_service}.py` | criar |
| `apps/__init__.py`, `apps/sim_manager/{__init__,app,main_window}.py` | criar |
| `apps/sim_manager/perspectives/{__init__}.py` + `simulation/{__init__,viewmodel,view}.py` | criar |
| `tests/test_sim_app_skeleton.py` | criar |

## 4. Decisões de Design

| Decisão | Escolha | Justificativa | ADR? |
|:--|:--|:--|:--:|
| VM puro vs Service puro | **VM puro**; Service Qt-touching | reuso de Worker/QThread; pureza crítica é o VM (service injetado) | não |
| threading no skeleton | `Worker` único + `simulate_batch` | pool efêmero é Fatia 3; batch pequeno não precisa | não |
| backend | `numba` fixo | evita JAX/TLS no skeleton; auto/jax = Fatia 5 | não |
| batch | fixo (3-camadas TIV, n_models pequeno) | geração estocástica é Fatia 3 | não |
| deepcopy do payload | metadata sim; H6 grande não (produced-and-released) | sem mutação concorrente no skeleton; documentado | não |

## 5. Riscos Técnicos e Mitigações

| Risco | Prob. | Impacto | Mitigação |
|:--|:--:|:--:|:--|
| Qt criado fora da main thread | baixa | alto | Worker SÓ emite sinais; View/QWidget só na main; AC-3 |
| sinal cross-thread sem QueuedConnection | média | médio | connect default = auto (Queued entre threads); slot no main |
| thread/worker coletado pelo GC mid-run | média | alto | BaseService guarda refs `(thread, worker)` |
| Numba JIT lento no teste | média | baixo | 1 smoke; aceito |
| VM importar Qt sem querer | baixa | médio | service injetado; teste de fronteira (AC-6) |

## 6. Estratégia de Teste

| Camada | O quê | AC |
|:--|:--|:--|
| puro | VM (validate/run com **service stub**; result_ready) | AC-1 |
| fidelidade | `SimulationService._run_simulation` → `simulate_batch` real → shape H6 | AC-2 |
| threading (xvfb) | `Worker` roda callable trivial → `finished` | AC-3 |
| e2e (xvfb) | perspectiva build + run → result_ready + plot | AC-4 |
| fronteira (subprocess) | VM sem Qt; core não importa gui/apps | AC-6 |
| regressão | SM GUI 16/16 (monólito intocado) | AC-6 |

## 7. GATE-P
- [x] Constituição sem violação; contratos exatos; sem ADR; teste cobre AC-1..AC-6 + fronteira + regressão.
