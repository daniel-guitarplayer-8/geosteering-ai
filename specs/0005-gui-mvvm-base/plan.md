---
Spec: 0005-gui-mvvm-base
Plano-de: spec.md
Status: planejado
Data: 2026-06-05
---

# Plano 0005 — Base MVVM (HOW)

> Pré-requisito: `spec.md` passou GATE-S.

## 1. Gate de Constituição

| Princípio | Aplicável? | Viola? | Como o plano cumpre |
|:--|:--:|:--:|:--|
| I Paridade (3 regimes) | não | não | não toca cálculo EM (`Converge-Em: n/a`) |
| II Errata imutável | não | não | — |
| III TF/Keras exclusivo | sim | não | sem torch |
| IV Python 3.13 | sim | não | — |
| V Config-parâmetro | parcial | não | `AppContext`/VM recebem dados; sem `globals()` |
| VI Logging | sim | não | `logging` (isolamento de callback do VMSignal) |
| VII D1–D14 | sim | não | mega-headers + docstrings + diagrama de camadas |
| VIII PT-BR | sim | não | — |
| IX SSoT | sim | não | `Backlog-Code: F-mvc-split` |
| X MVVM | sim | não | **núcleo da spec**: View⊥ViewModel(puro)⊥Model; VMSignal |
| XI Fundação | sim | não | `gui/` compartilhado; `core` não importa Qt; VM não importa Qt |
| XII Gates | sim | não | reviewers (Trilha F) + GUI suite xvfb |

**GATE-P: sem violação.** Não gera ADR (abstrações novas, reversíveis; alinhadas ao blueprint).

## 2. Arquitetura Técnica

```
┌── VIEW (Qt) ──────────────────────────────────────────────┐  importa Qt
│   MainWindowBase(QMainWindow): host de abas + statusbar    │
│   <subclasse App>  + Perspective.build_view → QWidget      │
└───────┬───────────────────────────────────────▲───────────┘
 binding │ (vm.method)                  notify    │ vm.signal.connect(slot_qt)
┌───────▼───────────────────────────────────────┴───────────┐  PURO (sem Qt)
│   BaseViewModel: props observáveis + VMSignal(changed,...)  │  ◄── testável sem pytest-qt
│   VMSignal: connect/disconnect/emit (lista de callbacks)    │
└───────┬────────────────────────────────────────────────────┘
 chama  │ (config validada)                                   ← Service/Worker = spec futura (0011)
┌───────▼────────────────────────────────────────────────────┐
│   MODEL = biblioteca (PipelineConfig/SimulationConfig/...)   │
└─────────────────────────────────────────────────────────────┘
```

## 3. Contratos / APIs (assinaturas exatas)

```python
# geosteering_ai/gui/viewmodels/signal.py  (PURO — sem Qt)
class VMSignal:
    """Sinal pub/sub puro-Python (lista de callbacks). Marshaling de thread = adapter da View."""
    def connect(self, callback: Callable[..., None]) -> None: ...      # idempotente
    def disconnect(self, callback: Callable[..., None]) -> None: ...   # no-op se ausente
    def emit(self, *args: Any, **kwargs: Any) -> None: ...             # isola exceções de cada cb (loga)
    def clear(self) -> None: ...

# geosteering_ai/gui/viewmodels/base.py  (PURO — sem Qt)
class BaseViewModel:
    changed: VMSignal                  # emitido em qualquer mudança de estado observável
    def to_dict(self) -> dict[str, Any]: ...     # estado serializável (base p/ .session/.gsproj)
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseViewModel": ...
    def _set(self, name: str, value: Any) -> bool: ...   # helper: set + emite changed se mudou

# geosteering_ai/gui/shell/context.py  (mínimo, extensível)
@dataclass
class AppContext:
    """Contexto compartilhado passado às perspectivas. Campos crescem em specs futuras."""
    app_name: str = "Geosteering AI"

# geosteering_ai/gui/shell/perspective.py  (ABC — sem Qt em runtime)
class Perspective(ABC):
    id: str; title: str; icon: str; order: int
    @abstractmethod
    def build_view(self, ctx: "AppContext") -> "QWidget": ...      # QWidget só em TYPE_CHECKING
    @abstractmethod
    def build_viewmodel(self, ctx: "AppContext") -> "BaseViewModel": ...  # PURO
    def on_activate(self) -> None: ...     # default no-op (lazy)
    def on_close(self) -> bool: ...        # default True (sem veto)

# geosteering_ai/gui/shell/main_window_base.py  (Qt — QMainWindow)
class MainWindowBase(QtWidgets.QMainWindow):
    def __init__(self, ctx: AppContext, *, parent=None) -> None: ...
    def add_perspective(self, p: Perspective) -> None: ...   # cria aba; build lazy no on_activate
```

## 4. Estrutura de Arquivos

| Arquivo | Ação | Conteúdo |
|:--|:--|:--|
| `gui/viewmodels/__init__.py` | criar | `__all__ = ["VMSignal", "BaseViewModel"]` |
| `gui/viewmodels/signal.py` | criar | `VMSignal` (puro) |
| `gui/viewmodels/base.py` | criar | `BaseViewModel` (puro) |
| `gui/shell/__init__.py` | criar | `__all__` (Perspective, AppContext, MainWindowBase) |
| `gui/shell/context.py` | criar | `AppContext` |
| `gui/shell/perspective.py` | criar | `Perspective` ABC (sem Qt runtime) |
| `gui/shell/main_window_base.py` | criar | `MainWindowBase` (Qt) |
| `tests/test_gui_mvvm_base.py` | criar | AC-1.x, AC-2.x, AC-3.x (puros) + AC-5.x (xvfb) |

## 5. Decisões de Design

| Decisão | Opções | Escolha | Justificativa | ADR? |
|:--|:--|:--|:--|:--:|
| VMSignal thread-safety | sinal Qt vs **pub/sub puro síncrono** | pub/sub puro | Princípio X (VM testável sem Qt); marshaling no adapter da View | não |
| Perspective importa Qt? | sim vs **TYPE_CHECKING** | TYPE_CHECKING | importável/testável sem Qt (AC-3.2) | não |
| Service/Worker nesta spec? | sim vs **defer p/ 0011** | defer | foco mínimo; threading entra com o 1º service real | não |
| isolamento de exceção em emit | propagar vs **isolar+logar** | isolar+logar | um callback ruim não derruba os demais (AC-1.5) | não |

## 6. Riscos Técnicos e Mitigações

| Risco | Prob. | Impacto | Mitigação |
|:--|:--:|:--:|:--|
| over-design | média | médio | escopo mínimo (4 primitivas); OUT explícito |
| `Perspective` puxar Qt | baixa | médio | `from __future__ import annotations` + `TYPE_CHECKING`; teste AC-3.2 |
| VMSignal memory leak (refs fortes) | baixa | baixo | `disconnect`/`clear` explícitos; documentar (sem weakref no MVP) |
| regressão GUI do SM | muito baixa | alto | arquivos NOVOS; SM não os usa ainda; AC-5.3 roda a suíte |

## 7. Estratégia de Teste

| Camada | O quê | Onde |
|:--|:--|:--|
| puro | VMSignal (connect/emit/disconnect/isolamento) | `tests/test_gui_mvvm_base.py` (AC-1.x) |
| puro | BaseViewModel (changed, to_dict/from_dict) | idem (AC-2.x) |
| puro | Perspective ABC (abstrato; sem Qt) | idem (AC-3.x) — teste roda mesmo sem PyQt |
| Qt (xvfb) | MainWindowBase instancia + add_perspective | idem (AC-5.1/5.2) |
| regressão | suíte GUI do SM 16/16 | `tests/test_simulation_manager_gui.py` (AC-5.3) |
| import-boundary | `viewmodels`/`Perspective` sem Qt; `core` sem `gui` | teste de import |

## 8. Critério de Pronto do Plano (GATE-P)
- [x] Constituição sem violação
- [x] Contratos com assinaturas exatas
- [x] Nenhum ADR bloqueante
- [x] Estratégia de teste cobre AC-1.x..AC-5.x + fronteira de import
