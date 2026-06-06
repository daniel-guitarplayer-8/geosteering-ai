---
Spec: 0005-gui-mvvm-base
Titulo: Base MVVM da fundação GUI (VMSignal · BaseViewModel · Perspective ABC · MainWindowBase)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: n/a   # abstrações de UI; não toca cálculo EM
Status: implementado
Released-As: v2.57
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-05
---

# Spec 0005 — Base MVVM da fundação `geosteering_ai/gui/`

## 1. Contexto e Problema

A spec 0004 estabeleceu o pacote `geosteering_ai/gui/` com `qt_compat`. Falta a **base MVVM**
sobre a qual o Simulation Manager (sobre `gui/`, spec 0011) e o Studio ALPHA (spec 0013) serão
construídos. Sem essas abstrações, ambos os apps reinventariam o binding View↔lógica e a casca.

O **Princípio X** (CONSTITUTION) exige MVVM com **ViewModel testável sem `pytest-qt`** (Python puro,
sem import de Qt). O blueprint [docs/architecture/04_ui_ux_mvvm.md](../../docs/architecture/04_ui_ux_mvvm.md)
§2.2 define o binding por sinal: `VMSignal` (pub/sub puro) que a View adapta a um slot Qt.

| Estado | Onde | Evidência |
|:--|:--|:--|
| ausente | `geosteering_ai/gui/viewmodels/` | diretório não existe |
| ausente | `geosteering_ai/gui/shell/` | diretório não existe |
| pronto | `geosteering_ai/gui/qt_compat.py` | fundação Qt (spec 0004) |

## 2. User Stories

| ID | Como… | Quero… | Para… | Prioridade |
|:--|:--|:--|:--|:--:|
| US-1 | Dev de ViewModel | escrever lógica de UII em Python PURO + notificar a View por `VMSignal` | testar o VM com pytest comum (sem QApplication) | Must |
| US-2 | Dev de app (SM/Studio) | herdar `MainWindowBase` + registrar `Perspective`s | montar a casca sem reescrever menu/abas/tema | Must |
| US-3 | Dev do Studio | um contrato `Perspective` (plugin) uniforme | adicionar perspectivas (Simulação/Treino/…) de forma plugável | Must |

## 3. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | `VMSignal`: pub/sub puro-Python (`connect`/`disconnect`/`emit`), **sem importar Qt** | Must | NOVO |
| RF-2 | `BaseViewModel`: base de VM (Python puro, sem Qt), com `VMSignal`s de mudança | Must | NOVO |
| RF-3 | `Perspective` ABC: contrato de plugin (`id/title/icon/order` + `build_view`/`build_viewmodel`/`on_activate`/`on_close`), importável **sem Qt** (QWidget via `TYPE_CHECKING`) | Must | NOVO |
| RF-4 | `AppContext`: contexto mínimo passado às perspectivas (extensível) | Should | NOVO |
| RF-5 | `MainWindowBase`(`QMainWindow`): host de perspectivas (abas) + statusbar + tema (reusa `qt_compat`) | Must | NOVO |

### RF-1 — Critérios de Aceite (VMSignal — puro)
- [ ] **AC-1.1**: `from geosteering_ai.gui.viewmodels.signal import VMSignal` funciona **sem Qt instalado** (não importa PyQt6/PySide6).
- [ ] **AC-1.2**: `s = VMSignal(); rec=[]; s.connect(rec.append); s.emit(7)` → `rec == [7]`.
- [ ] **AC-1.3**: `s.disconnect(cb)` para de notificar; reconectar o mesmo callback não duplica (idempotente).
- [ ] **AC-1.4**: emitir com múltiplos args repassa-os ao callback (`s.emit(a, b)` → `cb(a, b)`).
- [ ] **AC-1.5**: uma exceção em um callback NÃO impede os demais de receberem (isolamento), e é logada.

### RF-2 — Critérios de Aceite (BaseViewModel — puro)
- [ ] **AC-2.1**: `BaseViewModel` NÃO importa Qt (verificável: importável sem PyQt6).
- [ ] **AC-2.2**: subclasse com uma `property` setável emite o `VMSignal` `changed` ao mudar o valor.
- [ ] **AC-2.3**: `to_dict()`/`from_dict()` (round-trip) preserva o estado serializável (base p/ `.session`).

### RF-3 — Critérios de Aceite (Perspective ABC)
- [ ] **AC-3.1**: `Perspective` é ABC; instanciar uma subclasse SEM `build_view`/`build_viewmodel` levanta `TypeError`.
- [ ] **AC-3.2**: importar `perspective` **não** importa Qt em runtime (QWidget só em `TYPE_CHECKING`).
- [ ] **AC-3.3**: atributos `id/title/icon/order` declarados; `on_activate`/`on_close` têm default não-abstrato.

### RF-5 — Critérios de Aceite (MainWindowBase — Qt)
- [ ] **AC-5.1**: `MainWindowBase` é um `QMainWindow`; `add_perspective(p)` adiciona uma aba (lazy build no `on_activate`).
- [ ] **AC-5.2**: `MainWindowBase` instancia sob xvfb sem erro e expõe `statusBar()`.
- [ ] **AC-5.3**: a suíte GUI existente (`tests/test_simulation_manager_gui.py`) continua 16/16 (sem regressão).

## 4. Requisitos Não-Funcionais (RNF)

| ID | Categoria | Requisito | Métrica/Limite |
|:--|:--|:--|:--|
| RNF-1 | Paridade física | **N/A** (não toca EM) | declarado |
| RNF-2 | Testabilidade (Princípio X) | VMSignal/BaseViewModel/Perspective testáveis SEM pytest-qt | cobertura por pytest puro |
| RNF-3 | Fronteira de import | `viewmodels/` e `Perspective` NÃO importam Qt em runtime; `core` não importa `gui` | hook de import / teste |
| RNF-4 | Doc | D1–D14 (mega-header, docstrings, diagramas ASCII de camada) | conformes |
| RNF-5 | Plataforma | Python 3.13; `MainWindowBase` sob extra `[gui]` | — |

## 5. Escopo

### IN
- `gui/viewmodels/{__init__,signal,base}.py` (`VMSignal`, `BaseViewModel`).
- `gui/shell/{__init__,perspective,context,main_window_base}.py` (`Perspective` ABC, `AppContext`, `MainWindowBase`).
- Testes: `tests/test_gui_mvvm_base.py` (puros) + casos Qt sob xvfb.

### OUT (próximas specs)
- Camada de **Service** (`gui/services/`) + threading `Worker+moveToThread`/`EphemeralProcessRunner` → spec futura (com o 1º service real, **0011**).
- `plot_backends` → **0006**; `persistence/.session` → **0007**.
- Apps concretos (`apps/sim_manager`, `apps/studio`) e perspectivas reais → **0011/0013/0016+**.
- Descoberta de perspectivas por `entry_points` → quando houver perspectivas reais (0013).
- `QUndoStack`/undo-redo → spec futura.

## 6. [NEEDS CLARIFICATION]
- [x] ~~VMSignal e thread-safety (worker→GUI)~~ → **RESOLVIDO**: `VMSignal.emit` é SÍNCRONO (chama callbacks na thread chamadora). O *marshaling* worker→GUI é responsabilidade do **adapter da View** (conecta um slot Qt com `Qt.ConnectionType.QueuedConnection`) ou da camada de Service (spec futura). Documentado no `VMSignal` (Note). Mantém o VM puro/testável.
- [x] ~~`AppContext` agora ou depois~~ → **RESOLVIDO**: definir um `AppContext` MÍNIMO (dataclass extensível) agora, para o contrato `Perspective` ter um tipo; campos crescem em specs futuras (services/project).

**GATE-S: PASSOU** — 0 marcadores abertos.

## 7. Dependências e Riscos de Escopo

| Tipo | Item | Impacto |
|:--|:--|:--|
| Dep | spec 0004 (`gui/qt_compat`) | `MainWindowBase` reusa `qt_compat` |
| Risco | over-design das abstrações | mitigado: escopo mínimo (4 primitivas), sem Service/undo/entry_points |
| Risco | `Perspective` importar Qt e quebrar testabilidade | mitigado: `TYPE_CHECKING` p/ `QWidget`; AC-3.2 valida |
| Risco | `MainWindowBase` regredir a GUI do SM | baixo (arquivo NOVO; SM ainda não o usa); AC-5.3 roda a suíte |

## 8. Critério de Pronto da Spec (GATE-S)
- [x] 0 marcadores `[NEEDS CLARIFICATION]`
- [x] Todo RF tem ≥1 AC testável
- [x] Escopo IN/OUT explícito (Service/0006/0007/apps sequenciados)
- [x] `Produtos` e `Converge-Em` declarados
- [x] `Backlog-Code` (`F-mvc-split`) no ROADMAP §0
- [x] Nenhum princípio da CONSTITUTION violado (Princípio X — MVVM, VM puro)
