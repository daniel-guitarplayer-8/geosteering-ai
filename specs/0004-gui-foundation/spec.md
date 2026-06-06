---
Spec: 0004-gui-foundation
Titulo: Extração da fundação Qt para geosteering_ai/gui/ (Strangler Fig — qt_compat)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: n/a   # pura relocação de infraestrutura Qt; não toca cálculo EM
Status: implementado
Released-As: v2.57
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-05
---

# Spec 0004 — Fundação `geosteering_ai/gui/` (keystone da Fase 0)

## 1. Contexto e Problema

A infraestrutura Qt do Simulation Manager vive em `geosteering_ai/simulation/tests/` —
**código de produção dentro de `tests/`** (semântica errada). O módulo-fundação
`sm_qt_compat.py` (camada de binding PyQt6/PySide6, locale C, dark-mode) é importado por
**16 lugares** (o monólito `simulation_manager.py`, a maioria dos `sm_*.py` e `tests/`), mas
não existe um pacote `geosteering_ai/gui/` de 1ª classe que o Studio (futuro) e o SM possam
importar como fundação compartilhada. Esta é a **keystone bloqueante da Fase 0** (ver
[docs/architecture/04_ui_ux_mvvm.md](../../docs/architecture/04_ui_ux_mvvm.md) e
[ROADMAP §0](../../docs/ROADMAP.md)).

| Estado | Onde | Evidência |
|:--|:--|:--|
| ausente | `geosteering_ai/gui/` | diretório não existe |
| mal-alocado | `geosteering_ai/simulation/tests/sm_qt_compat.py` (298 LOC) | infra de produção em `tests/` |
| precedente | `simulation/tests/sm_io.py` | já é shim re-exportando de `simulation/io/tensor_dat.py` (v2.53) |

**Padrão a aplicar (Strangler Fig, idêntico ao precedente `sm_io.py`):** relocar a implementação
para o pacote de produção `geosteering_ai/gui/`, deixando um **shim de retrocompatibilidade** no
caminho antigo. Os 16 importadores (`from .sm_qt_compat import ...`, imports RELATIVOS) continuam
funcionando **sem alteração**.

## 2. User Stories

| ID | Como… | Quero… | Para… | Prioridade |
|:--|:--|:--|:--|:--:|
| US-1 | Dev do Studio | importar `from geosteering_ai.gui.qt_compat import ...` | construir a casca MVVM do Studio sobre a fundação compartilhada | Must |
| US-2 | Mantenedor do SM | que o Simulation Manager continue funcionando idêntico | não introduzir regressão na GUI existente | Must |
| US-3 | Dev da plataforma | instalar a fundação GUI via extra pip `[gui]` | empacotar SM/Studio sem puxar Qt no `core` | Should |

## 3. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | Criar pacote `geosteering_ai/gui/` (1ª classe) com `qt_compat` relocado | Must | NOVO |
| RF-2 | `simulation/tests/sm_qt_compat.py` vira shim re-exportando de `gui.qt_compat` | Must | NOVO |
| RF-3 | Os 16 importadores legados continuam funcionando sem alteração | Must | NOVO (verificar) |
| RF-4 | Extra pip `[gui]` (PyQt6) declarado em `pyproject.toml` | Should | NOVO |

### RF-1 — Critérios de Aceite
- [ ] **AC-1.1**: `from geosteering_ai.gui.qt_compat import QtCore, QtWidgets, Signal, QT_BINDING` funciona.
- [ ] **AC-1.2**: `geosteering_ai/gui/__init__.py` existe e o pacote é importável.
- [ ] **AC-1.3**: `gui/qt_compat` expõe os 16 nomes de `__all__` (QtCore, QtGui, QtWidgets, Qt, Signal, Slot, QThread, QObject, QT_AVAILABLE, QT_BINDING, QT_IMPORT_ERROR, check_qt_available, detect_os_dark_mode, enforce_c_locale, format_float, make_double_spin).

### RF-2/RF-3 — Critérios de Aceite (retrocompat — CRÍTICO)
- [ ] **AC-2.1**: `from geosteering_ai.simulation.tests.sm_qt_compat import QtCore` ainda funciona.
- [ ] **AC-2.2**: Os objetos re-exportados são os **MESMOS** (`gui.qt_compat.QtCore is sm_qt_compat.QtCore`, idem Signal/QThread/QT_BINDING).
- [ ] **AC-2.3**: A suíte GUI `tests/test_simulation_manager_gui.py` PASSA sob xvfb (sem regressão).
- [ ] **AC-2.4**: O Simulation Manager importa (`python -c "import geosteering_ai.simulation.tests.simulation_manager"`) sem erro.

### RF-4 — Critérios de Aceite
- [ ] **AC-4.1**: `pyproject.toml` declara extra `[gui]` com `PyQt6>=6.6` (binding default; `qt_compat` também aceita PySide6).

## 4. Requisitos Não-Funcionais (RNF)

| ID | Categoria | Requisito | Métrica/Limite |
|:--|:--|:--|:--|
| RNF-1 | Paridade física | **N/A** — não toca cálculo EM | declarado (`Converge-Em: n/a`) |
| RNF-2 | Retrocompat | zero alteração nos 16 importadores | imports relativos preservados via shim |
| RNF-3 | Doc | D1–D14 no `gui/qt_compat` + shim + `gui/__init__` | mega-header + docstrings |
| RNF-4 | Plataforma | Python 3.13; `core` NUNCA importa Qt | `gui/` é extra opcional `[gui]` |
| RNF-5 | Histórico git | preservar histórico do arquivo movido | `git mv` |

## 5. Escopo

### IN
- Pacote `geosteering_ai/gui/` + `gui/__init__.py` + `gui/qt_compat.py` (relocado de `sm_qt_compat.py`).
- Shim de retrocompat em `simulation/tests/sm_qt_compat.py`.
- Extra `[gui]` em `pyproject.toml`.
- Teste novo `tests/test_gui_foundation.py` (AC-1.x, AC-2.x).

### OUT (explicitamente — próximas specs)
- MVVM base (`VMSignal`, `MainWindowBase`, `Perspective` ABC) → **spec 0005**.
- `gui/plot_backends/` (PlotCanvas ABC + 4 backends) → **spec 0006**.
- `gui/persistence/` (.session atômico) → **spec 0007**.
- Mover widgets (`sm_widgets`, `sm_toast`, `sm_animation_bar`), threading (`sm_workers`), o monólito → futuro.

## 6. [NEEDS CLARIFICATION]
- [x] ~~Binding default do `[gui]`~~ → **RESOLVIDO**: `PyQt6` (já o default do projeto, em `[dev]`); `qt_compat` detecta PyQt6→PySide6. A decisão comercial PySide6 (LGPL) para o Studio é ADR futuro (D-A do blueprint), não bloqueia 0004.

**GATE-S: PASSOU** — 0 marcadores abertos.

**Desvios registrados na implementação (orgânicos, pós-revisão):**
- `detect_os_dark_mode` (em `gui/qt_compat.py`) recebeu `bool(...)` no retorno — Qt6 é
  não-tipado (`lightness()` é `Any`), o cast explícito mantém o módulo-fundação mypy-clean.
  Comportamento idêntico (`lightness() < 128`).
- Mensagem de `check_qt_available` generalizada de `[Simulation Manager]` para
  `[Geosteering AI GUI]` (módulo agora COMPARTILHADO por SM + Studio — Princípio X/XI).

## 7. Dependências e Riscos de Escopo

| Tipo | Item | Impacto |
|:--|:--|:--|
| Dep | spec 0001 (SDD bootstrap) | processo vigente |
| Risco | quebrar imports relativos dos 16 importadores | mitigado: shim re-exporta (precedente `sm_io.py`); AC-2.x valida |
| Risco | import circular `gui` ↔ `simulation` | baixo: `qt_compat` só importa `sys`/`typing` (autocontido) |
| Risco | GUI tests dependem do caminho antigo | mitigado: shim mantém o caminho; AC-2.3 roda a suíte |

## 8. Critério de Pronto da Spec (GATE-S)
- [x] 0 marcadores `[NEEDS CLARIFICATION]`
- [x] Todo RF tem ≥1 AC testável
- [x] Escopo IN/OUT explícito (0005/0006/0007 sequenciados)
- [x] `Produtos` e `Converge-Em` declarados
- [x] `Backlog-Code` (`F-mvc-split`) existe no ROADMAP §0
- [x] Nenhum princípio da CONSTITUTION violado (Princípio X/XI — fundação `gui/` compartilhada)
