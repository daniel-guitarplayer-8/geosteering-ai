---
Spec: 0004-gui-foundation
Plano-de: spec.md
Status: planejado
Data: 2026-06-05
---

# Plano 0004 — Fundação `geosteering_ai/gui/` (HOW)

> Pré-requisito: `spec.md` passou GATE-S.

## 1. Gate de Constituição

| Princípio | Aplicável? | Viola? | Como o plano cumpre |
|:--|:--:|:--:|:--|
| I Paridade (3 regimes) | não | não | **não toca cálculo EM** (`Converge-Em: n/a`) |
| II Errata imutável | não | não | nenhuma constante física |
| III TF/Keras exclusivo | sim | não | sem torch |
| IV Python 3.13 | sim | não | — |
| V Config-parâmetro | n/a | — | módulo de infra Qt (sem PipelineConfig) |
| VI Logging | sim | não | `sys.stderr.write` em `check_qt_available` é contrato de CLI (abort) — preservado |
| VII D1–D14 | sim | não | mega-header no `gui/qt_compat`, shim e `gui/__init__` |
| VIII PT-BR | sim | não | acentuação |
| IX SSoT | sim | não | `Backlog-Code: F-mvc-split` |
| X MVVM | sim | não | estabelece a fundação `gui/` (View⊥ViewModel⊥Model virá em 0005) |
| XI Fundação | sim | não | `gui/` é a fundação compartilhada SM/Studio; `core` não importa Qt |
| XII Gates/Scaler | sim | não | reviewers (Trilha F) + GUI tests |

**GATE-P: sem violação.** **Não gera ADR** (relocação mecânica reversível). Nota: o ADR de
**binding Qt default** (PyQt6 vs PySide6, D-A do blueprint) é decisão FUTURA (Studio comercial),
não bloqueia esta relocação — `qt_compat` já suporta ambos.

## 2. Arquitetura Técnica

```
ANTES                                      DEPOIS (Strangler Fig)
─────────────────────────────             ─────────────────────────────────────────────
simulation/tests/                          geosteering_ai/gui/            ◄── NOVO pacote
  sm_qt_compat.py  (impl, 298 LOC)           __init__.py
        ▲ from .sm_qt_compat                  qt_compat.py  (impl, movido via git mv)
        │ (16 importadores)                          ▲
  sm_*.py, simulation_manager.py,                    │ re-export
  tests/conftest*.py                         simulation/tests/
                                               sm_qt_compat.py  (SHIM ← re-exporta gui.qt_compat)
                                                     ▲ from .sm_qt_compat  (16 importadores INALTERADOS)
                                               sm_*.py, simulation_manager.py, tests/conftest*.py
```

- **Detecção de binding roda no IMPORT** de `gui/qt_compat` (linhas 129-137: `_try_pyqt6()`/
  `_try_pyside6()` populam os globals). O shim importa `gui.qt_compat` → dispara a detecção UMA vez
  → re-exporta os globals JÁ resolvidos. `from .sm_qt_compat import X` (importadores) pega o mesmo objeto.
- **`qt_compat` é autocontido** (só importa `sys`, `typing`) → move-se sem ajuste de imports internos.

## 3. Contratos / APIs

```python
# geosteering_ai/gui/__init__.py  (NOVO)
"""Pacote de fundação GUI (infra Qt compartilhada SM + Studio)."""
__all__: list[str] = []   # submódulos importados explicitamente (qt_compat, ...)

# geosteering_ai/gui/qt_compat.py  (movido de sm_qt_compat.py; conteúdo idêntico + header novo)
__all__ = [16 nomes]   # QtCore, QtGui, QtWidgets, Qt, Signal, Slot, QThread, QObject,
                       # QT_AVAILABLE, QT_BINDING, QT_IMPORT_ERROR, check_qt_available,
                       # detect_os_dark_mode, enforce_c_locale, format_float, make_double_spin

# geosteering_ai/simulation/tests/sm_qt_compat.py  (SHIM — substitui o conteúdo)
from geosteering_ai.gui.qt_compat import (  # re-export retrocompat (precedente sm_io.py)
    Qt, QObject, QThread, QT_AVAILABLE, QT_BINDING, QT_IMPORT_ERROR,
    QtCore, QtGui, QtWidgets, Signal, Slot,
    check_qt_available, detect_os_dark_mode, enforce_c_locale,
    format_float, make_double_spin,
)
__all__ = [ ...mesmos 16... ]
```

## 4. Estrutura de Arquivos

| Arquivo | Ação | Conteúdo |
|:--|:--|:--|
| `geosteering_ai/gui/__init__.py` | criar | pacote (mega-header D1, `__all__=[]`) |
| `geosteering_ai/gui/qt_compat.py` | **`git mv`** de `sm_qt_compat.py` | impl relocada; header atualizado (lar canônico) |
| `geosteering_ai/simulation/tests/sm_qt_compat.py` | reescrever | **shim** re-exportando de `gui.qt_compat` |
| `pyproject.toml` | modificar | extra `[gui]` (PyQt6); `package-data`/`find` já cobrem `geosteering_ai*` |
| `tests/test_gui_foundation.py` | criar | AC-1.x (novo caminho) + AC-2.x (retrocompat + identidade) |

## 5. Decisões de Design / ADRs

| Decisão | Opções | Escolha | Justificativa | ADR? |
|:--|:--|:--|:--|:--:|
| forma do shim | `sys.modules` alias vs **re-export explícito** | re-export explícito | precedente `sm_io.py`; ruff-clean; sem magia | não |
| mover qt_compat via | cópia vs **`git mv`** | `git mv` | preserva histórico (RNF-5) | não |
| escopo 0004 | só qt_compat vs +widgets/plot | **só qt_compat** | keystone cirúrgica; menor risco; 0005-0007 estendem | não |
| binding default `[gui]` | PyQt6 vs PySide6 | PyQt6 (agora) | já é o default do projeto; PySide6 comercial = ADR futuro D-A | não (ADR futuro) |

## 6. Riscos Técnicos e Mitigações

| Risco | Prob. | Impacto | Mitigação |
|:--|:--:|:--:|:--|
| shim não re-exporta um nome usado | baixa | alto | re-export EXPLÍCITO dos 16 `__all__`; AC-2.x + identidade (`is`) |
| import circular gui↔simulation | muito baixa | alto | `qt_compat` só importa sys/typing; `gui/__init__` não importa simulation |
| GUI tests quebram | baixa | alto | rodar `tests/test_simulation_manager_gui.py` sob xvfb (AC-2.3) |
| `git mv` + working tree sujo (0003 flutuante) | baixa | baixo | arquivos da 0004 são disjuntos (gui/ + sm_qt_compat + pyproject + test) |

## 7. Estratégia de Teste

| Camada | O quê | Onde |
|:--|:--|:--|
| import (novo) | `gui.qt_compat` expõe os 16 nomes | `tests/test_gui_foundation.py` (AC-1.x) |
| retrocompat | shim re-exporta + IDENTIDADE (`is`) | idem (AC-2.1, AC-2.2) |
| regressão GUI | suíte pytest-qt sob xvfb | `tests/test_simulation_manager_gui.py` (AC-2.3) |
| smoke | `import ...simulation_manager` sem erro | idem (AC-2.4) |
| paridade física | **N/A** (não toca EM) | — |
| lint/types | ruff + mypy nos arquivos novos | CI |

## 8. Critério de Pronto do Plano (GATE-P)
- [x] Tabela de constituição sem violação
- [x] Contratos (shim + __all__) exatos
- [x] Nenhum ADR bloqueante (binding default é ADR futuro, não-bloqueante)
- [x] Estratégia de teste cobre AC-1.x/AC-2.x + regressão GUI
