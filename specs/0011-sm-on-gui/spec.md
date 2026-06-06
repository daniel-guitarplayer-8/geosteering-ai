---
Spec: 0011-sm-on-gui
Titulo: SM sobre gui/ — Fase 0 (migração de consumidores + remoção dos 4 shims transitórios)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM]
Converge-Em: n/a   # migração de imports; não toca cálculo EM
Status: planejado
Released-As:
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-05
---

# Spec 0011 — SM sobre `gui/` · **Fase 0: de-shim**

## 0. Nota de escopo (importante)

A spec 0011 canônica (INDEX) é **grande**: casca MVVM `apps/sim_manager/` + ViewModels puros +
Services, estrangulando o monólito de 10.7k LOC (~semanas). **Esta spec executa apenas a FASE 0**:
fazer os consumidores dos 4 módulos já extraídos (0004/0006/0007) importarem de `gui/` **diretamente**
e **remover os 4 shims transitórios**. As Fases 1+ (app MVVM, perspectiva Simulação) ficam para
specs/iterações futuras (0011a/b…). Decisão do usuário (2026-06-05): "de-shim agora".

## 1. Contexto e Problema

As specs 0004/0006/0007 deixaram **4 shims de retrocompat** em `simulation/tests/` (re-exportando de
`gui/`) para não tocar o monólito. As próprias specs declaram: *"remoção do shim → quando o SM migrar
(0011)"*. Esta fase elimina a indireção: 26 import-sites passam a apontar para `gui/`, e os 4 shims
são deletados.

| Shim (deletar) | → destino canônico | Consumidores |
|:--|:--|:--:|
| `sm_qt_compat.py` | `geosteering_ai.gui.qt_compat` | monólito + 8 `sm_*.py` + 4 testes |
| `sm_plot_backends/` | `geosteering_ai.gui.plot_backends` | monólito (4) + testes de shim |
| `sm_plot_cache.py` | `geosteering_ai.gui.persistence.plot_cache` | monólito + `test_simulation_lru_cache` (7) |
| `sm_snapshot_persist.py` | `geosteering_ai.gui.persistence.snapshot` | monólito (2) |

**Limitação consciente:** o monólito continua em `simulation/tests/` e depende de **14 outros
`sm_*.py` não-extraídos** — a inversão de camada profunda persiste para esses (escopo das Fases 1+).

## 2. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | Migrar imports dos 8 `sm_*.py` de produção: `from .sm_qt_compat` → `gui.qt_compat` | Must | edição |
| RF-2 | Migrar os 10 import-sites do monólito (`simulation_manager.py`) p/ os 4 destinos `gui/` | Must | edição |
| RF-3 | Migrar consumidores de teste REAIS (`conftest_qt`, `test_simulation_manager_gui`, `test_simulation_parameters_seed`, `test_simulation_lru_cache`) p/ `gui/` | Must | edição |
| RF-4 | Remover os testes de **shim-identity** obsoletos (foundation/plot_backends/persistence) | Must | remoção |
| RF-5 | **Deletar** os 4 shims (`sm_qt_compat.py`, `sm_plot_backends/`, `sm_plot_cache.py`, `sm_snapshot_persist.py`) | Must | `git rm` |

### Critérios de Aceite
- [ ] **AC-1**: nenhum arquivo (exceto os deletados) contém `sm_qt_compat`/`sm_plot_backends`/`sm_plot_cache`/`sm_snapshot_persist` como **import** (grep limpo). Guards de fonte que mencionam o NOME em string (ex.: `"sm_qt_compat" not in src`) permanecem válidos.
- [ ] **AC-2**: os 4 shims não existem mais no disco.
- [ ] **AC-3** (regressão): suíte GUI do SM `tests/test_simulation_manager_gui.py` 16/16; `test_simulation_lru_cache.py` 7/7; toda a fundação gui/ (foundation/mvvm/plot_backends/persistence) verde — **sem regressão**.
- [ ] **AC-4**: `import geosteering_ai.simulation.tests.simulation_manager` (o monólito) funciona importando de `gui/` diretamente.

## 3. Requisitos Não-Funcionais (RNF)

| ID | Requisito | Limite |
|:--|:--|:--|
| RNF-1 | Paridade física **N/A** — só troca de imports | declarado |
| RNF-2 | Zero mudança de comportamento — imports apontam para os MESMOS objetos (identidade preservada) | regressão |
| RNF-3 | D8: comentários que referenciavam o shim (ex.: `sm_model_gen.py:482`) atualizados | conforme |

## 4. Escopo

### IN
- Migração de 26 import-sites + remoção de 5 grupos de testes de shim-identity + `git rm` dos 4 shims.

### OUT (Fases 1+ da 0011)
- `apps/sim_manager/` (casca MVVM), `SimulationViewModel`, `SimulationService`, perspectiva Simulação.
- Extração dos 14 `sm_*.py` restantes para `gui/`.
- Mover o monólito para fora de `simulation/tests/`.

## 5. [NEEDS CLARIFICATION]
- [x] ~~de-shim vs MVVM completo?~~ → **RESOLVIDO** (usuário): só de-shim (Fase 0) agora.

**GATE-S: PASSOU** — 0 marcadores abertos.

## 6. Dependências e Riscos

| Tipo | Item | Impacto/Mitigação |
|:--|:--|:--|
| Dep | 0004/0006/0007 (módulos em `gui/`) | os destinos existem; migração é troca de path |
| Risco | quebrar import do monólito (multi-line, submódulo) | mitigado: edição precisa + AC-3/AC-4 (regressão roda o SM) |
| Risco | remover teste de shim deixa AC de 0004/0006/0007 sem cobertura | aceito: o shim deixou de existir; a cobertura do destino `gui/` permanece nos testes de API |

## 7. Critério de Pronto (GATE-S)
- [x] 0 marcadores; todo RF com AC; escopo IN/OUT explícito; `Backlog-Code` no ROADMAP §0; nada toca física.
