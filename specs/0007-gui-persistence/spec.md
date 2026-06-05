---
Spec: 0007-gui-persistence
Titulo: Fundação de persistência gui/persistence (escrita atômica + SessionDocument + extração snapshot/cache)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: n/a   # persistência de estado de UI; não toca cálculo EM
Status: planejado
Released-As:
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-05
---

# Spec 0007 — `gui/persistence` (.session atômico)

## 1. Contexto e Problema

A fundação `gui/` (0004 qt_compat, 0005 MVVM, 0006 plot_backends) precisa de uma camada de
**persistência de estado de UI** compartilhada por SM e Studio. Hoje a persistência do
Simulation Manager está espalhada e **não é atômica**: `SnapshotPersistThread.run()` faz
`open(path, "w").write(...)` direto — um crash no meio da escrita corrompe o `.session`.

Esta spec faz **duas coisas** (≠ 0006, que foi só extração):
1. **Extrai** (Strangler-Fig) `sm_snapshot_persist.py` + `sm_plot_cache.py` → `gui/persistence/`.
2. **Endurece** (net-new): escrita **atômica** (write-temp → `os.replace`) + um `SessionDocument`
   reutilizável (`.session` JSON, sem pickle, forward-compat) — base do `.gsproj` (0018).

| Estado | Onde | Evidência |
|:--|:--|:--|
| a mover (puro numpy) | `simulation/tests/sm_plot_cache.py` | `LRUPlotCache` + `default_max_bytes` (294 LOC) |
| a mover (QThread) | `simulation/tests/sm_snapshot_persist.py` | `SnapshotPersistThread` (139 LOC) |
| 🐛 não-atômico | `…/sm_snapshot_persist.py:129-133` | comentário admite "write-temp-then-rename seria ideal" mas faz write direto |
| ausente | `geosteering_ai/gui/persistence/` | diretório não existe |
| permanece | `ExperimentState` em `simulation_manager.py:331` | embutido no monólito — migra em **0011** (não nesta spec) |

## 2. User Stories

| ID | Como… | Quero… | Para… | Prioridade |
|:--|:--|:--|:--|:--:|
| US-1 | Dev de app (SM/Studio) | salvar estado de UI em `.session` JSON de forma ATÔMICA | nunca corromper a sessão num crash/disco cheio | Must |
| US-2 | Dev de ViewModel | um `SessionDocument` (puro, sem Qt) com to/from JSON | serializar estado testável sem `pytest-qt` | Must |
| US-3 | Mantenedor do SM | imports legados (`sm_snapshot_persist`, `sm_plot_cache`) intactos | não quebrar o monólito (migração é 0011) | Must |
| US-4 | Dev do Studio | infra de cache/persistência compartilhada em `gui/` | reusar LRU/snapshot sem duplicar | Should |

## 3. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | `gui/persistence/atomic.py`: `atomic_write_text(path, text)` — tmpfile no mesmo dir → `flush`+`fsync` → `os.replace` (atômico); limpa o tmp em falha. PURO (sem Qt) | Must | NOVO |
| RF-2 | `gui/persistence/session.py`: `SessionDocument` (dataclass PURA) — `schema_version` + `data` dict; `to_json`/`from_json` (forward-compat: chaves desconhecidas preservadas; **sem pickle**); `save`/`load` atômicos | Must | NOVO |
| RF-3 | Mover `sm_plot_cache.py` → `gui/persistence/plot_cache.py` (`LRUPlotCache`, `default_max_bytes`) + shim | Must | MOVE |
| RF-4 | Mover `sm_snapshot_persist.py` → `gui/persistence/snapshot.py` (`SnapshotPersistThread`) + shim; fix `..sm_qt_compat`→`gui.qt_compat`; **usar `atomic_write_text`** (hardening) | Must | MOVE+FIX |
| RF-5 | Importar `gui.persistence` (atomic/session) **não** importa Qt (puro); `snapshot` importa Qt (é a camada async) | Must | INVARIANTE |
| RF-6 | Suíte `tests/test_gui_persistence.py` (atômico + crash-resistance + SessionDocument round-trip + shim + fronteira) | Must | NOVO |

### RF-1 — Critérios de Aceite (escrita atômica)
- [ ] **AC-1.1**: `atomic_write_text(p, "abc")` grava `"abc"` em `p` (cria dir pai).
- [ ] **AC-1.2** (atomicidade): se o conteúdo novo falha a meio (simulado), o arquivo ANTIGO permanece intacto (nunca truncado) e nenhum `.tmp-*` resiste.
- [ ] **AC-1.3**: `atomic_write_text` é PURO — importável sem Qt.

### RF-2 — Critérios de Aceite (SessionDocument)
- [ ] **AC-2.1**: `SessionDocument(data={"k": 1}).to_json()` → JSON; `from_json` reconstrói (round-trip).
- [ ] **AC-2.2** (forward-compat): `from_json` de um JSON com chave futura desconhecida **preserva** essa chave (não levanta).
- [ ] **AC-2.3**: `save(path)`/`load(path)` round-trip via escrita atômica; `SessionDocument` importável sem Qt.
- [ ] **AC-2.4** (sem pickle): o módulo não importa `pickle`; serialização é só JSON.

### RF-3/RF-4 — Critérios de Aceite (extração + shim + hardening)
- [ ] **AC-3.1**: `from geosteering_ai.gui.persistence import LRUPlotCache, default_max_bytes, atomic_write_text, SessionDocument` funciona (pacote **Qt-free**); `SnapshotPersistThread` (QThread) via submódulo `gui.persistence.snapshot` (não re-exportado no `__init__`, p/ manter o pacote sem Qt — Princípio X).
- [ ] **AC-3.2** (shim/identidade): `sm_plot_cache.LRUPlotCache is gui.persistence.plot_cache.LRUPlotCache`; idem `SnapshotPersistThread`.
- [ ] **AC-3.3** (hardening): `SnapshotPersistThread.run()` usa `atomic_write_text` (não `open().write()` direto) — guard de fonte.
- [ ] **AC-3.4** (regressão): coberto em DUAS camadas — (a) LOCAL: `test_shim_*_identity` (esta suíte) garante que o shim re-exporta os mesmos objetos; (b) EXTERNO: as suítes existentes `tests/test_simulation_lru_cache.py` (consumidor direto do shim) + `tests/test_simulation_manager_gui.py` continuam verdes (rodadas no GATE-V, não duplicadas aqui).

### RF-5 — Critérios de Aceite (fronteira)
- [ ] **AC-5.1** (subprocess): após `import geosteering_ai.gui.persistence.atomic` e `…session`, **nenhum** de `{PyQt6, PySide6}` em `sys.modules` (puros).
- [ ] **AC-5.2**: `core` (simulation/…) não importa `gui` (hook de fronteira preservado).

## 4. Requisitos Não-Funcionais (RNF)

| ID | Categoria | Requisito | Métrica/Limite |
|:--|:--|:--|:--|
| RNF-1 | Paridade física | **N/A** — persistência não toca cálculo EM | declarado |
| RNF-2 | Durabilidade | escrita atômica (tmp+fsync+`os.replace`) — crash-safe | AC-1.2 |
| RNF-3 | Segurança | **PROIBIDO pickle** (RCE em desserialização não-confiável) | AC-2.4 |
| RNF-4 | Testabilidade (Princípio X) | `atomic`/`session` puros (testáveis sem pytest-qt) | AC-1.3/2.3 |
| RNF-5 | Doc | D1–D14 (headers atualizados; docstrings) | conformes |

## 5. Escopo

### IN
- `gui/persistence/{__init__,atomic,session}.py` (NOVOS — atômico + SessionDocument).
- `git mv` `sm_plot_cache.py`→`plot_cache.py`, `sm_snapshot_persist.py`→`snapshot.py` (+ fix import + usar atômico).
- Shims (módulo) em `simulation/tests/sm_plot_cache.py` + `sm_snapshot_persist.py`.
- `tests/test_gui_persistence.py`.

### OUT (próximas specs)
- **`ProjectDocument` / `.gsproj`** (zip+JSON, lineage, `SecureArchiveReader`) → **0018** (depende desta).
- **Extração de `ExperimentState`** do monólito + fiação do SM ao `SessionDocument`/atomic → **0011**.
- Auto-save com debounce, `QUndoStack.setClean()` no save → spec futura (com o SM sobre gui/).

## 6. [NEEDS CLARIFICATION]
- [x] ~~Mover `ExperimentState` agora?~~ → **RESOLVIDO**: NÃO. Está embutido no monólito (`simulation_manager.py`); extraí-lo é a migração do SM (**0011**). 0007 entrega a infra (atomic + SessionDocument) que 0011 consumirá.
- [x] ~~`SessionDocument` herda `BaseViewModel`?~~ → **RESOLVIDO**: NÃO — é um documento de persistência (dataclass), não um ViewModel. Espelha o padrão `to_dict/from_dict` da 0005, mas é independente (um VM pode serializar-se PARA um SessionDocument).
- [x] ~~Onde a atomicidade vive?~~ → **RESOLVIDO**: helper PURO `atomic_write_text` em `gui/persistence/atomic.py` (reutilizável por session/snapshot/futuro .gsproj).

**GATE-S: PASSOU** — 0 marcadores abertos.

## 7. Dependências e Riscos de Escopo

| Tipo | Item | Impacto |
|:--|:--|:--|
| Dep | spec 0004 (`gui/qt_compat`) | `snapshot.py` (QThread) reusa Qt via `qt_compat` |
| Desbloqueia | spec 0018 (`.gsproj`) | `SessionDocument` + atomic são a base |
| Risco | `os.replace` cross-device (tmp em /tmp, dest em outro FS) | mitigado: tmpfile **no mesmo diretório** do destino (rename intra-FS) |
| Risco | `fsync` lento/indisponível (alguns FS) | baixo: try/except em torno do fsync; replace é o que garante atomicidade |
| Risco | quebrar `tests/test_simulation_lru_cache.py` | mitigado: shim re-exporta `default_max_bytes`/`LRUPlotCache`; AC-3.4 |

## 8. Critério de Pronto da Spec (GATE-S)
- [x] 0 marcadores `[NEEDS CLARIFICATION]`
- [x] Todo RF tem ≥1 AC testável
- [x] Escopo IN/OUT explícito (`.gsproj`/ExperimentState sequenciados p/ 0018/0011)
- [x] `Produtos` e `Converge-Em` declarados
- [x] `Backlog-Code` (`F-mvc-split`) no ROADMAP §0
- [x] Nenhum princípio da CONSTITUTION violado (sem pickle; puros testáveis; fronteira)
