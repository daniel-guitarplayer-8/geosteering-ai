---
Spec: 0007-gui-persistence
Plano-de: spec.md
Status: planejado
Data: 2026-06-05
---

# Plano 0007 — Persistência gui/ (escrita atômica + SessionDocument) (HOW)

> Pré-requisito: `spec.md` passou GATE-S.

## 1. Gate de Constituição

| Princípio | Aplicável? | Viola? | Como o plano cumpre |
|:--|:--:|:--:|:--|
| I Paridade (3 regimes) | não | não | persistência não toca cálculo EM |
| II Errata imutável | não | não | — |
| III TF/Keras exclusivo | sim | não | sem torch |
| IV Python 3.13 | sim | não | `tempfile`/`os.replace` stdlib |
| V Config-parâmetro | sim | não | `SessionDocument`/funções recebem dados por parâmetro; sem `globals()` |
| VI Logging | sim | não | sem `print()` |
| VII D1–D14 | sim | não | mega-headers + docstrings |
| VIII PT-BR | sim | não | — |
| IX SSoT | sim | não | `Backlog-Code: F-mvc-split` |
| X MVVM | sim | não | `atomic`/`session` PUROS (testáveis sem Qt); `snapshot` é a camada async Qt |
| XI Fundação | sim | não | `gui/persistence` compartilhado; `core` não importa `gui` |
| XII Gates | sim | não | reviewers + suíte + fronteira + crash-resistance |

**GATE-P: sem violação.** Sem ADR (extração + hardening reversíveis; padrão Strangler já ratificado).

## 2. Arquitetura Técnica

```
gui/persistence/
  atomic.py   (PURO)  atomic_write_text(path, text)  ──┐ usado por
  session.py  (PURO)  SessionDocument.save/load  ───────┤ (escrita crash-safe)
  snapshot.py (Qt)    SnapshotPersistThread.run() ──────┘  ← agora ATÔMICO
  plot_cache.py(numpy)LRUPlotCache, default_max_bytes
  __init__.py         re-exporta a API

ESCRITA ATÔMICA (RNF-2):
  tmp = mkstemp(dir=MESMO diretório do destino)   # rename intra-FS (evita EXDEV)
  write(tmp) → flush → fsync(tmp)                  # durabilidade
  os.replace(tmp, destino)                         # troca ATÔMICA POSIX
  (falha → unlink(tmp); destino antigo intacto)
```

`atomic`/`session` PUROS → o estado de UI serializa testável sem `QApplication` (Princípio X).
`snapshot` (QThread) e `plot_cache` são infra; o monólito os consome via shim (migração = 0011).

## 3. Contratos / APIs (assinaturas exatas)

```python
# gui/persistence/atomic.py  (PURO)
def atomic_write_text(path: str | os.PathLike, text: str, *, encoding: str = "utf-8") -> None:
    """Escreve `text` atomicamente (tmp no mesmo dir + fsync + os.replace).
    Garante: ou o arquivo antigo, ou o novo completo — nunca truncado."""

# gui/persistence/session.py  (PURO, sem pickle)
@dataclass
class SessionDocument:
    schema_version: int = 1
    data: dict[str, Any] = field(default_factory=dict)   # estado volátil de UI
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, text: str) -> "SessionDocument": ...   # forward-compat
    def save(self, path) -> None: ...                          # atomic_write_text
    @classmethod
    def load(cls, path) -> "SessionDocument": ...

# gui/persistence/snapshot.py  (Qt — QThread); RF-4 hardening:
#   run(): ANTES open(path,"w").write(json)  →  DEPOIS atomic_write_text(path, json)
#   import: ..sm_qt_compat  →  geosteering_ai.gui.qt_compat
```

## 4. Estrutura de Arquivos

| Arquivo | Ação | Conteúdo |
|:--|:--|:--|
| `gui/persistence/__init__.py` | criar | re-exporta API (`atomic_write_text`, `SessionDocument`, `LRUPlotCache`, `default_max_bytes`, `SnapshotPersistThread`) |
| `gui/persistence/atomic.py` | criar | `atomic_write_text` (puro) |
| `gui/persistence/session.py` | criar | `SessionDocument` (puro, sem pickle) |
| `gui/persistence/plot_cache.py` | `git mv` | de `sm_plot_cache.py` (numpy) |
| `gui/persistence/snapshot.py` | `git mv`+editar | de `sm_snapshot_persist.py`; fix import + usar atômico |
| `simulation/tests/sm_plot_cache.py` | criar (shim) | re-export de `gui.persistence.plot_cache` |
| `simulation/tests/sm_snapshot_persist.py` | criar (shim) | re-export de `gui.persistence.snapshot` |
| `tests/test_gui_persistence.py` | criar | AC-1.x/2.x/3.x/5.x |

## 5. Decisões de Design

| Decisão | Opções | Escolha | Justificativa | ADR? |
|:--|:--|:--|:--|:--:|
| tmpfile dir | /tmp vs **mesmo dir do destino** | mesmo dir | `os.replace` exige mesmo FS (evita `EXDEV` cross-device) | não |
| `SessionDocument` base | herda BaseViewModel vs **dataclass independente** | dataclass | é documento de persistência, não VM; espelha to/from-dict | não |
| atomicidade | embutida no snapshot vs **helper puro reutilizável** | helper | reusável por session/snapshot/.gsproj(0018); testável sem Qt | não |
| fsync | sempre vs **try/except** | try/except | alguns FS não suportam; `os.replace` é o que garante atomicidade | não |
| serialização | JSON vs pickle/YAML | **JSON** | Princípio (sem pickle — RCE); forward-compat trivial | não |

## 6. Riscos Técnicos e Mitigações

| Risco | Prob. | Impacto | Mitigação |
|:--|:--:|:--:|:--|
| `EXDEV` (rename cross-FS) | baixa | alto | tmpfile no MESMO diretório do destino |
| quebrar `test_simulation_lru_cache` | média | médio | shim re-exporta `default_max_bytes`/`LRUPlotCache`; AC-3.4 |
| hardening mudar comportamento do snapshot | baixa | médio | mesma semântica (grava json_text no path); + crash-safe; AC-3.3 |
| `snapshot` puxar Qt no import de `gui.persistence` | média | baixo | `__init__` importa snapshot lazy OU documenta; teste AC-5.1 cobre atomic/session puros |

## 7. Estratégia de Teste

| Camada | O quê | Onde |
|:--|:--|:--|
| puro | `atomic_write_text` grava + cria dir | `tests/test_gui_persistence.py` (AC-1.1) |
| puro (crash) | falha a meio → arquivo antigo intacto, sem `.tmp` resíduo | idem (AC-1.2) |
| puro | `SessionDocument` round-trip + forward-compat + sem pickle | idem (AC-2.x) |
| shim | identidade `LRUPlotCache`/`SnapshotPersistThread` | idem (AC-3.2) |
| hardening | guard: `snapshot.run` usa `atomic_write_text` | idem (AC-3.3) |
| fronteira (subprocess) | `atomic`/`session` sem Qt | idem (AC-5.1) |
| regressão | `test_simulation_lru_cache` + GUI SM | suites existentes (AC-3.4) |

## 8. Critério de Pronto do Plano (GATE-P)
- [x] Constituição sem violação
- [x] Contratos com assinaturas exatas
- [x] Nenhum ADR bloqueante
- [x] Estratégia de teste cobre AC-1.x..AC-5.x + crash-resistance + regressão
