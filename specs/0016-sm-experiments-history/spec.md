---
Spec: 0016-sm-experiments-history
Titulo: Fatia 6c — Experimentos & Histórico (.exp.json + snapshots com cache/reload/busca)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: gui/persistence  # sem física; só estado/persistência
Status: implementado
Released-As: v2.59
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-07
---

# Spec 0016 — Fatia 6c — Experimentos & Histórico

## 0. Escopo

Gestão de experimentos (`.exp.json`) + histórico de simulações da sessão (lista com
cache ●/○, busca, reload, info), na *secondary sidebar* do shell Antigravity.
Reusa `gui/persistence`. Modelo de experimento **PURO próprio** (sem acoplar ao
monólito). Sem física.

## 1. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW |
|:--|:--|:--:|
| RF-1 | `ExperimentState`/`SimulationSnapshot` PUROS (dataclasses JSON; to_dict/from_dict forward-compat) — NÃO importam do monólito | Must |
| RF-2 | `ExperimentsService` (Qt): create/load (atomic + `SnapshotPersistThread` async), `LRUPlotCache` p/ bundles de reload, recents (QSettings); VMSignals saved/error/cache_updated | Must |
| RF-3 | `ExperimentsViewModel` (PURO): experiment/snapshots/cache_keys/recents + new/open/save/add_snapshot/clear/mark_out_of_cache/select; VMSignals | Must |
| RF-4 | Painel Histórico (sidebar): toolbar Novo/Abrir/Salvar + busca + lista ●/○ + info + **double-click → recarrega** (`results.set_result(bundle)`); recentes | Must |
| RF-5 | Wire na perspectiva: `result_ready` → snapshot + cache.put(bundle) + add; double-click → cache.get → reabre na galeria OU aviso "fora do cache" | Must |

## 2. Critérios de Aceite

- [x] **AC-1** `.exp.json` roundtrip (to_dict/from_dict; forward-compat ignora chaves novas).
- [x] **AC-2** add_snapshot + cache.put; ●/○ reflete `snap_id in cache` APÓS put (eviction → ○).
- [x] **AC-3** reload: cache.get(snap_id) → bundle → `results.set_result`; ausente → aviso (sem crash).
- [x] **AC-4** busca filtra a lista; recents persiste/dedup (QSettings, top-10).
- [x] **AC-5** ViewModel PURO testável sem Qt; Service async não bloqueia.

## 3. Riscos/guards
- Reload fora do cache → mensagem clara (não crash); recente inválido → remove; "Limpar" → confirmação.

## 4. GATE-S
- [x] 0 marcadores; RF→AC; sem física; reuso de gui/persistence; sem acoplamento ao monólito.
