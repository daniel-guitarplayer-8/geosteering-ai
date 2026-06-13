# Sprint em execução

> Este arquivo contém o plano detalhado da sprint **em execução**.
> Após o merge, deve ser renomeado para snapshot imutável (convenção:
> `v2.X.md`) e este arquivo fica vazio.
>
> **Fonte canônica do backlog:** [docs/ROADMAP.md §0](../ROADMAP.md) (SSoT, ADR-0001).

---

## Iniciativa ativa — `F-mvc-split` (SM MVVM, Strangler Fig)

| Campo | Valor |
|:------|:------|
| **Code** | `F-mvc-split` |
| **Trilha** | F (MVC split do Simulation Manager) |
| **Prioridade** | **P1** · Status **IN_PROGRESS** (v2.57+) |
| **Branches recentes** | `feat/gui-0018` (mergeada via PR #47) · `feat/recover-archive-v258` (PR #48, infra) |
| **Referência de estado** | [docs/reports/v2.58_estado_4_produtos_2026-06-10.md](../reports/v2.58_estado_4_produtos_2026-06-10.md) |

### Contexto (atualizado 2026-06-10)

A construção do **Simulation Manager MVVM** (`apps/sim_manager/` + `geosteering_ai/gui/`),
por Strangler Fig em paralelo ao monólito intocado, é a frente P1 em execução. Já landaram na
`main`: fundação `gui/` (specs 0004-0007), Fatias **6a-6d** (execução/cancel, geologia+Hankel,
experimentos+histórico, galeria+geosinais), shell Antigravity, JAX-GPU em subprocesso TLS-safe,
e o **Lote 1+2** (paralelismo Numba/Fortran, física canônica tj/h1, BottomBar, scroll).

**Recém-entregue (v2.58, em revisão no PR #48):** recuperação do `archive/wip-stash-gui0013`
(reassembly vetorizado −49% pico de memória, flags ricas do `benchmark`, cache XLA estável +
`sm_io` shim, CHANGELOG SSoT). Infra geral (lib/CLI/docs) — não-GUI.

> A antiga **Sprint O0 (JAX GPU pre-flight, 2026-05-24)** está **CONCLUÍDA/SUPERSEDED**: a trilha
> A (JAX GPU) landou integralmente em v2.42-v2.48 (`A-jax-gpu-batched-api`, `A-jax-gpu-dispatcher`,
> agrupamento por geometria) — todos `DONE` no ROADMAP §0. O conteúdo foi arquivado deste CURRENT.

### Paridade restante — Fatias 6e-6i

| Fatia | Conteúdo | Análogo no monólito |
|:-----:|:---------|:--------------------|
| **6e** | Preferências (tema, paths, backend de plot, limites de cache LRU) | `PreferencesPage` |
| **6f** | Aba Benchmark (cenários A-H + tabela + CSV export) | `BenchmarkPage` |
| **6g** | Plot composer (grid de subplots, export PNG/PDF/EPS, DPI) | `PlotComposerDialog` |
| **6h** | DAT viewer (carregar `.dat/.out` sem re-simular) | `sm_dat_viewer.py` |
| **6i** | Perspectiva Resultados dedicada | `ResultsPage` |

### Próximo candidato (recomendado)

**Fatia 6e — Preferências.** Menor superfície, desbloqueia configuração persistente (tema,
paths, backend de plot) reutilizada pelas demais fatias. Gate por-fatia: paridade de
comportamento vs monólito + pytest-qt headless + `ruff/format/mypy` limpos +
**paridade física `<1e-12` preservada** (a GUI só consome `simulate_batch`, nunca recalcula).

### Invariantes inegociáveis (toda fatia)

1. **Física só via `dispatch.simulate_batch`** — a GUI nunca reimplementa cálculo EM.
2. **ViewModel puro** (sem import de Qt) — testável headless.
3. **Monólito intocado** — coexistência Strangler Fig; nenhuma regressão no SM de produção.
4. **CLI/GUI não se misturam** — fixes de lib/CLI vão p/ `main` + merge nas feature branches.

---

*Alinhado com ADR-0001. As versões `vX.Y` são atribuídas no primeiro commit de cada fatia/sprint;
o backlog priorizado vive em [docs/ROADMAP.md §0](../ROADMAP.md).*
