---
Spec: 0011a-sm-app-skeleton
Titulo: SM app MVVM — Fatia 1 (walking skeleton: apps/sim_manager + gui/services + gui/threading)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: simulate_batch  # SÓ orquestra; física converge em multi_forward.py (intocada)
Status: planejado
Released-As:
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-05
---

# Spec 0011a — Walking skeleton do app MVVM (Fatia 1 da 0011)

## 0. Nota de escopo

A 0011 canônica (app MVVM completo do SM) é um épico multi-fatia. Esta spec entrega a **Fatia 1 —
walking skeleton**: o caminho vertical MÍNIMO end-to-end que **prova toda a pilha MVVM**
(MainWindowBase + Perspective + VMSignal + ViewModel puro + Service + threading + plot_backends),
de-riscando o resto do épico. Fatias 2-5 (params completos, geração de modelos, ResultsView, JAX GPU)
são specs/iterações futuras. O **monólito permanece 100% funcional** — `apps/` é entry-point paralelo.

## 1. Contexto e Problema

A fundação `gui/` (0004-0007) provê casca MVVM, plot_backends e persistência, mas **não há app real
usando-a** nem a camada de **Service/threading** (que não existe em `gui/`). Sem um vertical end-to-end,
o épico 0011 fica sem prova de conceito e alto risco.

| Estado | Onde | Evidência |
|:--|:--|:--|
| ausente | `geosteering_ai/gui/services/` | diretório não existe |
| ausente | `geosteering_ai/gui/threading/` | diretório não existe |
| ausente | `apps/sim_manager/` | diretório não existe |
| pronto (reusar) | `simulate_batch` | `dispatch.py:233` (rota física <1e-12, TLS-safe) |

## 2. User Stories

| ID | Como… | Quero… | Para… | Prioridade |
|:--|:--|:--|:--|:--:|
| US-1 | Usuário do SM | abrir o app MVVM, setar 2 params, clicar **Run** e ver 1 plot do resultado | confirmar que a arquitetura MVVM funciona de ponta a ponta | Must |
| US-2 | Dev de ViewModel | `SimulationVM` PURO (sem Qt) testável | iterar lógica sem `pytest-qt` | Must |
| US-3 | Dev de app | uma camada `Service` que roda simulação **off-thread** e emite `VMSignal` | UI não congela; resultado marshalado à main thread | Must |

## 3. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | `gui/threading/`: `Worker(QObject)` + `WorkerSignals` — roda um callable off-thread, emite `finished(object)`/`error(str)` | Must | NOVO |
| RF-2 | `gui/services/`: `BaseService` (mantém `VMSignal`s + `_run_async(fn)`) | Must | NOVO |
| RF-3 | `gui/services/SimulationService` + `SimRequest`: `run(req)` constrói batch pequeno + chama `simulate_batch(backend="numba")` off-thread + emite `finished(result)`/`error`. **NÃO toca física** (só orquestra) | Must | NOVO |
| RF-4 | `apps/sim_manager/.../SimulationViewModel` PURO: estado de params + `validate()` + `run()` (delega a um service INJETADO); emite `changed`/`result_ready` | Must | NOVO |
| RF-5 | `apps/sim_manager/.../SimulatorView` (QWidget): inputs + Run + status + `PlotCanvas`; faz binding aos `VMSignal` do VM | Must | NOVO |
| RF-6 | `SimulationPerspective(Perspective)` + `SM_MainWindow(MainWindowBase)` + `app.py` (entry-point) | Must | NOVO |
| RF-7 | Testes: VM puro + Service fidelity-smoke (shape do H-tensor real) + threading + perspectiva offscreen + fronteira | Must | NOVO |

### Critérios de Aceite
- [ ] **AC-1** (VM puro): `SimulationViewModel` instancia e roda `validate()`/`run()` com um **service stub** SEM Qt (`pytest` puro); `run()` chama `service.run(request)` com os params; `result_ready` emite ao `service.finished`.
- [ ] **AC-2** (fidelidade): `SimulationService` constrói um batch válido e `simulate_batch(backend="numba")` retorna H6 com shape `(n_models, nTR, nAng, n_pos, nf, 9)` complex, finito. (Física intocada — só verifica shape/dtype.)
- [ ] **AC-3** (threading): `Worker` roda um callable em outra thread e emite `finished` (verificável com um callable trivial, sob xvfb).
- [ ] **AC-4** (end-to-end, offscreen): a `SimulationPerspective` constrói View+VM sob `MainWindowBase`; disparar `run()` produz `result_ready` e a View plota (smoke).
- [ ] **AC-5** (TLS-safe): o skeleton usa `backend="numba"` (batch pequeno) → JAX **não** inicializa → sem risco `_dl_allocate_tls_init`.
- [ ] **AC-6** (fronteira): `SimulationViewModel` importável **sem Qt**; `core` (simulation/models/…) não importa `gui`/`apps`; suíte GUI do SM 16/16 (sem regressão — monólito intocado).

## 4. Requisitos Não-Funcionais (RNF)

| ID | Requisito | Limite |
|:--|:--|:--|
| RNF-1 | **Paridade física intocada** — Service só chama `simulate_batch`; não copia kernel nem muda ordem de ops | <1e-12 preservado por construção |
| RNF-2 | ViewModel PURO (Princípio X) — testável sem `pytest-qt` (service injetado) | AC-1 |
| RNF-3 | Threading: zero Qt fora da main thread; `VMSignal` marshalado à main via `QueuedConnection`; refs anti-GC | AC-3 |
| RNF-4 | Monólito intocado (`apps/` paralelo) — regressão impossível por construção | AC-6 |
| RNF-5 | D1-D14 + PT-BR | conformes |

## 5. Escopo

### IN
- `gui/threading/{__init__,worker}.py`; `gui/services/{__init__,base,simulation_service}.py`.
- `apps/{__init__}.py`, `apps/sim_manager/{__init__,app,main_window}.py`, `apps/sim_manager/perspectives/simulation/{__init__,viewmodel,view}.py`.
- `tests/test_sim_app_skeleton.py`.

### OUT (fatias futuras)
- ParametersPage completa → VM (freqs/dips/TRs/h1/tj/p_med, validação, n_pos Fortran) — **Fatia 2**.
- Geração de modelos estocásticos + pool efêmero + cancel/pause (envolver `ModelGenerationThread`/`SimulationThread`) — **Fatia 3**.
- ResultsView (galeria, cache, `.session`) — **Fatia 4**.
- Seletor de backend auto/jax/fortran (= spec 0012) — **Fatia 5**.

## 6. [NEEDS CLARIFICATION]
- [x] ~~Service Qt-free ou Qt-touching?~~ → **RESOLVIDO**: Service é L2 (orquestração na GUI) e PODE tocar Qt (gerencia `Worker`/`QThread`); a PUREZA crítica é o **ViewModel** (service injetado por protocolo duck-typed → VM não importa o Service Qt).
- [x] ~~Reusar `SimulationThread`/pool efêmero agora?~~ → **RESOLVIDO**: NÃO no skeleton (batch pequeno, 1 chamada `simulate_batch` num `Worker`); o pool efêmero entra na Fatia 3.

**GATE-S: PASSOU** — 0 marcadores.

## 7. Dependências e Riscos

| Tipo | Item | Impacto/Mitigação |
|:--|:--|:--|
| Dep | 0004/0005/0006 (casca MVVM, plot_backends) | compostos pelo app |
| Risco | reimplementar threading mal (race/Qt off-thread) | mitigado: Worker mínimo + `QueuedConnection` + refs anti-GC + AC-3 |
| Risco | quebrar fidelidade | mitigado: Service só chama `simulate_batch`; AC-2 verifica shape do output REAL |
| Risco | Numba JIT lento no teste de fidelidade | aceito: 1 smoke; CI já paga warmup |

## 8. Critério de Pronto (GATE-S)
- [x] 0 marcadores; todo RF com AC; IN/OUT explícito; `Converge-Em: simulate_batch`; nada toca física diretamente.
