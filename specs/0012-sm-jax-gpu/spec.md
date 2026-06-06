---
Spec: 0012-sm-jax-gpu
Titulo: JAX GPU no SM MVVM — simulação off-main em SUBPROCESSO (TLS-safe) + seletor de backend
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: simulate_batch  # física intocada; só muda ONDE roda (subprocesso)
Status: planejado
Released-As:
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-06
---

# Spec 0012 — JAX GPU no Simulation Manager MVVM

## 0. Nota de escopo

O app MVVM fixa `backend="numba"` porque inicializar o JAX numa `QThread` estoura o TLS estático do
processo (`_dl_allocate_tls_init`) — a `libgomp` (Numba) não consegue alocar TLS após o init do CUDA.
A spec 0012 destrava o **JAX-GPU** rodando a simulação num **SUBPROCESSO spawn isolado** (o JAX inita
num processo próprio; o processo da GUI nunca importa JAX), com marshaling `Future → VMSignal`. Adiciona
um **seletor de backend** (numba/jax/auto) no VM/View. Física intocada (só muda ONDE roda). É o
"ProcessPool adiado" das Fatias 3/4.

**Validação empírica (RTX A6000, 2026-06-06):** `simulate_batch(jax)` na GPU OK; **paridade JAX-GPU vs
Numba `max|Δ|=4.38e-14` (<1e-12)**; `_run_simulation(jax)` num `ProcessPoolExecutor(mp_context="spawn")`
retorna resultado **bit-idêntico** (`Δ=0`) ao in-process.

## 1. Contexto e Problema

| Estado | Onde | Evidência |
|:--|:--|:--|
| crash TLS JAX-em-QThread | (raiz) | `_dl_allocate_tls_init` ao init CUDA numa QThread; skeleton fixa numba (`sim_request.py`) |
| dispatcher já roteia | `simulation/dispatch.py:233` | `simulate_batch(backend=auto/jax/numba)`; `_jax_gpu_available`, threshold n≥32 |
| padrão subprocesso provado | `simulation/tests/sm_workers.py` (v2.29) + CLI `_exec.py` | ProcessPool efêmero + spawn isola JAX/Numba |
| service hoje (in-thread) | `gui/services/base.py::_run_async` | Worker(QObject)+QThread → VMSignal (numba OK; JAX crasharia) |

## 2. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | `BaseService._run_in_subprocess(fn, *args)`: roda `fn` num `ProcessPoolExecutor(mp_context="spawn", max_workers=1)` (efêmero, 1 sim por vez), reusando o Worker/QThread só p/ BLOQUEAR no `future.result()` — o processo da GUI NÃO importa JAX. Resultado/erro via os VMSignal existentes (`finished`/`error`) | Must | NOVO |
| RF-2 | `SimulationService.run(request)`: despacha por backend — `numba` → `_run_async` (in-thread, caminho atual); `jax`/`auto` → `_run_in_subprocess` (TLS-safe) | Must | dispatch |
| RF-3 | `SimulationViewModel`: property `backend` (∈ {numba, jax, auto}) + validação; `run()` passa ao `SimRequest`. `SimulatorView`: combo "Backend:" | Must | evolução |
| RF-4 | Paridade física: JAX-GPU vs Numba (mesmos modelos) `max|Δ| < 1e-12` — teste gated em `_jax_gpu_available()` (skip sem GPU) | Must | fidelidade |
| RF-5 | Testes: subprocess marshaling (e2e jax via Worker→pool→VMSignal), VM backend (validação + run monta SimRequest), caminho numba in-thread INALTERADO | Must | testes |

### Critérios de Aceite
- [ ] **AC-1** (TLS-safe): a SIMULAÇÃO JAX (que INICIALIZA CUDA) roda num SUBPROCESSO spawn (pid ≠ pid da GUI), NÃO numa `QThread` da GUI — evita o crash `_dl_allocate_tls_init`. (NOTA: o *módulo* `jax` já é importado no processo via `opt_einsum`/TF no `import geosteering_ai` — INÓCUO, pois `import jax` ≠ init CUDA, que é lazy. O invariante é não INICIALIZAR CUDA numa QThread, não "não importar jax".)
- [ ] **AC-2** (dispatch): `SimulationService.run(SimRequest(backend="numba"))` usa o caminho in-thread (sem subprocesso); `backend in {jax, auto}` usa o subprocesso. Verificável.
- [ ] **AC-3** (marshaling): e2e — VM real + Service real, `backend="jax"`, `run()` → (subprocesso) → `result_ready`/`results` com `H6` finito correto na main thread.
- [ ] **AC-4** (paridade <1e-12): `simulate_batch(jax)` vs `(numba)` p/ os mesmos modelos: `max|Δ| < 1e-12` (gated em GPU).
- [ ] **AC-5** (VM): `backend` ∈ {numba, jax, auto}; valor inválido reprovado por `validate()`; `run()` monta `SimRequest(backend=...)`; testável com stub sem Qt.
- [ ] **AC-6** (regressão): caminho numba in-thread inalterado (Fatias 2-4 verdes); guard de re-entrância preservado; `_threads` prune preservado.

## 3. RNF

| ID | Requisito | Limite |
|:--|:--|:--|
| RNF-1 | **Fidelidade**: física só `simulate_batch` (<1e-12 JAX/Numba/Fortran); o subprocesso não altera ordem de ops | AC-4 |
| RNF-2 | **TLS-safe**: processo da GUI nunca importa JAX (init isolado no subprocesso spawn) | AC-1 |
| RNF-3 | **Pureza**: `_run_simulation`/`SimRequest` picklable (já são — módulo-nível + frozen dataclass) | declarado |
| RNF-4 | **Sem regressão**: caminho numba in-thread idêntico ao atual | AC-6 |
| RNF-5 | Monólito intocado | — |

## 4. Escopo

### IN
- `gui/services/base.py` (`_run_in_subprocess` + helper `_pool_run`); `gui/services/simulation_service.py` (dispatch);
  `apps/.../viewmodel.py` (backend property+validação); `apps/.../view.py` (combo Backend); `tests/test_sm_jax_gpu.py` (novo).

### OUT (futuro)
- **Pool PERSISTENTE + warmup** (amortizar o spawn+CUDA init ~s por run) → otimização futura (v1 = efêmero, correto).
- **Preflight TLS-safe** (decidir numba-vs-jax em NumPy antes de spawnar, p/ "auto" com batch pequeno não spawnar à toa) → futuro.
- Barra de progresso/cancel do subprocesso → Fatia de paridade (5a).
- Fortran backend na GUI → futuro.

## 5. [NEEDS CLARIFICATION]
- [x] ~~Subprocesso vs init-JAX-no-main?~~ → **RESOLVIDO**: subprocesso spawn (isolamento robusto; padrão v2.29; validado bit-a-bit).
- [x] ~~Pool efêmero vs persistente?~~ → **RESOLVIDO**: efêmero no v1 (simples/correto; persistente = otimização futura).
- [x] ~~start method?~~ → **RESOLVIDO**: `spawn` (evita fork+CUDA+Qt hazards; fork é o default mas é inseguro aqui).

**GATE-S: PASSOU** — 0 marcadores.

## 6. Dependências e Riscos

| Tipo | Item | Impacto/Mitigação |
|:--|:--|:--|
| Dep | 0011a (BaseService/Worker) + dispatcher | evolui o BaseService; reusa simulate_batch |
| Risco | spawn re-importa módulos (lento) + CUDA init por run (~s) | aceitável no v1 (GPU domina em batch grande); pool persistente = futuro |
| Risco | fork+CUDA+Qt = crash | mitigado: `mp_context="spawn"` (processo limpo) |
| Risco | GPU ausente (CI) | paridade/e2e jax gated em `_jax_gpu_available()` (skip) |
| Risco | erro no subprocesso (OOM/driver) | propaga via `error` VMSignal (Worker captura BaseException); UI mostra |

## 7. GATE-S
- [x] 0 marcadores; todo RF→AC; IN/OUT explícito; fidelidade (paridade <1e-12) + TLS-safety fixadas; validação empírica registrada; física intocada.
