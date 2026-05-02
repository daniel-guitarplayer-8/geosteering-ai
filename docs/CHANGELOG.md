# Changelog — Geosteering AI Simulation Manager

Todas as mudanças notáveis do Simulation Manager são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e
o projeto usa [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [2.17] — 2026-05-02

### Fix regressão de oversubscrição em CPUs Hyperthreading/SMT (3× em produção GUI)

Sprint 16 — investigação adicional revela que a regressão de **3× em produção GUI**
remanescente (38k mod/h vs 123k esperado pós-v2.16) era causada por **oversubscrição**:
a fórmula default `spin_threads = max(2, ncpu // (ncpu // 4))` em
[simulation_manager.py](../geosteering_ai/simulation/tests/simulation_manager.py)
produzia **4 workers × 4 threads = 16 threads** em CPU 8C/16T HT, excedendo
**2× os 8 cores físicos** (oversubscrição). O fix v2.16 (threading masking
observable) estabilizou o pool, mas defaults ruins permaneciam.

A v2.17 introduz **detecção de topologia CPU** (físicos vs lógicos) e
**recomendação inteligente** que respeita ``workers × threads ≤ cores físicos``,
eliminando oversubscrição em hardware com Hyperthreading (Intel) ou SMT
(AMD/ARM big-cores).

### Adicionado

- **Sprint 16.1 — `detect_cpu_topology()` portável** ([_workers.py:158-291](../geosteering_ai/simulation/_workers.py)):
  - Detecção em camadas: psutil → sysctl (macOS) → /proc/cpuinfo (Linux) →
    wmic (Windows) → heurística fallback
  - Retorna `(logical_cores, physical_cores, has_hyperthreading)`
  - Cache em variável global `_CPU_TOPOLOGY_CACHE` (topologia não muda em runtime)
  - NUNCA falha — em caso de erro retorna `(logical, logical, False)` (conservador)
- **Sprint 16.2 — `recommend_default_parallelism()`**
  ([_workers.py:294-353](../geosteering_ai/simulation/_workers.py)):
  - Para batch grande (>= 10 modelos): `(phys // 2, 2)` — Modo D híbrido
  - Para single-model: `(1, phys)` — Modo B multi-thread
  - Invariante: `workers × threads ≤ physical_cores` SEMPRE
- **Sprint 16.3 — Warning visual de oversubscrição na GUI**
  ([simulation_manager.py](../geosteering_ai/simulation/tests/simulation_manager.py)):
  - QLabel vermelho aparece quando `workers × threads > physical_cores`
  - Mensagem: "⚠ Oversubscrição: {W} × {T} = {N} threads em {phys} cores físicos"
  - Conectado via `valueChanged` aos spinboxes (atualização em tempo real)
- **Sprint 16.4 — Logging diagnóstico em SimulationThread.run()**
  ([sm_workers.py](../geosteering_ai/simulation/tests/sm_workers.py)):
  - Loga "CPU: N cores físicos · M threads lógicas (HT/SMT)" antes de cada simulação
  - Aviso explícito se `workers × threads > phys_cores`
- **Sprint 16.5 — Smoke tests CPU topology**
  ([tests/test_simulation_cpu_topology.py](../tests/test_simulation_cpu_topology.py)):
  - 19 testes (5 detection + 6 recommendation + 6 simulated hardware + 2 no-regression)
  - Cenários cobertos: Mac Intel 8C/16T, Apple Silicon M1, Linux Xeon 32C, dual-core
- **Tooltips expandidos**: explicam recomendação física vs lógica em todos os spinboxes

### Corrigido

- **Defaults da página Simulação** (`SimulationPage`): de `(4w × 4t = 16)` para
  `(4w × 2t = 8)` em hardware 8C/16T HT (= cores físicos exatos, sem oversubscrição)
- **Defaults da página Benchmark** (`BenchmarkPage`): mesma correção aplicada
  para Numba e Fortran
- **Label de CPU**: agora mostra "8 cores físicos · 16 threads lógicas (HT/SMT)"
  em vez de apenas "16 CPU cores disponíveis"

### Mudado

- **Função pública `detect_cpu_topology()`** exportada em
  [geosteering_ai/simulation/__init__.py](../geosteering_ai/simulation/__init__.py)
  para uso por outros componentes (treinamento DL, etc.)
- **Função pública `recommend_default_parallelism()`** exportada — ponto único
  de verdade para todos os defaults de paralelismo no projeto

### Notas de performance (Hardware Intel 8C/16T HT, macOS)

| Configuração | Threads totais | Cenário E (200 mod, 600 pts) | Speedup |
|---|---|---|---|
| **v2.16 default GUI** (4w × 4t) | 16 (oversub 2×) | 70k mod/h | baseline |
| **v2.17 default GUI** (4w × 2t) | 8 (= phys) | 85k mod/h | **+21%** |
| Cenário E benchmark (4w × 2t) | 8 (= phys) | 85k mod/h | idêntico ✓ |

Em workloads CPU-bound mais intensos (modelos com mais camadas, mais frequências
ou n_pos > 1000) e em arquiteturas com HT mais agressivo (Linux Xeon 32C/64T),
o ganho esperado é **30–50%**.

### Paridade Fortran

- **<1e-12 em 7 modelos canônicos** (Oklahoma 3/5/15/28, Devine 8, Hou 7,
  Viking Graben 10) — zero regressão física

### Validação

- **Pytest CPU topology** (test_simulation_cpu_topology.py):
  19/19 PASS em 1.38s
- **Pytest threading v2.16** (test_simulation_workers_threading.py):
  3/3 PASS em 4.22s (zero regressão)
- **Pytest I/O v2.16** (test_sm_workers_io.py):
  4/4 PASS em 2.73s (zero regressão)
- **Paridade Fortran 7 canônicos**: 7/7 PASS em 126.62s (<1e-12)

### Decisões deferidas

- **Tile/block processing (Sprint 15.6)** → mantido em v2.18 (mais ganho potencial
  agora que oversubscrição foi eliminada)
- **Pré-compute Hankel kernels TE/TM** → v2.18 (10–15% ganho, médio risco)
- **Apple Silicon M1/M2 sem HT**: defaults já corretos (phys = logical)

### Arquivos

**Modificados (4)**:
- `geosteering_ai/simulation/_workers.py` (+193/-0)
- `geosteering_ai/simulation/__init__.py` (+4/-0)
- `geosteering_ai/simulation/tests/simulation_manager.py` (+90/-22)
- `geosteering_ai/simulation/tests/sm_workers.py` (+30/-1)

**Criados (2)**:
- `tests/test_simulation_cpu_topology.py` (~250 LOC, 19 testes)
- `docs/reports/v2.17_2026-05-02.md` (relatório principal)

---

## [2.16] — 2026-05-01

### Fix regressão crítica de threading + Cenário E (production scale 600 pts) + I/O vetorizado

Sprint 15 de correção: identifica e corrige regressão de **4–8×** em
produção GUI introduzida pela combinação dos commits `0f92035` (`try/except
RuntimeError: pass` em `multi_forward.py`) e `e1c8864` (remoção de
`NUMBA_NUM_THREADS` env var nos workers) da v2.15. Adiciona Cenário E
para reproduzir a configuração real de produção (n_positions=600) e
vetoriza `write_dat_from_tensor` (≥3× speedup em I/O).

### Adicionado

- **Sprint 15.2 — Cenário E benchmark (production scale)**:
  - Novo `benchmark_scenario_e()` em [benchmarks/bench_v214_numba.py](../benchmarks/bench_v214_numba.py)
    (+95 LOC) — n_positions=600, single-freq, replica config GUI LWD
  - Novo flag CLI `--n-positions N` (default 30 microbench, 600 production)
  - `--all` agora roda 5 cenários (A–E)
  - Esperado em hardware 8C HT pós-fix: ≥120k mod/h
- **Sprint 15.1 — Smoke tests threading masking**:
  - [tests/test_simulation_workers_threading.py](../tests/test_simulation_workers_threading.py) (~270 LOC)
  - 3 testes: env var inheritance, logger.warning observable,
    simulate_multi em worker respeita num_threads
- **Sprint 15.4 — Smoke tests I/O vetorizado**:
  - [tests/test_sm_workers_io.py](../tests/test_sm_workers_io.py) (~200 LOC)
  - 4 testes: bit-exatness vs loop, z_obs 1D, rho None, performance ≥3×
- **Sprint 15.5 — Relatório técnico n_positions scaling**:
  - [docs/reports/v2.16_n_positions_scaling_analysis_2026-05-01.md](reports/v2.16_n_positions_scaling_analysis_2026-05-01.md)
  - Hot path identificado: `_numba/dipoles.py::hmd_tiv` recursão TE/TM
  - Análise de complexidade O(`n_pos × n_layers × n_filter_pts × n_freqs × n_TR × n_ang`)
  - Top 3 oportunidades documentadas (tile/block, pré-compute kernels, SIMD ufuncs)
- **Marker `slow` em pyproject.toml**:
  - Permite filtrar testes JIT cold-start em CI rápido com `-m 'not slow'`

### Corrigido

- **Sprint 15.1 — Threading masking observable em `multi_forward.py:880-907`**:
  - Substituído `try/except RuntimeError: pass` (silencioso) por
    `logger.warning` com diagnóstico (threads ativas, pool size, exception)
  - Adicionada verificação prévia `if current_active != cfg.num_threads`
    para evitar set redundante
  - Causa raiz da regressão 4–8× em produção GUI (v2.15)
- **Sprint 15.1 — `NUMBA_NUM_THREADS` setado no env do PAI antes do spawn**:
  - [`sm_workers.py::_acquire_numba_pool`](../geosteering_ai/simulation/tests/sm_workers.py)
    (+15 LOC) — workers spawn herdam env, Numba dimensiona pool corretamente
  - [`_workers.py::_acquire_pool`](../geosteering_ai/simulation/_workers.py)
    (+15 LOC) — espelho no pool nativo do core
  - Resultado: workers nascem com pool de threads = `n_threads` (não `cpu_count()`)

### Mudado

- **Sprint 15.4 — `write_dat_from_tensor` vetorizada**:
  - [sm_io.py](../geosteering_ai/simulation/tests/sm_io.py) (+50 / -28 LOC)
  - 5 loops Python aninhados (~1.8M iterações para 600 modelos × 600 pos
    × 1 freq) → broadcast + transpose + reshape NumPy
  - Speedup ≥3× em I/O (validado por `test_write_dat_vectorized_is_faster`)
  - Bit-exatness preservada (validada por 3 testes diferentes em
    `tests/test_sm_workers_io.py`)
- **CLAUDE.md linha 16**:
  - `SM v2.15` → `SM v2.16 (2026-05-01) — fix regressão threading + cenário E (600 pts) + I/O vetorizado`

### Notas de Performance (Hardware Intel 8C/16T HT, macOS)

| Cenário | Pré-fix v2.15 | Pós-fix v2.16 | Speedup |
|:-------:|:-------------:|:-------------:|:-------:|
| A (30 pts, 5k mod) | 753k mod/h¹ | **1.74M mod/h** | 2.31× |
| E (600 pts, 200 mod) | ~30k mod/h² | 86k mod/h | 2.87× |
| E (600 pts, 600 mod) | — | 111k mod/h | — |
| E (600 pts, 2000 mod) | — | **123k mod/h** | ~4.10× |

¹ Reportado em `v2.15_benchmark_hardware_2026-05-01.md`.
² Estimado por escalabilidade linear `n_pos` (753k / 20 ≈ 38k) — coerente com relato GUI 25–38k mod/h.

### Decisões Deferidas

- **Tile/block processing (Sprint 15.6)** — DEFERIDO para v2.17. Cenário E
  pós-fix entrega 4× speedup confirmado; risco/benefício de adicionar
  reordering de prange agora não justifica vs validação de paridade
  Fortran obrigatória em todos os 7 canônicos.
- **Pré-compute Hankel kernels TE/TM** — DEFERIDO para v2.18. Ganho
  estimado 10–15% mas requer revalidação de paridade.
- **SIMD ufuncs** — REJEITADO. Risco de quebrar paridade Fortran <1e-12
  por reordering FMA é inviolável.

### Pytest

- **Total esperado:** 172+ pass (165 v2.15 + 7 novos: 3 threading + 4 I/O)
- **Paridade Fortran:** 10/10 PASS em <1e-12 (zero regressão)
- **Smoke GUI:** 207+ OK (mantido)

---

## [2.15] — 2026-05-01

### Hardware validation, JIT cache observability, code review, fix CI v2.14

Sprint 14 de finalização: corrige CI quebrada do PR #33 (binário Fortran
incompatível em runner Linux), valida ganhos v2.14 em hardware real
(8 cores físicos × 2 threads), expande observability do triple-cache
JAX (bucketed/unified/chunked) e aplica P1 findings do code review
v2.13→v2.14 (zero P0 encontrado).

### Adicionado

- **Sprint 14.0 — Gate CI-safe `_tatu_runnable()`**:
  - Helper em [tests/_fortran_helpers.py](../tests/_fortran_helpers.py) (+73 LOC)
  - Detecta via `subprocess.run` se `tatu.x` é executável no OS atual
    (resolve `OSError [Errno 8] Exec format error` em CI Linux quando
    o repo contém binário macOS commitado)
  - Cache módulo-level memoiza resultado por sessão (sem custo recorrente)
  - 12 testes em `test_simulation_compare_fortran.py` e `test_simulation_multi.py`
    migram de `Path.exists()` para `_tatu_runnable()`
- **Sprint 14.1 — `--threads-per-worker` no benchmark**:
  - Novo arg CLI em [benchmarks/bench_v214_numba.py](../benchmarks/bench_v214_numba.py) (+30 LOC)
  - Helper `_configure_threads()` seta `OMP_NUM_THREADS`,
    `NUMBA_NUM_THREADS`, `MKL_NUM_THREADS` antes do worker pool fork
  - Default `2` (ideal para 4 workers × 2 threads = 8 cores físicos)
  - Total threads = `workers × threads_per_worker` (anti-oversubscription)
- **Sprint 14.3 — `get_jit_cache_info()` expandido**:
  - Função em [forward_pure.py](../geosteering_ai/simulation/_jax/forward_pure.py)
    (+80 LOC) reporta os 3 caches: `bucketed_size`, `unified_size`,
    `chunked_size`, `total_xla_programs`, `estimated_vram_mb`,
    `strategy_distribution`
  - Heurística VRAM: `Σ (3 × n × npt × 16 bytes) / 1024²` por entrada
    (proxy do tensor `common_arrays` shape `(3, n, npt, nf)` complex128)
  - Backward-compat preservada: chaves `n_entries`, `maxsize`, `keys`
    de v1.5.0 continuam apontando para `_BUCKET_JIT_CACHE`
  - 4 novos testes em
    [test_simulation_jax_sprint13.py](../tests/test_simulation_jax_sprint13.py)
    (~180 LOC): empty, after_simulate_unified, vram_estimate, idempotent
- **Relatórios técnicos**:
  - `docs/reports/v2.15_2026-05-01.md` — relatório técnico Sprint 14
  - `docs/reports/v2.15_benchmark_hardware_2026-05-01.md` — 4 cenários
    no hardware do usuário (8 cores físicos × 2 threads × 4 workers)
  - `docs/reports/v2.15_fastmath_dipoles_analysis_2026-05-01.md` —
    decisão técnica de NÃO aplicar fastmath em dipoles.py/kernel.py

### Mudado

- **Sprint 14.4 P1 — code review aplicado**:
  - Eliminado import duplicado de `_simulate_combined_prange` em
    [multi_forward.py](../geosteering_ai/simulation/multi_forward.py):867-876
    (bloco unificado em uma única tupla)
  - Loop triplo z_obs simplificado: O(nTR × nAngles × n_pos) com
    `break` imediato → O(nAngles × n_pos) sem inner loop, semântica
    bit-exata preservada (primeiro TR sempre amostrado)
  - Teste `test_fastmath_propagation_remains_false` agora inspeciona
    `targetoptions` do dispatcher Numba para detectar regressão
    (em vez de smoke + finitude apenas)

### Corrigido

- **CI PR #33 v2.14**:
  `Fortran_Gerador/tatu.x` é binário macOS arm64/x86_64 commitado no
  repositório como artefato de validação local. Em runner Linux x86_64
  o sistema rejeitava com `OSError [Errno 8] Exec format error`,
  fazendo 12 testes Fortran-dependentes falharem na CI mesmo com
  `Path.exists() == True`. Agora `_tatu_runnable()` testa execução
  real e os testes ficam *skipped* (não falham) em ambientes
  incompatíveis. Ver [tests/_fortran_helpers.py](../tests/_fortran_helpers.py).

### Notas técnicas

- **Decisão fastmath dipoles.py/kernel.py — NÃO APLICAR (oficial)**:
  Análise técnica documentada em
  `docs/reports/v2.15_fastmath_dipoles_analysis_2026-05-01.md`.
  Erro FMA acumulado em `hmd_tiv` (~600 ops × 2e-16 ≈ 1.2e-13)
  está em 0.83× do gate Fortran 1e-12 — qualquer ordering reordering
  pode quebrar paridade. `vmd` apresenta ~8e-14. Cancelamento
  catastrófico em recursão TE/TM (`1 + RTEup` com exp quase-iguais)
  amplificaria o erro FMA. Apenas `hankel.py` (4 funções dot-product)
  permanece com fastmath=True (decisão v2.14, validada).
- **Hardware spec testbed**: macOS Darwin 25.4.0 x86_64,
  16 logical cores (hyperthreaded), 8 physical cores, 64 GB RAM
  (Mac Pro / Mac Studio Intel).
- **JIT cache observability — uso recomendado**:
  ```python
  from geosteering_ai.simulation._jax.forward_pure import get_jit_cache_info
  info = get_jit_cache_info()
  print(f"Total XLA programs: {info['total_xla_programs']}")
  print(f"Estimated VRAM: {info['estimated_vram_mb']:.1f} MB")
  ```
  Útil em loops PINN longos (T4 16 GB / A100 40 GB) para detectar
  vazamento de VRAM antes de OOM.

### Testes (zero regressão vs v2.14)

| Suite | Testes | Status |
|:------|:------:|:------:|
| `test_simulation_compare_fortran.py` | 10 | PASS local macOS / SKIP CI Linux |
| `test_simulation_multi.py` (Fortran-deps) | 2 | PASS local / SKIP CI |
| `test_simulation_v213_optimizations.py` | 13 | PASS |
| `test_simulation_v214_prange_combined.py` | 8 | PASS |
| `test_simulation_v214_fastmath.py` | 6 | PASS (com introspecção P1) |
| `test_simulation_jax_sprint13.py` | 4 | PASS (NOVO) |

---

## [2.14] — 2026-05-01

### Otimizações Numba JIT — Sprints 13.3 + 13.4

Implementação das 2 otimizações Numba deferidas de v2.13:
- Sprint 13.3: `prange(n_combos * n_pos)` combinado para colapsar 24 transições
  Python→Numba em loop TR×ângulo serial, eliminando overhead fork/join
- Sprint 13.4: `fastmath=True` seletivo em 4 helpers hankel.py (dot-product FMA-safe)

Benchmark formal v2.14 com 4 cenários (single-freq, multi-freq, multi-TR, PINN).

### Adicionado

- **Sprint 13.3 — prange combinado TR×ângulo**:
  - Nova função `_simulate_combined_prange` em
    [forward.py](../geosteering_ai/simulation/forward.py) (~150 LOC)
    com `@njit(parallel=True, cache=True, nogil=True)`
  - Materialização pré-dispatch em `multi_forward.py`: flat geometry arrays
    `dz_halfs[n_combos]`, `r_halfs[n_combos]`, `cache_indices[n_combos]`
  - Deduplicação de cache via `key_to_idx` mapping — múltiplos combos
    com mesmo hordist compartilham entrada única
  - Stack de caches únicos: `u_unique[n_unique, nf, npt, n]`, idem para 9 arrays
  - Dispatch adaptativo: usa prange quando `n_combos >= 2`, fallback v2.13 para
    single-combo (preserva prange(n_pos) sem paralelismo insuficiente)
  - Nested prange serialização automática Numba: prange(n_total) →
    prange(nf) serializa sem regressão (validado Sprint 13.1)
- **Sprint 13.4 — fastmath seletivo**:
  - `@njit(fastmath=True)` em 4 funções hankel.py:
    `prepare_kr`, `integrate_j0`, `integrate_j1`, `integrate_j0_j1`
  - Justificativa: loops pure dot-product (FMA-safe, erro máx ~2e-14)
  - **INTOCADOS** (fastmath=False obrigatório):
    `propagation.common_arrays`, `common_factors` (recursão TE/TM orden-sensível),
    `dipoles.hmd_tiv`, `vmd` (fatores transmissão antes sums),
    `kernel._fields_in_freqs_kernel_cached`, `precompute_common_arrays_cache`
- **Benchmark CLI**:
  - Novo arquivo `benchmarks/bench_v214_numba.py` (~370 LOC)
  - 4 cenários: A (single-freq 30k mod), B (multi-freq 10 freqs, 30k),
    C (multi-TR×ang 3×5, 5k mod), D (PINN 50 calls, cache_persistent)
  - Métricas: modelos/hora (A/B/C), ms/chamada (D)
  - CLI: `python benchmarks/bench_v214_numba.py --scenario [A|B|C|D|--all]`

### Testes

- **Novo arquivo** `tests/test_simulation_v214_prange_combined.py`:
  - 8 testes Sprint 13.3 — paridade vs v2.13, single-combo fallback,
    multi-TR×multi-ang, deduplicação, Fortran parity <1e-12, determinismo
  - **Todos passam (2.30s)**
- **Novo arquivo** `tests/test_simulation_v214_fastmath.py`:
  - 6 testes Sprint 13.4 — paridade Fortran pós-fastmath (3-layer, oklahoma28),
    determinismo, validação propagation.py=False, smoke zero-regressão
  - **Todos passam (2.25s)**
- **Zero regressão**: 27/27 testes v2.13 + v2.14 PASS (2.45s) — backward-compat total

### Backward-compat

- Prange combinado (Sprint 13.3): fallback automático para v2.13 quando
  `n_combos < 2` ou `parallel=False` ou n_workers=1
- Fastmath (Sprint 13.4): aplicado SOMENTE em hankel.py — dipoles.py,
  propagation.py, kernel.py mantêm fastmath=False (ordem-sensível)
- API `simulate_multi()` 100% compatível: sem novos kwargs, comportamento
  idêntico quando `cache_persistent=False` (default) e `n_workers` não especificado

### Validado

- Paridade v2.13 → v2.14 bit-exata em single-TR×single-ang (fallback path)
- Paridade v2.13 → v2.14 determinística em multi-TR×multi-ang (prange combinado)
- Paridade Fortran <1e-12 em 3-layer + oklahoma28 simulado pós-fastmath hankel
- Thread-safety: prange(n_combos*n_pos) com índices exclusivos por construção
- Nested prange: serialização automática Numba validada (prange(n_total) →
  prange(nf) serializa)

### Pendências (v2.15+)

- Benchmark em hardware do usuário: validar ganhos reais vs v2.12
  (meta v2.13 Sprint 13.1: ≥1.5× multi-freq; meta v2.14 Sprint 13.3: ≥1.3× multi-TR)
- Fastmath=True seletivo em outras operações (dipoles.py factors, kernel.py,
  conforme validação Fortran <1e-12)
- Análise JIT caching: monitorar número de compilações XLA para Numba/JAX
  no backend híbrido

### Modificado

| Arquivo | Mudanças |
|:--------|:---------|
| `forward.py` | +`_simulate_combined_prange` (~150 LOC) |
| `multi_forward.py` | Materialização pré-dispatch + dispatch adaptativo (~100 LOC) |
| `_numba/hankel.py` | `@njit(fastmath=True)` em 4 funções (+4 LOC) |
| `tests/test_simulation_v214_prange_combined.py` | NOVO (8 testes, ~270 LOC) |
| `tests/test_simulation_v214_fastmath.py` | NOVO (6 testes, ~180 LOC) |
| `benchmarks/bench_v214_numba.py` | NOVO (4 cenários, ~370 LOC) |

---

## [2.13] — 2026-05-01

### Otimizações Numba JIT — Sprints 13.1 + 13.2 + 13.4

Implementação parcial das otimizações Numba previstas no relatório técnico
[v2.11_simulador_python_analise_paralelismo_2026-04-30.md §1.3 + §6.2](reports/v2.11_simulador_python_analise_paralelismo_2026-04-30.md).
Sprints 13.1, 13.2 e 13.4 entregues nesta release; Sprint 13.3 (prange combinado
TR×ângulo) e `fastmath=True` seletivo deferidos para v2.14 devido a complexidade
de refatoração do dispatcher.

### Adicionado

- **Sprint 13.1 — Vetorização de frequências**:
  - `prange(nf)` em `_fields_in_freqs_kernel_cached`
    ([_numba/kernel.py](../geosteering_ai/simulation/_numba/kernel.py)) e
    `precompute_common_arrays_cache`
  - `@njit(cache=True, parallel=True, nogil=True)` em ambas funções
  - Quando chamadas de contexto `prange` externo, Numba serializa o nível
    interno automaticamente (sem regressão)
- **Sprint 13.2 — Cache cross-call**:
  - Novo kwarg `cache_persistent: bool = False` em `simulate_multi` (opt-in)
  - Cache global thread-safe `_GLOBAL_HORDIST_CACHE` com chave
    `(round(hordist, 12), freqs_signature, n, eta_bytes, h_bytes)` —
    detecção bit-exata de qualquer variação numérica
  - Funções públicas exportadas em `geosteering_ai.simulation`:
    - `release_numba_cache() -> int` — libera cache, retorna count
    - `get_numba_cache_size() -> int` — diagnóstico
  - UI `closeEvent` chama 3 releases:
    `release_numba_pool()` + `release_pool()` + `release_numba_cache()`
- **Sprint 13.4 — `nogil=True` universal**:
  - Wrapper `njit` em `_numba/propagation.py` agora seta `nogil=True`
    como default — todas as funções `@njit` do simulador liberam GIL
  - Custo zero em performance; benefício direto em uso multi-thread
    (notebooks, treino offline, UI responsiva)
  - `fastmath=True` permanece **opt-in caso-a-caso** — preserva paridade
    Fortran <1e-12 nas recursões TE/TM e `common_arrays`

### Testes

- **Novo arquivo** `tests/test_simulation_v213_optimizations.py`:
  - 13 testes cobrindo Sprints 13.1, 13.2 e 13.4 — **todos passam (2.17s)**
  - Cobre: vetorização freqs, cache hit/miss/release/thread-safety,
    backward-compat v2.12, threading concorrente sem corrupção
- **Zero regressão**: 152/152 testes pré-existentes (workers + multi +
  numba_kernel + config) continuam passando em 13.12s

### Validado

- Paridade bit-exata entre chamadas (cache hit) — `assert_array_equal`
- Backward-compat total: API v2.12 single-model e batch (`models=[...]`)
  funcionam idênticas quando `cache_persistent=False` (default)
- Thread-safety: 4 ThreadPool workers concorrentes não corrompem resultados
- Cache miss correto em variações de freqs e perfis geológicos
- Smoke 30k-modelos: chamada multi-freq (`[20, 40, 60, 100] kHz`) operacional

### Pendências (v2.14+)

- **Sprint 13.3 — prange combinado TR×ângulo**: requer refatoração do
  dispatcher `multi_forward.py:732-818` para colapsar 2 loops Python em
  `prange(nTR*nAngles)` Numba via array de `cache_indices` materializadas
- **Sprint 13.4 — `fastmath=True` seletivo**: aplicar `fastmath=True` em
  Hankel quadratura e operações de decoupling após validação de paridade
  Fortran <1e-12 em 4 modelos canônicos
- **Benchmark formal v2.13**: script CLI cobrindo 4 cenários (single-freq,
  multi-freq, multi-TR, PINN) — adiado para validação final em hardware
  do usuário

### Modificado

| Arquivo | Mudanças |
|:--------|:---------|
| `_numba/kernel.py` | `prange(nf)` em 2 funções + nogil + parallel |
| `_numba/propagation.py` | Wrapper `njit` agora seta `nogil=True` default |
| `multi_forward.py` | +`cache_persistent` kwarg + cache global + 2 funções públicas |
| `simulation/__init__.py` | Exports: `release_numba_cache`, `get_numba_cache_size` |
| `tests/simulation_manager.py` | `closeEvent` chama 3 releases (pool ui + pool core + cache) |

---

## [2.12] — 2026-04-30

### Workers Nativos no `simulate_multi`

Implementação da Sprint 12 (relatório técnico
[v2.11_simulador_python_analise_paralelismo_2026-04-30.md](reports/v2.11_simulador_python_analise_paralelismo_2026-04-30.md))
que migra o suporte a paralelismo inter-modelo da camada UI
(`tests/sm_workers.py`) para o **core do simulador**
(`geosteering_ai/simulation/_workers.py`).

### Adicionado

- **Novo módulo `geosteering_ai/simulation/_workers.py`** (~530 LOC)
  com 4 modos de execução:
  - **A** (Single, 1w × 1t) — debug/pequeno
  - **B** (Multi-Thread, 1w × Nt) — 1 simulação grande
  - **C** (Workers, Mw × 1t) — batch n_pos baixo
  - **D** (Hybrid, Mw × Kt) — ★ DEFAULT PRODUÇÃO
- `MultiSimulationResultBatch` (frozen dataclass) em
  `geosteering_ai.simulation` com campos:
  `H_stack`, `z_obs`, `elapsed_s`, `throughput_mod_per_h`,
  `backend`, `n_workers`, `n_threads`, `mode`.
- `release_pool()` exportado em `geosteering_ai.simulation` para
  cleanup explícito.
- 3 novos kwargs em `simulate_multi`:
  - `models: Optional[List[Dict]]` — batch de modelos
  - `n_workers: Optional[int]` — número de processos do pool
  - `threads_per_worker: Optional[int]` — threads Numba por worker
    (`None` = auto anti-oversubscription)
- 2 novos campos em `SimulationConfig`:
  `n_workers`, `threads_per_worker` (defaults `None`).
- **Anti-oversubscription automático**: quando
  `threads_per_worker is None`, usa `eff = max(1, cpu // n_workers)`.
- **Benchmark** `benchmarks/bench_v212_workers.py` com 4 modos
  comparativos.
- **17 novos testes** em `tests/test_simulation_workers.py`
  (paridade A/B/C/D, validação de input, métricas).
- **9 novos testes** em `tests/test_simulation_config.py` para
  validação dos novos campos.

### Migração Simulation Manager

- `closeEvent` agora libera tanto `release_numba_pool()` (UI)
  quanto `release_pool()` (core) — cleanup completo de ambos pools.
- API nativa **disponível** mas UI mantém pool próprio para
  preservar Pause/Cancel cooperativo (v2.11) com checkpoints
  `_wait_if_paused` + `_cancel_requested`.

### Métricas de paridade

- Modo A vs Modo B: **bit-exato** (`assert_allclose rtol=1e-14`).
- Modo A vs Modo C: **<1e-12** (tolerância FP entre processos).
- Modo A vs Modo D: **<1e-12** (tolerância FP entre processos).

### Backward-compat

- `simulate_multi(rho_h=..., rho_v=..., esp=..., positions_z=...)`
  retorna `MultiSimulationResult` (v2.11) — comportamento atual.
- `simulate_multi(models=[...], n_workers=N)` retorna
  `MultiSimulationResultBatch` (v2.12) — caminho novo.
- API existente intocada. **Zero regressão** em pytest
  (734 simulation tests passing, 0 failed).

### Smoke tests

- 197 → 202 (+5: T24-T28).

---

## [2.11] — 2026-04-29

### Análise Causa-Raiz do Freezing GUI

Identificados 5 gargalos `O(N)` na main thread via profiling instrumentado
(`MainThreadHeartbeat`):

1. `generate_models(N)` — loop síncrono na main thread (3-30s para 30k modelos)
2. `appendPlainText` log — `O(N²/100²)` cumulativo (5-30s acumulativos)
3. `_refresh_keys` combo populate — `O(min(100, N))` síncrono
4. `_append_simulation_snapshot` JSON serialize — `O(N)` na main thread
5. Pool spawn first-time — `O(n_workers)` na worker thread (UX gap)

### Adicionado

- `ModelGenerationThread` — geração de modelos assíncrona em `QThread` separada
- `PhaseTimer` — instrumentação permanente com sinais Qt (`phase_started`, `phase_completed`)
- `WorkerProgressWidget` — barras individuais por worker com health status
- `CorrelationBySlice` — p-values granulares por frequência + UI tabbed
- `SnapshotPersistThread` — persistência de snapshot em `QThread`
- `MainThreadHeartbeat` — sentinel de gaps na main thread (debug)
- Painel "Cronologia da Simulação" com tempos exatos de cada fase
- Botões de Pause/Resume/Cancel com sinais cooperativos

### Corrigido

- GUI travava por tempo proporcional a N (qualquer quantidade de modelos)
- Buffer de log com flush throttled (1 Hz) substitui `appendPlainText` direto
- Combo populate usa `setModel(QStringListModel)` em batch
- Cancelamento limpo do pool persistente em `closeEvent`

### Métrica de sucesso

- `max_gap_ms < 50ms` na main thread para qualquer N (100, 1k, 10k, 30k)
- Latência click → primeiro feedback < 200ms

### Smoke tests

- 156 → 166 (+10: T17-T26)

---

## [2.10] — 2026-04-28

### Adicionado

- Pool persistente Numba (`_acquire_numba_pool`, `_numba_init_worker`,
  `release_numba_pool`) — workers spawn/import/JIT 1× por sessão
- Defer mechanism: `_pending_sim_trigger` + `_prewarm_numba_pool`
  auto-disparam simulação ao concluir warmup

### Corrigido

- 1ª simulação 3× mais lenta que subsequentes (overhead de spawn/import/JIT)
- Fallback p-value combinado em `_compute_pvalues` agora retorna `1.0`
  (conservador, assume não-significância) em vez de `np.mean(pvals)` inválido

### Smoke tests: 148 → 156 (+8: T15-T16)

---

## [2.9] — 2026-04-28

### Adicionado

- `NumbaPrimer(QThread)` — pré-aquecimento assíncrono de cache JIT na startup
- Status bar label "🔥 JIT Numba…" → "✓ JIT (Xs)"

### Corrigido

- Race condition de timing no `crosshair` matplotlib — `CrosshairManager`
  removido completamente (234 LOC)
- Cache JIT invalidado após atualização Numba (1ª simulação 3× mais lenta)
- Aviso `OMP: omp_set_nested deprecated` suprimido (`KMP_WARNINGS=FALSE`)
- Tema do canvas agora persiste corretamente entre sessões (`canvas/theme`
  unificado em `_qsettings()`)
- `NumbaPrimer` lazy start em `showEvent` + cleanup em `closeEvent`

### Removido

- `sm_crosshair.py` (234 LOC) + 11 seções em `simulation_manager.py`
- Shortcut `Ctrl+Shift+C` + botão de toolbar

### Smoke tests: 142 → 148 (+6: T13-T14)

---

## [2.8] — anterior

- Plot kinds dinâmicos (`_on_kind_mode_changed`)
- `CorrelationAnalysisDialog` com método selecionável (Pearson/Spearman/Kendall)
- Export CSV de matriz de correlação
- Smoke tests T9-T12

---

## [2.7a] — 2026-04-25 (PR #29)

- Migração `PyQt6` + `PySide6` (compatibilidade dupla via `sm_qt_compat.py`)
- Bug fixes diversos + polimento de UX
- Smoke tests T1-T5 (binding, ALIGN_*/ORIENT_*, dark mode,
  `CollapsibleGroupBox`, `PyQtGraphCanvas`)

---

## [2.6b] — anterior

- Bug fix A1 (referenciado em `feat/simulation-manager-v2.6b`)
- Multi-backend foundation
- Smoke tests T6-T8

---

## [2.5] — 2026-04-25 (PR #26)

- `PlotComposerDialog`
- Fix cache LRU multi-freq × angle
- Fortran multi-TR + JAX `chunk_size`
- 70/70 smoke tests; 1464/0 pytest
