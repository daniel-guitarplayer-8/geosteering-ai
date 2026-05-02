---
description: Sub-skill dedicada ao Simulation Manager (SM) Numba do Geosteering AI v2.0. Cobre arquitetura de pacote (multi_forward, forward, _numba/, _jax/), workers nativos (v2.12), otimizações Numba (v2.13: prange-freq, cache, nogil; v2.14: prange TR×ang, fastmath hankel), JIT cache observability (v2.15), fix regressão threading + I/O vetorizado (v2.16), fix oversubscrição HT/SMT (v2.17), hardware tuning, paridade Fortran <1e-12. Triggers SM: "simulation manager", "SM", "v2.10", "v2.11", "v2.12", "v2.13", "v2.14", "v2.15", "v2.16", "v2.17", "simulate_multi", "Workers Nativos", "prange", "cache_persistent", "release_numba_cache", "set_jit_cache_maxsize", "get_jit_cache_info", "fastmath", "hmd_tiv", "vmd", "common_arrays", "common_factors", "tatu.x exec format", "Cenário E", "n_positions", "write_dat_from_tensor", "NUMBA_NUM_THREADS spawn", "detect_cpu_topology", "recommend_default_parallelism", "oversubscrição", "hyperthreading", "HT/SMT", "physical cores", "logical cores".
---

# Sub-skill: Geosteering Simulation Manager (SM-Numba)

**Versão**: v2.17 (2026-05-02)
**Triggers principais**: simulate_multi, Simulation Manager, SM, v2.x SM, prange, fastmath, cache_persistent, get_jit_cache_info, tatu.x, Cenário E, n_positions, threading masking, oversubscrição, hyperthreading, detect_cpu_topology

Esta sub-skill é DEDICADA ao Simulation Manager Numba (`geosteering_ai/simulation/`).
Para o Simulador Python JAX (backend `_jax/`), use `geosteering-simulator-python`.
Para o Simulador Fortran (`Fortran_Gerador/`), use `geosteering-simulator-fortran`.

---

## §1 Identidade e Versões

| Versão | Data | Branch | Entrega Principal | Tests |
|:------:|:----:|:------:|:------------------|:-----:|
| v2.10 | 2026-04-28 | snapshot | Pool persistente Numba + p-value fallback | 156 |
| v2.11 | 2026-04-29 | feat/simulation-manager-v2.11 | ModelGenerationThread async, elimina freezing GUI | 197 |
| v2.12 | 2026-04-30 | feat/simulation-manager-v2.12 | Workers Nativos (4 modos A/B/C/D) | 202 |
| v2.13 | 2026-05-01 | feat/simulation-manager-v2.13 | Otimizações Numba (Sprints 13.1+13.2+13.4): prange(nf), cache cross-call, nogil universal | 165 (152+13) |
| v2.14 | 2026-05-01 | feat/simulation-manager-v2.14 | Sprint 13.3 prange TR×ang + Sprint 13.4 fastmath hankel.py + benchmark formal | 165+27 |
| v2.15 | 2026-05-01 | feat/simulation-manager-v2.15 | Hardware validation, JIT cache observability, code review P1, fix CI tatu.x exec format | 165+27+4 |
| v2.16 | 2026-05-01 | feat/simulation-manager-v2.16 | Fix regressão crítica de threading masking (4–8× em produção GUI) + Cenário E benchmark (600 pts) + I/O `write_dat_from_tensor` vetorizado ≥3× | 165+27+4+7 |
| **v2.17** | **2026-05-02** | **feat/simulation-manager-v2.17** | **Fix regressão de oversubscrição em CPUs HT/SMT (3× em produção GUI): `detect_cpu_topology()` + `recommend_default_parallelism()` + warning visual GUI + logging diagnóstico. Defaults físicos: 8C/16T HT → (4w × 2t = 8) em vez de (4w × 4t = 16)** | **165+27+4+7+19** |

---

## §2 Arquitetura

```
geosteering_ai/simulation/
├── __init__.py             ← exports públicos: simulate_multi, simulate, release_pool, etc.
├── config.py               ← SimulationConfig dataclass (errata, presets)
├── multi_forward.py        ← dispatcher: simulate_multi(rho_h/rv/esp/positions_z/...)
├── forward.py              ← simulate() shim + _simulate_combined_prange (Sprint 13.3)
├── _numba/
│   ├── propagation.py      ← @njit fastmath=False (TE/TM recursão)
│   ├── dipoles.py          ← hmd_tiv, vmd @njit (UNSAFE para fastmath, v2.15 oficial)
│   ├── kernel.py           ← _fields_in_freqs_kernel + _cached + precompute_common_arrays_cache
│   ├── hankel.py           ← @njit(fastmath=True) v2.14: prepare_kr, integrate_j0/j1/j0_j1
│   └── geometry.py         ← find_layers_tr_numba
├── _jax/                   ← backend JAX (escopo: geosteering-simulator-python)
├── _workers.py             ← run_batch (Workers Nativos v2.12, 4 modos)
├── filters/                ← Werthmüller 201pt, Kong 61pt, Anderson 801pt
├── io/                     ← model_in, binary_dat (Fortran 22-col)
├── postprocess/            ← compensation, tilted antennas
└── validation/             ← canonical_models, compare_fortran
```

**Pontos de entrada**:

- `simulate(rho_h, rho_v, esp, positions_z, ...)` — single-modelo, single-TR/ang
- `simulate_multi(...)` — multi-TR×ang, suporta `models=[...]` para batch
- `release_pool()` — libera worker pool (chamar antes de release_numba_cache)
- `release_numba_cache()` — limpa Numba JIT (free RAM)

---

## §3 Workers Nativos (v2.12)

`simulate_multi(models=[...], n_workers=N, threads_per_worker=T)` ativa
4 modos automaticamente baseado em `n_workers` × `threads_per_worker`:

| Modo | n_workers | threads/worker | Quando usar |
|:----:|:---------:|:--------------:|:-----------|
| A | 1 | 1 | Determinístico/paridade Fortran |
| B | 1 | >1 | Single-process, intra-prange |
| C | >1 | 1 | Multi-process, isolado |
| D | >1 | >1 | Default produção (PINN, batch 30k) |

**Anti-oversubscription**: total threads = `n_workers × threads_per_worker`,
deve ser ≤ `os.cpu_count()` (cores físicos preferíveis aos lógicos para
Numba — 8 físicos > 16 hyperthreaded).

---

## §4 Otimizações Numba (v2.13 + v2.14)

### v2.13 — prange(nf) + cache cross-call + nogil

- **Sprint 13.1**: `prange(nf)` em `_simulate_positions_njit_cached`
  vetoriza frequências (1.5× speedup multi-freq)
- **Sprint 13.2**: `cache_persistent=True` mantém common_arrays entre
  chamadas (5× speedup PINN cache hit teórico)
- **Sprint 13.4 (parcial)**: `nogil=True` universal em prange → permite
  ThreadPoolExecutor concorrente sem GIL contention

### v2.14 — prange TR×ang + fastmath hankel

- **Sprint 13.3**: `_simulate_combined_prange` em `forward.py` colapsa
  loops Python serial `for i_tr × for i_ang` em `prange(n_combos × n_pos)`
  flat (≥1.3× speedup multi-TR×ang)
- **Sprint 13.4 (final)**: `@njit(fastmath=True)` em 4 helpers de
  `hankel.py` (FMA-safe pure dot-product, erro acumulado ~2e-14)

---

## §5 Hardware Tuning

### CPU intra-worker (Numba threads)

```bash
# Setar ANTES de import numba (env vars lidos no primeiro import)
export NUMBA_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

Em código:
```python
import numba
numba.set_num_threads(2)  # apenas se chamado ANTES de qualquer prange
```

⚠️ **Limitação Numba**: `set_num_threads(N)` falha com `RuntimeError:
Cannot set NUMBA_NUM_THREADS to a different value once threads have
been launched` se chamado depois de prange. Mitigation: setar env var
antes do primeiro import.

### Anti-oversubscription

| Hardware | Cores físicos | Lógicos | Workers × Threads recomendado |
|:--------:|:-------------:|:-------:|:----------------------------:|
| Intel 8C/16T | 8 | 16 | 4 × 2 (= 8 físicos) |
| Apple M2 Pro | 10 | 10 | 5 × 2 (= 10) |
| Server 32C/64T | 32 | 64 | 8 × 4 (= 32) |
| Colab Pro T4 | 4 | 8 | 2 × 2 (= 4 físicos) |

---

## §6 Cache Cross-Call (`cache_persistent`)

```python
from geosteering_ai.simulation import simulate_multi, release_numba_cache

cfg = SimulationConfig(cache_persistent=True)
for iter in range(50):  # PINN training loop
    result = simulate_multi(rho_h=rho_h_iter, ..., cfg=cfg)
release_numba_cache()  # libera RAM ao final
```

Hit-rate ideal: cada `(hordist_round, n_layers, freqs_hash)` único
sobrevive entre chamadas. Em PINN com mesmo `(positions_z, dip_degs,
tr_spacings_m)`, hit-rate = 100% após 1ª chamada.

---

## §7 Paridade Fortran (gate <1e-12)

### Helper `_tatu_runnable()` (v2.15)

```python
from tests._fortran_helpers import _tatu_runnable
if _tatu_runnable():
    # rodar comparação Python ↔ Fortran
    ...
```

Resolve `OSError [Errno 8] Exec format error` em CI Linux quando o
repo contém `Fortran_Gerador/tatu.x` macOS commitado. Detecta via
`subprocess.run` real.

### Canônicos validados

`oklahoma_3`, `oklahoma_5`, `devine_8`, `oklahoma_15`, `oklahoma_28`,
`hou_7`, `viking_graben_10` — todos com max_diff < 1e-12 (Sprint 4.4).

---

## §8 Análise fastmath SAFE / UNSAFE (v2.15 oficial)

| Função | Arquivo:Linha | fastmath | Razão |
|:-------|:--------------|:--------:|:------|
| `prepare_kr` | hankel.py:90 | **True** | Pure scalar (~npt ops) |
| `integrate_j0` | hankel.py:121 | **True** | Dot-product (npt × 2 ops) |
| `integrate_j1` | hankel.py:156 | **True** | Dot-product |
| `integrate_j0_j1` | hankel.py:182 | **True** | Dot-product 2-em-1 |
| `hmd_tiv` | dipoles.py:179 | False | Recursão TE/TM, ~600 ops, FMA risk ~1.2e-13 |
| `vmd` | dipoles.py:703 | False | Recursão TE, ~400 ops, ~8e-14 |
| `_fields_in_freqs_kernel` | kernel.py:301 | False | Transitivo |
| `_fields_in_freqs_kernel_cached` | kernel.py:670 | False | Transitivo |
| `precompute_common_arrays_cache` | kernel.py:581 | False | Transitivo |
| `common_arrays`, `common_factors` | propagation.py | False | Recursão TE/TM cancelamento |

**Decisão**: NÃO aplicar fastmath em dipoles/kernel oficialmente (v2.15).
Ver `docs/reports/v2.15_fastmath_dipoles_analysis_2026-05-01.md`.

---

## §9 Backend JAX (referência cruzada)

`_jax/` é backend opt-in via `cfg.backend="jax"`. Estratégias:

- `bucketed`: 1 prog XLA por `(ct, cr, n, npt)` — para 22 camadas Oklahoma_28
  resulta em ~44 compilações (alto VRAM)
- `unified`: 1 prog XLA por `(n, npt)` — Oklahoma_28: 1 compilação
  (consolidação 44× → 1, ~250 MB vs 11 GB)
- `chunked`: padrão `unified` particionado em chunks de N posições
  (mitiga materialização de tensores intermediários grandes)

Detalhes em sub-skill `geosteering-simulator-python`.

---

## §10 JIT Cache Observability (v2.15)

```python
from geosteering_ai.simulation._jax.forward_pure import (
    get_jit_cache_info, clear_jit_cache, clear_unified_jit_cache,
)

info = get_jit_cache_info()
print(f"Total XLA programs: {info['total_xla_programs']}")
print(f"Bucketed: {info['bucketed_size']}, Unified: {info['unified_size']}, "
      f"Chunked: {info['chunked_size']}")
print(f"Estimated VRAM: {info['estimated_vram_mb']:.1f} MB")

# Em loops PINN, monitor para detectar vazamento
if info['estimated_vram_mb'] > 10000:  # 10 GB threshold
    clear_jit_cache()
    clear_unified_jit_cache()
```

Heurística VRAM: `Σ (3 × n × npt × 16 bytes) / 1024²` por entrada.
Conservadora (não inclui buffers `vmap`).

---

## §11 Benchmark CLI (v2.14 + v2.15)

```bash
# Cenário único
PYTHONPATH=. python benchmarks/bench_v214_numba.py --scenario A \
    --models 30000 --workers 4 --threads-per-worker 2

# Todos os 4 cenários (mesma config)
PYTHONPATH=. python benchmarks/bench_v214_numba.py --all \
    --models 30000 --freqs 10 --workers 4 --threads-per-worker 2
```

| Cenário | Modelos | Posições | Freqs | TR | Ang | Meta | Hardware 8C |
|:-------:|:-------:|:--------:|:-----:|:--:|:---:|:----:|:-----------:|
| A | 30k | 30 | 1 | 1 | 1 | zero regressão v2.12 | 1.74M mod/h (v2.16) |
| B | 30k | 30 | 10 | 1 | 1 | ≥1.5× v2.12 | 320k mod/h |
| C | 30k (×6=5k effetivo) | 30 | 2 | 3 | 5 | ≥1.3× v2.13 | 119k mod/h |
| D | 1 (×50 calls) | 30 | 4 | 1 | 1 | ≥5× sem cache | 13.4 ms/call |
| **E (v2.16)** | **2k** | **600** | **1** | **1** | **1** | **≥120k mod/h pós-fix** | **123k mod/h** |

⚠️ **Limitação `--all`**: `NUMBA_NUM_THREADS` não pode ser alterado
após threads ativas. Use 1 cenário por invocação se precisa variar
`threads_per_worker` entre cenários.

---

## §12 Troubleshooting

### `OSError: [Errno 8] Exec format error: tatu.x`

Causa: binário macOS commitado no repo, executado em Linux. Solução:
testes usam `_tatu_runnable()` desde v2.15 — atualizar local pull.

### `RuntimeError: Cannot set NUMBA_NUM_THREADS ...`

Causa: chamada a `numba.set_num_threads()` ou env var setado depois
de Numba já lançar threads. Solução: setar `NUMBA_NUM_THREADS` antes
do primeiro `import numba`. No benchmark, use `--threads-per-worker N`
em invocação única, sem misturar com runs anteriores no mesmo processo.

### Paridade Fortran > 1e-12

1. Confirmar `cfg.backend == "numba"` e não JAX (JAX tolerância 1e-10)
2. Confirmar `hankel_filter == "werthmuller_201pt"` (Fortran usa 201pt)
3. Limpar cache: `release_pool()` + `release_numba_cache()`
4. Verificar versão Numba (≥0.60 recomendado)
5. Reportar em GitHub com modelo + diff

### VRAM crescendo em Colab T4 PINN

Usar `get_jit_cache_info()` para monitorar e `clear_unified_jit_cache()`
periodicamente. LRU bound 64 entradas — ajuste com `set_jit_cache_maxsize(N)`.

---

## §14 Threading Masking — Causa Raiz da Regressão v2.15 (FIX v2.16)

### Sintoma

GUI do Simulation Manager produzindo 25–38k mod/h em produção (n_positions=600)
quando o esperado seria ≥150k mod/h. Microbenchmark Cenário A não detectou.

### Causa raiz (commits problemáticos)

1. **`0f92035`** — `multi_forward.py:880-886`: `try/except RuntimeError: pass`
   silencia falhas de `numba.set_num_threads()` sem fallback
2. **`e1c8864`** — remove `os.environ["NUMBA_NUM_THREADS"]` de
   `_numba_init_worker` (sm_workers.py) e `_simulate_worker_init`
   (_workers.py), deixando workers spawn com pool dimensionado por
   `cpu_count()` (16 em hyperthreaded de 8 cores)

### Fix v2.16 (Sprint 15.1)

```python
# multi_forward.py: try/except: pass → logger.warning observable
if cfg.num_threads > 0 and HAS_NUMBA:
    import numba as _numba
    current_active = _numba.get_num_threads()
    if current_active != cfg.num_threads:
        try:
            _numba.set_num_threads(cfg.num_threads)
        except RuntimeError as exc:
            logger.warning(
                "numba.set_num_threads(%d) falhou (threads ativas atual=%d, "
                "pool size NUMBA_NUM_THREADS=%d): %s. Performance pode degradar.",
                cfg.num_threads, current_active, _numba.config.NUMBA_NUM_THREADS, exc,
            )

# sm_workers.py::_acquire_numba_pool e _workers.py::_acquire_pool:
# Setar NUMBA_NUM_THREADS no env do PAI antes do spawn
os.environ["NUMBA_NUM_THREADS"] = str(n_threads)
os.environ["OMP_NUM_THREADS"] = str(n_threads)
ProcessPoolExecutor(..., mp_context=spawn, initializer=_numba_init_worker, ...)
```

### Validação

`tests/test_simulation_workers_threading.py` (3 testes):
1. `test_worker_inherits_numba_num_threads_from_parent_env` — env herdado
2. `test_set_num_threads_emits_warning_on_runtime_error` — log observable
3. `test_simulate_multi_in_worker_respects_n_threads` — E2E em worker spawn

### Throughput pós-fix (Hardware Intel 8C/16T HT)

| Cenário | Pré-fix v2.15 | Pós-fix v2.16 | Speedup |
|:-------:|:-------------:|:-------------:|:-------:|
| A (30 pts) | 753k mod/h | **1.74M mod/h** | 2.31× |
| E (600 pts) | ~30k mod/h | **123k mod/h** | ~4.1× |

---

## §15 I/O Vetorizado `write_dat_from_tensor` (v2.16, Sprint 15.4)

`geosteering_ai/simulation/tests/sm_io.py::write_dat_from_tensor` foi
reescrita substituindo 5 loops Python aninhados (~1.8M iterações para
600 modelos × 600 pos × 1 freq) por broadcast NumPy:

```python
# Permutação para ordem Fortran (m, itr, kt, fi, jm, ic)
H_perm = np.ascontiguousarray(np.transpose(H, (0, 1, 2, 4, 3, 5)))
H_flat = H_perm.reshape(total_records, 9)

# Broadcasts vetorizados
buf["col0"] = np.broadcast_to(model_ids[:,None,None,None,None], shape).ravel()
buf["col1"] = np.broadcast_to(z_view, shape).ravel()
buf["col2"] = np.broadcast_to(rho_h_at_obs[:,None,:,None,:], shape).ravel()
buf["col3"] = np.broadcast_to(rho_v_at_obs[:,None,:,None,:], shape).ravel()
for ic in range(9):
    buf[f"col{4+2*ic}"] = H_flat[:, ic].real
    buf[f"col{5+2*ic}"] = H_flat[:, ic].imag
```

**Speedup ≥3×** validado por `test_write_dat_vectorized_is_faster`.
**Bit-exatness** preservada (3 testes em `tests/test_sm_workers_io.py`).

---

## §16 Análise n_positions Scaling (v2.16, Sprint 15.5)

Documento dedicado: `docs/reports/v2.16_n_positions_scaling_analysis_2026-05-01.md`

**Hot path:** `_numba/dipoles.py::hmd_tiv` linhas 325–420 (~80% do tempo).
**Complexidade:** O(`n_pos × n_layers × n_filter_pts × n_freqs × n_TR × n_ang`).
**Escalabilidade `n_pos`:** linear. 30→600 = 20× redução de throughput
(confirmado: 1.74M / 123k ≈ 14× pós-fix).

### Top 3 oportunidades (deferidas para v2.17/v2.18)

| Otimização | Ganho est. | Risco | Status v2.16 |
|:-----------|:----------:|:-----:|:------------:|
| Tile/block processing (M=4 pos) | 15–25% | Baixo | Deferido v2.17 |
| Pré-compute Hankel kernels TE/TM | 10–15% | Médio | Deferido v2.18 |
| SIMD ufuncs NumPy | 20–40% | **Alto** | **REJEITADO** (paridade <1e-12) |

---

## §17 Oversubscrição em CPUs HT/SMT (v2.17, Sprint 16)

**Problema descoberto pós-v2.16**: GUI continuava a 38k mod/h em produção, mesmo
com fix de threading masking aplicado. **Causa raiz**: defaults da GUI
(`spin_workers = ncpu // 4`, `spin_threads = ncpu // (ncpu // 4)`) produziam
**oversubscrição** em CPUs com Hyperthreading (Intel) ou SMT (AMD/ARM):

| Hardware | `os.cpu_count()` | Default v2.16 (W × T) | Total threads | Cores físicos | Status |
|:---------|:----------------:|:---------------------:|:-------------:|:-------------:|:------:|
| Intel 8C/16T HT | 16 | 4 × 4 | 16 | 8 | ⚠ 2× oversub |
| Linux 32C/64T HT | 64 | 16 × 4 | 64 | 32 | ⚠ 2× oversub |
| Apple Silicon M1 | 8 | 2 × 4 | 8 | 8 | ✓ ok |

**Por que oversub é ruim em workloads CPU-bound (como Numba JIT prange)**:

1. **Cache L1/L2 compartilhada** entre hyperthreads do mesmo core físico → trashing
2. **FPU/ALU compartilhada** → contenção em operações aritméticas
3. **Context switch overhead** no scheduler do SO
4. **Numba TBB/OMP × N processos** = competição massiva por recursos

Em workloads CPU-bound puros, **1 thread/core físico** é tipicamente
30-50% mais rápido que 1 thread/hyperthread.

### Fix v2.17 (Sprint 16.1+16.2)

Nova API pública em `geosteering_ai/simulation/_workers.py`:

```python
from geosteering_ai.simulation import (
    detect_cpu_topology,        # Sprint 16.1 (v2.17)
    recommend_default_parallelism,  # Sprint 16.2 (v2.17)
)

# Detecção em camadas: psutil → sysctl → /proc/cpuinfo → wmic → heurística
logical, physical, has_ht = detect_cpu_topology()
# Em Mac Intel 8C/16T HT: (16, 8, True)
# Em Apple Silicon M1: (8, 8, False)
# Em Linux Xeon 32C/64T: (64, 32, True)

# Recomendação que respeita workers × threads ≤ physical_cores
n_workers, threads = recommend_default_parallelism()
# Mac 8C/16T HT: (4, 2) → 4 × 2 = 8 = phys ✓
# Apple Silicon M1: (4, 2) → 8 = phys ✓
# Linux 32C/64T: (16, 2) → 32 = phys ✓
```

### Recomendação Visual + Logging (Sprint 16.3+16.4)

Em `simulation_manager.py:SimulationPage`:

- **QLabel vermelho** aparece quando `workers × threads > physical_cores`
- Mensagem: "⚠ Oversubscrição: 4 × 4 = 16 threads em 8 cores físicos (2× sobrecarga)"
- Conectado via `valueChanged` aos spinboxes (atualização em tempo real)

Em `SimulationThread.run()`:

```python
self.log.emit(f"CPU: {phys} cores físicos · {logical} threads lógicas (HT/SMT)")
self.log.emit(f"  ⚠ AVISO: oversubscrição {factor:.2f}× ({total} threads > {phys} cores)")
```

### Métricas pós-fix (Hardware Intel 8C/16T HT, macOS)

| Configuração | Threads totais | Cenário E (200 mod, 600 pts) | Speedup |
|---|---|---|---|
| v2.16 default GUI (4w × 4t) | 16 (oversub 2×) | 70k mod/h | baseline |
| **v2.17 default GUI (4w × 2t)** | **8 (= phys)** | **85k mod/h** | **+21%** |
| Cenário E benchmark (4w × 2t) | 8 (= phys) | 85k mod/h | idêntico ✓ |

Em workloads de produção mais intensos (modelos com mais camadas, mais
frequências, ou n_pos > 1000) e em CPUs com HT mais agressivo (Linux Xeon
32C/64T), o ganho esperado é **30–50%**.

### Quando usar cada modo (Tabela de decisão)

| Cenário | n_models | Hardware | Defaults v2.17 | Modo |
|:--------|:--------:|:---------|:---------------|:----:|
| Single inversão / debug | 1 | qualquer | (1, phys) | B |
| Batch pequeno | 2-9 | qualquer | (1, phys) | B |
| Batch normal | 10-1000 | 4C+ | (phys // 2, 2) | D |
| Produção 30k | 10000+ | 8C+ HT | (4, 2) ou (8, 2) | D |
| Mac Apple Silicon (sem HT) | qualquer | 8C | (4, 2) | D |

### Validação

`tests/test_simulation_cpu_topology.py` — 19 testes:

- 5 testes de detecção (tupla, valores sãos, OS coerente, cache, fallback)
- 6 testes de recomendação (não-oversub, single-model, batch, determinismo)
- 6 testes de hardware simulado (Mac Intel, M1, Linux Xeon, low-end, dual-core, single)
- 2 testes de no-regression (cache persiste, recomendação determinística)

---

## §13 Referências

- `docs/CHANGELOG.md` — versões v2.10 a v2.17
- `docs/reports/v2.{N}_*.md` — relatórios técnicos por versão
- `docs/ROADMAP.md` — tabela de versões SM
- `docs/reports/v2.15_fastmath_dipoles_analysis_2026-05-01.md` — análise técnica fastmath
- `docs/reports/v2.15_benchmark_hardware_2026-05-01.md` — resultados hardware
- `docs/reports/v2.16_2026-05-01.md` — relatório principal v2.16 (threading masking + Cenário E)
- `docs/reports/v2.16_n_positions_scaling_analysis_2026-05-01.md` — análise n_positions
- `docs/reports/v2.17_2026-05-02.md` — relatório principal v2.17 (oversubscrição HT/SMT)
- `tests/_fortran_helpers.py` — helpers paridade Fortran
- `tests/test_simulation_v213_optimizations.py` — 13 testes v2.13
- `tests/test_simulation_v214_prange_combined.py` — 8 testes v2.14
- `tests/test_simulation_v214_fastmath.py` — 6 testes v2.14
- `tests/test_simulation_jax_sprint13.py` — 4 testes v2.15
- `tests/test_simulation_workers_threading.py` — 3 testes v2.16 (threading masking)
- `tests/test_sm_workers_io.py` — 4 testes v2.16 (I/O vetorizado)
- `tests/test_simulation_cpu_topology.py` — 19 testes v2.17 (CPU topology + oversub)

---

**Sub-skill criada em 2026-05-01 (v2.15). Última atualização: 2026-05-02 (v2.17).**
