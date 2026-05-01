---
description: Sub-skill dedicada ao Simulation Manager (SM) Numba do Geosteering AI v2.0. Cobre arquitetura de pacote (multi_forward, forward, _numba/, _jax/), workers nativos (v2.12), otimizações Numba (v2.13: prange-freq, cache, nogil; v2.14: prange TR×ang, fastmath hankel), JIT cache observability (v2.15), hardware tuning, paridade Fortran <1e-12. Triggers SM: "simulation manager", "SM", "v2.10", "v2.11", "v2.12", "v2.13", "v2.14", "v2.15", "simulate_multi", "Workers Nativos", "prange", "cache_persistent", "release_numba_cache", "set_jit_cache_maxsize", "get_jit_cache_info", "fastmath", "hmd_tiv", "vmd", "common_arrays", "common_factors", "tatu.x exec format".
---

# Sub-skill: Geosteering Simulation Manager (SM-Numba)

**Versão**: v2.15 (2026-05-01)
**Triggers principais**: simulate_multi, Simulation Manager, SM, v2.x SM, prange, fastmath, cache_persistent, get_jit_cache_info, tatu.x

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
| **v2.15** | **2026-05-01** | **feat/simulation-manager-v2.15** | **Hardware validation, JIT cache observability, code review P1, fix CI tatu.x exec format** | **165+27+4** |

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

| Cenário | Modelos | Freqs | TR | Ang | Meta | Hardware 8C |
|:-------:|:-------:|:-----:|:--:|:---:|:----:|:-----------:|
| A | 30k | 1 | 1 | 1 | zero regressão v2.12 | 753k mod/h |
| B | 30k | 10 | 1 | 1 | ≥1.5× v2.12 | 320k mod/h |
| C | 30k (×6=5k effetivo) | 2 | 3 | 5 | ≥1.3× v2.13 | (validação) |
| D | 1 (×50 calls) | 4 | 1 | 1 | ≥5× sem cache | 13.4 ms/call |

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

## §13 Referências

- `docs/CHANGELOG.md` — versões v2.10 a v2.15
- `docs/reports/v2.{N}_*.md` — relatórios técnicos por versão
- `docs/ROADMAP.md` — tabela de versões SM
- `docs/reports/v2.15_fastmath_dipoles_analysis_2026-05-01.md` — análise técnica fastmath
- `docs/reports/v2.15_benchmark_hardware_2026-05-01.md` — resultados hardware
- `tests/_fortran_helpers.py` — helpers paridade Fortran
- `tests/test_simulation_v213_optimizations.py` — 13 testes v2.13
- `tests/test_simulation_v214_prange_combined.py` — 8 testes v2.14
- `tests/test_simulation_v214_fastmath.py` — 6 testes v2.14
- `tests/test_simulation_jax_sprint13.py` — 4 testes v2.15

---

**Sub-skill criada em 2026-05-01 (v2.15). Atualizar a cada versão SM.**
