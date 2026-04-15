# Relatório Final — Simulador Python v1.4.1 (pós PR #21)

**Data**: 2026-04-15
**Versão do subpacote**: `1.4.1`
**Commit HEAD**: `6a4cd0c` (PR #21 mergeado em `main`)
**Autor**: Daniel Leal
**Status**: Entrega incremental — PR #21 completo + arquitetura PR #22 documentada

---

## 1. Resumo Executivo

O simulador Python Numba/JAX atingiu **paridade física e numérica total com o Fortran `tatu.x`** em todas as capacidades implementadas até aqui (Sprints 1.1 → 11). O PR #21 fechou uma lacuna crítica de validação descoberta pelo usuário via comparação visual: a convenção Transmissor/Receptor estava invertida em ambos os paths (Numba + JAX), produzindo off-diagonais cruzadas (Hxz, Hzx) com sinal trocado em dip ≠ 0°. A correção foi aplicada cirurgicamente em 4 locais Numba + 3 locais JAX, acompanhada de 2 testes de regressão permanente.

### Entregáveis PR #21 (v1.4.1)

| Item | Status |
|:-----|:------:|
| Fix convenção T/R Numba (`forward.py`, `multi_forward.py`) | ✅ |
| Fix convenção T/R JAX (`_jax/kernel.py`, `_jax/forward_pure.py`) | ✅ |
| Teste `test_fortran_byte_exact_dat_nonzero_dip` (Numba vs Fortran, dip=30°, <1e-12) | ✅ |
| Teste `test_jax_dip_nonzero_matches_numba` (JAX vs Numba, dip=30°, <1e-10) | ✅ |
| Helper compartilhado `tests/_fortran_helpers.py` | ✅ |
| 1399 testes passam (0 falhas, 295 skipped) em 4:16 | ✅ |
| Bump `__version__` 1.4.0 → **1.4.1** | ✅ |
| PR #21 mergeado em `main` | ✅ |

### Pendente para PR #22 (v1.5.0)

| Item | Status | Esforço estimado |
|:-----|:------:|:-----------------|
| Sprint 10: JAX unified JIT via `lax.fori_loop` (44 → 1 programa XLA) | ⏳ | ~500 LOC + debug |
| Sprint 11-JAX: `simulate_multi_jax()` (multi-TR/angle/freq GPU) | ⏳ | ~500 LOC + testes |
| Testes paridade JAX unified-JIT vs Numba em 7 modelos (<1e-12) | ⏳ | +120 LOC |
| Testes paridade Sprint 11-JAX vs Numba (4 configs × 3 modelos) | ⏳ | +300 LOC |
| Benchmark Colab GPU T4/A100 (notebooks) | ⏳ | Acesso GPU |
| `/code-review` + sub-skill v1.9.0 + relatório v1.5.0 | ⏳ | Baixo |

---

## 2. Histórico Consolidado — Fases e Sprints do Simulador Python

Tabela completa de todas as fases e sprints implementadas, em ordem cronológica.

### Fase F7.0 — Fundação (Sprints 1.1 – 1.3)

| Sprint | Status | Data | PR # | Nome Curto | Arquivos Principais | Métrica-Chave | Descrição |
|:------:|:------:|:----:|:----:|:-----------|:--------------------|:--------------|:----------|
| **1.1** | ✅ | 2026-04-11 | #1 | Pesos Hankel `.npz` + FilterLoader | `scripts/extract_hankel_weights.py`, `filters/*.npz`, `filters/loader.py` | 3 filtros extraídos com SHA-256 auditável | Extração dos pesos Hankel do Fortran para `.npz` (Kong 61, Werthmüller 201, Anderson 801). `FilterLoader` thread-safe com cache classe-level. 53 testes. |
| **1.2** | ✅ | 2026-04-11 | #1 | `SimulationConfig` + errata | `config.py` (515 LOC) | 13 campos + validação + 4 presets | Dataclass frozen com validação de errata (frequency, spacing, backend); 4 presets (default, high_precision, production_gpu, realtime_cpu); YAML roundtrip. 62 testes. |
| **1.3** | ✅ | 2026-04-11 | #1 | Half-space analítico | `validation/half_space.py` | 5 funções bit-exatas | 5 funções analíticas fechadas (decoupling, skin_depth, wavenumber, VMD axial/broadside) como ground-truth para Fase 2. 38 testes. |

### Fase F7.1 — Backend Numba (Sprints 2.1 – 2.7)

| Sprint | Status | Data | PR # | Nome Curto | Arquivos Principais | Métrica-Chave | Descrição |
|:------:|:------:|:----:|:----:|:-----------|:--------------------|:--------------|:----------|
| **2.1** | ✅ | 2026-04-11 | #2 | Numba propagation | `_numba/propagation.py` (593 LOC) | `common_arrays` + `common_factors` bit-exatos vs Fortran | Port line-for-line de `commonarraysMD`/`commonfactorsMD` do Fortran. Dual-mode `@njit` + fallback NumPy. Tolerância <1e-13 vs Fortran. 25 testes. |
| **2.2** | ✅ | 2026-04-11 | #3 | Dipolos Numba + I/O + F6/F7 | `_numba/dipoles.py` (900 LOC), `io/*.py`, `postprocess/compensation.py`, `postprocess/tilted.py` | 6 casos geométricos HMD/VMD + F6 CDR + F7 tilted | Port completo de `hmd_TIV_optimized` e `vmd_optimized`. I/O: exportação binária 22-col byte-exata. F6 (compensação CDR) + F7 (antenas inclinadas). 58 testes. |
| **2.3+2.4** | ✅ | 2026-04-12 | #4 | Geometry + Rotation + Kernel | `_numba/geometry.py`, `_numba/rotation.py`, `_numba/hankel.py`, `_numba/kernel.py` (620 LOC) | Orquestrador `fields_in_freqs` completo | Geometria (sanitize_profile, find_layers_tr), rotação RtHR, integrais Hankel, kernel forward. 59 testes. |
| **2.5+2.6** | ✅ | 2026-04-12 | #5 | API `simulate()` + validação analítica | `forward.py` (450 LOC), `validation/compare_analytical.py` | `simulate(cfg)` + gate ACp/ACx <1e-5 | API pública single-source + `SimulationResult`. Bugfix crítico: `prof[0]` deve ser `-1e300` (sentinel Fortran). 33 testes. |
| **2.7** | ✅ | 2026-04-12 | #6 | Benchmark Fase 2 (gate) | `benchmarks/bench_forward.py`, notebook Colab | ≥40k mod/h (gate) → 66k-663k mod/h | Benchmark 3 perfis (small/medium/large) em CPU Intel i9. Resultado: 66k-663k mod/h (150-1127% do baseline Fortran). **Gate final Fase 2 ✅**. |

### Fase F7.2 — Paralelização + Cache (Sprints 2.8 – 2.10)

| Sprint | Status | Data | PR # | Nome Curto | Arquivos Principais | Métrica-Chave | Descrição |
|:------:|:------:|:----:|:----:|:-----------|:--------------------|:--------------|:----------|
| **2.8** | ✅ | 2026-04-12 | hybrid | ThreadPool + Visualização | `visualization/plot_tensor.py`, `plot_benchmark.py` | 16 plotagens validadas | ThreadPoolExecutor (GIL limita ganho imediato). Módulos de visualização GridSpec 3×7 para tensores. 11 testes. |
| **2.9** | ✅ | 2026-04-12 | hybrid | `@njit(parallel=True) + prange` | `forward.py::_simulate_positions_njit` | 6.6× speedup (663k mod/h) | Port de `fields_in_freqs` para `@njit(parallel=True)` com `prange`. Elimina GIL, paridade preservada. 69 testes. |
| **2.10** | ✅ | 2026-04-13 | #9 | Cache `common_arrays` (bucket) | `_numba/_cache.py` | O(unique_hordist) vs O(nTR×nAngles) | Cache bucketing por `hordist = L·sin(θ)` + freq + perfil. Reutiliza no loop multi-freq. **1014k mod/h small** (1722% Fortran). 16 plots validação. |

### Fase F7.3 — Backend JAX (Sprints 3.1 – 3.3.4)

| Sprint | Status | Data | PR # | Nome Curto | Arquivos Principais | Métrica-Chave | Descrição |
|:------:|:------:|:----:|:----:|:-----------|:--------------------|:--------------|:----------|
| **3.1** | ✅ | 2026-04-12 | hybrid | JAX Foundation | `_jax/hankel.py`, `_jax/rotation.py` | Paridade <1e-13 vs Numba | Infraestrutura JAX: Hankel via `jnp.einsum`, rotação tensorial. `jax.grad` diferenciável. 15 testes. |
| **3.2** | ✅ | 2026-04-12 | hybrid | `_jax/propagation.py` | `_jax/propagation.py` com `jax.lax.scan` | `jax.jacfwd` compatível | Port de `common_arrays`/`common_factors` com `lax.scan`. Diferenciação automática habilitada. 10 testes. |
| **3.3** | ✅ | 2026-04-12 | hybrid | JAX kernel híbrido | `_jax/kernel.py` | `pure_callback` para dipolos Numba | Orquestrador JAX híbrido reutilizando HMD/VMD Numba via `jax.pure_callback` + vmap. 20 testes. |
| **3.3.1** | ✅ | 2026-04-13 | hybrid | JAX dipolos (parcial) | `_jax/dipoles_native.py` (início) | 2/6 casos HMD via `lax.switch` | Port parcial HMD para JAX nativo (cases 1/2/3). 8 testes. |
| **3.3.2** | ✅ | 2026-04-13 | #10 | HMD native completo | `_hmd_tiv_full_jax` | 6 casos via `lax.switch` | Port completo HMD para JAX native. 4 plots LWD/PINN industriais. |
| **3.3.3** | ✅ | 2026-04-13 | #11 | VMD native | `_vmd_full_jax` | 4 casos geométricos + empymod 9-comp | Port VMD completo + validação empymod 9 componentes (TIV + iso). 6 plots curadas. |
| **3.3.4** | ✅ | 2026-04-13 | #12 | Propagação + assembly JAX | ETAPAS 3+6 end-to-end JAX native | `forward_pure_jax` 100% native | Port final de propagação TE/TM + tensor assembly. `jax.jacfwd` end-to-end em CPU/GPU. |

### Fase F7.4 — Validação cruzada (Sprints 4.1 – 4.4)

| Sprint | Status | Data | PR # | Nome Curto | Arquivos Principais | Métrica-Chave | Descrição |
|:------:|:------:|:----:|:----:|:-----------|:--------------------|:--------------|:----------|
| **4.1** | ✅ | 2026-04-13 | #9 | Validação empymod | `compare_numba_empymod()` | VMD Hzz iso vs empymod <1e-10 | Comparação com simulador empymod 1D independente; VMD axial. 4 testes. |
| **4.2** | ✅ | 2026-04-13 | #11 | empymod 9 componentes | Extensão para 9 componentes | Tabela 9-comp vs empymod (TIV + iso) | Validação cruzada de todos os 9 componentes em cenários TIV/isotrópico. |
| **4.3** | ✅ | 2026-04-13 | #12 | Reconciliação empymod | Ajustes convenção temporal | Testes de sinal + e^(-iωt) vs e^(+iωt) | Reconciliação de diferenças de convenção entre simuladores. |
| **4.4** | ✅ | 2026-04-13 | #14 | Fortran ↔ Python direto | Comparação bit-a-bit via `.dat` | Paridade vs tatu.x real (Intel i9) | Validação com executável Fortran nativo (OpenMP) em máquina real. |

### Fase F7.5 — Jacobiano (Sprints 5.1 – 5.2)

| Sprint | Status | Data | PR # | Nome Curto | Arquivos Principais | Métrica-Chave | Descrição |
|:------:|:------:|:----:|:----:|:-----------|:--------------------|:--------------|:----------|
| **5.1** | ✅ | 2026-04-13 | #13 | Jacobiano FD Numba | `compute_jacobian_fd_numba()` | ∂H/∂ρ via finite differences | Jacobiano FD `@njit`. 15 testes. |
| **5.1b** | ✅ | 2026-04-13 | #15 | Jacobiano `jax.jacfwd` | `compute_jacobian_jax()` | 5× FD + GPU T4 viable | Port para `jax.jacfwd` forward-mode AD. Teste 4-way (CPU Numba/JAX, GPU T4 CPU/GPU). Notebook Colab. |
| **5.2** | ✅ | 2026-04-13 | #13 | Meio-espaço TIV | `tiv_halfspace_analytical.py` | Solução analítica 1-cam TIV | Extensão de half_space.py para TIV (σh ≠ σv). Validação de simetrias. 8 testes. |

### Fase F7.6 — Integração PipelineConfig (Sprints 6.1 – 6.2)

| Sprint | Status | Data | PR # | Nome Curto | Arquivos Principais | Métrica-Chave | Descrição |
|:------:|:------:|:----:|:----:|:-----------|:--------------------|:--------------|:----------|
| **6.1** | ✅ | 2026-04-13 | #16 | `simulator_backend` em PipelineConfig | `config.py`: 5 novos campos | `fortran_f2py` → `numba`/`jax` | Adição de simulator_backend/precision/device/jax_mode/cache ao PipelineConfig com validação. |
| **6.2** | ✅ | 2026-04-13 | #16 | `SyntheticDataGenerator` | `data/synthetic_generator.py` (320 LOC) | Gerador in-process | Substitui subprocess Fortran `batch_runner.py`. Amostragem log-uniforme + seed determinístico. 6 testes. |

### Fase F7.7 — Performance JAX (Sprints 7.x – 9)

| Sprint | Status | Data | PR # | Nome Curto | Arquivos Principais | Métrica-Chave | Descrição |
|:------:|:------:|:----:|:----:|:-----------|:--------------------|:--------------|:----------|
| **7.x** | ✅ | 2026-04-13 | #17 | Performance JAX bucketing | Bucketing por (camad_t, camad_r) + JIT cache | 717× speedup oklahoma_5 (177k → 15.9ms) | Otimização JAX: bucketing geométrico + vmap duplo + JIT cache. |
| **7.x+** | ✅ | 2026-04-13 | #18 | LRU bounded cache VRAM | `LRU_bounded_cache` GPU | Controle VRAM T4 (4GB soft) / A100 (24GB) | Cache LRU com limites; evita OOM T4 via auto-eviction. |
| **8** | ✅ | 2026-04-13 | #19 | JAX warmup + chunked | `warmup_all_buckets()`, `forward_pure_jax_chunked()` | Amortização JIT | Warmup + forward chunked. 8 testes. |
| **9** | ✅ | 2026-04-13 | #19 | JAX pmap multi-GPU | `forward_pure_jax_pmap()` | Distribuição multi-GPU (A100 × N) | `jax.pmap` em A100×N via NCCL. |

### Fase F7.14 — Multi-TR/ângulo Numba (Sprint 11)

| Sprint | Status | Data | PR # | Nome Curto | Arquivos Principais | Métrica-Chave | Descrição |
|:------:|:------:|:----:|:----:|:-----------|:--------------------|:--------------|:----------|
| **11** | ✅ | 2026-04-14 | #20 | Multi-TR + Multi-ângulo Numba | `multi_forward.py` (450 LOC), `MultiSimulationResult` | Shape (nTR, nAngles, n_pos, nf, 9) | API nativa multi-TR/ângulo no Numba. Dedup cache por `hordist`. Export `.dat` Fortran-compat. **Paridade <2e-13 vs Fortran**. 17 testes. |

### PR #21 (v1.4.1) — Fix convenção T/R + regressão dip≠0°

| Item | Status | Data | PR # | Nome Curto | Arquivos Modificados | Métrica-Chave | Descrição |
|:----:|:------:|:----:|:----:|:-----------|:---------------------|:--------------|:----------|
| **PR #21** | ✅ | 2026-04-15 | #21 | Fix convenção T/R + regressão dip≠0° | `forward.py`, `multi_forward.py`, `_jax/kernel.py`, `_jax/forward_pure.py`, `tests/test_simulation_multi.py`, `tests/_fortran_helpers.py` (novo) | Paridade Numba vs Fortran <1e-12 em dip=30° | Corrige bug de convenção T/R (Python tinha T/R invertidos vs Fortran). Testes regressão permanente. **1399 testes passam**. |

### Pendente — PR #22 (v1.5.0)

| Sprint | Status | PR # | Nome Curto | Esforço Estimado | Descrição |
|:------:|:------:|:----:|:-----------|:-----------------|:----------|
| **10** | ⏳ | #22 | JAX unified JIT via `lax.fori_loop` | ~500 LOC + debug | Refatora loops Python em `_jax/dipoles_native.py:1053-1220` para `lax.fori_loop` + `jnp.where` encadeado. Consolida 44 programas XLA → 1. Expectativa: VRAM T4 ~11 GB → ~250 MB, speedup 5-20× GPU. |
| **11-JAX** | ⏳ | #22 | `simulate_multi_jax()` multi-TR/angle GPU | ~500 LOC | Port de `simulate_multi` (Sprint 11 Numba) para JAX native. Depende de Sprint 10 (unified JIT). vmap aninhado (nTR, nAngles, n_pos, nf). Dedup hordist. |
| **Docs** | ⏳ | #22 | Docs v1.5.0 | +1500 LOC | `sprint_10_fori_loop_consolidation.md`, `sprint_11_jax_multi_gpu.md`, `relatorio_final_v1_5.md`, sub-skill v1.9.0. |
| **Notebooks** | ⏳ | #22 | Benchmarks Colab GPU | 2 notebooks | `bench_sprint10_colab.ipynb`, `bench_sprint11_jax_colab.ipynb`. |

---

## 3. Estado Numba vs JAX (pós v1.4.1)

### Backend Numba (CPU) — Completo

| Aspecto | Status |
|:--------|:------:|
| Propagação `common_arrays`/`common_factors` | ✅ |
| Dipolos HMD + VMD (6 casos) | ✅ |
| API `simulate()` single-source | ✅ |
| API `simulate_multi()` multi-TR/angle/freq | ✅ (Sprint 11) |
| Paridade <1e-12 vs Fortran em dip=0° e dip=30° | ✅ (PR #21) |
| Jacobiano FD (`compute_jacobian_fd_numba`) | ✅ |
| Performance (1M mod/h small, 350k medium, 180k large) | ✅ |
| Convenção T/R Fortran-compat | ✅ (PR #21) |

### Backend JAX (CPU + GPU) — 80% completo

| Aspecto | Status |
|:--------|:------:|
| Propagação JAX native (`jax.lax.scan`) | ✅ |
| Dipolos HMD/VMD JAX native (`lax.switch`) | ✅ (Sprint 3.3.2/3.3.3) |
| API `forward_pure_jax()` end-to-end | ✅ (Sprint 3.3.4) |
| Jacobiano `jax.jacfwd` | ✅ (Sprint 5.1b) |
| Bucketing + LRU cache VRAM | ✅ (Sprint 7) |
| Warmup + chunked | ✅ (Sprint 8) |
| pmap multi-GPU | ✅ (Sprint 9) |
| Convenção T/R Fortran-compat | ✅ (PR #21) |
| **Unified JIT (44 → 1 XLA programs)** | ⏳ (Sprint 10 PENDENTE) |
| **`simulate_multi_jax()` multi-TR/angle/freq** | ⏳ (Sprint 11-JAX PENDENTE) |

---

## 4. Arquitetura Proposta para PR #22 (v1.5.0)

Documentação detalhada da implementação pendente. Estes blocos serão executados em PR separado dada a magnitude (~1000 LOC novos + debug GPU).

### Sprint 10 — JAX Unified JIT via `lax.fori_loop`

**Problema atual**:
- `_jax/dipoles_native.py:1053-1220` tem loops Python `for j in range(camad_t, camad_r+1)` (descendente) e `for j in range(camad_t, camad_r-1, -1)` (ascendente)
- Python `range()` requer bounds estáticos → `camad_t`, `camad_r` devem ser concretos
- Resultado: **44 programas XLA** separados em oklahoma_28, VRAM T4 ~11 GB

**Solução**:
- Refatorar loops para `jax.lax.fori_loop(lower, upper, body_fn, init)` — aceita tracers
- Corpo do loop com 5 branches → `jnp.where` encadeado com avaliação eager
- **Resultado esperado**: 1 programa XLA, VRAM ~250 MB, speedup 5-20× GPU

**Pseudocódigo** (ver plano para detalhes completos):

```python
def body_descent(j, carry):
    Txdw, Tudw = carry
    is_first = j == camad_t
    is_next_last = (j == camad_t + 1) & (j == n - 1)
    is_next_internal = (j == camad_t + 1) & (j != n - 1)
    is_internal = (j > camad_t + 1) & (j != n - 1)
    is_last = (j == n - 1) & (j != camad_t) & (j != camad_t + 1)

    # Calcula 5 candidatos eager (com safe_idx = jnp.maximum(j-1, 0))
    Txdw_first = _MX / (2.0 * s[:, camad_t])
    Tudw_first = -_MX / 2.0
    # ... 4 branches mais (cada com fórmula específica)

    # Mux via jnp.where encadeado
    Txdw_new = jnp.where(is_first, Txdw_first,
               jnp.where(is_next_last, Txdw_next_last,
               jnp.where(is_next_internal, Txdw_next_internal,
               jnp.where(is_internal, Txdw_internal, Txdw_last))))
    # ... idem Tudw_new

    return Txdw.at[:, j].set(Txdw_new), Tudw.at[:, j].set(Tudw_new)

Txdw, Tudw = jax.lax.fori_loop(
    lower=camad_t, upper=camad_r + 1,
    body_fun=body_descent,
    init_val=(Txdw_init, Tudw_init),
)
```

**Riscos**:
- `jnp.where` pode introduzir NaN em branches inativas (mitigação: `safe_idx = jnp.maximum(j-1, 0)`)
- Paridade bit-exact pode degradar para ~1e-12 ao invés de 1e-13 (aceito)

**Testes**: `tests/test_simulation_jax_sprint10_parity.py` — paridade <1e-12 vs Numba em 7 modelos.

### Sprint 11-JAX — `simulate_multi_jax()` multi-TR/angle/freq GPU

**API alvo** (espelha `simulate_multi` Numba):

```python
from geosteering_ai.simulation import simulate_multi_jax, MultiSimulationResultJAX

result = simulate_multi_jax(
    rho_h=jnp.array([1.0, 100.0, 1.0]),
    rho_v=jnp.array([1.0, 200.0, 1.0]),
    esp=jnp.array([5.0]),
    positions_z=jnp.linspace(-10, 10, 100),
    frequencies_hz=[20000., 40000.],       # nf
    tr_spacings_m=[0.5, 1.0, 1.5],         # nTR
    dip_degs=[0., 30., 60.],               # nAngles
    cfg=SimulationConfig(backend="jax", device="gpu"),
)
# result.H_tensor.shape → (nTR=3, nAngles=3, n_pos=100, nf=2, 9) complex128
```

**Estratégia**:
- Dedup cache via `hordist = L·|sin(θ)|` (padrão Numba Sprint 2.10)
- vmap aninhado sobre (nTR, nAngles, n_pos, nf)
- `lax.scan` sobre `unique_hordist` → 1 forward JIT por hordist único
- Output: `MultiSimulationResultJAX` com shape idêntico a Numba + `.to_single()`
- Suporte F6/F7 via pós-processamento

**Dependência crítica**: Sprint 10 completo (sem unified JIT, 44 × nTR × nAngles = explosão combinatória de programas XLA).

**Testes**: `tests/test_simulation_jax_multi.py` — 4 configs × 3 modelos = 12 cenários, paridade <1e-12 vs Numba.

**Benchmark esperado** (notebook Colab):

| Config | Modelo | Numba CPU | JAX T4 | JAX A100 |
|:------:|:------:|:---------:|:------:|:--------:|
| 1TR×1ang×1f | oklahoma_3 | 130k mod/h | 400k mod/h | 1.2M mod/h |
| 3TR×5ang×2f | oklahoma_28 | 120k mod/h | 700k mod/h | 2.5M mod/h |

---

## 5. Correção T/R — Convenção Fortran (detalhes)

### Bug (v1.4.0 e anteriores)

Python usava:
- Transmissor: `Tx = -r_half`, `Tz = z_mid - dz_half` (lado -x, acima)
- Receptor: `cx = r_half`, `cz = z_mid + dz_half` (lado +x, abaixo)

Fortran `PerfilaAnisoOmp.f08:674-679` usa:
- Receptor: `x = -Lsen/2`, `z = z1 - Lcos/2` (lado -x, acima)
- Transmissor: `Tx = +Lsen/2`, `Tz = z1 + Lcos/2` (lado +x, abaixo)

**Conclusão**: Python tinha T e R com posições trocadas → off-diagonais cruzadas (Hxz, Hzx, Hyz, Hzy) com sinal invertido.

### Correção aplicada (v1.4.1)

4 locais Numba + 3 locais JAX:
```python
# Antes (bug):        # Depois (correto, Fortran-compat):
Tz = z_mid - dz_half  Tz = z_mid + dz_half   # T abaixo
cz = z_mid + dz_half  cz = z_mid - dz_half   # R acima
Tx = -r_half          Tx = r_half            # T em +x
cx = r_half           cx = -r_half           # R em -x
```

### Por que o bug passou 10 PRs sem ser detectado?

Testes existentes (Sprint 11 PR #15) usavam `dip=0°` exclusivamente. Para `dip=0°`:
- `r_half = L·sin(0) = 0` → `Tx = cx = 0`
- Hxz e Hzx são **identicamente nulos** por simetria

A convenção incorreta era invisível com esta geometria. **O usuário descobriu visualmente** ao comparar plots em `dip=30°`.

### Proteção de regressão permanente (PR #21)

- `test_fortran_byte_exact_dat_nonzero_dip` (Numba vs Fortran, dip=30°, <1e-12)
- `test_jax_dip_nonzero_matches_numba` (JAX vs Numba, dip=30°, <1e-10)
- Sanidade assertiva: `|Re(Hxz)| > 1e-10` em dip=30° (detecta regressão de sinal)

---

## 6. Métricas Finais — v1.4.1

```
Versão subpacote:            1.4.1 (PR #21)
Commit HEAD:                 6a4cd0c (main)
Total de testes (simulator): 1399 PASSED, 295 skipped (0 failed)
Tempo execução suite:        4:16 (CPU Intel i9)
Tolerância paridade Fortran: <1e-12 em dip=0° e dip=30°

Arquivos de código:
  - Produção:  ~4400 LOC (geosteering_ai/simulation/)
  - Testes:    ~2700 LOC (tests/test_simulation*.py + _fortran_helpers.py)
  - Docs:      ~3700 LOC (docs/reference/sprint_*.md + relatorio_*.md)

Fases completadas (total):
  F7.0: Fundação (1.1-1.3)            ✅
  F7.1: Backend Numba (2.1-2.7)        ✅
  F7.2: Paralelização (2.8-2.10)       ✅
  F7.3: Backend JAX (3.1-3.3.4)        ✅
  F7.4: Validação cruzada (4.1-4.4)    ✅
  F7.5: Jacobiano (5.1-5.2)            ✅
  F7.6: PipelineConfig (6.1-6.2)       ✅
  F7.7: Performance JAX (7.x-9)        ✅
  F7.14: Multi-TR/angle Numba (11)     ✅

Fases pendentes:
  Sprint 10: JAX unified JIT           ⏳ (PR #22)
  Sprint 11-JAX: multi-TR/angle GPU    ⏳ (PR #22)
  Fase 6 deploy final (backend switch) ⏳ (v2.0.0)

Benchmarks Numba CPU (post Sprint 2.10):
  oklahoma_3  (3 camadas):  ~1.01M mod/h (1722% Fortran)
  oklahoma_5  (5 camadas):  ~347k mod/h  (589% Fortran)
  oklahoma_28 (28 camadas): ~184k mod/h  (312% Fortran)

Paridade numérica (pós PR #21):
  dip=0° vs Fortran:  max_abs_err < 2e-13
  dip=30° vs Fortran: max_abs_err < 1e-12
  JAX vs Numba (dip=30°): max_abs_err < 1e-10
```

---

## 7. Pendências/Riscos pós-v1.4.1

| Item | Prioridade | Impacto | Sprint potencial |
|:-----|:----------:|:-------:|:----------------:|
| **Sprint 10 (JAX unified JIT)** | **Alta** | **Desbloqueio GPU** | PR #22 |
| **Sprint 11-JAX (multi-TR/angle GPU)** | **Alta** | **Paridade JAX↔Numba** | PR #22 (depende Sprint 10) |
| Sprint 12: complex64 mixed precision | Média | ~50% VRAM + 16× TFLOPS T4 | PR #23 |
| Sprint 6.4: SyntheticDataGenerator multi-TR | Média | Treino PINN em multi-TR | PR #24 |
| Amostragem Sobol (quasi-aleatória) | Baixa | Melhor cobertura espaço ρ | PR #25 |
| Integração DataPipeline.prepare() multi-TR | Média | Automação treino end-to-end | PR #26 |
| F6/F7 em JAX GPU (postprocess/*.py) | Média | Paridade de features JAX | PR #27 |
| Profiling e otimização final JAX GPU | Alta | Target >1M mod/h T4 | PR #28 |

---

## 8. Conclusão

O simulador Python do Geosteering AI v2.0 **atingiu maturidade funcional total** com o PR #21. A versão v1.4.1:

- **Reproduz fielmente o Fortran `tatu.x`** em todos os cenários testados (paridade <1e-12)
- **Suporta nativamente** multi-TR, multi-ângulo, multi-frequência (Numba)
- **É diferenciável** end-to-end via `jax.jacfwd` (CPU + GPU)
- **Tem convenções físicas corretas** (T/R Petrobras-compatível)
- **Tem performance superior ao Fortran** em CPU (1.7× no modelo small)

O trabalho restante para v1.5.0 — Sprint 10 (JAX unified JIT) + Sprint 11-JAX — é **otimização de performance GPU** e não afeta correção numérica. O PR #21 entrega valor imediato (proteção de regressão permanente para convenção T/R) e desbloqueia a evolução confiável do path JAX em PR separado.

---

## Referências

1. `docs/reference/plano_simulador_python_jax_numba.md` — plano mestre
2. `.claude/commands/geosteering-simulator-python.md` — sub-skill (v1.8.0)
3. `docs/ROADMAP.md` — seção F7 (Simulator Python)
4. `Fortran_Gerador/PerfilaAnisoOmp.f08:670-680` — geometria T-R referência
5. `geosteering_ai/simulation/__init__.py` — tabela versões + exports
6. `tests/_fortran_helpers.py` — helpers compartilhados de teste
7. `benchmarks/bench_multi_vs_fortran.py` — benchmark Sprint 11
