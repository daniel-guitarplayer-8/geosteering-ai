# Plano de Implementação — Simulador Python Otimizado (JAX + Numba)

**Projeto:** Geosteering AI v2.0 · **Data:** 2026-04-10  
**Objetivo:** Reimplementar o simulador EM 1D TIV Fortran (`PerfilaAnisoOmp.f08` v10.0) em Python puro com equivalência CPU (Fortran/OpenMP) e aceleração GPU (XLA/CUDA).

---

## 1. Resumo Executivo

O simulador Fortran atingiu **58.856 mod/h** em CPU (i9-9980HK, 8 threads) após 5 fases de otimização e está funcional/validado. O projeto Python **não possui** nenhum código nativo de forward modeling EM — toda simulação é delegada via `subprocess` ou `tatu_f2py.so`. Esse acoplamento impede três capacidades críticas para v2.0/v3.0:

1. **Auto-diferenciação** (`∂H/∂ρ` para PINNs) — inviável em Fortran via FD em tempo real
2. **GPU acceleration** — Fortran/OpenMP limitado a CPU; datasets > 1M modelos exigem horas
3. **On-the-fly data generation** durante treinamento — elimina I/O de disco como gargalo

**Recomendação arquitetural (consolidada dos 3 documentos de planejamento):**

| Track | Horizonte | Tecnologia | Papel |
|:---|:---|:---|:---|
| **Track 1** | Imediato | `tatu_f2py.so` (já existe) | Backend de validação e ground truth numérico |
| **Track 2** | Principal | **JAX (`jit`+`vmap`+`pmap`+`jacfwd`)** | Forward model + Jacobiano + GPU + integração DL |
| **Track 3** | Complementar | **Numba (`njit`+`prange`)** | CPU parity com OpenMP para nós sem GPU (Colab CPU, worker batch) |
| **Track 4** | Futuro | Numba CUDA | Kernels especializados para geração massiva de datasets (>10M modelos) |

**Decisão central:** implementar o simulador como **módulo único com dois backends comutáveis em runtime** (JAX e Numba), compartilhando a mesma topologia funcional. A API externa é única (`simulate(config) → H_tensor`), permitindo benchmarks A/B e fallback transparente.

---

## 2. Estado Atual — O que Existe vs O que Falta

### 2.1 O que existe em Python (do inventário)

| Categoria | Arquivos | Status |
|:---|:---|:---|
| **Wrapper Fortran** | `tatu_f2py.so` + `buildValidamodels.py` + `validate_jacobian.py` | Funcional |
| **Geração de modelos** | `fifthBuildTIVModels.py` (Sobol QMC) | Funcional, 1381 LOC |
| **Batch orchestration** | `batch_runner.py` (ProcessPoolExecutor) | Funcional, 1462 LOC |
| **Parser binário** | `geosteering_ai/data/loading.py` | Funcional |
| **Surrogate neural** | `geosteering_ai/models/surrogate.py` (TCN/ModernTCN) | Parcial — precisa treino |
| **PINN losses** | `geosteering_ai/losses/pinns.py` (8 cenários) | Funcional, mas dependente de Fortran para "oracle" |

### 2.2 O que **não** existe (gap para fechar)

| Componente | Criticidade | Localização futura |
|:---|:---:|:---|
| Quadratura Hankel (filtros J0/J1 Kong/Werthmüller/Anderson) | 🔴 Alta | `simulation/hankel.py` |
| `commonarraysMD` (recursão TE/TM por camada) | 🔴 Alta | `simulation/propagation.py` |
| `commonfactorsMD` (14 exponenciais por posição) | 🔴 Alta | `simulation/propagation.py` |
| `hmd_TIV_optimized` + `vmd_optimized` (kernels dipolo) | 🔴 Alta | `simulation/dipoles.py` |
| Rotação de tensor `RtHR` | 🟡 Média | `simulation/rotation.py` |
| Montagem 9-componentes + gather por posição | 🔴 Alta | `simulation/forward.py` |
| Jacobiano `jax.jacfwd` / FD centrada | 🟢 Baixa (trivial com JAX) | `simulation/jacobian.py` |
| Multi-TR loop (F1) | 🟡 Média | `simulation/forward.py` (vmap externo) |
| Tilted antennas (F7) | 🟢 Baixa (pós-processamento) | `simulation/postprocess.py` |
| Compensação midpoint (F6) | 🟢 Baixa (pós-processamento) | `simulation/postprocess.py` |
| Loader de pesos Hankel | 🟡 Média | `simulation/filters.py` |

**Ponto crítico:** os 3 conjuntos de pesos (Kong 61pt, Werthmüller 201pt, Anderson 801pt) estão **hardcoded no Fortran** em `filtersv2.f08`. A primeira tarefa técnica será **extrair** essas tabelas para arquivos `.npz` ou `.py` importáveis.

---

## 3. Decisões Arquiteturais Consolidadas

### 3.1 Por que JAX como backend principal

| Requisito v2.0/v3.0 | Satisfação JAX |
|:---|:---|
| Auto-diferenciação `∂H/∂ρ` (PINNs) | ✅ `jax.jacfwd` nativo, exato, sem overhead FD |
| GPU (NVIDIA + Apple Metal) | ✅ XLA compila para CUDA, ROCm, Metal sem reescrita |
| Batch de milhares de modelos | ✅ `jax.vmap` (auto-vectorização sem loops Python) |
| Distribuição multi-GPU | ✅ `jax.pmap` + sharding |
| Compatibilidade com Keras 3.x | ✅ Keras 3 roda nativamente sobre JAX |
| Funções puras (reprodutibilidade) | ✅ Paradigma funcional elimina mutação acidental |
| Compilação JIT agressiva | ✅ XLA faz kernel fusion automaticamente |
| Suporte a `complex128` em GPU | ⚠️ Suportado, mas precisa verificar performance vs `complex64` |

### 3.2 Por que Numba como backend CPU complementar

| Requisito | Satisfação Numba |
|:---|:---|
| Equivalência OpenMP Fortran | ✅ `prange` + `parallel=True` |
| Zero dependência de XLA/CUDA toolkit | ✅ Funciona em qualquer máquina com LLVM |
| Performance CPU ≥ 80% do Fortran | ✅ `@njit(fastmath=True, cache=True)` atinge esse patamar |
| Fallback quando GPU indisponível | ✅ Mesmo código, decorator diferente |
| Debug mais simples que JAX tracing | ✅ Stack traces Python legíveis |

### 3.3 Por que **não** PyTorch, CuPy, TensorFlow nativo, Taichi

| Descartado | Motivo |
|:---|:---|
| PyTorch | **Proibido** pelo CLAUDE.md (framework exclusivo é TF/Keras) |
| TensorFlow puro (sem JAX) | `tf.function` é menos eficiente em kernels com recursão; auto-diff complexa |
| CuPy | Exige reescrita CUDA manual; sem auto-diff |
| Taichi Lang | Ecossistema pequeno; baixa maturidade em auto-diff complex128 |
| Cython puro | Overhead de desenvolvimento > Numba; sem GPU |

### 3.4 Princípio de design: **dois backends, uma API**

```
┌─────────────────────────────────────────────┐
│  geosteering_ai.simulation.simulate(...)   │  ← API pública única
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   backend="jax"         backend="numba"
        │                     │
   ┌────┴────┐           ┌────┴────┐
   │ JAX JIT │           │ Numba   │
   │ + vmap  │           │ njit +  │
   │ + XLA   │           │ prange  │
   └────┬────┘           └────┬────┘
        │                     │
        ▼                     ▼
      GPU                  CPU (OpenMP)
```

A mesma **topologia funcional** (assinaturas, fluxo de dados, caches Phase 4) é escrita duas vezes, mas em módulos separados (`_jax.py` e `_numba.py`), compartilhando o orquestrador `forward.py` via dispatcher.

---

## 4. Stack Tecnológico Detalhado

### 4.1 Dependências novas

```toml
# pyproject.toml — extras proposto
[project.optional-dependencies]
simulation = [
    "jax[cuda12]>=0.4.30; platform_system=='Linux'",
    "jax[metal]>=0.4.30; platform_system=='Darwin'",
    "jaxlib>=0.4.30",
    "numba>=0.59",
    "scipy>=1.11",       # scipy.special.jn (validação)
    "empymod>=2.3",      # referência científica para validação cruzada
]
```

**`empymod`** é uma biblioteca de Werthmüller (mesmo autor do filtro de 201pt) — servirá como **terceira fonte de verdade** para validação (além do Fortran e da solução analítica de half-space).

### 4.2 Configuração em `PipelineConfig`

```python
# geosteering_ai/config.py — novos campos
@dataclass
class PipelineConfig:
    # ... campos existentes ...

    # ── Simulador Python ────────────────────────────────────
    simulator_backend: Literal["jax", "numba", "fortran_f2py"] = "fortran_f2py"
    simulator_precision: Literal["complex64", "complex128"] = "complex128"
    simulator_device: Literal["cpu", "gpu", "auto"] = "auto"
    simulator_filter: Literal["kong", "werthmuller", "anderson"] = "werthmuller"
    simulator_cache_phase4: bool = True      # replica Fase 4 Fortran
    simulator_jacobian_mode: Literal["fd_centered", "jax_jacfwd", "none"] = "none"
    simulator_validate_on_init: bool = False  # smoke test bit-exato vs Fortran
```

---

## 5. Arquitetura do Módulo `geosteering_ai/simulation/`

```
geosteering_ai/simulation/
├── __init__.py                 ← API pública: simulate(), SimulationConfig
├── config.py                   ← SimulationConfig (subconjunto de PipelineConfig)
├── forward.py                  ← Orquestrador: dispatcher backend, loop Multi-TR
├── _jax/
│   ├── __init__.py
│   ├── propagation.py          ← commonarraysMD_jax, commonfactorsMD_jax
│   ├── dipoles.py              ← hmd_tiv_jax, vmd_jax
│   ├── hankel.py               ← quadratura Hankel vectorizada
│   ├── rotation.py             ← RtHR_jax
│   ├── jacobian.py             ← jax.jacfwd wrapper
│   └── kernel.py               ← fieldsinfreqs_cached_jax
├── _numba/
│   ├── __init__.py
│   ├── propagation.py          ← commonarraysMD_numba (@njit)
│   ├── dipoles.py              ← hmd_tiv_numba, vmd_numba
│   ├── hankel.py               ← quadratura Hankel em prange
│   ├── rotation.py             ← RtHR_numba
│   └── kernel.py               ← fieldsinfreqs_cached_numba
├── filters/
│   ├── kong_61pt.npz           ← pesos extraídos de filtersv2.f08
│   ├── werthmuller_201pt.npz
│   ├── anderson_801pt.npz
│   └── loader.py               ← load_filter(name) → krJ0J1, wJ0, wJ1
├── postprocess.py              ← tilted (F7), compensação (F6)
├── geometry.py                 ← posTR, profundidades, interfaces
├── validation/
│   ├── half_space.py           ← solução analítica (2 camadas) para sanity check
│   ├── compare_fortran.py      ← diff JAX/Numba vs tatu_f2py
│   └── compare_empymod.py      ← diff vs empymod (2ª fonte de verdade)
└── benchmarks/
    ├── bench_forward.py        ← mod/h por backend+device+filter
    ├── bench_jacobian.py       ← speedup jax.jacfwd vs FD
    └── bench_scaling.py        ← batch size vs latência
```

---

## 6. Mapeamento Fortran → JAX/Numba (Estrutura Matemática)

### 6.1 Pipeline de dados preservado

```
Fortran (OpenMP)                     Python (JAX/Numba)
─────────────────                    ──────────────────
model.in → RunAnisoOmp               SimulationConfig (dataclass)
   │                                    │
   ▼                                    ▼
perfila1DanisoOMP                    simulate(config)
   │                                    │
   ├─ alloc ws_pool(0:T-1)              ├─ [JAX] pytree de buffers stateless
   │                                    │  [Numba] arrays pré-alocados por thread
   │                                    │
   ├─ LOOP itr=1..nTR                   ├─ jax.vmap(over dTR)  ou  for itr in prange(nTR)
   │  │                                 │     │
   │  ├─ pre-cache Phase 4              │     ├─ commonarraysMD → cache (npt, n, nf)
   │  │  (commonarraysMD por k)         │     │
   │  │                                 │     │
   │  ├─ !$omp parallel do k            │     ├─ jax.vmap(over theta)
   │  │  !$omp parallel do j            │     │     jax.vmap(over j)
   │  │  │                              │     │     │
   │  │  └─ fieldsinfreqs_cached_ws     │     │     └─ kernel(cache, posTR, ang)
   │  │                                 │     │          │
   │  │     ├─ commonfactorsMD          │     │          ├─ commonfactorsMD_jax
   │  │     ├─ hmd_TIV_optimized        │     │          ├─ hmd_tiv_jax
   │  │     ├─ vmd_optimized            │     │          ├─ vmd_jax
   │  │     └─ RtHR                     │     │          └─ rtHR_jax → H(3,3)
   │  │                                 │     │
   │  ├─ writes_files (.dat)            │     └─ return H_tensor[:, :, :, :]
   │  └─ [F10] compute_jacobian_fd      │
   │                                    └─ [opcional] jax.jacfwd(simulate, argnums=rho)
   └─ [F6] compensation post-proc          [opcional] postprocess.compensate()
```

### 6.2 Hot spots e estratégia de aceleração

| Subroutina Fortran | Custo relativo | Estratégia JAX | Estratégia Numba |
|:---|:---:|:---|:---|
| `commonarraysMD` (recursão TE/TM) | **~55%** | `@jax.jit` + vmap em `(npt, layer, freq)`; recursão via `jax.lax.scan` sobre camadas | `@njit(fastmath=True)` + `prange` em npt; recursão serial em camadas |
| `commonfactorsMD` (14 exp complex) | **~20%** | vmap em posições; XLA fusione exponenciais | `@njit` com arrays locais |
| `hmd_TIV_optimized_ws` | **~12%** | Puro NumPy (ops vectorizadas); sem branches | `@njit` inline |
| `vmd_optimized_ws` | **~8%** | idem HMD | idem |
| Quadratura Hankel (dot product 201pt) | **~3%** | `jnp.einsum('i,ij->j', w, kernel_values)` | `np.dot` em `@njit` |
| `RtHR` (rotação 3×3) | **~2%** | einsum | `@njit` |

**Insight crítico da análise Fortran:** a invariância de `commonarraysMD` em `j` (posição) é a **razão do speedup 4×** da Fase 4. Em JAX isso é preservado naturalmente: `vmap(compute_measurement)` recebe o cache como **closure** (argumento fechado), e XLA não recomputa.

### 6.3 Estrutura de caches Phase 4 em JAX

```python
# Pseudocódigo ilustrativo da estratégia
@jax.jit
def precompute_cache_phase4(r_k, freqs, layers_h, layers_v, filter_weights):
    """Cache de 9 arrays shape (npt, n_layers, nf) — invariante em j."""
    eta = 1.0 / jnp.stack([layers_h, layers_v], axis=-1)  # (n, 2)

    def per_frequency(freq):
        zeta = 1j * 2 * jnp.pi * freq * MU_0
        return commonarraysMD_jax(r_k, zeta, eta, filter_weights.krJ0J1)

    cache = jax.vmap(per_frequency)(freqs)  # (nf, 9, npt, n)
    return cache  # tuple ou pytree
```

A chave é que `cache` é um **pytree imutável**, e `jax.vmap` sobre posições fecha sobre ele sem copiar memória.

---

## 7. Estratégia de Paralelização — CPU e GPU

### 7.1 Paralelismo hierárquico (mapeado ao Fortran)

| Nível | Dimensão Fortran | JAX (GPU) | Numba (CPU) |
|:---:|:---|:---|:---|
| **L1** | Modelos em batch (externo) | `jax.vmap` → XLA batch | `prange` em loop externo |
| **L2** | `nTR` (pares T-R) | `jax.vmap` | Loop serial (nTR ≤ 3) |
| **L3** | `ntheta` (ângulos) | `jax.vmap` | `prange` (Fase 5b `if(ntheta>1)`) |
| **L4** | `nmed` (posições) — **principal** | `jax.vmap` | `prange schedule=guided` |
| **L5** | `nf` (frequências) | `jax.vmap` (inner) | loop serial ou `prange` |
| **L6** | `npt` (pontos Hankel) | vectorização implícita | SIMD via `fastmath` |
| **L7** | `2n` perturbações (Jacobiano) | `jax.jacfwd` (auto) ou vmap | `prange` |

**Decisão de schedule:** no Numba, replicar a Fase 2 do Fortran exige `prange` com `schedule='guided'` — mas Numba não expõe isso diretamente. **Workaround:** pré-computar `nmed(k)` e usar `prange` simples (tolerar ~5% de load imbalance vs OpenMP guided).

### 7.2 Estratégia GPU (JAX/XLA)

```python
# Topologia de compilação sugerida
@partial(jax.jit, static_argnames=('n_layers', 'nf', 'ntheta', 'nmed', 'filter_type'))
def simulate_jax(
    rho_h: jax.Array,        # (batch, n_layers)
    rho_v: jax.Array,        # (batch, n_layers)
    thicknesses: jax.Array,  # (batch, n_layers-1)
    freqs: jax.Array,        # (nf,)
    theta: jax.Array,        # (ntheta,)
    dTR: jax.Array,          # (nTR,)
    p_med: float,
    z1: float,
    tj: float,
    n_layers: int, nf: int, ntheta: int, nmed: int, filter_type: str,
):
    # ...
    return H_tensor  # (batch, nTR, ntheta, nmed, nf, 3, 3) complex128
```

**Pontos críticos para XLA:**

1. **`static_argnames`** é obrigatório para dimensões que participam de `reshape`/`broadcast`
2. Evitar `jnp.where` com branches grandes — XLA gera ambos os ramos; substituir por aritmética quando possível (e.g. `jnp.heaviside`)
3. `complex128` em GPU NVIDIA: funciona mas ~1.5× mais lento que `complex64`. Oferecer ambos via `simulator_precision`
4. **Donate buffers** no `jit` para reduzir cópias de memória em batches grandes

### 7.3 Estratégia CPU (Numba + OpenMP)

```python
@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def simulate_numba(rho_h, rho_v, thicknesses, freqs, theta, dTR, p_med, z1, tj):
    nTR, ntheta, nmed_max, nf = ...
    H_out = np.zeros((nTR, ntheta, nmed_max, nf, 3, 3), dtype=np.complex128)

    for itr in range(nTR):                      # serial (nTR pequeno)
        for k in numba.prange(ntheta):           # Fase 5b equivalente
            r_k = dTR[itr] * abs(np.sin(theta[k]))
            cache = _precompute_cache(r_k, freqs, ...)  # (9, nf, npt, n)

            for j in numba.prange(nmed_max):     # hot loop
                posTR = _geometry(j, theta[k], p_med, z1)
                H_out[itr, k, j, :, :, :] = _kernel(cache, posTR, theta[k])

    return H_out
```

**Nota:** `numba.prange` aninhado funciona mas com overhead — em Numba 0.59+ apenas o **loop mais externo** é efetivamente paralelizado. Estratégia: achatar `(k, j)` em um único índice linear quando `ntheta * nmed_max > num_cpus * 16`.

---

## 8. Plano de Implementação em Fases

### Fase 0 — Preparação (antes de codificar)

| # | Tarefa | Entregável |
|:---:|:---|:---|
| 0.1 | Extrair pesos Hankel (Kong/Werthmüller/Anderson) do `filtersv2.f08` | `simulation/filters/*.npz` |
| 0.2 | Escrever golden dataset de validação (100 modelos canônicos via `tatu_f2py`) | `tests/fixtures/golden_100.npz` |
| 0.3 | Implementar solução analítica de half-space (2 camadas, VTI) | `simulation/validation/half_space.py` |
| 0.4 | Criar skeleton do módulo `simulation/` com `__init__.py` vazios | Estrutura de diretórios |
| 0.5 | Adicionar dependências ao `pyproject.toml` (`jax`, `numba`, `empymod`) | `pyproject.toml` atualizado |
| 0.6 | Setup de testes: `tests/simulation/` com pytest fixtures | Infraestrutura de teste |

### Fase 1 — Backend Numba CPU (prioridade de mitigação de risco)

**Por que Numba primeiro?** Paradigma imperativo (igual ao Fortran), debug mais fácil, sem overhead de tracing. Serve como **ponte conceitual** e terceira fonte de verdade para validar o JAX depois.

| # | Tarefa | Dependências |
|:---:|:---|:---|
| 1.1 | `hankel.py` — quadratura digital `∫ f(kr) J_ν(kr·r) dkr` | 0.1 |
| 1.2 | `propagation.py` — `commonarraysMD` (recursão TE/TM) | 1.1 |
| 1.3 | `propagation.py` — `commonfactorsMD` (fatores de onda) | 1.2 |
| 1.4 | `dipoles.py` — `hmd_tiv` + `vmd` (kernels) | 1.2, 1.3 |
| 1.5 | `rotation.py` — `RtHR` (rotação de tensor 3×3) | — |
| 1.6 | `kernel.py` — `fieldsinfreqs_cached` (integra 1.2–1.5) | 1.1–1.5 |
| 1.7 | `forward.py` — orquestrador com loops Multi-TR + cache Phase 4 | 1.6 |
| 1.8 | Validação: `simulate_numba` vs `tatu_f2py` (erro < 1e-10) | 0.2, 0.3 |
| 1.9 | Benchmark: throughput mod/h em single model + batch | 1.7 |

**Critério de saída da Fase 1:** Numba single-thread atinge ≥ 80% da performance Fortran single-thread. Numba 8-thread atinge ≥ 70% da performance Fortran 8-thread.

### Fase 2 — Backend JAX CPU

| # | Tarefa | Dependências |
|:---:|:---|:---|
| 2.1 | Port de `hankel.py` para JAX (`jnp.einsum`) | 1.1 |
| 2.2 | Port de `commonarraysMD` com `jax.lax.scan` sobre camadas | 1.2 |
| 2.3 | Port de `commonfactorsMD` (vectorização direta) | 1.3 |
| 2.4 | Port de dipolos + rotação | 1.4, 1.5 |
| 2.5 | `kernel.py` JAX com `@jax.jit(static_argnames=...)` | 2.1–2.4 |
| 2.6 | `forward.py` JAX com `jax.vmap` hierárquico | 2.5 |
| 2.7 | Validação: `simulate_jax(cpu)` vs Numba (erro < 1e-12) | 1.8 |
| 2.8 | Validação cruzada com **empymod** (terceira fonte) | 2.7 |

### Fase 3 — Backend JAX GPU

| # | Tarefa | Dependências |
|:---:|:---|:---|
| 3.1 | Ajuste de `static_argnames` para evitar recompilação | 2.6 |
| 3.2 | Profiling com `jax.profiler` — identificar gargalos XLA | 3.1 |
| 3.3 | Suporte a `complex64` opcional (config `simulator_precision`) | 2.6 |
| 3.4 | `jax.vmap` externo para batch de modelos (10k–100k) | 3.1 |
| 3.5 | Validação numérica GPU vs CPU (tolerância maior: 1e-8) | 3.4 |
| 3.6 | Benchmark T4 / L4 / A100 (Colab Pro+) | 3.4 |

**Critério de saída da Fase 3:** ≥ 200k mod/h em T4, ≥ 500k mod/h em A100.

### Fase 4 — Jacobiano (F10 em Python)

| # | Tarefa | Dependências |
|:---:|:---|:---|
| 4.1 | `jax.jacfwd(simulate_jax, argnums=(rho_h, rho_v))` | 2.6 |
| 4.2 | Comparar precisão vs `compute_jacobian_fd` do Fortran | 4.1, 0.2 |
| 4.3 | Integrar em `geosteering_ai/losses/pinns.py` (cenário "physics") | 4.1 |
| 4.4 | Benchmark: `jax.jacfwd` vs FD centrada (speedup esperado 2n×) | 4.1 |

### Fase 5 — Features v10.0 (F1, F5, F6, F7)

| # | Tarefa |
|:---:|:---|
| 5.1 | **F1 Multi-TR** — `jax.vmap` sobre `dTR` (quase trivial) |
| 5.2 | **F5 Frequências arbitrárias** — já natural em JAX (dim vectorizada) |
| 5.3 | **F7 Tilted antennas** — `postprocess.py` puro (ops lineares) |
| 5.4 | **F6 Compensação midpoint** — `postprocess.py` (diff + log10) |

### Fase 6 — Integração com `geosteering_ai/`

| # | Tarefa |
|:---:|:---|
| 6.1 | Adapter: `simulation.simulate(config)` retorna array compatível com `loading.py` |
| 6.2 | Substituir backend `fortran_f2py` por `jax` em `losses/pinns.py` cenário oracle |
| 6.3 | Substituir subprocess+batch_runner por `jax.vmap` em geração on-the-fly |
| 6.4 | Callback de treinamento que gera dados por epoch com `simulation.simulate` |

### Fase 7 — Numba CUDA (opcional, longo prazo)

Kernels CUDA especializados para geração de datasets > 10M modelos. Apenas se JAX GPU se revelar insuficiente.

---

## 9. Passos Iniciais — Primeira Sprint (Fase 0 + início da Fase 1)

Ordem executável, sem ambiguidade:

### Sprint 1.1 — Infraestrutura

1. **Extrair pesos Hankel do Fortran**
   - Ler `Fortran_Gerador/filtersv2.f08` linhas 6–120 (Kong 61pt) e demais filtros
   - Escrever script `scripts/extract_hankel_weights.py` que parseia e salva:
     - `simulation/filters/kong_61pt.npz` — chaves `kr`, `wJ0`, `wJ1`
     - `simulation/filters/werthmuller_201pt.npz`
     - `simulation/filters/anderson_801pt.npz`
   - **Validar:** carregar com `np.load` e verificar shapes corretas (61, 201, 801)

2. **Gerar golden dataset de validação**
   - Criar `tests/simulation/fixtures/generate_golden.py`
   - Invocar `tatu_f2py.simulate_v8` com 100 modelos canônicos (n_layers ∈ {3, 5, 10, 20}, freq ∈ {1 kHz, 20 kHz, 100 kHz, 1 MHz}, ângulos ∈ {0°, 30°, 60°})
   - Salvar `golden_100.npz` com entradas + saídas
   - Este será o **ground truth** para todos os testes subsequentes

3. **Criar esqueleto do módulo**
   - `mkdir -p geosteering_ai/simulation/{_jax,_numba,filters,validation,benchmarks}`
   - Criar `__init__.py` vazios + `config.py` com `SimulationConfig` dataclass
   - Adicionar ao `geosteering_ai/__init__.py`: `from . import simulation`

4. **Setup de testes**
   - `tests/simulation/test_filters.py` — verifica carregamento dos pesos
   - `tests/simulation/test_golden.py` — placeholder que carrega golden e rodará os backends
   - Rodar `pytest tests/simulation/ -v` e garantir que tudo passa (mesmo vazio)

### Sprint 1.2 — Primeiro kernel Numba (MVP do forward)

5. **Solução analítica de half-space (validação independente)**
   - `simulation/validation/half_space.py` — implementar resposta HMD/VMD para 2 camadas homogêneas anisotrópicas
   - Referência: Løseth & Ursin (2007) ou Anderson (1979)
   - Teste unitário: resposta em espaço livre (ρ → ∞) deve convergir para solução estática

6. **`commonarraysMD` em NumPy puro** (ainda sem Numba)
   - Ler `PerfilaAnisoOmp.f08:1110-1198` e mapear para NumPy
   - Entradas: `r, freq, n_layers, h, eta, krJ0J1`
   - Saídas: tupla de 9 arrays complexos `(npt, n_layers)`
   - **Teste:** rodar em modelo de 2 camadas e comparar com half-space analítico

7. **`commonfactorsMD` em NumPy puro**
   - Mesma abordagem, ~14 exponenciais complexas
   - **Teste:** verificar que `exp(-u*z)` decai corretamente

8. **Kernels HMD/VMD em NumPy puro**
   - 6 componentes do tensor por chamada
   - **Teste:** tensor deve ser simétrico quando `ρ_h = ρ_v` (isotropia)

9. **Pipeline NumPy completo (protótipo lento)**
   - `forward_numpy(config) → H_tensor`
   - Rodar nos 100 modelos do golden dataset
   - **Critério:** erro máximo vs Fortran < 1e-8 (tolerância inicial)
   - **Se falhar:** diagnosticar qual componente diverge antes de seguir

### Sprint 1.3 — Acelerar com Numba

10. **Adicionar `@njit` nas funções corretas**
    - Começar pelas folhas (hankel quadrature), subir até o topo
    - Cada função deve ter um teste antes e depois da decoração
    - **Gotcha:** `np.complex128` + `fastmath=True` pode perder precisão — testar ambos

11. **Paralelizar o loop externo com `prange`**
    - Apenas no `forward_numba`, sobre índice achatado de `(itr, k, j)`
    - **Medir:** speedup vs single-thread em máquina do usuário (esperar 4–6× em 8 cores)

12. **Benchmark vs Fortran**
    - `bench_forward.py` — reportar mod/h nas duas implementações
    - Registrar em `docs/reference/` (relatório de baseline)

**Ponto de decisão após Sprint 1.3:** se Numba atinge ≥ 70% da performance Fortran CPU, prosseguir para Fase 2 (JAX). Se ficar abaixo de 50%, investigar gargalos (provavelmente quadratura Hankel ou cache miss) antes de atacar JAX.

---

## 10. Estratégia de Validação Numérica

### 10.1 Triângulo de verdade

```
          tatu_f2py (Fortran v10.0)
                   ▲
                   │
                   │  erro < 1e-10
                   │
        ┌──────────┼──────────┐
        │                     │
     Numba              empymod (Werthmüller)
        │                     │
        └──────────┬──────────┘
                   │
                   │  erro < 1e-8
                   ▼
                  JAX
```

Três fontes independentes cruzando-se:

| Par | Tolerância | Rationale |
|:---|:---:|:---|
| Numba vs Fortran | `1e-10` | Mesmo algoritmo, diferença só em `fastmath` |
| Numba vs empymod | `1e-6` | Algoritmos diferentes, mas mesma física |
| JAX CPU vs Numba | `1e-12` | Ambos NumPy-like, diferenças de ordem de operações |
| JAX GPU vs JAX CPU | `1e-8` | CUDA usa FMA agressivo, pequenas diferenças |
| JAX vs half-space analítico | `1e-4` | Quadratura finita (201pt) limita precisão |

### 10.2 Testes obrigatórios

```python
# tests/simulation/test_forward_parity.py
def test_numba_matches_fortran_golden():
    """100 modelos canônicos: erro máximo < 1e-10"""

def test_jax_cpu_matches_numba():
    """Mesmos 100 modelos: erro máximo < 1e-12"""

def test_jax_gpu_matches_cpu():
    """Apenas se GPU disponível, tolerância 1e-8"""

def test_half_space_analytical():
    """Modelo de 2 camadas isotrópicas: erro < 1e-4"""

def test_empymod_cross_check():
    """10 modelos simples: erro < 1e-6 vs empymod"""

def test_jacobian_fd_vs_jax_jacfwd():
    """∂H/∂ρ via FD (1e-4) vs jacfwd: erro relativo < 1e-6"""

def test_multi_tr_consistency():
    """F1: simulate(nTR=3) == concat(3× simulate(nTR=1))"""

def test_backward_compat_v9():
    """Desabilitando F10: bit-exato vs golden Fortran"""
```

### 10.3 CI

Adicionar workflow que roda os testes acima em cada commit em `geosteering_ai/simulation/`. GPU tests marcados como `@pytest.mark.gpu` e rodados apenas em runner com CUDA (GitHub Actions self-hosted ou Colab CI).

---

## 11. Performance Targets

| Implementação | Ambiente | Throughput mod/h | Latência batch=1 | Speedup vs Fortran |
|:---|:---|:---:|:---:|:---:|
| Fortran (baseline) | i9-9980HK 8T | 58.856 | ~0,06 s | 1,0× |
| Numba single-thread | i9-9980HK 1T | ≥ 8.000 | ~0,45 s | 0,14× |
| Numba 8 threads | i9-9980HK 8T | ≥ **40.000** | ~0,09 s | ≥ 0,7× |
| JAX CPU | i9-9980HK | ≥ 30.000 | ~0,12 s | ≥ 0,5× |
| JAX GPU T4 | Colab | ≥ **200.000** | ~0,018 s | **3,4×** |
| JAX GPU L4 | Colab Pro+ | ≥ 400.000 | ~0,009 s | 6,8× |
| JAX GPU A100 | Colab Pro+/on-prem | ≥ **500.000** | ~0,007 s | **8,5×** |

**Meta secundária — Jacobiano:**

| Método | Tempo/modelo+J | vs FD |
|:---|:---:|:---:|
| Fortran FD (F10 Estratégia C) | ~0,28 s | 1,0× |
| JAX `jacfwd` CPU | ~0,20 s | 1,4× |
| JAX `jacfwd` GPU A100 | **~0,02 s** | **14×** |

---

## 12. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|:---|:---:|:---:|:---|
| `complex128` em GPU mais lento que esperado | Média | Alto | Fornecer `complex64` via config; testar ambos no Fase 3 |
| Recursão TE/TM não vetoriza bem em JAX | Alta | Alto | Usar `jax.lax.scan` (loop compilado) em vez de `vmap` sobre camadas |
| Quadratura Hankel com 801pt (Anderson) estoura VRAM em batch grande | Média | Médio | Particionar batch com `jax.lax.map`; deixar Anderson como opt-in |
| Performance Numba inferior a 70% do Fortran | Baixa | Médio | Profiling detalhado; considerar AOT compilation (`numba.pycc`) |
| Divergência bit-exata com Fortran impossível (fastmath) | Alta | Baixo | Aceitar tolerância `1e-10`; documentar origem das diferenças |
| `empymod` usa algoritmo divergente → falsos positivos de erro | Média | Médio | Isolar empymod como "sanity check", não bloqueante |
| Apple Silicon Metal JAX menos maduro que CUDA | Alta | Baixo | CPU Numba como fallback no macOS |
| Auto-diff em função com `jax.lax.scan` tem overhead | Média | Médio | Benchmark `jacfwd` vs `jacrev` — para `n_layers ≪ n_outputs`, `jacfwd` vence |
| Refatoração quebra cenário "oracle" de PINNs | Baixa | Alto | Manter `fortran_f2py` como backend sempre disponível via config |
| Compilação XLA > 30s em primeira chamada | Alta | Baixo | Warm-up explícito no `TrainingLoop.__init__` + cache de compilação |

---

## 13. Entregáveis por Fase

| Fase | Entregável principal | Métrica de aceitação |
|:---:|:---|:---|
| 0 | Pesos Hankel extraídos + golden dataset + esqueleto | `pytest` verde |
| 1 | `simulate_numba(config)` funcional | ≥ 70% perf Fortran; erro < 1e-10 |
| 2 | `simulate_jax(config)` CPU | Paridade com Numba < 1e-12 |
| 3 | `simulate_jax(config)` GPU | ≥ 200k mod/h (T4) |
| 4 | `jacobian_jax(config)` | ≥ 3× speedup vs F10 FD |
| 5 | Features F1/F5/F6/F7 | Bit-exato vs Fortran |
| 6 | Integração em PINNs + on-the-fly | Treinamento end-to-end verde |
| 7 (opt) | Numba CUDA kernels | ≥ 1M mod/h em A100 |

---

## 14. Decisões Arquiteturais (FIXADAS em 2026-04-11)

As 6 questões pendentes foram respondidas pelo usuário e aplicadas no
código e documentação:

| #  | Questão                                | Decisão aplicada                                              |
|:--:|:---------------------------------------|:--------------------------------------------------------------|
| 1  | Ordem Numba ↔ JAX                      | **Numba primeiro**; JAX CPU JIT avaliado como opção paralela |
| 2  | Precisão default                       | **complex128** + `complex64` configurável (produção GPU)     |
| 3  | Filtro default                         | **werthmuller_201pt** (paridade filter_type=0 Fortran)        |
| 4  | Dependência empymod                    | **Incluída** (3ª fonte de validação cruzada)                  |
| 5  | Backend default PipelineConfig         | Permanece **`fortran_f2py`** até Fase 6                       |
| 6  | Branch de desenvolvimento              | **`feature/simulator-python`** (criada em 2026-04-11)         |

---

## 15. Estado de Execução (atualizado em 2026-04-11)

### 15.1 Sprints concluídas

| Sprint | Nome                                       | Status        | Data       | Testes |
|:------:|:-------------------------------------------|:--------------|:-----------|:------:|
| 1.1    | Extração dos pesos Hankel                 | ✅ Concluída | 2026-04-11 | 53/53 |
| 1.2    | SimulationConfig dataclass                 | ✅ Concluída | 2026-04-11 | 62/62 |
| 1.3    | Soluções analíticas half-space (5 casos)  | ✅ Concluída | 2026-04-11 | 38/38 |
|        | **TOTAL Fase 1 (Foundations)**            | **✅ Completa** | **2026-04-11** | **153/153 PASS em 1.81s** |

**Relatórios detalhados**:
- Sprint 1.1: [`relatorio_sprint_1_1_hankel.md`](relatorio_sprint_1_1_hankel.md)
- Sprints 1.2 + 1.3: [`relatorio_sprint_1_2_1_3_config_halfspace.md`](relatorio_sprint_1_2_1_3_config_halfspace.md)

Resumo dos artefatos entregues na Fase 1:

**Sprint 1.1 (Filtros Hankel)**:
- `scripts/extract_hankel_weights.py` — parser Fortran→.npz com SHA-256 auditável
- `geosteering_ai/simulation/__init__.py` (fachada pública)
- `geosteering_ai/simulation/filters/loader.py` — FilterLoader thread-safe + HankelFilter
- `geosteering_ai/simulation/filters/*.npz` (3 arquivos, 28 KB totais)
- `tests/test_simulation_filters.py` — **53/53 PASS**
- 7 correções pós-review aplicadas

**Sprint 1.2 (SimulationConfig)**:
- `geosteering_ai/simulation/config.py` — `SimulationConfig` dataclass frozen
  (13 campos, 4 presets, YAML roundtrip, errata validation)
- `tests/test_simulation_config.py` — **62/62 PASS**

**Sprint 1.3 (Half-space analítico)**:
- `geosteering_ai/simulation/validation/__init__.py`
- `geosteering_ai/simulation/validation/half_space.py` — 5 funções analíticas:
  `static_decoupling_factors`, `skin_depth`, `wavenumber_quasi_static`,
  `vmd_fullspace_axial`, `vmd_fullspace_broadside`
- `tests/test_simulation_half_space.py` — **38/38 PASS**

### 15.2 Sprints em andamento

Nenhuma — aguardando aprovação para iniciar Fase 2 (backend Numba).

### 15.3 Sprints próximas (Fase 2 — backend Numba CPU)

| Sprint | Nome                                           | Pré-requisitos  |
|:------:|:-----------------------------------------------|:----------------|
| 2.1    | `_numba/propagation.py` (commonarraysMD)       | Fase 1 ✅       |
| 2.2    | `_numba/dipoles.py` (hmd_TIV, vmd)             | Sprint 2.1      |
| 2.3    | `_numba/hankel.py` (quadratura digital)        | Sprint 2.2      |
| 2.4    | `_numba/kernel.py` (orchestrador forward)      | Sprint 2.3      |
| 2.5    | `forward.py` (API `simulate()` + backend dispatch) | Sprint 2.4  |
| 2.6    | Validação Numba vs soluções analíticas (half-space) | Sprint 2.5 |
| 2.7    | Benchmark CPU Numba (meta ≥ 40k mod/h)        | Sprint 2.6      |

---

## Referências

- Anderson, W.L. (1979). *Numerical integration of related Hankel transforms of orders 0 and 1 by adaptive digital filtering.* Geophysics, 44(7), 1287–1305.
- Løseth, L.O. & Ursin, B. (2007). *Electromagnetic fields in planarly layered anisotropic media.* Geophysical Journal International, 170(1), 44–80.
- Werthmüller, D. (2017). *An open-source full 3D electromagnetic modeler for 1D VTI media in Python: empymod.* Geophysics, 82(6), WB9–WB19.
- Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). *Physics-informed neural networks.* Journal of Computational Physics, 378, 686–707.
- Bradbury, J. et al. (2018). *JAX: composable transformations of Python+NumPy programs.* http://github.com/jax-ml/jax
- Lam, S.K. et al. (2015). *Numba: a LLVM-based Python JIT compiler.* SC '15 Companion, 1–6.
