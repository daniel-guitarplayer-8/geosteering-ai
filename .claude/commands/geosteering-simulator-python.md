---
name: geosteering-simulator-python
description: |
  Simulador Python otimizado (JAX + Numba) do Geosteering AI v2.0 — equivalente
  matemático do simulador Fortran PerfilaAnisoOmp/tatu.x. Dois backends
  intercambiáveis: Numba (CPU, njit+prange) e JAX (CPU/GPU/TPU, jit+vmap+pmap).
  Cobre: pacote geosteering_ai/simulation/, SimulationConfig, API simulate(),
  filtros Hankel digitais (Werthmüller 201pt default, Kong 61pt, Anderson 801pt),
  validação contra Fortran + empymod, benchmarks, roadmap de 7 fases.
  Triggers: "simulador Python", "simulação Python", "JAX", "Numba", "njit",
  "prange", "vmap", "pmap", "jit", "FilterLoader", "HankelFilter",
  "extract_hankel_weights", "filtersv2", "pesos Hankel", "abscissas",
  "SimulationConfig", "simulate(", "backend numba", "backend jax",
  "simulation/_jax", "simulation/_numba", "commonarraysMD Python",
  "commonfactorsMD Python", "hmd_TIV Python", "empymod", "half-space",
  "paridade Fortran", "Sprint 1.1", "Sprint 1.2", "Fase 0 simulador",
  "feature/simulator-python".
---

# Geosteering AI — Simulador Python Otimizado (JAX + Numba)

Pacote: `geosteering_ai/simulation/` — Simulador EM 1D TIV Python com dois
backends de alta performance (Numba CPU e JAX CPU/GPU), matematicamente
equivalente ao simulador Fortran `PerfilaAnisoOmp.f08` (v10.0, `tatu.x`).

Branch de desenvolvimento: `feature/simulator-python`.

---

## 1. Estado Atual e Roadmap

### 1.1 Versão e progresso

| Campo            | Valor                                                     |
|:-----------------|:----------------------------------------------------------|
| **Versão**       | **1.0.0** (Sprints 1.1-1.3 + 2.1-2.9 + **3.1-3.4** concluídas) |
| **Branch**       | `feature/simulator-python-sprint-3-complete`              |
| **Base**         | `main` (a partir do commit Sprint 2.7 `91e4634`)          |
| **Autor**        | Daniel Leal                                               |
| **Framework**    | NumPy 2.x + Numba 0.61+ + **JAX 0.4.38+** + empymod (valid.)  |
| **Precisão**     | `complex128` default + `complex64` via config (prod.)     |
| **Filtro default** | Werthmüller 201pt (mantém paridade Fortran filter_type=0) |
| **Testes**       | **1214 passed, 295 skipped** em 42.1s (+79 Sprint 2.9+3.2+3.3) |
| **Referência**   | `docs/reference/plano_simulador_python_jax_numba.md`      |

### 1.2 Fases do plano (7 fases)

| Fase | Nome                                    | Status      | Sprint(s) |
|:----:|:----------------------------------------|:------------|:----------|
|  0   | Setup (branch, deps, estrutura)         | ✅ Concluída | 1.1 ✅ |
|  1   | Foundations (filtros, config, analítico) | ✅ **Concluída** | 1.1 ✅, 1.2 ✅, 1.3 ✅ |
|  2   | Backend Numba CPU (paridade Fortran)    | ✅ **Concluída** | 2.1–2.9 ✅ (**2.9: @njit + prange, 6.6× speedup**) |
|  3   | Backend JAX (CPU+GPU, vmap+jit)         | ✅ **Concluída** | **3.1 ✅**, **3.2 ✅**, **3.3 ✅** (híbrido), **3.4 ✅** (Colab) |
|  4   | Validação cruzada (Fortran/Numba/empymod) | ⬜ Pendente | 4.1-4.3 |
|  5   | Jacobiano ∂H/∂ρ (jacfwd JAX, FD Numba)  | ⬜ Pendente | 5.1-5.2   |
|  6   | Integração no PipelineConfig (backend)  | ⬜ Pendente | 6.1-6.2   |
|  7   | Otimizações finais (pmap, XLA, caching) | ⬜ Pendente | 7.1-7.3   |

### 1.2d Sprint 2.9 — fields_in_freqs @njit (concluída 2026-04-12)

**Objetivo cumprido**: Port de `fields_in_freqs` para `@njit`, criando
`_fields_in_freqs_kernel` e `_compute_zrho_kernel`. Mesma estratégia para
`sanitize_profile → _sanitize_profile_kernel`. Habilitou
`_simulate_positions_njit` com `@njit(parallel=True)` + `prange` —
**speedup real de 6.6× em medium profile** (GIL eliminado do caminho crítico).

**Benchmarks Sprint 2.9 (parallel=True default):**
| Perfil | Sprint 2.7 | Sprint 2.9 | % Fortran |
|:---|---:|---:|---:|
| small  | 66k mod/h | **663k mod/h** | **1127.5%** ✅ |
| medium | 15k mod/h | **170k mod/h** | **289.6%** ✅ |
| large  | 3.6k mod/h | **26k mod/h** | 44.8% |

### 1.2e Modelos canônicos (Sprint 2.9)

Novo submódulo `validation/canonical_models.py` com **7 modelos geológicos
canônicos** para validação reprodutível:

| Id | Nome | Camadas | Tipo | Referência |
|:---|:---|:---:|:---|:---|
| oklahoma_3 | Oklahoma 3 | 3 | TIV | TR 32_2011 |
| oklahoma_5 | Oklahoma 5 | 5 | TIV gradual | TR 32_2011 |
| devine_8 | Devine 8 | 8 | Isotrópico | TR 32_2011 |
| oklahoma_15 | Oklahoma 15 | 15 | Isotrópico | TR 32_2011 |
| oklahoma_28 | Oklahoma 28 | 28 | TIV forte (ρv=2ρh) | TR 32_2011 |
| hou_7 | Hou et al. 7 | 7 | TIV | Hou 2006 |
| viking_graben_10 | Viking Graben 10 | 10 | TIV (N. Sea) | Eidesmo 2002 |

Wrappers de plotagem em `visualization/plot_canonical.py`:
- `plot_canonical_model(name, freq, TR, dip, ...)` → Figure
- `plot_all_canonical_models(output_dir)` → List[Path]

**69 testes** cobrindo shapes, ρ positivo, interfaces monotônicas,
simulate() funcional e plot wrappers.

### 1.2f Sprint 3.2 — _jax/propagation.py (concluída 2026-04-12)

**Port JAX** de `common_arrays` + `common_factors` usando:
- `jax.vmap` sobre eixo de camadas para constantes por camada
- `jax.lax.scan` para recursões TE/TM bottom-up e top-down
- Operações primitivas diferenciáveis (habilita `jax.grad`)

**Paridade JAX vs Numba**:
- 9 arrays de `common_arrays_jax`: **< 1e-13** (ULP float64)
- 6 fatores de `common_factors_jax`: **< 1e-16** (bit-exato)

**10 testes PASS** em 4 cenários (single_layer, 3-TIV, 5-iso, alta resistividade
1e6 Ω·m) × {shape, paridade}.

### 1.2g Sprint 3.3 — _jax/kernel.py híbrido (concluída 2026-04-12)

**Arquitetura pragmática**: `fields_in_freqs_jax_batch` reusa os 900 LOC
complexos de `hmd_tiv`+`vmd` (Numba) via `jax.pure_callback`, enquanto a
propagação roda em JAX puro (diferenciável).

**Vantagens do híbrido**:
1. Propagação JAX com `jax.lax.scan` — compilável por XLA, diferenciável
2. Dipolos reusados de Numba — sem re-implementação de 6 casos geométricos
3. Paridade numérica automática (< 1e-13 vs Numba puro)
4. Preparado para GPU — quando `pure_callback` for substituído por port
   JAX nativo dos dipolos (Sprint 3.3.1 futuro), todo o pipeline roda
   em GPU

### 1.2h Sprint 3.4 — GPU + Colab (concluída 2026-04-12)

**Notebook Colab**: `notebooks/bench_jax_gpu_colab.ipynb`
- Detecção automática CPU/GPU (via `nvidia-smi`)
- Instalação condicional: `jax[cuda12]` se GPU, `jax[cpu]` caso contrário
- Benchmark Numba CPU + JAX (CPU/GPU) + validação com 7 modelos canônicos
- Compatível com Colab Pro+ T4/L4/A100

**Compatibilidade local**:
- macOS: `pip install jax[cpu]` (JAX Metal é experimental, não
  recomendado por enquanto)
- Linux + CUDA: `pip install jax[cuda12]`
- Windows + CUDA: `pip install jax[cuda12]` (via WSL2 preferencialmente)

### 1.2i Bateria de testes consolidada (2026-04-12)

**Total: 1214 passed, 295 skipped** em 42.1s (CPU)

- 53 test_simulation_filters
- 87 test_simulation_config
- 38 test_simulation_half_space
- 25 test_simulation_numba_propagation (+1 correção bit-exato→rtol)
- 22 test_simulation_numba_dipoles
- 16 test_simulation_io
- 20 test_simulation_postprocess
- 16 test_simulation_numba_geometry
- 12 test_simulation_numba_rotation
- 11 test_simulation_numba_hankel
- 20 test_simulation_numba_kernel
- 18 test_simulation_forward
- 15 test_simulation_analytical_validation
- 11 test_simulation_benchmark
- 11 test_simulation_visualization (Sprint 2.8)
- 15 test_simulation_jax_foundation (Sprint 3.1)
- **69 test_simulation_canonical_models (Sprint 2.9)**
- **10 test_simulation_jax_propagation (Sprint 3.2)**

**PRs mergeados em `main`**: #1 (Fase 1), #2–#6 (Sprints 2.1-2.7). Sprints 2.8
+ 3.1 em branch `feature/simulator-python-sprint-2-8-and-3-1` (este commit).

### 1.2b Sprint 2.8 — Paralelização via ThreadPool + Visualização (concluída 2026-04-12)

**Finding principal**: `@njit(parallel=True, prange)` NÃO funciona porque
`fields_in_freqs` (kernel.py) é uma função Python pura que orquestra chamadas
a @njit kernels — não pode ser chamada de dentro de um contexto @njit sem
refatoração profunda. Usamos `ThreadPoolExecutor` como alternativa, mas o
speedup foi ≈ 1.0× porque o GIL é retido entre as chamadas @njit. A
infraestrutura (`forward.py:_simulate_positions_parallel` + `cfg.parallel`
flag) está preservada para quando `fields_in_freqs` for portado para @njit
(Sprint 2.9?) ou substituído por `vmap` JAX (Sprint 3.3+, onde XLA não sofre
de GIL).

**Default do config**: `parallel=False` (não ajuda atualmente, evita confusão).

**Novos módulos**:
- `geosteering_ai/simulation/visualization/__init__.py` — fachada pública
- `geosteering_ai/simulation/visualization/plot_tensor.py` — `plot_tensor_profile()`
  + `plot_resistivity_profile()` com layout GridSpec(3,7) (19 axes) —
  replica padrão de `buildValidamodels.py:571-628`
- `geosteering_ai/simulation/visualization/plot_benchmark.py` — 
  `plot_benchmark_comparison()` (2 painéis: throughput + % Fortran)

**Testes**: `tests/test_simulation_visualization.py` — **11 testes PASS** em
3.0s (4 TestResistivityProfile, 4 TestTensorProfile, 3 TestBenchmarkComparison).

**Convenções visuais**:
- Eixo y invertido (profundidade cresce para baixo)
- ρ em semilogx (cobre 1e-1 a 1e6 Ω·m)
- Paletas Re: azul, Im: vermelho (fidelidade `buildValidamodels.py:540-545`)
- Interfaces: `axhline` tracejadas pretas

### 1.2c Sprint 3.1 — JAX Foundation CPU (concluída 2026-04-12)

**Objetivo**: Fundação do backend JAX — portar módulos não-recursivos
(hankel, rotation) com paridade numérica < 1e-12 vs Numba.

**Instalação**: JAX 0.4.38 via `pip install jax[cpu]` — 99.7 MB jaxlib +
2.2 MB jax. Backend CPU default. `jax.config.update("jax_enable_x64", True)`
chamado no `_jax/__init__.py` para garantir complex128.

**Novos módulos**:
- `geosteering_ai/simulation/_jax/__init__.py` — fachada + `HAS_JAX` flag
- `geosteering_ai/simulation/_jax/hankel.py` — `integrate_j0`, `integrate_j1`,
  `integrate_j0_j1` via `jnp.einsum("i,i->", w, v)` + `@jax.jit`
- `geosteering_ai/simulation/_jax/rotation.py` — `build_rotation_matrix`
  (matriz R 3×3 via `jnp.stack`), `rotate_tensor` (`Rᵀ @ H @ R`) — **diferenciáveis
  via `jax.grad`/`jax.jacfwd`**

**Paridade medida**:
- `build_rotation_matrix`: **0.00e+00** (bit-exato) vs Numba
- `rotate_tensor`: **9.16e-16** (ULP float64) vs Numba
- `integrate_j0/j1` vs numpy: < 1e-13

**Testes**: `tests/test_simulation_jax_foundation.py` — **15 testes PASS**
em 5.0s:
- 5 Hankel (integrais, paridade numpy, consistência j0_j1, JIT cache)
- 5 BuildRotationMatrix (identidade, ortogonalidade R·Rᵀ=I, det=+1,
  paridade Numba, diferenciabilidade via `jax.grad`)
- 5 RotateTensor (identidade preserva, tr invariante, ‖·‖_F invariante,
  paridade Numba, composição R(α)·R(-α) = I)

**Deferido para Sprints 3.2–3.4**:
- Sprint 3.2: port de `_jax/propagation.py` (common_arrays + common_factors
  com `jax.lax.scan` para recursões TE/TM)
- Sprint 3.3: port de `_jax/dipoles.py` + `_jax/kernel.py` (orquestrador
  `fields_in_freqs_jax` com `vmap` sobre posições)
- Sprint 3.4: GPU support (pip install `jax[cuda12]` / `jax[metal]`) +
  benchmark T4/A100

### 1.3 Sprint 1.1 — Extração dos pesos Hankel (concluída 2026-04-11)

- `scripts/extract_hankel_weights.py` — parser do Fortran `filtersv2.f08`
  com regex robusta (aceita `1.23D+02`, `0.21D-28`, `.21D-28`, `1D0`),
  hash SHA-256 auditável e validação bit-a-bit.
- `geosteering_ai/simulation/filters/*.npz` — 3 artefatos (28 KB totais).
- `geosteering_ai/simulation/filters/loader.py` — `FilterLoader` com
  **cache classe-level thread-safe** (double-checked locking via
  `threading.Lock`, seguro para Fase 2 com workers paralelos) +
  `HankelFilter` (`@dataclass(frozen=True)` + arrays read-only).
- `tests/test_simulation_filters.py` — **53 testes**:
  - 11 bit-exact vs Fortran (Kong, Werthmüller, Anderson: primeiro/meio/último)
  - 13 API (canônico, aliases, cache)
  - 4 imutabilidade (arrays read-only, frozen)
  - 4 sincronia SHA-256 (gate Fortran ↔ .npz)
  - 21 restantes (shapes, semântica, Anderson expandido)
- 7 correções pós-review aplicadas (1 race condition, 2 regex, 1
  assertion, 1 spot-check, 1 fixture, 1 header D1).

### 1.4 Sprint 1.2 — SimulationConfig (concluída 2026-04-11)

- `geosteering_ai/simulation/config.py` — `SimulationConfig`
  (`@dataclass(frozen=True)`, 13 campos) com:
  - Validação de errata em `__post_init__` (ranges físicos, enums,
    conflitos mútuos backend × device).
  - 4 presets via `@classmethod`: `default()`, `high_precision()`,
    `production_gpu()`, `realtime_cpu()`.
  - Roundtrip YAML via `to_yaml/from_yaml` (lazy import PyYAML).
  - Roundtrip dict via `to_dict/from_dict` (ignora chaves extras
    com warning).
- `tests/test_simulation_config.py` — **62 testes**:
  - 11 defaults
  - 8 ranges numéricos
  - 12 enums (backend, dtype, device, hankel_filter)
  - 4 mutual exclusivity
  - 6 listas opcionais
  - 4 num_threads
  - 5 presets
  - 3 imutabilidade
  - 6 serialização (dict + YAML)
  - 3 igualdade + hash

### 1.5 Sprint 1.3 — Soluções analíticas half-space (concluída 2026-04-11)

- `geosteering_ai/simulation/validation/__init__.py` — fachada.
- `geosteering_ai/simulation/validation/half_space.py` — 5 funções
  puras NumPy com ground-truth analítico:
  1. `static_decoupling_factors(L)` → (ACp, ACx) — CLAUDE.md errata.
  2. `skin_depth(f, rho)` → δ em metros (Nabighian 1988).
  3. `wavenumber_quasi_static(f, rho)` → k complexo (Ward-Hohmann).
  4. `vmd_fullspace_axial(L, f, rho, m)` → Hz em `(0,0,L)`.
  5. `vmd_fullspace_broadside(L, f, rho, m)` → Hz em `(L,0,0)`.
- Convenção temporal e^(-iωt) (padrão geofísica / Moran-Gianzero 1979).
- `tests/test_simulation_half_space.py` — **38 testes**:
  - 7 decoupling factors (bit-exato vs CLAUDE.md)
  - 8 skin depth (fórmula, dependências 1/√f e √ρ, array broadcast)
  - 7 wavenumber (Im(k)>0, |k|·δ=√2)
  - 7 VMD axial (limite estático, linearidade momento, skin effect)
  - 5 VMD broadside (limite estático ACp, sinal negativo)
  - 4 cross-cutting (relações entre casos analíticos)

### 1.6 Bateria de testes consolidada

| Suíte                                      | Testes | Tempo  |
|:-------------------------------------------|:------:|:------:|
| tests/test_simulation_filters.py           |   53   | 0.82s  |
| tests/test_simulation_config.py            |   87   | ~0.8s  |
| tests/test_simulation_half_space.py        |   38   | ~0.2s  |
| tests/test_simulation_numba_propagation.py |   25   | ~0.6s  |
| tests/test_simulation_numba_dipoles.py     |   22   | ~0.5s  |
| tests/test_simulation_io.py                |   16   | ~0.2s  |
| tests/test_simulation_postprocess.py       |   20   | ~0.1s  |
| tests/test_simulation_numba_geometry.py    |   16   | ~0.3s  |
| tests/test_simulation_numba_rotation.py    |   12   | ~0.2s  |
| tests/test_simulation_numba_hankel.py      |   11   | ~0.1s  |
| tests/test_simulation_numba_kernel.py      |   20   | ~0.3s  |
| tests/test_simulation_forward.py           |   18   | ~0.4s  |
| tests/test_simulation_analytical_validation.py | 15 | ~0.3s  |
| tests/test_simulation_benchmark.py         |   11   | ~4.5s  |
| **TOTAL**                                  | **364** | **~6s** |

### 1.7 Sprint 2.1 — Backend Numba propagation (concluída 2026-04-11)

- `geosteering_ai/simulation/_numba/__init__.py` — fachada do backend
  Numba, re-export de `common_arrays`, `common_factors`, `HAS_NUMBA`.
- `geosteering_ai/simulation/_numba/propagation.py` — port Python+Numba
  de `commonarraysMD` e `commonfactorsMD` (Fortran `utils.f08:158-297`)
  com pipeline completo de propagação TE/TM e reflexão multi-camada.
  - **Dual-mode Numba**: funciona com ou sem Numba instalado (no-op
    decorator fallback), permitindo CI mínimo e debugging nativo.
  - **Decoradores**: `@njit(cache=True, fastmath=False, error_model="numpy")`
    — paridade bit-exata com Fortran `real(dp)` (sem FMA reorder).
  - **Guard de singularidade**: `hordist < 1e-12 → r = 0.01 m`
    (replica Fortran `utils.f08:195`).
  - **9 arrays invariantes** (common_arrays): `u`, `s`, `uh`, `sh`,
    `RTEdw`, `RTEup`, `RTMdw`, `RTMup`, `AdmInt` (shape `(npt, n)`
    complex128).
  - **6 fatores de onda** (common_factors): `Mxdw`, `Mxup`, `Eudw`,
    `Euup`, `FEdwz`, `FEupz` (shape `(npt,)` complex128).
- `tests/test_simulation_numba_propagation.py` — **25 testes**
  (todos PASS em 0.59s):
  - 4 shapes/dtypes/não-mutação
  - 5 limite 1-camada homogêneo isotrópico (s=u, RT=0, u²=kr²+zeta·σh)
  - 3 limite TIV (λ=4, s²=λ·(kr²+zeta·σv), u independente de σv)
  - 4 invariantes da recursão (RTEdw[n-1]=0, RTEup[0]=0, |RT|≤1)
  - 3 common_factors shapes/dtypes/count
  - 4 common_factors self-consistency (Mxdw=exp(-s·dz), FEdwz=exp(-u·dz))
  - 2 dual-mode Numba (HAS_NUMBA bool, chamada em qualquer modo)

**Gate 2.1 → 2.2**: ✅ Atingido — paridade estrutural e física 100%.

### 1.8 Sprint 2.2 — Dipolos + I/O + F6/F7 (concluída 2026-04-11)

**Objetivo**: Portar os kernels de dipolos magnéticos (`hmd_TIV`, `vmd`) para
Numba e adicionar 3 features opt-in: exportadores Fortran-compatíveis, F6
Compensação Midpoint e F7 Antenas Inclinadas.

**Módulos criados**:

- `geosteering_ai/simulation/_numba/dipoles.py` (~1300 LOC) — port
  line-for-line de `magneticdipoles.f08:91-624`:
  - `hmd_tiv(...)` — HMD com 2 polarizações (hmdx + hmdy) retorna 3
    arrays `(2,)` complex128 para `[Hx, Hy, Hz]_{dipolo}`.
  - `vmd(...)` — VMD retorna 3 escalares complex128 `(Hx, Hy, Hz)`.
  - 6 casos geométricos tratados: `camadR==0 and camadT!=0`,
    `camadR<camadT`, `camadR==camadT and z≤h0`, `camadR==camadT and z>h0`,
    `camadR>camadT and camadR!=n-1`, `camadR==n-1`.
  - Integração Hankel via `wJ0`/`wJ1` do `FilterLoader`.

- `geosteering_ai/simulation/io/model_in.py` (~360 LOC) —
  `export_model_in()` escreve arquivo ASCII idêntico ao formato
  `model.in` v10.0 (paridade `fifthBuildTIVModels.py`). Suporta F5, F6,
  F7 quando ativos no config. Opt-in via `cfg.export_model_in=True`.

- `geosteering_ai/simulation/io/binary_dat.py` (~400 LOC) —
  `export_binary_dat()` escreve `.dat` 22-col stream nativo + 
  `export_out_metadata()` escreve `info{filename}.out` texto. Round-trip
  byte-exato via `np.fromfile(path, dtype=DTYPE_22COL)`. Opt-in via
  `cfg.export_binary_dat=True`.

- `geosteering_ai/simulation/postprocess/compensation.py` (~240 LOC) —
  `apply_compensation(H_tensors, comp_pairs)` implementa F6 via fórmula
  CDR clássica: `H_comp = 0.5·(H_near+H_far)`, `Δφ[°]` e `Δα[dB]`.
  Retorna tupla `(H_comp, phase_diff_deg, atten_db)`.

- `geosteering_ai/simulation/postprocess/tilted.py` (~180 LOC) —
  `apply_tilted_antennas(H_tensor, tilted_configs)` implementa F7 via
  projeção `H_tilted = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]`.

**Ampliação de `SimulationConfig` (3 novos grupos)**:

- **Grupo 7 (I/O)**: `export_model_in`, `export_binary_dat`, `output_dir`,
  `output_filename` — todos default `False`/`"."`/`"simulation"`.
- **Grupo 8 (F6)**: `use_compensation: bool`, `comp_pairs: Tuple[Tuple[int, int], ...]`.
  Pré-requisito: `len(tr_spacings_m) >= 2`.
- **Grupo 9 (F7)**: `use_tilted_antennas: bool`,
  `tilted_configs: Tuple[Tuple[float, float], ...]`. Ranges: β ∈ [0°, 90°],
  φ ∈ [0°, 360°).

**Testes**:

- `tests/test_simulation_numba_dipoles.py` — **22 testes PASS + 1 skip**:
  - 4 shapes/dtypes/no-nan-inf
  - 5 decoupling limit (ACp, ACx bit-exato em σ→0)
  - 4 VMD analítico vs `vmd_fullspace_broadside` (< 1e-4, atol em ρ=100 Ω·m)
  - 6 alta resistividade ρ ∈ {1, 10², 10³, 10⁴, 10⁵, 10⁶} Ω·m
  - 2 reciprocidade T↔R
  - 2 compilação JIT (1 skip se Numba não instalado)

- `tests/test_simulation_io.py` — **16 testes PASS**:
  - 5 TestModelInBasic (opt-in, layout, filter_type)
  - 4 TestModelInFlags (F5, F6, F7, multi-TR)
  - 5 TestBinaryDatRoundTrip (opt-in, 172 bytes, round-trip, append, shape 2D)
  - 2 TestOutMetadata (conteúdo básico + F7)

- `tests/test_simulation_postprocess.py` — **20 testes PASS**:
  - 6 TestCompensationBasic (shapes, opt-in, identidade)
  - 4 TestCompensationPhysics (fórmula CDR, dB, φ, NaN guard)
  - 5 TestTiltedBasic (shapes, β=0 → Hzz, β=90° φ=0° → Hxz, β=90° φ=90° → Hyz)
  - 5 TestTiltedOrtogonality (combinações canônicas)

- `tests/test_simulation_config.py` +25 testes novos:
  - 5 TestGroup7Exporters, 7 TestGroup8Compensation, 6 TestGroup9TiltedAntennas,
    2 TestSprint22Integration, 5 ajustes em testes pré-existentes.

**Gate 2.2 → 2.3**: ✅ Atingido — 261/261 testes (+1 skip) PASS em 1.58s.
Decoupling ACp e ACx bit-exato vs CLAUDE.md errata. Alta resistividade
estável até 10⁶ Ω·m sem NaN/Inf.

### 1.9 Sprint 2.3 + 2.4 — Geometry + Rotation + Hankel + Kernel (concluída 2026-04-12)

**Objetivo**: Portar os módulos auxiliares (geometria de camadas, rotação Euler
RtHR, quadratura Hankel) e o orquestrador forward que amarra toda a cadeia
de cálculo do tensor H em uma função `fields_in_freqs`.

**Módulos criados** (Sprint 2.3):

- `_numba/geometry.py` — `sanitize_profile()` (constrói h/prof a partir de
  espessuras), `find_layers_tr()` (localiza camadas TX/RX), `layer_at_depth()`
  (camada para profundidade arbitrária). Port de `utils.f08:5-87,299-319`.
- `_numba/rotation.py` — `build_rotation_matrix(α, β, γ)` (matriz R 3×3),
  `rotate_tensor(α, β, γ, H)` (aplica Rᵀ·H·R). Port de `utils.f08:321-355`
  (Liu 2017 eq. 4.80). Convenção: ângulos em radianos.
- `_numba/hankel.py` — `prepare_kr()`, `integrate_j0()`, `integrate_j1()`,
  `integrate_j0_j1()`. Helpers para a quadratura digital de Hankel, usados
  como API de conveniência e documentação (os dipolos da Sprint 2.2 já fazem
  a integração inline para performance máxima).

**Módulo criado** (Sprint 2.4):

- `_numba/kernel.py` — `fields_in_freqs()` (orquestrador forward completo)
  + `compute_zrho()` (profundidade do ponto-médio + resistividades).
  Port de `fieldsinfreqs` (`PerfilaAnisoOmp.f08:937-993`). Recebe posição
  TR, dip, perfil geológico e frequências; retorna tensor H rotacionado
  `(nf, 9)` complex128.

**Testes** (59 novos):

- `test_simulation_numba_geometry.py` — **16 testes**: sanitize shapes,
  find_layers TX/RX em 7 posições canônicas, layer_at_depth.
- `test_simulation_numba_rotation.py` — **12 testes**: R identidade,
  ortogonalidade R·Rᵀ=I, det=+1, γ=π/2 troca Hxx↔Hyy, trace invariante,
  norma Frobenius invariante.
- `test_simulation_numba_hankel.py` — **11 testes**: prepare_kr, integrate
  J0/J1/J0J1 com constantes, complex, guard, filtro real.
- `test_simulation_numba_kernel.py` — **20 testes**: shape (nf,9), multi-freq,
  no-NaN/Inf, decoupling ACx/ACp/Hzz, VMD analítico < 1e-4, 3-camadas
  isotrópicas = full-space, TIV ≠ isotrópico, alta resistividade ρ 1..10⁶,
  dip_rad=0 vs π/2, compute_zrho.

**Gate 2.4 → 2.5**: ✅ Atingido — 320/320 PASS (+1 skip) em 0.99s. O
orquestrador `fields_in_freqs` reproduz ACp/ACx/Hzz analítico e funciona
com perfis multi-camada, TIV, alta resistividade e rotação dip.

---

## 2. Decisões de Arquitetura (fixadas pelo usuário)

| Questão                          | Decisão                                         |
|:---------------------------------|:------------------------------------------------|
| 1. Ordem de implementação        | **Numba primeiro**, depois JAX (com CPU JIT avaliado) |
| 2. Precisão default              | **complex128** (paridade Fortran) + `complex64` via config |
| 3. Filtro default                | **werthmuller_201pt** (mantém filter_type=0 Fortran)  |
| 4. Dependências                  | **inclui empymod** (validação cruzada)          |
| 5. Backend default PipelineConfig | Mantém **`fortran_f2py`** até Fase 6            |
| 6. Branch                        | **`feature/simulator-python`**                  |

---

## 3. Estrutura do Pacote

```
geosteering_ai/simulation/
├── __init__.py                ← ★ fachada (FilterLoader, HankelFilter, SimulationConfig)
├── config.py                  ← ★ SimulationConfig (Sprint 1.2)
├── forward.py                 ← [PENDENTE Fase 2-3] API simulate()
│
├── _numba/                    ← 🟡 EM CONSTRUÇÃO (Fase 2)
│   ├── __init__.py            ← ★ (Sprint 2.1, dual-mode Numba)
│   ├── propagation.py         ← ★ (Sprint 2.1) common_arrays + common_factors
│   ├── dipoles.py             ← ★ (Sprint 2.2) hmd_tiv + vmd (port Fortran)
│   ├── geometry.py            ← ★ (Sprint 2.3) sanitize_profile + find_layers_tr + layer_at_depth
│   ├── rotation.py            ← ★ (Sprint 2.3) build_rotation_matrix + rotate_tensor (RtHR)
│   ├── hankel.py              ← ★ (Sprint 2.3) prepare_kr + integrate_j0/j1 (helpers)
│   ├── kernel.py              ← ★ (Sprint 2.4) fields_in_freqs + compute_zrho (orquestrador)
│   └── jacobian.py            ← [PENDENTE Fase 5] ∂H/∂ρ via FD
│
├── io/                        ← ★ IMPLEMENTADO (Sprint 2.2) — opt-in
│   ├── __init__.py            ← fachada + re-exports
│   ├── model_in.py            ← export_model_in() Fortran-compatível
│   └── binary_dat.py          ← DTYPE_22COL + export_binary_dat + export_out_metadata
│
├── postprocess/               ← ★ IMPLEMENTADO (Sprint 2.2) — opt-in
│   ├── __init__.py            ← fachada + re-exports
│   ├── compensation.py        ← F6 apply_compensation (CDR midpoint)
│   └── tilted.py              ← F7 apply_tilted_antennas (β, φ projeção)
│
├── _jax/                      ← [PENDENTE Fase 3] backend CPU/GPU/TPU
│   ├── __init__.py
│   ├── propagation.py         ← jax.lax.scan para recursão de camadas
│   ├── dipoles.py             ← jit + vmap
│   ├── hankel.py              ← quadratura vetorizada
│   ├── rotation.py            ← RtHR diferenciável
│   ├── jacobian.py            ← jacfwd (automatic differentiation)
│   └── kernel.py              ← pmap multi-GPU
│
├── filters/                   ← ★ IMPLEMENTADO (Sprint 1.1)
│   ├── __init__.py            ← D1 completo
│   ├── loader.py              ← FilterLoader (thread-safe), HankelFilter
│   ├── README.md
│   ├── werthmuller_201pt.npz  ← filter_type=0 (default)
│   ├── kong_61pt.npz          ← filter_type=1 (rápido)
│   └── anderson_801pt.npz     ← filter_type=2 (preciso)
│
├── validation/                ← ★ IMPLEMENTADO parcial (Sprint 1.3)
│   ├── __init__.py            ← 6 exports públicos
│   ├── half_space.py          ← 5 casos analíticos fechados (Sprint 1.3)
│   ├── compare_fortran.py     ← [PENDENTE Fase 4] bit-exactness vs tatu.x
│   └── compare_empymod.py     ← [PENDENTE Fase 4] cross-check empymod
│
├── geometry.py                ← [PENDENTE Fase 2] dip, azimute, posições
├── postprocess.py             ← [PENDENTE Fase 2] FV, GS, compensação
│
└── benchmarks/                ← [PENDENTE Fase 4-7]
    ├── bench_forward.py       ← latência por modelo
    ├── bench_jacobian.py      ← latência ∂H/∂ρ
    └── bench_scaling.py       ← strong/weak scaling CPU e GPU
```

---

## 4. Filtros Hankel Digitais (Sprint 1.1)

### 4.1 Catálogo

| Arquivo                  | npt | filter_type | Uso                                     |
|:-------------------------|:---:|:-----------:|:----------------------------------------|
| `werthmuller_201pt.npz`  | 201 |    0 (★)    | Default, paridade Fortran               |
| `kong_61pt.npz`          |  61 |      1      | 3.3× mais rápido, precisão ok para DL   |
| `anderson_801pt.npz`     | 801 |      2      | Máxima precisão, referência              |

### 4.2 API do `FilterLoader`

```python
from geosteering_ai.simulation.filters import FilterLoader, HankelFilter

loader = FilterLoader()

# Carregamento (I/O só na primeira chamada, depois cache)
filt = loader.load("werthmuller_201pt")      # nome canônico
filt = loader.load("wer")                    # alias curto
filt = loader.load("0")                      # filter_type numérico

# HankelFilter é @dataclass(frozen=True) com arrays read-only
filt.abscissas     # np.ndarray(npt,) float64  — kr pontos (> 0, crescente)
filt.weights_j0    # np.ndarray(npt,) float64  — pesos J₀
filt.weights_j1    # np.ndarray(npt,) float64  — pesos J₁
filt.npt           # 201
filt.fortran_filter_type  # 0
filt.source_sha256 # hash do filtersv2.f08 no momento da extração
filt.description   # "Werthmüller 201pt — default do simulador Fortran"

# Listagem
loader.available()  # ['werthmuller_201pt', 'kong_61pt', 'anderson_801pt']

# Cache
loader.clear_cache()  # força re-leitura do disco
```

### 4.3 Fórmula matemática (quadratura digital)

```
F(r) = ∫₀^∞ f(kr) · Jν(kr · r) · dkr   (ν = 0 ou 1)

     ≈ (1/r) · Σᵢ f(aᵢ/r) · wᵢ(ν)       (quadratura digital)
```

onde `a = abscissas` e `w = weights_j{0,1}` são pré-computados e
carregados do `.npz`. Todos os 3 filtros são da família
"J₀-J₁ common-base" (mesmas abscissas para ambas as ordens).

### 4.3b Thread-safety do FilterLoader

O cache `_class_cache` é protegido por `threading.Lock` com **double-checked
locking**:

1. **Cache hit (caminho rápido)** — lê `_class_cache.get(key)` sem lock.
   A leitura do dict é atômica sob GIL do CPython, então não há corrida.
2. **Cache miss (caminho lento)** — adquire `_class_lock`, re-verifica,
   faz o I/O e inserção. Apenas uma thread por vez faz o trabalho pesado.

Esta proteção é crítica para a Fase 2 (Numba com `prange` ou workers
`concurrent.futures.ThreadPoolExecutor`), onde múltiplas threads vão
chamar `FilterLoader.load("werthmuller_201pt")` simultaneamente no
primeiro acesso. Sem o lock, duas threads construiriam dois
`HankelFilter` distintos, e o segundo escrito sobreporia o primeiro,
quebrando a garantia `a is b` que os consumidores assumem.

### 4.4 Regeneração dos `.npz`

Quando `Fortran_Gerador/filtersv2.f08` for modificado (novo filtro ou
correção de pesos), re-execute:

```bash
python scripts/extract_hankel_weights.py          # extração completa
python scripts/extract_hankel_weights.py --verify # só auditoria
python scripts/extract_hankel_weights.py --verbose # debug
```

O script:

1. Computa SHA-256 do `filtersv2.f08`.
2. Parseia as subrotinas `J0J1Wer`, `J0J1Kong`, `J0J1And` via regex robusta
   (aceita `1.23D+02`, `0.21D-28`, `.21D-28` e `1D0`).
3. Converte notação Fortran `1.23D+02` para Python `1.23E+02`.
4. Valida `len == npt_esperado` e `abscissas > 0, crescente`.
5. Grava `.npz` comprimido com metadata JSON (hash + timestamp + script).

---

## 5. SimulationConfig (Sprint 1.2)

### 5.1 API

```python
from geosteering_ai.simulation import SimulationConfig

# Defaults (frequency_hz=20000, tr_spacing_m=1.0, n_positions=600,
#           backend=fortran_f2py, dtype=complex128, device=cpu,
#           hankel_filter=werthmuller_201pt)
cfg = SimulationConfig()

# Customização manual
cfg = SimulationConfig(
    frequency_hz=400000.0,
    tr_spacing_m=1.5,
    backend="jax",
    device="gpu",
    dtype="complex64",
    hankel_filter="kong_61pt",
    frequencies_hz=[20000.0, 400000.0],  # F5 multi-frequência
    tr_spacings_m=[0.5, 1.0, 1.5],        # multi-TR
)

# Presets @classmethod
SimulationConfig.default()          # == SimulationConfig()
SimulationConfig.high_precision()   # anderson_801pt + complex128
SimulationConfig.production_gpu()   # jax + gpu + complex64 + kong_61pt
SimulationConfig.realtime_cpu()     # numba + cpu + complex128 + kong_61pt
```

### 5.2 Serialização

```python
# Dict (nativo do Python, sem dependências)
d = cfg.to_dict()
cfg_restored = SimulationConfig.from_dict(d)
assert cfg == cfg_restored

# YAML (requer pyyaml, lazy import)
cfg.to_yaml("configs/sim_production_gpu.yaml")
cfg_loaded = SimulationConfig.from_yaml("configs/sim_production_gpu.yaml")
```

### 5.3 Validação de errata (via `__post_init__`) — **expandida Sprint 2.1**

| Campo              | Regra                                                       |
|:-------------------|:------------------------------------------------------------|
| `frequency_hz`     | **10 ≤ f ≤ 2e6** (CSAMT baixa + ARC/PeriScope 2 MHz)       |
| `tr_spacing_m`     | **0.01 ≤ L ≤ 50** (curtas + deep-reading 20.43 m PeriScope HD) |
| `n_positions`      | 10 ≤ N ≤ 100_000                                            |
| `backend`          | `{fortran_f2py, numba, jax}`                               |
| `dtype`            | `{complex128, complex64}`                                   |
| `device`           | `{cpu, gpu}`                                                |
| `hankel_filter`    | `{werthmuller_201pt, kong_61pt, anderson_801pt}`           |
| conflito mútuo     | `backend=fortran_f2py` exige `device=cpu`                   |
| conflito mútuo     | `backend=numba` exige `device=cpu`                          |
| `frequencies_hz`   | se ≠ None: len ≥ 1, todos no range de `frequency_hz`       |
| `tr_spacings_m`    | se ≠ None: len ≥ 1, todos no range de `tr_spacing_m`       |
| `num_threads`      | -1 (auto) ou ≥ 1                                            |

**Motivação dos limites expandidos** (documentada em plano-mãe §16):

- `frequency_hz = 2 MHz`: paridade com ferramentas LWD dual-frequency
  (ARC6, PeriScope, EcoScope rodam 400 kHz + 2 MHz).
- `tr_spacing_m = 50 m`: cobre deep-reading PeriScope HD (20.43 m) +
  GeoSphere + margem. Limite físico real do filtro Werthmüller 201pt
  é ~30 m; Anderson 801pt cobre até ~1000 m.
- **Alta resistividade** (ρ > 1000 Ω·m) em carbonatos/sal/crosta seca
  é caso de uso legítimo — testes de paridade Numba vs analítico na
  Sprint 2.6 incluem `rho ∈ [1, 100, 1000, 10000, 100000]` Ω·m.
- Limite superior 2 MHz preserva validade da aproximação quasi-estática
  (|ωε/σ| < 1%) até ρ = 10 000 Ω·m. Acima de 2 MHz ou em dielétricos
  (ρ > 1e5 Ω·m com f > 1 MHz) é necessário modo "full EM" (fora do
  escopo da Fase 2).

### 5.4 Imutabilidade

`SimulationConfig` é `@dataclass(frozen=True)`. Para mutar um campo:

```python
import dataclasses
cfg2 = dataclasses.replace(cfg, frequency_hz=400000.0)
# cfg2 é uma NOVA instância re-validada pelo __post_init__
```

---

## 6. Validation (Sprint 1.3 — soluções analíticas)

### 6.1 API

```python
from geosteering_ai.simulation.validation import (
    MU_0,                          # 4π × 10⁻⁷ H/m
    static_decoupling_factors,     # Caso 1
    skin_depth,                    # Caso 2
    wavenumber_quasi_static,       # Caso 3
    vmd_fullspace_axial,           # Caso 4
    vmd_fullspace_broadside,       # Caso 5
)

# Caso 1: constantes de decoupling (limite estático)
ACp, ACx = static_decoupling_factors(L=1.0)
# ACp ≈ -0.079577 (errata CLAUDE.md, exato)
# ACx ≈ +0.159155 (errata CLAUDE.md, exato)

# Caso 2: skin depth (escalar ou array)
delta = skin_depth(20000.0, 1.0)      # 3.56 m (f=20kHz, ρ=1 Ω·m)
deltas = skin_depth(np.array([1e3, 1e4, 1e5]), 1.0)  # broadcast

# Caso 3: wavenumber complexo (quasi-estático)
k = wavenumber_quasi_static(20000.0, 1.0)
# k ≈ 0.281 + 0.281i  (Im(k) > 0 → decaimento com r)

# Caso 4: VMD full-space axial (receptor em (0,0,L))
H = vmd_fullspace_axial(L=1.0, frequency_hz=20000.0, resistivity_ohm_m=1.0)
# H ≈ 0.157 + 0.010i  A/m (moment-normalized)

# Caso 5: VMD full-space broadside (receptor em (L,0,0))
H = vmd_fullspace_broadside(L=1.0, frequency_hz=20000.0, resistivity_ohm_m=1.0)
# H ≈ -0.081 + 0.004i A/m
```

### 6.2 Convenções

- **Temporal**: e^(-iωt) (padrão geofísica, Moran-Gianzero 1979)
- **Quasi-estática**: k² = iωμ₀σ → k com parte imaginária positiva
- **Propagação**: e^(ikr) atenua para r crescente (Im(k) > 0)

### 6.3 Propriedades validadas pelos 38 testes

| Caso | Propriedade                                           |
|:----:|:------------------------------------------------------|
| 1    | ACp = -1/(4π L³) bit-exato; ACx = +1/(2π L³) bit-exato |
| 1    | Razão ACx/(-ACp) = 2 (invariante geométrica)         |
| 1    | Escalamento 1/L³                                      |
| 2    | δ = √(ρ/(πfμ₀)) bit-exato                            |
| 2    | δ ∝ 1/√f e δ ∝ √ρ                                    |
| 3    | Im(k) > 0 (atenuação), Re(k) = Im(k), |k|·δ = √2     |
| 4    | Limite estático → ACx·m (diff = 0 em f=1e-6)          |
| 4    | Linearidade no momento m                              |
| 4    | Skin effect: |H(f_alta)| < |H(f_baixa)|               |
| 5    | Limite estático → ACp·m (diff = 0 em f=1e-6)          |
| 5    | Sinal negativo (broadside)                             |
| Cross| Razão H_axial/H_broadside = -2 no limite estático    |
| Cross| Caso 1 == limites estáticos dos casos 4 e 5 (consist.)|

### 6.4 Papel na validação dos backends (Fases 2-3)

Os 5 casos analíticos servem como **ground-truth independente** para
comparar os backends Numba (Fase 2) e JAX (Fase 3) contra valores
exatos. Estratégia de validação:

```python
# Na Fase 2, após implementar _numba/kernel.py:
from geosteering_ai.simulation import SimulationConfig, simulate
from geosteering_ai.simulation.validation import vmd_fullspace_axial

cfg = SimulationConfig(backend="numba", ...)
H_numba = simulate(cfg, profile=homogeneous(rho=1.0), ...)  # 1 camada
H_analytic = vmd_fullspace_axial(L=1.0, frequency_hz=20000.0, resistivity_ohm_m=1.0)
assert abs(H_numba - H_analytic) < 1e-10  # paridade float64
```

---

## 7. Fluxo de Dados Alvo (Fases 2-3)

```
┌──────────────────────────────────────────────────────────────────────┐
│  SimulationConfig (profile ρh/ρv, frequências, geometria, backend) │
│      │                                                             │
│      ▼                                                             │
│  simulate(config)  ───►  backend='numba' ─► _numba/kernel.py      │
│      │                                                             │
│      │                   backend='jax'   ─► _jax/kernel.py        │
│      ▼                                                             │
│  forward pipeline:                                                 │
│    1. findlayersTR2well   (geometry.py)                            │
│    2. commonfactorsMD     (propagation.py)  ← ~20% custo          │
│    3. commonarraysMD      (propagation.py)  ← ~55% custo          │
│    4. HankelQuadrature    (hankel.py)       ← ~3%  custo          │
│    5. hmd_TIV / vmd       (dipoles.py)      ← ~20% custo          │
│    6. RtHR rotation       (rotation.py)     ← ~2%  custo          │
│      │                                                             │
│      ▼                                                             │
│  H_tensor  shape=(n_med, 9)  complex128                            │
│      │                                                             │
│      ▼                                                             │
│  postprocess: FV, GS, compensação midpoint F6, antenas F7        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8. Metas de Performance

| Backend              | Meta           | Referência                   |
|:---------------------|:---------------|:-----------------------------|
| **Numba CPU**        | ≥ 40.000 mod/h | Fortran/OpenMP baseline 58k  |
| **JAX CPU (JIT)**    | ≥ 40.000 mod/h | Avaliação adicional (item 1) |
| **JAX GPU T4**       | ≥ 200.000 mod/h | Colab Pro+                  |
| **JAX GPU A100**     | ≥ 500.000 mod/h | HPC / Colab Pro+            |
| **Jacobiano (jacfwd GPU A100)** | ≥ 10× JAX forward | Usando autodiff         |

---

## 9. Validação Cruzada (Fase 4)

Triângulo de validação para garantir correção matemática:

```
        Fortran (tatu.x)
          /        \
         /          \
        / ≤ 1e-10   \ ≤ 1e-10
       /             \
      /               \
  Numba ─── ≤ 1e-14 ── JAX
            ↑
         empymod
     (3ª fonte independente)
```

- **Tolerância** float64: `≤ 1e-14` (bit-idêntico modulo FMA).
- **Tolerância** float32: `≤ 1e-6` (uso em treino DL apenas).
- **Caso de referência 1**: meio homogêneo (half-space) com solução analítica.
- **Caso de referência 2**: 3 camadas contrastantes (padrão Anderson 1989).
- **Caso de referência 3**: perfil Sobol geo-realista (32 camadas).

---

## 10. Integração com Arquitetura v2.0

### 8.1 Chain-of-trust dos filtros

```
filtersv2.f08 (Fortran, 5559 linhas)
   │
   │  scripts/extract_hankel_weights.py (parser + SHA-256)
   ▼
geosteering_ai/simulation/filters/*.npz  (versionado)
   │
   │  FilterLoader (cache + validação)
   ▼
HankelFilter (imutável, arrays read-only)
   │
   ▼
_numba/hankel.py  ou  _jax/hankel.py   (consumidores)
```

### 8.2 Backend no `PipelineConfig`

Enquanto a Fase 6 não estiver pronta, o default permanece:

```python
@dataclass
class PipelineConfig:
    simulator_backend: str = "fortran_f2py"   # opções futuras: 'numba', 'jax'
    simulation_dtype: str = "complex128"      # 'complex64' para produção GPU
    hankel_filter: str = "werthmuller_201pt"  # ou 'kong_61pt', 'anderson_801pt'
```

A mudança para `backend='jax'` será **opcional** e avaliada no final
da Fase 3 (critério: paridade numérica + performance ≥ 90% do Fortran).

---

## 11. Padrões Aplicados (D1-D14)

Todo código novo do simulador Python segue rigorosamente os padrões
D1–D14 do `CLAUDE.md`:

- **D1** mega-header Unicode de 14 campos em cada .py
- **D2** cabeçalhos de seção com contexto
- **D3** diagramas ASCII em docstrings
- **D4** atributos de config documentados em grupos
- **D5/D6** docstrings Google-style em todas as funções/classes
- **D7** comentários inline semânticos
- **D8** `__all__` em todos os módulos
- **D9** `logging` (nunca `print`)
- **D10** constantes com unidades e significado físico
- **D11** tabelas ASCII em catálogos
- **D12** cross-references via `Note:` em docstrings
- **D13** branch comments com layout de saída
- **D14** diagrama de pipeline em módulos orquestradores

Idioma: **docstrings e comentários em PT-BR com acentuação correta**.

---

## 12. Comandos Úteis

```bash
# Rodar testes do simulador Python (filtros)
pytest tests/test_simulation_filters.py -v

# Re-extrair pesos Hankel do Fortran
python scripts/extract_hankel_weights.py

# Verificar sincronia dos .npz com filtersv2.f08
python scripts/extract_hankel_weights.py --verify

# Checar tamanho e metadata dos filtros
python -c "
from geosteering_ai.simulation.filters import FilterLoader
loader = FilterLoader()
for name in loader.available():
    f = loader.load(name)
    print(f'{name:20s}  npt={f.npt}  hash={f.source_sha256[:12]}...')
"
```

---

## 13. Referências

### 11.1 Documentos internos (MD)

- **Plano detalhado (seção-mãe)**:
  [`docs/reference/plano_simulador_python_jax_numba.md`](../../docs/reference/plano_simulador_python_jax_numba.md)
- **Simulador Fortran equivalente**:
  [`docs/reference/documentacao_simulador_fortran.md`](../../docs/reference/documentacao_simulador_fortran.md)
- **README dos filtros**:
  [`geosteering_ai/simulation/filters/README.md`](../../geosteering_ai/simulation/filters/README.md)
- **Arquitetura v2.0**:
  [`docs/ARCHITECTURE_v2.md`](../../docs/ARCHITECTURE_v2.md)
- **ROADMAP**: [`docs/ROADMAP.md`](../../docs/ROADMAP.md) — Fase F7 (Simulador Python)
- **Sub skill Fortran**: `.claude/commands/geosteering-simulator-fortran.md`
- **Sub skill v2**: `.claude/commands/geosteering-v2.md`

### 11.2 Bibliografia

- Werthmüller, D. (2017). "empymod: open-source 3D EM modeler for 1D VTI."
  *Geophysics* 82(6), WB9-WB19.
- Kong, F. N. (2007). "Hankel transform filters for dipole antenna radiation."
  *Geophysical Prospecting* 55(1), 83-89.
- Anderson, W. L. (1989). "A hybrid fast Hankel transform algorithm for EM
  modeling." *Geophysics* 54(2), 263-266.
- Frostig, R. et al. (2018). "Compiling machine learning programs via
  high-level tracing." *MLSys* (JAX design).
- Lam, S. K. et al. (2015). "Numba: a LLVM-based Python JIT compiler."
  *LLVM-HPC*.

---

## 14. Triggers para Uso desta Skill

Use esta sub skill quando o usuário mencionar:

- Simulador Python, simulação Python, backends JAX/Numba
- Funções específicas (`simulate()`, `FilterLoader`, `HankelFilter`)
- Pesos Hankel, filtros, abscissas, `extract_hankel_weights`
- Módulos `geosteering_ai/simulation/*`
- Comparação Python vs Fortran, paridade numérica, empymod
- Fases/Sprints do plano `plano_simulador_python_jax_numba.md`
- Branch `feature/simulator-python`
- JIT, vmap, pmap, jacfwd, njit, prange
- Meta de performance para CPU/GPU
- Hot kernels: `commonarraysMD`, `commonfactorsMD`, `hmd_TIV`, `vmd`
