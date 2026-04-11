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
| **Versão**       | 0.3.0 (Sprints 1.1, 1.2 e 1.3 concluídas)                 |
| **Branch**       | `feature/simulator-python`                                |
| **Base**         | `main` (a partir do commit F10 Jacobiano `f1bd6e9`)       |
| **Autor**        | Daniel Leal                                               |
| **Framework**    | NumPy 2.x + Numba 0.60+ + JAX 0.4.30+ + empymod (valid.)  |
| **Precisão**     | `complex128` default + `complex64` via config (prod.)     |
| **Filtro default** | Werthmüller 201pt (mantém paridade Fortran filter_type=0) |
| **Testes**       | **153/153 PASS** em 1.81s (53 filtros + 62 config + 38 half-space) |
| **Referência**   | `docs/reference/plano_simulador_python_jax_numba.md`      |

### 1.2 Fases do plano (7 fases, Fase 1 concluída)

| Fase | Nome                                    | Status      | Sprint(s) |
|:----:|:----------------------------------------|:------------|:----------|
|  0   | Setup (branch, deps, estrutura)         | ✅ Concluída | 1.1 ✅ |
|  1   | Foundations (filtros, config, analítico) | ✅ **Concluída** | 1.1 ✅, 1.2 ✅, 1.3 ✅ |
|  2   | Backend Numba CPU (paridade Fortran)    | ⬜ Pendente | 2.1-2.5   |
|  3   | Backend JAX (CPU+GPU, vmap+jit)         | ⬜ Pendente | 3.1-3.4   |
|  4   | Validação cruzada (Fortran/Numba/empymod) | ⬜ Pendente | 4.1-4.3 |
|  5   | Jacobiano ∂H/∂ρ (jacfwd JAX, FD Numba)  | ⬜ Pendente | 5.1-5.2   |
|  6   | Integração no PipelineConfig (backend)  | ⬜ Pendente | 6.1-6.2   |
|  7   | Otimizações finais (pmap, XLA, caching) | ⬜ Pendente | 7.1-7.3   |

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

| Suíte                              | Testes | Tempo  |
|:-----------------------------------|:------:|:------:|
| tests/test_simulation_filters.py   |   53   | 0.82s  |
| tests/test_simulation_config.py    |   62   | ~0.8s  |
| tests/test_simulation_half_space.py |   38   | ~0.2s  |
| **TOTAL**                          | **153** | **1.81s** |

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
├── _numba/                    ← [PENDENTE Fase 2] backend CPU (njit+prange)
│   ├── __init__.py
│   ├── propagation.py         ← commonarraysMD, commonfactorsMD
│   ├── dipoles.py             ← hmd_TIV, vmd
│   ├── hankel.py              ← quadratura Hankel digital
│   ├── rotation.py            ← RtHR (Euler tensor rotation)
│   ├── jacobian.py            ← ∂H/∂ρ via finite differences
│   └── kernel.py              ← orchestrador forward
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

### 5.3 Validação de errata (via `__post_init__`)

| Campo              | Regra                                                       |
|:-------------------|:------------------------------------------------------------|
| `frequency_hz`     | 100 ≤ f ≤ 1e6 (LWD comercial 2 kHz-400 kHz)               |
| `tr_spacing_m`     | 0.1 ≤ L ≤ 10.0                                             |
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
