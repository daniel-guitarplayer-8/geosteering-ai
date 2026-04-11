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
| **Versão**       | 0.1.0 (Sprint 1.1 concluída)                              |
| **Branch**       | `feature/simulator-python`                                |
| **Base**         | `main` (a partir do commit F10 Jacobiano `f1bd6e9`)       |
| **Autor**        | Daniel Leal                                               |
| **Framework**    | NumPy 2.x + Numba 0.60+ + JAX 0.4.30+ + empymod (valid.)  |
| **Precisão**     | `complex128` default + `complex64` via config (prod.)     |
| **Filtro default** | Werthmüller 201pt (mantém paridade Fortran filter_type=0) |
| **Referência**   | `docs/reference/plano_simulador_python_jax_numba.md`      |

### 1.2 Fases do plano (7 fases, 3 concluídas e 4 pendentes)

| Fase | Nome                                    | Status      | Sprint(s) |
|:----:|:----------------------------------------|:------------|:----------|
|  0   | Setup (branch, deps, estrutura)        | 🟡 Parcial  | 1.1 ✅    |
|  1   | Foundations (filtros, config, geometry) | 🟡 Parcial  | 1.1 ✅, 1.2, 1.3 |
|  2   | Backend Numba CPU (paridade Fortran)    | ⬜ Pendente | 2.1-2.5   |
|  3   | Backend JAX (CPU+GPU, vmap+jit)         | ⬜ Pendente | 3.1-3.4   |
|  4   | Validação cruzada (Fortran/Numba/empymod) | ⬜ Pendente | 4.1-4.3 |
|  5   | Jacobiano ∂H/∂ρ (jacfwd JAX, FD Numba)  | ⬜ Pendente | 5.1-5.2   |
|  6   | Integração no PipelineConfig (backend)  | ⬜ Pendente | 6.1-6.2   |
|  7   | Otimizações finais (pmap, XLA, caching) | ⬜ Pendente | 7.1-7.3   |

### 1.3 Sprint 1.1 — entregas concluídas (2026-04-11)

- `scripts/extract_hankel_weights.py` — parser do Fortran `filtersv2.f08`
  que extrai abscissas + pesos J₀ + pesos J₁ para Kong 61pt, Werthmüller
  201pt e Anderson 801pt, com validação bit-a-bit e hash SHA-256 auditável.
- `geosteering_ai/simulation/filters/*.npz` — 3 artefatos com os pesos
  (metadata JSON embutido). Total < 30 KB.
- `geosteering_ai/simulation/filters/loader.py` — `FilterLoader` + classe
  `HankelFilter` (`@dataclass(frozen=True)` com arrays read-only).
- `tests/test_simulation_filters.py` — 45 testes validando presença,
  bit-exactness, API de aliases, imutabilidade e sincronia com Fortran.
- `geosteering_ai/simulation/__init__.py` e
  `geosteering_ai/simulation/filters/README.md` — documentação.

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
├── __init__.py                ← fachada pública (FilterLoader hoje)
├── config.py                  ← [PENDENTE Sprint 1.2] SimulationConfig
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
│   ├── __init__.py
│   ├── loader.py              ← FilterLoader, HankelFilter
│   ├── README.md
│   ├── werthmuller_201pt.npz  ← filter_type=0 (default)
│   ├── kong_61pt.npz          ← filter_type=1 (rápido)
│   └── anderson_801pt.npz     ← filter_type=2 (preciso)
│
├── geometry.py                ← [PENDENTE Fase 2] dip, azimute, posições
├── postprocess.py             ← [PENDENTE Fase 2] FV, GS, compensação
│
├── validation/                ← [PENDENTE Fase 4] testes de referência
│   ├── half_space.py          ← solução analítica meio homogêneo
│   ├── compare_fortran.py     ← bit-exactness vs tatu.x
│   └── compare_empymod.py     ← cross-check com empymod (3ª fonte)
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
2. Parseia as subrotinas `J0J1Wer`, `J0J1Kong`, `J0J1And` via regex.
3. Converte notação Fortran `1.23D+02` para Python `1.23E+02`.
4. Valida `len == npt_esperado` e `abscissas > 0, crescente`.
5. Grava `.npz` comprimido com metadata JSON (hash + timestamp + script).

---

## 5. Fluxo de Dados Alvo (Fases 2-3)

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

## 6. Metas de Performance

| Backend              | Meta           | Referência                   |
|:---------------------|:---------------|:-----------------------------|
| **Numba CPU**        | ≥ 40.000 mod/h | Fortran/OpenMP baseline 58k  |
| **JAX CPU (JIT)**    | ≥ 40.000 mod/h | Avaliação adicional (item 1) |
| **JAX GPU T4**       | ≥ 200.000 mod/h | Colab Pro+                  |
| **JAX GPU A100**     | ≥ 500.000 mod/h | HPC / Colab Pro+            |
| **Jacobiano (jacfwd GPU A100)** | ≥ 10× JAX forward | Usando autodiff         |

---

## 7. Validação Cruzada (Fase 4)

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

## 8. Integração com Arquitetura v2.0

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

## 9. Padrões Aplicados (D1-D14)

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

## 10. Comandos Úteis

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

## 11. Referências

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

## 12. Triggers para Uso desta Skill

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
