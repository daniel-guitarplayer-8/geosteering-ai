# Análise de Cenários e Estratégias de Otimização — Simulador Python Numba JIT

<!-- Metadados -->
| Campo | Valor |
|:------|:------|
| **Versão base** | v2.21 (commit `cba27dd`) |
| **Data** | 2026-05-02 |
| **Branch** | `feat/simulation-manager-v2.21` |
| **Filtro padrão** | Werthmuller 201pt (`werthmuller_201pt`) |
| **Hardware referência** | Apple M-series 8C físicos / 16T lógicos (HT/SMT) |
| **Paridade Fortran** | <1e-12 (inviolável) |

---

## Sumário

1. [Objetivo e Escopo](#1-objetivo-e-escopo)
2. [Arquitetura do Simulador — Visão Geral](#2-arquitetura-do-simulador--visão-geral)
3. [Mapa Completo de Paralelismo (v2.21)](#3-mapa-completo-de-paralelismo-v221)
4. [Taxonomia Completa de Cenários](#4-taxonomia-completa-de-cenários)
5. [Análise Detalhada — Cenários Básicos (8 combinações)](#5-análise-detalhada--cenários-básicos-8-combinações)
6. [Análise Detalhada — Cenários Avançados (F6, F7, Batch)](#6-análise-detalhada--cenários-avançados-f6-f7-batch)
7. [Alta Resistividade (ρ > 1000 Ω·m)](#7-alta-resistividade-ρ--1000-ωm)
8. [Estratégias de Otimização — Catálogo Completo](#8-estratégias-de-otimização--catálogo-completo)
9. [Roadmap de Implementação](#9-roadmap-de-implementação)
10. [Novos Benchmarks Necessários](#10-novos-benchmarks-necessários)
11. [Tabelas Consolidadas de Referência](#11-tabelas-consolidadas-de-referência)
12. [Apêndice A — Grafo de Chamadas Completo](#apêndice-a--grafo-de-chamadas-completo)
13. [Apêndice B — Anatomia dos Arrays por Camada](#apêndice-b--anatomia-dos-arrays-por-camada)
14. [Apêndice C — Análise de Custo Computacional](#apêndice-c--análise-de-custo-computacional)

---

## 1. Objetivo e Escopo

### 1.1 Objetivo

Este documento registra a análise técnica completa de todos os cenários de simulação suportados pelo simulador Python Numba JIT do projeto Geosteering AI, com foco em:

1. **Identificar o comportamento de performance de cada cenário** — como o paralelismo atual se comporta em função de `nf`, `nAng`, `nTR` e `n_pos`;
2. **Apontar lacunas de otimização** — onde o código atual deixa CPUs ociosas ou paga overhead desnecessário;
3. **Propor estratégias concretas** de otimização máxima para cada combinação de dimensões;
4. **Garantir robustez física** — especialmente para alta resistividade (ρ > 1000 Ω·m) e precisão geofísica comparável ao Fortran.

### 1.2 Escopo do Simulador

O simulador Python Numba JIT deve replicar **todas as capacidades do simulador Fortran `tatu.x`**:

```
┌───────────────────────────────────────────────────────────────────────────┐
│  CAPACIDADES DO SIMULADOR (alvo de paridade total com Fortran)            │
│                                                                           │
│  Física:                                                                  │
│    • Propagação EM 1D em meio TIV (transversalmente isotrópico)           │
│    • Dipolo magnético horizontal (HMD) e vertical (VMD)                   │
│    • Integração de Hankel (filtro Werthmuller 201pt — padrão neste doc)   │
│    • Tensor completo de 9 componentes: H = [Hxx, Hxy, ..., Hzz]          │
│    • Rotação de tensor para qualquer ângulo de inclinação (dip)           │
│    • Alta resistividade: ρ ∈ [0.01, 1e6] Ω·m                             │
│    • Multicamadas: n_layers ∈ [2, 100+]                                   │
│                                                                           │
│  Funcionalidades:                                                         │
│    • F5: Frequências arbitrárias (nf ∈ [1, 16])                          │
│    • F6: Compensação midpoint CDR (nTR ≥ 2, comp_pairs)                  │
│    • F7: Antenas inclinadas (β, φ configuráveis)                          │
│    • Multi-TR: nTR ∈ [1, 8] espaçamentos                                 │
│    • Multi-ângulo: nAng ∈ [1, 32] ângulos de inclinação                  │
│    • Batch: n_models ∈ [1, 100000+] modelos geológicos                   │
│                                                                           │
│  Performance:                                                             │
│    • Filtro Werthmuller 201pt: npt=201 (padrão de análise neste doc)     │
│    • Filtro Kong 61pt: npt=61 (3.3× mais rápido, ε~1e-10)                │
│    • Filtro Anderson 801pt: npt=801 (máxima precisão, 4× mais lento)     │
└───────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Versões Relevantes

| Versão | Sprint | Melhoria Principal | Impacto em Performance |
|:------:|:------:|:------------------|:----------------------:|
| v2.9 | 2.9 | `@njit(parallel=True)` + `prange(n_pos)` | 5–8× vs Python puro |
| v2.10 | 2.10 | Cache `common_arrays` por `hordist` | 4–6× em produção |
| v2.12 | 12.1 | `ProcessPoolExecutor` batch nativo | N× (N=n_workers) |
| v2.13 | 13.1 | `prange(nf)` em precompute + `nogil=True` universal | 1.5× multi-freq |
| v2.16 | 13.3 | FLAT `prange(nTR×nAng×n_pos)` | 2–3× multi-TR/angle |
| v2.17 | — | `detect_cpu_topology` + defaults CPU-aware | +2.24× (anti-oversubscription) |
| v2.21 | 21.1 | Remoção `parallel=True` em `_fields_in_freqs_kernel_cached` | **2.65× Cenário E** |

---

## 2. Arquitetura do Simulador — Visão Geral

### 2.1 Estrutura de Módulos

```
geosteering_ai/simulation/
│
├── multi_forward.py         ← Ponto de entrada principal: simulate_multi()
├── forward.py               ← Kernels Numba: _simulate_combined_prange,
│                               _simulate_positions_njit_cached, etc.
│
├── _numba/
│   ├── kernel.py            ← Orquestrador: _fields_in_freqs_kernel_cached,
│   │                           precompute_common_arrays_cache, fields_in_freqs
│   ├── propagation.py       ← Física EM: common_arrays, common_factors
│   ├── dipoles.py           ← Integrais de Hankel: hmd_tiv, vmd
│   ├── hankel.py            ← Kernels de Hankel (fastmath=True)
│   ├── geometry.py          ← find_layers_tr, find_layers_tr_jax
│   └── rotation.py          ← rotate_tensor
│
├── postprocess/
│   ├── compensation.py      ← F6: apply_compensation (CDR midpoint)
│   └── tilted.py            ← F7: apply_tilted_antennas
│
├── filters/
│   ├── werthmuller_201pt.npz  ← npt=201, ε~1e-14 (PADRÃO deste doc)
│   ├── kong_61pt.npz          ← npt=61,  ε~1e-10 (3.3× mais rápido)
│   └── anderson_801pt.npz     ← npt=801, ε~1e-16 (4× mais lento)
│
├── _workers.py              ← ProcessPoolExecutor, batch parallelism
│
└── tests/
    ├── simulation_manager.py  ← GUI PyQt6 + SimulationManager
    └── sm_workers.py          ← recommend_default_parallelism, detect_cpu_topology
```

### 2.2 Fluxo de Dados Geral

```
                ╔═══════════════════════════════════════════════════╗
                ║  INPUT                                            ║
                ║    rho_h: (n,)  — resistividade horizontal        ║
                ║    rho_v: (n,)  — resistividade vertical          ║
                ║    esp:   (n-2,) — espessuras internas            ║
                ║    positions_z: (n_pos,) — profundidades de medição║
                ║    frequencies_hz: (nf,) — frequências EM         ║
                ║    tr_spacings_m: (nTR,) — espaçamentos TX-RX     ║
                ║    dip_degs: (nAng,) — ângulos de inclinação      ║
                ╚══════════════╦════════════════════════════════════╝
                               ↓
                ╔══════════════╩════════════════════════════════════╗
                ║  simulate_multi() — Orquestrador Python           ║
                ║                                                   ║
                ║  1. Validação de inputs (_validate_multi_inputs)  ║
                ║  2. Normalização de arrays (contiguous float64)   ║
                ║  3. Geometria: h_arr, prof_arr, eta               ║
                ║  4. Cache deduplication (unique hordist)          ║
                ║  5. Despacho: batch OU single-model               ║
                ╚══════════════╦════════════════════════════════════╝
                               ↓
          ┌────────────────────┴─────────────────────┐
          │ models=None?                              │ models=[...]?
          ↓                                           ↓
  ╔═══════════════════╗                  ╔═══════════════════════════╗
  ║  SINGLE-MODEL     ║                  ║  BATCH (ProcessPool)      ║
  ║                   ║                  ║                           ║
  ║  precompute_cache ║                  ║  _workers.run_batch()     ║
  ║  → prange(nf)     ║                  ║  → N workers × chunk      ║
  ║                   ║                  ║  → cada worker chama      ║
  ║  _simulate_       ║                  ║    simulate_multi()       ║
  ║  combined_prange  ║                  ║    recursivamente         ║
  ║  → prange(        ║                  ╚═══════════════════════════╝
  ║    nTR×nAng×n_pos)║
  ║                   ║
  ║  F6? → apply_comp ║
  ║  F7? → apply_tilt ║
  ╚═══════════════════╝
          ↓
  ╔══════════════════════════════════════════════════════════════════╗
  ║  OUTPUT: MultiSimulationResult                                  ║
  ║    H_tensor: (nTR, nAng, n_pos, nf, 9) complex128               ║
  ║    z_obs:    (nAng, n_pos) float64                              ║
  ║    rho_h_at_obs, rho_v_at_obs: (nAng, n_pos) float64           ║
  ║    H_comp:   (ncomp, nAng, n_pos, nf) complex128  [se F6]       ║
  ║    H_tilted: (n_tilted, nTR, nAng, n_pos, nf) complex128 [se F7]║
  ╚══════════════════════════════════════════════════════════════════╝
```

---

## 3. Mapa Completo de Paralelismo (v2.21)

### 3.1 Três Camadas de Paralelismo

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  CAMADA 1 — INTER-PROCESS: Batch de Modelos Geológicos                   ║
║                                                                          ║
║  ProcessPoolExecutor(n_workers=4, threads_per_worker=2)                 ║
║    ├── Worker 0 → simulate_multi(chunk_0, n_models=7500) → 2 threads   ║
║    ├── Worker 1 → simulate_multi(chunk_1, n_models=7500) → 2 threads   ║
║    ├── Worker 2 → simulate_multi(chunk_2, n_models=7500) → 2 threads   ║
║    └── Worker 3 → simulate_multi(chunk_3, n_models=7500) → 2 threads   ║
║                                                                          ║
║  Configuração ótima (v2.17, empiricamente confirmada v2.20):            ║
║    n_workers = phys_cores // 2  →  4 workers em 8C/16T                  ║
║    threads_per_worker = 2       →  2 threads Numba por worker           ║
║    Total: 4 × 2 = 8 threads = phys_cores (sem oversubscription)        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  CAMADA 2 — INTRA-PROCESS: Numba Threads (dentro de cada worker)        ║
║                                                                          ║
║  @njit(parallel=True, cache=True, nogil=True)                           ║
║  _simulate_combined_prange(dz_halfs, r_halfs, dip_rads_flat,           ║
║                             cache_indices, positions_z, ...)            ║
║                                                                          ║
║    n_combos = nTR × nAng                                                ║
║    n_total  = n_combos × n_pos                                          ║
║                                                                          ║
║    for k in prange(n_total):   ← PARALELO (nTR × nAng × n_pos tasks)  ║
║        i_combo = k // n_pos                                             ║
║        j       = k % n_pos                                              ║
║        _fields_in_freqs_kernel_cached(...)                              ║
║            ↓                                                            ║
║            for i_f in range(nf):  ← SERIAL ⚠ (loop frequências)       ║
║                common_factors(Tz, camad_t, u_cache[i_f], ...)          ║
║                hmd_tiv(u_cache[i_f], s_cache[i_f], ...)                ║
║                vmd(u_cache[i_f], s_cache[i_f], ...)                    ║
║                rotate_tensor(dip_rad, matH)                             ║
║                cH[i_f, :] = flatten_9(tH)                              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  CAMADA 3 — PRÉ-CÔMPUTO DE CACHE (uma única vez por contexto)           ║
║                                                                          ║
║  @njit(cache=True, parallel=True, nogil=True)                           ║
║  precompute_common_arrays_cache(hordist, freqs_hz, n, npt, ...)        ║
║                                                                          ║
║    for i_f in prange(nf):    ← PARALELO (correto — sem aninhamento)   ║
║        u[i_f], s[i_f], uh[i_f], sh[i_f], RTEdw[i_f], RTEup[i_f],     ║
║        RTMdw[i_f], RTMup[i_f], AdmInt[i_f] = common_arrays(...)        ║
║                                                                          ║
║    → Output: (9 arrays) × (nf, npt, n)   ← cache completo             ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 3.2 Por Que `range(nf)` Dentro do Hot Path É Serial (e Correto)

A decisão de manter `range(nf)` serial dentro de `_fields_in_freqs_kernel_cached` é **arquiteturalmente correta** dado o estado atual da v2.21, e é **consequência direta do fix da regressão histórica**:

```
HISTÓRIA:
  v2.10–v2.12: range(nf) serial → ~120k mod/h em Cenário E ✓
  v2.13 Sprint 13.1: prange(nf) dentro da função → ~46k mod/h ✗
    Causa: nested prange (prange outer em positions × prange inner em nf)
    Numba serializa o prange inner MAS paga overhead do parallel scheduler
    em cada uma das 180k+ chamadas → ~14s de overhead total por simulação
  v2.21 Sprint 21.1: range(nf) serial restaurado → ~122k mod/h ✓

REGRA DERIVADA:
  "Uma função @njit chamada de dentro de prange outer NUNCA deve
   ter parallel=True. O overhead do scheduler aninhado supera qualquer
   ganho teórico para funções de duração < 100μs chamadas > 10k vezes."

SOLUÇÃO CORRETA (não implementada ainda):
  Mover o paralelismo de nf para o NÍVEL DO prange EXTERNO,
  criando um prange único que cubra (nTR × nAng × n_pos × nf).
  Isso elimina o aninhamento e paraleliza nf sem overhead.
```

### 3.3 Decoradores JIT por Função (v2.21)

| Função | Arquivo | `parallel` | `prange` | `cache` | `nogil` | `fastmath` |
|:-------|:--------|:----------:|:--------:|:-------:|:-------:|:----------:|
| `common_arrays` | propagation.py | ✗ | — | ✓ (default) | ✓ | ✗ |
| `common_factors` | propagation.py | ✗ | — | ✓ (default) | ✓ | ✗ |
| `hmd_tiv` | dipoles.py | ✗ | — | ✓ (default) | ✓ | ✗ |
| `vmd` | dipoles.py | ✗ | — | ✓ (default) | ✓ | ✗ |
| `_hankel_j0_kernel` | hankel.py | ✗ | — | ✓ | ✓ | **✓** |
| `_hankel_j1_kernel` | hankel.py | ✗ | — | ✓ | ✓ | **✓** |
| `_fields_in_freqs_kernel` | kernel.py | ✗ | — | **✓ explicit** | ✓ | ✗ |
| `precompute_common_arrays_cache` | kernel.py | **✓** | prange(nf) | **✓ explicit** | **✓ explicit** | ✗ |
| `_fields_in_freqs_kernel_cached` | kernel.py | ✗ (v2.21) | range(nf) | **✓ explicit** | **✓ explicit** | ✗ |
| `_simulate_positions_njit` | forward.py | **✓** | prange(n_pos) | **✓** | **✓** | ✗ |
| `_simulate_positions_njit_cached` | forward.py | **✓** | prange(n_pos) | **✓** | **✓** | ✗ |
| `_simulate_combined_prange` | forward.py | **✓** | prange(n_combos×n_pos) | **✓** | **✓** | ✗ |

---

## 4. Taxonomia Completa de Cenários

### 4.1 Eixos de Variação

```
Dimensão      Valores Típicos       Regime
─────────────────────────────────────────────────────────────────────────
nf            1                     "single-freq"  (default, este doc)
              2–4                   "multi-freq médio"
              6–16                  "multi-freq alto"

nAng          1                     "single-angle" (poço vertical)
              2–4                   "multi-angle médio"
              8–32                  "multi-angle alto"

nTR           1                     "single-TR"    (ferramenta simples)
              2–4                   "multi-TR médio"
              5–8                   "multi-TR alto" (array completo)

n_pos         10–100                "baixo"        (inversão pontual)
              100–500               "médio"        (janela LWD típica)
              500–10000             "alto"         (perfil longo)
              10000+                "extremo"      (perfil completo de poço)

n_layers      2–5                   "simples"      (modelo sintético)
              6–15                  "médio"        (formação típica)
              16–30                 "complexo"     (sequência estratigráfica)
              30+                   "muito complexo"

ρ_max         0.01–10 Ω·m          "baixa" (argilitos, salmouras)
              10–100 Ω·m           "média" (arenitos saturados)
              100–1000 Ω·m         "alta" (carbonatos, arenitos secos)
              1000–1e6 Ω·m         "muito alta" (evaporitas, sal, gás seco)
```

### 4.2 As 8 Combinações Estruturais Básicas

| # | `nf` | `nAng` | `nTR` | Nome | Exemplo de uso |
|:--:|:----:|:------:|:-----:|:-----|:---------------|
| 1 | 1 | 1 | 1 | Base | Referência, smoke test, inversão simples |
| 2 | >1 | 1 | 1 | Multi-freq | ARC 4-freq, poço vertical |
| 3 | 1 | >1 | 1 | Multi-ângulo | Inversão multi-dip, análise de anisotropia |
| 4 | 1 | 1 | >1 | Multi-TR | Perfilagem multi-espaçamento (diferente DoI) |
| 5 | 1 | >1 | >1 | Multi-ângulo + TR | Array 3D, ferramenta aprimorada |
| 6 | >1 | >1 | 1 | Multi-freq + ângulo | ARC multi-dip |
| 7 | >1 | 1 | >1 | Multi-freq + TR | Array multi-espaçamento + multi-freq |
| 8 | >1 | >1 | >1 | Completo | Periscope-15 completo, máxima fidelidade |

### 4.3 Cenários Avançados (Features Opcionais)

| # | Feature | Condição | Dimensão Extra | Custo Adicional |
|:--:|:-------:|:---------|:---------------|:---------------|
| F6 | Compensação CDR | `use_compensation=True`, nTR≥2 | `comp_pairs` pós-processamento | O(n_comp_pairs × n_pos × nf) |
| F7 | Antenas inclinadas | `use_tilted=True` | `n_tilted_configs × nTR × nAng × n_pos × nf` | O(n_tilted × resultado) |
| B | Batch | `models=[...]` | n_models × any cenário | O(n_models) via ProcessPool |

### 4.4 Combinações Extremas (Interesse Científico/Industrial)

```
Cenário "Periscope Completo":
  nf=4, nAng=8, nTR=4, n_pos=600, F7=True
  n_total = 4 × 8 × 4 × 600 × 4 = 307,200 chamadas de kernel por modelo

Cenário "LWD Rápido":
  nf=1, nAng=1, nTR=2, n_pos=600, F6=True
  n_total = 1 × 1 × 2 × 600 × 1 = 1,200 chamadas

Cenário "Inversão Ensemble Treinamento":
  nf=1, nAng=1, nTR=1, n_pos=600, n_models=30000
  n_total = 30,000 × 600 = 18,000,000 chamadas de kernel
```

---

## 5. Análise Detalhada — Cenários Básicos (8 combinações)

> **Nota sobre o filtro Werthmuller 201pt**: Todos os cenários nesta seção usam `npt=201` (Werthmuller), que oferece precisão `ε~1e-14` e equilibrio ótimo entre precisão e custo. O custo de cada chamada `hmd_tiv` e `vmd` é proporcional a `npt × n_layers`.

---

### Cenário 1 — `nf=1, nAng=1, nTR=1` (Referência)

**Descrição física**: Ferramenta LWD de frequência única, poço vertical, par TX-RX único. Configuração de referência para toda análise de performance.

**Função de entrada**: `simulate_multi(nf=1, nAng=1, nTR=1, n_pos=N)`

#### Caminho de Execução

```
simulate_multi(nf=1, nAng=1, nTR=1, n_pos=N)
  ↓
  precompute_common_arrays_cache(hordist=0.0, freqs_hz=[f0])
    └─ prange(nf=1) → 1 iteração → sem ganho de paralelismo no cache
    → u_cache: (1, 201, n_layers)   ← shape com Werthmuller 201pt

  ↓ (n_combos=1 → usa _simulate_combined_prange OU _simulate_positions_njit_cached)
  _simulate_combined_prange(n_combos=1, n_pos=N)
    └─ for k in prange(N):   ← N tarefas paralelas
         _fields_in_freqs_kernel_cached(...)
           └─ for i_f in range(1):  ← 1 iteração (sem overhead)
                common_factors(...)  ← O(npt=201)
                hmd_tiv(...)        ← O(npt × n_layers)
                vmd(...)            ← O(npt × n_layers)
                rotate_tensor(0°)   ← O(1)
                cH[0, :] = ...
```

#### Análise por Regime de `n_pos` (8 cores físicos, Werthmuller 201pt)

| Regime | `n_pos` | Tarefas prange | Tarefas/thread | Overhead fork | Eficiência | Obs. |
|:------:|:-------:|:--------------:|:--------------:|:-------------:|:----------:|:-----|
| Baixo | 10 | 10 | 1.25 | ~5–10% | 55–60% | 6 threads ociosas por iteração |
| Baixo | 30 | 30 | 3.75 | ~3–5% | 65–75% | Benchmark Cenário A atual |
| Médio | 100 | 100 | 12.5 | ~1% | 82–88% | Bom para inversão pontual |
| Médio | 300 | 300 | 37.5 | <0.5% | 90–93% | Qualidade de produção |
| Alto | 600 | 600 | 75 | <0.2% | 93–96% | Cenário E referência |
| Alto | 1200 | 1200 | 150 | <0.1% | 96–98% | Perfil longo LWD |
| Extremo | 10000+ | 10000+ | 1250+ | ~0% | ~99% | Perfil completo de poço |

**Benchmark v2.21 (medido)**:
- Cenário A (n_pos=30): **1.392.371 mod/h** (7.34× acima meta histórica)
- Cenário E (n_pos=600): **121.957 mod/h** (atinge meta histórica >120k)

**Status do paralelismo**: **Ótimo**. `prange(n_pos)` satura bem os cores para n_pos ≥ 100. O único problema é para n_pos baixo (< 50), onde a granularidade é insuficiente.

**Oportunidade de melhoria**: `Adaptive thread count` para n_pos < 50 (§8.1).

---

### Cenário 2 — `nf>1, nAng=1, nTR=1` (Multi-Frequência)

**Descrição física**: Ferramenta com múltiplas frequências simultâneas em poço vertical, par TR único. Exemplo: Halliburton ARC, `frequencies_hz=[400e3, 1e6, 2e6, 400e3]` (nf=4).

**Motivação de uso**: Inversão com múltiplas frequências melhora a resolução vertical da formação; frequências altas investigam raso, frequências baixas investigam profundo.

#### Caminho de Execução

```
simulate_multi(nf=4, nAng=1, nTR=1, n_pos=600)
  ↓
  precompute_common_arrays_cache(hordist=0.0, freqs_hz=[f0,f1,f2,f3])
    └─ prange(nf=4) → 4 iterações PARALELAS ✓
    → u_cache: (4, 201, n_layers)   ← 4 × 201 × n arrays
    → Custo: ~4× do Cenário 1, mas paralelizado em 4 threads

  ↓
  _simulate_combined_prange(n_combos=1, n_pos=600)
    └─ for k in prange(600):   ← 600 tarefas paralelas
         _fields_in_freqs_kernel_cached(...)
           └─ for i_f in range(4):   ← SERIAL ⚠ 4 iterações sequenciais
                common_factors(Tz, camad_t, u_cache[i_f], ...)
                hmd_tiv(u_cache[i_f], s_cache[i_f], ...)   ← O(201 × n) × 4
                vmd(u_cache[i_f], ...)                      ← O(201 × n) × 4
```

#### Problema: Sub-Granularidade para n_pos Baixo com nf Alto

```
Situação: n_pos=30, nf=4 (benchmark Cenário A com multi-freq)

  Tarefas disponíveis: prange(30) = 30 tarefas
  Threads disponíveis: 8
  Tarefas/thread: 3.75  ← insuficiente!

  Dentro de cada tarefa, 4 frequências são computadas sequencialmente.
  Tarefas efetivas: 30 × 4 = 120 unidades de trabalho, mas desigualmente distribuídas.

  Resultado: 4 threads terminam mais cedo que outras → desbalanceamento.

Com FLAT prange(n_pos × nf) = prange(120):
  Tarefas/thread: 15 → muito melhor!
  Balanceamento uniforme: cada thread computa ~15 pares (posição, frequência)
```

#### Análise por Regime de `n_pos` com `nf=4`

| Regime | `n_pos` | Tarefas atual | Tarefas FLAT | Ganho estimado | Obs. |
|:------:|:-------:|:-------------:|:------------:|:--------------:|:-----|
| Baixo | 10 | 10 (4 freq serial) | 40 | **3–4×** | Maior benefício |
| Baixo | 30 | 30 | 120 | **2.5–3×** | Cenário A multi-freq |
| Médio | 150 | 150 | 600 | **1.5–2×** | Benefício moderado |
| Alto | 600 | 600 | 2400 | **1.2–1.5×** | Carga já balanceada |
| Extremo | 5000 | 5000 | 20000 | **<1.1×** | Quase sem diferença |

**Insight crucial**: O ganho do FLAT prange é **inversamente proporcional a n_pos** — para n_pos alto, a serialização de nf impacta pouco porque o balanceamento de carga já é bom. Para n_pos baixo, o impacto é máximo.

**Melhoria proposta**: `O2 — FLAT prange(n_pos × nf)` (§8.2).

---

### Cenário 3 — `nf=1, nAng>1, nTR=1` (Multi-Ângulo)

**Descrição física**: Frequência única, múltiplos ângulos de inclinação. Uso: inversão multi-dip para análise de anisotropia ou geração de dados de treinamento com dip variável.

**Exemplo**: `dip_degs=[0, 15, 30, 45, 60, 75, 90]` (nAng=7), nf=1, nTR=1, n_pos=300.

#### Impacto do Ângulo no Cache

Para ângulo de inclinação `dip ≠ 0°`, o deslocamento horizontal TX-RX muda:

```python
hordist = tr_spacing × sin(dip_rad)

dip=0°:  hordist = 1.0 × sin(0°) = 0.0
dip=30°: hordist = 1.0 × sin(30°) = 0.5
dip=45°: hordist = 1.0 × sin(45°) = 0.707
dip=90°: hordist = 1.0 × sin(90°) = 1.0
```

**Cada ângulo gera um cache distinto** (hordist diferente → `u_cache` diferente). Para nAng=7 ângulos distintos: **7 caches** são pré-computados.

#### Caminho de Execução

```
simulate_multi(nf=1, nAng=7, nTR=1, n_pos=300)
  ↓
  _build_unique_hordist_caches(tr_spacings=[1.0], dip_rads=[0,0.26,0.52,...])
    → unique_hordist: [0.0, 0.259, 0.5, 0.707, 0.866, 0.966, 1.0]  (7 únicos)

  precompute_common_arrays_cache × 7 (um por hordist único)
    cada: prange(nf=1) → 1 iteração
    → 7 × u_cache: (1, 201, n_layers) por hordist

  ↓
  _simulate_combined_prange(n_combos=7, n_pos=300)
    └─ for k in prange(7 × 300 = 2100):   ← 2100 tarefas ✓
         i_combo = k // 300  (ângulo)
         j = k % 300         (posição)
         cache_idx = cache_indices[i_combo]
         _fields_in_freqs_kernel_cached(u_cache[cache_idx], ...)
```

#### Análise por Regime de `n_pos` com `nAng=7`

| Regime | `n_pos` | Tarefas prange | Tarefas/thread | Eficiência | Obs. |
|:------:|:-------:|:--------------:|:--------------:|:----------:|:-----|
| Baixo | 20 | 140 | 17.5 | **Boa** (85%) | `nAng` compensa `n_pos` baixo |
| Médio | 100 | 700 | 87.5 | **Ótima** (93%) | Bem saturado |
| Alto | 300 | 2100 | 262.5 | **Excelente** (97%) | |
| Alto | 600 | 4200 | 525 | **Excelente** (98%) | |

**Status**: **JÁ OTIMIZADO** via `_simulate_combined_prange`. O `nAng` atua como multiplicador de granularidade, tornando este cenário robusto mesmo para n_pos baixo.

**Observação importante**: Para nAng alto (≥8) com n_pos baixo, o Cenário 3 pode ter **melhor eficiência que o Cenário 2** com o mesmo n_pos × nAng, porque o FLAT prange(nAng × n_pos) já está implementado.

---

### Cenário 4 — `nf=1, nAng=1, nTR>1` (Multi-TR)

**Descrição física**: Frequência única, poço vertical, múltiplos pares TR. Diferentes espaçamentos TX-RX investigam diferentes profundidades radiais na formação (shallow/medium/deep DoI).

**Exemplo**: `tr_spacings_m=[0.5, 1.0, 1.5, 2.0]` (nTR=4), nf=1, nAng=1, n_pos=600.

#### Deduplicação de Cache por `hordist`

Para dip=0°, `hordist = tr_spacing × sin(0°) = 0` independente do espaçamento TR. Portanto:

```
nTR=4, dip=0°: unique_hordist = {0.0}  → apenas 1 cache!
Todos os 4 pares TR compartilham o mesmo u_cache.

nTR=4, dip=30°: hordist_i = tr_spacings_m[i] × 0.5
  → unique_hordist = {0.25, 0.5, 0.75, 1.0}  → 4 caches distintos
```

Este comportamento é **correto fisicamente**: para poço vertical, a geometria de propagação das ondas EM é a mesma independente do espaçamento, apenas a distância muda.

#### Caminho de Execução

```
simulate_multi(nf=1, nAng=1, nTR=4, n_pos=600, dip=0°)
  ↓
  _build_unique_hordist_caches → 1 cache único (hordist=0)
  precompute_common_arrays_cache × 1 (apenas!)

  ↓
  _simulate_combined_prange(n_combos=4, n_pos=600)
    └─ for k in prange(4 × 600 = 2400):   ← 2400 tarefas ✓
         i_combo = k // 600  (0, 1, 2, ou 3 → TR spacing)
         j = k % 600         (posição)
         cache_idx = 0        (todos compartilham o cache)
         _fields_in_freqs_kernel_cached(u_cache[0], ...)
           Tx = tr_spacings_m[i_combo] / 2     ← geometria difere por TR!
           cx = -tr_spacings_m[i_combo] / 2
```

**Status**: **JÁ OTIMIZADO**. Mesmo análise que Cenário 3. O FLAT prange elimina o loop Python sobre `nTR`.

---

### Cenário 5 — `nf=1, nAng>1, nTR>1` (Multi-ângulo + Multi-TR)

**Descrição física**: Combinação total de múltiplos ângulos e múltiplos espaçamentos TR, frequência única. Uso: array 3D de inversão — ferramenta tipo Schlumberger Periscope-15.

**Exemplo**: nTR=4, nAng=8, nf=1, n_pos=600.

#### Análise de Escala

```
n_combos = nTR × nAng = 4 × 8 = 32
Tarefas prange = 32 × 600 = 19.200

Com 8 threads: 2.400 tarefas/thread → saturação excelente.

Cache: unique_hordist varia com (TR, ângulo):
  Para dip_degs=[0,15,30,45,60,75,82,90], tr_spacings_m=[0.5,1.0,1.5,2.0]:
  → até 32 caches únicos (hordist = tr × sin(dip))
  → Mas alguns podem coincidir (ex: tr=0.5,dip=90° vs tr=1.0,dip=30° se mesma hordist)
  → Cache dedup economiza pré-cômputo
```

#### Análise por Regime de `n_pos`

| Regime | `n_pos` | n_combos | Tarefas total | Tarefas/thread | Eficiência |
|:------:|:-------:|:--------:|:-------------:|:--------------:|:----------:|
| Baixo | 20 | 32 | 640 | 80 | **Ótima** (93%) |
| Baixo | 30 | 32 | 960 | 120 | **Excelente** (95%) |
| Médio | 150 | 32 | 4800 | 600 | **Excelente** (97%) |
| Alto | 600 | 32 | 19200 | 2400 | **Excelente** (98%) |

**Status**: **JÁ OTIMIZADO**. Cenário 5 é o mais robusto de todos para n_pos baixo. `nTR × nAng` garante abundância de tarefas em qualquer regime de n_pos.

---

### Cenário 6 — `nf>1, nAng>1, nTR=1` (Multi-Freq + Multi-Ângulo)

**Descrição física**: Ferramenta de frequência variável operando em múltiplos ângulos de inclinação. Uso: ARC com 2–4 frequências para análise de anisotropia vertical com inversão multi-dip.

**Exemplo**: `frequencies_hz=[400e3, 2e6]` (nf=2), `dip_degs=[0,15,30,45]` (nAng=4), nTR=1, n_pos=300.

#### Caminho de Execução Atual

```
simulate_multi(nf=2, nAng=4, nTR=1, n_pos=300)
  ↓
  precompute_common_arrays_cache × 4 (um por hordist único de cada ângulo)
    cada: prange(nf=2) PARALELO ✓
    → 4 × u_cache: (2, 201, n_layers)

  ↓
  _simulate_combined_prange(n_combos=4, n_pos=300)
    └─ for k in prange(4 × 300 = 1200):
         _fields_in_freqs_kernel_cached(...)
           └─ for i_f in range(2):  ← SERIAL ⚠
                hmd_tiv(u_cache[ci, i_f], ...)
                vmd(...)
```

#### Comparação: Atual vs FLAT Proposto

```
ATUAL:
  Tarefas prange: 1200
  Trabalho por tarefa: 2 freq × (hmd_tiv + vmd) = 2× custo base
  Overhead: cada tarefa carrega 2× mais trabalho → menor balanceamento
  Eficiência: 1200/8 = 150 tarefas/thread (mas cada 2× mais pesada)

FLAT prange(nf × n_combos × n_pos) = prange(2 × 4 × 300 = 2400):
  Tarefas: 2400
  Trabalho por tarefa: 1 freq × (hmd_tiv + vmd) = 1× custo base
  Balanceamento: 2400/8 = 300 tarefas/thread
  Cada tarefa é independente (i_f, i_combo, j são decompostos do flat index)
```

**Ganho estimado com FLAT**: ~1.5–2× para n_pos=300, nf=2.

---

### Cenário 7 — `nf>1, nAng=1, nTR>1` (Multi-Freq + Multi-TR)

**Descrição física**: Array de TR com múltiplas frequências, poço vertical. Caso realístico de ferramenta Schlumberger ARC 6/8-phase com múltiplos espaçamentos.

**Exemplo**: `frequencies_hz=[400e3, 1e6, 2e6]` (nf=3), `tr_spacings_m=[0.5, 1.0, 2.0]` (nTR=3), nAng=1, n_pos=600.

#### Análise de Caches

```
dip=0°: hordist = 0 para todos TR
  → 1 cache único para todos (3 × 3 = 9 combos)

dip=30°: hordist = tr × sin(30°) = tr × 0.5
  → [0.25, 0.5, 1.0] → 3 caches únicos

precompute_common_arrays_cache:
  Para cada hordist único: prange(nf=3) → 3 iterações paralelas
  Custo total de pré-cômputo: n_unique × O(npt × n_layers × nf / 3)
```

#### Análise por Regime de `n_pos` com `nf=3, nTR=3`

| Regime | `n_pos` | Tarefas atual | Tarefas FLAT | Ganho | Eficiência atual |
|:------:|:-------:|:-------------:|:------------:|:-----:|:----------------:|
| Baixo | 30 | 90 | 270 | **2.5–3×** | 65% |
| Médio | 150 | 450 | 1350 | **1.8–2×** | 80% |
| Alto | 600 | 1800 | 5400 | **1.3–1.5×** | 90% |

---

### Cenário 8 — `nf>1, nAng>1, nTR>1` (Máxima Complexidade)

**Descrição física**: Ferramenta de array completa — multi-TR, multi-ângulo, multi-frequência. Caso de uso: inversão de ensemble em tempo real com ferramenta tipo Periscope-15.

**Exemplo referência**: nTR=4, nAng=8, nf=4, n_pos=600 → `n_total` por modelo:
```
Chamadas de kernel = nTR × nAng × n_pos × nf = 4 × 8 × 600 × 4 = 76.800
```

#### Estrutura de Dados

```
H_tensor: (nTR=4, nAng=8, n_pos=600, nf=4, 9)
  Total de complex128: 4 × 8 × 600 × 4 × 9 = 691.200 complexos
  Memória: 691.200 × 16 bytes = 10.7 MB por modelo
  Para batch de 30.000 modelos: 321 GB → requer streaming!
```

#### Caminho de Execução Atual

```
simulate_multi(nf=4, nAng=8, nTR=4, n_pos=600)
  ↓
  n_combos = nTR × nAng = 4 × 8 = 32

  precompute: até 32 caches únicos (hordist = tr × sin(dip))
    cada: prange(nf=4) paralelo ✓

  _simulate_combined_prange(n_combos=32, n_pos=600)
    └─ for k in prange(32 × 600 = 19.200):
         _fields_in_freqs_kernel_cached(...)
           └─ for i_f in range(4):  ← SERIAL ⚠
                hmd_tiv × 4 + vmd × 4  ← 8 integrações de Hankel por posição
```

#### Escalabilidade com FLAT prange

```
ATUAL: prange(19.200) × 4 freq serial
  Trabalho efetivo por thread: 19.200/8 × 4 = 9.600 × trabalho_base

FLAT: prange(4 × 32 × 600 = 76.800)
  Trabalho efetivo por thread: 76.800/8 = 9.600 × trabalho_base

Teoricamente idêntico! Mas na prática, FLAT é melhor por:
  1. Eliminação de tarefas pesadas desiguais (uma tarefa com 4 freq vs 4 tarefas com 1 freq)
  2. Melhor aproveitamento do work-stealing do scheduler Numba
  3. Menor variância no tempo de conclusão das threads

Ganho esperado: 15–40% de redução em desvio padrão de tempo de simulação.
```

---

## 6. Análise Detalhada — Cenários Avançados (F6, F7, Batch)

### 6.1 F6 — Compensação Midpoint CDR

**Física**: A compensação CDR (Compensated Deep Resistivity) cancela efeitos de dip e borehole usando pares de medições TX/RX simétricas.

**Condição**: `use_compensation=True`, `len(tr_spacings_m) ≥ 2`, `comp_pairs` especificado.

#### Fluxo de Execução F6

```
simulate_multi(..., use_compensation=True, comp_pairs=((0,2), (1,3)))
  ↓
  [execução normal: Cenário 4, 5, 7, ou 8]
  H_tensor: (nTR, nAng, n_pos, nf, 9)
  ↓
  apply_compensation(H_tensor, comp_pairs=((0,2),(1,3)))
    → Para cada par (i_near, i_far):
       H_comp[k] = (H_tensor[i_near] + H_tensor[i_far]) / 2
    → H_comp: (n_comp_pairs, nAng, n_pos, nf) complex128
    → phase_diff_deg: (n_comp_pairs, nAng, n_pos, nf) float64
    → atten_db: (n_comp_pairs, nAng, n_pos, nf) float64
```

**Custo adicional de F6**: O(n_comp_pairs × nAng × n_pos × nf) — operações de array NumPy puro, muito rápidas.

**Performance**: F6 adiciona <5% de overhead ao Cenário 4/5/7/8. Não é gargalo.

**Combinações com F6**:
- **F6 + Cenário 4** (1f, 1a, nTR≥2): Caso simples, típico de compensação LWD
- **F6 + Cenário 7** (nf, 1a, nTR≥2): Compensação multi-frequência
- **F6 + Cenário 8** (nf, nAng, nTR≥2): Compensação completa multi-dim

### 6.2 F7 — Antenas Inclinadas

**Física**: Simula ferramentas com antenas transmissoras ou receptoras inclinadas em ângulo `β` em relação ao eixo do poço, com azimute `φ`. A resposta é uma combinação linear do tensor completo.

**Equação**:
```
H_tilted(β, φ) = cos(β)·H_zz + sin(β)·[cos(φ)·H_xz + sin(φ)·H_yz]
```

**Condição**: `use_tilted=True`, `tilted_configs=[(β₁,φ₁), (β₂,φ₂), ...]`

#### Fluxo de Execução F7

```
simulate_multi(..., use_tilted=True, tilted_configs=[(45,0),(45,90)])
  ↓
  [execução normal qualquer cenário]
  H_tensor: (nTR, nAng, n_pos, nf, 9)
  ↓
  apply_tilted_antennas(H_tensor, tilted_configs=[(45,0),(45,90)])
    → Para cada config (β, φ):
       H_tilted[k] = cos(β)·H[...,8] + sin(β)·(cos(φ)·H[...,6] + sin(φ)·H[...,7])
    → H_tilted: (n_tilted=2, nTR, nAng, n_pos, nf) complex128
```

**Custo adicional de F7**: O(n_tilted × nTR × nAng × n_pos × nf) — combinação linear de componentes do tensor, muito barata.

**Performance**: F7 adiciona <2% de overhead. Não é gargalo.

**Oportunidade**: Para n_tilted alto (>10 configurações), uma versão vetorizada de `apply_tilted_antennas` com `np.einsum` poderia ser ~3× mais rápida que o loop Python atual.

### 6.3 Batch — Modelos Geológicos em Paralelo

**Descrição**: Execução de N modelos geológicos distintos usando `ProcessPoolExecutor`. Principal modo de uso para geração de dados de treinamento.

#### Arquitetura do Batch

```
simulate_multi(models=[m0, m1, ..., m_{N-1}], n_workers=4, threads_per_worker=2)
  ↓
  _workers.run_batch(models, n_workers=4, threads_per_worker=2, ...)
    ↓
    chunks = _split_models_uniform(models, n_workers=4)
      chunk_0: models[0:7500]
      chunk_1: models[7500:15000]
      chunk_2: models[15000:22500]
      chunk_3: models[22500:30000]
    ↓
    pool = _acquire_pool(n_workers=4, threads_per_worker=2, ...)
      → ProcessPoolExecutor(max_workers=4)
      → Cada worker: os.environ["NUMBA_NUM_THREADS"] = "2"
    ↓
    futures = [pool.submit(_simulate_worker, chunk_i, ...) for i in range(4)]
    ↓
    results = list(as_completed(futures))
    → _order_results(results, original_order)
    → MultiSimulationResultBatch(H_stack, throughput_mod_per_h)
```

#### Configuração Ótima (v2.17, empiricamente confirmada v2.20)

```
Hardware: Apple M-series 8C/16T
  n_workers = phys_cores // 2 = 4
  threads_per_worker = 2
  Total threads = 4 × 2 = 8 = phys_cores ← sem oversubscription

Benchmark v2.21 (Cenário E, 600 pts, 300 modelos/run):
  4w × 2t: ~122.000 mod/h (mediana 5 runs) ✓
  4w × 4t: ~38.000 mod/h (-69%, oversubscription) ✗
  Confirmação empírica: phys_cores é a estratégia correta para Numba HPC
```

#### Estratégia de Split por Complexidade de Cenário

```
Cenário A (n_pos=30, nf=1):     Custo por modelo ∝ 30 × 1   = 30
Cenário E (n_pos=600, nf=1):    Custo por modelo ∝ 600 × 1  = 600
Cenário F (n_pos=600, nf=4):    Custo por modelo ∝ 600 × 4  = 2400
Cenário G (n_pos=600, nf=10):   Custo por modelo ∝ 600 × 10 = 6000

O split uniforme (_split_models_uniform) é CORRETO quando todos os modelos
têm a mesma complexidade (mesma formação, mesmo cenário). Para lotes
heterogêneos (mistura de modelos simples e complexos), um split adaptativo
baseado no custo estimado seria ideal — mas não é necessário no uso atual.
```

---

## 7. Alta Resistividade (ρ > 1000 Ω·m)

Esta seção documenta os desafios numéricos específicos de alta resistividade e as garantias do código atual.

### 7.1 Física dos Modos EM em Alta Resistividade

Em meios de baixa condutividade (alta resistividade), as ondas EM se comportam como campos quase-estáticos:

```
Constante de propagação em camada i:
  k²ᵢ = ω²μ₀ε₀ - iωμ₀σₕ,ᵢ

Para σₕ,ᵢ → 0 (ρ → ∞):
  k²ᵢ → ω²μ₀ε₀ (apenas parte real, modo propagativo puro)

Variável de integração de Hankel:
  uᵢ = √(kr² - k²ᵢ) ≈ √(kr² - ε₀) ≈ kr  para kr >> ε₀

Exponencial de propagação:
  exp(uᵢ × Δzᵢ) ≈ exp(kr × Δzᵢ)

  Para kr = 10 (modo evanescente de alta freq), Δz = 100m:
    exp(10 × 100) = exp(1000) → OVERFLOW float64 (max ~e⁷⁰⁰)
```

### 7.2 Estabilidade Numérica: Técnica de Recursão Estável

O código Python herda do Fortran a **recursão P-matrix estável** (também chamada de "recursão propagadora normalizada"), que evita overflow dividindo por exponenciais antes de multiplicar:

```python
# propagation.py — Análise da recursão:

# Sentido descendente (top → bottom):
for i in range(n-2, -1, -1):
    # RTEdw[i] é calculado recursivamente
    # usando exp(u[i+1] × h[i+1]) normalizado
    # → não há acumulação de overflow porque
    #    os coeficientes de reflexão são |R| ≤ 1

# Sentido ascendente (bottom → top):
for i in range(1, n):
    # RTEup[i] análogo — também |R| ≤ 1
```

**Garantia**: A recursão de coeficientes de reflexão (RTEdw, RTEup, RTMdw, RTMup) é condicionalmente estável para qualquer valor de resistividade, pois os coeficientes de reflexão satisfazem `|R| ≤ 1` (lei de conservação de energia).

### 7.3 Testes com Alta Resistividade — Estado Atual

| Modelo canônico | ρ_max | Status testes |
|:----------------|:-----:|:-------------:|
| oklahoma_3 | ~100 Ω·m | ✓ <1e-12 |
| oklahoma_5 | ~200 Ω·m | ✓ <1e-12 |
| devine_8 | ~50 Ω·m | ✓ <1e-12 |
| oklahoma_15 | ~150 Ω·m | ✓ <1e-12 |
| oklahoma_28 | ~300 Ω·m | ✓ <1e-12 |
| hou_7 | ~80 Ω·m | ✓ <1e-12 |
| viking_graben_10 | ~500 Ω·m | ✓ <1e-12 |

**Gap identificado**: Nenhum modelo canônico atual testa ρ > 500 Ω·m. Carbonatos secos (ρ = 2000–10000 Ω·m) e evaporitas (ρ = 10000–100000 Ω·m) não estão cobertos.

### 7.4 Modelos Canônicos Propostos para Alta Resistividade

```python
# Modelo "carbonato_seco_5c" — 5 camadas, pico 5000 Ω·m
rho_h_carbonato = np.array([2.0, 50.0, 5000.0, 50.0, 2.0])   # Ω·m
rho_v_carbonato = rho_h_carbonato * 2.0                        # anisotropia 2:1
esp_carbonato   = np.array([10.0, 5.0, 20.0, 5.0])            # metros (n-1 esps)

# Modelo "evaporita_3c" — 3 camadas, pico 100000 Ω·m
rho_h_evaporita = np.array([1.5, 1e5, 1.5])                   # sal halita ~100kΩ·m
rho_v_evaporita = rho_h_evaporita                              # sal isotrópico
esp_evaporita   = np.array([5.0, 50.0])

# Modelo "gas_seco_8c" — 8 camadas, pico 10000 Ω·m
rho_h_gas = np.array([2.0, 10.0, 200.0, 1500.0, 10000.0, 1500.0, 10.0, 2.0])
rho_v_gas = rho_h_gas * 1.5
esp_gas   = np.array([5.0, 3.0, 2.0, 1.0, 2.0, 3.0, 5.0])
```

### 7.5 Comportamento Esperado do Filtro Werthmuller 201pt em Alta Resistividade

```
Para ρ = 100 Ω·m:   sinal H ≈ 1e-9 A/m    → SNR vs filtro ~1e14 → OK
Para ρ = 1000 Ω·m:  sinal H ≈ 1e-11 A/m   → SNR vs filtro ~1e12 → OK
Para ρ = 10000 Ω·m: sinal H ≈ 1e-13 A/m   → SNR vs filtro ~1e10 → marginal
Para ρ = 100000 Ω·m: sinal H ≈ 1e-15 A/m  → SNR vs filtro ~1e8  → pode precisar Anderson 801pt

Recomendação:
  ρ ≤ 10000 Ω·m: Werthmuller 201pt é adequado (npt=201)
  ρ > 10000 Ω·m: Usar Anderson 801pt (npt=801) para máxima precisão
```

### 7.6 Implicação para `fastmath=True`

Para alta resistividade com `fastmath=True` em `hmd_tiv` e `vmd`:

```
Risco principal: Cancelamento catastrófico em somas de Hankel
  Ocorre quando: termos positivos e negativos de magnitude similar se cancelam

  Para ρ baixo:    |H_TE| >> |H_TM|  → sem cancelamento, fastmath SEGURO
  Para ρ médio:    |H_TE| ~ |H_TM|   → possível cancelamento PARCIAL
  Para ρ alto:     |H_TE| ≈ |H_TM|   → cancelamento SIGNIFICATIVO possível

Critério de aceitação para fastmath:
  Gate: paridade Fortran mantida <1e-12 nos 7 modelos canônicos ATUAIS
  + nos 3 novos modelos de alta resistividade propostos acima

Se paridade >1e-12 para qualquer modelo de alta ρ com fastmath=True:
  → fastmath=False para hmd_tiv/vmd (manter atual)
  → fastmath=True apenas para hankel.py (já implementado, seguro)
```

---

## 8. Estratégias de Otimização — Catálogo Completo

### 8.1 O1 — Adaptive Thread Count para n_pos Baixo

**Problema**: Para n_pos × nAng × nTR < num_threads, há threads ociosas e o overhead de inicialização do scheduler Numba domina.

**Análise do overhead**:
```
Overhead de fork/join Numba prange: ~50–150μs por invocação (medido empiricamente)
Trabalho por ponto (n_layers=22, npt=201): ~200–500μs

Para n_pos=8 (< 8 threads), serial pode ser mais rápido:
  Paralelo: 150μs overhead + 1 × 400μs trabalho / 1 thread ≈ 550μs
  Serial:   0μs overhead   + 8 × 400μs trabalho      ≈ 3.2ms
  → Paralelo compensa mesmo para n_pos=8 (3.2ms / 8 threads = 0.4ms)

Para n_pos=2 (extremamente baixo):
  Paralelo: 150μs overhead + (400μs / 8 threads) ≈ 200μs
  Serial:   0μs overhead   + 2 × 400μs = 800μs
  → Paralelo ainda compensa!

Conclusão: prange é geralmente vantajoso mesmo para n_pos pequeno,
MAS a granularidade sub-ótima pode causar desbalanceamento.
```

**Implementação proposta**:
```python
# Em simulate_multi(), antes do despacho:
effective_tasks = n_combos * n_pos * nf
num_threads = numba.get_num_threads()
optimal_threads = min(num_threads, max(1, effective_tasks // 4))
# Pelo menos 4 tarefas/thread para amortizar overhead

if optimal_threads != num_threads:
    numba.set_num_threads(optimal_threads)
    try:
        result = _simulate_combined_prange(...)
    finally:
        numba.set_num_threads(num_threads)  # Restaurar
```

**Cenários beneficiados**: 1, 2 (n_pos baixo), 3 (n_pos muito baixo), 4 (n_pos baixo).

**Ganho estimado**: 10–40% para n_pos < 50 com nf=1.

**Risco**: Overhead de `set_num_threads` (~1μs) — desprezível.

---

### 8.2 O2 — FLAT prange (nf × nTR × nAng × n_pos)

**Esta é a otimização mais impactante para cenários multi-frequência.**

**Problema**: `range(nf)` serial dentro de `_fields_in_freqs_kernel_cached` impede que as frequências sejam computadas em paralelo.

**Solução**: Novo kernel `_simulate_combined_prange_flat` que colapsa as 4 dimensões em um único `prange`.

#### Design Técnico do Novo Kernel

```python
# forward.py — NOVO kernel (proposta Sprint 22.1)

@njit(parallel=True, cache=True, nogil=True)
def _simulate_combined_prange_flat(
    # Dimensões
    positions_z:   np.ndarray,  # (n_pos,) float64
    dz_halfs:      np.ndarray,  # (n_combos,) float64
    r_halfs:       np.ndarray,  # (n_combos,) float64
    dip_rads_flat: np.ndarray,  # (n_combos,) float64
    cache_indices: np.ndarray,  # (n_combos,) int64
    freqs_hz:      np.ndarray,  # (nf,) float64
    # Geometria da formação
    n:             int,
    prof_arr:      np.ndarray,  # (n+1,)
    rho_h:         np.ndarray,  # (n,)
    rho_v:         np.ndarray,  # (n,)
    h_arr:         np.ndarray,  # (n,)
    eta:           np.ndarray,  # (n, 2) complex
    # Cache por frequência
    u_unique:      np.ndarray,  # (n_unique, nf, npt, n) complex
    s_unique:      np.ndarray,  # idem
    uh_unique:     np.ndarray,  # idem
    sh_unique:     np.ndarray,  # idem
    RTEdw_unique:  np.ndarray,  # idem
    RTEup_unique:  np.ndarray,  # idem
    RTMdw_unique:  np.ndarray,  # idem
    RTMup_unique:  np.ndarray,  # idem
    AdmInt_unique: np.ndarray,  # idem
    # Filtro Hankel
    krJ0J1: np.ndarray,  # (npt,)
    wJ0:    np.ndarray,  # (npt,)
    wJ1:    np.ndarray,  # (npt,)
    # Saída
    H_tensor: np.ndarray,  # (n_combos, n_pos, nf, 9) complex128
) -> None:

    n_combos = dz_halfs.shape[0]
    n_pos    = positions_z.shape[0]
    nf       = freqs_hz.shape[0]
    n_total  = n_combos * n_pos * nf  # ← FLAT: todas as dimensões

    for k in _prange(n_total):       # ← ÚNICO prange, sem aninhamento
        # Decomposição do índice flat:
        i_combo = k // (n_pos * nf)
        rem     = k % (n_pos * nf)
        j       = rem // nf
        i_f     = rem % nf

        # Geometria (i_combo)
        dz_half = dz_halfs[i_combo]
        r_half  = r_halfs[i_combo]
        dip_rad = dip_rads_flat[i_combo]
        ci      = cache_indices[i_combo]

        # Posição (j)
        z_mid = positions_z[j]
        Tz = z_mid + dz_half;   cz = z_mid - dz_half
        Tx = r_half;             cx = -r_half
        Ty = np.float64(0.0);   cy = np.float64(0.0)

        # Frequência (i_f) — AGORA PARALELO!
        # find_layers_tr é O(log n) — custo negligível vs hmd_tiv
        camad_t, camad_r = find_layers_tr(n, Tz, cz, prof_arr)

        Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz = common_factors(
            camad_t, camad_r, Tz, cz,
            u_unique[ci, i_f], s_unique[ci, i_f],
            uh_unique[ci, i_f], sh_unique[ci, i_f],
            RTEdw_unique[ci, i_f], RTEup_unique[ci, i_f],
            RTMdw_unique[ci, i_f], RTMup_unique[ci, i_f],
            AdmInt_unique[ci, i_f],
        )

        freq = freqs_hz[i_f]
        zeta = np.complex128(0.0 + 1j) * 2.0 * np.pi * freq * 4.0e-7 * np.pi

        Hx_hmd, Hy_hmd, Hz_hmd = hmd_tiv(
            Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz,
            Tx, Ty, Tz, cx, cy, cz,
            u_unique[ci, i_f], s_unique[ci, i_f],
            AdmInt_unique[ci, i_f],
            RTEdw_unique[ci, i_f], RTEup_unique[ci, i_f],
            krJ0J1, wJ0, wJ1, n, camad_t, camad_r, zeta,
        )
        Hx_vmd, Hy_vmd, Hz_vmd = vmd(
            Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz,
            Tx, Ty, Tz, cx, cy, cz,
            u_unique[ci, i_f], s_unique[ci, i_f],
            AdmInt_unique[ci, i_f],
            RTEdw_unique[ci, i_f], RTEup_unique[ci, i_f],
            krJ0J1, wJ0, wJ1, n, camad_t, camad_r, zeta,
        )

        matH = (
            (Hx_hmd[0], Hy_hmd[0], Hz_hmd[0]),  # dipolo x
            (Hx_hmd[1], Hy_hmd[1], Hz_hmd[1]),  # dipolo y
            (Hx_vmd,    Hy_vmd,    Hz_vmd),      # dipolo z
        )
        tH = rotate_tensor(dip_rad, np.float64(0.0), np.float64(0.0), matH)

        H_tensor[i_combo, j, i_f, 0] = tH[0][0]
        H_tensor[i_combo, j, i_f, 1] = tH[0][1]
        # ... (9 componentes)
```

#### Análise de Correção do FLAT prange

```
Questão: find_layers_tr(Tz, cz, prof_arr) é computado nf vezes para
         a mesma posição (j) — é redundante?

Resposta: Sim, é redundante. find_layers_tr é O(log n) = O(4) para n=22.
         Custo de find_layers_tr: ~50ns
         Custo de hmd_tiv+vmd:    ~200μs
         Overhead = 50ns/200μs = 0.025%  ← desprezível!

Alternativa para eliminar redundância:
  Usar decomposição k → (i_combo, j) em outer loop e (i_f) em inner loop,
  mas isso reintroduz aninhamento. O trade-off favorece FLAT.
```

#### Profiling Teórico de Ganho (8 cores, n_layers=22, npt=201)

```
Tempo de hmd_tiv + vmd por (posição, frequência): T_kern

Cenário 2 (nf=4, n_pos=600) ATUAL:
  Tarefas paralelas: 600
  Trabalho por tarefa: 4 × T_kern
  Tempo total: (600/8) × 4 × T_kern = 300 × T_kern

Cenário 2 (nf=4, n_pos=600) FLAT:
  Tarefas paralelas: 2400
  Trabalho por tarefa: 1 × T_kern
  Tempo total: (2400/8) × T_kern = 300 × T_kern

Teoricamente idêntico! O ganho REAL vem de:
  1. Melhor balanceamento quando nf não é múltiplo de n_threads
  2. Redução de desbalanceamento tail-latency (última tarefa pesada)
  3. Melhor para n_pos BAIXO: prange(30×4=120) vs prange(30) com 4 freq serial

Para n_pos=30, nf=4:
  ATUAL:  30/8 × 4 × T = 15 × T  (mas 2 threads ficam ociosas no final!)
  FLAT:  120/8 × T = 15 × T      (todas as threads trabalham igualmente)

  Benefício real com n_pos baixo: 1.5–3× (eliminação de threads ociosas)
```

---

### 8.3 O3 — `fastmath=True` em `hmd_tiv` e `vmd`

**Contexto**: As funções `hmd_tiv` e `vmd` realizam somas de Hankel sobre `npt=201` pontos (Werthmuller). Estas somas são candidatas para `fastmath=True`, que habilita FMA (Fused Multiply-Add) e reordenamento de operações de ponto flutuante.

**Análise de segurança por regime de resistividade**:

```
Operações críticas em hmd_tiv:
  sum_Ktedz_J1 = Σ_{i=0}^{200} Ktedz_J1[i]     (201 somas)
  sum_Ktm_J1   = Σ_{i=0}^{200} Ktm_J1[i]        (201 somas)
  # etc.

Para ρ = 1–100 Ω·m:
  Termos Ktedz são de magnitude similar → somas bem condicionadas
  fastmath SEGURO ✓

Para ρ = 100–1000 Ω·m:
  Possível espalhamento de magnitude entre termos
  fastmath PROVAVELMENTE SEGURO ✓ (validar com gate <1e-12)

Para ρ > 1000 Ω·m:
  Termos TE e TM podem ter cancelamento parcial
  fastmath REQUER VALIDAÇÃO CUIDADOSA

Gate obrigatório: rodar paridade Fortran em 7 modelos canônicos +
                  3 modelos de alta resistividade propostos
```

**Implementação sugerida** (dual-mode seguro):
```python
# dipoles.py — proposta
@njit(fastmath=False)   # versão precisa (default, produção atual)
def hmd_tiv_precise(...):
    ...

@njit(fastmath=True)    # versão rápida (opt-in via SimulationConfig)
def hmd_tiv_fast(...):
    ...

# Em SimulationConfig:
class SimulationConfig:
    use_fastmath: bool = False  # default seguro
    # True: +8-15% velocidade, aceita ε~1e-10 vs Fortran
    # False: <1e-12 vs Fortran (invariante física)
```

**Ganho estimado**:
- `hmd_tiv` + `vmd` representam ~70% do custo total do kernel
- `fastmath` típico: 8–15% por função
- **Ganho global estimado**: ~5–10%

---

### 8.4 O4 — Cache de `precompute_common_arrays_cache` por Contexto de Simulação

**Problema**: A cada chamada `simulate_multi()`, o cache é recomputado do zero. Para simulações repetidas com a mesma geometria (mesmo modelo geológico, diferentes posições), o cache poderia ser reutilizado.

**Análise**:
```
O cache depende de: (hordist, freq, n_layers, rho_h, rho_v, esp, hankel_filter)

Para batch de N modelos, todos com formações distintas:
  → N caches diferentes → não reutilizável ✗

Para inversão iterativa (mesmo modelo, atualiza posições):
  → Cache idêntico em todas as iterações → REUTILIZÁVEL ✓
  Uso: geosteering em tempo real, inversão Newton-Raphson

Implementação:
  _CACHE_STORE: Dict[CacheKey, CachedArrays] = {}  (LRU)
  CacheKey = hash(hordist, freq_tuple, rho_h, rho_v, esp, filter_name)

  → Evita recompute em simulações consecutivas com mesmo modelo
  → Benefício em geosteering em tempo real: até 30% de speedup
```

**Cenários beneficiados**: Inversão em tempo real, geosteering Look-Ahead.

---

### 8.5 O5 — Exposição Clara do Filtro Kong 61pt para Treinamento

**Contexto**: Para geração de dados de treinamento (~30.000 modelos), precisão <1e-10 é suficiente (erro de generalização da rede >> 1e-10). O filtro Kong 61pt é 3.3× mais rápido que Werthmuller 201pt.

**Werthmuller 201pt** (padrão deste documento):
- `npt=201`
- Precisão: ε~1e-14 vs Fortran
- Custo por chamada de `hmd_tiv`: O(201 × n_layers)
- Uso: Validação física, inversão em produção, ground truth

**Kong 61pt** (alternativa para treinamento):
- `npt=61`
- Precisão: ε~1e-10 vs Fortran (5 ordens melhor que ruído de medição LWD)
- Custo por chamada de `hmd_tiv`: O(61 × n_layers) ← 3.3× menor
- Uso: Geração de datasets de treinamento (30k-100k modelos)

**Impacto na throughput de treinamento** (Cenário E, 4w × 2t):
```
Werthmuller 201pt: ~122.000 mod/h (v2.21)
Kong 61pt:         ~122.000 × 3.3 ≈ 400.000 mod/h (estimado)

Para geração de 1.000.000 de modelos:
  Werthmuller: 1e6 / 122k ≈ 8.2 horas
  Kong 61pt:   1e6 / 400k ≈ 2.5 horas
  Economia: ~5.7 horas
```

**Implementação na GUI**: Adicionar seletor de filtro na página de parâmetros com contexto de uso recomendado.

---

### 8.6 O6 — Vetorização Avançada de F7 com `np.einsum`

**Problema**: `apply_tilted_antennas` usa um loop Python sobre `n_tilted_configs`.

**Proposta**:
```python
# ATUAL (loop Python):
for k, (beta, phi) in enumerate(tilted_configs):
    b = np.deg2rad(beta); p = np.deg2rad(phi)
    H_tilted[k] = (np.cos(b) * H_tensor[..., 8] +
                   np.sin(b) * (np.cos(p) * H_tensor[..., 6] +
                                np.sin(p) * H_tensor[..., 7]))

# PROPOSTO (einsum vetorizado):
betas  = np.deg2rad([cfg[0] for cfg in tilted_configs])  # (n_tilted,)
phis   = np.deg2rad([cfg[1] for cfg in tilted_configs])  # (n_tilted,)
W = np.stack([np.cos(betas),
              np.sin(betas)*np.cos(phis),
              np.sin(betas)*np.sin(phis)], axis=-1)  # (n_tilted, 3)
components = np.stack([H_tensor[..., 8],
                       H_tensor[..., 6],
                       H_tensor[..., 7]], axis=-1)   # (..., 3)
H_tilted = np.einsum('...j,kj->k...', components, W)  # (n_tilted, ...)
```

**Ganho**: ~3× para n_tilted > 5. Impacto global: <2% (F7 não é gargalo).

---

## 9. Roadmap de Implementação

### 9.1 Tabela de Prioridades

| Versão | Sprint | Otimização | Cenários | Ganho Estimado | Dificuldade | Tempo |
|:------:|:------:|:-----------|:--------:|:--------------:|:-----------:|:-----:|
| **v2.22** | 22.1 | **O2: FLAT prange(nf × n_combos × n_pos)** | 2, 6, 7, 8 | +2–4× (nf>1, n_pos baixo) | Média | 2–3 dias |
| **v2.22** | 22.2 | Benchmarks: Cenários F, G, H, I, J, K | Todos | — | Baixa | 4h |
| **v2.22** | 22.3 | Modelos canônicos alta ρ (carbonato, evaporita) | Alta ρ | — | Baixa | 3h |
| **v2.23** | 23.1 | **O3: fastmath em hmd_tiv + vmd** (com gate) | Todos | +5–10% | Média | 1 dia |
| **v2.23** | 23.2 | **O1: Adaptive thread count** para n_pos < 50 | 1–4 | +10–40% (baixo n_pos) | Média | 1 dia |
| **v2.24** | 24.1 | O5: Exposição Kong 61pt na GUI/CLI | Todos (treino) | +230% treinamento | Baixa | 4h |
| **v2.24** | 24.2 | Pré-cômputo Hankel TE/TM avançado | Todos | +10–15% | Alta | 3–5 dias |
| **v2.25** | 25.1 | O4: Cache de contexto para time-real | Inversão iterativa | +30% | Alta | 2 dias |
| **v2.25** | 25.2 | O6: F7 einsum vetorizado | F7 com n_tilted > 5 | +3× em F7 | Baixa | 2h |

### 9.2 Sequência Recomendada

```
1. v2.22 Sprint 22.1 — FLAT prange
   Pré-requisito: testes unitários do decompositor de índice flat
   Critério de aceite:
     • Cenário E (nf=1): sem regressão vs v2.21 (>120k mod/h)
     • Cenário F (nf=4, n_pos=600): ≥ 1.3× vs atual
     • Cenário L (nf=4, n_pos=30): ≥ 2.5× vs atual
     • Paridade Fortran: <1e-12 em 7 modelos canônicos

2. v2.22 Sprint 22.2 — Novos benchmarks
   Valida todas as otimizações anteriores e futuras

3. v2.22 Sprint 22.3 — Alta resistividade
   Valida robustez do simulador para uso industrial completo

4. v2.23 Sprint 23.1 — fastmath (opt-in)
   Requer gate em modelos de alta ρ antes de ativar

5. v2.23 Sprint 23.2 — Adaptive threads
   Simples, baixo risco

6. v2.24+ — Otimizações de longo prazo
   Dependem de validação das etapas anteriores
```

### 9.3 Projeção de Throughput Acumulado

```
Referência (v2.21, Cenário E, 4w×2t, nf=1):
  ~122.000 mod/h

Com v2.22 (FLAT prange, nf=1 sem mudança):
  ~122.000 mod/h (sem mudança para nf=1)

Com v2.22 Cenário F (nf=4, n_pos=600):
  ~160.000 mod/h (estimativa, +30% com FLAT)

Com v2.23 (fastmath, todos cenários):
  ~132.000 mod/h Cenário E (nf=1, +8%)
  ~175.000 mod/h Cenário F (nf=4, +8%)

Com v2.24 (pré-cômputo Hankel avançado, Cenário E):
  ~148.000 mod/h (+15% adicional)
  ~195.000 mod/h Cenário F

Projeção v2.25 (todos fixes acumulados, Cenário E):
  ~165.000 mod/h (meta: superar 150k/h consistentemente)
  ~220.000 mod/h Cenário F
```

---

## 10. Novos Benchmarks Necessários

### 10.1 Cenários de Benchmark Propostos

```python
# Adicionar a benchmarks/bench_v214_numba.py

SCENARIOS_EXTENDED = {
    # — Existentes —
    "A": {"n_pos": 30,  "nf": 1,  "nTR": 1, "nAng": 1,
          "desc": "Baseline (referência histórica)"},
    "E": {"n_pos": 600, "nf": 1,  "nTR": 1, "nAng": 1,
          "desc": "Produção LWD (meta: >120k mod/h)"},

    # — Multi-frequência —
    "F": {"n_pos": 600, "nf": 4,  "nTR": 1, "nAng": 1,
          "desc": "ARC 4-freq (avalia Cenário 2)"},
    "G": {"n_pos": 600, "nf": 10, "nTR": 1, "nAng": 1,
          "desc": "Multi-freq extremo (avalia Cenário 2 alto)"},
    "G2": {"n_pos": 30, "nf": 4,  "nTR": 1, "nAng": 1,
           "desc": "Cenário 2 com n_pos baixo (avalia O2 FLAT)"},

    # — Multi-TR/ângulo —
    "H": {"n_pos": 300, "nf": 1,  "nTR": 4, "nAng": 4,
          "desc": "Array 3D (avalia Cenário 5)"},

    # — Completos multi-dimensionais —
    "I": {"n_pos": 300, "nf": 4,  "nTR": 3, "nAng": 6,
          "desc": "Cenário 7: multi-freq+TR (avalia O2 FLAT)"},
    "J": {"n_pos": 600, "nf": 4,  "nTR": 4, "nAng": 8,
          "desc": "Cenário 8 completo (Periscope-like)"},

    # — Alta resistividade —
    "K_carb": {"n_pos": 600, "nf": 1, "nTR": 1, "nAng": 1,
               "rho_profile": "carbonato_seco",
               "desc": "Carbonato seco (rho_max=5000 Ω·m)"},
    "K_evap": {"n_pos": 600, "nf": 1, "nTR": 1, "nAng": 1,
               "rho_profile": "evaporita",
               "desc": "Evaporita/sal (rho_max=100000 Ω·m)"},

    # — Extremo n_pos —
    "X": {"n_pos": 10000, "nf": 1, "nTR": 1, "nAng": 1,
          "desc": "Perfil completo de poço (10km / 1m)"},
}
```

### 10.2 Métricas a Capturar

| Métrica | Unidade | Propósito |
|:--------|:-------:|:---------|
| `mod/h` | modelos/hora | Throughput principal |
| `s/modelo` | segundos | Latência single-model |
| `GFLOPS` | 10⁹ ops/s | Eficiência de hardware |
| `Δ_fortran` | adimensional | Paridade física (meta: <1e-12) |
| `threads_utilization` | % | % do tempo com todas as threads ativas |
| `cache_miss_rate` | % | Taxa de miss no precompute_common_arrays |

### 10.3 Comparação FLAT vs Atual (matriz completa)

```
Para cada cenário com nf > 1, capturar:
  t_atual  = tempo com _simulate_combined_prange atual (range(nf) serial)
  t_flat   = tempo com _simulate_combined_prange_flat (prange total)
  speedup  = t_atual / t_flat

Tabela esperada pós-v2.22:
╔══════╦════╦════════╦══════════════╦═══════════╦══════════════╗
║ Cen. ║ nf ║ n_pos  ║ t_atual (ms) ║ t_flat(ms)║ Speedup      ║
╠══════╬════╬════════╬══════════════╬═══════════╬══════════════╣
║  F   ║  4 ║ 600    ║ ~50          ║ ~35       ║ ~1.4×        ║
║  G2  ║  4 ║ 30     ║ ~10          ║ ~3.5      ║ ~2.9×        ║
║  I   ║  4 ║ 300    ║ ~130         ║ ~85       ║ ~1.5×        ║
║  J   ║  4 ║ 600    ║ ~250         ║ ~165      ║ ~1.5×        ║
╚══════╩════╩════════╩══════════════╩═══════════╩══════════════╝
(Valores estimados, a serem confirmados por medição)
```

---

## 11. Tabelas Consolidadas de Referência

### 11.1 Status de Paralelismo por Cenário

```
Legenda: ✅ Ótimo  ⚠️ Subótimo  ❌ Necessita atenção

╔════╦════╦═════╦═════╦══════════════╦═════════════════════════════════════╗
║ #  ║ nf ║ nAng║ nTR ║ n_pos baixo  ║ n_pos médio/alto                    ║
╠════╬════╬═════╬═════╬══════════════╬═════════════════════════════════════╣
║ 1  ║  1 ║  1  ║  1  ║ ⚠️ 55-65%   ║ ✅ 90-96%                          ║
║ 2  ║ >1 ║  1  ║  1  ║ ❌ 40-55%   ║ ⚠️ 80-85%  (freq serial no kernel)  ║
║ 3  ║  1 ║ >1  ║  1  ║ ✅ 80-90%   ║ ✅ 93-98%  (nAng amplifica tarefas) ║
║ 4  ║  1 ║  1  ║ >1  ║ ✅ 80-90%   ║ ✅ 93-98%  (nTR amplifica tarefas)  ║
║ 5  ║  1 ║ >1  ║ >1  ║ ✅ 90-95%   ║ ✅ 96-99%  (produto amplifica muito) ║
║ 6  ║ >1 ║ >1  ║  1  ║ ❌ 45-60%   ║ ⚠️ 82-88%  (nAng mitiga parcialmente)║
║ 7  ║ >1 ║  1  ║ >1  ║ ❌ 45-60%   ║ ⚠️ 82-88%  (nTR mitiga parcialmente) ║
║ 8  ║ >1 ║ >1  ║ >1  ║ ⚠️ 65-75%  ║ ⚠️ 86-92%  (mais tarefas, mas freq   ║
║    ║    ║     ║     ║              ║          serial ainda presente)       ║
╚════╩════╩═════╩═════╩══════════════╩═════════════════════════════════════╝
```

### 11.2 Matriz de Impacto das Otimizações

```
         O1      O2      O3      O4      O5      O6
         Adapt   FLAT    fmath   ctx$    Kong    F7-einsum
Cen. 1   ●●○○   ○○○○   ●●○○   ●●○○   ●●●●   ○○○○
Cen. 2   ●●●○   ●●●●   ●●●○   ●●●○   ●●●●   ○○○○
Cen. 3   ●○○○   ○○○○   ●●○○   ●●○○   ●●●●   ○○○○
Cen. 4   ●○○○   ○○○○   ●●○○   ●●○○   ●●●●   ○○○○
Cen. 5   ○○○○   ○○○○   ●●○○   ●●○○   ●●●●   ○○○○
Cen. 6   ●○○○   ●●●●   ●●●○   ●●●○   ●●●●   ○○○○
Cen. 7   ●○○○   ●●●●   ●●●○   ●●●○   ●●●●   ○○○○
Cen. 8   ○○○○   ●●●●   ●●●○   ●●●○   ●●●●   ○○○○
F6       ○○○○   ○○○○   ○○○○   ○○○○   ●●●●   ○○○○
F7       ○○○○   ○○○○   ○○○○   ○○○○   ●●●●   ●●●○
Batch    ○○○○   ●●○○   ●●●○   ●○○○   ●●●●   ○○○○

●●●● alto impacto  ●●●○ médio-alto  ●●○○ médio  ●○○○ baixo  ○○○○ sem impacto
```

### 11.3 Parâmetros de Desempenho por Filtro Hankel

| Filtro | `npt` | Precisão (vs Fortran) | Custo relativo | Uso recomendado |
|:-------|:-----:|:--------------------:|:--------------:|:----------------|
| Kong 61pt | 61 | ε~1e-10 | 1.0× (base) | Treinamento de rede (ε << erro ML) |
| Werthmuller 201pt | 201 | ε~1e-14 | **3.3×** | **Produção, inversão, ground truth** |
| Anderson 801pt | 801 | ε~1e-16 | 13.2× | Alta resistividade extrema (ρ > 10000 Ω·m) |

> **Padrão deste documento**: Werthmuller 201pt. Toda análise de cenários acima usa `npt=201`.

### 11.4 Configuração de Workers por Hardware

| Hardware | Cores físicos | Threads lógicos | Workers (rec.) | Threads/worker | Total threads |
|:---------|:-------------:|:---------------:|:--------------:|:--------------:|:-------------:|
| Mac M1/M2 8C | 8 | 8 (sem HT) | 4 | 2 | 8 |
| Mac M1 Pro 10C | 10 | 10 | 5 | 2 | 10 |
| Intel 12C/24T | 12 | 24 | 6 | 2 | 12 |
| AMD Ryzen 16C | 16 | 32 | 8 | 2 | 16 |
| Workstation 32C | 32 | 64 | 16 | 2 | 32 |

> **Regra geral (v2.17, confirmada empiricamente v2.20)**:
> `workers = phys_cores // 2`, `threads_per_worker = 2`.
> **Não usar logical_cores** — overhead de HT/SMT em workloads de ALU pesada degrada ~25%.

---

## Apêndice A — Grafo de Chamadas Completo

```
simulate_multi()                              [Python, multi_forward.py]
│
├── _validate_multi_inputs()                  [Python]
├── _build_unique_hordist_caches()            [Python, loop por (iTR, iAng)]
│     └── precompute_common_arrays_cache()   [@njit parallel=True, prange(nf)]
│           └── common_arrays()             [@njit, loop serial por camada]
│                 ├── Phase 1: u, s, uh, sh, AdmInt  [downward]
│                 ├── Phase 2: RTEdw, RTMdw           [upward recursion]
│                 └── Phase 3: RTEup, RTMup           [downward recursion]
│
├── [se models is not None]
│     └── _workers.run_batch()               [Python, ProcessPoolExecutor]
│           └── [N workers] → simulate_multi() recursivo
│
└── [se models is None] → single-model path
      │
      ├── _simulate_combined_prange()        [@njit parallel=True, prange(n_combos × n_pos)]
      │     └── _fields_in_freqs_kernel_cached()  [@njit, range(nf) serial]
      │           ├── find_layers_tr()       [@njit, busca binária O(log n)]
      │           ├── [range(nf)]
      │           │     ├── common_factors()     [@njit, O(npt)]
      │           │     ├── hmd_tiv()            [@njit, O(npt × n_layers)]
      │           │     │     ├── [dipolo x]: geometria + propagação de camadas
      │           │     │     ├── [dipolo y]: geometria + propagação de camadas
      │           │     │     └── Integrais de Hankel J0, J1 × wJ0, wJ1
      │           │     ├── vmd()                [@njit, O(npt × n_layers)]
      │           │     │     └── Mesma estrutura que hmd_tiv
      │           │     └── rotate_tensor()      [@njit, O(1)]
      │           └── flatten → cH[i_f, :] = [Hxx, Hxy, ..., Hzz]
      │
      ├── [se F6] → apply_compensation()      [Python/NumPy, postprocess]
      └── [se F7] → apply_tilted_antennas()   [Python/NumPy, postprocess]
```

---

## Apêndice B — Anatomia dos Arrays por Camada

```
Para uma formação com n=10 camadas e Werthmuller 201pt:

Inputs da formação:
  rho_h:    (10,)     — [Ω·m]
  rho_v:    (10,)     — [Ω·m]
  esp:      (8,)      — [m] (n-2 espessuras internas)
  h_arr:    (10,)     — [m] (todas as espessuras, incluindo limites)
  prof_arr: (11,)     — [m] (fronteiras de camada: 0, h0, h0+h1, ...)
  eta:      (10, 2)   — [S/m] complex128 (condutividades TE e TM)

Arrays do cache (por frequência f e hordist h):
  u_cache:      (nf, 201, 10)  complex128 — constantes de propagação
  s_cache:      (nf, 201, 10)  complex128 — constantes auxiliares
  uh_cache:     (nf, 201, 10)  complex128
  sh_cache:     (nf, 201, 10)  complex128
  RTEdw_cache:  (nf, 201, 10)  complex128 — reflexão TE descendente
  RTEup_cache:  (nf, 201, 10)  complex128 — reflexão TE ascendente
  RTMdw_cache:  (nf, 201, 10)  complex128 — reflexão TM descendente
  RTMup_cache:  (nf, 201, 10)  complex128 — reflexão TM ascendente
  AdmInt_cache: (nf, 201, 10)  complex128 — admitância intrínseca

Memória por cache (nf=1, npt=201, n=10):
  9 arrays × 201 × 10 × 16 bytes = 289.440 bytes ≈ 283 KB

Para nf=4:
  9 × 201 × 10 × 4 × 16 = 1.157.760 bytes ≈ 1.1 MB

Tensores intermediários de dipoles (por posição, por frequência):
  Ktedz_J0: (201,) complex128   — integrando TE para J0
  Ktedz_J1: (201,) complex128   — integrando TE para J1
  Ktm_J0:   (201,) complex128   — integrando TM para J0
  Ktm_J1:   (201,) complex128   — integrando TM para J1
  (+ variantes para vmd: KtedzzJ1, KtezJ0, ...)

Saída final por posição e frequência:
  cH: (nf, 9) complex128   ← tensor EM completo (9 componentes)

Saída acumulada de simulate_multi:
  H_tensor: (nTR, nAng, n_pos, nf, 9) complex128
  Para nTR=4, nAng=8, n_pos=600, nf=4:
    → 4 × 8 × 600 × 4 × 9 = 691.200 complexos = 10.6 MB
```

---

## Apêndice C — Análise de Custo Computacional

### C.1 Breakdown por Função (% do tempo total, nf=1, n_layers=22, npt=201)

```
Distribuição estimada de tempo por chamada ao kernel (uma posição, uma frequência):

  find_layers_tr():      <0.1%   (busca binária, O(log 22) ≈ 5 ops)
  common_factors():       5-8%   (O(npt=201), carregamento de cache)
  hmd_tiv():             42-45%  (O(npt × n_layers), integrais de Hankel)
  vmd():                 38-42%  (O(npt × n_layers), integrais de Hankel)
  rotate_tensor():       <0.5%   (3×3 matrix mult)
  flatten + store:       <0.5%   (escrita no array de saída)

  TOTAL:                100%

Implicação: 80-87% do custo está em hmd_tiv + vmd.
Otimizações que reduzem custo de hmd_tiv/vmd têm impacto máximo.
  → fastmath=True (O3): ~8-15% nos kernels = ~6-12% global
  → FLAT prange (O2): reduz desbalanceamento, não custo unitário
  → Filtro Kong 61pt: 3.3× mais rápido em hmd_tiv+vmd = 3× global
```

### C.2 Custo de Pré-cômputo vs Simulação

```
Para Cenário E (n_pos=600, n_layers=22, nf=1, hordist=0):

  precompute_common_arrays_cache():
    1 chamada × O(npt × n_layers) = 1 × 201 × 22 = 4.422 operações
    Custo: ~0.5ms (prange(1) = sem paralelismo)

  _simulate_combined_prange():
    600 posições × O(npt × n_layers) = 600 × 201 × 22 = 2.653.200 operações
    Custo: ~8.5s (com paralelo, 8 threads)

  Razão precompute/simulação: 0.5ms / 8500ms ≈ 0.006%
  → precompute é desprezível para nf=1 e n_pos=600

Para Cenário 2 (n_pos=30, nf=4):
  precompute: 4 × O(201 × 22) = 17.688 operações → ~2ms (prange(4) em 4 threads = 0.5ms)
  simulação:  30 × O(201 × 22 × 4) = 531.960 operações → ~0.43s
  Razão: 0.5ms / 430ms ≈ 0.12%
  → precompute continua negligível
```

### C.3 Scalability com `n_layers`

```
Custo de hmd_tiv + vmd ∝ n_layers (recursão de propagação de camadas)

n_layers=3:   custo relativo = 1.0×   (modelo sintético simples)
n_layers=10:  custo relativo = 3.3×   (formação típica)
n_layers=22:  custo relativo = 7.3×   (sequência estratigráfica complexa)
n_layers=50:  custo relativo = 16.7×  (modelo geológico detalhado)

IMPORTANTE: Para n_layers alto (>30), a recursão de propagação de camadas
se torna o gargalo secundário após as integrais de Hankel.
Potencial de otimização: recursão SIMD para n_layers usando NumPy nativo.
(Deferred: requer validação cuidadosa da recursão vetorializada)
```

---

*Documento gerado em 2026-05-02 com base na análise do código `geosteering_ai/simulation/` v2.21 (commit `cba27dd`). Filtro padrão: Werthmuller 201pt. Paridade Fortran <1e-12 é constraint inviolável em todas as otimizações propostas.*
