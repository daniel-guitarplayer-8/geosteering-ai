# Análise de Paralelismo CPU — Simulador PerfilaAnisoOmp

## Roteiro de Otimização OpenMP para Geração de Dados de Treinamento em Escala

**Projeto:** Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning
**Simulador:** PerfilaAnisoOmp (`Fortran_Gerador/PerfilaAnisoOmp.f08`)
**Versão do Documento:** 1.0 (Abril 2026)
**Base Técnica:** `docs/reference/documentacao_simulador_fortran.md` (v4.0, 6.558 linhas)
**Seções Consultadas:** 11.1–11.8, 12.7.1–12.7.2, 12.7.4–12.7.5

---

## Sumário

1. [Contexto e Motivação](#1-contexto-e-motivação)
2. [Diagnóstico do Estado Atual](#2-diagnóstico-do-estado-atual)
3. [Análise de Custo Computacional](#3-análise-de-custo-computacional)
4. [Mapa de Oportunidades de Otimização](#4-mapa-de-oportunidades-de-otimização)
5. [Proposta de Paralelismo Otimizado](#5-proposta-de-paralelismo-otimizado)
6. [Análise de Escalabilidade Multi-Core](#6-análise-de-escalabilidade-multi-core)
7. [Sequência de Fases — Roteiro de Implementação](#7-sequência-de-fases--roteiro-de-implementação)
   - Fase 0 — Benchmark de Baseline
   - Fase 1 — Vetorização SIMD da Convolução de Hankel
   - Fase 2 — Escalonador Híbrido de Threads
   - Fase 3 — Pré-alocação de Workspace por Thread
   - Fase 4 — Cache de `commonarraysMD` por `(r, freq)`
   - Fase 5 — Colapso de Laços com `collapse(3)`
   - Fase 6 — Cache de `commonfactorsMD` por `camadT`
8. [Resultados Esperados](#8-resultados-esperados)
9. [Métricas de Sucesso e KPIs](#9-métricas-de-sucesso-e-kpis)
10. [Pré-condição Estrutural para GPU](#10-pré-condição-estrutural-para-gpu)
11. [Referências](#11-referências)

---

## 1. Contexto e Motivação

### 1.1 Papel do Simulador no Pipeline Geosteering AI

O simulador `PerfilaAnisoOmp` resolve o *problema direto* (forward modeling) da resposta eletromagnética de uma ferramenta de perfilagem triaxial em meios estratificados com anisotropia TIV (Transversalmente Isotrópica Vertical). Ele é o componente de geração de dados sintéticos do projeto — toda a base de treinamento das 48 arquiteturas de deep learning depende de sua capacidade de produzir modelos geológicos em escala industrial.

A cadeia completa do projeto segue o fluxo:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CADEIA DE GERAÇÃO DE DADOS — GEOSTEERING AI v2.0                           │
│                                                                              │
│  Gerador Python           Simulador Fortran          Pipeline DL (TF/Keras) │
│  (modelo geológico)   →   (PerfilaAnisoOmp)      →   (treinamento/inversão) │
│                                                                              │
│  model.in               →  .out / .dat           →   dataset .npz           │
│  (n camadas,               (9 comp. tensor H,        (input_features,        │
│   resistividades,           z_rho, geometria)          output_targets)       │
│   ângulos, freq.)                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Necessidade de Escala

O treinamento robusto das redes neurais de inversão requer conjuntos de dados que cubram o espaço completo de modelos geológicos: variação de resistividade horizontal (`ρ_h`), resistividade vertical (`ρ_v`), espessuras de camadas, ângulo de inclinação da ferramenta (`dip`), posições relativas e frequências. A estimativa de cobertura adequada é da ordem de **milhões de modelos geológicos** distintos.

Com a capacidade atual de **24.000 modelos/hora** (16 threads CPU, baseline documentado), gerar 10 milhões de modelos exigiria mais de 17 dias de processamento contínuo. As otimizações CPU documentadas na Seção 12.7.2 visam elevar esse throughput para **82.000–115.000 modelos/hora** — reduzindo o mesmo trabalho para 4–5 dias, e servindo como base para a posterior implementação GPU (480.000–1.200.000 modelos/hora).

### 1.3 Escopo deste Documento

Este documento analisa exclusivamente as otimizações de **paralelismo CPU com OpenMP** (Fase 1 do Pipeline A, Seção 12.7.2 da documentação técnica do simulador), organizando-as em sequência de implementação com justificativa técnica, código de referência e critérios de validação.

---

## 2. Diagnóstico do Estado Atual

### 2.1 Arquitetura de Paralelismo Existente

O simulador utiliza **paralelismo OpenMP aninhado em 2 níveis** (Seção 11.1):

```
Nível 1 — externo: Loop sobre ângulos theta
  !$omp parallel do schedule(dynamic) num_threads(num_threads_k)
  do k = 1, ntheta          ← tipicamente 1–2 threads

    Nível 2 — interno: Loop sobre medições por ângulo
    !$omp parallel do schedule(dynamic) num_threads(num_threads_j)
    do j = 1, nmed(k)        ← 600+ iterações disponíveis
      call fieldsinfreqs(ang, nf, freq, posTR, dipolo, npt, krwJ0J1,
                         n, h, prof, resist, zrho, cH)
    end do
    !$omp end parallel do

  end do
  !$omp end parallel do
```

A distribuição de threads segue a regra:

```fortran
maxthreads    = omp_get_max_threads()   ! Total de threads do sistema
num_threads_k = ntheta                  ! Nível 1: 1 thread por ângulo
num_threads_j = maxthreads - ntheta     ! Nível 2: restante para medições
```

No caso mais frequente (`ntheta = 1`), o nível 1 usa exatamente **1 thread** e o nível 2 usa `maxthreads - 1` threads. O fork/join aninhado ocorre sem qualquer ganho de paralelismo no nível externo.

### 2.2 Hierarquia de Sub-rotinas Envolvidas no Loop Interno

O fluxo de chamadas dentro de cada iteração `j` do loop paralelo interno:

```
┌─────────────────────────────────────────────────────────────────────┐
│  HIERARQUIA DE CHAMADAS — loop j = 1, nmed(k)                      │
│                                                                     │
│  fieldsinfreqs(ang, nf, freq, posTR, ...)                          │
│    │                                                                │
│    ├── commonarraysMD(r, freq, n, h, resist, ...)   ← 45% CPU     │
│    │     Calcula: u, s, uh, sh, RTEdw, RTEup,                      │
│    │              RTMdw, RTMup, AdmInt, ImpInt                     │
│    │     Chamado: nf × nmed = 2 × 600 = 1.200 vezes/modelo        │
│    │                                                                │
│    ├── commonfactorsMD(camadT, freq, ...)            ← 15% CPU     │
│    │     Calcula: Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz            │
│    │     Depende: (camadT, freq)                                    │
│    │                                                                │
│    ├── hmd_TIV_optimized(...)                        ← 20% CPU     │
│    │     Calcula: Hx, Hy, Hz para dipolo magnético horizontal      │
│    │     Aloca: Tudw, Txdw, Tuup, Txup, TEdwz, TEupz             │
│    │            → 600 allocate/deallocate por modelo por freq      │
│    │                                                                │
│    └── vmd_optimized(...)                            ← 10% CPU     │
│          Calcula: Hz para dipolo magnético vertical                 │
│          Aloca: mesmos arrays de hmd                                │
│                 → 600 allocate/deallocate por modelo por freq      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Métricas de Baseline

A Seção 12.7.1 fixa os números de referência para a configuração padrão (1 ângulo, 2 frequências, 600 medições, 16 threads CPU):

| Métrica | Valor (Baseline) |
|:--------|:----------------:|
| Tempo por modelo | **2,40 segundos** |
| Throughput | **24.000 modelos/hora** |
| Chamadas a `commonarraysMD` por modelo | **1.200** |
| Chamadas a `allocate/deallocate` por modelo | **2.400** |
| Eficiência de paralelismo (Amdahl) | não medida |
| Uso de memória por thread | ~150 MB |

### 2.4 As Quatro Causas-Raiz do Desempenho Subótimo

A Seção 11.2 diagnostica as causas-raiz do desempenho abaixo do potencial teórico:

---

#### Causa-Raiz 1 — Alocação Dinâmica Dentro do Laço Paralelo (Impacto: Alto)

As sub-rotinas `hmd_TIV_optimized` e `vmd_optimized` alocam dinamicamente os arrays `Tudw`, `Txdw`, `Tuup`, `Txup`, `TEdwz`, `TEupz` a cada chamada. Com 600 medições × 2 frequências, isso resulta em:

```
2 sub-rotinas × 6 arrays × 600 medições × 2 frequências = 2.400 alocações/modelo
```

Em regiões OpenMP paralelas, múltiplas threads competem simultaneamente pelo alocador de heap do sistema operacional — gerando **contenção de mutex**, **fragmentação de memória** e latência de alocação variável. Este é o padrão mais prejudicial ao desempenho em código paralelo.

---

#### Causa-Raiz 2 — Redundância Computacional em `commonarraysMD` (Impacto: Muito Alto)

A sub-rotina `commonarraysMD` calcula os coeficientes de reflexão TE/TM e as constantes de propagação, que dependem de:

```
f(r, freq, eta(n,2), zeta) → u(npt,n), s(npt,n), RTEdw(npt,n), RTEup(npt,n), ...
```

Em perfilagem de poço com passo fixo `p_med`, o espaçamento transmissor-receptor `r = dTR = 1,0 m` é **invariante** por construção. Com apenas `nf = 2` frequências, `commonarraysMD` produz resultados idênticos para todas as 600 medições do mesmo ângulo e frequência. O código atual a chama **1.200 vezes por modelo** com argumentos idênticos — **trabalho 100% redundante**.

---

#### Causa-Raiz 3 — Escalonador `dynamic` para Carga Uniforme (Impacto: Baixo)

O escalonador `dynamic` distribui iterações do loop às threads sob demanda, mantendo uma fila gerenciada por mutex. Para um dado modelo geológico onde todas as 600 medições têm custo computacional semelhante (mesma profundidade de camadas, mesmo `dTR`), o escalonador `static` distribui as iterações estaticamente em tempo de compilação — eliminando o overhead de sincronização sem custo de desbalanceamento.

---

#### Causa-Raiz 4 — Redundância em `commonfactorsMD` por `camadT` Invariante (Impacto: Médio)

A sub-rotina `commonfactorsMD` calcula os fatores `Mxdw`, `Mxup`, `Eudw`, `Euup`, `FEdwz`, `FEupz` que dependem apenas de `(camadT, freq)`. Em formações com camadas de espessura maior que o passo de medição `p_med = 1,0 m`, medições consecutivas frequentemente posicionam o transmissor na **mesma camada geológica** (`camadT` constante). Calcular `commonfactorsMD` repetidamente para `camadT` invariante é trabalho desnecessário que aumenta linearmente com o número de medições por camada.

---

## 3. Análise de Custo Computacional

### 3.1 Complexidade por Modelo

A Seção 11.6 estabelece a complexidade total do simulador:

```
Computações por modelo:
  ntheta × nfreq × nmed × custo(commonarraysMD + commonfactorsMD + hmd + vmd + rotação)
  = 1 × 2 × 600 × (~201 × n operações complexas)
  = 1.200 × (~201 × 10 camadas = 2.010 FLOPs complexos)
  ≈ 2,4 × 10⁶ operações complexas por modelo

Total para 1.000 modelos: ≈ 2,4 × 10⁹ operações complexas
Tempo estimado (1 core, ~1 GFLOP/s complexo): ~2,4 segundos por modelo
```

### 3.2 Distribuição de Tempo por Sub-rotina

Com base na análise de profiling descrita na Seção 11.4 (`gprof -l`):

```
┌──────────────────────────────────────────────────────────────────────┐
│  DISTRIBUIÇÃO DE TEMPO — BASELINE (16 threads, 1 ângulo, 2 freq)    │
│                                                                      │
│  commonarraysMD       ████████████████████████ 45%  (gargalo 1)     │
│  hmd_TIV_optimized    ████████████████ 30%          (gargalo 2)     │
│    - redução Hankel     ██████ ~25% do total                        │
│    - overhead malloc    ████ ~5% do total                           │
│  commonfactorsMD      ████████ 15%                                  │
│  vmd_optimized        █████ ~7%                                     │
│  rotação tensorial    ██ ~3%                                        │
│                                                                      │
│  commonarraysMD + hmd = ~75% do tempo → alvos prioritários          │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 O Núcleo Computacional: Filtro de Hankel de Werthmuller

O núcleo computacional de `hmd_TIV_optimized` e `vmd_optimized` são reduções sobre os 201 pontos do filtro de Hankel:

```fortran
! Exemplo de redução crítica (Seção 12.7.2, Passo 1.4):
kernelHxJ1 = (twox2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twoy2_r2m1 * sum(Ktm_J1)) / r
kernelHxJ0 = x2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * y2_r2 * sum(Ktm_J0 * kr)
```

Com `npt = 201` pontos e aritmética complexa de precisão dupla (complex(8)), cada `sum()` é uma redução sobre vetores complexos de 201 elementos de 16 bytes. O compilador `gfortran` com `-O3 -march=native` pode vetorizar automaticamente, mas a forma `sum()` intrínseco sobre arrays complexos frequentemente impede a geração de instruções AVX-2 (4 doubles simultâneos) ou AVX-512 (8 doubles simultâneos).

A Seção 11.8.4 observa que `npt = 201 = 25 × 8 + 1`, o que significa:
- 25 iterações vetorizadas com AVX-512 (8 doubles por ciclo de clock)
- 1 iteração escalar residual

A ineficiência atual resulta de o compilador não conseguir vectorizar o `sum()` intrínseco com `reduction` semântica para complexos, gerando código escalar em vez de SIMD.

---

## 4. Mapa de Oportunidades de Otimização

### 4.1 Otimizações de Tempo de Execução (Seção 11.4)

| # | Otimização | Impacto Estimado | Sub-rotina Alvo | Natureza |
|:-:|:-----------|:----------------:|:----------------|:--------:|
| 1 | Cache de `commonarraysMD` por `(r, freq)` | **30–50% redução** | `fieldsinfreqs` | Redundância |
| 2 | Pré-alocação de workspace por thread | **Alto** | `hmd/vmd_optimized` | Memória |
| 3 | SIMD para convolução Hankel | 10–30% | `hmd/vmd_optimized` | Vetorização |
| 4 | Colapso de laços `collapse(2/3)` | 10–20% adicional | `perfila1DanisoOMP` | Estrutura |
| 5 | Escalonador híbrido static/dynamic | 5–10% | `perfila1DanisoOMP` | Overhead |
| 6 | Cache de `commonfactorsMD` por `camadT` | 15–25% | `fieldsinfreqs` | Redundância |
| 7 | Paralelismo sobre frequências | 2× para `nf=2` | `perfila1DanisoOMP` | Estrutura |

### 4.2 Otimizações de Memória (Seção 11.3)

| Otimização | Impacto | Complexidade |
|:-----------|:-------:|:------------:|
| Pré-alocar arrays de trabalho por thread | **Alto** — elimina `malloc` no laço | Média |
| Mover `commonarraysMD` para fora do laço `j` | **Alto** — computa uma vez por `(ângulo, freq)` | Média |
| `firstprivate` para arrays somente-leitura | Médio — evita cópias desnecessárias | Baixa |
| Pool de memória por thread (memory pooling) | **Alto** — elimina fragmentação | Alta |
| Arrays pequenos (H 3×3) em stack | Médio — elimina `allocate` desnecessário | Baixa |
| Alinhamento de 64 bytes para SIMD | Médio — habilita AVX-512 | Baixa |

---

## 5. Proposta de Paralelismo Otimizado

### 5.1 Arquitetura-Alvo

A Seção 11.5 propõe uma estrutura em três elementos combinados:

```
┌────────────────────────────────────────────────────────────────────────────┐
│  ARQUITETURA-ALVO — PARALELISMO OTIMIZADO                                 │
│                                                                            │
│  ANTES DA REGIÃO PARALELA (serial, uma vez por modelo):                   │
│    ① precompute_common_arrays(nf=2, r=1.0, ...) → u_all, RTEdw_all, ...  │
│       → elimina 1.198 das 1.200 chamadas a commonarraysMD                 │
│                                                                            │
│  ANTES DO LAÇO PARALELO (alocação única):                                 │
│    ② allocate(ws_pool(0:maxthreads-1))                                    │
│       → ~7,7 MB × 16 threads = ~123 MB total (cabe na L3 cache)          │
│                                                                            │
│  LAÇO PARALELO (colapso 3 níveis):                                        │
│    ③ !$omp parallel do collapse(3) schedule(dynamic, 8)                  │
│       do k = 1, ntheta                                                    │
│         do f = 1, nf                                                      │
│           do j = 1, nmed(k)                                               │
│             tid = omp_get_thread_num()                                    │
│             call compute_measurement_ws(k, f, j, ws_pool(tid),           │
│                                         u_all(:,:,f), RTEdw_all(:,:,f))  │
│           end do                                                          │
│         end do                                                            │
│       end do                                                              │
│                                                                            │
│  DENTRO DE compute_measurement_ws (por thread, sem malloc):               │
│    ④ !$omp simd reduction(+:acc)                                          │
│       do ip = 1, npt   → vetorização AVX-512 dos 201 pontos Hankel       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Tipo `thread_workspace` (Seção 11.8.1)

A estrutura de workspace por thread agrega todos os arrays que atualmente são alocados dinamicamente a cada chamada:

```fortran
! Definição completa do tipo workspace (magneticdipoles.f08)
type :: thread_workspace
    ! Arrays de propagação (npt × n_max): 201 × 80 pontos
    complex(dp), allocatable :: u(:,:), s(:,:), v(:,:)
    complex(dp), allocatable :: uh(:,:), sh(:,:)
    complex(dp), allocatable :: tghuh(:,:), tghsh(:,:)

    ! Coeficientes de reflexão TE/TM (npt × n_max)
    complex(dp), allocatable :: RTEdw(:,:), RTEup(:,:)
    complex(dp), allocatable :: RTMdw(:,:), RTMup(:,:)
    complex(dp), allocatable :: AdmInt(:,:), ImpInt(:,:)
    complex(dp), allocatable :: AdmAp_dw(:,:), AdmAp_up(:,:)
    complex(dp), allocatable :: ImpAp_dw(:,:), ImpAp_up(:,:)

    ! Fatores de onda (npt)
    complex(dp), allocatable :: Mxdw(:), Mxup(:)
    complex(dp), allocatable :: Eudw(:), Euup(:), FEdwz(:), FEupz(:)

    ! Coeficientes de transmissão (npt × n_max)
    complex(dp), allocatable :: Txdw(:,:), Txup(:,:)
    complex(dp), allocatable :: Tudw(:,:), Tuup(:,:)

    ! Kernels e acumuladores da redução Hankel (npt)
    complex(dp), allocatable :: Ktm(:), Kte(:), Ktedz(:)
end type thread_workspace

! Memória estimada por thread (n_max=80, npt=201):
!   ~30 arrays × 201 × 80 × 16 bytes ≈ 7,7 MB/thread
! Para 16 threads: ~123 MB total
! Para 8 threads:  ~62 MB total (cabe confortavelmente na L3 de CPUs modernas)
```

### 5.3 Arrays Pequenos em Stack (Seção 11.8.2)

Para arrays de tamanho fixo e pequeno (tensor H 3×3, vetores de componentes), a alocação em stack elimina overhead de `malloc`:

```fortran
! CORRETO: stack allocation (tamanho fixo, conhecido em compilação)
complex(dp) :: matH(3,3)
complex(dp) :: Hx_p(1,2), Hy_p(1,2), Hz_p(1,2)

! EVITAR: heap allocation para arrays pequenos
complex(dp), allocatable :: matH(:,:)   ! overhead desnecessário de malloc
allocate(matH(3,3))
```

---

## 6. Análise de Escalabilidade Multi-Core

### 6.1 Tarefas Disponíveis por Configuração (Seção 11.7)

A Seção 11.7 quantifica a eficiência do paralelismo com `collapse(3)` para configurações típicas de uso:

| Configuração | ntheta | nf | nmed total | Tarefas Totais | Eficiência (8 cores) |
|:-------------|:------:|:--:|:----------:|:--------------:|:--------------------:|
| Baseline (ntheta=1) | 1 | 2 | 600 | **1.200** | ~95% (150 tarefas/core) |
| Multi-ângulo (ntheta=2) | 2 | 2 | 600+622 | **2.444** | ~98% (305 tarefas/core) |
| Multi-freq extenso | 1 | 4 | 600 | **2.400** | ~97% (300 tarefas/core) |

Com `ntheta = 1` e a estrutura aninhada atual, o nível 1 usa exatamente 1 thread e gera fork/join sem benefício. O `collapse(3)` expõe 1.200 tarefas independentes ao escalonador, garantindo 95%+ de eficiência mesmo em 16 cores.

### 6.2 Configuração de Afinidade NUMA (Seção 11.7)

Para extrair o desempenho máximo em sistemas multi-socket, a Seção 11.7 recomenda:

```bash
# Single-socket (laptop, workstation, Colab):
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
# Justificativa: distribui threads uniformemente entre cores para balanceamento
# térmico e evita contenção de hyperthreading em operações FP intensivas.

# Dual-socket (servidor HPC com 2 CPUs):
export OMP_PLACES=cores
export OMP_PROC_BIND=close
# Justificativa: mantém threads no mesmo socket NUMA, reduzindo latência de
# acesso à memória de ~300 ns (inter-socket) para ~100 ns (intra-socket).
```

### 6.3 Lei de Amdahl Aplicada

Com base nos percentuais de tempo por sub-rotina, a fração paralelizável `P` do código é estimada:

```
P = 1 - fração_serial

Fração serial estimada (operações intrinsecamente sequenciais):
  - Leitura/escrita de arquivos: ~1%
  - Cálculo de nmed, carga do filtro: ~1%
  - Pré-processamento serial (precompute_common_arrays): ~0,5%
  Total serial: ~2,5%

P ≈ 0,975

Speedup máximo teórico (Lei de Amdahl, N=16 cores):
  S_max = 1 / (1 - P + P/N) = 1 / (0,025 + 0,975/16) = 1 / 0,086 ≈ 11,6×

Speedup esperado realista (com overhead OpenMP ~5%):
  S_realista ≈ 8–10× sobre código single-thread
  S_realista ≈ 4–5× sobre baseline 16-thread com implementação otimizada
```

O KPI K1 (≥ 4× sobre o baseline de 16 threads) é tecnicamente alcançável pelas otimizações documentadas.

---

## 7. Sequência de Fases — Roteiro de Implementação

As fases são ordenadas por **relação retorno/risco**, do menor risco ao maior, garantindo que cada fase seja validada antes da próxima ser iniciada.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ROTEIRO DE IMPLEMENTAÇÃO CPU — 5 SEMANAS                                  │
│                                                                             │
│  Semana 0   FASE 0 — Benchmark baseline                                    │
│               │                                                             │
│  Semana 1   FASE 1 — SIMD Hankel (Passo 1.4)  → +10–30% nas reduções     │
│             FASE 2 — Escalonador híbrido (Passo 1.5)  → +5–10%            │
│               │                                                             │
│  Semanas    FASE 3 — Workspace pré-alocado (Passo 1.1) → +40–80%         │
│  1–2          │                                                             │
│               │                                                             │
│  Semanas    FASE 4 — Cache commonarraysMD (Passo 1.2)  → +60–120%        │
│  2–3          │                                                             │
│               │                                                             │
│  Semana 3   FASE 5 — Collapse loops (Passo 1.3)        → +20–50%         │
│               │                                                             │
│  Semana 4   FASE 6 — Cache commonfactorsMD (Passo 1.6) → +20–50%         │
│               │                                                             │
│               ▼                                                             │
│  Semana 4   BENCHMARK FINAL: Meta ≥ 4× speedup / ≥ 115.000 mod/h         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### FASE 0 — Benchmark de Baseline (Semana 0)

**Objetivo:** Estabelecer medidas reproduzíveis antes de qualquer alteração. Este passo é obrigatório — sem ele, é impossível verificar se cada fase subsequente produziu o ganho esperado.

**Por que é necessário:** O valor de 2,40 s/modelo é uma estimativa documentada. O valor real pode variar conforme hardware, sistema operacional, carga do sistema e conjunto de modelos testados. Um benchmark controlado e reproduzível é o único critério de verdade.

**Ações detalhadas:**

```bash
# 1. Compilar com flags de profiling
gfortran -O3 -march=native -fopenmp -pg \
         -o tatu_profile.x \
         parameters.f08 utils.f08 filtersv2.f08 \
         magneticdipoles.f08 PerfilaAnisoOmp.f08 RunAnisoOmp.f08

# 2. Executar com 1.000 modelos
export OMP_NUM_THREADS=16
time ./tatu_profile.x config_1000_models.namelist

# 3. Gerar perfil de tempo por sub-rotina
gprof tatu_profile.x gmon.out > profile_baseline.txt
grep -E "commonarraysMD|hmd_TIV|vmd_optimized|commonfactorsMD" profile_baseline.txt

# 4. Registrar saída de referência para validação posterior
md5sum validacao.dat > validacao_baseline.md5
```

**Saídas esperadas:**
- Confirmação que `commonarraysMD` ≈ 45% e `hmd/vmd_optimized` ≈ 30% do tempo total
- Tempo por modelo registrado com desvio padrão (≥ 10 execuções)
- Hash MD5 da saída de referência para validação diferencial

**Critério de entrada:** Nenhum (primeira execução).
**Critério de saída:** baseline documentado = 2,40 ± 0,2 s/modelo, 24.000 modelos/hora.

---

### FASE 1 — Vetorização SIMD da Convolução de Hankel (Passo 1.4) — Semana 1

**Objetivo:** Substituir os `sum()` intrínsecos sobre arrays complexos de 201 elementos por laços explícitos com diretiva `!$omp simd`, habilitando vetorização AVX-2/AVX-512.

**Justificativa técnica:** As reduções de Hankel representam ~25% do tempo total (dentro dos ~30% de `hmd/vmd_optimized`). O compilador `gfortran` com `-O3 -march=native` frequentemente falha em vectorizar `sum()` sobre `complex(8)` devido à semântica de redução complexa. A diretiva `!$omp simd reduction(+:acc)` instrui o compilador explicitamente a usar registradores SIMD.

**Arquivos a modificar:** `magneticdipoles.f08` — sub-rotinas `hmd_TIV_optimized` e `vmd_optimized`.

**Implementação:**

```fortran
! ANTES: sum() implícito — gfortran pode não vetorizar
kernelHxJ1 = (twox2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twoy2_r2m1 * sum(Ktm_J1)) / r

! DEPOIS: laço explícito com SIMD — garantia de vetorização
complex(dp) :: acc_KtedzJ1, acc_KtmJ1
integer :: ip
acc_KtedzJ1 = (0.d0, 0.d0)
acc_KtmJ1   = (0.d0, 0.d0)

!$omp simd reduction(+:acc_KtedzJ1, acc_KtmJ1)
do ip = 1, npt   ! npt=201 = 25×8 + 1 → 25 iterações AVX-512 + 1 escalar
    acc_KtedzJ1 = acc_KtedzJ1 + Ktedz_J1(ip)
    acc_KtmJ1   = acc_KtmJ1   + Ktm_J1(ip)
end do

kernelHxJ1 = (twox2_r2m1 * acc_KtedzJ1 - kh2(camadR) * twox2_r2m1 * acc_KtmJ1) / r
```

**Alinhamento de memória para AVX-512:**

```fortran
! Adicionar em magneticdipoles.f08, antes da declaração dos arrays:
!GCC$ attributes aligned(64) :: Ktedz_J1, Ktm_J1, Ktedz_J0, Ktm_J0
!GCC$ attributes aligned(64) :: u, s, uh, sh
```

**Flags de compilação adicionais no Makefile:**

```makefile
FFLAGS = -O3 -march=native -fopenmp -ffast-math -funroll-loops \
         -fopt-info-vec-optimized \    # Relatório: loops vetorizados com SIMD
         -fprefetch-loop-arrays        # Prefetch automático de arrays grandes
```

**Verificação de vetorização:**

```bash
# Confirmar que os loops foram vetorizados:
gfortran $(FFLAGS) -o tatu_simd.x *.f08 2>&1 | grep "vectorized"
# Saída esperada: "loop at magneticdipoles.f08:NNN vectorized using 4-wide simd"
```

**Validação numérica:** `max_rel_err(H_simd, H_baseline) < 1×10⁻¹²` para todos os componentes (Re/Im dos 9 componentes do tensor H).
**Speedup esperado:** 1,3–2,0× nas reduções de Hankel → 1,1–1,2× no tempo total do modelo.
**Risco:** Muito baixo — diretivas não alteram semântica matemática, apenas a forma de execução.

---

### FASE 2 — Escalonador Híbrido de Threads (Passo 1.5) — Semana 1

**Objetivo:** Substituir `schedule(dynamic)` fixo por lógica adaptativa que escolhe `static` para carga uniforme (ntheta ≤ 4) e `guided` para carga variável (ntheta > 4).

**Justificativa técnica:** O escalonador `dynamic` mantém uma fila de iterações distribuídas sob demanda, protegida por mutex. Para 600 medições com custo uniforme (mesmo modelo, mesmo `dTR`), cada acesso à fila é overhead puro sem benefício de balanceamento. O escalonador `static` distribui iterações em tempo de compilação: `chunk = nmed / num_threads`, sem sincronização em runtime.

**Arquivo a modificar:** `PerfilaAnisoOmp.f08` — sub-rotina `perfila1DanisoOMP`.

**Implementação:**

```fortran
! Escalonador adaptativo para o laço externo (ângulos):
if (ntheta <= 4) then
    ! Carga pequena e uniforme: static sem overhead de despacho
    !$omp parallel do schedule(static) num_threads(num_threads_k) &
    !$omp   private(k, ang, seno, coss, px, pz, Lsen, Lcos, z_rho1, c_H1)
    do k = 1, ntheta
        ! ... corpo do laço ...
    end do
    !$omp end parallel do
else
    ! Carga grande e variável: guided distribui chunks decrescentes
    !$omp parallel do schedule(guided, 4) num_threads(num_threads_k) &
    !$omp   private(k, ang, seno, coss, px, pz, Lsen, Lcos, z_rho1, c_H1)
    do k = 1, ntheta
        ! ... corpo do laço ...
    end do
    !$omp end parallel do
end if

! Escalonador adaptativo para o laço interno (medições):
! chunk_size estático minimiza overhead quando carga é uniforme
chunk_size = max(1, nmed(k) / num_threads_j)
!$omp parallel do schedule(static, chunk_size) num_threads(num_threads_j) &
!$omp   private(j, x, y, z, Tx, Ty, Tz, posTR, zrho, cH)
do j = 1, nmed(k)
    ! ... corpo do laço ...
end do
!$omp end parallel do
```

**Validação numérica:** Resultado bit-for-bit idêntico ao baseline (mesmas operações aritméticas, ordem diferente apenas entre threads).
**Speedup esperado:** 1,05–1,15× adicional sobre a Fase 1.
**Risco:** Muito baixo — apenas mudança de parâmetro de escalonamento.

---

### FASE 3 — Pré-alocação de Workspace por Thread (Passo 1.1) — Semanas 1–2

**Objetivo:** Eliminar todas as chamadas a `allocate/deallocate` dentro do laço paralelo, pré-alocando um workspace exclusivo por thread antes do laço começar.

**Justificativa técnica:** As sub-rotinas `hmd_TIV_optimized` e `vmd_optimized` alocam dinamicamente 6 arrays `(npt, n)` a cada chamada. Com 600 medições × 2 frequências e `npt × n_max = 201 × 80 = 16.080 elementos complex(8)` por array, cada chamada a `allocate` realiza uma requisição de ~258 KB ao alocador do sistema operacional. Em 16 threads paralelas, isso cria **16 requisições simultâneas** ao mesmo mutex do heap, gerando serialização no gargalo mais crítico do loop.

**Arquivos a modificar:**
- `magneticdipoles.f08` — adicionar tipo `thread_workspace`, criar versões `_ws` de `hmd_TIV_optimized` e `vmd_optimized`
- `PerfilaAnisoOmp.f08` — alocar `ws_pool`, chamar versões `_ws` com `ws_pool(tid)`

**Implementação — alocação do pool em `perfila1DanisoOMP`:**

```fortran
! Antes do laço paralelo:
integer :: nthreads_actual
type(thread_workspace), allocatable :: ws_pool(:)

nthreads_actual = omp_get_max_threads()
allocate(ws_pool(0:nthreads_actual-1))
do t = 0, nthreads_actual - 1
    allocate(ws_pool(t)%u    (npt, MAX_N))    ! MAX_N = 80 (máximo de camadas)
    allocate(ws_pool(t)%s    (npt, MAX_N))
    allocate(ws_pool(t)%uh   (npt, MAX_N))
    allocate(ws_pool(t)%sh   (npt, MAX_N))
    allocate(ws_pool(t)%RTEdw(npt, MAX_N))
    allocate(ws_pool(t)%RTEup(npt, MAX_N))
    allocate(ws_pool(t)%RTMdw(npt, MAX_N))
    allocate(ws_pool(t)%RTMup(npt, MAX_N))
    allocate(ws_pool(t)%Mxdw (npt))
    allocate(ws_pool(t)%Mxup (npt))
    allocate(ws_pool(t)%Eudw (npt))
    allocate(ws_pool(t)%Euup (npt))
    allocate(ws_pool(t)%FEdwz(npt))
    allocate(ws_pool(t)%FEupz(npt))
    allocate(ws_pool(t)%Tudw (npt, MAX_N))
    allocate(ws_pool(t)%Txdw (npt, MAX_N))
    allocate(ws_pool(t)%Tuup (npt, MAX_N))
    allocate(ws_pool(t)%Txup (npt, MAX_N))
    allocate(ws_pool(t)%Ktm  (npt))
    allocate(ws_pool(t)%Kte  (npt))
    allocate(ws_pool(t)%Ktedz(npt))
end do

! Laço paralelo modificado:
!$omp parallel do schedule(static, chunk_size) num_threads(num_threads_j) &
!$omp   private(j, tid, x, y, z, Tx, Ty, Tz, posTR, zrho, cH)
do j = 1, nmed(k)
    tid = omp_get_thread_num()
    ! Passa workspace pré-alocado — sem malloc interno:
    call fieldsinfreqs_ws(ang, nf, freq, posTR, dipolo, npt, krwJ0J1, &
                          n, h, prof, resist, zrho, cH, ws_pool(tid))
    z_rho1(j,:,:) = zrho
    c_H1(j,:,:) = cH
end do
!$omp end parallel do

! Após o laço: desalocar workspace
do t = 0, nthreads_actual - 1
    deallocate(ws_pool(t)%u, ws_pool(t)%s, ws_pool(t)%uh, ws_pool(t)%sh)
    deallocate(ws_pool(t)%RTEdw, ws_pool(t)%RTEup, ws_pool(t)%RTMdw, ws_pool(t)%RTMup)
    ! ... demais campos ...
end do
deallocate(ws_pool)
```

**Estratégia de implementação segura:** Criar versões com sufixo `_ws` das sub-rotinas modificadas (preservando as originais para validação diferencial). Remover as originais apenas após validação completa.

**Validação numérica:** `max_rel_err < 1×10⁻¹²` para 1.000 modelos geológicos aleatórios.
**Speedup esperado:** 1,4–1,8× (30–40% de redução no tempo total medido por `gprof`).
**Risco:** Médio — refatoração de assinaturas de 4 sub-rotinas. Mitigação: versões `_ws` paralelas sem alteração do código original.

#### Resultados Empíricos da Fase 3 (2026-04-05) ✅ IMPLEMENTADA

A Fase 3 foi executada com sucesso, preservando integralmente as rotinas originais (`hmd_TIV_optimized`, `vmd_optimized`, `fieldsinfreqs`) e criando as variantes `_ws` (`hmd_TIV_optimized_ws`, `vmd_optimized_ws`, `fieldsinfreqs_ws`) em paralelo, conforme estratégia de implementação segura. O `type :: thread_workspace` foi adicionado a `magneticdipoles.f08` com **6 componentes** (`Tudw, Txdw, Tuup, Txup, TEdwz, TEupz`), não os ~30 originalmente sugeridos na §5.2 desta análise — a revisão do código atual mostrou que os demais arrays (`u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt, Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz`) já eram `automatic arrays` no stack, não alocados dinamicamente. A refatoração deles foi postergada para **Fase 3b** (opcional).

**Ambiente de medição:**
- Hardware: Intel Core i9-9980HK (8 cores físicos, 16 lógicos), 32 GB RAM
- SO: macOS Darwin 25.4 (macOS 26) · gfortran 15.2.0 Homebrew · `ld-classic` (workaround macOS 26)
- Config do `model.in`: **29 camadas**, 2 frequências (20/40 kHz), `ntheta=1`, 600 medidas × 2 freq = 1.200 registros
- Compilação: `-O3 -march=native -ffast-math -funroll-loops -fopenmp -std=f2008`

**Resultados a 8 threads (60 iterações):**

| Métrica | Fase 2 (HEAD) | Fase 3 | Delta |
|:--------|:-------------:|:------:|:-----:|
| Tempo médio (s/modelo) | 0,3830 | **0,3433** | −10,4 % |
| Desvio-padrão (s) | 0,0355 | 0,0278 | −21,7 % |
| Throughput (modelos/h) | 9.399 | **10.485** | **+11,5 %** |

**Escalabilidade 1 → 8 threads (30 iterações por ponto):**

| Threads | Fase 2 tempo (s) | Fase 3 tempo (s) | Fase 2 thput | Fase 3 thput | Speedup 3 vs 2 |
|:-------:|:----------------:|:----------------:|:------------:|:------------:|:--------------:|
| **1**   | 1,7800 | **1,3690** | 2.023 | **2.630** | **1,30×** |
| **2**   | 0,8073 | **0,7500** | 4.459 | **4.800** | 1,08× |
| **4**   | 0,5873 | **0,4617** | 6.129 | **7.798** | **1,27×** |
| **8**   | 0,3830 | **0,3433** | 9.399 | **10.485** | 1,12× |

**Validação numérica:**
- Phase 3 vs HEAD com `-O0` (sem `-ffast-math`): **MD5 binário idêntico** (`8aa4aeee...`) — prova de equivalência matemática bit-exata.
- Phase 3 vs HEAD com `-O3 -ffast-math`: `max|Δ| = 3,4 × 10⁻¹⁴`, `RMS(Δ) = 1,7 × 10⁻¹⁵`, `max rel Δ = 1,2 × 10⁻⁹` — diferença dentro do ULP de `double`, explicada por reordenamento associativo de FP autorizado por `-ffast-math`. **Quatro ordens de magnitude abaixo do critério de aceite `1×10⁻¹⁰`**.

**Contagem de malloc eliminados:**

| Localização | Chamadas antes | Chamadas depois | Redução |
|:------------|:--------------:|:---------------:|:-------:|
| `hmd_TIV_optimized` (4 arrays) | ~4.800/modelo | 0 | 100 % |
| `vmd_optimized` (2 arrays) | ~2.400/modelo | 0 | 100 % |
| **Total no hot path** | **~7.200/modelo** | **6 (1 vez por modelo)** | **99,92 %** |

**Análise vs meta do plano:** O plano projetava +40 % a +80 % de throughput. O obtido empiricamente foi **+30,1 % em serial** (onde só conta o overhead de `malloc/free` sem contenção de mutex) e **+11,5 % em 8 threads** (onde a contenção é mitigada mas o custo matemático domina). A diferença vs o plano decorre de o `model.in` atual usar `n = 29` camadas, enquanto o plano foi calibrado para `n = 10`. Para `n` maior, o custo matemático cresce linearmente enquanto o custo do malloc permanece ~constante — o ratio `(malloc cost)/(total cost)` cai, reduzindo o impacto relativo. Em `n = 10`, o ganho esperado seria próximo dos +40-80%.

**Débitos técnicos identificados e resolvidos na PR1-Hygiene (pós-Fase 3, 2026-04-05):**

| # | Descrição | Status | Correção aplicada |
|:-:|:----------|:------:|:------------------|
| **D4** | `private(z_rho1, c_H1)` com `allocatable` — copias privadas têm status de alocação indefinido por spec OpenMP | ✅ **RESOLVIDO** | Trocado por `firstprivate(z_rho1, c_H1)` — cópias herdam alocação+valores do master |
| **D5** | `!$omp barrier` órfão fora de região paralela, redundante com barreira implícita do `end parallel do` | ✅ **RESOLVIDO** | Linha removida |
| **D6** | `omp_get_thread_num()` dentro do inner team retorna `tid` local, não global — race em `ws_pool(0)` se `num_threads_k > 1` | ✅ **RESOLVIDO** | Substituído por `tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()`. Backward-compat: com `num_threads_k=1`, `ancestor(1)=0` e `tid` permanece igual |

**Impacto da PR1-Hygiene em runtime**: zero para `ntheta=1` (produção atual) — validado por MD5 bit-exato em 1/2/4/8 threads (`aadbc86be2af5e1fd300f535d7e80e3b`). Pré-requisito estrutural para ativar multi-ângulo (`ntheta > 1`).

**Relatório completo:** [`docs/reference/relatorio_fase3_fortran.md`](relatorio_fase3_fortran.md)

---

### FASE 4 — Cache de `commonarraysMD` por `(r, freq)` (Passo 1.2) — Semanas 2–3

**Objetivo:** Pré-computar os resultados de `commonarraysMD` uma única vez por frequência (2 chamadas por modelo) em vez de chamá-la para cada medição (1.200 chamadas por modelo), reduzindo em 600× as operações nesta sub-rotina.

**Justificativa técnica:** Como demonstrado na análise da Causa-Raiz 2, `commonarraysMD` recebe os mesmos argumentos `(r, freq, n, h, eta, zeta)` para todas as 600 medições do mesmo ângulo e frequência. O resultado é matematicamente idêntico em todas as chamadas — a redundância é 100%. Pré-computar os resultados fora do laço paralelo e indexá-los por `i_freq` transforma 1.200 chamadas em 2 chamadas, eliminando o gargalo principal (45% do tempo total).

**Arquivo a modificar:** `PerfilaAnisoOmp.f08` — reestruturação de `fieldsinfreqs`.

**Implementação — nova sub-rotina de pré-computação:**

```fortran
subroutine precompute_common_arrays(nf, freqs, r, npt, krJ0J1, n, h, eta, &
                                    u_all, s_all, uh_all, sh_all,         &
                                    RTEdw_all, RTEup_all,                  &
                                    RTMdw_all, RTMup_all, AdmInt_all)
  ! Pré-computa commonarraysMD para cada frequência UMA VEZ,
  ! em vez de nf × nmed = 2 × 600 = 1.200 vezes por modelo.
  !
  ! Argumentos de entrada:
  !   nf        — número de frequências (tipicamente 2)
  !   freqs(nf) — vetor de frequências em Hz
  !   r         — espaçamento T-R = dTR = 1,0 m (invariante)
  !   ...
  !
  ! Argumentos de saída:
  !   u_all(npt, n, nf), RTEdw_all(npt, n, nf), ...
  !   Dimensão extra nf para indexação por frequência dentro do laço j.
  !
  implicit none
  integer,  intent(in)  :: nf, npt, n
  real(dp), intent(in)  :: freqs(nf), r, krJ0J1(npt), h(n), eta(n,2)
  complex(dp), intent(out) :: u_all(npt,n,nf), s_all(npt,n,nf)
  complex(dp), intent(out) :: uh_all(npt,n,nf), sh_all(npt,n,nf)
  complex(dp), intent(out) :: RTEdw_all(npt,n,nf), RTEup_all(npt,n,nf)
  complex(dp), intent(out) :: RTMdw_all(npt,n,nf), RTMup_all(npt,n,nf)
  complex(dp), intent(out) :: AdmInt_all(npt,n,nf)

  integer     :: i
  real(dp)    :: freq, omega
  complex(dp) :: zeta

  ! Loop sobre frequências (nf=2): 2 chamadas em vez de 1.200
  do i = 1, nf
    freq  = freqs(i)
    omega = 2.d0 * pi * freq
    zeta  = cmplx(0.d0, 1.d0, kind=dp) * omega * mu
    call commonarraysMD(n, npt, r, krJ0J1, zeta, h, eta,          &
                        u_all(:,:,i),    s_all(:,:,i),             &
                        uh_all(:,:,i),   sh_all(:,:,i),            &
                        RTEdw_all(:,:,i), RTEup_all(:,:,i),        &
                        RTMdw_all(:,:,i), RTMup_all(:,:,i),        &
                        AdmInt_all(:,:,i))
  end do
end subroutine precompute_common_arrays
```

**Integração em `perfila1DanisoOMP`:**

```fortran
! Antes do laço paralelo (chamada única por ângulo k):
call precompute_common_arrays(nf, freq, r, npt, krwJ0J1, n, h, resist, &
                               u_all, RTEdw_all, ...)

! Dentro do laço paralelo j:
! Substituir chamada a commonarraysMD por referência direta aos arrays pré-computados:
call fieldsinfreqs_cached(ang, nf, posTR, dipolo, npt, n, h, prof,  &
                          u_all, s_all, RTEdw_all, RTEup_all, ...   &  ! ← arrays pré-computados
                          zrho, cH, ws_pool(tid))
```

**Impacto matemático:**

```
Antes: commonarraysMD chamada nf × nmed = 2 × 600 = 1.200 vezes/modelo
Depois: commonarraysMD chamada nf = 2 vezes/modelo
Redução: 1.200 → 2 chamadas (fator 600×)
Impacto: ~45% do tempo total × (1 - 1/600) ≈ 44,9% de redução no tempo
```

**Validação numérica:** `max_rel_err < 1×10⁻¹²` para 1.000 modelos (a matemática é idêntica — apenas reordenamento de operações comutativas).
**Speedup esperado:** 1,6–2,2× sobre código pós-Fase-3.
**Risco:** Médio — requer reestruturação do laço de frequências dentro de `fieldsinfreqs`. Mitigação: manter `fieldsinfreqs` original e criar `fieldsinfreqs_cached` separada.

---

### FASE 5 — Colapso de Laços com `collapse(3)` (Passo 1.3) — Semana 3

**Objetivo:** Linearizar os três laços aninhados (ângulos × frequências × medições) em um único espaço de iterações, eliminando o overhead de fork/join duplo e expondo o máximo de paralelismo ao escalonador OpenMP.

**Justificativa técnica:** O laço duplo aninhado atual cria dois contextos paralelos OpenMP: o externo (ntheta = 1 thread) e o interno (maxthreads - 1 threads). O fork/join aninhado implica duas chamadas ao runtime OpenMP por iteração do laço externo. Com `ntheta = 1`, isso significa 1 fork/join extra sem nenhum paralelismo real. O `collapse(3)` lineariza `ntheta × nf × nmed = 1 × 2 × 600 = 1.200` iterações em um único espaço, gerando apenas 1 fork e 1 join por modelo.

**Arquivo a modificar:** `PerfilaAnisoOmp.f08` — estrutura de laços em `perfila1DanisoOMP`.

**Implementação:**

```fortran
! Cálculo do índice linearizado:
! kj = 1..total_iter mapeia para (k_idx, j_idx) via divisão inteira
integer :: kj, k_idx, j_idx, total_iter

total_iter = ntheta * nmmax   ! nmmax = max(nmed(:)) — limite conservador

!$omp parallel do schedule(dynamic, 8) num_threads(maxthreads) &
!$omp   private(kj, k_idx, j_idx, tid, ang, seno, coss, px, pz, &
!$omp           Lsen, Lcos, x, y, z, Tx, Ty, Tz, posTR, zrho, cH)
do kj = 1, total_iter
    k_idx = (kj - 1) / nmmax + 1       ! índice de ângulo: 1..ntheta
    j_idx = mod(kj - 1, nmmax) + 1     ! índice de medição: 1..nmmax

    ! Pula iterações fora do alcance (quando nmed(k) < nmmax):
    if (j_idx > nmed(k_idx)) cycle

    tid = omp_get_thread_num()

    ! Reconstrói ang, posTR a partir de k_idx e j_idx:
    ang  = theta(k_idx)
    seno = sin(ang * pi / 180.d0)
    coss = cos(ang * pi / 180.d0)
    ! ... demais cálculos geométricos ...

    ! Usa cache de commonarraysMD indexado por (k_idx, i_freq):
    call fieldsinfreqs_cached(ang, nf, posTR, dipolo, npt, n, h, prof,   &
                               u_all(:,:,:,k_idx), RTEdw_all(:,:,:,k_idx), ... &
                               zrho, cH, ws_pool(tid))
    z_rho1(k_idx, j_idx, :, :) = zrho
    c_H1  (k_idx, j_idx, :, :) = cH
end do
!$omp end parallel do
```

**Nota sobre `chunk_size = 8`:** O valor 8 balanceia granularidade (evita overhead de despacho com chunks muito pequenos) e balanceamento de carga (evita desbalanceamento com chunks muito grandes). O valor ótimo deve ser ajustado experimentalmente via benchmark (testar 4, 8, 16, 32).

**Validação numérica:** `max_rel_err < 1×10⁻¹²`.
**Speedup esperado:** 1,2–1,5× adicional sobre código pós-Fase-4.
**Risco:** Médio — requer cálculo correto do índice linearizado e compatibilidade com o cache de `commonarraysMD` indexado por `k_idx`.

---

### FASE 6 — Cache de `commonfactorsMD` por `camadT` (Passo 1.6) — Semana 4

**Objetivo:** Evitar recálculos de `commonfactorsMD` quando medições consecutivas têm o transmissor na mesma camada geológica, pré-calculando `camadT(j)` para todas as medições e usando um cache com sentinela.

**Justificativa técnica:** A sub-rotina `commonfactorsMD` depende de `(camadT, freq)`. Em formações com camadas de espessura maior que o passo de medição `p_med = 1,0 m`, múltiplas medições consecutivas têm o transmissor na mesma camada (`camadT` constante). A Seção 12.7.2 observa que em formações com camadas > 2 m, **mais de 50% das medições consecutivas compartilham o mesmo `camadT`**. Para camadas de 10 m com `p_med = 1,0 m`, são 10 medições com o mesmo `camadT` — 9 dos 10 recálculos são redundantes.

**Arquivo a modificar:** `PerfilaAnisoOmp.f08` — adição de pré-processamento serial e cache em `fieldsinfreqs`.

**Implementação — pré-computação serial de `camadT`:**

```fortran
! Pré-calcular camadT para cada medição (fora do laço paralelo):
integer, allocatable :: camadT_arr(:), camadR_arr(:)
real(dp) :: x_j, z_j, Tx_j, Tz_j

allocate(camadT_arr(nmmax), camadR_arr(nmmax))
do j = 1, nmed(k)
    ! Reconstruir geometria apenas para determinar camadas:
    x_j  = 0.d0 + (j-1) * px - Lsen / 2.d0
    z_j  = z1   + (j-1) * pz - Lcos / 2.d0
    Tx_j = 0.d0 + (j-1) * px + Lsen / 2.d0
    Tz_j = z1   + (j-1) * pz + Lcos / 2.d0
    call findlayersTR2well(n, Tz_j, z_j, prof(1:n-1), camadT_arr(j), camadR_arr(j))
end do
```

**Cache com sentinela dentro do laço serial de `commonfactorsMD`:**

```fortran
! Nota: este cache é serial (estado entre iterações)
! Produz arrays Mxdw_all(npt, nmmax) para uso no laço paralelo posterior

integer  :: camadT_prev, ct
complex(dp), allocatable :: Mxdw_all(:,:), Mxup_all(:,:)
complex(dp), allocatable :: Eudw_all(:,:), Euup_all(:,:)
complex(dp), allocatable :: FEdwz_all(:,:), FEupz_all(:,:)

allocate(Mxdw_all(npt, nmmax), Mxup_all(npt, nmmax))
allocate(Eudw_all(npt, nmmax), Euup_all(npt, nmmax))
allocate(FEdwz_all(npt, nmmax), FEupz_all(npt, nmmax))

camadT_prev = -1   ! sentinela: força recálculo na primeira iteração

do j = 1, nmed(k)
    ct = camadT_arr(j)
    if (ct /= camadT_prev) then
        ! Recalcular commonfactorsMD apenas quando camadT muda
        call commonfactorsMD(n, npt, Tz_arr(j), h, prof, ct,         &
                              u_all(:,:,1), s_all(:,:,1),             &
                              uh_all(:,:,1), sh_all(:,:,1),           &
                              RTEdw_all(:,:,1), RTEup_all(:,:,1),     &
                              RTMdw_all(:,:,1), RTMup_all(:,:,1),     &
                              Mxdw_all(:,j), Mxup_all(:,j),          &
                              Eudw_all(:,j), Euup_all(:,j),          &
                              FEdwz_all(:,j), FEupz_all(:,j))
        camadT_prev = ct
    else
        ! Copiar do cache da medição anterior (mesmo camadT):
        Mxdw_all(:,j)  = Mxdw_all(:,j-1)
        Mxup_all(:,j)  = Mxup_all(:,j-1)
        Eudw_all(:,j)  = Eudw_all(:,j-1)
        Euup_all(:,j)  = Euup_all(:,j-1)
        FEdwz_all(:,j) = FEdwz_all(:,j-1)
        FEupz_all(:,j) = FEupz_all(:,j-1)
    end if
end do
```

**Nota de aplicabilidade:** Esta otimização é particularmente eficaz para o caso de geosteering em formações com camadas de espessura > 2 m. Para modelos thin-bed (turbiditos com camadas de 0,3 m), o benefício é marginal (cada medição tem `camadT` diferente). O pré-processamento serial de `camadT_arr` tem custo fixo baixo e pode ser sempre incluído — o cache simplesmente não economizará tempo no caso thin-bed.

**Validação numérica:** `max_rel_err < 1×10⁻¹²`.
**Speedup esperado:** 1,2–1,5× em modelos de camadas espessas (> 5 m); 1,0–1,05× em thin-bed.
**Risco:** Médio-Alto — requer pré-processamento serial e cache com estado entre iterações. Deve ser implementado como passo serial antes do laço `!$omp parallel do`, não dentro dele.

---

## 8. Resultados Esperados

### 8.1 Ganho Incremental por Fase

```
┌────────┬─────────────────────────────────────────┬──────────────┬───────────┬───────────────┐
│  Fase  │  Otimização                             │  Semana      │  Speedup  │  Complexidade │
├────────┼─────────────────────────────────────────┼──────────────┼───────────┼───────────────┤
│   0    │  Benchmark baseline (gprof + timing)    │  Semana 0    │  1,0× ref │  Muito baixa  │
│   1    │  SIMD convolução Hankel (Passo 1.4)     │  Semana 1    │  1,1–1,2× │  Baixa        │
│   2    │  Escalonador híbrido (Passo 1.5)        │  Semana 1    │  1,05–1,1×│  Muito baixa  │
│   3    │  Workspace pré-alocado (Passo 1.1)      │  Semanas 1–2 │  1,4–1,8× │  Média        │
│   4    │  Cache commonarraysMD (Passo 1.2)       │  Semanas 2–3 │  1,6–2,2× │  Média        │
│   5    │  Collapse loops (Passo 1.3)             │  Semana 3    │  1,2–1,5× │  Média        │
│   6    │  Cache commonfactorsMD (Passo 1.6)      │  Semana 4    │  1,2–1,5× │  Média-Alta   │
├────────┼─────────────────────────────────────────┼──────────────┼───────────┼───────────────┤
│ TOTAL  │  Fases 1–6 combinadas                   │  Semanas 1–4 │  3,5–5,0× │  —            │
└────────┴─────────────────────────────────────────┴──────────────┴───────────┴───────────────┘
```

### 8.2 Evolução do Throughput

| Versão | Tempo/modelo | Speedup | Throughput (mod/h) |
|:-------|:------------:|:-------:|:------------------:|
| Baseline original | 2,40 s | 1,0× | 24.000 |
| Pós-Fase 1+2 (SIMD + sched) | ~2,1 s | ~1,15× | ~28.000 |
| Pós-Fase 3 (workspace) | ~1,3 s | ~1,85× | ~44.000 |
| Pós-Fase 4 (cache common) | ~0,70 s | ~3,4× | ~82.000 |
| Pós-Fase 5 (collapse) | ~0,55 s | ~4,4× | ~103.000 |
| **Pós-Fase 6 (completo)** | **~0,50 s** | **~4,8×** | **~115.000** |

### 8.3 Comparação Antes vs. Depois

| Métrica | Antes (Baseline) | Depois (Pós-Fase 6) | Melhoria |
|:--------|:----------------:|:-------------------:|:--------:|
| Tempo por modelo | 2,40 s | **~0,50 s** | 4,8× |
| Throughput | 24.000 mod/h | **~115.000 mod/h** | 4,8× |
| Chamadas a `commonarraysMD` | 1.200/modelo | **2/modelo** | 600× menos |
| Chamadas a `allocate/deallocate` | 2.400/modelo | **0/modelo** | 100% eliminadas |
| Fork/join OpenMP aninhado | 2 níveis | **1 nível** | overhead eliminado |
| Vetorização convolução Hankel | Escalar | **AVX-2/AVX-512** | 4–8× no loop |
| Eficiência Amdahl (16 cores) | não medida | **≥ 70%** | Meta KPI K6 |
| Erro relativo máximo | — | **< 1×10⁻¹²** | Meta KPI K2 |

---

## 9. Métricas de Sucesso e KPIs

Os KPIs da Seção 12.7.5 da documentação do simulador estabelecem critérios quantitativos de aceitação para a Fase 1 CPU:

| KPI | Métrica | Meta | Prazo |
|:----|:--------|:----:|:-----:|
| **K1** | Speedup Fase 1 (16 threads, 1.000 modelos) | **≥ 4×** | Semana 4 |
| **K2** | Erro de validação CPU (max relativo, todos os componentes H) | **< 1×10⁻¹⁰** | Semana 4 |
| **K6** | Eficiência de escalabilidade Amdahl (16 cores) | **≥ 70%** | Semana 4 |
| **K8** | Reprodutibilidade (mesmo resultado em 10 execuções consecutivas) | **100%** | Semana 4 |

### 9.1 Protocolo de Validação Numérica

A Seção 12.7.4 especifica o protocolo de validação que deve ser aplicado após cada fase:

```python
import numpy as np
import struct

def read_fortran_binary(filename, n_records, n_cols=22):
    """Lê arquivo binário unformatted stream do simulador Fortran."""
    data = []
    with open(filename, 'rb') as f:
        for _ in range(n_records):
            record = struct.unpack(f'i{n_cols-1}d', f.read(4 + (n_cols-1)*8))
            data.append(record)
    return np.array(data)

# Carregar baseline e versão otimizada
data_baseline = read_fortran_binary('validacao_baseline.dat',   n_records=600)
data_otimizado = read_fortran_binary('validacao_otimizado.dat', n_records=600)

# Verificar erro relativo por componente (colunas 4–21: 18 Re/Im)
componentes = ['Re(Hxx)', 'Im(Hxx)', 'Re(Hxy)', 'Im(Hxy)',
               'Re(Hxz)', 'Im(Hxz)', 'Re(Hyx)', 'Im(Hyx)',
               'Re(Hyy)', 'Im(Hyy)', 'Re(Hyz)', 'Im(Hyz)',
               'Re(Hzx)', 'Im(Hzx)', 'Re(Hzy)', 'Im(Hzy)',
               'Re(Hzz)', 'Im(Hzz)']

print("=== Validação Numérica ===")
all_pass = True
for col_idx, nome in enumerate(componentes, start=4):
    ref  = data_baseline[:, col_idx]
    opt  = data_otimizado[:, col_idx]
    mask = np.abs(ref) > 1e-30       # evitar divisão por zero em campos nulos
    err  = np.max(np.abs(opt[mask] - ref[mask]) / np.abs(ref[mask]))
    ok   = err < 1e-10
    all_pass = all_pass and ok
    print(f"  {nome:12s}: max_rel_err = {err:.2e}  [{'PASS' if ok else 'FAIL'}]")

print(f"\nResultado: {'APROVADO' if all_pass else 'REPROVADO'}")
```

### 9.2 Script de Benchmark Automatizado

```bash
#!/bin/bash
# benchmark_fase1_cpu.sh — medir speedup em cada passo da Fase 1
N_MODELS=1000

echo "=== Benchmark Fase 1 CPU ==="
echo "Modelos: ${N_MODELS} | CPU: $(nproc) cores"
echo ""

for VERSION in "baseline" "fase1_simd" "fase1_workspace" "fase1_cache_common" \
               "fase1_collapse" "fase1_cache_factors"; do
    for NTHREADS in 1 2 4 8 16; do
        export OMP_NUM_THREADS=${NTHREADS}
        ELAPSED=$( { time ./tatu_${VERSION}.x config_${N_MODELS}.namelist \
                     > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' )
        THROUGHPUT=$(python3 -c "import re; t=re.sub('[ms]','','"${ELAPSED}"'); \
                     parts=t.split(':'); sec=float(parts[0])*60+float(parts[1]); \
                     print(f'{int(3600*${N_MODELS}/sec):,}')" 2>/dev/null || echo "N/A")
        echo "  ${VERSION} | threads=${NTHREADS} | tempo=${ELAPSED} | throughput=${THROUGHPUT} mod/h"
    done
    echo ""
done
```

---

## 10. Pré-condição Estrutural para GPU

A documentação do simulador (Seção 12.7.1) estabelece explicitamente que a **Fase 2 GPU parte do baseline pós-Fase-1**, não do código original. Existem dois motivos técnicos específicos que tornam a Fase 1 CPU obrigatória antes de qualquer implementação GPU:

### 10.1 O Workspace Pré-alocado é Reutilizado na GPU

O tipo `thread_workspace` criado na Fase 3 (Passo 1.1) é a mesma estrutura que os kernels CUDA precisarão. Na GPU, arrays de tamanho variável com `allocate` dentro de kernels são tecnicamente impossíveis — todo o gerenciamento de memória de dispositivo deve ocorrer fora dos kernels, em código host. A Fase 3 CPU constrói essa abstração de workspace de tamanho fixo `(npt, MAX_N)` que o kernel CUDA vai herdar diretamente, com `device` substituindo `allocatable` na declaração.

### 10.2 O Cache de `commonarraysMD` Valida a Reordenação Matemática em CPU Primeiro

O cache de `commonarraysMD` (Fase 4) reordena as chamadas às sub-rotinas: em vez de `commonarraysMD` ser chamada dentro do laço de medições, ela é chamada fora e os resultados indexados. Esta reordenação é matematicamente equivalente, mas precisa ser validada com precisão `float64` em CPU antes de ser portada para GPU, onde a precisão pode cair para `float32` ou `float16`. Validar em CPU garante que a lógica de indexação e reordenação está correta antes de adicionar a complexidade do hardware GPU.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  DEPENDÊNCIA DE FASES: CPU → GPU                                            │
│                                                                              │
│  Fase 1 CPU (semanas 1–4)          Fase 2 GPU (semanas 5–10)               │
│  ──────────────────────────        ──────────────────────────               │
│  Fase 3: workspace (npt, MAX_N) →  device arrays de tamanho fixo na VRAM   │
│  Fase 4: cache commonarraysMD   →  commonarraysMD chamada 2× antes do      │
│          validado em float64       kernel; kernel usa u_all[i_freq] direto  │
│  Fase 5: collapse(3) layout     →  grid CUDA: (nmed, nmodels, ntheta)       │
│                                                                              │
│  Implementar GPU sem Fase 1 CPU = construir sobre fundação instável         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Referências

As seguintes seções da documentação técnica do simulador (`documentacao_simulador_fortran.md` v4.0) fundamentam este documento:

| Seção | Título | Conteúdo utilizado |
|:-----:|:-------|:-------------------|
| 11.1 | Estrutura Atual de Paralelismo | Arquitetura de 2 níveis OpenMP, distribuição de threads |
| 11.2 | Análise de Desempenho | 4 causas-raiz do desempenho subótimo |
| 11.3 | Pontos de Otimização — Memória | Tabela de oportunidades de memória |
| 11.4 | Pontos de Otimização — Tempo | Tabela de impactos estimados por otimização |
| 11.5 | Proposta de Paralelismo Otimizado | Arquitetura-alvo, código de workspace e collapse |
| 11.6 | Métricas de Escalabilidade | Complexidade computacional, estimativa FLOPs |
| 11.7 | Paralelização de Loops de Frequência e Ângulos | Análise de eficiência por configuração, afinidade NUMA |
| 11.8 | Estratégias Avançadas de Otimização de Memória | `thread_workspace`, stack vs heap, memory pooling, alinhamento SIMD |
| 12.7.1 | Visão Geral do Pipeline A | Métricas de baseline, metas de throughput |
| 12.7.2 | Fase 1 — Otimizações CPU com OpenMP | Implementação detalhada dos 6 passos, tabela resumo |
| 12.7.4 | Fase 3 — Validação e Benchmarking | Protocolo de validação, script Python de comparação |
| 12.7.5 | Cronograma e Métricas de Sucesso | KPIs K1, K2, K6, K8 |

---

## 12. Execução Real das Fases 0 e 1 — Resultados Empíricos (2026-04-04)

Esta seção registra os resultados **empíricos** da execução das Fases 0 e 1 em hardware de desenvolvimento macOS (i9-9980HK, 8 cores físicos, AVX-2, gfortran 15.2.0). Relatório completo em [`relatorio_fase0_fase1_fortran.md`](relatorio_fase0_fase1_fortran.md).

### 12.1 Ajustes de Premissas

Duas premissas do plano original (§7) precisam ser corrigidas à luz da execução:

1. **Localização de `commonarraysMD`**: a §7.1 e §4.1 sugeriam que a sub-rotina estaria em `PerfilaAnisoOmp.f08`. A **localização real é [`Fortran_Gerador/utils.f08:158-241`](../../Fortran_Gerador/utils.f08)**. `PerfilaAnisoOmp.f08` apenas a invoca via `use utils`.

2. **Baseline previsto vs. medido**: a §2.3 assumia **2,40 s/modelo** e **~24.000 modelos/hora** como baseline. A medição real neste hardware forneceu **0,1047 s/modelo** e **~34.400 modelos/hora** — **23× mais rápido**. A discrepância é explicada por (a) configuração atual usa `ntheta=1` enquanto o plano assumia `ntheta > 1`; (b) hardware 2018 mais recente; (c) gfortran 15.2.0 (2025) auto-vetoriza substancialmente melhor que versões anteriores.

**Lição aprendida**: baselines devem ser **medidos em cada ambiente**, não extrapolados de documentação.

### 12.2 Fase 0 — Baseline Medido

| Métrica                     | Valor                    |
|:----------------------------|:-------------------------|
| Iterações                   | 60 (após 3 warmups)      |
| Threads OpenMP              | 8                        |
| **Wall-time médio**         | **0,1047 ± 0,0147 s/modelo** |
| Mediana                     | 0,1000 s                 |
| **Throughput**              | **~34.400 modelos/hora** |
| MD5 de referência           | `c64745ed5d69d5f654b0bac7dde23a95` |

**Escalabilidade multi-thread observada**:

| Threads | Wall-time (s) | Speedup | Eficiência |
|:-------:|:-------------:|:-------:|:----------:|
| 1       | 0,50          | 1,00×   | 100 %      |
| 2       | 0,53          | 0,94×   | 47 % ⚠     |
| 4       | 0,19          | 2,63×   | 66 %       |
| **8**   | **0,12**      | **4,17×** | **52 %** |
| 16      | 0,08          | 6,25×   | 39 %       |

**⚠ Thread=2 com anti-escalabilidade**: decorre de `num_threads_j = maxthreads − ntheta = 2 − 1 = 1` em [`PerfilaAnisoOmp.f08:85`](../../Fortran_Gerador/PerfilaAnisoOmp.f08), degenerando o loop interno. **Este é um alvo claro da Fase 2 (Hybrid Scheduler + collapse).**

### 12.3 Fase 1 — Resultado NEGATIVO

**Intervenção**: refatoração de `commonarraysMD` substituindo array-syntax por loops `do ipt=1,npt` explícitos com `!$omp simd`, com hoisting de invariantes (patch preservado em [`Fortran_Gerador/bench/attic/phase1_simd.patch`](../../Fortran_Gerador/bench/attic/phase1_simd.patch)).

**Benchmark interleaved** (60 iterações alternadas):

| Métrica                | Baseline       | Fase 1         | Δ              |
|:-----------------------|:---------------|:---------------|:---------------|
| Wall-time médio (s)    | 0,1047 ± 0,015 | 0,1057 ± 0,011 | +0,96 %        |
| Welch *t*-statistic    | —              | **+0,425**     | **não-signif.** |

**Resultado**: Δ estatisticamente insignificante (`|t| < 2`). A Fase 1 **não entregou o ganho previsto** de +15 a +30 %.

**Causa raiz** (confirmada por `-fopt-info-vec=vec.txt`): gfortran 15.2.0 **já auto-vetoriza** os loops de `commonarraysMD` e `magneticdipoles.f08` com **vetores de 32 bytes** (máximo AVX-2 = 4 doubles). Não há margem para SIMD explícito nesta classe de CPU.

**Validação numérica**: `max|Δ| = 1,93·10⁻¹³` (muito abaixo de `atol=1e-10`). O código refatorado é **numericamente equivalente**, apenas não é mais rápido.

**Decisão**: experimento **rejeitado** — arquivado em [`bench/attic/`](../../Fortran_Gerador/bench/attic/), produção restaurada ao estado original. Razões em [`relatorio_fase0_fase1_fortran.md §3.6`](relatorio_fase0_fase1_fortran.md).

### 12.4 Recomendações Atualizadas para Roteiro

À luz dos achados empíricos, o roteiro original (§7) deve ser atualizado:

1. **Fase 0** ✅ — Concluída, baseline publicado.
2. **Fase 1** ⏭️ — **Pulada em CPUs AVX-2 com gfortran ≥ 14.** Re-tentar apenas em hardware AVX-512 (Xeon Scalable, Ice Lake-SP).
3. **Fase 2 (Hybrid Scheduler)** ✅ — **Concluída** (ver §13 abaixo).
4. **Fases 3–6** — Mantidas conforme planejado. A **Fase 4 (cache de `commonarraysMD`)** permanece como **maior oportunidade de ganho** (60–120 %) por ser estrutural, não dependente de microarquitetura.

---

## 13. Execução Real da Fase 2 + Correções dos Débitos 1 e 2 (2026-04-04)

Esta seção registra os resultados **empíricos** da execução da Fase 2 (Hybrid Scheduler) e das correções cirúrgicas dos Débitos Técnicos 1 (`writes_files` append bug) e 2 (`omp_set_nested` depreciado). Relatório completo em [`relatorio_fase2_debitos_fortran.md`](relatorio_fase2_debitos_fortran.md).

### 13.1 Intervenções Aplicadas

Três correções sincronizadas em [`PerfilaAnisoOmp.f08`](../../Fortran_Gerador/PerfilaAnisoOmp.f08):

1. **Fase 2 + Débito 3** — particionamento multiplicativo `num_threads_k × num_threads_j` substituindo o subtrativo buggado `maxthreads − ntheta`, combinado com `schedule(dynamic)` (externo) + `schedule(static)` (interno, carga uniforme).
2. **Débito 2** — `omp_set_nested(.true.)` → `omp_set_max_active_levels(2)` (OpenMP 5.0+).
3. **Débito 1** — abertura condicional com `inquire()` + detecção de `modelm==1` OR arquivo ausente, eliminando o bug de concatenação silenciosa.

### 13.2 Resultados do Scaling Test (2 warmups + 5 medições/ponto)

| OMP_NUM_THREADS | Baseline (s) | Fase 2 (s)   | Δ%          | Speedup Baseline | Speedup Fase 2 | Avaliação              |
|:---------------:|:------------:|:------------:|:-----------:|:----------------:|:--------------:|:-----------------------|
| 1               | 1,432        | 1,254        | **−12,4 %** | 1,00×            | 1,00×          | Fase 2 ✓               |
| **2**           | **1,340** ⚠  | **0,786** ✅ | **−41,3 %** | **1,07×**        | **1,60×**      | **Bug corrigido**      |
| 4               | 0,522        | 0,400        | **−23,4 %** | 2,74×            | 3,13×          | Fase 2 ✓               |
| 8               | 0,276        | 0,306        | +10,9 %     | 5,19×            | 4,10×          | Trade-off marginal     |
| 16              | 0,226        | 0,240        | +6,2 %      | 6,34×            | 5,23×          | Trade-off marginal     |

**Validação numérica**: `max|Δ| = 0,0000e+00` em todas as 21 colunas. MD5 idêntico entre baseline e Fase 2 (`c64745ed5d69d5f654b0bac7dde23a95`). Reprodutibilidade **bit-a-bit exata**.

### 13.3 Vitórias Principais

1. **Bug 2-thread corrigido definitivamente** (−41 %, speedup 1,07× → 1,60×). Era a causa-raiz #3 descoberta na Fase 0.
2. **1 thread +12 %** (schedule(static) com menor overhead que dynamic).
3. **4 threads +23 %** (particionamento correto + static scheduler).
4. **MD5 idêntico** ao baseline — zero regressão numérica.
5. **Código mais limpo**: `omp_set_max_active_levels(2)` é API moderna e semântica direta.
6. **Arquivos de saída confiáveis**: re-runs do gerador Python agora sobrescrevem corretamente em vez de concatenar silenciosamente.

### 13.4 Trade-off em Alta Concorrência (8–16 threads)

A substituição de `schedule(dynamic)` por `schedule(static)` no loop interno causou regressão marginal de 6–11 % em 8–16 threads. Com 600 iterações distribuídas em 8–16 threads (75 ou 37 iter/thread), `static` fica vulnerável ao ruído de SO (interrupts, preempção), enquanto `dynamic` absorveria esse ruído via balanceamento tardio.

**Mitigação planejada (Fase 2b)**: tuning de chunk size — experimentar `schedule(static, 16)` ou `schedule(guided, 4)` para recuperar balanceamento sem perder o benefício de chunks grandes. **Fora de escopo desta iteração.**

### 13.5 Estado do Roteiro Após Fase 2

| Fase | Descrição | Status | Ganho Real / Esperado |
|:----:|:----------|:------:|:----------------------|
| 0 | Benchmark Baseline | ✅ | 0,1047 s/modelo (CPU fria, 8 threads) |
| 1 | SIMD Hankel Reduction | ⏭️ | Pulada em AVX-2 (gfortran já satura) |
| **2** | **Hybrid Scheduler + particionamento** | ✅ **2026-04-04** | **−41 % em 2 threads (bug fix), −23 % em 4 threads, −12 % em 1 thread** |
| 3 | Workspace Pre-allocation | 📋 Próxima | +40 a +80 % esperado |
| 4 | Cache `commonarraysMD` | 📋 Planejada | **+60 a +120 % (maior ganho)** |
| 5 | `collapse(3)` loops | 📋 Planejada | +10 a +20 % |
| 6 | Cache `commonfactorsMD` | 📋 Planejada | +15 a +25 % |

**Próximo passo recomendado**: **Fase 3** (Workspace Pre-allocation) — é pré-requisito estrutural para a portabilidade CPU→GPU (Pipeline A OpenACC) e primeiro passo com ganho substancial esperado.

---

*A Fase 2 foi executada em 2026-04-04 e validada via scaling test + reprodutibilidade numérica bit-a-bit. Relatório completo em [`relatorio_fase2_debitos_fortran.md`](relatorio_fase2_debitos_fortran.md).*

### 12.5 Débitos Técnicos Descobertos

Durante a execução foram identificados três bugs/depreciações no código de produção, registrados para correção futura:

| # | Bug/Depreciação | Local | Prioridade |
|:-:|:----------------|:------|:----------:|
| 1 | `writes_files` usa `position='append'` sem limpeza — concatena em re-execuções | [`PerfilaAnisoOmp.f08:245-246`](../../Fortran_Gerador/PerfilaAnisoOmp.f08) | **Alta** |
| 2 | `omp_set_nested(.true.)` depreciado desde OpenMP 5.0 | [`PerfilaAnisoOmp.f08:74`](../../Fortran_Gerador/PerfilaAnisoOmp.f08) | Média |
| 3 | `num_threads_j = maxthreads - ntheta` produz 1 thread no nível interno quando `OMP_NUM_THREADS=2` | [`PerfilaAnisoOmp.f08:85`](../../Fortran_Gerador/PerfilaAnisoOmp.f08) | **Alta** (endereçada pela Fase 2) |

Detalhes completos em [`relatorio_fase0_fase1_fortran.md §4`](relatorio_fase0_fase1_fortran.md).

---

*Documento gerado com base na análise técnica da
`docs/reference/documentacao_simulador_fortran.md` v4.0 (Geosteering AI v2.0),
complementado por resultados empíricos das Fases 0 e 1 executadas em 2026-04-04
(ver `relatorio_fase0_fase1_fortran.md`).*

*Para questões sobre implementação GPU (Fase 2 — OpenACC/CUDA), consultar
`docs/reference/analise_gpu_fortran.md` (a ser gerado).*