# Relatório Executivo — Fase 4: Cache de `commonarraysMD` por `(r, freq)` (Fortran)

| Campo | Valor |
|:------|:------|
| **Autor** | Daniel Leal (execução via Claude Code) |
| **Data** | 2026-04-05 |
| **Escopo** | `Fortran_Gerador/PerfilaAnisoOmp.f08` |
| **Referência** | [`analise_paralelismo_cpu_fortran.md` §7 Fase 4](analise_paralelismo_cpu_fortran.md) |
| **Roteiro** | Fase 4 de 6 (Cache commonarraysMD por (r, freq)) |
| **Fases anteriores** | Fase 0 (baseline), Fase 1 (arquivada), Fase 2 (Hybrid Scheduler), Fase 3 (Workspace Pre-alloc), PR1-Hygiene (D4/D5/D6) |
| **Commit pai** | `db997d2 fix(fortran): OpenMP hygiene — correcao de debitos D4/D5/D6` |

---

## 1. Resumo Executivo

A **Fase 4 — Cache de `commonarraysMD` por `(r, freq)`** eliminou **99,83 %** das chamadas a `commonarraysMD` por modelo (1.200 → `nf = 2`), explorando a invariância matemática `r = dTR × |sin(theta_k)|` (constante por ângulo devido à translação rígida da ferramenta LWD). O ganho empírico foi de **speedup 5,54× a 8 threads** (0,3433 s/modelo → 0,0620 s/modelo) — **muito acima** da meta documentada de 1,6–2,2×, indicando que `commonarraysMD` dominava o custo do simulador em ~82 % (bem mais que os 45 % sugeridos pelo profiling inicial). Throughput final: **58.064 modelos/h**, **242 %** da meta original do roteiro (24.000 modelos/h).

A validação numérica confirmou desvio máximo absoluto de **3,97 × 10⁻¹³** vs Fase 3 — três ordens de magnitude abaixo do critério `1 × 10⁻¹⁰`, compatível com reordenamento associativo autorizado por `-ffast-math`. As rotinas originais `fieldsinfreqs`, `fieldsinfreqs_ws` permanecem intactas para rollback instantâneo.

---

## 2. Mudanças de Código

### 2.1 `Fortran_Gerador/PerfilaAnisoOmp.f08`

**Adicionado** — 9 arrays de cache `allocatable` + hoist de `eta` em `perfila1DanisoOMP`:

```fortran
complex(dp), allocatable :: u_cache(:,:,:), s_cache(:,:,:)
complex(dp), allocatable :: uh_cache(:,:,:), sh_cache(:,:,:)
complex(dp), allocatable :: RTEdw_cache(:,:,:), RTEup_cache(:,:,:)
complex(dp), allocatable :: RTMdw_cache(:,:,:), RTMup_cache(:,:,:)
complex(dp), allocatable :: AdmInt_cache(:,:,:)
real(dp),    allocatable :: eta_shared(:,:)
```

Dimensão `(npt, n, nf)` para os caches + `(n, 2)` para `eta_shared`. Alocados uma única vez no início de `perfila1DanisoOMP` e desalocados ao final. Memória: `9 × 201 × 29 × 2 × 16 ≈ 1,68 MB` (n=29, nf=2) no heap — dominado pelo nível de cache L3 de CPUs modernas.

**Adicionado** — pré-cômputo serial dentro do loop `k`, antes do inner parallel:

```fortran
r_k = dTR * dabs(seno)
do ii = 1, nf
  omega_i = 2.d0 * pi * freq(ii)
  zeta_i  = cmplx(0.d0, 1.d0, kind=dp) * omega_i * mu
  call commonarraysMD(n, npt, r_k, krwJ0J1(:,1), zeta_i, h, eta_shared,   &
                      u_cache(:,:,ii),  s_cache(:,:,ii),                   &
                      uh_cache(:,:,ii), sh_cache(:,:,ii),                  &
                      RTEdw_cache(:,:,ii), RTEup_cache(:,:,ii),            &
                      RTMdw_cache(:,:,ii), RTMup_cache(:,:,ii),            &
                      AdmInt_cache(:,:,ii))
end do
```

A sanitização `if (hordist < eps) r = 1.d-2` da rotina `commonarraysMD` original é preservada — passamos `r_k` direto (que pode ser `0.0` quando `theta=0`), e a rotina aplica a proteção internamente, garantindo bit-equivalência.

**Adicionado** — nova subrotina `fieldsinfreqs_cached_ws` (~100 linhas) logo após `fieldsinfreqs_ws`:
- Recebe os 9 caches + `eta_in` como `intent(in)`.
- Delega para `hmd_TIV_optimized_ws` e `vmd_optimized_ws` passando slices `u_cache(:,:,i)` etc. — contíguas em column-major Fortran (sem cópia temporária, verificado em build).
- **commonfactorsMD permanece inline** pois depende de `camadT, Tz` (variáveis em `j`). Será tratada em Fase 6.
- **Preserva** `fieldsinfreqs_ws` original intacta — rollback em uma linha.

**Alterado** — chamada no inner parallel (1 linha de call):
```fortran
! Antes:
call fieldsinfreqs_ws(ws_pool(tid), ang, nf, freq, posTR, dipolo, npt, &
                      krwJ0J1, n, h, prof, resist, zrho, cH)

! Depois:
call fieldsinfreqs_cached_ws(ws_pool(tid), ang, nf, freq, posTR, dipolo, npt, &
                             krwJ0J1, n, h, prof, resist, eta_shared,         &
                             u_cache, s_cache, uh_cache, sh_cache,            &
                             RTEdw_cache, RTEup_cache,                         &
                             RTMdw_cache, RTMup_cache, AdmInt_cache,           &
                             zrho, cH)
```

Adicionado `default(shared)` explícito na diretiva `!$omp parallel do` interna para auto-documentação: todos os caches são compartilhados entre threads (somente leitura, sem race).

### 2.2 Contagem de chamadas a `commonarraysMD`

| Momento | Chamadas por modelo | Redução |
|:--------|:-------------------:|:-------:|
| Antes (Fase 3) | `nf × nmed = 2 × 600 = 1.200` | — |
| **Depois (Fase 4)** | `ntheta × nf = 1 × 2 = 2` | **99,83 %** |

Para a configuração de produção (`ntheta = 1, nf = 2, nmed = 600`), o fator de redução é exatamente **600×** — cada chamada computa os mesmos arrays `(npt, n)` que antes eram recomputados 600 vezes (uma por medição) e agora servem ao inner loop inteiro.

---

## 3. Validação Numérica

### 3.1 Metodologia

Comparação bit-a-bit dos arquivos `Inv0_15Dip1000_t5.dat` gerados por:
- **Fase 3 (commit c213b66)**: tatu.x com `fieldsinfreqs_ws` (commonarraysMD em cada iteração).
- **Fase 4**: tatu.x atual com `fieldsinfreqs_cached_ws` (commonarraysMD hoisted).
- **Mesmo `model.in`**: 29 camadas, 2 frequências (20/40 kHz), `ntheta=1`, 1.200 registros.
- **Mesmo host**: Intel i9-9980HK, macOS Darwin 25.4, gfortran 15.2 + ld-classic.
- **Mesmas flags**: `-O3 -march=native -ffast-math -funroll-loops -fopenmp -std=f2008`.
- **Mesmas threads**: `OMP_NUM_THREADS=8`.

### 3.2 Resultados

| Métrica | Valor | Critério (plano) | Status |
|:--------|:-----:|:----------------:|:------:|
| **max \|Δ\|** | **3,97 × 10⁻¹³** | ≤ 10⁻¹⁰ | ✅ 3 ordens de magnitude abaixo |
| **RMS(Δ)** | 2,02 × 10⁻¹⁴ | (ruído ULP) | ✅ |
| **max rel Δ** | 1,01 × 10⁻⁸ | (valores próximos a zero) | ✅ |
| **Linhas comparadas** | 1.200 × 21 = 25.200 valores | — | — |
| **Colunas inteiras** | todas idênticas | identidade | ✅ |
| **Determinismo entre threads** | MD5 idêntico em 1/2/4/8 threads | — | ✅ |

### 3.3 Interpretação

A diferença máxima de `3,97e-13` ocorre em um valor absoluto de ordem `~0,08` (linha 811, coluna 4 — `Re(Hxx)` de uma medição próxima a uma interface de alto contraste). O desvio relativo é `~4,7e-12`, compatível com:
1. **Reordenamento associativo** permitido por `-ffast-math`: operações como `a * b + c * d` podem ser fundidas em `FMA` ou reordenadas.
2. **Mudança no layout de registradores**: chamar `commonarraysMD` uma vez fora vs 600 vezes dentro altera o pressure de registradores e o scheduling de instruções pelo `-O3`.
3. **ULP de `double`** na faixa 0,08: `~1,1e-17`, portanto `3,97e-13` representa ~36.000 ULPs acumulados ao longo de 600 iterações somadas e produtos — totalmente dentro do esperado.

Em modo `-O0` (sem reordenamento), a saída seria bit-exata com Fase 3 — a matemática é idêntica, apenas a ordem de execução muda.

---

## 4. Benchmark

### 4.1 Ambiente

- **Hardware**: Intel Core i9-9980HK (8 cores físicos, 16 lógicos), 32 GB RAM
- **SO**: macOS Darwin 25.4.0 (macOS 26), Homebrew gfortran 15.2.0
- **Flags**: `-O3 -march=native -ffast-math -funroll-loops -fopenmp -std=f2008`
- **Linker**: `ld-classic` (workaround macOS 26)
- **Config**: `model.in` com 29 camadas, 2 frequências (20/40 kHz), 600 medidas/modelo
- **Protocolo**: 60 iterações a 8 threads; 30 iterações por ponto para scaling 1/2/4 threads

### 4.2 Resultados a 8 threads (60 iterações)

| Métrica | Fase 2 (HEAD) | Fase 3 | **Fase 4** | Delta 4 vs 3 |
|:--------|:-------------:|:------:|:----------:|:------------:|
| Tempo médio (s/modelo) | 0,3830 | 0,3433 | **0,0620** | **−81,9 %** |
| Desvio-padrão (s) | 0,0355 | 0,0278 | 0,0058 | −79,1 % |
| Throughput (modelos/h) | 9.399 | 10.485 | **58.064** | **+453,8 %** |

### 4.3 Escalabilidade 1 → 8 threads

| Threads | Fase 2 tempo (s) | Fase 3 tempo (s) | **Fase 4 tempo (s)** | Fase 4 thput (mod/h) | **Speedup 4 vs 3** |
|:-------:|:----------------:|:----------------:|:--------------------:|:--------------------:|:------------------:|
| **1**   | 1,7800 | 1,3690 | **0,2393** | 15.042 | **5,72×** |
| **2**   | 0,8073 | 0,7500 | **0,1390** | 25.899 | 5,40× |
| **4**   | 0,5873 | 0,4617 | **0,0820** | 43.902 | **5,63×** |
| **8**   | 0,3830 | 0,3433 | **0,0620** | 58.064 | 5,54× |

### 4.4 Speedup consolidado (vs Fase 2 baseline)

| Threads | Fase 2 → Fase 4 speedup | Throughput ganho |
|:-------:|:-----------------------:|:----------------:|
| 1 | **7,44×** | +644 % |
| 2 | 5,81× | +481 % |
| 4 | 7,16× | +616 % |
| 8 | **6,18×** | +518 % |

### 4.5 Análise dos resultados

- **Consistência do speedup ~5,5× entre thread counts**: indica que a economia é **dominada pelo custo bruto de `commonarraysMD`**, não por contenção de recurso paralelo. Cada thread executava as mesmas 1.200 chamadas por modelo, todas redundantes — eliminar 1.198 delas economiza ~82 % do tempo total, independente de contagem de threads.
- **Perfil revisado**: o profiling inicial (gprof pré-Fase 0) estimava `commonarraysMD` em ~45 % do tempo total. O ganho real de 82 % sugere que o profile original subestimava o custo (provavelmente por amostragem imprecisa do `gprof` com funções de curta duração) ou que as fases anteriores (Hybrid Scheduler, Workspace Pre-alloc) mudaram a distribuição relativa, tornando `commonarraysMD` ainda mais dominante.
- **Meta do roteiro atingida e ultrapassada**: a meta original era 24.000 modelos/h (documentada em `analise_paralelismo §6.4`). Fase 4 atinge **58.064 modelos/h**, **242 %** da meta — o simulador agora processa ~16,1 modelos por segundo em 8 threads, viabilizando datasets de 1.000 modelos em ~62 segundos.
- **Escalabilidade paralela saturada a partir de 4 threads**: 4t/1t = 2,92× (ideal 4,00×), 8t/1t = 3,86× (ideal 8,00×). A saturação é esperada em machine learning kernels e provavelmente reflete largura de banda de memória + sincronização no `findlayersTR2well`/`commonfactorsMD`. Fase 5 (`collapse(3)`) e Fase 6 (cache `commonfactorsMD`) podem melhorar.

### 4.6 Gargalo remanescente

Com `commonarraysMD` eliminada, o novo hot path é:
1. `commonfactorsMD` (~40-50 % do tempo restante) → **alvo da Fase 6**.
2. `hmd_TIV_optimized_ws` + `vmd_optimized_ws` (~30-40 %) → já otimizados em Fase 3.
3. Findlayers, setup, escrita binária (~10-15 %) → marginal.

---

## 5. Bugs e Débitos Identificados

### 5.1 Resolvidos nesta fase

- ✅ **B2 (hoist de `eta`)**: `eta(i,1) = 1/resist(i,1)` era recomputado a cada chamada de `fieldsinfreqs_ws`. Foi hoisted para `eta_shared` no escopo de `perfila1DanisoOMP`, eliminando `n × nf × nmed` divisões redundantes por modelo.

### 5.2 Pendentes (registrados, não corrigidos)

- **B1**: Cópia redundante `krJ0J1/wJ0/wJ1 = krwJ0J1(:,1..3)` em cada chamada de `fieldsinfreqs_cached_ws`. Economia marginal se removida.
- **B3 (D7)**: `private(zrho, cH)` com `allocatable` no inner parallel — mesma categoria de D4/D7, deveria ser `firstprivate`. Pré-requisito adicional para multi-ângulo robusto.
- **B5**: `krwJ0J1` alocado em `perfila1DanisoOMP` e nunca desalocado (leak de ~9,6 KB/modelo). Trivial.
- **B6**: Stride inconveniente de `zrho1(ntheta, nmmax, nf, 3)` para `writes_files`. Penalidade de cache em escrita.
- **B7**: Dummies `u, s, uh, sh, RTEdw, ...` em `hmd_TIV_optimized_ws, vmd_optimized_ws` sem atributo `contiguous` — migrar para maior segurança com slices de F2008.

Todos registrados para PR futuro de "OpenMP hygiene + code cleanup secundário".

---

## 6. Estado Atual do Simulador (pós-Fase 4)

| Fase | Status | Throughput (8 threads, n=29) | Speedup vs Fase 2 |
|:-----|:------:|:----------------------------:|:-----------------:|
| Fase 0 (baseline original) | ✅ arquivado | 34.400 mod/h (n=10) | referência histórica |
| Fase 1 (SIMD Hankel) | ⊘ arquivado | não mensurável | — |
| Fase 2 (Hybrid Scheduler) | ✅ `6ac51ca` | 9.399 mod/h | 1,00× |
| Fase 3 (Workspace Pre-alloc) | ✅ `c213b66` | 10.485 mod/h | 1,12× |
| **PR1 OpenMP Hygiene (D4/D5/D6)** | **✅ `db997d2`** | 10.485 (inalterado) | 1,12× |
| **Fase 4 (Cache commonarraysMD)** | **✅ este commit** | **58.064 mod/h** | **6,18×** |
| Fase 5 (`collapse(3)`) | 📋 futura | ganho marginal com ntheta=1 | — |
| Fase 6 (cache `commonfactorsMD`) | 📋 próxima | +40-60 % esperado sobre Fase 4 | — |

---

## 7. Próximos Passos Recomendados

1. **Fase 6 — Cache de `commonfactorsMD` por `camadT`**: novo gargalo dominante (~40-50 % do tempo pós-Fase-4). `commonfactorsMD` depende de `(camadT, freq)`, e em formações com camadas > 2 m, medições consecutivas compartilham o mesmo `camadT`. Estratégia: cache com sentinela `last_camadT_per_thread`, reuso quando `camadT` atual == prévio. Ganho esperado: +40-60 % sobre Fase 4.

2. **Fase 5 — `collapse(3)` nos loops `(ntheta × nf × nmed)`**: adiar até multi-ângulo ser ativado. Para `ntheta=1` atual, o ganho é marginal (fork/join extra único por modelo já custa < 1 ms).

3. **Correção dos débitos B1/B3/B5/B6/B7** em PR separado focado em OpenMP hygiene secundária + code cleanup.

4. **Teste em produção Linux**: rodar benchmark em servidor Linux (gfortran nativo, sem `ld-classic`). O ganho de Fase 4 deve ser preservado, possivelmente ainda maior sem o overhead do `ld-classic` workaround.

5. **Atualização do orquestrador Python `fifthBuildTIVModels.py`**: exibir a taxa atualizada de modelos/h (com throughput Fase 4 de ~58 k mod/h, 1.000 modelos levam ~1 minuto — ajustar mensagens de progresso para blocos menores).

6. **Rebaseline das curvas de convergência**: o dataset de treino do Deep Learning pode agora ser regenerado em <1 h em vez de ~2 h, reduzindo o ciclo de iteração de experimentos.

---

## 8. Arquivos Afetados

| Arquivo | Tipo | Mudanças |
|:--------|:----:|:---------|
| `Fortran_Gerador/PerfilaAnisoOmp.f08` | modificado | +~200 linhas: 9 caches + eta_shared + pré-cômputo no loop k + `fieldsinfreqs_cached_ws` + substituição no hot path |
| `Fortran_Gerador/bench/results/phase4_*.{md,txt}` | novo | relatórios de benchmark Fase 4 (60 iter 8t + scaling 1/2/4t) |
| `docs/reference/relatorio_fase4_fortran.md` | **novo** | este relatório |
| `docs/reference/analise_paralelismo_cpu_fortran.md` | atualizado | §7.4 com resultados empíricos |
| `docs/reference/documentacao_simulador_fortran.md` | atualizado | nova subseção sobre `fieldsinfreqs_cached_ws` |
| `docs/ROADMAP.md` | atualizado | F2.5.1 Fase 4 ✅ |

---

**Assinatura de validação:** Fase 4 concluída com ganho de **5,54× em 8 threads** (0,343 s → 0,062 s por modelo), validação numérica com `max|Δ| = 3,97 × 10⁻¹³` (3 ordens de magnitude abaixo do critério), zero warnings de compilação, MD5 determinístico em 1/2/4/8 threads. Meta do roteiro **ultrapassada em 242 %**. Pronta para commit em `origin/main`.
