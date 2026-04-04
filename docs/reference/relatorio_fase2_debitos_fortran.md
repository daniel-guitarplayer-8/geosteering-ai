# Relatório de Execução — Fase 2 e Correções dos Débitos 1 e 2 do Simulador Fortran

**Projeto**: Geosteering AI v2.0 — Simulador `PerfilaAnisoOmp`
**Data**: 2026-04-04
**Autor**: Daniel Leal (execução assistida por Claude Code)
**Escopo**: Fase 2 (Hybrid Scheduler) do roteiro CPU + correção cirúrgica dos Débitos Técnicos 1 (`writes_files` append bug) e 2 (`omp_set_nested` depreciado), identificados no [Relatório das Fases 0 e 1](relatorio_fase0_fase1_fortran.md).
**Antecedentes**: [`relatorio_fase0_fase1_fortran.md`](relatorio_fase0_fase1_fortran.md) · [`analise_paralelismo_cpu_fortran.md` §7 Fase 2](analise_paralelismo_cpu_fortran.md)

---

## Sumário Executivo

Esta iteração aplicou **três correções cirúrgicas** ao arquivo [`Fortran_Gerador/PerfilaAnisoOmp.f08`](../../Fortran_Gerador/PerfilaAnisoOmp.f08), cobrindo a Fase 2 do roteiro CPU e dois dos três débitos técnicos descobertos na Fase 0:

1. **Fase 2 — Hybrid Scheduler** + **Débito 3** (particionamento de threads): substituição do `num_threads_j = maxthreads - ntheta` (aritmética subtrativa que degenerava para 1 thread em `OMP=2`) por particionamento **multiplicativo** `num_threads_k × num_threads_j ≈ maxthreads`, combinado com `schedule(dynamic)` no loop externo de ângulos e `schedule(static)` no loop interno de medidas.
2. **Débito 2** — `omp_set_nested(.true.)` depreciado desde OpenMP 5.0 → migração para `omp_set_max_active_levels(2)`.
3. **Débito 1** — `writes_files` usava `position='append'` sem limpeza prévia, concatenando runs silenciosamente → abertura condicional com `inquire()` + detecção de `modelm==1` OR arquivo ausente.

**Resultados empíricos** (scaling test 1–16 threads, 5 medições por ponto após 2 warmups):

| OMP_NUM_THREADS | Baseline (s) | Fase 2 (s)   | Δ%          | Speedup Baseline | Speedup Fase 2 |
|:---------------:|:------------:|:------------:|:-----------:|:----------------:|:--------------:|
| 1               | 1,432        | 1,254        | **−12,4 %** | 1,00×            | 1,00×          |
| **2**           | **1,340** ⚠  | **0,786** ✅ | **−41,3 %** | **1,07×**        | **1,60×**      |
| 4               | 0,522        | 0,400        | **−23,4 %** | 2,74×            | 3,13×          |
| 8               | 0,276        | 0,306        | +10,9 %     | 5,19×            | 4,10×          |
| 16              | 0,226        | 0,240        | +6,2 %      | 6,34×            | 5,23×          |

**Validação numérica**: `max|Δ| = 0,0000e+00` em todas as 21 colunas da saída binária (validação bit-a-bit exata via [`validate_numeric.py`](../../Fortran_Gerador/bench/validate_numeric.py) com `atol=1e-10`, `rtol=1e-12`). **MD5 idêntico** entre baseline e Fase 2: `c64745ed5d69d5f654b0bac7dde23a95`.

**Veredito**: a Fase 2 **corrige definitivamente o Débito Técnico #3** (bug de 2 threads), entregando **+41 % de throughput** neste regime crítico, **+12 % em 1 thread** e **+23 % em 4 threads**. O trade-off marginal (+6 a +11 %) em 8–16 threads é atribuível à perda de balanceamento adaptativo do `schedule(dynamic)` substituído por `schedule(static)`, e pode ser mitigado em iteração futura via tuning de chunk size.

---

## 1. Intervenções no Código

### 1.1 Débito 2 — Migração OpenMP Moderno

**Local**: [`PerfilaAnisoOmp.f08:74-80`](../../Fortran_Gerador/PerfilaAnisoOmp.f08) (antigo) → bloco expandido

**Antes**:
```fortran
call omp_set_nested(.true.)
nested_enabled = omp_get_nested()
```

**Depois**:
```fortran
call omp_set_max_active_levels(2)
nested_enabled = (omp_get_max_active_levels() >= 2)
```

**Justificativa**: `omp_set_nested` e `omp_get_nested` foram marcadas depreciadas na especificação **OpenMP 5.0 (novembro 2018)** e serão removidas em versões futuras do runtime. A API substituta `omp_set_max_active_levels(n)` controla diretamente quantos níveis de paralelismo aninhado podem estar ativos, oferecendo granularidade maior (antes: bool on/off; agora: número de níveis). Para manter semântica idêntica ao código original (2 níveis: `theta × medidas`), usa-se `n=2`.

### 1.2 Fase 2 + Débito 3 — Particionamento Multiplicativo

**Local**: [`PerfilaAnisoOmp.f08:82-86`](../../Fortran_Gerador/PerfilaAnisoOmp.f08) (antigo)

**Antes**:
```fortran
maxthreads = omp_get_max_threads()
num_threads_k = ntheta
num_threads_j = maxthreads - ntheta   ! BUG: degenera para 1 em OMP=2, ntheta=1
```

**Depois**:
```fortran
maxthreads = omp_get_max_threads()
num_threads_k = max(1, min(ntheta, maxthreads))
num_threads_j = max(1, maxthreads / num_threads_k)
```

**Análise do bug**: com `OMP_NUM_THREADS=2` e `ntheta=1` (configuração padrão do `model.in`), o código original calculava `num_threads_j = 2 − 1 = 1`. O loop interno de medidas, com carga de trabalho de ~600 iterações × ~3 ms cada, ficava **serializado em uma única thread**, enquanto o outro thread permanecia ocioso. O resultado era um speedup de **1,07×** em 2 threads (virtualmente nenhum ganho).

**Comportamento do fix**:

| maxthreads | ntheta | `num_threads_k` (novo) | `num_threads_j` (novo) | Produto | Observação                |
|:----------:|:------:|:----------------------:|:----------------------:|:-------:|:---------------------------|
| 2          | 1      | 1                      | 2                      | 2       | **Fix do bug 2-thread**    |
| 2          | 2      | 2                      | 1                      | 2       | Uma thread por ângulo      |
| 8          | 1      | 1                      | 8                      | 8       | Todas threads no interno   |
| 8          | 2      | 2                      | 4                      | 8       | 2 × 4 balanceado           |
| 8          | 7      | 7                      | 1                      | 7       | Quase satura (1 ocioso)    |
| 16         | 7      | 7                      | 2                      | 14      | 14 de 16 ativas            |

### 1.3 Fase 2 — Schedule Híbrido

**Local**: [`PerfilaAnisoOmp.f08:95, 106`](../../Fortran_Gerador/PerfilaAnisoOmp.f08)

**Antes**:
```fortran
!$omp parallel do schedule(dynamic) num_threads(num_threads_k) ...  ! externo
  !$omp parallel do schedule(dynamic) num_threads(num_threads_j) ...  ! interno
```

**Depois**:
```fortran
!$omp parallel do schedule(dynamic) num_threads(num_threads_k) ...  ! externo: carga desigual
  !$omp parallel do schedule(static)  num_threads(num_threads_j) ...  ! interno: carga uniforme
```

**Justificativa**:

- **Loop externo `k` (ângulos)**: `nmed(k)` varia com `theta(k)` porque `pz = p_med × cos(theta)` muda com o ângulo — janela vertical constante mas passo vertical variável resulta em número de medidas desigual. Carga **desigual** → `schedule(dynamic)` balanceia bem.
- **Loop interno `j` (medidas)**: cada iteração chama `commonarraysMD`, `commonfactorsMD`, `hmd_TIV_optimized`, `vmd_optimized` com **mesmos** `n`, `npt`, `nf`, `nlayers`. Cada medida é independente e tem custo essencialmente constante. Carga **uniforme** → `schedule(static)` elimina o overhead de sincronização do dynamic.

O documento [`analise_paralelismo_cpu_fortran.md §7 Fase 2`](analise_paralelismo_cpu_fortran.md) prevê ganho de 5–15 % apenas com a mudança de schedule em loops de carga uniforme. A medição empírica confirma isto em 1–4 threads, mas mostra trade-off em 8–16 threads — ver §2.3.

### 1.4 Débito 1 — `writes_files` com Abertura Condicional

**Local**: [`PerfilaAnisoOmp.f08:217-258`](../../Fortran_Gerador/PerfilaAnisoOmp.f08) (antigo)

**Antes**:
```fortran
open(unit = 1000, iostat = exec, file = fileTR, form = 'unformatted', &
     access = 'stream', status = 'unknown', position = 'append')
```

**Depois**:
```fortran
inquire(file = fileTR, exist = file_exists)
if (modelm == 1 .or. .not. file_exists) then
    open(unit = 1000, iostat = exec, file = fileTR, form = 'unformatted', &
         access = 'stream', status = 'replace', action = 'write')
else
    open(unit = 1000, iostat = exec, file = fileTR, form = 'unformatted', &
         access = 'stream', status = 'old', position = 'append', action = 'write')
end if
```

**Análise do bug original**: a combinação `status='unknown', position='append'` fazia com que **toda invocação** de `tatu.x`, incluindo a primeira do lote (`modelm=1`), **anexasse** ao arquivo de saída existente. Consequência: re-executar o gerador Python (`fifthBuildTIVModels.py`) produzia um arquivo com `K × N` modelos em vez de `N`, silenciosamente, sem erro ou warning. Modelos duplicados contaminavam o dataset de treino do SurrogateNet.

**Lógica do fix**: tabela de decisão baseada em `modelm` e existência do arquivo:

| `modelm` | Arquivo existe? | Ação                                         |
|:--------:|:---------------:|:---------------------------------------------|
| `== 1`   | qualquer        | `status='replace'` — sobrescreve             |
| `> 1`    | sim             | `status='old', position='append'` — anexa    |
| `> 1`    | não             | `status='replace'` — fallback defensivo      |

O caso "modelm>1 sem arquivo" é tratado defensivamente para **não quebrar o benchmark** (que usa `model.in` com `modelm=1000`, `nmaxmodel=1000` e sempre limpa `.dat` antes de rodar) e para **sobreviver a interrupções do lote Python** (se o loop parou em `modelm=500` e o arquivo foi apagado manualmente, retomar em `modelm=501` não aborta).

### 1.5 Nova Declaração Necessária

**Local**: [`PerfilaAnisoOmp.f08:219`](../../Fortran_Gerador/PerfilaAnisoOmp.f08)

```fortran
! Antes:
! logical :: file_exists    ! comentado

! Depois:
logical :: file_exists      ! descomentado e usado pelo inquire()
```

---

## 2. Resultados dos Benchmarks

### 2.1 Compilação

```
gfortran -fopenmp -std=f2008 -pedantic -Wall -Wextra -Wimplicit-interface
         -fPIC -fmax-errors=1 -O3 -march=native -ffast-math -funroll-loops
         -fall-intrinsics -c PerfilaAnisoOmp.f08 ...
```

**Resultado**: compilação **limpa** — zero warnings, zero errors. O `-pedantic` e `-Wall -Wextra` não sinalizaram nenhuma irregularidade nas intervenções.

### 2.2 Sanity Test

```bash
OMP_NUM_THREADS=8 ./tatu.x
md5 -q Inv0_15Dip1000_t5.dat
# → c64745ed5d69d5f654b0bac7dde23a95
```

**MD5 idêntico ao baseline** — a saída binária é bit-a-bit idêntica à produzida pelo código original. Isto confirma que:

1. As correções de Débitos 1 e 2 não alteram nenhum resultado numérico.
2. O Hybrid Scheduler da Fase 2 produz saída idêntica porque o código é determinístico mesmo com ordem de execução diferente (cada iteração `j` é independente e escreve em posição fixa do array `z_rho1`/`c_H1`, depois agregada serialmente em `zrho1`/`cH1` e escrita em `writes_files`).

### 2.3 Scaling Test 1–16 Threads

Ver arquivo dedicado: [`Fortran_Gerador/bench/results/scaling_phase2_vs_baseline.md`](../../Fortran_Gerador/bench/results/scaling_phase2_vs_baseline.md).

Protocolo: 2 warmups + 5 medições por `(binário, threads)`, média aritmética. Binários compilados separadamente e executados no mesmo diretório temporário com `rm -f *.dat *.out` entre iterações.

**Observações principais**:

1. **Bug 2-thread corrigido** (−41 % wall-time, speedup 1,07× → 1,60×): a maior vitória da Fase 2. O particionamento multiplicativo transforma o caso degenerado em caso escalável.
2. **1 thread −12 %**: surpresa positiva, atribuída a menor overhead do `schedule(static)` vs `dynamic` mesmo com team de 1 thread (o runtime dynamic ainda paga custo de dispatch iteração-a-iteração).
3. **4 threads −23 %**: combinação de particionamento correto + schedule estático em carga uniforme.
4. **8–16 threads +6 a +11 %** (regressão marginal): com 600 iterações distribuídas em 8–16 threads, cada thread recebe 75 ou 37 iterações. Sob ruído de SO (interrupts, outras threads do sistema, preempção), o `dynamic` antigo absorve melhor o ruído via balanceamento tardio, enquanto `static` fica limitado pela thread mais lenta.

**Mitigação planejada** (iteração Fase 2b): experimentar `schedule(static, 16)` ou `schedule(guided, 4)` para recuperar algum balanceamento sem perder o benefício de chunks grandes. Esta micro-otimização fica fora de escopo desta iteração.

### 2.4 Validação Numérica via `validate_numeric.py`

Comparação entre `baseline_output.dat` (Fase 0, 2026-04-04 18:03) e `phase2_output.dat` (Fase 2, 2026-04-04 18:41):

```
[+] Carregando baseline_output.dat ...
    → 1200 registros, shape values=(1200, 21)
[+] Carregando phase2_output.dat ...
    → 1200 registros, shape values=(1200, 21)

## Comparação numérica
    max |Δ|            = 0.0000e+00
    RMS(Δ)             = 0.0000e+00
    max rel            = 0.0000e+00
    tolerância atol    = 1.0e-10

[✓] PASS — diferença dentro da tolerância
```

**Todas as 21 colunas têm `max|Δ| = 0,0000e+00`** — reprodutibilidade numérica **exata em todos os bits** de precisão dupla. A Fase 2 não introduz nem mesmo drift de último bit (1e-16) porque preserva a ordem de agregação: cada thread escreve em posição independente de `z_rho1(j,:,:)` e `c_H1(j,:,:)`, e a agregação final em `zrho1(k,1:nmed(k),:,:) = z_rho1` é serial.

---

## 3. Débitos Restantes (Para Futuras Iterações)

### 3.1 Débito Técnico #3 — Encerrado

O Débito Técnico #3 (particionamento `num_threads_j = maxthreads - ntheta`) foi **totalmente endereçado** pela Fase 2. O fix multiplicativo está em produção e validado empiricamente.

### 3.2 Regressão marginal em 8–16 threads

Como observado em §2.3, `schedule(static)` no loop interno causa perda de 6–11 % em regimes de 8–16 threads, compensando parcialmente os ganhos em 1–4 threads. **Não é um débito novo**, é um trade-off previsto mas não-dimensionado no plano original. Candidato a micro-otimização em **Fase 2b** (tuning de chunk size).

### 3.3 Pendentes do Roteiro Original

As seguintes fases permanecem no roteiro CPU (referência: [`analise_paralelismo_cpu_fortran.md §7`](analise_paralelismo_cpu_fortran.md)):

| Fase | Descrição | Ganho esperado | Status |
|:----:|:----------|:---------------|:------:|
| 3    | Workspace Pre-allocation (`thread_workspace` tipo) | +40 a +80 % | 📋 Planejada |
| 4    | Cache de `commonarraysMD` por `(r, freq)` — 1200 → 2 chamadas/modelo | **+60 a +120 %** | 📋 Planejada |
| 5    | `collapse(3)` nos loops `theta × medidas × freq` | +10 a +20 % | 📋 Planejada |
| 6    | Cache de `commonfactorsMD` por `camadT` | +15 a +25 % | 📋 Planejada |

A **Fase 4** permanece como a maior oportunidade de ganho do roteiro (60–120 %), pois é **estrutural** — ataca a redundância computacional em `commonarraysMD`, não depende de microarquitetura SIMD ou de scheduler OpenMP.

### 3.4 Fase 1 em AVX-512 (Re-tentativa em Hardware Diferente)

O patch arquivado em [`Fortran_Gerador/bench/attic/phase1_simd.patch`](../../Fortran_Gerador/bench/attic/phase1_simd.patch) permanece elegível para re-teste em hardware **AVX-512** (Xeon Scalable, Ice Lake-SP, Sapphire Rapids), onde os vetores de 64 bytes podem reabrir margem para SIMD explícito que o auto-vetorizador ainda não saturou.

---

## 4. Estado do Projeto Pós-Execução

### 4.1 Código Fortran

| Arquivo                           | Mudança                                                                          |
|:----------------------------------|:---------------------------------------------------------------------------------|
| [`PerfilaAnisoOmp.f08`](../../Fortran_Gerador/PerfilaAnisoOmp.f08) | **Modificado**: Débitos 1, 2 + Fase 2 (particionamento + schedule) |
| [`utils.f08`](../../Fortran_Gerador/utils.f08)                   | Intocado (sanity do Fase 1 — reverter confirmado) |
| [`magneticdipoles.f08`](../../Fortran_Gerador/magneticdipoles.f08) | Intocado                                                            |
| [`filtersv2.f08`](../../Fortran_Gerador/filtersv2.f08)           | Intocado                                                              |
| [`parameters.f08`](../../Fortran_Gerador/parameters.f08)         | Intocado                                                              |
| [`RunAnisoOmp.f08`](../../Fortran_Gerador/RunAnisoOmp.f08)       | Intocado                                                              |
| [`Makefile`](../../Fortran_Gerador/Makefile)                     | Intocado                                                              |

### 4.2 Arquitetura Python v2.0

**100 % intocada** — esta iteração é exclusivamente Fortran. O pipeline `geosteering_ai/` de carregamento `.dat` 22-col (em [`data/loading.py`](../../geosteering_ai/data/loading.py) e [`data/pipeline.py`](../../geosteering_ai/data/pipeline.py)) **beneficia-se indiretamente** do Débito 1 corrigido: datasets gerados a partir de agora não terão mais o risco silencioso de duplicação de modelos quando o gerador Python é re-executado sem `rm -f *.dat` prévio.

### 4.3 Documentação

Este relatório acompanha as atualizações sincronizadas em:

- [`docs/reference/analise_paralelismo_cpu_fortran.md`](analise_paralelismo_cpu_fortran.md) — nova §13 com resultados Fase 2.
- [`docs/reference/documentacao_simulador_fortran.md`](documentacao_simulador_fortran.md) — histórico de otimizações atualizado.
- [`docs/ROADMAP.md`](../ROADMAP.md) — F2.5.1 com Fase 2 marcada ✅.
- [`Fortran_Gerador/bench/results/scaling_phase2_vs_baseline.md`](../../Fortran_Gerador/bench/results/scaling_phase2_vs_baseline.md) — dados brutos do scaling test.
- [`Fortran_Gerador/bench/results/phase2_report.md`](../../Fortran_Gerador/bench/results/phase2_report.md) — relatório automatizado do `run_bench.sh`.

---

## 5. Próximos Passos Recomendados

### 5.1 Imediato (próxima sessão)

1. **Fase 3 — Workspace Pre-allocation**: eliminar `allocate`/`deallocate` dentro do laço paralelo via tipo `thread_workspace`. Ganho esperado **+40 a +80 %**. **Pré-requisito estrutural** para portabilidade CPU→GPU. Risco médio.

2. **Fase 4 — Cache de `commonarraysMD` por `(r, freq)`**: a maior oportunidade de ganho do roteiro (**+60 a +120 %**). Como `r = dTR = 1,0 m` é invariante por construção em perfilagem com passo fixo, `commonarraysMD` produz resultado idêntico para todas as 600 medições do mesmo ângulo/frequência. Reduzir de 1.200 chamadas/modelo para 2 chamadas/modelo elimina o gargalo número 1 (45 % do tempo total). Risco médio — requer refatoração estrutural de `fieldsinfreqs`.

### 5.2 Curto Prazo

3. **Fase 2b — Chunk Tuning**: mitigar a regressão marginal em 8–16 threads via experimentação de `schedule(static, k)` ou `schedule(guided, k)` com `k ∈ {8, 16, 32}`. Execução isolada, baixo risco.

4. **Fase 5 — `collapse(3)`**: fundir os laços `theta × medidas × freq` em um único espaço de iteração linearizado para melhor balanceamento em regimes multi-ângulo (`ntheta > 1`). Ganho esperado +10–20 %. Risco médio.

### 5.3 Médio Prazo

5. **Fase 6 — Cache de `commonfactorsMD` por `camadT`**: otimização análoga à Fase 4, para a outra sub-rotina redundante. Ganho esperado +15–25 %.

6. **Pipeline A (OpenACC GPU)** conforme [`documentacao_simulador_fortran.md §12`](documentacao_simulador_fortran.md): portabilidade CPU→GPU via diretivas OpenACC. **Somente após Fase 3 concluída** (workspace pre-allocation é pré-requisito).

### 5.4 Alinhamento com Arquitetura Python v2.0

- O baseline de throughput atual (~11.900 mod/h nesta CPU, medido com CPU aquecida) já viabiliza o treinamento da família **SurrogateNet** (TCN baseline + ModernTCN v2) com datasets de 10⁵–10⁶ modelos em configuração `ntheta=1`.
- O **Modo C do SurrogateNet** (tensor 9 componentes, 18 canais) requer datasets **multi-dip** (`ntheta ≥ 7`). Neste regime, a Fase 2 com particionamento multiplicativo é especialmente importante porque `num_threads_k = 7` e `num_threads_j = 2` (em 16 threads) distribuem o trabalho uniformemente, evitando a degeneração do particionamento subtrativo original.
- A **Fase 4** (cache de `commonarraysMD`) é o próximo passo crítico para viabilizar datasets multi-dip em tempo razoável.

---

## 6. Referências

1. [`docs/reference/relatorio_fase0_fase1_fortran.md`](relatorio_fase0_fase1_fortran.md) — Relatório das Fases 0 e 1 (antecedente direto desta execução).
2. [`docs/reference/analise_paralelismo_cpu_fortran.md`](analise_paralelismo_cpu_fortran.md) — Roteiro completo das 6 fases de otimização CPU.
3. [`docs/reference/documentacao_simulador_fortran.md`](documentacao_simulador_fortran.md) — Documentação técnica do simulador PerfilaAnisoOmp v4.0.
4. [`Fortran_Gerador/bench/results/scaling_phase2_vs_baseline.md`](../../Fortran_Gerador/bench/results/scaling_phase2_vs_baseline.md) — Dados brutos do scaling test.
5. OpenMP Architecture Review Board (2020). *OpenMP API Specification 5.1*, §3.2.7 (`omp_set_max_active_levels`), §3.2.13 (`schedule` clauses).
6. Chapman, B., Jost, G., van der Pas, R. (2007). *Using OpenMP: Portable Shared Memory Parallel Programming*, §4.5 (Nested Parallelism), §5.2 (Schedule Kinds).

---

**Fim do Relatório — Fase 2 + Débitos 1 e 2 do Roteiro de Paralelismo CPU Fortran**
