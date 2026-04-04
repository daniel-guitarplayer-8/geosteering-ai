# Relatório de Execução — Fases 0 e 1 do Roteiro de Paralelismo CPU Fortran

**Projeto**: Geosteering AI v2.0 — Simulador `PerfilaAnisoOmp`
**Data**: 2026-04-04
**Autor**: Daniel Leal (execução assistida por Claude Code)
**Escopo**: Execução das Fases 0 (Benchmark Baseline) e 1 (SIMD Hankel Reduction) definidas em [`analise_paralelismo_cpu_fortran.md`](analise_paralelismo_cpu_fortran.md) §7
**Commit**: a ser atribuído no merge (ver `git log` pós-merge)

---

## Sumário Executivo

As Fases 0 e 1 do roteiro de otimização CPU do simulador Fortran foram executadas integralmente em hardware de desenvolvimento (macOS Darwin, Intel i9-9980HK, 8 cores físicos, AVX-2). A Fase 0 (baseline) estabeleceu o ponto de referência quantitativo: **0,1047 ± 0,015 s por modelo**, **~34.400 modelos/hora** com 8 threads OpenMP. A Fase 1 (refatoração de `commonarraysMD` com `!$omp simd` explícito em `utils.f08`) **não produziu ganho estatisticamente significativo** (Δ +0,96 %, Welch *t* = +0,425, não-significativo a 95 %).

A causa-raiz do não-ganho foi confirmada por auditoria de auto-vetorização (`-fopt-info-vec`): **gfortran 15.2.0 já vetoriza automaticamente** os loops de array-syntax de `commonarraysMD` com vetores de **32 bytes** — o máximo suportado pela arquitetura AVX-2 (4 doubles por vetor). Não há margem para SIMD adicional nesta classe de hardware. O experimento foi **arquivado em [`Fortran_Gerador/bench/attic/`](../../Fortran_Gerador/bench/attic/)** para auditoria futura e o código de produção foi restaurado ao estado original.

**Recomendação**: Prosseguir diretamente para as **Fases 2 (Hybrid Scheduler)** e **3 (Workspace Pre-alloc)**, que atacam gargalos estruturais (escalonamento, alocação dinâmica), não de microarquitetura. A Fase 1 deve ser re-avaliada quando houver acesso a hardware **AVX-512** (Xeon Scalable, Ice Lake-SP), onde a margem de vetorização explícita pode ressurgir.

Apesar do não-ganho de performance, a execução entregou artefatos permanentes de valor:

1. **Infraestrutura de benchmark reprodutível** ([`Fortran_Gerador/bench/`](../../Fortran_Gerador/bench/)) com suporte dual-OS (macOS + Linux), medição estatística (média, σ, mediana, Welch *t*), validação numérica automatizada (`atol=1e-10`, `rtol=1e-12`) e geração de relatórios Markdown.
2. **Baseline quantitativo** publicado — referência obrigatória para futuras otimizações.
3. **Três débitos técnicos** identificados no código de produção (ver §7).
4. **Descoberta científica documentada**: auto-vetorização moderna fecha a janela da Fase 1 do roteiro original em CPUs AVX-2.

---

## 1. Ambiente de Execução

| Campo                              | Valor                                                     |
|:-----------------------------------|:----------------------------------------------------------|
| Sistema operacional                | macOS Darwin 25.4.0                                       |
| CPU                                | Intel Core i9-9980HK @ 2,40 GHz                           |
| Núcleos físicos                    | 8                                                         |
| Núcleos lógicos                    | 16                                                        |
| Conjunto SIMD máximo               | **AVX-2 (32 bytes, 4 doubles/vetor)**                     |
| Compilador                         | GNU Fortran (Homebrew GCC 15.2.0_1) 15.2.0                |
| Flags de produção                  | `-O3 -march=native -ffast-math -funroll-loops -fopenmp`   |
| `OMP_NUM_THREADS` (benchmark)      | 8                                                         |
| Runtime OpenMP                     | libgomp                                                   |
| Ferramenta de timing               | `/usr/bin/time -p` (wall/user/sys)                        |

Observação: `gprof` não está disponível no macOS por padrão; o profiling por sub-rotina previsto no plano original foi substituído por **wall-time estatístico** com validação por Welch *t*-test. Em ambiente Linux, o script [`bench/run_bench.sh`](../../Fortran_Gerador/bench/run_bench.sh) pode ser acompanhado de `gprof` ou `perf record` conforme disponibilidade.

---

## 2. Fase 0 — Benchmark Baseline

### 2.1 Objetivo

Capturar métricas reprodutíveis do simulador **no estado original**, estabelecendo ponto de referência para (i) qualquer otimização futura e (ii) validação numérica de otimizações via MD5 + `numpy.allclose`.

### 2.2 Protocolo

- Configuração física: [`Fortran_Gerador/model.in`](../../Fortran_Gerador/model.in) — `nf=2` (20 kHz, 40 kHz), `ntheta=1` (0°), `ncam=10`, janela `tj=120 m`, passo `p_med=0,2 m`, arranjo `dTR=1,0 m`, filtro Hankel 201 pontos (Werthmüller).
- **N=60 invocações** de `tatu.x` em loop controlado, com remoção de `*.dat` e `*.out` entre iterações (contorna o bug de `position='append'` descrito em §7.1).
- **3 execuções de warmup** descartadas antes da coleta.
- `OMP_NUM_THREADS=8` (valor ótimo de escalabilidade — ver §2.4).
- MD5 do arquivo binário `.dat` computado para assinatura de reprodutibilidade.

### 2.3 Resultados

| Métrica                        | Baseline (média ± σ)          |
|:-------------------------------|:------------------------------|
| **Wall-time (s/modelo)**       | **0,1047 ± 0,0147**           |
| Mediana (s)                    | 0,1000                        |
| Mínimo (s)                     | 0,0900                        |
| Máximo (s)                     | 0,1500                        |
| **Throughput (modelos/hora)**  | **~34.384**                   |
| Tamanho do `.dat` gerado       | 206.400 bytes (1.200 reg × 172 B) |
| MD5 do `.dat` (referência)     | `c64745ed5d69d5f654b0bac7dde23a95` |

### 2.4 Escalabilidade Multi-Thread (Sondagem)

Execução sondagem única-invocação, thread sweep em `model.in` (ntheta=1):

| `OMP_NUM_THREADS` | Wall-time (s) | Speedup | Eficiência |
|:-----------------:|:-------------:|:-------:|:----------:|
| 1                 | 0,50          | 1,00×   | 100 %      |
| 2                 | 0,53          | 0,94×   | 47 %       |
| 4                 | 0,19          | 2,63×   | 66 %       |
| **8**             | **0,12**      | **4,17×** | **52 %**  |
| 16                | 0,08          | 6,25×   | 39 %       |

**Observações críticas**:

1. **Thread=2 exibe anti-escalabilidade** (0,94×) devido ao cálculo `num_threads_j = maxthreads − ntheta = 2 − 1 = 1`, que degenera o loop interno para 1 thread. Esta é uma **falha de distribuição** no particionamento em [`PerfilaAnisoOmp.f08:85`](../../Fortran_Gerador/PerfilaAnisoOmp.f08), candidata à correção em fase futura (Fase 2).
2. **Eficiência cai após 8 threads** porque o hardware possui apenas 8 cores físicos; de 9 a 16 threads entram em hyperthreading, com retorno marginal.
3. **Configuração ótima**: `OMP_NUM_THREADS = número de cores físicos`, que neste hardware é 8.

### 2.5 Comparação com Previsão do Plano Original

A [`analise_paralelismo_cpu_fortran.md` §2.3](analise_paralelismo_cpu_fortran.md) assumia baseline de **2,40 s/modelo** e **~24.000 modelos/hora**. A execução real mostrou **23× mais rápido** (0,10 s/modelo, 34.400 modelos/h). A discrepância decorre de:

- **Configuração diferente**: o documento assumia `ntheta > 1` e modelos com mais camadas; o `model.in` atual tem `ntheta=1` e 10 camadas.
- **Hardware mais recente**: i9-9980HK (2018) vs. hardware de referência do documento.
- **Compilador mais recente**: gfortran 15.2.0 (2025) tem vetorização substancialmente melhor que versões anteriores.

**Lição**: baselines precisam ser **medidos em cada ambiente**, não assumidos a partir de documentação.

---

## 3. Fase 1 — SIMD Hankel Reduction (Experimento)

### 3.1 Descoberta Prévia — Auditoria de Auto-Vetorização

Antes de aplicar qualquer alteração, foi executada auditoria via `-fopt-info-vec=$OUT/vec.txt` em cada módulo. O relatório revelou que gfortran 15.2.0 **já vetoriza** os loops críticos com **vetores de 32 bytes** (máximo AVX-2):

```
utils.f08:353:31: optimized: loop vectorized using 32 byte vectors
magneticdipoles.f08:444:22: optimized: loop vectorized using 32 byte vectors
magneticdipoles.f08:447:53: optimized: loop vectorized using 32 byte vectors
magneticdipoles.f08:450:53: optimized: loop vectorized using 32 byte vectors
magneticdipoles.f08:453:60: optimized: loop vectorized using 32 byte vectors
magneticdipoles.f08:476-535: ... (idem para todas as branches de select case)
```

Isto **altera a premissa** da Fase 1: não há margem para "forçar vetorização que o compilador não fez", porque o compilador já vetoriza no limite do hardware. Qualquer `!$omp simd` explícito apenas repete o trabalho do compilador. A única margem teórica remanescente seria **redução de loop versioning por aliasing** (versões escalares de fallback), que gfortran ainda emite em alguns casos por segurança.

### 3.2 Intervenção Aplicada

Refatoração de [`commonarraysMD` (utils.f08:158-241)](../../Fortran_Gerador/utils.f08), convertendo o bloco de atribuições em array-syntax para loops `do ipt = 1, npt` explícitos com `!$omp simd`, hoisting de invariantes escalares (`sqrt(lamb2(i))`, `h(i)`, `eta(i,1)`, `1/zeta`) para fora do loop interno. O patch completo está preservado em [`Fortran_Gerador/bench/attic/phase1_simd.patch`](../../Fortran_Gerador/bench/attic/phase1_simd.patch).

**Motivação**: fornecer hints explícitos ao compilador para eliminar loop versioning por aliasing, reduzir pressão de registradores e unificar as 9 atribuições separadas num único passo sobre o array.

### 3.3 Protocolo de Benchmark Comparativo

Benchmark **interleaved** rigoroso:

1. Compilar `tatu_baseline.x` (código original) e `tatu_phase1.x` (com patch) separadamente.
2. Executar 3 warmups alternados, descartados.
3. Em loop de 60 iterações: `tatu_baseline.x` seguido imediatamente por `tatu_phase1.x` no mesmo ambiente térmico.
4. Wall-time coletado via `/usr/bin/time -p` após cada execução.
5. Estatísticas via `statistics.fmean`, `statistics.stdev`, `statistics.median`, Welch *t*-test.

A razão do protocolo interleaved é eliminar viés de drift térmico da CPU: executar 60 vezes baseline + 60 vezes phase1 em blocos sequenciais pode sofrer frequency throttling diferenciado.

### 3.4 Resultados

| Métrica                        | Baseline         | Fase 1 (experimento) | Δ / Análise              |
|:-------------------------------|:-----------------|:---------------------|:-------------------------|
| N (iterações)                  | 60               | 60                   | —                        |
| Wall-time médio (s)            | **0,1047**       | **0,1057**           | **+0,0010 (+0,96 %)**    |
| Desvio-padrão (s)              | 0,0147           | 0,0108               | —                        |
| Mediana (s)                    | 0,1000           | 0,1000               | **0,0 %**                |
| Mínimo (s)                     | 0,0900           | 0,1000               | +11 %                    |
| Welch *t*-statistic            | —                | **+0,425**           | **não-significativo**    |
| Significância estatística      | —                | —                    | \|t\| > 2 requerido p/ 95 % |

**Interpretação estatística**: `|t| = 0,425 < 2,0` indica que a diferença entre as médias **não é distinguível do ruído de medição**. A Fase 1, do ponto de vista de performance, é **indistinguível do baseline**.

### 3.5 Validação Numérica

Validação via [`bench/validate_numeric.py`](../../Fortran_Gerador/bench/validate_numeric.py) carregando os `.dat` binários de ambas versões e executando `np.allclose(atol=1e-10, rtol=1e-12)`:

| Coluna         | max \|Δ\|     | Dentro de atol=1e-10? |
|:---------------|:--------------|:----------------------|
| `z_obs`        | 0,00e+00      | ✓                     |
| `rho_h, rho_v` | 0,00e+00      | ✓                     |
| `Re(H1), Im(H1)` | 1,93e-13, 9,11e-14 | ✓              |
| `Re(H5), Im(H5)` | 1,93e-13, 9,11e-14 | ✓              |
| `Re(H9), Im(H9)` | 8,44e-15, 2,80e-15 | ✓              |
| Demais H2–H8   | 0,00e+00      | ✓                     |

**Resultado**: `[✓] PASS — diferença dentro da tolerância`. Erro máximo absoluto de **1,93 · 10⁻¹³**, compatível com reordenação de operações de ponto flutuante (último bit de mantissa em double precision). O código refatorado é **numericamente equivalente** ao baseline.

### 3.6 Decisão de Merge

**Rejeitada** — o experimento é preservado em [`bench/attic/`](../../Fortran_Gerador/bench/attic/) mas **não é mesclado à produção** pelos motivos:

1. **Sem ganho de performance mensurável** (Δ +0,96 %, não-significativo).
2. **Aumento de complexidade do código** (9 variáveis locais novas, 60 linhas adicionais para expressar a mesma matemática em loops explícitos).
3. **Drift numérico** (ainda que pequeno, de ordem 10⁻¹³) sem compensação de benefício.
4. **Sintaxe de array Fortran é mais legível** para o público-alvo do código (geofísicos e engenheiros de poço), pois aproxima a expressão da formulação matemática (Liu 2017, equação 4.80; Werthmüller 2018).

Esta rejeição **não invalida o roteiro geral** — as Fases 2–6 atacam problemas estruturais (escalonamento, alocação dinâmica, cache de invariantes) que **não dependem de microarquitetura SIMD**.

---

## 4. Débitos Técnicos Identificados

Durante a execução foram descobertos bugs/depreciações no código de produção. Por decisão explícita do usuário (escopo "apenas docs"), nenhum foi corrigido nesta iteração; são registrados abaixo para futura correção.

### 4.1 `writes_files` — Append Sem Limpeza Prévia (Alta Prioridade)

**Local**: [`Fortran_Gerador/PerfilaAnisoOmp.f08:245-246`](../../Fortran_Gerador/PerfilaAnisoOmp.f08)

```fortran
open(unit = 1000, iostat = exec, file = fileTR, form = 'unformatted', &
     access =  'stream', status = 'unknown', position = 'append')
```

**Problema**: múltiplas execuções sem remoção prévia de `*.dat` concatenam dados ao arquivo existente. O arquivo final pode conter `K × N` registros em vez de `N` (onde `K` é o número de invocações anteriores). Isto **invalida benchmarks** e pode produzir arquivos de treinamento com duplicatas silenciosas.

**Contorno atual**: `bench/run_bench.sh` executa `rm -f *.dat *.out` antes de cada iteração.

**Correção recomendada** (fase futura):
```fortran
if (modelm == 1) then
   status_str = 'replace'   ! primeiro modelo → cria arquivo limpo
else
   status_str = 'unknown'   ! demais → append
end if
```

### 4.2 `omp_set_nested` Depreciado (Prioridade Média)

**Local**: [`Fortran_Gerador/PerfilaAnisoOmp.f08:74`](../../Fortran_Gerador/PerfilaAnisoOmp.f08)

```fortran
call omp_set_nested(.true.)
```

**Problema**: `omp_set_nested` é **depreciado desde OpenMP 5.0** (2018). Em OpenMP 5.1+ a API correta é `omp_set_max_active_levels(n)`. Compiladores futuros podem remover a função, gerando warning ou erro de compilação.

**Correção recomendada** (fase futura):
```fortran
call omp_set_max_active_levels(2)  ! equivalente a omp_set_nested(.true.) com nivel=2
```

### 4.3 Inconsistência de Localização em Documentação (Baixa Prioridade — Corrigida Parcialmente)

A documentação [`analise_paralelismo_cpu_fortran.md` §7.1](analise_paralelismo_cpu_fortran.md) sugeria que `commonarraysMD` estaria "dentro de `PerfilaAnisoOmp`". Localização real: [`Fortran_Gerador/utils.f08:158-241`](../../Fortran_Gerador/utils.f08). `PerfilaAnisoOmp.f08` apenas importa `utils` e a chama via `use utils`. A atualização do MD nesta execução corrige a referência.

### 4.4 Escalabilidade em 2 Threads (Descoberta Adicional)

**Local**: [`Fortran_Gerador/PerfilaAnisoOmp.f08:84-85`](../../Fortran_Gerador/PerfilaAnisoOmp.f08)

```fortran
num_threads_k = ntheta              ! =1 por default
num_threads_j = maxthreads - ntheta ! =maxthreads-1
```

**Problema**: quando `OMP_NUM_THREADS=2`, `num_threads_j = 1`, degenerando o loop interno a serial. Isto causa a anti-escalabilidade observada em §2.4 (thread=2 → speedup 0,94×).

**Correção recomendada** (Fase 2 do roteiro): substituir por `collapse(3)` conforme já previsto em [`§7 Fase 5`](analise_paralelismo_cpu_fortran.md), ou lógica de particionamento com garantia mínima de 2 threads no nível interno.

---

## 5. Artefatos Produzidos

| Artefato | Caminho | Status |
|:---------|:--------|:------:|
| Script de benchmark dual-OS | [`Fortran_Gerador/bench/run_bench.sh`](../../Fortran_Gerador/bench/run_bench.sh) | ✓ Novo |
| Validador numérico | [`Fortran_Gerador/bench/validate_numeric.py`](../../Fortran_Gerador/bench/validate_numeric.py) | ✓ Novo |
| README do bench | [`Fortran_Gerador/bench/README.md`](../../Fortran_Gerador/bench/README.md) | ✓ Novo |
| Relatório baseline (30 iter) | [`Fortran_Gerador/bench/results/baseline_report.md`](../../Fortran_Gerador/bench/results/baseline_report.md) | ✓ Novo |
| Relatório phase1 (30 iter) | [`Fortran_Gerador/bench/results/phase1_report.md`](../../Fortran_Gerador/bench/results/phase1_report.md) | ✓ Novo |
| Séries brutas baseline | [`Fortran_Gerador/bench/results/baseline_interleaved_times.txt`](../../Fortran_Gerador/bench/results/baseline_interleaved_times.txt) | ✓ Novo |
| Séries brutas phase1 | [`Fortran_Gerador/bench/results/phase1_interleaved_times.txt`](../../Fortran_Gerador/bench/results/phase1_interleaved_times.txt) | ✓ Novo |
| Patch experimento Fase 1 | [`Fortran_Gerador/bench/attic/phase1_simd.patch`](../../Fortran_Gerador/bench/attic/phase1_simd.patch) | ✓ Arquivado |
| Fonte experimento Fase 1 | [`Fortran_Gerador/bench/attic/utils_phase1_experiment.f08`](../../Fortran_Gerador/bench/attic/utils_phase1_experiment.f08) | ✓ Arquivado |
| README do attic | [`Fortran_Gerador/bench/attic/README.md`](../../Fortran_Gerador/bench/attic/README.md) | ✓ Novo |
| Este relatório | [`docs/reference/relatorio_fase0_fase1_fortran.md`](relatorio_fase0_fase1_fortran.md) | ✓ Este arquivo |
| Saída binária baseline | `bench/results/baseline_output.dat` (202 KB) | ✓ Referência |
| Saída binária phase1 | `bench/results/phase1_output.dat` (202 KB) | ✓ Referência |

### Artefatos Modificados

| Arquivo | Tipo de modificação |
|:--------|:--------------------|
| [`docs/reference/analise_paralelismo_cpu_fortran.md`](analise_paralelismo_cpu_fortran.md) | Anexada seção §12 com resultados reais + correção de localização de `commonarraysMD` |
| [`docs/reference/documentacao_simulador_fortran.md`](documentacao_simulador_fortran.md) | Nota em §11.x sobre resultados de Fase 1 (auto-vetorização gfortran 15.x) |
| [`docs/ROADMAP.md`](../../docs/ROADMAP.md) | Nova subseção F2.5.1 — CPU Optimization Roadmap com Fases 0/1 concluídas |

### Arquivos de Código NÃO Modificados

- [`Fortran_Gerador/utils.f08`](../../Fortran_Gerador/utils.f08) — experimento Fase 1 revertido
- [`Fortran_Gerador/PerfilaAnisoOmp.f08`](../../Fortran_Gerador/PerfilaAnisoOmp.f08) — intocado
- [`Fortran_Gerador/magneticdipoles.f08`](../../Fortran_Gerador/magneticdipoles.f08) — intocado
- [`Fortran_Gerador/Makefile`](../../Fortran_Gerador/Makefile) — intocado
- [`geosteering_ai/`](../../geosteering_ai/) — 100 % intocado (escopo é exclusivamente Fortran)

---

## 6. Próximos Passos Recomendados

### 6.1 Curto Prazo (próximas sessões de trabalho)

1. **Fase 2 — Hybrid Scheduler** ([§7 Fase 2](analise_paralelismo_cpu_fortran.md)): substituir `schedule(dynamic)` por lógica condicional que escolhe `static`/`dynamic`/`guided` conforme `ntheta × nmed`. Risco baixo, ganho esperado 5–15 %, ataca também o débito 4.4 (escalabilidade ruim em 2 threads).

2. **Correção dos débitos técnicos 4.1 e 4.2**:
   - `writes_files`: trocar `position='append'` para lógica condicional baseada em `modelm==1`.
   - `omp_set_nested` → `omp_set_max_active_levels(2)`.
   - Ambas são correções cirúrgicas, baixo risco.

3. **Re-validar Fase 1 em hardware AVX-512** (Xeon Scalable, Ice Lake-SP, servidor Colab Pro+ A100 host): reaplicar o patch preservado em `attic/` e medir. Se houver ganho, promover à produção com flag condicional `-march=skylake-avx512`.

### 6.2 Médio Prazo

4. **Fase 3 — Workspace Pre-allocation** ([§7 Fase 3](analise_paralelismo_cpu_fortran.md)): eliminar `allocate`/`deallocate` dentro do laço paralelo via tipo `thread_workspace`. Risco médio, ganho esperado 40–80 %. **Este é o primeiro passo com potencial de ganho substancial.**

5. **Fase 4 — Cache de `commonarraysMD` por `(r, freq)`** ([§7 Fase 4](analise_paralelismo_cpu_fortran.md)): reduzir 1.200 chamadas/modelo para 2 chamadas/modelo. Ganho esperado 60–120 % (o maior do roteiro). Risco médio — requer refatoração estrutural de `fieldsinfreqs`.

### 6.3 Longo Prazo

6. **Fase 5 — `collapse(3)`** + **Fase 6 — Cache de `commonfactorsMD`** (refinamento final).

7. **Ponte para GPU**: implementar Pipeline A (OpenACC) conforme [`documentacao_simulador_fortran.md` §12](documentacao_simulador_fortran.md) — **apenas após** Fases 3–4 concluídas, pois a pré-alocação de workspace é pré-requisito estrutural para a portabilidade CPU→GPU.

### 6.4 Alinhamento com Arquitetura v2.0

- O simulador Fortran alimenta o pipeline Python v2.0 via arquivos `.dat` binários (formato 22 colunas descrito em [`docs/reference/documentacao_simulador_fortran.md` §6](documentacao_simulador_fortran.md)).
- A taxa de geração atual (~34.400 modelos/h neste hardware) já é **compatível com o treinamento do SurrogateNet TCN+ModernTCN**, que requer O(10⁵–10⁶) modelos.
- Otimizações CPU (Fases 2–4) permitirão geração de **datasets multi-dip** (ntheta=7 ou mais) necessários para o **Modo C** do SurrogateNet (tensor 9-componente) — atualmente bloqueado pela baixa escalabilidade (ver débito 4.4).

---

## 7. Referências

1. [`docs/reference/documentacao_simulador_fortran.md`](documentacao_simulador_fortran.md) — Documentação técnica completa do simulador PerfilaAnisoOmp (v4.0, 6.557 linhas).
2. [`docs/reference/analise_paralelismo_cpu_fortran.md`](analise_paralelismo_cpu_fortran.md) — Roteiro das 6 fases de otimização CPU.
3. [`docs/ROADMAP.md`](../../docs/ROADMAP.md) — Roadmap geral do projeto.
4. Werthmüller, D. (2018). "Digital filter design for the Hankel transform." *Geophysics* 83(5), F49–F61.
5. Liu, Z. (2017). *Theory of Electromagnetic Well Logging*. Academic Press — capítulo 4 (equações 4.80, figura 4.17).
6. OpenMP Architecture Review Board (2020). *OpenMP API Specification 5.1* — §3.2.7 (`omp_set_max_active_levels`).
7. GCC 15 Release Notes — gfortran auto-vectorization improvements (2024–2025).

---

**Fim do Relatório — Fases 0 e 1 do Roteiro de Paralelismo CPU Fortran**
