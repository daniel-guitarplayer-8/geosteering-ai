# Attic — Experimentos Arquivados

Este diretório guarda experimentos de otimização tentados durante a execução das Fases 0 e 1 do roteiro de paralelismo CPU que **não foram mesclados ao código de produção** por não terem entregado o ganho previsto. Preserva-se aqui para futura referência e auditoria científica.

## `utils_phase1_experiment.f08` + `phase1_simd.patch`

**Experimento**: refatoração de `commonarraysMD` (em [`../../utils.f08`](../../utils.f08)) transformando a sintaxe de array Fortran (`u(:,i) = sqrt(kr*kr - kh2(i))`) em loops `do ipt = 1, npt` explícitos anotados com `!$omp simd`, com invariantes do loop interno (`sqrt(lamb2(i))`, `h(i)`, `eta(i,1)`, `inv_zeta`, `kh2(i)`, `kv2(i)`) promovidos a escalares locais antes do loop vetorizado.

**Motivação**: a [`analise_paralelismo_cpu_fortran.md` §7.1](../../../docs/reference/analise_paralelismo_cpu_fortran.md) previa ganho de +15 a +30 % em `commonarraysMD` ao fornecer hints SIMD explícitos ao compilador, eliminando o loop versioning por aliasing observado em gfortran 14.x e anteriores.

**Resultado experimental (hardware de dev)**:

| Métrica (OMP_NUM_THREADS=8)    | Baseline       | Fase 1 (experimento) | Δ            |
|:-------------------------------|:---------------|:---------------------|:-------------|
| Wall-time médio (s/modelo)     | 0,1047 ± 0,015 | 0,1057 ± 0,011       | +0,96 %      |
| Throughput (modelos/h)         | 34 384         | 34 060               | −0,94 %      |
| Welch *t*-statistic            | —              | **+0,425**           | não-signif.  |
| Erro numérico (max \|Δ\|)      | —              | 1,93 · 10⁻¹³         | ≪ atol=1e-10 |

Protocolo: 60 iterações interleaved (alternadas entre os dois binários) em host macOS Darwin 25, Intel i9-9980HK (8 cores físicos, AVX-2), gfortran 15.2.0 Homebrew, `-O3 -march=native -ffast-math -funroll-loops -fopenmp`, pós-warmup (3 execuções descartadas).

**Causa raiz do não-ganho** (confirmada por `-fopt-info-vec`): o gfortran 15.2.0 **já auto-vetoriza** os loops de array-syntax em `commonarraysMD` com vetores de 32 bytes (máximo suportado pela CPU alvo, AVX-2 = 4 doubles por vetor). A anotação `!$omp simd` explícita não tem margem para adicionar paralelismo vetorial além do que o compilador já produz.

**Validação numérica**: a refatoração produz diferença de ordem 10⁻¹³ vs baseline (reordenação de operações de ponto flutuante), bem abaixo da tolerância acordada de 10⁻¹⁰. Portanto o código é **numericamente equivalente**, apenas não é mais rápido.

**Decisão**: **não mesclar** à produção. A versão original (array-syntax Fortran) é mais concisa e expressa a intenção matemática de forma mais próxima da formulação física, sendo preferível do ponto de vista de manutenção quando não há ganho de performance.

**Conclusão para o roteiro de paralelismo CPU**: a Fase 1 (SIMD Hankel Reduction) é **superada pela auto-vetorização moderna** em CPUs com AVX-2 e gfortran ≥ 14. Em CPUs com **AVX-512** (Xeon Scalable, Ice Lake) e/ou compiladores mais antigos (gfortran 11/12), o ganho pode ser recuperado — mas esse hardware não está disponível para o desenvolvimento atual. **Pular a Fase 1 neste hardware é legítimo**; recomenda-se ir diretamente à **Fase 2 (Hybrid Scheduler)** e à **Fase 3 (Workspace Pre-alloc)** que abordam gargalos estruturais, não de microarquitetura.

## Como re-aplicar o experimento

```bash
# 1. Aplicar o patch experimental
cd Fortran_Gerador
patch -p0 < bench/attic/phase1_simd.patch

# 2. Recompilar e rodar bench
bash bench/run_bench.sh --label phase1_retest --iters 60 --threads 8 --keep

# 3. Reverter (restaurar baseline)
patch -p0 -R < bench/attic/phase1_simd.patch
```
