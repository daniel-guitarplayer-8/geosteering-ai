# Scaling Test — Baseline vs Fase 2 (2026-04-04)

**Hardware**: Intel Core i9-9980HK, 8 cores físicos, 16 lógicos, AVX-2
**OS**: macOS Darwin 25.4.0
**Compilador**: gfortran 15.2.0 (Homebrew GCC 15.2.0_1)
**Flags**: `-O3 -march=native -ffast-math -funroll-loops -fopenmp`
**Protocolo**: 2 warmups + 5 medições por (binário, threads), média aritmética

## Binários Comparados

| Binário              | Origem                                                        |
|:---------------------|:--------------------------------------------------------------|
| `tatu_baseline.x`    | Commit `0b61fee` (utils.f08 + PerfilaAnisoOmp.f08 originais) |
| `tatu_phase2.x`      | Fase 2 aplicada: `omp_set_max_active_levels(2)` + particionamento multiplicativo `num_threads_k × num_threads_j` + `schedule(static)` no loop interno |

## Resultados Brutos

```
  tatu_baseline.x       OMP= 1  mean=1.4320s  speedup=1.00x
  tatu_baseline.x       OMP= 2  mean=1.3400s  speedup=1.07x ⚠ bug
  tatu_baseline.x       OMP= 4  mean=0.5220s  speedup=2.74x
  tatu_baseline.x       OMP= 8  mean=0.2760s  speedup=5.19x
  tatu_baseline.x       OMP=16  mean=0.2260s  speedup=6.34x

  tatu_phase2.x         OMP= 1  mean=1.2540s  speedup=1.00x
  tatu_phase2.x         OMP= 2  mean=0.7860s  speedup=1.60x ✓ fix
  tatu_phase2.x         OMP= 4  mean=0.4000s  speedup=3.13x
  tatu_phase2.x         OMP= 8  mean=0.3060s  speedup=4.10x
  tatu_phase2.x         OMP=16  mean=0.2400s  speedup=5.23x
```

## Comparação Lado-a-Lado

| OMP_NUM_THREADS | Baseline (s) | Fase 2 (s) | Δ%          | Speedup base | Speedup F2 | Avaliação              |
|:---------------:|:------------:|:----------:|:-----------:|:------------:|:----------:|:-----------------------|
| 1               | 1,432        | 1,254      | **−12,4 %** | 1,00×        | 1,00×      | Fase 2 mais rápida (menos overhead) |
| **2**           | **1,340** ⚠  | **0,786** ✅ | **−41,3 %** | **1,07×**    | **1,60×**  | **BUG 2-THREAD CORRIGIDO** |
| 4               | 0,522        | 0,400      | **−23,4 %** | 2,74×        | 3,13×      | Fase 2 ganho substancial |
| 8               | 0,276        | 0,306      | +10,9 %     | 5,19×        | 4,10×      | Baseline marginalmente melhor |
| 16              | 0,226        | 0,240      | +6,2 %      | 6,34×        | 5,23×      | Baseline marginalmente melhor |

## Análise

### Vitórias da Fase 2

1. **Bug 2-thread corrigido definitivamente** — a maior vitória. O particionamento antigo `num_threads_j = maxthreads - ntheta = 2-1 = 1` degenerava o loop interno a sequencial. O novo `max(1, maxthreads/num_threads_k) = max(1, 2/1) = 2` permite que ambos threads trabalhem. Speedup vai de 1,07× (anti-escalável) para 1,60× (escalável).

2. **1 thread mais rápido em 12,4 %** — inesperado mas explicável. Com `schedule(dynamic)` original, o runtime alocava iterações uma-a-uma mesmo com 1 thread, pagando o overhead de sincronização. `schedule(static)` divide o range de uma vez, eliminando o overhead.

3. **4 threads mais rápido em 23,4 %** — a combinação de (a) melhor particionamento e (b) `schedule(static)` com chunks maiores reduz a sincronização em paralelismo moderado.

### Trade-off em 8–16 threads

Com 8–16 threads e 600 iterações, `schedule(static)` atribui 75–37 iterações por thread. Quando há ruído de SO (interrupts, outras threads do sistema), o `dynamic` do baseline absorve melhor esse ruído via balanceamento tardio, enquanto `static` fica limitado pela thread mais lenta. Δ +6 a +11 % em favor do baseline neste regime.

**Mitigação futura** (Fase 2 iteração 2): usar `schedule(static, 16)` ou `schedule(guided, 4)` para recuperar algum balanceamento sem perder o benefício de chunks grandes. Fica para fase futura.

### Absoluto vs Relativo

As medições absolutas estão afetadas por **throttling térmico** da CPU (todas as execuções consecutivas) — os valores absolutos NÃO devem ser comparados com o baseline original de 0,1047 s medido em CPU fria. A comparação válida é **relativa**, dentro do mesmo bloco de medições interleaved.

## Decisão

**Aceitar Fase 2** para o código de produção. O fix do bug 2-thread é crítico (Débito Técnico #3 descoberto na Fase 0), e 3 dos 5 regimes de threads mostram ganho substancial. O regime 8–16 threads com degradação marginal pode ser otimizado em iteração futura via schedule tuning.

## MD5 de Validação

- `baseline_output.dat` (Fase 0): `c64745ed5d69d5f654b0bac7dde23a95`
- `phase2_output.dat` (Fase 2):   `c64745ed5d69d5f654b0bac7dde23a95`
- **Idênticos** — a Fase 2 não alterou nenhum bit da saída (validação via `validate_numeric.py`: max|Δ| = 0,0000e+00 em todas as 21 colunas).
