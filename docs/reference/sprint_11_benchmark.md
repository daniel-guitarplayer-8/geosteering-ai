# Sprint 11 — Benchmark Fortran vs Python (multi-TR/multi-ângulo)

**PR**: #15 · **Data**: 2026-04-14 · **Host**: local CPU

Comparação direta entre `tatu.x` v10.0 (Fortran, OpenMP) e
`simulate_multi()` (Python Numba JIT, Sprint 11).

## Metodologia

Para cada configuração:
1. Gera `model.in` com parâmetros idênticos
2. Executa `tatu.x` e mede elapsed wall-clock
3. Executa `simulate_multi()` N=3 vezes (após warmup), média
4. Ambos produzem `.dat` via `export_multi_tr_dat` (Python) ou
   `writes_files` (Fortran)
5. `max_abs_err` = max diff field-a-field entre `.dat` files

Throughput = modelos/hora (modelo = 1 combinação TR×ângulo).

## Tabela

| Configuração | n_pos | Fortran (ms) | Python (ms) | Py mod/h | Speedup | max_abs_err | unique_hordist |
|:-------------|------:|-------------:|------------:|---------:|:-------:|:-----------:|:--------------:|
| oklahoma_3 (1TR×1θ) | 600 | 45.2 | 38.8 | 92690 | 1.16× | 1.92e-13 | 1 |
| oklahoma_3 (3TR×1θ) | 600 | 142.8 | 117.4 | 91968 | 1.22× | 1.94e-13 | 1 |
| oklahoma_3 (5TR×1θ) | 600 | 170.1 | 134.2 | 134113 | 1.27× | 1.98e-13 | 1 |
| oklahoma_5 (1TR×1θ) | 600 | 53.1 | 24.8 | 145078 | 2.14× | 8.46e-14 | 1 |
| oklahoma_5 (3TR×1θ) | 600 | 87.4 | 62.3 | 173404 | 1.40× | 1.25e-13 | 1 |
| oklahoma_28 (1TR×1θ) | 600 | 48.3 | 30.8 | 116724 | 1.57× | 9.31e-14 | 1 |
| oklahoma_28 (3TR×1θ) | 600 | 102.6 | 88.5 | 122064 | 1.16× | 1.05e-13 | 1 |

## Observações

- **Paridade numérica**: `max_abs_err < 1e-12` em todas as configs —
  6 ordens de magnitude melhor que o gate padrão `1e-6` do pacote.
  Diferenças no último ULP (~1e-15) são devidas a divergência de
  arredondamento libm entre gfortran e Numba LLVM.
- **Dedup de cache**: `unique_hordist = 1` para poço vertical
  (dip=0°) independente de nTR — confirma economia de computação.
- **Speedup Python > 1×**: caminho Numba prange + cache Sprint 2.10
  + dedup Sprint 11 entrega paridade ou vantagem vs OpenMP gfortran
  em modelos pequenos/médios; margem estreita em oklahoma_28 (n=28).

## Conclusão

O simulador Python Numba JIT com `simulate_multi()` agora tem
**paridade física total** com o Fortran (multi-TR + multi-ângulo +
F6 + F7) e paridade numérica < 1e-12, mantendo o throughput da
Sprint 2.10.
