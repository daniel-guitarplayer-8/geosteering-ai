# Benchmark Local — Python Numba JIT vs Fortran tatu.x (OpenMP)

**Data**: 2026-04-17 23:49:58  
**Host**: CPU local (16 threads, NUMBA_NUM_THREADS=16, OMP_NUM_THREADS=16)  
**Numba warmup**: 1 chamada (JIT) descartada; timing = média de N=3 chamadas  
**Fortran**: `tatu.x` v10.0 via subprocess; OpenMP herdado do ambiente  

## Metodologia

- **Modelo = 1 chamada**: `mod/h` conta chamadas completas do simulador (1 `simulate_multi()` ou 1 execução de `tatu.x`), independente de número de sub-configurações (TR × ângulo × freq) produzidas por chamada.
- **Submodelo**: 1 combinação TR × ângulo × frequência. `ms/submodelo` reporta custo computacional por resposta EM.
- **Speedup = Fortran / Numba**: razão de tempos absolutos por chamada. Valores >1× indicam Numba mais rápido.
- **Paridade (`max_abs_err`)**: comparação record-a-record dos arquivos `.dat` exportados. Apenas reportada para Config A (dip=0°, n_pos idêntico entre simuladores).

## Configurações

| Config | Freqs (kHz) | TRs (m) | Dips (°) | n_pos | Submodelos/chamada |
|:------:|:-----------:|:-------:|:--------:|:-----:|:------------------:|
| **A** | 20 | 1.0 | 0 | 600 | 1 |
| **B** | 20, 40 | 1.0 | 0, 30 | 600 | 4 |

## Tabela 1 — Config A (baseline: 1 freq × 1 TR × 1 dip × 600 pos)

| Modelo | n_pos | Numba (ms) | Fortran (ms) | Numba mod/h | Fortran mod/h | Speedup | max_abs_err |
|:-------|------:|-----------:|-------------:|------------:|--------------:|:-------:|:-----------:|
| Oklahoma 3 camadas (TIV) | 600 | 17.8 | 59.2 | 202,560 | 60,805 | 3.33× | 1.42e-13 |
| Oklahoma 5 camadas (TIV) | 600 | 21.0 | 43.6 | 171,556 | 82,575 | 2.08× | 8.16e-14 |
| Oklahoma 28 camadas (TIV forte) | 600 | 28.5 | 48.3 | 126,361 | 74,460 | 1.70× | 1.33e-13 |
| Devine 8 camadas (isotrópico) | 600 | 18.4 | 44.4 | 195,729 | 81,171 | 2.41× | 1.60e-13 |
| Hou 7 camadas (TIV) | 600 | 22.7 | 45.9 | 158,760 | 78,348 | 2.03× | 1.68e-13 |
| Viking Graben 10 camadas (TIV) | 600 | 17.3 | 45.1 | 207,581 | 79,878 | 2.60× | 1.72e-13 |

## Tabela 2 — Config B (carga 4×: 2 freqs × 1 TR × 2 dips × 600 pos)

| Modelo | Numba (ms) | Fortran (ms) | ms/submodelo Numba | ms/submodelo Fortran | Numba mod/h | Fortran mod/h | Speedup |
|:-------|-----------:|-------------:|-------------------:|---------------------:|------------:|--------------:|:-------:|
| Oklahoma 3 camadas (TIV) | 73.8 | 108.2 | 18.45 | 27.05 | 48,790 | 33,276 | 1.47× |
| Oklahoma 5 camadas (TIV) | 83.9 | 114.2 | 20.98 | 28.56 | 42,895 | 31,513 | 1.36× |
| Oklahoma 28 camadas (TIV forte) | 92.5 | 105.8 | 23.12 | 26.46 | 38,931 | 34,013 | 1.14× |
| Devine 8 camadas (isotrópico) | 93.4 | 102.9 | 23.34 | 25.73 | 38,555 | 34,985 | 1.10× |
| Hou 7 camadas (TIV) | 71.0 | 108.6 | 17.75 | 27.15 | 50,705 | 33,148 | 1.53× |
| Viking Graben 10 camadas (TIV) | 80.8 | 107.3 | 20.19 | 26.83 | 44,582 | 33,549 | 1.33× |

## Tabela 3 — Experimento 30.000 perfis aleatórios (20 camadas, ρh>1000 Ω·m)

Configuração: 1 freq (20 kHz), 1 TR (1 m), 1 dip (0°), 600 posições; perfis log-uniformes ρh∈[1000,10000] Ω·m, TIV λ∈[1,3], esp∈[0.5, 5] m.

| Métrica | Numba | Fortran  (subset 300, extrapolado para 30,000) |
|:--------|------:|--------:|
| Total de modelos | 30,000 | 300 | 
| Tempo total (s) | 1215.3 | 30.0 |
| Tempo por modelo (ms) | 40.51 | 100.06 |
| Throughput (mod/h) | 88,869 | 35,979 |
| Total extrapolado 30k (s) | — | 3001.7 |
| **Speedup Fortran/Numba** | — | **2.47×** |

## Estatísticas agregadas

- **Config A** — Numba mod/h médio: **177,091** (min: 126,361, max: 207,581)
- **Config B** — Numba mod/h médio: **44,076** (min: 38,555, max: 50,705)
- **Speedup médio Config A** (Fortran/Numba): **2.36×** (min: 1.70×, max: 3.33×)
- **Speedup médio Config B** (Fortran/Numba): **1.32×** (min: 1.10×, max: 1.53×)
