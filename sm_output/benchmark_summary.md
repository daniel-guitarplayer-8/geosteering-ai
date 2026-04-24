# Benchmark Simulation Manager — Numba JIT vs Fortran tatu.x

## Config C — uniforme · f=[2k,6k]Hz · TR=[8.19,20.43]m · dip=0° · h1=10m · p_med=0.200m
| Modelo | Numba ms/mod | Fortran ms/mod | Numba mod/h | Fortran mod/h | Speedup |
|:-------|------------:|--------------:|-----------:|--------------:|:-------:|
| oklahoma_3 | 14.93 | 82.84 | 241,066 | 43,459 | 5.55× |
| oklahoma_5 | 21.12 | 83.44 | 170,465 | 43,142 | 3.95× |
| oklahoma_28 | 99.02 | 104.33 | 36,357 | 34,505 | 1.05× |
| devine_8 | 46.36 | 80.84 | 77,645 | 44,531 | 1.74× |
| hou_7 | 27.64 | 87.87 | 130,230 | 40,970 | 3.18× |
| viking_graben_10 | 128.03 | 74.74 | 28,119 | 48,168 | 0.58× |

