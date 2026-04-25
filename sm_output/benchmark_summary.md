# Benchmark Simulation Manager — Numba JIT vs Fortran tatu.x

## Config A — 1 freq · 1 TR · 1 dip · 600 pos
| Modelo | Numba ms/mod | Fortran ms/mod | Numba mod/h | Fortran mod/h | Speedup |
|:-------|------------:|--------------:|-----------:|--------------:|:-------:|
| oklahoma_3 | 15.48 | 67.40 | 232,575 | 53,414 | 4.35× |
| oklahoma_5 | 19.17 | 55.80 | 187,839 | 64,518 | 2.91× |
| oklahoma_28 | 30.22 | 56.01 | 119,131 | 64,274 | 1.85× |
| devine_8 | 18.68 | 54.91 | 192,708 | 65,565 | 2.94× |
| hou_7 | 18.54 | 54.97 | 194,126 | 65,489 | 2.96× |
| viking_graben_10 | 20.23 | 54.05 | 177,962 | 66,599 | 2.67× |

## Config B — 2 freq · 1 TR · 2 dips · 600 pos
| Modelo | Numba ms/mod | Fortran ms/mod | Numba mod/h | Fortran mod/h | Speedup |
|:-------|------------:|--------------:|-----------:|--------------:|:-------:|
| oklahoma_3 | 64.66 | 126.59 | 55,672 | 28,439 | 1.96× |
| oklahoma_5 | 85.24 | 130.66 | 42,233 | 27,552 | 1.53× |
| oklahoma_28 | 104.63 | 118.93 | 34,408 | 30,270 | 1.14× |
| devine_8 | 68.97 | 118.76 | 52,193 | 30,314 | 1.72× |
| hou_7 | 66.81 | 127.07 | 53,881 | 28,331 | 1.90× |
| viking_graben_10 | 78.11 | 122.99 | 46,091 | 29,270 | 1.57× |

