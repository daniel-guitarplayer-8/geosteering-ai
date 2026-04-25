# Benchmark Simulation Manager — Numba JIT vs Fortran tatu.x

## Config C — uniforme · f=[2k,6k]Hz · TR=[8.19,20.43]m · dip=0° · h1=10m · p_med=0.200m
| Modelo | Numba ms/mod | Fortran ms/mod | Numba mod/h | Fortran mod/h | Speedup |
|:-------|------------:|--------------:|-----------:|--------------:|:-------:|
| oklahoma_3 | 15.41 | 112.75 | 233,616 | 31,928 | 7.32× |
| oklahoma_5 | 31.66 | 115.53 | 113,700 | 31,161 | 3.65× |
| oklahoma_28 | 105.22 | 134.00 | 34,215 | 26,866 | 1.27× |
| devine_8 | 58.98 | 122.62 | 61,037 | 29,360 | 2.08× |
| hou_7 | 33.38 | 119.67 | 107,856 | 30,083 | 3.59× |
| viking_graben_10 | 160.46 | 102.08 | 22,435 | 35,268 | 0.64× |

