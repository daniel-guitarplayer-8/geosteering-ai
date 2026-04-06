# Relatório de Benchmark — phase3b5b

**Gerado em**: 2026-04-06T00:22:17Z
**Script**: `bench/run_bench.sh`

## Ambiente

| Campo                | Valor                                   |
|:---------------------|:----------------------------------------|
| Sistema operacional  | Darwin                                |
| CPU                  | Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz                              |
| Núcleos físicos      | 8                             |
| Núcleos lógicos      | 16                              |
| Compilador           | GNU Fortran (Homebrew GCC 15.2.0_1) 15.2.0                           |
| `OMP_NUM_THREADS`  | 8                                |
| Flags extras         | (nenhuma)             |

## Configuração do Modelo (`model.in`)

- Frequências: 2 (20 kHz, 40 kHz)
- Ângulo: 1 (0°)
- Camadas: 10
- Filtro Hankel: 201 pontos (Werthmüller J0/J1)
- Medidas por modelo: ~600 (janela 120 m, passo 0,2 m)

## Resultados

| Métrica                        | Valor                   |
|:-------------------------------|:------------------------|
| Iterações                      | 60                      |
| Wall-time médio (s/modelo)     | 0.0668                   |
| Desvio-padrão (s)              | 0.0050                  |
| Mínimo (s)                     | 0.0600                   |
| Máximo (s)                     | 0.0800                   |
| Mediana (s)                    | 0.0700                 |
| **Throughput (modelos/hora)**  | **53865.3**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `3d3c309fd1aa121f8b4166268552814c`                           |

## Série Bruta (segundos)

```
0.07
0.07
0.06
0.06
0.06
0.06
0.06
0.07
0.06
0.06
0.07
0.07
0.07
0.07
0.07
0.06
0.06
0.07
0.07
0.06
0.06
0.07
0.07
0.06
0.07
0.07
0.07
0.06
0.06
0.07
0.06
0.06
0.07
0.07
0.07
0.07
0.07
0.07
0.07
0.06
0.07
0.07
0.07
0.07
0.07
0.07
0.07
0.07
0.06
0.08
0.07
0.07
0.07
0.07
0.06
0.07
0.07
0.07
0.07
0.06
```
