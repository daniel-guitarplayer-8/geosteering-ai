# Relatório de Benchmark — baseline

**Gerado em**: 2026-04-04T20:58:28Z
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
| Iterações                      | 30                      |
| Wall-time médio (s/modelo)     | 0.1137                   |
| Desvio-padrão (s)              | 0.0185                  |
| Mínimo (s)                     | 0.0900                   |
| Máximo (s)                     | 0.1600                   |
| Mediana (s)                    | 0.1100                 |
| **Throughput (modelos/hora)**  | **31671.6**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `c64745ed5d69d5f654b0bac7dde23a95`                           |

## Série Bruta (segundos)

```
0.15
0.11
0.11
0.10
0.14
0.11
0.10
0.10
0.14
0.10
0.11
0.10
0.10
0.15
0.11
0.11
0.10
0.14
0.09
0.10
0.10
0.11
0.11
0.11
0.11
0.13
0.16
0.10
0.11
0.10
```
