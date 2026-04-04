# Relatório de Benchmark — phase2

**Gerado em**: 2026-04-04T21:43:38Z
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
| Wall-time médio (s/modelo)     | 0.2292                   |
| Desvio-padrão (s)              | 0.0166                  |
| Mínimo (s)                     | 0.1900                   |
| Máximo (s)                     | 0.2600                   |
| Mediana (s)                    | 0.2300                 |
| **Throughput (modelos/hora)**  | **15709.1**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `c64745ed5d69d5f654b0bac7dde23a95`                           |

## Série Bruta (segundos)

```
0.23
0.22
0.22
0.19
0.22
0.22
0.23
0.21
0.23
0.23
0.21
0.20
0.22
0.22
0.21
0.21
0.22
0.22
0.21
0.20
0.22
0.22
0.22
0.22
0.21
0.21
0.22
0.21
0.23
0.23
0.22
0.22
0.24
0.26
0.22
0.25
0.24
0.24
0.23
0.25
0.25
0.23
0.25
0.25
0.25
0.23
0.25
0.25
0.23
0.24
0.26
0.25
0.23
0.25
0.25
0.23
0.24
0.26
0.25
0.22
```
