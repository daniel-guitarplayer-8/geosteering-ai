# Relatório de Benchmark — phase5_t1

**Gerado em**: 2026-04-05T23:59:45Z
**Script**: `bench/run_bench.sh`

## Ambiente

| Campo                | Valor                                   |
|:---------------------|:----------------------------------------|
| Sistema operacional  | Darwin                                |
| CPU                  | Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz                              |
| Núcleos físicos      | 8                             |
| Núcleos lógicos      | 16                              |
| Compilador           | GNU Fortran (Homebrew GCC 15.2.0_1) 15.2.0                           |
| `OMP_NUM_THREADS`  | 1                                |
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
| Wall-time médio (s/modelo)     | 0.2583                   |
| Desvio-padrão (s)              | 0.0172                  |
| Mínimo (s)                     | 0.2400                   |
| Máximo (s)                     | 0.3300                   |
| Mediana (s)                    | 0.2600                 |
| **Throughput (modelos/hora)**  | **13935.5**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `3d3c309fd1aa121f8b4166268552814c`                           |

## Série Bruta (segundos)

```
0.25
0.26
0.30
0.26
0.26
0.33
0.26
0.26
0.26
0.25
0.26
0.26
0.25
0.26
0.26
0.25
0.25
0.25
0.26
0.26
0.24
0.25
0.25
0.25
0.25
0.24
0.26
0.24
0.26
0.26
```
