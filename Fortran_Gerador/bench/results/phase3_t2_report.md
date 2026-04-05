# Relatório de Benchmark — phase3_t2

**Gerado em**: 2026-04-05T18:02:58Z
**Script**: `bench/run_bench.sh`

## Ambiente

| Campo                | Valor                                   |
|:---------------------|:----------------------------------------|
| Sistema operacional  | Darwin                                |
| CPU                  | Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz                              |
| Núcleos físicos      | 8                             |
| Núcleos lógicos      | 16                              |
| Compilador           | GNU Fortran (Homebrew GCC 15.2.0_1) 15.2.0                           |
| `OMP_NUM_THREADS`  | 2                                |
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
| Wall-time médio (s/modelo)     | 0.7500                   |
| Desvio-padrão (s)              | 0.0292                  |
| Mínimo (s)                     | 0.7100                   |
| Máximo (s)                     | 0.8500                   |
| Mediana (s)                    | 0.7450                 |
| **Throughput (modelos/hora)**  | **4800.0**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `aadbc86be2af5e1fd300f535d7e80e3b`                           |

## Série Bruta (segundos)

```
0.77
0.77
0.74
0.71
0.74
0.73
0.72
0.75
0.75
0.71
0.72
0.74
0.76
0.77
0.76
0.74
0.73
0.74
0.75
0.75
0.77
0.82
0.75
0.76
0.74
0.77
0.73
0.73
0.73
0.85
```
