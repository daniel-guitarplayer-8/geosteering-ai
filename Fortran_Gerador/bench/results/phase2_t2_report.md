# Relatório de Benchmark — phase2_t2

**Gerado em**: 2026-04-05T18:09:26Z
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
| Wall-time médio (s/modelo)     | 0.8073                   |
| Desvio-padrão (s)              | 0.0186                  |
| Mínimo (s)                     | 0.7800                   |
| Máximo (s)                     | 0.8600                   |
| Mediana (s)                    | 0.8000                 |
| **Throughput (modelos/hora)**  | **4459.1**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `602d82205e4ab33ef8a82f9cf1350fe0`                           |

## Série Bruta (segundos)

```
0.86
0.81
0.80
0.81
0.80
0.80
0.83
0.80
0.79
0.86
0.82
0.82
0.81
0.79
0.82
0.80
0.80
0.81
0.80
0.79
0.79
0.79
0.78
0.80
0.80
0.80
0.82
0.81
0.82
0.79
```
