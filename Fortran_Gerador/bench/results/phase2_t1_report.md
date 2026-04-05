# Relatório de Benchmark — phase2_t1

**Gerado em**: 2026-04-05T18:08:07Z
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
| Wall-time médio (s/modelo)     | 1.7800                   |
| Desvio-padrão (s)              | 0.4390                  |
| Mínimo (s)                     | 1.5900                   |
| Máximo (s)                     | 3.5300                   |
| Mediana (s)                    | 1.6400                 |
| **Throughput (modelos/hora)**  | **2022.5**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `602d82205e4ab33ef8a82f9cf1350fe0`                           |

## Série Bruta (segundos)

```
1.71
1.69
1.64
1.66
1.65
2.38
3.53
1.71
3.05
1.65
1.61
1.63
1.63
1.62
1.62
1.65
1.64
1.61
1.59
1.83
1.62
1.63
1.63
1.59
1.59
1.64
1.72
1.60
1.63
1.65
```
