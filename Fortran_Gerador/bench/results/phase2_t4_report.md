# Relatório de Benchmark — phase2_t4

**Gerado em**: 2026-04-05T18:10:35Z
**Script**: `bench/run_bench.sh`

## Ambiente

| Campo                | Valor                                   |
|:---------------------|:----------------------------------------|
| Sistema operacional  | Darwin                                |
| CPU                  | Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz                              |
| Núcleos físicos      | 8                             |
| Núcleos lógicos      | 16                              |
| Compilador           | GNU Fortran (Homebrew GCC 15.2.0_1) 15.2.0                           |
| `OMP_NUM_THREADS`  | 4                                |
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
| Wall-time médio (s/modelo)     | 0.5873                   |
| Desvio-padrão (s)              | 0.1665                  |
| Mínimo (s)                     | 0.4500                   |
| Máximo (s)                     | 1.1700                   |
| Mediana (s)                    | 0.5450                 |
| **Throughput (modelos/hora)**  | **6129.4**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `602d82205e4ab33ef8a82f9cf1350fe0`                           |

## Série Bruta (segundos)

```
0.49
0.54
0.48
0.58
0.52
0.54
0.99
1.17
0.87
0.78
0.68
0.55
0.60
0.65
0.56
0.58
0.58
0.52
0.50
0.61
0.45
0.56
0.46
0.46
0.45
0.45
0.46
0.57
0.48
0.49
```
