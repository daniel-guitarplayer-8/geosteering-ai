# Relatório de Benchmark — phase4_t2

**Gerado em**: 2026-04-05T19:20:22Z
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
| Wall-time médio (s/modelo)     | 0.1390                   |
| Desvio-padrão (s)              | 0.0055                  |
| Mínimo (s)                     | 0.1300                   |
| Máximo (s)                     | 0.1600                   |
| Mediana (s)                    | 0.1400                 |
| **Throughput (modelos/hora)**  | **25899.3**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `1e4f36fa8f0bcd21f3700dc445c2894d`                           |

## Série Bruta (segundos)

```
0.14
0.14
0.14
0.14
0.14
0.14
0.13
0.14
0.14
0.14
0.14
0.14
0.13
0.13
0.14
0.14
0.14
0.14
0.14
0.16
0.14
0.14
0.14
0.14
0.14
0.13
0.13
0.14
0.14
0.14
```
