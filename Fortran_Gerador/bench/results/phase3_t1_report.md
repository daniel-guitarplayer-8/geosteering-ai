# Relatório de Benchmark — phase3_t1

**Gerado em**: 2026-04-05T18:01:48Z
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
| Wall-time médio (s/modelo)     | 1.3690                   |
| Desvio-padrão (s)              | 0.0594                  |
| Mínimo (s)                     | 1.2800                   |
| Máximo (s)                     | 1.6000                   |
| Mediana (s)                    | 1.3600                 |
| **Throughput (modelos/hora)**  | **2629.7**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `aadbc86be2af5e1fd300f535d7e80e3b`                           |

## Série Bruta (segundos)

```
1.32
1.34
1.44
1.34
1.33
1.39
1.38
1.30
1.31
1.28
1.31
1.38
1.35
1.39
1.36
1.33
1.60
1.35
1.33
1.34
1.37
1.42
1.37
1.42
1.45
1.40
1.36
1.37
1.36
1.38
```
