# Relatório de Benchmark — phase3

**Gerado em**: 2026-04-05T17:59:45Z
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
| Wall-time médio (s/modelo)     | 0.3433                   |
| Desvio-padrão (s)              | 0.0278                  |
| Mínimo (s)                     | 0.2900                   |
| Máximo (s)                     | 0.4100                   |
| Mediana (s)                    | 0.3400                 |
| **Throughput (modelos/hora)**  | **10485.4**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `aadbc86be2af5e1fd300f535d7e80e3b`                           |

## Série Bruta (segundos)

```
0.35
0.30
0.32
0.33
0.29
0.34
0.31
0.30
0.30
0.31
0.33
0.32
0.31
0.30
0.32
0.32
0.32
0.33
0.34
0.35
0.33
0.31
0.33
0.34
0.32
0.33
0.35
0.33
0.34
0.32
0.34
0.35
0.32
0.38
0.36
0.33
0.35
0.35
0.34
0.37
0.34
0.38
0.38
0.35
0.37
0.35
0.37
0.38
0.35
0.37
0.36
0.38
0.39
0.37
0.38
0.34
0.36
0.41
0.39
0.40
```
