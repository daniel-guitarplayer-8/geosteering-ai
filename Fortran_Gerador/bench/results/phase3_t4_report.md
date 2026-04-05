# Relatório de Benchmark — phase3_t4

**Gerado em**: 2026-04-05T18:04:01Z
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
| Wall-time médio (s/modelo)     | 0.4617                   |
| Desvio-padrão (s)              | 0.0207                  |
| Mínimo (s)                     | 0.4300                   |
| Máximo (s)                     | 0.5200                   |
| Mediana (s)                    | 0.4600                 |
| **Throughput (modelos/hora)**  | **7797.8**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `aadbc86be2af5e1fd300f535d7e80e3b`                           |

## Série Bruta (segundos)

```
0.45
0.43
0.44
0.43
0.47
0.45
0.44
0.44
0.44
0.46
0.46
0.47
0.45
0.47
0.45
0.47
0.45
0.49
0.47
0.45
0.45
0.46
0.45
0.47
0.47
0.49
0.48
0.52
0.49
0.49
```
