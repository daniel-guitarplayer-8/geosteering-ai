# Relatório de Benchmark — phase2_n29

**Gerado em**: 2026-04-05T17:57:55Z
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
| Wall-time médio (s/modelo)     | 0.3830                   |
| Desvio-padrão (s)              | 0.0355                  |
| Mínimo (s)                     | 0.3100                   |
| Máximo (s)                     | 0.4600                   |
| Mediana (s)                    | 0.3800                 |
| **Throughput (modelos/hora)**  | **9399.5**         |

## Saída Binária

| Campo        | Valor                              |
|:-------------|:-----------------------------------|
| Arquivo      | `Inv0_15Dip1000_t5.dat`                     |
| Tamanho      | 206400 bytes                    |
| MD5          | `602d82205e4ab33ef8a82f9cf1350fe0`                           |

## Série Bruta (segundos)

```
0.34
0.37
0.31
0.32
0.36
0.32
0.35
0.35
0.34
0.37
0.34
0.33
0.37
0.34
0.36
0.37
0.35
0.39
0.34
0.37
0.38
0.35
0.38
0.38
0.37
0.40
0.37
0.40
0.37
0.41
0.39
0.39
0.42
0.36
0.38
0.37
0.40
0.41
0.36
0.40
0.36
0.39
0.39
0.37
0.39
0.37
0.43
0.42
0.45
0.42
0.43
0.44
0.41
0.44
0.41
0.43
0.41
0.46
0.42
0.46
```
