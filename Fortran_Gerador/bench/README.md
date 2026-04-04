# Bench — Infraestrutura de Benchmark do Simulador Fortran

Diretório de benchmarking do simulador `PerfilaAnisoOmp` / `tatu.x`, criado para execução das **Fases 0 e 1** do roteiro de otimização CPU descrito em [`../../docs/reference/analise_paralelismo_cpu_fortran.md`](../../docs/reference/analise_paralelismo_cpu_fortran.md) §7.

## Objetivo

Medir o desempenho do simulador de forma **reprodutível e comparável** entre versões do código, detectando:

- **Throughput**: modelos/segundo com diferentes números de threads OpenMP.
- **Speedup**: razão entre tempo de execução sequencial e paralelo.
- **Regressões numéricas**: garantia de que otimizações não alteram a saída em norma L² acima de `1e-10`.

## Protocolo de Benchmark

O benchmark roda `tatu.x` **N vezes consecutivas** (`--iters N`, default 30) sobre o mesmo `model.in`, removendo os arquivos de saída entre as execuções (imposto pela semântica `position='append'` de [`writes_files`](../PerfilaAnisoOmp.f08) — ver §Débitos Técnicos). Cada iteração mede *wall time* via `/usr/bin/time -p`, e o script reporta média ± desvio-padrão.

### Invariantes físicos

O `model.in` padrão (herdado da raiz do repositório) define:

| Parâmetro          | Valor        | Unidade                    |
|:-------------------|:-------------|:---------------------------|
| `nf`               | 2            | frequências (20/40 kHz)    |
| `ntheta`           | 1            | ângulos de inclinação      |
| `theta(1)`         | 0,0          | graus                      |
| `h1`               | 10,0         | m (altura 1º ponto-médio)  |
| `tj`               | 120,0        | m (janela vertical)        |
| `p_med`            | 0,2          | m (passo entre medidas)    |
| `dTR`              | 1,0          | m (arranjo T-R)            |
| `ncam`             | 10           | camadas                    |
| Filtro             | 201 pontos   | Werthmüller J0/J1          |

## Arquivos

| Arquivo                          | Finalidade                                                                  |
|:---------------------------------|:----------------------------------------------------------------------------|
| [`run_bench.sh`](run_bench.sh)   | Script principal dual-OS (macOS + Linux). Detecta SO, compila, executa, md5|
| [`validate_numeric.py`](validate_numeric.py) | Compara `.dat` binários de baseline e otimização (numpy `allclose`)|
| [`results/`](results/)           | Saídas dos benchmarks (não versionadas além dos relatórios)                |
| [`results/baseline_*.md`](results/) | Relatórios gerados pelos benchmarks                                       |

## Uso

```bash
# 1. Fase 0 — baseline (sem alterações no código Fortran)
cd Fortran_Gerador
bash bench/run_bench.sh --label baseline --iters 30 --threads 8

# 2. Fase 1 — após aplicação de diretivas SIMD
bash bench/run_bench.sh --label phase1 --iters 30 --threads 8

# 3. Validação numérica
python3 bench/validate_numeric.py \
    bench/results/baseline_output.dat \
    bench/results/phase1_output.dat
```

## Ambientes Suportados

| SO             | Compilador              | Comando timing         | Contagem cores                    |
|:---------------|:------------------------|:-----------------------|:----------------------------------|
| **macOS**      | gfortran (Homebrew)     | `/usr/bin/time -p`     | `sysctl -n hw.physicalcpu`        |
| **Linux**      | gfortran (distro)       | `/usr/bin/time -p`     | `nproc`                           |

`OMP_PLACES=cores` e `OMP_PROC_BIND=close` são aplicados em Linux; no macOS são ignorados silenciosamente pelo runtime OpenMP do GCC (limitação conhecida).

## Débitos Técnicos Detectados

Durante a preparação deste benchmark foram identificados **bugs de reprodutibilidade** no código de produção:

1. **`writes_files` em `PerfilaAnisoOmp.f08:246`** usa `position='append'` com `status='unknown'`. Múltiplas execuções sem limpeza concatenam dados ao arquivo de saída (o arquivo final pode conter 1 + N modelos em vez de 1). O `run_bench.sh` contorna isso removendo `*.dat` e `*.out` antes de cada iteração. **Correção recomendada (fase futura)**: trocar para `status='replace'` quando `modelm==1` e `position='append'` apenas para `modelm>1`.

2. **`omp_set_nested(.true.)` em `PerfilaAnisoOmp.f08:74`** — API depreciada desde OpenMP 5.0. Substituir por `OMP_MAX_ACTIVE_LEVELS=2` (variável de ambiente) ou `call omp_set_max_active_levels(2)` em fase futura.

3. **Documentação `analise_paralelismo_cpu_fortran.md` §7.1** localiza `commonarraysMD` "dentro de `PerfilaAnisoOmp`" — na verdade está em [`utils.f08`](../utils.f08) linhas 158–241. Correção aplicada na atualização dos MDs (ver relatório final).

## Referências

- [`docs/reference/documentacao_simulador_fortran.md`](../../docs/reference/documentacao_simulador_fortran.md) §11 — análise OpenMP e profiling esperado
- [`docs/reference/analise_paralelismo_cpu_fortran.md`](../../docs/reference/analise_paralelismo_cpu_fortran.md) §7 — roteiro das 6 fases de otimização
- [`docs/reference/relatorio_fase0_fase1_fortran.md`](../../docs/reference/relatorio_fase0_fase1_fortran.md) — relatório de execução destas Fases 0/1 (gerado por este benchmark)
