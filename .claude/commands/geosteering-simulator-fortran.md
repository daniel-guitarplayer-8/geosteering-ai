---
name: geosteering-simulator-fortran
description: |
  Simulador EM 1D TIV em Fortran (PerfilaAnisoOmp / tatu.x) do Geosteering AI.
  Cobre: 7 módulos Fortran, formato model.in v9.0, formato binário 22-col,
  Makefile (tatu.x, f2py, debug), otimizações OpenMP (Fases 0-5), Features F5/F6/F7,
  Filtro Adaptativo (Kong/Werthmuller/Anderson),
  scripts Python (fifthBuildTIVModels, buildValidamodels, batch_runner).
  Triggers: "Fortran", "tatu.x", "simulador", "PerfilaAnisoOmp", "model.in",
  "Hankel", "f2py", "batch_runner", "OpenMP", "thread_workspace", "cache",
  "multi-TR", "antenas inclinadas", "tilted", "compilar", "make",
  "compensação", "midpoint", "Kong", "Anderson", "filtro adaptativo".
---

# Geosteering AI — Simulador Fortran EM 1D TIV

Diretório: `Fortran_Gerador/` — Simulador EM 1D para meios TIV via transformada de Hankel.

---

## 1. Hierarquia de Módulos

```
parameters.f08         ← Constantes físicas (π, μ₀, ε₀, dp)
  ├─ filtersv2.f08     ← Filtros Hankel (J0/J1: Werth 201pt, Kong 61pt, Key, Anderson 801pt)
  ├─ utils.f08         ← commonarraysMD, commonfactorsMD, findlayersTR2well, RtHR
  ├─ magneticdipoles.f08 ← hmd_TIV_optimized(_ws), vmd_optimized(_ws), thread_workspace type
  └─ omp_lib           ← OpenMP runtime

PerfilaAnisoOmp.f08 (module DManisoTIV)  ← Solver principal
  └─ perfila1DanisoOMP: loop multi-TR × multi-θ × multi-f × medidas

RunAnisoOmp.f08 (program CampoMag1DAnisotropico)  ← Main: lê model.in, chama solver

tatu_f2py_wrapper.f08 (module tatu_wrapper)  ← Interface Python: simulate(), simulate_v8()
```

## 2. Subroutines Principais

| Módulo | Subroutine | Finalidade |
|:-------|:-----------|:-----------|
| **DManisoTIV** | `perfila1DanisoOMP` | Solver master (v9.0): loop itr→k→ii→j com OpenMP |
| | `fieldsinfreqs` | Legacy: calcula H(f) para 1 posição |
| | `fieldsinfreqs_ws` | Phase 3: com thread_workspace pré-alocado |
| | `fieldsinfreqs_cached_ws` | Phase 4: com cache de u/s/RT arrays |
| | `writes_files` | Output: .dat binário + .out info (multi-TR, F7 tilted) |
| | `writes_compensation_files` | F6: .dat compensado + atenuação (dB) |
| **magneticdipoles** | `hmd_TIV_optimized_ws` | Dipolo magnético horizontal (HMD) com workspace |
| | `vmd_optimized_ws` | Dipolo magnético vertical (VMD) com workspace |
| **utils** | `commonarraysMD` | Pré-computa u, s, uh, sh, RT*, AdmInt para Hankel |
| | `commonfactorsMD` | Fatores de reflexão/transmissão TE/TM por camada |
| | `findlayersTR2well` | Encontra índice da camada para profundidade z |
| | `RtHR` | Rotação do tensor H por ângulos de Euler |
| **filtersv2** | `J0J1Wer` | Filtro Hankel Werthmuller 201pt (default, precisão 10⁻⁶) |
| | `J0J1Kong` | Filtro Hankel Kong 61pt (rápido, precisão 10⁻⁴) |
| | `J0J1And` | Filtro Hankel Anderson 801pt (máxima precisão 10⁻⁸) |
| **tatu_wrapper** | `simulate` | Interface f2py v7.0 (sempre Werthmuller) |
| | `simulate_v8` | Interface f2py v9.0 (F5 + F6 + F7 + Filtro Adaptativo) |

## 3. Formato model.in (v9.0)

```
nf                    !número de frequências
freq(1)               !frequência 1 (Hz)
...freq(nf)
ntheta                !número de ângulos
theta(1)              !ângulo 1 (graus)
...theta(ntheta)
h1                    !altura 1º ponto-médio T-R (m)
tj                    !janela de investigação (m)
p_med                 !passo entre medidas (m)
nTR                   !número de pares T-R (Feature 1)
dTR(1)                !espaçamento T-R 1 (m)
...dTR(nTR)
filename              !nome base dos arquivos de saída
ncam                  !número de camadas
resh(1) resv(1)       !resistividades ρ_h, ρ_v camada 1 (Ω·m)
...resh(ncam) resv(ncam)
esp(2)...esp(ncam-1)  !espessuras das ncam-2 camadas internas (m)
modelm nmaxmodel      !modelo atual e nº máximo
[F5: use_arb_freq]    !(v8.0+ opcional, via iostat)
[F7: use_tilted]      !(v8.0+ opcional)
[F7: n_tilted]
[F7: beta(i) phi(i)]  !ângulo de tilt e azimute por config
[F6: use_compensation]!(v9.0 opcional, 0=desab. 1=hab.)
[F6: n_comp_pairs]    !(só se F6=1)
[F6: near(i) far(i)]  !pares T-R para compensação
[filter_type]         !(v9.0 opcional, 0=Werth 1=Kong 2=Anderson)
```

**nTR=1**: saída `{filename}.dat` (sem sufixo). **nTR>1**: `{filename}_TR{k}.dat`.
**F6**: saída `{filename}_COMP{p}.dat` e `{filename}_COMP{p}_ATT.dat` (compensação).

## 4. Formato Binário .dat (22 colunas)

```
col0  = med (int32)     — índice da medida
col1  = z_obs            — profundidade (m)
col2  = ρ_h              — resistividade horizontal (Ω·m)
col3  = ρ_v              — resistividade vertical (Ω·m)
col4-5   = Re/Im(Hxx)   — componente planar
col6-7   = Re/Im(Hxy)
col8-9   = Re/Im(Hxz)
col10-11 = Re/Im(Hyx)
col12-13 = Re/Im(Hyy)
col14-15 = Re/Im(Hyz)
col16-17 = Re/Im(Hzx)
col18-19 = Re/Im(Hzy)
col20-21 = Re/Im(Hzz)   — componente axial
```

172 bytes/registro. Layout: k(ângulo) → j(freq) → i(medida).

## 5. Build (Makefile)

| Target | Comando | Resultado |
|:-------|:--------|:----------|
| `make` | gfortran -O3 -march=native -fopenmp | `tatu.x` (executável) |
| `make f2py_wrapper` | numpy f2py | `tatu_f2py.so` (módulo Python) |
| `make debug_O0` | -O0 -fno-fast-math | Build determinístico (validação bit-exata) |
| `make run_python` | python fifthBuildTIVModels.py | Gera modelos TIV |
| `make all` | tatu.x + run_python | Build completo |
| `make clean` | rm build/ | Limpa artefatos |

## 6. Otimizações OpenMP (Fases 0-5)

| Fase | Técnica | Speedup |
|:-----|:--------|:--------|
| 0 | Baseline (serial) | 1× |
| 2 | Hybrid scheduler (outer θ + inner z) | ~4-6× |
| 3 | thread_workspace pré-alocado (elimina malloc/free) | ~10-12× |
| 3b | Workspace estendido (12 campos: Mxdw, Eudw, FEdwz...) | Robustez ncam≥30 |
| 4 | Cache commonarraysMD: u_cache(npt,n,nf,ntheta) | ~15-20× |
| 5 | schedule(guided, 16) + paralelismo adaptativo | ~20-25× (~58K mod/h) |

**thread_workspace type:** Buffers pré-alocados por thread (Tudw, Txdw, Tuup, Txup, TEdwz, TEupz + 6 reflection factors).

**Cache 4D:** `u_cache(npt, n, nf, ntheta)` — cada ângulo k escreve em slice independente, eliminando race condition.

## 7. Features v9.0

**F5 — Frequências Arbitrárias:** `use_arb_freq=1` habilita nf ∈ [1, 16]. Guard para nf>2 quando desabilitado.

**F6 — Compensação Midpoint (v9.0):** `use_compensation=1` com n_comp_pairs pares (near_itr, far_itr):
```
H_comp = (H_near + H_far) / 2          — tensor compensado (CDR)
Δφ = arg(H_near) − arg(H_far)          — diferença de fase (graus)
Δα = 20·log₁₀(|H_near|/|H_far|)       — atenuação (dB)
```
Requer nTR ≥ 2 (F1). Saída: `_COMP{i}.dat` (H_comp + phase_diff) + `_COMP{i}_ATT.dat` (atten_dB).
Custo negligível (~162 μs), pós-processamento sobre cH_all_tr.

**F7 — Antenas Inclinadas:** `use_tilted=1` com n_tilted configurações (β, φ):
```
H_tilted(β, φ) = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]
```
5 flops/ponto — negligível. Post-processamento sobre tensor H existente.

**Filtro Adaptativo (v9.0):** `filter_type` seleciona o filtro de Hankel:
```
filter_type=0: Werthmuller 201pt (default, precisão 10⁻⁶)
filter_type=1: Kong 61pt (3.3× mais rápido, precisão 10⁻⁴)
filter_type=2: Anderson 801pt (4× mais lento, precisão 10⁻⁸)
```
Kong recomendado para geração de datasets de treinamento; Anderson para validação.

## 8. Scripts Python

| Script | Função |
|:-------|:-------|
| `fifthBuildTIVModels.py` | Gerador de modelos TIV via Sobol QMC + Dirichlet (8 ensembles) |
| `buildValidamodels.py` | Validação com 6 modelos canônicos (Oklahoma, Devine, Hou et al.) |
| `batch_runner.py` | Orquestrador paralelo: ProcessPoolExecutor + concatenação binária |
| `bench/run_bench.sh` | Benchmark: wall-time, MD5, estatísticas |

## 9. Constantes Físicas (parameters.f08)

```fortran
dp = kind(1.d0)              ! double precision
pi = 3.141592653589793238...  ! π (15 dígitos)
mu = 4e-7 × π                ! μ₀ = 1.257e-6 H/m
epsilon = 8.85e-12            ! ε₀ F/m
Iw = 1.0                     ! corrente do transmissor (A)
```

## 10. Uso

```bash
# Compilar e executar
make && ./tatu.x              # lê model.in, gera *.dat

# Benchmark (12 threads, 60 iterações)
OMP_NUM_THREADS=12 bash bench/run_bench.sh --label prod --iters 60

# f2py (Python)
make f2py_wrapper
python -c "import tatu_f2py; zrho, cH = tatu_f2py.simulate(...)"
```

---

## 11. Documentação Detalhada (consulta on-demand)

Para informações detalhadas que excedem o escopo desta skill, consulte:

| Documento | Conteúdo | Tamanho |
|:----------|:---------|--------:|
| `docs/reference/documentacao_simulador_fortran_otimizado.md` | Documentação completa v10.0 (20 seções) | 381 KB |
| `docs/reference/analise_paralelismo_cpu_fortran.md` | OpenMP detalhado (6 fases, resultados empíricos) | 83 KB |
| `docs/reference/analise_novos_recursos_simulador_fortran.md` | F5/F7 + recursos futuros (1.5D, invasão, ∂H/∂ρ) | 66 KB |
| `docs/reference/modelo_geologico.md` | Modelos geológicos e geração estocástica | 35 KB |
| `docs/reference/relatorio_fase0_fase1_fortran.md` | Baseline + SIMD (Fase 0-1) | 23 KB |
| `docs/reference/relatorio_fase2_debitos_fortran.md` | Hybrid scheduler (Fase 2) | 22 KB |
| `docs/reference/relatorio_fase3_fortran.md` | Thread workspace (Fase 3) | 17 KB |
| `docs/reference/relatorio_fase4_fortran.md` | Cache commonarraysMD (Fase 4) | 18 KB |
| `docs/reference/relatorio_validacao_final_fortran.md` | Validação bit-exata final | 14 KB |

**Instrução ao Claude:** Ao precisar de detalhes sobre fórmulas matemáticas, diagramas
de fluxo OpenMP, ou análise de performance, leia o documento relevante diretamente
com `Read(file_path="docs/reference/...", offset=X, limit=Y)`.
