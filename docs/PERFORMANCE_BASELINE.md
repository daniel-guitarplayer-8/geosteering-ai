# Performance Baseline — Simulador Python (Numba JIT + JAX GPU)

| Campo | Valor |
|:------|:------|
| **Documento** | Baseline canônico de throughput do simulador (Numba CPU + JAX GPU) |
| **Versão atual** | v2.43 (Numba baseline v2.34 + JAX GPU baseline Sprint A1.6) |
| **Última atualização** | 2026-05-23 |
| **Hook anti-regressão** | [`.claude/hooks/check-perf-regression.sh`](../.claude/hooks/check-perf-regression.sh) |
| **Baseline JSON** | [`.claude/perf_baseline.json`](../.claude/perf_baseline.json) |
| **Seção JAX GPU** | [§9 abaixo](#9-jax-gpu-t4-baseline-sprint-a16) |

---

## 1. Tabela de Cenários Canônicos

| Cenário | n_pos | nf | nTR | nAng | n_models | Throughput baseline (mod/h) | Versão | Notas |
|:-------:|:-----:|:--:|:---:|:----:|:--------:|:---------------------------:|:------:|:------|
| **A** | 1 | 1 | 1 | 1 | 1000 | >800,000 | v2.29.1 | Single-pos rápido — meta empírica 3M+ |
| **B** | 100 | 1 | 1 | 1 | 1000 | >300,000 | v2.21 | Multi-pos baseline |
| **C** | 100 | 4 | 1 | 1 | 1000 | >100,000 | v2.21 | Multi-freq |
| **D** | 1 | 1 | 4 | 1 | 1000 | >200,000 | v2.21 | Multi-TR single-pos |
| **E** | 600 | 1 | 1 | 1 | 1000 | >140,000 | v2.29.1 | Inv0Dip 0° (DEFAULT) — meta 150k |
| **E** | 600 | 1 | 1 | 1 | 2000 | >95,000 | v2.29.2 | Config padrão usuário n=2000 |
| **E** | 600 | 1 | 1 | 1 | 200 | >65,000 | v2.29.3 | Baseline do hook anti-regressão |
| **F** | 100 | 4 | 4 | 1 | 500 | >50,000 | v2.29.1 | Multi-freq + multi-TR |
| **Multi-freq+dip** | 100 | 2 | 1 | 2 | 200 | >130,000 | v2.29.1 | Reprodutor do bug v2.29.1 |

---

## 2. Notas sobre Variabilidade

Throughput **NÃO é constante**: depende fortemente de:

1. **Warmup amortizado**: simulações menores (n=200) pagam fração maior do
   custo de JIT cold-start. Tabela mostra throughput por n_models distinto.

   | n_models | Throughput E (warm) | Razão |
   |:--------:|:-------------------:|:------|
   | 200 | ~65–70k mod/h | Warmup ainda significativo |
   | 500 | ~80–85k mod/h | Warmup amortizado parcialmente |
   | 1000 | ~92–95k mod/h | Próximo do steady-state |
   | 2000 | ~95–97k mod/h | Steady-state |

2. **Cache JIT em disco** (`__pycache__/*.nbi/.nbc`):
   - Cold cache (após `rm -rf __pycache__`): pode ser 50% mais lento
   - Warm cache: throughput estabilizado

3. **Thermal throttling** (especialmente Apple Silicon Mac):
   - Após 5+ minutos contínuos de simulação, CPU pode reduzir clock
   - Variação esperada: ±10% entre runs

4. **Outros processos no sistema**: editor de código, browser, Spotlight, etc.

**Conclusão**: variabilidade de ±10% entre execuções consecutivas é normal.
Regressão é considerada apenas se throughput cair **<90% do baseline** de
forma consistente (3 medições).

---

## 2.1. Warm-Cache Baseline (CI — Sprint v2.34)

A partir do **Sprint v2.34**, o workflow GitHub Actions executa um step
explícito de warmup (`geosteering-warmup --verbose`) antes do step de
benchmark. Isso **isola o cold-start JIT/LLVM do tempo medido** e produz uma
métrica `E_n200_warm` comparável entre PRs.

**Por que duas métricas (cold vs warm)?**

| Métrica | Quando medir | Propósito |
|:--------|:-------------|:----------|
| `E_n200` (cold) | Sem warmup prévio | Baseline histórico do hook local de devs |
| `E_n200_warm` (warm, CI) | Pós `geosteering-warmup` | Detectar regressões pós-merge sem ruído de cold-start |

**Interpretação**:

- `E_n200_warm` ≥ 110% de `E_n200` (cold) é esperado (cache JIT/LLVM quente).
- Queda > 10% em `E_n200_warm` entre runs consecutivas do CI indica
  regressão real no caminho hot do simulador (não em I/O de warmup).
- O hook local (`check-perf-regression.sh`) continua usando `E_n200` (cold)
  porque devs raramente rodam warmup antes de medir localmente.

**Onde fica registrada a métrica warm**: `.claude/perf_baseline.json` campo
`scenarios.E_n200_warm`. Placeholder atual (`null`) deve ser substituído pela
mediana de 3 runs do CI pós-merge do Sprint v2.34.

**Step do CI** (em `.github/workflows/ci.yml`):

```yaml
- name: Warm up JIT/LLVM cache (Sprint v2.32+)
  run: geosteering-warmup --verbose
  timeout-minutes: 5

- name: Benchmark smoke (cenário E n=200)
  run: python -m geosteering_ai.cli benchmark --scenario E --n 200 | tee benchmark_ci.log
  timeout-minutes: 10
  continue-on-error: true
```

`continue-on-error: true` é intencional: o objetivo é **observabilidade
histórica**, não gating. Regressões reais são caçadas via comparação manual
do log entre runs consecutivas no GitHub Actions.

---

## 3. Hook Anti-Regressão

O hook `.claude/hooks/check-perf-regression.sh` automatiza validação:

```bash
bash .claude/hooks/check-perf-regression.sh
# Saída esperada:
# [check-perf-regression] Cenário E n=200: 69336 mod/h
# [check-perf-regression] ✓ PASS (100% do baseline)
```

**Configuração**:

- Threshold: 90% (configurável via `THRESHOLD_PCT` env var)
- Cenário default: E (configurável via `SCENARIO` env var)
- n_models default: 200 (configurável via `N_MODELS` env var)

**Trigger sugerido** (em `settings.json`):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "command": ".claude/hooks/check-perf-regression.sh",
        "when": "tool_input.command =~ 'git commit'"
      }
    ]
  }
}
```

Não-bloqueante (`exit 0` mesmo em alerta) — apenas alerta no log.

---

## 4. Processo de Atualização

**Quando atualizar o baseline**:

1. **Otimização intencional**: nova versão melhora throughput em >+5%.
   Atualize valor em `.claude/perf_baseline.json` E nesta tabela.

2. **Hardware do desenvolvedor mudou**: caso novo dev assuma com máquina
   diferente, criar entrada separada e documentar context.

3. **Mudança de scenário canônico** (raro): coordenar com revisor físico.

**Como atualizar**:

```bash
# 1. Rode benchmark 3x (warm cache)
python -m geosteering_ai.cli benchmark --scenario E --n 1000  # x3

# 2. Calcule mediana
# 3. Atualize JSON:
python -c "
import json
with open('.claude/perf_baseline.json', 'r+') as f:
    data = json.load(f)
    data['scenarios']['E_n1000']['throughput_mod_h'] = NEW_VALUE
    data['scenarios']['E_n1000']['measured_at'] = '2026-XX-XXTHH:MM:SSZ'
    data['scenarios']['E_n1000']['version'] = 'vX.Y.Z'
    f.seek(0); f.truncate()
    json.dump(data, f, indent=2)
"

# 4. Atualize esta tabela
# 5. Commit: docs(perf): bump baseline E_n1000 → NEW mod/h (vX.Y.Z)
```

---

## 5. Histórico de Versões

| Versão | Data | Medição (E n=1000 warm) | Notas |
|:------:|:----:|:-----------------------:|:------|
| v2.21 | 2026-05-02 | 121,957 mod/h | Causa-raiz KB-013 corrigida |
| v2.22.4 | 2026-05-08 | ~125,000 mod/h | FLAT prange opt-in |
| v2.23 | 2026-05-10 | ~130,000 mod/h | Fastmath + threads adaptativos |
| v2.29 | 2026-05-11 | ~140,000 mod/h | Back to Basics (ephemeral pool) |
| v2.29.1 | 2026-05-11 | 145,202 mod/h | Fix NUMBA_NUM_THREADS no pai |
| v2.29.2 | 2026-05-11 | ~95,000 mod/h (n=2000) | Cache LRU configurável + auto-detect |
| v2.29.3 | 2026-05-11 | 95,124 mod/h (n=2000) | Confirmado NÃO há regressão v2.29.2 |

**Importante**: medições v2.29.2 e v2.29.3 com **n=2000** (config padrão
canônica do usuário). Medições anteriores com n=1000. Não confundir: o
overhead de warmup é fixo em ~5 s, então throughput n=1000 > n=2000 nominal
mas n=2000 reflete melhor o uso real.

---

## 6. Metas (Sprint v2.30+)

| Meta | Valor | Sprint |
|:-----|:------|:------:|
| Cenário E n=1000 warm | >150,000 mod/h | v2.30+ |
| Cenário E n=2000 warm | >100,000 mod/h | v2.30 |
| Cenário F (multi-freq+TR) | >60,000 mod/h | v2.30 |
| Cenário A | >1,000,000 mod/h | v2.31 (FLAT prange optimization) |
| pytest-qt golden path test | Verde em CI | v2.27 |

---

## 7. Referências

- [`analise_cenarios_otimizacao_simulador_numba.md`](reference/analise_cenarios_otimizacao_simulador_numba.md):
  análise detalhada por cenário com flamegraphs
- [`reports/v2.29.3_2026-05-11.md`](reports/v2.29.3_2026-05-11.md): relatório
  da investigação de regressão pós-v2.29.2 (não confirmada)
- [`reports/v2.29.1_2026-05-11.md`](reports/v2.29.1_2026-05-11.md): fix
  NUMBA_NUM_THREADS no pai (recuperou 150k em reprodutor)

---

## 8. TF Training Throughput (Sprint v2.40)

**Adicionado em**: v2.40 (2026-05-18) — I2.2 MCP colab-bridge + tf.data + Mixed Precision

Esta seção complementa as métricas do simulador Numba (Cenários A–H) com
**throughput de treinamento TensorFlow** em GPU Colab Pro+. Diferentemente
do simulador (CPU Mac M1/Intel), os benchmarks de treinamento são executados
**remotamente** via notebook `notebooks/colab_templates/benchmark_tfdata_mp16.ipynb`,
disparado pelo MCP `colab-mcp`.

### 8.1 Métrica e Metodologia

| Item | Valor |
|:---|:---|
| Métrica primária | `samples/sec` (mediana de 5 runs) |
| Modelo | ResNet_18 (default validado) |
| Dataset sintético | 1024 samples × seq_len=10 × 5 features |
| Batch size | 64 |
| Épocas por run | 3 (apenas para medir throughput; sem convergência) |
| Warmup | 1 epoch descartada (compilação XLA + cache GPU) |
| Reject threshold | stdev/mediana > 5% → re-run |
| Hardware Colab | T4 (free tier) ou A100 (Pro+ se disponível) |

### 8.2 Configurações Comparadas (4 cells)

| Config | `use_mixed_precision` | `use_xla` | tf.data tuning |
|:---|:---:|:---:|:---|
| **C1 baseline** | False | False | defaults |
| **C2 mp16** | True | False | defaults |
| **C3 mp16+XLA** | True | True | defaults |
| **C4 mp16+XLA+tf.data** | True | True | `tf_shuffle_buffer_size=4096`, `tf_num_parallel_calls=8`, `tf_prefetch_buffer_size=4` |

### 8.3 Baseline Esperado (a preencher via notebook)

| Config | T4 (samples/sec) | A100 (samples/sec) | Speedup vs C1 |
|:---|:---:|:---:|:---:|
| C1 baseline fp32 | _placeholder_ | _placeholder_ | 1.00x |
| C2 mp16 | _placeholder_ | _placeholder_ | _esperado ≥1.15x_ |
| C3 mp16+XLA | _placeholder_ | _placeholder_ | _esperado ≥1.30x_ |
| C4 mp16+XLA+tf.data | _placeholder_ | _placeholder_ | _esperado ≥1.40x_ |

JSON exportado por cada run: `docs/perf_baselines/v2.40_tf_training_{device}_{timestamp}.json`.

### 8.4 Gate de Aceitação Sprint v2.40

- **C2 (mp16)** deve ter speedup ≥ **1.15x** vs C1 em T4 (gate mínimo)
- Stdev/mediana < 5% em todas configs (qualidade da medição)
- Modelo construído com `build_model_with_mp_policy(config)` (D5 do plano)
  garantindo camadas em `compute_dtype='float16'` quando flag ativa

### 8.5 Configuração `PipelineConfig` (Sprint v2.40 D6)

4 campos novos parametrizam tf.data sem hardcodes:

```python
config = PipelineConfig(
    # Mixed Precision (já existia, agora com ordem correta de setup)
    use_mixed_precision=True,
    use_xla=True,
    # tf.data tuning (NOVOS em v2.40)
    tf_shuffle_buffer_size=10000,   # default; 0 desativa, max 100000
    tf_num_parallel_calls=-1,       # -1 = AUTOTUNE em runtime
    tf_prefetch_buffer_size=-1,     # -1 = AUTOTUNE em runtime
    tf_cache_eval=True,             # cache val/test (default legado)
)
```

### 8.6 Workflow de Re-medição

```bash
# Via MCP colab-bridge (manual ou via Claude Code)
# 1. Disparar via prompt natural: "Rode benchmark_tfdata_mp16 em Colab T4"
# 2. Skill geosteering-colab-mcp invoca MCP colab-mcp
# 3. Notebook executa 4 configs × 5 runs
# 4. JSON salvo em Drive
# 5. Download manual + commit em docs/perf_baselines/

# Ou manual (sem MCP):
# Upload benchmark_tfdata_mp16.ipynb para Colab → Runtime GPU → Run all
```

---

## 9. JAX GPU T4 Baseline (Sprint A1.6)

**Estabelecida em**: v2.43 (2026-05-23) — Sprint A1.6 `A-jax-gpu-benchmark-redesign`

Esta seção complementa a baseline Numba CPU (§1-§7) e TF Training (§8)
com a **baseline canônica oficial do simulador JAX GPU** em NVIDIA T4.

### 9.1 Validação Experimental

| Item | Valor |
|:-----|:------|
| Hardware | NVIDIA Tesla T4 (15 GB VRAM, n1-standard-4) |
| Plataforma | Google Colab Pro+ |
| Commit baseline | [`a06cf12`](https://github.com/daniel-guitarplayer-8/geosteering-ai/commit/a06cf122ef9b6a25b878915f2c2af4db9de53e2b) |
| Notebook | [`validate_jax_gpu_v240.ipynb`](../notebooks/colab_templates/validate_jax_gpu_v240.ipynb) |
| API | `simulate_multi_jax_batched` (Sprint A1.5, v2.42) |
| JAX version | 0.4.38+ (X64 enabled, `JAX_PLATFORMS="cuda,cpu"`) |
| Filtro Hankel | Werthmüller 201pt (paridade Fortran) |
| Precisão | `complex128` (sem mixed precision) |
| Paridade Fortran | 164/164 testes `pytest -m gpu` PASS (<1e-12) |
| Gate de aceitação | ≥1.5× Numba T4 LOCAL em A, B, E |
| Resultado | ✅ **GATE APROVADO** (A: 2.56×, B: 2.86×, E: 1.90×) |

### 9.2 Tabela Baseline (mediana hot, 4 runs)

| Cenário | n_pos | nf×TR×Ang | n_models | Throughput (mod/h) | vs Numba T4 | Threshold 90% | Membro do gate |
|:-------:|:-----:|:---------:|:--------:|-------------------:|:-----------:|--------------:|:--------------:|
| **A** |   1 | 1×1×1 | 50 | **6 899 537** | 2.56× | 6 209 583 | ✅ Sim |
| **B** | 100 | 1×1×1 | 50 |   **257 629** | 2.86× |   231 866 | ✅ Sim |
| **C** | 100 | 4×1×1 | 50 |   **151 073** | 5.70× |   135 966 | — |
| **D** |   1 | 1×4×1 | 50 | **4 931 078** | 5.11× | 4 437 970 | — |
| **E** | 600 | 1×1×1 | 50 |    **43 021** | 1.90× |    38 719 | ✅ Sim |
| **F** | 100 | 4×4×1 | 20 |    **36 468** | 4.51× |    32 821 | — |
| **G** | 100 | 4×4×4 |  5 |     **9 197** | 4.39× |     8 277 | — |
| **H** | 100 | 8×8×8 |  5 |        (OOM)  |    —  |        —  | — (N/A em T4) |

### 9.3 N_MODELS Calibrado por Cenário (T4 15 GB VRAM)

A memória XLA escala como `n_models × nTR × nAng × per_unit_buffer`.
Para evitar OOM em T4:

```python
N_MODELS_PER_SCENARIO = {
    "A": 50, "B": 50, "C": 50, "D": 50, "E": 50,  # gate scenarios — sem pressão VRAM
    "F": 20,  # 4 TR × 1 Ang — 50 → OOM 17.8 GB; 20 → ~7 GB
    "G": 5,   # 4 TR × 4 Ang — 4× pior que F
    "H": 5,   # 8 TR × 8 Ang — OOM mesmo com 5 (deferido A100)
}
```

Cenários A/B/C/D/E (gate + multi-config simples) usam **n_models=50** sem
risco de OOM. F/G/H usam valores reduzidos calibrados empiricamente.

### 9.4 Política de Não-Regressão

Toda alteração no caminho hot do simulador JAX deve:

1. **Manter paridade Fortran <1e-12** — pytest `-m gpu` 164/164 PASS
2. **Não regredir gate**: razão hot/Numba_T4_LOCAL ≥1.5× em A, B, E
3. **Não regredir throughput** ≥90% do baseline em A-G executáveis
4. **Documentar** qualquer mudança >+5%: bump `.claude/perf_baseline.json`
   + nova entrada na §9.6 (histórico)
5. **Consultar skill `geosteering-jax`** antes de editar `_jax/`

### 9.5 Workflow de Re-medição

```bash
# 1. Abrir Google Colab Pro+ com GPU T4 (Runtime → Change runtime type → GPU)
# 2. Upload notebooks/colab_templates/validate_jax_gpu_v240.ipynb
# 3. Em Cell 3, ajustar GIT_TAG="vX.YZ" (tag a validar)
# 4. Run All — espera-se ~30 min total (pytest + warmup + benchmark)
# 5. JSON gerado em /content/drive/MyDrive/Geosteering_AI/sprint_a16/
# 6. Download manual + commit em docs/perf_baselines/
# 7. Se gate PASS e throughput ≥ baseline:
#    - Bump .claude/perf_baseline.json::jax_gpu_t4
#    - Append entrada §9.6 abaixo
#    - Atualizar relatório docs/reports/vX.YZ_jax_gpu_baseline_t4_*.md
```

### 9.6 Histórico de Versões JAX GPU

| Versão | Data | Hardware | Gate (A/B/E) | A_hot | E_hot | Notas |
|:------:|:----:|:--------:|:------------:|:-----:|:-----:|:------|
| v2.43 | 2026-05-23 | T4 Colab | ✅ PASS (2.56×, 2.86×, 1.90×) | 6 899 537 | 43 021 | **Baseline canônica oficial** — Sprint A1.6 com batched API (batched-**unified**) |
| v2.44 | 2026-05-29 | A6000 local | n/a (ref local) | — | — | Sprint O4 batched-**bucketed** — ver §9.8 |

### 9.8 Sprint O4 — Batched-Bucketed (A6000 local, v2.44)

`simulate_multi_jax_batched` passou a usar o kernel **bucketed** (vmap dos kernels
de bucket sobre o eixo de modelos) quando a geometria é compartilhada entre os
modelos do batch (regime PINN/on-the-fly), em vez do `unified` hardcodado. O
`E_hot = 43 021` da baseline T4 v2.43 era medido com **batched-unified** — o
"teto de ~42.8k" era artefato desse caminho, não limite do simulador.

| Métrica (A6000 48 GB, c128) | batched-unified (pré-O4) | batched-bucketed (O4) | Ganho |
|:----------------------------|:------------------------:|:---------------------:|:-----:|
| On-the-fly 32 modelos × 600 pos | 65 100 mod/h | **1 466 000 mod/h** | **22.5×** |
| Cenário H (8×8×8 = 512 cfg) | OOM ~110 GB (pulado) | roda c/ `chunk_size_models=8` (32 600 mod/h) | destrava |

> **Gate**: a baseline oficial permanece `jax_gpu_t4` (§9.2). A seção
> `jax_gpu_a6000_o4` no JSON é **referência local** (não-gate). O `E_hot` T4
> deve ser **re-medido pós-O4** em Colab (batched agora usa bucketed). Ver
> [relatório v2.44](reports/v2.44_sprint_O4_batched_bucketed_2026-05-29.md).

### 9.7 Referências JAX GPU

- [Sprint A1.6 snapshot](sprints/v2.43.md) — notebook rewrite + 8 bug fixes
- [Sprint A1.5 snapshot](sprints/v2.42.md) — `simulate_multi_jax_batched` API
- [Relatório completo v2.43](reports/v2.43_jax_gpu_baseline_t4_2026-05-23.md) —
  análise detalhada desta baseline (paridade, warmup, speedups, OOM, próximos passos)
- [JSON bruto da medição](perf_baselines/sprint_a16_jax_batched_benchmark_t4_20260523_181558.json)
- [Audit v2.40.4](reports/v2.40.4_auditoria_resultados_sprint_a1.md) — 8 bugs metodológicos originais
- Skill `geosteering-jax` — expertise no backend JAX (uso obrigatório antes de tocar `_jax/`)
