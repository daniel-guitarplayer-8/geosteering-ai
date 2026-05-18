# Performance Baseline — Simulador Python Numba JIT

| Campo | Valor |
|:------|:------|
| **Documento** | Baseline canônico de throughput do simulador |
| **Versão atual** | v2.34 |
| **Última atualização** | 2026-05-15 |
| **Hook anti-regressão** | [`.claude/hooks/check-perf-regression.sh`](../.claude/hooks/check-perf-regression.sh) |
| **Baseline JSON** | [`.claude/perf_baseline.json`](../.claude/perf_baseline.json) |

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
