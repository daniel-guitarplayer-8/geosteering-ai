# Performance Baseline — Simulador Python Numba JIT

| Campo | Valor |
|:------|:------|
| **Documento** | Baseline canônico de throughput do simulador |
| **Versão atual** | v2.29.3 |
| **Última atualização** | 2026-05-11 |
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
