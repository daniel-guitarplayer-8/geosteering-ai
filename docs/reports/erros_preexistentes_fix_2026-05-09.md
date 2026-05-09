# Relatório Técnico — Conserto de Erros Pré-Existentes (Fase 1 Bonus)

**Data**: 2026-05-09
**Branch**: `feat/fase1-fundacao-multiagente`
**Commits desta sessão**: 2 (`4509160` fix + `3e88592` docs sync)
**Modelo**: Claude Opus 4.7 (operador)
**Documento base**: `docs/reports/arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md`

---

## 1. Sumário Executivo

Esta sessão complementar à Fase 1 (§22.1) atacou os **3 erros pré-existentes**
em `geosteering_ai/simulation/multi_forward.py` que vinham sendo bypassados via
`SKIP=mypy,ruff` em commits dos últimos 5 dias. Todos foram **corrigidos com fixes
triviais (~15 linhas)**, sem regressão funcional. Adicionalmente, sincronizamos
`CLAUDE.md` e `docs/ROADMAP.md` que estavam declarando `v2.21` (desatualizados em
~7 dias) para `v2.22.6` (estado real).

| Métrica | Antes | Depois |
|:--------|:-----:|:------:|
| Erros ruff em `multi_forward.py` | 1 (F821) | **0** |
| Erros mypy em `multi_forang.py` | 2 (Optional widening) | **0** |
| Necessidade de `SKIP=mypy,ruff` | sim (5 commits) | **não** |
| CodeRabbit findings (`--agent --base main`) | 0 | **0** (mantido) |
| `pytest test_known_bugs.py` | 11/11 | 11/11 |
| `pytest test_simulation_v22_flat_prange.py` | 27/27 | 27/27 |
| `pytest tools/{pv,np}-mcp/tests/` | 32/32 | 32/32 |
| Doc `CLAUDE.md` Simulation Manager | v2.21 (2026-05-02) | **v2.22.6 (2026-05-09)** |
| Doc `ROADMAP.md` versão | 1.1 (Abril 2026) | **1.2 (Maio 2026)** |

---

## 2. Investigação Pré-Sessão

### 2.1 Como os erros foram identificados

A sessão anterior (Fase 1 fundação multi-agente) precisou usar `SKIP=mypy,ruff`
em 4 dos 10 commits. A mensagem operador foi:

> *"Os erros F821 MultiSimulationResultBatch (forward-ref) e mypy em comp_pairs/
> tilted_configs são pré-existentes (não introduzidos pelas minhas mudanças —
> estão documentados como deferred no plano)."*

Nesta sessão, **dois agentes Explore em paralelo** fizeram diagnóstico completo:

| Agente | Foco | Output |
|:-------|:-----|:-------|
| #1 | Investigação dos erros (`ruff check`, `mypy`, `git blame`, `grep`) | 3 erros confirmados, todos pré-existentes; recomendação `FIX_NOW` para todos |
| #2 | Mapeamento de gaps em §22 + KBs + docs obsoletos | Doc obsoletas (ROADMAP+CLAUDE 7 dias), I1.7 .worktreeinclude pendente, MCPs Fase 2-3 não criados (esperado) |

### 2.2 Resultados do diagnóstico — outros gaps

| Item | Status | Decisão |
|:-----|:------:|:--------|
| 22 `pytest.skip/xfail` em `tests/` | OK | Todos legítimos (TF/Fortran/JAX/known bugs) |
| 1 `TODO(v1.1)` em `consensus-mcp-server` | OK | Não-crítico, agendado para v1.1 do MCP |
| `CLAUDE.md` declara v2.21 (real: v2.22.5/v2.22.6) | ❌ → ✅ | Corrigido nesta sessão |
| `docs/ROADMAP.md` declara v2.21 | ❌ → ✅ | Corrigido nesta sessão |
| `.worktreeinclude` (I1.7) ausente | DEFER | Não-bloqueante para Fase 2; criar em I1.7 dedicado |
| MCPs `colab-bridge` (I2.2), `mlflow-tracker` (I3.1) ausentes | DEFER | Esperado (são Fase 2/3) |
| `.backups/`, `latex-skill.zip`, `Resultados_Relatorio_2026/` em untracked | OK | Já em `.gitignore`; não são lixo "público" |

---

## 3. Resposta às Perguntas do Usuário

### 3.1 *"Que erros pré-existentes são esses?"*

| # | Tipo | Localização original | Mensagem |
|:-:|:-----|:--------------------|:---------|
| 1 | **F821 (ruff)** | `multi_forward.py:676` | `Undefined name MultiSimulationResultBatch` |
| 2 | **mypy arg-type** | `multi_forward.py:1258` | `Argument "comp_pairs" to "apply_compensation" has incompatible type "tuple[tuple[int, int], ...] \| None"; expected "tuple[tuple[int, int], ...]"` |
| 3 | **mypy arg-type** | `multi_forward.py:1271` | `Argument "tilted_configs" to "apply_tilted_antennas" has incompatible type "tuple[tuple[float, float], ...] \| None"; expected "tuple[tuple[float, float], ...]"` |

**Erro 1 (F821)** — A assinatura `simulate_multi() -> Union["MultiSimulationResult", "MultiSimulationResultBatch"]`
usa **forward-reference em string** para `MultiSimulationResultBatch`. Essa
classe está definida em `geosteering_ai/simulation/_workers.py:742` (Sprint
12.1, v2.12) e é importada **lazy** dentro de `simulate_multi` para evitar custo
de spawn do pool de workers quando o usuário não opt-in. Como nenhum import
top-level fornece o nome ao escopo do módulo, ruff `F821` reclama.

**Erros 2+3 (mypy)** — `simulate_multi` declara `comp_pairs: Optional[Tuple[...]] = None`
e `tilted_configs: Optional[Tuple[...]] = None`, mas `apply_compensation`
e `apply_tilted_antennas` (em `postprocess/`) exigem **non-None** explícito.
O código histórico está dentro de `if use_compensation:` e `if use_tilted:`,
mas mypy **não consegue narrowing automático** baseado em flag bool externa
sem uma checagem `is not None` direta na variável.

### 3.2 *"Os erros pré-existentes foram corrigidos?"*

**SIM** — todos os 3, nesta sessão, no commit `4509160`. Validação automatizada:

```bash
ruff check geosteering_ai/simulation/multi_forward.py
# All checks passed!

mypy --ignore-missing-imports --follow-imports=silent geosteering_ai/simulation/multi_forward.py
# Success: no issues found in 1 source file
```

Pre-commit hook do commit `4509160` rodou **sem `SKIP=mypy,ruff`** e passou
todos os 9 hooks (ruff, ruff-format, mypy, anti-patterns, paridade Fortran).

### 3.3 *"Por que eles haviam sido deferidos?"*

Razões históricas levantadas via `git blame` + análise de mensagens de commit:

| Razão | Quando | Magnitude |
|:------|:------:|:---------:|
| Sprints anteriores priorizaram **features sobre type-check cleanliness** (Sprint 11 introduziu Optional widening; Sprint 12 introduziu forward-ref por design lazy import) | 2026-04-14 a 2026-05-01 | Política de projeto |
| Os 3 erros **não causavam falhas em runtime** no caminho feliz (caso típico `use_compensation=False`, `use_tilted=False`) | desde introdução | Risco baixo |
| **Política documentada** `SKIP=mypy,ruff` em CLAUDE.md para issues pré-existentes — bypassa hook em troca de velocidade de iteração em sprints urgentes (v2.13 v2.14 vs regressão de produção) | 2026-04-30 a 2026-05-09 | Trade-off explícito |
| Fixes exigem **type narrowing manual** + decisão de `TYPE_CHECKING` import vs runtime import — não é one-line, exige análise | sempre | Inércia técnica |

### 3.4 *"Quais os riscos desses erros não terem sido consertados?"*

**Risco 1 — F821 (forward-ref)**: TYPE_CHECKING_ONLY. Python em runtime trata
strings como referências adiadas — não há crash. Mypy/Pyright só reportam ruído.
Risco real: **baixo** (apenas estética + dificuldade de detectar novos F821).

**Risco 2 — mypy arg-type comp_pairs**: **RUNTIME CRASH POTENCIAL**. Se um
usuário chamar `simulate_multi(use_compensation=True, comp_pairs=None)`,
o código entra em `if use_compensation:` e tenta passar `None` a
`apply_compensation()`, que faz `for near, far in comp_pairs:` → `TypeError:
'NoneType' is not iterable`. Risco real: **médio-alto** (improvável mas
possível, sem mensagem informativa).

**Risco 3 — mypy arg-type tilted_configs**: idêntico ao Risco 2 para F7.

**Risco transversal — acumulação de SKIP**: cada commit que toca
`multi_forward.py` precisa `SKIP=mypy,ruff`, o que **mascara novos erros
mypy/ruff** introduzidos pelo commit. Discrepância CodeRabbit (0 findings) vs
mypy local (3 findings) corrói confiança na suite de checks.

### 3.5 *"Eles devem ser corrigidos agora?"*

**SIM, e foram.** Justificativa:

1. **Custo trivial** — 15 linhas de mudança, sem decisão arquitetural
2. **Sem risco regressional** — paridade Fortran <1e-12 + 27 testes paridade FLAT bit-exato preservados
3. **Desbloqueio operacional** — próximos commits do projeto **não precisam mais de `SKIP=mypy,ruff`** em `multi_forward.py`
4. **Runtime safety** — Risco 2/3 (RUNTIME CRASH potencial) eliminado com `ValueError` explícito

### 3.6 *"Existem outros erros pré-existentes que devem ser consertados?"*

Investigação completa do agente Explore #1 mapeou **todos** os erros pré-existentes:

```bash
ruff check geosteering_ai/   # exit 0 (apenas multi_forward.py F821, agora corrigido)
mypy geosteering_ai/         # 2 erros (apenas multi_forward.py, agora corrigidos)
```

**Outros achados não-erros**:

| Achado | Categoria | Decisão |
|:-------|:---------:|:--------|
| 22 `pytest.skip/xfail` | Tests legítimos (TF/Fortran/JAX/known_bugs) | OK |
| 1 `TODO(v1.1)` em `consensus-mcp-server` | Roadmap futuro | DEFER |
| `.backups/`, `latex-skill-main.zip`, `Resultados_Relatorio_2026/` | Diretórios untracked, em `.gitignore` | OK |
| `old_geosteering_ai/` | Referência histórica para análise KB-013 | MANTER |

**Conclusão**: Não há outros erros pré-existentes em código de produção
após esta sessão. Working tree em estado canônico.

---

## 4. Soluções Aplicadas

### 4.1 Fix #1 — F821 via `TYPE_CHECKING`

**Antes**:
```python
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
# ... código ...
def simulate_multi(...) -> Union["MultiSimulationResult", "MultiSimulationResultBatch"]:
```

**Depois**:
```python
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union
# ... código ...

# TYPE_CHECKING-only: resolve forward-ref de `MultiSimulationResultBatch` sem
# import circular. Em runtime, o import real é lazy dentro de `simulate_multi`
# (linha ~795+), pois Workers Nativos (Sprint 12.1) é opt-in via cfg.
if TYPE_CHECKING:
    from geosteering_ai.simulation._workers import MultiSimulationResultBatch
```

**Custo runtime**: ZERO (block executa apenas em mypy/IDE).

### 4.2 Fix #2 — type narrowing `comp_pairs`

**Antes**:
```python
if use_compensation:
    H_comp, phase_diff_deg, atten_db = apply_compensation(
        H_tensors_per_tr=H_tensor,
        comp_pairs=comp_pairs,  # mypy: Optional[Tuple] não compatível com Tuple
    )
```

**Depois**:
```python
if use_compensation:
    # apply_compensation espera (nTR, ntheta, nmeds, nf, 9) → layout exato.
    # Type narrowing: use_compensation=True implica comp_pairs is not None
    # (validado por SimulationConfig.__post_init__ Sprint 2.2).
    if comp_pairs is None:
        raise ValueError(
            "use_compensation=True requer comp_pairs (tuple de pares (near, far))."
        )
    H_comp, phase_diff_deg, atten_db = apply_compensation(
        H_tensors_per_tr=H_tensor,
        comp_pairs=comp_pairs,
    )
```

**Benefício**: mensagem clara de erro guia o usuário; mypy narrowing automático
pós o `raise`.

### 4.3 Fix #3 — type narrowing `tilted_configs`

Análogo ao Fix #2, para o bloco `if use_tilted:`.

---

## 5. Validação Multi-Camadas

### 5.1 Pre-commit hooks (commit `4509160` SEM SKIP)

| Hook | Status |
|:-----|:------:|
| trim trailing whitespace | ✅ Passed |
| fix end of files | ✅ Passed |
| ruff (ANTES `Failed`) | **✅ Passed** |
| ruff-format | ✅ Passed (1 file reformatted) |
| mypy (ANTES `Failed`) | **✅ Passed** |
| Anti-patterns (KB-013/018/019) | ✅ Passed |
| Paridade Fortran <1e-12 | ✅ Passed |

### 5.2 Smoke pytest

| Suite | Resultado | Tempo |
|:------|:---------:|------:|
| `tests/test_known_bugs.py` | 11/11 PASS | 1.4s |
| `tools/physics-validator-mcp/tests/` | 15/15 PASS | 0.3s |
| `tools/numba-profiler-mcp/tests/` | 17/17 PASS (incl. 2 slow) | 1.5s |
| `tests/test_simulation_v22_flat_prange.py` | **27/27 PASS** (paridade FLAT vs legacy bit-exato) | 119.7s |
| **Total** | **70/70 PASS** | ~123s |

### 5.3 CodeRabbit

```
{"type":"complete","status":"review_completed","findings":0}
```

**0 findings** mantidos após os 2 novos commits.

---

## 6. Documentação Sincronizada

### 6.1 `CLAUDE.md` (linha 16)

**Antes**: `Simulation Manager: v2.21 (2026-05-02)` ← desatualizado em 7 dias

**Depois**: `Simulation Manager: v2.22.6 (2026-05-09)` com sumário completo da
Fase 1 fundação multi-agente (I1.2 + I1.9 + I1.10 + working tree clean).

### 6.2 `docs/ROADMAP.md`

**Antes**: versão doc 1.1 (Abril 2026), última linha v2.21

**Depois**: versão doc 1.2 (Maio 2026), 4 novas linhas:
- v2.22 (FLAT prange 4D)
- v2.22.4 (default True)
- v2.22.5 (skills agent-config-override)
- v2.22.6 (Fase 1 completa)

Cross-reference para `docs/reports/fase1_fundacao_multiagente_completa_2026-05-09.md`.

---

## 7. Estatísticas Acumuladas (Branch `feat/fase1-fundacao-multiagente`)

### 7.1 Commits totais (12)

```
3e88592 docs(sync): atualizar CLAUDE.md + ROADMAP.md (v2.21 → v2.22.6)
4509160 fix(sim): corrigir 3 erros pré-existentes em multi_forward.py
75c35a2 docs(fase1): relatorio fundacao multi-agente completa (Fase 1 §22.1)
d09233a fix(mcp): aplicar findings CodeRabbit (Fase 1 final)
93f3499 test(mcp): isolate test imports per MCP via importlib
71945f5 test(mcp): add numba-profiler handler + stdio tests (I1.10)
a1dd3e1 chore(mcp): pin mcp>=1.0,<2.0 in numba-profiler requirements
8902001 feat(mcp): wire numba-profiler handlers + benchmarks (I1.10)
b2a96da test(mcp): add physics-validator handler + stdio tests (I1.9)
e938dcd feat(mcp): wire physics-validator handlers to production fns (I1.9)
a007755 feat(sim): add get_jit_cache_info to multi_forward (I1.10 prep)
4ab8dee feat(skills): add geosteering-simulator-numba (I1.2 Fase 1)
```

### 7.2 Diff vs main (atualizado)

| Arquivo | Linhas |
|:--------|------:|
| `.claude/commands/geosteering-simulator-numba.md` | +384 |
| `geosteering_ai/simulation/__init__.py` | +3 |
| `geosteering_ai/simulation/multi_forward.py` | +127 (+107 prep + 20 fix) |
| `tools/numba-profiler-mcp/*` | +817 |
| `tools/physics-validator-mcp/*` | +755 |
| `docs/reports/fase1_*.md` | +407 |
| `docs/reports/erros_preexistentes_fix_*.md` | (este arquivo) |
| `CLAUDE.md` (sync) | +1 / -1 |
| `docs/ROADMAP.md` (sync) | +5 / -2 |
| **Total** | **~+2500 / -252** |

---

## 8. Roadmap §22 — Próximos Passos (Fase 2)

Com Fase 1 100% completa **+ working tree em estado canônico (0 SKIP)**, a próxima
sessão pode iniciar **Fase 2 §22.2 Active Workflows** (Mês 2, ~54h) sem inércia
técnica:

### 8.1 Recomendação imediata

**Opção A** (recomendada — 1 hora):
```bash
git checkout main && git pull --ff-only
git merge --no-ff feat/fase1-fundacao-multiagente
git tag -a v2.22.6 -m "Fase 1 fundação multi-agente completa + fix erros pré-existentes"
git push origin main --follow-tags
```

**Opção B** — continuar nesta branch para acumular Sprint v2.23 (fastmath
dual-mode, ~8-12h). Não recomendado — branch já tem 12 commits e PR ficaria
difícil de revisar.

### 8.2 Próximos itens em ordem (§22.2)

| Item | Descrição | Esforço | Modelo |
|:-----|:----------|:--------|:-------|
| **I1.7** | `.worktreeinclude` + testes (Fase 1 deferred-blocker) | ~1h | Sonnet 4.6 |
| **I2.1** | Hook PreToolUse `validate-physics-edit.sh` (chama MCP physics-validator) | 4h | Sonnet 4.6 |
| **I2.2** | Hook PostToolUse `bench-after-perf-edit.sh` (chama MCP numba-profiler) | 4h | Sonnet 4.6 |
| **I2.3** | Slash `/sprint` orquestração — invoca physics + perf reviewers paralelo | 6h | Opus 4.7 |
| **I2.4** | Slash `/profile` — wrapper de `numba-profiler.profile_kernel` | 2h | Sonnet 4.6 |
| **I2.5** | Slash `/parity` — wrapper de `physics-validator.run_canonical_models` | 2h | Sonnet 4.6 |

### 8.3 Sprints simulator paralelos

| Sprint | Descrição | Esforço |
|:------:|:----------|:--------|
| **v2.23** | fastmath dual-mode + adaptive thread count | ~8-12h |
| **v2.24** | Hankel pré-cômputo + Kong UI opt-in (Werthmüller default) | 3-5 dias |
| **v2.25** | Alta resistividade gate (pré-sal Brazil) | 2-3 dias |
| **v2.27** | JAX vmap_real flip default (pós-validação Colab T4/A100) | 2 dias |

---

## 9. Conclusão

A sessão de "limpeza" complementar à Fase 1 corrigiu integralmente os **3 erros
pré-existentes** em `multi_forward.py` (1 ruff F821 + 2 mypy arg-type) que
vinham acumulando "noise" no pre-commit nos últimos 5 dias. Os fixes são
triviais (15 linhas), zero-risco e **eliminam a necessidade de `SKIP=mypy,ruff`
em commits futuros do projeto**.

Adicionalmente, sincronizamos `CLAUDE.md` e `docs/ROADMAP.md` (que declaravam
`v2.21` enquanto a realidade era `v2.22.6`) e validamos com:

- ✅ `ruff check`: All checks passed
- ✅ `mypy --ignore-missing-imports`: Success: no issues found
- ✅ Pre-commit hook: 9/9 hooks Passed (sem SKIP)
- ✅ pytest combinado (KB + 2 MCPs + paridade FLAT): **70/70 PASS**
- ✅ CodeRabbit `--agent --base main`: **0 findings**

A branch `feat/fase1-fundacao-multiagente` está **pronta para merge em main**
como `v2.22.6`. Estado do projeto: working tree canônico, Fase 1 §22.1 100%
completa, Fase 2 §22.2 totalmente desbloqueada.

**Aguardando novas instruções.**
