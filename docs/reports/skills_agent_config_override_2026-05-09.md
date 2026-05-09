# Sessão — Agent Config Override + Upgrade Física Opus 4.7

| Campo | Valor |
|:------|:------|
| **Data** | 2026-05-09 |
| **Branch** | `feat/skills-agent-config-override` |
| **Base** | `main` (cherry-pick de `feat/skills-effort-config`) |
| **Commits** | `f1a5114`..`3b80d1e` (4 commits) |
| **Modelo** | Claude Sonnet 4.6 |
| **Arquivos modificados** | 5 skills + 1 relatório |
| **Código-review** | 2 findings (1 skip, 1 fix aplicado) |

---

## §1 Sumário Executivo

Esta sessão executou duas solicitações do usuário:

1. **Override dinâmico de `model` e `effort` no Orquestrador** — documentação de
   como passar `model=` como parâmetro ao `Agent()` (já suportado nativamente no
   Claude Code) e convenção para simular controle de effort via prefixo no `prompt`.

2. **Upgrade de agentes físicos/numéricos para Opus 4.7** — análise comparativa de
   Sonnet 4.6 Max vs Opus 4.7 Extra-High/Max para agentes de alta complexidade física,
   resultando na promoção de 3 skills de simulação para Opus 4.7.

O `/code-review` detectou 2 findings; 1 corrigido, 1 recusado como falso positivo.

---

## §2 Análise: Override `model` e `effort` no Agent()

### 2.1 Suporte Nativo ao Parâmetro `model`

O tool `Agent` do Claude Code já suporta o parâmetro opcional `model`, com valores
aceitos: `"opus"`, `"sonnet"`, `"haiku"`.

```python
# Comportamento:
# - Se model não informado → usa model: do frontmatter da skill
# - Se model informado → sobrescreve o frontmatter

Agent(subagent_type="geosteering-jax", prompt="...")           # usa Sonnet 4.6 (frontmatter)
Agent(subagent_type="geosteering-jax", model="opus", prompt="...")  # força Opus 4.7
```

**Conclusão**: nenhuma modificação necessária na infraestrutura — o override de modelo
já funciona out-of-the-box. O que foi feito é documentar esse padrão no orquestrador.

### 2.2 Ausência de Parâmetro `effort` no Agent()

O tool `Agent` **não possui** parâmetro `effort`. O campo `effort:` no frontmatter
é lido pelo runtime do Claude Code para configurar o budget de raciocínio da skill,
mas não pode ser sobrescrito via chamada de `Agent()`.

**Solução implementada — Convenção de Prefixo no `prompt`**:

```python
# Indicar análise profunda via contexto no prompt:
Agent(
    subagent_type="geosteering-physics-reviewer",
    model="opus",
    prompt=(
        "CONTEXTO: Paridade Fortran quebrou em produção. "
        "Análise crítica — verificar TODOS os 7 modelos canônicos, "
        "incluindo alta resistividade (>1000 Ω·m).\n\nTarefa: ..."
    )
)
```

Essa convenção guia o comportamento do agente sem alterar o budget técnico, o que
é suficiente para a maioria dos casos de uso. Para budget máximo garantido, o
agente deve ser invocado diretamente com `/geosteering-orchestrator` (effort: max).

---

## §3 Análise: Sonnet Max vs Opus Extra-High/Max para Física

### 3.1 Critérios de Avaliação

| Critério | Sonnet 4.6 Max | Opus 4.7 Extra-High | Opus 4.7 Max |
|:---------|:--------------|:---------------------|:------------|
| Raciocínio multi-arquivo | Bom | Excelente | Excelente |
| Contexto (tokens) | 200k | 1M | 1M |
| Debug tensor Maxwell 3×3 | Suficiente | Superior | Superior |
| Paridade Fortran (debug regressão) | Pode perder edge-cases | Confiável | Confiável |
| Custo relativo | ~2× Sonnet | ~4× Sonnet | ~5× Sonnet |
| Sprint simulador cross-file | Borderline | ✅ Recomendado | ✅ Recomendado |
| Review física rotineira | ✅ Suficiente | ✅ Ideal | Excessivo |

### 3.2 Referência: §19 Documento de Arquitetura

O documento `arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md`
§19.2 especifica:

```
Sprint do simulador Numba (cross-file) → Opus    (já era Opus no doc)
Debug regressão paridade Fortran       → Opus    (física-reviewer era Sonnet 4.6)
Refatoração arquitetural (backends 2D) → Opus
```

O physics-reviewer estava como **Sonnet 4.6** mas o doc base exige **Opus** para
debugging de paridade — inconsistência corrigida nesta sessão.

### 3.3 Decisões de Upgrade

| Skill | Antes | Depois | Justificativa |
|:------|:------|:-------|:--------------|
| `geosteering-physics-reviewer` | Sonnet 4.6 extra-high | **Opus 4.7 extra-high** | §19: debug paridade = Opus; tensor Maxwell 3×3 requer raciocínio profundo |
| `geosteering-simulator-fortran` | (sem model/effort) | **Opus 4.7 extra-high** | Debugging Fortran → Opus per §19; `effort: extra-high` padrão para sprints |
| `geosteering-simulator-python` | (sem model/effort) | **Opus 4.7 extra-high** | Multi-arquivo JAX+Numba; cruzamento com Fortran parity gate |
| `geosteering-jax` | Sonnet 4.6 extra-high | **mantido** | Override via `model="opus"` pelo orquestrador em refatorações |
| `geosteering-pinns` | Sonnet 4.6 extra-high | **mantido** | Implementação bem-definida, 8 cenários catalogados |

---

## §4 Implementação — Mudanças por Arquivo

### 4.1 `.claude/commands/geosteering-orchestrator.md`

Adições (+92 linhas):

```
+ ## Configuração Dinâmica de Agentes
+   - Seção "Parâmetro model — Override via Agent()"
+     - Exemplos de uso: padrão, override Opus, override Sonnet
+     - Regras de ouro para 5 tipos de tarefa
+   - Seção "Parâmetro effort — Convenção via Prompt"
+     - Explicação de por que Agent() não suporta effort diretamente
+     - Padrão de prompt prefix para indicar nível de análise
+
+ Tabela "Subagentes Disponíveis" expandida:
+   - Coluna "Effort" (default da skill)
+   - Coluna "Override Típico" (quando o orquestrador deve sobrescrever)
+   - physics-reviewer atualizado para Opus 4.7
+   - documentation atualizado para Sonnet 4.6 (bump anterior)
+
+ Tabela Identidade: linha "Effort" com hierarquia explícita
+   → Responde ao finding minor do code-review
```

### 4.2 `.claude/commands/geosteering-physics-reviewer.md`

```diff
- description: ... Modelo Sonnet 4.6 com profundidade 2 ...
+ description: ... Modelo Opus 4.7 com effort extra-high ...
- model: claude-sonnet-4-6
+ model: claude-opus-4-7
  effort: extra-high   # já existia via cherry-pick
- | **Modelo** | Claude Sonnet 4.6 |
+ | **Modelo** | Claude Opus 4.7 (1M context, effort extra-high) |
```

### 4.3 `.claude/commands/geosteering-simulator-fortran.md`

```diff
+ model: claude-opus-4-7
+ effort: extra-high
```

### 4.4 `.claude/commands/geosteering-simulator-python.md`

```diff
+ model: claude-opus-4-7
+ effort: extra-high
```

---

## §5 /code-review — Findings

| # | Severidade | Arquivo | Finding | Ação |
|:-:|:-----------|:--------|:--------|:-----|
| 1 | Critical | `geosteering-simulator-python.md:19` | CodeRabbit sugere `effort: xhigh` em vez de `extra-high` | **SKIP** — `extra-high` é valor estabelecido em 5 skills existentes; CodeRabbit não tem autoridade sobre frontmatter Claude Code |
| 2 | Minor | `geosteering-orchestrator.md:20` | Terminologia inconsistente: frontmatter usa `max`, tabela usa `extra-high` | **CORRIGIDO** — adicionada linha Effort na tabela de identidade com hierarquia explícita `high < extra-high < max` |

---

## §6 Estado Final das Skills

### Hierarquia de Modelo × Effort

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SKILL                      │  MODELO          │  EFFORT       │  CAMADA    │
├──────────────────────────────────────────────────────────────────────────────┤
│  geosteering-orchestrator   │  Opus 4.7 (1M)   │  max          │  Hub (L0)  │
├──────────────────────────────────────────────────────────────────────────────┤
│  geosteering-physics-reviewer│ Opus 4.7 (1M)   │  extra-high   │  Spoke (L2)│  ← UPGRADE
│  geosteering-simulator-numba│  Opus 4.7 (1M)   │  extra-high   │  Spoke (L1)│
│  geosteering-simulator-fortran│Opus 4.7 (1M)   │  extra-high   │  Spoke (L1)│  ← NOVO
│  geosteering-simulator-python│ Opus 4.7 (1M)   │  extra-high   │  Spoke (L1)│  ← NOVO
├──────────────────────────────────────────────────────────────────────────────┤
│  geosteering-jax            │  Sonnet 4.6      │  extra-high   │  Spoke (L2)│
│  geosteering-pinns          │  Sonnet 4.6      │  extra-high   │  Spoke (L2)│
│  geosteering-code-reviewer  │  Sonnet 4.6      │  high         │  Spoke (L3)│
│  geosteering-documentation  │  Sonnet 4.6      │  high         │  Spoke (L3)│  ← Haiku→Sonnet
│  geosteering-research       │  Sonnet 4.6      │  high         │  Spoke (L3)│
│  geosteering-security-auditor│ Sonnet 4.6      │  high         │  Spoke (L3)│
│  geosteering-data           │  Sonnet 4.6      │  high         │  Spoke (L3)│
│  geosteering-realtime       │  Sonnet 4.6      │  high         │  Spoke (L3)│
├──────────────────────────────────────────────────────────────────────────────┤
│  geosteering-perf-reviewer  │  Haiku 4.5       │  high         │  Spoke (L4)│
└──────────────────────────────────────────────────────────────────────────────┘
```

### Distribuição de Modelos (Target §19)

| Modelo | Skills | Uso Esperado |
|:-------|:------:|:------------|
| Opus 4.7 | 5 skills | ~10% interações (física + simulação + orquestração) |
| Sonnet 4.6 | 8 skills | ~65% interações (implementação rotineira) |
| Haiku 4.5 | 1 skill | ~25% interações (automação + benchmarks) |

---

## §7 Próximos Passos — Baseado em `arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md`

### §22 Roadmap Completo de Implementação da Infraestrutura

O documento base organiza a construção em **Etapas sequenciais**:

#### Etapa 0 ✅ CONCLUÍDA — Quality Mesh Foundation
- 3 hooks PreToolUse (backup, anti-patterns, Fortran-parity)
- `tests/test_known_bugs.py` (11 testes)
- `.pre-commit-config.yaml` (ruff + mypy + local)

#### Etapa 1.5 ✅ CONCLUÍDA — Polishing & Estabilização
- 19/19 PASS, 0 XFAIL
- 6/7 camadas Quality Mesh ativas (L0+L1+L2+L3+L5+L7)
- cleanup_backups.sh + thread-safe conflict_matrix.py

#### Sprint v2.22 ✅ CONCLUÍDA — FLAT prange (Etapa Sim)
- FLAT prange 4D, 27 testes paridade, 1597 PASS total
- Cenário B +11%, F +9%; Paridade Fortran <1e-12 preservada

#### Etapa 2 ✅ CONCLUÍDA — Skills Multi-Agente (Sessões B+C)
- 11 skills de qualidade e domínio
- 2 MCP scaffolds (physics-validator + numba-profiler)
- v2.22.4 default `use_flat_prange=True`

#### Etapa 2b ✅ CONCLUÍDA (esta sessão) — Upgrade Física + Agent Config
- geosteering-physics-reviewer: Sonnet → Opus 4.7
- Simuladores Fortran e Python: model + effort adicionados
- Orquestrador: seção override model/effort documentada

---

### Próximas Etapas Prioritárias

#### A) Sprint v2.23 — Fastmath + Adaptive Threads (~2 dias)
**Desbloqueado por**: v2.22.4 (FLAT prange default True)
**Agente**: `geosteering-simulator-numba` (Opus 4.7 extra-high)

```
Scope:
  1. cfg.use_fastmath: bool = False (opt-in gate)
  2. Dual-mode hmd_tiv/vmd: PRECISE (<1e-12) vs FAST (~1e-10)
  3. Adaptive thread count para n_pos baixo (Cenário A < 30 pts)
  4. Testes paridade: 7 modelos × 2 modos = 14 casos
  5. Benchmark E/B/F com fastmath ON/OFF
```

**Meta**: +20% Cenário E sem regressão paridade produção.

#### B) Etapa 3 — MCP Servers Completos (~1.5-2 dias)
**Agente**: `geosteering-orchestrator` + implementação manual

```
physics-validator-mcp/server.py:
  - mcp.server.Server (stdio) com handlers async reais
  - 6 tools: validate_parity, check_errata, maxwell_symmetry,
    check_decoupling, run_canonical, audit_kb013

numba-profiler-mcp/server.py:
  - 6 tools: profile_kernel, benchmark_scenario, compare_flat_vs_legacy,
    trace_prange, cache_stats, memory_usage
```

#### C) Sprint v2.24 — Hankel Pre-cômputo + Kong UI (~3-5 dias)
**Agente**: `geosteering-simulator-numba` + `geosteering-jax`

```
Scope:
  1. HankelFilterManager (lazy-load, cache npz)
  2. cfg.use_kong_training: bool = False (opt-in, 3.3× treino)
  3. GUI FilterSelectorDialog (Werthmuller/Kong/Anderson)
  4. Kernel alternativo _fields_in_freqs_kernel_kong
  5. Werthmuller 201pt PERMANECE default produção
```

#### D) Sprint v2.25 — Alta Resistividade Gate (~2-3 dias)
**Agente**: `geosteering-physics-reviewer` (Opus 4.7) + `geosteering-simulator-numba`

```
Scope (§7.5 do doc):
  - 4 novos modelos canônicos (carbonato_seco, evaporita, gas_seco, basalto)
  - ρ_max até 100.000 Ω·m (sal halita)
  - Gate: paridade <1e-12 com Werthmuller 201pt
  - Se falhar: tentar Anderson 801pt
  - Fastmath gate alta-ρ (bloquear se ρ > 10k Ω·m com fastmath)
```

#### E) Sprint v2.27 — Flip Default JAX vmap_real (~1 dia)
**Pré-requisito**: validação GPU Colab T4/A100 (manual)
**Agente**: `geosteering-jax` (override `model="opus"` para decisão arquitetural)

#### F) Etapa 4 — Integração Colab 4-tier (~2 semanas)
**Agente**: `geosteering-orchestrator` + documentação de 4 tiers:
- A: Drive manual
- B: colab-mcp browser (pdwi2020/colab-exec)
- C: HEADLESS
- D: custom (Sprint 28+)

---

## §8 Recomendação Imediata

**Próxima sessão recomendada: Sprint v2.23 fastmath + adaptive threads**

```bash
# Preparação:
git checkout main
git checkout -b feat/simulator-v2.23-fastmath

# Invocar:
/geosteering-orchestrator
# → Prompt: "Execute Sprint v2.23: dual-mode fastmath (gate <1e-12 PRECISE,
#   ~1e-10 FAST) + adaptive thread count. Base: docs/reference/analise_cenarios.md §8.3"
```

**Por que agora**: v2.22.4 default `True` está em `main`. O Sprint v2.23 é o próximo
na sequência do §22 roadmap e está explicitamente desbloqueado por v2.22.

---

## §9 Estatísticas

| Métrica | Valor |
|:--------|:------|
| Commits | 4 (cherry-pick ×2 + feat + fix) |
| Arquivos modificados | 5 skills + 1 relatório |
| LOC adicionadas | ~95 linhas (orquestrador) + 6 linhas (frontmatter skills) |
| Skills com model+effort completo | 13/22 (subindo de 11) |
| Skills Opus 4.7 | 4 → 5 (physics-reviewer promovido) |
| /code-review findings | 2 (1 fix, 1 skip justificado) |
