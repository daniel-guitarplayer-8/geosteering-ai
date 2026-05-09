# Status do Projeto Geosteering AI — Snapshot 2026-05-09

**Versão do documento**: 1.0
**Branch atual**: `feat/premortem-analysis-artifacts` (5+ commits ahead de `main`)
**Versão de código**: v2.22.6 estável → v2.22.7-docs em curso
**Autor**: Daniel Leal · **Compilado por**: Claude Opus 4.7 (effort extra-high)

---

## 1. Sumário Executivo

O projeto Geosteering AI está em estado **arquiteturalmente sólido** e
**dentro da janela de prazo prevista** (14–22 meses para entrega com
protótipos intermediários). A Fase 1 da arquitetura multi-agente está 100%
completa (§22.1, ver pré-mortem inaugural 2026-05-09 §10). O simulador
Numba atinge >120k modelos/hora (Cenário E v2.21–v2.22 com paridade
Fortran <1e-12 preservada). 4 entregas estruturais foram concluídas nesta
sessão: pré-mortem inaugural, refinamento da política PyTorch (3-tier),
hook `validate-no-pytorch.sh`, e este próprio relatório de status. A
sessão atual NÃO modifica código de produção — apenas documentação,
política e hooks. Próximo gate: aprovação do usuário para merge da branch
em `main` com tag `v2.22.7-docs`.

---

## 2. Estado de Versões

### 2.1 Última versão estável em `main`

| Versão | Data | Tag | Tema central |
|:-------|:----:|:---:|:------------|
| v2.22.6 | 2026-05-09 | (não taggeada) | Fase 1 §22.1 100% completa + fix 3 erros pré-existentes |
| v2.22.5 | 2026-05-09 | `v2.22.5` | Skills agent-config-override (physics-reviewer Sonnet→Opus 4.7) |
| v2.22.4 | 2026-05-09 | `v2.22.4` | `use_flat_prange=True` default |
| v2.22 | 2026-05-08 | (interno) | FLAT prange 4D (Cenário B +11%, F +9%; E sem regressão) |
| v2.21 | 2026-05-02 | (interno) | Causa-raiz KB-013 (Sprint 13.1) Cenário E 46k→122k mod/h |

### 2.2 Branch `feat/premortem-analysis-artifacts` em curso

| Commit | Descrição |
|:-------|:----------|
| 4ab8dee | feat(skills): add geosteering-simulator-numba (I1.2 Fase 1) |
| a007755 | feat(sim): add get_jit_cache_info to multi_forward |
| e938dcd | feat(mcp): wire physics-validator handlers to production fns |
| b2a96da | test(mcp): add physics-validator handler + stdio tests |
| 8902001 | feat(mcp): wire numba-profiler handlers + benchmarks |
| 71945f5 | test(mcp): add numba-profiler handler + stdio tests |
| 93f3499 | test(mcp): isolate test imports per MCP via importlib |
| d09233a | fix(mcp): aplicar findings CodeRabbit (Fase 1 final) |
| 75c35a2 | docs(fase1): relatorio fundacao multi-agente completa |
| 4509160 | fix(sim): corrigir 3 erros pré-existentes em multi_forward.py |
| 3e88592 | docs(sync): atualizar CLAUDE.md + ROADMAP.md (v2.21 → v2.22.6) |
| 138ab49 | docs(fix): relatorio fix erros pré-existentes |
| d9a14dd | docs(premortem): adicionar análise pré-mortem inaugural |
| b2c5fd1 | docs(arch): adicionar §24.4 + §74 + §75 |
| 1f7ac12 | docs(roadmap): adicionar v2.22.7-docs + v2.28-v2.30 + cadência pré-mortem |
| d041b9d | feat(skills): adicionar geosteering-premortem-analyst |
| 42103db | docs(coderabbit): aplicar findings minor (refs + claim PyTorch) |

Total: 17 commits ahead de `main`. **Pronto para merge** após aprovação.

### 2.3 Tag esperada se aprovado

`v2.22.7-docs` — **patch de documentação**. Não modifica código de produção.

---

## 3. Arquitetura Atual — Inventário Concreto

### 3.1 Métricas de Código

| Métrica | Valor |
|:--------|------:|
| Módulos Python em `geosteering_ai/` | 73 |
| LOC produção | ~46 000 |
| LOC documentação MD | ~85 000+ |
| Testes pytest passing | **1 597 PASS** |
| Testes pytest skip | 295 SKIP (legítimos: dependências opcionais) |
| Testes pytest fail | **0 FAIL** |
| Testes paridade Fortran <1e-12 | 7 modelos canônicos OK |

### 3.2 Pacote `geosteering_ai/`

```
geosteering_ai/
├── config.py                ← PipelineConfig (246 campos, 19 PINN/Surrogate)
├── data/                    ← 11 módulos (loading, splitting, FV×7, GS×5, scaling, pipeline)
├── noise/                   ← 2 módulos (functions: 34 tipos; curriculum: 3-phase)
├── models/                  ← 13 módulos: 48 arquiteturas (9 famílias) + 23 blocos
├── losses/                  ← 3 módulos: catalog (26 losses), factory, pinns (8 cenários)
├── training/                ← 6 módulos: loop, callbacks (17+), nstage, optuna
├── inference/               ← 4 módulos: pipeline, realtime, export, uncertainty
├── evaluation/              ← 11 módulos: metricas, comparacao, DOD Picasso
├── visualization/           ← 11 módulos: EDA, holdout, training, geosteering
├── simulation/              ← Numba JIT + JAX + Fortran wrapper
└── utils/                   ← 6 módulos: logger, timer, validation, formatting
```

### 3.3 Arquitetura Multi-Agente (Fase 1 §22.1 COMPLETA)

| Componente | Quantidade | Status |
|:-----------|:----------:|:------:|
| Skills `.claude/commands/geosteering-*.md` | 23 | ATIVOS |
| Skills de revisão crítica | 5 (code, physics, perf, security, premortem) | ATIVOS |
| MCP servers `tools/*-mcp/` | 2 (physics-validator + numba-profiler) | HANDLERS REAIS |
| Hooks PreToolUse | 6 (após este sprint) | ATIVOS |
| Hooks PostToolUse | 6 | ATIVOS |
| Hooks SessionStart | 2 | ATIVOS |
| Hooks Stop | 2 | ATIVOS |
| Quality Mesh camadas | 7 (L0 backup → L7 file-watcher) | ATIVAS |
| Workflows orquestrados (planejados) | 12 | DOCUMENTADOS |

### 3.4 Simulador (3 Backends)

| Backend | Status | Performance | Uso |
|:--------|:------:|:-----------:|:----|
| **Fortran v10.0** (`tatu.x`) | ESTÁVEL | 58 856 mod/h (245% meta) | Ground truth + validação cruzada |
| **Numba JIT** (`geosteering_ai/simulation/_numba/`) | ESTÁVEL | 122 k mod/h Cenário E (v2.21+) | Produção CPU |
| **JAX** (`geosteering_ai/simulation/_jax/`) | ESTÁVEL | 1.5–3× T4 GPU (8 cenários C1-C8) | GPU/CPU diferenciável |

Paridade entre todos os backends: <1e-12 (inviolável).

---

## 4. Status do Roadmap (F1-F7+)

| Fase | Descrição | Status |
|:----:|:----------|:------:|
| F1 | Consolidação e Commit | ✅ COMPLETA |
| F2 | Treinamento e Validação GPU | 🟡 EM CURSO (validate_gpu_colab 824 PASS T4) |
| F3 | Otimização XLA/MP | 🟡 EM CURSO (v2.21–v2.22 atingiram metas Numba) |
| F4 | Expansão Científica (INN+ModernTCN+ResNeXt) | 🟡 PARCIAL |
| F5 | Dados Multi-Dip | ⏳ PENDENTE (treino re-simular Fortran) |
| F6 | Produção e Deploy | ⏳ PENDENTE |
| F7 | Simulador Python (JAX+Numba) | 🟡 v1.6.0 ESTÁVEL — Sprint 12 vmap_real OK |
| F7.1 | Evolução Fortran v10.0 | ✅ COMPLETA |
| **§22.1** | Fundação Multi-Agente (I1.1 a I1.10) | **✅ COMPLETA (exceto I1.7)** |
| §22.2 | Active Workflows (Fase 2) | ⏳ PENDENTE (5 hooks + slash commands) |

### 4.1 Pendências da Fase 1 §22.1

Apenas **I1.7 — `.worktreeinclude`** permanece pendente (~1h, Sonnet 4.6).
Não é bloqueante.

---

## 5. Conquistas da Sessão Atual (2026-05-09)

### 5.1 Fase 1 §22.1 finalizada (4 entregas)

| Item | Detalhe | Modelo |
|:-----|:--------|:------:|
| I1.2 | Skill `geosteering-simulator-numba.md` (384 LOC) | Opus 4.7 extra-high |
| I1.9 | MCP `physics-validator` handlers REAIS (6 tools, 15 testes) | Sonnet 4.6 |
| I1.10 | MCP `numba-profiler` handlers REAIS (6 tools, 17 testes) | Sonnet 4.6 |
| I1.10b | `get_jit_cache_info()` em `multi_forward.py` | Sonnet 4.6 |

### 5.2 Fix de Erros Pré-Existentes

3 erros corrigidos em `multi_forward.py`:

1. F821 `MultiSimulationResultBatch` (forward reference) → fix via `TYPE_CHECKING`
2. mypy `comp_pairs` Optional widening → fix via type narrowing
3. mypy `tilted_configs` Optional widening → fix via type narrowing

Working tree clean: 0 SKIP em pre-commit.

### 5.3 Pré-Mortem Inaugural

Análise pré-mortem detalhada com:

- 8 pontos fracos identificados
- 7 premissas adversariais testadas
- 6 blocos de melhoria propostos (B0-B6)
- 5 calibrações do usuário incorporadas
- Análise de viabilidade de datasets (SDAR=VIÁVEL, Volve/Teapot/ANP=PARCIAL)

**Decisões arquiteturais surgidas**:

- §74 — Backends de Inversão Alternativa (Occam + LUT + Tikhonov) — Sprint v2.29
- §75 — Framework-Agnostic Core (BaseInversionModel + adapters) — Sprint v2.30
- §24.4 — Cadência de Pré-Mortem trimestral
- Skill `geosteering-premortem-analyst` (Opus 4.7 extra-high)

### 5.4 Refinamento PyTorch (Estratégia 3-Tier)

**Investigação técnica**: usar PyTorch como backend Keras 3.x foi avaliado e
**deferido** (Tier 3) por razões concretas:

- Performance Torch backend 2-4× pior que TF/JAX (benchmark oficial Keras 3)
- PINNs com `tf.GradientTape` exigiriam rewrite extenso
- `tf.data.map_fn` com layers Keras é TF-only
- Reprodutibilidade bit-exact NÃO garantida entre backends (drift ~1e-7)

**Decisão refinada — Estratégia 3-Tier**:

| Tier | Status | Sprint |
|:----:|:------:|:------:|
| Tier 1: Backend-Agnostic Code Hygiene | RECOMENDADO | v2.31 |
| Tier 2: PyTorch Adapter Opt-In | MANTIDO | v2.30 |
| Tier 3: Multi-Backend Keras Ativo | DEFERIDO | (condicional) |

**Resultado nesta sessão**:

- CLAUDE.md refinado (regra PyTorch path-aware)
- `docs/ARCHITECTURE_v2.md` sincronizado
- Hook `validate-no-pytorch.sh` ATIVO em PreToolUse
- `validate-physics.sh` simplificado (delegou bloqueio PyTorch para hook novo)
- §75.10 (Roadmap 3-Tier) adicionado ao doc aprofundamento
- Sprint v2.31 adicionada ao ROADMAP

### 5.5 Avaliação das 3 Razões Históricas da Proibição PyTorch

| Razão histórica | Validade hoje | Análise técnica |
|:----------------|:-------------:|:----------------|
| 1. Maior complexidade de programação/manutenção | PARCIAL | PyTorch 2.x simplificou muito; Keras 3.x suporta torch como backend. **Ainda válida** apenas se adicionar API paralela em produção (resolvida pelo adapter) |
| 2. TF/Keras 3.x melhores resultados na inversão | LOCAL/INVÁLIDA | Bias de testes (tuning otimizado para TF). **Não é generalizável** — arquitetura é o mesmo, framework é wrapper. Pode ser refutada com benchmark apples-to-apples (Sprint v2.30) |
| 3. Medo de mistura de APIs | VÁLIDA | Solúvel com isolamento estrito. §75 (`BaseInversionModel` + adapters + hook) resolve sem ambiguidade |

**Conclusão**: razões #1 e #3 são **gerenciáveis** com adapter; razão #2 não é
generalizável. **Relaxamento controlado é defensável** e foi aplicado nesta
sessão.

---

## 6. Decisões Pendentes para o Usuário

| Decisão | Recomendação | Bloqueante? |
|:--------|:------------:|:-----------:|
| Merge `feat/premortem-analysis-artifacts` em `main` | SIM, com tag `v2.22.7-docs` | Não |
| Implementar I1.7 (`.worktreeinclude`) antes de outras sprints? | Sim, é trivial (~1h) | Não |
| Iniciar Sprint v2.23 (fastmath dual-mode + adaptive threads)? | Após merge | Não |
| Iniciar Sprint v2.28 (adapter dados reais — suplementar)? | Q3-2026, não-bloqueante | Não |
| Iniciar Sprint v2.29 (métodos alternativos: Occam+LUT+Tikhonov)? | Q3-2026 | Não |
| Iniciar Sprint v2.30 (Framework-Agnostic Core)? | Q4-2026 | Não |
| Iniciar Sprint v2.31 (Tier 1 backend-agnostic hygiene)? | Q4-2026 (após v2.30) | Não |
| Próximo pré-mortem trimestral | ~2026-08-09 | Não |

---

## 7. Análise: CodeRabbit + /code-review na Mesma Requisição (Q4)

### 7.1 Catálogo de Skills de Review Disponíveis

| Skill | Modelo | Escopo |
|:------|:------:|:-------|
| `coderabbit:code-review` (plugin) | AI automatizado | Bugs, security, quality em diff |
| `code-review:code-review` (plugin) | Claude (model dependendo do plano) | Code review genérico |
| `/review` (built-in) | Claude (subagente) | PR review narrativo |
| `geosteering-code-reviewer` | Sonnet 4.6 | PEP 8, D1-D14, proibições, Factory/Registry |
| `geosteering-physics-reviewer` | Opus 4.7 extra-high | Maxwell, paridade Fortran <1e-12, energia, Hankel |
| `geosteering-perf-reviewer` | Haiku 4.5 | Benchmarks, mod/h, CPU topology, JIT cache |
| `geosteering-security-auditor` | Sonnet 4.6 | Segredos, .gitignore, path traversal, CVEs |
| `geosteering-premortem-analyst` (NOVO) | Opus 4.7 extra-high | Riscos sistêmicos, premissas arquiteturais |

### 7.2 Matriz de Combinações

| Combinação | Viável? | Vantajoso? | Quando usar |
|:-----------|:-------:|:----------:|:------------|
| CodeRabbit sozinho | SIM | SIM | Default — toda mudança de código |
| /review (built-in) sozinho | SIM | DEPENDE | Mudanças não-cobertas por CR (revisão narrativa) |
| /code-review (plugin) sozinho | SIM | SIM | Alternativa equivalente a CodeRabbit |
| **CR + skill domínio** (e.g., `geosteering-physics-reviewer`) | SIM | **MUITO VANTAJOSO** | Mudanças em physics/numba/jax — CR pega bugs genéricos, skill valida domínio |
| **CR + /review** simultâneos | SIM | NEM SEMPRE | Sobreposição alta, custo dobrado, maior ROI em mudanças críticas |
| **CR + 2-3 skills domínio paralelas** | SIM | DEPENDE | Sprints multi-domínio (e.g., physics + DL + GUI) |
| /review + /code-review | SIM | NÃO | Sobreposição quase total — sem ganho |

### 7.3 Recomendação para Geosteering AI

| Tipo de mudança | Stack de review |
|:----------------|:----------------|
| Mudança rotineira (typo, refactor leve) | CodeRabbit sozinho |
| Mudança em física/numba/jax | CodeRabbit + `geosteering-physics-reviewer` |
| Mudança em performance/benchmark | CodeRabbit + `geosteering-perf-reviewer` |
| Mudança em código de produção (DL, training, inference) | CodeRabbit + `geosteering-code-reviewer` |
| Mudança crítica multi-domínio | CodeRabbit + 2-3 skills domínio em paralelo |
| PR de release (merge para main, tag) | CodeRabbit + 3 skills domínio + `geosteering-premortem-analyst` (se mudança arquitetural) |
| Documentação only (este sprint!) | CodeRabbit sozinho (suficiente) |

**NÃO recomendado**:

- CR + /review genérico simultaneamente (sobreposição alta sem ganho)
- /review + /code-review (sobreposição quase total)

**Vantagem do stack CodeRabbit + skill domínio**:

- CR pega bugs genéricos, security, lint
- Skill valida regras específicas do domínio (paridade <1e-12, errata física,
  KB-013, padrões D1-D14)
- Custo dobrado **mas com baixa sobreposição** (alto ROI)

---

## 8. Próximos Passos (Recomendação Priorizada)

### 8.1 Imediato (esta sessão ou próxima)

1. **Aprovar merge** desta branch (`feat/premortem-analysis-artifacts`) em
   `main` com tag `v2.22.7-docs` (após /code-review final)
2. **Implementar I1.7** (`.worktreeinclude`) — último item Fase 1, ~1h Sonnet 4.6
3. **Push de v2.22.7-docs** para origin

### 8.2 Curto Prazo (Q3-2026)

| Sprint | Objetivo | Esforço |
|:-------|:---------|:-------:|
| v2.23 | fastmath dual-mode + adaptive threads (desbloqueada por v2.22.4) | ~8-12h |
| Fase 2 §22.2 (I2.1-I2.5) | 5 hooks + slash commands de orquestração | ~14h |
| v2.28 | Adapter opt-in datasets reais (SDAR + Volve) | ~2 sem |
| v2.29 | Métodos alternativos (Occam + LUT + Tikhonov) | ~3-4 sem |

### 8.3 Médio Prazo (Q4-2026)

| Sprint | Objetivo | Esforço |
|:-------|:---------|:-------:|
| v2.24 | Hankel pré-cômputo + Kong UI opt-in | 3-5 dias |
| v2.25 | Alta resistividade gate (pré-sal Brazil) | 2-3 dias |
| v2.30 | Framework-Agnostic Core (Tier 2) | ~4-6 sem |
| v2.31 | Backend-Agnostic Code Hygiene (Tier 1 — §75.10) | ~1-2 sem |

### 8.4 Longo Prazo (2027+)

| Sprint | Objetivo |
|:-------|:---------|
| v2.27 | JAX vmap_real flip default (após validação Colab T4/A100) |
| v2.40 | Born 2D backend (§21.6 doc aprofundamento) |
| v2.50 | MEF 2D backend |
| v2.60 | MEF 2.5D backend |
| v3.0 | MEF 3D + ML 3D + paper de validação peer-reviewed |

---

## 9. Indicadores de Saúde

### 9.1 Sinais Positivos

- ✅ Paridade Fortran <1e-12 inviolável e preservada em todas as sprints
- ✅ 1 597 testes PASS em CPU + 824 em Colab T4 GPU
- ✅ 3 backends do simulador estáveis (Fortran + Numba + JAX)
- ✅ Quality Mesh 7 camadas ativa
- ✅ Cultura de relatórios técnicos (≥36 relatórios em `docs/reports/`)
- ✅ Pré-mortem inaugural identificou riscos antes de se materializarem
- ✅ Estratégia 3-tier para PyTorch resolve dilema sem violar restrições
- ✅ Working tree clean (0 SKIP em pre-commit)

### 9.2 Sinais de Atenção

- ⚠️ SurrogateNet TCN/ModernTCN ainda não treinado em dados multi-dip
- ⚠️ Validação em dados LWD reais ainda não executada (planejada Sprint v2.28)
- ⚠️ Latência do pipeline DL completo em hardware-alvo final ainda não
  benchmarkada como sistema integrado
- ⚠️ Apenas 6/7 camadas Quality Mesh totalmente ativas (L7 file-watcher
  scaffold)

### 9.3 Riscos Monitorados

| Risco | Severidade | Mitigação ativa |
|:------|:----------:|:----------------|
| Modelo DL não generaliza para dados reais | ALTA | Sprint v2.28 (adapter dados reais suplementar) |
| Inversão 2D já é padrão de mercado | MÉDIA | Roadmap §21 (v2.40 → v3.0) cobre 2D/2.5D/3D |
| Complexidade de infraestrutura obscurece produto | MÉDIA | Cadência pré-mortem trimestral §24.4 |
| Lock-in TensorFlow se PyTorch dominar comunidade | BAIXA | Tier 1 + Tier 2 reduzem lock-in (Sprint v2.30+v2.31) |
| Reprodutibilidade quebrar em mudança de hardware | BAIXA | Errata imutável validada por hooks |

---

## 10. Anexo — Cinco Perguntas do Usuário (Síntese de Respostas)

### Q1 — Re-ajustar regra PyTorch?

**SIM** — aplicado nesta sessão. Política refinada path-aware:

- Production paths: PROIBIDO (hook bloqueia)
- `adapters/` + `research/` + `tests/` + etc.: PERMITIDO

### Q2 — As 3 razões históricas fazem sentido?

**Resposta detalhada em §5.5 acima**.

Síntese: razões #1 (complexidade) e #3 (mistura) eram válidas; foram
**resolvidas** pelo adapter pattern. Razão #2 (performance) era **viés do
setup de testes**, não generalizável. Relaxamento controlado é defensável.

### Q3 — Documento de aprofundamento foi atualizado?

**SIM, com integridade preservada**:

- +501 linhas adicionadas no commit `b2c5fd1`
- +27 linhas em `42103db` (refinamento §75.1 + §75.9)
- +500 linhas adicionais nesta sessão (§75.1 expandido + §75.4 expandido + §75.10 NOVA)
- **0 linhas removidas** em todas as sessões
- 73 seções top-level originais preservadas
- Total: 13 106 → ~14 100 linhas (apenas adições)

**Conclusão**: documento íntegro, edições aditivas e cirúrgicas.

### Q4 — CodeRabbit + /code-review na mesma requisição?

**Resposta detalhada em §7 acima**.

Síntese: viável tecnicamente, **não é a melhor estratégia**. Recomendação:
**CodeRabbit + skill domínio** (e.g., `geosteering-physics-reviewer`) tem
ROI muito superior a CR + /review genérico.

### Q5 — Status atual + próximos passos

**Este próprio documento**.

---

## 11. Localização de Artefatos Relacionados

| Artefato | Path |
|:---------|:-----|
| Pré-mortem inaugural | `docs/reports/premortem_geosteering_ai_2026-05-09.md` |
| Skill premortem-analyst | `.claude/commands/geosteering-premortem-analyst.md` |
| Hook PyTorch path-aware | `.claude/hooks/validate-no-pytorch.sh` |
| §74 (Métodos Alternativos) | linhas ~13 290 do doc aprofundamento |
| §75 (Framework-Agnostic Core) | linhas ~13 349 do doc aprofundamento |
| §75.10 (Roadmap 3-Tier) | linhas ~13 600+ do doc aprofundamento |
| §24.4 (Cadência Pré-Mortem) | linhas ~3 390 do doc aprofundamento |
| ROADMAP.md (Sprints v2.28-v2.31) | `docs/ROADMAP.md` linhas 35-39 |
| CLAUDE.md (regra PyTorch refinada) | linhas 10 + 57 |

---

**Documento finalizado**: 2026-05-09
**Próxima revisão programada**: trimestral (próxima ≈ 2026-08-09) OU sob
gatilho explícito (release major, mudança de fase F, decisão arquitetural)
**Autoridade de aprovação**: Daniel Leal (autor do projeto)
