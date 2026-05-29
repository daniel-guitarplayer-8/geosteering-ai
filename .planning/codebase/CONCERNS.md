# Codebase Concerns

**Analysis Date:** 2026-05-24
**Project:** Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning
**Sprint atual:** v2.35 (branch `feat/sprint-o1-quick-wins`)
**LOC produção:** ~46k | **Testes:** 1665+ PASS / 295 SKIP / 0 FAIL

---

## 1. Known Bugs (Catalog KB-001..KB-019)

Catálogo oficial em `docs/known_bugs.md`. Todos resolvidos na versão indicada
e protegidos por hook `check-anti-patterns.sh` + teste de regressão nomeado `test_kbXXX_*`.

| ID | Severidade | Versão intro | Versão fix | Descrição | Arquivos afetados |
|:--:|:----------:|:------------:|:----------:|:----------|:------------------|
| KB-001 | ALTA | v1.0 | v1.0.1 | Sinal trocado no decoupling `Hxz` — fator positivo vs negativo para HMD | `geosteering_ai/simulation/_numba/dipoles.py` |
| KB-002 | MÉDIA | v2.0 | v2.0.5 | Curriculum 3-fase off-by-one em epoch=0 (`epoch/total` vs `(epoch+1)/total`) | `geosteering_ai/noise/curriculum.py` |
| KB-013 | **CRÍTICA** | v2.13 | v2.21 | `@njit(parallel=True)` aninhado dentro de `prange` outer — overhead acumulava ~14 s/run em Cenário E; regressão 122 k → 46 k mod/h (-62%) sem alterar paridade | `geosteering_ai/simulation/_numba/kernel.py` |
| KB-018 | ALTA | v2.18 | v2.19 | `rng_seed=42` hardcoded na GUI — gerador produzia sempre a mesma sequência de modelos | `geosteering_ai/simulation/tests/simulation_manager.py` (linha 8088 era o foco; corrigido via `get_rng_seed()`) |
| KB-019 | MÉDIA | v2.19 | v2.20 | Defaults de threading usavam `os.cpu_count()` (lógicos) em vez de `physical_cores` → oversubscrição 4w×4t em CPUs HT/SMT → -25 % throughput | `geosteering_ai/simulation/tests/sm_workers.py` |

**Estado atual:** 5 KBs catalogados, todos marcados como "corrigidos". Nenhum KB aberto (estado OPEN) no catálogo. Gap: última entrada é KB-019; numeração salta de KB-002 para KB-013, indicando que KBs intermediários podem não ter sido formalizados.

**Prevenção ativa** via `.claude/anti-patterns.txt` (13 regras BLOCK/WARN):
- `KB-013`: bloqueia `@njit(...parallel=True` em `*_numba/kernel.py`
- `KB-018`: bloqueia `rng_seed\s*=\s*42` em `*simulation_manager.py`
- `KB-019`: bloqueia combinação `4w×4t` hardcoded em `*sm_workers.py`
- `KB-002`: avisa `epoch / total_epochs` sem `+1` em `*noise/curriculum.py`
- + 9 regras de errata física (PyTorch, globals, print, constantes físicas)

---

## 2. Technical Debt

### 2.1 Monolito `simulation_manager.py` — Strangler Fig Pendente

**Issue:** `geosteering_ai/simulation/tests/simulation_manager.py` contém **10.759 LOC** (cresceu de 5.512 em v2.21 para 10.759 em v2.29.3 — +95% em 11 sprints). Token cost estimado ~80.000 tokens (~16% da janela de 500k Opus). Ritmo médio: +480 LOC/sprint; em 9 sprints adicionais atinge ~15k LOC.

**Localização incorreta:** o arquivo está em `simulation/tests/` (caminho de testes), mas é o módulo de GUI principal. Decisão arquitetural incorreta documentada como débito desde v2.18.

**Pirâmide invertida:** o agregador (1 arquivo, 10.759 LOC) é maior que a soma dos módulos satélite (17 arquivos `sm_*.py`, 9.337 LOC).

**Análise de risco:** `docs/reports/mvc_simulation_manager_studio_analysis_2026-05-18.md` classifica risco de não-refatoração como "médio-alto".

**Fix approach:** Estratégia Strangler Fig em 3 fases (preparação → split incremental → controllers), planejada para Sprints v2.41–v2.45. Pré-requisitos pendentes: API REST I2.7 e Dockerfile I2.8.

**Safe modification:** Para alterar `simulation_manager.py`, ler `docs/reports/mvc_simulation_manager_studio_analysis_2026-05-18.md` § 1.2 (mapa MVC implícito dos 17 módulos). Alterações devem preferir os módulos `sm_*.py` correspondentes.

### 2.2 Módulos Planejados Inexistentes (`adapters/`, `inversion/`, `research/`)

**Issue:** CLAUDE.md e o documento de arquitetura referenciam módulos críticos que não existem fisicamente no pacote:

| Módulo planejado | Caminho esperado | Status |
|:----------------|:----------------|:-------|
| Framework-Agnostic Core | `geosteering_ai/adapters/` | Não existe — diretório ausente |
| PyTorch adapter | `geosteering_ai/adapters/pytorch_adapter.py` | Não existe |
| TF adapter | `geosteering_ai/adapters/tf_adapter.py` | Não existe |
| ONNX adapter | `geosteering_ai/adapters/onnx_adapter.py` | Não existe |
| Inversão Occam | `geosteering_ai/inversion/occam.py` | Não existe |
| Look-up Table | `geosteering_ai/inversion/lut.py` | Não existe |
| Tikhonov | `geosteering_ai/inversion/tikhonov.py` | Não existe |
| Módulos de pesquisa | `geosteering_ai/research/` | Não existe |
| Real data adapter | `geosteering_ai/data/loaders/real_data_adapter.py` | Não existe |

**Impacto:** CLAUDE.md declara `from geosteering_ai.adapters import get_adapter("pytorch")` como caminho correto de acesso PyTorch — mas esse código falharia com `ImportError` em runtime.

**Fix approach:** Sprint v2.30 (planejada) cria `adapters/`. Para os métodos de inversão alternativos, Sprint pós-v3.0 per §74 do documento de aprofundamento.

### 2.3 Placeholder `make_multitask` em Produção

**Issue:** `geosteering_ai/losses/catalog.py` (linhas ~935–963) implementa `make_multitask` como MSE simples com `logger.warning` explícito: "placeholder — retorna MSE simples. Kendall et al. (2018) sera implementado em v2.1". A versão atual é v2.35 — o placeholder sobreviveu 35+ sprints. Qualquer experimento que selecione a loss `multitask` (entrada #22 no `LossFactory`) obtém silenciosamente um MSE sem incerteza de tarefa.

**Files:** `geosteering_ai/losses/catalog.py:941–963`, `geosteering_ai/losses/factory.py` (registry #22)

**Fix approach:** Implementar Kendall et al. (2018) com variáveis treináveis `log_sigma_i` por tarefa, ou remover do registry até que esteja implementado.

### 2.4 TODOs em Callbacks de Treinamento

**Issue:** `geosteering_ai/training/callbacks.py` contém dois TODOs explícitos para segmentação por janela deslizante (linhas 1484 e 1553). A implementação atual não particiona a sequência em janelas para comparar perda no centro vs bordas (`edge effects`), o que pode distorcer métricas de validação em configurações causal (realtime).

**Files:** `geosteering_ai/training/callbacks.py:1484`, `geosteering_ai/training/callbacks.py:1553`

**Fix approach:** Implementar sliding window de `config.sequence_length` para isolar edge effects; documentar no callback `EdgeEffectsCallback`.

### 2.5 Regra de Errata Imutável Não Automaticamente Validada em Todos os Paths

**Issue:** As constantes físicas críticas são validadas em `PipelineConfig.__post_init__()` — mas somente quando um `PipelineConfig` é instanciado. Módulos que operam fora do `PipelineConfig` (notebooks ad-hoc, scripts standalone, código legado em `Arquivos_Projeto_Claude/`) podem usar valores incorretos sem acionamento da validação.

**Files:** `geosteering_ai/config.py:__post_init__` (validações), `.claude/anti-patterns.txt` (cobertura parcial via hook)

**Fix approach:** O hook `reinject-errata.sh` já existe em `.claude/hooks/` — garantir que seja acionado em pre-commit para todos os paths `.py`.

### 2.6 `simulation_manager.py` em `tests/` — Path Inconsistente com Pytest

**Issue:** O módulo de GUI `simulation_manager.py` vive em `geosteering_ai/simulation/tests/`, o que cria ambiguidade: pytest pode tentar coletá-lo como módulo de teste quando invocado com `pytest geosteering_ai/` sem markers. O marcador `gui` e o filtro `xvfb-run` no CI mitigam parcialmente, mas o path é estruturalmente incorreto.

**Fix approach:** Parte da migração Strangler Fig — mover para `geosteering_ai/simulation/gui/` como primeira fase.

---

## 3. Security Concerns

### 3.1 Ausência de `.env` — Positivo, mas Sem Validação de Segredos em CI

**Current state:** Não há arquivo `.env` na raiz do projeto — verificado. Sem segredos em texto claro no repositório. `settings.local.json` em `.claude/` contém permissões de shell específicas (padrões de Bash permitidos), sem tokens ou chaves de API.

**Gap:** Não há verificação automática de secrets no CI (ex: `truffleHog`, `git-secrets`, ou GitHub Secret Scanning configurado explicitamente). O hook `validate-scientific-refs.sh` existe mas abrange referências científicas, não varredura de segredos.

**Files:** `.claude/settings.local.json`, `.claude/hooks/` (sem hook de varredura de segredos)

**Recommendations:** Ativar GitHub Secret Scanning no repositório. Adicionar `detect-secrets` ou `truffleHog` ao pre-commit.

### 3.2 MCP Servers Locais — Sem Autenticação/RBAC

**Issue:** Os dois MCP servers (`tools/physics-validator-mcp/` e `tools/numba-profiler-mcp/`) rodam como processos locais sem autenticação. O handler `physics-validator` expõe ferramentas de validação de física (6 tools) e o `numba-profiler` expõe profiling de JIT (6 tools). Em ambiente compartilhado ou Colab, qualquer processo com acesso ao socket MCP pode invocar essas ferramentas.

**Current mitigation:** Uso exclusivamente local (desenvolvimento + CI). Sem binding externo.

**Fix approach:** Adicionar token de autenticação simples (`Authorization: Bearer`) nos handlers se algum MCP server for exposto além de localhost.

### 3.3 `NUMBA_CACHE_DIR` em Tmpfs — Permissões Restritivas Aplicadas

**State:** `geosteering_ai/cli/main.py` define `NUMBA_CACHE_DIR` como `$TMPDIR/geosteering_numba_cache` com permissões `0o700`. CodeRabbit apontou este finding em v2.31 e foi corrigido. Permissões estão corretas.

### 3.4 Binário Fortran `tatu.x` sem Assinatura

**Issue:** `Fortran_Gerador/tatu.x` é um binário compilado incluído no repositório sem checksum ou assinatura verificável. Qualquer substituição do binário passaria despercebida no pipeline de CI a menos que os testes de paridade Fortran falhem.

**Current mitigation:** 10/10 testes de paridade Fortran com tolerância <1e-12 detectariam alteração funcional. Alteração de backdoor não computacional não seria detectada.

**Fix approach:** Adicionar `sha256sum tatu.x` ao CI e verificar contra valor esperado em `Fortran_Gerador/tatu.x.sha256`.

---

## 4. Performance Fragility

### 4.1 Numba JIT Cold-Start — Custo Estrutural e História de Regressões

**Problem:** O pipeline de simulação Numba tem cold-start de JIT que representa custo fixo significativo por sessão. Histórico de regressões de throughput relacionadas a este problema:

| Sprint | Problema | Throughput E | Fix |
|:-------|:---------|:------------:|:----|
| v2.13 (KB-013) | `parallel=True` aninhado em `_fields_in_freqs_kernel_cached` | 122k → 46k mod/h | v2.21: remove `parallel=True` |
| v2.16 | Threading masking (4–8× em produção GUI) | — | Fix imediato |
| v2.18 | `t0_sim` antes do warmup → throughput reportado incorretamente | 38k (falso) | Mover `t0_sim` pós-warmup |
| v2.19 | `rng_seed=42` + oversubscrição HT/SMT | 189k → 802k (A); E estagnado | Corrigir KB-018/KB-019 |
| v2.25–v2.28 | 4 tentativas de warmup sintético falharam; coverage incompleta (Warmups A/B sem paths anisotrópico+dip>0) | 55k mod/h (regressão) | v2.29: reverter para ephemeral pool |
| v2.29.1 | `NUMBA_NUM_THREADS` setado no worker (spawn bootstrap) | RuntimeError | Mover para PAI com try/finally |

**Current state (v2.35):** Arquitetura ephemeral pool (v2.29) + `geosteering-warmup` standalone (v2.32) + warmup integrado no CI (v2.34). Baseline CI: E_n200_warm = 105.423 mod/h (threshold 90% = 94.881). Cenário E cold = 69.336 mod/h.

**Files:** `geosteering_ai/simulation/tests/sm_workers.py`, `geosteering_ai/cli/main.py` (warmup entry point), `.claude/perf_baseline.json`

**Fragility:** A sensibilidade ao JIT warmup significa que qualquer alteração na ordem de imports em `_numba/` pode reintroduzir cold-start parcial. O mecanismo atual (warmup inline com dados reais do `chunk[0]`) é robusto mas silencioso — falhas de warmup não levantam exceção, apenas degradam throughput.

### 4.2 Oversubscrição em CPUs HT/SMT — Risco Recorrente

**Problem:** A combinação `n_workers × threads_per_worker > physical_cores` degrada throughput em CPUs com Hyper-Threading. Já ocorreu em v2.19 (KB-019). A proteção atual é `recommend_default_parallelism()` + warning KB-019 se `n_workers × threads > 4× physical_cores`.

**Fragility:** A proteção é um warning, não um block. Usuário pode sobrescrever na GUI ou via CLI `--workers N --threads M` sem bloqueio técnico.

**Files:** `geosteering_ai/simulation/tests/sm_workers.py` (`recommend_default_parallelism()`), `geosteering_ai/simulation/config.py` (`__post_init__` com warning KB-019)

### 4.3 Sistema de 5 Camadas Anti-Regressão — Complexidade de Manutenção

**Current state (v2.29.3):** 5 camadas de defesa:
1. Hook `.claude/hooks/check-perf-regression.sh` (WARN-only, não blocking)
2. `.claude/perf_baseline.json` (baseline Cenário E n=200 + H stress)
3. Skill `geosteering-perf-baseline` (reviewer perf-aware)
4. 37+ testes de throughput/paridade
5. Processo humano de benchmark side-by-side

**Fragility:** A camada 1 é WARN-only (`continue-on-error: true` no CI). Uma regressão de throughput pode ser mergeada sem bloqueio automático. O baseline só cobre Cenário E e H — cenários B e F (+11%/+9% após FLAT prange) não têm threshold definido.

**Fix approach:** Promover `check-perf-regression.sh` para BLOCK após definir thresholds para B e F.

### 4.4 LRU Cache Dinâmico — Risco de OOM em Multi-Frequência

**Problem:** `LRUPlotCache` em `geosteering_ai/simulation/tests/sm_plot_cache.py` usa auto-detect de RAM (10% RAM, piso 500 MB, teto 4 GB via psutil). Tensor histórico multi-freq×multi-angle pode exceder 5× esse limite: `complex128` 1000 mod × 2 TR × 4 dips × 600 pos × 4 freq × 9 comp × 16 B = **2,77 GB**. Com limite de 4 GB, cache 90% cheio antes do usuário perceber.

**Current mitigation:** Mensagem dinâmica no diálogo + QSettings persistente. QSpinBox permite ajuste manual.

**Files:** `geosteering_ai/simulation/tests/sm_plot_cache.py`, `geosteering_ai/simulation/tests/simulation_manager.py:8811`

---

## 5. Architectural Fragility

### 5.1 Strangler Fig — Risco de Crescimento do Monolito Durante Refatoração

**Problem:** A estratégia Strangler Fig para `simulation_manager.py` requer congelar o crescimento do arquivo durante a migração. Porém, sprints recentes continuaram adicionando features ao monolito (+480 LOC/sprint médio). A refatoração está planejada para Sprints v2.41–v2.45, mas os pré-requisitos (I2.7 API REST, I2.8 Dockerfile) ainda não foram entregues.

**Risk score:** Análise `docs/reports/mvc_simulation_manager_studio_analysis_2026-05-18.md` classifica como "médio-alto". Token cost de 80k tokens por sessão com Opus limita a capacidade de revisão deste arquivo.

**Safe modification:** Toda nova feature para `simulation_manager.py` deve ser implementada primeiro nos módulos `sm_*.py` correspondentes e apenas referenciada no orquestrador.

### 5.2 Dependência Single-Vendor TF/Keras

**Problem:** 100% do pipeline de produção DL usa TensorFlow 2.x / Keras 3.x. PyTorch tem >70% de market share em DL geofísica 2024-2026. Colaborações externas e modelos pré-treinados (BERT geofísico, Foundation Models geológicos) requerem conversão.

**Current mitigation:** Framework-Agnostic Core (§75 do documento de aprofundamento) planejado como `geosteering_ai/adapters/` — mas o diretório **não existe ainda**. Hook `validate-no-pytorch.sh` bloqueia imports PyTorch em production paths.

**Files:** Módulos em `geosteering_ai/{models,losses,training,inference}/` (todos TF-exclusivos)

**Fix approach:** Sprint v2.30 (planejada) deve criar `adapters/` com `BaseInversionModel` + adapters TF/PyTorch/ONNX.

### 5.3 Paridade Fortran <1e-12 — Restrição Sagrada com Ponto Único de Verdade

**Problem:** A paridade com `Fortran_Gerador/tatu.x` é o critério de correção física central (10/10 testes canônicos). Porém:
1. `tatu.x` é um binário sem código fonte público auditável neste repositório
2. Se `tatu.x` tiver simplificações na modelagem EM (ex: ignorar efeitos de poço, acoplamento galvânico), a paridade perfeita propaga esses erros com fidelidade absoluta
3. Não existe segunda fonte de verdade institucional independente além do Fortran

**Current mitigation:** Pre-mortem (§4.3) identificou este risco. Recomendação de adicionar modelos canônicos SDAR/SPWLA RtSIG como segunda fonte ainda não implementada.

**Files:** `Fortran_Gerador/tatu.x` (binário), `tests/test_simulation_compare_fortran.py` (10 modelos canônicos)

**Fix approach:** Integrar modelos canônicos SDAR (OSTI 1501648) como validação suplementar.

### 5.4 PyQt6/PySide2 Threading — QObject Cross-Thread Pitfalls

**Problem:** `simulation_manager.py` gerencia múltiplas threads Qt (`SimulationThread`, `ModelGenerationThread`, `BenchmarkThread`, `SnapshotPersistThread`, `PoolWarmupThread` removida em v2.29, `MainThreadHeartbeat`). Sinais Qt são thread-safe quando usados corretamente, mas emit direto de método não-Signal de uma thread secundária causa crash silencioso ou undefined behavior em PyQt6.

**Current mitigation:** v2.33 adicionou `pytest-qt>=4.4` com 16 testes GUI cobrindo pause/resume cooperativo e sinais. Cobertura ~25% da GUI.

**Files:** `geosteering_ai/simulation/tests/sm_workers.py` (`SimulationThread`), `geosteering_ai/simulation/tests/sm_heartbeat.py` (`MainThreadHeartbeat`), `tests/conftest_qt.py`

**Fragility:** 75% da GUI sem cobertura de teste. Novos widgets adicionados ao monolito não têm testes automáticos.

### 5.5 JAX `lax.switch` — Restrição de Shapes Idênticos

**Problem:** `geosteering_ai/simulation/_jax/dipoles_native.py:683` documenta explicitamente que `lax.switch` exige que todos os branches tenham EXATAMENTE a mesma forma de saída. Qualquer adição de novo dipole type que retorne shape diferente quebra o dispatcher silenciosamente em modo eager e com erro em JIT.

**Files:** `geosteering_ai/simulation/_jax/dipoles_native.py:683`

**Fix approach:** Documentar constraint em `simulation/config.py` e adicionar teste de shape assertion para cada branch.

---

## 6. Validation Gaps

### 6.1 Datasets Reais — Viabilidade Classificada em Pré-Mortem

| Dataset | Status | Razão |
|:--------|:------:|:------|
| SDAR (SPWLA RtSIG) | **VIÁVEL** | Sintético institucional peer-reviewed, modelos canônicos OSTI 1501648 |
| Volve | **PARCIAL** | Disponível, mas requer adaptação de formato LAS → 22-col tensor |
| Teapot Dome | **PARCIAL** | Disponível, mas cobertura LWD limitada |
| ANP | **PARCIAL** | Dados brasileiros, acesso burocrático |
| NGDS | **NÃO VIÁVEL** | Sem medições LWD compatíveis |
| Penobscot | **NÃO VIÁVEL** | Sísmica, sem resistividade LWD |

**Gap crítico:** O modelo nunca foi treinado ou validado em dados reais. O ciclo é fechado: Fortran gera → DL aprende → DL é validado contra Fortran. `real_data_adapter.py` planejado mas inexistente.

**Files:** `geosteering_ai/data/` (sem loaders de dados reais), `docs/reports/premortem_geosteering_ai_2026-05-09.md` §3.1

### 6.2 SurrogateNet — Treinamento Multi-Dip Pendente

**Problem:** `SurrogateNet` TCN (127M parâmetros) e `SurrogateNet v2 ModernTCN` (204M parâmetros) estão implementados em `geosteering_ai/models/` mas nunca foram treinados em configuração multi-dip. O treinamento requer re-simulação Fortran com dip > 0° e GPU Colab Pro+ A100 — ainda pendente.

**Files:** `geosteering_ai/models/` (SurrogateNet TCN, ModernTCN), `MEMORY.md` ("Treino SurrogateNet: re-simular Fortran multi-dip + treinar TCN/ModernTCN Colab GPU" como lacuna restante)

### 6.3 PINNs — 8 Cenários Implementados, Nenhum Validado em Poços Reais

**Problem:** 8 cenários PINN estão implementados em `geosteering_ai/losses/pinns.py` (1.975 LOC). Os testes verificam forward pass e gradientes, não convergência física ou acurácia em inversão real. Cenário petrofísica (constraints Archie + Klein) está explicitamente incompleto no `MEMORY.md`.

**Files:** `geosteering_ai/losses/pinns.py`, `geosteering_ai/losses/catalog.py` (8 cenários PINN)

### 6.4 INN — Forward+Latent Loss Incompleto

**Problem:** `InvertibleNeuralNetwork` (INN) em `geosteering_ai/inference/uncertainty.py` implementa sampling posterior e reporta UQ 10× mais rápido que Ensemble. Porém o treinamento completo (loss combinada `L_forward + lambda × L_latent`) nunca foi executado end-to-end. O módulo usa apenas `L_forward` por default.

**Files:** `geosteering_ai/inference/uncertainty.py`, `MEMORY.md` ("Treino INN completo: forward+latent loss")

### 6.5 Complexidade Combinatória Sem Benchmark de Acurácia

**Problem:** 48 arquiteturas × 26 losses × 34 tipos de ruído = ~42.432 configurações combinatórias. Nenhum benchmark de acurácia de inversão geológica foi executado sistematicamente. A "melhor" arquitetura não está definida com base em evidência. Pré-mortem §3.2 identificou isso como ponto fraco principal.

**Mitigation proposta:** Reduzir a 10–12 arquiteturas com critério documentado (§5 B2 do pré-mortem). Não implementado até 2026-05-24.

**Files:** `geosteering_ai/models/` (48 arquiteturas), `docs/reports/premortem_geosteering_ai_2026-05-09.md` §3.2

### 6.6 Quality Mesh Cobre Qualidade de Código, Não Qualidade Científica

**Problem:** As 7 camadas do Quality Mesh (hooks, pre-commit, static analysis, anti-patterns, backups, multi-agent reviews, Fortran parity) verificam consistência interna. Não verificam: acurácia de inversão calibrada, incerteza ECE honesta, comparação com métodos estabelecidos (Occam, Tikhonov). Pré-mortem §4.6 classifica isso como premissa adversarial sustentada.

**Fix approach:** Adicionar "Layer 8 — Validação Científica" ao Quality Mesh: benchmarks SDAR + métodos analíticos + calibration curve (ECE). Ainda não implementado.

---

## 7. Process & Maintenance

### 7.1 Hierarquia de Planejamento ADR-0001 — Enforçada por Hook

**State:** ADR-0001 define hierarquia de 6 documentos canônicos (`docs/INDEX.md` → `docs/ROADMAP.md §0` → `docs/sprints/CURRENT.md` → ...). Hook `.claude/hooks/check-version-references.sh` bloqueia commits que referenciam versões futuras fora do `ROADMAP.md`.

**Gap:** `docs/sprints/CURRENT.md` e `docs/sprints/v2.X.md` mencionados em ADR-0001 podem não estar sincronizados com o branch ativo. Verificar antes de criar fases de planejamento.

**Files:** `docs/INDEX.md`, `docs/ROADMAP.md`, `docs/decisions/ADR-0001.md`, `.claude/hooks/check-version-references.sh`

### 7.2 Relatórios em `docs/reports/` — Proliferação sem Índice Navegável

**State:** 30+ arquivos MD em `docs/reports/` (verificado 2026-05-24). Nomes misturam snake_case, datas, versões e prefixos inconsistentes. `docs/INDEX.md` existe mas pode estar desatualizado para relatórios recentes.

**Maintenance risk:** Claude Code lê `docs/reports/` via MEMORY.md como índice secundário. Relatórios novos adicionados sem entrada no MEMORY.md ficam "órfãos" e podem ser ignorados em sessões futuras.

**Files:** `docs/reports/` (~30 arquivos), `/Users/daniel/.claude/projects/-Users-daniel-Geosteering-AI/memory/MEMORY.md`

### 7.3 `.claude/worktrees/` — Worktrees Dangling

**State:** `find` retornou arquivos duplicados em `.claude/worktrees/suspicious-goldwasser/geosteering_ai/` (ex: `callbacks.py`, `noise/functions.py` com linha idêntica). Worktree com nome `suspicious-goldwasser` pode ser artefato de sessão anterior não limpo.

**Maintenance risk:** pytest descoberto em worktrees pode falhar se esses módulos divergirem do working tree principal.

**Files:** `/Users/daniel/Geosteering_AI/.claude/worktrees/suspicious-goldwasser/`

**Fix approach:** `git worktree list` para verificar; `git worktree remove .claude/worktrees/suspicious-goldwasser` se obsoleto.

### 7.4 `docs/decisions/` — ADRs Podem Estar Sub-Documentadas

**State:** Apenas ADR-0001 é referenciado explicitamente em CLAUDE.md. Decisões arquiteturais críticas (Framework-Agnostic Core §75, Hexagonal+DDD §50, MVC refactor) vivem em relatórios de sessão em `docs/reports/`, não em ADRs formais.

**Maintenance risk:** Decisões em relatórios são difíceis de localizar por novos agentes/colaboradores. `docs/decisions/ADR-XXXX.md` deveria ser o único SSoT para decisões arquiteturais (Regra R1 ADR-0001).

**Files:** `docs/decisions/` (apenas ADR-0001 referenciado), `docs/reports/arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md` (§74, §75 — decisões críticas sem ADR formal)

---

## 8. Active TODOs / FIXMEs no Código

Grep executado em `geosteering_ai/` em 2026-05-24. Resultado filtrado por anotações de ação real (excluindo uso legítimo de "todos" em PT-BR e referências de metadados):

| Arquivo | Linha | Tipo | Descrição |
|:--------|------:|:----:|:----------|
| `geosteering_ai/training/callbacks.py` | 1484 | TODO | Implementar segmentação por janela deslizante para comparar perda no centro vs bordas da janela (edge effects) |
| `geosteering_ai/training/callbacks.py` | 1553 | TODO | Implementar segmentação por janela deslizante — particionar sequência (duplicado do acima; mesma feature) |
| `geosteering_ai/losses/catalog.py` | 960 | PLACEHOLDER | `make_multitask`: retorna MSE simples; Kendall et al. (2018) prometido desde v2.1, nunca implementado |

**Nota:** O codebase tem densidade de TODOs formais muito baixa (2 únicos com `:` formatado + 1 placeholder explícito). A maioria das pendências são rastreadas em `docs/ROADMAP.md`, `MEMORY.md` e `docs/reports/` — não em comentários no código. Isso é intencional mas significa que `grep TODO` não captura o backlog real.

---

## Resumo Executivo de Prioridades

| Prioridade | Item | Impacto | Esforço |
|:----------:|:-----|:-------:|:-------:|
| **P0** | Monolito `simulation_manager.py` (10.759 LOC em `tests/`) — crescimento contínuo | ALTO (16% janela Opus) | ALTO (Strangler Fig, pré-req I2.7+I2.8) |
| **P0** | `geosteering_ai/adapters/` inexistente — CLAUDE.md referencia caminho que falha em runtime | ALTO | MÉDIO |
| **P1** | Validação com dados reais — ciclo fechado Fortran→DL→Fortran | CRÍTICO (científico) | ALTO |
| **P1** | `make_multitask` placeholder em produção (35+ sprints) | MÉDIO | BAIXO |
| **P1** | INN treinamento incompleto (sem L_latent) | MÉDIO | MÉDIO |
| **P2** | Segunda fonte de verdade além de `tatu.x` (SDAR/SPWLA RtSIG) | MÉDIO | MÉDIO |
| **P2** | `check-perf-regression.sh` como WARN-only (regressão pode ser mergeada) | MÉDIO | BAIXO |
| **P2** | Worktree dangling `suspicious-goldwasser` | BAIXO | BAIXO |
| **P3** | GitHub Secret Scanning + `detect-secrets` no pre-commit | BAIXO | BAIXO |
| **P3** | `sha256sum tatu.x` no CI | BAIXO | BAIXO |

---

*Concerns audit: 2026-05-24*
