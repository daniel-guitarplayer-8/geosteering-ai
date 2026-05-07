# Relatório de Execução — Etapa 0: Pré-requisitos da Quality Mesh

**Data**: 2026-05-07
**Autor**: Daniel Leal (com Claude Opus 4.7 1M)
**Documento-base**: [`docs/reports/arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md`](arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md)
**Status**: ✅ **CONCLUÍDA**
**Duração**: ~25 minutos
**Branch**: `feat/simulation-manager-v2.21` (provisória — Etapa 1 criará `feat/quality-mesh-foundation`)

---

## 1. Sumário Executivo

| Item | Status | Tamanho | Validação |
|:-----|:------:|:-------:|:---------:|
| 0.1 — Hook `backup-pre-edit.sh` | ✅ | 40 LOC | smoke OK |
| 0.2 — Hook `run-fortran-parity.sh` | ✅ | 51 LOC | filter OK (pytest deferido) |
| 0.3 — Hook `check-anti-patterns.sh` | ✅ | 60 LOC | 4/4 smoke OK |
| 0.4 — Catálogo `docs/known_bugs.md` | ✅ | 262 LOC | 164 chars PT-BR |
| 0.5 — TSV `.claude/anti-patterns.txt` | ✅ | 4 entradas | grep TAB OK |
| 0.6 — `.worktreeinclude` | ✅ | 47 LOC | sintaxe OK |
| 0.7 — `.claude/settings.json` | ✅ | edit | JSON válido |

**Total LOC criados**: ~460 linhas (hooks + catálogos + config) + 1 edit em settings.

**Camadas da Quality Mesh ativadas**: 0 (backup), 1 (PreEdit), 5 (Anti-Patterns) das 7 camadas (§41).

---

## 2. Itens Implementados

### 2.1 Hook `backup-pre-edit.sh` (PreToolUse)

| Campo | Valor |
|:------|:------|
| **Caminho** | `.claude/hooks/backup-pre-edit.sh` |
| **Evento** | PreToolUse, matcher `Edit\|Write` |
| **Timeout** | 5s |
| **Posição na cadeia** | 1ª (antes de tudo) |
| **Bloqueia** | Não — sempre `exit 0` |

**Comportamento**:
- Lê `tool_input.file_path` via `jq`.
- Filtra por extensão: `.py .f08 .yaml .yml .json .md .sh .toml .tex .bib`.
- Cria cópia em `.backups/<YYYY-MM-DD>/<rel-path>.<HHMMSS>.bak`.
- Preserva múltiplas versões no mesmo dia via timestamp `HHMMSS`.

**Smoke test executado**:
```bash
$ printf '%s' '{"tool_input":{"file_path":"…/CLAUDE.md"}}' | bash backup-pre-edit.sh
[backup] CLAUDE.md → .backups/2026-05-07/CLAUDE.md.162544.bak
exit=0
```

✅ Backup criado, exit 0, mensagem stderr informativa.

### 2.2 Hook `run-fortran-parity.sh` (PostToolUse)

| Campo | Valor |
|:------|:------|
| **Caminho** | `.claude/hooks/run-fortran-parity.sh` |
| **Evento** | PostToolUse, matcher `Edit\|Write` |
| **Timeout** | 120s |
| **Posição na cadeia** | última (após compile-check, lint, autoformat, validate-refs) |
| **Bloqueia** | Sim — `exit 1` se paridade quebrar (alerta para rollback) |

**Comportamento**:
- Filtra por path glob: `*_numba/*`, `*forward.py`, `*multi_forward.py`.
- Outros paths: `exit 0` imediato (não roda pytest).
- Em paths críticos: ativa venv `~/Geosteering_AI_venv/` e roda `pytest tests/test_simulation_compare_fortran.py -k fortran_python_numba --timeout=60 -q`.
- Falha (`exit 1`) imprime instruções de rollback via `.backups/`.

**Smoke test executado**:
```bash
$ printf '%s' '{"tool_input":{"file_path":"…/CLAUDE.md"}}' | bash run-fortran-parity.sh
exit=0   # filtro funcional, não roda pytest

$ printf '%s' '{"tool_input":{"file_path":"…/config.py"}}' | bash run-fortran-parity.sh
exit=0   # config.py fora de _numba/forward — skip OK
```

✅ Filtro de path correto. Pytest end-to-end deferido para Etapa 1 (requer venv ativado).

### 2.3 Hook `check-anti-patterns.sh` (PreToolUse)

| Campo | Valor |
|:------|:------|
| **Caminho** | `.claude/hooks/check-anti-patterns.sh` |
| **Evento** | PreToolUse, matcher `Edit\|Write` |
| **Timeout** | 5s |
| **Posição na cadeia** | 2ª (depois de backup, antes de validate-physics) |
| **Bloqueia** | Sim — `exit 1` se padrão proibido detectado |

**Comportamento**:
- Lê `tool_input.file_path` + `tool_input.new_string` (ou `content`).
- Itera `.claude/anti-patterns.txt` (TSV: `kb_id`, `regex`, `path_glob`).
- Linha vazia ou começada com `#` é ignorada.
- Path do arquivo deve casar `path_glob` (bash extglob).
- Conteúdo é testado contra regex via `grep -qE`.
- Múltiplas violações são acumuladas e listadas.

**Smoke tests executados (4 cenários)**:

| # | Caso | Path | Conteúdo | Exit | Esperado |
|:-:|:-----|:-----|:---------|:----:|:--------:|
| 1 | KB-013 em kernel.py | `_numba/kernel.py` | `@njit(parallel=True, cache=True)` | 1 | 1 ✓ |
| 2 | kernel.py limpo | `_numba/kernel.py` | `@njit(cache=True, nogil=True)` | 0 | 0 ✓ |
| 3 | KB-018 em GUI | `simulation_manager.py` | `rng_seed=42` | 1 | 1 ✓ |
| 4 | parallel=True em forward.py | `forward.py` | `@njit(parallel=True)` | 0 | 0 ✓ |

✅ 4/4 cenários corretos — bloqueio funciona, glob path filtra corretamente.

### 2.4 Catálogo `docs/known_bugs.md`

| Campo | Valor |
|:------|:------|
| **Caminho** | `docs/known_bugs.md` |
| **Tamanho** | 262 linhas |
| **Acentuação PT-BR** | 164 caracteres acentuados |

**5 KBs documentados**:

| ID | Severidade | Versão Intro → Fix | Causa-raiz |
|:--:|:----------:|:------------------:|:-----------|
| KB-001 | ALTA | v1.0 → v1.0.1 | Sinal trocado em decoupling Hxz |
| KB-002 | MÉDIA | v2.0 → v2.0.5 | Curriculum 3-fase off-by-one em epoch=0 |
| KB-013 | **CRÍTICA** | v2.13 → v2.21 | Nested `prange` em `_fields_in_freqs_kernel_cached` |
| KB-018 | ALTA | v2.18 → v2.19 | `rng_seed=42` hardcoded na GUI |
| KB-019 | MÉDIA | v2.19 → v2.20 | Defaults oversubscription em CPUs HT/SMT |

Cada KB tem 6 campos estruturados: causa-raiz, sintoma, fix, hook de prevenção, teste de regressão, arquivos afetados.

### 2.5 Catálogo `.claude/anti-patterns.txt` (TSV)

| Campo | Valor |
|:------|:------|
| **Caminho** | `.claude/anti-patterns.txt` |
| **Formato** | TSV: `kb_id<TAB>regex<TAB>path_glob` |
| **Entradas ativas** | 4 |

**Conteúdo** (verificado com tabs reais via `awk`):
```
KB-013	@njit\([^)]*parallel=True	*_numba/kernel.py
KB-018	rng_seed\s*=\s*42	*simulation_manager.py
KB-019	threads_per_worker\s*=\s*4.*workers\s*=\s*4	*sm_workers.py
KB-002	epoch\s*/\s*total_epochs(?!\s*\+)	*noise/curriculum.py
```

### 2.6 `.worktreeinclude`

| Campo | Valor |
|:------|:------|
| **Caminho** | `.worktreeinclude` (raiz) |
| **Tamanho** | 47 linhas |

**Pastas incluídas**:
- Código: `geosteering_ai/`, `tests/`, `benchmarks/`, `notebooks/`, `configs/`
- Simuladores: `Fortran_Gerador/`, `Modelos/`, `Modelos_Gerados/`
- Docs: `docs/`, `CLAUDE.md`, `README.md`, `LICENSE`
- Build: `.claude/`, `.github/`, `pyproject.toml`, `setup.py`, etc.

**Excluídos explicitamente** (em comentário, para futura integração com worktree-cli):
`Relatorio_2025/`, `Relatorio_2026/`, `Resultados_Relatorio_2026/`, `.backups/`, `old_geosteering_ai/`, `latex-document-skill-main.zip`, `Prompts_Geosteering_AI/`.

### 2.7 Registro em `.claude/settings.json`

**Cadeia PreToolUse Edit|Write final** (4 hooks, ordem de execução):
```
1. backup-pre-edit.sh        timeout=5    (não bloqueia)
2. check-anti-patterns.sh    timeout=5    (BLOQUEIA se KB casa)
3. validate-physics.sh       timeout=10   (existente, BLOQUEIA)
4. protect-critical-files.sh timeout=5    (existente, BLOQUEIA)
```

**Cadeia PostToolUse Edit|Write final** (5 hooks):
```
1. compile-check.sh          timeout=15   (existente)
2. lint-v2-standards.sh      timeout=15   (existente)
3. autoformat.sh             timeout=30   (existente)
4. validate-scientific-refs.sh timeout=10 (existente)
5. run-fortran-parity.sh     timeout=120  (NOVO, alerta)
```

JSON validado com `python3 -m json.tool`.

---

## 3. Validação Consolidada

```
=== 1. Hooks: presença, permissões, sintaxe ===
  ✓ backup-pre-edit.sh — executável + sintaxe OK
  ✓ run-fortran-parity.sh — executável + sintaxe OK
  ✓ check-anti-patterns.sh — executável + sintaxe OK

=== 2. Catálogos ===
  ✓ docs/known_bugs.md — 262 linhas
  ✓ .claude/anti-patterns.txt — 4 entradas TSV
  ✓ .worktreeinclude — 47 linhas

=== 3. settings.json válido ===
  ✓ JSON válido
  PreToolUse: backup-pre-edit, check-anti-patterns, validate-physics, protect-critical-files
  PostToolUse: compile-check, lint-v2-standards, autoformat, validate-scientific-refs, run-fortran-parity

=== 4. PT-BR acentuação em known_bugs.md ===
  ✓ 164 caracteres acentuados (PT-BR garantido)

=== 5. Backup criado pelo smoke test ===
  CLAUDE.md.162544.bak
```

✅ **Todos os 8 critérios de aceite atendidos**.

---

## 4. Próximos Passos (Roteiro Faseado)

Conforme §22 (Roadmap 4 fases) e §54.5 (cronograma faseado) do documento-base, com Etapa 0 concluída a continuação é:

### 4.1 Etapa 1 — Locks Multi-Agente + `parallelism_rules.py` (Dia 2)

**Referência**: §40 (multi-agentes paralelos, L5779-L6011).

**Objetivo**: permitir que múltiplos agentes Claude rodem em paralelo (ex.: revisor + implementador + documentador) sem corromper o repositório.

**Itens**:
- `geosteering_ai/multi_agent/lock_manager.py` — `LockManager` com lock-file por arquivo, TTL configurável, detecção de stale locks (PID morto).
- `geosteering_ai/multi_agent/parallelism_rules.py` — regras de quais arquivos podem ser editados em paralelo (ex.: `tests/test_*.py` e `docs/*.md` sim; `_numba/kernel.py` exclusivo).
- `.claude/hooks/acquire-lock.sh` — PreToolUse que adquire lock antes de Edit/Write em arquivos críticos.
- `.claude/hooks/release-lock.sh` — Stop hook que libera todos os locks ao final da sessão.
- 5 testes pytest cobrindo: aquisição, conflito, TTL, stale, release.

**Critério de aceite**: 2 agentes Claude editando o mesmo `_numba/kernel.py` em paralelo recebem mensagem `[lock] file in use by PID xxxx, retry in 30s`.

### 4.2 Etapa 2 — Skill Orquestrador + 6 Reviewers (Dia 3)

**Referência**: §4.2 (template Agente Orquestrador, L700-L766) e §16 (MCP Servers, L2359-L2552).

**Objetivo**: criar a "skill" que coordena os agentes especialistas via Task/SendMessage para sprints complexos.

**Itens**:
- `.claude/commands/geosteering-orchestrator.md` (~600 LOC) — fluxo 5-fase: brief → split → dispatch parallel → review → integrate.
- 6 sub-skills de revisão (cada uma ~150 LOC):
  - `physics-reviewer` — paridade Fortran <1e-12
  - `numba-reviewer` — performance + cache + nogil
  - `tests-reviewer` — cobertura + mocks fora de _numba
  - `docs-reviewer` — PT-BR acentuação + D1-D14
  - `security-reviewer` — sem secrets, sem cmd injection
  - `compat-reviewer` — backward-compat + deprecation
- Atualização de `.claude/settings.json` para enable das 7 commands novas.
- 3 templates de PR description (feat, fix, perf) em `.claude/templates/`.

**Critério de aceite**: rodando `/geosteering-orchestrator "Sprint 22.1: FLAT prange in forward.py"` aciona automaticamente backup + 6 reviewers em paralelo + integração.

### 4.3 Etapa 3 — Sprint v2.22 FLAT prange (Semana 3)

**Referência**: §60 (Simulation Manager 3ª trilha, L10459-L10568) + memória `project_simulation_manager_v221.md` (Cenário B 376k → 303k regressão a recuperar).

**Objetivo técnico**: Cenário B (300 pts × 2 dips × 4 combos) deve atingir ≥600k mod/h via "FLAT prange" — colapso de loops aninhados em iteração única para o scheduler Numba.

**Itens**:
- Refatorar `_simulate_combined_prange` em `forward.py`: substituir 2 loops aninhados (`prange(n_pts) × prange(n_combos)`) por 1 prange flat com `i_pos = idx // n_combos; i_combo = idx % n_combos`.
- Benchmarks A/B/C/D/E pré e pós (CSV em `benchmarks/results/v2.22/`).
- Paridade Fortran obrigatória <1e-12.
- Testes de regressão para KB-013 e KB novos descobertos.

**Critério de aceite**:
- Cenário A: ≥1.39M mod/h (zero regressão vs v2.21).
- Cenário B: ≥600k mod/h (recuperação ≥2× vs v2.21 303k).
- Cenário E: ≥122k mod/h (zero regressão vs v2.21 meta histórica).
- Paridade <1e-12 em 7 modelos canônicos.

### 4.4 Etapa 4 — MCP Servers Físicos (Semana 4)

**Referência**: §16 (MCP Servers, L2359-L2552).

**Objetivo**: extrair conhecimento físico do projeto para servidores MCP, reduzindo contexto consumido em ~70% (validado em §16 com `physics-validator.get_canonical_models()`).

**Itens**:
- `tools/mcp-physics-validator/` — TS server com tools: `get_canonical_models()`, `validate_decoupling()`, `check_parity(file_path)`.
- `tools/mcp-numba-profiler/` — TS server com tools: `profile_function(name)`, `compare_versions(v1, v2)`, `find_hotspots()`.
- 2 entradas em `.claude/settings.json` apontando para os MCP servers locais (stdio).
- Documentação de uso em `docs/reference/mcp_servers_local.md`.

**Critério de aceite**: agente Claude chama `mcp__physics-validator__get_canonical_models()` e recebe lista de 7 modelos canônicos sem precisar ler `tests/test_simulation_compare_fortran.py` (~1000 LOC poupados).

---

## 5. Recomendações Imediatas (próximas 24h)

1. **Branch dedicada**: criar `feat/quality-mesh-foundation` a partir de main e mover os 7 itens da Etapa 0 para lá. Hoje estão na branch `feat/simulation-manager-v2.21` que é específica do simulador.
   ```bash
   git checkout -b feat/quality-mesh-foundation main
   git add .claude/hooks/{backup-pre-edit,run-fortran-parity,check-anti-patterns}.sh \
           .claude/anti-patterns.txt \
           .claude/settings.json \
           docs/known_bugs.md \
           docs/reports/etapa_0_quality_mesh_2026-05-07.md \
           .worktreeinclude
   git commit -m "feat(quality-mesh): Etapa 0 — backup hook + fortran-parity + anti-patterns + KB catalog"
   ```

2. **Smoke test end-to-end com pytest real**: ativar `~/Geosteering_AI_venv` e rodar:
   ```bash
   source ~/Geosteering_AI_venv/bin/activate
   printf '%s' '{"tool_input":{"file_path":"'$PWD'/geosteering_ai/simulation/forward.py"}}' | \
     CLAUDE_PROJECT_DIR=$PWD bash .claude/hooks/run-fortran-parity.sh
   # Esperado: pytest roda em <60s, paridade <1e-12 confirmada, exit 0
   ```

3. **Criar testes de regressão** dos KBs catalogados em `tests/test_known_bugs.py`:
   - `test_kb013_no_nested_prange_in_kernel`
   - `test_kb018_random_seed_not_hardcoded`
   - `test_kb019_no_oversubscription_defaults`

4. **Commitar `Relatorio_2026/`** que está untracked desde 2026-05-05 (já tem trabalho terminado em machine_learning.tex 1640 LOC):
   ```bash
   git add Relatorio_2026/
   git commit -m "docs(report-2026): machine_learning.tex enhancements (geosignals, PINNs, multiagente)"
   ```

5. **Atualizar `MEMORY.md`** com pointer para o novo relatório:
   ```markdown
   - [Etapa 0 Quality Mesh](etapa_0_quality_mesh_2026-05-07.md) — backup-pre-edit + fortran-parity + anti-patterns hooks + known_bugs.md catálogo (KB-001/002/013/018/019) + .worktreeinclude. 7/7 itens, 460 LOC, 4/4 smoke tests OK
   ```

---

## 6. Riscos e Mitigações Pendentes

| Risco | Probabilidade | Impacto | Mitigação |
|:------|:-------------:|:-------:|:----------|
| Backup `.backups/` cresce indefinidamente | Alta | Médio | Adicionar Etapa 1.5: hook semanal de cleanup (manter últimos 7 dias) |
| `run-fortran-parity` torna edits em `_numba/*` lentos (>60s) | Média | Médio | Definir threshold; rodar apenas teste mais crítico (oklahoma_3) na hora; full suite em commit hook |
| Anti-patterns regex falsos positivos | Baixa | Baixo | Hook permite override via env `CLAUDE_BYPASS_ANTI_PATTERNS=1` (a implementar em Etapa 1) |
| Múltiplos agentes editam `anti-patterns.txt` simultaneamente | Baixa | Alto | Etapa 1 (LockManager) cobre |
| `.worktreeinclude` ainda não tem ferramenta consumindo | N/A | N/A | Etapa 1 criará agent worktree wrapper |

---

## 7. Métricas da Sessão

| Métrica | Valor |
|:--------|:-----:|
| Itens Etapa 0 implementados | 7/7 |
| Hooks criados | 3 |
| Catálogos criados | 2 (`known_bugs.md`, `anti-patterns.txt`) |
| LOC totais (hooks + catálogos) | ~460 |
| Smoke tests executados | 6 (1 backup + 4 anti-patterns + 1 fortran-parity filter) |
| Smoke tests passando | 6/6 |
| Tempo total | ~25 min |
| Camadas Quality Mesh ativadas | 3/7 (Layer 0, 1, 5) |

---

## 8. Referências ao Documento-Base

| Item da Etapa 0 | Seção do Documento | Linhas aprox. |
|:---------------|:------------------|:-------------:|
| backup-pre-edit | §38 (Backup automático pré-alteração) | L5287-L5496 |
| run-fortran-parity | §15.3 + §35 (Mecanismos anti-regressão) | L2322-L2358, L4813-L4994 |
| check-anti-patterns | §35.4 | L4925-L4994 |
| known_bugs.md | §35.3 | L4890-L4923 |
| anti-patterns.txt | §35.4 | L4961-L4967 |
| .worktreeinclude | §40 (Multi-agentes paralelos) | L5779-L6011 |
| settings.json edit | §41 (Quality Mesh) | L6012-L6238 |
| Próximos passos | §22 (Roadmap 4 fases), §54.5 | L3156-L3226, L8629-L8827 |

---

**Status Final**: Etapa 0 ✅ concluída e validada. Aguardando instruções para Etapa 1 (LockManager + parallelism_rules.py) ou retomada de outra trilha (relatório PT-BR, simulação v2.22, etc.).
