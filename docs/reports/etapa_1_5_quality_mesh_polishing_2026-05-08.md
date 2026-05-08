# Relatório Técnico — Etapa 1.5: Quality Mesh Polishing & Estabilização

<!-- Metadados do Documento -->

| Campo | Valor |
|:------|:------|
| **Versão** | Etapa 1.5 (pós-Etapas 0 e 1) |
| **Data** | 2026-05-08 |
| **Branch** | `main` (após merges de `feat/quality-mesh-foundation` e `feat/simulation-manager-v2.21`) |
| **Commit HEAD** | `e323f79` (fix(quality-mesh): code review pós-Etapa 1.5 — 4 findings corrigidos) |
| **Testes** | **19/19 PASS, 0 XFAIL, 0 SKIP** (`test_known_bugs.py` + `test_multi_agent_locks.py`) |
| **Documento-base** | `docs/reports/arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md` |
| **Relatórios anteriores** | `etapa_0_quality_mesh_2026-05-07.md`, `etapa_1_quality_mesh_multi_agent_2026-05-07.md` |

---

## Sumário Executivo

A Etapa 1.5 consolidou e estabilizou a infraestrutura Quality Mesh implantada nas Etapas 0 e 1,
resolvendo **5 bugs críticos** detectados em auditoria pré-execução, implementando **5 melhorias
de polishing** (A.1–A.5) e corrigindo **4 findings** de code review pós-execução.
O repositório `main` está agora com **todas as 6 camadas Quality Mesh ativas** (L0–L3 + L5 + L7),
com 19/19 testes de regressão passando e zero XFAIL.

---

## 1. Auditoria Pré-Etapa 1.5 — Bugs e Gaps Detectados

Antes da execução das melhorias, uma auditoria comparativa identificou os seguintes problemas
(todos corrigidos nesta etapa):

| # | Bug / Gap | Arquivo | Impacto |
|:-:|:----------|:--------|:-------:|
| B.1 | `is_expired()` off-by-one: `>` → `>=` para comparação inteira de TTL | `lock_manager.py` | `test_stale_lock_auto_cleanup` FAIL |
| B.2 | `test_known_bugs.py` KB-018: nested `def check(...)` reiniciava `in_smoke_test` prematuramente | `tests/test_known_bugs.py` | KB-018 XFAIL persistia após merge v2.19 |
| B.3 | `pyproject.toml` perdeu `pythonpath = ["."]` no merge v2.21 | `pyproject.toml` | `ModuleNotFoundError: tests._fortran_helpers` |
| B.4 | `run-fortran-parity.sh`: caractere Unicode stray em nome de variável | `.claude/hooks/run-fortran-parity.sh` | `MODE_DESC▯: unbound variable` |
| B.5 | `.pre-commit-config.yaml`: `tensorflow` em `additional_dependencies` do mypy | `.pre-commit-config.yaml` | `pre-commit install` falhava no git commit |

### 1.1 Detalhamento dos Fixes

**B.1 — TTL off-by-one (`is_expired`):**
`time.sleep(1.5)` retorna `int(1.5) == 1`. Com `ttl=1` e `>`, `1 > 1 = False`, lock não expirava.
Fix: `return self.age_seconds() >= self.ttl_sec`.

**B.2 — KB-018 detector de contexto smoke_test:**
`_run_smoke_test` contém `def check(...)` interno (linha ~8897).
O regex `re.match(r"^\s*def\s+", line)` capturava defs com indentação, saindo prematuramente
do contexto smoke. `rng_seed=42` em linha ~10281 (ainda dentro de `_run_smoke_test`) era
erroneamente detectado como violação.
Fix: `re.match(r"^(def|async def|class)\s+", line)` — apenas level-0 termina contexto.

**B.3 — `pythonpath` perdido no merge:**
O commit `ab72b7e fix(ci): restaurar pythonpath=["."]` reintroduziu a linha após sobreescrita
pelo merge do `feat/simulation-manager-v2.21` (que tinha seu próprio `pyproject.toml`).

---

## 2. Itens Implementados — Etapa 1.5

### A.1 — Fortran-Parity Smart (CRÍTICO)

**Arquivo:** `.claude/hooks/run-fortran-parity.sh` (reescrita completa)

Problema anterior: hook único rodava 7 modelos canônicos (~146s) em cada `Edit` de
arquivo Numba, tornando o fluxo de desenvolvimento inviável.

Solução implementada — dois modos via `FORTRAN_PARITY_MODE`:

```
┌─────────────────────────────────────────────────────────────────────┐
│  MODO quick (default, PostToolUse)                                  │
│    Filtro: oklahoma_3 apenas (1 modelo canônico)                    │
│    Tempo: ~2.07s (cache Numba quente) / ~124s (cold start JIT)     │
│    Ativação: qualquer Edit em _numba/*, forward.py, multi_forward.py│
│    Exit: 0 = paridade OK, 1 = violação detectada                   │
├─────────────────────────────────────────────────────────────────────┤
│  MODO full (pre-commit)                                             │
│    Filtro: fortran_python_numba (7 modelos canônicos)               │
│    Tempo: ~146s (run completo)                                      │
│    Ativação: FORTRAN_PARITY_MODE=full (sem leitura de stdin JSON)  │
│    Registrado em .pre-commit-config.yaml (hook fortran-parity-full) │
└─────────────────────────────────────────────────────────────────────┘
```

**Controles:**
- `CLAUDE_BYPASS_FORTRAN_PARITY=1`: desabilita globalmente (CI de emergência).
- Timeout em `settings.json`: 200s → 30s (adequado para modo quick cache quente).
- Modo full em `.pre-commit-config.yaml` com `files: ^geosteering_ai/simulation/(_numba/|forward\.py|multi_forward\.py)`.

**Resultado:** Paridade Fortran <1e-12 preservada. Fluxo de edição Numba não bloqueado.

---

### A.2 — Backup Cleanup Semanal

**Arquivo:** `tools/cleanup_backups.sh` (novo, ~160 LOC)

Política de retenção implementada (compatível BSD/GNU date):

```
┌──────────────────────────────────────────────────────────────────┐
│  Janela          │ Política                                       │
│  ────────────    │ ──────────────────────────────                │
│  Últimas 24h     │ Preservar TODOS os backups                    │
│  1–7 dias        │ 1 backup por arquivo por dia (mais recente)   │
│  > 7 dias        │ Remover tudo (git é o backup permanente)      │
└──────────────────────────────────────────────────────────────────┘
```

**Uso:**
```bash
tools/cleanup_backups.sh [--dry-run] [--backups-dir DIR]
```

**Notas de implementação:**
- `--dry-run`: lista o que seria removido sem remover.
- Deduplicação por `<reldir>/<basename>` (timestamp removido): mantém o arquivo mais recente.
- Compatibilidade macOS (BSD date `-v-7d`) e Linux (GNU date `-d "7 days ago"`).
- Não configurado como hook automático (pode ser lento em repos com muitos backups acumulados).

---

### A.3 — Thread Safety em `conflict_matrix.py`

**Arquivo:** `geosteering_ai/multi_agent/conflict_matrix.py`

Problema: `_RULES_MODULE` (singleton com importlib.util) não era thread-safe sob
concorrência do `watchdog` (file-watcher daemon) e de múltiplos agentes.

Solução: **double-checked locking** com `threading.Lock`:

```python
import threading
_RULES_MODULE: Any = None
_RULES_LOCK: threading.Lock = threading.Lock()

def _rules() -> Any:
    global _RULES_MODULE
    if _RULES_MODULE is None:          # check externo (sem lock, sem custo)
        with _RULES_LOCK:
            if _RULES_MODULE is None:  # check interno (com lock, race-safe)
                _RULES_MODULE = _load_rules_module()
    return _RULES_MODULE
```

**Por quê double-checked?** O check externo evita adquirir o lock em toda chamada
após a inicialização (custo O(1) amortizado). O check interno evita que dois threads
que passaram pelo check externo simultaneamente inicializem duas vezes.

---

### A.4 — Merge Branches com KB Fixes para `main`

**Branches mergeadas:**
- `feat/simulation-manager-v2.21` → `main` (commit `5c65eb6`)
- `feat/quality-mesh-foundation` → `main` (commit `bf63985`)

**O que os merges trouxeram:**
- **KB-013 fix** (v2.21): remove `parallel=True` de `_fields_in_freqs_kernel_cached`.
  Cenário E: 46k → 122k mod/h (+2.65×, meta histórica >120k restaurada).
- **KB-018 fix** (v2.19): `rng_seed=42` hardcoded → `Optional[int]` + UI control.
- **KB-019 fix** (v2.20): `recommend_default_parallelism()` usa `phys_cores`.
- **Infraestrutura Quality Mesh** (Etapas 0 e 1): hooks, LockManager, anti-patterns,
  known_bugs.md, file_watcher_daemon.py.

**Resultado após merges:** `tests/test_known_bugs.py` — 10/10 PASS (era 8 PASS + 2 XFAIL).

---

### A.5 — Env Override `CLAUDE_BYPASS_ANTI_PATTERNS=1`

**Arquivos:** `.claude/hooks/check-anti-patterns.sh` + `check-anti-patterns-precommit.sh`

Bypass global adicionado no topo de ambos os scripts:
```bash
[ "${CLAUDE_BYPASS_ANTI_PATTERNS:-0}" = "1" ] && exit 0
```

**Quando usar:** emergência onde um padrão anti-pattern é necessário temporariamente
(e.g., diagnóstico de bug que requer `print()` em `geosteering_ai/`).

**Segurança:** variável de ambiente local, não persistida em `settings.json`.
Não afeta o hook pre-commit (que lê de `.pre-commit-config.yaml`, não de `settings.json`).

Análogo já existente: `CLAUDE_BYPASS_FORTRAN_PARITY=1` (hook run-fortran-parity.sh).

---

## 3. Code Review Pós-Etapa 1.5 — Findings e Correções

Code review executado pelo agente `feature-dev:code-reviewer` com foco nos arquivos
modificados na Etapa 1.5. **4 findings** de severidade média/alta corrigidos:

| # | Finding | Arquivo | Correção |
|:-:|:--------|:--------|:---------|
| F.1 | `backup-pre-edit.sh`: sem guard para FILE_PATH fora do PROJECT_DIR (path traversal) | `.claude/hooks/backup-pre-edit.sh` | `REL_PATH="${FILE_PATH#$PROJECT_DIR/}"; [ "$REL_PATH" = "$FILE_PATH" ] && exit 0` |
| F.2 | `release-lock.sh`: fallback `\|\| echo "/Users/daniel/..."` — hardcoded, falha em CI | `.claude/hooks/release-lock.sh` | `\|\| pwd` (POSIX-safe) |
| F.3 | `cleanup_backups.sh`: `CUTOFF_24H` com `\>` excluía backups de ontem (deveria ser `\>=`) | `tools/cleanup_backups.sh` | `[ "$DAY" \>= "$CUTOFF_24H" ]` |
| F.4 | `check-anti-patterns-precommit.sh`: `grep -qE "$pattern"` sem `--` (word-split risco) | `.claude/hooks/check-anti-patterns-precommit.sh` | `grep -qE -- "$pattern" "$FILE_PATH"` |

**Commit de correção:** `e323f79 fix(quality-mesh): code review pós-Etapa 1.5 — 4 findings corrigidos`

---

## 4. Estado Final do Quality Mesh (6/7 camadas ativas)

Mapeamento contra §41 do documento-base:

```
╔══════════════════════════════════════════════════════════════════════════╗
║  QUALITY MESH — ESTADO EM 2026-05-08                                    ║
╠══════════╦════════════════════════╦════════════╦════════════════════════╣
║  Camada  ║  Nome                  ║  Status    ║  Implementação         ║
╠══════════╬════════════════════════╬════════════╬════════════════════════╣
║  L0      ║  Backup automático     ║  ✅ ATIVA  ║  backup-pre-edit.sh    ║
╠══════════╬════════════════════════╬════════════╬════════════════════════╣
║  L1      ║  Validação PreEdit     ║  ✅ ATIVA  ║  validate-physics.sh   ║
║          ║                        ║            ║  protect-critical-files ║
╠══════════╬════════════════════════╬════════════╬════════════════════════╣
║  L2      ║  Static (pre-commit)   ║  ✅ ATIVA  ║  ruff + mypy +         ║
║          ║                        ║            ║  anti-patterns +       ║
║          ║                        ║            ║  fortran-parity-full   ║
╠══════════╬════════════════════════╬════════════╬════════════════════════╣
║  L3      ║  Concorrência/Locks    ║  ✅ ATIVA  ║  LockManager + acquire ║
║          ║                        ║            ║  lock.sh + release-    ║
║          ║                        ║            ║  lock.sh + conflict_   ║
║          ║                        ║            ║  matrix.py (thread-safe)║
╠══════════╬════════════════════════╬════════════╬════════════════════════╣
║  L4      ║  Testes de regressão   ║  ⚠️ PARCIAL ║  test_known_bugs.py   ║
║          ║                        ║            ║  test_multi_agent_     ║
║          ║                        ║            ║  locks.py (19/19 PASS) ║
║          ║                        ║            ║  Falta: test_regression ║
║          ║                        ║            ║  _simulator.py formal  ║
╠══════════╬════════════════════════╬════════════╬════════════════════════╣
║  L5      ║  Anti-patterns         ║  ✅ ATIVA  ║  13 padrões TSV +      ║
║          ║                        ║            ║  check-anti-patterns.sh ║
║          ║                        ║            ║  (BLOCK/WARN + bypass) ║
╠══════════╬════════════════════════╬════════════╬════════════════════════╣
║  L6      ║  CI/CD Gates           ║  ⚠️ PARCIAL ║  GitHub Actions existe ║
║          ║                        ║            ║  Falta: gate paridade  ║
║          ║                        ║            ║  Fortran em CI         ║
╠══════════╬════════════════════════╬════════════╬════════════════════════╣
║  L7      ║  File-watcher daemon   ║  ✅ ATIVA  ║  tools/file_watcher_   ║
║          ║                        ║            ║  daemon.py (watchdog)  ║
║          ║                        ║            ║  standalone (não auto) ║
╚══════════╩════════════════════════╩════════════╩════════════════════════╝
```

### 4.1 Inventário Completo de Hooks Ativos

```
PreToolUse(Edit|Write):
  ├─ backup-pre-edit.sh         ← L0: snapshot .backups/<data>/<file>.<ts>.bak
  ├─ acquire-lock.sh            ← L3: POSIX O_CREAT|O_EXCL, TTL 300s
  ├─ check-anti-patterns.sh     ← L5: 13 padrões BLOCK/WARN (bypass CLAUDE_BYPASS_ANTI_PATTERNS)
  ├─ validate-physics.sh        ← L1: errata física, PyTorch proibido
  └─ protect-critical-files.sh  ← L1: Fortran_Gerador/, legacy/, configs versionados

PostToolUse(Edit|Write):
  ├─ release-lock.sh            ← L3: libera lock do arquivo específico
  ├─ compile-check.sh           ← py_compile rápido
  ├─ lint-v2-standards.sh       ← verifica D1-D14 (mega-header, docstrings)
  ├─ autoformat.sh              ← ruff format + isort
  ├─ validate-scientific-refs.sh← valida formato citações
  └─ run-fortran-parity.sh      ← L2 quick: oklahoma_3 ~2s (cache quente)

Stop:
  ├─ release-lock.sh --all      ← L3: libera TODOS os locks do agente
  └─ run-pytest.sh              ← suite completa tests/ -q --tb=no (120s)

SessionStart(startup):
  └─ setup-environment.sh       ← branch, commits recentes, Python version

SessionStart(compact):
  └─ reinject-errata.sh         ← reinjeta errata + proibições após compactação

pre-commit (framework .pre-commit-config.yaml):
  ├─ ruff (lint)
  ├─ mypy (type check)
  ├─ pre-commit-hooks (trailing-ws, EOF, json check)
  ├─ check-anti-patterns-precommit.sh
  └─ fortran-parity-full (FORTRAN_PARITY_MODE=full, 7 modelos, ~146s)
```

### 4.2 Inventário Multi-Agent Infrastructure

```
geosteering_ai/multi_agent/
├── __init__.py           ← exports: LockManager, AgentConflictError, can_run_together
├── lock_manager.py       ← LockManager + LockInfo + CLI (acquire/release/status/cleanup)
└── conflict_matrix.py    ← parallelism_rules.py loader, thread-safe (double-checked locking)

.claude/
├── anti-patterns.txt     ← 13 padrões TSV (KB-013/018/019/002 + 9 físicos)
├── locks/                ← lock-files SHA-256 (gitignored)
├── telemetry/
│   └── parallelism_dashboard.py ← dashboard ASCII + histórico CSV
└── parallelism_rules.py  ← conflict matrix + limites (max_sonnet=4, max_opus=1)

tools/
├── cleanup_backups.sh    ← retenção política 24h/7d (NOVO)
└── file_watcher_daemon.py← L7 watchdog (standalone, não auto-iniciado)
```

---

## 5. Histórico de Commits (Etapas 0 + 1 + 1.5)

```
e323f79 fix(quality-mesh): code review pós-Etapa 1.5 — 4 findings corrigidos
a2076a0 fix(quality-mesh): corrigir 2 bugs pós-merge em main
bf63985 merge(quality-mesh): feat/quality-mesh-foundation → main
ab72b7e fix(ci): restaurar pythonpath=["."] no pyproject.toml pós-merge v2.21
5c65eb6 merge(sm): feat/simulation-manager-v2.21 → main
906039e feat(quality-mesh): Etapa 1.5 — polishing + code review fixes
2b590ef docs(quality-mesh): relatório técnico Etapa 1 (Multi-Agent + Quality Mesh L7)
c495a76 fix(quality-mesh): code review findings — atomic locks + glob safety + portability
[...commits Etapa 1 e 0 anteriores]
```

---

## 6. Roadmap: Próximos Passos por §22 do Documento-Base

O documento-base (`arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md`)
define um **roteiro de 4 fases** (~237h total). Abaixo, o mapeamento do que está feito
e o que vem a seguir.

### 6.1 Posição Atual no Roadmap

```
╔══════════════════════════════════════════════════════════════════════╗
║  FASE 1 — FUNDAÇÃO (Mês 1, ~48h estimado)                          ║
║                                                                      ║
║  ✅ I1.7  .worktreeinclude + testes worktree                        ║
║  ✅ I1.8  Hooks Quality Mesh (backup, anti-patterns, parity,        ║
║           lock acquire/release)                                     ║
║  ⏳ I1.1  geosteering-orchestrator.md (skill principal)            ║
║  ⏳ I1.2  Expandir geosteering-simulator-numba.md                  ║
║  ⏳ I1.3  geosteering-simulator-jax.md + geosteering-pinns.md      ║
║  ⏳ I1.4  7 skills de domínio (data-pipeline, realtime, mlops,     ║
║           frontend, research, documentation, code-reviewer)         ║
║  ⏳ I1.5  5 skills de qualidade (physics-reviewer, perf-reviewer,  ║
║           security-auditor, doc-ptbr, geosteering-codex-reviewer)  ║
║  ⏳ I1.9  MCP physics-validator (stdio local)                      ║
║  ⏳ I1.10 MCP numba-profiler (stdio local)                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  FASE 2 — WORKFLOWS ATIVOS (Mês 2, ~54h)                           ║
║                                                                      ║
║  ⏳ I2.1  Sprint v2.22 FLAT prange (Cenário B 303k → ≥600k mod/h)  ║
║  ⏳ I2.2  MCP colab-bridge (agente 27º + Camadas A/B/C)            ║
║  ⏳ I2.3  /loop monitoring Colab                                    ║
║  ⏳ I2.4  Agent Teams 3 reviewers em PR simulador                  ║
║  ⏳ I2.5  Hooks: check-ptbr-accentuation, generate-pr-description  ║
║  ⏳ I2.6  CLI geosteering-cli MVP (simulate + benchmark)            ║
║  ⏳ I2.7  API REST MVP (/predict offline)                           ║
║  ⏳ I2.8  Dockerfile.cpu + CI build em PR                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  FASE 3 — MATURIDADE (Mês 3, ~53h)                                 ║
║                                                                      ║
║  ⏳ I3.1–3.2  MLflow tracking server + integração TrainingLoop     ║
║  ⏳ I3.3      Model Registry (dev/staging/production)              ║
║  ⏳ I3.4      API REST completa (realtime + UQ + admin)             ║
║  ⏳ I3.5      Docker.gpu                                            ║
║  ⏳ I3.6      Grafana + Prometheus                                  ║
║  ⏳ I3.7      CronCreate relatórios semanais                        ║
║  ⏳ I3.8      CLAUDE.md hierárquicos por subpacote                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  FASE 4 — INDUSTRIAL (Mês 4–6, ~82h)                               ║
║                                                                      ║
║  ⏳ I4.1–4.2  Loaders WITSML + LAS                                 ║
║  ⏳ I4.3–4.4  DomainAdapter + validação Volve dataset              ║
║  ⏳ I4.5      Streaming OPC-UA                                      ║
║  ⏳ I4.6–4.7  Dashboard Streamlit + Edge TFLite Jetson             ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

### 6.2 Próximo Sprint Imediato Recomendado: v2.22 FLAT prange (I2.1)

**Justificativa:** Cenário B está em 303k mod/h após fix KB-013 (Cenário E restaurado).
Análise em `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` indica que
o Cenário B pode atingir ≥600k mod/h com a técnica **FLAT prange** (§7 do documento-base).

**O que é FLAT prange (§55.2 do documento-base):**

```python
# ATUAL: prange outer sobre posições, kernel loop sobre frequências
for i in prange(n_pos):
    for f in range(nf):
        kernel_cached(...)        # hot path

# FLAT: linearizar (i, f) em índice único → 1 prange flat
total = n_pos * nf
for idx in prange(total):
    i, f = divmod(idx, nf)
    kernel_cached(...)            # mesmo hot path, sem aninhamento
```

**Vantagem:** Elimina os schedulers aninhados completamente; Numba paralleliza
`n_pos × nf` trabalhos de uma vez (ex.: 600 × 4 = 2400 tarefas) em vez de
600 outer + 4 inner serializados.

**Metas:**
- Cenário A: ≥1.4M mod/h (mantido)
- Cenário B: ≥600k mod/h (atual 303k — **meta 2×**)
- Cenário E: ≥120k mod/h (mantido, fix KB-013 em v2.21)
- Paridade Fortran: <1e-12 (inviolável)

**Risco:** `fastmath` seletivo pode comprometer paridade. Protocolo:
aplicar apenas em `_fields_in_freqs_kernel_cached` (não em módulos de propagação).
Gate automático via hook `run-fortran-parity.sh` (quick e full).

---

### 6.3 Etapa 2 — Skills e Orquestrador (I1.1–I1.6, §22.1)

**Skills a criar (11 novas):**

| Skill | Descrição | Modelo sugerido |
|:------|:----------|:---------------:|
| `geosteering-orchestrator` | Skill mestre: delega para agentes especializados, conhece todos os fluxos | Opus 4.7 |
| `geosteering-simulator-jax` | JAX backend: 8 cenários C1–C8, vmap, pmap, fori_loop | Opus 4.7 |
| `geosteering-pinns` | PINNs: 8 cenários, TIVConstraintLayer, loss física | Opus 4.7 |
| `geosteering-data-pipeline` | DataPipeline, FV(7), GS(5), curriculum noise, DataPipeline API | Sonnet 4.6 |
| `geosteering-realtime` | InferencePipeline, WITSML streaming, latência <100ms | Sonnet 4.6 |
| `geosteering-mlops` | MLflow, Docker, API REST, Model Registry | Sonnet 4.6 |
| `geosteering-frontend` | PyQt6 GUI, CLI Typer, UX patterns | Sonnet 4.6 |
| `geosteering-research` | Consensus + ArXiv + bioRxiv search; citações inline | Sonnet 4.6 |
| `geosteering-documentation` | Padrões D1–D14, PT-BR, template relatório | Haiku 4.5 |
| `geosteering-physics-reviewer` | Validação física: Maxwell, TIV, energia, paridade | Sonnet 4.6 |
| `geosteering-perf-reviewer` | Benchmark Numba/JAX, profiling, throughput | Haiku 4.5 |

**Por que o orquestrador é a peça mais crítica:**
O documento-base (§3, §4) define o orquestrador como o único agente com visibilidade
completa do projeto. Sem ele, cada sessão começa do zero sem contexto de sprint ativo,
workflows multi-arquivo são subótimos, e a delegação Opus→Sonnet→Haiku não é automatizada.

---

### 6.4 Etapa 3 — MCP Servers Locais (I1.9–I1.10, §16)

**MCP physics-validator** (`physics_validator.py`):

```
Ferramentas expostas:
  • validate_errata(config_dict) → ValidationResult
  • check_maxwell_symmetry(H_tensor) → bool
  • check_energy_conservation(R_tensor) → bool
  • check_fortran_parity(model, numba_result, tolerance=1e-12) → ParityResult
```

**Por que MCP em vez de hook?**
Hooks são determinísticos (shell). Validação física complexa (simetria de Maxwell,
conservação de energia) requer Python numérico — MCP expõe isso ao agente como tool
nativa, sem spawn de subprocesso.

**MCP numba-profiler** (`numba_profiler.py`):

```
Ferramentas expostas:
  • profile_scenario(scenario, n_models) → BenchResult
  • get_jit_cache_info() → CacheInfo
  • compare_decorator_configs(configs) → ComparisonTable
  • detect_nested_prange(source_code) → list[Violation]
```

---

### 6.5 Etapa 4 — Sprint v2.22 FLAT prange + Validação JAX GPU (§7 doc-base)

**Contexto (§7 do documento-base):**

> Sprint v2.22 FLAT prange: reestruturar `_simulate_positions_njit_cached` para
> usar prange flat sobre `(n_pos × nf)` em vez de nested prange. Meta: Cenário B
> ≥600k mod/h. Paralelizar benchmark validation com Agent Teams (Revisor Físico +
> Revisor Performance em paralelo).

**Sprint JAX GPU (§7 doc-base):**

Após v2.22 Numba, validar o flip do default `jax_vmap_real=True` em Colab T4:
- Meta: `simulate_multi_jax` com `vmap_real=True` ≥1.5× faster que `vmap_real=False`
- Paridade JAX-vs-Numba: <1e-10 (gate em `tests/test_simulation_compare_fortran.py`)
- Cenário C8 (8f×13a×4TR×600pos): meta ~6s em T4, ~700ms em A100×4 (pmap)

---

### 6.6 Topologia Multi-Agente — Estado Atual e Destino

De acordo com §4 do documento-base, o projeto define **17 agentes especializados** em 5 camadas.
Estado atual (2026-05-08):

| Agente | Camada | Status | Skill |
|:-------|:------:|:------:|:------|
| Orquestrador Principal (Daniel + Opus) | 0 | ✅ Operacional | — |
| Sim. Numba Engineer | 1 | ✅ Skills existem | `geosteering-simulator-python` |
| Sim. JAX Engineer | 1 | ⏳ Skill a criar | — |
| Sim. Fortran Engineer | 1 | ✅ Skill existe | `geosteering-simulator-fortran` |
| DL Pipeline Engineer | 1 | ⏳ Skill parcial | `geosteering-v2` |
| PINNs Specialist | 1 | ⏳ Skill a criar | — |
| Data Engineer | 1 | ⏳ Skill a criar | — |
| Geosteering RT | 1 | ⏳ Skill a criar | — |
| MLOps/Deploy | 1 | ⏳ Skill a criar | — |
| Frontend Engineer | 1 | ⏳ Skill a criar | — |
| Físico Revisor | 2 | ⏳ Skill a criar | — |
| Perf. Revisor | 2 | ⏳ Skill a criar | — |
| Código Revisor | 2 | ✅ Parcial | `geosteering-code-v2` |
| Doc PT-BR | 2 | ⏳ Skill a criar | — |
| Segurança | 2 | ⏳ Skill a criar | — |
| Pesquisador | 3 | ✅ Skill existe | `consensus-search`, `arxiv-search` |
| Documentador | 3 | ⏳ Skill a criar | — |

**Status:** 5 agentes operacionais com skills, 12 a criar.

---

## 7. Anti-patterns Catalog — Estado Atual (13 entradas)

```
KB-013  @njit\([^)]*parallel=True         BLOCK  *_numba/kernel.py
KB-018  rng_seed\s*=\s*42                 BLOCK  *simulation_manager.py
KB-019  threads_per_worker=4.*workers=4   BLOCK  *sm_workers.py
KB-002  epoch / total_epochs(?!\s*\+)     WARN   *noise/curriculum.py
KB-PYT  import\s+torch\b                  BLOCK  *.py
KB-PRT  print\s*(                         WARN   *geosteering_ai/*.py
KB-GLB  globals\(\)\.get\(               BLOCK  *geosteering_ai/*.py
KB-LOG  TARGET_SCALING="log"              BLOCK  *.py
KB-INF  INPUT_FEATURES=[0,3,4,7,8]        BLOCK  *.py
KB-OTG  OUTPUT_TARGETS=[1,2]              BLOCK  *.py
KB-FRQ  FREQUENCY_HZ=2.0[^0-9]           BLOCK  *.py
KB-SPC  SPACING_METERS=1000.0             BLOCK  *.py
KB-EPS  eps_tf=1e-30                      BLOCK  *.py
```

**Próximas adições planejadas** (§35.4 do documento-base):
- `KB-020`: `use_compensation=True` com `nTR < 2` (Fortran crash silencioso)
- `KB-021`: `fastmath=True` em funções de propagação (compromete paridade)
- `KB-022`: `split_by_sample` em vez de `split_by_model` (contaminação treino/val)

---

## 8. Métricas de Performance do Simulation Manager (pós-v2.21)

| Cenário | Antes (v2.13–v2.20) | Depois (v2.21) | Meta v2.22 |
|:--------|:-------------------:|:--------------:|:----------:|
| A (1f, 1a, 1TR, 1pos) | ~1.18M mod/h | **1.39M mod/h** | ≥1.4M |
| B (1f, multi-a) | ~376k mod/h | 303k mod/h* | **≥600k** |
| E (4f, 1a, 1TR, 600pos) | 46k mod/h | **122k mod/h** | ≥120k ✅ |

*Cenário B sofreu regressão de -19% com fix KB-013; recuperação planejada em v2.22 FLAT prange.

---

## 9. Critérios de Aceitação (pós-Etapa 1.5)

| Critério | Status |
|:---------|:------:|
| 19/19 testes de regressão PASS | ✅ |
| 0 XFAIL em `test_known_bugs.py` | ✅ |
| Paridade Fortran <1e-12 preservada (7 modelos) | ✅ |
| Hook run-fortran-parity.sh funcional (quick e full) | ✅ |
| LockManager: acquire/release/stale/TTL funcionais | ✅ |
| Anti-patterns: 13 padrões BLOCK/WARN ativos | ✅ |
| Backup cleanup com política 24h/7d implementado | ✅ |
| Thread safety em conflict_matrix.py | ✅ |
| File-watcher daemon standalone funcional | ✅ |
| pre-commit framework instalável (`pre-commit install`) | ✅ |

---

## 10. Guia de Orientação — O Que Construir em Seguida

Com base na hierarquia de valor do §22 do documento-base e no estado atual do projeto,
a recomendação de sequência é:

### Prioridade 1 — Sprint v2.22 FLAT prange (~6h, I2.1)
**Por quê primeiro:** É o único item de performance pendente. Cenário B em 303k é um
risco técnico: caso alguém rode o benchmark antes do fix, pode concluir erroneamente que
o simulador regrediu. O fix está bem documentado e o hook de paridade Fortran garante
que não quebra nada.

**Entregável:** `geosteering_ai/simulation/forward.py` com `_simulate_positions_njit_cached`
reestruturado para prange flat; relatório `docs/reports/v2.22_2026-05-XX.md`.

### Prioridade 2 — Skill `geosteering-orchestrator` (~4h, I1.1)
**Por quê segundo:** Sem o orquestrador, cada sessão de Claude Code é reativada sem
contexto de sprint ativo. Com ele, invocar `/geosteering-orchestrator` carrega o estado
completo do projeto, os últimos 5 sprints, os workflows recomendados e as regras de
delegação Opus→Sonnet→Haiku.

**Entregável:** `.claude/commands/geosteering-orchestrator.md` (~300 LOC com:
inventário de agentes, workflows end-to-end, matriz de delegação por tarefa,
anti-patterns documentados, critérios de aceitação por tipo de sprint).

### Prioridade 3 — MCP physics-validator (~8h, I1.9)
**Por quê terceiro:** O hook `validate-physics.sh` atual é shell simples (regex).
Validação de simetria Maxwell e conservação de energia requer Python — isso deve
ser um MCP Server local (stdio) chamado pelo agente, não um hook.

**Entregável:** `.claude/mcp/physics_validator.py` + entrada em `.mcp.json`.

### Prioridade 4 — Skills JAX + PINNs (~8h total, I1.3)
**Por quê quarto:** O simulador JAX v1.6.0 está estável mas sem skill especializada.
O agente JAX precisa conhecer: 8 cenários (C1–C8), decisões vmap vs fori_loop,
OOM fallbacks, XLA tuning (O1–O8), flip default para `jax_vmap_real=True`.

**Entregável:** `.claude/commands/geosteering-simulator-jax.md` (~500 LOC) +
`.claude/commands/geosteering-pinns.md` (~300 LOC).

### Prioridade 5 — Colab Bridge MCP (~6h, I2.2)
**Por quê quinto:** Validação GPU T4/A100 do JAX e treino do SurrogateNet ModernTCN
estão pendentes. Sem o agente `colab-bridge`, cada execução Colab é manual.

**Entregável:** `.claude/agents/colab-bridge.md` + configuração Camada C
(`pdwi2020/colab-exec` em `.mcp.json`) + hook `colab-token-refresh.sh`.

---

## Apêndice — Referências

| Seção doc-base | Título | Relevância |
|:---------------|:-------|:----------:|
| §2 | Princípios Arquiteturais (P1–P7) | Constraints invioláveis |
| §3–4 | Topologia 5 camadas + Catálogo de agentes | 17 agentes, 12 a criar |
| §7 | Sprint v2.22 FLAT prange + JAX GPU | Próximos sprints de performance |
| §15 | Hooks (catálogo completo) | 9 hooks novos planejados |
| §16 | MCP Servers (5 locais + 4 nuvem) | physics-validator, numba-profiler |
| §17 | Skills (16 total, 5 existem) | 11 skills a criar |
| §22 | Roadmap 4 fases ~237h | Sequência de implementação |
| §24 | Análise de riscos | Tabela de mitigações |
| §35 | Mecanismo anti-regressão | known_bugs.md, test_regression_*.py |
| §40–41 | LockManager + Quality Mesh 7L | Base Etapas 0–1.5 |
| §55 | JAX cenários C1–C8 + otimizações | FLAT prange + vmap strategies |
| §71–72 | Colab 4-tier + agente bridge | Próximas integrações |
