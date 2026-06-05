> **Blueprint de Arquitetura do Geosteering AI — Cross-Cutting (Segurança · Observabilidade · Testes · Config).** Índice: [README.md](README.md) · Constituição SDD: [../../specs/CONSTITUTION.md](../../specs/CONSTITUTION.md) · Roadmap: [../../specs/ROADMAP.md](../../specs/ROADMAP.md). Gerado 2026-06-05 (workflow multi-agente + revisão crítica).

## 0. Mapa transversal — onde cada preocupação cruza os 4 produtos

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  PREOCUPAÇÃO ↓  / PRODUTO →   Lib+API   CLI    Studio   Sim. Manager           │
├──────────────────────────────────────────────────────────────────────────────┤
│  Segurança (auth)              ●●●      ○      ●●        ○                      │
│  Segurança (OT rig)            ●●●      ○      ●●●       ○   (só Fase 2 Studio) │
│  Segredos (gitleaks/.env)      ●●●      ●●     ●●        ●                      │
│  Observabilidade (OTel/Prom)   ●●●      ●      ○         ○                      │
│  Observabilidade (Sentry)      ○        ○      ●●●       ●●●                    │
│  Logging estruturado D9        ●●●      ●●●    ●●●       ●●●  (já existe)        │
│  Testes (pirâmide)             ●●●      ●●     ●●●       ●●●                     │
│  Config frozen + errata        ●●●      ●●●    ●●●       ●●●  (já existe)        │
│  Estado (.gsproj/.session)     ○        ○      ●●●       ●●                      │
│  Resiliência (jax→numba)       ●●       ●●     ●●●       ●●●  (parcial)          │
│  Docs D1-D14 / ADR             ●●●      ●●●    ●●●       ●●●  (já existe)        │
└──────────────────────────────────────────────────────────────────────────────┘
  ● = crítico  ○ = não-aplicável/baixo   Tudo converge em geosteering_ai/utils/ e config
```

Princípio-mestre: **um único módulo transversal por preocupação**, importado por todos os produtos. Zero duplicação. A fundação compartilhada (`geosteering_ai/utils/` já existe; `geosteering_ai/platform/` a criar) é a casa dos cross-cutting.

---

## 1. SEGURANÇA

### 1.1 OT no rig — Arquitetura Purdue + Zero Trust (Studio Fase 2)

A inferência realtime durante perfuração toca a rede OT do rig. Modelo de referência: **IEC 62443 + Purdue**, com o Studio operando **edge-first** (nunca no caminho de controle do BHA — apenas decision-support).

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  PURDUE / IEC 62443 — POSIÇÃO DO GEOSTEERING AI                          │
├─────────────────────────────────────────────────────────────────────────┤
│  L4/L5  Corporativo (nuvem)   ── Registry MLflow, dashboards Grafana     │
│    ▲  jump-server + MFA (sem VPN always-on)                              │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ DMZ industrial (data diode/ETP gateway)──  │
│    ▲  ETP/WITSML JWT+TLS 1.3, mTLS opcional                             │
│  L3   Operações do rig        ── Studio EDGE (decision-support, RO)      │
│    ▲  só LÊ telemetria; NUNCA escreve no controle                       │
│  L2   Supervisório (SCADA)    ── MWD/LWD aggregator  (fora do escopo)    │
│  L1   Controle (PLC/BHA)      ── PROIBIDO acesso do Studio              │
│  L0   Sensores fundo de poço  ──                                        │
└─────────────────────────────────────────────────────────────────────────┘
  Regra-dura OT-1: o Studio é SEMPRE read-only em relação a L1/L2.
                   "Steering command" = sugestão para humano, nunca atuação.
```

| # | Controle | Decisão concreta | Status |
|:-:|:---------|:-----------------|:------:|
| OT-1 | Isolamento de controle | Studio nunca importa cliente de escrita L1/L2; só consome telemetria via `ingest/`. Hook nega imports de bibliotecas de controle (PLC/Modbus-write) | planejado |
| OT-2 | Sem VPN always-on | Sincronização registry/telemetria via **jump-server bastion + MFA**, sessão efêmera. Studio funciona 100% offline; sync é pull manual on-demand | planejado |
| OT-3 | Edge-first | Inferência roda local (A6000 ou laptop CPU); zero dependência de nuvem no caminho crítico. Telemetria de saída é **store-and-forward** | parcial (inferência local já existe) |
| OT-4 | ETP/WITSML seguro | `ingest/etp_client.py`: WebSocket **TLS 1.3 + JWT** (Bearer); valida `exp`/`aud`/`iss`; refresh por jump-server. WITSML 2.0 via ETP; LAS/DLIS são import de arquivo (sem rede) | ausente |
| OT-5 | Data diode / DMZ | Saída para L4 (registry) atravessa gateway unidirecional quando exigido pelo operador; degrada para "sem sync" sem quebrar | planejado |
| OT-6 | Least-privilege ingest | Cliente ETP usa credencial **somente-leitura** de telemetria; segredo nunca em disco do rig (keyring OS / env injetada) | planejado |

### 1.2 Segredos — sem nada no git

Estado atual: `.gitignore:47-49` cobre `.env`, `*.key`, `credentials.json`; **não há gitleaks** no `.pre-commit-config.yaml` nem no CI.

| # | Controle | Ação concreta | Arquivo | Status |
|:-:|:---------|:--------------|:--------|:------:|
| SEC-1 | gitleaks pre-commit | Adicionar hook `gitleaks` (gate local + bloqueia commit) | `.pre-commit-config.yaml` | ausente |
| SEC-2 | gitleaks CI | Step `gitleaks detect --redact` antes do pytest (gating) | `.github/workflows/ci.yml` | ausente |
| SEC-3 | .env canônico | `.env.example` versionado (chaves vazias); `.env` real ignorado; loader `platform/secrets.py` (env > keyring > erro) | novo | ausente |
| SEC-4 | gitignore estado | Adicionar `*.gsproj`, `*.session`, `*.npz` de campo, `*.dlis`, `*.las` reais ao `.gitignore` (dados de poço = confidenciais) | `.gitignore` | ausente |
| SEC-5 | Reuso do auditor | Acionar skill `geosteering-security-auditor` em PRs sensíveis (já existe) e hook que bloqueia `.env`/keys no diff | `.claude/` | parcial |

### 1.3 Auth da API REST

Estado atual: API tem CORS-hardening, body-size cap, exception handlers tipados (`api/app.py:154-304`) mas **sem autenticação**.

```text
┌──────────────────────────────────────────────────────────────────────┐
│  API AUTH — CAMADAS (defesa em profundidade)                         │
├──────────────────────────────────────────────────────────────────────┤
│  1. Body-size cap 16 MiB        [implementado] app.py:167            │
│  2. CORS allowlist explícita    [implementado] app.py:147            │
│  3. API-Key (X-API-Key)         [ausente]  Dependency simples MVP    │
│  4. JWT Bearer (OAuth2)         [planejado] produção multi-tenant    │
│  5. Rate-limit por chave        [ausente]  slowapi / token-bucket    │
│  6. Audit log (quem/quando)     [ausente]  request_id já existe      │
└──────────────────────────────────────────────────────────────────────┘
```

| Fase | Mecanismo | Decisão | Status |
|:-----|:----------|:--------|:------:|
| MVP+1 | API-Key | `Depends(require_api_key)` lendo `GEOSTEERING_API_KEYS` (CSV hash); 401 sem chave. Reusa o padrão `Settings` frozen de `dependencies.py` | ausente |
| Prod | JWT/OAuth2 | `OAuth2PasswordBearer` + scopes (`predict`, `admin`); mesma origem do JWT do ETP (issuer único) | planejado |
| Prod | Rate-limit | `slowapi` por API-key, default 60 req/min; 429 padronizado via `ErrorResponse` existente | planejado |
| Prod | Audit | Estender o middleware `log_requests` (app.py:195) para emitir evento estruturado com principal | parcial |

### 1.4 Privacidade de dados de poço

| # | Controle | Decisão | Status |
|:-:|:---------|:--------|:------:|
| PRIV-1 | Classificação | Dados de campo (LAS/DLIS/WITSML reais) = **confidenciais do operador**; nunca commitados, nunca em telemetria de crash (ver Sentry §2.4) | planejado |
| PRIV-2 | Logs sem PII de poço | Logger nunca emite coordenadas de poço/nome de operador em nível INFO; só IDs anônimos. Errata: `freq/spacing/seq_len` podem logar (físicos, não sensíveis) | planejado |
| PRIV-3 | Retenção `.gsproj` | Projeto do Studio cifrável em repouso (opt-in, `cryptography` Fernet) quando contém dados reais | planejado |
| PRIV-4 | Export scrub | Relatórios/figuras exportados removem metadados de poço a menos que explicitamente incluídos | planejado |

---

## 2. OBSERVABILIDADE

### 2.1 Logging estruturado (D9) — fundação já existe

`geosteering_ai/utils/logger.py` é o ponto único (regra: nunca `print`, exceto stdout de CLI). Evolução: **JSON opt-in** para a API/serviços.

| # | Item | Decisão | Status |
|:-:|:-----|:--------|:------:|
| LOG-1 | Logger central | `utils/logger.py` único; todos os módulos `logging.getLogger(__name__)` | implementado |
| LOG-2 | JSON formatter | Modo `GEOSTEERING_LOG_FORMAT=json` (API/server) → `python-json-logger`; texto colorido em dev/CLI | parcial |
| LOG-3 | request_id correlação | Já propagado em `app.py:201`; estender para `trace_id` (OTel) | parcial |
| LOG-4 | CLI stdout vs log | CLI usa stdout para resultado, logger para diagnóstico (já é o padrão) | implementado |

### 2.2 Telemetria da API — OpenTelemetry + Prometheus (opt-in)

```text
┌──────────────────────────────────────────────────────────────────────┐
│  API SERVER — pipeline de telemetria (OPT-IN via env)               │
├──────────────────────────────────────────────────────────────────────┤
│  Request → middleware log_requests (existe)                          │
│            ├─ OTel span (trace_id, span_id)   [GEOSTEERING_OTEL=1]   │
│            ├─ Prometheus counter+histogram     [/metrics]            │
│            │     • http_requests_total{route,status}                 │
│            │     • predict_latency_seconds (p50/p95/p99)             │
│            │     • model_load_seconds (cold-start TF)                │
│            │     • inference_batch_size                              │
│            └─ structured log (JSON)                                  │
│  OTLP exporter → Collector → (Tempo/Jaeger + Prometheus + Grafana)  │
└──────────────────────────────────────────────────────────────────────┘
  Default OFF: sem env var, zero overhead e zero dependência nova carregada.
```

| # | Sinal | Implementação | Status |
|:-:|:------|:--------------|:------:|
| OBS-1 | Métricas Prometheus | `prometheus-client` + rota `/metrics` (gated por env); reusa `time.perf_counter` já medido em `app.py:203` | ausente |
| OBS-2 | Tracing OTel | `opentelemetry-instrumentation-fastapi`, exporter OTLP opt-in; injeta `trace_id` no log | ausente |
| OBS-3 | Métricas de modelo | Histogram de latência de inferência + counter de `mc_samples` (UQ) + gauge de modelo carregado | ausente |
| OBS-4 | Health rico | `/health` já existe; adicionar `/health/ready` (modelo carregado) vs `/health/live` | parcial |

### 2.3 SLO / SLA / latência

| Serviço | SLI | SLO | Erro budget | Como medir |
|:--------|:----|:----|:------------|:-----------|
| API `/predict` | latência p99 | < 100 ms (sem UQ) / < 400 ms (MC Dropout 30x) | 1% req/mês | histogram OTel |
| API `/predict` | disponibilidade | 99.5% (warm) | — | `up` + 5xx rate |
| Inferência realtime (Studio) | latência por sample | < 100 ms causal (já é meta `realtime.py`) | — | Sentry perf span |
| Cold-start modelo | tempo 1º load | < 5 s (lazy, ver `dependencies.py:265`) | — | `model_load_seconds` |
| Simulação (SM/Studio) | throughput | não-regressão vs `.claude/perf_baseline.json` | — | benchmark gate (existe) |

### 2.4 Crash-reporting desktop (Studio + SM) — Sentry com privacidade

| # | Item | Decisão | Status |
|:-:|:-----|:--------|:------:|
| OBS-5 | Sentry opt-in | `sentry-sdk` ativado só com consentimento explícito no 1º run (dialog Qt); `GEOSTEERING_TELEMETRY=0` desliga tudo | ausente |
| OBS-6 | Scrub agressivo | `before_send` remove paths de `.gsproj`/`.las`/`.dlis`, coordenadas, nome de operador (PRIV-2) | ausente |
| OBS-7 | Sem dados de poço | Breadcrumbs nunca incluem `raw_data`/arrays; só metadados de UI (qual perspectiva, qual backend) | ausente |
| OBS-8 | Release health | Tag de versão (`__version__`) + commit hash para correlacionar crash↔release | ausente |

---

## 3. ESTRATÉGIA DE TESTES

### 3.1 Pirâmide por camada

```text
                ╱╲          E2E / load (poucos, lentos, caros)
               ╱  ╲         • Studio fluxo completo (pytest-qt + síntese→treino→infere)
              ╱────╲        • API load (locust/k6, p99 sob carga)
             ╱      ╲       INTEGRAÇÃO (médio)
            ╱────────╲      • DataPipeline raw→FV→GS→scale  • dispatch jax→numba
           ╱          ╲     • API TestClient  • ingest LAS→22col
          ╱────────────╲    GOLDEN / PARIDADE (críticos, determinísticos)
         ╱              ╲   • paridade Fortran <1e-12 (SAGRADA)  • jax vs numba <1e-10
        ╱                ╲  • golden de modelo (forward fixo→saída fixa)
       ╱──────────────────╲ UNIT (muitos, rápidos, sem Qt, sem TF pesado)
      ╱                    ╲• ViewModels MVVM puros  • config errata  • losses/metrics
     ╱──────────────────────╲• validators de schema  • secrets loader  • formatters
```

### 3.2 Tipos de teste e cobertura-alvo por produto

| Tipo | Como | Produto-alvo | Cobertura atual → meta |
|:-----|:-----|:-------------|:----------------------:|
| Unit ViewModel (sem Qt) | ViewModel puro Python, mock do model; `pytest` headless puro | Studio | 0% → 90% |
| Unit config/errata | `test_config.py` assert errata; já robusto | Todos | alto (mantém) |
| Integração pipeline | raw→split→FV→GS→scale shapes/leakage | Lib | alto (mantém) |
| Integração dispatch | `dispatch.py:_resolve_backend` mockando `_jax_gpu_available` (já testável) | Lib/SM | parcial → 95% |
| pytest-qt GUI headless | `xvfb-run` + `QT_QPA_PLATFORM=offscreen` (já no CI) | Studio + SM | ~25% → 60% |
| Golden de modelo | input fixo (seed) → saída salva `.npz`; tolerância; detecta drift de arquitetura | Lib | ausente → cobrir Tier-1 |
| Paridade física | Fortran <1e-12 (existe, hook `run-fortran-parity.sh`) | Lib/SM | implementado (inviolável) |
| Paridade JAX↔Numba | <1e-10 (existe) | Lib/SM | implementado |
| API E2E | `TestClient` /health+/predict + auth (quando existir) | API | parcial → 90% |
| Load testing | `locust`/`k6` contra `/predict`, valida p99 SLO §2.3 | API | ausente → planejado |
| Regressão visual | baseline PNG de plots `visualization/` via `pytest-mpl` (`--mpl-baseline`) | Studio/Lib | ausente → Tier-1 plots |
| Regressão perf | benchmark gate vs `perf_baseline.json` (existe) | SM/Lib | implementado |

### 3.3 Regras de teste por produto (acionáveis)

| Produto | Regra-dura | Onde |
|:--------|:-----------|:-----|
| Studio | **ViewModel testável sem importar Qt** (MVVM); 0 lógica em `.ui`/View | `gui/viewmodels/test_*.py` |
| SM | Toda mudança no caminho Numba JIT roda `geosteering-perf-baseline` | hook PR (existe) |
| API | Todo endpoint novo: teste 200 + 4xx + 5xx + auth | `tests/test_api_*.py` |
| CLI | `--backend auto` testado mockando GPU (não exige A6000 no CI) | `tests/test_cli_*.py` |
| Lib | Paridade Fortran NUNCA pode quebrar — gate de merge | `geosteering-physics-reviewer` |

---

## 4. CONFIG & GESTÃO DE ESTADO

### 4.1 Hierarquia de config (já parcialmente implementada)

```text
┌──────────────────────────────────────────────────────────────────────┐
│  CAMADAS DE CONFIGURAÇÃO (frozen → derivado → estado)                │
├──────────────────────────────────────────────────────────────────────┤
│  PipelineConfig (frozen, errata __post_init__)   [config.py:74]      │
│  SimulationConfig (frozen, errata __post_init__) [simulation/config] │
│  api.Settings (frozen, env-driven)               [dependencies.py:123]│
│       ↓ presets YAML (configs/*.yaml — robusto, nstage, realtime)    │
│       ↓ overlay runtime (CLI flags, env) — re-valida via replace()   │
│       ↓ ESTADO de sessão/projeto (.session / .gsproj) — NÃO é config │
└──────────────────────────────────────────────────────────────────────┘
  Regra: config é IMUTÁVEL e validada; estado é mutável e serializável.
         Mutação de config = dataclasses.replace() → re-dispara __post_init__.
```

| # | Item | Decisão | Status |
|:-:|:-----|:--------|:------:|
| CFG-1 | Frozen + errata | `PipelineConfig` (config.py:74,670) e `SimulationConfig` (config.py:264,702) frozen com asserts | implementado |
| CFG-2 | Presets YAML | `from_yaml` + presets de classe (`robusto()`, `nstage()`, `realtime()`) | implementado |
| CFG-3 | api.Settings | frozen, env-driven, `lru_cache` (dependencies.py:149) | implementado |
| CFG-4 | Versão de config | Campo `config_schema_version` no YAML + migração explícita ao carregar `.gsproj` antigo | ausente |
| CFG-5 | Serialização no manifest | `PipelineConfig→JSON` no manifest de reprodutibilidade (gap citado no digest) | parcial |

### 4.2 Estado de sessão e projeto

| Artefato | Conteúdo | Formato | Produto | Status |
|:---------|:---------|:--------|:--------|:------:|
| `.session` | UI volátil: zoom, painel ativo, último diretório | JSON | SM (existe `sm_snapshot_persist.py`) / Studio | parcial |
| `.gsproj` | Projeto completo: ref de config (YAML), datasets, modelos, resultados, layout | **zip+manifest JSON** (não pickle — segurança) | Studio | ausente |
| Manifest | config + seed + git hash + splits (reprodutibilidade) | JSON | Lib (existe `create_manifest`) | implementado |

```text
.gsproj  (zip container — NUNCA pickle, evita RCE)
├── manifest.json        ← schema_version, git_hash, criado_em, app_version
├── config.yaml          ← PipelineConfig serializado (rastreável)
├── sim_config.yaml      ← SimulationConfig (se houver simulação)
├── datasets/refs.json   ← caminhos + checksums (dados não embutidos por padrão)
├── models/<id>.keras    ← modelos treinados (opcional, ou ref)
└── results/             ← figuras, métricas, relatórios MD
```

| # | Item | Decisão | Status |
|:-:|:-----|:--------|:------:|
| ST-1 | Formato seguro | `.gsproj` = zip + JSON; **proibido pickle** (desserialização não confiável = RCE) | planejado |
| ST-2 | Versionamento | `schema_version` no manifest; loader migra ou recusa graciosamente versões futuras | planejado |
| ST-3 | Checksums de dataset | refs com SHA-256; aviso se dataset mudou (reprodutibilidade) | planejado |
| ST-4 | Autosave `.session` | Studio/SM autosave a cada N s (debounced) — base já em `sm_snapshot_persist.py` | parcial |

---

## 5. ERROR HANDLING & RESILIÊNCIA

### 5.1 Matriz de erros

```text
┌──────────────────────────────────────────────────────────────────────┐
│  CLASSE DE ERRO        ESTRATÉGIA            EXEMPLO                  │
├──────────────────────────────────────────────────────────────────────┤
│  Errata física         FAIL-FAST (assert)    freq=2.0 → AssertionError│
│  Backend indisponível  FALLBACK gracioso     jax GPU ausente → numba  │
│  Modelo não carregado  503 tipado            ModelNotLoadedError       │
│  Payload inválido      422 Pydantic          raw_data shape errado     │
│  Sessão corrompida     RECUPERAÇÃO parcial   carrega último .session ok│
│  Telemetria ETP caiu   STORE-AND-FORWARD     buffer local, retry      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Decisões de resiliência

| # | Cenário | Estratégia concreta | Status |
|:-:|:--------|:--------------------|:------:|
| RES-1 | Errata física | Fail-fast no `__post_init__` (config.py:692, simulation/config.py:717) — NUNCA degradar silenciosamente valor físico | implementado |
| RES-2 | jax→numba fallback | `dispatch._resolve_backend` com `numba_fallback`; `_jax_gpu_available()` captura toda exceção → Numba (dispatch.py:69-85). Logar WARN, nunca crashar | parcial |
| RES-3 | Modelo ausente API | `ModelNotLoadedError`→503, `ModelLoadFailedError`→503 tipados (dependencies.py:88-102) | implementado |
| RES-4 | Cold-start TF | Lazy load no 1º `/predict`, double-checked locking (dependencies.py:251); idempotente | implementado |
| RES-5 | Recuperação de sessão | Ao crashar, Studio/SM oferece restaurar último `.session` autosalvo; se corrompido, carrega default sem perder o resto | planejado |
| RES-6 | Idempotência API | `/predict` puro (sem efeito colateral); `x-request-id` permite dedup no cliente (app.py:201) | parcial |
| RES-7 | Telemetria store-forward | Saída ETP/registry bufferiza localmente; retry exponencial; nunca bloqueia inferência (OT-3) | planejado |
| RES-8 | Warmup gracioso | `geosteering-warmup --auto` degrada JAX→só-Numba sem falhar (CI já faz, ci.yml:67) | implementado |

---

## 6. DOCUMENTAÇÃO (D1-D14) & GOVERNANÇA

### 6.1 Aplicação de D1-D14 aos transversais

| Padrão | Aplicação no cross-cutting | Status |
|:-------|:---------------------------|:------:|
| D1 mega-header | Todo módulo novo (`platform/secrets.py`, `ingest/`, `gui/viewmodels/`) | regra |
| D3 diagramas ASCII | Pipeline de auth, Purdue, telemetria — como neste doc | regra |
| D5/D6 docstrings | Funções/classes públicas de segurança e config | regra |
| D9 logging | **já é regra-dura** (nunca print); estende para JSON | implementado |
| D12 cross-ref | Toda função de segurança referencia o ADR correspondente | regra |
| PT-BR acentuado | Todo doc/comentário (este incluso) | regra |

### 6.2 Governança — ADRs novos a criar

| ADR | Título | Decide |
|:----|:-------|:-------|
| ADR-0010 | Modelo de auth da API (API-Key→JWT) | mecanismo, ordem de adoção |
| ADR-0011 | Telemetria opt-in (OTel/Prometheus/Sentry) | consentimento, scrub, default OFF |
| ADR-0012 | Formato `.gsproj` (zip+JSON, anti-pickle) | container, versionamento |
| ADR-0013 | OT no rig: Studio read-only L1/L2 | isolamento Purdue, edge-first |
| ADR-0014 | gitleaks + pip-audit gating no CI | gates de segurança |

Regra ADR-0001 (já vigente): só `docs/ROADMAP.md` define versões futuras; estes transversais entram no backlog por **code** (ex: `X-security-otrig`, `X-obs-otel`) até o 1º commit da sprint.

### 6.3 CI — gates de segurança a adicionar

```text
.github/workflows/ci.yml  (ordem proposta dos novos steps)
  setup python ─→ install ─→ [NOVO gitleaks] ─→ [NOVO pip-audit] ─→
  compile ─→ warmup ─→ pytest ─→ gui(xvfb) ─→ benchmark ─→ mypy ─→
  [NOVO bandit -r geosteering_ai/platform geosteering_ai/api]
```

| Gate | Comando | Gating? |
|:-----|:--------|:-------:|
| gitleaks | `gitleaks detect --redact --no-banner` | SIM (bloqueia) |
| pip-audit | `pip-audit -r requirements` | SIM (CVE high) |
| bandit | `bandit -r geosteering_ai/{platform,api,ingest}` | SIM (módulos de borda) |
| ruff/mypy | já existem (ci.yml:94) | SIM |

---

## 7. Roadmap transversal priorizado (síntese acionável)

| Prioridade | Item | Produtos | Esforço | Bloqueia |
|:----------:|:-----|:---------|:-------:|:---------|
| P0 | gitleaks + pip-audit + bandit no CI (SEC-1,2; gate §6.3) | Todos | baixo | credibilidade |
| P0 | gitignore `.gsproj/.las/.dlis/.npz` (SEC-4) | Todos | trivial | privacidade |
| P1 | API-Key auth (§1.3 MVP+1) | API | baixo | qualquer deploy |
| P1 | ViewModel testável sem Qt (§3.3) | Studio | médio | toda a casca Studio |
| P1 | `.gsproj` zip+JSON anti-pickle (ST-1,2) | Studio | médio | persistência |
| P2 | OTel/Prometheus opt-in (OBS-1,2) | API | médio | SLO observável |
| P2 | Sentry desktop com scrub (OBS-5,6,7) | Studio/SM | médio | crash visibility |
| P2 | jax→numba fallback robusto + WARN (RES-2) | Lib/SM/CLI | baixo | `--backend auto` |
| P3 | ETP JWT+TLS client (OT-4) | Studio F2 | alto | campo real |
| P3 | Recuperação de sessão (RES-5) | Studio/SM | médio | UX robustez |
