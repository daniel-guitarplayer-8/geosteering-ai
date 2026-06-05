> **Blueprint de Arquitetura do Geosteering AI — API Design.** Índice: [README.md](README.md) · Constituição SDD: [../../specs/CONSTITUTION.md](../../specs/CONSTITUTION.md) · Roadmap: [../../specs/ROADMAP.md](../../specs/ROADMAP.md). Gerado 2026-06-05 (workflow multi-agente + revisão crítica).

## API Design — 3 Superfícies Programáveis do Geosteering AI

Documento de contrato versionável para as três superfícies de consumo do backend (`geosteering_ai/`): a **Biblioteca Python (SDK)**, a **API REST (FastAPI)** e a **CLI (`geosteering-cli`)**. O Studio e o Simulation Manager consomem a **camada SDK** (importação direta), nunca a REST nem a CLI — portanto o SDK é a superfície de contrato mais crítica do portfólio.

```
┌───────────────────────────────────────────────────────────────────────────┐
│  PIRÂMIDE DE SUPERFÍCIES (todas convergem na física: multi_forward.py)     │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   CLI (geosteering-cli)        REST (geosteering-api)                      │
│        │                            │                                     │
│        └──────────┬─────────────────┘   ← consumidores externos / scripts │
│                   ▼                                                        │
│        ┌──────────────────────┐                                           │
│        │  SDK Python público  │  ← Studio MVVM + Simulation Manager       │
│        │  (geosteering_ai.*)  │     importam AQUI (nunca REST/CLI)        │
│        └──────────┬───────────┘                                           │
│                   ▼                                                        │
│   PipelineConfig · ModelRegistry · TrainingLoop · InferencePipeline ·     │
│   RealtimeInference · simulate_batch · DataPipeline                       │
│                   ▼                                                        │
│        simulation/multi_forward.py  (Numba <1e-12 / JAX <1e-10 paridade)  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 1. BIBLIOTECA Python (SDK)

### 1.1 Problema atual — versão e tipagem

| Item | Estado atual | Evidência | Ação |
|:-----|:-------------|:----------|:-----|
| Versão hardcoded | `__version__ = "2.0.0"` duplicado | `geosteering_ai/__init__.py:63` + `pyproject.toml:7` | SSoT (§1.2) |
| CLI desincronizada | `GEOSTEERING_CLI_VERSION = "v2.56"` separado | `cli/_main.py:119` | unificar via `importlib.metadata` |
| `py.typed` | **AUSENTE** (`find` retornou vazio) | — | criar marker PEP 561 |
| Política de deprecação | inexistente | — | adotar PEP 387 (§1.4) |
| Fronteira público/interno | implícita (prefixo `_`) | `_jax/`, `_numba/`, `_main.py` | formalizar `__all__` + tabela (§1.3) |

### 1.2 Semver real (PEP 440) — fim do `2.0.0` hardcoded

**Decisão**: SSoT da versão em `pyproject.toml`; o pacote lê via `importlib.metadata`. A constante `v2.56` da CLI é o histórico de *sprint* (ortogonal ao semver do pacote) — renomear para `__sprint_tag__` e deixar `--version` reportar o semver do pacote.

```
┌─────────────────────────────────────────────────────────────────┐
│  FLUXO DE VERSÃO (uma fonte → todas as superfícies)              │
├─────────────────────────────────────────────────────────────────┤
│  pyproject.toml [project].version = "2.41.0"   ◄── SSoT          │
│        │                                                        │
│        ▼  importlib.metadata.version("geosteering-ai")          │
│  geosteering_ai.__version__ ─┬─► SDK  __version__               │
│                              ├─► REST  /health.version, /v1     │
│                              └─► CLI  `version` / `--version`    │
└─────────────────────────────────────────────────────────────────┘
```

```python
# geosteering_ai/__init__.py  (substitui linha 63)
from importlib.metadata import PackageNotFoundError, version as _pkg_version
try:
    __version__ = _pkg_version("geosteering-ai")
except PackageNotFoundError:          # editable/source tree sem instalar
    __version__ = "0.0.0.dev0"
```

**Política semver aplicada ao domínio**:

| Mudança | Bump | Exemplo concreto |
|:--------|:-----|:-----------------|
| Quebra de assinatura pública / remoção de export do `__all__` | **MAJOR** | remover `theta=` de `InferencePipeline.predict` |
| Novo export, novo kwarg opcional, nova arquitetura no `ModelRegistry` | **MINOR** | adicionar `backend="auto"` à CLI; nova loss |
| Bugfix sem mudança de assinatura | **PATCH** | corrigir EPS float32 |
| **Quebra de paridade Fortran <1e-12** | **MAJOR + nota física** | mudança de filtro Hankel default |

Regra-dura: alteração na **errata física** (`FREQUENCY_HZ`, `INPUT_FEATURES=[1,4,5,20,21]`, `OUTPUT_TARGETS=[2,3]`, `TARGET_SCALING="log10"`) é sempre MAJOR e exige ADR.

### 1.3 Superfície pública estável vs interna

`geosteering_ai/__init__.py:300` já define `__all__` (~37 símbolos). Formalizo a fronteira em **3 tiers de contrato**:

```
┌──────────────┬───────────────────────────────────────┬──────────────────────────┐
│  Tier        │  Garantia                             │  Localização             │
├──────────────┼───────────────────────────────────────┼──────────────────────────┤
│  PÚBLICO     │  Semver estrito · deprec. ≥2 minors   │  topo de geosteering_ai/ │
│  (estável)   │  Re-exportado no __init__.__all__      │  via __init__            │
├──────────────┼───────────────────────────────────────┼──────────────────────────┤
│  PROVISÓRIO  │  @experimental · pode mudar em MINOR   │  submódulo público +     │
│  (beta)      │  emite GeosteeringExperimentalWarning  │  warning                 │
├──────────────┼───────────────────────────────────────┼──────────────────────────┤
│  INTERNO     │  SEM garantia · pode mudar em PATCH    │  prefixo _ (_jax/,       │
│  (privado)   │  NÃO importar de fora do pacote        │  _numba/, _workers, etc.)│
└──────────────┴───────────────────────────────────────┴──────────────────────────┘
```

**Tabela de superfície ESTÁVEL (contrato congelado em v2.41)** — assinaturas reais verificadas no código:

| Símbolo público | Tipo | Assinatura-chave (verificada) | Arquivo:linha | Status |
|:----------------|:-----|:------------------------------|:--------------|:-------|
| `PipelineConfig` | dataclass | `from_yaml(path) · robusto() · nstage(n) · realtime(model_type)` | `config.py` | implementado |
| `DataPipeline` | classe | `prepare(dataset_path) -> PreparedData · build_train_map_fn(noise_var)` | `data/pipeline.py` | implementado |
| `ModelRegistry` | classe | `build(config) -> keras.Model · list_available() · get_info(name)` | `models/registry.py` | implementado |
| `build_model` | função | `build_model(config) -> keras.Model` | `models/` | implementado |
| `is_causal_compatible` | função | `is_causal_compatible(model_type) -> bool` | `models/` | implementado |
| `build_loss_fn` | função | `build_loss_fn(config) -> Callable` | `losses/` | implementado |
| `TrainingLoop` | classe | `__init__(config)` · `fit(...)` (L694) · `run(...)` (L917) | `training/loop.py:289` | implementado |
| `InferencePipeline` | classe | `predict(raw, *, theta, freq, return_uncertainty, mc_samples)` · `load(path)` | `inference/pipeline.py:104` | implementado |
| `RealtimeInference` | classe | `update(sample) -> result` · `reset()` | `inference/realtime.py:102` | implementado |
| `simulate_batch` | função | `simulate_batch(rho_h_batch, rho_v_batch, esp_batch, positions_z, *, backend="auto", ...) -> (H, info)` | `simulation/dispatch.py:233` | implementado |
| `CurriculumSchedule` | classe | schedule 3-phase noise | `noise/` | implementado |
| `__version__` | str | semver | `__init__.py:63` | parcial (hardcoded) |

**Contratos das fachadas principais** (invariantes que o semver protege):

```
┌─ PipelineConfig ─────────────────────────────────────────────────────────┐
│ INVARIANTE: __post_init__ valida errata. Passar config como PARÂMETRO     │
│ (nunca globals). from_yaml round-trip estável. Presets de classe são API. │
├─ ModelRegistry ──────────────────────────────────────────────────────────┤
│ INVARIANTE: build(config) é determinístico p/ mesma seed. 48 arquiteturas │
│ estáveis por nome; remover nome = MAJOR. list_available() = contrato.     │
├─ InferencePipeline ──────────────────────────────────────────────────────┤
│ INVARIANTE: predict() retorna ndarray OU (mean,std) se return_uncertainty.│
│ Saída em Ω·m (domínio físico, pós inverse-scaling). raw_data shape        │
│ (N, seq, 22). load(path) carrega SavedModel + scalers + config manifest.  │
├─ RealtimeInference ──────────────────────────────────────────────────────┤
│ INVARIANTE: causal-only (model_type ∈ causal_compatible). update() é      │
│ O(1) amortizado (buffer circular FIFO). Latência alvo <100ms/sample.      │
├─ simulate_batch ─────────────────────────────────────────────────────────┤
│ INVARIANTE: backend="auto" expõe _resolve_backend (árvore medida). Saída  │
│ (H_tensor, info). Paridade JAX↔Numba <1e-10. dtype c128 default = paridade│
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Política de deprecação (PEP 387)

| Fase | Duração mínima | Mecanismo | Visível em |
|:-----|:--------------|:----------|:-----------|
| 1. Anúncio | release N | `DeprecationWarning` + nota no CHANGELOG + `@deprecated("2.43", "use X")` | docstring + runtime |
| 2. Janela de migração | ≥2 minor releases | símbolo funcional, warning persiste | logs |
| 3. Remoção | release N+2 (MAJOR se público) | `AttributeError` / remoção do `__all__` | — |

Implementação concreta: decorator `geosteering_ai/utils/deprecation.py::deprecated(since, alternative, removal)` que emite `DeprecationWarning` (categoria `GeosteeringDeprecationWarning` própria, filtrável). Experimental usa `GeosteeringExperimentalWarning`.

### 1.5 py.typed / PEP 561

| Ação | Detalhe |
|:-----|:--------|
| Criar `geosteering_ai/py.typed` (arquivo vazio) | sinaliza pacote tipado a mypy/pyright dos consumidores |
| `pyproject.toml` | adicionar `[tool.setuptools.package-data] geosteering_ai = ["py.typed"]` |
| Gate CI | `mypy --strict` sobre a **superfície pública** (já há `[tool.mypy] python_version="3.13"` em `pyproject.toml:134`) |

---

## 2. REST (FastAPI)

### 2.1 Estado atual (verificado)

App em `geosteering_ai/api/app.py:117` (`create_app()` factory). Já implementado: CORS configurável, middleware body-size (413), middleware logging com `x-request-id`, exception handlers tipados (`ModelNotLoadedError`/`ModelLoadFailedError`→503, `RuntimeError`→500). Schemas Pydantic v2 em `schemas.py`. **Auth: AUSENTE** (`grep` por api_key/jwt/bearer = 0 resultados). **Versionamento de path: AUSENTE** (rotas montadas em `/health`, `/predict` sem prefixo).

### 2.2 Versionamento `/v1` (decisão)

Montar todos os roteadores de domínio sob `APIRouter(prefix="/v1")`. `/health` permanece **sem prefixo** (probe de infra é ortogonal ao contrato de negócio). OpenAPI `info.version` = semver do pacote (§1.2). Quebra de schema = `/v2` coexistente; `/v1` entra em deprecação anunciada via header `Deprecation` (RFC 8594).

```
GET  /health                 ← sem versão (liveness/readiness, <1ms, sem TF)
GET  /v1/...                  ← contrato de negócio versionado
GET  /openapi.json /docs /redoc  (toggle via GEOSTEERING_API_DOCS_ENABLED)
```

### 2.3 Tabela de endpoints (atual + propostos)

| Método | Path | Request (Pydantic) | Response | Auth | Status |
|:-------|:-----|:-------------------|:---------|:-----|:-------|
| GET | `/health` | — | `HealthResponse{status,version,model_loaded,model_path}` | nenhuma | **implementado** (`routes/health.py`) |
| POST | `/v1/predict` | `PredictRequest{raw_data[N,seq,22], theta?, freq?, return_uncertainty?, mc_samples?}` | `PredictResponse{predictions, uncertainty?, shape, latency_ms, model_type, mc_samples?}` | X-API-Key | **parcial** (existe em `/predict`, sem auth/prefixo) |
| POST | `/v1/predict:batch` | `PredictBatchRequest{items[], parallelism?}` | `PredictBatchResponse{results[], n_ok, n_err}` | X-API-Key | **ausente** |
| POST | `/v1/simulate` | `SimulateRequest{rho_h_batch, rho_v_batch, esp_batch, positions_z, frequencies_hz?, dips?, tr_spacings?, backend="auto", dtype}` | `SimulateResponse{H_shape, backend, reason, n_geometry_groups, elapsed_s, download_uri?}` | X-API-Key | **ausente** (envolve `simulate_batch`) |
| POST | `/v1/jobs/train` | `TrainJobRequest{config(yaml\|inline), dataset_uri, callbacks?}` | `JobAccepted{job_id, status="queued", poll_uri}` (**202**) | X-API-Key (scope `train`) | **ausente** (async) |
| GET | `/v1/jobs/{job_id}` | — | `JobStatus{job_id, state, progress, metrics?, artifact_uri?, error?}` | X-API-Key | **ausente** |
| DELETE | `/v1/jobs/{job_id}` | — | `204` | X-API-Key (scope `train`) | **ausente** |
| GET | `/v1/models` | query `?causal=true&tier=1` | `ModelList{items:[{name,family,tier,causal,params}]}` | X-API-Key | **ausente** (envolve `ModelRegistry.list_available`) |
| GET | `/v1/models/{name}` | — | `ModelInfo{name,family,tier,causal,refs,input_shape}` | X-API-Key | **ausente** |
| POST | `/v1/registry/models` | `RegisterModelRequest{name, artifact_uri, metrics, config}` | `RegisteredModel{id, version}` (**201**) | X-API-Key (scope `registry:write`) | **ausente** (MLflow — gap-chave) |

**Diagrama do job assíncrono de treino** (TF não bloqueia o event loop):

```
POST /v1/jobs/train ──► 202 {job_id, poll_uri}
                          │
            (enfileira em backend de jobs: in-proc ThreadPool no MVP;
             Celery/RQ + GPU local A6000 em produção — TrainingLoop.run)
                          │
  GET /v1/jobs/{id} ──► {state: queued→running(progress)→succeeded|failed}
                          │
                  artifact_uri ──► SavedModel/TFLite/ONNX p/ /v1/registry
```

### 2.4 Schemas Pydantic — adições

Constantes da errata já espelhadas em `schemas.py:64-83` (`N_COLUMNS=22`, ranges theta/freq, `MC_SAMPLES_MIN/MAX`, `MAX_N_SAMPLES=1024`). Novos schemas seguem o mesmo padrão de `field_validator` fail-fast:

| Schema | Validações fail-fast |
|:-------|:---------------------|
| `SimulateRequest` | `len(rho_h)==len(rho_v)`; `esp.shape[-1]==n-2`; `backend ∈ {auto,jax,numba}`; `dtype ∈ {complex128,complex64}`; cap `n_models ≤ MAX_N_MODELS` |
| `TrainJobRequest` | `config` valida via `PipelineConfig.__post_init__` (reusa errata); `dataset_uri` scheme allowlist (`file://`, `s3://`) |
| `ModelList` query | `tier ∈ {1,2,3}`; `causal` bool |

### 2.5 Auth, rate-limit, segurança edge (IEC 62443)

```
┌────────────────────────────────────────────────────────────────────────┐
│  CADEIA DE SEGURANÇA (ordem de middleware — outer→inner)                 │
├────────────────────────────────────────────────────────────────────────┤
│  1. TLS termination (reverse proxy — fora do escopo FastAPI)            │
│  2. Body-size 413  ◄── JÁ IMPLEMENTADO (app.py:167)                     │
│  3. Rate-limit 429 (slowapi, por API-Key)        ◄── ausente            │
│  4. Auth: X-API-Key (MVP) → JWT Bearer + scopes (prod)  ◄── ausente     │
│  5. request_id + logging  ◄── JÁ IMPLEMENTADO (app.py:195)              │
│  6. Route handler                                                       │
└────────────────────────────────────────────────────────────────────────┘
```

| Camada | MVP | Produção | Mapeamento IEC 62443 |
|:-------|:----|:---------|:---------------------|
| AuthN | `X-API-Key` (header), hash em env/secret | JWT Bearer (OIDC), rotação | FR1 Identification & Authentication |
| AuthZ | binário (key válida = full) | scopes (`predict`, `train`, `registry:write`) | FR2 Use Control |
| Rate-limit | — | `slowapi` por key, 429 + `Retry-After` | FR7 Resource Availability (anti-DoS) |
| Integridade | body-size 413 (feito) | + checksum de artefatos no registry | FR3 System Integrity |
| Auditoria | logging request_id (feito) | log estruturado imutável de `train`/`registry` | FR6 Timely Response to Events |
| Confidencialidade | TLS no proxy | mTLS zona OT↔IT | FR4 Data Confidentiality |

Dependency FastAPI concreta (a injetar em todas as rotas `/v1` exceto `/health`):

```python
# geosteering_ai/api/security.py
async def require_api_key(x_api_key: str = Header(...)) -> ApiKeyContext:
    if not _verify(x_api_key):            # hash-compare contra GEOSTEERING_API_KEYS
        raise HTTPException(401, "API key inválida ou ausente")
    return ApiKeyContext(scopes=_scopes_for(x_api_key))
```

---

## 3. CLI (`geosteering-cli`)

### 3.1 Estado atual (verificado)

Entry points em `pyproject.toml:108-114`: `geosteering-cli`, `geosteering-warmup`, `geosteering-api`. Parser em `cli/_main.py:310` com 3 subcomandos: `simulate`, `benchmark`, `version`. Args comuns de backend em `_add_common_backend_args` (`_main.py:213`). Exit codes documentados em `main()` (0/1/2). `--json` e `--quiet` já existem.

### 3.2 CRÍTICO — `--backend auto` (default)

**Bug de exposição**: `_main.py:226-231` declara `choices=["numba","jax"], default="numba"`, mas `dispatch._resolve_backend` (`dispatch.py:94`) **já implementa e testa** `"auto"` (árvore de decisão medida). A CLI não expõe a capacidade que o backend já tem.

```diff
# cli/_main.py:226  _add_common_backend_args
  p.add_argument(
      "--backend",
-     choices=["numba", "jax"],
-     default="numba",
+     choices=["auto", "numba", "jax"],
+     default="auto",
      help="backend: auto (padrão, dispatch medido) | numba (CPU) | jax (GPU)",
  )
```

```
┌─ --backend auto (default proposto) ───────────────────────────────────────┐
│  geosteering-cli simulate --models 5000 --geometry templates              │
│        │                                                                  │
│        ▼  _resolve_backend(n_models, n_geometry_groups, gpu?)             │
│   ┌────────────┴────────────┐                                            │
│   n_models≥32 & agrupável & GPU?   →  jax   (motivo no info.reason)       │
│   senão                            →  numba (CPU pool efêmero)            │
└───────────────────────────────────────────────────────────────────────────┘
```

Compat: usuário que fixava `--backend numba` continua válido (MINOR, não MAJOR — só amplia `choices` e troca default para a opção mais inteligente).

### 3.3 Tabela de comandos (atual + propostos)

| Comando | Args principais | Saída | Fachada SDK | Status |
|:--------|:----------------|:------|:------------|:-------|
| `simulate` | `--models --n-pos --frequencies --dips --tr-spacings --backend --geometry --out --format{npz,dat,none} --json` | tabela / npz / .dat 22-col | `simulate_batch` / `simulate_multi` | **implementado** (`--backend` sem `auto`) |
| `benchmark` | `--scenario{A..H} --n --backend --compare-backends --repeat --json` | throughput mod/h | dispatch | **implementado** |
| `version` | — | `Geosteering AI CLI vX.Y` | `__version__` | **parcial** (constante separada) |
| `warmup` | `--verbose` (entry point separado `geosteering-warmup`) | timing JIT | `_warmup_numba_tier2_sync` | **implementado** (não-subcomando) |
| `generate-dataset` | `--config --models --out --format{npz,tfrecord} --seed --noise` | dataset 22-col + manifest | `SyntheticDataGenerator` + `DataPipeline` | **ausente** |
| `train` | `--config(yaml) --dataset --epochs --model --out --resume` | checkpoints + history | `TrainingLoop.run` | **ausente** |
| `infer` (alias `predict`) | `--model(path) --input --theta --freq --uncertainty --out --json` | predições Ω·m | `InferencePipeline.load().predict` | **ausente** |
| `registry` | `list \| add --artifact \| show NAME \| promote NAME` | tabela de modelos | `ModelRegistry` + MLflow | **ausente** |
| `serve` | `--host --port --model --workers` | sobe `geosteering-api` | `api.cli:main` | **parcial** (entry point separado) |

### 3.4 Convenções (contrato versionável)

```
┌─ EXIT CODES (estende main() _main.py:535) ─────────────────────────────────┐
│  0  sucesso                                                                │
│  1  erro de execução (modelo inválido, OSError, falha de backend)         │
│  2  argumento inválido / subcomando ausente (argparse SystemExit)         │
│  3  PROPOSTO: erro de configuração (PipelineConfig.__post_init__ falhou)  │
│  4  PROPOSTO: erro de I/O de artefato (dataset/checkpoint não encontrado) │
│ 130 PROPOSTO: interrompido (SIGINT) — captura KeyboardInterrupt           │
└───────────────────────────────────────────────────────────────────────────┘
```

| Convenção | Regra | Estado |
|:----------|:------|:-------|
| `--json` | saída machine-readable no stdout; logs humanos no **stderr** | implementado (`simulate`/`benchmark`) → **estender a todos** |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | substitui `logging.basicConfig` fixo em `_main.py:559` | **ausente** (hoje hardcoded INFO) |
| `--quiet` | suprime tabela, mantém erros + linha grep-able | implementado |
| stdout limpo | só `version`/resultados `--json` escrevem stdout (exceção D9 documentada `_main.py:586`) | implementado |
| shell completion | `argcomplete` (`eval "$(register-python-argcomplete geosteering-cli)")` | **ausente** → adicionar |
| `--version` global | flag top-level além do subcomando `version` (convenção POSIX) | **ausente** |

---

## Resumo de gaps acionáveis por superfície

| Superfície | Gap | Esforço | Prioridade |
|:-----------|:----|:--------|:-----------|
| SDK | `importlib.metadata` (remover `2.0.0` hardcoded ×2) | baixo | alta |
| SDK | criar `py.typed` + package-data | trivial | alta |
| SDK | `utils/deprecation.py` + warnings PEP 387/experimental | médio | média |
| REST | `security.py` X-API-Key dependency + 401 | baixo | alta |
| REST | prefixo `/v1` em roteadores de negócio | trivial | alta |
| REST | `/v1/simulate`, `/v1/models`, jobs `/v1/jobs/train` (202) | alto | média |
| REST | rate-limit `slowapi` (429) | baixo | média |
| CLI | `--backend auto` default (expor `_resolve_backend`) | trivial | **crítica** |
| CLI | `--log-level` + `--version` global + exit codes 3/4/130 | baixo | alta |
| CLI | subcomandos `train`/`infer`/`generate-dataset`/`registry` | alto | média |
