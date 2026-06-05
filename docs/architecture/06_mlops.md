> **Blueprint de Arquitetura do Geosteering AI — MLOps.** Índice: [README.md](README.md) · Constituição SDD: [../../specs/CONSTITUTION.md](../../specs/CONSTITUTION.md) · Roadmap: [../../specs/ROADMAP.md](../../specs/ROADMAP.md). Gerado 2026-06-05 (workflow multi-agente + revisão crítica).

# Arquitetura MLOps — Geosteering AI

> Escopo: tornar o backend (~90% pronto) **operável, auditável e promovível** sem violar nenhuma restrição inviolável (TF/Keras exclusivo, paridade Fortran `<1e-12`, errata física, Python 3.13 conda, A6000 local, Config-como-parâmetro). Tudo on-prem na A6000 — sem dependência de nuvem (alinhado à descontinuação do Colab, v2.44).
>
> Princípio-mestre: **estender a cultura de paridade `<1e-12` do simulador para o ciclo de vida de modelos treinados**. Um modelo só promove se reproduzir golden-tests bit-a-bit-tolerante; um runtime quantizado só promove se passar gate de paridade vs. baseline FP32.

---

## 0. Mapa de gaps → componentes MLOps propostos

| # | Gap (CONTEXTO) | Onde fica hoje | Componente MLOps proposto | Status |
|:-:|:---------------|:---------------|:--------------------------|:------:|
| 1 | MLflow/registry AUSENTE | — | `geosteering_ai/mlops/registry/` (MLflow on-prem) | ausente → planejado |
| 2 | Lineage parcial (git/seed manuais) | `evaluation/manifest.py:105` | `ModelVersion` entidade + auto-captura | parcial → planejado |
| 3 | Sem autolog de treino | `training/loop.py:694` (`fit()`) | `MLflowAutologCallback` (Keras) | ausente → planejado |
| 4 | Golden-test só p/ física | `tests/*parity*` | `tests/golden/` saída de inferência por versão | ausente → planejado |
| 5 | INT8 só dynamic-range | `inference/export.py:182,216` | full-INT8 + gate de paridade pré/pós | parcial → planejado |
| 6 | Sem runtime ONNX/LiteRT empacotado | `export.py` exporta, não serve | `mlops/runtime/` (ORT + LiteRT) | parcial → planejado |
| 7 | Sem drift/model card/audit | — | `mlops/monitoring/` | ausente → planejado |
| 8 | UQ sem P10/P50/P90+CRPS+coverage | `inference/uncertainty.py`, `visualization/uncertainty.py:649` | `mlops/uq_calibration/` | parcial → planejado |
| 9 | Validação de campo real | — (gap #1 credibilidade) | `mlops/validation/` (Volve/Goliat gate) | ausente → planejado |

> Decisão de pacote: tudo novo entra em **`geosteering_ai/mlops/`** (subpacote pip-installable). Não polui `simulation/` (paridade sagrada) nem `gui/`. Reutiliza `evaluation/manifest.py`, `inference/export.py`, `inference/uncertainty.py`, `training/loop.py` por composição — zero duplicação de física.

---

## 1. Ciclo MLOps — Diagrama ASCII

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                         GEOSTEERING AI — CICLO MLOps ON-PREM (A6000)                    │
└──────────────────────────────────────────────────────────────────────────────────────┘

   [SIMULADOR/DADOS]                  [TREINO + HPO]                    [REGISTRY]
   ┌───────────────┐  on-the-fly      ┌──────────────────┐  autolog     ┌───────────────────┐
   │ SyntheticGen  │ ───────────────► │ TrainingLoop     │ ───────────► │ MLflow (on-prem)  │
   │ (Numba/JAX)   │  noise→FV→GS     │ + Optuna HPO     │  params,     │ ──────────────    │
   │ paridade      │                  │ + MLflowAutolog  │  metrics,    │ ModelVersion(N)   │
   │ Fortran<1e-12 │                  │   Callback       │  curvas,     │  ├ git_hash       │
   └──────┬────────┘                  └────────┬─────────┘  artefatos   │  ├ gen_version    │
          │ gen_version+seed                   │                        │  ├ seed+dataset   │
          ▼ + dataset_hash                     ▼                        │  ├ config_sha256  │
   ┌───────────────┐                  ┌──────────────────┐              │  └ stage: STAGING │
   │ DATASET CARD  │                  │  Manifest        │              └─────────┬─────────┘
   │ (lineage tag) │ ◄────────────────│ create_manifest()│                        │
   └───────────────┘                  │ evaluation/      │                        │ promote?
                                       │ manifest.py:105  │                        ▼
                                       └──────────────────┘              ┌───────────────────┐
                                                                         │  GATES DE PROMOÇÃO │
   ┌──────────────────────────────────────────────────────────────────► │  (CI + manual)     │
   │                                                                     ├───────────────────┤
   │  ┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐   │ G1 golden infer   │
   │  │ GOLDEN-TESTS   │   │ QUANTIZAÇÃO INT8 │   │ UQ CALIBRADA     │   │ G2 paridade INT8  │
   │  │ inferência/ver │──►│ + gate paridade  │──►│ P10/P50/P90 CRPS │──►│ G3 coverage≥nom   │
   │  │ tests/golden/  │   │ FP32 vs INT8     │   │ coverage test    │   │ G4 model card OK  │
   │  └────────────────┘   └──────────────────┘   └──────────────────┘   │ G5 validação real │
   │                                                                     └─────────┬─────────┘
   │                                                                               │ pass
   │                                                                               ▼
   │   [RUNTIME PRODUÇÃO]                                              ┌───────────────────┐
   │   ┌──────────────────┐         ┌──────────────────┐              │ stage: PRODUCTION │
   │   │ ONNX Runtime     │         │ TFLite/LiteRT    │              │ (ou TECH PREVIEW  │
   │   │ batch/offline    │         │ INT8 realtime    │              │  se G5 incompleto)│
   │   │ API REST /predict│         │ <100ms Studio/SM │              └─────────┬─────────┘
   │   └────────┬─────────┘         └────────┬─────────┘                        │
   │            │  inferências                │  inferências                     │
   │            ▼                             ▼                                  │
   │   ┌──────────────────────────────────────────────────┐                     │
   │   │           MONITORAMENTO EM PRODUÇÃO               │                     │
   │   │  ├ drift KS/PSI/Wasserstein sobre Hxx/Hzz         │ ───── alerta ──────►│ rollback /
   │   │  ├ trilha de auditoria (inference log assinado)   │      drift/OOD      │ archive
   │   │  ├ extrapolação fora de ranges do model card      │                     │
   │   │  └ coverage observado vs nominal (online)         │ ────────────────────┘
   │   └──────────────────────────────────────────────────┘
   │                                                                  ▲
   └──────────────────── retrain trigger (drift sustentado) ─────────┘
```

---

## 2. Model Registry + Lineage

### 2.1 Decisão estrutural — colisão de nomes `ModelRegistry`

`geosteering_ai/models/registry.py:834` já define `ModelRegistry` — mas é um **factory de arquiteturas** (48 funções `build_*`), NÃO um registry de versões. **Não renomear** (quebraria `training/loop.py:847` e exports). O novo componente chama-se **`ModelStore`** (camada de domínio) sobre o **MLflow Model Registry** (camada de infra).

```
geosteering_ai/mlops/registry/
├── model_version.py    ← ModelVersion (entidade de domínio, dataclass frozen)
├── model_store.py      ← ModelStore (fachada sobre MLflow client)
├── lineage.py          ← capture_lineage() — git_hash, gen_version, dataset_hash
└── stages.py           ← PromotionStage enum + transições válidas
```

### 2.2 `ModelVersion` — entidade de domínio

Composição (não herança) sobre o `manifest` existente. O manifest (`evaluation/manifest.py:105`) já serializa 121 campos de config — `ModelVersion` o **embute** e adiciona o que falta:

```
┌──────────────────────────────────────────────────────────────────────┐
│  ModelVersion (frozen dataclass)                                       │
├──────────────────────────────────────────────────────────────────────┤
│  name              str       "geosteering-resnet18-p1"                 │
│  version           int       N (auto-incremento MLflow)                │
│  stage             PromotionStage  STAGING|PRODUCTION|ARCHIVED|PREVIEW │
│  ── LINEAGE (4-tupla de reprodutibilidade) ─────────────────────────── │
│  git_hash          str       git rev-parse HEAD (auto, NÃO manual)     │
│  generator_version str       SyntheticDataGenerator.GENERATOR_VERSION  │
│  seed              int       data/synthetic_generator.py:481           │
│  dataset_hash      str       sha256 do .npz / dataset card             │
│  ── CONFIG ─────────────────────────────────────────────────────────── │
│  config_sha256     str       sha256(config.to_dict()) — chave dedup    │
│  manifest          dict      create_manifest() inteiro (121 campos)    │
│  ── ARTEFATOS ─────────────────────────────────────────────────────── │
│  savedmodel_uri    str       mlflow artifact path                      │
│  onnx_uri          str|None  runtime batch                             │
│  tflite_int8_uri   str|None  runtime realtime                          │
│  scaler_uri        str       ScalerRegistry persistido (clean-fit)     │
│  ── GATES ─────────────────────────────────────────────────────────── │
│  golden_passed     bool      tests/golden/                             │
│  int8_parity_db    float     desvio máx FP32 vs INT8                   │
│  coverage_p90      float     UQ calibrada                              │
│  field_validated   bool      Volve/Goliat → gate produto vs preview    │
└──────────────────────────────────────────────────────────────────────┘
```

| Atributo | Fonte hoje | Lacuna a fechar |
|:---------|:-----------|:----------------|
| `git_hash` | `manifest extra` manual (`report.py:492`) | **auto-capturar** via `subprocess git rev-parse HEAD` em `lineage.py` (hoje é string opcional do chamador) |
| `generator_version` | **inexistente** | adicionar `GENERATOR_VERSION="1.0.0"` em `synthetic_generator.py` + injetar em `metadata` (`:477`) |
| `seed` | `metadata["seed"]` (`:481`) | já existe — apenas propagar |
| `dataset_hash` | inexistente | sha256 do `.npz` gerado (ex.: `benchmarks/results/sprintc_surrogate_dataset.npz`) |
| `config_sha256` | inexistente | `hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True))` — dedup de runs idênticos |

### 2.3 Promotion stages

```
   ┌─────────┐  golden+int8+coverage   ┌────────────┐  field_validated?   ┌────────────┐
   │ STAGING │ ──────────────────────► │ candidate  │ ─── não ──────────► │ TECH       │
   └─────────┘  (G1..G4 pass)          │ (gate G5)  │                     │ PREVIEW    │
        ▲                              └─────┬──────┘ ─── sim (G5 pass) ─► │ PRODUCTION │
        │                                    │                            └─────┬──────┘
        │ novo treino                        │ falha gate                       │ drift/
        │                                    ▼                                  ▼ superseded
   (sempre entra)                       (bloqueado)                        ┌────────────┐
                                                                           │  ARCHIVED  │
                                                                           └────────────┘
```

| Stage | Quem promove | Pré-condição | Runtime servido |
|:------|:-------------|:-------------|:----------------|
| `STAGING` | automático pós-treino | manifest válido | nenhum (só registro) |
| `TECH PREVIEW` | mantenedor (manual) | G1–G4 ✓, **G5 incompleto** | ONNX (offline only), disclaimer obrigatório |
| `PRODUCTION` | mantenedor + ADR | G1–G5 ✓ | ONNX (batch) + LiteRT INT8 (realtime) |
| `ARCHIVED` | automático ou manual | superseded / drift sustentado | nenhum |

> **Regra-dura**: transição `→ PRODUCTION` exige ADR (`docs/decisions/ADR-XXXX.md`) e é a **única** que requer human-in-the-loop explícito (§6).

---

## 3. Experiment Tracking — `TrainingLoop → MLflow autolog`

### 3.1 Integração não-invasiva

`TrainingLoop.fit()` (`training/loop.py:694`) chama `model.fit(... callbacks)`. MLflow entra como **mais um callback Keras**, respeitando o padrão Factory existente (`build_callbacks`) — zero mudança no caminho crítico.

```
geosteering_ai/mlops/tracking/
├── autolog_callback.py   ← MLflowAutologCallback(keras.callbacks.Callback)
└── optuna_bridge.py      ← MLflowOptunaCallback (nested runs por trial)
```

| Hook Keras | Loga em MLflow |
|:-----------|:---------------|
| `on_train_begin` | `config.to_dict()` (121 params), `git_hash`, `seed`, `generator_version`, mixed-precision policy |
| `on_epoch_end` | `loss`, `val_loss`, `R2Score`, `PerComponentMetric` (rho_h/rho_v), `AnisotropyRatioError`, LR atual |
| `on_train_end` | `TrainingResult.best_epoch/best_val_loss` (`loop.py:120`), `training_time`, curvas (PNG via `visualization/`), SavedModel |

> Decisão: **callback custom em vez de `mlflow.tensorflow.autolog()`**. O autolog genérico não captura métricas geofísicas custom (`AnisotropyRatioError`) nem respeita o N-Stage merging de history (`loop.py:300`). O callback custom lê `self.history` acumulado.

### 3.2 Optuna HPO → nested runs

`training/optuna_hpo.py:163` (`create_search_space`) e `run_hpo` já existem (opt-in `config.use_optuna`). Integração: cada `trial` vira um **nested MLflow run** sob o run-pai do estudo. `study.best_params` → registra `ModelVersion` STAGING automaticamente.

```
   run-pai "hpo-resnet18-2026-06"   (sampler=TPE, pruner=Hyperband)
   ├── nested run trial-0  params={lr, dropout, kernel} → val_loss=0.061
   ├── nested run trial-1  ... → val_loss=0.058   ◄── best
   └── nested run trial-N  ...
        └─► best_params → ModelStore.register(stage=STAGING)
```

---

## 4. Golden-Tests de Modelo — estender paridade `<1e-12` a modelos treinados

### 4.1 Filosofia

A paridade Fortran `<1e-12` (simulador) e JAX-vs-Numba `<1e-10` são **golden-tests de física**. Estendemos a mesma disciplina a **modelos treinados**: a saída de inferência de uma versão registrada deve ser **determinística e reproduzível** sob input fixo.

```
tests/golden/
├── fixtures/
│   ├── golden_input_22col.npz       ← input canônico congelado (z_obs fixo)
│   └── golden_output_v{N}.npz        ← saída de referência por ModelVersion
├── test_inference_golden.py          ← G1: saída por versão
└── test_int8_parity.py               ← G2: paridade pré/pós quantização
```

### 4.2 Tabela de golden-tests

| Gate | Teste | Tolerância | Justificativa |
|:----:|:------|:-----------|:--------------|
| **G1** | saída SavedModel(vN) vs `golden_output_vN.npz` | `< 1e-6` (rtol) | float32 + não-determinismo cuDNN; trava regressão de export/scaler |
| **G2a** | ONNX Runtime vs SavedModel (mesma vN) | `< 1e-5` | conversão tf2onnx não pode alterar predição |
| **G2b** | **TFLite INT8 vs FP32** (paridade de quantização) | `< 0.5 dB` em atenuação derivada / `< 2%` RMSE em rho | **gate físico**: quantização não pode degradar inversão além do erro de medição LWD |
| **G3** | coverage P90 ≥ 0.88 (nominal 0.90) | tolerância −0.02 | UQ não pode estar mal-calibrada |

> **Decisão sobre tolerância G1**: `1e-6` (não `1e-12`). Modelos treinados em float32 + kernels cuDNN não-determinísticos não atingem `1e-12` — exigir isso seria fisicamente incorreto. A disciplina é mantida; o número reflete o regime float32 (consistente com `eps_tf=1e-12` ser piso, não alvo de inferência).

### 4.3 Fluxo de criação de golden

```
register(vN) ──► roda inferência em golden_input_22col.npz ──► salva golden_output_vN.npz
                                                                       │
   PR futuro que altere export/scaler/runtime ──► CI roda G1 ──► compara ──► FALHA se Δ>1e-6
```

---

## 5. Runtime de Produção — ONNX Runtime + TFLite/LiteRT INT8

### 5.1 Dois runtimes, dois alvos

| Runtime | Alvo | Consumidor | Latência | Quantização |
|:--------|:-----|:-----------|:---------|:------------|
| **ONNX Runtime** | batch / offline | API REST `/predict`, pós-poço | sem SLA rígido | FP32 (ou FP16 opt) |
| **TFLite/LiteRT INT8** | realtime causal | Studio dashboard, SM streaming | **< 100 ms/sample** | full-INT8 (gate G2b) |

### 5.2 Gate de quantização — fechar o gap de `export.py:216`

`inference/export.py:182` faz só **dynamic-range** (pesos INT8, ativações float32) e declara full-INT8 não-implementado (`:216-218`). Proposta concreta:

```
geosteering_ai/mlops/runtime/
├── quantize.py     ← export_tflite_int8(model, representative_dataset, config)
├── ort_session.py  ← ONNXRuntimeSession (wrapper batch)
└── litert_session.py ← LiteRTSession (wrapper realtime <100ms)
```

```
┌─────────────────────────────────────────────────────────────────────┐
│  PIPELINE DE QUANTIZAÇÃO COM GATE                                     │
├─────────────────────────────────────────────────────────────────────┤
│  1. representative_dataset ← 100–500 amostras CLEAN do val set       │
│     (NUNCA ruidoso — mesma regra do scaler fit-on-clean)             │
│  2. TFLiteConverter.optimizations=[DEFAULT]                          │
│     converter.representative_dataset = rep_gen                       │
│     target_spec.supported_ops = [TFLITE_BUILTINS_INT8]              │
│  3. GATE G2b: roda FP32 e INT8 no golden_input                       │
│       Δrho_RMSE < 2%  E  Δatenuação < 0.5 dB ?                       │
│         ├─ SIM → tflite_int8_uri registrado em ModelVersion          │
│         └─ NÃO → promoção BLOQUEADA, log do desvio por componente    │
└─────────────────────────────────────────────────────────────────────┘
```

> Decisão: `representative_dataset` extraído de dados **limpos** do val set, alinhado à regra inviolável "scaler fit em dados LIMPOS". Quantizar com ativações vistas em dados ruidosos enviesaria os ranges INT8 para o regime de ruído.

---

## 6. Monitoramento em Produção

### 6.1 Drift sobre o sinal físico (Hxx/Hzz)

```
geosteering_ai/mlops/monitoring/
├── drift.py        ← KS / PSI / Wasserstein sobre features de entrada
├── model_card.py   ← ModelCard (ranges físicos válidos + extrapolação)
├── audit.py        ← AuditTrail (log assinado de inferências)
└── alerts.py       ← thresholds + canais (log estruturado + arquivo)
```

| Métrica | Aplicada a | Threshold de alerta | Ação |
|:--------|:-----------|:--------------------|:-----|
| **KS** (Kolmogorov-Smirnov) | dist. de `Re/Im Hxx`, `Re/Im Hzz` vs treino | D > 0.1 (p<0.01) | warning |
| **PSI** (Population Stability Index) | mesmas features, binned | PSI > 0.2 | warning / PSI > 0.25 retrain trigger |
| **Wasserstein** | magnitude de Hzz (4 ordens de grandeza) | W normalizado > 0.15 | warning |
| **Extrapolação** | `freq`, `spacing`, `rho` fora do model card | qualquer fora-range | bloqueio + disclaimer reforçado |

> Decisão: drift medido **sobre o tensor EM de entrada (Hxx/Hzz)**, não sobre a saída (rho). O sinal EM é a fronteira física entre sintético e campo real — é onde o domain shift sintético→real se manifesta primeiro. Reusa estatística já presente em `visualization/uncertainty.py:649` (coverage online).

### 6.2 Model Card — limites físicos válidos

| Campo do Model Card | Fonte | Exemplo |
|:--------------------|:------|:--------|
| Ranges de treino | errata física imutável | `freq ∈ [100, 1e6] Hz`, `spacing ∈ [0.1, 10] m`, `seq_len ∈ [10, 1e5]` |
| Range de `rho` treinado | dataset card | `0.5–2000 Ω·m` (extrapolação além = preview only) |
| Anisotropia TIV coberta | dataset card | `λ = ρv/ρh ∈ [1, 10]` |
| Validação de campo | gate G5 | "Volve público: R²=0.xx; Goliat: pendente" |
| Limitações declaradas | manual | "não validado em formações carbonáticas; multi-freq testado em 3 freqs" |

### 6.3 Trilha de auditoria

```
┌────────────────────────────────────────────────────────────────┐
│  AuditTrail — append-only, 1 registro por inferência            │
├────────────────────────────────────────────────────────────────┤
│  timestamp · model_version (N) · model_stage · config_sha256    │
│  input_hash · output (rho_h,rho_v,DTB) · uq (P10/P50/P90)        │
│  drift_flags · extrapolation_flag · operator_id · disclaimer_ack │
└────────────────────────────────────────────────────────────────┘
   → JSONL assinado (sha256 encadeado) — reconstrói QUALQUER decisão
```

---

## 7. Governança (domínio crítico/regulado)

| Regra de governança | Implementação concreta | Status |
|:--------------------|:-----------------------|:------:|
| Promoção a PRODUCTION exige ADR | `docs/decisions/ADR-XXXX.md` + hook valida link no commit de promoção | planejado |
| Human-in-the-loop em geosteering | API/Studio retornam **disclaimer obrigatório** + `disclaimer_ack` na auditoria; decisão de poço **nunca** automática | planejado |
| Reprodutibilidade total | `ModelVersion` 4-tupla (git+gen+seed+dataset) → `re-train` produz `config_sha256` idêntico | parcial → planejado |
| Tech Preview vs Production | gate G5 (validação de campo) separa os dois; preview só serve ONNX offline | planejado |
| Rastreabilidade de dados | dataset card com `generator_version` + `seed` + `dataset_hash` | ausente → planejado |
| Sem PyTorch em produção | runtime usa ONNX/LiteRT exportados de **Keras**; adapter isolado nunca entra no registry | implementado (regra) |

> **Disclaimer padrão (human-in-the-loop)**: toda resposta de inferência em estágio PRODUCTION ou PREVIEW carrega: *"Saída de modelo de inversão 1D — apoio à decisão, não substitui interpretação geofísica. Validar contra LWD/sísmica antes de decisão de trajetória."* Em PREVIEW, acrescenta: *"Modelo NÃO validado em campo real."*

---

## 8. Validação Científica — gate "produto" vs "tech preview"

### 8.1 Gate G5 — validação de campo real

```
┌────────────────────────────────────────────────────────────────────┐
│  PLANO DE VALIDAÇÃO DE CAMPO (G5)                                    │
├────────────────────────────────────────────────────────────────────┤
│  Fase A — Volve (dataset público Equinor)                           │
│    ├ ingest LAS/DLIS/WITSML → 22-col (depende geosteering_ai/ingest/)│
│    ├ inferência batch via ONNX Runtime                              │
│    └ métrica: R²(rho_h), RMSE, DTB error vs interpretação publicada  │
│  Fase B — Goliat-like / poço operacional (sob NDA)                  │
│    └ mesmo protocolo, dataset proprietário                          │
│                                                                     │
│  CRITÉRIO DE GATE:                                                  │
│    G5 ✓  (R² ≥ limiar acordado em ADR)  →  elegível a PRODUCTION    │
│    G5 ✗  ou ausente                      →  no máximo TECH PREVIEW   │
└────────────────────────────────────────────────────────────────────┘
```

> Dependência dura: G5 depende de `geosteering_ai/ingest/` (LAS/DLIS/WITSML), hoje **AUSENTE** — é o **gap #1 de credibilidade**. Sem ingest, G5 é inalcançável e **todo modelo fica travado em TECH PREVIEW**. Isto deve ser explícito no ROADMAP §0.

### 8.2 Calibração de UQ — fechar gap de `uncertainty.py`

`inference/uncertainty.py` produz mean/std/CI95 (MC Dropout/Ensemble/INN) mas **não** P10/P50/P90 calibrados, CRPS, nem coverage-test programático (só plot em `visualization/uncertainty.py:649`). Proposta:

```
geosteering_ai/mlops/uq_calibration/
├── quantiles.py    ← P10/P50/P90 a partir de amostras MC/ensemble
├── crps.py         ← CRPS (Continuous Ranked Probability Score)
└── coverage.py     ← coverage test programático (gate G3)
```

| Métrica UQ | Definição operacional | Gate |
|:-----------|:----------------------|:----:|
| **Coverage P90** | fração de y_true dentro de [P5, P95] | ≥ 0.88 (G3) |
| **CRPS** | erro probabilístico (menor = melhor) | reportado no model card |
| **Calibração** | curva observado vs nominal (`uncertainty.py:649`) | desvio máx < 0.05 |

---

## 9. Dependências e impacto no ambiente

| Dependência nova | Versão | Justificativa | Já em pyproject? |
|:-----------------|:-------|:--------------|:----------------:|
| `mlflow` | `>=2.10` | registry + tracking on-prem (A6000) | **não** → adicionar `[mlops]` |
| `onnxruntime-gpu` | `>=1.17` | runtime batch (A6000 CUDA12) | não → adicionar |
| `tf2onnx` | (já lazy em `export.py:312`) | export ONNX | declarar em `[mlops]` |
| `scipy.stats` | (transitivo) | KS / Wasserstein | já presente |
| `optuna` | `>=3.0` | HPO | **sim** (`pyproject:42`) |
| `joblib` | — | persistência de scaler (`scaling.py:153`) | verificar |

> Decisão: novo extra `[mlops]` no `pyproject.toml`, separado de `[all]`, para manter o dev CPU-only leve (lazy imports, padrão já usado em `export.py:312` e `uncertainty.py`).

---

## 10. Roteiro de implementação (ordem de dependência)

```
   FASE 1 (fundação)          FASE 2 (gates)            FASE 3 (produção)
   ┌──────────────────┐       ┌──────────────────┐      ┌──────────────────┐
   │ lineage auto-capt │  ──►  │ golden-tests G1  │ ──►  │ runtime ONNX+ORT  │
   │ GENERATOR_VERSION │       │ INT8 + gate G2b  │      │ drift+model card  │
   │ ModelVersion+Store│       │ UQ calib + G3    │      │ audit trail       │
   │ MLflow autolog    │       │                  │      │ governança/ADR    │
   └──────────────────┘       └──────────────────┘      └──────────────────┘
                                                                  │
                                          (bloqueado por ingest)  ▼
                                                          ┌──────────────────┐
                                                          │ G5 Volve/Goliat   │
                                                          │ produto≠preview   │
                                                          └──────────────────┘
```

| Fase | Entregável mínimo | Desbloqueia |
|:----:|:------------------|:------------|
| 1 | `mlops/registry/` + autolog callback + `GENERATOR_VERSION` | rastreabilidade real (hoje git/seed manuais) |
| 2 | `tests/golden/` + full-INT8 + UQ calibrada | promoção STAGING→PREVIEW automatizada |
| 3 | runtimes empacotados + monitoramento + governança | servir em produção com auditoria |
| 4 | `ingest/` + G5 | **único caminho para PRODUCTION** (sai de preview) |
