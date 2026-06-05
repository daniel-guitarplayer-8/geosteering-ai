> **Blueprint de Arquitetura do Geosteering AI — Data Flow.** Índice: [README.md](README.md) · Constituição SDD: [../../specs/CONSTITUTION.md](../../specs/CONSTITUTION.md) · Roadmap: [../../specs/ROADMAP.md](../../specs/ROADMAP.md). Gerado 2026-06-05 (workflow multi-agente + revisão crítica).

# Data Flow Completo — Geosteering AI

> Esta seção projeta o fluxo de dados ponta-a-ponta dos 4 produtos. Cada fluxo cita
> `arquivo:linha` reais quando o componente já existe e marca **status**:
> `implementado` | `parcial` | `ausente` | `planejado`. A âncora física é o
> **contrato 22-col** (`geosteering_ai/simulation/io/binary_dat.py:103` —
> `DTYPE_22COL`), imutável, que une simulação, treino, inferência e ingestão.

---

## 0. Mapa de Produtos × Estágios do Data Flow

Onde cada um dos 4 produtos *entra* no fluxo de dados (consome/produz):

```
┌──────────────────────┬───────────┬──────────┬────────────┬───────────────┐
│  Estágio do fluxo    │ Biblioteca│   CLI    │   Studio   │ Simulation Mgr│
│                      │  + API    │          │ (flagship) │  (só sim.)    │
├──────────────────────┼───────────┼──────────┼────────────┼───────────────┤
│ 1 Gerar dataset      │  produz   │ produz   │  produz    │  produz       │
│ 2 Treino DL          │  produz   │  ──(*)   │  produz    │   ──          │
│ 3 Inferência offline │ produz/srv│  ──(*)   │  produz    │   ──          │
│ 4 Geonav. realtime   │  srv(API) │   ──     │  produz    │   ──          │
│ 5 Ingestão real      │ biblioteca│   ──     │  consome   │   ──          │
└──────────────────────┴───────────┴──────────┴────────────┴───────────────┘
(*) CLI hoje só cobre simulate/benchmark/version; train/predict CLI = planejado.
srv = serve via REST FastAPI (geosteering_ai/api/).
```

**Fundação compartilhada SM ↔ Studio** (reconciliação): ambos consomem
`geosteering_ai/gui/` (**ausente** — a extrair de `simulation/tests/sm_*.py`) +
`geosteering_ai/simulation/`. SM = `gui/ + simulation/` apenas. Studio =
`gui/ + biblioteca inteira`. A física **nunca** é duplicada — todos convergem em
`geosteering_ai/simulation/multi_forward.py`.

---

## CONTRATOS DE DADOS (definição canônica)

Antes dos fluxos, os 5 contratos que os atravessam.

### C1. Registro 22-col (`DTYPE_22COL`) — IMUTÁVEL

Fonte de verdade: `geosteering_ai/simulation/io/binary_dat.py:103-119`.
172 bytes/registro = 1×int32 + 21×float64.

```
┌─────┬──────────────┬──────────┬──────────────────────────────────────────┐
│ Col │ Campo        │ dtype    │ Significado físico                        │
├─────┼──────────────┼──────────┼──────────────────────────────────────────┤
│  0  │ i            │ int32    │ índice da medição (posição ao longo MD)   │
│  1  │ z_obs        │ float64  │ profundidade observada (m) — NUNCA escala │
│  2  │ rho_h        │ float64  │ resistividade horizontal (Ω·m) — target   │
│  3  │ rho_v        │ float64  │ resistividade vertical (Ω·m) — target     │
│ 4-5 │ Re/Im Hxx    │ float64  │ tensor EM linha 1                         │
│ 6-7 │ Re/Im Hxy    │ float64  │                                           │
│ 8-9 │ Re/Im Hxz    │ float64  │                                           │
│10-11│ Re/Im Hyx    │ float64  │ tensor EM linha 2                         │
│12-13│ Re/Im Hyy    │ float64  │                                           │
│14-15│ Re/Im Hyz    │ float64  │                                           │
│16-17│ Re/Im Hzx    │ float64  │ tensor EM linha 3                         │
│18-19│ Re/Im Hzy    │ float64  │                                           │
│20-21│ Re/Im Hzz    │ float64  │ (Hzz axial)                               │
└─────┴──────────────┴──────────┴──────────────────────────────────────────┘
```

Contratos derivados (errata imutável, validada em `data/loading.py:107`):
`INPUT_FEATURES = [1, 4, 5, 20, 21]` (z_obs, Re/Im Hxx, Re/Im Hzz) ·
`OUTPUT_TARGETS = [2, 3]` (rho_h, rho_v). `COL_MAP_22` em `data/loading.py:119`.

Status: **implementado** (escrita `io/tensor_dat.py:129`, leitura `data/loading.py:359`).

### C2. Manifest JSON (lineage de dataset/modelo)

Hoje existe `safe_json_dump`/`NumpyEncoder` (`utils/io.py:118,45`) e
`SimulationConfig.to_dict/from_dict` (`simulation/config.py:1151,1166`), mas
**não há `manifest.py` consolidando dataset+config+git+seed**. Proposto
(**planejado** `geosteering_ai/data/manifest.py` ou `utils/manifest.py`):

```jsonc
{
  "schema_version": "1.0",
  "kind": "dataset" | "model",
  "created_utc": "2026-06-05T12:00:00Z",
  "git_hash": "5672804",          // dirty flag se working tree suja
  "seed": 42,
  "sim_config": { /* SimulationConfig.to_dict() */ },
  "pipeline_config": { /* PipelineConfig (gap survey: serialização JSON ausente) */ },
  "dataset": {
    "format": "dat_22col" | "hdf5",
    "path": "datasets/scenarioH_512.dat",
    "n_models": 2000, "n_positions": 600, "n_freqs": 3,
    "sha256": "…", "bytes": 1234567
  },
  "split": { "strategy": "by_model[P1]", "train": 1600, "val": 200, "test": 200 },
  "backend": { "name": "jax_gpu" | "numba", "reason": "auto: …",
               "parity_fortran": "<1e-12", "parity_jax_numba": "<1e-10" }
}
```

### C3. `.gsproj` (projeto Studio) e `.session` (Simulation Manager)

```
┌─────────────┬──────────────────┬──────────────────────────────────────────┐
│ Artefato    │ Produto          │ Conteúdo                                  │
├─────────────┼──────────────────┼──────────────────────────────────────────┤
│ .session    │ Simulation Mgr   │ SimRequest(s) + estado GUI + refs .dat    │
│  (zip/json) │  (planejado)     │ + snapshot perf. Persist: sm_snapshot_    │
│             │                  │ persist.py (base existente).              │
│ .gsproj     │ Studio (flagship)│ Superset: refs dataset(manifest) +        │
│  (zip)      │  (planejado)     │ ModelVersion + trajetória + camadas de    │
│             │                  │ interpretação + .session embutível.       │
└─────────────┴──────────────────┴──────────────────────────────────────────┘
```
Regra: `.gsproj` ⊇ `.session` (Studio abre `.session` do SM; SM não abre `.gsproj`).
Ambos guardam **referências** a `.dat`/`.hdf5` por path+sha256 (manifest), nunca
embutem o tensor bruto. Status: **planejado** (base `sm_snapshot_persist.py`,
`sm_io.py`).

### C4. Trajetória (MD ↔ TVD)

Contrato para Fases 2 (realtime/ingestão). **ausente** (`geosteering_ai/ingest/`
inexistente). Proposto — array estruturado + survey:

```
trajectory := { md[m], inc[deg], azi[deg], tvd[m], ns[m], ew[m] }   # minimum-curvature
mapping:  z_obs (22-col) ↔ tvd  via interpolação por MD
```

### C5. ModelVersion (registry de modelos)

**ausente** (MLflow/registry gap). Hoje só `InferencePipeline.save/load`
(`inference/pipeline.py:402,459`) serializa modelo+scalers. Proposto wrapper de
lineage que liga `manifest(model)` → SavedModel/TFLite/ONNX +
métricas + dataset-pai.

---

## FLUXO 1 — Geração de Dataset Sintético

Status global: **implementado** (caminho `.dat`); **parcial** (HDF5, manifest).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SimulationConfig          PipelineConfig (sampling rho/dip/TR/freq)           │
│  config.py:265             synthetic_generator.py:154                          │
│      │ frequencies_hz=[20k,100k,400k]  (multi-freq JÁ implementada)            │
│      ▼                                                                          │
│  SyntheticDataGenerator.generate_batch()   synthetic_generator.py:196          │
│      │  amostra N modelos geológicos (camadas, rho_h, rho_v, dip)              │
│      ▼                                                                          │
│  simulate_batch(models, backend="auto")    dispatch.py:233                     │
│      │   _resolve_backend (dispatch.py:94) — árvore medida:                    │
│      │   ┌───────────────────────────────────────────────────────┐            │
│      │   │ sem GPU JAX ........→ numba (16w×4t, prange)           │            │
│      │   │ n_models < thr .....→ numba (GPU subocupada)           │            │
│      │   │ geometria não-agrup→ numba                             │            │
│      │   │ GPU + n≥thr + agrup→ jax_gpu (grouped/bucketed vmap)   │            │
│      │   └───────────────────────────────────────────────────────┘            │
│      ▼                                                                          │
│  multi_forward.py  (FÍSICA ÚNICA — paridade Fortran <1e-12)                    │
│      │   H_tensor: complex128 shape (n_models, n_meds, n_cfg, 9)               │
│      ▼                                                                          │
│  write_dat_from_tensor()    io/tensor_dat.py:129  (VETORIZADO, bit-exato)     │
│      │   → .dat 22-col binário (DTYPE_22COL, 172 B/rec)                        │
│      │     [multi-config: .dat cobre só TR₀/dip₀/freq₀ — aviso gen:382]        │
│      ▼                                                                          │
│  manifest.json  (PLANEJADO — git_hash + seed + sim_config + sha256)           │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Etapa | Componente | `arquivo:linha` | Status |
|:--|:--|:--|:--|
| Config sim | `SimulationConfig` | `simulation/config.py:265` | implementado |
| Multi-freq | `frequencies_hz` | `simulation/config.py:373` | implementado |
| Amostragem | `SyntheticDataGenerator` | `data/synthetic_generator.py:154` | implementado |
| Dispatch auto | `simulate_batch` / `_resolve_backend` | `simulation/dispatch.py:233,94` | implementado |
| Física | `multi_forward` | `simulation/multi_forward.py` | implementado (<1e-12) |
| Escrita `.dat` | `write_dat_from_tensor` | `simulation/io/tensor_dat.py:129` | implementado |
| Escrita HDF5 | — | (a criar `io/hdf5.py`) | **ausente** |
| Manifest | `create_manifest` | (a criar) | **planejado** |
| On-the-fly (sem disco) | `generate_batch(build_dat_22col=False)` | `data/synthetic_generator.py:214` | implementado |

**Decisão acionável:** adicionar `io/hdf5.py` (datasets multi-config completos, não
só TR₀/dip₀/freq₀ — resolve o aviso de `synthetic_generator.py:382`) e
`manifest.py`. O `.dat` permanece como formato de paridade/legado Fortran; HDF5
vira formato primário para datasets multi-config + multi-freq.

**Entradas de produto:** SM gera via `sm_model_gen.py`+`sm_workers.py` → `.dat` +
`.session`. CLI via `geosteering-cli simulate` (`cli/simulate.py`). Studio via
perspectiva de Simulação (reimplementa MVVM sobre `simulate_batch`).

---

## FLUXO 2 — Treino DL (on-the-fly noise→FV→GS→scale)

Status global: **parcial** — pipeline e treino prontos; lineage MLflow **ausente**.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  dataset .dat/.hdf5 + manifest                                                 │
│      │                                                                          │
│      ▼  load_binary_dat (data/loading.py:359) → array (rows, 22) raw          │
│      ▼  apply_decoupling (ACp/ACx)  → raw decoupled                            │
│      ▼  split_by_model [P1] (splitting.py)  — ZERO leakage (split por modelo)  │
│      │                                                                          │
│      ▼  fit_scaler em DADOS LIMPOS (FV+GS clean, temporário)   ◀── REGRA        │
│      │     (scaler NUNCA fitado em ruído; val/test transformados offline)      │
│      ▼                                                                          │
│  DataPipeline (data/pipeline.py) — tf.data.map ON-THE-FLY:                     │
│   ┌──────────────────────────────────────────────────────────────┐            │
│   │ train_raw → noise(A/m) → FV_tf(noisy) → GS_tf(noisy) → scale  │            │
│   │   z_obs (col 1) protegido (nunca recebe ruído nem escala)     │            │
│   │   GS veem ruído ✓ (fidelidade LWD). 34 tipos + curriculum 3-φ │            │
│   └──────────────────────────────────────────────────────────────┘            │
│      ▼                                                                          │
│  ModelRegistry().build(config)   (48 arquiteturas; 26 causal-compat)          │
│  LossFactory.get / build_combined (26 losses + 8 PINNs)                        │
│      ▼                                                                          │
│  TrainingLoop: compile → fit → causal finetuning (10 ep. fixo — gap)          │
│      ▼                                                                          │
│  MLflow lineage (PLANEJADO):  params + metrics + dataset(manifest) + git      │
│      ▼                                                                          │
│  ModelVersion → InferencePipeline.save (pipeline.py:402) [SavedModel+scalers]  │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Etapa | Componente | Status |
|:--|:--|:--|
| Load/decoupling | `data/loading.py:359` + `apply_decoupling` | implementado |
| Split por modelo P1 | `data/splitting.py` | implementado |
| Scaler em clean | `data/scaling.py` (per-group: StandardScaler EM + RobustScaler GS) | implementado |
| Pipeline on-the-fly | `data/pipeline.py` (`tf.data.map`) | implementado |
| Noise 34 tipos + curriculum | `noise/` | implementado |
| Modelos / Losses | `models/` `ModelRegistry`, `losses/` `LossFactory` | implementado |
| TrainingLoop + N-Stage | `training/` | implementado |
| Causal finetuning | `training/` (epochs hardcoded=10) | parcial (gap: não-configurável) |
| Serialização PipelineConfig→manifest | — | **ausente** (gap survey data) |
| MLflow / registry | — | **ausente** |

**Decisões acionáveis:** (1) serializar `PipelineConfig` em JSON no manifest
(resolve gap "rastreabilidade"); (2) `training/lineage.py` opcional que loga em
MLflow (`mlflow.tensorflow.autolog`) ligando run → `manifest(dataset)` → git_hash;
(3) tornar `causal_finetune_epochs` um campo de `PipelineConfig`.

**Entradas de produto:** Biblioteca (`from geosteering_ai.training import
TrainingLoop`). Studio: wizard "Train" sobre os mesmos objetos (Expert Mode expõe
Optuna). CLI: subcomando `train` **planejado**. SM **não** treina.

---

## FLUXO 3 — Inferência Offline (perfil 22-col → relatório)

Status global: **parcial** — predict+UQ prontos; UQ calibrada P10/P50/P90 **ausente**.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  modelo (ModelVersion / SavedModel)   +   perfil 22-col (.dat ou array)        │
│      │                                                                          │
│      ▼  InferencePipeline.predict()   inference/pipeline.py:193                │
│   ┌──────────────────────────────────────────────────────────────┐            │
│   │ raw 22-col → FV → GS → scale → model.predict                  │            │
│   │            → inverse_target_scaling (10^y' : log10→Ω·m)       │            │
│   │ args: theta=, freq=, return_uncertainty=, n_mc=30            │            │
│   └──────────────────────────────────────────────────────────────┘            │
│      ▼                                                                          │
│  UncertaintyEstimator (inference/uncertainty.py:217)                          │
│   ┌──────────────────────────────────────────────────────────────┐            │
│   │ mc_dropout (286) | ensemble (390) | inn (489)                 │            │
│   │ → UncertaintyResult: mean, std, CI95 (uncertainty.py:139)     │            │
│   │ [P10/P50/P90 calibrado + CRPS/coverage — AUSENTE]             │            │
│   └──────────────────────────────────────────────────────────────┘            │
│      ▼                                                                          │
│  Pós-proc geofísico:                                                            │
│   • DTB labels [P5]  (data/boundaries.py)                                       │
│   • curtain 2D ρ_h/ρ_v + bandas incerteza (visualization/geosteering)         │
│   • Picasso DOD (visualization/picasso)                                         │
│      ▼                                                                          │
│  Relatório Markdown (evaluation/report) + figuras (PDF/PNG/SVG)                │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Etapa | Componente | `arquivo:linha` | Status |
|:--|:--|:--|:--|
| Predict + inverse scaling | `InferencePipeline.predict` | `inference/pipeline.py:193` | implementado |
| UQ MC Dropout/Ensemble/INN | `UncertaintyEstimator` | `inference/uncertainty.py:217-607` | implementado |
| UQ calibrada P10/50/90 + CRPS | — | — | **ausente** (gap credibilidade) |
| DTB | `data/boundaries.py` | — | implementado |
| Curtain / DOD | `visualization/{geosteering,picasso}` | — | implementado |
| Relatório | `evaluation/report` | — | implementado |
| Validação campo real | — | — | **ausente** (gap #1) |

**Decisões acionáveis:** estender `UncertaintyResult` (`uncertainty.py:139`) com
quantis calibrados (conformal/quantile-calibration) + métricas CRPS/coverage;
adicionar `evaluation/calibration.py`. Serve também via `POST /predict`
(`api/routes/predict.py:89`, **sem auth** no MVP).

**Entradas de produto:** Biblioteca/API (REST). Studio: perspectiva de
Interpretação (curtain interativa). CLI `predict` **planejado**. SM **não** infere.

---

## FLUXO 4 — Geonavegação Realtime (Fase 2)

Status global: **parcial** — motor causal pronto; bridge WITSML/steering/export **ausente**.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  RIG  ──WITSML/ETP (streaming)──►  IngestService          [AUSENTE]            │
│                                    geosteering_ai/ingest/  (a criar)           │
│      │  PWLS/tool-channels → normaliza → 22-col (C1)                           │
│      │  trajetória: MD ↔ TVD (minimum-curvature; z_obs↔tvd)  (C4)             │
│      ▼                                                                          │
│  RealtimeInference.update(measurement)   inference/realtime.py:183             │
│   ┌──────────────────────────────────────────────────────────────┐            │
│   │ buffer circular deque(maxlen=window_size=seq_len) (rt:170)    │            │
│   │ retorna None até buffer cheio; depois 1 predição/amostra      │            │
│   │ modelo causal-compatible (TCN/WaveNet/Mamba) <31ms            │            │
│   └──────────────────────────────────────────────────────────────┘            │
│      ▼                                                                          │
│  UQ on-arrival (mc_dropout)  → CI95 por amostra                               │
│      ▼                                                                          │
│  DTB / steering decision engine          [AUSENTE]                            │
│      ▼                                                                          │
│  Plot 60 fps (RealtimeMonitor, visualization/realtime) + dashboard            │
│      ▼                                                                          │
│  Export OSDU / WITSML-out                 [AUSENTE]                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Etapa | Componente | Status |
|:--|:--|:--|
| Bridge WITSML/ETP in | `ingest/witsml_client.py` | **ausente** |
| Normalização → 22-col | `ingest/pwls_to_22col.py` | **ausente** |
| MD↔TVD | `ingest/trajectory.py` | **ausente** |
| Inferência incremental | `RealtimeInference` (`inference/realtime.py:102`) | implementado |
| UQ on-arrival | `UncertaintyEstimator` | implementado |
| Steering engine | — | **ausente** |
| Plot 60fps | `visualization/realtime` (`RealtimeMonitor`) | implementado |
| Export OSDU | `ingest/osdu_export.py` | **ausente** |

**Decisão acionável:** o motor (`realtime.py`) já entrega valor; o trabalho de
Fase 2 é o anel externo `ingest/` + `steering`. Studio é o **único** produto que
fecha o ciclo realtime (SM é só simulação; API serve `predict` mas não orquestra
o loop de perfuração).

---

## FLUXO 5 — Ingestão de Dado Real (Fase 2)

Status global: **ausente** (`geosteering_ai/ingest/` inexistente).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  LAS / DLIS / WITSML (arquivo ou stream)                                       │
│      │                                                                          │
│      ▼  Reader específico:                                                     │
│        LASReader  (lasio)   ·  DLISReader (dlisio)  ·  WITSMLClient            │
│      ▼  Normalização de canais → ordem/unidades 22-col (C1)                    │
│      ▼  errata fail-fast (mesmo gate do PipelineConfig.__post_init__):         │
│   ┌──────────────────────────────────────────────────────────────┐            │
│   │ assert 100 ≤ freq_hz ≤ 1e6 ; 0.1 ≤ spacing ≤ 10              │            │
│   │ assert TARGET_SCALING=="log10" ; INPUT_FEATURES==[1,4,5,20,21]│            │
│   │ valida espaçamento/freq do tool vs config → rejeita se incompat│            │
│   └──────────────────────────────────────────────────────────────┘            │
│      ▼  metadados LWD (tool rotation, borehole corr, mud filtrate) → manifest │
│      ▼  array 22-col  →  data/loading.py (mesmo path do sintético)            │
│      ▼  DomainAdapter (fine_tune synthetic→field) — já existe                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Etapa | Componente | Status |
|:--|:--|:--|
| LAS → 22-col | `ingest/las_reader.py` (lasio) | **ausente** |
| DLIS → 22-col | `ingest/dlis_reader.py` (dlisio) | **ausente** |
| WITSML → 22-col | `ingest/witsml_client.py` | **ausente** |
| Errata fail-fast | reutilizar `PipelineConfig.__post_init__` | implementado (regra; aplicar no reader) |
| Domain adaptation | `DomainAdapter.fine_tune` | implementado |
| Validação anisotropia TIV | `inspection.validate_tiv_consistency()` | **ausente** (gap) |

**Decisão acionável:** criar `geosteering_ai/ingest/` com um `BaseReader` que
**sempre** devolve `np.ndarray (rows, 22)` conforme C1 e roda errata fail-fast
antes de retornar — garante que dado real e sintético compartilhem 100% do
pipeline a jusante (Fluxos 2-3-4). Studio consome via wizard "Importar dados de
campo".

---

## Resumo de Gaps Priorizados (data flow)

```
┌──────────────────────────────────────┬──────────┬───────────────────────────┐
│ Gap                                  │ Bloqueia │ Artefato a criar          │
├──────────────────────────────────────┼──────────┼───────────────────────────┤
│ geosteering_ai/ingest/ (LAS/DLIS/    │ F4,F5    │ ingest/ + BaseReader      │
│  WITSML/ETP, MD↔TVD)                 │ Fase 2   │                           │
│ manifest.py (lineage dataset+modelo) │ F1,F2,F3 │ data/manifest.py          │
│ HDF5 multi-config completo           │ F1       │ simulation/io/hdf5.py     │
│ UQ calibrada P10/50/90 + CRPS        │ F3,F4    │ evaluation/calibration.py │
│ MLflow registry / ModelVersion       │ F2,F3    │ training/lineage.py       │
│ geosteering_ai/gui/ (fundação SM/Std)│ SM,Studio│ gui/ ← extrair sm_*.py    │
│ steering engine + OSDU export        │ F4       │ inference/steering.py     │
│ CLI train/predict + --backend auto   │ CLI      │ cli/{train,predict}.py    │
└──────────────────────────────────────┴──────────┴───────────────────────────┘
```
