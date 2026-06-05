# Índice de Specs — Geosteering AI

Tabela viva: cada spec ↔ `Backlog-Code` (ROADMAP §0) ↔ produto ↔ fase ↔ status. A **fila ordenada
por dependência** e as ondas (O0–O7) estão em [ROADMAP.md](ROADMAP.md).

**Legenda de produto:** LIB = Biblioteca/API · CLI = geosteering-cli · SM = Simulation Manager ·
STU = Studio (flagship) · PLAT = Plataforma/MLOps/CI.
**Status:** `planejado` → `em-spec` → `em-plan` → `em-impl` → `implementado`.

| Spec | Título | Produto(s) | Fase | Depende de | Status |
|:--|:--|:--|:--:|:--|:--|
| [0001](0001-sdd-bootstrap/) | Bootstrap SDD: `specs/` + Constituição (12 princípios) | PLAT | 0 | — | em-impl* |
| [0002](0002-version-ssot/) | SSoT de versão (`importlib.metadata`) + `py.typed` + **bump 2.0.0→2.56.0** | LIB | 0 | 0001 | planejado |
| [0003](0003-cli-backend-auto/) | CLI `--backend auto` (expõe `dispatch._resolve_backend`) | CLI | 0 | 0001 | **implementado** em `feat/cli-backend-auto` (GATE-V ✓ — testes/ruff/mypy verdes; aguarda revisão/merge) |
| [0004](0004-gui-foundation/) | Extração de `geosteering_ai/gui/` a partir de `sm_*.py` (Strangler Fig — `qt_compat`) | SM, STU | 0 | 0001 | **implementado** em `feat/gui-foundation` (GATE-V ✓ — GUI suite 16/16 xvfb, ruff/mypy verdes; aguarda revisão/merge) |
| [0005](0005-gui-mvvm-base/) | Fundação MVVM: `VMSignal` + `MainWindowBase` + `Perspective` ABC | SM, STU | 0 | 0004 | planejado |
| [0006](0006-gui-plot-backends/) | `gui/plot_backends`: `PlotCanvas` ABC + 4 backends + tokens de tema | SM, STU | 0 | 0004 | planejado |
| [0007](0007-gui-persistence/) | `gui/persistence`: `.session` atômico (write-temp→`os.replace`) | SM, STU | 0 | 0004 | planejado |
| [0008](0008-sdk-semver-tiers/) | SDK: 3 tiers de API + deprecação PEP 387 + `mypy --strict` no `__all__` | LIB | 1 | 0002 | planejado |
| [0009](0009-api-hardening/) | API `/v1` + X-API-Key + rate-limit + jobs 202 | LIB | 1 | 0002 | planejado |
| [0010](0010-cli-train-infer/) | CLI `train`/`infer` (reusa `TrainingLoop` + `InferencePipeline`) | CLI | 1 | 0003 | planejado |
| [0011](0011-sm-on-gui/) | SM app sobre `gui/` (casca + 1 perspectiva Simulação) | SM | 1 | 0005 | planejado |
| [0012](0012-sm-jax-gpu/) | SM JAX GPU via `simulate_batch(backend="auto")` | SM | 1 | 0011 | planejado |
| [0013](0013-studio-alpha/) | Studio ALPHA: casca MVVM + Perspectiva Simulação | STU | 1 | 0005 | planejado |
| [0014](0014-runtime-onnx-tflite/) | Runtime produção: ONNX + TFLite INT8 + golden parity (G1/G2b) | LIB | 2 | 0008 | planejado |
| [0015](0015-manifest-scaler-registry/) | `data/manifest.py` lineage + **`ScalerRegistry`** (dep dura) | LIB | 2 | 0008 | planejado |
| [0016](0016-studio-pers-treino/) | Studio Perspectiva Treino (ModelRegistry + LossFactory + curriculum) | STU | 2 | 0013 | planejado |
| [0017](0017-studio-pers-inferencia/) | Studio Perspectiva Inferência (UQ MC Dropout/Ensemble) | STU | 2 | 0013 | planejado |
| [0018](0018-gsproj/) | `.gsproj` projeto durável (zip+JSON, lineage DDD, **sem pickle**, `SecureArchiveReader`) | STU | 3 | 0007 | planejado |
| [0019](0019-ingest-las/) | `ingest/` BaseReader→22-col (LAS via lasio + errata fail-fast + MD↔TVD) | LIB | 3 | 0015 | planejado |
| [0019b](0019b-ingest-witsml/) | `ingest/` WITSML 2.0/ETP + WITS L0 + DLIS (dlisio) | LIB | 3 | 0019 | planejado |
| [0020](0020-uq-calibration/) | `evaluation/calibration.py`: P10/P50/P90 + CRPS + coverage (gate G3) | LIB, STU | 2 | 0017 | planejado |
| [0021](0021-studio-pers-realtime/) | Studio Perspectiva Realtime (curtain 60fps + bandas UQ) | STU | 3 | 0020 | planejado |
| [0022](0022-wellstate-manager/) | `WellStateManager` (histórico multi-poço durável) | STU | 3 | 0018 | planejado |
| [0024](0024-mlflow-modelstore/) | `registry/` ModelStore sobre MLflow on-prem + autolog | PLAT | 4 | 0015 | planejado |
| [0025](0025-installers-conda/) | Instaladores conda-constructor (Win/Linux/macOS) SM+Studio | PLAT | 4 | 0013 | planejado |
| [0026](0026-field-validation-g5/) | Validação de campo Volve (gate G5: PRODUCTION vs TECH PREVIEW) | LIB, STU | 4 | 0019 | planejado |

\* **0001 está sendo materializado por este bootstrap** (criação de `specs/`, Constituição,
templates, ROADMAP). A spec formal `0001-sdd-bootstrap/spec.md` pode ser preenchida
retroativamente para registro.

> A spec **0023** foi intencionalmente omitida pelo planejamento (numeração não-contígua é
> permitida; números nunca são reusados — Princípio IX).
