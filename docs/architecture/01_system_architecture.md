> **Blueprint de Arquitetura do Geosteering AI — Arquitetura do Sistema.** Índice: [README.md](README.md) · Constituição SDD: [../../specs/CONSTITUTION.md](../../specs/CONSTITUTION.md) · Roadmap: [../../specs/ROADMAP.md](../../specs/ROADMAP.md). Gerado 2026-06-05 (workflow multi-agente + revisão crítica).

## Arquitetura do Sistema — Geosteering AI (Visão de Plataforma, 4 Produtos)

### 1. Princípio Reitor

Um único pacote pip-installable (`geosteering_ai/`) é a **fundação física e de DL**. Sobre ela existem **duas camadas compartilhadas novas** (`gui/` Qt e `ingest/` de dados reais) e **quatro produtos** que consomem a fundação por composição, nunca por duplicação. A física de forward (`simulation/multi_forward.py`) é o **ponto de convergência único** — SM, Studio, CLI, API e Lib todos passam por ele. A paridade Fortran `<1e-12` é invariante de plataforma, não de produto.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRODUTOS (camada de aplicação)                        │
│                                                                               │
│  ┌────────────┐   ┌────────────┐   ┌────────────────┐   ┌──────────────────┐  │
│  │ P1 Lib+API │   │ P2 CLI     │   │ P3 Studio       │   │ P4 Simulation    │  │
│  │ (importável│   │ geosteering│   │ (FLAGSHIP)      │   │    Manager (SM)  │  │
│  │  + REST)   │   │   -cli     │   │  app desktop    │   │  GUI só-simul.   │  │
│  │ [parcial]  │   │ [parcial]  │   │  MVVM [ausente] │   │ [parcial 18,9kL] │  │
│  └─────┬──────┘   └─────┬──────┘   └────────┬───────┘   └────────┬─────────┘  │
│        │                │                   │                    │            │
│        │ importa        │ importa           │ importa            │ importa    │
│        │ lib            │ lib               │ lib INTEIRA        │ gui/+simul │
│        ▼                ▼                   ▼   + gui/            ▼  APENAS    │
└────────┼────────────────┼───────────────────┼────────────────────┼───────────┘
         │                │                   │                    │
┌────────┼────────────────┼───────────────────┼────────────────────┼───────────┐
│        ▼                ▼                   ▼                    ▼            │
│  ╔══════════════════════════════════════════════════════════════════════╗   │
│  ║   CAMADA GUI COMPARTILHADA — geosteering_ai/gui/   [NOVO/ausente]     ║   │
│  ║   qt_compat · plot_backends/ (PlotCanvas ABC) · models/ (VOs) ·        ║   │
│  ║   services/ (SimulationService, BenchmarkService) · widgets · style    ║   │
│  ║   Extraída de simulation/tests/sm_*.py (8,2k LOC modular)              ║   │
│  ╚══════════════════════════════════════════════════════════════════════╝   │
│        │                │                   │                    │            │
│  ╔══════════════════════════════════════════════════════════════════════╗   │
│  ║   CAMADA DE INGESTÃO — geosteering_ai/ingest/   [NOVO/ausente]        ║   │
│  ║   witsml/etp · wits_l0 · las (lasio) · dlis (dlisio) · pwls_map ·      ║   │
│  ║   trajectory (MD↔TVD min-curvature) → tensor 22-col (errata-checked)   ║   │
│  ╚══════════════════════════════════════════════════════════════════════╝   │
│        │                │                   │                    │            │
│  ╔══════════════════════════════════════════════════════════════════════╗   │
│  ║          BIBLIOTECA-NÚCLEO  geosteering_ai/   [~90% implementado]     ║   │
│  ║                                                                        ║   │
│  ║  config.py (PipelineConfig, 246 campos, errata __post_init__)          ║   │
│  ║  ┌──────────────┬──────────────┬──────────────┬──────────────────┐    ║   │
│  ║  │ simulation/  │ data/        │ models/(48)  │ inference/       │    ║   │
│  ║  │ _numba _jax  │ noise/       │ losses/(26+  │ pipeline realtime│    ║   │
│  ║  │ multi_forward│ FV·GS·scale  │   8 PINN)    │ uncertainty(UQ)  │    ║   │
│  ║  │ dispatch.py  │ ingest→22col │ training/    │ export(SM/TFLite/│    ║   │
│  ║  │ io/ filters/ │ evaluation/  │ registry     │   ONNX)          │    ║   │
│  ║  └──────────────┴──────────────┴──────────────┴──────────────────┘    ║   │
│  ║  registry/ (MLflow lineage) [NOVO] · visualization/ · utils/           ║   │
│  ╚══════════════════════════════════════════════════════════════════════╝   │
│        │                                                                      │
│  ╔══════════════════════════════════════════════════════════════════════╗   │
│  ║   PARIDADE-ORÁCULO (validação)  —  Fortran tatu.x / PerfilaAnisoOmp    ║   │
│  ║   <1e-12 float64 (Fortran) · <1e-10 (JAX-vs-Numba)  [implementado]    ║   │
│  ╚══════════════════════════════════════════════════════════════════════╝   │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Reconciliação SM ↔ Studio (a confusão a evitar):**

| Aspecto | Simulation Manager (P4) | Studio (P3, flagship) |
|:--------|:------------------------|:----------------------|
| Escopo de import | `gui/` + `simulation/` **apenas** | `gui/` + biblioteca **inteira** |
| Casca de aplicação | Reusa serviços de `gui/services/` direto | MVVM próprio (ViewModels sem import Qt) |
| Perspectiva de Simulação | É o produto inteiro | **Reimplementa** a perspectiva como uma aba entre várias |
| Esqueleto | `sm_*.py` modular (8,2k), **não** o monólito 10,7k | nasce limpo sobre `gui/` |
| Física | `multi_forward.py` | `multi_forward.py` (mesmo) |
| Dados reais | Não (só sintético/simul.) | Sim (`ingest/`, Fase 2) |

O monólito `geosteering_ai/simulation/tests/simulation_manager.py` (10.759 LOC) **não é o esqueleto** de nenhum produto. É código a ser estrangulado (Strangler Fig). A infra que vira `gui/` é a família `sm_*.py`.

---

### 2. Estrutura de Pacotes Proposta (monorepo, fronteiras explícitas)

```
geosteering-ai/                          (repo raiz — monorepo)
├── pyproject.toml                       extras: [sim][train][api][gui][ingest][studio][all]
├── geosteering_ai/                      ◄══ BIBLIOTECA-NÚCLEO (pip-installable)
│   ├── config.py                        [impl]   PipelineConfig — ponto único de verdade
│   ├── simulation/                      [impl]   Numba+JAX, dispatch.py auto, io/, filters/
│   │   ├── multi_forward.py             [impl]   CONVERGÊNCIA física de TODOS os produtos
│   │   ├── dispatch.py                  [impl]   backend="auto" (já existe na lib)
│   │   ├── _numba/ _jax/ _workers.py    [impl]
│   │   ├── io/ {binary_dat,model_in,    [impl]   22-col byte-exact Fortran
│   │   │        tensor_dat}.py
│   │   ├── validation/                  [impl]   compare_fortran, compare_empymod
│   │   └── tests/sm_*.py                [impl]   ◄── DOAR p/ gui/ (não é teste, é infra Qt)
│   ├── data/                            [impl]   loading,splitting,FV,GS,scaling,pipeline
│   ├── noise/                           [impl]   on-the-fly, curriculum 3-fase
│   ├── models/  (48 arq, 9 famílias)    [impl]   ModelRegistry, INN, SurrogateNet
│   ├── losses/  (26 + 8 PINN)           [impl]   LossFactory.build_combined()
│   ├── training/                        [impl]   TrainingLoop, NStage, callbacks, Optuna
│   ├── inference/                       [impl]   pipeline, realtime (deque), uncertainty, export
│   ├── evaluation/                      [impl]   metrics, dod, geosteering, report
│   │   └── calibration.py               [NOVO]   ◄── CRPS, coverage, reliability (gap UQ)
│   ├── visualization/                   [impl]   EDA, picasso, curtain, realtime monitor
│   ├── cli/                             [impl]   _main, simulate, benchmark, warmup, _backend
│   ├── api/                             [impl]   FastAPI: app, routes/{health,predict}, cli
│   │   ├── auth.py                      [NOVO]   ◄── X-API-Key / JWT (gap multi-tenant)
│   │   └── routes/simulate.py           [NOVO]   ◄── POST /simulate (forward via lib)
│   ├── ingest/                          ◄══ NOVO/ausente — conectores de dados reais
│   │   ├── witsml_etp.py                [NOVO]   cliente ETP/WebSocket (wss, JWT) 2.0
│   │   ├── wits_l0.py                   [NOVO]   cliente WITS L0 TCP/serial
│   │   ├── las_reader.py                [NOVO]   LAS 2.0/3.0 via lasio
│   │   ├── dlis_reader.py               [NOVO]   DLIS/RP66 via dlisio
│   │   ├── pwls_map.py                  [NOVO]   normalizador mnemônicos PWLS 3.0 → 22-col
│   │   └── trajectory.py                [NOVO]   MD↔TVD minimum-curvature (survey)
│   ├── registry/                        ◄══ NOVO/ausente — governança de modelos
│   │   ├── mlflow_store.py              [NOVO]   MLflow Tracking + Model Registry on-prem
│   │   ├── model_card.py                [NOVO]   ranges físicos válidos, limites extrapolação
│   │   └── drift.py                     [NOVO]   KS/PSI/Wasserstein sobre Hxx/Hzz
│   ├── gui/                             ◄══ NOVO/ausente — infra Qt COMPARTILHADA
│   │   ├── qt_compat.py                 [migrar] de sm_qt_compat.py (PyQt6+PySide6)
│   │   ├── plot_backends/               [migrar] de sm_plot_backends/ (PlotCanvas ABC)
│   │   ├── widgets.py / toast.py        [migrar] de sm_widgets.py, sm_toast.py
│   │   ├── models/                      [NOVO]   Value Objects DDD (SimRequest, WellSession)
│   │   ├── services/                    [NOVO]   SimulationService, BenchmarkService
│   │   └── persistence/                 [migrar] de sm_snapshot_persist.py (cache LRU)
│   └── utils/                           [impl]   logger, timer, validation
│
└── apps/                               ◄══ NOVO — produtos GUI (não pip-lib)
    ├── simulation_manager/              [parcial] P4 — gui/ + simulation/ APENAS
    │   ├── __main__.py                  entry: geosteering-sm
    │   ├── main_window.py               casca fina sobre gui/services
    │   └── pages/simulator_page.py      (estrangula simulation_manager.py monólito)
    └── studio/                          ◄══ NOVO/ausente (0% código) — P3 FLAGSHIP
        ├── __main__.py                  entry: geosteering-studio
        ├── viewmodels/                  [NOVO]   MVVM puro (testável sem import Qt)
        ├── views/                       [NOVO]   PyQt6 + LiquidGlass + PyQtGraph
        └── perspectives/                [NOVO]   Simulação · Treino · Inversão · Realtime
```

**Status por componente:**

| Componente | Status | Evidência |
|:-----------|:-------|:----------|
| `simulation/` (Numba+JAX+dispatch) | **implementado** | `simulation/dispatch.py`, paridade `validation/compare_fortran.py` |
| `data/ noise/ models/ losses/ training/ inference/` | **implementado** | digest dl-core/data: maturidade "pronto" |
| `cli/` (simulate/benchmark/warmup) | **parcial** | falta `--backend auto` (só `numba\|jax`), ver `cli/_backend.py` |
| `api/` REST | **parcial** | `api/routes/{health,predict}.py` MVP, sem auth, sem `/simulate` |
| `evaluation/calibration.py` (CRPS/coverage) | **ausente** | gap #4 UQ (digest dl-literature) |
| `gui/` | **ausente** | confirmado `NO geosteering_ai/gui/`; infra vive em `simulation/tests/sm_*.py` |
| `ingest/` | **ausente** | confirmado `NO geosteering_ai/ingest/`; zero refs WITSML/LAS/DLIS |
| `registry/` (MLflow) | **ausente** | sem lineage de artefatos de modelo |
| `apps/simulation_manager/` | **parcial** | monólito 10,7k + sm_*.py 8,2k em `simulation/tests/` |
| `apps/studio/` | **ausente** | 0% código |

---

### 3. Topologia de Deployment

```
┌──────────────── DESKTOP (workstation geofísico / data van) ─────────────────┐
│                                                                              │
│   conda-constructor → instalador nativo (.exe/.deb/.pkg) [rota napari]       │
│   ┌───────────────────────────┐      ┌───────────────────────────────────┐  │
│   │ geosteering-studio (P3)   │      │ geosteering-sm (P4)               │  │
│   │ MVVM · PyQtGraph 30fps ·  │      │ só simulação · Numba+JAX local    │  │
│   │ vispy p/ volume · A6000   │      │ benchmark A/B/30k                 │  │
│   └────────────┬──────────────┘      └─────────────┬─────────────────────┘  │
│                └──────────────┬─────────────────────┘                        │
│                               ▼ in-process import (sem rede)                  │
│                   geosteering_ai (lib) + JAX cuda12 (RTX A6000 48GB)          │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────── SERVING LOCAL (inferência congelada) ───────────────────────┐
│   ONNX Runtime  →  batch/offline cross-platform                              │
│   TFLite/LiteRT (INT8 + delegates) → realtime LWD <100ms                      │
│   golden-test paridade pré/pós-quantização INT8 (cultura <1e-12 estendida)   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────── API REST OPCIONAL (multi-poço / SaaS interno) ──────────────┐
│   Docker (Dockerfile.cpu, non-root, /health HEALTHCHECK)                      │
│   FastAPI (lazy TF) · /health <1ms · /predict <50ms · /simulate [NOVO]        │
│   + X-API-Key/JWT [NOVO] · rate-limit [NOVO] · MLflow registry [NOVO]         │
│   K8s manifests [planejado] · reverse proxy (TLS) responsável externo         │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────── EDGE NO RIG (Fase 2) ───────────────────────────────────────┐
│   edge-first: inferência local (JAX/Numba/TFLite) no rig/data van            │
│   ingest/ ETP (wss://443, JWT+TLS) ── consome WITSML 2.0 / WITS L0           │
│   envia SÓ predições + UQ ao centro remoto (IEC 62443 / Zero Trust)          │
│   destino corporativo: OSDU (WellLog + WellboreTrajectory via manifest)      │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Alvo | Empacotamento | Runtime inferência | Status |
|:-----|:--------------|:-------------------|:-------|
| Desktop Studio/SM | conda-constructor (Win/Linux/macOS) | in-process TF/JAX (A6000) | planejado (conda já existe) |
| Serving local | wheel + ONNX RT / TFLite | ONNX RT (batch) · TFLite INT8 (realtime) | parcial (export pronto, INT8 não validado on-rig) |
| API REST | `Dockerfile.cpu` multi-stage | FastAPI lazy-TF singleton | parcial (MVP sem auth) |
| Edge rig | conda/container ARM-friendly | TFLite INT8 + delegates | planejado (Fase 2) |

---

### 4. Decisão: Monorepo vs Multi-Repo

**Recomendação: MONOREPO com pacotes e fronteiras de import enforçadas, com extração futura do Studio como opção contingente.**

```
┌────────────────────┬──────────────────────────┬──────────────────────────────┐
│ Critério           │ Monorepo (RECOMENDADO)   │ Multi-repo                   │
├────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Paridade física    │ 1 fonte multi_forward;   │ risco de drift entre repos;  │
│ <1e-12             │ teste-oráculo único ✓    │ duplicação tentadora ✗       │
│ gui/+ingest novos  │ refactor atômico c/ lib  │ versão-lock cross-repo ✗     │
│ Velocidade dev     │ 1 PR atravessa camadas ✓ │ N PRs coordenados ✗          │
│ CI                 │ 1 matriz, cache GHA ✓     │ N pipelines ✗                │
│ Licenciamento      │ Studio comercial mistura  │ separação limpa MIT/comerc ✓ │
│                    │ com lib MIT (atenção) ⚠   │                              │
│ Tamanho contexto   │ monólito 10,7k pesa ⚠     │ isolado ✗ p/ reuso           │
└────────────────────┴──────────────────────────┴──────────────────────────────┘
```

Enquanto o Studio é alpha e a fundação está em fluxo (gui/+ingest/+registry novos, MVVM em definição), o **acoplamento atômico do monorepo vence** o overhead de coordenação. Fronteiras se enforçam por: (a) hook `validate-import-boundaries.sh` (apps/ nunca importam de `simulation/tests/`; lib nunca importa de apps/), (b) extras pip (`[gui]`, `[studio]`) que mantêm a lib instalável sem Qt. **Gatilho de extração do Studio** para repo privado próprio: quando (1) a API da lib estabilizar (`gui/` e `ingest/` GA) **e** (2) houver requisito comercial/licença que exija separação MIT↔proprietário. Aí o Studio passa a consumir `geosteering_ai` como dependência versionada PyPI/privada.

---

### 5. Stack Tecnológica Consolidada

| Camada | Tecnologia | Papel | Restrição |
|:-------|:-----------|:------|:----------|
| DL produção | **TensorFlow 2.x / Keras 3.x** | 48 arq, 26 losses, training, inference | EXCLUSIVO — PyTorch só `adapters/pytorch_adapter.py` |
| Forward GPU/diff | **JAX (cuda12)** | vmap/pmap, jacfwd, batched-bucketed | paridade JAX-vs-Numba <1e-10 |
| Forward CPU | **Numba (njit+prange)** | kernel HMD/VMD, caminho crítico | nunca `parallel=True` aninhado (KB-013) |
| Oráculo | **Fortran (tatu.x)** | validação `<1e-12`, não em runtime | paridade sagrada |
| GUI | **Qt6 (PyQt6 + PySide6 dual)** | SM + Studio via `gui/qt_compat` | `worker-object+moveToThread` (não subclassar QThread) |
| Plot | **PyQtGraph (30fps) + vispy (volume)** | curtain, realtime, picasso | alinhado à literatura |
| API | **FastAPI + Pydantic v2 + uvicorn** | REST `/predict /simulate /health` | lazy-TF, +auth pendente |
| Registry/MLOps | **MLflow 3.x (on-prem A6000)** | lineage modelo (git+dataset+seed), drift | NOVO |
| Serving | **ONNX Runtime + TFLite/LiteRT** | batch + realtime INT8 | golden-test pré/pós-INT8 |
| Ingestão | **lasio, dlisio, ETP/WebSocket, PWLS** | dados reais → 22-col | NOVO, errata fail-fast |
| Empacotamento | **conda-constructor + setuptools/pip** | instalador desktop + wheel | Python 3.13, conda `Geosteering_AI` |
| Container | **Docker (Dockerfile.cpu)** | API serving | non-root, +K8s pendente |

---

### 6. ADRs a Registrar

| ADR | Título | Decisão a documentar |
|:----|:-------|:---------------------|
| ADR-0002 | Monorepo com fronteiras de import + extração contingente do Studio | recomendação §4; gatilhos de split |
| ADR-0003 | `gui/` como camada Qt compartilhada (SM e Studio) | extrair `sm_*.py` → `gui/`; SM=gui+simulation, Studio=gui+lib inteira |
| ADR-0004 | Estrangulamento do monólito `simulation_manager.py` (Strangler Fig) | 10,7k LOC não é esqueleto; migração faseada p/ `apps/simulation_manager/` |
| ADR-0005 | MVVM no Studio (ViewModels sem import Qt) | testabilidade sem pytest-qt; PipelineConfig como Model |
| ADR-0006 | `ingest/` e formato 22-col como contrato de fronteira | WITSML/ETP/LAS/DLIS + PWLS → 22-col errata-checked; MD↔TVD min-curvature |
| ADR-0007 | Model Registry (MLflow on-prem) + golden-test de inferência por versão | estende tripé config+seed+versão ao artefato treinado |
| ADR-0008 | Runtime de produção congelado: ONNX RT (batch) + TFLite INT8 (realtime) | paridade pré/pós-quantização como gate |
| ADR-0009 | CLI `--backend auto` reusando `simulation/dispatch.py` | unifica heurística lib↔CLI (crossover n=32, anti-OOM 80GB) |
| ADR-0010 | API hardening pré-Studio-ALPHA (X-API-Key, rate-limit, `/simulate`, pip-audit) | bloqueante comercial |
| ADR-0011 | Métricas de calibração de UQ (CRPS, coverage, reliability) em `evaluation/calibration.py` | gap de credibilidade científica |
| ADR-0012 | Arquitetura edge-first no rig (IEC 62443 / Zero Trust) + export OSDU | Fase 2; só predições+UQ saem do rig |
| ADR-0013 | Validação de campo real como gate de credibilidade (estilo Goliat/GWC-SPE) | gap #1; benchmark reproduzível antes de claim comercial |

> Nota: ADR-0001 (hierarquia de planejamento / SSoT 4-doc) já existe e é preservado; esta numeração continua a sequência.
