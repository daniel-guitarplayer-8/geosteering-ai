# Geosteering AI — Arquitetura de Software v2.0

## Pipeline de Inversao Geofisica com Deep Learning

| Atributo | Valor |
|:---------|:------|
| **Projeto** | Inversao 1D de Resistividade via Deep Learning para Geosteering |
| **Versao** | v5.0.15 → v2.0 (arquitetura de software) |
| **Autor** | Daniel Leal |
| **Framework** | TensorFlow 2.x / Keras **EXCLUSIVO** (PyTorch PROIBIDO) |
| **Ambiente** | Desenvolvimento: VSCode + Claude Code · Versionamento: GitHub · Execucao: Google Colab Pro+ (GPU) |
| **Linguagem** | Python 3.10+ · Variaveis em ingles · Comentarios/docs em PT-BR |
| **Repositorio** | `github.com/daniel-leal/geosteering-ai` |

---

## 1. Visao Geral e Missao

### 1.1. Missao do Projeto

Reproduzir, com **fidelidade fisica**, a inversao eletromagnetica em tempo real
atraves de arquiteturas de Deep Learning. O sistema integrado deve ser capaz de
utilizar **componentes EM + geosinais e/ou Feature Views** como features para
inversao de resistividade em cenarios de:

- **Inferencia offline** (acausal, batch completo, 44 arquiteturas)
- **Inferencia causal realtime** para geosteering (sliding window, 27 arquiteturas compatíveis)
- **Ambientes ruidosos** com noise on-the-fly (fidelidade com medicoes LWD reais)

### 1.2. Principio Fisico Central

Em operacoes reais de LWD (Logging While Drilling), a ferramenta mede campos
eletromagneticos **com ruido instrumental**. Geosinais e Feature Views em producao
sao computados a partir dessas medicoes ruidosas. O pipeline deve reproduzir
essa realidade:

```
CADEIA FISICA REAL (ferramenta LWD):
  Campos EM brutos (com ruido) → Feature Views → Geosinais → Inversao

CADEIA DO SOFTWARE (on-the-fly):
  EM raw (sintetico) → Noise(σ) → FV_tf(noisy) → GS_tf(noisy) → Scale → Modelo DL
```

A cadeia on-the-fly e a **unica** implementacao de ruido no pipeline v2.0.
Ruido offline (pre-computado) foi removido por violar a fidelidade fisica
quando Feature Views e Geosinais estao ativos.

### 1.3. Por Que Uma Arquitetura de Software

O projeto atingiu **71.899 linhas** em 50+ arquivos, com **574 chamadas
`globals().get()`**, **~1.185 FLAGS**, e **44 arquiteturas**. O formato notebook-flat
original nao suporta mais a complexidade do sistema:

| Problema Diagnosticado | Impacto | Solucao v2.0 |
|:-----------------------|:--------|:-------------|
| 574 `globals().get()` sem validacao estatica | Defaults divergentes causaram bugs (S15-S22) | PipelineConfig dataclass |
| C40 "god cell" (3.583L, 125 branches) | Impossivel rastrear combinacao de FLAGS | CallbackFactory |
| Funcoes com CC > 50 (5 funcoes) | Intestableis, regressao frequente | Decomposicao em sub-funcoes |
| Double-processing FV/GS (C22+C24) | GS computados de dados limpos | DataPipeline com cadeia unica |
| Noise no dominio normalizado (pos-scaling) | Diverge da fisica (pre-scaling) | Noise em raw EM (A/m) |
| Dead code (75+ linhas em C24) | `_tf_scale_em`/`_tf_scale_gs` nunca usados | Ativados no Step 4 |
| Sem testes unitarios | Regressoes detectadas por inspecao manual | pytest por modulo |
| Sem versionamento formal | Rollback impossivel, reprodutibilidade manual | GitHub + tags + YAML |

---

## 2. Infraestrutura: GitHub + VSCode + Colab

### 2.1. Fluxo de Trabalho Tripartite

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  1. DESENVOLVIMENTO (VSCode + Claude Code)                           │
│     /Users/daniel/Geosteering_AI/                                    │
│     ├── geosteering_ai/          ← Claude Code edita AQUI (local)   │
│     ├── tests/                   ← Claude Code roda pytest AQUI     │
│     └── .git/                    ← Repositorio Git LOCAL            │
│                                                                       │
│     Ciclo: editar → pytest → git commit → git push                   │
│              │                                                        │
│              ▼                                                        │
│  2. GITHUB (github.com/daniel-leal/geosteering-ai)                  │
│     ├── CI automatico (GitHub Actions)                               │
│     │   ├── py_compile em todos os modulos                           │
│     │   ├── pytest tests/ (CPU-only: config, shapes, logica)        │
│     │   ├── mypy type checking                                       │
│     │   └── Status: ✓ ou ✗ no PR                                    │
│     ├── Branches: main ← develop ← feature/*                        │
│     ├── Releases: tags v2.0.0, v2.1.0, ...                          │
│     └── Issues: rastreio de bugs e features                          │
│              │                                                        │
│              ▼                                                        │
│  3. EXECUCAO (Google Colab Pro+ com GPU)                             │
│     Celula 1: !pip install git+https://github.com/daniel-leal/      │
│                   geosteering-ai.git@v2.1.0                          │
│     Celula 2: config = PipelineConfig.from_yaml("robusto.yaml")     │
│     Celula 3: data = DataPipeline(config).prepare(dataset_path)     │
│     Celula 4: model = ModelRegistry().build(config)                  │
│     Celula 5: history = TrainingLoop(config, model, ...).run()       │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2. Branching Strategy

```
main                    ← Releases estaveis (tags v2.x.x), Colab producao
  │
  └── develop           ← Integracao continua de features
        │
        ├── feature/phase0-config-utils      ← Fase 0: fundacao
        ├── feature/phase1-models            ← Fase 1: arquiteturas
        ├── feature/phase2-data-pipeline     ← Fase 2: dados + noise
        ├── feature/phase3-training          ← Fase 3: treinamento
        ├── feature/phase4-eval-vis          ← Fase 4: avaliacao
        └── fix/geosignal-call-signature     ← Correcoes pontuais
```

| Branch | Proposito | Merge para | Protecao |
|--------|-----------|:----------:|:--------:|
| `main` | Releases estaveis | — | CI obrigatorio + tag |
| `develop` | Integracao de features | `main` (via PR + tag) | CI obrigatorio |
| `feature/*` | Uma feature/fase | `develop` (via PR) | CI obrigatorio |
| `fix/*` | Correcao pontual | `develop` (via PR) | CI obrigatorio |

### 2.3. CI: GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install
        run: pip install -e ".[dev]"

      - name: Compile check
        run: python -m compileall geosteering_ai/ -q

      - name: Tests (CPU-only)
        run: pytest tests/ -v --ignore=tests/test_gpu.py

      - name: Type check
        run: mypy geosteering_ai/ --ignore-missing-imports
```

Testes de GPU (treinamento real) rodam no Colab, nao no CI.
O CI valida compilacao, types, shapes, e logica pura.

### 2.4. Reprodutibilidade via Release Tags

```yaml
# Registro de experimento (salvo junto ao modelo treinado)
experiment:
  code_version: "v2.1.0"              # tag do GitHub
  config: "configs/nstage_n2.yaml"     # preset usado
  seed: 42                             # seed global
  date: "2026-03-25"
  dataset: "arranjoTR1_60k"
  result:
    val_loss: 0.1243
    best_epoch: 87
    r2_clean: 0.97
```

Para reproduzir:
```bash
pip install git+https://github.com/daniel-leal/geosteering-ai.git@v2.1.0
```
```python
config = PipelineConfig.from_yaml("configs/nstage_n2.yaml")
# Resultado identico (mesmo codigo + config + seed)
```

### 2.5. O Que Vai e Nao Vai no GitHub

```gitignore
# .gitignore — Dados e modelos ficam no Google Drive
*.dat                    # Dados binarios do Fortran (grandes)
*.out                    # Metadata do simulador
*.keras                  # Modelos treinados
*.h5                     # Pesos
checkpoints/             # Checkpoints de treinamento
experiments/             # Resultados de experimentos
__pycache__/
*.pyc
.env
```

---

## 3. Estrutura de Diretorios

```
github.com/daniel-leal/geosteering-ai/
│
├── geosteering_ai/                     ← PACOTE PYTHON
│   ├── __init__.py                     ← Versao, imports publicos
│   ├── config.py                       ← PipelineConfig dataclass
│   │
│   ├── data/                           ← Modulo de dados
│   │   ├── __init__.py
│   │   ├── loader.py                   ← parse_out_metadata, load_binary_dat
│   │   ├── splitter.py                 ← split_by_geological_model [P1]
│   │   ├── decoupling.py              ← ACp/ACx, apply_decoupling
│   │   ├── feature_views.py           ← 6 FV (numpy + TF)
│   │   ├── geosignals.py             ← 5 familias GS (numpy + TF)
│   │   ├── scaling.py                 ← fit/transform scalers, per-group [P3]
│   │   └── pipeline.py               ← DataPipeline class (orquestrador)
│   │
│   ├── noise/                          ← Modulo de ruido (ON-THE-FLY EXCLUSIVO)
│   │   ├── __init__.py
│   │   ├── functions.py               ← gaussian, multiplicative, uniform, dropout (TF ops)
│   │   ├── curriculum.py              ← CurriculumSchedule (3-phase ramp)
│   │   └── utils.py                   ← apply_raw_em_noise numpy (visualizacao/debug)
│   │
│   ├── models/                         ← Modulo de arquiteturas (44 modelos)
│   │   ├── __init__.py
│   │   ├── registry.py                ← ModelRegistry class (44 entradas)
│   │   ├── blocks.py                  ← 23 blocos Keras reutilizaveis
│   │   ├── cnn.py                     ← ResNet_18/34/50, ConvNeXt, Inception, CNN_1D (7)
│   │   ├── rnn.py                     ← LSTM, BiLSTM, CNN_LSTM, CNN_BiLSTM_ED (4)
│   │   ├── tcn.py                     ← TCN, TCN_Advanced (2)
│   │   ├── transformer.py            ← Transformer, TFT, PatchTST, Autoformer, iTransformer (6)
│   │   ├── unet.py                    ← 14 U-Net variantes (offline-only)
│   │   ├── decomposition.py          ← N-BEATS, N-HiTS (2)
│   │   ├── advanced.py               ← DNN, FNO, DeepONet, GeoAttention (4)
│   │   └── geosteering.py            ← WaveNet, Causal_Transformer, Informer,
│   │                                    Mamba_S4, Encoder_Forecaster (5 nativas causais)
│   │
│   ├── losses/                         ← Modulo de funcoes de perda (26 losses)
│   │   ├── __init__.py
│   │   ├── catalog.py                 ← 26 losses individuais (13 gen + 4 geo + 9 adv)
│   │   ├── factory.py                 ← LossFactory + build_combined_loss()
│   │   └── geophysical.py            ← losses #14-#17 (geo-aware, TARGET_SCALING-aware)
│   │
│   ├── training/                       ← Modulo de treinamento
│   │   ├── __init__.py
│   │   ├── callbacks.py               ← build_callbacks() factory + 26 callbacks
│   │   ├── loop.py                    ← TrainingLoop class (compile + fit)
│   │   ├── nstage.py                  ← NStageTrainer (N stages + mini-curriculum)
│   │   ├── optuna_hpo.py             ← OptunaOptimizer (opt-in)
│   │   └── metrics.py                ← R2Score, PerComponentMetric, AnisotropyRatioError
│   │
│   ├── inference/                      ← Modulo de inferencia (offline + realtime)
│   │   ├── __init__.py
│   │   ├── pipeline.py               ← InferencePipeline (FV+GS+scalers serializaveis)
│   │   ├── realtime.py               ← SlidingWindowInference, GeoSteeringSession
│   │   └── export.py                 ← save/load modelo + scalers + config
│   │
│   ├── evaluation/                     ← Modulo de avaliacao
│   │   ├── __init__.py
│   │   ├── metrics.py                ← compute_metrics(), per_component, anisotropy
│   │   ├── comparison.py             ← compare_models(), benchmark_table
│   │   └── sensitivity.py            ← sensitivity_analysis()
│   │
│   ├── visualization/                  ← Modulo de visualizacao
│   │   ├── __init__.py
│   │   ├── holdout.py                ← plot_holdout_sample (clean + noisy overlay)
│   │   ├── training_curves.py        ← loss/metrics curves, dual validation
│   │   ├── picasso.py                ← DOD plots, multi-freq/angle (P5)
│   │   ├── eda.py                    ← exploratory data analysis
│   │   ├── realtime_monitor.py       ← trajetoria do poco, faixa de incerteza
│   │   └── style.py                  ← LaTeX config, color schemes, 4K export
│   │
│   └── utils/                          ← Utilitarios compartilhados
│       ├── __init__.py
│       ├── logger.py                  ← setup_logger, ColoredFormatter, ANSI C class
│       ├── timer.py                   ← format_time, timer_decorator
│       ├── validation.py             ← ValidationTracker
│       ├── formatting.py             ← print_header, format_number, format_bytes
│       ├── system.py                  ← is_colab, has_gpu, gpu_memory_info
│       └── io.py                      ← safe_mkdir, safe_json_dump, ensure_dirs
│
├── configs/                            ← Presets de configuracao (YAML)
│   ├── baseline.yaml                  ← P1: FV=identity, GS=off, noise=off
│   ├── robusto.yaml                   ← E-Robusto: noise 8%, curriculum, LR=1e-4
│   ├── nstage_n2.yaml                ← N-Stage N=2 + mini-curriculum
│   ├── nstage_n3.yaml                ← N-Stage N=3 (exploratorio)
│   ├── geosinais_p4.yaml            ← P4: GS on, noise on-the-fly
│   └── realtime_causal.yaml          ← Geosteering: modo causal
│
├── tests/                              ← Testes unitarios (pytest)
│   ├── test_config.py                ← Validacao PipelineConfig + errata
│   ├── test_data_pipeline.py         ← Loading, splitting, scaling, on-the-fly chain
│   ├── test_noise.py                 ← Noise functions, curriculum 3-phase
│   ├── test_feature_views.py         ← 6 FV: numpy/TF consistency
│   ├── test_geosignals.py            ← 5 familias: numpy/TF consistency
│   ├── test_models.py                ← Forward pass em todas 44 arquiteturas
│   ├── test_losses.py                ← 26 losses forward pass + gradients
│   └── test_training.py              ← 1-epoch smoke test (CPU)
│
├── notebooks/                          ← Notebooks Colab (orquestradores finos)
│   ├── train_colab.ipynb             ← Treinamento completo (~15 celulas)
│   ├── evaluate_colab.ipynb          ← Avaliacao + visualizacao
│   ├── geosteering_colab.ipynb       ← Inferencia realtime / geosteering
│   └── eda_colab.ipynb               ← Exploracao de dados + Picasso DOD
│
├── docs/                               ← Documentacao do projeto
│   ├── ARCHITECTURE_v2.md            ← ESTE DOCUMENTO
│   ├── physics/                       ← Contexto fisico/geofisico
│   │   ├── em_tensor.md              ← Tensor EM 3x3, 22-col format
│   │   ├── decoupling.md             ← ACp/ACx, formulas
│   │   ├── geosignals.md             ← 5 familias (USD, UAD, UHR, UHA, U3DF)
│   │   └── feature_views.md          ← 6 transformacoes (identity, logH, fases...)
│   └── references/                    ← PDFs e artigos de referencia
│
├── Fortran_Gerador/                    ← Simulador EM Fortran (gerador de dados)
│   ├── PerfilaAnisoOmp.f08           ← Codigo principal do simulador
│   ├── *.dat                          ← Dados binarios 22-col (NO GITHUB — .gitignore)
│   └── *.out                          ← Metadata (NO GITHUB — .gitignore)
│
├── legacy/                             ← Codigo legado (preservado para referencia)
│   ├── Arquivos_Projeto_Claude/       ← C0-C47 .py originais
│   └── Skill/                         ← Skill docs v5.0.15
│
├── .github/
│   └── workflows/
│       ├── ci.yml                     ← Testes automaticos (push/PR)
│       └── release.yml                ← Tag → release
│
├── pyproject.toml                     ← Build system (pip installable)
├── .gitignore                         ← *.dat, *.keras, checkpoints, __pycache__
├── CLAUDE.md                          ← Instrucoes persistentes Claude Code (atualizado)
└── README.md                          ← Documentacao publica do repositorio
```

### 3.1. Diagrama de Dependencias entre Modulos

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  configs/*.yaml ──→ config.py (PipelineConfig)                          │
│                         │                                                │
│         ┌───────────────┼───────────────┐                               │
│         │               │               │                                │
│         ▼               ▼               ▼                                │
│    data/            models/         losses/                              │
│    ├── loader       ├── registry    ├── catalog                         │
│    ├── splitter     ├── blocks      ├── factory                         │
│    ├── decoupling   ├── cnn         └── geophysical                     │
│    ├── feature_views├── rnn                                              │
│    ├── geosignals   ├── transformer     noise/                          │
│    ├── scaling      ├── unet            ├── functions                   │
│    └── pipeline ◄───┤   ...             ├── curriculum                  │
│         │           └── geosteering     └── utils                       │
│         │               │                    │                           │
│         │               ▼                    │                           │
│         │          training/                 │                           │
│         ├─────────→├── callbacks ◄───────────┘                          │
│         │          ├── loop                                              │
│         │          ├── nstage                                            │
│         │          └── metrics                                           │
│         │               │                                                │
│         │               ▼                                                │
│         │          inference/        evaluation/      visualization/     │
│         └─────────→├── pipeline      ├── metrics      ├── holdout       │
│                    ├── realtime      ├── comparison    ├── picasso       │
│                    └── export        └── sensitivity   └── realtime      │
│                                                                          │
│  utils/ (logger, timer, validation, formatting, system, io)             │
│  └── importado por TODOS os modulos acima                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Principios de Organizacao

| Principio | Aplicacao |
|:----------|:---------|
| **Single Responsibility** | Cada modulo .py tem uma unica responsabilidade (ex: scaling.py so escala) |
| **Inversao de Dependencia** | Modulos dependem de `config.py` + `utils/`, nao uns dos outros |
| **DRY (Don't Repeat Yourself)** | Geosteering e modo (flag), nao modulo duplicado |
| **Open-Closed** | Adicionar arquitetura = `registry.register()`, nao editar factory |
| **Separation of Concerns** | Dados, modelos, treinamento, inferencia em modulos separados |
| **Fail-Fast** | `PipelineConfig.__post_init__()` valida ANTES de qualquer execucao |

---

## 4. Design Patterns

### 4.1. Configuration Object — PipelineConfig

**Substitui:** 574 chamadas `globals().get()` dispersas por um ponto unico de verdade.

```python
# geosteering_ai/config.py

@dataclass
class PipelineConfig:
    """Configuracao unica e validada do pipeline.

    Cada FLAG do pipeline e um campo tipado com default explicito.
    Validacao automatica no __post_init__ garante errata e consistencia.
    Serializavel para YAML (reprodutibilidade) e dict (logging).

    Presets:
        PipelineConfig.baseline()       # P1: sem noise, sem GS
        PipelineConfig.robusto()        # E-Robusto S21: noise 8%, curriculum
        PipelineConfig.nstage(n=3)      # N-Stage + mini-curriculum
        PipelineConfig.geosinais_p4()   # P4: GS on, noise on-the-fly
        PipelineConfig.realtime()       # Geosteering: modo causal

    Example:
        >>> config = PipelineConfig.from_yaml("configs/robusto.yaml")
        >>> config.noise_level_max
        0.08
    """

    # ── Fisica ───────────────────────────────────────────────────────
    frequency_hz: float = 20000.0          # NUNCA 2.0 (Errata v4.4.5)
    spacing_meters: float = 1.0            # NUNCA 1000.0 (Errata v4.4.5)
    sequence_length: int = 600             # NUNCA 601
    input_features: List[int] = field(default_factory=lambda: [1, 4, 5, 20, 21])
    output_targets: List[int] = field(default_factory=lambda: [2, 3])
    target_scaling: str = "log10"          # NUNCA "log"

    # ── Dados ────────────────────────────────────────────────────────
    split_by_model: bool = True            # [P1] NUNCA split por amostra
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    use_dual_validation: bool = True       # [P2] val_clean + val_noisy

    # ── Feature Views + Geosinais ────────────────────────────────────
    feature_view: str = "identity"         # 6 opcoes: identity, H1_logH2, etc.
    use_geosignal_features: bool = False   # [P4] Ativa geosinais on-the-fly
    geosignal_set: str = "usd_uhr"        # Familias: usd_uhr, full_1d, full_3d
    eps_tf: float = 1e-12                  # [P7] NUNCA 1e-30

    # ── Scaling ──────────────────────────────────────────────────────
    scaler_type: str = "standard"
    use_per_group_scalers: bool = True     # [P3] StandardScaler(EM) + RobustScaler(GS)
    gs_scaler_type: str = "robust"

    # ── Noise (ON-THE-FLY EXCLUSIVO) ─────────────────────────────────
    use_noise: bool = True
    noise_level_max: float = 0.08          # E-Robusto S21
    noise_types: List[str] = field(default_factory=lambda: ["gaussian"])
    noise_weights: List[float] = field(default_factory=lambda: [1.0])
    use_curriculum: bool = True            # 3-phase: clean → ramp → estavel
    epochs_no_noise: int = 10
    noise_ramp_epochs: int = 80

    # ── Arquitetura ──────────────────────────────────────────────────
    model_type: str = "ResNet_18"          # 44 opcoes no ModelRegistry
    inference_mode: str = "offline"        # "offline" ou "realtime"
    use_causal_mode: bool = False          # Auto-True quando realtime
    output_channels: int = 2               # 2=[rho_h,rho_v], 4=[+sigma], 6=[+DTB]

    # ── Treinamento ──────────────────────────────────────────────────
    learning_rate: float = 1e-4            # E-Robusto S21
    epochs: int = 400
    batch_size: int = 32
    optimizer: str = "adamw"
    early_stopping_patience: int = 60

    # ── N-Stage Training ─────────────────────────────────────────────
    use_nstage: bool = False               # Mutuamente exclusivo com curriculum
    n_training_stages: int = 2
    nstage_stage1_epochs: int = 15
    stage_lr_decay: float = 0.5
    use_stage_mini_curriculum: bool = True
    stage_ramp_fraction: float = 0.25

    # ── Loss ─────────────────────────────────────────────────────────
    loss_type: str = "rmse"                # 26 opcoes no LossFactory
    use_look_ahead_loss: bool = False
    use_dtb_loss: bool = False
    use_pinns: bool = False

    # ── Paths ────────────────────────────────────────────────────────
    base_dir: str = "/content/drive/MyDrive/Geosteering_AI"
    global_seed: int = 42

    def __post_init__(self):
        """Validacao centralizada — fail-fast."""
        # Errata fisicas (imutaveis)
        assert self.frequency_hz == 20000.0, "NUNCA 2.0 Hz"
        assert self.spacing_meters == 1.0, "NUNCA 1000.0 m"
        assert self.sequence_length == 600, "NUNCA 601"
        assert self.target_scaling == "log10", "NUNCA 'log'"
        assert self.input_features == [1, 4, 5, 20, 21]
        assert self.output_targets == [2, 3]
        # Mutual exclusivity
        if self.use_nstage:
            assert not self.use_curriculum, "N-Stage e Curriculum mutuamente exclusivos"
        # Ranges
        assert 0.0 <= self.noise_level_max <= 1.0
        assert len(self.noise_types) == len(self.noise_weights)

    @property
    def needs_onthefly_fv_gs(self) -> bool:
        """True se FV/GS devem ser computados on-the-fly (pos-noise)."""
        return (
            self.use_noise
            and (self.feature_view not in ("identity", "raw", None)
                 or self.use_geosignal_features)
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig": ...
    def to_yaml(self, path: str) -> None: ...
    def to_dict(self) -> dict: ...
```

### 4.2. Registry + Factory (Models, Losses, Callbacks)

**Substitui:** Cascatas if/elif e dicts globais por classes extensiveis.

```python
# geosteering_ai/models/registry.py

class ModelRegistry:
    """Registro central de 44 arquiteturas com validacao causal."""

    def register(self, name, build_fn, tier, category, causal_compatible, cell): ...
    def build(self, config: PipelineConfig) -> tf.keras.Model: ...
    def list_causal_compatible(self) -> List[str]: ...

# geosteering_ai/losses/factory.py

class LossFactory:
    """Factory para 26 funcoes de perda com prioridade Morales > Prob > base."""

    @classmethod
    def get(cls, config: PipelineConfig) -> Callable: ...
    @classmethod
    def register(cls, name: str) -> Callable: ...  # decorador

# geosteering_ai/training/callbacks.py

def build_callbacks(config, model, noise_level_var) -> List[Callback]:
    """Monta lista de callbacks — substitui 550L imperativas de C40."""
    ...
```

### 4.3. DataPipeline — Cadeia Fisicamente Correta

**Substitui:** C22+C23+C24 com estado global mutavel.
**Garante:** raw → noise → FV → GS → scale (unica passagem, sem double-processing).

```python
# geosteering_ai/data/pipeline.py

class DataPipeline:
    """Pipeline de dados unificado.

    Dois modos automaticos (config.needs_onthefly_fv_gs):

    MODO OFFLINE (FV=identity, GS=off, OU noise=off):
      prepare(): FV+GS+scale offline em train/val/test
      train_map_fn: noise ONLY (sobre dados ja processados)

    MODO ON-THE-FLY (FV ou GS ativos COM noise):
      prepare(): fit scaler em clean FV+GS, val/test offline, train=RAW
      train_map_fn: noise → FV_tf → GS_tf → scale_tf (cadeia completa)
    """

    def __init__(self, config: PipelineConfig): ...

    def prepare(self, dataset_path: str) -> DataSplits:
        """Carrega, divide, fita scalers, prepara splits."""
        # 1. Load .dat + parse .out
        # 2. Decoupling (ACp/ACx)
        # 3. Split por modelo geologico [P1]
        # 4. FV+GS em clean → fit scaler (temporario)
        # 5. Val/Test: FV+GS+scale offline
        # 6. Train: RAW se on-the-fly, FV+GS+scale se offline
        ...

    def build_train_map_fn(self, noise_level_var) -> Callable:
        """Constroi cadeia on-the-fly via closure (sem globals)."""
        # Captura: config, fv_fn_tf, gs_fn_tf, scale_em_tf, scale_gs_tf
        def map_fn(x, y):
            # Step 1: Noise em raw EM (A/m)
            # Step 2: FV sobre EM ruidoso
            # Step 3: GS sobre EM ruidoso (fisicamente correto)
            # Step 4: Scaling (scaler fitado em clean)
            return x, y
        return map_fn

    def build_inference_pipeline(self) -> InferencePipeline:
        """Exporta FV+GS+scalers para inferencia identica ao treino."""
        ...
```

### 4.4. Strategy Pattern (Noise Curriculum)

```python
# geosteering_ai/noise/curriculum.py

class CurriculumSchedule:
    """3-phase noise ramp: clean → ramp linear → estavel."""

    def get_noise_level(self, epoch: int) -> float:
        if epoch < self.epochs_clean: return 0.0                    # Fase 1
        ramp = epoch - self.epochs_clean
        if ramp < self.ramp_epochs: return self.max * ramp / self.ramp_epochs  # Fase 2
        return self.max                                              # Fase 3
```

---

## 5. Cadeia de Dados — Detalhamento Tecnico

### 5.1. Fluxo Completo On-the-Fly

```
┌─────────────────────────────────────────────────────────────────────┐
│                  CADEIA FISICAMENTE CORRETA v2.0                     │
│                                                                      │
│  .dat (22-col binario, Fortran PerfilaAnisoOmp)                     │
│       │                                                              │
│       ▼                                                              │
│  LOAD + PARSE .out (theta, freq, nmeds, n_models)                   │
│       │                                                              │
│       ▼                                                              │
│  DECOUPLING EM                                                       │
│    Re{Hxx} -= ACp = -1/(4πL³),  L=1.0m                             │
│    Re{Hzz} -= ACx = +1/(2πL³)                                      │
│       │                                                              │
│       ▼                                                              │
│  SPLIT POR MODELO GEOLOGICO [P1]                                     │
│    train (70%) ∩ val (15%) ∩ test (15%) = ∅ em model_ids           │
│       │                                                              │
│       ├── train.x: RAW EM expandido (EXPANDED_INPUT_FEATURES)       │
│       ├── val.x, test.x: raw EM                                     │
│       │                                                              │
│       ▼                                                              │
│  FIT SCALERS (em dados LIMPOS — FV+GS aplicados temporariamente)    │
│    train_clean → FV(clean) → GS(clean) → scaler_em.fit()           │
│                                           scaler_gs.fit()           │
│    (resultado temporario descartado — train.x permanece RAW)        │
│       │                                                              │
│       ▼                                                              │
│  TRANSFORMAR VAL/TEST OFFLINE (para avaliacao consistente)           │
│    val  → FV → GS → scale → val_clean_ds                            │
│    test → FV → GS → scale → test_ds                                 │
│       │                                                              │
│       ▼                                                              │
│  tf.data.Dataset.map(train_map_fn) — POR BATCH, CADA EPOCA          │
│                                                                      │
│    ┌─────────────────────────────────────────────────────────────┐   │
│    │  train.x_raw ──→ STEP 1: NOISE σ(epoch)                    │   │
│    │                      σ = noise_level_var (curriculum/N-Stage)│   │
│    │                      Ruido em A/m (dominio fisico)          │   │
│    │                          │                                   │   │
│    │                          ▼                                   │   │
│    │                   STEP 2: FEATURE VIEW (FV_tf)              │   │
│    │                      Sobre EM RUIDOSO                       │   │
│    │                      Ex: log|H|, amplitude, fase            │   │
│    │                          │                                   │   │
│    │                          ▼                                   │   │
│    │                   STEP 3: GEOSINAIS (GS_tf)                 │   │
│    │                      Sobre EM RUIDOSO → fisicamente correto │   │
│    │                      USD, UHR, etc. (att dB + fase deg)     │   │
│    │                      Propagacao nao-linear do ruido          │   │
│    │                          │                                   │   │
│    │                          ▼                                   │   │
│    │                   STEP 4: SCALING                            │   │
│    │                      scaler_em (StandardScaler) em features  │   │
│    │                      scaler_gs (RobustScaler) em geosinais   │   │
│    │                      Fitados em dados LIMPOS FV+GS           │   │
│    │                          │                                   │   │
│    │                          ▼                                   │   │
│    │                      (x_scaled, y) → MODELO DL              │   │
│    └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  INFERENCIA (InferencePipeline — mesma cadeia, sem noise):           │
│    raw_novo → FV → GS → scale → model.predict()                    │
│            → inverse_target_scaling() → rho_h, rho_v (Ω·m)        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2. Por Que Noise On-the-Fly Exclusivo

O ruido offline (cenarios 3A-3D do pipeline legado v5.0.15) foi **removido**
do pipeline v2.0 porque viola a fidelidade fisica quando FV e GS estao ativos:

```
OFFLINE (REMOVIDO — violacao fisica):
  train → FV(clean) → GS(clean) → scale → copiar K vezes + noise aditivo
                            │
                       GS nunca veem ruido
                       noise aditivo em dB ≠ propagacao fisica
                       Ex: GS_clean=0.35dB ± 0.08 vs GS_noisy_EM=2.26dB

ON-THE-FLY (UNICO — fisicamente correto):
  train_raw → noise(A/m) → FV(noisy) → GS(noisy) → scale
                                             │
                                        GS veem ruido
                                        Propagacao nao-linear correta
```

**Componentes removidos na v2.0:**
- `Noisy3DDataGenerator` class
- FLAGS: `DATA_SCENARIO` (selector), `OFFLINE_K_COPIES`
- Logica de tier-switching offline
- Cenarios 1A, 1B, 3A, 3B, 3C, 3D

**Preservado como utilitario:**
- `noise/utils.py`: `apply_raw_em_noise()` numpy para visualizacao e debugging

### 5.3. Dual-Mode: Offline vs Realtime

O modo de inferencia e controlado por `config.inference_mode`:

| Aspecto | Offline (padrao) | Realtime (geosteering) |
|:--------|:-----------------|:----------------------|
| Dados de entrada | Batch completo (n, seq, feat) | Sliding window (1, W, feat) |
| Rede | Acausal (44 arqs) | Causal (27 arqs compativeis) |
| Saida | (batch, N_MEDIDAS, 2) | (1, W, 2-6) |
| Incerteza | Opcional (ensemble UQ) | Automatica (NLL) |
| Padding Conv1D | `"same"` | `"causal"` |
| Treinamento | Direto | Fase 1 acausal + Fase 2 finetune causal |

O geosteering **nao e um modulo separado** — e um modo de operacao que ativa
comportamentos especificos em modulos existentes:

- `models/geosteering.py`: 5 arquiteturas nativas causais (exclusivas)
- `inference/realtime.py`: SlidingWindowInference, GeoSteeringSession (exclusivas)
- `visualization/realtime_monitor.py`: Painel operacional (exclusiva)
- Todos os demais modulos: comportamento condicional via `config.use_causal_mode`

---

## 6. Notebook Colab como Orquestrador

### 6.1. Exemplo: train_colab.ipynb (~15 celulas)

```python
# ── Celula 1: Setup ──────────────────────────────────────────────────
!pip install git+https://github.com/daniel-leal/geosteering-ai.git@v2.1.0 -q
from geosteering_ai import PipelineConfig
from geosteering_ai.data import DataPipeline
from geosteering_ai.models import ModelRegistry
from geosteering_ai.training import TrainingLoop

# ── Celula 2: Configuracao ───────────────────────────────────────────
config = PipelineConfig.from_yaml("/content/drive/MyDrive/configs/robusto.yaml")
print(config)

# ── Celula 3: Dados ──────────────────────────────────────────────────
pipeline = DataPipeline(config)
data = pipeline.prepare("/content/drive/MyDrive/datasets/arranjoTR1")
print(f"Train: {data.train.x.shape}, Val: {data.val.x.shape}")

# ── Celula 4: Modelo ─────────────────────────────────────────────────
model = ModelRegistry().build(config)
model.summary()

# ── Celula 5: Treinamento ────────────────────────────────────────────
trainer = TrainingLoop(config, model, pipeline, data)
history = trainer.run()

# ── Celula 6: Avaliacao ──────────────────────────────────────────────
from geosteering_ai.evaluation import evaluate_model
results = evaluate_model(model, data.test, pipeline, config)

# ── Celula 7: Salvar ─────────────────────────────────────────────────
from geosteering_ai.inference import export_model
export_model(model, pipeline, config, "/content/drive/MyDrive/models/v1")
```

### 6.2. Comparacao: Notebook Legado vs v2.0

| Aspecto | Legado (74 celulas) | v2.0 (~15 celulas) |
|---------|:-------------------:|:------------------:|
| Linhas no notebook | ~71.900 | ~500 |
| Tempo para ler | Horas | Minutos |
| Risco de celula errada | Alto | Baixo |
| Testabilidade | Nenhuma | pytest por modulo |
| Reprodutibilidade | Manual (copiar FLAGS) | config.yaml + tag GitHub |
| Adicionar arquitetura | Editar 3 celulas | `registry.register()` |
| Rollback | Impossivel | `git checkout v2.0.0` |
| Deploy Colab | Copiar .py para Drive | `pip install git+...@tag` |

---

## 7. Estrategia de Migracao

### 7.1. Abordagem: EXTRAIR + REFATORAR + CRIAR

| Categoria | Linhas Legado | Linhas v2.0 | Descricao |
|:---------:|:------------:|:-----------:|:----------|
| EXTRAIR | 20.150 (28%) | ~20.150 | Mover como esta (build_*, losses, utils) |
| REFATORAR | 13.649 (19%) | ~7.700 | Reestruturar (data pipeline, callbacks, training) |
| CRIAR | — | ~8.900 | Novo (config, tests, eval, vis, notebooks) |
| DESCARTAR | 9.812 (14%) | 0 | Boilerplate de celula, dead code, noise offline |
| **TOTAL** | **71.899** | **~36.750** | **-49% linhas, +testes, +GitHub, +YAML** |

### 7.2. Fases de Implementacao

```
FASE 0: FUNDACAO
  ├── pyproject.toml + geosteering_ai/__init__.py
  ├── config.py (PipelineConfig + presets + YAML)
  ├── utils/ (logger, timer, validation, formatting, system, io)
  ├── tests/test_config.py
  ├── GitHub: git init, .gitignore, ci.yml
  └── VALIDACAO: pytest tests/test_config.py ✓

FASE 1: MODELOS (EXTRAIR)
  ├── models/blocks.py (23 blocos ← C27)
  ├── models/cnn.py, rnn.py, tcn.py, transformer.py, unet.py, etc. (← C28-C36A)
  ├── models/registry.py (ModelRegistry ← C37)
  ├── tests/test_models.py (forward pass 44 arqs)
  └── VALIDACAO: build + dummy forward para cada arquitetura ✓

FASE 2: DADOS + NOISE (REFATORAR)
  ├── data/loader.py, splitter.py, decoupling.py (← C19, C21)
  ├── data/feature_views.py, geosignals.py (← C22)
  ├── data/scaling.py (← C23)
  ├── data/pipeline.py (DataPipeline — NOVO orquestrador)
  ├── noise/functions.py, curriculum.py (← C24)
  ├── tests/test_data_pipeline.py, test_noise.py
  └── VALIDACAO: carregar dataset real, shapes, split P1 ✓

FASE 3: LOSSES + TRAINING (EXTRAIR + REFATORAR)
  ├── losses/catalog.py, factory.py (← C41)
  ├── training/callbacks.py (CallbackFactory ← C40)
  ├── training/loop.py, nstage.py (← C43)
  ├── training/metrics.py (← C42)
  ├── tests/test_losses.py, test_training.py
  └── VALIDACAO: treinar 5 epocas ResNet_18, comparar val_loss ✓

FASE 4: REPRODUCAO DO BASELINE (GATE DE QUALIDADE)
  ├── notebooks/train_colab.ipynb
  ├── configs/robusto.yaml, nstage_n2.yaml
  ├── Treinar ResNet_18 E-Robusto completo no Colab
  └── VALIDACAO: val_loss ≈ 0.16 (S21) ou melhor ✓
      Se divergir > 10%: PARAR e investigar antes de prosseguir

FASE 5: INFERENCIA + AVALIACAO + VISUALIZACAO (CRIAR)
  ├── inference/pipeline.py, realtime.py, export.py
  ├── evaluation/metrics.py, comparison.py, sensitivity.py
  ├── visualization/holdout.py, training_curves.py, picasso.py, eda.py
  ├── notebooks/evaluate_colab.ipynb, geosteering_colab.ipynb, eda_colab.ipynb
  └── VALIDACAO: pipeline completo end-to-end ✓

FASE 6: LIMPEZA + RELEASE
  ├── Mover Arquivos_Projeto_Claude/ → legacy/
  ├── README.md publico
  ├── git tag v2.0.0
  └── Release GitHub ✓
```

### 7.3. Compatibilidade durante Migracao

Adapter para usar PipelineConfig no notebook legado (durante transicao):

```python
def inject_config_as_globals(config: PipelineConfig):
    """Injeta FLAGS do config no namespace global para celulas legadas."""
    for name, value in config.to_dict().items():
        globals()[name.upper()] = value
```

---

## 8. Testes

### 8.1. Estrutura

```python
# tests/test_config.py
def test_errata_validation():
    with pytest.raises(AssertionError): PipelineConfig(frequency_hz=2.0)

def test_mutual_exclusivity():
    with pytest.raises(AssertionError): PipelineConfig(use_nstage=True, use_curriculum=True)

def test_yaml_roundtrip(tmp_path):
    config = PipelineConfig.robusto()
    config.to_yaml(tmp_path / "test.yaml")
    loaded = PipelineConfig.from_yaml(tmp_path / "test.yaml")
    assert config == loaded

# tests/test_data_pipeline.py
def test_split_no_leakage(): ...        # P1: train ∩ val ∩ test = ∅
def test_scaler_fit_on_clean(): ...     # Scaler fitado em dados limpos
def test_onthefly_chain_shape(): ...    # noise→FV→GS→scale preserva shapes
def test_noise_preserves_zobs(): ...    # Noise so afeta EM, nao z

# tests/test_models.py
@pytest.mark.parametrize("model_type", ModelRegistry().list_all())
def test_forward_pass(model_type):
    config = PipelineConfig(model_type=model_type)
    model = ModelRegistry().build(config)
    dummy = tf.random.normal((2, 100, config.n_features))
    output = model(dummy)
    assert output.shape == (2, 100, config.output_channels)

# tests/test_noise.py
def test_curriculum_3phase():
    schedule = CurriculumSchedule(PipelineConfig.robusto())
    assert schedule.get_noise_level(0) == 0.0       # clean
    assert schedule.get_noise_level(50) > 0.0       # ramp
    assert schedule.get_noise_level(200) == 0.08    # estavel
```

### 8.2. Execucao

```bash
# Local (antes de cada commit)
pytest tests/ -v --tb=short

# CI (automatico em cada push/PR)
# Definido em .github/workflows/ci.yml

# GPU (no Colab, manual, apos merge em develop)
# Treinar 5 epocas → comparar val_loss com baseline
```

---

## 9. Documentacao

### 9.1. Hierarquia

```
PRIORIDADE 1: docs/ARCHITECTURE_v2.md    ← ESTE DOCUMENTO (arquitetura geral)
PRIORIDADE 2: CLAUDE.md                  ← Regras e proibicoes para Claude Code
PRIORIDADE 3: configs/*.yaml             ← Presets auto-documentados
PRIORIDADE 4: Docstrings no codigo       ← API reference (Google-style PT-BR)
PRIORIDADE 5: docs/physics/              ← Contexto fisico (tensor EM, GS, FV)
PRIORIDADE 6: notebooks/*.ipynb          ← Exemplos de uso
PRIORIDADE 7: legacy/Skill/             ← Guias historicos (referencia)
```

### 9.2. CLAUDE.md Simplificado para v2.0

```
1. Identidade do Projeto (mantida)
2. Proibicoes Absolutas (mantida)
3. Valores Fisicos Criticos — Errata (mantida)
4. Arquitetura de Software → ARCHITECTURE_v2.md
5. Code Patterns: PipelineConfig, Factory, DataPipeline
6. Testes Obrigatorios: pytest antes de commit
7. GitHub: branching, CI, releases
```

---

## 10. Valores Fisicos Criticos (Errata Imutavel)

Estes valores sao validados pelo `PipelineConfig.__post_init__()` e pelo CI:

```python
assert FREQUENCY_HZ == 20000.0        # NUNCA 2.0 Hz
assert SPACING_METERS == 1.0          # NUNCA 1000.0 m
assert SEQUENCE_LENGTH == 600         # NUNCA 601
assert TARGET_SCALING == "log10"      # NUNCA "log"
assert INPUT_FEATURES == [1,4,5,20,21]  # NUNCA [0,3,4,7,8]
assert OUTPUT_TARGETS == [2,3]        # NUNCA [1,2]
assert EPS_TF == 1e-12               # NUNCA 1e-30 (float32)

# Decoupling (L = SPACING_METERS = 1.0 m):
#   ACp = -1/(4π×L³) ≈ -0.079577  (planar: Hxx, Hyy)
#   ACx = +1/(2π×L³) ≈ +0.159155  (axial: Hzz)
#   Relacao: ACx = -2×ACp

# Formato 22-col (ATIVO):
#   Col 0=meds (meta), Col 1=zobs, Col 2=res_h, Col 3=res_v
#   Cols 4-21: tensor EM 3×3 (9 componentes × Re/Im)
#   Col 4/5=Re/Im(Hxx), Col 20/21=Re/Im(Hzz)
```

---

## 11. Resumo de Design Decisions

| Decisao | Justificativa |
|:--------|:-------------|
| **PipelineConfig dataclass** | Ponto unico de verdade, elimina 574 globals().get() |
| **Pacote Python + pip install** | Testavel, versionavel, IDE-friendly, Colab-compatible |
| **GitHub + CI + tags** | Reprodutibilidade, rollback, colaboracao, deploy automatico |
| **On-the-fly exclusivo** | Unico path fisicamente correto para FV+GS com noise |
| **Factory pattern** | Extensivel, encapsulado, elimina cascatas if/elif |
| **DataPipeline class** | Cadeia raw→noise→FV→GS→scale, sem double-processing |
| **Notebook fino (~15 celulas)** | Legivel, reprodutivel, impossivel executar errado |
| **Testes unitarios** | Detectar regressoes antes do treino |
| **Presets YAML** | Um arquivo define todo o experimento |
| **Migracao incremental** | Coexistencia legado + pacote durante transicao |
| **Geosteering como modo** | DRY — flag no config, nao modulo duplicado |

---

## 12. Contagens de Referencia

| Componente | Quantidade |
|:-----------|:---------:|
| Arquiteturas (ModelRegistry) | 44 (39 standard + 5 geosteering) |
| Nativas causais | 6 (WaveNet, Causal_Transformer, TCN, Mamba_S4, LSTM, Encoder_Forecaster) |
| Causais incompativeis (offline-only) | 17 (BiLSTM, U-Nets, DeepONet) |
| Causais adaptaveis | 21 |
| Funcoes de perda (LossFactory) | 26 (13 gen + 4 geo + 9 adv) |
| Feature Views | 6 (identity, H1_logH2, logH1_logH2, 3× fase/razao) |
| Familias de Geosinais | 5 (USD, UAD, UHR, UHA, U3DF) |
| Tipos de ruido (on-the-fly) | 4 (gaussian, multiplicative, uniform, dropout) |
| Perspectivas | P1-P5 (baseline → Picasso/DTB) |
| Solucoes obrigatorias | P1-P8 (split modelo, dual val, per-group, ...) |

---

*Documento gerado em 2026-03-25 | Geosteering AI v5.0.15 → v2.0*
*Autor: Daniel Leal | Assistente: Claude Code*
*Repositorio: github.com/daniel-leal/geosteering-ai*
