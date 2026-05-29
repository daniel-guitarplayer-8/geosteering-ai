# Codebase Structure

**Analysis Date:** 2026-05-24

## Directory Layout (Root)

```
Geosteering_AI/                         ← Raiz do repositório (cwd ao lançar `claude`)
├── geosteering_ai/                     ← Pacote pip-installable principal (~97k LOC)
├── tests/                              ← Suíte pytest (90 arquivos test_*.py)
├── configs/                            ← Presets YAML de PipelineConfig (6 arquivos)
├── docs/                               ← Documentação técnica e decisional
├── Fortran_Gerador/                    ← Simulador Fortran (tatu.x) + Makefile
├── notebooks/                          ← Jupyter/Colab (benchmarks + validação GPU)
├── tools/                              ← Servidores MCP (3 servidores)
├── benchmarks/                         ← Scripts de benchmark de performance
├── scripts/                            ← Utilitários de suporte (count pytest, etc.)
├── sm_experiments/                     ← Resultados de experimentos SM (.exp.json)
├── sm_output/                          ← Saída binária Fortran gerada (.dat/.out)
├── tutorials/                          ← Tutoriais de uso do pacote
├── legacy/                             ← Código legado C0-C47 (referência histórica)
├── old_geosteering_ai/                 ← Snapshot pré-v2.0 (referência para SM)
├── .claude/                            ← Automação Claude Code (hooks, skills, templates)
├── .agents/                            ← Regras de agentes externos (Antigravity RTK)
├── .github/workflows/                  ← CI/CD GitHub Actions (ci.yml, docker.yml)
├── .planning/codebase/                 ← Documentos de análise GSD (este arquivo)
├── pyproject.toml                      ← Manifesto do pacote + entry points + pytest
├── CLAUDE.md                           ← Instruções do projeto para Claude Code
├── Dockerfile.cpu                      ← Imagem Docker CPU para deploy
├── .pre-commit-config.yaml             ← Hooks pre-commit (ruff, mypy, local)
├── .mcp.json                           ← Configuração dos servidores MCP
└── w{0-3}_sim_batch.dat                ← Dados de simulação em cache (~2 GB cada)
```

## Pacote Principal: `geosteering_ai/`

```
geosteering_ai/
├── __init__.py
├── config.py                    ← PipelineConfig dataclass (1615 LOC, 246 campos)
│
├── data/                        ← Carregamento, split, feature views, pipeline
│   ├── loading.py               ← Carrega .dat Fortran em tensores numpy
│   ├── splitting.py             ← Split por modelo geológico (P1 — nunca por amostra)
│   ├── feature_views.py         ← 7 feature views (FV) — transformações de entrada
│   ├── geosignals.py            ← 5 modos GeoSteering (GS) — features contextuais
│   ├── scaling.py               ← StandardScaler fit em dados limpos (sem ruído)
│   ├── pipeline.py              ← DataPipeline — cadeia raw→noise→FV→GS→scale (1120 LOC)
│   ├── boundaries.py            ← DTB/DFB P5 (distância às fronteiras)
│   ├── inspection.py            ← Inspeção e estatísticas de dataset
│   ├── sampling.py              ← Amostragem estratificada
│   ├── second_order.py          ← Features de segunda ordem (gradientes)
│   ├── surrogate_data.py        ← Dados de treino para SurrogateNet
│   └── synthetic_generator.py  ← Gerador de modelos geológicos sintéticos
│
├── noise/                       ← Ruído on-the-fly
│   ├── functions.py             ← 34 tipos de ruído (2329 LOC)
│   └── curriculum.py           ← Curriculum 3-phase (fases crescentes de ruído)
│
├── models/                      ← 48 arquiteturas DL (9 famílias)
│   ├── registry.py              ← ModelRegistry — factory central (886 LOC)
│   ├── blocks.py                ← 23 blocos reutilizáveis (1952 LOC)
│   ├── cnn.py                   ← Família CNN (ResNet-18/34, TCN, etc.)
│   ├── rnn.py                   ← Família RNN/LSTM
│   ├── tcn.py                   ← TCN e ModernTCN
│   ├── transformer.py           ← Família Transformer
│   ├── unet.py                  ← Família U-Net
│   ├── hybrid.py                ← Modelos híbridos (ResNeXt_LSTM, etc.)
│   ├── decomposition.py         ← Decomposição espectral
│   ├── advanced.py              ← INN, ResNeXt, arquiteturas avançadas
│   ├── geosteering.py           ← Modelos específicos para geosteering
│   └── surrogate.py             ← SurrogateNet TCN + ModernTCN
│
├── losses/                      ← 26 loss functions + PINNs
│   ├── catalog.py               ← Catálogo completo de losses (1165 LOC)
│   ├── factory.py               ← LossFactory — seleção por config
│   └── pinns.py                 ← 8 cenários PINN + TIVConstraintLayer (1975 LOC)
│
├── training/                    ← Loop de treino + callbacks
│   ├── loop.py                  ← TrainingLoop principal (1151 LOC)
│   ├── callbacks.py             ← 17+ callbacks Keras (3538 LOC — maior módulo DL)
│   ├── metrics.py               ← Métricas customizadas TensorFlow
│   ├── nstage.py                ← Treino N-Stage progressivo
│   ├── optuna_hpo.py            ← Integração Optuna para HPO
│   └── adaptation.py           ← Adaptação de domínio
│
├── inference/                   ← Pipeline de inferência
│   ├── pipeline.py              ← InferencePipeline offline
│   ├── realtime.py              ← Inferência em tempo real (sliding window)
│   ├── export.py                ← Export TFLite / ONNX / SavedModel
│   └── uncertainty.py          ← UQ: MC Dropout + Ensemble + INN
│
├── evaluation/                  ← Métricas e relatórios
│   ├── metrics.py               ← Métricas geofísicas de inversão
│   ├── comparison.py            ← Comparação entre modelos
│   ├── geosteering_metrics.py   ← Métricas específicas de geosteering
│   ├── geosteering_report.py    ← Relatório de geosteering
│   ├── dod.py                   ← DOD Picasso 6 métodos
│   ├── predict.py               ← Predição batch com métricas
│   ├── realtime_comparison.py   ← Comparação realtime vs offline
│   ├── advanced.py              ← Métricas avançadas
│   ├── config_report.py         ← Relatório de configuração
│   ├── report.py                ← Relatório geral de avaliação
│   └── manifest.py             ← Manifesto de experimentos
│
├── visualization/               ← Plots e visualizações
│   ├── eda.py                   ← Análise exploratória de dados (1323 LOC)
│   ├── training.py              ← Curvas de treinamento
│   ├── holdout.py               ← Visualização de holdout
│   ├── error_maps.py            ← Mapas de erro geoespaciais
│   ├── geosteering.py           ← Visualização de trajetória geosteering
│   ├── picasso.py               ← Visualizador Picasso
│   ├── realtime.py              ← Plot realtime
│   ├── uncertainty.py           ← Visualização de incerteza (UQ)
│   ├── optuna_viz.py            ← Dashboard Optuna (1231 LOC)
│   └── export.py               ← Export de figuras
│
├── simulation/                  ← Simulador EM 1D (Python — 4 backends)
│   ├── config.py                ← SimulationConfig dataclass (1247 LOC)
│   ├── forward.py               ← Forward pass de posição única
│   ├── multi_forward.py         ← Forward multi-posição / multi-ângulo (1344 LOC)
│   ├── _workers.py              ← Pool efêmero ProcessPoolExecutor + warmup JIT
│   ├── _jacobian.py             ← Jacobiano ∂H/∂ρ numérico
│   │
│   ├── _numba/                  ← Backend Numba JIT (caminho crítico de produção)
│   │   ├── kernel.py            ← Kernel principal cached Numba (1052 LOC)
│   │   ├── propagation.py       ← Propagação de ondas EM em camadas
│   │   ├── dipoles.py           ← Dipolos magnéticos HMD/VMD
│   │   ├── geometry.py          ← Geometria de ferramentas LWD
│   │   ├── hankel.py            ← Transformada de Hankel digital
│   │   └── rotation.py         ← Rotações de tensor EM (fastmath)
│   │
│   ├── _jax/                    ← Backend JAX (GPU — aceleração XLA)
│   │   ├── kernel.py            ← Kernel JAX via pure_callback → Numba
│   │   ├── multi_forward.py     ← Multi-forward JAX vmap (1250 LOC)
│   │   ├── forward_pure.py      ← Forward pass JAX puro (1379 LOC)
│   │   ├── dipoles_unified.py   ← Dipolos HMD+VMD unified JIT
│   │   ├── dipoles_native.py    ← Dipolos nativos JAX (1995 LOC)
│   │   ├── geometry_jax.py      ← Geometria tracer-safe JAX
│   │   ├── propagation.py       ← Propagação JAX jit/vmap
│   │   ├── hankel.py            ← Hankel JAX
│   │   └── rotation.py         ← Rotações JAX
│   │
│   ├── filters/                 ← Filtros de Hankel digitais (dados empacotados)
│   │   ├── loader.py            ← FilterLoader — carrega .npz por tipo
│   │   ├── werthmuller_201pt.npz ← Filtro padrão 201 pontos
│   │   ├── kong_61pt.npz        ← Filtro rápido 61 pontos (3.3× mais rápido)
│   │   └── anderson_801pt.npz  ← Filtro alta precisão 801 pontos
│   │
│   ├── io/                      ← I/O binário compatível com Fortran
│   │   ├── binary_dat.py        ← Leitura/escrita de .dat single-TR
│   │   ├── binary_dat_multi.py  ← Leitura/escrita de .dat multi-TR vetorizado
│   │   └── model_in.py         ← Parser de model.in (formato Fortran)
│   │
│   ├── postprocess/             ← Pós-processamento de tensores EM
│   │   ├── compensation.py      ← Compensação midpoint (F6)
│   │   └── tilted.py           ← Antenas inclinadas (F7)
│   │
│   ├── validation/              ← Validação física do simulador
│   │   ├── canonical_models.py  ← 10 modelos canônicos (paridade Fortran <1e-12)
│   │   ├── compare_fortran.py   ← Comparação Python vs tatu.x
│   │   ├── compare_analytical.py ← Comparação vs solução analítica half-space
│   │   ├── compare_empymod.py   ← Comparação vs empymod
│   │   └── half_space.py       ← Solução analítica half-space TIV
│   │
│   ├── benchmarks/              ← Benchmarks de throughput do simulador
│   │   └── bench_forward.py    ← Cenários A-H (benchmarks padronizados)
│   │
│   ├── visualization/           ← Plots específicos do simulador
│   │   ├── plot_benchmark.py    ← Plots de throughput
│   │   ├── plot_benchmark_advanced.py
│   │   ├── plot_canonical.py    ← Plots de modelos canônicos
│   │   ├── plot_geophysical.py  ← Plots geofísicos EM
│   │   ├── plot_ml.py           ← Plots ML + simulador integrado
│   │   ├── plot_physics.py      ← Plots de física EM
│   │   └── plot_tensor.py      ← Plots do tensor EM 9 componentes
│   │
│   └── tests/                  ← PyQt6 GUI Simulation Manager (~10.7k LOC)
│       ├── simulation_manager.py ← Janela principal Qt (10759 LOC — maior arquivo)
│       ├── sm_workers.py        ← SimulationThread + pool efêmero (1103 LOC)
│       ├── sm_plots.py          ← Widgets de plot (1772 LOC)
│       ├── sm_qt_compat.py      ← Compatibilidade Qt5/Qt6
│       ├── sm_model_gen.py      ← ModelGenerationThread (async)
│       ├── sm_plot_cache.py     ← LRUPlotCache configurável
│       ├── sm_benchmark.py      ← Widget de benchmark integrado
│       ├── sm_canonical_profiles.py ← Perfis canônicos na GUI
│       ├── sm_correlation.py    ← Análise de correlação
│       ├── sm_dat_viewer.py     ← Visualizador de .dat
│       ├── sm_heartbeat.py      ← MainThreadHeartbeat (anti-freeze)
│       ├── sm_io.py             ← I/O da GUI
│       ├── sm_layers_dialog.py  ← Diálogo de camadas geológicas
│       ├── sm_phase_timer.py    ← PhaseTimer (temporização de fases)
│       ├── sm_animation_bar.py  ← Barra de progresso animada
│       ├── sm_snapshot_persist.py ← Persistência de snapshots
│       ├── sm_toast.py          ← Notificações toast
│       ├── sm_widgets.py        ← Widgets auxiliares
│       └── sm_plot_backends/   ← Backends de plot (mpl, plotly, pyqtgraph, vispy)
│
├── cli/                         ← Comandos de linha de comando
│   ├── _main.py                 ← Entry point principal (geosteering-cli)
│   ├── benchmark.py             ← Subcomando benchmark (Cenários A-H)
│   ├── simulate.py              ← Subcomando simulate
│   ├── warmup.py                ← Entry point geosteering-warmup (aquecimento JIT)
│   └── __main__.py             ← python -m geosteering_ai.cli
│
├── api/                         ← API REST FastAPI
│   ├── app.py                   ← Aplicação FastAPI
│   ├── cli.py                   ← Entry point geosteering-api (uvicorn)
│   ├── dependencies.py          ← Injeção de dependências
│   ├── schemas.py               ← Esquemas Pydantic v2
│   └── routes/                 ← Rotas da API
│       ├── health.py            ← GET /health
│       └── predict.py          ← POST /predict
│
├── multi_agent/                 ← Infraestrutura multi-agente
│   ├── conflict_matrix.py       ← Matriz de conflitos entre agentes (thread-safe)
│   └── lock_manager.py         ← LockManager com PID checks (psutil)
│
└── utils/                       ← Utilitários transversais
    ├── logger.py                ← Logger estruturado (NUNCA usar print())
    ├── timer.py                 ← Timer de operações
    ├── validation.py            ← Validações de entrada
    ├── formatting.py            ← Formatação de saída
    ├── system.py                ← Detecção de CPU/GPU e topologia
    └── io.py                   ← I/O genérico
```

## Simulador Fortran: `Fortran_Gerador/`

```
Fortran_Gerador/
├── PerfilaAnisoOmp.f08          ← Módulo principal TIV + OpenMP (núcleo EM)
├── RunAnisoOmp.f08              ← Runner / orquestrador de simulação
├── magneticdipoles.f08          ← Dipolos magnéticos HMD/VMD Fortran
├── filtersv2.f08                ← Filtros de Hankel (Werthmuller/Kong/Anderson)
├── parameters.f08               ← Parâmetros físicos e constantes
├── utils.f08                    ← Utilitários Fortran
├── tatu_f2py_wrapper.f08        ← Wrapper f2py para chamada Python
├── tatu.x                       ← Binário de produção (gfortran -O3 + OpenMP)
├── tatu_dbg.x                   ← Binário debug (-O0)
├── tatu_{phase2,phase4}_{O0,O3}.x ← Binários de sprints anteriores
├── Makefile                     ← Build system (detecta macOS/Linux, ld-classic)
├── model.in                     ← Modelo geológico de entrada (formato texto)
├── model.in.{n10,n15_*}         ← Modelos canônicos de referência
├── batch_runner.py              ← Runner Python para simulação em lote
├── buildValidamodels.py         ← Construção de modelos de validação
├── fifthBuildTIVModels.py       ← Construção de modelos TIV (5ª geração)
├── validate_jacobian.py         ← Validação do Jacobiano ∂H/∂ρ
├── geradorFortran_fifth/        ← Gerador de datasets (5ª versão)
│   └── lib/                    ← Biblioteca do gerador
├── build/                       ← Arquivos objeto (.o) e módulos (.mod) compilados
│   └── lib/                    ← ld wrapper (ld-classic em macOS)
├── bench/                       ← Scripts de benchmark do Fortran
│   ├── validate_numeric.py      ← Validação numérica rápida
│   ├── validate_numeric_extensive.py ← Validação extensa
│   ├── run_bench.sh             ← Script de benchmark
│   └── attic/                  ← Versões anteriores de scripts
└── validacao_plots/             ← Plots de validação gerados
```

## Testes: `tests/`

```
tests/
├── conftest.py                  ← Fixtures globais + alinhamento QT_API
├── conftest_qt.py               ← Fixtures Qt: qt_binding, mock_simulation_thread
├── __init__.py
│
├── test_config.py               ← Errata PipelineConfig, mutual exclusivity, YAML roundtrip
├── test_models.py               ← Forward pass para todas as 48 arquiteturas
├── test_data_pipeline.py        ← Shapes, split P1, scaler fit em dados limpos
├── test_noise.py                ← Curriculum 3-phase, noise preserva z_obs
├── test_losses.py               ← Forward pass + gradientes para 26 losses
├── test_pinns.py                ← 8 cenários PINN + TIVConstraintLayer
├── test_training.py             ← TrainingLoop, callbacks, N-Stage
├── test_inference.py            ← InferencePipeline, realtime, UQ
├── test_evaluation.py           ← Métricas, comparação
├── test_visualization.py        ← Plots (smoke tests)
├── test_surrogate.py            ← SurrogateNet TCN + ModernTCN
├── test_utils.py                ← Logger, timer, validation, formatting
├── test_boundaries.py           ← DTB/DFB P5
├── test_multi_agent_locks.py    ← LockManager + conflict matrix
│
├── test_api_health.py           ← GET /health
├── test_api_predict.py          ← POST /predict (TestClient httpx)
├── test_api_schemas.py          ← Esquemas Pydantic v2
│
├── test_cli_mvp.py              ← CLI simulate/benchmark/version
├── test_cli_warmup_*.py         ← geosteering-warmup entry point
│
├── test_simulation_*.py         ← ~50 arquivos de teste do simulador
│   ├── test_simulation_config.py
│   ├── test_simulation_forward.py
│   ├── test_simulation_multi.py
│   ├── test_simulation_workers*.py      ← Pool efêmero, threading
│   ├── test_simulation_numba_*.py       ← Kernel, geometry, dipoles, etc.
│   ├── test_simulation_jax_*.py         ← Foundation, multi, sprint10-13, GPU
│   ├── test_simulation_compare_fortran.py ← Paridade Fortran <1e-12
│   ├── test_simulation_canonical_models.py ← 10 modelos canônicos
│   ├── test_simulation_v21*.py          ← Testes de sprints específicas
│   └── test_simulation_manager_gui.py  ← 16 testes pytest-qt (marker: gui)
│
├── test_known_bugs.py           ← 11 testes de bugs conhecidos (KB-001/002/013/018/019)
├── test_legacy_integration.py   ← Integração com código legado
├── test_hooks_i25.py            ← Testes dos hooks I2.5
├── test_sprint_v224.py          ← Testes da sprint v2.24
├── test_strategies_bc.py        ← Estratégias B e C de simulação
├── test_perf_baseline_h.py      ← Cenário H (stress 512 combos)
├── test_worktree_config.py      ← Configuração de worktrees
├── test_pipeline_simulator_backend.py ← Pipeline + backend simulador
└── _fortran_helpers.py          ← Helpers para testes Fortran (não é test_*)
```

## Configuração: `configs/`

```
configs/
├── baseline.yaml                ← Configuração base mínima
├── robusto.yaml                 ← Configuração robusta para produção
├── geosinais_p4.yaml            ← GeoSteering modo P4 (4 features de contexto)
├── nstage_n2.yaml               ← Treino N-Stage com 2 estágios
├── nstage_n3.yaml               ← Treino N-Stage com 3 estágios
└── realtime_causal.yaml         ← Inferência realtime (Conv1D causal)
```

## Documentação: `docs/`

```
docs/
├── INDEX.md                     ← Porta de entrada — "onde encontro X?" (SSoT)
├── ARCHITECTURE_v2.md           ← Arquitetura completa v2.0 (documento principal)
├── ROADMAP.md                   ← Backlog priorizado (SSoT do futuro — §0 é canônico)
├── CHANGELOG.md                 ← Releases Keep-a-Changelog
├── PERFORMANCE_BASELINE.md      ← Baselines de throughput por cenário (A-H)
├── MIGRATION_GUIDE.md           ← Guia de migração legado → v2.0
├── known_bugs.md                ← Bugs conhecidos KB-001/002/013/018/019
├── documentacao_*.md            ← Docs por módulo (losses, models, noises, pinns, etc.)
│
├── decisions/                   ← Architectural Decision Records
│   ├── ADR-0001-arquitetura-documentacao.md ← ADR principal (hierarquia de planejamento)
│   └── archive/                ← ADRs arquivados
│
├── sprints/                     ← Snapshots imutáveis de sprints
│   ├── CURRENT.md               ← Sprint em execução
│   ├── v2.40.md / v2.41.md / v2.42.md / v2.43.md ← Sprints concluídas
│   └── v2.40.1.md              ← Hotfix
│
├── reports/                     ← Relatórios técnicos gerados (v{VERSION}_{DATE}.md)
│
├── physics/                     ← Contexto físico do domínio
│   ├── errata_valores.md        ← Errata de valores físicos críticos
│   ├── onboarding.md            ← Introdução ao domínio geofísico
│   └── perspectivas.md         ← Perspectivas de pesquisa
│
├── perf_baselines/              ← Baselines de performance históricos
│
└── reference/                  ← Catálogos de referência técnica
    ├── arquiteturas_detalhado.md
    ├── arquiteturas_resumo.md
    ├── losses_catalog.md
    ├── noise_catalog.md
    ├── modelo_geologico.md
    ├── second_order_features.md
    ├── plano_simulador_python_jax_numba.md
    ├── analise_cenarios_otimizacao_simulador_numba.md
    ├── relatorio_*.md           ← Relatórios de sprints do simulador
    └── sprint_*.md             ← Detalhes de sprints específicas
```

## Automação Claude Code: `.claude/`

```
.claude/
├── settings.json                ← Configuração principal (hooks, permissões)
├── settings.local.json          ← Overrides locais (não versionado)
├── anti-patterns.txt            ← 13 padrões proibidos (TSV 4 colunas)
├── ptbr-words.txt               ← 60+ pares PT-BR para accentuação obrigatória
├── perf_baseline.json           ← Baselines de performance por cenário CLI
├── parallelism_rules.py         ← Regras de paralelismo para agentes
├── recovery.sh                  ← Script de recuperação de estado
├── state.json.template          ← Template de estado para agentes
│
├── commands/                   ← Skills / comandos slash disponíveis
│   ├── geosteering-v2.md        ← Skill PRINCIPAL — domínio físico + padrões v2.0
│   ├── geosteering-orchestrator.md ← Orquestrador multi-agente
│   ├── geosteering-simulator-python.md ← Skill simulador Python (JAX+Numba)
│   ├── geosteering-simulator-fortran.md ← Skill simulador Fortran
│   ├── geosteering-simulator-numba.md ← Skill Numba JIT
│   ├── geosteering-simulation-manager.md ← Skill GUI Simulation Manager
│   ├── geosteering-physics-reviewer.md ← Revisor de física EM
│   ├── geosteering-perf-reviewer.md ← Revisor de performance
│   ├── geosteering-code-reviewer.md ← Revisor de código
│   ├── geosteering-premortem-analyst.md ← Análise pré-mortem
│   ├── consensus-search.md      ← Pesquisa científica Semantic Scholar + ArXiv
│   ├── arxiv-search.md          ← Busca ArXiv (sem API key)
│   └── geosteering-{data,jax,losses,models,physics,pinns,realtime,research}.md
│
├── hooks/                       ← Hooks de automação (PreToolUse / PostToolUse)
│   ├── backup-pre-edit.sh       ← Backup automático antes de editar
│   ├── check-anti-patterns.sh   ← Bloqueia padrões proibidos (BLOCK/WARN)
│   ├── validate-no-pytorch.sh   ← Bloqueia PyTorch em production paths
│   ├── validate-physics.sh      ← Valida valores físicos críticos
│   ├── check-ptbr-accentuation.sh ← Valida acentuação PT-BR obrigatória
│   ├── check-perf-regression.sh ← Alerta de regressão de performance (WARN)
│   ├── check-version-references.sh ← Enforça R1 ADR-0001 (versões em ROADMAP)
│   ├── run-fortran-parity.sh    ← Executa paridade <1e-12 vs tatu.x
│   ├── run-pytest.sh            ← Executa pytest após edição
│   ├── compile-check.sh         ← Verifica compilação Python
│   ├── autoformat.sh            ← ruff format
│   ├── generate-pr-description.sh ← Geração automática de PR description
│   ├── lint-v2-standards.sh     ← Validação padrões D1-D14
│   ├── protect-critical-files.sh ← Protege arquivos críticos
│   ├── acquire-lock.sh / release-lock.sh ← Locks multi-agente
│   ├── reinject-errata.sh       ← Reinjeta errata em config.py
│   ├── setup-environment.sh     ← Configura ambiente Python
│   └── colab-token-refresh.sh  ← Refresh de token Colab MCP
│
├── templates/                   ← Templates de documentos
│   ├── report_template.md       ← Template obrigatório para relatórios MD
│   └── pr_description_template.md ← Template de PR description
│
├── scripts/
│   └── worktree-create.sh       ← Criação de git worktrees
│
└── telemetry/
    └── parallelism_dashboard.py ← Dashboard de paralelismo de agentes
```

## Ferramentas MCP: `tools/`

```
tools/
├── file_watcher_daemon.py       ← Daemon de monitoramento de arquivos (watchdog)
├── cleanup_backups.sh           ← Limpeza de .backups/ antigos
│
├── consensus-mcp-server/        ← MCP Server: pesquisa científica
│   ├── server.py                ← 6 tools (search_papers, etc.)
│   └── requirements.txt
│
├── physics-validator-mcp/       ← MCP Server: validação física EM
│   ├── server.py                ← 6 tools (validate_config, etc.)
│   ├── tests/
│   └── requirements.txt
│
└── numba-profiler-mcp/          ← MCP Server: profiling Numba JIT
    ├── server.py                ← 6 tools (profile_jit, etc.)
    ├── tests/
    └── requirements.txt
```

## Notebooks: `notebooks/`

```
notebooks/
├── main.ipynb                   ← Notebook principal de treino
├── validate_gpu_colab.ipynb     ← Validação GPU (824 passed, Colab Pro+ T4)
├── validate_jax_unified_gpu.ipynb ← Validação JAX unified GPU
├── bench_*.ipynb                ← Benchmarks JAX GPU (PR #14b/d/e/f)
├── benchmark_jax_gpu_vs_numba.ipynb ← Comparação JAX GPU vs Numba
├── sprint_3_3_4_2_validation.ipynb ← Validação sprints 3.3/4.2
│
└── colab_templates/             ← Templates para Colab Pro+ GPU
    ├── __README.md
    ├── benchmark_tfdata_mp16.ipynb ← Benchmark tf.data multiprocessing 16 cores
    ├── train_v240_mp16.ipynb    ← Treino v2.40 MP16
    ├── validate_jax_gpu_v240.ipynb ← Validação JAX GPU v2.40
    ├── validate_sprint_o0_o1_gpu.ipynb ← Validação Sprint O0/O1 T4/A100
    └── validate_sprint_o1_gpu_tests.ipynb ← Testes Sprint O1 GPU
```

## Benchmarks: `benchmarks/`

```
benchmarks/
├── bench_forward_colab.ipynb    ← Benchmark forward Colab
├── bench_multi_vs_fortran.py    ← Python multi vs Fortran
├── bench_numba_vs_fortran_local.py ← Numba vs Fortran local
├── bench_sprint10_unified_vs_bucketed.py
├── bench_sprint12_regression.py ← Regressão Sprint 12
├── bench_v212_workers.py        ← Workers v2.12
├── bench_v214_numba.py          ← Numba v2.14
├── bench_v22_flat_prange.py     ← FLAT prange v2.22
├── bench_v236_tile_block.py     ← Tile block v2.36
├── bench_v237_cross_platform.py ← Cross-platform v2.37
├── bench_v237_flat_heuristic.py ← Flat heuristic v2.37
├── profile_freezing_baseline.py ← Profiling de freeze GUI
└── results/                    ← Resultados salvos de benchmarks
```

## Convenções de Nomenclatura

**Arquivos Python:**
- `snake_case.py` para todos os módulos: `feature_views.py`, `multi_forward.py`
- Prefixo `_` para backends privados: `_numba/`, `_jax/`, `_workers.py`, `_jacobian.py`
- Prefixo `sm_` para módulos do Simulation Manager: `sm_workers.py`, `sm_plots.py`
- Prefixo `test_` obrigatório para testes pytest
- Prefixo `bench_` para scripts de benchmark

**Identificadores Python:**
- Variáveis e funções: `snake_case` em inglês
- Classes: `PascalCase` em inglês
- Constantes: `UPPER_SNAKE_CASE` em inglês

**Comentários e docstrings:**
- Sempre em PT-BR com acentuação obrigatória
- `implementação`, `configuração`, `função` — nunca `implementacao`

**Diretórios:**
- Pacotes Python: `snake_case/`
- Diretórios de configuração: `.claude/`, `.agents/`, `.github/`
- Diretórios históricos: `Arquivos_Projeto_Claude/`, `Relatorio_2025/` (PascalCase legado)

**Arquivos de documentação:**
- Relatórios técnicos: `docs/reports/v{VERSION}_{YYYY-MM-DD}.md`
- Sprints: `docs/sprints/v{X}.{Y}.md`
- ADRs: `docs/decisions/ADR-{NNNN}-{slug}.md`
- Skills: `.claude/commands/{skill-name}.md`

**Fortran:**
- Módulos: `PascalCase.f08` (ex: `PerfilaAnisoOmp.f08`, `RunAnisoOmp.f08`)
- Binários: `tatu{_variant}.x`

## Onde Adicionar Novo Código

### Nova arquitetura de modelo DL
- Implementação: `geosteering_ai/models/{familia}.py` (ou `advanced.py` para arquiteturas únicas)
- Registro: adicionar entrada em `geosteering_ai/models/registry.py` (ModelRegistry)
- Teste de forward pass: `tests/test_models.py`
- Documentação: `docs/reference/arquiteturas_detalhado.md`

### Nova loss function
- Implementação: `geosteering_ai/losses/catalog.py` (catálogo existente)
- Factory: atualizar `geosteering_ai/losses/factory.py` (LossFactory)
- Testes: `tests/test_losses.py`
- Documentação: `docs/reference/losses_catalog.md`

### Novo tipo de ruído
- Implementação: `geosteering_ai/noise/functions.py` (catálogo existente de 34 tipos)
- Integração no currículo: `geosteering_ai/noise/curriculum.py`
- Testes: `tests/test_noise.py`
- Documentação: `docs/reference/noise_catalog.md`

### Novo campo em PipelineConfig
- Implementação: `geosteering_ai/config.py` (PipelineConfig.__post_init__ valida)
- Preset YAML (se novo preset): `configs/{nome}.yaml`
- Testes de errata: `tests/test_config.py`

### Nova feature view (FV) ou GeoSignal (GS)
- Feature View: `geosteering_ai/data/feature_views.py`
- GeoSignal: `geosteering_ai/data/geosignals.py`
- Testes: `tests/test_data_pipeline.py`

### Nova rota na API REST
- Rota: `geosteering_ai/api/routes/{nome}.py`
- Schema Pydantic: `geosteering_ai/api/schemas.py`
- Registro: `geosteering_ai/api/app.py` (include_router)
- Testes: `tests/test_api_{nome}.py`

### Novo subcomando CLI
- Implementação: `geosteering_ai/cli/{nome}.py`
- Registro: `geosteering_ai/cli/_main.py` + entry point em `pyproject.toml`
- Testes: `tests/test_cli_*.py`

### Nova otimização do simulador Numba
- Kernel: `geosteering_ai/simulation/_numba/kernel.py`
- Dipoles/geometry/propagation: respectivos módulos em `_numba/`
- Testes de paridade Fortran: `tests/test_simulation_compare_fortran.py`
- Testes de especialização JIT: `tests/test_simulation_numba_specializations.py`

### Nova funcionalidade JAX/GPU
- Kernel/forward: `geosteering_ai/simulation/_jax/`
- Testes: `tests/test_simulation_jax_{nome}.py`
- Benchmark Colab: `notebooks/colab_templates/validate_{nome}_gpu.ipynb`

### Nova validação física
- Validação: `geosteering_ai/simulation/validation/{tipo}.py`
- Testes: `tests/test_simulation_{tipo}.py`

### Novo hook Claude Code
- Script: `.claude/hooks/{nome}.sh`
- Registro: `.claude/settings.json` (seção hooks)

### Nova skill / agente
- Skill: `.claude/commands/{geosteering-nome}.md`
- Segue formato das skills existentes (ver `geosteering-v2.md`)

### Novo servidor MCP
- Diretório: `tools/{nome}-mcp/`
- Arquivo: `tools/{nome}-mcp/server.py` (6 tools mínimo)
- Testes: `tools/{nome}-mcp/tests/`
- Registro: `.mcp.json`

### Novo relatório técnico
- Arquivo: `docs/reports/v{VERSION}_{YYYY-MM-DD}.md`
- Template: `.claude/templates/report_template.md`
- Changelog: adicionar entrada em `docs/CHANGELOG.md`

### Novo benchmark de performance
- Script: `benchmarks/bench_{nome}.py`
- Cenário padronizado: adicionar em `geosteering_ai/simulation/benchmarks/bench_forward.py`
- Baseline: `.claude/perf_baseline.json` + `docs/PERFORMANCE_BASELINE.md`

## Diretórios Especiais

**`.backups/`**
- Propósito: Backups automáticos criados pelo hook `backup-pre-edit.sh` antes de cada edição
- Gerado: Sim (pelo hook PreToolUse)
- Committed: Não (em `.gitignore`)
- Limpeza: `tools/cleanup_backups.sh`

**`.claude/locks/`**
- Propósito: Locks de arquivo para coordenação multi-agente
- Gerado: Sim (pelos hooks acquire-lock.sh / release-lock.sh)
- Committed: Não

**`.remember/`**
- Propósito: Memória persistente de sessões Claude Code
- Gerado: Sim (pelo Claude Code)
- Committed: Não (em `.gitignore`)

**`geosteering_ai.egg-info/`**
- Propósito: Metadados de instalação editable (`pip install -e .`)
- Gerado: Sim
- Committed: Não

**`sm_output/`**
- Propósito: Saída binária gerada pelo Fortran ou pelo GUI Simulation Manager
- Gerado: Sim
- Committed: Não

**`sm_experiments/`**
- Propósito: Resultados de experimentos do Simulation Manager (.exp.json, .csv)
- Gerado: Sim (parcialmente)
- Committed: Parcialmente (sumários .csv e .md)

**`old_geosteering_ai/`**
- Propósito: Snapshot pré-v2.0, usado como referência para o Simulation Manager
- Gerado: Não
- Committed: Sim (referência histórica — NÃO modificar)

**`legacy/`**
- Propósito: Código legado C0-C47 (pré-v2.0)
- Gerado: Não
- Committed: Sim (referência histórica — NÃO modificar)

---

*Análise de estrutura: 2026-05-24*
