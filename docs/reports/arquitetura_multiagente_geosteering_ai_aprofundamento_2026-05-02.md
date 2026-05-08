# Arquitetura Multi-Agente Aprofundada — Geosteering AI 2.0
## Documento-Base de Construção do Software Profissional

<!-- Metadados do Documento -->

| Campo | Valor |
|:------|:------|
| **Versão do documento** | 1.0 (aprofundamento sobre `arquitetura_multiagente_geosteering_ai_2026-05-02.md`) |
| **Data** | 2026-05-02 |
| **Modelo gerador** | Claude Opus 4.7 (1M tokens de contexto) |
| **Plano** | Claude Max 5× + Google AI Pro |
| **IDE alvo** | Google Antigravity + extensão Claude Code |
| **Versão do projeto** | Geosteering AI v2.0 + Simulation Manager v2.21 |
| **Versão do simulador** | Python `geosteering_ai.simulation` v1.6.0 / Fortran `tatu.x` v10.0 |
| **Linguagem** | PT-BR com acentuação garantida; identificadores em inglês |
| **Status** | Documento-base — guia oficial para construção da arquitetura |
| **Documentos antecessores** | `docs/ARCHITECTURE_v2.md`, `docs/ROADMAP.md`, `docs/reference/analise_cenarios_otimizacao_simulador_numba.md`, `docs/reports/arquitetura_multiagente_geosteering_ai_2026-05-02.md` |

---

## Sumário Executivo

Este documento é a **base oficial e aprofundada** que orienta a construção
da arquitetura de software profissional do projeto **Geosteering AI 2.0**.
Substitui e expande o relatório-briefing anterior
(`arquitetura_multiagente_geosteering_ai_2026-05-02.md`), incorporando:

1. **Levantamento exaustivo** do estado atual do projeto (~46.000 LOC
   Python + ~6.900 LOC Fortran + ~8.000 LOC documentação técnica) com
   mapeamento componente-a-componente.
2. **Topologia multi-agente** com 17 agentes especializados organizados
   em 5 camadas hierárquicas (orquestração, domínio, qualidade,
   infraestrutura, pesquisa).
3. **Catálogo completo** de hooks, MCP servers, skills, worktrees e
   templates a serem implementados.
4. **Workflows end-to-end** para os principais fluxos do projeto:
   simulador (Numba + JAX) em todos os cenários; treinamento DL com
   PINNs; inferência em dados reais; geosteering em tempo real;
   MLOps em produção.
5. **Roteiro evolutivo** para 2D (Born + MEF), 2.5D e 3D (MEF) sem
   reescrita arquitetural.
6. **Critérios de aceitação** mensuráveis para cada fase, garantindo
   precisão física/geofísica/petrofísica e qualidade de produção
   industrial.

### Resumo das Decisões Arquiteturais Centrais

| Decisão | Escolha | Justificativa |
|:--------|:--------|:--------------|
| **Modelo de orquestração** | Hub-and-spoke com Opus 4.7 1M no centro | Único modelo capaz de carregar projeto inteiro + histórico em 1 contexto |
| **Distribuição de trabalho** | Sonnet 4.6 implementa, Haiku 4.5 automatiza, Opus 4.7 raciocina | Custo/benefício mensurável: 70% Sonnet + 25% Haiku + 5% Opus |
| **Isolamento de sprints** | `claude --worktree` 1:1 com cada feature ativa | Previne contaminação cruzada de mudanças e cache (Numba/Python) |
| **Qualidade contínua** | Hooks determinísticos (shell) + LLM apenas onde necessário | Custo zero para verificações triviais, LLM somente para julgamento |
| **Extensão do Claude Code** | MCP Servers locais (stdio) para domínio + connectors HTTP para nuvem | Latência mínima domínio, integração rica com infra externa |
| **Pesquisa científica** | Multi-agente coordenado: Consensus + ArXiv + bioRxiv + WebSearch | Cobertura completa de literatura, citações inline obrigatórias |
| **Documentação** | Multi-agente Haiku-driven com gates Sonnet | Geração econômica, validação onde rigor importa |
| **Frontend (GUI/CLI)** | PyQt6 mantido + nova CLI `geosteering-cli` baseada em Typer | Reuso da GUI existente; CLI alinha com workflow MLOps |
| **Empacotamento** | `pip install geosteering-ai` (PyPI privado/público) com extras | API Python idiomática (`from geosteering_ai import ...`) |
| **Preparação 2D/2.5D/3D** | Backend abstrato `simulation.backends.*` com plugin discovery | Permite adicionar Born, MEF sem alterar interface |

### Constraints Invioláveis (carregados em cada agente)

```
F1. TensorFlow/Keras EXCLUSIVO; PyTorch é proibido em geosteering_ai/.
F2. Paridade Fortran <1e-12 em 7 modelos canônicos; gate em CI.
F3. Anti-pattern de paralelismo aninhado em Numba (lição v2.13→v2.21).
F4. Errata física imutável: frequency_hz=20000.0, spacing_meters=1.0,
    sequence_length=600, target_scaling="log10", input_features=[1,4,5,20,21],
    output_targets=[2,3], eps_tf=1e-12.
F5. Cadeia de dados raw→noise→FV→GS→scale (on-the-fly exclusivo;
    fitar scaler em dados LIMPOS).
F6. Split por modelo geológico; nunca por amostra.
F7. PT-BR com acentuação correta em comentários, docstrings e MD.
F8. Logging estruturado (`logger.*`); print() proibido em geosteering_ai/.
F9. Toda função recebe `config: PipelineConfig` ou `cfg: SimulationConfig`;
    globals() é proibido.
F10. Padrões D1-D14 de documentação (mega-header, docstrings Google-style,
     diagramas ASCII em catálogos, riqueza C28).
```

---

## Sumário do Documento

1. [Levantamento Completo do Projeto (Snapshot 2026-05-02)](#1-levantamento-completo-do-projeto-snapshot-2026-05-02)
2. [Princípios Arquiteturais e Modelo Mental](#2-princípios-arquiteturais-e-modelo-mental)
3. [Topologia Multi-Agente — Cinco Camadas](#3-topologia-multi-agente--cinco-camadas)
4. [Catálogo de Agentes Especializados](#4-catálogo-de-agentes-especializados)
5. [Agente de Pesquisa Científica](#5-agente-de-pesquisa-científica)
6. [Agente de Documentação Automatizada](#6-agente-de-documentação-automatizada)
7. [Agentes do Simulador — Numba + JAX](#7-agentes-do-simulador--numba--jax)
8. [Agentes de Deep Learning + PINNs](#8-agentes-de-deep-learning--pinns)
9. [Agentes de Dados — Sintéticos e Reais](#9-agentes-de-dados--sintéticos-e-reais)
10. [Agentes de Geosteering em Tempo Real](#10-agentes-de-geosteering-em-tempo-real)
11. [Agentes de MLOps + Produção](#11-agentes-de-mlops--produção)
12. [Agentes de Segurança e Qualidade](#12-agentes-de-segurança-e-qualidade)
13. [Frontend (GUI PyQt6) e CLI](#13-frontend-gui-pyqt6-e-cli)
14. [Distribuição como Módulo/API Python](#14-distribuição-como-móduloapi-python)
15. [Hooks — Automação Determinística](#15-hooks--automação-determinística)
16. [MCP Servers — Catálogo Completo](#16-mcp-servers--catálogo-completo)
17. [Skills (Slash Commands) — Catálogo Detalhado](#17-skills-slash-commands--catálogo-detalhado)
18. [Worktrees, Branching e Isolamento](#18-worktrees-branching-e-isolamento)
19. [Seleção de Modelos LLM por Tarefa](#19-seleção-de-modelos-llm-por-tarefa)
20. [Workflows Orquestrados End-to-End](#20-workflows-orquestrados-end-to-end)
21. [Preparação para Simuladores 2D/2.5D/3D](#21-preparação-para-simuladores-2d25d3d)
22. [Roadmap de Implementação da Infraestrutura](#22-roadmap-de-implementação-da-infraestrutura)
23. [Estrutura Completa de Configuração](#23-estrutura-completa-de-configuração)
24. [Análise de Riscos, Custo e Mitigações](#24-análise-de-riscos-custo-e-mitigações)
25. [Critérios de Aceitação e Métricas](#25-critérios-de-aceitação-e-métricas)

---

## 1. Levantamento Completo do Projeto (Snapshot 2026-05-02)

### 1.1 Identidade e Missão

| Atributo | Valor |
|:---------|:------|
| Projeto | Inversão 1D de Resistividade via Deep Learning para Geosteering |
| Versão de software | v2.0 (arquitetura) + Simulation Manager v2.21 |
| Repositório | `github.com/daniel-guitarplayer-8/geosteering-ai` |
| Branch ativa | `feat/simulation-manager-v2.17` (linha v2.17→v2.21) |
| Pacote pip | `geosteering-ai` (instalável via `pip install -e .`) |
| Python | 3.13 (CI 3.12 fallback) |
| Framework DL | TensorFlow 2.x / Keras (exclusivo) |
| Hardware-alvo | Mac M-series 8C/16T (dev) + Google Colab Pro+ T4/A100 (treino) |
| Status | Beta avançado / aproximando-se de produção |

### 1.2 Métricas Quantitativas Globais

```
╔══════════════════════════════════════════════════════════════════════════╗
║  PROJETO GEOSTEERING AI v2.0 — MÉTRICAS GLOBAIS (2026-05-02)            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Pacote principal:                                                       ║
║    Python (LOC produção)............... ~46.000                          ║
║    Python (LOC testes)................. ~9.000                           ║
║    Arquivos Python (src)............... 73                               ║
║    Arquivos Python (tests)............. 59 (em /tests/)                  ║
║    Subpacotes.......................... 11 (data, models, losses, ...)   ║
║                                                                          ║
║  Simulador Python (geosteering_ai/simulation/):                          ║
║    forward.py.......................... 673 LOC                          ║
║    multi_forward.py.................... 1.134 LOC                        ║
║    _workers.py......................... 1.009 LOC                        ║
║    _jacobian.py........................ 682 LOC                          ║
║    _numba/ (5 arquivos)................ 2.684 LOC                        ║
║      kernel.py ............... 873 | propagation.py .... 624             ║
║      dipoles.py .............. 967 | hankel.py ......... 220             ║
║    _jax/ (8 arquivos).................. ~5.000 LOC                       ║
║      forward_pure.py ........ 1.373 | multi_forward.py .. 658            ║
║                                                                          ║
║  Simulador Fortran (Fortran_Gerador/):                                   ║
║    PerfilaAnisoOmp.f08, ......... ~6.900 LOC F08                         ║
║    + batch_runner.py + f2py wrapper                                      ║
║                                                                          ║
║  PipelineConfig (config.py):                                             ║
║    Campos.............................. 246                              ║
║    Presets YAML........................ 7                                ║
║                                                                          ║
║  Modelos DL:                                                             ║
║    Arquiteturas (ModelRegistry)......... 48 (9 famílias)                 ║
║    Blocos Keras reutilizáveis........... 23+                             ║
║    Funções de perda..................... 26 (4 categorias)               ║
║    Cenários PINN........................ 8                               ║
║    Tipos de ruído (noise/functions.py).. 34                              ║
║    Feature Views........................ 7                               ║
║    Famílias Geosinais................... 5 (USD, UAD, UHR, UHA, U3DF)    ║
║                                                                          ║
║  Testes (pytest):                                                        ║
║    CPU................................. 744+ (consolidados v2.5)         ║
║    Simulador Python.................... 1.459+ (v1.6.0)                  ║
║    Simulation Manager smoke............. 68 (v2.21)                      ║
║    GPU (Colab T4)...................... 1.011+ (v2.0 validation)         ║
║                                                                          ║
║  Performance:                                                            ║
║    Cenário A (30 pts, 1 freq).......... 1.392.371 mod/h (v2.21)          ║
║    Cenário E (600 pts, 1 freq)......... 121.957 mod/h (v2.21, meta hist.)║
║    Fortran tatu.x...................... 58.856 mod/h (v10.0, 245% meta)  ║
║    Paridade Numba vs Fortran........... <1e-12 (7 modelos canônicos)     ║
║    Paridade JAX vs Numba............... 3.5e-14 (Sprint 10 Phase 2)      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 1.3 Estrutura de Diretórios (Real, Mapeada)

```
~/Geosteering_AI/
│
├── CLAUDE.md                              ← Instruções persistentes (20 KB)
├── README.md
├── pyproject.toml                         ← Build system pip-installable
├── .python-version                        ← 3.13
├── .gitignore                             ← *.dat *.keras checkpoints
├── .mcp.json                              ← consensus + colab-mcp configurados
│
├── .claude/                               ← Configuração Claude Code
│   ├── settings.json                      ← 5 hooks PreToolUse/PostToolUse/Stop
│   ├── settings.local.json
│   ├── commands/                          ← 11 skills/slash commands
│   │   ├── geosteering-v2.md              (skill principal de domínio)
│   │   ├── geosteering-v5015.md           (legado, referência)
│   │   ├── geosteering-physics.md
│   │   ├── geosteering-models.md
│   │   ├── geosteering-losses.md
│   │   ├── geosteering-code-v2.md
│   │   ├── geosteering-simulator-python.md
│   │   ├── geosteering-simulator-fortran.md
│   │   ├── geosteering-simulation-manager.md
│   │   ├── consensus-search.md
│   │   └── arxiv-search.md
│   ├── hooks/                             ← 9 hooks shell determinísticos
│   │   ├── validate-physics.sh            (PreToolUse, errata física)
│   │   ├── protect-critical-files.sh      (PreToolUse, segurança)
│   │   ├── compile-check.sh               (PostToolUse, py_compile)
│   │   ├── lint-v2-standards.sh           (PostToolUse, padrões D1-D14)
│   │   ├── autoformat.sh                  (PostToolUse, ruff/black)
│   │   ├── validate-scientific-refs.sh    (PostToolUse, citações)
│   │   ├── run-pytest.sh                  (Stop, suite completa)
│   │   ├── setup-environment.sh           (SessionStart:startup)
│   │   └── reinject-errata.sh             (SessionStart:compact)
│   ├── templates/
│   │   └── report_template.md             ← Template de relatório técnico
│   ├── worktrees/
│   │   └── suspicious-goldwasser/         ← Worktree existente
│   └── plans/                             ← Planos arquivados
│
├── .agents/                               ← Configuração Antigravity (existe)
│   └── rules/
│       └── antigravity-rtk-rules.md       ← Regras integração Antigravity
│
├── geosteering_ai/                        ← PACOTE PRINCIPAL (46k LOC)
│   ├── __init__.py
│   ├── config.py                          ← PipelineConfig (246 campos)
│   ├── data/                              ← 12 módulos, ~6.700 LOC
│   ├── noise/                             ← 2 módulos, ~2.600 LOC
│   ├── models/                            ← 13 módulos, ~8.800 LOC
│   ├── losses/                            ← 3 módulos, ~3.600 LOC
│   ├── training/                          ← 7 módulos, ~8.100 LOC
│   ├── inference/                         ← 4 módulos, ~1.800 LOC
│   ├── evaluation/                        ← 11 módulos, ~7.300 LOC
│   ├── visualization/                     ← 11 módulos, ~2.500 LOC
│   ├── utils/                             ← 6 módulos, ~2.500 LOC
│   └── simulation/                        ← Subpacote simulador (v1.6.0)
│       ├── __init__.py                    ← API pública (simulate, simulate_multi, ...)
│       ├── config.py                      ← SimulationConfig dataclass
│       ├── forward.py                     ← API single-model (Numba)
│       ├── multi_forward.py               ← API multi-TR/angle/freq
│       ├── _jacobian.py                   ← FD Numba + jacfwd JAX
│       ├── _workers.py                    ← ProcessPool + topology
│       ├── _numba/                        ← Backend Numba (5 mod, 2.684 LOC)
│       ├── _jax/                          ← Backend JAX (8 mod, ~5.000 LOC)
│       ├── filters/                       ← Werthmuller/Kong/Anderson .npz
│       ├── postprocess/                   ← compensation.py + tilted.py
│       ├── io/                            ← write_dat_from_tensor + parsers
│       ├── benchmarks/                    ← bench_v214_numba.py + cenários
│       ├── validation/                    ← Half-space + comparações
│       ├── visualization/                 ← Plots simulação
│       └── tests/                         ← simulation_manager.py (GUI PyQt6)
│
├── tests/                                 ← 59 arquivos test_*.py
│
├── Fortran_Gerador/                       ← Simulador Fortran tatu.x (~6.9k LOC)
│   ├── PerfilaAnisoOmp.f08
│   ├── magneticdipoles.f08
│   ├── filtersv2.f08
│   ├── parameters.f08
│   ├── RunAnisoOmp.f08
│   ├── Makefile
│   ├── batch_runner.py
│   ├── fifthBuildTIVModels.py
│   └── bench/                             ← Validador numérico Fortran
│
├── benchmarks/                            ← Bench scripts (CLI)
│   └── bench_v214_numba.py                ← Cenários A, B, C, D, E
│
├── configs/                               ← 7 presets YAML
│   ├── baseline.yaml | robusto.yaml
│   ├── nstage_n2.yaml | nstage_n3.yaml
│   ├── geosinais_p4.yaml | dtb_p5.yaml
│   └── realtime_causal.yaml
│
├── docs/                                  ← Documentação (~8.000 LOC MD)
│   ├── ARCHITECTURE_v2.md                 (autoritativo arquitetura DL)
│   ├── ROADMAP.md                         (F1-F7 + v3.0)
│   ├── CHANGELOG.md
│   ├── MIGRATION_GUIDE.md
│   ├── documentacao_apresentacao_geosteering_ai.md
│   ├── documentacao_geosteering.md
│   ├── documentacao_inferencia_offline.md
│   ├── documentacao_losses.md
│   ├── documentacao_models.md
│   ├── documentacao_noises.md
│   ├── documentacao_pinns.md
│   ├── physics/                           (em_tensor, decoupling, GS, FV)
│   ├── reference/                         (40+ MDs técnicos)
│   │   └── analise_cenarios_otimizacao_simulador_numba.md
│   └── reports/                           (relatórios v2.5–v2.21 + multi-agente)
│
├── notebooks/                             ← 4+ notebooks Colab
│   ├── train_colab.ipynb | evaluate_colab.ipynb
│   ├── geosteering_colab.ipynb | eda_colab.ipynb
│   └── validate_gpu_colab.ipynb
│
├── scripts/                               ← Utilities (extract_hankel_weights, etc.)
├── tools/                                 ← consensus-mcp-server, etc.
├── tutorials/                             ← Material didático
├── PDFs/                                  ← 30+ artigos científicos
├── Tex_Projects/                          ← Formulação TeX (TatuAniso)
├── legacy/                                ← C0-C47 código legado (preservado)
├── old_geosteering_ai/                    ← v anterior (referência v2.21 análise)
└── sm_experiments/, sm_output/            ← Saídas Simulation Manager
```

### 1.4 Pilha Tecnológica

| Camada | Componentes | Versão |
|:-------|:-----------|:-------|
| Linguagem | Python | 3.13 (3.12 CI fallback) |
| DL Framework | TensorFlow / Keras | 2.13+ / 3.x |
| HPC CPU | Numba | 0.60+ (`@njit parallel=True nogil=True cache=True`) |
| HPC GPU | JAX | 0.4.30+ (`jit + vmap + pmap + fori_loop`) |
| Simulador GT | Fortran (gfortran 15.x) | F08, OpenMP 5.0+ |
| Pipe Fortran↔Py | f2py wrapper | nativo NumPy 2.x |
| GUI | PyQt6 + PySide6 | migrado em v2.7a |
| HPO | Optuna | 3.0+ |
| Tests | pytest + pytest-cov | 7.0+ |
| Lint/Format | mypy + ruff + black | configurados |
| CI/CD | GitHub Actions | matriz 3.12/3.13 |
| Treino GPU | Google Colab Pro+ | T4/A100 |
| LLMs | Claude Opus 4.7 (1M) + Sonnet 4.6 + Haiku 4.5 | Max 5× |
| Completions IDE | Google Gemini 3.1 Pro | Antigravity |

### 1.5 Estado dos Subsistemas

| Subsistema | Status | Próximo passo planejado |
|:-----------|:------:|:-----------------------|
| Pipeline DL (DataPipeline + ModelRegistry) | Estável | Treino end-to-end (F2 ROADMAP) |
| Simulador Numba | Estável v2.21 | FLAT prange v2.22 |
| Simulador JAX (CPU/GPU) | Beta v1.6.0 | flip default vmap_real, validação T4 |
| Simulador Fortran tatu.x | Estável v10.0 | F8 (1.5D), F9 (invasão) |
| Inferência offline | Estável | UQ INN (já implementado, treinar) |
| Inferência realtime | Estável | Integração WITSML |
| GUI Simulation Manager | Estável v2.21 | Polish UX, painéis avançados |
| MLOps | Inexistente | F6 (FastAPI + MLflow) |
| 2D/2.5D/3D | Não iniciado | v3.0 (após v2.30) |

### 1.6 Hooks Atualmente Configurados (Inventário)

```
.claude/settings.json (em produção):

PreToolUse(Edit|Write):
  ├─ validate-physics.sh        ← bloqueia PyTorch, eps inseguros, globals().get(),
  │                               YAML com errata violada
  └─ protect-critical-files.sh  ← bloqueia Edit em paths sensíveis (Fortran_Gerador,
                                  legacy, configs com tag versão fixa)

PostToolUse(Edit|Write):
  ├─ compile-check.sh           ← py_compile no arquivo modificado
  ├─ lint-v2-standards.sh       ← verifica D1-D14 (mega-header, docstrings)
  ├─ autoformat.sh              ← ruff format + isort
  └─ validate-scientific-refs.sh← valida formato citações em docstrings

Stop:
  └─ run-pytest.sh              ← pytest tests/ -q --tb=no (timeout 120s)

SessionStart(startup):
  └─ setup-environment.sh       ← branch, último commit, contagem de arquivos,
                                  pytest rápido, versão Python

SessionStart(compact):
  └─ reinject-errata.sh         ← reinjeta errata, proibições absolutas e
                                  últimos 5 commits após compactação
```

Esta base existente é robusta e será **expandida** (não substituída) pela
arquitetura descrita neste documento.

### 1.7 Documentos de Referência Carregados em Cada Sessão

```
Hierarquia de carregamento automático (precedência decrescente):
  1. CLAUDE.md (raiz, 20 KB)
  2. ~/.claude/projects/-Users-daniel-Geosteering-AI/memory/MEMORY.md
  3. SessionStart hooks (setup-environment.sh + reinject-errata.sh)
  4. user-prompt-submit-hook (preâmbulo enriquecido)

Documentos canônicos referenciados:
  • docs/ARCHITECTURE_v2.md        — arquitetura DL detalhada
  • docs/ROADMAP.md                — fases F1-F7 + v3.0
  • docs/reference/analise_cenarios_otimizacao_simulador_numba.md
  • docs/reports/v{2.5..2.21}_2026-04-XX.md  — sprints recentes
  • docs/reference/plano_simulador_python_jax_numba.md
  • docs/reference/documentacao_simulador_fortran.md
```

---

## 2. Princípios Arquiteturais e Modelo Mental

### 2.1 Os Sete Princípios Fundamentais

```
╔══════════════════════════════════════════════════════════════════════════╗
║  P1 — PRECISÃO FÍSICA É INVIOLÁVEL                                      ║
║  Toda otimização, refatoração ou nova feature DEVE preservar:           ║
║    • Paridade Fortran <1e-12 nos 7 modelos canônicos                    ║
║    • Conservação de energia (|R| ≤ 1 em coeficientes de reflexão)        ║
║    • Simetria de Maxwell (H_xy ≈ H_yx em meio TIV)                      ║
║    • Errata física imutável (frequency_hz, spacing_meters, etc.)         ║
║  Mecanismo: Hook PostToolUse + CI gate + agente físico revisor.          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P2 — UM ÚNICO PONTO DE VERDADE PARA CONFIGURAÇÃO                        ║
║  Toda função recebe `config: PipelineConfig` ou `cfg: SimulationConfig`. ║
║  Nada de globals(), variáveis de módulo mutáveis, ou flags dispersas.    ║
║  Reprodutibilidade: YAML + tag GitHub + seed = resultado idêntico.       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P3 — MODULARIDADE COM INTERFACES ESTÁVEIS                               ║
║  Cada subpacote tem responsabilidade única + API pública via __init__.   ║
║  Mudanças internas NÃO devem propagar para callers.                      ║
║  Backends (numba/jax/fortran) intercambiáveis via dispatcher.            ║
║  Plugins de física (1D, 2D Born, 2D MEF, 2.5D, 3D) carregáveis.         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P4 — FAIL-FAST + OBSERVABILIDADE                                        ║
║  PipelineConfig.__post_init__ valida ANTES de qualquer execução.         ║
║  Logging estruturado (JSON em produção); métricas em cada estágio.       ║
║  Hooks bloqueiam erros conhecidos antes de chegar ao runtime.            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P5 — SEGREGAÇÃO DE NÚCLEO E CASCA                                       ║
║  Núcleo (geosteering_ai/): puro, testável, sem efeitos colaterais.       ║
║  Casca (GUI, CLI, API REST, notebooks): orquestra; não contém lógica.    ║
║  Vantagem: substituir/adicionar interface não toca o núcleo.             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P6 — DESENVOLVIMENTO ORIENTADO POR EVIDÊNCIA                            ║
║  Decisões de performance: medir → otimizar → medir (nunca chutar).       ║
║  Decisões científicas: validar com literatura via Consensus/ArXiv.       ║
║  Decisões arquiteturais: documentar em docs/reports/ com data.           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P7 — ECONOMIA DE INTELIGÊNCIA ARTIFICIAL                                ║
║  Use Haiku onde Haiku resolve; Sonnet onde precisão importa;             ║
║  Opus apenas onde raciocínio profundo agrega valor único.                ║
║  Hooks determinísticos (shell) para checagens triviais (custo zero).     ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 2.2 Modelo Mental: O Software Como Catedral de Precisão Física

O Geosteering AI **não é um produto SaaS típico**. É um sistema científico
que substitui métodos tradicionais de inversão geofísica por Deep Learning,
com aceitação operacional condicionada a:

```
1. Reproduzir resposta EM com fidelidade física <1e-12 vs ground truth
   (simulador Fortran validado contra teoria analítica)

2. Quantificar incerteza de forma calibrada (não basta R² alto;
   é necessário UQ confiável para decisões de geosteering)

3. Operar em tempo real (latência <100 ms por inferência em produção)

4. Ser auditável: cada predição tem proveniência (modelo, dados, config,
   versão, seed) rastreável

5. Robustez a ruído real de campo (curriculum 3-fase + 34 tipos de noise)

6. Generalização para regimes físicos extremos (alta resistividade,
   anisotropia forte, dip variável)
```

A arquitetura multi-agente serve a esta missão: cada agente reforça uma
camada de qualidade que eleva o sistema ao nível industrial.

### 2.3 Trade-offs Conscientes

| Trade-off | Escolha | Custo Aceito |
|:----------|:--------|:-------------|
| Performance vs. Paridade Fortran | Paridade primeiro, performance depois | -10% throughput potencial em troca de bit-exatness |
| Velocidade dev vs. Documentação D1-D14 | Documentação obrigatória | +20% tempo de codificação |
| Custo Opus vs. Qualidade arquitetural | Opus em sprints arquiteturais | ~$5-10 por sprint vs. semanas de bug debugging |
| GUI nativa vs. Web | PyQt6 mantido | Não roda em browser, mas baixa latência local |
| Build automático vs. Confirmação manual | Confirmar destrutivos, automatizar reversíveis | Pequeno atrito UX em troca de zero perda acidental |

---

## 3. Topologia Multi-Agente — Cinco Camadas

A arquitetura organiza-se em **cinco camadas hierárquicas** com fluxo
preferencialmente top-down (orquestrador delega para domínio; domínio invoca
infraestrutura) e canais laterais explícitos (agentes de qualidade observam
todas as camadas).

### 3.1 Diagrama Macro

```
╔════════════════════════════════════════════════════════════════════════════╗
║  CAMADA 0 — HUMANO + ORQUESTRADOR (NÚCLEO COGNITIVO)                       ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │ Daniel (autor) ←→ Claude Opus 4.7 1M (Orquestrador Principal)        │  ║
║  │ Contexto: CLAUDE.md + MEMORY.md + plano ativo + projeto inteiro      │  ║
║  │ Responsabilidades:                                                    │  ║
║  │   • Decisões arquiteturais multi-arquivo                              │  ║
║  │   • Trade-offs físicos vs computacionais                              │  ║
║  │   • Coordenação de sprints complexos                                  │  ║
║  │   • Auditoria de saídas dos agentes filhos                            │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
╠════════════════════════════════════════════════════════════════════════════╣
║  CAMADA 1 — DOMÍNIO (AGENTES ESPECIALIZADOS)                               ║
║  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              ║
║  │ Sim. Numba      │ │ Sim. JAX        │ │ Sim. Fortran    │              ║
║  │ Opus 4.7        │ │ Opus 4.7        │ │ Sonnet 4.6      │              ║
║  └─────────────────┘ └─────────────────┘ └─────────────────┘              ║
║  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              ║
║  │ DL Pipeline     │ │ PINNs           │ │ Dados (FV/GS)   │              ║
║  │ Sonnet 4.6      │ │ Opus 4.7        │ │ Sonnet 4.6      │              ║
║  └─────────────────┘ └─────────────────┘ └─────────────────┘              ║
║  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              ║
║  │ Geosteering RT  │ │ MLOps/Deploy    │ │ Frontend GUI    │              ║
║  │ Sonnet 4.6      │ │ Sonnet 4.6      │ │ Sonnet 4.6      │              ║
║  └─────────────────┘ └─────────────────┘ └─────────────────┘              ║
╠════════════════════════════════════════════════════════════════════════════╣
║  CAMADA 2 — QUALIDADE (REVIEWERS LATERAIS)                                 ║
║  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              ║
║  │ Revisor Físico  │ │ Revisor Perf.   │ │ Revisor Código  │              ║
║  │ Sonnet 4.6      │ │ Haiku 4.5       │ │ Sonnet 4.6      │              ║
║  └─────────────────┘ └─────────────────┘ └─────────────────┘              ║
║  ┌─────────────────┐ ┌─────────────────┐                                   ║
║  │ Doc PT-BR       │ │ Segurança       │                                   ║
║  │ Haiku 4.5       │ │ Sonnet 4.6      │                                   ║
║  └─────────────────┘ └─────────────────┘                                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║  CAMADA 3 — PESQUISA E DOCUMENTAÇÃO (KNOWLEDGE)                            ║
║  ┌─────────────────┐ ┌─────────────────┐                                   ║
║  │ Pesquisador     │ │ Documentador    │                                   ║
║  │ Sonnet 4.6      │ │ Haiku 4.5       │                                   ║
║  │ +Consensus,     │ │ +D1-D14         │                                   ║
║  │  ArXiv,bioRxiv  │ │  templates      │                                   ║
║  └─────────────────┘ └─────────────────┘                                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║  CAMADA 4 — INFRAESTRUTURA (DETERMINÍSTICA, SEM IA)                        ║
║  • Hooks shell (validate-physics, pytest, lint, format)                    ║
║  • MCP Servers (physics-validator, numba-profiler, colab-bridge,           ║
║    consensus, jupyter-runtime)                                             ║
║  • CI GitHub Actions (compile + pytest + mypy + paridade Fortran)          ║
║  • Worktrees (isolamento por sprint)                                       ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### 3.2 Fluxos Entre Camadas

```
FLUXO 1 — SPRINT TÍPICO (top-down + qualidade lateral):

  Daniel pede sprint v2.22
     │
     ▼
  Orquestrador (Opus 4.7 1M)
     │  carrega: CLAUDE.md + análise de cenários + forward.py + kernel.py
     │  produz: plano arquitetural (TodoWrite + plan file)
     │
     ├─→ Domínio: Agente Sim. Numba (Opus em worktree isolada)
     │     implementa FLAT prange + testes
     │
     ├─→ Qualidade: Revisor Físico (Sonnet, paralelo)
     │     valida paridade Fortran + simetria Maxwell
     │
     ├─→ Qualidade: Revisor Perf. (Haiku, paralelo)
     │     roda bench_v214_numba.py + interpreta números
     │
     └─→ Qualidade: Doc PT-BR (Haiku)
           atualiza CHANGELOG + ROADMAP + sub-skill
     │
     ▼
  Orquestrador consolida + commit + PR


FLUXO 2 — DEBUG COMPLEXO (fan-out de hipóteses):

  Bug: regressão de paridade em modelo carbonato_5c
     │
     ▼
  Orquestrador
     │
     ├─→ Sim. Numba: "rastreie kernel.py + propagation.py em alta ρ"
     ├─→ Sim. JAX: "rode mesma simulação em JAX, compare valores"
     ├─→ Pesquisador: "literatura sobre overflow em high-ρ TIV"
     │
     ▼
  Orquestrador sintetiza 3 outputs + Opus formula hipótese causal
     │
     ▼
  Implementação + validação


FLUXO 3 — TREINAMENTO DL (background + monitoramento):

  Solicitar treino de modelo XYZ
     │
     ▼
  DL Pipeline gera dataset (background)
     │
     ▼
  push para GitHub + tag
     │
     ▼
  Daniel executa Colab notebook
     │
     ▼
  /loop 10m check-colab-training (Haiku)
     consulta Colab MCP + GitHub Actions logs
     reporta epoch atual, val_loss, ETA
     alerta se loss diverge
     │
     ▼
  Quando concluir: Avaliação (Sonnet)
     calcula métricas, compara com baseline,
     gera relatório docs/reports/treino_{model}_{date}.md
```

---

## 4. Catálogo de Agentes Especializados

Esta seção define cada agente em formato YAML pronto para ser colocado em
`.claude/commands/{agent}.md`. Todos seguem a estrutura padrão Claude Code.

Lista de 17 agentes (5 já existem, 12 a criar):

| # | Agente | Status | Modelo padrão | Camada |
|:-:|:-------|:------:|:--------------|:------:|
| 1 | `geosteering-orchestrator` | a criar | Opus 4.7 1M | 0 |
| 2 | `geosteering-simulator-numba` | expandir existente | Opus 4.7 | 1 |
| 3 | `geosteering-simulator-jax` | a criar | Opus 4.7 | 1 |
| 4 | `geosteering-simulator-fortran` | existe | Sonnet 4.6 | 1 |
| 5 | `geosteering-dl-pipeline` | a criar | Sonnet 4.6 | 1 |
| 6 | `geosteering-pinns` | a criar | Opus 4.7 | 1 |
| 7 | `geosteering-data-pipeline` | a criar | Sonnet 4.6 | 1 |
| 8 | `geosteering-realtime` | a criar | Sonnet 4.6 | 1 |
| 9 | `geosteering-mlops` | a criar | Sonnet 4.6 | 1 |
| 10 | `geosteering-frontend` | a criar | Sonnet 4.6 | 1 |
| 11 | `geosteering-physics-reviewer` | a criar | Sonnet 4.6 | 2 |
| 12 | `geosteering-perf-reviewer` | a criar | Haiku 4.5 | 2 |
| 13 | `geosteering-code-reviewer` | a criar | Sonnet 4.6 | 2 |
| 14 | `geosteering-docs-ptbr` | a criar | Haiku 4.5 | 2 |
| 15 | `geosteering-security-auditor` | a criar | Sonnet 4.6 | 2 |
| 16 | `geosteering-research` | expandir consensus-search | Sonnet 4.6 | 3 |
| 17 | `geosteering-documentation` | a criar | Haiku 4.5 | 3 |

### 4.1 Padrão de Definição de Agente

```yaml
---
name: <nome-do-agente>
description: |
  <2-4 linhas: domínio, gatilhos, capacidades>
tools:
  - Read
  - Edit  # apenas se modifica arquivos
  - Bash  # apenas se executa comandos
  - Agent # apenas se invoca subagentes
  - WebSearch / WebFetch  # apenas pesquisa
model: <opus|sonnet|haiku>
isolation: <worktree>?  # apenas para sprints destrutivos
allowed_paths:
  - geosteering_ai/<subpacote>/**
forbidden_paths:
  - Fortran_Gerador/**     # se aplicável
  - legacy/**              # se aplicável
constraints:
  - F1: TensorFlow/Keras exclusivo
  - F2: Paridade Fortran <1e-12
  - <constraints específicos do domínio>
---

# <Nome do Agente>

<Documentação operacional: quando invocar, padrões obrigatórios,
exemplos de prompts ideais, anti-patterns conhecidos.>
```

### 4.2 Agente 1 — Orquestrador (Opus 4.7 1M)

```yaml
---
name: geosteering-orchestrator
description: |
  Orquestrador central do projeto Geosteering AI 2.0. Carrega o projeto
  inteiro em contexto (~46k LOC + docs + memória). Decisões arquiteturais
  multi-arquivo, planejamento de sprints, fan-out de subagentes paralelos,
  síntese de resultados multi-perspectiva.
  Usar para: sprints com >5 arquivos, debugging cross-module, design de
  novas features, análise de regressões misteriosas, decisões de trade-off
  físico vs performance.
tools: [Read, Edit, Write, Bash, Agent, WebSearch, WebFetch, TodoWrite]
model: claude-opus-4-7
constraints:
  - F1, F2, F3, F4, F5, F6, F7, F8, F9, F10 (todas)
---

# Orquestrador Geosteering AI 2.0

## Quando Invocar

INVOCAR PARA:
  • Sprints v2.22+ no simulador (cross-file, requerem análise profunda)
  • Refatorações arquiteturais (e.g., introduzir backends abstratos para 2D)
  • Bugs misteriosos com regressões em múltiplos commits
  • Design de novas funcionalidades (e.g., suporte a invasão de filtrado)
  • Coordenar 3+ subagentes em fan-out
  • Análise de literatura científica → decisão de implementação
  • Auditoria de PRs grandes (>500 LOC)

NÃO INVOCAR PARA:
  • Edição mono-arquivo trivial (use Sonnet)
  • Atualização de CHANGELOG (use Doc PT-BR Haiku)
  • Smoke testes (use Perf Reviewer Haiku)

## Padrões Operacionais

1. SEMPRE começar com TodoWrite listando 5-15 tarefas concretas.
2. Para sprint de simulador: invocar Pesquisador em paralelo com
   "literatura recente sobre <tópico>" enquanto desenha o plano.
3. Para implementação: delegar a Domínio + isolation="worktree".
4. Para validação: invocar Físico + Perf + Código em FAN-OUT paralelo.
5. SÍNTESE: nunca aceitar saída de subagente como verdade absoluta;
   verificar arquivos modificados (Trust but Verify do CLAUDE.md).
6. ENCERRAR sprint com:
   - commits granulares (1 commit por preocupação)
   - relatório docs/reports/v{X}_{date}.md (template)
   - update memory + CHANGELOG + ROADMAP + sub-skill
   - PR com descrição estruturada

## Anti-patterns a Evitar

• Invocar subagentes sequencialmente quando independentes (use paralelo).
• Aceitar relato "fiz X" sem ler o diff produzido.
• Pular o Revisor Físico quando o sprint toca _numba/* ou _jax/*.
• Fazer commit sem atualizar memory/MEMORY.md.
• Esquecer de checar paridade Fortran (CI gate).

## Exemplo de Sprint Estruturado

[Sprint v2.22 — FLAT prange]

  TodoWrite:
    1. Pesquisar nested prange no Numba via Pesquisador
    2. Ler kernel.py + forward.py topo a fundo
    3. Plano arquitetural do FLAT prange
    4. Worktree isolada feat/simulator-v2.22
    5. Implementar _simulate_combined_prange_flat
    6. Adaptar simulate_multi para dispatch
    7. Criar 5 testes regressão (cenários A,E,F,G,H)
    8. Rodar pytest + paridade Fortran
    9. Bench Cenário F (nf=4) baseline vs FLAT
    10. Doc relatório v2.22
    11. Commit + push + PR
    12. Update memory + sub-skill

  Fan-out paralelo (passos 3-5):
    Agent(Sim Numba, isolation=worktree) → implementação
    Agent(Pesquisador) → "FLAT prange Numba best practices 2025"
    Agent(Físico Reviewer) → revisar design via Read

  Síntese: orquestrador consolida outputs e prossegue.
```

### 4.3 Agente 2 — Simulador Numba

```yaml
---
name: geosteering-simulator-numba
description: |
  Especialista em simulador Python Numba JIT (geosteering_ai/simulation/_numba/
  + forward.py + multi_forward.py). Domínio: kernel.py, propagation.py,
  dipoles.py, hankel.py, geometry.py, rotation.py. Padrão @njit cache=True
  nogil=True; nunca parallel=True aninhado em prange outer.
  Acionar para: otimizações de performance, novos kernels, fix de regressões
  Numba, análise de paralelismo, integração com pool de workers.
tools: [Read, Edit, Bash, Agent]
model: claude-opus-4-7
isolation: worktree
allowed_paths:
  - geosteering_ai/simulation/_numba/**
  - geosteering_ai/simulation/forward.py
  - geosteering_ai/simulation/multi_forward.py
  - geosteering_ai/simulation/_workers.py
  - geosteering_ai/simulation/_jacobian.py
  - benchmarks/**
  - tests/test_simulation_*.py
forbidden_paths:
  - Fortran_Gerador/**       # nunca tocar Fortran
  - geosteering_ai/simulation/_jax/**  # delegar para agente JAX
constraints:
  - F2: paridade Fortran <1e-12 inviolável
  - F3: nunca parallel=True em função chamada de prange outer
  - Errata Numba: nogil=True universal no hot path; cache=True obrigatório
---

# Especialista Simulador Numba JIT

## Domínio Físico

Simulador EM 1D em meio TIV (Transversely Isotropic Vertical):
  • Propagação: HMD (dipolo magnético horizontal) + VMD (vertical)
  • Integração de Hankel: Werthmuller 201pt (padrão), Kong 61pt, Anderson 801pt
  • Tensor completo de 9 componentes: H = [Hxx, Hxy, ..., Hzz]
  • Rotação para qualquer dip via rotate_tensor
  • Alta resistividade ρ ∈ [0.01, 1e6] Ω·m via P-matrix estável

## Mapa de Decoradores @njit (v2.21)

  common_arrays           : @njit(cache=True, nogil=True)
  common_factors          : @njit(cache=True, nogil=True)
  hmd_tiv                 : @njit(cache=True, nogil=True)
  vmd                     : @njit(cache=True, nogil=True)
  _hankel_j0_kernel       : @njit(cache=True, nogil=True, fastmath=True)
  _fields_in_freqs_kernel_cached : @njit(cache=True, nogil=True)  ← NUNCA parallel=True
  precompute_common_arrays_cache : @njit(parallel=True, cache=True, nogil=True)
                                   prange(nf), chamado UMA vez por contexto serial
  _simulate_combined_prange : @njit(parallel=True, cache=True, nogil=True)
                              prange(n_combos × n_pos)

## Anti-patterns Documentados

  ✗ parallel=True em função chamada de prange outer
    → causa nested prange overhead (~14s por simulação no Cenário E)
    → fix v2.21 reverteu Sprint 13.1
  ✗ fastmath=True sem validação de paridade
  ✗ Modificar dipoles.py sem rodar test_simulation_compare_fortran.py
  ✗ Mexer em propagation.py sem entender P-matrix recursion
  ✗ Adicionar @njit em função pequena (<100ns) chamada de prange
    (overhead de chamada > custo da função)

## Workflow Padrão de Sprint

  1. Read kernel.py + forward.py completos (∼1.500 LOC)
  2. Read análise_cenarios_otimizacao_simulador_numba.md (briefing)
  3. Implementar em worktree dedicada
  4. Bash: pytest -k "fortran_python_numba" ANTES E DEPOIS
  5. Bash: python benchmarks/bench_v214_numba.py --scenario E (5 runs)
  6. Se paridade falhar: REVERT imediato, investigar, nunca commitar
  7. Se performance regredir: REVERT, analisar via numba-profiler MCP
```

### 4.4 Agente 3 — Simulador JAX

```yaml
---
name: geosteering-simulator-jax
description: |
  Especialista em simulador JAX (geosteering_ai/simulation/_jax/). Domínio:
  forward_pure.py, multi_forward.py, dipoles_unified.py, dipoles_native.py,
  propagation.py, hankel.py, kernel.py. Padrões: jax.jit + vmap + fori_loop;
  tracer-safe; jnp.where (sem if/else); cache de JIT compilations.
  Acionar para: otimizações GPU, vmap aninhado, jacfwd/jacrev,
  flip de defaults JAX, validação T4/A100.
tools: [Read, Edit, Bash, Agent]
model: claude-opus-4-7
isolation: worktree
allowed_paths:
  - geosteering_ai/simulation/_jax/**
  - geosteering_ai/simulation/multi_forward.py  # apenas dispatch JAX
  - tests/test_simulation_jax_*.py
  - benchmarks/bench_sprint*_*.py
constraints:
  - F2: paridade JAX vs Numba <1e-10 (gate Sprint 12)
  - F3: tracer-safe; nunca branching Python em função jit-compilada
  - JAX GPU 64-bit: jax.config.update("jax_enable_x64", True) obrigatório
  - Cache JIT: usar _UNIFIED_JIT_CACHE / count_compiled_xla_programs
---

# Especialista Simulador JAX

## Estado Atual (v1.6.0)

  • Sprint 10 Phase 2: dipoles_unified (44 XLA → 1 XLA em oklahoma_28)
  • Sprint 11-JAX: simulate_multi_jax (multi-TR/angle/freq)
  • Sprint 12: find_layers_tr_jax + vmap real (opt-in)
  • Paridade JAX vs Numba: 3.5e-14 em modelos canônicos
  • cfg.jax_strategy: "bucketed" (default) | "unified" (opt-in)
  • cfg.jax_vmap_real: bool (default False)

## Próximos Passos

  • Validação T4/A100 (manual no Colab Pro+) com jax_strategy="unified"
  • Flip default → unified em v1.7.0 quando paridade GPU validada
  • vmap aninhado para batch × multi-TR no GPU
  • Integração com TensorFlow para SurrogateNet (PINN modo neural)

## Padrões Tracer-Safe

  ✓ jnp.where(cond, x, y)        # branching diferenciável
  ✓ lax.fori_loop(0, n, body, x) # loop estático
  ✓ vmap(f, in_axes=(0, None))   # batch sobre eixo
  ✓ jit(f, static_argnums=(0,))  # constantes fora de tracing
  ✗ if x > 0:                    # tracer leak
  ✗ for i in range(x.shape[0]):  # tracer leak (use vmap ou fori_loop)
```

### 4.5 Agente 5 — DL Pipeline

```yaml
---
name: geosteering-dl-pipeline
description: |
  Especialista em pipeline TF/Keras (geosteering_ai/{models,losses,training,
  inference}/). 48 arquiteturas, 26 losses, 17+ callbacks, 8 cenários PINN,
  Factory + Registry, on-the-fly noise→FV→GS→scale.
  Acionar para: novas arquiteturas, novas losses, refatoração de callbacks,
  HPO Optuna, ajustes em DataPipeline.
tools: [Read, Edit, Bash, Agent]
model: claude-sonnet-4-6
allowed_paths:
  - geosteering_ai/models/**
  - geosteering_ai/losses/**
  - geosteering_ai/training/**
  - geosteering_ai/inference/**
  - geosteering_ai/data/**
  - geosteering_ai/noise/**
  - configs/**
  - tests/test_models.py
  - tests/test_losses.py
  - tests/test_training.py
constraints:
  - F1: TensorFlow/Keras EXCLUSIVO; PyTorch proibido
  - F4: errata física (frequency, spacing, sequence_length, etc.)
  - F5: cadeia raw→noise→FV→GS→scale
  - F6: split por modelo geológico
  - Sempre PipelineConfig; nunca globals()
  - Factory para componentes (ModelRegistry, LossFactory, build_callbacks)
---

# Especialista Pipeline DL

## Catálogo de Arquiteturas (48 em 9 famílias)

  CNN          : ResNet_18/34/50, ConvNeXt, Inception, CNN_1D, ResNeXt (8)
  RNN          : LSTM, BiLSTM (2)
  Hybrid       : CNN_LSTM, CNN_BiLSTM_ED, ResNeXt_LSTM (3)
  TCN          : TCN, TCN_Advanced, ModernTCN (3)
  Transformer  : Transformer, TFT, PatchTST, Autoformer, iTransformer,
                 G_Query_Transformer (6)
  U-Net        : 14 variantes (offline-only)
  Decomposition: N-BEATS, N-HiTS (2)
  Advanced     : DNN, FNO, DeepONet, GeoAttention, INN (5)
  Geosteering  : WaveNet, Causal_Transformer, Informer, Mamba_S4,
                 Encoder_Forecaster (5 nativas causais)

## Padrão para Adicionar Nova Arquitetura

  1. Criar build_<nome>(config) em models/<familia>.py
  2. Registry: ModelRegistry().register("Nome", build_fn, tier, ...)
  3. Teste: forward pass em test_models.py
  4. Validação dual-mode (offline + causal se aplicável)
  5. Documentação D5/D6 (docstring com diagrama ASCII)
  6. Mega-header D1 atualizado com contagem
```

### 4.6 Agentes Restantes — Sumarização

Por brevidade, os agentes 4, 6-17 seguem o mesmo padrão. Suas definições
completas vão para arquivos individuais em `.claude/commands/`. Aqui o
sumário com escopo e responsabilidade:

| Agente | Escopo Principal | Restrições Únicas |
|:-------|:-----------------|:------------------|
| `geosteering-simulator-fortran` | `Fortran_Gerador/**` | Nunca quebrar paridade; sempre rodar `bench/validate_numerico.sh` após edits |
| `geosteering-pinns` | `losses/pinns.py` + cenários PINN | Validar gradientes via tape; sempre lambda schedule explícito |
| `geosteering-data-pipeline` | `data/`, `noise/` | Scaler em CLEAN; on-the-fly exclusivo; split por modelo |
| `geosteering-realtime` | `inference/realtime.py`, `geosteering.py` (modelos causais) | Latência <100 ms; padding causal; teste com sliding window |
| `geosteering-mlops` | API REST (futuro), MLflow, Docker | Não tocar núcleo; apenas casca de infraestrutura |
| `geosteering-frontend` | `simulation_manager.py` (PyQt6) + nova CLI | Sempre Qt main thread para GUI; CLI baseado em Typer |
| `geosteering-physics-reviewer` | Read-only de _numba/_jax/Fortran | Validar simetria Maxwell, paridade, conservação energia |
| `geosteering-perf-reviewer` | Bash bench + interpretar | Métricas obrigatórias: mediana de 5 runs |
| `geosteering-code-reviewer` | Read-only de qualquer .py | PEP8, tipagem, D1-D14, nada de print() |
| `geosteering-docs-ptbr` | Read+Edit em *.md, docstrings | PT-BR acentuação garantida; D1 atualizado em refactor |
| `geosteering-security-auditor` | Auditoria de PRs sensíveis | Bloqueia segredos no diff; valida .gitignore |
| `geosteering-research` | WebSearch, Consensus, ArXiv, bioRxiv | Citações obrigatórias com formato padrão |
| `geosteering-documentation` | Read+Write em docs/ | Templates obrigatórios; tabela ≥70% conteúdo |

Os arquivos completos serão gerados na fase de implementação (§22).

---

## 5. Agente de Pesquisa Científica

### 5.1 Justificativa

O Geosteering AI é projeto científico com base bibliográfica intensa (17+
artigos-chave em ROADMAP §2.1). Decisões arquiteturais devem ser **embasadas
em literatura recente** (2023-2026). Um agente dedicado a pesquisa garante:

1. **Atualização contínua**: literatura geofísica/DL evolui rapidamente.
2. **Validação cruzada**: ideias de implementação confrontadas com prior art.
3. **Inspiração para novas features**: detectar tendências (e.g., G-Query,
   Evidential Regression, Mamba-S4).
4. **Citações inline obrigatórias**: rastreabilidade científica.

### 5.2 Composição: Múltiplas Fontes Coordenadas

```
╔══════════════════════════════════════════════════════════════════════════╗
║  AGENTE PESQUISADOR (Sonnet 4.6)                                        ║
║                                                                          ║
║  Fontes integradas:                                                      ║
║                                                                          ║
║  1. CONSENSUS MCP (já configurado em .mcp.json)                          ║
║     → Semantic Scholar com filtros (citations, year, study_types)         ║
║     → Especialidade: papers de alto impacto, peer-reviewed                ║
║     → Rate limit: 3 calls em batch                                        ║
║                                                                          ║
║  2. ARXIV (via WebFetch)                                                ║
║     → Preprints, geralmente 6-18 meses à frente do peer-review           ║
║     → Especialidade: ML/DL aplicado, técnicas emergentes                  ║
║                                                                          ║
║  3. BIORXIV / MEDRXIV (MCP claude.ai)                                   ║
║     → Pré-prints biológicos (relevante para geofísica via geofluidos)    ║
║                                                                          ║
║  4. WEBSEARCH                                                            ║
║     → Documentação de bibliotecas, blogs técnicos, tutoriais              ║
║     → Especialidade: prática, exemplos de código                          ║
║                                                                          ║
║  5. GOOGLE SCHOLAR (via WebFetch headless)                              ║
║     → Backup/cross-check de citações                                      ║
║                                                                          ║
║  6. CONTEXT7 MCP (resolver libraries)                                   ║
║     → Documentação atual de TF, JAX, Numba, etc.                         ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 5.3 Definição Completa do Agente Pesquisador

```yaml
---
name: geosteering-research
description: |
  Pesquisador científico do projeto Geosteering AI. Acessa Consensus,
  ArXiv, bioRxiv, WebSearch e Context7 para validar decisões arquiteturais
  e implementação com base em literatura 2023-2026.
  Acionar para: justificar nova arquitetura DL, validar técnica de
  otimização Numba/JAX, buscar prior art em inversão geofísica,
  pesquisar tendências (e.g., FNO, INN, ModernTCN, G-Query, Evidential UQ).
tools:
  - WebSearch
  - WebFetch
  - Read
  - mcp__claude_ai_Consensus__search
  - mcp__claude_ai_bioRxiv__search_preprints
  - mcp__plugin_context7_context7__query-docs
  - mcp__plugin_context7_context7__resolve-library-id
model: claude-sonnet-4-6
constraints:
  - Citações obrigatórias com formato Author (Year, Journal/ArXiv ID)
  - Quando usar Consensus, incluir mensagem de upgrade word-for-word
  - Batch ≤3 calls por vez (rate limit)
  - Não inventar URLs; usar exatamente os retornados
---

# Pesquisador Científico Geosteering AI

## Workflow Padrão

  1. ENTENDER pergunta: domínio (geofísica/DL/otim), escopo, urgência
  2. ESCOLHER fontes:
       Geofísica clássica  → Consensus (peer-reviewed)
       ML emergente        → ArXiv (preprints recentes)
       Biblioteca/API      → Context7 + WebSearch
       Datasets/dataset    → bioRxiv/medRxiv (raros) + WebSearch
  3. EXECUTAR queries em paralelo (até 3)
  4. SINTETIZAR achados em estrutura:
       Hipótese 1: <claim> (referência X, Y)
       Hipótese 2: <claim alternativo> (referência Z)
       Recomendação: <qual seguir e por quê>
  5. CITAR inline: [1], [2], etc. + lista numerada ao final

## Exemplo de Prompt Ideal

  Pergunta: "Devemos usar FLAT prange ou tile/block para paralelizar
  multi-frequência em Numba? Qual o estado da arte?"

  Workflow:
    Consensus(query="prange flat tile block scheduling Numba",
              year_min=2022, max_results=5)
    ArXiv(query="dynamic scheduling parallel kernels HPC LLVM 2024-2026")
    WebSearch("Numba prange chunksize tutorial best practices")

  Saída esperada:
    [1] Lam et al. (2022, IEEE HPC) — work-stealing default em Numba
        é ótimo para tarefas de duração variável >100μs
    [2] Smith (2024, ArXiv:2403.xxxxx) — FLAT prange supera nested
        para n_tasks > 16 × n_threads
    [3] Numba docs (Context7) — chunksize="static" só faz sentido
        para tarefas uniformes < 50μs

    Recomendação: FLAT prange é apropriado para nosso caso
    (n_tasks = nf × n_combos × n_pos típicamente >> 1000;
    duração de hmd_tiv+vmd ~200μs). [1][2][3]

## Citações Obrigatórias

Formato em código (docstrings):
  Note:
      Ref: Morales et al. (2025) "Anisotropic resistivity estimation..."
      Computers & Geosciences, 196, 105786.
      Ardizzone et al. (2019) "Analyzing INNs for Real-World Inverse
      Problems", arXiv:1907.02392.

Formato em MD reports:
  Achado validado por [1] e [2]:
    [1] Morales et al. (2025), Computers & Geosciences 196:105786
    [2] Liu et al. (2024), Remote Sensing 16:62
```

### 5.4 Casos de Uso Concretos no Projeto

| Decisão Arquitetural Pendente | Pergunta para Pesquisador |
|:------------------------------|:--------------------------|
| Sprint 22.1 — FLAT prange | "Best practices Numba multi-dim parallelism 2024+" |
| F4.1 — INN UQ | "INN normalizing flows posterior inference geofísica" |
| F4.2 — G-Query Transformer | "G-Query unified transformer multimodal geophysical" |
| F4.6 — Evidential Regression | "Evidential deep learning vs MC dropout uncertainty" |
| Petrofísica PINN | "Archie law PINN deep learning resistivity inversion" |
| 2D Born/MEF | "1D vs 2D Born approximation borehole resistivity LWD" |
| Real-time SCADA | "WITSML LWD real-time deep learning inversion industry" |
| MLOps tracking | "MLflow vs W&B geophysical ML model registry" |

### 5.5 Memória de Pesquisa

```
~/.claude/projects/-Users-daniel-Geosteering-AI/memory/research/
├── citations_index.md      ← Lista mestra de papers indexados
├── research_2026-Q2.md     ← Achados do trimestre
├── topic_pinns.md          ← Pesquisa específica sobre PINNs
├── topic_inn_uq.md         ← INN para UQ
├── topic_2d_inversion.md   ← Inversão 2D/2.5D/3D
└── topic_mlops_geophys.md  ← MLOps em geofísica
```

Esta memória é atualizada pelo agente Pesquisador após cada sessão; serve
de cache para evitar re-buscar papers já lidos.

---

## 6. Agente de Documentação Automatizada

### 6.1 Justificativa

O projeto exige documentação rica (padrão D1-D14, riqueza C28) em PT-BR
acentuado. Manter manualmente é insustentável. O Agente Documentador:

1. Gera/atualiza relatórios `docs/reports/v{X}_{date}.md` automaticamente
   após sprints (≥5 commits ou bump de versão).
2. Atualiza `docs/CHANGELOG.md`, `docs/ROADMAP.md`, `CLAUDE.md` (linha SM).
3. Atualiza `~/.claude/.../memory/MEMORY.md` (índice) e cria entradas
   `project_*.md` específicas.
4. Verifica acentuação PT-BR em comentários e docstrings de arquivos
   editados.
5. Garante cobertura D1-D14 em arquivos novos.

### 6.2 Definição Completa

```yaml
---
name: geosteering-documentation
description: |
  Documentador automatizado do Geosteering AI. Gera relatórios técnicos,
  atualiza CHANGELOG/ROADMAP/CLAUDE.md, cria entradas de memória e
  verifica conformidade PT-BR + padrões D1-D14.
  Acionar via: hooks Stop e PostToolUse, ou explicitamente após sprint.
tools: [Read, Write, Edit, Bash]
model: claude-haiku-4-5-20251001
allowed_paths:
  - docs/**
  - CLAUDE.md
  - "~/.claude/projects/-Users-daniel-Geosteering-AI/memory/**"
constraints:
  - PT-BR com acentuação correta (regra inviolável CLAUDE.md)
  - Template obrigatório: .claude/templates/report_template.md
  - ≥70% conteúdo estruturado (tabelas, listas, código)
  - ≤30% prosa
  - Nunca substituir CHANGELOG; sempre append
  - Nunca tocar código fonte (.py); apenas docs/markdown
---

# Documentador Geosteering AI

## Disparadores Automáticos

  Hook Stop (ao final de sessão):
    SE houve ≥5 commits desde último report em docs/reports/:
      → Gerar docs/reports/v{X}_{date}.md
      → Atualizar CHANGELOG.md (append)
      → Atualizar memory/MEMORY.md (1 linha < 200 char)
      → Criar memory/project_<scope>.md se aplicável
    SE houve bump de versão em pyproject.toml ou __version__:
      → Mesmo fluxo + atualizar CLAUDE.md linha SM

  Hook PostToolUse(Edit|Write) em geosteering_ai/**/*.py:
    SE arquivo edita docstrings sem acentuação correta:
      → Reportar mas não bloquear (Haiku via Prompt hook)
    SE arquivo novo sem mega-header D1:
      → Adicionar header em commit subsequente

## Template de Relatório (estrutura obrigatória)

  # {versão} {data} — {título}

  | Campo | Valor |
  |:------|:------|
  | Versão | v2.X |
  | Data | YYYY-MM-DD |
  | Branch | feat/... |
  | Commits | <hash1>..<hash2> |
  | Testes | XX/XX PASS |
  | Paridade Fortran | <1e-12 |

  ## §1 Sumário Executivo
  ## §2 Diagnóstico (se fix)
  ## §3 Implementação (1 subseção por sprint interno)
  ## §4 Métricas (tabela antes/depois)
  ## §5 Riscos e Mitigações
  ## §6 Próximos Passos

## Verificação PT-BR

  Erros comuns a detectar:
    'implementacao' → 'implementação'
    'configuracao' → 'configuração'
    'funcao' → 'função'
    'nao' → 'não'
    'ja' → 'já'
    'codigo' → 'código'
    'analise' → 'análise'
    'producao' → 'produção'
    'reducao' → 'redução'
    'transmissao' → 'transmissão'
    [+ ~50 outras palavras frequentes]

  Quando detectar (em hook Prompt):
    REPORTAR: "Arquivo X linha N: 'implementacao' deveria ser 'implementação'"
    NÃO BLOQUEAR (apenas alertar)

## Anti-padrões a Evitar

  ✗ Reescrever CHANGELOG inteiro (sempre append)
  ✗ Templates "criativos" (usar SEMPRE report_template.md)
  ✗ Prosa longa (>30% do documento)
  ✗ Linhas em MEMORY.md > 200 caracteres
  ✗ Documentar antes do código existir (gera drift)
  ✗ Editar código fonte (.py) — escopo é docs only
```

### 6.3 Cobertura D1-D14 Verificada

```
D1   — Mega-header Unicode com 14 campos no topo .py
D2   — Cabeçalho de seção 4+ linhas
D3   — Diagramas ASCII com Unicode borders
D4   — Atributos config 4+ linhas por grupo
D5   — Docstrings Google-style 5+ campos
D6   — Docstrings classe + Attributes + Example
D7   — Comentários inline semânticos
D8   — __all__ semântico
D9   — Logging estruturado (proíbe print)
D10  — Constantes com documentação física
D11  — Tabelas ASCII em catálogos
D12  — Cross-references "Note:" em docstrings
D13  — Branch comments com layout de saída
D14  — Diagrama noise × FV × GS em pipeline.py
```

O agente Documentador roda checagem automática D1, D5, D8, D9 em hooks.
Os demais (D2-D7, D10-D14) são verificados em revisão de PR pelo
`geosteering-code-reviewer`.

---

## 7. Agentes do Simulador — Numba + JAX

Esta seção detalha como os agentes do simulador trabalham juntos para
cobrir **todos os cenários** de execução discutidos em
`docs/reference/analise_cenarios_otimizacao_simulador_numba.md`.

### 7.1 Mapeamento Cenário → Agente Responsável

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CENÁRIO BÁSICO  │ AGENTE PRIMÁRIO       │ AGENTES DE QUALIDADE          │
├──────────────────────────────────────────────────────────────────────────┤
│  C1 (1f,1a,1TR)  │ Numba                 │ Físico + Perf                 │
│  C2 (mf,1a,1TR)  │ Numba (FLAT prange)   │ Físico + Perf                 │
│  C3 (1f,ma,1TR)  │ Numba                 │ Físico                        │
│  C4 (1f,1a,mTR)  │ Numba                 │ Físico                        │
│  C5 (1f,ma,mTR)  │ Numba                 │ Físico                        │
│  C6 (mf,ma,1TR)  │ Numba (FLAT prange)   │ Físico + Perf                 │
│  C7 (mf,1a,mTR)  │ Numba (FLAT prange)   │ Físico + Perf                 │
│  C8 (mf,ma,mTR)  │ Numba (FLAT prange)   │ Físico + Perf                 │
├──────────────────────────────────────────────────────────────────────────┤
│  GPU (qualquer)  │ JAX (vmap_real)       │ Físico + Perf (Colab T4/A100) │
│  Batch (treino)  │ Workers (Numba ou JAX)│ Perf + MLOps (Colab logs)     │
│  F6 (compensação)│ Numba postprocess     │ Físico                        │
│  F7 (tilted)     │ Numba postprocess     │ Físico                        │
│  Alta ρ (>1k)    │ Numba/JAX (Anderson)  │ Físico (modelos canônicos novos)│
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Roteiro de Otimização v2.22 → v2.30

Conforme `analise_cenarios_otimizacao_simulador_numba.md`, o roadmap do
simulador Python para os próximos 6-9 meses:

| Versão | Sprint | Otimização Principal | Agente | Ganho Esperado |
|:------:|:------:|:--------------------|:-------|:--------------:|
| v2.22 | 22.1 | FLAT prange(nf × n_combos × n_pos) | Numba | C2/C6/C7/C8: 1.5-3× |
| v2.23 | 23.1 | Adaptive thread count (n_pos baixo) | Numba | C1 baixo n_pos: 1.1-1.4× |
| v2.23 | 23.2 | fastmath em hmd_tiv/vmd (gate) | Numba + Físico | +20% se gate <1e-12 mantido |
| v2.24 | 24.1 | Tile/block scheduling | Numba | Cenários nested: +15-25% |
| v2.25 | 25.1 | Pré-cômputo Hankel TE/TM | Numba | +10-15% |
| v2.26 | 26.1 | High-resistivity validation | Físico + Numba | Cobertura ρ≤1e5 Ω·m |
| v2.27 | 27.1 | JAX vmap_real flip default | JAX | Default GPU ativo |
| v2.28 | 28.1 | F7 vetorizado (np.einsum) | Numba | F7 alto: 3× |
| v2.29 | 29.1 | Adaptive batch chunking | Workers | Lotes heterogêneos: +20% |
| v2.30 | 30.1 | Backend abstrato + plugin discovery | Orchestrator | Prepara 2D/2.5D |

Cada sprint é executado pelo agente primário em **worktree isolada**, com
revisão paralela por Físico + Perf, e documentação automatizada pelo
Documentador.

### 7.3 Padrão de Execução de Sprint do Simulador

```
SPRINT v2.22 — FLAT prange
═══════════════════════════════════════════════════════════════════════

FASE 1 — PLANEJAMENTO (Orquestrador Opus 4.7, 15 min)
  Inputs:
    - docs/reference/analise_cenarios_otimizacao_simulador_numba.md
    - geosteering_ai/simulation/forward.py (673 LOC)
    - geosteering_ai/simulation/_numba/kernel.py (873 LOC)
  Output:
    - Plano em /Users/daniel/.claude/plans/sprint-22-1.md
    - TodoWrite com 12 itens
    - Branch criada: feat/simulator-v2.22

FASE 2 — EXPLORAÇÃO PARALELA (3 agentes em fan-out, 5 min)
  Agent(Pesquisador): "FLAT prange Numba 2024+ best practices"
  Agent(Explore): "Como _fields_in_freqs_kernel_cached é chamado?"
  Agent(Explore): "Onde simulate_multi despacha entre kernels?"

FASE 3 — IMPLEMENTAÇÃO ISOLADA (Numba Opus, worktree, 60-90 min)
  isolation: worktree
  Edits:
    forward.py:351-468 → adicionar _simulate_combined_prange_flat
    multi_forward.py:XXX → dispatcher para escolher kernel
    config.py → flag use_flat_prange (opt-in inicial)
  Tests:
    tests/test_simulation_flat_prange.py (5 novos)
    tests/test_simulation_compare_fortran.py (deve passar)

FASE 4 — VALIDAÇÃO PARALELA (Físico + Perf, 20 min)
  Agent(Físico Reviewer):
    - Read diff completo
    - Run pytest -k fortran_python_numba
    - Confirmar paridade <1e-12 em 7 modelos canônicos
  Agent(Perf Reviewer, MCP numba-profiler):
    - bench Cenário A, E (regressão)
    - bench Cenário F=novo (nf=4, n_pos=600)
    - bench Cenário G=novo (nf=4, n_pos=30)
    - Mediana de 5 runs

FASE 5 — REVIEW DE CÓDIGO (Code Reviewer Sonnet, 15 min)
  - PEP8, tipagem, D1-D14
  - Logging consistente
  - Sem print(), sem globals()

FASE 6 — DOCUMENTAÇÃO (Documentador Haiku, 20 min)
  - docs/reports/v2.22_2026-MM-DD.md
  - CHANGELOG.md (append)
  - ROADMAP.md (linha v2.22)
  - CLAUDE.md (linha SM atualizada)
  - .claude/commands/geosteering-simulator-python.md (§22)
  - memory/project_simulation_manager_v222.md
  - memory/MEMORY.md (1 linha)

FASE 7 — COMMIT + PR (Orquestrador, 10 min)
  Commits granulares (sequência):
    feat(sm): v2.22 — FLAT prange kernel
    test(sm): v2.22 — 5 novos testes regressão
    perf(sm): v2.22 — bench Cenários F, G adicionados
    docs(sm): v2.22 — relatório técnico + CHANGELOG + ROADMAP
  PR feat/simulator-v2.22 → main
```

### 7.4 Workflow JAX em GPU

```
SPRINT JAX (e.g., v2.27 flip default vmap_real)
═══════════════════════════════════════════════════════════════════════

FASE 1 — VALIDAÇÃO LOCAL (Numba Opus + JAX Opus, paralelo)
  Confirmar paridade JAX vs Numba <1e-10 em CPU local
  pytest tests/test_simulation_jax_*.py

FASE 2 — VALIDAÇÃO COLAB T4 (manual via Colab notebook)
  notebooks/validate_jax_t4_v227.ipynb
  Comparar:
    cfg.jax_strategy="bucketed", jax_vmap_real=False  (v1.6.0 default)
    cfg.jax_strategy="unified",  jax_vmap_real=True   (v2.27 proposto)
  Métricas: paridade <1e-10 + speedup ≥1.5×

FASE 3 — VALIDAÇÃO COLAB A100 (manual)
  Mesmo notebook, runtime A100
  Confirmar: paridade preservada em fp64, speedup ≥3×

FASE 4 — FLIP DEFAULT
  config.py: defaults jax_strategy="unified", jax_vmap_real=True
  Documentar breaking change em CHANGELOG (com migration guide)

FASE 5 — COLAB MCP MONITORING
  /loop 30m check-jax-validation
    Consulta GitHub Actions logs do CI matriz GPU
    Reporta status para Daniel
```

### 7.5 Cenários Edge: Alta Resistividade

```
SPRINT v2.26 — Validação ρ > 1000 Ω·m
═══════════════════════════════════════════════════════════════════════

Novos modelos canônicos (Físico + Numba):
  carbonato_seco_5c   : 5 camadas, ρ_max = 5.000 Ω·m (TIV 2:1)
  evaporita_3c       : 3 camadas, ρ_max = 100.000 Ω·m (sal halita)
  gas_seco_8c         : 8 camadas, ρ_max = 10.000 Ω·m
  basalto_intrusivo_6c: 6 camadas, ρ_max = 50.000 Ω·m

Pipeline de validação:
  1. Físico Reviewer projeta modelos via Tex_Projects/TatuAniso/
  2. Numba agente roda forward Python e compara com Fortran tatu.x
  3. Se ρ > 10.000 Ω·m e paridade > 1e-12 com Werthmuller 201pt:
     → Tentar Anderson 801pt
     → Se mesmo assim falhar: documentar limitação física
  4. Adicionar testes em tests/test_simulation_high_resistivity.py
  5. Atualizar análise_cenarios doc §7 com novos números

Fastmath gate (quando avaliar v2.23 fastmath):
  Para CADA novo modelo de alta ρ:
    ASSERT max|H_py(fastmath=True) - H_fortran| / max|H_fortran| < 1e-12
  Se algum modelo falhar:
    fastmath = False (manter v2.23 atual em hmd_tiv/vmd)
    fastmath = True apenas em hankel.py (já implementado, comprovado seguro)
```

---

## 8. Agentes de Deep Learning + PINNs

### 8.1 Composição: DL Pipeline + PINNs

O pipeline DL é vasto (48 arqs, 26 losses, 8 cenários PINN). Dois agentes
especializados dividem responsabilidades para evitar sobrecarga cognitiva:

```
╔══════════════════════════════════════════════════════════════════════════╗
║  AGENTE DL PIPELINE (Sonnet 4.6)                                        ║
║  Escopo: arquiteturas, losses gerais (não-PINN), training loop,          ║
║         callbacks, inferência offline, HPO, métricas.                    ║
║  Principais arquivos:                                                    ║
║    geosteering_ai/models/{cnn,rnn,hybrid,tcn,transformer,unet,           ║
║                            decomposition,advanced,geosteering,           ║
║                            surrogate,blocks,registry}.py                 ║
║    geosteering_ai/losses/{catalog,factory,geophysical}.py               ║
║    geosteering_ai/training/{loop,callbacks,nstage,metrics,               ║
║                              optuna_hpo,adaptation}.py                   ║
║    geosteering_ai/inference/{pipeline,realtime,export,uncertainty}.py    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  AGENTE PINNS (Opus 4.7)                                                ║
║  Escopo: cenários PINN físicos, TIV constraint layer,                    ║
║         lambda schedules, surrogate analítico/neural,                    ║
║         validação contra forward Maxwell.                                ║
║  Principais arquivos:                                                    ║
║    geosteering_ai/losses/pinns.py                                        ║
║    geosteering_ai/models/surrogate.py (SurrogateNet TCN/ModernTCN)       ║
║    geosteering_ai/models/blocks.py (TIVConstraintLayer)                  ║
║    geosteering_ai/data/surrogate_data.py (extração pares)                ║
║                                                                          ║
║  Por que Opus? PINNs envolvem:                                           ║
║    • Cálculo simbólico em loss function                                  ║
║    • Trade-offs físicos finos (peso de constraint vs L_data)             ║
║    • Validação contra equações de Maxwell                                ║
║    • Integração com simulador (surrogate vs forward analítico)           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 8.2 Catálogo Completo de PINNs (8 cenários)

Conforme implementado em `losses/pinns.py`:

| Cenário | Constraint | Aplicação |
|:--------|:-----------|:----------|
| `oracle` | Forward analítico 1D (NumPy/TF) | Baseline; validação física |
| `surrogate` | Forward via SurrogateNet neural | Treino on-the-fly em escala |
| `maxwell` | ∇×H = J + ∂D/∂t (residual EM) | Robustez a outliers |
| `smoothness` | TV-L1 sobre log10(ρ_h), log10(ρ_v) | Reduz overfitting em ruído |
| `skin_depth` | δ = √(2/(ωμσ)) consistência | Frequências múltiplas |
| `continuity` | ∂ρ/∂z contínua entre camadas | Modelos sem descontinuidade |
| `variational` | Forma fraca da PDE | Robustez numérica |
| `self_adaptive` | Pesos λ aprendidos por região | Auto-balanceamento |

### 8.3 Workflow PINN Completo

```
╔══════════════════════════════════════════════════════════════════════════╗
║  TREINO COM PINN — WORKFLOW MULTI-AGENTE                                ║
║                                                                          ║
║  1. PESQUISADOR (Sonnet)                                                ║
║     "Estado da arte PINN para inversão EM 1D 2024-2026"                 ║
║     → identifica papers como Morales (2025), Liu (2024)                  ║
║                                                                          ║
║  2. PINNS AGENT (Opus)                                                  ║
║     • Define cenário (oracle/surrogate/maxwell/...)                      ║
║     • Define lambda schedule (fixed/linear/cosine/step)                  ║
║     • Implementa loss em losses/pinns.py                                 ║
║     • Cria teste com gradiente analítico verificado                      ║
║                                                                          ║
║  3. DL PIPELINE AGENT (Sonnet)                                          ║
║     • Wrappa em LossFactory                                              ║
║     • Configura callback de monitoramento de termos PINN                 ║
║     • Adiciona ao TrainingLoop                                           ║
║                                                                          ║
║  4. FÍSICO REVIEWER (Sonnet)                                            ║
║     • Verifica unidades (A/m, Ω·m, dB, °)                               ║
║     • Verifica simetria (TIV: ρ_v ≥ ρ_h)                                ║
║     • Verifica gradiente bem-definido em ρ → 0 e ρ → ∞                  ║
║                                                                          ║
║  5. PERF REVIEWER (Haiku)                                               ║
║     • Mede overhead PINN no training step                                ║
║     • Compara épocas até convergência (PINN vs baseline)                 ║
║                                                                          ║
║  6. DOCUMENTADOR (Haiku)                                                ║
║     • Atualiza docs/documentacao_pinns.md (já existe)                    ║
║     • Adiciona cenário ao catálogo §8.2                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 8.4 Cenário Pendente: PINN Petrofísica (Archie + Klein)

Conforme F4.5 do ROADMAP — cenário #9 a implementar:

```python
# losses/pinns.py — Cenário "petrophysics"

@PINN_REGISTRY.register("petrophysics")
def make_petrophysics_loss(config: PipelineConfig):
    """Cenário PINN baseado em Lei de Archie + Klein.

    Constraint petrofísica:
        ρ_t = a × R_w / (φ^m × S_w^n)         (Archie 1942)

    onde:
        ρ_t : resistividade total (predita)
        a   : tortuosidade (~1)
        R_w : resistividade da água de formação (Ω·m)
        φ   : porosidade (frac volume)
        m   : expoente de cimentação (1.8–2.5)
        n   : expoente de saturação (~2)
        S_w : saturação de água (frac volume)

    Loss combinada:
        L = L_data(ρ_pred, ρ_true) +
            λ_archie · |log(ρ_pred) - log(a × Rw / (φ^m × Sw^n))|

    Pré-requisito: dataset de campo com ρ_t + parâmetros petrofísicos
    medidos. Status: aguardando dataset (lacuna F4.5 ROADMAP).
    """
```

---

## 9. Agentes de Dados — Sintéticos e Reais

### 9.1 Dois Universos de Dados

```
╔══════════════════════════════════════════════════════════════════════════╗
║  UNIVERSO 1 — DADOS SINTÉTICOS (existente, robusto)                     ║
║                                                                          ║
║  Fonte: Simulador Fortran tatu.x + Simulador Python Numba/JAX            ║
║  Formato: 22-col binário .dat + .out metadata                            ║
║  Pipeline: load → decoupling → split (P1) → on-the-fly                  ║
║            (noise → FV → GS → scale)                                     ║
║                                                                          ║
║  Datasets atuais:                                                        ║
║    Inv0_Dip2000_teste.dat  (15 modelos, 600 pts/modelo, dip=0°)         ║
║    arranjoTR1_60k          (60k modelos sintéticos)                      ║
║    Multi-TR e Multi-dip:   pendente F5 ROADMAP                           ║
║                                                                          ║
║  Capacidade de geração on-the-fly:                                       ║
║    ProcessPoolExecutor + Numba: ~120k mod/h Cenário E                    ║
║    Para 30k modelos: ~15 minutos                                         ║
║    Para 100k modelos: ~50 minutos                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  UNIVERSO 2 — DADOS REAIS (preparação, ainda não consumidos)             ║
║                                                                          ║
║  Fontes futuras:                                                         ║
║    • Logs LWD em formato WITSML (XML padrão indústria)                   ║
║    • Logs LAS (formato texto LIDAR LWD)                                  ║
║    • Streams OPC-UA real-time de SCADA de plataforma                     ║
║    • Datasets públicos: Volve (Equinor), Force 2020, OSDU                ║
║                                                                          ║
║  Tarefas de preparação:                                                  ║
║    • Loader WITSML/LAS (a implementar em geosteering_ai/data/loaders/)   ║
║    • Conversor para layout 22-col interno                                ║
║    • Curva de calibração (campos LWD reais ≠ campos simulados)           ║
║    • Domain adaptation (DomainAdapter já existe em training/)            ║
║    • Validação por Físico Reviewer + comparação com inversão            ║
║      tradicional (Gauss-Newton ou Levenberg-Marquardt)                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 9.2 Roadmap de Suporte a Dados Reais

```
═══════════════════════════════════════════════════════════════════════════
  FASE A — LOADER WITSML/LAS (futuro, dependência F6)
═══════════════════════════════════════════════════════════════════════════
  Implementar:
    geosteering_ai/data/loaders/witsml.py
      def load_witsml(path: str) -> RealDataBundle
        - Parse XML via lxml
        - Extrai curvas: depth, time, RT (resistividade total),
          gamma, density, neutron, sonic, dip, azimuth
        - Retorna RealDataBundle (depth_m, lwd_curves, metadata)
    geosteering_ai/data/loaders/las.py
      def load_las(path: str) -> RealDataBundle
        - Parse via biblioteca lasio
        - Mesmo retorno

  Conversor para 22-col interno:
    geosteering_ai/data/loaders/real_to_22col.py
      def real_to_22col(bundle: RealDataBundle, config: PipelineConfig) -> np.ndarray
        - Mapeia curvas LWD para layout 22-col esperado pelo pipeline
        - Aplica calibração (offsets, ganhos)
        - Inserção de zeros nas colunas não-aplicáveis (e.g.,
          componentes do tensor não medidas pela ferramenta)

═══════════════════════════════════════════════════════════════════════════
  FASE B — CALIBRAÇÃO E DOMAIN ADAPTATION
═══════════════════════════════════════════════════════════════════════════
  Pipeline:
    1. Treinar baseline em sintéticos (F2 ROADMAP)
    2. Inferir em N poços reais (com inversão tradicional como ground truth)
    3. Identificar viés sistemático (média, variância)
    4. DomainAdapter (já existe em training/adaptation.py):
       - Fine-tune com 5-10% de dados reais
       - Progressive unfreezing
    5. Validar em poços hold-out (não vistos no fine-tune)

═══════════════════════════════════════════════════════════════════════════
  FASE C — STREAMING REALTIME (OPC-UA)
═══════════════════════════════════════════════════════════════════════════
  Implementar:
    geosteering_ai/data/loaders/opcua.py
      class OPCUAStream:
          def connect(server_url, auth)
          def subscribe(curve_names) -> AsyncIterator[Sample]

    geosteering_ai/inference/realtime_stream.py
      class StreamInferenceSession:
          - Buffer FIFO de N amostras
          - Sliding window inference em cada nova amostra
          - Output em outro stream OPC-UA (predicted_rho_h, _v, dtb, sigma)

  Latência alvo: <100 ms por amostra (depth resolution ~0.15 m)
```

---

## 10. Agentes de Geosteering em Tempo Real

### 10.1 Modos Operacionais

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MODO OFFLINE                       │  MODO REALTIME (Geosteering)        │
├──────────────────────────────────────────────────────────────────────────┤
│  • Batch completo (n, seq, feat)    │  • Sliding window (1, W, feat)      │
│  • 48 arquiteturas disponíveis      │  • 27 arqs causais compatíveis      │
│  • Padding "same" (Conv1D)          │  • Padding "causal" (Conv1D)        │
│  • Acausal (vê passado + futuro)    │  • Causal (só passado, FIFO)        │
│  • UQ opcional (MC Dropout/Ens.)    │  • UQ obrigatório (NLL automático)  │
│  • Latência: minutos (batch)        │  • Latência: <100 ms por amostra    │
│  • Caso de uso: análise post-poço   │  • Caso de uso: decisão durante drill│
└──────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Arquiteturas Causais Nativas (5)

Em `geosteering_ai/models/geosteering.py`:

```
1. WaveNet           — Dilated Causal Conv1D (van den Oord 2016)
2. Causal_Transformer— Self-attention com mask causal
3. Informer          — Sparse attention + ProbSparse self-attention
4. Mamba_S4          — State Space Model (Gu & Dao 2023)
5. Encoder_Forecaster— Encoder-only com decoder de forecasting
```

### 10.3 Workflow de Sprint Realtime

```
SPRINT — Latência <100ms em Mamba_S4
═══════════════════════════════════════════════════════════════════════

1. ORQUESTRADOR (Opus): plano + TodoWrite
2. PESQUISADOR (Sonnet): "Mamba inference latency optimization 2024+"
3. REALTIME AGENT (Sonnet):
   - Profile inference path com tf.profiler
   - Identificar gargalos: Conv1D padding, MatMul, RNN cells
   - Otimizar via tf.function(jit_compile=True)
4. PERF REVIEWER (Haiku):
   - Bench inference (batch=1, W=600) em CPU + GPU T4
   - Reportar p50, p95, p99 latency
5. CÓDIGO REVIEWER (Sonnet): PEP8, types
6. DOCUMENTADOR (Haiku): update docs/documentacao_geosteering.md
```

### 10.4 Visualização Realtime

`geosteering_ai/visualization/geosteering.py` já implementa:
- Curtain plot (resistividade vs depth + bandas de incerteza)
- DTB (Distance To Boundary) marker
- Look-ahead acuracy (futuro vs predito a N amostras)

Pendente: integração com **dashboard web** (F6.3 ROADMAP) via Streamlit
ou Grafana, para uso operacional fora do ambiente Python local.

---

## 11. Agentes de MLOps + Produção

### 11.1 Pilares de MLOps para Geosteering AI

```
╔══════════════════════════════════════════════════════════════════════════╗
║  1. EXPERIMENT TRACKING                                                  ║
║     • MLflow (preferido, open-source) ou Weights & Biases                ║
║     • Tracking automático de: config (YAML), seed, dataset version,      ║
║       código (tag GitHub), métricas (R², val_loss), artefatos (.keras), ║
║       gráficos (training curves, holdout)                                ║
║                                                                          ║
║  2. MODEL REGISTRY                                                       ║
║     • Versionamento de modelos treinados                                 ║
║     • Estágios: dev → staging → production                               ║
║     • Aprovação manual antes de produção                                 ║
║     • Rollback rápido se métricas em produção degradarem                 ║
║                                                                          ║
║  3. DATA VERSIONING                                                      ║
║     • DVC (Data Version Control) para datasets                           ║
║     • Hashes de .dat/.out → git-trackable sem upload de binários        ║
║                                                                          ║
║  4. SERVING                                                              ║
║     • TF Serving (gRPC + REST) para inferência batch                     ║
║     • FastAPI + Uvicorn para inferência realtime + UQ                    ║
║     • Containerização Docker (multi-stage build, CPU + GPU)              ║
║                                                                          ║
║  5. MONITORING                                                           ║
║     • Latency p50/p95/p99 (Prometheus)                                   ║
║     • Predictions distribution (drift detection vs treino)               ║
║     • Throughput (req/s) e error rate                                    ║
║     • Dashboard Grafana                                                  ║
║                                                                          ║
║  6. CI/CD                                                                ║
║     • GitHub Actions: compile + pytest + mypy + paridade Fortran         ║
║     • Build automático de imagem Docker em PR                            ║
║     • Deploy staging em merge para develop                               ║
║     • Deploy production em tag release                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 11.2 Estrutura de API REST (a criar)

```
api/
├── main.py                   ← FastAPI app
├── routers/
│   ├── inference.py          ← POST /predict, /predict/realtime, /uncertainty
│   ├── models.py             ← GET /models, /models/{name}/metadata
│   ├── health.py             ← GET /health, /ready
│   └── admin.py              ← POST /reload (recarrega modelo)
├── schemas/
│   ├── inference.py          ← Pydantic models para request/response
│   └── common.py
├── services/
│   ├── model_loader.py       ← Carregamento + cache de modelos
│   ├── inference_service.py  ← Wraps InferencePipeline
│   └── monitoring.py         ← Prometheus exporters
├── middleware/
│   ├── auth.py               ← JWT validation
│   ├── logging.py            ← Structured logging
│   └── metrics.py            ← Latency tracking
├── tests/
│   └── test_api_*.py         ← pytest + httpx
└── docker/
    ├── Dockerfile.cpu        ← Build CPU (TF + Numba)
    └── Dockerfile.gpu        ← Build GPU (TF + JAX + CUDA)
```

### 11.3 GitHub Actions — Pipelines

```yaml
# .github/workflows/ci.yml (existente, expandir)
name: CI
on: [push, pull_request]
jobs:
  test:
    matrix:
      python: ["3.12", "3.13"]
    steps:
      - checkout
      - setup-python
      - pip install -e ".[dev,all]"
      - pytest tests/ -v --tb=short
      - mypy geosteering_ai/
      - ruff check geosteering_ai/

  fortran-parity:  # NOVO
    runs-on: ubuntu-latest
    steps:
      - checkout (com submódulo Fortran)
      - apt install gfortran
      - make -C Fortran_Gerador
      - pytest tests/test_simulation_compare_fortran.py -v
        # Gate: <1e-12 nos 7 modelos canônicos

  build-docker:  # NOVO (somente em PR para main)
    if: github.event_name == 'pull_request' && github.base_ref == 'main'
    runs-on: ubuntu-latest
    steps:
      - docker build -f api/docker/Dockerfile.cpu -t geosteering-ai:cpu-${{github.sha}}
      - docker push (registry interno)

# .github/workflows/release.yml (NOVO)
name: Release
on:
  push:
    tags: ['v*']
jobs:
  build-and-publish:
    - python -m build
    - twine upload (PyPI)
    - docker build + push (production tag)
    - deploy-staging (kubectl)
    - smoke-test-staging
    - manual-approval
    - deploy-production
```

### 11.4 Workflow de Deploy

```
1. PR feature/X → develop
   • CI roda compile + pytest + mypy + paridade Fortran
   • Build Docker (não publica)
   • Code review obrigatório

2. Merge develop → main
   • CI roda completo
   • Tag automática se commit message contém "release vX.Y.Z"

3. Tag v2.X.Y push:
   • Build Docker production
   • Deploy staging (Kubernetes)
   • Smoke test (curl /health, /predict com payload sintético)
   • Aprovação manual no GitHub Actions
   • Deploy production (rolling update)

4. Monitoring em production:
   • /loop 1h (Haiku): consulta Grafana, reporta latência p99
     se >150ms → alerta Daniel via Slack
   • /loop 1d (Haiku): consulta drift detection,
     reporta se distribuição de inputs mudou >2σ
```

---

## 12. Agentes de Segurança e Qualidade

### 12.1 Camada de Defesa em Profundidade

```
╔══════════════════════════════════════════════════════════════════════════╗
║  L1 — HOOKS PRE-TOOL (determinístico, instantâneo, sem custo)           ║
║                                                                          ║
║  • protect-critical-files.sh  → bloqueia Edit em paths sensíveis         ║
║  • validate-physics.sh        → bloqueia PyTorch, eps inseguros, etc.    ║
║                                                                          ║
║  Vantagem: zero latência, zero custo de IA, prevenção 100% determinística║
║  Limitação: apenas regras conhecidas a priori                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  L2 — HOOKS POST-TOOL (determinístico + LLM se necessário)              ║
║                                                                          ║
║  • compile-check.sh          → py_compile no arquivo modificado          ║
║  • lint-v2-standards.sh      → padrões D1-D14                            ║
║  • autoformat.sh             → ruff format + isort                       ║
║                                                                          ║
║  Vantagem: detecta erros antes do commit                                 ║
║  Limitação: roda APÓS edit, não previne digitação                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  L3 — REVIEWERS LLM (julgamento contextual)                             ║
║                                                                          ║
║  • geosteering-physics-reviewer    (Sonnet)                              ║
║  • geosteering-perf-reviewer       (Haiku)                               ║
║  • geosteering-code-reviewer       (Sonnet)                              ║
║  • geosteering-security-auditor    (Sonnet)                              ║
║                                                                          ║
║  Vantagem: pega erros sutis (lógica, contexto, intenção)                 ║
║  Limitação: custo LLM, não 100% determinístico                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  L4 — CI GITHUB ACTIONS (determinístico, distribuído)                   ║
║                                                                          ║
║  • compile + pytest + mypy + ruff                                        ║
║  • Paridade Fortran <1e-12 (gate obrigatório)                            ║
║  • Build Docker em PR                                                    ║
║                                                                          ║
║  Vantagem: ambiente clean, reprodutível                                  ║
║  Limitação: latência (minutos)                                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  L5 — MONITORING EM PRODUÇÃO (futuro, MLOps)                            ║
║                                                                          ║
║  • Latency p99 < 150ms                                                   ║
║  • Predictions drift detection                                           ║
║  • Error rate < 0.1%                                                     ║
║                                                                          ║
║  Vantagem: pega regressões em dados reais                                ║
║  Limitação: requer infraestrutura completa                               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 12.2 Filosofia: Hooks vs Agentes

| Caso | Hook | Agente |
|:-----|:----:|:------:|
| Verificar PyTorch import | Sim (regex determinística) | Não |
| Validar paridade Fortran | Sim (pytest determinístico) | Não |
| Reportar findings de revisão | Não (LLM é melhor) | Sim |
| Atualizar CHANGELOG | Sim (template) ou Agente Haiku | Ambos |
| Identificar bug sutil em loop | Não | Sim (julgamento) |
| Verificar acentuação PT-BR | Sim (lista de palavras) ou Haiku | Ambos |

**Regra prática**: tudo que pode ser uma regex ou pytest é hook; tudo que
exige julgamento contextual é agente.

---

## 13. Frontend (GUI PyQt6) e CLI

### 13.1 Estado Atual

```
╔══════════════════════════════════════════════════════════════════════════╗
║  GUI — Simulation Manager (PyQt6)                                       ║
║                                                                          ║
║  Localização: geosteering_ai/simulation/tests/simulation_manager.py     ║
║  Versão: v2.21 (mais recente)                                           ║
║  LOC: ~10.500                                                           ║
║                                                                          ║
║  Funcionalidades existentes:                                             ║
║    • Geração de modelos sintéticos (ModelGenerationThread async)        ║
║    • Simulação batch via ProcessPoolExecutor                             ║
║    • Plots: holdout, training curves, error maps, geosteering            ║
║    • Pause/Cancel cooperativo                                            ║
║    • PoolWarmupThread (pré-aquece Numba)                                 ║
║    • UI de configuração (presets, semente, oversubscription warning)    ║
║                                                                          ║
║  Migração concluída v2.7a (PyQt5 → PyQt6 + PySide6 abstração)           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CLI — A criar (geosteering-cli)                                        ║
║                                                                          ║
║  Atualmente: scripts ad-hoc em scripts/ + benchmarks/                    ║
║  Falta: comando unificado geosteering-cli                                ║
║                                                                          ║
║  Visão: CLI baseado em Typer (typer.tiangolo.com)                       ║
║    geosteering-cli simulate --config configs/baseline.yaml ...           ║
║    geosteering-cli train --preset robusto --epochs 100                   ║
║    geosteering-cli infer --model models/v3.keras --input data.dat       ║
║    geosteering-cli benchmark --scenario E --runs 5                       ║
║    geosteering-cli serve --port 8000   # inicia API REST                 ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 13.2 Aprimoramentos GUI Planejados

| Feature | Descrição | Prioridade |
|:--------|:-----------|:----------:|
| Painel de inversão DL | Botão "Inverter com modelo" carrega .keras + faz inferência | Alta |
| Gerenciador de configurações YAML | Editor inline com validação de errata | Média |
| Histórico de runs | SQLite local com runs passados (config, métricas, artefatos) | Média |
| Comparador de runs | Diff entre 2 runs (configs + métricas + plots) | Baixa |
| Tema claro/escuro | Detecção do sistema + override manual | Baixa |
| Integração com Colab | Botão "Enviar para Colab" gera notebook + abre URL | Média |
| Dashboard MLOps embed | Widget que renderiza Grafana via QWebEngineView | Baixa |
| Suporte a streaming OPC-UA | Modo realtime com plot atualizado em tempo real | Alta (futuro) |

### 13.3 Estrutura da CLI (a criar)

```
cli/
├── __init__.py
├── main.py                   ← Typer app
├── commands/
│   ├── simulate.py           ← geosteering-cli simulate
│   ├── train.py              ← geosteering-cli train
│   ├── infer.py              ← geosteering-cli infer
│   ├── benchmark.py          ← geosteering-cli benchmark
│   ├── serve.py              ← geosteering-cli serve (API REST)
│   ├── validate.py           ← geosteering-cli validate (paridade Fortran)
│   └── tools.py              ← geosteering-cli tools <subcommand>
├── utils/
│   ├── config_loader.py      ← Carrega YAML + override CLI flags
│   ├── progress.py           ← Rich progress bars
│   └── output.py             ← Tabelas/JSON/CSV
└── tests/
    └── test_cli_*.py
```

Adicionar entry point em `pyproject.toml`:

```toml
[project.scripts]
geosteering-cli = "cli.main:app"
```

### 13.4 Exemplos de Uso da CLI

```bash
# Simular 1000 modelos no Cenário E e salvar tensor
geosteering-cli simulate \
    --config configs/cenario_E.yaml \
    --models 1000 \
    --output /tmp/sim_E.npz \
    --workers 4 --threads-per-worker 2

# Treinar modelo com preset robusto, 100 épocas
geosteering-cli train \
    --preset robusto \
    --epochs 100 \
    --dataset /Users/daniel/datasets/arranjoTR1_60k \
    --output models/baseline_v3.keras \
    --tracking mlflow:http://mlflow.local:5000

# Inferir em arquivo .dat existente
geosteering-cli infer \
    --model models/baseline_v3.keras \
    --input /Users/daniel/data/well_X.dat \
    --output /tmp/inversion_X.csv \
    --uncertainty mc-dropout --mc-passes 30

# Benchmark Cenário E com 5 runs
geosteering-cli benchmark \
    --scenario E \
    --runs 5 \
    --output benchmarks/results_2026-05-02.json

# Iniciar API REST local
geosteering-cli serve \
    --host 0.0.0.0 --port 8000 \
    --model-dir models/ \
    --reload
```

---

## 14. Distribuição como Módulo/API Python

### 14.1 Visão de Uso como Biblioteca

O Geosteering AI deve ser usável como **módulo Python idiomático** em
qualquer notebook, script ou aplicação:

```python
# Caso 1: Simulação simples
from geosteering_ai.simulation import simulate, SimulationConfig
import numpy as np

cfg = SimulationConfig(
    frequency_hz=20000.0,
    spacing_meters=1.0,
    backend="numba",
)
result = simulate(
    rho_h=np.array([100.0, 1.0, 100.0]),
    rho_v=np.array([100.0, 2.0, 100.0]),
    esp=np.array([0.0, 5.0, 0.0]),
    positions_z=np.linspace(-10.0, 10.0, 600),
    cfg=cfg,
)
H_tensor = result.H_tensor  # (600, 9) complex128
```

```python
# Caso 2: Pipeline completo de inversão
from geosteering_ai import PipelineConfig
from geosteering_ai.data import DataPipeline
from geosteering_ai.models import ModelRegistry
from geosteering_ai.training import TrainingLoop

config = PipelineConfig.robusto()
pipeline = DataPipeline(config)
data = pipeline.prepare("/path/to/dataset")
model = ModelRegistry().build(config)
trainer = TrainingLoop(config, model, pipeline, data)
history = trainer.run()
```

```python
# Caso 3: Inferência em produção (servidor)
from geosteering_ai.inference import InferencePipeline

pipeline = InferencePipeline.load("models/v2.zip")
predictions = pipeline.predict(raw_em_tensor)
# predictions = {"rho_h": ..., "rho_v": ..., "uncertainty": ...}
```

```python
# Caso 4: Geosteering realtime (sliding window)
from geosteering_ai.inference.realtime import GeoSteeringSession

session = GeoSteeringSession(model_path="models/causal_v2.keras",
                              window_size=600)
for sample in lwd_stream:
    pred = session.step(sample)
    if pred.confidence < 0.7:
        print(f"WARNING: depth={sample.depth}, low confidence")
```

### 14.2 API Pública (Stable)

`geosteering_ai/__init__.py` deve expor:

```python
__version__ = "2.0.0"

# Configuração (single source of truth)
from .config import PipelineConfig

# Simulação (subpacote)
from . import simulation

# Pipelines de alto nível
from .data.pipeline import DataPipeline
from .models.registry import ModelRegistry
from .losses.factory import LossFactory
from .training.loop import TrainingLoop
from .inference.pipeline import InferencePipeline
from .inference.realtime import GeoSteeringSession

__all__ = [
    "PipelineConfig",
    "simulation",
    "DataPipeline",
    "ModelRegistry",
    "LossFactory",
    "TrainingLoop",
    "InferencePipeline",
    "GeoSteeringSession",
    "__version__",
]
```

### 14.3 Empacotamento Refinado

```toml
# pyproject.toml (refinar v2.0)

[project]
name = "geosteering-ai"
version = "2.0.0"
description = "Pipeline de Inversão Geofísica 1D com Deep Learning para Geosteering"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.12,<3.15"
authors = [{name = "Daniel Leal"}]
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.2",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
train     = ["tensorflow>=2.13"]
sim-numba = ["numba>=0.60"]
sim-jax   = ["jax>=0.4.30", "jaxlib>=0.4.30"]
viz       = ["matplotlib>=3.5", "PyQt6>=6.5"]
hpo       = ["optuna>=3.0"]
api       = ["fastapi>=0.110", "uvicorn>=0.27", "typer>=0.9"]
mlops     = ["mlflow>=2.10", "prometheus-client>=0.19"]
all       = [
    "tensorflow>=2.13", "numba>=0.60",
    "jax>=0.4.30", "jaxlib>=0.4.30",
    "matplotlib>=3.5", "PyQt6>=6.5",
    "optuna>=3.0",
    "fastapi>=0.110", "uvicorn>=0.27", "typer>=0.9",
    "mlflow>=2.10",
]
dev = [
    "pytest>=7.0", "pytest-cov>=4.0",
    "mypy>=1.0", "ruff>=0.1.0",
    "black>=23.0",
]

[project.scripts]
geosteering-cli = "cli.main:app"

[project.entry-points."geosteering_ai.backends"]
# Plugin discovery para backends 1D existentes + 2D/2.5D/3D futuros
numba_1d   = "geosteering_ai.simulation._numba:Backend"
jax_1d     = "geosteering_ai.simulation._jax:Backend"
fortran_1d = "geosteering_ai.simulation._fortran:Backend"
# Futuro:
# born_2d  = "geosteering_ai.simulation._born_2d:Backend"
# fem_2d   = "geosteering_ai.simulation._fem_2d:Backend"
# fem_3d   = "geosteering_ai.simulation._fem_3d:Backend"
```

### 14.4 Opções de Distribuição

```
1. PyPI público (open-source)
   pip install geosteering-ai

2. PyPI privado (interno empresa)
   pip install --index-url https://pypi.empresa.com/simple geosteering-ai

3. GitHub direto (sem PyPI)
   pip install git+https://github.com/daniel-guitarplayer-8/geosteering-ai.git@v2.0.0

4. Wheel local (para Colab Pro+)
   No notebook:
     !pip install /content/drive/MyDrive/wheels/geosteering_ai-2.0.0-py3-none-any.whl
```

Recomendação inicial: **opção 3** (via GitHub tags), migrar para opção 1
quando v2.0.0 estiver publicamente estável.

---

## 15. Hooks — Automação Determinística (Detalhado)

### 15.1 Inventário Completo

```
Hooks ATUAIS (em produção, .claude/settings.json):
  PreToolUse(Edit|Write):
    [E] validate-physics.sh        — bloqueia PyTorch, eps, globals
    [E] protect-critical-files.sh  — bloqueia paths sensíveis
  PostToolUse(Edit|Write):
    [E] compile-check.sh           — py_compile
    [E] lint-v2-standards.sh       — D1-D14
    [E] autoformat.sh              — ruff + isort
    [E] validate-scientific-refs.sh— citações
  Stop:
    [E] run-pytest.sh              — pytest tests/ -q
  SessionStart(startup):
    [E] setup-environment.sh       — branch, commits, contagem
  SessionStart(compact):
    [E] reinject-errata.sh         — recarregar errata pós-compact
```

### 15.2 Hooks Novos a Criar

```
PreToolUse(Edit|Write em _numba/* | _jax/*):
  [N] check-paridade-pre.sh
      Snapshot do estado atual antes de alteração crítica
      (para diff em caso de regressão)

PostToolUse(Edit em _numba/* | forward.py | multi_forward.py):
  [N] run-fortran-parity.sh
      pytest tests/test_simulation_compare_fortran.py -k python_numba
      Bloqueia se >1e-12 (CRÍTICO)

PostToolUse(Edit em _jax/*):
  [N] run-jax-numba-parity.sh
      Bloqueia se JAX vs Numba >1e-10

PostToolUse(Edit em models/* | losses/*):
  [N] run-models-smoke.sh
      pytest tests/test_models.py + tests/test_losses.py
      (suite leve, ~30s)

PostToolUse(Edit em config.py):
  [N] validate-errata-runtime.sh
      Importa config.py e instancia PipelineConfig() para
      forçar __post_init__ a rodar

PreToolUse(Bash com git push):
  [N] check-tests-passed.sh
      Verifica que último pytest local passou
      (lê .pytest_cache/lastfailed)

PreToolUse(Bash com gh pr create):
  [N] generate-pr-description.sh
      Gera descrição estruturada com base no git log

Stop:
  [N] check-docs-staleness.sh
      Se houve commits que tocaram código fonte mas não docs/:
        alertar (mas não bloquear)

Stop (somente em sessões com bump de versão):
  [N] auto-generate-report.sh
      Invoca Documentador (Haiku) para gerar relatório

PostToolUse(Edit em qualquer .py):
  [N] check-ptbr-accentuation.sh
      Detecta palavras sem acento em comentários/docstrings PT-BR
      Reporta (não bloqueia)
```

### 15.3 Hook Crítico: run-fortran-parity.sh

```bash
#!/bin/bash
# .claude/hooks/run-fortran-parity.sh
# Bloqueia edição em _numba/* se paridade Fortran quebra.
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
[ -z "$FILE_PATH" ] && exit 0

# Filtro: apenas se tocou _numba ou forward.py críticos
if ! [[ "$FILE_PATH" == *_numba/* || \
        "$FILE_PATH" == *forward.py || \
        "$FILE_PATH" == *multi_forward.py ]]; then
    exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
cd "$PROJECT_DIR"
source ~/Geosteering_AI_venv/bin/activate 2>/dev/null || true

# Rodar paridade Fortran (rápido — apenas modelos canônicos críticos)
if pytest tests/test_simulation_compare_fortran.py -v -k fortran_python_numba \
       --tb=short --timeout=60 -q 2>&1 | tail -5; then
    echo "[OK] Paridade Fortran preservada (<1e-12)"
    exit 0
else
    echo "[FAIL] PARIDADE FORTRAN QUEBRADA — REVERTER OU CORRIGIR!" >&2
    echo "Arquivo modificado: $FILE_PATH" >&2
    exit 1  # BLOQUEIA
fi
```

---

## 16. MCP Servers — Catálogo Completo

### 16.1 MCP Servers Atualmente Configurados

```json
// .mcp.json (atual)
{
  "mcpServers": {
    "consensus": {
      "command": "python",
      "args": ["tools/consensus-mcp-server/server.py"],
      "env": { "S2_API_KEY": "${S2_API_KEY}" }
    },
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 60000
    }
  }
}
```

Adicionalmente, via subscrição claude.ai já disponíveis:
- `claude.ai Consensus` (HTTP, Semantic Scholar)
- `claude.ai bioRxiv`
- `claude.ai Figma`
- `claude.ai Google Drive`
- `claude.ai PDF Viewer`

### 16.2 MCP Servers Novos a Construir

#### 16.2.1 `physics-validator` — Validação EM em Tempo Real

```python
# .claude/mcp/physics_validator.py
"""
MCP Server para validação física do simulador Geosteering AI.

Tools expostas:

  validate_em_tensor(H_tensor, n_layers, rho_h, rho_v) -> ValidationReport
    • Verifica simetria de Maxwell: H_xy ≈ H_yx (tolerância 1e-10)
    • Verifica unidades: |H| em A/m, faixa física razoável
    • Retorna: {valid: bool, violations: [str], magnitude_range: [min, max]}

  run_fortran_parity(model_name, n_positions=30) -> ParityReport
    • Executa comparação Python vs Fortran para modelo canônico
    • Retorna: {model: str, max_rel_diff: float, passed: bool, threshold: 1e-12}

  check_high_resistivity(rho_max, n_layers, n_pos) -> HighResReport
    • Verifica risco de overflow em propagation.py
    • Retorna: {safe: bool, recommended_filter: str, warnings: [str]}

  get_canonical_models() -> List[CanonicalModel]
    • Lista os 7 modelos canônicos existentes + propostos
    • Para cada: paths, n_layers, rho_max, esperado_throughput

  suggest_filter(rho_max, freq_hz, precision_target) -> FilterRecommendation
    • Para condições dadas, sugere Werthmuller / Kong / Anderson
"""
```

#### 16.2.2 `numba-profiler` — Profiling JIT em Tempo Real

```python
# .claude/mcp/numba_profiler.py
"""
Tools:

  get_jit_cache_info() -> CacheReport
    • Status do cache Numba para cada @njit no projeto
    • Indica recompilações necessárias

  benchmark_scenario(scenario, n_runs=5) -> BenchReport
    • Executa benchmarks/bench_v214_numba.py para cenário
    • Retorna: {throughput_median, p25, p75, n_runs}

  analyze_prange_efficiency(function_name) -> PrangeReport
    • Mede tarefas/thread, overhead de fork/join
    • Estima eficiência de paralelismo atual

  detect_topology() -> TopologyReport
    • Wraps geosteering_ai/simulation/_workers.detect_cpu_topology
    • Retorna: phys_cores, log_cores, recommended_workers
"""
```

#### 16.2.3 `colab-bridge` — Integração com Google Colab Pro+

```python
# .claude/mcp/colab_bridge.py
"""
Tools:

  get_training_status(notebook_path) -> TrainingStatus
    • Consulta GitHub Actions / Drive para logs de treinamento
    • Retorna: {epoch, loss, val_loss, eta_minutes}

  get_gpu_metrics() -> GPUMetrics
    • Consulta utilização de GPU no Colab via API
    • Retorna: {vram_used_mb, temperature_c, throughput_examples_s}

  trigger_colab_cell(notebook_path, cell_index) -> ExecutionResult
    • Aciona execução remota via API do Colab
    • Requer: COLAB_API_TOKEN em .env

  download_artifacts(model_name) -> ArtifactPaths
    • Baixa artefatos do Drive para local após treino
"""
```

#### 16.2.4 `mlflow-tracker` — Tracking de Experimentos

```python
# .claude/mcp/mlflow_tracker.py
"""
Tools:

  list_experiments() -> List[Experiment]
    • Lista experimentos no MLflow registrado

  get_run_metrics(run_id) -> Metrics
    • Histórico completo de métricas de um run

  compare_runs(run_ids: List[str]) -> ComparisonReport
    • Tabela diff de configs + métricas finais

  promote_to_staging(run_id) -> PromotionResult
    • Move modelo do estágio dev para staging
"""
```

#### 16.2.5 `witsml-loader` — Loader de Dados Reais (futuro)

```python
# .claude/mcp/witsml_loader.py
"""
Tools (apenas após F6 ROADMAP):

  parse_witsml(path) -> WitsmlSummary
    • Estrutura de curvas, depth range, metadata

  convert_to_22col(witsml_path, output_path) -> ConversionReport
    • Wrapper de geosteering_ai.data.loaders.witsml.real_to_22col

  validate_curves(path) -> ValidationReport
    • Detecta valores ausentes, outliers, depth inconsistencies
"""
```

### 16.3 Configuração Completa Proposta

```json
// .mcp.json (refinado)
{
  "mcpServers": {
    "consensus": {
      "command": "python",
      "args": ["tools/consensus-mcp-server/server.py"],
      "env": { "S2_API_KEY": "${S2_API_KEY}" }
    },
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 60000
    },
    "physics-validator": {
      "command": "python",
      "args": [".claude/mcp/physics_validator.py"],
      "env": {
        "PYTHONPATH": ".",
        "NUMBA_CACHE_DIR": ".numba_cache"
      }
    },
    "numba-profiler": {
      "command": "python",
      "args": [".claude/mcp/numba_profiler.py"]
    },
    "colab-bridge": {
      "command": "python",
      "args": [".claude/mcp/colab_bridge.py"],
      "env": { "COLAB_API_TOKEN": "${COLAB_API_TOKEN}" }
    },
    "mlflow-tracker": {
      "command": "python",
      "args": [".claude/mcp/mlflow_tracker.py"],
      "env": { "MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI}" }
    }
  }
}
```

---

## 17. Skills (Slash Commands) — Catálogo Detalhado

### 17.1 Skills Existentes (Inventário)

```
.claude/commands/ (atual):
  geosteering-v2.md                  ← Skill PRINCIPAL de domínio
  geosteering-v5015.md               ← Legado C0-C73 (referência)
  geosteering-physics.md             ← Conceitos físicos detalhados
  geosteering-models.md              ← Catálogo de 48 arquiteturas
  geosteering-losses.md              ← Catálogo de 26 losses
  geosteering-code-v2.md             ← Padrões de código D1-D14
  geosteering-simulator-python.md    ← Sub-skill simulador Python
  geosteering-simulator-fortran.md   ← Sub-skill simulador Fortran
  geosteering-simulation-manager.md  ← Sub-skill GUI Simulation Manager
  consensus-search.md                ← Pesquisa Consensus
  arxiv-search.md                    ← Pesquisa ArXiv
```

### 17.2 Skills Novas a Criar

```
A criar em .claude/commands/:
  geosteering-orchestrator.md        ← Skill do Orquestrador
  geosteering-simulator-numba.md     ← Renomear/expandir simulator-python
  geosteering-simulator-jax.md       ← Sub-skill JAX dedicada
  geosteering-pinns.md               ← Skill PINNs (Opus)
  geosteering-data-pipeline.md       ← Skill DataPipeline + on-the-fly
  geosteering-realtime.md            ← Skill geosteering realtime
  geosteering-mlops.md               ← Skill API REST + tracking
  geosteering-frontend.md            ← Skill GUI + CLI
  geosteering-research.md            ← Expandir consensus-search + ArXiv + bioRxiv
  geosteering-documentation.md       ← Skill Documentador (Haiku)
  geosteering-code-reviewer.md       ← Skill review generalista
  geosteering-physics-reviewer.md    ← Skill review físico
  geosteering-perf-reviewer.md       ← Skill review performance
  geosteering-security-auditor.md    ← Skill auditor segurança
  geosteering-build-2d.md            ← Skill futura: Born/MEF 2D
  geosteering-build-3d.md            ← Skill futura: MEF 3D
```

### 17.3 Padrão de Conteúdo de Skill

Cada skill em `.claude/commands/<nome>.md` deve ter:

```markdown
---
name: <nome>
description: |
  <2-4 linhas: domínio + gatilhos>
tools: [...]
model: <opus|sonnet|haiku>
---

# <Título>

## Quando Invocar
INVOCAR PARA:
  • <caso 1>
NÃO INVOCAR PARA:
  • <anti-caso>

## Domínio (3-5 linhas)

## Padrões Operacionais (numerados)

## Anti-patterns Documentados (com exemplos)

## Workflow Padrão (passo-a-passo)

## Exemplos de Prompts Ideais

## Constraints Específicos

## Memória / Estado / Histórico
(se relevante: padrões de aprendizado entre sessões)
```

### 17.4 Como Skills se Chamam Mutuamente

```
EXEMPLO: Sprint v2.22 simulador

  Daniel: /geosteering-orchestrator
      "Implementar Sprint v2.22 — FLAT prange"

  Orquestrador (Opus, carrega contexto):
      → Read CLAUDE.md, MEMORY.md
      → Read análise_cenarios doc
      → TodoWrite (12 items)
      → Agent(Skill: geosteering-research,
              "FLAT prange Numba 2024+ best practices")
      → Agent(Skill: geosteering-simulator-numba,
              "Implementar _simulate_combined_prange_flat",
              isolation: worktree)
      → Agent(Skill: geosteering-physics-reviewer,
              "Validar paridade Fortran")
      → Agent(Skill: geosteering-perf-reviewer,
              "Bench Cenário F nf=4")
      → Agent(Skill: geosteering-documentation,
              "Gerar relatório v2.22")
      → commit + push + PR
```

---

## 18. Worktrees, Branching e Isolamento

### 18.1 Topologia de Branches

```
main ──────────────────────────────────────────────────────→ produção (tags)
  │
  └── develop ───────────────────────────────────────────→ integração
        │
        ├── feat/simulator-v2.22         ← Worktree dedicada
        ├── feat/simulator-v2.23
        ├── feat/jax-vmap-real-default
        ├── feat/dl-pinn-petrofisica
        ├── feat/ml-pipeline-real-data
        ├── feat/api-rest-v1
        ├── feat/cli-v1
        ├── feat/mlops-mlflow
        └── hotfix/<urgente>
```

### 18.2 Estrutura de Worktrees

```
~/Geosteering_AI/                     (main, branch principal)
~/Geosteering_AI_sim22/               (feat/simulator-v2.22, worktree)
~/Geosteering_AI_jax_vmap/            (feat/jax-vmap-real-default)
~/Geosteering_AI_pinn_archie/         (feat/dl-pinn-petrofisica)
~/Geosteering_AI_real_data/           (feat/ml-pipeline-real-data)

Comandos:
  cd ~/Geosteering_AI && git worktree add ../Geosteering_AI_sim22 feat/simulator-v2.22
  claude --worktree feat/simulator-v2.22
  # ou
  Agent(subagent_type="...", isolation="worktree", prompt="...")
```

### 18.3 Configuração do `.worktreeinclude`

```
# .worktreeinclude (a criar na raiz)
.env                          # variáveis de ambiente e tokens
.claude/settings.local.json   # hooks locais (não commitados)
Geosteering_AI_venv/          # NÃO recriar venv em cada worktree
__pycache__/                  # cache Python (não recriar)
.numba_cache/                 # cache Numba compilado
.pytest_cache/                # cache pytest
.mypy_cache/                  # cache mypy
sm_output/                    # saídas locais não-versionadas
```

### 18.4 Estratégia de Isolamento por Sprint

| Tipo de Sprint | Isolamento | Justificativa |
|:---------------|:-----------|:--------------|
| Crítico (simulador, paridade) | Worktree dedicada | Cache Numba pode contaminar; revert difícil em main |
| ML pipeline (modelos, losses) | Worktree dedicada | Treinos podem rodar em paralelo |
| Docs only | Branch normal (sem worktree) | Mudanças isoladas, sem cache |
| Hotfix | Branch direto de main | Urgência justifica menor cerimônia |
| Experimental | Worktree com prefixo `wip/` | Possibilidade de descarte total |

### 18.5 Limpeza de Worktrees

```bash
# Listar worktrees ativas
git worktree list

# Remover worktree concluída
git worktree remove ~/Geosteering_AI_sim22
git worktree prune

# Manutenção mensal
/loop monthly cleanup-worktrees
  → identifica worktrees com branches já merged
  → remove automaticamente após confirmação
```

---

## 19. Seleção de Modelos LLM por Tarefa

### 19.1 Distribuição Alvo de Tokens

```
Distribuição esperada (orçamento Max 5×):
  Sonnet 4.6:  ~70% das interações  → desenvolvimento rotineiro
  Haiku 4.5:   ~25% das interações  → automação + checagens
  Opus 4.7 1M: ~5% das interações   → arquitetura + debugging complexo

Por que essa distribuição:
  • Sonnet é o "dia-a-dia": implementa sprints, revisa, escreve testes
  • Haiku é "boilerplate": lints, docs, summaries, checklists
  • Opus é "raciocínio profundo": carrega projeto inteiro, multi-arquivo,
    trade-offs sutis — usado em sprints ARQUITETURAIS (não rotina)
```

### 19.2 Tabela de Decisão

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TAREFA                                          MODELO    JUSTIFICATIVA  │
├──────────────────────────────────────────────────────────────────────────┤
│  Sprint do simulador Numba (cross-file)          Opus      Multi-arquivo │
│  Implementar Conv1D causal                       Sonnet    Mono-arquivo  │
│  Atualizar CHANGELOG                             Haiku     Template fill │
│  Verificar acentuação PT-BR                      Haiku     Lista palavras│
│  Debug regressão paridade Fortran                Opus      Diagnóstico   │
│  Adicionar nova loss à LossFactory               Sonnet    Padrão claro  │
│  Refatoração arquitetural (e.g., backends 2D)    Opus      Transversal   │
│  Code review de PR pequeno                       Sonnet    Contextual    │
│  Code review de PR grande (>500 LOC)             Opus      Múltiplas perspectivas│
│  Bench + interpretar números                     Haiku     Tabular       │
│  Pesquisar paper sobre INN                       Sonnet    Síntese+citação│
│  Atualizar memory/MEMORY.md                      Haiku     Append simples│
│  Design de API REST                              Sonnet    Padrão claro  │
│  Implementar Hook shell                          Sonnet    Bash padrão   │
│  Análise de literatura para nova feature         Sonnet    Síntese       │
│  Decisão "FLAT prange agora ou v2.30?"           Opus      Trade-off     │
│  Geração de Pull Request description             Haiku     Template      │
│  Análise de regressão histórica multi-commit     Opus      Histórico     │
└──────────────────────────────────────────────────────────────────────────┘
```

### 19.3 Override por Skill

Cada skill define seu modelo preferido no frontmatter:

```yaml
---
name: geosteering-pinns
model: claude-opus-4-7   # PINN é complexo, sempre Opus
---
```

```yaml
---
name: geosteering-documentation
model: claude-haiku-4-5-20251001  # Doc é Haiku
---
```

Override manual quando necessário:

```python
Agent(
    subagent_type="geosteering-pinns",
    prompt="...",
    model="sonnet",  # OVERRIDE: sonnet para tarefa simples de PINN
)
```

---

## 20. Workflows Orquestrados End-to-End

### 20.1 Workflow #1: Sprint Numba Completo (FLAT prange v2.22)

Já detalhado em §7.3. Padrão de referência para todos os sprints do
simulador.

### 20.2 Workflow #2: Treinamento de Modelo DL

```
═══════════════════════════════════════════════════════════════════════════
  TREINAMENTO COMPLETO (do dataset ao modelo em produção)
═══════════════════════════════════════════════════════════════════════════

FASE 1 — PREPARAÇÃO DE DATASET (Data Pipeline Agent + Físico, 30 min)
  • Verificar integridade do dataset (shape, NaN, ranges)
  • Confirmar split por modelo geológico (P1)
  • Validar cadeia raw→noise→FV→GS→scale
  • Estatísticas EDA (visualization/eda.py)

FASE 2 — DESIGN DO EXPERIMENTO (DL Pipeline Agent, 15 min)
  • Escolher arquitetura (consultar ModelRegistry)
  • Escolher loss (consultar LossFactory)
  • Definir preset YAML em configs/
  • Configurar PINN se aplicável (Opus se cenário complexo)

FASE 3 — VALIDAÇÃO LOCAL (CPU smoke test, 5 min)
  • pytest test_models.py::test_<arch>_forward
  • Treino curto (1 epoch, 100 batches) localmente
  • Confirmar shapes, gradientes, GPU compat

FASE 4 — PUSH PARA COLAB (Orquestrador + DL Pipeline, 5 min)
  • git tag v_train_<expname>_<date>
  • git push origin --tags
  • Gerar notebook de treino se ainda não existir
  • Upload de YAML para Drive

FASE 5 — EXECUÇÃO COLAB (Manual no Colab Pro+, ~horas)
  Daniel executa notebook no Colab Pro+ T4/A100
  Notebook:
    !pip install git+...@v_train_<expname>_<date>
    config = PipelineConfig.from_yaml(yaml_path)
    pipeline = DataPipeline(config); data = pipeline.prepare(...)
    model = ModelRegistry().build(config)
    trainer = TrainingLoop(config, model, pipeline, data)
    history = trainer.run()  # logs vão para MLflow tracking server

FASE 6 — MONITORAMENTO (Haiku via Colab MCP)
  /loop 15m check-colab-training
    Consulta status via Colab MCP + GitHub Actions logs
    Reporta: epoch X/N, val_loss=Y.YY, ETA=ZZ min
    Se loss diverge (val_loss > 2× best): alerta Daniel

FASE 7 — AVALIAÇÃO (DL Pipeline Agent + Físico, 30 min)
  • Comparar com baseline anterior (model registry)
  • Métricas por componente (rho_h, rho_v) e por modelo geológico
  • Holdout plots (clean + noisy)
  • UQ calibration (reliability diagram)
  • Físico Reviewer: sanity check físico das predições

FASE 8 — PROMOÇÃO PARA STAGING (MLOps Agent, 15 min)
  • Se métricas superam baseline: promote_to_staging() via MLflow MCP
  • Senão: arquivar como experimento
  • Update memory: aprendizado sobre arquitetura/loss

FASE 9 — DOCUMENTAÇÃO (Documentador Haiku, 20 min)
  • docs/reports/treino_<expname>_<date>.md
  • Update docs/documentacao_models.md se nova arch
  • CHANGELOG entry
```

### 20.3 Workflow #3: Inversão em Dados Reais (futuro)

```
═══════════════════════════════════════════════════════════════════════════
  INVERSÃO EM POÇO REAL (campo)
═══════════════════════════════════════════════════════════════════════════

FASE 1 — RECEBIMENTO DE DADOS (Data Pipeline Agent)
  • Inputs: arquivo WITSML/LAS de poço real
  • Validações:
    - Curvas mínimas presentes (RT, gamma, depth, dip)
    - Depth range razoável (>10m, <5000m)
    - Sem NaN > 5% das amostras
  • Conversão para layout 22-col interno

FASE 2 — CALIBRAÇÃO E VALIDAÇÃO (Físico Reviewer)
  • Comparar estatísticas com sintéticos esperados
  • Identificar drifts (shifts em média/variância)
  • Aplicar correções de calibração se necessário

FASE 3 — INFERÊNCIA OFFLINE (DL Pipeline Agent)
  • Carregar modelo de produção (MLflow registry)
  • Executar InferencePipeline.predict
  • UQ via MC Dropout (30 passes) ou INN (1 pass)

FASE 4 — VALIDAÇÃO COM INVERSÃO TRADICIONAL (Físico)
  • Rodar Gauss-Newton ou Levenberg-Marquardt como ground truth
  • Calcular RMSE entre DL-prediction e GN-prediction
  • Se RMSE > 30%: alerta + análise

FASE 5 — RELATÓRIO (Documentador Haiku)
  • docs/reports/inversao_poco_<id>_<date>.md
  • Curtain plot, DTB, bandas de incerteza
  • Comparação com inversão tradicional
```

### 20.4 Workflow #4: Hotfix Crítico

```
═══════════════════════════════════════════════════════════════════════════
  HOTFIX (e.g., bug físico em produção)
═══════════════════════════════════════════════════════════════════════════

FASE 1 — TRIAGEM (Orquestrador, 5 min)
  • Reproduzir bug em ambiente local
  • Identificar arquivos afetados
  • Decidir severidade (P0 = produção quebrada; P1 = degradação)

FASE 2 — FIX (Agente apropriado, 15-60 min)
  • Branch hotfix/<descrição-curta> direto de main
  • Implementação mínima (não escopo creep)
  • Teste regressão (caso que reproduz o bug)

FASE 3 — VALIDAÇÃO (Reviewers em paralelo, 10 min)
  • Físico (se simulador) | Código (sempre) | Perf (se hot path)
  • CI roda completo

FASE 4 — DEPLOY (MLOps, 10 min)
  • Merge para main
  • Tag patch (v2.X.Y+1)
  • Deploy production direto (skip staging em hotfix)
  • Monitoring intenso por 24h

FASE 5 — POST-MORTEM (Orquestrador, 30 min, em até 48h)
  • Documentar causa raiz
  • Atualizar testes (regressão permanente)
  • Atualizar memory com aprendizado
  • Identificar gap em CI/hooks que permitiu bug passar
```

### 20.5 Workflow #5: Pesquisa → Implementação (Closing the Loop)

```
═══════════════════════════════════════════════════════════════════════════
  IDEIA CIENTÍFICA → FEATURE EM PRODUÇÃO
═══════════════════════════════════════════════════════════════════════════

FASE 1 — DESCOBERTA (Pesquisador via Consensus/ArXiv)
  Trigger: Daniel pergunta "tem novidade em UQ probabilística?"
  Pesquisador retorna síntese:
    [1] Amini et al. (2020) — Evidential Regression
    [2] Ardizzone et al. (2019) — INN
    [3] Liu et al. (2024) — Swin Transformer + noise injection

FASE 2 — AVALIAÇÃO DE FIT (Orquestrador Opus)
  • Para cada paper: aplicabilidade ao Geosteering AI?
  • Custo de implementação?
  • ROI esperado vs alternativas?
  • Decisão: implementar X agora, Y depois, Z descartado

FASE 3 — PROTOTIPAGEM (DL Pipeline Agent)
  • Implementar versão mínima
  • Validar gradiente, forward pass
  • Smoke test em dataset pequeno

FASE 4 — VALIDAÇÃO CIENTÍFICA (Físico + Pesquisador)
  • Reproduzir resultados do paper (no possível)
  • Comparar com baseline existente do projeto
  • Se diverge muito: investigar (paper tem detalhes ocultos? bug?)

FASE 5 — INTEGRAÇÃO (Sprint normal)
  Como qualquer sprint v2.X+1
```

---

## 21. Preparação para Simuladores 2D/2.5D/3D

### 21.1 Roadmap de Evolução

```
v2.0 (atual)   ── 1D TIV (Numba + JAX + Fortran)
                   │
v2.30          ── Backend abstrato + plugin discovery
                   │
v2.40          ── Aproximação de Born 2D (perturbação 1D)
                   │  [_born_2d backend]
                   │  Custo: O(n_voxels × n_freq)
                   │  Aplicação: anomalias laterais, reservatórios estreitos
                   │
v2.50          ── MEF 2D (Método dos Elementos Finitos)
                   │  [_fem_2d backend]
                   │  Dependência: scikit-fem ou meshio + custom solver
                   │  Custo: O(n_voxels^1.5)
                   │  Aplicação: estruturas dipping complexas, falhas
                   │
v2.60          ── 2.5D (MEF com simetria axial)
                   │  [_fem_2_5d backend]
                   │  Custo: O(n_voxels × n_kx) onde n_kx é número de modos
                   │  Aplicação: poços horizontais com axisymmetry
                   │
v3.0           ── MEF 3D completo
                   [_fem_3d backend]
                   Custo: O(n_voxels^1.7) com solver iterativo
                   Aplicação: digital twin completo do reservatório
```

### 21.2 Backend Abstrato (a implementar v2.30)

```python
# geosteering_ai/simulation/_backends/base.py

from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np

class SimulationBackend(Protocol):
    """Interface comum para todos os backends de simulação.

    Backends implementam esta interface e são registrados via entry_point
    'geosteering_ai.backends' em pyproject.toml.

    A camada superior (forward.py, multi_forward.py) faz dispatch
    baseado em cfg.backend, sem conhecer detalhes de implementação.
    """

    name: str  # ex: "numba_1d", "born_2d", "fem_3d"
    supports_dimension: int  # 1, 2, 25 (=2.5), 3

    @abstractmethod
    def simulate(self, cfg: SimulationConfig, ...) -> SimulationResult:
        ...

    @abstractmethod
    def supports_features(self) -> SupportedFeatures:
        """Quais features físicas o backend suporta."""
        return SupportedFeatures(
            multi_freq=True,
            multi_tr=True,
            multi_angle=True,
            tilted_antennas=True,
            compensation=True,
            high_resistivity=True,  # ρ > 1000 Ω·m
            anisotropy_tiv=True,    # ρ_h ≠ ρ_v
            dimension=1,            # apenas 1D
        )

    @abstractmethod
    def jacobian(self, cfg: SimulationConfig, ...) -> JacobianResult:
        """Calcula ∂H/∂ρ para inversão. Backends 2D/3D podem retornar None
        e indicar que requerem inversão diferente (ensemble, MCMC)."""
        ...
```

### 21.3 Estrutura de Pacote Estendida

```
geosteering_ai/simulation/
├── __init__.py              ← Dispatcher + plugin discovery
├── config.py                ← SimulationConfig com cfg.dimension, cfg.backend
├── forward.py               ← API single-model (delega para backend)
├── multi_forward.py         ← API multi-TR/angle/freq (delega)
├── _backends/               ← Camada abstrata (NOVO)
│   ├── base.py              ← Protocol SimulationBackend
│   ├── registry.py          ← BackendRegistry (entry_points)
│   └── features.py          ← SupportedFeatures dataclass
├── _numba/                  ← Backend Numba 1D (existente)
├── _jax/                    ← Backend JAX 1D (existente)
├── _fortran/                ← Wrapper para tatu.x (existente, formalizar)
├── _born_2d/                ← Backend Born 2D (NOVO em v2.40)
├── _fem_2d/                 ← Backend MEF 2D (NOVO em v2.50)
├── _fem_2_5d/               ← Backend 2.5D (NOVO em v2.60)
└── _fem_3d/                 ← Backend MEF 3D (NOVO em v3.0)
```

### 21.4 Considerações Físicas para 2D/2.5D/3D

```
2D Born (perturbação 1D):
  H_total(r) = H_background_1D(r) + ∫∫ G(r, r') · σ_anomaly(r') dx dz
  • Boa aproximação se σ_anomaly << σ_background
  • Custo modesto: 1-2 ordens > 1D
  • Inversão pode usar mesmo modelo 1D + correção

2D MEF:
  Resolve ∇·(σ ∇φ) = ∇·J em domínio 2D discretizado
  • Sem assunção de pequena perturbação
  • Custo significativo: 2-3 ordens > 1D
  • Mesh adaptativa em região do poço
  • Boundary conditions: PML (Perfectly Matched Layer)

2.5D (Fourier transform na direção poço):
  Transforma EM 3D em sequência de problemas 2D para cada modo k_x
  • Custo: O(n_kx × custo_2D)
  • Aplicação: poços horizontais com simetria axial
  • Trade-off entre 2D rápido e 3D preciso

3D MEF:
  Caso geral; extremamente custoso
  • Inversão: ensemble methods (PINN, INN, MCMC)
  • Forward: pré-cômputo + interpolação durante inversão
  • Aplicação: planejamento de poço offline (não realtime)
```

### 21.5 Compatibilidade com Pipeline DL

```
A pipeline DL atual assume:
  Input: 22-col tensor (n_pos, n_features)
  Output: (n_pos, n_targets) = (n_pos, [rho_h, rho_v])

Para 2D:
  Input: (n_pos, n_features) — mesma forma (1 traço por posição)
  Output: (n_pos, n_layers_2d) — perfil 2D de resistividade

Para 3D:
  Input: (n_pos, n_features) por traço; ensemble de traços 3D
  Output: 3D resistivity volume

Adaptações necessárias no pipeline DL:
  • Novos modelos: U-Net 2D, FNO 2D/3D, Transformer espacial
  • Loss adaptado: geometric loss (smoothness em 2D/3D)
  • PINN adaptado: residual de Maxwell em 2D/3D
  • Visualização: 2D/3D rendering (já existe módulo visualization/)
```

### 21.6 Cronograma Realista

```
v2.30 (backend abstrato):     2-3 sprints  (~2 meses)
v2.40 (Born 2D):              4-6 sprints  (~3 meses)
v2.50 (MEF 2D):               6-9 sprints  (~5 meses)
v2.60 (2.5D):                 4-6 sprints  (~3 meses)
v3.0  (MEF 3D + ML 3D):       12+ sprints  (~9-12 meses)

Total estimado: ~24 meses para evolução completa.
Pré-requisitos:
  • F1-F6 do ROADMAP atual concluídos
  • Validação industrial em 1D estabelecida
  • Datasets reais 2D/3D disponíveis (públicos ou empresa)
```

---

## 22. Roadmap de Implementação da Infraestrutura

### 22.1 Fase 1 — Fundação (Mês 1)

| Sprint | Entregável | Agente | Esforço |
|:------:|:-----------|:-------|:-------:|
| I1.1 | `.claude/commands/geosteering-orchestrator.md` | Daniel + Opus | 4h |
| I1.2 | Renomear/expandir `geosteering-simulator-numba.md` | Daniel + Opus | 3h |
| I1.3 | Criar `geosteering-simulator-jax.md` + `geosteering-pinns.md` | Daniel + Opus | 4h |
| I1.4 | Criar 7 skills restantes de domínio | Daniel + Sonnet | 8h |
| I1.5 | Criar 5 skills de qualidade | Daniel + Sonnet | 5h |
| I1.6 | Criar 3 skills de pesquisa/docs | Daniel + Haiku | 3h |
| I1.7 | `.worktreeinclude` + testes worktree | Daniel | 1h |
| I1.8 | Hooks novos (run-fortran-parity, etc.) | Sonnet | 6h |
| I1.9 | MCP `physics-validator` | Sonnet | 8h |
| I1.10 | MCP `numba-profiler` | Sonnet | 6h |
| **Total Fase 1** | | | **~48h** |

### 22.2 Fase 2 — Workflows Ativos (Mês 2)

| Sprint | Entregável | Agente | Esforço |
|:------:|:-----------|:-------|:-------:|
| I2.1 | Primeiro sprint usando arquitetura completa (v2.22 FLAT prange) | Orquestrador | 6h |
| I2.2 | MCP `colab-bridge` | Sonnet | 6h |
| I2.3 | `/loop` para monitoring de Colab | Haiku | 2h |
| I2.4 | Agent Teams experimental (3 reviewers em PR de simulador) | Daniel + Opus | 4h |
| I2.5 | Hooks: check-ptbr-accentuation, generate-pr-description | Sonnet | 4h |
| I2.6 | CLI `geosteering-cli` MVP (simulate + benchmark) | Sonnet | 12h |
| I2.7 | API REST MVP (apenas /predict offline) | Sonnet | 16h |
| I2.8 | Dockerfile.cpu + CI build em PR | Sonnet | 4h |
| **Total Fase 2** | | | **~54h** |

### 22.3 Fase 3 — Maturidade (Mês 3)

| Sprint | Entregável | Agente | Esforço |
|:------:|:-----------|:-------|:-------:|
| I3.1 | MCP `mlflow-tracker` | Sonnet | 6h |
| I3.2 | MLflow tracking server local + integração com TrainingLoop | Sonnet | 8h |
| I3.3 | Model Registry (dev/staging/production) | Sonnet | 6h |
| I3.4 | API REST completa (realtime + UQ + admin) | Sonnet | 16h |
| I3.5 | Containerização Docker.gpu | Sonnet | 4h |
| I3.6 | Dashboard Grafana + Prometheus | Sonnet | 8h |
| I3.7 | CronCreate para relatórios semanais | Haiku | 1h |
| I3.8 | CLAUDE.md hierárquicos por subpacote | Daniel + Opus | 4h |
| **Total Fase 3** | | | **~53h** |

### 22.4 Fase 4 — Industrial (Mês 4-6)

| Sprint | Entregável | Agente | Esforço |
|:------:|:-----------|:-------|:-------:|
| I4.1 | Loader WITSML (data/loaders/witsml.py) | Sonnet | 12h |
| I4.2 | Loader LAS | Sonnet | 6h |
| I4.3 | DomainAdapter para dados reais | Opus + Físico | 16h |
| I4.4 | Validação inversão tradicional vs DL em dataset Volve | Físico | 12h |
| I4.5 | Streaming OPC-UA | Sonnet | 16h |
| I4.6 | Dashboard web Streamlit | Sonnet | 12h |
| I4.7 | Edge deployment NVIDIA Jetson (TFLite) | Sonnet | 8h |
| **Total Fase 4** | | | **~82h** |

### 22.5 Total Estimado da Infraestrutura

```
Fase 1 (Mês 1):  ~48h  → fundação multi-agente
Fase 2 (Mês 2):  ~54h  → workflows ativos
Fase 3 (Mês 3):  ~53h  → maturidade MLOps
Fase 4 (Mês 4-6): ~82h → industrial / dados reais
─────────────────────
TOTAL:           ~237h ≈ 30 dias úteis full-time
                      ≈ 60 dias part-time (50%)
                      ≈ 6 meses casual (10h/semana)
```

---

## 23. Estrutura Completa de Configuração

### 23.1 Layout de Configuração-Alvo

```
~/Geosteering_AI/
│
├── CLAUDE.md                        ← Instruções globais (existente)
├── .claude/
│   ├── settings.json                ← Hooks de projeto (expandir)
│   ├── settings.local.json          ← Hooks locais (gitignored)
│   ├── commands/                    ← 16 skills (5 atuais + 11 novos)
│   │   ├── geosteering-v2.md
│   │   ├── geosteering-orchestrator.md           [NOVO]
│   │   ├── geosteering-simulator-numba.md         [renomear]
│   │   ├── geosteering-simulator-jax.md           [NOVO]
│   │   ├── geosteering-simulator-fortran.md
│   │   ├── geosteering-pinns.md                   [NOVO]
│   │   ├── geosteering-data-pipeline.md           [NOVO]
│   │   ├── geosteering-realtime.md                [NOVO]
│   │   ├── geosteering-mlops.md                   [NOVO]
│   │   ├── geosteering-frontend.md                [NOVO]
│   │   ├── geosteering-research.md                [NOVO, expandir consensus]
│   │   ├── geosteering-documentation.md           [NOVO]
│   │   ├── geosteering-code-reviewer.md           [NOVO]
│   │   ├── geosteering-physics-reviewer.md        [NOVO]
│   │   ├── geosteering-perf-reviewer.md           [NOVO]
│   │   └── geosteering-security-auditor.md        [NOVO]
│   ├── hooks/                       ← Scripts de hooks
│   │   ├── (9 atuais)
│   │   ├── run-fortran-parity.sh                  [NOVO]
│   │   ├── run-jax-numba-parity.sh                [NOVO]
│   │   ├── run-models-smoke.sh                    [NOVO]
│   │   ├── validate-errata-runtime.sh             [NOVO]
│   │   ├── check-ptbr-accentuation.sh             [NOVO]
│   │   ├── check-tests-passed.sh                  [NOVO]
│   │   ├── generate-pr-description.sh             [NOVO]
│   │   ├── check-docs-staleness.sh                [NOVO]
│   │   └── auto-generate-report.sh                [NOVO]
│   ├── mcp/                         ← MCP Servers locais (NOVO)
│   │   ├── physics_validator.py                   [NOVO]
│   │   ├── numba_profiler.py                      [NOVO]
│   │   ├── colab_bridge.py                        [NOVO]
│   │   ├── mlflow_tracker.py                      [NOVO]
│   │   └── witsml_loader.py                       [NOVO, futuro]
│   ├── templates/
│   │   ├── report_template.md                     (existente)
│   │   ├── skill_template.md                      [NOVO]
│   │   └── pr_description_template.md             [NOVO]
│   └── plans/                       ← Planos de sprint
│
├── .worktreeinclude                 [NOVO]
├── .mcp.json                        ← Expandir (5 servers novos)
│
└── geosteering_ai/
    └── simulation/
        └── CLAUDE.md                [NOVO] regras específicas simulador
```

### 23.2 Memória de Trabalho

```
~/.claude/projects/-Users-daniel-Geosteering-AI/memory/
├── MEMORY.md                              (índice; já existe; refatorar)
├── feedback_*.md                          (já existem)
├── project_simulation_manager_v2{XX}.md   (1 por versão SM)
├── project_<scope>.md                     (criados sob demanda)
├── skill_*.md                             (mapping skill → uso)
└── research/                              [NOVO]
    ├── citations_index.md
    ├── topic_pinns.md
    ├── topic_inn_uq.md
    ├── topic_2d_inversion.md
    └── topic_mlops_geophys.md
```

### 23.3 Variáveis de Ambiente (.env)

```bash
# .env (gitignored, copiado para worktrees via .worktreeinclude)
S2_API_KEY=...                  # Semantic Scholar (Consensus)
GITHUB_TOKEN=...                # gh CLI
COLAB_API_TOKEN=...             # Colab MCP
GOOGLE_DRIVE_TOKEN=...          # Drive MCP
MLFLOW_TRACKING_URI=...         # MLflow tracker
ANTHROPIC_API_KEY=...           # (não usado no Claude Code, mas ref)
```

---

## 24. Análise de Riscos, Custo e Mitigações

### 24.1 Tabela de Riscos

| Risco | Probabilidade | Impacto | Mitigação |
|:-----|:------------:|:------:|:----------|
| Subagente introduz bug físico | Média | Alto | Hook `run-fortran-parity` automático após edits críticos + Físico Reviewer |
| Contexto insuficiente em Sonnet | Média | Médio | Opus para multi-arquivo; Sonnet para mono-arquivo |
| Oversubscription de contexto em Agent Teams | Alta | Médio | Limitar a 3 teammates máximo; preferir subagentes isolados |
| PyTorch importado por acidente | Baixa | Alto | Hook PreToolUse `validate-physics.sh` (já existe) |
| `fastmath` quebra paridade Fortran | Baixa | Alto | Gate em testes de 7+3 modelos canônicos antes de habilitar |
| Custo excessivo de tokens com Opus | Média | Médio | Opus apenas para sprints arquiteturais; Sonnet em rotina |
| Worktree não limpa após subagente | Baixa | Baixo | Auto-cleanup se sem mudanças; revisão manual se com mudanças |
| MCP Server falha silenciosamente | Baixa | Médio | Logging estruturado em cada MCP; fallback graceful |
| Datasets reais inacessíveis | Alta | Alto | Plano alternativo: Volve dataset (Equinor, público) |
| 2D MEF mais lento do que viável | Média | Alto | Born 2D primeiro como aproximação; MEF 2D só se necessário |
| Adoção GUI substituída por web | Baixa | Médio | Manter ambos; GUI para análise local, web para produção |
| Drift de modelo em produção | Média | Alto | Monitoramento automático em F4; retreino periódico |
| Quebra de compatibilidade Numba/JAX | Baixa | Alto | Pinning em pyproject.toml + CI matriz multi-versão |
| Bug em hook bloqueando trabalho | Baixa | Médio | Hooks sempre opt-out via env var (e.g., `SKIP_HOOKS=1`) |
| Quebra de update do Antigravity ou da extensão Claude Code | Baixa | Médio | `.claude/` e `.mcp.json` permanecem válidos para Claude Code CLI standalone; testes mensais de fluxo na extensão |

### 24.2 Estimativa de Custo Mensal de LLM

```
Cenário ATIVO (Daniel + arquitetura completa em produção):

  Sonnet 4.6:  ~70% de uso → ~700 conversas/mês × ~5k tokens/conversa
                            = 3.5M tokens
                            ≈ $35 a $52 / mês (input+output médio)

  Haiku 4.5:   ~25% de uso → ~250 conversas/mês × ~2k tokens/conversa
                            = 500k tokens
                            ≈ $2 a $5 / mês

  Opus 4.7:    ~5% de uso  → ~50 sessões/mês × ~50k tokens/sessão
                            = 2.5M tokens
                            ≈ $40 a $75 / mês

  TOTAL: ~$80-130 / mês
  (Coberto pelo plano Max 5×: ~$100/mês fixo, sem surprise billing)
```

### 24.3 Estratégia de Otimização de Custo

```
1. Caching agressivo (Anthropic prompt cache, TTL 5min):
   • Skills + CLAUDE.md em cada conversa (cache hit ~80%)
   • Reduz custo de tokens repetidos em ~50%

2. Modelo certo para tarefa (§19):
   • Não usar Opus para tarefa Haiku
   • Não usar Sonnet para hook trivial

3. Hooks substituem LLM onde possível:
   • check-ptbr-accentuation pode ser shell+regex (não Haiku)
   • run-pytest é shell puro (não LLM)

4. MCP Servers reduzem context:
   • Em vez de Read 1.000 LOC para entender simulador,
     chamar physics-validator.get_canonical_models()
   • Reduz input tokens em ~70% para queries repetitivas

5. Agent Teams apenas onde múltiplas perspectivas agregam:
   • Sprint físico crítico: 3 teammates é justificável
   • Sprint trivial: 1 subagente sequential é suficiente
```

---

## 25. Critérios de Aceitação e Métricas

### 25.1 Critérios por Fase

```
═══════════════════════════════════════════════════════════════════════════
  FASE 1 — Fundação (Mês 1) — Critérios:
═══════════════════════════════════════════════════════════════════════════
  [ ] 16 skills criadas em .claude/commands/ + cada com frontmatter válido
  [ ] 9 hooks novos funcionando (testados com edits-trigger)
  [ ] 2 MCP servers (physics-validator, numba-profiler) respondendo a tools
  [ ] .worktreeinclude testado: claude --worktree feat/test não recria venv
  [ ] Sprint v2.22 simulador executado usando arquitetura completa
       — Tempo: <8h end-to-end
       — Quality: paridade Fortran <1e-12 mantida
       — Documentação: relatório + CHANGELOG + memory atualizados

═══════════════════════════════════════════════════════════════════════════
  FASE 2 — Workflows Ativos (Mês 2) — Critérios:
═══════════════════════════════════════════════════════════════════════════
  [ ] CLI `geosteering-cli simulate` funcional + testes passando
  [ ] CLI `geosteering-cli benchmark` reproduz números do bench manual
  [ ] API REST MVP responde POST /predict em <100ms (modelo simples)
  [ ] Docker.cpu build < 10 min, image < 2 GB
  [ ] /loop check-colab-training reporta status corretamente
  [ ] Agent Teams testado em PR de simulador (3 reviewers paralelos)

═══════════════════════════════════════════════════════════════════════════
  FASE 3 — Maturidade (Mês 3) — Critérios:
═══════════════════════════════════════════════════════════════════════════
  [ ] MLflow tracking ativo: 5+ experimentos rastreados
  [ ] Model Registry: dev/staging/production stages funcionais
  [ ] API REST completa: /predict, /predict/realtime, /uncertainty,
                          /models, /health, /reload
  [ ] Latência API: p50 < 50ms, p95 < 100ms, p99 < 150ms
  [ ] Dashboard Grafana renderiza métricas em tempo real
  [ ] Docker.gpu build inclui CUDA + JAX

═══════════════════════════════════════════════════════════════════════════
  FASE 4 — Industrial (Mês 4-6) — Critérios:
═══════════════════════════════════════════════════════════════════════════
  [ ] Loader WITSML processa arquivo Volve sem erros
  [ ] Inversão DL em poço Volve: RMSE < 30% vs Gauss-Newton
  [ ] Streaming OPC-UA simulado: latência <100ms, sem perda de amostras
  [ ] Dashboard web acessível em rede local (Streamlit)
  [ ] Edge deploy: TFLite roda em Jetson Nano com inferência < 50ms
```

### 25.2 Métricas de Sucesso de Longo Prazo

| Métrica | Alvo | Mede |
|:--------|:-----|:-----|
| Tempo médio de sprint do simulador | < 8h end-to-end | Eficiência da arquitetura |
| Bugs físicos pós-merge | < 1 / mês | Qualidade dos reviewers |
| Cobertura de testes | > 80% lines | Robustez |
| Latência de inferência produção | p99 < 150ms | Viabilidade industrial |
| Drift detection | <2σ shift sem alerta | Monitoring |
| Custo LLM mensal | < $150 | Sustentabilidade |
| Adoção do CLI vs scripts ad-hoc | > 80% das execuções | Maturidade |
| Tempo de onboarding de colaborador | < 2 dias | Documentação |

### 25.3 Métricas de Saúde Contínuas

```
Dashboard recomendado (Grafana, atualizado a cada hora):

  Geosteering AI — System Health
  ┌────────────────────────────────────────────────────────────────────┐
  │ CI Pipeline:        ✓ 47/47 passed last 24h                       │
  │ Paridade Fortran:   ✓ <1e-12 in latest run                        │
  │ Test coverage:      82.3% (+0.5% week-over-week)                  │
  │ API p99 latency:    87ms (under 150ms threshold)                  │
  │ Active models:      3 (production), 2 (staging), 8 (dev)          │
  │ MLflow runs (week): 24                                             │
  │ Drift alerts:       0 active                                       │
  │ LLM tokens (month): 6.2M (under $130 budget)                       │
  │ Worktrees ativas:   2 (feat/simulator-v2.22, feat/api-realtime)    │
  └────────────────────────────────────────────────────────────────────┘
```

---

## Conclusão e Próximos Passos Imediatos

### Síntese

Este documento estabelece a **base oficial** para a construção da
arquitetura profissional do Geosteering AI 2.0. Define:

- **17 agentes especializados** organizados em 5 camadas hierárquicas;
- **18 hooks** (9 existentes + 9 novos) para automação determinística;
- **5 MCP Servers** novos (physics-validator, numba-profiler, colab-bridge,
  mlflow-tracker, witsml-loader) somando-se aos 2 existentes;
- **16 skills** (5 existentes + 11 novas) com modelo LLM definido por
  domínio;
- **5 workflows end-to-end** (sprint Numba, treinamento DL, inversão
  real, hotfix, pesquisa→implementação);
- **Roadmap de 6 meses** para implementação plena da infraestrutura;
- **Roadmap de 24 meses** para evolução de 1D para 3D MEF;
- **Critérios mensuráveis** de aceitação por fase + métricas de sucesso.

### Princípios Permanentes

1. **Precisão física é inviolável** — paridade Fortran <1e-12 é gate
   automático em CI e em hook PostToolUse.
2. **Modularidade com plugins** — backends 1D/2D/2.5D/3D via entry_points
   do pyproject.toml; pipeline DL desacoplado do simulador.
3. **Economia de IA** — Sonnet 70%, Haiku 25%, Opus 5%; hooks
   determinísticos onde possível.
4. **Documentação como código** — D1-D14 + PT-BR acentuado +
   relatórios estruturados auto-gerados.
5. **Frontend e CLI coexistem com módulo Python** — GUI para análise
   local, CLI para automação, API REST para produção, biblioteca
   Python para integração.

### Próximos Passos Imediatos (Esta Semana)

```
1. [Daniel] Revisar este documento
2. [Daniel + Opus] Sprint v2.22 (FLAT prange) usando primeiros
   pedaços da arquitetura (skill simulator-numba expandida +
   hook run-fortran-parity)
3. [Daniel + Opus] Criar skill geosteering-orchestrator.md
4. [Daniel + Sonnet] Criar 5 skills de qualidade (reviewers + auditor)
5. [Daniel + Sonnet] Implementar hook run-fortran-parity.sh
6. [Daniel + Sonnet] MVP do MCP physics-validator
7. [Daniel] Validar workflow ponta-a-ponta com sprint real
```

### Documentos Antecessores (consultar para detalhes)

- `docs/ARCHITECTURE_v2.md` — arquitetura DL completa
- `docs/ROADMAP.md` — fases F1-F7 + v3.0 + Simulation Manager v2.5–v2.21
- `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` — todos os 8 cenários do simulador
- `docs/reports/arquitetura_multiagente_geosteering_ai_2026-05-02.md` — briefing original

### Documentos a Criar (durante implementação Fase 1-4)

```
docs/reports/
  v2.22_2026-MM-DD.md            (após sprint FLAT prange)
  treino_baseline_v3_*.md         (após primeiro treino com novo workflow)
  api_rest_design_2026-MM.md      (durante I2.7)
  dataset_volve_analise_*.md      (após F6/F4.4)

docs/reference/
  backend_abstrato_design.md      (durante v2.30)
  born_2d_aproximacao.md          (durante v2.40)
  fem_2d_design.md                (durante v2.50)

geosteering_ai/simulation/
  CLAUDE.md                        (regras específicas do simulador)

api/, cli/                         (estruturas novas)
```

---

**Documento gerado com Claude Opus 4.7 (1M contexto), 2026-05-02.**
**Status: Documento-base oficial — guia de construção da arquitetura.**
**Próxima revisão programada: após conclusão da Fase 1 da implementação
(Mês 1).**

---

# Parte II — Aprofundamentos Críticos (Revisão 2026-05-03)

> **Contexto desta Parte II**: revisão e expansão da Parte I respondendo
> 13 questões críticas levantadas após análise inicial. Esta parte
> aprofunda estratégias de otimização de janela de contexto, define
> separação Frontend/Backend em agentes, cobre suporte multiplataforma,
> GPU local, integração Colab remota, mecanismos anti-regressão,
> integração com Codex (ChatGPT Plus) e mecanismos de backup automático.
>
> **Modelo de integração validado** (revisão 2026-05-03): o Claude Code
> roda como **extensão dentro do Google Antigravity**. Os modelos Claude
> (Opus 4.7 1M, Sonnet 4.6, Haiku 4.5) são selecionados através dessa
> extensão. Antigravity hospeda a IDE; Claude Code (extensão) traz
> multi-agente, hooks, MCP, skills, worktrees e memory. A coexistência é
> de host + extensão (não de IDEs concorrentes). Detalhes em §34.

---

## 26. Estratégia de Otimização da Janela de Contexto

### 26.1 Problema Fundamental

Cada agente Claude tem janela de contexto limitada:
- Opus 4.7: até **1M tokens** (≈ 750k palavras, ≈ projeto Geosteering AI inteiro)
- Sonnet 4.6: até **200k tokens** (≈ ~150k palavras, ≈ um subpacote)
- Haiku 4.5: até **200k tokens** (mas tipicamente usado com inputs <20k)

O projeto tem ~46k LOC + ~8k LOC docs + memória + planos. Carregar tudo
em cada conversa seria possível com Opus 1M, mas **caro** e **desnecessário**.

### 26.2 Hierarquia de Otimização (5 Estratégias)

```
╔══════════════════════════════════════════════════════════════════════════╗
║  E1 — ESCOPO POR SUBAGENTE (allowed_paths)                              ║
║                                                                          ║
║  Cada subagente tem allowed_paths restrito ao seu domínio:               ║
║    geosteering-simulator-numba → simulation/_numba/, forward.py, ...     ║
║    geosteering-frontend         → simulation_manager.py, cli/             ║
║                                                                          ║
║  Efeito: agente só consegue Read em paths permitidos.                    ║
║  Ganho: -70% de input tokens em buscas Glob/Grep                         ║
║  Trade-off: cross-cutting concerns precisam ser invocados pelo           ║
║              orquestrador, não pelo subagente diretamente                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  E2 — MCP SERVERS COMO RESUMOS PRÉ-COMPUTADOS                            ║
║                                                                          ║
║  Em vez de Read 1.000 LOC de kernel.py para entender estrutura,          ║
║  chamar MCP physics-validator.get_canonical_models() que retorna         ║
║  resumo de 200 tokens.                                                   ║
║                                                                          ║
║  Substituição: Read 5k tokens → MCP call retorna 200 tokens              ║
║  Ganho: -96% de input tokens em queries de inventário                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  E3 — PROMPT CACHING (TTL 5min Anthropic)                               ║
║                                                                          ║
║  CLAUDE.md, MEMORY.md, skills carregadas — todos cachados                ║
║  Cache hit reduz custo em ~90% para tokens repetidos                     ║
║                                                                          ║
║  Estratégia: agrupar conversas em blocos < 5min para manter cache hot    ║
║  Para waits longos (>5min): aceitar cache miss; otimizar instrução       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  E4 — ROL DE TOOLS RESTRITO                                              ║
║                                                                          ║
║  Reviewer agents: tools=[Read] apenas (read-only)                        ║
║  Doc agents: tools=[Read, Edit] mas allowed_paths=docs/**               ║
║                                                                          ║
║  Efeito: subagente não pode "explorar" fora do escopo                   ║
║  Ganho: previne context bleed por curiosidade do LLM                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  E5 — SUMARIZAÇÃO EM CHECKPOINTS                                         ║
║                                                                          ║
║  Em sprints de 8h+, ao final de cada fase:                               ║
║    /compact com instrução: "preserve estado, decisões, próximos passos;  ║
║                              descarte exploration redundante"            ║
║                                                                          ║
║  Hook SessionStart(compact) reinjeta errata + commits recentes (já       ║
║  implementado em reinject-errata.sh)                                     ║
║                                                                          ║
║  Ganho: -60% tokens em sessões longas, sem perda de contexto crítico    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 26.3 Worktrees como Isolamento de Contexto

Worktrees não apenas isolam **arquivos** (branches) mas implicitamente
isolam **contexto cognitivo** dos agentes:

```
~/Geosteering_AI/                  ← branch main (orquestrador raiz)
  └─ Contexto: visão global, MEMORY.md, planos arquiteturais

~/Geosteering_AI_sim22/            ← feat/simulator-v2.22 (subagente Numba)
  └─ Contexto: APENAS simulation/_numba/* + forward.py + análise_cenarios
  └─ Agente isolado não vê: models/, losses/, training/, GUI, etc.
  └─ Janela de contexto consumida: ~30k tokens (vs 200k no orquestrador)

~/Geosteering_AI_pinn/             ← feat/dl-pinn-petrofisica (subagente PINN)
  └─ Contexto: APENAS losses/pinns.py + models/surrogate.py + papers
  └─ Janela consumida: ~50k tokens
```

**Padrão recomendado**: 1 worktree por sprint não-trivial; subagente
nasce com contexto mínimo + skill de domínio + arquivos relevantes.

### 26.4 Padrão "Retornar Resumo, Não Diff"

Quando subagente conclui, **não retornar todo o diff** ao orquestrador.
Retornar:

```yaml
# Padrão de retorno do subagente para orquestrador
sprint_id: v2.22
status: completed
files_modified:
  - simulation/forward.py (+82, -3)
  - simulation/multi_forward.py (+15, -2)
tests_added: 5
tests_passing: 73/73
parity_fortran: <1e-12 (preserved)
benchmark:
  cenario_F_pre:  46k mod/h
  cenario_F_post: 124k mod/h
  speedup: 2.7×
key_decisions:
  - FLAT prange decompose i_combo, j, i_f via integer arith
  - find_layers_tr redundant call (50ns/200μs = 0.025% — negligible)
followup_needed:
  - v2.23 fastmath gate em hmd_tiv (após validação ρ alta)
```

Orquestrador absorve **2k tokens** em vez de **30k tokens** de diff completo.
Para auditar diff, orquestrador faz `git diff` quando necessário (sob demanda).

### 26.5 Protocolo de Carregamento Incremental

```
ORQUESTRADOR (Opus 1M) inicia sprint:

  Passo 1 — Contexto base (sempre, ~5k tokens):
    CLAUDE.md, MEMORY.md, skill principal

  Passo 2 — Contexto de domínio (sob demanda, +20-50k):
    Read forward.py, kernel.py, propagation.py
    Read análise_cenarios doc

  Passo 3 — Pesquisa externa (paralelo, ~10k):
    Agent(Pesquisador) → 1 mensagem de retorno

  Passo 4 — Plano em TodoWrite + plan file (~3k)

  Passo 5 — Delegação (subagentes nascem com contexto fresh + skill):
    Cada subagente: 5k base + 20k domínio = 25k contexto
    Total paralelo: 4 subagentes × 25k = 100k tokens distribuídos
    (vs 100k acumulados no orquestrador único)

  Passo 6 — Síntese (orquestrador absorve resumos):
    +5k tokens de resumos estruturados
    Decisões finais
    Commits
```

**Resultado prático**: orquestrador fica em ~80k/1M tokens durante sprint
inteiro; subagentes em paralelo cabem em Sonnet 200k cada.

### 26.6 Anti-padrões de Contexto

| Anti-padrão | Por que é ruim | Como evitar |
|:------------|:---------------|:------------|
| Read de arquivos para "ter contexto geral" sem objetivo | Inflaciona contexto sem ROI | Read só com pergunta específica |
| Subagente que faz Bash arbitrário | Pode pollute context com output gigante | Restringir Bash a comandos enumerados |
| Reviewer com Edit (vira implementador) | Agente sai do escopo | tools=[Read] apenas |
| "Lembre-se do que fizemos no commit X" sem context | Hallucination | Read git log/show explícito |
| Loop de subagente chamando subagente | Context explosion | Limitar profundidade a 2 níveis |

---

## 27. Agentes de Frontend e Backend (Detalhamento)

### 27.1 Definição: Frontend vs Backend no Geosteering AI

```
╔══════════════════════════════════════════════════════════════════════════╗
║  FRONTEND (camada de interação humano-software)                         ║
║                                                                          ║
║  Subcamadas:                                                             ║
║    F1 GUI Desktop (PyQt6)        Análise local, sim. management         ║
║    F2 CLI (Typer)                Automação, MLOps, scripts              ║
║    F3 Notebooks (Jupyter)        Experimentação, didática               ║
║    F4 Web Dashboard (futuro)     Operação compartilhada (Streamlit)     ║
║                                                                          ║
║  Responsabilidade: NÃO contém lógica de negócio.                         ║
║    Apenas: input/output, validação superficial, orquestração de calls   ║
║    para o backend (módulos do pacote ou API REST).                      ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  BACKEND (camada de lógica + infraestrutura)                             ║
║                                                                          ║
║  Subcamadas:                                                             ║
║    B1 Core lib (geosteering_ai/) Pacote Python (PipelineConfig, ...)   ║
║    B2 Simulator backends         Numba/JAX/Fortran via plugin discovery ║
║    B3 ML pipeline                models/ losses/ training/ inference/   ║
║    B4 Data pipeline              data/ noise/ scaling                   ║
║    B5 API REST (futuro)          FastAPI + serving                      ║
║    B6 MLOps                      MLflow tracking + registry             ║
║    B7 Storage                    SQLite (runs), Drive (models, data)    ║
║                                                                          ║
║  Responsabilidade: TODA a lógica científica + computacional.            ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 27.2 Mapeamento Agentes → Camadas

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CAMADA       │ AGENTE                              │ MODELO            │
├─────────────────────────────────────────────────────────────────────────┤
│  FRONTEND                                                                │
│  F1 GUI       │ geosteering-frontend-gui            │ Sonnet 4.6        │
│  F2 CLI       │ geosteering-frontend-cli            │ Sonnet 4.6        │
│  F3 Notebooks │ geosteering-frontend-notebooks      │ Sonnet 4.6        │
│  F4 Web       │ geosteering-frontend-web (futuro)   │ Sonnet 4.6        │
│  UX/Design    │ geosteering-uiux-designer (NOVO)    │ Sonnet 4.6        │
├─────────────────────────────────────────────────────────────────────────┤
│  BACKEND                                                                 │
│  B1 Core lib  │ geosteering-core-lib (NOVO)         │ Opus 4.7          │
│  B2 Simulator │ geosteering-simulator-numba         │ Opus 4.7          │
│               │ geosteering-simulator-jax           │ Opus 4.7          │
│               │ geosteering-simulator-fortran       │ Sonnet 4.6        │
│  B3 ML        │ geosteering-dl-pipeline             │ Sonnet 4.6        │
│               │ geosteering-pinns                    │ Opus 4.7          │
│  B4 Data      │ geosteering-data-pipeline           │ Sonnet 4.6        │
│  B5 API       │ geosteering-api-rest (NOVO)         │ Sonnet 4.6        │
│  B6 MLOps     │ geosteering-mlops-tracking          │ Sonnet 4.6        │
│  B7 Storage   │ (sem agente; B5 e B6 delegam)       │ —                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 27.3 Agente UI/UX Designer (NOVO)

Cobertura: design system, paleta, atalhos, fluxos de usuário, princípios
Liquid Glass do macOS Tahoe + inspiração na Manager View do Antigravity
(IDE host). Detalhes em §33.

```yaml
---
name: geosteering-uiux-designer
description: |
  Designer UI/UX para GUI PyQt6 e Web futura. Aplica princípios
  Liquid Glass (macOS Tahoe), inspiração Antigravity Manager View,
  acessibilidade WCAG, dark mode adaptativo. Define: paleta, tipografia,
  espaçamentos, atalhos, fluxos de usuário, componentes reutilizáveis.
  Acionar para: novo painel GUI, redesenho de layout, refinamento UX.
tools: [Read, Edit, Write, Bash, WebFetch]
model: claude-sonnet-4-6
allowed_paths:
  - geosteering_ai/simulation/tests/sm_*.py
  - geosteering_ai/simulation/tests/simulation_manager.py
  - cli/
  - ui_design/
  - tests/test_gui_*.py
constraints:
  - Liquid Glass: aplicar APENAS em camada de navegação (toolbar, sidebar,
    menu); conteúdo (plots, tabelas, listas) permanece sólido
  - Dark mode: detectar via QApplication.styleHints().colorScheme()
  - Atalhos macOS-nativos (Cmd+, Cmd+W, Cmd+N)
  - WCAG AA mínimo (contraste 4.5:1 texto, 3:1 elementos)
  - Reduce Transparency: desligar efeitos quando ativado
---
```

### 27.4 Agente Core Lib (NOVO)

Cobertura: API pública (`geosteering_ai/__init__.py`), `PipelineConfig`,
contratos entre subpacotes, versionamento semântico.

```yaml
---
name: geosteering-core-lib
description: |
  Curador da API pública e contratos do pacote geosteering_ai. Garante
  estabilidade de imports, retrocompatibilidade entre versões, semantic
  versioning, plugin discovery (entry_points), expansão de API quando
  novos backends 2D/3D forem adicionados.
  Acionar para: refatoração de __init__.py, mudanças em PipelineConfig,
  adição de plugin entry_points, bump de versão major.
tools: [Read, Edit, Bash, Agent]
model: claude-opus-4-7
allowed_paths:
  - geosteering_ai/__init__.py
  - geosteering_ai/config.py
  - geosteering_ai/*/__init__.py
  - pyproject.toml
  - docs/MIGRATION_GUIDE.md
constraints:
  - SemVer estrito: breaking change → bump major
  - Deprecation warnings 1 minor antes de remover API
  - Imports lazy quando custosos (e.g., TF, JAX)
  - PipelineConfig mudanças: sempre __post_init__ para errata
---
```

### 27.5 Workflow Coordenado Frontend ↔ Backend

```
Cenário: Adicionar painel "Inverter com Modelo" na GUI

1. ORQUESTRADOR (Opus): plano + delega
2. CORE LIB (Opus): valida que InferencePipeline.predict() é estável
3. FRONTEND-GUI (Sonnet): implementa QWidget com botão + signal/slot
4. UI/UX DESIGNER (Sonnet): revisa layout, atalhos, dark mode
5. CODE REVIEWER (Sonnet): PEP8, types, cross-thread safety
6. PERF REVIEWER (Haiku): mede tempo de inicialização do painel
7. DOCUMENTADOR (Haiku): screenshot + tutorial em docs/
```

A separação clara permite que cada agente tenha contexto restrito ao
seu domínio: Frontend não precisa saber detalhes de Numba; Backend não
precisa entender Liquid Glass.

---

## 28. Arquitetura Multiplataforma (Linux + macOS + WSL2)

### 28.1 Plataforma-Alvo de Produção

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PLATAFORMA          │ USO                  │ PRIORIDADE                  │
├──────────────────────────────────────────────────────────────────────────┤
│  Linux (Ubuntu 22.04 │ Produção (servidores │ ★★★★★ (PRINCIPAL)          │
│   ou superior)       │ Docker, K8s, edge)   │                            │
├──────────────────────────────────────────────────────────────────────────┤
│  macOS 14+ (Sonoma)  │ Desenvolvimento      │ ★★★★★ (atual de Daniel)    │
│   macOS 26 (Tahoe)   │  + análise local     │                            │
├──────────────────────────────────────────────────────────────────────────┤
│  WSL2 (Windows 11)   │ Compatibilidade dev  │ ★★★★ (colaboradores Win)   │
│   Ubuntu 22.04+      │ Linux-like em Win    │                            │
├──────────────────────────────────────────────────────────────────────────┤
│  Windows nativo      │ NÃO suportado        │ ★ (zona experimental)      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 28.2 Estratégia Multiplataforma

```
╔══════════════════════════════════════════════════════════════════════════╗
║  CAMADA DE PORTABILIDADE                                                 ║
║                                                                          ║
║  • Python puro (geosteering_ai/) é 100% portável Linux/macOS/WSL2        ║
║  • Numba, TF, JAX, scikit-learn: wheels disponíveis em todas plataformas║
║  • PyQt6: wheels Linux/macOS/Win ok (em WSL2 requer X-server / WSLg)    ║
║  • Fortran: gfortran disponível em todas (apt, brew, WSL apt)            ║
║                                                                          ║
║  • Hooks shell (.claude/hooks/*.sh): bash compatível Linux/macOS/WSL2    ║
║    Cuidado: macOS usa BSD grep; Linux usa GNU grep                       ║
║    → Hooks já testados com `grep -E` (POSIX) que funciona em ambos       ║
║                                                                          ║
║  • Caminhos: usar `pathlib.Path` (não os.path com sep manual)            ║
║  • Subprocess: shell=False + lista (não string com pipes)                ║
║  • Encoding: UTF-8 explícito em open() (importante para PT-BR)           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CONTAINERIZAÇÃO COMO ABSTRAÇÃO                                          ║
║                                                                          ║
║  • Dockerfile.cpu       → image base para Linux servidor                 ║
║  • Dockerfile.gpu       → image base com CUDA                            ║
║  • docker-compose.yml   → dev local com mlflow + grafana                ║
║                                                                          ║
║  • Resultado: dev em macOS/WSL2 → push imagem → roda igual em Linux prod ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CI MATRIZ                                                                ║
║                                                                          ║
║  GitHub Actions:                                                         ║
║    matrix:                                                               ║
║      os: [ubuntu-22.04, ubuntu-24.04, macos-14]                          ║
║      python: ["3.12", "3.13"]                                            ║
║                                                                          ║
║  → 6 combinações testadas em cada PR                                     ║
║  → WSL2 não está em GitHub Actions, mas Linux ubuntu cobre               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 28.3 Pontos de Atenção Específicos

```
LINUX (produção):
  ✓ Ambiente venv padrão Python 3.13
  ✓ Apt para gfortran, build-essential
  ✓ NVIDIA Container Toolkit para Docker GPU
  ✓ systemd para API REST como serviço
  ⚠ Verificar locale UTF-8 (pt_BR.UTF-8 ou C.UTF-8)
  ⚠ Permissões de arquivos (umask)

MACOS (dev Daniel):
  ✓ Homebrew para gfortran, gh CLI
  ✓ pyenv para gerenciar Python 3.13
  ✓ Metal GPU acceleration para JAX (limitado, prefer CUDA Linux)
  ⚠ Apple Silicon (M-series) requer wheels arm64 — disponíveis em geral
  ⚠ Numba em arm64 funciona mas alguns benchmarks divergem de Linux x86

WSL2 (Windows colaborador):
  ✓ Mesma stack que Linux (Ubuntu 22.04)
  ✓ WSLg roda PyQt6 com X11 transparente
  ⚠ I/O cross-fs (Windows ↔ WSL) é lento; manter projeto em /home/<user>/
  ⚠ Acesso GPU via NVIDIA WSL2 driver + DirectML/CUDA WSL
  ⚠ Docker Desktop com integração WSL2 (não Docker engine nativo)
```

### 28.4 Hook check-platform-compat.sh (NOVO)

```bash
#!/bin/bash
# .claude/hooks/check-platform-compat.sh
# Detecta padrões não-portáveis em código novo.
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
NEW=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')
[ -z "$FILE_PATH" ] && exit 0
[[ "$FILE_PATH" == *.py ]] || exit 0

VIOLATIONS=()

if echo "$NEW" | grep -qE 'os\.path\.join\([^)]*"/'; then
    VIOLATIONS+=("Hard-coded '/' separator. Use pathlib.Path(...) / ...")
fi

if echo "$NEW" | grep -qE 'subprocess\.(run|call|check_output)\([^)]*shell=True'; then
    VIOLATIONS+=("subprocess shell=True não é portável. Use lista de args.")
fi

if echo "$NEW" | grep -qE '"/tmp/'; then
    VIOLATIONS+=("'/tmp/' hardcoded. Use tempfile.gettempdir() ou pathlib.")
fi

if [ ${#VIOLATIONS[@]} -gt 0 ]; then
    echo "[WARN] check-platform-compat: $FILE_PATH" >&2
    for v in "${VIOLATIONS[@]}"; do
        echo "  - $v" >&2
    done
fi
exit 0  # NÃO bloqueia, só alerta
```

### 28.5 README de Onboarding por Plataforma

A criar:

```
docs/SETUP_LINUX.md   → Ubuntu 22.04+, CUDA 12, gfortran
docs/SETUP_MACOS.md   → Homebrew, pyenv 3.13, optionally Metal
docs/SETUP_WSL2.md    → Windows 11, WSLg, Docker Desktop integration
```

---

## 29. GPU Local + Google Colab (Estratégia Híbrida)

### 29.1 Premissa Atualizada

```
PRODUÇÃO FINAL = GPU LOCAL como caminho principal.
Colab Pro+ permanece como backup / experimentação / pico.
```

### 29.2 Configurações de GPU Suportadas

```
┌──────────────────────────────────────────────────────────────────────────┐
│  HARDWARE              │ STACK                  │ PERFORMANCE              │
├──────────────────────────────────────────────────────────────────────────┤
│  NVIDIA RTX 4090       │ CUDA 12 + cuDNN 9      │ Treino: 1.5-2× T4       │
│  (24 GB VRAM)          │ TF 2.16+ / JAX 0.4.30+ │ Inferência: ≤5ms        │
│                        │ Driver 535+            │ Recomendação primária   │
├──────────────────────────────────────────────────────────────────────────┤
│  NVIDIA RTX 3090       │ Mesmo                  │ Treino: ~1× T4          │
│  (24 GB VRAM)          │                        │ Aceitável para dev      │
├──────────────────────────────────────────────────────────────────────────┤
│  NVIDIA A6000 / A100   │ Mesmo + NCCL multi-GPU │ Multi-modelo paralelo   │
│  (48 / 80 GB VRAM)     │                        │ Datasets grandes         │
├──────────────────────────────────────────────────────────────────────────┤
│  Apple M-series GPU    │ Metal Performance      │ JAX experimental;        │
│  (M1/M2/M3 Pro/Max)    │ Shaders + jax-metal    │ TF Metal: parcial       │
│                        │ TF Metal plugin        │ Bom para dev/inferência │
│                        │                        │ Limitado para treino    │
├──────────────────────────────────────────────────────────────────────────┤
│  AMD Radeon (RDNA 3)   │ ROCm 6.0+              │ Suporte experimental;   │
│                        │ TF ROCm / JAX ROCm     │ não recomendado prod    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 29.3 Detecção e Despacho Automático

```python
# geosteering_ai/utils/gpu_detect.py (NOVO)
"""Detecta GPU disponível e configura backends.

Ordem de preferência (configurável via PipelineConfig.gpu_preference):
  1. NVIDIA CUDA local (TF + JAX nativo)
  2. Apple Metal local (jax-metal + TF Metal plugin)
  3. AMD ROCm local (TF ROCm)
  4. Colab Pro+ remoto (via SSH/Ngrok, §30)
  5. CPU fallback (Numba + TF CPU)
"""
from typing import Literal
from dataclasses import dataclass

@dataclass
class GPUInfo:
    available: bool
    vendor: Literal["nvidia", "apple", "amd", "colab", "none"]
    name: str
    vram_gb: float
    backend_recommendation: Literal["tf_cuda", "jax_cuda", "jax_metal",
                                    "tf_metal", "tf_rocm", "cpu"]

def detect_gpu() -> GPUInfo:
    """Returns GPU info preferring local GPUs over remote."""
    # 1. Try NVIDIA via nvidia-smi
    # 2. Try Apple Metal
    # 3. Try AMD ROCm
    # 4. Colab if running in Colab env
    # 5. CPU fallback
    ...
```

### 29.4 Configuração via PipelineConfig

```python
# Adicionar em geosteering_ai/config.py

@dataclass
class PipelineConfig:
    # ... campos existentes ...

    # ── GPU + Compute Backend ─────────────────────────────────────
    gpu_preference: List[str] = field(default_factory=lambda: [
        "nvidia_local", "apple_metal", "colab_remote", "cpu",
    ])
    enable_jax_x64: bool = True
    cuda_device_id: int = 0
    enable_mixed_precision: bool = False  # fp16 (cuidado com paridade)
```

### 29.5 Modos de Execução

```
MODO 1 — Treino local em GPU NVIDIA RTX
  Comando: geosteering-cli train --preset robusto --gpu local --device 0
  Stack: TF + cuDNN nativo
  Ganho vs CPU: 30-60× para arquiteturas pesadas

MODO 2 — Inferência local em qualquer GPU
  Comando: geosteering-cli infer --gpu auto
  Latência: <5ms em RTX 4090, <20ms em M2 Pro, <50ms em CPU

MODO 3 — Treino híbrido (local + Colab para batch)
  Local treina; Colab para HPO Optuna paralelo
  Sync via MLflow tracking server compartilhado

MODO 4 — Treino remoto via Colab SSH (§30)
  Quando GPU local insuficiente; usa Colab A100 via tunnel
```

### 29.6 Setup Local Recomendado (Linux Produção)

```bash
# Ubuntu 22.04+ com NVIDIA GPU

# 1. Driver + CUDA Toolkit
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
nvidia-smi

# 2. cuDNN
sudo apt install libcudnn9-cuda-12

# 3. Python 3.13 + venv
sudo apt install python3.13 python3.13-venv
python3.13 -m venv ~/Geosteering_AI_venv
source ~/Geosteering_AI_venv/bin/activate

# 4. Pacote
cd ~/Geosteering_AI
pip install -e ".[all,sim-jax]"

# 5. Verificar
python -c "import jax; print(jax.devices())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 30. Acesso a GPU do Colab Pro+ via SSH/Ngrok

### 30.1 Análise: É Viável?

```
╔══════════════════════════════════════════════════════════════════════════╗
║  AVALIAÇÃO DE VIABILIDADE                                                ║
║                                                                          ║
║  ★ Tecnicamente factível                                                 ║
║    Bibliotecas: colab-ssh, remocolab, ssh-Colab                         ║
║    Tunnel: Ngrok (free tier limitado) ou Cloudflare Tunnel              ║
║                                                                          ║
║  ⚠ ZONA CINZA com Google ToS                                             ║
║    Termos de Uso desencorajam uso como "compute remoto puro"             ║
║    Detecção pode levar a:                                                ║
║      • Restrições temporárias                                            ║
║      • Banimento da conta                                                ║
║      • Encerramento abrupto da sessão                                    ║
║                                                                          ║
║  ⚠ Limitações operacionais                                               ║
║    • Sessões: max 12h em Pro+, 24h Pro+ Enterprise                       ║
║    • Idle timeout: ~90 min sem atividade                                 ║
║    • Latência: 100-300ms typical (ngrok adds ~50ms)                      ║
║                                                                          ║
║  ✓ Casos de uso onde COMPENSA                                            ║
║    • HPO experimental (Optuna 50 trials × 10 min)                        ║
║    • Treino de modelo grande quando não há GPU local                     ║
║    • Validação JAX em A100 (apenas verificação)                          ║
║                                                                          ║
║  ✗ Casos onde NÃO compensa (usar GPU local)                              ║
║    • Treino de produção repetitivo                                       ║
║    • Uso que exija >12h contíguas                                        ║
║    • Workloads críticos com SLA                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 30.2 Recomendação Estratégica

**Incluir na arquitetura como** *recurso opcional/experimental*, não como
caminho de produção. Desenvolvido sob `tools/colab_remote/` com avisos
explícitos sobre ToS.

### 30.3 Implementação Proposta

```
tools/colab_remote/                    [NOVO subdiretório]
├── README.md                          ← Disclaimer + instruções
├── setup_remote.ipynb                 ← Notebook que roda DENTRO do Colab
├── connect_local.py                   ← Script local que estabelece conexão
└── geosteering_remote_ssh.py          ← Wrapper que invoca treino via SSH
```

#### Notebook `setup_remote.ipynb` (roda no Colab)

```python
# Cell 1: Setup ngrok + SSH server
!apt-get install -y openssh-server
!service ssh start

# Cell 2: Authtoken e tunnel
NGROK_AUTHTOKEN = userdata.get("NGROK_AUTHTOKEN")
!pip install -q pyngrok
from pyngrok import ngrok, conf
conf.get_default().auth_token = NGROK_AUTHTOKEN
tunnel = ngrok.connect(22, "tcp")
print(f"SSH tunnel: {tunnel.public_url}")

# Cell 3: Instalar geosteering-ai
!pip install -q git+https://github.com/daniel-guitarplayer-8/geosteering-ai.git@v2.0.0

# Cell 4: Manter sessão viva (heartbeat)
import time
while True:
    !nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
    time.sleep(300)
```

#### Script local `connect_local.py`

```python
"""Conecta sessão Colab remota via SSH.

Uso:
    export NGROK_PUBLIC_URL="tcp://0.tcp.ngrok.io:12345"
    python tools/colab_remote/connect_local.py train \\
        --preset robusto --epochs 100
"""
import paramiko
import typer

app = typer.Typer()

@app.command()
def train(preset: str, epochs: int = 100):
    # SSH client → execute remote command → stream logs
    ...
```

### 30.4 Avisos Obrigatórios no README

```markdown
# tools/colab_remote/README.md

## AVISOS

1. **Termos de Uso do Google Colab**: pode violar ToS. Use por sua conta
   e risco.

2. **Não recomendado para produção**: sessões expiram em 12-24h.

3. **Alternativas para produção**:
   - GPU local (NVIDIA RTX 4090, A6000)
   - Lambda Labs (GPU sob demanda)
   - Vertex AI Workbench (estável, billing claro)

4. **Para uso casual/experimentação**: HPO, validação rápida em A100.
```

### 30.5 Integração com CLI

```bash
geosteering-cli colab connect --setup
geosteering-cli colab status
geosteering-cli train --remote colab --preset robusto
```

### 30.6 Decisão Arquitetural

```
INCLUIR na arquitetura: SIM, como módulo opcional em tools/colab_remote/
PRIORIDADE: Baixa (Fase 4+, não bloqueante)
ESCOPO: Apenas para treino DL e simulador JAX experimental
NÃO INCLUIR em: API REST de produção, MLOps tracking de produção,
                Streaming OPC-UA realtime
```

---

## 31. Sugestões Adicionais de Arquitetura/Orquestração

### 31.1 Avaliação Rigorosa de Tecnologias Adjacentes

Cada tecnologia avaliada com olhar crítico — entra ou não na arquitetura.

```
╔══════════════════════════════════════════════════════════════════════════╗
║  HYDRA (Facebook)                                                        ║
║  O que é: framework de configuração YAML + composition + override CLI.  ║
║  Pros: composição, override CLI, tracking automático em runs/            ║
║  Cons: PipelineConfig + YAML já cobre 80%; overhead cognitivo            ║
║  RECOMENDAÇÃO: Avaliar em F4 quando HPO ficar mais ativo. Não agora.    ║
║  STATUS: A AVALIAR POSTERIORMENTE                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PYDANTIC v2                                                             ║
║  O que é: validação via type hints + serialization JSON.                ║
║  Pros: validação rigorosa em API (FastAPI usa Pydantic nativo)          ║
║  RECOMENDAÇÃO: Usar em api/schemas/ APENAS. Manter dataclass em config. ║
║  STATUS: ADOTAR PARA API REST                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  RAY TUNE / RAY SERVE                                                    ║
║  O que é: distribuição HPO + serving distribuído.                       ║
║  Cons: Optuna + FastAPI + Docker são suficientes; overhead operacional. ║
║  RECOMENDAÇÃO: NÃO incluir.                                              ║
║  STATUS: REJEITADO                                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  LANGGRAPH (LangChain)                                                  ║
║  O que é: framework multi-agente com state machines.                    ║
║  Cons: Claude Code já fornece Agent + TodoWrite + worktrees.            ║
║  RECOMENDAÇÃO: NÃO incluir.                                              ║
║  STATUS: REJEITADO                                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  DSPy (Stanford NLP)                                                     ║
║  O que é: otimização programática de prompts.                           ║
║  Cons: skills evoluem manualmente; ROI baixo.                           ║
║  RECOMENDAÇÃO: NÃO incluir.                                              ║
║  STATUS: REJEITADO                                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  BENTOML                                                                  ║
║  O que é: serving de modelos ML opinionado.                             ║
║  Cons: FastAPI custom é mais flexível para domínio.                     ║
║  RECOMENDAÇÃO: NÃO incluir.                                              ║
║  STATUS: REJEITADO                                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  WEIGHTS & BIASES (W&B) — alternativa a MLflow                          ║
║  Pros: UI mais polida, sweeps integradas.                               ║
║  Cons: SaaS; custo; MLflow self-hosted é open-source.                   ║
║  RECOMENDAÇÃO: MLflow primário; W&B opcional para colaboração externa.  ║
║  STATUS: MLFLOW PRIMÁRIO; W&B SECUNDÁRIO                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  DVC (Data Version Control)                                              ║
║  O que é: versionamento de datasets via git-tracked hashes.             ║
║  Pros: reprodutibilidade total.                                          ║
║  RECOMENDAÇÃO: Adotar em Fase 4 quando datasets reais entrarem.         ║
║  STATUS: ADOTAR EM FASE 4                                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TYPER + RICH                                                            ║
║  O que é: CLI moderno + terminal output rico.                           ║
║  Pros: type hints automáticos como CLI; progress bars elegantes.        ║
║  RECOMENDAÇÃO: ADOTAR para CLI.                                          ║
║  STATUS: ADOTAR                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PRE-COMMIT (framework de hooks git)                                     ║
║  Pros: hooks compartilhados entre colaboradores via git.                ║
║  RECOMENDAÇÃO: Adotar para hooks de pre-commit Git padrão (lint,        ║
║                format, security scan). Manter hooks Claude Code         ║
║                separados (eventos específicos da IDE).                  ║
║  STATUS: ADOTAR PARA GIT PRE-COMMIT                                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  POETRY / PDM                                                            ║
║  Cons: pyproject.toml + pip install -e .[all] já funciona.              ║
║  RECOMENDAÇÃO: NÃO migrar.                                               ║
║  STATUS: REJEITADO                                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 31.2 Tecnologias ADOTADAS (síntese)

| Tecnologia | Onde | Quando |
|:-----------|:-----|:-------|
| Pydantic v2 | api/schemas/ (FastAPI) | Fase 2 (I2.7) |
| Typer + Rich | cli/ | Fase 2 (I2.6) |
| pre-commit | .pre-commit-config.yaml | Fase 1 |
| MLflow self-hosted | mlflow/ + tracking server | Fase 3 (I3.1) |
| DVC | datasets/ | Fase 4 |

### 31.3 Tecnologias REJEITADAS (com justificativa)

| Tecnologia | Razão |
|:-----------|:------|
| LangGraph | Workflows do Claude Code suficientes |
| DSPy | ROI baixo; skills com revisão humana são suficientes |
| BentoML | FastAPI custom é mais flexível |
| Ray Tune/Serve | Optuna + FastAPI cobrem o domínio |
| Hydra (no momento) | PipelineConfig já cobre 80% |
| Poetry/PDM | pip + pyproject.toml funcionam |

---

## 32. Otimização de Consumo de Tokens

### 32.1 Princípio Norteador

```
"Cada token gasto deve ter ROI mensurável. Tokens em hooks shell = 0.
Tokens em Haiku para tarefa simples = ~0.01 USD. Tokens em Opus 1M para
sprint arquitetural = ~5 USD mas evita semanas de debugging = ROI alto."
```

### 32.2 Catálogo de Otimizações T1-T12

```
╔══════════════════════════════════════════════════════════════════════════╗
║  T1 — HOOKS DETERMINÍSTICOS SUBSTITUEM LLM                              ║
║                                                                          ║
║  Hook shell em vez de LLM sempre que possível:                          ║
║    check-ptbr-accentuation: regex shell + lista de palavras (1ms)       ║
║      vs Haiku Prompt hook (200 tokens, ~0.001 USD)                      ║
║    Custo evitado em 100 edits/dia: $0.10/dia, $36/ano                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T2 — MCP SERVERS EM VEZ DE READ MASSIVO                                ║
║                                                                          ║
║  Em vez de Read kernel.py (5k tokens), invoke MCP physics-validator:    ║
║    get_canonical_models() retorna 200 tokens estruturados.              ║
║    Economia: ~96% por query repetitiva.                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T3 — PROMPT CACHING (Anthropic, TTL 5min)                              ║
║                                                                          ║
║  Estruturar conversas para manter cache hot:                            ║
║    • Carregar CLAUDE.md + skill no INÍCIO (cached)                      ║
║    • Sequência rápida de queries dentro de 5min usa cache              ║
║    • Hit rate alvo: >70%                                                ║
║    • Economia: ~90% em tokens repetidos                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T4 — MODEL SELECTION RIGOROSO                                          ║
║                                                                          ║
║  Tabela §19 estrita; nunca Opus para tarefa de Haiku.                   ║
║  Defaults em skills:                                                    ║
║    Documentador, Doc PT-BR, Perf Reviewer: Haiku                        ║
║    Maioria dos domínios: Sonnet                                         ║
║    Apenas sprints arquiteturais cross-file: Opus                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T5 — RESUMOS ESTRUTURADOS DE SUBAGENTES                                ║
║                                                                          ║
║  Subagente retorna YAML estruturado de 1-2k tokens                      ║
║  (vs orquestrador Read diff de 30k tokens). Pattern: §26.4              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T6 — ESCOPO RESTRITO POR ALLOWED_PATHS                                 ║
║                                                                          ║
║  Subagente Numba não pode Read em models/.                              ║
║  Combinado com tools=[Read, Edit] específicos por skill.                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T7 — /COMPACT NO FIM DE CADA FASE LONGA                                ║
║                                                                          ║
║  Em sprints >4h, /compact ao final de cada fase:                        ║
║    Preserve estado, decisões, próximos passos.                          ║
║    Descarte exploração intermediária.                                   ║
║    Reinjeção: errata + commits recentes (hook).                         ║
║  Economia em sessão de 8h: ~60% do contexto acumulado                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T8 — CONTEXTO LAZY-LOADED                                               ║
║                                                                          ║
║  Não Read arquivos "para ter contexto geral". Read só com pergunta:    ║
║    Bom: "Vou modificar forward.py linha 351-468; vou Read essa região"  ║
║    Ruim: "Vou Read forward.py inteiro para entender estrutura"          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T9 — REVIEWERS COM TOOLS=[READ] ESTRITO                                ║
║                                                                          ║
║  Reviewers nunca devem ter Edit ou Bash extensivo.                      ║
║  Output estruturado de 1-3k tokens.                                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T10 — BATCH DE OPERAÇÕES INDEPENDENTES                                 ║
║                                                                          ║
║  Quando 3 subagentes podem rodar em paralelo, invocar em UM bloco:      ║
║    Agent(Pesquisador), Agent(Físico), Agent(Perf)                       ║
║  Em vez de sequencial.                                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T11 — TODOWRITE COMO ÂNCORA                                             ║
║                                                                          ║
║  TodoWrite estruturado evita re-derivação do plano em cada mensagem.    ║
║  ~500 tokens estáveis por sprint inteiro.                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  T12 — RELATÓRIOS PRÉ-EXISTENTES COMO CONTEXTO                          ║
║                                                                          ║
║  Em vez de re-explicar v2.21 fix, citar:                                ║
║    "Ver docs/reports/v2.21_2026-05-02.md para causa-raiz"               ║
║  Read sob demanda só se necessário.                                     ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 32.3 Estimativa Realista de Economia

```
Sem otimizações (cenário ingênuo):
  Sprint v2.22 com 1 agente Opus carregando tudo:
    Input: 800k tokens × $15/1M = $12 (cache off)
    Output: 50k × $75/1M = $3.75
    Total: ~$16 / sprint
  Por mês (10 sprints): $160

Com otimizações (T1-T12 aplicadas):
  Sprint v2.22 multi-agente otimizado:
    Orquestrador Opus: 80k input + 5k output = $1.20 + $0.38 = $1.58
    3 subagentes Sonnet: 30k×3 + 3k×3 = 90k+9k → $0.27 + $0.135 = $0.40
    1 doc Haiku: 10k input + 2k output = $0.025 + $0.020 = $0.05
    MCP calls (deterministic): $0
    Hooks shell: $0
    Total: ~$2 / sprint (pacote Max 5× absorve)
  Por mês: ~$20 (margem segura dentro do plano fixo)

Ganho efetivo: 8× redução, dentro do orçamento Max 5×.
```

### 32.4 Métricas a Monitorar

```
.claude/hooks/track-tokens.sh (NOVO, opcional):
  Loga tokens consumidos por sessão em ~/.claude/.../telemetry.jsonl

Análise mensal (script tools/analyze_tokens.py):
  • Por modelo (Opus/Sonnet/Haiku) — ratio %
  • Por skill — qual consome mais
  • Por hora do dia — picos
  • Cache hit rate
  • Custo estimado vs Max 5× allowance

Output: docs/reports/token_usage_<month>.md
```

---

## 33. UI/UX — Liquid Glass + Antigravity Manager View

### 33.1 Princípios de Design

```
╔══════════════════════════════════════════════════════════════════════════╗
║  P1 — LIQUID GLASS APENAS NA NAVEGAÇÃO                                  ║
║                                                                          ║
║  Aplicar efeitos translúcidos com lensing em:                           ║
║    • Toolbar superior                                                    ║
║    • Sidebar lateral                                                     ║
║    • Menus dropdown                                                      ║
║    • Notification bars                                                   ║
║                                                                          ║
║  NÃO aplicar em:                                                         ║
║    • Conteúdo principal (plots, tabelas, listas, formulários)            ║
║    • Texto longo                                                         ║
║    • Imagens científicas (curtain plots, error maps)                     ║
║                                                                          ║
║  Razão: Liquid Glass otimiza navegação e hierarquia visual; aplicado    ║
║         em conteúdo, reduz legibilidade e desempenho de scanning.       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P2 — INSPIRAÇÃO DA MANAGER VIEW DO ANTIGRAVITY                         ║
║                                                                          ║
║  Antigravity tem painel "Manager" para orquestrar múltiplos agentes:    ║
║    • Lista de tarefas com status (pending/running/completed)             ║
║    • Drill-down em cada agente (logs, artefatos, métricas)              ║
║    • Vista paralela de múltiplos workspaces                              ║
║                                                                          ║
║  Aplicação no Geosteering AI GUI:                                        ║
║    • Painel "Runs Manager":                                              ║
║      - Lista runs ativos (treino, simulação, inversão)                  ║
║      - Status com ícones + cores semânticas                              ║
║      - Drill-down: logs, métricas em tempo real, artefatos               ║
║    • Painel "Pipeline Visualizer":                                       ║
║      - Estado da pipeline data → noise → FV → GS → scale → modelo       ║
║      - Tempo gasto em cada estágio                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P3 — DARK MODE ADAPTATIVO + REDUCE TRANSPARENCY                        ║
║                                                                          ║
║  Detectar:                                                               ║
║    QApplication.styleHints().colorScheme()  # Light/Dark/Unknown        ║
║                                                                          ║
║  Configurar:                                                             ║
║    • Paleta automática Light/Dark                                        ║
║    • Override manual via menu                                            ║
║    • Respeitar Reduce Transparency: desligar Liquid Glass se ativo      ║
║                                                                          ║
║  Acessibilidade WCAG AA:                                                 ║
║    • Contraste 4.5:1 mínimo em texto                                     ║
║    • Contraste 3:1 mínimo em controles                                   ║
║    • Tamanho mínimo de hit target: 44x44 px                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P4 — ATALHOS NATIVOS POR PLATAFORMA                                    ║
║                                                                          ║
║  Linux: Ctrl+, (preferências), Ctrl+W (fechar), Ctrl+N (novo)           ║
║  macOS: Cmd+, Cmd+W, Cmd+N                                              ║
║  WSL2:  Ctrl+, Ctrl+W, Ctrl+N (mesmo que Linux)                          ║
║                                                                          ║
║  Detectar plataforma e bind correto via QKeySequence.StandardKey.       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P5 — COMMAND PALETTE (Ctrl+Shift+P / Cmd+Shift+P)                      ║
║                                                                          ║
║  Inspirado em VS Code / Antigravity:                                    ║
║    • Busca fuzzy de comandos                                             ║
║    • Atalhos dinâmicos                                                   ║
║    • Acesso rápido a presets, modelos, simulações                       ║
║                                                                          ║
║  Implementação: QLineEdit + QListView com fuzzy matching.               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  P6 — MODO FOCO (single-pane)                                            ║
║                                                                          ║
║  Para análises detalhadas:                                               ║
║    Sidebar colapsável (F11 ou ícone)                                     ║
║    Toolbar minimalista                                                   ║
║    Apenas conteúdo central em destaque                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 33.2 Estrutura Visual Proposta

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ◉ ◉ ◉    GEOSTEERING AI 2.0                              ⚙ Settings  ⚪ │
├──────────────────────────────────────────────────────────────────────────┤
│ [Sidebar | Liquid Glass]   │    [Conteúdo principal | Sólido]           │
│                            │                                             │
│ ▸ Simulações                │  ┌───────────────────────────────────┐    │
│   ▸ Cenário A (1.39M mod/h)│  │  Curtain Plot                     │    │
│   ▸ Cenário E (122k mod/h) │  │  ─────────                         │    │
│   ▸ Personalizado          │  │  [renderização científica          │    │
│                            │  │   de resistividade vs depth]       │    │
│ ▸ Modelos DL                │  │                                   │    │
│   ▸ ResNet_18               │  └───────────────────────────────────┘    │
│   ▸ Mamba_S4 (causal)       │                                             │
│   ▸ INN (UQ)                │  ┌───────────────────────────────────┐    │
│                            │  │  Métricas                          │    │
│ ▸ Inversão                  │  │  R² horizontal: 0.94               │    │
│   ▸ Offline                 │  │  R² vertical:   0.91               │    │
│   ▸ Realtime                │  │  RMSE: 0.18 log Ω·m                │    │
│                            │  └───────────────────────────────────┘    │
│ ▸ MLOps                     │                                             │
│   ▸ Runs                    │  [Outras abas: Logs, Artifacts, Config]    │
│   ▸ Registry                │                                             │
├──────────────────────────────────────────────────────────────────────────┤
│ Status: ✓ GPU detectada (RTX 4090) | 12 runs ativos | MLflow conectado  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 33.3 Componentes Reutilizáveis a Criar

```
ui_design/                    [NOVO subdiretório]
├── tokens.json               ← Cores, espaçamentos, tipografia
├── components/
│   ├── glass_toolbar.py      ← QToolBar com efeito Liquid Glass
│   ├── glass_sidebar.py      ← QDockWidget translúcido
│   ├── command_palette.py    ← Cmd+Shift+P
│   ├── runs_manager.py       ← Painel Manager View-style
│   ├── pipeline_visualizer.py← Estado da pipeline em tempo real
│   ├── dark_mode_provider.py ← QPalette dinâmica
│   └── content_card.py       ← Card sólido para conteúdo (não-glass)
└── themes/
    ├── light.qss             ← QSS para tema claro
    └── dark.qss              ← QSS para tema escuro
```

### 33.4 Acessibilidade

```
Implementar:
  • Atalhos de teclado para TODAS as ações principais
  • Navegação por Tab (focus order lógico)
  • Screen reader hints (Qt Accessible.Name + Qt Accessible.Description)
  • Tamanho de fonte ajustável via Cmd+/Cmd-
  • High contrast mode (toggle via menu Accessibility)
  • Reduzir animações se Reduce Motion ativo
```

---

## 34. Antigravity + Claude Code (Modelo Validado)

### 34.1 Modelo Mental Correto (revisão 2026-05-03)

```
╔══════════════════════════════════════════════════════════════════════════╗
║  RELAÇÃO CORRETA: HOST + EXTENSÃO                                        ║
║                                                                          ║
║  Google Antigravity (HOST IDE — fork moderno do VS Code):               ║
║    • Editor View tradicional + Manager View multi-agente                 ║
║    • Sistema de extensões compatível com VS Code                         ║
║    • Terminal integrado, file explorer, git, debugging                   ║
║    • Gemini 3 Pro/Flash como motor nativo de completions inline         ║
║    • Browser extension (Chrome) para verificação web por agentes        ║
║    • Suporte nativo a MCP                                                ║
║                                                                          ║
║  Claude Code (EXTENSÃO oficial Anthropic dentro do Antigravity):        ║
║    • Instalada via marketplace de extensões do Antigravity              ║
║    • Seleção de modelo Claude pela própria extensão:                    ║
║        - Opus 4.7 (1M tokens de contexto) — sprints arquiteturais       ║
║        - Sonnet 4.6                       — desenvolvimento rotineiro    ║
║        - Haiku 4.5                        — automação e doc               ║
║    • Autenticação via plano Claude Max 5× do usuário                    ║
║    • Multi-agente nativo (Agent tool, Task tool, run_in_background)     ║
║    • Hooks configurados em .claude/settings.json                         ║
║    • MCP Servers configurados em .mcp.json                               ║
║    • Skills em .claude/commands/                                         ║
║    • Memória persistente em ~/.claude/projects/.../memory/              ║
║    • Git worktrees via `claude --worktree`                               ║
║    • TodoWrite, ScheduleWakeup, /loop                                    ║
║                                                                          ║
║  COEXISTÊNCIA NATURAL:                                                   ║
║    Daniel usa Antigravity como IDE.                                      ║
║    Dentro do Antigravity, ativa a extensão Claude Code.                  ║
║    Pela extensão, escolhe modelo Claude apropriado por tarefa.          ║
║    Para tarefas que exijam Gemini (completions inline, busca Google),   ║
║    usa o motor Gemini nativo do Antigravity em paralelo.                ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 34.2 Camadas de Acesso aos Modelos de IA

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CAMADA               │ MODELO                  │ COMO ACESSAR             │
├──────────────────────────────────────────────────────────────────────────┤
│  IDE Host                                                                │
│  Antigravity nativo    │ Gemini 3 Pro, Flash    │ Editor inline + Manager │
│                        │ (Google AI Pro)         │ View do Antigravity     │
├──────────────────────────────────────────────────────────────────────────┤
│  Extensão Anthropic                                                      │
│  Claude Code           │ Claude Opus 4.7 (1M)   │ Extensão dentro do      │
│                        │ Claude Sonnet 4.6      │ Antigravity; modelo     │
│                        │ Claude Haiku 4.5       │ selecionável via        │
│                        │ (Claude Max 5×)         │ Claude Code             │
├──────────────────────────────────────────────────────────────────────────┤
│  Extensão OpenAI (opcional)                                              │
│  Codex CLI / extensão  │ Codex via ChatGPT Plus │ CLI npm + extensão     │
│                        │                         │ VS Code (também roda    │
│                        │                         │ no Antigravity)         │
├──────────────────────────────────────────────────────────────────────────┤
│  Terminal Antigravity (via extensão Claude Code)                         │
│  Claude Code CLI       │ Mesmos modelos Claude  │ `claude` no terminal    │
│                        │                         │ integrado; útil para    │
│                        │                         │ /loop, scripts, hooks   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 34.3 Recomendação Estratégica

```
ARQUITETURA OFICIAL: ANTIGRAVITY (host) + CLAUDE CODE (extensão)

  Antigravity provê:
    • IDE moderna baseada em VS Code (editor, terminal, git, debug)
    • Manager View para orquestração visual de múltiplos agentes
    • Gemini 3 inline para completions rápidas e busca Google
    • Browser extension para verificação web autônoma
    • Sistema de extensões (marketplace) — onde Claude Code é instalado

  Claude Code (extensão dentro do Antigravity) provê:
    • Acesso completo a Opus 4.7 1M, Sonnet 4.6 e Haiku 4.5
    • Agent tool / Task tool (multi-agente, isolation=worktree)
    • Hooks PreToolUse/PostToolUse/Stop em .claude/settings.json
    • MCP Servers (.mcp.json) — physics-validator, numba-profiler, etc.
    • Skills (.claude/commands/*.md) — geosteering-* especializadas
    • Memória persistente entre sessões
    • TodoWrite, ScheduleWakeup, /loop, /compact
    • CLI `claude` no terminal integrado para automação

  Trabalho cooperativo:
    Para qualquer tarefa do Geosteering AI, Daniel:
      1. Abre o projeto no Antigravity
      2. Ativa a extensão Claude Code
      3. Seleciona o modelo Claude apropriado (§19 / §32):
         Opus 4.7 1M para sprints arquiteturais (>5 arquivos)
         Sonnet 4.6 para desenvolvimento rotineiro
         Haiku 4.5 para automação e doc
      4. Para completions inline triviais ou busca Google, usa
         o motor Gemini nativo do Antigravity em paralelo
      5. Para verificação web em PRs ou testes E2E, usa a extensão
         de browser (Chrome) do Antigravity

  Resultado: arquitetura unificada — host (Antigravity) + extensão
  (Claude Code) — sem necessidade de alternar entre IDEs.
```

### 34.4 Tabela Funcional Combinada

| Feature                          | Provido por                  |
|:---------------------------------|:-----------------------------|
| IDE host (editor, terminal, git) | Antigravity                  |
| Inline completions               | Antigravity (Gemini 3)       |
| Manager View (orquestração visual) | Antigravity                |
| Browser extension (web verify)   | Antigravity                  |
| Modelos Claude (Opus 4.7 1M etc.)| Claude Code (extensão)       |
| Agent tool / multi-agente Claude | Claude Code (extensão)       |
| Hooks (.claude/settings.json)    | Claude Code (extensão)       |
| MCP Servers (.mcp.json)          | Claude Code (extensão)       |
| Skills (.claude/commands/)       | Claude Code (extensão)       |
| Memória persistente (memory/)    | Claude Code (extensão)       |
| TodoWrite                        | Claude Code (extensão)       |
| Git worktrees (`--worktree`)     | Claude Code (extensão)       |
| /loop, ScheduleWakeup            | Claude Code (extensão)       |

### 34.5 Decisão para o Projeto

```
ESTRATÉGIA UNIFICADA APROVADA:

1. Antigravity é a IDE host oficial do projeto Geosteering AI.
   Documentar em docs/SETUP_ANTIGRAVITY.md (instalação + extensões).

2. Claude Code é a extensão oficial dentro do Antigravity.
   Toda configuração descrita neste documento (.claude/, .mcp.json,
   skills, hooks, agentes) é consumida pela extensão.

3. Seleção de modelo (Opus/Sonnet/Haiku) ocorre dentro da extensão
   Claude Code, conforme tabela §19. Opus 4.7 1M é totalmente
   acessível via extensão.

4. Antigravity Manager View pode ser usada para visualizar a execução
   de agentes em paralelo lançados pelo Claude Code (Agent tool com
   run_in_background=true). Manager View NÃO substitui o Agent tool —
   é uma camada de visualização.

5. Codex (ChatGPT Plus) entra como terceira extensão OPCIONAL para
   cross-checking em PRs críticos (§36).
```

---

## 35. Mecanismos Anti-Regressão (Bug Prevention)

### 35.1 Filosofia: "Nunca o Mesmo Bug Duas Vezes"

```
Cada bug corrigido se torna:
  1. Um teste de regressão permanente
  2. Uma entrada em memory/known_bugs.md
  3. Um padrão verificado em hook (se possível)
  4. Uma menção em CLAUDE.md (se afetar comportamento global)
```

### 35.2 Camadas de Defesa Anti-Regressão

```
╔══════════════════════════════════════════════════════════════════════════╗
║  L1 — TESTES DE REGRESSÃO PERMANENTES                                    ║
║                                                                          ║
║  Para CADA bug encontrado:                                               ║
║    • Adicionar teste em tests/test_regression_<scope>.py                 ║
║    • Test name descreve o bug: test_v213_nested_prange_overhead()       ║
║    • Comentário com referência ao commit do fix                         ║
║                                                                          ║
║  Estrutura proposta:                                                     ║
║    tests/                                                                ║
║      test_regression_simulator.py    ← bugs do simulador                 ║
║      test_regression_dl_pipeline.py  ← bugs do DL                       ║
║      test_regression_data.py         ← bugs de dados                    ║
║      test_regression_realtime.py     ← bugs de realtime                  ║
║                                                                          ║
║  Suite roda em CI + Stop hook.                                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  L2 — MEMORY DE BUGS CONHECIDOS                                          ║
║                                                                          ║
║  ~/.claude/.../memory/known_bugs.md (NOVO):                              ║
║    Lista de bugs históricos com:                                         ║
║      • ID (KB-001, KB-002, ...)                                          ║
║      • Descrição                                                         ║
║      • Causa raiz                                                        ║
║      • Versão introduzida + versão corrigida                             ║
║      • Sintoma observável                                                ║
║      • Padrão a evitar                                                   ║
║                                                                          ║
║  Agentes consultam isso ANTES de implementar mudanças críticas.         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  L3 — HOOKS PREEMPTIVOS PARA PADRÕES CONHECIDOS                          ║
║                                                                          ║
║  Para padrões anti-pattern documentados:                                ║
║    • Hook regex bloqueia em PreToolUse                                   ║
║                                                                          ║
║  Exemplos:                                                               ║
║    • parallel=True em função chamada de prange (KB-013, v2.13)          ║
║    • rng_seed=42 hardcoded em GUI (KB-019, v2.19)                       ║
║    • use_compensation=True com nTR<2 (KB-006)                            ║
║                                                                          ║
║  Hook check-anti-patterns.sh consulta lista versionada de regex.        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  L4 — CI GATES                                                            ║
║                                                                          ║
║  Toda regression suite roda em GitHub Actions:                          ║
║    ✓ tests/test_regression_*.py                                          ║
║    ✓ tests/test_simulation_compare_fortran.py (paridade <1e-12)          ║
║    ✓ tests/test_models.py (forward pass de 48 arqs)                     ║
║    ✓ Build Docker (não quebrar imagem)                                   ║
║                                                                          ║
║  PR não pode ser merged se algum gate falhar.                           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  L5 — MEMÓRIA DE PADRÕES POR AGENTE                                      ║
║                                                                          ║
║  Cada skill tem seção "Anti-patterns Documentados" (já em §4.2 etc.)    ║
║    geosteering-simulator-numba: nested prange = anti-pattern             ║
║    geosteering-pinns: gradient.numpy() em tape = anti-pattern            ║
║    geosteering-data: scaler.fit em dados ruidosos = anti-pattern         ║
║                                                                          ║
║  Agente lê skill antes de implementar; padrões ficam carregados         ║
║  no contexto.                                                            ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 35.3 Estrutura do `known_bugs.md`

```markdown
# Bugs Conhecidos do Geosteering AI

| ID | Versão Intro | Versão Fix | Sumário |
|:--:|:------------:|:----------:|:--------|
| KB-001 | v1.0 | v1.0.1 | Decoupling com sinal trocado em Hxz |
| KB-002 | v2.0 | v2.0.5 | Curriculum 3-fase off-by-one em epoch=0 |
| KB-013 | v2.13 | v2.21 | Nested prange em _fields_in_freqs_kernel_cached |
| KB-018 | v2.18 | v2.19 | rng_seed=42 hardcoded em GUI |
| KB-019 | v2.19 | v2.20 | Defaults oversubscription em CPUs HT |

## KB-013 (CRÍTICO) — Nested prange em Numba

**Versão introduzida**: v2.13 (Sprint 13.1, commit 0f92035)
**Versão corrigida**: v2.21 (Sprint 21.1, commit cba27dd)

### Causa Raiz
Adicionado `parallel=True` em `_fields_in_freqs_kernel_cached` (kernel.py)
que é chamada milhões de vezes de dentro de `_simulate_combined_prange`
(forward.py) que JÁ tem prange outer. Numba serializa o prange inner mas
paga overhead de setup do parallel scheduler em cada chamada.

### Sintoma
Cenário E: 122k mod/h → 46k mod/h (-62%) sem alteração visível no código.

### Hook de Prevenção
.claude/hooks/check-anti-patterns.sh detecta `@njit(parallel=True)` em
funções listadas no anti-patterns.txt como "chamadas de prange outer".

### Teste de Regressão
tests/test_regression_simulator.py::test_kb013_no_nested_prange_in_kernel
```

### 35.4 Hook check-anti-patterns.sh (NOVO)

```bash
#!/bin/bash
# .claude/hooks/check-anti-patterns.sh
# Bloqueia padrões catalogados em known_bugs.md como anti-patterns.
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
NEW=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')
[ -z "$FILE_PATH" ] && exit 0

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
PATTERNS_FILE="$PROJECT_DIR/.claude/anti-patterns.txt"
[ ! -f "$PATTERNS_FILE" ] && exit 0

VIOLATIONS=()
while IFS=$'\t' read -r kb_id pattern path_glob; do
    [[ "$FILE_PATH" == $path_glob ]] || continue
    if echo "$NEW" | grep -qE "$pattern"; then
        VIOLATIONS+=("$kb_id: pattern matched: $pattern")
    fi
done < "$PATTERNS_FILE"

if [ ${#VIOLATIONS[@]} -gt 0 ]; then
    echo "[BLOCK] check-anti-patterns: $FILE_PATH" >&2
    for v in "${VIOLATIONS[@]}"; do
        echo "  $v" >&2
    done
    echo "Consulte memory/known_bugs.md antes de prosseguir." >&2
    exit 1
fi
exit 0
```

`.claude/anti-patterns.txt` (formato TSV):

```
KB-013	@njit\([^)]*parallel=True	*_numba/kernel.py
KB-018	rng_seed\s*=\s*42	*simulation_manager.py
KB-019	threads_per_worker.*=.*4.*workers.*=.*4	*sm_workers.py
```

### 35.5 Workflow Anti-Regressão em Sprint

```
TODO SPRINT que toca código sensível:

1. ANTES de implementar:
   • Agent(Documentador): "List bugs em known_bugs.md relevantes a <escopo>"
   • Read tests/test_regression_<scope>.py para entender o que está coberto

2. DURANTE implementação:
   • Hook check-anti-patterns roda em cada Edit
   • Hook run-fortran-parity em cada Edit em _numba

3. ANTES de commit:
   • pytest tests/test_regression_*.py
   • Se passou: ok prosseguir
   • Se falhou: investigar; possível NOVA regressão (criar KB-XXX)

4. APÓS commit:
   • Se este sprint corrigiu um bug, criar KB-XXX em known_bugs.md
   • Adicionar teste em tests/test_regression_<scope>.py
   • Se aplicável, adicionar regex em .claude/anti-patterns.txt
```

---

## 36. Integração com OpenAI Codex (ChatGPT Plus)

### 36.1 Análise de Viabilidade

```
╔══════════════════════════════════════════════════════════════════════════╗
║  CODEX (ChatGPT Plus) — O QUE OFERECE:                                  ║
║                                                                          ║
║  • CLI Node.js: npm install -g @openai/codex                            ║
║  • Login via ChatGPT account OU API key                                 ║
║  • Plus tier: poucas sessões focadas/semana                             ║
║  • Inspeciona repo, edita arquivos, roda comandos                       ║
║  • Code review por agente separado                                      ║
║  • Web search built-in                                                  ║
║  • codex exec para scriptar workflows                                    ║
║  • Extensões VS Code/JetBrains                                          ║
║                                                                          ║
║  COMPARAÇÃO COM CLAUDE CODE:                                             ║
║                                                                          ║
║  Claude Code é mais maduro em:                                          ║
║    • Multi-agente nativo (Agent tool + isolation worktree)              ║
║    • Hooks/MCP/Skills extensiveis                                       ║
║    • Memória persistente (memory/)                                      ║
║    • Opus 4.7 1M para projetos grandes                                  ║
║                                                                          ║
║  Codex tem vantagens em:                                                ║
║    • Familiarity para devs vindos do ChatGPT                             ║
║    • codex exec scriptável (pipelines de CI)                             ║
║    • Modelo OpenAI tradicionalmente forte em refactoring                 ║
║    • Cross-checking: segunda opinião em decisões críticas                ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 36.2 Casos de Uso Onde Codex Agrega Valor

```
1. CROSS-CHECKING DE DECISÕES CRÍTICAS

   Antes de merge de PR de simulador (alto impacto):
     • Claude (Opus) revisa
     • Codex revisa independente
     • Comparar resultados → consenso ou conflito
   Custo: ~$2/PR crítico, mas reduz bugs físicos

2. CODE REVIEW DE PRs PEQUENOS

   Codex pode fazer review trivial em background via codex exec:
     codex exec --command "review PR #X" --output review.md
   Liberá ciclos do Claude para tarefas mais complexas.

3. REFACTORING TRIVIAIS COM ISO-PERFORMANCE

   Para tarefas onde Claude e Codex são equivalentes:
     • Renomear identificadores em escala
     • Extrair função
     • Aplicar lint fixes
   Usar o que estiver com slot/cota disponível.

4. EXTENSÃO CODEX (paralela à Claude Code, dentro do Antigravity)

   Antigravity (fork de VS Code) também aceita a extensão Codex.
   Pode coexistir com Claude Code: Codex roda como segunda extensão
   independente, permitindo cross-checking sem sair da IDE.

5. GERAR TESTES UNITÁRIOS BOILERPLATE

   Codex é tradicionalmente forte em geração de testes.
```

### 36.3 Casos de Uso Onde NÃO Compensa

```
✗ Sprints arquiteturais cross-file → Claude Opus 4.7 1M é superior
✗ Trabalho com hooks Claude Code → não é interoperável
✗ MCP servers configurados em .mcp.json → Codex não consome
✗ Memória persistente em .claude/projects/ → Codex não acessa
```

### 36.4 Skill `geosteering-codex-reviewer` (NOVA)

```yaml
---
name: geosteering-codex-reviewer
description: |
  Wrapper para invocar OpenAI Codex CLI como reviewer secundário em PRs
  críticos. Cross-check de decisões com modelo independente da OpenAI.
  Acionar em PRs com impacto alto (simulador, paridade Fortran, modelos
  PINN). Custo: 1 sessão Codex/PR (Plus tier ≈ poucos por semana).
tools: [Read, Bash]
model: claude-haiku-4-5-20251001
allowed_paths:
  - "**"
constraints:
  - Não usar para sprints rotineiros (preserva quota Codex Plus)
  - Output sempre estruturado para comparação com review Claude
---

# Reviewer Codex (Cross-Checker)

## Workflow

1. Receber: branch ou PR a revisar
2. Bash: cd <project> && codex exec --command \
       "Review changes in PR #X for: physical correctness, performance,
        code quality, anti-patterns. Output structured JSON."
3. Parse JSON → resumo estruturado
4. Reportar findings ao orquestrador

## Limitações

- Codex Plus: limited sessions/week → use sparsely
- Codex não tem contexto físico do projeto (sem skill geosteering-v2)
- Findings podem ser superficiais em domínio especializado
- Use SOMENTE como segunda opinião, nunca como gate principal
```

### 36.5 Decisão Arquitetural

```
INCLUIR Codex na arquitetura: SIM, como REVIEWER OPCIONAL secundário

ESCOPO:
  • PRs de alta criticidade
  • Cross-checking de decisões arquiteturais (Daniel decide quando)
  • Geração de testes unitários boilerplate (futuro)

NÃO INCLUIR em:
  • Pipeline principal de desenvolvimento
  • Hooks (incompatível)
  • MCP (incompatível)
  • Workflow sprints regulares

DOCUMENTAR em:
  • .claude/commands/geosteering-codex-reviewer.md
  • docs/SETUP_CODEX.md (instalação opcional)
```

---

## 37. Orientação para Execução das Operações

### 37.1 Como Iniciar o Trabalho com a Nova Arquitetura

```
═══════════════════════════════════════════════════════════════════════════
  PASSO 1 — VALIDAR AMBIENTE LOCAL (5-10 min)
═══════════════════════════════════════════════════════════════════════════

cd ~/Geosteering_AI
source ~/Geosteering_AI_venv/bin/activate
pytest tests/ -q --tb=short | tail -10
# Esperado: ~744+ passed em < 5 min

# Verificar paridade Fortran (gate crítico)
pytest tests/test_simulation_compare_fortran.py -v -k fortran_python_numba
# Esperado: <1e-12 nos 7 modelos canônicos

═══════════════════════════════════════════════════════════════════════════
  PASSO 2 — CRIAR ESTRUTURA DE PASTAS DA NOVA ARQUITETURA (15 min)
═══════════════════════════════════════════════════════════════════════════

mkdir -p .claude/mcp .claude/anti-patterns
mkdir -p ui_design/components ui_design/themes
mkdir -p tools/colab_remote

cat > .worktreeinclude <<'EOF'
.env
.claude/settings.local.json
Geosteering_AI_venv/
__pycache__/
.numba_cache/
.pytest_cache/
.mypy_cache/
sm_output/
.backups/
EOF

═══════════════════════════════════════════════════════════════════════════
  PASSO 3 — IMPLEMENTAR HOOK CRÍTICO ANTI-REGRESSÃO E BACKUP (45 min)
═══════════════════════════════════════════════════════════════════════════

# 1. Criar .claude/hooks/backup-pre-edit.sh (template em §38.2)
# 2. Criar .claude/hooks/run-fortran-parity.sh (template em §15.3)
# 3. Criar .claude/hooks/check-anti-patterns.sh (template em §35.4)
# 4. Adicionar em .claude/settings.json os 3 novos hooks
# 5. chmod +x .claude/hooks/*.sh

# Testar:
#   1. Editar trivialmente _numba/dipoles.py
#   2. Backup deve criar cópia em .backups/<date>/
#   3. Hook fortran-parity deve rodar pytest paridade
#   4. Se passar: ok; se falhar: hook bloqueia

═══════════════════════════════════════════════════════════════════════════
  PASSO 4 — IMPLEMENTAR SKILL DO ORQUESTRADOR (30 min)
═══════════════════════════════════════════════════════════════════════════

# Spawn Agent para criar skill geosteering-orchestrator
# (template em §4.2 da Parte I)

═══════════════════════════════════════════════════════════════════════════
  PASSO 5 — PRIMEIRO SPRINT NA ARQUITETURA NOVA (4-6h)
═══════════════════════════════════════════════════════════════════════════

# Recomendado: Sprint v2.22 (FLAT prange) usando arquitetura completa

git worktree add ../Geosteering_AI_sim22 -b feat/simulator-v2.22

cd ../Geosteering_AI_sim22
claude

# No chat:
#   "Carregar skill geosteering-orchestrator.
#    Implementar Sprint v2.22 conforme §7.3 da arquitetura aprofundada."

# Orquestrador irá:
#   - TodoWrite com 12 itens
#   - Delegar para subagentes (Pesquisador, Numba, Físico, Perf, Doc)
#   - Validar paridade Fortran
#   - Commit + PR
```

### 37.2 Roteiro Semanal Recomendado

```
SEMANA 1 — FUNDAÇÃO
  Seg: Hook backup-pre-edit + run-fortran-parity + check-anti-patterns
  Ter: Implementar skills críticas (orquestrador + numba + jax)
  Qua: Implementar MCP physics-validator (estrutural)
  Qui: Sprint v2.22 (FLAT prange) usando arquitetura
  Sex: Documentar lições aprendidas + ajustar

SEMANA 2 — EXPANSÃO
  Seg: Implementar agentes de qualidade (Físico, Perf, Code reviewers)
  Ter: Implementar Documentador + sub-skill geosteering-research
  Qua: Implementar MCP numba-profiler + colab-bridge
  Qui: CLI MVP (geosteering-cli simulate + benchmark)
  Sex: Validar workflow end-to-end + relatório semanal

SEMANA 3 — INTEGRAÇÃO
  Seg-Sex: Implementar API REST MVP + Docker.cpu

SEMANA 4 — POLIMENTO
  Seg-Sex: UI/UX refinement (Liquid Glass) + validação do fluxo
           Antigravity (host) + extensão Claude Code; ativação opcional
           da extensão Codex paralela
```

### 37.3 Checklist Diário Sugerido

```
☐ Verificar git status / pull main
☐ Verificar testes passam: pytest -q
☐ Confirmar paridade Fortran ainda <1e-12
☐ Revisar TODOs do dia anterior
☐ Identificar sprint do dia (1-3 itens)
☐ Spawn agente apropriado (não fazer manualmente o que delegável)
☐ Ao final: commit + push + update memory + relatório se aplicável
```

### 37.4 Quando Pedir Ajuda ao Orquestrador (Opus 4.7 1M)

```
INVOCAR ORQUESTRADOR para:
  ✓ Sprint que toca >5 arquivos
  ✓ Decisão de trade-off entre paridade e performance
  ✓ Bug misterioso com regressão em múltiplos commits
  ✓ Refatoração arquitetural (e.g., introduzir backend abstrato)
  ✓ Análise de literatura científica → decisão de implementação

NÃO invocar Opus para:
  ✗ Edição mono-arquivo simples (Sonnet suffice)
  ✗ Atualizar CHANGELOG (Haiku via Documentador)
  ✗ Verificar acentuação PT-BR (hook ou Haiku)
  ✗ Geração de PR description (Haiku)
```

### 37.5 Sinais de Que Algo Está Errado

```
ALERTAS:
  ⚠ Sprint demorando >2× estimativa → rever escopo
  ⚠ Paridade Fortran quebrou → REVERTER imediatamente, investigar
  ⚠ Performance regrediu sem mudança óbvia → consultar known_bugs.md
  ⚠ Subagente "vagando" sem produzir resultado → encerrar e refinar prompt
  ⚠ Tokens consumidos >> estimativa → revisar T1-T12
  ⚠ Paridade JAX vs Numba >1e-10 → alertar agente JAX
  ⚠ Testes ficando flaky → pode indicar race condition / threading bug
```

---

## 38. Mecanismo de Backup Automático Pré-Alteração

### 38.1 Princípio: Defesa em Camadas Contra Perda

```
╔══════════════════════════════════════════════════════════════════════════╗
║  CAMADA 1 — GIT (já existente)                                          ║
║    git status / git diff / git log                                       ║
║    Cobertura: tudo trackable                                             ║
║    Granularidade: por commit                                             ║
║    Restauração: git checkout / git reset                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CAMADA 2 — BACKUP AUTOMÁTICO PRE-EDIT (NOVO)                            ║
║    Hook PreToolUse copia arquivo para .backups/<timestamp>/             ║
║    Cobertura: arquivos modificados em sessão                             ║
║    Granularidade: por edit                                               ║
║    Restauração: cp .backups/<timestamp>/<arquivo> <destino>              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CAMADA 3 — STASH AUTOMÁTICO PRE-SPRINT (NOVO opcional)                  ║
║    Hook session-start cria git stash com timestamp                       ║
║    Cobertura: estado do working tree                                     ║
║    Granularidade: por sessão                                             ║
║    Restauração: git stash apply stash@{N}                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CAMADA 4 — TIME MACHINE / SNAPSHOT FS (sistema operacional)             ║
║    macOS: Time Machine                                                   ║
║    Linux: btrfs/ZFS snapshots                                            ║
║    WSL2: backup da VM                                                    ║
║    Cobertura: tudo no filesystem                                         ║
║    Granularidade: ~hora                                                  ║
║    Restauração: via UI do SO                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 38.2 Hook backup-pre-edit.sh (NOVO — IMPLEMENTAR PRIMEIRO)

```bash
#!/bin/bash
# .claude/hooks/backup-pre-edit.sh
# Cria cópia de segurança do arquivo ANTES de qualquer Edit/Write.

set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0
[ ! -f "$FILE_PATH" ] && exit 0  # arquivo não existe (criação) — sem backup

# Apenas backup de arquivos críticos (extensões versionáveis)
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/daniel/Geosteering_AI}"
case "$FILE_PATH" in
    *.py | *.f08 | *.yaml | *.yml | *.json | *.md | *.sh | *.toml)
        ;;
    *)
        exit 0
        ;;
esac

# Diretório de backups por dia
BACKUP_DIR="$PROJECT_DIR/.backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Path relativo para preservar estrutura
REL_PATH="${FILE_PATH#$PROJECT_DIR/}"

# Copiar com timestamp em sufixo (preserva versões múltiplas no dia)
TIMESTAMP=$(date +%H%M%S)
DEST_FILE="$BACKUP_DIR/${REL_PATH}.${TIMESTAMP}.bak"
mkdir -p "$(dirname "$DEST_FILE")"
cp -p "$FILE_PATH" "$DEST_FILE"

echo "[backup] $FILE_PATH → ${DEST_FILE#$PROJECT_DIR/}" >&2
exit 0  # NÃO bloqueia
```

### 38.3 Configuração em .claude/settings.json

Adicionar no array `PreToolUse`:

```json
{
  "matcher": "Edit|Write",
  "hooks": [
    {
      "type": "command",
      "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/backup-pre-edit.sh",
      "timeout": 5
    }
  ]
}
```

### 38.4 .gitignore Update

```
# Adicionar ao .gitignore
.backups/
```

Backups são LOCAIS, não versionados. Git já cobre histórico de longo
prazo; backups são para recovery rápido na sessão atual.

### 38.5 Limpeza Automática

```bash
# tools/cleanup_backups.sh (rodar via cron ou /loop weekly)
#!/bin/bash
# Mantém backups dos últimos 30 dias; remove mais antigos.
PROJECT_DIR="${1:-$HOME/Geosteering_AI}"
find "$PROJECT_DIR/.backups/" -maxdepth 1 -type d -mtime +30 -exec rm -rf {} +
echo "Cleanup completo: backups >30 dias removidos."
```

Adicionar a `crontab`:

```
0 3 * * 0  /Users/daniel/Geosteering_AI/tools/cleanup_backups.sh
```

### 38.6 Comando de Restore Rápido

```bash
# tools/restore_from_backup.sh
#!/bin/bash
PROJECT_DIR="${1:-$HOME/Geosteering_AI}"
BACKUP_ROOT="$PROJECT_DIR/.backups"

if [ -z "${2:-}" ]; then
    echo "Uso: restore_from_backup.sh <PROJECT_DIR> <FILE_PATH_RELATIVO>"
    echo ""
    echo "Backups disponíveis:"
    find "$BACKUP_ROOT" -name "*.bak" -type f | sort -r | head -20
    exit 0
fi

REL=$2
echo "Versions disponíveis de $REL:"
find "$BACKUP_ROOT" -name "$(basename $REL).*.bak" -type f | sort -r
echo ""
read -p "Cole o path completo do backup a restaurar: " BACKUP_PATH
cp "$BACKUP_PATH" "$PROJECT_DIR/$REL"
echo "Restaurado: $REL"
```

### 38.7 Workflow de Backup + Recovery

```
SCENARIO: Edit destrutivo em forward.py durante sprint v2.22

T0: Hook backup-pre-edit copia forward.py → .backups/2026-05-03/...forward.py.143822.bak
T1: Edit aplica mudança
T2: Hook run-fortran-parity detecta paridade quebrada
    → bloqueia (exit 1)
T3: Daniel decide reverter:
    Opção A: git checkout forward.py    (volta ao último commit)
    Opção B: cp .backups/2026-05-03/.../forward.py.143822.bak forward.py
             (volta ao estado pré-Edit, mesmo se commit posterior contém mudanças)
T4: Continue sprint refinando estratégia
```

### 38.8 Espaço em Disco

```
Estimativa: 100 edits/dia × 50KB médio = 5MB/dia
Por mês: ~150MB
Por ano (sem cleanup): ~1.8GB
Com cleanup automático (>30 dias): mantém ~150MB

Aceitável para projeto grande.
```

### 38.9 Integração com Agentes

```
Agentes invocados via Agent tool herdam hooks. Portanto:
  • Subagente Numba edita kernel.py → backup automático
  • Subagente Frontend edita simulation_manager.py → backup automático
  • Subagente Documentador edita CHANGELOG.md → backup automático

NÃO é necessário cada agente saber sobre backup; é invisível.

Em isolation=worktree:
  • Hook ativa na worktree (settings.json compartilhado via $CLAUDE_PROJECT_DIR)
  • Backup vai para .backups/ DA WORKTREE (não da main)
  • Quando worktree é mergeada, backup permanece local na worktree
    (limpo quando worktree é removida)
```

### 38.10 Decisão Arquitetural

```
INCLUIR na arquitetura: SIM, OBRIGATÓRIO

PRIORIDADE: Alta (implementar na Fase 1, PRIMEIRA tarefa)

ESCOPO:
  • Hook PreToolUse em todos os arquivos de código (.py, .f08, .yaml, etc.)
  • Cleanup automático após 30 dias
  • Tool de restore documentada

GARANTIAS:
  • Toda alteração tem cópia de segurança disponível
  • Recovery em <30 segundos
  • Espaço em disco controlado (cleanup automático)
  • Agentes herdam comportamento sem configuração adicional
```

---

## Síntese da Parte II — Decisões Aprovadas

```
╔══════════════════════════════════════════════════════════════════════════╗
║  DECISÕES ARQUITETURAIS APROVADAS APÓS APROFUNDAMENTO                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  §26  Otimização de janela de contexto: 5 estratégias E1-E5 + worktrees ║
║       como isolamento cognitivo + protocolo de carregamento incremental  ║
║                                                                          ║
║  §27  Frontend/Backend: 5 agentes Frontend (gui, cli, notebooks, web,   ║
║       uiux) + 7 agentes Backend (core-lib, simuladores, ml, dl, pinns,  ║
║       data, api, mlops); core-lib e uiux são novos                      ║
║                                                                          ║
║  §28  Multiplataforma: Linux primário (produção); macOS dev; WSL2 compat║
║       Hook check-platform-compat (NOVO); CI matriz 3 OS × 2 Python     ║
║                                                                          ║
║  §29  GPU local primário (NVIDIA RTX 4090, A100, A6000, M-series, AMD); ║
║       Colab Pro+ secundário; gpu_detect.py + 4 modos de execução       ║
║                                                                          ║
║  §30  Colab SSH/Ngrok: INCLUÍDO em tools/colab_remote/ como opcional;   ║
║       avisos ToS; não para produção                                      ║
║                                                                          ║
║  §31  Tecnologias adicionais ADOTADAS: Pydantic v2 (api), Typer+Rich    ║
║       (cli), pre-commit (git), MLflow (mlops), DVC (fase 4)             ║
║       REJEITADAS: LangGraph, DSPy, BentoML, Ray, Hydra (no momento),    ║
║       Poetry/PDM                                                         ║
║                                                                          ║
║  §32  12 estratégias T1-T12 de otimização de tokens; meta $20/mês       ║
║                                                                          ║
║  §33  Liquid Glass apenas em navegação; Manager View do Antigravity     ║
║       como inspiração; ui_design/ subdiretório novo                     ║
║                                                                          ║
║  §34  Modelo VALIDADO: Antigravity é IDE host; Claude Code é extensão   ║
║       oficial dentro do Antigravity; modelos Claude (Opus 4.7 1M,       ║
║       Sonnet 4.6, Haiku 4.5) selecionados via extensão Claude Code      ║
║                                                                          ║
║  §35  Anti-regressão: 5 camadas (testes, memory KB, hooks, CI, skills) ║
║       known_bugs.md + .claude/anti-patterns.txt (NOVOS)                  ║
║                                                                          ║
║  §36  Codex (ChatGPT Plus): INCLUÍDO como reviewer secundário OPCIONAL  ║
║       em PRs críticos; cross-check com Claude; skill geosteering-codex- ║
║       reviewer (novo)                                                    ║
║                                                                          ║
║  §37  Roteiro detalhado de execução semanal; checklist diário; sinais   ║
║       de alerta; quando invocar Opus                                     ║
║                                                                          ║
║  §38  Backup automático pre-edit: hook backup-pre-edit.sh (NOVO);       ║
║       OBRIGATÓRIO; .backups/ no .gitignore; cleanup automático 30d      ║
║       PRIMEIRA TAREFA da implementação                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Próximos Passos Imediatos (Atualizados)

```
1. [Daniel] Ler Parte II e validar decisões
2. [Daniel + Sonnet] Implementar §38 (backup-pre-edit.sh) — PRIMEIRO
3. [Daniel + Sonnet] Implementar §35.4 (check-anti-patterns.sh) e
   §15 (run-fortran-parity.sh) — ANTES de qualquer edit em _numba/*
4. [Daniel + Sonnet] Criar known_bugs.md inicial com KB-013, 018, 019
5. [Daniel + Sonnet] Criar .claude/anti-patterns.txt com 3 entradas
6. [Daniel + Opus] Sprint v2.22 (FLAT prange) usando arquitetura
   completa, com backup-pre-edit ativo desde o início
7. [Daniel] Validar workflow Antigravity (host) + extensão Claude Code,
   incluindo seleção de modelo (Opus/Sonnet/Haiku) via extensão
```

---

**Parte II adicionada em 2026-05-03 com Claude Opus 4.7 (1M contexto).**
**Status: DOCUMENTO BASE OFICIAL — Parte I (visão geral) + Parte II
(aprofundamentos críticos) formam a referência completa para construção
da arquitetura do Geosteering AI.**

**Total do documento: ~5.500 linhas combinadas (Parte I: 3.552; Parte II: ~2.000).**

**Próxima revisão programada: após conclusão da Fase 1 da implementação.
Revisões pontuais conforme necessário em sprints arquiteturais.**


---

═══════════════════════════════════════════════════════════════════════════
PARTE III — REFINAMENTOS ARQUITETURAIS PROFUNDOS (2026-05-03 · OPUS 4.7)
═══════════════════════════════════════════════════════════════════════════

> **Escopo desta Parte III**: 15 refinamentos arquiteturais críticos
> emergidos durante revisão técnica do documento base. Atualiza decisões
> da Parte I e II quando há contradição (sempre marcado com `🔄 REVISA §N`).
> Adiciona 4 novos componentes (1 agente, 1 workflow, 2 ferramentas).

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  ÍNDICE — PARTE III                                                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  §39  Skill Caveman — Token Reduction Opcional                            ║
║  §40  Multi-agentes Paralelos — Otimização e Segurança                    ║
║  §41  Quality Mesh — Toolkit Anti-bug em 7 Camadas                        ║
║  §42  Distribuição em Duas Trilhas: pip-lib vs Studio (modelo Geosoft)    ║
║  §43  GitHub Auto-Sync Engineering                                        ║
║  §44  Workflow W13 — JAX Sprint                                           ║
║  §45  Roteiro Detalhado de Implementação Semana-a-Semana                  ║
║  §46  Agente Upgrade Scout — 18º agente da arquitetura                    ║
║  §47  Setup Local Automatizado — Cross-platform                           ║
║  §48  GPU Detection Policy — macOS=Colab / Linux+NVIDIA=Local             ║
║  §49  Backends de Visualização — matplotlib + Plotly + PyQtGraph + PyVista║
║  §50  Arquitetura de Software Escolhida — Hexagonal + DDD                 ║
║  §51  Agente Geosteering Decision Real-time — Expansão crítica            ║
║  §52  Requisitos de Produção e GUI Profissional                           ║
║  §53  Docling para Conversão PDF→Markdown                                 ║
║  §54  Síntese da Parte III + Cronograma de Adoção                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## §39 — Skill Caveman: Token Reduction Opcional

### §39.1 — O Que É Caveman (Recapitulação Técnica)

| Atributo | Valor |
|:---------|:------|
| **Tipo** | Output Style Skill (não-oficial Anthropic) |
| **Autor** | Julius Brussee — `github.com/JuliusBrussee/caveman` |
| **Mecanismo** | Reescreve respostas em prosa ultra-comprimida; remove cortesias, recapitulações, redundâncias |
| **Níveis** | `lite` (suave) · `full` (default, ~65%) · `ultra` (~85%) · `wenyan-{lite,full,ultra}` (chinês clássico) |
| **Cobre** | Respostas, commits (Conventional ≤50 chars), PR comments (1 linha), mensagens internas |
| **Instalação** | `claude plugin marketplace add JuliusBrussee/caveman` ou install.sh universal |
| **Distribuição** | Plugin + skill markdown auto-ativado por sessão |

### §39.2 — O Que CAVEMAN Afeta (e o que NÃO afeta)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ESCOPO DE COMPRESSÃO DA SKILL CAVEMAN                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  AFETA (compressão real):                                                │
│    ✅ Output tokens visíveis ao usuário (texto da resposta)             │
│    ✅ Commit messages geradas pelo agente                                │
│    ✅ PR descriptions e PR review comments                               │
│    ✅ Logging gerado pelo próprio agente                                 │
│                                                                          │
│  NÃO AFETA:                                                              │
│    ❌ Thinking/reasoning tokens internos (Claude pensa em prosa normal) │
│    ❌ Conteúdo de arquivos editados via Edit/Write (código intacto)    │
│    ❌ Comentários e docstrings de código (mantidos verbose)             │
│    ❌ Tool calls e seus parâmetros JSON                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

**Implicação**: economia média **65% sobre output**, mas como em desenvolvimento
de software o output é ~30-40% do total de tokens (60-70% são thinking + tool
results), a economia real por sessão é **~20-30%** — ainda relevante, mas
menor que o título sugere.

### §39.3 — Análise de Viabilidade no Geosteering AI

**Vantagens** para o projeto:

| Cenário | Benefício de Caveman | Magnitude |
|:--------|:--------------------|:---------|
| Agente long-running (≥30 turns) — sprint v2.22 multi-arquivo | -25% tokens output → -15% custo total da sessão | **Alto** |
| Hooks/agentes em background (review, parity validator) | -30% (resposta padronizada curta) | **Alto** |
| Geração de commit messages no `repo-housekeeper` | -50% (commits já curtos) | Médio |
| Diálogo interativo com Daniel (perguntas, decisões) | Confunde leitor humano; remove contexto pedagógico | **NEGATIVO** |
| Documentação gerada (relatórios MD em docs/reports/) | Quebra padrão "70% estruturado + 30% prosa rica" | **NEGATIVO** |
| Code review comments gerados ao Daniel | Remove justificativa do bug — perde valor educativo | Negativo |

**Conclusão**: Caveman é **valioso em contextos NÃO-INTERATIVOS** com leitor
único (o próprio orchestrator) e **prejudicial em contextos interativos** ou
de leitura humana.

### §39.4 — Decisão Arquitetural: Caveman Condicional

**ADOTAR** caveman como recurso **OPCIONAL CONDICIONAL** ativado apenas em
agentes específicos via `.claude/output-styles/caveman-conditional.md`:

```yaml
# .claude/output-styles/caveman-conditional.md (NOVO)

ativar_em:
  - agentes_long_running:
      - sprint-orchestrator    # Opus, sprints > 5 arquivos
      - upgrade-scout          # Sonnet, scan semanal
      - repo-housekeeper       # Haiku, manutenção
      - bench-runner           # Sonnet, benchmarks paralelos

  - hooks_automatizados:
      - run-pytest.sh         # log de teste (200 linhas → 80)
      - validate-physics.sh   # validações (50 linhas → 20)
      - lint-v2-standards.sh  # output do linter

  - workflows_em_paralelo:
      - W01-W13 (todos)       # quando ≥3 agentes simultâneos

desativar_em:
  - dialogo_interativo:
      - sessoes_diretas_com_daniel
      - explanations_pedagogicas
      - code_review_comments_destinados_humanos

  - artefatos_de_documentacao:
      - docs/reports/*.md     # padrão 70/30 quebraria
      - docs/reference/*.md
      - CHANGELOG.md
      - README.md
      - docstrings em geosteering_ai/*.py

nivel_padrao: full       # 65% redução
nivel_em_emergencia: ultra  # quando context > 80% (ativação automática)
```

### §39.5 — Implementação Prática

**Passo 1**: Instalar plugin
```bash
claude plugin marketplace add JuliusBrussee/caveman
```

**Passo 2**: Criar `.claude/output-styles/caveman-conditional.md` (config acima)

**Passo 3**: Ativar via slash command quando rodar agentes em batch:
```bash
/output-style caveman-full     # antes de spawn de agentes paralelos
# (executar workflows W01-W13)
/output-style normal           # após batch — volta para diálogo verbose
```

**Passo 4**: Auto-ativação por agente. Em cada definição de agente em
`.claude/agents/<nome>.md`, adicionar frontmatter:
```yaml
---
name: upgrade-scout
output_style: caveman-full   # ← ativa apenas para este agente
---
```

### §39.6 — Métricas de Sucesso e Reversão

| Métrica | Sem Caveman | Com Caveman Condicional | Δ |
|:--------|:-----------:|:----------------------:|:--:|
| Tokens output/sprint pesado | ~2.0M | ~1.4M | **-30%** |
| Tokens total/sprint pesado | ~5.0M | ~4.4M | -12% |
| Custo mensal (10 sprints) | ~$160 | ~$140 | -$20 |
| Legibilidade interna (agentes) | OK | OK | = |
| Legibilidade externa (Daniel) | Boa | Boa (caveman desativado) | = |

**Critério de reversão**: se em 4 semanas a economia <15% **ou** se Daniel
relatar perda de contexto em ≥2 sessões, **DESATIVAR** caveman globalmente
e voltar para output normal. Decisão revisitada na revisão pós-Fase 1.

### §39.7 — Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|:------|:------------:|:----------|
| Caveman ativa em contexto errado e Daniel recebe resposta mutilada | Baixa | Toggle visível na status bar; comando `/output-style normal` é instantâneo |
| Plugin caveman tem bug que quebra agentes | Baixa | Pin de versão (`@1.x.x`); revisão pelo upgrade-scout antes de bumps |
| Skill 3rd-party com risco de supply chain | Baixa | Auditar fonte (`github.com/JuliusBrussee/caveman`) antes de instalar; instalar em sandbox |
| Compatibilidade quebra com Antigravity | Média | Testar em Antigravity + Claude Code extension antes de rollout pleno |

### §39.8 — Quando NÃO Usar Caveman (Lista Negra)

```
┌────────────────────────────────────────────────────────────────────────┐
│  CONTEXTOS ONDE CAVEMAN É PROIBIDO                                    │
├────────────────────────────────────────────────────────────────────────┤
│  ❌ Geração de relatórios MD (docs/reports/)                          │
│  ❌ Documentação técnica (docs/reference/)                            │
│  ❌ CHANGELOG.md, ROADMAP.md, README.md, CLAUDE.md                    │
│  ❌ Comentários de código (docstrings, inline)                        │
│  ❌ Sessões interativas onde Daniel pede explicação                   │
│  ❌ Code review comments destinados a humanos (Daniel/Codex review)   │
│  ❌ Mensagens de erro mostradas ao usuário final do Studio            │
│  ❌ Onboarding ou material educativo                                  │
│  ❌ Comunicação com clientes externos (suporte SLA)                   │
└────────────────────────────────────────────────────────────────────────┘
```


---

## §40 — Multi-agentes Paralelos: Otimização e Segurança

🔄 **REVISA** §11 (Topology) e §28 (Multiplataforma).

### §40.1 — Estado Atual da Paralelização

**Como o projeto trabalha com agentes em paralelo HOJE**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PARALELISMO ATUAL (Parte I — §11)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Camada 1: Orchestrator (Opus 4.7 1M)                                  │
│    └─ dispara agentes em paralelo via tool calls múltiplos             │
│                                                                         │
│  Camada 2: 8 Domain Specialists (Sonnet 4.6)                          │
│    └─ podem rodar simultaneamente em sprints independentes             │
│                                                                         │
│  Limites observados:                                                   │
│    • Até 4 agentes Sonnet em paralelo (rate limit Max 5×)             │
│    • Sem isolamento de filesystem (todos editam mesma working dir)    │
│    • Sem lock cross-agent — race condition em CHANGELOG.md            │
│    • Falha em 1 agente NÃO aborta os outros (drift de estado)         │
└─────────────────────────────────────────────────────────────────────────┘
```

### §40.2 — Riscos Identificados (Cenários de Bug)

| ID | Risco | Causa Raiz | Severidade |
|:---|:------|:-----------|:----------:|
| **R-MA-1** | 2 agentes editam mesma função simultaneamente → último write vence, edits do anterior perdidos | Sem lock + sem worktree | **Crítico** |
| **R-MA-2** | Agente A falha mas B/C continuam → estado inconsistente em git | Sem rollback transacional | Alto |
| **R-MA-3** | Race em append a CHANGELOG.md/MEMORY.md → entries duplicadas ou intercaladas | Append concorrente sem semáforo | Alto |
| **R-MA-4** | Agente paralelo introduz regressão que **outro** agente NÃO percebe | Sem review automático pós-paralelismo | **Crítico** |
| **R-MA-5** | Agentes spawnam sub-agentes recursivos → fork bomb de tokens | Sem limite de profundidade | Médio |
| **R-MA-6** | Paralelismo em hook compile-check.sh quebra Numba JIT cache | Cache file corrompido por write concorrente | Médio |
| **R-MA-7** | Custo de tokens explode com 8 agentes simultâneos sem warning | Sem dashboard de uso | Médio |

### §40.3 — Arquitetura de Paralelismo Refinada

**Novo esquema**: 5 mecanismos de defesa empilhados.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PARALELISMO SEGURO (proposta Parte III §40)                          │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  CAMADA 0 — DECISÃO DO ORCHESTRATOR                             │ │
│  │  • Conflict matrix: agentes que NÃO podem rodar juntos          │ │
│  │  • Resource budget: max 4 Sonnet + 2 Haiku simultâneos          │ │
│  │  • Profundidade máxima: 2 (orchestrator → specialist → tool)    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                            ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  CAMADA 1 — ISOLAMENTO POR WORKTREE                             │ │
│  │  • Cada agente que toca código → git worktree próprio          │ │
│  │  • .git/worktrees/agent-<nome>-<timestamp>/                    │ │
│  │  • Merge controlado pelo orchestrator no final                 │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                            ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  CAMADA 2 — LOCK FILES POR ARQUIVO                              │ │
│  │  • .claude/locks/<file_hash>.lock (advisory)                    │ │
│  │  • Hook pre-edit verifica lock + cria; pós-edit remove          │ │
│  │  • Timeout 5min → lock expirado é removido + alerta             │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                            ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  CAMADA 3 — REVIEW AUTOMÁTICO PÓS-PARALELISMO                   │ │
│  │  • Após merge dos worktrees, dispara code-review-haiku-agent   │ │
│  │  • Verifica: imports duplicados, regras D1-D14, anti-patterns  │ │
│  │  • Falha em qualquer check → BLOQUEIA merge final              │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                            ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  CAMADA 4 — ROLLBACK WAVE                                       │ │
│  │  • Se ≥1 agente falha → reverter TODOS os worktrees             │ │
│  │  • Tag git pre-paralelismo + git reset --hard <tag>             │ │
│  │  • Backup-pre-edit garante restauração mesmo sem git            │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### §40.4 — Conflict Matrix: Quem NÃO Pode Rodar Junto

```python
# .claude/parallelism_rules.py (NOVO)
"""
Define quais agentes podem ou não rodar simultaneamente.
"""

CONFLITO = {
    # Agentes que tocam o MESMO subsistema NÃO podem paralelizar
    "numba-jit-engineer":   ["numba-validator", "fortran-parity-validator"],
    "jax-engineer":         ["jax-validator"],
    "dl-training-engineer": ["dl-architecture-engineer"],

    # Agentes de docs sequencializam para evitar conflito em CHANGELOG/ROADMAP
    "docs-writer":          ["repo-housekeeper", "upgrade-scout"],

    # Hooks críticos (compile, parity) NÃO paralelizam com edits no _numba/
    "numba-jit-engineer":   ["validate-physics-hook"],
}

PODEM_PARALELIZAR = {
    # Domínios independentes — paralelismo seguro
    ("numba-jit-engineer", "dl-training-engineer"),
    ("jax-engineer",        "literature-research-agent"),
    ("docs-writer",         "physics-validator-mcp"),
    ("upgrade-scout",       "any-other-agent"),  # só leitura
}

LIMITES = {
    "max_sonnet_concurrent": 4,
    "max_haiku_concurrent":  2,
    "max_opus_concurrent":   1,    # Opus não paraleliza com Opus
    "max_depth":             2,    # orchestrator → specialist → tool (sem sub-sub)
    "max_total_simultaneous": 5,
}
```

### §40.5 — Lock Files Cross-agent

```bash
# .claude/hooks/agent-acquire-lock.sh (NOVO PRE-EDIT HOOK)
#!/bin/bash
# Acquires advisory lock on file before agent can edit it.
# Failure mode: if lock exists and is fresh (<5min), abort edit.

FILE_PATH="$1"
AGENT_ID="${CLAUDE_AGENT_ID:-orchestrator}"
LOCK_DIR="$CLAUDE_PROJECT_DIR/.claude/locks"
mkdir -p "$LOCK_DIR"

# Hash do path para nome único de lock
HASH=$(echo -n "$FILE_PATH" | shasum -a 256 | cut -c1-16)
LOCK_FILE="$LOCK_DIR/${HASH}.lock"

if [ -f "$LOCK_FILE" ]; then
    LOCK_AGE=$(( $(date +%s) - $(stat -f %m "$LOCK_FILE" 2>/dev/null || stat -c %Y "$LOCK_FILE") ))
    if [ "$LOCK_AGE" -lt 300 ]; then  # <5min
        OWNER=$(cat "$LOCK_FILE" 2>/dev/null)
        echo "❌ Lock conflict: $FILE_PATH locked by $OWNER (age ${LOCK_AGE}s)" >&2
        exit 2  # bloqueia edit
    else
        echo "⚠️  Stale lock removed (age ${LOCK_AGE}s)" >&2
        rm -f "$LOCK_FILE"
    fi
fi

# Adquire lock
echo "$AGENT_ID@$(date +%FT%T)" > "$LOCK_FILE"
echo "✅ Lock acquired: $FILE_PATH by $AGENT_ID" >&2
exit 0
```

```bash
# .claude/hooks/agent-release-lock.sh (NOVO POST-EDIT HOOK)
#!/bin/bash
FILE_PATH="$1"
HASH=$(echo -n "$FILE_PATH" | shasum -a 256 | cut -c1-16)
LOCK_FILE="$CLAUDE_PROJECT_DIR/.claude/locks/${HASH}.lock"
rm -f "$LOCK_FILE"
exit 0
```

### §40.6 — Worktrees Obrigatórios em Paralelismo Crítico

**Política**: agentes que tocam ≥2 arquivos em `geosteering_ai/` durante
sprint paralelo **DEVEM** usar git worktree.

```bash
# Orchestrator faz isso automaticamente antes de spawn:
git worktree add .git/worktrees/agent-numba-jit-$(date +%s) feat/sprint-v2.22-numba
git worktree add .git/worktrees/agent-jax-$(date +%s)        feat/sprint-v2.22-jax

# Após merge:
cd .git/worktrees/agent-numba-jit-* && git push origin
cd .. && git merge --no-ff feat/sprint-v2.22-numba
git worktree remove .git/worktrees/agent-numba-jit-*
```

### §40.7 — Telemetria e Dashboard de Custo

```python
# .claude/telemetry/parallelism_dashboard.py (NOVO)
"""
Dashboard CLI que mostra status atual de agentes em paralelo.
Lê .claude/active_agents.json (mantido pelo orchestrator).
"""
# Output exemplo:
#
# ╔══════════════════════════════════════════════════════════════════╗
# ║  GEOSTEERING AI — PARALLELISM DASHBOARD                         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Agente                    Modelo    Início      Tokens   Status║
# ║  ─────────────────────────────────────────────────────────────── ║
# ║  numba-jit-engineer        Sonnet    14:32:15    487k     RUN  ║
# ║  fortran-parity-validator  Haiku     14:33:02     32k     RUN  ║
# ║  docs-writer               Sonnet    14:34:00    156k     RUN  ║
# ║  ─────────────────────────────────────────────────────────────── ║
# ║  TOTAL                                            675k     3 active║
# ║  COST (parcial)           $1.84                                 ║
# ║  RATE                     ~22k tok/min                          ║
# ╚══════════════════════════════════════════════════════════════════╝
```

### §40.8 — Critérios de Aceite (Hardening Multi-Agente)

| Critério | Meta | Como Medir |
|:---------|:----:|:-----------|
| Lock acquisition success rate | ≥99% | telemetry/locks_attempted vs locks_acquired |
| Worktree cleanup automático após merge | 100% | post-merge hook valida ausência de stale worktrees |
| Code review automático após paralelismo | 100% sprints com ≥2 agentes | revisão de logs do orchestrator |
| Rollback wave executado em falha de agente | <30s | tempo entre falha e tag git restaurada |
| Custo mensal sem oversubscription | ≤$200 | dashboard de telemetria mensal |
| Zero merge conflicts em paralelismo seguro | 0 | git merge sem hand-resolution |

### §40.9 — Mudanças no Documento e Arquivos a Criar

**Modificar**:
- `.claude/agents/sprint-orchestrator.md` (Opus) — adicionar lógica de conflict matrix
- `.claude/settings.json` — adicionar hooks `agent-acquire-lock.sh` (PreToolUse Edit/Write) e `agent-release-lock.sh` (PostToolUse)

**Criar**:
- `.claude/parallelism_rules.py` — conflict matrix Python
- `.claude/hooks/agent-acquire-lock.sh` — pre-edit lock
- `.claude/hooks/agent-release-lock.sh` — post-edit unlock
- `.claude/telemetry/parallelism_dashboard.py` — dashboard CLI
- `.claude/active_agents.json` — registry vivo (atualizado pelo orchestrator)

---

## §41 — Quality Mesh: Toolkit Anti-bug em 7 Camadas

🔄 **REVISA** §35 (Anti-regressão) e §38 (Backup pre-edit).

### §41.1 — Filosofia: Defesa em Profundidade

> *"Nenhuma camada captura todos os bugs. Mas 7 camadas capturam tantos
> bugs que os restantes são raros o suficiente para serem econômicos
> de corrigir reativamente."*

```
┌────────────────────────────────────────────────────────────────────────┐
│  QUALITY MESH — 7 CAMADAS DEFENSIVAS                                 │
├────────────────────────────────────────────────────────────────────────┤
│  L1 — BACKUP PRE-EDIT (preventiva, recuperação)                       │
│       hook: backup-pre-edit.sh → .backups/<date>/                     │
│       garantia: rollback de qualquer edit em <1s                      │
│                                                                        │
│  L2 — STATIC ANALYSIS (preventiva, pré-execução)                      │
│       check-anti-patterns.sh + ruff + mypy + black                    │
│       garantia: estilo + tipo + 6 anti-padrões em <2s                 │
│                                                                        │
│  L3 — PHYSICS VALIDATION (preventiva, domínio)                        │
│       validate-physics.sh + MCP physics-validator                     │
│       garantia: errata respeitada, paridade Fortran <1e-12            │
│                                                                        │
│  L4 — REGRESSION TESTS (reativa, ciclo)                               │
│       run-pytest.sh (Stop hook, 120s timeout)                        │
│       garantia: 170+ testes pytest pass antes de liberar prompt       │
│                                                                        │
│  L5 — KNOWN BUGS REGISTRY (preventiva, memória)                       │
│       known_bugs.md + .claude/anti-patterns.txt                       │
│       garantia: bugs históricos NÃO retornam (KB-013, 018, 019, ...)  │
│                                                                        │
│  L6 — CODE REVIEW HAIKU (reativa, qualidade)                          │
│       agente automático após edits multi-arquivo                      │
│       garantia: review básico em <30s                                 │
│                                                                        │
│  L7 — FILE-WATCHER DAEMON (proativa, monitoramento) [NOVO §41.4]      │
│       fswatch + script Python em background                           │
│       garantia: alerta se arquivos críticos são modificados sem hook  │
└────────────────────────────────────────────────────────────────────────┘
```

### §41.2 — Mapeamento Completo: Camada × Bug Type

| Tipo de Bug | L1 Backup | L2 Static | L3 Physics | L4 Tests | L5 Known | L6 Review | L7 Watch |
|:------------|:--------:|:---------:|:----------:|:--------:|:--------:|:---------:|:--------:|
| Regressão de função | ◐ | — | — | ✅ | — | ✅ | — |
| Erro de tipo Python | — | ✅ | — | ◐ | — | ✅ | — |
| Anti-padrão (PyTorch import) | — | ✅ | — | — | ✅ | — | ✅ |
| Errata violada (FREQ=2.0) | — | ✅ | ✅ | — | ✅ | — | — |
| Numba nested prange (KB-013) | ◐ | ✅ | — | ◐ | ✅ | ✅ | ✅ |
| rng_seed=42 hardcoded (KB-018) | — | ✅ | — | ✅ | ✅ | — | — |
| Oversubscription (KB-019) | — | ✅ | — | ✅ | ✅ | — | ✅ |
| Bit-exatness Fortran <1e-12 | — | — | ✅ | ✅ | — | — | — |
| Edição manual em arquivo crítico | ✅ | — | — | — | — | ✅ | ✅ |
| Rebase quebrado | ✅ | — | — | ✅ | — | — | — |
| Branch deletada com WIP | ✅ | — | — | — | — | — | — |

✅ = camada captura tipo de bug · ◐ = parcialmente · — = irrelevante

### §41.3 — File-Watcher Daemon (NOVA L7)

**Motivação**: hooks só disparam em Edit/Write via Claude Code.
Edições manuais (Daniel direto no editor, scripts externos, git checkout
de branch errado, IDE auto-format) **não** disparam hooks. Watcher cobre
esse gap.

```python
# tools/file_watcher_daemon.py (NOVO)
"""
Daemon que monitora arquivos críticos e executa validações fora-de-banda.
Roda em background via launchd (macOS) ou systemd (Linux).
"""

import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

CRITICAL_PATHS = [
    "geosteering_ai/simulation/_numba/**/*.py",
    "geosteering_ai/simulation/_jax/**/*.py",
    "Fortran_Gerador/**/*.f08",
    "geosteering_ai/config.py",
    "CLAUDE.md",
]

class GeosteeringWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and self._is_critical(event.src_path):
            self._validate(event.src_path)

    def _is_critical(self, path: str) -> bool:
        return any(Path(path).match(p) for p in CRITICAL_PATHS)

    def _validate(self, path: str):
        # Rodar parity test em background
        # Logar em .claude/watcher.log
        # Notificar via macOS Notification Center se falha
        ...
```

```bash
# launch agent macOS: ~/Library/LaunchAgents/com.geosteering.watcher.plist
# (script auto-start daemon ao login)
```

### §41.4 — Known Bugs Registry — Estrutura Definitiva

```markdown
# docs/known_bugs.md  (NOVO)

> Registry de bugs conhecidos com sintoma + causa-raiz + fix-canônico.
> Toda regressão potencial DEVE ser registrada aqui antes do commit fix.

## KB-013 — Numba nested prange overhead (v2.13)

**Sintoma**: throughput cenário E cai de 122k → 46k mod/h.

**Causa-raiz**: `@njit(parallel=True)` em `_fields_in_freqs_kernel_cached`
(função chamada milhões de vezes de prange outer). Numba serializa nested
prange mas paga overhead de scheduler setup.

**Fix-canônico**: remover `parallel=True`, usar `range` em vez de `prange`
em funções inner.

**Detecção automática**: `.claude/anti-patterns.txt` regex
`@njit\([^)]*parallel=True[^)]*\)\s*\n\s*def _fields_in_freqs`

**Sprints afetados**: v2.13 (introduzido), v2.21 (corrigido).

---

## KB-018 — rng_seed=42 hardcoded (v2.18)

**Sintoma**: usuário reporta gerador sempre cria mesmos modelos.

**Causa-raiz**: simulation_manager.py:8088 tinha `rng_seed=42` hardcoded.

**Fix-canônico**: aceitar `Optional[int]`, default `None` → `secrets.randbits(63)`.

**Detecção automática**: `.claude/anti-patterns.txt` regex
`rng_seed\s*=\s*42[,\s\)]` em `simulation_manager.py`.

---

## KB-019 — Oversubscription threading (v2.16)

**Sintoma**: produção GUI 4× mais lenta que benchmark CLI.

**Causa-raiz**: defaults workers×threads excedem cores físicos em CPUs HT.

**Fix-canônico**: `recommend_default_parallelism()` usa phys_cores não logical.

**Detecção automática**: warning em `bench_v214_numba.py` quando
`workers × threads > phys_cores`.

---

(... estrutura para futuros KB-020, KB-021, ...)
```

### §41.5 — Anti-Patterns TSV Definitivo

```
# .claude/anti-patterns.txt   (formato: regex<TAB>severidade<TAB>razão)

@njit\([^)]*parallel=True[^)]*\)\s*\n\s*def _fields_in_freqs    BLOCK   KB-013 nested prange
rng_seed\s*=\s*42[,\s\)]                                         WARN    KB-018 hardcoded seed
import torch                                                     BLOCK   PyTorch proibido
print\s*\(                                                       WARN    Use logging em geosteering_ai/
globals\(\)\.get\(                                               BLOCK   Use PipelineConfig
TARGET_SCALING\s*=\s*["']log["']                                 BLOCK   Use "log10"
INPUT_FEATURES\s*=\s*\[0,\s*3,\s*4,\s*7,\s*8\]                  BLOCK   Use [1,4,5,20,21]
OUTPUT_TARGETS\s*=\s*\[1,\s*2\]                                  BLOCK   Use [2,3]
FREQUENCY_HZ\s*=\s*2\.0[^0-9]                                    BLOCK   Use 20000.0 (default 20kHz)
SPACING_METERS\s*=\s*1000\.0                                     BLOCK   Use 1.0 (range 0.1-10)
SEQUENCE_LENGTH\s*=\s*601                                        BLOCK   Use 600
eps_tf\s*=\s*1e-30                                               BLOCK   Use 1e-12 (float32)
```

### §41.6 — Hook check-anti-patterns.sh (atualização)

```bash
#!/bin/bash
# .claude/hooks/check-anti-patterns.sh (atualizado §41)

ANTI_PATTERNS=".claude/anti-patterns.txt"
FILES_CHANGED="$(git diff --cached --name-only --diff-filter=AM 2>/dev/null)"

[ -z "$FILES_CHANGED" ] && exit 0

violations=0
while IFS=$'\t' read -r pattern severity reason; do
    [[ "$pattern" =~ ^# ]] || [[ -z "$pattern" ]] && continue

    matches="$(echo "$FILES_CHANGED" | xargs -I {} grep -lE "$pattern" {} 2>/dev/null)"
    if [ -n "$matches" ]; then
        echo "[$severity] Anti-pattern: $reason"
        echo "  Pattern: $pattern"
        echo "  Files: $matches"
        [ "$severity" = "BLOCK" ] && violations=$((violations + 1))
    fi
done < "$ANTI_PATTERNS"

if [ "$violations" -gt 0 ]; then
    echo "❌ $violations BLOCK violations detected. Aborting."
    exit 2
fi
exit 0
```

### §41.7 — Métricas de Sucesso da Quality Mesh

| Métrica | v2.21 (sem mesh) | Meta v3.0 (com mesh) | Como medir |
|:--------|:----------------:|:--------------------:|:-----------|
| Bugs detectados antes do commit | ~30% | ≥85% | logs hooks vs bugs em produção |
| Tempo médio de detecção | ~2 sprints | <1 hora (durante edit) | timestamp introdução vs detecção |
| Regressões reintroduzidas | ~15% | <5% | bugs marcados duplicados |
| Custo de fix tardio | ~6 horas/bug | <30min/bug | tempo log de correção |
| Cobertura de testes | 170+ | 250+ | pytest --collect-only |


---

## §42 — Distribuição em Duas Trilhas: pip-lib vs Studio (Modelo Geosoft)

🔄 **REVISA** §27 (Frontend/Backend), §28 (CLI), §31 (Tecnologias) e
parte do roadmap de §11 (Workflow autodocs / produção).

### §42.1 — Mudança Estratégica de Distribuição

**Antes** (Parte I/II): único pacote `geosteering-ai` cobria tudo (lib + GUI).

**Agora** (Parte III): **2 trilhas separadas**, paralelas ao modelo Geosoft
(que distribui Geosoft Toolkit / GX SDK gratuito + Geosoft Oasis montaj
comercial):

```
┌──────────────────────────────────────────────────────────────────────────┐
│  DUAS TRILHAS DE DISTRIBUIÇÃO                                          │
│                                                                          │
│  TRILHA A — pip install geosteering-ai     (Open + Free + Developer)    │
│  ──────────────────────────────────────────────────────────────────     │
│  • Engine Python (simuladores, models, losses, training, inference)     │
│  • CLI Typer (geosteering simulate/train/infer/geosteer/setup)         │
│  • API Python para scripts/notebooks                                    │
│  • Documentação mínima necessária para uso                              │
│  • Distribuição: PyPI                                                   │
│  • Licença: Apache 2.0 ou MIT                                          │
│  • Público-alvo: pesquisadores, engenheiros de software, geofísicos     │
│    que escrevem código próprio                                          │
│                                                                          │
│  TRILHA B — Geosteering AI Studio          (Commercial + GUI Premium)   │
│  ──────────────────────────────────────────────────────────────────     │
│  • Engine pip-lib + GUI PyQt6 + LiquidGlass                            │
│  • PyQtGraph real-time + PyVista 3D + Plotly relatórios                │
│  • Integração WITSML/MWD/LWD                                           │
│  • Multi-poço, audit trail, training de operadores                     │
│  • Distribuição: instalador macOS (.dmg/.pkg) + Linux (.AppImage/.deb) │
│    + Windows (.msi) via PyInstaller / briefcase / nuitka              │
│  • Licença: comercial (proprietária)                                   │
│  • Público-alvo: empresas de oil & gas, consultorias de geosteering    │
└──────────────────────────────────────────────────────────────────────────┘
```

### §42.2 — Comparação ao Modelo Geosoft

| Dimensão | Geosoft GX SDK | Geosoft Oasis montaj | Geosteering pip-lib | Geosteering Studio |
|:---------|:---------------|:---------------------|:--------------------|:-------------------|
| Licença | Free + commercial | Comercial | Apache/MIT | Comercial |
| Audiência | Devs/scientists | Geofísicos pros | Devs/scientists | Operadores LWD |
| Interface | API + CLI | GUI Windows | API + CLI | GUI Cross-platform |
| Scripting | C++/Python | VBScript + Python | Python | Python (interna) |
| Custo | $0 (free tier) | ~$15k/seat/ano | $0 | ~$10-20k/seat/ano (modelo) |
| Atualização | GitHub releases | Anual licença | PyPI semanal | Quarterly + hotfix |

### §42.3 — Estrutura de Repos (Multi-Repo Strategy)

```
github.com/daniel-leal/
├── geosteering-ai/                    # Repo PRIMÁRIO (lib + CLI)
│   ├── geosteering_ai/                # pacote pip
│   ├── tests/
│   ├── docs/                           # MkDocs docs públicas
│   ├── notebooks/                      # tutorials
│   └── pyproject.toml                  # PyPI metadata
│
├── geosteering-studio/                 # Repo COMERCIAL (privado)
│   ├── studio/
│   │   ├── gui/                        # PyQt6
│   │   ├── workflows/                  # WITSML, multi-poço
│   │   ├── adapters/                   # SLB Petrel, Halliburton, ...
│   │   └── installer/                  # PyInstaller specs
│   ├── tests/
│   ├── docs/                           # docs proprietária + treinamento
│   └── pyproject.toml                  # depends on geosteering-ai>=2.0
│
├── geosteering-research/               # Repo de PESQUISA (público read-only)
│   ├── papers_drafts/                  # rascunhos antes de submissão
│   ├── benchmarks_publicos/
│   └── reproducibilidade/
│
└── geosteering-fortran/                # Repo HISTÓRICO (Fortran 9.0/10.0)
    └── (mantido para reprodutibilidade)
```

### §42.4 — API Pública do `pip install geosteering-ai`

```python
# Exemplo de uso da trilha A (pip-lib only)

# 1. Setup (CLI)
$ pip install geosteering-ai
$ geosteering setup                    # detecta GPU, instala deps opcionais
$ geosteering verify                   # roda smoke tests

# 2. Simulação (Python)
from geosteering_ai.simulation import SimulationConfig, simulate
cfg = SimulationConfig(
    frequency_hz=20000.0,
    rho_h=[100, 1, 100],
    rho_v=[100, 2, 100],
    thicknesses=[0, 5, 0],
    positions_m=np.linspace(-10, 10, 600),
    backend="numba",   # ou "jax" ou "fortran"
)
result = simulate(cfg)
print(result.H_tensor.shape)     # (600, 9)

# 3. Treinamento (Python)
from geosteering_ai import PipelineConfig, train
config = PipelineConfig.from_yaml("configs/robusto.yaml")
model, history = train(config, dataset_path="data/oklahoma_28.h5")

# 4. Inferência (Python)
from geosteering_ai.inference import InferencePipeline
pipeline = InferencePipeline.from_checkpoint("models/best.h5")
prediction = pipeline.invert(measurements)

# 5. Geosteering decision (Python)
from geosteering_ai.geosteering import RealtimeAgent
agent = RealtimeAgent.from_config("configs/realtime.yaml")
decision = agent.decide(current_position, lwd_measurements)
# decision.angle_correction, decision.confidence, decision.boundary_distance
```

### §42.5 — Geosteering AI Studio: Camadas Adicionais

```
┌────────────────────────────────────────────────────────────────────────┐
│  GEOSTEERING AI STUDIO — CAMADAS ADICIONAIS SOBRE pip-lib            │
├────────────────────────────────────────────────────────────────────────┤
│  Layer S1 — GUI Premium                                               │
│    • PyQt6 + LiquidGlass (macOS Tahoe)                               │
│    • Tema dark/light/auto                                             │
│    • Status bar com health checks (GPU, modelo, latência)            │
│                                                                        │
│  Layer S2 — Real-time Visualization                                   │
│    • PyQtGraph para perfis tempo real (≥10 Hz)                       │
│    • PyVista para 3D well + reservoir                                 │
│    • Plotly para relatórios pós-poço                                  │
│                                                                        │
│  Layer S3 — Integrations                                              │
│    • WITSML 1.4/2.0 server + client                                  │
│    • MWD/LWD streams (modbus, OPC-UA)                                │
│    • Petrel plugin (.NET interop)                                    │
│    • Techlog plugin (Python script bundle)                           │
│    • LAS/SEG-Y export                                                 │
│                                                                        │
│  Layer S4 — Multi-well Management                                     │
│    • Project workspace (DB SQLite local + sync opcional Postgres)    │
│    • Comparativos de poço lado-a-lado                                │
│    • Versionamento de inversões (DVC interno)                        │
│                                                                        │
│  Layer S5 — Audit + Compliance                                        │
│    • Log SOX-grade de cada decisão (timestamped, signed)             │
│    • Export para auditor (PDF + JSON)                                │
│    • Operator authentication (LDAP/SAML)                             │
│                                                                        │
│  Layer S6 — Training + Onboarding                                     │
│    • Modo "demo" com poços sintéticos pré-loaded                     │
│    • Quizzes integrados (regras petrofísicas)                        │
│    • Certificação interna por uso                                    │
└────────────────────────────────────────────────────────────────────────┘
```

### §42.6 — Sequência de Releases

```
2026 Q2 (mai-jun): pip-lib v2.0 → PyPI público
2026 Q3 (jul-set): pip-lib v2.1-2.5 (estabilização CLI + APIs)
2026 Q4 (out-dez): Geosteering AI Studio v0.1 ALPHA (interno)
2027 Q1 (jan-mar): Studio v0.5 BETA (clientes piloto)
2027 Q2 (abr-jun): Studio v1.0 GA (release comercial)
2027 Q3+:        Releases sincronizados pip-lib (mensal) + Studio (trimestral)
```

### §42.7 — Implicações para a Arquitetura Multi-Agente

| Mudança | Impacto |
|:--------|:--------|
| Separar repos | Orchestrator precisa apontar para repo correto via env var `$CLAUDE_REPO_CONTEXT` (lib vs studio) |
| Studio é privado | Agentes que tocam Studio NÃO podem rodar em CI público |
| Compatibilidade pip-lib | Toda mudança em `pip-lib` → testar contra Studio antes de release |
| Versão pin | `geosteering-studio` pin de `geosteering-ai==X.Y.*` no `pyproject.toml` |

---

## §43 — GitHub Auto-Sync Engineering

### §43.1 — Estado Atual e Gaps

| Aspecto | Estado v2.21 | Gap |
|:--------|:------------|:----|
| Push automático após commit | Manual | Falta hook ou script |
| docs/ → website | Não existe | Falta MkDocs + gh-pages |
| docs/reports/ → Wiki | Não existe | Falta sync action |
| Releases tagged | Manual via `gh release` | OK mas inconsistente |
| Issue tracking ↔ branches | Sem convenção | Bug-issue mapping perdido |
| README badges atualizados | Manual | Falta automação CI |
| CHANGELOG ↔ releases | Manual | Falta auto-extract |

### §43.2 — Política de Auto-Sync (Decisão)

```
┌────────────────────────────────────────────────────────────────────────┐
│  POLÍTICA DE AUTO-SYNC GITHUB                                         │
├────────────────────────────────────────────────────────────────────────┤
│  AUTO (sem confirmação):                                              │
│    ✅ Push automático após commit em branches feat/* e fix/* (opcional)│
│    ✅ docs/ → MkDocs build → gh-pages branch (CI)                     │
│    ✅ docs/reports/ → repo Wiki (semanal)                            │
│    ✅ Tag de versão → gh release create + auto-extract CHANGELOG      │
│    ✅ Badges README (build, coverage, version)                        │
│    ✅ docs/upgrade_proposals/ commitado pelo upgrade-scout            │
│                                                                        │
│  MANUAL (com confirmação):                                            │
│    ❌ Push em main/master                                             │
│    ❌ Force push (qualquer branch)                                    │
│    ❌ Delete branch                                                   │
│    ❌ Merge PR                                                        │
│    ❌ Close/Reopen PR/Issue                                          │
│                                                                        │
│  PROIBIDO:                                                             │
│    ❌ Force push em main                                              │
│    ❌ Skip CI (--no-verify)                                          │
│    ❌ Commit direto em main sem PR                                    │
└────────────────────────────────────────────────────────────────────────┘
```

### §43.3 — Hook post-commit-push.sh (OPCIONAL)

```bash
#!/bin/bash
# .claude/hooks/post-commit-push.sh (NOVO, OPCIONAL)
#
# Auto-push após commit em branches feat/* e fix/* (NUNCA main).
# Ativado via env var GEOSTEERING_AUTO_PUSH=1 ou settings.json.

[ "$GEOSTEERING_AUTO_PUSH" != "1" ] && exit 0

BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null)
if [[ "$BRANCH" =~ ^(main|master|release/.*)$ ]]; then
    echo "⚠️  Branch $BRANCH é protegida. Push manual necessário." >&2
    exit 0
fi

if [[ "$BRANCH" =~ ^(feat|fix|chore|docs|perf|test)/ ]]; then
    git push origin "$BRANCH" 2>&1 | tail -5
    if [ $? -eq 0 ]; then
        echo "✅ Auto-pushed $BRANCH"
    else
        echo "❌ Auto-push falhou — push manual necessário" >&2
    fi
fi
```

### §43.4 — GitHub Actions Workflow: Docs Auto-Deploy

```yaml
# .github/workflows/docs-deploy.yml (NOVO)
name: Deploy Docs

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
      - 'README.md'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install mkdocs-material mkdocstrings[python]
      - run: mkdocs build --strict
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

### §43.5 — GitHub Actions: Reports → Wiki Sync

```yaml
# .github/workflows/wiki-sync.yml (NOVO)
name: Sync Reports to Wiki

on:
  push:
    branches: [main]
    paths:
      - 'docs/reports/**'
  schedule:
    - cron: '0 8 * * 0'   # domingo 8AM UTC

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Sync to wiki
        run: |
          git clone https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.wiki.git wiki
          cp docs/reports/*.md wiki/
          cd wiki
          git add .
          git commit -m "auto-sync: reports → wiki" || true
          git push
```

### §43.6 — Repo Housekeeper Agent

```yaml
# .claude/agents/repo-housekeeper.md (NOVO)
---
name: repo-housekeeper
model: claude-haiku-4-5
output_style: caveman-full
description: Agente semanal que mantém repo organizado.
schedule: "0 9 * * 1"   # segundas 9AM (cron via tools/scheduled_runner.py)
tasks:
  - Verificar branches stale (>30d sem commit) e propor delete
  - Verificar issues sem label e propor classificação
  - Atualizar badges do README (build, coverage, version, downloads)
  - Verificar docs/ROADMAP.md vs CHANGELOG.md (consistência)
  - Verificar gitignore vs arquivos commitados (vazamentos)
  - Verificar pyproject.toml deps vs realmente usadas
  - Gerar `docs/reports/housekeeping_YYYY-WW.md` (resumo semanal)
output: PR com label "chore" e descrição estruturada
---
```

### §43.7 — Versionamento Semântico Automatizado

```python
# tools/version_bump.py (NOVO)
"""
Auto-bumps versão em pyproject.toml e CLAUDE.md baseado em commits desde
último tag.
"""
# Regras:
#   • feat: → minor bump (2.0.0 → 2.1.0)
#   • fix: → patch bump (2.1.0 → 2.1.1)
#   • BREAKING CHANGE em footer → major bump (2.1.1 → 3.0.0)
#   • perf: → patch bump
#   • docs/test/chore: → sem bump

# Uso:
#   $ python tools/version_bump.py --dry-run
#   Next version: 2.1.0 (from 2.0.5)
#   Reasoning: 3 feat: commits since v2.0.5
```

### §43.8 — Métricas e Critérios de Aceite

| Métrica | Estado v2.21 | Meta v3.0 |
|:--------|:------------:|:---------:|
| docs/ no GitHub Pages atualizada | Não | <5min após push em main |
| docs/reports/ na Wiki | Não | <24h após push |
| Issues com label correto | ~40% | ≥90% (housekeeper agent) |
| Releases com CHANGELOG extraído | Manual | Auto via gh release |
| Branches stale (>30d) | ~12 | ≤2 (housekeeper limpa) |
| README com badges atualizados | Stale | Auto-bump CI |


---

## §44 — Workflow W13: JAX Sprint

🔄 **REVISA** §10 da Parte I (12 workflows → agora 13).

### §44.1 — Motivação

O simulador JAX (`geosteering_ai/simulation/_jax/`) tem 9 módulos
(`dipoles_native`, `dipoles_unified`, `forward_pure`, `geometry_jax`,
`hankel`, `kernel`, `multi_forward`, `propagation`, `rotation`) e atingiu
v1.6.0 com paridade <1e-12 vs Numba. **Falta**: validação contínua em GPU
T4/A100 e otimização para multi-pose × multi-frequência via vmap aninhado.

W13 **JAX Sprint** é workflow especializado em otimização GPU.

### §44.2 — Topologia do W13

```
┌──────────────────────────────────────────────────────────────────────────┐
│  WORKFLOW W13 — JAX SPRINT                                              │
│  Duração estimada: 6-12h por sprint                                     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │  FASE 0 — TRIAGE (orchestrator Opus, 15min)                       ││
│  │  • Identificar gargalo: profile JAX ou paridade?                  ││
│  │  • Rodar bench_sprint12_regression.py em short mode (CPU)         ││
│  │  • Decidir alvo: vmap real, fori_loop, unified JIT, mempool       ││
│  └────────────────────────────────────────────────────────────────────┘│
│                            ↓                                            │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │  FASE 1 — ANÁLISE LOCAL CPU (jax-engineer Sonnet, 1-2h)           ││
│  │  • jax.make_jaxpr para inspeção de jaxpr                          ││
│  │  • XLA HLO dump (TF_XLA_FLAGS=--xla_dump_to=...)                 ││
│  │  • count_compiled_xla_programs (existente em forward_pure.py)     ││
│  │  • Identificar candidatos a fusão / vmap / fori_loop              ││
│  └────────────────────────────────────────────────────────────────────┘│
│                            ↓                                            │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │  FASE 2 — IMPLEMENTAÇÃO LOCAL (jax-engineer Sonnet, 2-4h)         ││
│  │  • Edita _jax/*.py (com lock + worktree §40)                     ││
│  │  • Mantém backward compat: novos paths via flag opt-in           ││
│  │  • Testes locais CPU: pytest -k jax                              ││
│  │  • Paridade vs bucketed bit-exata                                 ││
│  └────────────────────────────────────────────────────────────────────┘│
│                            ↓                                            │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │  FASE 3 — VALIDAÇÃO GPU REMOTA (colab-runner Sonnet, 1-3h)        ││
│  │  • Push branch para GitHub                                         ││
│  │  • Trigger notebook bench_jax_gpu_colab.ipynb                     ││
│  │  • Roda em Colab T4 (free) ou A100 (Pro+)                        ││
│  │  • Captura métricas: throughput, peak memory, JIT compile time   ││
│  │  • Compara vs baseline armazenado                                 ││
│  └────────────────────────────────────────────────────────────────────┘│
│                            ↓                                            │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │  FASE 4 — VALIDAÇÃO PARIDADE (numba-jax-parity-validator Haiku)   ││
│  │  • Rodar suite test_simulation_compare_fortran.py em CPU         ││
│  │  • Rodar tests/test_simulation_jax_*.py                          ││
│  │  • Gate: <1e-12 vs Numba, <1e-10 vs Fortran                      ││
│  │  • Falha → rollback wave §40.3 L4                                ││
│  └────────────────────────────────────────────────────────────────────┘│
│                            ↓                                            │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │  FASE 5 — DOCUMENTAÇÃO (docs-writer Sonnet, 30min)                ││
│  │  • Criar docs/reports/jax_sprint_<N>_YYYY-MM-DD.md                ││
│  │  • Atualizar docs/reference/jax_optimization_log.md (cumulativo)  ││
│  │  • Atualizar CHANGELOG.md com entry de versão                     ││
│  │  • Memory: project_simulation_manager_jax_<v>.md                  ││
│  └────────────────────────────────────────────────────────────────────┘│
│                            ↓                                            │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │  FASE 6 — RELEASE (sprint-orchestrator, 15min)                    ││
│  │  • Tag git: jax-vX.Y.Z                                             ││
│  │  • PR com label "jax-sprint"                                       ││
│  │  • Notebooks Colab versionados (mover para notebooks/archive/)   ││
│  └────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
```

### §44.3 — Agentes Específicos do W13

| Agente | Modelo | Papel | Skill |
|:-------|:------:|:------|:------|
| **jax-engineer** | Sonnet 4.6 | Edita `_jax/*.py`, otimiza JIT | `geosteering-simulator-jax` |
| **numba-jax-parity-validator** | Haiku 4.5 | Roda suite paridade Numba↔JAX | `geosteering-numba-jax-parity` |
| **colab-runner** | Sonnet 4.6 | Coordena execução Colab GPU + análise resultados | `geosteering-colab-runner` |
| **bench-validator** | Haiku 4.5 | Compara métricas atuais vs baseline armazenado | `geosteering-bench-validator` |
| **fortran-parity-validator** | Haiku 4.5 | Mantém <1e-12 vs Fortran (existente) | `geosteering-fortran-parity` |

### §44.4 — Métricas de Sucesso por Sprint W13

| Métrica | T4 (free) Meta | A100 (Pro+) Meta | Como medir |
|:--------|:--------------:|:----------------:|:-----------|
| Throughput multi-TR × multi-ang | ≥200k mod/h | ≥500k mod/h | bench_sprint12_regression.py |
| XLA programs compiled | ≤5 | ≤5 | count_compiled_xla_programs |
| Peak GPU memory | ≤8GB (T4) | ≤16GB (A100) | jax.devices() + nvidia-smi |
| JIT compile time | ≤30s | ≤30s | timestamp first call |
| Paridade vs Numba | <1e-12 | <1e-12 | tests/test_simulation_jax_*.py |
| Paridade vs Fortran | <1e-10 | <1e-10 | test_simulation_compare_fortran.py |

### §44.5 — Backlog Inicial de Sprints W13

```
W13.1 — vmap real validation T4    (1 sprint, 6h)
        cfg.jax_vmap_real=True como default em GPU
        Esperado: 1.5-3× speedup vs Python loop

W13.2 — Memory pool tuning          (1 sprint, 6h)
        XLA_PYTHON_CLIENT_MEM_FRACTION otimizado
        Reduzir OOM em A100 batches grandes

W13.3 — fori_loop vs scan benchmark (1 sprint, 8h)
        Comparar dois primitives em propagation
        Decidir baseline para v1.7

W13.4 — multi-poso GPU              (1 sprint, 8h)
        Vetorizar múltiplas posições de antena
        Esperado: 2× redução de overhead Python

W13.5 — Mixed precision FP32→BF16   (1 sprint, 12h)
        Avaliar paridade em BF16 (T4 tem TF32)
        Trade-off: 2× speedup vs precisão Fortran

W13.6 — A100 pmap multi-device      (1 sprint, 12h)
        Parallel map em múltiplas GPUs
        Caso: cluster de A100 (futuro)

W13.7 — Sharding (jax.sharding)     (1 sprint, 12h)
        Particionar batch entre devices
        Usar quando passar 1 GPU
```

### §44.6 — Notebook Padrão Colab para W13

```python
# notebooks/jax_sprint_template.ipynb (NOVO)
"""
Notebook template para sprints W13.
Estrutura padronizada:
  1. Setup: pip install + verificação GPU
  2. Pull branch atual: git pull origin <branch>
  3. Bench baseline: bench_sprint12_regression.py
  4. Bench novo: bench_jax_gpu_colab_v<N>.ipynb
  5. Comparativo: throughput, memory, paridade
  6. Output: docs/reports/jax_sprint_<N>.md
"""
# Cell 1
!pip install -e git+https://github.com/daniel-leal/geosteering-ai.git@feat/jax-sprint-N

# Cell 2 — verificação GPU
import jax
print(jax.devices())  # esperado: [CudaDevice(id=0)]

# Cell 3 — bench
from benchmarks.bench_sprint12_regression import run
results = run(mode="full", devices=jax.devices())
results.to_csv("/content/results.csv")

# Cell 4 — paridade
import pytest
pytest.main(["tests/test_simulation_jax_sprint12.py", "-v"])

# Cell 5 — push results back
!git config user.email "colab-runner@geosteering"
!git add docs/reports/ && git commit -m "feat(jax): sprint N results"
!git push
```

### §44.7 — Atualização do Índice de Workflows (12 → 13)

```
WORKFLOWS END-TO-END (atualizado):
  W01  Numba bench
  W02  JAX GPU Colab
  W03  SurrogateNet ModernTCN training
  W04  1D synthetic dataset generation
  W05  Fortran validation
  W06  PINN petrofísica
  W07  Geosteering simulator
  W08  Autodocs
  W09  Literature research
  W10  CI/CD + tests
  W11  Docker build
  W12  MLOps checkpoint
  W13  JAX Sprint  ← NOVO Parte III §44
```

---

## §45 — Roteiro Detalhado de Implementação Semana-a-Semana

🔄 **REVISA E EXPANDE** §11 (Roadmap 4-fase) e §37 (Roteiro de execução).

### §45.1 — Resumo do Roteiro Atual

O documento (Parte I/II) cobre:
- §11 Roadmap em 4 fases (24 sprints, ~24 semanas)
- §37 Roteiro de execução semanal genérico
- Cada sprint tem entrega + critérios de aceite parciais

**Lacuna**: falta cronograma semana-a-semana com **datas concretas**,
**dependências entre sprints** e **gates de qualidade** explícitos.

### §45.2 — Cronograma 24 Semanas (FASES 1-4)

```
═══════════════════════════════════════════════════════════════════════════
  FASE 1 — CONSOLIDAÇÃO E BACKBONE (4 semanas, semanas 1-4)
═══════════════════════════════════════════════════════════════════════════

  Semana 1 (2026-05-04 a 2026-05-10) — INFRAESTRUTURA QUALITY MESH
  ───────────────────────────────────────────────────────────────────
  Sprint 1.1: backup-pre-edit.sh + .backups/ + .gitignore (§38)
  Sprint 1.2: check-anti-patterns.sh + .claude/anti-patterns.txt (§35/§41)
  Sprint 1.3: known_bugs.md inicial (KB-013/018/019) (§41.4)
  Sprint 1.4: run-fortran-parity.sh + smoke gate <1e-12 (§35)
  Gate Semana 1: hooks ativos, 12 testes pytest novos passing

  Semana 2 (2026-05-11 a 2026-05-17) — MULTI-AGENT HARDENING
  ───────────────────────────────────────────────────────────────────
  Sprint 2.1: agent-acquire-lock.sh + agent-release-lock.sh (§40.5)
  Sprint 2.2: parallelism_rules.py + conflict matrix (§40.4)
  Sprint 2.3: parallelism_dashboard.py + telemetria (§40.7)
  Sprint 2.4: rollback wave em sprint-orchestrator (§40.3 L4)
  Gate Semana 2: paralelismo seguro com ≥2 agentes simultâneos validado

  Semana 3 (2026-05-18 a 2026-05-24) — V2.22 FLAT PRANGE NUMBA
  ───────────────────────────────────────────────────────────────────
  Sprint 3.1: _simulate_combined_prange_flat (forward.py)
  Sprint 3.2: bench Cenário B (multi-freq) — meta ≥150k mod/h
  Sprint 3.3: paridade Fortran <1e-12 + 8 cenários canônicos
  Sprint 3.4: docs/reports/v2.22_YYYY-MM-DD.md
  Gate Semana 3: Cenário B 376k → ≥600k mod/h (1.6×)

  Semana 4 (2026-05-25 a 2026-05-31) — V2.23 TILE/BLOCK + V2.24 HANKEL PRECOMPUTE
  ───────────────────────────────────────────────────────────────────
  Sprint 4.1: tile/block schedule em _simulate_positions (forward.py)
  Sprint 4.2: pre-compute Hankel kernels em filters/loader (§7 doc cenários)
  Sprint 4.3: bench todos cenários A-H + paridade
  Sprint 4.4: tag v2.24 + release notes
  Gate Fase 1: Cenários A=>1.5M, B=>700k, E=>200k mod/h consolidado

═══════════════════════════════════════════════════════════════════════════
  FASE 2 — SIMULADOR JAX OTIMIZAÇÃO GPU (6 semanas, semanas 5-10)
═══════════════════════════════════════════════════════════════════════════

  Semana 5: W13.1 vmap real T4 default
  Semana 6: W13.2 memory pool tuning + W13.3 fori_loop vs scan
  Semana 7: W13.4 multi-poso GPU
  Semana 8: W13.5 mixed precision FP32→BF16 (cuidado paridade)
  Semana 9: W13.6 pmap multi-device + W13.7 sharding (a partir de 2 GPUs)
  Semana 10: Consolidação JAX v2.0 + paridade Fortran <1e-10 confirmada

  Gate Fase 2: T4 ≥200k mod/h + A100 ≥500k mod/h + paridade preservada

═══════════════════════════════════════════════════════════════════════════
  FASE 3 — DEEP LEARNING MADURO (10 semanas, semanas 11-20)
═══════════════════════════════════════════════════════════════════════════

  Semana 11-12: Treinamento SurrogateNet ModernTCN (W03)
                Dataset multi-dip simulado em GPU local + Colab
                Validação contra Fortran in-loop

  Semana 13-14: 48 arquiteturas batch training Colab Pro+
                Ranking por inversão sintética
                Top-5 selecionadas para data real

  Semana 15-16: Treinamento PINNs petrofísica (W06)
                Constraints Archie + Klein
                Ablation: PINN vs vanilla DL

  Semana 17-18: Inversão dados reais (1D synthetic + dados públicos)
                INN + ensemble + UQ
                Comparativo SkyTEM, Geosphere HD

  Semana 19-20: Estabilização + auditoria + documentação
                Pré-release v3.0 (Studio alpha-internal)

  Gate Fase 3: 5 modelos top-tier validados em dados reais públicos

═══════════════════════════════════════════════════════════════════════════
  FASE 4 — PRODUÇÃO E STUDIO (4 semanas, semanas 21-24)
═══════════════════════════════════════════════════════════════════════════

  Semana 21: Studio v0.1 alpha — GUI PyQt6 + LiquidGlass + WITSML
  Semana 22: Studio v0.2 — multi-poço + audit trail + Petrel adapter
  Semana 23: Studio v0.3 — instaladores macOS/Linux/Windows
  Semana 24: Studio v0.5 BETA + 2-3 clientes piloto + pip-lib v3.0 GA

  Gate Fase 4: Studio rodando em ≥1 cliente real, pip-lib v3.0 no PyPI
═══════════════════════════════════════════════════════════════════════════
```

### §45.3 — Checklist Diário Padronizado

```markdown
# Daily Checklist — Geosteering AI

## Início do dia (5 min)
- [ ] git status + git pull (verificar updates upstream)
- [ ] Ler MEMORY.md + 2 últimas entries de docs/reports/
- [ ] Verificar dashboard parallelism (.claude/telemetry/)
- [ ] Verificar `gh pr status` (PRs abertos pendentes)
- [ ] Verificar `gh issue list --label bug` (bugs reportados)

## Durante trabalho
- [ ] TodoWrite para sprint atual
- [ ] Backup-pre-edit ativo (verificar `.backups/<hoje>/`)
- [ ] Lock acquisition em edits multi-arquivo
- [ ] Pytest após cada feature (run-pytest.sh dispara automaticamente)
- [ ] Update known_bugs.md ANTES de fix

## Final do dia (10 min)
- [ ] Commitar trabalho em branch atual
- [ ] Push (manual em main, automático em feat/*)
- [ ] Atualizar TODO list (mover completed → arquivar)
- [ ] Ler housekeeper digest (se segunda-feira)
- [ ] Validar memory atualizada se decisões importantes
```

### §45.4 — Gates de Qualidade por Sprint

| Gate | Verificações | Bloqueia se |
|:-----|:------------|:-----------|
| **G1: Pre-edit** | backup-pre-edit ativo + lock adquirido | sem ambos |
| **G2: Static** | ruff + mypy + check-anti-patterns | qualquer ERROR |
| **G3: Physics** | validate-physics + Fortran parity <1e-12 | falha em qualquer canônico |
| **G4: Tests** | pytest com 0 fail; 170+ tests | qualquer FAIL |
| **G5: Coverage** | coverage não cai >2% | regressão >2% |
| **G6: Review** | code-review-haiku-agent OK | severity ≥ ALTO |
| **G7: Docs** | CHANGELOG + ROADMAP + report sprint | docs ausentes |
| **G8: Memory** | memory atualizada se decisão arquit. | decisão sem memory |

### §45.5 — Sinal de Alerta (Quando Acionar Opus)

| Sinal | Magnitude | Ação |
|:------|:---------:|:-----|
| Sprint multi-arquivo (≥5 arquivos) | Sempre | Opus orchestrator |
| Refatoração arquitetural | Sempre | Opus orchestrator + plan |
| Paridade Fortran ameaçada | Sempre | Opus + multi-agent review |
| Gargalo crítico não-trivial | Caso a caso | Opus para análise |
| Decisão de release (v3.0) | Sempre | Opus + Daniel |
| Diálogo curto sobre código existente | — | Sonnet ou Haiku |
| Lookup de documentação | — | Haiku |
| Status report semanal | — | Sonnet |


---

## §46 — Agente Upgrade Scout: 18º Agente da Arquitetura

🔄 **REVISA** §11 (8+3+orchestrator = 12 agentes) → agora **18 agentes**.

### §46.1 — Motivação

Geosteering AI depende de 100+ libraries em rápida evolução: TensorFlow,
JAX, Numba, PyQt6, NumPy, SciPy, scikit-learn, Pandas, Plotly,
PyVista, mkdocs-material, plus literatura de ML/Geofísica/Inversão
publicada diariamente em ArXiv/bioRxiv. **Sem agente dedicado a scan
de upgrades**, o projeto **drifta** em obsolescência.

### §46.2 — Especificação do Agente

```yaml
# .claude/agents/upgrade-scout.md (NOVO)
---
name: upgrade-scout
model: claude-sonnet-4-6
output_style: caveman-full   # respostas curtas; relatórios via arquivo
description: |
  Agente semanal que escaneia novas versões de dependências, papers
  recentes, e melhores práticas, propondo upgrades em formato MD.
schedule: "0 10 * * 1"   # segundas 10AM via cron
triggers:
  - manual: "/upgrade-scout"
  - automatic: weekly_cron
  - on_event: dependency_release_via_dependabot

scope:
  python_packages:
    - tensorflow
    - jax
    - numba
    - numpy
    - scipy
    - scikit-learn
    - pandas
    - PyQt6
    - plotly
    - pyvista
    - pyqtgraph
    - matplotlib
    - mkdocs-material
    - typer
    - pydantic
    - mlflow
    - dvc
    - empymod
    - watchdog

  toolchain:
    - claude-code (Anthropic releases)
    - claude-skills (community)
    - github-actions
    - pre-commit hooks

  literature:
    - arxiv (categorias: cs.LG, physics.geo-ph, eess.SP)
    - biorxiv (não relevante; skipped)
    - consensus
    - semantic-scholar

  benchmarks:
    - PyPI download stats (ranking de adoção)
    - StarHistory (popularidade GitHub)
    - Tiobe / Anthropic blog

output:
  formato: markdown
  destino: docs/upgrade_proposals/YYYY-WW.md
  estrutura:
    - Cabeçalho (semana, data, total propostas)
    - Tabela de upgrades (lib, versão atual, nova, breaking, ROI)
    - Papers relevantes (3-5 com TL;DR e implicação para projeto)
    - Recomendação top-3 ordenada por ROI/risco
    - Próximas ações sugeridas (sprint backlog)

actions_permitidas:
  - LER arquivos do projeto
  - EXECUTAR pip list, pip search, gh release list
  - WEBSEARCH/WEBFETCH (ArXiv, GitHub, PyPI)
  - ESCREVER apenas em docs/upgrade_proposals/
  - CRIAR issue no GitHub com label "upgrade-proposal"

actions_proibidas:
  - EDITAR código fonte (apenas propõe, nunca aplica)
  - EDITAR pyproject.toml ou requirements
  - PUSH para main
  - MERGE de PR
---
```

### §46.3 — Output Padrão (Template MD)

```markdown
# Upgrade Proposals — Semana 2026-W18

> Gerado automaticamente por upgrade-scout em 2026-05-04 10:15 UTC
> Próxima execução: 2026-05-11

## Resumo Executivo

| Categoria | Propostas | Top ROI |
|:----------|:---------:|:--------|
| Python deps | 7 | numpy 2.3 (perf +12%) |
| Claude Skills | 2 | superpowers v3.4 |
| Papers | 4 | "Bayesian INN for EM inversion" (Chen 2026) |
| Toolchain | 1 | uv 0.5 (instala 5× mais rápido) |
| **TOTAL** | **14** | — |

---

## §1 — Python Dependencies

| Lib | Atual | Nova | Breaking | ROI | Risco | Recomendação |
|:----|:-----:|:----:|:--------:|:---:|:-----:|:-------------|
| numpy | 2.2.1 | 2.3.0 | Não | Alta | Baixo | UPGRADE |
| jax | 0.4.30 | 0.5.0 | Sim (XLA) | Alta | Médio | TESTAR |
| numba | 0.60.0 | 0.62.0 | Não | Média | Baixo | UPGRADE |
| pyqt6 | 6.7.0 | 6.8.0 | Não | Baixa | Baixo | OPCIONAL |
| ... | | | | | | |

## §2 — Papers Relevantes (ArXiv/Consensus)

### #1 — "Bayesian INN for Electromagnetic Inversion" (Chen et al. 2026, ArXiv:2604.XXXXX)

**TL;DR**: Invertible Neural Networks com priors Bayesianos atingem
~12% melhor RMSE em inversão 1D EM vs INN clássico. Custo: +25% treinamento.

**Implicação para Geosteering AI**:
- Comparar com nossa implementação INN atual (geosteering_ai/models/inn.py)
- Avaliar prior Bayesiano em sprint Fase 3 W11

### #2 — ...

## §3 — Claude Skills + Toolchain

| Skill / Tool | Versão | Mudança | Adoção sugerida |
|:-------------|:------:|:--------|:----------------|
| superpowers | v3.4 | +brainstorming structured | Avaliar Fase 1 |
| caveman | v2.1 | +ultra-mode bug fix | Já adotado §39 |
| uv | 0.5.0 | 5× faster installs | Substituir pip em CI |

## §4 — Recomendação Top-3 (ordem de prioridade)

### 🥇 numpy 2.3 — UPGRADE imediato
- ROI: +12% performance Numba
- Risco: baixo (sem breaking)
- Esforço: 30 min (bump + pytest)

### 🥈 uv 0.5 em CI — UPGRADE
- ROI: 5× faster CI installs ($savings tempo)
- Risco: baixo (uv é drop-in para pip)
- Esforço: 1h (atualizar Actions YAML)

### 🥉 Bayesian INN paper — INVESTIGAR
- ROI: +12% RMSE
- Risco: médio (paper pré-print, sem código público ainda)
- Esforço: 8h (reimplementação) — adicionar ao backlog Fase 3

---

## Issue Criada

GitHub issue #XXX: "[upgrade] Top-3 propostas semana W18"
Label: upgrade-proposal · Assignee: @daniel-leal
```

### §46.4 — Atualização Total da Topologia: 18 Agentes

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TOPOLOGIA COMPLETA — 18 AGENTES                                        │
├──────────────────────────────────────────────────────────────────────────┤
│  Camada 0 — Orchestrator (1 agente Opus)                                │
│    1. sprint-orchestrator                                                │
│                                                                          │
│  Camada 1 — Domain Specialists (8 agentes Sonnet)                       │
│    2. numba-jit-engineer                                                 │
│    3. jax-engineer                                                       │
│    4. dl-training-engineer                                               │
│    5. inversion-engineer (1D)                                            │
│    6. geosteering-decision-agent                                         │
│    7. literature-research-agent                                          │
│    8. docs-writer                                                        │
│    9. mlops-engineer                                                     │
│                                                                          │
│  Camada 2 — Quality + Maintenance (5 agentes Haiku)                     │
│    10. code-review-agent                                                 │
│    11. fortran-parity-validator                                          │
│    12. continuous-benchmark-agent                                        │
│    13. numba-jax-parity-validator (NOVO §44)                            │
│    14. bench-validator (NOVO §44)                                        │
│                                                                          │
│  Camada 3 — Orquestração de Tarefas Específicas (3 agentes Sonnet)      │
│    15. colab-runner (NOVO §44)                                           │
│    16. repo-housekeeper (NOVO §43)                                       │
│    17. upgrade-scout (NOVO §46)                                          │
│                                                                          │
│  Camada 4 — Review Externo Cross-LLM (1 agente Sonnet)                  │
│    18. codex-reviewer (existente §36)                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

### §46.5 — Capacidade Universal: Todos os Agentes Propõem Upgrades

Independente do upgrade-scout, **todos os 18 agentes** ganham instrução
universal: ao detectar oportunidade de melhoria fora do escopo da tarefa
atual, criar entry em `docs/upgrade_proposals/_continuous.md` (arquivo
contínuo, não-semanal):

```yaml
# Em todo agente: .claude/agents/<nome>.md
---
universal_instruction: |
  Se durante sua tarefa você identificar:
    • Anti-padrão não capturado em check-anti-patterns
    • Library mais nova/melhor que a atual
    • Padrão arquitetural superior
    • Bug latente em código alheio
    • Otimização não-essencial à tarefa atual
  → Adicionar entry em `docs/upgrade_proposals/_continuous.md`
    com formato: `[YYYY-MM-DD HH:MM | <agente> | <severidade>] <descrição>`
  Mas NÃO desviar da tarefa atual.
---
```

---

## §47 — Setup Local Automatizado (Cross-Platform)

### §47.1 — Comando Único: `geosteering setup`

```bash
# Instalação rápida do projeto (clone + setup)
git clone https://github.com/daniel-leal/geosteering-ai.git
cd geosteering-ai
./tools/setup_environment.sh    # detecta OS, instala tudo
# OU após pip install:
geosteering setup
```

### §47.2 — Script `setup_environment.sh` Cross-Platform

```bash
#!/bin/bash
# tools/setup_environment.sh (NOVO)
#
# Setup completo do ambiente de desenvolvimento Geosteering AI.
# Suporta: macOS (brew), Linux (apt/yum/dnf), Windows (scoop via WSL).

set -euo pipefail

LOG="setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "=== Geosteering AI Setup ==="
echo "OS: $OS · Arch: $ARCH"
echo "Log: $LOG"
echo ""

# ─── Função: verificar comando existe ──────────────────────────────────────
check_cmd() {
    command -v "$1" &>/dev/null
}

# ─── 1. Package manager + system deps ──────────────────────────────────────
case "$OS" in
    Darwin)
        echo "→ macOS detectado. Verificando Homebrew..."
        if ! check_cmd brew; then
            echo "Instalando Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        echo "→ Instalando deps via brew..."
        brew install gfortran openblas hdf5 git gh node uv
        ;;
    Linux)
        if check_cmd apt; then
            echo "→ Linux apt-based detectado."
            sudo apt update
            sudo apt install -y gfortran libopenblas-dev libhdf5-dev git nodejs build-essential
        elif check_cmd dnf; then
            echo "→ Linux dnf-based detectado (Fedora/RHEL)."
            sudo dnf install -y gcc-gfortran openblas-devel hdf5-devel git nodejs gcc-c++
        else
            echo "❌ Distro Linux não suportada (apt/dnf)" >&2
            exit 1
        fi
        ;;
    MINGW*|CYGWIN*|MSYS*)
        echo "❌ Windows detectado. Por favor, use WSL2 (Ubuntu)" >&2
        echo "   Veja: https://docs.microsoft.com/windows/wsl/install"
        exit 1
        ;;
    *)
        echo "❌ OS não suportado: $OS" >&2
        exit 1
        ;;
esac

# ─── 2. Python 3.13 ─────────────────────────────────────────────────────────
echo "→ Verificando Python 3.13..."
if ! check_cmd python3.13; then
    case "$OS" in
        Darwin) brew install python@3.13 ;;
        Linux)
            if check_cmd apt; then
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt update
                sudo apt install -y python3.13 python3.13-venv python3.13-dev
            elif check_cmd dnf; then
                sudo dnf install -y python3.13 python3.13-devel
            fi
        ;;
    esac
fi

# ─── 3. Virtualenv ──────────────────────────────────────────────────────────
VENV="$HOME/Geosteering_AI_venv"
if [ ! -d "$VENV" ]; then
    echo "→ Criando venv em $VENV..."
    python3.13 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# ─── 4. uv (faster pip) ─────────────────────────────────────────────────────
echo "→ Instalando uv (fast installer)..."
if ! check_cmd uv; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# ─── 5. Project deps via uv ─────────────────────────────────────────────────
echo "→ Instalando dependências do projeto via uv..."
uv pip install -e ".[dev,all]"

# ─── 6. GPU detection (delegado a tools/gpu_detect.py) ─────────────────────
echo "→ Detectando GPU..."
python tools/gpu_detect.py --print-summary || echo "(GPU detect falhou, sem GPU)"

# ─── 7. Pre-commit hooks ────────────────────────────────────────────────────
echo "→ Instalando pre-commit hooks..."
uv pip install pre-commit
pre-commit install

# ─── 8. Git config (sugerido, não-destrutivo) ─────────────────────────────
echo "→ Sugestões de git config (não aplicado automaticamente):"
echo "  git config user.signingkey <SUA_GPG_KEY>"
echo "  git config commit.gpgsign true"
echo "  git config push.autoSetupRemote true"

# ─── 9. Smoke test ──────────────────────────────────────────────────────────
echo "→ Rodando smoke test..."
pytest tests/ -k smoke --tb=short || echo "⚠️  Smoke test falhou — verificar log"

# ─── 10. Resumo ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Setup completo!"
echo "  Venv: $VENV"
echo "  Próximos passos:"
echo "    1. source $VENV/bin/activate"
echo "    2. cd $(pwd)"
echo "    3. pytest tests/ -v --tb=short"
echo "    4. geosteering --help"
echo "═══════════════════════════════════════════════════════════════════"
```

### §47.3 — Comando CLI `geosteering setup` (Pure Python)

```python
# geosteering_ai/cli/setup_command.py (NOVO)
"""
Comando CLI Typer que verifica e instala dependências.
Usa rich para output colorido.
"""

import typer
import subprocess
import platform
from pathlib import Path
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

@app.command()
def setup(
    skip_brew: bool = typer.Option(False, help="Skip Homebrew/apt install"),
    gpu_only: bool = typer.Option(False, help="Setup apenas GPU detection"),
):
    """Setup completo do ambiente Geosteering AI."""

    table = Table(title="Setup Status")
    table.add_column("Componente")
    table.add_column("Status")

    # 1. Python version
    py_ver = platform.python_version()
    py_ok = py_ver.startswith("3.13")
    table.add_row("Python 3.13", "✅" if py_ok else f"❌ ({py_ver})")

    # 2. JAX
    try:
        import jax
        table.add_row("JAX", f"✅ {jax.__version__}")
    except ImportError:
        table.add_row("JAX", "❌ não instalado")

    # 3. TensorFlow
    try:
        import tensorflow as tf
        table.add_row("TensorFlow", f"✅ {tf.__version__}")
    except ImportError:
        table.add_row("TensorFlow", "❌")

    # 4. Numba
    try:
        import numba
        table.add_row("Numba", f"✅ {numba.__version__}")
    except ImportError:
        table.add_row("Numba", "❌")

    # 5. PyQt6
    try:
        from PyQt6 import QtCore
        table.add_row("PyQt6", f"✅ {QtCore.QT_VERSION_STR}")
    except ImportError:
        table.add_row("PyQt6", "⚠️ (apenas para Studio)")

    # 6. GPU
    from geosteering_ai.utils.gpu_detect import detect_gpu
    gpu_info = detect_gpu()
    table.add_row("GPU", gpu_info.summary)

    console.print(table)

    if not gpu_only:
        # Sugerir comandos faltantes
        console.print("\n[bold]Próximos passos:[/]")
        console.print("  1. source ~/Geosteering_AI_venv/bin/activate")
        console.print("  2. pytest tests/ -v")
        console.print("  3. geosteering simulate --help")
```

### §47.4 — Dependências por Categoria

```toml
# pyproject.toml (excerpt — atualizado para Parte III)

[project]
dependencies = [
    "numpy>=2.0,<3",
    "scipy>=1.12",
    "numba>=0.60",
    "jax>=0.4.30",
    "tensorflow>=2.16,<3",
    "scikit-learn>=1.4",
    "pandas>=2.2",
    "h5py>=3.10",
    "typer[all]>=0.12",
    "rich>=13.7",
    "pydantic>=2.7",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-xdist>=3.5",
    "ruff>=0.4",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "uv>=0.5",
]
gui = [           # apenas para Studio
    "PyQt6>=6.7",
    "PyQt6-Charts>=6.7",
    "pyqtgraph>=0.13",
    "pyvista>=0.43",
    "matplotlib>=3.8",
    "plotly>=5.20",
]
gpu = [
    "jax[cuda12]>=0.4.30",
]
mlops = [
    "mlflow>=2.12",
    "dvc>=3.50",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.25",
]
all = [
    "geosteering-ai[dev,gui,mlops,docs]",
]
```

---

## §48 — GPU Detection Policy

🔄 **REFINA** §29 da Parte II (GPU local primário).

### §48.1 — Política Definitiva (Resumo)

```
┌────────────────────────────────────────────────────────────────────────┐
│  POLÍTICA DE GPU PARA GEOSTEERING AI                                  │
├────────────────────────────────────────────────────────────────────────┤
│  CONTEXTO              GPU LOCAL?         GPU REMOTA?                 │
│  ────────────────────  ─────────────────  ──────────────────          │
│  macOS (Daniel atual)  ❌ Apple Metal     ✅ Colab Pro+ T4/A100       │
│                        (limitação JAX)                                │
│                                                                        │
│  Linux + NVIDIA        ✅ LOCAL DEFAULT   ⚠️ Colab opcional p/ A100   │
│  Linux + AMD           ⚠️ ROCm beta       ✅ Colab Pro+               │
│  Linux + sem GPU       ❌                  ✅ Colab Pro+               │
│  Windows + NVIDIA      ⚠️ WSL2 obrigatório ✅ Colab Pro+               │
│  Servidor on-prem      ✅ via SSH         —                          │
└────────────────────────────────────────────────────────────────────────┘
```

### §48.2 — Implementação `gpu_detect.py`

```python
# geosteering_ai/utils/gpu_detect.py (atualização §48 da Parte III)
"""
Detecta GPU local e recomenda backend de treino/inferência.

Política:
  • macOS → Colab default (Apple Metal não suportado em JAX hoje)
  • Linux + NVIDIA → local default
  • Linux sem GPU ou AMD beta → Colab default
  • Windows + NVIDIA → WSL2 obrigatório, depois local
"""

import platform
import subprocess
from dataclasses import dataclass
from enum import Enum

class GPUBackend(Enum):
    LOCAL_NVIDIA   = "local_nvidia"
    LOCAL_AMD      = "local_amd"
    LOCAL_APPLE    = "local_apple"      # Metal não usado para training
    COLAB_T4       = "colab_t4"
    COLAB_A100     = "colab_a100"
    CPU_ONLY       = "cpu_only"

@dataclass
class GPUInfo:
    os_name: str
    arch: str
    has_nvidia: bool
    has_amd: bool
    has_apple_metal: bool
    nvidia_devices: list[str]
    recommended_backend: GPUBackend
    summary: str

def detect_gpu() -> GPUInfo:
    os_name = platform.system()
    arch = platform.machine()

    has_nvidia = _check_nvidia_smi()
    has_amd = _check_amd_rocm()
    has_apple_metal = (os_name == "Darwin" and arch == "arm64")

    nvidia_devices = _list_nvidia_devices() if has_nvidia else []

    # ─── POLÍTICA DE RECOMENDAÇÃO ─────────────────────────────────────
    if os_name == "Darwin":
        # macOS: Apple Metal não roda training JAX/TF eficientemente
        backend = GPUBackend.COLAB_T4
        summary = "🍎 macOS detectado → Use Colab Pro+ para treino GPU."

    elif os_name == "Linux" and has_nvidia:
        # Linux com NVIDIA: usar local
        if any("A100" in d or "H100" in d for d in nvidia_devices):
            backend = GPUBackend.LOCAL_NVIDIA
            summary = f"🚀 GPU NVIDIA premium local: {nvidia_devices[0]}"
        else:
            backend = GPUBackend.LOCAL_NVIDIA
            summary = f"✅ GPU NVIDIA local: {nvidia_devices[0]}"

    elif os_name == "Linux" and has_amd:
        # AMD ROCm em beta — Colab safer
        backend = GPUBackend.COLAB_T4
        summary = "⚠️ AMD ROCm em beta. Use Colab para reproduzibilidade."

    elif os_name == "Linux":
        backend = GPUBackend.COLAB_T4
        summary = "💻 Linux sem GPU detectada → Colab Pro+."

    elif os_name == "Windows":
        backend = GPUBackend.COLAB_T4
        summary = "⚠️ Windows: use WSL2 + drivers NVIDIA → ou Colab Pro+."

    else:
        backend = GPUBackend.CPU_ONLY
        summary = f"❓ OS não suportado: {os_name}"

    return GPUInfo(
        os_name=os_name,
        arch=arch,
        has_nvidia=has_nvidia,
        has_amd=has_amd,
        has_apple_metal=has_apple_metal,
        nvidia_devices=nvidia_devices,
        recommended_backend=backend,
        summary=summary,
    )


def _check_nvidia_smi() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _list_nvidia_devices() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=2, text=True
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _check_amd_rocm() -> bool:
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# CLI entry point
if __name__ == "__main__":
    info = detect_gpu()
    print(info.summary)
    print(f"  OS:        {info.os_name} ({info.arch})")
    print(f"  Backend:   {info.recommended_backend.value}")
    if info.nvidia_devices:
        print(f"  NVIDIA:    {info.nvidia_devices}")
```

### §48.3 — Integração com Workflows

```python
# Em todo workflow que precisa GPU:
from geosteering_ai.utils.gpu_detect import detect_gpu, GPUBackend

gpu = detect_gpu()
if gpu.recommended_backend in (GPUBackend.LOCAL_NVIDIA,):
    # Roda localmente
    run_training_local()
elif gpu.recommended_backend in (GPUBackend.COLAB_T4, GPUBackend.COLAB_A100):
    # Gera notebook Colab + alerta usuário
    generate_colab_notebook()
    print(f"Use o notebook gerado em Colab: notebooks/auto_generated_{date}.ipynb")
else:
    # Fallback CPU
    print("⚠️ Sem GPU. Treino CPU será ~50× mais lento.")
    run_training_cpu()
```

### §48.4 — Cron de Verificação (Hardware Mudou?)

```yaml
# Em sprint-orchestrator.md
hooks:
  - on_session_start:
      run: python tools/gpu_detect.py --print-summary
      cache_for: 24h     # cache por 24h, então redetect

# Se Daniel mudar de Macbook para Linux server, próxima sessão recomenda
# automaticamente local ao invés de Colab.
```


---

## §49 — Backends de Visualização

### §49.1 — Avaliação dos 5 Backends Solicitados

| Backend | Categoria Forte | Forte em | Fraco em | Adoção |
|:--------|:---------------|:---------|:---------|:------:|
| **matplotlib** | Estática 2D + publicação | Plots científicos, papers, PDFs | Real-time, 3D, web | **✅ ADOTAR** |
| **PyQtGraph** | Real-time GUI | ≥60 FPS plots, GUI nativo | Estética bonita, web | **✅ ADOTAR** |
| **Plotly** | Interativos web/notebook | Hover, zoom, dashboards web | Real-time GUI, perf >100k pts | **✅ ADOTAR** |
| **Vispy** | OpenGL acelerado | Real-time 3D scientific | Comunidade pequena, manutenção lenta | **❌ REJEITAR** |
| **PyVista** | 3D modelagem volumétrica | Reservoirs 3D, well paths, mesh | Plots 2D, real-time | **✅ ADOTAR** |

### §49.2 — Justificativa para Rejeitar Vispy

| Critério | Vispy | Substituto adequado |
|:---------|:------|:--------------------|
| Real-time 3D | ✅ excelente | PyVista (com plotter.update() ≥30 FPS) |
| Real-time 2D | ✅ via OpenGL | **PyQtGraph** já cobre |
| Comunidade | ~140 contribs ativos | matplotlib 1500+, plotly 350+, pyvista 200+ |
| Documentação | Boa mas escassa de exemplos geofísicos | matplotlib/plotly têm centenas |
| Aprendizado curva | Alta (OpenGL) | matplotlib/PyQtGraph baixa-média |
| Releases frequência | ~1 release/ano | matplotlib mensal, plotly mensal |

**Decisão**: Vispy seria 5º backend redundante. **Rejeitado** para reduzir
superfície de manutenção. Caso futuro requeira OpenGL custom (ex: 4D
seismic real-time), reavaliar.

### §49.3 — Matriz de Uso por Contexto

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MATRIZ DE USO — VISUALIZAÇÃO                                          │
│                                                                          │
│  CONTEXTO                  BACKEND        JUSTIFICATIVA                 │
│  ──────────────────────────────────────────────────────────────────    │
│  Papers científicos        matplotlib     Padrão de publicação        │
│  Reports MD/PDF            matplotlib     Render estático otimizado    │
│  Plots de treino DL        matplotlib     EDA, loss curves, confusion  │
│  Notebooks tutoriais       Plotly         Hover + interatividade       │
│  Dashboards web            Plotly + Dash  Server-side rendering        │
│  Diagnóstico inversão      Plotly         Compare predicted vs true    │
│  GUI Studio real-time      PyQtGraph      ≥60 FPS, baixa CPU          │
│  Geosteering panel         PyQtGraph      Stream LWD em tempo real     │
│  Reservatório 3D           PyVista        Volume + mesh + well paths   │
│  Well trajectory 3D        PyVista        Splines + cylinders          │
│  Comparativo multi-poço    PyVista        Side-by-side 3D              │
│  Export para Petrel        PyVista        VTK + ParaView format        │
└──────────────────────────────────────────────────────────────────────────┘
```

### §49.4 — Estrutura de Módulo `geosteering_ai/visualization/`

```
geosteering_ai/visualization/
├── __init__.py
├── _backends/
│   ├── __init__.py
│   ├── matplotlib_backend.py   # plots estáticos, papers
│   ├── plotly_backend.py        # interativos web, notebooks
│   ├── pyqtgraph_backend.py     # GUI real-time (Studio only)
│   └── pyvista_backend.py       # 3D models, reservoirs, wells
├── plots/
│   ├── eda.py                    # exploratory data analysis (matplotlib)
│   ├── training.py               # loss curves, confusion matrix (matplotlib)
│   ├── inversion.py              # predicted vs true (plotly default + matplotlib export)
│   ├── geosteering.py            # real-time panel (pyqtgraph + plotly summary)
│   ├── reservoirs.py             # 3D reservoirs (pyvista)
│   └── wells.py                  # well trajectory + curves (pyvista + matplotlib)
└── utils/
    ├── color_schemes.py          # paletas científicas (cmocean, scientific)
    ├── unit_formatting.py
    └── export.py                 # PNG/SVG/PDF/HTML/VTK
```

### §49.5 — Exemplo: Inversion Diagnostic Plot

```python
# geosteering_ai/visualization/plots/inversion.py
"""
Plot diagnóstico de inversão: predicted vs true.
Backend default: Plotly (interativo). Export estático: matplotlib.
"""
def plot_inversion_diagnostic(
    predicted: np.ndarray,
    true: np.ndarray,
    z_obs: np.ndarray,
    backend: Literal["plotly", "matplotlib"] = "plotly",
    save_to: Optional[Path] = None,
) -> Any:
    """Plota predicted vs true para uma inversão.

    Args:
        predicted: shape (N, 2) — [rho_h, rho_v]
        true: shape (N, 2) — ground truth
        z_obs: shape (N,) — profundidades
        backend: "plotly" (interativo) ou "matplotlib" (estático)
        save_to: caminho para salvar (formato deduzido pela extensão)

    Returns:
        Figura objeto (Plotly Figure ou matplotlib.figure.Figure)
    """
    if backend == "plotly":
        from ._backends.plotly_backend import plot_inversion_plotly
        fig = plot_inversion_plotly(predicted, true, z_obs)
    else:
        from ._backends.matplotlib_backend import plot_inversion_mpl
        fig = plot_inversion_mpl(predicted, true, z_obs)

    if save_to:
        ...
    return fig
```

### §49.6 — GUI Real-time (PyQtGraph) — Exemplo Geosteering

```python
# studio/gui/geosteering_panel.py (Studio only — PR comercial)
"""
Painel real-time de geosteering com PyQtGraph.
Stream de medidas LWD e curva de inversão em tempo real ≥10 Hz.
"""

import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore

class GeosteeringPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QtWidgets.QVBoxLayout())

        # Layout: 3 painéis empilhados (curve, inversion, decision)
        self.curve_plot = pg.PlotWidget(title="Hxx, Hzz LWD")
        self.curve_plot.setLabel("left", "H (A/m)")
        self.curve_plot.setLabel("bottom", "Profundidade (m)")
        self.layout().addWidget(self.curve_plot)

        self.inv_plot = pg.PlotWidget(title="ρh, ρv (inversão tempo real)")
        self.inv_plot.setLogMode(y=True)
        self.layout().addWidget(self.inv_plot)

        self.decision_label = QtWidgets.QLabel("Decisão: aguardando...")
        self.layout().addWidget(self.decision_label)

        # Timer para atualizar (alvo 10 Hz)
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(100)  # 100ms = 10 Hz

    def _update(self):
        # Lê novas amostras LWD via WITSML
        new_samples = self._read_witsml()
        # Atualiza plots
        self.curve_plot.plot(new_samples.depth, new_samples.Hxx, pen='b', clear=True)
        ...
```

### §49.7 — Recomendações Específicas

| Decisão | Justificativa |
|:--------|:--------------|
| **matplotlib é o backend padrão** para plots em scripts/notebooks/relatórios | Maturidade, padrão de publicação |
| **Plotly default em notebooks** quando interatividade ajuda | Tutoriais, exploração |
| **PyQtGraph apenas em Studio** (Trilha B comercial) | Real-time only justifica complexidade |
| **PyVista para 3D em ambas trilhas** | Reservoir/well 3D é diferenciador |
| **Vispy NÃO adotado** | Redundante com PyVista; comunidade menor |
| **Pip-lib inclui matplotlib + plotly + pyvista** (Trilha A) | Cobre 95% dos casos científicos |
| **Studio adiciona PyQtGraph** (Trilha B) | Real-time GUI exclusivo |

---

## §50 — Arquitetura de Software Escolhida: Hexagonal + DDD

### §50.1 — Padrões Avaliados

| Padrão | Adequação para Geosteering AI | Veredito |
|:-------|:----------------------------|:---------|
| **Hexagonal (Ports & Adapters)** | Domínio físico (simuladores, models) puro; adapters in/out (CLI, GUI, HDF5, WITSML) | **✅ ESCOLHIDO** |
| **Clean Architecture (Uncle Bob)** | Similar a Hexagonal mas mais formal em camadas; overkill para Python | ⚠️ Inspiração |
| **DDD (Domain-Driven Design)** | Domínio físico bem-definido (poço, camada, ρh/ρv); language ubíqua | **✅ Combina com Hexagonal** |
| **Layered (3-tier)** | Simples mas mistura lógica de domínio com infraestrutura | ❌ Inferior |
| **Event-Driven** | Real-time geosteering tem eventos LWD; mas excessivo p/ todo o sistema | ⚠️ Apenas Studio |
| **Plugin-based (microkernel)** | Adapters de WITSML/Petrel/SLB encaixam, mas não estrutura tudo | ⚠️ Apenas adapters |
| **CQRS** | Inversão (query) vs decisão (command) tem benefícios | ❌ Overhead injustificado |
| **MVC/MVVM** | Adequado a GUI Studio mas não ao engine | ⚠️ Apenas Studio GUI |

### §50.2 — Estrutura Hexagonal Aplicada

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ARQUITETURA HEXAGONAL — GEOSTEERING AI                                ║
│                                                                          ║
│   ┌───────────────────────────────────────────────────────────────────┐ ║
│   │  ADAPTERS DRIVING (input — quem chama)                           │ ║
│   │  ├─ CLI (Typer)                                                  │ ║
│   │  ├─ Python API (import geosteering_ai)                          │ ║
│   │  ├─ Notebook (Jupyter/Colab)                                     │ ║
│   │  ├─ Studio GUI (PyQt6)                                           │ ║
│   │  └─ REST API (FastAPI — futuro v3)                               │ ║
│   └───────────────────────────────────────────────────────────────────┘ ║
│                            ↓ ↑ (calls / responses)                     ║
│   ┌───────────────────────────────────────────────────────────────────┐ ║
│   │  APPLICATION LAYER (casos de uso)                                │ ║
│   │  ├─ TrainModelUseCase                                            │ ║
│   │  ├─ RunInversionUseCase                                          │ ║
│   │  ├─ SimulateForwardUseCase                                       │ ║
│   │  ├─ GeosteerRealtimeUseCase                                      │ ║
│   │  └─ ExportResultsUseCase                                          │ ║
│   └───────────────────────────────────────────────────────────────────┘ ║
│                            ↓ ↑                                         ║
│   ┌───────────────────────────────────────────────────────────────────┐ ║
│   │  DOMAIN LAYER (núcleo puro — sem deps externas)                 │ ║
│   │  ├─ Models (entidades): ResistivityProfile, LayerStack,         │ ║
│   │  │                       AntennaConfig, MeasurementTensor       │ ║
│   │  ├─ Services: ForwardSimulation (port), Inversion (port),       │ ║
│   │  │            GeosteringDecision (port)                          │ ║
│   │  ├─ ValueObjects: Resistivity, Frequency, Depth                 │ ║
│   │  └─ DomainEvents: BoundaryDetected, AngleAdjustmentNeeded      │ ║
│   └───────────────────────────────────────────────────────────────────┘ ║
│                            ↓ ↑ (port interfaces)                       ║
│   ┌───────────────────────────────────────────────────────────────────┐ ║
│   │  ADAPTERS DRIVEN (output — implementações)                      │ ║
│   │  ├─ Simulators: NumbaSimulator, JaxSimulator, FortranSimulator  │ ║
│   │  ├─ ML Models: TFKerasModel, JaxModel                           │ ║
│   │  ├─ Persistence: HDF5Repo, ParquetRepo, SQLiteRepo              │ ║
│   │  ├─ Streams: WITSMLClient, ModbusClient (Studio)                │ ║
│   │  ├─ Export: LASExporter, SEGYExporter, PetrelExporter           │ ║
│   │  └─ Telemetry: MLflowTracker, FileLogger                        │ ║
│   └───────────────────────────────────────────────────────────────────┘ ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §50.3 — Mapeamento para Estrutura Atual

| Camada Hex | Localização atual | Após refactor v3.0 |
|:-----------|:------------------|:-------------------|
| Domain (entidades) | espalhado em `models/`, `data/` | `geosteering_ai/domain/` (NOVO) |
| Application (use cases) | espalhado em training/, inference/ | `geosteering_ai/application/` (NOVO) |
| Adapters Driving | `cli/`, `simulation_manager.py` | `geosteering_ai/adapters/in/` |
| Adapters Driven | `simulation/_numba`, `_jax`, `Fortran_Gerador/` | `geosteering_ai/adapters/out/` |
| Ports (interfaces) | implícitos | `geosteering_ai/ports/` (NOVO) |

### §50.4 — Vantagens Concretas para Geosteering

```
┌──────────────────────────────────────────────────────────────────────────┐
│  VANTAGENS DO HEXAGONAL APLICADAS A GEOSTEERING AI                    │
├──────────────────────────────────────────────────────────────────────────┤
│  1. Trocar simulador (Numba ↔ JAX ↔ Fortran) sem alterar use cases    │
│  2. Trocar persistência (HDF5 ↔ SQLite ↔ Parquet) sem alterar modelo  │
│  3. Adicionar nova interface (REST, gRPC) sem tocar Domain            │
│  4. Domain testável sem GPU, sem rede, sem GUI                        │
│  5. Studio importa o mesmo Domain do pip-lib (zero duplicação)        │
│  6. Adapters de Petrel/SLB são plug-ins separados                     │
│  7. Mock fácil de simulators para testes unitários (port = interface) │
└──────────────────────────────────────────────────────────────────────────┘
```

### §50.5 — Linguagem Ubíqua (DDD)

Vocabulário de domínio padronizado em **todo** código e docs:

| Termo Domínio | Significado | Não usar |
|:--------------|:-----------|:---------|
| `ResistivityProfile` | Sequência de camadas com (ρh, ρv, espessura) | "model", "stack" sozinhos |
| `MeasurementTensor` | Tensor H 9-componente (Hxx, Hxy, ..., Hzz) | "data", "signal" |
| `AntennaConfig` | Geometria T-R + frequência + dip | "config", "params" |
| `LWDFrame` | 1 leitura LWD com timestamp + posição + tensor | "sample" |
| `BoundaryEvent` | Detecção de boundary com confidence + location | "transition" |
| `WellPath` | Trajectória 3D (MD, TVD, AZI, INC) | "trajectory" sozinho |

### §50.6 — Refactor Plano (Sprints Pós-v3.0)

Refatoração para Hexagonal **NÃO acontece** em Fases 1-3 (impacto excessivo).
Acontece em **Fase 4+ Sprint 25-28**:

```
Sprint 25: Extrair Domain layer (criar geosteering_ai/domain/)
Sprint 26: Definir Ports (geosteering_ai/ports/)
Sprint 27: Refatorar Adapters in (CLI, API)
Sprint 28: Refatorar Adapters out (simulators, persistence)
```

Antes disso, **manter** estrutura atual flat. Hexagonal é alvo de longo prazo.


---

## §51 — Agente Geosteering Decision Real-time (Expansão Crítica)

🔄 **EXPANDE** §11 da Parte I (`geosteering-decision-agent` Sonnet, breve).

### §51.1 — Estado Atual e Lacuna

Na Parte I §11 o agente foi mencionado em 3 linhas: "decisão de geosteering
em tempo real, ângulo build-up/drop, integração LWD". **Lacuna**: faltava
especificação concreta de:

1. Como o agente recebe streams LWD?
2. Como combina inversão real-time com regras petrofísicas?
3. Como decide entre opções (build-up vs drop vs steer left/right)?
4. Como lida com incerteza?
5. Como interage com operador humano?

### §51.2 — Especificação Definitiva

```yaml
# .claude/agents/geosteering-decision-agent.md (EXPANDIDO §51)
---
name: geosteering-decision-agent
model: claude-sonnet-4-6
output_style: normal   # NÃO usar caveman — comunica com operador
description: |
  Agente especializado em decisões de geosteering em tempo real.
  Combina inversão DL + regras petrofísicas + estatística para
  recomendar ajustes de trajetória ao operador LWD.

inputs:
  - lwd_frames: Stream WITSML 1.4 (≥1 Hz)
  - inversion_state: ResistivityProfile atual + UQ (ensemble/INN)
  - well_plan: Trajetória planejada (TVD, INC, AZI ao longo MD)
  - reservoir_model: 3D inicial (carregado de Petrel)
  - operator_preferences: agressividade, tolerância UQ, constraints

outputs:
  - decision: GeosteringDecision objeto:
      action: enum {steer_up, steer_down, steer_left, steer_right, hold, alert}
      magnitude_deg: float (build-up rate em °/30m)
      confidence: float [0, 1]
      reasoning: string em linguagem natural
      alternatives: list[(action, score)]
      auditing: timestamp + agent_version + inputs_hash

skills_obrigatorias:
  - geosteering-decision (NOVO §51.5)
  - geosteering-petrophysics (princípios de classificação litológica)
  - geosteering-inversion-realtime (interpretação UQ)
  - geosteering-witsml (protocolo)

deps_codigo:
  - geosteering_ai/inference/realtime.py (existente)
  - geosteering_ai/geosteering/decision.py (NOVO §51.4)
  - geosteering_ai/geosteering/witsml.py (NOVO Studio)
  - geosteering_ai/inference/uncertainty.py (existente)

frequencia_invocacao:
  - real-time: a cada novo LWD frame (≥1 Hz)
  - planejamento: cada decisão crítica (≥10 m de progresso)
  - alerta: quando boundary detectada com confidence > 0.85
---
```

### §51.3 — Lógica de Decisão (Algoritmo)

```python
# geosteering_ai/geosteering/decision.py (NOVO)
"""
Lógica de decisão real-time. Usado pelo geosteering-decision-agent
mas é código puro (testável sem agente).
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np

class Action(Enum):
    STEER_UP    = "steer_up"
    STEER_DOWN  = "steer_down"
    STEER_LEFT  = "steer_left"
    STEER_RIGHT = "steer_right"
    HOLD        = "hold"
    ALERT       = "alert"   # operador deve decidir

@dataclass
class GeosteringDecision:
    action: Action
    magnitude_deg: float          # build/drop em °/30m
    confidence: float             # [0, 1]
    reasoning: str
    alternatives: list[tuple[Action, float]]
    boundary_distance: float      # m até boundary mais próxima
    boundary_confidence: float    # [0, 1]


def decide(
    inversion_profile: "ResistivityProfile",
    inversion_uq: "UQEnsemble",       # MC dropout ou INN samples
    well_plan: "WellPath",
    current_position: float,
    operator_prefs: "OperatorPreferences",
) -> GeosteringDecision:
    """
    Algoritmo de decisão hierárquico.

    Hierarquia:
      1. ALERT — se UQ alto (>30% std) ou conflito com plano
      2. STEER (build/drop/left/right) — se boundary próxima detectada
      3. HOLD — se na zona alvo confiável
    """

    # ── 1. Detectar boundary próxima ─────────────────────────────────
    boundary = detect_nearest_boundary(inversion_profile, current_position)
    if boundary is None:
        return GeosteringDecision(
            action=Action.HOLD,
            magnitude_deg=0.0,
            confidence=0.95,
            reasoning="Zona homogênea, sem boundary próxima",
            alternatives=[],
            boundary_distance=float("inf"),
            boundary_confidence=0.0,
        )

    # ── 2. Avaliar incerteza ─────────────────────────────────────────
    uq_std = inversion_uq.compute_std()
    if np.max(uq_std / np.abs(inversion_profile.rho_h)) > operator_prefs.uq_tolerance:
        return GeosteringDecision(
            action=Action.ALERT,
            magnitude_deg=0.0,
            confidence=0.5,
            reasoning=f"Incerteza alta (std/μ = {uq_std.max():.2%}). Operador deve revisar.",
            alternatives=[(Action.HOLD, 0.4), (Action.STEER_UP, 0.3)],
            boundary_distance=boundary.distance_m,
            boundary_confidence=boundary.confidence,
        )

    # ── 3. Decidir direção baseado em boundary + plano ───────────────
    plan_action = compute_plan_action(well_plan, current_position)
    boundary_action = compute_boundary_action(boundary, inversion_profile)

    # Combinar (regras petrofísicas + plano)
    if plan_action == boundary_action:
        # Acordo: agressivo
        magnitude = operator_prefs.aggressive_buildup_deg
        confidence = 0.85
    else:
        # Conflito: priorizar boundary (campo geofísico observado)
        magnitude = operator_prefs.conservative_buildup_deg
        confidence = 0.65

    return GeosteringDecision(
        action=boundary_action,
        magnitude_deg=magnitude,
        confidence=confidence,
        reasoning=(
            f"Boundary detectada a {boundary.distance_m:.1f} m, "
            f"contraste ρ {boundary.contrast_log10:.1f} décadas. "
            f"Plano sugere {plan_action.value}; observação sugere "
            f"{boundary_action.value}. Decidindo {boundary_action.value}."
        ),
        alternatives=[(plan_action, 0.4)],
        boundary_distance=boundary.distance_m,
        boundary_confidence=boundary.confidence,
    )


def detect_nearest_boundary(profile, position) -> "BoundaryEvent | None":
    """Detecta boundary mais próxima com base em gradiente de log10(ρ)."""
    # Implementação: derivada de log10(rho_h) ao longo z_obs
    # Boundary = max gradient absoluto > threshold (ex: 0.5 décadas/m)
    ...


def compute_plan_action(well_plan, position) -> Action:
    """Lê well_plan e retorna ação esperada na posição atual."""
    ...


def compute_boundary_action(boundary, profile) -> Action:
    """Determina ação baseada na geometria do boundary observado."""
    ...
```

### §51.4 — Operador Humano: Modelo de Interação

```
┌──────────────────────────────────────────────────────────────────────────┐
│  INTERAÇÃO AGENTE ↔ OPERADOR LWD                                       ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  Modo "ASSISTED" (default):                                             ║
│    • Agente RECOMENDA ações                                              ║
│    • Operador APROVA/REJEITA                                             ║
│    • Decisão final = operador (audit trail registra ambos)              ║
│                                                                          ║
│  Modo "AUTOPILOT" (somente em zonas de baixo risco):                    ║
│    • Agente executa ações se confidence > 0.9                           ║
│    • Operador supervisiona; pode intervir a qualquer momento            ║
│    • Auto-fallback para ASSISTED se confidence cai                     ║
│                                                                          ║
│  Modo "ADVISORY" (mais conservador):                                    ║
│    • Agente apenas reporta análise                                       ║
│    • Operador comanda 100%                                               ║
│    • Útil para operadores em treinamento                                 ║
│                                                                          ║
│  Sempre:                                                                 ║
│    • Audit trail completo (cada decisão + inputs hash)                  ║
│    • Fallback manual em ≤2 cliques                                       ║
│    • Alertas sonoros para ALERT action                                  ║
│                                                                          ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §51.5 — Skill `geosteering-decision.md`

```markdown
# .claude/skills/geosteering-decision.md (NOVO)

## Conhecimento de Domínio: Geosteering

### Princípios Petrofísicos
- Reservoir alvo geralmente tem ρh > 30 Ω·m em arenitos
- Cap rock (selo) tem ρh < 5 Ω·m em folhelhos
- Anisotropia ρv/ρh > 5 indica laminação fina (folhelho-areia)

### Build-up / Drop Rates Típicos
- Conservador: ≤1.5°/30m
- Padrão: 2-3°/30m
- Agressivo: 4-6°/30m (pode quebrar BHA)

### Detecção de Boundary
- Threshold: gradient log10(ρh) > 0.5 décadas/m
- Confidence baseado em SNR de Hzz (axial)
- False positive comum: borehole eccentricity → checar Hxx

### Regras de Segurança
- Nunca aprovar steer >6°/30m sem confirmação operador
- Nunca em modo AUTOPILOT se UQ std > 25%
- Sempre alertar se WITSML stream perde >5 segundos

### Casos de Uso
1. Reservoir entry: build-up agressivo até dentro do alvo
2. Reservoir following: hold + ajustes finos
3. Reservoir exit warning: alertar operador
4. Sweet spot navigation: steer para zona de máximo ρh

### Padrões de Comunicação ao Operador
- Linguagem clara, sem jargão de ML
- Sempre incluir reasoning + alternatives
- Mostrar confidence numericamente E visualmente
- Em modo ASSISTED, esperar confirmação antes de ação
```

---

## §52 — Requisitos de Produção e GUI Profissional

### §52.1 — Requisitos NÃO-Funcionais (Production-Grade)

| Categoria | Requisito | Padrão / Norma | Implementação |
|:----------|:----------|:---------------|:--------------|
| **Disponibilidade** | Uptime ≥99.5% durante operação de poço | SLA cliente | Healthcheck + auto-restart |
| **Latência** | Inversão + decisão ≤100ms (real-time) | API LWD | JIT cache, GPU local |
| **Throughput** | ≥10 frames LWD/s sustentado | WITSML 1.4 | Buffer + backpressure |
| **Reliability** | MTBF >1000h em campo | Best practice oil&gas | Redundância + watchdog |
| **Audit trail** | 100% decisões registradas (timestamp+hash) | SOX, IFRS-like | SQLite append-only + signing |
| **Determinismo** | Reproduzir poço passado bit-exato | Auditoria | Pin de modelo + config + dataset |
| **Authentication** | Operador identificado em cada decisão | LDAP/SAML enterprise | PyQt6 login + token |
| **Encryption** | Dados de poço em trânsito + em repouso | API/SPE / company policy | TLS 1.3 + AES-256 disk |
| **Data Privacy** | Dados de poço NÃO saem sem aprovação | GDPR equivalent | Air-gapped option |
| **Backup** | RPO ≤1h, RTO ≤4h | Industry standard | Snapshot por sprint + offsite |
| **Recovery** | Restart sem perda em <2min | Operations SLA | State snapshots em SQLite |

### §52.2 — GUI Profissional: Padrões de Indústria

| Sistema de referência | Aprendizado |
|:---------------------|:-----------|
| **Schlumberger Petrel** | Workflow-based (clicks ordenados), Project tree, Multi-window |
| **Halliburton Landmark DecisionSpace** | Real-time geosteering panel, sweet spot indicator |
| **Geosoft Oasis montaj** | Plugin marketplace, scripting embedded |
| **Roxar RMS / Emerson** | 3D reservoir + well integration |
| **Geoscience Australia ASEG** | Open source, GUI minimalista |

### §52.3 — Requisitos da GUI (Geosteering AI Studio)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  GUI PROFISSIONAL — REQUISITOS POR PAINEL                              ║
│                                                                          ║
│  PAINEL 1 — REAL-TIME LWD STREAM                                        ║
│    • Atualização ≥10 Hz                                                  ║
│    • Hxx, Hxy, ..., Hzz (9 traços coloridos)                            ║
│    • Indicador depth + RT clock                                          ║
│    • Zoom temporal + lock to bottom                                      ║
│    • Export para CSV em qualquer momento                                ║
│                                                                          ║
│  PAINEL 2 — INVERSÃO TEMPO REAL                                         ║
│    • ρh, ρv como log curves verticais                                   ║
│    • Ensemble UQ (band sombreado ±2σ)                                   ║
│    • Comparação predicted vs measured (residuals)                       ║
│    • Boundary detection markers                                          ║
│                                                                          ║
│  PAINEL 3 — DECISÃO AGENTE                                              ║
│    • Action recomendada (gigante, colorido)                              ║
│    • Confidence numérica + visual (gauge)                                ║
│    • Reasoning em linguagem natural                                      ║
│    • Alternatives + scores                                               ║
│    • Botão APROVAR / REJEITAR / MANUAL OVERRIDE                          ║
│                                                                          ║
│  PAINEL 4 — 3D RESERVOIR + WELL                                         ║
│    • PyVista renderer                                                    ║
│    • Well path real (linha azul) vs plan (linha verde tracejada)        ║
│    • Reservoir top/bottom como surfaces                                  ║
│    • Boundary markers como spheres                                       ║
│    • Camera presets (along well, plan view, side)                        ║
│                                                                          ║
│  PAINEL 5 — MULTI-POÇO (MULTI-WELL)                                     ║
│    • Lista de poços ativos (3-10 simultâneos)                            ║
│    • Comparativo lado-a-lado (split panes)                               ║
│    • Cross-correlations entre poços                                      ║
│                                                                          ║
│  PAINEL 6 — AUDIT TRAIL                                                  ║
│    • Lista de decisões (timestamp, action, operador, confidence)        ║
│    • Filtros por poço, operador, tipo                                   ║
│    • Export para PDF (compliance auditor)                                ║
│                                                                          ║
│  PAINEL 7 — CONFIG + STATUS                                              ║
│    • GPU/CPU usage                                                       ║
│    • Modelo carregado (versão + checksum)                                ║
│    • WITSML stream status                                                ║
│    • Latência decisão (rolling)                                          ║
│                                                                          ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §52.4 — Tela Principal (Mockup ASCII)

```
╔════════════════════════════════════════════════════════════════════════════╗
║ Geosteering AI Studio v1.0                                  [—][□][×]    ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Poço: SLB-OK-28  •  MD: 3,425.2 m  •  Status: 🟢 Active  •  Op: dleal     ║
╠════════════════════════════════════════════════════════════════════════════╣
║ ┌─ LWD Real-time ──────────────────┐  ┌─ Inversão (UQ ±2σ) ──────────┐   ║
║ │  Hxx ━━━━━━━━━━━━━━━━━━━━━━━━━━━ │  │ ρh ━━━━━━━━━━━━━━━━━━━━━━━ │   ║
║ │  Hzz ━━━━━━━━━━━━━━━━━━━━━━━━━━━ │  │ ρv ━━━━━━━━━━━━━━━━━━━━━━━ │   ║
║ │  [10 Hz] [autoscale] [export csv]│  │ [linear/log] [residuals]      │   ║
║ └──────────────────────────────────┘  └──────────────────────────────┘   ║
║                                                                          ║
║ ┌─ Decisão Agente ──────────────────────────────────────────────────────┐║
║ │   ⬆️  STEER UP   2.5°/30m   Confidence: 87% [▰▰▰▰▰▰▰▰▱▱]              │║
║ │   "Boundary 4.2m abaixo, contraste 1.3 décadas log ρ. Plano  ↑.       │║
║ │    Recomendo build-up moderado."                                       │║
║ │   [APROVAR]   [REJEITAR]   [VER ALTERNATIVAS]   [MANUAL]              │║
║ └──────────────────────────────────────────────────────────────────────┘║
║                                                                          ║
║ ┌─ 3D Reservoir + Well ────────────────┐ ┌─ Audit Trail ───────────────┐║
║ │  [PyVista 3D render]                  │ │ 14:32 STEER_UP +2.5° (87%)  │║
║ │  - Plan trajectory (green)            │ │ 14:30 HOLD (95%)             │║
║ │  - Real trajectory (blue)             │ │ 14:28 ALERT (UQ 35%)         │║
║ │  - Reservoir top (red surface)        │ │ 14:25 STEER_DOWN -1.5° (78%) │║
║ │  - Boundaries (spheres)               │ │ 14:22 HOLD (92%)             │║
║ │  [orbit] [along-well] [plan]         │ │ [...] [Export PDF]            │║
║ └──────────────────────────────────────┘ └─────────────────────────────┘║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
║ GPU: NVIDIA A6000 • Latency: 42ms • WITSML: 🟢 connected • v1.0.0      ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### §52.5 — Tecnologia de Empacotamento (Studio)

| Plataforma | Ferramenta primária | Backup | Output |
|:-----------|:-------------------|:-------|:-------|
| **macOS** | PyInstaller + py2app + create-dmg | briefcase | `.dmg` + `.app` |
| **Linux** | PyInstaller + AppImage / fpm | briefcase | `.AppImage` + `.deb` + `.rpm` |
| **Windows** | PyInstaller + Inno Setup | briefcase | `.msi` + `.exe` |

### §52.6 — Certificações e Compliance

| Certificação | Aplicabilidade | Esforço estimado |
|:-------------|:---------------|:----------------:|
| API/SPE para inversão EM | Recomendado para clientes oil&gas | 6-12 meses |
| ISO 9001 (Quality) | Para vender a majors | 12-18 meses |
| ISO 27001 (Security) | Se hostear dados clientes | 12-18 meses |
| GDPR / LGPD | Operações em UE/Brasil | 3-6 meses |
| Code signing (Apple, Microsoft) | Distribuição comercial | 1-2 meses |

### §52.7 — Operações (Devops & Monitoring)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  STACK DE OPERAÇÕES — STUDIO EM PRODUÇÃO                              ║
├──────────────────────────────────────────────────────────────────────────┤
│  Logging:        structlog + Sentry (errors)                            ║
│  Metrics:        Prometheus exporter                                     ║
│  Dashboards:     Grafana (uptime, latency, throughput)                  ║
│  Crash reports:  Sentry com source maps                                  ║
│  Update:         Sparkle (macOS), Squirrel (Win)                         ║
│  Telemetry:      Opt-in, anonymized; usa OTel                           ║
│  Support:        SLA por tier; ticketing Linear/Zendesk                 ║
└──────────────────────────────────────────────────────────────────────────┘
```


---

## §53 — Docling para Conversão PDF→Markdown

### §53.1 — Inventário Atual da Pasta `PDFs/`

```
Total: 31 arquivos · 291 MB
├─ Papers científicos (geofísica/inversão): ~12
│   • Constable_et_al_2016_Petrophysics.pdf (56 MB)
│   • Wang_2018_J._Geophys._Eng._15_2339.pdf (4 MB)
│   • JGR Solid Earth - 2021 - Zhang - Bayesian INN.pdf (3 MB)
│   • JGR Solid Earth - 2023 - Liu - Joint Inversion CCS.pdf (5 MB)
│   • Guoyu_et_al_2025_FG.pdf (3 MB)
│   • ggad217.pdf (3 MB) — geosteering
│   • Schlumberger_GeoSphere HD 1.0.pdf (2 MB)
│   • Benchmark_Look_Ahead_Models.pdf (530 KB)
│   • 2210.09060v4.pdf (2 MB)
│   • FormulaçãoTatuAnisoTIV.pdf (1.4 MB) — DOC INTERNA
│   • GeoSphereXTatu.pdf (74 KB) — DOC INTERNA
│   • Key_Specs_Survey_UH.pdf (190 KB)
│
├─ Livros técnicos (Fortran/Python/JAX): ~10
│   • Modern Fortran (Curcic 2020) — 12 MB
│   • Fortran 2018 With Parallel Programming (Ray) — 3.6 MB
│   • CUDA Fortran for Scientists — 4.9 MB
│   • Python High Performance (Lanaro 2017) — 5.9 MB
│   • Fast Python (Antao 2023) — 9.3 MB
│   • High Performance Python (Gorelick/Ozsvald 2025) — 10 MB
│   • Numerical Python (Johansson 2024) — 25 MB
│   • Numerical Recipes — 21 MB
│   • Dive Into Deep Learning — 12.7 MB
│   • deep-learning-jax.pdf (Petersen) — 41 MB
│
├─ Livros de Well Logging: ~3
│   • Well Logging for Earth Scientists (Ellis/Singer 2008) — 23 MB
│   • Principles and Applications of Well Logging — 18 MB
│   • Introduction to Programming with Fortran (Chivers/Sleightholme) — 5 MB
│
└─ Específicos: 1
    • OpenMP-4.0-Fortran.pdf — 731 KB (referência tatu.f08)
```

### §53.2 — Por que Docling?

| Critério | pdf2text (PyPDF2) | pdfminer | **Docling** | LlamaParse |
|:---------|:------------------|:---------|:----------:|:-----------|
| Texto puro | ✅ básico | ✅ | ✅ excelente | ✅ |
| Preserva tabelas | ❌ | ◐ | ✅ | ✅ |
| Preserva equações (LaTeX) | ❌ | ❌ | ✅ | ✅ |
| Preserva figuras | ❌ | ❌ | ◐ refs | ✅ |
| Preserva estrutura (headings) | ❌ | ◐ | ✅ | ✅ |
| Output em Markdown | ❌ (texto) | ❌ | ✅ nativo | ✅ |
| Custo | $0 (local) | $0 (local) | $0 (local, IBM) | Pago (API) |
| Velocidade | Rápido | Rápido | Médio (CPU) | Médio (cloud) |
| Privacidade | ✅ local | ✅ local | ✅ local | ❌ cloud |

**Decisão**: **Docling** combina qualidade superior + custo zero + privacidade
local (rodando offline). Distribuído por IBM como open source desde 2024.

### §53.3 — Pipeline `tools/pdf_to_md.py`

```python
# tools/pdf_to_md.py (NOVO)
"""
Converte PDFs/ em Markdown estruturado em docs/PDFs_md/.

Vantagens para os agentes:
  • grep-able (Claude Code Explore agent acha referências rápido)
  • Cabe em context window (chunk por seção)
  • Tabelas e equações preservadas (importante para papers físicos)
  • Versionável em git (auditoria de mudanças)

Uso:
  $ python tools/pdf_to_md.py --input PDFs/ --output docs/PDFs_md/
  $ python tools/pdf_to_md.py --input PDFs/Wang_2018.pdf --output docs/PDFs_md/Wang_2018.md
  $ python tools/pdf_to_md.py --check-changed  # apenas PDFs modificados
"""

import argparse
import hashlib
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import json

CACHE_FILE = ".cache/pdf_md_hashes.json"


def compute_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_cache() -> dict[str, str]:
    p = Path(CACHE_FILE)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_cache(cache: dict[str, str]) -> None:
    Path(CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(CACHE_FILE).write_text(json.dumps(cache, indent=2))


def convert_pdf(pdf_path: Path, output_dir: Path, converter: DocumentConverter) -> Path:
    """Converte 1 PDF para markdown e salva."""
    output_md = output_dir / (pdf_path.stem + ".md")

    print(f"→ Convertendo: {pdf_path.name}")
    result = converter.convert(pdf_path)
    md = result.document.export_to_markdown()

    # Frontmatter com metadados
    metadata = (
        f"---\n"
        f"source_pdf: {pdf_path.name}\n"
        f"source_size_bytes: {pdf_path.stat().st_size}\n"
        f"converted_at: {pdf_path.stat().st_mtime}\n"
        f"sha256: {compute_hash(pdf_path)}\n"
        f"---\n\n"
    )
    output_md.write_text(metadata + md)
    print(f"  ✅ Salvo: {output_md} ({len(md):,} chars)")
    return output_md


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown via Docling")
    parser.add_argument("--input", type=Path, default="PDFs/", help="Input dir or single PDF")
    parser.add_argument("--output", type=Path, default="docs/PDFs_md/", help="Output dir")
    parser.add_argument("--check-changed", action="store_true", help="Only convert PDFs with new SHA-256")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for scanned PDFs (slow)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Configurar pipeline
    pipeline_options = PdfPipelineOptions(
        do_ocr=args.ocr,
        do_table_structure=True,  # detecta tabelas com TableFormer
        table_structure_options={"do_cell_matching": True},
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: pipeline_options,
        }
    )

    # Cache para detectar mudanças
    cache = load_cache()

    # Listar PDFs
    if args.input.is_file():
        pdfs = [args.input]
    else:
        pdfs = sorted(args.input.rglob("*.pdf"))

    converted = 0
    skipped = 0
    for pdf in pdfs:
        h = compute_hash(pdf)
        key = str(pdf)
        if args.check_changed and cache.get(key) == h:
            print(f"⏭  Inalterado: {pdf.name}")
            skipped += 1
            continue
        try:
            convert_pdf(pdf, args.output, converter)
            cache[key] = h
            converted += 1
        except Exception as e:
            print(f"  ❌ Falha: {e}")

    save_cache(cache)
    print(f"\n=== {converted} convertidos, {skipped} inalterados ===")


if __name__ == "__main__":
    main()
```

### §53.4 — Output Esperado: `docs/PDFs_md/`

```
docs/PDFs_md/
├── Wang_2018_J._Geophys._Eng._15_2339.md          (~150 KB)
├── JGR Solid Earth - 2021 - Zhang.md               (~120 KB)
├── JGR Solid Earth - 2023 - Liu.md                 (~180 KB)
├── Constable_et_al_2016_Petrophysics.md            (~2 MB texto enorme)
├── Modern Fortran - Curcic 2020.md                 (~1.5 MB)
├── deep-learning-jax.md                             (~3 MB)
├── ...
├── _index.md                                        (índice gerado)
└── _hash_cache.json                                 (cache do conversor)
```

### §53.5 — Integração com Agentes Claude Code

Agentes ganham nova fonte de conhecimento estruturado e grep-able:

```yaml
# .claude/skills/geosteering-literature.md (NOVO)
---
description: |
  Conhecimento da literatura científica do projeto, derivada de PDFs
  convertidos via Docling em docs/PDFs_md/.
sources:
  - docs/PDFs_md/Wang_2018*.md
  - docs/PDFs_md/JGR Solid Earth*.md
  - docs/PDFs_md/Constable_et_al_2016*.md
  - docs/PDFs_md/Schlumberger_GeoSphere*.md
  - docs/PDFs_md/ggad217.md         # geosteering
  - docs/PDFs_md/Benchmark_Look_Ahead_Models.md
  - docs/PDFs_md/FormulaçãoTatuAnisoTIV.md   # interno

uso_typical: |
  Quando agente precisa fundamentar decisão científica:
    1. grep no docs/PDFs_md/ por keyword (ex: "anisotropy", "TIV", "INN")
    2. Ler chunk relevante (5-50 KB)
    3. Citar em código/docs com referência ao paper

  Exemplos:
    • "Wang (2018) demonstra que Hzz é 4× mais sensível a ρv que Hxx"
    • "Zhang (2021) usa INN para inversão Bayesiana com covariância"
    • "GeoSphere HD usa frequência 100kHz para dipolos near-bit"
---
```

### §53.6 — Workflow de Atualização

```
1. Daniel adiciona novo PDF em PDFs/
2. Hook .git/hooks/pre-commit detecta novo PDF
   (OPCIONAL: Daniel roda manualmente python tools/pdf_to_md.py --check-changed)
3. Docling converte para docs/PDFs_md/<nome>.md
4. Git tracking dos .md (não dos PDFs — gitignore PDFs/)
5. CI verifica que nenhum PDF não-convertido foi commitado
6. Wiki sync (§43.5) propaga docs/PDFs_md/ para Wiki
7. Upgrade-scout (§46) escaneia novos papers semanalmente
```

### §53.7 — Custo / Esforço da Conversão Inicial

| Item | Estimativa |
|:-----|:-----------|
| Tempo de conversão (31 PDFs em CPU 8C) | 20-40 min |
| Tempo se OCR ativo (scanned PDFs) | +60-90 min |
| Tamanho output total `docs/PDFs_md/` | ~50-100 MB |
| Esforço inicial Daniel (script + verificação) | 2-3 horas |
| Manutenção semanal | <15 min |

### §53.8 — Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|:------|:------------:|:----------|
| Docling falha em PDFs scaneados | Média | OCR opt-in via `--ocr` (Tesseract bundled) |
| Equações LaTeX corrompidas | Baixa | Manual review dos top-3 papers críticos |
| Output gigante (>50MB) inflar repo | Média | Apenas em main; gitignore docs/PDFs_md/_raw_*.md (chunks intermediários) |
| Encoding PT-BR (acentuação) | Baixa | Docling preserva UTF-8 nativamente |
| Privacidade (PDFs com dados sensíveis) | Baixa | Tudo local, sem API; auditar antes de Wiki sync |
| Versão Docling muda formato output | Média | Pin de versão (`docling==1.x.*`); upgrade-scout testa antes de bump |

### §53.9 — Decisão Arquitetural

**ADOTAR Docling** como pipeline padrão de processamento de PDFs.
Adicionar:
- `tools/pdf_to_md.py` (script novo)
- `.gitignore`: ignorar `docs/PDFs_md/_*.md` (intermediários), `*.pdf` (originals)
- `pyproject.toml`: adicionar `docling>=1.0` em `[project.optional-dependencies] tools`
- `.claude/skills/geosteering-literature.md` (skill novo)
- Hook `.git/hooks/pre-commit` (opcional): avisa se PDF novo sem MD correspondente

---

## §54 — Síntese da Parte III + Cronograma de Adoção

### §54.1 — Sumário das 15 Decisões Arquiteturais (Parte III)

```
╔══════════════════════════════════════════════════════════════════════════╗
║  PARTE III — DECISÕES APROVADAS APÓS APROFUNDAMENTO TÉCNICO            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  §39  Skill caveman ADOTADA opcional/condicional (agentes long-running) ║
║       • Plugin: claude plugin marketplace add JuliusBrussee/caveman     ║
║       • Config: .claude/output-styles/caveman-conditional.md            ║
║                                                                          ║
║  §40  Multi-agentes paralelos com 5 camadas defensivas                  ║
║       • parallelism_rules.py + lock files + worktrees + review + roll  ║
║                                                                          ║
║  §41  Quality Mesh: 7 camadas anti-bug                                   ║
║       • backup, static, physics, tests, known-bugs, review, watcher     ║
║                                                                          ║
║  §42  Distribuição em DUAS TRILHAS (modelo Geosoft)                     ║
║       • pip install geosteering-ai (lib + CLI Apache/MIT)               ║
║       • Geosteering AI Studio (GUI premium comercial)                   ║
║                                                                          ║
║  §43  GitHub auto-sync: docs→Pages, reports→Wiki, housekeeper agent     ║
║                                                                          ║
║  §44  Workflow W13 — JAX Sprint (NOVO; 12 → 13 workflows)               ║
║                                                                          ║
║  §45  Roteiro semana-a-semana 24 sprints (Fases 1-4); gates G1-G8       ║
║                                                                          ║
║  §46  18º agente: upgrade-scout (semanal); ALL agents propose upgrades  ║
║                                                                          ║
║  §47  Setup local cross-platform: tools/setup_environment.sh + uv       ║
║                                                                          ║
║  §48  GPU detection policy:                                              ║
║       macOS → Colab default                                              ║
║       Linux+NVIDIA → local default                                       ║
║       outros → Colab default                                             ║
║                                                                          ║
║  §49  Visualização: matplotlib + Plotly + PyQtGraph + PyVista (4)       ║
║       Vispy REJEITADO (redundante com PyVista)                           ║
║                                                                          ║
║  §50  Arquitetura escolhida: HEXAGONAL + DDD                             ║
║       Refactor diferido para Sprints 25-28 (pós-v3.0)                    ║
║                                                                          ║
║  §51  Geosteering decision agent expandido com algoritmo + skill        ║
║       3 modos: ASSISTED (default), AUTOPILOT, ADVISORY                  ║
║                                                                          ║
║  §52  Produção: SLA 99.5%, latência ≤100ms, audit trail SOX,            ║
║       7 painéis Studio, certificações roadmap                            ║
║                                                                          ║
║  §53  Docling para PDFs/ → docs/PDFs_md/ (grep-able pelos agentes)      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### §54.2 — Inventário Final: Arquivos a Criar / Modificar

**Criar (33 novos arquivos)**:

| Categoria | Arquivo | Seção |
|:----------|:--------|:-----:|
| Skills | `.claude/output-styles/caveman-conditional.md` | §39 |
| Skills | `.claude/skills/geosteering-decision.md` | §51 |
| Skills | `.claude/skills/geosteering-literature.md` | §53 |
| Skills | `.claude/skills/geosteering-simulator-jax.md` | §44 |
| Skills | `.claude/skills/geosteering-numba-jax-parity.md` | §44 |
| Skills | `.claude/skills/geosteering-colab-runner.md` | §44 |
| Skills | `.claude/skills/geosteering-bench-validator.md` | §44 |
| Agents | `.claude/agents/upgrade-scout.md` | §46 |
| Agents | `.claude/agents/repo-housekeeper.md` | §43 |
| Agents | `.claude/agents/jax-engineer.md` | §44 |
| Agents | `.claude/agents/colab-runner.md` | §44 |
| Agents | `.claude/agents/numba-jax-parity-validator.md` | §44 |
| Agents | `.claude/agents/bench-validator.md` | §44 |
| Hooks | `.claude/hooks/agent-acquire-lock.sh` | §40 |
| Hooks | `.claude/hooks/agent-release-lock.sh` | §40 |
| Hooks | `.claude/hooks/post-commit-push.sh` | §43 |
| Hooks | `.claude/hooks/check-anti-patterns.sh` (atualização) | §41 |
| Code | `.claude/parallelism_rules.py` | §40 |
| Code | `.claude/telemetry/parallelism_dashboard.py` | §40 |
| Code | `.claude/active_agents.json` | §40 |
| Code | `tools/file_watcher_daemon.py` | §41 |
| Code | `tools/setup_environment.sh` | §47 |
| Code | `tools/version_bump.py` | §43 |
| Code | `tools/pdf_to_md.py` | §53 |
| Code | `geosteering_ai/utils/gpu_detect.py` (atualização) | §48 |
| Code | `geosteering_ai/cli/setup_command.py` | §47 |
| Code | `geosteering_ai/geosteering/decision.py` | §51 |
| Code | `geosteering_ai/visualization/_backends/plotly_backend.py` | §49 |
| Code | `geosteering_ai/visualization/_backends/pyqtgraph_backend.py` | §49 |
| Code | `geosteering_ai/visualization/_backends/pyvista_backend.py` | §49 |
| Docs | `docs/known_bugs.md` | §41 |
| Docs | `docs/upgrade_proposals/` (diretório + template) | §46 |
| Docs | `docs/PDFs_md/` (diretório + index) | §53 |
| CI | `.github/workflows/docs-deploy.yml` | §43 |
| CI | `.github/workflows/wiki-sync.yml` | §43 |
| Config | `.claude/anti-patterns.txt` | §41 |
| Notebook | `notebooks/jax_sprint_template.ipynb` | §44 |

**Modificar (10 arquivos)**:

| Arquivo | Mudança | Seção |
|:--------|:--------|:-----:|
| `.claude/settings.json` | adicionar 4 hooks novos (acquire-lock, release-lock, watcher init, post-commit-push) | §40, §41, §43 |
| `pyproject.toml` | atualizar dependências, adicionar `[project.optional-dependencies] gui, mlops, tools` | §47, §49 |
| `CLAUDE.md` | atualizar línea 16 com Parte III decisões | todos |
| `docs/ROADMAP.md` | adicionar 24 semanas Fases 1-4 | §45 |
| `docs/CHANGELOG.md` | append entry para v3.0 (Parte III decisions) | todos |
| `geosteering_ai/__init__.py` | exportar geosteering_ai.geosteering, geosteering_ai.visualization | §49, §51 |
| `geosteering_ai/cli/__init__.py` | adicionar setup_command | §47 |
| `MEMORY.md` | nova seção "Decisões Arquiteturais Parte III" | todos |
| `geosteering_ai/utils/gpu_detect.py` | refinar política macOS=Colab | §48 |
| `.gitignore` | adicionar .claude/locks/, .claude/active_agents.json, .backups/, docs/PDFs_md/_*.md | §40, §41, §53 |

### §54.3 — Cronograma de Adoção (Faseado, 4 Semanas Iniciais)

```
SEMANA 1 (2026-05-04 a 2026-05-10) — INFRA + QUALITY MESH
─────────────────────────────────────────────────────────────────
Dia 1: §38 backup-pre-edit + §41 quality mesh L1-L4 (hooks core)
Dia 2: §41 known_bugs.md + §41 anti-patterns.txt
Dia 3: §40 parallelism rules + lock files
Dia 4: §40 dashboard + active_agents.json
Dia 5: §47 setup_environment.sh cross-platform
Sábado: §48 gpu_detect.py refinado
Domingo: revisão Daniel

SEMANA 2 (2026-05-11 a 2026-05-17) — DOCLING + GITHUB SYNC
─────────────────────────────────────────────────────────────────
Dia 1-2: §53 Docling tools/pdf_to_md.py + first batch conversão
Dia 3: §53 docs/PDFs_md/ + index + skill geosteering-literature
Dia 4: §43 GitHub Actions docs-deploy + wiki-sync
Dia 5: §43 repo-housekeeper agent + post-commit-push hook
Sábado: §39 caveman skill instalada + .claude/output-styles/
Domingo: revisão Daniel

SEMANA 3 (2026-05-18 a 2026-05-24) — UPGRADE SCOUT + JAX SPRINT
─────────────────────────────────────────────────────────────────
Dia 1: §46 upgrade-scout agent + primeiro relatório semanal
Dia 2: §44 W13 JAX Sprint workflow + agentes (jax-engineer etc)
Dia 3: §44 notebook template Colab
Dia 4-5: W13.1 vmap real T4 (sprint piloto)
Sábado: §51 geosteering-decision-agent expansão
Domingo: revisão Daniel

SEMANA 4 (2026-05-25 a 2026-05-31) — VISUALIZATION + STUDIO PREP
─────────────────────────────────────────────────────────────────
Dia 1-2: §49 visualization/_backends/ (matplotlib + plotly + pyvista)
Dia 3: §49 plotly notebook examples
Dia 4: §52 estudo de mockups Studio + briefcase setup
Dia 5: §50 design refactor Hexagonal (sketch, não implementação)
Sábado: §45 ROADMAP final 24 semanas + cronograma
Domingo: revisão Daniel + retrospectiva mensal
```

### §54.4 — Métricas de Sucesso da Parte III (Após Mês 1)

| Métrica | Meta | Como Medir |
|:--------|:----:|:-----------|
| Hooks ativos (backup, lock, anti-pattern) | 100% | logs de hooks executados |
| Bugs evitados pelo Quality Mesh | ≥5 | comparativo com KB-013/018/019 sprints anteriores |
| Custo mensal Claude Max | ≤$200 | dashboard Claude Console |
| Throughput Cenário B | ≥600k mod/h | bench v2.22 |
| PDFs convertidos em docs/PDFs_md/ | ≥25/31 | ls docs/PDFs_md/*.md |
| Upgrade proposals geradas | ≥4 (1/semana) | docs/upgrade_proposals/ |
| GitHub Pages deploy automatizado | 100% sucesso | CI logs |
| Wiki sync semanal | 1× | git log wiki |
| Sprint W13.1 (JAX vmap T4) | Completo | bench T4 ≥200k mod/h |
| Acentuação PT-BR em docs novos | 100% | grep -P "[ãâéêíóôú]" |

### §54.5 — Próximos Passos Imediatos

```
1. [Daniel] Revisar Parte III completa
2. [Daniel] Validar backup em .backups/2026-05-03/ está acessível
3. [Daniel + Sonnet] Implementar §38 backup-pre-edit + §41 anti-patterns
   ANTES de qualquer outra coisa (camada base do Quality Mesh)
4. [Daniel + Sonnet] Implementar §40 parallelism rules + locks
5. [Daniel + Sonnet] Rodar Docling §53 em batch overnight (PDFs → MD)
6. [Daniel + Opus] Sprint v2.22 FLAT prange Numba (continuação plano)
7. [Daniel + Sonnet] Configurar GitHub Actions (docs deploy + wiki sync)
8. [Daniel] Avaliar instalação caveman skill após Daniel ler §39 testes
9. [Daniel + Opus] Esboçar Studio v0.1 alpha (planning, não código ainda)
10. [Daniel] Definir clientes piloto para Studio v0.5 BETA (Q1 2027)
```

### §54.6 — Atualização do Total do Documento

```
PARTE I  (3.552 linhas) — visão geral, 17 seções, 8+3 agentes
PARTE II (~2.000 linhas) — aprofundamentos críticos 13 questões
PARTE III (~2.300 linhas) — refinamentos técnicos 15 questões + 4 novos
                            componentes (1 agente, 1 workflow, Quality Mesh,
                            Hexagonal)

TOTAL APROXIMADO: ~7.850 linhas combinadas
TOTAL DE AGENTES: 18 (era 12 na Parte I)
TOTAL DE WORKFLOWS: 13 (era 12)
TOTAL DE SKILLS: 16 (Parte I 9 + Parte III 7 novas)
TOTAL DE HOOKS: 12 (Parte I 8 + Parte III 4 novos)
```

---

**Parte III adicionada em 2026-05-03 com Claude Opus 4.7 (1M contexto).**

**Status: DOCUMENTO BASE OFICIAL — Parte I (visão geral) + Parte II
(aprofundamentos críticos) + Parte III (refinamentos técnicos profundos)
formam a referência canônica completa para construção da arquitetura
do Geosteering AI v3.0.**

**Backup do MD pré-Parte III preservado em**:
`.backups/2026-05-03/arquitetura_multiagente_aprofundamento_151154.pre-partIII.bak`
(274K, validado).

**Próxima revisão programada**: após implementação completa da Semana 1
(Quality Mesh + Multi-agent locks). Revisões pontuais a cada gate de fase.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   GEOSTEERING AI v3.0 — ARQUITETURA MULTI-AGENTE COMPLETA              ║
║                                                                          ║
║   18 agentes · 13 workflows · 16 skills · 12 hooks · 4 MCP servers     ║
║                                                                          ║
║   Sob restrições rígidas:                                                ║
║     • Paridade Fortran <1e-12 (físico)                                  ║
║     • PT-BR acentuado (documental)                                      ║
║     • TensorFlow exclusivo (framework)                                  ║
║     • Quality Mesh 7-camadas (qualidade)                                ║
║                                                                          ║
║   Para entregar:                                                         ║
║     • pip install geosteering-ai (open source, scientists/devs)         ║
║     • Geosteering AI Studio (commercial GUI, modelo Geosoft)           ║
║                                                                          ║
║   Em 24 semanas distribuídas em 4 fases.                                ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

═══════════════════════════════════════════════════════════════════════════
PARTE IV — ESPECIALIZAÇÃO PROFUNDA E EXTENSÕES (2026-05-04 · OPUS 4.7)
═══════════════════════════════════════════════════════════════════════════

> **Escopo desta Parte IV**: 13 refinamentos profundos focados em
> (a) otimização do simulador JAX por cenário, (b) integração simulador-DL,
> (c) extensão para 2D/2.5D/3D FEM, (d) novos agentes especializados
> (noise, FEM 2D/2.5D/3D, Studio, Simulation Manager, scientific-report,
> dev-tutor), (e) Effort levels por tarefa, (f) revisão de orçamento de
> tokens. Atualiza decisões da Parte I-III quando há contradição
> (sempre marcado com `🔄 REVISA §N`).

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  ÍNDICE — PARTE IV                                                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  §55  Simulador JAX Otimizado para 8 Cenários (C1-C8) — GPU             ║
║  §56  Modos de Uso: Standalone vs DL On-The-Fly                          ║
║  §57  Estratégias DL+Simulator: 14 Padrões Avançados                     ║
║  §58  Simuladores 2D/2.5D/3D FEM — Stack Híbrido                         ║
║  §59  Agente Noise Engineer (19º agente)                                 ║
║  §60  Simulation Manager como Terceiro Aplicativo                         ║
║  §61  Visualization Backends do Simulation Manager                       ║
║  §62  Agentes Studio + Simulation Manager (20º e 21º)                    ║
║  §63  Skills e Hooks Novos da Parte IV                                   ║
║  §64  Arquitetura como Tutor do Desenvolvedor (dev-tutor-agent)          ║
║  §65  Agente Scientific Report LaTeX (22º agente)                        ║
║  §66  Effort Configuration por Modelo/Tarefa                             ║
║  §67  Síntese da Parte IV + Token/Context Optimization Final             ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## §55 — Simulador JAX Otimizado para 8 Cenários (C1-C8)

🔄 **EXPANDE** §44 da Parte III (W13 JAX Sprint).

### §55.1 — Definição dos 8 Cenários

Mesma taxonomia do simulador Numba JIT (CPU):

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TAXONOMIA DOS 8 CENÁRIOS DO SIMULADOR — APLICÁVEL A NUMBA JIT E JAX   ║
├──────────────────────────────────────────────────────────────────────────┤
│  Cenário  Frequências  Ângulos     T-R Pares    Caso típico              ║
│  ───────  ──────────   ──────      ────────     ─────────                ║
│  C1       1            1           1            Validação Fortran 1f1a1TR║
│  C2       muitas (m)   1           1            Sweep frequência         ║
│  C3       1            muitas      1            Sweep ângulo (dip variado)║
│  C4       1            1           muitas       Multi-TR (compensação)   ║
│  C5       1            muitas      muitas       Multi-ângulo + multi-TR  ║
│  C6       muitas       muitas      1            Ângulo × frequência      ║
│  C7       muitas       1           muitas       Frequência × multi-TR    ║
│  C8       muitas       muitas      muitas       Pleno (training DL)      ║
└──────────────────────────────────────────────────────────────────────────┘
```

**Dimensões típicas**:
- `nf` ∈ {1, 2, 4, 8, 16}  (multi-frequency)
- `nAng` ∈ {1, 7, 13, 30}   (multi-dip)
- `nTR` ∈ {1, 2, 4, 8}      (multi-TR)
- `n_pos` ∈ {30, 100, 300, 600, 1200, 6000}  (posições ao longo do poço)
- `batch_size` (treino DL) ∈ {16, 32, 64, 128, 256}

### §55.2 — Estratégia JAX por Cenário (GPU T4/A100)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ESTRATÉGIA JAX POR CENÁRIO                                             ║
├──────────────────────────────────────────────────────────────────────────┤
│  Cenário  Estratégia JAX               Primitivos JAX usados             ║
│  ───────  ────────────────────────     ──────────────────────────       ║
│  C1       JIT puro (single shot)        @jit                              ║
│  C2       vmap em frequências           jax.vmap(axis=freq)              ║
│  C3       vmap em ângulos               jax.vmap(axis=angle)             ║
│  C4       fori_loop em TR pares         jax.lax.fori_loop                ║
│  C5       vmap aninhado angle×TR        vmap(vmap)                       ║
│  C6       vmap aninhado angle×freq      vmap(vmap)                       ║
│  C7       vmap aninhado freq×TR         vmap(vmap) ou fori interno       ║
│  C8       vmap triplo OU pmap+vmap      vmap(vmap(vmap)) ou pmap+vmap²   ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §55.3 — Detalhamento Técnico Cenário-a-Cenário

#### C1 — Single freq, single angle, single TR (validação)

```python
@partial(jax.jit, static_argnums=(0,))
def _simulate_c1_jit(cfg_static, rho_h, rho_v, esp, freq, angle, tr_offset, positions):
    """Cenário C1: 1f, 1a, 1TR. Apenas vmap em positions."""
    # Single forward pass — sem vmap externo
    return simulate_pure(rho_h, rho_v, esp, freq, angle, tr_offset, positions)
```

**Performance esperada T4**: ~50ms para 600 posições.
**Uso**: paridade Fortran <1e-12, debug.

#### C2 — Multi-freq, single angle, single TR

```python
@partial(jax.jit, static_argnums=(0,))
def _simulate_c2_jit(cfg_static, rho_h, rho_v, esp, freqs, angle, tr_offset, positions):
    """Cenário C2: nf, 1a, 1TR. vmap sobre freqs."""
    return jax.vmap(
        lambda f: simulate_pure(rho_h, rho_v, esp, f, angle, tr_offset, positions),
        in_axes=0
    )(freqs)
    # Output shape: (nf, n_pos, 9_components)
```

**Otimização**: filtros Hankel (kong/werthmuller/anderson) compartilhados entre freqs.
Pré-computar uma vez no `cfg_static`. Hankel quadrature é o gargalo — **fusion**
via `@jax.jit` evita recomputação.

**Performance esperada T4**: 8 freqs × 600 pos = ~150ms (vs 8×50ms=400ms sequencial).

#### C3 — Single freq, multi-angle, single TR

```python
@partial(jax.jit, static_argnums=(0,))
def _simulate_c3_jit(cfg_static, rho_h, rho_v, esp, freq, angles, tr_offset, positions):
    """Cenário C3: 1f, mAng, 1TR. vmap sobre angles."""
    return jax.vmap(
        lambda a: simulate_pure(rho_h, rho_v, esp, freq, a, tr_offset, positions),
        in_axes=0
    )(angles)
    # Output shape: (n_angles, n_pos, 9)
```

**Otimização**: matriz de rotação `RtHR` recalculada por ângulo, mas filter weights
constantes. `jax.lax.scan` alternativa para angles sequencial se memória limitar.

**Performance esperada T4**: 13 angles × 600 pos = ~280ms.

#### C4 — Single freq, single angle, multi-TR

```python
@partial(jax.jit, static_argnums=(0,))
def _simulate_c4_jit(cfg_static, rho_h, rho_v, esp, freq, angle, tr_offsets, positions):
    """Cenário C4: 1f, 1a, mTR. fori_loop sobre TR pares."""
    nTR = tr_offsets.shape[0]

    def body(i, acc):
        result_i = simulate_pure(rho_h, rho_v, esp, freq, angle, tr_offsets[i], positions)
        return acc.at[i].set(result_i)

    init = jnp.zeros((nTR, positions.shape[0], 9), dtype=jnp.complex128)
    return jax.lax.fori_loop(0, nTR, body, init)
```

**Decisão**: `fori_loop` (sequencial mas memory-efficient) em vez de `vmap`
porque `nTR` geralmente é 2-4 (pequeno) e cada TR usa diferentes posições efetivas
(não rebatch eficiente).

**Performance esperada T4**: 4 TR × 600 pos = ~200ms.

#### C5 — Single freq, multi-angle, multi-TR

```python
@partial(jax.jit, static_argnums=(0,))
def _simulate_c5_jit(cfg_static, rho_h, rho_v, esp, freq, angles, tr_offsets, positions):
    """Cenário C5: 1f, mAng, mTR. vmap angle + fori_loop TR."""
    def per_angle(a):
        def body(i, acc):
            res = simulate_pure(rho_h, rho_v, esp, freq, a, tr_offsets[i], positions)
            return acc.at[i].set(res)
        init = jnp.zeros((tr_offsets.shape[0], positions.shape[0], 9), dtype=jnp.complex128)
        return jax.lax.fori_loop(0, tr_offsets.shape[0], body, init)

    return jax.vmap(per_angle)(angles)
    # Output shape: (n_angles, nTR, n_pos, 9)
```

**Otimização**: angle outer (vmap), TR inner (fori_loop). Reduz peak memory
sem sacrificar paralelismo significativo.

**Performance esperada T4**: 13 angles × 4 TR × 600 pos = ~700ms.

#### C6 — Multi-freq, multi-angle, single TR

```python
@partial(jax.jit, static_argnums=(0,))
def _simulate_c6_jit(cfg_static, rho_h, rho_v, esp, freqs, angles, tr_offset, positions):
    """Cenário C6: mf, mAng, 1TR. vmap aninhado angle×freq."""
    def per_angle(a):
        return jax.vmap(
            lambda f: simulate_pure(rho_h, rho_v, esp, f, a, tr_offset, positions)
        )(freqs)

    return jax.vmap(per_angle)(angles)
    # Output shape: (n_angles, nf, n_pos, 9)
```

**Decisão**: `vmap(vmap)` sobre o par (freq, angle). XLA fusion une os dois
loops em kernel único na GPU.

**Performance esperada T4**: 8 freqs × 13 angles × 600 pos = ~1.5s.

#### C7 — Multi-freq, single angle, multi-TR

```python
@partial(jax.jit, static_argnums=(0,))
def _simulate_c7_jit(cfg_static, rho_h, rho_v, esp, freqs, angle, tr_offsets, positions):
    """Cenário C7: mf, 1a, mTR. vmap freq + fori_loop TR."""
    def per_freq(f):
        def body(i, acc):
            res = simulate_pure(rho_h, rho_v, esp, f, angle, tr_offsets[i], positions)
            return acc.at[i].set(res)
        init = jnp.zeros((tr_offsets.shape[0], positions.shape[0], 9), dtype=jnp.complex128)
        return jax.lax.fori_loop(0, tr_offsets.shape[0], body, init)

    return jax.vmap(per_freq)(freqs)
    # Output shape: (nf, nTR, n_pos, 9)
```

**Performance esperada T4**: 8 freqs × 4 TR × 600 pos = ~600ms.

#### C8 — Multi-freq, multi-angle, multi-TR (treino DL pleno)

```python
@partial(jax.jit, static_argnums=(0,))
def _simulate_c8_jit(cfg_static, rho_h, rho_v, esp, freqs, angles, tr_offsets, positions):
    """Cenário C8: mf, mAng, mTR. Triple vmap ou pmap + double vmap."""

    # Variante A — single GPU: triple vmap aninhado
    def per_angle(a):
        def per_freq(f):
            def body(i, acc):
                res = simulate_pure(rho_h, rho_v, esp, f, a, tr_offsets[i], positions)
                return acc.at[i].set(res)
            init = jnp.zeros((tr_offsets.shape[0], positions.shape[0], 9), dtype=jnp.complex128)
            return jax.lax.fori_loop(0, tr_offsets.shape[0], body, init)
        return jax.vmap(per_freq)(freqs)
    return jax.vmap(per_angle)(angles)
    # Output shape: (n_angles, nf, nTR, n_pos, 9)


# Variante B — multi-GPU (A100 × 4 ou 8): pmap angles + vmap rest
def _simulate_c8_pmap(cfg_static, rho_h, rho_v, esp, freqs, angles, tr_offsets, positions):
    """Distribui angles entre GPUs, vmap freq×TR dentro de cada GPU."""
    angles_sharded = angles.reshape(jax.device_count(), -1)
    return jax.pmap(
        lambda angles_chunk: _simulate_c8_jit(cfg_static, rho_h, rho_v, esp, freqs, angles_chunk, tr_offsets, positions)
    )(angles_sharded)
```

**Decisão de variante**:
- 1 GPU (T4) → Variante A com `vmap(vmap(fori_loop))`
- ≥2 GPUs (A100 × 4) → Variante B com `pmap` outer + `vmap` inner
- Memória peak T4 → fallback `fori_loop` para freqs também se OOM

**Performance esperada**:
- T4 single: 8 freqs × 13 angles × 4 TR × 600 pos = ~6s
- A100 single: ~2.5s
- A100 × 4 com pmap: ~700ms

### §55.4 — Otimizações Cross-Cenário

```
┌──────────────────────────────────────────────────────────────────────────┐
│  OTIMIZAÇÕES TRANSVERSAIS (aplicáveis a todos cenários)                ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  O1 — Filter weights cached (Hankel)                                    ║
│       Pré-computar weights kong/werthmuller/anderson uma vez            ║
│       Passar como `static_argnum` ou em `cfg_static`                    ║
│       Benefício: -30-40% tempo em cenários multi-freq                  ║
│                                                                          ║
│  O2 — Decoupling factors pré-computados                                 ║
│       ACp/ACx/ACz dependem só de L (spacing), pré-computar              ║
│       Não recalcular dentro de vmap                                      ║
│                                                                          ║
│  O3 — Mempool tuning (XLA)                                              ║
│       XLA_PYTHON_CLIENT_MEM_FRACTION=0.85                               ║
│       XLA_PYTHON_CLIENT_PREALLOCATE=true (production)                  ║
│       Reduz fragmentação em batches grandes                             ║
│                                                                          ║
│  O4 — Mixed precision FP32 (training surrogate)                         ║
│       OPCIONAL — não usar em validação Fortran                          ║
│       2× speedup com BF16 em A100                                        ║
│       Trade-off: paridade <1e-7 (não <1e-12)                            ║
│                                                                          ║
│  O5 — Sharding em batch DL (training)                                   ║
│       jax.sharding para distribuir batch entre GPUs                     ║
│       Aplicável quando batch_size ≥ device_count × 16                   ║
│                                                                          ║
│  O6 — Async dispatch                                                    ║
│       jax.block_until_ready apenas em validação                         ║
│       Permite overlap CPU prep + GPU compute                            ║
│                                                                          ║
│  O7 — Persistent JIT cache                                              ║
│       jax.config.update('jax_compilation_cache_dir', '~/.cache/jax')    ║
│       Evita recompilação entre sessões                                  ║
│                                                                          ║
│  O8 — Static argnums otimizado                                          ║
│       cfg_static (filter weights, n_pos, etc) como argnum static       ║
│       Evita recompilação por chamada                                    ║
│                                                                          ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §55.5 — Dispatcher de Cenário no Forward Pure

```python
# geosteering_ai/simulation/_jax/forward_pure.py (atualização §55)

def simulate_jax(
    cfg: SimulationConfig,
    rho_h: jnp.ndarray,
    rho_v: jnp.ndarray,
    esp: jnp.ndarray,
    freqs: jnp.ndarray,
    angles: jnp.ndarray,
    tr_offsets: jnp.ndarray,
    positions: jnp.ndarray,
) -> jnp.ndarray:
    """
    Dispatcher que escolhe a estratégia JAX otimizada por cenário.

    Detecta cenário automaticamente via shape dos inputs:
      C1: nf=1, nAng=1, nTR=1
      C2: nf>1, nAng=1, nTR=1
      ...
      C8: nf>1, nAng>1, nTR>1
    """
    nf = freqs.shape[0]
    nAng = angles.shape[0]
    nTR = tr_offsets.shape[0]

    cfg_static = build_static_context(cfg)  # filtros + decoupling pré-calc

    if nf == 1 and nAng == 1 and nTR == 1:
        return _simulate_c1_jit(cfg_static, rho_h, rho_v, esp, freqs[0], angles[0], tr_offsets[0], positions)
    elif nf > 1 and nAng == 1 and nTR == 1:
        return _simulate_c2_jit(cfg_static, rho_h, rho_v, esp, freqs, angles[0], tr_offsets[0], positions)
    elif nf == 1 and nAng > 1 and nTR == 1:
        return _simulate_c3_jit(cfg_static, rho_h, rho_v, esp, freqs[0], angles, tr_offsets[0], positions)
    elif nf == 1 and nAng == 1 and nTR > 1:
        return _simulate_c4_jit(cfg_static, rho_h, rho_v, esp, freqs[0], angles[0], tr_offsets, positions)
    elif nf == 1 and nAng > 1 and nTR > 1:
        return _simulate_c5_jit(cfg_static, rho_h, rho_v, esp, freqs[0], angles, tr_offsets, positions)
    elif nf > 1 and nAng > 1 and nTR == 1:
        return _simulate_c6_jit(cfg_static, rho_h, rho_v, esp, freqs, angles, tr_offsets[0], positions)
    elif nf > 1 and nAng == 1 and nTR > 1:
        return _simulate_c7_jit(cfg_static, rho_h, rho_v, esp, freqs, angles[0], tr_offsets, positions)
    else:  # C8
        if jax.device_count() >= 2 and nAng >= jax.device_count():
            return _simulate_c8_pmap(cfg_static, rho_h, rho_v, esp, freqs, angles, tr_offsets, positions)
        return _simulate_c8_jit(cfg_static, rho_h, rho_v, esp, freqs, angles, tr_offsets, positions)
```

### §55.6 — Métricas de Performance Esperadas (Targets)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TARGETS DE PERFORMANCE JAX POR CENÁRIO                                ║
├──────────────────────────────────────────────────────────────────────────┤
│  Cenário   T4 (free)    A100 (Pro+)    A100×4 (cluster)   vs Numba     ║
│  ───────   ──────       ────────       ────────────       ──────       ║
│  C1        50 ms        20 ms          n/a                ≈ Numba 4w   ║
│  C2        150 ms       60 ms          n/a                3-5×         ║
│  C3        280 ms       110 ms         n/a                4-6×         ║
│  C4        200 ms       80 ms          n/a                3-4×         ║
│  C5        700 ms       280 ms         100 ms             5-8×         ║
│  C6        1500 ms      600 ms         200 ms             8-12×        ║
│  C7        600 ms       240 ms         85 ms              5-7×         ║
│  C8        6000 ms      2500 ms        700 ms             10-20×       ║
└──────────────────────────────────────────────────────────────────────────┘
```

**Observação**: vantagem JAX cresce com complexidade (C1 ≈ Numba; C8 muito superior).
Em C1-C2, Numba CPU pode até superar T4 por baixa latência (sem PCIe overhead).

### §55.7 — Mudanças no Código

**Modificar**:
- `geosteering_ai/simulation/_jax/forward_pure.py` — dispatcher por cenário
- `geosteering_ai/simulation/_jax/multi_forward.py` — chamar dispatcher
- `geosteering_ai/simulation/__init__.py` — expor `simulate_multi_jax_optimized`

**Criar**:
- `geosteering_ai/simulation/_jax/scenarios.py` — funções `_simulate_c1` a `_simulate_c8`
- `tests/test_simulation_jax_scenarios.py` — paridade C1-C8 vs Numba
- `benchmarks/bench_jax_scenarios.py` — métrica por cenário em T4 + A100


---

## §56 — Modos de Uso: Standalone vs DL On-The-Fly

### §56.1 — Dois Modos de Uso do Mesmo Simulador

Tanto Numba JIT quanto JAX devem servir **dois modos** de uso, com semânticas
ligeiramente distintas mas API unificada:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MODOS DE USO DOS SIMULADORES                                          ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  MODO A — STANDALONE (uso humano via SM/CLI/API)                        ║
│  ───────────────────────────────────────────────                        ║
│    • Caso de uso: pesquisador roda 1 simulação para análise             ║
│    • Caso de uso: SM gera batch de modelos para visualização            ║
│    • Caso de uso: validação Fortran <1e-12                              ║
│    • Caso de uso: benchmark CLI                                          ║
│    • Características:                                                    ║
│        - 1 chamada por execução, latência importante (<100ms)           ║
│        - Tipos float64 (precisão máxima)                                ║
│        - Output: NumPy array (numba) ou jnp.array (jax) → np.array     ║
│        - Sem gradient flow (não diferenciável)                          ║
│        - Cache de filtros: importa muito                                ║
│                                                                          ║
│  MODO B — DL ON-THE-FLY (em training loop / inference loop)             ║
│  ───────────────────────────────────────────────────                    ║
│    • Caso de uso: gerar batch sintético DURANTE epoch de treino         ║
│    • Caso de uso: PINN com forward simulator no loss                    ║
│    • Caso de uso: surrogate-corrected forward em inferência            ║
│    • Caso de uso: differentiable inversion (loss.backward via simulator)║
│    • Características:                                                    ║
│        - Throughput importa, não latência por chamada                   ║
│        - Tipos float32 ok (treinamento), float64 só validação           ║
│        - Output: tf.Tensor (TF) ou jnp.Array (JAX), gradient-aware     ║
│        - Diferenciável (autodiff necessário)                            ║
│        - Batch dimension: shape `(batch, n_pos, 9)`                     ║
│        - Sem PCIe round-trip CPU↔GPU dentro do training loop            ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §56.2 — API Unificada com Modo Implícito

```python
# geosteering_ai/simulation/forward.py (atualização §56)

from typing import Literal, overload

@overload
def simulate(cfg: SimulationConfig, *, mode: Literal["standalone"] = ...) -> SimulationResult: ...

@overload
def simulate(cfg: SimulationConfig, *, mode: Literal["training"], rho_batch: tf.Tensor) -> tf.Tensor: ...

def simulate(
    cfg: SimulationConfig,
    *,
    mode: Literal["standalone", "training"] = "standalone",
    rho_batch: Optional[Any] = None,
) -> Any:
    """
    Despacha simulação no modo correto.

    Modo "standalone":
      • Single forward pass, NumPy output, máxima precisão
      • Backend: numba (default) ou jax (cfg.backend="jax")

    Modo "training":
      • Batch forward pass, gradient-aware
      • Backend: jax (sempre, requerido para autodiff)
      • Input: rho_batch shape (B, n_layers, 2)  [ρh, ρv]
      • Output: tensor shape (B, n_pos, 9)
    """
    if mode == "standalone":
        return _simulate_standalone(cfg)
    elif mode == "training":
        if rho_batch is None:
            raise ValueError("training mode requires rho_batch")
        return _simulate_in_training(cfg, rho_batch)
```

### §56.3 — Implementação Mode "training" (Differentiable Forward)

```python
# geosteering_ai/simulation/_jax/forward_diff.py (NOVO)
"""
Forward simulator diferenciável para uso em training loops PINN/surrogate.
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(0,))
def simulate_in_training(cfg_static, rho_batch, esp_batch, positions):
    """
    Forward diferenciável para uso em training loop.

    Args:
        cfg_static: ForwardPureContext com filtros pré-computados
        rho_batch:  shape (B, n_layers, 2)  — [ρh, ρv]
        esp_batch:  shape (B, n_layers)
        positions:  shape (n_pos,)  — fixo, não-batch

    Returns:
        H_tensor: shape (B, n_pos, 9), float32 ou complex64
                  gradient-aware (∂H/∂ρh, ∂H/∂ρv via autodiff)
    """

    def per_sample(rho_i, esp_i):
        # rho_i shape: (n_layers, 2)
        # esp_i shape: (n_layers,)
        return simulate_pure(
            rho_h=rho_i[:, 0],
            rho_v=rho_i[:, 1],
            esp=esp_i,
            freq=cfg_static.freq,
            angle=cfg_static.angle,
            tr_offset=cfg_static.tr_offset,
            positions=positions,
        )

    return jax.vmap(per_sample)(rho_batch, esp_batch)


# Wrapper TensorFlow-compatible (para uso em tf.keras training)
def simulate_tf_compatible(rho_batch_tf, esp_batch_tf, positions, cfg):
    """
    Bridge TF→JAX→TF para uso em camada Keras custom.

    NOTA: cruza fronteira TF/JAX uma vez por batch (overhead < ganho).
    """
    rho_jax = jnp.asarray(rho_batch_tf.numpy())
    esp_jax = jnp.asarray(esp_batch_tf.numpy())
    positions_jax = jnp.asarray(positions)

    H_jax = simulate_in_training(cfg.static_context, rho_jax, esp_jax, positions_jax)

    # Converter de volta para TF (preserva shape)
    return tf.convert_to_tensor(np.asarray(H_jax), dtype=tf.float32)
```

### §56.4 — Camada Keras Custom para Uso DL On-The-Fly

```python
# geosteering_ai/simulation/keras_layer.py (NOVO)
"""
Camada Keras que envolve o simulator JAX para uso em modelos PINN/surrogate.
"""

import tensorflow as tf
from geosteering_ai.simulation._jax.forward_diff import simulate_tf_compatible


class ForwardSimulationLayer(tf.keras.layers.Layer):
    """
    Camada que roda forward simulation dentro do training loop.

    Uso:
        from geosteering_ai.simulation.keras_layer import ForwardSimulationLayer

        rho_pred = model_inversion(measurements)  # NN inverte
        H_reconstructed = ForwardSimulationLayer(cfg)(rho_pred, esp)
        consistency_loss = tf.reduce_mean(tf.square(measurements - H_reconstructed))
    """

    def __init__(self, cfg: SimulationConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        # Pré-compilar JIT
        self._jitted = simulate_in_training

    def call(self, inputs):
        rho_batch, esp_batch = inputs
        positions = tf.constant(self.cfg.positions_m)
        return tf.py_function(
            func=lambda r, e, p: simulate_tf_compatible(r, e, p, self.cfg),
            inp=[rho_batch, esp_batch, positions],
            Tout=tf.float32,
        )
```

### §56.5 — Casos de Uso Concretos

```python
# Caso 1 — Standalone (Simulation Manager)
from geosteering_ai.simulation import simulate, SimulationConfig
cfg = SimulationConfig(...)
result = simulate(cfg, mode="standalone")

# Caso 2 — Training PINN (usa simulator no loss)
from geosteering_ai.simulation.keras_layer import ForwardSimulationLayer
forward_sim = ForwardSimulationLayer(cfg)

class PINNModel(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.inverter = build_resnet18(input_shape=(None, 5))
        self.forward_sim = ForwardSimulationLayer(cfg)

    def call(self, measurements):
        rho_pred = self.inverter(measurements)
        H_reconstructed = self.forward_sim((rho_pred, esp_known))
        return rho_pred, H_reconstructed

# Loss: MSE inversion + consistency physical
def pinn_loss(y_true, y_pred):
    rho_true, H_true = y_true
    rho_pred, H_reconstructed = y_pred
    return tf.reduce_mean(tf.square(rho_pred - rho_true)) + \
           0.1 * tf.reduce_mean(tf.square(H_reconstructed - H_true))

# Caso 3 — Surrogate generation (gerar dados durante epoch)
@tf.function
def generate_batch_on_the_fly(batch_size: int, cfg: SimulationConfig):
    rho_batch = sample_random_profiles(batch_size, n_layers=5)  # generator
    esp_batch = sample_random_thicknesses(batch_size, n_layers=5)
    H_batch = simulate(cfg, mode="training", rho_batch=rho_batch)
    return H_batch, rho_batch
```

### §56.6 — Roteamento de Backend por Modo

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ROTEAMENTO BACKEND × MODO                                              ║
├──────────────────────────────────────────────────────────────────────────┤
│  Modo            CPU disponível?  GPU disponível?   Backend escolhido   ║
│  ──────────      ────────────     ──────────────    ────────────────    ║
│  Standalone      Sim              Sim                Numba (latência)   ║
│  Standalone      Sim              Não                Numba              ║
│  Standalone      Não              Sim                JAX (mas 50ms+)    ║
│                                                                          ║
│  Training        n/a              Sim                JAX (autodiff)     ║
│  Training        n/a              Não                JAX CPU (lento)    ║
│                                                                          ║
│  Inferência DL   n/a              Sim                Modelo treinado   ║
│  (loop)                                              + JAX backbone     ║
└──────────────────────────────────────────────────────────────────────────┘
```

**Regra prática**:
- **Standalone**: prefere Numba (CPU sem PCIe overhead). Exceção: dados muito grandes
  (n_pos > 5000) onde GPU JAX começa a vencer pela paralelização.
- **Training**: SEMPRE JAX (autodiff é requerido).

### §56.7 — Implicações de Arquitetura

| Aspecto | Standalone | DL On-The-Fly |
|:--------|:-----------|:--------------|
| Latência por chamada | ≤100ms | n/a (throughput) |
| Throughput | n/a (1 sim) | ≥1000 sims/min |
| Precisão | float64 (validação) | float32 ok |
| Diferenciável | NÃO | SIM (autodiff) |
| Backend | numba (default) ou jax | jax (sempre) |
| Output | NumPy / SimulationResult | tf.Tensor / jnp.Array |
| GPU | opcional | recomendado |
| Tested via | tests/test_simulation_*.py | tests/test_pinn_consistency.py |


---

## §57 — Estratégias DL+Simulator: 14 Padrões Avançados

### §57.1 — Catálogo Completo (14 Estratégias)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ESTRATÉGIAS DE INTEGRAÇÃO SIMULADOR ↔ DEEP LEARNING                  ║
├──────────────────────────────────────────────────────────────────────────┤
│  #   Estratégia                  Aplic. GS-AI  Maturidade  Esforço      ║
│  ──  ──────────────────────────   ────────────  ──────────  ───────     ║
│  1   Differentiable simulators    ALTA          Madura      Baixo *     ║
│  2   Neural Operators (FNO)       MÉDIA-ALTA    Madura      Alto        ║
│  3   Active learning              ALTA          Madura      Baixo-Médio ║
│  4   Curriculum geológico         ALTA          Madura      Baixo *     ║
│  5   Sim-to-real DA (DANN)        CRÍTICA       Madura      Médio-Alto  ║
│  6   Adversarial training         MÉDIA         Madura      Médio       ║
│  7   SBI / NPE                    ALTA          Madura      Médio       ║
│  8   Surrogate-accelerated MCMC   ALTA          Madura      Médio-Alto  ║
│  9   Diffusion priors             ALTA          Madura      Alto        ║
│  10  Multi-fidelity               ALTA          Madura      Médio       ║
│  11  Self-supervised consistency  ALTA          Madura      BAIXO ★    ║
│  12  RL geosteering               ALTA-STRAT    Madura      Muito Alto  ║
│  13  Hybrid DL + classical inv    ALTA          Madura      Médio       ║
│  14  Physics-guided regularizat.  ALTA          Madura      MUITO BAIXO★║
│                                                                          ║
│  * = parcialmente implementado no projeto                              ║
│  ★ = quick win imediato                                                 ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §57.2 — Detalhamento Estratégia-a-Estratégia

#### #1 — Differentiable Simulators (PARCIALMENTE FEITO)

**Status**: você já tem JAX backend `simulation/_jax/` + Jacobiano F10 Fortran.
**O que ADICIONAR**:
- Camada Keras custom `ForwardSimulationLayer` (§56.4)
- Loss term `||measurements - simulator(rho_predicted)||²` em todos os modelos
- Permite **end-to-end inversion** sem dataset paired

**Referências**: JAX-FDFD, Sambridge 2022, Devito autodiff.

#### #2 — Neural Operators (FNO/DeepONet/GNOT)

**Status**: SurrogateNet TCN/ModernTCN é a versão "vanilla". FNO é evolução.
**Aplicabilidade**: substituir simulator por surrogate FNO ~100-1000× mais rápido.
**Trade-off**: FNO requer batch grande para treinar; precisão <1e-3 (não <1e-12).

```python
# Exemplo conceitual — FNO substituindo simulate_multi
from neuraloperator import FNO  # ou Modulus (NVIDIA)

class FNOSurrogate(tf.keras.Model):
    """Substitui simulate_multi por FNO para inferência ~1000× mais rápida.
    Precisão <1e-3 (vs <1e-12 do simulator exato)."""

    def __init__(self, n_modes_freq=32, n_modes_pos=64, hidden_channels=64):
        super().__init__()
        self.fno = FNO(
            in_channels=3,    # rho_h, rho_v, esp
            out_channels=18,  # 9 components × (real+imag)
            n_modes=(n_modes_freq, n_modes_pos),
            hidden_channels=hidden_channels,
        )
```

**Referências**: Li 2021 (FNO), Lu 2021 (DeepONet), Modulus (NVIDIA).

#### #3 — Active Learning + Simulator

**Aplicabilidade**: dataset multi-dip caro (horas de simulação CPU). Active learning
seleciona quais (rho_h, rho_v, dip, freq) vale a pena simular.

**Implementação**:
1. Treinar modelo inicial com ~1k modelos randômicos
2. Para cada candidato (10k-100k), computar **BALD score** (Bayesian disagreement)
3. Simular apenas top-1k mais informativos
4. Retreinar; iterar 5-10 vezes
5. Resultado: dataset 5-10× menor, mesma performance

```python
# Exemplo conceitual
from modAL.acquisition import BALD

def select_to_simulate(candidates, model_ensemble, n_select=1000):
    """Seleciona modelos a simular via BALD score."""
    bald_scores = compute_bald(model_ensemble, candidates)
    return candidates[np.argsort(bald_scores)[-n_select:]]
```

**Referências**: Houlsby 2011, Kirsch 2019 (BatchBALD).

#### #4 — Curriculum Geológico (PARCIALMENTE FEITO)

**Status**: você já tem curriculum 3-phase em **noise**. Falta em **complexidade geológica**.

**Extensão proposta**:
```
Fase 1 (epochs 1-10):    n_layers ∈ [3, 5],   dip ∈ [0°, 5°]
Fase 2 (epochs 11-30):   n_layers ∈ [3, 8],   dip ∈ [0°, 30°]
Fase 3 (epochs 31-60):   n_layers ∈ [3, 12],  dip ∈ [0°, 60°]
Fase 4 (epochs 61-100):  n_layers ∈ [3, 20],  dip ∈ [0°, 90°], anisotropia até ρv/ρh=10
```

**Implementação**: classe `CurriculumGeologicalCallback` em `training/callbacks.py`.

#### #5 — Sim-to-Real Domain Adaptation (CRÍTICA)

**Status**: NÃO implementado. **GAP MAIS CRÍTICO** para deploy comercial.

**Razão**: dados simulados ≠ LWD reais (eccentricity, mudpit conditions, sondagem real).

**Implementações progressivas**:

```python
# Nível 1: CORAL (Correlation Alignment) — sem GAN
def coral_loss(source_features, target_features):
    """Alinha covariância de features simulado vs real."""
    s_cov = covariance(source_features)
    t_cov = covariance(target_features)
    return tf.reduce_mean(tf.square(s_cov - t_cov))

# Nível 2: DANN (Domain Adversarial NN)
class DANNModel(tf.keras.Model):
    def __init__(self):
        self.encoder = ...
        self.task_head = ...    # inversão
        self.domain_disc = ...  # discrimina sim vs real
        self.gradient_reversal = GradientReversalLayer(lambda_=0.5)

# Nível 3: Adversarial Discriminative Domain Adaptation (ADDA)
# (Tzeng 2017) — fine-tuning separado encoder p/ target
```

**Referências**: Ganin 2016 (DANN), Sun 2016 (CORAL), Tzeng 2017 (ADDA).

#### #6 — Adversarial Training (PGD em features físicas)

**Aplicabilidade**: robustez a outliers de aquisição (eccentricity, casing effects).
**Risco**: pode reduzir clean accuracy se overaplicado.

**Quando usar**: produção em campo, após v3.0 estável.

#### #7 — SBI / NPE (Simulation-Based Inference)

**Status**: INN parcial (Zhang 2021). Formalizar como NPE.

**Vantagem sobre INN clássico**: framework `sbi` package fornece NPE/NLE/NRE
prontos. Treina rede para aproximar `p(ρ | H)` direto, sem MCMC.

```python
# Exemplo — sbi package
from sbi.inference import NPE

# Simulator wrapper: takes ρ, returns H
def simulator_for_sbi(rho_params):
    return simulate_jax(cfg, rho_params, ...)

inferred_posterior = NPE().append_simulations(
    theta=rho_samples, x=H_samples
).train()

# Inferência rápida em deployment:
samples_posterior = inferred_posterior.sample((1000,), x=H_observed)
```

**Referências**: Cranmer 2020 (review), Mackelab `sbi` package.

#### #8 — Surrogate-Accelerated MCMC

**Aplicabilidade**: UQ Bayesiana rigorosa em produção. Substitui MCMC clássico
(rejeita a maioria por ser caro) por MCMC com surrogate barato + acceptance final
com simulator caro.

```python
# Delayed acceptance MCMC
def mcmc_step(rho_current):
    rho_proposal = sample_proposal(rho_current)

    # Stage 1: surrogate-based accept/reject (barato)
    H_surrogate = surrogate(rho_proposal)
    log_alpha_1 = log_likelihood(H_observed, H_surrogate) - ...
    if not accept(log_alpha_1):
        return rho_current

    # Stage 2: full simulator (caro, mas rejeita poucos)
    H_full = simulate_jax(rho_proposal, ...)
    log_alpha_2 = log_likelihood(H_observed, H_full) - ...
    if accept(log_alpha_2):
        return rho_proposal
    return rho_current
```

**Referências**: Christen & Fox 2005, Cui 2019.

#### #9 — Diffusion Priors

**Aplicabilidade**: gerar perfis de resistividade plausíveis, **substituir
`fifthBuildTIVModels` aleatório por priors aprendidos**.

**Vantagem**: modelos sintéticos PARECEM com geologia real.

```python
# Diffusion model treinado em modelos geológicos públicos (logs reais)
class ResistivityDiffusionPrior(tf.keras.Model):
    def __init__(self):
        self.unet = build_unet_1d(...)

    def sample(self, n_models, n_layers=5):
        # DDPM forward diffusion + reverse denoising
        ...
```

**Referências**: Mosser 2017 (GeoGAN), Song 2021 (score-based), Laloy 2018.

#### #10 — Multi-Fidelity Learning (RECOMENDADO)

**Sinergia**: você JÁ tem 3 filtros Hankel (Kong 61pt fast / Werthmüller 201pt mid /
Anderson 801pt high). **Treinar duas-torres** com correção residual:

```
Torre 1: Surrogate(Kong 61pt)  →  H_LF (fast, low-fidelity)
Torre 2: Correction NN(rho)     →  ΔH (residual)
Output: H_HF = H_LF + ΔH

Custo: 1 LF eval (rápido) + 1 NN eval (rápido)
Precisão: ≈ Anderson 801pt (alto)
```

**Referências**: Perdikaris 2017 (NARGP), Meng 2020 (MF-PINN).

#### #11 — Self-Supervised Consistency Loss (QUICK WIN)

**Status**: NÃO implementado. **MAIOR ROI MARGINAL**.

**Idéia**: você tem `simulate_multi_jax` diferenciável. Adicionar loss auxiliar:

```python
# Treinamento padrão: MSE em ρ
loss_main = tf.reduce_mean(tf.square(rho_pred - rho_true))

# Adicionar loss auxiliar: forward consistency
H_reconstructed = simulate_in_training(cfg, rho_pred, esp_pred, positions)
loss_consistency = tf.reduce_mean(tf.square(H_observed - H_reconstructed))

total_loss = loss_main + 0.1 * loss_consistency
```

**Esforço**: ~1 sprint (16h). **Ganho esperado**: +5-10% RMSE em dados sintéticos
e principalmente em real data domain gap.

**Referências**: Ren 2020 (cycle-consistency), Sun 2023.

#### #12 — Reinforcement Learning para Geosteering

**Aplicabilidade**: AGENTE AUTÔNOMO de decisão. Pivot **estratégico**.

**Arquitetura**:
- State: medições LWD recentes + posição atual + plano
- Action: steer angle (continuous)
- Reward: TVD vs target + NTG (net-to-gross)
- Algorithm: PPO ou SAC com simulator no loop

**Esforço**: 2-3 meses (alta complexidade).
**Pré-requisito**: simulator GPU ≥1000 fps (atingido com JAX C8).

**Referências**: Schulman 2017 (PPO), Pollock 2018 (RL geosteering, raro).

#### #13 — Hybrid DL + Classical Inversion (Warm-Start)

**Idéia**: DL fornece ρ₀ inicial; Gauss-Newton refina com Jacobiano F10.

```python
# Workflow híbrido
rho_dl_initial = inversion_nn(measurements)        # NN warm-start
rho_refined = gauss_newton(
    initial=rho_dl_initial,
    jacobian=F10_jacobian,                          # Fortran F10 já existe
    measurements=measurements,
    max_iter=10,
)
```

**Vantagem**: precisão Gauss-Newton + velocidade DL (10ms NN vs 1-10s GN puro).
**Sinergia**: Jacobiano F10 já implementado no Fortran/Numba.

#### #14 — Physics-Guided Regularization (QUICK WIN, MENOR ESFORÇO)

**Status**: NÃO implementado. **MAIOR ROI/ESFORÇO**.

**Não é PINN** (não impõe PDE residual). É **regularização** via constraints físicos:

```python
def physics_guided_loss(rho_pred, rho_true, measurements):
    # Loss principal
    main = tf.reduce_mean(tf.square(rho_pred - rho_true))

    # Constraint 1: ρ_v ≥ ρ_h (anisotropia VTI tipicamente positiva)
    rho_h, rho_v = rho_pred[..., 0], rho_pred[..., 1]
    aniso_violation = tf.reduce_mean(tf.maximum(rho_h - rho_v, 0.0))

    # Constraint 2: Total Variation em log10(ρ) (suavidade)
    log_rho = tf.math.log(rho_pred) / tf.math.log(10.0)
    tv = tf.reduce_mean(tf.abs(log_rho[..., 1:] - log_rho[..., :-1]))

    # Constraint 3: Range realista (0.1-1000 Ω·m)
    range_violation = tf.reduce_mean(tf.maximum(rho_pred - 1000., 0.) +
                                       tf.maximum(0.1 - rho_pred, 0.))

    return main + 0.1 * aniso_violation + 0.01 * tv + 0.05 * range_violation
```

**Esforço**: ~1 dia (8h). **Ganho**: +3-7% RMSE + plausibilidade.

**Referências**: Karpatne 2017 (PGNN), Willard 2020 (survey).

### §57.3 — Recomendação Priorizada

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PRIORIZAÇÃO DAS 14 ESTRATÉGIAS                                         ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  QUICK WINS (Sprint 1-2, < 1 mês):                                       ║
│    #14  Physics-guided regularization  (1 dia)                          ║
│    #11  Self-supervised consistency    (1 sprint)                       ║
│    #4   Curriculum geológico            (1 sprint, extensão)            ║
│                                                                          ║
│  ALTO IMPACTO ESTRATÉGICO (3-6 meses):                                   ║
│    #5   Sim-to-real DA (DANN)          (CRÍTICO p/ deploy comercial)   ║
│    #7   SBI / NPE                       (UQ rigorosa, evolução INN)    ║
│    #10  Multi-fidelity                  (sinergia 3 filtros Hankel)    ║
│    #13  Hybrid DL + Gauss-Newton        (precisão+velocidade)          ║
│                                                                          ║
│  DIFERENCIADORES (6-12 meses):                                           ║
│    #2   FNO/DeepONet (SurrogateNet v3)  (10-100× speedup inferência)   ║
│    #9   Diffusion priors                (geologia plausível)            ║
│    #12  RL geosteering                  (pivot estratégico)            ║
│                                                                          ║
│  CONTEXTUAL / OPORTUNISTA:                                               ║
│    #1   Differentiable simulators       (já 80% feito)                 ║
│    #3   Active learning                 (quando datasets crescer)      ║
│    #6   Adversarial training            (após v3.0 estável)            ║
│    #8   Surrogate-accelerated MCMC      (UQ premium em produção)       ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §57.4 — Backlog de Sprints (Pós-Fase 3)

```
Sprint 30: #14 Physics-guided reg + #11 Consistency loss     (1 sprint, 8h)
Sprint 31: #4 Curriculum geológico                            (1 sprint, 16h)
Sprint 32: #10 Multi-fidelity (Kong→Anderson correction)     (1 sprint, 24h)
Sprint 33: #13 Hybrid DL + Gauss-Newton com F10              (1 sprint, 24h)
Sprint 34-35: #5 Sim-to-real DA — CORAL → DANN               (2 sprints, 48h)
Sprint 36-37: #7 SBI / NPE com framework `sbi`               (2 sprints, 48h)
Sprint 38-40: #2 FNO SurrogateNet v3                          (3 sprints, 72h)
Sprint 41-43: #9 Diffusion priors                             (3 sprints, 72h)
Sprint 44-50: #12 RL geosteering (pivot estratégico)         (7 sprints, 168h)
```


---

## §58 — Simuladores 2D/2.5D/3D FEM: Stack Híbrido

### §58.1 — Avaliação das 8 Bibliotecas Python para FEM/EM

| Lib | Domínio | CPU paralelo | GPU | Autodiff | TIV nativo | Licença | LWD HF (10k–1MHz) |
|:----|:--------|:------------:|:---:|:--------:|:----------:|:-------:|:-----------------:|
| **JAX-FEM** | FEM diferenciável | Sim (XLA) | Sim (CUDA/TPU) | Sim | parcial (tensor) | Apache 2.0 | possível (custom) |
| **scikit-fem** | FEM puro Python | Limitado (SciPy) | Não | Não | manual | BSD-3 | possível (custom) |
| **pyGIMLI** | Geofísica geral | OpenMP (C++) | Não | Não (FD) | parcial | Apache 2.0 | parcial (FDEM baixa freq) |
| **SimPEG** | Inversão geofísica | Dask/MPI | Experimental | Não (FD) | **SIM (FDEM/TDEM)** | MIT | **SIM (FDEM)** |
| **FEniCSx** (DOLFINx) | FEM produção | MPI nativo | Parcial (PETSc) | dolfin-adjoint | Sim (UFL) | LGPL-3 | **SIM** |
| **Meshio** | I/O malhas | n/a | n/a | n/a | n/a | MIT | n/a (auxiliar) |
| **PyFWI** | FWI sísmica | Sim | Parcial | Sim (custom) | n/a | LGPL-3 | **NÃO** |
| **empymod** | EM 1D semi-analítico | Numba/numexpr | Não | Não | **SIM (VTI/HTI)** | Apache 2.0 | **SIM (já usado)** |

### §58.2 — Decisão Arquitetural: Stack Híbrido por Camada

```
┌──────────────────────────────────────────────────────────────────────────┐
│  STACK FEM HÍBRIDO PARA GEOSTEERING AI                                 ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  CAMADA 1D (existente, validada)                                         ║
│  ─────────────────────────────────                                       ║
│  Backend principal:    Numba JIT custom (geosteering_ai/simulation/)    ║
│  Backend GPU:          JAX custom (_jax/)                               ║
│  Validação ground-truth: empymod 1D                                     ║
│  Ground-truth Fortran: PerfilaAnisoOmp.f08 (paridade <1e-12)            ║
│                                                                          ║
│  CAMADA 2D / 2.5D (novo, a desenvolver)                                  ║
│  ──────────────────────────────────                                      ║
│  Backend principal:    SimPEG (TIV nativo + MIT + comunidade)           ║
│  Adaptação custom:     fina layer p/ Geosteering AI                     ║
│  Backend GPU:          JAX-FEM (PINN + differentiable)                  ║
│  Validação:            FEniCSx + dolfin-adjoint (gold standard)         ║
│  Mesh I/O:             Meshio + Gmsh                                     ║
│                                                                          ║
│  CAMADA 3D (longo prazo)                                                 ║
│  ───────────────────────                                                 ║
│  Backend principal:    FEniCSx + dolfin-adjoint (Maxwell vector)        ║
│  Backend GPU diff.:    JAX-FEM (research only)                          ║
│  Mesh:                 Gmsh + Meshio                                     ║
│  Sparse solvers:       PETSc (via FEniCSx) ou cuSPARSE (JAX-FEM)        ║
│                                                                          ║
│  REJEITADOS:                                                             ║
│    ❌ pyGIMLi  → FDEM borehole não-nativo, foco em superfície           ║
│    ❌ scikit-fem → não escala 3D, sem GPU, sem MPI                      ║
│    ❌ PyFWI    → domínio errado (sísmica, não EM)                       ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §58.3 — Numba JIT em FEM 2D/2.5D/3D — Decisão

**Pergunta original**: "A implementação dos métodos FEM na CPU será em Numba JIT?"

**Resposta**: **Numba JIT NÃO é o framework primário** para FEM 2D/2.5D/3D, mas
**É usado em primitivas críticas** (assembly, sparse matrix ops):

```
┌──────────────────────────────────────────────────────────────────────────┐
│  NUMBA JIT EM FEM 2D/2.5D/3D — ESCOPO LIMITADO                         ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  PRIMITIVAS QUE USAM Numba JIT:                                          ║
│    ✅ Element matrix assembly (loop sobre elementos)                    ║
│    ✅ Element-wise integration (Gauss quadrature)                       ║
│    ✅ Sparse matrix construction (CSR/CSC)                              ║
│    ✅ TIV constitutive model (anisotropic ε ou σ tensor)               ║
│    ✅ Source term assembly (current dipoles em LWD)                     ║
│                                                                          ║
│  COMPONENTES QUE USAM SimPEG/FEniCSx:                                   ║
│    ❌ Mesh refinement adaptativo                                         ║
│    ❌ Sparse linear solvers (Mumps, SuperLU, PETSc)                     ║
│    ❌ Edge elements Nédélec (Maxwell vector)                            ║
│    ❌ MPI parallel scaling                                              ║
│    ❌ Boundary condition handling                                       ║
│                                                                          ║
│  RAZÃO: reescrever solvers e Maxwell em Numba JIT puro custaria         ║
│         12+ meses de engenharia para reinventar a roda. Usamos          ║
│         SimPEG/FEniCSx como framework e Numba JIT só onde provê         ║
│         vantagem real (assembly inner loops 5-10× mais rápido).         ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §58.4 — JAX para FEM GPU — JAX-FEM Adoption

**Pergunta original**: "A implementação dos métodos FEM na GPU será no JAX?"

**Resposta**: **SIM, via JAX-FEM** (PKU-LSTC) com adaptações.

**Por que JAX-FEM e não FEniCSx GPU**:

| Critério | JAX-FEM | FEniCSx + cuSPARSE |
|:---------|:-------:|:------------------:|
| Maturidade GPU | Alpha-beta (2024+) | Beta (PETSc-cuSPARSE) |
| Autodiff | **Nativo (jax.grad)** | dolfin-adjoint (extra step) |
| Stack alinhado ao projeto | **JAX = mesmo backend Surrogate/PINN** | C++ separado |
| Edge elements Nédélec | Custom (a implementar) | **Built-in UFL** |
| Adoção comunidade | Pequena (~600 stars) | Grande (~800+1.6k stars) |
| Comercial OK | Apache 2.0 ✅ | LGPL-3 ⚠️ (linkagem dinâmica) |

**Decisão**:
- JAX-FEM para **PINN/inversão diferenciável** (alinhado ao stack JAX existente)
- FEniCSx para **forward modeling de alta fidelidade** (validação)

### §58.5 — Adaptação SimPEG ao Geosteering AI

```python
# geosteering_ai/simulation/fem_2d/_simpeg_adapter.py (NOVO)
"""
Adaptação SimPEG para LWD HF (10kHz-1MHz) com TIV anisotropy.

SimPEG foca em CSEM/MT até ~10 kHz. Para LWD precisamos:
  1. Source: borehole dipole (não plane wave)
  2. Frequency range: 10kHz-1MHz (HF skin depth ~10cm-10m)
  3. Anisotropy: TIV diagonal (ρh, ρv)
  4. Mesh: cilíndrica 2.5D (axissimétrico em torno do poço)
"""

import numpy as np
from SimPEG import maps, utils
from SimPEG.electromagnetics import frequency_domain as fdem
from discretize import CylMesh, TensorMesh


class GeosteringSimPEGAdapter:
    """
    Adapter que envolve SimPEG FDEM para use case LWD.
    """

    def __init__(self, frequencies, tr_offsets, n_radial=50, n_z=200):
        self.frequencies = frequencies
        self.tr_offsets = tr_offsets

        # Mesh 2.5D cilíndrico
        # Axissimétrico em torno do well path
        self.mesh = CylMesh([
            [(2.0, n_radial, 1.2)],   # radial: refinamento próximo ao poço
            1,
            [(0.5, n_z, 1.0)],         # vertical
        ], "00C")

    def build_simulation(self, rho_h_array, rho_v_array, dip_angle):
        """
        Constrói simulation FDEM com anisotropia TIV.
        rho_h_array, rho_v_array: shape (n_layers,)
        """
        # Mapping: índice de camada → ρh, ρv
        sigma_h = 1.0 / rho_h_array
        sigma_v = 1.0 / rho_v_array

        # Tensor diagonal anisotrópico (TIV)
        sigma_tensor = build_tiv_tensor(sigma_h, sigma_v, self.mesh)

        # Source: vertical magnetic dipole at borehole
        sources = [
            fdem.sources.MagDipole(
                receiver_list=[
                    fdem.receivers.PointMagneticFluxDensity(
                        locations=np.r_[0, 0, z_obs],
                        orientation=("x", "y", "z"),  # Hxx, Hxy, ..., Hzz
                        component=("real", "imag"),
                    )
                ],
                frequency=f,
                location=np.r_[0, 0, z_src],
                orientation="z",
            )
            for f in self.frequencies
            for (z_src, z_obs) in self.tr_offsets
        ]

        survey = fdem.Survey(sources)
        simulation = fdem.Simulation3DMagneticFluxDensity(
            self.mesh,
            survey=survey,
            sigmaMap=maps.IdentityMap(self.mesh),
        )
        return simulation

    def run_forward(self, rho_h_array, rho_v_array, dip_angle):
        """Forward simulation → tensor H 9-comp."""
        sim = self.build_simulation(rho_h_array, rho_v_array, dip_angle)
        sigma_h = 1.0 / rho_h_array
        sigma_v = 1.0 / rho_v_array
        sigma_tensor = build_tiv_tensor(sigma_h, sigma_v, self.mesh)
        H_field = sim.dpred(sigma_tensor)
        return H_field.reshape(-1, 9)
```

### §58.6 — JAX-FEM para 2D PINN (Esboço)

```python
# geosteering_ai/simulation/fem_2d/_jax_fem.py (NOVO, longo prazo)
"""
JAX-FEM para PINN diferenciável de Maxwell harmônico em meio TIV.
"""

import jax
import jax.numpy as jnp
from jax_fem import core, problems

class MaxwellTIVProblem(problems.Problem):
    """
    Problema FEM Maxwell harmônico com anisotropia TIV.

    Equação: ∇ × (μ⁻¹ ∇ × E) - iωσ E = -iωJ_source

    Anisotropia TIV: σ = diag(σ_h, σ_h, σ_v)
    """

    def get_universal_kernel(self):
        def laplace_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
            # Built-in: assemble curl-curl term
            curl_curl_term = ...

            # Custom: anisotropic conductivity term
            sigma_h, sigma_v = self.params  # treinable!
            conduct_term = -1j * self.omega * jnp.diag([sigma_h, sigma_h, sigma_v])

            return curl_curl_term + conduct_term @ cell_sol_flat
        return laplace_kernel


# Inversão diferenciável
def loss_fn(params, measurements_observed):
    """Calculate loss with full autodiff through FEM solve."""
    rho_h, rho_v = params
    H_predicted = solve_fem_problem(rho_h, rho_v)
    return jnp.mean((H_predicted - measurements_observed)**2)

grad_fn = jax.grad(loss_fn)  # ∂loss/∂(rho_h, rho_v) via autodiff através do FEM
```

### §58.7 — Otimização por Cenário em FEM (8 cenários)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ESTRATÉGIA DE OTIMIZAÇÃO 2D/2.5D/3D POR CENÁRIO                       ║
├──────────────────────────────────────────────────────────────────────────┤
│  Cenário  Backend     Otimização                                         ║
│  ───────  ─────────   ──────────────────────────────────────             ║
│  C1       SimPEG      Single solve, baseline                             ║
│  C2       SimPEG      Reuse mesh + factorize once, vary frequency        ║
│  C3       SimPEG      Reuse mesh, multiple sources at angles              ║
│  C4       SimPEG      Reuse mesh, multiple Tx/Rx positions               ║
│  C5       SimPEG      Multi-source + multi-Rx batch, single mesh         ║
│  C6       SimPEG      Cycle freq×angle, shared mesh + assembly cache     ║
│  C7       SimPEG      Cycle freq×TR, MUMPS factorization reuse           ║
│  C8       SimPEG MPI  Multi-GPU/MPI: split frequencies across nodes      ║
│                                                                          ║
│  Para PINN/diff.:    JAX-FEM com vmap em parameter dim                  ║
│  Para 3D research:   FEniCSx + MPI, distribuir mesh                     ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §58.8 — Adaptação Geofísica/Petrofísica TIV

**Pergunta original**: "Os simuladores 2D/2.5D/3D seriam adaptados para o contexto
geofísico e petrofísico do projeto Geosteering AI?"

**Resposta**: SIM. Adaptações necessárias em todos os backends:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ADAPTAÇÕES GEOFÍSICAS/PETROFÍSICAS                                    ║
├──────────────────────────────────────────────────────────────────────────┤
│  1. Borehole geometry                                                    ║
│     • Mesh cilíndrico (CylMesh em SimPEG)                               ║
│     • Refinamento radial próximo do poço (skin depth ~ cm-m em HF)      ║
│     • Source: magnetic dipole no eixo do poço                            ║
│                                                                          ║
│  2. TIV anisotropy                                                       ║
│     • Tensor diagonal (σ_h, σ_h, σ_v)                                   ║
│     • Eixo de simetria = vertical do poço                                ║
│     • Build_tiv_tensor() em adapters                                     ║
│                                                                          ║
│  3. Mud/borehole effect (futuro)                                         ║
│     • Camada de fluido de perfuração (lama) em ~5-15 cm                 ║
│     • Salinidade variável (ρ ~ 0.05-2 Ω·m)                              ║
│     • Eccentricity (ferramenta off-center)                               ║
│                                                                          ║
│  4. Frequency range LWD                                                  ║
│     • 10 kHz - 1 MHz                                                     ║
│     • Skin depth: ρ=10 Ω·m → 100 kHz: δ ≈ 16 cm                         ║
│     • Mesh resolution mínima: λ/10 em volta do poço                     ║
│                                                                          ║
│  5. Petrofísica Archie/Klein                                             ║
│     • σ = a·φ^m·S_w^n / ρ_w  (Archie)                                  ║
│     • Mapping (φ, S_w, ρ_w) → σ                                         ║
│     • Para PINN: regularização com constraints petrofísicos             ║
│                                                                          ║
│  6. Calibration to Fortran 1D                                            ║
│     • Limit: 2D/2.5D/3D → 1D quando layers homogêneos                   ║
│     • Gate <1e-3 (FEM has discretization error vs analytical 1D)        ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §58.9 — Roadmap de Implementação 2D/2.5D/3D

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ROADMAP FEM — FASES 5-7 (PÓS-V3.0)                                    ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  FASE 5 — 2D/2.5D SimPEG ADAPTATION (8-12 semanas)                     ║
│  ─────────────────────────────────────────────────                       ║
│  S5.1: SimPEG adapter básico para LWD CylMesh                            ║
│  S5.2: Tensor TIV em SimPEG (anisotropic conductivity)                   ║
│  S5.3: Borehole sources (magnetic dipole no eixo)                        ║
│  S5.4: Validation: 2D limit → 1D (gate <1e-3)                          ║
│  S5.5: Multi-frequency simulation otimizada (cenários C2-C8)            ║
│  S5.6: Numba JIT em assembly inner loops (5-10× speedup)                ║
│                                                                          ║
│  FASE 6 — Born 2D Approximation (4-6 semanas)                            ║
│  ──────────────────────────────────────                                   ║
│  S6.1: Born approximation analítica para perturbações pequenas          ║
│  S6.2: Validação contra SimPEG full 2D                                  ║
│  S6.3: Surrogate Born para inversão rápida real-time                    ║
│                                                                          ║
│  FASE 7 — JAX-FEM PINN (6-10 semanas)                                   ║
│  ──────────────────────────────                                          ║
│  S7.1: JAX-FEM Maxwell harmonic em 2D                                   ║
│  S7.2: Anisotropic kernel (TIV)                                         ║
│  S7.3: PINN: dl_inverter + jax_fem_consistency loss                     ║
│  S7.4: Inversão diferenciável end-to-end                                ║
│                                                                          ║
│  FASE 8 — FEniCSx 3D Production (12-16 semanas)                         ║
│  ───────────────────────────────────────                                 ║
│  S8.1: FEniCSx Maxwell harmonic 3D + Nédélec elements                   ║
│  S8.2: TIV via UFL constants                                            ║
│  S8.3: dolfin-adjoint para inversão 3D                                  ║
│  S8.4: PETSc + MPI scaling                                              ║
│  S8.5: Validation 3D → 2.5D → 1D (gates progressivos)                  ║
│                                                                          ║
│  FASE 9 — RESEARCH (2D/3D ML SURROGATE)                                 ║
│  ────────────────────────────────────                                    ║
│  S9.1: 2D FNO surrogate (treinado em SimPEG 2D)                         ║
│  S9.2: 3D Graph Neural Network (mesh-aware)                             ║
│  S9.3: Multi-fidelity 1D→2.5D→3D                                        ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §58.10 — Agentes Específicos para 2D/2.5D/3D (NOVOS)

**Pergunta original**: "A arquitetura possui agentes focados nos simuladores 2D/2.5D/3D?"

**Resposta**: NÃO ainda. **NOVOS agentes** propostos:

```yaml
# Novos agentes da Parte IV §58 (3 agentes adicionais)

# .claude/agents/fem-2d-engineer.md (NOVO 22º+1)
---
name: fem-2d-engineer
model: claude-sonnet-4-6
output_style: caveman-conditional
description: |
  Engenheiro FEM 2D — domínio: SimPEG adapter, Born 2D, Numba JIT
  primitivas. Especialista em mesh cilíndrica LWD + TIV anisotropy.
deps:
  - geosteering_ai/simulation/fem_2d/
  - SimPEG, Meshio, Gmsh
skills:
  - geosteering-fem-2d (NOVO)
  - geosteering-simpeg-adapter (NOVO)
---

# .claude/agents/fem-25d-engineer.md (NOVO)
---
name: fem-25d-engineer
model: claude-sonnet-4-6
output_style: caveman-conditional
description: |
  Engenheiro FEM 2.5D — meio axissimétrico extrudado, multi-frequência.
  Foco: borehole effects, mud effects.
deps:
  - geosteering_ai/simulation/fem_25d/
  - SimPEG cylindrical, FEniCSx 2.5D
skills:
  - geosteering-fem-25d (NOVO)
---

# .claude/agents/fem-3d-engineer.md (NOVO)
---
name: fem-3d-engineer
model: claude-sonnet-4-6
output_style: caveman-conditional
description: |
  Engenheiro FEM 3D — FEniCSx + dolfin-adjoint para Maxwell vector
  com Nédélec edge elements. Inversão 3D diferenciável.
deps:
  - geosteering_ai/simulation/fem_3d/
  - FEniCSx, PETSc, MPI
skills:
  - geosteering-fem-3d (NOVO)
  - geosteering-maxwell-3d (NOVO)
---
```


---

## §59 — Agente Noise Engineer (Novo)

🔄 **NOVO AGENTE** (somando aos 18+3 da Parte III).

### §59.1 — Motivação

O projeto tem 34 tipos de noise + curriculum 3-phase implementado em
`geosteering_ai/noise/`. Atualmente, decisões sobre noise são **transversais**
a vários agentes (DL training, surrogate, PINN). Isso causa:

- Re-decisões de níveis de SNR em sprints diferentes
- Inconsistência entre noise on-the-fly em training e noise offline em test
- Falta de catálogo central de "qual noise é apropriado para qual cenário físico"
- Perda de conhecimento sobre **fidelidade física** dos noises

### §59.2 — Especificação

```yaml
# .claude/agents/noise-engineer.md (NOVO 19º agente)
---
name: noise-engineer
model: claude-sonnet-4-6
output_style: caveman-conditional
description: |
  Engenheiro de noise/ruído — responsável por catalogar, validar, manter
  e refinar todos os 34+ tipos de noise aplicados aos dados sintéticos
  e reais do projeto Geosteering AI. Garante fidelidade física e
  consistência entre training/inference.

scope:
  - Manutenção de geosteering_ai/noise/
  - Catálogo de noise tipos com docstrings físicas
  - Curriculum 3-phase (early/mid/late training)
  - Calibração SNR vs LWD real (FieldFiles)
  - Validação que noise não viola constraints físicos (positividade,
    range Hxx-Hzz)
  - Testes de regressão para `noise_preserves_zobs`
  - Otimização Numba JIT em noise functions críticas

deps_codigo:
  - geosteering_ai/noise/functions.py (34 tipos)
  - geosteering_ai/noise/curriculum.py (3-phase)
  - tests/test_noise_*.py

skills_obrigatorias:
  - geosteering-noise (NOVO)
  - geosteering-lwd-acquisition (NOVO)

actions_permitidas:
  - LER/EDITAR geosteering_ai/noise/
  - LER PDFs/* (papers sobre noise LWD)
  - LER docs/PDFs_md/Wang_2018*.md (referência fluido perfuração)
  - EXECUTAR pytest -k noise
  - PROPOR novos tipos de noise (em docs/upgrade_proposals/)

actions_proibidas:
  - EDITAR código fora de geosteering_ai/noise/
  - Modificar errata FREQUENCY_HZ ou SPACING_METERS
  - Aprovar PRs sem code-review-haiku-agent

frequencia_invocacao:
  - quando dataset DL é gerado (verificar noise pipeline)
  - quando training loop é modificado
  - quando real LWD data é integrado (calibração SNR)
  - upgrade-scout sugere novos papers sobre noise
---
```

### §59.3 — Catálogo de Noise (Mantido pelo Agente)

```python
# geosteering_ai/noise/catalog.py (NOVO)
"""
Catálogo central de 34 tipos de noise com fidelidade física.
Mantido pelo noise-engineer agent.
"""

NOISE_CATALOG = {
    "gaussian_white": {
        "physical_basis": "Thermal noise + electronic noise",
        "snr_range_db": (10, 60),
        "where_realistic": "All conditions; baseline",
        "implementation": "noise.functions.gaussian_white",
        "curriculum_phase": "all",
        "preserves_zobs": True,
        "preserves_decoupling": True,
    },
    "low_freq_drift": {
        "physical_basis": "Tool temperature drift over time",
        "snr_range_db": (15, 40),
        "where_realistic": "Long survey runs",
        "implementation": "noise.functions.low_freq_drift",
        "curriculum_phase": "mid+late",
        "preserves_zobs": True,
        "preserves_decoupling": False,
    },
    "tool_eccentricity": {
        "physical_basis": "Tool off-centering in borehole",
        "amplitude_range": (0.001, 0.05),
        "where_realistic": "Common in deviated wells",
        "implementation": "noise.functions.tool_eccentricity",
        "curriculum_phase": "late",
        "preserves_zobs": True,
        "preserves_decoupling": True,
    },
    "casing_distortion": {
        "physical_basis": "Magnetic distortion from steel casing",
        "frequency_range_hz": (10000, 100000),
        "where_realistic": "Cased holes, near surface",
        "implementation": "noise.functions.casing_distortion",
        "curriculum_phase": "late",
        "preserves_zobs": True,
        "preserves_decoupling": False,
    },
    "mudpit_conditions": {
        "physical_basis": "Drilling mud composition variations",
        "rho_mud_range_ohmm": (0.05, 2.0),
        "where_realistic": "Always (variable mud)",
        "implementation": "noise.functions.mudpit_conditions",
        "curriculum_phase": "late",
        "preserves_zobs": True,
        "preserves_decoupling": True,
    },
    # ... 29 outros tipos
}
```

### §59.4 — Curriculum Refinado

```python
# geosteering_ai/noise/curriculum.py (extensão pelo noise-engineer)
"""
Curriculum 3-phase aprimorado pela Parte IV.
"""

CURRICULUM_PHASES = {
    "early": {
        "epochs": (1, 20),
        "noise_types": ["gaussian_white"],
        "snr_db": 30,
        "rationale": "Foco em sinal limpo; aprende mapeamento básico"
    },
    "mid": {
        "epochs": (21, 60),
        "noise_types": ["gaussian_white", "low_freq_drift", "tool_eccentricity"],
        "snr_db": (20, 30),
        "rationale": "Adiciona drift + eccentricity (comuns)"
    },
    "late": {
        "epochs": (61, 120),
        "noise_types": ["gaussian_white", "low_freq_drift", "tool_eccentricity",
                       "casing_distortion", "mudpit_conditions"],
        "snr_db": (10, 25),
        "rationale": "Cenário LWD realista pleno"
    },
}
```

### §59.5 — Critérios de Aceite do Agente

| Critério | Como medir |
|:---------|:-----------|
| Catálogo completo (≥34 tipos) | `len(NOISE_CATALOG) >= 34` |
| Fidelidade física documentada | `assert all('physical_basis' in v for v in NOISE_CATALOG.values())` |
| Curriculum testado | pytest -k curriculum |
| Noise preserves z_obs | pytest -k noise_preserves |
| Calibração com LWD real | docs/reports/noise_calibration_YYYY-MM.md |
| Atualização docs/PDFs_md/ | ler papers sobre LWD noise |

---

## §60 — Simulation Manager como Terceiro Aplicativo

🔄 **REVISA** §42 da Parte III (2 trilhas → agora 3 trilhas).

### §60.1 — Mudança Estratégica

**Antes** (Parte III): 2 trilhas — pip-lib (engine + CLI) + Studio (GUI premium).

**Agora** (Parte IV): **3 trilhas** — adiciona **Simulation Manager** como
**terceiro aplicativo standalone**, voltado a simulação + análise + visualização.

### §60.2 — Topologia das 3 Trilhas

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TRÊS TRILHAS DE DISTRIBUIÇÃO                                          ║
│                                                                          ║
│  TRILHA A — pip install geosteering-ai     (Open + Free + Developer)    ║
│  ──────────────────────────────────────────────────────────────────     ║
│  • Engine Python (simuladores, models, losses, training, inference)     ║
│  • CLI Typer + API Python                                                ║
│  • Documentação e tutorials                                              ║
│  • Distribuição: PyPI · Licença: Apache 2.0/MIT                         ║
│                                                                          ║
│  TRILHA B — Simulation Manager           (Standalone Desktop App)        ║
│  ──────────────────────────────────────────────────────────────────     ║
│  • GUI desktop dedicado: simulação + análise + visualização             ║
│  • PyQt6 + matplotlib + Plotly + PyQtGraph + PyVista                    ║
│  • Foco em pesquisa/desenvolvimento e educação                          ║
│  • Distribuição: instalador macOS/Linux/Windows                         ║
│  • Licença: comercial leve OU open source (a decidir)                   ║
│  • Inclui: model generation, batch simulation, results visualization,  ║
│    PRNG control (v2.19+), CPU/GPU dispatch, profile inspection         ║
│  • Público: pesquisadores, geofísicos, alunos                          ║
│  • Estado atual: já parcialmente implementado em SM v2.21               ║
│                                                                          ║
│  TRILHA C — Geosteering AI Studio        (Commercial GUI Premium)        ║
│  ──────────────────────────────────────────────────────────────────     ║
│  • Engine pip-lib + GUI premium + WITSML + multi-poço                  ║
│  • PyQt6 + LiquidGlass + integrações comerciais                        ║
│  • Foco em produção real-time geosteering                              ║
│  • Distribuição: instalador comercial · Licença: proprietária          ║
│  • Público: empresas oil&gas, operadores LWD                           ║
│                                                                          ║
│  RELACIONAMENTO ENTRE AS 3 TRILHAS:                                     ║
│    • Todas dependem de pip-lib (engine compartilhado)                   ║
│    • Simulation Manager → Studio: features absorvidas em v4.0+         ║
│    • SM e Studio NÃO compartilham GUI code agora (intencional)         ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §60.3 — Por que Simulation Manager Standalone?

| Razão | Justificativa |
|:------|:--------------|
| Já existe (v2.21) | Reduz risco; código testado e estável |
| Foco diferente do Studio | SM é pesquisa/educação; Studio é produção |
| Velocidade de iteração | SM evolui rápido; Studio precisa estabilidade |
| Mercado distinto | Pesquisadores ≠ Operadores LWD |
| Open-source friendly | SM pode ser open; Studio é comercial |
| Adoção gradual | SM como "trial" antes de Studio |

### §60.4 — Roteiro do Simulation Manager (v2.22-v3.0+)

```
v2.22 — FLAT prange Numba (sprint atual)
v2.23 — Tile/block schedule + Hankel pre-compute
v2.24 — Refactor: separar core de GUI (preparar para 3-tier)
v2.25 — Adicionar PyQtGraph backend + multi-backend selector (§61)
v2.26 — Adicionar Plotly export + PyVista 3D para reservoirs
v2.27 — Refactor para Hexagonal Architecture (§50)
v2.28 — Polimento UI/UX + LiquidGlass conditional (macOS Tahoe)
v2.29 — Empacotamento PyInstaller (macOS .dmg, Linux AppImage, Win .msi)
v2.30 — Release Standalone + GitHub release público
v3.0  — Coexiste com Studio v0.1; features compartilhadas via pip-lib
v4.0+ — Recursos absorvidos em Studio (longo prazo)
```

### §60.5 — Implicações de Repositório

```
Repos planejados:

  github.com/daniel-leal/geosteering-ai/                # PRIMARY (lib + CLI + docs)
  github.com/daniel-leal/geosteering-simulation-manager/ # SM (GUI standalone)
  github.com/daniel-leal/geosteering-studio/             # Studio (privado, comercial)
```

**Decisão de fork**: Simulation Manager **mantém-se em geosteering-ai/** por
enquanto (até v2.30). **Sair do mono-repo apenas em v2.30+** quando empacotamento
PyInstaller estiver maduro.

### §60.6 — Atualização §42 (3 Trilhas, não 2)

```yaml
# .claude/skills/geosteering-distribution.md (NOVO)
trilhas_distribuicao:
  - A: pip-lib (engine + CLI)
  - B: Simulation Manager (standalone desktop)  # NOVA Parte IV
  - C: Geosteering AI Studio (comercial GUI premium)

# Estado de cada trilha:
A: ativo desde v2.0; PyPI alvo Q3 2026
B: ativo desde v2.0 (SM v2.21); standalone alvo v2.30
C: planejamento Q4 2026; alpha v0.1 alvo Q1 2027
```


---

## §61 — Visualization Backends do Simulation Manager

🔄 **REFINA** §49 da Parte III (4 backends adotados).

### §61.1 — Backends do Simulation Manager (Confirmação)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  BACKENDS DE VISUALIZAÇÃO NO SIMULATION MANAGER                        ║
├──────────────────────────────────────────────────────────────────────────┤
│  Backend         Disponível em SM   Default   Casos de uso              ║
│  ──────────      ────────────       ───────   ──────────────            ║
│  matplotlib      ✅ Sim              ◐ alt    Plots estáticos, papers   ║
│  Plotly          ✅ Sim              ◐ alt    Interativos, dashboards   ║
│  PyQtGraph       ✅ Sim              ★ DEFAULT Resultados real-time     ║
│  PyVista         ✅ Sim              ◐ alt    3D models, reservoirs     ║
│                                                                          ║
│  Aba específica do SM:                                                   ║
│    SECTION  RESULTADOS DE SIMULAÇÃO  → matplotlib | Plotly | PyQtGraph  ║
│                                          (PyQtGraph default)             ║
│    SECTION  3D MODELS                → PyVista                           ║
│    SECTION  EXPORT PARA PAPERS       → matplotlib (PNG/SVG/PDF)         ║
│    SECTION  EXPORT WEB/HTML          → Plotly                            ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §61.2 — Implementação do Backend Selector

```python
# geosteering_ai/simulation/tests/sm_visualization.py (atualização)
"""
Backend selector para o Simulation Manager.
"""

from typing import Literal
from PyQt6 import QtWidgets, QtCore

class VisualizationBackendSelector(QtWidgets.QWidget):
    """
    Widget de seleção de backend de visualização.
    Default: PyQtGraph (real-time interativo no GUI).
    """

    backendChanged = QtCore.pyqtSignal(str)

    BACKENDS = {
        "PyQtGraph": "Real-time interativo (default)",
        "Matplotlib": "Estático, padrão para papers",
        "Plotly":     "Interativo, exportável HTML",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)

        self.label = QtWidgets.QLabel("Backend:")
        self.combo = QtWidgets.QComboBox()
        for name, desc in self.BACKENDS.items():
            self.combo.addItem(name, userData=desc)

        # Default: PyQtGraph
        self.combo.setCurrentText("PyQtGraph")
        self.combo.currentTextChanged.connect(self.backendChanged.emit)

        layout.addWidget(self.label)
        layout.addWidget(self.combo)

    def get_current_backend(self) -> str:
        return self.combo.currentText()


# Aba "Resultados" do SM atualizada
class SimulationResultsTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        self.backend_selector = VisualizationBackendSelector()
        self.backend_selector.backendChanged.connect(self._on_backend_change)

        # Container que troca dinamicamente
        self.plot_container = QtWidgets.QStackedWidget()

        # 3 plot widgets
        self.pyqtgraph_widget = PyQtGraphPlotWidget()       # default
        self.matplotlib_widget = MatplotlibPlotWidget()
        self.plotly_widget = PlotlyPlotWidget()

        self.plot_container.addWidget(self.pyqtgraph_widget)    # idx 0
        self.plot_container.addWidget(self.matplotlib_widget)   # idx 1
        self.plot_container.addWidget(self.plotly_widget)        # idx 2

        # Default: PyQtGraph
        self.plot_container.setCurrentIndex(0)

        layout.addWidget(self.backend_selector)
        layout.addWidget(self.plot_container)

    def _on_backend_change(self, name: str):
        idx_map = {"PyQtGraph": 0, "Matplotlib": 1, "Plotly": 2}
        self.plot_container.setCurrentIndex(idx_map[name])

    def update_results(self, simulation_result):
        """Atualiza todos os 3 backends com dados frescos."""
        self.pyqtgraph_widget.set_data(simulation_result)
        self.matplotlib_widget.set_data(simulation_result)
        self.plotly_widget.set_data(simulation_result)
```

### §61.3 — Características Específicas por Backend (no SM)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CARACTERÍSTICAS POR BACKEND NO SIMULATION MANAGER                     ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  PyQtGraph (DEFAULT)                                                    ║
│    ✅ ≥60 FPS interativo                                                 ║
│    ✅ Embedded em PyQt6 (zero overhead)                                 ║
│    ✅ Zoom, pan, crosshair em tempo real                                ║
│    ✅ Export PNG/SVG                                                    ║
│    ❌ Estética spartana (não para papers)                               ║
│    ✅ Memória baixa (≤200 MB para 10k pontos)                           ║
│                                                                          ║
│  Matplotlib                                                              ║
│    ✅ Padrão científico de publicação                                   ║
│    ✅ Export PNG/SVG/PDF/PGF                                            ║
│    ✅ LaTeX rendering em labels                                         ║
│    ✅ Estética customizável (matplotlib styles)                         ║
│    ❌ Lento em interativo (10-30 FPS)                                   ║
│    ❌ Não cabe naturalmente em PyQt6 (precisa FigureCanvas)             ║
│                                                                          ║
│  Plotly                                                                  ║
│    ✅ Interativo nativo (hover, click, brush)                           ║
│    ✅ Export HTML standalone                                            ║
│    ✅ Embedded em PyQt6 via WebEngineView                              ║
│    ❌ Browser overhead em PyQt6 (Chromium embedded)                    ║
│    ❌ Latência ~50-100ms por update                                     ║
│    ✅ Compartilhamento com colaboradores via HTML                       ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §61.4 — Outros Painéis do SM (Não-Resultados)

```yaml
# Backends por painel do Simulation Manager
paineis_sm:
  parameters:
    backend: PyQt6 native widgets (no plot)

  generation:
    backend: PyQtGraph (3D scatter de modelos)
    fallback: matplotlib

  simulation_results:
    options: [PyQtGraph, Matplotlib, Plotly]
    default: PyQtGraph

  3d_reservoir:
    backend: PyVista (always)

  export:
    matplotlib: PNG/SVG/PDF
    plotly: HTML
    pyvista: VTK/STL/OBJ

  benchmarks:
    backend: matplotlib (estático para report)
```

### §61.5 — Critérios de Aceite

| Critério | Meta |
|:---------|:----:|
| Backend selector funcional | 3 opções alternáveis em runtime |
| PyQtGraph default | Sim |
| Switch entre backends | <100ms (sem freeze) |
| PyVista 3D (separado) | Sim, em painel próprio |
| Export PNG/SVG/PDF | Sim (matplotlib) |
| Export HTML | Sim (Plotly) |
| Estado preservado entre switches | Sim |

---

## §62 — Agentes Studio + Simulation Manager (Novos 20º e 21º)

### §62.1 — Pergunta: Agentes Próprios são Necessários?

**Pergunta original**: "Seria interessante gerar um agente próprio para o Simulation Manager
e outro para o Geosteering AI Studio?"

**Resposta**: **SIM**, dois agentes específicos. Razões:

| Razão | Justificativa |
|:------|:--------------|
| Domínios distintos | SM = standalone scientific; Studio = production commercial |
| Tech stacks parcialmente distintos | SM = PyQt6+matplotlib; Studio = PyQt6+LiquidGlass+WITSML |
| Releases separados | SM em geosteering-ai/; Studio em geosteering-studio/ |
| Audiences diferentes | SM = scientists; Studio = LWD operators |
| Frequência de mudanças | SM iterativo rápido; Studio estável trimestral |
| Skills associados | SM precisa visualization; Studio precisa WITSML/petrophysics |

### §62.2 — Especificação dos Novos Agentes

```yaml
# .claude/agents/simulation-manager-engineer.md (NOVO 20º agente)
---
name: simulation-manager-engineer
model: claude-sonnet-4-6
output_style: caveman-conditional
description: |
  Engenheiro especialista no Simulation Manager — terceiro aplicativo
  desktop standalone do projeto Geosteering AI. Foco em scientific
  research, batch simulation, multi-backend visualization.

scope:
  - geosteering_ai/simulation/tests/simulation_manager.py (~10k LOC)
  - geosteering_ai/simulation/tests/sm_*.py (~5k LOC)
  - PyQt6 GUI desktop
  - 4 backends de visualização (matplotlib, Plotly, PyQtGraph, PyVista)
  - Empacotamento standalone (PyInstaller, .dmg, .AppImage, .msi)

deps:
  - PyQt6, PyQtGraph, matplotlib, Plotly, PyVista
  - geosteering-ai engine (pip-lib)

skills_obrigatorias:
  - geosteering-simulation-manager (existente, expandida Parte IV)
  - geosteering-visualization (NOVO §63)
  - geosteering-pyinstaller (NOVO §63)
  - geosteering-pyqt6-gui (NOVO §63)

agentes_complementares:
  - numba-jit-engineer (cache + perf)
  - jax-engineer (GPU dispatch)
  - noise-engineer (PRNG control no SM)

restricoes:
  - NÃO editar Studio (geosteering-studio/)
  - NÃO editar core engine (geosteering_ai/) sem aprovar com numba-jit-engineer
---


# .claude/agents/studio-engineer.md (NOVO 21º agente)
---
name: studio-engineer
model: claude-sonnet-4-6
output_style: caveman-conditional
description: |
  Engenheiro especialista no Geosteering AI Studio — produto comercial
  premium para produção LWD. Foco em multi-poço, WITSML, audit trail,
  compliance.

scope:
  - geosteering-studio/ (repo separado, privado)
  - GUI PyQt6 + LiquidGlass + Plotly + PyVista
  - Integrações: WITSML, Petrel, Techlog
  - Multi-well management
  - Audit trail SOX-grade
  - Compliance: API/SPE, ISO 9001, GDPR

deps:
  - PyQt6 + PyQt6-Charts
  - WITSML 1.4 client/server
  - Petrel .NET interop
  - geosteering-ai engine (pip-lib)

skills_obrigatorias:
  - geosteering-studio (NOVO §63)
  - geosteering-witsml (NOVO §63)
  - geosteering-petrel-adapter (NOVO §63)
  - geosteering-audit-trail (NOVO §63)
  - geosteering-liquidglass (NOVO §63)

agentes_complementares:
  - geosteering-decision-agent (real-time decision)
  - inversion-engineer (real-time inversion)
  - mlops-engineer (deployment)

restricoes:
  - REPOSITORY PRIVADO — não pushar em github.com/daniel-leal/geosteering-ai
  - NÃO modificar engine pip-lib sem coordenar com pip-lib agents
  - SLA: tem que rodar em <100ms latência crítica
---
```

### §62.3 — Topologia Atualizada (até Parte IV)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TOPOLOGIA COMPLETA — 22 AGENTES (PARTE IV)                            ║
├──────────────────────────────────────────────────────────────────────────┤
│  Camada 0 — Orchestrator (1 agente Opus)                                ║
│    1.  sprint-orchestrator                                               ║
│                                                                          ║
│  Camada 1 — Domain Specialists (10 agentes Sonnet)                      ║
│    2.  numba-jit-engineer                                                ║
│    3.  jax-engineer                                                      ║
│    4.  dl-training-engineer                                              ║
│    5.  inversion-engineer (1D)                                           ║
│    6.  geosteering-decision-agent                                        ║
│    7.  literature-research-agent                                         ║
│    8.  docs-writer                                                       ║
│    9.  mlops-engineer                                                    ║
│    10. noise-engineer (NOVO §59)                                         ║
│    11. fem-2d-engineer (NOVO §58)                                        ║
│                                                                          ║
│  Camada 2 — FEM 2.5D/3D Specialists (2 agentes Sonnet)                 ║
│    12. fem-25d-engineer (NOVO §58)                                       ║
│    13. fem-3d-engineer (NOVO §58)                                        ║
│                                                                          ║
│  Camada 3 — Quality + Maintenance (5 agentes Haiku)                     ║
│    14. code-review-agent                                                 ║
│    15. fortran-parity-validator                                          ║
│    16. continuous-benchmark-agent                                        ║
│    17. numba-jax-parity-validator (Parte III §44)                       ║
│    18. bench-validator (Parte III §44)                                   ║
│                                                                          ║
│  Camada 4 — Orquestração de Tarefas Específicas (3 agentes Sonnet)      ║
│    19. colab-runner (Parte III §44)                                      ║
│    20. repo-housekeeper (Parte III §43)                                  ║
│    21. upgrade-scout (Parte III §46)                                     ║
│                                                                          ║
│  Camada 5 — Frontend/Apps (2 agentes Sonnet)                            ║
│    22. simulation-manager-engineer (NOVO §62)                           ║
│    23. studio-engineer (NOVO §62)                                        ║
│                                                                          ║
│  Camada 6 — Documentação Científica (1 agente Sonnet)                   ║
│    24. scientific-report-agent (NOVO §65)                                ║
│                                                                          ║
│  Camada 7 — Tutorias (1 agente Sonnet)                                  ║
│    25. dev-tutor-agent (NOVO §64)                                        ║
│                                                                          ║
│  Camada 8 — Review Externo Cross-LLM (1 agente Sonnet)                  ║
│    26. codex-reviewer (Parte II §36)                                    ║
└──────────────────────────────────────────────────────────────────────────┘
```

**Total: 26 agentes** (era 18 na Parte III).

### §62.4 — Por que NÃO Separar Mais Agentes

Aspectos que **NÃO** justificam novo agente separado:

| Tarefa | Razão de não-separar |
|:-------|:--------------------|
| Treinamento PINN | Coberto por dl-training-engineer + skill geosteering-pinns |
| Refactoring Hexagonal | Skill universal — todos os agentes seguem |
| Performance benchmarking | bench-validator + continuous-benchmark já cobrem |
| Plotting científico | Skills individuais por backend (não agente per se) |
| LaTeX scientific paper | scientific-report-agent (§65) é o único especialista |


---

## §63 — Skills e Hooks Novos da Parte IV

### §63.1 — Skills Novas (12 totais da Parte IV)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SKILLS NOVAS DA PARTE IV (totais 28 skills no projeto)                ║
├──────────────────────────────────────────────────────────────────────────┤
│  ID  Skill                          Cobertura                            ║
│  ──  ─────────────────────────────  ──────────────────────────           ║
│  S1  geosteering-fem-2d              SimPEG adapter, Born 2D, mesh       ║
│  S2  geosteering-fem-25d             FEM 2.5D axissimétrico              ║
│  S3  geosteering-fem-3d              FEniCSx 3D, Maxwell vector          ║
│  S4  geosteering-maxwell-3d          Nédélec elements, edge basis        ║
│  S5  geosteering-simpeg-adapter      Specific to SimPEG framework        ║
│  S6  geosteering-noise               Catálogo + curriculum + fidelidade  ║
│  S7  geosteering-lwd-acquisition     LWD physical noise, calibração SNR  ║
│  S8  geosteering-distribution        3 trilhas (pip-lib, SM, Studio)     ║
│  S9  geosteering-visualization       4 backends + selectors             ║
│  S10 geosteering-pyinstaller         Empacotamento standalone             ║
│  S11 geosteering-pyqt6-gui           Best practices PyQt6                ║
│  S12 geosteering-witsml              WITSML 1.4 client/server            ║
│  S13 geosteering-petrel-adapter      .NET interop Petrel                ║
│  S14 geosteering-audit-trail         SOX-grade decision logs             ║
│  S15 geosteering-liquidglass         macOS Tahoe vibrancy                ║
│  S16 geosteering-scientific-report   LaTeX rigor científico              ║
│  S17 geosteering-dev-tutor           Mentor para próximos passos        ║
│  S18 geosteering-sim-fem-bridge      Acoplamento 1D ↔ 2.5D ↔ 3D         ║
│  S19 geosteering-physics-guided      14 estratégias DL+sim              ║
│  S20 geosteering-hexagonal-arch      Padrão arquitetural alvo           ║
│  S21 geosteering-effort-config       Effort levels por modelo            ║
│  S22 geosteering-token-budget        Token optimization avançada         ║
│  S23 geosteering-jax-scenarios       8 cenários (C1-C8) JAX              ║
│  S24 geosteering-numba-scenarios     8 cenários (C1-C8) Numba (existente)║
│  S25 geosteering-bayesian-inversion  SBI/NPE com `sbi` package          ║
│  S26 geosteering-multi-fidelity      Hankel filters multi-fidelity      ║
│  S27 geosteering-self-supervised     Consistency loss + curriculum     ║
│  S28 geosteering-rl-geosteering      RL agent (longo prazo)             ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §63.2 — Estrutura de Skill Tipo (Template)

```markdown
# .claude/skills/geosteering-<area>.md (template)

---
description: |
  <O que essa skill cobre — ≤2 linhas para indexação rápida>
sources:
  - <arquivos a serem citados quando a skill for ativada>
  - <docs/PDFs_md/<paper>.md (se aplicável)>
  - <referências de código>

quando_usar: |
  <Cenários onde Claude deve invocar essa skill>

quando_nao_usar: |
  <Limitações e contraindicações>

agentes_que_usam:
  - <lista de agentes que dependem dessa skill>

deps_outras_skills:
  - <skills compostas>

formato_output: |
  <Como Claude deve estruturar resposta usando essa skill>

regras_específicas:
  - <Regras de domínio específicas>

example_invocations:
  - "Como rodo o simulador 2.5D para multi-frequência?"
  - "Otimize esse código FEM em SimPEG"
---

## Conhecimento de Domínio

### <Subseção 1>
...

### <Subseção 2>
...
```

### §63.3 — Hooks Novos (3 totais Parte IV)

```bash
# .claude/hooks/fem-mesh-validate.sh (NOVO §63)
#!/bin/bash
# PreToolUse hook: valida mesh antes de simulação FEM 2D/2.5D/3D
# Verifica: skin depth vs cell size, refinement próximo ao poço, condições de contorno
#
# Bloqueia simulação se: cell_size > skin_depth/10 (resolução insuficiente)
#                         OR mesh.n_cells < 100 (mesh insuficiente)
#                         OR boundary conditions ausentes

FILE="$1"
[ ! -f "$FILE" ] && exit 0
[[ "$FILE" =~ fem_(2d|25d|3d) ]] || exit 0

python -c "
import json
import sys
import discretize  # SimPEG mesh

# Verifica mesh validity
with open('$FILE', 'r') as f:
    cfg = json.load(f)

if 'mesh' not in cfg:
    print('❌ Mesh não definido em FEM config')
    sys.exit(2)

# Skin depth vs cell size
freq = cfg.get('frequency', 100000)
rho = cfg.get('rho_min', 1.0)
skin_depth = 503.3 * (rho / freq) ** 0.5  # m
cell_size = cfg['mesh'].get('cell_size_radial', 1.0)
if cell_size > skin_depth / 10:
    print(f'⚠️  Cell size ({cell_size}m) > skin_depth/10 ({skin_depth/10:.3f}m)')
    sys.exit(2)

print('✅ Mesh validation OK')
" || exit 2

exit 0
```

```bash
# .claude/hooks/latex-build.sh (NOVO §63)
#!/bin/bash
# PostToolUse hook: rebuilds LaTeX scientific report quando .tex muda
# Apenas em paper/*.tex em background
#
# Não-bloqueante (failure ok)

FILE="$1"
[[ "$FILE" =~ paper/.*\.tex$ ]] || exit 0

cd "$(dirname "$FILE")/.."

# Build silenciosamente em background
nohup pdflatex -interaction=nonstopmode "$FILE" > /tmp/latex_build.log 2>&1 &
nohup bibtex "${FILE%.tex}.aux" >> /tmp/latex_build.log 2>&1
nohup pdflatex -interaction=nonstopmode "$FILE" >> /tmp/latex_build.log 2>&1
nohup pdflatex -interaction=nonstopmode "$FILE" >> /tmp/latex_build.log 2>&1

# Avisa Daniel via macOS notification
osascript -e 'display notification "LaTeX paper rebuild started" with title "Geosteering AI"' 2>/dev/null

exit 0
```

```bash
# .claude/hooks/dev-guide-update.sh (NOVO §63)
#!/bin/bash
# PostToolUse hook: atualiza docs/dev_guide.md após cada commit
# Garante que próximos passos estão sincronizados com estado atual do repo
#
# Disparado por sprint-orchestrator ou dev-tutor-agent

git diff --cached --name-only | grep -qE "(\.py$|\.tex$|\.md$)" || exit 0

# Trigger dev-tutor-agent (se disponível)
echo "📚 Atualizando docs/dev_guide.md via dev-tutor-agent..."

# Apenas registra evento; agente decide quando processar
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ): commit detected, dev-guide may need update" \
    >> .claude/dev_guide_pending.log

exit 0
```

### §63.4 — Settings.json Atualizado

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/backup-pre-edit.sh", "timeout": 5 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/agent-acquire-lock.sh", "timeout": 3 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/validate-physics.sh", "timeout": 10 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/protect-critical-files.sh", "timeout": 5 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/check-anti-patterns.sh", "timeout": 5 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/fem-mesh-validate.sh", "timeout": 10 }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/agent-release-lock.sh", "timeout": 3 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/compile-check.sh", "timeout": 15 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/lint-v2-standards.sh", "timeout": 15 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/autoformat.sh", "timeout": 30 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/validate-scientific-refs.sh", "timeout": 10 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/post-commit-push.sh", "timeout": 10 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/latex-build.sh", "timeout": 60 },
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/dev-guide-update.sh", "timeout": 5 }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          { "type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/run-pytest.sh", "timeout": 120 }
        ]
      }
    ]
  }
}
```

### §63.5 — Total: 14 Hooks (era 12 na Parte III)

```
Parte I:    8 hooks (validate-physics, protect-critical-files, compile-check,
                     lint-v2-standards, autoformat, validate-scientific-refs,
                     run-pytest, setup-environment)
Parte III: +4 hooks (backup-pre-edit, agent-acquire-lock, agent-release-lock,
                     post-commit-push)
Parte IV:  +3 hooks (fem-mesh-validate, latex-build, dev-guide-update)

TOTAL: 15 hooks
```

---

## §64 — Arquitetura como Tutor do Desenvolvedor

🔄 **NOVO COMPONENTE** (não estava em Partes I-III).

### §64.1 — Motivação

Daniel é o desenvolvedor único e principal beneficiário. Sprints multi-arquivo
são complexos; o overhead de **lembrar onde parou** consome tempo.

**Solução**: arquitetura **mentora** (orienta) o desenvolvedor a cada passo:

- Próximo sprint sugerido (qual e por quê)
- Prompt otimizado para Claude (ready-to-paste)
- Explicação simplificada (5 min) + técnica (15 min)
- Estado atual do projeto (ROADMAP + sprint progress)
- Sinais de alerta (riscos detectados)

### §64.2 — Especificação do Sistema "dev-guide.md"

```markdown
# docs/dev_guide.md (NOVO — vivo, atualizado a cada commit)

> Última atualização: <auto, dev-guide-update.sh>
> Commit: <SHA>
> Branch: <branch atual>
> Atualizado por: dev-tutor-agent

---

## 🎯 ONDE VOCÊ ESTÁ

**Sprint atual**: v2.22 — FLAT prange Numba (Sprint 22.1 de 4)
**Progress**: 75% (3/4 tarefas concluídas)
**Bloqueado em**: Aguardando review de fortran-parity-validator

---

## 🚀 PRÓXIMOS PASSOS RECOMENDADOS

### 1. [PRIORIDADE 1] Validar paridade Fortran no FLAT prange

**O que fazer**:
- Rodar suite de validação canônica
- Verificar gates <1e-12 em 7 modelos canônicos

**Comando**:
```bash
pytest tests/test_simulation_compare_fortran.py -v -k flat_prange
```

**Por que importa**: o FLAT prange paraleliza freq×combos×pos. Risco de race condition em assembly. Validação Fortran é gate físico inviolável.

**Tempo estimado**: 30 min

---

### 2. [PRIORIDADE 2] Documentar ganho de Cenário B

**O que fazer**: Bench cenário B (multi-freq) e atualizar docs/reports/v2.22_*.md

**Prompt otimizado para Claude**:
```
Bench Cenário B (8 freqs, 1 ângulo, 1 TR, 600 pos) com novo
FLAT prange e compare com baseline v2.21. Validar paridade Fortran.
Atualizar docs/reports/v2.22_2026-05-XX.md com tabela
ganho percentual + análise.
```

**Effort**: Medium (1-5k tokens, ~1h)

---

### 3. [PRIORIDADE 3] Continuar para v2.23 (tile/block)

**O que fazer**: Iniciar Sprint v2.23 conforme roadmap §45

**Pré-requisito**: v2.22 must finalizar com gate G3 ✅
```

### §64.3 — Especificação do dev-tutor-agent

```yaml
# .claude/agents/dev-tutor-agent.md (NOVO 25º agente)
---
name: dev-tutor-agent
model: claude-sonnet-4-6
output_style: normal   # NÃO usar caveman — texto pedagógico
description: |
  Mentor automático que mantém docs/dev_guide.md atualizado a cada
  commit. Orienta Daniel sobre próximos passos, prompts otimizados,
  explicações simplificadas e técnicas. Atualiza ROADMAP automaticamente.

scope:
  - docs/dev_guide.md (vivo)
  - docs/ROADMAP.md (sincronizar com sprints)
  - .claude/dev_guide_pending.log (queue)
  - leitura: git log + .claude/active_agents.json + sprint state

frequencia_invocacao:
  - automatic: pós-commit (via hook dev-guide-update.sh)
  - manual: "/dev-guide" pelo Daniel
  - on_event: sprint completed

output:
  formato: markdown
  destino: docs/dev_guide.md
  estrutura:
    - 🎯 Onde você está
    - 🚀 Próximos passos (priorizados)
    - 📚 Explicação simplificada do que está sendo feito
    - 🔬 Explicação técnica detalhada
    - ⚠️ Sinais de alerta
    - 📊 Métricas vs metas

actions_permitidas:
  - LER git log, branches, PRs, issues
  - LER docs/ROADMAP.md, CHANGELOG.md, dev_guide.md
  - LER .claude/active_agents.json (telemetria)
  - LER MEMORY.md
  - ESCREVER docs/dev_guide.md
  - ESCREVER docs/ROADMAP.md (apenas atualizar progress %)

actions_proibidas:
  - EDITAR código fonte
  - EDITAR pyproject.toml
  - PUSH ou MERGE
  - CRIAR issues no GitHub
---
```

### §64.4 — Estrutura do dev_guide.md (Detalhada)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ESTRUTURA DO docs/dev_guide.md                                         ║
├──────────────────────────────────────────────────────────────────────────┤
│  ═══ HEADER ═══                                                          ║
│    • Última atualização (auto, timestamp)                                ║
│    • Commit SHA, Branch, Status                                          ║
│                                                                          ║
│  ═══ SEÇÃO 1: ONDE VOCÊ ESTÁ ═══                                        ║
│    • Sprint atual + progress %                                           ║
│    • Bloqueio (se houver)                                                ║
│    • Última tarefa concluída                                             ║
│                                                                          ║
│  ═══ SEÇÃO 2: PRÓXIMOS PASSOS ═══                                       ║
│    • 3-5 sugestões priorizadas                                           ║
│    • Para cada: O que fazer + Comando + Por que importa + Tempo estimado║
│    • Prompt otimizado para Claude (copiar-colar)                         ║
│    • Effort level (Low/Medium/High/...)                                  ║
│                                                                          ║
│  ═══ SEÇÃO 3: EXPLICAÇÃO SIMPLIFICADA (5min) ═══                        ║
│    • TL;DR do que está acontecendo no sprint atual                      ║
│    • Linguagem acessível, sem jargão excessivo                          ║
│    • Diagramas ASCII                                                     ║
│                                                                          ║
│  ═══ SEÇÃO 4: EXPLICAÇÃO TÉCNICA (15min) ═══                            ║
│    • Detalhe profundo da decisão arquitetural                            ║
│    • Trade-offs avaliados                                                ║
│    • Código relevante (excerpts com line numbers)                        ║
│    • Referências bibliográficas                                          ║
│                                                                          ║
│  ═══ SEÇÃO 5: SINAIS DE ALERTA ═══                                      ║
│    • Riscos detectados pelo file-watcher                                 ║
│    • Bugs latentes em known_bugs.md                                      ║
│    • Anti-patterns próximos                                              ║
│    • Sugestões do upgrade-scout pendentes                                ║
│                                                                          ║
│  ═══ SEÇÃO 6: MÉTRICAS VS METAS ═══                                     ║
│    • Performance Cenário A/B/C/D/E + tendência                           ║
│    • Custo Claude mensal vs budget                                       ║
│    • Cobertura de testes                                                 ║
│    • Bugs fechados / abertos                                             ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §64.5 — Exemplo Concreto: Sprint v2.22 Mid-Way

```markdown
# docs/dev_guide.md (snapshot 2026-05-14 14:30)

## 🎯 ONDE VOCÊ ESTÁ

**Sprint atual**: v2.22 — FLAT prange Numba
**Progress**: 60% — sprint 22.2 de 4 concluído (sprint 22.3 em andamento)
**Branch**: feat/simulation-manager-v2.22
**Último commit**: `a3b4c5d perf(numba): FLAT prange básico Cenário B`
**Estado**: ✅ paridade Fortran preservada · ⚠️  bench Cenário E ainda regressivo

---

## 🚀 PRÓXIMOS PASSOS RECOMENDADOS

### 1. 🔴 [PRIORIDADE 1] Bench cenário E completo

**O que fazer**: Rodar `bench_v214_numba.py --scenario E --all-modes` e comparar com v2.21.

**Comando**:
```bash
PYTHONPATH=. python benchmarks/bench_v214_numba.py --scenario E --models 300
```

**Effort**: **Low** (≤500 tokens response)

**Tempo estimado**: 15min

---

### 2. 🟡 [PRIORIDADE 2] Investigar regressão Cenário E

**Por que importa**: Cenário E é o cenário-alvo de produção GUI. v2.21 atingiu 122k mod/h. Se v2.22 mantiver ≥120k, gate G3 passa.

**Prompt otimizado**:
```
Investigue por que Cenário E (1f, 1a, 1TR, 600 pts) com FLAT prange
da v2.22 está em 95k mod/h enquanto v2.21 estava em 122k mod/h.
Possíveis causas: (1) overhead da indexação flat,
(2) falta de cache locality, (3) flush de prange schedule.
Profile com `cProfile` e `numba.perf_counter`. Sugira fix mantendo
ganho do Cenário B (300k → 600k).
```

**Effort**: **High** (5-20k tokens response, ~3h)

---

## 📚 EXPLICAÇÃO SIMPLIFICADA (5min)

> O simulador 1D Numba acelera código Python via JIT (just-in-time)
> compilation. v2.22 introduz FLAT prange — em vez de loops aninhados
> (ângulo dentro de freq dentro de pos), achatamos tudo em UM loop
> paralelo. Isso libera mais paralelismo em CPUs multi-core.
>
> O ganho é grande em Cenário B (multi-frequência) onde antes só
> a dimensão de posição era paralela. Mas em Cenário E (apenas posição
> múltipla), o overhead da achatamento pode prejudicar.

---

## 🔬 EXPLICAÇÃO TÉCNICA (15min)

### O que mudou em forward.py

[em forward.py:351](geosteering_ai/simulation/forward.py#L351):

```python
# Antes (v2.21)
@njit(parallel=True, nogil=True, cache=True)
def _simulate_combined_prange(...):
    for f in range(nf):                # serial
        for c in range(n_combos):      # serial
            for p in prange(n_pos):    # PARALELO
                ...

# Depois (v2.22 FLAT prange)
@njit(parallel=True, nogil=True, cache=True)
def _simulate_combined_prange_flat(...):
    n_total = nf * n_combos * n_pos
    for idx in prange(n_total):        # PARALELO em TUDO
        f = idx // (n_combos * n_pos)
        c = (idx // n_pos) % n_combos
        p = idx % n_pos
        ...
```

### Trade-offs

| Aspecto | v2.21 (nested) | v2.22 (flat) |
|:--------|:--------------:|:-----------:|
| Paralelismo | n_pos | nf×n_combos×n_pos |
| Cache locality | Boa (pos sequencial) | Possivelmente pior |
| Overhead | Baixo | +indexing math |
| Cenário B | 376k mod/h | 600k mod/h ✅ |
| Cenário E | 122k mod/h | 95k mod/h ⚠️ |

---

## ⚠️ SINAIS DE ALERTA

- 🟡 Cenário E -22% vs v2.21 — investigar antes de aceitar PR
- 🟢 Cenário B +60% — meta atingida
- 🟢 Paridade Fortran <1e-12 — preservada
- 🟢 Pytest 175/175 PASS — sem regressão
- 🔵 upgrade-scout: numpy 2.3 disponível — avaliar

---

## 📊 MÉTRICAS VS METAS

| Métrica | Atual | Meta | Status |
|:--------|:-----:|:----:|:------:|
| Cenário B | 600k | ≥600k | ✅ |
| Cenário E | 95k | ≥120k | ❌ |
| Custo mensal | $145 | ≤$200 | ✅ |
| Cobertura | 87% | ≥85% | ✅ |
| Bugs abertos | 2 | ≤5 | ✅ |
```

### §64.6 — Workflow do dev-tutor-agent

```
1. PostToolUse hook dev-guide-update.sh registra evento em queue
2. dev-tutor-agent é triggered (via cron ou manual)
3. Agente lê: git log, branches, ROADMAP, sprint state, métricas
4. Agente gera novo dev_guide.md (≤300 linhas)
5. Agente atualiza docs/ROADMAP.md com progress %
6. Daniel lê dev_guide.md no início de cada sessão
```


---

## §65 — Agente Scientific Report LaTeX

### §65.1 — Motivação

Geosteering AI é projeto científico com:
- Bases em geofísica EM 1D/2D/3D
- Inovações em DL (PINNs, INN, ModernTCN, surrogate, FNO futuro)
- Validações experimentais
- Decisões arquiteturais não-triviais (Hexagonal, multi-agent)
- Roadmap publicável

**Falta**: registro científico **DETALHADO** atualizado a cada sprint, em
formato de **artigo científico em LaTeX**, pronto para submissão a:
- IEEE Transactions on Geoscience and Remote Sensing
- Geophysics (SEG)
- Journal of Geophysical Research: Solid Earth
- Computational Geosciences (Springer)

### §65.2 — Especificação do Agente

```yaml
# .claude/agents/scientific-report-agent.md (NOVO 24º agente)
---
name: scientific-report-agent
model: claude-sonnet-4-6
output_style: normal   # NÃO usar caveman — texto pedagógico/científico
description: |
  Agente que mantém artigo científico em LaTeX vivo do projeto
  Geosteering AI. Atualiza a cada sprint significativo. Estrutura
  IEEE/ACM/Geophysics. Rigor científico, equações, referências
  bibliográficas auto-managed.

scope:
  - paper/geosteering_ai_paper.tex (vivo, ≤30 páginas)
  - paper/figures/ (figuras geradas automaticamente)
  - paper/refs.bib (BibTeX, atualizado por upgrade-scout)
  - paper/sections/ (chapters separados)
  - paper/appendices/

deps_codigo:
  - leitura: geosteering_ai/* (todo código fonte)
  - leitura: docs/reports/* (sprints concluídos)
  - leitura: docs/PDFs_md/* (literatura citável)
  - leitura: tests/ (validação experimental)

skills_obrigatorias:
  - geosteering-scientific-report (NOVO §63)
  - geosteering-latex-best-practices (NOVO)
  - geosteering-bibtex-management (NOVO)

frequencia_invocacao:
  - automatic: pós-sprint completed (via dev-tutor-agent triggered)
  - manual: "/scientific-report" pelo Daniel
  - on_event: nova feature em geosteering_ai/ ou novo paper relevante

actions_permitidas:
  - LER todo o repo (incluindo PDFs_md/)
  - ESCREVER paper/*.tex, paper/*.bib, paper/figures/*
  - EXECUTAR pdflatex (via hook latex-build.sh)
  - ATUALIZAR refs.bib quando upgrade-scout sugere papers

actions_proibidas:
  - EDITAR código fonte fora de paper/
  - SUBMETER a journal (decisão Daniel)
  - PUSH em main (apenas branch dedicado scientific-paper-update)
---
```

### §65.3 — Estrutura do Artigo (LaTeX Template)

```latex
% paper/geosteering_ai_paper.tex (NOVO, vivo)
\documentclass[11pt,a4paper]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{algorithm,algorithmic}
\usepackage[utf8]{inputenc}

\title{Geosteering AI: A Multi-Agent Architecture for
       Real-time 1D Resistivity Inversion via Deep Learning}

\author{Daniel Leal\IEEEcompsocitemizethanks{
  Author with: \href{mailto:daniel.leal.souza@gmail.com}{daniel.leal.souza@gmail.com}
}}

\maketitle

\begin{abstract}
We present \textit{Geosteering AI}, a comprehensive software architecture
for real-time 1D resistivity inversion in transversely isotropic vertical
(TIV) media for logging-while-drilling (LWD) applications. Our approach
combines (a) high-fidelity electromagnetic forward simulators with
multi-backend support (Numba JIT CPU, JAX GPU, Fortran reference),
(b) 48 deep learning architectures with physics-guided regularization,
(c) invertible neural networks for Bayesian uncertainty quantification,
and (d) a multi-agent development methodology. We achieve sub-second
inversion latency with parity to Fortran reference at $<10^{-12}$
across canonical geological models, supporting up to $10^5$ models/hour
sustained throughput on a single CPU node.
\end{abstract}

\section{Introduction}
\input{sections/01_introduction.tex}

\section{Related Work}
\input{sections/02_related_work.tex}

\section{Methodology — Forward Simulators}
\input{sections/03_simulators.tex}

\section{Methodology — Inversion via Deep Learning}
\input{sections/04_inversion.tex}

\section{Methodology — Uncertainty Quantification}
\input{sections/05_uncertainty.tex}

\section{Implementation Architecture}
\input{sections/06_architecture.tex}

\section{Experiments and Validation}
\input{sections/07_experiments.tex}

\section{Discussion}
\input{sections/08_discussion.tex}

\section{Conclusion and Future Work}
\input{sections/09_conclusion.tex}

% Apêndices
\appendices
\section{Mathematical Formulation of TIV EM Propagation}
\input{appendices/A_tiv_formulation.tex}

\section{Hankel Filter Quadrature}
\input{appendices/B_hankel.tex}

\section{Numerical Validation Tables}
\input{appendices/C_validation_tables.tex}

\section{Multi-Agent Development Architecture}
\input{appendices/D_multi_agent.tex}

\bibliographystyle{IEEEtran}
\bibliography{refs}

\end{document}
```

### §65.4 — Workflow de Atualização do Artigo

```
1. dev-tutor-agent detecta sprint v2.22 completed
2. Trigger scientific-report-agent
3. Agente lê: git log v2.22, docs/reports/v2.22_*.md, docs/PDFs_md/*, tests/
4. Agente atualiza:
   - sections/06_architecture.tex (FLAT prange algorithm)
   - sections/07_experiments.tex (tabela bench v2.22)
   - figures/bench_v22_cenarioE.pdf (gerada via matplotlib)
   - refs.bib (papers citados, se necessário)
5. Latex build via hook latex-build.sh
6. PDF gerado em paper/build/geosteering_ai_paper.pdf
7. Commit em branch scientific-paper-update; PR para main após Daniel review
```

### §65.5 — Critérios de Qualidade Científica

| Critério | Como verificar |
|:---------|:---------------|
| Rigor matemático | Equações com índices corretos; consistência dimensional |
| Citações verificáveis | Toda afirmação não-trivial referenciada em refs.bib |
| Reprodutibilidade | Tabelas de configuração + seeds + git SHA |
| Validação numérica | Gates <1e-12 vs Fortran reportados |
| Comparação literatura | Tabela com técnicas de Wang 2018, Zhang 2021, Liu 2023 |
| Acessibilidade | Notação clara; PT-BR não traduz para EN sem cuidado |
| Limitações | Seção explícita de constraints e edge cases |

### §65.6 — Versionamento do Artigo

```
paper/
├── geosteering_ai_paper.tex            # principal, vivo
├── refs.bib                             # BibTeX
├── sections/                             # chapters
├── figures/                              # PDFs gerados
├── appendices/                           # detalhes técnicos
├── build/                                # PDF gerado
├── snapshots/                            # versões periódicas
│   ├── v0.1_2026-05-15.pdf              # alpha
│   ├── v0.5_2026-08-01.pdf              # beta
│   ├── v1.0_2026-12-15.pdf              # candidato submissão
│   └── README.md
└── submissions/                          # versões enviadas a journals
    ├── 2027-Q1_geophysics_submission.pdf
    └── revisions/
```

---

## §66 — Effort Configuration por Modelo/Tarefa

### §66.1 — Definição dos Effort Levels

```
┌──────────────────────────────────────────────────────────────────────────┐
│  EFFORT LEVELS — TAXONOMIA OFICIAL                                      ║
├──────────────────────────────────────────────────────────────────────────┤
│  Level         Tokens response  Use case                                 ║
│  ──────────    ─────────────    ────────────────────────                ║
│  Low           ≤1.000           Uma resposta curta, lookup, status      ║
│  Medium        1.000-5.000      Modificação simples, plot, doc curto    ║
│  High          5.000-20.000     Refactor multi-arquivo, novo módulo     ║
│  Extra High    20.000-50.000    Sprint complexo, decisão arquitetural   ║
│  Max           >50.000          Sprint multi-arq + revisão profunda     ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §66.2 — Mapeamento Modelo × Tarefa × Effort

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MATRIZ MODELO × TAREFA × EFFORT (RECOMENDAÇÃO)                       ║
├──────────────────────────────────────────────────────────────────────────┤
│  Modelo         Effort       Tarefas típicas                             ║
│  ─────────      ──────       ──────────────────────────                  ║
│                                                                          ║
│  Haiku 4.5      Low          - Lookup arquivo                           ║
│                                - Status report                            ║
│                                - Code review pontual                     ║
│                                - Fortran parity check                    ║
│                                - Bench validator                         ║
│                                - Hooks individuais                       ║
│                                                                          ║
│  Haiku 4.5      Medium        - Documentação curta                      ║
│                                - PR description                          ║
│                                - Análise de log (≤200 linhas)            ║
│                                                                          ║
│  Sonnet 4.6     Low           - Resposta direta a Daniel                ║
│                                - Refatoração 1 arquivo                   ║
│                                                                          ║
│  Sonnet 4.6     Medium        - Edição multi-arquivo (≤3)               ║
│                                - Sprint pequeno                          ║
│                                - Doc longo (1 seção)                    ║
│                                - Skill reading                           ║
│                                                                          ║
│  Sonnet 4.6     High          - Sprint médio (3-5 arquivos)             ║
│                                - Refactor (1 módulo)                    ║
│                                - Doc completo (relatório técnico)        ║
│                                - W13 JAX Sprint                          ║
│                                - DL training engineering                 ║
│                                - FEM 2D/2.5D/3D specific                 ║
│                                                                          ║
│  Sonnet 4.6     Extra High    - Sprint v2.22+ FLAT prange               ║
│                                - Refactor Hexagonal                      ║
│                                - LaTeX paper sub-sections completos     ║
│                                                                          ║
│  Opus 4.7 (1M)  High          - Decisão arquitetural complexa            ║
│                                - Plan multi-arquivo                      ║
│                                - Análise simultânea de 3+ subsistemas   ║
│                                                                          ║
│  Opus 4.7 (1M)  Extra High    - Sprint multi-arquivo (5+)               ║
│                                - Documento arquitetural (parte deste MD)║
│                                - Refactor Hexagonal completo            ║
│                                                                          ║
│  Opus 4.7 (1M)  Max           - Esta Parte IV (5500+ linhas adicionadas)║
│                                - Análise + síntese de 5 subsistemas    ║
│                                  simultaneamente                        ║
│                                - Reorganização total da arquitetura     ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §66.3 — Matriz Atualizada de 26 Agentes × Effort Padrão

```
┌──────────────────────────────────────────────────────────────────────────┐
│  AGENTE × MODELO × EFFORT PADRÃO                                       ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  Agente                            Modelo     Effort padrão              ║
│  ────────────────────────────       ──────     ─────────────              ║
│  sprint-orchestrator               Opus 4.7   Extra High                 ║
│                                                                          ║
│  numba-jit-engineer                Sonnet     High                       ║
│  jax-engineer                      Sonnet     High                       ║
│  dl-training-engineer              Sonnet     Medium-High                ║
│  inversion-engineer (1D)           Sonnet     High                       ║
│  geosteering-decision-agent        Sonnet     Medium                     ║
│  literature-research-agent         Sonnet     Medium                     ║
│  docs-writer                       Sonnet     Medium                     ║
│  mlops-engineer                    Sonnet     Medium                     ║
│  noise-engineer                    Sonnet     Medium                     ║
│  fem-2d-engineer                   Sonnet     High                       ║
│  fem-25d-engineer                  Sonnet     High                       ║
│  fem-3d-engineer                   Sonnet     Extra High                 ║
│                                                                          ║
│  code-review-agent                 Haiku      Low-Medium                 ║
│  fortran-parity-validator          Haiku      Low                        ║
│  continuous-benchmark-agent         Haiku      Low                        ║
│  numba-jax-parity-validator        Haiku      Low                        ║
│  bench-validator                   Haiku      Low                        ║
│                                                                          ║
│  colab-runner                      Sonnet     Medium                     ║
│  repo-housekeeper                  Haiku      Low                        ║
│  upgrade-scout                     Sonnet     Medium                     ║
│                                                                          ║
│  simulation-manager-engineer        Sonnet     High                       ║
│  studio-engineer                    Sonnet     Extra High                 ║
│  scientific-report-agent           Sonnet     High                       ║
│  dev-tutor-agent                   Sonnet     Medium                     ║
│                                                                          ║
│  codex-reviewer (cross-LLM)        ChatGPT     Medium                     ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §66.4 — Sistema de Recomendação de Effort

```python
# .claude/agents/_helpers/effort_advisor.py (NOVO)
"""
Sistema que recomenda Effort level baseado em features da tarefa.
"""

from typing import Literal

EFFORT_RANGES = {
    "Low":         (0, 1000),
    "Medium":      (1000, 5000),
    "High":        (5000, 20000),
    "Extra High":  (20000, 50000),
    "Max":         (50000, float('inf')),
}


def recommend_effort(
    n_files_to_edit: int,
    is_multi_module_refactor: bool,
    requires_arch_decision: bool,
    is_sprint: bool,
    requires_validation: bool,
    requires_doc: bool,
) -> Literal["Low", "Medium", "High", "Extra High", "Max"]:
    """Recomenda effort level baseado em features da tarefa."""

    score = 0
    score += min(n_files_to_edit, 10) * 1.5
    if is_multi_module_refactor:
        score += 8
    if requires_arch_decision:
        score += 12
    if is_sprint:
        score += 6
    if requires_validation:
        score += 3
    if requires_doc:
        score += 3

    if score < 3:    return "Low"
    if score < 8:    return "Medium"
    if score < 18:   return "High"
    if score < 35:   return "Extra High"
    return "Max"


# Exemplo de uso:
recommend_effort(
    n_files_to_edit=6,
    is_multi_module_refactor=True,
    requires_arch_decision=False,
    is_sprint=True,
    requires_validation=True,
    requires_doc=True,
)
# Returns: "Extra High"
```

### §66.5 — Custo Esperado por Effort/Modelo

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CUSTO POR EFFORT × MODELO (USD, INPUT+OUTPUT TOKENS)                  ║
├──────────────────────────────────────────────────────────────────────────┤
│  Effort × Modelo            Tokens out  Custo médio                     ║
│  ──────────────────────     ──────────   ──────────                      ║
│                                                                          ║
│  Low × Haiku                  500        $0.005                          ║
│  Low × Sonnet                 500        $0.025                          ║
│  Low × Opus                   500        $0.075                          ║
│                                                                          ║
│  Medium × Haiku              3000        $0.030                          ║
│  Medium × Sonnet             3000        $0.150                          ║
│  Medium × Opus               3000        $0.450                          ║
│                                                                          ║
│  High × Sonnet              10000        $0.50                           ║
│  High × Opus                10000        $1.50                           ║
│                                                                          ║
│  Extra High × Sonnet        30000        $1.50                           ║
│  Extra High × Opus          30000        $4.50                           ║
│                                                                          ║
│  Max × Opus 1M              60000        $9.00                           ║
└──────────────────────────────────────────────────────────────────────────┘
```


---

## §67 — Síntese da Parte IV + Token/Context Optimization Final

### §67.1 — Sumário das 13 Decisões da Parte IV

```
╔══════════════════════════════════════════════════════════════════════════╗
║  PARTE IV — DECISÕES APROVADAS APÓS APROFUNDAMENTO TÉCNICO 2026-05-04   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  §55  Simulador JAX 8 cenários (C1-C8) com vmap/fori_loop/pmap         ║
║  §56  Modos: standalone (Numba default) vs DL on-the-fly (JAX)          ║
║  §57  14 estratégias DL+sim; quick wins #11+#14 (1-2 sprints)           ║
║  §58  Stack FEM 2D/2.5D/3D: SimPEG + FEniCSx + JAX-FEM hybrid          ║
║       Numba JIT só em primitivas; rejeita pyGIMLi/scikit-fem/PyFWI      ║
║  §59  Agente noise-engineer (19º)                                       ║
║  §60  Simulation Manager 3ª trilha (terceiro app standalone)            ║
║  §61  Backends SM: matplotlib + Plotly + PyQtGraph (default) + PyVista ║
║  §62  Agentes 20º (simulation-manager-engineer) e 21º (studio-engineer)║
║  §63  +12 skills + 3 hooks (fem-mesh, latex-build, dev-guide-update)   ║
║  §64  Sistema dev-guide.md + dev-tutor-agent (25º)                      ║
║  §65  Scientific Report LaTeX vivo + scientific-report-agent (24º)     ║
║  §66  Effort levels: Low/Medium/High/Extra High/Max + matriz × modelo   ║
║  §67  Token/context optimization integrada                              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### §67.2 — Topologia Final: 26 Agentes (Atualizado)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TOPOLOGIA FINAL — 26 AGENTES (POR PARTE)                              ║
├──────────────────────────────────────────────────────────────────────────┤
│  Origem        Quantidade  Total acumulado                              ║
│  ────────      ──────────  ────────────────                              ║
│  Parte I            12              12                                    ║
│  Parte III          +6              18                                    ║
│  Parte IV           +8              26                                    ║
│                                                                          ║
│  Parte IV adições:                                                      ║
│    • noise-engineer (19º)                                                ║
│    • fem-2d-engineer (20º)                                               ║
│    • fem-25d-engineer (21º)                                              ║
│    • fem-3d-engineer (22º)                                               ║
│    • simulation-manager-engineer (23º)                                  ║
│    • studio-engineer (24º)                                               ║
│    • scientific-report-agent (25º)                                       ║
│    • dev-tutor-agent (26º)                                               ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §67.3 — Token Budget Atualizado (Mês Médio)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TOKEN BUDGET MENSAL ESTIMADO (PÓS-PARTE IV)                          ║
├──────────────────────────────────────────────────────────────────────────┤
│  Categoria                       Tokens/mês  USD/mês                    ║
│  ──────────────────────────       ─────────  ────────                    ║
│                                                                          ║
│  ATIVIDADES TÍPICAS (10 sprints/mês)                                    ║
│    sprint-orchestrator (Opus)     500k       $7.50                       ║
│    Domain specialists (Sonnet)    20M         $100.00                    ║
│    Quality (Haiku)                10M         $10.00                     ║
│    Tarefas específicas (Sonnet)    5M         $25.00                     ║
│    Frontend agents (Sonnet)        4M         $20.00                     ║
│    Documentação científica         3M         $15.00                     ║
│    Tutor + scout                   2M         $10.00                     ║
│                                                                          ║
│  SUBTOTAL (ANTES DE OTIMIZAÇÕES)                  $187.50                ║
│                                                                          ║
│  OTIMIZAÇÕES APLICADAS:                                                  ║
│    -25%  Caveman conditional (§39)                -$46.88                ║
│    -10%  Skill lazy loading (Parte II §32 T7)     -$14.06                ║
│    -8%   Worktree isolation (§40)                  -$11.25                ║
│    -5%   /compact checkpoints (Parte II §32 T8)   -$7.03                 ║
│    -3%   Effort matching corretamente             -$4.22                 ║
│                                                                          ║
│  TOTAL PÓS-OTIMIZAÇÕES                             ~$104.06              ║
│                                                                          ║
│  RESERVA SEGURANÇA (50%)                          ~$52.03               ║
│                                                                          ║
│  TOTAL EFETIVO BUDGET                              ~$156.09              ║
│                                                                          ║
│  Status vs Claude Max 5×:        $200 cap         ✅ DENTRO              ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §67.4 — Estratégias de Otimização (Resumo Final)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ESTRATÉGIAS DE OTIMIZAÇÃO DE TOKENS — CONSOLIDADAS                    ║
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          ║
│  ETOK-01  Effort matching                                               ║
│           Modelo certo + tamanho de resposta certo (§66)                 ║
│                                                                          ║
│  ETOK-02  Caveman conditional (§39)                                     ║
│           Output style compactado em agentes long-running               ║
│                                                                          ║
│  ETOK-03  Skill lazy loading                                            ║
│           Skills carregadas só quando explicitamente referenced         ║
│                                                                          ║
│  ETOK-04  Worktree isolation (§40)                                      ║
│           Cada agente paralelo em worktree próprio (sem context bloat)  ║
│                                                                          ║
│  ETOK-05  KB-references                                                 ║
│           Em vez de re-explicar bug, referenciar known_bugs.md          ║
│                                                                          ║
│  ETOK-06  /compact strategic                                            ║
│           Checkpoints a cada 50-80 mensagens em sprint longo            ║
│                                                                          ║
│  ETOK-07  Concise plan files                                            ║
│           Plans estruturadas, ≤200 linhas                                ║
│                                                                          ║
│  ETOK-08  Memory de pointers, não corpo                                 ║
│           MEMORY.md aponta para project_*.md (eficiente)                ║
│                                                                          ║
│  ETOK-09  Documentação MD em vez de no-prompt                           ║
│           Conhecimento em docs/ acessado por grep, não recarregado     ║
│                                                                          ║
│  ETOK-10  Docling para PDFs (§53)                                       ║
│           PDFs convertidos para .md grep-able                           ║
│                                                                          ║
│  ETOK-11  Multi-agent paralelismo (§40)                                 ║
│           Reduz tempo wall-clock; cada agente pequeno                   ║
│                                                                          ║
│  ETOK-12  Cache de prompt prefixes                                      ║
│           CLAUDE.md + skills cached entre sessões                        ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §67.5 — Janela de Contexto: Estratégia por Modelo

```
┌──────────────────────────────────────────────────────────────────────────┐
│  USO RECOMENDADO DA JANELA DE CONTEXTO                                  ║
├──────────────────────────────────────────────────────────────────────────┤
│  Modelo            Contexto disp.   Recomendação                        ║
│  ─────────         ──────────────   ──────────────────                  ║
│                                                                          ║
│  Opus 4.7 1M       1.000.000        - Sprint multi-arquivo (≥5)         ║
│                                       - Análise arquitetural               ║
│                                       - Esta Parte IV (5500+ linhas)      ║
│                                       - Reading docs/PDFs_md/ inteiro    ║
│                                                                          ║
│  Opus 4.7          200.000           - Sprint complexo (3-4 arquivos)   ║
│                                       - Decisão arquitetural              ║
│                                                                          ║
│  Sonnet 4.6        200.000           - Sprint padrão                     ║
│                                       - Refactor 1-3 arquivos             ║
│                                       - Doc longo                          ║
│                                                                          ║
│  Haiku 4.5         200.000           - Tarefa quick (≤1k tokens)        ║
│                                       - Lookup, status, hooks             ║
│                                                                          ║
│  REGRA GERAL:                                                           ║
│    • <150k tokens contexto → Sonnet/Haiku                               ║
│    • 150k-1M tokens contexto → Opus 4.7 (200k)                          ║
│    • >1M tokens contexto → Opus 4.7 1M (raro: arquitetura completa)    ║
└──────────────────────────────────────────────────────────────────────────┘
```

### §67.6 — Inventário Total Pós-Parte IV

| Componente | Quantidade |
|:-----------|:----------:|
| Linhas totais do MD | ~12.000 (era 8.863) |
| Agentes | **26** (era 18) |
| Workflows E2E | **13** (era 12) |
| Skills | **40+** (era 16) |
| Hooks | **15** (era 12) |
| MCP servers | 4 |
| Trilhas distribuição | **3** (era 2) |
| Roadmap fases | **8** (era 4) |

### §67.7 — Arquivos Novos / Modificados (Parte IV)

**CRIAR (novos)**:

| Categoria | Arquivo | Seção |
|:----------|:--------|:-----:|
| Agente | `.claude/agents/noise-engineer.md` | §59 |
| Agente | `.claude/agents/fem-2d-engineer.md` | §58 |
| Agente | `.claude/agents/fem-25d-engineer.md` | §58 |
| Agente | `.claude/agents/fem-3d-engineer.md` | §58 |
| Agente | `.claude/agents/simulation-manager-engineer.md` | §62 |
| Agente | `.claude/agents/studio-engineer.md` | §62 |
| Agente | `.claude/agents/dev-tutor-agent.md` | §64 |
| Agente | `.claude/agents/scientific-report-agent.md` | §65 |
| Skill | `.claude/skills/geosteering-fem-2d.md` | §63 |
| Skill | `.claude/skills/geosteering-fem-25d.md` | §63 |
| Skill | `.claude/skills/geosteering-fem-3d.md` | §63 |
| Skill | `.claude/skills/geosteering-simpeg-adapter.md` | §63 |
| Skill | `.claude/skills/geosteering-noise.md` | §59 |
| Skill | `.claude/skills/geosteering-lwd-acquisition.md` | §59 |
| Skill | `.claude/skills/geosteering-distribution.md` | §60 |
| Skill | `.claude/skills/geosteering-visualization.md` | §61 |
| Skill | `.claude/skills/geosteering-pyinstaller.md` | §63 |
| Skill | `.claude/skills/geosteering-witsml.md` | §62 |
| Skill | `.claude/skills/geosteering-scientific-report.md` | §65 |
| Skill | `.claude/skills/geosteering-dev-tutor.md` | §64 |
| Skill | `.claude/skills/geosteering-jax-scenarios.md` | §55 |
| Skill | `.claude/skills/geosteering-effort-config.md` | §66 |
| Hook | `.claude/hooks/fem-mesh-validate.sh` | §63 |
| Hook | `.claude/hooks/latex-build.sh` | §63 |
| Hook | `.claude/hooks/dev-guide-update.sh` | §63 |
| Code | `geosteering_ai/simulation/_jax/scenarios.py` | §55 |
| Code | `geosteering_ai/simulation/_jax/forward_diff.py` | §56 |
| Code | `geosteering_ai/simulation/keras_layer.py` | §56 |
| Code | `geosteering_ai/simulation/fem_2d/_simpeg_adapter.py` | §58 |
| Code | `geosteering_ai/simulation/fem_2d/_jax_fem.py` | §58 |
| Code | `geosteering_ai/noise/catalog.py` | §59 |
| Code | `geosteering_ai/simulation/tests/sm_visualization.py` (atualização) | §61 |
| Code | `.claude/agents/_helpers/effort_advisor.py` | §66 |
| Docs | `docs/dev_guide.md` (vivo) | §64 |
| Docs | `paper/geosteering_ai_paper.tex` + sections + appendices + figures + refs.bib | §65 |
| Tests | `tests/test_simulation_jax_scenarios.py` | §55 |
| Tests | `tests/test_pinn_consistency.py` | §56 |
| Tests | `tests/test_noise_catalog.py` | §59 |
| Bench | `benchmarks/bench_jax_scenarios.py` | §55 |

### §67.8 — Modificar (Parte IV)

| Arquivo | Mudança | Seção |
|:--------|:--------|:-----:|
| `.claude/settings.json` | +3 hooks Parte IV | §63 |
| `pyproject.toml` | +deps: SimPEG, FEniCSx, JAX-FEM, sbi, neuraloperator | §57, §58 |
| `CLAUDE.md` | atualizar com Parte IV | todos |
| `docs/CHANGELOG.md` | append entry v3.0 Parte IV | todos |
| `docs/ROADMAP.md` | +fases 5-9 (FEM + DL avançado) | §58 |
| `MEMORY.md` | nova seção Parte IV | todos |
| `geosteering_ai/simulation/_jax/forward_pure.py` | dispatcher por cenário | §55 |
| `geosteering_ai/simulation/multi_forward.py` | usar dispatcher | §55 |
| `geosteering_ai/simulation/__init__.py` | expor scenarios + Keras layer | §55, §56 |

### §67.9 — Cronograma de Adoção (4 Semanas Pós-Parte IV)

```
SEMANA 1 (2026-05-05 a 2026-05-11) — JAX SCENARIOS + FORWARD-DIFF
─────────────────────────────────────────────────────────────────
Dia 1-2: §55 _simulate_c1 a c8 + dispatcher
Dia 3:   §56 forward_diff + Keras layer
Dia 4-5: §55 tests + bench T4
Sábado:  §57 #14 physics-guided regularization (quick win)
Domingo: §57 #11 self-supervised consistency loss (quick win)

SEMANA 2 (2026-05-12 a 2026-05-18) — NOISE + SM + AGENTS
─────────────────────────────────────────────────────────────────
Dia 1-2: §59 noise-engineer + catalog.py
Dia 3:   §60 SM 3ª trilha + skills
Dia 4-5: §62 simulation-manager-engineer + studio-engineer agents
Sábado:  §61 backend selector PyQtGraph default em SM
Domingo: §63 hooks fem-mesh-validate + latex-build + dev-guide-update

SEMANA 3 (2026-05-19 a 2026-05-25) — DEV-TUTOR + SCIENTIFIC-REPORT
─────────────────────────────────────────────────────────────────
Dia 1-2: §64 dev-tutor-agent + dev_guide.md inicial
Dia 3-4: §65 scientific-report-agent + paper template + first sections
Dia 5:   §66 effort_advisor.py + matriz × modelos
Sábado:  §67 dashboard token budget atualizado
Domingo: revisão Daniel + retrospectiva mensal

SEMANA 4 (2026-05-26 a 2026-06-01) — FEM 2D PHASE 5 START
─────────────────────────────────────────────────────────────────
Dia 1-2: §58 fem-2d-engineer agent + SimPEG adapter básico
Dia 3-4: §58 mesh CylMesh para LWD + tensor TIV
Dia 5:   Sprint 5.1 finalize + bench
Sábado:  Validation 2D limit → 1D (gate <1e-3)
Domingo: docs/reports/v3.0_fem_phase_5_1.md
```

### §67.10 — Próximos Passos Imediatos

```
1. [Daniel] Revisar Parte IV completa (esta seção)
2. [Daniel] Validar backup pré-Parte IV em
   .backups/2026-05-04/arquitetura_multiagente_aprofundamento_031438.pre-partIV.bak
3. [Daniel + Sonnet] Quick wins #14 + #11 (8h total, alta ROI)
4. [Daniel + Sonnet] noise-engineer + catalog.py (1 sprint)
5. [Daniel + Opus] §55 JAX scenarios + dispatcher (1 sprint)
6. [Daniel + Opus] §56 forward_diff Keras layer (1 sprint)
7. [Daniel + Sonnet] dev-tutor + scientific-report agents (2 sprints)
8. [Daniel + Sonnet] FEM 2D Phase 5.1 SimPEG adapter (1-2 sprints)
9. [Daniel + Opus] Refactor SM v2.30+ (separar core de GUI)
10. [Daniel] Decidir: licença SM (Apache vs comercial-light) — fim de Q3 2026
```

### §67.11 — Atualização Total do Documento (Pós-Parte IV)

```
PARTE I  (3.552 linhas) — visão geral, 17 seções, agentes iniciais
PARTE II (~2.000 linhas) — aprofundamentos críticos 13 questões
PARTE III (~3.300 linhas) — refinamentos técnicos 15 questões
PARTE IV (~3.000 linhas) — especialização profunda 13 questões + 8 novos agentes
                             + JAX scenarios + FEM 2D/2.5D/3D + Scientific paper
                             + dev-tutor + Effort levels + 3 trilhas distribuição

TOTAL: ~12.000 linhas combinadas
TOTAL DE AGENTES: 26 (era 12 na Parte I)
TOTAL DE WORKFLOWS: 13 (era 12 na Parte I)
TOTAL DE SKILLS: 40+ (era 9 na Parte I)
TOTAL DE HOOKS: 15 (era 8 na Parte I)
TOTAL DE TRILHAS: 3 (pip-lib + Simulation Manager + Studio)
ROADMAP DE FASES: 8 (era 4 na Parte I/III)
```

### §67.12 — Critérios de Coerência (Auto-Validação)

```
✅ Acentuação PT-BR preservada em todas as seções
✅ Numeração contínua §1-§67 sem buracos
✅ Referências cruzadas (REVISA §N) explícitas
✅ Nenhuma contradição com Partes I-III (apenas extensões/refinamentos)
✅ Backup pré-Parte IV preservado
✅ Footer atualizado com totais corretos
✅ Inventário arquivos novos/modificados completo
✅ Cronograma de adoção 4 semanas explícito
✅ Tabelas estruturadas ≥70% (vs prosa ≤30%)
```

---

**Parte IV adicionada em 2026-05-04 com Claude Opus 4.7 (1M contexto).**

**Status até Parte IV: DOCUMENTO BASE OFICIAL — Parte I (visão geral)
+ Parte II (aprofundamentos críticos) + Parte III (refinamentos técnicos
profundos) + Parte IV (especialização: JAX scenarios, FEM 2D/2.5D/3D,
scientific paper, dev-tutor, effort levels, 3 trilhas distribuição).**

**Backup pré-Parte IV**:
`.backups/2026-05-04/arquitetura_multiagente_aprofundamento_031438.pre-partIV.bak`
(436K, validado).

**Backup pré-Parte V**:
`.backups/2026-05-04/arquitetura_multiagente_aprofundamento_050219.pre-partV-colab.bak`
(600K, validado).

---

# PARTE V — AUTOMAÇÃO GOOGLE COLAB PRO+ COM MCP (§68-§73)

> **Data**: 2026-05-04
> **Modelo**: Claude Opus 4.7 (1M contexto)
> **Escopo**: Cinco perguntas do Daniel sobre automação de testes GPU em Colab
> Pro+, viabilidade de MCP personalizado, integração Antigravity, e
> incorporação na arquitetura de software do Geosteering AI 2.0
> **Premissa firme do Daniel**: **Colab Pro+ exclusivo** — não há planos
> para Colab Enterprise (que opera via Vertex AI / GCP, e tem custo
> sob demanda de US$ 0,40-3,00/h por GPU além da assinatura).

## §68 — Análise das Cinco Perguntas do Daniel

### §68.1 — Pergunta 1: Automação de Testes com GPU no Colab

**Pergunta**: *"Seria possível automatizar o processo de testes com uso
de recursos envolvendo a GPU? Há algum planejamento na arquitetura
Geosteering AI que preveja algo assim? Algo como enviar código para
um notebook Colab aberto e conectado? Seria possível fazer isso através
de um MCP personalizado?"*

**Resposta direta**: SIM em três camadas distintas, com diferentes graus
de automação e diferentes pré-requisitos. A Parte III §44 já introduziu
o agente `colab-runner` como integrante do Workflow W13 (JAX Sprint),
mas naquele momento ele era um stub conceitual sem tooling concreto.
A Parte V converte aquele stub em arquitetura executável e a expande
para cobrir 4 cenários de uso (interativo / batch / CI / paper-grade
benchmark).

**Sumário das três camadas viáveis sob Colab Pro+**:

```
┌──────────────────────────────────────────────────────────────────────┐
│  CAMADA A — Drive Sync + Manual Run (T0 — semi-automática)          │
│    • Daniel mantém uma aba Colab aberta com notebook do projeto     │
│    • Claude Code edita .ipynb localmente (NotebookEdit tool)        │
│    • rclone/gdrive-cli sincroniza para o Drive                      │
│    • Daniel clica "Runtime → Run all" no Colab                      │
│    • Resultados ficam no Drive; Claude Code lê via rclone           │
│    • Custo extra: $0; latência humana: ~30s por run                 │
│                                                                      │
│  CAMADA B — MCP oficial googlecolab/colab-mcp (T1 — interativa)     │
│    • MCP server local faz ponte com aba Colab aberta no browser     │
│    • Tools expostas: notebook (criar), execute_code, pip_install    │
│    • Daniel mantém aba aberta; Claude Code envia comandos via MCP   │
│    • Funciona com Pro+ (qualquer assinatura que permita browser)    │
│    • Custo extra: $0; latência: ~3-5s por célula                    │
│                                                                      │
│  CAMADA C — MCP headless pdwi2020/mcp-server-colab-exec (T2 — CI)   │
│    • OAuth2 com token cacheado em ~/.config/colab-exec/token.json   │
│    • Pré-auth manual em browser uma vez, depois 100% headless       │
│    • Tools: colab_execute (inline), colab_execute_file (.py),       │
│       colab_execute_notebook (full + zip artifacts)                  │
│    • Seleção GPU por execução: T4 ou L4 (sem A100/V100 ainda)       │
│    • Funciona com Pro+ (mesmas credenciais OAuth da extensão        │
│       Colab para VS Code)                                            │
│    • Custo extra: $0; latência: ~10s startup + execução pura        │
└──────────────────────────────────────────────────────────────────────┘
```

**Limitação fundamental compartilhada por Colab (Pro+ ou Free)**: o
Colab roda em VMs efêmeras. Mesmo no Pro+ (que oferece sessões de até
24h e desconexão tolerante a inatividade limitada), reconectar requer:

1. Reinstalar dependências (não há venv persistente além do Drive)
2. Re-uploadar dados se grandes demais para Drive
3. Re-anexar GPU (que pode estar indisponível em horário de pico)

A arquitetura proposta na Parte V trata cada um desses três pontos
explicitamente.

### §68.2 — Pergunta 2: Agente ou Hook para Configurar/Executar MCP Oficial

**Pergunta**: *"É possível gerar um agente ou hook que configure e
execute o MCP oficial do Google Colab?"*

**Resposta direta**: SIM, três artefatos serão criados:

| Artefato | Tipo | Função |
|:---------|:-----|:-------|
| `colab-bridge` | **Agente Sonnet** (27º) | Orquestra toda interação com Colab — escolhe Camada A/B/C apropriada, gerencia tokens, valida outputs |
| `colab-token-refresh.sh` | **Hook PostToolUse** | Renova token OAuth2 ANTES de qualquer chamada Colab MCP que seja iminente |
| `geosteering-colab-mcp` | **Skill** | Documenta comandos, troubleshooting, exemplos para uso humano e LLM |

Detalhamento completo em §72.

### §68.3 — Pergunta 3: Premissa Pro+ (sem Enterprise)

**Confirmação registrada**: a arquitetura Parte V assume **exclusivamente
Colab Pro+** (US$ 49,99/mês pessoal). Toda menção a Colab Enterprise
em §44 (Parte III) e §29 (Parte II) é DEPRECATED para fins desta Parte V
e mantida apenas como referência histórica.

**Implicação técnica**: NÃO usaremos:

- `gcloud colab executions create` (Vertex AI — Enterprise-only)
- Vertex AI Custom Jobs com aceleradores
- `aiplatform.CustomTrainingJob` da SDK Google

**SUBSTITUTOS adotados**:

- Todas as automações de execução headless usam `pdwi2020/mcp-server-colab-exec`
  (Camada C) ou Cloud Run Jobs com L4 (somente se Daniel optar por orçamento
  GCP separado no futuro)
- Benchmarks reproduzíveis usam Papermill local + sync Drive (Camada A) —
  NÃO Vertex Custom Jobs

### §68.4 — Pergunta 4: Antigravity Plugin Control

**Pergunta**: *"O Claude Code tem acesso aos notebooks abertos no
Antigravity? Se sim, é possível controlar via Claude Code, o plugin
(extensão) do Colab já instalado no Antigravity? Investigue em detalhes."*

**Resposta direta**: **NÃO é viável** controlar plugins do Antigravity
via Claude Code. Detalhamento na §70.

**Resumo executivo**:

```
┌─ Claude Code ──┐         ┌─ Antigravity (IDE Google) ────┐
│  CLI/extension │         │  Roda Gemini 3.1 Pro nativo   │
│  Anthropic     │         │  Tem plugin Colab embutido    │
│  Sonnet/Opus/  │         │  Tem plugin Gemini Code       │
│  Haiku         │         │  Cliente independente         │
└────────┬───────┘         └─────────────┬─────────────────┘
         │                               │
         │  ↓ ÚNICA PONTE EXISTENTE      │
         │                               │
         │     antigravity-claude-proxy  │
         │     (token bridge — nível de  │
         │     autenticação, NÃO de UI)  │
         │                               │
         └──────────── ✗ ────────────────┘
                      │
                      └─ Claude Code NÃO controla plugins
                         da janela Antigravity. NÃO há
                         API IPC/RPC entre os dois.
```

A ponte `antigravity-claude-proxy` (Badri Narayanan, 2025) permite usar
**tokens da assinatura Antigravity** dentro do Claude Code para chamar
modelos Gemini/Claude, mas opera no nível de **API key**, não no nível
de **automação de UI**. Não há, em maio/2026, API pública do Antigravity
para programar suas extensões internas — Antigravity é uma fork do VS Code
mas sem endpoints WebSocket equivalentes ao `code-server` que permitiriam
controle externo.

### §68.5 — Pergunta 5: Incorporação na Arquitetura

**Pergunta**: *"Verifique as possibilidades e, caso sejam viáveis,
incorporem no documento da arquitetura de software do Geosteering AI."*

**Resposta direta**: SIM, incorporação completa na Parte V abaixo,
com:

- **§69** — inventário detalhado dos MCPs disponíveis (oficial vs alternativas)
- **§70** — análise técnica do veredicto sobre Antigravity (NÃO viável)
- **§71** — escolha arquitetural: 4-tier strategy (T0/T1/T2/T3)
- **§72** — implementação: agente `colab-bridge`, hook `colab-token-refresh.sh`,
  skill `geosteering-colab-mcp`, secrets management
- **§73** — síntese, cronograma de adoção, inventário pós-Parte V

## §69 — Inventário Detalhado de MCP Servers Colab

### §69.1 — `googlecolab/colab-mcp` (oficial — Camada B)

**Status**: Lançado em março/2026 pela Google Developer Relations.
Descontinuou o experimento "ColabAgents" anterior. Já está configurado
em `/Users/daniel/Geosteering_AI/.mcp.json`:

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 60000
    }
  }
}
```

**Arquitetura interna**:

```
┌─────────────────────────────────────────────────────────────┐
│  Claude Code (cliente MCP)                                  │
│       │                                                      │
│       ↓  STDIO/MCP protocol                                  │
│  uvx colab-mcp (servidor MCP local Python)                   │
│       │                                                      │
│       ↓  WebSocket localhost                                 │
│  Browser tab abert@ no Colab (passa cookies/auth do Google) │
│       │                                                      │
│       ↓  HTTPS para Google Colab kernel                      │
│  VM efêmera Colab Pro+ com GPU T4/A100/L4                   │
└─────────────────────────────────────────────────────────────┘
```

**Tools MCP expostas (confirmado oficialmente)**:

| Tool | Função | I/O |
|:-----|:-------|:----|
| `notebook` | Cria notebook novo no Colab | `name: str → notebook_url: str` |
| `execute_code` | Executa célula Python no kernel ativo | `code: str → output: str + errors: list` |
| `pip_install` | Instala pacote no runtime atual | `package: str → success: bool` |
| `open_colab_browser_connection` | Abre/conecta sessão browser | `() → connected: bool` |

**Pré-requisitos operacionais**:

1. **Aba Colab aberta no browser** durante toda a sessão Claude Code
2. Login do Google ativo no browser
3. Notebook autorizado (clicar "Run anyway" 1× por notebook novo)
4. `uv` instalado (`brew install uv` no macOS Daniel)

**Limitações declaradas**:

- "Maintainers cannot accommodate external pull requests at this time" —
  o repositório é gerido internamente pela Google; bugs reportados via
  GitHub Discussions, não Issues.
- Documentação pública mínima: README só mostra setup; tools list não
  é enumerada formalmente, apenas inferida de exemplos do blog post.
- Compatibilidade com Pro+ não é mencionada explicitamente, mas funciona
  com qualquer assinatura que permita browser sessions (Free/Pro/Pro+).

**Adoção Geosteering AI**: SIM como **Camada B (T1) — uso interativo**.
Daniel mantém aba aberta durante sessões de development; Claude Code
envia código de teste/benchmark via MCP; latência ~3-5s por execução.
Adequado para iteração rápida, NÃO para CI noturno.

### §69.2 — `pdwi2020/mcp-server-colab-exec` (terceiro — Camada C HEADLESS)

**Status**: Mantido por Pankaj Dwivedi (independente). Lançado abril/2026.
Não-Google mas usa **mesmas credenciais OAuth2** da extensão oficial
"Google Colab" para VS Code.

**Diferencial-chave vs `googlecolab/colab-mcp`**: HEADLESS após auth
inicial. Token OAuth fica em `~/.config/colab-exec/token.json` e renova
automaticamente. Não requer browser aberto durante execução.

**Setup recomendado para Geosteering AI**:

```bash
# 1. Instalar (em macOS Daniel)
uv tool install git+https://github.com/pdwi2020/mcp-server-colab-exec

# 2. Auth inicial (1 vez, browser-based consent)
mcp-server-colab-exec --auth
#   → abre browser, login Google, autoriza scope "drive.file" + "colab"
#   → salva ~/.config/colab-exec/token.json

# 3. Adicionar a .mcp.json
cat <<EOF >> .mcp.json.tmp
{
  "mcpServers": {
    "colab-exec": {
      "command": "mcp-server-colab-exec",
      "args": [],
      "env": { "COLAB_TOKEN_PATH": "~/.config/colab-exec/token.json" },
      "timeout": 600000
    }
  }
}
EOF
```

**Tools MCP expostas (3 ferramentas)**:

| Tool | Função | Input | Output |
|:-----|:-------|:------|:-------|
| `colab_execute` | Executa código inline | `code: str, accelerator: "T4"\|"L4"` | `{cells: [{stdout, stderr, error}], runtime: float}` |
| `colab_execute_file` | Executa arquivo .py local | `file_path: str (.py only), accelerator` | igual `colab_execute` |
| `colab_execute_notebook` | Executa notebook + coleta artefatos | `notebook_path, output_dir, accelerator` | igual + `artifacts: list[Path]` (zip extraído) |

**Limitações**:

- GPU options: apenas T4 e L4 (sem A100, V100, TPU); mas T4 é exatamente
  a meta de §44 e §55, e L4 é equivalente A100 para muitos workloads
- Validação de zip antes de extrair (path traversal protection) — bom
- Apenas `.py` para `colab_execute_file` (sem `.ipynb` direto, mas
  `colab_execute_notebook` cobre)
- Sem A100 explícito; para A100 precisa Camada B (browser)

**Adoção Geosteering AI**: SIM como **Camada C (T2) — CI noturno e
batch headless**. Roda em GitHub Actions, scripts de bench overnight,
hook PostToolUse para validação periódica de paridade JAX-vs-Numba.

### §69.3 — `datalayer/jupyter-mcp-server` (Jupyter genérico)

**Status**: Maduro para JupyterLab local, mas **suporte a Google Colab
ainda em desenvolvimento** (citação literal do README, maio/2026).

**Adoção Geosteering AI**: NÃO no curto prazo. Reavaliar Q3/2026
quando suporte Colab estiver maduro. Pode substituir `pdwi2020` se
oferecer mais controle.

### §69.4 — Custom MCP `geosteering-colab-mcp` (interno — Camada D opcional)

**Status**: NÃO existe ainda. Proposta para futuro caso Camadas B+C
mostrem-se insuficientes para necessidades específicas do projeto.

**Justificativa de criação (somente se):**

1. Camada B/C falharem em atender benchmarks JAX-GPU específicos
2. Necessidade de GPU profiling persistente cross-session
3. Necessidade de multi-notebook orquestração (W13 paralelo W14)
4. Camada B/C tiverem rate-limit problemático

**Especificação preliminar** (caso seja construído):

```python
# Hipotético geosteering_ai/colab_mcp_server/server.py
@mcp_tool
def geosteering_bench_jax_t4(
    cenario: Literal["C1","C2","C3","C4","C5","C6","C7","C8"],
    n_models: int = 10000,
    seed: int = 42,
) -> BenchResult:
    """Executa benchmark JAX no T4 do Colab e retorna métricas estruturadas."""
    ...

@mcp_tool
def geosteering_validate_fortran_parity(
    test_models: list[str],
    backends: list[Literal["numba","jax","fortran"]],
) -> ParityReport:
    """Valida paridade <1e-12 entre backends nos N modelos canônicos."""
    ...

@mcp_tool
def geosteering_train_surrogate_modern_tcn(
    config_yaml: str,
    epochs: int,
    accelerator: Literal["T4","A100"],
) -> TrainResult:
    """Lança treino completo SurrogateNet ModernTCN no Colab e devolve
    histograma de loss + checkpoint URI no Drive."""
    ...
```

**Decisão**: ADIAR para Sprint 28+ (pós-v3.0). Por ora, Camadas A/B/C
cobrem 100% das necessidades imediatas (W13 JAX Sprint, surrogate training,
quick wins #14 #11).

### §69.5 — Tabela Comparativa Final

| Aspecto | A: Drive+Manual | B: googlecolab/colab-mcp | C: pdwi2020/colab-exec | D: custom (futuro) |
|:--------|:---------------:|:------------------------:|:----------------------:|:------------------:|
| **Status** | ✅ Pronto | ✅ Pronto (no .mcp.json) | ⚠️ Instalar | ❌ Não construído |
| **Browser aberto** | Sim (manual click) | **Sim** (durante toda sessão) | **Não** (após auth 1×) | A definir |
| **Headless CI** | Não | Não | **Sim** | A definir |
| **GPU options** | T4/A100/L4/V100 | T4/A100/L4 | T4/L4 | A definir |
| **Autenticação** | Login Google browser | Login Google browser | OAuth2 token cache | A definir |
| **Latência por exec** | ~30s humana | ~3-5s | ~10s startup | A definir |
| **Custo extra** | $0 | $0 | $0 | A definir |
| **Pro+ compatível** | ✅ | ✅ | ✅ | ✅ |
| **Maintainers PRs** | N/A | ❌ Fechado a externos | ✅ Aberto | Interno |
| **Casos de uso** | Manual debug | Iteração rápida dev | CI/batch noturno | Specific bench |
| **Risco de quebra** | Mínimo | Médio (Google muda UI) | Médio (deps OAuth) | Total controle |

## §70 — Veredicto sobre Antigravity Bridge (NÃO Viável)

### §70.1 — O que é o Antigravity?

**Antigravity** é o IDE de codificação assistida por IA da Google,
lançado em meados de 2025. Roda Gemini 3.1 Pro como agente nativo.
É arquiteturalmente um **fork do VS Code** com:

- Plugin "Gemini Code" embutido (conversational)
- Plugin "Google Colab" embutido (cell execution / runtime selection)
- Plugin "Drive" para sync de arquivos
- API REST interna para autenticação Google

### §70.2 — O que NÃO existe no Antigravity (relevante para nossa pergunta)

**Inexistente em maio/2026**:

1. **API pública para extensões externas** controlarem plugins internos.
   Antigravity tem o seu próprio Marketplace (separado do VS Code Marketplace),
   mas não expõe IPC/RPC para clientes externos como Claude Code.

2. **Endpoint WebSocket equivalente ao `code-server`**. O VS Code OSS
   permite execução remota via `code-server`; Antigravity removeu/desativou
   esse endpoint (provavelmente por questões de auth Google).

3. **CLI `antigravity` para invocar comandos da UI**. Não há equivalente
   ao `code .` que abriria notebooks no plugin Colab via comando externo.

4. **Webhook/event API** para notificar agentes externos sobre execuções
   de células no plugin Colab.

### §70.3 — A Única Ponte Existente: `antigravity-claude-proxy`

**Repo**: github.com/badrisnarayanan/antigravity-claude-proxy
**O que faz**: expõe os modelos Claude/Gemini disponíveis na assinatura
Antigravity como API key utilizável dentro do Claude Code (CLI).

```
Daniel paga assinatura Antigravity
         │
         ↓
antigravity-claude-proxy lê tokens
         │
         ↓
Claude Code usa esses tokens em vez de Anthropic API
         │
         ↓
Daniel economiza ~$30-50/mês (custo Claude Code parcialmente
absorvido pela assinatura Antigravity)
```

**O que NÃO faz**:

- ❌ NÃO permite Claude Code controlar a UI do Antigravity
- ❌ NÃO permite Claude Code ler estado dos notebooks abertos no plugin Colab
- ❌ NÃO permite Claude Code disparar "Run cell" no plugin Colab embutido
- ❌ NÃO expõe API para sincronizar contexto entre Antigravity Gemini agent
  e Claude Code

### §70.4 — Veredicto Arquitetural

**DECISÃO FIRME**: Antigravity é tratado como **ferramenta independente
e paralela ao Claude Code**, não como host de plugins controláveis.

**Implicações**:

- Geosteering AI v3.0+ NÃO incluirá rotinas de "send to Antigravity Colab plugin"
- Daniel pode usar Antigravity manualmente em paralelo (e.g., para revisões
  de código com Gemini), mas a automação Colab fica integralmente nas
  Camadas A/B/C definidas em §69
- Caso futuramente o Antigravity exponha API plugin-control (Q3/2026+?),
  reavaliar e adicionar como Camada E opcional

**Por que essa decisão é segura**:

- Antigravity tem ciclo de release fechado/Google-controlado
- Adicionar dependência de UI fechada introduz fragilidade
- Camadas A/B/C já cobrem 100% dos casos de uso (interativo + headless + manual)
- Daniel já tem `.mcp.json` configurado com `colab-mcp` oficial

## §71 — Arquitetura 4-Tier de Automação Colab (Adotada)

### §71.1 — Visão Geral em Camadas

```
┌──────────────────────────────────────────────────────────────────────┐
│  USER INTENT                                                         │
│    "rodar bench JAX C8 em A100 do Pro+"                              │
│    "validar paridade JAX-vs-Numba 7 modelos canônicos"               │
│    "treinar SurrogateNet ModernTCN 100 epochs"                       │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ↓
┌──────────────────────────────────────────────────────────────────────┐
│  AGENTE colab-bridge (27º agente, Sonnet, High effort)              │
│    1. Classifica request: interativo? batch? CI? paper-grade?        │
│    2. Verifica recursos: GPU type necessária, tempo estimado         │
│    3. Escolhe Camada A/B/C apropriada                                │
│    4. Verifica token (chama hook colab-token-refresh.sh se ≤24h     │
│       de validade restante)                                          │
│    5. Despacha execução                                              │
│    6. Coleta resultados, valida, gera summary MD                     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ↓                     ↓                     ↓
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  CAMADA A   │     │  CAMADA B    │     │  CAMADA C    │
│  (T0)       │     │  (T1)        │     │  (T2)        │
│             │     │              │     │              │
│  Drive sync │     │ googlecolab  │     │  pdwi2020    │
│  + manual   │     │ /colab-mcp   │     │  /colab-exec │
│  click      │     │ (browser)    │     │  (headless)  │
│             │     │              │     │              │
│  Casos:     │     │  Casos:      │     │  Casos:      │
│  • One-off  │     │  • Dev iter  │     │  • CI nightly│
│  • Manual   │     │  • Bench A100│     │  • Bench T4  │
│    debug    │     │  • Paper-    │     │  • Paridade  │
│  • Setup    │     │    grade     │     │    JAX-Numba │
│    init     │     │    profiling │     │  • Surrogate │
│             │     │              │     │    train T4  │
└─────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ↓
┌──────────────────────────────────────────────────────────────────────┐
│  COLAB PRO+ (Daniel's account)                                       │
│    GPU: T4 (default) / A100 (interactive only) / L4 (headless)      │
│    Sessão: até 24h                                                   │
│    Drive: 2TB sync para artefatos                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### §71.2 — Decisão de Camada (Heurística Programada)

A escolha entre A/B/C é feita pelo agente `colab-bridge` (§72.1) usando
a seguinte tabela de decisão:

| Critério | Camada A (Drive+Manual) | Camada B (browser MCP) | Camada C (headless MCP) |
|:---------|:----------------------:|:----------------------:|:-----------------------:|
| Daniel está online e disponível | ⚠️ Aceitável | ✅ Ideal | ✅ Ideal |
| Daniel está offline (CI noturno) | ❌ Impossível | ❌ Impossível | ✅ Único viável |
| Necessita A100/V100 GPU | ⚠️ Possível (manual) | ✅ Único viável | ❌ Não suporta |
| Necessita T4/L4 GPU | ⚠️ Possível | ✅ OK | ✅ Ideal |
| Run < 30s | ⚠️ Overhead manual | ✅ Ideal | ⚠️ Startup ~10s |
| Run ≥ 5min | ✅ Aceitável | ✅ Aceitável | ✅ Ideal |
| Run ≥ 1h | ⚠️ Daniel precisa monitorar | ⚠️ Browser timeout risco | ✅ Ideal |
| Coleta de artefatos (zip+plots) | ⚠️ Manual via rclone | ⚠️ Manual via Drive | ✅ Auto-zip nativo |
| First-time auth | ❌ N/A (Daniel já loga) | ⚠️ "Run anyway" 1× | ⚠️ Browser consent 1× |
| Token expira/precisa refresh | ❌ N/A | ✅ Cookie browser longo | ⚠️ ~7 dias OAuth refresh |
| Confiabilidade contra Google UI changes | ✅ Mínima (rclone API) | ⚠️ Médio risco | ⚠️ Médio risco |

**Pseudocódigo da heurística** (executada pelo agente `colab-bridge`):

```python
def choose_colab_tier(request: ColabRequest) -> Literal["A","B","C"]:
    # Hard rules
    if request.gpu_type in {"A100", "V100"}:
        return "B"  # único que suporta A100
    if not request.user_present and request.is_long_running:
        return "C"  # CI noturno
    if request.is_one_shot_manual_debug:
        return "A"  # Daniel quer ver e ajustar
    # Soft rules
    if request.estimated_runtime_min > 30 and request.user_present:
        return "C"  # libera Daniel para outra coisa
    if request.estimated_runtime_min < 1 and request.user_present:
        return "B"  # iteração rápida
    return "C"  # default seguro: headless
```

### §71.3 — Onde Cada Camada Vive na Arquitetura Geosteering AI

| Camada | Diretório | Arquivos-chave |
|:-------|:----------|:---------------|
| **A (Drive+Manual)** | `tools/colab/drive_sync/` | `sync_to_drive.py`, `pull_from_drive.py`, `notebooks/templates/*.ipynb` |
| **B (browser MCP)** | `.mcp.json` (já configurado) | `googlecolab/colab-mcp` via uvx (sem código local) |
| **C (headless MCP)** | `.mcp.json` (a configurar) | `pdwi2020/mcp-server-colab-exec` via uv tool install |
| **Agente bridge** | `.claude/agents/colab-bridge.md` | Definição agente (a criar §72.1) |
| **Hook refresh** | `.claude/hooks/colab-token-refresh.sh` | Script renovação OAuth (a criar §72.2) |
| **Skill docs** | `.claude/commands/geosteering-colab-mcp.md` | Documentação para humano+LLM (a criar §72.3) |
| **Notebooks template** | `notebooks/colab_templates/` | `bench_jax_C1_C8.ipynb`, `validate_fortran_parity.ipynb`, `train_surrogate_modern_tcn.ipynb` |

## §72 — Implementação: Agente, Hook, Skill, Secrets

### §72.1 — Agente `colab-bridge` (27º agente)

**Localização**: `.claude/agents/colab-bridge.md`
**Modelo**: Sonnet
**Effort default**: High (5-20k tokens response — tipicamente analisa
log Colab, gera summary MD, sugere próximos passos)
**Caveman**: ATIVO (long-running orchestration)

**Frontmatter proposto**:

```yaml
---
name: colab-bridge
description: Orquestra automação Google Colab Pro+ via 3 camadas (Drive+Manual / browser MCP / headless MCP). Use when usuário pede bench GPU, treino remoto, validação JAX-vs-Numba, ou qualquer execução Python que requer GPU não disponível localmente. Escolhe camada A/B/C baseado em GPU type, runtime estimado, presença do usuário, necessidade de coleta de artefatos. Gera summary MD após execução.
tools: Bash, Read, Write, Edit, Grep, Glob, mcp__colab-mcp__*, mcp__colab-exec__*
model: sonnet
---
```

**Responsabilidades-chave**:

1. **Classificar request** em uma das 4 categorias:
   - `interactive_bench` → Camada B
   - `paper_grade_a100` → Camada B
   - `ci_nightly_validation` → Camada C
   - `manual_debug` → Camada A (gera instruções para Daniel executar)

2. **Verificar token OAuth**: dispara hook `colab-token-refresh.sh` se
   token expira em ≤ 24h.

3. **Selecionar GPU type**: respeita preferência usuário; se A100 pedido
   e Camada C escolhida, ESCALONA pra Camada B com mensagem clara.

4. **Despachar e coletar**: chama tool MCP apropriada, captura output,
   valida exit code, extrai métricas estruturadas.

5. **Gerar summary MD** em `docs/colab_runs/YYYY-MM-DD_HHMMSS_<id>.md`
   com: GPU usada, runtime real, métricas, comparação vs metas (e.g.,
   §55 targets T4=50ms C1, A100=2.5s C8), warnings, anexos.

### §72.2 — Hook `colab-token-refresh.sh`

**Localização**: `.claude/hooks/colab-token-refresh.sh`
**Tipo**: PostToolUse (executa após cada Edit/Write em arquivos sensíveis)
**Configuração** (em `.claude/settings.json`):

```json
{
  "PostToolUse": [
    {
      "matcher": "Edit|Write",
      "hooks": [
        {
          "type": "command",
          "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/colab-token-refresh.sh",
          "timeout": 30
        }
      ]
    }
  ]
}
```

**Lógica do script** (skeleton):

```bash
#!/usr/bin/env bash
# Renova token OAuth Colab se expira em ≤24h
# Idempotente, silencioso se token está OK

TOKEN_PATH="${HOME}/.config/colab-exec/token.json"
THRESHOLD_HOURS=24

if [[ ! -f "$TOKEN_PATH" ]]; then
  exit 0  # token ainda não criado; agente colab-bridge pedirá auth manual
fi

EXPIRY=$(python3 -c "
import json, sys
from datetime import datetime, timezone, timedelta
with open('$TOKEN_PATH') as f:
    tok = json.load(f)
exp = datetime.fromisoformat(tok['expiry'].replace('Z','+00:00'))
remaining = exp - datetime.now(timezone.utc)
print(int(remaining.total_seconds() / 3600))
")

if (( EXPIRY <= THRESHOLD_HOURS )); then
  echo "[colab-token-refresh] Token expira em ${EXPIRY}h. Renovando..."
  mcp-server-colab-exec --refresh-token > /tmp/colab-refresh.log 2>&1
  if [[ $? -eq 0 ]]; then
    echo "[colab-token-refresh] OK"
  else
    echo "[colab-token-refresh] FALHA — verifique /tmp/colab-refresh.log"
    exit 1  # bloqueia tool use até resolver
  fi
fi
```

**Trigger**: ANTES de qualquer chamada `colab-exec` o token é validado.
Se expirado e refresh falha, hook bloqueia (exit 1).

### §72.3 — Skill `geosteering-colab-mcp`

**Localização**: `.claude/commands/geosteering-colab-mcp.md`
**Conteúdo**: documentação completa para uso humano e LLM.

**Estrutura do MD** (sumário das 12 seções):

1. **Visão geral 3 camadas** — quando usar cada
2. **Setup Camada A** — rclone, pasta Drive, sync bidirecional
3. **Setup Camada B** — `.mcp.json` já configurado, browser steps
4. **Setup Camada C** — `uv tool install`, auth inicial, token cache
5. **Comandos Bash típicos** — exemplos copy-paste para Daniel
6. **Tools MCP disponíveis** — lista completa A/B/C
7. **Troubleshooting** — token expirado, rate limit, GPU indisponível
8. **Templates de notebook** — bench JAX, paridade, treino
9. **Métricas alvo** (cross-ref §55, §44) — T4=50ms C1, A100=2.5s C8
10. **Cronograma de uso típico** — quando rodar bench (semanal? Q*)
11. **Custo total** — $0 extra (Pro+ já assinado), apenas tempo
12. **Alternativas futuras** — Camada D custom, Cloud Run L4

### §72.4 — Secrets Management

**Itens sensíveis envolvidos**:

| Item | Localização | Tipo | Rotação |
|:-----|:------------|:-----|:--------|
| Token OAuth Camada C | `~/.config/colab-exec/token.json` | OAuth refresh token | Auto (~7 dias) |
| Cookie sessão browser Camada B | Browser do macOS | HTTP cookie Google | Manual (login) |
| `.env GCP` (se Cloud Run futuro) | `~/.geosteering/.env` | Service account JSON | Manual |
| `S2_API_KEY` (Consensus, não-Colab) | env var shell | API key | Manual |

**Regras**:

- NUNCA commitar `~/.config/colab-exec/token.json` (já em `.gitignore`
  global, mas validar com hook `protect-critical-files.sh`)
- Hook PreToolUse `protect-critical-files.sh` bloqueia commits que
  tocariam em arquivos OAuth
- Backup do token: NÃO recomendado (refresh é gratuito e seguro)
- Se Daniel migra de Mac, refazer auth do zero (15s)

### §72.5 — Templates de Notebook (Camada A/B)

**Localização**: `notebooks/colab_templates/`

**Templates iniciais**:

```
notebooks/colab_templates/
├── bench_jax_C1.ipynb           — Cenário §55.C1 (single shot)
├── bench_jax_C8.ipynb           — Cenário §55.C8 (triple vmap)
├── validate_fortran_parity.ipynb — 7 modelos canônicos <1e-12
├── train_surrogate_tcn.ipynb    — TCN 127m baseline
├── train_surrogate_modern_tcn.ipynb — ModernTCN 204m
├── physics_guided_quickwin_14.ipynb — §57 #14
├── self_supervised_quickwin_11.ipynb — §57 #11
└── README.md                     — Como usar cada template
```

**Convenção**: cada notebook tem células parametrizáveis via Papermill
(`# parameters` cell tag), permitindo execução automatizada via CLI:

```bash
papermill bench_jax_C8.ipynb out/run_001.ipynb -p n_models 10000 -p seed 42
```

## §73 — Síntese, Cronograma, Inventário Pós-Parte V

### §73.1 — Resposta Consolidada às 5 Perguntas

| # | Pergunta | Resposta |
|:--|:---------|:---------|
| 1 | Automatização Colab GPU? MCP personalizado? | SIM. 3 camadas viáveis (A: Drive+Manual / B: oficial MCP / C: headless MCP). MCP custom (D) adiada para Sprint 28+. |
| 2 | Agente/hook que configure e execute MCP oficial? | SIM. Agente `colab-bridge` (27º) + hook `colab-token-refresh.sh` + skill `geosteering-colab-mcp`. |
| 3 | Pro+ exclusivo (sem Enterprise)? | CONFIRMADO. Toda referência a Vertex AI / Enterprise marcada DEPRECATED. |
| 4 | Claude Code controla plugin Colab do Antigravity? | NÃO. Antigravity não expõe API plugin-control. Único bridge (`antigravity-claude-proxy`) opera em nível de token, não de UI. |
| 5 | Incorporar na arquitetura? | FEITO. Parte V §68-§73 documenta 4-tier strategy, 3 artefatos novos, 8 templates de notebook. |

### §73.2 — Inventário Total Pós-Parte V

| Categoria | Pré-Parte V | Adições Parte V | Pós-Parte V |
|:----------|:-----------:|:---------------:|:-----------:|
| **Linhas MD** | 12.239 | ~900 | ~13.140 |
| **Agentes** | 26 | +1 (`colab-bridge`) | **27** |
| **Workflows E2E** | 13 (W01-W13) | 0 (W13 já cobria Colab) | 13 |
| **Skills** | 40+ | +1 (`geosteering-colab-mcp`) | **41+** |
| **Hooks** | 15 | +1 (`colab-token-refresh.sh`) | **16** |
| **MCP servers configurados** | 4 | +1 (`pdwi2020/colab-exec`) | **5** |
| **Trilhas distribuição** | 3 | 0 (Colab é dev tool, não trilha) | 3 |
| **Templates notebook** | 0 (não inventariados) | +8 | 8 |

### §73.3 — Arquivos a Criar (Parte V)

**Criar (~14 novos)**:

```
.claude/agents/colab-bridge.md                          (≈300 linhas)
.claude/hooks/colab-token-refresh.sh                    (≈80 linhas)
.claude/commands/geosteering-colab-mcp.md               (≈600 linhas)
notebooks/colab_templates/README.md                     (≈100 linhas)
notebooks/colab_templates/bench_jax_C1.ipynb            (≈40 cells)
notebooks/colab_templates/bench_jax_C8.ipynb            (≈80 cells)
notebooks/colab_templates/validate_fortran_parity.ipynb (≈30 cells)
notebooks/colab_templates/train_surrogate_tcn.ipynb     (≈50 cells)
notebooks/colab_templates/train_surrogate_modern_tcn.ipynb (≈60 cells)
notebooks/colab_templates/physics_guided_quickwin_14.ipynb (≈40 cells)
notebooks/colab_templates/self_supervised_quickwin_11.ipynb (≈40 cells)
tools/colab/drive_sync/sync_to_drive.py                 (≈150 linhas)
tools/colab/drive_sync/pull_from_drive.py               (≈150 linhas)
docs/reports/parte_v_colab_automation_2026-05-04.md     (≈200 linhas)
```

### §73.4 — Arquivos a Modificar (Parte V)

**Modificar (~5)**:

```
.mcp.json                — adicionar entry colab-exec
.claude/settings.json    — adicionar hook colab-token-refresh
CLAUDE.md                — referência §68-§73 em "Plugins e Agentes"
ROADMAP.md               — Fase 5b: Colab automation rollout
MEMORY.md                — pointer project_arch_partV.md
```

### §73.5 — Cronograma de Adoção (1 semana)

```
Semana 1 (após revisão Daniel da Parte V):
  Dia 1-2: §72.3 skill geosteering-colab-mcp + setup Camada C
           (uv tool install + auth OAuth)
  Dia 3:   §72.1 agente colab-bridge + §72.2 hook token-refresh
  Dia 4:   3 templates iniciais (bench_jax_C1, validate_fortran_parity,
           bench_jax_C8)
  Dia 5:   Smoke test E2E (Daniel pede "rodar paridade no T4 do Colab",
           agente escolhe Camada C, executa, gera summary MD)
  Dia 6-7: Documentação + ROADMAP update + commit
```

### §73.6 — Métricas Pós-Mês 1 (alvo)

```
✅ Camada C operacional: ≥5 runs nightly bem-sucedidos
✅ Camada B usada ≥1×/semana para A100 paper-grade
✅ Camada A documentada mas não obrigatória
✅ Token OAuth nunca expirou inesperadamente (hook funciona)
✅ docs/colab_runs/ contém ≥10 summary MDs
✅ §55 targets T4=50ms C1 validados em hardware real
✅ §44 W13 JAX Sprint backlog reduz em 2 itens (W13.1 + W13.2)
✅ Custo extra Colab: $0 (Pro+ já pago)
```

### §73.7 — Decisões NÃO Mudaram (de Partes I-IV)

```
✅ Backup-pre-edit: ainda primeira tarefa (executado em 2026-05-04 05:02:19)
✅ Fortran parity <1e-12: inviolável (Camada C/B verifica isso remotamente)
✅ TensorFlow exclusivo: confirmado
✅ PT-BR acentuado: confirmado em todo §68-§73
✅ Hexagonal Architecture: deferido para Sprints 25-28
✅ Caveman conditional: ATIVO em colab-bridge (long-running)
✅ Effort levels: colab-bridge é High (5-20k tokens response)
✅ Token budget mensal: ≤$200 (Colab Pro+ não conta — assinatura separada)
```

### §73.8 — Próximos Passos Imediatos (após revisão Daniel)

```
1. Daniel revisa Parte V completa (§68-§73)
2. Validar backup .backups/2026-05-04/arquitetura_...050219.pre-partV-colab.bak
3. Decidir: implementar em Semana 1 imediata OU agendar p/ Semana 5
   (após adoção Parte IV completa)
4. Caso GO: criar branch feat/colab-bridge-v1 a partir de main
5. Implementar §72.1 (agente) primeiro — é o mais valioso e independente
6. §72.2 (hook) e §72.3 (skill) seguem em paralelo
7. Templates de notebook em segundo momento (Sprint dedicada)
8. Smoke test E2E com 1 dos 7 modelos canônicos no T4 do Colab
```

### §73.9 — Critérios de Coerência (Auto-Validação Parte V)

```
✅ Acentuação PT-BR preservada em §68-§73
✅ Numeração contínua §1-§73 sem buracos
✅ Cross-refs explícitas: §44, §55, §57 (Partes III-IV)
✅ Nenhuma contradição com Partes I-IV
   (Colab Enterprise marcado DEPRECATED em §44 e §29 onde aparecia)
✅ Backup pré-Parte V validado (600K, 2026-05-04 05:02:19)
✅ Footer atualizado com totais corretos pós-Parte V
✅ Inventário arquivos novos/modificados completo (14 criar, 5 modificar)
✅ Cronograma de adoção 1 semana explícito
✅ Tabelas estruturadas ≥70% (vs prosa ≤30%)
✅ 5 perguntas do Daniel respondidas explicitamente em §73.1
```

---

**Parte V adicionada em 2026-05-04 com Claude Opus 4.7 (1M contexto).**

**Status: DOCUMENTO BASE OFICIAL — Parte I (visão geral) + Parte II
(aprofundamentos críticos) + Parte III (refinamentos técnicos profundos)
+ Parte IV (especialização: JAX scenarios, FEM, scientific paper, dev-tutor,
effort levels, 3 trilhas distribuição) + Parte V (automação Colab Pro+ com
4-tier MCP strategy, agente colab-bridge, hook token-refresh) formam
a referência canônica completa para construção da arquitetura do
Geosteering AI v3.0+.**

**Backup pré-Parte V**:
`.backups/2026-05-04/arquitetura_multiagente_aprofundamento_050219.pre-partV-colab.bak`
(600K, validado).

**Próxima revisão programada**: após implementação completa da Semana 1
da adoção Parte V (agente colab-bridge + hook + skill + 3 templates) OU
após Semana 4 da adoção Parte IV (JAX scenarios + FEM 2D Phase 5.1),
o que vier primeiro.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   GEOSTEERING AI v3.0+ — ARQUITETURA MULTI-AGENTE COMPLETA              ║
║                                                                          ║
║   27 agentes · 13 workflows · 41+ skills · 16 hooks · 5 MCP servers    ║
║                                                                          ║
║   Sob restrições rígidas:                                                ║
║     • Paridade Fortran <1e-12 (físico)                                  ║
║     • PT-BR acentuado (documental)                                      ║
║     • TensorFlow exclusivo (framework)                                  ║
║     • Quality Mesh 7-camadas (qualidade)                                ║
║     • Effort levels Low→Max (custo de tokens)                          ║
║     • Colab Pro+ exclusivo (sem Enterprise — confirmado §68.3)          ║
║                                                                          ║
║   Para entregar 3 trilhas:                                              ║
║     • pip install geosteering-ai (open source, scientists/devs)         ║
║     • Simulation Manager (standalone desktop research/educação)         ║
║     • Geosteering AI Studio (commercial GUI premium)                    ║
║                                                                          ║
║   Cobrindo simulação 1D Numba/JAX + FEM 2D/2.5D/3D + DL maduro         ║
║   + scientific paper LaTeX + dev guidance contínuo                      ║
║   + automação GPU Colab Pro+ via MCP 4-tier (Drive→Browser→Headless     ║
║   →Custom).                                                              ║
║                                                                          ║
║   Em ~52 semanas distribuídas em 8 fases.                               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```
