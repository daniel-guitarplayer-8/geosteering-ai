# Análise Arquitetural — MVC para Simulation Manager e Geosteering AI Studio

**Data:** 2026-05-18
**Autor:** Daniel Leal (assistido por Claude Opus 4.7)
**Tipo:** Análise arquitetural / Viabilidade / Roadmap
**Escopo:** GUI Simulation Manager atual (~10.7k LOC monolíticas + 17 módulos `sm_*.py`) e futuro Geosteering AI Studio (Trilha B comercial)
**Status:** Investigação — **não implementa código** (conforme solicitado pelo usuário)
**Branch atual:** `main` (commit `0dab72b` — v2.40 mergeada)
**Versão do projeto:** v2.40 (Sprint v2.40 C13 — MCP colab-bridge + tf.data + mp16)

---

## Sumário Executivo

| Pergunta | Resposta |
|:---------|:---------|
| O projeto **já prevê** MVC para a GUI? | **Sim** — Sprints 25–28 pós-v3.0 (~5 sprints, 6–7 semanas) |
| Há **decisão arquitetural escrita**? | **Sim** — §50 `arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md` (Hexagonal+DDD) + §5 `v2.29.3_questions_2026-05-11.md` (esboço MVC) + §7 `v2.39_proximos_passos_roadmap_2026-05-17.md` (Hex⊕MVC) |
| MVC e Hexagonal são **conflitantes**? | **Não** — confirmadamente complementares: Hexagonal = topologia do sistema; MVC = estrutura interna do *adapter* GUI |
| Studio é **evolução** do Simulation Manager? | **Não** — é aplicação **separada** (repo privado `geosteering-studio`, Trilha B comercial); SM é dev tool interno (pip-lib, Trilha A) |
| Pré-requisitos para iniciar refactor MVC | (1) `pytest-qt` ✅ entregue v2.33 · (2) API REST I2.7 ⏳ pendente · (3) Dockerfile I2.8 ⏳ pendente |
| Risco de **não fazer** o refactor | **Médio-alto** — `simulation_manager.py` cresceu **+95% em 11 sprints** (5.512→10.759 LOC v2.21→v2.29.3); custo cognitivo e token-cost aumentam linearmente |
| Recomendação central | **Estratégia Strangler Fig** em 3 fases (preparação · split incremental · controllers) iniciando **APÓS** v3.0 ou em paralelo a I2.7/I2.8 como sprint v2.41–v2.45 opcional |

---

## 1. Estado Atual da GUI — Inventário Físico

### 1.1 O "Monolito" `simulation_manager.py`

| Métrica | Valor | Comentário |
|:--------|:------|:-----------|
| Arquivo | [geosteering_ai/simulation/tests/simulation_manager.py](../../geosteering_ai/simulation/tests/simulation_manager.py) | ~471 KB · path indicia débito (em `tests/`, não em `gui/`) |
| Linhas de código (LOC) | **10.759** | Cresceu de 5.512 (v2.21) para 10.759 (v2.29.3) — **+95% em 11 sprints** |
| Classes Qt | **15** (`SimulationSnapshot`, `ExperimentState`, `NewExperimentDialog`, `WelcomeWidget`, `ParametersPage`, `SimulatorPage`, `ConfigDRowEditor`, `ConfigDDialog`, `BenchmarkPage`, `SaveFigureDialog`, `PlotComposerDialog`, `ResultsPage`, `PreferencesPage`, `SimulatorTab`, `MainWindow`) | Mistura UI + state + threading + I/O |
| Função `main()` + smoke test | linhas 9092, 10720 | CLI embutido no mesmo arquivo |
| Token cost estimado | **~80.000 tokens** | **~16% de janela 500k** — limita uso em contexto Opus 1M |
| Tendência | +480 LOC/sprint média | Em ritmo atual atinge ~15k LOC em mais 9 sprints |

### 1.2 Módulos `sm_*.py` — A Modularização "Invisível"

O projeto **já possui** modularização granular (17 módulos, ~9.3k LOC) — o que está faltando é apenas **agregação e separação MVC**:

| Módulo | LOC | Camada MVC implícita |
|:-------|:----:|:----------------------|
| [sm_plots.py](../../geosteering_ai/simulation/tests/sm_plots.py) | 1.772 | **View** (renderização matplotlib) |
| [sm_workers.py](../../geosteering_ai/simulation/tests/sm_workers.py) | 1.103 | **Controller** (SimulationThread, SimRequest) + **Service** (NumbaPrimer) |
| [sm_correlation.py](../../geosteering_ai/simulation/tests/sm_correlation.py) | 911 | **Service** (análise) + **View** (dialog) |
| [sm_benchmark.py](../../geosteering_ai/simulation/tests/sm_benchmark.py) | 830 | **Controller** (BenchmarkThread) + **Model** (BenchRecord) |
| [sm_model_gen.py](../../geosteering_ai/simulation/tests/sm_model_gen.py) | 670 | **Controller** (ModelGenerationThread) + **Model** (GenConfig) |
| [sm_layers_dialog.py](../../geosteering_ai/simulation/tests/sm_layers_dialog.py) | 424 | **View** |
| [sm_heartbeat.py](../../geosteering_ai/simulation/tests/sm_heartbeat.py) | 407 | **Service** (UI thread health) |
| [sm_qt_compat.py](../../geosteering_ai/simulation/tests/sm_qt_compat.py) | 298 | **Infra** (Qt adapter) |
| [sm_phase_timer.py](../../geosteering_ai/simulation/tests/sm_phase_timer.py) | 293 | **Service** |
| [sm_plot_cache.py](../../geosteering_ai/simulation/tests/sm_plot_cache.py) | 294 | **Model** (LRUPlotCache) |
| [sm_canonical_profiles.py](../../geosteering_ai/simulation/tests/sm_canonical_profiles.py) | 239 | **Model** (configs) |
| [sm_animation_bar.py](../../geosteering_ai/simulation/tests/sm_animation_bar.py) | 208 | **View** (widget) |
| [sm_toast.py](../../geosteering_ai/simulation/tests/sm_toast.py) | 197 | **View** (notificações) |
| [sm_widgets.py](../../geosteering_ai/simulation/tests/sm_widgets.py) | 195 | **View** (helpers) |
| [sm_dat_viewer.py](../../geosteering_ai/simulation/tests/sm_dat_viewer.py) | 180 | **View** + **Service** (parser .dat) |
| [sm_io.py](../../geosteering_ai/simulation/tests/sm_io.py) | 177 | **Service** (I/O) |
| [sm_snapshot_persist.py](../../geosteering_ai/simulation/tests/sm_snapshot_persist.py) | 139 | **Controller** (SnapshotPersistThread) |
| **Subtotal `sm_*.py`** | **9.337** | Distribuído em 17 arquivos |
| `simulation_manager.py` | 10.759 | Concentração problemática |
| **TOTAL GUI** | **20.096** | LOC totais |

**Observação crítica:** A pirâmide está invertida — a *agregação* (1 arquivo) é maior que a soma das *partes* (17 arquivos). Em arquiteturas saudáveis o orquestrador é fino.

### 1.3 Testabilidade — Estado Pós-v2.33

| Aspecto | Antes v2.33 | Após v2.33 (atual) | Após MVC (proposto) |
|:--------|:-----------:|:------------------:|:-------------------:|
| Cobertura GUI | ~0% | ~25% (16 testes) | 90%+ |
| Framework pytest-qt | ❌ | ✅ `pytest-qt>=4.4` em `[dev]` | ✅ expandido |
| Fixtures Qt | ❌ | ✅ `qt_binding`, `mock_simulation_thread`, `mock_sim_request` | ✅ + fixtures por controller |
| CI headless | ❌ | ✅ `xvfb-run -a pytest` | ✅ + visual regression |
| Marker `gui` | ❌ | ✅ | ✅ + marker `controller`, `view`, `model` |
| Golden path smoke | ❌ | ✅ Cenário E n=10 → simulation_finished | ✅ + cenários A,B,F,G,H |

---

## 2. Estado Atual da Decisão Arquitetural — O Que Já Foi Decidido

### 2.1 Cronologia das Decisões

| Data | Documento | Decisão |
|:-----|:----------|:--------|
| 2026-05-02 | [aprofundamento §50](arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md) | **Hexagonal + DDD escolhidos** como arquitetura central. MVC/MVVM marcado `⚠️ Apenas Studio GUI` (linha 7905) |
| 2026-05-02 | [aprofundamento §50.6](arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md) | Refactor Hexagonal **adiado para Sprints 25–28 pós-v3.0** |
| 2026-05-02 | [aprofundamento §42](arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md) | **Trilha A (pip-lib) + Trilha B (Studio)** — modelo Geosoft confirmado. Studio = repositório privado separado |
| 2026-05-11 | [v2.29.3_questions §5](v2.29.3_questions_2026-05-11.md) | **Esboço MVC concreto** (`gui/{models,views,controllers,services}/`); custo 5 sprints + ROI quantificado |
| 2026-05-13 | v2.33 (CHANGELOG) | **pytest-qt entregue** — desbloqueia refactor MVC |
| 2026-05-17 | [v2.39_proximos_passos §7](v2.39_proximos_passos_roadmap_2026-05-17.md) | **Complementariedade Hex⊕MVC** formalizada (citação literal §7.2) |
| 2026-05-18 | Hoje | Esta análise |

### 2.2 Citações-Chave Literais

**Sobre a complementariedade** — [`v2.39_proximos_passos_roadmap_2026-05-17.md`](v2.39_proximos_passos_roadmap_2026-05-17.md) §7.2:

> "A arquitetura Hexagonal define a **topologia do sistema** (o `geosteering_ai/` core é independente de qualquer UI). O MVC organiza a **estrutura interna do adapter GUI**. São complementares, não concorrentes."

**Sobre o escopo do MVC** — [`aprofundamento §50.1`](arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md) tabela de padrões:

> "MVC/MVVM | Adequado a GUI Studio mas não ao engine | ⚠️ Apenas Studio GUI"

**Sobre o adiamento** — [`aprofundamento §50.6`](arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md):

> "Refatoração para Hexagonal **NÃO acontece** em Fases 1-3 (impacto excessivo). Acontece em **Fase 4+ Sprint 25-28**. (…) Antes disso, **manter** estrutura atual flat. Hexagonal é alvo de longo prazo."

**Sobre o ROI** — [`v2.29.3_questions §5.5`](v2.29.3_questions_2026-05-11.md):

> "Onboarding novo dev ~2 sem → ~1 sem (-50%); Tempo nova feature ~3-4d → ~1-2d (-50%); Bugs regressão UI ~1-2/sprint → ~0-1/sprint (-50%); Testabilidade lógica 60% → 95% (+35%)"

### 2.3 Camadas Stratificadas (S1–S6) — Já Definidas

Do [`v2.39_proximos_passos_roadmap_2026-05-17.md`](v2.39_proximos_passos_roadmap_2026-05-17.md) §7.1:

```
┌──────────────────────────────────────────────────────────────────┐
│  S1 — Presentation Layer     (PyQt6 UI: MainWindow + Dialogs)    │
│  S2 — Controller Layer       (MVC: SessionController, etc.)      │
│  S3 — Service Layer          (StudioSession, InversionService)   │
│  S4 — Domain Layer           (geosteering_ai/ package — core)    │
│  S5 — Infrastructure Layer   (REST API, Colab, Cloud Storage)    │
│  S6 — Data Layer             (SQLite/PostgreSQL + File System)   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Análise de Viabilidade — 5 Eixos

### 3.1 Eixo Técnico — Viabilidade Estrutural

| Critério | Estado | Score | Justificativa |
|:---------|:-------|:----:|:--------------|
| Modularização-base | 17 `sm_*.py` separados | 🟢 8/10 | Foundation pronta; falta apenas **agregação** |
| Testabilidade | pytest-qt + 16 testes em v2.33 | 🟡 6/10 | Cobertura insuficiente para refactor seguro de monolito de 10.7k LOC |
| Domain layer | Espalhado em `models/`, `data/` | 🟡 5/10 | Hexagonal exige `geosteering_ai/domain/` (não criado) |
| Adapters | CLI ✅, REST ⏳, Notebook ✅ | 🟡 6/10 | I2.7 (REST API) **bloqueia** Studio |
| Acoplamento entre classes | Alto (Signal/Slot implícitos) | 🔴 4/10 | Inventário Signal/Slot inexistente |
| Inversão de dependência | Mínima | 🔴 3/10 | Widgets chamam `simulate_multi()` diretamente |
| **Subtotal** | | **5.3/10** | **Refactor é necessário mas não emergencial** |

### 3.2 Eixo Negócio — Studio como Produto Comercial

| Critério | Estado | Comentário |
|:---------|:-------|:-----------|
| Distribuição definida | **Trilha A pip-lib** (MIT/Apache, PyPI) + **Trilha B Studio** (privado, comercial) | Modelo Geosoft confirmado |
| Repositório Studio | `github.com/daniel-leal/geosteering-studio` (privado, depende pip-lib ≥2.0) | Ainda não criado |
| Cronograma | ALPHA 2026 Q4 · BETA 2027 Q1 · GA 2027 Q2 | ~6 meses até ALPHA |
| Stack Studio | PyQt6 + LiquidGlass + PyQtGraph + PyVista 3D | Diferente do SM (matplotlib only) |
| Features exclusivas Studio | Real-time WITSML · multi-poço · Petrel/SLB plugins · PyQtGraph 60fps | Justifica refactor MVC |
| Bloqueante crítico | **API REST I2.7 + Dockerfile I2.8** | Sem isso, Studio não pode existir |

**Veredito de Negócio**: O MVC é **pré-requisito qualitativo** (não bloqueante) para Studio ALPHA. Sem MVC, o Studio nasceria com débito desde o dia 1.

### 3.3 Eixo Token-Cost — Custo Cognitivo de LLM

```
Crescimento de simulation_manager.py:
  v2.21 (2026-05-02): 5.512 LOC  → ~44k tokens (8.8% de 500k)
  v2.29.3 (2026-05-11): 10.759 LOC  → ~80k tokens (16% de 500k)
  v2.45 (proj. 2026-06): ~14k LOC  → ~108k tokens (21.6% de 500k)
  v3.0 (proj. 2026-Q4): ~17k LOC  → ~135k tokens (27% de 500k)
```

**Impacto no Claude Code**:
- A cada `Read` completo do arquivo gasta ~16% do contexto disponível
- Subagentes Explore frequentemente precisam dele → multiplicam o custo
- Edits cirúrgicos exigem ler regiões de ±100 linhas (alívio parcial)

**Mitigação imediata (Fase 0)**: dividir `simulation_manager.py` em **8 sub-módulos** preservando interface pública — **independente** do refactor MVC completo.

### 3.4 Eixo Risco — Riscos de FAZER vs NÃO FAZER

| Risco | Probabilidade | Severidade | Mitigação |
|:------|:-------------:|:----------:|:----------|
| **Risco de fazer:** introduzir regressão funcional | Média | Alta | pytest-qt cobertura ≥80% antes do refactor + Strangler Fig |
| **Risco de fazer:** quebrar Signal/Slot implícitos | Alta | Média | Inventário Signal/Slot via novo hook `check-signal-orphans.sh` |
| **Risco de fazer:** atrasar features funcionais | Média | Média | Sprints intercalados (1 sprint MVC + 1 sprint feature) |
| **Risco de fazer:** double-maintenance durante migração | Alta | Baixa | Strangler Fig limita duplicação a ~3 sprints |
| **Risco de NÃO fazer:** Studio nasce com mesma dívida | Alta | Alta | — sem mitigação além de refactor |
| **Risco de NÃO fazer:** SM excede 20k LOC e fica impagável | Média | Alta | — perda de produtividade composta |
| **Risco de NÃO fazer:** token-cost limita uso do Opus 1M | Alta | Média | Mitigação parcial via split (Fase 0) |
| **Risco de NÃO fazer:** Onboarding de novos devs degrada | Média | Média | Documentação extensa parcialmente compensa |

### 3.5 Eixo Esforço — Dimensionamento

| Item | Sprints | Esforço | Pré-requisito |
|:-----|:-------:|:-------:|:--------------|
| pytest-qt setup | ✅ v2.33 | concluído | — |
| Cobertura pytest-qt ≥80% | 1 sprint | 1 sem | pytest-qt ✅ |
| Inventário Signal/Slot + skills `mvc-architect` + `qt-tester` | 0.5 sprint | 3 dias | — |
| Fase 0 — split físico (`simulation_manager/` package) | 1 sprint | 1 sem | cobertura ≥80% |
| Fase 1 — extrair Models | 1 sprint | 1 sem | Fase 0 |
| Fase 2 — extrair Controllers (Strangler) | 2 sprints | 2 sem | Fase 1 |
| Fase 3 — Service layer + Studio scaffolding | 1 sprint | 1 sem | API REST I2.7 ✅ |
| **TOTAL** | **5.5 sprints** | **~6.5 semanas** | |

---

## 4. Propostas de Melhorias — 9 Recomendações

### 4.1 (R1) **Estratégia Strangler Fig** ao invés de Big-Bang

**Problema**: O refactor MVC tradicional ("apague tudo e reescreva") é proibitivamente arriscado em 10.7k LOC de GUI com Signal/Slot implícitos.

**Proposta**: Aplicar o padrão *Strangler Fig* de Martin Fowler:

```
Fase 0 — Split físico (preserva interface pública):
   simulation_manager.py (10.759 LOC)
   ↓
   simulation_manager/  (package)
   ├── __init__.py          ← re-exporta tudo (compat 100%)
   ├── main_window.py       ← MainWindow apenas (~600 LOC)
   ├── pages/
   │   ├── parameters.py    ← ParametersPage (~1.220 LOC)
   │   ├── simulator.py     ← SimulatorPage (~725 LOC)
   │   ├── benchmark.py     ← BenchmarkPage (~475 LOC)
   │   ├── results.py       ← ResultsPage (~2.005 LOC)
   │   └── preferences.py   ← PreferencesPage (~890 LOC)
   ├── dialogs/
   │   ├── new_experiment.py
   │   ├── plot_composer.py
   │   ├── save_figure.py
   │   ├── config_d.py
   │   └── welcome.py
   └── state/
       ├── snapshot.py      ← SimulationSnapshot
       └── experiment.py    ← ExperimentState (376 LOC)
```

**Benefício imediato (sem refactor MVC)**:
- Token-cost: 80k → 8 × 10k (cada submódulo isolado)
- Testabilidade granular
- Code-review focado

**Compatibilidade**: `from geosteering_ai.simulation.tests.simulation_manager import MainWindow` continua funcionando via `__init__.py` re-export.

### 4.2 (R2) Promover `gui/` a Pacote First-Class

**Problema**: A GUI vive em `geosteering_ai/simulation/tests/` — semanticamente confuso (não é teste).

**Proposta**: Mover para `geosteering_ai/gui/` em movimento atômico:

```diff
- geosteering_ai/simulation/tests/simulation_manager.py
- geosteering_ai/simulation/tests/sm_*.py
+ geosteering_ai/gui/__init__.py
+ geosteering_ai/gui/simulation_manager/  (package — vide R1)
+ geosteering_ai/gui/widgets/  (ex sm_widgets, sm_animation_bar, sm_toast)
+ geosteering_ai/gui/plots/  (ex sm_plots, sm_plot_cache, sm_plot_backends/)
+ geosteering_ai/gui/threads/  (ex sm_workers, sm_model_gen, sm_snapshot_persist, sm_heartbeat, sm_phase_timer)
+ geosteering_ai/gui/dialogs/  (ex sm_layers_dialog, sm_dat_viewer)
+ geosteering_ai/gui/services/  (ex sm_io, sm_canonical_profiles)
+ geosteering_ai/gui/qt_compat.py  (ex sm_qt_compat)
+ tests/test_simulation_manager.py  (testes ficam onde devem ficar)
```

**Risco**: Imports espalhados — mitigar com `geosteering_ai/simulation/tests/__init__.py` que re-exporta de `gui.*` (período de transição 1 sprint).

### 4.3 (R3) **DDD Value Objects** Mapeados a Dataclasses Existentes

**Problema**: Linguagem ubíqua do DDD (§50.5) não é aplicada — `ExperimentState`, `SimRequest`, `BenchRecord` usam nomes técnicos.

**Proposta**: Manter nomes técnicos (compatibilidade) e adicionar **aliases DDD** + propriedades semânticas:

| Dataclass atual | Alias DDD | Localização |
|:---------------|:----------|:------------|
| `ExperimentState` | `WellSession` ou `SimulationSession` | `gui/models/session.py` |
| `SimulationSnapshot` | `SimulationOutcome` | `gui/models/outcome.py` |
| `SimRequest` | `SimulationRequest` (já é) → manter | `simulation/_workers.py` |
| `BenchRecord` | `BenchmarkMeasurement` | `gui/models/benchmark.py` |
| `GenConfig` | `ModelGenerationSpec` | `gui/models/generation.py` |
| `LRUPlotCache` | `PlotMemoizationCache` | `gui/models/plot_cache.py` |
| Modelo Numba (esp, rho_h, rho_v) | `ResistivityProfile` (§50.5) | `geosteering_ai/domain/profile.py` (NOVO) |

**Vantagem**: Studio importará `ResistivityProfile` do `domain/` sem precisar conhecer Numba/JAX. Pip-lib mantém código existente sem breaking change.

### 4.4 (R4) Camada **Service** como Ponte SM↔Studio

**Problema**: SM atual chama `simulate_multi()` diretamente; Studio fará o mesmo → duplicação de wiring (threading, error handling, progress reporting).

**Proposta**: Extrair `gui/services/`:

```
gui/services/
├── simulation_service.py  ← wrapper sobre simulate_multi() com:
│                             - progress signal
│                             - cancel/pause cooperativo (preserva v2.11)
│                             - error normalization
│                             - lazy import de Numba
│                             - integração com NumbaPrimer (v2.9)
├── plot_service.py        ← wrapper sobre matplotlib + cache:
│                             - LRUPlotCache (já existe)
│                             - render_async() para Studio (PyQtGraph)
│                             - export PNG/PDF/SVG unificado
├── benchmark_service.py   ← wrapper sobre Cenários A–H:
│                             - .claude/perf_baseline.json check
│                             - PerformanceBaseline alert
│                             - integração CI
├── file_service.py        ← wrapper sobre I/O:
│                             - .exp.json (SM)
│                             - .h5/.zarr (futuro, audit v2.39 menciona)
│                             - .las/.dlis (Studio)
└── inference_service.py   ← wrapper sobre InferencePipeline:
                              - SHARED com Studio (Trilha A→Trilha B)
```

**Ganho duplo**: SM ganha testabilidade imediata; Studio reusa Services do dia 0.

### 4.5 (R5) **Controllers Explícitos** — O Maior Débito MVC

**Problema**: Os Controllers **não existem hoje**. A lógica de orquestração vive nos próprios `QWidget`s (`SimulatorPage.start_simulation`, etc.).

**Proposta**: Criar `gui/controllers/` com Controllers thin (≤300 LOC cada):

| Controller | Responsabilidade | Observa | Comanda |
|:-----------|:-----------------|:--------|:--------|
| `MainController` | Coordenação cross-tab; lifecycle | `MainWindow`, todos os tabs | Service-layer geral |
| `SimulationController` | Iniciar/parar/pausar simulação | `SimulatorPage`, `SimulationThread` | `SimulationService` |
| `BenchmarkController` | Coordenar Cenários A–H | `BenchmarkPage`, `BenchmarkThread` | `BenchmarkService` |
| `ExperimentController` | Carregar/salvar .exp.json | `MainWindow.file_menu` | `FileService` |
| `PlotController` | Compor plots, exportar | `PlotComposerDialog`, `ResultsPage` | `PlotService` |
| `PreferencesController` | QSettings + cache config | `PreferencesPage` | — (direto QSettings) |
| `ModelGenController` | Estocástica + canônica | `LayersDialog`, `NewExperimentDialog` | — (sm_model_gen) |

**Padrão**: cada Controller recebe Service no construtor (injeção de dependência manual — sem framework necessário).

### 4.6 (R6) Nova Skill `geosteering-mvc-architect` (Opus 4.7)

**Problema**: Refactor MVC requer julgamento arquitetural recorrente; cada agente sem contexto MVC re-derivaria padrões.

**Proposta** (já mencionada em [v2.29.3_questions §5.6](v2.29.3_questions_2026-05-11.md), agora expandida):

```yaml
name: geosteering-mvc-architect
model: claude-opus-4-7
effort: extra-high
description: |
  Especialista em arquitetura MVC do GUI Simulation Manager e
  Geosteering AI Studio. Garante separação Model/View/Controller,
  invariantes de testabilidade Qt, e padrão Strangler Fig.

  Triggers: "MVC", "Controller", "ViewModel", "refactor GUI",
  "gui/", "simulation_manager refactor", "Studio architecture".
```

**Tarefas típicas**:
- Auditar se algum widget importa `simulate_multi()` direto (anti-padrão)
- Validar que Models não importam Qt (Qt-free dataclasses)
- Sugerir extração de Controller quando widget passa de 500 LOC
- Verificar que cada Controller tem ≥1 teste pytest-qt

### 4.7 (R7) Hooks Anti-Regressão MVC

**Proposta** — 3 hooks novos em `.claude/hooks/`:

| Hook | Evento | Função |
|:-----|:-------|:-------|
| `validate-mvc-separation.sh` | PreToolUse (Edit/Write em `gui/`) | BLOCK: imports `gui/models/*` → `gui/views/*` ou `simulation/_numba/*` |
| `check-signal-orphans.sh` | PostToolUse (Edit em `gui/`) | WARN: `QtCore.Signal` declarado sem `.emit()` ou `.connect()` (excluir whitelist) |
| `check-gui-test-coverage.sh` | Stop | WARN: novo Controller sem teste `tests/test_*_controller.py` |

### 4.8 (R8) Métricas de Qualidade MVC em `.claude/perf_baseline.json`

**Proposta**: Ampliar baseline (já existe para performance) com baseline arquitetural:

```jsonc
{
  "performance": { /* já existe */ },
  "architecture": {
    "simulation_manager_py_loc": { "current": 10759, "target": "<3000" },
    "monolithic_classes": { "current": 15, "target": "<5" },
    "controllers_count": { "current": 0, "target": ">=7" },
    "services_count": { "current": 0, "target": ">=5" },
    "pytest_qt_tests": { "current": 16, "target": ">=80" },
    "qt_coverage_pct": { "current": 25, "target": ">=80" },
    "token_cost_simulation_manager_k": { "current": 80, "target": "<25" }
  }
}
```

Hook `check-arch-regression.sh` falha CI se métricas pioram.

### 4.9 (R9) Documentação Arquitetural Específica — `docs/gui/`

**Proposta**: Criar `docs/gui/` com 5 documentos sob padrões D1–D14:

| Documento | Conteúdo |
|:----------|:---------|
| `docs/gui/architecture.md` | Diagrama MVC + camadas S1–S6 (já em §7.1 v2.39) |
| `docs/gui/signal_slot_inventory.md` | Inventário automático Signal/Slot via grep + parser |
| `docs/gui/migration_map.md` | Mapping antigo → novo (atualizado a cada refactor) |
| `docs/gui/pytest_qt_guide.md` | Padrões de teste pytest-qt para Controllers/Services/Views |
| `docs/gui/studio_handoff.md` | Como Studio (repo privado) consome `gui/services/` e `domain/` |

---

## 5. Roadmap Proposto — 3 Caminhos Alternativos

### 5.1 Caminho A — "Conservador" (alinha com roadmap atual)

```
v2.41–v2.45 (Jun 2026):  features funcionais (Catálogo Ruído, DTB, SurrogateNet, F5)
v2.46–v2.50 (Jul-Set 2026):  I2.7 API REST + I2.8 Dockerfile + MLOps
v3.0 (Out 2026):  major release; Studio scaffolding inicia
v3.0.1+ Sprints 25-28:  refactor MVC completo
```

**Pros**: alinhado ao `aprofundamento §50.6`; baixo risco.
**Contras**: `simulation_manager.py` pode chegar a 15k LOC; Studio nasce em paralelo ao refactor (sobrecarga).

### 5.2 Caminho B — "Strangler Antecipado" (recomendado)

```
v2.41 (Jun 2026):  Fase 0 — split físico simulation_manager.py em package
                   (1 sprint, preserva interface)
v2.42 (Jun 2026):  Mover gui/ para top-level + cobertura pytest-qt ≥80%
v2.43–v2.44 (Jul 2026):  I2.7 API REST + I2.8 Dockerfile (em paralelo)
v2.45 (Ago 2026):  Fase 1 — extrair Models + DDD Value Objects
v2.46–v2.47 (Set 2026):  Fase 2 — Controllers (Strangler 1 página por sprint)
v2.48 (Out 2026):  Fase 3 — Service layer consolidado
v3.0 (Out-Nov 2026):  major release; Studio escolhe Services prontas
```

**Pros**: Studio ALPHA (2026 Q4) inicia em arquitetura limpa; token-cost cai antes; cada fase é mergeable.
**Contras**: 8 sprints "arquiteturais" diluem capacidade para features; requer disciplina de Strangler.

### 5.3 Caminho C — "Big-Bang v3.0" (não recomendado)

```
v2.41–v2.49 (Jun-Out 2026):  apenas features funcionais
v3.0 (Nov 2026):  refactor completo MVC em 1 release massiva
```

**Pros**: Sem retrabalho intermediário.
**Contras**: 1 PR de ~15k LOC alteradas; risco extremo de regressão; pytest-qt golden path não cobre a totalidade; perda de momentum.

### 5.4 Quadro Comparativo

| Critério | A — Conservador | B — Strangler Antecipado | C — Big-Bang |
|:---------|:---------------:|:------------------------:|:------------:|
| Risco técnico | 🟢 Baixo | 🟡 Médio | 🔴 Alto |
| Tempo até Studio ALPHA limpo | 🔴 Alto (refactor pós-Studio) | 🟢 Baixo (refactor antes) | 🟡 Médio |
| Esforço total | 🟢 Distribuído | 🟡 Frontloaded | 🔴 Bloco único |
| Risco de regressão | 🟢 Baixo | 🟡 Baixo-Médio | 🔴 Alto |
| Velocity features durante refactor | 🟢 Mantida | 🟡 Reduzida 50% | 🔴 Zero |
| Token-cost relief | 🔴 Tarde | 🟢 Cedo | 🟡 Tarde |
| Alinhamento com `aprofundamento §50.6` | 🟢 Total | 🟡 Antecipa | 🟡 Antecipa |
| **Score ponderado** | **6.5/10** | **8.0/10** | **3.5/10** |

---

## 6. Análise de Aprimoramento do Projeto — O Que o MVC Habilita

### 6.1 Aprimoramentos Diretos (Curto Prazo)

| Aprimoramento | Como o MVC habilita | Métrica de impacto |
|:--------------|:--------------------|:-------------------|
| **Token-cost ↓ no Claude Code** | `simulation_manager.py` 10.7k LOC → 8× ~1.3k LOC | -80% por arquivo; -50% por interação |
| **Bug fix velocity ↑** | Controllers isolados; bug em UI ≠ bug em workflow | -50% MTTR (mean time to repair) |
| **Onboarding ↓** | Modelo de cabeças bem definido | -50% rampa para novos devs |
| **Test coverage GUI** | pytest-qt pode mockar Services | 25% → 85% |
| **Code review focado** | Diff por camada (M, V, ou C) | -40% tempo de review |
| **Reuse cross-app** | SM e Studio compartilham Services | 0% → ~60% reuso |

### 6.2 Aprimoramentos Estratégicos (Médio Prazo)

| Aprimoramento | Por que precisa de MVC? |
|:--------------|:------------------------|
| **Studio comercial (Trilha B)** | Sem MVC + Hexagonal, Studio duplica wiring; com MVC, Studio importa Services |
| **REST API I2.7** | API REST e Studio compartilham `InversionService`, `SimulationService` |
| **CLI estendido v2.41+** | `geosteering-cli invert` chama mesma `InversionService` do GUI |
| **Real-time WITSML (Studio)** | Service `WITSMLStreamService` plugs em controllers existentes |
| **Multi-poço (Studio)** | `MultiWellSession` model + `MultiWellController` orchestra; SM permanece single-well |
| **Plugin Petrel/SLB (Studio)** | Hexagonal `adapters/out/petrel.py`; SM e Studio ambos beneficiam |
| **Inversão 2D/3D futura (v3.x)** | Models `ResistivityProfile2D`, `ResistivityVolume3D` + Services dedicados |

### 6.3 Aprimoramentos Defensivos (Riscos Mitigados)

| Risco evitado | Como o MVC mitiga? |
|:--------------|:-------------------|
| **`simulation_manager.py` excede 20k LOC** | Split físico (Fase 0) impede crescimento concentrado |
| **Regressões de UI invisíveis** | pytest-qt + visual regression em PR |
| **Studio nasce com mesma dívida do SM** | Studio importa `gui/services/` em vez de re-escrever |
| **Multi-poço requer reescrever SM** | Multi-poço encaixa em Services existentes |
| **Migração JAX/Numba expõe internals para UI** | Services abstraem backend (UI não conhece JAX) |
| **Time-to-feature degrada com tempo** | Cada feature toca 2–3 arquivos ≤500 LOC |
| **Onboarding de dev junior demora 4+ semanas** | Estrutura MVC é vocabulário comum |

### 6.4 Aprimoramentos no Ecossistema Multi-Agente

A arquitetura multi-agente (Parte III/IV/V do aprofundamento) **se beneficia** de MVC:

| Agente | Benefício MVC |
|:-------|:--------------|
| `geosteering-orchestrator` | Pode delegar "edit em UI" vs "edit em service" para sub-agentes especializados |
| `geosteering-simulation-manager` (skill) | Documentação reflete pacote real, não monolito |
| **NOVO** `geosteering-mvc-architect` | Existe especificamente para garantir invariantes |
| **NOVO** `geosteering-qt-tester` | Especializado em pytest-qt + Signal/Slot |
| **NOVO** `geosteering-studio-planner` | Coordena features Studio vs SM |
| MCP `physics-validator` | Continua validando `domain/` (puro, sem Qt) |
| MCP `numba-profiler` | Continua validando `simulation/` (intocado) |

---

## 7. Conclusões e Recomendações Finais

### 7.1 Respostas Diretas à Pergunta do Usuário

**"Em que momento o projeto prevê a estruturação MVC?"**

- **Documentado:** Sprints 25–28 pós-v3.0 (Out-Nov 2026+) conforme [`aprofundamento §50.6`](arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md).
- **Esboço técnico já existe:** [`v2.29.3_questions §5.4`](v2.29.3_questions_2026-05-11.md) detalha estrutura `gui/{models,views,controllers,services}/`.
- **Pré-requisitos:** pytest-qt ✅ (v2.33) · API REST I2.7 ⏳ · Dockerfile I2.8 ⏳.
- **Complementaridade Hex⊕MVC:** confirmada literalmente em [`v2.39_proximos_passos §7.2`](v2.39_proximos_passos_roadmap_2026-05-17.md).

**"Disserte sobre as possibilidades"**

Três caminhos viáveis (§5.1–5.3), com **recomendação para Caminho B (Strangler Antecipado)**, distribuindo o refactor em 8 sprints v2.41–v2.48 (~6 meses).

**"Analise e investigue de modo detalhado a viabilidade"**

5 eixos analisados (§3.1–3.5): técnico 5.3/10, negócio favorável, token-cost crítico, risco médio mitigável, esforço ~6.5 semanas distribuídas.

### 7.2 Decisões de Engenharia (Não-Implementadas, Conforme Solicitado)

| Decisão | Recomendação | Justificativa |
|:--------|:-------------|:--------------|
| **Quando** | Caminho B — iniciar v2.41 (próxima sprint) | Token-cost crítico; Studio nasce em Q4 |
| **Como** | Strangler Fig em 4 fases (0, 1, 2, 3) | Big-bang é proibitivo em 10.7k LOC |
| **Onde** | Novo pacote `geosteering_ai/gui/` | Sair de `simulation/tests/` (semanticamente errado) |
| **O quê primeiro** | Fase 0 — split físico (preserva interface) | Risco zero; ganho imediato em token-cost |
| **Bloqueante crítico** | pytest-qt ≥80% antes da Fase 2 | Refactor sem rede de teste = regressão garantida |
| **Skills a criar** | `mvc-architect` (Opus 4.7), `qt-tester` (Sonnet 4.6), `studio-planner` (Opus 4.7) | Garantir invariantes; reduzir token-cost recorrente |
| **Hooks a criar** | `validate-mvc-separation`, `check-signal-orphans`, `check-gui-test-coverage` | Anti-regressão automática |
| **Métricas** | Ampliar `perf_baseline.json` com seção `architecture` | Quantificar débito ao longo do tempo |

### 7.3 Próximas Ações Sugeridas (Aguardando Decisão do Usuário)

1. **Aprovar/Negar Caminho B** (Strangler Antecipado v2.41–v2.48).
2. **Se aprovado**: criar branch `feat/v2.41-mvc-phase-0-split` e iniciar Fase 0 (split físico — risco zero).
3. **Em paralelo**: gerar skill `geosteering-mvc-architect` para acompanhar refactor.
4. **Documentar** decisão em [`docs/ARCHITECTURE_v2.md`](../ARCHITECTURE_v2.md) §76 (novo) — "MVC para GUI Adapter".
5. **Atualizar** [`docs/ROADMAP.md`](../ROADMAP.md) Sprints v2.41–v2.48 com escopo MVC concreto.

---

## 8. Referências

### 8.1 Documentos Internos Consultados

| Documento | Seção Relevante | Conteúdo |
|:----------|:----------------|:---------|
| [arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md](arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md) | §42, §50, §50.6 | Trilhas A+B · Hexagonal+DDD · plano sprint 25-28 |
| [v2.29.3_questions_2026-05-11.md](v2.29.3_questions_2026-05-11.md) | §5.1–§5.7 | Esboço MVC concreto · custo+ROI · skills propostas |
| [v2.39_proximos_passos_roadmap_2026-05-17.md](v2.39_proximos_passos_roadmap_2026-05-17.md) | §7.1, §7.2, §7.3 | Camadas S1–S6 · Hex⊕MVC · cronograma Studio |
| [sprints_v2_25_v2_27_planning.md](../reference/sprints_v2_25_v2_27_planning.md) | §347–§409 | Considerar MVC v2.27 · O5 GUI dependente |
| [CHANGELOG.md](../CHANGELOG.md) | v2.33, v2.40 | pytest-qt entregue · mp16 + colab-bridge |
| [ROADMAP.md](../ROADMAP.md) | Marcos v2.40 | Sprint atual v2.40 — pré-v2.41 |
| [CLAUDE.md](../../CLAUDE.md) | Identidade, Studio | Trilha A pip-lib + Trilha B Studio comercial |

### 8.2 Arquivos de Código Inspecionados

- [simulation_manager.py](../../geosteering_ai/simulation/tests/simulation_manager.py) (10.759 LOC, 15 classes Qt)
- [sm_workers.py](../../geosteering_ai/simulation/tests/sm_workers.py), [sm_plots.py](../../geosteering_ai/simulation/tests/sm_plots.py), [sm_benchmark.py](../../geosteering_ai/simulation/tests/sm_benchmark.py), [sm_correlation.py](../../geosteering_ai/simulation/tests/sm_correlation.py), [sm_model_gen.py](../../geosteering_ai/simulation/tests/sm_model_gen.py) (5 maiores módulos `sm_*.py`)
- Inventário completo dos 17 módulos `sm_*.py` (§1.2)

### 8.3 Padrões Arquiteturais Referenciados

| Padrão | Fonte | Aplicação |
|:-------|:------|:----------|
| Strangler Fig | Martin Fowler (2004) | Refactor incremental sem big-bang |
| Hexagonal (Ports & Adapters) | Alistair Cockburn (2005) | Topologia do sistema (já decidido §50) |
| DDD (Domain-Driven Design) | Eric Evans (2003) | Linguagem ubíqua (já decidido §50.5) |
| MVC | Trygve Reenskaug (1979) | Estrutura interna do adapter GUI |
| Injeção de dependência manual | Mark Seemann (2019) | Service → Controller via construtor |

---

**Fim do relatório.**

**Próximas ações:** Aguardar instruções do usuário sobre qual caminho (A, B ou C) prosseguir, e se deseja iniciar Fase 0 (split físico) ou criar primeiro a skill `geosteering-mvc-architect`.

**Status do branch:** `main` (limpo após v2.40); este relatório é o **único** artefato gerado nesta interação.
