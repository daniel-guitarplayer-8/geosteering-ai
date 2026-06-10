# Geosteering AI — Status de Portfólio dos 4 Produtos

**Data:** 2026-06-04
**Autor:** Daniel Leal (assistido por Claude Opus 4.8 + workflow multi-agente)
**Tipo:** Levantamento de status · Arquitetura de portfólio de produtos
**Escopo:** Levantamento completo e detalhado do status atual do Geosteering AI organizado pelos **4 produtos** do projeto, **acrescentando** (sem restringir) ao relatório [geosteering_ai_studio_estruturacao_requisitos_2026-06-04.md](geosteering_ai_studio_estruturacao_requisitos_2026-06-04.md).
**Status:** Investigação — não implementa código. Complementa o relatório de requisitos.
**Versão do projeto:** v2.56 · **Método:** Workflow multi-agente (7 agentes: 5 levantamentos por produto + síntese + crítica adversarial; ~489k tokens; 185 tool-uses) cruzado com leitura direta do código.

---

## Os 4 Produtos do Geosteering AI

| # | Produto | Natureza | Estado | Backends de simulação |
|:-:|:--------|:---------|:------:|:----------------------|
| **1** | **Biblioteca / API Python** | Módulos importáveis + API REST (FastAPI) | BETA avançado (~85%) | Numba + JAX (+ Fortran via paridade) |
| **2** | **CLI** (`geosteering-cli`) | Linha de comando | MVP (~60%) | Numba (default) + JAX (opt-in) |
| **3** | **Geosteering AI Studio** ⭐ | Software desktop **flagship** — absorve todos os recursos | **0% código** (backend ~90% pronto) | herda Numba + JAX (+ Fortran) |
| **4** | **Simulation Manager** | GUI **só de simulação** | Produção (~75%, com débito) | Numba + **Fortran** · **JAX ausente** |

> **Premissa do usuário (incorporada):** o **Studio é o produto principal**, que absorverá todos os recursos num único software. Os outros três são produtos próprios sobre a mesma biblioteca-núcleo. O **Simulation Manager é um produto separado e focado SÓ em simulação** (Numba JIT + Fortran + JAX GPU).

### Achados-chave deste levantamento (novos em relação ao relatório anterior)

| Achado | Evidência | Severidade |
|:-------|:----------|:----------:|
| **CLI não expõe `--backend auto`** — força `numba`\|`jax` manual; o dispatcher `simulate_batch("auto")` da lib não é exposto | `cli/_main.py:222` `choices=["numba","jax"]`, `default="numba"` vs `dispatch.py` `_resolve_backend` | **Alta** (maior ROI/menor custo do portfólio) |
| **Simulation Manager NÃO tem JAX GPU** — só Numba + Fortran | `sm_workers.py:185,1078` `backend="numba"` hardcoded; só `run_numba_chunk`/`run_fortran_chunk`; sem `run_jax_chunk` | **Alta** (gap vs. visão "Numba+Fortran+JAX") |
| **A CLI se auto-rotula como "Simulation Manager v2.56"** — confunde a fronteira de produto | `cli/_main.py:113` `SIMULATION_MANAGER_VERSION`; banner em `_main.py:581` | **Média** (confusão de produto) |
| **Studio é 0% de código** — não existe `geosteering_studio/` nem `geosteering_ai/gui/` | grep confirmou: zero `.py` com "studio"; `gui/` ausente | **Alta** (é o produto principal) |
| **Versão fragmentada** — raiz `2.0.0`, CLI `v2.56`, API `v2.39.0`, simulation `v1.6.0` | `__init__.py:63`, `_main.py:113`, `api/__init__.py:62` | **Média** |
| **Os 3 níveis (básica/intermediária/avançada) não têm definição mensurável** em código ou docs | grep retornou zero; este relatório os define (Parte IV) | **Média** (design a fixar) |

---

## Parte I — Visão de Portfólio

Os 4 produtos são **fachadas sobre uma única biblioteca-núcleo** (`geosteering_ai/`, ~81,6k LOC de backend). A biblioteca é a fundação; CLI, REST, Simulation Manager e Studio são interfaces sobre ela. **Ponto crítico:** os 3 produtos que simulam (CLI, SM, Studio-futuro) convergem no **mesmo caminho de computação** (`multi_forward.py` / `_numba`) — **não há duplicação de física, apenas de orquestração/UI**.

```
                 ┌─────────────────────────────────────────────────────────────┐
                 │   ⭐ PRODUTO 3 · GEOSTEERING AI STUDIO  (FLAGSHIP — 0% código) │
                 │   GUI Qt6 desktop · absorve TODOS os recursos num 1 software   │
                 │   5 perspectivas: Simulação · Treino · Inferência · Realtime · │
                 │   Registry  |  Fase 1 = sintético+sim · Fase 2 = dados reais   │
                 └───────┬───────────────┬───────────────┬───────────────┬───────┘
        orquestra (L2)   │               │               │   absorve infra Qt (Fase 0)
            ┌────────────┘               │               │               └────────────┐
            ▼                            ▼               ▼                             ▼
   ┌─────────────────┐        ┌──────────────────┐  ┌──────────────┐      ┌────────────────────────┐
   │ PRODUTO 2 · CLI │        │ PRODUTO 1 · API  │  │ PRODUTO 1 ·  │      │ PRODUTO 4 · SIMULATION │
   │ geosteering-cli │        │ REST (FastAPI)   │  │ Biblioteca   │      │ MANAGER (GUI Qt)        │
   │ simulate·bench  │        │ /health /predict │  │ importável   │      │ Numba+Fortran (JAX ✗)   │
   │ Numba✓ JAX✓     │        │ geosteering-api  │  │ ~80 exports  │      │ ~18,96k LOC em tests/   │
   │ auto ✗          │        │ (MVP sem auth)   │  │              │      │ (débito estrutural)     │
   └────────┬────────┘        └────────┬─────────┘  └──────┬───────┘      └───────────┬────────────┘
            │ import                   │ import            │ import                   │ qt_compat·plot
            └──────────────────────────┴───────────────────┴──────────────────────────┘ backends·threading
                                                  ▼ (todos consomem — MESMO multi_forward.py)
   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
   │     PRODUTO 1 · BIBLIOTECA-NÚCLEO  geosteering_ai/  (~81,6k LOC · BETA avançado · v2.0.0)      │
   │  config → data·simulation·models·noise → training·losses → inference·evaluation·visualization  │
   │  GAPS-núcleo: gui/ (ausente) · ingest/ WITSML·LAS·DLIS (ausente) · MLflow registry (ausente)   │
   └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Parte II — Status Detalhado por Produto

### Produto 1 — Biblioteca / API Python

**O que é:** o SDK importável (`import geosteering_ai`) + a API REST (FastAPI) que expõe a inferência por HTTP. É o produto que um desenvolvedor/pesquisador usa para programar.

**Superfície pública (importável):**

| Módulo | Exporta |
|:-------|:--------|
| `geosteering_ai` (topo) | `PipelineConfig`, `__version__`, ~80 símbolos em `__all__` |
| `.data` | `DataPipeline`, `PreparedData` |
| `.models` | `ModelRegistry`, `build_model`, `list_available_models`, `is_causal_compatible` |
| `.losses` | `LossFactory`, `build_loss_fn`, `VALID_LOSS_TYPES` |
| `.training` | `TrainingLoop`, `NStageTrainer`, `build_callbacks`, `build_metrics` |
| `.inference` | `InferencePipeline`, `RealtimeInference`, `export_saved_model/tflite/onnx`, `UncertaintyEstimator` |
| `.evaluation` | `compute_all_metrics`, `evaluate_predictions`, `compare_models`, `compare_modes` |
| `.visualization` | `plot_curtain`, `plot_dtb_profile`, `plot_picasso_dod`, `RealtimeMonitor`, … (14 plots) |
| `.simulation` | `simulate`, `simulate_batch` (dispatcher `auto`), `simulate_multi`, `simulate_multi_jax(_batched/_grouped)`, `compute_jacobian(_fd_numba/_jax)`, `SimulationConfig`, `FilterLoader` |
| `.noise` | `apply_noise_tf`, `CurriculumSchedule` |
| **API REST** | `GET /health`, `POST /predict` (22-col → ρ_h/ρ_v em Ω·m + MC Dropout); launcher `geosteering-api` |

**Maturidade:** BETA avançado (~85%). Núcleo funcional completo, arquitetura modular (Factory/Registry), presets YAML estáveis (`baseline`/`robusto`/`nstage`/`geosinais_p4`/`realtime`), extras pip (`[train]`,`[sim]`,`[viz]`,`[hpo]`,`[api]`,`[all]`,`[dev]`), type hints em ~89% dos arquivos.

**Gaps para virar SDK público:**
- **Versão hardcoded `2.0.0`** (`__init__.py:63`) que não evolui com merges; subcomponentes desalinhados (API `v2.39.0`, simulation `v1.6.0`).
- **Sem publicação PyPI**, sem `py.typed` (PEP 561), sem política de deprecação (PEP 387).
- Sem distinção formal API estável vs interna; sem `pytest-cov`/badge; sem exemplos/notebooks no repo.

---

### Produto 2 — CLI (`geosteering-cli`)

**O que é:** acesso por linha de comando aos simuladores, sem escrever Python. Entry points: `geosteering-cli`, `geosteering-warmup` (aquecimento JIT/LLVM), `geosteering-api` (launcher REST, separado).

**Superfície:**

| Comando | Função | Flags-chave |
|:--------|:-------|:------------|
| `simulate` | Gera N modelos sintéticos via `simulate_multi`/dispatcher | `--models --n-pos --frequencies --dips --tr-spacings --geometry --backend --dtype --jax-strategy --warmup --repeat --compare-backends --jax-chunk-size --out --format {npz,dat,none}` |
| `benchmark` | Throughput em mod/h | `--scenario {A..H}` (8 cenários, H=512 combos) `--n --list-scenarios` + flags de backend |
| `version` | Exibe `SIMULATION_MANAGER_VERSION` | — |
| `geosteering-warmup` | Aquece Numba+LLVM (síncrono) | `--verbose --version` |

**Maturidade:** MVP production-ready (~60%). Backends Numba (default) + JAX (opt-in) com **fallback gracioso** (JAX sem GPU → Numba/CPU com aviso) e pré-voo de agrupabilidade (v2.55, evita crash TLS). Observabilidade: tabela ASCII + JSON, wall-clock transparente (v2.56), comparação de backends lado-a-lado.

**Gaps (como produto):**
- 🔴 **`--backend auto` ausente** — `cli/_main.py:222` força `numba|jax`; o dispatcher `simulate_batch("auto")` (já testado na lib) **não é exposto**. O requisito do usuário ("CLI por padrão associada AOS DOIS simuladores Numba JIT E JAX GPU") pede exatamente `auto` como default.
- 🔴 **Sem comandos de treino/inferência/predict** — escopo atual = só simulação + benchmark; DL é exposto só via REST.
- 🔴 **Sem `generate-dataset`** persistente (modelos são randomizados in-memory).
- 🟡 **Banner se auto-rotula "Geosteering AI Simulation Manager v2.56"** (`_main.py:581`) — confunde a CLI com o produto SM; versão hardcoded.
- 🟡 Sem man pages, sem shell completion, sem `--log-level`.
- 🟡 **D9 (nunca `print`) qualificado:** a CLI usa `print()` em entry-points (`_main.py:155,180,581`, `warmup.py`) — aceitável em CLI user-facing, mas a alegação "D9-compliant" deve ser ressalvada.

---

### Produto 3 — Geosteering AI Studio (FLAGSHIP) ⭐

**O que é:** o software desktop principal que **absorve todos os recursos** (simulação, treino, inferência offline, geonavegação realtime, análise) num único app Qt6, lidando com **dados reais E sintéticos**. **Estado: 0% de código** — backend ~90% pronto; falta GUI/orquestração + ingestão de dado real.

Detalhado na Parte IV (fases, níveis de simulação, absorção). Resumo de estado:

| Componente | Implementação | Testes | Produção |
|:-----------|:-------------:|:------:|:--------:|
| Backend (`geosteering_ai/`) | 90% | ~alta | Sim (sintético) |
| Studio como pacote | **0%** | — | — |

---

### Produto 4 — Simulation Manager (GUI só de simulação)

**O que é:** GUI Qt6 especializada em **simulação EM 1D TIV** com backends **Numba JIT + Fortran (tatu.x)**, geração estocástica de modelos, benchmark Numba-vs-Fortran e visualização multi-backend. Lançado por `python -m geosteering_ai.simulation.tests.simulation_manager`.

**Superfície (4 abas):** `Simulador` (parâmetros, geração estocástica, seletor de backend, play/pause/resume/cancel), `Benchmark` (Config A/B/C + 30k, throughput/speedup/paridade), `Resultados` (galeria de snapshots, 4 backends de plot, composer/export), `Preferências` (caminhos, tema, cache LRU).

**Maturidade:** Produção (~75%, status "Produção" nos headers), mas com débito estrutural (vive em `tests/`, monólito 10,7k LOC, 25% cobertura pytest-qt).

| Capacidade | Status | Evidência |
|:-----------|:------:|:----------|
| Geração estocástica (Sobol/Halton/MT/Normal) | implementado | `sm_model_gen.py` (7 famílias RNG) |
| Execução paralela **Numba + Fortran** | implementado | `sm_workers.py` `run_numba_chunk`/`run_fortran_chunk`; ProcessPool efêmero v2.29 (anti-hang) |
| Benchmark Numba vs Fortran | implementado | `sm_benchmark.py` (6 perfis canônicos, `speedup_fortran_over_numba`) |
| Visualização 4 backends (mpl/pyqtgraph/plotly/vispy) | implementado | `sm_plot_backends/` (ABC + Factory) |
| Persistência `.exp.json` + cache LRU 500 MB | implementado | `sm_snapshot_persist.py`, `sm_plot_cache.py` |
| Visualizador `.dat` 22-col + correlação | implementado | `sm_dat_viewer.py`, `sm_correlation.py` |
| **Integração JAX GPU** | **ausente** | `sm_workers.py:185,1078` hardcoda `backend="numba"`; sem `run_jax_chunk` |

**Gap principal (vs. visão do usuário "Numba+Fortran+JAX"):** integrar **JAX GPU** ao SM. Esforço ~1–2 sprints: (1) remover hardcode `backend="numba"`; (2) `backend` em `SimRequest`; (3) `run_jax_chunk` (vmap/pmap via `simulate_batch`); (4) dropdown de backend na UI; (5) teste de paridade JAX vs Numba <1e-10. **Risco a mitigar:** a inicialização CUDA/TLS pode colidir com o ProcessPool efêmero (base da paridade) — o mesmo crash TLS que a CLI já teve de mitigar no pré-voo v2.55.

---

## Parte III — Matrizes de Portfólio

### III.1 Produto × Maturidade

| Produto | Existe hoje | Backend pronto? | GUI? | Distribuição | Gap #1 |
|:--------|:-----------:|:----------------|:-----|:-------------|:-------|
| **1 · Biblioteca/API** | ~85% | é o próprio backend | n/a | `pip install -e`; **PyPI ausente** | PyPI + `py.typed` + semver real |
| **1 · API REST** | ~40% (MVP) | `/health`+`/predict` | não (HTTP) | entry point `geosteering-api`, extra `[api]` | Sem auth/rate-limit |
| **2 · CLI** | ~60% (MVP) | `simulate`/`benchmark` | não (ASCII+JSON) | entry point `geosteering-cli` | **`--backend auto` ausente** |
| **3 · Studio** ⭐ | **~5%** (Fase 0/1: 0–10%) | ~90% reutilizável | **não existe** | sem instalador | GUI/orquestração inexistente |
| **4 · Simulation Manager** | ~75% (Produção+débito) | Numba+Fortran; **JAX ausente** | sim (Qt6, 4 abas) | **não pip-instalável** (`python -m …`) | Integração JAX GPU |

### III.2 Subsistema-backend × Produto

●=consumo direto · ◐=parcial/planejado · ○=não consome

| Subsistema (`geosteering_ai/`) | Lib | CLI | API | SM | Studio |
|:-------------------------------|:---:|:---:|:---:|:--:|:------:|
| **simulation/** (Numba+JAX+Fortran) | ● | ● | ○ | ● (JAX ◐) | ● |
| **models/** (48 arqs) | ● | ○ | ◐ | ○ | ● |
| **losses/** (26 + 8 PINN) | ● | ○ | ○ | ○ | ● |
| **training/** (Loop, N-Stage, Optuna) | ● | ○ (gap: sem `train`) | ○ | ○ | ● |
| **data/** (on-the-fly, FV/GS) | ● | ◐ (in-memory) | ◐ | ◐ (gen própria) | ● |
| **inference/** (offline + realtime) | ● | ○ (gap: sem `predict`) | ● | ○ | ● |
| **evaluation/** | ● | ○ | ○ | ◐ (paridade) | ● |
| **visualization/** (14 plots) | ● | ○ (ASCII) | ○ | ◐ (4 backends próprios) | ● |
| **noise/** (34 tipos) | ● | ○ | ○ | ○ | ● |
| **gui/** (qt_compat, plot_backends) | ○ **(ausente)** | ○ | ○ | ◐ (em `tests/`) | ● (após Fase 0) |
| **ingest/** (WITSML/LAS/DLIS) | ○ **(ausente)** | ○ | ○ | ○ | ◐ (bloqueador Fase 2) |

**Observação:** o Studio é o **único produto que consome TODOS os subsistemas** — daí seu papel de flagship absorvente. CLI/API/SM consomem fatias estreitas.

---

## Parte IV — O Studio como Flagship

### IV.1 Fase 1 (sintético + simulação) vs Fase 2 (dados reais)

| | **FASE 1 — Sintético + Simulação (MVP v0.1)** | **FASE 2 — Geonavegação Realtime em Poço Real (v0.2–v0.3)** |
|:--|:--|:--|
| **Objetivo** | Ciclo end-to-end em dado sintético, sem rig | Conectar a rig real, streaming → inversão causal |
| **Dados** | Simuladores TIV Numba/JAX + datasets sintéticos + replay | WITSML/ETP/WITS + LAS/DLIS de poço real |
| **Capacidades** | Gerar dataset · treinar (48 arqs, PINN, curriculum) · inferência offline · curtain/DTB/Picasso · UQ MC Dropout · relatórios · sessão | Bridge WITSML 2.0/ETP · WITS L0 · LAS/DLIS readers · MD↔TVD · `WellStateManager` · RealtimeInference incremental · alarmes · multi-poço · SLA latência medido |
| **Estado backend** | ~90% pronto (GUI/orquestração a construir) | Ingestão = **0** (gap #1); RealtimeInference existe mas re-roda janela inteira |
| **Bloqueador** | Fase 0 (refactor da GUI) | Módulo `ingest/` + validação de campo (Goliat) |

### IV.2 Os 3 níveis de simulação realtime — definição mensurável

> **Correção de honestidade (crítica):** estes níveis **não existiam definidos** em código nem docs. Abaixo a proposta concreta. **Multi-frequência JÁ está implementada no simulador** (`config.frequencies_hz`, `multi_forward.py`, CLI `--frequencies`) — o gap é apenas o *default escalar* do `PipelineConfig` de DL + a UI. **UQ calibrada P10/P50/P90 + CRPS/coverage, look-ahead e `WellStateManager` NÃO existem** — são o que define o nível Avançado e precisa ser construído.

| Dimensão | **Básico** | **Intermediário** | **Avançado** |
|:---------|:-----------|:------------------|:-------------|
| **Modelo** | ResNet-18 causal (Tier 1) | ResNet-18/Informer/Mamba causal | WaveNet/Causal-Transformer nativo |
| **Frequência** | Mono (20 kHz) | Mono (multi-freq disponível¹) | **Multi-freq nativa** (já no simulador¹) |
| **Physics (PINN)** | Não | Opcional (λ-slider) | **Obrigatória** (Maxwell/TIV/cross-grad/look-ahead²) |
| **UQ** | Nenhuma / CI naive | MC Dropout (CI 95%) | **P10/P50/P90 calibrada + CRPS/coverage** (a construir³) |
| **Curriculum** | Off | 3-phase | N-Stage (5 stages) + ρ-oversampling |
| **Backend** | Numba/JAX | JAX (preferido) | JAX obrigatória (A6000) |
| **Latência-alvo** | <100 ms | <60 ms | <50 ms (SLA a medir⁴) |
| **Dados** | Sintético/real pré-processado | Sint→Real (DomainAdapter) | Real validado (Goliat) |
| **Preset** | `geosteering-basic` | `geosteering-robust` | `geosteering-expert` |

¹ Implementado no simulador; falta UI + default não-escalar no pipeline DL. ² `look_ahead_loss` existe; motor de cenários não. ³ **Não existe** — diferenciador a construir. ⁴ Nunca medido fim-a-fim.

### IV.3 Mapa de absorção (como cada produto é integrado ao Studio)

| Produto | Grau de absorção no Studio | Como |
|:--------|:--------------------------:|:-----|
| **1 · Biblioteca** | **100%** (import puro) | Serviços L2 reexportam `simulation`/`models`/`training`/`inference`/`visualization` |
| **1 · API REST** | ~10% hoje, ~60% futuro | `InferenceService` dual-mode: local (`InferencePipeline`) ou remoto (`HttpClient /predict` edge no rig) |
| **2 · CLI** | ~20% hoje, ~50% futuro | Standalone p/ batch; futuro: perspectiva Simulação chama via subprocess; **compartilham `simulation.*`** |
| **4 · Simulation Manager** | **infra Qt 100%, arquitetura ~60%** | Strangler Fig: `qt_compat`, `plot_backends`, threading, persistência migram p/ `gui/` e viram o esqueleto do Studio |

---

## Parte V — Reconciliação Simulation Manager ↔ Studio (revisada 2026-06-05)

> **Esclarecimento do usuário (2026-06-05):** o Simulation Manager (SM) e o Geosteering AI Studio são **dois produtos totalmente diferentes**. O Studio deve possuir **todos os recursos** do SM, mas o SM deve trabalhar **apenas** com as simulações **Numba JIT, JAX GPU e Fortran**. A formulação anterior — "o código do SM é absorvido e vira o esqueleto do Studio" — era **imprecisa**; esta seção a corrige.

### V.1 Por que um monólito NÃO pode ser o "esqueleto" — e o que realmente é reaproveitado

O "SM", no nível de código, **não é um monólito único** — são **duas naturezas de código**:

| Natureza | Conteúdo | ~LOC | Reaproveitável? |
|:---------|:---------|:----:|:----------------|
| **(A) Infraestrutura já modular** (`sm_*.py`) | `sm_qt_compat` (binding PyQt6/PySide6), `sm_plot_backends/` (ABC + 4 backends, 60 fps), `sm_workers` (threading anti-hang v2.29 + ProcessPool), `sm_snapshot_persist` (persistência async), `sm_plot_cache` (LRU 500 MB), widgets/toast | ~8,2k | **Sim, diretamente** — já é código desacoplado e testado |
| **(B) Monólito de aplicação** (`simulation_manager.py`) | `MainWindow` + `*Page` (Parameters/Simulator/Benchmark/Results/Preferences) + diálogos + estado embutido + orquestração | ~10,7k | **Não como está** — UI+estado+I/O acoplados, sem MVVM |

**Resposta direta à sua pergunta:** um monólito **não pode** ser o esqueleto, e **não é**. O que é reaproveitável é a **infraestrutura (A)** — que já é modular — e os **padrões comprovados** que o SM resolveu ao longo de ~11 sprints (compat de binding Qt, 4 backends de plot, threading que não trava a UI, persistência async, cache LRU). O monólito (B) é a **casca de aplicação do próprio SM** e **não** é reusado como esqueleto do Studio.

O "esqueleto" correto é o pacote **compartilhado `geosteering_ai/gui/`** que **emerge** ao extrairmos a infra (A) de `simulation/tests/` para um pacote de 1ª classe. Construir/arrumar o SM é o que **força esse `gui/` a existir**; o Studio então **também** constrói sobre ele. O SM é o **primeiro consumidor / campo de provas** dessa fundação — não o esqueleto literal.

### V.2 Modelo correto: dois produtos separados, uma fundação compartilhada

```
                       geosteering_ai/  (biblioteca-núcleo)
                       ├── simulation/   (Numba + JAX + Fortran)        ◀── compartilhado
                       ├── models/ losses/ training/ data/ inference/ ...
                       └── gui/          (NOVO — extraído da infra sm_*.py) ◀── compartilhado
                              ▲                                   ▲
              importa SÓ      │                                   │   importa TUDO
        ┌─────────────────────┘                                   └────────────────────┐
   SIMULATION MANAGER (produto A)                          GEOSTEERING AI STUDIO (produto B, flagship)
   = gui/ + simulation/  APENAS                            = gui/ + biblioteca INTEIRA
   • Numba JIT · JAX GPU · Fortran                         • simulação + treino + inferência
   • simulação interativa + benchmark                      • + realtime + dados reais (Fase 2)
   • casca própria (monólito OU refatorada)                • casca PRÓPRIA com MVVM (dia 1)
```

| Aspecto | Simulation Manager | Geosteering AI Studio |
|:--------|:-------------------|:----------------------|
| **Produto** | Separado, focado | Separado, flagship |
| **Importa** | `gui/` + `simulation/` apenas | `gui/` + biblioteca inteira |
| **Backends** | Numba + Fortran (+ **JAX a integrar**) | herda os mesmos |
| **Escopo** | **SÓ simulação** (sem treino/inferência/realtime/dado real) | **Tudo**, incl. perspectiva de Simulação que reimplementa as features do SM sobre o `gui/` |
| **Casca (shell)** | `simulation_manager.py` (monólito próprio; refator opcional) | shell **novo** com MVVM (obrigatório — app multi-domínio) |

### V.3 O que é "absorvido" — com precisão

- **Absorvido de fato (reuso real):** a **infraestrutura Qt** (`gui/`, ex-`sm_*.py`) + o **backend de simulação** (`simulation/`). Ambos compartilhados por **importação**.
- **"Todos os recursos do SM" no Studio:** o Studio entrega as features de simulação do SM **reconstruindo a perspectiva de Simulação sobre o mesmo `gui/`** (o backend já faz o trabalho pesado) — **não** importando a aplicação SM.
- **NÃO absorvido:** a casca de aplicação do SM (`simulation_manager.py`) — cada produto tem a sua. O SM permanece um app **separado e focado em simulação**.

### V.4 Implicação prática (desbloqueia o cronograma)

Como o reuso real é a **infra `sm_*.py` (já modular)**, a "Fase 0" que o Studio precisa é **extrair `sm_*.py` → `geosteering_ai/gui/`** (movimentos de baixo risco) — **não** refatorar o monólito inteiro do SM. O `simulation_manager.py` pode **permanecer monolítico** (é um app focado, tolerável) ou ser refatorado de forma **independente** — **não bloqueia o Studio**. Isso é mais barato e desacopla o progresso do Studio do refactor do SM. (Correção em relação ao relatório de requisitos, que tratava o refactor MVVM do monólito inteiro como bloqueador único.)

### V.5 Prova de NÃO-duplicação de física (inalterada)

```
CLI      →  simulate_batch  →  dispatch  →  multi_forward.py  ┐
SM       →  run_numba_chunk →  simulate_multi  ───────────────┤→  MESMO compute path
Studio   →  SimulationService → simulate_batch → multi_forward ┘     (_numba / _jax)
```

Os três convergem no **mesmo `multi_forward.py`** — a duplicação é só de **orquestração/UI**, **nunca de física**. Esse ponto único é a fronteira biblioteca↔produtos.

---

## Parte VI — Roadmap por Produto

| Produto | Marcos |
|:--------|:-------|
| **1 · Biblioteca → SDK** | M1: PyPI + `py.typed` + tags semver reais (sincronizar subcomponentes) · M2: política de deprecação + API reference auto-gerada · M3: `pytest-cov` + multi-versão TF |
| **1 · API REST** | M1: hardening (X-API-Key, rate-limit, CORS restrito, pip-audit) · M2: `POST /simulate` (expor simulador) · M3: observabilidade (OpenTelemetry) |
| **2 · CLI** | **M1: `--backend auto`** (expor dispatcher; default `auto` herdando o pré-voo de geometria) · M2: comandos `train`/`infer`/`predict` · M3: `generate-dataset` + man pages + shell completion + renomear banner "Geosteering AI CLI" |
| **4 · Simulation Manager** | **M1: JAX GPU** (1–2 sprints — `run_jax_chunk`, dropdown UI, paridade) · M2: visualizador HD/relatórios · M3: coexistir com Studio (specialist sobre `gui/` compartilhado) |
| **3 · Studio** ⭐ | **Fase 0: refactor GUI** (extrair `tests/`→`gui/`, MVVM, pytest-qt ≥80%) — bloqueadora · **Fase 1: MVP offline sintético** (5 perspectivas) · Fase 1.5: instaladores conda/constructor · **Fase 2: realtime + dados reais** (`ingest/`, WellStateManager, validação Goliat) |

---

## Parte VII — Correções de Honestidade (da crítica adversarial)

Verificadas no código; aplicam-se também ao relatório anterior onde indicado.

| # | Correção | Detalhe |
|:--:|:--|:--|
| 1 | **Multi-frequência JÁ é implementada** | `config.py:373` `frequencies_hz: Optional[List[float]]`; `multi_forward.py:667`; CLI `--frequencies`. O gap é só **UI + default escalar `frequency_hz`** no pipeline DL — **não** o simulador. (Corrige RF-A06 e os "pilares A/C" do relatório anterior.) |
| 2 | **Paridade é `<1e-12`, não `<1e-10`** | A errata sagrada Fortran é `<1e-12` (164/164 PASS). O `<1e-10` é a tolerância **JAX-vs-Numba**, não Fortran. Padronizar para não confundir. |
| 3 | **Throughput T4/A100 está deprecado (v2.44)** | Baseline canônica = **A6000 local** (`jax_gpu_a6000_v244`). Citar "200k mod/h T4" como maturidade está desalinhado; usar número A6000 ou marcar T4 como histórico. |
| 4 | **CLI tem 8 cenários (A–H)**, não A–G | `_main.py:440` `choices=["A","B","C","D","E","F","G","H"]`; H = estresse 512 combos. |
| 5 | **`geosteering-warmup` é aquecimento JIT/LLVM**, não validação | `cli/warmup.py` (síncrono, bloqueia até aquecer). |
| 6 | **UQ "P10/P50/P90 calibrada" não está pronta** | MC Dropout/Ensemble existem; **percentis calibrados + CRPS/coverage + MDN não**. Rotular como **alvo**, não "pronto" (consistente com a Parte VII do relatório anterior). |
| 7 | **CLI se rotula como "Simulation Manager"** | Renomear banner para "Geosteering AI CLI" — a CLI **não** é o SM; reduz confusão de fronteira de produto. |

---

## Parte VIII — Decisões Abertas (requerem o usuário)

Complementam as Q1–Q8 do relatório de requisitos, com foco no portfólio:

| # | Decisão | Impacto |
|:--:|:--|:--|
| **QP1** | **Esquema único de versionamento** (lib semver 2.x vs CLI/SM v2.56) — sincronizar `SIMULATION_MANAGER_VERSION` com `pyproject`/git tags | Pré-requisito do Studio; resolve confusão de produto |
| **QP2** | **`--backend auto` como default da CLI** — herda o pré-voo de agrupabilidade (anti-crash TLS) ou mantém `numba` seguro em CPU/CI? | Alinha CLI ao requisito "ambos por padrão" |
| **QP3** | **Sequência Fase 0** — refactor MVVM do SM (`tests/`→`gui/`) **precede** o Studio, ou começam em paralelo com risco de fork de código Qt? | Cronograma de SM e Studio |
| **QP4** | **JAX GPU no SM agora ou junto com o `gui/` compartilhado?** | O SM ganha JAX antes ou depois da Fase 0? |
| **QP5** | **Distribuição GUI** — criar extra `[gui]` + instalador desktop é marco do SM ou só do Studio? Hoje nenhum produto GUI é pip-instalável | Empacotamento |
| **QP6** | **Definição oficial dos 3 níveis** (básica/intermediária/avançada) — adotar os critérios mensuráveis da Parte IV.2? | Vira spec do Studio |

---

## Parte IX — Recomendações

1. **Maior ROI / menor custo:** expor **`--backend auto`** na CLI (~10 linhas; a árvore `dispatch._resolve_backend` já existe e é testada). Fecha o gap mais citado e alinha CLI↔biblioteca↔Studio numa decisão de backend.
2. **Integrar JAX GPU ao Simulation Manager** (M1, 1–2 sprints) — completa a visão "Numba+Fortran+JAX"; mitigar colisão CUDA/TLS com o ProcessPool (precedente: pré-voo v2.55).
3. **Fase 0 antes de qualquer linha do Studio** — Strangler Fig de `simulation/tests/` → `geosteering_ai/gui/` (precedente concreto: `sm_io.py` virou shim em v2.53). Sem isso, o Studio copia código (dívida dupla) ou trava.
4. **Unificar versionamento e nomenclatura** — derivar `SIMULATION_MANAGER_VERSION` de `importlib.metadata`; renomear o banner da CLI para "Geosteering AI CLI".
5. **Adotar os 3 níveis mensuráveis** (IV.2) e expor honestamente que "Avançado" depende de software a construir (UQ calibrada, look-ahead, WellStateManager).
6. **Manter `multi_forward.py` como ponto único** de física — a fronteira biblioteca↔produtos; qualquer refactor preserva a não-duplicação.

---

## Anexo — Conexão com o relatório anterior

Este documento **acrescenta** a visão de portfólio de 4 produtos ao [relatório de requisitos](geosteering_ai_studio_estruturacao_requisitos_2026-06-04.md), **sem restringir**. Ajustes de reconciliação:
- O relatório anterior tratou o **SM como "semente a ser promovida"**. Aqui isso é **precisado**: o **código** do SM é a semente (absorvido via Strangler Fig → esqueleto do Studio); o **produto-conceito** SM sobrevive como modo "simulação-only" (Parte V).
- O relatório anterior listou **multi-frequência como gap principal** (pilares A/C); aqui isso é **corrigido**: o simulador já a implementa; o gap é só UI + default escalar do pipeline DL (Parte VII.1).
- Mantêm-se válidos: Fase 0 bloqueadora, ingestão `ingest/` ausente como bloqueador da Fase 2, validação de campo como gate de credibilidade, e as decisões abertas Q1–Q8.

*Documento de análise exploratória. Não modifica `ROADMAP.md`/`CHANGELOG.md` — a formalização da Trilha G e dos produtos aguarda decisão (Parte VIII), conforme ADR-0001 R1.*
