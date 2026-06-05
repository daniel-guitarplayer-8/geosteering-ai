# Roadmap SDD — Geosteering AI (etapas em ondas)

**Data:** 2026-06-05 · **TODAS as etapas seguem SDD** (`spec.md → plan.md → tasks.md` antes de
qualquer código). Este roadmap encaixa o SDD entre `docs/ROADMAP.md §0` (SSoT) e os snapshots de
sprint (ADR-0001 R1/R2). Fila ordenada e índice em [INDEX.md](INDEX.md). Princípios em
[CONSTITUTION.md](CONSTITUTION.md). Arquitetura em [`../docs/architecture/`](../docs/architecture/).

---

## 0. Princípio de Sequenciamento (lido da revisão crítica)

> **O Studio é a estrela-guia, mas seu caminho crítico passa pela fundação.** Sequenciar por
> **dependência técnica**, não por desejo comercial. Este é um projeto de alta cadência com
> capacidade concentrada — o roadmap escolhe **um foco por onda** e protege a fundação compartilhada.

**Ordem tecnicamente forçada (não negociável por prazo):**

```
(1) ScalerRegistry + manifest        → reprodutibilidade de TUDO (pré-requisito de golden-tests)
        │
(2) registry/ModelStore + golden     → governança do que JÁ existe
        │
(3) gui/ extraído (Strangler Fig)    → desbloqueia SM modular (menor escopo) E Studio
        │
(4) ingest/ + calibration.py + G5    → desbloqueia campo real (credibilidade #1)
```

O **Simulation Manager modular (SM sobre `gui/`)** é o **entregável-foco de menor risco** que
ainda exercita TODA a fundação `gui/` — é o caminho natural antes do Studio ALPHA completo. O
Studio nasce sobre a mesma fundação, logo em seguida, com as perspectivas adicionadas
incrementalmente.

---

## 1. Diagrama de Fases × 4 Trilhas de Produto

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│  FASE 0        FASE 1          FASE 2           FASE 3            FASE 4                         │
│  FUNDAÇÃO      MVP             REALTIME         PRODUTO          DIFERENCIADORES                │
│  (desbloqueio) (sintético)     (causal/UQ)      (campo real)     (governança científica)        │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Lib/API ─► 0001 SDD ───► 0008 SDK semver ─► 0014 ONNX/TFLite ─► 0019 ingest LAS ─► 0024 MLflow  │
│            0002 SSoT      py.typed/tiers      + golden (G1/G2b)   0019b WITSML       ModelStore   │
│                          0009 API /v1 auth   0015 manifest+      0020 calibration   0026 G5      │
│                                              ScalerRegistry      CRPS/coverage                   │
│ CLI ─────► 0003 --backend auto ─► 0010 train/infer ─► (consome SDK) ─► (consome ingest) ──────►  │
│ SM ──────► 0004 EXTRAI gui/ ──► 0011 SM sobre gui/ ─► 0012 SM JAX GPU ─► (estável) ───────────►  │
│  └─FUNDAÇÃO 0005 MVVM base       casca + 1 perspect.   dispatch auto                             │
│    gui/     0006 plot_backends   (Simulação)                                                     │
│             0007 persist .session                                                               │
│ Studio ──┴─► (espera gui/) ──► 0013 Studio ALPHA ─► 0016 Pers.Treino ─► 0021 Pers.Realtime ───► │
│                                 casca + Pers.Simul.   0017 Pers.Inferên.  0022 WellState         │
│                                 (Perspective ABC)     0020 UQ calibrada   0025 instaladores      │
│                                                       0018 .gsproj         0026 validação G5     │
│ ═══ GATES TRANSVERSAIS (todas as ondas) ═══                                                     │
│ G-PHYS: paridade Fortran <1e-12 (alvo validation_portable) ─ G-QT: pytest-qt 25→40→60→80%       │
│ G-GOLD: golden modelo rtol<1e-6 ─ G-SEC: gitleaks+pip-audit+bandit ─ G-PERF: A6000 nightly      │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

GARGALO: 0004 (extrair gui/) desbloqueia SM modular E Studio simultaneamente.
Física NUNCA duplicada: todos convergem em simulation/multi_forward.py + dispatch.py:94.
```

---

## 2. Capacidade, Risco e Cortes (da revisão crítica — ler antes de prometer prazos)

| Item | Realidade | Decisão |
|:--|:--|:--|
| **Escopo total** | 4 produtos; gaps somados (gui/ + ingest/ 4 formatos + registry/MLflow + calibration + Studio 0%→ALPHA + 5 workflows CI + ScalerRegistry + golden INT8) = **anos** de trabalho concentrado | Foco **um produto por onda**; SM modular é o foco de 2026 |
| **Studio ALPHA Q4 2026** | 0% de código, depende de `gui/` + MVVM ainda não iniciados — **maior risco de cronograma** | **Rebaixar** de "comercial Q4 2026" para "após SM modular validado"; ALPHA realista em 2027 |
| **Validação científica** | G5 depende de `ingest/` (ausente) que depende de PWLS+MD↔TVD (ausentes) — campo real está a **2 níveis** de dependência | "PRODUCTION" exige dataset **representativo da geologia-alvo** |
| **Volve ≠ geologia-alvo** | Volve é arenito (Mar do Norte); o caso real provável é pré-sal/carbonatos. Domain shift **duplo** (sintético→real + Volve→pré-sal) | Aquisição de dataset de campo representativo = **item de prioridade máxima, NÃO-software** |
| **Sem dataset de campo** | G5 cai só em Volve (insuficiente) | Produto fica **TECH PREVIEW** com disclaimer human-in-the-loop, **não** "PRODUCTION" |
| **MLflow on-prem disco** | cada ModelVersion = SavedModel+ONNX+TFLite+scaler+golden×N_freq | Definir footprint + **cota+GC automático ANTES** de ligar autolog Optuna |

---

## 3. Milestones e Gates de Aceite (BLOQUEANTES)

| Milestone | Onda | Gate de aceite |
|:--|:--:|:--|
| **M0 — SDD vivo + Fundação `gui/`** | O0–O1 | `specs/0001` mergeada; `gui/` extraído e importável por SM e Studio; `gui/` **nunca** importado por `models/losses/training/...` (hook de fronteira); `core` nunca importa Qt; CLI `--backend auto` (choice); SSoT de versão única; **bump semver 2.0.0→2.56.0**; paridade Fortran <1e-12 BLOQUEANTE server-side (alvo `validation_portable`) e inalterada |
| **M1 — MVP 4 produtos (sintético)** | O2–O3 | CLI `train`/`infer`; SM roda sobre `gui/` (1 perspectiva) + JAX GPU; Studio ALPHA abre+simula+plota (Perspective ABC); SDK `py.typed`+tiers+`mypy --strict`; API `/v1`+X-API-Key+jobs 202; pytest-qt ≥40%; G-SEC verde |
| **M2 — Realtime + UQ + golden** | O4–O5 | Studio perspectivas Treino+Inferência+Realtime; golden G1 (rtol<1e-6) + G2b (INT8 Δρ_RMSE<2%/Δaten<0.5dB); UQ P10/P50/P90 coverage P90≥0.88 (G3); ONNX+TFLite congelados; `manifest.py`+`ScalerRegistry` completos; drift KS/PSI sobre Hzz; pytest-qt ≥60% |
| **M3 — Produto campo real** | O6–O7 | `ingest/` LAS→WITSML 22-col fail-fast errata + MD↔TVD; `.gsproj` durável (sem pickle, `SecureArchiveReader`); `WellStateManager`; `registry/` MLflow+autolog; instaladores conda Win/Linux; `release.yml` por tag; pytest-qt ≥80% BLOQUEANTE; **G5 Volve (ADR de aceite) → PRODUCTION**; sem G5 → TECH PREVIEW |

---

## 4. Paralelização por Onda

| Onda | Lib/API | CLI | SM | Studio | Plataforma/MLOps |
|:--:|:--|:--|:--|:--|:--|
| **O0** Bootstrap | spec 0001 + SSoT 0002 | spec 0003 | spec 0004 (extração `gui/`) | aguarda `gui/` | `check-spec-completeness.sh`; decompor `ci.yml` em 5 workflows; G-SEC; alvo `validation_portable` |
| **O1** Fundação `gui/` | impl 0002 (`importlib.metadata` + bump 2.56.0) | impl 0003 (choice `auto`, default `numba`+DeprecationWarning) | **impl 0004+0005+0006+0007** (extrai `sm_*.py`→`gui/`; MVVM base; .session atômico) | scaffold `apps/studio` (vazio) | fortran-parity BLOQUEANTE server-side; G-QT 25% |
| **O2** MVP-A | impl 0008 (SDK tiers + `py.typed`) | impl 0010 (`train`/`infer`) | impl 0011 (SM sobre `gui/`, 1 perspectiva) | **impl 0013 Studio ALPHA** (Perspective ABC + Pers.Simulação) | golden scaffold `test_golden_models.py` |
| **O3** MVP-B | impl 0009 (API `/v1`+X-API-Key+jobs 202) | (consome SDK) | impl 0012 (**SM JAX GPU**) | impl 0016 (Pers.Treino) | MLflow scaffold; cota+GC disco |
| **O4** Realtime | impl 0014 (ONNX+TFLite INT8+golden G1/G2b) | (golden via CLI) | estabilização | impl 0017 (Pers.Inferência: UQ) | impl 0020 `calibration.py` (gate G3); impl 0024 ModelStore+autolog |
| **O5** Realtime+UQ | impl 0015 (`manifest.py`+`ScalerRegistry`) | — | — | impl 0021 (Pers.Realtime 60fps); 0022 WellState | G-QT 60%; golden multi-freq; drift KS/PSI |
| **O6** Campo real-A | impl 0019 (`ingest/` LAS+errata+MD↔TVD) | (`infer` aceita LAS) | — | impl 0018 (`.gsproj` sem pickle) | conda-constructor scaffold |
| **O7** Campo real-B | impl 0019b (WITSML 2.0/ETP) | — | — | impl 0025 (instaladores) + 0026 (Volve G5) | G-QT 80% BLOQUEANTE; G5 PRODUCTION vs TECH PREVIEW |

**Serialização (SDD):** specs que tocam `gui/` ou `multi_forward.py` adquirem lock
(`acquire-lock.sh`); specs puramente CLI/API/Lib paralelizam livremente.

---

## 5. Decisões Concretas Embutidas

1. **`gui/` in-tree, extra opcional `[gui]`** — `core` nunca importa Qt; `gui/` nunca importado pelo núcleo de DL. Binding default em ADR (PySide6 LGPL recomendado p/ Studio comercial; `qt_compat` suporta ambos).
2. **CLI `--backend auto` é MINOR** — amplia `choices` em `cli/_main.py:228`; **mantém `default="numba"` por 1 minor com `DeprecationWarning`** (PEP 387) antes de mudar default para `auto`.
3. **SSoT de versão** = `importlib.metadata.version('geosteering-ai')`, **após bump 2.0.0→2.56.0**; `GEOSTEERING_CLI_VERSION`→`__sprint_tag__` derivado do semver.
4. **`ModelStore` (pesos treinados, MLflow) ≠ `ModelRegistry` (48 arquiteturas)** — NÃO renomear o existente; criar `registry/model_store.py`. **`ScalerRegistry` é dependência dura** (spec 0015), não questão aberta.
5. **G5 depende de `ingest/`** — todo modelo fica TECH PREVIEW até `ingest/` existir.
6. **`.gsproj` = zip+JSON com `schema_version`, PROIBIDO pickle** + `SecureArchiveReader` (anti path-traversal/zip-bomb); `.gsproj ⊇ .session`.
7. **Trajetória (MD↔TVD) é entidade de domínio de 1ª classe** (editável, versionada no `.gsproj`), não detalhe de `ingest/`.
8. **Três tolerâncias** (Princípio I): física <1e-12 · JAX-Numba <1e-10 · inferência rtol<1e-6.

---

## 6. Questões Abertas → ADR antes da Onda O2

Ratificar como `docs/decisions/ADR-XXXX.md` (lista completa em [`../docs/architecture/README.md §4`](../docs/architecture/README.md)):
binding Qt (D-A) · licenciamento/repo (D-B) · prioridade `gui/`×`ingest/` (D-C) · backend MLflow (D-D) ·
alvo WITSML (D-E) · métrica do gate G5 (D-F) · determinismo golden G1 (D-G) · rampa pytest-qt (D-H).

---

## 7. Próximo passo no fluxo guiado

A **spec semente** [`0003-cli-backend-auto/spec.md`](0003-cli-backend-auto/spec.md) já está escrita
(menor risco, 1 produto, não toca fundação). O fluxo guiado a seguir:

1. **Você revisa** a spec 0003 (resolve qualquer `[NEEDS CLARIFICATION]`) → GATE-S.
2. **Escrevemos** `0003/plan.md` (HOW) → GATE-P.
3. **Escrevemos** `0003/tasks.md` (decomposição) → GATE-T.
4. **Implementamos** tarefa-a-tarefa com commits atômicos → GATE-I → GATE-V → merge.

Em paralelo, a spec keystone [`0004-gui-foundation`](0004-gui-foundation/) (extração do `gui/`) é o
gargalo da Fase 0 — recomendado escrevê-la logo após validar o processo com a 0003.
