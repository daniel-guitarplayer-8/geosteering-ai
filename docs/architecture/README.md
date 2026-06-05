# Arquitetura do Geosteering AI — Blueprint Mestre

**Versão:** 1.0 · **Data:** 2026-06-05 · **Autor:** Daniel Leal (assistido por Claude Opus 4.8 + workflow multi-agente de 10 agentes + revisão adversarial)
**Status:** Blueprint de design — base canônica para o desenvolvimento via **Spec-Driven Development (SDD)**.

> Este diretório é o **design completo da arquitetura** da suite Geosteering AI (IA/DL para perfilagem de poços, geosteering em tempo real, simulação, geofísica e petrofísica). Ele complementa, sem restringir, os relatórios [geosteering_ai_portfolio_4_produtos_status_2026-06-04.md](../reports/geosteering_ai_portfolio_4_produtos_status_2026-06-04.md) e [geosteering_ai_studio_estruturacao_requisitos_2026-06-04.md](../reports/geosteering_ai_studio_estruturacao_requisitos_2026-06-04.md). O **processo** de desenvolvimento (SDD) vive em [`../../specs/`](../../specs/).

---

## Índice dos Componentes

| # | Documento | Cobre |
|:-:|:----------|:------|
| 1 | [01_system_architecture.md](01_system_architecture.md) | Monorepo, camadas, pacotes, fundação compartilhada `gui/`+`simulation/`, topologia de deployment, stack, ADRs |
| 2 | [02_data_flow.md](02_data_flow.md) | 5 fluxos (geração de dataset, treino, inferência offline, realtime streaming, ingestão real), contratos de dados |
| 3 | [03_api_design.md](03_api_design.md) | 3 superfícies: SDK Python (tiers/semver/py.typed), REST `/v1` (auth, jobs), CLI (`--backend auto`, train/infer) |
| 4 | [04_ui_ux_mvvm.md](04_ui_ux_mvvm.md) | Pacote `gui/` (MVVM), apps SM e Studio, perspectivas-plugin, threading, design system, testabilidade |
| 5 | [05_cicd.md](05_cicd.md) | Pipelines por produto, gates (paridade, pytest-qt, golden, segurança), release automation, branch strategy |
| 6 | [06_mlops.md](06_mlops.md) | Model registry/lineage (MLflow), golden-tests, runtime (ONNX/TFLite), drift, governança, validação científica |
| 7 | [07_cross_cutting.md](07_cross_cutting.md) | Segurança (OT/IEC 62443), observabilidade, estratégia de testes, config/estado, erro/resiliência, D1–D14 |

**Processo (SDD):** [`../../specs/CONSTITUTION.md`](../../specs/CONSTITUTION.md) (princípios invioláveis) · [`../../specs/README.md`](../../specs/README.md) (o fluxo) · [`../../specs/ROADMAP.md`](../../specs/ROADMAP.md) (etapas) · [`../../specs/templates/`](../../specs/templates/) (spec/plan/tasks).

---

## 1. Visão Executiva

O Geosteering AI é uma **família de 4 produtos sobre uma biblioteca-núcleo única** (`geosteering_ai/`, ~81,6k LOC, ~90% pronto). A física de forward (`simulation/multi_forward.py`) é o **ponto de convergência único** — todos os produtos passam por ele; **a física nunca é duplicada**.

```
                       ┌──────────────────────────────────────────────┐
                       │  ⭐ STUDIO (flagship) — gui/ + biblioteca      │
                       │     casca MVVM própria · 5 perspectivas        │
                       └───────┬───────────────┬───────────────┬───────┘
   importa lib  ┌──────────────┘     importa   │   importa     └──────────────┐ importa
        ▼       │                gui/+simulation│   lib                        │ lib
  ┌───────────┐ │              ┌────────────────▼─┐                  ┌──────────▼──────────┐
  │ CLI       │ │              │ SIMULATION MANAGER│                  │ BIBLIOTECA / API     │
  │ geosteer- │ │              │ SÓ simulação      │                  │ (importável + REST)  │
  │ ing-cli   │ │              │ Numba·JAX·Fortran │                  │                      │
  └─────┬─────┘ │              └─────────┬─────────┘                  └──────────┬───────────┘
        └───────┴────────────────────────┴──────── todos consomem ──────────────┘
                                          ▼
        ┌────────────────────────────────────────────────────────────────────────┐
        │  FUNDAÇÃO COMPARTILHADA                                                  │
        │  geosteering_ai/gui/  (infra Qt extraída de sm_*.py) [NOVO]              │
        │  geosteering_ai/simulation/  (multi_forward.py — CONVERGÊNCIA física)    │
        │  geosteering_ai/{config,data,models,losses,training,inference,...}       │
        │  [NOVO] ingest/ · registry/ · evaluation/calibration.py                  │
        └────────────────────────────────────────────────────────────────────────┘
```

**Os 4 produtos e seu estado:**

| Produto | Estado | Fundação que consome | Gap #1 |
|:--------|:------:|:---------------------|:-------|
| **Biblioteca / API** | ~85% (BETA) | toda a biblioteca | semver real (2.0.0 congelado) + PyPI + `py.typed` |
| **CLI** | ~60% (MVP) | `cli/` + `simulation/` | `--backend auto` + comandos `train`/`infer` |
| **Simulation Manager** | ~75% (Produção, débito) | `gui/` + `simulation/` **apenas** | **JAX GPU** + extração para `gui/` |
| **Geosteering AI Studio** ⭐ | **0% código** | `gui/` + biblioteca inteira | a casca MVVM + perspectivas (nasce sobre `gui/`) |

**O caminho crítico** (verificado e sequenciado por dependência): a fundação `gui/` (extraída da infra modular `sm_*.py`) desbloqueia **simultaneamente** o SM modular e o Studio. O `gui/` é o gargalo da Fase 0.

---

## 2. Decisões-Âncora (verificadas no código)

| Decisão | Resumo |
|:--------|:-------|
| **Monorepo com fronteiras explícitas** | Um repo; `core` nunca importa Qt; `gui/` nunca importado por `models/losses/training/...`. Extração futura do Studio para repo separado é decisão de empacotamento, não de arquitetura. |
| **Fundação compartilhada `gui/` + `simulation/`** | SM e Studio são produtos **separados** que compartilham `gui/` (infra Qt) e `simulation/` (backend). O monólito `simulation_manager.py` **não** é o esqueleto — a infra modular `sm_*.py` é. |
| **Física convergente única** | `simulation/multi_forward.py` é o ponto único; toda spec que toca cálculo EM declara `Converge-Em: multi_forward.py`. |
| **MVVM para toda GUI** | View (Qt) ⊥ ViewModel (sem import Qt, testável) ⊥ Model (biblioteca). Habilita pytest puro sem pytest-qt. |
| **TF/Keras exclusivo em produção** | PyTorch só via adapter isolado. Runtime congelado: ONNX RT (batch) + TFLite INT8 (realtime). |
| **SDD entre o ROADMAP §0 e os snapshots de sprint** | SDD não cria SSoT concorrente (ADR-0001 R1/R2). |

---

## 3. Correções Aplicadas pela Revisão Adversarial (importante)

A revisão crítica encontrou imprecisões que **alteram decisões**. Estão incorporadas nos documentos e devem virar ADRs:

| ID | Correção | Onde está tratada |
|:--:|:---------|:------------------|
| **C1 (P0)** | **Contradição da paridade Fortran no CI:** o alvo `portable` do `Fortran_Gerador/Makefile:62` mantém `-ffast-math`, mas a validação numérica exige `-fno-fast-math` (linha 56). Tornar o gate `<1e-12` bloqueante usando o binário `portable` validaria contra FP **reordenado**. **Ação:** criar alvo `validation_portable` (`-O0 -march=x86-64-v2 -fno-fast-math -fsignaling-nans`) e usá-lo no gate; **ADR obrigatório**. | [05_cicd.md](05_cicd.md), [CONSTITUTION §I](../../specs/CONSTITUTION.md) |
| **C2 (P0)** | **Três regimes de tolerância** (não um): física Fortran `complex128` **<1e-12**; JAX-vs-Numba `complex64` **<1e-10**; inferência treinada `float32` **<1e-6**. A Constituição os enumera para evitar que specs futuras rejeitem golden-tests legítimos. | [CONSTITUTION §I](../../specs/CONSTITUTION.md), [06_mlops.md](06_mlops.md) |
| **C3 (P1)** | **`ScalerRegistry` é vaporware** (só uma string em `data/scaling.py:153`). É **dependência dura** de reprodutibilidade (scaler fit-on-clean faz parte do contrato dataset→modelo→inferência); sem ele o golden-test G1 é irreprodutível. Elevado a item de fundação (spec 0015, par do ModelStore). | [06_mlops.md](06_mlops.md), [ROADMAP](../../specs/ROADMAP.md) |
| **C4 (P1)** | **Semver do pacote congelado em 2.0.0** após 56+ sprints. Bumpar para `2.56.0` **antes** de adotar `importlib.metadata` como SSoT, senão a SSoT oficializa uma versão obsoleta. | [03_api_design.md](03_api_design.md) |
| **C5 (P1)** | **Segurança de `.gsproj`/`.session`:** além de proibir pickle, exigir `SecureArchiveReader` (sem path-traversal/zip-bomb/symlink; limites de JSON; sem refs HDF5 externas) — o arquivo cruza a fronteira campo↔escritório. | [07_cross_cutting.md](07_cross_cutting.md) |
| **C6 (P2)** | **`--backend auto`:** adicionar `auto` a `choices` agora, mas manter `default="numba"` por 1 minor com `DeprecationWarning` (PEP 387) — evita regressão silenciosa de scripts/CI. | [03_api_design.md](03_api_design.md), [spec 0003](../../specs/0003-cli-backend-auto/spec.md) |
| **C7 (gap)** | **Capacidade solo-dev vs 4 produtos:** sequenciar por **dependência**, não por desejo comercial. O Studio é a estrela-guia, mas seu caminho crítico passa pela fundação + SM modular. Ver [ROADMAP §Capacidade](../../specs/ROADMAP.md). | [ROADMAP](../../specs/ROADMAP.md) |
| **C8 (gap)** | **Validação geológica:** Volve (arenito Mar do Norte) é **necessário mas insuficiente** para a geologia-alvo (pré-sal/carbonatos). "PRODUCTION" exige dataset de campo **representativo** — item de backlog de **prioridade máxima, não-software**. | [06_mlops.md](06_mlops.md), [ROADMAP](../../specs/ROADMAP.md) |
| **C9 (gap)** | **SLO realtime sem piso de hardware:** reescrever como "<100 ms no perfil **HW-FIELD-MIN** (a definir)" + benchmark nesse perfil como gate de promoção. | [07_cross_cutting.md](07_cross_cutting.md) |
| **C10 (gap)** | **Trajetória (MD↔TVD) é entidade de domínio de 1ª classe** (editável, versionada no `.gsproj`), não um detalhe de `ingest/` — geosteering É navegação na trajetória. | [02_data_flow.md](02_data_flow.md) |

---

## 4. Decisões Abertas que Precisam de ADR (antes da Onda O2)

Consolidadas dos componentes e do roadmap. Cada uma vira `docs/decisions/ADR-XXXX.md`:

| # | Decisão | Recomendação default |
|:-:|:--------|:---------------------|
| D-A | Binding Qt canônico do Studio (PyQt6 GPL/comercial vs **PySide6 LGPL**) | PySide6 (Studio comercial); `qt_compat` suporta ambos |
| D-B | Licenciamento/repo do Studio (monorepo misto vs extração para repo privado) | Monorepo agora; extrair no/antes do BETA |
| D-C | Prioridade `gui/` vs `ingest/` | `gui/` primeiro (Fases 0–3 rodam com sintético); `ingest/`+G5 a partir da O6 |
| D-D | Backend MLflow (SQLite+FS local vs PostgreSQL+S3/MinIO) | SQLite+FS na A6000 p/ Fase 1; cota+GC **antes** de ligar autolog |
| D-E | Alvo WITSML (1.4.1.1 arquivo vs 2.0/ETP streaming) | LAS+arquivo no MVP; ETP 2.0 como onda seguinte |
| D-F | Métrica de aceite do gate G5 (R² vs DTB-error-m vs concordância publicada) | ADR de domínio; confirmar acesso ao Goliat (NDA) |
| D-G | Determinismo do golden-test G1 (`TF_DETERMINISTIC_OPS=1` vs rtol 1e-6) | rtol 1e-6 com nota; determinístico em nightly |
| D-H | Rampa do gate pytest-qt (cov-diff só em código novo vs rampa global 25→80%) | cov-diff ≥80% em código novo + rampa global por onda |

---

## 5. Como usar este blueprint

1. **Leia** este README + a [Constituição](../../specs/CONSTITUTION.md) (princípios invioláveis).
2. **Consulte** o componente relevante (01–07) ao escrever uma spec.
3. **Siga** o [ROADMAP](../../specs/ROADMAP.md): cada etapa é uma `spec → plan → tasks → implement → verify`.
4. **Ratifique** as decisões abertas (§4) como ADRs antes da Onda O2.

*Gerado por workflow multi-agente (8 designers + roadmap + crítica adversarial), cruzado com leitura direta do código e dos relatórios prévios. Não modifica `docs/ROADMAP.md`/`CHANGELOG.md` — a formalização aguarda ratificação (ADR-0001 R1).*
