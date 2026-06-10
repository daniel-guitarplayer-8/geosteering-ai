# Geosteering AI Studio — Levantamento de Status, Pesquisa de Produto e Análise de Requisitos

**Data:** 2026-06-04
**Autor:** Daniel Leal (assistido por Claude Opus 4.8 + workflow multi-agente)
**Tipo:** Análise de produto · Levantamento de status · Análise de requisitos · Estruturação de software
**Escopo:** Estruturar o **Geosteering AI Studio** — o produto que transforma o backend `geosteering_ai/` (simulador + DL + inferência) em uma solução de produção para **geonavegação em tempo real via Deep Learning**, treino de novas redes, inferência offline e análise geofísica/petrofísica.
**Status:** Investigação — **não implementa código**. Aguarda decisões do usuário (Parte VIII).
**Versão do projeto:** v2.56 (branch `feat/v2.45-followups`)
**Método:** Workflow multi-agente (15 agentes — 7 levantamentos internos + 4 pesquisas externas + 3 sínteses + 1 crítica adversarial; ~1,47M tokens; 381 tool-uses) cruzado com leitura direta de código e dos relatórios prévios (`mvc_simulation_manager_studio_analysis_2026-05-18.md`, `premortem_geosteering_ai_2026-05-09.md`, `v2.39_proximos_passos_roadmap_2026-05-17.md`).

---

## Sumário Executivo

| Pergunta | Resposta |
|:---------|:---------|
| O backend (física, DL, dados, inferência) está pronto? | **Sim, em grande parte.** Simulador (paridade Fortran <1e-12), 48 arquiteturas DL, 26 losses + 8 PINNs, pipeline de dados on-the-fly e inferência causal estão **implementados e validados em sintético**. O gap não é física nem DL. |
| Qual é o gap real para o produto? | **GUI realtime de produção + ingestão de dado real (WITSML/ETP/LAS/DLIS) + governança de modelos + validação de campo.** O Pilar D (geonavegação em poço real) concentra **8 de 19** requisitos novos. |
| O Studio já existia no planejamento? | **Sim** — concebido como "Trilha B comercial" (repo privado, ALPHA 2026 Q4) em [mvc_simulation_manager_studio_analysis](mvc_simulation_manager_studio_analysis_2026-05-18.md) e [v2.39_proximos_passos §7](v2.39_proximos_passos_roadmap_2026-05-17.md). **Esta análise expande o escopo** para os 4 pilares unificados que o usuário declarou. |
| Existe GUI reaproveitável? | **Sim** — o *Simulation Manager* (~18,9k LOC Qt: 10.759 no monólito + ~8,2k nos `sm_*.py`), mas vive em `simulation/tests/` (débito), é monolítico (sem MVC) e tem só ~25% de cobertura pytest-qt. |
| O "DL-first" é diferencial de mercado? | **Sim, qualificado.** Incumbentes (SLB Neuro, HAL LOGIX) tratam DL como componente acoplado a hardware LWD. Posição recomendada: **software-first / DL-native** no espaço ROGII/Petrolink/Corva, **não** no espaço OFS. |
| Maior risco do projeto | **Validação de campo real** (não é técnico). A paridade Fortran prova o *forward*, não a generalização da inversão DL em dado de poço real. Sem caso estilo Goliat + UQ calibrada, o claim comercial é indefensável. |
| Prontidão do material para decisão | **~70% para decisão técnica · ~45% para decisão de negócio.** Faltam decisões de licenciamento/monetização, segurança OT regulatória e um plano de validação científica como *gate* de release. |
| Recomendação central | **Construir o MVP offline (v0.1) primeiro** (entrega valor end-to-end sem depender de rig), **precedido da Fase 0** (extrair a GUI de `tests/` → pacote de 1ª classe + refactor MVVM). Tratar realtime/WITSML como Fase 2 e validação de campo como *gate* do rótulo "produto". |

**Correção de honestidade (apontada pela crítica adversarial):** três alegações do material precisam ser ajustadas antes de virarem compromisso comercial:
1. **UQ "P10/P50/P90 nativa (INN)" NÃO existe pronta** no backend. O que existe é MC Dropout (CI 95% gaussiano) + Ensemble + INN como *perturbação de entrada* (sensibilidade, não posterior). Percentis calibrados, métricas de calibração (coverage/CRPS) e MDN são **gaps abertos** — e são justamente o diferenciador de mercado central. Tratá-los como **alvo a construir** (RF-B11, RF-D10), não como ativo pronto.
2. **Latência "realtime" tem três alvos divergentes** (<100 ms produto, <31 ms sliding-window, <10 ms SLO) e **nenhum microbenchmark fim-a-fim** foi medido. Dado que a telemetria *mud-pulse* impõe atraso de segundos-a-minutos, o diferenciador honesto é **qualidade + UQ da inversão**, não latência sub-100 ms.
3. **Domain adaptation está marcado "pronto" mas nunca exercido em campo real** — é o gap mais crítico de credibilidade segundo a literatura.

---

## Parte I — Levantamento do Estado Atual (o que já existe)

### I.1 Métricas globais do código

| Subpacote | LOC | Papel |
|:----------|----:|:------|
| `simulation/` | 48.744 | Motor EM 1D TIV — backends Numba (CPU) + JAX (GPU), paridade Fortran <1e-12 |
| `training/` | 7.718 | TrainingLoop, N-Stage, 16 callbacks, HPO Optuna, DomainAdapter |
| `models/` | 7.611 | 48 arquiteturas (9 famílias) + ModelRegistry + 23 blocos |
| `data/` | 7.153 | Loading 22-col, P1 split, 6 FV, 5 Geosinais, 8 scalers, DataPipeline, DTB, SurrogateDataset |
| `visualization/` | 6.488 | EDA, Picasso/DOD, curtain, DTB, UQ, realtime monitor, Optuna |
| `evaluation/` | 5.901 | Métricas, comparação, Picasso analítico, relatórios, realtime vs offline, manifesto |
| `losses/` | 3.693 | 26 losses + 8 cenários PINN + TIV constraint |
| `cli/` | 3.617 | `geosteering-cli` (simulate/benchmark/warmup) |
| `noise/` | 2.640 | 34 tipos on-the-fly + curriculum 3-phase |
| `utils/` | 2.186 | Logger, timer, validação |
| `inference/` | 1.983 | InferencePipeline, RealtimeInference, UQ, export (SavedModel/TFLite/ONNX) |
| `api/` | 1.541 | FastAPI REST (`/health`, `/predict`) |
| `multi_agent/` | 622 | LockManager, ConflictMatrix (infra de dev) |
| **GUI Simulation Manager** | **~18.960** | `simulation/tests/simulation_manager.py` (10.759) + 17 `sm_*.py` (~8.201) |

> **Nota da crítica:** a síntese de arquitetura citou "~20,2k LOC" de GUI; a medição real é **~18,96k** (~6% inflado). Isso importa porque a tese de "reaproveita 60–80%" depende dessa base — com cobertura de teste a só 25%, o esforço de refactor está **subestimado**.

### I.2 Maturidade por subsistema (do produto)

| Subsistema | Estado | Prontidão p/ produção | Observação-chave |
|:-----------|:------:|:---------------------:|:-----------------|
| **Simulador EM 1D (Numba+JAX)** | Implementado | Pronto | Paridade Fortran <1e-12 (163 testes); dispatcher auto-backend medido; ~92–95k mod/h (E, Numba), ~210k (JAX GPU) |
| **DL — modelos/losses/PINNs** | Implementado | Pronto (sintético) | 48 arqs (26 causal-compatible), LossFactory, 8 PINNs; **SurrogateNet não convergido** (falta GPU-hours, não é bloqueio técnico) |
| **Treino — Loop/N-Stage/HPO** | Implementado | Pronto | Causal finetuning automático; gaps menores (Optuna sem UI, λ-PINN sem clamp, finetuning 10 ep. por constante de módulo) |
| **Dados — pipeline on-the-fly** | Implementado | Pronto (sintético) | noise→FV→GS→scale (GS veem ruído); **só consome `.dat`/`.hdf5` sintético** |
| **Inferência offline** | Implementado | Pronto | 6 passos determinísticos, serializável, UQ MC Dropout/Ensemble |
| **Inferência realtime (causal)** | Implementado | Quase | `RealtimeInference` (deque maxlen=600) **pronto na biblioteca**, mas **0% integrado** a sistema real de poço; retorna `None` até o buffer encher (600 medições) e **re-roda a janela inteira a cada update** |
| **Avaliação + Visualização** | Implementado | Pronto | curtain, DTB, Picasso, error maps, dashboards, UQ — **altamente reutilizável na GUI** |
| **API REST** | Implementado | MVP/quase | `/predict` 22-col + MC Dropout; **sem auth, sem rate-limit, singleton por processo** |
| **CLI + Docker + CI/CD** | Implementado | Quase | CLI production-ready; Docker CPU sem CUDA; CI verde; **sem release automation, sem model registry** |
| **GUI (Simulation Manager)** | Implementado | Precisa-trabalho | Monolítico (10,7k LOC), em `tests/`, 25% cobertura, sem MVC/MVVM |
| **Ingestão de dado real** | **Ausente** | Pesquisa | **0 referências** a WITSML/ETP/WITS/LAS/DLIS/OSDU no pacote; `lasio`/`dlisio` não declarados |
| **Governança de modelos** | **Ausente** | Pesquisa | Sem MLflow/registry, sem lineage formal, versão `v2.0.0` hardcoded |

### I.3 O coração do produto: como a inferência realtime funciona hoje

`geosteering_ai/inference/realtime.py` (`RealtimeInference`) é o motor de geonavegação:

```
Medições LWD (1 vetor 22-col por ponto de profundidade)
   → deque(maxlen=sequence_length=600)  [FIFO circular]
   → quando buffer cheio: stack → (1, 600, 22) → InferencePipeline.predict()
   → retorna predição (1, 600, n_targets) em Ω·m
```

**Limitações de produção observadas no código:**
- Retorna `None` durante o *warmup* (primeiras 600 medições descartadas) — frio no início da seção.
- **Re-executa a janela completa a cada nova medição** (não é incremental) — custo redundante; o estado causal do modelo não é mantido entre updates.
- Consome uma **lista iterável**, não um stream pub/sub — falta o adaptador on-arrival (callback) para ETP/WITS.
- Não há `WellStateManager` (profundidade/seção/ângulo/histórico), nem persistência, nem multi-poço sincronizado.

### I.4 A semente da GUI: o Simulation Manager

A GUI existente **já é uma aplicação de abas** (`QTabWidget`: Simulador / Benchmark / Resultados / Preferências) com ativos diretamente reaproveitáveis:

| Ativo | Reuso p/ Studio | Origem |
|:------|:---------------:|:-------|
| `qt_compat` (PyQt6/PySide6 neutro) | 100% | `sm_qt_compat.py` |
| Backends de plot (ABC + mpl/pyqtgraph/plotly/vispy, 60 fps) | 100% | `sm_plot_backends/` |
| Threading anti-hang (ProcessPool efêmero v2.29, 150k+ mod/h) | Refactor → Service | `sm_workers.py:470` (ainda subclassa `QThread`) |
| Persistência `.exp.json` + cache LRU 500 MB | 100% (evolui p/ `.session`) | `sm_snapshot_persist.py`, `sm_plot_cache.py` |
| Geração de modelos de camadas (estocástica + canônica) | Adaptável multi-poço | `sm_model_gen.py`, `sm_layers_dialog.py` |
| Widgets/toast/animação | 100% | `sm_widgets.py`, `sm_toast.py`, `sm_animation_bar.py` |

**Débitos:** monólito de 10,7k LOC (~80k tokens, 16% de uma janela de 500k), em `tests/`, sem MVC/MVVM, Signal/Slot implícitos, `QSettings` global. Já analisado em [mvc_simulation_manager_studio_analysis_2026-05-18.md](mvc_simulation_manager_studio_analysis_2026-05-18.md) (estratégia *Strangler Fig*, R1–R9).

---

## Parte II — Pesquisa de Mercado e Literatura

### II.1 Panorama do software comercial de geosteering (2023–2026)

A tendência do biênio é a transição de "geosteering assistido" para **geosteering autônomo closed-loop**: SLB Neuro (dez/2024), Halliburton LOGIX (jul/2025) e a colocação totalmente automatizada offshore Guyana (2026). Academicamente, um "robô de geosteering" (particle filter + RL) **superou a média de especialistas humanos** em benchmark (SPE Journal 2025).

**Dois segmentos:**

| Segmento | Players | Moat | Relevância p/ o Studio |
|:---------|:--------|:-----|:-----------------------|
| **OFS integrado** (hardware LWD + SW + serviço) | SLB (GeoSphere/PeriScope/Neuro), Halliburton (EarthStar/LOGIX), Baker Hughes (VisiTrak/AutoTrak) | O par **ferramenta↔software** gera o dado ultraprofundo | **NÃO competir aqui** — exige hardware downhole |
| **Software independente** (vendor-agnostic) | **ROGII StarSteer**, Petrolink, Corva, Enverus/GVERSE, SMART4D | Plataforma de interpretação sobre WITSML | **Espaço-alvo do Studio** |

**Checklist de mercado** (o que um software de produção precisa ter):
- **Interpretação:** curtain/Earth Model dinâmico · correlação de logs (Ghosting/flatten/TVT) · inversão determinística **+ estocástica P10/P50/P90** · DTB multi-camada · ahead-of-bit/look-ahead (4D) · log sintético dinâmico.
- **Tempo real:** ingestão **WITSML/WITS/LAS/ETP** multi-fonte (streams ~1 s) · atualização proativa do modelo · colaboração cloud.
- **Automação:** what-if/projeção de trajetória · auto-steer/dip-picking por ML · closed-loop com RSS.
- **Entrega:** desktop + web + mobile + cloud; API/extensibilidade (Python).

**Posicionamento recomendado:** "**software-first / DL-native**" no espaço ROGII/Petrolink/Corva. Diferenciais *defensáveis* (mas que exigem trabalho): (1) inversão DL sub-segundo com **UQ calibrada P10/P50/P90**, (2) **PINNs** garantindo consistência Maxwell/TIV (incumbentes não anunciam), (3) extensibilidade Python-first. **Ingestão WITSML é requisito de sobrevivência, não diferencial.**

### II.2 Padrões de dados e integração realtime

| Padrão | Papel | Implicação p/ o Studio |
|:-------|:------|:-----------------------|
| **WITSML 1.4.1.1** (SOAP/polling) | Mais difundido em rigs reais | Suportar p/ compatibilidade legada |
| **WITSML 2.0 / ETP** (WebSocket, Avro, JWT+TLS) | Alvo estratégico (OSDU DAG só aceita 2.0) | Alvo de roadmap; <1 s rig→remoto |
| **WITS L0** (TCP/serial) | Legado simples | Bom **primeiro conector MVP** |
| **LAS 2.0/3.0** (`lasio`) / **DLIS RP66** (`dlisio`) | Batch/replay/validação offline | Leitores de arquivo → 22-col |
| **PWLS 3.0** | Normalização cross-vendor de mnemônicos | Mapear RES_DEEP/ATR/A2RD/RV → 22-col |
| **OSDU** | Destino corporativo (WellLog/Trajectory) | Exportação pós-ALPHA |

**Pontos críticos:**
- O LWD de propagação mede **phase shift + attenuation** em 400 kHz/2 MHz → mapeável a ρ_h/ρ_v (cols 2–3) e ao tensor Hxx/Hzz (cols 4–5, 20–21) do formato 22-col interno.
- **Gargalo é a taxa de chegada (mud-pulse: s–min), não a inferência** → arquitetura **edge-first** (inferência no rig, envia só predição+UQ).
- Segurança OT: rede do rig regida por **IEC 62443/Purdue**, **sem VPN always-on**, jump-server + MFA + Zero Trust.
- **Trajetória (MD↔TVD via minimum curvature) não cabe nas 22-col** — precisa de estrutura própria.

> **Nota de viabilidade:** `lasio` (LAS), `welly` (well logs) e `pyvista` (3D) **já estão instalados no ambiente conda `Geosteering_AI`** (env-only, não declarados em `pyproject.toml` — ver memória `geoscience-libs-installed`). Ou seja, a prototipagem de ingestão LAS (RF-C08) e de visualização 3D futura não precisa esperar instalação — falta apenas **declarar as deps** (grupo opt-in, ex.: `[ingest]`) e escrever os leitores. `dlisio` (DLIS) ainda não está instalado.

### II.3 Estado da arte — Deep Learning para inversão LWD/geosteering

| Eixo | Estado da arte | Alinhamento do projeto |
|:-----|:---------------|:----------------------:|
| Arquitetura | ResNet 1D compacta domina forward/inverse de baixa latência; U-Net/BiLSTM/Transformer por modalidade | **Alinhado/à frente** (amplitude de 48 arqs cobre todas as famílias) |
| UQ | INN/normalizing flows (posterior), MDN (multimodal), Bayesian DL, ensemble smoother (FlexIES) | **Parcial** — tem INN (como perturbação), MC Dropout, Ensemble; **faltam percentis calibrados, MDN, métricas de calibração** |
| Física | PINN (resíduo Maxwell, cross-gradients, look-ahead) melhora robustez sob ruído | **Alinhado** (`losses/pinns.py`, cross_gradient, look_ahead) |
| Ruído | Ruído no treino + augmentation + camada de ruído | **Alinhado e fisicamente correto** (on-the-fly, GS veem ruído) |
| **Validação de campo** | **Caso real (Goliat) vs inversão proprietária é mandatório** para credibilidade | **GAP CRÍTICO** — validado só em sintético |
| Generalização sintético→campo | Domain adaptation/transfer learning é "o gap mais subestimado" | **GAP** — DomainAdapter existe mas nunca exercido em campo |
| Compressão | AutoML/NAS comprime 4–14× p/ deploy downhole | Oportunidade (48 arqs = bom espaço de busca) |

**Conclusão da literatura:** o projeto está **no estado da arte ou à frente em amplitude arquitetural, riqueza de losses/PINN e correção física do ruído**. Para credibilidade de **produção** faltam 3 validações *mandatórias*: (a) caso de campo real vs inversão proprietária, (b) domain adaptation sintético→real, (c) **métricas de calibração de incerteza** (coverage, reliability diagram, CRPS).

### II.4 MLOps e UX de aplicação científica desktop

| Tema | Recomendação | Estado atual |
|:-----|:-------------|:------------:|
| Model registry/lineage | **MLflow 3.x on-prem** (no A6000): git hash + dataset + seed | Ausente (só código é versionado) |
| Runtime de produção | **ONNX Runtime** (batch) + **TFLite/LiteRT INT8** (realtime); INT8 −60% latência, −75% tamanho | Export existe; **sem golden-test pré/pós-INT8** |
| Empacotamento | **conda/constructor** (rota napari) > PyInstaller p/ deps JAX/Qt/CUDA | Ambiente conda já existe; sem instaladores |
| GUI | **MVVM** (ViewModel sem import Qt → testável) | Monólito Qt, 25% cobertura |
| Threading | worker-object + `moveToThread` > subclassar `QThread`; `deepcopy` em sinais mutáveis | `sm_workers.py:470` subclassa QThread |
| Governança/regulação | Geosteering = domínio **crítico/regulado** (model-risk US$500M+); exige model card + drift (KS/PSI) + auditoria | Ausente |
| Plotting | pyqtgraph (≤100k pts) + vispy/OpenGL (10M+) | **Já adotado** nos backends |

---

## Parte III — Visão de Produto e Posicionamento

### III.1 Definição

> **Geosteering AI Studio** é uma aplicação científica desktop **DL-first** (Python/Qt6) que unifica, num único shell, todo o ciclo de geonavegação: **simulação EM 1D TIV → geração de datasets → treino/avaliação de redes de inversão → inferência offline/análise → geonavegação causal em tempo real** (DTB/look-ahead com UQ). O Studio **não recomeça**: promove a GUI Qt já madura e consome `geosteering_ai/` como biblioteca de domínio.

### III.2 Reconciliação com o planejamento prévio

O Studio **já estava concebido**, mas com escopo mais estreito. Esta análise expande e reconcilia:

| Dimensão | Visão prévia (mai/2026) | Visão atual (esta análise) |
|:---------|:------------------------|:---------------------------|
| Natureza | App comercial **separada** ("Trilha B", repo privado), distinta do Simulation Manager (dev-tool) | **Produto unificado** cobrindo os 4 pilares; SM é a **semente** a ser promovida |
| Foco | GUI realtime + multi-poço + WITSML | **+ treino de novas redes + inferência offline + análise geofísica** como pilares de 1ª classe |
| Cronograma | ALPHA Q4 2026 · BETA Q1 2027 · GA Q2 2027 | Mantém, **mas condicionado a Fase 0 + decisões de negócio** (ver Parte VIII) |
| Arquitetura | 6 camadas S1–S6, Hexagonal⊕MVVM | Compatível; detalhada em camadas L0–L5 (Parte V) |
| Repo/licença | `geosteering-studio` privado importando pip-lib MIT | **Decisão ainda aberta** (Parte VIII, Q3) |

### III.3 Os 4 pilares × cobertura atual

| Pilar | Descrição | Backend pronto? | Gap principal |
|:------|:----------|:---------------:|:--------------|
| **A — Simulação realtime via DL** | Forward EM + geração de datasets sintéticos | ✅ Alto | UI sobre `generate_batch`; multi-frequência |
| **B — Treino de novas redes DL** | Selecionar/treinar/avaliar arquiteturas causais + PINNs | ✅ Alto | UI de treino; Optuna dashboard; métricas de calibração de UQ |
| **C — Inferência offline + análise** | Inverter perfis, EDA, Picasso/DTB, relatórios | ✅ Alto | **Ingestão LAS/DLIS**; multi-frequência |
| **D — Geonavegação tempo real** | Streaming de poço → inversão causal → DTB/steering | ⚠️ Parcial | **Bridge WITSML/ETP, WellStateManager, steering engine** (8/19 itens novos) |

---

## Parte IV — Análise de Requisitos

> Legenda **Cobertura:** COBERTO (existe e validado) · PARCIAL (backend existe, falta UI/integração) · NOVO (não existe). **MoSCoW:** Must/Should/Could/Won't-now.

### IV.1 Personas e casos de uso

| Persona | Objetivo | Pilares | Bloqueadores atuais |
|:--------|:---------|:-------:|:--------------------|
| **Geofísico de geosteering** | Posicionar o poço na zona-alvo em tempo real com DTB + incerteza confiável | D, A | Sem bridge WITSML/ETP; sem steering engine; latência CPU 100–150 ms |
| **Engenheiro de poço / well-site** | Supervisionar perfuração, receber alertas, exportar relatório | D, C | Sem WellStateManager; alarmes só em log; sem multi-poço sincronizado |
| **Pesquisador de DL** | Treinar/comparar arquiteturas causais + PINNs e validar contra física | B, A | Optuna sem UI; sem métricas de calibração; MDN ausente; SurrogateNet não convergido |
| **Data scientist** | EDA, governança de versões, reprodutibilidade | C, B | Sem model registry/lineage; ingestão só sintética; sem drift monitoring |

### IV.2 Requisitos funcionais — resumo de cobertura

| Pilar | RF total | COBERTO | PARCIAL | NOVO |
|:------|:--------:|:-------:|:-------:|:----:|
| A — Simulação realtime DL | 8 | 3 | 3 | 2 |
| B — Treino de redes DL | 14 | 6 | 3 | 5 |
| C — Inferência offline + análise | 12 | 7 | 1 | 4 |
| D — Geonavegação tempo real | 13 | 2 | 3 | 8 |
| **Total** | **47** | **18 (38%)** | **10 (21%)** | **19 (40%)** |

**Leitura crítica:** A/B/C estão majoritariamente cobertos (backend maduro). O **Pilar D concentra 8 de 19 itens NOVOS** — o valor comercial do Studio depende de software que **ainda não existe**: bridge WITSML/ETP, WellStateManager, steering engine e governança de modelos.

### IV.3 Requisitos funcionais detalhados (seleção dos Must e dos NOVOS críticos)

**Pilar A — Simulação / geração de dados**

| ID | Requisito | MoSCoW | Cobertura |
|:---|:----------|:------:|:---------:|
| RF-A01 | Forward EM 1D TIV, paridade Fortran <1e-10 | Must | COBERTO |
| RF-A02 | Dispatcher auto-backend (JAX⇄Numba) + guard OOM | Must | COBERTO |
| RF-A03 | Gerar dataset on-demand (≤30k modelos) via UI | Must | PARCIAL |
| RF-A06 | Suporte multi-frequência (10 MHz/400 kHz/20 kHz) | Should | PARCIAL (`config.frequency_hz` escalar) |

**Pilar B — Treino**

| ID | Requisito | MoSCoW | Cobertura |
|:---|:----------|:------:|:---------:|
| RF-B01 | Selecionar arquitetura (48) com filtro causal | Must | COBERTO |
| RF-B03 | Ruído on-the-fly (34) + curriculum 3-phase | Must | COBERTO |
| RF-B04 | Losses geofísicas + 8 PINNs (Maxwell/TIV) | Must | COBERTO |
| RF-B06 | Domain adaptation sintético→campo via wizard | Must | PARCIAL (nunca testado em campo) |
| RF-B11 | **Métricas de calibração de UQ (coverage/CRPS)** | Should | **NOVO** |
| RF-B13 | MDN dedicada (multimodal ahead-of-bit) | Could | NOVO |

**Pilar C — Inferência offline / análise**

| ID | Requisito | MoSCoW | Cobertura |
|:---|:----------|:------:|:---------:|
| RF-C01 | Inferência offline + inverse scaling (log10→Ω·m) | Must | COBERTO |
| RF-C04 | Visualizações (curtain, DTB, Picasso DOD) | Must | COBERTO |
| RF-C08 | **Ingerir LAS 2.0/3.0 → 22-col** | Must | **NOVO** |
| RF-C09 | Ingerir DLIS/RP66 → 22-col | Should | NOVO |
| RF-C10 | Normalizar mnemônicos cross-vendor (PWLS 3.0) | Should | NOVO |
| RF-C12 | Conversão MD↔TVD (minimum curvature) | Should | NOVO |

**Pilar D — Geonavegação realtime**

| ID | Requisito | MoSCoW | Cobertura |
|:---|:----------|:------:|:---------:|
| RF-D01 | Inferência causal streaming (sliding window FIFO) | Must | COBERTO |
| RF-D02 | Validar causalidade (ΔR² <15%, fail-fast) | Must | COBERTO |
| RF-D04 | **Bridge WITSML 1.4.1.1 + ETP 2.0** | Must | **NOVO (bloqueador #1)** |
| RF-D05 | Adaptador stream-on-arrival (pub/sub) | Must | NOVO |
| RF-D06 | **WellStateManager** (prof./seção/ângulo/histórico) | Must | NOVO |
| RF-D07 | Dashboard realtime ao vivo (ρ + CI, DTB, latência) | Must | PARCIAL |
| RF-D09 | Steering rules engine (decisão automática) | Should | NOVO |
| RF-D10 | **UQ P10/P50/P90 nativa** (paridade StarSteer) | Should | PARCIAL→NOVO* |
| RF-D11 | Multi-poço (100+) com estado sincronizado | Should | NOVO |
| RF-D12 | MWD feedback loop (closed-loop) | Won't-now | NOVO |

> *RF-D10: o material original marcou PARCIAL; a verificação de código mostra que **percentis calibrados não existem** (só CI 95% gaussiano) → trata-se de **NOVO** para o claim de mercado.

### IV.4 Requisitos não-funcionais (seleção)

| ID | Categoria | Alvo | Estado |
|:---|:----------|:-----|:------:|
| RNF-01 | Latência realtime fim-a-fim | **definir 1 SLA** (ver Parte VIII Q2); CPU 100–150 ms hoje | PARCIAL |
| RNF-04 | Paridade física Fortran | <1e-12 (f64) — inviolável | COBERTO |
| RNF-06 | Escalabilidade multi-poço | 100+ concorrentes | NÃO ATENDE (singleton por processo) |
| RNF-07 | Segurança OT (IEC 62443/Purdue/Zero Trust) | edge-first, jump-server+MFA, ETP JWT+TLS | NOVO |
| RNF-08 | AuthN/AuthZ na API (JWT/OAuth2/RBAC) | + rate limiting | NOVO (MVP sem auth) |
| RNF-12 | Instaladores nativos Win/Linux/macOS | conda/constructor | NOVO |
| RNF-14 | Model registry + lineage | MLflow on-prem (A6000) | NOVO |
| RNF-15 | Model card + drift (KS/PSI) + auditoria | domínio regulado | NOVO |
| RNF-17 | Manifesto config+seed+git_hash | JSON | COBERTO |
| RNF-18 | Golden-test de inferência por versão | paridade pré/pós-INT8 | NOVO |
| RNF-19 | Refactor GUI MVVM (ViewModel testável) | ≥80% pytest-qt (hoje 25%) | PARCIAL |

### IV.5 Restrições invioláveis

| Restrição | Implicação para o Studio |
|:----------|:-------------------------|
| **TF/Keras exclusivo em produção** (PyTorch proibido) | Runtime = TF SavedModel/TFLite (+ ONNX só interop) |
| **Python 3.13** (3.12 fallback CI) | Sem 3.14+; ambiente conda `Geosteering_AI` |
| **Paridade Fortran <1e-12 sagrada** | Estende-se a golden-test de modelos (RNF-18) |
| **GPU A6000 local** (Colab descontinuado v2.44) | Dev GPU + MLflow on-prem local |
| **Errata física imutável** (`__post_init__`) | Ingestão (RF-C08) deve validar fail-fast reusando isso |
| **Logging (nunca print) + Config como parâmetro (nunca globals) + D1–D14** | Obrigatório em todo código novo (`ingest/`, steering) |

---

## Parte V — Arquitetura Proposta do Studio

### V.1 Camadas

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       GEOSTEERING AI STUDIO (desktop Qt6)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│  L5  SHELL (Qt)         MainWindow · perspectivas/abas · QUndoStack · temas     │
│        ▲ reaproveita simulation_manager.py MainWindow + QTabWidget              │
├──────────────────────────────────────────────────────────────────────────────┤
│  L4  VIEWMODELS (MVVM, sem import Qt → testáveis sem pytest-qt)                 │
├──────────────────────────────────────────────────────────────────────────────┤
│  L3  DOMÍNIO (DDD)      Workspace · Project · Well · Run · ModelVersion          │
├──────────────────────────────────────────────────────────────────────────────┤
│  L2  SERVIÇOS           SimulationSvc · TrainingSvc · InferenceSvc · IngestSvc · │
│                         RegistrySvc · BenchmarkSvc   (orquestram o backend)      │
├──────────────────────────────────────────────────────────────────────────────┤
│  L1  BACKEND  geosteering_ai/ (pip-lib v2.56) — simulation/models/losses/       │
│               training/data/inference/evaluation/visualization — REUSO 100%      │
├──────────────────────────────────────────────────────────────────────────────┤
│  L0  DADOS    projects/*.gsproj · MLflow registry · datasets *.hdf5/*.dat ·     │
│               wells *.session.h5 · ingest WITS/ETP/LAS/DLIS (a construir)        │
└──────────────────────────────────────────────────────────────────────────────┘
            ▲ Acoplamento de UI = ZERO no backend (biblioteca importável)
```

### V.2 Perspectivas (abas orientadas ao fluxo do geofísico)

| Perspectiva | Função | Backend consumido | Origem na GUI |
|:------------|:-------|:------------------|:--------------|
| **Simulação & Datasets** | Editar `SimulationConfig`, forward, gerar dataset, validar paridade | `simulate_batch`, `SyntheticDataGenerator` | Existe (`SimulatorPage`) |
| **Treinamento** | Selecionar arq./losses/PINN/curriculum; progresso; Optuna (Expert) | `ModelRegistry`, `LossFactory`, `TrainingLoop` | Novo (consome lib) |
| **Inferência / Análise** | Predizer perfis, inverse scaling, EDA, Picasso/DOD, relatórios | `InferencePipeline`, `evaluation/`, `visualization/` | Parcial (`ResultsPage`) |
| **Geonavegação Realtime** | Stream→buffer→causal→DTB/UQ→curtain 60 fps | `RealtimeInference`, `ingest/` (novo) | **Novo (núcleo)** |
| **Model Registry / Avaliação** | Versionar, comparar, model card, golden-tests, drift | `evaluation/comparison`, MLflow (novo) | Novo |

### V.3 Decisão-chave: promover a GUI a pacote de 1ª classe

A GUI vive em `simulation/tests/` (código de produção em `tests/` — semântica errada). **Precedente:** na v2.53, `sm_io.py` virou shim re-exportando de `simulation/io/tensor_dat.py`. **Aplicar o mesmo *Strangler Fig* à GUI inteira.** Estrutura proposta:

```
geosteering_studio/  (ou geosteering_ai/gui/ — decisão Parte VIII Q3)
├── app.py                  ← bootstrap QApplication + plugin loader
├── shell/                  ← L5 (main_window, perspective_host, qt_compat, theme, undo)
├── perspectives/           ← plugins (1 por aba; Perspective ABC via entry_points)
│   ├── simulation/ training/ inference/ realtime/ registry/   {view,viewmodel}.py
├── viewmodels/             ← L4 puros (ZERO import Qt → testáveis)
├── domain/                 ← L3 DDD (workspace, project, well, run, model_version)
├── services/               ← L2 (simulation/training/inference/ingest/registry + threading/worker)
├── plotting/backends/      ← reuso 100% (de sm_plot_backends/)
├── ingest/                 ← NOVO (wits_client, etp_client, las_reader, dlis_reader, pwls_mapper, trajectory)
└── persistence/            ← snapshot, plot_cache
```

### V.4 Fluxo de inferência realtime (núcleo do produto)

```
 RIG (OT, IEC 62443/Zero Trust)              STUDIO (edge-first, data van / A6000)
┌──────────────┐  WITS L0 / ETP wss://       ┌──────────────────────────────────────┐
│ LWD telemetry│── (JWT+TLS, <1s) ──────────▶│ IngestService                          │
│ mud-pulse    │  phase/atten 400k/2MHz       │  ├ WITS/ETP client (pub/sub Avro)       │
│ (s–min)      │                              │  ├ PWLS 3.0 mapper → 22-col            │
└──────────────┘                              │  └ trajectory MD↔TVD (min curvature)  │
   gargalo = TAXA DE CHEGADA, não throughput  └──────────────┬───────────────────────┘
                                                      callback on-arrival
                                          ┌────────────────────▼─────────────────────┐
                                          │ RealtimeInference.update() [deque 600]    │
                                          │  → modelo CAUSAL (TCN/WaveNet/Mamba)       │
                                          │  → UncertaintyEstimator (→ P10/P50/P90*)   │
                                          └────────────────────┬─────────────────────┘
                                          ┌────────────────────▼─────────────────────┐
                                          │ RealtimeViewModel → PlotCanvas (60fps):   │
                                          │ curtain ρh/ρv + bandas UQ, DTB,           │
                                          │ [steering rules → MWD]  (gap, Won't v0.1)  │
                                          └──────────────────────────────────────────┘
                                          predições+UQ ──ETP──▶ cloud/OSDU (arquivo)
```
*P10/P50/P90 + calibração são **a construir** (RF-B11, RF-D10).

### V.5 Deployment

| Eixo | Decisão | Justificativa |
|:-----|:--------|:--------------|
| Empacotamento desktop | **conda/constructor** (Win/Linux/macOS) | Deps JAX/Qt/CUDA pesadas; conda já em uso |
| Serving local | MLflow registry on-prem + **ONNX RT** (batch) + **TFLite INT8** (realtime) | INT8 −60% latência, −75% tamanho |
| API REST opcional | FastAPI existente + endurecer (X-API-Key, rate limit, CORS) | MVP sem auth hoje |
| Edge no well-site | Inferência local; envia só predição+UQ via ETP | IEC 62443/Zero Trust (sem VPN always-on) |
| Cloud/corporativo | Exportar WellLog/Trajectory p/ **OSDU** (ETP 2.0) | Destino corporativo natural |

### V.6 ADRs propostos (decisões a registrar)

| ADR | Decisão |
|:----|:--------|
| ADR-S01 | Promover a GUI a pacote de 1ª classe (Strangler Fig de `tests/`) |
| ADR-S02 | Separação Trilha A (pip-lib sem Qt) × Trilha B (Studio Qt) |
| ADR-S03 | MVVM (ViewModel sem Qt; meta pytest-qt ≥80%) |
| ADR-S04 | Threading: worker-object+moveToThread (I/O) × ProcessPool efêmero (Numba) |
| ADR-S05 | Perspectivas como plugins (`entry_points`) |
| ADR-S06 | Camada `ingest/` (WITS/ETP/LAS/DLIS + PWLS 3.0) — ETP 2.0 alvo estratégico |
| ADR-S07 | `RealtimeInference` adaptado a stream pub/sub on-arrival |
| ADR-S08 | Model Registry + lineage (MLflow on-prem A6000) |
| ADR-S09 | Runtime congelado: ONNX RT (batch) + TFLite INT8 (realtime) + golden-test |
| ADR-S10 | Empacotamento via conda/constructor |
| ADR-S11 | Governança de domínio crítico (model card + drift + auditoria) |
| ADR-S12 | Persistência `.exp.json`→`.gsproj`/`.session.h5` (DDD) |
| ADR-S13 | UQ de produção: reportar posterior (INN) + coverage/calibração |
| ADR-S14 | **Validação de campo sintético→real obrigatória antes de claim comercial** |

---

## Parte VI — Catálogo de Features, MVP e Roadmap

### VI.1 MVP — Studio v0.1 (valor end-to-end offline)

**Objetivo:** um geofísico **carrega um dataset → usa/treina um modelo → inferência offline com visualização e incerteza → exporta relatório**, numa GUI desktop empacotada — **sem depender de rig** (evita o bloqueador WITSML).

| **IN — v0.1** | **OUT — pós-v0.1** |
|:--|:--|
| GUI desktop empacotada (split Fase 0) | Streaming realtime de rig (WITSML/ETP/WITS) |
| Pipeline offline carregar→treinar→inferir→visualizar | Multi-poço concorrente + WellStateManager |
| UQ visual (MC Dropout, CI 95%) | Steering rules engine + MWD closed-loop |
| DTB + curtain + comparação de modelos | Ingestão de campo real (LAS/DLIS/WITSML) |
| Exportação de relatório MD/PDF | Model registry com lineage + A/B testing |
| Session save/open (1 poço) | Undo/Redo, drift monitoring, OSDU export |
| Geração de dataset sintético on-demand | Quantização INT8 on-rig, instaladores nativos |

**Critério de aceite v0.1:** end-to-end em dataset sintético; UQ renderizada; relatório gerado; paridade Fortran preservada. **Pré-requisito bloqueante: split físico da GUI (Fase 0).**

### VI.2 Roadmap em fases

| Fase | Objetivo | Features-chave | Dependências | Marco de validação |
|:--:|:--|:--|:--|:--|
| **0 — Fundação** | GUI importável e testável | Split `tests/→gui/`; refactor MVVM; pytest-qt ≥80%; worker-object+moveToThread | Nenhuma | GUI standalone; ≥80% cobertura; simulador sem regressão; paridade 10/10 |
| **1 — MVP Offline (v0.1)** | Ciclo carregar→treinar→inferir→visualizar | Editor `SimulationConfig`; gerador UI; seletor causal; treino c/ presets; inferência; curtain+UQ+DTB; comparação; relatório; session | Fase 0 | Aceite §VI.1; demo reprodutível |
| **2 — Realtime Streaming** | Geonavegação ao vivo + dado real | **Bridge WITSML 2.0/ETP**; WITS L0; PWLS→22-col; MD↔TVD; stream-on-arrival; monitor 60 fps; alarmes; **latência E2E medida**; replay | Fase 1 + `ingest/` | Inferência ≥1×/s com WITS/ETP; SLA medido; ΔR² <15% |
| **3 — Produto Robusto** | Governança, multi-poço, distribuição | Model Registry+lineage (MLflow); model card; golden-tests; multi-poço+WellStateManager; Undo/Redo; API hardening; instaladores conda; PostgreSQL; observabilidade | Fase 2 | Registry versiona artefato; ≥3 poços; instalador nativo; pip-audit verde |
| **4 — Diferenciadores** | Credibilidade de mercado + closed-loop | **UQ P10/P50/P90 + calibração (coverage/CRPS)**; what-if/ahead-of-bit; drift (KS/PSI); INT8 on-rig; A/B/canary; steering+MWD; OSDU; **validação de campo real (Goliat)** | Fase 3 | **Caso de campo real vs inversão proprietária**; calibração reportada; INT8 on-rig validado |

**Sequenciamento de risco:** Fase 0 é **bloqueante crítico** (sem ela o Studio reescreveria ~60% da GUI). O bridge WITSML (Fase 2) é o **gap #1 de sobrevivência**, mas é adiado para depois do MVP offline, que entrega valor sem rig.

### VI.3 Nova Trilha "G — Studio/Produto" no ROADMAP

O Studio **não cabe** nas 6 Trilhas existentes (A–F são todas da pip-lib). Proposta: **Trilha G — Studio/Produto** (camada de orquestração + GUI comercial que consome A–F).

| Atributo | Valor |
|:--|:--|
| **Tema** | GUI realtime comercial + orquestração end-to-end |
| **Diretório** | `geosteering_ai/gui/` (após Fase 0) + repo da app comercial |
| **Escopo** | Session/project mgmt, multi-poço, undo/redo, streaming UI, registry, instaladores, what-if, alarmes |
| **Fora de escopo** | Física, kernels Numba/JAX, losses (ficam em A–C como dependência) |
| **Fronteira D↔G** | parser/normalizador→22-col em **D** (`data/loaders/`, `ingest/`); conector streaming + bridge realtime em **G** |

---

## Parte VII — Riscos Críticos e Correções de Honestidade

> Da crítica adversarial. Estes pontos precisam ser **decididos/corrigidos** antes de o material virar compromisso comercial ou de cronograma.

| # | Risco / correção | Severidade | Ação recomendada |
|:--:|:--|:--:|:--|
| C1 | **Validação de campo real ausente** (gap #1). Paridade Fortran prova o forward, não a inversão DL em dado real. Sem dataset de campo no repo nem parceria. | **Alta** | Definir caso de referência (tentar **Volve público/Equinor** antes de assumir parceria); critério quantitativo (coverage/CRPS vs inversão comercial); **gate de release**: sem campo validado → rótulo "**TECH PREVIEW**", não "produto" |
| C2 | **UQ "P10/P50/P90 nativa" vendida como pronta — é gap aberto.** Só há MC Dropout (CI 95% gaussiano), Ensemble e INN-como-perturbação. Sem percentis calibrados, coverage/CRPS, MDN. | **Alta** | Reposicionar como **alvo (Fase 4)**; implementar percentis + métricas de calibração; é o diferenciador de mercado central |
| C3 | **Segurança OT regulatória não traduzida em requisitos.** Rig é OT (IEC 62443/Purdue); API atual sem auth alguma. | **Alta** | RNF de segurança OT explícitos (edge-first, jump-server+MFA, JWT+TLS, modelo de ameaça) antes de piloto em rig |
| C4 | **Governança/regulação (model-risk) fora de escopo.** Decisão de steering = domínio regulado (penalidades US$500M+). | **Alta** | Model card + drift (KS/PSI) + trilha de auditoria + disclaimer human-in-the-loop |
| C5 | **Viabilidade de equipe/cronograma/orçamento.** ALPHA Q4 acumula MVVM (5,5 sprints) ∥ `ingest/` (XL) ∥ WellStateManager ∥ steering ∥ campo. Precedente: 4 semanas de JAX "eclipsaram a Trilha C". GPU-hours não orçados. | **Alta** | Dimensionar pessoa-sprint do caminho crítico; promover **GPU runner CI (E-github-gpu)** de P4 a decisão de bloqueio |
| C6 | **Monetização/licenciamento não resolvido.** Repetido 3× sem convergência. pip-lib MIT pode permitir concorrente recriar o Studio. | **Alta** | Decidir modelo (per-well/per-seat/SaaS); estratégia de licença que proteja a Trilha B; license-check nos instaladores |
| C7 | **SLA de latência: 3 alvos divergentes, 0 medições.** Mud-pulse impõe s–min de atraso físico. | **Alta** | **Um** SLA operacional medido em hardware-alvo; reposicionar o claim para **qualidade+UQ**, não latência sub-100 ms |
| C8 | **`ingest/` subdimensionado** (marcado junto a multi-freq "L"; é um **épico XL** próprio). MD↔TVD não tratado. | **Alta** | Épico de ingestão com sequência de conectores; MD↔TVD minimum-curvature como estrutura separada |
| C9 | **Multi-frequência é pré-requisito de ingestão, não enhancement.** Simulador aceita até 2 MHz; pipeline DL é escalar (20 kHz). | **Média** | Mapear phase/attenuation→ρ_h/ρ_v→22-col; tratar antes/junto da ingestão |
| C10 | **Reaproveitamento da GUI superestimado** (80%/60%) vs 25% de cobertura e refactor não iniciado. | **Média** | Golden-tests de inferência + regressão visual antes do refactor; recalibrar estimativa |
| C11 | **Steering engine/MWD closed-loop sem decisão de escopo** (implicação de responsabilidade enorme). | **Média** | ADR de escopo ALPHA: **human-in-the-loop** como default (recomendação de mercado) |
| C12 | **Risco do proxy DNN fora-de-distribuição** não tratado. | **Média** | Model-error assessment; propagar incerteza quando entrada de campo sai da distribuição sintética |

---

## Parte VIII — Decisões Abertas (requerem o usuário)

Estas decisões aparecem repetidas nas sínteses **sem convergência** e bloqueiam o detalhamento. São o foco das próximas instruções.

| # | Decisão | Opções | Impacto |
|:--:|:--|:--|:--|
| **Q1** | **Escopo do ALPHA** | (a) MVP **offline** primeiro (recomendado) · (b) realtime desde já | Define a Fase 1 e o caminho crítico |
| **Q2** | **SLA de latência realtime** | <100 ms / <31 ms / "qualidade+UQ acima de latência" | Define RNF-01 e o claim de marketing |
| **Q3** | **Repo/licença do Studio** | (a) `geosteering_studio/` top-level privado/comercial · (b) `geosteering_ai/gui/` subpacote opt-in `[studio]` | Define ADR-S02, monetização, acoplamento |
| **Q4** | **Primeiro conector de ingestão** | WITS L0 (mais simples) / WITSML 1.4.1.1 (mais difundido) / ETP 2.0 (alvo OSDU) | Define a Fase 2 e compatibilidade com rigs |
| **Q5** | **human-in-the-loop vs closed-loop** | (a) só predições+UQ supervisionadas (recomendado) · (b) steering automático | Responsabilidade legal + escopo |
| **Q6** | **Validação de campo** | Volve público / SDAR/Teapot / parceria com operadora / só sintético ("TECH PREVIEW") | **Gate de credibilidade** do rótulo "produto" |
| **Q7** | **Multi-poço na v1.0?** | requisito v1.0 vs Fase 4 | Depende do cliente-alvo (operadora vs serviço single-well); muda a arquitetura de dados |
| **Q8** | **Orçamento GPU/equipe** | GPU runner CI (E-github-gpu); GPU-hours p/ SurrogateNet; dimensão de equipe | Viabiliza cronograma e governança de modelos |

---

## Parte IX — Recomendações e Próximos Passos

### IX.1 Recomendação central

1. **Fase 0 primeiro** — extrair a GUI de `simulation/tests/` para pacote de 1ª classe (`geosteering_ai/gui/`) + refactor MVVM + pytest-qt ≥80%. É bloqueante e de risco controlado (Strangler Fig já desenhado em [mvc_simulation_manager_studio_analysis](mvc_simulation_manager_studio_analysis_2026-05-18.md)).
2. **MVP offline (v0.1)** — entrega valor end-to-end sem depender de rig; usa ~90% de código existente.
3. **Tratar realtime/WITSML como Fase 2** e **validação de campo como gate da Fase 4** (rótulo "TECH PREVIEW" até lá).
4. **Corrigir o discurso de UQ e latência** (Parte VII C2, C7) antes de qualquer material comercial.

### IX.2 Ações imediatas sugeridas (sem implementar ainda — aguardando instruções)

| Ação | Tipo | Observação |
|:-----|:-----|:-----------|
| Responder Q1–Q8 (Parte VIII) | Decisão | Desbloqueia o detalhamento de sprints |
| Formalizar **Trilha G** + Studio no `docs/ROADMAP.md §0` | Governança | Por ADR-0001 R1, o ROADMAP é o SSoT do futuro — **só após decisão do usuário** |
| Criar **ADR-0002** (escopo ALPHA: human-in-the-loop) e **ADR-0003** (repo/licença) | Governança | Resolve Q1/Q3/Q5 |
| Especificar o **épico de ingestão** (`geosteering_ai/ingest/`) | Planejamento | Sequência de conectores + PWLS + MD↔TVD |
| Definir **plano de validação científica** (caso de campo + critério quantitativo) | Pesquisa | Gate do rótulo "produto" |

### IX.3 Conexão com o que já existe (não duplicar)

- A estratégia **Strangler Fig**, a promoção `gui/` e o cronograma ALPHA já estão em [mvc_simulation_manager_studio_analysis_2026-05-18.md](mvc_simulation_manager_studio_analysis_2026-05-18.md) (R1–R9, 3 caminhos A/B/C). Este relatório **expande** com os 4 pilares, requisitos e a Trilha G.
- A arquitetura **S1–S6 / Hexagonal⊕MVVM** está em [v2.39_proximos_passos §7](v2.39_proximos_passos_roadmap_2026-05-17.md) — compatível com as camadas L0–L5 aqui.
- Os riscos C1 (campo), C7 (latência), C12 (proxy OOD) ecoam o [premortem_geosteering_ai_2026-05-09.md](premortem_geosteering_ai_2026-05-09.md) (§3.1 gap de dados reais, §3.4 latência, §3.8 validação científica) — **convergência forte** de duas análises independentes reforça que estes são os riscos reais.

---

## Anexo A — Fontes de Pesquisa (seleção)

**Mercado:** ROGII StarSteer 2026.1 · Halliburton LOGIX (2025) + Guyana automatizado (2026) · SLB Neuro/GeoSphere HD/PeriScope Edge (2024) · Baker Hughes VisiTrak/AutoTrak (2024) · Petrolink/Corva · *Geosteering Robot vs Human Experts* (SPE Journal 2025) · *Real-Time 2.5D DL Inversion of LWD Resistivity* (SPWLA 2021/2022).

**Padrões de dados:** WITSML 1.4.1.1/2.0 + ETP (Energistics/OSDU) · WITS L0 · LAS/DLIS (`lasio`/`dlisio`) · PWLS 3.0 · OSDU · IEC 62443/Purdue.

**Literatura DL:** INN para UQ de resistividade ultraprofunda (C&G 2025) · Review de PINN-inversão (Geophysics 2024) · Physics-guided DL inversion de LWD ruidoso (GJI 2023) · *Strategic Geosteering Workflow with UQ and DL — Goliat Field* (Geophysics 2024) · cross-gradients (SEG 2021) · MDN multimodal (2022) · model-error assessment de proxy DNN (2022) · NAS para resistividade (GJI 2023).

**MLOps/UX:** MLflow 3.x · ONNX RT vs TFLite/LiteRT (edge) · conda/constructor (napari) · MVVM para PyQt · PyQtGraph/VisPy (SciPy) · governança de modelos (ml-ops.org) · drift (Evidently).

*(Lista completa de URLs nos artefatos do workflow: `/tmp/studio_parts/research_*.md`.)*

---

*Documento de análise exploratória (não segue o template de sprint). Gerado a pedido do usuário. Não modifica `ROADMAP.md`/`CHANGELOG.md` — a formalização da Trilha G e dos ADRs aguarda decisão (Parte VIII), conforme ADR-0001 R1/R2.*
