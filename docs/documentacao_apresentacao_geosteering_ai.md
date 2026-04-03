# Geosteering AI v2.0 — Apresentação Geral do Projeto

> **Documento de entrada** para novos colaboradores, revisores e stakeholders.
> Visão completa do sistema de inversão 1D de resistividade EM via Deep Learning
> para geosteering em tempo real.

---

## Sumário

| Seção | Título | Página |
|:-----:|:-------|:-------|
| 1 | [Introdução](#1-introdução) | Contexto, motivação e analogia |
| 2 | [O Problema Físico](#2-o-problema-físico) | Resistividade, anisotropia TIV, ferramenta LWD |
| 3 | [Arquitetura de Software v2.0](#3-arquitetura-de-software-v20) | Pacote, PipelineConfig, Factory |
| 4 | [Pipeline de Dados](#4-pipeline-de-dados) | DataPipeline, on-the-fly, perspectivas |
| 5 | [Modelos de Deep Learning](#5-modelos-de-deep-learning) | 48 arquiteturas em 9 famílias |
| 6 | [Funções de Perda](#6-funções-de-perda) | 26 losses, PINNs, LossFactory |
| 7 | [Ruído e Curriculum](#7-ruído-e-curriculum) | 34 tipos, curriculum 3 fases |
| 8 | [Treinamento](#8-treinamento) | TrainingLoop, N-Stage, callbacks |
| 9 | [Inferência](#9-inferência) | Offline, realtime, UQ, export |
| 10 | [Avaliação e Visualização](#10-avaliação-e-visualização) | Métricas, DOD Picasso, dashboards |
| 11 | [Roadmap de Desenvolvimento](#11-roadmap-de-desenvolvimento) | Fases F1–F6 |
| 12 | [Documentação do Projeto](#12-documentação-do-projeto) | Catálogo de documentos |
| 13 | [Configuração e Presets](#13-configuração-e-presets) | YAML presets, PipelineConfig |
| 14 | [Referências Bibliográficas](#14-referências-bibliográficas) | ~20 referências-chave |

---

## 1. Introdução

### 1.1 O que é o Geosteering AI?

O **Geosteering AI** é um sistema de **inversão 1D de resistividade eletromagnética
via Deep Learning** projetado para aplicações de **geosteering** — o direcionamento
em tempo real de poços durante a perfuração horizontal.

O objetivo central é traduzir medições eletromagnéticas (EM) capturadas por
ferramentas LWD (*Logging While Drilling*) em **mapas de resistividade** do
subsolo, permitindo ao operador de poço tomar decisões de direcionamento com
informação geofísica atualizada a cada ponto de medição.

### 1.2 Ficha Técnica

| Atributo | Valor |
|:---------|:------|
| **Autor** | Daniel Leal |
| **Versão** | v2.0 (arquitetura de software) |
| **Framework** | TensorFlow 2.x / Keras **EXCLUSIVO** |
| **Ambiente de dev** | VSCode + Claude Code |
| **Ambiente de treino** | Google Colab Pro+ GPU (T4/A100) |
| **CI/CD** | GitHub Actions |
| **Linguagem** | Python 3.10+ |
| **Convenção** | Variáveis em inglês · Comentários/docs em PT-BR |
| **Repositório** | `github.com/daniel-leal/geosteering-ai` |
| **Pacote** | `geosteering_ai/` (pip installable) |
| **Estatísticas** | 73 arquivos Python · ~46.000 LOC · 744 testes CPU |

> **Nota:** PyTorch é **PROIBIDO** em qualquer parte do pipeline. Todo o código
> utiliza exclusivamente TensorFlow 2.x e a API Keras.

### 1.3 Analogia Médica

Para facilitar a compreensão de não-especialistas, a inversão EM para
geosteering pode ser entendida por analogia com a tomografia computadorizada:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ANALOGIA: CT SCAN vs LWD                        │
├───────────────────────┬─────────────────────────────────────────────┤
│  Medicina             │  Geosteering                               │
├───────────────────────┼─────────────────────────────────────────────┤
│  Raios X              │  Ondas eletromagnéticas (20 kHz)           │
│  Corpo humano         │  Formações rochosas do subsolo             │
│  Scanner CT           │  Ferramenta LWD no fundo do poço          │
│  Imagem reconstruída  │  Perfil de resistividade (ρh, ρv)         │
│  Algoritmo de recon.  │  Rede neural profunda (Deep Learning)      │
│  Radiologista         │  Operador de geosteering                   │
└───────────────────────┴─────────────────────────────────────────────┘
```

Assim como um scanner CT emite radiação e reconstrói uma imagem interna do
corpo, a ferramenta LWD emite campos EM e o sistema de Deep Learning reconstrói
o perfil de resistividade das camadas geológicas ao redor do poço.

### 1.4 Motivação

A inversão EM convencional é um **problema mal-posto** (*ill-posed*): múltiplos
modelos de resistividade podem produzir respostas EM indistinguíveis dentro do
ruído de medição. Métodos iterativos tradicionais (e.g., Gauss-Newton, Levenberg-
Marquardt) são lentos e requerem bons pontos de partida.

Redes neurais profundas aprendem o mapeamento inverso diretamente dos dados,
oferecendo:

- **Velocidade**: inferência em milissegundos (vs. minutos/horas)
- **Robustez**: generalização para modelos geológicos não vistos
- **Automatização**: eliminação da necessidade de *initial guess*
- **Tempo real**: compatibilidade com geosteering operacional

---

## 2. O Problema Físico

### 2.1 Resistividade Elétrica

A **resistividade** (ρ, medida em Ω·m) é a propriedade geofísica primária para
diferenciação de fluidos e litologias no subsolo:

```
┌──────────────────────────────────────────────────────────────────┐
│              RESISTIVIDADE TÍPICA POR LITOLOGIA                  │
├────────────────────┬────────────────┬────────────────────────────┤
│  Material          │  ρ (Ω·m)      │  Observação                │
├────────────────────┼────────────────┼────────────────────────────┤
│  Água salgada      │  0.1 – 5      │  Condutor (baixa ρ)        │
│  Folhelho (shale)  │  1 – 10       │  Condutor a moderado       │
│  Arenito c/ óleo   │  50 – 500     │  Resistivo (alta ρ)        │
│  Carbonato         │  100 – 10000  │  Muito resistivo           │
│  Anidrita          │  1000+        │  Extremamente resistivo    │
└────────────────────┴────────────────┴────────────────────────────┘
```

A capacidade de distinguir **água** (baixa ρ) de **hidrocarboneto** (alta ρ) é
a base da perfilagem de resistividade e, por extensão, do geosteering.

### 2.2 Anisotropia TIV

Formações sedimentares apresentam **anisotropia transversal isotrópica vertical**
(TIV — *Transverse Isotropy with a Vertical axis of symmetry*):

- **ρh** (*horizontal resistivity*): resistividade no plano de acamamento
- **ρv** (*vertical resistivity*): resistividade perpendicular ao acamamento

A relação fundamental é:

```
ρv ≥ ρh   (sempre)

Razão de anisotropia:  λ = √(ρv / ρh) ≥ 1.0
```

Em folhelhos laminados, a razão λ pode exceder 10. Modelos que ignoram a
anisotropia produzem inversões com artefatos significativos.

**O Geosteering AI inverte ambas as componentes (ρh e ρv) simultaneamente**,
capturando a anisotropia TIV completa.

### 2.3 Ferramenta LWD e Tensor EM

A ferramenta LWD (*Logging While Drilling*) opera com:

- **Transmissor**: bobina de 20 kHz (frequência padrão)
- **Receptores**: espaçamento L = 1.0 m do transmissor
- **Resposta**: tensor magnético H completo (3×3 = 9 componentes complexas)

```
┌─────────────────────────────────────────────────────────────────┐
│                    TENSOR H (3×3)                               │
│                                                                 │
│         ┌─────────┬─────────┬─────────┐                        │
│         │  Hxx    │  Hxy    │  Hxz    │                        │
│    H =  │  Hyx    │  Hyy    │  Hyz    │   (campo magnético     │
│         │  Hzx    │  Hzy    │  Hzz    │    secundário, A/m)    │
│         └─────────┴─────────┴─────────┘                        │
│                                                                 │
│    Cada componente é complexa: Re(Hij) + j·Im(Hij)             │
│    Total: 9 componentes × 2 (Re + Im) = 18 valores reais      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Formato de 22 Colunas

Os dados de simulação (gerados por software Fortran 1D) seguem o formato
padronizado de **22 colunas**:

```
┌─────┬──────────┬───────────────────────────────────────────────────┐
│ Col │ Variável │ Descrição                                         │
├─────┼──────────┼───────────────────────────────────────────────────┤
│  0  │ index    │ Índice sequencial do ponto de medição             │
│  1  │ z_obs    │ Profundidade de observação (m)                    │
│  2  │ ρh       │ Resistividade horizontal (Ω·m) — TARGET          │
│  3  │ ρv       │ Resistividade vertical (Ω·m)   — TARGET          │
│  4  │ Re(Hxx)  │ Parte real de Hxx  — INPUT                       │
│  5  │ Im(Hxx)  │ Parte imaginária de Hxx — INPUT                  │
│  6  │ Re(Hxy)  │ Parte real de Hxy                                │
│  7  │ Im(Hxy)  │ Parte imaginária de Hxy                          │
│  8  │ Re(Hxz)  │ Parte real de Hxz                                │
│  9  │ Im(Hxz)  │ Parte imaginária de Hxz                          │
│ 10  │ Re(Hyx)  │ Parte real de Hyx                                │
│ 11  │ Im(Hyx)  │ Parte imaginária de Hyx                          │
│ 12  │ Re(Hyy)  │ Parte real de Hyy                                │
│ 13  │ Im(Hyy)  │ Parte imaginária de Hyy                          │
│ 14  │ Re(Hyz)  │ Parte real de Hyz                                │
│ 15  │ Im(Hyz)  │ Parte imaginária de Hyz                          │
│ 16  │ Re(Hzx)  │ Parte real de Hzx                                │
│ 17  │ Im(Hzx)  │ Parte imaginária de Hzx                          │
│ 18  │ Re(Hzy)  │ Parte real de Hzy                                │
│ 19  │ Im(Hzy)  │ Parte imaginária de Hzy                          │
│ 20  │ Re(Hzz)  │ Parte real de Hzz  — INPUT                       │
│ 21  │ Im(Hzz)  │ Parte imaginária de Hzz — INPUT                  │
└─────┴──────────┴───────────────────────────────────────────────────┘
```

### 2.5 Features de Entrada e Targets de Saída

**Baseline (P1 — 5 features):**

```
INPUT_FEATURES = [1, 4, 5, 20, 21]
  → z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)

OUTPUT_TARGETS = [2, 3]
  → ρh, ρv   (em escala log10)
```

**Justificativa física:**
- **Hxx** (componente planar): sensível a variações horizontais de resistividade
- **Hzz** (componente axial): sensível a variações verticais de resistividade
- **z_obs**: posição absoluta, necessária para contextualizar a medição
- Componentes Re + Im capturam tanto a atenuação quanto a fase do sinal EM

**Escala dos targets:**
- Resistividade em **log10** (NUNCA "log" natural)
- Justificativa: ρ varia 4+ ordens de magnitude (0.1 → 10000 Ω·m)
- log10 normaliza a faixa dinâmica para treinamento estável

### 2.6 Constantes de Decoupling

O decoupling remove a resposta do espaço livre (*free-space*) do sinal medido:

```python
# Para espaçamento L = 1.0 m:
ACp = -1 / (4 * π * L³) ≈ -0.079577   # planar: Hxx, Hyy
ACx = +1 / (2 * π * L³) ≈ +0.159155   # axial:  Hzz
```

---

## 3. Arquitetura de Software v2.0

### 3.1 Visão Geral do Pacote

O Geosteering AI v2.0 é um **pacote Python pip-installable** organizado em 11
subpacotes com responsabilidades claramente definidas:

```
geosteering_ai/
├── __init__.py
├── config.py              ← PipelineConfig dataclass (246 campos)
│
├── data/                  ← Carregamento, split, FV, GS, scaling, pipeline
│   ├── loading.py         │  Leitura de arquivos .out (22 colunas)
│   ├── splitting.py       │  Split por modelo geológico [P1]
│   ├── feature_views/     │  7 visões de features (FV)
│   ├── geosignals/        │  5 famílias de geosinais (GS)
│   ├── scaling.py         │  8 scalers (fit em dados LIMPOS)
│   ├── pipeline.py        │  DataPipeline (orquestrador)
│   ├── inspection.py      │  EDA e validação de dados
│   ├── boundaries.py      │  DTB — Distance To Boundary [P5]
│   ├── sampling.py        │  Amostragem estratificada
│   ├── second_order.py    │  Derivadas de 2ª ordem
│   └── surrogate_data.py  │  Dados para SurrogateNet
│
├── noise/                 ← Ruído on-the-fly
│   ├── functions.py       │  34 tipos de ruído em 6 categorias
│   └── curriculum.py      │  Curriculum 3 fases (Clean→Ramp→Stable)
│
├── models/                ← 48 arquiteturas + ModelRegistry
│   ├── registry.py        │  ModelRegistry (Factory)
│   ├── cnn.py             │  ResNet, ConvNeXt, Inception, ResNeXt
│   ├── tcn.py             │  TCN, TCN_Advanced
│   ├── modern_tcn.py      │  ModernTCN
│   ├── rnn.py             │  LSTM, BiLSTM
│   ├── hybrid.py          │  CNN_LSTM, CNN_BiLSTM_ED, ResNeXt_LSTM
│   ├── unet.py            │  7 bases × 2 (±atenção) = 14 variantes
│   ├── transformer.py     │  Transformer, TFT, PatchTST, etc.
│   ├── advanced.py        │  DNN, FNO, DeepONet, INN
│   ├── geosteering.py     │  WaveNet, Mamba_S4, Informer, etc.
│   ├── blocks.py          │  23 blocos reutilizáveis
│   ├── surrogate.py       │  SurrogateNet (TCN + ModernTCN)
│   └── layers.py          │  Camadas customizadas (TIVConstraint)
│
├── losses/                ← 26 funções de perda + PINNs
│   ├── catalog.py         │  Catálogo completo (26 losses)
│   ├── factory.py         │  LossFactory + build_combined
│   └── pinns.py           │  8 cenários PINN + TIV constraint
│
├── training/              ← Loop de treinamento
│   ├── loop.py            │  TrainingLoop (compile → fit → finetune)
│   ├── callbacks.py       │  17+ callbacks customizados
│   ├── metrics.py         │  Métricas de treinamento
│   ├── nstage.py          │  Treinamento N-Stage (N=2,3,4)
│   ├── optuna.py          │  HPO com Optuna
│   └── adaptation.py      │  Adaptação de domínio
│
├── inference/             ← Inferência
│   ├── pipeline.py        │  InferencePipeline (batch offline)
│   ├── realtime.py        │  RealtimeInference (sliding window)
│   ├── export.py          │  Export: SavedModel, TFLite, ONNX
│   └── uncertainty.py     │  UQ: MC Dropout, Ensemble, INN
│
├── evaluation/            ← Avaliação
│   ├── metrics.py         │  R², RMSE, MAE, MBE, MAPE
│   ├── comparison.py      │  Comparação entre modelos
│   ├── picasso.py         │  DOD Picasso (6 métodos)
│   ├── reports.py         │  Relatórios Markdown + JSON
│   └── ...                │  (11 módulos total)
│
├── visualization/         ← Visualização
│   ├── eda.py             │  Análise exploratória
│   ├── holdout.py         │  Plots de holdout
│   ├── training.py        │  Curvas de treinamento
│   ├── error_maps.py      │  Mapas de erro
│   ├── geosteering.py     │  Dashboards de geosteering
│   └── ...                │  (11 módulos total)
│
└── utils/                 ← Utilitários
    ├── logger.py          │  Logging estruturado (NUNCA print)
    ├── timer.py           │  Medição de tempo
    ├── validation.py      │  Validações de entrada
    ├── formatting.py      │  Formatação de saída
    ├── system.py          │  Info do sistema (GPU, memória)
    └── io.py              │  I/O de arquivos
```

### 3.2 PipelineConfig — Ponto Único de Verdade

O `PipelineConfig` é um **dataclass** Python com **246 campos** que centraliza
toda a configuração do pipeline. Nenhuma função lê variáveis globais — todas
recebem `config: PipelineConfig` como parâmetro.

```
┌─────────────────────────────────────────────────────────────────┐
│                     PipelineConfig                               │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Dados       │  │  Modelo      │  │  Treinamento │          │
│  │  • seq_len   │  │  • model_type│  │  • epochs    │          │
│  │  • features  │  │  • filters   │  │  • lr        │          │
│  │  • targets   │  │  • blocks    │  │  • batch_size│          │
│  │  • scaler    │  │  • dropout   │  │  • optimizer │          │
│  │  • FV / GS   │  │  • causal    │  │  • scheduler │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Ruído       │  │  Loss        │  │  PINN        │          │
│  │  • noise_type│  │  • loss_type │  │  • pinn_on   │          │
│  │  • std       │  │  • weights   │  │  • scenario  │          │
│  │  • curriculum│  │  • combined  │  │  • lambda    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  __post_init__() → Validações da Errata Imutável               │
│  • FREQUENCY_HZ ∈ [100, 1e6]   (default: 20000.0)             │
│  • SPACING_METERS ∈ [0.1, 10]  (default: 1.0)                 │
│  • SEQUENCE_LENGTH ∈ [10, 1e5] (default: 600)                  │
│  • TARGET_SCALING == "log10"                                    │
│  • INPUT_FEATURES == [1,4,5,20,21]                              │
│  • OUTPUT_TARGETS == [2,3]                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Padrões de Projeto

O código v2.0 segue três padrões fundamentais:

**1. Factory Pattern — componentes intercambiáveis**

```python
model    = ModelRegistry().build(config)          # 48 arquiteturas
loss_fn  = LossFactory.get(config)                # 26 losses
callbacks = build_callbacks(config, model, noise)  # 17+ callbacks
```

**2. PipelineConfig como parâmetro — NUNCA globals**

```python
# CORRETO:
def build_model(config: PipelineConfig) -> tf.keras.Model:
    ...

# PROIBIDO:
def build_model():
    model_type = globals().get("MODEL_TYPE", "ResNet_18")  # NÃO!
```

**3. DataPipeline com cadeia explícita**

```python
pipeline = DataPipeline(config)
data = pipeline.prepare(dataset_path)
map_fn = pipeline.build_train_map_fn(noise_var)
```

### 3.4 Estatísticas do Código

| Métrica | Valor |
|:--------|:------|
| Arquivos Python | 73 |
| Linhas de código (LOC) | ~46.000 |
| Testes unitários (CPU) | 744 passed |
| Testes com GPU | 1011+ passed |
| Campos PipelineConfig | 246 (19 PINN/Surrogate) |
| Presets YAML | 7 |
| Documentos .md | 48+ |

---

## 4. Pipeline de Dados

### 4.1 Fluxo Completo

O `DataPipeline` orquestra toda a cadeia de dados, desde o carregamento até
o `tf.data.Dataset` pronto para treinamento:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PIPELINE DE DADOS COMPLETO                        │
│                                                                     │
│  ┌─────────┐    ┌──────────┐    ┌──────┐    ┌──────┐    ┌───────┐ │
│  │  .out   │ →  │ load_22  │ →  │ split│ →  │ FV   │ →  │ scale │ │
│  │ (Fortran│    │ (22 col) │    │ [P1] │    │ (7)  │    │ (fit  │ │
│  │  files) │    │          │    │      │    │      │    │ clean)│ │
│  └─────────┘    └──────────┘    └──────┘    └──────┘    └───────┘ │
│                                                                     │
│  TREINAMENTO (on-the-fly via tf.data.map):                         │
│  ┌──────────┐    ┌──────────┐    ┌──────┐    ┌───────┐            │
│  │ raw data │ →  │ noise(A/m│ →  │ FV   │ →  │ GS    │ → scale   │
│  │ (limpo)  │    │ on-the-  │    │ (com │    │ (veem │   → modelo │
│  │          │    │ fly)     │    │noise)│    │noise) │            │
│  └──────────┘    └──────────┘    └──────┘    └───────┘            │
│                                                                     │
│  VAL/TEST (offline, sem ruído):                                    │
│  ┌──────────┐    ┌──────┐    ┌──────┐    ┌───────┐                │
│  │ raw data │ →  │ FV   │ →  │ GS   │ →  │ scale │ → modelo      │
│  │ (limpo)  │    │(clean│    │(clean│    │(trans-│                │
│  │          │    │)     │    │)     │    │form)  │                │
│  └──────────┘    └──────┘    └──────┘    └───────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Princípios Fundamentais

1. **Split por modelo geológico** — NUNCA por amostra individual.
   Evita vazamento de informação (*data leakage*) entre treino e teste.

2. **Scaler fitado em dados LIMPOS** — O scaler (e.g., StandardScaler) é
   ajustado nos dados clean (FV + GS sem ruído). Dados ruidosos de treino
   são transformados usando os parâmetros do scaler limpo.

3. **On-the-fly é o ÚNICO path correto** — O ruído é adicionado a cada epoch
   via `tf.data.map`, garantindo que:
   - Cada epoch vê realizações diferentes de ruído
   - As Feature Views (FV) são calculadas sobre dados ruidosos
   - Os Geosinais (GS) veem o ruído → fidelidade física com LWD real

4. **Noise offline com GS é PROIBIDO** — Se o ruído fosse adicionado após as
   GS, os geosinais seriam calculados sobre dados limpos e depois corrompidos
   artificialmente. Isso viola a física do instrumento LWD.

### 4.3 As 5 Perspectivas

O sistema suporta 5 perspectivas de dados, cada uma com foco diferente:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      5 PERSPECTIVAS DE DADOS                        │
├──────┬──────────────┬────────────┬──────────────────────────────────┤
│ Persp│ Nome         │ Features   │ Objetivo                         │
├──────┼──────────────┼────────────┼──────────────────────────────────┤
│  P1  │ Baseline     │ 5 (H+z)   │ Inversão padrão, 1 freq, 0° dip │
│  P2  │ Angle        │ 5+ângulo   │ Generalização multi-dip          │
│  P3  │ Frequency    │ 5×N_freq   │ Múltiplas frequências            │
│  P4  │ Geosignals   │ 5+GS      │ Atributos geofísicos derivados   │
│  P5  │ DTB          │ 5+dtb     │ Distance To Boundary             │
└──────┴──────────────┴────────────┴──────────────────────────────────┘
```

### 4.4 Feature Views (FV)

As 7 Feature Views transformam os dados brutos em representações otimizadas
para a rede neural:

| FV | Nome | Transformação | Motivação |
|:---|:-----|:-------------|:----------|
| FV0 | Raw | Sem transformação | Baseline |
| FV1 | Log_magnitude | log10(\|H\|) | Comprime faixa dinâmica |
| FV2 | Phase | φ(H) = atan2(Im, Re) | Informação de fase |
| FV3 | Amplitude_Phase | (\|H\|, φ) | Representação polar |
| FV4 | H1_logH2 | H1 cru, log10\|H2\| | Híbrido: SNR + compressão |
| FV5 | Normalized | H / max(\|H\|) | Normalização por amplitude |
| FV6 | Differential | ΔH entre receptores | Sensibilidade local |

### 4.5 Geosinais (GS)

As 5 famílias de geosinais extraem atributos geofísicos derivados:

| GS | Família | Atributos | Significado Físico |
|:---|:--------|:----------|:------------------|
| GS1 | Atenuação/Fase | att, phase_diff | Propriedades tradicionais LWD |
| GS2 | Impedância | Z, R, X | Impedância de onda EM |
| GS3 | Anisotropia | λ, ρh/ρv ratio | Indicadores de anisotropia TIV |
| GS4 | Derivadas | dH/dz, d²H/dz² | Sensibilidade a interfaces |
| GS5 | Estatísticos | mean, std, skew | Momentos estatísticos locais |

### 4.6 Scalers

| Scaler | Descrição | Uso Recomendado |
|:-------|:----------|:---------------|
| Standard | (x - μ) / σ | Default para dados EM |
| MinMax | (x - min) / (max - min) | Quando range é conhecido |
| Robust | (x - median) / IQR | Dados com outliers |
| MaxAbs | x / max(\|x\|) | Preserva esparsidade |
| Log | log10(x + ε) | Dados positivos com faixa larga |
| PowerTransformer | Yeo-Johnson | Aproximar normalidade |
| QuantileTransformer | Quantil → normal | Robustez extrema |
| Identity | Sem transformação | Debugging/referência |

---

## 5. Modelos de Deep Learning

### 5.1 Visão Geral: 48 Arquiteturas em 9 Famílias

O Geosteering AI implementa **48 arquiteturas de redes neurais** organizadas em
**9 famílias**, cobrindo desde CNNs clássicas até modelos de estado-espaço:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  48 ARQUITETURAS — 9 FAMÍLIAS                       │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │   CNN (8)    │  │   TCN (3)   │  │   RNN (2)   │                │
│  │  ResNet-18 ★ │  │  TCN        │  │  LSTM       │                │
│  │  ResNet-34   │  │  TCN_Adv    │  │  BiLSTM     │                │
│  │  ResNet-50   │  │  ModernTCN  │  │             │                │
│  │  ConvNeXt    │  │             │  │             │                │
│  │  Inception   │  └─────────────┘  └─────────────┘                │
│  │  ResNeXt     │                                                   │
│  │  ...         │  ┌─────────────┐  ┌─────────────┐                │
│  └─────────────┘  │ Hybrid (3)  │  │ UNet (14)   │                │
│                    │ CNN_LSTM    │  │ 7 bases ×   │                │
│  ┌─────────────┐  │ CNN_BiED    │  │ 2 (±attn)   │                │
│  │ Transf. (6) │  │ ResNeXt_LSTM│  │             │                │
│  │ Transformer  │  └─────────────┘  └─────────────┘                │
│  │ TFT          │                                                   │
│  │ PatchTST     │  ┌─────────────┐  ┌─────────────┐                │
│  │ Autoformer   │  │ Decomp. (2) │  │ GeoSteer(5) │                │
│  │ iTransformer │  │ N-BEATS     │  │ WaveNet     │                │
│  │ ...          │  │ N-HiTS      │  │ CausalTF    │                │
│  └─────────────┘  └─────────────┘  │ Informer    │                │
│                                     │ Mamba_S4    │                │
│  ┌─────────────┐                    │ Enc_Forecast│                │
│  │ Advanced (5) │                    └─────────────┘                │
│  │ DNN          │                                                   │
│  │ FNO          │  ★ = Default (ResNet-18)                          │
│  │ DeepONet     │                                                   │
│  │ Geo_Attn     │  Todas suportam dual-mode:                       │
│  │ INN          │    "same"   → Offline (acausal)                  │
│  └─────────────┘    "causal" → Realtime (causal)                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Detalhamento por Família

#### CNN — Redes Convolucionais (8 modelos)

| Modelo | Blocos | Parâmetros típicos | Destaque |
|:-------|:------:|:------------------:|:---------|
| **ResNet-18** ★ | 8 | ~1M | Default, skip connections |
| ResNet-34 | 16 | ~2M | Mais profundo, mesma estrutura |
| ResNet-50 | 16 | ~3M | Bottleneck blocks |
| ConvNeXt | 12 | ~2M | Design modernizado (2022) |
| Inception | 9 | ~2M | Multi-scale kernels |
| ResNeXt | 8 | ~2M | Grouped convolutions |
| Simple_CNN | 4 | ~200K | Baseline minimalista |
| Deep_CNN | 12 | ~1.5M | CNN profunda sem skip |

#### TCN — Temporal Convolutional Networks (3 modelos)

| Modelo | Camadas | Dilatação | Destaque |
|:-------|:-------:|:---------:|:---------|
| TCN | 6 | 1,2,4,8,16,32 | Causal convolutions clássico |
| TCN_Advanced | 8 | 1→128 | + SE blocks + residual |
| **ModernTCN** | 8 | 1→128 | Depth-wise separable, SOTA |

#### RNN — Redes Recorrentes (2 modelos)

| Modelo | Direção | Destaque |
|:-------|:-------:|:---------|
| LSTM | Unidirecional | Memória longa, sequencial |
| BiLSTM | Bidirecional | Contexto completo (offline) |

#### Hybrid — Modelos Híbridos (3 modelos)

| Modelo | Componentes | Destaque |
|:-------|:-----------|:---------|
| CNN_LSTM | ResNet + LSTM | Features locais + contexto |
| CNN_BiLSTM_ED | ResNet + BiLSTM + Decoder | Encoder-Decoder |
| ResNeXt_LSTM | ResNeXt + LSTM | Grouped conv + sequencial |

#### UNet — Redes Encoder-Decoder (14 modelos)

Baseadas na arquitetura U-Net com skip connections entre encoder e decoder:

| Base | Sem Atenção | Com Atenção |
|:-----|:----------:|:----------:|
| ResNet-18 | UNet_ResNet18 | UNet_ResNet18_Attn |
| ResNet-34 | UNet_ResNet34 | UNet_ResNet34_Attn |
| ConvNeXt | UNet_ConvNeXt | UNet_ConvNeXt_Attn |
| Inception | UNet_Inception | UNet_Inception_Attn |
| TCN | UNet_TCN | UNet_TCN_Attn |
| LSTM | UNet_LSTM | UNet_LSTM_Attn |
| Transformer | UNet_TF | UNet_TF_Attn |

#### Transformer — Modelos de Atenção (6 modelos)

| Modelo | Atenção | Destaque |
|:-------|:--------|:---------|
| Transformer | Multi-head | Baseline, self-attention |
| TFT | Multi-horizon | Temporal Fusion Transformer |
| PatchTST | Patched | Patches como tokens |
| Autoformer | Auto-correlation | Decomposição série temporal |
| iTransformer | Inverted | Atenção sobre features |
| Crossformer | Cross-dim | Atenção cruzada dim×tempo |

#### Decomposition — Modelos de Decomposição (2 modelos)

| Modelo | Bases | Destaque |
|:-------|:-----:|:---------|
| N-BEATS | Trend+Season | Backcast/forecast explicável |
| N-HiTS | Multi-rate | Amostragem hierárquica |

#### Advanced — Modelos Avançados (5 modelos)

| Modelo | Tipo | Destaque |
|:-------|:-----|:---------|
| DNN | Feedforward | MLP profundo, referência |
| FNO | Operador | Fourier Neural Operator |
| DeepONet | Operador | Branch/trunk architecture |
| Geo_Attn | Atenção | Atenção geofísica customizada |
| **INN** | Invertível | Posterior sampling, UQ nativo |

#### Geosteering — Modelos para Tempo Real (5 modelos)

| Modelo | Tipo | Latência | Destaque |
|:-------|:-----|:--------:|:---------|
| WaveNet | Causal Conv | Baixa | Dilatação causal, streaming |
| Causal_TF | Causal Attn | Média | Atenção causal masked |
| Informer | ProbSparse | Média | Atenção esparsa O(L·log L) |
| Mamba_S4 | State-Space | Muito baixa | Recorrência linear O(L) |
| Enc_Forecast | Enc→Dec | Média | Encoder + forecasting head |

### 5.3 ModelRegistry — Factory Pattern

Todas as arquiteturas são instanciadas via `ModelRegistry`, eliminando
blocos `if/elif` extensos:

```python
# Construção de qualquer modelo via config:
model = ModelRegistry().build(config)

# Equivalente a:
# if config.model_type == "ResNet_18": ...
# elif config.model_type == "TCN": ...
# (× 48 arquiteturas) → NÃO!
```

### 5.4 Dual-Mode: Offline vs Causal

Todos os modelos suportam operação em dois modos:

```
┌──────────────────────────────────────────────────────────────────┐
│  MODO OFFLINE (acausal) — padding="same"                        │
│  • Acesso ao contexto futuro e passado                          │
│  • Usado em processamento batch (pós-perfuração)                │
│  • Máxima acurácia                                              │
├──────────────────────────────────────────────────────────────────┤
│  MODO CAUSAL — padding="causal"                                 │
│  • Apenas contexto passado (sem leak do futuro)                 │
│  • Usado em geosteering em tempo real                           │
│  • Compatível com sliding window                                │
│  • Ligeira perda de acurácia nas bordas                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Funções de Perda

### 6.1 Catálogo: 26 Losses

O sistema implementa **26 funções de perda** organizadas em 4 categorias:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    26 FUNÇÕES DE PERDA                               │
├──────────────────────┬──────────────────────────────────────────────┤
│  GENÉRICAS (13)      │  MSE, RMSE, MAE, Huber, LogCosh,           │
│                      │  MSLE, MAPE, Quantile, Tukey,              │
│                      │  CauchyNLL, StudentT_NLL,                  │
│                      │  HeteroscedasticNLL, LaplacianNLL          │
├──────────────────────┼──────────────────────────────────────────────┤
│  GEOFÍSICAS (4)      │  WeightedMSE_TIV, Log10_MSE,              │
│                      │  AnisotropyAware, BoundaryFocused          │
├──────────────────────┼──────────────────────────────────────────────┤
│  GEOSTEERING (2)     │  CausalWeighted, SlidingWindow             │
├──────────────────────┼──────────────────────────────────────────────┤
│  AVANÇADAS (7)       │  FocalLoss, DiceLoss, TverskyLoss,         │
│                      │  ContrastiveLoss, TripletLoss,             │
│                      │  SSIM_Loss, PerceptualLoss                 │
└──────────────────────┴──────────────────────────────────────────────┘
```

### 6.2 Losses Geofísicas — Detalhe

| Loss | Fórmula Simplificada | Motivação Física |
|:-----|:--------------------|:-----------------|
| WeightedMSE_TIV | w_h·MSE(ρh) + w_v·MSE(ρv) | Peso diferenciado por componente |
| Log10_MSE | MSE(log10(ρ_pred), log10(ρ_true)) | Opera no espaço log natural |
| AnisotropyAware | MSE + λ·\|ρv_pred - ρh_pred\|₋ | Penaliza ρv < ρh (violação TIV) |
| BoundaryFocused | MSE × w(z), w alto em interfaces | Maior peso em transições de camada |

### 6.3 PINNs — Physics-Informed Neural Networks

O sistema implementa **8 cenários PINN** que incorporam leis físicas como
termos de regularização na função de perda:

| Cenário | Lei Física | Regularização |
|:--------|:----------|:-------------|
| 1 | TIV constraint (ρv ≥ ρh) | Penalidade em violações |
| 2 | Continuidade lateral | Suavidade entre pontos adjacentes |
| 3 | Maxwell (∇×E = -jωμH) | Consistência EM |
| 4 | Condições de contorno | ρ → ρ_background nas bordas |
| 5 | Conservação de corrente | ∇·J = 0 |
| 6 | Reciprocidade | H_ij = H_ji (simetria do tensor) |
| 7 | Monotonicidade camada | ρ constante dentro de camada |
| 8 | Petrofísica (futuro) | Archie + Klein constraints |

**Loss combinada com PINN:**

```
L_total = L_data + λ₁·L_TIV + λ₂·L_smooth + λ₃·L_physics + ...
```

O peso λ de cada termo PINN pode seguir um schedule (crescente durante o
treinamento) para estabilidade.

### 6.4 LossFactory

```python
# Loss simples:
loss_fn = LossFactory.get(config)

# Loss combinada (data + PINN):
loss_fn = LossFactory.build_combined(config)
# Automaticamente monta L_data + Σ λᵢ·L_pinnᵢ
```

---

## 7. Ruído e Curriculum

### 7.1 Motivação

Dados de simulação Fortran são **limpos** (sem ruído). Dados LWD reais contêm
ruído de diversas fontes: eletrônica, vibrações de perfuração, efeitos de
borehole, etc. Para que o modelo generalize para dados reais, é essencial
treinar com **ruído realista**.

### 7.2 Catálogo: 34 Tipos de Ruído em 6 Categorias

```
┌─────────────────────────────────────────────────────────────────────┐
│                    34 TIPOS DE RUÍDO                                 │
├──────────────────────┬──────────────────────────────────────────────┤
│  GAUSSIANO (6)       │  gaussian, gaussian_relative,               │
│                      │  gaussian_snr, gaussian_complex,            │
│                      │  gaussian_correlated, gaussian_heterosc     │
├──────────────────────┼──────────────────────────────────────────────┤
│  INSTRUMENTAL (6)    │  thermal, quantization, calibration,        │
│                      │  drift, saturation, cross_talk              │
├──────────────────────┼──────────────────────────────────────────────┤
│  AMBIENTAL (6)       │  borehole, vibration, formation,            │
│                      │  temperature, pressure, mud_invasion        │
├──────────────────────┼──────────────────────────────────────────────┤
│  COMBINADO (6)       │  realistic_lwd, field_like, harsh,          │
│                      │  mild, moderate, custom_blend               │
├──────────────────────┼──────────────────────────────────────────────┤
│  ADVERSARIAL (6)     │  spike, dropout_noise, phase_shift,         │
│                      │  frequency_leak, systematic_bias,           │
│                      │  amplitude_modulation                       │
├──────────────────────┼──────────────────────────────────────────────┤
│  DISTRIBUIÇÃO (4)    │  uniform, laplacian, student_t, cauchy      │
└──────────────────────┴──────────────────────────────────────────────┘
```

### 7.3 Curriculum 3 Fases

O treinamento com ruído segue um **curriculum de 3 fases** que introduz
complexidade gradualmente:

```
┌─────────────────────────────────────────────────────────────────────┐
│                CURRICULUM DE RUÍDO — 3 FASES                        │
│                                                                     │
│  Noise                                                              │
│  Level                                                              │
│  ▲                                                                  │
│  │                              ┌─────────────────────              │
│  │                             ╱  Fase 3: STABLE                    │
│  │  σ_final ─ ─ ─ ─ ─ ─ ─ ─ ╱   (σ = σ_final)                    │
│  │                          ╱                                       │
│  │                        ╱   Fase 2: RAMP                          │
│  │                      ╱     (σ: 0 → σ_final)                     │
│  │                    ╱                                              │
│  │  0 ──────────────┘   Fase 1: CLEAN                               │
│  │                      (σ = 0, sem ruído)                          │
│  └──────────────────────────────────────────────────────────── ►    │
│     0        e₁              e₂              e₃       epochs        │
│              (clean_end)     (ramp_end)                              │
└─────────────────────────────────────────────────────────────────────┘
```

- **Fase 1 (Clean)**: O modelo aprende a mapear H → ρ sem distração de ruído
- **Fase 2 (Ramp)**: Ruído aumenta linearmente, forçando robustez gradual
- **Fase 3 (Stable)**: Ruído no nível final, simula condições operacionais

### 7.4 Princípio Físico: On-the-Fly

```
train_raw → noise(A/m) → FV_tf(noisy) → GS_tf(noisy) → scale → modelo
                                              │
                                    GS veem ruído ✓ (fidelidade LWD)
```

O ruído é adicionado **antes** das Feature Views e Geosinais, garantindo que
todos os atributos derivados reflitam o ruído real do instrumento. Isso é
crítico para fidelidade física.

---

## 8. Treinamento

### 8.1 TrainingLoop

O `TrainingLoop` encapsula todo o ciclo de treinamento:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                                  │
│                                                                     │
│  1. compile(model, optimizer, loss_fn, metrics)                     │
│     │                                                               │
│  2. fit(train_ds, val_ds, epochs, callbacks)                        │
│     │  ├── DualValidation (clean + noisy val)                       │
│     │  ├── NoiseCurriculum (3 fases)                                │
│     │  ├── PINN λ schedule                                          │
│     │  ├── LR schedule (cosine, warmup)                             │
│     │  ├── EarlyStopping                                            │
│     │  ├── ModelCheckpoint                                          │
│     │  └── ... (17+ callbacks)                                      │
│     │                                                               │
│  3. causal_finetune(model, causal_ds)  [opcional]                   │
│     │  └── Fine-tuning com padding causal para geosteering          │
│     │                                                               │
│  4. evaluate(test_ds) → métricas finais                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Treinamento N-Stage

O N-Stage divide o treinamento em N etapas com complexidade crescente:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   N-STAGE TRAINING (N=3)                            │
│                                                                     │
│  Stage 1: Clean                                                     │
│  • Sem ruído, loss simples (MSE)                                    │
│  • LR alto, epochs: 50                                              │
│  • Objetivo: convergência rápida                                    │
│                                                                     │
│  Stage 2: Robust                                                    │
│  • Ruído gaussiano, loss robusta (Huber)                            │
│  • LR médio, epochs: 100                                            │
│  • Objetivo: generalização                                          │
│                                                                     │
│  Stage 3: Fine-tune                                                 │
│  • Ruído realista (realistic_lwd), loss combinada + PINN            │
│  • LR baixo, epochs: 50                                             │
│  • Objetivo: precisão final + constraints físicos                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 Callbacks (17+)

| Callback | Função |
|:---------|:-------|
| DualValidation | Avalia em val limpo E val ruidoso |
| NoiseCurriculum | Controla as 3 fases de ruído |
| PINN_Schedule | Aumenta λ dos termos PINN gradualmente |
| LRWarmupCosine | Warmup linear + cosine decay |
| EarlyStopping | Para treinamento se val_loss estagna |
| ModelCheckpoint | Salva melhor modelo |
| TensorBoard | Logging para visualização |
| GradientMonitor | Monitora norma dos gradientes |
| WeightDecaySchedule | Schedule de weight decay |
| MetricLogger | Log de métricas customizadas |
| ProfilerCallback | Profiling de performance |
| MemoryMonitor | Monitora uso de memória GPU |
| NaNTerminate | Para se loss = NaN |
| CSVLogger | Log em CSV |
| ReduceLROnPlateau | Reduz LR se métrica estagna |
| TimeEstimator | Estima tempo restante |
| StageTransition | Gerencia transições N-Stage |

### 8.4 Otimização de Hiperparâmetros

O sistema integra **Optuna** para busca automática de hiperparâmetros:

- Espaço de busca: LR, batch_size, dropout, filtros, blocos, loss, noise_std
- Estratégias: TPE (*Tree-structured Parzen Estimator*), CMA-ES
- Poda (*pruning*): MedianPruner para trials sem promessa
- Métricas: R² no val clean como objetivo principal

### 8.5 Mixed Precision e XLA

O pipeline está preparado para otimizações de performance:

- **Mixed Precision (FP16)**: reduz uso de memória e acelera em GPUs modernas
- **XLA Compilation**: compilação just-in-time para grafos otimizados
- **tf.data optimization**: prefetch, interleave, cache

---

## 9. Inferência

### 9.1 Modo Offline — InferencePipeline

Para processamento batch de dados já adquiridos (pós-perfuração):

```
┌─────────────────────────────────────────────────────────────────────┐
│                  INFERÊNCIA OFFLINE                                  │
│                                                                     │
│  dados_22col.out → load → FV → GS → scale → modelo → ρh, ρv       │
│                                                                     │
│  • Processa arquivo completo de uma vez                             │
│  • Usa padding="same" (acausal) para máxima acurácia                │
│  • Suporta batching para arquivos grandes                           │
│  • Saída: arrays NumPy ou DataFrame com ρh, ρv preditos            │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Modo Realtime — RealtimeInference

Para geosteering operacional durante a perfuração:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  INFERÊNCIA REALTIME                                 │
│                                                                     │
│  ┌──────────────────────────────────────┐                           │
│  │  Sliding Window (causal)             │                           │
│  │  ┌───┬───┬───┬───┬───┬───┬───┐      │                           │
│  │  │ t │t-1│t-2│t-3│t-4│t-5│...│      │                           │
│  │  └───┴───┴───┴───┴───┴───┴───┘      │                           │
│  │  ↓ novo ponto a cada ~30 segundos    │                           │
│  │  → predição ρh(t), ρv(t)            │                           │
│  └──────────────────────────────────────┘                           │
│                                                                     │
│  • Padding="causal" (sem acesso ao futuro)                         │
│  • Latência: ~10ms por predição (GPU)                              │
│  • Buffer circular para eficiência de memória                      │
│  • Modelos recomendados: WaveNet, Mamba_S4, Causal_TF             │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.3 Quantificação de Incerteza (UQ)

O sistema implementa 3 métodos de UQ:

```
┌─────────────────────────────────────────────────────────────────────┐
│              QUANTIFICAÇÃO DE INCERTEZA                              │
├──────────────────┬──────────────────────────────────────────────────┤
│  MC Dropout      │  • N forward passes com dropout ativado         │
│                  │  • Média = predição, Std = incerteza            │
│                  │  • Simples, funciona com qualquer modelo        │
│                  │  • N típico: 50–100 passes                      │
├──────────────────┼──────────────────────────────────────────────────┤
│  Ensemble        │  • K modelos treinados independentemente        │
│                  │  • Média do ensemble = predição                 │
│                  │  • Spread = incerteza epistêmica                │
│                  │  • K típico: 5–10 modelos                       │
├──────────────────┼──────────────────────────────────────────────────┤
│  INN             │  • Invertible Neural Network                    │
│                  │  • Sampling direto da distribuição posterior     │
│                  │  • 10× mais rápido que MC Dropout               │
│                  │  • Incerteza aleatória + epistêmica             │
│                  │  • Modelo precisa ser treinado como INN         │
└──────────────────┴──────────────────────────────────────────────────┘
```

### 9.4 Exportação

| Formato | Uso | Tamanho Típico |
|:--------|:----|:--------------:|
| SavedModel | TensorFlow Serving, deploy cloud | ~10 MB |
| TFLite | Edge devices, mobile | ~3 MB |
| ONNX | Interoperabilidade, outros frameworks | ~5 MB |

---

## 10. Avaliação e Visualização

### 10.1 Métricas de Avaliação

| Métrica | Fórmula | Uso |
|:--------|:--------|:----|
| **R²** | 1 - SS_res / SS_tot | Qualidade geral (alvo: > 0.95) |
| **RMSE** | √(mean(ε²)) | Erro típico em log10(Ω·m) |
| **MAE** | mean(\|ε\|) | Erro absoluto médio |
| **MBE** | mean(ε) | Viés (bias) sistemático |
| **MAPE** | mean(\|ε/y\|) × 100 | Erro percentual |

Todas as métricas são calculadas **por componente** (ρh e ρv separadamente)
e **por modelo geológico** para análise granular.

### 10.2 DOD Picasso — Depth of Detection

O **DOD Picasso** visualiza a capacidade de detecção do modelo em função
da profundidade e da resistividade, usando 6 métodos:

| Método | Descrição |
|:-------|:----------|
| Absolute Error | Mapa de \|ε\| no espaço (z, ρ) |
| Relative Error | Mapa de \|ε/ρ_true\| |
| R² Local | R² em janelas móveis |
| Confidence | Intervalos de confiança (UQ) |
| Gradient | Sensibilidade ∂ρ_pred/∂H |
| Binary | Detecção acima/abaixo de threshold |

### 10.3 Visualizações Disponíveis

```
┌─────────────────────────────────────────────────────────────────────┐
│                   MÓDULOS DE VISUALIZAÇÃO                           │
├──────────────────┬──────────────────────────────────────────────────┤
│  EDA             │  Distribuições, correlações, histogramas,        │
│                  │  boxplots, heatmaps de features                  │
├──────────────────┼──────────────────────────────────────────────────┤
│  Holdout         │  Predição vs verdade, scatter plots,            │
│                  │  perfis de resistividade, residuais              │
├──────────────────┼──────────────────────────────────────────────────┤
│  Training        │  Loss curves, LR schedule, gradient norms,      │
│                  │  métricas por epoch, noise schedule              │
├──────────────────┼──────────────────────────────────────────────────┤
│  Error Maps      │  Erro espacial (z × modelo), heatmaps,          │
│                  │  distribuição de erro por camada                 │
├──────────────────┼──────────────────────────────────────────────────┤
│  Geosteering     │  Dashboard em tempo real, trajeto do poço,      │
│                  │  bandas de incerteza, alertas de boundary       │
├──────────────────┼──────────────────────────────────────────────────┤
│  DOD Picasso     │  6 métodos de profundidade de detecção           │
├──────────────────┼──────────────────────────────────────────────────┤
│  Comparação      │  Modelo A vs B, tabelas de ranking,              │
│                  │  radar charts, análise estatística               │
└──────────────────┴──────────────────────────────────────────────────┘
```

### 10.4 Relatórios Automatizados

O sistema gera relatórios automatizados em dois formatos:

- **Markdown**: relatório legível com tabelas, gráficos referenciados e
  conclusões automáticas
- **JSON manifest**: metadados estruturados para integração programática
  (CI/CD, dashboards, comparações automáticas)

---

## 11. Roadmap de Desenvolvimento

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ROADMAP — FASES F1 a F6                          │
├──────┬──────────────────┬──────────┬────────────────────────────────┤
│ Fase │ Nome             │ Status   │ Descrição                      │
├──────┼──────────────────┼──────────┼────────────────────────────────┤
│  F1  │ Consolidação     │ ✅ Done  │ Migração notebook → pacote,    │
│      │                  │          │ 73 módulos, 744 testes,        │
│      │                  │          │ CI GitHub Actions              │
├──────┼──────────────────┼──────────┼────────────────────────────────┤
│  F2  │ Treinamento GPU  │ 🔄 WIP  │ Notebooks Colab para treino    │
│      │                  │          │ com T4/A100, validação GPU     │
│      │                  │          │ (824 testes passed)            │
├──────┼──────────────────┼──────────┼────────────────────────────────┤
│  F3  │ Otimização       │ 📋 Plan  │ XLA compilation, mixed         │
│      │                  │          │ precision (FP16), tf.data      │
│      │                  │          │ pipeline optimization          │
├──────┼──────────────────┼──────────┼────────────────────────────────┤
│  F4  │ Novas Arqs       │ ✅ Done  │ ModernTCN, INN, ResNeXt,       │
│      │                  │ (parcial)│ ResNeXt_LSTM implementados.    │
│      │                  │          │ Treino INN completo pendente   │
├──────┼──────────────────┼──────────┼────────────────────────────────┤
│  F5  │ Dados Multi-Dip  │ 📋 Plan  │ Re-simulação Fortran com       │
│      │                  │          │ múltiplos ângulos de dip.      │
│      │                  │          │ Tensor completo (9 comp, 18ch) │
├──────┼──────────────────┼──────────┼────────────────────────────────┤
│  F6  │ Deploy           │ 📋 Plan  │ API REST (FastAPI), Docker,    │
│      │                  │          │ TFLite para edge, monitoring   │
└──────┴──────────────────┴──────────┴────────────────────────────────┘
```

### Lacunas Conhecidas

| Lacuna | Descrição | Dependência |
|:-------|:----------|:-----------|
| Treino SurrogateNet | Re-simular Fortran multi-dip + treinar TCN/ModernTCN | F5 (dados) |
| Surrogate Modo C | Tensor completo (9 comp, 18ch) | F5 (dados multi-dip) |
| PINN Petrofísica | Constraints Archie + Klein | Cenário PINN 8 |
| Treino INN completo | Forward + latent loss (L_forward + λ·L_latent) | F2 (GPU) |

---

## 12. Documentação do Projeto

### 12.1 Catálogo de Documentos

```
┌─────────────────────────────────────────────────────────────────────┐
│                  DOCUMENTAÇÃO DO PROJETO                            │
├──────────────────────────────────┬──────────────────────────────────┤
│  DOCUMENTO                       │  CONTEÚDO                       │
├──────────────────────────────────┼──────────────────────────────────┤
│  docs/                           │                                  │
│  ├── ARCHITECTURE_v2.md          │  Arquitetura completa v2.0      │
│  ├── ROADMAP.md                  │  Roadmap de desenvolvimento     │
│  ├── MIGRATION_GUIDE.md          │  Guia notebook → pacote         │
│  ├── documentacao_apresentacao_  │  ESTE DOCUMENTO (visão geral)   │
│  │   geosteering_ai.md           │                                  │
│  ├── documentacao_noises.md      │  34 tipos de ruído detalhados   │
│  ├── documentacao_losses.md      │  26 losses com fórmulas         │
│  ├── documentacao_models.md      │  48 arquiteturas detalhadas     │
│  ├── documentacao_inferencia_    │  Pipeline de inferência offline │
│  │   offline.md                  │                                  │
│  ├── documentacao_geosteering.md │  Modelos causais + realtime     │
│  ├── documentacao_pinns.md       │  8 cenários PINN               │
│  │                               │                                  │
│  ├── physics/                    │                                  │
│  │   ├── errata_valores.md       │  Errata imutável de constantes  │
│  │   ├── perspectivas.md         │  5 perspectivas (P1–P5)        │
│  │   └── onboarding.md          │  Introdução à física EM/LWD    │
│  │                               │                                  │
│  └── reference/                  │                                  │
│      ├── arquiteturas_resumo.md  │  Tabela-resumo das 48 arqs     │
│      ├── losses_catalog.md       │  Catálogo completo de losses    │
│      ├── noise_catalog.md        │  Catálogo completo de ruídos    │
│      └── consensus_integration.md│  Integração pesquisa científica │
├──────────────────────────────────┼──────────────────────────────────┤
│  CLAUDE.md                       │  Instruções para Claude Code    │
│  configs/*.yaml                  │  Presets de configuração (7)    │
│  notebooks/*.ipynb               │  Notebooks de treino/validação  │
└──────────────────────────────────┴──────────────────────────────────┘
```

### 12.2 Hierarquia de Consulta

Para qualquer dúvida sobre o projeto, consulte na seguinte ordem:

| Prioridade | Documento | Quando |
|:----------:|:----------|:-------|
| 1ª | `docs/ARCHITECTURE_v2.md` | Decisões arquiteturais |
| 2ª | `CLAUDE.md` | Regras, proibições, patterns |
| 3ª | `geosteering_ai/config.py` | FLAGS, defaults, validações |
| 4ª | `configs/*.yaml` | Presets de configuração |
| 5ª | Docstrings no código | API e uso de funções |
| 6ª | `docs/physics/` | Contexto físico |
| 7ª | `docs/reference/` | Catálogos |

---

## 13. Configuração e Presets

### 13.1 Presets Disponíveis

O sistema oferece **presets pré-configurados** para cenários comuns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRESETS DE CONFIGURAÇÃO                           │
├──────────────────────┬──────────────────────────────────────────────┤
│  Preset              │  Descrição                                   │
├──────────────────────┼──────────────────────────────────────────────┤
│  baseline()          │  P1 sem ruído. Para debugging e              │
│                      │  validação de pipeline. ResNet-18,           │
│                      │  MSE loss, sem curriculum.                   │
├──────────────────────┼──────────────────────────────────────────────┤
│  robusto()           │  E-Robusto S21 (default). Ruído gaussiano   │
│                      │  com curriculum 3 fases, Huber loss,        │
│                      │  ResNet-18. Configuração recomendada.       │
├──────────────────────┼──────────────────────────────────────────────┤
│  nstage(n=3)         │  N-Stage training com N etapas de           │
│                      │  complexidade crescente. Clean → Robust     │
│                      │  → Fine-tune com PINN.                      │
├──────────────────────┼──────────────────────────────────────────────┤
│  geosinais_p4()      │  Perspectiva P4 com geosinais on-the-fly.  │
│                      │  Adiciona atributos GS1–GS5 aos inputs.    │
│                      │  Requer mais features de entrada.            │
├──────────────────────┼──────────────────────────────────────────────┤
│  realtime()          │  Geosteering causal. Modelos WaveNet ou     │
│                      │  Mamba_S4 com padding causal, sliding       │
│                      │  window, latência mínima.                   │
└──────────────────────┴──────────────────────────────────────────────┘
```

### 13.2 Uso via Python

```python
from geosteering_ai.config import PipelineConfig

# Preset direto:
config = PipelineConfig.robusto()

# Ou via YAML:
config = PipelineConfig.from_yaml("configs/robusto.yaml")

# Override de campos:
config = PipelineConfig.robusto()
config.model_type = "TCN_Advanced"
config.epochs = 200
```

### 13.3 Valores Críticos (Errata Imutável)

Estes valores são validados automaticamente no `__post_init__()` do
PipelineConfig e **nunca devem ser alterados**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ERRATA IMUTÁVEL                                   │
├────────────────────────┬──────────────────┬─────────────────────────┤
│  Parâmetro             │  Valor Correto   │  Valor PROIBIDO         │
├────────────────────────┼──────────────────┼─────────────────────────┤
│  FREQUENCY_HZ          │  20000.0         │  2.0                    │
│  SPACING_METERS        │  1.0             │  1000.0                 │
│  SEQUENCE_LENGTH       │  600             │  601                    │
│  TARGET_SCALING        │  "log10"         │  "log"                  │
│  INPUT_FEATURES        │  [1,4,5,20,21]   │  [0,3,4,7,8]           │
│  OUTPUT_TARGETS        │  [2,3]           │  [1,2]                  │
│  eps_tf (float32)      │  1e-12           │  1e-30                  │
└────────────────────────┴──────────────────┴─────────────────────────┘
```

---

## 14. Referências Bibliográficas

### Inversão EM e Geosteering

1. **Shahriari, M. et al.** "A deep learning approach to the inversion of
   borehole resistivity measurements." *Computational Geosciences*, 24(2),
   971–994, 2020. — Pioneiro em DL para inversão de resistividade em poço.

2. **Alyaev, S. et al.** "Modeling extra-deep electromagnetic logs using a
   deep neural network." *Geophysics*, 86(3), E269–E281, 2021. — DNN para
   logs EM de longo alcance.

3. **Larsen, E. et al.** "Deep learning for efficient geosteering."
   *Journal of Petroleum Science and Engineering*, 212, 110234, 2022.
   — Framework de geosteering com redes neurais.

4. **Rammay, M.H. et al.** "Multi-resolution deep learning for formation
   evaluation from LWD measurements." *Petrophysics*, 63(5), 2022.
   — Avaliação de formação multi-resolução.

### Deep Learning — Arquiteturas Base

5. **He, K. et al.** "Deep Residual Learning for Image Recognition."
   *CVPR*, 2016. — ResNet: skip connections para redes profundas.

6. **Bai, S. et al.** "An Empirical Evaluation of Generic Convolutional and
   Recurrent Networks for Sequence Modeling." *arXiv:1803.01271*, 2018.
   — TCN: convoluções temporais superam RNNs em muitas tarefas.

7. **Donghao, L. & Xue, W.** "ModernTCN: A Modern Pure Convolution Structure
   for General Time Series Analysis." *ICLR*, 2024. — ModernTCN: depth-wise
   separable convolutions para séries temporais.

8. **Liu, Z. et al.** "A ConvNet for the 2020s." *CVPR*, 2022.
   — ConvNeXt: CNN modernizada competitiva com Transformers.

9. **Xie, S. et al.** "Aggregated Residual Transformations for Deep Neural
   Networks." *CVPR*, 2017. — ResNeXt: grouped convolutions.

10. **Ronneberger, O. et al.** "U-Net: Convolutional Networks for Biomedical
    Image Segmentation." *MICCAI*, 2015. — U-Net: encoder-decoder com skip
    connections.

### Deep Learning — Transformers e Atenção

11. **Vaswani, A. et al.** "Attention Is All You Need." *NeurIPS*, 2017.
    — Transformer: mecanismo de self-attention.

12. **Lim, B. et al.** "Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting." *Int. J. Forecasting*, 37(4),
    2021. — TFT: transformer interpretável para séries temporais.

13. **Nie, Y. et al.** "A Time Series is Worth 64 Words: Long-term Forecasting
    with Transformers." *ICLR*, 2023. — PatchTST: patching para séries longas.

14. **Zhou, H. et al.** "Informer: Beyond Efficient Transformer for Long
    Sequence Time-Series Forecasting." *AAAI*, 2021. — ProbSparse attention.

### Deep Learning — Modelos Avançados

15. **Oreshkin, B.N. et al.** "N-BEATS: Neural basis expansion analysis for
    interpretable time series forecasting." *ICLR*, 2020. — Decomposição
    interpretável.

16. **Li, Z. et al.** "Fourier Neural Operator for Parametric Partial
    Differential Equations." *ICLR*, 2021. — FNO: operador neural no
    domínio de Fourier.

17. **Lu, L. et al.** "Learning nonlinear operators via DeepONet based on the
    universal approximation theorem of operators." *Nature Machine Intelligence*,
    3, 218–229, 2021. — DeepONet: aproximação de operadores.

18. **Gu, A. et al.** "Efficiently Modeling Long Sequences with Structured
    State Spaces." *ICLR*, 2022. — S4/Mamba: modelos de estado-espaço.

19. **Ardizzone, L. et al.** "Analyzing Inverse Problems with Invertible
    Neural Networks." *ICLR*, 2019. — INN: redes invertíveis para problemas
    inversos com quantificação de incerteza.

### Physics-Informed Neural Networks

20. **Raissi, M. et al.** "Physics-informed neural networks: A deep learning
    framework for solving forward and inverse problems involving nonlinear
    partial differential equations." *Journal of Computational Physics*, 378,
    686–707, 2019. — Framework PINN original.

### Geofísica e Petrofísica

21. **Anderson, B.** "Modeling and Inversion Methods for the Interpretation
    of Resistivity Logging Tool Response." *DUP Science*, 2001.
    — Referência para modelagem de ferramentas de resistividade.

22. **Ellis, D.V. & Singer, J.M.** "Well Logging for Earth Scientists."
    *Springer*, 2nd ed., 2007. — Referência geral de perfilagem de poço.

---

## Glossário

| Sigla | Significado |
|:------|:-----------|
| **EM** | Eletromagnético |
| **LWD** | Logging While Drilling — perfilagem durante a perfuração |
| **TIV** | Transverse Isotropy with Vertical axis of symmetry |
| **ρh** | Resistividade horizontal (Ω·m) |
| **ρv** | Resistividade vertical (Ω·m) |
| **FV** | Feature View — representação transformada dos dados |
| **GS** | Geosignal — atributo geofísico derivado |
| **DTB** | Distance To Boundary — distância até interface de camada |
| **PINN** | Physics-Informed Neural Network |
| **UQ** | Uncertainty Quantification |
| **INN** | Invertible Neural Network |
| **TCN** | Temporal Convolutional Network |
| **HPO** | Hyperparameter Optimization |
| **DOD** | Depth of Detection |
| **SNR** | Signal-to-Noise Ratio |
| **XLA** | Accelerated Linear Algebra (compilador TF) |
| **FP16** | Half-precision floating point |

---

> **Geosteering AI v2.0** — Inversão 1D de Resistividade EM via Deep Learning
>
> Autor: Daniel Leal | Framework: TensorFlow 2.x / Keras
>
> 73 módulos · 48 arquiteturas · 26 losses · 34 ruídos · 744 testes
>
> *"Traduzindo ondas eletromagnéticas em mapas de resistividade,
> uma predição de cada vez."*
