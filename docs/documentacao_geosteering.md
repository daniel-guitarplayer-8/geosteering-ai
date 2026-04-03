# Documentação Técnica: Geosteering em Tempo Real — Geosteering AI v2.0

> **Versão:** 2.0 &nbsp;|&nbsp; **Autor:** Daniel Leal &nbsp;|&nbsp; **Data:** Abril 2026
> **Framework:** TensorFlow 2.x / Keras &nbsp;|&nbsp; **Pacote:** `geosteering_ai/`

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Conceitos Fundamentais de Geosteering](#2-conceitos-fundamentais-de-geosteering)
3. [Arquiteturas Causais Nativas (5)](#3-arquiteturas-causais-nativas-5)
4. [RealtimeInference — Inferência em Tempo Real](#4-realtimeinference--inferência-em-tempo-real)
5. [Modo Dual: Offline vs Realtime](#5-modo-dual-offline-vs-realtime)
6. [Validação de Compatibilidade Causal](#6-validação-de-compatibilidade-causal)
7. [Constraintes de Tempo Real para LWD](#7-constraintes-de-tempo-real-para-lwd)
8. [Tutorial Rápido](#8-tutorial-rápido)
9. [Melhorias Futuras](#9-melhorias-futuras)
10. [Referências](#10-referências)

---

## 1. Visão Geral

### O que é Geosteering?

**Geosteering** é o processo de ajustar a trajetória de um poço **em tempo real**,
utilizando dados de LWD (Logging While Drilling), para manter a broca dentro da
zona de reservatório — maximizando o contato com a formação produtiva e evitando
zonas de água ou rocha selante.

### Pipeline Completo de Geosteering

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE GEOSTEERING EM TEMPO REAL                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌───────────┐    ┌─────────────┐    ┌──────────────┐   │
│  │ 1. LWD   │───▶│ 2. Tele-  │───▶│ 3. Aqui-    │───▶│ 4. Inversão  │   │
│  │  Tool    │    │  metria   │    │  sição      │    │  DL (★)      │   │
│  │ (fundo   │    │ (MWD      │    │ (dados EM   │    │ (este        │   │
│  │  do poço)│    │  pulso)   │    │  brutos)    │    │  projeto)    │   │
│  └──────────┘    └───────────┘    └─────────────┘    └──────┬───────┘   │
│                                                             │           │
│                                                             ▼           │
│                  ┌──────────────┐    ┌──────────────────────────────┐    │
│                  │ 6. Controle  │◀───│ 5. Interpretação             │    │
│                  │  de Trajetó- │    │ (perfil de resistividade →   │    │
│                  │  ria (MWD)   │    │  posição relativa à camada)  │    │
│                  └──────────────┘    └──────────────────────────────┘    │
│                                                                          │
└────────────────────────────────────────────────────────────────────────────┘
```

O **Geosteering AI v2.0** implementa o **passo 4** — inversão 1D de resistividade
via Deep Learning — com latência inferior a **31 ms** por predição, compatível
com o ciclo de decisão de geosteering em tempo real.

### Capacidades de Tempo Real

| Métrica                | Valor                                       |
|:-----------------------|:--------------------------------------------|
| **Latência por ponto** | < 31 ms (GPU) / < 80 ms (CPU)             |
| **Arquiteturas causais nativas** | 5 (WaveNet, Causal_Transformer, Informer, Mamba_S4, Encoder_Forecaster) |
| **Arquiteturas adaptáveis** | 25 (via `padding='causal'` ou máscara) |
| **Total realtime**     | 30 arquiteturas compatíveis                |
| **Classe principal**   | `RealtimeInference` (`inference/realtime.py`) |
| **Mecanismo**          | Sliding window buffer + `InferencePipeline` |
| **Tipos de ruído**     | 34 (robustez a condições reais de campo)   |
| **Quantificação de incerteza** | MC Dropout, Ensemble, INN         |

---

## 2. Conceitos Fundamentais de Geosteering

### 2.1 O Problema de Geosteering

Em perfuração direcional, o objetivo é manter a broca **dentro do reservatório**
— uma camada de rocha porosa e permeável que contém hidrocarbonetos. O desafio:

```
         Superfície
         ══════════════════════════════════════
                    ╲
                     ╲  Poço vertical
                      ╲
                       ╲
         ───────────────╲────────────────────── Topo do reservatório
                         ╲▸▸▸▸▸▸▸▸▸▸▸▸▸▸▸▸▸▸  Trecho horizontal
         ─────────────────────────────────────── Base do reservatório
                                                 (zona de água)

         ▸▸▸▸ = trajetória desejada (dentro do reservatório)
```

Se a broca sai do reservatório para cima, entra na rocha selante (sem produção).
Se sai para baixo, pode atingir a zona de água (produção indesejada). Decisões
de correção devem ser tomadas **em segundos**, não em horas.

### 2.2 Ferramentas LWD e Medições EM

As ferramentas **LWD (Logging While Drilling)** são instrumentos acoplados ao
BHA (Bottom Hole Assembly) que adquirem medições **enquanto perfuram**:

- **Transmissor EM:** emite sinal eletromagnético a frequência fixa (default: 20 kHz)
- **Receptores:** medem campo magnético complexo (componentes real e imaginária)
- **Espaçamento:** distância transmissor-receptor (default: 1.0 m)
- **Tensor EM:** 9 componentes (Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz)

```
┌─────────────────────────────────────────────────────────┐
│              FERRAMENTA LWD — ESQUEMA SIMPLIFICADO      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│    ┌────┐         L = 1.0 m          ┌────┐            │
│    │ Tx │◄───────────────────────────▶│ Rx │            │
│    │    │   (espaçamento padrão)      │    │            │
│    └────┘                             └────┘            │
│  Transmissor                        Receptor            │
│  (20 kHz)                     (mede H complexo)         │
│                                                         │
│  Componentes medidas:                                   │
│    • Hxx, Hyy (planar)  →  ACp = -1/(4πL³) ≈ -0.0796  │
│    • Hzz (axial)        →  ACx = +1/(2πL³) ≈ +0.1592  │
│                                                         │
│  Features de entrada (5):                               │
│    [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)]        │
│    Índices no formato 22-colunas: [1, 4, 5, 20, 21]    │
│                                                         │
│  Targets de saída (2):                                  │
│    [ρ₁, ρ₂] (resistividades das camadas)               │
│    Índices no formato 22-colunas: [2, 3]                │
│    Escala: log10 (NUNCA "log" natural)                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Por Que Tempo Real Importa

O ciclo de decisão de geosteering opera em **tempo quase-real**:

1. **A cada ~5 segundos**, um novo ponto de medição é adquirido
2. O operador de geosteering precisa atualizar o modelo geológico
3. Decisões de correção de trajetória devem ser comunicadas ao sondador
4. O tempo entre medição e decisão deve ser **mínimo**

Com inversão convencional (e.g., mínimos quadrados iterativos), cada ponto
pode levar **minutos** para inverter. Com Deep Learning, a inversão leva
**< 31 ms** — compatível com o ciclo de decisão.

### 2.4 Restrição de Causalidade

Em tempo real, o modelo **não pode ver o futuro**:

```
┌────────────────────────────────────────────────────────────────┐
│                 RESTRIÇÃO DE CAUSALIDADE                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Tempo ──────────────────────────────▶                         │
│                                                                │
│  ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ● ○ ○ ○ ○ ○ ○ ○ ○          │
│  ├─── passado (visível) ──────┤ ↑  ├── futuro (invisível) ──┤ │
│                                │ │                             │
│                           agora│ └─ ponto de predição          │
│                                                                │
│  ■ = dados já adquiridos (usáveis)                             │
│  ● = ponto atual (predição aqui)                               │
│  ○ = dados futuros (NÃO disponíveis em tempo real)             │
│                                                                │
│  Consequência: apenas arquiteturas CAUSAIS são válidas         │
│  para deploy em tempo real. BiLSTM e UNet são INCOMPATÍVEIS.   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.5 Janela Deslizante (Sliding Window)

O mecanismo de inferência em tempo real utiliza uma **janela deslizante**
de tamanho fixo `W = config.sequence_length` (default: 600):

```
┌────────────────────────────────────────────────────────────┐
│               SLIDING WINDOW (W = 600)                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  t=1    Buffer: [m₁]                          → None      │
│  t=2    Buffer: [m₁, m₂]                     → None      │
│  ...    (preenchendo)                          → None      │
│  t=600  Buffer: [m₁, m₂, ..., m₆₀₀]         → pred₆₀₀   │
│  t=601  Buffer: [m₂, m₃, ..., m₆₀₁]         → pred₆₀₁   │
│  t=602  Buffer: [m₃, m₄, ..., m₆₀₂]         → pred₆₀₂   │
│                                                            │
│  ← m₁ descartado (FIFO)                                   │
│                                                            │
│  Cada predição usa exatamente W medições mais recentes.    │
│  O buffer circular descarta a medição mais antiga quando   │
│  uma nova chega e o buffer já está cheio.                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 2.6 Ciclo de Decisão

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│  Nova    │───▶│  Buffer  │───▶│ Inversão │───▶│  Perfil  │───▶│ Decisão │
│ medição  │    │ update   │    │  DL      │    │  ρ(z)    │    │ de      │
│ (5 seg)  │    │ (< 1ms)  │    │ (< 31ms) │    │          │    │ trajet. │
└─────────┘    └──────────┘    └──────────┘    └──────────┘    └─────────┘
     │                                                              │
     │                    Ciclo total < 50 ms                       │
     └──────────────────────────────────────────────────────────────┘
```

---

## 3. Arquiteturas Causais Nativas (5)

O Geosteering AI v2.0 possui **5 arquiteturas projetadas nativamente para
causalidade** — ou seja, sua estrutura interna **garante** que apenas dados
passados e presentes são utilizados, sem necessidade de adaptação.

### Resumo Comparativo

```
┌──────────────────────┬────────────┬───────────────┬───────────┬──────────────────┐
│  Arquitetura         │  Parâms    │  Campo Recep. │  Complex. │  Mecanismo       │
│                      │  (~aprox)  │  (amostras)   │           │  Causal          │
├──────────────────────┼────────────┼───────────────┼───────────┼──────────────────┤
│  WaveNet             │  ~300K     │  256           │  O(L)     │  padding=causal  │
│  Causal_Transformer  │  ~700K     │  ilimitado*   │  O(L²)    │  máscara triang. │
│  Informer            │  ~300K     │  ilimitado*   │  O(L logL)│  ProbSparse mask │
│  Mamba_S4            │  ~200K     │  muito longo  │  O(L)     │  SSM causal      │
│  Encoder_Forecaster  │  ~400K     │  histórico    │  O(L)     │  LSTM + CNN caus.│
├──────────────────────┼────────────┼───────────────┼───────────┼──────────────────┤
│  * limitado pela memória disponível                                              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 WaveNet

**Convoluções dilatadas com gates multiplicativas** — originalmente proposta
para geração de áudio (van den Oord et al., 2016), adaptada para inversão
geofísica 1D.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WAVENET — ARQUITETURA                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input (batch, None, 5)                                             │
│    │                                                                │
│    ▼                                                                │
│  Causal Conv1D (residual_channels, kernel=1)   ← projeção inicial  │
│    │                                                                │
│    ▼                                                                │
│  ┌─── Bloco Residual × N_LAYERS ──────────────────────────────┐    │
│  │                                                             │    │
│  │   Dilated Causal Conv1D (dilation=2^i, padding='causal')   │    │
│  │     │                                                       │    │
│  │     ├──▶ tanh(·)  ──┐                                      │    │
│  │     │                ├──▶  × (gate multiplicativo)          │    │
│  │     └──▶ sigmoid(·) ─┘      │                               │    │
│  │                              ├──▶ Conv1D(1) ──▶ skip_sum   │    │
│  │                              └──▶ Conv1D(1) ──▶ + residual │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│    │                                                                │
│    ▼                                                                │
│  Soma de todas as skip connections                                  │
│    │                                                                │
│    ▼                                                                │
│  ReLU → Conv1D(1) → ReLU → Dense(2, 'linear')                     │
│    │                                                                │
│    ▼                                                                │
│  Output (batch, None, 2)  ← [ρ₁, ρ₂] em escala log10             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Mecanismo causal:** `padding='causal'` no Keras garante que a saída no tempo
`t` depende apenas de entradas em tempos `≤ t`. Nenhuma informação futura vaza.

**Campo receptivo com dilatações dobradas:**

```
  Camada    Dilatação    Campo Receptivo Cumulativo
  ─────     ─────────    ──────────────────────────
    0          1              3  (kernel_size=3)
    1          2              7
    2          4              15
    3          8              31
    4          16             63
    5          32             127
    6          64             255
    7          128            511  ← cobre quase todo o buffer W=600
```

Com 8 camadas e kernel_size=3, o campo receptivo atinge **511 amostras**,
cobrindo a maioria da janela deslizante de 600 pontos.

**Pontos fortes:**
- Causalidade nativa via `padding='causal'` — sem adaptação necessária
- Eficiência computacional O(L) — complexidade linear no comprimento
- Gates multiplicativas capturam interações não-lineares entre features EM
- Skip connections facilitam fluxo de gradientes em redes profundas
- ~300K parâmetros — modelo leve para deploy em edge

---

### 3.2 Causal_Transformer

**Transformer com máscara de atenção causal triangular** — adapta a arquitetura
de self-attention (Vaswani et al., 2017) para processamento sequencial estrito.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CAUSAL TRANSFORMER — ARQUITETURA                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input (batch, None, 5)                                             │
│    │                                                                │
│    ▼                                                                │
│  Dense(d_model) + Positional Encoding (aprendido)                  │
│    │                                                                │
│    ▼                                                                │
│  ┌─── Bloco Transformer × N_LAYERS ───────────────────────────┐    │
│  │                                                             │    │
│  │   ┌─────────────────────────────────────────────────────┐  │    │
│  │   │  Multi-Head Causal Self-Attention                   │  │    │
│  │   │                                                     │  │    │
│  │   │  Máscara triangular inferior:                       │  │    │
│  │   │  ┌─────────────┐                                    │  │    │
│  │   │  │ 1 0 0 0 0 0 │  Q·K^T é mascarado com -∞        │  │    │
│  │   │  │ 1 1 0 0 0 0 │  nas posições futuras,            │  │    │
│  │   │  │ 1 1 1 0 0 0 │  garantindo que a atenção          │  │    │
│  │   │  │ 1 1 1 1 0 0 │  no tempo t só veja ≤ t.          │  │    │
│  │   │  │ 1 1 1 1 1 0 │                                    │  │    │
│  │   │  │ 1 1 1 1 1 1 │  use_causal_mask=True             │  │    │
│  │   │  └─────────────┘                                    │  │    │
│  │   └─────────────────────────────────────────────────────┘  │    │
│  │     │                                                       │    │
│  │     ▼                                                       │    │
│  │   LayerNorm + Residual                                      │    │
│  │     │                                                       │    │
│  │     ▼                                                       │    │
│  │   Feed-Forward Network (Dense → ReLU → Dense)              │    │
│  │     │                                                       │    │
│  │     ▼                                                       │    │
│  │   LayerNorm + Residual                                      │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│    │                                                                │
│    ▼                                                                │
│  Dense(2, 'linear')                                                │
│    │                                                                │
│    ▼                                                                │
│  Output (batch, None, 2)                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Mecanismo causal:** A máscara triangular inferior aplicada ao produto Q·K^T
atribui peso -∞ a todas as posições futuras antes do softmax. Após softmax,
esses pesos se tornam zero — eliminando qualquer contribuição de dados futuros.

**Positional encoding aprendido:** Em vez de senos/cossenos fixos, utiliza
embeddings treináveis que aprendem a importância relativa de cada posição
na sequência de medições LWD.

**Pontos fortes:**
- Campo receptivo ilimitado — cada posição atende a **todo o passado**
- Paralelizável durante treino (diferente de RNNs)
- Capta dependências de longo alcance entre medições distantes
- ~700K parâmetros — modelo de capacidade intermediária

**Limitação:** Complexidade O(L²) na atenção — para sequências muito longas
(L > 10000), considerar Informer ou Mamba_S4.

---

### 3.3 Informer

**Atenção esparsa ProbSparse com complexidade O(L log L)** — proposta por
Zhou et al. (2021) para previsão de séries temporais longas.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      INFORMER — ARQUITETURA                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input (batch, None, 5)                                             │
│    │                                                                │
│    ▼                                                                │
│  Dense(d_model) + Positional Encoding                              │
│    │                                                                │
│    ▼                                                                │
│  ┌─── Encoder × N_LAYERS ─────────────────────────────────────┐    │
│  │                                                             │    │
│  │  ProbSparse Self-Attention (causal mask)                    │    │
│  │    │                                                        │    │
│  │    │  Em vez de computar atenção completa (L×L),           │    │
│  │    │  seleciona apenas os top-u queries mais "ativas"       │    │
│  │    │  baseado na divergência KL com distribuição uniforme.  │    │
│  │    │                                                        │    │
│  │    │  u = c · ln(L) queries ativas → O(L log L)            │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  LayerNorm + Residual                                       │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  Feed-Forward + Distilling Layer                            │    │
│  │    (Conv1D + MaxPool → reduz comprimento pela metade)       │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│    │                                                                │
│    ▼                                                                │
│  ┌─── Decoder × N_DEC_LAYERS ─────────────────────────────────┐    │
│  │                                                             │    │
│  │  Causal Self-Attention (máscara triangular)                 │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  Cross-Attention (decoder → encoder)                        │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  Feed-Forward                                               │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│    │                                                                │
│    ▼                                                                │
│  Dense(2, 'linear') → Output (batch, None, 2)                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Mecanismo causal:** Máscara causal no decoder, análoga ao Causal_Transformer.
A atenção esparsa ProbSparse preserva a propriedade causal ao mascarar posições
futuras antes da seleção de queries ativas.

**Pontos fortes:**
- Complexidade O(L log L) — viável para sequências muito longas
- Distilling layers reduzem progressivamente o comprimento da sequência
- ~300K parâmetros — eficiente em memória e computação
- Ideal quando `sequence_length` é grande (> 2000 pontos)

---

### 3.4 Mamba_S4

**Modelo de espaço de estados (SSM) com gate seletivo** — inspirado em
S4 (Gu et al., 2022) e Mamba (Gu & Dao, 2024), implementado via convoluções
dilatadas causais (aproximação compatível com TensorFlow/Keras).

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MAMBA_S4 — ARQUITETURA                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input (batch, None, 5)                                             │
│    │                                                                │
│    ▼                                                                │
│  Dense(d_model) — projeção de entrada                              │
│    │                                                                │
│    ▼                                                                │
│  ┌─── Bloco SSM × N_LAYERS ───────────────────────────────────┐    │
│  │                                                             │    │
│  │  ┌── Branch A: Convolução Causal ────────────────────────┐ │    │
│  │  │                                                        │ │    │
│  │  │  DWConv1D(d=1, causal) → SiLU                        │ │    │
│  │  │  DWConv1D(d=4, causal) → SiLU                        │ │    │
│  │  │  DWConv1D(d=16, causal) → SiLU                       │ │    │
│  │  │                                                        │ │    │
│  │  │  3 dilatações (1, 4, 16) → campo receptivo ~64       │ │    │
│  │  │  DWConv = Depthwise separable (eficiente)             │ │    │
│  │  │                                                        │ │    │
│  │  └────────────────────────────────────────────────────────┘ │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  ┌── Selective Gate ─────────────────────────────────────┐  │    │
│  │  │                                                        │  │    │
│  │  │  gate = sigmoid(Dense(x))                             │  │    │
│  │  │  output = conv_out × gate                             │  │    │
│  │  │                                                        │  │    │
│  │  │  O gate "seleciona" quais features temporais são      │  │    │
│  │  │  relevantes — análogo ao mecanismo seletivo do Mamba. │  │    │
│  │  │                                                        │  │    │
│  │  └────────────────────────────────────────────────────────┘  │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  Dense(d_model) + Residual + LayerNorm                      │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│    │                                                                │
│    ▼                                                                │
│  Dense(2, 'linear') → Output (batch, None, 2)                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Mecanismo causal:** Convoluções depthwise com `padding='causal'` em todas
as 3 camadas dilatadas. O gate seletivo opera point-wise (sem dependência
temporal adicional).

**Aproximação SSM via convoluções dilatadas:** Como TensorFlow/Keras não
possui implementação nativa de SSM recorrente (como o Mamba original em
PyTorch/CUDA), a arquitetura aproxima o comportamento de memória longa
do S4 usando convoluções dilatadas com dilatações `d = {1, 4, 16}`,
combinando campo receptivo amplo com complexidade linear O(L).

**Pontos fortes:**
- Complexidade **linear** O(L) — o mais eficiente para sequências longas
- Memória de longo alcance via dilatações empilhadas
- Gate seletivo filtra informação irrelevante adaptivamente
- ~200K parâmetros — o mais leve das 5 arquiteturas causais
- Ideal para deploy em hardware limitado (edge computing)

---

### 3.5 Encoder_Forecaster

**LSTM encoder + CNN decoder causal** — padrão de nowcasting adaptado
para inversão geofísica. O encoder LSTM captura o estado latente da
sequência, e o decoder CNN projeta a saída ponto a ponto.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ENCODER_FORECASTER — ARQUITETURA                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input (batch, None, 5)                                             │
│    │                                                                │
│    ▼                                                                │
│  ┌─── Encoder: LSTM ──────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  LSTM(units=128, return_sequences=True)                     │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  LSTM(units=128, return_sequences=True)                     │    │
│  │    │                                                        │    │
│  │    │  LSTM é nativamente causal:                            │    │
│  │    │  h_t = f(h_{t-1}, x_t) — apenas estado passado        │    │
│  │    │  e entrada atual. Nenhum lookahead.                    │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  Sequência de estados ocultos (batch, None, 128)            │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│    │                                                                │
│    ▼                                                                │
│  ┌─── Decoder: CNN Causal ────────────────────────────────────┐    │
│  │                                                             │    │
│  │  Conv1D(64, 3, padding='causal') → BN → ReLU              │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  Conv1D(32, 3, padding='causal') → BN → ReLU              │    │
│  │    │                                                        │    │
│  │    │  Padding causal no decoder garante que a projeção      │    │
│  │    │  no tempo t usa apenas estados ocultos ≤ t.            │    │
│  │    │                                                        │    │
│  │    ▼                                                        │    │
│  │  Dense(2, 'linear')                                         │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│    │                                                                │
│    ▼                                                                │
│  Output (batch, None, 2)                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Mecanismo causal dual:**
1. **Encoder LSTM:** nativamente causal — o estado oculto `h_t` depende
   apenas de `h_{t-1}` e `x_t` (equações de gates LSTM)
2. **Decoder CNN:** `padding='causal'` em todas as camadas Conv1D

**Pontos fortes:**
- Padrão de nowcasting bem estabelecido na literatura de previsão
- LSTM captura dependências temporais de forma natural
- Decoder CNN adiciona refinamento local sem perder causalidade
- ~400K parâmetros — balanceamento entre capacidade e eficiência
- Robusto para sequências de comprimento variável

---

## 4. RealtimeInference — Inferência em Tempo Real

A classe `RealtimeInference` (`geosteering_ai/inference/realtime.py`) é o
componente central para deploy de modelos em cenários de geosteering.

### 4.1 Estrutura da Classe

```python
class RealtimeInference:
    """Pipeline de inferência em tempo real para geosteering.

    Mantém um buffer circular (sliding window) de tamanho W
    e executa predições incrementais a cada nova medição.

    Attributes:
        pipeline: InferencePipeline configurado com modelo causal.
        buffer: collections.deque de tamanho máximo W (sequence_length).
        n_updates: Contador de medições processadas desde o início.

    Example:
        >>> config = PipelineConfig(model_type="WaveNet", use_causal=True)
        >>> rt = RealtimeInference(config, model_path="model.keras")
        >>> pred = rt.update(new_measurement)
    """

    def __init__(self, config: PipelineConfig, model_path: str):
        self.pipeline = InferencePipeline(config, model_path)
        self.buffer = deque(maxlen=config.sequence_length)
        self.n_updates = 0

    def update(self, measurement: np.ndarray) -> Optional[np.ndarray]:
        """Processa uma nova medição e retorna predição se buffer cheio.

        Args:
            measurement: Vetor de features (N_FEATURES,) = (5,).
                [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)]

        Returns:
            np.ndarray de shape (2,) com [ρ₁, ρ₂] em log10,
            ou None se o buffer ainda não está cheio.
        """
        self.buffer.append(measurement)
        self.n_updates += 1

        if len(self.buffer) < self.buffer.maxlen:
            return None  # Fase de preenchimento

        # Buffer cheio → empilhar e predizer
        window = np.stack(list(self.buffer))          # (W, 5)
        window = np.expand_dims(window, axis=0)       # (1, W, 5)
        prediction = self.pipeline.predict(window)    # (1, W, 2)
        return prediction[0, -1, :]                   # (2,) último ponto

    @property
    def is_ready(self) -> bool:
        """Retorna True se o buffer está cheio e pronto para predição."""
        return len(self.buffer) == self.buffer.maxlen

    @property
    def buffer_fill(self) -> float:
        """Fração de preenchimento do buffer (0.0 a 1.0)."""
        return len(self.buffer) / self.buffer.maxlen
```

### 4.2 Diagrama de Operação

```
┌──────────────────────────────────────────────────────────────────────┐
│               REALTIMEINFERENCE — FLUXO DE OPERAÇÃO                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Medição LWD                                                        │
│   (5 features)                                                       │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  update(measurement)                                         │    │
│  │                                                              │    │
│  │  1. buffer.append(measurement)      ← deque FIFO            │    │
│  │  2. n_updates += 1                                           │    │
│  │                                                              │    │
│  │  3. if len(buffer) < W:                                      │    │
│  │       return None  ← "ainda preenchendo"                     │    │
│  │                                                              │    │
│  │  4. window = stack(buffer)          ← (W, 5)                │    │
│  │  5. window = expand_dims(window)    ← (1, W, 5)             │    │
│  │  6. pred = pipeline.predict(window) ← (1, W, 2)             │    │
│  │  7. return pred[0, -1, :]           ← (2,) último timestep  │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐                                                     │
│  │  [ρ₁, ρ₂]   │  ← resistividades em log10                        │
│  │  ou None     │  ← se buffer incompleto                            │
│  └─────────────┘                                                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 Buffer Circular (deque)

O buffer é implementado como `collections.deque(maxlen=W)`:

```
┌────────────────────────────────────────────────────────────────┐
│                   BUFFER CIRCULAR — deque(maxlen=W)            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Estado 1 (preenchendo, t < W):                                │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬   ┬───┐                   │
│  │ m₁│ m₂│ m₃│ m₄│   │   │   │   │...│   │                   │
│  └───┴───┴───┴───┴───┴───┴───┴───┴   ┴───┘                   │
│  ├── preenchido ──┤├──── vazio ───────────┤                    │
│  is_ready = False       buffer_fill = 4/W                      │
│                                                                │
│  Estado 2 (cheio, t = W):                                      │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬   ┬────┐                  │
│  │ m₁│ m₂│ m₃│ m₄│ m₅│ m₆│ m₇│ m₈│...│m_W │                  │
│  └───┴───┴───┴───┴───┴───┴───┴───┴   ┴────┘                  │
│  is_ready = True        buffer_fill = 1.0                      │
│                                                                │
│  Estado 3 (deslizando, t = W+1):                               │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┬   ┬──────┐                │
│  │ m₂│ m₃│ m₄│ m₅│ m₆│ m₇│ m₈│ m₉│...│m_W+1 │                │
│  └───┴───┴───┴───┴───┴───┴───┴───┴   ┴──────┘                │
│  ↑ m₁ descartado automaticamente (FIFO)                        │
│  is_ready = True        buffer_fill = 1.0                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Propriedades da deque:**
- **O(1)** para append e pop (extremidades)
- **maxlen** garante tamanho fixo sem gerenciamento manual
- Descarte automático do elemento mais antigo quando cheio
- Thread-safe para operações atômicas (append/pop)

---

## 5. Modo Dual: Offline vs Realtime

O Geosteering AI v2.0 suporta dois modos de operação, controlados pela
flag `use_causal` no `PipelineConfig`:

```
┌──────────────────────────────────────────────────────────────────────┐
│                  MODO DUAL — OFFLINE vs REALTIME                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  config.use_causal = False          config.use_causal = True         │
│  ┌──────────────────────┐          ┌──────────────────────┐          │
│  │  MODO OFFLINE        │          │  MODO REALTIME       │          │
│  │                      │          │                      │          │
│  │  padding = "same"    │          │  padding = "causal"  │          │
│  │  Batch completo      │          │  Streaming ponto     │          │
│  │  Acausal (vê tudo)   │          │  Causal (só passado) │          │
│  │  Pós-processamento   │          │  Tempo real          │          │
│  │                      │          │                      │          │
│  │  Uso: interpretação  │          │  Uso: geosteering    │          │
│  │  de poços já         │          │  durante perfuração  │          │
│  │  perfurados          │          │                      │          │
│  └──────────────────────┘          └──────────────────────┘          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Tabela Comparativa

| Aspecto                    | Offline (`use_causal=False`)        | Realtime (`use_causal=True`)       |
|:---------------------------|:------------------------------------|:-----------------------------------|
| **Acesso aos dados**       | Batch completo (poço inteiro)       | Streaming (medição por medição)    |
| **Causalidade**            | Acausal — vê passado E futuro       | Causal — apenas passado e presente |
| **Latência**               | Sem restrição (segundos a minutos)  | < 31 ms por predição               |
| **Padding Conv1D**         | `"same"` (simétrico)                | `"causal"` (assimétrico)           |
| **Atenção Transformer**    | Bidirecional (sem máscara)          | Máscara triangular inferior        |
| **LSTM**                   | Bidirecional (BiLSTM) possível      | Forward-only (unidirecional)       |
| **Shape de entrada**       | `(batch, L, features)` — L fixo    | `(1, W, features)` — janela W     |
| **Shape de saída**         | `(batch, L, targets)` — L pontos   | `(2,)` — último ponto apenas      |
| **Uso típico**             | Interpretação pós-perfuração        | Geosteering durante perfuração     |
| **Arquiteturas válidas**   | Todas as 48                         | 30 (5 nativas + 25 adaptáveis)    |
| **BiLSTM / UNet**          | Válidos                             | INCOMPATÍVEIS (veem futuro)        |
| **Quantificação incerteza**| Ensemble (múltiplos modelos)        | MC Dropout (único modelo)          |

### Exemplo de Configuração

```python
# ── Modo offline (pós-perfuração) ──────────────────────────
config_offline = PipelineConfig(
    model_type="ResNet_18",
    use_causal=False,       # padding="same", bidirecional
    sequence_length=600,
)

# ── Modo realtime (geosteering) ────────────────────────────
config_realtime = PipelineConfig(
    model_type="WaveNet",
    use_causal=True,        # padding="causal", unidirecional
    sequence_length=600,
)
# OU usando preset:
config_realtime = PipelineConfig.realtime(model_type="WaveNet")
```

---

## 6. Validação de Compatibilidade Causal

Nem todas as arquiteturas são compatíveis com o modo causal. O sistema
valida automaticamente a compatibilidade antes do deploy.

### 6.1 Critérios de Compatibilidade

```
┌────────────────────────────────────────────────────────────────────┐
│            CRITÉRIOS DE COMPATIBILIDADE CAUSAL                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ✓ COMPATÍVEL:                                                     │
│    • Conv1D com padding='causal' (WaveNet, TCN, etc.)             │
│    • LSTM forward-only (Encoder_Forecaster)                        │
│    • Transformer com use_causal_mask=True                          │
│    • SSM com convoluções causais (Mamba_S4)                        │
│    • Qualquer operação point-wise (Dense, BN, Dropout)            │
│                                                                    │
│  ✗ INCOMPATÍVEL:                                                   │
│    • BiLSTM (vê sequência completa em ambas direções)              │
│    • UNet com skip connections simétricas (encoder vê decoder)     │
│    • Conv1D com padding='same' sem mascaramento                    │
│    • Attention sem máscara causal                                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 6.2 ModelRegistry.is_causal_compatible()

O `ModelRegistry` mantém metadados de cada arquitetura, incluindo
compatibilidade causal:

```python
# Verificação automática no registry
registry = ModelRegistry()

# Consultar compatibilidade
registry.is_causal_compatible("WaveNet")              # True (nativa)
registry.is_causal_compatible("Causal_Transformer")    # True (nativa)
registry.is_causal_compatible("ResNet_18")             # True (adaptável)
registry.is_causal_compatible("BiLSTM_ResNet")         # False
registry.is_causal_compatible("UNet_1D")               # False

# Listar todas as arquiteturas causais
causal_models = registry.list_causal_compatible()
# → ['WaveNet', 'Causal_Transformer', 'Informer', 'Mamba_S4',
#    'Encoder_Forecaster', 'ResNet_18', 'ResNet_34', ...]
```

### 6.3 Classificação das 48 Arquiteturas

```
┌──────────────────────────┬────────────────────┬──────────────────────────┐
│  Categoria               │  Qtd.              │  Exemplos                │
├──────────────────────────┼────────────────────┼──────────────────────────┤
│  Causal nativa           │   5                │  WaveNet, Mamba_S4,      │
│  (projetada para causal) │                    │  Informer, Causal_Trans, │
│                          │                    │  Encoder_Forecaster      │
├──────────────────────────┼────────────────────┼──────────────────────────┤
│  Adaptável para causal   │  25                │  ResNet_18/34/50,        │
│  (padding='causal' ou    │                    │  TCN_1/2/3, LSTM,        │
│  máscara adicionada)     │                    │  GRU, ResNeXt,           │
│                          │                    │  ModernTCN, etc.         │
├──────────────────────────┼────────────────────┼──────────────────────────┤
│  Incompatível com causal │  18                │  BiLSTM_ResNet,          │
│  (requer futuro)         │                    │  UNet_1D, BiGRU,         │
│                          │                    │  Bi_Transformer, etc.    │
├──────────────────────────┼────────────────────┼──────────────────────────┤
│  TOTAL                   │  48                │                          │
└──────────────────────────┴────────────────────┴──────────────────────────┘
```

### 6.4 Validação em Tempo de Build

Quando `use_causal=True`, o `ModelRegistry.build()` verifica automaticamente:

```python
def build(self, config: PipelineConfig) -> tf.keras.Model:
    """Constrói modelo a partir da configuração.

    Raises:
        ValueError: Se use_causal=True e a arquitetura é incompatível.
    """
    if config.use_causal and not self.is_causal_compatible(config.model_type):
        raise ValueError(
            f"Arquitetura '{config.model_type}' é INCOMPATÍVEL com modo causal. "
            f"Use uma das {len(self.list_causal_compatible())} arquiteturas compatíveis."
        )
    # ... build model ...
```

---

## 7. Constraintes de Tempo Real para LWD

O deploy de modelos de inversão em cenários reais de geosteering impõe
restrições rigorosas que vão além da simples acurácia de predição.

### 7.1 Requisitos Operacionais

```
┌──────────────────────────────────────────────────────────────────────────┐
│              CONSTRAINTES DE TEMPO REAL PARA LWD                         │
├──────────────────┬────────────────────┬──────────────────────────────────┤
│  Constrainte     │  Requisito         │  Como o sistema atende           │
├──────────────────┼────────────────────┼──────────────────────────────────┤
│  Latência        │  < 10 ms (ideal)   │  Modelos leves (200K-700K       │
│                  │  < 31 ms (máximo)  │  parâmetros), inferência         │
│                  │                    │  single-batch GPU/CPU            │
├──────────────────┼────────────────────┼──────────────────────────────────┤
│  Causalidade     │  Sem lookahead     │  5 arquiteturas nativas +        │
│                  │  (futuro invisível)│  25 adaptáveis com padding       │
│                  │                    │  causal e máscaras               │
├──────────────────┼────────────────────┼──────────────────────────────────┤
│  Sequencialidade │  Processamento     │  RealtimeInference com buffer    │
│                  │  medição a medição │  circular deque(maxlen=W)        │
├──────────────────┼────────────────────┼──────────────────────────────────┤
│  Robustez a      │  Ruído EM,         │  34 tipos de ruído no treino     │
│  ruído           │  interferência,    │  (on-the-fly curriculum),        │
│                  │  drift térmico     │  gaussiano + correlacionado      │
├──────────────────┼────────────────────┼──────────────────────────────────┤
│  Quantificação   │  Barra de erro     │  MC Dropout (single model),      │
│  de incerteza    │  nas predições     │  Ensemble, INN (posterior        │
│  (UQ)            │                    │  sampling 10× mais rápido)       │
├──────────────────┼────────────────────┼──────────────────────────────────┤
│  Disponibilidade │  24/7 durante      │  Export TF SavedModel/ONNX,      │
│                  │  perfuração        │  inferência CPU como fallback,   │
│                  │  (dias a semanas)  │  sem dependências pesadas        │
├──────────────────┼────────────────────┼──────────────────────────────────┤
│  Memória         │  < 100 MB          │  Modelos 200K-700K parâmetros,   │
│                  │  (edge device)     │  buffer W×5 floats ≈ 12 KB      │
├──────────────────┼────────────────────┼──────────────────────────────────┤
│  Reprodutibilidade│ Mesma entrada →   │  PipelineConfig YAML + seed +    │
│                  │  mesma saída       │  tag GitHub = determinístico     │
└──────────────────┴────────────────────┴──────────────────────────────────┘
```

### 7.2 Latência por Componente

```
┌─────────────────────────────────────────────────────────┐
│           BREAKDOWN DE LATÊNCIA (GPU — T4)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  buffer.append()          │  < 0.01 ms    │  █          │
│  np.stack(buffer)         │  ~ 0.1 ms     │  █          │
│  np.expand_dims()         │  < 0.01 ms    │  █          │
│  pipeline.predict()       │  ~ 25 ms      │  ████████   │
│  resultado[-1, :]         │  < 0.01 ms    │  █          │
│  ──────────────────────────────────────────────────     │
│  TOTAL                    │  ~ 25.1 ms    │             │
│                                                         │
│  Margem de segurança:     │  ~6 ms                      │
│  Budget máximo:           │  31 ms                      │
│                                                         │
│  Em CPU (fallback):       │  ~ 70-80 ms                 │
│  (aceitável para ciclo de decisão de 5 s)               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Tutorial Rápido

### 8.1 Exemplo 1 — Configurar Inferência em Tempo Real

```python
import numpy as np
from geosteering_ai.config import PipelineConfig
from geosteering_ai.inference.realtime import RealtimeInference

# ── Configuração usando preset realtime ────────────────────
config = PipelineConfig.realtime(model_type="WaveNet")
# Equivalente a:
# config = PipelineConfig(
#     model_type="WaveNet",
#     use_causal=True,
#     sequence_length=600,
#     frequency_hz=20000.0,
#     spacing_meters=1.0,
#     target_scaling="log10",
#     input_features=[1, 4, 5, 20, 21],
#     output_targets=[2, 3],
# )

# ── Inicializar pipeline de tempo real ─────────────────────
rt = RealtimeInference(config, model_path="models/wavenet_causal.keras")

print(f"Buffer size: {config.sequence_length}")   # 600
print(f"Ready: {rt.is_ready}")                     # False
print(f"Fill: {rt.buffer_fill:.1%}")               # 0.0%
```

### 8.2 Exemplo 2 — Processar Fluxo de Medições

```python
import numpy as np
from geosteering_ai.config import PipelineConfig
from geosteering_ai.inference.realtime import RealtimeInference
import logging

logger = logging.getLogger("geosteering_ai")

# ── Setup ──────────────────────────────────────────────────
config = PipelineConfig.realtime(model_type="WaveNet")
rt = RealtimeInference(config, model_path="models/wavenet_causal.keras")

# ── Simular fluxo de medições LWD ─────────────────────────
# Em produção, estas viriam da telemetria MWD em tempo real
n_measurements = 1000
measurements = np.random.randn(n_measurements, 5)  # Exemplo sintético

for i, measurement in enumerate(measurements):
    prediction = rt.update(measurement)

    if prediction is None:
        # Buffer ainda preenchendo
        if i % 100 == 0:
            logger.info(
                f"Preenchendo buffer: {rt.buffer_fill:.1%} "
                f"({i}/{config.sequence_length})"
            )
    else:
        # Predição disponível
        rho1_log10, rho2_log10 = prediction
        rho1 = 10 ** rho1_log10  # Resistividade em Ohm.m
        rho2 = 10 ** rho2_log10

        if i % 50 == 0:
            logger.info(
                f"t={i}: rho1={rho1:.2f} Ohm.m, "
                f"rho2={rho2:.2f} Ohm.m"
            )

logger.info(f"Total de predições: {rt.n_updates - config.sequence_length + 1}")
```

### 8.3 Exemplo 3 — Escolher Modelo Causal para Geosteering

```python
from geosteering_ai.config import PipelineConfig
from geosteering_ai.models.registry import ModelRegistry

registry = ModelRegistry()

# ── Listar todas as arquiteturas compatíveis com causal ───
causal_models = registry.list_causal_compatible()
print(f"Total de modelos causais: {len(causal_models)}")
# → 30

# ── Filtrar por critério ──────────────────────────────────
# Cenário 1: Latência mínima (edge device, CPU)
# → Mamba_S4 (~200K params, O(L))
config_edge = PipelineConfig.realtime(model_type="Mamba_S4")

# Cenário 2: Máxima acurácia (GPU disponível)
# → Causal_Transformer (~700K params, atenção completa)
config_gpu = PipelineConfig.realtime(model_type="Causal_Transformer")

# Cenário 3: Sequências muito longas (L > 2000)
# → Informer (~300K params, O(L log L))
config_long = PipelineConfig(
    model_type="Informer",
    use_causal=True,
    sequence_length=5000,  # Sequência longa
)

# Cenário 4: Balanceamento geral
# → WaveNet (~300K params, campo receptivo 511)
config_balanced = PipelineConfig.realtime(model_type="WaveNet")

# ── Verificar compatibilidade antes de build ──────────────
for model_type in ["WaveNet", "BiLSTM_ResNet", "Mamba_S4", "UNet_1D"]:
    compat = registry.is_causal_compatible(model_type)
    status = "COMPATIVEL" if compat else "INCOMPATIVEL"
    print(f"  {model_type:25s} → {status}")

# Output:
#   WaveNet                   → COMPATIVEL
#   BiLSTM_ResNet             → INCOMPATIVEL
#   Mamba_S4                  → COMPATIVEL
#   UNet_1D                   → INCOMPATIVEL
```

### 8.4 Exemplo 4 — Inferência com Quantificação de Incerteza

```python
from geosteering_ai.config import PipelineConfig
from geosteering_ai.inference.realtime import RealtimeInference
from geosteering_ai.inference.uncertainty import MCDropoutPredictor
import numpy as np

# ── MC Dropout para UQ em tempo real ──────────────────────
config = PipelineConfig.realtime(model_type="WaveNet")
rt = RealtimeInference(config, model_path="models/wavenet_causal.keras")

# MC Dropout: N forward passes com dropout ativo
mc_predictor = MCDropoutPredictor(
    model=rt.pipeline.model,
    n_samples=30,  # 30 amostras MC
)

# Após buffer cheio:
window = np.random.randn(1, 600, 5)  # Exemplo
predictions = mc_predictor.predict(window)  # (30, 1, 600, 2)

# Estatísticas de incerteza no último ponto
mean_pred = predictions[:, 0, -1, :].mean(axis=0)   # (2,) média
std_pred = predictions[:, 0, -1, :].std(axis=0)     # (2,) desvio

print(f"rho1: {10**mean_pred[0]:.2f} +/- {std_pred[0]:.3f} log10(Ohm.m)")
print(f"rho2: {10**mean_pred[1]:.2f} +/- {std_pred[1]:.3f} log10(Ohm.m)")
```

---

## 9. Melhorias Futuras

O pipeline de geosteering em tempo real está planejado para evoluir em
múltiplas direções:

### 9.1 Inversão 2.5D

Atualmente, o sistema realiza **inversão 1D** — assume um modelo geológico
de camadas horizontais sob o ponto de medição. A extensão para **2.5D**
consideraria a geometria real da camada (mergulho, curvatura):

```
┌────────────────────────────────────────────────────────────┐
│  1D (atual)              │  2.5D (futuro)                  │
│                          │                                  │
│  ─── camada 1 ───────    │  ─── camada 1 ──╲               │
│  ─── camada 2 ───────    │  ─── camada 2 ────╲             │
│  (horizontais)           │  (mergulho variável)             │
│                          │                                  │
│  Ref: SPE/SPWLA 2021     │  Requer: dados multi-dip        │
└────────────────────────────────────────────────────────────┘
```

### 9.2 Predição Look-Ahead (DTB)

Usando informações de **DTB (Distance to Boundary)** — já implementado
como feature P5 no pipeline — o modelo poderia predizer a posição da
fronteira de camada **antes** de alcançá-la:

- Antecipar saída do reservatório em 3-5 metros
- Dar mais tempo ao operador para correção de trajetória
- Combinar com DTB para alertas proativos

### 9.3 Fusão Multi-Ferramenta

Integrar medições de **múltiplas ferramentas LWD** simultaneamente:

- Resistividade (EM) + Densidade (gamma-gamma) + Neutrão
- Fusão de sensores aumenta robustez e reduz ambiguidade
- Requer arquitetura multi-input com branches dedicadas

### 9.4 Janela Adaptativa

Em vez de `sequence_length` fixo, ajustar o tamanho da janela
dinamicamente baseado na geologia:

- Camadas finas → janela menor (alta resolução)
- Camadas espessas → janela maior (mais contexto)
- Critério: taxa de variação do sinal EM

### 9.5 Integração com Interface Humano-Máquina (HMI)

Conectar o pipeline de inferência diretamente ao software de geosteering
do operador:

- Visualização em tempo real do perfil de resistividade invertido
- Barras de incerteza (MC Dropout / INN) no display
- Alertas automáticos quando incerteza excede threshold
- Integração com WITSML (padrão da indústria para dados de poço)

---

## 10. Referências

### Geosteering e Geofísica de Poço

1. **Constable, M. V. et al.** (2016). "Looking Ahead of the Bit with
   Electromagnetic Tools." *SPE Drilling & Completion*, 31(03), 165-177.
   — Fundamentação do uso de EM para geosteering com look-ahead.

2. **Morales, D. P. et al.** (2025). "Physics-Informed Neural Networks
   for Electromagnetic Inversion in Geosteering." *Geophysics*.
   — PINNs aplicadas a inversão EM, integração de equações de Maxwell
   como constraints no treinamento.

### Arquiteturas de Deep Learning

3. **van den Oord, A. et al.** (2016). "WaveNet: A Generative Model for
   Raw Audio." *arXiv:1609.03499*.
   — Convoluções dilatadas causais com gates multiplicativas. Base da
   arquitetura WaveNet adaptada para inversão geofísica.

4. **Vaswani, A. et al.** (2017). "Attention Is All You Need."
   *NeurIPS 2017*.
   — Arquitetura Transformer original. Base para Causal_Transformer
   com máscara triangular.

5. **Zhou, H. et al.** (2021). "Informer: Beyond Efficient Transformer
   for Long Sequence Time-Series Forecasting." *AAAI 2021*.
   — Atenção esparsa ProbSparse O(L log L) para séries temporais longas.
   Base da arquitetura Informer.

6. **Gu, A. et al.** (2022). "Efficiently Modeling Long Sequences with
   Structured State Spaces (S4)." *ICLR 2022*.
   — Modelos de espaço de estados para sequências longas com complexidade
   linear. Fundamentação teórica do Mamba_S4.

7. **Gu, A. & Dao, T.** (2024). "Mamba: Linear-Time Sequence Modeling
   with Selective State Spaces." *COLM 2024*.
   — Mecanismo de gate seletivo sobre SSM. Inspiração para a implementação
   do Mamba_S4 em TensorFlow/Keras via convoluções dilatadas.

### Deep Learning Aplicado a Geociências

8. **He, K. et al.** (2016). "Deep Residual Learning for Image
   Recognition." *CVPR 2016*.
   — Skip connections que estabilizam gradientes em redes profundas.
   Base das famílias ResNet e ResNeXt no Geosteering AI.

9. **Shi, X. et al.** (2015). "Convolutional LSTM Network: A Machine
   Learning Approach for Precipitation Nowcasting." *NeurIPS 2015*.
   — Padrão encoder-forecaster para predição sequencial. Inspiração
   para a arquitetura Encoder_Forecaster.

10. **Hochreiter, S. & Schmidhuber, J.** (1997). "Long Short-Term
    Memory." *Neural Computation*, 9(8), 1735-1780.
    — Arquitetura LSTM com gates de esquecimento. Fundamentação do
    encoder LSTM no Encoder_Forecaster.

---

> **Documento gerado para o Geosteering AI v2.0**
> **Pacote:** `geosteering_ai/` &nbsp;|&nbsp; **Framework:** TensorFlow 2.x / Keras
> **Referência arquitetural:** `docs/ARCHITECTURE_v2.md`
