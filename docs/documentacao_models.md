# Documentação Técnica: Arquiteturas de Rede Neural — Geosteering AI v2.0

> **Projeto:** Inversão 1D de Resistividade via Deep Learning para Geosteering  
> **Módulo:** `geosteering_ai/models/` (13 módulos Python)  
> **Versão:** v2.0  
> **Autor:** Daniel Leal  
> **Framework:** TensorFlow 2.x / Keras (exclusivo)  
> **Data:** Abril 2026  

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Conceitos Fundamentais](#2-conceitos-fundamentais)
3. [ModelRegistry — Factory Pattern](#3-modelregistry--factory-pattern)
4. [Catálogo Completo — 48 Arquiteturas](#4-catálogo-completo--48-arquiteturas)
5. [Detalhamento por Família](#5-detalhamento-por-família)
   - 5.1 [CNN (8 arquiteturas)](#51-cnn-8-arquiteturas)
   - 5.2 [TCN (3 arquiteturas)](#52-tcn-3-arquiteturas)
   - 5.3 [RNN (2 arquiteturas)](#53-rnn-2-arquiteturas)
   - 5.4 [Híbrido (3 arquiteturas)](#54-híbrido-3-arquiteturas)
   - 5.5 [U-Net (14 arquiteturas)](#55-u-net-14-arquiteturas)
   - 5.6 [Transformer (6 arquiteturas)](#56-transformer-6-arquiteturas)
   - 5.7 [Decomposição (2 arquiteturas)](#57-decomposição-2-arquiteturas)
   - 5.8 [Avançado (5 arquiteturas)](#58-avançado-5-arquiteturas)
   - 5.9 [Geosteering (5 arquiteturas)](#59-geosteering-5-arquiteturas)
6. [Blocos Reutilizáveis (23+)](#6-blocos-reutilizáveis-23)
7. [Modo Dual: Offline vs Realtime](#7-modo-dual-offline-vs-realtime)
8. [SurrogateNet — Forward Model Neural](#8-surrogatenet--forward-model-neural)
9. [Tutorial Rápido — Escolha de Modelo por Cenário](#9-tutorial-rápido--escolha-de-modelo-por-cenário)
10. [Melhorias Futuras](#10-melhorias-futuras)
11. [Referências](#11-referências)

---

## 1. Visão Geral

O módulo `geosteering_ai/models/` implementa **48 arquiteturas de rede neural** organizadas
em **9 famílias**, todas projetadas para a tarefa de **inversão 1D de resistividade
eletromagnética** a partir de sinais de ferramentas LWD (Logging While Drilling).

### Paradigma Seq2Seq

Todas as arquiteturas seguem o paradigma **sequence-to-sequence (seq2seq)**, preservando
a dimensão temporal do sinal de entrada. Isto é fundamental: a tarefa NÃO é classificação,
mas sim predição ponto-a-ponto ao longo do perfil de poço.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PARADIGMA SEQ2SEQ                               │
│                                                                     │
│   Input:  (batch, seq_len, n_features)                             │
│              │         │         │                                   │
│              │         │         └── 5 baseline (z_obs + 4 EM)     │
│              │         │             até 15+ com GS/P2/P3/2ª ordem │
│              │         │                                            │
│              │         └── config.sequence_length                   │
│              │             (derivado do .out, default 600)          │
│              │                                                      │
│              └── tamanho do batch                                   │
│                                                                     │
│   Output: (batch, seq_len, output_channels)                        │
│              │         │         │                                   │
│              │         │         ├── 2: (ρ_h, ρ_v)                 │
│              │         │         ├── 4: (ρ_h, ρ_v, σ_h, σ_v)      │
│              │         │         └── 6: (ρ_h, ρ_v, σ_h, σ_v, DTB) │
│              │         │                                            │
│              │         └── preservado (mesma seq_len)               │
│              └── mesmo batch                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Organização em Tiers

O `ModelRegistry` classifica cada arquitetura em 3 níveis de maturidade:

| Tier | Significado | Critério |
|:----:|:------------|:---------|
| **1** | Validado, produção | Testado em GPU, métricas publicadas, estável |
| **2** | Comprovado, recomendado | Forward pass validado, aguarda treino completo |
| **3** | Experimental | Implementação funcional, em fase de avaliação |

### Dimensões de Entrada e Saída

| Configuração | `n_features` | Composição |
|:------------|:------------:|:-----------|
| P1 baseline | 5 | z_obs + Re(Hxx) + Im(Hxx) + Re(Hzz) + Im(Hzz) |
| P2 (GS incluído) | 7–10 | P1 + geosignals (atenuação, defasagem, ...) |
| P3 (2ª ordem) | 12–15+ | P2 + derivadas + features de segunda ordem |

| Configuração | `output_channels` | Composição |
|:------------|:-----------------:|:-----------|
| Padrão | 2 | ρ_h (resistividade horizontal), ρ_v (vertical) |
| Com incerteza | 4 | ρ_h, ρ_v + σ_h, σ_v (desvios-padrão) |
| Com DTB (P5) | 6 | ρ_h, ρ_v + σ_h, σ_v + DTB (dist. ao topo/base) |

---

## 2. Conceitos Fundamentais

Esta seção apresenta os conceitos teóricos essenciais para compreender as
arquiteturas implementadas no módulo.

### 2.1 Seq2Seq para Inversão 1D

Na inversão geofísica, cada ponto de medição ao longo do poço produz um
conjunto de sinais EM que deve ser mapeado para propriedades do meio
(resistividades). A rede recebe a sequência completa de medições e produz
a sequência correspondente de propriedades — preservando a dimensão temporal.

```
Medições LWD (seq_len pontos)        Propriedades do Meio (seq_len pontos)
┌───┬───┬───┬───┬───┬───┐           ┌───┬───┬───┬───┬───┬───┐
│ m₁│ m₂│ m₃│ m₄│...│ mₙ│  ──DL──▶ │ ρ₁│ ρ₂│ ρ₃│ ρ₄│...│ ρₙ│
└───┴───┴───┴───┴───┴───┘           └───┴───┴───┴───┴───┴───┘
  5 features por ponto                2–6 targets por ponto
```

### 2.2 Campo Receptivo (Receptive Field)

O campo receptivo determina quantos pontos de medição (em metros, dado
`SPACING_METERS = 1.0`) a rede "enxerga" para produzir cada saída.
Um campo receptivo maior captura dependências de longo alcance no perfil,
essencial para detectar camadas finas ou transições graduais.

```
Campo receptivo por arquitetura (exemplos):
┌────────────────────┬─────────────────────┬──────────────┐
│  Arquitetura       │  Mecanismo          │  Campo (m)   │
├────────────────────┼─────────────────────┼──────────────┤
│  CNN_1D            │  3 conv k=3         │  ~7          │
│  ResNet_18         │  4 stages k=3       │  ~35         │
│  TCN               │  dilação 2^(N-1)    │  ~255        │
│  ModernTCN         │  DWConv k=51        │  ~204        │
│  WaveNet           │  dilação empilhada  │  ~512        │
│  Transformer       │  atenção global     │  seq_len     │
│  UNet              │  pooling + skip     │  seq_len     │
└────────────────────┴─────────────────────┴──────────────┘
```

### 2.3 Causal vs Acausal

A distinção entre modos causal e acausal é crítica para geosteering em
tempo real.

```
ACAUSAL (padding='same'):                CAUSAL (padding='causal'):
usa passado + futuro                     usa APENAS passado

  ←──────→                                 ←──────
  contexto                                 contexto
      │                                        │
      ▼                                        ▼
  ┌───────────────────┐                  ┌───────────────────┐
  │ ...m₃ m₄ [m₅] m₆ m₇...│             │ ...m₃ m₄ [m₅] ░░░│
  └───────────────────┘                  └───────────────────┘
         ↓                                      ↓
     pred(m₅)                               pred(m₅)

  ✓ Offline (batch completo)             ✓ Realtime (streaming)
  ✗ Requer sequência completa            ✓ Sliding window
```

**Regra:** Em operações de geosteering realtime, APENAS arquiteturas com
suporte causal podem ser utilizadas. O `ModelRegistry` valida esta
restrição automaticamente.

### 2.4 Skip Connections

Conexões residuais (skip connections) permitem que o gradiente flua
diretamente pelas camadas, mitigando o problema de degradação em redes
profundas.

```
         ┌──────────────────┐
  x ────▶│  Conv → BN → ReLU │────▶ (+) ────▶ output
  │      │  Conv → BN        │       ▲
  │      └──────────────────┘       │
  └─────────────────────────────────┘
              skip connection
              (identidade ou projeção 1×1)
```

Utilizadas em: ResNet, UNet, N-BEATS, N-HiTS, InceptionTime, ResNeXt.

### 2.5 Mecanismos de Atenção

Dois tipos principais de atenção são empregados:

- **Atenção global (Transformer):** Cada ponto atende a todos os outros da
  sequência, capturando dependências de longo alcance. Complexidade O(L²).
- **Recalibração de canal (SE — Squeeze-and-Excitation):** Pondera a
  importância relativa de cada canal/filtro. Custo computacional mínimo.

```
SE Block:
  features (B, L, C) → GlobalAvgPool → Dense(C/r) → ReLU
                                        → Dense(C)  → Sigmoid → scale
```

### 2.6 Convoluções Dilatadas

Convoluções dilatadas expandem o campo receptivo exponencialmente sem
aumentar o número de parâmetros ou utilizar pooling.

```
Dilação = 1:    ■ ■ ■           campo = 3
Dilação = 2:    ■ . ■ . ■       campo = 5
Dilação = 4:    ■ . . . ■ . . . ■   campo = 9

Empilhamento (TCN): d=1,2,4,8,16,32,64,128 → campo = 255
```

Utilizadas em: TCN, TCN_Advanced, ModernTCN, WaveNet, Mamba_S4.

---

## 3. ModelRegistry — Factory Pattern

O `ModelRegistry` é a fábrica central de modelos do Geosteering AI v2.0.
Ele encapsula a criação de qualquer uma das 48 arquiteturas através de uma
interface unificada, eliminando condicionais espalhados pelo código.

### 3.1 Estrutura Interna

```python
# Registro interno: dict nome → metadados
_REGISTRY = {
    "ResNet_18": RegistryEntry(
        family="CNN",
        build_fn=_build_resnet18,      # lazy import
        tier=1,
        description="4 stages × 2 residual blocks, SE opcional",
        causal_compatible=True,
    ),
    # ... 47 outras entradas
}
```

### 3.2 API Pública

```python
from geosteering_ai.models import ModelRegistry

registry = ModelRegistry()

# ── Construir modelo a partir de PipelineConfig ──────────────────
model = registry.build(config)
# Internamente: registry._REGISTRY[config.model_type].build_fn(config)

# ── Listar modelos disponíveis ───────────────────────────────────
all_models = registry.list_available_models()            # 48 entradas
cnn_models = registry.list_available_models(family="CNN")  # 8 entradas

# ── Verificar compatibilidade causal ─────────────────────────────
assert registry.is_causal_compatible("TCN")      # True
assert not registry.is_causal_compatible("BiLSTM")  # False
```

### 3.3 Lazy Imports

Cada função `build_fn` utiliza importação tardia (lazy import) para evitar
carregar todas as 48 implementações na memória durante a inicialização:

```python
def _build_resnet18(config: PipelineConfig) -> tf.keras.Model:
    from geosteering_ai.models.cnn import build_resnet18
    return build_resnet18(
        input_shape=(None, config.n_features),
        output_channels=config.output_channels,
        use_causal=config.use_causal,
        **config.model_kwargs,
    )
```

### 3.4 Validação Automática

O `ModelRegistry.build()` valida automaticamente:

1. Se `config.model_type` existe no registro
2. Se `config.use_causal=True`, se o modelo é `causal_compatible`
3. Se `config.n_features` e `config.output_channels` são válidos
4. Emite warning para modelos Tier 3 (experimentais)

---

## 4. Catálogo Completo — 48 Arquiteturas

A tabela abaixo lista todas as 48 arquiteturas implementadas, organizadas
por família. O símbolo ★ indica o modelo default (`ResNet_18`).

| # | Nome | Família | Tier | Causal | ~Parâmetros | Melhor Uso |
|--:|:-----|:--------|:----:|:------:|:------------|:-----------|
| 1 | CNN_1D | CNN | 1 | ✓ | ~50K | Baseline simples, ablação |
| 2 | **ResNet_18** ★ | CNN | 1 | ✓ | ~400K | Default, pesquisa offline |
| 3 | ResNet_34 | CNN | 2 | ✓ | ~600K | Modelos geológicos complexos |
| 4 | ResNet_50 | CNN | 2 | ✓ | ~800K | Alta capacidade, datasets grandes |
| 5 | ConvNeXt | CNN | 2 | ✓ | ~300K | Modernização, B=1 estável |
| 6 | InceptionNet | CNN | 2 | ✓ | ~400K | Features multi-escala |
| 7 | InceptionTime | CNN | 2 | ✓ | ~200K | Multi-escala eficiente |
| 8 | ResNeXt | CNN | 3 | ✓ | ~450K | Multi-path, alta cardinalidade |
| 9 | TCN | TCN | 1 | ✓ | ~500K | Realtime causal nativo |
| 10 | TCN_Advanced | TCN | 2 | ✓ | ~1.2M | TCN + SE + atenção |
| 11 | ModernTCN | TCN | 2 | ✓ | ~250K | Eficiência, campo 204m |
| 12 | LSTM | RNN | 1 | ✓ | ~500K | Séries temporais, causal |
| 13 | BiLSTM | RNN | 1 | ✗ | ~1M | Offline, contexto bidirecional |
| 14 | CNN_LSTM | Híbrido | 2 | ✓ | ~600K | Local + temporal |
| 15 | CNN_BiLSTM_ED | Híbrido | 2 | ✗ | ~900K | Encoder-decoder offline |
| 16 | ResNeXt_LSTM | Híbrido | 3 | ✓ | ~700K | Multi-escala temporal |
| 17 | UNet_Base | U-Net | 2 | ✗ | ~500K | Segmentação de camadas |
| 18 | UNet_Attention_Base | U-Net | 2 | ✗ | ~550K | UNet + attention gates |
| 19 | UNet_ResNet18 | U-Net | 1 | ✗ | ~700K | Encoder ResNet, offline |
| 20 | UNet_Attention_ResNet18 | U-Net | 1 | ✗ | ~750K | Melhor UNet geral |
| 21 | UNet_ResNet34 | U-Net | 2 | ✗ | ~900K | Encoder profundo |
| 22 | UNet_Attention_ResNet34 | U-Net | 2 | ✗ | ~950K | ResNet34 + atenção |
| 23 | UNet_ResNet50 | U-Net | 2 | ✗ | ~1.1M | Encoder bottleneck |
| 24 | UNet_Attention_ResNet50 | U-Net | 2 | ✗ | ~1.15M | ResNet50 + atenção |
| 25 | UNet_ConvNeXt | U-Net | 2 | ✗ | ~600K | Encoder moderno |
| 26 | UNet_Attention_ConvNeXt | U-Net | 2 | ✗ | ~650K | ConvNeXt + atenção |
| 27 | UNet_Inception | U-Net | 2 | ✗ | ~700K | Multi-escala encoder |
| 28 | UNet_Attention_Inception | U-Net | 2 | ✗ | ~750K | Inception + atenção |
| 29 | UNet_EfficientNet | U-Net | 3 | ✗ | ~600K | MBConv encoder |
| 30 | UNet_Attention_EfficientNet | U-Net | 3 | ✗ | ~650K | EfficientNet + atenção |
| 31 | Transformer | Transformer | 2 | ✓ | ~700K | Contexto global |
| 32 | Simple_TFT | Transformer | 2 | ✓ | ~300K | Feature selection |
| 33 | TFT | Transformer | 2 | ✓ | ~600K | Features heterogêneas |
| 34 | PatchTST | Transformer | 2 | ✓ | ~400K | Longas sequências |
| 35 | Autoformer | Transformer | 3 | ✓ | ~400K | Decomposição sazonal |
| 36 | iTransformer | Transformer | 3 | ✓ | ~300K | Atenção sobre features |
| 37 | N_BEATS | Decomposição | 2 | ✓ | ~600K | Interpretável |
| 38 | N_HiTS | Decomposição | 2 | ✓ | ~700K | Multi-escala hierárquica |
| 39 | DNN | Avançado | 1 | ✓ | ~300K | Ablação (sem temporal) |
| 40 | FNO | Avançado | 3 | ✓ | ~200K | Dados periódicos |
| 41 | DeepONet | Avançado | 3 | ✗ | ~300K | Operator learning |
| 42 | Geophysical_Attention | Avançado | 3 | ✓ | ~400K | Physics-aware |
| 43 | INN | Avançado | 3 | ✓ | ~300K | UQ via sampling |
| 44 | WaveNet | Geosteering | 1 | ✓ | ~300K | Realtime referência |
| 45 | Causal_Transformer | Geosteering | 2 | ✓ | ~700K | Atenção causal |
| 46 | Informer | Geosteering | 2 | ✓ | ~300K | ProbSparse eficiente |
| 47 | Mamba_S4 | Geosteering | 3 | ✓ | ~200K | Complexidade linear |
| 48 | Encoder_Forecaster | Geosteering | 2 | ✓ | ~400K | Nowcasting |

---

## 5. Detalhamento por Família

### 5.1 CNN (8 arquiteturas)

A família CNN é a base do projeto, com a arquitetura default `ResNet_18`
sendo a mais utilizada e validada.

#### CNN_1D — Baseline Simples

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  Conv1D(32, k=3) → BN → ReLU → Dropout                 │
│    ↓                                                     │
│  Conv1D(64, k=3) → BN → ReLU → Dropout                 │
│    ↓                                                     │
│  Conv1D(128, k=3) → BN → ReLU → Dropout                │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~50K parâmetros | Tier 1 | Causal: ✓
```

Modelo mais simples do catálogo. Útil como baseline para ablação, verificando
se arquiteturas mais complexas realmente trazem ganho. O campo receptivo
é limitado (~7 pontos), insuficiente para capturar dependências de longo
alcance em modelos geológicos com camadas espessas.

#### ResNet_18 ★ — Default do Projeto

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  Stem: Conv1D(64, k=7) → BN → ReLU → Dropout           │
│    ↓                                                     │
│  Stage 1: 2× ResidualBlock(64)   + SE opcional          │
│  Stage 2: 2× ResidualBlock(128)  + SE opcional          │
│  Stage 3: 2× ResidualBlock(256)  + SE opcional          │
│  Stage 4: 2× ResidualBlock(512)  + SE opcional          │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~400K parâmetros | Tier 1 | Causal: ✓
```

Arquitetura principal do projeto. O stem com kernel 7 cobre ~7m de
profundidade no perfil de poço, ideal para detectar contrastes de camada
espessa. Os 4 estágios com 2 blocos residuais cada (total: 8 blocos)
permitem aprender representações hierárquicas. O bloco SE (Squeeze-and-
Excitation) opcional recalibra a importância de cada canal.

**Referência:** He et al. "Deep Residual Learning for Image Recognition"
(CVPR 2016) — skip connections estabilizam gradientes em redes profundas.

#### ResNet_34

Versão mais profunda com configuração de blocos (3, 4, 6, 3) por estágio,
totalizando 16 blocos residuais. Maior capacidade para datasets extensos
ou modelos geológicos com alta variabilidade.

~600K parâmetros | Tier 2 | Causal: ✓

#### ResNet_50

Utiliza blocos bottleneck (Conv1×1 → Conv3×3 → Conv1×1) com fator de
redução 4, reduzindo o custo computacional enquanto mantém alta capacidade.
Recomendado para datasets com milhares de modelos geológicos.

~800K parâmetros | Tier 2 | Causal: ✓

#### ConvNeXt — Design Moderno

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  Stem: Conv1D(96, k=4, stride=4) → LayerNorm            │
│    ↓                                                     │
│  Stage 1–4: N× ConvNeXtBlock                            │
│    │  DWConv(k=7) → LayerNorm → Conv1×1(4C) → GELU     │
│    │  → Conv1×1(C) → residual                           │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~300K parâmetros | Tier 2 | Causal: ✓
```

Transposição do ConvNeXt (Liu et al., 2022) para 1D. Substitui BatchNorm
por LayerNorm e ReLU por GELU, resultando em estabilidade superior com
batch size 1 — comum em cenários de geosteering realtime.

#### InceptionNet — Multi-Escala

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  InceptionModule (paralelo):                             │
│    ├── Conv1D(k=9)   → captura dependências curtas      │
│    ├── Conv1D(k=19)  → captura dependências médias      │
│    ├── Conv1D(k=39)  → captura dependências longas      │
│    └── MaxPool(k=3)  → preserva features dominantes     │
│    → Concatenação no eixo de canais                      │
│    ↓                                                     │
│  N× InceptionModule                                      │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~400K parâmetros | Tier 2 | Causal: ✓
```

Kernels de tamanhos variados capturam simultaneamente dependências de
diferentes escalas espaciais: camadas finas (k=9, ~9m), médias (k=19,
~19m) e espessas (k=39, ~39m).

#### InceptionTime — Inception Eficiente

Variante do InceptionNet com blocos residuais, reduzindo o número de
parâmetros em ~50% enquanto mantém a capacidade multi-escala. Recomendado
quando eficiência computacional é prioritária.

~200K parâmetros | Tier 2 | Causal: ✓

#### ResNeXt — Multi-Path com Alta Cardinalidade

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  Stem: Conv1D(64, k=7) → BN → ReLU                     │
│    ↓                                                     │
│  N× ResNeXtBlock:                                        │
│    │  Conv1×1(d×C) → GroupedConv(k=3, groups=C)         │
│    │  → Conv1×1(out) → BN → residual                   │
│    │  C=32 caminhos (cardinalidade), d=4 (bottleneck)   │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~450K parâmetros | Tier 3 | Causal: ✓
```

Implementa convoluções agrupadas (grouped convolutions) com cardinalidade
C=32 e dimensão de bottleneck d=4, seguindo Xie et al. (2017). Cada grupo
aprende uma sub-representação independente, aumentando a diversidade de
features sem explodir o número de parâmetros.

**Referência:** Xie et al. "Aggregated Residual Transformations for Deep
Neural Networks" (CVPR 2017).

---

### 5.2 TCN (3 arquiteturas)

As Temporal Convolutional Networks são a família preferida para
geosteering em tempo real, graças ao suporte causal nativo e ao campo
receptivo exponencialmente crescente via convoluções dilatadas.

#### TCN — Temporal Convolutional Network

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  TCN Stack:                                              │
│    │  Block d=1:   Conv(k=3, d=1)  → BN → ReLU → Drop  │
│    │  Block d=2:   Conv(k=3, d=2)  → BN → ReLU → Drop  │
│    │  Block d=4:   Conv(k=3, d=4)  → BN → ReLU → Drop  │
│    │  Block d=8:   Conv(k=3, d=8)  → BN → ReLU → Drop  │
│    │  ...                                                │
│    │  Block d=128: Conv(k=3, d=128) → BN → ReLU → Drop │
│    │  (cada bloco com skip connection residual)          │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~500K parâmetros | Tier 1 | Causal: ✓ (nativo)
Campo receptivo: ~255 pontos (~255m com SPACING=1.0)
```

Dilação dobrando a cada camada: 1, 2, 4, 8, 16, 32, 64, 128. O campo
receptivo cresce exponencialmente como RF = 2^N - 1 (N = número de blocos).
Com 8 blocos, RF = 255 pontos.

**Referência:** Bai et al. "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling" (2018).

#### TCN_Advanced — TCN com SE e Atenção

Extensão do TCN com múltiplos stacks, blocos SE (recalibração de canal) e
camada de self-attention opcional no topo. O multi-stack permite combinar
campos receptivos de diferentes escalas.

~1.2M parâmetros | Tier 2 | Causal: ✓

#### ModernTCN — TCN de Nova Geração

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  Stem: Conv1D → LayerNorm                                │
│    ↓                                                     │
│  4× ModernTCNBlock:                                      │
│    │  DWConv1D(k=51) → LayerNorm                        │
│    │  → ConvFFN: Conv1×1(4C) → GELU → Conv1×1(C)       │
│    │  → residual                                         │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~250K parâmetros | Tier 2 | Causal: ✓
Campo receptivo: ~204m (4 blocos × k=51)
```

Inspirado no ConvNeXt, utiliza Depthwise Convolutions com kernel grande
(k=51) + ConvFFN (Conv1×1 como projeção) + LayerNorm. Alcança campo
receptivo de ~204m com **50% menos parâmetros** que o TCN clássico.

**Referência:** Luo & Wang "ModernTCN: A Modern Pure Convolution Structure
for General Time Series Analysis" (ICLR 2024).

---

### 5.3 RNN (2 arquiteturas)

Redes recorrentes capturam dependências temporais via estados ocultos.
Apesar de mais lentas que CNNs/TCNs no treinamento (não paralelizáveis),
oferecem modelagem natural de sequências.

#### LSTM — Long Short-Term Memory

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  LSTM(256, return_sequences=True)                        │
│    ↓                                                     │
│  LSTM(256, return_sequences=True)                        │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~500K parâmetros | Tier 1 | Causal: ✓ (nativo, forward-only)
```

LSTM forward-only com `return_sequences=True` para manter a saída seq2seq.
Nativamente causal — cada célula processa apenas informação passada.

#### BiLSTM — Bidirectional LSTM

Versão bidirecional que processa a sequência em ambas as direções,
concatenando os estados ocultos. Oferece melhor desempenho offline às custas
da incompatibilidade causal.

~1M parâmetros | Tier 1 | Causal: ✗ (INCOMPATÍVEL)

---

### 5.4 Híbrido (3 arquiteturas)

Combinações de CNN + RNN que aproveitam extração local de features (CNN)
seguida de modelagem temporal (RNN).

#### CNN_LSTM

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  Conv1D(64, k=3) → BN → ReLU → Dropout                 │
│  Conv1D(128, k=3) → BN → ReLU → Dropout                │
│  Conv1D(256, k=3) → BN → ReLU → Dropout                │
│    ↓                                                     │
│  LSTM(256, return_sequences=True)                        │
│  LSTM(128, return_sequences=True)                        │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~600K parâmetros | Tier 2 | Causal: ✓
```

As camadas CNN extraem features locais (padrões EM de curto alcance),
enquanto as LSTMs modelam dependências temporais de longo alcance ao longo
do perfil de poço.

#### CNN_BiLSTM_ED — Encoder-Decoder

Arquitetura encoder-decoder: CNN codifica features, BiLSTM processa
bidirecionalmente, decoder reconstrói a sequência de saída. Apenas offline.

~900K parâmetros | Tier 2 | Causal: ✗

#### ResNeXt_LSTM

Combina 3 blocos ResNeXt (convoluções agrupadas, multi-path) com 2 camadas
LSTM. O ResNeXt captura features multi-escala enquanto o LSTM modela
transições temporais. Arquitetura experimental de alta capacidade.

~700K parâmetros | Tier 3 | Causal: ✓

---

### 5.5 U-Net (14 arquiteturas)

A família U-Net é a maior do catálogo, com 7 variantes de encoder × 2
(com/sem attention gates). Todas são **CAUSAL_INCOMPATIBLE** devido às
skip connections entre encoder e decoder.

#### Estrutura Geral U-Net

```
┌──────────────────────────────────────────────────────────────────┐
│                         U-Net 1D                                  │
│                                                                   │
│  Encoder                    Decoder                               │
│  ┌─────────┐               ┌─────────┐                           │
│  │ Level 1 │──skip conn──▶│ Level 1 │ → Output                  │
│  │ (32 ch) │               │ (32 ch) │                           │
│  └────┬────┘               └────▲────┘                           │
│       ↓ pool                    ↑ upsample                       │
│  ┌─────────┐               ┌─────────┐                           │
│  │ Level 2 │──skip conn──▶│ Level 2 │                           │
│  │ (64 ch) │               │ (64 ch) │                           │
│  └────┬────┘               └────▲────┘                           │
│       ↓ pool                    ↑ upsample                       │
│  ┌─────────┐               ┌─────────┐                           │
│  │ Level 3 │──skip conn──▶│ Level 3 │                           │
│  │(128 ch) │               │(128 ch) │                           │
│  └────┬────┘               └────▲────┘                           │
│       ↓ pool                    ↑ upsample                       │
│  ┌─────────┐               ┌─────────┐                           │
│  │ Level 4 │──skip conn──▶│ Level 4 │                           │
│  │(256 ch) │               │(256 ch) │                           │
│  └────┬────┘               └────▲────┘                           │
│       ↓                         ↑                                 │
│  ┌──────────────────────────────┐                                 │
│  │      Bottleneck (512 ch)     │                                 │
│  └──────────────────────────────┘                                 │
└──────────────────────────────────────────────────────────────────┘
Profundidade: 4 níveis | Base: 32 filtros
```

#### Variantes de Encoder

```
┌──────────────────────────┬────────────────────────────────────────┐
│  Encoder                 │  Características                       │
├──────────────────────────┼────────────────────────────────────────┤
│  Base                    │  Conv1D simples, leve                  │
│  ResNet18                │  Blocos residuais, 2 por nível         │
│  ResNet34                │  Blocos residuais, mais profundo       │
│  ResNet50                │  Bottleneck blocks, alta capacidade    │
│  ConvNeXt                │  DWConv + LayerNorm + GELU, moderno   │
│  Inception               │  Multi-escala paralelo                 │
│  EfficientNet            │  MBConv (mobile bottleneck), eficiente │
└──────────────────────────┴────────────────────────────────────────┘
```

#### Attention Gates

As variantes com attention gates (`UNet_Attention_*`) adicionam um
mecanismo que pondera as skip connections, permitindo que o decoder
selecione quais features do encoder são mais relevantes para cada posição:

```
Attention Gate:
  skip (encoder) ──▶ W_s ──┐
                            ├──▶ ReLU → W_ψ → Sigmoid → α
  gating (decoder) ──▶ W_g ─┘
                                                        │
  output = skip × α    (α ∈ [0,1] por posição)
```

**Todas as 14 variantes U-Net são CAUSAL_INCOMPATIBLE** — as skip
connections entre encoder e decoder requerem acesso a toda a sequência.
Uso exclusivamente offline.

---

### 5.6 Transformer (6 arquiteturas)

A família Transformer implementa mecanismos de atenção para capturar
dependências globais na sequência de medições.

#### Transformer — Vanilla

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  Learned Positional Encoding                             │
│    ↓                                                     │
│  N× TransformerEncoderBlock:                             │
│    │  MultiHeadAttention(heads=8)                        │
│    │  → Add & LayerNorm                                  │
│    │  → FFN: Dense(4d) → ReLU → Dense(d)               │
│    │  → Add & LayerNorm                                  │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~700K parâmetros | Tier 2 | Causal: ✓ (com máscara triangular)
```

Implementação padrão com codificação posicional aprendida (não sinusoidal).
A máscara triangular habilita o modo causal. Complexidade O(L²) no
comprimento da sequência.

**Referência:** Vaswani et al. "Attention Is All You Need" (NeurIPS 2017).

#### Simple_TFT — Temporal Fusion Transformer Simplificado

Versão simplificada do TFT com GRN (Gated Residual Network) como bloco
base + camada Transformer. Feature selection via GRN, sem VSN completo.

~300K parâmetros | Tier 2 | Causal: ✓

#### TFT — Temporal Fusion Transformer

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  VSN (Variable Selection Network)                        │
│    → seleciona features relevantes por contexto          │
│    ↓                                                     │
│  GRN (Gated Residual Network)                            │
│    → extrai features não-lineares com skip               │
│    ↓                                                     │
│  Transformer Layer (MHA + FFN)                           │
│    ↓                                                     │
│  Gated Output Layer                                      │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~600K parâmetros | Tier 2 | Causal: ✓
```

O TFT lida com features heterogêneas (estáticas, conhecidas no futuro,
observadas) — ideal quando se combinam z_obs (estática ao longo do well
path), sinais EM (observados) e geosignals (derivados).

**Referência:** Lim et al. "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (IJF, 2021).

#### PatchTST — Patch Time Series Transformer

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, seq_len, N_FEATURES)                      │
│    ↓                                                     │
│  Patch Embedding:                                        │
│    sequência dividida em patches de tamanho P             │
│    (seq_len/P tokens ao invés de seq_len)                │
│    ↓                                                     │
│  Positional Encoding                                     │
│    ↓                                                     │
│  N× TransformerEncoder                                   │
│    ↓                                                     │
│  Head: projeção linear para output                       │
└──────────────────────────────────────────────────────────┘
~400K parâmetros | Tier 2 | Causal: ✓
Complexidade: O(L log L) — eficiente para longas sequências
```

Tokenização por patches reduz o número de tokens de L para L/P,
viabilizando atenção em sequências longas (>1000 pontos).

**Referência:** Nie et al. "A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers" (ICLR 2023).

#### Autoformer — Decomposição Autocorrelativa

Decompõe automaticamente a série em componentes de tendência e sazonalidade,
utilizando autocorrelação (ao invés de dot-product attention) para capturar
periodicidades intrínsecas.

~400K parâmetros | Tier 3 | Causal: ✓

**Referência:** Wu et al. "Autoformer: Decomposition Transformers with
Auto-Correlation for Long-Term Series Forecasting" (NeurIPS 2021).

#### iTransformer — Atenção Invertida

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, seq_len, N_FEATURES)                      │
│    ↓                                                     │
│  Inversão: atenção sobre FEATURES (não sobre tempo)      │
│    → cada feature é um "token"                           │
│    → a dimensão temporal é a "embedding"                 │
│    ↓                                                     │
│  TransformerEncoder (atenção entre features)              │
│    ↓                                                     │
│  Projeção para output                                    │
└──────────────────────────────────────────────────────────┘
~300K parâmetros | Tier 3 | Causal: ✓
```

Inverte a dimensão de atenção: ao invés de atender sobre posições temporais,
atende sobre variáveis/features. Especialmente eficaz quando as correlações
entre features (e.g., Re(Hxx) vs Im(Hzz)) são mais informativas que as
correlações temporais locais.

**Referência:** Liu et al. "iTransformer: Inverted Transformers Are
Effective for Time Series Forecasting" (ICLR 2023).

---

### 5.7 Decomposição (2 arquiteturas)

Arquiteturas que decompõem o sinal em componentes interpretáveis.

#### N-BEATS — Neural Basis Expansion

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, seq_len, N_FEATURES)                      │
│    ↓                                                     │
│  N× Stack (empilhamento residual):                       │
│    │  M× Block:                                          │
│    │    FC layers → θ_b (backcast), θ_f (forecast)      │
│    │    basis expansion: g_b(θ_b), g_f(θ_f)             │
│    │    residual: x ← x - g_b(θ_b)                     │
│    │    saída acumulada: y ← y + g_f(θ_f)              │
│    ↓                                                     │
│  Output: soma das previsões de todos os blocos           │
└──────────────────────────────────────────────────────────┘
~600K parâmetros | Tier 2 | Causal: ✓
```

A expansão em base (basis expansion) permite interpretabilidade: cada bloco
remove uma componente do sinal (backcast) e adiciona à previsão (forecast).
As bases podem ser polinomiais (tendência) ou harmônicas (sazonalidade).

**Referência:** Oreshkin et al. "N-BEATS: Neural Basis Expansion Analysis
for Interpretable Time Series Forecasting" (ICLR 2020).

#### N-HiTS — N-BEATS Hierárquico

Extensão do N-BEATS com pooling multi-escala hierárquico em 3 níveis
(pool_sizes = 1, 4, 8). Cada nível opera em resolução diferente, capturando
padrões de diferentes escalas temporais com maior eficiência.

~700K parâmetros | Tier 2 | Causal: ✓

**Referência:** Challu et al. "N-HiTS: Neural Hierarchical Interpolation
for Time Series Forecasting" (AAAI 2023).

---

### 5.8 Avançado (5 arquiteturas)

Arquiteturas especializadas que exploram paradigmas menos convencionais.

#### DNN — MLP por Ponto

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, seq_len, N_FEATURES)                      │
│    ↓                                                     │
│  TimeDistributed(Dense(256)) → ReLU → Dropout            │
│  TimeDistributed(Dense(128)) → ReLU → Dropout            │
│  TimeDistributed(Dense(64))  → ReLU → Dropout            │
│    ↓                                                     │
│  TimeDistributed(Dense(output_channels, 'linear'))       │
└──────────────────────────────────────────────────────────┘
~300K parâmetros | Tier 1 | Causal: ✓
```

Processa cada ponto de medição independentemente via TimeDistributed —
sem qualquer modelagem temporal. Serve como baseline de ablação: se uma
CNN/TCN/Transformer não supera o DNN significativamente, a informação
temporal não está sendo explorada adequadamente.

#### FNO — Fourier Neural Operator

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, seq_len, N_FEATURES)                      │
│    ↓                                                     │
│  Lifting: Conv1×1 (N_FEATURES → d_model)                │
│    ↓                                                     │
│  N× FourierLayer:                                        │
│    │  FFT → multiplicação por modos → iFFT              │
│    │  + Conv1×1 (transformação local)                    │
│    │  → GELU → residual                                  │
│    ↓                                                     │
│  Projection: Conv1×1 (d_model → output_channels)        │
└──────────────────────────────────────────────────────────┘
~200K parâmetros | Tier 3 | Causal: ✓
```

Opera no domínio de frequência, multiplicando os modos de Fourier por
pesos aprendíveis. Particularmente eficaz para dados com periodicidades
(e.g., sequências com camadas alternantes resistivas/condutivas).

**Referência:** Li et al. "Fourier Neural Operator for Parametric Partial
Differential Equations" (ICLR 2021).

#### DeepONet — Operator Learning

```
┌──────────────────────────────────────────────────────────┐
│  Branch Net (sinal de entrada):                          │
│    Input (batch, seq_len, N_FEATURES)                    │
│    → FC layers → b ∈ R^p                                │
│                                                          │
│  Trunk Net (localizações):                               │
│    Input (batch, seq_len, 1)  [posições z]              │
│    → FC layers → t ∈ R^p                                │
│                                                          │
│  Output = Σᵢ bᵢ × tᵢ  (dot product)                   │
└──────────────────────────────────────────────────────────┘
~300K parâmetros | Tier 3 | Causal: ✗
```

Aprende o operador de inversão como produto de duas redes: branch (processa
o sinal EM) e trunk (processa as localizações espaciais). Paradigma de
operator learning — generaliza para diferentes configurações de medição.

**Referência:** Lu et al. "Learning Nonlinear Operators via DeepONet Based
on the Universal Approximation Theorem of Operators" (Nature Machine
Intelligence, 2021).

#### Geophysical_Attention — Atenção com Consciência Física

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, seq_len, N_FEATURES)                      │
│    ↓                                                     │
│  CNN Encoder: 3× Conv1D → BN → ReLU                    │
│    ↓                                                     │
│  Physics-Aware Attention:                                │
│    │  Máscara baseada no skin depth EM:                  │
│    │    δ = √(2ρ / ωμ)                                  │
│    │  Pontos além do skin depth recebem                  │
│    │  atenuação na atenção                               │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~400K parâmetros | Tier 3 | Causal: ✓
```

Incorpora conhecimento físico diretamente no mecanismo de atenção: a
profundidade de penetração (skin depth) limita o alcance da atenção,
refletindo a física real da propagação EM em formações rochosas.

#### INN — Invertible Neural Network

```
┌──────────────────────────────────────────────────────────────────┐
│  Input x ∈ R^d                                                   │
│    ↓                                                             │
│  8× Affine Coupling Layer (Real-NVP):                           │
│    │  Split: x → (x₁, x₂)                                      │
│    │  Forward:                                                   │
│    │    y₁ = x₁                                                 │
│    │    y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁)                         │
│    │  (s, t são redes densas)                                    │
│    │                                                             │
│    │  Inverse (analítico, sem iteração):                        │
│    │    x₂ = (y₂ - t(y₁)) ⊙ exp(-s(y₁))                      │
│    │    x₁ = y₁                                                 │
│    ↓                                                             │
│  Output y ∈ R^d  (+ latent z para UQ)                           │
│                                                                   │
│  ┌──────────────────────────────────────────┐                    │
│  │  Para UQ (Quantificação de Incerteza):   │                    │
│  │  1. Amostrar z ~ N(0, I)                 │                    │
│  │  2. Inversão: x = f⁻¹(y_obs, z)        │                    │
│  │  3. Repetir N vezes → distribuição       │                    │
│  │  → 10× mais rápido que MC Dropout        │                    │
│  └──────────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
~300K parâmetros | Tier 3 | Causal: ✓
```

Rede totalmente invertível via Affine Coupling Layers (Real-NVP). A
invertibilidade analítica permite amostrar da distribuição posterior sem
custo adicional de forward passes (como no MC Dropout), resultando em
quantificação de incerteza ~10× mais rápida.

**Referência:** Ardizzone et al. "Analyzing Inverse Problems with
Invertible Neural Networks" (ICLR 2019).

---

### 5.9 Geosteering (5 arquiteturas) — Causais Nativas

Família dedicada a operações de geosteering em tempo real. Todas as
arquiteturas são **nativamente causais** — projetadas desde a concepção
para operar com sliding window, sem acesso a medições futuras.

#### WaveNet — Convoluções Gated Dilatadas

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, None, N_FEATURES)                         │
│    ↓                                                     │
│  Causal Conv1D (entrada)                                 │
│    ↓                                                     │
│  N× WaveNetBlock:                                        │
│    │  Dilated Causal Conv1D(d=2^i)                      │
│    │    ↓                                                │
│    │  Split → tanh(a) × σ(b)  (gated activation)       │
│    │    ↓                                                │
│    │  Conv1×1 (residual) ─→ skip connection             │
│    │  Conv1×1 (skip)     ─→ acumulador de skip          │
│    ↓                                                     │
│  Soma dos skips → ReLU → Conv1×1 → ReLU → Conv1×1      │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~300K parâmetros | Tier 1 | Causal: ✓ (nativo)
```

A ativação gated (tanh × sigmoid) permite controle fino do fluxo de
informação, enquanto as convoluções causais dilatadas garantem campo
receptivo amplo sem violar a causalidade.

**Referência:** Oord et al. "WaveNet: A Generative Model for Raw Audio"
(2016).

#### Causal_Transformer

Transformer com máscara triangular superior no mecanismo de atenção,
garantindo que cada posição atenda apenas a posições anteriores ou iguais.
Combina a capacidade de modelagem global do Transformer com restrição
causal estrita.

~700K parâmetros | Tier 2 | Causal: ✓ (nativo)

#### Informer — ProbSparse Attention

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, seq_len, N_FEATURES)                      │
│    ↓                                                     │
│  Positional Encoding                                     │
│    ↓                                                     │
│  N× InformerLayer:                                       │
│    │  ProbSparse Self-Attention:                         │
│    │    seleciona top-u queries dominantes               │
│    │    → O(L log L) ao invés de O(L²)                  │
│    │  → FFN                                              │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~300K parâmetros | Tier 2 | Causal: ✓ (nativo)
```

A ProbSparse Attention seleciona apenas as queries mais informativas
(baseado na divergência KL com distribuição uniforme), reduzindo a
complexidade de O(L²) para O(L log L). Viável para sequências com
milhares de pontos.

**Referência:** Zhou et al. "Informer: Beyond Efficient Transformer for
Long Sequence Time-Series Forecasting" (AAAI 2021).

#### Mamba_S4 — State-Space Model

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, seq_len, N_FEATURES)                      │
│    ↓                                                     │
│  Stem: Conv1D → LayerNorm                                │
│    ↓                                                     │
│  N× S4-like Block:                                       │
│    │  DWConv1D stack (aproximação do kernel S4)          │
│    │  → LayerNorm → GELU → residual                     │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~200K parâmetros | Tier 3 | Causal: ✓ (nativo)
Complexidade: O(L) — linear no comprimento da sequência
```

Aproxima o modelo de espaço de estados (State-Space Model) via stacks de
DWConv, alcançando complexidade **linear O(L)** — a mais eficiente do
catálogo para sequências muito longas. Modelo mais leve (~200K params).

**Referências:** Gu et al. "Efficiently Modeling Long Sequences with
Structured State Spaces" (ICLR 2022); Gu & Dao "Mamba: Linear-Time
Sequence Modeling with Selective State Spaces" (2024).

#### Encoder_Forecaster — Padrão Nowcasting

```
┌──────────────────────────────────────────────────────────┐
│  Input (batch, seq_len, N_FEATURES)                      │
│    ↓                                                     │
│  Encoder: 2× LSTM (codifica sequência observada)        │
│    → hidden state h, cell state c                        │
│    ↓                                                     │
│  Decoder: CNN 1D (decodifica previsão)                   │
│    → usa h, c como condicionamento                       │
│    ↓                                                     │
│  Dense(output_channels, 'linear')                        │
└──────────────────────────────────────────────────────────┘
~400K parâmetros | Tier 2 | Causal: ✓ (nativo)
```

Padrão de nowcasting adaptado para geosteering: o encoder LSTM comprime
a sequência observada em um estado latente, e o decoder CNN gera a
predição. Inspirado em arquiteturas de previsão meteorológica de curto prazo.

---

## 6. Blocos Reutilizáveis (23+)

O módulo implementa 23+ blocos construtivos reutilizáveis que são
compartilhados entre as arquiteturas:

| # | Bloco | Descrição | Usado em |
|--:|:------|:----------|:---------|
| 1 | `residual_block_1d` | Conv → BN → Act → Conv → BN + skip | ResNet_18/34, UNets |
| 2 | `bottleneck_block_1d` | Conv1×1 → Conv3×3 → Conv1×1 + skip (redução 4×) | ResNet_50, UNet_ResNet50 |
| 3 | `conv_next_block` | DWConv → LN → Conv1×1(4C) → GELU → Conv1×1(C) | ConvNeXt, UNet_ConvNeXt |
| 4 | `se_block` | GAP → Dense(C/r) → ReLU → Dense(C) → Sigmoid | ResNet + SE, TCN_Advanced |
| 5 | `dilated_causal_block` | Conv1D causal com dilação crescente | TCN, WaveNet |
| 6 | `inception_module` | Conv k=9,19,39 + MaxPool paralelos | InceptionNet/Time, UNet_Inception |
| 7 | `mbconv_block` | DWConv → SE → Conv1×1 (Mobile Bottleneck) | UNet_EfficientNet |
| 8 | `gated_activation_block` | tanh(a) × sigmoid(b) | WaveNet |
| 9 | `tcn_residual_block` | Dilated Conv → BN → Act + skip 1×1 | TCN, TCN_Advanced |
| 10 | `self_attention_block` | MHA + Add & LN | Transformer, TCN_Advanced |
| 11 | `transformer_encoder_block` | MHA → Add&LN → FFN → Add&LN | Transformer, TFT, PatchTST |
| 12 | `autocorr_block` | Autocorrelação + decomposição | Autoformer |
| 13 | `patch_embedding_block` | Segmentação em patches + projeção | PatchTST |
| 14 | `grn_block` | Gated Residual Network (ELU + gate) | TFT, Simple_TFT |
| 15 | `vsn_block` | Variable Selection Network | TFT |
| 16 | `ita_block` | Inverted Token Attention | iTransformer |
| 17 | `series_decomp_block` | Média móvel → tendência + sazonalidade | Autoformer, N-HiTS |
| 18 | `output_projection` | Dense(output_channels, 'linear') | Todos os modelos |
| 19 | `attention_block` | Attention gate para UNet | UNet_Attention_* |
| 20 | `film_layer` | Feature-wise Linear Modulation (γ, β) | Condicional |
| 21 | `static_injection_stem` | Injeção de features estáticas no stem | TFT, CNN condicional |
| 22 | `_causal_depthwise_conv1d` | DWConv com padding causal manual | ModernTCN (modo causal) |
| 23 | `affine_coupling_layer` | Split → s,t → scale+translate (Real-NVP) | INN |

### Exemplo de Uso

```python
from geosteering_ai.models.blocks import residual_block_1d, se_block

# ── Bloco residual com SE ──────────────────────────────────────
x = residual_block_1d(
    x,
    filters=128,
    kernel_size=3,
    padding="same",       # ou "causal"
    use_se=True,
    se_ratio=16,
    dropout_rate=0.1,
)
```

---

## 7. Modo Dual: Offline vs Realtime

Uma das características centrais do Geosteering AI v2.0 é o suporte
dual-mode: o mesmo código pode operar tanto em modo offline (processamento
em batch de dados históricos) quanto em modo realtime (geosteering com
sliding window).

```
┌──────────────────────────────────────────────────────────────────┐
│                    MODO DUAL                                      │
│                                                                   │
│  "same"   →  Offline (acausal, batch completo, pós-perfuração)  │
│  "causal" →  Realtime (causal, sliding window, durante perfuração│
└──────────────────────────────────────────────────────────────────┘
```

### Compatibilidade por Arquitetura

| Arquitetura | Offline | Realtime | Observações |
|:------------|:-------:|:--------:|:------------|
| **Causal Nativo** | | | |
| TCN | ✓ | ✓ | Dilação causal nativa |
| LSTM | ✓ | ✓ | Forward-only nativo |
| WaveNet | ✓ | ✓ | Gated causal convolutions |
| Mamba_S4 | ✓ | ✓ | State-space, O(L) |
| Encoder_Forecaster | ✓ | ✓ | LSTM encoder nativo |
| ModernTCN | ✓ | ✓ | DWConv causal manual |
| **Causal Adaptável** | | | |
| ResNet_18/34/50 | ✓ | ✓ | padding='same' → 'causal' |
| ConvNeXt | ✓ | ✓ | DWConv causal adaptável |
| CNN_1D | ✓ | ✓ | Conv1D causal simples |
| InceptionNet/Time | ✓ | ✓ | Kernels multi-escala causais |
| ResNeXt | ✓ | ✓ | Grouped conv causal |
| CNN_LSTM | ✓ | ✓ | CNN causal + LSTM |
| ResNeXt_LSTM | ✓ | ✓ | ResNeXt causal + LSTM |
| Transformer | ✓ | ✓ | Máscara triangular |
| Simple_TFT / TFT | ✓ | ✓ | GRN + máscara causal |
| PatchTST | ✓ | ✓ | Patches + máscara |
| Autoformer | ✓ | ✓ | Decomposição causal |
| iTransformer | ✓ | ✓ | Atenção sobre features |
| Causal_Transformer | ✓ | ✓ | Máscara nativa |
| Informer | ✓ | ✓ | ProbSparse causal |
| N_BEATS / N_HiTS | ✓ | ✓ | Basis expansion causal |
| DNN | ✓ | ✓ | Per-point (trivialmente causal) |
| FNO | ✓ | ✓ | Fourier + truncamento |
| Geophysical_Attention | ✓ | ✓ | Máscara física causal |
| INN | ✓ | ✓ | Coupling layers causais |
| TCN_Advanced | ✓ | ✓ | Multi-stack causal |
| **Causal Incompatível** | | | |
| BiLSTM | ✓ | ✗ | Bidirecional requer futuro |
| UNet_* (14 variantes) | ✓ | ✗ | Skip connections encoder→decoder |
| CNN_BiLSTM_ED | ✓ | ✗ | BiLSTM no encoder |
| DeepONet | ✓ | ✗ | Branch/Trunk requer sequência completa |

### Seleção Automática no config

```python
# ── Offline (padrão) ─────────────────────────────────────
config = PipelineConfig(model_type="ResNet_18", use_causal=False)

# ── Realtime ─────────────────────────────────────────────
config = PipelineConfig(model_type="TCN", use_causal=True)

# ── Erro: modelo incompatível com causal ────────────────
config = PipelineConfig(model_type="BiLSTM", use_causal=True)
# → ValueError: "BiLSTM não é compatível com modo causal"
```

---

## 8. SurrogateNet — Forward Model Neural

O SurrogateNet é uma rede neural treinada para substituir o solver Fortran
de modelagem EM direta (forward model). Seu uso principal é no cálculo da
loss PINN (Cenário 2) e no Modo C do pipeline.

### SurrogateNet Clássico (TCN)

```
┌──────────────────────────────────────────────────────────┐
│  Input: modelo geológico (batch, seq_len, n_geo_params)  │
│    ↓                                                     │
│  6× TCN Block:                                           │
│    │  Dilated Causal Conv1D (dilação 1, 2, 4, 8, 16, 32)│
│    │  → BN → ReLU → Dropout → residual                  │
│    ↓                                                     │
│  Dense → sinais EM simulados                             │
│    ↓                                                     │
│  Output: (batch, seq_len, n_em_signals)                  │
└──────────────────────────────────────────────────────────┘
~127m campo receptivo (6 blocos, dilação até 32, k=3)
```

### SurrogateNet v2 (ModernTCN)

```
┌──────────────────────────────────────────────────────────┐
│  Input: modelo geológico (batch, seq_len, n_geo_params)  │
│    ↓                                                     │
│  Stem: Conv1D → LayerNorm                                │
│    ↓                                                     │
│  4× ModernTCN Block:                                     │
│    │  DWConv1D(k=51) → LayerNorm                        │
│    │  → ConvFFN: Conv1×1(4C) → GELU → Conv1×1(C)       │
│    │  → residual                                         │
│    ↓                                                     │
│  Dense → sinais EM simulados                             │
│    ↓                                                     │
│  Output: (batch, seq_len, n_em_signals)                  │
└──────────────────────────────────────────────────────────┘
~204m campo receptivo | 50% menos parâmetros que v1
```

### Comparação

| Atributo | SurrogateNet v1 (TCN) | SurrogateNet v2 (ModernTCN) |
|:---------|:---------------------:|:---------------------------:|
| Blocos | 6 TCN | 4 ModernTCN |
| Kernel | k=3, dilatado | k=51, DWConv |
| Campo receptivo | ~127m | ~204m |
| Parâmetros relativos | 100% | ~50% |
| LayerNorm | Não (BatchNorm) | Sim |
| Ativação | ReLU | GELU |

### Uso no Pipeline PINN

```python
# ── Cenário 2: PINN com SurrogateNet ────────────────────
config = PipelineConfig(
    model_type="ResNet_18",
    pinn_scenario=2,
    surrogate_type="ModernTCN",  # ou "TCN"
)

# O SurrogateNet substitui o Fortran na loss PINN:
# L_total = L_data + λ × L_physics(surrogate)
```

---

## 9. Tutorial Rápido — Escolha de Modelo por Cenário

Árvore de decisão para selecionar a arquitetura mais adequada:

```
                        ┌─── Qual é o cenário? ───┐
                        │                          │
                   ┌────┴────┐               ┌─────┴─────┐
                   │ OFFLINE │               │ REALTIME   │
                   │ (batch) │               │ (streaming)│
                   └────┬────┘               └─────┬─────┘
                        │                          │
              ┌─────────┼─────────┐          ┌─────┼──────────┐
              │         │         │          │     │          │
         ┌────┴───┐ ┌───┴────┐ ┌──┴──┐  ┌───┴──┐ ┌┴────┐ ┌──┴───┐
         │Pesquisa│ │Segmen- │ │ UQ  │  │Rápido│ │Long │ │ UQ   │
         │ geral  │ │tação   │ │     │  │      │ │seq  │ │      │
         └───┬────┘ └───┬────┘ └──┬──┘  └──┬───┘ └──┬──┘ └──┬───┘
             │          │         │         │        │        │
         ResNet_18  UNet_Att_   INN      TCN ou  PatchTST    INN
          (★ default) ResNet18         ModernTCN  ou Mamba
                                       ou WaveNet
```

### Recomendações Detalhadas

| Cenário | 1ª Opção | 2ª Opção | Justificativa |
|:--------|:---------|:---------|:--------------|
| Pesquisa offline geral | **ResNet_18** ★ | ConvNeXt | Validado, estável, boa relação custo/benefício |
| Segmentação de camadas | UNet_Attention_ResNet18 | UNet_Attention_ResNet34 | Skip connections preservam resolução |
| Geosteering realtime | TCN | ModernTCN | Causal nativo, campo receptivo amplo |
| Realtime leve | ModernTCN | WaveNet | 50% menos parâmetros, LayerNorm estável |
| Quantificação de incerteza | INN | probabilistic_nll | Sampling analítico, 10× mais rápido |
| Features multi-escala | InceptionTime | ResNeXt | Kernels/grupos paralelos |
| Sequências longas (>2000) | PatchTST | Mamba_S4 | O(L log L) / O(L) |
| Ablação (sem temporal) | DNN | CNN_1D | Baseline sem contexto temporal |
| Dados periódicos | FNO | Autoformer | Domínio de frequência / autocorrelação |
| Features heterogêneas | TFT | Simple_TFT | Variable Selection Network |

### Exemplo Completo

```python
from geosteering_ai.config import PipelineConfig
from geosteering_ai.models import ModelRegistry

# ── Cenário: geosteering realtime com ModernTCN ─────────
config = PipelineConfig(
    model_type="ModernTCN",
    use_causal=True,
    sequence_length=600,
    n_features=5,
    output_channels=2,
)

# ── Construir modelo via Registry ────────────────────────
registry = ModelRegistry()
model = registry.build(config)

# ── Verificar ────────────────────────────────────────────
model.summary()
# Total params: ~250,000
# Input:  (None, None, 5)
# Output: (None, None, 2)
```

---

## 10. Melhorias Futuras

O catálogo de arquiteturas está em expansão contínua. As seguintes
direções estão planejadas:

### G-Query Transformer (Jiang et al., 2025)

Transformer multimodal com queries geofísicas especializadas. Combina
sinais EM, dados de perfuração e informações geológicas prévias em um
framework unificado de atenção cruzada.

### Evidential Regression

Quantificação de incerteza em um único forward pass, sem necessidade de
múltiplas amostragens (como MC Dropout) ou redes invertíveis (como INN).
Modela diretamente os parâmetros de uma distribuição sobre a saída.

### Perceiver IO

Arquitetura com capacidade de processar entradas e saídas de dimensões
arbitrárias, sem restrição de alinhamento temporal. Potencial para
inversão multi-ferramenta (combinando dados de diferentes sensores LWD).

### Vision Transformer (ViT) para 1D

Adaptação do Vision Transformer para séries temporais 1D, tratando
segmentos do perfil como "patches" visuais. Combinação de patch
embedding com atenção global.

---

## 11. Referências

### Arquiteturas CNN

- **He, K. et al.** "Deep Residual Learning for Image Recognition."
  *CVPR*, 2016. — Fundamento dos ResNets: skip connections para redes
  profundas.

- **Xie, S. et al.** "Aggregated Residual Transformations for Deep Neural
  Networks." *CVPR*, 2017. — ResNeXt: convoluções agrupadas com
  cardinalidade como nova dimensão.

- **Liu, Z. et al.** "A ConvNet for the 2020s." *CVPR*, 2022. —
  ConvNeXt: modernização de CNNs com design inspirado em Transformers
  (DWConv + LayerNorm + GELU).

### Redes Temporais

- **Bai, S. et al.** "An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling." *arXiv:1803.01271*, 2018. —
  TCN: convoluções dilatadas causais superam RNNs em múltiplas tarefas.

- **Luo, D. & Wang, X.** "ModernTCN: A Modern Pure Convolution Structure
  for General Time Series Analysis." *ICLR*, 2024. — ModernTCN: DWConv
  com kernels grandes + ConvFFN para séries temporais.

### Transformers

- **Vaswani, A. et al.** "Attention Is All You Need." *NeurIPS*, 2017. —
  Fundamento da família Transformer: mecanismo de self-attention.

- **Lim, B. et al.** "Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting." *International Journal of
  Forecasting*, 2021. — TFT: variable selection + gated residual + atenção.

- **Nie, Y. et al.** "A Time Series is Worth 64 Words: Long-term
  Forecasting with Transformers." *ICLR*, 2023. — PatchTST: tokenização
  por patches para eficiência em longas sequências.

- **Wu, H. et al.** "Autoformer: Decomposition Transformers with
  Auto-Correlation for Long-Term Series Forecasting." *NeurIPS*, 2021. —
  Autoformer: decomposição tendência+sazonalidade com autocorrelação.

- **Liu, Y. et al.** "iTransformer: Inverted Transformers Are Effective
  for Time Series Forecasting." *ICLR*, 2023. — iTransformer: atenção
  invertida sobre a dimensão de features.

- **Zhou, H. et al.** "Informer: Beyond Efficient Transformer for Long
  Sequence Time-Series Forecasting." *AAAI*, 2021. — Informer: ProbSparse
  attention com complexidade O(L log L).

### Decomposição

- **Oreshkin, B. et al.** "N-BEATS: Neural Basis Expansion Analysis for
  Interpretable Time Series Forecasting." *ICLR*, 2020. — N-BEATS:
  empilhamento residual com expansão em base interpretável.

- **Challu, C. et al.** "N-HiTS: Neural Hierarchical Interpolation for
  Time Series Forecasting." *AAAI*, 2023. — N-HiTS: pooling hierárquico
  multi-escala para capturar padrões em diferentes resoluções.

### Métodos Avançados

- **Li, Z. et al.** "Fourier Neural Operator for Parametric Partial
  Differential Equations." *ICLR*, 2021. — FNO: aprendizado de operadores
  no domínio de Fourier.

- **Lu, L. et al.** "Learning Nonlinear Operators via DeepONet Based on
  the Universal Approximation Theorem of Operators." *Nature Machine
  Intelligence*, 2021. — DeepONet: branch-trunk para operator learning.

- **Ardizzone, L. et al.** "Analyzing Inverse Problems with Invertible
  Neural Networks." *ICLR*, 2019. — INN: redes invertíveis para problemas
  inversos com quantificação de incerteza.

### State-Space e Geosteering

- **Gu, A. et al.** "Efficiently Modeling Long Sequences with Structured
  State Spaces." *ICLR*, 2022. — S4: modelos de espaço de estados
  estruturados para sequências longas.

- **Gu, A. & Dao, T.** "Mamba: Linear-Time Sequence Modeling with
  Selective State Spaces." *arXiv:2312.00752*, 2024. — Mamba: seleção
  dinâmica de estados com complexidade linear.

- **Oord, A. van den et al.** "WaveNet: A Generative Model for Raw Audio."
  *arXiv:1609.03499*, 2016. — WaveNet: convoluções causais gated dilatadas,
  referência para geração autorregressiva.

---

> **Nota:** Este documento é parte da documentação técnica do Geosteering
> AI v2.0. Para informações sobre a arquitetura de software completa,
> consulte `docs/ARCHITECTURE_v2.md`. Para o guia de migração do código
> legado, consulte `docs/MIGRATION_GUIDE.md`.
