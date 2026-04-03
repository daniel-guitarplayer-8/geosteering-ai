# Catálogo Detalhado de Arquiteturas de Rede Neural — Geosteering AI v2.0

## Pipeline de Inversão Geofísica com Deep Learning

**Autor:** Daniel Leal
**Versão:** v2.0 (atualizado v2.0.1 — Abril 2026)
**Framework:** TensorFlow 2.x / Keras (exclusivo)
**Total de Arquiteturas:** 48 (9 famílias: CNN 8, TCN 3, RNN 2, Híbrido 3, U-Net 14, Transformer 6, Decomposição 2, Avançado 5, Geosteering 5)
**Data:** Abril 2026

---

## 1. Introdução — O Problema de Inversão EM e o Papel das Arquiteturas

### 1.1 Contexto Físico

A inversão geofísica de dados eletromagnéticos (EM) provenientes de ferramentas LWD (*Logging While Drilling*) consiste em estimar propriedades petrofísicas do subsolo — especificamente a **resistividade horizontal (ρh)** e a **resistividade vertical (ρv)** — a partir de medições do tensor de campo magnético **H** registradas ao longo de um poço.

A ferramenta LWD opera a uma frequência de **20.000 Hz** (20 kHz) com espaçamento transmissor-receptor de **1,0 m**, transmitindo um campo EM que interage com as formações rochosas circundantes. A resposta EM é governada pelas equações de Maxwell em meios anisotrópicos (simetria TIV — Transversally Isotropic with a Vertical axis of symmetry), onde:

- **ρh** (resistividade horizontal) controla a resposta no plano de acamamento
- **ρv** (resistividade vertical) controla a resposta perpendicular ao acamamento
- A razão **λ = √(ρv/ρh)** define o **coeficiente de anisotropia**

O tensor magnético medido **H** possui 9 componentes (3×3), das quais o pipeline utiliza as componentes **Hxx** (coplanar) e **Hzz** (coaxial), cada uma com partes real e imaginária, totalizando **5 features de entrada**: `[zobs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)]`.

### 1.2 Formulação como Problema de Deep Learning

O problema de inversão 1D é formulado como um mapeamento **sequência-a-sequência (seq2seq)**:

```
Entrada:  (batch, N_MEDIDAS, N_FEATURES)   →  Medições EM ao longo do poço
Saída:    (batch, N_MEDIDAS, 2)             →  [ρh, ρv] em escala log10
```

Onde:
- **N_MEDIDAS = 600** (profundidades de observação para θ=0°)
- **N_FEATURES = 5** (baseline P1) a **17** (P2+P3+P4 completo)
- **OUTPUT_CHANNELS = 2** (ρh, ρv), opcionalmente 4 (+σ) ou 6 (+DTB, ρ_adj)

Cada arquitetura deve **preservar a dimensão temporal** N_MEDIDAS, pois cada profundidade de observação deve produzir sua própria estimativa de resistividade. Isto exclui modelos que reduzem a sequência a um vetor único (classificação).

### 1.3 Dual-Mode: Offline vs. Realtime (Geosteering)

O pipeline opera em dois modos:

| Aspecto | Offline (inversão padrão) | Realtime (geosteering) |
|:--------|:--------------------------|:-----------------------|
| **Dados** | Perfil completo do poço | Janela deslizante (sliding window) |
| **Rede** | Acausal (vê futuro + passado) | **Causal** (só vê passado) |
| **Saída** | `(batch, N_MEDIDAS, 2)` | `(1, W, 2-6)` com incerteza |
| **Latência** | Não importa | Crítica (< 1s por amostra) |
| **Padding** | `"same"` | `"causal"` |
| **Uso** | Pós-processamento de logs | **Tomada de decisão em tempo real** |

Para **geosteering causal**, a rede **não pode acessar dados futuros** — cada predição na profundidade `d` deve depender apenas das medições em profundidades `≤ d`. Esta restrição elimina algumas arquiteturas (U-Nets, BiLSTM) e exige adaptações em outras.

---

## 2. Classificação das 44 Arquiteturas

### 2.1 Organização por Família

| Família | Qtd | Arquiteturas | Célula |
|:--------|:---:|:-------------|:------:|
| **CNN** | 9 | ResNet-18/34/50, ConvNeXt, InceptionNet, InceptionTime, CNN_1D, TCN, TCN_Advanced | C28-C30 |
| **RNN** | 2 | LSTM, BiLSTM | C31 |
| **Híbrido** | 2 | CNN_LSTM, CNN_BiLSTM_ED | C32 |
| **U-Net** | 14 | UNet_Inversion, Attention_UNet, + 12 variantes com backbones | C33 |
| **Transformer** | 6 | Transformer, Simple_TFT, TFT, PatchTST, Autoformer, iTransformer | C34 |
| **Decomposição** | 2 | N-BEATS, N-HiTS | C35 |
| **Operador** | 2 | FNO, DeepONet | C36 |
| **Atenção** | 1 | Geophysical_Attention | C36 |
| **Dense** | 1 | DNN | C36 |
| **Geosteering** | 5 | WaveNet, Causal_Transformer, Informer, Mamba_S4, Encoder_Forecaster | C36A |

### 2.2 Compatibilidade Causal (Geosteering)

| Categoria | Qtd | Arquiteturas |
|:----------|:---:|:-------------|
| **Nativas causais** | 6 | WaveNet, Causal_Transformer, TCN, Mamba_S4, LSTM, Encoder_Forecaster |
| **Adaptáveis** (causal mask ou padding) | 21 | ResNet-18/34/50, ConvNeXt, InceptionNet/Time, CNN_1D, CNN_LSTM, TCN_Advanced, Transformer, Simple_TFT, TFT, PatchTST, Autoformer, iTransformer, N-BEATS, N-HiTS, DNN, FNO*, Geophysical_Attention |
| **Incompatíveis** (offline only) | 17 | BiLSTM, CNN_BiLSTM_ED, todas as 14 U-Nets, DeepONet |

*FNO: FFT global viola causalidade estritamente, mas é adaptável com restrição espectral.*

### 2.3 Tiers de Recomendação

| Tier | Nível | Critério | Arquiteturas Exemplo |
|:----:|:------|:---------|:---------------------|
| **1** | Alta | Validadas, estáveis, bom tradeoff | ResNet-18 ★, ResNet-34/50, ConvNeXt, InceptionTime, CNN_1D, TCN, Attention_UNet, UNet_ResNet*, UNet_Attention_* |
| **2** | Média | Funcionais, requerem tuning | TCN_Advanced, Transformer, TFT, PatchTST, Autoformer, iTransformer, N-BEATS, N-HiTS, InceptionNet |
| **3** | Baixa | Experimentais ou nicho | LSTM, FNO, DeepONet, Geophysical_Attention, DNN |
| **G** | Geosteering | Especializadas em realtime causal | WaveNet, Causal_Transformer, Informer, Mamba_S4, Encoder_Forecaster |

---

## 3. Família CNN — Redes Convolucionais (9 arquiteturas)

As CNNs são a base do pipeline de inversão EM. Convoluções 1D operam ao longo da dimensão de profundidade, extraindo padrões locais nas medições EM que correlacionam com variações de resistividade. O **campo receptivo** (*receptive field*) da rede determina quantas profundidades vizinhas influenciam cada predição — um conceito análogo à **profundidade de investigação** da ferramenta LWD.

### 3.1 ResNet-18 ★ (DEFAULT)

**Referência:** He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
**Célula:** C28 | **Tier:** 1 (Alta) | **Causal:** Adaptável
**Parâmetros:** blocks=[2,2,2,2], filters=[64,128,256,512], kernel_size=3

**Descrição:**
ResNet-18 é a arquitetura **padrão (default)** do pipeline. Utiliza blocos residuais com conexões de atalho (*skip connections*) que permitem treinar redes profundas sem degradação de gradiente. Cada bloco consiste em duas convoluções 1D com BatchNormalization e ativação ReLU, mais uma conexão residual que soma o input ao output do bloco.

**Aplicação à Inversão EM:**
- O campo receptivo cresce progressivamente com a profundidade da rede (18 camadas), capturando correlações entre medições EM em diferentes profundidades
- As conexões residuais preservam informação de alta frequência das medições EM brutas, essencial para detectar limites finos de camadas geológicas (resolução vertical)
- A progressão de filtros [64→512] permite extrair features de complexidade crescente: das variações brutas de Re/Im(Hxx/Hzz) até padrões abstratos de anisotropia

**Vantagens:**
- Treinamento estável e rápida convergência (gradientes bem comportados)
- Bom tradeoff entre capacidade e velocidade de inferência
- Amplamente validado em problemas geofísicos similares (Wang 2018, Noh & Verdin 2022)
- Adaptável a modo causal com `padding="causal"`

**Desvantagens:**
- Campo receptivo limitado pelo kernel_size=3 e depth=18 (~36 profundidades)
- Não captura dependências de longo alcance tão bem quanto Transformers
- Pode ser excessivo para modelos geológicos simples (1-3 camadas)

---

### 3.2 ResNet-34

**Referência:** He et al. (2016)
**Célula:** C28 | **Tier:** 1 | **Causal:** Adaptável
**Parâmetros:** blocks=[3,4,6,3], filters=[64,128,256,512], kernel_size=3

**Descrição:**
Variante mais profunda do ResNet com 34 camadas. A configuração [3,4,6,3] concentra mais blocos nos estágios intermediários (128 e 256 filtros), ampliando o campo receptivo.

**Aplicação à Inversão EM:**
- Campo receptivo maior que o ResNet-18 (~68 profundidades), capturando correlações de mais longo alcance
- Os 6 blocos no estágio de 256 filtros permitem modelar relações complexas de anisotropia em zonas com múltiplas camadas finas
- Indicado quando o modelo geológico tem muitas camadas ou variações graduais de resistividade

**Vantagens:**
- Maior capacidade representacional que ResNet-18
- Estabilidade de treinamento preservada pelos blocos residuais
- Melhor desempenho em modelos geológicos complexos (>5 camadas)

**Desvantagens:**
- ~2× mais parâmetros que ResNet-18, aumentando tempo de treinamento
- Risco de overfitting em datasets pequenos (<1000 modelos geológicos)
- Ganho marginal sobre ResNet-18 em cenários simples

---

### 3.3 ResNet-50

**Referência:** He et al. (2016) — versão bottleneck
**Célula:** C28 | **Tier:** 1 | **Causal:** Adaptável
**Parâmetros:** blocks=[3,4,6,3], filters=[64,128,256,512], expansion=4, kernel_size=3

**Descrição:**
ResNet-50 usa **blocos bottleneck** (1×1 → 3×1 → 1×1) com fator de expansão 4. A convolução 1×1 inicial reduz a dimensionalidade, a 3×1 processa a informação espacial, e a 1×1 final expande de volta. Isto permite redes mais profundas com menos parâmetros por bloco.

**Aplicação à Inversão EM:**
- A redução de dimensionalidade no bottleneck atua como um gargalo de informação que força a rede a aprender representações compactas das medições EM
- As dimensões pós-expansão [256, 512, 1024, 2048] oferecem enorme capacidade para capturar a relação não-linear entre campos EM e resistividade em meios anisotrópicos
- Projeção final (projection shortcut) de 512→2048 dimensões permite processar o espaço latente completo

**Vantagens:**
- Eficiência computacional superior ao ResNet-34 para profundidade equivalente
- Bottleneck força representações compactas (efeito regularizador)
- Excelente em datasets grandes com alta variabilidade de modelos geológicos

**Desvantagens:**
- Complexidade de implementação (bottleneck + projection shortcuts)
- Pode ser over-parameterizado para inversão 1D simples
- Tempo de treinamento significativamente maior

---

### 3.4 ConvNeXt

**Referência:** Liu et al. "A ConvNet for the 2020s" (CVPR 2022)
**Célula:** C28 | **Tier:** 1 | **Causal:** Adaptável
**Parâmetros:** depths=[3,3,9,3], dims=[96,192,384,768], kernel_size=7

**Descrição:**
ConvNeXt moderniza a arquitetura CNN incorporando insights dos Vision Transformers: **Depthwise Separable Convolutions**, **Layer Normalization** (em vez de BatchNorm), ativação **GELU**, e **Layer Scale** (peso aprendível por canal). O kernel_size=7 (maior que o padrão 3) amplia o campo receptivo por camada.

**Aplicação à Inversão EM:**
- Depthwise convolutions processam cada canal EM independentemente antes de misturá-los, o que é fisicamente sensato: Re(Hxx), Im(Hxx), Re(Hzz) e Im(Hzz) têm sensibilidades diferentes a ρh e ρv
- Layer Scale com inicialização 1e-6 permite que a rede "ignore" blocos desnecessários nas primeiras épocas, facilitando o treinamento em dados EM com alta variância
- O kernel_size=7 cobre uma janela de profundidade maior por camada, capturando melhor a resposta EM difusa (skin depth)

**Vantagens:**
- Desempenho competitivo com Transformers, mas com complexidade O(N) em vez de O(N²)
- Depthwise convolutions respeitam a independência física entre componentes EM
- Treinamento mais estável que ResNets clássicos (GELU + LayerNorm + LayerScale)

**Desvantagens:**
- Layer Normalization pode ser mais lenta que BatchNorm em GPU
- Kernel_size=7 aumenta custo computacional por camada
- Requer datasets maiores para aproveitar a maior capacidade

---

### 3.5 InceptionNet

**Referência:** Szegedy et al. "Going Deeper with Convolutions" (CVPR 2015)
**Célula:** C28 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** n_modules=[3,3,3], filters=[32,64,128], kernel_sizes=[1,3,5]

**Descrição:**
InceptionNet usa **módulos Inception** com 4 ramos paralelos: convoluções 1×1, 3×1 e 5×1 mais MaxPool. Cada ramo captura padrões em escalas diferentes, e os resultados são concatenados. Convoluções 1×1 de bottleneck reduzem a dimensionalidade antes dos kernels maiores.

**Aplicação à Inversão EM:**
- Os três kernels (1, 3, 5) capturam simultaneamente variações locais (limites de camada), médias (transições graduais) e de longo alcance (tendências regionais) nas medições EM
- Isto é análogo à resolução multi-escala da ferramenta EM: a componente Hzz tem maior profundidade de investigação que Hxx, e ambas integram informação em diferentes escalas espaciais
- Particularmente adequado quando o modelo geológico contém tanto camadas finas (1-2m) quanto espessas (>10m)

**Vantagens:**
- Captura multi-escala sem necessidade de pipelines separados
- Bottleneck 1×1 mantém eficiência computacional
- Boa detecção de limites de camada em diferentes escalas

**Desvantagens:**
- Muitos hiperparâmetros (filtros por ramo, número de módulos, bottleneck sizes)
- Concatenação dos ramos aumenta a dimensão do feature map progressivamente
- Pode ser redundante quando as variações de resistividade são predominantemente de uma escala

---

### 3.6 InceptionTime

**Referência:** Fawaz et al. "InceptionTime: Finding AlexNet for Time Series Classification" (DAMI 2020)
**Célula:** C28 | **Tier:** 1 | **Causal:** Adaptável
**Parâmetros:** n_modules=6, bottleneck_filters=32, residual_every=3

**Descrição:**
InceptionTime adapta o módulo Inception para séries temporais, adicionando **conexões residuais a cada 3 módulos** e um **bottleneck 1×1 antes de cada módulo**. Combina a captura multi-escala do Inception com a estabilidade de treinamento do ResNet.

**Aplicação à Inversão EM:**
- Conexões residuais a cada 3 módulos previnem degradação em redes profundas (6+ módulos)
- O bottleneck antes de cada módulo comprime as features EM antes da análise multi-escala, forçando representações eficientes
- Originalmente projetado para séries temporais (como logs de poço), oferecendo desempenho superior ao Inception clássico em dados 1D

**Vantagens:**
- Estado-da-arte para classificação de séries temporais
- Combinação de multi-escala (Inception) + estabilidade (residual)
- Especificamente otimizado para dados 1D sequenciais

**Desvantagens:**
- 6 módulos × 4 ramos = 24 caminhos, aumentando complexidade de debug
- Bottleneck pode limitar a expressividade para sinais EM com alta dimensionalidade (P4 geosinais)

---

### 3.7 CNN_1D

**Referência:** Arquitetura clássica de ConvNet 1D
**Célula:** C29 | **Tier:** 1 | **Causal:** Adaptável
**Parâmetros:** 6 camadas simétricas [32, 64, 128, 128, 64, 32], kernel_size=5

**Descrição:**
CNN puro com 6 camadas Conv1D em configuração simétrica (encoder-decoder sem skip connections). Cada camada aplica convolução, BatchNorm, ativação ReLU e dropout. A simetria [32→128→32] permite expansão e contração do espaço de features.

**Aplicação à Inversão EM:**
- Estrutura simétrica encoder-decoder imita o processo físico: as primeiras camadas extraem features crescentemente abstratas das medições EM, e as últimas reconstroem o perfil de resistividade
- Kernel_size=5 oferece campo receptivo moderado por camada (5 profundidades)
- Modelo baseline simples para validação e comparação

**Vantagens:**
- Implementação extremamente simples e rápida
- Poucos hiperparâmetros para tunar
- Bom baseline para comparação com modelos mais complexos
- Tempo de treinamento e inferência mínimos

**Desvantagens:**
- Sem conexões residuais — dificuldade em redes mais profundas
- Campo receptivo total limitado (~30 profundidades com 6 camadas)
- Não captura dependências de longo alcance

---

### 3.8 TCN (Temporal Convolutional Network)

**Referência:** Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)
**Célula:** C30 | **Tier:** 1 | **Causal:** **Nativo**
**Parâmetros:** num_filters=64, kernel_size=3, dilations=[1,2,4,8,16,32], dropout=0.1

**Descrição:**
TCN usa **convoluções causais dilatadas** (*dilated causal convolutions*) que expandem o campo receptivo exponencialmente sem aumentar o número de parâmetros. Com dilatações [1,2,4,...,32], uma rede de 6 camadas tem campo receptivo efetivo de 63 posições.

**Aplicação à Inversão EM / Geosteering:**
- **Nativo causal**: cada predição na profundidade `d` usa apenas medições em `d-62` a `d`, sem informação futura — ideal para geosteering realtime
- As dilatações crescentes simulam a crescente profundidade de investigação da ferramenta EM: camadas iniciais capturam variações locais (limites de camada próximos), camadas profundas capturam tendências regionais (resistividade de background)
- O campo receptivo de 63 profundidades (63m com SPACING=1.0m) cobre a faixa típica de investigação da ferramenta GeoSphere HD (~30m)

**Vantagens:**
- **Causalidade nativa** — zero vazamento de informação futura
- Campo receptivo exponencial com poucos parâmetros
- Treinamento paralelo (ao contrário de RNNs)
- Excelente latência para geosteering realtime

**Desvantagens:**
- Campo receptivo fixo (determinado pelas dilatações) — não adaptativo
- Pode ter dificuldade com dependências que excedem o campo receptivo
- Não modela explicitamente a física (skin depth, anisotropia)

---

### 3.9 TCN_Advanced

**Referência:** Bai et al. (2018) + Dauphin et al. "Language Modeling with Gated Convolutional Networks" (2017)
**Célula:** C30 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** Similar ao TCN + gated_linear_units=True

**Descrição:**
Versão avançada do TCN que adiciona **Gated Linear Units (GLU)** ao design. GLU aplica um mecanismo de gating sigmóide que permite à rede suprimir seletivamente informação em cada profundidade, similar a um portão de LSTM mas sem recorrência.

**Aplicação à Inversão EM:**
- GLU gating permite suprimir medições EM ruidosas ou ambíguas em certas profundidades
- Particularmente útil em zonas de transição gradual onde a resposta EM é uma mistura complexa das camadas adjacentes
- Combinação de campo receptivo exponencial (dilated conv) + seletividade (GLU)

**Vantagens:**
- Gating melhora a qualidade da inversão em zonas de transição
- Flexibilidade: adaptável a modo causal quando necessário
- Regularização implícita pelo mecanismo de gating

**Desvantagens:**
- ~2× mais parâmetros que TCN básico (GLU duplica os filtros)
- Mais complexo de tunar (interação entre dilatações e gating)
- Benefício marginal em modelos geológicos simples

---

## 4. Família RNN — Redes Recorrentes (2 arquiteturas)

RNNs processam a sequência de medições EM posição por posição, mantendo um estado oculto (*hidden state*) que acumula informação ao longo da profundidade. Isto é análogo a um geofísico que interpreta um log progressivamente, integrando o contexto das medições anteriores.

### 4.1 LSTM (Long Short-Term Memory)

**Referência:** Hochreiter & Schmidhuber (1997)
**Célula:** C31 | **Tier:** 3 | **Causal:** **Nativo**
**Parâmetros:** units=128, n_layers=2, dropout=0.2, return_sequences=True

**Descrição:**
LSTM é uma RNN com portões de entrada, esquecimento e saída que regulam o fluxo de informação no estado de célula. Processa a sequência de forma unidirecional (de cima para baixo no poço), sendo nativamente causal.

**Aplicação à Inversão EM / Geosteering:**
- **Nativo causal**: processa profundidades sequencialmente sem acesso ao futuro
- O estado oculto acumula um "resumo" das medições EM anteriores, similar a como um geofísico mantém contexto mental ao interpretar um log
- Os portões de esquecimento permitem "resetar" o contexto ao cruzar um limite de camada significativo

**Vantagens:**
- Causalidade nativa sem modificação
- Modela dependências de longo alcance via estado de célula
- Interpretabilidade: ativações dos portões indicam quais medições são relevantes

**Desvantagens:**
- **Treinamento sequencial** — não paralelizável, lento em sequências longas (N=600)
- Latência alta para geosteering (processar 600 posições sequencialmente)
- Gradiente pode degradar em sequências muito longas apesar dos portões
- Inferior a TCN em benchmarks empíricos para séries temporais (Bai et al. 2018)

---

### 4.2 BiLSTM (Bidirectional LSTM)

**Referência:** Schuster & Paliwal (1997)
**Célula:** C31 | **Tier:** 2 | **Causal:** **Incompatível**
**Parâmetros:** units=128, n_layers=2, dropout=0.2, return_sequences=True

**Descrição:**
BiLSTM processa a sequência em ambas as direções (topo→base e base→topo) e concatena os estados ocultos. Isto fornece contexto bidirecional em cada posição.

**Aplicação à Inversão EM (offline):**
- Em modo offline (pós-processamento), ter acesso a medições futuras e passadas melhora significativamente a inversão em zonas de transição
- O processamento reverso (base→topo) captura informação "de baixo para cima" que é fisicamente relevante: a resistividade de uma camada profunda influencia as medições em profundidades mais rasas (efeito de *shoulder bed*)
- A concatenação forward + backward fornece representação completa do contexto local

**Vantagens:**
- Contexto bidirecional superior ao LSTM unidirecional para inversão offline
- Captura efeitos de shoulder bed (influência de camadas adjacentes)
- Melhor resolução em limites de camada do que LSTM unidirecional

**Desvantagens:**
- **Incompatível com geosteering** — requer dados futuros
- Mesmo problema de sequencialidade do LSTM (lento para N=600)
- ~2× parâmetros do LSTM (duas direções)

---

## 5. Família Híbrida CNN+RNN (2 arquiteturas)

Modelos híbridos combinam a capacidade de extração de features locais das CNNs com a modelagem de dependências temporais das RNNs.

### 5.1 CNN_LSTM

**Célula:** C32 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** cnn_filters=[64,128], lstm_units=128, kernel_size=3

**Descrição:**
CNN_LSTM aplica camadas Conv1D iniciais para extrair features locais das medições EM, depois alimenta um LSTM que modela dependências de longo alcance. A CNN atua como encoder de features e o LSTM como processador sequencial.

**Aplicação à Inversão EM:**
- A CNN extrai padrões locais (variações de Re/Im perto de limites de camada)
- O LSTM integra esses padrões ao longo da profundidade, mantendo contexto
- Fisicamente análogo a: primeiro detectar features locais da resposta EM, depois integrá-las em um modelo regional de resistividade

**Vantagens:**
- Combina extração local (CNN) com contexto global (LSTM)
- CNN reduz a dimensionalidade antes do LSTM, melhorando eficiência
- Adaptável a modo causal (LSTM já é unidirecional)

**Desvantagens:**
- LSTM no final ainda é sequencial (gargalo de velocidade)
- Interação CNN-LSTM pode ser difícil de tunar (dimensões, dropout)

---

### 5.2 CNN_BiLSTM_ED (Encoder-Decoder)

**Célula:** C32 | **Tier:** 1 | **Causal:** **Incompatível**
**Parâmetros:** cnn_filters=[64,128], lstm_units=128, kernel_size=3

**Descrição:**
Arquitetura encoder-decoder completa: CNN encoder extrai features, BiLSTM processa bidireccionalmente, CNN decoder reconstrói o perfil de resistividade. Skip connections conectam encoder e decoder.

**Aplicação à Inversão EM (offline):**
- Estrutura enc-dec é natural para inversão: encoder comprime medições EM em representação latente, decoder reconstrói perfil de resistividade
- BiLSTM bidirecional no meio captura contexto completo do poço
- Skip connections preservam detalhes de alta frequência (limites de camada)

**Vantagens:**
- Melhor desempenho offline entre os modelos híbridos
- Skip connections preservam resolução vertical
- Tier 1 — validado e estável

**Desvantagens:**
- **Incompatível com geosteering** (BiLSTM requer dados futuros)
- Complexidade de implementação elevada
- Treinamento lento (BiLSTM sequencial)

---

## 6. Família U-Net (14 arquiteturas)

U-Nets são arquiteturas encoder-decoder com **skip connections simétricas** que conectam cada nível do encoder ao nível correspondente do decoder. O encoder faz downsampling (MaxPool/Conv stride) e o decoder faz upsampling, retornando à resolução original. Todas as U-Nets são **incompatíveis com modo causal** porque as skip connections entre encoder e decoder criam dependências não-causais.

### 6.1 Princípio Geral no Contexto de Inversão EM

O encoder da U-Net comprime progressivamente as medições EM em representações de baixa resolução mas alto nível semântico (tendências regionais de resistividade). O decoder reconstrói o perfil detalhado, usando as skip connections para recuperar detalhes de alta frequência (limites de camada). Isto é análogo à inversão em duas etapas: primeiro estimar a macro-estrutura geológica, depois refinar com detalhes locais.

### 6.2 UNet_Inversion

**Célula:** C33 | **Tier:** 2 | **Causal:** Incompatível
**Parâmetros:** filters=[64,128,256], kernel_size=3

U-Net clássica adaptada para inversão 1D. Encoder com Conv1D + MaxPool1D, decoder com UpSampling1D + Conv1D, skip connections via Concatenate.

### 6.3 Attention_UNet

**Célula:** C33 | **Tier:** 1 | **Causal:** Incompatível
**Parâmetros:** filters=[64,128,256], kernel_size=3, attention_filters=auto

Adiciona **Attention Gates** (Oktay et al. 2018) nas skip connections. O gate aprende a ponderar quais features do encoder são relevantes para cada posição do decoder. Isto é particularmente útil em inversão EM onde nem todas as profundidades do encoder contêm informação igualmente relevante — zonas homogêneas têm features redundantes que o gate pode suprimir.

### 6.4 UNet_ResNet18 / UNet_ResNet34 / UNet_ResNet50

**Célula:** C33 | **Tier:** 1 | **Causal:** Incompatível

Substituem o encoder Conv1D simples por backbones ResNet. Blocos residuais no encoder permitem extrair features mais profundas sem degradação de gradiente. ResNet-50 usa bottleneck blocks com expansion=4.

### 6.5 UNet_Attention_ResNet18 / 34 / 50

**Célula:** C33 | **Tier:** 1 | **Causal:** Incompatível

Combinam encoder ResNet + decoder com Attention Gates. Representam a **combinação mais poderosa**: encoder profundo (ResNet) + skip connections seletivas (Attention). UNet_Attention_ResNet18 é destacada como a combinação de maior valor para inversão EM offline.

### 6.6 UNet_ConvNeXt / UNet_Attention_ConvNeXt

**Célula:** C33 | **Tier:** 2 / 1 | **Causal:** Incompatível

Encoder ConvNeXt (Liu 2022) com DepthwiseConv + LayerNorm + GELU. Modernização do encoder U-Net que se beneficia dos avanços do ConvNeXt. A variante com Attention Gate adiciona seletividade nas skip connections.

### 6.7 UNet_Inception / UNet_Attention_Inception

**Célula:** C33 | **Tier:** 2 | **Causal:** Incompatível

Encoder com módulos Inception multi-escala (kernels 1, 3, 5). Captura variações em diferentes escalas espaciais simultaneamente no encoder. A variante com Attention Gate permite seleção dinâmica de features multi-escala.

### 6.8 UNet_EfficientNet / UNet_Attention_EfficientNet

**Célula:** C33 | **Tier:** 2 / 1 | **Causal:** Incompatível
**Referência:** Tan & Le "EfficientNet: Rethinking Model Scaling for CNNs" (ICML 2019)

Encoder com blocos MBConv (Mobile Inverted Bottleneck): expand → DepthwiseConv → Squeeze-and-Excite → project. Ativação Swish/SiLU. Extremamente eficiente em parâmetros. A variante com Attention Gate é Tier 1.

### 6.9 Vantagens e Desvantagens Comuns das U-Nets

**Vantagens:**
- Excelente preservação de detalhes via skip connections (resolução vertical)
- Encoder-decoder naturalmente adequado para inversão (compressão → reconstrução)
- 14 variantes permitem escolher o encoder ideal para cada cenário
- Attention Gates melhoram significativamente a qualidade (supressão seletiva)

**Desvantagens:**
- **Todas incompatíveis com geosteering realtime** (skip connections não-causais)
- MaxPool1D no encoder reduz resolução — depende do upsample para reconstruir
- Muitas variantes podem confundir a seleção de modelo
- Custo computacional elevado (encoder + decoder + skip connections)

---

## 7. Família Transformer (6 arquiteturas)

Transformers usam **mecanismo de atenção** (*self-attention*) que permite a cada posição na sequência "olhar" para todas as outras posições, capturando dependências de alcance arbitrário. No contexto de inversão EM, isto permite que cada profundidade considere medições em profundidades distantes que podem influenciar a resposta EM local (efeito de shoulder bed, skin depth).

### 7.1 Transformer (Encoder)

**Referência:** Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
**Célula:** C34 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** d_model=128, n_heads=8, n_layers=4, d_ff=512

**Descrição:**
Transformer Encoder padrão com Multi-Head Attention, Positional Encoding e Feed-Forward Network. Cada camada aplica self-attention (captura dependências globais) seguida de FFN point-wise (transformação não-linear local).

**Aplicação à Inversão EM:**
- Self-attention permite que cada profundidade "consulte" todas as outras profundidades, capturando dependências de longo alcance como shoulder bed effects e variações regionais de resistividade
- 8 cabeças de atenção podem especializar-se em diferentes aspectos: padrões de limites de camada, tendências de background, correlações entre Hxx e Hzz, etc.
- Positional Encoding informa a posição relativa no poço, mantendo a noção de "distância" entre profundidades

**Vantagens:**
- Captura dependências de alcance arbitrário (não limitado por campo receptivo)
- Paralelizável (ao contrário de RNNs)
- Interpretabilidade via mapas de atenção
- Adaptável a modo causal via causal mask

**Desvantagens:**
- Complexidade O(N²) com N=600 — custo quadrático significativo
- Sem bias indutivo para localidade (toda posição tem peso igual a priori)
- Requer Positional Encoding para noção de ordem
- Pode focar em correlações espúrias em datasets pequenos

---

### 7.2 Simple_TFT (Temporal Fusion Transformer — Simplificado)

**Referência:** Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2021) — variante simplificada
**Célula:** C34 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** d_model=128, n_heads=4, n_layers=2, d_ff=256

**Descrição:**
Versão simplificada do TFT com Gated Linear Units (GLU) e gated skip connections. O GLU permite supressão seletiva de features, e o skip connection gated permite que a rede "pule" os blocos de atenção quando eles não são informativos.

**Aplicação à Inversão EM:**
- GLU gating é útil para lidar com a variabilidade das medições EM: em zonas homogêneas, o gate pode suprimir a atenção (informação redundante), enquanto em zonas de transição, o gate ativa a atenção total
- Skip connections gated preservam informação direta das medições EM quando a atenção não agrega valor

**Vantagens:**
- Gating melhora estabilidade de treinamento
- Mais leve que o TFT completo
- Bom tradeoff entre complexidade e desempenho

**Desvantagens:**
- Sem Variable Selection Network (VSN) — não seleciona features automaticamente
- Interpretabilidade limitada comparado ao TFT completo

---

### 7.3 TFT (Temporal Fusion Transformer — Completo)

**Referência:** Lim et al. (2021) — implementação completa
**Célula:** C34 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** d_model=128, n_heads=4, n_layers=2, d_ff=256, grn_hidden=128

**Descrição:**
Implementação completa do TFT com **Gated Residual Network (GRN)** e **Variable Selection Network (VSN)**. O VSN aplica um GRN por feature de entrada, pondera-as com softmax, e seleciona automaticamente quais features são mais relevantes. GRN pós-atenção com Dense→ELU→Dense→GLU→LayerNorm→Residual.

**Aplicação à Inversão EM:**
- VSN é **particularmente valioso** para inversão EM com features expandidas (P2-P4): quando temos 5-17 features (incluindo θ, f, geosinais), o VSN aprende automaticamente quais features são mais informativas para a inversão em cada profundidade
- GRN pós-atenção permite supressão seletiva de informação redundante
- A combinação VSN + GRN oferece **interpretabilidade**: os pesos do VSN indicam a importância relativa de Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz), θ, f, geosinais

**Vantagens:**
- **Variable Selection** — seleção automática de features EM relevantes
- **Interpretabilidade** — pesos do VSN indicam importância das variáveis
- GRN + GLU para regularização via gating
- Ideal para perspectivas P2-P4 (muitas features)

**Desvantagens:**
- Complexidade elevada (VSN + GRN + MHA + skip gated)
- Requer mais dados para convergir (mais parâmetros)
- n_features deve ser conhecido em build time

---

### 7.4 PatchTST

**Referência:** Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)
**Célula:** C34 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** patch_size=16, d_model=128, n_heads=8, n_layers=3

**Descrição:**
PatchTST segmenta a sequência em **patches** (janelas fixas de 16 posições) antes de alimentar o Transformer. Cada patch é comprimido via Conv1D stride em um vetor d_model, reduzindo o comprimento da sequência de N=600 para N/16=37 patches. Após o Transformer, cada patch é expandido de volta para 16 posições.

**Aplicação à Inversão EM:**
- Reduce complexidade de O(600²)=360.000 para O(37²)=1.369 — **260× mais eficiente**
- Cada patch de 16 profundidades (16m com SPACING=1.0m) cobre uma faixa espacial comparável ao comprimento de onda do campo EM na frequência de operação
- Patches fornecem representações locais ricas antes da atenção global, combinando localidade (intra-patch) com dependências de longo alcance (inter-patch)
- O upsample de volta preserva N_MEDIDAS usando Dense → reshape → slice

**Vantagens:**
- **Eficiência O((N/P)²)** — dramática redução de custo computacional
- Representações locais ricas (Conv1D por patch) antes da atenção global
- Estado-da-arte para forecasting de séries temporais longas
- Patch_size=16 naturalmente compatível com a escala de investigação EM

**Desvantagens:**
- Perda de resolução intra-patch (16 posições comprimidas em 1 vetor)
- Upsample pode introduzir artefatos nos limites entre patches
- Requer que seq_len seja compatível com patch_size (padding se necessário)

---

### 7.5 Autoformer

**Referência:** Wu et al. "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (NeurIPS 2021)
**Célula:** C34 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** d_model=128, n_heads=8, n_layers=2, moving_avg=25

**Descrição:**
Autoformer introduz dois conceitos: (1) **Series Decomposition** — decomposição em trend (média móvel) e seasonal (residual) aplicada entre cada bloco, e (2) **Auto-Correlation** — substituição do self-attention por correlação temporal via FFT, com complexidade O(N log N) em vez de O(N²).

**Aplicação à Inversão EM:**
- Decomposição trend/seasonal é fisicamente interpretável: o trend corresponde à resistividade de background (macro-escala) e o seasonal corresponde às variações finas (limites de camada, camadas finas)
- Moving_avg=25 (25m) é comparável ao DOI (Depth of Investigation) da ferramenta EM a 20 kHz
- Auto-Correlation via FFT captura periodicidades na resposta EM — relevante para modelos com camadas repetitivas (flysch, turbiditos)

**Vantagens:**
- Decomposição trend/seasonal com interpretação geofísica direta
- O(N log N) via FFT — mais eficiente que atenção padrão O(N²)
- Periodicidade capturada naturalmente (FFT)
- Moving_avg=25 alinhado com escala de investigação da ferramenta

**Desvantagens:**
- Decomposição assume que trend é capturável por média móvel — pode falhar para variações bruscas
- Auto-Correlation assume que a relação entre posições é periódica — nem sempre verdade em geologia
- Implementação FFT pode ter edge effects nas bordas da sequência

---

### 7.6 iTransformer

**Referência:** Liu et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024)
**Célula:** C34 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** d_model=64, n_heads=4, n_layers=2, d_ff=128

**Descrição:**
iTransformer **inverte** a dimensão da atenção: em vez de atenção entre posições temporais (standard), aplica atenção entre **features** (variáveis). Cada feature EM é tratada como um "token", e a atenção captura dependências inter-variáveis. Implementado via `_InvertedMultiHeadAttention` com projeções Q/K/V no eixo d_model (estático) e transposição interna para feature-attention. A FFN usa Conv1D(kernel=1) para preservar a dimensão temporal dinâmica.

**Aplicação à Inversão EM:**
- Captura dependências entre as variáveis EM (Re/Im de Hxx e Hzz) que são fisicamente acopladas pelas equações de Maxwell em meios anisotrópicos
- A relação entre Re(Hxx) e Im(Hzz), por exemplo, contém informação sobre o coeficiente de anisotropia λ — o iTransformer pode aprender essa relação diretamente
- Particularmente valioso nas perspectivas P2-P4 onde há muitas features inter-relacionadas (θ, f, geosinais como atenuação, phase shift, geosinal de anisotropia)

**Vantagens:**
- Captura dependências inter-variáveis (fisicamente motivado para EM anisotrópico)
- Complementar ao Transformer padrão (temporal) — pode ser combinado
- Leve (d_model=64, menos parâmetros que Transformer padrão)
- Eficiente para muitas features (P4: até 17 features)

**Desvantagens:**
- Não captura dependências temporais diretamente (apenas inter-features)
- A FFN Conv1D(kernel=1) point-wise limita a interação temporal
- Requer combinação com mecanismo temporal para capturar ambas as dimensões

---

## 8. Família Decomposição (2 arquiteturas)

Modelos baseados em decomposição expressam a saída como combinação linear de funções base aprendidas, oferecendo interpretabilidade parcial.

### 8.1 N-BEATS

**Referência:** Oreshkin et al. "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting" (ICLR 2020)
**Célula:** C35 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** n_stacks=2, n_blocks=3, hidden_units=256, theta_dim=4

**Descrição:**
N-BEATS decompõe a saída em funções base aprendidas (*basis expansion*). Cada bloco produz coeficientes θ que parametrizam funções base, e a saída é a combinação linear dessas bases. Blocos são empilhados em stacks com conexões residuais entre eles.

**Aplicação à Inversão EM:**
- Funções base aprendidas podem corresponder a padrões geofísicos: constante (camada homogênea), linear (gradiente de resistividade), sigmoide (limite de camada)
- Stacks residuais permitem decomposição hierárquica: primeiro stack captura macro-escala, segundo stack refina detalhes
- θ_dim=4 → 4 funções base por bloco é suficiente para capturar variações de resistividade em modelos 1D

**Vantagens:**
- Interpretabilidade parcial via funções base
- Decomposição hierárquica (macro → detalhe)
- Treinamento estável com conexões residuais inter-bloco
- Sem suposição de localidade ou periodicidade

**Desvantagens:**
- Necessita de ajuste fino do theta_dim e número de stacks/blocks
- Não modela explicitamente a física EM
- Pode ter dificuldade com descontinuidades abruptas (limites de camada nítidos)

---

### 8.2 N-HiTS

**Referência:** Challu et al. "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (AAAI 2023)
**Célula:** C35 | **Tier:** 2 | **Causal:** Adaptável
**Parâmetros:** n_stacks=2, n_blocks=3, hidden_units=256, pool_sizes=[2,4,8]

**Descrição:**
N-HiTS estende o N-BEATS com **pooling hierárquico multi-resolução**. Cada bloco opera em uma resolução diferente (pool_sizes=[2,4,8]), permitindo captura simultânea de padrões em múltiplas escalas.

**Aplicação à Inversão EM:**
- Pooling [2,4,8] opera em escalas de 2m, 4m e 8m — captura desde limites de camada finos até variações de bloco
- Hierarquia multi-resolução é análoga à resolução variável da ferramenta EM: componentes próximas ao Tx/Rx "veem" melhor camadas finas, componentes distantes "veem" melhor camadas grossas
- Interpolação hierárquica no decoder permite reconstrução suave do perfil de resistividade

**Vantagens:**
- Multi-resolução hierárquica (análogo à resolução EM variável)
- Mais eficiente que N-BEATS (pooling reduz computação)
- Combina macro e micro escala naturalmente

**Desvantagens:**
- Pool sizes fixos podem não ser ótimos para todas as frequências/espaçamentos
- Interpolação pode suavizar excessivamente limites de camada abruptos

---

## 9. Família Operador (2 arquiteturas)

Operadores neurais aprendem o mapeamento entre espaços de funções, não apenas entre vetores. Isto é particularmente relevante para inversão EM, onde queremos aprender o operador que mapeia "perfil de medições EM" → "perfil de resistividade".

### 9.1 FNO (Fourier Neural Operator)

**Referência:** Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" (ICLR 2021)
**Célula:** C36 | **Tier:** 3 | **Causal:** Adaptável*
**Parâmetros:** modes=16, width=64, n_layers=4

**Descrição:**
FNO opera no domínio de frequência via FFT. Cada camada aplica: (1) transformada para o domínio de Fourier, (2) multiplicação com pesos aprendidos nos primeiros `modes` modos de Fourier, (3) transformada inversa. Isto captura padrões periódicos e globais eficientemente.

**Aplicação à Inversão EM:**
- A inversão EM é fundamentalmente um problema de equações diferenciais parciais (Maxwell em meio anisotrópico) — FNO é especificamente projetado para PDE operators
- A transformada de Fourier dos perfis EM frequentemente revela padrões periódicos (camadas alternantes)
- `modes=16` captura os 16 modos de frequência mais significativos do perfil

**Vantagens:**
- Fundamentação teórica sólida para problemas de PDE (Maxwell)
- Captura padrões globais eficientemente via FFT
- Independente de discretização (generaliza para diferentes N_MEDIDAS)
- O(N log N) via FFT

**Desvantagens:**
- FFT global viola causalidade estritamente (*adaptável com restrição espectral)
- Pode capturar correlações espúrias em perfis não-periódicos
- Menos intuitivo que CNNs para dados 1D
- Tier 3: requer mais validação em inversão EM

---

### 9.2 DeepONet (Deep Operator Network)

**Referência:** Lu et al. "Learning Nonlinear Operators via DeepONet" (Nature Machine Intelligence, 2021)
**Célula:** C36 | **Tier:** 3 | **Causal:** **Incompatível**
**Parâmetros:** branch_layers=[128,128], trunk_layers=[128,128]

**Descrição:**
DeepONet consiste em duas redes: **Branch** (processa o input funcional — medições EM) e **Trunk** (processa os pontos de avaliação — profundidades). O output é o produto escalar dos outputs de Branch e Trunk. Baseado no teorema universal de aproximação de operadores (Chen & Chen 1995).

**Aplicação à Inversão EM:**
- Branch processa o perfil completo de medições EM (input global)
- Trunk processa cada profundidade individualmente, parametrizando a saída
- Fundamentação teórica: o operador F: medições_EM → resistividade é um operador não-linear entre espaços de funções, exatamente o que DeepONet aproxima

**Vantagens:**
- Fundamentação teórica forte (teorema de aproximação universal de operadores)
- Separação Branch/Trunk permite generalização para diferentes discretizações
- Potencialmente superior para problemas de operador puro

**Desvantagens:**
- **Incompatível com geosteering** — Branch processa input global
- Implementação complexa (duas redes + produto escalar)
- Treinamento instável sem inicialização cuidadosa
- Tier 3: experimental neste contexto

---

## 10. Atenção Geofísica e Dense (2 arquiteturas)

### 10.1 Geophysical_Attention

**Célula:** C36 | **Tier:** 3 | **Causal:** Adaptável
**Parâmetros:** d_model=128, n_heads=4, n_layers=3

**Descrição:**
Variante de Transformer com atenção customizada para dados geofísicos. Incorpora bias de posição baseado na distância entre profundidades e pesos de atenção inicializados para favorecer vizinhança local (prior geofísico: medições próximas são mais correlacionadas).

**Aplicação à Inversão EM:**
- Prior de vizinhança local respeita a física: a resposta EM em uma profundidade é mais influenciada por formações próximas (skin depth limita a profundidade de investigação)
- Bias posicional decai com a distância, imitando o decaimento do campo EM

**Vantagens:**
- Incorpora priors geofísicos diretamente na arquitetura
- Mais focado que Transformer genérico para dados de poço

**Desvantagens:**
- Priors fixos podem não ser ótimos para todas as formações
- Tier 3: experimental, requer mais validação

---

### 10.2 DNN (Deep Neural Network)

**Célula:** C36 | **Tier:** 3 | **Causal:** Adaptável
**Parâmetros:** hidden_layers=[256, 128, 64], dropout=0.2

**Descrição:**
Rede densa (feedforward) simples com TimeDistributed Dense. Cada profundidade é processada independentemente — sem interação entre posições vizinhas.

**Aplicação à Inversão EM:**
- Baseline mais simples possível: aprende o mapeamento point-wise de medições EM → resistividade em cada profundidade
- Não captura dependências espaciais — cada profundidade é independente
- Útil como referência para quantificar o benefício de modelos com contexto espacial

**Vantagens:**
- Implementação trivial
- Treinamento extremamente rápido
- Baseline universal para comparação

**Desvantagens:**
- Sem contexto espacial (cada profundidade é independente)
- Não captura shoulder bed effects, limites de camada, etc.
- Tipicamente inferior a qualquer modelo com contexto temporal

---

## 11. Família Geosteering (5 arquiteturas especializadas)

Estas arquiteturas são **especificamente projetadas para geosteering realtime**, priorizando causalidade, latência e estimativa de incerteza.

### 11.1 WaveNet

**Referência:** van den Oord et al. "WaveNet: A Generative Model for Raw Audio" (2016)
**Célula:** C36A | **Tier:** G | **Causal:** **Nativo**
**Parâmetros:** residual_channels=64, skip_channels=256, dilations=[1,2,...,512], n_stacks=2

**Descrição:**
WaveNet usa **convoluções causais dilatadas empilhadas** com conexões residuais e skip connections. Cada camada tem dilatação crescente (1,2,4,...,512), e múltiplos stacks repetem o padrão. O campo receptivo total com 2 stacks × 10 camadas é de 2×1023 = 2046 posições.

**Aplicação ao Geosteering:**
- **Causalidade nativa**: projetado para geração autoregressiva, cada output depende apenas de inputs anteriores
- Campo receptivo de 2046m cobre muito além do DOI da ferramenta (~30m), capturando tendências regionais
- Skip connections de cada camada para a saída preservam informação multi-escala
- Originalmente projetado para áudio (44.1 kHz) — dados EM a 600 amostras são comparativamente curtos, permitindo processamento rápido

**Vantagens:**
- **Melhor tradeoff causalidade + campo receptivo** entre as arquiteturas causais
- Latência constante O(1) por nova amostra (após preenchimento do buffer)
- Skip connections multi-escala preservam detalhes e contexto
- Treinamento paralelo (todas as convoluções operam em paralelo)

**Desvantagens:**
- Muitos parâmetros (residual + skip channels × layers × stacks)
- Campo receptivo fixo (não adaptativo)
- Gating fixo (sigmoid × tanh) — menos flexível que GLU

---

### 11.2 Causal_Transformer

**Referência:** Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
**Célula:** C36A | **Tier:** G | **Causal:** **Nativo**
**Parâmetros:** d_model=128, n_heads=8, n_layers=6, d_ff=512, use_kv_cache=True

**Descrição:**
Transformer com **causal mask nativa** (estilo GPT) — a atenção em cada posição é mascarada para ver apenas posições anteriores ou iguais. Com **KV-cache** (*Key-Value Cache*) para inferência incremental: em realtime, apenas a nova posição é processada, reutilizando os K/V anteriores.

**Aplicação ao Geosteering:**
- **Causal mask**: garante que posição `d` veja apenas `d'≤d`, ideal para geosteering
- **KV-cache**: em inferência realtime, cada nova profundidade é processada em O(N) em vez de O(N²), com N apenas para o cálculo de atenção da nova posição contra o cache
- Atenção global (mesmo mascarada) permite capturar padrões de longo alcance no histórico do poço
- 6 camadas × 8 cabeças oferecem alta capacidade para modelar relações complexas entre medições EM históricas e resistividade atual

**Vantagens:**
- Captura dependências de longo alcance com causalidade
- KV-cache para inferência incremental rápida
- Interpretabilidade via mapas de atenção (quais profundidades passadas são mais informativas)

**Desvantagens:**
- Complexidade O(N²) no treinamento (O(N) com KV-cache na inferência)
- 6 camadas × 512 FFN = muitos parâmetros
- Sem bias de localidade (pode focar em posições distantes irrelevantes)
- Requer sequências longas de treinamento para aprender atenção causal eficaz

---

### 11.3 Informer

**Referência:** Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021)
**Célula:** C36A | **Tier:** G | **Causal:** Via máscara
**Parâmetros:** d_model=128, n_heads=8, enc_layers=3, dec_layers=2, factor=5

**Descrição:**
Informer usa **ProbSparse self-attention** que seleciona apenas as top-K queries mais informativas (baseado em KL-divergência), reduzindo complexidade de O(N²) para O(N log N). Também inclui **distilling** (convolução + MaxPool entre camadas) para comprimir a sequência progressivamente.

**Aplicação ao Geosteering:**
- ProbSparse attention é ideal para logs EM onde muitas profundidades são redundantes (zonas homogêneas) — o mecanismo de seleção foca nas profundidades mais informativas (limites de camada)
- Distilling progressivo comprime as 600 posições, focando nos padrões essenciais
- Complexidade O(N log N) permite processar sequências longas rapidamente

**Vantagens:**
- **O(N log N)** — escalável para sequências longas
- Seleção automática de profundidades mais informativas
- Distilling comprime informação redundante (zonas homogêneas)

**Desvantagens:**
- Probabilistic sparse attention adiciona variabilidade ao treinamento
- Distilling com MaxPool pode perder detalhes finos
- Implementação complexa

---

### 11.4 Mamba/S4 (State Space Model)

**Referência:** Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR 2022) + Gu & Dao "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
**Célula:** C36A | **Tier:** G | **Causal:** **Nativo**
**Parâmetros:** d_model=128, d_state=16, d_conv=4, expand=2, n_layers=6

**Descrição:**
Mamba/S4 são **modelos de espaço de estados** (*State Space Models* — SSMs) que processam sequências em tempo linear O(N) usando recorrência estruturada. O S4 parametriza a matriz de transição de estados usando matrizes diagonalizáveis (HiPPO), permitindo campo receptivo infinito teórico com O(1) por step.

**Aplicação ao Geosteering:**
- **O(1) por step de inferência** — a menor latência possível para geosteering realtime
- **Campo receptivo infinito teórico** — o estado acumula informação de todo o histórico sem custo adicional
- **Seletividade** (Mamba): mecanismo de gating que permite ao modelo decidir quais informações armazenar/esquecer no estado, análogo a um geofísico experiente que sabe quando atualizar sua interpretação
- d_state=16 → 16 dimensões de estado comprimem toda a informação histórica relevante

**Vantagens:**
- **Menor latência** — O(1) por step de inferência realtime
- Campo receptivo infinito sem custo computacional crescente
- Seletividade (Mamba) para reter/esquecer informação adaptivamente
- **Causalidade nativa** — processamento puramente unidirecional
- Treinamento paralelo O(N) via convolução no domínio de Fourier

**Desvantagens:**
- Implementação complexa (HiPPO initialization, discretização)
- d_state=16 pode ser insuficiente para modelos geológicos muito complexos
- Sem mapas de atenção interpretáveis (estado oculto é opaco)
- Tecnologia relativamente nova, menos validada em geofísica

---

### 11.5 Encoder_Forecaster

**Célula:** C36A | **Tier:** G | **Causal:** **Nativo**
**Parâmetros:** enc_units=128, dec_units=128, enc_layers=3, dec_layers=2

**Descrição:**
Arquitetura seq2seq com **encoder** que processa as medições EM históricas e **forecaster (decoder)** que projeta a resistividade futura. O encoder comprime o contexto histórico em um vetor latente, e o decoder realiza **look-ahead explícito** — prevê a resistividade nas próximas W posições à frente da posição atual.

**Aplicação ao Geosteering:**
- **Look-ahead explícito**: prevê resistividade não apenas na posição atual, mas também nas próximas W posições — antecipa mudanças de camada antes de chegar nelas
- Encoder acumula contexto de todo o histórico de medições
- Decoder pode gerar predições para W steps futuros autoregressivamente
- Em geosteering, antecipar um limite de camada 5-10m à frente permite ao direcional tomar decisões de correção de trajetória a tempo

**Vantagens:**
- **Look-ahead** — antecipa mudanças geológicas futuras
- Encoder-decoder natural para o problema de inversão
- Causalidade garantida (encoder só vê passado, decoder projeta futuro)
- Estimativa de incerteza natural (variabilidade das predições futuras)

**Desvantagens:**
- Qualidade do look-ahead degrada com a distância (incerteza cresce)
- Decoder autoregressivo é sequencial na inferência
- Requer definição do horizonte W de predição

---

## 12. Tabela Comparativa Completa

### 12.1 Resumo Geral

| # | Arquitetura | Família | Tier | Causal | Complexidade | Parâmetros | Força Principal | Fraqueza Principal |
|:-:|:------------|:--------|:----:|:------:|:------------:|:----------:|:----------------|:-------------------|
| 1 | ResNet-18 ★ | CNN | 1 | Adapt. | O(N) | ~500K | Baseline estável | Campo receptivo limitado |
| 2 | ResNet-34 | CNN | 1 | Adapt. | O(N) | ~1M | Maior capacidade | Mais lento |
| 3 | ResNet-50 | CNN | 1 | Adapt. | O(N) | ~800K | Bottleneck eficiente | Complexo |
| 4 | ConvNeXt | CNN | 1 | Adapt. | O(N) | ~1.5M | Moderno, DepthConv | Kernel=7 custoso |
| 5 | InceptionNet | CNN | 2 | Adapt. | O(N) | ~600K | Multi-escala | Muitos hiperparams |
| 6 | InceptionTime | CNN | 1 | Adapt. | O(N) | ~700K | Série temporal SOTA | Complexo |
| 7 | CNN_1D | CNN | 1 | Adapt. | O(N) | ~200K | Simples e rápido | Sem residuals |
| 8 | TCN | CNN | 1 | **Nat.** | O(N) | ~300K | Causal dilated | Campo fixo |
| 9 | TCN_Adv | CNN | 2 | Adapt. | O(N) | ~600K | GLU gating | 2× parâmetros |
| 10 | LSTM | RNN | 3 | **Nat.** | O(N²)* | ~400K | Causal nativo | Lento (sequencial) |
| 11 | BiLSTM | RNN | 2 | Incomp. | O(N²)* | ~800K | Bidirecional | Offline only |
| 12 | CNN_LSTM | Híb. | 2 | Adapt. | O(N) | ~500K | Local + global | LSTM gargalo |
| 13 | CNN_BiLSTM_ED | Híb. | 1 | Incomp. | O(N²)* | ~1M | Enc-Dec completo | Offline only |
| 14-27 | U-Nets (14) | U-Net | 1-2 | Incomp. | O(N) | ~1-5M | Skip connections | Offline only |
| 28 | Transformer | Trans. | 2 | Adapt. | O(N²) | ~1M | Atenção global | Custo quadrático |
| 29 | Simple_TFT | Trans. | 2 | Adapt. | O(N²) | ~800K | GLU gating | Sem VSN |
| 30 | TFT | Trans. | 2 | Adapt. | O(N²) | ~1.2M | VSN + GRN | Complexo |
| 31 | PatchTST | Trans. | 2 | Adapt. | O((N/P)²) | ~900K | Eficiente | Perda intra-patch |
| 32 | Autoformer | Trans. | 2 | Adapt. | O(NlogN) | ~800K | Decomposição | FFT edges |
| 33 | iTransformer | Trans. | 2 | Adapt. | O(d²) | ~300K | Feature attention | Sem temporal |
| 34 | N-BEATS | Decomp. | 2 | Adapt. | O(N) | ~500K | Basis expansion | Descontinuidades |
| 35 | N-HiTS | Decomp. | 2 | Adapt. | O(N) | ~400K | Multi-resolução | Pool fixo |
| 36 | FNO | Oper. | 3 | Adapt.* | O(NlogN) | ~300K | PDE theory | FFT global |
| 37 | DeepONet | Oper. | 3 | Incomp. | O(N) | ~500K | Op. universal | Branch global |
| 38 | Geo_Attn | Attn. | 3 | Adapt. | O(N²) | ~600K | Prior geofísico | Experimental |
| 39 | DNN | Dense | 3 | Adapt. | O(N) | ~100K | Baseline mínimo | Sem contexto |
| 40 | WaveNet | Geo | G | **Nat.** | O(N) | ~1.5M | Skip + dilated | Muitos params |
| 41 | Causal_Tr | Geo | G | **Nat.** | O(N²) | ~2M | KV-cache | Custo treinamento |
| 42 | Informer | Geo | G | Mask | O(NlogN) | ~1.5M | Sparse attention | Complexo |
| 43 | Mamba/S4 | Geo | G | **Nat.** | **O(N)** | ~500K | **O(1) inferência** | Novo, menos validado |
| 44 | Enc_Forec | Geo | G | **Nat.** | O(N) | ~700K | Look-ahead | Degrada com distância |

*RNNs: O(N²) refere-se ao custo total de treinamento (N steps × N hidden). Inferência é O(N).

### 12.2 Recomendações por Cenário de Uso

| Cenário | Recomendação Primária | Alternativas | Justificativa |
|:--------|:----------------------|:-------------|:--------------|
| **Baseline offline** | ResNet-18 ★ | CNN_1D, ResNet-34 | Estável, validado, Tier 1 |
| **Offline alta qualidade** | Attention_UNet | UNet_Attention_ResNet18, CNN_BiLSTM_ED | Skip connections + attention |
| **Offline muitas features (P2-P4)** | TFT | iTransformer, Transformer | VSN para seleção de features |
| **Offline modelos complexos** | ResNet-50, ConvNeXt | InceptionTime, PatchTST | Alta capacidade |
| **Geosteering — latência mínima** | Mamba/S4 | TCN | O(1) inferência, causal nativo |
| **Geosteering — campo receptivo máx.** | WaveNet | Causal_Transformer | 2046 posições, skip multi-escala |
| **Geosteering — interpretabilidade** | Causal_Transformer | Encoder_Forecaster | Mapas de atenção |
| **Geosteering — antecipação** | Encoder_Forecaster | — | Look-ahead explícito |
| **Eficiência computacional** | PatchTST | N-HiTS, CNN_1D | O((N/P)²), patch-based |
| **Séries temporais longas (N>1000)** | PatchTST, Autoformer | Informer, N-HiTS | Sub-quadrático |
| **Multi-escala** | InceptionTime | N-HiTS, Autoformer | Kernels paralelos [1,3,5] |
| **Fundamentação teórica (PDE)** | FNO | DeepONet | Fourier Neural Operator |

---

## 13. Impacto da Causalidade no Geosteering

### 13.1 Por Que a Causalidade é Crítica

Em operações de geosteering, a broca avança continuamente e decisões de direcionamento devem ser tomadas **em tempo real** com base apenas nas medições já adquiridas. Se a rede utiliza informação futura (medições ainda não realizadas), o modelo é **inaplicável em campo**.

A causalidade no contexto da inversão EM significa:
- **Padding "causal"**: convoluções só usam posições anteriores ou iguais
- **Causal mask**: atenção mascarada para Q/K onde K.pos > Q.pos são zerados
- **return_sequences=True** + unidirecional: RNNs processam de cima para baixo

### 13.2 Impacto na Qualidade da Inversão

A restrição causal **sempre degrada** a qualidade da inversão em relação ao modo offline:

| Aspecto | Offline (acausal) | Realtime (causal) | Degradação |
|:--------|:------------------|:------------------|:-----------|
| Resolução em limites de camada | Alta | Média-Baixa | Sem "previsão" da transição abaixo |
| Shoulder bed compensation | Bidirecional | Unidirecional | Perda da compensação bottom-up |
| Estimativa de anisotropia | Contexto completo | Contexto parcial | λ = √(ρv/ρh) menos preciso |
| Latência de resposta | N/A | Depende da arquitetura | TCN/Mamba: baixa, Transformer: média |
| Estimativa de incerteza | Opcional | **Automática** | Incerteza é essencial em realtime |

### 13.3 Hierarquia de Recomendação para Geosteering

```
1. Mamba/S4      → O(1) inferência, campo receptivo infinito, causal nativo
2. WaveNet       → Campo receptivo 2046, skip multi-escala, causal nativo
3. TCN           → Simples, eficiente, causal nativo, campo 63
4. Causal_Trans. → Atenção global causal, KV-cache, interpretável
5. Enc_Forecaster→ Look-ahead explícito, antecipa mudanças
6. ResNet-18     → Adaptável (padding causal), baseline sólido
```

---

## 14. Relação com Conceitos Geofísicos

### 14.1 Skin Depth e Campo Receptivo

O **skin depth** δ = √(2ρ/(ωμ₀)) define a profundidade de penetração do campo EM. A 20 kHz:

| ρ (Ω·m) | δ (m) | Comentário |
|:---------|:------|:-----------|
| 1 | 3.6 | Folhelho (alta condutividade) |
| 10 | 11.3 | Arenito argiloso |
| 100 | 35.6 | Arenito limpo |
| 1000 | 112.5 | Carbonato compacto |

O **campo receptivo** da rede deve ser compatível com o skin depth:
- TCN (63 posições = 63m): cobre skin depth para ρ ≤ 100 Ω·m
- WaveNet (2046 posições): cobre qualquer skin depth prático
- Mamba/S4 (infinito teórico): sempre adequado

### 14.2 Resolução Vertical e Tipo de Arquitetura

A **resolução vertical** (capacidade de distinguir camadas finas) depende do tipo de arquitetura:
- **U-Nets com skip connections**: melhor resolução (detalhes preservados via skips)
- **CNNs profundas (ResNet-50, ConvNeXt)**: boa resolução com kernels pequenos
- **Transformers**: resolução depende do Positional Encoding (pode ser limitada)
- **RNNs**: resolução limitada pela "memória" do estado oculto

### 14.3 Anisotropia TIV e Seleção de Features

O coeficiente de anisotropia λ = √(ρv/ρh) é extraído da relação entre as componentes Hxx (sensível a ρh) e Hzz (sensível a ρv). Arquiteturas que modelam explicitamente a interação entre features são vantajosas:
- **iTransformer**: atenção entre features captura a relação Hxx↔Hzz diretamente
- **TFT com VSN**: seleção automática de features revela quais componentes são mais informativas para ρh vs. ρv
- **InceptionNet**: kernels multi-escala capturam relação Hxx↔Hzz em diferentes escalas espaciais

### 14.4 Ângulo de Incidência (Perspectiva P2) e Sequências Multi-Ângulo

Em poços direcionais, o ângulo θ entre o poço e as camadas varia, modificando a resposta EM. O pipeline usa `Input(shape=(None, N_FEATURES))` onde `None` permite sequências de comprimento variável (multi-ângulo). Isto é crucial para:
- **Transformers**: N_MEDIDAS pode variar entre ângulos sem retraining
- **CNNs com padding="same"**: operam em qualquer comprimento
- **N-BEATS/N-HiTS com theta_dim**: funções base se adaptam ao comprimento

---

## 15. Conclusão

O pipeline v5.0.15 oferece **44 arquiteturas** organizadas em 10 famílias, cobrindo desde baselines simples (DNN, CNN_1D) até modelos de estado-da-arte especializados (Mamba/S4, TFT, PatchTST). A seleção da arquitetura deve considerar:

1. **Modo de operação**: offline (todas as 44) vs. realtime/geosteering (27 compatíveis)
2. **Complexidade do modelo geológico**: simples (ResNet-18, CNN_1D) vs. complexo (ResNet-50, TFT, U-Nets)
3. **Número de features (Perspectiva)**: P1 baseline (5) → qualquer; P2-P4 (6-17) → TFT, iTransformer
4. **Latência**: batch (qualquer) vs. realtime (Mamba < WaveNet < TCN < Causal_Transformer)
5. **Interpretabilidade**: TFT (pesos VSN), Causal_Transformer (mapas de atenção), N-BEATS (funções base)
6. **Dataset size**: pequeno (<1K modelos) → ResNet-18, CNN_1D; grande (>10K) → ResNet-50, TFT, U-Nets

O **ResNet-18** permanece como default recomendado por seu excelente tradeoff entre desempenho, estabilidade e velocidade. Para geosteering realtime, **Mamba/S4** oferece a menor latência e **WaveNet** o maior campo receptivo causal.

---

*Catálogo Detalhado de Arquiteturas — Pipeline de Inversão Geofísica v5.0.15*
*Daniel Leal — Março 2026*
