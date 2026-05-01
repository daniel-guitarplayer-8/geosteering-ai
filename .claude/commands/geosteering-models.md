---
name: geosteering-models
description: |
  Sub-skill de arquiteturas de redes neurais do Geosteering AI v2.0. Catálogo completo das
  48 arquiteturas em 9 famílias com tiers de maturidade, compatibilidade causal, arquivo-fonte,
  referências bibliográficas e guia de seleção. Inclui SurrogateNet (TCN 127M + ModernTCN 204M),
  INN para UQ probabilística, ResNeXt e ModernTCN (adicionados 2026-04).
  Use para questões sobre arquiteturas, seleção de modelo, causal mode, tiers, famílias.
  Triggers: "arquitetura", "modelo", "ResNet", "TCN", "LSTM", "U-Net", "Transformer",
  "WaveNet", "Mamba", "FNO", "DeepONet", "INN", "ModernTCN", "ResNeXt", "SurrogateNet",
  "causal", "tier", "família", "geosteering model", "ModelRegistry", "build_model",
  "causal_compatible", "seq2seq", "realtime model".
  v2.0 | Última atualização: 2026-04-29 | Sub-skill da geosteering-v2
---

# Geosteering AI v2.0 — Catálogo de Arquiteturas (48 Modelos, 9 Famílias)

> **Sub-skill especializada** — Arquiteturas de redes neurais.
> Skill principal: `geosteering-v2` | Física: `geosteering-physics` | Código: `geosteering-code-v2` | Losses: `geosteering-losses`

---

## 1. Visão Geral

### 1.1 Formulação do Problema

```
Entrada: X ∈ ℝ^(batch × 600 × N_features)
  N_features = 5   (baseline P1: zobs + Re(Hxx) + Im(Hxx) + Re(Hzz) + Im(Hzz))
  N_features += 2K (com K famílias de geosinais P4 ativas)

Saída: Y ∈ ℝ^(batch × 600 × 2)
  Canal 0: rho_h (escala log10)
  Canal 1: rho_v (escala log10)

Regime: seq2seq (sequência→sequência, 600 pontos para 600 pontos)
         NUNCA classificação ou regressão escalar
```

**Invariante absoluta:** Toda arquitetura DEVE preservar a dimensão temporal.
A saída `(batch, N_MEDIDAS, 2)` é obrigatória para TODAS as 48 arquiteturas.

### 1.2 Tiers de Maturidade

| Tier | Significado | Ação recomendada |
|:----:|:------------|:-----------------|
| **1** | Validado, estável, resultado publicado | Usar em produção |
| **2** | Implementado, testado, performance conhecida | Testar antes de deploy |
| **3** | Experimental, pode precisar de tuning | Pesquisa e exploração |

### 1.3 Compatibilidade Causal

Arquiteturas **incompatíveis com modo causal** (use_causal_mode=True):

```python
_CAUSAL_INCOMPATIBLE = frozenset({
    "BiLSTM",           # usa backward pass — acesso a amostras futuras
    "CNN_BiLSTM_ED",    # encoder bidirecional
    # Todas 14 U-Nets — skip connections do encoder inteiro:
    "UNet_Base", "UNet_Attention",
    "UNet_ResNet18", "UNet_Attention_ResNet18",
    "UNet_ResNet34", "UNet_Attention_ResNet34",
    "UNet_ResNet50", "UNet_Attention_ResNet50",
    "UNet_ConvNeXt", "UNet_Attention_ConvNeXt",
    "UNet_Inception", "UNet_Attention_Inception",
    "UNet_EfficientNet", "UNet_Attention_EfficientNet",
    "N_BEATS", "N_HiTS",    # decomposição com look-ahead implícito
})
# Todas as demais 30 arquiteturas são causal-compatible
```

---

## 2. Catálogo por Família

### 2.1 CNN (8 arquiteturas)

| Arquitetura | Tier | Causal | Arquivo | Referência |
|:------------|:----:|:------:|:--------|:-----------|
| **ResNet_18** ★ | 1 | ✅ | `models/cnn.py` | He et al. (2016) CVPR |
| **ResNet_34** | 1 | ✅ | `models/cnn.py` | He et al. (2016) CVPR |
| **ResNet_50** | 2 | ✅ | `models/cnn.py` | He et al. (2016) CVPR |
| **ConvNeXt** | 1 | ✅ | `models/cnn.py` | Liu et al. (2022) CVPR |
| **InceptionNet** | 2 | ✅ | `models/cnn.py` | Szegedy et al. (2015) |
| **InceptionTime** | 2 | ✅ | `models/cnn.py` | Fawaz et al. (2020) |
| **CNN_1D** | 1 | ✅ | `models/cnn.py` | — (baseline simples) |
| **ResNeXt** *(novo 2026-04)* | 2 | ✅ | `models/cnn.py` | Xie et al. (2017) |

**ResNet_18** é o **default** — Tier 1, validado, bom tradeoff performance/complexidade.
**ResNeXt:** CNN com grouped convolutions (cardinality > 1); maior capacidade com mesmo FLOPs.

**Como funciona ResNet-18 para inversão 1D:**

```
Input (batch, None, N_FEATURES)
  ↓
Stem: Conv1D(64, 7) → BN → ReLU → Dropout
  ↓
Stage 1: 2× ResidualBlock(64)  — campo receptivo ~7m
Stage 2: 2× ResidualBlock(128) — campo receptivo ~15m
Stage 3: 2× ResidualBlock(256) — campo receptivo ~31m
Stage 4: 2× ResidualBlock(512) — campo receptivo ~63m
  ↓
Output: Conv1D(output_channels, 1, 'linear')  → (batch, N, 2)
```

Kernel size 3 → cada camada aumenta campo receptivo em ~3m (SPACING_METERS=1,0 m).
Após 4 estágios: ~63m de contexto — suficiente para inversão típica.

### 2.2 TCN (3 arquiteturas)

| Arquitetura | Tier | Causal | Arquivo | Referência |
|:------------|:----:|:------:|:--------|:-----------|
| **TCN** | 1 | ✅ | `models/tcn.py` | Bai et al. (2018) arXiv |
| **TCN_Advanced** | 2 | ✅ | `models/tcn.py` | — (extensão interna) |
| **ModernTCN** *(novo 2026-04)* | 2 | ✅ | `models/tcn.py` | Luo et al. (2024) |

**TCN:** Temporal Convolutional Network com dilated causal convolutions.
Dilation rates: [1, 2, 4, 8, ...] — campo receptivo exponencial sem custo quadrático.

**ModernTCN (2026-04):** DWConv largo + ConvFFN + LayerNorm.
- DWConv (depthwise-separable com kernel largo): captura dependências de longo alcance
- ConvFFN (Feed-Forward com convolução): substitui o projetor linear do Transformer
- Arquitetura pura-convolucional mas com propriedades de Transformer

### 2.3 RNN (2 arquiteturas)

| Arquitetura | Tier | Causal | Arquivo | Referência |
|:------------|:----:|:------:|:--------|:-----------|
| **LSTM** | 1 | ✅ | `models/rnn.py` | Hochreiter & Schmidhuber (1997) |
| **BiLSTM** | 2 | ❌ | `models/rnn.py` | — |

**LSTM:** Causal nativo — processa sequência forward, sem acesso a amostras futuras.
**BiLSTM:** Bidirecional — processa forward + backward simultaneamente. INCOMPATÍVEL com realtime.

### 2.4 Hybrid (3 arquiteturas)

| Arquitetura | Tier | Causal | Arquivo | Referência |
|:------------|:----:|:------:|:--------|:-----------|
| **CNN_LSTM** | 2 | ✅ | `models/hybrid.py` | — (combinação interna) |
| **CNN_BiLSTM_ED** | 2 | ❌ | `models/hybrid.py` | — |
| **ResNeXt_LSTM** *(novo 2026-04)* | 2 | ✅ | `models/hybrid.py` | — |

**CNN_LSTM:** Extrator CNN (campo receptivo local) + LSTM (contexto temporal global).
**ResNeXt_LSTM:** ResNeXt como extrator de features + LSTM para capturar tendências do perfil.

### 2.5 U-Net (14 arquiteturas)

Todas as 14 variantes U-Net: **Tier 2, CAUSAL INCOMPATÍVEL**.

```
UNet_Base                    UNet_Attention
UNet_ResNet18                UNet_Attention_ResNet18
UNet_ResNet34                UNet_Attention_ResNet34
UNet_ResNet50                UNet_Attention_ResNet50
UNet_ConvNeXt                UNet_Attention_ConvNeXt
UNet_Inception               UNet_Attention_Inception
UNet_EfficientNet            UNet_Attention_EfficientNet
```

Arquivo: `models/unet.py`

**Por que U-Net para inversão 1D?** A estrutura encoder-decoder com skip connections permite
que o modelo combine contexto global (decoder) com detalhes locais (skip connections do encoder).
Funciona como segmentação semântica 1D da sequência de resistividade.

**Por que são incompatíveis com modo causal?** Os skip connections conectam camadas do encoder
ao decoder — a ativação de um ponto Z usa informações de pontos anteriores E posteriores
(encoder processou toda a sequência antes do decoder começar).

### 2.6 Transformer (6 arquiteturas)

| Arquitetura | Tier | Causal | Arquivo | Referência |
|:------------|:----:|:------:|:--------|:-----------|
| **Transformer** | 2 | ✅ | `models/transformer.py` | Vaswani et al. (2017) NeurIPS |
| **Simple_TFT** | 2 | ✅ | `models/transformer.py` | — (simplificado) |
| **TFT** | 3 | ✅ | `models/transformer.py` | Lim et al. (2021) Int. J. Forecasting |
| **PatchTST** | 2 | ✅ | `models/transformer.py` | Nie et al. (2023) ICLR |
| **Autoformer** | 3 | ✅ | `models/transformer.py` | Wu et al. (2021) NeurIPS |
| **iTransformer** | 3 | ✅ | `models/transformer.py` | Liu et al. (2024) ICLR |

**PatchTST:** Divide a sequência em patches (janelas) e aplica self-attention entre patches.
Mais eficiente que Transformer padrão para séries longas — N_patches << N_seq.

### 2.7 Decomposition (2 arquiteturas)

| Arquitetura | Tier | Causal | Arquivo | Referência |
|:------------|:----:|:------:|:--------|:-----------|
| **N_BEATS** | 2 | ❌ | `models/decomposition.py` | Oreshkin et al. (2020) ICLR |
| **N_HiTS** | 2 | ❌ | `models/decomposition.py` | Challu et al. (2023) AAAI |

Ambos decompõem a série em componentes via stacks especializados (trend + seasonality).
**Incompatíveis com modo causal** porque os stacks processam toda a sequência.

### 2.8 Advanced (5 arquiteturas)

| Arquitetura | Tier | Causal | Arquivo | Referência |
|:------------|:----:|:------:|:--------|:-----------|
| **DNN** | 1 | ✅ | `models/advanced.py` | — (baseline ponto a ponto) |
| **FNO** | 3 | ✅ | `models/advanced.py` | Li et al. (2021) ICLR |
| **DeepONet** | 3 | ✅ | `models/advanced.py` | Lu et al. (2021) Nat. Mach. Intel. |
| **Geophysical_Attention** | 2 | ✅ | `models/advanced.py` | — (atenção física) |
| **INN** *(novo 2026-04)* | 3 | ✅ | `models/advanced.py` | Ardizzone et al. (2019) |

**FNO (Fourier Neural Operator):** Opera no domínio de frequência via FFT.
Captura padrões periódicos na sequência de resistividade — útil quando há variação cíclica de camadas.

**DeepONet:** Aprende operadores funcionais, não funções fixas.
Potencial para generalizar a diferentes espaçamentos e frequências sem re-treinar.

**INN (Invertible Neural Network):** Rede bijetiva — forward pass (EM→resistividade)
é exatamente invertível. Permite **amostragem do posterior probabilístico**:
```python
# Forward: EM → resistividade
rho = inn_model.forward(x_em)
# Inverse: resistividade → distribuição de EM compatíveis
x_em_samples = inn_model.inverse(rho, n_samples=100)
# Posterior sampling: múltiplas soluções de resistividade para o mesmo dado EM
rho_samples = inn_model.sample_posterior(x_em, n_samples=100)
```

**Vantagem do INN sobre MC Dropout:** 10× mais rápido para estimativa de incerteza.
**Desvantagem:** Tier 3 — requer tuning cuidadoso das loss functions forward + latent.

### 2.9 Geosteering (5 arquiteturas)

Todas projetadas para modo causal nativo — ideais para inferência em tempo real:

| Arquitetura | Tier | Causal | Arquivo | Referência |
|:------------|:----:|:------:|:--------|:-----------|
| **WaveNet** ★ | 1 | ✅ | `models/geosteering.py` | Oord et al. (2016) arXiv |
| **Causal_Transformer** | 2 | ✅ | `models/geosteering.py` | Vaswani + masking causal |
| **Informer** | 2 | ✅ | `models/geosteering.py` | Zhou et al. (2021) AAAI |
| **Mamba_S4** | 3 | ✅ | `models/geosteering.py` | Gu et al. (2022, 2024) |
| **Encoder_Forecaster** | 2 | ✅ | `models/geosteering.py` | — (arquitetura interna) |

**WaveNet:** Dilated causal convolutions + skip connections. Tier 1 para geosteering.
Receptive field exponencial: com 10 camadas de dilation [1,2,4,...,512] → 1.023 amostras de contexto.

**Mamba_S4:** State Space Model linear com hardware-aware scan. Tier 3 — experimental.
Promissor para substituir Transformers em sequências longas com latência menor.

---

## 3. SurrogateNet (Simulador Surrogate)

SurrogateNet aprende a mapear parâmetros de modelo geológico → tensor EM, atuando como
um **substituto** do simulador Fortran com latência ~1.000× menor:

| Variante | Parâmetros | Backbone | Arquivo | Status |
|:---------|:----------:|:--------:|:--------|:-------|
| **SurrogateNet v1** | 127M | TCN | `models/surrogate.py` | Implementado |
| **SurrogateNet v2** *(2026-04)* | 204M | ModernTCN | `models/surrogate.py` | Implementado |

**Entrada do SurrogateNet:** parâmetros do modelo geológico `(rho_h, rho_v, esp, z_pos, ...)`
**Saída:** tensor EM 9 componentes × 2 (Re+Im) = 18 canais por ponto de medição

**Status de treinamento:** Implementação completa; aguarda re-simulação Fortran multi-dip
para dataset de treinamento. SurrogateNet Modo C (tensor completo, 9 comp, 18 ch) aguarda dados multi-dip.

---

## 4. Guia de Seleção de Arquitetura

### 4.1 Por Uso

```
Pesquisa offline (sem restrição causal):
  → ResNet_18 ★  (default, Tier 1, testado, resultado publicado)
  → ConvNeXt     (alternativa CNN moderna)
  → UNet_ResNet18 (melhor para perfis com transições abruptas)
  → N_BEATS       (quando decomposição trend+seasonality é útil)

Geosteering em tempo real (causal obrigatório):
  → WaveNet ★    (default realtime, Tier 1, dilated causal, 1023 amostras de contexto)
  → TCN           (alternativa causal eficiente)
  → LSTM          (mais simples, bom baseline causal)
  → Causal_Transformer (se atenção longa for necessária)

Alta incerteza / quantificação probabilística (UQ):
  → INN           (posterior sampling 10× mais rápido que MC Dropout)
  → Qualquer arq + MC Dropout (config.mc_dropout_samples > 0)
  → Ensemble      (config.ensemble_size > 1)

Capacidade máxima (sem restrição de latência):
  → ResNet_50     (mais profundo)
  → SurrogateNet v2 (ModernTCN 204M — mas uso como surrogate, não inversão)

Baseline para comparação:
  → DNN           (ponto a ponto, sem contexto temporal)
  → CNN_1D        (CNN simples, 1 estágio)
```

### 4.2 Por Família — Trade-offs

| Família | Vantagem | Desvantagem | Melhor para |
|:--------|:---------|:------------|:-----------|
| CNN | Eficiente, campo receptivo controlado | Contexto limitado a kernel×dilação | Inversão offline padrão |
| TCN | Causal nativo, contexto exponencial | Menos flexível que RNN para estado | Geosteering |
| RNN | Estado oculto = memória contínua | LSTM gradiente vanishing em seq longas | Perfis com deriva lenta |
| Hybrid | Extração local + memória global | Mais parâmetros, treino mais lento | Quando CNN+LSTM ambos ajudam |
| U-Net | Contexto global + detalhes locais | Sem causal, custo de memória alto | Inversão offline de qualidade máxima |
| Transformer | Atenção global, interpretável | Quadrático em memória, lento | Perfis longos com dependências distantes |
| Decomposition | Explícito trend/seasonal | Sem causal, hipótese de decomposição | Dados com padrão periódico claro |
| Advanced | Operadores funcionais, UQ | Tier 3, requer tuning | Pesquisa, UQ probabilística |
| Geosteering | Projetado para realtime | Específico do domínio | Produção geosteering |

---

## 5. API do ModelRegistry

```python
from geosteering_ai.models.registry import ModelRegistry, is_causal_compatible

# Construção via registry (padrão obrigatório):
registry = ModelRegistry()
model = registry.build(config)           # constrói modelo completo

# Metadados:
info = registry.get_model_info("ResNet_18")
# info = {
#     "name": "ResNet_18",
#     "family": "CNN",
#     "tier": 1,
#     "causal_compatible": True,
#     "description": "ResNet-18 — skip connections, 8 blocos residuais, Tier 1 validado"
# }

# Validação de causalidade:
is_causal_compatible("WaveNet")    # → True
is_causal_compatible("BiLSTM")    # → False
is_causal_compatible("UNet_Base") # → False

# Lista de disponíveis por família:
cnn_models = [k for k, v in registry.families.items() if v == "CNN"]

# Validação automática: se use_causal_mode=True e model é incompatível → ValueError
config = PipelineConfig(model_type="BiLSTM", use_causal_mode=True)
model = registry.build(config)  # ValueError: "BiLSTM não é causal-compatible"
```

---

## 6. Curriculum Learning e Modo de Treinamento

### 6.1 3 Fases de Curriculum

```
Fase 1 — Clean (ep 0 → EPOCHS_NO_NOISE):
  noise_level = 0.0 — modelo aprende mapeamento limpo

Fase 2 — Ramp (ep EPOCHS_NO_NOISE → EPOCHS_NO_NOISE + NOISE_RAMP_EPOCHS):
  noise_level = NOISE_LEVEL_MAX × (ep - start) / NOISE_RAMP_EPOCHS
  modelo se adapta gradualmente ao ruído

Fase 3 — Stable (ep > EPOCHS_NO_NOISE + NOISE_RAMP_EPOCHS):
  noise_level = NOISE_LEVEL_MAX — treino no nível máximo de ruído
```

### 6.2 N-Stage Training

Para modelos mais robustos, o N-Stage itera o curriculum N vezes com variância crescente:
```python
config = PipelineConfig.nstage(n=3)
# Estágio 1: curriculum clean→ramp→stable com noise_max=σ₁
# Estágio 2: curriculum com noise_max=σ₂ > σ₁ (re-treino a partir do checkpoint)
# Estágio 3: curriculum com noise_max=σ₃ > σ₂
```

---

## 7. 5 Perspectivas — Impacto nas Arquiteturas

| P# | Mudança | Impacto na arquitetura |
|:--:|:--------|:-----------------------|
| P1 | Baseline | N_FEATURES = 5 (entrada padrão) |
| P2 | Ângulo como feature | N_FEATURES = 6 (theta na entrada) |
| P3 | Frequência como feature | N_FEATURES = 6–7 (log10(f) na entrada) |
| P4 | Geosinais | N_FEATURES = 5 + 2K (K famílias ativas) |
| P5 | DTB como target | output_channels = 4 ou 6 (rho_h, rho_v + DTB) |

Todas as 48 arquiteturas aceitam N_FEATURES variável (parâmetro dinâmico via config).
A saída `output_channels` é configurável via `config.output_channels`.

---

## 8. Referências Bibliográficas das Arquiteturas

| Arquitetura | Referência |
|:------------|:-----------|
| ResNet | He et al. (2016) "Deep Residual Learning for Image Recognition" CVPR |
| ConvNeXt | Liu et al. (2022) "A ConvNet for the 2020s" CVPR |
| ResNeXt | Xie et al. (2017) "Aggregated Residual Transformations" CVPR |
| TCN | Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional..." arXiv |
| ModernTCN | Luo et al. (2024) "ModernTCN: A Modern Pure Convolution Structure..." |
| LSTM | Hochreiter & Schmidhuber (1997) Neural Computation |
| Transformer | Vaswani et al. (2017) "Attention Is All You Need" NeurIPS |
| TFT | Lim et al. (2021) Int. J. Forecasting |
| PatchTST | Nie et al. (2023) "A Time Series Is Worth 64 Words" ICLR |
| Autoformer | Wu et al. (2021) "Autoformer: Decomposition Transformers" NeurIPS |
| iTransformer | Liu et al. (2024) "iTransformer: Inverted Transformers" ICLR |
| FNO | Li et al. (2021) "Fourier Neural Operator" ICLR |
| DeepONet | Lu et al. (2021) "Learning nonlinear operators via DeepONet" Nat. Mach. Intel. |
| WaveNet | Oord et al. (2016) "WaveNet: A Generative Model for Raw Audio" arXiv |
| Mamba/S4 | Gu et al. (2022) ICLR 2022; Gu & Dao (2024) arXiv |
| N-BEATS | Oreshkin et al. (2020) "N-BEATS: Neural basis expansion..." ICLR |
| N-HiTS | Challu et al. (2023) "N-HiTS: Neural Hierarchical..." AAAI |
| INN | Ardizzone et al. (2019) "Analyzing Inverse Problems..." ICLR |

---

*Sub-skill de arquiteturas — Geosteering AI v2.0 | Última atualização: 2026-04-29*
