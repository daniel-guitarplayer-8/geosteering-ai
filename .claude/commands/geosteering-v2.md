---
name: geosteering-v2
description: |
  Skill definitiva para o Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning.
  Cobre TRES dominios: (1) Fisica de geofisica/petrofisica/EM para LWD/geosteering,
  (2) Codigo v2.0 com PipelineConfig/Factory/DataPipeline (NUNCA globals/print),
  (3) Deep Learning aplicado a geociencias (44 arquiteturas, 26 losses, 43 noise types).
  Triggers: "inversao", "resistividade", "geosteering", "EM", "LWD", "tensor magnetico",
  "anisotropia", "TIV", "skin depth", "decoupling", "Feature View", "geosinal", "Picasso",
  "DTB", "noise", "curriculum", "arquitetura", "loss", "PipelineConfig", "DataPipeline",
  "petrofisica", "perfilagem", "porosidade", "permeabilidade", "saturacao", "fator de formacao".
  v2.0 | TensorFlow/Keras (PyTorch PROIBIDO) | PipelineConfig | Factory Pattern
  44 arquiteturas (39 standard + 5 geosteering), 26 losses, 43 noise types, 5 perspectivas P1-P5.
---

# Geosteering AI v2.0 — Skill de Dominio Fisico e Desenvolvimento

## 1. Identidade do Projeto

| Atributo | Valor |
|:---------|:------|
| **Projeto** | Inversao 1D de Resistividade via Deep Learning para Geosteering |
| **Versao** | **v2.0** (arquitetura de software modular) |
| **Autor** | Daniel Leal |
| **Framework** | TensorFlow 2.x / Keras **EXCLUSIVO** (PyTorch PROIBIDO) |
| **Ambiente** | VSCode + Claude Code (dev) · GitHub CI (teste) · Google Colab Pro+ GPU (exec) |
| **Pacote** | `geosteering_ai/` (pip installable) |
| **Config** | `PipelineConfig` dataclass (NUNCA `globals().get()`) |
| **Referencia** | `docs/ARCHITECTURE_v2.md` (documento completo da arquitetura) |

---

## 2. Fontes Autoritativas e Ordem de Consulta

| Prioridade | Fonte | Quando Consultar |
|:----------:|:------|:----------------|
| 1a | `docs/physics/errata_valores.md` | Qualquer codigo com constantes fisicas |
| 2a | `geosteering_ai/config.py` | PipelineConfig, defaults, validacoes |
| 3a | `docs/ARCHITECTURE_v2.md` | Decisoes arquiteturais |
| 4a | `CLAUDE.md` | Regras, proibicoes, code patterns |
| 5a | `docs/physics/perspectivas.md` | Features P2-P5, interacoes |
| 6a | `docs/reference/losses_catalog.md` | 26 losses, decisao de escolha |
| 7a | `docs/reference/noise_catalog.md` | 43 noise types, curriculum |
| 8a | `docs/reference/arquiteturas_resumo.md` | 44 arquiteturas, tiers, causal compat |
| 9a | `docs/physics/onboarding.md` | Onboarding, fisica basica |
| 10a | `docs/MIGRATION_GUIDE.md` | Mapeamento legado C0-C47 → v2.0 |

**Regra:** Ao gerar codigo → consultar ERRATA + config.py primeiro.
Ao responder perguntas de fisica → consultar docs/physics/ primeiro.
Em caso de duvida → CLAUDE.md prevalece.

---

## 3. Proibicoes Absolutas (Fail-Fast)

Qualquer violacao dessas regras invalida o codigo gerado:

| Proibicao | Correto | Errado |
|:----------|:--------|:-------|
| PyTorch em qualquer parte | TensorFlow/Keras | `import torch` |
| `globals().get()` | `config.param` via PipelineConfig | `globals().get("FLAG", default)` |
| `print()` | `logger.info()` via logging | `print("resultado")` |
| `FREQUENCY_HZ = 2.0` | `20000.0` (20 kHz) | `2.0` |
| `SPACING_METERS = 1000.0` | `1.0` (1 metro) | `1000.0` |
| `SEQUENCE_LENGTH = 601` | `600` | `601` |
| `TARGET_SCALING = "log"` | `"log10"` | `"log"` |
| `INPUT_FEATURES = [0,3,4,7,8]` | `[1,4,5,20,21]` (22-col) | `[0,3,4,7,8]` |
| `OUTPUT_TARGETS = [1,2]` | `[2,3]` (22-col) | `[1,2]` |
| `eps = 1e-30` | `1e-12` (float32 safe) | `1e-30` |
| Split por amostra | `split_by_model=True` (P1) | Split random |
| Scaler fit em dados ruidosos | Fit em dados LIMPOS | Fit apos noise |
| Noise offline com FV/GS | On-the-fly dentro de tf.data.map | Noise antes de split |
| Funcao sem config param | `def f(config: PipelineConfig)` | `def f():` com globals |

---

## 4. Fisica Fundamental — Inversao EM 1D para LWD

### 4.1 O Problema Fisico

**Objetivo:** Traduzir medidas eletromagneticas (EM) feitas por ferramentas dentro de pocos de petroleo
em mapas de resistividade eletrica das rochas ao redor do poco usando redes neurais profundas.

**Analogia medica:**
- Ondas EM (20 kHz) → equivalente a raios-X
- Rochas (folhelho, arenito, carbonato) → equivalente ao corpo humano
- Ferramenta LWD no poco → equivalente ao tomografo CT
- Rede neural profunda → equivalente a reconstrucao de imagem
- Saida: rho_h (horizontal) e rho_v (vertical) → equivalente a densidade de tecido

### 4.2 Resistividade (rho, Ohm.m)

A resistividade eletrica eh a propriedade petrofisica primaria para diferenciacao de fluidos:

| Material | Resistividade (Ohm.m) | Significado |
|:---------|:---------------------:|:------------|
| Agua salgada (formacao) | 0.1 - 5 | Condutiva (ions dissolvidos) |
| Folhelho (shale) | 1 - 10 | Moderadamente condutivo |
| Arenito com agua | 5 - 50 | Intermediario |
| Arenito com oleo | 50 - 500 | Resistivo (hidrocarboneto) |
| Carbonato compacto | 100 - 10000 | Muito resistivo |
| Sal | >10000 | Extremamente resistivo |

**Importancia para geosteering:** Contraste de resistividade indica fronteiras entre
camadas geologicas (reservatorio vs selante) — informacao critica para manter o poco
dentro da zona produtora.

### 4.3 Anisotropia TIV (Transversalmente Isotropica Vertical)

Rochas sedimentares conduzem corrente eletrica de forma diferente nas direcoes
horizontal e vertical, como a fibra da madeira:

```
rho_h (horizontal) — resistividade ao longo das camadas
rho_v (vertical)   — resistividade perpendicular as camadas
SEMPRE rho_v >= rho_h em TIV (ratio tipico 1.5-10x)
```

**Consequencia no pipeline:** O modelo SEMPRE produz 2 saidas por ponto: (rho_h, rho_v).
Output shape: `(batch, N_MEDIDAS, 2)` onde `N_MEDIDAS = 600`.

### 4.4 Ferramenta LWD (Logging While Drilling)

A ferramenta eh um arranjo de antenas montado no BHA (Bottom Hole Assembly):

```
Transmissor → emite onda EM a 20 kHz
Receptores → medem tensor magnetico 3x3 (9 componentes complexas)

Tensor H (3x3):
    | Hxx  Hxy  Hxz |
H = | Hyx  Hyy  Hyz |    (cada componente: Re + Im = 18 valores reais)
    | Hzx  Hzy  Hzz |
```

**Antenas tilted (inclinadas):** Permitem medidas direcionais (azimutais).
Componentes-chave do pipeline:
- `Hxx` (planar) — sensivel a camadas horizontais
- `Hzz` (axial) — sensivel a bulk resistivity
- `Hxz`, `Hzx` (cross) — sensiveis a fronteiras e dip

### 4.5 Formato de Dados 22 Colunas

O dataset `.dat` (Fortran binary) tem 22 colunas por ponto de medicao:

```
Col  0: meds          — indice de medicao (metadata, NUNCA feature)
Col  1: zobs          — profundidade observada (metros) → INPUT_FEATURE
Col  2: res_h         — resistividade horizontal (Ohm.m) → OUTPUT_TARGET
Col  3: res_v         — resistividade vertical (Ohm.m) → OUTPUT_TARGET
Col  4: Re(Hxx)       — parte real de Hxx → INPUT_FEATURE
Col  5: Im(Hxx)       — parte imaginaria de Hxx → INPUT_FEATURE
Col  6: Re(Hxy)       — parte real de Hxy
Col  7: Im(Hxy)       — parte imaginaria de Hxy
...
Col 20: Re(Hzz)       — parte real de Hzz → INPUT_FEATURE
Col 21: Im(Hzz)       — parte imaginaria de Hzz → INPUT_FEATURE
```

**INPUT_FEATURES = [1, 4, 5, 20, 21]** → zobs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)
**OUTPUT_TARGETS = [2, 3]** → res_h, res_v

### 4.6 Skin Depth (Profundidade de Penetracao)

A onda EM penetra no meio ate uma profundidade delta:

```
delta = sqrt(2 / (omega * mu * sigma))

Para f = 20 kHz, rho = 10 Ohm.m:
  omega = 2*pi*20000 = 125663.7 rad/s
  mu = 4*pi*1e-7 H/m
  sigma = 1/10 = 0.1 S/m
  delta = sqrt(2 / (125663.7 * 4*pi*1e-7 * 0.1)) = ~11.3 m
```

**Consequencia:** DOI (Depth of Investigation) = ~1.5-3.0 m para configuracao tipica.
Frequencias menores → maior penetracao, menor resolucao vertical.
Frequencias maiores → menor penetracao, maior resolucao vertical.

### 4.7 Decoupling EM (Remocao do Acoplamento Direto)

O sinal EM medido inclui o acoplamento direto Tx-Rx (campo no espaco livre),
que deve ser removido para isolar a resposta da formacao:

```python
# Constantes de decoupling (L = SPACING_METERS = 1.0 m)
ACp = -1 / (4 * pi * L**3)   # ≈ -0.079577 (planar: Hxx, Hyy)
ACx = +1 / (2 * pi * L**3)   # ≈ +0.159155 (axial: Hzz)

# Aplicacao:
H_formation_xx = H_measured_xx - ACp   # Remove acoplamento planar
H_formation_zz = H_measured_zz - ACx   # Remove acoplamento axial
```

### 4.8 Feature Views (6 Transformacoes de Componentes EM)

Transformacoes sobre as 4 componentes EM [Re(H1), Im(H1), Re(H2), Im(H2)]:

| View | Canal 0 | Canal 1 | Canal 2 | Canal 3 |
|:-----|:--------|:--------|:--------|:--------|
| **identity** | Re(H1) | Im(H1) | Re(H2) | Im(H2) |
| **raw** | Re(H1) | Im(H1) | Re(H2) | Im(H2) |
| **H1_logH2** | Re(H1) | Im(H1) | log10\|H2\| | phi(H2) |
| **logH1_logH2** | log10\|H1\| | phi(H1) | log10\|H2\| | phi(H2) |
| **IMH1_IMH2_razao** | Im(H1) | Im(H2) | \|H1\|/\|H2\| | phi(H1)-phi(H2) |
| **IMH1_IMH2_lograzao** | Im(H1) | Im(H2) | log10(\|H1\|/\|H2\|) | phi(H1)-phi(H2) |

H1 = Hxx (planar), H2 = Hzz (axial).
SEMPRE log10 (NUNCA ln). Versoes numpy e TF CONSISTENTES.

### 4.9 Geosinais (P4) — 5 Familias

Razoes compensadas em ganho entre componentes do tensor EM:

| Familia | Formula | Detecta | Canais |
|:--------|:--------|:--------|:------:|
| **USD** | (ZZ+XZ)/(ZZ-XZ) × (ZZ-ZX)/(ZZ+ZX) | Fronteiras de camada | att + phase |
| **UAD** | (ZZ+XZ)/(ZZ-XZ) × (ZZ+ZX)/(ZZ-ZX) | Dip/anisotropia | att + phase |
| **UHR** | -2·ZZ/(XX+YY) | Bulk resistivity | att + phase |
| **UHA** | XX/YY | Anisotropia | att + phase |
| **U3DF** | (ZZ+YZ)/(ZZ-YZ) × (ZZ-ZY)/(ZZ+ZY) | Indicador 3D | att + phase |

**Cada familia produz 2 canais:** atenuacao (dB) e deslocamento de fase (graus).

**Dependencias EM por familia:**
```python
FAMILY_EM_DEPS = {
    "USD": {"ZZ", "XZ", "ZX"},
    "UAD": {"ZZ", "XZ", "ZX"},
    "UHR": {"ZZ", "XX", "YY"},
    "UHA": {"XX", "YY"},
    "U3DF": {"ZZ", "YZ", "ZY"},
}
# Hxx e Hzz sao SEMPRE auto-incluidos no baseline
```

### 4.10 Picasso Plots e DTB (P5)

**Picasso Plot:** Mapa 2D de DOD (Depth of Detection) em funcao de (Rt1, Rt2):
- Eixo X: Resistividade da camada 1
- Eixo Y: Resistividade da camada 2
- Diagonal Rt1=Rt2: "zona cega" (sem contraste)
- DOD tipico do pipeline: ~1.5-3.0 metros

**DTB (Distance to Boundary):** Distancia do poco ate a fronteira geologica mais proxima.
Pode ser adicionado como target adicional (output_channels: 2 → 4 ou 6).

---

## 5. Petrofisica Fundamental

### 5.1 Relacoes Petrofisicas Basicas

**Lei de Archie (1942):** Relacao entre resistividade e porosidade/saturacao:
```
Rt = a * Rw / (phi^m * Sw^n)

Onde:
  Rt = resistividade da formacao (Ohm.m)
  Rw = resistividade da agua de formacao (Ohm.m)
  phi = porosidade (fracao)
  Sw = saturacao de agua (fracao)
  a = constante de tortuosidade (~1.0)
  m = expoente de cimentacao (~2.0)
  n = expoente de saturacao (~2.0)
```

**Implicacao:** Conhecendo Rt (saida do modelo) e Rw (dado do poco),
pode-se estimar phi e Sw — propriedades criticas para avaliacao de reservatorio.

### 5.2 Fator de Formacao

```
F = a / phi^m = Rt / Rw (quando Sw = 1.0, formacao 100% saturada com agua)
```

### 5.3 Classificacao de Litologia por Resistividade

| Litologia | rho_h tipico | rho_v tipico | rho_v/rho_h |
|:----------|:-------------|:-------------|:------------|
| Folhelho | 1-10 | 3-30 | 3-5 |
| Arenito limpo | 20-500 | 20-500 | ~1 |
| Arenito argiloso | 5-50 | 10-100 | 2-5 |
| Carbonato | 100-10000 | 100-10000 | ~1 |
| Evaporito (sal) | >10000 | >10000 | ~1 |

---

## 6. Geosteering — Operacao em Tempo Real

### 6.1 O que eh Geosteering

Ajuste da trajetoria do poco EM TEMPO REAL usando medidas LWD para manter
o poco dentro da zona produtora (reservatorio), evitando folhelhos e contatos de fluidos.

### 6.2 Cadeia Completa de Geosteering

```
1. Downhole:  LWD/MWD transmite ondas EM → receptores medem tensor H
2. Telemetria: Mud pulse / EM / wired drill pipe → superficie
3. Aquisicao:  WITSML/OPC data streams (superficie)
4. Inversao:   ★ ESTE PIPELINE ★ — EM → (rho_h, rho_v) via DL
5. Interpretacao: Deteccao de camadas + calculo DTB
6. HMI:        Interface para geologo/engenheiro de poco
7. Controle:   Comandos direcionais ao BHA
```

**O pipeline eh o nucleo de inversao (etapa 4):** recebe dados EM brutos
e produz perfis de resistividade em milissegundos (<31 ms), permitindo
decisoes em tempo real.

### 6.3 Constraintes de Tempo Real

| Constrainte | Requisito | Status Pipeline |
|:------------|:----------|:----------------|
| Latencia | <10 ms/amostra | <31 ms (inversao) |
| Causalidade | Sem dados futuros | 27/44 arqs causal-compatible |
| Sequencialidade | Dados a cada ~5 s | Sliding window pronto |
| Robustez | Ruido real, dropouts | 43 noise types + curriculum |
| Incerteza | Confianca quantificada | MC dropout + ensemble |
| Disponibilidade | 24/7 durante perfuracao | InferencePipeline (P6) serializado |

### 6.4 Modo Dual (Offline vs Realtime)

| Aspecto | Offline (Pesquisa) | Realtime (Geosteering) |
|:--------|:---|:---|
| Dados | Perfil completo do poco | Sliding window (600 amostras) |
| Rede | Acausal (futuro+passado) | Causal (somente passado) |
| Output | (batch, 600, 2) | (1, W, 2-6) + incerteza |
| Latencia | Nao critica | <1 s por amostra |
| Padding | `"same"` | `"causal"` |
| Uso | Pos-processamento | Suporte a decisao em tempo real |

---

## 7. Deep Learning Aplicado a Geofisica

### 7.1 Formulacao do Problema

```
Entrada: X ∈ R^(batch × 600 × N_features)
  N_features = 5 (baseline P1: zobs + 4 EM)
  N_features += 2*K (com K familias de geosinais ativos)

Saida: Y ∈ R^(batch × 600 × 2)
  Canal 0: rho_h (log10 scale)
  Canal 1: rho_v (log10 scale)
```

**Regra critica:** Toda arquitetura DEVE preservar a dimensao temporal.
Isso eh seq2seq (sequencia→sequencia), NAO classificacao.

### 7.2 Cadeia de Dados Fisicamente Correta

```
train_raw → noise(A/m) → FV_tf(noisy) → GS_tf(noisy) → scale → modelo
                                              |
                                    GS veem ruido ✓ (fidelidade LWD)
```

**Regras:**
1. Scaler fitado em dados LIMPOS (FV+GS aplicados em clean, temporario)
2. Val/test transformados offline (sem noise, FV+GS+scale)
3. Train permanece raw para noise on-the-fly via tf.data.map()
4. NUNCA aplicar noise offline quando FV ou GS estao ativos

### 7.3 Curriculum Learning (3 Fases)

```
Fase 1 — Clean (ep 0 a EPOCHS_NO_NOISE):
  noise_level = 0.0
  Modelo aprende mapeamento limpo

Fase 2 — Ramp (ep EPOCHS_NO_NOISE a EPOCHS_NO_NOISE + NOISE_RAMP_EPOCHS):
  noise_level = NOISE_LEVEL_MAX × (ep - start) / NOISE_RAMP_EPOCHS
  Modelo se adapta gradualmente ao ruido

Fase 3 — Stable (ep > EPOCHS_NO_NOISE + NOISE_RAMP_EPOCHS):
  noise_level = NOISE_LEVEL_MAX
  Modelo treina no nivel maximo de ruido
```

### 7.4 44 Arquiteturas (9 Familias)

| Familia | Arquiteturas | Causal |
|:--------|:-------------|:-------|
| **CNN** | ResNet-18★, ResNet-34, ResNet-50, ConvNeXt, InceptionNet, InceptionTime, CNN_1D | Adaptavel |
| **TCN** | TCN, TCN_Advanced | Nativo |
| **RNN** | LSTM, BiLSTM | Nativo / Incomp |
| **Hibrido** | CNN_LSTM, CNN_BiLSTM_ED | Parcial |
| **U-Net** | 14 variantes (Base + Attention + ResNet + ConvNeXt + Inception + EfficientNet) | Incompativel |
| **Transformer** | Transformer, Simple_TFT, TFT, PatchTST, Autoformer, iTransformer | Adaptavel |
| **Decomposicao** | N-BEATS, N-HiTS | Adaptavel |
| **Operador** | FNO, DeepONet | Adaptavel / Incomp |
| **Geosteering** | WaveNet, Causal_Transformer, Informer, Mamba_S4, Encoder_Forecaster | Nativo |

**Default:** ResNet-18★ (Tier 1, validado, estavel, bom tradeoff).

### 7.5 26 Loss Functions

| Categoria | Count | Exemplos |
|:----------|:-----:|:---------|
| **Genericas** | 13 | MSE, RMSE, MAE, MBE, Huber, Log-Cosh |
| **Geofisicas** | 4 | log_scale_aware, adaptive_log_scale, robust_log_scale |
| **Geosteering** | 2 | probabilistic_nll, look_ahead_weighted |
| **Avancadas v5.0.15** | 6 | DILATE, Encoder-Decoder, Multi-Task, Sobolev H1, Cross-Gradient, Spectral |
| **Hibrida** | 1 | morales_physics_hybrid |

**Gangorra:** Losses base + look_ahead + DTB + PINNs combinados com pesos aprendidos.

### 7.6 Target Scaling (8 Metodos)

| Metodo | Transformacao | Dominio |
|:-------|:-------------|:--------|
| **log10** (default) | log10(rho) | Comprime escala logaritmica |
| log | ln(rho) | Compressao natural |
| sqrt | sqrt(rho) | Compressao leve |
| power | rho^0.25 | Compressao intermediaria |
| minmax | (x-min)/(max-min) | Normaliza [0,1] |
| standard | (x-mu)/sigma | Normaliza N(0,1) |
| robust | (x-median)/IQR | Robusto a outliers |
| none | rho | Sem transformacao |

### 7.7 PINNs (Physics-Informed Neural Networks)

Integracao de constraintes fisicas na funcao de perda:

```python
L_total = L_data + lambda_physics * L_physics

L_physics pode incluir:
  - Residuo da equacao de difusao EM
  - Constrainte de suavidade (Sobolev H1)
  - Consistencia rho_h/rho_v (cross-gradient)
  - Hard constraint layer (Morales 2025)
```

**Referencia principal:** Morales et al. (2025) — PINN para inversao EM triaxial
com quantificacao de incerteza e constrainte de anisotropia.

---

## 8. Padroes de Codigo v2.0 (OBRIGATORIOS)

### 8.1 PipelineConfig como Parametro (NUNCA globals)

```python
# CORRETO:
def build_model(config: PipelineConfig) -> tf.keras.Model:
    n_features = config.n_features
    output_channels = config.output_channels
    ...

# PROIBIDO:
def build_model():
    model_type = globals().get("MODEL_TYPE", "ResNet_18")  # NAO!
```

### 8.2 Factory Pattern para Componentes

```python
# CORRETO:
model = ModelRegistry().build(config)
loss_fn = LossFactory.get(config)
callbacks = build_callbacks(config, model, noise_var)

# PROIBIDO:
if MODEL_TYPE == "ResNet_18":
    model = build_resnet18(...)
elif MODEL_TYPE == "CNN_1D":
    ...
```

### 8.3 DataPipeline com Cadeia Explicita

```python
# CORRETO:
pipeline = DataPipeline(config)
data = pipeline.prepare(dataset_path)        # raw → split → fit_scaler
map_fn = pipeline.build_train_map_fn(noise_var)  # noise → FV → GS → scale

# PROIBIDO:
x_train = apply_feature_view(x_train, view)  # offline, sem noise
x_train = apply_noise(x_train)               # noise apos FV → fisicamente errado
```

### 8.4 Presets YAML para Reprodutibilidade

```python
config = PipelineConfig.from_yaml("configs/robusto.yaml")
# OU presets de classe:
config = PipelineConfig.baseline()        # P1: sem noise, sem GS
config = PipelineConfig.robusto()         # E-Robusto S21 defaults
config = PipelineConfig.nstage(n=3)       # N-Stage training
config = PipelineConfig.geosinais_p4()    # P4 com geosinais
config = PipelineConfig.realtime()        # Geosteering causal mode
```

### 8.5 Logging (NUNCA print)

```python
import logging
logger = logging.getLogger(__name__)

# CORRETO:
logger.info("Modelo compilado: %s, params: %d", config.model_type, model.count_params())
logger.warning("Scaler nao fitado — usando defaults")

# PROIBIDO:
print(f"Modelo: {MODEL_TYPE}")
```

---

## 9. Estrutura do Pacote v2.0

```
geosteering_ai/
├── __init__.py
├── config.py              ← PipelineConfig dataclass (ponto unico de verdade)
├── data/
│   ├── __init__.py
│   ├── loading.py         ← parse .out, load .dat (22-col), segregate by angle
│   ├── splitting.py       ← split by geological model (P1), stratified
│   ├── feature_views.py   ← 6 Feature Views (numpy + TF)
│   ├── geosignals.py      ← 5 familias GS (numpy + TF)
│   ├── scaling.py         ← 8 scalers + 8 target scalings + per-group [P3]
│   └── pipeline.py        ← DataPipeline: raw → split → fit → tf.data.map
├── noise/
│   ├── __init__.py
│   ├── functions.py       ← apply_raw_em_noise() (43 noise types)
│   └── curriculum.py      ← CurriculumSchedule: 3-phase + N-Stage
├── models/
│   ├── __init__.py
│   ├── blocks.py          ← 23 blocos Keras reutilizaveis
│   ├── cnn.py             ← ResNet-18/34/50, ConvNeXt, Inception, CNN_1D
│   ├── tcn.py             ← TCN, TCN_Advanced
│   ├── rnn.py             ← LSTM, BiLSTM
│   ├── hybrid.py          ← CNN_LSTM, CNN_BiLSTM_ED
│   ├── unet.py            ← 14 variantes U-Net
│   ├── transformer.py     ← 6 transformers
│   ├── decomposition.py   ← N-BEATS, N-HiTS
│   ├── advanced.py        ← DNN, FNO, DeepONet, Geophysical_Attention
│   ├── geosteering.py     ← WaveNet, Causal_Transformer, Informer, Mamba_S4, Encoder_Forecaster
│   └── registry.py        ← ModelRegistry: 44 entradas + build()
├── losses/
│   ├── __init__.py
│   ├── catalog.py         ← 26 loss functions
│   └── factory.py         ← LossFactory.get(config)
├── training/
│   ├── __init__.py
│   ├── loop.py            ← TrainingLoop.run()
│   ├── callbacks.py       ← build_callbacks(config, model, noise_var)
│   ├── metrics.py         ← R2Score, PerComponentMetric, AnisotropyRatioError
│   ├── nstage.py          ← N-Stage training (multi-stage noise adaptation)
│   └── optuna_hpo.py      ← HPO com Optuna
├── inference/
│   ├── __init__.py
│   ├── pipeline.py        ← InferencePipeline (P6): FV+GS+scalers, joblib save/load
│   ├── realtime.py        ← Sliding window inference
│   └── export.py          ← SavedModel, TFLite, ONNX
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py         ← Metricas numpy (MSE, RMSE, R2, MBE)
│   └── comparison.py      ← Comparacao entre modelos
├── visualization/
│   ├── __init__.py
│   ├── holdout.py         ← Plots holdout clean+noisy
│   ├── picasso.py         ← Picasso DOD plots
│   ├── eda.py             ← EDA plots
│   └── realtime.py        ← Monitor de inferencia em tempo real
└── utils/
    ├── __init__.py
    ├── logger.py          ← ColoredFormatter, setup_logger
    ├── timer.py           ← timer_decorator, ProgressTracker
    ├── validation.py      ← ValidationTracker
    ├── formatting.py      ← print_header, format_params
    ├── system.py          ← is_colab(), has_gpu()
    └── io.py              ← File I/O helpers
```

---

## 10. 5 Perspectivas (P1-P5) — Features Adicionais

| P# | Nome | Features Adicionadas | Versao |
|:--:|:-----|:--------------------|:------:|
| P1 | Baseline | z + 4 EM (5 features) | v5.0.1 |
| P2 | Angulo | theta como feature (6 features) | v5.0.8 |
| P3 | Frequencia | f como feature (6-7 features), log10 norm | v5.0.12 |
| P4 | Geosinais | USD, UAD, UHR, UHA, U3DF (2 canais/familia) | v5.0.13 |
| P5 | Picasso/DTB | DOD validation + DTB como target (6 output channels) | v5.0.15 |

**Cadeia on-the-fly com P4/P5:**
```
raw Re/Im → noise → FV → GS → Scale → Model
                                 ↑
                       GS veem o ruido ✓ (fisicamente correto)
```

---

## 11. Referencias Bibliograficas Principais

### 11.1 Livros-Texto

| Referencia | Topico |
|:-----------|:-------|
| Ellis & Singer (2008) "Well Logging for Earth Scientists" | Perfilagem de pocos, fisica EM, interpretacao |
| Misra, Li & He (2019) "Machine Learning for Subsurface Characterization" | ML aplicado a geofisica e petrofisica |
| "Principles and Applications of Well Logging" | Fundamentos de perfilagem |
| "Dive Into Deep Learning" | Deep Learning fundamental |

### 11.2 Artigos Cientificos — Inversao EM e Deep Learning

| Referencia | Contribuicao |
|:-----------|:-------------|
| Morales et al. (2025) "Anisotropic resistivity estimation... PINN" | PINNs para inversao EM triaxial com UQ |
| Noh & Verdin (2022) Petrophysics | Estimativa petrofisica com DL |
| Wang et al. (2018) J. Geophys. Eng. 15:2339 | Analise de sensibilidade ARM + inversao Gauss-Newton |
| Constable et al. (2016) Petrophysics | Look-ahead EM (EMLA) — deteccao 5-30m |

### 11.3 Artigos — Geosteering e Look-Ahead

| Referencia | Contribuicao |
|:-----------|:-------------|
| Guoyu et al. (2025) Energies 18:3014 | Geometric factors para look-ahead EM |
| Benchmark Look-Ahead Models | Casos de teste padrao (vertical, deviated, thin-bed) |
| GeoSphere HD (Schlumberger) | Ferramenta comercial de referencia |

### 11.4 Artigos — Metodos e Algoritmos

| Referencia | Contribuicao |
|:-----------|:-------------|
| gxag017 (2026) | Modified Propagator Coefficient para modelagem 1D rapida |
| Bai et al. (2022) arXiv:2210.09060 | Tutorial PINNs para mecanica computacional |

### 11.5 Arquiteturas de Redes Neurais

| Arquitetura | Referencia |
|:------------|:-----------|
| ResNet | He et al. (2016) CVPR |
| ConvNeXt | Liu et al. (2022) CVPR |
| Transformer | Vaswani et al. (2017) NeurIPS |
| TFT | Lim et al. (2021) Int. J. Forecasting |
| PatchTST | Nie et al. (2023) ICLR |
| FNO | Li et al. (2021) ICLR |
| DeepONet | Lu et al. (2021) Nature Machine Intelligence |
| WaveNet | Oord et al. (2016) arXiv |
| Mamba/S4 | Gu et al. (2022, 2024) |
| N-BEATS | Oreshkin et al. (2020) ICLR |

### 11.6 Ferramentas Comerciais LWD

| Ferramenta | Fabricante | Caracteristica |
|:-----------|:-----------|:---------------|
| GeoSphere HD | Schlumberger | DOI: 1.5-3.0m, multi-frequencia |
| EarthStar/BrightStar | Halliburton | Look-around + look-ahead |
| Visitrak | Baker Hughes | Geosteering service |
| ACPR | COSL | Azimuthal propagation resistivity |

---

## 12. Comparacao com Pipeline Legado (v5.0.15)

| Aspecto | Legado (v5.0.15) | v2.0 |
|:--------|:----------------|:-----|
| Formato | 74 celulas Colab (C0-C73) | Pacote Python pip installable |
| Config | ~1.185 FLAGS via `globals().get()` | `PipelineConfig` dataclass |
| Componentes | if/elif imperativo | Registry + Factory Pattern |
| Dados | Disperso em C19-C26 | `DataPipeline` com cadeia unica |
| Testes | Manual | pytest automatizado + CI |
| Output | `print()` direto | `logging` estruturado |
| Reprodutibilidade | FLAGS no notebook | YAML + tag GitHub + seed |
| Legado | `.claude/commands/geosteering-v5015.md` | Esta skill (`geosteering-v2.md`) |

**Regra de migracao:** Ao consultar codigo legado, SEMPRE traduzir para padroes v2.0.
O legado serve APENAS como referencia de logica de dominio, NAO de padrao de codigo.

---

## 13. Workflow de Desenvolvimento

```
ANTES de implementar:
  1. Consultar docs/ARCHITECTURE_v2.md para decisoes arquiteturais
  2. Consultar docs/physics/ para questoes fisicas
  3. Planejar com superpowers:writing-plans se 3+ etapas

DURANTE implementacao:
  4. Codigo SEMPRE recebe config: PipelineConfig
  5. Factory/Registry para componentes
  6. logging (NUNCA print)
  7. Consultar context7-plugin para docs TF/Keras

APOS implementar:
  8. pytest tests/ -v --tb=short
  9. Revisar com code-reviewer agent
  10. Verificar com superpowers:verification-before-completion
```

---

## 14. Checklist Rapido para Geracao de Codigo

Antes de entregar QUALQUER codigo, verificar:

- [ ] Usa `config: PipelineConfig` como parametro (nao globals)
- [ ] Usa `logging` (nao print)
- [ ] Constantes fisicas corretas (20000.0, 1.0, 600, [1,4,5,20,21], [2,3])
- [ ] `eps = 1e-12` (nao 1e-30)
- [ ] Target scaling = "log10" (nao "log")
- [ ] Split by model (nao por amostra)
- [ ] Scaler fit em dados limpos
- [ ] Noise on-the-fly (nao offline) quando FV/GS ativos
- [ ] Output preserva dimensao temporal: (batch, N, channels)
- [ ] TensorFlow/Keras (nao PyTorch)
- [ ] Testes incluidos ou existentes para o modulo
- [ ] Mega-header D1 presente no modulo
- [ ] Secoes PARTE com headers D2 e 4+ linhas de contexto
- [ ] Docstrings Google-style D5/D6 em todas as funcoes/classes
- [ ] Diagramas ASCII D3 quando 3+ caminhos/categorias

---

## 15. Padroes de Documentacao v2.0 (D1-D10 Adaptados)

O codigo v2.0 DEVE manter o mesmo nivel de detalhamento do legado,
adaptado para a estrutura de pacote Python modular.

### D1. Mega-Header Unicode (OBRIGATORIO — todos os modulos)

Cada arquivo `.py` em `geosteering_ai/` DEVE ter um mega-header no topo:

```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: {{subpacote}}/{{nome_modulo}}.py                                 ║
# ║  Bloco: {{N}} — {{Nome do Bloco}}                                         ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • {{Bullet 1 — acao principal}}                                        ║
# ║    • {{Bullet 2 — acao secundaria}}                                       ║
# ║    • {{Bullet 3+ — outras acoes}}                                         ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig),                                ║
# ║                {{modulo}} ({{simbolos}})                                   ║
# ║  Exports: ~{{N}} funcoes/classes — ver __all__                            ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao {{X.Y}}                              ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ║    {{vX.Y.Z}} — {{descricao da mudanca}}                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
```

**Campos OBRIGATORIOS (14):**
1. Modulo (subpacote/nome)
2. Bloco (numero e nome)
3. Projeto (Geosteering AI v2.0)
4. Autor (Daniel Leal)
5. Framework (TF/Keras + proibicao PyTorch)
6. Ambiente (VSCode + CI + Colab)
7. Pacote (geosteering_ai pip installable)
8. Config (PipelineConfig, NUNCA globals)
9. Proposito (3+ bullets com •)
10. Dependencias (lista EXPLICITA de modulos e simbolos)
11. Exports (contagem e referencia a __all__)
12. Ref (secao do ARCHITECTURE_v2.md)
13. Historico (versao + data + descricao)

### D2. Cabecalho de Secao (OBRIGATORIO — todas as secoes logicas)

```python
# ════════════════════════════════════════════════════════════════════════════
# SECAO: {{TITULO EM MAIUSCULAS}}
# ════════════════════════════════════════════════════════════════════════════
# {{Linha 1: Descricao do proposito desta secao}}
# {{Linha 2: Contexto tecnico/fisico relevante}}
# {{Linha 3: Relacao com padroes P## ou perspectivas}}
# {{Linha 4: Referencia cruzada a outros modulos ou docs}}
# ──────────────────────────────────────────────────────────────────────────
```

**Requisito minimo:** >=4 linhas de comentario contextual ANTES do codigo.

### D3. Diagramas ASCII (OBRIGATORIO se >= 3 caminhos/categorias)

Usar caracteres Unicode para bordas: `┌ ─ ┬ ┐ │ ├ ┼ ┤ └ ┴ ┘`

Situacoes obrigatorias:
- Fluxos de dados com >= 3 etapas
- Mapeamentos semanticos (categorias de componentes)
- Cascatas de transformacao (FV, GS, scaling)
- Catalogos de componentes (arquiteturas, losses, noise)
- Formulas fisicas com multiplas variaveis
- Layout de dados (22-col, tensor EM)

### D4. Atributos de Config (OBRIGATORIO — bloco de 4+ linhas por grupo)

No `config.py`, cada grupo de atributos do PipelineConfig DEVE ter:

```python
# ── GRUPO: Nome do Grupo ─────────────────────────────────────────────────
# Descricao: [Frase COMPLETA descrevendo o grupo]
# Relacao: [Como afeta o pipeline — quais modulos usam]
# Ref: [Secao do ARCHITECTURE_v2.md ou docs/physics/]
# Nota: [Versao onde foi introduzido ou alterado]
atributo_1: tipo = default  # descricao inline
atributo_2: tipo = default  # descricao inline
```

### D5. Docstrings de Funcoes (OBRIGATORIO — Google-style, 5+ campos)

```python
def nome_funcao(
    data: np.ndarray,
    config: PipelineConfig,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Descricao concisa em uma linha.

    Descricao detalhada em um ou mais paragrafos. Explicar contexto
    fisico, algoritmo, logica de dominio. Mencionar padroes P## ou
    perspectivas relevantes. Citar referencia bibliografica se aplicavel.

    A cadeia de dados segue: raw → noise → FV → GS → scale.
    Este modulo implementa a etapa {{X}} da cadeia.

    Args:
        data: Array (n_rows, n_feat) ou (n_seq, seq_len, n_feat).
            Layout esperado: [prefix, z, Re(H1), Im(H1), Re(H2), Im(H2)].
        config: Configuracao do pipeline. Atributos usados:
            - config.feature_view: Nome da Feature View
            - config.eps_tf: Epsilon para estabilidade numerica
        verbose: Se True, loga detalhes intermediarios.

    Returns:
        np.ndarray: Array com mesma shape, canais EM transformados.
            4 colunas EM substituidas conforme Feature View selecionada.

    Raises:
        ValueError: Se feature_view nao esta em VALID_VIEWS.
        AssertionError: Se data.ndim nao eh 2 ou 3.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig(feature_view="logH1_logH2")
        >>> result = nome_funcao(data, config)
        >>> assert result.shape == data.shape

    Note:
        Referencia: docs/ARCHITECTURE_v2.md secao 4.3.
        Bug fix v2.0: Legado usava ln (numpy) vs log10 (TF). Agora ambos
        usam log10 consistentemente.
    """
```

**Campos OBRIGATORIOS:** descricao, Args (com tipos e restricoes),
Returns (com formato), Raises, Example, Note.

### D6. Docstrings de Classes (OBRIGATORIO — Attributes + Example)

```python
@dataclass
class NomeDaClasse:
    """Descricao da classe em uma linha.

    Descricao detalhada com contexto fisico e proposito arquitetural.
    Explicar como se encaixa no pipeline e quais modulos a utilizam.

    Attributes:
        attr1 (tipo): Descricao — uso semantico no pipeline.
            Restricoes: [range, enum, etc.]
        attr2 (tipo): Descricao — uso semantico.

    Example:
        >>> obj = NomeDaClasse(param1=valor1, param2=valor2)
        >>> print(obj.attr1)

    Note:
        Utilizado em: data/pipeline.py (DataPipeline.prepare()),
        training/loop.py (TrainingLoop.run()).
        Ref: docs/ARCHITECTURE_v2.md secao {{X.Y}}.
    """
```

### D7. Comentarios Inline Semanticos

```python
# ── Componentes EM (4 colunas do tensor H) ───────────────────────────────
re_h1 = data[:, em_start]       # Re(Hxx) — componente planar
im_h1 = data[:, em_start + 1]   # Im(Hxx) — componente planar
re_h2 = data[:, em_start + 2]   # Re(Hzz) — componente axial
im_h2 = data[:, em_start + 3]   # Im(Hzz) — componente axial
```

### D8. Inventario de Exports (OBRIGATORIO — todo modulo com __all__)

```python
# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Constantes ────────────────────────────────────────────────────────
    "EPS",
    "VALID_VIEWS",
    # ── Funcoes numpy ─────────────────────────────────────────────────────
    "apply_feature_view",
    # ── Funcoes TensorFlow ────────────────────────────────────────────────
    "apply_feature_view_tf",
]
```

### D9. Logging Estruturado (OBRIGATORIO — ao inves de print)

```python
import logging
logger = logging.getLogger(__name__)

# ── Inicio de operacao ────────────────────────────────────────────────────
logger.info("Aplicando Feature View '%s' em %d amostras", view, n_samples)

# ── Resultado ──────────────────────────────────────────────────────────────
logger.info("Feature View aplicada: shape=%s, view='%s'", result.shape, view)
logger.debug("  Canais EM transformados: [%d:%d]", em_start, em_start + 4)
```

### D10. Constantes com Documentacao Fisica

```python
# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES FISICAS E DE CONFIGURACAO
# ════════════════════════════════════════════════════════════════════════════
# Valores criticos validados pela Errata v4.4.5 + v5.0.15.
# Qualquer alteracao DEVE ser aprovada e documentada.
# Ref: docs/physics/errata_valores.md
# ──────────────────────────────────────────────────────────────────────────

# Epsilon seguro para float32 — protege contra underflow em log/divisao.
# NUNCA usar 1e-30 (causa subnormais em float32, gradientes explodidos).
# Ref: Errata v5.0.15, IEEE 754 float32 min normal ≈ 1.175e-38.
EPS = 1e-12

# Feature Views validas — 6 transformacoes sobre componentes EM.
# identity/raw: passthrough. H1_logH2: H1 raw + H2 log10.
# logH1_logH2: ambos log10. IMH1_IMH2_*: partes imaginarias + razao.
# Ref: docs/physics/perspectivas.md secao Feature Views.
VALID_VIEWS = {"identity", "raw", "H1_logH2", "logH1_logH2",
               "IMH1_IMH2_razao", "IMH1_IMH2_lograzao"}
```

### D11. Tabelas de Formulas ASCII (OBRIGATORIO — catalogos de componentes)

Quando o modulo implementa um catalogo de transformacoes ou componentes
(Feature Views, Geosignal Families, Loss Functions, Noise Types),
incluir uma tabela ASCII dentro do comentario de secao mostrando
TODAS as variantes com suas formulas e canais de saida.

```python
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  6 Feature Views Canonicas (Ref: docs/physics/perspectivas.md):         │
#   │                                                                          │
#   │  View               │ Canal 0    │ Canal 1  │ Canal 2      │ Canal 3    │
#   │  ───────────────────┼────────────┼──────────┼──────────────┼────────────│
#   │  identity / raw     │ Re(H1)     │ Im(H1)   │ Re(H2)       │ Im(H2)    │
#   │  H1_logH2           │ Re(H1)     │ Im(H1)   │ log10|H2|    │ φ(H2)     │
#   │  logH1_logH2        │ log10|H1|  │ φ(H1)    │ log10|H2|    │ φ(H2)     │
#   │  IMH1_IMH2_razao    │ Im(H1)     │ Im(H2)   │ |H1|/|H2|   │ Δφ        │
#   │  IMH1_IMH2_lograzao │ Im(H1)     │ Im(H2)   │ log10(ratio) │ Δφ        │
#   │                                                                          │
#   │  H1 = Hxx (planar), H2 = Hzz (axial)                                   │
#   │  |H| = √(Re² + Im² + ε),  φ(H) = arctan2(Im, Re)                      │
#   │  ε = 1e-12 (float32 safe — NUNCA 1e-30)                                │
#   │  SEMPRE log10 (NUNCA ln — bug fix v2.0)                                │
#   └──────────────────────────────────────────────────────────────────────────┘
```

### D12. Cross-References em Docstrings (OBRIGATORIO — Note: section)

Toda funcao publica DEVE incluir uma secao `Note:` na docstring com:
1. Onde mais esta funcao eh usada no pipeline
2. Referencia a secao da documentacao
3. Bug fixes relevantes (se aplicavel)
4. Restricoes criticas (errata, guards numericos)

```python
    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._apply_fv_gs() (modo offline)
            - data/pipeline.py: build_train_map_fn() (modo on-the-fly)
            - tests/test_data_pipeline.py: TestFeatureViews
        Ref: docs/ARCHITECTURE_v2.md secao 4.3.
        Bug fix v2.0: Legado (C22) usava ln (numpy) vs log10 (TF).
            Agora ambos usam log10 consistentemente.
        Guard numerico: EPS = 1e-12 (NUNCA 1e-30 em float32).
```

### D13. Branch Comments com Layout de Saida (OBRIGATORIO — transformacoes)

Cada branch `if/elif` que transforma dados DEVE ter um comentario
mostrando o layout exato dos canais de saida:

```python
    if view == "H1_logH2":
        # ── H1_logH2: H1 cru preserva SNR em alta atenuacao,
        #    H2 log10-transformado comprime faixa dinamica larga de Hzz.
        #    Saida: [Re(H1), Im(H1), log10|H2|, φ(H2)]
        #    Motivacao fisica: Hzz varia 4+ ordens de magnitude,
        #    log10 estabiliza gradientes e melhora convergencia.
        result[:, em_start + 2] = _safe_log10(mag_h2)
        result[:, em_start + 3] = phi_h2

    elif view == "logH1_logH2":
        # ── logH1_logH2: Ambos H1 e H2 em escala logaritmica.
        #    Saida: [log10|H1|, φ(H1), log10|H2|, φ(H2)]
        #    Motivacao fisica: Magnitude e fase capturam toda a informacao
        #    do sinal complexo. Log10 comprime faixa para melhor treinamento.
        result[:, em_start]     = _safe_log10(mag_h1)
        result[:, em_start + 1] = phi_h1
        result[:, em_start + 2] = _safe_log10(mag_h2)
        result[:, em_start + 3] = phi_h2
```

### D14. Interacao Noise × FV × GS (OBRIGATORIO — pipeline.py)

O modulo pipeline.py DEVE documentar explicitamente como o noise
interage com Feature Views e Geosinais:

```python
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  INTERACAO NOISE × FV × GS (Fidelidade Fisica LWD)                     │
# ├──────────────────────────────────────────────────────────────────────────┤
# │                                                                          │
# │  CORRETO (on-the-fly, dentro de tf.data.map):                           │
# │    raw Re/Im → noise(σ) → FV(noisy) → GS(noisy) → scale → modelo      │
# │                                    │                                     │
# │                          GS veem ruido ✓ (fidelidade LWD)               │
# │                                                                          │
# │  ERRADO (offline, bug do legado):                                        │
# │    raw Re/Im → FV(clean) → GS(clean) → scale → noise → modelo          │
# │                                    │                                     │
# │                          GS nunca veem ruido ✗ (bias sistematico)        │
# │                                                                          │
# │  REGRAS:                                                                 │
# │    1. Scaler SEMPRE fitado em dados LIMPOS (FV+GS clean, temporario)    │
# │    2. Val/test transformados offline (sem noise, FV+GS+scale)            │
# │    3. Train permanece raw para noise on-the-fly via tf.data.map()       │
# │    4. NUNCA aplicar noise offline quando FV ou GS estao ativos           │
# └──────────────────────────────────────────────────────────────────────────┘
```
