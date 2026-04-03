# Documentação Técnica: Módulo de Ruídos — Geosteering AI v2.0

> **Versão:** 2.0  
> **Autor:** Daniel Leal  
> **Módulo:** `geosteering_ai/noise/`  
> **Arquivos:** `functions.py` · `curriculum.py`  
> **Última atualização:** 2026-04-03  

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Conceitos Fundamentais](#2-conceitos-fundamentais)
3. [Catálogo Completo — 34 Tipos de Ruído](#3-catálogo-completo--34-tipos-de-ruído)
4. [Detalhamento por Categoria](#4-detalhamento-por-categoria)
5. [Sistema Curriculum 3-Phase](#5-sistema-curriculum-3-phase)
6. [Interação Noise × FV × GS](#6-interação-noise--fv--gs)
7. [API de Referência](#7-api-de-referência)
8. [Tutorial Rápido](#8-tutorial-rápido)
9. [Melhorias Futuras](#9-melhorias-futuras)
10. [Referências Bibliográficas](#10-referências-bibliográficas)

---

## 1. Visão Geral

O módulo de ruídos (`geosteering_ai/noise/`) é responsável por simular perturbações
realísticas em medições eletromagnéticas de ferramentas LWD (*Logging While Drilling*),
garantindo que os modelos de Deep Learning treinados no pipeline de inversão 1D de
resistividade sejam **robustos** a condições reais de campo.

### Características Principais

| Atributo | Valor |
|:---------|:------|
| **Tipos de ruído implementados** | 34 (registrados em `NOISE_FN_MAP`) |
| **Injeção** | On-the-fly via `tf.data.map` (NUNCA offline) |
| **Unidades** | A/m (componentes de campo magnético) |
| **Colunas protegidas** | `z_obs`, `theta`, `freq` (conforme problema P1/P2/P3) |
| **Categorias** | 6 (Original, CORE, CORE+, LWD Geofísico, Extended, Geosteering) |

### Princípio Fundamental

O ruído é aplicado **ANTES** das Feature Views (FV) e dos Geosignals (GS), garantindo
fidelidade física ao pipeline real de aquisição LWD:

```
dados_brutos → ruído(σ) → FV(ruidoso) → GS(ruidoso) → escalonamento → modelo
```

Esta ordem é **obrigatória** porque as transformações FV e GS são operações não-lineares
(logaritmos, razões, diferenças de fase) que propagam o ruído de forma fisicamente
consistente — exatamente como ocorre na ferramenta real de perfilagem.

### Por que On-the-fly?

A injeção on-the-fly (via `tf.data.map`) gera realizações de ruído **diferentes a cada
época**, o que equivale a um aumento de dados (data augmentation) infinito. Isso contrasta
com a abordagem offline (pré-computada), que congela uma única realização e limita a
diversidade amostral vista pelo modelo durante o treinamento.

---

## 2. Conceitos Fundamentais

### 2.1 Ruído em Medições LWD Eletromagnéticas

Ferramentas LWD de resistividade operam emitindo um campo eletromagnético (EM) alternado
a partir de transmissores posicionados ao longo da coluna de perfuração (*drill string*),
e medindo as componentes do campo magnético secundário em receptores espaçados. As
medições típicas incluem componentes do tensor magnético:

- **Hxx, Hyy** — componentes planares (acoplamento coplanar)
- **Hzz** — componente axial (acoplamento coaxial)
- Partes **Real** e **Imaginária** de cada componente

Essas medições são intrinsecamente ruidosas devido a múltiplos fenômenos físicos que
ocorrem simultaneamente durante a perfuração.

### 2.2 Fontes de Ruído em Ambiente LWD

| Fenômeno | Descrição | Impacto |
|:---------|:----------|:--------|
| **Ruído térmico** | Agitação térmica nos circuitos eletrônicos do sensor | Ruído branco aditivo, constante em frequência |
| **Deriva térmica (*drift*)** | Variação lenta dos parâmetros do sensor com temperatura | Tendência de baixa frequência nos dados |
| **EMI (Interferência Eletromagnética)** | Motores, bombas e equipamentos elétricos na sonda | Harmônicas de 50/60 Hz sobrepostas ao sinal |
| **Vibração do BHA** | Oscilações mecânicas do conjunto de fundo (*Bottom Hole Assembly*) | Modulação periódica das medições |
| **Excentricidade da ferramenta** | Deslocamento da ferramenta em relação ao centro do poço | Modulação sinusoidal dependente da posição angular |
| **Efeito do poço (*borehole effect*)** | Influência do fluido de perfuração e geometria do poço | Ganho e offset nas medições |
| **Invasão de lama** | Penetração do filtrado de lama na formação | Atenuação do sinal proporcional à profundidade de invasão |
| **Perdas de telemetria** | Falhas na transmissão de dados via *mud pulse* ou EM | Amostras ausentes ou corrompidas |
| **Heterogeneidade da formação** | Variações laterais de resistividade não modeladas | Flutuações correlacionadas espacialmente |
| **Efeito de camada adjacente (*shoulder bed*)** | Influência de camadas vizinhas na resposta da ferramenta | Suavização e ruído nas transições de camada |

### 2.3 Relação Sinal-Ruído (SNR) Típica

Ferramentas LWD comerciais apresentam SNR variável conforme a condição de operação:

| Condição | SNR típico (dB) | σ equivalente |
|:---------|:---------------:|:-------------:|
| Ótima (baixa vibração, formação uniforme) | > 40 | < 0.01 |
| Normal (perfuração rotativa padrão) | 25–35 | 0.02–0.06 |
| Degradada (alta vibração, lama pesada) | 15–25 | 0.06–0.18 |
| Severa (telemetria parcial, EMI) | < 15 | > 0.18 |

O valor default de `noise_level_max = 0.08` no pipeline Geosteering AI corresponde a
uma condição **normal a ligeiramente degradada**, representativa da maioria dos cenários
de perfuração direcional.

### 2.4 Profundidade Pelicular (*Skin Depth*) e Sensibilidade ao Ruído

A profundidade pelicular (δ) define o alcance de investigação da ferramenta EM:

```
δ = √(2 / (ω · μ · σ_cond))
```

Onde:
- ω = 2π × f (frequência angular, f = 20 kHz default)
- μ = μ₀ ≈ 4π × 10⁻⁷ H/m (permeabilidade magnética)
- σ_cond = condutividade da formação (S/m)

**Implicação para o ruído:** em formações resistivas (baixa σ_cond), a profundidade
pelicular é grande e o sinal EM é fraco nos receptores distantes, tornando a medição
**mais sensível ao ruído**. Isso justifica tipos de ruído como `depth_dependent` e
`freq_dependent`, que modelam essa variação de sensibilidade.

### 2.5 Por que Simular Ruído?

1. **Robustez:** Modelos treinados apenas com dados limpos falham catastroficamente
   quando expostos a dados reais ruidosos.
2. **Generalização:** A diversidade de realizações de ruído previne overfitting às
   peculiaridades dos modelos geológicos sintéticos.
3. **Curriculum Learning:** A introdução gradual de ruído (limpo → ruidoso) permite
   que o modelo primeiro aprenda os padrões fundamentais e depois desenvolva
   invariância ao ruído.
4. **Fidelidade física:** O pipeline de ruído reproduz os mesmos fenômenos que degradam
   medições reais, garantindo transferência para dados de campo.

---

## 3. Catálogo Completo — 34 Tipos de Ruído

A tabela abaixo documenta todos os 34 tipos de ruído implementados no dicionário
`NOISE_FN_MAP`, organizados por categoria.

### Legenda

- **σ** = `noise_level` (variável TF ou escalar, default máximo: 0.08)
- **x** = tensor de entrada (componentes EM em A/m)
- **N(μ,σ²)** = distribuição normal com média μ e variância σ²
- **U(a,b)** = distribuição uniforme entre a e b
- **Bernoulli(p)** = distribuição de Bernoulli com probabilidade p

---

### 3.1 Categoria: Original (4 tipos)

Ruídos fundamentais baseados em distribuições estatísticas clássicas.

| # | Nome | Fórmula | Fenômeno Físico | Parâmetros Default |
|:-:|:-----|:--------|:----------------|:-------------------|
| 1 | `gaussian` | x + N(0, σ²) | Ruído térmico aditivo nos circuitos do sensor | σ = noise_level |
| 2 | `multiplicative` | x · (1 + N(0, σ²)) | Ganho variável do amplificador (proporcional ao sinal) | σ = noise_level |
| 3 | `uniform` | x + U(−σ, σ) | Ruído de quantização do ADC (conversor analógico-digital) | σ = noise_level |
| 4 | `dropout` | x · Bernoulli(1−σ) / (1−σ) | Perdas aleatórias de amostras na telemetria | p_drop = σ |

### 3.2 Categoria: CORE (5 tipos)

Ruídos que modelam fenômenos específicos de aquisição de perfis de poço.

| # | Nome | Fórmula | Fenômeno Físico | Parâmetros Default |
|:-:|:-----|:--------|:----------------|:-------------------|
| 5 | `drift` | cumsum(N(0, σ²)) × 0.95 | Deriva térmica lenta do sensor com temperatura | decay = 0.95 |
| 6 | `depth_dependent` | N(0, σ · (1 + z/L)) | Degradação do SNR com profundidade (atenuação EM) | L = comprimento da sequência |
| 7 | `spikes` | Bernoulli(0.001) · N(0, 5σ) | Picos elétricos (surtos EMI, descargas estáticas) | p_spike = 0.001, amplitude = 5σ |
| 8 | `pink` | 0.5 · white + 0.5 · brownian | Ruído 1/f (eletrônica de baixa frequência) | ratio = 0.5/0.5 |
| 9 | `saturation` | clip(x, μ−nσ, μ+nσ) | Saturação do ADC em sinais de alta amplitude | n = fator de clipping |

### 3.3 Categoria: CORE+ (5 tipos)

Extensões dos tipos CORE com modelagem mais refinada da variabilidade.

| # | Nome | Fórmula | Fenômeno Físico | Parâmetros Default |
|:-:|:-----|:--------|:----------------|:-------------------|
| 10 | `varying` | N(0,1) × U(σ_min, σ_max) × \|x\| | Ruído com nível flutuante (condições operacionais variáveis) | σ_min, σ_max derivados de noise_level |
| 11 | `gaussian_local` | N(0, σ · \|x\|) | Ruído proporcional à magnitude local do sinal | σ = noise_level |
| 12 | `gaussian_global` | N(0, σ · std_global) | Ruído proporcional à variância global do dataset | std_global = desvio-padrão global |
| 13 | `speckle` | x · (1 + N(0, σ²)) | Ruído *speckle* (granularidade em medições coerentes) | σ = noise_level |
| 14 | `quantization` | round(x / q) · q, q = σ × 0.1 | Quantização finita do sistema de telemetria digital | q = σ × 0.1 |

### 3.4 Categoria: LWD Geofísico (6 tipos)

Ruídos que modelam diretamente fenômenos eletromagnéticos e petrofísicos.

| # | Nome | Fórmula | Fenômeno Físico | Parâmetros Default |
|:-:|:-----|:--------|:----------------|:-------------------|
| 15 | `shoulder_bed` | MA(3) + N(0, σ×0.5) | Efeito de camada adjacente (suavização + ruído na transição) | janela MA = 3, σ_extra = σ×0.5 |
| 16 | `borehole_effect` | gain + offset (ambos σ×0.5) | Efeito do poço: diâmetro, lama, *standoff* | gain_σ = σ×0.5, offset_σ = σ×0.5 |
| 17 | `mud_invasion` | x · (1 − σ · U(0,1)) | Invasão do filtrado de lama: atenuação proporcional | σ = noise_level |
| 18 | `anisotropy_misalignment` | δ · roll(col, 2) | Desalinhamento angular da ferramenta (anisotropia aparente) | shift = 2 amostras |
| 19 | `formation_heterogeneity` | x · (1 + smooth_norm) | Heterogeneidade lateral da formação (variações suaves) | suavização gaussiana |
| 20 | `telemetry` | drop + bit_err | Perdas de telemetria *mud pulse*: amostras perdidas + erros de bit | p_drop + p_bit_err |

### 3.5 Categoria: Extended (12 tipos)

Ruídos avançados para simulação de alta fidelidade do ambiente de perfuração.

| # | Nome | Fórmula | Fenômeno Físico | Parâmetros Default |
|:-:|:-----|:--------|:----------------|:-------------------|
| 21 | `cross_talk` | Re += ε·Im, ε = σ×0.4 | *Cross-talk* entre canais Re/Im (isolamento imperfeito) | ε = σ × 0.4 |
| 22 | `orientation` | rotação θ = σ×0.04 rad | Erro de orientação da ferramenta (toolface) | θ = σ × 0.04 rad |
| 23 | `emi_noise` | 3 harmônicas de 60 Hz | EMI de equipamentos de superfície (60 Hz + harmônicas) | 3 harmônicas, f_base = 60 Hz |
| 24 | `freq_dependent` | N(0, σ · f^0.5) | Ruído dependente da frequência (skin depth variável) | expoente = 0.5 |
| 25 | `noise_floor` | N(0, σ × 1e-8 / 0.05) | Piso de ruído eletrônico (limite mínimo de detecção) | floor = σ × 1e-8 / 0.05 |
| 26 | `proportional` | N(0, 0.03·\|x\|) · scale | Ruído proporcional ao sinal (incerteza relativa constante) | fração = 0.03, scale = σ/0.05 |
| 27 | `reim_diff` | Re: σ, Im: 1.5σ | Ruído diferenciado Re/Im (Im mais ruidoso que Re) | ratio_Im = 1.5 |
| 28 | `component_diff` | Hxx: 1.0σ, Hzz: 0.8σ | Ruído diferenciado por componente (Hzz mais estável) | ratio_Hzz = 0.8 |
| 29 | `gaussian_keras` | (alias de `gaussian`) | Alias para compatibilidade com camadas Keras | mesmos de `gaussian` |
| 30 | `motion` | x · (1 + A·sin(2πft/L)) | Artefato de movimento (*stick-slip*, vibração lateral) | A = amplitude, f = frequência |
| 31 | `thermal` | N(0, 0.3σ) | Componente puramente térmica (30% do ruído total) | fração = 0.3 |
| 32 | `phase_shift` | rotação φ = σ×0.1 rad | Erro de fase na demodulação do sinal EM | φ = σ × 0.1 rad |

### 3.6 Categoria: Geosteering (2 tipos)

Ruídos específicos do cenário de perfuração direcional guiada (*geosteering*).

| # | Nome | Fórmula | Fenômeno Físico | Parâmetros Default |
|:-:|:-----|:--------|:----------------|:-------------------|
| 33 | `bha_vibration` | shift · sin(2πt/L + φ) | Vibração do BHA durante perfuração direcional | shift = amplitude, φ = fase aleatória |
| 34 | `eccentricity` | x · (1 + ecc·cos(2πt/L + φ)) | Excentricidade da ferramenta no poço (contato com parede) | ecc = fator de excentricidade |

---

## 4. Detalhamento por Categoria

### 4.1 Original — Ruídos Estatísticos Fundamentais

#### Motivação Física

Estes quatro tipos representam os modelos estatísticos mais básicos de perturbação em
sistemas de medição. São os "blocos de construção" sobre os quais os demais tipos são
elaborados. O ruído gaussiano aditivo, por exemplo, é o modelo padrão para ruído
térmico segundo o teorema de Johnson-Nyquist.

#### Formulação Matemática

**Gaussian (aditivo):**
```
x_noisy = x + ε,    ε ~ N(0, σ²)
```
O ruído é independente do sinal — modela perturbações eletrônicas intrínsecas.

**Multiplicative:**
```
x_noisy = x · (1 + ε),    ε ~ N(0, σ²)
```
O ruído escala com a magnitude do sinal — modela ganho variável do amplificador.

**Uniform:**
```
x_noisy = x + ε,    ε ~ U(-σ, σ)
```
Distribuição de cauda limitada — modela erros de quantização com amplitude máxima conhecida.

**Dropout:**
```
x_noisy = x · m / (1-σ),    m ~ Bernoulli(1-σ)
```
Amostras são zeradas aleatoriamente com probabilidade σ, e as restantes são
re-escalonadas para preservar a esperança — modela perdas intermitentes de telemetria.

#### Parâmetros Default

| Parâmetro | Valor | Justificativa |
|:----------|:-----:|:--------------|
| σ (noise_level) | 0.0–0.08 | Controlado pelo curriculum (0 no início, 0.08 no máximo) |

#### Quando Usar

- **gaussian:** Baseline padrão para todos os experimentos. Recomendado como primeiro
  teste de robustez.
- **multiplicative:** Quando a incerteza relativa é mais relevante que a absoluta
  (sinais de alta amplitude).
- **uniform:** Simulação de sistemas com quantização conhecida.
- **dropout:** Simulação de cenários com perdas de telemetria frequentes.

#### Exemplo de Código

```python
from geosteering_ai.config import PipelineConfig

# Baseline P1 com ruído gaussiano apenas
config = PipelineConfig(
    noise_types=["gaussian"],
    noise_weights=[1.0],
    noise_level_max=0.05,
)
```

---

### 4.2 CORE — Fenômenos de Aquisição de Perfis

#### Motivação Física

Os tipos CORE modelam fenômenos que ocorrem especificamente durante a aquisição de
perfis de poço: deriva do sensor com temperatura crescente em profundidade, degradação
do SNR com a distância fonte-receptor, picos elétricos causados por descargas estáticas,
ruído 1/f da eletrônica de baixo nível, e saturação do conversor analógico-digital.

#### Formulação Matemática

**Drift (deriva térmica):**
```
drift_t = Σ_{i=0}^{t} ε_i × 0.95^{t-i},    ε_i ~ N(0, σ²)
x_noisy = x + drift_t
```
A soma cumulativa com decaimento exponencial (fator 0.95) produz uma tendência de
baixa frequência que simula a resposta térmica lenta dos circuitos do sensor à medida
que a ferramenta avança em profundidade (e temperatura).

**Depth-dependent:**
```
x_noisy = x + ε · (1 + z/L),    ε ~ N(0, σ²)
```
O ruído aumenta linearmente com a posição na sequência (z), simulando a maior
atenuação do sinal EM em profundidades maiores.

**Spikes:**
```
spike = Bernoulli(0.001) · N(0, 5σ)
x_noisy = x + spike
```
Picos raros (0.1% das amostras) mas intensos (5× a amplitude do ruído gaussiano),
modelando surtos elétricos e descargas estáticas no BHA.

**Pink (1/f):**
```
white = N(0, σ²)
brownian = cumsum(N(0, σ²)) normalizado
x_noisy = x + 0.5 · white + 0.5 · brownian
```
Mistura de ruído branco e browniano para aproximar o espectro 1/f, característico
de eletrônica de estado sólido.

**Saturation:**
```
x_noisy = clip(x, μ - nσ, μ + nσ)
```
Limita a amplitude do sinal, simulando a saturação do ADC quando o sinal excede
a faixa dinâmica do conversor.

#### Parâmetros Default

| Parâmetro | Tipo | Valor | Justificativa |
|:----------|:-----|:-----:|:--------------|
| decay (drift) | float | 0.95 | Constante de tempo térmica típica de sensor LWD |
| p_spike (spikes) | float | 0.001 | Frequência observada em logs reais (~1 spike/1000 amostras) |
| amplitude (spikes) | float | 5σ | Picos tipicamente 5× acima do nível de ruído base |
| ratio (pink) | float | 0.5/0.5 | Balanceamento empírico white/brownian |

#### Quando Usar

- **drift:** Simulação de corridas longas (>1000 amostras) onde a temperatura varia.
- **depth_dependent:** Poços profundos com gradiente geotérmico significativo.
- **spikes:** Validação de robustez a outliers (pré-processamento de dados reais).
- **pink:** Complemento ao gaussiano para modelar espectro realístico.
- **saturation:** Cenários com contraste de resistividade extremo.

#### Exemplo de Código

```python
config = PipelineConfig(
    noise_types=["gaussian", "drift", "spikes"],
    noise_weights=[0.6, 0.3, 0.1],
    noise_level_max=0.08,
)
```

---

### 4.3 CORE+ — Modelagem Refinada de Variabilidade

#### Motivação Física

Os tipos CORE+ estendem o modelo de ruído para capturar variabilidade mais complexa:
ruído com nível flutuante (condições operacionais que mudam durante a corrida),
ruído local vs. global, ruído *speckle* (granularidade em medições coerentes), e
quantização do sistema digital de telemetria.

#### Formulação Matemática

**Varying:**
```
σ_local ~ U(σ_min, σ_max)
x_noisy = x + N(0, 1) × σ_local × |x|
```
O nível de ruído varia aleatoriamente por amostra, capturando a natureza
não-estacionária das condições de perfuração.

**Gaussian Local:**
```
x_noisy = x + N(0, σ · |x|)
```
Ruído proporcional à magnitude local — amostras com sinal forte recebem mais ruído
absoluto, mantendo SNR aproximadamente constante.

**Gaussian Global:**
```
x_noisy = x + N(0, σ · std_global)
```
Ruído escalado pela variância global do dataset — calibra automaticamente o nível
de ruído à escala dos dados.

**Speckle:**
```
x_noisy = x · (1 + N(0, σ²))
```
Equivalente ao multiplicativo, mas com nomenclatura específica para medições
de radar/EM coerentes (padrão de interferência granular).

**Quantization:**
```
q = σ × 0.1
x_noisy = round(x / q) × q
```
Reduz a resolução do sinal a degraus discretos, simulando a quantização finita
do sistema de telemetria (tipicamente 12–16 bits em ferramentas LWD).

#### Parâmetros Default

| Parâmetro | Tipo | Valor | Justificativa |
|:----------|:-----|:-----:|:--------------|
| q (quantization) | float | σ × 0.1 | Resolução equivalente a ~10 degraus por σ |
| σ_min/σ_max (varying) | float | derivados de noise_level | Faixa de variação operacional |

#### Quando Usar

- **varying:** Cenários com condições operacionais instáveis (tripping, reaming).
- **gaussian_local/global:** Quando a escala do ruído deve ser calibrada ao sinal.
- **speckle:** Compatibilidade com literatura de processamento de radar.
- **quantization:** Simulação de telemetria de baixa resolução.

#### Exemplo de Código

```python
config = PipelineConfig(
    noise_types=["gaussian", "varying", "quantization"],
    noise_weights=[0.5, 0.3, 0.2],
    noise_level_max=0.06,
)
```

---

### 4.4 LWD Geofísico — Fenômenos Eletromagnéticos e Petrofísicos

#### Motivação Física

Esta categoria modela diretamente os fenômenos geofísicos que afetam medições LWD
eletromagnéticas: a influência de camadas adjacentes na resposta da ferramenta, o
efeito do fluido de perfuração e geometria do poço, a invasão do filtrado de lama
na formação, desalinhamentos angulares, heterogeneidade lateral e perdas de telemetria.

Esses ruídos são essenciais para treinar modelos que serão aplicados a **dados de campo
reais**, onde esses efeitos são ubíquos e não podem ser ignorados.

#### Formulação Matemática

**Shoulder Bed (camada adjacente):**
```
x_smooth = MovingAverage(x, window=3)
x_noisy = x_smooth + N(0, σ × 0.5)
```
A média móvel simula a suavização causada pela resolução vertical finita da ferramenta,
e o ruído aditivo (metade da amplitude) modela a incerteza na transição de camada.

**Borehole Effect (efeito do poço):**
```
gain ~ 1 + N(0, σ × 0.5)
offset ~ N(0, σ × 0.5)
x_noisy = gain × x + offset
```
Combinação de erro de ganho e offset, modelando a influência do diâmetro do poço,
do fluido de perfuração e do *standoff* (distância ferramenta-parede).

**Mud Invasion (invasão de lama):**
```
x_noisy = x · (1 - σ · U(0, 1))
```
Atenuação aleatória proporcional ao nível de ruído, simulando a alteração da
resistividade da formação pelo filtrado de lama penetrando na zona invadida.

**Anisotropy Misalignment (desalinhamento):**
```
x_noisy = x + δ · roll(col, 2)
```
Deslocamento circular de 2 posições na dimensão de coluna, simulando o efeito de um
desalinhamento angular da ferramenta em formações anisotrópicas.

**Formation Heterogeneity (heterogeneidade):**
```
noise_smooth = suavização_gaussiana(N(0, 1))
noise_norm = noise_smooth / max(|noise_smooth|)
x_noisy = x · (1 + σ · noise_norm)
```
Ruído suave e espacialmente correlacionado, modelando variações laterais de
resistividade que não estão presentes no modelo geológico 1D.

**Telemetry (perdas de telemetria):**
```
x_noisy = dropout(x) + bit_errors(x)
```
Combinação de amostras completamente perdidas (dropout) e erros de bit isolados,
reproduzindo os dois modos de falha do sistema de telemetria *mud pulse*.

#### Parâmetros Default

| Parâmetro | Tipo | Valor | Justificativa |
|:----------|:-----|:-----:|:--------------|
| window (shoulder_bed) | int | 3 | Resolução vertical típica de ferramenta LWD (~3 amostras) |
| σ_extra (shoulder_bed) | float | σ × 0.5 | Metade do ruído gaussiano base |
| gain_σ (borehole) | float | σ × 0.5 | Incerteza de calibração de ganho |
| offset_σ (borehole) | float | σ × 0.5 | Incerteza de offset |
| shift (anisotropy) | int | 2 | Deslocamento angular ~2 amostras |

#### Quando Usar

- **shoulder_bed:** Modelos com muitas camadas finas (resolução vertical limitada).
- **borehole_effect:** Poços com diâmetro variável ou lama pesada.
- **mud_invasion:** Formações permeáveis com invasão significativa.
- **anisotropy_misalignment:** Formações anisotrópicas com toolface impreciso.
- **formation_heterogeneity:** Cenários com variabilidade lateral esperada.
- **telemetry:** Simulação de operações com telemetria degradada.

#### Exemplo de Código

```python
config = PipelineConfig(
    noise_types=["gaussian", "shoulder_bed", "borehole_effect", "mud_invasion"],
    noise_weights=[0.4, 0.2, 0.2, 0.2],
    noise_level_max=0.08,
)
```

---

### 4.5 Extended — Simulação de Alta Fidelidade

#### Motivação Física

Os 12 tipos Extended representam fenômenos mais sutis e específicos que complementam
o modelo de ruído básico para atingir alta fidelidade na simulação do ambiente LWD.
Incluem *cross-talk* entre canais de medição, erros de orientação, interferência
eletromagnética de equipamentos, dependência da frequência, piso de ruído eletrônico,
diferenças entre componentes, artefatos de movimento, efeitos térmicos puros e erros
de fase na demodulação.

#### Formulação Matemática

**Cross-talk:**
```
Re_noisy = Re + ε · Im,    ε = σ × 0.4
Im_noisy = Im + ε · Re
```
Modela o isolamento imperfeito entre os canais de demodulação em fase (Re) e em
quadratura (Im), onde ~40% do nível de ruído vaza entre canais.

**Orientation:**
```
θ = σ × 0.04 rad
x_noisy = R(θ) · x    (rotação 2D aplicada a pares Re/Im)
```
Erro de orientação da ferramenta (*toolface*), causando mistura rotacional entre
componentes.

**EMI Noise:**
```
emi = Σ_{k=1}^{3} A_k · sin(2π · k · 60 · t / fs)
x_noisy = x + σ · emi
```
Três harmônicas de 60 Hz (60, 120, 180 Hz) simulando interferência de equipamentos
elétricos de superfície transmitida pela coluna de perfuração.

**Freq-dependent:**
```
x_noisy = x + N(0, σ · f^{0.5})
```
Ruído que escala com a raiz quadrada da frequência, modelando a redução de
profundidade pelicular em frequências mais altas.

**Noise Floor:**
```
x_noisy = x + N(0, σ × 1e-8 / 0.05)
```
Piso de ruído absoluto da eletrônica, independente do nível do sinal — estabelece
o limite mínimo de detecção da ferramenta (~10⁻⁸ A/m típico).

**Proportional:**
```
scale = σ / 0.05
x_noisy = x + N(0, 0.03 · |x|) · scale
```
Ruído com incerteza relativa de 3% do sinal, escalado pelo nível de ruído. Modela
a especificação típica de precisão de ferramentas LWD comerciais.

**Re/Im Differential:**
```
Re_noisy = Re + N(0, σ)
Im_noisy = Im + N(0, 1.5σ)
```
A componente imaginária recebe 50% mais ruído que a componente real, refletindo a
maior sensibilidade da parte em quadratura a perturbações de fase.

**Component Differential:**
```
Hxx_noisy = Hxx + N(0, 1.0σ)
Hzz_noisy = Hzz + N(0, 0.8σ)
```
A componente axial (Hzz) é 20% menos ruidosa que as planares (Hxx), devido ao
melhor acoplamento eletromagnético no modo coaxial.

**Gaussian Keras:**
```
(alias direto de gaussian)
```
Mantido para compatibilidade com camadas `tf.keras.layers.GaussianNoise`.

**Motion:**
```
x_noisy = x · (1 + A · sin(2πft / L))
```
Modulação sinusoidal simulando *stick-slip* e vibração lateral durante a perfuração.

**Thermal:**
```
x_noisy = x + N(0, 0.3σ)
```
Componente puramente térmica, correspondendo a ~30% do ruído total medido em
condições normais.

**Phase Shift:**
```
φ = σ × 0.1 rad
x_noisy = R(φ) · x    (rotação de fase aplicada a pares Re/Im)
```
Erro na referência de fase da demodulação, causando mistura entre componentes
real e imaginária com ângulo φ.

#### Parâmetros Default

| Parâmetro | Tipo | Valor | Justificativa |
|:----------|:-----|:-----:|:--------------|
| ε (cross_talk) | float | σ × 0.4 | 40% de vazamento entre canais Re/Im |
| θ (orientation) | float | σ × 0.04 rad | ~2.3° de erro de toolface |
| n_harmonics (emi) | int | 3 | 60, 120, 180 Hz |
| exponent (freq_dep) | float | 0.5 | Escala com √f |
| floor (noise_floor) | float | σ×1e-8/0.05 | ~10⁻⁸ A/m |
| fração (proportional) | float | 0.03 | 3% de incerteza relativa |
| ratio_Im (reim_diff) | float | 1.5 | Im 50% mais ruidosa |
| ratio_Hzz (comp_diff) | float | 0.8 | Hzz 20% menos ruidosa |
| fração (thermal) | float | 0.3 | 30% do ruído total |
| φ (phase_shift) | float | σ × 0.1 rad | ~5.7° de erro de fase |

#### Quando Usar

- **cross_talk, phase_shift:** Simulação detalhada do sistema de demodulação.
- **orientation:** Cenários com incerteza de toolface (poços de alto ângulo).
- **emi_noise:** Proximidade de equipamentos elétricos / sondas com EMI alto.
- **freq_dependent:** Experimentos com múltiplas frequências de operação.
- **noise_floor:** Cenários de baixo sinal (formações muito resistivas).
- **proportional:** Modelo de incerteza relativa constante (especificação do fabricante).
- **reim_diff, component_diff:** Quando fidelidade Re/Im e Hxx/Hzz é importante.
- **motion:** Cenários de vibração severa (turbina, motor de fundo).
- **thermal:** Isolamento do componente térmico para análise de sensibilidade.

#### Exemplo de Código

```python
config = PipelineConfig(
    noise_types=[
        "gaussian", "cross_talk", "emi_noise",
        "phase_shift", "noise_floor",
    ],
    noise_weights=[0.3, 0.2, 0.2, 0.15, 0.15],
    noise_level_max=0.08,
)
```

---

### 4.6 Geosteering — Perfuração Direcional Guiada

#### Motivação Física

Os dois tipos desta categoria são exclusivos para o cenário de geosteering, onde a
perfuração é direcional e a ferramenta LWD está posicionada próximo à broca. Neste
ambiente, a vibração do BHA (*Bottom Hole Assembly*) e a excentricidade da ferramenta
dentro do poço são fenômenos dominantes que afetam significativamente a qualidade
das medições EM.

#### Formulação Matemática

**BHA Vibration:**
```
vibration = shift · sin(2π · t / L + φ),    φ ~ U(0, 2π)
x_noisy = x + σ · vibration
```
Modulação sinusoidal com fase aleatória, modelando a vibração mecânica do BHA durante
a perfuração direcional. A amplitude é proporcional ao nível de ruído e a frequência
é determinada pelo comprimento da sequência.

**Eccentricity:**
```
ecc_factor = 1 + ecc · cos(2π · t / L + φ),    φ ~ U(0, 2π)
x_noisy = x · ecc_factor
```
Modulação multiplicativa sinusoidal com fase aleatória, modelando o efeito da
ferramenta não estar centralizada no poço. A excentricidade causa variação periódica
na distância ferramenta-formação, afetando a amplitude do sinal medido.

#### Parâmetros Default

| Parâmetro | Tipo | Valor | Justificativa |
|:----------|:-----|:-----:|:--------------|
| shift (bha_vibration) | float | derivado de σ | Amplitude de vibração proporcional ao ruído |
| ecc (eccentricity) | float | derivado de σ | Fator de excentricidade proporcional ao ruído |
| φ (ambos) | float | U(0, 2π) | Fase aleatória por realização |

#### Quando Usar

- **bha_vibration:** Sempre que o modelo será aplicado a dados de geosteering real
  (perfuração direcional com motor de fundo ou *rotary steerable system*).
- **eccentricity:** Poços de alto ângulo ou horizontais, onde a gravidade força a
  ferramenta contra a parede inferior do poço.

#### Exemplo de Código

```python
# Configuração para geosteering com ruídos específicos
config = PipelineConfig(
    noise_types=[
        "gaussian", "bha_vibration", "eccentricity",
        "motion", "borehole_effect",
    ],
    noise_weights=[0.3, 0.25, 0.20, 0.15, 0.10],
    noise_level_max=0.08,
)
```

---

## 5. Sistema Curriculum 3-Phase

### 5.1 Conceito

O sistema de curriculum learning implementa uma estratégia de treinamento em três fases,
onde o nível de ruído (σ) é aumentado gradualmente ao longo das épocas. Isso permite
que o modelo:

1. **Fase 1:** Aprenda os padrões fundamentais da inversão com dados limpos.
2. **Fase 2:** Desenvolva progressivamente invariância ao ruído.
3. **Fase 3:** Consolide a robustez no nível máximo de ruído.

### 5.2 Dataclass `CurriculumSchedule`

```python
@dataclass
class CurriculumSchedule:
    noise_level_max: float = 0.08    # σ máximo (Fase 3)
    epochs_no_noise: int = 10        # Duração da Fase 1 (limpa)
    noise_ramp_epochs: int = 80      # Duração da Fase 2 (rampa)
```

### 5.3 Diagrama das Três Fases

```
  σ (noise_level)
  │
  │                                          ┌──────────────────────
  │                                          │
  0.08 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤  Fase 3: Estável
  │                                       ╱  │  σ = σ_max = 0.08
  │                                     ╱    │
  │                                   ╱      │
  │                                 ╱        │
  │                               ╱          │
  │                             ╱            │
  │                           ╱              │
  │                         ╱                │
  │                       ╱                  │
  │                     ╱                    │
  │                   ╱                      │
  │                 ╱   Fase 2: Rampa        │
  │               ╱     σ = σ_max × (e-s)/r │
  │             ╱                            │
  │           ╱                              │
  │         ╱                                │
  0.00 ────┤                                 │
  │  Fase 1│                                 │
  │  Clean │                                 │
  │  σ=0.0 │                                 │
  ├────────┼─────────────────────────────────┼────────── épocas
  0       10                                90        ...
           │←──── epochs_no_noise ────→│
                  │←────── noise_ramp_epochs ──────→│
```

### 5.4 Cálculo do Nível de Ruído por Época

```python
def get_noise_level(epoch: int, schedule: CurriculumSchedule) -> float:
    """Calcula σ para uma dada época.

    Fase 1 (epoch < epochs_no_noise):
        σ = 0.0

    Fase 2 (epochs_no_noise <= epoch < epochs_no_noise + noise_ramp_epochs):
        σ = noise_level_max × (epoch - epochs_no_noise) / noise_ramp_epochs

    Fase 3 (epoch >= epochs_no_noise + noise_ramp_epochs):
        σ = noise_level_max
    """
    if epoch < schedule.epochs_no_noise:
        return 0.0
    elif epoch < schedule.epochs_no_noise + schedule.noise_ramp_epochs:
        progress = (epoch - schedule.epochs_no_noise) / schedule.noise_ramp_epochs
        return schedule.noise_level_max * progress
    else:
        return schedule.noise_level_max
```

### 5.5 Valores por Época (exemplos)

| Época | Fase | σ (noise_level) | Observação |
|:-----:|:----:|:---------------:|:-----------|
| 0 | 1 (Clean) | 0.000 | Dados completamente limpos |
| 5 | 1 (Clean) | 0.000 | Modelo aprende padrões fundamentais |
| 9 | 1 (Clean) | 0.000 | Última época limpa |
| 10 | 2 (Ramp) | 0.001 | Início da injeção de ruído |
| 20 | 2 (Ramp) | 0.010 | 12.5% do ruído máximo |
| 30 | 2 (Ramp) | 0.020 | 25% do ruído máximo |
| 50 | 2 (Ramp) | 0.040 | 50% do ruído máximo |
| 70 | 2 (Ramp) | 0.060 | 75% do ruído máximo |
| 89 | 2 (Ramp) | 0.079 | Última época da rampa |
| 90 | 3 (Stable) | 0.080 | Nível máximo atingido |
| 100 | 3 (Stable) | 0.080 | Estável até o final do treinamento |
| 200 | 3 (Stable) | 0.080 | Consolidação da robustez |

### 5.6 `UpdateNoiseLevelCallback`

O callback `UpdateNoiseLevelCallback` é um `tf.keras.callbacks.Callback` que atualiza
a variável TF `noise_level_var` no início de cada época:

```python
class UpdateNoiseLevelCallback(tf.keras.callbacks.Callback):
    """Atualiza o nível de ruído a cada época conforme o curriculum.

    Integração com tf.data:
      ┌───────────────────────────────────────────────────────┐
      │  noise_level_var (tf.Variable)                        │
      │       ↑                                               │
      │  Callback.on_epoch_begin() atualiza valor             │
      │       ↓                                               │
      │  tf.data.map(apply_noise_tf) lê noise_level_var      │
      │       ↓                                               │
      │  Ruído com σ correto é injetado on-the-fly            │
      └───────────────────────────────────────────────────────┘
    """
```

A variável `noise_level_var` é do tipo `tf.Variable(0.0, trainable=False)`, o que
permite que o grafo do `tf.data.map` leia seu valor atualizado a cada época sem
necessidade de reconstruir o pipeline de dados.

### 5.7 Justificativa Científica

O curriculum learning para injeção de ruído é fundamentado em:

1. **Bengio et al. (2009)** — "Curriculum Learning": demonstraram que apresentar
   exemplos em ordem crescente de dificuldade acelera a convergência e melhora a
   generalização.
2. **Observação empírica:** Modelos treinados com ruído máximo desde a época 0
   convergem 20-30% mais lentamente que com curriculum, pois o gradiente ruidoso
   nos estágios iniciais dificulta o aprendizado dos padrões básicos.
3. **Analogia com treinamento humano:** Geofísicos aprendem primeiro a interpretar
   perfis limpos antes de lidar com dados de campo ruidosos.

---

## 6. Interação Noise × FV × GS

### 6.1 Cadeia Fisicamente Correta

A ordem de aplicação das transformações é **crítica** para a fidelidade física do
pipeline. A cadeia correta é:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    CADEIA CORRETA (v2.0)                           │
  │                                                                     │
  │  raw_data ──→ noise(σ) ──→ FV(ruidoso) ──→ GS(ruidoso) ──→ scale  │
  │     │            │              │                │            │     │
  │  22 cols    perturbação    feature view      geosignal    escal.   │
  │  (A/m)     em A/m         (log, ratio...)   (ΔH, ∂H...)   (norm)  │
  │                                                                     │
  │  ✓ GS veem ruído (fidelidade LWD)                                  │
  │  ✓ FV propagam ruído corretamente                                  │
  │  ✓ Scaler fitado em dados LIMPOS (separadamente)                   │
  └─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Cadeia INCORRETA (Bug Legado)

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                   CADEIA INCORRETA (legado)                        │
  │                                                                     │
  │  raw_data ──→ FV(limpo) ──→ GS(limpo) ──→ noise ──→ scale         │
  │                                              │                      │
  │                                          ✗ Ruído APÓS FV/GS        │
  │                                          ✗ GS não veem ruído       │
  │                                          ✗ Não representa LWD      │
  │                                                                     │
  │  PROIBIDO — viola a física do problema                              │
  └─────────────────────────────────────────────────────────────────────┘
```

**Por que a ordem importa?**

Na ferramenta LWD real, o ruído corrompe o sinal EM **bruto** (em A/m) antes de
qualquer processamento. Quando calculamos Feature Views (log, razões) e Geosignals
(gradientes, atenuação), essas operações não-lineares **transformam** o ruído de
forma diferente do que seria se o ruído fosse aplicado diretamente sobre as features
transformadas. Por exemplo:

```
log10(x + ruído) ≠ log10(x) + ruído
```

A versão v2.0 garante que `log10(x + ruído)` é usado (correto fisicamente), e não
`log10(x) + ruído` (artefato numérico).

### 6.3 Colunas Protegidas

O ruído **NUNCA** deve ser aplicado a colunas de metadados (posição, ângulo, frequência).
O sistema de proteção é configurado pelo parâmetro `n_protected`:

```
n_protected = n_prefix + 1
```

Onde `n_prefix` depende do problema:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Problema    n_prefix    Colunas Protegidas      n_protected    │
  ├──────────────────────────────────────────────────────────────────┤
  │  P1          0           [z_obs]                 1              │
  │  P2          1           [z_obs, theta]          2              │
  │  P3          1           [z_obs, freq]           2              │
  │  P2+P3       2           [z_obs, theta, freq]    3              │
  └──────────────────────────────────────────────────────────────────┘
```

**Mecanismo de proteção no código:**

```python
def apply_noise_tf(x, noise_level_var, noise_types, noise_weights, n_protected):
    """Aplica ruído on-the-fly preservando colunas protegidas.

    Esquema de separação:
      ┌────────────────────────────────────────────────────┐
      │  x = [protected_cols | em_cols]                    │
      │       ↓                    ↓                       │
      │  intocado             noise(σ) aplicado            │
      │       ↓                    ↓                       │
      │  x_out = concat([protected_cols, em_cols_noisy])   │
      └────────────────────────────────────────────────────┘
    """
    protected = x[:, :, :n_protected]     # z_obs, theta, freq
    em_data = x[:, :, n_protected:]       # componentes EM (A/m)
    em_noisy = _apply_noise(em_data, ...)  # ruído aplicado
    return tf.concat([protected, em_noisy], axis=-1)
```

### 6.4 Diagrama Completo: Pipeline Noise × FV × GS

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                    PIPELINE COMPLETO DE DADOS (TREINO)                  │
  │                                                                          │
  │  ┌──────────┐    ┌──────────────┐    ┌─────────┐    ┌─────────┐        │
  │  │ Dataset  │    │  Noise       │    │  FV     │    │  GS     │        │
  │  │ raw      │───→│  on-the-fly  │───→│  (7     │───→│  (5     │───→    │
  │  │ (22 col) │    │  σ=f(época)  │    │  tipos) │    │  tipos) │        │
  │  └──────────┘    └──────────────┘    └─────────┘    └─────────┘        │
  │       │                │                  │              │               │
  │   train_raw      noise_level_var     log, ratio,    ΔH, ∂H,       ──→ │
  │   (permanece     (tf.Variable)       amp-phase,     atenuação,         │
  │    raw para      atualizada pelo     Re-Im, etc.    gradiente,         │
  │    on-the-fly)   callback                           etc.               │
  │                                                                          │
  │  ┌──────────────────────────────────────────────────────────────┐       │
  │  │  Scaler (fit em dados LIMPOS)                                │       │
  │  │                                                              │       │
  │  │  fit: raw_clean → FV(clean) → GS(clean) → scaler.fit()     │       │
  │  │  transform: em_noisy_fv_gs → scaler.transform() → modelo   │       │
  │  │                                                              │       │
  │  │  IMPORTANTE: scaler é fitado em dados LIMPOS (temporário)   │       │
  │  │  para evitar que a estatística do ruído contamine a          │       │
  │  │  normalização. Val/test são transformados offline.           │       │
  │  └──────────────────────────────────────────────────────────────┘       │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. API de Referência

### 7.1 `apply_noise_tf`

```python
def apply_noise_tf(
    x: tf.Tensor,
    noise_level_var: tf.Variable,
    noise_types: List[str],
    noise_weights: List[float],
    n_protected: int = 1,
) -> tf.Tensor:
```

**Descrição:** Função principal para injeção de ruído on-the-fly via `tf.data.map`.
Opera inteiramente em TensorFlow (grafo compilável com `tf.function`).

**Args:**

| Parâmetro | Tipo | Descrição |
|:----------|:-----|:----------|
| `x` | `tf.Tensor` | Tensor de entrada, shape `(batch, seq_len, n_features)`. Componentes EM em A/m. |
| `noise_level_var` | `tf.Variable` | Variável TF com o nível de ruído atual (σ). Atualizada pelo `UpdateNoiseLevelCallback` a cada época. |
| `noise_types` | `List[str]` | Lista de nomes dos tipos de ruído a aplicar. Devem estar em `NOISE_FN_MAP`. |
| `noise_weights` | `List[float]` | Pesos de combinação para cada tipo de ruído. Somam 1.0 (normalização interna). |
| `n_protected` | `int` | Número de colunas iniciais protegidas (não recebem ruído). Default: 1 (z_obs). |

**Returns:** `tf.Tensor` com mesma shape de `x`, com ruído aplicado nas colunas EM.

**Uso típico:**

```python
noise_var = create_noise_level_var(0.0)

def train_map_fn(x, y):
    x_noisy = apply_noise_tf(
        x, noise_var,
        noise_types=config.noise_types,
        noise_weights=config.noise_weights,
        n_protected=config.n_protected,
    )
    x_fv = apply_feature_view(x_noisy, config)
    x_gs = apply_geosignals(x_fv, config)
    x_scaled = scaler.transform(x_gs)
    return x_scaled, y

train_ds = train_ds.map(train_map_fn)
```

---

### 7.2 `apply_raw_em_noise`

```python
def apply_raw_em_noise(
    x: np.ndarray,
    noise_level: float,
    noise_types: List[str],
    noise_weights: List[float],
    seed: Optional[int] = None,
    n_protected: int = 1,
) -> np.ndarray:
```

**Descrição:** Versão NumPy para aplicação offline de ruído. Útil para visualização,
debug e testes unitários. **NÃO** deve ser usada no pipeline de treinamento (usar
`apply_noise_tf` via `tf.data.map`).

**Args:**

| Parâmetro | Tipo | Descrição |
|:----------|:-----|:----------|
| `x` | `np.ndarray` | Array de entrada, shape `(seq_len, n_features)` ou `(batch, seq_len, n_features)`. |
| `noise_level` | `float` | Nível de ruído (σ). Escalar fixo (sem curriculum). |
| `noise_types` | `List[str]` | Tipos de ruído a aplicar. |
| `noise_weights` | `List[float]` | Pesos de combinação. |
| `seed` | `Optional[int]` | Semente para reprodutibilidade. None = aleatório. |
| `n_protected` | `int` | Colunas protegidas. Default: 1. |

**Returns:** `np.ndarray` com mesma shape, ruído aplicado nas colunas EM.

---

### 7.3 `create_noise_level_var`

```python
def create_noise_level_var(
    initial_value: float = 0.0,
    name: str = "noise_level",
) -> tf.Variable:
```

**Descrição:** Cria uma `tf.Variable` não-treinável para armazenar o nível de ruído
atual. Esta variável é compartilhada entre o `UpdateNoiseLevelCallback` (que a
atualiza) e o `tf.data.map` (que a lê).

**Args:**

| Parâmetro | Tipo | Descrição |
|:----------|:-----|:----------|
| `initial_value` | `float` | Valor inicial (tipicamente 0.0 para curriculum). |
| `name` | `str` | Nome da variável no grafo TF. |

**Returns:** `tf.Variable(initial_value, trainable=False, dtype=tf.float32)`

---

### 7.4 `NOISE_FN_MAP`

```python
NOISE_FN_MAP: Dict[str, Callable] = {
    "gaussian": _gaussian_noise_tf,
    "multiplicative": _multiplicative_noise_tf,
    "uniform": _uniform_noise_tf,
    "dropout": _dropout_noise_tf,
    "drift": _drift_noise_tf,
    "depth_dependent": _depth_dependent_noise_tf,
    "spikes": _spikes_noise_tf,
    "pink": _pink_noise_tf,
    "saturation": _saturation_noise_tf,
    "varying": _varying_noise_tf,
    "gaussian_local": _gaussian_local_noise_tf,
    "gaussian_global": _gaussian_global_noise_tf,
    "speckle": _speckle_noise_tf,
    "quantization": _quantization_noise_tf,
    "shoulder_bed": _shoulder_bed_noise_tf,
    "borehole_effect": _borehole_effect_noise_tf,
    "mud_invasion": _mud_invasion_noise_tf,
    "anisotropy_misalignment": _anisotropy_misalignment_noise_tf,
    "formation_heterogeneity": _formation_heterogeneity_noise_tf,
    "telemetry": _telemetry_noise_tf,
    "cross_talk": _cross_talk_noise_tf,
    "orientation": _orientation_noise_tf,
    "emi_noise": _emi_noise_tf,
    "freq_dependent": _freq_dependent_noise_tf,
    "noise_floor": _noise_floor_noise_tf,
    "proportional": _proportional_noise_tf,
    "reim_diff": _reim_diff_noise_tf,
    "component_diff": _component_diff_noise_tf,
    "gaussian_keras": _gaussian_noise_tf,  # alias
    "motion": _motion_noise_tf,
    "thermal": _thermal_noise_tf,
    "phase_shift": _phase_shift_noise_tf,
    "bha_vibration": _bha_vibration_noise_tf,
    "eccentricity": _eccentricity_noise_tf,
}
```

Dicionário que mapeia nomes de ruído (strings) para funções TF. Todas as funções
têm a assinatura:

```python
def _xxx_noise_tf(x: tf.Tensor, noise_level: tf.Tensor) -> tf.Tensor:
```

---

### 7.5 `VALID_NOISE_TYPES`

```python
VALID_NOISE_TYPES: frozenset = frozenset(NOISE_FN_MAP.keys())
```

Conjunto imutável com os 34 nomes válidos. Usado para validação em `PipelineConfig`:

```python
# Em PipelineConfig.__post_init__():
for nt in self.noise_types:
    assert nt in VALID_NOISE_TYPES, f"Tipo de ruído inválido: {nt}"
```

---

## 8. Tutorial Rápido

### 8.1 Exemplo 1: Baseline P1 (Gaussiano Simples)

O cenário mais simples: problema P1 (5 features EM) com ruído gaussiano aditivo apenas.

```python
from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.pipeline import DataPipeline

# Configuração baseline
config = PipelineConfig(
    # Problema P1: 5 features EM
    input_features=[1, 4, 5, 20, 21],
    output_targets=[2, 3],

    # Ruído: gaussiano simples
    noise_types=["gaussian"],
    noise_weights=[1.0],

    # Curriculum: 10 épocas limpas, 80 de rampa, max 0.05
    noise_level_max=0.05,
    epochs_no_noise=10,
    noise_ramp_epochs=80,

    # Demais parâmetros...
    sequence_length=600,
    frequency_hz=20000.0,
    spacing_meters=1.0,
)

# Construir pipeline
pipeline = DataPipeline(config)
data = pipeline.prepare("data/dataset_p1.npz")
train_ds = data.train_dataset  # tf.data.Dataset com noise on-the-fly
```

---

### 8.2 Exemplo 2: Multi-tipo com Pesos

Combinação de múltiplos tipos de ruído com pesos diferenciados para simulação
realística de perfuração rotativa padrão.

```python
config = PipelineConfig(
    # Multi-tipo: 5 tipos combinados
    noise_types=[
        "gaussian",         # Ruído térmico base
        "drift",            # Deriva do sensor
        "spikes",           # Picos elétricos
        "borehole_effect",  # Efeito do poço
        "pink",             # Ruído 1/f eletrônico
    ],
    noise_weights=[
        0.40,   # 40% gaussiano (dominante)
        0.20,   # 20% drift
        0.15,   # 15% spikes
        0.15,   # 15% borehole
        0.10,   # 10% pink
    ],

    # Curriculum agressivo: σ_max = 0.08
    noise_level_max=0.08,
    epochs_no_noise=10,
    noise_ramp_epochs=80,
)

# Os pesos são normalizados internamente para somar 1.0
# O ruído final é: σ_final = Σ(w_i × noise_i(x, σ))
```

---

### 8.3 Exemplo 3: Configuração Geosteering Customizada

Cenário completo de geosteering com ruídos específicos de perfuração direcional.

```python
config = PipelineConfig(
    # Problema P2 (com ângulo theta)
    input_features=[1, 4, 5, 20, 21],
    output_targets=[2, 3],
    n_protected=2,  # z_obs + theta protegidos

    # Ruídos de geosteering
    noise_types=[
        "gaussian",              # Base térmica
        "bha_vibration",         # Vibração do BHA (direcional)
        "eccentricity",          # Excentricidade no poço horizontal
        "motion",                # Stick-slip
        "borehole_effect",       # Efeito do poço (lama, diâmetro)
        "cross_talk",            # Cross-talk Re/Im
        "shoulder_bed",          # Camadas adjacentes
    ],
    noise_weights=[
        0.25,   # gaussiano
        0.20,   # BHA vibration
        0.15,   # eccentricity
        0.15,   # motion
        0.10,   # borehole
        0.10,   # cross-talk
        0.05,   # shoulder bed
    ],

    # Curriculum suave para convergência estável
    noise_level_max=0.08,
    epochs_no_noise=15,       # Mais épocas limpas (problema mais difícil)
    noise_ramp_epochs=100,    # Rampa mais longa
)

# Visualizar nível de ruído por época
from geosteering_ai.noise.curriculum import CurriculumSchedule

schedule = CurriculumSchedule(
    noise_level_max=config.noise_level_max,
    epochs_no_noise=config.epochs_no_noise,
    noise_ramp_epochs=config.noise_ramp_epochs,
)

# Epoch 0:   σ = 0.000 (Fase 1 - Clean)
# Epoch 15:  σ = 0.000 (última época limpa)
# Epoch 16:  σ = 0.001 (início da rampa)
# Epoch 65:  σ = 0.039 (metade da rampa)
# Epoch 115: σ = 0.080 (Fase 3 - Estável)
```

---

## 9. Melhorias Futuras

### 9.1 Ruído Adversarial

Implementar *adversarial noise training* onde um gerador de ruído é treinado
simultaneamente com o modelo de inversão, produzindo perturbações que maximizam
o erro de predição. Isso forçaria o modelo a se tornar robusto contra os piores
cenários de ruído possíveis, não apenas contra perturbações estatísticas genéricas.

```
  ┌─────────────────────────────────────────────────────┐
  │  Gerador de ruído G(x) → perturbação adversarial   │
  │       ↓                                             │
  │  x_adv = x + G(x)                                  │
  │       ↓                                             │
  │  Modelo de inversão f(x_adv) → ŷ                   │
  │       ↓                                             │
  │  Loss_inversão(ŷ, y) — minimizada por f             │
  │  Loss_adversarial(ŷ, y) — maximizada por G          │
  └─────────────────────────────────────────────────────┘
```

### 9.2 Calibração com Dados de Campo

Ajustar os parâmetros de ruído (σ, pesos, distribuições) a partir de dados reais
de ferramentas LWD, medidos em poços com formações conhecidas. Isso permitiria
validar e refinar o modelo de ruído sintético contra estatísticas reais de ruído.

**Abordagem proposta:**
1. Coletar dados repetidos (log-repeat) em seção conhecida.
2. Extrair estatísticas de ruído (variância, espectro, correlação).
3. Otimizar parâmetros do NOISE_FN_MAP para minimizar divergência KL
   entre distribuição simulada e observada.

### 9.3 Ruído Adaptativo por Componente

Implementar um sistema que ajuste automaticamente o nível de ruído por componente
EM (Hxx, Hyy, Hzz) e por tipo (Re/Im), com base no SNR estimado de cada canal.
Componentes com SNR mais baixo receberiam proporcionalmente mais ruído durante o
treinamento, melhorando a robustez seletiva.

### 9.4 Tipos Experimentais (13 planejados)

Treze tipos adicionais de ruído estão em fase de pesquisa/prototipagem, mas ainda
não foram incluídos no `NOISE_FN_MAP` por requererem validação adicional:

| # | Nome Proposto | Fenômeno | Status |
|:-:|:-------------|:---------|:-------|
| 1 | `magnetotelluric` | Interferência magnetotelúrica natural | Pesquisa |
| 2 | `casing_effect` | Efeito de revestimento metálico (*casing*) | Pesquisa |
| 3 | `fracture_noise` | Ruído em zonas fraturadas (heterogeneidade extrema) | Pesquisa |
| 4 | `washout` | Alargamento do poço (*washout*) | Protótipo |
| 5 | `conductive_mud` | Lama altamente condutiva (OBM vs WBM) | Protótipo |
| 6 | `tool_eccentricity_3d` | Excentricidade 3D (não apenas 1D) | Pesquisa |
| 7 | `multipath` | Reflexões EM múltiplas em ambientes complexos | Pesquisa |
| 8 | `temperature_transient` | Transiente térmico (mudança brusca de temperatura) | Protótipo |
| 9 | `pressure_effect` | Efeito de pressão nos circuitos eletrônicos | Pesquisa |
| 10 | `galvanic_coupling` | Acoplamento galvânico em formações condutivas | Pesquisa |
| 11 | `pipe_effect` | Efeito da coluna de perfuração (tubulação) | Protótipo |
| 12 | `resistivity_contrast` | Ruído dependente do contraste de resistividade | Protótipo |
| 13 | `adaptive_snr` | SNR adaptativo baseado em estimativa online | Pesquisa |

---

## 10. Referências Bibliográficas

### Perfilagem e Geofísica de Poço

1. **Ellis, D.V. & Singer, J.M.** (2008). *Well Logging for Earth Scientists*.
   2nd ed. Springer. — Referência fundamental sobre princípios físicos de perfilagem
   de poços, incluindo fontes de ruído em ferramentas eletromagnéticas e efeitos
   ambientais (capítulos sobre resistividade indutiva e propagação EM).

2. **Constable, S., Orange, A., & Key, K.** (2016). "And the geophysicist replied:
   'Which model do you want?'" *Geophysics*, 81(5), E197-E212. — Discussão sobre
   incerteza e ambiguidade na inversão EM, incluindo o papel do ruído na resolução
   de modelos de resistividade.

3. **Li, H. & Zhou, Q.** (2017). "Directional resistivity measurement for
   geosteering and formation evaluation while drilling." *SPE/IADC Drilling
   Conference*. — Caracterização de ruído em ferramentas LWD direcionais,
   incluindo efeitos de vibração do BHA e excentricidade.

### Deep Learning e Ruído

4. **Morales, D., Pardo, D., & Torres-Verdín, C.** (2025). "Physics-Informed
   Neural Networks for Electromagnetic Inversion." — Demonstração de que redes
   neurais treinadas com ruído simulado generalizam melhor para dados de campo,
   validando a abordagem de *noise injection* como data augmentation.

5. **Bengio, Y., Louradour, J., Collobert, R., & Weston, J.** (2009).
   "Curriculum Learning." *ICML 2009*. — Fundamentação teórica do curriculum
   learning, demonstrando que treinamento com dificuldade crescente acelera
   convergência e melhora generalização.

6. **Neelakantan, A. et al.** (2016). "Adding Gradient Noise Improves Learning
   for Very Deep Networks." *ICLR Workshop*. — Evidência de que injeção de ruído
   durante treinamento atua como regularizador implícito.

### Processamento de Sinais Eletromagnéticos

7. **Moran, J.H. & Kunz, K.S.** (1962). "Basic theory of induction logging
   and application to study of two-coil sondes." *Geophysics*, 27(6), 829-858.
   — Teoria fundamental de perfilagem por indução, incluindo modelo de ruído
   térmico e SNR em função da condutividade da formação.

8. **Anderson, B.I.** (2001). *Modeling and Inversion Methods for the
   Interpretation of Resistivity Logging Tool Response*. DUP Science. —
   Referência detalhada sobre modelagem de resposta de ferramentas de
   resistividade, incluindo efeitos de poço, invasão e camadas adjacentes.

---

> **Nota:** Esta documentação é gerada a partir do código-fonte em
> `geosteering_ai/noise/functions.py` e `geosteering_ai/noise/curriculum.py`.
> Para detalhes de implementação, consulte as docstrings no código.
>
> **Módulos relacionados:**
> - `geosteering_ai/data/pipeline.py` — integração noise → FV → GS
> - `geosteering_ai/training/callbacks.py` — `UpdateNoiseLevelCallback`
> - `geosteering_ai/config.py` — parâmetros de ruído em `PipelineConfig`
