# Documentação Técnica: Funções de Perda e PINNs — Geosteering AI v2.0

> **Módulo:** `geosteering_ai/losses/` (`catalog.py`, `factory.py`, `pinns.py`)
> **Versão:** v2.0 — Abril 2026
> **Autor:** Daniel Leal
> **Framework:** TensorFlow 2.x / Keras (EXCLUSIVO)

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Conceitos Fundamentais de Funções de Perda](#2-conceitos-fundamentais-de-funções-de-perda)
3. [Catálogo Completo — 26 Funções de Perda](#3-catálogo-completo--26-funções-de-perda)
4. [Detalhamento por Categoria](#4-detalhamento-por-categoria)
5. [Cenários PINN (Physics-Informed Neural Networks)](#5-cenários-pinn-physics-informed-neural-networks)
6. [TIV Constraint Layer](#6-tiv-constraint-layer)
7. [LossFactory — Montagem de Loss Combinada](#7-lossfactory--montagem-de-loss-combinada)
8. [Tutorial Rápido](#8-tutorial-rápido)
9. [Melhorias Futuras](#9-melhorias-futuras)
10. [Referências](#10-referências)

---

## 1. Visão Geral

O módulo de losses do Geosteering AI v2.0 é responsável por definir as funções
objetivo que guiam o treinamento das redes neurais de inversão 1D de resistividade
eletromagnética. A arquitetura é composta por três submódulos:

| Submódulo      | Arquivo       | Conteúdo                                        |
|:---------------|:--------------|:------------------------------------------------|
| **Catálogo**   | `catalog.py`  | 26 funções de perda (4 categorias)              |
| **Fábrica**    | `factory.py`  | `LossFactory` — montagem e combinação de losses |
| **PINNs**      | `pinns.py`    | 8 cenários de regularização física + TIV Layer  |

### Números-chave

- **26** funções de perda catalogadas (13 genéricas + 4 geofísicas + 2 geosteering + 7 avançadas)
- **8** cenários PINN de regularização informada por física
- **1** camada de restrição TIV (Transverse Isotropy with Vertical axis)
- **`LossFactory`** com dois métodos principais: `get()` (loss única) e `build_combined()` (composição)

### Invariantes do Módulo

```python
# Domínio de operação: TODAS as losses operam em log10
TARGET_SCALING = "log10"    # NUNCA "log"

# Epsilon numérico: seguro para float32
EPS = 1e-12                 # NUNCA 1e-30

# Shapes de entrada esperados:
#   y_true: (batch, seq_len, output_channels)
#   y_pred: (batch, seq_len, output_channels)
# onde output_channels = len(OUTPUT_TARGETS) = 2 (ρ_h, ρ_v em log10)
```

### Diagrama de Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MÓDULO DE LOSSES                              │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │  catalog.py   │   │  factory.py  │   │       pinns.py         │  │
│  │              │   │              │   │                        │  │
│  │  26 losses   │◄──│ LossFactory  │──►│  8 cenários PINN      │  │
│  │  (A/B/C/D)   │   │  get()       │   │  TIVConstraintLayer   │  │
│  │              │   │  build_      │   │  Lambda scheduling    │  │
│  │              │   │  combined()  │   │                        │  │
│  └──────────────┘   └──────────────┘   └────────────────────────┘  │
│         │                  │                       │                │
│         └──────────────────┼───────────────────────┘                │
│                            ▼                                        │
│                   L_total = L_data + λ × L_physics                 │
│                     + w_la × L_look_ahead                          │
│                     + w_dtb × L_dtb                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Conceitos Fundamentais de Funções de Perda

### 2.1. Papel da Loss na Otimização por Gradiente Descendente

A função de perda (loss function) quantifica a discrepância entre a predição do
modelo `ŷ` e o valor verdadeiro `y`. Durante o treinamento, o otimizador (Adam,
SGD, etc.) calcula os gradientes da loss em relação aos parâmetros da rede e
atualiza os pesos na direção que minimiza essa discrepância:

```
θ_{t+1} = θ_t − η × ∇_θ L(y, ŷ)
```

onde `η` é a taxa de aprendizado e `∇_θ L` é o gradiente da loss em relação aos
parâmetros `θ`.

A escolha da loss influencia diretamente:

- **Velocidade de convergência** — gradientes suaves (MSE) convergem mais rápido
  em regiões próximas ao ótimo; gradientes constantes (MAE) convergem mais rápido
  longe do ótimo.
- **Robustez a outliers** — MSE amplifica erros grandes quadraticamente; MAE os
  trata linearmente; Huber combina ambos.
- **Qualidade das interfaces** — losses geofísicas penalizam especificamente a
  suavização excessiva de contrastes de resistividade entre camadas.
- **Estabilidade numérica** — o domínio log10 e o epsilon adequado previnem
  gradientes explosivos ou `NaN`.

### 2.2. O Domínio log10

Todas as losses do Geosteering AI operam no domínio log10 da resistividade.
Isto é fundamental porque a resistividade varia ao longo de várias ordens de
magnitude (tipicamente de 0.1 a 10.000 Ω·m):

```
Domínio linear:    ρ ∈ [0.1, 10000] Ω·m     (5 ordens de magnitude)
Domínio log10:     log10(ρ) ∈ [-1, 4]        (faixa compacta de 5 unidades)
```

**Interpretação de erros no domínio log10:**

| Erro em log10 | Erro relativo em ρ linear | Interpretação                  |
|:--------------:|:-------------------------:|:-------------------------------|
| 0.01           | ~2.3%                     | Excelente — quase exato        |
| 0.05           | ~12.2%                    | Bom — aceitável para produção  |
| 0.10           | ~25.9%                    | Moderado — fronteira de uso    |
| 0.30           | ~99.5%                    | Ruim — erro de quase 2×        |
| 1.00           | ~900%                     | Péssimo — uma ordem errada     |

A relação é: `erro_relativo = 10^(erro_log10) − 1`. Portanto, um RMSE de 0.1
no domínio log10 corresponde a aproximadamente 25% de erro relativo na
resistividade linear.

### 2.3. Comportamento do Gradiente

```
┌───────────────────────────────────────────────────────────────────┐
│  Comportamento do Gradiente por Tipo de Loss                     │
│                                                                   │
│  MSE (quadrática):   ∂L/∂ŷ = 2(ŷ − y)/N                        │
│    → gradiente PROPORCIONAL ao erro                              │
│    → grande perto de outliers, pequeno perto do ótimo            │
│    → convergência rápida final, instável com outliers            │
│                                                                   │
│  MAE (L1):           ∂L/∂ŷ = sign(ŷ − y)/N                     │
│    → gradiente CONSTANTE (±1/N)                                  │
│    → robusto a outliers, convergência lenta perto do ótimo       │
│    → não-diferenciável em ŷ = y (subgradiente)                   │
│                                                                   │
│  Huber (híbrida):    ∂L/∂ŷ = (ŷ−y)/N se |e|≤δ, δ×sign(e)/N    │
│    → quadrática dentro de [-δ, δ], linear fora                   │
│    → combina precisão do MSE com robustez do MAE                 │
│                                                                   │
│  Log-cosh (C∞):      ∂L/∂ŷ = tanh(ŷ − y)/N                     │
│    → suave em toda parte (infinitamente diferenciável)            │
│    → MAE-like para erros grandes, MSE-like para erros pequenos   │
└───────────────────────────────────────────────────────────────────┘
```

### 2.4. Sensibilidade a Outliers e Robustez

Em perfis de resistividade, outliers surgem por:

- **Ruído EM** — especialmente em alta atenuação (camadas condutivas)
- **Efeitos de borda** — transições abruptas entre camadas geram artefatos
- **Dados corrompidos** — falhas instrumentais no LWD

Losses quadráticas (MSE) penalizam outliers com peso `e²`, amplificando sua
influência no gradiente. Para cenários com ruído elevado, losses robustas
(MAE, Huber, log-cosh) são preferíveis.

### 2.5. Necessidade de Losses Geofísicas

Losses genéricas tratam todos os pontos da sequência igualmente. Porém, em
inversão geofísica, existem requisitos específicos:

- **Nitidez de interfaces:** A transição entre camadas geológicas deve ser
  abrupta (descontinuidade de resistividade). MSE tende a suavizar essas
  transições, produzindo perfis "borrados".
- **Supressão de oscilações:** Redes profundas podem gerar oscilações espúrias
  (artefatos de Gibbs) próximo a interfaces. Penalidades de variação total (TV)
  suprimem essas oscilações.
- **Subestimação de resistividade:** Em camadas finas ou muito resistivas,
  modelos tendem a subestimar a resistividade. Penalidades assimétricas
  compensam esse viés.
- **Resolução vertical:** A resolução EM é limitada pela profundidade de skin
  `δ = √(2ρ/ωμ)`. Losses que incorporam esse limite físico são mais realistas.

### 2.6. Regularização Informada por Física

O conceito de PINN (Physics-Informed Neural Network) combina a loss baseada em
dados com um termo de regularização derivado das leis físicas:

```
L_total = L_data(y, ŷ) + λ × L_physics(ŷ)
```

onde `L_physics` pode ser:
- Resíduo de uma PDE (Maxwell, Helmholtz)
- Consistência com o forward model (Surrogate)
- Restrições de suavidade ou continuidade
- Limites físicos de resolução (skin depth)

O parâmetro `λ` controla o peso relativo da física versus dados. Estratégias
de scheduling de `λ` permitem que o modelo primeiro aprenda com os dados e
depois incorpore gradualmente as restrições físicas.

---

## 3. Catálogo Completo — 26 Funções de Perda

### Tabela Resumo

| #  | Nome                       | Cat. | Fórmula (resumida)                                     | Gradiente              | Quando Usar                                    |
|:--:|:---------------------------|:----:|:--------------------------------------------------------|:-----------------------|:-----------------------------------------------|
| 1  | `mse_loss`                 | A    | `mean((y−ŷ)²)`                                         | Quadrático             | Baseline, dados limpos                         |
| 2  | `rmse_loss`                | A    | `√(mean((y−ŷ)²) + ε)`                                  | Quadr. atenuado        | Erro interpretável em log10                    |
| 3  | `mae_loss`                 | A    | `mean(|y−ŷ|)`                                           | Constante (±1/N)       | Dados ruidosos, robustez a outliers            |
| 4  | `mbe_loss`                 | A    | `mean(ŷ−y)`                                             | Constante (+1/N)       | Diagnóstico de viés sistemático                |
| 5  | `rse_loss`                 | A    | `SS_res / (SS_tot + ε)`                                 | Normalizado            | Comparação entre datasets                      |
| 6  | `rae_loss`                 | A    | `Σ|e| / (Σ|y−ȳ| + ε)`                                  | L1 normalizado         | Versão L1 do RSE                               |
| 7  | `mape_loss`                | A    | `mean(|e|/(|y|+ε)) × 100`                               | Inverso proporcional   | Erro percentual, cuidado com y≈0               |
| 8  | `msle_loss`                | A    | `mean((log(y)−log(ŷ))²)`                                | Log-quadrático         | Domínio log-log (dupla compressão)             |
| 9  | `rmsle_loss`               | A    | `√(MSLE + ε)`                                           | Log-quadr. atenuado    | Versão interpretável do MSLE                   |
| 10 | `nrmse_loss`               | A    | `RMSE / (max−min + ε)`                                  | Normalizado por range  | Comparação entre modelos geológicos            |
| 11 | `rrmse_loss`               | A    | `RMSE / (|mean(y)| + ε)`                                | Relativo à média       | Erro relativo à magnitude média                |
| 12 | `huber_loss`               | A    | Híbrida MSE/MAE (δ=1.0)                                | Adaptativo             | Compromisso precisão/robustez                  |
| 13 | `log_cosh_loss`            | A    | `mean(log(cosh(ŷ−y)))`                                  | C∞ suave (tanh)        | Máxima suavidade, sem pontos angulosos         |
| 14 | `log_scale_aware`          | B    | `RMSE + α·Interf + β·Osc + γ·Under`                    | Composto, warmup       | Treino geofísico padrão                        |
| 15 | `adaptive_log_scale`       | B    | #14 + gangorra β(noise)                                 | Adaptativo a ruído     | Curriculum noise crescente                     |
| 16 | `robust_log_scale`         | B    | `Huber + 4 termos + TV`                                 | Robusto + TV           | Dados muito ruidosos com interfaces            |
| 17 | `adaptive_robust`          | B    | `Huber + lógica inversa de ruído`                       | Decrescente com ruído  | Curriculum com redução automática de peso      |
| 18 | `probabilistic_nll`        | C    | NLL Gaussiana (μ + log_var)                             | Incerteza-guiado       | Quantificação de incerteza (UQ)                |
| 19 | `look_ahead_weighted`      | C    | `Σ w[i]·e² com w[i]=exp(−rate·i/N)`                    | Ponderado exponencial  | Geosteering — prioriza profundidades recentes  |
| 20 | `DILATE`                   | D    | `α·Soft-DTW + (1−α)·TDI`                               | Alinhamento temporal   | Sequências com deslocamento temporal           |
| 21 | `encoder_decoder`          | D    | `(1−w)·MSE(pred) + w·MSE(recon)`                       | Dual                   | Regularização por reconstrução                 |
| 22 | `multitask`                | D    | PLACEHOLDER                                             | —                      | Futuro: Kendall uncertainty-weighted           |
| 23 | `sobolev_h1`               | D    | `MSE + λ·MSE(dy/dz, dŷ/dz)`                            | Inclui derivadas       | Suavidade H¹ do perfil                         |
| 24 | `cross_gradient`           | D    | `MSE + λ·mean((∇ρ_h × ∇ρ_v)²)`                        | Acoplamento anisotr.   | Restrição de anisotropia ρ_h/ρ_v              |
| 25 | `spectral`                 | D    | `(1−λ)·MSE + λ·MSE(|FFT|)`                             | Frequência + espaço    | Preservar conteúdo espectral                   |
| 26 | `morales_physics_hybrid`   | D    | `ω·MSE + (1−ω)·MAE, ω(epoch)`                          | Annealing L1→L2        | Inspirado em Morales et al. (2025)             |

**Legenda de categorias:**
- **A** — Genéricas (13 losses)
- **B** — Geofísicas (4 losses)
- **C** — Geosteering (2 losses)
- **D** — Avançadas (7 losses)

---

## 4. Detalhamento por Categoria

### 4.1. Categoria A — Losses Genéricas (13)

#### Motivação Física

As losses genéricas formam a base sobre a qual as demais categorias são
construídas. Elas não incorporam conhecimento específico do domínio geofísico,
mas são essenciais como baselines e como componentes de losses compostas.

No domínio log10, essas métricas adquirem significado físico direto: um MSE
de 0.01 em log10(ρ) corresponde a ~2.3% de erro relativo na resistividade,
enquanto um MSE de 0.1 corresponde a ~25.9%.

#### 4.1.1. MSE — Mean Squared Error (#1)

```
L_MSE = (1/N) × Σᵢ (yᵢ − ŷᵢ)²
```

- **Gradiente:** `∂L/∂ŷᵢ = 2(ŷᵢ − yᵢ) / N` — proporcional ao erro
- **Parâmetros:** Nenhum
- **Quando usar:** Baseline padrão para dados limpos com distribuição aproximadamente
  Gaussiana de erros. Primeiro experimento em qualquer novo dataset.
- **Limitações:** Sensível a outliers; pode suavizar interfaces abruptas.

#### 4.1.2. RMSE — Root Mean Squared Error (#2)

```
L_RMSE = √(mean((y − ŷ)²) + ε)
```

- **Gradiente:** Atenuado em relação ao MSE pelo fator `1 / (2 × RMSE)`
- **Parâmetros:** `ε = 1e-12` (estabilidade numérica)
- **Quando usar:** Quando se deseja erro na mesma unidade de log10(ρ). Um RMSE
  de 0.05 é diretamente interpretável como "erro médio de 0.05 unidades log10".
- **Nota:** Preferível ao MSE para report de métricas; comportamento de
  otimização idêntico (mesmos mínimos).

#### 4.1.3. MAE — Mean Absolute Error (#3)

```
L_MAE = (1/N) × Σᵢ |yᵢ − ŷᵢ|
```

- **Gradiente:** `∂L/∂ŷᵢ = sign(ŷᵢ − yᵢ) / N` — constante em magnitude
- **Parâmetros:** Nenhum
- **Quando usar:** Dados com outliers ou ruído não-Gaussiano. Corresponde ao
  estimador da mediana (vs. MSE que estima a média).
- **Limitações:** Gradiente não-diferenciável em `ŷ = y`; convergência mais lenta
  próximo ao ótimo.

#### 4.1.4. MBE — Mean Bias Error (#4)

```
L_MBE = (1/N) × Σᵢ (ŷᵢ − yᵢ)
```

- **Gradiente:** `∂L/∂ŷᵢ = 1/N` — constante
- **Quando usar:** Não como loss de treinamento, mas como **diagnóstico** de viés
  sistemático. MBE > 0 indica superestimação; MBE < 0 indica subestimação.
- **Nota:** O valor ótimo é MBE = 0, mas isso não garante que as predições
  individuais sejam corretas (erros podem se cancelar).

#### 4.1.5. RSE — Relative Squared Error (#5)

```
L_RSE = Σᵢ (yᵢ − ŷᵢ)² / (Σᵢ (yᵢ − ȳ)² + ε)
```

- **Gradiente:** Normalizado pela variância total dos dados
- **Parâmetros:** `ε = 1e-12`
- **Quando usar:** Comparação entre datasets com variâncias diferentes.
  RSE < 1.0 significa que o modelo é melhor que predizer a média.
  Relação com R²: `R² = 1 − RSE`.

#### 4.1.6. RAE — Relative Absolute Error (#6)

```
L_RAE = Σᵢ |yᵢ − ŷᵢ| / (Σᵢ |yᵢ − ȳ| + ε)
```

- **Gradiente:** L1 normalizado pela dispersão absoluta dos dados
- **Parâmetros:** `ε = 1e-12`
- **Quando usar:** Versão L1 (robusta) do RSE. Mesma interpretação: RAE < 1.0
  indica modelo superior à predição da média.

#### 4.1.7. MAPE — Mean Absolute Percentage Error (#7)

```
L_MAPE = (100/N) × Σᵢ |yᵢ − ŷᵢ| / (|yᵢ| + ε)
```

- **Gradiente:** Inversamente proporcional a `|y|` — amplifica gradientes onde
  `y` é pequeno
- **Parâmetros:** `ε = 1e-12`
- **Quando usar:** Quando se deseja erro em porcentagem. Cuidado: no domínio
  log10, valores próximos de zero (ρ ≈ 1 Ω·m) têm `log10(ρ) ≈ 0`, o que
  pode causar instabilidade. Preferir RMSE ou MAE na maioria dos casos.

#### 4.1.8. MSLE — Mean Squared Logarithmic Error (#8)

```
L_MSLE = (1/N) × Σᵢ (log(yᵢ + 1) − log(ŷᵢ + 1))²
```

- **Gradiente:** Log-quadrático — compressão dupla (já estamos em log10, aplica log natural)
- **Parâmetros:** Nenhum
- **Quando usar:** Cenários extremos onde mesmo o domínio log10 não comprime
  suficientemente a faixa dinâmica. Raramente necessário.

#### 4.1.9. RMSLE — Root Mean Squared Logarithmic Error (#9)

```
L_RMSLE = √(MSLE + ε)
```

- **Parâmetros:** `ε = 1e-12`
- **Quando usar:** Versão interpretável do MSLE.

#### 4.1.10. NRMSE — Normalized RMSE (#10)

```
L_NRMSE = RMSE / (max(y) − min(y) + ε)
```

- **Gradiente:** Normalizado pelo range dos dados
- **Parâmetros:** `ε = 1e-12`
- **Quando usar:** Comparação de desempenho entre modelos geológicos com
  diferentes ranges de resistividade. NRMSE ∈ [0, 1] (idealmente < 0.1).

#### 4.1.11. RRMSE — Relative RMSE (#11)

```
L_RRMSE = RMSE / (|mean(y)| + ε)
```

- **Gradiente:** Normalizado pela magnitude média
- **Parâmetros:** `ε = 1e-12`
- **Quando usar:** Erro relativo à escala média dos dados. Útil quando o range
  é menos informativo que a magnitude típica.

#### 4.1.12. Huber Loss (#12)

```
L_Huber = (1/N) × Σᵢ h(eᵢ)
onde h(e) = 0.5 × e²         se |e| ≤ δ
          = δ × (|e| − 0.5δ)  se |e| > δ
```

- **Gradiente:** Quadrático dentro de `[-δ, δ]`, linear fora
- **Parâmetros:** `δ = 1.0` (default)
- **Quando usar:** Compromisso entre precisão (MSE) e robustez (MAE). O δ
  controla a transição: δ pequeno → mais robusto; δ grande → mais preciso.
  Valor δ=1.0 em log10 equivale a uma ordem de magnitude de resistividade.

#### 4.1.13. Log-Cosh Loss (#13)

```
L_logcosh = (1/N) × Σᵢ log(cosh(ŷᵢ − yᵢ))
```

- **Gradiente:** `∂L/∂ŷᵢ = tanh(ŷᵢ − yᵢ) / N` — suave C∞
- **Parâmetros:** Nenhum
- **Quando usar:** Quando se deseja máxima suavidade do gradiente. Comporta-se
  como MSE para erros pequenos (`|e| ≪ 1`) e como MAE para erros grandes
  (`|e| ≫ 1`), sem a descontinuidade da Huber.

---

### 4.2. Categoria B — Losses Geofísicas (4)

#### Motivação Física

Na inversão de resistividade EM, o perfil de saída deve respeitar características
geológicas específicas:

1. **Interfaces abruptas** entre camadas (folhelho/arenito) devem ser nítidas
2. **Oscilações espúrias** (artefatos de Gibbs) devem ser suprimidas
3. **Subestimação** de camadas resistivas finas deve ser penalizada
4. **Adaptação ao ruído** do curriculum training é necessária

As losses geofísicas incorporam esses requisitos como termos de penalidade
adicionais ao erro base.

#### 4.2.1. Log-Scale Aware Loss (#14)

```
L_lsa = L_RMSE + α × L_interface + β × L_oscillation + γ × L_underestimation
```

Onde:

```
L_interface   = mean(w_interface × |e|)
                w_interface[i] = 1 se |dy/dz[i]| > threshold, 0 caso contrário

L_oscillation = mean(|d²ŷ/dz²|)
                penaliza curvatura excessiva (oscilações de alta frequência)

L_under       = mean(max(0, y − ŷ)²)
                penaliza apenas subestimação (ŷ < y)
```

- **Parâmetros:** `α = 0.3`, `β = 0.1`, `γ = 0.2` (defaults), warmup scheduling
- **Warmup:** Os termos geofísicos são ativados gradualmente nas primeiras
  `N_warmup` epochs para evitar instabilidade inicial
- **Quando usar:** Loss geofísica padrão. Recomendada como primeiro passo além
  do baseline RMSE.

```
Scheduling de warmup:
  ┌────────────────────────────────────────────────┐
  │ epoch 0──────────N_warmup──────────N_total     │
  │   α=0              α=0.3 (fixo)                │
  │   β=0              β=0.1 (fixo)                │
  │   γ=0              γ=0.2 (fixo)                │
  │   ~~~rampa linear~~~│                          │
  └────────────────────────────────────────────────┘
```

#### 4.2.2. Adaptive Log-Scale Loss (#15)

Extensão da #14 com mecanismo de **gangorra** (seesaw): o peso `β` das
oscilações se ajusta automaticamente com base no nível de ruído atual:

```
β_eff = β × (1 − noise_level / noise_max)
```

- **Lógica:** Com ruído alto, oscilações "reais" são mais difíceis de distinguir
  de artefatos. A penalidade de oscilação é reduzida para evitar que o modelo
  suavize demais o sinal ruidoso.
- **Parâmetros:** Herda de #14 + `noise_level_var` (tf.Variable)
- **Quando usar:** Curriculum noise com ruído crescente (3-phase).

#### 4.2.3. Robust Log-Scale Loss (#16)

```
L_robust = L_Huber + α × L_interface + β × L_oscillation
         + γ × L_underestimation + δ_tv × L_TV
```

Onde `L_TV` é a variação total (Total Variation):

```
L_TV = mean(|ŷ[i+1] − ŷ[i]|)
```

- **Base:** Huber (robusta) em vez de RMSE
- **Termo adicional:** TV promove perfis "blocky" (degraus nítidos), ideais
  para modelos geológicos de camadas planas
- **Parâmetros:** `δ_huber = 1.0`, `δ_tv = 0.05` (defaults)
- **Quando usar:** Dados muito ruidosos onde se deseja perfis nítidos.

#### 4.2.4. Adaptive Robust Loss (#17)

Extensão da #16 com lógica **inversa** de adaptação ao ruído:

```
Todas as penalidades REDUZEM com aumento do ruído:
  α_eff = α × (1 − noise_level / noise_max)
  β_eff = β × (1 − noise_level / noise_max)
  γ_eff = γ × (1 − noise_level / noise_max)
```

- **Filosofia:** Com ruído muito alto, a rede deve focar apenas no erro base
  (Huber), sem distração das penalidades geofísicas que podem conflitar com
  o sinal ruidoso.
- **Quando usar:** Curriculum 3-phase com ruído extremo na fase 3.

---

### 4.3. Categoria C — Losses de Geosteering (2)

#### Motivação Física

Em operações de geosteering (guiamento direcional de poço horizontal), o objetivo
não é apenas reconstruir o perfil de resistividade, mas **tomar decisões em tempo
real** sobre a trajetória do poço. Isso implica:

1. **Incerteza quantificada** — saber não apenas `ρ̂` mas também `σ²(ρ̂)` para
   decisões informadas
2. **Priorização temporal** — medições mais recentes são mais relevantes para
   a decisão atual de steering

#### 4.3.1. Probabilistic NLL — Negative Log-Likelihood Gaussiana (#18)

```
L_NLL = (1/2N) × Σᵢ [log(σᵢ²) + (yᵢ − μᵢ)² / σᵢ²]
```

O modelo prediz **dois outputs** por canal: a média `μ` e o log-variância
`log(σ²)`. Isso permite quantificação de incerteza (UQ) nativa.

- **Output shape:** `(batch, seq_len, 2 × output_channels)` — primeira metade é
  `μ`, segunda metade é `log_var`
- **Gradiente:** Forçado a explicar a variância; penaliza tanto erros grandes
  quanto incertezas desnecessariamente largas
- **Parâmetros:** Nenhum adicional (o modelo precisa ser adaptado)
- **Quando usar:** UQ para decisão de geosteering. O `σ` predito indica zonas
  de alta incerteza (interfaces, camadas finas, baixo SNR).

```
Interpretação do output:
  ┌────────────────────────────────────────────────────────┐
  │  Predição com UQ:                                     │
  │                                                        │
  │  ŷ = [μ_ρh, μ_ρv, log_var_ρh, log_var_ρv]           │
  │                                                        │
  │  ρ_h ± σ_h   em log10 → resistividade horizontal     │
  │  ρ_v ± σ_v   em log10 → resistividade vertical       │
  │                                                        │
  │  σ alto → zona de incerteza (decisão cautelosa)       │
  │  σ baixo → predição confiável (decisão agressiva)     │
  └────────────────────────────────────────────────────────┘
```

#### 4.3.2. Look-Ahead Weighted Loss (#19)

```
L_la = Σᵢ w[i] × (yᵢ − ŷᵢ)²
onde w[i] = exp(−rate × i / N)
```

Os pesos decaem exponencialmente ao longo da sequência, priorizando as
**profundidades mais recentes** (mais próximas da broca).

- **Parâmetros:** `rate` — taxa de decaimento exponencial (default: 3.0)
- **Gradiente:** Ponderado — erros recentes contribuem mais para a atualização
- **Quando usar:** Geosteering em tempo real, onde a decisão de steering depende
  principalmente das últimas medições.

```
Perfil de pesos (rate=3.0, N=600):
  w[0]   = 1.000  ← mais recente (maior peso)
  w[150] = 0.472
  w[300] = 0.223
  w[450] = 0.105
  w[599] = 0.050  ← mais antigo (menor peso)
```

---

### 4.4. Categoria D — Losses Avançadas (7)

#### Motivação Física

As losses avançadas incorporam técnicas de vanguarda da literatura de deep learning
e processamento de sinais, adaptadas ao problema de inversão 1D. Incluem
alinhamento temporal (DTW), regularização espectral (FFT), penalidade de gradiente
(Sobolev) e esquemas adaptativos inspirados na literatura recente.

#### 4.4.1. DILATE — DIstortion Loss with Alignment and Temporal Elasticity (#20)

```
L_DILATE = α × L_soft_DTW + (1 − α) × L_TDI
```

Onde:
- **Soft-DTW** (Cuturi, 2013): versão diferenciável do Dynamic Time Warping.
  Mede a similaridade entre sequências permitindo deformações temporais.
- **TDI** (Temporal Distortion Index): penaliza deslocamentos temporais
  residuais após alinhamento.

- **Parâmetros:** `α = 0.5` (balanceamento DTW vs TDI), `γ_sdtw = 0.01`
  (suavização do Soft-DTW)
- **Quando usar:** Quando as predições estão corretas em forma mas deslocadas
  temporalmente (ex.: interfaces preditas 2-3 pontos antes ou depois).
- **Custo computacional:** O(N²) em memória e tempo — usar com cautela em
  sequências longas (`seq_len > 1000`).

#### 4.4.2. Encoder-Decoder Loss (#21)

```
L_ed = (1 − w) × MSE(y, ŷ) + w × MSE(x, x̂)
```

Onde `x̂` é a reconstrução do input pelo decoder (autoencoder regularization).

- **Parâmetros:** `w = 0.1` (peso da reconstrução, default)
- **Quando usar:** Arquiteturas com encoder compartilhado (ex.: UNet). A
  regularização por reconstrução força o encoder a aprender representações
  mais ricas.

#### 4.4.3. Multitask Loss (#22)

```
PLACEHOLDER — Implementação futura
```

Reservado para a loss multi-tarefa de Kendall et al. (2018), que pondera
automaticamente múltiplas losses usando incerteza homoscedástica:

```
L_multi = Σ_t (1/2σ_t²) × L_t + log(σ_t)
```

#### 4.4.4. Sobolev H¹ Loss (#23)

```
L_sobolev = MSE(y, ŷ) + λ × MSE(dy/dz, dŷ/dz)
```

A derivada `dy/dz` é aproximada por diferenças finitas ao longo da dimensão
de profundidade (sequência):

```
dy/dz ≈ (y[i+1] − y[i]) / Δz    (Δz = SPACING_METERS = 1.0 m)
```

- **Parâmetros:** `λ = 0.1` (peso do termo de gradiente, default)
- **Motivação:** Penalizar discrepâncias nos gradientes do perfil — não basta
  que os valores estejam corretos, os **gradientes** (taxa de variação da
  resistividade com a profundidade) também devem coincidir.
- **Quando usar:** Quando as interfaces são o foco principal (detecção de limites
  de camadas).

#### 4.4.5. Cross-Gradient Loss (#24)

```
L_cg = MSE(y, ŷ) + λ × mean((∇ρ_h × ∇ρ_v)²)
```

O produto vetorial dos gradientes de ρ_h e ρ_v penaliza mudanças
descorrelacionadas entre resistividade horizontal e vertical.

- **Parâmetros:** `λ = 0.05` (default)
- **Motivação:** Em meios TIV (isotropia transversal com eixo vertical), as
  variações de ρ_h e ρ_v devem ser estruturalmente correlacionadas — ambas
  mudam nas mesmas interfaces geológicas. O cross-gradient impõe essa restrição.
- **Quando usar:** Inversão anisotrópica (ρ_h ≠ ρ_v), especialmente em
  sequências de folhelhos e arenitos.

#### 4.4.6. Spectral Loss (#25)

```
L_spectral = (1 − λ) × MSE(y, ŷ) + λ × MSE(|FFT(y)|, |FFT(ŷ)|)
```

O termo espectral compara os espectros de amplitude (magnitudes da FFT) dos
perfis verdadeiro e predito.

- **Parâmetros:** `λ = 0.3` (peso espectral, default)
- **Motivação:** Assegurar que o conteúdo de frequência do perfil predito seja
  correto — evita que a rede "invente" oscilações de alta frequência ou
  suavize excessivamente o sinal.
- **Quando usar:** Quando artefatos espectrais (ringing, aliasing) são observados
  nas predições.

#### 4.4.7. Morales Physics-Hybrid Loss (#26)

```
L_morales = ω(epoch) × MSE(y, ŷ) + (1 − ω(epoch)) × MAE(y, ŷ)
```

Com annealing de `ω` ao longo do treinamento:

```
Fase 1 (epochs 0 → T₁):    ω ≈ 0 → dominância MAE (L1, robusto)
Fase 2 (epochs T₁ → T₂):   ω cresce linearmente
Fase 3 (epochs T₂ → fim):   ω ≈ 1 → dominância MSE (L2, preciso)
```

- **Inspiração:** Morales et al. (2025, GJI) — a transição L1→L2 permite que a
  rede primeiro aprenda a estrutura geral (interfaces, via L1 robusto) e depois
  refine os valores (via L2 preciso).
- **Parâmetros:** `T₁`, `T₂` — epochs de transição; acessados via `epoch_var`
  (tf.Variable)
- **Quando usar:** Treino de referência seguindo a metodologia Morales.

```
Annealing ω(epoch):
  ┌────────────────────────────────────────────────────────┐
  │ ω                                                      │
  │ 1.0 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐           │
  │                                    ╱       │           │
  │                                 ╱          │           │
  │                              ╱             │           │
  │ 0.0 ────────────────────╱                  │           │
  │       Fase 1 (L1)  │  Fase 2   │ Fase 3 (L2)         │
  │                    T₁          T₂                      │
  └────────────────────────────────────────────────────────┘
```

---

## 5. Cenários PINN (Physics-Informed Neural Networks)

### 5.1. Conceito Geral

Os cenários PINN adicionam um termo de regularização física à loss de dados:

```
L_total = L_data(y, ŷ) + λ × L_physics(ŷ)
```

O termo `L_physics` não requer labels adicionais — é calculado exclusivamente a
partir das predições `ŷ` e das leis físicas do eletromagnetismo, geologia ou
petrofísica.

### 5.2. Scheduling do λ

O parâmetro `λ` controla o peso relativo da física. Quatro estratégias de
scheduling estão disponíveis, cada uma com 3 fases:

```
┌───────────────────────────────────────────────────────────────────┐
│  Estratégias de Scheduling de λ                                  │
│                                                                   │
│  1. Fixed:   λ constante durante todo o treino                   │
│     ────────────────────────────────                             │
│                                                                   │
│  2. Linear:  λ cresce linearmente de 0 até λ_max                 │
│     ╱──────────────────────────────                              │
│                                                                   │
│  3. Cosine:  λ segue meio ciclo cosseno (suave)                  │
│     ╭───────────────────────────                                 │
│                                                                   │
│  4. Step:    λ = 0 até epoch T, depois λ = λ_max                 │
│     ─────────┐                                                   │
│              └──────────────────                                 │
└───────────────────────────────────────────────────────────────────┘

3 fases comuns:
  Fase 1 — Warmup:  λ = 0 (modelo aprende apenas com dados)
  Fase 2 — Ramp:    λ cresce (introdução gradual da física)
  Fase 3 — Hold:    λ = λ_max (física totalmente ativa)
```

### 5.3. Cenário 1 — Oracle

```
L_oracle = mean((ρ_pred − ρ_reference)²)
```

Compara as predições diretamente com o perfil de referência do Fortran
(forward model exato). Serve como **limite superior** de desempenho — se a
rede não consegue aprender com a loss Oracle, o problema está na arquitetura,
não na loss.

- **Requer:** Perfis de referência do simulador Fortran
- **Uso:** Validação e debugging; NÃO para produção

### 5.4. Cenário 2 — Surrogate

Três modos de complexidade crescente:

```
┌──────────────────────────────────────────────────────────────────┐
│  Surrogate Modes                                                 │
│                                                                   │
│  Mode A — Magnitude |H|:                                         │
│    Forward analítico: |H| = AC × exp(−L/δ)                      │
│    δ = √(2ρ / (ωμ))    (skin depth)                             │
│    AC depende da geometria:                                      │
│      ACp = −1/(4πL³) ≈ −0.079577  (planar: Hxx, Hyy)           │
│      ACx = +1/(2πL³) ≈ +0.159155  (axial: Hzz)                 │
│                                                                   │
│  Mode B — Complex (Re/Im):                                       │
│    H_complex = AC × exp(−(1+j)L/δ)                              │
│    Re(H) = AC × exp(−L/δ) × cos(L/δ)                           │
│    Im(H) = AC × exp(−L/δ) × sin(L/δ)                           │
│                                                                   │
│  Mode C — Neural Forward:                                        │
│    SurrogateNet(ρ_pred) → H_pred                                │
│    L_surr = MSE(H_pred, H_measured)                              │
│    (requer SurrogateNet pré-treinado)                            │
└──────────────────────────────────────────────────────────────────┘
```

**Constantes de decoupling (L = 1.0 m):**

| Componente | AC                    | Valor numérico | Tipo    |
|:-----------|:----------------------|:--------------:|:--------|
| Hxx, Hyy   | −1/(4πL³)             | −0.079577      | Planar  |
| Hzz        | +1/(2πL³)             | +0.159155      | Axial   |

- **Mode A:** Mais simples; usa apenas magnitudes. Adequado para primeiros
  experimentos PINN.
- **Mode B:** Inclui fase; mais informativo mas requer cuidado com wrapping
  de fase.
- **Mode C:** Usa SurrogateNet treinado como forward model diferenciável.
  Máxima fidelidade, mas requer pré-treinamento.

### 5.5. Cenário 3 — Maxwell (Helmholtz PDE)

```
L_maxwell = mean(|d²E/dz² + k²E|²)
```

Onde `k² = jωμσ = jωμ/ρ` é o número de onda ao quadrado.

O campo elétrico `E` deve satisfazer a equação de Helmholtz 1D. O resíduo
desta PDE, calculado a partir de `ρ_pred`, é usado como regularização.

- **Derivadas:** Calculadas por diferenças finitas de segunda ordem
- **Complexidade:** Requer reconstrução de `E` a partir de `ρ_pred` e `H_measured`
- **Quando usar:** Cenário avançado; forte restrição física mas computacionalmente
  intensivo.

### 5.6. Cenário 4 — Smoothness

```
L_smooth = 0.7 × L_Tikhonov + 0.3 × L_TV
```

Combinação de duas regularizações clássicas:

```
L_Tikhonov = mean((dρ_pred/dz)²)     → L2 (suave)
L_TV       = mean(|dρ_pred/dz|)        → L1 (blocky)
```

- **L2 (Tikhonov):** Penaliza gradientes grandes, produzindo perfis suaves
- **L1 (TV):** Penaliza variação total, produzindo perfis em degraus
- **Combinação 70/30:** Compromisso entre suavidade e nitidez
- **Quando usar:** Cenário simples de regularização; não requer forward model.

### 5.7. Cenário 5 — Skin Depth

```
L_skin = mean(max(0, |dρ/dz| − 1/δ)²)
```

Penaliza gradientes de resistividade que excedem o limite de resolução imposto
pela profundidade de skin:

```
δ = √(2ρ / (ωμ₀))

Para f = 20 kHz, ρ = 1 Ω·m:
  δ = √(2 × 1 / (2π × 20000 × 4π × 10⁻⁷))
  δ ≈ 3.56 m
```

- **Interpretação:** A ferramenta LWD não pode resolver variações mais finas
  que ~δ metros. Impor esse limite previne que a rede "invente" detalhes
  abaixo da resolução instrumental.
- **Quando usar:** Para resultados fisicamente realistas em termos de resolução.

### 5.8. Cenário 6 — Continuity

```
L_continuity = mean(|dρ_pred/dz|)
```

Regularização L1 pura — promove perfis **blocky** (constantes por partes),
que são a representação geológica natural de camadas sedimentares.

- **Quando usar:** Modelos geológicos com camadas planas e contrastes nítidos.
- **Diferença do TV na Smoothness:** Aqui é L1 puro (100% TV), sem componente L2.

### 5.9. Cenário 7 — Variational (Deep Ritz)

```
L_variational = ∫ [(dE/dz)² + k²E²] dz    (forma fraca)
```

Baseado no método Deep Ritz (E & Yu, 2018): em vez de minimizar o resíduo
forte da PDE, minimiza o funcional variacional equivalente.

- **Vantagem:** Requer apenas derivadas de primeira ordem (vs. segunda ordem no
  cenário Maxwell), com melhor estabilidade numérica.
- **Quando usar:** Alternativa ao cenário Maxwell com melhor convergência.

### 5.10. Cenário 8 — Self-Adaptive

```
L_adaptive = mean(w(z) × |residual(z)|²)
onde w(z) é aprendido por uma rede de atenção auxiliar
```

Os pesos `w(z)` são aprendidos durante o treinamento, permitindo que a rede
concentre esforço nas regiões de maior resíduo (tipicamente interfaces entre
camadas).

- **Inspiração:** Lu et al. (2021) — self-adaptive PINNs
- **Quando usar:** Cenário mais avançado; útil quando os resíduos são
  heterogeneamente distribuídos ao longo do perfil.

---

## 6. TIV Constraint Layer

### 6.1. Conceito Físico

Em sedimentos com isotropia transversal de eixo vertical (TIV), a resistividade
vertical é sempre maior ou igual à resistividade horizontal:

```
ρ_v ≥ ρ_h    →    log10(ρ_v) ≥ log10(ρ_h)
```

Esta é uma propriedade física fundamental de meios estratificados: correntes
horizontais fluem preferencialmente através das camadas condutivas (path de
menor resistência), enquanto correntes verticais devem atravessar todas as
camadas (incluindo as resistivas).

### 6.2. Implementação como Soft Constraint

Em vez de impor `ρ_v ≥ ρ_h` como hard constraint (que pode causar problemas de
gradiente), a TIVConstraintLayer adiciona uma penalidade suave:

```python
violation = max(0, log10(ρ_h) − log10(ρ_v))
L_tiv = mean(violation²)
```

A penalidade é zero quando a restrição é satisfeita (`ρ_v ≥ ρ_h`) e cresce
quadraticamente com a magnitude da violação.

### 6.3. Razões Típicas ρ_v/ρ_h por Litologia

```
┌────────────────────────────────────────────────────────────────┐
│  Razões de Anisotropia TIV por Litologia                      │
│                                                                │
│  Litologia            │  ρ_v / ρ_h  │  Notas                  │
│  ─────────────────────┼─────────────┼────────────────────────  │
│  Folhelho puro        │  3× — 5×    │  Alta anisotropia        │
│  Folhelho arenoso     │  2× — 5×    │  Variável com cimento    │
│  Arenito limpo        │  ~1×        │  Quasi-isotrópico        │
│  Arenito com argila   │  1.5× — 3×  │  Depende da distribuição │
│  Carbonato            │  1× — 2×    │  Fraturamento aumenta    │
│  Evaporito            │  ~1×        │  Isotrópico              │
└────────────────────────────────────────────────────────────────┘
```

### 6.4. Integração na Loss Total

O termo TIV é adicionado via PipelineConfig:

```python
L_total = L_data + λ_pinn × L_physics + λ_tiv × L_tiv
```

O peso `λ_tiv` é tipicamente menor que `λ_pinn`, pois a restrição TIV é uma
condição necessária, não um constraint forte da PDE.

---

## 7. LossFactory — Montagem de Loss Combinada

### 7.1. Visão Geral

A `LossFactory` é responsável por montar a loss final a partir da configuração
do `PipelineConfig`. Dois métodos principais:

| Método           | Retorno                    | Uso                                      |
|:-----------------|:---------------------------|:-----------------------------------------|
| `get(config)`    | Função de loss única       | Baseline simples (ex.: `"rmse_loss"`)    |
| `build_combined(config)` | Função combinada  | Loss composta com PINN + look-ahead + DTB |

### 7.2. Árvore de Decisão do `build_combined()`

```
build_combined(config)
  │
  ├── config.loss_type == "morales_physics_hybrid"?
  │     └── SIM → Modo exclusivo Morales (retorna direto)
  │           Morales gerencia internamente o annealing L1→L2
  │           NÃO combina com outros termos
  │
  └── NÃO → Modo combinado padrão:
        │
        ├── 1. Loss base = LossFactory.get(config.loss_type)
        │     peso = w_base = max(1.0 − w_la − w_dtb, 0.01)
        │
        ├── 2. Look-ahead? (config.use_look_ahead)
        │     └── SIM → + w_la × look_ahead_weighted
        │
        ├── 3. DTB? (config.use_dtb)
        │     └── SIM → + w_dtb × dtb_loss
        │
        └── 4. PINN? (config.use_pinns)
              └── SIM → + λ_pinn × L_physics(cenário)
```

### 7.3. Normalização de Pesos

Os pesos são normalizados para garantir estabilidade:

```python
w_la  = config.look_ahead_weight      # ex.: 0.2
w_dtb = config.dtb_weight             # ex.: 0.1
w_base = max(1.0 - w_la - w_dtb, 0.01)  # ex.: 0.7 (mínimo 0.01)
```

A soma `w_base + w_la + w_dtb` é sempre ≤ 1.0, com `w_base` tendo piso de 0.01
para garantir que a loss base nunca seja completamente anulada.

### 7.4. Variáveis Dinâmicas

A `LossFactory` injeta `tf.Variable` para parâmetros que mudam durante o treino:

| Variável             | Tipo         | Atualizado por          | Usado em                    |
|:---------------------|:-------------|:------------------------|:----------------------------|
| `epoch_var`          | `tf.int32`   | Callback de época       | Warmup, Morales annealing   |
| `noise_level_var`    | `tf.float32` | Curriculum scheduler    | Adaptive losses (#15, #17)  |
| `pinns_lambda_var`   | `tf.float32` | Lambda scheduler        | Peso PINN                   |

Essas variáveis são atualizadas externamente (via callbacks) e lidas internamente
pelas losses, permitindo comportamento adaptativo sem recompilação do grafo.

---

## 8. Tutorial Rápido

### 8.1. Exemplo 1 — Baseline Simples com RMSE

```python
from geosteering_ai.config import PipelineConfig
from geosteering_ai.losses.factory import LossFactory

# Configuração mínima
config = PipelineConfig(
    loss_type="rmse_loss",
    use_pinns=False,
    use_look_ahead=False,
    use_dtb=False,
)

# Obter loss
loss_fn = LossFactory.get(config)

# Usar no modelo
model.compile(optimizer="adam", loss=loss_fn)
```

### 8.2. Exemplo 2 — Loss Geofísica com Warmup

```python
from geosteering_ai.config import PipelineConfig
from geosteering_ai.losses.factory import LossFactory
import tensorflow as tf

# Configuração com loss geofísica
config = PipelineConfig(
    loss_type="log_scale_aware",
    use_pinns=False,
    use_look_ahead=False,
    use_dtb=False,
)

# Variável de época para warmup
epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False)

# Obter loss (a factory injeta epoch_var automaticamente)
loss_fn = LossFactory.get(config)

# Compilar e treinar
model.compile(optimizer="adam", loss=loss_fn)

# No callback de época, atualizar:
# epoch_var.assign(current_epoch)
```

### 8.3. Exemplo 3 — Loss Combinada com PINN + Look-Ahead + DTB

```python
from geosteering_ai.config import PipelineConfig
from geosteering_ai.losses.factory import LossFactory

# Configuração completa
config = PipelineConfig(
    loss_type="robust_log_scale",
    
    # PINN — cenário Surrogate Mode A
    use_pinns=True,
    pinn_scenario="surrogate",
    pinn_surrogate_mode="A",
    pinn_lambda=0.01,
    pinn_lambda_schedule="cosine",
    
    # Look-ahead para geosteering
    use_look_ahead=True,
    look_ahead_weight=0.2,
    look_ahead_rate=3.0,
    
    # DTB (Distance to Boundary)
    use_dtb=True,
    dtb_weight=0.1,
)

# Montar loss combinada
# L_total = 0.7 × robust_log_scale
#         + 0.2 × look_ahead_weighted
#         + 0.1 × dtb_loss
#         + λ(epoch) × L_surrogate_A
loss_fn = LossFactory.build_combined(config)

model.compile(optimizer="adam", loss=loss_fn)
```

### Diagrama da Loss Combinada do Exemplo 3

```
┌─────────────────────────────────────────────────────────────────┐
│                    L_total (Exemplo 3)                          │
│                                                                 │
│  ┌─────────────────────────┐                                    │
│  │  robust_log_scale (0.7) │──┐                                 │
│  │  Huber + 4 termos + TV  │  │                                 │
│  └─────────────────────────┘  │                                 │
│                               │                                 │
│  ┌─────────────────────────┐  │     ┌─────────┐                │
│  │  look_ahead (0.2)       │──┼────►│ L_total │                │
│  │  exp decay weights      │  │     └─────────┘                │
│  └─────────────────────────┘  │          ▲                      │
│                               │          │                      │
│  ┌─────────────────────────┐  │          │                      │
│  │  dtb_loss (0.1)         │──┘          │                      │
│  │  boundary proximity     │             │                      │
│  └─────────────────────────┘             │                      │
│                                          │                      │
│  ┌─────────────────────────┐             │                      │
│  │  Surrogate A (λ cosine) │─────────────┘                      │
│  │  |H| = AC × exp(−L/δ)  │                                    │
│  └─────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Melhorias Futuras

### 9.1. Kendall Multi-Task Uncertainty (#22)

Completar a implementação placeholder `multitask` com a loss de Kendall et al.
(2018), que pondera automaticamente múltiplas tarefas usando incerteza
homoscedástica aprendida:

```
L_multi = Σ_t (1 / 2σ_t²) × L_t + log(σ_t)
```

Os pesos `1/σ_t²` são aprendidos durante o treino, permitindo que tarefas com
maior incerteza recebam menor peso automaticamente.

**Aplicação no Geosteering AI:** Ponderar automaticamente a loss de ρ_h vs. ρ_v,
já que a inversão de resistividade horizontal é tipicamente mais bem-condicionada
que a vertical.

### 9.2. Evidential Regression Loss

Loss baseada em distribuições de evidência (Amini et al., 2020), que modela a
incerteza epistêmica e aleatória simultaneamente usando uma distribuição
Normal-Inverse-Gamma:

```
L_evidential = L_NLL + λ × L_evidence_regularizer
```

**Vantagem sobre NLL Gaussiana:** Distingue incerteza do modelo (epistêmica) de
incerteza dos dados (aleatória), sem necessidade de MC Dropout ou ensembles.

### 9.3. Adversarial Training Loss

Incorporar um discriminador que distingue perfis reais de preditos:

```
L_total = L_data + λ_adv × L_adversarial
L_adversarial = −log(D(ŷ))    (generator loss)
```

**Motivação:** Perfis gerados por redes de inversão frequentemente carecem de
"realismo geológico" — são corretos em média mas não parecem perfis reais.
Um discriminador treinado em perfis reais pode forçar maior realismo.

### 9.4. Multi-Frequency Loss Weighting

Para ferramentas LWD multi-frequência, ponderar a loss por frequência:

```
L_multi_freq = Σ_f w_f × L_data(y_f, ŷ_f)
```

Onde `w_f` é maior para frequências com melhor SNR. Frequências mais altas
(shallow reading) teriam peso diferente de frequências mais baixas (deep reading).

---

## 10. Referências

### Artigos Científicos

1. **Morales, G. et al.** (2025). "Physics-Informed Neural Networks for
   Electromagnetic Inversion of Well-Log Data." *Geophysical Journal
   International (GJI)*. — Inspiração para a loss `morales_physics_hybrid`
   (#26) e para os cenários PINN Surrogate.

2. **Cuturi, M.** (2013). "Sinkhorn Distances: Lightspeed Computation of
   Optimal Transport." *Advances in Neural Information Processing Systems
   (NeurIPS)*. — Base teórica para o Soft-DTW usado na loss DILATE (#20).

3. **Kendall, A., Gal, Y., & Cipolla, R.** (2018). "Multi-Task Learning
   Using Uncertainty to Weigh Losses for Scene Geometry and Semantics."
   *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
   — Framework para a futura loss multitask (#22).

4. **Bai, J., Rabczuk, T., Gupta, A., Alzubaidi, L., & Gu, Y.** (2022).
   "A Physics-Informed Neural Network Technique Based on a Modified Loss
   Function for Computational 2D and 3D Solid Mechanics." *Computational
   Mechanics*. — Tutorial e boas práticas para PINNs.

5. **E, W. & Yu, B.** (2018). "The Deep Ritz Method: A Deep Learning-Based
   Numerical Method for Solving Variational Problems." *Communications in
   Mathematics and Statistics*. — Base para o cenário PINN Variational (#7).

6. **Lu, L., Meng, X., Mao, Z., & Karniadakis, G.E.** (2021). "DeepXDE:
   A Deep Learning Library for Solving Differential Equations." *SIAM Review*.
   — Referência para PINNs self-adaptive (#8).

7. **Amini, A., Schwarting, W., Soleimany, A., & Rus, D.** (2020). "Deep
   Evidential Regression." *Advances in Neural Information Processing Systems
   (NeurIPS)*. — Referência para futura evidential regression loss.

### Módulos Fonte

| Arquivo                           | Conteúdo                                          |
|:----------------------------------|:--------------------------------------------------|
| `geosteering_ai/losses/catalog.py`| 26 funções de perda (Categorias A/B/C/D)          |
| `geosteering_ai/losses/factory.py`| `LossFactory` — get() e build_combined()          |
| `geosteering_ai/losses/pinns.py`  | 8 cenários PINN + TIVConstraintLayer              |
| `geosteering_ai/config.py`        | `PipelineConfig` — todos os hiperparâmetros       |

### Documentação Relacionada

| Documento                          | Relação                                           |
|:-----------------------------------|:--------------------------------------------------|
| `docs/ARCHITECTURE_v2.md`          | Arquitetura geral do pacote                       |
| `docs/reference/noise_catalog.md`  | Catálogo de ruído (34 tipos)                      |
| `docs/physics/`                    | Contexto físico (tensor EM, GS, FV)               |
| `CLAUDE.md`                        | Regras e proibições do projeto                    |

---

> **Gerado automaticamente pela documentação técnica do Geosteering AI v2.0.**
> **Última atualização:** Abril 2026.
