---
name: geosteering-losses
description: |
  Sub-skill de loss functions do Geosteering AI v2.0. Catálogo completo das 26 losses em
  4 categorias (A: 13 genéricas, B: 4 geofísicas, C: 2 geosteering, D: 7 avançadas),
  API do LossFactory (get + build_combined), 8 cenários PINN com residuos físicos e
  constraints de anisotropia. Guia completo de seleção por problema/regime.
  Use para questões sobre qual loss usar, como combinar losses, PINNs, LossFactory.
  Triggers: "loss", "função de perda", "MSE", "RMSE", "MAE", "Huber", "log_scale",
  "log_cosh", "DILATE", "Sobolev", "spectral", "morales", "probabilistic", "look_ahead",
  "multitask", "cross_gradient", "LossFactory", "PINN", "physics loss", "residuo físico",
  "constraint anisotropia", "gangorra", "combined loss", "build_combined", "VALID_LOSS_TYPES".
  v2.0 | Última atualização: 2026-04-29 | Sub-skill da geosteering-v2
---

# Geosteering AI v2.0 — Catálogo de Loss Functions (26 Losses)

> **Sub-skill especializada** — Loss functions e PINNs.
> Skill principal: `geosteering-v2` | Física: `geosteering-physics` | Código: `geosteering-code-v2` | Modelos: `geosteering-models`

---

## 1. Visão Geral

### 1.1 Arquivos

| Arquivo | Conteúdo |
|:--------|:---------|
| `geosteering_ai/losses/catalog.py` | 26 loss functions (46.2 KB) |
| `geosteering_ai/losses/factory.py` | `LossFactory.get()` + `build_combined()` |
| `geosteering_ai/losses/pinns.py` | 8 cenários PINN (106.0 KB) |

### 1.2 As 26 Losses — Sumário por Categoria

```
A. Genéricas (13):  mse, rmse, mae, mbe, rse, rae, mape, msle, rmsle, nrmse, rrmse, huber, log_cosh
B. Geofísicas (4):  log_scale_aware, adaptive_log_scale, robust_log_scale, adaptive_robust
C. Geosteering (2): probabilistic_nll, look_ahead_weighted
D. Avançadas (7):   dilate, enc_decoder, multitask, sobolev, cross_gradient, spectral, morales_physics_hybrid
```

---

## 2. Categoria A — Genéricas (13 losses)

### 2.1 Losses Base

| Nome | Fórmula | Quando Usar |
|:-----|:--------|:------------|
| `mse` | mean((ŷ - y)²) | Default baseline, sensível a outliers |
| `rmse` | sqrt(mean((ŷ - y)²)) | Mesma escala que targets, mais interpretável |
| `mae` | mean(\|ŷ - y\|) | Robusto a outliers, convergência mais lenta |
| `mbe` | mean(ŷ - y) | Diagnóstico de bias sistemático |
| `rse` | sum((ŷ-y)²) / sum((ȳ-y)²) | Normalizado por variância do target |
| `rae` | sum(\|ŷ-y\|) / sum(\|ȳ-y\|) | Normalizado por desvio médio do target |

### 2.2 Losses de Escala Relativa

| Nome | Fórmula | Quando Usar |
|:-----|:--------|:------------|
| `mape` | mean(\|ŷ-y\|/\|y\|) × 100 | Erro percentual — problemático perto de 0 |
| `msle` | mean((log(1+ŷ) - log(1+y))²) | Penaliza sub-estimação mais que super-estimação |
| `rmsle` | sqrt(msle) | Versão interpretável do MSLE |
| `nrmse` | rmse / (y_max - y_min) | RMSE normalizado pelo range do target |
| `rrmse` | rmse / mean(\|y\|) | RMSE relativo à magnitude média |

### 2.3 Losses Robustas

| Nome | Fórmula | Quando Usar |
|:-----|:--------|:------------|
| `huber` | δ²(√(1+(e/δ)²)-1) | Robusto a outliers, diferenciável em 0 |
| `log_cosh` | mean(log(cosh(ŷ-y))) | Suavização de MAE, robusto e diferenciável |

**Huber vs. MAE:** Huber usa approximação quadrática próximo de 0 (∝ MSE) e linear longe de 0
(∝ MAE). Parâmetro δ controla a transição. Default δ=1.0 para resistividade em escala log10.

---

## 3. Categoria B — Geofísicas (4 losses)

Losses projetadas especificamente para a escala logarítmica de resistividade:

### 3.1 `log_scale_aware`

```
L = mean(w_i × (ŷ_i - y_i)²)

w_i = 1 + alpha × |y_i|   (pesos proporcionais à magnitude do target)

Motivação: Resistividade em log10 → erros em valores altos (carbonatos >1000 Ohm.m)
têm impacto geológico MAIOR que erros equivalentes em valores baixos (folhelhos 1-10 Ohm.m).
```

### 3.2 `adaptive_log_scale`

```
L = mean((ŷ - y)² / (|y| + eps))

Divisão pela magnitude: normaliza o erro pela escala local do target.
Equivale a minimizar erro relativo em escala log.
```

### 3.3 `robust_log_scale`

```
L = mean(Huber(ŷ - y) × (1 + |y|))

Combina robustez do Huber com sensibilidade à escala logarítmica.
Melhor quando há outliers no conjunto de treinamento.
```

### 3.4 `adaptive_robust`

```
Combina adaptive_log_scale + Huber para lidar com:
  - Faixa dinâmica larga (4+ ordens de magnitude de resistividade)
  - Outliers geofísicos (frente de invasão, zonas de alta contraste)
  - Gradientes estáveis em toda a escala log10
```

**Quando usar losses geofísicas:** Sempre que `TARGET_SCALING = "log10"` (padrão).
As losses da categoria B são projetadas para a faixa [-1, +4] do log10(rho).

---

## 4. Categoria C — Geosteering (2 losses)

### 4.1 `probabilistic_nll` — Negative Log-Likelihood

```
L = mean(log(σ²) + (ŷ_mean - y)²/σ²)

Modelo produz: ŷ_mean (previsão) + ŷ_var (variância aleatória)
output_channels deve ser dobrado: 2 → 4 (mean_h, mean_v, var_h, var_v)

Uso: InferencePipeline em modo probabilístico
     Quantificação de incerteza (UQ) end-to-end
```

### 4.2 `look_ahead_weighted` — Look-Ahead Ponderado

```
L = sum_t(w_t × (ŷ_t - y_t)²)

w_t = exp(gamma × (t - T_current))   (pesos maiores para amostras futuras)
gamma: taxa de decaimento (padrão: 0.1)

Motivação: Em geosteering, prever resistividade FRENTE à broca (look-ahead)
é mais valioso que prever resistividade já perfurada (look-around).
Ref: Constable et al. (2016) — look-ahead EM detection 5–30m
```

**Quando usar:** Quando o objetivo é maximizar a acurácia de look-ahead para decisões
de trajetória, sacrificando ligeiramente a acurácia retrospectiva.

---

## 5. Categoria D — Avançadas (7 losses)

### 5.1 `dilate` — Dynamic Time Lag

```
L_dilate = alpha × L_shape + (1-alpha) × L_temporal

L_shape:    correlação de forma via DTW (Dynamic Time Warping)
L_temporal: penaliza deslocamentos temporais (lag) na predição
alpha:      balanço shape vs. temporal (padrão: 0.5)

Ref: Chang et al. (2019) "DILATE: DIscriminative Loss for shApe and Time positioning"
```

**Uso:** Quando o modelo tende a produzir predições corretas em forma mas deslocadas no tempo.
Comum em inversões de zonas de transição abrupta (fronteiras de camada).

### 5.2 `enc_decoder` — Encoder-Decoder Loss

```
L = L_recon + beta × L_latent

L_recon:  reconstrução do target no espaço de saída
L_latent: regularização do espaço latente (variância do encoder)
```

### 5.3 `multitask` — Multi-Task Learning

```
L = w_h × L(rho_h) + w_v × L(rho_v) + sum_k(w_k × L_k)

Combina losses de múltiplos targets (rho_h, rho_v, DTB, etc.)
com pesos aprendidos ou fixos.
```

### 5.4 `sobolev` — Sobolev H1

```
L = ||ŷ - y||²_L2 + lambda × ||∇ŷ - ∇y||²_L2

Penaliza diferenças no gradiente temporal além das diferenças de valores.
Garante suavidade do perfil de resistividade predito.

Ref: Czarnecki et al. (2017) — Sobolev training para redes neurais
```

**Uso:** Quando o perfil predito apresenta oscilações espúrias entre amostras consecutivas.

### 5.5 `cross_gradient` — Cross-Gradient

```
L = ||ŷ - y||² + mu × ||(∇rho_h × ∇rho_v)||

Penaliza quando gradientes de rho_h e rho_v são perpendiculares.
Promove que as duas curvas de resistividade tenham transições na mesma profundidade.

Motivação física: rho_h e rho_v descrevem a MESMA rocha — devem ter fronteiras
de camada nas mesmas posições no perfil.
```

### 5.6 `spectral` — Spectral Loss

```
L = L_time + gamma × L_freq

L_time: loss padrão no domínio temporal
L_freq: MSE no domínio de frequência (via FFT)

Penaliza diferenças espectrais, capturando padrões de frequência no perfil.
Útil quando o modelo perde texturas periódicas (variaçõetricamente regulares de camadas).
```

### 5.7 `morales_physics_hybrid` — Híbrida Morales 2025

```
L = L_data + lambda_aniso × L_anisotropy + lambda_smooth × L_smooth + lambda_bc × L_bc

L_data:       MSE nos targets (rho_h, rho_v)
L_anisotropy: ||max(0, rho_h - rho_v)||² (constraint TIV: rho_v >= rho_h)
L_smooth:     Sobolev H1 (suavidade do perfil)
L_bc:         Condição de contorno (resistividade nas bordas)

Ref: Morales et al. (2025) "Anisotropic resistivity estimation using PINN with uncertainty"
```

**Esta é a loss mais completa do projeto.** Combina fidelidade aos dados com 3 constraints físicos.

---

## 6. API do LossFactory

### 6.1 `LossFactory.get(config)` — Loss Base

```python
from geosteering_ai.losses.factory import LossFactory

# Uso básico:
loss_fn = LossFactory.get(config)
# Equivalente a: resolver config.loss_type → função ou closure compilada

# Exemplos:
config = PipelineConfig(loss_type="rmse")
loss_fn = LossFactory.get(config)   # → função rmse_loss direta

config = PipelineConfig(loss_type="dilate", dilate_alpha=0.5)
loss_fn = LossFactory.get(config)   # → closure make_dilate(config)

# Uso no modelo:
model.compile(optimizer="adam", loss=loss_fn)
```

### 6.2 `LossFactory.build_combined(config)` — Loss Combinada (Gangorra)

```python
# A loss combinada empilha múltiplas losses com pesos:
loss_fn = LossFactory.build_combined(config)

# Fluxo interno:
# 1. loss_base = LossFactory.get(config)           ← loss principal
# 2. loss_combined = loss_base
# 3. Se config.use_look_ahead_loss:
#      loss_combined += config.look_ahead_weight × look_ahead_weighted(config)
# 4. Se config.output_channels > 2:                 ← DTB ativo (P5)
#      loss_combined += config.dtb_weight × dtb_loss(config)
# 5. Se config.use_physics_loss:                    ← PINN ativo
#      loss_combined += config.physics_weight × pinn_loss(config)

# Exemplo: loss completa para geosteering com DTB e PINN
config = PipelineConfig(
    loss_type="log_scale_aware",
    use_look_ahead_loss=True, look_ahead_weight=0.3,
    output_channels=4,        dtb_weight=0.2,
    use_physics_loss=True,    physics_weight=0.1,
)
loss_fn = LossFactory.build_combined(config)
```

### 6.3 VALID_LOSS_TYPES — Lista Completa

```python
from geosteering_ai.losses.factory import VALID_LOSS_TYPES

VALID_LOSS_TYPES = [
    # A: Genéricas
    "mse", "rmse", "mae", "mbe", "rse", "rae",
    "mape", "msle", "rmsle", "nrmse", "rrmse",
    "huber", "log_cosh",
    # B: Geofísicas
    "log_scale_aware", "adaptive_log_scale",
    "robust_log_scale", "adaptive_robust",
    # C: Geosteering
    "probabilistic_nll", "look_ahead_weighted",
    # D: Avançadas
    "dilate", "enc_decoder", "multitask",
    "sobolev", "cross_gradient", "spectral",
    "morales_physics_hybrid",
]
```

---

## 7. PINNs — Physics-Informed Neural Networks

### 7.1 Formulação Geral

```python
L_total = L_data + lambda_physics × L_physics

L_data:    loss padrão (qualquer das 26)
L_physics: residuo de equações físicas ou constraints geofísicas
lambda_physics: peso do termo físico (hyperparâmetro, default: 0.1)
```

### 7.2 8 Cenários PINN (losses/pinns.py)

| # | Cenário | L_physics | Motivação |
|:-:|:--------|:----------|:----------|
| 1 | **Difusão EM** | Resíduo ∇²H = iωμσH | Equação física da propagação EM |
| 2 | **Suavidade** | \|\|∇rho_h\|\|² + \|\|∇rho_v\|\|² | Perfis fisicamente contínuos |
| 3 | **Anisotropia TIV** | \|max(0, rho_h - rho_v)\|² | rho_v ≥ rho_h em TIV |
| 4 | **Consistência decoupling** | \|H_meas - H_form - ACp\|² | Decoupling inverso verificado |
| 5 | **Condição de fronteira** | \|rho(z=0) - rho_ref\|² | BC na superfície |
| 6 | **Cross-gradient** | \|(∇rho_h) × (∇rho_v)\| | Fronteiras coincidentes |
| 7 | **Archie constraint** | Penaliza Sw > 1 ou Sw < 0 | Física petrofísica |
| 8 | **Morales completo** | Cenários 2 + 3 + 5 + 6 | Inversão com UQ (Morales 2025) |

**Arquivo:** `losses/pinns.py` (106 KB — o arquivo mais extenso do projeto).

### 7.3 Configuração PINN

```python
config = PipelineConfig(
    loss_type="rmse",               # loss de dados
    use_physics_loss=True,           # ativar PINN
    physics_scenario=3,              # cenário de anisotropia TIV
    physics_weight=0.1,              # lambda_physics
    anisotropy_weight=1.0,           # peso específico do constraint TIV
)
loss_fn = LossFactory.build_combined(config)
```

---

## 8. Guia de Seleção de Loss

### 8.1 Por Regime de Treinamento

| Regime | Loss Recomendada | Alternativa |
|:-------|:----------------|:-----------|
| Baseline rápido | `rmse` | `mse` |
| Produção padrão | `log_scale_aware` | `adaptive_log_scale` |
| Robustez a outliers | `robust_log_scale` | `huber` |
| Geosteering realtime | `log_scale_aware` + `look_ahead_weighted` | `rmse` + `look_ahead_weighted` |
| Com DTB (P5) | `log_scale_aware` + multitask | `morales_physics_hybrid` |
| Inversão física | `morales_physics_hybrid` | `log_scale_aware` + PINN cenário 8 |
| UQ probabilística | `probabilistic_nll` | qualquer + MC Dropout |
| Suavidade de perfil | `sobolev` | `log_scale_aware` + PINN cenário 2 |

### 8.2 Decisão Rápida

```
1. Há UQ (incerteza quantificada)?
   SIM → probabilistic_nll ou qualquer + config.mc_dropout_samples > 0

2. Há targets adicionais (DTB)?
   SIM → multitask ou LossFactory.build_combined(config, use_dtb=True)

3. Há constraints físicas importantes?
   SIM → morales_physics_hybrid ou PINN (cenários 3, 6, 8)

4. Perfil com oscilações espúrias?
   SIM → sobolev ou spectral

5. Perfil OK mas atrasado (lag)?
   SIM → dilate

6. DEFAULT (sem necessidades especiais):
   → log_scale_aware (melhor que rmse/mse para resistividade em log10)
```

### 8.3 Combinações Comprovadas

```python
# Configuração E-Robusto S21 (validada em produção):
config = PipelineConfig.robusto()
# loss_type="log_scale_aware", physics_weight=0.0, sem look_ahead

# Geosteering completo com DTB:
config = PipelineConfig(
    loss_type="log_scale_aware",
    output_channels=4,          # rho_h, rho_v, DTB_up, DTB_down
    use_look_ahead_loss=True,
    look_ahead_weight=0.3,
    dtb_weight=0.2,
)

# Pesquisa com constraints físicos:
config = PipelineConfig(
    loss_type="morales_physics_hybrid",
    use_physics_loss=True,
    physics_scenario=8,         # Morales completo
    physics_weight=0.1,
)
```

---

## 9. Target Scaling — Impacto nas Losses

As losses operam sobre os **targets já escalonados** (pós `TARGET_SCALING = "log10"`).

| Scaling | Range de rho (Ohm.m) | Range log10 | Impacto na loss |
|:--------|:-------------------:|:-----------:|:----------------|
| **log10** (default) | 0,1 – 10.000 | -1 a +4 | Faixa [5 unidades], MSE bem condicionado |
| `none` (bruto) | 0,1 – 10.000 | — | MSE dominado por valores altos (>1000) |
| `log` (ln) | 0,1 – 10.000 | -2,3 a +9,2 | INCORRETO para o pipeline — PROIBIDO |

**Por que `log` é PROIBIDO:** Legado usava ln (numpy) e log10 (TF) inconsistentemente.
O scaler foi fitado em log10 — usar `"log"` cria incompatibilidade silenciosa entre
scaler.inverse_transform e a escala real do modelo.

---

*Sub-skill de loss functions — Geosteering AI v2.0 | Última atualização: 2026-04-29*
