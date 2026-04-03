# Catálogo de Ruídos — Pipeline v2.0

## Referencia rapida para C12 (FLAGS) e C24 (implementacao)

---

## 1. Visao Geral

| Categoria              | Qtd | Enabled Default | Descricao                                       |
|:-----------------------|:---:|:---------------:|:------------------------------------------------|
| CORE                   |   9 |       Sim       | Fenomenos fisicos reais na cadeia de medicao LWD |
| EXTENDED               |  13 |       Nao       | Fenomenos LWD secundarios ou operacionais        |
| EXPERIMENTAL           |  13 |       Não       | Baixa relevância para LWD EM ou exploratórios (não implementados no v2.0) |
| Geofísico LWD (R1-R6) |   6 |       Não       | Efeitos geofísicos especializados                |
| Geosteering (R7-R8)   |   2 |       Não       | BHA vibration + eccentricity                     |
| **TOTAL implementado** |**34**|     **9**      | 9 CORE + 12 EXTENDED + 6 LWD + 5 originais + 2 geosteering |
| **TOTAL catalogado**   |**43**|     **9**      | Inclui 13 EXPERIMENTAL não implementados         |

**Nota v2.0:** 34 tipos efetivamente implementados em `noise/functions.py` (NOISE_FN_MAP).
Os 13 EXPERIMENTAL estão catalogados mas NÃO implementados no v2.0.
**Implementação:** `geosteering_ai/noise/functions.py` (apply_raw_em_noise via tf.data pipeline)

---

## 2. CORE — 9 Tipos (Enabled por Padrao)

| #  | Nome             | Tier | Formula                              | Parametros Default                | Fenomeno Fisico               |
|:--:|:-----------------|:----:|:-------------------------------------|:----------------------------------|:------------------------------|
|  1 | varying          |  A   | x + N(0,1) * U(s_min, s_max) * \|x\|| sigma_min=0.01, sigma_max=0.10 | Heteroscedastico legado        |
|  2 | gaussian_local   |  A   | x + N(0, pct * \|x\|)               | pct=0.05                          | Calibracao local              |
|  3 | gaussian_global  |  A   | x + N(0, pct * std_global)           | pct=0.05                          | Erro de medicao global        |
|  4 | speckle          |  A   | x * (1 + N(0, intensity^2))          | intensity=0.05                    | Gain do amplificador          |
|  5 | drift            |  A   | x + cumsum(N(0, std^2)) * phi        | phi=0.95, std=0.01                | Deriva termica da eletronica  |
|  6 | quantization     |  A   | round(x / q) * q                     | bits=16                           | Resolucao ADC                 |
|  7 | saturation       |  A   | clip(x, -x_max, +x_max)             | clip_percentile=99.5              | Saturacao ADC                 |
|  8 | depth_dependent  |  B   | sigma(z) = sigma_0 * (1 + alpha * z) | alpha=0.001, base_sigma=0.03      | Atenuacao EM (skin depth)     |
|  9 | pink             |  B   | FFT -> 1/f^alpha -> IFFT             | alpha=1.0, intensity=0.05         | Flicker noise eletronico      |

---

## 3. EXTENDED — 13 Tipos (Disabled por Padrao)

| #  | Nome             | Tier | Descricao curta                              | Parametros Default                            |
|:--:|:-----------------|:----:|:---------------------------------------------|:----------------------------------------------|
| 10 | cross_talk       |  A   | Acoplamento capacitivo Re/Im                 | epsilon=0.02                                  |
| 11 | orientation      |  A   | Rotacao mandril (mistura Hxx/Hyy)            | delta_deg=2.0                                 |
| 12 | emi_noise        |  A   | EMI 60 Hz + harmonicos do rig                | fundamental=60, n_harmonics=3, amplitude=0.01 |
| 13 | freq_dependent   |  A   | Noise floor ~ f^alpha                        | alpha=0.5, ref_hz=20000.0                     |
| 14 | noise_floor      |  A   | Limite de deteccao do instrumento            | floor_value=1e-8                              |
| 15 | proportional     |  A   | Erro proporcional ~3% do sinal               | level=0.03                                    |
| 16 | reim_diff        |  A   | Im mais ruidoso que Re                       | imag_factor=1.5                               |
| 17 | component_diff   |  A   | Sensibilidades Hxx != Hyy != Hzz             | factors: Hxx=1.0, Hyy=1.2, Hzz=0.8           |
| 18 | gaussian_keras   |  B   | GaussianNoise Keras puro (aditivo)           | stddev=0.05                                   |
| 19 | motion           |  B   | Vibracao BHA via splines aleatorias          | amplitude=0.02, frequency=5.0, velocity=10.0  |
| 20 | thermal          |  B   | Johnson-Nyquist (175 C / 448 K)              | temperature_k=448.0, resistance_ohm=50.0      |
| 21 | spikes           |  B   | Outliers transitorios (5 sigma)              | probability=0.001, magnitude=5.0              |
| 22 | dropouts         |  B   | Valores zerados por falha de comunicacao     | probability=0.001                             |

---

## 4. EXPERIMENTAL — 13 Tipos (Disabled por Padrao)

| #  | Nome                  | Tier | Descricao curta                            |
|:--:|:----------------------|:----:|:-------------------------------------------|
| 23 | uniform               |  B   | Quantizacao simplificada U(-a, +a)         |
| 24 | arma                  |  B   | Autocorrelacao temporal ARMA               |
| 25 | fractal               |  B   | Brownian motion 1/f^2                      |
| 26 | step                  |  B   | Recalibracao abrupta (Heaviside)           |
| 27 | mixture               |  B   | Gaussiano + salt & pepper                  |
| 28 | phase_shift           |  B   | Erro de fase no demodulador                |
| 29 | synthetic_geological  |  B   | Camadas finas nao modeladas (senoidal)     |
| 30 | poisson               |  C   | Shot noise (NAO relevante para EM)         |
| 31 | salt_pepper           |  C   | Outliers binarios min/max                  |
| 32 | lognormal             |  C   | Distribuicao assimetrica                   |
| 33 | rayleigh              |  C   | Envoltoria de ruido I/Q                    |
| 34 | rician                |  C   | Rayleigh + componente deterministica       |
| 35 | spectral_custom       |  C   | Filtro espectral customizado               |

**Tiers:** A = alta relevancia LWD, B = media/situacional, C = baixa relevancia EM

---

## 5. Geofisicos LWD — R1-R6 (v5.0.3)

| ID | Nome                       | Fenomeno                           | Impacto EM                                   | FLAG principal                       |
|:--:|:---------------------------|:-----------------------------------|:---------------------------------------------|:-------------------------------------|
| R1 | Shoulder Bed               | Camadas adjacentes (H leakage)     | Erro em camadas finas (<1 m)                 | USE_SHOULDER_BED_NOISE               |
| R2 | Borehole Effect            | Rugosidade do poco (washout)       | Erro proporcional ao caliper                 | USE_BOREHOLE_EFFECT_NOISE            |
| R3 | Mud Invasion               | Filtrado na formacao               | Altera rho_t aparente (fator 0.8 default)    | USE_INVASION_NOISE                   |
| R4 | Aniso Misalign             | Eixo ferramenta vs TIV             | Mistura Hxx <-> Hzz (delta 2 deg)           | USE_ANISOTROPY_MISALIGNMENT_NOISE    |
| R5 | Formation Heterogeneity    | Variabilidade intra-camada de rho  | **UNICO tipo que altera TARGETS** (5%)       | USE_TARGET_PERTURBATION              |
| R6 | Telemetry                  | Erros MWD/LWD transmissao          | Dropout (0.1%) + bit error (1e-4 BER)        | USE_TELEMETRY_NOISE                  |

> **ATENCAO:** R5 (formation_heterogeneity) e o UNICO tipo que perturba os targets (rho_h, rho_v).
> Todos os outros tipos operam exclusivamente nas features de entrada.

---

## 6. Geosteering — R7-R8 (v5.0.7)

| ID | Nome             | Fenomeno                                    | Parametros Default         | FLAG principal                |
|:--:|:-----------------|:--------------------------------------------|:---------------------------|:------------------------------|
| R7 | BHA Vibration    | Posicao do sensor no BHA — shift via roll    | std=0.5 m                  | USE_BHA_OFFSET_NOISE          |
| R8 | Eccentricity     | Excentricidade da ferramenta no poco         | max=10.0 mm                | USE_TOOL_ECCENTRICITY_NOISE   |

Ambos operam em modo dual (offline + realtime). Ativados tipicamente apenas em cenarios de geosteering.

---

## 7. Curriculum Learning — 3 Fases

```
Fase 1 (epoca 0 -> EPOCHS_NO_NOISE):       dados limpos (noise_level = 0)
Fase 2 (rampa de NOISE_RAMP_EPOCHS epocas): ruido crescente BASE_NOISE -> MAX_NOISE
Fase 3 (estavel):                           ruido no MAX_NOISE (0.075 default)
```

FLAGS de controle (C12 PARTE 2):

| FLAG               | Default | Descricao                                  |
|:-------------------|:--------|:-------------------------------------------|
| USE_CURRICULUM_NOISE | True  | Switch mestre do curriculum                |
| EPOCHS_NO_NOISE    | 15      | Epocas iniciais sem ruido                  |
| NOISE_RAMP_EPOCHS  | 150     | Epocas da rampa crescente                  |
| BASE_NOISE         | 0.005   | Nivel minimo de ruido (inicio da rampa)    |
| MAX_NOISE          | 0.075   | Nivel maximo de ruido (fim da rampa)       |

**Gangorra beta:** quando noise_level sobe, beta_eff sobe (mais peso em suavidade na loss).
Implementado via tf.Variable(current_noise_level) atualizado por UpdateNoiseLevelCallback em C40.

---

## 8. Modos de Combinacao

| NOISE_COMBINATION_MODE | Comportamento                                       |
|:-----------------------|:----------------------------------------------------|
| `"sequential"`         | Empilha tipos na ordem do catalogo                  |
| `"additive"`           | Soma todos os ruidos simultaneamente                |
| `"random_single"`      | Seleciona aleatoriamente UM tipo por batch          |

FLAGS relacionadas: NOISE_MODEL (`"simple"` / `"physical"` / `"hybrid"` / `"custom"`),
NOISE_APPLICATION_MODE (`"coherent"` / `"incoherent"` / `"both"`),
NOISE_APPLICATION_DOMAIN (`"features"` / `"targets"` / `"both"`).

---

## 9. Pipeline de Ruido Adaptado (v5.0.15+)

### 9.1 Cenarios de Ruido — 3 Modos

| Modo | Cenarios | Quando Usar | Curriculum | Fidelidade |
|:-----|:---------|:-----------|:-----------|:-----------|
| Clean (sem ruido) | 1A-1D | Baseline/sanity check | N/A | Baixa |
| **On-the-fly** | **2A-2D** | **Producao (PADRAO)** | **Nativo** | **Alta** |
| Off-line | 3A-3D | Reprodutibilidade exata | Tier-switch | Media |

### 9.2 Ordem de Operacoes com Ruido

**REGRA CRITICA:** Noise DEVE ser aplicado em Re/Im BRUTAS, ANTES de FV e GS.

```
On-the-fly (dentro de tf.data.Dataset.map):
  1. Noise em Re/Im brutas (apply_noise_tf)
  2. Feature View em dados ruidosos (apply_feature_view_tf)
  3. Geosinais do tensor ruidoso (compute_geosignal_tf)  [P7: eps_tf guards]
  4. Scaling per-group (scaler_em + scaler_gs)             [P3: fit em clean]

Off-line (loop em C24):
  for k in range(K_COPIES):
    1. apply_raw_em_noise(x_clean, noise_level=LEVELS[k])
    2. apply_feature_view(x_noisy_k)
    3. compute_geosignal_features(x_noisy_k)
    4. scaler.transform(x_k)
```

### 9.3 Interacao Ruido x FV/GS

| Componente | Impacto do Ruido | Guard Necessario |
|:-----------|:----------------|:----------------|
| Feature View (Re/Im -> mag/phase) | Nao-linear: mag=sqrt(re^2+im^2) amplifica ruido | eps_tf em sqrt |
| USD/UAD (razoes ZZ/XZ/ZX) | Alta sensibilidade: denominador proximo de zero | tf.where guard |
| UHR (-2*ZZ/(XX+YY)) | Media sensibilidade: soma no denominador estabiliza | eps_tf standard |
| UHA (XX/YY) | Media sensibilidade: razao simples | eps_tf standard |
| U3DF (razoes ZZ/YZ/ZY) | Alta sensibilidade: similar a USD | tf.where guard |

### 9.4 Validacao Dual [P2]

```
val_clean_ds: dados limpos, escalados -> monitora generalizacao
val_noisy_ds: mesma noise que train -> monitora robustez

Gap = |val_loss - val_noisy_loss|
  Gap alto: modelo sensivel ao ruido -> aumentar curriculum
  Gap baixo: modelo robusto -> curriculum suficiente
```

### 9.5 FLAGS Novas do Pipeline Adaptado

| FLAG | Default | Celula | Descricao |
|:-----|:--------|:------:|:----------|
| DATA_SCENARIO | "2D" | C4 | Cenario padrao (1A-3D) |
| SPLIT_BY_MODEL | True | C4 | Split por modelo geologico [P1] |
| USE_DUAL_VALIDATION | True | C4 | val_clean + val_noisy [P2] |
| NOISE_GAP_THRESHOLD | 0.5 | C14 | Limiar do gap [P2] |
| OFFLINE_K_COPIES | 5 | C12 | Copias off-line |
| OFFLINE_NOISE_LEVELS | [...] | C12 | Niveis por copia [P5] |

---

*Catalogo de Ruidos — Pipeline v5.0.15+ — 43 tipos + Pipeline Adaptado (P1-P8)*
