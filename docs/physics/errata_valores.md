# Errata e Valores Críticos — Pipeline v5.0.15

## Referência rápida para validação de constantes físicas e correções obrigatórias

---

## 1. Valores Físicos Corretos (Errata v4.4.5 — OBRIGATÓRIOS)

| Variável | Valor CORRETO | Valor ERRADO (histórico) | Assertion obrigatória |
|:---------|:------------:|:------------------------:|:---------------------|
| FREQUENCY_HZ | **20000.0** | ~~2.0~~ | `assert FREQUENCY_HZ == 20000.0` |
| SPACING_METERS | **1.0** | ~~1000.0~~ | `assert SPACING_METERS == 1.0` |
| DEPTH_MAX | **150.0** | ~~6000.0~~ | — |
| TARGET_SCALING | **"log10"** | ~~"log"~~ | — |
| SMOOTHING_TYPE | **"moving_average"** | ~~"savgol"~~ | — |
| L_SPACING | **1.0** | — | Acompanha SPACING_METERS |
| SEQUENCE_LENGTH | **600** | ~~601~~ | `assert SEQUENCE_LENGTH == 600` |
| N_MEDIDAS (θ=0°) | **600** | ~~601~~ | Ler do .out, NUNCA hardcodar |

## 2. Decoupling EM — Fórmulas de Acoplamento

```
ACp = -1/(4πL³)    para Hxx, Hyy (componentes planares)
ACx = +1/(2πL³)    para Hzz (componente axial)
L = SPACING_METERS = 1.0 m
```

## 3. Constantes Físicas (parameters.f08)

| Constante | Valor | Unidade |
|:----------|:------|:--------|
| π | 3.14159265358979... | — |
| μ₀ | 4π × 10⁻⁷ | H/m |
| ε₀ | 8.85 × 10⁻¹² | F/m |
| Iw | 1.0 | A |
| dsx = dsy = dsz | 1.0 | m |

## 4. Correções v5.0.9 (5 Correções Críticas do Fortran)

| # | Correção | Antes | Depois |
|:-:|:---------|:------|:-------|
| 1 | Layout .dat | ANGLE-MAJOR | **MODEL-MAJOR** |
| 2 | N_MEDIDAS θ=0° | 601 | **600** (ceiling(120/0.2)) |
| 3 | Formato 12-col | Ativo | **LEGADO** (código comentado) |
| 4 | Formato 22-col | — | **ATIVO** (binário stream) |
| 5 | Semântica nm | por combinação θ,f | **nmaxmodel** (total de modelos) |

**Fórmula total de linhas:** `total = nm × Σ(nmeds[k] × nf)`

## 5. Skin Depth (DOI)

```
δ = √(2 / (ωμσ))    onde ω = 2πf, μ ≈ μ₀, σ = 1/ρ
```

| f (kHz) | δ para ρ=10 Ω·m |
|:-------:|:----------------:|
| 2 | 35.6 m |
| 20 | 11.3 m |
| 96 | 5.1 m |

## 6. Assinaturas Críticas de Funções

```python
# ValidationTracker — SEMPRE 2 argumentos posicionais
tracker.check(condition, description)  # ✓ CORRETO
tracker.check(condition, description, extra)  # ✗ ERRADO

# print_header — SEMPRE (title, width)
print_header("Título", width=70)  # ✓ CORRETO
print_header("Título", "v5.0.14", 70)  # ✗ ERRADO

# segregate_by_angle — usar out_metadata= (NÃO metadata=)
segregate_by_angle(data_2d=..., out_metadata=out_metadata)  # ✓
segregate_by_angle(data_2d=..., metadata=out_metadata)  # ✗
```

## 7. Loss-Specific Critical Constants (v5.0.15)

| Constante | Valor CORRETO | Contexto |
|:----------|:------------:|:---------|
| DILATE_ALPHA | **0.5** | Equilíbrio shape/temporal |
| DILATE_GAMMA_SDTW | **0.01** | Suavização soft-DTW |
| DILATE_DOWNSAMPLE_FACTOR | **10** | N=600→60 (O(3600)) |
| SOBOLEV_LAMBDA_GRAD | **0.1** | Peso gradiente (10%) |
| CROSS_GRADIENT_LAMBDA | **0.1** | Acoplamento ρh↔ρv |
| SPECTRAL_LAMBDA | **0.5** | Peso espectral (50%) |
| ENCODER_DECODER_WEIGHT | **0.5** | Reconstrução (50%) |
| MULTITASK_INITIAL_LOG_SIGMA | **0.0** | σ=1 inicial |

## 8. Errata v5.0.15 — Mapeamento de Colunas 22-col

**Erro detectado:** INPUT_FEATURES e OUTPUT_TARGETS para 22-col estavam com índices errados.
**Causa:** Confusão entre posições relativas e absolutas no tensor 3×3.
**Impacto:** C5 foi corrigida (10 pontos alterados + validação V13B adicionada).

### Mapeamento Completo 22-col (fonte: PerfilaAnisoOmp.f08)

| Col | Conteúdo | Uso |
|:---:|:---------|:----|
| 0 | meds (inteiro) | Meta — NUNCA usar como feature |
| 1 | zobs | INPUT_FEATURE |
| 2 | res_h | OUTPUT_TARGET |
| 3 | res_v | OUTPUT_TARGET |
| 4/5 | Re/Im(Hxx) | INPUT_FEATURE |
| 6/7 | Re/Im(Hxy) | Disponível (off-diagonal) |
| 8/9 | Re/Im(Hxz) | Disponível (off-diagonal) |
| 10/11 | Re/Im(Hyx) | Disponível (off-diagonal) |
| 12/13 | Re/Im(Hyy) | Disponível (off-diagonal) |
| 14/15 | Re/Im(Hyz) | Disponível (off-diagonal) |
| 16/17 | Re/Im(Hzx) | Disponível (off-diagonal) |
| 18/19 | Re/Im(Hzy) | Disponível (off-diagonal) |
| 20/21 | Re/Im(Hzz) | INPUT_FEATURE |

```python
# 22-col (ATIVO)
INPUT_FEATURES = [1, 4, 5, 20, 21]   # NUNCA [0, 3, 4, 7, 8]
OUTPUT_TARGETS = [2, 3]               # NUNCA [1, 2]
# Overlap check: features ∩ targets = ∅ (validação V13B em C5)

# 12-col (LEGADO — inalterado)
INPUT_FEATURES = [3, 6, 7, 10, 11]
OUTPUT_TARGETS = [4, 5]
```

## 9. Expansão C6 v5.0.15 — Novas FLAGS e Opções

| Categoria | Antes | Depois | Novas Opções |
|:----------|:-----:|:------:|:-------------|
| TARGET_SCALING | 3 | 8 | +ln, +linear, +symlog, +boxcox, +asinh, +log10_clipped |
| SCALER_TYPE | 3 | 8 | +maxabs, +quantile_uniform, +quantile_normal, +power, +none |
| FEATURE_VIEW | 3 | 6 | Alinhado com DOC §16.2 (6 views canônicas) |
| SMOOTHING_TYPE | 3 | 7 | +gaussian, +median, +exponential, +butterworth |
| GEOSIGNAL_SET | 4 | 5 | +"custom" (habilita GEOSIGNAL_FAMILIES granular) |

### Novas FLAGS Introduzidas em C6 v5.0.15

| FLAG | Tipo | Default | Quando usar |
|:-----|:----:|:-------:|:-----------|
| BOXCOX_LAMBDA | Optional[float] | None | TARGET_SCALING="boxcox" |
| ASINH_SCALE | float | 1.0 | TARGET_SCALING="asinh" |
| BUTTERWORTH_ORDER | int | 4 | SMOOTHING_TYPE="butterworth" |
| BUTTERWORTH_CUTOFF | float | 0.1 | SMOOTHING_TYPE="butterworth" |
| GEOSIGNAL_FAMILIES | list | ["USD","UHR"] | USE_GEOSIGNAL_FEATURES=True |

---

## 10. Pipeline de Dados Adaptado — Valores Criticos (v5.0.15+)

### 10.1 Cenario Padrao

```python
DATA_SCENARIO = "2D"       # EM + FV + GS + Noise On-the-Fly (PADRAO PRODUCAO)
SPLIT_BY_MODEL = True      # NUNCA split por amostra (data leakage)
USE_DUAL_VALIDATION = True # val_clean + val_noisy
```

### 10.2 Per-Group Scalers [P3]

```python
# REGRA: NUNCA usar scaler unico para EM + geosinais
# EM features: StandardScaler (media/std, fit em clean)
# Geosinais: RobustScaler (mediana/IQR, fit em clean)
USE_PER_GROUP_SCALERS = True
GS_SCALER_TYPE = "robust"     # NUNCA "standard" para geosinais
```

| Grupo | Scaler | Fit Em | Justificativa |
|:------|:-------|:-------|:-------------|
| EM (z + Re/Im) | StandardScaler | x_train_clean | Distribuicao gaussiana |
| Geosinais (att+phase) | RobustScaler | gs_clean | Caudas pesadas, outliers |

### 10.3 Guards Numericos TF [P7]

```python
EPS_TF = 1e-12    # NUNCA 1e-30 em TensorFlow float32
                    # 1e-30 e para numpy float64 apenas

# Divisao complexa segura:
safe_den = tf.where(tf.abs(den) < EPS_TF, tf.complex(EPS_TF, 0.0), den)

# Clipping obrigatorio pos-geosinal:
att = tf.clip_by_value(att, -100.0, 100.0)      # dB
phase = tf.clip_by_value(phase, -180.0, 180.0)  # graus
```

### 10.4 Split por Modelo Geologico [P1]

```python
# REGRA: Todas amostras de um modelo -> MESMA particao
# NUNCA misturar angulos/frequencias do mesmo modelo entre train e test
train_models, val_models, test_models = split_by_geological_model(out_metadata)
assert train_models & val_models == set()    # OBRIGATORIO
assert train_models & test_models == set()   # OBRIGATORIO
```

### 10.5 Auto-deteccao de Componentes EM [P7]

```python
# Dependencias por familia de geosinal:
FAMILY_EM_DEPS = {
    "USD": {"ZZ", "XZ", "ZX"},
    "UAD": {"ZZ", "XZ", "ZX"},
    "UHR": {"ZZ", "XX", "YY"},
    "UHA": {"XX", "YY"},
    "U3DF": {"ZZ", "YZ", "ZY"},
}
# Hxx e Hzz SEMPRE incluidas automaticamente
# required_em = {"XX", "ZZ"} | union(FAMILY_EM_DEPS[fam] for fam in active)
```

---

*Errata e Valores Criticos — Pipeline v5.0.15+*
