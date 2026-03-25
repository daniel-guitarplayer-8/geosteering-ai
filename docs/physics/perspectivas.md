# Perspectivas P2–P5 — Resumo de FLAGS e Interações

## Pipeline de Inversão Geofísica com Deep Learning v5.0.15

---

## 1. Visão Geral

| P# | Nome | Versão | O Que Adiciona | N_FEATURES |
|:--:|:-----|:------:|:---------------|:----------:|
| P1 | Baseline | v5.0.1 | z + 4 componentes EM (Re/Im H1, Re/Im H2) | 5 |
| P2 | θ como feature | v5.0.8 | Ângulo de inclinação injetado | 6 |
| P3 | f como feature | v5.0.12 | Frequência injetada | 6–7 |
| P4 | Geosinais | v5.0.13 | USD, UAD, UHR, UHA, U3DF | 9–17 |
| P5 | Picasso/DTB | v5.0.15 | Validação DOD + DTB como target | — |

**N_FEATURES máximo teórico:** 17 (P2+P3+P4 full_3d)

## 2. Perspectiva 2 — θ como Feature (v5.0.8)

### FLAGS (C5)
```python
USE_MULTI_ANGLE: bool = False           # Switch mestre
USE_THETA_AS_FEATURE: bool = False      # Injetar θ como feature
MULTI_ANGLE_STRATEGY: str = "segregate" # "segregate" | "padded"
ANGLE_BATCH_MODE: str = "homogeneous"   # "homogeneous" | "mixed"
ANGLE_PAD_VALUE: float = 0.0
USE_ANGLE_MASK: bool = False
ANGLE_SAMPLING_WEIGHTS: list = None
PRIMARY_ANGLE_INDEX: int = 0
```

### Mecanismo
- **22-col (ATIVO):** θ NÃO existe no .dat → injetado por `segregate_by_angle(inject_theta=True)`
- **12-col (LEGADO):** θ na coluna 2 → adicionado ao INPUT_FEATURES
- N_MEDIDAS varia com θ: 600 (0°), 622 (15°), 693 (30°), 1200 (60°)

### Retrocompatibilidade
`USE_MULTI_ANGLE=False` → v5.0.7

## 3. Perspectiva 3 — f como Feature (v5.0.12)

### FLAGS (C5)
```python
USE_FREQ_AS_FEATURE: bool = False       # Switch mestre
FREQ_NORMALIZATION: str = "log10"       # "log10" | "khz" | "mhz" | "raw"
_FREQ_INJECT_MODE: str = "none"         # Derivada automática
```

### Mecanismo
- **22-col:** f injetado externamente por `segregate_by_angle(inject_freq=True)`
- **12-col:** f da coluna 1 adicionada ao INPUT_FEATURES
- Normalização recomendada: log10 (20kHz→4.301, 96kHz→4.982)
- Otimização: `freq_norm_cache` pré-computa normalização FORA dos loops

### Interação com P2
Ortogonais e complementares. Ordem: [θ] [f_norm] [z] [Re(H1)] [Im(H1)] [Re(H2)] [Im(H2)]

### Retrocompatibilidade
`USE_FREQ_AS_FEATURE=False` → v5.0.11

## 4. Perspectiva 4 — Geosinais (v5.0.13)

### FLAGS (C6)
```python
USE_GEOSIGNAL_FEATURES: bool = False    # Switch mestre
GEOSIGNAL_SET: str = "usd_uhr"         # "usd_uhr" | "usd_uhr_uha" | "full_1d" | "full_3d" | "custom"
GEOSIGNAL_MODE: str = "hybrid"          # "pure" | "hybrid" | "append"
GEOSIGNAL_CONVENTION: str = "geosphere" # "geosphere" | "tatu"
GEOSIGNAL_FAMILIES: list = ["USD","UHR"]# Seleção granular (v5.0.15) — ativo quando SET="custom"
                                        # ou auto-populado a partir do SET escolhido
```

### 5 Famílias de Geosinais
| Família | Sensibilidade | Fórmula | Disponível |
|:--------|:-------------|:--------|:-----------|
| USD | Boundaries | (ZZ+XZ)/(ZZ−XZ) × (ZZ−ZX)/(ZZ+ZX) | 22-col apenas |
| UAD | Dip/anisotropia | (ZZ+XZ)/(ZZ−XZ) × (ZZ+ZX)/(ZZ−ZX) | 22-col apenas |
| UHR | Resistividade bulk | −2·ZZ/(XX+YY) | 22-col e 12-col |
| UHA | Anisotropia | XX/YY | 22-col e 12-col |
| U3DF | Indicador 3D | (ZZ+YZ)/(ZZ−YZ) × (ZZ−ZY)/(ZZ+ZY) | 22-col apenas |

### Estratégia Recomendada
**B (Híbrida):** manter componentes brutas + adicionar geosinais
→ N_FEATURES = 9 (z + 4 EM + USDA + USDP + UHRA + UHRP)

### Auto-config C5/C6
```python
# Auto-população SET → FAMILIES (C6):
_SET_TO_FAMILIES = {
    "usd_uhr":     ["USD", "UHR"],
    "usd_uhr_uha": ["USD", "UHR", "UHA"],
    "full_1d":     ["USD", "UAD", "UHR", "UHA"],
    "full_3d":     ["USD", "UAD", "UHR", "UHA", "U3DF"],
    "custom":      None,  # usa GEOSIGNAL_FAMILIES diretamente
}
# 12-col auto-filtering: apenas {"UHR", "UHA"} disponíveis (falta off-diagonal)

# Contagem de features (C5):
_n_geosignal_features = _compute_geosignal_count(GEOSIGNAL_FAMILIES)
N_FEATURES = len(INPUT_FEATURES) + _n_inject_external + _n_geosignal_features
```

### Retrocompatibilidade
`USE_GEOSIGNAL_FEATURES=False` → v5.0.12

## 5. Perspectiva 5 — Picasso Plots + DTB (v5.0.15)

### FLAGS (C14/C15/C10)
```python
# Picasso Plots
USE_PICASSO_PLOT: bool = False
PICASSO_R_MIN: float = 0.1
PICASSO_R_MAX: float = 1000.0
PICASSO_R_STEPS: int = 50
PICASSO_THRESHOLD_ATT: float = 0.25     # dB
PICASSO_THRESHOLD_PS: float = 1.5       # graus
PICASSO_GEOSIGNALS: list = ["USDA", "USDP", "UHRA", "UHRP"]
USE_SENSITIVITY_PLOT: bool = False

# DTB como target
USE_DTB_AS_TARGET: bool = False
DTB_MAX_FROM_PICASSO: float = 3.0       # metros
DTB_SCALING: str = "linear"             # "linear" | "log" | "normalized"
```

### Picasso Plots — 5 Usos
1. Validação da P4 (geosinais vs literatura)
2. Prior constraints para DTB (upper bound = DOD máximo)
3. Análise de erro condicionada ao cenário (Rt1×Rt2)
4. Visualização diagnóstica (erro DL sobreposto ao Picasso)
5. Feature engineering informada (blind spots → componentes brutas)

### DTB como Target
- C20 computa DTB_up e DTB_down por ponto de medição
- OUTPUT_CHANNELS = 6: [ρh, ρv, DTB_up, DTB_down, ρ_up, ρ_down]
- L_DTB com clipping em [0, DTB_MAX_FROM_PICASSO]

### Retrocompatibilidade
`USE_PICASSO_PLOT=False` + `USE_DTB_AS_TARGET=False` → v5.0.13

## 5B. Losses Geosteering Avançado (v5.0.15)

### FLAGS (C10)
```python
USE_DILATE_LOSS: bool = False             # DILATE (shape + temporal)
DILATE_ALPHA: float = 0.5                 # Balanço shape/temporal
DILATE_GAMMA_SDTW: float = 0.01           # Suavização soft-DTW
DILATE_DOWNSAMPLE_FACTOR: int = 10        # N=600→60

USE_ENCODER_DECODER_LOSS: bool = False    # Forward surrogate
ENCODER_DECODER_WEIGHT: float = 0.5       # Peso reconstrução

USE_MULTITASK_LEARNED_LOSS: bool = False  # Kendall 2018
MULTITASK_INITIAL_LOG_SIGMA: float = 0.0  # σ=1 inicial

USE_SOBOLEV_LOSS: bool = False            # Gradient matching
SOBOLEV_LAMBDA_GRAD: float = 0.1          # Peso gradiente

USE_CROSS_GRADIENT_LOSS: bool = False     # ρh↔ρv coupling
CROSS_GRADIENT_LAMBDA: float = 0.1        # Peso coupling

USE_SPECTRAL_LOSS: bool = False           # FFT matching
SPECTRAL_LAMBDA: float = 0.5             # Peso espectral
```

### Interação com P4/P5
- **DILATE + P4:** Geosinais como features + tolerância a shifts → melhor tracking de boundaries
- **Sobolev + P5 (DTB):** Gradient matching preserva transições → DTB labels mais precisos
- **Cross-Gradient + P4:** Acoplamento ρh↔ρv complementa geosinais direccionais (USD/UAD)
- **Spectral:** Captura padrões de frequência espacial em perfis de resistividade
- **Encoder-Decoder:** Requer surrogate neural do simulador Fortran (futuro)

### Retrocompatibilidade
Todas `USE_*=False` → v5.0.14 (zero impacto em fluxos existentes)

## 6. Interações entre Perspectivas

```
     P2 (θ)  ←→  P3 (f)   : Ortogonais, ambas injetadas
       ↓           ↓
     P4 (geosinais)        : Usa componentes brutas como entrada
       ↓                     Cross-Gradient acopla ρh↔ρv (USD/UAD)
     P5 (Picasso/DTB)      : Valida P4, define constraints para DTB
       ↓                     Sobolev preserva transições DTB
     Losses Avançadas       : DILATE (shape+temporal), Spectral (FFT),
                              Multitask (Kendall), Encoder-Decoder (futuro)
```

Todas aplicam-se a AMBOS os modos (offline e realtime).
θ e f são propriedades da ferramenta → NÃO violam causalidade.
Losses avançadas (§5B) são opt-in → `USE_*=False` preserva v5.0.14.

## 7. Interação FV/GS com Ruído On-the-Fly (v5.0.15+)

### Pipeline Adaptado — Cenário 2D (Padrão)

Quando noise on-the-fly está ativo (cenários 2x), Feature Views e Geosinais
NÃO podem ser computados estaticamente. Devem ser recalculados APÓS o ruído.

```
Cadeia CORRETA (on-the-fly, dentro de tf.data.map):
  raw Re/Im → noise → FV → GS → Scale → Modelo

Cadeia ERRADA (estática, antes do tf.data):
  raw Re/Im → FV → GS → Scale → noise → Modelo
  ↑ FV e GS nunca veem ruído → viés sistemático
```

### Auto-detecção de Componentes EM para Geosinais

Quando `USE_GEOSIGNAL_FEATURES=True` + noise on-the-fly, o dataset
DEVE armazenar TODAS as componentes EM necessárias:

```python
required_em = {"XX", "ZZ"}  # SEMPRE obrigatórias (Hxx e Hzz)
for fam in GEOSIGNAL_FAMILIES:
    required_em |= FAMILY_EM_DEPS[fam]

EXPANDED_INPUT_FEATURES = compute_expanded_input_features(
    INPUT_FEATURES, GEOSIGNAL_FAMILIES, _COL_MAP_22
)
# Ex: USD+UHR → {XX,ZZ,XZ,ZX,YY} → cols [1,4,5,8,9,12,13,16,17,20,21]
```

### Per-Group Scalers [P3]

Geosinais (atenuação em dB + fase em graus) têm distribuição
fundamentalmente diferente de features EM (Re/Im com ordens de
magnitude variáveis). Usar scaler único DEGRADA a normalização.

| Grupo | Scaler | Fit Em | Justificativa |
|:------|:-------|:-------|:-------------|
| EM (z + Re/Im) | StandardScaler | x_train_clean (pós-FV) | Dist. ~gaussiana |
| Geosinais (att+phase) | RobustScaler | gs_clean (pré-noise) | Caudas pesadas |

### Impacto nas Perspectivas

| Perspectiva | Impacto do Pipeline Adaptado |
|:------------|:----------------------------|
| P2 (θ) | Nenhum — θ é coluna injetada, não EM |
| P3 (f) | Nenhum — f é coluna injetada, não EM |
| P4 (GS) | **MAIOR IMPACTO** — GS recomputados on-the-fly, auto-detecção EM, per-group scalers |
| P5 (DTB) | DTB é target, não feature — não afetado diretamente |

### Novas FLAGS (v5.0.15+)

```python
DATA_SCENARIO: str = "2D"              # Cenário padrão (1A-3D)
SPLIT_BY_MODEL: bool = True            # Split por modelo geológico [P1]
USE_DUAL_VALIDATION: bool = True       # val_clean + val_noisy [P2]
USE_PER_GROUP_SCALERS: bool = True     # Scalers separados EM/GS [P3]
GS_SCALER_TYPE: str = "robust"         # RobustScaler para GS [P3]
EPS_TF: float = 1e-12                  # Guard numérico TF [P7]
EXPORT_INFERENCE_PIPELINE: bool = True # Serializa FV+GS+scalers [P6]
NOISE_GAP_THRESHOLD: float = 0.5      # Limiar gap dual val [P2]
```

---

*Perspectivas P2–P5 — Pipeline v5.0.15+*
