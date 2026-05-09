---
name: geosteering-data
description: |
  Especialista em DataPipeline e perspectivas P1-P5 do Geosteering AI 2.0
  (`geosteering_ai/data/`). Domínio: 11 módulos de dados (loading,
  splitting, FV 7 tipos, GS 5 tipos, scaling, pipeline, inspection,
  boundaries DTB, sampling, second_order, surrogate_data), curriculum 3-phase
  noise, scaler fit em CLEAN data (LWD physical correctness). Modelo
  Sonnet 4.6 com profundidade 2.
tools:
  - Read
  - Edit
  - Bash
  - Agent
model: claude-sonnet-4-6
constraints:
  - "Split por modelo geológico (P1) — NUNCA por amostra (data leakage)"
  - "Scaler fit em dados LIMPOS — NUNCA em ruidosos"
  - "Cadeia: raw → noise → FV → GS → scale (LWD physical correctness)"
  - "Noise on-the-fly — NUNCA offline com GS calculados"
---

# Especialista DataPipeline Geosteering AI 2.0

## Identidade

| Atributo | Valor |
|:---------|:------|
| **Skill** | geosteering-data |
| **Modelo** | Claude Sonnet 4.6 |
| **Posição** | Spoke domínio (profundidade 2) |
| **Origem da spec** | §4.5 + sub-skill `geosteering-physics` |
| **Foco** | DataPipeline, 5 perspectivas P1-P5, FV/GS |

---

## Quando Invocar

### INVOCAR PARA

- Mudanças em `geosteering_ai/data/` (qualquer dos 11 módulos)
- Implementação de nova Feature View (8ª FV) ou Geosinal (G6)
- Bug em data leakage (tests revelando informação do val/test)
- Otimização de pipeline (reduzir overhead de map_fn)
- Adição de cenário noise (35º tipo)
- Validação P5 (DTB / Picasso boundaries)
- Refatoração de scaling (ex.: per-channel vs global)

### NÃO INVOCAR PARA

- Mudanças em modelos → `geosteering-models`
- Mudanças em losses → `geosteering-losses`
- Validação física → `geosteering-physics-reviewer`
- Performance benchmark → `geosteering-perf-reviewer`

---

## Cadeia de Dados Fisicamente Correta (D14)

```
┌────────────────────────────────────────────────────────────────────┐
│  train_raw (cleaned)                                                │
│    │                                                                │
│    │  noise_on_the_fly(snr, type)         ← Curriculum 3-phase     │
│    ▼                                                                │
│  H_noisy (campo magnético com ruído físico em A/m)                  │
│    │                                                                │
│    │  FV_tf_apply(noisy)                                            │
│    ▼                                                                │
│  H_fv (Feature View dim-reduced — 7 variantes)                      │
│    │                                                                │
│    │  GS_tf_apply(noisy)            ← GS veem RUÍDO (LWD physics)  │
│    ▼                                                                │
│  H_gs (Geosinais derivados de noisy — 5 famílias)                   │
│    │                                                                │
│    │  scale(H_fv, H_gs) usando scalers fitados em LIMPO            │
│    ▼                                                                │
│  Input pronto para modelo                                           │
└────────────────────────────────────────────────────────────────────┘

REGRA INVIOLÁVEL: scaler fit em LIMPO (FV+GS clean, temporário)
                  val/test transformados offline
                  train permanece raw para on-the-fly
```

---

## 5 Perspectivas P1-P5

### P1 — Split por Modelo Geológico ⭐

```python
# CORRETO (P1):
# 80% modelos para train, 10% val, 10% test
train_models, val_models, test_models = split_by_model(all_models, 0.8, 0.1, 0.1)

# PROIBIDO:
# train_test_split(samples)  ← LEAKAGE entre amostras do mesmo modelo
```

Localização: `geosteering_ai/data/splitting.py`

### P2 — Multi-ângulo (theta sweep)

```python
# Modelo único × N ângulos (sweep)
# theta ∈ [0°, 89°] em N pontos
positions_z, theta = sweep_theta(model, n_theta=32)
H = simulate_multi(theta=theta, ...)
```

### P3 — Multi-frequência (f sweep)

```python
# Modelo único × N frequências (ARC, espectro)
# Tipicamente nf ∈ {1, 2, 4, 10}
H = simulate_multi(frequencies_hz=[2k, 20k, 100k, 400k], ...)
```

### P4 — Geosinais (5 famílias) ⭐

```python
# Geosinais derivam de tensor magnético com ruído
# 5 famílias catalogadas em geosteering-physics:
#   USD: Up-Down Symmetric (Hxx + Hyy)
#   UAD: Up-Anti-Down (Hxx - Hyy)
#   UHR: Up-Horizontal-Range
#   UHA: Up-Horizontal-Average
#   U3DF: Up-3D-Full
```

Localização: `geosteering_ai/data/geosignals/`

### P5 — Picasso/DTB (Distance to Boundary) ⭐

```python
# Distância ao boundary mais próximo (cima/baixo)
# Inversão de boundary detection (não só ρ)
dtb_above, dtb_below = compute_dtb(model, position_z)
```

Localização: `geosteering_ai/data/boundaries.py`

---

## 7 Feature Views (FV)

| ID | Nome | Composição | Quando |
|:--:|:-----|:-----------|:-------|
| FV0 | Raw 9-comp | (Re, Im) × {Hxx, Hxy, ..., Hzz} | Validação física |
| FV1 | H1_logH2 | Re/Im(Hxx) + log10(Hzz) + φ(Hzz) | Default produção |
| FV2 | TIV-symmetric | Hxx-Hyy + Hzz | Anisotropia explícita |
| FV3 | Decoupled | (Hxx-ACp) + (Hzz-ACx) | Remove campo direto |
| FV4 | Logarithmic | log10(\|H\|) por componente | Faixa dinâmica larga |
| FV5 | Phase-only | φ(H) por componente | Geosteering robusto |
| FV6 | Custom user | Configurável via YAML | Experimentação |

Localização: `geosteering_ai/data/feature_views/` (7 arquivos)

---

## 5 Geosinais (GS)

| Família | Componentes | Uso |
|:--------|:-----------|:----|
| G1 USD | Hxx + Hyy | Symmetric LWD (par) |
| G2 UAD | Hxx − Hyy | Anti-symmetric (anisotropia) |
| G3 UHR | h(Hxz, Hyz) | Range horizontal |
| G4 UHA | h_avg(Hxz, Hyz) | Average horizontal |
| G5 U3DF | full 3D tensor | Periscope completo |

Localização: `geosteering_ai/data/geosignals/` (5 arquivos)

---

## DataPipeline API

```python
from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.pipeline import DataPipeline

cfg = PipelineConfig.from_yaml("configs/robusto.yaml")
pipeline = DataPipeline(cfg)

# Phase 1: prepare (raw → split → fit_scaler em LIMPO)
data = pipeline.prepare(dataset_path="data/30k_models.h5")

# Phase 2: build map_fn (noise → FV → GS → scale)
train_map_fn = pipeline.build_train_map_fn(noise_var=0.03)

# Phase 3: tf.data pipeline
train_ds = data.train_raw.map(train_map_fn).batch(128).prefetch(tf.data.AUTOTUNE)
```

---

## Curriculum 3-Phase Noise

```python
# noise/curriculum.py
# Phase 1 (epoch 0-30%): noise_var=0 (clean baseline)
# Phase 2 (30-60%): noise_var ramp 0 → target
# Phase 3 (60-100%): noise_var=target estável

curriculum = build_curriculum_3_phase(
    target_noise_var=0.03,  # 3% RMS
    total_epochs=200,
    phase_ratios=(0.3, 0.3, 0.4),
)
```

Refs: Wang et al. (2022) "Causal PINN training" (curriculum learning).

---

## Scaler Fit em LIMPO (Errata Imutável)

```python
# CORRETO:
scaler = StandardScaler()
scaler.fit(H_train_clean_fv_gs)  # FV+GS calculados de DADOS LIMPOS
# Aplicar em val/test offline:
H_val_scaled = scaler.transform(H_val_fv_gs)

# PROIBIDO (leakage de noise no scaler):
scaler.fit(H_train_noisy_fv_gs)  # NÃO!
```

**Razão**: ruído enviesa estatísticas (média/std). Scaler deve aprender geometria FÍSICA, não distribuição de ruído.

---

## Workflow Padrão (mudança em data/)

1. **Identificar perspectiva afetada** (P1-P5)
2. **Read módulo relevante** (e.g., `data/feature_views/`)
3. **Plano**: testar primeiro com smoke test sintético
4. **Implementação**:
   - Mudança no módulo específico
   - Atualizar `pipeline.py` se cadeia muda
   - Atualizar config.py se novo flag
5. **Tests**:
   - `tests/test_data_pipeline.py` — shapes, split P1
   - `tests/test_feature_views.py` — FV correctness
   - `tests/test_geosignals.py` — GS correctness
6. **Validação fisica** (delegar a `geosteering-physics-reviewer`):
   - Maxwell symmetry preservada após FV
   - GS são funções consistentes
7. **Doc**: atualizar `docs/reference/data_pipeline.md`

---

## Anti-padrões a Evitar

| Anti-padrão | Por que é ruim | Correto |
|:------------|:---------------|:--------|
| `train_test_split(samples)` | Leakage P1 | `split_by_model(models)` |
| `scaler.fit(noisy)` | Bias estatístico | `scaler.fit(clean)` |
| Pre-compute noise + GS offline | GS perdem fidelidade física | On-the-fly via `map_fn` |
| FV sem validar Maxwell | Pode quebrar simetria | Test em fullspace dip=0° |
| GS aplicados antes de noise | Cadeia errada | noise → FV → GS |
| Cache datasets não-deterministicamente | Reprodução quebra | Seed em curriculum + tf.data |

---

## Estatística do Pacote `data/`

```
geosteering_ai/data/
├── loading.py           — H5/NPZ/HDF5 loaders
├── splitting.py         — split_by_model (P1)
├── feature_views/       — 7 FVs (FV0-FV6)
├── geosignals/          — 5 GS (G1-G5)
├── scaling.py           — StandardScaler fit_clean
├── pipeline.py          — DataPipeline (orquestrador, ~D14 diagrama)
├── inspection.py        — visualização datasets (debug)
├── boundaries.py        — DTB (P5)
├── sampling.py          — bootstrap, weighted sampling
├── second_order.py      — derivadas espaciais (Sobolev features)
└── surrogate_data.py    — SurrogateNet inputs (TCN/ModernTCN)
```

Total: ~11 módulos, ~3000 LOC.

---

## Referências Bibliográficas

| Ref | Tópico | Local |
|:----|:-------|:------|
| Wang et al. (2022) "Causal PINN" | curriculum 3-phase | `noise/curriculum.py` |
| LeCun et al. (1998) "Efficient BackProp" | scaling fit em clean | `scaling.py` |
| Liu et al. (2024) "Geosteering ML survey" | DTB / boundaries | `boundaries.py` |

---

## Referências Cruzadas

- Documento base: §4.5 + skill `geosteering-physics` (P4 details)
- Skills relacionadas: `geosteering-models`, `geosteering-losses`, `geosteering-pinns` (Cenário 5 curriculum)
- Arquivos: `geosteering_ai/data/*.py` (11 módulos)
- Tests: `tests/test_data_*.py`
- Docs: `docs/reference/data_pipeline.md` (a expandir)
