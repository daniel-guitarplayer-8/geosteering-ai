# Modelo Geológico — Referência Técnica Completa

## Geosteering AI v2.0 · Pipeline de Inversão 1D de Resistividade

---

## Índice

1. [Definição e Conceito Fundamental](#1-definição-e-conceito-fundamental)
2. [Origem dos Dados — Simulador Fortran](#2-origem-dos-dados--simulador-fortran)
3. [Estrutura do Arquivo `.dat` e `.out`](#3-estrutura-do-arquivo-dat-e-out)
4. [Layout de 22 Colunas por Medição](#4-layout-de-22-colunas-por-medição)
5. [Estrutura de Dados 3D no Pipeline](#5-estrutura-de-dados-3d-no-pipeline)
6. [Split por Modelo Geológico — Princípio P1](#6-split-por-modelo-geológico--princípio-p1)
7. [O Modelo Geológico no Dataset e no tf.data](#7-o-modelo-geológico-no-dataset-e-no-tfdata)
8. [Container DataSplits](#8-container-datasplits)
9. [Metadados — OutMetadata e AngleGroup](#9-metadados--outmetadata-e-anglegroup)
10. [Decoupling EM — Remoção do Acoplamento Direto](#10-decoupling-em--remoção-do-acoplamento-direto)
11. [Cadeia Completa de Carregamento](#11-cadeia-completa-de-carregamento)
12. [Valores Físicos Críticos e Errata](#12-valores-físicos-críticos-e-errata)
13. [Referências no Código](#13-referências-no-código)
14. [Tabela de Referência Rápida](#14-tabela-de-referência-rápida)

---

## 1. Definição e Conceito Fundamental

Um **modelo geológico**, no contexto do Geosteering AI, é um **cenário sintético de distribuição
de resistividade elétrica** ao longo de uma coluna de profundidade de poço, gerado pelo simulador
numérico Fortran **PerfilaAnisoOmp**.

Em termos físicos, é uma configuração específica de camadas geológicas — folhelhos, arenitos,
carbonatos, anidrita, etc. — com seus respectivos valores de:

- **ρ_h** — resistividade horizontal (Ohm·m)
- **ρ_v** — resistividade vertical (Ohm·m)

O simulador resolve as equações de Maxwell para a geometria de ferramenta LWD especificada e
calcula qual seria a **resposta eletromagnética** (tensor H em A/m) medida pela ferramenta em
cada ponto de profundidade ao longo daquele perfil geológico.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODELO GEOLÓGICO — Fluxo Conceitual                                        │
│                                                                             │
│  Entrada (perfil de resistividade):                                         │
│    Camada 1: z = 0–30 m   → ρ_h = 5 Ω·m   (folhelho)                     │
│    Camada 2: z = 30–80 m  → ρ_h = 100 Ω·m  (arenito saturado de óleo)    │
│    Camada 3: z = 80–150 m → ρ_h = 8 Ω·m   (folhelho)                     │
│                                ↓                                            │
│  Simulador Fortran (PerfilaAnisoOmp):                                       │
│    Resolve equações de Maxwell para ferramenta EM LWD                      │
│    Frequência: 20 kHz  |  Espaçamento Tx-Rx: 1,0 m                        │
│                                ↓                                            │
│  Saída (resposta EM, para cada z em [0, 150 m]):                           │
│    Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz), ...  [A/m]                        │
│    ρ_h(z), ρ_v(z)  [Ohm·m]  (targets para inversão DL)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Analogia direta:**
- O simulador Fortran é o "oráculo" que sabe a física perfeita
- O modelo geológico é um caso de treino: "dadas estas medições EM, qual é ρ_h e ρ_v?"
- A rede de DL aprende a inversão: medições EM → resistividades

---

## 2. Origem dos Dados — Simulador Fortran

O simulador **PerfilaAnisoOmp** (Fortran, compilado com OpenMP) gera os dados de treinamento.
Ele resolve numericamente o problema direto EM para centenas de milhares de modelos geológicos
distintos, criando um dataset sintético supervisionado.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Geração do Dataset Sintético                                                │
│                                                                              │
│  Para cada modelo geológico k = 0, 1, ..., N_MODELS-1:                     │
│                                                                              │
│    1. Gera perfil aleatório de ρ_h(z), ρ_v(z) com N_CAMADAS camadas        │
│    2. Simula resposta EM para cada z em [0, profundidade_max]               │
│    3. Escreve seq_len linhas (22 colunas) no arquivo .dat                   │
│                                                                              │
│  Parâmetros da simulação (do arquivo .out):                                 │
│    n_models   = número total de modelos (ex: 10.000)                        │
│    theta_list = ângulos de mergulho simulados [graus] (ex: [0, 30, 60])    │
│    freq_list  = frequências [Hz] (ex: [20000.0])                            │
│    nmeds_list = medições por ângulo (ex: [600] para θ=0°)                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Por que θ = 0° → seq_len = 600?**

Para o dataset padrão (poço vertical, ângulo de inclinação θ = 0°), o simulador calcula a
resposta em 600 pontos de profundidade espaçados de 0,25 m, cobrindo 150 metros de coluna:

```
seq_len = 600 pontos
espaçamento = 0,25 m/ponto
profundidade total = 600 × 0,25 = 150 m
```

> **IMPORTANTE:** `seq_len` **não é hardcoded como 600** — é derivado do arquivo `.out` e
> armazenado em `config.sequence_length`. Datasets multi-dip (θ ≠ 0°) podem ter valores
> diferentes. O valor 600 é apenas o default para `PipelineConfig` (dataset Inv0Dip).

---

## 3. Estrutura do Arquivo `.dat` e `.out`

### 3.1 Arquivo `.out` — Metadados

O arquivo `.out` é um header de texto de **exatamente 4 linhas**, gerado pelo PerfilaAnisoOmp:

```
Linha 1: nt  nf  nm           ← n_angles, n_freqs, n_models
Linha 2: θ₀  θ₁  ...  θₙₜ₋₁  ← ângulos em graus
Linha 3: f₀  f₁  ...  fₙf₋₁  ← frequências em Hz
Linha 4: m₀  m₁  ...  mₙₜ₋₁  ← medições por ângulo (nmeds_list)
```

**Exemplo real (dataset Inv0Dip, θ=0°, 20 kHz, 10.000 modelos):**
```
1  1  10000
0.0
20000.0
600
```

**Sem o `.out`, é impossível saber onde termina um modelo e começa o próximo no `.dat`.**

### 3.2 Arquivo `.dat` — Dados Binários

O arquivo `.dat` é um **binário Fortran** com registros de tamanho fixo:

```
Tamanho por registro: 172 bytes
  = 1 × int32  (4 bytes)   → coluna 0 (índice de medição)
  + 21 × float64 (168 bytes) → colunas 1–21 (dados geofísicos)

Total de registros: N_MODELS × seq_len
```

Os modelos geológicos são **concatenados sequencialmente** no `.dat`:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Arquivo .dat (N_models × seq_len linhas totais)                             │
│                                                                              │
│  [MODELO 0: registros 0 a 599]          → linhas 1–600                      │
│  [MODELO 1: registros 600 a 1199]       → linhas 601–1200                   │
│  [MODELO 2: registros 1200 a 1799]      → linhas 1201–1800                  │
│  ...                                                                         │
│  [MODELO N-1: registros (N-1)×600 a N×600-1]                                │
│                                                                              │
│  Cada registro: 22 valores (1 int32 + 21 float64 = 172 bytes)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layout de 22 Colunas por Medição

Cada registro de 172 bytes corresponde a **uma medição** em um ponto de profundidade dentro
de um modelo geológico. O layout completo das 22 colunas:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYOUT 22 COLUNAS — .dat Fortran binary (172 bytes/registro)               │
├────┬──────────────┬──────────────────────────────────────────────────────────┤
│ Col│ Nome         │ Descrição e Papel no Pipeline                            │
├────┼──────────────┼──────────────────────────────────────────────────────────┤
│  0 │ meds         │ Índice de medição (int32 — NUNCA feature, apenas metadata)│
│  1 │ zobs         │ Profundidade observada [m] → INPUT_FEATURE [idx=1]       │
│  2 │ res_h        │ Resistividade horizontal [Ω·m] → OUTPUT_TARGET [idx=2]  │
│  3 │ res_v        │ Resistividade vertical [Ω·m] → OUTPUT_TARGET [idx=3]    │
│ 4-5│ Re/Im(Hxx)   │ Tensor EM coplanar (INPUT) — recebe decoupling ACp      │
│ 6-7│ Re/Im(Hxy)   │ Cross-component (off-diagonal) — base para Geosinais    │
│ 8-9│ Re/Im(Hxz)   │ Cross-component (off-diagonal)                          │
│10-11│Re/Im(Hyx)   │ Cross-component (off-diagonal)                          │
│12-13│Re/Im(Hyy)   │ Tensor EM coplanar — recebe decoupling ACp              │
│14-15│Re/Im(Hyz)   │ Cross-component — base para Geosinais                   │
│16-17│Re/Im(Hzx)   │ Cross-component — base para Geosinais                   │
│18-19│Re/Im(Hzy)   │ Cross-component — base para Geosinais                   │
│20-21│Re/Im(Hzz)   │ Tensor EM coaxial (INPUT) — recebe decoupling ACx       │
└────┴──────────────┴──────────────────────────────────────────────────────────┘
```

**Errata Imutável — Índices Críticos:**

```python
# Validado automaticamente por PipelineConfig.__post_init__()
INPUT_FEATURES  = [1, 4, 5, 20, 21]   # zobs + Re/Im(Hxx) + Re/Im(Hzz)
OUTPUT_TARGETS  = [2, 3]               # res_h + res_v
TARGET_SCALING  = "log10"              # NUNCA "log"

# Erros históricos que JAMAIS devem ser repetidos:
# INPUT_FEATURES = [0, 3, 4, 7, 8]   ← PROIBIDO (colunas erradas)
# OUTPUT_TARGETS = [1, 2]             ← PROIBIDO (inclui zobs como target)
# TARGET_SCALING = "log"              ← PROIBIDO (base natural, não base 10)
```

**Tensor EM completo:**

```
    ┌              ┐
    │ Hxx  Hxy  Hxz│
H = │ Hyx  Hyy  Hyz│   (cada componente é complexa: Re + j·Im)
    │ Hzx  Hzy  Hzz│
    └              ┘

Colunas de features usadas no baseline (5 features):
  [zobs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)]
  índices: [1, 4, 5, 20, 21]
```

---

## 5. Estrutura de Dados 3D no Pipeline

Após o carregamento do `.dat` e o reshape pelo `segregate_by_angle()`, os dados têm
**forma 3D** com semântica clara:

```python
# Shape após load_dataset():
x: np.ndarray  # (n_models, seq_len, n_features)   — features EM
y: np.ndarray  # (n_models, seq_len, n_targets)    — targets de resistividade
z: np.ndarray  # (n_models, seq_len)               — profundidade em metros

# Exemplo com dataset padrão (10.000 modelos, θ=0°, baseline 5 features):
x.shape = (10000, 600, 5)    # 10k modelos × 600 pontos × 5 features
y.shape = (10000, 600, 2)    # 10k modelos × 600 pontos × [ρ_h, ρ_v]
z.shape = (10000, 600)       # 10k modelos × 600 pontos de profundidade [m]
```

**Semântica dos eixos:**

| Eixo | Dimensão | Semântica | Operação Permitida |
|:----:|:--------:|:----------|:-------------------|
| 0 | n_models | Índice do modelo geológico | Shuffle, split por modelo |
| 1 | seq_len | Ponto de medição dentro do modelo | Preservado intacto (sequência temporal) |
| 2 | channels | Canal de feature/target | Seleção por índice, escalonamento |

> **Regra crítica:** A dimensão 0 (modelos) é a única que pode ser **embaralhada e dividida**
> no split. A dimensão 1 (sequência) é **SEMPRE preservada intacta** — ela representa a ordem
> espacial das medições dentro de um poço, que o modelo de DL deve aprender.

---

## 6. Split por Modelo Geológico — Princípio P1

### 6.1 Por que o split por amostra é proibido?

**Problema:** Se fizéssemos split aleatório por linha (amostra), medições do mesmo modelo
geológico poderiam cair em partições diferentes:

```
Modelo #42 (600 medições):
  ← Medições 0–419   → train   (70%)
  ← Medições 420–599 → test    (30%)
```

**Resultado catastrófico:**
- A rede aprende o perfil de resistividade de TODA a coluna do modelo #42 durante o treino
- As métricas de teste ficam artificialmente infladas (o modelo "memorizou" o perfil)
- O modelo **não generaliza** para novos cenários geológicos nunca vistos
- Esta é a forma mais grave de **data leakage** em séries temporais geofísicas

### 6.2 O Split Correto — Granularidade de Modelos

O split deve garantir que cada modelo geológico pertença a **exatamente uma** partição:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Split Model-Wise (zero data leakage)                                        │
│                                                                              │
│  Modelos geologicos:                                                         │
│    [M0, M1, M2, M3, M4, M5, M6, M7, M8, M9]  ← 10 modelos                │
│                  ↓  shuffle(seed=42)                                         │
│    [M3, M7, M1, M9, M0, M5, M4, M8, M2, M6]  ← permutação determinística  │
│     ├──────────────────────┤├──────┤├──────┤                                │
│           TRAIN (70%)       VAL(15%) TEST(15%)                               │
│                                                                              │
│  Garantias formais:                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  train_ids ∩ val_ids  = ∅   (zero overlap)                          │   │
│  │  train_ids ∩ test_ids = ∅   (zero overlap)                          │   │
│  │  val_ids   ∩ test_ids = ∅   (zero overlap)                          │   │
│  │  |train| + |val| + |test| = N_MODELS  (partição completa)           │   │
│  │  Cada modelo pertence a EXATAMENTE UM split                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Implementação — `split_model_ids()`

```python
# geosteering_ai/data/splitting.py

def split_model_ids(
    n_models: int,
    train_ratio: float = 0.70,
    val_ratio: float  = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Particiona IDs de modelos em train/val/test — zero data leakage."""
    rng = np.random.default_rng(seed)   # RNG moderno, independente do estado global
    indices = np.arange(n_models)
    rng.shuffle(indices)                # shuffle determinístico por seed

    n_train = int(n_models * train_ratio)
    n_val   = int(n_models * val_ratio)

    train_ids = set(indices[:n_train].tolist())
    val_ids   = set(indices[n_train : n_train + n_val].tolist())
    test_ids  = set(indices[n_train + n_val :].tolist())

    # Assertivas de integridade — falha imediata se houver bug
    assert train_ids & val_ids  == set(), "Overlap train/val"
    assert train_ids & test_ids == set(), "Overlap train/test"
    assert val_ids   & test_ids == set(), "Overlap val/test"

    return train_ids, val_ids, test_ids
```

### 6.4 Implementação — `apply_split()`

```python
def apply_split(
    angle_group: AngleGroup,
    train_ids: Set[int],
    val_ids: Set[int],
    test_ids: Set[int],
) -> DataSplits:
    """Aplica máscaras booleanas por modelo geológico."""
    model_ids = angle_group.model_ids  # (n_seq,) — ID do modelo por sequência

    # np.isin: vetorizado, eficiente para grandes arrays
    train_mask = np.isin(model_ids, sorted(train_ids))
    val_mask   = np.isin(model_ids, sorted(val_ids))
    test_mask  = np.isin(model_ids, sorted(test_ids))

    return DataSplits(
        x_train=angle_group.x[train_mask],
        y_train=angle_group.y[train_mask],
        z_train=angle_group.z_meters[train_mask],   # z em metros — NUNCA escalado
        x_val  =angle_group.x[val_mask],
        y_val  =angle_group.y[val_mask],
        z_val  =angle_group.z_meters[val_mask],     # z em metros — NUNCA escalado
        x_test =angle_group.x[test_mask],
        y_test =angle_group.y[test_mask],
        z_test =angle_group.z_meters[test_mask],    # z em metros — NUNCA escalado
        train_model_ids=train_ids,
        val_model_ids  =val_ids,
        test_model_ids =test_ids,
    )
```

### 6.5 Motivação Física

O princípio P1 é justificado pelo domínio geofísico:

- **Correlação intra-modelo:** medições do mesmo modelo geológico são altamente correlacionadas
  espacialmente (o mesmo perfil de resistividade gera toda a sequência). São **dependentes**, não
  i.i.d. como assume o split aleatório padrão.

- **Generalização para geologias novas:** em campo real, o modelo DL receberá medições de
  reservatórios nunca simulados. A avaliação deve refletir esse cenário.

- **Equivalente ao walk-forward em séries temporais:** assim como modelos de previsão são
  avaliados em períodos futuros (nunca vistos), modelos de inversão geofísica devem ser avaliados
  em cenários geológicos nunca vistos durante o treino.

---

## 7. O Modelo Geológico no Dataset e no tf.data

### 7.1 Carregamento e Reshape

```python
# Carregamento do .dat → array 2D
raw_data = load_binary_dat(dat_path)   # shape: (N_MODELS × seq_len, 22)

# Decoupling EM
decoupled = apply_decoupling(raw_data, config)  # remove acoplamento direto Tx-Rx

# Segregação MODEL-MAJOR → AngleGroup 3D
angle_groups = segregate_by_angle(decoupled, metadata, config)
# angle_groups[0.0].x.shape = (N_MODELS, 600, 5)  ← modelo geológico na dim 0
```

### 7.2 tf.data.Dataset — Unidade de Treino

No pipeline `tf.data`, **cada exemplo de treinamento é um modelo geológico completo**
(sequência de seq_len medições):

```python
# tf.data.Dataset por modelo geológico:
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# Cada (x, y) = um modelo geológico completo
#   x: (600, 5)  — seq_len pontos × 5 features
#   y: (600, 2)  — seq_len pontos × 2 targets [ρ_h, ρ_v]

# batch(32) = 32 modelos geológicos por batch
train_ds = (train_ds
    .shuffle(buffer_size=n_train_models)
    .batch(32)
    .map(noise_map_fn)   # on-the-fly: noise → FV → GS → scale
    .prefetch(tf.data.AUTOTUNE)
)
```

**Shape por batch:**
```
x_batch: (32, 600, 5)    → 32 modelos × 600 pontos × 5 features
y_batch: (32, 600, 2)    → 32 modelos × 600 pontos × [ρ_h, ρ_v]
```

### 7.3 Cadeia On-the-Fly por Modelo Geológico

Para cada batch, a cadeia on-the-fly é aplicada **por modelo geológico** dentro do `map()`:

```
x_batch_raw (32, 600, 5)
       ↓
  apply_noise_tf()        ← ruído sobre Re/Im brutas [A/m]
       ↓
  apply_feature_view_tf() ← FV: transforma Re/Im → mag/phase/razões
       ↓
  compute_geosignal_tf()  ← GS: computa geosinais sobre dados ruidosos
       ↓
  scale()                 ← escalonamento per-group (scaler fit em dados LIMPOS)
       ↓
x_batch_proc (32, 600, n_features_final)
```

> **Regra crítica de fidelidade física:** O ruído DEVE ser aplicado sobre as componentes
> Re/Im brutas (A/m), ANTES de FV e GS. Aplicar ruído após FV viola a física do instrumento,
> pois FV é uma transformação não-linear (mag = √(Re² + Im²) amplifica o ruído de forma
> diferente). GS deve ver o sinal ruidoso para aprender a ser robusto.

---

## 8. Container DataSplits

O `DataSplits` é o contêiner de resultado do split, exportado por `splitting.py`:

```python
@dataclass
class DataSplits:
    """Resultado do split por modelo geológico [P1].

    Cada campo _train/_val/_test contém os modelos geológicos
    alocados a cada partição. z_meters é preservado separadamente
    (NUNCA escalado pelo scaler — usado apenas para plots e avaliação).
    """

    # Arrays 3D de features — shape: (n_split_models, seq_len, n_features)
    x_train: np.ndarray
    x_val:   np.ndarray
    x_test:  np.ndarray

    # Arrays 3D de targets — shape: (n_split_models, seq_len, n_targets)
    y_train: np.ndarray
    y_val:   np.ndarray
    y_test:  np.ndarray

    # Profundidade em metros — shape: (n_split_models, seq_len)
    # Preservada SEPARADAMENTE — nunca entra no scaler
    z_train: np.ndarray
    z_val:   np.ndarray
    z_test:  np.ndarray

    # IDs dos modelos geológicos em cada partição
    train_model_ids: Set[int]
    val_model_ids:   Set[int]
    test_model_ids:  Set[int]
```

**Exemplo com dataset padrão (10.000 modelos, split 70/15/15):**

| Campo | Shape | Descrição |
|:------|:------|:----------|
| `x_train` | (7000, 600, 5) | Features de 7.000 modelos de treino |
| `y_train` | (7000, 600, 2) | Targets de 7.000 modelos de treino |
| `z_train` | (7000, 600) | Profundidade dos modelos de treino |
| `x_val` | (1500, 600, 5) | Features de 1.500 modelos de validação |
| `y_val` | (1500, 600, 2) | Targets de 1.500 modelos de validação |
| `x_test` | (1500, 600, 5) | Features de 1.500 modelos de teste |
| `y_test` | (1500, 600, 2) | Targets de 1.500 modelos de teste |
| `train_model_ids` | Set(7000) | IDs dos modelos de treino (ex: {3,7,1,...}) |

> **Nota sobre `z_meters`:** A profundidade em metros não é escalonada em nenhum momento.
> O scaler é fitado sobre `x` (features) e nunca toca `z`. Isso preserva a interpretabilidade
> física dos perfis de profundidade durante a avaliação e visualização.

---

## 9. Metadados — OutMetadata e AngleGroup

### 9.1 OutMetadata

Encapsula as informações do arquivo `.out`:

```python
@dataclass
class OutMetadata:
    n_angles: int           # número de ângulos simulados
    n_freqs: int            # número de frequências simuladas
    n_models: int           # total de modelos geológicos
    theta_list: List[float] # ângulos em graus (ex: [0.0, 30.0, 60.0])
    freq_list: List[float]  # frequências em Hz (ex: [20000.0])
    nmeds_list: List[int]   # medições por ângulo (ex: [600])

    # Derivados (auto-calculados em __post_init__):
    total_rows: int         # N_MODELS × sum(nmeds × n_freqs)
    rows_per_model: int     # sum(nmeds × n_freqs) para um modelo
```

**Exemplo de uso:**

```python
metadata = parse_out_metadata("dataset/Inv0Dip.out")
print(f"Modelos: {metadata.n_models}")        # 10000
print(f"Ângulos: {metadata.theta_list}")      # [0.0]
print(f"Frequências: {metadata.freq_list}")   # [20000.0]
print(f"Medições/modelo: {metadata.nmeds_list}")  # [600]
print(f"Total de registros: {metadata.total_rows}")  # 6000000
```

### 9.2 AngleGroup

Dados segregados para um ângulo de inclinação específico, em layout MODEL-MAJOR:

```python
@dataclass
class AngleGroup:
    theta: float            # ângulo de inclinação (ex: 0.0, 30.0, 60.0)
    x: np.ndarray           # features: shape (n_seq, seq_len, n_features)
    y: np.ndarray           # targets:  shape (n_seq, seq_len, n_targets)
    z_meters: np.ndarray    # profundidade: shape (n_seq, seq_len)
    model_ids: np.ndarray   # ID do modelo: shape (n_seq,)
    nmeds: int              # medições por sequência neste ângulo
```

Para datasets multi-ângulo, há um `AngleGroup` por valor de θ no dicionário retornado
por `segregate_by_angle()`:

```python
angle_groups = segregate_by_angle(data, metadata, config)
# angle_groups = {
#     0.0:  AngleGroup(theta=0.0,  x.shape=(10000, 600, 5), ...),
#     30.0: AngleGroup(theta=30.0, x.shape=(10000, 720, 5), ...),
#     60.0: AngleGroup(theta=60.0, x.shape=(10000, 900, 5), ...),
# }
# Cada ângulo pode ter seq_len diferente (derivado do nmeds_list do .out)
```

---

## 10. Decoupling EM — Remoção do Acoplamento Direto

Antes de qualquer uso das componentes EM, o pipeline aplica o **decoupling** para remover
o acoplamento direto (coupling direto Tx→Rx no ar, sem a presença da formação):

```
ACp = -1 / (4π × L³) ≈ -0,079577  [A/m]   (componentes planares: Hxx, Hyy)
ACx = +1 / (2π × L³) ≈ +0,159155  [A/m]   (componente axial: Hzz)

para L = SPACING_METERS = 1,0 m

H_decoupled_xx = H_xx - ACp
H_decoupled_zz = H_zz - ACx
```

```python
# Aplicado em apply_decoupling() — geosteering_ai/data/loading.py
# O decoupling é aplicado sobre as partes REAIS das componentes principais.
# Componentes cross (Hxy, Hxz, etc.) não recebem decoupling (acoplamento zero no ar).
```

> **Significado físico:** O instrumento LWD mede o campo total (acoplamento direto + resposta
> da formação). O decoupling extrai apenas a componente de resposta da formação, que é o sinal
> de interesse para inversão. Sem decoupling, as redes tendem a aprender o offset constante
> em vez da variação de resistividade.

---

## 11. Cadeia Completa de Carregamento

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Pipeline de Carregamento — data/loading.py                                  │
│                                                                              │
│  1. parse_out_metadata("dataset.out")                                        │
│        ↓ OutMetadata(n_models=10000, seq_len=600, freq=[20kHz], θ=[0°])    │
│                                                                              │
│  2. load_binary_dat("dataset.dat")                                           │
│        ↓ np.ndarray shape: (6.000.000, 22)  [raw, 172 bytes/registro]       │
│                                                                              │
│  3. apply_decoupling(raw_data, config)                                       │
│        ↓ np.ndarray shape: (6.000.000, 22)  [ACp e ACx removidos]          │
│                                                                              │
│  4. segregate_by_angle(data, metadata, config)                               │
│        ↓ Dict[theta → AngleGroup]                                            │
│           angle_groups[0.0].x.shape = (10000, 600, 5)  ← MODEL-MAJOR 3D   │
│           angle_groups[0.0].model_ids.shape = (10000,)                      │
│                                                                              │
│  5. split_angle_group(angle_groups[config.theta], config)                    │
│        ↓ DataSplits                                                          │
│           x_train.shape = (7000, 600, 5)                                    │
│           x_val.shape   = (1500, 600, 5)                                    │
│           x_test.shape  = (1500, 600, 5)                                    │
│                                                                              │
│  6. DataPipeline.prepare() — target_scaling + scaler fit on clean          │
│        ↓ PreparedData (splits escalados, scaler, metadata)                   │
│                                                                              │
│  7. tf.data.Dataset — on-the-fly: noise → FV → GS → scale → model         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Valores Físicos Críticos e Errata

Todos os parâmetros abaixo são validados automaticamente por `PipelineConfig.__post_init__()`:

```python
# Validações fail-fast — erros detectados na instanciação, não em runtime
assert 100.0 <= frequency_hz <= 1e6           # Default: 20000.0 Hz (20 kHz)
assert 0.1 <= spacing_meters <= 10.0          # Default: 1.0 m
assert 10 <= sequence_length <= 100000        # Default: 600 (Inv0Dip θ=0°)
assert target_scaling == "log10"              # NUNCA "log" (base natural)
assert input_features == [1, 4, 5, 20, 21]   # NUNCA [0, 3, 4, 7, 8]
assert output_targets == [2, 3]              # NUNCA [1, 2]
assert eps_tf >= 1e-15                        # Default 1e-12, float32-safe
```

**Significado físico dos valores padrão:**

| Parâmetro | Valor | Significado Físico |
|:----------|:-----:|:-------------------|
| `frequency_hz` | 20000 | Frequência de operação da ferramenta LWD (20 kHz) |
| `spacing_meters` | 1.0 | Distância Transmissor–Receptor na ferramenta (1 m) |
| `sequence_length` | 600 | Medições por modelo (150 m de coluna, passo 0,25 m) |
| `target_scaling` | "log10" | Resistividade varia 4+ ordens de grandeza (log necessário) |
| `ACp` | −0,079577 A/m | Acoplamento direto Tx-Rx para componentes planares |
| `ACx` | +0,159155 A/m | Acoplamento direto Tx-Rx para componente axial |

---

## 13. Referências no Código

| Componente | Arquivo | Descrição |
|:-----------|:--------|:----------|
| `PipelineConfig` | `geosteering_ai/config.py` | Configuração central — `sequence_length`, `input_features`, `output_targets` |
| `OutMetadata` | `geosteering_ai/data/loading.py:175` | Metadados do `.out` — n_models, seq_len, frequências, ângulos |
| `AngleGroup` | `geosteering_ai/data/loading.py:222` | Dados MODEL-MAJOR para um ângulo θ |
| `parse_out_metadata()` | `geosteering_ai/data/loading.py:270` | Parsing das 4 linhas do `.out` |
| `load_binary_dat()` | `geosteering_ai/data/loading.py` | Leitura binária Fortran — 172 bytes/registro |
| `apply_decoupling()` | `geosteering_ai/data/loading.py` | Remove ACp e ACx das componentes EM |
| `segregate_by_angle()` | `geosteering_ai/data/loading.py` | Reshape 2D→3D, segregação MODEL-MAJOR |
| `DataSplits` | `geosteering_ai/data/splitting.py:81` | Container resultado do split |
| `split_model_ids()` | `geosteering_ai/data/splitting.py:159` | Particiona IDs de modelos — zero leakage |
| `apply_split()` | `geosteering_ai/data/splitting.py:241` | Máscaras booleanas por modelo geológico |
| `split_angle_group()` | `geosteering_ai/data/splitting.py:309` | Wrapper conveniente (split_model_ids + apply_split) |
| `DataPipeline` | `geosteering_ai/data/pipeline.py` | Orquestra toda a cadeia loading→split→scale |

**Testes relacionados:**

| Teste | Arquivo | O que valida |
|:------|:--------|:-------------|
| `TestParseOut` | `tests/test_data_pipeline.py` | Parsing correto do `.out` (3 casos) |
| `TestSplitting` | `tests/test_data_pipeline.py` | Zero leakage, shapes, z_meters preservado |
| `TestErrata` | `tests/test_config.py` | Validação dos índices críticos no `__post_init__` |

---

## 14. Tabela de Referência Rápida

| Aspecto | Detalhe |
|:--------|:--------|
| **Definição** | Cenário sintético de resistividade gerado pelo simulador Fortran PerfilaAnisoOmp |
| **Geração** | Problema direto: perfil ρ(z) → resposta EM H(z) — resolve equações de Maxwell |
| **Arquivo de dados** | `.dat` (binário Fortran, 172 bytes/registro) + `.out` (metadados, 4 linhas texto) |
| **Registros por modelo** | `seq_len` linhas × 22 colunas (derivado do `.out`, default 600) |
| **seq_len default** | 600 pontos (θ=0°, espaçamento 0,25 m, 150 m de coluna) |
| **Leitura do seq_len** | Sempre do `.out` via `parse_out_metadata()` — NUNCA hardcoded |
| **Formato binário** | 1 × int32 (índice) + 21 × float64 (dados) = 172 bytes/registro |
| **Shape 3D** | `(n_models, seq_len, channels)` após reshape MODEL-MAJOR |
| **Features de entrada** | Colunas [1, 4, 5, 20, 21] → [zobs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)] |
| **Targets de saída** | Colunas [2, 3] → [ρ_h, ρ_v] em log10(Ohm·m) |
| **Decoupling** | ACp ≈ -0,0796 A/m (planares) e ACx ≈ +0,1592 A/m (axial), L = 1,0 m |
| **Split correto [P1]** | Por índice de modelo (dimensão 0) — NUNCA por linha individual |
| **Motivação do split** | Correlação intra-modelo impede split por amostra — seria data leakage |
| **Ratios padrão** | 70% treino / 15% validação / 15% teste (por modelo geológico) |
| **Determinismo** | Seed `config.global_seed` (default 42) → splits reprodutíveis |
| **z_meters** | Preservado separado — NUNCA escalado pelo scaler |
| **Unidade de batch** | 1 modelo geológico completo = (seq_len, n_features) por exemplo |
| **Exemplo de dataset** | 10.000 modelos × 600 pontos = 6.000.000 registros totais |
| **Distribuição típica** | 7.000 treino / 1.500 val / 1.500 teste (por modelo) |
| **Garantia de integridade** | `assert train_ids ∩ val_ids = ∅` e idem para test — fail-fast |
| **Classe container** | `DataSplits` (dataclass) — exportada por `data/splitting.py` |
| **Validação automática** | `PipelineConfig.__post_init__()` verifica toda a errata |

---

*Referência Técnica — Modelo Geológico — Geosteering AI v2.0 · Pipeline v5.0.15+*
