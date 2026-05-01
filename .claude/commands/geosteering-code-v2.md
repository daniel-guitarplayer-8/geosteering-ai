---
name: geosteering-code-v2
description: |
  Sub-skill de padrões de código do Geosteering AI v2.0. Cobre TODOS os padrões obrigatórios:
  PipelineConfig como parâmetro (NUNCA globals), Factory/Registry, DataPipeline, presets YAML,
  logging (NUNCA print), estrutura de pacote, e padrões de documentação D1-D14 com exemplos
  completos. Use para qualquer questão de como escrever/estruturar código no projeto.
  Triggers: "PipelineConfig", "globals", "Factory", "Registry", "DataPipeline", "logging",
  "print()", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12",
  "D13", "D14", "mega-header", "docstring", "comentário", "padrão de código", "código v2",
  "código v2.0", "pacote", "estrutura", "proibição", "errata", "checklist".
  v2.0 | Última atualização: 2026-04-29 | Sub-skill da geosteering-v2
---

# Geosteering AI v2.0 — Padrões de Código e Documentação

> **Sub-skill especializada** — Padrões obrigatórios de código e documentação.
> Skill principal: `geosteering-v2` | Física: `geosteering-physics` | Modelos: `geosteering-models` | Losses: `geosteering-losses`

---

## 1. Proibições Absolutas (Fail-Fast)

Qualquer violação invalida o código gerado. Verificar ANTES de escrever qualquer linha:

| Proibição | Correto | Errado | Motivo |
|:----------|:--------|:-------|:-------|
| PyTorch | TensorFlow/Keras | `import torch` | Framework exclusivo |
| `globals().get()` | `config.param` via PipelineConfig | `globals().get("FLAG", d)` | Ponto único de verdade |
| `print()` | `logger.info()` via logging | `print("resultado")` | Observabilidade |
| `FREQUENCY_HZ = 2.0` | `20000.0` (20 kHz) | Qualquer outro valor | Errata física |
| `SPACING_METERS = 1000.0` | `1.0` (1 metro) | Qualquer outro valor | Errata física |
| `SEQUENCE_LENGTH = 601` | `600` | `601` ou outro | Errata física |
| `TARGET_SCALING = "log"` | `"log10"` | `"log"` (ln) | Log natural ≠ log10 |
| `INPUT_FEATURES = [0,3,4,7,8]` | `[1,4,5,20,21]` (22-col) | Qualquer outro set | Errata física |
| `OUTPUT_TARGETS = [1,2]` | `[2,3]` (22-col) | Qualquer outro set | Errata física |
| `eps = 1e-30` | `1e-12` (float32 safe) | `1e-30` | Causa subnormais float32 |
| Split por amostra | `split_by_model=True` (P1) | Split random | Leakage geológico |
| Scaler fit em dados ruidosos | Fit em dados LIMPOS | Fit após noise | Bias sistemático |
| Noise offline com FV/GS | On-the-fly em `tf.data.map` | Noise antes do split | Infidelidade física |
| Função sem `config` | `def f(config: PipelineConfig)` | `def f():` com globals | Testabilidade |

---

## 2. Padrões de Código Obrigatórios

### 2.1 PipelineConfig como Parâmetro (NUNCA globals)

```python
# ✅ CORRETO:
def build_model(config: PipelineConfig) -> tf.keras.Model:
    n_features = config.n_features        # acesso via config
    output_channels = config.output_channels
    use_causal = config.use_causal_mode
    return ModelRegistry().build(config)

# ❌ PROIBIDO:
def build_model():
    model_type = globals().get("MODEL_TYPE", "ResNet_18")  # globals!
    n_features = N_FEATURES                                  # constante global!
    return build_resnet18(n_features=n_features)             # if/elif imperativo!
```

**Por quê:** Toda função que lê globals não é testável (estado global oculto), não é reprodutível
(comportamento depende de ordem de execução) e não é serializável (YAML roundtrip impossível).

### 2.2 Factory/Registry para Componentes

```python
# ✅ CORRETO:
model   = ModelRegistry().build(config)        # despacha para build_* correto
loss_fn = LossFactory.get(config)              # resolve loss_type automaticamente
loss_fn = LossFactory.build_combined(config)   # base + look_ahead + DTB + PINNs
cbs     = build_callbacks(config, model, noise_var)

# ❌ PROIBIDO:
if MODEL_TYPE == "ResNet_18":
    model = build_resnet18(...)
elif MODEL_TYPE == "TCN":
    model = build_tcn(...)
# ... 48 branches imperativas
```

### 2.3 DataPipeline com Cadeia Explícita

```python
# ✅ CORRETO:
pipeline = DataPipeline(config)
data = pipeline.prepare(dataset_path)            # raw → split → fit_scaler(clean)
map_fn = pipeline.build_train_map_fn(noise_var)  # noise → FV → GS → scale

ds_train = tf.data.Dataset.from_tensor_slices(data.x_train_raw)
ds_train = ds_train.map(map_fn).batch(config.batch_size).prefetch(2)

# ❌ PROIBIDO:
x_train = apply_feature_view(x_train, view)   # offline, sem noise
x_train = apply_noise(x_train)                # noise após FV → fisicamente errado
scaler.fit(x_train_noisy)                     # fit em dados ruidosos → bias
```

### 2.4 Presets YAML para Reprodutibilidade

```python
# Via arquivo YAML:
config = PipelineConfig.from_yaml("configs/robusto.yaml")

# Via presets de classe (shortcuts reprodutíveis):
config = PipelineConfig.baseline()        # P1: sem noise, sem GS, ResNet-18
config = PipelineConfig.robusto()         # E-Robusto S21 defaults
config = PipelineConfig.nstage(n=3)       # N-Stage training (3 estágios)
config = PipelineConfig.geosinais_p4()    # P4 com geosinais USD+UAD+UHR
config = PipelineConfig.realtime()        # Geosteering causal mode, WaveNet
```

### 2.5 Logging Estruturado (NUNCA print)

```python
import logging
logger = logging.getLogger(__name__)

# ✅ CORRETO:
logger.info("Modelo compilado: %s, params: %d", config.model_type, model.count_params())
logger.warning("Scaler não fitado — usando defaults conservadores")
logger.debug("Shape após Feature View '%s': %s", config.feature_view, x.shape)
logger.error("loss_type='%s' inválido. Opções: %s", lt, VALID_LOSS_TYPES)

# ❌ PROIBIDO:
print(f"Modelo: {MODEL_TYPE}")
print("Época:", epoch, "loss:", loss)
```

---

## 3. Estrutura do Pacote v2.0

```
geosteering_ai/
├── __init__.py
├── config.py              ← PipelineConfig dataclass (ponto único de verdade)
├── data/
│   ├── loading.py         ← parse .out, load .dat (22-col), segregate by angle
│   ├── splitting.py       ← split by geological model (P1), stratified
│   ├── feature_views.py   ← 6 Feature Views (numpy + TF)
│   ├── geosignals.py      ← 5 famílias GS (numpy + TF)
│   ├── scaling.py         ← 8 scalers + 8 target scalings + per-group [P3]
│   ├── pipeline.py        ← DataPipeline: raw → split → fit → tf.data.map
│   ├── boundaries.py      ← DTB, detecção de fronteiras (P5)
│   ├── sampling.py        ← Amostragem estratificada
│   ├── second_order.py    ← Estatísticas de segunda ordem
│   └── surrogate_data.py  ← Geração de dados para SurrogateNet
├── noise/
│   ├── functions.py       ← apply_raw_em_noise() (34+ noise types)
│   └── curriculum.py      ← CurriculumSchedule: 3-phase + N-Stage
├── models/
│   ├── blocks.py          ← 23 blocos Keras reutilizáveis
│   ├── cnn.py             ← ResNet-18/34/50, ConvNeXt, Inception, CNN_1D, ResNeXt
│   ├── tcn.py             ← TCN, TCN_Advanced, ModernTCN
│   ├── rnn.py             ← LSTM, BiLSTM
│   ├── hybrid.py          ← CNN_LSTM, CNN_BiLSTM_ED, ResNeXt_LSTM
│   ├── unet.py            ← 14 variantes U-Net
│   ├── transformer.py     ← 6 transformers
│   ├── decomposition.py   ← N_BEATS, N_HiTS
│   ├── advanced.py        ← DNN, FNO, DeepONet, Geophysical_Attention, INN
│   ├── geosteering.py     ← WaveNet, Causal_Transformer, Informer, Mamba_S4, EF
│   ├── surrogate.py       ← SurrogateNet (TCN 127M + ModernTCN 204M)
│   └── registry.py        ← ModelRegistry: 48 entradas + build()
├── losses/
│   ├── catalog.py         ← 26 loss functions (A: genéricas, B: geofísicas, C: GS, D: avançadas)
│   ├── factory.py         ← LossFactory.get() + build_combined()
│   └── pinns.py           ← 8 cenários PINN (residuos físicos, constraints)
├── training/
│   ├── loop.py            ← TrainingLoop.run()
│   ├── callbacks.py       ← build_callbacks(config, model, noise_var) — 17+ callbacks
│   ├── metrics.py         ← R2Score, PerComponentMetric, AnisotropyRatioError
│   ├── nstage.py          ← N-Stage training (multi-stage noise adaptation)
│   └── optuna_hpo.py      ← HPO com Optuna
├── inference/
│   ├── pipeline.py        ← InferencePipeline (P6): FV+GS+scalers, joblib save/load
│   ├── realtime.py        ← Sliding window inference
│   ├── export.py          ← SavedModel, TFLite, ONNX
│   └── uncertainty.py     ← MC Dropout + Ensemble + INN posterior sampling
├── evaluation/
│   ├── metrics.py         ← Métricas numpy (MSE, RMSE, R2, MBE)
│   └── comparison.py      ← Comparação entre modelos
├── visualization/
│   ├── holdout.py         ← Plots holdout clean+noisy
│   ├── picasso.py         ← Picasso DOD plots (6 métodos)
│   ├── eda.py             ← EDA plots
│   └── realtime.py        ← Monitor de inferência em tempo real
└── utils/
    ├── logger.py          ← ColoredFormatter, setup_logger
    ├── timer.py           ← timer_decorator, ProgressTracker
    ├── validation.py      ← ValidationTracker
    ├── formatting.py      ← print_header, format_params
    ├── system.py          ← is_colab(), has_gpu()
    └── io.py              ← File I/O helpers
```

---

## 4. Padrões de Documentação D1–D14

O código v2.0 DEVE manter a mesma profundidade documental do legado C28.
O código é um **documento de referência executável** — lido como tutorial técnico.

### D1 — Mega-Header Unicode (OBRIGATÓRIO — todos os módulos)

Cada arquivo `.py` em `geosteering_ai/` DEVE ter no topo:

```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: {{subpacote}}/{{nome_modulo}}.py                                  ║
# ║  Bloco: {{N}} — {{Nome do Bloco}}                                          ║
# ║                                                                             ║
# ║  Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning      ║
# ║  Autor: Daniel Leal                                                         ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)          ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec)  ║
# ║  Pacote: geosteering_ai (pip installable)                                  ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                   ║
# ║                                                                             ║
# ║  Propósito:                                                                 ║
# ║    • {{Bullet 1 — ação principal}}                                         ║
# ║    • {{Bullet 2 — ação secundária}}                                        ║
# ║    • {{Bullet 3+ — outras ações}}                                          ║
# ║                                                                             ║
# ║  Dependências: config.py (PipelineConfig),                                  ║
# ║                {{módulo}} ({{símbolos}})                                    ║
# ║  Exports: ~{{N}} funções/classes — ver __all__                             ║
# ║  Ref: docs/ARCHITECTURE_v2.md seção {{X.Y}}                               ║
# ║                                                                             ║
# ║  Histórico:                                                                 ║
# ║    v2.0.0 (2026-03) — Implementação inicial                               ║
# ║    {{vX.Y.Z}} — {{descrição da mudança}}                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
```

**14 campos obrigatórios:** Módulo, Bloco, Projeto, Autor, Framework, Ambiente, Pacote, Config,
Propósito (3+ bullets), Dependências (lista explícita), Exports, Ref, Histórico.

### D2 — Cabeçalho de Seção (OBRIGATÓRIO — todas as seções lógicas)

```python
# ════════════════════════════════════════════════════════════════════════════
# SEÇÃO: {{TÍTULO EM MAIÚSCULAS}}
# ════════════════════════════════════════════════════════════════════════════
# {{Linha 1: Descrição do propósito desta seção}}
# {{Linha 2: Contexto técnico/físico relevante}}
# {{Linha 3: Relação com padrões P## ou perspectivas}}
# {{Linha 4: Referência cruzada a outros módulos ou docs}}
# ──────────────────────────────────────────────────────────────────────────
```

**Requisito mínimo:** ≥ 4 linhas de comentário contextual ANTES do código.

### D3 — Diagramas ASCII (OBRIGATÓRIO se ≥ 3 caminhos/categorias)

Usar bordas Unicode: `┌ ─ ┬ ┐ │ ├ ┼ ┤ └ ┴ ┘ ║ ═`

Situações obrigatórias:
- Fluxos de dados com ≥ 3 etapas
- Mapeamentos semânticos (categorias de componentes)
- Cascatas de transformação (FV, GS, scaling)
- Catálogos de componentes (arquiteturas, losses, noise)
- Fórmulas físicas com múltiplas variáveis
- Layout de dados (22-col, tensor EM)

### D4 — Atributos de Config (OBRIGATÓRIO — bloco de 4+ linhas por grupo)

```python
# ── GRUPO: Nome do Grupo ─────────────────────────────────────────────────
# Descrição: [Frase COMPLETA descrevendo o grupo]
# Relação: [Como afeta o pipeline — quais módulos usam]
# Ref: [Seção do ARCHITECTURE_v2.md ou docs/physics/]
# Nota: [Versão onde foi introduzido ou alterado]
atributo_1: tipo = default  # descrição inline
atributo_2: tipo = default  # descrição inline
```

### D5 — Docstrings de Funções (OBRIGATÓRIO — Google-style, 5+ campos)

```python
def aplicar_feature_view(
    data: np.ndarray,
    config: PipelineConfig,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Transforma componentes EM brutas conforme a Feature View selecionada.

    Aplica uma das 6 Feature Views canônicas ao subconjunto de colunas EM
    [Re(H1), Im(H1), Re(H2), Im(H2)] do tensor de entrada (22-col).
    SEMPRE usa log10 (nunca ln). Versões numpy e TF são consistentes desde v2.0.

    A cadeia de dados segue: raw → noise → FV → GS → scale.
    Esta função implementa a etapa FV (Feature View) da cadeia.

    Args:
        data: Array (n_rows, n_feat) ou (n_seq, seq_len, n_feat).
            Layout esperado após loading: cols 0-3 = prefix, col 4 = zobs,
            cols 5-8 = [Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)].
        config: Configuração do pipeline. Atributos usados:
            - config.feature_view: Nome da Feature View (ver VALID_VIEWS)
            - config.eps_tf: Epsilon para estabilidade numérica (default: 1e-12)
        verbose: Se True, loga shape e view ativa via logger.debug.

    Returns:
        np.ndarray: Array com mesma shape, colunas EM substituídas conforme FV.
            4 colunas EM [5:9] são transformadas in-place (cópia retornada).

    Raises:
        ValueError: Se config.feature_view não está em VALID_VIEWS.
        AssertionError: Se data.ndim não é 2 ou 3.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig(feature_view="logH1_logH2")
        >>> out = aplicar_feature_view(data, config)
        >>> assert out.shape == data.shape

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._apply_fv_gs() (modo offline)
            - data/pipeline.py: build_train_map_fn() (modo on-the-fly, TF)
            - tests/test_data_pipeline.py: TestFeatureViews
        Ref: docs/ARCHITECTURE_v2.md seção 4.3.
        Bug fix v2.0: Legado (C22) usava ln (numpy) vs log10 (TF).
            Agora ambos usam log10 consistentemente.
        Guard numérico: EPS = 1e-12 (NUNCA 1e-30 em float32).
    """
```

**6 campos obrigatórios:** descrição (1 linha), parágrafos de contexto, Args (com tipos e restrições),
Returns (com formato), Raises, Example, Note (com cross-references).

### D6 — Docstrings de Classes (OBRIGATÓRIO — Attributes + Example)

```python
@dataclass
class DataPipeline:
    """Pipeline de dados completo: raw → split → scaler → tf.data.

    Encapsula a cadeia de preparação de dados para treinamento e validação.
    Garante que: (1) scaler é fitado em dados limpos; (2) noise é on-the-fly
    via tf.data.map; (3) split é por modelo geológico (P1) para evitar leakage.

    Attributes:
        config (PipelineConfig): Configuração completa do pipeline.
            Controla feature_view, geosignal_families, noise_level_max, etc.
        _scaler_fitted (bool): True após prepare() ser chamado com sucesso.
            Acessar data.x_train_raw antes de prepare() levanta RuntimeError.

    Example:
        >>> pipeline = DataPipeline(config)
        >>> data = pipeline.prepare("dataset/")    # split + fit scaler
        >>> map_fn = pipeline.build_train_map_fn(noise_var)
        >>> ds_train = tf.data.Dataset.from_tensor_slices(data.x_train_raw)
        >>> ds_train = ds_train.map(map_fn).batch(32).prefetch(2)

    Note:
        Utilizado em: training/loop.py (TrainingLoop.run()).
        Ref: docs/ARCHITECTURE_v2.md seção 3.2 (DataPipeline).
        Invariante: NUNCA chamar build_train_map_fn() antes de prepare().
    """
```

### D7 — Comentários Inline Semânticos

```python
# ── Componentes EM (4 colunas do tensor H) ───────────────────────────────
re_h1 = data[:, em_start]       # Re(Hxx) — componente planar (sensível a rho_h)
im_h1 = data[:, em_start + 1]   # Im(Hxx) — componente planar
re_h2 = data[:, em_start + 2]   # Re(Hzz) — componente axial (sensível a bulk rho)
im_h2 = data[:, em_start + 3]   # Im(Hzz) — componente axial
```

Comentários inline DEVEM incluir **significado físico**, não apenas nome do campo.

### D8 — Inventário de Exports (OBRIGATÓRIO — todo módulo)

```python
# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventário completo de símbolos exportados por este módulo.
# Agrupados semanticamente para facilitar navegação e import.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Constantes ────────────────────────────────────────────────────────
    "EPS",
    "VALID_VIEWS",
    # ── Funções numpy ─────────────────────────────────────────────────────
    "apply_feature_view",
    # ── Funções TensorFlow ────────────────────────────────────────────────
    "apply_feature_view_tf",
]
```

### D9 — Logging Estruturado (OBRIGATÓRIO)

```python
import logging
logger = logging.getLogger(__name__)

# ── Início de operação ────────────────────────────────────────────────────
logger.info("Aplicando Feature View '%s' em %d amostras", view, n_samples)
# ── Resultado ─────────────────────────────────────────────────────────────
logger.info("Feature View aplicada: shape=%s, view='%s'", result.shape, view)
logger.debug("  Colunas EM transformadas: [%d:%d]", em_start, em_start + 4)
```

### D10 — Constantes com Documentação Física

```python
# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS E DE CONFIGURAÇÃO
# ════════════════════════════════════════════════════════════════════════════
# Valores críticos validados pela Errata v4.4.5 + v5.0.15.
# Qualquer alteração DEVE ser aprovada e documentada.
# Ref: docs/physics/errata_valores.md
# ──────────────────────────────────────────────────────────────────────────

# Epsilon seguro para float32 — protege contra underflow em log/divisão.
# NUNCA usar 1e-30 (causa subnormais em float32, gradientes explodidos).
# Ref: Errata v5.0.15, IEEE 754 float32 min normal ≈ 1,175e-38.
EPS = 1e-12

# Feature Views válidas — 6 transformações sobre componentes EM.
# identity/raw: passthrough. H1_logH2: H1 raw + H2 log10.
# Ref: docs/physics/perspectivas.md seção Feature Views.
VALID_VIEWS = {"identity", "raw", "H1_logH2", "logH1_logH2",
               "IMH1_IMH2_razao", "IMH1_IMH2_lograzao"}
```

### D11 — Tabelas de Fórmulas ASCII (OBRIGATÓRIO em catálogos)

```python
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  6 Feature Views Canônicas (Ref: docs/physics/perspectivas.md):          │
#   │                                                                           │
#   │  View               │ Canal 0    │ Canal 1  │ Canal 2       │ Canal 3    │
#   │  ───────────────────┼────────────┼──────────┼───────────────┼────────────│
#   │  identity / raw     │ Re(H1)     │ Im(H1)   │ Re(H2)        │ Im(H2)    │
#   │  H1_logH2           │ Re(H1)     │ Im(H1)   │ log10|H2|     │ φ(H2)     │
#   │  logH1_logH2        │ log10|H1|  │ φ(H1)    │ log10|H2|     │ φ(H2)     │
#   │  IMH1_IMH2_razao    │ Im(H1)     │ Im(H2)   │ |H1|/|H2|    │ Δφ        │
#   │  IMH1_IMH2_lograzao │ Im(H1)     │ Im(H2)   │ log10(ratio)  │ Δφ        │
#   │                                                                           │
#   │  H1 = Hxx (planar), H2 = Hzz (axial)                                    │
#   │  |H| = √(Re² + Im² + ε),  φ(H) = arctan2(Im, Re)                       │
#   │  ε = 1e-12 (float32 safe — NUNCA 1e-30)                                 │
#   │  SEMPRE log10 (NUNCA ln — bug fix v2.0)                                 │
#   └──────────────────────────────────────────────────────────────────────────┘
```

### D12 — Cross-References em Docstrings (OBRIGATÓRIO — seção Note)

Toda função pública DEVE ter seção `Note:` com:
1. Onde mais esta função é usada no pipeline
2. Referência à seção da documentação
3. Bug fixes relevantes (se aplicável)
4. Restrições críticas (errata, guards numéricos)

### D13 — Branch Comments com Layout de Saída (OBRIGATÓRIO em transformações)

```python
    if view == "H1_logH2":
        # ── H1_logH2: H1 cru preserva SNR em alta atenuação,
        #    H2 log10-transformado comprime faixa dinâmica larga de Hzz.
        #    Saída: [Re(H1), Im(H1), log10|H2|, φ(H2)]
        #    Motivação física: Hzz varia 4+ ordens de magnitude,
        #    log10 estabiliza gradientes e melhora convergência.
        result[:, em_start + 2] = _safe_log10(mag_h2)
        result[:, em_start + 3] = phi_h2

    elif view == "logH1_logH2":
        # ── logH1_logH2: Ambos H1 e H2 em escala logarítmica.
        #    Saída: [log10|H1|, φ(H1), log10|H2|, φ(H2)]
        #    Motivação física: magnitude + fase capturam toda a informação
        #    do sinal complexo; log10 comprime faixa para melhor treinamento.
        result[:, em_start]     = _safe_log10(mag_h1)
        result[:, em_start + 1] = phi_h1
        result[:, em_start + 2] = _safe_log10(mag_h2)
        result[:, em_start + 3] = phi_h2
```

### D14 — Diagrama Noise × FV × GS (OBRIGATÓRIO em pipeline.py)

```python
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  INTERAÇÃO NOISE × FV × GS (Fidelidade Física LWD)                      │
# ├──────────────────────────────────────────────────────────────────────────┤
# │                                                                           │
# │  CORRETO (on-the-fly, dentro de tf.data.map):                            │
# │    raw Re/Im → noise(σ) → FV(noisy) → GS(noisy) → scale → modelo       │
# │                                    │                                      │
# │                          GS veem ruído ✓ (fidelidade LWD)                │
# │                                                                           │
# │  ERRADO (offline, bug do legado C28):                                    │
# │    raw Re/Im → FV(clean) → GS(clean) → scale → noise → modelo           │
# │                                    │                                      │
# │                          GS nunca veem ruído ✗ (bias sistemático)        │
# │                                                                           │
# │  REGRAS:                                                                  │
# │    1. Scaler SEMPRE fitado em dados LIMPOS (FV+GS clean, temporário)     │
# │    2. Val/test transformados offline (sem noise, FV+GS+scale)             │
# │    3. Train permanece raw para noise on-the-fly via tf.data.map()        │
# │    4. NUNCA aplicar noise offline quando FV ou GS estão ativos            │
# └──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Checklist Rápido para Geração de Código

Antes de entregar QUALQUER código ao usuário, verificar TODOS os itens:

- [ ] Usa `config: PipelineConfig` como parâmetro (não globals)
- [ ] Usa `logging` (não print)
- [ ] Constantes físicas corretas (`20000.0`, `1.0`, `600`, `[1,4,5,20,21]`, `[2,3]`)
- [ ] `eps = 1e-12` (não `1e-30`)
- [ ] Target scaling = `"log10"` (não `"log"`)
- [ ] Split by model (não por amostra)
- [ ] Scaler fit em dados LIMPOS
- [ ] Noise on-the-fly (não offline) quando FV/GS ativos
- [ ] Output preserva dimensão temporal: `(batch, N, channels)`
- [ ] TensorFlow/Keras (não PyTorch)
- [ ] Testes incluídos ou existentes para o módulo
- [ ] Mega-header D1 presente no módulo
- [ ] Seções com headers D2 e 4+ linhas de contexto
- [ ] Docstrings Google-style D5/D6 em todas as funções/classes
- [ ] Diagramas ASCII D3 quando ≥ 3 caminhos/categorias
- [ ] Cross-references D12 em funções públicas

---

## 6. Workflow de Desenvolvimento

```
ANTES de implementar:
  1. Consultar docs/ARCHITECTURE_v2.md para decisões arquiteturais
  2. Consultar docs/physics/ para questões físicas
  3. Planejar com Plan agent se 3+ etapas

DURANTE implementação:
  4. Código SEMPRE recebe config: PipelineConfig
  5. Factory/Registry para componentes
  6. logging (NUNCA print)
  7. Consultar context7-plugin para docs TF/Keras

APÓS implementar:
  8. pytest tests/ -v --tb=short (obrigatório antes de qualquer commit)
  9. Revisar com feature-dev:code-reviewer
  10. Verificar com superpowers:verification-before-completion
```

---

## 7. Hierarquia de Consulta

| Prioridade | Documento | Quando Consultar |
|:----------:|:----------|:----------------|
| 1ª | `docs/physics/errata_valores.md` | Qualquer código com constantes físicas |
| 2ª | `geosteering_ai/config.py` | PipelineConfig, defaults, validações |
| 3ª | `docs/ARCHITECTURE_v2.md` | Decisões arquiteturais |
| 4ª | `CLAUDE.md` | Regras, proibições, code patterns |
| 5ª | `docs/physics/perspectivas.md` | Features P2-P5, interações |
| 6ª | `docs/reference/losses_catalog.md` | 26 losses, decisão de escolha |
| 7ª | `docs/reference/noise_catalog.md` | 34+ noise types, curriculum |
| 8ª | `docs/reference/arquiteturas_resumo.md` | 48 arquiteturas, tiers, causal compat |

---

*Sub-skill de padrões de código — Geosteering AI v2.0 | Última atualização: 2026-04-29*
