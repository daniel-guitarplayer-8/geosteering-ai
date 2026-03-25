# Geosteering AI — Instrucoes para Claude Code v2.0

## Identidade do Projeto

| Atributo | Valor |
|:---------|:------|
| **Projeto** | Inversao 1D de Resistividade via Deep Learning para Geosteering |
| **Versao** | v2.0 (arquitetura de software) |
| **Autor** | Daniel Leal |
| **Framework** | TensorFlow 2.x / Keras **EXCLUSIVO** (PyTorch PROIBIDO) |
| **Ambiente** | VSCode + Claude Code (dev) · GitHub (CI) · Google Colab Pro+ GPU (exec) |
| **Linguagem** | Python 3.10+ · Variaveis em ingles · Comentarios/docs em PT-BR |
| **Repositorio** | `github.com/daniel-leal/geosteering-ai` |
| **Pacote** | `geosteering_ai/` (pip installable) |
| **Referencia** | `docs/ARCHITECTURE_v2.md` (documento completo da arquitetura) |

---

## Proibicoes Absolutas

- **PyTorch** — PROIBIDO em qualquer parte do pipeline
- **FREQUENCY_HZ = 2.0** — o valor correto e 20000.0
- **SPACING_METERS = 1000.0** — o valor correto e 1.0
- **SEQUENCE_LENGTH = 601** — o valor correto e 600
- **TARGET_SCALING = "log"** — o valor correto e "log10"
- **INPUT_FEATURES = [0, 3, 4, 7, 8]** — o correto e [1, 4, 5, 20, 21] (22-col)
- **OUTPUT_TARGETS = [1, 2]** — o correto e [2, 3] (22-col)
- **eps_tf = 1e-30** — usar 1e-12 para float32
- **Split por amostra** — SEMPRE split por modelo geologico [P1]
- **Scaler fit em dados ruidosos** — SEMPRE fit em dados LIMPOS
- **Noise offline com GS** — on-the-fly e o UNICO path fisicamente correto
- **globals().get()** — usar PipelineConfig (ponto unico de verdade)
- **print()** — usar logging (logger do utils/)

---

## Valores Fisicos Criticos (Errata Imutavel)

```python
# Validados automaticamente por PipelineConfig.__post_init__()
assert FREQUENCY_HZ == 20000.0         # NUNCA 2.0 Hz
assert SPACING_METERS == 1.0           # NUNCA 1000.0 m
assert SEQUENCE_LENGTH == 600          # NUNCA 601
assert TARGET_SCALING == "log10"       # NUNCA "log"
assert INPUT_FEATURES == [1,4,5,20,21] # NUNCA [0,3,4,7,8]
assert OUTPUT_TARGETS == [2,3]         # NUNCA [1,2]

# Decoupling (L = 1.0 m):
#   ACp = -1/(4*pi*L^3) ≈ -0.079577  (planar: Hxx, Hyy)
#   ACx = +1/(2*pi*L^3) ≈ +0.159155  (axial: Hzz)
```

---

## Arquitetura de Software

### Estrutura do Pacote

```
geosteering_ai/
├── config.py              ← PipelineConfig dataclass
├── data/                  ← Loading, splitting, FV, GS, scaling, DataPipeline
├── noise/                 ← On-the-fly noise (gaussian, curriculum)
├── models/                ← 44 arquiteturas + ModelRegistry
├── losses/                ← 26 losses + LossFactory
├── training/              ← TrainingLoop, callbacks, N-Stage
├── inference/             ← InferencePipeline, realtime, export
├── evaluation/            ← Metricas, comparacao
├── visualization/         ← Plots, Picasso, EDA
└── utils/                 ← Logger, timer, validation, formatting
```

### Code Patterns Obrigatorios

**1. PipelineConfig como parametro (NUNCA globals)**
```python
# CORRETO:
def build_model(config: PipelineConfig) -> tf.keras.Model:
    ...

# PROIBIDO:
def build_model():
    model_type = globals().get("MODEL_TYPE", "ResNet_18")  # NAO!
```

**2. Factory Pattern para componentes**
```python
model = ModelRegistry().build(config)        # NAO: build_model(MODEL_TYPE)
loss_fn = LossFactory.get(config)            # NAO: if LOSS_TYPE == "rmse": ...
callbacks = build_callbacks(config, model, noise_var)  # NAO: 550 linhas imperativas
```

**3. DataPipeline com cadeia explicita**
```python
pipeline = DataPipeline(config)
data = pipeline.prepare(dataset_path)        # raw → split → fit_scaler
map_fn = pipeline.build_train_map_fn(noise_var)  # noise → FV → GS → scale
```

**4. Presets YAML para reprodutibilidade**
```python
config = PipelineConfig.from_yaml("configs/robusto.yaml")
# OU presets de classe:
config = PipelineConfig.robusto()
config = PipelineConfig.nstage(n=3)
config = PipelineConfig.geosinais_p4()
config = PipelineConfig.realtime(model_type="WaveNet")
```

### Cadeia de Dados Fisicamente Correta

```
train_raw → noise(A/m) → FV_tf(noisy) → GS_tf(noisy) → scale → modelo
                                              │
                                    GS veem ruido ✓ (fidelidade LWD)
```

Scaler fitado em dados LIMPOS (FV+GS clean, temporario).
Val/test transformados offline. Train permanece raw para on-the-fly.

---

## Workflow de Desenvolvimento

### Ciclo: Editar → Testar → Commitar → CI → Treinar

```
1. Claude Code edita geosteering_ai/*.py (local)
2. pytest tests/ (local, CPU)
3. git commit + push → GitHub
4. GitHub Actions CI: compile + pytest + mypy
5. Google Colab: pip install git+...@tag → treinar com GPU
```

### Testes Obrigatorios

Antes de qualquer commit:
```bash
pytest tests/ -v --tb=short
```

Testes minimos por modulo:
- `test_config.py` — errata, mutual exclusivity, YAML roundtrip
- `test_models.py` — forward pass para cada arquitetura
- `test_data_pipeline.py` — shapes, split P1, scaler fit on clean
- `test_noise.py` — curriculum 3-phase, noise preserva zobs
- `test_losses.py` — forward pass + gradients para 26 losses

---

## Plugins e Agentes Especializados

### Agentes Claude Code Utilizados

| Agente | Uso | Quando Invocar |
|:-------|:----|:---------------|
| **Explore** (subagent_type) | Busca em codebase, leitura de arquivos, analise | Entender codigo existente, encontrar funcoes |
| **Plan** (subagent_type) | Planejamento de implementacao | Antes de tarefas com 3+ etapas |
| **feature-dev:code-reviewer** | Revisao de codigo | Apos implementar modulo, antes de commit |
| **feature-dev:code-explorer** | Analise de features existentes | Entender padrao de codigo legado |
| **feature-dev:code-architect** | Design de features | Planejar novo modulo ou refatoracao |
| **code-simplifier** | Simplificacao e limpeza | Apos modulo pronto, otimizar legibilidade |
| **general-purpose** | Pesquisa, tarefas complexas | Tarefas multi-step que requerem autonomia |

### Plugins MCP Disponiveis

| Plugin | Uso |
|:-------|:----|
| **context7-plugin** | Buscar documentacao atualizada de TensorFlow, Keras, NumPy, scikit-learn |
| **Figma** | (Disponivel, nao utilizado neste projeto) |

### Skills Disponiveis

| Skill | Uso |
|:------|:----|
| **geosteering-v5015** | Skill LEGADA para celulas C0-C73. Usar apenas para referencia ao codigo legado |
| **feature-dev** | Desenvolvimento guiado de features com foco em arquitetura |
| **code-review** | Revisao de PRs e codigo |
| **context7-plugin:docs** | Busca de documentacao de bibliotecas |

### Workflow com Agentes (Recomendado)

```
ANTES de implementar:
  1. Explore agent → entender codigo existente e dependencias
  2. Plan agent → planejar implementacao com etapas claras

DURANTE implementacao:
  3. Claude Code → editar arquivos, rodar testes
  4. context7 → consultar docs do TF/Keras quando necessario

APOS implementar:
  5. code-reviewer agent → revisar bugs, seguranca, qualidade
  6. code-simplifier agent → otimizar legibilidade
  7. pytest → validar automaticamente
```

---

## Hierarquia de Consulta

| Prioridade | Documento | Quando Consultar |
|:----------:|:----------|:----------------|
| 1a | `docs/ARCHITECTURE_v2.md` | Qualquer decisao arquitetural |
| 2a | `CLAUDE.md` (este arquivo) | Regras, proibicoes, code patterns |
| 3a | `geosteering_ai/config.py` | FLAGS, defaults, validacoes |
| 4a | `configs/*.yaml` | Presets de configuracao |
| 5a | Docstrings no codigo | API e uso de funcoes |
| 6a | `docs/physics/` | Contexto fisico (tensor EM, GS, FV) |
| 7a | `docs/reference/` | Catalogos (arquiteturas, losses, noise) |
| 8a | `legacy/` | Codigo legado C0-C47 (referencia historica) |

---

## Principios de Engenharia

- **Config como parametro:** Toda funcao recebe `config: PipelineConfig`, nunca le globals
- **Factory para componentes:** ModelRegistry, LossFactory, build_callbacks
- **Um unico path de dados:** DataPipeline com cadeia raw→noise→FV→GS→scale
- **Testes antes de commit:** pytest obrigatorio, CI no GitHub
- **Impacto minimo:** Alterar apenas o necessario, sem over-engineering
- **Fail-fast:** PipelineConfig valida errata no __post_init__
- **Reprodutibilidade:** config YAML + tag GitHub + seed = resultado identico
- **DRY:** Geosteering e modo (flag), nao modulo duplicado
