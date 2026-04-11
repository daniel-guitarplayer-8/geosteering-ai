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
- **FREQUENCY_HZ = 2.0** — o default e 20000.0 (range valido: 100–1e6 Hz, derivado do .out)
- **SPACING_METERS = 1000.0** — o default e 1.0 (range valido: 0.1–10.0 m)
- **SEQUENCE_LENGTH = 601** — o default e 600 (range valido: 10–100000, derivado do .out)
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

## Regras de Documentação (Invioláveis)

- **Acentuação em PT-BR obrigatória** — TODOS os documentos MD em língua portuguesa (Brasil)
  e TODA documentação de código gerada (comentários, docstrings, headers) DEVEM ter acentuação
  garantida. Nunca escrever "implementacao", "configuracao", "nao", "funcao" etc. — sempre
  "implementação", "configuração", "não", "função". Esta regra se aplica a qualquer arquivo
  gerado ou editado, incluindo comentários em blocos de código Fortran, Python e YAML.

- **Geração de relatórios MD — perguntar antes** — Por padrão, NÃO gere arquivos MD de
  relatórios/análises sem solicitação explícita do usuário. Se a geração parecer importante
  para documentar uma decisão ou análise, PERGUNTE PRIMEIRO ao usuário antes de criar o
  arquivo. Exceção: arquivos MD que são atualização de documentos existentes no `docs/`
  (ex.: ROADMAP.md, documentacao_simulador_fortran_otimizado.md) podem ser editados sem
  confirmação quando o contexto da tarefa indica claramente que é esperado.

---

## Valores Fisicos Criticos (Errata Imutavel)

```python
# Validados automaticamente por PipelineConfig.__post_init__()
assert 100.0 <= FREQUENCY_HZ <= 1e6   # Default 20000.0 (20 kHz)
assert 0.1 <= SPACING_METERS <= 10.0  # Default 1.0 m
assert 10 <= SEQUENCE_LENGTH <= 100000 # Default 600 (Inv0Dip 0 graus)
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
├── models/                ← 48 arquiteturas (9 famílias) + ModelRegistry
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

## Padroes de Documentacao (D1-D14)

Todo codigo em `geosteering_ai/` DEVE seguir os padroes D1-D14 definidos na
skill `geosteering-v2` (secao 15). Resumo:

| Padrao | Requisito | Onde |
|:-------|:----------|:-----|
| **D1** | Mega-header Unicode com 14 campos | Topo de cada modulo .py |
| **D2** | Cabecalho de secao com 4+ linhas de contexto | Cada bloco logico |
| **D3** | Diagramas ASCII com Unicode borders | >= 3 caminhos/categorias |
| **D4** | Atributos de config com 4+ linhas por grupo | config.py |
| **D5** | Docstrings Google-style com 5+ campos | Todas as funcoes |
| **D6** | Docstrings de classes com Attributes + Example | Todas as classes |
| **D7** | Comentarios inline semanticos | Operacoes de dominio |
| **D8** | Inventario de exports com __all__ semantico | Todos os modulos |
| **D9** | Logging estruturado (NUNCA print) | Toda saida |
| **D10** | Constantes com documentacao fisica | Valores criticos |
| **D11** | Tabelas de formulas ASCII em catalogos | Catalogos de componentes |
| **D12** | Cross-references Note: em docstrings | Toda funcao publica |
| **D13** | Branch comments com layout de saida | Toda transformacao if/elif |
| **D14** | Diagrama noise × FV × GS | pipeline.py |

Referencia completa: `/geosteering-v2` secao 15.

### Nivel de Riqueza Documental (padrao C28 legado)

O codigo v2.0 DEVE ter a mesma profundidade de documentacao do legado C28.
O codigo e um **documento de referencia executavel** — lido como tutorial tecnico.
Cada funcao publica DEVE incluir:

**1. Diagramas ASCII de arquitetura dentro de docstrings (OBRIGATORIO)**
```python
def build_resnet18(input_shape, output_channels, use_causal, **kwargs):
    """Constroi modelo ResNet-18 para inversao 1D de resistividade.

    Estrutura do modelo:
      ┌──────────────────────────────────────────────────────────┐
      │  Input (batch, None, N_FEATURES)                        │
      │    ↓                                                    │
      │  Stem: Conv1D(64, 7) → BN → ReLU → Dropout            │
      │    ↓                                                    │
      │  Stage 1: 2× ResidualBlock(64)  + SE opcional          │
      │  Stage 2: 2× ResidualBlock(128) + SE opcional          │
      │  ...                                                    │
      │  Output: Dense(output_channels, 'linear')               │
      └──────────────────────────────────────────────────────────┘
    """
```

**2. Significado fisico de CADA parametro (OBRIGATORIO)**
```python
    Args:
        input_shape: Shape (None, N_FEATURES). None permite sequencia
            de comprimento variavel (multi-angulo). Default: (None, 5)
            para P1 baseline [z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)].
        kernel_size: Tamanho do kernel Conv1D. Default: 3.
            Kernel 3 corresponde a ~3 metros de campo receptivo por camada
            (SPACING_METERS=1.0). Kernels maiores (7, 11) capturam
            dependencias de longo alcance em modelos com camadas espessas.
```

**3. Referencias bibliograficas com contribuicao explicita (OBRIGATORIO)**
```python
    Note:
        Ref: He et al. "Deep Residual Learning for Image Recognition"
        (CVPR 2016) — skip connections estabilizam gradientes em redes
        profundas, permitindo treinar 4 estagios sem degradacao.
```

**4. Explicacao dual-mode (offline vs causal) quando aplicavel**
```python
    Dual-mode:
      ┌────────────────────────────────────────────────────────┐
      │  "same"   →  Offline (acausal, batch completo)        │
      │  "causal" →  Realtime (causal, sliding window)        │
      └────────────────────────────────────────────────────────┘
```

**5. Blocos de comentario com 4+ linhas antes de cada operacao significativa**
```python
    # ── Stem: Conv1D(64, 7) → BN → Act ────────────────────────
    # Convolution inicial de campo receptivo maior (7×1) para capturar
    # features EM de baixa frequencia. strides=1 preserva N_MEDIDAS.
    # Kernel 7 cobre ~7m de profundidade no perfil de poco,
    # ideal para detectar contrastes de camada espessa.
    x = Conv1D(64, 7, padding=_padding, strides=1, ...)(x)
```

**6. Tabelas ASCII em modulos que implementam catalogos**
```python
    #   ┌──────────────────────┬────────────┬────────────────────────┐
    #   │  Modelo              │  Blocos    │  Tipo                  │
    #   ├──────────────────────┼────────────┼────────────────────────┤
    #   │  ResNet_18 (★)       │  8         │  ResidualBlock         │
    #   │  ResNet_34           │  16        │  ResidualBlock         │
    #   │  ...                 │            │                        │
    #   └──────────────────────┴────────────┴────────────────────────┘
```

**7. Branch comments com layout de saida em transformacoes**
```python
    if view == "H1_logH2":
        # ── H1_logH2: H1 cru preserva SNR em alta atenuacao,
        #    H2 log10-transformado comprime faixa dinamica larga de Hzz.
        #    Saida: [Re(H1), Im(H1), log10|H2|, φ(H2)]
        #    Motivacao fisica: Hzz varia 4+ ordens de magnitude.
```

**Regra:** Se o codigo legado C28 documentava uma operacao com N linhas
de contexto, a versao v2.0 DEVE ter pelo menos N linhas equivalentes.
O codigo e lido por geofisicos e engenheiros de poco — nao apenas
por desenvolvedores de software.

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
| **consensus** | Busca de artigos cientificos (Semantic Scholar + ArXiv). Opcao A — MCP Server (padrao) |
| **Figma** | (Disponivel, nao utilizado neste projeto) |

### Skills Disponiveis

| Skill | Uso |
|:------|:----|
| **geosteering-v2** | Skill PRINCIPAL v2.0 — dominio fisico (geofisica/petrofisica/EM), padroes de codigo v2.0 (PipelineConfig/Factory), DL aplicado a geociencias. Usar para TODAS as questoes do projeto |
| **geosteering-simulator-python** | Simulador Python otimizado (JAX + Numba). Cobre: `geosteering_ai/simulation/`, FilterLoader, HankelFilter, `extract_hankel_weights.py`, plano de 7 fases, metas de performance CPU/GPU. Usar para qualquer questão sobre o simulador Python em desenvolvimento em `feature/simulator-python` |
| **geosteering-simulator-fortran** | Simulador Fortran v10.0 (`PerfilaAnisoOmp.f08` / `tatu.x`). Cobre: `Fortran_Gerador/`, módulos F08, Makefile, OpenMP, Features F5/F6/F7/F10, Jacobiano ∂H/∂ρ, f2py wrapper |
| **geosteering-v5015** | Skill LEGADA para celulas C0-C73. Usar apenas para referencia ao codigo legado |
| **consensus-search** | Pesquisa cientifica multi-fonte — Semantic Scholar + ArXiv + WebSearch. Fase A (imediata) |
| **arxiv-search** | Busca em repositorios abertos — ArXiv + Semantic Scholar. Opcao B (sem API key) |
| **feature-dev** | Desenvolvimento guiado de features com foco em arquitetura |
| **code-review** | Revisao de PRs e codigo |
| **context7-plugin:docs** | Busca de documentacao de bibliotecas |

### Integracao Cientifica (Consensus)

O projeto integra pesquisa cientifica ao fluxo de desenvolvimento via 3 fases:

| Fase | Implementacao | Status |
|:-----|:-------------|:-------|
| **A** | Skill `/consensus-search` via WebFetch | Ativa |
| **B** | MCP Server `tools/consensus-mcp-server/` | Pronto (ativar em settings.json) |
| **C** | Hook `validate-scientific-refs.sh` | Pronto (ativar em settings.json) |

4 opcoes de busca disponiveis:

| Opcao | Tipo | API Key | Recomendacao |
|:------|:-----|:-------:|:-------------|
| **A** | MCP Server (padrao) | Opcional (S2_API_KEY) | Desenvolvimento continuo |
| **B** | Skill /arxiv-search | Nenhuma | Exploracao rapida |
| **C** | WebSearch/WebFetch direto | Nenhuma | Buscas pontuais |
| **D** | Hook automatico | Nenhuma | Lembrete passivo |

Ref: `docs/reference/consensus_integration.md` (guia completo).

### Workflow com Agentes (Recomendado)

```
ANTES de implementar:
  1. Explore agent → entender codigo existente e dependencias
  2. Plan agent → planejar implementacao com etapas claras
  3. /consensus-search → validar decisoes com literatura cientifica

DURANTE implementacao:
  4. Claude Code → editar arquivos, rodar testes
  5. context7 → consultar docs do TF/Keras quando necessario
  6. /arxiv-search → buscar preprints para novas tecnicas

APOS implementar:
  7. code-reviewer agent → revisar bugs, seguranca, qualidade
  8. code-simplifier agent → otimizar legibilidade
  9. pytest → validar automaticamente
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
