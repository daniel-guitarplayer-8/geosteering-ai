# Pipeline de Inversao Geofisica com Deep Learning — v5.0.15

## Identidade do Projeto

| Atributo | Valor |
|:---------|:------|
| **Projeto** | Pipeline de Inversao 1D de Resistividade via Deep Learning |
| **Versao** | v5.0.15 |
| **Autor** | Daniel Leal |
| **Framework** | TensorFlow 2.x / Keras **EXCLUSIVO** (PyTorch PROIBIDO) |
| **Ambiente** | Google Colab Pro+ com GPU + Google Drive montado |
| **Linguagem** | Python 3.10+, variaveis em ingles, comentarios/docs em PT-BR |
| **Celulas** | 74 (C0-C73) em 8 secoes operacionais |
| **FLAGS** | ~1.185 com defaults conservadores |
| **Arquiteturas** | 44 modelos (39 standard + 5 geosteering) |
| **Losses** | 26 funcoes (13 genericas + 4 geofisicas + 8 geosteering + 1 hybrid) |
| **Perspectivas** | P1 (baseline), P2 (theta), P3 (f), P4 (geosinais), P5 (Picasso/DTB) |

---

## Proibicoes Absolutas

- **PyTorch** — PROIBIDO em qualquer parte do pipeline
- **Hardcoding de caminhos** — usar variaveis de C3 (BASE_DIR, DATASET_DIR, etc.)
- **Hardcoding de N_MEDIDAS** — ler do .out via C19
- **fit() do scaler em dados ruidosos** — SEMPRE em x_train_clean
- **Remover funcoes/FLAGS na limpeza** — APENAS variaveis temporarias
- **FREQUENCY_HZ = 2.0** — o valor correto e 20000.0
- **SEQUENCE_LENGTH = 601** — o valor correto e 600
- **TARGET_SCALING = "log"** — o valor correto e "log10"
- **SPACING_METERS = 1000.0** — o valor correto e 1.0
- **print()** — usar logger.info() para toda saida
- **Shape fixo no Input** — usar Input(shape=(None, N_FEATURES))
- **metadata= em segregate_by_angle** — usar out_metadata=
- **Formato 12-col como ativo** — 22-col e o formato ATIVO, 12-col e LEGADO
- **INPUT_FEATURES = [0, 3, 4, 7, 8]** — o valor correto e [1, 4, 5, 20, 21]
- **OUTPUT_TARGETS = [1, 2]** — o valor correto e [2, 3]
- **Split por amostra** — PROIBIDO: usar split por MODELO GEOLOGICO (P1)
- **Scaler unico para EM+GS** — usar per-group scalers: StandardScaler(EM) + RobustScaler(GS) (P3)
- **eps_tf = 1e-30 em TensorFlow** — usar eps_tf = 1e-12 para float32 (P7)
- **FV/GS estaticos com ruido on-the-fly** — recomputar FV/GS APOS noise no map_fn (P7/P8)

---

## Valores Fisicos Criticos (Errata v4.4.5 + v5.0.15)

```python
assert FREQUENCY_HZ == 20000.0,    "Errata v4.4.5: NUNCA 2.0 Hz"
assert SPACING_METERS == 1.0,      "Errata v4.4.5: NUNCA 1000.0 m"
assert SEQUENCE_LENGTH == 600,     "NUNCA 601"
# TARGET_SCALING = "log10"         (NUNCA "log")
# DEPTH_MAX = 150.0                (NUNCA 6000.0)
# Decoupling: ACp = -1/(4*pi*L^3), ACx = +1/(2*pi*L^3), L = 1.0 m

# Errata v5.0.15 — Mapeamento de colunas 22-col (fonte: PerfilaAnisoOmp.f08):
# INPUT_FEATURES = [1, 4, 5, 20, 21]  (NUNCA [0, 3, 4, 7, 8])
#   Col 1=zobs, Col 4=Re(Hxx), Col 5=Im(Hxx), Col 20=Re(Hzz), Col 21=Im(Hzz)
# OUTPUT_TARGETS = [2, 3]              (NUNCA [1, 2])
#   Col 2=res_h, Col 3=res_v
# Col 0=meds (meta, NAO usar como feature)
```

---

## Assinaturas Criticas de Funcoes

```python
ValidationTracker.check(condition, description)     # SEMPRE 2 args
print_header(title, width)                          # SEMPRE (title, width)
segregate_by_angle(data_2d=..., out_metadata=...)   # out_metadata= (NAO metadata=)
```

---

## Estrutura do Pipeline (74 Celulas)

```
SECAO 0: INFRAESTRUTURA       C0-C2     (3)   -> Logger, seeds, imports
SECAO 1: CONFIGURACAO/FLAGS   C3-C18    (16)  -> ~1.159 FLAGS
SECAO 2: PREPARACAO DE DADOS  C19-C26   (8)   -> Datasets tf.data
SECAO 3: ARQUITETURAS         C27-C39   (13)  -> Modelo Keras
SECAO 4: TREINAMENTO          C40-C47   (8)   -> Modelo treinado
SECAO 5: AVALIACAO            C48-C57   (10)  -> Metricas
SECAO 6: VISUALIZACAO         C58-C65   (8)   -> Plots/relatorios
SECAO 7: GEOSTEERING          C66-C73   (8)   -> Inferencia realtime
```

### Status de Implementacao
- **C0:** EXPANDIDA v5.0.15 — 1800 linhas, 46 exports, 17 PARTEs
- **C1-C7:** RECONSTRUIDAS do zero como v5.0.15 nativo
- **C8-C73:** A implementar do zero

---

## Estrategia de Implementacao: Arquivos .py Individuais

### Abordagem
Cada celula e gerada como arquivo `.py` individual. Ao final, um script
`assemble_notebook.py` monta todos os `.py` em um notebook `.ipynb` para o Colab.

### Convencao de Nomes
- Formato: `C{N}_{NomeDescritivo}.py` (ex: `C19_DataLoading_OutMetadata.py`)
- Diretorio: `/Users/daniel/Geosteering_AI/Arquivos_Projeto_Claude/`

### Protocolo de Retomada entre Sessoes
```
1. Claude le CLAUDE.md (automatico) + MEMORY.md (automatico)
2. Revisar secao "Licoes Aprendidas" em MEMORY.md (se existir)
3. MEMORY.md indica: "ultima_celula_gerada = C{K}"
4. Claude lista arquivos C*.py para confirmar
5. Continua a partir de C{K+1}
```

### Montagem Final do Notebook
- Script `assemble_notebook.py` le todos os C*.py em ordem
- Insere celulas Markdown entre secoes
- Gera `Pipeline_Inversao_Geofisica_v5015.ipynb` para Google Colab Pro+

---

## Shapes Obrigatorios

```python
# Entrada: SEMPRE (batch, None, N_FEATURES) — None para multi-angulo
# Saida:   SEMPRE (batch, N_MEDIDAS, OUTPUT_CHANNELS)
# OUTPUT_CHANNELS: 2=[rho_h, rho_v] | 4=[+sigma] | 6=[+DTB, rho_adj]
# Input(shape=(None, N_FEATURES)) — NUNCA shape fixo
```

---

## Formato de Dados

- **22-col = ATIVO** (tensor completo 3x3, binario stream)
- **12-col = LEGADO** (codigo Fortran comentado)
- Layout = **MODEL-MAJOR** (NAO ANGLE-MAJOR)
- theta e f **NAO existem** no .dat de 22-col -> obtidos do .out
- `total_linhas = nm * SUM(nmeds[k] * nf)`

---

## Pipeline de Dados Adaptado (v5.0.15+ — Solucoes P1-P8)

### Cenario Padrao: 2D (EM + FV + GS + Noise On-the-Fly)

O cenario **2D** e o padrao recomendado para producao e geosteering.
Detalhes completos: `Skill/PIPELINE_EXECUTION_FLOWS.md`

### 8 Solucoes Obrigatorias

| # | Problema | Solucao | Celulas |
|:-:|:---------|:--------|:--------|
| P1 | Data leakage (split por amostra) | Split por **modelo geologico** | C19 |
| P2 | Assimetria train/val | **Validacao dual** (val_clean + val_noisy) | C24, C40 |
| P3 | Scaler EM+GS misturado | **Per-group scalers** + RobustScaler p/ GS | C23 |
| P4 | Replicacao targets off-line | **Shuffle buffer >= K*N** + sample_weight | C24 |
| P5 | Off-line vs curriculum | **Tier-switching** ou usar on-the-fly | C24, C40 |
| P6 | Inferencia inconsistente | **InferencePipeline** exportavel | C25 |
| P7 | NaN/Inf geosinais TF | **eps_tf=1e-12** + tf.where guards | C22, C24 |
| P8 | Vies estatico FV/GS clean | Documentar como **baseline only** | C22, doc |

### Ordem de Execucao com Ruido On-the-Fly (Cenarios 2x)

```
C19(load + split modelo) -> C22(registra FV/GS ops) -> C23(fit scaler clean) -> C24(orquestra)
                                                                                    |
                                          tf.data.Dataset.map(train_map_fn):        |
                                            1. Noise em Re/Im brutas                |
                                            2. Feature View (se != identity)         |
                                            3. Geosinais on-the-fly (se habilitado)  |
                                            4. Scaling per-group                     |
```

### Novas FLAGS do Pipeline Adaptado

| FLAG | Tipo | Default | Celula | Descricao |
|:-----|:----:|:-------:|:------:|:----------|
| DATA_SCENARIO | str | "2D" | C4 | Cenario de dados (1A-3D) |
| SPLIT_BY_MODEL | bool | True | C4 | Split por modelo geologico [P1] |
| USE_DUAL_VALIDATION | bool | True | C4 | Validacao dual [P2] |
| USE_PER_GROUP_SCALERS | bool | True | C6 | Scalers separados EM/GS [P3] |
| GS_SCALER_TYPE | str | "robust" | C6 | Scaler para geosinais [P3] |
| EPS_TF | float | 1e-12 | C22 | Guard numerico TF [P7] |
| EXPORT_INFERENCE_PIPELINE | bool | True | C25 | Exportar InferencePipeline [P6] |
| NOISE_GAP_THRESHOLD | float | 0.5 | C14 | Limiar gap val_clean-val_noisy [P2] |
| PINNS_HARD_CONSTRAINT | bool | False | C13 | Hard constraint layer na saida (Morales) |
| PINNS_CONSTRAINT_ACTIVATION | str | "softplus" | C13 | Ativacao da constraint layer |
| PINNS_LAMBDA_SCHEDULE | str | "linear" | C13 | Schedule de annealing do lambda |
| PINNS_PHYSICS_NORM | str | "l2" | C13 | Norma L_physics (l1/l2/huber) |
| PINNS_DATA_NORM | str | "l2" | C13 | Norma L_data com PINNs ativo |
| USE_PETROPHYSICAL_INVERSION | bool | False | C13 | Modulo Klein TI pos-inversao (Morales) |
| USE_PHYSICAL_CONSTRAINT_LAYER | bool | False | C7 | Constraint layer na saida (Morales) |
| CONSTRAINT_ACTIVATION | str | "softplus" | C7 | Ativacao da constraint layer |
| USE_MORALES_HYBRID_LOSS | bool | False | C10 | Loss L2+L1 hibrida (Morales) |
| MORALES_PHYSICS_OMEGA | float | 0.85 | C10 | Peso omega da fisica (Morales) |
| USE_ENSEMBLE_UQ | bool | False | C14 | UQ via perturbacao ensemble (Morales) |
| N_ENSEMBLE_UQ | int | 500 | C14 | N realizacoes do ensemble UQ |
| UQ_ENSEMBLE_KAPPA | float | 0.05 | C14 | Nivel de ruido relativo (kappa=5%) |
| USE_FIELD_DATA | bool | False | C5 | Dados de campo mistos (Morales) |
| FIELD_DATA_RATIO | float | 0.3 | C5 | Proporcao campo/sintetico |

### Novos Componentes

- **InferencePipeline** (C25): Serializa FV+GS+scalers para inferencia identica ao treino [P6]
- **DualValidationCallback** (C40): Avalia val_clean e val_noisy, monitora gap [P2]
- **EpochTierCallback** (C40): Troca datasets off-line por tier de ruido [P5]
- **compute_expanded_input_features()** (C22): Auto-detecta componentes EM para GS [P7]
- **split_by_geological_model()** (C19): Split por indice de modelo, nao por amostra [P1]

---

## Dual-Mode (Offline vs Realtime)

| Aspecto | Offline (padrao) | Realtime |
|:--------|:-----------------|:---------|
| Dados | Batch completo | Sliding window |
| Rede | Acausal (44 arqs) | Causal (27 arqs) |
| Saida | (batch, N, 2) | (1, W, 2-6) |
| Incerteza | Opcional | Automatica |
| Padding | "same" | "causal" |

---

## Hierarquia de Consulta (Arquivos Autoritativos)

Ao gerar codigo ou responder perguntas, SEMPRE consultar nesta ordem:

| Prioridade | Arquivo | Quando consultar |
|:----------:|:--------|:----------------|
| 1a | `Arquivos_Projeto_Claude/Skill/ERRATA_E_VALORES.md` | Qualquer codigo com constantes fisicas |
| 2a | `Arquivos_Projeto_Claude/Skill/CHECKLIST.md` | Revisao ou auditoria de celula |
| 3a | `Arquivos_Projeto_Claude/Skill/ARQUITETURAS.md` | Codigo de C28-C37, Model Factory |
| 4a | `Arquivos_Projeto_Claude/Skill/PERSPECTIVAS.md` | FLAGS de P2/P3/P4/P5, interacoes |
| 5a | `Arquivos_Projeto_Claude/Skill/LOSSES_CATALOG.md` | 26 losses (13 gen + 4 geo + 8 geosteering + 1 hybrid) |
| 6a | `Arquivos_Projeto_Claude/Skill/NOISE_CATALOG.md` | 43 tipos de ruido, curriculum learning |
| 7a | `Arquivos_Projeto_Claude/Skill/PIPELINE_EXECUTION_FLOWS.md` | 12 cenarios de dados, solucoes P1-P8 **NOVO** |
| 8a | `Arquivos_Projeto_Claude/Skill/SECTION3_GUIDE.md` | Geracao C27-C39, build_* template, shapes |
| 9a | `Arquivos_Projeto_Claude/Skill/SKILL.md` | Workflow de geracao, regras gerais |
| 10a | `Arquivos_Projeto_Claude/MANUAL_IMPLEMENTACAO_COMPLETO_v5014.md` | Inputs/outputs/deps de QUALQUER celula (C0-C73) |
| 11a | `Arquivos_Projeto_Claude/DOCUMENTACAO_COMPLETA_SOFTWARE_v5_0_14.md` | Detalhes tecnicos de qualquer secao (1-39) |
| 12a | `Arquivos_Projeto_Claude/ONBOARDING_PIPELINE_v5014.md` | Contexto fisico/geofisico, novo desenvolvedor |

---

## 44 Arquiteturas — Compatibilidade Causal

| Categoria | Arquiteturas | Total |
|:----------|:-------------|:-----:|
| **Nativas causais** | WaveNet, Causal_Transformer, TCN, Mamba_S4, LSTM, Encoder_Forecaster | 6 |
| **Adaptaveis** | ResNet_18/34/50, ConvNeXt, InceptionNet/Time, CNN_1D, CNN_LSTM, TCN_Advanced, Simple_TFT, TFT, PatchTST, Autoformer, iTransformer, N-HiTS, N-BEATS, DNN, FNO, Geophysical_Attention, Transformer | 21 |
| **Incompativeis (offline only)** | BiLSTM, CNN_BiLSTM_ED, UNet_Inversion, UNet_ResNet18/34/50, UNet_ConvNeXt, UNet_Inception, UNet_EfficientNet, UNet_Attention_* (6 variantes), DeepONet | 17 |

Toda arquitetura DEVE preservar N_MEDIDAS no output:
```python
# Conv1D: strides=1, padding="same" (ou "causal" em realtime)
# RNN: return_sequences=True
# Dense: TimeDistributed(Dense(OUTPUT_CHANNELS, activation='linear'))
#        OU activation=CONSTRAINT_ACTIVATION se USE_PHYSICAL_CONSTRAINT_LAYER=True
```

---

## Perspectivas P2-P5

| P# | Nome | Versao | N_FEATURES |
|:--:|:-----|:------:|:----------:|
| P1 | Baseline | v5.0.1 | 5 (z + 4 EM) |
| P2 | theta como feature | v5.0.8 | 6 |
| P3 | f como feature | v5.0.12 | 6-7 |
| P4 | Geosinais | v5.0.13 | 9-17 |
| P5 | Picasso/DTB | v5.0.15 | — (validacao + DTB target) |

**N_FEATURES maximo teorico:** 17 (P2+P3+P4 full_3d)

---

## Padrao de Codigo por Celula

Cada celula DEVE seguir este template:

```
1. Mega-Header (calcado com dupla borda) com 8+ campos
2. Timer _t0_c{N}
3. Dependency guard (_c{N}_required)
4. print_header() + 2 linhas DIM
5. PARTEs numeradas com separadores
6. ValidationTracker._vt_c{N}
7. Inventario _c{N}_exports
8. Limpeza P17 (lista explicita de temporarias)
9. Timer _t1_c{N} + format_time
10. Banner de sucesso: CELULA C{N} EXECUTADA COM SUCESSO
```

### Declaracao de FLAGS (Padrao Obrigatorio)

```python
# -- FLAG_NAME (type) -----------------------------------------------
# Descricao: [O que esta flag faz]
# Valores validos: [Lista de valores ou restricoes]
# Default: [Valor padrao com justificativa]
# Impacto: [Quais celulas usam e como]
# Referenciado em: [Lista de celulas]
# Nota de versao: [Historico de versao]
FLAG_NAME: type = default_value
```

---

## Padrao de Documentacao Detalhada (Obrigatorio)

O pipeline exige documentacao rica e detalhada em todo codigo gerado.
Codigo sem documentacao adequada NAO sera aceito. Esta secao define os
requisitos minimos de documentacao para cada tipo de elemento do codigo.

Os arquivos `C0_Logger_Seeds_Utilitarios.py` e
`C4_FLAGS_Gerais_InferenceMode.py` sao os exemplares canonicos
de documentacao. Toda celula nova DEVE seguir o mesmo nivel de detalhe.

### D1. Mega-Header Unicode (Obrigatorio — todas as celulas)

Caixa com bordas duplas Unicode (caracteres ╔ ═ ╗ ║ ╚ ╝) contendo
NO MINIMO 12 campos. NUNCA omitir campos.

```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECAO {S} — {NOME_DA_SECAO}                                              ║
# ║  C{N} — {Nome Descritivo da Celula}                                       ║
# ║                                                                            ║
# ║  Pipeline de Inversao Geofisica com Deep Learning — v5.0.15               ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: Google Colab Pro+ com GPU + Google Drive montado                ║
# ║  Secao: {S}/7 | Celula: {N}/73                                            ║
# ║  Perspectivas: P1-P5 (baseline -> Picasso/DTB)                            ║
# ║  Arquiteturas: 44 modelos (39 standard + 5 geosteering)                   ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    * {Bullet 1 — acao principal}                                           ║
# ║    * {Bullet 2 — acao secundaria}                                          ║
# ║    * {Bullet 3+ — outras acoes}                                            ║
# ║    * ~{N} FLAGS com defaults conservadores (se celula de FLAGS)            ║
# ║                                                                            ║
# ║  Dependencias: C0 ({lista de simbolos}),                                   ║
# ║                C{X} ({lista de simbolos})                                  ║
# ║  Exports: ~{N} {tipo} (ver inventario)                                     ║
# ║  Proxima: C{N+1} — {Nome da proxima celula}                               ║
# ║  DOC REF: §{secao}                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
```

Campos OBRIGATORIOS do Mega-Header (12 minimo):
1. Secao (numero e nome)
2. Celula (numero e nome descritivo)
3. Pipeline (nome e versao)
4. Autor
5. Framework (com mencao a proibicao de PyTorch)
6. Ambiente
7. Secao/Celula (formato "{S}/7 | Celula: {N}/73")
8. Perspectivas
9. Arquiteturas
10. Proposito (3+ bullets detalhados)
11. Dependencias (com lista EXPLICITA de simbolos usados)
12. Exports (com contagem e tipo)
13. Proxima celula
14. DOC REF (referencia a secao na documentacao)

### D2. Cabecalhos de PARTE (Obrigatorio — todas as PARTEs)

Cada PARTE DEVE ter NO MINIMO 4 linhas de comentario contextual ANTES
do codigo executavel. O cabecalho descreve: proposito, contexto fisico/
tecnico, relacao com padroes (P##), e referencias cruzadas.

```python
# ══════════════════════════════════════════════════════════════════════════════
# PARTE {N}: {TITULO EM MAIUSCULAS}
# ══════════════════════════════════════════════════════════════════════════════
# {Linha 1: Descricao do proposito desta PARTE}
# {Linha 2: Contexto tecnico/fisico relevante}
# {Linha 3: Relacao com padroes P## ou versoes}
# {Linha 4: Referencia cruzada a outras celulas ou secoes da DOC}
# ──────────────────────────────────────────────────────────────────────────────
```

Exemplo CORRETO (C4, Parte 3):
```python
# ══════════════════════════════════════════════════════════════════════════════
# PARTE 3: INFERENCE_MODE — SWITCH MESTRE (v5.0.7)
# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE_MODE e o switch mestre que controla o dual-mode do pipeline:
#
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │  "offline"  ->  Inversao 1D padrao (acausal, batch, sem incerteza)  │
#   │  "realtime" ->  Geosteering (causal, sliding window, incerteza)     │
#   └──────────────────────────────────────────────────────────────────────┘
# ──────────────────────────────────────────────────────────────────────────────
```

### D3. Diagramas ASCII (Obrigatorio em condicoes especificas)

Diagramas ASCII com bordas Unicode (caracteres ┌ ─ ┬ ┐ │ ├ ┼ ┤ └ ┴ ┘)
sao OBRIGATORIOS nas seguintes situacoes:

1. **Fluxos com >= 3 caminhos** (ex: offline vs realtime, file_scheme 1 vs 2)
2. **Mapeamentos semanticos** (ex: cores ANSI, formatos de dados, categorias)
3. **Cascatas de auto-configuracao** (ex: INFERENCE_MODE -> FLAGS derivadas)
4. **Catalogos de componentes** (ex: pacotes pip, arquiteturas, losses)
5. **Formulas fisicas** (ex: decoupling ACp/ACx, skin depth)
6. **Relacoes entre celulas** (ex: dependencias, fluxo de dados)

Formatos aceitos:
```
# Tabela:
#   ┌───────────┬──────────┬─────────┐
#   │ Coluna 1  │ Coluna 2 │ Coluna 3│
#   ├───────────┼──────────┼─────────┤
#   │ Valor A   │ Valor B  │ Valor C │
#   └───────────┴──────────┴─────────┘

# Caixa de fluxo:
#   ┌──────────────────────────────────────────┐
#   │  Descricao do fluxo ou estado            │
#   │    -> Consequencia 1                     │
#   │    -> Consequencia 2                     │
#   └──────────────────────────────────────────┘

# Diagrama de catalogo:
#   ┌──────────────────────────────────────────┐
#   │  GRUPO A                                 │
#   │  ┌────────┬────────┬────────┐            │
#   │  │ Item 1 │ Item 2 │ Item 3 │            │
#   │  └────────┴────────┴────────┘            │
#   ├──────────────────────────────────────────┤
#   │  GRUPO B                                 │
#   │  ┌────────┬────────┐                     │
#   │  │ Item 4 │ Item 5 │                     │
#   │  └────────┴────────┘                     │
#   └──────────────────────────────────────────┘
```

### D4. Declaracao de FLAGS — Bloco Completo de 6 Linhas (Obrigatorio)

TODA FLAG DEVE ter o bloco completo de 6 linhas. Nao condensar para 2-3 linhas.
Mesmo FLAGS simples (bool on/off) DEVEM usar o formato completo.

```python
# -- FLAG_NAME (type) -----------------------------------------------
# Descricao: [O que esta flag faz — frase completa, nao abreviada]
# Valores validos: [Lista EXPLICITA de todos os valores aceitos]
# Default: [Valor padrao com justificativa tecnica]
# Impacto: [Quais celulas usam e COMO — nao apenas numeros]
# Referenciado em: [Lista de celulas com contexto]
# Nota de versao: [Versao onde foi introduzida ou alterada]
FLAG_NAME: type = default_value
```

Exemplo CORRETO:
```python
# -- USE_CAUSAL_MODE (bool) -------------------------------------------------
# Descricao: Forca modo causal em todas as arquiteturas.
# Valores validos: True, False.
# Default: False (auto-True quando INFERENCE_MODE="realtime").
# Impacto: C27-C39 (padding causal), C40 (treinamento).
# Referenciado em: C4 (auto-config), C7 (arquitetura), C27-C39.
# Nota de versao: v5.0.7.
USE_CAUSAL_MODE: bool = False
```

Exemplo INCORRETO (condensado — NAO ACEITO):
```python
# -- USE_CAUSAL_MODE (bool)
# Forca modo causal. Default: False. Impacto: C27-C39.
USE_CAUSAL_MODE: bool = False
```

### D5. Docstrings de Funcoes (Obrigatorio — Google-style)

Toda funcao publica DEVE ter docstring completa no estilo Google com
TODOS os 5 campos: descricao, Args, Returns, Example, Note.
Funcoes privadas (_prefixo) podem omitir Example.

```python
def nome_funcao(
    param1: tipo1,
    param2: tipo2 = default,
    log: Optional[logging.Logger] = None,
) -> TipoRetorno:
    """Descricao concisa da funcao em uma linha.

    Descricao detalhada em um ou mais paragrafos. Explicar o contexto,
    algoritmo ou logica. Mencionar padroes relevantes (P##).

    Args:
        param1: Descricao do parametro 1 com tipo e restricoes.
        param2: Descricao do parametro 2. Default: {default}.
                Restricoes adicionais se houver.
        log:    Logger para saida. Default: None (usa logger global).

    Returns:
        TipoRetorno: Descricao do valor retornado com formato esperado.
            Se retorna Dict, listar chaves principais.

    Example:
        >>> resultado = nome_funcao(valor1, valor2)
        >>> print(resultado)

    Note:
        Referenciado em: C{X} ({contexto}), C{Y} ({contexto}).
        Assinatura critica: {se aplicavel, ex: "SEMPRE 2 args"}.
    """
```

### D6. Docstrings de Classes (Obrigatorio)

Toda classe DEVE ter docstring com campo Attributes listando TODOS os
atributos publicos. Se a classe tiver metodos publicos, cada metodo
DEVE ter sua propria docstring (ver D5).

```python
class NomeDaClasse:
    """Descricao da classe em uma linha.

    Descricao detalhada com contexto e proposito.

    Attributes:
        attr1: Descricao do atributo 1 — uso semantico.
        attr2: Descricao do atributo 2 — uso semantico.
        attr3: Descricao do atributo 3 — uso semantico.

    Example:
        >>> obj = NomeDaClasse()
        >>> print(obj.attr1)

    Note:
        Referenciado em: C{X}-C{Y} ({contexto}).
    """
```

### D7. Comentarios Inline Semanticos (Recomendado)

Usar separadores semanticos com `# ── nome ──` para agrupar blocos
logicos DENTRO de uma PARTE quando houver sub-agrupamentos.

```python
# ── Estilos de texto ─────────────────────────────────────────────────
BOLD = "\033[1m"
DIM  = "\033[2m"

# ── Cores padrao (foreground) ────────────────────────────────────────
RED   = "\033[31m"
GREEN = "\033[32m"

# ── Alias para compatibilidade ─────────────────────────────────────
set_seeds = set_all_seeds
```

### D8. Inventario de Exports (Obrigatorio — todas as celulas)

O inventario DEVE ser um dicionario completo com tipo e valor de cada
export. Se VERBOSE=True, listar todos os itens no log.

```python
_c{N}_exports: dict = {
    # {Grupo semantico 1}
    "EXPORT_1": f"tipo — {valor}",
    "EXPORT_2": f"tipo — {valor}",
    # {Grupo semantico 2}
    "EXPORT_3": f"tipo — {valor}",
}

logger.info(f"  {C.DIM}Total de exports: {len(_c{N}_exports)}{C.RESET}")

if VERBOSE:
    for _name, _desc in _c{N}_exports.items():
        logger.info(f"  {C.BRIGHT_CYAN}{_name:<38}{C.RESET} {C.DIM}{_desc}{C.RESET}")
```

### D9. Resumo Final da Celula (Obrigatorio)

Toda celula DEVE terminar com bloco de resumo com metricas-chave
da execucao (tempo, contagem de exports, FLAGS principais, etc.).

```python
_t1_c{N} = _time.perf_counter()
_elapsed_c{N} = _t1_c{N} - _t0_c{N}

logger.info(f"\n{'═' * 60}")
logger.info(f"{C.BOLD}{C.GREEN}✓ CELULA C{N} EXECUTADA COM SUCESSO{C.RESET}")
logger.info(f"{'═' * 60}")
logger.info(f"  Tempo ...................... {C.BRIGHT_WHITE}{format_time(_elapsed_c{N})}{C.RESET}")
logger.info(f"  {Metrica 1} ............... {C.BRIGHT_WHITE}{valor1}{C.RESET}")
logger.info(f"  {Metrica 2} ............... {C.BRIGHT_WHITE}{valor2}{C.RESET}")
logger.info(f"{'═' * 60}")
logger.info(f"{C.DIM}Proxima: C{N+1} — {Nome da proxima celula}{C.RESET}\n")
```

### D10. Banner de PARTE com logger (Obrigatorio — todas as PARTEs)

Cada PARTE executavel DEVE iniciar com banner logger padronizado:

```python
logger.info(f"\n{C.BOLD}{C.BLUE}▶ Parte {N}: {Titulo da Parte}{C.RESET}")
logger.info(f"{'─' * 50}")
```

### Resumo dos Padroes de Documentacao

| Padrao | Elemento | Requisito Minimo |
|:------:|:---------|:----------------|
| D1 | Mega-Header | 12+ campos, bordas Unicode ╔═╗ |
| D2 | Cabecalho PARTE | >= 4 linhas contexto + separador ── |
| D3 | Diagramas ASCII | Obrigatorio se >= 3 caminhos/categorias |
| D4 | FLAGS | Bloco COMPLETO de 6 linhas (NUNCA condensar) |
| D5 | Docstring funcao | Google-style: Args/Returns/Example/Note |
| D6 | Docstring classe | Attributes listando TODOS atributos |
| D7 | Comentarios inline | Separadores # ── nome ── para sub-grupos |
| D8 | Inventario exports | Dict completo com tipo+valor, log VERBOSE |
| D9 | Resumo final | Tempo + metricas + proxima celula |
| D10 | Banner PARTE | logger ▶ + separador ─ |

---

## Mapa de Diretorios do Projeto

```
/Users/daniel/Geosteering_AI/
|-- CLAUDE.md                              <- Este arquivo (instrucoes persistentes)
|-- Arquivos_Projeto_Claude/               <- Codigo principal e documentacao
|   |-- Skill/                             <- Skill config e referencias
|   |   |-- SKILL.md                       <- Regras de geracao de codigo (v5.0.15)
|   |   |-- ERRATA_E_VALORES.md            <- Constantes fisicas criticas
|   |   |-- CHECKLIST.md                   <- 71 itens de validacao
|   |   |-- ARQUITETURAS.md                <- 44 modelos
|   |   |-- PERSPECTIVAS.md                <- P2-P5 FLAGS e interacoes
|   |   |-- LOSSES_CATALOG.md              <- 25 losses (4 categorias) **NOVO v5.0.15**
|   |   |-- NOISE_CATALOG.md               <- 43 tipos de ruido (5 familias) **NOVO v5.0.15**
|   |   |-- SECTION3_GUIDE.md              <- Guia C27-C39 arquiteturas **NOVO v5.0.15**
|   |   |-- PIPELINE_EXECUTION_FLOWS.md    <- 12 cenarios dados + P1-P8 **NOVO v5.0.15+**
|   |-- ANALISE_COMPARATIVA_PINN_MORALES_2025.md <- Analise PINN vs Morales **NOVO**
|   |   |-- geosteering-v5014.skill        <- Pacote da Skill v5.0.14 (legado)
|   |   +-- geosteering-v5015.skill        <- Pacote da Skill v5.0.15 (12 arquivos, 643 KB)
|   |-- C0_Logger_Seeds_Utilitarios.py     <- Celula C0 (v5.0.15 EXPANDIDA)
|   |-- C1_Instalacao_Dependencias.py      <- Celula C1 (v5.0.15 RECONSTRUIDA)
|   |-- C2_Imports_Verificacao.py          <- Celula C2 (v5.0.15 RECONSTRUIDA)
|   |-- C3_Paths_Diretorios_File_Scheme.py <- Celula C3 (v5.0.15 RECONSTRUIDA)
|   |-- C4_FLAGS_Gerais_InferenceMode.py   <- Celula C4 (v5.0.15 RECONSTRUIDA)
|   |-- C5_FLAGS_Dados_*.py                <- Celula C5 (v5.0.15 RECONSTRUIDA)
|   |-- C6_FLAGS_Preprocessamento_*.py     <- Celula C6 (v5.0.15 RECONSTRUIDA)
|   |-- C7_FLAGS_Arquitetura_*.py          <- Celula C7 (v5.0.15 RECONSTRUIDA)
|   |-- DOCUMENTACAO_COMPLETA_SOFTWARE_v5_0_14.md  <- Referencia tecnica principal
|   |-- MANUAL_IMPLEMENTACAO_COMPLETO_v5014.md     <- Manual de impl. C0-C73 (1.756 linhas) **NOVO**
|   |-- MANUAL_IMPLEMENTACAO_ESQUELETO_v5014.md    <- Versao esqueleto do manual **NOVO**
|   |-- ONBOARDING_PIPELINE_v5014.md               <- Guia de integracao **NOVO**
|   |-- ANALISE_ESTRUTURA_SKILL_v5014.md           <- Analise da estrutura da Skill **NOVO**
|   |-- AddNoise_OnTheFly2.py              <- Sistema de ruido on-the-fly
|   |-- novas_loss.py                      <- 25 loss functions (v5.0.15)
|   +-- assemble_notebook.py               <- Script de montagem do .ipynb
|-- Fortran_Gerador/                       <- Gerador Fortran (simulador EM)
|   |-- PerfilaAnisoOmp.f08               <- Codigo principal do simulador
|   |-- *.dat                              <- Dados binarios (22-col, MODEL-MAJOR)
|   +-- *.out                              <- Metadata (theta, freq, nmeds, nm)
|-- Py_Geosteering/                        <- Codigo Python legado
|   |-- Codigos/                           <- Scripts de treinamento
|   +-- Referencias/                       <- Livros e artigos PDF
|-- PDFs/                                  <- Literatura de referencia
|-- Markdowns/                             <- Documentacao historica (v5.0-v5.0.6)
+-- Plotagens/                             <- Saida de visualizacoes
```

---

## Referencias Bibliograficas (PDFs Disponiveis)

### Em Arquivos_Projeto_Claude/ (colecao principal consolidada)

#### Livros
- `Darwin V Ellis...Well Logging for Earth Scientists 2008 Springer` (22 MB)
- `Siddharth Misra, Hao Li, Jiabo He - Machine Learning for Subsurface Characterization (2019)` (86 MB) **NOVO**
- `Louis Owen - Hyperparameter Tuning with Python (2022, Packt)` (11 MB) **NOVO**
- `Manu Joseph - Modern Time Series Forecasting with Python (2022, Packt)` (26 MB)
- `Milan Curcic - Modern Fortran (2020, Manning)` (12 MB)
- `[Livro] Dive Into Deep Learning` (12 MB)
- `Principles and Applications of Well Logging` (18 MB)

#### Artigos e Manuais Tecnicos
- `Schlumberger_GeoSphere HD 1.0.pdf` (2 MB) — especificacoes da ferramenta comercial
- `Wang_2018_J._Geophys._Eng._15_2339.pdf` (3.8 MB) — inversao EM com DL
- `GeoSphereXTatu.pdf` (72 KB) — GeoSphere Tatu comparison
- `Noh_Verdin_2022_Petrophysics.pdf` (3.3 MB) — Noh & Verdin 2022 **NOVO**
- `Anisotropic resistivity estimation...borehole triaxial EM.pdf` (13 MB) **NOVO**
- `ggad217.pdf` — Electrical properties + ML + Neural networks + Inverse theory **NOVO**
- `gxag017.pdf` — artigo cientifico (geofisica) **NOVO**
- `2111.07490v1.pdf` — arxiv preprint (10 paginas) **NOVO**
- `2170-5584-2-PB.pdf` — artigo cientifico **NOVO**
- `Manuscript_v8.pdf` — manuscrito (5 paginas) **NOVO**
- `Key_Specs_Survey_UH.pdf` — especificacoes tecnicas **NOVO**

### Em PDFs/ (literatura adicional)
- Constable et al. 2016 — Petrophysics

### Em Py_Geosteering/Referencias/
- Dive Into Deep Learning, Modern Fortran, Modern Time Series (copias)

---

## Principios de Engenharia

### Logica Simples, Documentacao Completa, Impacto Minimo

- **Logica simples:** Simplificar a logica de cada alteracao. Evitar over-engineering.
  Documentacao DEVE ser rica (D1-D10), mas a logica deve ser a mais clara possivel.
- **Sem preguica:** Encontrar causas raiz. Sem solucoes temporarias. Padrao de engenheiro senior.
- **Impacto minimo:** Alterar apenas o necessario. Nao introduzir efeitos colaterais ou regressoes.
- **Elegancia equilibrada:** Para alteracoes NAO triviais, perguntar "existe uma forma mais correta
  e completa?" — mas NAO complicar correcoes simples e obvias.
- **Correcao autonoma:** Ao detectar bug durante implementacao, corrigir imediatamente e reportar.
  Apontar para logs, erros e validacoes — e resolver sem necessitar troca de contexto do usuario.

### Orquestracao do Fluxo de Trabalho

- **Planejamento obrigatorio:** Usar modo de planejamento para QUALQUER tarefa nao trivial
  (3+ etapas, decisoes arquiteturais, ou propagacao cross-celula).
- **Pare e replaneje:** Se algo der errado durante implementacao, PARAR imediatamente e replanejar.
  NAO continuar cascateando erros.
- **Subagentes:** Usar subagentes para pesquisa, exploracao e analise paralela.
  Manter a janela de contexto principal limpa. Uma tarefa por subagente.
  IMPORTANTE: Subagentes DEVEM receber contexto dos valores criticos e proibicoes.
- **Verificacao antes de conclusao:** NUNCA marcar tarefa como concluida sem provar que funciona
  (py_compile, validate_pipeline.py, ou demonstracao de correção).
  Meta-pergunta: "Um engenheiro senior aprovaria isso?"

### Ciclo de Autoaperfeicoamento

- Apos QUALQUER correcao do usuario: registrar o padrao em MEMORY.md (secao Licoes Aprendidas).
- Escrever regras que evitem o mesmo erro no futuro.
- Revisar licoes no inicio de cada sessao via Protocolo de Retomada.
- Iterar implacavelmente nestas licoes ate que a taxa de erros diminua.

---

## Workflow de Geracao de Codigo (5 Passos)

### Passo 1 — Identificar
Identificar a celula (C0-C73), secao (0-7), nome e proposito.

### Passo 2 — Consultar
1. Ler ERRATA_E_VALORES.md para valores criticos
2. Ler secao relevante em DOCUMENTACAO_COMPLETA_v5_0_14.md
3. Se envolve P2-P5, ler PERSPECTIVAS.md
4. Se envolve arquiteturas, ler ARQUITETURAS.md
5. Se envolve dados/ruido/FV/GS (C19-C26, C40-C47), ler PIPELINE_EXECUTION_FLOWS.md

### Passo 3 — Planejar (se nao trivial)
Para tarefas com 3+ etapas ou decisoes arquiteturais:
1. Usar modo de planejamento antes de gerar codigo
2. Identificar TODAS as celulas impactadas (propagacao cross-celula)
3. Verificar se a alteracao introduz regressoes em celulas existentes

### Passo 4 — Gerar
Gerar codigo seguindo a estrutura canonica (mega-header, PARTEs, timer, validacao, exports, cleanup).

### Passo 5 — Verificar e Provar
Executar CHECKLIST.md (50 itens). Verificar em particular:
- Shapes: (batch, N_MEDIDAS, N_FEATURES) -> (batch, N_MEDIDAS, OUTPUT_CHANNELS)
- Errata: FREQUENCY_HZ=20000.0, SEQUENCE_LENGTH=600
- Causalidade: se realtime, padding='causal'
- Nenhum import de PyTorch
- py_compile OK em TODOS os arquivos alterados
- validate_pipeline.py sem regressoes (0 CRITICAL, 0 HIGH)
- NUNCA declarar sucesso sem demonstrar que compila e passa validacao

---

## Checklist Resumido (Itens Mais Criticos)

- [ ] Sem imports de PyTorch
- [ ] FREQUENCY_HZ == 20000.0
- [ ] SPACING_METERS == 1.0
- [ ] SEQUENCE_LENGTH == 600
- [ ] TARGET_SCALING == "log10"
- [ ] Shapes consistentes (batch, None, N_FEATURES) -> (batch, N_MEDIDAS, OUTPUT_CHANNELS)
- [ ] N_MEDIDAS do .out (nunca hardcoded)
- [ ] Caminhos via variaveis de C3
- [ ] Scaler ajustado em dados LIMPOS
- [ ] plt.close() apos cada visualizacao
- [ ] Seeds fixadas para reprodutibilidade
- [ ] Type hints em todas as assinaturas
- [ ] Docstrings em funcoes publicas
- [ ] f-strings (nao % ou .format())
- [ ] Compatibilidade dual-mode (offline + realtime)
- [ ] segregate_by_angle usa out_metadata= (nao metadata=)
- [ ] Layout MODEL-MAJOR (nao ANGLE-MAJOR)
- [ ] Formato 22-col ATIVO, 12-col LEGADO
- [ ] INPUT_FEATURES 22-col == [1, 4, 5, 20, 21] (NUNCA [0, 3, 4, 7, 8])
- [ ] OUTPUT_TARGETS 22-col == [2, 3] (NUNCA [1, 2])
- [ ] INPUT_FEATURES e OUTPUT_TARGETS sem overlap (features ∩ targets = ∅)
- [ ] [P1] Split por modelo geologico (NUNCA por amostra) — train∩val∩test = ∅ em model_ids
- [ ] [P2] Se noise on-the-fly: val_clean_ds E val_noisy_ds ambos presentes
- [ ] [P3] Se USE_GEOSIGNAL_FEATURES: per-group scalers (StandardScaler EM + RobustScaler GS)
- [ ] [P3] Scaler GS fit em geosinais LIMPOS (nao ruidosos)
- [ ] [P4] Se off-line: shuffle_buffer >= K_COPIES * n_train
- [ ] [P5] Se off-line + curriculum: usar tier-switching ou recomendar on-the-fly
- [ ] [P6] InferencePipeline exportado com FV+GS+scalers serializados
- [ ] [P7] eps_tf = 1e-12 (nao 1e-30) em operacoes TF float32
- [ ] [P7] tf.where guards em divisoes complexas de geosinais TF
- [ ] [P7] tf.clip_by_value atenuacao [-100,100] dB e fase [-180,180] graus
- [ ] [P8] Cenarios clean (1x) documentados como BASELINE (nao producao)
- [ ] DATA_SCENARIO padrao = "2D" (maxima fidelidade fisica)
- [ ] [Morales] Se PINNS_HARD_CONSTRAINT=True: constraint layer aplicada em build_model (C37)
- [ ] [Morales] Se USE_ENSEMBLE_UQ=True: N_ENSEMBLE_UQ > 0 e UQ_ENSEMBLE_KAPPA ∈ (0, 1)
- [ ] [Morales] Se USE_MORALES_HYBRID_LOSS=True: MORALES_PHYSICS_OMEGA ∈ (0, 1)
- [ ] [Morales] Se USE_FIELD_DATA=True: FIELD_DATA_PATH não-vazio
