---
name: geosteering-v5015
description: |
  Skill de geração de código para o Pipeline de Inversão Geofísica com Deep Learning v5.0.15.
  Use SEMPRE que o usuário pedir para criar, editar, revisar ou depurar qualquer célula (C0–C73),
  ou perguntar sobre FLAGS, arquiteturas, dados, losses, ruído, geosinais, Picasso Plots, DTB,
  Feature Views, Target Scaling, ou qualquer componente de inversão EM para LWD/geosteering.
  Use quando solicitar documentos (.docx, .pptx, .xlsx) sobre o projeto, ou perguntas sobre
  geofísica de poço, tensor magnético, anisotropia TIV, skin depth, decoupling EM.
  Triggers: "célula C", "FLAGS", "pipeline", "inversão", "geosteering", "resistividade",
  "PerfilaAnisoOmp", "multi-ângulo", "multi-frequência", "segregate_by_angle", termos LWD.
  v5.0.15 | TensorFlow/Keras (PyTorch PROIBIDO) | P2(θ) P3(f) P4(geosinais) P5(Picasso/DTB)
  74 células, ~1.159 FLAGS, 39 seções, 26 arquiteturas, 25 losses, 43 tipos de ruído.
---

# Pipeline de Inversão Geofísica com Deep Learning — Skill v5.0.15

## 1. Fontes Autoritativas e Ordem de Consulta

SEMPRE consultar os references desta skill ANTES de gerar código ou responder:

| Prioridade | Arquivo | Quando consultar | Linhas |
|:----------:|:--------|:----------------|:------:|
| 1ª | `references/ERRATA_E_VALORES.md` | Qualquer código com constantes físicas | ~95 |
| 2ª | `references/CHECKLIST.md` | Revisão ou auditoria de célula | ~130 |
| 3ª | `references/ARQUITETURAS.md` | Código de C28–C37, Model Factory | ~75 |
| 4ª | `references/PERSPECTIVAS.md` | FLAGS de P2/P3/P4/P5, interações | ~185 |
| 5ª | `references/LOSSES_CATALOG.md` | Geração de C41, configuração de C10 | ~200 |
| 6ª | `references/SECTION3_GUIDE.md` | Geração de C27–C39 (arquiteturas) | ~180 |
| 7ª | `references/NOISE_CATALOG.md` | Geração de C24/C40–C44 (ruído) | ~150 |
| 8ª | `references/MANUAL_IMPLEMENTACAO.md` | Inputs/outputs/deps de qualquer célula | ~1.760 |
| 9ª | `references/DOC_COMPLETA_v5015.md` | Detalhes de qualquer seção (§1–§39) | ~8.530 |
| 10ª | `references/ONBOARDING.md` | Quando usuário é novo no projeto | ~260 |
| 11ª | `references/novas_loss.py` | Implementações TF das 25 losses | ~720 |

**Regra:** Ao gerar código → consultar ERRATA primeiro, depois MANUAL, depois DOC.
Ao responder perguntas → consultar DOC primeiro. Em caso de dúvida → DOC prevalece.

---

## 2. Identidade do Projeto

| Atributo | Valor |
|:---------|:------|
| Versão | **v5.0.15** |
| Autor | Daniel Leal |
| Framework | TensorFlow 2.x / Keras **EXCLUSIVO** (PyTorch PROIBIDO) |
| Ambiente | Google Colab Pro+ com GPU + Google Drive montado |
| Código | Python 3.10+, variáveis em inglês, comentários/docs em PT-BR |
| Cells | 74 (C0–C73) em 8 seções operacionais |
| FLAGS | ~1.159 com defaults conservadores |
| Arquiteturas | 26 modelos (21 + 5 geosteering) |
| Losses | 25 funções (13 genéricas + 4 geofísicas + 8 geosteering) |
| Perspectivas | P2 (θ), P3 (f), P4 (geosinais), P5 (Picasso/DTB) |

---

## 3. Workflow de Geração de Código (4 Passos)

### Passo 1 — Identificar

Identificar a célula (C0–C73), seção (0–7), nome e propósito.
Consultar `references/MANUAL_IMPLEMENTACAO.md` para inputs/outputs/deps.

### Passo 2 — Consultar

1. Ler `references/ERRATA_E_VALORES.md` para valores críticos
2. Ler seção relevante em `references/DOC_COMPLETA_v5015.md`
3. Se envolve P2–P5, ler `references/PERSPECTIVAS.md`
4. Se envolve arquiteturas (C27–C39), ler `references/SECTION3_GUIDE.md` + `references/ARQUITETURAS.md`
5. Se envolve losses (C10/C41), ler `references/LOSSES_CATALOG.md` + `references/novas_loss.py`
6. Se envolve ruído (C12/C24/C40–C44), ler `references/NOISE_CATALOG.md`

### Passo 3 — Gerar

Gerar código seguindo a estrutura canônica com documentação rica (D1–D10):

```
╔══════════════════════════════════════════════════════════╗
║  SEÇÃO {S} — {NOME_SEÇÃO}                                ║
║  C{N} — {Nome Descritivo}                                 ║
║  Pipeline v5.0.15 | Autor: Daniel Leal                    ║
║  Framework: TF/Keras (PyTorch PROIBIDO)                   ║
║  Ambiente: Google Colab Pro+ com GPU                      ║
║  Seção: {S}/7 | Célula: {N}/73                            ║
║  Perspectivas: P1–P5                                      ║
║  Arquiteturas: 26 modelos                                 ║
║  Propósito: • bullet 1 • bullet 2 • bullet 3+            ║
║  Dependências: C0 ({símbolos}), C{X} ({símbolos})         ║
║  Exports: ~{N} {tipo}                                     ║
║  Próxima: C{N+1} — {Nome}                                 ║
║  DOC REF: §{secção}                                       ║
╚══════════════════════════════════════════════════════════╝

Timer → Dependency Guard → Banner →
PARTEs numeradas (≥4 linhas contexto cada, diagramas ASCII) →
FLAGS (bloco COMPLETO 6 linhas) →
Funções (docstring Args/Returns/Example/Note) →
ValidationTracker → Inventário exports → Limpeza → Timer final
```

**IMPORTANTE:** Antes de gerar, consultar os exemplares canônicos:
- Célula de infraestrutura/funções → C0 como referência
- Célula de FLAGS → C4 como referência
- Célula de instalação/verificação → C1 como referência

### Passo 4 — Verificar

Executar checklist de `references/CHECKLIST.md` (71 itens: 50 base + 15 doc + 6 v5.0.15).
Verificar em particular:

**Código:**
- Shapes: (batch, N_MEDIDAS, N_FEATURES) → (batch, N_MEDIDAS, 2)
- Errata: FREQUENCY_HZ=20000.0, SEQUENCE_LENGTH=600
- Causalidade: se realtime, padding='causal'
- Nenhum import de PyTorch

**Documentação (D1–D10):**
- D1: Mega-Header com 12+ campos e bordas Unicode
- D2: Cada PARTE com ≥4 linhas de contexto
- D3: Diagramas ASCII quando ≥3 caminhos/categorias
- D4: TODA FLAG com bloco COMPLETO de 6 linhas (NUNCA condensar)
- D5: Funções com docstring Args/Returns/Example/Note
- D6: Classes com Attributes listando TODOS atributos
- D8: Inventário exports como Dict agrupado com log VERBOSE

---

## 4. Regras Críticas (Resumo)

### 4.1 Valores Físicos (NUNCA errar)

```python
assert FREQUENCY_HZ == 20000.0,  "Errata v4.4.5: NUNCA 2.0 Hz"
assert SPACING_METERS == 1.0,    "Errata v4.4.5: NUNCA 1000.0 m"
# SEQUENCE_LENGTH = 600 (NUNCA 601)
# TARGET_SCALING = "log10" (NUNCA "log")
```

### 4.2 Shapes Obrigatórios

```python
# Entrada: SEMPRE (batch, None, N_FEATURES) — None para multi-ângulo
# Saída:   SEMPRE (batch, N_MEDIDAS, OUTPUT_CHANNELS)
# OUTPUT_CHANNELS: 2=[ρh,ρv] | 4=[+σ] | 6=[+DTB,ρ_adj]
# Input(shape=(None, N_FEATURES)) — NUNCA shape fixo
```

### 4.3 Formato de Dados

- **22-col = ATIVO** (tensor completo 3×3, binário stream)
- **12-col = LEGADO** (código Fortran comentado)
- Layout = **MODEL-MAJOR** (NÃO ANGLE-MAJOR)
- θ e f **NÃO existem** no .dat de 22-col → obtidos do .out
- `total_linhas = nm × Σ(nmeds[k] × nf)`

### 4.4 Proibições Absolutas

- **PyTorch** — PROIBIDO em qualquer parte
- **Hardcoding** de caminhos — usar variáveis de C3
- **Hardcoding** de N_MEDIDAS — ler do .out
- **fit() do scaler** em dados ruidosos — SEMPRE x_train_clean
- **Remover funções/FLAGS** na limpeza — APENAS temporárias
- **FREQUENCY_HZ = 2.0** — é 20000.0
- **N_MEDIDAS = 601** — é 600 para θ=0°

### 4.5 Assinaturas Críticas

```python
ValidationTracker.check(condition, description)     # 2 args
print_header(title, width)                           # 2 args
segregate_by_angle(data_2d=..., out_metadata=...)    # out_metadata=
```

---

## 5. Quick Reference — Estrutura do Pipeline

### 5.1 Células por Seção

```
SEÇÃO 0: INFRAESTRUTURA         C0–C2     (3)   → Logger, seeds, imports
SEÇÃO 1: CONFIGURAÇÃO/FLAGS     C3–C18    (16)  → ~1.159 FLAGS
SEÇÃO 2: PREPARAÇÃO DE DADOS    C19–C26   (8)   → Datasets tf.data
SEÇÃO 3: ARQUITETURAS           C27–C39   (13)  → Modelo Keras
SEÇÃO 4: TREINAMENTO            C40–C47   (8)   → Modelo treinado
SEÇÃO 5: AVALIAÇÃO              C48–C57   (10)  → Métricas
SEÇÃO 6: VISUALIZAÇÃO           C58–C65   (8)   → Plots/relatórios
SEÇÃO 7: GEOSTEERING            C66–C73   (8)   → Inferência realtime
```

### 5.2 Células Já Implementadas (C0–C26)

**Seção 0 — Infraestrutura (C0–C2):**
- C0: EXPANDIDA (1800 linhas, 46 exports, 17 PARTEs) — **exemplar canônico**
- C1–C2: RECONSTRUÍDAS v5.0.15

**Seção 1 — FLAGS (C3–C18, 16 células, ~310 FLAGS):**
- C3: Paths/diretórios | C4: FLAGS gerais (**exemplar canônico**) | C5: FLAGS dados
- C6: Preprocessamento | C7: Arquitetura | C8: Skip connections | C9: Treinamento
- C10: Loss (~62 FLAGS, 25 losses) | C11: Regularização | C12: Augmentation (~129 FLAGS)
- C13: PINNs | C14: Avaliação | C15: EDA | C16: Catalogação | C17: ARCH_PARAMS | C18: Optuna

**Seção 2 — Dados (C19–C26, 8 células, ~120 exports, 76 validações):**
- C19: Parse .out/.dat, segregate MODEL-MAJOR, tf.data (1712 linhas)
- C20: Geological model parser, DTB labels P5 (978 linhas)
- C21: Decoupling EM ACp/ACx (818 linhas)
- C22: Feature Views, geosinais P4 (1185 linhas)
- C23: Split, target scaling, feature normalization (1052 linhas)
- C24: tf.data.Dataset + Noisy3DDataGenerator (1393 linhas)
- C25: Dataset optimization, validation (1025 linhas)
- C26: EDA plots, Picasso DOD P5 (1451 linhas)

**C27–C73:** A implementar do zero. **Próxima fase: C27–C39 (Arquiteturas).**

### 5.3 FLAGS Principais

| FLAG | Default | Descrição |
|:-----|:--------|:----------|
| INFERENCE_MODE | "offline" | Switch mestre dual-mode |
| MODEL_TYPE | "ResNet_18" | Arquitetura da rede |
| FREQUENCY_HZ | 20000.0 | Frequência do transmissor |
| SPACING_METERS | 1.0 | Espaçamento Tx-Rx |
| TARGET_SCALING | "log10" | Escala dos targets |
| USE_MULTI_ANGLE | False | Perspectiva 2 (θ) |
| USE_FREQ_AS_FEATURE | False | Perspectiva 3 (f) |
| USE_GEOSIGNAL_FEATURES | False | Perspectiva 4 (geosinais) |
| USE_PICASSO_PLOT | False | Perspectiva 5 (Picasso) |
| USE_DTB_AS_TARGET | False | Perspectiva 5 (DTB) |
| FILE_SCHEME | 1 | 1=split interno, 2=dois pares |
| OUTPUT_CHANNELS | 2 | 2/4/6 canais de saída |

### 5.4 Dual-Mode

| Aspecto | Offline (padrão) | Realtime |
|:--------|:-----------------|:---------|
| Dados | Batch completo | Sliding window |
| Rede | Acausal (26 arqs) | Causal (20 arqs) |
| Saída | (batch, N, 2) | (1, W, 2-6) |
| Incerteza | Opcional | Automática |

### 5.5 Losses — Quick Reference (25 funções)

| Cat. | Qtd | Versão | Exemplos | Detalhes |
|:-----|:---:|:------:|:---------|:---------|
| Genéricas | 13 | v5.0.1+ | mse, mae, huber, log_cosh | Keras built-in |
| Geofísicas | 4 | v5.0.3+ | log_scale_aware + 3 variantes | Gangorra β |
| Geosteering | 2 | v5.0.7 | probabilistic_nll, look_ahead | Incerteza/realtime |
| Geo. Avançado | 6 | v5.0.15 | dilate, sobolev, spectral, cross_gradient, multitask, encoder_decoder | Opt-in |

Para detalhes: `references/LOSSES_CATALOG.md`. Implementações: `references/novas_loss.py`.

### 5.6 Próxima Fase — C27–C39 (Arquiteturas)

```
C27: Blocos básicos (ConvBlock1D, ResidualBlock, SEBlock, S4Layer, ...)
C28: ResNet_18 / ResNet_34
C29: CNN_1D
C30: TCN / TCN_Advanced
C31: LSTM / BiLSTM
C32: CNN_LSTM / CNN_BiLSTM_ED
C33: U-Nets (4 variantes)
C34: Transformer / TFT
C35: N-BEATS / N-HiTS
C36: FNO / DeepONet / GeoAttn + Geosteering #22–#26
C37: Model Factory (build_model dispatch)
C38: Configuração Optuna
C39: Função Objetivo Optuna
```

Para detalhes: `references/SECTION3_GUIDE.md`. Hiperparâmetros: C17 (ARCH_PARAMS).

---

## 6. Padrões de Código e Documentação (51 Padrões + 10 Documentação)

### Referência: CLAUDE.md §"Padrão de Documentação Detalhada"

Os padrões de documentação D1–D10 estão definidos em detalhe no CLAUDE.md.
Os arquivos `C0_Logger_Seeds_Utilitarios.py` e `C4_FLAGS_Gerais_InferenceMode.py`
são os exemplares canônicos. Ao gerar qualquer célula, CONSULTAR esses exemplares.

### Estruturais (P01–P10)

- P01: Mega-Header Unicode (╔══╗) com **12+ campos** (ver D1 no CLAUDE.md):
  Seção, Célula, Pipeline, Autor, Framework, Ambiente, Seção/Célula,
  Perspectivas, Arquiteturas, Propósito (3+ bullets), Dependências
  (lista EXPLÍCITA de símbolos), Exports, Próxima célula, DOC REF
- P02: Banner print_header() + 2 linhas DIM (seção + descrição)
- P03: Timer _t0/_t1 com _elapsed_c{N} + format_time()
- P04: PARTEs numeradas com separadores ═ e **≥4 linhas contexto** (ver D2)
- P05: Diagramas ASCII inline com bordas Unicode ┌─┬─┐ (ver D3):
  - OBRIGATÓRIO quando ≥3 caminhos/categorias
  - OBRIGATÓRIO para cascatas de auto-configuração
  - OBRIGATÓRIO para catálogos de componentes
  - OBRIGATÓRIO para fórmulas físicas
- P06: FLAGS em SCREAMING_SNAKE_CASE com **bloco COMPLETO de 6 linhas** (ver D4):
  Descrição, Valores válidos, Default, Impacto, Referenciado em, Nota de versão
  — NUNCA condensar para 2-3 linhas
- P07: FLAGS existem com default ANTES de serem usadas
- P08: Código para Google Colab, células independentes
- P09: Caminhos via variáveis de C3
- P10: Seeds fixadas em C0

### Implementação (P11–P17)

- P11: Type hints em 100% das assinaturas
- P12: Docstrings Google-style **completas** em funções públicas (ver D5):
  - OBRIGATÓRIO: descrição + Args + Returns + Example + Note
  - Args: todos os parâmetros com tipo, restrições e default
  - Returns: tipo com formato esperado; se Dict, listar chaves
  - Example: uso básico com >>> (pode omitir em funções privadas)
  - Note: referências cruzadas a células e assinaturas críticas
- P13: try/except em toda operação de I/O
- P14: Validação com contadores [VN] via ValidationTracker
- P15: Cores ANSI semânticas (12 tipos — ver tabela em C0 PARTE 2)
- P16: Log alinhado com pontos (`.` padding até coluna fixa)
- P17: Limpeza de temporárias (lista EXPLÍCITA — nunca del por padrão)

### Runtime (P18–P22)

- P18: plt.close() após cada visualização
- P19: f-strings para formatação (nunca % ou .format())
- P20: .keras (modelos), .joblib (scalers), .json (configs)
- P21: tf.data.Dataset com .batch().prefetch(AUTOTUNE).cache()
- P22: Mixed precision quando GPU suportar

### Qualidade (Q1–Q15)

- Q1: Argumentos nomeados ↔ parâmetros da assinatura
- Q5: Valores constantes em loops → pré-computar em cache
- Q8: Acumulação em lista + stack final
- Q11: Lógica train/holdout fatorada
- Q15: Diagramas ASCII para fluxos com mais de 2 caminhos

### Documentação Detalhada (D1–D10) — Resumo

| Padrão | Elemento | Requisito |
|:------:|:---------|:----------|
| D1 | Mega-Header | 12+ campos, bordas ╔═╗, propósito 3+ bullets |
| D2 | Cabeçalho PARTE | ≥4 linhas contexto, separador ══ e ── |
| D3 | Diagramas ASCII | Obrigatório se ≥3 caminhos/categorias |
| D4 | FLAGS | Bloco COMPLETO 6 linhas (NUNCA condensar) |
| D5 | Docstring função | Google-style: Args/Returns/Example/Note |
| D6 | Docstring classe | Attributes com TODOS atributos públicos |
| D7 | Comentários inline | Separadores `# ── nome ──` para sub-grupos |
| D8 | Inventário exports | Dict com tipo+valor, agrupado, log VERBOSE |
| D9 | Resumo final | Tempo + métricas + próxima célula |
| D10 | Banner PARTE | `logger ▶` + separador `─ × 50` |

### Exemplares Canônicos (usar como referência ao gerar código)

Ao gerar qualquer célula, SEMPRE usar estes arquivos como referência de
padrão de documentação:

| Arquivo | Tipo | Padrões demonstrados |
|:--------|:-----|:--------------------|
| `C0_Logger_Seeds_Utilitarios.py` | Infraestrutura (classes + funções) | D1, D2, D3, D5, D6, D7, D8, D9, D10 |
| `C4_FLAGS_Gerais_InferenceMode.py` | FLAGS (configuração) | D1, D2, D3, D4, D8, D9, D10 |
| `C1_Instalacao_Dependencias.py` | Instalação (pip + verificação) | D1, D2, D3, D8, D9, D10 |

**Regra:** Se em dúvida sobre nível de documentação, consultar o exemplar
canônico correspondente ao tipo de célula sendo gerada.

(Detalhes completos dos padrões D1–D10: CLAUDE.md §"Padrão de Documentação Detalhada")

---

## 7. Quando Consultar Cada Reference

| Situação | Reference a consultar |
|:---------|:---------------------|
| Gerar qualquer célula | ERRATA → MANUAL → DOC §correspondente |
| Auditar/revisar célula | CHECKLIST + ERRATA |
| Gerar C27–C39 (arquiteturas) | SECTION3_GUIDE + ARQUITETURAS + DOC §11 |
| Gerar C40–C41 (losses) | LOSSES_CATALOG + novas_loss.py + DOC §19 |
| Gerar C24/C40–C44 (ruído) | NOISE_CATALOG + DOC §20 |
| Código com P2/P3/P4/P5 | PERSPECTIVAS + DOC §25–§28 |
| Perguntas técnicas | DOC §correspondente |
| Usuário novo no projeto | ONBOARDING |
| Detalhes de FLAGS | DOC §9 + §32 |
| Detalhes de losses | LOSSES_CATALOG → DOC §19 |
| Detalhes de ruído | NOISE_CATALOG → DOC §20 |
| Geosteering realtime | DOC §15 + §24 |
| Formato .dat/.out | DOC §6 |

---

## 8. Skills Complementares de Formato

Para deliverables em formatos específicos, consultar TAMBÉM:
- `.docx` → `/mnt/skills/public/docx/SKILL.md`
- `.pptx` → `/mnt/skills/public/pptx/SKILL.md`
- `.xlsx` → `/mnt/skills/public/xlsx/SKILL.md`
- `.pdf` → `/mnt/skills/public/pdf/SKILL.md`
- React/HTML → `/mnt/skills/public/frontend-design/SKILL.md`

---

*Skill Geosteering v5.0.15 — Pipeline de Inversão Geofísica com Deep Learning*
