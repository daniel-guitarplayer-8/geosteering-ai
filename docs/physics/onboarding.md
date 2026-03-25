# ONBOARDING — Pipeline de Inversão Geofísica com Deep Learning

## Guia de Integração para Novos Desenvolvedores — v5.0.15

**Projeto:** Inversion and Geosteering AI  
**Autor:** Daniel Leal  
**Público-alvo:** Desenvolvedor Python/TensorFlow sem background em geofísica  
**Tempo de leitura:** ~2 horas  
**Pré-requisitos:** Python 3.10+, TensorFlow 2.x/Keras, NumPy, Google Colab  
**Referência completa:** DOCUMENTACAO_COMPLETA_SOFTWARE_v5_0_15.md

---

## 1. O Que Este Projeto Faz

### 1.1 Em Uma Frase

Traduzimos medições eletromagnéticas feitas por ferramentas dentro de um poço de petróleo em mapas de resistividade elétrica das rochas ao redor, usando redes neurais profundas.

### 1.2 Analogia

Pense em uma tomografia médica: raios-X atravessam o corpo e um computador reconstrói a imagem interna. Nosso projeto faz algo análogo:

- Em vez de raios-X → **ondas eletromagnéticas** (20 kHz)
- Em vez do corpo humano → **rochas no subsolo** (folhelhos, arenitos, carbonatos)
- Em vez de um tomógrafo → **ferramenta LWD** (Logging While Drilling) dentro do poço
- Em vez de reconstrução analítica → **rede neural profunda** (deep learning)
- A "imagem" resultante → **perfil de resistividade** (ρh horizontal, ρv vertical)

### 1.3 Por Que Importa

Saber a resistividade das rochas em tempo real permite **geosteering** — ajustar a trajetória de perfuração para manter a broca no reservatório de petróleo. Ferramentas comerciais (GeoSphere HD da Schlumberger) fazem isso com inversão física tradicional (Monte Carlo, Gauss-Newton). Nosso projeto usa deep learning: milissegundos vs minutos.

### 1.4 O Fluxo Básico

```
   ENTRADA                    REDE NEURAL                     SAÍDA
┌───────────────┐         ┌──────────────┐            ┌──────────────────┐
│ Sinais EM     │         │              │            │ Perfil de ρh     │
│ medidos no    │   →     │  ResNet_18   │     →      │ Perfil de ρv     │
│ poço (tensor  │         │  (ou outras  │            │ (ponto a ponto)  │
│ magnético H)  │         │  25 opções)  │            │                  │
└───────────────┘         └──────────────┘            └──────────────────┘
  (batch, 600, 5)                                       (batch, 600, 2)
```

---

## 2. A Física em 5 Minutos

### 2.1 Resistividade

Resistividade (ρ, Ω·m) mede quanto uma rocha resiste à corrente elétrica. Óleo/gás são resistivos (50–500 Ω·m), água salgada é condutiva (1–10 Ω·m). Diferenciar os dois é o objetivo principal.

### 2.2 Anisotropia TIV

Rochas sedimentares conduzem melhor na horizontal (ρh) que na vertical (ρv) — como madeira conduz melhor ao longo das fibras. Nosso pipeline SEMPRE estima ambos: output = (batch, N_MEDIDAS, **2**) = [ρh, ρv].

### 2.3 Ferramenta LWD

A ferramenta LWD é um conjunto de antenas EM na coluna de perfuração. Transmissor emite onda, receptores medem. Antenas tilted → medição direcional → **tensor magnético 3×3** (9 componentes complexas). Usamos 22 colunas no simulador; configuração padrão: 5 features (z + Re/Im de Hxx e Hzz).

### 2.4 Skin Depth

A onda EM penetra a uma profundidade δ = √(2/ωμσ). Frequências baixas penetram mais (maior DOI), frequências altas dão melhor resolução vertical. Nosso simulador: f=20 kHz.

### 2.5 Geosinais

Razões gain-compensated entre componentes do tensor: **USD** (detecção de boundaries), **UHR** (resistividade bulk), **UHA** (anisotropia), **UAD** (dip), **U3DF** (indicador 3D). Cada um produz atenuação (dB) e phase shift (°).

### 2.6 Picasso Plots

Mapa de capacidade de detecção (DOD) em função das resistividades de 2 camadas. A diagonal Rt1=Rt2 é "zona cega". DOD do nosso simulador: ~1.5–3.0 m.

---

## 3. Arquitetura do Pipeline

### 3.1 Visão Geral: 74 Células em 8 Seções

```
SEÇÃO 0: INFRAESTRUTURA         C0–C2     (3 células)   → Logger, seeds, imports
SEÇÃO 1: CONFIGURAÇÃO/FLAGS     C3–C18    (16 células)  → ~1.159 FLAGS
SEÇÃO 2: PREPARAÇÃO DE DADOS    C19–C26   (8 células)   → Datasets tf.data
SEÇÃO 3: ARQUITETURAS           C27–C39   (13 células)  → Modelo Keras
SEÇÃO 4: TREINAMENTO            C40–C47   (8 células)   → Modelo treinado
SEÇÃO 5: AVALIAÇÃO              C48–C57   (10 células)  → Métricas
SEÇÃO 6: VISUALIZAÇÃO           C58–C65   (8 células)   → Plots/relatórios
SEÇÃO 7: GEOSTEERING            C66–C73   (8 células)   → Inferência realtime
```

### 3.2 FLAGS

~1.159 variáveis de configuração com defaults seguros. **Todos os defaults = pipeline baseline** (ResNet_18, identity view, θ=0°, f=20kHz, log_scale_aware_loss). Qualquer variação é opt-in.

### 3.3 Dual-Mode: Offline vs Realtime

Controlado por `INFERENCE_MODE`: "offline" (pesquisa, acausal, 26 arquiteturas) ou "realtime" (geosteering, causal, 20 arquiteturas, sliding window, incerteza).

### 3.4 As 5 Perspectivas

| P# | Nome | Adiciona | Versão |
|:--:|:-----|:---------|:------:|
| P1 | Baseline | z + 4 EM (5 features) | v5.0.1 |
| P2 | θ como feature | Ângulo de inclinação | v5.0.8 |
| P3 | f como feature | Frequência | v5.0.12 |
| P4 | Geosinais | USD, UHR como features | v5.0.13 |
| P5 | Picasso/DTB | Validação + DTB como target | v5.0.15 |

---

## 4. Conceitos-Chave

### 4.1 Formato dos Dados

**`.dat`** = medições EM (binário stream, 22 colunas, layout MODEL-MAJOR)
**`.out`** = metadados (ângulos, frequências, N_MEDIDAS por ângulo)

**MODEL-MAJOR:** todos os dados do modelo 1 primeiro, depois modelo 2, etc.
**Fórmula:** `total_linhas = nm × Σ(nmeds[k] × nf)`

### 4.2 Shapes — Regra de Ouro

```
ENTRADA:  (batch, N_MEDIDAS, N_FEATURES)   típico: (batch, 600, 5)
SAÍDA:    (batch, N_MEDIDAS, 2)            SEMPRE [ρh, ρv]
```

N_MEDIDAS varia com θ: 600 (0°), 622 (15°), 693 (30°), 1200 (60°).

**Regra de preservação temporal:** toda arquitetura DEVE manter N_MEDIDAS no output = input.

### 4.3 Valores Que NUNCA Devem Estar Errados

| Variável | CORRETO | ~~ERRADO~~ |
|:---------|:-------:|:----------:|
| FREQUENCY_HZ | **20000.0** | ~~2.0~~ |
| SPACING_METERS | **1.0** | ~~1000.0~~ |
| SEQUENCE_LENGTH | **600** | ~~601~~ |
| TARGET_SCALING | **"log10"** | ~~"log"~~ |

---

## 5. Estrutura do Código

### 5.1 Template de Célula

```python
# ╔═══════════════════════════════════════════════════╗
# ║  CÉLULA C{N} — {NOME}                             ║
# ║  Pipeline v5.0.15 | Autor: Daniel Leal             ║
# ╚═══════════════════════════════════════════════════╝

print_header(f"C{N} — {NOME}", width=70)
_t0 = time.time()

# PARTE 1 — ...
# PARTE 2 — ...

tracker = ValidationTracker("C{N}")
tracker.check(condição, "descrição")
tracker.summary()

_cN_exports = [...]
del _temp_vars  # NUNCA deletar funções ou FLAGS
```

### 5.2 NUNCA Fazer

- NUNCA usar PyTorch
- NUNCA hardcodar caminhos (usar variáveis de C3)
- NUNCA hardcodar N_MEDIDAS como 601
- NUNCA usar FREQUENCY_HZ = 2.0
- NUNCA deletar funções ou FLAGS na limpeza
- NUNCA fazer fit do scaler em dados ruidosos

---

## 6. Primeiros Passos

### 6.1 Setup

```python
from google.colab import drive
drive.mount(\'/content/drive\')
import os
os.chdir(\'/content/drive/MyDrive/Inversion_Geosteering_Claude_V3\')
```

### 6.2 Execute C0–C7

Execute sequencialmente. Checkpoint após C7:
```python
assert INFERENCE_MODE == "offline"
assert MODEL_TYPE == "ResNet_18"
assert FREQUENCY_HZ == 20000.0
assert N_FEATURES == 5
```

---

## 7. Onde Encontrar O Quê

| Preciso de... | Veja DOC §... |
|:--------------|:--------------|
| Contrato de dados (.dat/.out) | §6 |
| FLAGS completas | §9 + §32 |
| 26 Arquiteturas | §11 |
| 25 Loss functions (13 genéricas + 4 geofísicas + 8 geosteering) | §19 |
| 39 Tipos de ruído | §20 |
| Multi-ângulo (P2) | §25 |
| Multi-frequência (P3) | §26 |
| Geosinais (P4) | §27 |
| Picasso Plots / DTB (P5) | §28 |
| Geosteering realtime | §15 + §24 |
| Checklist (50 itens) | §36 |
| Qualidade Q1–Q15 | §37 |
| Padrões P01–P51 | SKILL_PIPELINE_v5012.md |
| Errata v4.4.5 | §34 |

### 7.1 Loss Functions — 25 Funções (v5.0.15)

O pipeline oferece **25 loss functions** organizadas em 3 categorias:

| Categoria | Qtd | Exemplos |
|:----------|:---:|:---------|
| **Genéricas** | 13 | MSE, MAE, Huber, log-cosh, quantile, etc. |
| **Geofísicas** | 4 | log_scale_aware, anisotropy_ratio, boundary_weighted, skin_depth_weighted |
| **Geosteering** | 8 | dtb_aware, trajectory_risk + 6 novas (v5.0.15) |

**6 novas losses geosteering (v5.0.15):**

- **DILATE** — alinhamento temporal (shape + temporal) para séries preditas vs reais
- **Encoder-Decoder** — loss latent-space para arquiteturas autoencoder
- **Multi-Task Learned** — ponderação automática de múltiplos objetivos via incerteza homoscedástica
- **Sobolev H1** — penaliza diferenças no gradiente (derivada primeira) do perfil predito
- **Cross-Gradient** — impõe consistência estrutural entre ρh e ρv (gradientes cruzados)
- **Spectral** — compara espectros de frequência (FFT) dos perfis predito vs real

---

## 8. Glossário

| Termo | Definição |
|:------|:----------|
| **BHA** | Bottom Hole Assembly — ferramentas na ponta da coluna |
| **Boundary** | Interface entre camadas rochosas |
| **Causal** | Rede que só usa dados passados (tempo real) |
| **Decoupling** | Remoção do acoplamento direto Tx-Rx |
| **DOD** | Depth of Detection — distância máx. de detecção |
| **DTB** | Distance-to-Boundary — distância ao boundary |
| **Feature View** | Transformação das componentes EM |
| **Geosinal** | Razão gain-compensated do tensor |
| **Geosteering** | Ajuste da trajetória em tempo real |
| **LWD** | Logging While Drilling |
| **MODEL-MAJOR** | Layout .dat: modelo 1 completo, depois modelo 2 |
| **N_MEDIDAS** | Pontos de medição (600 para θ=0°) |
| **Picasso Plot** | Mapa de DOD no espaço (Rt1, Rt2) |
| **ρh, ρv** | Resistividade horizontal e vertical (targets) |
| **Seq2seq** | Mapeamento ponto-a-ponto |
| **Skin depth** | Profundidade de penetração EM |
| **TIV** | Transversalmente Isotrópico Vertical |
| **USD/UHR** | Geosinais para boundaries / resistividade |

---

*Onboarding v5.0.15 — Para detalhes: DOCUMENTACAO_COMPLETA_SOFTWARE_v5_0_15.md*
