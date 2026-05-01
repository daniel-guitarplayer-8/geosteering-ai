---
name: geosteering-physics
description: |
  Sub-skill de domínio físico do Geosteering AI v2.0. Cobre TODA a física de geofísica /
  petrofísica / eletromagnetismo para LWD e geosteering: tensor EM 3×3, resistividade TIV,
  skin depth, decoupling, Feature Views (6), Geosinais P4 (5 famílias), Picasso/DTB, Archie.
  Use para qualquer questão de física, fórmulas, grandezas e interpretação geológica.
  Triggers: "física", "resistividade", "anisotropia", "TIV", "tensor", "Hxx", "Hzz",
  "skin depth", "decoupling", "ACp", "ACx", "Feature View", "FV", "geosinais", "GS",
  "USD", "UAD", "UHR", "UHA", "U3DF", "Picasso", "DTB", "Archie", "petrofísica",
  "porosidade", "saturação", "LWD", "BHA", "22 colunas", "22-col", "zobs", "errata".
  v2.0 | Última atualização: 2026-04-29 | Sub-skill da geosteering-v2
---

# Geosteering AI — Domínio Físico e Geofísico

> **Sub-skill especializada** — Física, petrofísica e eletromagnetismo para LWD/geosteering.
> Skill principal: `geosteering-v2` | Código: `geosteering-code-v2` | Modelos: `geosteering-models` | Losses: `geosteering-losses`

---

## 1. O Problema Físico

### 1.1 Objetivo e Analogia

**Objetivo:** Traduzir medidas eletromagnéticas (EM) feitas por ferramentas dentro de poços de
petróleo em mapas de resistividade elétrica das rochas ao redor do poço usando redes neurais profundas.

```
Analogia médica:
  Ondas EM (20 kHz)               → equivalente a raios-X
  Rochas (folhelho, arenito, carb.) → equivalente ao corpo humano
  Ferramenta LWD no poço          → equivalente ao tomógrafo CT
  Rede neural profunda            → equivalente à reconstrução de imagem
  Saída: rho_h (horizontal) + rho_v (vertical) → equivalente a densidade de tecido
```

### 1.2 Cadeia Completa de Geosteering

```
1. Downhole:  LWD/MWD transmite ondas EM → receptores medem tensor H
2. Telemetria: Mud pulse / EM / wired drill pipe → superfície
3. Aquisição:  WITSML/OPC data streams (superfície)
4. Inversão:   ★ ESTE PIPELINE ★ — EM → (rho_h, rho_v) via DL
5. Interpretação: Detecção de camadas + cálculo DTB
6. HMI:        Interface para geólogo/engenheiro de poço
7. Controle:   Comandos direcionais ao BHA
```

O pipeline é o núcleo de inversão **(etapa 4)**: recebe dados EM brutos e produz perfis de
resistividade em milissegundos (<31 ms), permitindo decisões em tempo real durante a perfuração.

---

## 2. Resistividade Elétrica

### 2.1 Tabela de Resistividades Típicas

| Material | Resistividade (Ohm.m) | Significado para geosteering |
|:---------|:---------------------:|:-----------------------------|
| Água salgada (formação) | 0,1 – 5 | Condutiva (íons dissolvidos) — indica zona úmida |
| Folhelho (shale) | 1 – 10 | Moderadamente condutivo — selante |
| Arenito com água | 5 – 50 | Intermediário |
| **Arenito com óleo** | **50 – 500** | **Resistivo — zona produtora** |
| Carbonato compacto | 100 – 10.000 | Muito resistivo |
| Sal (evaporito) | >10.000 | Extremamente resistivo |

**Importância para geosteering:** O contraste de resistividade indica fronteiras entre camadas
geológicas (reservatório vs. selante) — informação crítica para manter o poço dentro da zona produtora.

### 2.2 Anisotropia TIV (Transversalmente Isotrópica Vertical)

Rochas sedimentares conduzem corrente elétrica de forma **diferente** nas direções horizontal e
vertical, como a fibra da madeira:

```
rho_h (horizontal) — resistividade ao longo das camadas (plano de estratificação)
rho_v (vertical)   — resistividade perpendicular às camadas (direção de compactação)

INVARIANTE FÍSICO: rho_v >= rho_h em TIV
  Razão típica rho_v / rho_h: 1,5 – 10× (folhelhos podem chegar a 20×)
```

**Consequência no pipeline:** O modelo SEMPRE produz **2 saídas por ponto**: `(rho_h, rho_v)`.
Shape de saída: `(batch, N_MEDIDAS, 2)` onde `N_MEDIDAS = 600`.

**Por que 2 saídas?** Ferramentas LWD modernas medem tensor completo H (9 componentes) e as
componentes planar (Hxx) e axial (Hzz) respondem diferentemente a rho_h e rho_v — a rede neural
explora essa diferença de sensibilidade para separar os dois valores.

---

## 3. Ferramenta LWD e Tensor EM

### 3.1 Geometria da Ferramenta

```
Bottom Hole Assembly (BHA):
  ┌─────────────────────────────────────────┐
  │  TX (Transmissor) — emite EM a 20 kHz  │
  │       ↕  L = 1,0 m (SPACING_METERS)    │
  │  RX1 (Receptor 1)                      │
  │       ↕  L = 1,0 m                     │
  │  RX2 (Receptor 2) — compensação       │
  └─────────────────────────────────────────┘
  Arranjo típico: 2 transmissores + 2 receptores (compensação de ganho)
```

### 3.2 Tensor Magnético H (3×3)

```
         | Hxx  Hxy  Hxz |
H =      | Hyx  Hyy  Hyz |    (cada componente: Re + Im = 18 valores reais)
         | Hzx  Hzy  Hzz |

Componentes-chave:
  Hxx (planar)  — sensível a camadas horizontais, rho_h
  Hzz (axial)   — sensível a bulk resistivity
  Hxz, Hzx (cross) — sensíveis a fronteiras e dip angular
  Hxy, Hyx (azimutais) — indicadores de heterogeneidade lateral
```

**Antenas tilted (inclinadas):** Permitem medidas direcionais (azimutais) → importante para
detecção assimétrica de fronteiras (acima vs. abaixo do poço).

---

## 4. Formato de Dados — 22 Colunas

O dataset `.dat` (Fortran binary) tem **22 colunas por ponto de medição**:

```
Col  0: meds     — índice de medição (metadata, NUNCA feature)
Col  1: zobs     — profundidade observada (metros) → INPUT_FEATURE ★
Col  2: res_h    — resistividade horizontal (Ohm.m) → OUTPUT_TARGET ★
Col  3: res_v    — resistividade vertical (Ohm.m) → OUTPUT_TARGET ★
Col  4: Re(Hxx)  — parte real de Hxx → INPUT_FEATURE ★
Col  5: Im(Hxx)  — parte imaginária de Hxx → INPUT_FEATURE ★
Col  6: Re(Hxy)  — parte real de Hxy
Col  7: Im(Hxy)  — parte imaginária de Hxy
Col  8: Re(Hxz)  — parte real de Hxz
Col  9: Im(Hxz)  — parte imaginária de Hxz
Col 10: Re(Hyx)  — parte real de Hyx
Col 11: Im(Hyx)  — parte imaginária de Hyx
Col 12: Re(Hyy)  — parte real de Hyy
Col 13: Im(Hyy)  — parte imaginária de Hyy
Col 14: Re(Hyz)  — parte real de Hyz
Col 15: Im(Hyz)  — parte imaginária de Hyz
Col 16: Re(Hzx)  — parte real de Hzx
Col 17: Im(Hzx)  — parte imaginária de Hzx
Col 18: Re(Hzy)  — parte real de Hzy
Col 19: Im(Hzy)  — parte imaginária de Hzy
Col 20: Re(Hzz)  — parte real de Hzz → INPUT_FEATURE ★
Col 21: Im(Hzz)  — parte imaginária de Hzz → INPUT_FEATURE ★
```

### 4.1 Errata — Valores Críticos (IMUTÁVEIS)

```python
# Validados automaticamente por PipelineConfig.__post_init__()
INPUT_FEATURES  = [1, 4, 5, 20, 21]    # zobs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)
OUTPUT_TARGETS  = [2, 3]               # res_h, res_v
FREQUENCY_HZ    = 20000.0              # 20 kHz (range: 100–1e6 Hz)
SPACING_METERS  = 1.0                  # 1,0 m (range: 0,1–10,0 m)
SEQUENCE_LENGTH = 600                  # amostras por perfil (range: 10–100.000)
TARGET_SCALING  = "log10"              # NUNCA "log" (ln ≠ log10)
eps_tf          = 1e-12               # NUNCA 1e-30 (causa subnormais em float32)
```

**Razão do TARGET_SCALING = "log10":** Resistividade varia 4+ ordens de magnitude (0,1 a 10.000+
Ohm.m). A transformação log10 comprime a faixa para ~[-1, +4], estabilizando gradientes durante
o treinamento. `"log"` (logaritmo natural) produz escala incompatível com os scalers e causa
divergência silenciosa.

---

## 5. Skin Depth e Profundidade de Investigação

### 5.1 Fórmula do Skin Depth

```
delta = sqrt(2 / (omega * mu * sigma))

Parâmetros:
  omega = 2π × f         (frequência angular)
  mu    = 4π × 10⁻⁷ H/m (permeabilidade magnética — vácuo ≈ rocha)
  sigma = 1 / rho        (condutividade)

Para f = 20 kHz, rho = 10 Ohm.m:
  omega = 2π × 20000 = 125.663,7 rad/s
  sigma = 0,1 S/m
  delta = sqrt(2 / (125663,7 × 4π×10⁻⁷ × 0,1)) ≈ 11,3 m
```

### 5.2 DOI (Depth of Investigation)

```
DOI ≈ 1,5 – 3,0 m  (configuração típica: f=20 kHz, L=1 m)

Relações gerais:
  Frequência menor → maior penetracao, menor resolução vertical
  Frequência maior → menor penetração, maior resolução vertical
  Espaçamento maior → maior DOI
```

**Para geosteering look-ahead (Constable et al. 2016):** frequências menores (1–2 kHz) + arranjos
longos (5–30 m) permitem detecção de fronteiras até 30 m à frente da broca.

---

## 6. Decoupling EM

O sinal EM medido inclui o **acoplamento direto Tx-Rx** (campo no espaço livre), que deve ser
removido para isolar a resposta da formação:

### 6.1 Constantes de Decoupling

```python
# Derivadas analiticamente para arranjo coaxial (L = SPACING_METERS = 1,0 m)
# Convenção temporal: e^(-iωt)
# Ref: Moran & Gianzero (1979)

ACp = -1 / (4 * pi * L**3)   # ≈ -0,079577 T/A  (planar: Hxx, Hyy)
ACx = +1 / (2 * pi * L**3)   # ≈ +0,159155 T/A  (axial: Hzz)

# Aplicação:
H_formation_xx = H_measured_xx - ACp   # Remove acoplamento planar
H_formation_zz = H_measured_zz - ACx   # Remove acoplamento axial
```

**Por que ACx = -2 × ACp?** No limite estático (f→0), a razão H_axial/H_broadside = -2 para
um dipolo magnético vertical — invariante geométrico que serve como gate de validação analítica.

### 6.2 Relação com os Dados 22-col

Os dados já chegam **sem o acoplamento** (pós-decoupling, realizado pelo firmware da ferramenta
ou pelo simulador Fortran). Os valores nas colunas 4, 5, 20, 21 são os campos de formação H_formation.

---

## 7. Feature Views (6 Transformações)

As Feature Views transformam as 4 componentes EM brutas `[Re(H1), Im(H1), Re(H2), Im(H2)]`
onde H1 = Hxx (planar) e H2 = Hzz (axial):

```
┌──────────────────────────┬────────────────┬───────────┬──────────────────┬──────────────┐
│  Feature View            │ Canal 0        │ Canal 1   │ Canal 2          │ Canal 3      │
├──────────────────────────┼────────────────┼───────────┼──────────────────┼──────────────┤
│  identity / raw          │ Re(H1)         │ Im(H1)    │ Re(H2)           │ Im(H2)       │
│  H1_logH2               │ Re(H1)         │ Im(H1)    │ log10|H2|        │ φ(H2)        │
│  logH1_logH2            │ log10|H1|      │ φ(H1)     │ log10|H2|        │ φ(H2)        │
│  IMH1_IMH2_razao        │ Im(H1)         │ Im(H2)    │ |H1|/|H2|        │ Δφ           │
│  IMH1_IMH2_lograzao     │ Im(H1)         │ Im(H2)    │ log10(|H1|/|H2|) │ Δφ           │
└──────────────────────────┴────────────────┴───────────┴──────────────────┴──────────────┘

Onde: |H| = sqrt(Re² + Im² + ε),  φ(H) = arctan2(Im, Re),  ε = 1e-12
SEMPRE log10 (NUNCA ln) — consistência numpy + TF garantida desde v2.0
```

### 7.1 Escolha de Feature View

| View | Quando usar | Vantagem |
|:-----|:------------|:---------|
| `identity` / `raw` | Baseline, diagnóstico | Sem transformação, interpretável |
| `H1_logH2` | Quando H2 (Hzz) tem alta faixa dinâmica | Preserva SNR de H1, comprime H2 |
| `logH1_logH2` | Geral recomendado | Ambos comprimidos, melhor convergência |
| `IMH1_IMH2_razao` | Quando razão é informativa | Ratio independe de ganho absoluto |
| `IMH1_IMH2_lograzao` | Faixa dinâmica larga em ratio | Comprime razão exponencial |

---

## 8. Geosinais (Perspectiva P4) — 5 Famílias

Razões compensadas em ganho entre componentes do tensor EM, projetadas para detectar
características geológicas específicas:

```
┌──────────┬─────────────────────────────────────────────┬───────────────────┬────────┐
│ Família  │ Fórmula                                      │ Detecta           │ Canais │
├──────────┼─────────────────────────────────────────────┼───────────────────┼────────┤
│ USD      │ (ZZ+XZ)/(ZZ-XZ) × (ZZ-ZX)/(ZZ+ZX)          │ Fronteiras camada │ 2      │
│ UAD      │ (ZZ+XZ)/(ZZ-XZ) × (ZZ+ZX)/(ZZ-ZX)          │ Dip/anisotropia   │ 2      │
│ UHR      │ -2·ZZ / (XX+YY)                              │ Bulk resistivity  │ 2      │
│ UHA      │ XX / YY                                      │ Anisotropia       │ 2      │
│ U3DF     │ (ZZ+YZ)/(ZZ-YZ) × (ZZ-ZY)/(ZZ+ZY)          │ Indicador 3D      │ 2      │
└──────────┴─────────────────────────────────────────────┴───────────────────┴────────┘

Cada família produz 2 canais: atenuação (dB) e deslocamento de fase (graus).
```

### 8.1 Dependências EM por Família

```python
FAMILY_EM_DEPS = {
    "USD":  {"ZZ", "XZ", "ZX"},      # USD/UAD: cross-components (dip-sensitive)
    "UAD":  {"ZZ", "XZ", "ZX"},
    "UHR":  {"ZZ", "XX", "YY"},      # UHR: diagonal (bulk resistivity)
    "UHA":  {"XX", "YY"},             # UHA: planar ratio (azimuthal anisotropy)
    "U3DF": {"ZZ", "YZ", "ZY"},      # U3DF: Y-plane crosses (3D indicator)
}
# Hxx e Hzz são SEMPRE auto-incluídos no baseline P1
```

### 8.2 Canais com P4 Ativo

```python
# N_FEATURES com famílias ativas:
N_FEATURES = 5 + 2 * K   # K = número de famílias de geosinais ativas

# Exemplos:
# P1 (sem GS): 5 features  [zobs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)]
# P4 USD+UHR:  9 features  [5 base + 4 GS]
# P4 todos:   15 features  [5 base + 10 GS]
```

**Invariante da cadeia de dados:** Os Geosinais DEVEM ver o ruído durante o treinamento:
```
raw Re/Im → noise(σ) → FV(noisy) → GS(noisy) → scale → modelo
```
Aplicar GS em dados limpos e depois adicionar ruído é **fisicamente incorreto** — os geosinais
(razões compensadas) têm sensibilidade não-linear ao ruído, e treinar com GS limpos cria
um bias sistemático não observado em campo.

---

## 9. Picasso Plots e DTB (Perspectiva P5)

### 9.1 Picasso Plot (DOD — Depth of Detection)

```
Mapa 2D de DOD em função de (Rt1, Rt2):
  Eixo X: Resistividade da camada 1 (Ohm.m, log-scale)
  Eixo Y: Resistividade da camada 2 (Ohm.m, log-scale)
  Diagonal Rt1 = Rt2: "zona cega" (sem contraste — DOD indefinida)
  Cores: DOD em metros (maior = melhor detecção)

DOD típico do pipeline: 1,5 – 3,0 m
DOD look-ahead (multi-freq): até 30 m (modo EMLA)
```

### 9.2 DTB (Distance to Boundary)

```
DTB = Distância do poço até a fronteira geológica mais próxima (metros)

Configuração com DTB ativo (P5):
  output_channels: 2 → 4  (rho_h, rho_v, DTB_acima, DTB_abaixo)
  output_channels: 2 → 6  (com incertezas por canal)

Importância operacional: permite ao geólogo de geosteering tomar decisões
de ajuste de trajetória ANTES de cruzar a fronteira, não depois.
```

---

## 10. Petrofísica Fundamental

### 10.1 Lei de Archie (1942)

```
Rt = a × Rw / (phi^m × Sw^n)

Onde:
  Rt  = resistividade da formação (Ohm.m) — SAÍDA DO MODELO
  Rw  = resistividade da água de formação (Ohm.m) — dado de campo
  phi = porosidade (fração) — a estimar
  Sw  = saturação de água (fração) — a estimar
  a   = constante de tortuosidade (~1,0)
  m   = expoente de cimentação (~2,0 para arenitos limpos)
  n   = expoente de saturação (~2,0)
```

**Implicação prática:** Conhecendo Rt (saída do modelo) e Rw (dado do poço), pode-se estimar phi
e Sw — propriedades críticas para avaliação quantitativa de reservatório.

### 10.2 Fator de Formação

```
F = a / phi^m = Rt / Rw   (quando Sw = 1,0 — formação 100% saturada com água)
```

### 10.3 Classificação de Litologia

| Litologia | rho_h típico | rho_v típico | rho_v/rho_h |
|:----------|:------------:|:------------:|:-----------:|
| Folhelho | 1–10 | 3–30 | 3–5 |
| Arenito limpo | 20–500 | 20–500 | ~1 |
| Arenito argiloso | 5–50 | 10–100 | 2–5 |
| Carbonato | 100–10.000 | 100–10.000 | ~1 |
| Evaporito (sal) | >10.000 | >10.000 | ~1 |

---

## 11. Modo Dual: Offline vs. Realtime

| Aspecto | Offline (Pesquisa) | Realtime (Geosteering) |
|:--------|:-------------------|:----------------------|
| Dados | Perfil completo do poço (600 amostras) | Sliding window (últimas 600) |
| Rede | Acausal (futuro + passado disponíveis) | Causal (somente passado) |
| Output | `(batch, 600, 2)` | `(1, W, 2-6)` + incerteza |
| Latência | Não crítica | <1 s por amostra |
| Padding Conv1D | `"same"` | `"causal"` |
| Uso | Pós-processamento, pesquisa | Suporte a decisão em tempo real |

**Restrições de tempo real:**

| Requisito | Meta | Status do Pipeline |
|:----------|:----:|:-----------------:|
| Latência | <10 ms/amostra | <31 ms (inversão) |
| Causalidade | Sem dados futuros | 30 arqs causal-compatible |
| Robustez | Ruído real, dropouts | 34+ noise types + curriculum |
| Incerteza | Confiança quantificada | MC Dropout + Ensemble + INN |

---

## 12. Simulador EM — Contexto Técnico

### 12.1 Simuladores Disponíveis

| Simulador | Backend | Performance | Status |
|:----------|:-------:|:-----------:|:------:|
| Fortran `tatu.x` v10.0 | OpenMP | 58.856 mod/h | Produção |
| Python Numba | ProcessPool | ~175k mod/h (2ª sim.) | Produção (SM v2.11) |
| Python JAX | XLA/GPU | 1,5–3× Numba (GPU) | Beta (PR #25) |

### 12.2 Errata do Simulador Fortran (v9.0/v10.0)

O simulador Fortran usa a **convenção de transmissor (T) em +x/+z abaixo** e **receptor (R) em
-x/-z acima**. Esta convenção é oposta à intuição geométrica mas validada contra Moran-Gianzero 1979.

**Paridade garantida:** Diferença máxima Numba vs. Fortran < 1e-12 (5 ordens abaixo do gate 1e-12).

---

## 13. Referências Bibliográficas

| Referência | Contribuição |
|:-----------|:-------------|
| Ellis & Singer (2008) "Well Logging for Earth Scientists" | Física EM, perfilagem, interpretação |
| Moran & Gianzero (1979) | Convenção temporal e^(-iωt), fórmulas ACp/ACx |
| Morales et al. (2025) "Anisotropic resistivity estimation... PINN" | PINNs para inversão EM com UQ |
| Constable et al. (2016) Petrophysics | Look-ahead EM (EMLA) — detecção 5–30 m |
| Wang et al. (2018) J. Geophys. Eng. 15:2339 | Análise de sensibilidade ARM + inversão GN |
| Guoyu et al. (2025) Energies 18:3014 | Geometric factors para look-ahead EM |
| gxag017 (2026) | Modified Propagator Coefficient para modelagem 1D rápida |
| Archie (1942) | Relação resistividade–porosidade–saturação |

---

*Sub-skill de domínio físico — Geosteering AI v2.0 | Última atualização: 2026-04-29*
