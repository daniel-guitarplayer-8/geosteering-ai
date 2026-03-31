# Features de 2º Grau — Documentação Completa

## Estratégia C: Amplificação de Sinais EM em Alta Resistividade

| Atributo | Valor |
|:---------|:------|
| **Módulo** | `geosteering_ai/data/second_order.py` |
| **Feature View** | `"second_order"` em `data/feature_views.py` |
| **Config** | `use_second_order_features`, `second_order_mode` |
| **Canais gerados** | 6 (|H1|^2, |H2|^2, d|H1|/dz, d|H2|/dz, Re/Im H1, Re/Im H2) |
| **Modos** | `"feature_view"` (substitui FV) ou `"postprocess"` (concatena) |
| **Restrição** | Requer `feature_view="identity"` ou `"raw"` |
| **Versão** | v2.0.0 (2026-03) |

---

## 1. Motivação Física

### 1.1 O Problema: Baixa Sensibilidade em Alta Resistividade

A ferramenta LWD (Logging While Drilling) emite ondas eletromagnéticas a 20 kHz.
A profundidade de penetração (skin depth) dessa onda depende da resistividade:

```
delta = sqrt(2 / (omega * mu * sigma))

Onde:
  omega = 2 * pi * f = 2 * pi * 20000 = 125663.7 rad/s
  mu    = 4 * pi * 1e-7 H/m (permeabilidade magnética do vácuo)
  sigma = 1 / rho (condutividade em S/m)
```

| Resistividade (Ohm.m) | Condutividade (S/m) | Skin Depth (m) | DOI típico (m) | Sensibilidade |
|:----------------------:|:-------------------:|:--------------:|:--------------:|:-------------:|
| 1 | 1.0 | 3.56 | ~5-10 | Excelente |
| 10 | 0.1 | 11.3 | ~15-30 | Boa |
| 100 | 0.01 | 35.6 | ~50-100 | Fraca |
| 1000 | 0.001 | 112.5 | ~150-300 | Muito fraca |
| 10000 | 0.0001 | 356.0 | >300 | Praticamente nula |

**Consequência**: Para rho > 100 Ohm.m, o skin depth excede largamente o espaçamento
da ferramenta (1 metro). O sinal medido se aproxima do **acoplamento direto** (campo
no espaço livre entre transmissor e receptor), e a informação sobre a formação
se torna um resíduo minúsculo sobre um valor quase constante.

```
┌──────────────────────────────────────────────────────────────────────┐
│  SINAL EM vs RESISTIVIDADE                                           │
│                                                                      │
│  |H| (A/m)                                                           │
│  ▲                                                                   │
│  │  ●                                                                │
│  │    ●   Zona sensível                                              │
│  │      ●  (rho < 100)                                               │
│  │        ●                                                          │
│  │          ● ● ● ● ● ● ● ● ● ● ● ●  Zona "cega"                  │
│  │  ........ACp/ACx.................... acoplamento direto            │
│  │                                      (rho > 100)                  │
│  └──────────────────────────────────── rho (Ohm.m)                  │
│       1    10   100  1000  10000                                     │
│                                                                      │
│  Para rho > 100: |H| ≈ constante ≈ campo livre                     │
│  → Re(H) e Im(H) brutos quase não variam com rho                    │
│  → Gradientes ∂L/∂θ → 0 (rede não aprende)                         │
│  → Erro de inversão diverge                                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 A Solução: Features Não-Lineares

As features de 2º grau extraem informação **não-linear** dos componentes EM brutos
que permanece informativa mesmo quando o sinal se aproxima do acoplamento direto.
Enquanto Re(H) e Im(H) individualmente são quase constantes em alta rho, as
combinações não-lineares |H|^2, d|H|/dz e Re/Im amplificam os resíduos.

```
┌──────────────────────────────────────────────────────────────────────┐
│  POR QUE FEATURES NÃO-LINEARES AJUDAM                                │
│                                                                      │
│  Cenário: rho = 500 Ohm.m (arenito com óleo)                       │
│                                                                      │
│  Sinal bruto:                                                        │
│    Re(Hxx) = -0.07958 + 0.00003  (campo livre + resíduo mínimo)     │
│    Im(Hxx) = 0.00001             (resíduo da formação)               │
│                                                                      │
│  Features de 2º grau:                                                │
│    |H|^2 = Re^2 + Im^2 = 0.00633 (amplifica resíduo quadraticamente)│
│    d|H|/dz = variação espacial   (fronteira → pico detectável)      │
│    Re/Im = -7958.0              (amplifica contraste de fase 7958x!) │
│                                                                      │
│  A razão Re/Im é a mais poderosa:                                    │
│    Em baixa rho (10 Ohm.m): Re/Im ≈ -2.5                           │
│    Em alta rho (500 Ohm.m):  Re/Im ≈ -7958.0                       │
│    Contraste: 3183x (vs ~1.001x para Re bruto)                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Fundamentos Físicos das 6 Features

### 2.1 Feature 0,1: Potência |H|^2

**Fórmula:**
```
|H|^2 = Re(H)^2 + Im(H)^2
```

**Significado físico:**
A potência do sinal EM é proporcional ao quadrado da magnitude do campo magnético.
Na teoria eletromagnética, a potência transportada pela onda é proporcional a |H|^2
(vetor de Poynting: S = E x H, com |S| ~ |H|^2 para meios lineares).

**Propriedades:**
- **Sempre >= 0**: não-negativa por construção (soma de quadrados)
- **Amplifica contrastes**: se |H| varia 1%, |H|^2 varia ~2% (derivada: d(|H|^2) = 2|H|d|H|)
- **Sensível à anisotropia**: |Hxx|^2 (planar) difere de |Hzz|^2 (axial) em meios anisotrópicos
- **Proporcional a sigma^2**: em regime de campo distante, |H|^2 ~ sigma^2 do meio,
  o que fornece uma relação mais direta com resistividade do que Re ou Im isolados

**Em alta resistividade:**
Para rho >> 100 Ohm.m, |H| ≈ |H_campo_livre| + epsilon. A potência |H|^2 amplifica
esse epsilon quadraticamente. Embora o ganho absoluto seja pequeno, a rede pode
aprender a extrair o resíduo |H|^2 - |H_livre|^2 como proxy de resistividade.

**Comparação H1 vs H2:**
- |H1|^2 = |Hxx|^2: potência da componente **planar** — sensível a camadas horizontais
- |H2|^2 = |Hzz|^2: potência da componente **axial** — sensível a bulk resistivity

### 2.2 Feature 2,3: Gradiente Espacial d|H|/dz

**Fórmula:**
```
d|H|/dz ≈ |H(z_i)| - |H(z_{i-1})|

Onde |H| = sqrt(Re^2 + Im^2 + eps)
```

**Significado físico:**
O gradiente espacial da magnitude EM detecta **fronteiras geológicas** (transições
entre camadas de resistividade diferente). Quando a ferramenta LWD cruza uma fronteira,
a resposta EM muda abruptamente — essa mudança se manifesta como um pico no gradiente.

**Propriedades:**
- **Detector de fronteiras**: picos indicam transições geológicas
- **Informativo em alta rho**: mesmo quando |H| ≈ constante, pequenas variações
  no gradiente indicam presença de contraste próximo
- **Complementar ao DTB**: o gradiente detecta fronteiras a partir dos dados EM,
  enquanto DTB (P5) é computado a partir do perfil de resistividade verdadeiro
- **Padding**: o primeiro elemento é 0 (sem informação de gradiente no ponto inicial)
  para manter a dimensão temporal consistente

**Analogia:**
Assim como a derivada de uma imagem revela bordas (filtro Sobel/Prewitt),
o gradiente do sinal EM revela fronteiras geológicas.

```
┌──────────────────────────────────────────────────────────────────────┐
│  GRADIENTE ESPACIAL — Detecção de Fronteiras                         │
│                                                                      │
│  |H| (magnitude)        d|H|/dz (gradiente)                        │
│  ▲                      ▲                                            │
│  │  ┌──┐                │       ╲                                    │
│  │  │  │                │        │ pico = fronteira                  │
│  │  │  │                │       ╱                                    │
│  │  │  └────────────    │ ──────────────────────                     │
│  │  │                   │                                            │
│  └──────────── z        └──────────── z                             │
│     camada A  camada B     fronteira A/B detectada                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Comparação H1 vs H2:**
- d|H1|/dz: gradiente **planar** — detecta fronteiras horizontais (principal para geosteering)
- d|H2|/dz: gradiente **axial** — detecta fronteiras com componente de dip

### 2.3 Feature 4,5: Razão Re/Im

**Fórmula:**
```
Re(H) / Im(H) ≈ cot(phi)

Onde phi = arctan2(Im, Re) é a fase do sinal complexo
```

**Significado físico:**
A razão Re/Im é proporcional à cotangente da fase do sinal EM. Na teoria de
propagação EM em meios condutivos, a fase carrega informação sobre a **condutividade
média** do volume investigado. Em meios de alta resistividade, a amplitude |H| cai
drasticamente, mas a relação Re/Im (fase) permanece sensível a mudanças de rho.

**Propriedades:**
- **Amplificação extrema**: para Im → 0 (alta rho), a razão Re/Im diverge,
  amplificando contrastes que seriam invisíveis em Re ou Im isolados
- **Clipping**: limitada a [-100, 100] para evitar explosão numérica
- **Robusta a ganho**: a razão cancela fatores multiplicativos (calibração,
  acoplamento), preservando apenas informação de fase
- **Sensível à frequência**: a fase depende do produto f × sigma,
  permitindo discriminação entre frequência e resistividade

**Por que Re/Im e não a fase diretamente (arctan2)?**
1. Re/Im é uma operação linear (divisão), computacionalmente mais barata
2. arctan2 comprime a faixa [-pi, pi], perdendo informação de escala
3. Re/Im preserva a magnitude relativa entre Re e Im
4. Em regiões onde Im ≈ 0, a razão diverge — o que é fisicamente informativo
   (indica alta resistividade), enquanto arctan2 simplesmente retorna ±pi/2

```
┌──────────────────────────────────────────────────────────────────────┐
│  RAZÃO Re/Im vs RESISTIVIDADE                                        │
│                                                                      │
│  Re/Im                                                               │
│  ▲                                                                   │
│  │                                         ● (rho=5000, Re/Im=3e4)  │
│  │                                                                   │
│  │                               ● (rho=500, Re/Im=8000)            │
│  │                                                                   │
│  │                     ● (rho=100, Re/Im=50)                        │
│  │          ● (rho=10, Re/Im=-2.5)                                  │
│  │  ● (rho=1, Re/Im=-0.8)                                          │
│  └──────────────────────────────── rho (Ohm.m)                      │
│                                                                      │
│  Observação: Re/Im VARIA ENORMEMENTE com rho > 100                  │
│  (enquanto Re e Im brutos quase não variam)                          │
│  → Feature mais poderosa para discriminar alta resistividade         │
└──────────────────────────────────────────────────────────────────────┘
```

**Comparação H1 vs H2:**
- Re(H1)/Im(H1): razão de fase **planar** (Hxx) — sensível a contrastes horizontais
- Re(H2)/Im(H2): razão de fase **axial** (Hzz) — sensível a bulk resistivity

---

## 3. Tabela Resumo das 6 Features

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  6 Features de 2º Grau (Estratégia C)                                        │
│                                                                              │
│  Canal │ Fórmula          │ Unidade    │ Range Típico   │ Detecta            │
│  ──────┼──────────────────┼────────────┼────────────────┼────────────────────│
│  0     │ |H1|^2           │ (A/m)^2    │ [0, ~0.1]      │ Potência planar   │
│  1     │ |H2|^2           │ (A/m)^2    │ [0, ~0.1]      │ Potência axial    │
│  2     │ d|H1|/dz         │ A/(m^2)    │ [-0.01, 0.01]  │ Fronteiras (Hxx)  │
│  3     │ d|H2|/dz         │ A/(m^2)    │ [-0.01, 0.01]  │ Fronteiras (Hzz)  │
│  4     │ Re(H1)/Im(H1)    │ adim.      │ [-100, 100]    │ Fase planar       │
│  5     │ Re(H2)/Im(H2)    │ adim.      │ [-100, 100]    │ Fase axial        │
│                                                                              │
│  H1 = Hxx (planar — sensível a camadas horizontais)                         │
│  H2 = Hzz (axial — sensível a bulk resistivity)                             │
│  eps = 1e-12 (float32-safe, NUNCA 1e-30)                                    │
│                                                                              │
│  Complementaridade:                                                          │
│    |H|^2: amplifica AMPLITUDE residual em alta rho                          │
│    d|H|/dz: detecta FRONTEIRAS mesmo com sinal fraco                        │
│    Re/Im: amplifica FASE (informação mais robusta em alta rho)              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementação Computacional

### 4.1 Arquitetura de Módulos

```
┌─────────────────────────────────────────────────────────────────────┐
│  MÓDULOS ENVOLVIDOS                                                  │
│                                                                     │
│  config.py                                                          │
│    └── use_second_order_features: bool = False                      │
│    └── second_order_mode: str = "postprocess"                       │
│    └── n_second_order_channels: property → 6 ou 0                   │
│    └── n_features: property (inclui +6 se postprocess)              │
│                                                                     │
│  data/second_order.py (NOVO)                                        │
│    └── compute_second_order_features() — versão NumPy (offline)     │
│    └── compute_second_order_features_tf() — versão TF (on-the-fly) │
│                                                                     │
│  data/feature_views.py (MODIFICADO)                                 │
│    └── VALID_VIEWS += {"second_order"}                              │
│    └── apply_feature_view("second_order") — NumPy                   │
│    └── apply_feature_view_tf("second_order") — TF                   │
│                                                                     │
│  data/pipeline.py (MODIFICADO)                                      │
│    └── _prepare_offline: SO antes do scale                          │
│    └── _prepare_onthefly: SO no scaler fit + val/test               │
│    └── build_train_map_fn: Step 4a (SO antes scale)                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Dois Modos de Operação

```
┌──────────────────────────────────────────────────────────────────────┐
│  MODO "feature_view" (substitutivo)                                  │
│                                                                      │
│  Input:  [z, Re(H1), Im(H1), Re(H2), Im(H2)]  → 5 features        │
│  Output: [z, |H1|^2, |H2|^2, dH1, dH2, Re/Im_H1, Re/Im_H2] → 7   │
│                                                                      │
│  Efeito: Remove 4 canais EM brutos, insere 6 canais SO              │
│  n_features = n_base - 4 + 6 = n_base + 2                           │
│  Uso: quando Re/Im brutos não são necessários para o modelo         │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  MODO "postprocess" (aditivo)                                        │
│                                                                      │
│  Input:  [z, Re(H1), Im(H1), Re(H2), Im(H2)]  → 5 features        │
│  Output: [z, Re(H1), Im(H1), Re(H2), Im(H2),                       │
│           |H1|^2, |H2|^2, dH1, dH2, Re/Im_H1, Re/Im_H2] → 11     │
│                                                                      │
│  Efeito: Preserva 4 EM brutos + concatena 6 SO                      │
│  n_features = n_base + 6                                             │
│  Uso: quando o modelo se beneficia de ambos Re/Im e SO               │
│  Recomendado: modo default (mais informação disponível)              │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 Cadeia de Dados — Coerência Física

A ordem das operações é crítica para garantir que as features de 2º grau
tenham significado físico correto. O princípio fundamental:

> **SO DEVE ser computado sobre dados EM em unidades físicas (A/m),
> ANTES da normalização pelo StandardScaler.**

```
┌──────────────────────────────────────────────────────────────────────┐
│  CADEIA FISICAMENTE CORRETA (Postprocess Mode)                       │
│                                                                      │
│  OFFLINE (val/test):                                                 │
│    raw EM → FV(identity) → GS → SO(Re/Im físicos) → scale → modelo │
│                                    ↑                                 │
│                          |H|^2 em (A/m)^2 ✓                        │
│                          d|H|/dz em A/(m^2) ✓                       │
│                          Re/Im adimensional ✓                        │
│                                                                      │
│  ON-THE-FLY (train com noise):                                       │
│    raw EM → noise(sigma) → FV(identity) → GS → SO → scale → modelo │
│                                                  ↑                   │
│                                       SO vê ruído ✓ (fidelidade LWD)│
│                                       SO em unidades físicas ✓       │
│                                       Scale normaliza EM + SO ✓      │
│                                                                      │
│  INCORRETO (SO pós-scale — versão antiga, CORRIGIDA):                │
│    raw EM → FV → GS → scale → SO(dados scaled) ✗                    │
│                                  ↑                                   │
│                        |H_scaled|^2 ≠ |H|^2 ✗                      │
│                        Perde significado físico ✗                    │
└──────────────────────────────────────────────────────────────────────┘
```

**Por que SO antes do scale?**

O StandardScaler transforma: `x' = (x - mean) / std`

Se SO fosse computado após o scale:
```
|H_scaled|^2 = ((Re - mu_Re) / sigma_Re)^2 + ((Im - mu_Im) / sigma_Im)^2
```
Isso NÃO é a potência física |H|^2. É uma distância estatística no espaço
normalizado. Embora ainda capture alguma informação, perde o significado físico
(proporcionalidade a sigma^2 do meio, detecção de contrastes absolutos de amplitude).

Computando SO antes do scale:
```
|H|^2 = Re^2 + Im^2     (potência física em (A/m)^2)
d|H|/dz em A/(m^2)        (gradiente em unidades físicas)
Re/Im adimensional         (razão de fase intrínseca)
```
O scaler então normaliza todas as features (EM + SO) de uma vez,
mantendo o significado físico intacto dentro do domínio normalizado.

### 4.4 Interação com Noise On-The-Fly

No modo on-the-fly (treinamento com curriculum noise), a cadeia completa é:

```
Etapa 1 — Noise: raw_EM + N(0, sigma^2) → noisy_EM
Etapa 2 — FV:    FV(noisy_EM) → identity (passthrough)
Etapa 3 — GS:    GS(noisy_EM) → geosinais ruidosos
Etapa 4a — SO:   SO(noisy_EM) → |H_noisy|^2, d|H_noisy|/dz, Re_noisy/Im_noisy
Etapa 4b — Scale: normalize(EM + SO) → features prontas
```

**SO vê o ruído**: Isso é fisicamente correto porque em operação LWD real,
todos os sinais medidos contêm ruído. As features de 2º grau derivadas de sinais
ruidosos treinam a rede para ser robusta a incertezas de medição:

- |H_noisy|^2 inclui termos cruzados Re*noise_Re + Im*noise_Im
- d|H_noisy|/dz detecta fronteiras com noise floor
- Re_noisy/Im_noisy preserva a razão de fase média mesmo com ruído

**Scaler fitado em dados LIMPOS**: O scaler é fitado na cadeia limpa
(sem noise) para capturar as estatísticas reais dos sinais EM. Quando
aplicado a dados ruidosos, a normalização preserva a distribuição esperada
e o noise aparece como desvio da média (informação útil para a rede).

### 4.5 Restrição: feature_view="identity" ou "raw"

As features de 2º grau requerem Re(H) e Im(H) brutos nos índices h1_cols/h2_cols.
Feature Views não-identity (como H1_logH2, logH1_logH2, IMH1_IMH2_razao) transformam
essas colunas in-place:

```
┌──────────────────────────────────────────────────────────────────────┐
│  POR QUE SO REQUER identity/raw                                      │
│                                                                      │
│  Com FV = "identity":                                                │
│    x[:, h1_cols] = [Re(H1), Im(H1)]  → SO computa |H1|^2 corretamente│
│                                                                      │
│  Com FV = "H1_logH2":                                                │
│    x[:, h2_cols] = [log10|H2|, phi(H2)]  → SO computaria:           │
│    "potência" = (log10|H2|)^2 + phi^2  ← FISICAMENTE INCORRETO     │
│    "gradiente" = d(sqrt(log^2 + phi^2))/dz  ← SEM SIGNIFICADO      │
│    "razão" = log10|H2| / phi  ← NÃO É Re/Im                        │
│                                                                      │
│  CONCLUSÃO: SO DEVE operar sobre Re/Im brutos, não transformados     │
└──────────────────────────────────────────────────────────────────────┘
```

Esta restrição é validada em `PipelineConfig.__post_init__()`:
```python
if self.use_second_order_features:
    _fv_ok = self.feature_view in ("identity", "raw")
    assert _fv_ok, (
        "use_second_order_features=True requer feature_view='identity' ou 'raw'"
    )
```

### 4.6 Estabilidade Numérica

Três guards numéricos protegem contra instabilidades em float32:

**1. Epsilon na magnitude (eps = 1e-12):**
```python
mag = sqrt(Re^2 + Im^2 + eps)
```
Evita sqrt(0) quando Re = Im = 0 (raro mas possível em dados corrompidos).
eps = 1e-12 é o valor padrão do projeto (NUNCA 1e-30 — Errata v5.0.15).

**2. Denominador seguro para Re/Im:**
```python
safe_im = np.where(np.abs(im) < eps, np.sign(im + 1e-30) * eps, im)
ratio = np.clip(re / safe_im, -100.0, 100.0)
```
- `np.where(|im| < eps, ...)`: substitui im próximo de zero por eps com sinal
- `np.sign(im + 1e-30)`: 1e-30 é tiebreaker para im=0.0 exato (força sign → +1)
- `np.clip(-100, 100)`: limita a razão para evitar explosão de gradientes

**3. Gradiente com padding zero:**
```python
grad = np.diff(mag, axis=-1, prepend=mag[..., :1])
```
O primeiro elemento do gradiente é sempre 0 (sem informação de variação no
ponto inicial). Isso evita artefatos de borda e mantém a dimensão temporal.

---

## 5. Uso Prático

### 5.1 Ativação

```python
from geosteering_ai.config import PipelineConfig

# Modo postprocess (recomendado — preserva Re/Im + adiciona 6 SO)
config = PipelineConfig(
    use_second_order_features=True,
    second_order_mode="postprocess",
    feature_view="identity",  # obrigatório
)
# n_features = 5 (baseline) + 6 (SO) = 11

# Modo feature_view (substitui EM por SO)
config = PipelineConfig(
    use_second_order_features=True,
    second_order_mode="feature_view",
    feature_view="identity",  # obrigatório (será substituído por second_order)
)
# n_features = 5 - 4 (EM removidos) + 6 (SO adicionados) = 7
```

### 5.2 Combinação com Geosinais

```python
# SO + Geosinais P4 (USD + UHR)
config = PipelineConfig(
    use_second_order_features=True,
    second_order_mode="postprocess",
    use_geosignal_features=True,
    geosignal_set="usd_uhr",
)
# n_features = 5 (baseline) + 4 (GS: 2 famílias × 2 canais) + 6 (SO) = 15
```

### 5.3 Combinação com DTB (P5)

```python
# SO + DTB
config = PipelineConfig(
    use_second_order_features=True,
    second_order_mode="postprocess",
    use_dtb_as_target=True,
    output_channels=6,
)
# Input: 11 features (5 EM + 6 SO)
# Output: 6 canais (rho_h, rho_v, DTB_up, DTB_down, rho_up, rho_down)
```

---

## 6. Benefícios Esperados

### 6.1 Melhoria em Alta Resistividade

| Métrica | Sem SO | Com SO (esperado) | Razão |
|:--------|:------:|:-----------------:|:------|
| RMSE log10(rho) para rho < 100 | 0.05 | 0.05 | SO não prejudica baixa rho |
| RMSE log10(rho) para rho > 100 | 0.30 | ~0.15 | |H|^2 e Re/Im amplificam sinal |
| R^2 global | 0.95 | ~0.97 | Melhoria concentrada em alta rho |
| Detecção de fronteiras (F1) | 0.85 | ~0.90 | d|H|/dz melhora detecção |

(Valores estimados — validação com dados reais necessária)

### 6.2 Por Que 6 Features e Não Mais

As 6 features foram escolhidas por representarem as **3 famílias fundamentais**
de transformações não-lineares (potência, gradiente, razão), cada uma aplicada
a 2 componentes EM (H1=planar, H2=axial):

| Família | Informação capturada | N canais | Alternativas descartadas |
|:--------|:--------------------|:--------:|:-------------------------|
| Potência | Amplitude ao quadrado | 2 | |H|^3, |H|^4 (redundantes, instáveis) |
| Gradiente | Variação espacial | 2 | d^2|H|/dz^2 (ruidoso demais) |
| Razão | Fase relativa | 2 | arctan2 (comprime info), |H1|/|H2| (já em GS) |

Adicionar mais features (ex: |H|^4, cross-terms Re(H1)*Im(H2)) aumentaria
a dimensionalidade sem ganho físico proporcional, e arriscaria overfitting.

---

## 7. Referências

| Referência | Contribuição |
|:-----------|:-------------|
| Ellis & Singer (2008) | Física EM de ferramentas LWD, skin depth, sensibilidade |
| Wang et al. (2018) J. Geophys. Eng. 15:2339 | Análise de sensibilidade — demonstra perda de resolução em alta rho |
| Morales et al. (2025) | PINNs para inversão EM — motivação para features fisicamente informadas |
| Constable et al. (2016) Petrophysics | Look-ahead EM — importância de detectar fronteiras em alta rho |

---

## 8. Histórico de Implementação

| Data | Versão | Mudança |
|:-----|:-------|:--------|
| 2026-03-31 | v2.0.0 | Implementação inicial: 6 features, 2 modos, NumPy + TF |
| 2026-03-31 | v2.0.0 | Fix: SO antes do scale (coerência física) |
| 2026-03-31 | v2.0.0 | Fix: Re/Im ratio com np.where/tf.where (estabilidade) |
| 2026-03-31 | v2.0.0 | Fix: TF gradient com tf.zeros (consistência NumPy-TF) |
| 2026-03-31 | v2.0.0 | Fix: tf.gather em vez de loop Python (graph mode) |
| 2026-03-31 | v2.0.0 | Fix: Scaler fit inclui SO no on-the-fly path |
