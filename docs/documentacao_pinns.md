# Documentação Técnica: Physics-Informed Neural Networks (PINNs) — Geosteering AI v2.0

> **Versão:** 2.0  
> **Autor:** Daniel Leal  
> **Última atualização:** 2026-04-03  
> **Módulo:** `geosteering_ai/losses/pinns.py`  
> **Referência:** `docs/ARCHITECTURE_v2.md` § Losses / PINNs

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Conceitos Fundamentais](#2-conceitos-fundamentais)
3. [Lambda Scheduling — Controle Temporal](#3-lambda-scheduling--controle-temporal)
4. [Cenário 1: Oracle (Referência Simulada)](#4-cenário-1-oracle-referência-simulada)
5. [Cenário 2: Surrogate (Forward Model)](#5-cenário-2-surrogate-forward-model)
6. [Cenário 3: Maxwell (Resíduo PDE)](#6-cenário-3-maxwell-resíduo-pde)
7. [Cenário 4: Smoothness (Tikhonov + TV)](#7-cenário-4-smoothness-tikhonov--tv)
8. [Cenário 5: Skin Depth (Resolução EM)](#8-cenário-5-skin-depth-resolução-em)
9. [Cenário 6: Continuity (L1 Esparso)](#9-cenário-6-continuity-l1-esparso)
10. [Cenário 7: Variational (Deep Ritz)](#10-cenário-7-variational-deep-ritz)
11. [Cenário 8: Self-Adaptive](#11-cenário-8-self-adaptive)
12. [TIV Constraint Layer](#12-tiv-constraint-layer)
13. [SurrogateNet — Forward Model Neural](#13-surrogatenet--forward-model-neural)
14. [build_pinns_loss() — Factory](#14-build_pinns_loss--factory)
15. [Integração no Pipeline de Treinamento](#15-integração-no-pipeline-de-treinamento)
16. [Tutorial Rápido](#16-tutorial-rápido)
17. [Melhorias Futuras](#17-melhorias-futuras)
18. [Referências](#18-referências)

---

## 1. Visão Geral

Physics-Informed Neural Networks (PINNs) integram leis físicas diretamente na
função de perda da rede neural, criando um **regularizador baseado em física**
que guia o aprendizado para soluções fisicamente consistentes.

### Princípio Central

```
L_total = L_data + λ × L_physics
```

Onde:

| Componente    | Descrição                                                        |
|:--------------|:-----------------------------------------------------------------|
| `L_data`      | Loss de dados (RMSE, Huber, etc.) — fidelidade às observações    |
| `L_physics`   | Loss de física (PDE, constraints, etc.) — consistência física    |
| `λ` (lambda)  | Peso dinâmico que controla o balanço entre dados e física        |

### Inventário de Cenários Implementados

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    8 Cenários PINN                              │
  ├─────────┬───────────────┬───────────────────────────────────────┤
  │  Nº     │  Nome         │  Tipo de Constraint                  │
  ├─────────┼───────────────┼───────────────────────────────────────┤
  │  1      │  Oracle       │  Referência simulada (Fortran)       │
  │  2      │  Surrogate    │  Forward model (analítico/neural)    │
  │  3      │  Maxwell      │  Resíduo PDE (Helmholtz 1D)          │
  │  4      │  Smoothness   │  Tikhonov L2 + TV L1                 │
  │  5      │  Skin Depth   │  Resolução EM adaptativa             │
  │  6      │  Continuity   │  L1 esparso (perfil blocky)          │
  │  7      │  Variational  │  Deep Ritz (forma fraca)             │
  │  8      │  Self-Adaptive│  Pesos automáticos por gradiente     │
  ├─────────┼───────────────┼───────────────────────────────────────┤
  │  Extra  │  TIV Layer    │  ρ_v ≥ ρ_h (anisotropia TIV)        │
  └─────────┴───────────────┴───────────────────────────────────────┘
```

### Lambda Scheduling

4 estratégias de agendamento para `λ`:

- **fixed** — salto imediato após warmup
- **linear** — rampa linear (padrão)
- **cosine** — curva-S suave
- **step** — degrau na metade da rampa

### Integração no Pipeline

```
build_pinns_loss(config)  →  LossFactory.build_combined()  →  TrainingLoop
         │                           │                            │
         ▼                           ▼                            ▼
   Cenário PINN              L_data + λ×L_physics         Callback λ(epoch)
   + TIV opcional            + w_tiv×L_tiv
```

---

## 2. Conceitos Fundamentais

### 2.1 O que são PINNs?

PINNs (Physics-Informed Neural Networks) foram formalizadas por **Raissi, Perdikaris
e Karniadakis (2019)** como uma classe de redes neurais que incorporam equações
diferenciais parciais (PDEs) e outras leis físicas diretamente na função de perda.

A ideia central é que, além de minimizar o erro em relação aos dados observados,
a rede também minimiza o **resíduo** de equações físicas conhecidas. Isso permite:

1. **Reduzir a necessidade de dados** — a física complementa os dados escassos
2. **Garantir consistência física** — soluções respeitam leis fundamentais
3. **Melhorar generalização** — regularização baseada em domínio físico
4. **Resolver problemas mal-postos** — a física restringe o espaço de soluções

### 2.2 Regularização Baseada em Física vs. Puramente Data-Driven

```
  ┌───────────────────────────────────────────────────────────────────┐
  │                                                                   │
  │   Data-Driven Puro              PINN (Physics-Informed)          │
  │   ─────────────────             ───────────────────────          │
  │                                                                   │
  │   L = L_data                    L = L_data + λ × L_physics       │
  │                                                                   │
  │   • Aprende apenas              • Aprende com dados              │
  │     dos dados                     E leis físicas                 │
  │   • Pode violar física          • Soluções fisicamente           │
  │   • Precisa de MUITOS             consistentes                   │
  │     dados para generalizar      • Generaliza melhor com          │
  │   • Overfitting em ruído          poucos dados                   │
  │                                 • Regularização natural          │
  │                                                                   │
  └───────────────────────────────────────────────────────────────────┘
```

### 2.3 Por que PINNs para Inversão EM?

A inversão 1D de resistividade a partir de dados EM de LWD (Logging While Drilling)
é um **problema mal-posto** por natureza:

- **Não-unicidade:** Múltiplos perfis de resistividade podem gerar o mesmo sinal EM
- **Sensibilidade ao ruído:** Pequenas perturbações nos dados → grandes variações na solução
- **Ill-conditioning:** A matriz Jacobiana do problema direto é mal-condicionada

PINNs atacam esses problemas ao restringir o espaço de soluções àquelas que
satisfazem as equações de Maxwell (ou aproximações delas), eliminando soluções
não-físicas que um modelo puramente data-driven poderia produzir.

### 2.4 Soft Constraints vs. Hard Constraints

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   Soft Constraint (loss penalty)    Hard Constraint (layer)      │
  │   ─────────────────────────────     ──────────────────────       │
  │                                                                  │
  │   • Adicionado à loss function     • Embutido na arquitetura    │
  │   • L += λ × violation²           • Saída da rede é             │
  │   • Pode ser violado se λ          projetada para satisfazer    │
  │     for pequeno                     a constraint                 │
  │   • Flexível, fácil de             • Garantia absoluta          │
  │     implementar                    • Mais complexo de            │
  │                                      implementar                 │
  │                                                                  │
  │   Exemplo:                          Exemplo:                     │
  │   L_tiv = mean(violation²)          TIVConstraintLayer:          │
  │   onde violation =                  ρ_v = ρ_h + softplus(Δ)     │
  │     max(0, log10ρ_h - log10ρ_v)                                │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

No Geosteering AI v2.0, **todos os 8 cenários PINN usam soft constraints**
(loss penalty), enquanto o **TIV Constraint Layer** pode operar como soft
constraint (loss penalty) ou hard constraint (camada de projeção).

### 2.5 Balanço de Lambda

O parâmetro `λ` é crítico para o desempenho das PINNs:

| λ muito alto              | λ muito baixo                | λ balanceado               |
|:--------------------------|:-----------------------------|:---------------------------|
| Física domina             | Física ignorada              | Equilíbrio ótimo           |
| Modelo ignora dados       | Modelo ignora física         | Fidelidade + consistência  |
| Underfitting nos dados    | Soluções não-físicas         | Generalização robusta      |
| Convergência lenta        | Overfitting no ruído         | Convergência estável       |

**Regra prática:** Iniciar com `λ = 0` (warmup), rampar gradualmente até
`λ_target`, permitindo que a rede primeiro aprenda os dados antes de impor
constraints de física. Este é o princípio do **curriculum learning** aplicado
a PINNs.

---

## 3. Lambda Scheduling — Controle Temporal

### 3.1 Função Principal

```python
def compute_lambda_schedule(
    epoch: int,
    warmup: int,
    ramp: int,
    target: float,
    strategy: str = "linear"
) -> float:
    """Calcula o valor de lambda para o epoch atual.

    Args:
        epoch: Epoch atual (0-indexed).
        warmup: Número de epochs com λ=0 (fase de warmup).
        ramp: Número de epochs para rampar de 0 a target.
        target: Valor final de λ após a rampa.
        strategy: Estratégia de scheduling ("fixed", "linear",
            "cosine", "step").

    Returns:
        Valor de λ para o epoch atual.
    """
```

### 3.2 Três Fases do Scheduling

```
  λ
  ▲
  │                              ┌─────────── Hold (λ = target)
  │                             ╱│
  │                            ╱ │
  │                           ╱  │
  │                          ╱   │
  │                         ╱    │
  │                        ╱     │
  │  Warmup (λ=0)        ╱  Ramp│
  │ ─────────────────────╱      │
  │                      │      │
  └──────────────────────┼──────┼──────────────► epoch
  0                   warmup  warmup+ramp
```

| Fase     | Epochs              | Valor de λ     | Propósito                            |
|:---------|:--------------------|:---------------|:-------------------------------------|
| Warmup   | `0` → `warmup-1`   | `λ = 0`        | Rede aprende dados sem física        |
| Ramp     | `warmup` → `warmup+ramp-1` | `0 → target` | Introdução gradual de física   |
| Hold     | `warmup+ramp` → ∞  | `λ = target`   | Peso final estável                   |

### 3.3 Quatro Estratégias

#### 3.3.1 Fixed

```python
# Salto imediato após warmup — SEM rampa gradual
if epoch < warmup:
    return 0.0
return target
```

```
  λ
  ▲
  │          ┌──────────────────────── target
  │          │
  │          │
  │          │
  │──────────┘
  └──────────┼─────────────────────► epoch
           warmup
```

**Uso:** Quando a constraint de física é simples e a rede tolera um salto
abrupto (ex.: Smoothness, Continuity).

#### 3.3.2 Linear (PADRÃO)

```python
# Rampa linear de 0 a target
progress = (epoch - warmup) / ramp  # 0.0 → 1.0
return target * progress
```

```
  λ
  ▲
  │                    ┌────────── target
  │                   ╱
  │                  ╱
  │                 ╱
  │                ╱
  │               ╱
  │──────────────╱
  └──────────────┼─────┼──────────► epoch
              warmup  warmup+ramp
```

**Uso:** Estratégia padrão e mais estável. Recomendada para a maioria dos
cenários, especialmente Maxwell e Surrogate.

#### 3.3.3 Cosine (Curva-S)

```python
# Curva-S suave via cosseno
progress = (epoch - warmup) / ramp
return target * 0.5 * (1 - cos(π * progress))
```

```
  λ
  ▲
  │                      ┌──────── target
  │                    ╱──
  │                  ╱
  │                ╱
  │              ╱
  │            ──╱
  │──────────╱
  └──────────┼─────┼──────────────► epoch
           warmup  warmup+ramp
```

A curva-S tem **aceleração lenta no início** e **desaceleração suave no final**,
evitando transições abruptas. A fórmula cosseno garante derivada zero nos
extremos:

```
λ(t) = target × 0.5 × (1 - cos(π × t))

onde t = (epoch - warmup) / ramp ∈ [0, 1]

t=0.0 → λ = 0.0          (início suave)
t=0.5 → λ = target/2     (ponto médio)
t=1.0 → λ = target       (chegada suave)
```

**Uso:** Cenários sensíveis a transições abruptas (ex.: Maxwell, Self-Adaptive).

#### 3.3.4 Step (Degrau)

```python
# Degrau na metade da rampa
progress = (epoch - warmup) / ramp
if progress >= 0.5:
    return target
return 0.0
```

```
  λ
  ▲
  │                 ┌───────────── target
  │                 │
  │                 │
  │                 │
  │─────────────────┘
  └──────────┼──────┼─────────────► epoch
           warmup  warmup+ramp/2
```

**Uso:** Quando se deseja um período mais longo de treinamento puramente
data-driven antes de introduzir física (ex.: cenários com dados ruidosos).

### 3.4 Configuração via PipelineConfig

```python
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="smoothness",
    pinns_lambda_target=0.1,       # λ final
    pinns_warmup_epochs=10,        # 10 epochs sem física
    pinns_ramp_epochs=20,          # 20 epochs de rampa
    pinns_lambda_strategy="cosine" # curva-S
)
```

---

## 4. Cenário 1: Oracle (Referência Simulada)

### 4.1 Conceito

O cenário Oracle compara a predição da rede `ρ_pred` diretamente com uma
**referência simulada** `ρ_reference` gerada pelo modelo direto (forward model)
do código Fortran. É o cenário mais simples e barato computacionalmente.

### 4.2 Formulação Matemática

```
L_oracle = norm(ρ_pred - ρ_reference)
```

Onde `norm` pode ser:

| Norma   | Fórmula                            | Característica                   |
|:--------|:-----------------------------------|:---------------------------------|
| `l2`    | `mean((ρ_pred - ρ_ref)²)`         | Penaliza outliers fortemente     |
| `l1`    | `mean(|ρ_pred - ρ_ref|)`          | Robusta a outliers               |
| `huber` | Huber loss com δ=1.0              | Híbrida: L2 perto, L1 longe     |

### 4.3 Caso de Uso

```
  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
  │ Modelo Geol. │────▶│ Fortran FWD  │────▶│ ρ_reference     │
  │ (sintético)  │     │ Model        │     │ (ground truth)  │
  └─────────────┘     └──────────────┘     └────────┬────────┘
                                                     │
                                                     ▼
  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
  │ Dados EM    │────▶│ Rede Neural  │────▶│ ρ_pred          │
  │ (+ ruído)   │     │ (inversão)   │     │                 │
  └─────────────┘     └──────────────┘     └────────┬────────┘
                                                     │
                                          L_oracle = ║ρ_pred - ρ_ref║
```

**Vantagens:**
- Complexidade computacional mínima: O(N) por amostra
- Funciona como "teacher forcing" — referência confiável guia o aprendizado
- Ideal para dados sintéticos com ground truth disponível

**Limitações:**
- Requer `ρ_reference` no dataset (disponível apenas para dados sintéticos)
- Não aplicável a dados de campo (sem ground truth)

### 4.4 Configuração

```python
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="oracle",
    pinns_oracle_norm="l2",        # "l1", "l2", "huber"
    pinns_lambda_target=0.05,
)
```

---

## 5. Cenário 2: Surrogate (Forward Model)

O cenário Surrogate implementa um **modelo direto** (forward model) que
calcula o sinal EM esperado a partir da resistividade predita pela rede,
comparando com o sinal EM observado. Três modos estão disponíveis.

### 5.1 Modo A — Magnitude Analítica

#### Formulação Física

O skin depth (profundidade de penetração) para um meio condutor homogêneo é:

```
δ = sqrt(2ρ / (ωμ₀))

onde:
  ρ  = resistividade (Ω·m)
  ω  = 2π × f = 2π × 20000 = 125663.7 rad/s
  μ₀ = 4π × 10⁻⁷ H/m (permeabilidade magnética do vácuo)
```

A magnitude do campo magnético a uma distância `L` da fonte é:

```
|H| = AC × exp(-L / δ)

onde:
  L  = SPACING_METERS = 1.0 m
  AC = constante geométrica dependente da orientação do dipolo
```

#### Constantes Geométricas (Decoupling)

Para espaçamento L = 1.0 m:

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Orientação      │  Fórmula          │  Valor               │
  ├──────────────────┼───────────────────┼──────────────────────┤
  │  Planar (Hxx)    │  ACp = 1/(4πL³)  │  0.079577 A/m        │
  │  Axial  (Hzz)    │  ACx = 1/(2πL³)  │  0.159155 A/m        │
  └──────────────────┴───────────────────┴──────────────────────┘

  Nota: ACp refere-se a componentes planares (Hxx, Hyy).
        ACx refere-se à componente axial (Hzz).
        Relação: ACx = 2 × ACp (dipolo magnético em meio homogêneo).
```

#### Saída

```
Output Modo A: [log10|Hxx|, log10|Hzz|]
```

Apenas magnitudes — **sem informação de fase**. Útil para diagnósticos
rápidos, mas perde sensibilidade a gradientes e limites de camada.

### 5.2 Modo B — Complexo Analítico (RECOMENDADO)

#### Formulação Física

O campo magnético complexo para um dipolo em meio condutor:

```
H = AC × exp(-(1 + j) × L / δ)
```

Separando em parte real e imaginária:

```
Re(H) = AC × exp(-L/δ) × cos(-L/δ)
Im(H) = AC × exp(-L/δ) × sin(-L/δ)
```

Onde o expoente complexo `-(1+j)L/δ` gera simultaneamente:
- **Atenuação exponencial:** `exp(-L/δ)` — magnitude decai com a distância
- **Rotação de fase:** `cos(-L/δ)` e `sin(-L/δ)` — fase varia com a distância

#### Saída

```
Output Modo B: [Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)]
```

4 canais de saída vs. 2 do Modo A. A **fase carrega informação sobre
gradientes e limites de camada** que a magnitude sozinha não captura.

#### Por que Modo B é Recomendado?

```
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │   Modo A (Magnitude)           Modo B (Complexo)              │
  │   ──────────────────           ─────────────────              │
  │                                                                │
  │   • 2 outputs                  • 4 outputs                    │
  │   • Sem fase                   • Fase + magnitude             │
  │   • Perde gradientes           • Sensível a boundaries        │
  │   • OK para diagnóstico        • RECOMENDADO para treino      │
  │                                                                │
  │   log10|H| perde sinal         Re(H), Im(H) preservam         │
  │   da derivada em               informação direcional           │
  │   interfaces de camada         nas interfaces                  │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

### 5.3 Modo C — Surrogate Neural

#### Conceito

Em vez de usar fórmulas analíticas (que assumem meio homogêneo), o Modo C
utiliza uma **rede neural pré-treinada** (SurrogateNet) como modelo direto.
A SurrogateNet foi treinada para reproduzir a resposta do código Fortran
(modelo 1D de camadas), capturando efeitos de heterogeneidade que os modos
analíticos A e B não capturam.

#### Arquiteturas Disponíveis

| Modelo            | Blocos | Dilatação | Kernel | Campo Receptivo | Parâmetros |
|:------------------|:------:|:---------:|:------:|:---------------:|:----------:|
| SurrogateNet v1   | 6 TCN  | 1→32      | causal | 127 m           | ~2M        |
| SurrogateNet v2   | 4 ModernTCN | —    | DWConv k=51 | 204 m     | ~5M        |

#### Carregamento

```python
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="surrogate",
    pinns_surrogate_mode="C",
    surrogate_model_path="models/surrogate_v2_moderntcn.keras",
)
```

A SurrogateNet é carregada via `surrogate_model_path` e congelada (`trainable=False`)
durante o treinamento da rede de inversão.

#### Pipeline do Modo C

```
  ┌────────────┐                    ┌──────────────────┐
  │ Rede de    │ ──► ρ_pred ──────▶ │ SurrogateNet     │ ──► H_pred
  │ Inversão   │                    │ (congelada)      │
  └────────────┘                    └──────────────────┘
                                              │
                                    L_surrogate = ║H_pred - H_obs║²
                                              │
                                              ▼
  ┌────────────┐                    backprop through SurrogateNet
  │ H_obs      │ ◄────────────────  (gradientes fluem para ρ_pred)
  │ (medido)   │
  └────────────┘
```

### 5.4 Configuração Completa

```python
# Modo A — Magnitude analítica
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="surrogate",
    pinns_surrogate_mode="A",           # magnitude apenas
)

# Modo B — Complexo analítico (RECOMENDADO)
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="surrogate",
    pinns_surrogate_mode="B",           # real + imaginário
)

# Modo C — Surrogate neural
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="surrogate",
    pinns_surrogate_mode="C",           # rede pré-treinada
    surrogate_model_path="path/to/model.keras",
)
```

---

## 6. Cenário 3: Maxwell (Resíduo PDE)

### 6.1 Equação Governante

A equação de Helmholtz 1D para o campo elétrico em um meio condutor:

```
d²E/dz² + k²E = 0
```

Onde o número de onda é:

```
k² = ωμ₀σ = ωμ₀ × 10^(-ρ_log)

com:
  ω   = 2π × FREQUENCY_HZ = 2π × 20000 Hz
  μ₀  = 4π × 10⁻⁷ H/m
  σ   = condutividade = 1/ρ = 10^(-ρ_log)
  ρ_log = log10(ρ) — saída da rede em escala logarítmica
```

### 6.2 Resíduo PDE

O resíduo é formulado em termos da resistividade predita `ρ_log(z)`:

```
Curvature = (d²ρ_log/dz²)²
Coupling  = (dρ_log/dz)² × k²

Residual = (Curvature + Coupling) / (1 + k² + ε)
```

Onde `ε = 1e-12` (estabilidade numérica para float32).

### 6.3 Interpretação Física

```
  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │  Curvature = (d²ρ/dz²)²                                     │
  │  ─────────────────────────                                    │
  │  Penaliza variações abruptas de 2ª ordem na resistividade.  │
  │  Em zonas condutivas (k² grande), o denominador relaxa       │
  │  a penalidade — camadas finas condutivas são permitidas.     │
  │                                                               │
  │  Coupling = (dρ/dz)² × k²                                   │
  │  ────────────────────────                                     │
  │  Acopla o gradiente de resistividade ao número de onda.      │
  │  Gradientes fortes em zonas de alto k² (condutivas) são      │
  │  menos penalizados que em zonas resistivas.                  │
  │                                                               │
  │  Denominador = 1 + k² + ε                                    │
  │  ──────────────────────────                                   │
  │  Normalização adaptativa: zonas condutivas (k² >> 1)         │
  │  toleram mais variação; zonas resistivas (k² << 1) são       │
  │  mais restringidas. Isso reflete a física real da resolução  │
  │  EM — campos EM resolvem melhor em meios resistivos.         │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
```

### 6.4 Cálculo das Derivadas

As derivadas são calculadas via **diferenças finitas** ao longo da dimensão
espacial (z):

```python
# 1ª derivada (central differences)
dρ_dz = (ρ[:, 2:, :] - ρ[:, :-2, :]) / (2 * dz)

# 2ª derivada (central differences)
d2ρ_dz2 = (ρ[:, 2:, :] - 2*ρ[:, 1:-1, :] + ρ[:, :-2, :]) / (dz²)

# onde dz = SPACING_METERS = 1.0 m
```

### 6.5 Configuração

```python
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="maxwell",
    pinns_lambda_target=0.01,          # λ baixo — PDE é forte regularizador
    pinns_lambda_strategy="cosine",    # rampa suave
    pinns_warmup_epochs=15,
    pinns_ramp_epochs=30,
)
```

**Nota:** O cenário Maxwell é o mais caro computacionalmente (requer 2ª
derivada), mas é o mais fisicamente fundamentado.

---

## 7. Cenário 4: Smoothness (Tikhonov + TV)

### 7.1 Formulação

Combinação ponderada de regularização Tikhonov (L2) e Total Variation (L1):

```
L_smoothness = 0.7 × L2_Tikhonov + 0.3 × L1_TV
```

Onde:

```
L2_Tikhonov = mean((dρ_log/dz)²)     — suprime oscilações
L1_TV       = mean(|dρ_log/dz|)       — preserva bordas
```

### 7.2 Motivação Geológica

```
  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │  Realidade Geológica:                                        │
  │  ─────────────────────                                        │
  │  Camadas sedimentares são essencialmente CONSTANTES            │
  │  dentro de cada camada, com transições ABRUPTAS               │
  │  entre camadas adjacentes.                                    │
  │                                                               │
  │     ρ                                                         │
  │     ▲                                                         │
  │     │     ┌──────────┐                                        │
  │     │     │ Arenito  │ ← constante                           │
  │     │─────┘          └──────┐                                 │
  │     │                       │ Folhelho ← constante           │
  │     │                       └───────────                      │
  │     └──────────────────────────────────► z                    │
  │                                                               │
  │  L2 (Tikhonov): penaliza oscilações DENTRO das camadas       │
  │  L1 (TV): permite transições ABRUPTAS entre camadas          │
  │                                                               │
  │  A combinação 70/30 favorece suavidade dentro das camadas    │
  │  enquanto preserva as interfaces geológicas.                 │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
```

### 7.3 Pesos 0.7/0.3

A escolha dos pesos reflete a geologia de reservatórios:

- **0.7 (L2):** Dentro de uma camada, a resistividade é constante — qualquer
  oscilação é artefato numérico ou ruído. L2 suprime essas oscilações.
- **0.3 (L1):** Entre camadas, a transição é abrupta (boundary). L1 (Total
  Variation) preserva descontinuidades sem penalizar excessivamente.

### 7.4 Configuração

```python
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="smoothness",
    pinns_lambda_target=0.1,     # Smoothness tolera λ mais alto
    pinns_lambda_strategy="linear",
)
```

---

## 8. Cenário 5: Skin Depth (Resolução EM)

### 8.1 Conceito

A resolução vertical de uma ferramenta EM é limitada pelo **skin depth** δ.
Variações de resistividade menores que δ não são detectáveis pelo instrumento.
Este cenário penaliza gradientes de resistividade que excedem o limite de
resolução local.

### 8.2 Formulação

```
Penalty(z) = max(0, |dρ_log/dz| - 1/δ(z))²

L_skin_depth = mean(Penalty(z))
```

Onde o skin depth adaptativo é:

```
δ(z) = sqrt(2 / (ωμ₀σ(z)))
     = sqrt(2ρ(z) / (ωμ₀))

com σ(z) = 10^(-ρ_log(z))
```

### 8.3 Comportamento Adaptativo

```
  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │  Zona CONDUTIVA (ρ baixo, σ alto):                           │
  │    δ pequeno → 1/δ GRANDE → constraint APERTADA              │
  │    Gradientes devem ser suaves                                │
  │    (campo EM atenua rapidamente, resolução baixa)            │
  │                                                               │
  │  Zona RESISTIVA (ρ alto, σ baixo):                           │
  │    δ grande → 1/δ PEQUENO → constraint FROUXA                │
  │    Gradientes podem ser mais fortes                           │
  │    (campo EM penetra mais, resolução alta)                   │
  │                                                               │
  │                                                               │
  │  Limite de resolução:                                        │
  │  ─────────────────────                                        │
  │                                                               │
  │    δ(1 Ω·m, 20kHz) ≈ 3.6 m    → constraint apertada         │
  │    δ(10 Ω·m, 20kHz) ≈ 11.3 m  → constraint moderada         │
  │    δ(100 Ω·m, 20kHz) ≈ 35.6 m → constraint frouxa           │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
```

### 8.4 Configuração

```python
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="skin_depth",
    pinns_lambda_target=0.05,
    pinns_lambda_strategy="cosine",
)
```

---

## 9. Cenário 6: Continuity (L1 Esparso)

### 9.1 Formulação

```
L_continuity = mean(|dρ_log/dz|)
```

Regularização L1 pura sobre o gradiente — a forma mais simples de
regularização espacial.

### 9.2 Efeito: Perfis Blocky

A norma L1 no gradiente promove **esparsidade** nos gradientes — a maioria
dos pontos terá gradiente zero (constante), com poucos pontos de transição
abrupta. Isso produz perfis **blocky** (piecewise constant):

```
  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │  ρ                                                            │
  │  ▲                                                            │
  │  │        ┌──────────┐          ← gradiente = 0              │
  │  │        │          │                                        │
  │  │────────┘          │          ← gradiente ≠ 0 (esparso)    │
  │  │                   │                                        │
  │  │                   └──────────  ← gradiente = 0            │
  │  │                                                            │
  │  └──────────────────────────────────────────────────► z       │
  │                                                               │
  │  Resultado: perfil BLOCKY (piecewise constant)               │
  │  Ideal para sedimentos estratificados                         │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
```

### 9.3 Caso de Uso

- **Sedimentos estratificados:** Camadas horizontais com resistividade constante
- **Geosteering em tempo real:** Perfis blocky são mais fáceis de interpretar
  pelo geólogo durante a perfuração
- **Baixo custo computacional:** Apenas 1ª derivada, O(N)

### 9.4 Configuração

```python
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="continuity",
    pinns_lambda_target=0.1,
    pinns_lambda_strategy="fixed",   # Simples o suficiente para step
)
```

---

## 10. Cenário 7: Variational (Deep Ritz)

### 10.1 Conceito

Baseado no método **Deep Ritz** (E & Yu, 2018), este cenário formula o
problema como minimização de um funcional variacional (forma fraca), em vez
de resolver o resíduo da PDE (forma forte, cenário Maxwell).

### 10.2 Formulação

```
L_variational = mean((dρ_log/dz)² / (1 + k²))
```

Comparação com o cenário Maxwell:

```
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │  Maxwell (forma forte):                                       │
  │    Curvature = (d²ρ/dz²)²     ← derivada 2ª                  │
  │    Coupling  = (dρ/dz)² × k²  ← derivada 1ª × k²             │
  │    Residual  = (C + Co) / (1 + k² + ε)                        │
  │                                                                │
  │  Variational (forma fraca):                                   │
  │    L = mean((dρ/dz)² / (1 + k²))  ← apenas derivada 1ª      │
  │                                                                │
  │  Vantagens do Variational:                                    │
  │    • Apenas 1ª derivada (vs 2ª do Maxwell)                    │
  │    • ~60% do custo computacional                              │
  │    • Mais estável numericamente                               │
  │    • Menos sensível a ruído                                   │
  │                                                                │
  │  Desvantagens:                                                │
  │    • Menos preciso fisicamente                                │
  │    • Não captura efeitos de 2ª ordem                          │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

### 10.3 O Denominador `(1 + k²)`

O denominador atua como normalização adaptativa, assim como no cenário Maxwell:

- **k² grande (condutor):** denominador grande → penalidade reduzida
- **k² pequeno (resistor):** denominador ≈ 1 → penalidade integral

Isso reflete a física: em meios condutivos, o campo EM é fortemente atenuado
e tem resolução limitada — variações de resistividade são menos "visíveis".

### 10.4 Configuração

```python
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="variational",
    pinns_lambda_target=0.05,
    pinns_lambda_strategy="linear",
    pinns_warmup_epochs=10,
    pinns_ramp_epochs=20,
)
```

---

## 11. Cenário 8: Self-Adaptive

### 11.1 Conceito

O cenário Self-Adaptive elimina a necessidade de ajustar hiperparâmetros
manuais ao usar **pesos adaptativos baseados no gradiente** da predição.
Regiões com gradientes altos (limites de camada) recebem automaticamente
mais peso.

### 11.2 Formulação

```
w(z) = softplus(|dρ_log/dz|)           — peso adaptativo (atenção)
R(z) = residual da PDE em z            — resíduo local

L_self_adaptive = mean(w × R²) / mean(w)
```

Onde `softplus(x) = log(1 + exp(x))` é uma função suave e positiva.

### 11.3 Comportamento

```
  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │  Gradiente alto (interface de camada):                       │
  │    |dρ/dz| grande → softplus grande → w grande               │
  │    → R² nessa região recebe MAIS peso                        │
  │    → Rede foca em acertar os limites de camada               │
  │                                                               │
  │  Gradiente baixo (interior da camada):                       │
  │    |dρ/dz| ≈ 0 → softplus ≈ log(2) ≈ 0.69 → w moderado     │
  │    → R² nessa região recebe MENOS peso                       │
  │    → Rede não desperdiça capacidade em zonas constantes      │
  │                                                               │
  │  Efeito: ATENÇÃO AUTOMÁTICA nos limites de camada            │
  │                                                               │
  │     w(z)                                                      │
  │     ▲                                                         │
  │     │    ╱╲              ╱╲                                   │
  │     │   ╱  ╲            ╱  ╲         ← picos em interfaces   │
  │     │──╱────╲──────────╱────╲────────                        │
  │     └──────────────────────────────► z                        │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘
```

### 11.4 Vantagens

- **Zero hiperparâmetros** a ajustar para o mecanismo de ponderação
- **Auto-regulação:** mais peso onde a física é mais difícil (boundaries)
- **Adaptativo:** responde automaticamente à complexidade do modelo geológico
- A normalização por `mean(w)` previne que a loss escale com o número de
  interfaces

### 11.5 Configuração

```python
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="self_adaptive",
    pinns_lambda_target=0.05,
    pinns_lambda_strategy="cosine",    # Cosine recomendado
    pinns_warmup_epochs=10,
    pinns_ramp_epochs=25,
)
```

---

## 12. TIV Constraint Layer

### 12.1 Conceito Físico

Em sedimentos com **anisotropia TIV** (Transverse Isotropy with a Vertical
axis of symmetry), a resistividade vertical é SEMPRE maior ou igual à
resistividade horizontal:

```
ρ_v ≥ ρ_h    (sempre válido para TIV sedimentar)
```

Em escala logarítmica:

```
log10(ρ_v) ≥ log10(ρ_h)
```

Esta é uma **lei física fundamental** de sedimentos estratificados:
correntes horizontais encontram caminhos de menor resistência (paralelo),
enquanto correntes verticais devem atravessar todas as camadas (série).

### 12.2 Implementação como Soft Constraint

```python
violation = tf.maximum(0.0, log10_rho_h - log10_rho_v)
L_tiv = tf.reduce_mean(violation ** 2)
```

A penalidade é zero quando `ρ_v ≥ ρ_h` (satisfeita) e cresce quadraticamente
quando violada.

### 12.3 Razões de Anisotropia Típicas

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Litologia         │  λ = ρ_v/ρ_h  │  Observação           │
  ├────────────────────┼───────────────┼───────────────────────┤
  │  Folhelho          │  3× — 5×      │  Alta anisotropia      │
  │  Folhelho arenoso  │  2× — 5×      │  Moderada a alta       │
  │  Areia limpa       │  ~1×          │  Isotrópico            │
  │  Carbonato         │  1× — 3×      │  Variável              │
  │  Evaporito         │  ~1×          │  Isotrópico            │
  └────────────────────┴───────────────┴───────────────────────┘

  Nota: Razões > 10× são raras e geralmente indicam erro de inversão
  ou ambiente exótico (coal seams, thin laminations).
```

### 12.4 Independência do Flag use_pinns

O TIV constraint pode ser ativado **independentemente** dos cenários PINN:

```python
# TIV sem PINN
config = PipelineConfig(
    use_pinns=False,
    use_tiv_constraint=True,    # Ativo mesmo sem PINNs
    tiv_weight=0.01,
)

# TIV com PINN
config = PipelineConfig(
    use_pinns=True,
    pinns_scenario="smoothness",
    use_tiv_constraint=True,    # Somado à loss PINN
    tiv_weight=0.01,
)
```

### 12.5 Loss Combinada com TIV

```
L_total = w_base × L_data + λ × L_physics + w_tiv × L_tiv

onde:
  w_base = 1.0 (peso da loss de dados)
  λ      = pinns_lambda (scheduling dinâmico)
  w_tiv  = tiv_weight (fixo, tipicamente 0.01)
```

---

## 13. SurrogateNet — Forward Model Neural

### 13.1 SurrogateNet v1 (TCN)

Arquitetura baseada em **Temporal Convolutional Network** (TCN):

```
  ┌──────────────────────────────────────────────────────────────┐
  │  SurrogateNet v1 — TCN                                      │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  Input: ρ(z) — perfil de resistividade                      │
  │    ↓                                                         │
  │  Block 1: Conv1D(64, k=3, dilation=1)  → BN → ReLU         │
  │  Block 2: Conv1D(64, k=3, dilation=2)  → BN → ReLU         │
  │  Block 3: Conv1D(64, k=3, dilation=4)  → BN → ReLU         │
  │  Block 4: Conv1D(64, k=3, dilation=8)  → BN → ReLU         │
  │  Block 5: Conv1D(64, k=3, dilation=16) → BN → ReLU         │
  │  Block 6: Conv1D(64, k=3, dilation=32) → BN → ReLU         │
  │    ↓                                                         │
  │  Dense(N_outputs, 'linear')                                  │
  │    ↓                                                         │
  │  Output: H(z) — sinal EM predito                            │
  │                                                              │
  │  Campo receptivo: Σ(dilation × (k-1)) + 1                   │
  │    = 2×(1+2+4+8+16+32) + 1 = 127 pontos = 127 m            │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

**Características:**
- 6 blocos com dilatação exponencial (1, 2, 4, 8, 16, 32)
- Convoluções causais (apenas olham para trás no espaço)
- Batch Normalization em cada bloco
- Campo receptivo de 127 metros — cobre a maioria dos modelos geológicos

### 13.2 SurrogateNet v2 (ModernTCN)

Arquitetura baseada em **ModernTCN** com convoluções depthwise separáveis:

```
  ┌──────────────────────────────────────────────────────────────┐
  │  SurrogateNet v2 — ModernTCN                                │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  Input: ρ(z) — perfil de resistividade                      │
  │    ↓                                                         │
  │  Block 1: DWConv(k=51) → LayerNorm → ConvFFN → Residual    │
  │  Block 2: DWConv(k=51) → LayerNorm → ConvFFN → Residual    │
  │  Block 3: DWConv(k=51) → LayerNorm → ConvFFN → Residual    │
  │  Block 4: DWConv(k=51) → LayerNorm → ConvFFN → Residual    │
  │    ↓                                                         │
  │  Dense(N_outputs, 'linear')                                  │
  │    ↓                                                         │
  │  Output: H(z) — sinal EM predito                            │
  │                                                              │
  │  Campo receptivo: N_blocks × (k-1) + 1                      │
  │    = 4 × (51-1) + 1 = 201 pontos ≈ 204 m (com padding)     │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

**Características:**
- 4 blocos ModernTCN com kernel grande (k=51)
- Depthwise Separable Convolutions — ~5M parâmetros
- LayerNorm (mais estável que BN para séries temporais)
- ConvFFN (Feed-Forward Network convolucional)
- Campo receptivo de ~204 metros — superior ao v1

### 13.3 Comparação v1 vs v2

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Atributo          │  v1 (TCN)      │  v2 (ModernTCN)       │
  ├────────────────────┼────────────────┼───────────────────────┤
  │  Blocos            │  6             │  4                     │
  │  Campo receptivo   │  127 m         │  204 m                 │
  │  Normalização      │  BatchNorm     │  LayerNorm             │
  │  Kernel            │  3 (dilatado)  │  51 (depthwise)        │
  │  Parâmetros        │  ~2M           │  ~5M                   │
  │  Estabilidade      │  Boa           │  Superior              │
  │  Status            │  Produção      │  Recomendado           │
  └────────────────────┴────────────────┴───────────────────────┘
```

---

## 14. build_pinns_loss() — Factory

### 14.1 Árvore de Decisão

```
  build_pinns_loss(config)
  │
  ├── use_pinns = False?
  │   └── return None
  │
  ├── Selecionar cenário:
  │   ├── "oracle"        → _build_oracle_loss(config)
  │   ├── "surrogate"     → _build_surrogate_loss(config)
  │   ├── "maxwell"       → _build_maxwell_loss(config)
  │   ├── "smoothness"    → _build_smoothness_loss(config)
  │   ├── "skin_depth"    → _build_skin_depth_loss(config)
  │   ├── "continuity"    → _build_continuity_loss(config)
  │   ├── "variational"   → _build_variational_loss(config)
  │   └── "self_adaptive" → _build_self_adaptive_loss(config)
  │
  ├── use_tiv_constraint = True?
  │   └── Adicionar _build_tiv_loss(config)
  │
  └── return combined_loss_fn, lambda_var
```

### 14.2 Lambda Variable — Segurança em Grafo TF

O `λ` é implementado como `tf.Variable` (e **NÃO** como variável Python)
para compatibilidade com o modo de execução em grafo do TensorFlow:

```python
# CORRETO: tf.Variable é graph-safe
lambda_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="pinns_lambda")

def combined_loss(y_true, y_pred):
    l_data = base_loss(y_true, y_pred)
    l_physics = physics_loss(y_true, y_pred)
    return l_data + lambda_var * l_physics    # lambda_var é tf.Variable ✓

# PROIBIDO: .numpy() dentro de closure quebra tf.function
def combined_loss_ERRADO(y_true, y_pred):
    current_lambda = some_python_var.numpy()  # NÃO! Não funciona em grafo
    return l_data + current_lambda * l_physics
```

### 14.3 PINNsLambdaCallback

O callback atualiza `lambda_var` a cada epoch usando `compute_lambda_schedule`:

```python
class PINNsLambdaCallback(tf.keras.callbacks.Callback):
    """Atualiza lambda PINN a cada epoch via scheduling.

    O callback usa tf.Variable.assign() para atualizar o peso
    de forma graph-safe, sem precisar de .numpy() ou eager mode.
    """

    def __init__(self, lambda_var, config):
        self.lambda_var = lambda_var
        self.config = config

    def on_epoch_begin(self, epoch, logs=None):
        new_lambda = compute_lambda_schedule(
            epoch=epoch,
            warmup=self.config.pinns_warmup_epochs,
            ramp=self.config.pinns_ramp_epochs,
            target=self.config.pinns_lambda_target,
            strategy=self.config.pinns_lambda_strategy,
        )
        self.lambda_var.assign(new_lambda)
```

---

## 15. Integração no Pipeline de Treinamento

### 15.1 Fluxo Completo

```
  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐
  │ PipelineConfig│──▶│ LossFactory  │──▶│ build_combined()     │
  │              │   │ .get(config) │   │                      │
  │ use_pinns=T  │   │              │   │ base_loss = RMSE     │
  │ scenario=X   │   │              │   │ + build_pinns_loss() │
  │ use_tiv=T    │   │              │   │ + build_tiv_loss()   │
  └──────────────┘   └──────────────┘   └──────────┬───────────┘
                                                    │
                                                    ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  L_combined = w_base × L_data + λ × L_physics + w_tiv × L_tiv│
  └──────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  TrainingLoop                                                │
  │    • model.compile(loss=L_combined)                          │
  │    • callbacks = [PINNsLambdaCallback(lambda_var, config)]   │
  │    • model.fit(... callbacks=callbacks ...)                  │
  └──────────────────────────────────────────────────────────────┘
```

### 15.2 Ciclo de Vida do lambda_var

```
  Epoch 0                    Epoch W               Epoch W+R           Epoch N
  │                          │                     │                   │
  │  Warmup: λ=0             │  Ramp: 0→target     │  Hold: λ=target  │
  │  (rede aprende dados)    │  (física gradual)   │  (treino final)  │
  │                          │                     │                   │
  ▼                          ▼                     ▼                   ▼
  PINNsLambdaCallback.on_epoch_begin(epoch)
       │
       ▼
  lambda_var.assign(compute_lambda_schedule(epoch, W, R, target, strategy))
       │
       ▼
  L_combined usa lambda_var atualizado durante todo o epoch
```

### 15.3 Métricas de Monitoramento

Durante o treinamento com PINNs, as seguintes métricas são logadas:

| Métrica           | Descrição                                    |
|:------------------|:---------------------------------------------|
| `loss`            | Loss total combinada                         |
| `data_loss`       | Componente de dados (L_data)                 |
| `physics_loss`    | Componente de física (L_physics × λ)         |
| `tiv_loss`        | Componente TIV (L_tiv × w_tiv), se ativo     |
| `pinns_lambda`    | Valor atual de λ                             |
| `val_loss`        | Loss de validação (apenas L_data, sem PINN)  |

**Nota importante:** A loss de validação (`val_loss`) tipicamente **não inclui**
o termo PINN, pois o objetivo é avaliar a qualidade da predição em relação
aos dados reais, sem o viés da regularização de física.

---

## 16. Tutorial Rápido

### 16.1 Exemplo 1: Smoothness PINN (Mais Simples)

O cenário mais simples para começar — não requer dados extras nem modelos
auxiliares.

```python
from geosteering_ai.config import PipelineConfig

config = PipelineConfig(
    # ── Dados ──────────────────────────────────
    dataset_path="data/inv0dip_22col.npy",

    # ── Modelo ─────────────────────────────────
    model_type="ResNet_18",

    # ── Loss base ──────────────────────────────
    loss_type="rmse",

    # ── PINN: Smoothness ───────────────────────
    use_pinns=True,
    pinns_scenario="smoothness",
    pinns_lambda_target=0.1,
    pinns_warmup_epochs=10,
    pinns_ramp_epochs=20,
    pinns_lambda_strategy="linear",
)
```

**O que acontece:**
1. Epochs 0–9: Rede treina apenas com RMSE (λ=0)
2. Epochs 10–29: λ cresce linearmente de 0 a 0.1
3. Epoch 30+: λ=0.1 fixo, loss = RMSE + 0.1 × Smoothness

### 16.2 Exemplo 2: Surrogate Complexo com TIV

Cenário mais completo — forward model analítico complexo + constraint TIV.

```python
config = PipelineConfig(
    # ── Dados ──────────────────────────────────
    dataset_path="data/inv0dip_22col.npy",

    # ── Modelo ─────────────────────────────────
    model_type="WaveNet",

    # ── Loss base ──────────────────────────────
    loss_type="huber",

    # ── PINN: Surrogate Modo B ─────────────────
    use_pinns=True,
    pinns_scenario="surrogate",
    pinns_surrogate_mode="B",           # Complexo (Re + Im)
    pinns_lambda_target=0.05,
    pinns_warmup_epochs=15,
    pinns_ramp_epochs=25,
    pinns_lambda_strategy="cosine",     # Curva-S suave

    # ── TIV Constraint ─────────────────────────
    use_tiv_constraint=True,
    tiv_weight=0.01,
)
```

**O que acontece:**
1. Epochs 0–14: Rede treina com Huber + TIV (λ_pinn=0, w_tiv=0.01)
2. Epochs 15–39: λ_pinn cresce via cosseno de 0 a 0.05
3. Epoch 40+: L = Huber + 0.05 × Surrogate_B + 0.01 × TIV

### 16.3 Exemplo 3: Self-Adaptive com Cosine Schedule

Cenário avançado — pesos automáticos, sem hiperparâmetros de ponderação.

```python
config = PipelineConfig(
    # ── Dados ──────────────────────────────────
    dataset_path="data/inv0dip_22col.npy",

    # ── Modelo ─────────────────────────────────
    model_type="InceptionTime",

    # ── Loss base ──────────────────────────────
    loss_type="rmse",

    # ── PINN: Self-Adaptive ────────────────────
    use_pinns=True,
    pinns_scenario="self_adaptive",
    pinns_lambda_target=0.05,
    pinns_warmup_epochs=10,
    pinns_ramp_epochs=25,
    pinns_lambda_strategy="cosine",
)
```

**O que acontece:**
1. Epochs 0–9: Treino puramente data-driven
2. Epochs 10–34: Self-Adaptive gradualmente ativado (cosine S-curve)
3. A rede automaticamente foca mais nos limites de camada (gradientes altos)
4. Nenhum hiperparâmetro de ponderação espacial a ajustar

---

## 17. Melhorias Futuras

### 17.1 PINN Petrofísica (Archie + Klein)

Incorporar constraints petrofísicas na loss:

```
Lei de Archie:    Rt = a × Rw / (φ^m × Sw^n)
Modelo de Klein:  Relação empírica ρ_h/ρ_v com V_shale

L_petro = violation(Archie) + violation(Klein)
```

**Status:** Conceitual — requer integração com dados petrofísicos (porosidade,
saturação de água, volume de argila).

### 17.2 Multi-Frequency PINN

Explorar múltiplas frequências simultaneamente para melhorar a resolução:

```
L_multi_freq = Σ_f  λ_f × L_physics(f)

onde cada frequência f tem seu próprio skin depth δ(f)
e contribui com resolução em diferentes escalas.
```

**Status:** Requer dados multi-frequência do simulador Fortran.

### 17.3 Adversarial PINN Training

Treinar um discriminador que distingue perfis fisicamente consistentes
de inconsistentes:

```
L_adv = L_data + λ × L_physics + γ × L_adversarial
```

**Status:** Pesquisa — resultados preliminares na literatura são promissores
para problemas de inversão geofísica.

### 17.4 Full INN Posterior

Integrar PINNs com Invertible Neural Networks (INN) para quantificação de
incerteza informada por física:

```
L_INN = L_forward + λ × L_latent

onde L_forward incorpora constraints físicas no espaço de dados
e L_latent regulariza o espaço latente para distribuição gaussiana.
```

**Status:** Arquitetura INN implementada; treino forward+latent pendente.

---

## 18. Referências

### Referências Primárias

1. **Morales, I.D.R., Misra, S., Pardo, D., & Torres-Verdín, C. (2025).**
   "Physics-Informed Neural Networks for Anisotropic Electromagnetic Forward
   Modeling in Earth's Subsurface."
   *Geophysics.* — Aplicação direta de PINNs para modelagem EM em meios
   anisotrópicos, incluindo TIV. Base teórica para os cenários Surrogate
   e Maxwell deste projeto.

2. **Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).**
   "Physics-Informed Neural Networks: A Deep Learning Framework for Solving
   Forward and Inverse Problems Involving Nonlinear Partial Differential
   Equations."
   *Journal of Computational Physics, 378, 686-707.* — Artigo seminal que
   formalizou PINNs. Framework geral de soft constraints via loss function.

3. **Bai, J., Rabczuk, T., Gupta, A., Alzubaidi, L., & Gu, Y. (2022).**
   "A Physics-Informed Neural Network Technique Based on a Modified Loss
   Function for Computational 2D and 3D Solid Mechanics."
   *Computational Mechanics.* — Tutorial abrangente sobre implementação de
   PINNs, incluindo estratégias de balanceamento de lambda.

### Referências de Métodos Específicos

4. **E, W. & Yu, B. (2018).**
   "The Deep Ritz Method: A Deep Learning-Based Numerical Method for Solving
   Variational Problems."
   *Communications in Mathematics and Statistics, 6(1), 1-12.* — Base teórica
   para o cenário Variational (Deep Ritz). Formulação variacional como
   alternativa ao resíduo PDE.

5. **Ward, S.H. & Hohmann, G.W. (1988).**
   "Electromagnetic Theory for Geophysical Applications."
   In *Electromagnetic Methods in Applied Geophysics, Volume 1: Theory.*
   SEG. — Referência clássica para teoria EM em geofísica. Base para as
   equações de Helmholtz e skin depth usadas nos cenários Maxwell e
   Skin Depth.

### Referências Complementares

6. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**
   "Deep Residual Learning for Image Recognition."
   *CVPR.* — Arquitetura ResNet usada como backbone na rede de inversão.

7. **Bai, S., Kolter, J.Z., & Koltun, V. (2018).**
   "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
   for Sequence Modeling."
   *arXiv:1803.01271.* — TCN (Temporal Convolutional Network) — base para
   SurrogateNet v1.

8. **Donghao, L. & Xue, W. (2024).**
   "ModernTCN: A Modern Pure Convolution Structure for General Time Series
   Analysis."
   *ICLR 2024.* — ModernTCN com Depthwise Separable Convolutions — base
   para SurrogateNet v2.

---

> **Documento gerado para:** Geosteering AI v2.0  
> **Módulos relacionados:** `geosteering_ai/losses/pinns.py`, `geosteering_ai/losses/catalog.py`,
> `geosteering_ai/losses/factory.py`, `geosteering_ai/models/surrogate.py`,
> `geosteering_ai/training/callbacks.py`  
> **Consulte também:** `docs/ARCHITECTURE_v2.md`, `docs/documentacao_losses.md`,
> `docs/documentacao_models.md`
