# Documentação Técnica: Inferência Offline — Geosteering AI v2.0

> **Módulo:** `geosteering_ai/inference/`
> **Versão:** 2.0
> **Autor:** Daniel Leal
> **Framework:** TensorFlow 2.x / Keras (exclusivo)
> **Última atualização:** 2026-04-03

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Conceitos Fundamentais](#2-conceitos-fundamentais)
3. [InferencePipeline — Classe Principal](#3-inferencepipeline--classe-principal)
4. [Cadeia de Processamento Detalhada](#4-cadeia-de-processamento-detalhada)
5. [Quantificação de Incerteza (UQ)](#5-quantificação-de-incerteza-uq)
6. [Exportação de Modelos](#6-exportação-de-modelos)
7. [Serialização e Reprodutibilidade](#7-serialização-e-reprodutibilidade)
8. [Tutorial Rápido](#8-tutorial-rápido)
9. [Melhorias Futuras](#9-melhorias-futuras)
10. [Referências](#10-referências)

---

## 1. Visão Geral

A **inferência offline** no Geosteering AI v2.0 corresponde ao processamento em
lote (*batch processing*) de um perfil de poço completo, realizado após a
aquisição dos dados eletromagnéticos (EM). Diferentemente da inferência em tempo
real (realtime/streaming), o modo offline opera sobre o conjunto integral de
medições, permitindo o uso de convoluções acausais e acesso bidirecional à
sequência temporal.

### Características Principais

| Atributo               | Valor / Descrição                                          |
|:-----------------------|:-----------------------------------------------------------|
| **Modo de operação**   | Batch — perfil completo processado de uma só vez           |
| **Classe principal**   | `InferencePipeline`                                        |
| **Cadeia interna**     | FV → GS → scale → model.predict → inverse_scaling          |
| **Entrada**            | Dados brutos 22 colunas `.dat` — shape `(N, seq_len, 22)` |
| **Saída**              | Resistividade em Ohm.m — shape `(N, seq_len, 2)`          |
| **Targets**            | `rho_h` (resistividade horizontal) e `rho_v` (vertical)   |
| **Domínio de escala**  | Predição em log10 → inversa para Ohm.m linear              |
| **Serialização**       | `model.keras` + `scalers.joblib` + `config.yaml`          |

### Diagrama de Alto Nível

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    INFERÊNCIA OFFLINE — VISÃO GERAL                │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │   Dados brutos (.dat)     Artefatos salvos                         │
  │   ┌──────────────┐        ┌──────────────────┐                     │
  │   │ (N, seq, 22) │        │  model.keras     │                     │
  │   └──────┬───────┘        │  scalers.joblib  │                     │
  │          │                │  config.yaml     │                     │
  │          │                └────────┬─────────┘                     │
  │          │                         │                               │
  │          ▼                         ▼                               │
  │   ┌─────────────────────────────────────────┐                      │
  │   │          InferencePipeline              │                      │
  │   │  ┌─────┐  ┌────┐  ┌───────┐  ┌───────┐│                      │
  │   │  │ FV  │→ │ GS │→ │ Scale │→ │ Model ││                      │
  │   │  └─────┘  └────┘  └───────┘  └───────┘│                      │
  │   └───────────────────┬─────────────────────┘                      │
  │                       │                                            │
  │                       ▼                                            │
  │             ┌─────────────────┐                                    │
  │             │ inverse_scaling │                                    │
  │             │  10^y → Ohm.m   │                                    │
  │             └────────┬────────┘                                    │
  │                      │                                             │
  │                      ▼                                             │
  │            ┌──────────────────┐                                    │
  │            │ (N, seq_len, 2)  │                                    │
  │            │  rho_h, rho_v    │                                    │
  │            │  em Ohm.m        │                                    │
  │            └──────────────────┘                                    │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Conceitos Fundamentais

### 2.1. O que é inferência offline em geofísica?

Na perfilagem de poços direcionais (LWD — *Logging While Drilling*), os sensores
eletromagnéticos coletam dados continuamente durante a perfuração. A **inferência
offline** refere-se ao processamento posterior (*post-processing*) desses perfis,
após o término da corrida de perfilagem. Nesse cenário, o geofísico dispõe de
todo o perfil de resistividade para análise, sem restrições de latência.

Contextos típicos de uso:

- **Validação de modelos:** Comparar predições do modelo de deep learning com
  inversões convencionais (1D analítica, métodos iterativos).
- **Pesquisa e desenvolvimento:** Testar novas arquiteturas, losses e
  hiperparâmetros em dados históricos.
- **Análise retrospectiva:** Reprocessar poços antigos com modelos atualizados
  para melhorar a interpretação geológica.
- **Calibração de incerteza:** Executar múltiplas passagens (MC Dropout,
  ensembles) para quantificar a confiança das predições.

### 2.2. Diferença entre Offline e Realtime

| Aspecto                | Offline (Batch)                     | Realtime (Streaming)                |
|:-----------------------|:------------------------------------|:------------------------------------|
| **Dados disponíveis**  | Perfil completo                     | Janela deslizante                   |
| **Tipo de convolução** | Acausal (`padding="same"`)          | Causal (`padding="causal"`)         |
| **Latência**           | Irrelevante (post-processing)       | Crítica (< 1 segundo por amostra)  |
| **Uso de memória**     | Proporcional ao tamanho do perfil   | Fixo (tamanho da janela)            |
| **Caso de uso**        | Validação, pesquisa, QC             | Decisão durante perfuração          |
| **Acesso temporal**    | Bidirecional (passado + futuro)     | Unidirecional (somente passado)     |
| **Precisão esperada**  | Máxima (usa contexto completo)      | Levemente inferior (sem futuro)     |

### 2.3. Formato dos dados de entrada

Os dados brutos do simulador Fortran possuem **22 colunas** por ponto de
medição, organizadas conforme o padrão do arquivo `.dat`/`.out`:

```
  Coluna   Conteúdo                        Unidade
  ──────   ────────────────────────────     ───────
  0        z_true (profundidade real)       metros
  1        z_obs  (profundidade observada)  metros
  2        rho_h  (resistividade horiz.)    Ohm.m  ← target
  3        rho_v  (resistividade vert.)     Ohm.m  ← target
  4        Re(Hxx)                          A/m
  5        Im(Hxx)                          A/m
  6        Re(Hyy)                          A/m
  7        Im(Hyy)                          A/m
  8        Re(Hxy)                          A/m
  9        Im(Hxy)                          A/m
  10       Re(Hyx)                          A/m
  11       Im(Hyx)                          A/m
  12       Re(Hxz)                          A/m
  13       Im(Hxz)                          A/m
  14       Re(Hzx)                          A/m
  15       Im(Hzx)                          A/m
  16       Re(Hyz)                          A/m
  17       Im(Hyz)                          A/m
  18       Re(Hzy)                          A/m
  19       Im(Hzy)                          A/m
  20       Re(Hzz)                          A/m
  21       Im(Hzz)                          A/m
```

As **features de entrada** selecionadas para o pipeline P1 (baseline) são:

```python
INPUT_FEATURES = [1, 4, 5, 20, 21]
# → z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)
```

Os **targets de saída** são:

```python
OUTPUT_TARGETS = [2, 3]
# → rho_h, rho_v
```

### 2.4. Escala dos targets: domínio log10

A resistividade elétrica varia ao longo de várias ordens de magnitude
(tipicamente 0.1 a 10000 Ohm.m). Para estabilizar o treinamento e melhorar a
convergência, os targets são transformados para o domínio **log10**:

```
y_train = log10(rho)          # Treinamento em log10
y_pred  = model.predict(x)    # Predição em log10
rho     = 10^y_pred           # Inversão para Ohm.m
```

> **IMPORTANTE:** O valor correto de `TARGET_SCALING` é `"log10"`, nunca `"log"`
> (logaritmo natural). Essa distinção é crítica: confundir log10 com ln introduz
> um fator de ~2.3× nas predições de resistividade.

---

## 3. InferencePipeline — Classe Principal

A classe `InferencePipeline` encapsula toda a lógica necessária para transformar
dados brutos EM em predições de resistividade. Ela coordena a cadeia completa de
processamento: extração de features, transformações físicas (FV, GS),
normalização, predição pelo modelo treinado e inversão de escala.

### 3.1. Construtor

```python
class InferencePipeline:
    """Pipeline completo de inferência offline para inversão 1D de resistividade.

    Encadeia Feature View → Geosignals → Scaling → Model → Inverse Scaling,
    garantindo que os dados passem pelas mesmas transformações aplicadas
    durante o treinamento. Os parâmetros do scaler são congelados no
    momento do treinamento para reprodutibilidade.

    Attributes:
        config: PipelineConfig com todos os hiperparâmetros do pipeline.
        model: Modelo Keras treinado (tf.keras.Model).
        scaler_params: Dicionário com parâmetros do scaler (mean, std ou
            min, max) fitados em dados limpos durante o treinamento.

    Example:
        >>> config = PipelineConfig.from_yaml("configs/robusto.yaml")
        >>> pipeline = InferencePipeline.load("artifacts/run_042/")
        >>> rho_pred = pipeline.predict(raw_data)
        >>> # rho_pred.shape == (N, seq_len, 2)  # Ohm.m
    """

    def __init__(
        self,
        config: PipelineConfig,
        model: tf.keras.Model,
        scaler_params: dict,
    ):
```

**Parâmetros:**

| Parâmetro        | Tipo               | Descrição                                              |
|:-----------------|:-------------------|:-------------------------------------------------------|
| `config`         | `PipelineConfig`   | Configuração completa do pipeline (246 campos)         |
| `model`          | `tf.keras.Model`   | Modelo Keras treinado, pronto para `model.predict()`   |
| `scaler_params`  | `dict`             | Parâmetros de normalização fitados em dados limpos     |

### 3.2. Método `predict`

```python
def predict(
    self,
    raw_data: np.ndarray,
    theta: float = 0.0,
    freq: float = 20000.0,
    return_uncertainty: bool = False,
    mc_samples: int = 30,
) -> Union[np.ndarray, UncertaintyResult]:
    """Executa inferência offline sobre dados brutos 22 colunas.

    Cadeia de processamento:
      raw_data → extract_features → inject_theta_freq → FV → GS →
      scale → model.predict → inverse_target_scaling → rho (Ohm.m)

    Args:
        raw_data: Dados brutos do simulador, shape (N, seq_len, 22).
            Cada amostra contém seq_len pontos de medição com 22 colunas
            (profundidade, componentes EM, targets). Os targets nas
            colunas [2,3] NÃO são usados na predição — servem apenas
            para validação posterior.
        theta: Ângulo de mergulho da ferramenta em graus. Default 0.0
            (horizontal). Usado nos pipelines P2/P3 que injetam theta
            como feature adicional para inversão multi-ângulo.
        freq: Frequência de operação do sensor EM em Hz. Default 20000.0
            (20 kHz), correspondente ao sensor LWD padrão simulado.
            Range válido: 100–1e6 Hz.
        return_uncertainty: Se True, executa quantificação de incerteza
            (MC Dropout) e retorna UncertaintyResult ao invés de ndarray.
        mc_samples: Número de passagens Monte Carlo quando
            return_uncertainty=True. Default 30. Valores típicos: 20–100.

    Returns:
        np.ndarray: Resistividade predita, shape (N, seq_len, 2), em Ohm.m.
            Canal 0 = rho_h (horizontal), Canal 1 = rho_v (vertical).
        UncertaintyResult: Se return_uncertainty=True, retorna dataclass
            com mean, std, ci_lower, ci_upper, method e n_samples.

    Note:
        O scaler é fitado em dados LIMPOS (sem ruído) durante o treinamento.
        Os dados de inferência podem conter ruído real de campo — o scaler
        aplicará a mesma normalização, e o modelo foi treinado para ser
        robusto a ruído via curriculum noise durante o treinamento.
    """
```

### 3.3. Métodos `save` e `load`

```python
def save(self, path: str) -> None:
    """Salva artefatos de inferência em diretório.

    Cria três arquivos no diretório especificado:
      - model.keras    → modelo Keras completo (pesos + arquitetura)
      - scalers.joblib → parâmetros do scaler serializados
      - config.yaml    → snapshot da PipelineConfig

    Args:
        path: Caminho do diretório de destino. Criado se não existir.

    Example:
        >>> pipeline.save("artifacts/run_042/")
        # Gera:
        #   artifacts/run_042/model.keras
        #   artifacts/run_042/scalers.joblib
        #   artifacts/run_042/config.yaml
    """

@classmethod
def load(cls, path: str) -> "InferencePipeline":
    """Reconstrói InferencePipeline a partir de artefatos salvos.

    Carrega model.keras, scalers.joblib e config.yaml do diretório
    especificado e reconstrói o pipeline completo, pronto para predict().

    Args:
        path: Caminho do diretório contendo os 3 artefatos.

    Returns:
        InferencePipeline: Pipeline reconstruído e pronto para uso.

    Raises:
        FileNotFoundError: Se algum dos 3 artefatos estiver ausente.
        ValueError: Se config.yaml contiver valores inválidos (errata).

    Example:
        >>> pipeline = InferencePipeline.load("artifacts/run_042/")
        >>> rho = pipeline.predict(raw_data)
    """
```

---

## 4. Cadeia de Processamento Detalhada

A inferência offline segue uma cadeia de 7 etapas, cada uma com significado
físico bem definido. A ordem das etapas é idêntica à aplicada durante o
treinamento (exceto pela ausência de injeção de ruído, que é exclusiva do
treinamento).

### Diagrama Completo da Cadeia

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │               CADEIA DE PROCESSAMENTO — INFERÊNCIA OFFLINE           │
  ├───────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │  ETAPA 1: Extração de Features                                       │
  │  ┌──────────────────────────────────────────────────┐                 │
  │  │ raw (N, seq, 22) → features (N, seq, 5)         │                 │
  │  │ INPUT_FEATURES = [1, 4, 5, 20, 21]              │                 │
  │  │ → z_obs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)   │                 │
  │  └──────────────────────┬───────────────────────────┘                 │
  │                         │                                             │
  │                         ▼                                             │
  │  ETAPA 2: Injeção theta/freq (P2/P3)                                 │
  │  ┌──────────────────────────────────────────────────┐                 │
  │  │ Se multi-ângulo: concat([theta, freq], features) │                 │
  │  │ (N, seq, 5) → (N, seq, 7)  [opcional]           │                 │
  │  └──────────────────────┬───────────────────────────┘                 │
  │                         │                                             │
  │                         ▼                                             │
  │  ETAPA 3: Feature View (FV)                                          │
  │  ┌──────────────────────────────────────────────────┐                 │
  │  │ Transformação das componentes EM                 │                 │
  │  │ 7 modos: identity, H1_logH2, logH1_logH2, ...   │                 │
  │  │ (N, seq, 5) → (N, seq, N_FV)                    │                 │
  │  └──────────────────────┬───────────────────────────┘                 │
  │                         │                                             │
  │                         ▼                                             │
  │  ETAPA 4: Geosignals (GS)                                           │
  │  ┌──────────────────────────────────────────────────┐                 │
  │  │ Computa sinais derivados USD/UHR                 │                 │
  │  │ 5 modos: none, USD, UHR, USD_UHR, full          │                 │
  │  │ (N, seq, N_FV) → (N, seq, N_FV + N_GS)          │                 │
  │  │ Tipicamente +4 canais (USD_UHR)                  │                 │
  │  └──────────────────────┬───────────────────────────┘                 │
  │                         │                                             │
  │                         ▼                                             │
  │  ETAPA 5: Normalização (Scale)                                       │
  │  ┌──────────────────────────────────────────────────┐                 │
  │  │ Aplica scaler fitado em dados LIMPOS             │                 │
  │  │ StandardScaler: (x - mean) / std                 │                 │
  │  │ Parâmetros congelados no treinamento             │                 │
  │  └──────────────────────┬───────────────────────────┘                 │
  │                         │                                             │
  │                         ▼                                             │
  │  ETAPA 6: Predição do Modelo                                         │
  │  ┌──────────────────────────────────────────────────┐                 │
  │  │ model.predict(x_scaled)                          │                 │
  │  │ (N, seq, N_in) → (N, seq, 2)  em log10          │                 │
  │  │ Canal 0: log10(rho_h)                            │                 │
  │  │ Canal 1: log10(rho_v)                            │                 │
  │  └──────────────────────┬───────────────────────────┘                 │
  │                         │                                             │
  │                         ▼                                             │
  │  ETAPA 7: Inversão de Escala                                         │
  │  ┌──────────────────────────────────────────────────┐                 │
  │  │ rho = 10^y_pred                                  │                 │
  │  │ (N, seq, 2) log10 → (N, seq, 2) Ohm.m           │                 │
  │  │ rho_h ∈ [0.1, 10000] Ohm.m (típico)             │                 │
  │  │ rho_v ∈ [0.1, 10000] Ohm.m (típico)             │                 │
  │  └──────────────────────┴───────────────────────────┘                 │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘
```

### 4.1. Etapa 1 — Extração de Features

A primeira etapa seleciona as 5 colunas relevantes do array de 22 colunas:

```python
# ── Extração de features do array bruto ────────────────────
# Seleciona as 5 colunas definidas em INPUT_FEATURES.
# z_obs (col 1): profundidade observada — ancora espacial do modelo.
# Re(Hxx), Im(Hxx) (cols 4,5): componente planar do tensor EM.
# Re(Hzz), Im(Hzz) (cols 20,21): componente axial do tensor EM.
# Essas 5 features capturam a resposta EM primária de um sensor
# triaxial operando em formação anisotrópica (TIV).
features = raw_data[:, :, INPUT_FEATURES]  # (N, seq, 5)
```

### 4.2. Etapa 2 — Injeção de theta/freq

Nos pipelines multi-ângulo (P2, P3), o ângulo de mergulho da ferramenta
(`theta`) e a frequência de operação (`freq`) são injetados como colunas
adicionais:

```python
# ── Injeção de parâmetros instrumentais (P2/P3 apenas) ────
# theta: ângulo de mergulho em graus (0° = horizontal).
#   Varia de 0° a 90° conforme a trajetória do poço.
#   Influencia diretamente o acoplamento EM com as camadas.
# freq: frequência de operação em Hz (default 20 kHz).
#   Determina a profundidade de investigação do sensor EM.
if config.inject_theta_freq:
    theta_col = np.full((N, seq, 1), theta)
    freq_col = np.full((N, seq, 1), freq)
    features = np.concatenate([theta_col, freq_col, features], axis=-1)
```

### 4.3. Etapa 3 — Feature View (FV)

A Feature View transforma as componentes EM brutas em representações mais
adequadas para o aprendizado. O projeto implementa **7 modos de FV**:

| Modo FV           | Transformação                                      | Motivação Física                                  |
|:------------------|:---------------------------------------------------|:--------------------------------------------------|
| `identity`        | Sem transformação                                  | Baseline — dados como medidos                     |
| `H1_logH2`        | H1 cru, log10\|H2\|, fase(H2)                     | Comprime faixa dinâmica de Hzz (4+ ordens)        |
| `logH1_logH2`     | log10\|H1\|, fase(H1), log10\|H2\|, fase(H2)      | Ambas componentes comprimidas                     |
| `amplitude_phase`  | \|H1\|, fase(H1), \|H2\|, fase(H2)                | Separa magnitude e fase explicitamente            |
| `log_amp_phase`   | log10\|H1\|, fase(H1), log10\|H2\|, fase(H2)      | Log-amplitude + fase                              |
| `normalized`      | H/\|H_max\| por amostra                           | Remove escala absoluta                            |
| `attenuation`     | Atenuação + defasagem entre Hxx e Hzz              | Sinais de perfilagem comercial (AT, PS)           |

```python
# ── Aplicação do Feature View ──────────────────────────────
# O FV é aplicado para transformar componentes EM brutas (A/m)
# em representações que facilitem o aprendizado pelo modelo.
# A escolha do FV depende das características do dataset:
#   - identity: simples, bom para dados bem condicionados
#   - H1_logH2: recomendado para Hzz com alta variância
#   - attenuation: emula sinais comerciais (Schlumberger, Baker)
features_fv = apply_feature_view(features, config.feature_view)
```

### 4.4. Etapa 4 — Geosignals (GS)

Os Geosignals são sinais derivados das componentes EM que realçam contrastes de
resistividade em fronteiras de camada. A família **USD** (Up-Symmetrized
Directional) é sensível à direção da fronteira, enquanto **UHR** (Ultra-High
Resolution) oferece resolução vertical superior.

```python
# ── Cálculo dos Geosignals ────────────────────────────────
# GS derivam sinais fisicamente interpretáveis a partir do
# tensor EM, amplificando contrastes em interfaces de camada.
# Adicionam tipicamente +4 canais ao vetor de features:
#   USD_Re, USD_Im: sensíveis à assimetria do campo EM
#   UHR_Re, UHR_Im: alta resolução vertical (~0.5m)
# Os GS veem o ruído presente nos dados (fidelidade LWD).
features_gs = compute_geosignals(features_fv, config.geosignal_mode)
```

### 4.5. Etapa 5 — Normalização

A normalização utiliza os parâmetros (média, desvio padrão) fitados em dados
**limpos** durante o treinamento. Esse design é intencional: o scaler não deve
ser contaminado pelo ruído.

```python
# ── Normalização com scaler fitado em dados limpos ────────
# O scaler (StandardScaler) foi fitado em dados SEM ruído
# durante o treinamento. Mesmo que os dados de inferência
# contenham ruído real de campo, a normalização preserva
# a mesma transformação linear: x_scaled = (x - μ_clean) / σ_clean.
# Isso garante consistência entre treino e inferência.
x_scaled = (features_gs - scaler_params["mean"]) / scaler_params["std"]
```

### 4.6. Etapa 6 — Predição do Modelo

O modelo Keras executa a predição sobre os dados normalizados. A saída está no
domínio log10:

```python
# ── Predição pelo modelo treinado ─────────────────────────
# O modelo recebe (N, seq, N_features) normalizado e retorna
# (N, seq, 2) no domínio log10:
#   Canal 0: log10(rho_h) — resistividade horizontal
#   Canal 1: log10(rho_v) — resistividade vertical
# O padding "same" preserva seq_len na saída (modo offline).
y_pred_log10 = model.predict(x_scaled, batch_size=config.batch_size)
```

### 4.7. Etapa 7 — Inversão de Escala

A última etapa converte o domínio log10 de volta para Ohm.m:

```python
# ── Inversão de escala: log10 → Ohm.m ─────────────────────
# A operação 10^y inverte a transformação log10 aplicada aos
# targets durante o treinamento. O resultado é resistividade
# em unidades lineares (Ohm.m), diretamente comparável com
# perfis de resistividade convencionais (LLD, LLS, MSFL).
# Faixa típica: 0.1–10000 Ohm.m para formações sedimentares.
rho_pred = np.power(10.0, y_pred_log10)  # (N, seq, 2) em Ohm.m
```

---

## 5. Quantificação de Incerteza (UQ)

A quantificação de incerteza é essencial em aplicações de geosteering, onde
decisões de perfuração dependem da confiabilidade das predições. O módulo
`inference/uncertainty.py` implementa três métodos complementares, encapsulados
na classe `UncertaintyEstimator`.

### 5.1. Métodos Disponíveis

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │             MÉTODOS DE QUANTIFICAÇÃO DE INCERTEZA                    │
  ├───────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
  │  │   MC Dropout     │  │    Ensemble     │  │      INN        │      │
  │  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤      │
  │  │ 1 modelo         │  │ K modelos       │  │ 1 modelo         │      │
  │  │ N forward passes │  │ 1 pass cada     │  │ perturbação      │      │
  │  │ training=True    │  │ desacordo=UQ    │  │ sensibilidade    │      │
  │  │ mean ± std       │  │ mean ± std      │  │ propagação σ     │      │
  │  │ Bayesiano aprox. │  │ frequentista    │  │ 10× mais rápido  │      │
  │  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘
```

### 5.2. MC Dropout — Aproximação Bayesiana

O método MC Dropout (Gal & Ghahramani, 2016) executa múltiplas passagens forward
com dropout ativo (`training=True`), gerando uma distribuição empírica das
predições:

```python
# ── MC Dropout: N passagens com dropout ativo ──────────────
# Cada passagem forward desativa neurônios aleatoriamente,
# simulando amostragem de uma distribuição posterior sobre
# os pesos da rede (aproximação Bayesiana variacional).
# A variância entre passagens indica a incerteza epistêmica.
predictions = []
for i in range(mc_samples):
    y_i = model(x_scaled, training=True)  # dropout ativo
    predictions.append(y_i)

# Stack e estatísticas no domínio log10
preds = np.stack(predictions, axis=0)      # (mc_samples, N, seq, 2)
mean_log10 = np.mean(preds, axis=0)        # (N, seq, 2)
std_log10  = np.std(preds, axis=0)         # (N, seq, 2)

# Intervalo de confiança 95% (no domínio log10)
ci_lower = mean_log10 - 1.96 * std_log10
ci_upper = mean_log10 + 1.96 * std_log10
```

| Parâmetro     | Valor Típico | Descrição                                    |
|:-------------|:-------------|:---------------------------------------------|
| `mc_samples`  | 30           | Número de passagens Monte Carlo              |
| `training`    | `True`       | Ativa dropout durante inferência             |
| `CI`          | 95%          | Intervalo: mean ± 1.96 × std                |

### 5.3. Ensemble — Desacordo entre Modelos

O método Ensemble (Lakshminarayanan et al., 2017) treina K modelos
independentemente (seeds diferentes, possivelmente arquiteturas diferentes) e
usa o desacordo entre suas predições como medida de incerteza:

```python
# ── Ensemble: K modelos independentes ──────────────────────
# Cada modelo foi treinado com inicialização diferente,
# potencialmente com shuffles distintos dos dados.
# O desacordo entre modelos captura tanto incerteza epistêmica
# quanto variância na superfície de otimização.
ensemble_preds = []
for model_k in ensemble_models:
    y_k = model_k.predict(x_scaled)
    ensemble_preds.append(y_k)

preds = np.stack(ensemble_preds, axis=0)   # (K, N, seq, 2)
mean_log10 = np.mean(preds, axis=0)
std_log10  = np.std(preds, axis=0)
```

| Parâmetro   | Valor Típico | Descrição                                  |
|:-----------|:-------------|:-------------------------------------------|
| `K`         | 5            | Número de modelos no ensemble              |
| Custo       | K × treino   | Treinar K modelos independentes            |
| Diversidade | Seeds, LR    | Fontes de diversidade entre modelos        |

### 5.4. INN — Perturbação de Entrada

O método INN (*Input Noise Injection*, inspirado em Ardizzone et al., 2019)
perturba a entrada com ruído gaussiano de amplitude controlada e mede a
sensibilidade da saída à perturbação:

```python
# ── INN: Perturbação de entrada (σ=0.1) ───────────────────
# Injeta ruído gaussiano na entrada normalizada e mede
# a variação na saída. Regiões onde o modelo é sensível
# a pequenas perturbações indicam alta incerteza.
# Vantagem: 10× mais rápido que MC Dropout (sem dropout).
perturbed_preds = []
for i in range(n_perturbations):
    noise = np.random.normal(0, sigma, x_scaled.shape)
    y_i = model.predict(x_scaled + noise)
    perturbed_preds.append(y_i)

preds = np.stack(perturbed_preds, axis=0)
mean_log10 = np.mean(preds, axis=0)
std_log10  = np.std(preds, axis=0)
```

| Parâmetro         | Valor Típico | Descrição                                 |
|:-----------------|:-------------|:------------------------------------------|
| `sigma`           | 0.1          | Amplitude da perturbação gaussiana        |
| `n_perturbations` | 30           | Número de perturbações                    |
| Velocidade        | ~10× MC      | Sem overhead de dropout ativo             |

### 5.5. UncertaintyResult — Dataclass de Saída

```python
@dataclass
class UncertaintyResult:
    """Resultado da quantificação de incerteza.

    Todos os campos de predição estão em Ohm.m (após inversão log10).
    Os intervalos de confiança são calculados no domínio log10 e depois
    convertidos para Ohm.m via 10^x.

    Attributes:
        mean: Predição média, shape (N, seq_len, 2). Ohm.m.
        std: Desvio padrão das predições, shape (N, seq_len, 2). Ohm.m.
        ci_lower: Limite inferior do IC 95%, shape (N, seq_len, 2). Ohm.m.
        ci_upper: Limite superior do IC 95%, shape (N, seq_len, 2). Ohm.m.
        method: Método utilizado ("mc_dropout", "ensemble" ou "inn").
        n_samples: Número de amostras/passagens utilizadas.

    Example:
        >>> result = pipeline.predict(data, return_uncertainty=True)
        >>> print(f"rho_h = {result.mean[0,:,0]:.2f} ± {result.std[0,:,0]:.2f}")
        >>> print(f"IC 95%: [{result.ci_lower[0,:,0]:.2f}, {result.ci_upper[0,:,0]:.2f}]")
    """
    mean: np.ndarray
    std: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    method: str
    n_samples: int
```

### 5.6. Comparação entre Métodos

| Critério               | MC Dropout          | Ensemble            | INN                 |
|:----------------------|:---------------------|:--------------------|:--------------------|
| **Modelos necessários** | 1                   | K (tipicamente 5)   | 1                   |
| **Custo de treino**     | Normal               | K × normal          | Normal              |
| **Custo de inferência** | N × forward pass    | K × forward pass    | N × forward pass    |
| **Tipo de incerteza**   | Epistêmica (aprox.) | Epistêmica + aleatória | Sensibilidade     |
| **Fundamentação**       | Bayesiano variacional| Frequentista       | Análise perturbativa|
| **Velocidade relativa** | 1×                   | ~0.5× (K modelos)  | ~10× (sem dropout) |
| **Calibração**          | Boa                  | Excelente           | Razoável            |

---

## 6. Exportação de Modelos

O módulo `inference/export.py` permite exportar modelos treinados para diferentes
formatos, adequados a cenários de implantação (*deployment*) variados.

### 6.1. Formatos Suportados

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │                    FORMATOS DE EXPORTAÇÃO                            │
  ├───────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │  ┌─────────────────┐                                                  │
  │  │   SavedModel    │  TF Serving, produção, API REST/gRPC            │
  │  │   (~50–200 MB)  │  Grafo completo + variáveis + assinaturas       │
  │  └─────────────────┘                                                  │
  │                                                                       │
  │  ┌─────────────────┐                                                  │
  │  │     TFLite      │  Dispositivos edge, microcontroladores          │
  │  │   (~5–50 MB)    │  float32 ou int8 quantizado                     │
  │  └─────────────────┘                                                  │
  │                                                                       │
  │  ┌─────────────────┐                                                  │
  │  │      ONNX       │  Interoperabilidade (PyTorch, C#, Java)         │
  │  │   (~10–100 MB)  │  opset=15, cross-platform                      │
  │  └─────────────────┘                                                  │
  │                                                                       │
  │  ┌─────────────────┐                                                  │
  │  │   model.keras   │  Formato nativo Keras — treino + inferência     │
  │  │   (~10–100 MB)  │  Serialização padrão do pipeline                │
  │  └─────────────────┘                                                  │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘
```

### 6.2. Tabela Comparativa

| Formato       | Tamanho Típico | Caso de Uso                  | Quantização   | Plataforma           |
|:-------------|:---------------|:-----------------------------|:-------------|:---------------------|
| **SavedModel** | 50–200 MB     | TF Serving, produção         | Não (float32) | TensorFlow           |
| **TFLite**     | 5–50 MB       | Edge, mobile, embarcado      | float32/int8  | Android, IoT, Edge   |
| **ONNX**       | 10–100 MB     | Cross-platform, C++/C#       | float32       | ONNX Runtime, todos  |
| **model.keras**| 10–100 MB     | Desenvolvimento, Colab       | Não (float32) | TensorFlow/Keras     |

### 6.3. Exemplos de Exportação

```python
from geosteering_ai.inference.export import export_savedmodel, export_tflite, export_onnx

# ── SavedModel para TF Serving ────────────────────────────
# Exporta grafo completo com assinaturas para inferência REST/gRPC.
# Inclui metadados de versão para versionamento no servidor.
export_savedmodel(model, "exports/savedmodel/1/")

# ── TFLite com quantização int8 ───────────────────────────
# Reduz tamanho ~4× e acelera inferência em hardware edge.
# Requer dataset representativo para calibração da quantização.
export_tflite(
    model,
    "exports/model_int8.tflite",
    quantize="int8",
    representative_dataset=calibration_data,
)

# ── ONNX para interoperabilidade ──────────────────────────
# opset=15 garante compatibilidade com ONNX Runtime 1.12+.
# Permite inferência em ambientes sem TensorFlow instalado.
export_onnx(model, "exports/model.onnx", opset=15)
```

---

## 7. Serialização e Reprodutibilidade

### 7.1. Artefatos de Serialização

O método `save()` da `InferencePipeline` cria exatamente **3 artefatos** que,
juntos, permitem reproduzir a inferência de forma idêntica:

```
  artifacts/run_042/
  ├── model.keras        ← Modelo Keras (arquitetura + pesos)
  ├── scalers.joblib     ← Parâmetros do scaler (mean, std)
  └── config.yaml        ← Snapshot completo da PipelineConfig
```

| Artefato          | Conteúdo                                          | Formato            |
|:-----------------|:--------------------------------------------------|:-------------------|
| `model.keras`     | Arquitetura do modelo + pesos treinados           | Keras nativo       |
| `scalers.joblib`  | Média e desvio padrão fitados em dados limpos     | joblib (sklearn)   |
| `config.yaml`     | Todos os 246 campos da PipelineConfig             | YAML legível       |

### 7.2. Garantias de Reprodutibilidade

A combinação dos 3 artefatos garante reprodutibilidade completa:

1. **`config.yaml`** registra todos os hiperparâmetros, incluindo `feature_view`,
   `geosignal_mode`, `input_features`, `output_targets`, `target_scaling` e
   `sequence_length`. Qualquer alteração em qualquer campo é detectada.

2. **`scalers.joblib`** preserva os parâmetros de normalização exatamente como
   foram calculados durante o treinamento. O scaler **nunca** é refitado em
   dados de inferência.

3. **`model.keras`** contém a arquitetura exata (incluindo camadas customizadas)
   e os pesos treinados. A combinação de seed + config + dados de treino
   determina univocamente os pesos.

### 7.3. Fluxo de Serialização

```
  TREINAMENTO                          INFERÊNCIA
  ──────────                           ──────────
  config → fit scaler                  load config.yaml
         → treinar modelo              load scalers.joblib
         → save()                      load model.keras
           │                              │
           ├── model.keras ──────────────→┤
           ├── scalers.joblib ───────────→┤
           └── config.yaml ──────────────→┤
                                          │
                                          ▼
                                   InferencePipeline
                                   pronto para predict()
```

### 7.4. Versionamento

O projeto utiliza **tags GitHub** para versionar conjuntos de artefatos:

```bash
# Criar tag associada ao treinamento
git tag -a v2.0-run042-resnet18 -m "ResNet-18, robusto preset, 100 epochs"
git push origin v2.0-run042-resnet18

# Instalar versão específica no Colab
pip install git+https://github.com/daniel-leal/geosteering-ai@v2.0-run042-resnet18
```

---

## 8. Tutorial Rápido

### 8.1. Exemplo 1 — Predição Offline Básica

```python
import numpy as np
from geosteering_ai.config import PipelineConfig
from geosteering_ai.inference.pipeline import InferencePipeline

# ── Carregar pipeline salvo ────────────────────────────────
# O diretório contém model.keras, scalers.joblib e config.yaml
# gerados ao final do treinamento via pipeline.save().
pipeline = InferencePipeline.load("artifacts/run_042/")

# ── Carregar dados brutos ─────────────────────────────────
# Dados do simulador Fortran: shape (N_modelos, seq_len, 22)
# Cada modelo geológico tem seq_len=600 pontos de medição
# com 22 colunas (profundidade, componentes EM, targets).
raw_data = np.load("data/test_models.npy")
print(f"Shape dos dados: {raw_data.shape}")
# → Shape dos dados: (50, 600, 22)

# ── Executar predição ─────────────────────────────────────
# Retorna resistividade em Ohm.m, shape (50, 600, 2).
# Canal 0: rho_h (horizontal), Canal 1: rho_v (vertical).
rho_pred = pipeline.predict(raw_data)
print(f"Shape da predição: {rho_pred.shape}")
# → Shape da predição: (50, 600, 2)

# ── Verificar faixa de valores ────────────────────────────
print(f"rho_h: min={rho_pred[:,:,0].min():.2f}, max={rho_pred[:,:,0].max():.2f} Ohm.m")
print(f"rho_v: min={rho_pred[:,:,1].min():.2f}, max={rho_pred[:,:,1].max():.2f} Ohm.m")
# → rho_h: min=0.35, max=512.78 Ohm.m
# → rho_v: min=0.42, max=1023.45 Ohm.m
```

### 8.2. Exemplo 2 — Predição com Incerteza (MC Dropout)

```python
# ── Predição com quantificação de incerteza ───────────────
# MC Dropout: 30 passagens forward com dropout ativo.
# Retorna UncertaintyResult com mean, std, IC 95%.
result = pipeline.predict(
    raw_data,
    return_uncertainty=True,
    mc_samples=30,
)

# ── Acessar resultados ────────────────────────────────────
print(f"Método: {result.method}")
# → Método: mc_dropout

print(f"Amostras MC: {result.n_samples}")
# → Amostras MC: 30

# ── Predição média e incerteza para o primeiro modelo ─────
rho_h_mean = result.mean[0, :, 0]     # (600,) Ohm.m
rho_h_std  = result.std[0, :, 0]      # (600,) Ohm.m
rho_h_lo   = result.ci_lower[0, :, 0] # (600,) Ohm.m — IC 95% inferior
rho_h_hi   = result.ci_upper[0, :, 0] # (600,) Ohm.m — IC 95% superior

# ── Visualizar com banda de incerteza ─────────────────────
import matplotlib.pyplot as plt

depth = raw_data[0, :, 1]  # z_obs (coluna 1)

fig, ax = plt.subplots(figsize=(4, 12))
ax.fill_betweenx(depth, rho_h_lo, rho_h_hi, alpha=0.3, label="IC 95%")
ax.plot(rho_h_mean, depth, "b-", linewidth=1.5, label="Predição (média)")
ax.set_xlabel("rho_h (Ohm.m)")
ax.set_ylabel("Profundidade (m)")
ax.set_xscale("log")
ax.invert_yaxis()
ax.legend()
ax.set_title("Resistividade Horizontal — MC Dropout (30 amostras)")
plt.tight_layout()
plt.show()
```

### 8.3. Exemplo 3 — Exportação para TFLite (Edge)

```python
from geosteering_ai.inference.export import export_tflite

# ── Exportar modelo para TFLite ───────────────────────────
# Formato otimizado para dispositivos edge e embarcados.
# float32 preserva precisão total; int8 reduz tamanho ~4×.

# Opção A: float32 (máxima precisão)
export_tflite(
    pipeline.model,
    "exports/geosteering_float32.tflite",
    quantize=None,
)

# Opção B: int8 quantizado (mínimo tamanho)
# Requer dados representativos para calibração.
calibration_data = raw_data[:10]  # 10 amostras para calibração

export_tflite(
    pipeline.model,
    "exports/geosteering_int8.tflite",
    quantize="int8",
    representative_dataset=calibration_data,
)

# ── Verificar tamanhos ────────────────────────────────────
import os
size_f32 = os.path.getsize("exports/geosteering_float32.tflite") / 1e6
size_i8  = os.path.getsize("exports/geosteering_int8.tflite") / 1e6
print(f"float32: {size_f32:.1f} MB")
print(f"int8:    {size_i8:.1f} MB")
# → float32: 12.3 MB
# → int8:    3.1 MB
```

---

## 9. Melhorias Futuras

### 9.1. Paralelismo em Predição Batch

Atualmente, o `model.predict()` processa o batch inteiro de uma vez. Para
arquivos muito grandes (milhares de modelos geológicos), implementar predição
em chunks com paralelismo via `tf.data.Dataset`:

```python
# Futuro: predição em chunks paralelos
dataset = tf.data.Dataset.from_tensor_slices(x_scaled)
dataset = dataset.batch(256).prefetch(tf.data.AUTOTUNE)
predictions = model.predict(dataset)
```

### 9.2. Otimização com ONNX Runtime

Para inferência em produção sem TensorFlow, migrar para ONNX Runtime com
otimizações de grafo:

```python
# Futuro: inferência via ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
result = session.run(None, {"input": x_scaled})
```

### 9.3. Pipeline Multi-Modelo (Ensemble Automatizado)

Implementar orquestração automática de K modelos para ensemble inference,
com agregação configurável (média, mediana, votação ponderada):

```python
# Futuro: ensemble automatizado
ensemble = EnsemblePipeline.load("artifacts/ensemble_K5/")
result = ensemble.predict(raw_data, aggregation="weighted_mean")
```

### 9.4. Streaming Offline para Arquivos Grandes

Para perfis extremamente longos (seq_len > 100000), implementar processamento
em janelas deslizantes com overlap e *stitching*:

```python
# Futuro: streaming offline com overlap
result = pipeline.predict_streaming(
    raw_data,
    window_size=10000,
    overlap=500,
    stitching="cosine_blend",
)
```

### 9.5. Cache de Predições

Implementar cache persistente baseado em hash dos dados de entrada + versão do
modelo, evitando reprocessamento de dados já analisados:

```python
# Futuro: cache de predições
pipeline = InferencePipeline.load("artifacts/", cache_dir=".cache/predictions/")
rho = pipeline.predict(raw_data)  # Calcula e salva no cache
rho = pipeline.predict(raw_data)  # Lê do cache (instantâneo)
```

---

## 10. Referências

### 10.1. Quantificação de Incerteza

1. **Gal, Y. & Ghahramani, Z.** (2016). "Dropout as a Bayesian Approximation:
   Representing Model Uncertainty in Deep Learning." *Proceedings of the 33rd
   International Conference on Machine Learning (ICML)*, pp. 1050–1059.
   — Fundamentação teórica do MC Dropout: dropout durante inferência aproxima
   inferência Bayesiana variacional, permitindo estimar incerteza epistêmica
   sem modificar a arquitetura do modelo.

2. **Lakshminarayanan, B., Pritzel, A. & Blundell, C.** (2017). "Simple and
   Scalable Predictive Uncertainty Estimation using Deep Ensembles." *Advances
   in Neural Information Processing Systems (NeurIPS)*, vol. 30.
   — Deep Ensembles como alternativa frequentista ao MC Dropout: treinar K
   modelos com inicializações diferentes captura diversidade na superfície
   de perda, produzindo estimativas de incerteza bem calibradas.

3. **Ardizzone, L., Lüth, C., Kruse, J., Rother, C. & Köthe, U.** (2019).
   "Guided Image Generation with Conditional Invertible Neural Networks."
   *arXiv:1907.02392*.
   — Redes Neurais Invertíveis (INN) para amostragem eficiente da distribuição
   posterior. A invertibilidade permite estimar incerteza via perturbação de
   entrada, sem múltiplas passagens forward com dropout.

### 10.2. Deep Learning para Geofísica

4. **Shahriari, M., Pardo, D., Picón, A., Galdran, A., Del Ser, J. &
   Torres-Verdín, C.** (2020). "A deep learning approach to the inversion of
   borehole resistivity measurements." *Computational Geosciences*, 24,
   pp. 971–994.
   — Inversão de perfis de resistividade via redes neurais profundas,
   demonstrando viabilidade de substituir métodos iterativos tradicionais
   por modelos data-driven para LWD.

5. **Alyaev, S., Suter, E., Bratvold, R.B., et al.** (2021). "A decision
   support system for multi-target geosteering." *Journal of Petroleum
   Science and Engineering*, 199, 108172.
   — Framework de apoio à decisão para geosteering, integrando inversão de
   resistividade em tempo real com otimização de trajetória de poço.

### 10.3. Arquiteturas de Redes Neurais

6. **He, K., Zhang, X., Ren, S. & Sun, J.** (2016). "Deep Residual Learning
   for Image Recognition." *Proceedings of the IEEE Conference on Computer
   Vision and Pattern Recognition (CVPR)*, pp. 770–778.
   — Skip connections em redes residuais, permitindo treinar redes profundas
   (18–152 camadas) sem degradação de gradientes. Base da família ResNet
   implementada no Geosteering AI.

7. **van den Oord, A., Dieleman, S., Zen, H., et al.** (2016). "WaveNet:
   A Generative Model for Raw Audio." *arXiv:1609.03499*.
   — Convoluções causais dilatadas para processamento sequencial, com campo
   receptivo exponencialmente crescente. Adaptada para inferência realtime
   no Geosteering AI (modo causal).

---

> **Documento gerado para o projeto Geosteering AI v2.0**
> Inversão 1D de Resistividade Eletromagnética via Deep Learning
> Autor: Daniel Leal | Framework: TensorFlow 2.x / Keras
