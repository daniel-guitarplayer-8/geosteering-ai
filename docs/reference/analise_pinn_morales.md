# Analise Comparativa: PINN do Pipeline v5.0.15 vs. Morales et al. (2025)

**Documento:** Analise tecnica comparativa entre a implementacao PINN do Pipeline de
Inversao Geofisica v5.0.15 e a PINN descrita no artigo "Anisotropic resistivity
estimation and uncertainty quantification from borehole triaxial electromagnetic
induction measurements: Gradient-based inversion and physics-informed neural network"
(Morales, Dos Santos & Guevara, 2025).

**Data:** 2026-03-13
**Autor:** Daniel Leal (com assistencia de Claude)
**Versao do Pipeline:** v5.0.15
**Referencia do Artigo:** Morales et al., Geophysical Journal International, 2025

---

## Sumario Executivo

Este documento apresenta uma analise exaustiva das diferencas fundamentais entre
a implementacao PINN (Physics-Informed Neural Network) do nosso Pipeline de Inversao
Geofisica v5.0.15 e a PINN proposta por Morales, Dos Santos & Guevara (2025).

A analise revela que, embora ambos os sistemas utilizem redes neurais informadas
por fisica para problemas de inversao em geofisica de poco, eles resolvem
**problemas complementares na cadeia de interpretacao** e adotam filosofias
arquiteturais fundamentalmente diferentes.

Foram identificados **7 insights incorporaveis** ao nosso pipeline, classificados
por prioridade (alta, media, estrategica), com impacto potencial nas celulas
C7, C10, C13, C27-C36A, C40, C48-C57 e C66-C73.

---

## Indice

1. [Contexto e Motivacao](#1-contexto-e-motivacao)
2. [Dimensao 1: Problema Fisico Resolvido](#2-dimensao-1-problema-fisico-resolvido)
3. [Dimensao 2: Tipo de Aprendizado](#3-dimensao-2-tipo-de-aprendizado)
4. [Dimensao 3: Formulacao da Loss e Peso da Fisica](#4-dimensao-3-formulacao-da-loss-e-peso-da-fisica)
5. [Dimensao 4: Arquitetura da Rede](#5-dimensao-4-arquitetura-da-rede)
6. [Dimensao 5: Constraint de Saida](#6-dimensao-5-constraint-de-saida)
7. [Dimensao 6: Escala dos Dados de Treinamento](#7-dimensao-6-escala-dos-dados-de-treinamento)
8. [Dimensao 7: Quantificacao de Incerteza (UQ)](#8-dimensao-7-quantificacao-de-incerteza-uq)
9. [Dimensao 8: Contexto Espacial](#9-dimensao-8-contexto-espacial)
10. [Dimensao 9: Modelo Fisico Forward](#10-dimensao-9-modelo-fisico-forward)
11. [Dimensao 10: Otimizador e Hiperparametros](#11-dimensao-10-otimizador-e-hiperparametros)
12. [Tabela Resumo Comparativa](#12-tabela-resumo-comparativa)
13. [Insights Incorporaveis — Analise Detalhada](#13-insights-incorporaveis--analise-detalhada)
14. [Cadeia de Interpretacao Integrada](#14-cadeia-de-interpretacao-integrada)
15. [Plano de Acoes Recomendadas](#15-plano-de-acoes-recomendadas)
16. [Conclusoes](#16-conclusoes)
17. [Referencias](#17-referencias)

---

## 1. Contexto e Motivacao

### 1.1 Nosso Pipeline (v5.0.15)

O Pipeline de Inversao Geofisica com Deep Learning v5.0.15 e um sistema completo
de inversao 1D de resistividade a partir de medidas eletromagneticas (EM) triaxiais
em ambiente de perfuracao direcional (LWD — Logging While Drilling). O pipeline:

- Recebe componentes complexas do tensor EM (Re/Im de Hxx, Hzz) geradas pelo
  simulador Fortran `PerfilaAnisoOmp.f08`
- Prediz perfis de resistividade horizontal (rho_h) e vertical (rho_v) ao longo
  da profundidade
- Suporta 44 arquiteturas seq2seq, 25 funcoes de perda, 43 tipos de ruido
- Possui modulo PINN opcional (USE_PINNS=False por padrao) com 3 cenarios:
  "oracle", "surrogate", "maxwell"
- Configurado em C13 (7 FLAGS) com schedule de lambda (warmup + ramp)

### 1.2 Morales et al. (2025)

O artigo propoe uma PINN para inversao petrofisica: dado os perfis de resistividade
horizontal (Rh) e vertical (Rv) ja estimados, determinar a concentracao volumetrica
de folhelho (Csh) e a resistividade do arenito limpo (Rss) utilizando as equacoes
de Klein (1993) para meios transversalmente isotropicos (TI).

A PINN de Morales opera em um estagio **posterior** ao nosso pipeline na cadeia
de interpretacao petrofisica.

---

## 2. Dimensao 1: Problema Fisico Resolvido

Esta e a diferenca mais fundamental e estrutural entre os dois sistemas.

### Nosso Pipeline

```
Problema: Inversao Geofisica
H_tensor(theta, f, L) --> [DL] --> rho_h(z), rho_v(z)

Entrada: componentes complexas do tensor EM (Re/Im de Hxx, Hzz)
         para multiplos angulos theta, frequencias f, espacamentos L
Saida:   perfis de resistividade rho_h e rho_v ao longo de z
Fisica:  Equacoes de Maxwell em meio TIV estratificado
         (Fortran PerfilaAnisoOmp.f08 -- solver numerico completo)
Complexidade: Altissima -- PDE em 3D + anisotropia + multi-frequencia
```

### Morales et al.

```
Problema: Inversao Petrofisica
[Rv(z), Rh(z)] --> [PINN] --> Csh(z), Rss(z)

Entrada: perfis de resistividade horizontal e vertical
Saida:   concentracao volumetrica de folhelho (Csh) e
         resistividade do arenito (Rss)
Fisica:  Equacoes de Klein (1993) -- 2 equacoes algebricas TI
         Rv = Csh*Rsh_v + (1-Csh)*Rss           (circuito serie)
         1/Rh = Csh/Rsh_h + (1-Csh)/Rss         (circuito paralelo)
Complexidade: Baixa -- sistema de 2 equacoes analiticas diferenciaveis
```

### Consequencia Critica

As equacoes de Klein sao diretamente diferenciaveis e podem ser embutidas como
camada Keras sem nenhum solver externo. As equacoes de Maxwell do nosso problema
requerem o simulador Fortran — que **nao e diferenciavel** — tornando a
incorporacao da fisica muito mais complexa.

Isso explica por que:
- Morales pode usar omega=0.85 (fisica dominante) sem instabilidade
- Nosso pipeline usa PINNS_LAMBDA=0.01 (conservador) com warmup+ramp
- Morales nao precisa de labels diretos (a fisica e suficiente)
- Nosso pipeline depende de labels supervisionados como sinal principal

---

## 3. Dimensao 2: Tipo de Aprendizado

Esta e a diferenca conceitual mais profunda entre os dois sistemas.

### Comparativo

| Aspecto                   | Nosso Pipeline                          | Morales et al.                            |
|:--------------------------|:----------------------------------------|:------------------------------------------|
| **Labels diretos?**       | **SIM** — rho_h, rho_v do simulador     | **NAO** — Csh e Rss nunca sao fornecidos  |
| **Como aprende?**         | Minimiza erro vs labels + reg. fisica   | Aprende pela consistencia com Klein       |
| **Paradigma**             | **Supervisionado** + regularizacao      | **Physics-only** / auto-supervisionado    |
| **Sinal principal**       | Labels gerados pelo simulador Fortran   | Equacoes fisicas embutidas na loss        |
| **Sinal secundario**      | Termos fisicos na loss (1%)             | Data mismatch (15%)                       |

### Detalhamento

**Nosso pipeline:** A rede recebe pares (X, Y) onde X sao as medidas EM e Y sao
os perfis de resistividade correspondentes, gerados pelo simulador Fortran. O
treinamento e fundamentalmente supervisionado. O modulo PINN adiciona um termo
de regularizacao L_physics que corresponde a apenas ~1% da loss total (com
PINNS_LAMBDA=0.01). A rede poderia aprender razoavelmente bem mesmo sem PINNs.

**Morales:** A rede NUNCA "ve" os valores corretos de Csh e Rss durante o
treinamento. Ela aprende a estima-los exclusivamente porque, quando substitui
suas predicoes nas equacoes de Klein, os Rv_hat e Rh_hat calculados devem
coincidir com os Rv e Rh medidos. E um regime de aprendizado fundamentalmente
diferente onde a fisica NAO e regularizacao — e o **unico sinal de aprendizado
significativo** (85% da loss).

### Implicacao

O paradigma de Morales e mais elegante do ponto de vista fisico (aprende sem
labels), mas so e viavel porque as equacoes de Klein sao simples, analiticas
e diferenciaveis. Para o nosso problema (equacoes de Maxwell), replicar este
paradigma exigiria um surrogate neural diferenciavel do simulador Fortran —
o que a loss `encoder_decoder` (#21 no catalogo) ja preve parcialmente.

---

## 4. Dimensao 3: Formulacao da Loss e Peso da Fisica

### Nosso Pipeline

```
L_total = L_data + PINNS_LAMBDA x L_physics

PINNS_LAMBDA = 0.01 (default)

Distribuicao (defaults):
  L_data ..... ~99% da loss total   <-- DOMINANTE
  L_physics ..  ~1% da loss total   <-- regularizacao secundaria

Schedule:  lambda_eff = 0 (warmup 10 epocas)
           lambda_eff ramp linear (ramp 20 epocas)
           lambda_eff = PINNS_LAMBDA (pleno)

L_data = qualquer das 25 losses (ex: log_scale_aware)
       = w_mse*RMSE + alpha*Interface + beta*Oscillation + gamma*Underest.

L_physics depende do PINNS_SCENARIO:
  "oracle"    = ||H_pred - H_fortran||^2    (compara componentes EM)
  "surrogate" = ||H_pred - H_surrogate||^2  (neural surrogate do Fortran)
  "maxwell"   = ||nabla^2 E + k^2 E||^2     (Helmholtz via GradientTape)
```

### Morales et al.

```
L_total = omega x L_physics + (1-omega) x L_data

omega = 0.85 (fixo, determinado empiricamente via grid search)

Distribuicao:
  L_physics ... 85% da loss total  <-- DOMINANTE
  L_data .....  15% da loss total  <-- complementar

L_physics = ||W_d * (x - x_hat(theta))||^2 + lambda*||theta||^2
          = MSE ponderada (W_d) + L2 regularizacao
            onde x_hat(theta) = forward_Klein(Csh_pred, Rss_pred)

L_data = ||x - x_hat(theta)||_1    (L1 data mismatch)

W_d = matriz de ponderacao adaptativa baseada no IGR (Indice Gamma Ray):
  W_d = diag([1/(Rv/IGR), 1/(Rh/IGR)])
  Normaliza a contribuicao de cada medida pela escala geologica

Sem schedule -- omega = 0.85 desde a epoca 1
```

### Insight Chave: Normas Diferentes

Morales usa **normas diferentes** para cada componente:
- **L2 (MSE)** para o termo de fisica — penaliza mais fortemente grandes desvios
  das equacoes fisicas
- **L1 (MAE)** para o mismatch de dados — robusta a outliers nas medidas

Esta combinacao intencional tem justificativa tecnica solida e poderia ser
adotada em nosso pipeline como uma nova variante de loss.

### Grid Search de omega

Morales testou omega em {0.3, 0.5, 0.7, 0.85, 0.95, 0.99} e encontrou omega=0.85
como otimo. Valores menores (omega < 0.7) degradavam a consistencia fisica;
valores maiores (omega > 0.95) subajustavam os dados. A relacao 85/15 entre
fisica e dados e um ponto de equilibrio empiricamente validado.

---

## 5. Dimensao 4: Arquitetura da Rede

### Nosso Pipeline

```
- 44 arquiteturas disponiveis (ResNet, CNN, LSTM, Transformer,
  U-Net, TCN, WaveNet, N-BEATS, Mamba/S4, FNO, etc.)
- Seq2Seq: (batch, N_MEDIDAS, N_FEATURES) -> (batch, N_MEDIDAS, OUT_CH)
  N_MEDIDAS = 600, N_FEATURES = 5-17, OUT_CH = 2-6
- Dezenas a centenas de milhoes de parametros
- Contexto espacial: a rede "ve" os 600 pontos de profundidade juntos
- Suporte a causalidade: padding='causal' no modo realtime
- Restricao de saida: SOFT (via termos de loss)
- Celulas: C27-C36A (blocos e arquiteturas), C37 (factory + registry)
```

### Morales et al.

```
- 1 arquitetura fixa: FC feed-forward simples
- Ponto-a-ponto: [Rv, Rh, Rsh_v, Rsh_h] -> [Csh, Rss]
  (4 escalares -> 2 escalares por profundidade)
- 23.402 parametros totais
- SEM contexto espacial: cada profundidade processada independentemente
- Estrutura:
    Input(4)
     -> Linear(4->150) -> BatchNorm -> tanh -> Dropout(0.1)
     -> Linear(150->150) -> BatchNorm -> tanh -> Dropout(0.1)
     -> Linear(150->2)
     -> CONSTRAINT LAYER: sigmoid(Csh) | ReLU(Rss)   <-- HARD CONSTRAINT
- Total: 31 min de treino (RTX 3080), 0.5 ms de inferencia
```

### Comparativo Detalhado

| Aspecto                | Nosso Pipeline                    | Morales et al.                    |
|:-----------------------|:----------------------------------|:----------------------------------|
| **N. de arquiteturas** | 44 (configuravel via MODEL_NAME)  | 1 (FC fixa)                       |
| **Tipo**               | Seq2Seq (sequencia completa)      | Ponto-a-ponto (depth-by-depth)    |
| **Parametros**         | ~1M a ~100M (depende da arq.)     | 23.402                             |
| **Input shape**        | (batch, 600, 5-17)                | (batch, 4)                         |
| **Output shape**       | (batch, 600, 2-6)                 | (batch, 2)                         |
| **Ativacao hidden**    | ReLU/GELU/SiLU (configuravel)     | tanh (fixo)                        |
| **Ativacao output**    | linear (sem constraint)           | sigmoid + ReLU (hard constraint)   |
| **Normalizacao**       | BatchNorm/LayerNorm (opcional)     | BatchNorm (fixo)                   |
| **Dropout**            | Configuravel (USE_DROPOUT, DROPOUT_RATE) | 0.1 (fixo)                 |
| **Contexto espacial**  | Sim (Conv1D, LSTM, Attention)     | Nao (ponto independente)           |
| **Causalidade**        | Suportada (padding='causal')      | N/A (ponto-a-ponto)               |

---

## 6. Dimensao 5: Constraint de Saida (Hard vs. Soft)

### Comparativo

| Aspecto              | Nosso Pipeline                              | Morales et al.                        |
|:---------------------|:--------------------------------------------|:--------------------------------------|
| **Tipo**             | Soft constraint                             | **Hard constraint**                   |
| **Como implementado**| `UnderestimationPenalty` na loss             | Camada sigmoid/ReLU na arquitetura    |
| **Garantia**         | Probabilistica — penaliza, nao impede       | **Absoluta** — impossivel violar      |
| **rho > 0**          | Penalizacao indireta via loss               | softplus/ReLU garante por construcao  |
| **Csh in [0,1]**     | N/A (nosso target e rho, nao Csh)           | sigmoid garante por construcao        |
| **Csh + (1-Csh)=1**  | N/A                                         | Garantido pelo sigmoid               |

### Detalhamento

**Nosso pipeline:** A camada final de TODAS as 44 arquiteturas e:
```python
TimeDistributed(Dense(OUTPUT_CHANNELS, activation='linear'))
```
Nao ha nenhuma restricao arquitetural que impeca a rede de produzir valores
negativos de resistividade. A `UnderestimationPenalty` em `log_scale_aware_loss`
penaliza subestimacao, mas nao impede fisicamente valores invalidos.

**Morales:** A camada final inclui restricoes hard:
```python
# Pseudocodigo da constraint layer de Morales
output_raw = Dense(2)(hidden)
Csh = sigmoid(output_raw[:, 0])   # Garante Csh in [0, 1]
Rss = relu(output_raw[:, 1])      # Garante Rss >= 0
```
Isso elimina por construcao qualquer predicao fisicamente invalida.

### Consequencia

Nosso pipeline pode, em principio, produzir predicoes de resistividade negativas
em regioes de treinamento dificeis (especialmente em escala log10 onde valores
muito negativos correspondem a resistividades sub-unitarias). Uma constraint
layer `softplus` (suave, diferenciavel) resolveria este problema.

---

## 7. Dimensao 6: Escala dos Dados de Treinamento

| Aspecto              | Nosso Pipeline                              | Morales et al.                        |
|:---------------------|:--------------------------------------------|:--------------------------------------|
| **Dados de treino**  | Milhares de modelos x 600 prof x N_ang x N_freq | 3.200 medidas (1 sint. + 1 campo) |
| **Origem dos dados** | Simulador Fortran (100% sintetico)          | Simulador 3D-UTAPWeLS + campo real    |
| **Diversidade geol.**| Alta (muitos modelos, variacao theta, f)    | Moderada (4 casos totais)             |
| **Dados de campo**   | Nao utilizados (apenas sinteticos)          | Usados no treinamento                 |
| **Split**            | Por modelo geologico [P1]                   | Hold-out 80/20 simples                |

### Implicacao

Morales demonstra que e possivel obter bons resultados com apenas 3.200 amostras
quando a fisica e fortemente embutida (omega=0.85). Isso sugere que, se
aumentarmos o peso do termo de fisica em nosso pipeline, poderiamos potencialmente
reduzir a quantidade de dados sinteticos necessarios — embora a complexidade
das equacoes de Maxwell torne isso muito mais desafiador.

A inclusao de dados de campo reais por Morales e um aspecto que nosso pipeline
ainda nao contempla e que poderia melhorar a generalizacao.

---

## 8. Dimensao 7: Quantificacao de Incerteza (UQ)

### Comparativo

| Aspecto              | Nosso Pipeline                              | Morales et al.                        |
|:---------------------|:--------------------------------------------|:--------------------------------------|
| **Metodo principal** | MC Dropout (USE_MC_DROPOUT_INFERENCE)       | Perturbacao Ensemble (N=1000)         |
| **Metodo secundario**| NLL loss (#18: probabilistic_nll)           | --                                    |
| **Tipo de incerteza**| Epistemica (MC) + Aletoria (NLL)            | Aletoria (variancia dos dados)        |
| **Implementacao**    | N forward passes com Dropout ativo          | N forward passes com input ruidoso    |
| **N de amostragens** | Configuravel (tipico: 50-100)               | 1.000 (fixo no artigo)                |
| **Ruido aplicado**   | Dropout nas camadas ocultas                 | kappa=5% na entrada (perturbacao)     |
| **Saida UQ**         | mu +/- sigma por profundidade               | Intervalo P10-P90 por profundidade    |
| **Tempo de UQ**      | N x tempo_inferencia                        | ~0.5 s total (1000 runs)             |
| **Implementado?**    | FLAGS definidas, impl. em C48-C57           | Totalmente validado em 4 datasets     |
| **Independe da arq?**| NAO — requer camadas Dropout                | **SIM** — funciona com qualquer rede  |

### Detalhamento do Metodo Ensemble (Morales)

O metodo de perturbacao ensemble de Morales e elegante pela sua simplicidade:

1. Dado um perfil de entrada [Rv, Rh, Rsh_v, Rsh_h]
2. Gerar N=1000 versoes perturbadas: x_perturb = x + kappa * std(x) * N(0,1)
3. Inferir Csh e Rss para cada versao perturbada
4. Calcular percentis P10 e P90 das 1000 predicoes
5. O intervalo P10-P90 e a estimativa de incerteza

**Vantagem principal:** Nao requer nenhuma modificacao na arquitetura da rede.
Funciona com qualquer modelo ja treinado. E complementar ao MC Dropout.

**Validacao no artigo:** Morales demonstra que o intervalo P10-P90 correlaciona
bem com o erro real de predicao — regioes de alta incerteza coincidem com
regioes de alto erro (ex: transicoes de camada).

---

## 9. Dimensao 8: Contexto Espacial

| Aspecto              | Nosso Pipeline                              | Morales et al.                        |
|:---------------------|:--------------------------------------------|:--------------------------------------|
| **Contexto**         | **Sequencia completa** — 600 pontos juntos  | **Ponto-a-ponto** — cada prof isolada |
| **Vantagem**         | Captura dependencias geologicas longas       | Extremamente rapido, escalavel        |
| **Desvantagem**      | Mais lento, requer sequencia completa        | Sem contexto de camadas vizinhas      |
| **Modo realtime**    | Sliding window causal                        | Inferencia por ponto (nativo)         |
| **Padding**          | 'same' (offline) ou 'causal' (realtime)     | N/A (sem convolucao)                  |

### Implicacao

O processamento ponto-a-ponto de Morales e naturalmente compativel com
inferencia em tempo real (0.5 ms por ponto). Nosso pipeline no modo realtime
requer sliding window com padding causal, o que adiciona complexidade.

Porem, nosso contexto espacial permite que a rede aprenda padroes geologicos
que transcendem um unico ponto de profundidade — por exemplo, a correlacao
entre camadas adjacentes, a continuidade lateral de horizontes, e a suavidade
natural dos perfis geologicos. Morales nao captura essas dependencias.

---

## 10. Dimensao 9: Modelo Fisico Forward

### Comparativo

| Aspecto              | Nosso Pipeline                              | Morales et al.                        |
|:---------------------|:--------------------------------------------|:--------------------------------------|
| **Modelo forward**   | PerfilaAnisoOmp.f08 (Fortran)               | Equacoes de Klein (analiticas)        |
| **Complexidade**     | PDE Maxwell em meio TIV estratificado       | 2 equacoes algebricas simples         |
| **Diferenciavel?**   | **NAO** (solver numerico Fortran)            | **SIM** (analitico, diretamente)      |
| **Embutivel em TF?** | Requer surrogate neural ou GradientTape     | Sim — como camada Lambda/funcao       |
| **Custo por avaliacao**| Segundos (solver completo)                 | Microsegundos (2 equacoes)            |
| **Cenarios PINNs**   | 3 (oracle, surrogate, maxwell)              | 1 (Klein embutido na loss)            |

### Detalhamento

**Nosso pipeline** enfrenta o desafio fundamental de que o modelo forward
(equacoes de Maxwell em meio anisotrópico) nao pode ser diretamente embutido
como camada diferenciavel. Por isso, oferecemos 3 cenarios:

1. **Oracle:** Usa dados pre-computados do Fortran como referencia. Requer que
   o Fortran ja tenha sido executado para os mesmos parametros. Estavel mas
   limitado a dados existentes.

2. **Surrogate:** Treina uma rede neural separada para aproximar o forward model.
   Diferenciavel por construcao, mas introduz erro de aproximacao.

3. **Maxwell:** Implementa a equacao de Helmholtz diretamente via GradientTape
   (derivadas de segunda ordem). Mais fiel a fisica, mas computacionalmente
   caro e numericamente desafiador.

**Morales** tem a vantagem de equacoes simples e analiticas que podem ser
implementadas como:
```python
def klein_forward(Csh, Rss, Rsh_v, Rsh_h):
    Rv_pred = Csh * Rsh_v + (1 - Csh) * Rss
    Rh_pred = 1.0 / (Csh / Rsh_h + (1 - Csh) / Rss)
    return Rv_pred, Rh_pred
```
Isso permite calculo de gradientes via autograd sem nenhuma aproximacao.

---

## 11. Dimensao 10: Otimizador e Hiperparametros

| Aspecto              | Nosso Pipeline                              | Morales et al.                        |
|:---------------------|:--------------------------------------------|:--------------------------------------|
| **Otimizador**       | Adam (default), AdamW, SGD (configuravel)   | AdamW (fixo)                          |
| **Learning rate**    | Configuravel (C9: LEARNING_RATE)            | 0.001 (fixo)                          |
| **Weight decay**     | Via L2/ElasticNet regularization (C11)      | 1e-5 (nativo do AdamW)               |
| **Batch size**       | Configuravel (C9: BATCH_SIZE)               | 32                                     |
| **Epocas**           | Configuravel (C9: N_EPOCHS)                 | 300                                    |
| **Early stopping**   | Sim (configuravel)                          | Sim (patience=20)                      |
| **LR scheduler**     | Configuravel (ReduceLROnPlateau, etc.)      | Nao mencionado                         |
| **Tempo de treino**  | Minutos a horas (GPU dependente)            | 31 min (RTX 3080)                      |
| **Tempo inferencia** | Ms a segundos (batch seq2seq)               | ~0.5 ms por perfil                     |

### Observacao sobre AdamW

Morales valida empiricamente o uso de AdamW com weight_decay=1e-5 para PINNs.
Nosso pipeline ja suporta AdamW como opcao (C9), mas o default e Adam sem
weight decay. A validacao de Morales reforça que AdamW pode ser preferivel
quando PINNs esta ativo, pois o weight decay auxilia na estabilidade do
treinamento com termos de fisica dominantes.

---

## 12. Tabela Resumo Comparativa

```
+----------------------+------------------------------+------------------------------+
| Dimensao             | Nosso Pipeline v5.0.15       | Morales et al. 2025          |
+----------------------+------------------------------+------------------------------+
| Problema             | Inversao geofisica (H->rho)  | Inversao petrofisica (rho->C)|
| Fisica               | Maxwell (PDE, complexa)      | Klein TI (2 eq. algebricas)  |
| Aprendizado          | Supervisionado + fisica      | Physics-only (sem labels)    |
| Peso da fisica       | 1% (PINNS_LAMBDA=0.01)      | 85% (omega=0.85)             |
| Arquitetura          | 44 arqs, seq2seq, >1M params | FC simples, 23K params       |
| Constraint saida     | Soft (via loss)              | Hard (sigmoid/ReLU layer)    |
| Escala entrada       | (batch, 600, 5-17)           | (batch, 4) -- ponto a ponto  |
| Contexto espacial    | Sim -- sequencia completa    | Nao -- independente          |
| lambda schedule      | Warmup 10 + ramp 20 epocas   | omega fixo desde epoca 1     |
| Norma loss physics   | L2 (MSE)                     | L2 (MSE) + L1 (MAE) hibrido |
| Dados treino         | Sinteticos (Fortran)         | Sinteticos + campo real      |
| Incerteza (UQ)       | MC Dropout / NLL             | Perturbacao ensemble P10-P90 |
| Inferencia           | Ms a segundos (batch)        | ~0.5 ms por perfil           |
| Escopo               | Inversao completa (fase toda)| Modulo petrofisico           |
| Posicao na cadeia    | H tensor -> rho_h, rho_v     | rho_h, rho_v -> Csh, Rss    |
| Forward model        | Fortran (NAO diferenciavel)  | Klein (diferenciavel)        |
| Ativacao output      | linear (sem restricao)       | sigmoid + ReLU (hard bound)  |
| Otimizador default   | Adam                         | AdamW                        |
| Dados de campo       | Nao utilizados               | Sim (2 datasets reais)       |
+----------------------+------------------------------+------------------------------+
```

---

## 13. Insights Incorporaveis -- Analise Detalhada

### PRIORIDADE ALTA

#### Insight 1 — Constraint Layer (Hard Physical Bounds)

**Problema identificado:** A ausencia de constraint layer em nosso pipeline e a
lacuna tecnica mais concreta. Todas as 44 arquiteturas usam `activation='linear'`
na camada final, permitindo valores fisicamente invalidos.

**Solucao proposta por Morales:** sigmoid/ReLU na camada final. Em nosso contexto,
`softplus` (suave, sempre positiva, diferenciavel) e mais adequada.

**Aplicacao em nosso pipeline:**

```python
# ATUAL (toda arquitetura em C27-C36A):
TimeDistributed(Dense(OUTPUT_CHANNELS, activation='linear'))

# PROPOSTO (futuro, via FLAG):
if USE_PHYSICAL_CONSTRAINT_LAYER:
    # softplus garante output > 0 (em escala log10)
    # ou, se output ja esta em log10, nao precisa de constraint
    # pois exp(qualquer_valor) > 0
    TimeDistributed(Dense(OUTPUT_CHANNELS, activation='softplus'))
else:
    TimeDistributed(Dense(OUTPUT_CHANNELS, activation='linear'))
```

**Consideracao importante:** Se nosso pipeline opera em escala log10
(TARGET_SCALING="log10"), a saida da rede ja esta em dominio log e nao
ha restricao natural de positividade necessaria (log10(rho) pode ser
qualquer valor real). A constraint layer seria mais relevante se a saida
fosse em escala linear (rho diretamente).

**Celulas impactadas:** C7 (FLAG nova), C27-C36A (todas as arquiteturas)
**Nova FLAG sugerida:** `USE_PHYSICAL_CONSTRAINT_LAYER: bool = False`
**Nova FLAG sugerida:** `CONSTRAINT_ACTIVATION: str = "softplus"` (opcoes: "softplus", "relu", "sigmoid")

---

#### Insight 2 — UQ via Perturbacao Ensemble

**Problema identificado:** Nossa UQ atual (MC Dropout) requer Dropout ativo
durante inferencia e depende da presenca de camadas Dropout na arquitetura
escolhida. Nem todas as 44 arquiteturas possuem Dropout.

**Solucao proposta por Morales:** Perturbacao ensemble — adiciona ruido gaussiano
na entrada e coleta N predicoes para calcular P10-P90.

**Vantagens:**
- Arquitetura-agnostico — funciona com qualquer dos 44 modelos
- Complementar ao MC Dropout (mede incerteza aletoria, nao epistemica)
- Rapido — N=1000 inferencias com rede pequena ~0.5s

**Implementacao proposta para C48-C57:**

```python
def ensemble_uncertainty(model, x_input, N=1000, kappa=0.05):
    """UQ via perturbacao ensemble (Morales et al., 2025).

    Gera N versoes perturbadas da entrada e coleta predicoes.
    O intervalo P10-P90 estima a incerteza aletoria.

    Args:
        model:   Modelo Keras treinado.
        x_input: Tensor de entrada (batch, N_MEDIDAS, N_FEATURES).
        N:       Numero de perturbacoes (default: 1000).
        kappa:   Nivel de ruido relativo (default: 0.05 = 5%).

    Returns:
        Dict com 'p10', 'p50', 'p90', 'width' (P90-P10).
    """
    x_std = tf.math.reduce_std(x_input, axis=1, keepdims=True)
    predictions = []
    for _ in range(N):
        noise = tf.random.normal(tf.shape(x_input)) * kappa * x_std
        x_perturb = x_input + noise
        pred = model(x_perturb, training=False)
        predictions.append(pred)
    stack = tf.stack(predictions, axis=0)  # (N, batch, N_MEDIDAS, OUT_CH)
    p10 = tfp.stats.percentile(stack, 10.0, axis=0)
    p50 = tfp.stats.percentile(stack, 50.0, axis=0)
    p90 = tfp.stats.percentile(stack, 90.0, axis=0)
    return {"p10": p10, "p50": p50, "p90": p90, "width": p90 - p10}
```

**Celulas impactadas:** C48-C57 (avaliacao), C66-C73 (geosteering)
**Novas FLAGS sugeridas:**
- `USE_ENSEMBLE_UQ: bool = False`
- `UQ_ENSEMBLE_N: int = 1000`
- `UQ_ENSEMBLE_NOISE_KAPPA: float = 0.05`

---

### PRIORIDADE MEDIA

#### Insight 3 — Combinacao de Normas Diferentes (L2 fisica + L1 dados)

**Observacao:** Morales usa L2 (MSE) para o termo de fisica e L1 (MAE) para o
mismatch de dados. Esta combinacao intencional tem justificativa:
- L2 para fisica: penaliza mais desvios grandes das equacoes -> forca aderencia
- L1 para dados: robusta a outliers nas medidas brutas (ruido, artefatos)

**Aplicacao em nosso pipeline:** Quando USE_PINNS=True, o termo L_physics poderia
usar L2 (MSE) enquanto L_data usa L1 (MAE), criando uma nova variante de loss.

**Celulas impactadas:** C10 (novas FLAGS), C40 (implementacao)
**Novas FLAGS sugeridas:**
- `PINNS_PHYSICS_NORM: str = "l2"` (opcoes: "l1", "l2")
- `PINNS_DATA_NORM: str = "l1"` (opcoes: "l1", "l2")

---

#### Insight 4 — Calibracao do Peso PINNS_LAMBDA

**Observacao:** Nosso PINNS_LAMBDA=0.01 e muito conservador comparado ao omega=0.85
de Morales. Porem, ha razao arquitetural: nossas equacoes de Maxwell sao
nao-lineares e o gradiente de L_physics pode ser instavel.

**Proposta:** No cenario PINNS_SCENARIO="surrogate" (onde o forward model e uma
rede neural suave e diferenciavel), aumentar PINNS_LAMBDA para 0.1-0.5 poderia
ser explorado com seguranca. Incluir como configuracao recomendada no SKILL.md.

**Celulas impactadas:** C13 (ajuste de default), C40 (treinamento)
**Acao:** Documentar recomendacao de PINNS_LAMBDA por cenario:
- oracle:    PINNS_LAMBDA = 0.01 - 0.1  (Fortran como referencia)
- surrogate: PINNS_LAMBDA = 0.1 - 0.5   (neural surrogate, suave)
- maxwell:   PINNS_LAMBDA = 0.001 - 0.01 (GradientTape, instavel)

---

#### Insight 5 — Dados Mistos Sinteticos + Campo Real

**Observacao:** Morales treina com mix sintetico+campo e obtem boa generalizacao.
Nosso pipeline usa exclusivamente dados do Fortran.

**Proposta:** Quando dados de campo reais estiverem disponiveis, adicionar FLAG
para mistura-los ao treinamento. Morales usa proporcao ~50/50.

**Celulas impactadas:** C4 ou C5 (FLAGS), C19 (data loading), C24 (datasets)
**Novas FLAGS sugeridas:**
- `USE_FIELD_DATA: bool = False`
- `FIELD_DATA_PATH: str = ""`
- `FIELD_DATA_RATIO: float = 0.5`

---

#### Insight 6 — Matriz de Ponderacao Adaptativa W_d

**Observacao:** Morales usa W_d = diag([1/(Rv/IGR), 1/(Rh/IGR)]) para normalizar
a contribuicao de cada medida pela escala geologica (Indice Gamma Ray). Medidas
com valores absolutos maiores recebem pesos menores.

**Aplicacao em nosso pipeline:** Ponderacao por amostra (sample_weight) baseada
em propriedades geologicas do modelo (angulo, profundidade, SNR). Ja previsto
parcialmente no DATA_SCENARIO "2D" via P4 (sample_weight no tf.data).

**Celulas impactadas:** C24 (sample_weight no map_fn), C40 (treinamento)

---

### PRIORIDADE ESTRATEGICA (Longo Prazo)

#### Insight 7 — Integracao Sequencial como Pipeline Completo

**Descoberta chave:** Os dois sistemas resolvem problemas sequenciais
complementares na mesma cadeia de interpretacao petrofisica.

A PINN de Morales poderia ser integrada como modulo opcional de pos-processamento
nas celulas de geosteering (C66-C73), convertendo os perfis de rho_h e rho_v
preditos pelo nosso pipeline em propriedades petrofisicas (Csh, Rss).

Ver diagrama completo na Secao 14.

---

## 14. Cadeia de Interpretacao Integrada

```
Dados Brutos LWD (ferramenta EM triaxial)
     |
     v
[Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz), zobs]   <-- INPUT_FEATURES
     |
     v
+---------------------------------------+
|  NOSSO PIPELINE v5.0.15               |
|  44 arquiteturas seq2seq               |
|  Loss: log_scale_aware + PINNs         |
|  UQ: MC Dropout + Ensemble (futuro)    |
+---------------------------------------+
     |
     v
[rho_h(z), rho_v(z)]    <-- OUTPUT_TARGETS (cols 2,3 do formato 22-col)
     |
     +---> Perfil de resistividade (entregavel primario)
     |
     v
+---------------------------------------+
|  MORALES PINN (modulo futuro C6x)     |  <-- INTEGRACAO FUTURA
|  FC simples, physics-only              |
|  Equacoes Klein TI                     |
|  UQ: Perturbacao Ensemble P10-P90      |
+---------------------------------------+
     |
     v
[Csh(z), Rss(z)]  <-- Propriedades petrofisicas
                       (fracao de folhelho, resistividade do arenito)
     |
     v
Caracterizacao petrofisica completa em tempo real
(integravel ao geosteering C66-C73)
```

### Beneficios da Integracao

1. **Cadeia completa automatizada:** De dados EM brutos a propriedades
   petrofisicas em tempo real
2. **UQ propagada:** A incerteza em rho_h/rho_v propaga-se para Csh/Rss
   via ensemble perturbation
3. **Sem custo computacional significativo:** A PINN de Morales tem apenas
   23K parametros e inferencia ~0.5ms
4. **Validacao cruzada:** Se Csh predito e fisicamente inconsistente
   (ex: Csh > 1), indica problemas na predicao de rho

---

## 15. Plano de Acoes Recomendadas

| # | Insight                               | Prioridade   | Celulas          | Novas FLAGS                           |
|:--|:--------------------------------------|:-------------|:-----------------|:--------------------------------------|
| 1 | Constraint layer (softplus/ReLU)      | **Alta**     | C7, C27-C36A     | USE_PHYSICAL_CONSTRAINT_LAYER         |
| 2 | UQ ensemble (N=1000, kappa=5%)        | **Alta**     | C48-C57, C66-C73 | UQ_ENSEMBLE_N, UQ_ENSEMBLE_NOISE_KAPPA|
| 3 | L2+L1 hibrido nas normas              | Media        | C10, C40         | PINNS_PHYSICS_NORM, PINNS_DATA_NORM   |
| 4 | PINNS_LAMBDA aumento no surrogate     | Media        | C13, C40         | Ajuste de default por cenario         |
| 5 | Dados mistos sintetico+campo          | Media        | C5, C19, C24     | USE_FIELD_DATA, FIELD_DATA_RATIO      |
| 6 | Ponderacao adaptativa W_d             | Media        | C24, C40         | (via sample_weight existente)         |
| 7 | Morales PINN como pos-inversao        | Estrategica  | C66-C73          | USE_PETROPHYSICAL_INVERSION           |

### Cronograma Sugerido

**Fase 6 (C40-C47 — Treinamento):**
- Incorporar Insights 3, 4, 6 nas celulas de treinamento

**Fase 7 (C48-C57 — Avaliacao):**
- Incorporar Insight 2 (UQ Ensemble) como metodo de UQ complementar

**Fase 8 (C58-C65 — Visualizacao):**
- Plots de incerteza P10-P90 do ensemble

**Fase 9 (C66-C73 — Geosteering):**
- Incorporar Insight 7 (Morales PINN como modulo opcional)
- Integrar UQ Ensemble no fluxo realtime

**Fase 5 revisao (C27-C39 — Arquiteturas):**
- Incorporar Insight 1 (Constraint Layer) — requer revisao de C27-C36A

---

## 16. Conclusoes

### 16.1 Resumo das Diferencas

Os dois sistemas PINN diferem em **10 dimensoes fundamentais**. A diferenca mais
critica e o **tipo de aprendizado**: nosso pipeline e supervisionado com fisica
como regularizacao (1%), enquanto Morales e physics-only com dados como
complemento (15%). Esta diferenca e consequencia direta da complexidade dos
modelos fisicos envolvidos (Maxwell vs. Klein).

### 16.2 Complementaridade

A descoberta mais valiosa desta analise e que os dois sistemas sao
**complementares e sequenciais** na cadeia de interpretacao petrofisica.
A integracao do modulo Morales como pos-processamento em C66-C73 criaria
um pipeline end-to-end de dados EM brutos a propriedades petrofisicas.

### 16.3 Lacunas Identificadas

As duas lacunas mais concretas do nosso pipeline, reveladas pela comparacao:

1. **Ausencia de constraint layer hard** — solucionavel com softplus/ReLU
   (Insight 1, prioridade alta)

2. **UQ limitada a MC Dropout** — expandivel com perturbacao ensemble
   (Insight 2, prioridade alta)

### 16.4 Validacao do Design Atual

A comparacao tambem valida varias decisoes do nosso pipeline:

- O schedule lambda com warmup+ramp e **superior** ao omega fixo de Morales
  para nosso problema (Maxwell e mais complexo e requer ramp gradual)
- O contexto espacial (seq2seq) e uma **vantagem significativa** sobre o
  processamento ponto-a-ponto de Morales
- Os 3 cenarios PINNs (oracle, surrogate, maxwell) oferecem **mais
  flexibilidade** que o unico cenario de Morales
- A diversidade de 44 arquiteturas permite **exploracao** que a FC fixa
  de Morales nao oferece

---

## 17. Referencias

1. **Morales, J. C. R., Dos Santos, W. L. R., & Guevara, S. L.** (2025).
   Anisotropic resistivity estimation and uncertainty quantification from
   borehole triaxial electromagnetic induction measurements: Gradient-based
   inversion and physics-informed neural network.
   *Geophysical Journal International*, ggaf101.
   DOI: 10.1093/gji/ggaf101

2. **Klein, J. D.** (1993). Resistivity anisotropy.
   *In Development in Petrophysics*, Geological Society of London, 122-127.

3. **Pipeline de Inversao Geofisica com Deep Learning v5.0.15** —
   Documentacao interna (CLAUDE.md, SKILL.md, ERRATA_E_VALORES.md).

4. **PerfilaAnisoOmp.f08** — Simulador Fortran de perfilagem EM em meio
   anisotrópico, codigo-fonte em Fortran_Gerador/.

---

*Documento gerado em 2026-03-13. Versao do pipeline: v5.0.15.*
*Analise realizada com base nos arquivos C10, C13, novas_loss.py e artigo completo.*
