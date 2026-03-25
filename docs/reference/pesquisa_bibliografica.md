# Pesquisa Bibliográfica — Pipeline de Inversão Geofísica com Deep Learning v5.0.15

## Revisão Completa da Literatura Científica e Técnica

**Projeto:** Pipeline de Inversão 1D de Resistividade via Deep Learning
**Versão:** v5.0.15
**Autor da revisão:** Daniel Leal
**Data:** Março 2026
**Framework:** TensorFlow 2.x / Keras (exclusivo)
**Escopo:** Geofísica, Petrofísica, Inteligência Artificial, Deep Learning, Inversão EM, LWD, Geosteering

---

## Sumário

1. [Introdução e Motivação](#1-introdução-e-motivação)
2. [Fundamentos de Perfilagem Eletromagnética e Petrofísica](#2-fundamentos-de-perfilagem-eletromagnética-e-petrofísica)
3. [Inversão EM Tradicional — Métodos Clássicos](#3-inversão-em-tradicional--métodos-clássicos)
4. [Deep Learning para Inversão EM em LWD](#4-deep-learning-para-inversão-em-em-lwd)
5. [Geosteering com Inteligência Artificial](#5-geosteering-com-inteligência-artificial)
6. [Redes Neurais Informadas por Física (PINNs)](#6-redes-neurais-informadas-por-física-pinns)
7. [Arquiteturas de Deep Learning — Referências das 44 Implementações](#7-arquiteturas-de-deep-learning--referências-das-44-implementações)
8. [Funções de Perda — Referências das 25 Losses](#8-funções-de-perda--referências-das-25-losses)
9. [Estratégias de Treinamento e Augmentação de Dados](#9-estratégias-de-treinamento-e-augmentação-de-dados)
10. [Quantificação de Incerteza em DL para Geofísica](#10-quantificação-de-incerteza-em-dl-para-geofísica)
11. [Geosinais e Medidas Direcionais](#11-geosinais-e-medidas-direcionais)
12. [Ferramentas Comerciais LWD](#12-ferramentas-comerciais-lwd)
13. [Livros e Textos Fundamentais](#13-livros-e-textos-fundamentais)
14. [Referências Compiladas](#14-referências-compiladas)

---

## 1. Introdução e Motivação

O presente documento constitui uma revisão bibliográfica abrangente da literatura
científica e técnica relevante ao Pipeline de Inversão Geofísica com Deep Learning
v5.0.15. O pipeline implementa um sistema de inversão 1D de resistividade a partir
de medidas eletromagnéticas (EM) de ferramentas de perfilagem durante a perfuração
(LWD — Logging While Drilling), utilizando redes neurais profundas como substituto
computacionalmente eficiente dos métodos tradicionais de inversão iterativa.

### 1.1 Contexto do Problema

A inversão de dados eletromagnéticos de poço é um problema inverso mal-posto:
dadas as medidas de campo magnético (componentes do tensor H), deseja-se
reconstruir a distribuição de resistividade elétrica da formação rochosa ao redor
do poço. Tradicionalmente, este problema é resolvido por métodos iterativos
(Gauss-Newton, Occam, Levenberg-Marquardt) que requerem múltiplas avaliações
do problema direto (forward modeling) — processo computacionalmente intensivo
que impede a inversão em tempo real durante a perfuração.

O avanço recente do Deep Learning oferece uma alternativa: treinar redes neurais
com dados sintéticos gerados por simuladores EM de alta fidelidade, criando um
mapeamento direto medidas → resistividade que opera em milissegundos — viabilizando
a inversão em tempo real para aplicações de geosteering.

### 1.2 Escopo da Revisão

Esta revisão cobre:

- **Fundamentos físicos:** Eletromagnetismo de poço, propagação EM em meios
  anisotrópicos (TIV), skin depth, acoplamento de bobinas triaxiais
- **Métodos tradicionais:** Inversão iterativa, regularização, modelagem 1D/2D/2.5D
- **Deep Learning aplicado:** CNNs, RNNs, Transformers, U-Nets para inversão EM
- **Geosteering:** Sistemas de decisão em tempo real, navegação direcional
- **Arquiteturas:** Referências das 44 arquiteturas implementadas no pipeline
- **Losses:** Referências das 25 funções de perda (genéricas, geofísicas, geosteering)
- **Treinamento:** Curriculum learning, noise injection, data augmentation
- **Incerteza:** MC Dropout, NLL gaussiana, métodos bayesianos
- **Ferramentas comerciais:** GeoSphere HD, EarthStar, DeepLook-EM

### 1.3 Relação com o Pipeline v5.0.15

O pipeline implementa 74 células (C0–C73) organizadas em 8 seções operacionais,
suportando 44 arquiteturas de rede neural (39 standard + 5 geosteering), 25 funções
de perda, 43 tipos de ruído e 5 perspectivas de dados (P1–P5). Cada referência
nesta revisão está mapeada para as células e componentes específicos do pipeline
onde sua contribuição é aplicada.

---

## 2. Fundamentos de Perfilagem Eletromagnética e Petrofísica

### 2.1 Perfilagem de Poço — Visão Geral

A perfilagem de poço (well logging) é o processo de medição contínua de
propriedades físicas das formações rochosas atravessadas pelo poço de petróleo.
As medidas são adquiridas por ferramentas (sondes) que descem ao interior do
poço, registrando perfis (logs) de propriedades como resistividade elétrica,
porosidade, densidade, velocidade sônica e radioatividade natural.

A obra de referência fundamental é **Ellis e Singer (2008)** [L1], que cobre
de forma abrangente os princípios físicos de todas as ferramentas de perfilagem:

> Ellis, D. V., & Singer, J. M. (2008). *Well Logging for Earth Scientists*
> (2nd ed.). Springer. ISBN 978-1-4020-3738-2.

Este livro cobre em seus 27 capítulos: medidas de resistividade (indução,
laterolog, propagação), porosidade (nêutron, densidade, sônico), radioatividade
natural (gamma ray), ressonância magnética nuclear (NMR), e perfilagem durante
a perfuração (LWD/MWD). O capítulo sobre ferramentas de propagação EM é
particularmente relevante para o pipeline, pois detalha o princípio de
funcionamento das ferramentas LWD que geram os dados de entrada.

**Relevância para o pipeline:** C5 (INPUT_FEATURES), C19 (data loading),
C21 (decoupling EM). Conceitos de skin depth, fator geométrico, atenuação
e phase shift são utilizados diretamente no cálculo dos geosinais (C22).

### 2.2 Machine Learning para Caracterização de Subsuperfície

A aplicação de técnicas de aprendizado de máquina em petrofísica e
caracterização de reservatórios é consolidada em **Misra, Li e He (2019)** [L2]:

> Misra, S., Li, H., & He, J. (2019). *Machine Learning for Subsurface
> Characterization*. Gulf Professional Publishing. ISBN 978-0-12-817736-5.

Este livro de 86 MB (disponível no projeto) cobre de forma sistemática:
- Regressão e classificação de litofácies a partir de perfis de poço
- Redes neurais artificiais (ANNs) para estimativa de porosidade e saturação
- Support Vector Machines (SVMs) para identificação de zonas produtoras
- Clustering não-supervisionado para agrupamento de fácies
- Deep Learning aplicado a dados sísmicos e de perfilagem
- Técnicas de transfer learning entre poços de diferentes campos

**Relevância para o pipeline:** Fundamentação teórica geral de ML aplicado
a dados de poço. Conceitos de feature engineering, normalização de dados
de perfilagem e avaliação de modelos preditivos em contexto petrofísico.

### 2.3 Princípios de Perfilagem de Poço

O livro **Principles and Applications of Well Logging** [L3] (disponível
no projeto como texto em chinês) complementa a visão de Ellis e Singer com
foco em aplicações práticas de interpretação de perfis, incluindo:
- Avaliação de formações com ferramentas convencionais e especiais
- Interpretação quantitativa de perfis em formações complexas
- Métodos de análise integrada de múltiplos perfis

**Relevância para o pipeline:** Contexto prático de como as medidas EM
são interpretadas na indústria, informando a escolha de OUTPUT_TARGETS
(resistividade horizontal ρh e vertical ρv).

### 2.4 Física da Propagação EM em Formações Anisotrópicas

A propagação de ondas eletromagnéticas em formações rochosas com
anisotropia de resistividade é governada pelas equações de Maxwell em
meios condutivos. Para formações com isotropia transversal vertical (TIV),
a resistividade é descrita por dois parâmetros:

- **ρh** (resistividade horizontal): resistividade medida paralelamente
  às camadas de estratificação
- **ρv** (resistividade vertical): resistividade medida perpendicularmente
  às camadas

A relação de anisotropia λ = √(ρv/ρh) é tipicamente > 1.0 em formações
laminadas (folhelhos intercalados com arenitos), podendo atingir valores
de 2 a 10 em casos extremos.

O **skin depth** (profundidade de penetração) é dado por:

```
δ = √(2 / (ωμσ))
```

onde ω = 2πf é a frequência angular, μ é a permeabilidade magnética
(≈ μ₀ para a maioria das rochas) e σ = 1/ρ é a condutividade.
Para a frequência do pipeline (f = 20.000 Hz) e ρ = 10 Ω·m:

```
δ = √(2 / (2π × 20000 × 4π×10⁻⁷ × 0.1)) ≈ 11.3 m
```

**Relevância para o pipeline:** FREQUENCY_HZ = 20000.0 (C5), SPACING_METERS = 1.0 (C5).
O skin depth determina a profundidade de investigação da ferramenta e é
um parâmetro crítico na modelagem direta (PerfilaAnisoOmp.f08).

O artigo de **Carvalho, Régis e Silva (2022)** [A9] investiga especificamente
os efeitos de borehole nos logs coaxiais e coplanares de ferramentas triaxiais
em formações laminadas com folhelhos anisotrópicos:

> Carvalho, P. R., Régis, C., & Silva, V. S. (2022). Borehole Effects on
> Coaxial and Coplanar Logs of Triaxial Tools in Laminated Formations with
> Anisotropic Shale Host. *Brazilian Journal of Geophysics*, 40(3), 411–420.
> DOI: 10.22564/brjg.v40i3.2170

Os autores comparam resultados de um programa 3D de Elementos Finitos
Vetoriais (com borehole) contra um código 1D analítico (sem borehole),
demonstrando que:
- A configuração coplanar vertical mostra efeito de horn mais forte nas
  fronteiras de pacotes laminados que a configuração coaxial
- O efeito de skin é mais pronunciado no coplanar para meios condutivos
- As sensibilidades à anisotropia e ao borehole são opostas: para ângulos
  pequenos o coaxial é menos sensível, e para ângulos grandes o coplanar
  é menos sensível

**Relevância para o pipeline:** C21 (decoupling EM), C22 (geosinais).
Os resultados informam a escolha de componentes do tensor H (Hxx, Hzz)
como features de entrada e a formulação dos geosinais USD, UAD, UHR, UHA.

---

## 3. Inversão EM Tradicional — Métodos Clássicos

### 3.1 Inversão Gauss-Newton Regularizada

O método tradicional de inversão de dados EM de poço é baseado na
minimização iterativa de uma função objetivo que combina o misfit entre
dados observados e sintéticos com um termo de regularização:

```
Φ(m) = ‖d_obs - F(m)‖² + λ‖Rm‖²
```

onde m é o vetor de parâmetros do modelo (resistividades), F(m) é o
operador de modelagem direta, d_obs são os dados observados, R é a
matriz de regularização e λ é o parâmetro de regularização.

**Wang et al. (2018)** [A6] apresentam uma análise sistemática de
sensibilidade e inversão de medidas azimutais de resistividade LWD:

> Wang, L., Li, H., Fan, Y., & Wu, Z. (2018). Sensitivity analysis and
> inversion processing of azimuthal resistivity logging-while-drilling
> measurements. *Journal of Geophysics and Engineering*, 15, 2339–2349.
> DOI: 10.1088/1742-2140/aacbf4

Os autores desenvolvem uma abordagem de inversão Gauss-Newton rápida e
regularizada com as seguintes otimizações:
- Janela deslizante ao longo da profundidade (depth sliding window)
- Limites e valores iniciais otimizados a partir de análise de sensibilidade
- Aceleração via cálculo analítico do Jacobiano para modelos 1D

**Relevância para o pipeline:** C19 (data loading — formato similar),
C21 (decoupling). A abordagem de janela deslizante é análoga ao
SEQUENCE_LENGTH = 600 pontos do pipeline.

### 3.2 Inversão de Medidas Extra-Deep Azimutais

**Wang et al. (2019)** [A7] estendem o trabalho anterior para medidas
azimutais de ultra-profundidade (EDARM):

> Wang, L., Deng, S., Zhang, P., Cao, Y., Fan, Y., & Yuan, X. (2019).
> Detection performance and inversion processing of logging-while-drilling
> extra-deep azimuthal resistivity measurements. *Petroleum Science*.
> DOI: 10.1007/s12182-019-00374-4

Contribuições relevantes:
- Nova definição de profundidade de detecção considerando a incerteza
  da inversão causada por ruído nos dados
- Teoria bayesiana com amostragem MCMC (Markov Chain Monte Carlo)
  para processamento rápido de dados EDARM
- Demonstração de que a capacidade de detecção aumenta com maior
  espaçamento entre transmissor e receptor e maior contraste de resistividade

**Relevância para o pipeline:** C12 (noise models), C22 (geosinais).
A definição de profundidade de detecção baseada em incerteza é análoga
à quantificação de incerteza via MC Dropout no pipeline (C4, FLAGS de
incerteza).

### 3.3 Modelagem 1D Triaxial Eficiente

**Guo et al. (2024)** [A8] introduzem um método eficiente de simulação
para perfilagem EM triaxial em formações 1D anisotrópicas:

> Guo, W., Wang, L., Wang, N., Qiao, P., Zeng, Z., & Yang, K. (2024).
> Efficient 1D Modeling of Triaxial Electromagnetic Logging in Uniaxial
> and Biaxial Anisotropic Formations Using Virtual Boundaries and
> Equivalent Resistivity Schemes. *Geophysics* (gxag017).

O método propõe:
- Fórmulas compactas para campos EM em meios uniaxiais anisotrópicos (UA)
  que transformam o cálculo em 3 pares de amplitudes desacoplados
- Esquema de fronteiras virtuais para garantir que todas as fronteiras
  de formação estejam em camadas sem fonte
- Algoritmo MPC (Modified Propagator Coefficient) para cálculo das
  amplitudes sem operações de matriz complexas
- Extensão para formações biaxiais anisotrópicas (BA) via aproximação

**Relevância para o pipeline:** O simulador PerfilaAnisoOmp.f08 do
projeto implementa princípios similares de modelagem 1D triaxial. Os
dados gerados (.dat de 22 colunas) contêm o tensor completo 3×3 de
campos magnéticos (Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz).

### 3.4 Inversão Multidimensional de UDAR

**Saputra, Torres-Verdín et al. (2026)** [A10] representam o
estado-da-arte em inversão multidimensional de medidas UDAR
(Ultradeep Azimuthal Resistivity):

> Saputra, W., Torres-Verdín, C., Ambia, J., et al. (2026). Recent
> Developments and Verifications of Multidimensional and Data-Adaptive
> Inversion of Borehole UDAR Measurements. *Petrophysics*, 67(1), 173–189.
> DOI: 10.30632/PJV67N1-2026a12

Este artigo descreve:
- Algoritmos de modelagem e inversão rápidos baseados em solução
  de volumes finitos das equações de Maxwell em grid adaptativo de Lebedev
- Integração com modelos geológicos 3D locais e trajetórias arbitrárias
- Solver block-based para eficiência computacional
- Validação com dados de campo e modelos sintéticos complexos

**Relevância para o pipeline:** A10 representa o baseline de alta
fidelidade contra o qual as inversões por DL devem ser comparadas.
A abordagem multidimensional inspira a extensão futura do pipeline
para inversão 2D/2.5D/3D.

---

## 4. Deep Learning para Inversão EM em LWD

Esta é a seção central da revisão, cobrindo os trabalhos que fundamentam
diretamente o pipeline v5.0.15. O grupo de pesquisa de Noh, Torres-Verdín
e Pardo é a principal referência, com uma série de publicações progressivas
que evoluíram de inversão 1D simples para inversão 2.5D com detecção de
falhas e tratamento de ruído.

### 4.1 Inversão 2.5D com Deep Learning para Geosteering (Noh et al., 2022)

> Noh, K., Torres-Verdín, C., & Pardo, D. (2022). Real-Time 2.5D
> Inversion of LWD Resistivity Measurements Using Deep Learning for
> Geosteering Applications Across Faulted Formations. *Petrophysics*,
> 63(4), 506–518. DOI: 10.30632/PJV63N4-2022a2

**Contribuições principais:**

Este é o artigo seminal que mais diretamente inspira o pipeline v5.0.15.
Os autores desenvolvem um workflow de inversão DL para problemas inversos
2.5D que emprega **quatro arquiteturas DL independentes**:

1. **Classificador de estrutura geológica:** Identifica o tipo de
   estrutura (sem falta, com fronteira de camada, com plano de falha)
2. **Inversor sem cruzamento:** Reconstrói resistividade quando não há
   cruzamento de fronteiras ou falhas
3. **Inversor com fronteira:** Reconstrói quando há cruzamento de
   fronteira de camada
4. **Inversor com falha:** Reconstrói quando há cruzamento de plano
   de falha

Cada arquitetura emprega **camadas convolucionais** (CNNs) treinadas com
dados sintéticos gerados por um simulador EM de alta ordem baseado em
elementos finitos adaptativos (hp-FEM). O treinamento utiliza conjuntos de
dados de 10.000 a 50.000 modelos geológicos com variação paramétrica de:
- Resistividade horizontal e vertical de cada camada
- Ângulo de mergulho (dip angle)
- Posição e orientação do plano de falha
- Espaçamentos e frequências da ferramenta

**Resultados quantitativos:**
- Inversão em < 1 segundo por janela (vs. minutos/horas para inversão
  iterativa convencional)
- Erro RMS < 10% para modelos com até 5 camadas e 1 falha
- Detecção correta do tipo de estrutura em > 95% dos casos

**Relevância para o pipeline:**
- C19: O formato de dados e split por modelo geológico (P1) são
  inspirados neste trabalho
- C27–C36: As arquiteturas CNN 1D do pipeline seguem o paradigma de
  usar convoluções ao longo da dimensão espacial (profundidade medida)
- C37: O Model Factory permite selecionar entre 44 arquiteturas,
  incluindo variantes inspiradas neste trabalho
- C40–C43: O loop de treinamento segue o paradigma de treinar com
  dados sintéticos e validar com modelos geológicos separados

### 4.2 Inversão DL Robusta a Ruído (Noh et al., 2021/2023)

> Noh, K., Pardo, D., & Torres-Verdín, C. (2021). Deep-Learning
> Inversion Method for the Interpretation of Noisy Logging-While-Drilling
> Resistivity Measurements. arXiv:2111.07490v1. (Preprint para IEEE
> Transactions on Geoscience and Remote Sensing)

> Noh, K., Pardo, D., & Torres-Verdín, C. (2023). Physics-guided
> deep-learning inversion method for the interpretation of noisy
> logging-while-drilling resistivity measurements. *Geophysical Journal
> International*, 235, 150–165. DOI: 10.1093/gji/ggad217

**Contribuições principais:**

Estes dois artigos (o preprint arXiv e a versão final no GJI) são
fundamentais para o pipeline v5.0.15 pois tratam especificamente do
**efeito do ruído nas medidas sobre a qualidade da inversão DL**.

Os autores testam **três abordagens** para lidar com ruído:

1. **Noise injection no training set:** Adicionar ruído gaussiano às
   medidas do conjunto de treinamento (abordagem mais simples)
2. **Data augmentation com réplicas ruidosas:** Replicar o dataset
   K vezes, cada réplica com uma realização diferente de ruído
   (equivalente ao nosso cenário 3x off-line)
3. **Noise layer na arquitetura DL:** Adicionar uma camada de ruído
   estocástico diretamente na rede neural durante o treinamento

**Resultados-chave:**
- As três abordagens produzem um **efeito de denoising**, melhorando
  significativamente a robustez da inversão a ruído
- A abordagem 2 (data augmentation) é a mais eficaz para níveis de
  ruído moderados a altos
- A combinação de data augmentation com physics-guided loss
  (regularização baseada em física) produz os melhores resultados
- O "physics-guided" refere-se a incorporar constraints do problema
  direto na função de perda (similar à nossa encoder-decoder loss)

**Relevância direta para o pipeline:**

| Conceito do artigo | Implementação no pipeline |
|:-------------------|:-------------------------|
| Noise injection simples | Cenário 2x (on-the-fly noise em C24) |
| Data augmentation com réplicas | Cenário 3x (off-line em C24) |
| Noise layer estocástico | AddNoise_OnTheFly2.py (43 tipos) |
| Physics-guided loss | Encoder-Decoder loss (#21 em C41) |
| Split por modelo geológico | P1 — split_by_geological_model() em C19 |
| Validação com e sem ruído | P2 — DualValidationCallback em C40 |

### 4.3 Inversão 2.5D com Detecção de Falhas (Noh et al., Manuscript v8)

> Noh, K., Pardo, D., & Torres-Verdín, C. (2023+). 2.5D Deep Learning
> Inversion of LWD and Deep-Sensing EM Measurements Across Formations
> with Dipping Faults. Manuscript v8 (submetido a IEEE TGRS).

**Contribuições adicionais ao artigo de 2022:**
- Módulo "look-around" para detecção de falhas com mergulho arbitrário
- Comparação entre usar apenas medidas LWD curtas vs. combinar LWD
  com medidas deep-sensing
- Demonstração de que medidas deep-sensing reduzem significativamente
  a incerteza na inversão
- Validação da aplicabilidade de inversão DL 2.5D em tempo real para
  geosteering em formações com falhas

**Relevância para o pipeline:** Inspira a perspectiva P5 (Picasso/DTB)
onde o modelo prediz não apenas resistividades mas também
distance-to-boundary (DTB) e resistividades adjacentes.

### 4.4 Estimativa Anisotrópica com PINNs e Incerteza (Morales et al., 2025) — ANÁLISE DETALHADA

> Morales, M. M., Eghbali, A., Raheem, O., Pyrcz, M. J., &
> Torres-Verdín, C. (2025). Anisotropic resistivity estimation and
> uncertainty quantification from borehole triaxial electromagnetic
> induction measurements: Gradient-based inversion and physics-informed
> neural network. *Computers & Geosciences*, 196, 105786.
>
> **Instituição:** The University of Texas at Austin
> **Grupo:** Formation Evaluation (FE) + Digital Reservoir Characterization Technology (DIRECT)
> **Repositório:** github.com/misaelmorales/Anisotropic-Resistivity-Inversion

#### 4.4.1 Problema Abordado

O artigo trata da inversão de resistividade anisotrópica a partir de medidas
triaxiais de indução eletromagnética em poço. O objetivo é estimar a concentração
volumétrica de folhelho (C_sh) e a resistividade do arenito (R_ss) a partir dos
perfis de resistividade horizontal (R_h) e vertical (R_v), assumindo meio
transversalmente isotrópico (TI — Transverse Isotropy).

O modelo físico subjacente é o sistema de equações de Klein (1993, 1996):

```
┌──────────────────────────────────────────────────────────────────────┐
│  Rv = Csh · Rsh_v + (1 - Csh) · Rss        (circuito em SÉRIE)    │
│  1/Rh = Csh/Rsh_h + (1 - Csh)/Rss          (circuito em PARALELO) │
│                                                                      │
│  onde: Rsh_v, Rsh_h = resistividades do folhelho puro (v, h)       │
│        Csh = concentração volumétrica de folhelho [0, 1]            │
│        Rss = resistividade do arenito (> 0)                         │
└──────────────────────────────────────────────────────────────────────┘
```

Este sistema é **idêntico** ao modelo TIV usado no nosso simulador
PerfilaAnisoOmp.f08, onde a resistividade horizontal (ρh = col 2) e vertical
(ρv = col 3) são os OUTPUT_TARGETS do pipeline.

#### 4.4.2 Três Métodos Comparados

O artigo compara três métodos de inversão:

| Método | Acurácia | Tempo | Estabilidade |
|:-------|:--------:|:-----:|:------------:|
| **Solução analítica** | 28-69% (R²) | Instantâneo | Colapsa quando Csh → 0 ou 1 |
| **Inversão por gradiente** (CG não-linear) | 98-99% | Minutos | Estável com L-curve |
| **PINN** | 91-99% | ~0.5 ms | Estável com constraint layer |

A solução analítica (fórmula quadrática para Csh) é **numericamente instável** —
colapsa em casos práticos. Tanto a inversão por gradiente quanto o PINN são
estáveis, mas o PINN é **~10⁶ vezes mais rápido**.

#### 4.4.3 Arquitetura da PINN

A rede é um feed-forward simples com **constraint layer**:

```
┌───────────────────────────────────────────────────────────────┐
│                    ARQUITETURA PINN                           │
│                                                               │
│  Input (4): [Rv, Rh, Rsh_v, Rsh_h]                          │
│      ↓                                                        │
│  Hidden 1: Linear(4→150) → BatchNorm → tanh → Dropout        │
│      ↓                                                        │
│  Hidden 2: Linear(150→150) → BatchNorm → tanh → Dropout      │
│      ↓                                                        │
│  Output (2): Linear(150→2)                                   │
│      ↓                                                        │
│  CONSTRAINT LAYER:                                            │
│    Csh → sigmoid (garante [0, 1])                            │
│    Rss → ReLU    (garante > 0)                               │
│                                                               │
│  Total: 23.402 parâmetros treináveis                         │
│  Treinamento: AdamW, lr=0.001, weight_decay=1e-5, 300 epochs │
│  Tempo de treino: ~31 min (RTX 3080)                         │
│  Tempo de inferência: ~0.5 ms                                │
└───────────────────────────────────────────────────────────────┘
```

**Insight crucial — Constraint Layer:** A aplicação de sigmoid para saídas
limitadas [0,1] e ReLU para saídas positivas é diretamente aplicável ao nosso
pipeline, onde ρh e ρv devem ser estritamente positivos.

#### 4.4.4 Loss Function Informada por Física

A função de perda é uma combinação ponderada de dois termos:

```
L_total = ω · L_physics + (1 - ω) · L_data

onde:
  L_physics = ||W_d · e(θ)||² + λ||θ||²    (MSE ponderada + L2 regularização)
  L_data    = ||x - x̂(θ)||₁               (L1 data mismatch)
  e(θ)      = x - x̂(θ)                    (erro de simulação)
  ω         = 0.85                          (determinado empiricamente)
  W_d       = [1/(Rv/IGR), 1/(Rh/IGR)]ᵀ   (ponderação adaptativa)
```

**Mecanismo chave:** A rede prediz (Csh, Rss), e os valores preditos são
usados para **calcular numericamente** R̂v e R̂h via Eq.(1). O erro entre
as medidas reais (Rv, Rh) e as simuladas (R̂v, R̂h) é a loss de física.
A rede **nunca vê labels diretos** de Csh ou Rss — aprende pela consistência
com as equações físicas.

**Insights para o pipeline:**

1. **ω = 0.85:** Peso de ~85% para o termo de física. Nosso C13 já tem
   `PINN_WEIGHT` — este valor serve como referência empírica inicial.

2. **Ponderação adaptativa W_d:** Normalização por Gamma Ray Index (IGR).
   Conceito transferível: ponderar medidas EM por profundidade ou ângulo
   para equilibrar contribuições de diferentes depths no batch.

3. **L2 regularização no loss (λ||θ||²):** Estabiliza a inversão —
   nosso C11 já tem `USE_L2_REGULARIZATION`, que cumpre papel similar.

4. **Combinação L2 (physics) + L1 (data):** A escolha de normas diferentes
   para cada componente (MSE para física, MAE para dados) é intencional —
   L1 é mais robusta a outliers nos dados, L2 penaliza mais desvios
   grandes das equações físicas.

#### 4.4.5 Quantificação de Incerteza

O artigo propõe um método prático de UQ por perturbação ensemble:

```
┌──────────────────────────────────────────────────────────────────┐
│  MÉTODO DE INCERTEZA POR PERTURBAÇÃO ENSEMBLE                   │
│                                                                  │
│  1. Gerar N=1000 perturbações: x̃ = x + ε, ε ~ N(0, κσ_x)     │
│  2. Inferir θ̂ para cada perturbação                             │
│  3. Calcular percentis P10 e P90 em cada profundidade            │
│  4. Incerteza média: U(θj) = (1/Z) Σ [P90(θj_z) - P10(θj_z)]  │
│                                                                  │
│  κ = 5% (nível de ruído)                                        │
│  Resultados: U(Csh) ≈ 0.044 v/v, U(Rss) ≈ 0.228 Ωm           │
└──────────────────────────────────────────────────────────────────┘
```

**Observações importantes:**
- A incerteza em Csh é mais **consistente ao longo da profundidade**,
  enquanto Rss pode superestimar em certas regiões
- R̂h tem menor incerteza que R̂v em todos os 4 datasets
- A incerteza localiza-se em zonas de transição litológica

#### 4.4.6 Datasets de Validação

| Dataset | Tipo | Profundidade | Amostras | Contexto Geológico |
|:--------|:-----|:-------------|:--------:|:-------------------|
| Sintético 1 | 3D-UTAPWeLS | 5485-5680 ft | 779 | 12 camadas puro shale/arenito |
| Sintético 2 | 3D-UTAPWeLS | 5091-5194 ft | 415 | 22 camadas com shale disperso |
| Campo 1 | Norte da África | 9720-10110 ft | 1560 | Turbiditos laminados, alta mergulho |
| Campo 2 | Mar do Norte (Horda) | 6293-9078 ft | 11142 | Draupne Fm, deepwater |

O artigo demonstra que o PINN generaliza bem para dados de campo reais,
mesmo tendo sido treinado com apenas 3200 medidas (1 sintético + 1 campo).

#### 4.4.7 Inversão com Medidas Convencionais (sem triaxial)

Quando logs triaxiais (Rv, Rh) não estão disponíveis, o artigo propõe usar
apenas o log de resistividade profunda (RT90) com:
- C̃sh estimado pelo Gamma Ray Index (IGR)
- R̃ss estimado por inversão usando apenas a componente horizontal de Eq.(1)

Resultados: MAPE ~11.6-24.0%, R² ~69-80%. **Menos preciso**, mas viável
como alternativa quando triaxial não está disponível.

**Relevância:** Demonstra flexibilidade do método — nosso pipeline poderia
oferecer modo degradado com menos features de entrada.

#### 4.4.8 Contribuições Específicas ao Pipeline v5.0.15

| Insight do Artigo | Célula(s) Impactada(s) | Ação Recomendada |
|:-------------------|:----------------------:|:-----------------|
| **Constraint layer** (sigmoid/ReLU) | C27-C36 | Adicionar camada final que garante ρ > 0 via softplus ou ReLU |
| **Physics-informed loss** (ω=0.85) | C10, C13, C40 | Calibrar PINN_WEIGHT com ω=0.85 como default |
| **Ponderação adaptativa W_d** | C10, C24 | Implementar sample_weight adaptativo baseado em propriedades |
| **Combinação L2+L1** | C10 | Considerar loss híbrida MSE(physics) + MAE(data) |
| **UQ por perturbação ensemble** | C48-C57 | Implementar avaliação de incerteza via perturbação (κ=5%) |
| **Shale properties adaptativas** | C20 | Wavelet peak-finding para propriedades variáveis com profundidade |
| **Treinamento com dados mistos** | C24 | Combinar dados sintéticos + campo para melhor generalização |
| **Modo degradado sem triaxial** | C4, C66-C73 | Prever modo com INPUT_FEATURES reduzido |
| **Forward model embarcado no loss** | C10, C40 | Incorporar forward EM model na loss durante treino |
| **AdamW optimizer** | C9, C40 | AdamW (com weight decay) como alternativa ao Adam padrão |

#### 4.4.9 Relação com o Simulador PerfilaAnisoOmp.f08

O modelo TI usado por Morales et al. (Eq. 1 do artigo) é **fisicamente
equivalente** ao que nosso simulador Fortran calcula:

- Nosso simulador resolve o problema **direto**: dado um modelo de camadas
  com ρh e ρv, calcula as componentes do tensor H (Hxx, Hzz, etc.)
- Morales resolve o problema **inverso petrofísico**: dado Rv e Rh, estima
  Csh e Rss (que determinam Rv e Rh via modelo TI)
- Nosso pipeline resolve o **inverso geofísico**: dadas as componentes de H,
  estima ρh e ρv diretamente via deep learning

A cadeia completa seria:

```
Modelo geológico → [PerfilaAnisoOmp] → H tensor → [Nosso Pipeline DL] → ρh, ρv
                                                                           ↓
                                                              [Morales PINN] → Csh, Rss
```

Assim, o trabalho de Morales et al. poderia ser um **estágio subsequente**
ao nosso pipeline, ou poderia ser **integrado** como um módulo adicional
nas células C66-C73 (Geosteering).

---

## 5. Geosteering com Inteligência Artificial

### 5.1 Sistema de Decisão para Geosteering (Alyaev et al.)

O grupo de pesquisa da NORCE (Norwegian Research Centre), liderado por
Sergey Alyaev, desenvolveu uma série progressiva de trabalhos sobre
sistemas de decisão para geosteering assistido por IA:

> Alyaev, S., Suter, E., & Bratvold, R. B. (2019). A decision support
> system for multi-target geosteering. *Journal of Petroleum Science
> and Engineering*, 183, 106381.

> Alyaev, S., et al. (2021). Deep learning for prediction of complex
> geology ahead of drilling. In *Springer* (APPEA Conference).

> Alyaev, S., et al. (2022). Direct Multi-Modal Inversion of Geophysical
> Logs Using Deep Learning. *Earth and Space Science*, 9,
> e2021EA002186.

> Alyaev, S., et al. (2022). Strategic Geosteering Workflow with
> Uncertainty Quantification and Deep Learning: A Case Study on the
> Goliat Field. arXiv:2210.15548.

> Alyaev, S., et al. (2021). Probabilistic forecasting for geosteering
> in fluvial successions using a generative adversarial network.
> *First Break*, 39(7).

> Alyaev, S., et al. (2023). Optimal Sequential Decision-Making in
> Geosteering: A Reinforcement Learning Approach. arXiv:2310.04772.

> Alyaev, S., et al. (2025). High-precision geosteering via
> reinforcement learning and particle filters. *Computational
> Geosciences*. DOI: 10.1007/s10596-025-10352-y

**Evolução da pesquisa:**

1. **2019 — Decision Support:** Sistema probabilístico multi-target
   que assimila medidas em tempo real para atualizar modelos de terra
   e recomendar decisões de steering

2. **2021 — Deep Learning:** Duas inovações:
   - GAN (Generative Adversarial Network) para representação de
     modelos de terra complexos
   - Forward DNN (FDNN) como surrogate do simulador EM

3. **2021 — Forecasting com GAN:** Uso de GANs para previsão
   probabilística da geologia à frente da broca em sucessões fluviais

4. **2022 — Inversão Multi-Modal:** DL para inversão direta e
   simultânea de múltiplos perfis geofísicos

5. **2022 — Caso de Campo (Goliat):** Aplicação do workflow com
   quantificação de incerteza no campo de Goliat (Mar de Barents)

6. **2023 — Reinforcement Learning:** Formulação do geosteering como
   problema de decisão sequencial, resolvido via RL

7. **2025 — RL + Particle Filters:** Combinação de RL com filtros de
   partículas para geosteering de alta precisão

**Relevância para o pipeline:**
- C66–C73 (Seção 7 — Geosteering): A arquitetura de inferência
  realtime do pipeline é inspirada no workflow de Alyaev et al.
- C4 (INFERENCE_MODE = "realtime"): O dual-mode offline/realtime
  é motivado pela necessidade de inversão em tempo real
- C36A (5 arquiteturas geosteering): WaveNet, Causal_Transformer,
  Informer, Mamba_S4, Encoder_Forecaster são projetadas para
  inferência causal (somente dados passados) como requerido para
  geosteering em tempo real

### 5.2 Surrogate Models para Simulação EM

O conceito de usar redes neurais como substitutos (surrogates) do
simulador EM forward é central tanto para Alyaev et al. quanto para
o pipeline v5.0.15:

- **Forward DNN (FDNN):** Rede treinada para mapear
  parâmetros_de_modelo → resposta_EM, substituindo o simulador
  Fortran em aplicações que requerem milhares de avaliações
  (inversão bayesiana, MCMC)

- **Inverse DNN:** Rede treinada para mapear
  resposta_EM → parâmetros_de_modelo, que é exatamente o que o
  pipeline v5.0.15 implementa

A loss **Encoder-Decoder** (#21, Araya-Polo et al., 2018) do
pipeline combina ambos: o FDNN congelado é usado como "verificador
físico" durante o treinamento do Inverse DNN.

---

## 6. Redes Neurais Informadas por Física (PINNs)

### 6.1 Framework PINN Original

> Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
> Physics-informed neural networks: A deep learning framework for
> solving forward and inverse problems involving nonlinear partial
> differential equations. *Journal of Computational Physics*, 378,
> 686–707.

Os PINNs incorporam as equações diferenciais da física diretamente
na função de perda da rede neural, forçando a solução aprendida a
satisfazer as leis físicas (equações de Maxwell, no caso EM). A
função de perda típica de um PINN é:

```
L = L_data + λ_physics * L_physics
```

onde L_data é o misfit dados-predição e L_physics mede a violação
das equações governantes (resíduo da PDE).

**Relevância para o pipeline:**
- C13 (FLAGS PINNs): 7 FLAGS dedicadas a treinamento physics-informed
- Losses #14–#17 (geofísicas): Incorporam constraints físicos
  (suavidade, penalização de oscilação, consistency de anisotropia)
- Loss #21 (Encoder-Decoder): Usa forward model neural como
  "verificador de física"
- Loss #24 (Cross-Gradient): Structural constraint que força
  co-localização de transições ρh ↔ ρv

### 6.2 PINNs para Mecânica Computacional

> Bai, J., Jeong, H., Batuwatta-Gamage, C. P., et al. (2022).
> An introduction to programming Physics-Informed Neural Network-based
> computational solid mechanics. arXiv:2210.09060v4.

Embora focado em mecânica dos sólidos, este artigo fornece uma
introdução prática à programação de PINNs com TensorFlow/Keras,
incluindo:
- Implementação de condições de contorno via "hard constraints"
  (modificação da arquitetura) vs. "soft constraints" (penalização)
- Estratégias de amostragem adaptativa de pontos de colocação
- Técnicas de balanceamento entre termos da loss

**Relevância para o pipeline:** Referência de implementação para as
FLAGS de PINNs em C13.

### 6.3 PINNs para Inversão de Resistividade Anisotrópica (Morales et al., 2025)

> Morales, M. M., et al. (2025). Anisotropic resistivity estimation and
> uncertainty quantification from borehole triaxial electromagnetic induction
> measurements: Gradient-based inversion and physics-informed neural network.
> *Computers & Geosciences*, 196, 105786.

Este é o primeiro trabalho publicado a aplicar PINNs especificamente à inversão
de resistividade anisotrópica, comparando com inversão por gradiente e solução
analítica. Ver **Seção 4.4** para análise detalhada.

**Contribuição ao estado-da-arte das PINNs:**

1. **PINN sem labels diretos:** A rede nunca recebe C_sh ou R_ss como
   targets. Aprende exclusivamente pela consistência com as equações de Klein
   (1993) — um caso de "unsupervised physics-informed learning"

2. **Constraint layer como hard constraint:** Em vez de penalizar violações
   de limites na loss (soft constraint), Morales usa sigmoid/ReLU como
   camada final (hard constraint), garantindo fisicalidade por construção

3. **Speedup de ~10⁶:** Demonstra a viabilidade de PINNs para aplicações
   real-time em LWD/geosteering — inferência em ~0.5 ms vs minutos

4. **UQ nativa:** O método de perturbação ensemble é computacionalmente
   viável apenas porque a PINN é rápida (~0.5 ms × 1000 = ~0.5 s total)

### 6.4 Review de PINNs em Sistemas de Energia Subsuperfície

> Latrach, A., Malki, M. L., Morales, M., Mehana, M., & Rabiei, M. (2024).
> A critical review of physics-informed machine learning applications in
> subsurface energy systems. *Geoenergy Science and Engineering*, 212938.

> Karniadakis, G. E., et al. (2021). Physics-informed machine learning.
> *Nature Reviews Physics*, 3(6), 422–440.

Estas duas revisões fornecem panorama abrangente das aplicações de PINNs
em ciências da Terra, incluindo reservatórios, sequestro de CO₂, e
geomecânica. Karniadakis et al. (2021) é a referência fundacional que
cunhou o termo "physics-informed machine learning" no contexto mais amplo.

---

## 7. Arquiteturas de Deep Learning — Referências das 44 Implementações

O pipeline v5.0.15 implementa 44 arquiteturas de rede neural, organizadas
em famílias. Cada arquitetura tem um ou mais papers de referência.

### 7.1 Redes Convolucionais (CNNs)

#### 7.1.1 ResNet (Residual Networks) — 3 variantes

> He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning
> for Image Recognition. In *CVPR* (pp. 770–778).

O ResNet introduziu as **conexões residuais** (skip connections), que
permitem o treinamento de redes muito profundas ao resolver o problema
de degradação do gradiente. A ideia central é aprender resíduos
F(x) = H(x) - x em vez da transformação completa H(x):

```
output = F(x) + x  (residual connection)
```

O pipeline implementa três variantes:
- **ResNet_18** (C28): 18 camadas, blocos BasicBlock (2 conv 3×1).
  **Arquitetura DEFAULT** do pipeline por ser o melhor equilíbrio
  entre capacidade e eficiência para inversão 1D
- **ResNet_34** (C28): 34 camadas, blocos BasicBlock. Para modelos
  mais complexos com mais camadas geológicas
- **ResNet_50** (C28): 50 camadas, blocos BottleneckBlock (1×1, 3×1, 1×1).
  Maior capacidade, inspirado em aplicações de visão computacional

**Adaptação para inversão EM:** Convoluções 2D → Conv1D ao longo da
dimensão de profundidade (N_MEDIDAS = 600). Strides = 1 e padding = "same"
preservam a dimensão temporal. Para modo causal, padding = "causal".

#### 7.1.2 ConvNeXt

> Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S.
> (2022). A ConvNet for the 2020s. In *CVPR*.

ConvNeXt moderniza a arquitetura CNN tradicional adotando princípios de
design dos Transformers:
- Convoluções depthwise separáveis (como em MobileNet)
- Kernels grandes (7×7, adaptados para 7×1 no pipeline)
- Layer Normalization em vez de Batch Normalization
- Ativação GELU em vez de ReLU
- Fewer activation functions e normalization layers

**Relevância:** Alternativa moderna ao ResNet com melhor trade-off
computação/precisão. Implementada em C28.

#### 7.1.3 InceptionNet / InceptionTime

> Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
> Rethinking the Inception Architecture for Computer Vision. In *CVPR*.

> Fawaz, H. I., Lucas, B., Forestier, G., et al. (2020). InceptionTime:
> Finding AlexNet for Time Series Classification. *Data Mining and
> Knowledge Discovery*, 34, 1936–1962.

O módulo Inception aplica convoluções com **múltiplos tamanhos de kernel
em paralelo** (1×1, 3×1, 5×1, 7×1) e concatena os resultados. Isso
permite capturar padrões em múltiplas escalas espaciais simultaneamente.

InceptionTime adapta especificamente o Inception para **séries temporais**,
com:
- Ensemble de múltiplos módulos Inception
- Conexões residuais entre módulos
- Demonstração de desempenho state-of-the-art em 128 datasets do
  UCR Archive

**Relevância para o pipeline:** Multi-scale feature extraction é
particularmente útil para inversão EM onde transições de resistividade
ocorrem em escalas variadas (fronteiras abruptas vs. gradientes suaves).
Implementadas em C28.

#### 7.1.4 CNN_1D

Arquitetura convolucional genérica 1D com 6 camadas simétricas
[32, 64, 128, 128, 64, 32] filtros. Sem paper específico — é uma
arquitetura baseline simples e rápida. Implementada em C29.

### 7.2 Convolucionais Temporais

#### 7.2.1 TCN (Temporal Convolutional Network)

> Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation
> of Generic Convolutional and Recurrent Networks for Sequence Modeling.
> arXiv:1803.01271.

O TCN é uma arquitetura **nativamente causal** que combina:
- **Convoluções causais:** output[t] depende apenas de input[≤t]
- **Dilatação exponencial:** dilation = 2^i para camada i, criando
  campo receptivo exponencialmente grande sem aumentar parâmetros
- **Conexões residuais:** estabilizam o treinamento profundo

O campo receptivo de um TCN com L camadas e dilation = 2^i é:

```
RF = 1 + 2 * (kernel_size - 1) * (2^L - 1)
```

Para kernel_size = 3 e L = 8: RF = 1 + 2 × 2 × 255 = 1021 > 600 (N_MEDIDAS).

**Relevância:** **Nativo causal** — ideal para geosteering realtime.
O TCN é uma das 6 arquiteturas nativamente causais do pipeline.
Implementado em C30.

#### 7.2.2 WaveNet

> van den Oord, A., et al. (2016). WaveNet: A Generative Model for
> Raw Audio. arXiv:1609.03499.

WaveNet foi originalmente proposto para geração de áudio, mas sua
arquitetura de **convoluções causais dilatadas com gated activations**
é altamente adequada para sequências temporais:
- Convolução causal (sem acesso a dados futuros)
- Dilatação exponencial (campo receptivo amplo)
- Gated activation: z = tanh(W_f * x) ⊙ σ(W_g * x)
- Conexões residuais e skip connections

**Relevância para o pipeline:** WaveNet é uma das 5 arquiteturas
dedicadas a **geosteering** (C36A), projetada para inversão causal de
alta qualidade. Sua estrutura gated é particularmente eficaz para
capturar transições abruptas de resistividade (fronteiras de camadas).

### 7.3 Redes Recorrentes (RNNs)

#### 7.3.1 LSTM (Long Short-Term Memory)

> Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
> *Neural Computation*, 9(8), 1735–1780.

O LSTM resolve o problema de desvanecimento de gradiente das RNNs
tradicionais através de um sistema de **gates** (portões):
- **Input gate:** controla quanta nova informação entra na célula
- **Forget gate:** controla quanta informação anterior é descartada
- **Output gate:** controla quanta informação da célula é exposta
- **Cell state:** memória de longo prazo que flui com modificações mínimas

**Relevância:** LSTM é **nativamente causal** — cada saída depende
apenas de entradas passadas e presentes. Com return_sequences=True,
preserva N_MEDIDAS. Implementado em C31.

#### 7.3.2 BiLSTM (Bidirectional LSTM)

> Schuster, M., & Paliwal, K. K. (1997). Bidirectional Recurrent Neural
> Networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.

BiLSTM processa a sequência em ambas as direções (frente e trás),
concatenando as representações. Isso permite que cada posição tenha
contexto do passado E do futuro.

**Relevância:** **Incompatível com modo causal** — usa informação
futura. Adequado apenas para inversão offline. Implementado em C31.

### 7.4 Transformers

#### 7.4.1 Transformer Original

> Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is
> All You Need. In *NeurIPS*.

O mecanismo de **self-attention** permite que cada posição da sequência
atenda a todas as outras posições, capturando dependências de longo
alcance:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Multi-Head Attention executa h projeções paralelas, cada uma com
dimensão d_k = d_model / h. O Transformer completo combina multi-head
attention com feed-forward networks, residual connections e layer
normalization.

**Relevância:** Base de 8 arquiteturas Transformer do pipeline.
Adaptável para modo causal via máscara triangular inferior.

#### 7.4.2 TFT (Temporal Fusion Transformer)

> Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal
> Fusion Transformers for Interpretable Multi-horizon Time Series
> Forecasting. *International Journal of Forecasting*, 37(4), 1748–1764.

O TFT introduz componentes especializados para séries temporais:
- **GRN (Gated Residual Network):** Processamento adaptativo com
  skip connection gated
- **VSN (Variable Selection Network):** Seleção automática de
  features relevantes por timestep
- **Multi-head attention interpretável:** Pesos de atenção que
  indicam quais timesteps são mais importantes
- **Quantile regression:** Previsão de múltiplos quantis para
  estimativa de incerteza

**Relevância:** O pipeline implementa duas versões:
- Simple_TFT (C34): Versão simplificada sem GRN/VSN
- TFT completo (C34): Versão full com GRN + VSN

O VSN é particularmente relevante para inversão EM pois permite
que a rede aprenda automaticamente quais componentes do tensor H
são mais informativos para cada posição de profundidade.

#### 7.4.3 Informer

> Zhou, H., Zhang, S., Peng, J., et al. (2021). Informer: Beyond
> Efficient Transformer for Long Sequence Time-Series Forecasting.
> In *AAAI*.

O Informer resolve o problema de complexidade O(N²) do Transformer
original via:
- **ProbSparse self-attention:** Seleção dos top-k queries mais
  informativos, reduzindo complexidade para O(N log N)
- **Self-attention distilling:** Redução progressiva do comprimento
  da sequência entre camadas
- **Generative decoder:** Previsão direta de múltiplos passos

**Relevância:** Eficiência para sequências longas (N_MEDIDAS = 600).
Implementado em C34 como arquitetura geosteering.

#### 7.4.4 PatchTST

> Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
> A Time Series is Worth 64 Words: Long-term Forecasting with
> Transformers. In *ICLR*.

PatchTST divide a série temporal em **patches** (subsegmentos) e
aplica o Transformer sobre estes patches:
- **Patching:** Segmentos de comprimento P com stride S
- **Channel-independence:** Cada variável processada independentemente
- Redução drástica do comprimento da sequência de entrada
  (N/S tokens vs. N tokens)

**Relevância:** Eficiência computacional. Para N_MEDIDAS = 600 com
patch_size = 16 e stride = 8: 75 tokens (vs. 600). Implementado em C34.

#### 7.4.5 Autoformer

> Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer:
> Decomposition Transformers with Auto-Correlation for Long-Term Series
> Forecasting. In *NeurIPS*.

O Autoformer combina:
- **Series decomposition:** Separação trend-seasonal em cada camada
  (moving average para trend, resíduo para seasonal)
- **Auto-Correlation mechanism:** Substitui self-attention por
  correlação no domínio da frequência, capturando dependências
  periódicas eficientemente

**Relevância:** A decomposição trend-seasonal pode ser análoga à
separação de componentes de resistividade (tendência regional vs.
variações locais em fronteiras). Implementado em C34.

#### 7.4.6 iTransformer

> Liu, Y., Hu, T., Li, H., Lu, J., Zhao, H., & Long, M. (2024).
> iTransformer: Inverted Transformers Are Effective for Time Series
> Forecasting. In *ICLR*.

O iTransformer **inverte** a dimensão de atenção:
- Transformer convencional: atenção ao longo do **tempo**
- iTransformer: atenção ao longo das **variáveis** (features)

Cada variável (feature) é tratada como um "token", e a atenção
captura correlações entre variáveis em vez de dependências temporais.
A dimensão temporal é processada por um feed-forward network.

**Relevância:** Para inversão EM, as correlações entre componentes
do tensor H (Hxx, Hzz) e geosinais são tão importantes quanto as
dependências espaciais. Implementado em C34.

#### 7.4.7 Causal Transformer

Variante do Transformer com máscara causal triangular inferior
(GPT-style), garantindo que a atenção em cada posição acesse apenas
posições anteriores. **Nativamente causal**, adequado para geosteering
realtime. Implementado em C36A.

### 7.5 Decomposição e Análise de Séries

#### 7.5.1 N-BEATS

> Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020).
> N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series
> Forecasting. In *ICLR*.

N-BEATS decompõe a previsão em **blocos empilhados**, cada um com
uma expansão de base (basis expansion) que gera backcast e forecast:

```
forecast_block = θ_f^T · V_f  (projeção em base de funções)
backcast_block = θ_b^T · V_b
```

As bases podem ser:
- **Generic:** Projeção linear aprendida
- **Trend:** Polinômios de baixa ordem
- **Seasonality:** Harmônicos de Fourier

**Relevância:** A decomposição interpretável pode separar componentes
de resistividade em tendência (regional) e variação local (fronteiras).
Implementado em C35.

#### 7.5.2 N-HiTS

> Challu, C., Olivares, K. G., Oreshkin, B. N., et al. (2023).
> N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.
> In *AAAI*.

N-HiTS estende N-BEATS com **interpolação hierárquica**, onde diferentes
blocos operam em diferentes resoluções temporais via pooling:
- Blocos de alta resolução capturam detalhes finos
- Blocos de baixa resolução capturam tendências de longo prazo
- Interpolação reconstrói o sinal na resolução original

**Relevância:** Multi-resolução é análoga a capturar variações de
resistividade em diferentes escalas (micro-camadas vs. formações).
Implementado em C35.

### 7.6 Modelos de Espaço de Estados

#### 7.6.1 S4 (Structured State Space Sequence Model)

> Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long
> Sequences with Structured State Spaces. In *ICLR*.

S4 modela sequências como um sistema dinâmico linear:

```
x'(t) = Ax(t) + Bu(t)
y(t) = Cx(t) + Du(t)
```

onde A é uma matriz estruturada (HiPPO — High-order Polynomial
Projection Operator) que captura eficientemente a história da
sequência. A discretização permite convolução eficiente no
domínio da frequência: O(N log N).

#### 7.6.2 Mamba

> Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling
> with Selective State Spaces. In *ICLR*.

Mamba estende S4 com **seletividade dependente da entrada**:
- Parâmetros B, C, Δ são funções da entrada (não fixos)
- Mecanismo de seleção permite que o modelo "ignore" inputs
  irrelevantes e "lembre" informações importantes
- Complexidade **O(N)** linear no comprimento da sequência
- Hardware-aware algorithm para eficiência em GPU

**Relevância para o pipeline:** Mamba/S4 é uma das 5 arquiteturas
geosteering (C36A), oferecendo a **menor latência** de todas as
44 arquiteturas — O(1) por novo ponto de dados na inferência
incremental. Ideal para geosteering realtime com restrição de
latência < 100 ms.

### 7.7 Operadores Neurais

#### 7.7.1 FNO (Fourier Neural Operator)

> Li, Z., Kovachki, N., Azizzadenesheli, K., et al. (2021). Fourier
> Neural Operator for Parametric Partial Differential Equations.
> In *ICLR*.

O FNO aprende mapeamentos entre espaços de funções (operadores)
usando camadas no domínio da frequência:

```
v_{l+1} = σ(W_l · v_l + K_l(v_l))
K_l(v) = F^{-1}(R_l · F(v))
```

onde F é a FFT e R_l é uma transformação linear aprendida no
espaço de Fourier. Isso permite capturar padrões globais com
custo O(N log N).

**Relevância:** O mapeamento campo_EM → resistividade é um
mapeamento entre funções — exatamente o cenário para operadores
neurais. FNO pode capturar a relação integral entre medidas e
modelo. Implementado em C36. Nota: a FFT global viola causalidade
estrita (adaptável com restrição espectral).

#### 7.7.2 DeepONet (Deep Operator Network)

> Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021).
> Learning nonlinear operators via DeepONet based on the universal
> approximation theorem of operators. *Nature Machine Intelligence*, 3,
> 218–229.

DeepONet usa duas sub-redes:
- **Branch net:** Processa a função de entrada (medidas EM)
- **Trunk net:** Processa as coordenadas de avaliação (profundidades)
- Saída: produto interno dos outputs de branch e trunk

**Relevância:** Fundamentação teórica sólida via teorema de
aproximação universal de operadores (Chen & Chen, 1995).
Implementado em C36. **Incompatível com modo causal** (offline only).

### 7.8 U-Nets e Variantes (14 variantes)

#### 7.8.1 U-Net Original

> Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net:
> Convolutional Networks for Biomedical Image Segmentation. In *MICCAI*
> (pp. 234–241). Springer.

U-Net é uma arquitetura encoder-decoder com **skip connections**
simétricas entre encoder e decoder:

```
Encoder: x → [↓32] → [↓64] → [↓128] → [↓256]  (downsample)
                ↕          ↕          ↕          ↕
Decoder:         [↑128] ← [↑64] ← [↑32] ← [↑16]  (upsample)
                          → output
```

As skip connections permitem que detalhes de alta resolução do
encoder sejam preservados no decoder, resultando em segmentação
precisa de bordas.

**Relevância para inversão EM:** A estrutura encoder-decoder é
análoga à compressão e reconstrução de perfis de resistividade.
As skip connections preservam transições abruptas (fronteiras de
camadas). No pipeline, Conv1D substitui Conv2D, e strides/upsampling
operam ao longo da dimensão de profundidade.

O pipeline implementa 14 variantes de U-Net (C33) combinando:
- **Backbones:** ResNet_18, ResNet_34, ResNet_50, ConvNeXt,
  InceptionNet, EfficientNet
- **Attention:** Com e sem Attention Gates

#### 7.8.2 Attention U-Net

> Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look
> for the Pancreas. arXiv:1804.03999.

Adiciona **Attention Gates (AG)** nas skip connections da U-Net,
permitindo que o decoder "foque" nas regiões mais relevantes do
encoder:

```
AG(x_skip, g_decoder) = σ(ψ(σ(W_x · x + W_g · g + b))) ⊙ x_skip
```

O gate aprende a ponderar cada posição da skip connection com base
no contexto do decoder.

**Relevância:** Para inversão EM, as AGs permitem que a rede focalize
nas regiões próximas a fronteiras de camadas (onde a informação é mais
crítica) e atenue regiões homogêneas. Implementado em C33.

#### 7.8.3 EfficientNet (backbone)

> Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling
> for Convolutional Neural Networks. In *ICML*.

EfficientNet propõe um método de **compound scaling** que escala
uniformemente profundidade, largura e resolução:

```
depth = α^φ, width = β^φ, resolution = γ^φ
α · β² · γ² ≈ 2
```

Usa blocos **MBConv** (Mobile Inverted Bottleneck Conv) com
Squeeze-and-Excitation.

**Relevância:** Backbone eficiente para U-Nets quando memória GPU é
limitada (Google Colab). Implementado em C33 (UNet_EfficientNet,
UNet_Attention_EfficientNet).

### 7.9 Mecanismos de Atenção

#### 7.9.1 Squeeze-and-Excitation (SE) Networks

> Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
> In *CVPR*.

O bloco SE recalibra as respostas de cada canal (filtro) via
atenção por canal:

```
SE(x) = x ⊙ σ(W_2 · ReLU(W_1 · GAP(x)))
```

onde GAP é Global Average Pooling e W_1, W_2 são transformações
lineares com redução de dimensão (ratio r).

**Relevância:** Implementado como USE_SE_BLOCK em C8 (FLAGS de skip
connections). Permite que a rede aprenda automaticamente quais filtros
são mais relevantes para cada input — útil quando diferentes componentes
EM têm importância variável. Propagado a todas as 44 arquiteturas via
C27 (blocos reutilizáveis).

### 7.10 Arquiteturas Híbridas e Especializadas

#### 7.10.1 CNN-LSTM e CNN-BiLSTM Encoder-Decoder

Combinam CNNs para extração de features locais com LSTMs para
modelagem de dependências sequenciais:

- **CNN-LSTM** (C32): Conv1D blocks → LSTM layers → TimeDistributed Dense.
  Adaptável para modo causal
- **CNN-BiLSTM-ED** (C32): Encoder Conv1D+BiLSTM, Decoder LSTM com
  repeat vector. **Incompatível com modo causal**

#### 7.10.2 DNN (Deep Neural Network)

Rede fully-connected simples com TimeDistributed Dense layers.
Baseline minimalista para comparação. Implementado em C36.

#### 7.10.3 Geophysical Attention

Mecanismo de atenção customizado para dados geofísicos, combinando
self-attention com positional encoding baseado em profundidade
(em vez de posição sequencial genérica). Implementado em C36.

#### 7.10.4 Encoder-Forecaster (Seq2Seq)

Arquitetura Sequence-to-Sequence com encoder que comprime a janela
de entrada e forecaster que gera a predição. **Nativa causal** com
look-ahead explícito para geosteering (C36A). Inspirada na arquitetura
seq2seq de Sutskever et al. (2014).

---

## 8. Funções de Perda — Referências das 25 Losses

### 8.1 Losses Genéricas (13)

As 13 losses genéricas (MSE, RMSE, MAE, MBE, RSE, RAE, MAPE, MSLE,
RMSLE, NRMSE, RRMSE, Huber, Log-Cosh) são métricas padrão da
literatura de machine learning e não requerem citações específicas.
Operam no domínio log10 (TARGET_SCALING = "log10") conforme definido
em C6. Referência geral:

> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
> MIT Press. Capítulo 8: Optimization for Training Deep Models.

### 8.2 Losses Geofísicas (4)

As 4 losses geofísicas (log_scale_aware, adaptive_log_scale,
robust_log_scale, adaptive_robust_log_scale) são desenvolvidas
especificamente para o pipeline v5.0.15, combinando:

- **InterfaceError:** Penalização extra em regiões próximas a
  fronteiras de camadas (detectadas via gradiente do target)
- **OscillationPenalty:** Penalização de oscilações espúrias na
  predição (smoothness constraint)
- **UnderestimationPenalty:** Penalização assimétrica que pune
  subestimação mais que superestimação (motivada fisicamente:
  subestimar resistividade é mais perigoso para geosteering)
- **Mecanismo gangorra (adaptive_*):** Beta efetivo varia com o
  nível de ruído (curriculum noise)

### 8.3 Losses Geosteering v5.0.7 (2)

#### 8.3.1 Probabilistic NLL

Loss de Negative Log-Likelihood gaussiana para estimativa de
incerteza, inspirada em:

> Nix, D. A., & Weigend, A. S. (1994). Estimating the mean and
> variance of the target probability distribution. In *IEEE
> International Conference on Neural Networks*.

#### 8.3.2 Look-Ahead Weighted

Loss com ponderação exponencial que penaliza mais erros em posições
"à frente" na sequência, forçando a rede a antecipar transições
geológicas. Conceito original para geosteering.

### 8.4 Losses Avançadas v5.0.15 (6)

#### 8.4.1 DILATE (Loss #20)

> Le Guen, V., & Thome, N. (2019). Shape and Time Distortion Loss
> for Training Deep Time Series Forecasting Models. In *NeurIPS*.

DILATE combina **soft-DTW** (Dynamic Time Warping suavizado) para
alinhamento temporal com MSE de gradientes para preservação de forma:

```
L_DILATE = α · softDTW(y_ds, ŷ_ds) / N_ds + (1-α) · MSE(dy/dz, dŷ/dz)
```

O softDTW permite que a loss tolere **deslocamentos espaciais** entre
predição e target — crucial para inversão EM onde a posição exata de
uma fronteira de camada pode estar ligeiramente deslocada.

**Implementação:** Soft-DTW implementado em TF puro via programação
dinâmica. Downsampling por fator 10 para reduzir complexidade O(N²).

#### 8.4.2 Encoder-Decoder Loss (#21)

> Araya-Polo, M., Jennings, J., Adler, A., & Dahlke, T. (2018).
> Deep-learning tomography. *The Leading Edge*, 37(1), 58–66.

Combina MSE do espaço de parâmetros com MSE do espaço de dados via
um forward model neural congelado:

```
L = (1-λ) · MSE(y, ŷ) + λ · MSE(F(ŷ), F(y))
```

onde F(·) é uma rede neural pré-treinada que mapeia
resistividade → resposta EM (surrogate do simulador Fortran).

**Relevância:** Força consistência física — a predição de
resistividade deve, quando re-simulada, reproduzir as medidas EM
de entrada.

#### 8.4.3 Multi-Task Learned Loss (#22)

> Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning
> Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.
> In *CVPR*.

Aprende automaticamente os pesos relativos de múltiplas sub-losses
via **incerteza homoscedástica**:

```
L = Σ_k [ exp(-2s_k)/2 · L_k + s_k ]
```

onde s_k = log(σ_k) são parâmetros treináveis. Tarefas com maior
incerteza recebem peso menor automaticamente.

**Relevância:** Quando o pipeline usa múltiplas losses simultaneamente
(ex: MSE + Interface + Oscillation), o Multi-Task Learned balanceia
automaticamente os termos sem necessidade de ajuste manual de pesos.

#### 8.4.4 Sobolev H1 Loss (#23)

> Czarnecki, W. M., Osindero, S., Jaderberg, M., Swirszcz, G., &
> Pascanu, R. (2017). Sobolev Training for Neural Networks. In *NeurIPS*.

Penaliza diferenças entre gradientes (derivadas) da predição e do target:

```
L = MSE(y, ŷ) + λ · MSE(dy/dz, dŷ/dz)
```

**Relevância:** O termo de gradiente força a rede a preservar
**transições abruptas** de resistividade em fronteiras de camadas.
Sem este termo, redes tendem a suavizar excessivamente as transições.

#### 8.4.5 Cross-Gradient Loss (#24)

> Gallardo, L. A., & Meju, M. A. (2003). Characterization of
> heterogeneous near-surface materials by joint 2D inversion of dc
> resistivity and seismic data. *Geophysical Research Letters*, 30(13),
> 1658.

O cross-gradient structural constraint força **co-localização** de
transições em diferentes propriedades:

```
τ(z) = (dρh/dz) × (dρv/dz)
L = MSE(y, ŷ) + λ · MSE(τ_true, τ_pred)
```

τ > 0 indica transição concordante (ρh e ρv mudam juntas).
τ ≈ 0 indica transição discordante ou estabilidade.

**Relevância:** Em formações anisotrópicas, as fronteiras de camadas
devem produzir transições simultâneas em ρh e ρv. O cross-gradient
garante que a rede respeite esta constraint estrutural.

#### 8.4.6 Spectral Loss (#25)

> Jiang, L., et al. (2021). Focal Frequency Loss for Image
> Reconstruction and Synthesis. In *ICCV*.

Penaliza diferenças no domínio da frequência:

```
L = MSE(y, ŷ) + λ · MSE(log(1+|FFT(y)|), log(1+|FFT(ŷ)|))
```

**Relevância:** Captura e penaliza **oscilações espúrias de alta
frequência** na predição, que são um artefato comum em inversões
de resistividade em zonas de alta resistividade.

---

## 9. Estratégias de Treinamento e Augmentação de Dados

### 9.1 Curriculum Learning

> Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
> Curriculum Learning. In *ICML*.

Curriculum learning apresenta dados ao modelo em **ordem crescente
de dificuldade**, análogo ao currículo escolar. No pipeline v5.0.15:

- **Noise curriculum:** Treina primeiro com dados limpos ou ruído baixo,
  depois aumenta progressivamente o nível de ruído
- **Tier-switching (P5):** EpochTierCallback troca entre datasets com
  níveis crescentes de ruído a cada K épocas
- **On-the-fly noise (P7):** Nível de ruído aumenta continuamente via
  UpdateNoiseLevelCallback

**Relevância:** C12 (43 tipos de ruído), C24 (curriculum scheduling),
C40 (EpochTierCallback), C44 (curriculum learning dedicado).

### 9.2 Data Augmentation com Noise Injection

O pipeline implementa 43 tipos de ruído organizados em 5 famílias
(C12, NOISE_CATALOG.md):

1. **Ruído instrumental (8):** Gaussian, phase, multiplicative,
   quantization, thermal, ADC, clock jitter, cross-talk
2. **Ruído ambiental (8):** Telluric, powerline, mud motor,
   drillstring vibration, temperature drift, BHA offset, tool
   eccentricity, standoff
3. **Ruído geológico (9):** Thin bed, invasion, shoulder bed,
   anisotropy variation, rugosity, formation heterogeneity,
   washout, fracture, vugs
4. **Ruído de processamento (9):** Decimation, interpolation,
   baseline drift, calibration error, depth matching, stacking,
   convolution, deconvolution, filtering
5. **Ruído composto (9):** LWD standard, LWD harsh, offshore,
   HTHP, carbonate, laminated, salt proximity, unconventional, custom

**Fundamentação:** Noh et al. (2021/2023) [A3/A2] demonstram que
a injeção de ruído durante o treinamento é essencial para robustez
da inversão DL a medidas reais ruidosas.

### 9.3 Otimizadores

#### 9.3.1 Adam

> Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic
> Optimization. In *ICLR*.

Adam combina momentum (média móvel dos gradientes) com RMSProp
(média móvel dos quadrados dos gradientes) para adaptação automática
da taxa de aprendizado por parâmetro. O pipeline usa LEARNING_RATE = 1e-3
(default do Adam) com possibilidade de warmup e decay.

#### 9.3.2 AdamW

> Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay
> Regularization. In *ICLR*.

AdamW "desacopla" o weight decay da atualização do gradiente,
resolvendo o problema de que L2 regularization em Adam é diferente
de weight decay verdadeiro. Disponível como OPTIMIZER = "adamw" em C9.

### 9.4 Learning Rate Scheduling

#### 9.4.1 Cosine Annealing

> Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient
> Descent with Warm Restarts. In *ICLR*.

Decaimento cosseno suave do learning rate:

```
lr(t) = lr_min + 0.5 · (lr_max - lr_min) · (1 + cos(π · t / T))
```

Implementado como LR_SCHEDULER_TYPE = "cosine" em C9, com variante
"warmup_cosine" que inclui warmup linear nas primeiras épocas.

#### 9.4.2 ReduceLROnPlateau

Callback nativo do Keras que reduz o LR quando a métrica monitorada
(val_loss) para de melhorar. **Default** do pipeline por ser
adaptativo — só reduz quando necessário.

### 9.5 Hyperparameter Optimization

> Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).
> Optuna: A Next-generation Hyperparameter Optimization Framework.
> In *KDD*.

Optuna é o framework de otimização de hiperparâmetros do pipeline
(C18, C38, C45), oferecendo:
- **Tree-structured Parzen Estimator (TPE):** Sampler bayesiano
  para exploração eficiente do espaço de hiperparâmetros
- **Pruning:** Early stopping de trials não-promissores via
  MedianPruner ou PercentilePruner
- **Dashboard:** Visualização interativa de trials e importância
  de hiperparâmetros

**Relevância:** C18 (FLAGS Optuna), C38 (config study), C45 (loop Optuna).

### 9.6 Normalização

#### 9.6.1 Batch Normalization

> Ioffe, S., & Szegedy, S. (2015). Batch Normalization: Accelerating
> Deep Network Training by Reducing Internal Covariate Shift. In *ICML*.

Normaliza ativações por mini-batch: z_norm = (z - μ_B) / √(σ²_B + ε).
Usado como default na maioria das arquiteturas CNN do pipeline.

#### 9.6.2 Layer Normalization

> Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization.
> arXiv:1607.06450.

Normaliza ao longo da dimensão de features (em vez de batch).
Preferido em Transformers e RNNs onde batch size pode ser pequeno.

### 9.7 Regularização

#### 9.7.1 Dropout

> Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., &
> Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural
> Networks from Overfitting. *JMLR*, 15(1), 1929–1958.

Desativa aleatoriamente uma fração p dos neurônios durante o treinamento,
forçando redundância na representação aprendida. O pipeline usa
DROPOUT_RATE = 0.1 (default, C11). MC Dropout para incerteza (Seção 10).

---

## 10. Quantificação de Incerteza em DL para Geofísica

### 10.1 MC Dropout (Monte Carlo Dropout)

> Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian
> Approximation: Representing Model Uncertainty in Deep Learning.
> In *ICML*.

Gal e Ghahramani demonstram que manter o Dropout ativado durante a
**inferência** (não apenas treinamento) e executar múltiplas predições
é equivalente a uma aproximação de inferência variacional bayesiana:

```
μ(x) = (1/T) Σ_{t=1}^{T} f_θ_t(x)     # média das T predições
σ²(x) = (1/T) Σ_{t=1}^{T} f_θ_t(x)² - μ(x)²  # variância epistêmica
```

**Relevância para o pipeline:**
- C4: USE_MC_DROPOUT_INFERENCE flag
- C11: DROPOUT_RATE e USE_MC_DROPOUT_INFERENCE
- C7: OUTPUT_CHANNELS = 4 para saída (μ_ρh, μ_ρv, σ_ρh, σ_ρv)
- C66–C73: Incerteza automática em modo geosteering realtime

A incerteza é particularmente importante em geosteering: decisões
de steering com alta incerteza devem ser sinalizadas ao operador.

### 10.2 NLL Gaussiana

A loss probabilistic_nll (#18) treina a rede para predizer
simultaneamente média e variância:

```
L = log(σ) + (y - μ)² / (2σ²)
```

Isso permite estimativa de incerteza **aleatória** (dados) em
adição à incerteza **epistêmica** (modelo) do MC Dropout.

### 10.3 Incerteza em Inversão EM via Perturbação Ensemble (Morales et al., 2025)

Conforme detalhado na Seção 4.4, Morales et al. [A5] apresentam um método
prático e eficiente de quantificação de incerteza para inversão EM:

**Método:** Gerar N=1000 realizações com ruído gaussiano aditivo (κ=5% da
variância dos dados) e computar a distribuição dos parâmetros estimados.
A incerteza é quantificada pelo intervalo interpercentil P10-P90.

**Resultados quantitativos por dataset:**

| Dataset | U(Csh) [v/v] | U(Rss) [Ωm] | U(R̂v) [Ωm] | U(R̂h) [Ωm] |
|:--------|:-------------|:-------------|:------------|:------------|
| Sintético 1 | 0.019 | 0.147 | 0.196 | 0.026 |
| Sintético 2 | 0.029 | 0.206 | 0.202 | 0.037 |
| Campo 1 | 0.056 | 0.354 | 0.265 | 0.126 |
| Campo 2 | 0.071 | 0.204 | 0.208 | 0.123 |

**Padrões observados:**
- Csh tem incerteza mais consistente ao longo da profundidade
- Rss superestima em zonas de transição litológica
- R̂h sempre tem menor incerteza que R̂v
- Dados de campo mostram incerteza ~2-4× maior que sintéticos

**Vantagem sobre MC Dropout:** O método ensemble é agnóstico à arquitetura
(não requer Dropout durante inferência) e é ~10⁶× mais rápido que
re-inverter com gradient-based para cada perturbação.

**Relevância para o pipeline:**
- C48-C57 (Avaliação): Implementar avaliação de incerteza via perturbação
  com κ configurável como FLAG (ex: `UQ_NOISE_LEVEL = 0.05`)
- C66-C73 (Geosteering): No modo realtime, inferir N perturbações em
  paralelo para fornecer intervalos de confiança ao operador
- C4: OUTPUT_CHANNELS = 4 (com σ) é complementar — uma rede que prediz
  μ e σ diretamente, combinada com perturbação ensemble, forneceria
  tanto incerteza aleatória (σ) quanto epistêmica (ensemble)

---

## 11. Geosinais e Medidas Direcionais

### 11.1 Definição de Geosinais

Os geosinais são combinações das componentes do tensor de campo
magnético H que realçam informações geológicas específicas. A
referência técnica principal é o documento interno **GeoSphereXTatu**
[T4] e as especificações da ferramenta **GeoSphere HD** [T1].

O pipeline implementa 5 famílias de geosinais (C22):

| Família | Nome | Fórmula | Informação Geológica |
|:--------|:-----|:--------|:---------------------|
| USD | Up-Down Symmetry | (ZZ+XZ)/(ZZ-XZ) × (ZZ-ZX)/(ZZ+ZX) | Fronteiras de camadas |
| UAD | Up-Down Asymmetry | (ZZ+XZ)/(ZZ-XZ) × (ZZ+ZX)/(ZZ-ZX) | Dip e anisotropia |
| UHR | Homogeneous Resistivity | -2ZZ / (XX+YY) | Resistividade bulk |
| UHA | Horizontal Anisotropy | XX / YY | Razão de anisotropia |
| U3DF | 3D Formation | (ZZ+YZ)/(ZZ-YZ) × (ZZ-ZY)/(ZZ+ZY) | Detecção 3D |

Cada família produz **atenuação** (amplitude ratio em dB) e
**phase shift** (diferença de fase em graus) a partir das componentes
complexas do tensor H.

### 11.2 Look-Ahead e Depth of Investigation

> Constable, M. V., et al. (2016). Looking Ahead of the Bit While
> Drilling: From Vision to Reality. *Petrophysics*, 57(5), 426–446.

Este artigo seminal descreve o desenvolvimento do protótipo EMLA
(Electromagnetic Look-Ahead) para detecção de propriedades de
formação à frente da broca:

- Transmissor de baixa frequência a 1.8 m atrás da broca
- 2-3 receptores espaçados no drillstring
- Mesma tecnologia de sensores do look-around commercial (GeoSphere)
- Capacidade de detecção de contrastes de resistividade a vários
  metros à frente da broca

**Relevância:** Inspira a perspectiva P5 (DTB — Distance to Boundary)
e a loss look_ahead_weighted (#19) do pipeline.

### 11.3 Ferramentas de Look-Ahead — Estado da Arte

> Li, G., Wu, Z., Liao, X., et al. (2025). Optimization and Analysis
> of Sensitive Areas for Look-Ahead Electromagnetic Logging-While-Drilling
> Based on Geometric Factors. *Energies*, 18(12), 3014.

> Liu, R., Zhang, W., Chen, W., et al. (2025). Factors and detection
> capability of look-ahead logging while drilling (LWD) tools.
> *Petroleum Science*, 22, 850–867.

Estes artigos recentes (2025) analisam os fatores geométricos e a
capacidade de detecção de ferramentas look-ahead, demonstrando as
limitações e oportunidades para deep learning:

- A capacidade de detecção depende criticamente do espaçamento
  transmissor-receptor, frequência e contraste de resistividade
- Ferramentas convencionais têm zona cega à frente da broca em
  poços de alto ângulo
- O DL pode potencialmente "extrair" informações look-ahead das
  medidas look-around via treinamento com modelos que incluem
  variações à frente da broca

---

## 12. Ferramentas Comerciais LWD

### 12.1 Schlumberger GeoSphere HD

> Schlumberger. *GeoSphere HD 1.0 — High-Definition Reservoir
> Mapping-While-Drilling Service*. Technical Specifications. [T1]

GeoSphere HD é a ferramenta comercial de referência para medidas
UDAR (Ultradeep Azimuthal Resistivity):
- Múltiplos espaçamentos transmissor-receptor (até 30+ metros)
- Múltiplas frequências (tipicamente 6 kHz a 96 kHz)
- Medidas azimutais (0°, 90°, 180°, 270°)
- Profundidade de investigação > 30 metros
- Geosinais USD, UAD, UHR, UHA computados em tempo real

### 12.2 Halliburton EarthStar

A ferramenta EarthStar da Halliburton oferece capacidades similares
ao GeoSphere, com:
- Ultra-deep azimuthal resistivity measurements
- Inversão 2D em tempo real via LOGIX automated geosteering
- Integração com sistemas de geosteering automatizado

### 12.3 Especificações Comparativas

> University of Houston. (2016). *A Survey on Definitions of Some Key
> Service Specs of LWD Deep Azimuthal Resistivity Tools*. [T3]

Este documento compara as definições de especificações-chave entre
fabricantes (Schlumberger, Halliburton, Baker Hughes, Weatherford):

- **DOI (Depth of Investigation):** Definições variam entre fabricantes
  (50% response, 2 skin depths, geometric factor)
- **Resolução vertical:** Determinada pelo espaçamento e frequência
- **Sensibilidade direcional:** Capacidade de detectar azimute e
  distância de fronteiras remotas

### 12.4 Benchmark SDAR

> SDAR Work Group. (2020). *Benchmark Models for Look-Ahead
> Applications*. [T2]

Modelos benchmark padronizados para comparação de ferramentas de
diferentes fabricantes:
- Caso 1: Poço vertical (vertical well)
- Caso 2: Poço desviado (deviated well)
- Caso 3: Camada fina entre camadas (thin-bed in between)

---

## 13. Livros e Textos Fundamentais

### 13.1 Deep Learning — Teoria e Prática

> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
> MIT Press. ISBN 978-0-262-03561-3.

O "livro do Deep Learning" por excelência. Cobre de forma rigorosa:
- Álgebra linear, probabilidade e teoria da informação (Parte I)
- Redes feedforward, CNNs, RNNs, regularização (Parte II)
- Modelos generativos, autoencoders, representações (Parte III)

> Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive
> Into Deep Learning*. [L6] (Disponível no projeto)

Livro interativo com código executável (TensorFlow/PyTorch/JAX).
Cobre de forma prática implementações de CNNs, RNNs, Transformers,
GANs e técnicas modernas de DL.

### 13.2 Séries Temporais com Python

> Joseph, M. (2022). *Modern Time Series Forecasting with Python*.
> Packt. [L4]

Cobre implementações modernas de:
- N-BEATS, N-HiTS, TFT e outros modelos state-of-the-art
- Pipelines de dados para séries temporais
- Avaliação e comparação de modelos

### 13.3 Tuning de Hiperparâmetros

> Owen, L. (2022). *Hyperparameter Tuning with Python*. Packt. [L3]

Cobre estratégias de otimização de hiperparâmetros incluindo:
- Grid search, random search, Bayesian optimization
- Optuna e outros frameworks
- Transfer learning de hiperparâmetros

### 13.4 Fortran Moderno

> Curcic, M. (2020). *Modern Fortran: Building Efficient Parallel
> Applications*. Manning. [L5]

Relevante para o simulador EM PerfilaAnisoOmp.f08, que é escrito
em Fortran 2008 moderno com paralelização OpenMP.

---

## 14. Referências Compiladas

### 14.1 Artigos Científicos

[A1] Noh, K., Torres-Verdín, C., & Pardo, D. (2022). Real-Time 2.5D Inversion of LWD Resistivity Measurements Using Deep Learning for Geosteering Applications Across Faulted Formations. *Petrophysics*, 63(4), 506–518. DOI: 10.30632/PJV63N4-2022a2

[A2] Noh, K., Pardo, D., & Torres-Verdín, C. (2023). Physics-guided deep-learning inversion method for the interpretation of noisy logging-while-drilling resistivity measurements. *Geophysical Journal International*, 235, 150–165. DOI: 10.1093/gji/ggad217

[A3] Noh, K., Pardo, D., & Torres-Verdín, C. (2021). Deep-Learning Inversion Method for the Interpretation of Noisy Logging-While-Drilling Resistivity Measurements. arXiv:2111.07490v1.

[A4] Noh, K., Pardo, D., & Torres-Verdín, C. (2023+). 2.5D Deep Learning Inversion of LWD and Deep-Sensing EM Measurements Across Formations with Dipping Faults. Manuscript v8 (IEEE TGRS).

[A5] Morales, M. M., Eghbali, A., Raheem, O., Pyrcz, M. J., & Torres-Verdín, C. (2025). Anisotropic resistivity estimation and uncertainty quantification from borehole triaxial electromagnetic induction measurements. *Computers & Geosciences*, 196, 105786.

[A6] Wang, L., Li, H., Fan, Y., & Wu, Z. (2018). Sensitivity analysis and inversion processing of azimuthal resistivity logging-while-drilling measurements. *Journal of Geophysics and Engineering*, 15, 2339–2349. DOI: 10.1088/1742-2140/aacbf4

[A7] Wang, L., Deng, S., Zhang, P., Cao, Y., Fan, Y., & Yuan, X. (2019). Detection performance and inversion processing of logging-while-drilling extra-deep azimuthal resistivity measurements. *Petroleum Science*. DOI: 10.1007/s12182-019-00374-4

[A8] Guo, W., Wang, L., Wang, N., Qiao, P., Zeng, Z., & Yang, K. (2024). Efficient 1D Modeling of Triaxial Electromagnetic Logging in Uniaxial and Biaxial Anisotropic Formations Using Virtual Boundaries and Equivalent Resistivity Schemes. *Geophysics* (gxag017).

[A9] Carvalho, P. R., Régis, C., & Silva, V. S. (2022). Borehole Effects on Coaxial and Coplanar Logs of Triaxial Tools in Laminated Formations with Anisotropic Shale Host. *Brazilian Journal of Geophysics*, 40(3), 411–420. DOI: 10.22564/brjg.v40i3.2170

[A10] Saputra, W., Torres-Verdín, C., Ambia, J., et al. (2026). Recent Developments and Verifications of Multidimensional and Data-Adaptive Inversion of Borehole UDAR Measurements. *Petrophysics*, 67(1), 173–189. DOI: 10.30632/PJV67N1-2026a12

[A11] Constable, M. V., et al. (2016). Looking Ahead of the Bit While Drilling: From Vision to Reality. *Petrophysics*, 57(5), 426–446.

[A12] Li, G., Wu, Z., Liao, X., et al. (2025). Optimization and Analysis of Sensitive Areas for Look-Ahead EM LWD Based on Geometric Factors. *Energies*, 18(12), 3014. DOI: 10.3390/en18123014

[A13] Liu, R., Zhang, W., Chen, W., et al. (2025). Factors and detection capability of look-ahead logging while drilling (LWD) tools. *Petroleum Science*, 22, 850–867.

[A14] Alyaev, S., Suter, E., & Bratvold, R. B. (2019). A decision support system for multi-target geosteering. *Journal of Petroleum Science and Engineering*, 183, 106381.

[A15] Alyaev, S., et al. (2021). Deep learning for prediction of complex geology ahead of drilling. Springer (APPEA).

[A16] Alyaev, S., et al. (2022). Direct Multi-Modal Inversion of Geophysical Logs Using Deep Learning. *Earth and Space Science*, 9, e2021EA002186.

[A17] Alyaev, S., et al. (2022). Strategic Geosteering Workflow with Uncertainty Quantification and Deep Learning: A Case Study on the Goliat Field. arXiv:2210.15548.

[A18] Alyaev, S., et al. (2021). Probabilistic forecasting for geosteering in fluvial successions using a generative adversarial network. *First Break*, 39(7).

[A19] Alyaev, S., et al. (2023). Optimal Sequential Decision-Making in Geosteering: A Reinforcement Learning Approach. arXiv:2310.04772.

[A20] Alyaev, S., et al. (2025). High-precision geosteering via reinforcement learning and particle filters. *Computational Geosciences*. DOI: 10.1007/s10596-025-10352-y

[A21] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.

[A22] Bai, J., et al. (2022). An introduction to programming PINN-based computational solid mechanics. arXiv:2210.09060v4.

[A23] Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422–440.

[A24] Klein, J. D. (1993). Induction log anisotropy corrections. *The Log Analyst*, 34(02).

[A25] Klein, J. D., & Martin, P. (1997). The petrophysics of electrically anisotropic reservoirs. *The Log Analyst*, 38(03), 25–36.

[A26] Hagiwara, T. (1996). EM log response to anisotropic resistivity in thinly laminated formations. *SPE Formation Evaluation*, 11(04), 211–217.

[A27] Pardo, D., & Torres-Verdín, C. (2015). Fast 1D inversion of logging-while-drilling resistivity measurements for improved estimation in high-angle and horizontal wells. *Geophysics*, 80(2), E111–E124.

[A28] Latrach, A., Malki, M. L., Morales, M., Mehana, M., & Rabiei, M. (2024). A critical review of physics-informed machine learning applications in subsurface energy systems. *Geoenergy Science and Engineering*, 212938.

### 14.2 Papers de Arquiteturas

[AR1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*, 770–778.

[AR2] Liu, Z., Mao, H., Wu, C.-Y., et al. (2022). A ConvNet for the 2020s. *CVPR*.

[AR3] Szegedy, C., Vanhoucke, V., Ioffe, S., et al. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR*.

[AR4] Fawaz, H. I., et al. (2020). InceptionTime: Finding AlexNet for Time Series Classification. *DMKD*, 34, 1936–1962.

[AR5] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271.

[AR6] van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio. arXiv:1609.03499.

[AR7] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

[AR8] Schuster, M., & Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks. *IEEE Trans. Signal Processing*, 45(11), 2673–2681.

[AR9] Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

[AR10] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *Int. J. Forecasting*, 37(4), 1748–1764.

[AR11] Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI*.

[AR12] Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR*.

[AR13] Wu, H., et al. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. *NeurIPS*.

[AR14] Liu, Y., et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR*.

[AR15] Oreshkin, B. N., et al. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. *ICLR*.

[AR16] Challu, C., et al. (2023). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. *AAAI*.

[AR17] Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR*.

[AR18] Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *ICLR*.

[AR19] Li, Z., et al. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. *ICLR*.

[AR20] Lu, L., et al. (2021). Learning nonlinear operators via DeepONet. *Nature Machine Intelligence*, 3, 218–229.

[AR21] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*, 234–241.

[AR22] Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. arXiv:1804.03999.

[AR23] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML*.

[AR24] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. *CVPR*.

### 14.3 Papers de Losses

[LO1] Le Guen, V., & Thome, N. (2019). Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models. *NeurIPS*.

[LO2] Araya-Polo, M., et al. (2018). Deep-learning tomography. *The Leading Edge*, 37(1), 58–66.

[LO3] Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses. *CVPR*.

[LO4] Czarnecki, W. M., et al. (2017). Sobolev Training for Neural Networks. *NeurIPS*.

[LO5] Gallardo, L. A., & Meju, M. A. (2003). Characterization of heterogeneous near-surface materials by joint 2D inversion. *GRL*, 30(13), 1658.

[LO6] Jiang, L., et al. (2021). Focal Frequency Loss for Image Reconstruction and Synthesis. *ICCV*.

### 14.4 Papers de Treinamento e Otimização

[TR1] Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.

[TR2] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.

[TR3] Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*.

[TR4] Bengio, Y., et al. (2009). Curriculum Learning. *ICML*.

[TR5] Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*.

[TR6] Ioffe, S., & Szegedy, S. (2015). Batch Normalization. *ICML*.

[TR7] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv:1607.06450.

[TR8] Srivastava, N., et al. (2014). Dropout. *JMLR*, 15(1), 1929–1958.

[TR9] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.

### 14.5 Livros

[L1] Ellis, D. V., & Singer, J. M. (2008). *Well Logging for Earth Scientists* (2nd ed.). Springer.

[L2] Misra, S., Li, H., & He, J. (2019). *Machine Learning for Subsurface Characterization*. Gulf Professional Publishing.

[L3] Owen, L. (2022). *Hyperparameter Tuning with Python*. Packt.

[L4] Joseph, M. (2022). *Modern Time Series Forecasting with Python*. Packt.

[L5] Curcic, M. (2020). *Modern Fortran: Building Efficient Parallel Applications*. Manning.

[L6] Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive Into Deep Learning*.

[L7] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

### 14.6 Manuais Técnicos

[T1] Schlumberger. *GeoSphere HD 1.0 — High-Definition Reservoir Mapping-While-Drilling Service*.

[T2] SDAR Work Group. (2020). *Benchmark Models for Look-Ahead Applications*.

[T3] University of Houston. (2016). *A Survey on Definitions of Some Key Service Specs of LWD Deep Azimuthal Resistivity Tools*.

[T4] *GeoSphereXTatu — Comparação de sinais GeoSphere vs TatuAniso1D*. Documento interno.

---

## Resumo Quantitativo

| Categoria | Quantidade |
|:----------|:---------:|
| Artigos — DL + Inversão EM (core) | 5 |
| Artigos — Modelagem EM / Perfilagem | 5 |
| Artigos — Look-Ahead / Ferramentas | 3 |
| Artigos — Geosteering com IA | 7 |
| Artigos — PINNs e Physics-Informed ML | 4 |
| Artigos — Petrofísica EM / Anisotropia | 4 |
| Papers — Arquiteturas (44 arqs) | 24 |
| Papers — Losses (25 losses) | 6 |
| Papers — Treinamento / Otimização | 9 |
| Livros | 7 |
| Manuais Técnicos | 4 |
| **Total de referências** | **78** |

---

*Pesquisa Bibliográfica — Pipeline de Inversão Geofísica com Deep Learning v5.0.15*
*Última atualização: Março 2026*
*Última revisão: Incorporação de análise detalhada de Morales et al. (2025) — PINNs para inversão anisotrópica*
