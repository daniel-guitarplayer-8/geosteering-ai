# Documentação Completa do Simulador Fortran — PerfilaAnisoOmp

## Simulação Eletromagnética 1D para Meios Anisotrópicos TIV com Paralelismo OpenMP

**Projeto:** Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning
**Autor do Simulador:** Daniel Leal
**Linguagem:** Fortran 2008 (gfortran) com extensões OpenMP
**Localização:** `Fortran_Gerador/`
**Versão do Documento:** 2.0 (Abril 2026)

---

## Sumário

1. [Introdução e Motivação](#1-introdução-e-motivação)
2. [Fundamentos Físicos e Geofísicos](#2-fundamentos-físicos-e-geofísicos)
3. [Formulação Matemática Completa](#3-formulação-matemática-completa)
4. [Arquitetura do Software](#4-arquitetura-do-software)
5. [Módulos Fortran — Análise Detalhada](#5-módulos-fortran--análise-detalhada)
6. [Arquivo de Entrada model.in](#6-arquivo-de-entrada-modelin)
7. [Arquivos de Saída (.dat e .out)](#7-arquivos-de-saída-dat-e-out)
8. [Sistema de Build (Makefile)](#8-sistema-de-build-makefile)
9. [Gerador de Modelos Geológicos (Python)](#9-gerador-de-modelos-geológicos-python)
10. [Paralelismo OpenMP — Análise e Otimização](#10-paralelismo-openmp--análise-e-otimização)
11. [Análise de Viabilidade CUDA (GPU)](#11-análise-de-viabilidade-cuda-gpu)
12. [Análise de Reimplementação em Python Otimizado](#12-análise-de-reimplementação-em-python-otimizado)
13. [Integração com o Pipeline Geosteering AI v2.0](#13-integração-com-o-pipeline-geosteering-ai-v20)
14. [Referências Bibliográficas](#14-referências-bibliográficas)
15. [Apêndices](#15-apêndices)

---

## 1. Introdução e Motivação

### 1.1 Contexto do Problema

O simulador `PerfilaAnisoOmp` resolve o *problema direto* (forward modeling) da resposta eletromagnética (EM) de uma ferramenta de perfilagem triaxial em meios estratificados horizontalmente com anisotropia TIV (Transversalmente Isotrópica Vertical). Este tipo de modelagem é fundamental para a geração de dados sintéticos de treinamento para redes neurais de inversão.

A cadeia completa do projeto Geosteering AI segue o fluxo:

```
Gerador Python          Simulador Fortran         Pipeline DL (Python/TF)
(modelos geológicos) --> (forward EM 1D TIV)  --> (inversão EM via rede neural)
  fifthBuildTIV            PerfilaAnisoOmp           geosteering_ai/
  Models.py                   tatu.x
       |                        |                        |
  Parâmetros               Campos EM               Perfis de rho
  geológicos               tensoriais              estimados
  (rho_h, rho_v,           (tensor H 3x3)          (rho_h, rho_v)
   espessuras)                                      log10 scale
```

### 1.2 O que é Forward Modeling (Modelagem Direta)

O **forward modeling** (modelagem direta ou problema direto) é o processo de calcular a resposta
física de um instrumento de medição dado um modelo conhecido do meio. Em termos concretos,
o simulador recebe como entrada um modelo geológico completamente especificado — isto é,
o número de camadas, suas espessuras, e as resistividades horizontal e vertical de cada
camada — e calcula como uma ferramenta EM triaxial "veria" esse modelo. O resultado é o
tensor completo de campo magnético H(3x3) em cada posição de medição ao longo do poço.

O forward modeling é o passo complementar à **inversão** (problema inverso), onde se deseja
determinar as propriedades do meio a partir das medições. Na abordagem do projeto Geosteering AI,
o forward modeling é utilizado para gerar milhares de pares (modelo geológico, resposta EM)
que servem como dados de treinamento supervisionado para redes neurais. A rede neural aprende
o mapeamento inverso: dada uma resposta EM, estimar as resistividades do meio.

Matematicamente, se $\mathbf{d}$ representa os dados observados e $\mathbf{m}$ o modelo
geológico, o forward modeling computa a função:

```
d = F(m)    (forward: modelo → dados)

A rede neural aprende a função inversa aproximada:
m_est = G(d)  ≈  F^{-1}(d)    (inversão: dados → modelo estimado)
```

A qualidade da inversão via rede neural depende diretamente da fidelidade e diversidade
dos dados de treinamento gerados pelo forward modeling. Por isso, o simulador Fortran deve
ser fisicamente preciso e capaz de gerar grandes volumes de dados eficientemente.

### 1.3 Geração de Dados de Treinamento

O conceito central do projeto é substituir a inversão EM tradicional (iterativa, lenta,
computacionalmente custosa) por uma rede neural treinada em dados sintéticos. O processo
de geração de dados de treinamento segue estas etapas:

1. **Geração de modelos geológicos aleatórios:** O script Python `fifthBuildTIVModels.py`
   gera milhares de modelos geológicos 1D com parâmetros amostrados via Sobol quasi-Monte Carlo.
   Cada modelo especifica número de camadas (3-80), espessuras, e resistividades
   horizontais/verticais. Os cenários cobrem desde modelos "amigáveis" (camadas grossas,
   contrastes moderados) até "patológicos" (camadas finas, contrastes extremos).

2. **Simulação EM (forward modeling):** Para cada modelo geológico, o simulador Fortran
   calcula a resposta EM completa ao longo de uma janela de 120 metros com passo de 0.2 m,
   resultando em 600 posições de medição. Em cada posição, o tensor H(3x3) completo é
   calculado para 1-2 frequências (20 kHz e 40 kHz).

3. **Armazenamento em formato binário:** Os resultados são salvos em arquivos `.dat`
   no formato binário Fortran stream, com 22 colunas por registro (1 índice int32 +
   21 float64: profundidade, rho_h, rho_v, e 18 componentes reais/imaginárias do tensor H).

4. **Consumo pelo pipeline Python:** O módulo `geosteering_ai/data/loading.py` lê os
   arquivos `.dat` e organiza os dados em arrays NumPy de forma (n_modelos, seq_len, n_features),
   prontos para alimentar o pipeline de treinamento da rede neural.

### 1.4 Objetivo do Simulador

Dado um modelo geológico 1D (camadas horizontais com resistividades anisotrópicas), o simulador calcula o tensor completo de campo magnético `H(3x3)` medido por uma ferramenta triaxial LWD (Logging While Drilling) em cada posição de medição ao longo do poço. O resultado é armazenado em formato binário `.dat` para consumo pelo pipeline de Deep Learning.

### 1.5 Escopo da Documentação

Este documento apresenta:
- Fundamentos físicos completos (equações de Maxwell em meios TIV)
- Formulação matemática detalhada (decomposição TE/TM, transformada de Hankel, coeficientes de reflexão recursivos)
- Análise linha a linha do código Fortran
- Estrutura de dados de entrada e saída
- Análise completa de paralelismo e otimização
- Viabilidade de portabilidade para CUDA e Python

---

## 2. Fundamentos Físicos e Geofísicos

### 2.1 O Problema Eletromagnético em Meios Estratificados

A perfilagem EM de poços utiliza fontes dipolares magnéticas operando em baixa frequência (tipicamente 20 kHz a 2 MHz) para sondar as propriedades elétricas das formações rochosas ao redor do poço. O campo eletromagnético emitido pelo transmissor penetra no meio geológico e é modificado pelas propriedades de resistividade de cada camada. Os receptores medem o campo resultante, que contém informação sobre a distribuição de resistividade.

### 2.2 Anisotropia TIV (Transversalmente Isotrópica Vertical)

Rochas sedimentares apresentam anisotropia elétrica intrínseca devida à laminação deposicional. No modelo TIV, a condutividade é descrita por um tensor diagonal:

```
         ┌           ┐
         │ sigma_h 0     0     │
sigma =  │ 0     sigma_h 0     │
         │ 0     0     sigma_v │
         └           ┘

Onde:
  sigma_h = 1/rho_h  (condutividade horizontal, S/m)
  sigma_v = 1/rho_v  (condutividade vertical, S/m)
  rho_h = resistividade horizontal (Ohm.m)
  rho_v = resistividade vertical (Ohm.m)

Constraint físico: rho_v >= rho_h (SEMPRE em rochas sedimentares TIV)

Coeficiente de anisotropia:
  lambda = sqrt(sigma_h / sigma_v) = sqrt(rho_v / rho_h)
  Range no dataset do projeto: 1.0 <= lambda <= sqrt(2) ~ 1.414
  (Em formações reais, lambda pode exceder sqrt(2); o range acima
   reflete a distribuição de treinamento do gerador fifthBuildTIVModels.py)
```

**Por que rho_v >= rho_h em rochas sedimentares?**

A anisotropia elétrica TIV em rochas sedimentares tem origem na **microestrutura laminada**
dos sedimentos. Quando sedimentos de granulometrias diferentes se depositam alternadamente
(por exemplo, lâminas de areia e folhelho), cada lâmina possui uma resistividade diferente.
Em escala macroscópica, esse empilhamento de lâminas finas se comporta como um material
anisotrópico equivalente:

```
  Corrente horizontal (paralela às lâminas):
    ┌────────────────────────────────────────┐
    │  Areia (rho_a alto)   ───►  I_h        │  Corrente flui por AMBAS
    │  Folhelho (rho_f baixo) ───►  I_h      │  as lâminas em PARALELO
    └────────────────────────────────────────┘
    → Resistência equivalente: média harmônica (dominada por rho_f baixo)
    → Resultado: rho_h BAIXO

  Corrente vertical (perpendicular às lâminas):
    ┌────────────────────────────────────────┐
    │  Areia (rho_a alto)     ↓ I_v          │  Corrente cruza TODAS
    │  ─────────────────────────────────     │  as lâminas em SÉRIE
    │  Folhelho (rho_f baixo)  ↓ I_v         │
    └────────────────────────────────────────┘
    → Resistência equivalente: média aritmética (influenciada por rho_a alto)
    → Resultado: rho_v ALTO
```

Matematicamente, para N lâminas de espessura $h_i$ e resistividade $\rho_i$:

```
  1/rho_h = sum(h_i / rho_i) / sum(h_i)    (média harmônica → dominada por valores BAIXOS)
  rho_v   = sum(h_i * rho_i) / sum(h_i)    (média aritmética → influenciada por valores ALTOS)
```

Como a média aritmética é sempre >= média harmônica (desigualdade AM-HM), segue que
`rho_v >= rho_h` sempre. A igualdade só ocorre quando todas as lâminas têm a mesma
resistividade (meio isotrópico, lambda = 1).

Em formações com alternância de areia e folhelho, o coeficiente de anisotropia lambda
pode atingir valores de 2 a 10 ou mais. No dataset de treinamento do projeto, o range
é conservador (1.0 a sqrt(2) ~ 1.414) para focar na faixa mais comum em reservatórios.

**Implicação no código Fortran:** Cada camada `i` possui dois valores de resistividade `resist(i,1) = rho_h` e `resist(i,2) = rho_v`, e as condutividades são calculadas como `eta(i,1) = 1/rho_h`, `eta(i,2) = 1/rho_v`.

### 2.3 Geometria do Meio Estratificado

O meio é composto por `n` camadas horizontais planas, empilhadas verticalmente:

```
    z = -infinity    ┌──────────────────────────────────────┐
                     │  Camada 1 (semi-espaço superior)     │
                     │  rho_h(1), rho_v(1)                  │
                     │  espessura: infinita                  │
    z = prof(1) ─────├──────────────────────────────────────┤
                     │  Camada 2                             │
                     │  rho_h(2), rho_v(2)                  │
                     │  espessura: h(2)                      │
    z = prof(2) ─────├──────────────────────────────────────┤
                     │  Camada 3                             │
                     │  rho_h(3), rho_v(3)                  │
                     │  espessura: h(3)                      │
    z = prof(3) ─────├──────────────────────────────────────┤
                     │  ...                                  │
    z = prof(n-1) ───├──────────────────────────────────────┤
                     │  Camada n (semi-espaço inferior)      │
                     │  rho_h(n), rho_v(n)                  │
                     │  espessura: infinita                  │
    z = +infinity    └──────────────────────────────────────┘
```

**Convenção de fronteiras no código:**
- `prof(0) = -1e300` (sentinela para semi-espaço superior)
- `prof(1) = h(1) = 0` (primeira interface, atribuído diretamente)
- `prof(k) = prof(k-1) + h(k)` para `k = 2, ..., n-1`
- `prof(n) = +1e300` (sentinela para semi-espaço inferior)

A utilização de sentinelas `+/-1e300` elimina condicionais nos limites das exponenciais, evitando divisões por zero e garantindo estabilidade numérica.

### 2.4 Ferramenta Triaxial LWD

A ferramenta simulada é composta por:
- **1 transmissor** com 3 dipolos magnéticos ortogonais (Mx, My, Mz)
- **1 receptor** com 3 receptores ortogonais (Hx, Hy, Hz)
- **Espaçamento T-R:** `dTR` metros (default: 1.0 m)

O tensor de campo magnético medido pela ferramenta triaxial completa é:

```
       ┌                ┐
       │ Hxx  Hxy  Hxz  │
  H =  │ Hyx  Hyy  Hyz  │    (9 componentes complexas)
       │ Hzx  Hzy  Hzz  │
       └                ┘

Onde H_ij = campo na direção j devido ao dipolo na direção i.

Linha 1: campo do DMH-x (Horizontal Magnetic Dipole, direção x)
Linha 2: campo do DMH-y (Horizontal Magnetic Dipole, direção y)
Linha 3: campo do DMV   (Vertical Magnetic Dipole, direção z)
```

**Por que 9 componentes importam?**

Em um meio isotrópico com poço vertical, o tensor H possui simetrias que reduzem o
número de componentes independentes. Por exemplo, Hxy = Hyx, Hxz = Hzx = 0, Hyy = Hxx,
e Hz* = 0 para dipolos horizontais. Porém, em meios TIV com poço inclinado (theta != 0),
essas simetrias se quebram e **todas as 9 componentes** se tornam independentes e
carregam informação distinta sobre a formação:

```
  Poço vertical (theta = 0) em TIV:     Poço inclinado em TIV:
  ┌              ┐                       ┌                ┐
  │ Hxx  0    0  │                       │ Hxx  Hxy  Hxz  │  ← 9 independentes
  │ 0    Hxx  0  │  ← 2 independentes    │ Hyx  Hyy  Hyz  │     (tensor cheio)
  │ 0    0   Hzz │                       │ Hzx  Hzy  Hzz  │
  └              ┘                       └                ┘
```

As componentes diagonais (Hxx, Hyy, Hzz) são sensíveis principalmente à resistividade
na direção do respectivo dipolo. As componentes off-diagonal (Hxy, Hxz, etc.) são
especialmente sensíveis à presença de interfaces entre camadas e à inclinação do poço,
fornecendo informação sobre a geometria da formação que não está disponível nos
componentes diagonais isoladamente.

No pipeline Geosteering AI, o modo P1 (baseline) utiliza apenas 5 features selecionadas
das 22 colunas: zobs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz). Porém, modos futuros (Modo C)
podem explorar o tensor completo de 9 componentes (18 valores reais) para melhorar a
resolução da inversão em cenários com poço inclinado.

**Antenas inclinadas (tilted coils):**

Ferramentas LWD modernas (como a Schlumberger Rt Scanner ou a Halliburton EarthStar)
utilizam antenas inclinadas em relação ao eixo do poço, além das antenas axiais e
transversais tradicionais. Antenas inclinadas geram combinações lineares dos 9 componentes
do tensor, o que permite medir a anisotropia e a geometria das camadas com maior robustez.
O simulador `PerfilaAnisoOmp` calcula o tensor completo H(3x3), o que permite simular
qualquer configuração de antenas inclinadas via combinação linear pós-cálculo.

### 2.5 Arranjo T-R e Geometria de Perfilagem

O poço possui inclinação `theta` em relação à vertical. A ferramenta percorre o poço com passo `p_med` metros, cobrindo uma janela de investigação `tj` metros:

```
                         θ (ângulo de inclinação)
                        /
                       /
    z1 = -h1   ◯ R  /  (receptor, início da janela, acima de T)
                  |/
                  ◯ T  (transmissor, dTR metros abaixo do R)
                 /|
                / |
               /  |  pz = p_med * cos(θ)  (passo vertical)
              /   |  px = p_med * sin(θ)  (passo horizontal)
             /    |
    ◯ R+1  /     |
          ◯ T+1  |
         .       |
         .       |
         .       |
    z1 + tj ◯ R_final
              ◯ T_final

  nmed = ceil(tj / pz)  (número de medições por ângulo)
```

**Posições do Transmissor e Receptor para a j-ésima medição:**

```
Receptor (variável `x`, `z` no código):
  x = 0 + (j-1) * px - Lsen/2
  z = z1 + (j-1) * pz - Lcos/2

Transmissor (variável `Tx`, `Tz` no código):
  Tx = 0 + (j-1) * px + Lsen/2
  Tz = z1 + (j-1) * pz + Lcos/2

Onde:
  Lsen = dTR * sin(theta)
  Lcos = dTR * cos(theta)
  px = p_med * sin(theta)
  pz = p_med * cos(theta)
```

**Nota de configuração:** O transmissor está **abaixo** dos receptores, conforme configuração padrão dos arranjos da Petrobras.

### 2.6 Constantes Físicas do Simulador

| Constante | Símbolo | Valor | Unidade | Significado |
|:----------|:--------|:------|:--------|:------------|
| Permeabilidade magnética | mu | 4pi x 10^-7 | H/m | Permeabilidade do vácuo |
| Permissividade elétrica | epsilon | 8.85 x 10^-12 | F/m | Permissividade do vácuo |
| Tolerância numérica | eps | 10^-9 | - | Limiar para singularidades |
| Tolerância angular | del | 0.1 | graus | Resolução angular mínima |
| Corrente do dipolo | Iw | 1.0 | A | Corrente do transmissor |
| Momento dipolar | mx, my, mz | 1.0 | A.m^2 | Momento do dipolo magnético |

### 2.7 Skin Depth e Profundidade de Investigação (DOI)

O **skin depth** (profundidade pelicular ou de penetração) é a distância na qual a
amplitude da onda EM decai por um fator de 1/e (~37%) em relação ao valor na fonte.
É o parâmetro fundamental que determina o alcance da investigação da ferramenta:

```
  delta = sqrt(2 / (omega * mu * sigma))

Onde:
  omega = 2*pi*f  (frequência angular, rad/s)
  mu = 4*pi*10^-7 H/m  (permeabilidade magnética)
  sigma = condutividade do meio (S/m)

Valores típicos para f = 20 kHz (FREQUENCY_HZ do projeto):

  ┌──────────────────────────────────────────────────────────────┐
  │  Formação              │  rho (Ohm.m)  │  delta (m)          │
  ├────────────────────────┼───────────────┼─────────────────────┤
  │  Folhelho salino       │  0.3          │  0.62               │
  │  Folhelho              │  1.0          │  1.13               │
  │  Arenito saturado      │  10           │  3.56               │
  │  Arenito com óleo      │  100          │  11.25              │
  │  Carbonato compacto    │  1000         │  35.59              │
  │  Sal/anidrita          │  10000        │  112.54             │
  └──────────────────────────────────────────────────────────────┘
```

**Implicações para a profundidade de investigação (DOI):**

A DOI real da ferramenta é tipicamente 2-3 vezes o skin depth, dependendo da
relação sinal-ruído do instrumento. Para a configuração padrão do projeto
(f = 20 kHz, dTR = 1.0 m):

- Em meios condutivos (rho < 1 Ohm.m): DOI ~ 1-2 m. A ferramenta "vê"
  apenas a camada imediata.
- Em meios resistivos (rho > 100 Ohm.m): DOI ~ 20-30 m. A ferramenta
  detecta interfaces distantes.
- Frequências mais baixas aumentam o skin depth (e a DOI), mas reduzem
  a resolução vertical.

O skin depth também explica por que o simulador usa frequências na faixa de
kHz (20 kHz, 40 kHz): frequências muito altas (MHz) teriam skin depth de
centímetros em meios condutivos, insuficiente para sondar além do poço.
Frequências muito baixas (Hz) teriam skin depth de quilômetros, mas com
resolução vertical inadequada para geosteering.

### 2.8 Decoupling — Remoção do Acoplamento Direto

O campo magnético medido pelo receptor é composto por duas contribuições:
o **acoplamento direto** (campo do transmissor que chega ao receptor sem
interagir com o meio) e o **campo secundário** (campo refletido/refratado
pelas interfaces das camadas). Para a inversão, interessa apenas o campo
secundário, pois ele carrega a informação sobre as propriedades do meio.

O **decoupling** é a operação de subtrair o acoplamento direto do campo
total medido. Para dipolos magnéticos em espaço livre, o acoplamento
direto analítico é:

```
Campo do dipolo magnético em espaço livre (r = distância T-R):

  Para componentes PLANARES (Hxx, Hyy):
    AC_p = -1 / (4 * pi * L^3)
    AC_p ≈ -0.079577  para L = 1.0 m

  Para componente AXIAL (Hzz):
    AC_x = +1 / (2 * pi * L^3)
    AC_x ≈ +0.159155  para L = 1.0 m

Onde L = espaçamento T-R (SPACING_METERS = 1.0 m no projeto).
```

No pipeline Geosteering AI, o decoupling é implementado pelas **views de geosinais (GS)**
no módulo `geosteering_ai/data/geosignals.py`. As views subtraem os valores AC_p e AC_x
das componentes correspondentes do tensor H antes de alimentar a rede neural.

**Nota importante:** O simulador Fortran NÃO realiza o decoupling — ele armazena o campo
total (acoplamento direto + campo secundário). O decoupling é feito exclusivamente
no pipeline Python, permitindo que diferentes estratégias de decoupling sejam testadas
sem recomputar a simulação.

```
  Campo total       =  Acoplamento direto  +  Campo secundário
  (armazenado .dat)    (analítico, AC)        (contém info geológica)

  No pipeline:
    campo_decoupled = campo_total - AC
    → Usado como input da rede neural
```

---

## 3. Formulação Matemática Completa

### 3.1 Equações de Maxwell em Meios TIV

O campo EM em meios com anisotropia TIV obedece às equações de Maxwell no domínio da frequência (convenção `exp(-i*omega*t)`):

```
∇ × E = i*omega*mu*H           (Lei de Faraday)
∇ × H = sigma_tensor * E + J   (Lei de Ampère)

Onde:
  omega = 2*pi*f  (frequência angular, rad/s)
  mu = 4*pi*10^-7 H/m  (permeabilidade, assumida isotrópica)
  sigma_tensor = diag(sigma_h, sigma_h, sigma_v)  (condutividade TIV)
  J = fonte dipolar (corrente do transmissor)
```

**Sobre a corrente de deslocamento:**

A lei de Ampère completa inclui o termo de corrente de deslocamento:

```
∇ × H = sigma * E + J + dD/dt

No domínio da frequência:
∇ × H = (sigma - i*omega*epsilon) * E + J
```

O simulador **desconsidera** a corrente de deslocamento (`-i*omega*epsilon*E`).
Esta simplificação é válida quando a corrente de condução domina sobre a corrente
de deslocamento, ou seja, quando:

```
  sigma >> omega * epsilon

Para f = 20 kHz e rho = 1000 Ohm.m (caso MAIS resistivo do dataset):
  sigma = 0.001 S/m
  omega * epsilon = 2*pi*20000 * 8.85e-12 ≈ 1.11e-6 S/m

  Razão: sigma / (omega*epsilon) = 0.001 / 1.11e-6 ≈ 900

Mesmo no caso mais desfavorável, a corrente de condução é 900 vezes maior
que a corrente de deslocamento. Para formações típicas (rho < 100 Ohm.m),
a razão excede 10^4, tornando a simplificação extremamente precisa.
```

Esta é a chamada **aproximação quasi-estática**, válida para frequências
na faixa de kHz a MHz em formações geológicas. A aproximação falha apenas
para frequências muito altas (> 100 MHz) ou meios extremamente resistivos
(rho > 10^6 Ohm.m), ambos fora do escopo da perfilagem LWD.

### 3.2 Impeditividade (zeta)

```
zeta = i * omega * mu

No código (Fortran):
  zeta = cmplx(0, 1.d0, kind=dp) * omega * mu

Significado físico: zeta relaciona o campo elétrico ao campo magnético
nas equações de Maxwell. É a impedância intrínseca do meio multiplicada
pelo número de onda.
```

### 3.3 Números de Onda e Constantes de Propagação

Para cada camada `i` com condutividades `eta_h = 1/rho_h` e `eta_v = 1/rho_v`:

```
Número de onda horizontal ao quadrado:
  kh^2(i) = -zeta * eta_h(i) = -i*omega*mu*sigma_h(i)

Número de onda vertical ao quadrado:
  kv^2(i) = -zeta * eta_v(i) = -i*omega*mu*sigma_v(i)

Constantes de propagação (dependem do parâmetro espectral kr):
  u(i) = sqrt(kr^2 - kh^2(i))   (propagação horizontal, modo TE)
  v(i) = sqrt(kr^2 - kv^2(i))   (propagação vertical)
  s(i) = lambda(i) * v(i)       (propagação TIV, modo TM)

Onde:
  lambda(i) = sqrt(sigma_h(i) / sigma_v(i))  (coeficiente de anisotropia)
  kr = variável de integração no domínio espectral (Hankel)
```

**No código Fortran (`commonarraysMD`):**

```fortran
kh2(i) = -zeta * eta(i,1)           ! kh^2 = -i*omega*mu*sigma_h
kv2(i) = -zeta * eta(i,2)           ! kv^2 = -i*omega*mu*sigma_v
lamb2(i) = eta(i,1) / eta(i,2)      ! (sigma_h/sigma_v) = lambda^2
u(:,i) = sqrt(kr*kr - kh2(i))       ! constante horiz (TE)
v(:,i) = sqrt(kr*kr - kv2(i))       ! constante vert
s(:,i) = sqrt(lamb2(i)) * v(:,i)    ! lambda * v (TM)
```

### 3.4 Decomposição TE/TM para Meios TIV

O campo EM em meios estratificados TIV é decomposto em dois modos independentes:

**Modo TE (Transverse Electric):** Componente do campo elétrico tangencial às interfaces.
- Governado pela constante de propagação `u` (horizontal)
- Associado à admitância intrínseca `AdmInt = u / zeta`

**Modo TM (Transverse Magnetic):** Componente do campo magnético tangencial às interfaces.
- Governado pela constante de propagação `s` (TIV)
- Associado à impedância intrínseca `ImpInt = s / eta_h`

```
                       Campo EM total
                      /               \
                   Modo TE            Modo TM
                  (u, AdmInt)        (s, ImpInt)
                  /      \           /      \
              RTEup   RTEdw      RTMup   RTMdw
           (reflexão) (reflexão)(reflexão)(reflexão)
```

**Intuição física para a decomposição TE/TM:**

A decomposição TE/TM é a separação do campo EM em duas polarizações independentes
relativas às interfaces horizontais das camadas. Essa decomposição é possível porque,
em meios estratificados 1D com anisotropia TIV, as interfaces são planas e horizontais,
e o tensor de condutividade é diagonal com simetria cilíndrica em torno do eixo z.

- **Modo TE:** O campo elétrico é inteiramente tangencial (horizontal) às interfaces.
  Como a corrente flui paralelamente às lâminas, o modo TE "vê" apenas a condutividade
  horizontal sigma_h. Sua constante de propagação `u` depende somente de kh^2.
  O modo TE é **insensível à anisotropia** em poço vertical — a resposta TE de um
  meio TIV é idêntica à de um meio isotrópico com sigma = sigma_h.

- **Modo TM:** O campo magnético é inteiramente tangencial às interfaces, e o campo
  elétrico possui componente vertical (perpendicular às lâminas). Por isso, o modo TM
  "sente" tanto sigma_h quanto sigma_v. Sua constante de propagação `s = lambda*v`
  depende da razão sigma_h/sigma_v (anisotropia). O modo TM é o **único canal
  de informação sobre rho_v** em configurações com poço vertical.

Esta separação é análoga à decomposição de ondas ópticas em polarização s e p
na reflexão em uma interface plana. A grande vantagem é que cada modo pode ser
resolvido independentemente (equação escalar de 2ª ordem), reduzindo o problema
vetorial 3D a dois problemas escalares 1D.

### 3.5 Coeficientes de Reflexão Recursivos

Os coeficientes de reflexão são calculados recursivamente das fronteiras mais externas para a camada do transmissor, utilizando a fórmula de estabilidade numérica baseada em `tanh`:

**Direção descendente (de cima para baixo):**

```
Admitância aparente descendente:
  AdmAp_dw(n) = AdmInt(n)    (semi-espaço inferior: sem reflexão)

  AdmAp_dw(i) = AdmInt(i) * [AdmAp_dw(i+1) + AdmInt(i) * tanh(u(i)*h(i))]
                              / [AdmInt(i) + AdmAp_dw(i+1) * tanh(u(i)*h(i))]

Coeficiente de reflexão TE descendente:
  RTEdw(n) = 0    (sem reflexão na última camada)
  RTEdw(i) = [AdmInt(i) - AdmAp_dw(i+1)] / [AdmInt(i) + AdmAp_dw(i+1)]
```

**Fórmula `tanh` no código (estabilidade numérica):**

```fortran
! Em vez de tanh(x) diretamente, o código usa:
tghuh(:,i) = (1.d0 - exp(-2.d0 * uh(:,i))) / (1.d0 + exp(-2.d0 * uh(:,i)))

! Que é matematicamente equivalente a tanh(uh) mas evita overflow
! quando uh é grande (camadas espessas ou alta frequência).
! Para uh >> 1: tanh(uh) -> 1.0 (correto)
! Para uh << 1: tanh(uh) -> uh (correto)
```

**Direção ascendente (de baixo para cima):**

```
Admitância aparente ascendente:
  AdmAp_up(1) = AdmInt(1)    (semi-espaço superior: sem reflexão)

  AdmAp_up(i) = AdmInt(i) * [AdmAp_up(i-1) + AdmInt(i) * tanh(u(i)*h(i))]
                              / [AdmInt(i) + AdmAp_up(i-1) * tanh(u(i)*h(i))]

Coeficiente de reflexão TE ascendente:
  RTEup(1) = 0    (sem reflexão na primeira camada)
  RTEup(i) = [AdmInt(i) - AdmAp_up(i-1)] / [AdmInt(i) + AdmAp_up(i-1)]
```

As mesmas fórmulas se aplicam ao modo TM substituindo `AdmInt` por `ImpInt`, `u` por `s`, e `uh` por `sh`.

### 3.6 Fatores de Onda Refletida na Camada do Transmissor

A sub-rotina `commonfactorsMD` calcula os fatores de onda refletida especificamente para a camada onde o transmissor está localizado. Estes fatores encapsulam as múltiplas reflexões dentro da camada fonte:

**Modo TM (fatores Mxdw, Mxup):**

```
den_TM = 1 - RTMdw(T) * RTMup(T) * exp(-2*s*h(T))

Mxdw = [exp(-s*(prof(T) - h0)) + RTMup(T) * exp(s*(prof(T-1) - h0 - h(T)))] / den_TM
Mxup = [exp(s*(prof(T-1) - h0)) + RTMdw(T) * exp(-s*(prof(T) - h0 + h(T)))] / den_TM

Onde:
  T = camadT (índice da camada do transmissor)
  h0 = profundidade do transmissor
  prof(T), prof(T-1) = fronteiras da camada T
```

**Modo TE (fatores Eudw, Euup, FEdwz, FEupz):**

```
den_TE = 1 - RTEdw(T) * RTEup(T) * exp(-2*u*h(T))

Eudw = [exp(-u*(prof(T) - h0)) - RTEup(T) * exp(u*(prof(T-1) - h0 - h(T)))] / den_TE
Euup = [exp(u*(prof(T-1) - h0)) - RTEdw(T) * exp(-u*(prof(T) - h0 + h(T)))] / den_TE

FEdwz = [exp(-u*(prof(T) - h0)) + RTEup(T) * exp(u*(prof(T-1) - h(T) - h0))] / den_TE
FEupz = [exp(u*(prof(T-1) - h0)) + RTEdw(T) * exp(-u*(prof(T) + h(T) - h0))] / den_TE
```

**Nota sobre sinais:** Os fatores `Eu` (modo TE) possuem sinais opostos aos fatores `Mx` (modo TM) nos termos de reflexão. Isso reflete a diferença de condições de contorno entre os modos TE (continuidade de E tangencial) e TM (continuidade de H tangencial) nas interfaces.

### 3.7 Coeficientes de Transmissão entre Camadas

Quando o receptor está em uma camada diferente do transmissor, os campos são calculados via coeficientes de transmissão recursivos:

**Receptor abaixo do transmissor (camadR > camadT):**

```
Txdw(T) = mx / (2*s(T))    (termo fonte na camada T)
Tudw(T) = -mx / 2          (termo fonte TE)

Para j = T+1 até camadR:
  Txdw(j) = s(j-1) * Txdw(j-1) * (...exponenciais...) / [(1 - RTMdw(j)*exp(-2*sh(j))) * s(j)]
  Tudw(j) = u(j-1) * Tudw(j-1) * (...exponenciais...) / [(1 - RTEdw(j)*exp(-2*uh(j))) * u(j)]
```

**Receptor acima do transmissor (camadR < camadT):** Fórmulas análogas, com direção ascendente e coeficientes RTMup/RTEup.

### 3.8 Transformada de Hankel via Filtros Digitais

A passagem do domínio espectral (kr) para o domínio espacial (r) é realizada pela transformada de Hankel, implementada via filtro digital:

```
f(r) = integral_0^inf F(kr) * Jn(kr*r) * kr * d(kr)

Onde Jn é a função de Bessel de primeira espécie de ordem n.

Implementação via filtro digital (Werthmuller, 201 pontos):

  f(r) ≈ (1/r) * sum_{i=1}^{npt} F(kr_i/r) * w_i

Onde:
  kr_i = abscissas do filtro (tabeladas)
  w_i = pesos do filtro (tabelados, diferentes para J0 e J1)
  npt = 201 pontos (filtro Werthmuller)
```

**Por que a transformada de Hankel funciona (simetria cilíndrica):**

A transformada de Hankel é a ferramenta matemática natural para problemas com
**simetria cilíndrica**. Em meios estratificados 1D, as interfaces são planas e
horizontais, e as fontes dipolares geram campos com dependência azimutal simples
(cos(phi) ou sin(phi) para HMD, e nenhuma dependência azimutal para VMD).

Quando aplicamos a transformada de Fourier 2D nas coordenadas horizontais (x, y)
das equações de Maxwell, obtemos integrais no espaço (kx, ky). Devido à simetria
cilíndrica do meio, podemos converter essas integrais duplas em uma integral simples
sobre o número de onda radial kr = sqrt(kx^2 + ky^2), usando a identidade:

```
  integral_{-inf}^{+inf} integral_{-inf}^{+inf} F(kx,ky) * exp(i*(kx*x+ky*y)) dkx dky
  = 2*pi * integral_0^{inf} F(kr) * J_n(kr*r) * kr * d(kr)

  Onde r = sqrt(x^2 + y^2) e J_n é a função de Bessel de ordem n.
  n = 0 para VMD (simetria axial completa)
  n = 0 e n = 1 para HMD (dependência cos(phi))
```

Esta redução de integral dupla para simples é a razão pela qual o cálculo é
computacionalmente tratável. Em vez de avaliar uma integral 2D (O(N^2) pontos),
avaliamos uma soma 1D de 201 termos — uma economia de várias ordens de magnitude.

O filtro digital substitui a integração numérica (quadratura) por uma soma
ponderada com coeficientes pré-calculados, tornando a avaliação extremamente
eficiente: ~201 multiplicações complexas e uma soma por componente de campo.

**Filtros disponíveis no código (`filtersv2.f08`):**

| Filtro | Pontos | Referência | Uso |
|:-------|:------:|:-----------|:----|
| Kong | 61 | Kong (2007) | Rápido, precisão moderada |
| Key | - | Key (2012) | Precisão padrão |
| **Werthmuller** | **201** | **Werthmuller (2006)** | **Usado no simulador** |
| Anderson | 801 | Anderson (1982) | Alta precisão |

O simulador utiliza o filtro Werthmuller de 201 pontos (`npt = 201`), que oferece excelente balanço entre precisão e desempenho.

### 3.9 Campos do Dipolo Magnético Horizontal (HMD) em Meio TIV

O campo magnético do HMD é calculado a partir dos kernels espectrais Ktm (modo TM) e Kte (modo TE), convolvidos com funções de Bessel J0 e J1:

```
Para o HMD na direção x (hmdx):

Hx = [(2x^2/r^2 - 1) * sum(Ktedz * wJ1)/r - kh^2*(2y^2/r^2 - 1) * sum(Ktm * wJ1)/r
      - x^2/r^2 * sum(Ktedz * wJ0 * kr) + kh^2*y^2/r^2 * sum(Ktm * wJ0 * kr)] / (2*pi*r)

Hy = xy/r^2 * [sum(Ktedz * wJ1 + kh^2 * Ktm * wJ1)/r
               - sum((Ktedz * wJ0 + kh^2 * Ktm * wJ0) * kr)/2] / (pi*r)

Hz = -x * sum(Kte * wJ1 * kr^2) / (r * 2*pi*r)
```

**Propriedade de simetria do HMDy:** O campo do dipolo y é obtido por rotação de 90 graus do HMDx:

```
HMDy: x -> y, y -> -x
Hx(hmdy) = Hy(hmdx)
Hy(hmdy) = expressão com (2y^2/r^2 - 1) e x^2/r^2 (permutados)
Hz(hmdy) = -y * sum(Kte * wJ1 * kr^2) / (r * 2*pi*r)
```

O modo `'hmdxy'` calcula ambos simultaneamente, evitando recomputação dos kernels.

### 3.10 Campos do Dipolo Magnético Vertical (VMD) em Meio TIV

O VMD excita apenas o modo TE (por simetria axial, não há acoplamento TM):

```
Hx = -x * sum(KtedzzJ1 * kr^2) / (2*pi*r) / r

Hy = -y * sum(KtedzzJ1 * kr^2) / (2*pi*r) / r

Hz = sum(KtezJ0 * kr^3) / (2*pi*zeta) / r

Onde:
  KtezJ0 = fac * wJ0     (kernel z com Bessel J0)
  KtezJ1 = fac * wJ1     (kernel z com Bessel J1)
  KtedzzJ1 = AdmInt * KtezJ1  (kernel derivada z com J1)
```

### 3.11 Rotação do Tensor para Orientação Arbitrária da Ferramenta

O tensor `H` calculado no sistema de coordenadas geológico (x, y, z fixos) deve ser rotacionado para o sistema de coordenadas da ferramenta. A rotação é definida por três ângulos de Euler (alpha, beta, gamma):

```
H_ferramenta = R^T * H_geológico * R

Onde R é a matriz de rotação (Liu, 2017, eq. 4.80):

R = [  cos(a)*cos(b)*cos(g) - sin(b)*sin(g)    -cos(a)*cos(b)*sin(g) - sin(b)*cos(g)    sin(a)*cos(b) ]
    [  cos(a)*sin(b)*cos(g) + cos(b)*sin(g)    -cos(a)*sin(b)*sin(g) + cos(b)*cos(g)    sin(a)*sin(b) ]
    [ -sin(a)*cos(g)                             sin(a)*sin(g)                            cos(a)        ]

Na perfilagem LWD:
  alpha = theta (inclinação do poço)
  beta = 0      (azimute, assumido zero)
  gamma = 0     (rotação da ferramenta, assumida zero)
```

---

## 4. Arquitetura do Software

### 4.1 Estrutura de Módulos

O simulador é composto por 6 arquivos-fonte Fortran organizados em módulos com dependências hierárquicas:

```
  ┌─────────────────────────────────────────────────────────┐
  │  RunAnisoOmp.f08 (Programa Principal)                   │
  │    ├── Lê model.in                                      │
  │    └── Chama perfila1DanisoOMP(...)                      │
  ├─────────────────────────────────────────────────────────┤
  │  PerfilaAnisoOmp.f08 (Módulo DManisoTIV)                │
  │    ├── perfila1DanisoOMP (loop principal + OpenMP)       │
  │    ├── fieldsinfreqs (campos EM por frequência)          │
  │    ├── writes_files (escrita .dat/.out)                  │
  │    └── write_results (escrita de resultados auxiliares)  │
  ├─────────────────────────────────────────────────────────┤
  │  magneticdipoles.f08 (Campos Dipolares)                 │
  │    ├── hmd_TIV_optimized (Dipolo Horiz. TIV)            │
  │    └── vmd_optimized (Dipolo Vert. TIV)                 │
  ├─────────────────────────────────────────────────────────┤
  │  utils.f08 (Utilidades)                                 │
  │    ├── sanitize_hprof_well (prepara geometria)          │
  │    ├── findlayersTR2well (localiza camadas T/R)         │
  │    ├── commonarraysMD (constantes de propagação)         │
  │    ├── commonfactorsMD (fatores de onda refletida)       │
  │    ├── RtHR (rotação tensor)                            │
  │    └── layer2z_inwell, int2str, real2str, etc.          │
  ├─────────────────────────────────────────────────────────┤
  │  filtersv2.f08 (module filterscommonbase)                 │
  │    ├── J0J1Kong (61 pontos)                             │
  │    ├── J0J1Key                                          │
  │    ├── J0J1Wer (201 pontos) ← USADO                    │
  │    └── J0J1And (801 pontos)                             │
  ├─────────────────────────────────────────────────────────┤
  │  parameters.f08 (Constantes Físicas)                    │
  │    └── dp, pi, mu, epsilon, eps, del, Iw, mx, my, mz   │
  └─────────────────────────────────────────────────────────┘
```

### 4.2 Grafo de Dependências

```
parameters.f08
      │
      ├──────────────────────┐
      v                      v
filtersv2.f08          magneticdipoles.f08
(filterscommonbase)
      │                      │
      v                      │
   utils.f08 ◄───────────────┘
      │
      v
PerfilaAnisoOmp.f08  (usa todos acima + omp_lib)
      │
      v
RunAnisoOmp.f08      (programa principal)
```

### 4.3 Fluxo de Execução Completo

```
1. RunAnisoOmp lê model.in
      │
2. Chama perfila1DanisoOMP(...)
      │
      ├── 2a. Calcula nmed(theta) para cada ângulo
      ├── 2b. Carrega filtro Werthmuller (201 pontos)
      ├── 2c. Monta geometria (sanitize_hprof_well)
      ├── 2d. Configura OpenMP (nested parallelism)
      │
      ├── 2e. LOOP PARALELO sobre ângulos k = 1..ntheta
      │       │
      │       ├── LOOP PARALELO sobre medições j = 1..nmed(k)
      │       │       │
      │       │       ├── Calcula posições T-R
      │       │       └── Chama fieldsinfreqs(...)
      │       │               │
      │       │               ├── Identifica camadas T, R
      │       │               ├── Para cada frequência f:
      │       │               │     ├── commonarraysMD (u,s,RTEdw,RTEup,RTMdw,RTMup)
      │       │               │     ├── commonfactorsMD (Mxdw,Mxup,Eudw,Euup)
      │       │               │     ├── hmd_TIV_optimized (HMD x,y)
      │       │               │     ├── vmd_optimized (VMD z)
      │       │               │     ├── Monta tensor H(3x3)
      │       │               │     └── RtHR (rotação)
      │       │               └── Armazena zrho, cH
      │       └── Coleta resultados do loop j
      └── Coleta resultados do loop k
      │
3. writes_files(...)
      │
      ├── Escreve .out (metadata)
      └── Escreve .dat (binário, stream)
```

### 4.4 Inventário de Linhas de Código

| Arquivo | LOC | Propósito |
|:--------|----:|:----------|
| `parameters.f08` | 19 | Constantes físicas e parâmetros numéricos |
| `filtersv2.f08` | 5559 | Coeficientes tabelados dos filtros digitais |
| `utils.f08` | 383 | Utilidades: geometria, propagação, reflexão, rotação |
| `magneticdipoles.f08` | 540 | Campos HMD e VMD em meios TIV |
| `PerfilaAnisoOmp.f08` | 304 | Módulo principal: loop de perfilagem + I/O |
| `RunAnisoOmp.f08` | 54 | Programa principal: leitura de model.in |
| **Total Fortran** | **6859** | |
| `fifthBuildTIVModels.py` | ~900 | Gerador de modelos geológicos |
| `Makefile` | 83 | Sistema de build |
| **Total do projeto** | **~7842** | |

---

## 5. Módulos Fortran — Análise Detalhada

### 5.1 parameters.f08 — Constantes Físicas

Módulo mínimo que define constantes fundamentais com precisão dupla (`dp = kind(1.d0)`):

| Variável | Valor | Tipo | Descrição |
|:---------|:------|:-----|:----------|
| `sp` | `kind(1.e0)` | `integer, parameter` | Precisão simples (~7 dígitos) |
| `dp` | `kind(1.d0)` | `integer, parameter` | Precisão dupla (~15 dígitos) |
| `qp` | `selected_real_kind(30)` | `integer, parameter` | Precisão quádrupla (disponível, não usado) |
| `pi` | 3.14159265... | `real(dp), parameter` | Pi com 37 dígitos decimais |
| `mu` | 4e-7 * pi | `real(dp), parameter` | Permeabilidade do vácuo (H/m) |
| `epsilon` | 8.85e-12 | `real(dp), parameter` | Permissividade do vácuo (F/m) |
| `eps` | 1e-9 | `real(dp), parameter` | Tolerância numérica para singularidades |
| `del` | 0.1 | `real(dp), parameter` | Tolerância angular (graus) |
| `Iw` | 1.0 | `real(dp), parameter` | Corrente do dipolo (A) |
| `dsx, dsy, dsz` | 1.0 | `real(dp), parameter` | Área dos dipolos elétricos (m^2, não utilizado) |
| `mx, my, mz` | 1.0 | `real(dp), parameter` | Momentos dipolares magnéticos (A.m^2) |

### 5.2 filtersv2.f08 (module filterscommonbase) — Filtros Digitais para Transformada de Hankel

**Nota:** O arquivo chama-se `filtersv2.f08`, mas o módulo Fortran declarado dentro dele é `module filterscommonbase`. O módulo principal (`PerfilaAnisoOmp.f08`) o referencia via `use filterscommonbase`.

Módulo que armazena os coeficientes tabelados de 4 filtros digitais para transformadas de Hankel. Cada filtro fornece:
- `absc(npt)`: Abscissas (pontos de amostragem no domínio espectral)
- `wJ0(npt)`: Pesos para convolução com função de Bessel J0
- `wJ1(npt)`: Pesos para convolução com função de Bessel J1

**Filtro utilizado no simulador:** `J0J1Wer` com `npt = 201` (Werthmuller).

A transformada de Hankel via filtro digital é uma técnica clássica em geofísica EM que substitui a integração numérica direta (quadratura) por uma soma ponderada. Os coeficientes são pré-calculados offline e tabelados, tornando a avaliação extremamente eficiente (~201 multiplicações por ponto).

### 5.3 utils.f08 — Funções Utilitárias

#### 5.3.1 sanitize_hprof_well

Converte espessuras de camadas em arrays de profundidade com sentinelas:

```
Input:  esp(1:n) = [0, h2, h3, ..., h(n-1), 0]
                    ^                         ^
              semi-espaço              semi-espaço
              superior                 inferior

Output: h(1:n) = esp  (espessuras, com h(1)=h(n)=0)
        prof(0:n) = [-1e300, 0, h2, h2+h3, ..., +1e300]
                      ^                           ^
               sentinela                    sentinela
               (evita overflow)             (evita overflow)
```

As sentinelas `+/-1e300` eliminam condicionais nos cálculos de exponenciais nas camadas extremas, simplificando o código de propagação.

#### 5.3.2 findlayersTR2well

Identifica em qual camada estão o transmissor (camadT) e o receptor (camadR), dado o array de profundidades das interfaces. A busca é feita de baixo para cima (`do i = n-1, 2, -1`) para eficiência em cenários onde T e R estão em camadas profundas.

#### 5.3.3 commonarraysMD — Constantes de Propagação e Reflexão

Esta é a sub-rotina mais importante do ponto de vista físico. Calcula, para cada ponto do filtro (`npt = 201`) e cada camada (`n`):

1. Números de onda: `kh^2`, `kv^2`
2. Constantes de propagação: `u`, `v`, `s = lambda*v`
3. Admitâncias/impedâncias intrínsecas: `AdmInt`, `ImpInt`
4. Produtos espessura × constante: `uh = u*h`, `sh = s*h`
5. Tanh estabilizado: `tghuh`, `tghsh`
6. Coeficientes de reflexão recursivos (TE e TM, ascendentes e descendentes)

**Complexidade computacional:** O(npt * n) por chamada, onde `npt = 201` e `n = número de camadas`.

**Algoritmo passo a passo:**

```
commonarraysMD(n, npt, hordist, krJ0J1, zeta, h, eta, ...):

  Passo 1 — Preparação das abscissas do filtro:
    Para i = 1..npt:
      kr(i) = absc(i) / hordist    (escala abscissas pelo espaçamento horizontal)
    Este passo converte as abscissas adimensionais do filtro em números de onda
    físicos (1/m), dividindo pela distância horizontal T-R.

  Passo 2 — Números de onda por camada:
    Para j = 1..n:
      kh2(j) = -zeta * eta(j,1)    (kh² = -iωμσ_h)
      kv2(j) = -zeta * eta(j,2)    (kv² = -iωμσ_v)
      lamb2(j) = eta(j,1)/eta(j,2) (λ² = σ_h/σ_v)
    Nota: kh2 e kv2 são complexos puros (parte real = 0) na aproximação
    quasi-estática. O sinal negativo garante que u e s tenham parte
    real positiva (onda decaindo com a distância).

  Passo 3 — Constantes de propagação (npt × n):
    Para i = 1..npt, j = 1..n:
      u(i,j) = sqrt(kr(i)² - kh2(j))   (modo TE)
      v(i,j) = sqrt(kr(i)² - kv2(j))   (intermediário)
      s(i,j) = sqrt(lamb2(j)) * v(i,j)  (modo TM, inclui anisotropia)
    A raiz quadrada complexa usa a convenção Re(sqrt) > 0 para garantir
    decaimento exponencial das ondas.

  Passo 4 — Admitâncias e impedâncias intrínsecas:
    AdmInt(i,j) = u(i,j) / zeta         (admitância TE)
    ImpInt(i,j) = s(i,j) / eta(j,1)     (impedância TM)

  Passo 5 — Produtos espessura × constante (npt × n):
    uh(i,j) = u(i,j) * h(j)    (argumento da tanh para TE)
    sh(i,j) = s(i,j) * h(j)    (argumento da tanh para TM)
    Para semi-espaços (h=0): uh=sh=0, e tanh(0)=0 (correto).

  Passo 6 — Tanh estabilizada:
    tghuh(i,j) = (1 - exp(-2*uh(i,j))) / (1 + exp(-2*uh(i,j)))
    tghsh(i,j) = (1 - exp(-2*sh(i,j))) / (1 + exp(-2*sh(i,j)))

  Passo 7 — Recursão descendente (j = n → 2):
    AdmAp_dw(n) = AdmInt(n)    (condição de contorno: semi-espaço)
    Para j = n-1, n-2, ..., 2:
      AdmAp_dw(j) = AdmInt(j) * (AdmAp_dw(j+1) + AdmInt(j)*tghuh(j))
                                / (AdmInt(j) + AdmAp_dw(j+1)*tghuh(j))
      RTEdw(j) = (AdmInt(j) - AdmAp_dw(j+1)) / (AdmInt(j) + AdmAp_dw(j+1))
    (Análogo para ImpAp_dw e RTMdw usando ImpInt e tghsh)

  Passo 8 — Recursão ascendente (j = 1 → n-1):
    AdmAp_up(1) = AdmInt(1)    (condição de contorno: semi-espaço)
    Para j = 2, 3, ..., n-1:
      AdmAp_up(j) = AdmInt(j) * (AdmAp_up(j-1) + AdmInt(j)*tghuh(j))
                                / (AdmInt(j) + AdmAp_up(j-1)*tghuh(j))
      RTEup(j) = (AdmInt(j) - AdmAp_up(j-1)) / (AdmInt(j) + AdmAp_up(j-1))
    (Análogo para ImpAp_up e RTMup)
```

#### 5.3.4 commonfactorsMD — Fatores de Onda da Camada Fonte

Calcula os 6 fatores de onda refletida (Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz) para a camada do transmissor. Estes fatores encapsulam as reflexões múltiplas dentro da camada fonte e são reutilizados para todos os dipolos.

**Otimização:** Esta sub-rotina só precisa ser recalculada quando a distância horizontal `r` entre T e R muda, ou quando a camada do transmissor muda. Em configurações coaxiais (r constante), o custo é amortizado.

#### 5.3.5 RtHR — Rotação do Tensor Magnético

Implementa a rotação `R^T * H * R` conforme Liu (2017), equação 4.80. Recebe os três ângulos de Euler (alpha, beta, gamma) e o tensor H(3x3) complexo, retornando o tensor rotacionado no sistema de coordenadas da ferramenta.

Na perfilagem inclinada, `alpha = theta` (inclinação do poço), `beta = gamma = 0`.

### 5.4 magneticdipoles.f08 — Campos Dipolares

#### 5.4.1 hmd_TIV_optimized — Dipolo Magnético Horizontal

Sub-rotina principal para o cálculo do campo do HMD em meio TIV. Trata 6 configurações geométricas:

| Caso | Condição | Descrição |
|:-----|:---------|:----------|
| 1 | `camadR == 1 .and. camadT /= 1` | Receptor no semi-espaço superior |
| 2 | `camadR < camadT` | Receptor acima do transmissor (camada intermediária) |
| 3 | `camadR == camadT .and. z <= h0` | Mesma camada, receptor acima de T |
| 4 | `camadR == camadT .and. z > h0` | Mesma camada, receptor abaixo de T |
| 5 | `camadR > camadT .and. camadR /= n` | Receptor abaixo do transmissor (camada intermediária) |
| 6 | `camadR == n` | Receptor no semi-espaço inferior |

**Explicação dos 6 casos geométricos:**

Os 6 casos surgem porque o cálculo dos campos requer expressões diferentes dependendo
da posição relativa entre transmissor e receptor no modelo de camadas:

```
  Caso 1 (semi-espaço superior):
    O receptor está no semi-espaço superior e o transmissor em outra camada.
    O campo é puramente transmitido de baixo para cima através de todas as
    interfaces entre camadT e a superfície. Usa coeficientes TEupz/TMup.

  Caso 2 (receptor acima, camada intermediária):
    O receptor está acima do transmissor, mas não no semi-espaço. O campo
    inclui transmissão ascendente e reflexões nas interfaces acima e abaixo
    do receptor. Usa coeficientes de transmissão recursivos ascendentes.

  Caso 3 (mesma camada, receptor acima):
    T e R na mesma camada, com R acima de T. O campo inclui o termo direto
    (exp(-u*|z-h0|)) mais reflexões nas fronteiras da camada. Os fatores
    Euup/Mxup capturam as reflexões vindas de cima.

  Caso 4 (mesma camada, receptor abaixo):
    Simétrico ao Caso 3. Usa fatores Eudw/Mxdw para reflexões de baixo.
    Nota: Casos 3 e 4 cobrem a situação mais comum no projeto (poço vertical,
    dTR = 1.0 m), onde T e R frequentemente estão na mesma camada.

  Caso 5 (receptor abaixo, camada intermediária):
    Simétrico ao Caso 2, com transmissão descendente.

  Caso 6 (semi-espaço inferior):
    Simétrico ao Caso 1, com transmissão para o semi-espaço inferior.
```

Para cada caso, os kernels espectrais `Ktm` (modo TM) e `Kte` (modo TE) são calculados usando os coeficientes de transmissão e reflexão apropriados, e então convolvidos com os pesos do filtro de Hankel.

**Modos de dipolo suportados:**
- `'hmdx'`: Apenas dipolo horizontal x (saída: Hx(1,1), Hy(1,1), Hz(1,1))
- `'hmdy'`: Apenas dipolo horizontal y
- `'hmdxy'`: Ambos simultaneamente (saída: Hx(1,2), Hy(1,2), Hz(1,2))

O modo `'hmdxy'` é o utilizado no simulador, pois calcula ambos os dipolos compartilhando os kernels.

#### 5.4.2 vmd_optimized — Dipolo Magnético Vertical

O VMD excita apenas o modo TE (simetria axial). A estrutura é similar ao HMD, com 6 casos geométricos. Os coeficientes de transmissão `TEdwz`/`TEupz` são calculados recursivamente.

### 5.5 PerfilaAnisoOmp.f08 — Módulo Principal

#### 5.5.1 perfila1DanisoOMP — Loop de Perfilagem

Função principal do simulador. Etapas:

1. **Cálculo de nmed:** Número de medições por ângulo baseado na janela `tj` e passo `p_med`
2. **Carregamento do filtro:** `J0J1Wer(201, ...)` — filtro de Werthmuller
3. **Preparação da geometria:** `sanitize_hprof_well(n, esp, h, prof)`
4. **Configuração OpenMP:** Paralelismo aninhado com threads dinâmicos
5. **Loop duplo paralelo:**
   - Loop externo: ângulos (`k = 1..ntheta`), `num_threads_k = ntheta`
   - Loop interno: medições (`j = 1..nmed(k)`), `num_threads_j = maxthreads - ntheta`
6. **Coleta e escrita:** `writes_files(...)` escreve .dat e .out

#### 5.5.2 fieldsinfreqs — Campos em Todas as Frequências

Para cada posição T-R, calcula o tensor H completo em todas as frequências. Cadeia de chamadas:

```
findlayersTR2well → commonarraysMD → commonfactorsMD
                                         │
                    ┌────────────────────┤
                    v                    v
            hmd_TIV_optimized    vmd_optimized
                    │                    │
                    v                    v
               matH(1:2,:)          matH(3,:)
                         │
                         v
                    RtHR(ang, 0, 0, matH)
                         │
                         v
                      cH(f,:) = [tH(1,1), tH(1,2), ..., tH(3,3)]
```

#### 5.5.3 writes_files — Escrita de Saída

**Arquivo .out (metadados):** Escrito apenas quando `modelm == nmaxmodel` (último modelo):

```
Linha 1: nt nf nmaxmodel     (núm. ângulos, núm. frequências, núm. modelos)
Linha 2: theta(1) ... theta(nt)   (ângulos)
Linha 3: freq(1) ... freq(nf)    (frequências)
Linha 4: nmeds(1) ... nmeds(nt)  (núm. medições por ângulo)
```

**Arquivo .dat (dados binários):** Escrito em modo `stream` (sem registros fixos), `append`:

```
Para cada ângulo k, frequência j, medição i:
  write(1000) i, zobs, rho_h, rho_v, Re(H11), Im(H11), ..., Re(H33), Im(H33)
              ^    ^      ^      ^     ^                                    ^
           int32  real64  real64  real64   9 × (2 × real64 = 16 bytes)
           4 bytes 8 bytes 8 bytes 8 bytes    = 144 bytes
                                                                    
  Total por registro: 4 + 3×8 + 18×8 = 4 + 24 + 144 = 172 bytes
  Formato: 1 int32 + 21 float64 = 22 valores por registro
```

### 5.6 RunAnisoOmp.f08 — Programa Principal

Programa sequencial que:
1. Obtém o diretório corrente via `getcwd`
2. Lê o arquivo `model.in` sequencialmente
3. Aloca arrays de resistividade e espessura
4. Chama `perfila1DanisoOMP(...)` com todos os parâmetros

---

## 6. Arquivo de Entrada model.in

### 6.1 Estrutura Completa

O arquivo `model.in` é lido sequencialmente pelo programa principal. Cada linha contém um ou mais valores:

```
Linha  Variável            Tipo      Descrição
─────  ─────────────────  ────────  ────────────────────────────────────────────
  1    nf                 integer   Número de frequências
  2    freq(1)            real(dp)  Frequência 1 (Hz)
  ...  freq(nf)           real(dp)  Frequência nf (Hz)
  +1   ntheta             integer   Número de ângulos de inclinação
  +2   theta(1)           real(dp)  Ângulo 1 (graus, 0-90)
  ...  theta(ntheta)      real(dp)  Ângulo ntheta (graus)
  +1   h1                 real(dp)  Altura do 1º ponto-médio T-R (m, acima da 1ª interface)
  +1   tj                 real(dp)  Tamanho da janela de investigação (m)
  +1   p_med              real(dp)  Passo entre medições (m)
  +1   dTR                real(dp)  Distância Transmissor-Receptor (m)
  +1   filename           char      Nome base dos arquivos de saída
  +1   ncam               integer   Número total de camadas (incluindo semi-espaços)
  +2   resist(1,1:2)      real(dp)  rho_h, rho_v da camada 1 (semi-espaço superior)
  ...  resist(ncam,1:2)   real(dp)  rho_h, rho_v da camada ncam
  +1   esp(2)             real(dp)  Espessura da camada 2 (m)
  ...  esp(ncam-1)        real(dp)  Espessura da camada ncam-1 (m)
  +1   modelm nmaxmodel   integer   Modelo atual e total de modelos
```

### 6.2 Exemplo Comentado (model.in atual)

```
2                    ! nf = 2 frequências
20000.0              ! freq(1) = 20 kHz
40000.0              ! freq(2) = 40 kHz
1                    ! ntheta = 1 ângulo
0.0                  ! theta(1) = 0 graus (poço vertical)
10.0                 ! h1 = 10 m acima da 1ª interface
120.0                ! tj = 120 m de janela de investigação
0.2                  ! p_med = 0.2 m entre medições
1.0                  ! dTR = 1.0 m de espaçamento T-R
Inv0_15Dip1000_t5    ! nome dos arquivos de saída
10                   ! ncam = 10 camadas
1.38    1.38         ! Camada 1 (semi-espaço): rho_h=1.38, rho_v=1.38
1.76    1.76         ! Camada 2
107.07  107.08       ! Camada 3 (alta resistividade — possível reservatório)
102.19  102.2        ! Camada 4
1474.11 1474.26      ! Camada 5 (resistividade muito alta — possível sal/carbonato)
2.58    2.58         ! Camada 6
9.48    9.48         ! Camada 7
0.35    0.35         ! Camada 8 (baixíssima — água salgada)
582.74  582.82       ! Camada 9
0.3     0.3          ! Camada 10 (semi-espaço inferior)
0.5                  ! esp(2) = 0.5 m
42.03                ! esp(3) = 42.03 m
0.57                 ! esp(4) = 0.57 m (camada fina!)
3.71                 ! esp(5) = 3.71 m
27.4                 ! esp(6) = 27.4 m
12.63                ! esp(7) = 12.63 m
11.91                ! esp(8) = 11.91 m
1.26                 ! esp(9) = 1.26 m
1000 1000            ! modelo 1000 de 1000
```

### 6.3 Observações Importantes

1. **Espessuras:** As camadas 1 e ncam são semi-espaços (espessura infinita), então `esp(1) = esp(ncam) = 0` é atribuído em `RunAnisoOmp.f08` (programa principal) antes de chamar `perfila1DanisoOMP`.

2. **Frequências:** O simulador padrão usa 2 frequências (20 kHz e 40 kHz). A primeira é a mesma do pipeline (`FREQUENCY_HZ = 20000.0`).

3. **Número de medições:** Para `theta = 0`, `pz = p_med = 0.2`, `nmed = ceil(120/0.2) = 600`, consistente com `SEQUENCE_LENGTH = 600` do pipeline.

4. **Anisotropia neste exemplo:** O modelo mostra anisotropia muito fraca (`lambda ~ 1.0` em quase todas as camadas). Em cenários reais, `lambda` varia de 1.0 a sqrt(2).

---

## 7. Arquivos de Saída (.dat e .out)

### 7.1 Arquivo .out — Metadados

Arquivo texto com 4 linhas, escrito apenas pelo último modelo da série:

```
Formato:
  Linha 1: nt  nf  nmaxmodel
  Linha 2: theta(1)  theta(2) ... theta(nt)
  Linha 3: freq(1)  freq(2)  ... freq(nf)
  Linha 4: nmeds(1)  nmeds(2) ... nmeds(nt)
```

**Exemplos de .out existentes:**

| Arquivo | nt | nf | nmodels | Ângulos | Frequências | nmeds |
|:--------|:--:|:--:|:-------:|:--------|:------------|:------|
| t2 | 2 | 1 | 1000 | 0, 15 | 20000 | 600, 622 |
| t4 | 2 | 2 | 1000 | 0, 15 | 20000, 40000 | 600, 622 |
| t5 | 1 | 2 | 1000 | 0 | 20000, 40000 | 600 |

### 7.2 Arquivo .dat — Dados Binários

Arquivo binário no formato Fortran `stream` (sem registros fixos), escrito em modo `append`.

**Formato por registro:**

| Posição | Variável | Tipo | Bytes | Descrição |
|:--------|:---------|:-----|------:|:----------|
| 0 | i | int32 | 4 | Índice da medição |
| 4 | zobs | float64 | 8 | Profundidade do ponto-médio T-R (m) |
| 12 | rho_h | float64 | 8 | Resistividade horizontal verdadeira (Ohm.m) |
| 20 | rho_v | float64 | 8 | Resistividade vertical verdadeira (Ohm.m) |
| 28 | Re(Hxx) | float64 | 8 | Parte real de H(1,1) |
| 36 | Im(Hxx) | float64 | 8 | Parte imaginária de H(1,1) |
| 44 | Re(Hxy) | float64 | 8 | Parte real de H(1,2) |
| 52 | Im(Hxy) | float64 | 8 | Parte imaginária de H(1,2) |
| 60 | Re(Hxz) | float64 | 8 | Parte real de H(1,3) |
| 68 | Im(Hxz) | float64 | 8 | Parte imaginária de H(1,3) |
| 76 | Re(Hyx) | float64 | 8 | Parte real de H(2,1) |
| 84 | Im(Hyx) | float64 | 8 | Parte imaginária de H(2,1) |
| 92 | Re(Hyy) | float64 | 8 | Parte real de H(2,2) |
| 100 | Im(Hyy) | float64 | 8 | Parte imaginária de H(2,2) |
| 108 | Re(Hyz) | float64 | 8 | Parte real de H(2,3) |
| 116 | Im(Hyz) | float64 | 8 | Parte imaginária de H(2,3) |
| 124 | Re(Hzx) | float64 | 8 | Parte real de H(3,1) |
| 132 | Im(Hzx) | float64 | 8 | Parte imaginária de H(3,1) |
| 140 | Re(Hzy) | float64 | 8 | Parte real de H(3,2) |
| 148 | Im(Hzy) | float64 | 8 | Parte imaginária de H(3,2) |
| 156 | Re(Hzz) | float64 | 8 | Parte real de H(3,3) |
| 164 | Im(Hzz) | float64 | 8 | Parte imaginária de H(3,3) |

**Total por registro:** 4 + 21 x 8 = **172 bytes**

### 7.3 Leitura no Pipeline Python (geosteering_ai)

O pipeline Python (`geosteering_ai/data/loading.py`) lê o .dat interpretando os registros binários de 172 bytes. O pseudocódigo ilustrativo abaixo mostra a lógica conceitual (a implementação real em `loading.py` utiliza `np.frombuffer` com mapeamento `COL_MAP_22`):

```python
# Pseudocódigo ilustrativo (NÃO é o código real de loading.py):
dtype = np.dtype([
    ('meds', np.int32),       # Col 0: índice de medição
    ('values', np.float64, 21) # Cols 1-21: zobs, rho_h, rho_v, 18 EM
])
data = np.fromfile(filepath, dtype=dtype)

# Reorganização em 22 colunas:
# Col 0:  meds (metadata)
# Col 1:  zobs → INPUT_FEATURE
# Col 2:  rho_h → OUTPUT_TARGET
# Col 3:  rho_v → OUTPUT_TARGET
# Col 4:  Re(Hxx) → INPUT_FEATURE
# Col 5:  Im(Hxx) → INPUT_FEATURE
# ...
# Col 20: Re(Hzz) → INPUT_FEATURE
# Col 21: Im(Hzz) → INPUT_FEATURE
```

### 7.4 Tamanho dos Arquivos

Para 1000 modelos, 1 ângulo, 2 frequências, 600 medições:

```
Tamanho = nmodels × ntheta × nfreq × nmeds × 172 bytes
        = 1000 × 1 × 2 × 600 × 172
        = 206,400,000 bytes ≈ 197 MB
```

---

## 8. Sistema de Build (Makefile)

### 8.1 Estrutura do Makefile

O Makefile segue a convenção padrão de projetos Fortran com:

| Seção | Descrição |
|:------|:----------|
| SETTINGS | Binário (`tatu.x`), extensões, diretório de build (`./build`) |
| DEPENDENCIES | Ordem de compilação: `parameters → filtersv2 → utils → magneticdipoles → PerfilaAnisoOmp → RunAnisoOmp` |
| COMPILER | `gfortran` com flags de produção |
| TARGETS | `$(binary)`, `run_python`, `all`, `clean`, `cleanall` |

### 8.2 Flags de Compilação

**Flags de desenvolvimento (comentadas, para debug):**

```makefile
-g             # Símbolos de debug
-fcheck=all    # Verificação de bounds, overflow, etc.
-fbacktrace    # Stack trace em caso de erro
```

Para ativar o modo de desenvolvimento, descomente `flags = $(development_flags_gfortran)` e comente `flags = $(production_flags_gfortran)` no Makefile.

**Flags de produção (ativas):**

```makefile
-J$(build)     # Diretório para arquivos .mod
-std=f2008     # Padrão Fortran 2008
-pedantic      # Avisos de conformidade com o padrão
-Wall -Wextra  # Todos os avisos
-Wimplicit-interface  # Erro em interfaces implícitas
-fPIC          # Posição independente (permite linking dinâmico)
-fmax-errors=1 # Para na primeira falha
-O3            # Otimização agressiva
-march=native  # Otimização para a arquitetura local
-ffast-math    # Otimizações matemáticas agressivas
-funroll-loops # Desenrola loops para otimização
-fall-intrinsics # Habilita todas as funções intrínsecas
```

**Flags OpenMP:** `-fopenmp` adicionada tanto na compilação quanto na linkagem.

### 8.3 Targets

| Target | Descrição |
|:-------|:----------|
| `$(binary)` (`tatu.x`) | Compila e linka todos os .o em um executável |
| `run_python` | Executa `fifthBuildTIVModels.py` (gerador de modelos) |
| `all` | Compila + executa Python |
| `clean` | Remove `./build/` |
| `cleanall` | Remove `./build/`, `tatu.x`, `*.dat`, `*.out` |

### 8.4 Ordem de Compilação

```
parameters.f08 ──> build/parameters.o
filtersv2.f08  ──> build/filtersv2.o
utils.f08      ──> build/utils.o      (depende de parameters)
magneticdipoles.f08 ──> build/magneticdipoles.o (depende de parameters)
PerfilaAnisoOmp.f08 ──> build/PerfilaAnisoOmp.o (depende de todos acima + omp_lib)
RunAnisoOmp.f08 ──> build/RunAnisoOmp.o (depende de parameters, DManisoTIV)

Linkagem: gfortran -fopenmp [flags] -o tatu.x build/*.o
```

### 8.5 Notas e Recomendações

- **`-ffast-math`** pode afetar a precisão de operações com NaN e infinitos. Para validação numérica, recomenda-se compilar sem esta flag.
- **`-march=native`** otimiza para a CPU local, mas o binário não é portável.
- O diretório `./build` é criado automaticamente via `$(shell mkdir -p $(build))`.

---

## 9. Gerador de Modelos Geológicos (Python)

### 9.1 Visão Geral

O script `fifthBuildTIVModels.py` gera modelos geológicos aleatórios usando amostragem Sobol Quasi-Monte Carlo (`scipy.stats.qmc.Sobol`). Os modelos são escritos em `model.in` e simulados pelo Fortran em sequência.

### 9.2 Parâmetros dos Modelos Geológicos

| Parâmetro | Range | Distribuição | Descrição |
|:----------|:------|:------------|:----------|
| `n_layers` | 3-80 | Empírica (ponderada) ou uniforme | Número de camadas |
| `rho_h` | 0.05-1500 Ohm.m | Log-uniforme (Sobol) | Resistividade horizontal |
| `lambda` | 1.0-sqrt(2) | Uniforme (Sobol), correlacionada com rho_h | Coeficiente de anisotropia |
| `rho_v` | calculado | `lambda^2 * rho_h` | Resistividade vertical |
| `espessuras` | 0.1-50+ m | Sobol + stick-breaking | Espessuras das camadas internas |

### 9.3 Cenários de Geração (6 Geradores)

| Cenário | Função | nmodels | ncam | Espessuras | Contrastes | Ruído |
|:--------|:-------|:-------:|:----:|:-----------|:-----------|:------|
| **Baseline empírico** | `baseline_empirical_2` | 18000 | 3-30 (pesos empíricos) | Standard (min 1.0 m) | Natural | Não |
| **Baseline uniforme** | `baseline_ncamuniform_2` | 9000 | 3-80 (uniforme) | Standard (min 0.5 m) | Natural | Não |
| **Camadas grossas** | `baseline_thick_thicknesses_2` | 9000 | 3-14 | Grossas (min 10 m, p=0.7) | Forçados em grossas | Não |
| **Desfavorável empírico** | `unfriendly_empirical_2` | 12000 | 3-30 (pesos) | Finas (min 0.2 m, p=0.6) | Forçados (5x, p=0.5) | Não |
| **Desfavorável ruidoso** | `unfriendly_noisy_2` | 12000 | 3-30 (pesos) | Finas + ruído (3%) | Forçados + ruído (5%) | Sim |
| **Patológico** | `generate_pathological_models_2` | 4500 | 3-28 | Muito finas (min 0.1 m) | Extremos (10x, p=0.7) | Sim (7%) |

### 9.4 Funções Auxiliares e Vantagens do Sobol QMC

| Função | Descrição |
|:-------|:----------|
| `log_transform_2` | Transforma Sobol [0,1] para escala log-uniforme [min, max] |
| `uniform_transform_2` | Transforma Sobol [0,1] para escala linear [min, max] |
| `generate_thicknesses_2` | Sobol stick-breaking com espessura mínima garantida |
| `generate_thick_thicknesses_2` | Forçamento de camadas grossas (>15 m) com probabilidade `p_thick` |
| `generate_thin_thicknesses_2` | Forçamento de camadas finas (~min) com probabilidade `p_thin` |
| `conditional_rho_h_sampling_core_2` | Força contrastes entre camadas adjacentes |
| `conditional_rho_h_with_thickness_2` | Força resistividades extremas em camadas grossas |
| `correlated_lambda_sampling_core_2` | Lambda correlacionado com resistividade (rho alto → lambda alto) |

**Sobol Quasi-Monte Carlo vs Pseudo-Aleatório:**

A escolha de sequências Sobol (quasi-aleatórias) em vez de geradores pseudo-aleatórios
tradicionais (como NumPy `np.random`) é deliberada e traz vantagens significativas para
a geração de modelos geológicos:

```
  Pseudo-aleatório (Monte Carlo):         Sobol (Quasi-Monte Carlo):
  ┌────────────────────────────┐          ┌────────────────────────────┐
  │ • • •  •    •   •  •      │          │ •   •   •   •   •   •     │
  │    •  •    • • •   •   •  │          │   •   •   •   •   •   •   │
  │  •  •  •••   •      • •   │          │ •   •   •   •   •   •     │
  │    •     •  •   ••  •     │          │   •   •   •   •   •   •   │
  │  •  •  •  •      • •  •   │          │ •   •   •   •   •   •     │
  └────────────────────────────┘          └────────────────────────────┘
  Clusters e vazios aleatórios             Cobertura uniforme garantida
```

Vantagens do Sobol para geração de modelos geológicos:

1. **Cobertura uniforme do espaço de parâmetros:** Sequências Sobol preenchem o
   hipercubo [0,1]^d de forma mais uniforme que amostras aleatórias, evitando
   clusters e lacunas. Para d = ncam + ncam + (ncam-2) dimensões, isso garante
   que todas as combinações de (rho_h, lambda, espessura) sejam representadas.

2. **Convergência mais rápida:** A taxa de convergência do QMC é O(1/N) com
   constante pequena, versus O(1/sqrt(N)) para MC. Isso significa que ~1000
   amostras Sobol cobrem o espaço tão bem quanto ~10000 amostras pseudo-aleatórias.

3. **Reprodutibilidade determinística:** Para um mesmo `seed` e `dimensão`, a
   sequência Sobol é idêntica, garantindo reprodutibilidade perfeita dos modelos
   geológicos gerados.

4. **Correlação cruzada baixa:** Cada dimensão da sequência Sobol é quase
   independente das demais, evitando correlações espúrias entre parâmetros
   que distorceriam a distribuição dos modelos.

### 9.5 Fluxo do Gerador

```
Para cada modelo i = 1..nmodels:
  1. Sorteia ncam (número de camadas)
  2. Gera amostra Sobol de dimensão (ncam + ncam + ncamint-1)
  3. Fatia amostra em: rho_h_portion, lambda_portion, thickness_portion
  4. Transforma rho_h (log-uniforme)
  5. Aplica forçamento condicional de contrastes
  6. Transforma lambda (uniforme, correlacionada)
  7. Calcula rho_v = lambda^2 * rho_h
  8. Gera espessuras (stick-breaking + min)
  9. Escreve model.in
  10. Executa tatu.x via subprocess
  11. Resultado: append no .dat
```

### 9.6 Loop de Execução via Subprocess

O gerador Python orquestra a execução do simulador Fortran via `subprocess.run()`.
Para cada modelo geológico gerado, o fluxo é:

```python
# Pseudocódigo do loop de execução (simplificado):
for i in range(1, nmodels + 1):
    # 1. Gera parâmetros do modelo i
    ncam, rho_h, rho_v, espessuras = generate_model(i, ...)

    # 2. Escreve model.in com os parâmetros
    write_model_in(
        nf=2, freqs=[20000.0, 40000.0],
        ntheta=1, thetas=[0.0],
        h1=10.0, tj=120.0, p_med=0.2, dTR=1.0,
        filename=output_name,
        ncam=ncam, resist=np.column_stack([rho_h, rho_v]),
        esp=espessuras,
        modelm=i, nmaxmodel=nmodels
    )

    # 3. Executa o simulador Fortran
    result = subprocess.run(
        ['./tatu.x'],
        cwd=fortran_dir,
        capture_output=True,
        timeout=300  # 5 minutos por modelo (segurança)
    )

    # 4. Verifica sucesso
    if result.returncode != 0:
        logger.error(f"Modelo {i} falhou: {result.stderr}")
        continue

    # O resultado é appendado ao .dat automaticamente pelo Fortran
```

**Detalhes importantes do subprocess:**

- O Fortran lê `model.in` do diretório corrente e escreve `.dat` e `.out` no
  mesmo diretório. O `cwd=fortran_dir` garante que os arquivos sejam encontrados.
- O modo `append` do Fortran (`position='append'`) acumula os resultados de todos
  os modelos em um único arquivo `.dat`, sem necessidade de concatenação posterior.
- O `.out` (metadados) é escrito apenas quando `modelm == nmaxmodel`, então não é
  sobrescrito a cada modelo.
- Para 1000 modelos com ~2.4 s/modelo, o tempo total é ~40 minutos (single-thread)
  ou ~5-10 minutos com OpenMP em 8 cores.

---

## 10. Paralelismo OpenMP — Análise e Otimização

### 10.1 Estrutura Atual de Paralelismo

O simulador utiliza **paralelismo OpenMP aninhado em 2 níveis**:

```
Nível 1 (externo): Ângulos theta
  !$omp parallel do schedule(dynamic) num_threads(num_threads_k)
  do k = 1, ntheta   ← tipicamente 1-2 threads

    Nível 2 (interno): Medições por ângulo
    !$omp parallel do schedule(dynamic) num_threads(num_threads_j)
    do j = 1, nmed(k)   ← 600+ threads disponíveis
      ...
    end do
    !$omp end parallel do

  end do
  !$omp end parallel do
```

**Distribuição de threads:**

```fortran
maxthreads = omp_get_max_threads()      ! Total de threads disponíveis
num_threads_k = ntheta                  ! Threads para ângulos (1-2)
num_threads_j = maxthreads - ntheta     ! Restante para medições
```

### 10.2 Análise de Desempenho

**Problemas identificados:**

1. **Desbalanceamento de carga no nível externo:** Com `ntheta = 1` (comum), o loop externo usa apenas 1 thread, desperdiçando o segundo nível aninhado.

2. **Alocação de memória dentro do loop paralelo:** Cada chamada a `commonarraysMD` e `hmd_TIV_optimized` aloca arrays temporários. Isso causa:
   - Contenção no alocador de memória (malloc)
   - Fragmentação de heap
   - Overhead de alocação/desalocação repetida

3. **Escalonamento `dynamic` para loop regular:** Se todas as medições têm custo computacional similar (mesmo modelo), `static` seria mais eficiente (sem overhead de despacho).

4. **Redundância computacional:** `commonarraysMD` calcula constantes de propagação e coeficientes de reflexão que dependem apenas do modelo (n, resist, freq), não da posição T-R. Em medições onde T e R estão na mesma camada (maioria dos casos em poços verticais), `commonarraysMD` produz o mesmo resultado para todos os `j`.

### 10.3 Pontos de Otimização — Memória

| Otimização | Impacto | Complexidade |
|:-----------|:--------|:-------------|
| **Pré-alocar arrays de trabalho por thread** | Alto — elimina malloc dentro do loop | Média |
| **Mover `commonarraysMD` para fora do loop j** | Alto — computa uma vez por (ângulo, freq) quando r é constante | Média |
| **Usar `firstprivate` para arrays somente-leitura** | Médio — evita cópias desnecessárias | Baixa |
| **Reduzir arrays temporários em hmd_TIV** | Médio — Tudw/Txdw/etc. alocados com tamanho mínimo | Média |
| **Pool de memória por thread** | Alto — elimina fragmentação | Alta |

### 10.4 Pontos de Otimização — Tempo de Execução

| Otimização | Impacto Estimado | Descrição |
|:-----------|:----------------|:----------|
| **Cache de `commonarraysMD`** | 30-50% redução | Computa uma vez por (camadT, freq), reutiliza para todos os j com mesmo camadT |
| **Paralelismo sobre frequências** | 2x para nf=2 | Adicionar nível de paralelismo sobre frequências |
| **Colapso de loops** | 10-20% | `!$omp parallel do collapse(2)` para ângulos × medições |
| **SIMD para convolução Hankel** | 10-30% | `!$omp simd` nos sums de kernels × pesos do filtro |
| **Scheduler híbrido** | 5-10% | `static` quando nmed uniforme, `dynamic` quando variável |
| **Batch de medições** | 15-25% | Agrupar medições por camadT para reutilizar coeficientes de reflexão |

### 10.5 Proposta de Paralelismo Otimizado

```fortran
! PROPOSTA: Colapso de loops + pré-alocação + cache de coeficientes
!
! 1. Pré-alocar workspace por thread:
!$omp parallel private(tid, ws)
tid = omp_get_thread_num()
ws = workspace_pool(tid)  ! Arrays pré-alocados

! 2. Loop colapsado (ângulos × medições):
!$omp do schedule(dynamic, chunk_size) collapse(2)
do k = 1, ntheta
  do j = 1, nmmax
    if (j > nmed(k)) cycle
    ! ... computa usando ws ...
  end do
end do
!$omp end do
!$omp end parallel

! 3. Cache de commonarraysMD por (camadT, freq):
! Identificar camadT para cada posição j (depende apenas de Tz)
! Agrupar por camadT e computar commonarraysMD uma vez por grupo
```

**Exemplo concreto de pré-alocação de workspace:**

```fortran
! Definição do tipo workspace (arrays de trabalho por thread):
type :: thread_workspace
    complex(dp), allocatable :: u(:,:), s(:,:)       ! (npt, n)
    complex(dp), allocatable :: uh(:,:), sh(:,:)     ! (npt, n)
    complex(dp), allocatable :: RTEdw(:,:), RTEup(:,:)  ! (npt, n)
    complex(dp), allocatable :: RTMdw(:,:), RTMup(:,:)  ! (npt, n)
    complex(dp), allocatable :: Mxdw(:), Mxup(:)     ! (npt)
    complex(dp), allocatable :: Eudw(:), Euup(:)     ! (npt)
    complex(dp), allocatable :: Hx(:,:), Hy(:,:), Hz(:,:) ! (npt, 2)
end type

! Alocação única antes do loop paralelo:
type(thread_workspace) :: ws_pool(0:maxthreads-1)
do t = 0, maxthreads - 1
    allocate(ws_pool(t)%u(npt, n_max))
    allocate(ws_pool(t)%s(npt, n_max))
    ! ... demais arrays ...
end do

! Uso dentro do loop paralelo (sem malloc!):
!$omp parallel private(tid)
tid = omp_get_thread_num()
!$omp do schedule(dynamic, 16)
do j = 1, nmed
    call commonarraysMD_preallocated(ws_pool(tid), ...)
    call hmd_TIV_preallocated(ws_pool(tid), ...)
end do
!$omp end do
!$omp end parallel

! Desalocação após o loop:
do t = 0, maxthreads - 1
    deallocate(ws_pool(t)%u)
    ! ...
end do
```

### 10.6 Métricas de Escalabilidade

Para 1000 modelos, 1 ângulo, 2 frequências, 600 medições:

```
Computações por modelo:
  = ntheta × nfreq × nmed × (commonarraysMD + commonfactorsMD + hmd + vmd + rotação)
  = 1 × 2 × 600 × (~201*n operações complexas)
  = 1200 × (~201*10 = 2010 FLOPs complexos)
  ≈ 2.4 × 10^6 operações complexas por modelo

Total para 1000 modelos: ≈ 2.4 × 10^9 operações
Tempo estimado (single core, ~1 GFLOP/s complex): ~2.4 segundos por modelo
```

---

## 11. Análise de Viabilidade CUDA (GPU)

### 11.1 Identificação de Kernels Paralelizáveis

| Kernel | Dimensão | Paralelismo | Adequação GPU |
|:-------|:---------|:------------|:-------------|
| `commonarraysMD` | npt × n | Independente por ponto do filtro | **Alta** — 201 threads independentes |
| `commonfactorsMD` | npt | Independente por ponto do filtro | **Alta** — 201 threads |
| Convolução Hankel (sum) | npt | Redução paralela | **Alta** — redução clássica |
| Loop medições | nmed | Independente por medição | **Alta** — 600+ threads |
| Loop frequências | nf | Independente por frequência | **Média** — apenas 2 frequências |
| Recursão reflexão | n (sequencial) | **Dependência de dados** | **Baixa** — n camadas sequenciais |

### 11.2 Estratégia de Implementação CUDA

```
GPU Grid:
  Block dimension: (npt=201 threads, 1, 1) — um thread por ponto do filtro
  Grid dimension:  (nmed, nf, ntheta) — uma thread-block por medição/freq/ângulo

Kernel 1: compute_propagation_constants
  Input: kr(201), eta(n,2), zeta, h(n)
  Output: u(201,n), s(201,n), uh(201,n), sh(201,n)
  Paralelismo: cada thread computa para um kr_i

Kernel 2: compute_reflection_coefficients
  Input: u, s, uh, sh, AdmInt, ImpInt
  Output: RTEdw, RTEup, RTMdw, RTMup
  ATENÇÃO: recursão sequencial sobre camadas (n)
  Estratégia: cada thread processa um kr_i independentemente,
              mas faz a recursão sobre n camadas sequencialmente

Kernel 3: compute_wave_factors
  Input: RTEdw, RTEup, RTMdw, RTMup, camadT, h0
  Output: Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz
  Paralelismo: independente por kr_i

Kernel 4: compute_hmd_vmd
  Input: todos os fatores + geometria
  Output: Hx, Hy, Hz para HMD e VMD
  Inclui redução paralela para convolução Hankel

Kernel 5: assemble_tensor_and_rotate
  Input: H_hmd, H_vmd, ângulos
  Output: tensor H(3,3) rotacionado
  Paralelismo: independente por medição
```

### 11.3 Desafios CUDA

1. **Recursão dos coeficientes de reflexão:** Os coeficientes são calculados de forma recursiva (camada n para 1, e 1 para n). Cada passo depende do anterior, impedindo paralelismo nesta dimensão. Porém, os 201 pontos do filtro são independentes, permitindo 201 threads executando a recursão simultaneamente.

2. **Alocação dinâmica:** O código Fortran aloca arrays de tamanho variável (`Tudw`, `Txdw`) dependendo da posição relativa T-R. Na GPU, estes arrays devem ser pré-alocados com tamanho máximo.

3. **Divergência de branch:** Os 6 casos geométricos em `hmd_TIV_optimized` causam divergência de warp em GPUs NVIDIA. Estratégia: agrupar medições por caso geométrico para minimizar divergência.

4. **Memória:** Para n=80 camadas, npt=201: arrays de tamanho (201, 80) × complex64 = 201 × 80 × 16 bytes = ~257 KB por medição. Caberiam ~800 medições simultâneas em 200 MB de memória global.

### 11.4 Análise Detalhada de Memória GPU

Para uma estimativa mais precisa do uso de memória na GPU, considere os arrays
necessários por medição:

```
Arrays por medição (n_max = 80 camadas, npt = 201):

  Constantes de propagação:
    u(npt, n)       = 201 × 80 × 16 bytes = 257,280 bytes (251 KB)
    s(npt, n)       = 201 × 80 × 16 bytes = 257,280 bytes
    v(npt, n)       = 201 × 80 × 16 bytes = 257,280 bytes

  Produtos espessura:
    uh(npt, n)      = 201 × 80 × 16 bytes = 257,280 bytes
    sh(npt, n)      = 201 × 80 × 16 bytes = 257,280 bytes

  Tanh estabilizada:
    tghuh(npt, n)   = 201 × 80 × 16 bytes = 257,280 bytes
    tghsh(npt, n)   = 201 × 80 × 16 bytes = 257,280 bytes

  Coeficientes de reflexão (4 arrays):
    RTEdw, RTEup, RTMdw, RTMup = 4 × 257,280 = 1,029,120 bytes

  Fatores de onda (6 arrays, npt cada):
    Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz = 6 × 201 × 16 = 19,296 bytes

  Admitâncias/impedâncias (2 arrays):
    AdmInt, ImpInt = 2 × 257,280 = 514,560 bytes

  Subtotal por medição: ~3.1 MB

Para GPU com 16 GB (T4, Colab Pro+):
  Memória útil (excluindo overhead CUDA): ~14 GB
  Medições simultâneas (batch): 14,000 / 3.1 ≈ 4,500 medições
  Como nmed = 600, isso permite processar ~7 modelos em batch simultâneo.

Para GPU com 80 GB (A100):
  Medições simultâneas: 75,000 / 3.1 ≈ 24,000 medições
  Modelos em batch: ~40 modelos simultâneos.
```

A GPU é particularmente vantajosa quando múltiplos modelos são processados em batch,
pois o overhead de transferência CPU→GPU é amortizado e a ocupação dos SMs é maximizada.

### 11.5 Estimativa de Speedup

| Componente | CPU (1 core) | GPU (estimado) | Speedup |
|:-----------|:-------------|:---------------|:--------|
| `commonarraysMD` | ~0.5 ms/call | ~0.01 ms/call | 50x |
| Convolução Hankel | ~0.2 ms/sum | ~0.002 ms/sum | 100x |
| Loop medições (600) | 600 × serial | 600 × paralelo | ~100x |
| Overhead transferência | 0 | ~1 ms/modelo | - |
| **Total por modelo** | ~400 ms | ~10 ms | **~40x** |

### 11.6 Frameworks CUDA Recomendados

| Framework | Linguagem | Vantagem | Desvantagem |
|:----------|:----------|:---------|:------------|
| **CUDA Fortran (PGI/NVHPC)** | Fortran | Reuso máximo do código | Compilador proprietário |
| **CUDA C/C++** | C/C++ | Ecossistema maduro, cuBLAS | Reescrita completa |
| **OpenACC** | Fortran | Mínimo de alteração no código | Desempenho menor que CUDA nativo |
| **hipSYCL** | C++ | Portabilidade AMD/NVIDIA | Complexidade adicional |

**Recomendação:** Para prototipagem rápida, **OpenACC** com `nvfortran` (NVIDIA HPC SDK). Para produção, **CUDA C** com wrappers Fortran.

---

## 12. Análise de Reimplementação em Python Otimizado

### 12.1 Motivação

Uma versão Python do simulador permitiria:
- Integração direta com o pipeline `geosteering_ai` (sem subprocess)
- Geração de dados on-the-fly durante o treinamento
- Eliminação da dependência de compilação Fortran
- Facilidade de extensão (novos modelos de ferramenta, multi-frequência)
- Potencial implementação de backpropagation through physics (PINNs)

### 12.2 Bibliotecas para Computação de Alto Desempenho em Python

| Biblioteca | Tipo | Paralelismo | Adequação |
|:-----------|:-----|:------------|:----------|
| **NumPy** | CPU, vetorial | Multi-threaded (BLAS) | Baseline — boa para prototipagem |
| **Numba** | CPU, JIT | Multi-threaded, SIMD | **Excelente** — compilação JIT com tipagem |
| **Numba CUDA** | GPU | CUDA kernels | **Alta** — acesso direto a GPU |
| **CuPy** | GPU, vetorial | CUDA automático | **Alta** — drop-in NumPy para GPU |
| **JAX** | CPU/GPU/TPU | XLA compilation | **Alta** — gradientes automáticos |
| **TensorFlow** | CPU/GPU/TPU | XLA compilation | **Média** — overhead de framework |
| **PyTorch** | CPU/GPU | CUDA | **PROIBIDO** no projeto |
| **empymod** | CPU | Numba JIT | **Referência** — simulador EM 1D existente |

### 12.3 Estratégia de Implementação Python

#### 12.3.1 Fase 1: Protótipo NumPy Vetorizado

```python
# Vetorização do loop de medições:
# Em vez de iterar sobre j=1..nmed, computar todas as medições de uma vez

# Posições T-R (vetorizado para todas as medições):
j = np.arange(nmed)
Tz = z1 + j * pz + Lcos / 2    # shape: (nmed,)
Rz = z1 + j * pz - Lcos / 2    # shape: (nmed,)

# commonarraysMD vetorizado:
# kr: (npt,), u: (npt, n), => broadcast automático
# Resultado: arrays de shape (npt, n) para cada frequência
```

#### 12.3.2 Fase 2: Compilação JIT com Numba

```python
@numba.njit(parallel=True, cache=True)
def compute_reflection_coefficients(kr, eta, zeta, h, n, npt):
    """Calcula coeficientes de reflexão recursivos (Numba JIT)."""
    u = np.empty((npt, n), dtype=np.complex128)
    s = np.empty((npt, n), dtype=np.complex128)
    RTEdw = np.empty((npt, n), dtype=np.complex128)
    RTMdw = np.empty((npt, n), dtype=np.complex128)
    
    for i in numba.prange(npt):  # Paralelo sobre pontos do filtro
        for j in range(n):
            kh2 = -zeta * eta[j, 0]
            u[i, j] = np.sqrt(kr[i]**2 - kh2)
            # ... recursão sequencial sobre camadas
    
    return u, s, RTEdw, RTMdw
```

#### 12.3.3 Fase 3: GPU via Numba CUDA ou CuPy

```python
@numba.cuda.jit
def kernel_commonarraysMD(kr, eta, zeta, h, n, u, s, RTEdw, RTMdw):
    """CUDA kernel para constantes de propagação."""
    i = numba.cuda.grid(1)  # Thread index = ponto do filtro
    if i < npt:
        for j in range(n):
            kh2 = -zeta * eta[j, 0]
            u[i, j] = cmath.sqrt(kr[i]**2 - kh2)
            # ... recursão sobre camadas
```

### 12.4 Referência: empymod

O pacote Python `empymod` (Werthmuller, 2017) já implementa modelagem EM 1D/3D para meios anisotrópicos usando Numba. Principais características:

- Transformadas de Hankel via filtros digitais (mesmos filtros do simulador Fortran)
- Suporte a meios TIV e anisotropia geral
- Compilação JIT via Numba
- Interface NumPy-compatível

**Exemplo de API do empymod (para comparação):**

```python
import empymod

# Cálculo de campo EM para dipolo horizontal em meio TIV:
# empymod usa coordenadas (x, y, z) com z positivo para baixo
result = empymod.bipole(
    src=[0, 0, 500, 0, 0],     # [x1, x2, y1, y2, z] do transmissor
    rec=[100, 0, 500, 0, 0],   # [x1, x2, y1, y2, z] do receptor
    depth=[0, 300, 500, 700],   # interfaces entre camadas (m)
    res=[1e20, 10, 1, 100, 50], # resistividades por camada (Ohm.m)
    # Para TIV, res pode ser um array 2D: res=[rho_h, rho_v] por camada
    aniso=[1, 1, 1.5, 1, 1.2], # lambda = sqrt(rho_v/rho_h) por camada
    freqtime=20000,              # frequência (Hz)
    verb=0
)
# result: array complexo com componentes do campo EM

# Comparação com o simulador Fortran:
# empymod calcula campo para um par T-R por chamada,
# enquanto o Fortran calcula para 600 posições T-R de uma vez.
# Para uso em loop de perfilagem, empymod seria ~10x mais lento
# sem vetorização adicional.
```

**Limitações do empymod para o projeto:**
- Não suporta diretamente o arranjo triaxial com rotação
- Não otimizado para loop de perfilagem (muitas posições T-R)
- Não integrado com TensorFlow para backpropagation

**Recomendação:** Usar empymod como referência e validação, mas implementar versão customizada otimizada para o caso de uso específico do pipeline.

### 12.5 Diferenciação Automática com JAX para PINNs

Uma das motivações mais fortes para reimplementar o simulador em Python é a possibilidade
de utilizar **diferenciação automática** (AD) para calcular gradientes do forward model
em relação aos parâmetros do meio. Isso é essencial para PINNs (Physics-Informed Neural
Networks), onde o gradiente do simulador aparece como termo de regularização na loss.

**JAX** é particularmente adequado para este caso:

```python
import jax
import jax.numpy as jnp

@jax.jit
def forward_em_1d(rho_h, rho_v, thicknesses, freq, z_obs):
    """Forward model EM 1D em JAX (diferenciável)."""
    sigma_h = 1.0 / rho_h
    sigma_v = 1.0 / rho_v
    omega = 2 * jnp.pi * freq
    zeta = 1j * omega * mu
    # ... cálculo do tensor H via Hankel (implementação JAX) ...
    return H_tensor  # (9,) complexo

# Gradiente do forward model em relação a rho_h e rho_v:
grad_fn = jax.grad(forward_em_1d, argnums=(0, 1))
d_H_d_rho_h, d_H_d_rho_v = grad_fn(rho_h, rho_v, ...)

# Uso em PINNs:
# loss_pinn = loss_data + lambda * ||F(rho_pred) - d_obs||^2
# onde F é o forward model diferenciável
```

Vantagens do JAX sobre outras opções:

1. **`jax.grad`** computa gradientes exatos (não numéricos) via AD reverso,
   com custo computacional ~2-3x o forward pass.
2. **`jax.jit`** compila o forward model via XLA, atingindo desempenho
   comparável a Numba JIT.
3. **`jax.vmap`** vetoriza automaticamente sobre batch de modelos,
   eliminando loops Python.
4. **Compatibilidade GPU/TPU** nativa, sem alteração de código.

A principal limitação do JAX é que a recursão dos coeficientes de reflexão
(loop sobre camadas com dependência de dados) requer `jax.lax.scan` em vez
de loops Python, o que exige reescrita cuidadosa. Além disso, operações com
números complexos em JAX requerem atenção à convenção de branch cuts em
funções como `sqrt`.

### 12.6 Estimativa de Desempenho Python

| Implementação | Tempo/modelo (est.) | vs Fortran | Notas |
|:-------------|:-------------------|:-----------|:------|
| Python puro | ~60 s | 150x mais lento | Inviável |
| NumPy vetorizado | ~4 s | 10x mais lento | Aceitável para protótipo |
| **Numba CPU (JIT)** | **~0.5 s** | **~1.2x mais lento** | **Viável para produção** |
| **Numba CUDA** | **~0.02 s** | **~20x mais rápido** | **Ideal para treinamento** |
| CuPy | ~0.1 s | ~4x mais rápido | Bom com batching |
| JAX (XLA) | ~0.05 s | ~8x mais rápido | Permite gradientes |

### 12.7 Roteiro de Implementação Python

```
Fase 1 (Protótipo — 2-3 semanas):
  ├── Implementar commonarraysMD em NumPy vetorizado
  ├── Implementar hmd_TIV e vmd em NumPy
  ├── Validar contra Fortran (erro relativo < 1e-10)
  └── Benchmark: tempo/modelo NumPy vs Fortran

Fase 2 (Otimização CPU — 2-3 semanas):
  ├── Reescrever kernels críticos com Numba @njit
  ├── Paralelismo sobre pontos do filtro (prange)
  ├── Cache de coeficientes de reflexão por modelo
  └── Benchmark: tempo/modelo Numba vs Fortran

Fase 3 (GPU — 3-4 semanas):
  ├── CUDA kernels para commonarraysMD + Hankel
  ├── Batching de múltiplos modelos na GPU
  ├── Integração com pipeline tf.data
  └── Benchmark: throughput (modelos/segundo)

Fase 4 (Integração — 2 semanas):
  ├── Módulo geosteering_ai/simulation/forward.py
  ├── Interface com PipelineConfig
  ├── Modo surrogado: rede neural substitui simulador
  └── Testes de integração com pipeline completo
```

### 12.8 Arquitetura Proposta para o Módulo Python

```
geosteering_ai/simulation/
├── __init__.py
├── forward.py        ← ForwardSimulator: interface principal
├── propagation.py    ← Constantes de propagação, reflexão (Numba)
├── dipoles.py        ← HMD e VMD para meios TIV (Numba)
├── hankel.py         ← Transformada de Hankel via filtros (Numba)
├── rotation.py       ← Rotação do tensor (Numba)
├── filters.py        ← Coeficientes tabelados (Werthmuller, etc.)
├── geometry.py       ← Geometria do poço e arranjo T-R
├── cuda_kernels.py   ← Kernels CUDA (Numba CUDA) [opcional]
└── validation.py     ← Comparação com Fortran
```

---

## 13. Integração com o Pipeline Geosteering AI v2.0

### 13.1 Cadeia Atual (Fortran)

```
fifthBuildTIVModels.py → model.in → tatu.x (Fortran) → .dat → geosteering_ai/data/loading.py
                              |          |                 |
                        Parâmetros    Simulação        Leitura binária
                        geológicos    EM 1D TIV        22 colunas
```

### 13.2 Cadeia Futura (Python Integrado)

```
geosteering_ai/simulation/forward.py → geosteering_ai/data/pipeline.py
                |                              |
          Simulação EM 1D TIV           Split → Noise → FV → GS → Scale
          (Numba/CUDA, on-the-fly)              |
                                          tf.data.Dataset
```

### 13.3 Benefícios da Integração

1. **Eliminação de I/O:** Dados gerados in-memory, sem escrita/leitura de .dat
2. **Geração on-the-fly:** Novos modelos geológicos gerados durante treinamento
3. **Data augmentation física:** Perturbação de parâmetros geológicos (espessuras, resistividades) como augmentation
4. **PINNs:** Backpropagation through o simulador para constraintes físicas
5. **Reprodutibilidade:** Seed único controla toda a cadeia (geração + simulação + treinamento)

### 13.4 Correspondência Fortran → Pipeline v2.0

| Parâmetro Fortran | Config v2.0 | Valor Default | Descrição |
|:------------------|:------------|:-------------|:----------|
| `freq(1)` | `config.frequency_hz` | 20000.0 | Frequência principal (Hz) |
| `dTR` | `config.spacing_meters` | 1.0 | Espaçamento T-R (m) |
| `nmed(1)` | `config.sequence_length` | 600 | Número de medições |
| `resist(:,1)` | `targets[:, 0]` (col 2) | - | rho_h (Ohm.m) |
| `resist(:,2)` | `targets[:, 1]` (col 3) | - | rho_v (Ohm.m) |
| `cH(f,1:9)` | `features[:, 4:21]` (cols 4-21) | - | Tensor H (18 valores reais) |
| `zobs` | `features[:, 1]` (col 1) | - | Profundidade (m) |

### 13.5 Mapeamento .dat → Estrutura de 22 Colunas do Pipeline

O pipeline Python (`geosteering_ai/data/loading.py`) interpreta os registros binários
de 172 bytes do .dat e os reorganiza em uma estrutura de **22 colunas** que é a
representação canônica dos dados ao longo de todo o pipeline:

```
  Registro binário .dat (172 bytes):
  ┌─────┬──────┬──────┬──────┬─────────────────────────────────────────────┐
  │ i   │ zobs │ rho_h│ rho_v│ Re Im Re Im ... Re Im Re Im Re Im Re Im Re Im │
  │int32│ f64  │ f64  │ f64  │ Hxx    Hxy    Hxz    Hyx    Hyy    Hyz    Hzx    Hzy    Hzz │
  └─────┴──────┴──────┴──────┴─────────────────────────────────────────────┘
    Col0  Col1   Col2   Col3   Col4,5  Col6,7  Col8,9  Col10,11  ...   Col20,21

  Mapeamento para o pipeline (COL_MAP_22):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Coluna  │  Conteúdo          │  Papel no Pipeline                      │
  ├──────────┼────────────────────┼─────────────────────────────────────────┤
  │  0       │  índice medição    │  Metadata (não usado como feature)      │
  │  1       │  zobs (m)          │  INPUT_FEATURE[0] — profundidade        │
  │  2       │  rho_h (Ohm.m)    │  OUTPUT_TARGET[0] — target 1            │
  │  3       │  rho_v (Ohm.m)    │  OUTPUT_TARGET[1] — target 2            │
  │  4       │  Re(Hxx) (A/m)    │  INPUT_FEATURE[1] — campo EM            │
  │  5       │  Im(Hxx) (A/m)    │  INPUT_FEATURE[2] — campo EM            │
  │  6-7     │  Re,Im(Hxy)       │  Disponível (não usado em P1)           │
  │  8-9     │  Re,Im(Hxz)       │  Disponível                             │
  │  10-11   │  Re,Im(Hyx)       │  Disponível                             │
  │  12-13   │  Re,Im(Hyy)       │  Disponível                             │
  │  14-15   │  Re,Im(Hyz)       │  Disponível                             │
  │  16-17   │  Re,Im(Hzx)       │  Disponível                             │
  │  18-19   │  Re,Im(Hzy)       │  Disponível                             │
  │  20      │  Re(Hzz) (A/m)    │  INPUT_FEATURE[3] — campo EM            │
  │  21      │  Im(Hzz) (A/m)    │  INPUT_FEATURE[4] — campo EM            │
  └──────────┴────────────────────┴─────────────────────────────────────────┘
```

**Configuração P1 (baseline):**

```python
# Em PipelineConfig (geosteering_ai/config.py):
INPUT_FEATURES = [1, 4, 5, 20, 21]
    # [zobs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz)]
    # → 5 features: profundidade + 2 componentes diagonais (planar + axial)

OUTPUT_TARGETS = [2, 3]
    # [rho_h, rho_v]
    # → 2 targets em escala log10 (TARGET_SCALING = "log10")
```

O modo P1 seleciona apenas Hxx (componente planar) e Hzz (componente axial) porque:
- Hxx é sensível à resistividade horizontal (via modo TE + TM)
- Hzz é sensível à resistividade vertical (via modo TE apenas, em poço vertical)
- Juntos, Hxx e Hzz fornecem informação complementar para resolver rho_h e rho_v

As 14 colunas restantes (cols 6-19) contêm as componentes off-diagonal e a segunda
componente diagonal (Hyy ≈ Hxx em poço vertical), que ficam disponíveis para modos
futuros (P2 tensor parcial, Modo C tensor completo).

---

## 14. Referências Bibliográficas

### 14.1 Modelagem EM em Meios Estratificados

1. **Anderson, W.L.** (1982). "Fast Hankel Transforms Using Related and Lagged Convolutions". *ACM Transactions on Mathematical Software*, 8(4), 344-368. — Filtro de 801 pontos para transformada de Hankel.

2. **Werthmuller, D.** (2006). "EMMOD — Electromagnetic Modelling". *Report, TU Delft*. — Filtros digitais de 201 pontos para J0/J1 utilizados no simulador.

3. **Kong, F.N.** (2007). "Hankel transform filters for dipole antenna radiation in a conductive medium". *Geophysical Prospecting*, 55(1), 83-89. — Filtro de 61 pontos.

4. **Key, K.** (2012). "Is the fast Hankel transform faster than quadrature?". *Geophysics*, 77(3), F21-F30. — Comparação de desempenho de filtros.

### 14.2 Anisotropia TIV em Perfilagem de Poços

5. **Anderson, B., Barber, T., & Habashy, T.** (2002). "The Interpretation of Multicomponent Induction Logs in the Presence of Dipping, Anisotropic Formations". *SPWLA 43rd Annual Logging Symposium*. — Resposta de ferramentas triaxiais em meios TIV.

6. **Liu, C.** (2017). *Theory of Electromagnetic Well Logging*. Elsevier. — Referência principal para a rotação do tensor (eq. 4.80).

### 14.3 Decomposição TE/TM e Coeficientes de Reflexão

7. **Chew, W.C.** (1995). *Waves and Fields in Inhomogeneous Media*. IEEE Press. — Formulação TE/TM para meios estratificados com anisotropia.

8. **Ward, S.H. & Hohmann, G.W.** (1988). "Electromagnetic Theory for Geophysical Applications". In *Electromagnetic Methods in Applied Geophysics*, Vol. 1, SEG. — Fundamentação teórica das equações de Maxwell em geofísica.

### 14.4 Dipolos Magnéticos em Meios TIV

9. **Zhong, L., Li, J., Bhardwaj, A., Shen, L.C., & Liu, R.C.** (2008). "Computation of Triaxial Induction Logging Tools in Layered Anisotropic Dipping Formations". *IEEE Transactions on Geoscience and Remote Sensing*, 46(4), 1148-1163.

10. **Davydycheva, S., Druskin, V., & Habashy, T.** (2003). "An efficient finite-difference scheme for electromagnetic logging in 3D anisotropic inhomogeneous media". *Geophysics*, 68(5), 1525-1536.

### 14.5 Software de Modelagem EM

11. **Werthmuller, D.** (2017). "An open-source full 3D electromagnetic modeller for 1D VTI media in Python: empymod". *Geophysics*, 82(6), WB9-WB19. — Implementação Python de referência com Numba.

### 14.6 Quasi-Monte Carlo e Geração de Modelos

12. **Sobol, I.M.** (1967). "On the distribution of points in a cube and the approximate evaluation of integrals". *USSR Computational Mathematics and Mathematical Physics*, 7(4), 86-112. — Sequências quasi-aleatórias para amostragem uniforme.

### 14.7 Inversão EM via Deep Learning

13. **Morales, A. et al.** (2025). "Physics-Informed Neural Networks for Triaxial Electromagnetic Inversion with Uncertainty Quantification". — PINN para inversão EM triaxial com constrainte TIV.

### 14.8 Computação GPU para Geofísica

14. **Weiss, C.J.** (2013). "Project APhiD: A Lorenz-gauged A-Phi decomposition for parallelized computation of ultra-broadband electromagnetic induction in a fully heterogeneous Earth". *Computers & Geosciences*, 58, 40-52. — GPU para modelagem EM.

15. **Commer, M. & Newman, G.A.** (2004). "A parallel finite-difference approach for 3D transient electromagnetic modeling with galvanic sources". *Geophysics*, 69(5), 1192-1202.

---

## 15. Apêndices

### 15.1 Apêndice A — Tabela de Sub-rotinas

| Sub-rotina | Módulo | Argumentos | Descrição |
|:-----------|:-------|:-----------|:----------|
| `perfila1DanisoOMP` | DManisoTIV | modelm, nmaxmodel, mypath, nf, freq, ntheta, theta, h1, tj, dTR, p_med, n, resist, esp, filename | Loop principal de perfilagem |
| `fieldsinfreqs` | DManisoTIV | ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, n, h, prof, resist, zrho, cH | Campos em todas as frequências para um ponto T-R |
| `writes_files` | DManisoTIV | modelm, nmaxmodel, mypath, zrho, cH, nt, theta, nf, freq, nmeds, filename | Escrita de .dat e .out |
| `write_results` | DManisoTIV | results, nk, nj, ni, arq, filename | Escrita de resultados auxiliares |
| `sanitize_hprof_well` | utils | n, esp, h, prof | Prepara geometria com sentinelas |
| `findlayersTR2well` | utils | n, h0, z, prof, camadT, camad | Identifica camadas T e R |
| `sanitizedata2well` | utils | n, h0, z, esp, camadT, camad, h, prof | Versão combinada (geometria + camadas) |
| `commonarraysMD` | utils | n, npt, hordist, krJ0J1, zeta, h, eta, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt | Constantes de propagação e reflexão |
| `commonfactorsMD` | utils | n, npt, h0, h, prof, camadT, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz | Fatores de onda da camada fonte |
| `layer2z_inwell` | utils | n, z, profs | Retorna índice da camada para profundidade z |
| `RtHR` | utils | alpha, beta, gamma, H | Rotação do tensor H: R^T*H*R |
| `hmd_TIV_optimized` | magneticdipoles | Tx, Ty, h0, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, prof, zeta, eta, cx, cy, z, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, Mxdw, Mxup, Eudw, Euup, Hx_p, Hy_p, Hz_p, dipolo | Campo do HMD em meio TIV |
| `vmd_optimized` | magneticdipoles | Tx, Ty, h0, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, prof, zeta, cx, cy, z, u, uh, AdmInt, RTEdw, RTEup, FEdwz, FEupz, Hx_p, Hy_p, Hz_p | Campo do VMD em meio TIV |
| `J0J1Kong` | filterscommonbase | npt, absc, wJ0, wJ1 | Filtro Kong (61 pts) |
| `J0J1Key` | filterscommonbase | npt, absc, wJ0, wJ1 | Filtro Key |
| `J0J1Wer` | filterscommonbase | npt, absc, wJ0, wJ1 | Filtro Werthmuller (201 pts) |
| `J0J1And` | filterscommonbase | npt, absc, wJ0, wJ1 | Filtro Anderson (801 pts) |

### 15.2 Apêndice B — Glossário

| Termo | Descrição |
|:------|:----------|
| **TIV** | Transversalmente Isotrópico Vertical — modelo de anisotropia com sigma_h != sigma_v |
| **LWD** | Logging While Drilling — perfilagem durante a perfuração |
| **HMD** | Horizontal Magnetic Dipole — fonte dipolar horizontal |
| **VMD** | Vertical Magnetic Dipole — fonte dipolar vertical |
| **TE** | Transverse Electric — modo com E tangencial às interfaces |
| **TM** | Transverse Magnetic — modo com H tangencial às interfaces |
| **Transformada de Hankel** | Integral de Bessel para passagem do domínio espectral para espacial |
| **DOI** | Depth of Investigation — profundidade de investigação da ferramenta |
| **BHA** | Bottom Hole Assembly — conjunto de ferramentas na extremidade do drill string |
| **Skin depth** | Profundidade de penetração da onda EM: delta = sqrt(2/(omega*mu*sigma)) |
| **Impeditividade** | zeta = i*omega*mu — constante que relaciona E e H |
| **Admitância intrínseca** | Y = u/zeta — admitância do modo TE por camada |
| **Impedância intrínseca** | Z = s/eta_h — impedância do modo TM por camada |
| **Decoupling** | Remoção do acoplamento direto T-R para isolar o campo secundário |
| **Forward modeling** | Cálculo da resposta do instrumento dado um modelo conhecido do meio |
| **QMC** | Quasi-Monte Carlo — amostragem com sequências de baixa discrepância (Sobol) |

### 15.3 Apêndice C — Mapeamento de Variáveis Fortran → Python

| Variável Fortran | Tipo | Variável Python (pipeline) | Descrição |
|:-----------------|:-----|:--------------------------|:----------|
| `resist(i,1)` | `real(dp)` | `y[:, :, 0]` (após log10) | rho_h |
| `resist(i,2)` | `real(dp)` | `y[:, :, 1]` (após log10) | rho_v |
| `cH(f,1)` | `complex(dp)` | `x[:, :, 1]` (Re) + `x[:, :, 2]` (Im) | Hxx |
| `cH(f,9)` | `complex(dp)` | `x[:, :, 19]` (Re) + `x[:, :, 20]` (Im) | Hzz |
| `zrho(f,1)` | `real(dp)` | `x[:, :, 0]` | z_obs (profundidade) |
| `nmed` | `integer` | `config.sequence_length` | 600 |
| `freq(1)` | `real(dp)` | `config.frequency_hz` | 20000.0 |
| `dTR` | `real(dp)` | `config.spacing_meters` | 1.0 |
| `n` | `integer` | `model['n_layers']` | Número de camadas |
| `esp(2:n-1)` | `real(dp)` | `model['thicknesses']` | Espessuras internas |

### 15.4 Apêndice D — Validação de Consistência Fortran-Python

Para garantir que o pipeline Python lê corretamente os dados do Fortran, os seguintes testes de consistência devem ser executados:

```python
# Teste 1: Número de registros
n_records_expected = sum(nmed) * nf * nmaxmodel
n_records_actual = os.path.getsize(filepath) // 172
assert n_records_actual == n_records_expected

# Teste 2: Ranges físicos
assert np.all(rho_h > 0)          # Resistividade positiva
assert np.all(rho_v >= rho_h)     # Constraint TIV
assert np.all(np.isfinite(H_real + H_imag))  # Sem NaN/Inf

# Teste 3: Simetria do tensor para theta=0 (poço vertical)
# Para poço vertical, Hxy = Hyx (simetria)
assert np.allclose(Re_Hxy, Re_Hyx, rtol=1e-10)
assert np.allclose(Im_Hxy, Im_Hyx, rtol=1e-10)
```

---

*Documentação do Simulador Fortran PerfilaAnisoOmp — Geosteering AI v2.0*
*Versão 2.0 — Abril 2026 — Pipeline v5.0.15+*
