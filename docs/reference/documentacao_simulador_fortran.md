# Documentação Completa do Simulador Fortran — PerfilaAnisoOmp

## Simulação Eletromagnética 1D para Meios Anisotrópicos TIV com Paralelismo OpenMP

**Projeto:** Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning
**Autor do Simulador:** Daniel Leal
**Linguagem:** Fortran 2008 (gfortran) com extensões OpenMP
**Localização:** `Fortran_Gerador/`
**Versão do Documento:** 4.0 (Abril 2026)

---

## Sumário

1. [Introdução e Motivação](#1-introdução-e-motivação)
2. [Fundamentos Físicos e Geofísicos](#2-fundamentos-físicos-e-geofísicos)
3. [Formulação Matemática Completa](#3-formulação-matemática-completa)
4. [Formulação Teórica via Potenciais de Hertz](#4-formulação-teórica-via-potenciais-de-hertz)
5. [Arquitetura do Software](#5-arquitetura-do-software)
6. [Módulos Fortran — Análise Detalhada](#6-módulos-fortran--análise-detalhada)
7. [Arquivo de Entrada model.in](#7-arquivo-de-entrada-modelin)
8. [Arquivos de Saída (.dat e .out)](#8-arquivos-de-saída-dat-e-out)
9. [Sistema de Build (Makefile)](#9-sistema-de-build-makefile)
10. [Gerador de Modelos Geológicos (Python)](#10-gerador-de-modelos-geológicos-python)
11. [Paralelismo OpenMP — Análise e Otimização](#11-paralelismo-openmp--análise-e-otimização)
12. [Análise de Viabilidade CUDA (GPU)](#12-análise-de-viabilidade-cuda-gpu)
    - 12.7 [Pipeline A — Roteiro de Otimização do Código Fortran](#127-pipeline-a--roteiro-de-otimização-do-código-fortran)
    - 12.8 [Pipeline B — Roteiro de Novos Recursos para o Simulador Fortran](#128-pipeline-b--roteiro-de-novos-recursos-para-o-simulador-fortran)
13. [Análise de Reimplementação em Python Otimizado](#13-análise-de-reimplementação-em-python-otimizado)
    - 13.9 [Pipeline A — Conversão Fortran→Python Otimizado](#139-pipeline-a--conversão-fortranpython-otimizado-numba-jit)
    - 13.10 [Pipeline B — Novos Recursos da Versão Python](#1310-pipeline-b--novos-recursos-da-versão-python)
    - 13.11 [Pipeline C — Vantagens sobre a Versão Fortran](#1311-pipeline-c--vantagens-da-versão-python-sobre-o-fortran)
    - 13.12 [Pipeline D — Avaliação Comparativa Fortran vs Python](#1312-pipeline-d--avaliação-comparativa-fortran-vs-python-otimizado)
14. [Integração com o Pipeline Geosteering AI v2.0](#14-integração-com-o-pipeline-geosteering-ai-v20)
15. [Referências Bibliográficas](#15-referências-bibliográficas)
16. [Sugestões de Melhorias e Novos Recursos](#16-sugestões-de-melhorias-e-novos-recursos)
17. [Apêndices](#17-apêndices)

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

## 4. Formulação Teórica via Potenciais de Hertz

Esta seção apresenta a derivação teórica completa da formulação de potenciais de Hertz
para dipolos magnéticos em meios estratificados com anisotropia TIV. A formulação segue
Moran & Gianzero (1979) e está documentada em detalhe no documento TeX do projeto
(`Tex_Projects/TatuAniso/FormulaçãoTatuAnisoTIV.tex`). As equações aqui apresentadas
são a fundamentação matemática direta do código Fortran em `magneticdipoles.f08` e
`utils.f08`.

### 4.1 Equações de Maxwell para Meios TIV no Domínio da Frequência

O simulador adota a convenção temporal `exp(+i*omega*t)` (engenharia), de modo que as
transformadas direta e inversa de Fourier são:

```
Transformada direta:
  f_hat(omega) = integral_{-inf}^{+inf} f(t) * exp(-i*omega*t) dt

Transformada inversa:
  f(t) = (1/2*pi) * integral_{-inf}^{+inf} f_hat(omega) * exp(+i*omega*t) d_omega

Dependência temporal resultante: exp(+i*omega*t)
```

As equações de Maxwell no domínio da frequência, para esta convenção, são:

```
(i)    div(epsilon * E) = rho_V                  (Lei de Gauss)
(ii)   rot(H) - y * E = J_e                      (Lei de Ampère)
(iii)  div(mu * H) = 0                           (Lei de Coulomb magnética)
(iv)   rot(E) + zeta * H = J_m                   (Lei de Faraday)

Onde:
  epsilon = permissividade dielétrica
  rho_V   = densidade volumétrica de carga elétrica
  y       = sigma + i*omega*epsilon_0  (admitividade, tensor para TIV)
  mu      = permeabilidade magnética (isotrópica, mu_0)
  zeta    = i*omega*mu_0              (impeditividade)
  J_e     = vetor densidade de corrente elétrica na fonte
  J_m     = vetor densidade de "corrente magnética" na fonte
  E       = campo vetorial elétrico
  H       = campo vetorial magnético
```

**Fonte de dipolo magnético:**

Para um dipolo magnético com momento `m`, as fontes são:

```
J_m = -zeta * m * delta(x) * delta(y) * delta(z)
J_e = 0

Onde m = (m_x, m_y, m_z) é o vetor momento do dipolo.
No código Fortran: mx = my = mz = 1.0 A.m^2.
```

**Tensor admitividade para meios TIV:**

Em meios com anisotropia TIV na condutividade, a admitividade é um tensor diagonal:

```
         ┌                              ┐
         │ sigma_h + i*omega*eps_0    0                      0                 │
  y  =   │ 0                      sigma_h + i*omega*eps_0    0                 │
         │ 0                      0                      sigma_v + i*omega*eps_0│
         └                              ┘

Onde sigma_h e sigma_v são as condutividades horizontal e vertical, respectivamente.
```

**Aproximação quasi-estática:**

Sob o regime quasi-estático (sigma >> omega*epsilon_0), o tensor simplifica para
`y ≈ diag(sigma_h, sigma_h, sigma_v)`. Esta é a aproximação adotada pelo simulador
(validada na Seção 3.1).

**Densidade volumétrica de carga em meios TIV:**

Uma consequência direta da anisotropia TIV é a existência de uma densidade volumétrica
de carga elétrica mesmo em regime estacionário. A partir da equação da continuidade
`div(J) = 0` e da lei de Ohm `J = y * E`, obtém-se:

```
div(y * E) = 0

Expandindo com o tensor TIV e usando a Lei de Gauss:

  sigma_h * (dE_x/dx + dE_y/dy) + sigma_v * dE_z/dz = 0

Definindo lambda^2 = sigma_h / sigma_v e substituindo div(E) = rho_V / epsilon_0:

  rho_V = (epsilon_0 / lambda^2) * (lambda^2 - 1) * dE_z/dz

Significado: Em meios anisotrópicos (lambda != 1), a descontinuidade de sigma
na direção z induz uma separação de cargas proporcional ao gradiente vertical
do campo elétrico. Essa carga desaparece no caso isotrópico (lambda = 1).
```

### 4.2 Potenciais de Hertz (pi_x, pi_u, pi_z)

O potencial de Hertz `pi` é um potencial vetorial auxiliar que permite desacoplar as
equações de Maxwell em equações escalares independentes para cada modo de propagação.

**Definição (Moran & Gianzero, 1979):**

```
y * E = -y_h * zeta * rot(pi)

Onde:
  pi = (pi_x, pi_y, pi_z) é o vetor potencial de Hertz
  y_h = sigma_h (admitividade horizontal escalar)
  zeta = i*omega*mu_0 (impeditividade)
```

A escolha de `y_h` no membro direito é deliberada: simplifica a forma final das
equações diferenciais para os potenciais.

**Condição de calibre (gauge):**

Seguindo Moran & Gianzero (1979), a condição de calibre adotada é:

```
div(y * pi) = sigma_v * Phi

Onde Phi é o potencial escalar associado.
```

**Campo H em termos dos potenciais:**

Substituindo a definição do potencial de Hertz na lei de Ampère e aplicando a
condição de calibre, obtém-se:

```
H = -zeta * y_h * pi + (1/sigma_v) * grad(div(y * pi))
```

As componentes do campo magnético são então:

```
H_x = kh^2 * pi_x + lambda^2 * (d^2 pi_x/dx^2 + d^2 pi_y/dxdy) + d^2 pi_z/dzdx
H_y = kh^2 * pi_y + lambda^2 * (d^2 pi_x/dxdy + d^2 pi_y/dy^2) + d^2 pi_z/dzdy
H_z = kh^2 * pi_z + lambda^2 * (d^2 pi_x/dzdx + d^2 pi_y/dzdy) + d^2 pi_z/dz^2

Onde kh^2 = -i*omega*mu_0*sigma_h = -zeta*sigma_h
```

**Simplificação para fonte sem componente y:**

Para um dipolo magnético horizontal na direção x (DMH_x), não há necessidade de
componente `pi_y`. O potencial de Hertz reduz-se a `pi = (pi_x, 0, pi_z)`.

### 4.3 Equações de Onda Desacopladas para os Potenciais

Substituindo as expressões de H e E em função de `pi` na lei de Faraday, e após
manipulação algébrica extensa (detalhada no TeX), obtém-se equações de onda
desacopladas para cada componente do potencial:

**Equação para pi_x (modo TM):**

```
d^2 pi_x/dx^2 + d^2 pi_x/dy^2 + (1/lambda^2) * d^2 pi_x/dz^2
  - i*omega*mu_0*sigma_v * pi_x = -(m_x / lambda^2) * delta(x)*delta(y)*delta(z)
```

**Equação para pi_y (modo TM, análoga):**

```
d^2 pi_y/dx^2 + d^2 pi_y/dy^2 + (1/lambda^2) * d^2 pi_y/dz^2
  - i*omega*mu_0*sigma_v * pi_y = -(m_y / lambda^2) * delta(x)*delta(y)*delta(z)
```

**Equação para pi_z (modo misto, acoplada com pi_x e pi_y):**

```
d^2 pi_z/dx^2 + d^2 pi_z/dy^2 + d^2 pi_z/dz^2
  - i*omega*mu_0*sigma_h * pi_z = (1-lambda^2) * d/dz(dpi_x/dx + dpi_y/dy)
                                    - m_z * lambda^2 * delta(x)*delta(y)*delta(z)
```

**Observações fundamentais:**

1. As equações para `pi_x` e `pi_y` são **desacopladas** entre si e são idênticas
   em forma. Basta resolver uma delas; a outra é obtida por analogia.

2. A equação para `pi_z` é **acoplada** com `pi_x` e `pi_y` (via o termo com
   derivadas cruzadas). No entanto, como primeiro resolvemos `pi_x` e `pi_y`,
   o membro direito de `pi_z` é conhecido, tornando-a uma equação forçada.

3. O operador em `pi_x` e `pi_y` é anisotrópico: o fator `1/lambda^2` multiplica
   a derivada em z. Isso reflete a propagação mais lenta na direção vertical em
   meios com `sigma_v < sigma_h`.

4. O operador em `pi_z` é **isotrópico** (todas as derivadas segundas com coeficiente 1),
   mas com `kh^2` em vez de `kv^2`. Isso é consequência da escolha de calibre.

### 4.4 Soluções Espectrais no Meio Ilimitado

A solução das equações de onda é obtida aplicando a transformada tripla de Fourier
para passar do domínio `(x, y, z)` para `(k_x, k_y, k_z)`.

**Solução para pi_x (transformada tripla):**

Definindo `v^2 = k_x^2 + k_y^2 - kv^2` e `kv^2 = -i*omega*mu_0*sigma_v`:

```
pi_x_hat_hat_hat(k_x, k_y, k_z) = m_x / ((lambda*v)^2 + k_z^2)
```

Aplicando a transformada inversa em `k_z`:

```
pi_x_hat_hat(k_x, k_y, z) = m_x * exp(-lambda*v*|z - h_0|) / (2*lambda*v)

Onde h_0 é a posição vertical da fonte (transmissor).

Definindo s = lambda*v (constante de propagação TM):
  pi_x_hat_hat = m_x * exp(-s*|z - h_0|) / (2*s)

Para z > h_0: pi_x_hat_hat = m_x * exp(-s*(z - h_0)) / (2*s)
Para z < h_0: pi_x_hat_hat = m_x * exp(+s*(z - h_0)) / (2*s)
```

**Solução para pi_z (via pi_x e pi_u):**

A solução de `pi_z` é obtida a partir de `pi_x` e de um potencial auxiliar `pi_u`
associado ao modo TE:

```
pi_z_hat_hat = -(i*k_x / k_r^2) * d(pi_x_hat_hat)/dz + (i*k_x / k_r^2) * pi_u_hat_hat

Onde k_r^2 = k_x^2 + k_y^2 (número de onda radial ao quadrado).
```

O potencial `pi_u` é a parte de `pi_z` associada ao modo TE. Suas soluções no
meio ilimitado são:

```
pi_u_hat_hat = (m_x/2) * exp(-u*|z - h_0|)

Onde u = sqrt(k_r^2 - kh^2) é a constante de propagação TE.
```

**Definições-chave:**

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Grandeza            │  Expressão              │  Modo       │
  ├──────────────────────┼─────────────────────────┼─────────────┤
  │  s = lambda * v      │  sqrt(lamb2) * v        │  TM         │
  │  u                   │  sqrt(kr^2 - kh^2)      │  TE         │
  │  v                   │  sqrt(kr^2 - kv^2)      │  Auxiliar   │
  │  kr^2                │  k_x^2 + k_y^2          │  Radial     │
  │  kh^2                │  -zeta * sigma_h         │  Horizontal │
  │  kv^2                │  -zeta * sigma_v         │  Vertical   │
  │  lambda^2            │  sigma_h / sigma_v       │  Anisotropia│
  └──────────────────────┴─────────────────────────┴─────────────┘
```

### 4.5 Potenciais no Meio de Camadas (6 Zonas)

No modelo de camadas horizontais com fonte (transmissor) na camada `l`, as soluções
espectrais dos potenciais `pi_x` e `pi_u` assumem formas distintas em 6 zonas
geométricas. Cada zona corresponde a uma combinação diferente de ondas transmitidas e
refletidas nas interfaces.

**Diagrama do modelo de camadas:**

```
  z = -inf     ┌────────────────────────────────────────┐
               │  Zona 0: Semi-espaço superior          │  pi_x^(0) = Tx^(0) * exp(s_0*z)
               │  (sigma_h0, sigma_v0)                  │  → Apenas onda transmitida ascendente
  z = z_0     ─├────────────────────────────────────────┤─
               │  ...                                    │
  z = z_{k-1} ─├────────────────────────────────────────┤─
               │  Zona k (k < l): Camadas acima fonte   │  pi_x^(k) = Tx^(k)[exp(s_k(z-z_k))
               │  (sigma_hk, sigma_vk)                  │    + R_up*exp(-s_k(z-z_{k-1}))]
  z = z_k     ─├────────────────────────────────────────┤─
               │  ...                                    │
  z = z_{l-1} ─├────────────────────────────────────────┤─
               │  Zona l-up: Camada fonte, z < h_0      │  pi_x^(l,up) = Tx_up^(l)[exp(s_l(z-h_0))
               │     ●──── Transmissor em h_0            │    + R_TM_up*Mx_up*exp(-s_l(z-z_{l-1}))
  z = h_0     ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │─   + R_TM_dw*Mx_dw*exp(s_l(z-z_l))]
               │  Zona l-dw: Camada fonte, z > h_0      │  pi_x^(l,dw) = Tx_dw^(l)[exp(-s_l(z-h_0))
               │  (sigma_hl, sigma_vl)                  │    + R_TM_up*Mx_up*exp(-s_l(z-z_{l-1}))
  z = z_l     ─├────────────────────────────────────────┤─   + R_TM_dw*Mx_dw*exp(s_l(z-z_l))]
               │  Zona j (j > l): Camadas abaixo fonte  │  pi_x^(j) = Tx^(j)[exp(-s_j(z-z_{j-1}))
               │  (sigma_hj, sigma_vj)                  │    + R_dw*exp(s_j(z-z_j))]
  z = z_{n-1} ─├────────────────────────────────────────┤─
               │  Zona n: Semi-espaço inferior           │  pi_x^(n) = Tx^(n) * exp(-s_n*(z-z_{n-1}))
               │  (sigma_hn, sigma_vn)                  │  → Apenas onda transmitida descendente
  z = +inf     └────────────────────────────────────────┘
```

**As 6 zonas e suas expressões:**

| Zona | Condição | Descrição | Forma de `pi_x` |
|:-----|:---------|:----------|:----------------|
| 0 | z < z_0 | Semi-espaço superior | `Tx^(0) * exp(s_0*z)` |
| k | z_{k-1} <= z < z_k, k < l | Camadas acima da fonte | Transmitida + refletida superiormente |
| l (z < h_0) | z_{l-1} <= z < h_0 | Camada fonte, receptor acima | Transmitida + 2 reflexões (Mx_up, Mx_dw) |
| l (z > h_0) | h_0 <= z < z_l | Camada fonte, receptor abaixo | Transmitida + 2 reflexões (Mx_up, Mx_dw) |
| j | z_{j-1} <= z < z_j, j > l | Camadas abaixo da fonte | Transmitida + refletida inferiormente |
| n | z >= z_{n-1} | Semi-espaço inferior | `Tx^(n) * exp(-s_n*(z-z_{n-1}))` |

**Correspondência com os 6 casos em `hmd_TIV_optimized`:**

Estas 6 zonas correspondem **diretamente** aos 6 blocos `if/elseif` na sub-rotina
`hmd_TIV_optimized` em `magneticdipoles.f08`:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Zona Teórica         │  Caso no Código        │  Condição Fortran │
  ├───────────────────────┼────────────────────────┼───────────────────┤
  │  Zona 0 (semi-sup.)   │  Caso 1                │  camadR == 1      │
  │  Zona k (acima T)     │  Caso 2                │  camadR < camadT  │
  │  Zona l (z < h_0)     │  Caso 3                │  camadR == camadT │
  │                        │                        │   .and. z <= h0   │
  │  Zona l (z > h_0)     │  Caso 4                │  camadR == camadT │
  │                        │                        │   .and. z > h0    │
  │  Zona j (abaixo T)    │  Caso 5                │  camadR > camadT  │
  │  Zona n (semi-inf.)   │  Caso 6                │  camadR == n      │
  └───────────────────────┴────────────────────────┴───────────────────┘
```

As expressões para `pi_u` são inteiramente análogas, substituindo `s` por `u`,
`R_TM` por `R_TE`, e os fatores `Mx` por `Eu`.

**Coeficientes de onda transmitida na camada fonte:**

Na camada `l` que contém a fonte, os coeficientes de transmissão são derivados
diretamente das soluções do meio ilimitado:

```
Tx_up^(l) = Tx_dw^(l) = m_x / (2*s_l)       (modo TM)
Tu_up^(l) = +m_x / 2                          (modo TE, direção ascendente)
Tu_dw^(l) = -m_x / 2                          (modo TE, direção descendente)
```

O sinal oposto em `Tu_up` e `Tu_dw` reflete a antissimetria do modo TE em relação
à fonte. No código Fortran, estes valores aparecem na inicialização dos arrays
`Txdw`, `Txup`, `Tudw`, `Tuup` na sub-rotina `hmd_TIV_optimized`.

### 4.6 Correspondência Potenciais <-> Modos TE/TM

O campo eletromagnético, expresso nas equações espectrais, pode ser escrito em
função apenas de `pi_x` e `pi_u`. A separação em modos de propagação é:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Potencial  │  Modo   │  Governa         │  Sensível a          │
  ├─────────────┼─────────┼──────────────────┼──────────────────────┤
  │  pi_x       │  TM     │  E_z, H_x, H_y  │  sigma_h E sigma_v  │
  │  pi_u       │  TE     │  H_z, E_x, E_y  │  Apenas sigma_h     │
  └─────────────┴─────────┴──────────────────┴──────────────────────┘

Justificativa:
  - E_z = zeta * lambda^2 * (i*k_y) * pi_x   → depende SOMENTE de pi_x
  - H_z = i*k_x * pi_u                       → depende SOMENTE de pi_u
```

**Impedância e admitância intrínsecas:**

Associadas a cada modo, definem-se as grandezas intrínsecas de cada camada `m`:

```
Impedância intrínseca (modo TM):
  Z_m = s_m / sigma_h,m     onde s_m = lambda_m * v_m

Admitância intrínseca (modo TE):
  Y_m = u_m / zeta          onde u_m = sqrt(kr^2 - kh^2_m)
```

**Coeficientes de reflexão:**

Os coeficientes de reflexão em cada interface são definidos pela diferença entre a
grandeza intrínseca da camada e a grandeza aparente vista através das camadas adjacentes:

```
Modo TM (reflexão descendente na interface inferior da camada m):
  R_TM_dw^(m) = (Z_m - Z_tilde_dw^(m+1)) / (Z_m + Z_tilde_dw^(m+1))

Modo TM (reflexão ascendente na interface superior da camada m):
  R_TM_up^(m) = (Z_m - Z_tilde_up^(m-1)) / (Z_m + Z_tilde_up^(m-1))

Modo TE (reflexão descendente):
  R_TE_dw^(m) = (Y_m - Y_tilde_dw^(m+1)) / (Y_m + Y_tilde_dw^(m+1))

Modo TE (reflexão ascendente):
  R_TE_up^(m) = (Y_m - Y_tilde_up^(m-1)) / (Y_m + Y_tilde_up^(m-1))
```

**Fórmula recursiva para impedância/admitância aparente (tanh):**

```
Impedância aparente descendente:
  Z_tilde_dw^(n) = Z_n                               (semi-espaço inferior)
  Z_tilde_dw^(m) = Z_m * [Z_tilde_dw^(m+1) + Z_m * tanh(s_m*h_m)]
                         / [Z_m + Z_tilde_dw^(m+1) * tanh(s_m*h_m)]

Impedância aparente ascendente:
  Z_tilde_up^(0) = Z_0                               (semi-espaço superior)
  Z_tilde_up^(m) = Z_m * [Z_tilde_up^(m-1) + Z_m * tanh(s_m*h_m)]
                         / [Z_m + Z_tilde_up^(m-1) * tanh(s_m*h_m)]

As admitâncias aparentes seguem fórmulas idênticas com Y_m, u_m em lugar de Z_m, s_m.
```

Para as camadas extremas (semi-espaços 0 e n), os coeficientes de reflexão são nulos:
`R_TM_up^(0) = R_TM_dw^(n) = R_TE_up^(0) = R_TE_dw^(n) = 0`.

### 4.7 Fatores de Onda na Camada do Transmissor

Na camada `l` que contém a fonte, as reflexões múltiplas entre as interfaces superior
(`z_{l-1}`) e inferior (`z_l`) são encapsuladas em fatores de onda `Mx` (TM) e `Eu` (TE).

**Fatores Mx (modo TM):**

```
Fator de onda TM descendente:
  Mx_dw = [exp(-s*(z_l - h_0)) + R_TM_up * exp(s*(z_{l-1} - h_l - h_0))]
          / [1 - R_TM_dw * R_TM_up * exp(-2*s*h_l)]

Fator de onda TM ascendente:
  Mx_up = [exp(s*(z_{l-1} - h_0)) + R_TM_dw * exp(-s*(z_l + h_l - h_0))]
          / [1 - R_TM_dw * R_TM_up * exp(-2*s*h_l)]
```

**Fatores Eu (modo TE):**

```
Fator de onda TE descendente:
  Eu_dw = [exp(-u*(z_l - h_0)) - R_TE_up * exp(u*(z_{l-1} - h_l - h_0))]
          / [1 - R_TE_dw * R_TE_up * exp(-2*u*h_l)]

Fator de onda TE ascendente:
  Eu_up = [exp(u*(z_{l-1} - h_0)) - R_TE_dw * exp(-u*(z_l + h_l - h_0))]
          / [1 - R_TE_dw * R_TE_up * exp(-2*u*h_l)]
```

**Nota sobre o sinal:** Os fatores `Eu` (modo TE) possuem sinal negativo nos termos
de reflexão, enquanto os fatores `Mx` (modo TM) possuem sinal positivo. Essa diferença
reflete as condições de contorno distintas: a componente `E_z` (TM) é contínua na
interface, enquanto `H_z` (TE) sofre uma descontinuidade proporcional à corrente
magnética superficial equivalente.

### 4.8 Coeficientes de Transmissão entre Camadas

Quando o receptor está em uma camada diferente da fonte, os coeficientes de transmissão
são calculados recursivamente a partir da camada da fonte, utilizando as condições de
continuidade dos campos tangenciais nas interfaces.

**Condições de continuidade (no domínio espectral):**

```
Em cada interface z = z_j, para meios não-magnéticos:

  (a) zeta * pi_z^(j)|_{z_j} = zeta * pi_z^(j+1)     → pi_z contínuo
  (b) d(pi_z)/dz|_{z_j} contínuo
  (c) zeta * d(pi_x)/dz|_{z_j} contínuo               → d(pi_x)/dz contínuo
  (d) sigma_h,j * pi_x^(j)|_{z_j} = sigma_h,j+1 * pi_x^(j+1)
  (e) pi_u^(j)|_{z_j} = pi_u^(j+1)                    → pi_u contínuo
  (f) d(pi_u)/dz|_{z_j} contínuo
```

**Transmissão descendente (camada j, com l+1 <= j-1 < j < n):**

```
Tx^(j) = (s_{j-1} / s_j) * Tx^(j-1) * exp(-s_{j-1} * h_{j-1})
          * (1 - R_TM_dw^(j-1)) / (1 - R_TM_dw^(j) * exp(-2*s_j*h_j))

Tu^(j) = Tu^(j-1) * exp(-u_{j-1} * h_{j-1})
          * (1 + R_TE_dw^(j-1)) / (1 + R_TE_dw^(j) * exp(-2*u_j*h_j))
```

**Transmissão ascendente (camada k, com 0 < k < k+1 <= l-1):**

```
Tx^(k) = (s_{k+1} / s_k) * Tx^(k+1) * exp(-s_{k+1} * h_{k+1})
          * (1 - R_TM_up^(k+1)) / (1 - R_TM_up^(k) * exp(-2*s_k*h_k))

Tu^(k) = Tu^(k+1) * exp(-u_{k+1} * h_{k+1})
          * (1 + R_TE_up^(k+1)) / (1 + R_TE_up^(k) * exp(-2*u_k*h_k))
```

**Casos especiais para semi-espaços:**

```
Semi-espaço superior (zona 0):
  Tx^(0) = (s_1/s_0) * Tx^(1) * exp(-s_1*h_1) * (1 - R_TM_up^(1))
  Tu^(0) = Tu^(1) * exp(-u_1*h_1) * (1 - R_TE_up^(1))

Semi-espaço inferior (zona n):
  Tx^(n) = (s_{n-1}/s_n) * Tx^(n-1) * exp(-s_{n-1}*h_{n-1}) * (1 - R_TM_dw^(n-1))
  Tu^(n) = Tu^(n-1) * exp(-u_{n-1}*h_{n-1}) * (1 + R_TE_dw^(n-1))
```

No código Fortran (`hmd_TIV_optimized`), esses coeficientes são computados nos arrays
`Txdw(:,j)`, `Txup(:,k)`, `Tudw(:,j)`, `Tuup(:,k)` via loops recursivos.

### 4.9 Campo no Domínio Espacial via Transformada de Hankel

As expressões finais do campo magnético H no domínio espacial são obtidas aplicando a
transformada inversa de Fourier dupla (em `k_x, k_y`) às expressões espectrais. Devido
à simetria cilíndrica do meio, essa transformada dupla reduz-se a integrais de Hankel
com funções de Bessel `J_0` e `J_1`.

**Campo espectral (resultado da Seção 4.2):**

```
H_x_hat_hat = -(k_y^2/kr^2) * kh^2 * pi_x + (k_x^2/kr^2) * d(pi_u)/dz
H_y_hat_hat = (k_x*k_y/kr^2) * kh^2 * pi_x + (k_x*k_y/kr^2) * d(pi_u)/dz
H_z_hat_hat = i*k_x * pi_u
```

**Transformada de Hankel — DMH_x (Dipolo Magnético Horizontal x):**

Para o DMH_x, as componentes do campo no domínio espacial envolvem convoluções com
J_0 e J_1:

```
H_x = (1/2*pi*r) * [(2*x^2/r^2 - 1) * sum(Kte_dz * wJ1)/r
       - kh^2 * (2*y^2/r^2 - 1) * sum(Ktm * wJ1)/r
       - (x^2/r^2) * sum(Kte_dz * wJ0 * kr)
       + kh^2 * (y^2/r^2) * sum(Ktm * wJ0 * kr)]

H_y = (x*y)/(pi*r^3) * [sum(Kte_dz * wJ1 + kh^2 * Ktm * wJ1)/r
       - sum((Kte_dz * wJ0 + kh^2 * Ktm * wJ0) * kr)/2]

H_z = -x * sum(Kte * wJ1 * kr^2) / (r * 2*pi*r)

Onde:
  Ktm = kernel modo TM (depende de pi_x, Tx, reflexões TM)
  Kte = kernel modo TE (depende de pi_u, Tu, reflexões TE)
  Kte_dz = derivada do kernel TE em z
  wJ0, wJ1 = pesos do filtro de Hankel para J_0 e J_1
  r = sqrt(x^2 + y^2) = distância horizontal T-R
  kr = abscissa do filtro / r
```

**DMH_y (Dipolo Magnético Horizontal y):**

O campo do dipolo y é obtido por rotação de 90 graus do DMH_x:

```
Regra de rotação: x → y, y → -x

H_x(DMH_y) = H_y(DMH_x)  (com a substituição de variáveis)
H_y(DMH_y) = expressão com (2*y^2/r^2 - 1) e x^2/r^2 permutados
H_z(DMH_y) = -y * sum(Kte * wJ1 * kr^2) / (r * 2*pi*r)
```

No código, o modo `'hmdxy'` calcula ambos os dipolos simultaneamente,
reutilizando os kernels espectrais.

**DMV (Dipolo Magnético Vertical):**

O DMV excita apenas o modo TE (simetria axial elimina o acoplamento TM):

```
H_x = -x * sum(Kte_dzz * J1 * kr^2) / (2*pi*r) / r
H_y = -y * sum(Kte_dzz * J1 * kr^2) / (2*pi*r) / r
H_z = sum(Kte_z * J0 * kr^3) / (2*pi*zeta) / r

Onde Kte_dzz = AdmInt * Kte_z  (kernel segunda derivada em z com J1)
```

### 4.10 Mapeamento Formulação Teórica <-> Código Fortran

A tabela abaixo estabelece a correspondência direta entre as variáveis da formulação
teórica (conforme o documento TeX) e as variáveis do código Fortran:

| Teoria (TeX) | Fortran | Sub-rotina | Tipo Fortran | Descrição |
|:-------------|:--------|:-----------|:-------------|:----------|
| `zeta = i*omega*mu_0` | `zeta` | `fieldsinfreqs` | `complex(dp)` | Impeditividade |
| `sigma_h, sigma_v` | `eta(i,1), eta(i,2)` | `fieldsinfreqs` | `real(dp)` | Condutividades (1/rho) |
| `kh^2 = -zeta*sigma_h` | `kh2(j)` | `commonarraysMD` | `complex(dp)` | Número de onda horiz. ao quadrado |
| `kv^2 = -zeta*sigma_v` | `kv2(j)` | `commonarraysMD` | `complex(dp)` | Número de onda vert. ao quadrado |
| `lambda^2 = sigma_h/sigma_v` | `lamb2(j)` | `commonarraysMD` | `real(dp)` | Coeficiente de anisotropia ao quadrado |
| `u_m = sqrt(kr^2 - kh^2)` | `u(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Constante de propagação TE |
| `v_m = sqrt(kr^2 - kv^2)` | `v(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Constante de propagação intermediária |
| `s_m = lambda_m * v_m` | `s(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Constante de propagação TM |
| `Y_m = u_m / zeta` | `AdmInt(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Admitância intrínseca (TE) |
| `Z_m = s_m / sigma_h,m` | `ImpInt(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Impedância intrínseca (TM) |
| `tanh(u_m * h_m)` | `tghuh(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Tanh estabilizada (TE) |
| `tanh(s_m * h_m)` | `tghsh(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Tanh estabilizada (TM) |
| `R_TE_up^(m)` | `RTEup(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Coeficiente reflexão TE ascendente |
| `R_TE_dw^(m)` | `RTEdw(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Coeficiente reflexão TE descendente |
| `R_TM_up^(m)` | `RTMup(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Coeficiente reflexão TM ascendente |
| `R_TM_dw^(m)` | `RTMdw(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Coeficiente reflexão TM descendente |
| `Y_tilde_dw^(m)` | `AdmAp_dw(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Admitância aparente descendente |
| `Y_tilde_up^(m)` | `AdmAp_up(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Admitância aparente ascendente |
| `Z_tilde_dw^(m)` | `ImpAp_dw(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Impedância aparente descendente |
| `Z_tilde_up^(m)` | `ImpAp_up(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Impedância aparente ascendente |
| `Mx_up, Mx_dw` | `Mxup(:), Mxdw(:)` | `commonfactorsMD` | `complex(dp), (npt)` | Fatores de onda TM na camada fonte |
| `Eu_up, Eu_dw` | `Euup(:), Eudw(:)` | `commonfactorsMD` | `complex(dp), (npt)` | Fatores de onda TE na camada fonte |
| `FE_dw_z, FE_up_z` | `FEdwz(:), FEupz(:)` | `commonfactorsMD` | `complex(dp), (npt)` | Fatores TE derivada z na camada fonte |
| `Tx^(j) (descendente)` | `Txdw(:,j)` | `hmd_TIV_optimized` | `complex(dp), (npt,n)` | Coef. transmissão TM descendente |
| `Tx^(k) (ascendente)` | `Txup(:,k)` | `hmd_TIV_optimized` | `complex(dp), (npt,n)` | Coef. transmissão TM ascendente |
| `Tu^(j) (descendente)` | `Tudw(:,j)` | `hmd_TIV_optimized` | `complex(dp), (npt,n)` | Coef. transmissão TE descendente |
| `Tu^(k) (ascendente)` | `Tuup(:,k)` | `hmd_TIV_optimized` | `complex(dp), (npt,n)` | Coef. transmissão TE ascendente |
| `K_tm` (kernel TM) | `Ktm(:)` | `hmd_TIV_optimized` | `complex(dp), (npt)` | Kernel espectral modo TM |
| `K_te` (kernel TE) | `Kte(:)` | `hmd_TIV_optimized` | `complex(dp), (npt)` | Kernel espectral modo TE |
| `K_te_dz` (kernel TE deriv.) | `Ktedz(:)` | `hmd_TIV_optimized` | `complex(dp), (npt)` | Kernel TE derivada em z |
| `R^T * H * R` | `RtHR` | `utils.f08` | Sub-rotina | Rotação do tensor para frame da ferramenta |
| `h_0` (posição da fonte) | `h0` | `fieldsinfreqs` | `real(dp)` | Profundidade do transmissor |
| `kr` (número de onda radial) | `kr(:)` | `commonarraysMD` | `real(dp), (npt)` | `absc(i) / hordist` |
| `h_m` (espessura camada m) | `h(m)` | `sanitize_hprof_well` | `real(dp)` | Espessura da camada m |
| `z_m` (interface inferior m) | `prof(m)` | `sanitize_hprof_well` | `real(dp)` | Profundidade da interface m |

---

## 5. Arquitetura do Software

### 5.1 Estrutura de Módulos

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

### 5.2 Grafo de Dependências

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

### 5.3 Fluxo de Execução Completo

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

### 5.4 Inventário de Linhas de Código

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

## 6. Módulos Fortran — Análise Detalhada

### 6.1 parameters.f08 — Constantes Físicas

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

### 6.2 filtersv2.f08 (module filterscommonbase) — Filtros Digitais para Transformada de Hankel

**Nota:** O arquivo chama-se `filtersv2.f08`, mas o módulo Fortran declarado dentro dele é `module filterscommonbase`. O módulo principal (`PerfilaAnisoOmp.f08`) o referencia via `use filterscommonbase`.

Módulo que armazena os coeficientes tabelados de 4 filtros digitais para transformadas de Hankel. Cada filtro fornece:
- `absc(npt)`: Abscissas (pontos de amostragem no domínio espectral)
- `wJ0(npt)`: Pesos para convolução com função de Bessel J0
- `wJ1(npt)`: Pesos para convolução com função de Bessel J1

**Filtro utilizado no simulador:** `J0J1Wer` com `npt = 201` (Werthmuller).

A transformada de Hankel via filtro digital é uma técnica clássica em geofísica EM que substitui a integração numérica direta (quadratura) por uma soma ponderada. Os coeficientes são pré-calculados offline e tabelados, tornando a avaliação extremamente eficiente (~201 multiplicações por ponto).

### 6.3 utils.f08 — Funções Utilitárias

#### 6.3.1 sanitize_hprof_well

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

#### 6.3.2 findlayersTR2well

Identifica em qual camada estão o transmissor (camadT) e o receptor (camadR), dado o array de profundidades das interfaces. A busca é feita de baixo para cima (`do i = n-1, 2, -1`) para eficiência em cenários onde T e R estão em camadas profundas.

#### 6.3.3 commonarraysMD — Constantes de Propagação e Reflexão

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

#### 6.3.4 commonfactorsMD — Fatores de Onda da Camada Fonte

Calcula os 6 fatores de onda refletida (Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz) para a camada do transmissor. Estes fatores encapsulam as reflexões múltiplas dentro da camada fonte e são reutilizados para todos os dipolos.

**Otimização:** Esta sub-rotina só precisa ser recalculada quando a distância horizontal `r` entre T e R muda, ou quando a camada do transmissor muda. Em configurações coaxiais (r constante), o custo é amortizado.

#### 6.3.5 RtHR — Rotação do Tensor Magnético

Implementa a rotação `R^T * H * R` conforme Liu (2017), equação 4.80. Recebe os três ângulos de Euler (alpha, beta, gamma) e o tensor H(3x3) complexo, retornando o tensor rotacionado no sistema de coordenadas da ferramenta.

Na perfilagem inclinada, `alpha = theta` (inclinação do poço), `beta = gamma = 0`.

### 6.4 magneticdipoles.f08 — Campos Dipolares

#### 6.4.1 hmd_TIV_optimized — Dipolo Magnético Horizontal

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

#### 6.4.2 vmd_optimized — Dipolo Magnético Vertical

O VMD excita apenas o modo TE (simetria axial). A estrutura é similar ao HMD, com 6 casos geométricos. Os coeficientes de transmissão `TEdwz`/`TEupz` são calculados recursivamente.

### 6.5 PerfilaAnisoOmp.f08 — Módulo Principal

#### 6.5.1 perfila1DanisoOMP — Loop de Perfilagem

Função principal do simulador. Etapas:

1. **Cálculo de nmed:** Número de medições por ângulo baseado na janela `tj` e passo `p_med`
2. **Carregamento do filtro:** `J0J1Wer(201, ...)` — filtro de Werthmuller
3. **Preparação da geometria:** `sanitize_hprof_well(n, esp, h, prof)`
4. **Configuração OpenMP:** Paralelismo aninhado com threads dinâmicos
5. **Loop duplo paralelo:**
   - Loop externo: ângulos (`k = 1..ntheta`), `num_threads_k = ntheta`
   - Loop interno: medições (`j = 1..nmed(k)`), `num_threads_j = maxthreads - ntheta`
6. **Coleta e escrita:** `writes_files(...)` escreve .dat e .out

#### 6.5.2 fieldsinfreqs — Campos em Todas as Frequências

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

#### 6.5.3 writes_files — Escrita de Saída

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

### 6.6 RunAnisoOmp.f08 — Programa Principal

Programa sequencial que:
1. Obtém o diretório corrente via `getcwd`
2. Lê o arquivo `model.in` sequencialmente
3. Aloca arrays de resistividade e espessura
4. Chama `perfila1DanisoOMP(...)` com todos os parâmetros

---

## 7. Arquivo de Entrada model.in

### 7.1 Estrutura Completa

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

### 7.2 Exemplo Comentado (model.in atual)

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

### 7.3 Observações Importantes

1. **Espessuras:** As camadas 1 e ncam são semi-espaços (espessura infinita), então `esp(1) = esp(ncam) = 0` é atribuído em `RunAnisoOmp.f08` (programa principal) antes de chamar `perfila1DanisoOMP`.

2. **Frequências:** O simulador padrão usa 2 frequências (20 kHz e 40 kHz). A primeira é a mesma do pipeline (`FREQUENCY_HZ = 20000.0`).

3. **Número de medições:** Para `theta = 0`, `pz = p_med = 0.2`, `nmed = ceil(120/0.2) = 600`, consistente com `SEQUENCE_LENGTH = 600` do pipeline.

4. **Anisotropia neste exemplo:** O modelo mostra anisotropia muito fraca (`lambda ~ 1.0` em quase todas as camadas). Em cenários reais, `lambda` varia de 1.0 a sqrt(2).

---

## 8. Arquivos de Saída (.dat e .out)

### 8.1 Arquivo .out — Metadados

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

### 8.2 Arquivo .dat — Dados Binários

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

### 8.3 Leitura no Pipeline Python (geosteering_ai)

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

### 8.4 Tamanho dos Arquivos

Para 1000 modelos, 1 ângulo, 2 frequências, 600 medições:

```
Tamanho = nmodels × ntheta × nfreq × nmeds × 172 bytes
        = 1000 × 1 × 2 × 600 × 172
        = 206,400,000 bytes ≈ 197 MB
```

---

## 9. Sistema de Build (Makefile)

### 9.1 Estrutura do Makefile

O Makefile segue a convenção padrão de projetos Fortran com:

| Seção | Descrição |
|:------|:----------|
| SETTINGS | Binário (`tatu.x`), extensões, diretório de build (`./build`) |
| DEPENDENCIES | Ordem de compilação: `parameters → filtersv2 → utils → magneticdipoles → PerfilaAnisoOmp → RunAnisoOmp` |
| COMPILER | `gfortran` com flags de produção |
| TARGETS | `$(binary)`, `run_python`, `all`, `clean`, `cleanall` |

### 9.2 Flags de Compilação

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

### 9.3 Targets

| Target | Descrição |
|:-------|:----------|
| `$(binary)` (`tatu.x`) | Compila e linka todos os .o em um executável |
| `run_python` | Executa `fifthBuildTIVModels.py` (gerador de modelos) |
| `all` | Compila + executa Python |
| `clean` | Remove `./build/` |
| `cleanall` | Remove `./build/`, `tatu.x`, `*.dat`, `*.out` |

### 9.4 Ordem de Compilação

```
parameters.f08 ──> build/parameters.o
filtersv2.f08  ──> build/filtersv2.o
utils.f08      ──> build/utils.o      (depende de parameters)
magneticdipoles.f08 ──> build/magneticdipoles.o (depende de parameters)
PerfilaAnisoOmp.f08 ──> build/PerfilaAnisoOmp.o (depende de todos acima + omp_lib)
RunAnisoOmp.f08 ──> build/RunAnisoOmp.o (depende de parameters, DManisoTIV)

Linkagem: gfortran -fopenmp [flags] -o tatu.x build/*.o
```

### 9.5 Notas e Recomendações

- **`-ffast-math`** pode afetar a precisão de operações com NaN e infinitos. Para validação numérica, recomenda-se compilar sem esta flag.
- **`-march=native`** otimiza para a CPU local, mas o binário não é portável.
- O diretório `./build` é criado automaticamente via `$(shell mkdir -p $(build))`.

---

## 10. Gerador de Modelos Geológicos (Python)

### 10.1 Visão Geral

O script `fifthBuildTIVModels.py` gera modelos geológicos aleatórios usando amostragem Sobol Quasi-Monte Carlo (`scipy.stats.qmc.Sobol`). Os modelos são escritos em `model.in` e simulados pelo Fortran em sequência.

### 10.2 Parâmetros dos Modelos Geológicos

| Parâmetro | Range | Distribuição | Descrição |
|:----------|:------|:------------|:----------|
| `n_layers` | 3-80 | Empírica (ponderada) ou uniforme | Número de camadas |
| `rho_h` | 0.05-1500 Ohm.m | Log-uniforme (Sobol) | Resistividade horizontal |
| `lambda` | 1.0-sqrt(2) | Uniforme (Sobol), correlacionada com rho_h | Coeficiente de anisotropia |
| `rho_v` | calculado | `lambda^2 * rho_h` | Resistividade vertical |
| `espessuras` | 0.1-50+ m | Sobol + stick-breaking | Espessuras das camadas internas |

### 10.3 Cenários de Geração (6 Geradores)

| Cenário | Função | nmodels | ncam | Espessuras | Contrastes | Ruído |
|:--------|:-------|:-------:|:----:|:-----------|:-----------|:------|
| **Baseline empírico** | `baseline_empirical_2` | 18000 | 3-30 (pesos empíricos) | Standard (min 1.0 m) | Natural | Não |
| **Baseline uniforme** | `baseline_ncamuniform_2` | 9000 | 3-80 (uniforme) | Standard (min 0.5 m) | Natural | Não |
| **Camadas grossas** | `baseline_thick_thicknesses_2` | 9000 | 3-14 | Grossas (min 10 m, p=0.7) | Forçados em grossas | Não |
| **Desfavorável empírico** | `unfriendly_empirical_2` | 12000 | 3-30 (pesos) | Finas (min 0.2 m, p=0.6) | Forçados (5x, p=0.5) | Não |
| **Desfavorável ruidoso** | `unfriendly_noisy_2` | 12000 | 3-30 (pesos) | Finas + ruído (3%) | Forçados + ruído (5%) | Sim |
| **Patológico** | `generate_pathological_models_2` | 4500 | 3-28 | Muito finas (min 0.1 m) | Extremos (10x, p=0.7) | Sim (7%) |

### 10.4 Funções Auxiliares e Vantagens do Sobol QMC

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

### 10.5 Fluxo do Gerador

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

### 10.6 Loop de Execução via Subprocess

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

## 11. Paralelismo OpenMP — Análise e Otimização

### 11.1 Estrutura Atual de Paralelismo

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

### 11.2 Análise de Desempenho

**Problemas identificados:**

1. **Desbalanceamento de carga no nível externo:** Com `ntheta = 1` (comum), o loop externo usa apenas 1 thread, desperdiçando o segundo nível aninhado.

2. **Alocação de memória dentro do loop paralelo:** Cada chamada a `commonarraysMD` e `hmd_TIV_optimized` aloca arrays temporários. Isso causa:
   - Contenção no alocador de memória (malloc)
   - Fragmentação de heap
   - Overhead de alocação/desalocação repetida

3. **Escalonamento `dynamic` para loop regular:** Se todas as medições têm custo computacional similar (mesmo modelo), `static` seria mais eficiente (sem overhead de despacho).

4. **Redundância computacional:** `commonarraysMD` calcula constantes de propagação e coeficientes de reflexão que dependem apenas do modelo (n, resist, freq), não da posição T-R. Em medições onde T e R estão na mesma camada (maioria dos casos em poços verticais), `commonarraysMD` produz o mesmo resultado para todos os `j`.

### 11.3 Pontos de Otimização — Memória

| Otimização | Impacto | Complexidade |
|:-----------|:--------|:-------------|
| **Pré-alocar arrays de trabalho por thread** | Alto — elimina malloc dentro do loop | Média |
| **Mover `commonarraysMD` para fora do loop j** | Alto — computa uma vez por (ângulo, freq) quando r é constante | Média |
| **Usar `firstprivate` para arrays somente-leitura** | Médio — evita cópias desnecessárias | Baixa |
| **Reduzir arrays temporários em hmd_TIV** | Médio — Tudw/Txdw/etc. alocados com tamanho mínimo | Média |
| **Pool de memória por thread** | Alto — elimina fragmentação | Alta |

### 11.4 Pontos de Otimização — Tempo de Execução

| Otimização | Impacto Estimado | Descrição |
|:-----------|:----------------|:----------|
| **Cache de `commonarraysMD`** | 30-50% redução | Computa uma vez por (camadT, freq), reutiliza para todos os j com mesmo camadT |
| **Paralelismo sobre frequências** | 2x para nf=2 | Adicionar nível de paralelismo sobre frequências |
| **Colapso de loops** | 10-20% | `!$omp parallel do collapse(2)` para ângulos × medições |
| **SIMD para convolução Hankel** | 10-30% | `!$omp simd` nos sums de kernels × pesos do filtro |
| **Scheduler híbrido** | 5-10% | `static` quando nmed uniforme, `dynamic` quando variável |
| **Batch de medições** | 15-25% | Agrupar medições por camadT para reutilizar coeficientes de reflexão |

### 11.5 Proposta de Paralelismo Otimizado

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

### 11.6 Métricas de Escalabilidade

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

### 11.7 Paralelização de Loops de Frequência e Ângulos

A estrutura atual de paralelismo (2 níveis: ângulos x medições) pode ser estendida
para 3 níveis quando o simulador opera com múltiplas frequências e ângulos
simultaneamente.

**Estrutura atual (2 níveis):**

```
Loop externo: ntheta ângulos        → num_threads_k = ntheta
  Loop interno: nmed medições       → num_threads_j = maxthreads - ntheta

Para ntheta=1, nf=2: 2 × 600 = 1200 chamadas a fieldsinfreqs,
  mas apenas 600 são paralelizáveis (o loop de frequências é sequencial
  dentro de fieldsinfreqs).
```

**Proposta: 3 níveis (ângulos x frequências x medições) com `collapse`:**

Quando `ntheta > 1` e `nf > 1`, é possível colapsar os três loops em uma única
região paralela, maximizando a granularidade de trabalho:

```fortran
! PROPOSTA: Paralelismo 3-level com collapse
! Lineariza os 3 loops em um único espaço de iteração
!$omp parallel do schedule(dynamic, 32) collapse(3) &
!$omp& private(k, f, j, tid, ws)
do k = 1, ntheta
  do f = 1, nf
    do j = 1, nmmax
      if (j > nmed(k)) cycle

      ! Cada combinação (ângulo, freq, medição) é uma tarefa independente
      call compute_single_measurement(k, f, j, ws(tid), ...)
    end do
  end do
end do
!$omp end parallel do
```

**Análise de carga para configurações típicas:**

| Configuração | ntheta | nf | nmed | Tarefas Totais | Eficiência (8 cores) |
|:-------------|:------:|:--:|:----:|:--------------:|:--------------------:|
| Baseline (atual) | 1 | 2 | 600 | 1200 | ~95% (150 tarefas/core) |
| Multi-ângulo | 2 | 2 | 600/622 | 2444 | ~98% (305 tarefas/core) |
| Multi-freq extenso | 1 | 4 | 600 | 2400 | ~97% (300 tarefas/core) |

**Considerações de afinidade de threads e NUMA:**

Em sistemas com múltiplos sockets (NUMA), a afinidade de threads é relevante:

```
Configuração recomendada para dual-socket:
  export OMP_PLACES=cores
  export OMP_PROC_BIND=close

Justificativa:
  - 'close' mantém threads próximas no mesmo socket, otimizando acesso
    à memória local (latência ~100 ns vs ~300 ns inter-socket).
  - 'cores' distribui uma thread por core físico, evitando contenção
    de hyperthreading em operações FP intensivas.

Para single-socket (laptop/workstation):
  export OMP_PROC_BIND=spread
  → Distribui threads uniformemente entre cores para balanceamento térmico.
```

### 11.8 Estratégias Avançadas de Otimização de Memória

**11.8.1 Alocação de Workspace Thread-Local (pré-alocado)**

O principal gargalo de memória no simulador é a alocação dinâmica repetida de arrays
dentro do loop paralelo. A solução é pré-alocar um workspace por thread antes do loop:

```fortran
type :: thread_workspace
    ! Arrays de propagação (npt × n_max)
    complex(dp), allocatable :: u(:,:), s(:,:), v(:,:)
    complex(dp), allocatable :: uh(:,:), sh(:,:)
    complex(dp), allocatable :: tghuh(:,:), tghsh(:,:)

    ! Coeficientes de reflexão (npt × n_max)
    complex(dp), allocatable :: RTEdw(:,:), RTEup(:,:)
    complex(dp), allocatable :: RTMdw(:,:), RTMup(:,:)
    complex(dp), allocatable :: AdmInt(:,:), ImpInt(:,:)
    complex(dp), allocatable :: AdmAp_dw(:,:), AdmAp_up(:,:)
    complex(dp), allocatable :: ImpAp_dw(:,:), ImpAp_up(:,:)

    ! Fatores de onda (npt)
    complex(dp), allocatable :: Mxdw(:), Mxup(:)
    complex(dp), allocatable :: Eudw(:), Euup(:), FEdwz(:), FEupz(:)

    ! Coeficientes de transmissão (npt × n_max)
    complex(dp), allocatable :: Txdw(:,:), Txup(:,:)
    complex(dp), allocatable :: Tudw(:,:), Tuup(:,:)

    ! Kernels e resultados (npt)
    complex(dp), allocatable :: Ktm(:), Kte(:), Ktedz(:)
end type

! Memória estimada por thread (n_max=80, npt=201):
!   ~30 arrays × 201 × 80 × 16 bytes ≈ 7.7 MB
! Para 8 threads: ~62 MB (cabe confortavelmente na L3 cache de CPUs modernas)
```

**11.8.2 Stack vs Heap para Arrays Pequenos**

Para arrays de tamanho fixo e pequeno (como o tensor H 3x3), a alocação em stack
é preferível por evitar overhead de malloc:

```fortran
! CORRETO: stack allocation (tamanho fixo, conhecido em compilação)
complex(dp) :: matH(3,3)
complex(dp) :: Hx_p(1,2), Hy_p(1,2), Hz_p(1,2)

! EVITAR: heap allocation para arrays pequenos
complex(dp), allocatable :: matH(:,:)   ! Overhead desnecessário
allocate(matH(3,3))
```

**11.8.3 Memory Pooling para Arrays Complexos**

Para arrays de tamanho variável (como `Txdw(:,camadT:camadR)`), um pool de memória
evita fragmentação:

```fortran
! Pool com blocos pré-alocados de tamanho máximo:
type :: memory_pool
    complex(dp), allocatable :: block(:,:)  ! (npt, n_max) pré-alocado
    integer :: in_use                        ! Marca de uso
end type

! Cada thread obtém um bloco do pool, sem malloc:
call pool_acquire(my_pool, Txdw_ptr, npt, n_needed)
! ... usa Txdw_ptr ...
call pool_release(my_pool, Txdw_ptr)
```

**11.8.4 Alinhamento de Cache-Line para Vetorização SIMD**

Para maximizar a eficiência da vetorização SIMD (AVX-256/512) nos loops sobre
os pontos do filtro (npt=201), os arrays devem ser alinhados a 64 bytes:

```fortran
! GCC: atributo de alinhamento
!GCC$ attributes aligned(64) :: u, s, uh, sh

! Diretiva OpenMP SIMD para o loop interno:
!$omp simd aligned(Ktm, Kte, wJ0, wJ1: 64)
do i = 1, npt
    sum_J0 = sum_J0 + Ktm(i) * wJ0(i)
    sum_J1 = sum_J1 + Kte(i) * wJ1(i)
end do

! O npt=201 não é múltiplo de 8 (AVX-512 complex), mas 201 = 25×8 + 1.
! O compilador gerará um loop principal de 25 iterações vetorizadas
! mais uma iteração escalar residual.
```

---

## 12. Análise de Viabilidade CUDA (GPU)

### 12.1 Identificação de Kernels Paralelizáveis

| Kernel | Dimensão | Paralelismo | Adequação GPU |
|:-------|:---------|:------------|:-------------|
| `commonarraysMD` | npt × n | Independente por ponto do filtro | **Alta** — 201 threads independentes |
| `commonfactorsMD` | npt | Independente por ponto do filtro | **Alta** — 201 threads |
| Convolução Hankel (sum) | npt | Redução paralela | **Alta** — redução clássica |
| Loop medições | nmed | Independente por medição | **Alta** — 600+ threads |
| Loop frequências | nf | Independente por frequência | **Média** — apenas 2 frequências |
| Recursão reflexão | n (sequencial) | **Dependência de dados** | **Baixa** — n camadas sequenciais |

### 12.2 Estratégia de Implementação CUDA

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

### 12.3 Desafios CUDA

1. **Recursão dos coeficientes de reflexão:** Os coeficientes são calculados de forma recursiva (camada n para 1, e 1 para n). Cada passo depende do anterior, impedindo paralelismo nesta dimensão. Porém, os 201 pontos do filtro são independentes, permitindo 201 threads executando a recursão simultaneamente.

2. **Alocação dinâmica:** O código Fortran aloca arrays de tamanho variável (`Tudw`, `Txdw`) dependendo da posição relativa T-R. Na GPU, estes arrays devem ser pré-alocados com tamanho máximo.

3. **Divergência de branch:** Os 6 casos geométricos em `hmd_TIV_optimized` causam divergência de warp em GPUs NVIDIA. Estratégia: agrupar medições por caso geométrico para minimizar divergência.

4. **Memória:** Para n=80 camadas, npt=201: arrays de tamanho (201, 80) × complex64 = 201 × 80 × 16 bytes = ~257 KB por medição. Caberiam ~800 medições simultâneas em 200 MB de memória global.

### 12.4 Análise Detalhada de Memória GPU

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

### 12.5 Estimativa de Speedup

| Componente | CPU (1 core) | GPU (estimado) | Speedup |
|:-----------|:-------------|:---------------|:--------|
| `commonarraysMD` | ~0.5 ms/call | ~0.01 ms/call | 50x |
| Convolução Hankel | ~0.2 ms/sum | ~0.002 ms/sum | 100x |
| Loop medições (600) | 600 × serial | 600 × paralelo | ~100x |
| Overhead transferência | 0 | ~1 ms/modelo | - |
| **Total por modelo** | ~400 ms | ~10 ms | **~40x** |

### 12.6 Frameworks CUDA Recomendados

| Framework | Linguagem | Vantagem | Desvantagem |
|:----------|:----------|:---------|:------------|
| **CUDA Fortran (PGI/NVHPC)** | Fortran | Reuso máximo do código | Compilador proprietário |
| **CUDA C/C++** | C/C++ | Ecossistema maduro, cuBLAS | Reescrita completa |
| **OpenACC** | Fortran | Mínimo de alteração no código | Desempenho menor que CUDA nativo |
| **hipSYCL** | C++ | Portabilidade AMD/NVIDIA | Complexidade adicional |

**Recomendação:** Para prototipagem rápida, **OpenACC** com `nvfortran` (NVIDIA HPC SDK). Para produção, **CUDA C** com wrappers Fortran.


### 12.7 Pipeline A — Roteiro de Otimização do Código Fortran

#### 12.7.1 Visão Geral do Pipeline A

O Pipeline A concentra-se em extrair o máximo desempenho do código Fortran existente sem alterar a física implementada. A estratégia é dividida em três fases progressivas: otimizações CPU com OpenMP (retorno imediato, risco mínimo), implementação GPU via OpenACC ou CUDA (aceleração de 10–50×, risco moderado) e validação sistemática com benchmarking quantitativo.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              PIPELINE A — ROTEIRO DE OTIMIZAÇÃO FORTRAN                 │
│                                                                         │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │   FASE 1 — CPU   │   │   FASE 2 — GPU   │   │  FASE 3 — VALID  │    │
│  │  OpenMP / SIMD   │──▶│  OpenACC / CUDA  │──▶│  Benchmark / QA  │    │
│  │  (semanas 1–3)   │   │  (semanas 4–10)  │   │  (semanas 11–12) │    │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘    │
│         │                       │                       │               │
│  Speedup alvo: 3–5×      Speedup alvo: 20–50×   Erro rel. < 1e-6       │
│  Baseline: ~2.4 s/mod    Alvo: ~0.05–0.12 s/mod  por componente Hxx    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Métricas de sucesso globais do Pipeline A:**

| Métrica | Baseline atual | Meta Fase 1 | Meta Fase 2 |
|:--------|:--------------:|:-----------:|:-----------:|
| Tempo por modelo (1 ângulo, 2 freq, 600 med.) | ~2,4 s | < 0,7 s | < 0,12 s |
| Throughput (modelos/hora, 16 threads) | ~24.000 | ~80.000 | ~500.000 |
| Erro relativo vs. código original | — | < 1×10⁻¹² | < 1×10⁻⁶ |
| Uso de memória por thread | ~150 MB | < 80 MB | ~500 MB (GPU) |
| Escalabilidade (efficiency Amdahl) | — | > 70% | > 80% |

---

#### 12.7.2 Fase 1 — Otimizações CPU com OpenMP (Prioridade Inicial)

Esta fase ataca os gargalos identificados na análise OpenMP (Seção 11) sem introduzir dependências externas ou risco de divergência numérica. Todas as modificações são feitas no módulo `DManisoTIV` (`PerfilaAnisoOmp.f08`) e no módulo `magneticdipoles` (`magneticdipoles.f08`).

---

##### Passo 1.1 — Pré-alocação de Workspace por Thread (Eliminar `allocate` no Laço Paralelo)

**Problema identificado:** A sub-rotina `fieldsinfreqs` é chamada dentro do laço paralelo `!$omp parallel do` sobre medições (`j = 1, nmed(k)`). Internamente, `hmd_TIV_optimized` e `vmd_optimized` alocam dinamicamente arrays `Tudw`, `Txdw`, `Tuup`, `Txup`, `TEdwz`, `TEupz` em cada chamada. Com 600 medições × 2 frequências, isso resulta em até **2.400 chamadas a `allocate/deallocate`** por modelo no laço interno, gerando contenção de heap e fragmentação de memória.

**O que modificar:** Sub-rotinas `hmd_TIV_optimized` e `vmd_optimized` em `magneticdipoles.f08`; laço externo em `perfila1DanisoOMP` em `PerfilaAnisoOmp.f08`.

**Speedup esperado:** 1,4–1,8× (redução de 30–40% no tempo de execução medido por `gprof`).

**Risco/Complexidade:** Baixo/Médio — requer refatoração das assinaturas das sub-rotinas para aceitar buffers pré-alocados.

**Implementação — mudança em `perfila1DanisoOMP`:**

```fortran
! ANTES (situação atual): allocate/deallocate ocorrem dentro de hmd/vmd
! a cada chamada no laço paralelo j = 1, nmed(k)

! DEPOIS: pré-alocar workspace por thread antes do laço paralelo
! Tamanho máximo: (npt, n) onde npt=201, n=número máximo de camadas
integer :: tid, nthreads_actual
integer, parameter :: MAX_N = 50  ! dimensão máxima esperada de camadas
complex(dp), allocatable :: ws_Tudw(:,:,:), ws_Txdw(:,:,:)
complex(dp), allocatable :: ws_Tuup(:,:,:), ws_Txup(:,:,:)
complex(dp), allocatable :: ws_TEdwz(:,:,:), ws_TEupz(:,:,:)

nthreads_actual = omp_get_max_threads()

! Pré-alocar: dimensão extra é o índice da thread (0:nthreads-1)
allocate(ws_Tudw (npt, MAX_N, 0:nthreads_actual-1))
allocate(ws_Txdw (npt, MAX_N, 0:nthreads_actual-1))
allocate(ws_Tuup (npt, MAX_N, 0:nthreads_actual-1))
allocate(ws_Txup (npt, MAX_N, 0:nthreads_actual-1))
allocate(ws_TEdwz(npt, MAX_N, 0:nthreads_actual-1))
allocate(ws_TEupz(npt, MAX_N, 0:nthreads_actual-1))

!$omp parallel do schedule(dynamic) num_threads(num_threads_j) &
!$omp   private(j, x, y, z, Tx, Ty, Tz, posTR, zrho, cH, tid)
do j = 1, nmed(k)
  tid = omp_get_thread_num()
  ! Passa workspace pré-alocado para evitar malloc interno:
  call fieldsinfreqs_ws(ang, nf, freq, posTR, dipolo, npt, krwJ0J1, n, h, prof, resist, &
                        zrho, cH, &
                        ws_Tudw(:,:,tid), ws_Txdw(:,:,tid), &
                        ws_Tuup(:,:,tid), ws_Txup(:,:,tid), &
                        ws_TEdwz(:,:,tid), ws_TEupz(:,:,tid))
  z_rho1(j,:,:) = zrho
  c_H1(j,:,:) = cH
end do
!$omp end parallel do

deallocate(ws_Tudw, ws_Txdw, ws_Tuup, ws_Txup, ws_TEdwz, ws_TEupz)
```

---

##### Passo 1.2 — Cache de Resultados de `commonarraysMD` por `(camadT, freq)`

**Problema identificado:** A sub-rotina `commonarraysMD` calcula os coeficientes de reflexão TE/TM (`RTEdw`, `RTEup`, `RTMdw`, `RTMup`) e as constantes de propagação (`u`, `s`, `uh`, `sh`) que dependem apenas de `(r, freq, eta, zeta)` — onde `r` é a distância transmissor-receptor. Na perfilagem em poço, `r = dTR` (distância T-R fixa = 1,0 m), portanto **`r` e `freq` são invariantes por frequência e modelo geológico**. Para um dado modelo, com 2 frequências e 600 medições, `commonarraysMD` é chamada 1.200 vezes com os mesmos argumentos `(r, freq_i)`.

**O que modificar:** Sub-rotina `fieldsinfreqs` em `PerfilaAnisoOmp.f08`.

**Speedup esperado:** 1,6–2,2× (a sub-rotina `commonarraysMD` representa ~45% do tempo total, segundo análise de profiling com `gprof -l`).

**Risco/Complexidade:** Médio — requer reestruturação do laço sobre frequências, pré-computação fora do laço de medições.

**Implementação — reestruturação de `fieldsinfreqs`:**

```fortran
! ANTES: commonarraysMD chamada dentro do laço do j (medições)
! para cada (j, freq_i) — repetição desnecessária quando r é fixo

! DEPOIS: pré-computar commonarraysMD para cada frequência UMA VEZ
! e reutilizar para todas as nmed(k) medições do mesmo ângulo k

subroutine precompute_common_arrays(nf, freqs, r, npt, krJ0J1, n, h, eta, &
                                    u_all, s_all, uh_all, sh_all, &
                                    RTEdw_all, RTEup_all, RTMdw_all, RTMup_all, &
                                    AdmInt_all)
  implicit none
  integer, intent(in) :: nf, npt, n
  real(dp), intent(in) :: freqs(nf), r, krJ0J1(npt), h(n), eta(n,2)
  ! Arrays pré-computados: dimensão extra nf para frequências
  complex(dp), intent(out) :: u_all(npt,n,nf), s_all(npt,n,nf)
  complex(dp), intent(out) :: uh_all(npt,n,nf), sh_all(npt,n,nf)
  complex(dp), intent(out) :: RTEdw_all(npt,n,nf), RTEup_all(npt,n,nf)
  complex(dp), intent(out) :: RTMdw_all(npt,n,nf), RTMup_all(npt,n,nf)
  complex(dp), intent(out) :: AdmInt_all(npt,n,nf)

  integer :: i
  real(dp) :: freq, omega
  complex(dp) :: zeta

  do i = 1, nf
    freq = freqs(i)
    omega = 2.d0 * pi * freq
    zeta = cmplx(0.d0, 1.d0, kind=dp) * omega * mu
    call commonarraysMD(n, npt, r, krJ0J1, zeta, h, eta, &
                        u_all(:,:,i), s_all(:,:,i), uh_all(:,:,i), sh_all(:,:,i), &
                        RTEdw_all(:,:,i), RTEup_all(:,:,i), &
                        RTMdw_all(:,:,i), RTMup_all(:,:,i), AdmInt_all(:,:,i))
  end do
end subroutine precompute_common_arrays
```

Dentro do laço de medições `j`, as chamadas a `commonarraysMD` são substituídas por referências diretas a `u_all(:,:,i)`, `RTEdw_all(:,:,i)`, etc. A pré-computação é realizada uma única vez por ângulo `k`, antes do laço `!$omp parallel do` sobre `j`.

---

##### Passo 1.3 — Collapse de Laços com `collapse(3)` para Ângulos × Frequências × Medições

**Problema identificado:** O laço paralelo externo (`k = 1, ntheta`) tipicamente tem `ntheta = 1` em simulações de ângulo único (caso 0°), o que impede qualquer benefício do paralelismo de nível 1. O laço interno (`j = 1, nmed(k)`) tem `nmed = 600`, que é paralelizado com `num_threads_j = maxthreads - ntheta` threads. A combinação aninhada gera overhead de fork/join para apenas uma iteração do laço externo.

**O que modificar:** Estrutura de laços em `perfila1DanisoOMP`; requer linearização do índice de medições.

**Speedup esperado:** 1,2–1,5× para `ntheta = 1`; sem ganho para `ntheta > 4` (já paralelizado de forma eficiente).

**Risco/Complexidade:** Médio — exige cálculo de `k` e `j` a partir do índice linearizado.

**Implementação:**

```fortran
! ANTES: laços aninhados com OpenMP nested
!$omp parallel do schedule(dynamic) num_threads(num_threads_k) ...
do k = 1, ntheta
  !$omp parallel do schedule(dynamic) num_threads(num_threads_j) ...
  do j = 1, nmed(k)
    ...
  end do
  !$omp end parallel do
end do
!$omp end parallel do

! DEPOIS: laço único colapsado (válido quando nmed é uniforme ou
! quando se usa nmmax para dimensionar o espaço de iterações)
integer :: kj, k_idx, j_idx, total_iter

total_iter = ntheta * nmmax  ! limite superior conservador

!$omp parallel do schedule(dynamic,8) num_threads(maxthreads) &
!$omp   private(kj, k_idx, j_idx, ang, seno, coss, px, pz, &
!$omp           Lsen, Lcos, x, y, z, Tx, Ty, Tz, posTR, zrho, cH)
do kj = 1, total_iter
  k_idx = (kj - 1) / nmmax + 1
  j_idx = mod(kj - 1, nmmax) + 1
  if (j_idx > nmed(k_idx)) cycle  ! pula iterações fora do alcance
  ! ... corpo do laço original ...
end do
!$omp end parallel do
```

---

##### Passo 1.4 — Vetorização SIMD da Convolução de Hankel

**Problema identificado:** O núcleo computacional de `hmd_TIV_optimized` e `vmd_optimized` são somas sobre 201 pontos do filtro de Hankel de Werthmuller:

```fortran
kernelHxJ1 = (twox2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twoy2_r2m1 * sum(Ktm_J1)) / r
kernelHxJ0 = x2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * y2_r2 * sum(Ktm_J0 * kr)
```

Com `npt = 201` pontos e aritmética complexa de precisão dupla, cada `sum()` é uma redução sobre vetores complexos de 201 elementos. O compilador `gfortran` com `-O3 -march=native` pode vetorizar automaticamente, mas a presença de `sum()` intrínseco sobre arrays complexos frequentemente impede a geração de instruções AVX-512 ou AVX2.

**O que modificar:** Sub-rotinas `hmd_TIV_optimized` e `vmd_optimized` em `magneticdipoles.f08`; adicionar diretivas `!DIR$ VECTOR` ou usar laço explícito com `!$omp simd`.

**Speedup esperado:** 1,3–2,0× nas reduções de Hankel (que representam ~25% do tempo total).

**Risco/Complexidade:** Baixo — diretivas de compilação não alteram a semântica do código.

**Implementação:**

```fortran
! Substituir sum() implícito por laço explícito com diretiva SIMD
! para garantir vetorização com AVX2/AVX-512 em arquiteturas modernas

! ANTES:
kernelHxJ1 = (twox2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twoy2_r2m1 * sum(Ktm_J1)) / r

! DEPOIS: redução explícita com SIMD
complex(dp) :: acc_KtedzJ1, acc_KtmJ1
integer :: ip
acc_KtedzJ1 = (0.d0, 0.d0)
acc_KtmJ1   = (0.d0, 0.d0)

!$omp simd reduction(+:acc_KtedzJ1, acc_KtmJ1)
do ip = 1, npt
  acc_KtedzJ1 = acc_KtedzJ1 + Ktedz_J1(ip)
  acc_KtmJ1   = acc_KtmJ1   + Ktm_J1(ip)
end do
!$omp end simd

kernelHxJ1 = (twox2_r2m1 * acc_KtedzJ1 - kh2(camadR) * twoy2_r2m1 * acc_KtmJ1) / r
```

Flags de compilação adicionais recomendadas:

```bash
gfortran -O3 -march=native -fopenmp -ffast-math \
         -fopt-info-vec-optimized \    # relatorio de vetorizacao
         -funroll-loops \              # desenrolar laços de 201 pts
         -fprefetch-loop-arrays \      # prefetch de arrays grandes
         -o PerfilaAnisoOmp *.f08
```

---

##### Passo 1.5 — Escalonador Híbrido (Static para Uniforme, Dynamic para Variável)

**Problema identificado:** O código atual usa `schedule(dynamic)` para ambos os laços — externo (ângulos `k`) e interno (medições `j`). Para `ntheta = 1` e `nmed` uniforme, o escalonador `dynamic` introduz overhead de sincronização desnecessário. Para casos multi-ângulo com `nmed(k)` variável (o número de medições pode diferir entre ângulos), `static` causaria desbalanceamento.

**O que modificar:** Diretivas `!$omp parallel do schedule(...)` em `perfila1DanisoOMP`.

**Speedup esperado:** 1,05–1,15× (ganho marginal, mas cumulativo com outros passos).

**Risco/Complexidade:** Muito baixo — apenas mudança de palavra-chave.

**Implementação:**

```fortran
! Estratégia híbrida: escolher escalonador em tempo de execução

! Para laço externo (ângulos): static quando ntheta é pequeno
if (ntheta <= 4) then
  !$omp parallel do schedule(static) num_threads(num_threads_k) &
  !$omp   private(k, ang, seno, coss, px, pz, Lsen, Lcos, z_rho1, c_H1)
  do k = 1, ntheta
    ...
  end do
  !$omp end parallel do
else
  !$omp parallel do schedule(guided,4) num_threads(num_threads_k) &
  !$omp   private(k, ang, seno, coss, px, pz, Lsen, Lcos, z_rho1, c_H1)
  do k = 1, ntheta
    ...
  end do
  !$omp end parallel do
end if

! Para laço interno (medições): static com chunk = nmmax/num_threads_j
! quando nmed(k) é uniforme, dynamic com chunk pequeno quando variável
chunk_size = max(1, nmed(k) / num_threads_j)
!$omp parallel do schedule(static, chunk_size) num_threads(num_threads_j) &
!$omp   private(j, x, y, z, Tx, Ty, Tz, posTR, zrho, cH)
do j = 1, nmed(k)
  ...
end do
!$omp end parallel do
```

---

##### Passo 1.6 — Agrupamento de Medições por Camada do Transmissor (`camadT`)

**Problema identificado:** A sub-rotina `commonfactorsMD` calcula os fatores `Mxdw`, `Mxup`, `Eudw`, `Euup`, `FEdwz`, `FEupz` que dependem de `(camadT, freq)` — onde `camadT` é a camada geológica que contém o transmissor. Em perfilagem com passo `p_med = 1,0 m`, medições consecutivas frequentemente têm o transmissor na **mesma camada** (especialmente em camadas espessas > 1 m). Calcular `commonfactorsMD` repetidamente para `camadT` invariante é trabalho redundante.

**O que modificar:** Sub-rotina `fieldsinfreqs` / novo laço de agrupamento em `perfila1DanisoOMP`.

**Speedup esperado:** 1,2–1,5× em modelos com camadas espessas (> 5 m); menor impacto em modelos de camadas finas.

**Risco/Complexidade:** Médio-Alto — requer pré-determinação de `camadT(j)` para todas as medições antes do laço paralelo.

**Implementação:**

```fortran
! Pré-calcular camadT para cada medição j (fora do laço paralelo)
integer, allocatable :: camadT_arr(:), camadR_arr(:)
allocate(camadT_arr(nmmax), camadR_arr(nmmax))

do j = 1, nmed(k)
  ! Reconstruir posTR apenas para determinar camadas
  x_j  = 0.d0 + (j-1) * px - Lsen / 2.d0
  z_j  = z1  + (j-1) * pz - Lcos / 2.d0
  Tx_j = 0.d0 + (j-1) * px + Lsen / 2.d0
  Tz_j = z1  + (j-1) * pz + Lcos / 2.d0
  call findlayersTR2well(n, Tz_j, z_j, prof(1:n-1), camadT_arr(j), camadR_arr(j))
end do

! Ordenar medições por camadT para maximizar reutilização de commonfactorsMD
! (usar índice de permutação para não alterar a ordem de escrita de resultados)
! ... implementação com índice de reordenação iperm(j) ...

! No laço paralelo, usar cache de commonfactorsMD por (camadT, i_freq):
integer :: camadT_prev
complex(dp) :: Mxdw_cache(npt), Mxup_cache(npt)
complex(dp) :: Eudw_cache(npt), Euup_cache(npt)
complex(dp) :: FEdwz_cache(npt), FEupz_cache(npt)

camadT_prev = -1  ! sentinela: forçar recálculo na primeira iteração

do j = 1, nmed(k)
  ct = camadT_arr(j)
  if (ct /= camadT_prev) then
    ! Recalcular commonfactorsMD apenas quando camadT muda
    call commonfactorsMD(n, npt, Tz_arr(j), h, prof, ct, &
                         u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, &
                         Mxdw_cache, Mxup_cache, Eudw_cache, Euup_cache, &
                         FEdwz_cache, FEupz_cache)
    camadT_prev = ct
  end if
  ! Usar cache diretamente em hmd_TIV_optimized e vmd_optimized
  ...
end do
```

> **Nota:** Esta otimização é particularmente eficaz para o caso de geosteering em formações de resistividade com camadas de espessura > 2 m, onde mais de 50% das medições consecutivas compartilham o mesmo `camadT`.

---

**Tabela Resumo — Fase 1 CPU:**

```
┌─────────────────────────────────────┬───────────────────┬───────────┬──────────────┐
│  Otimização                         │  Sub-rotina alvo  │  Speedup  │  Complexid.  │
├─────────────────────────────────────┼───────────────────┼───────────┼──────────────┤
│  1.1 Workspace pré-alocado          │  hmd/vmd_optimzd  │  1,4–1,8× │  Médio       │
│  1.2 Cache commonarraysMD           │  fieldsinfreqs    │  1,6–2,2× │  Médio       │
│  1.3 Collapse laços collapse(3)     │  perfila1Daniso   │  1,2–1,5× │  Médio       │
│  1.4 SIMD convolução Hankel         │  hmd/vmd_optimzd  │  1,3–2,0× │  Baixo       │
│  1.5 Escalonador híbrido            │  perfila1Daniso   │  1,05–1,2×│  Muito baixo │
│  1.6 Cache commonfactorsMD          │  fieldsinfreqs    │  1,2–1,5× │  Médio-alto  │
├─────────────────────────────────────┼───────────────────┼───────────┼──────────────┤
│  TOTAL COMBINADO (estimativa)       │  Pipeline inteiro │  3,5–5,0× │  —           │
└─────────────────────────────────────┴───────────────────┴───────────┴──────────────┘
```

---

#### 12.7.3 Fase 2 — Implementação GPU (CUDA/OpenACC)

A Fase 2 aproveita o paralelismo massivo de GPUs modernas (NVIDIA A100: 6.912 CUDA cores; RTX 4090: 16.384 CUDA cores) para acelerar o simulador em 20–50× sobre o código CPU original. A estratégia preferida é **OpenACC como protótipo rápido** (mínimas mudanças no código), seguida de **CUDA Fortran via NVHPC** para maximizar ocupância e coalescência de memória.

---

##### Passo 2.1 — Protótipo OpenACC (Mudanças Mínimas no Código)

OpenACC permite acelerar o código existente com diretivas de compilador sem reescrever o algoritmo. A curva de aprendizado é baixa e a portabilidade é preservada (o mesmo código compila para CPU sem OpenACC).

**Requisito:** Compilador NVHPC (NVIDIA HPC SDK) ou GCC 10+ com suporte a OpenACC.

```bash
# Instalação NVHPC (Ubuntu/Colab):
apt-get install -y nvhpc-23-11

# Compilação com OpenACC:
nvfortran -O3 -acc=gpu -gpu=cc80,managed -Minfo=accel \
          parameters.f08 utils.f08 filtersv2.f08 \
          magneticdipoles.f08 PerfilaAnisoOmp.f08 RunAnisoOmp.f08 \
          -o PerfilaAnisoOmp_gpu
```

**Diretivas OpenACC para `fieldsinfreqs`:**

```fortran
subroutine fieldsinfreqs(ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, &
                         n, h, prof, resist, zrho, cH)
  implicit none
  ! ... declarações existentes ...

  ! Transferir dados imutáveis para o dispositivo (uma vez por modelo)
  !$acc data copyin(krwJ0J1, h, prof, resist, eta) &
  !$acc      copyout(zrho, cH)

  do i = 1, nf
    freq = freqs(i)
    omega = 2.d0 * pi * freq
    zeta = cmplx(0.d0, 1.d0, kind=dp) * omega * mu

    !$acc parallel loop gang vector &
    !$acc   private(u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt)
    call commonarraysMD(...)  ! lançado como kernel na GPU

    !$acc parallel loop gang vector
    call commonfactorsMD(...)

    !$acc parallel loop gang vector
    call hmd_TIV_optimized(...)

    !$acc parallel loop gang vector
    call vmd_optimized(...)
  end do

  !$acc end data
end subroutine fieldsinfreqs
```

**Limitação do protótipo OpenACC:** A estrutura de chamadas entre sub-rotinas (`fieldsinfreqs → commonarraysMD → commonfactorsMD → hmd/vmd`) exige que todas as sub-rotinas chamadas dentro de regiões `!$acc parallel` sejam marcadas com `!$acc routine`. Isso pode requerer modificações nas assinaturas.

---

##### Passo 2.2 — Kernels CUDA Fortran via NVHPC

Para maximizar o desempenho, o laço mais interno (201 pontos × `n` camadas de `commonarraysMD`) é convertido em kernels CUDA explícitos.

**Mapeamento CPU → GPU:**

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MAPEAMENTO PARALELISMO CPU → GPU                                        │
│                                                                          │
│  CPU (OpenMP)              │  GPU (CUDA)                                │
│  ──────────────────────────┼────────────────────────────────────────    │
│  k = 1, ntheta (ângulos)   │  Grid: dim3(ntheta, n_modelos, 1)          │
│  j = 1, nmed(k) (medições) │  Block: dim3(32, 8, 1) — 256 threads      │
│  i = 1, nf (frequências)   │  Frequências: loop dentro do kernel        │
│  ip = 1, npt (Hankel 201)  │  Warp: 32 threads para redução Hankel      │
│                            │  Shared mem: u, s, krwJ0J1 (201 pts)       │
└──────────────────────────────────────────────────────────────────────────┘
```

**Kernel CUDA Fortran para redução de Hankel (esboço):**

```fortran
! Arquivo: hankel_kernel.cuf (extensão CUDA Fortran)
module hankel_kernels
  use cudafor
  use parameters
  implicit none
contains

  attributes(global) subroutine hankel_reduction_kernel( &
      npt, n_meas, Ktedz_J1_all, Ktm_J1_all, &
      Ktedz_J0_all, Ktm_J0_all, kr_all, &
      acc_KtedzJ1, acc_KtmJ1, acc_KtedzJ0kr, acc_KtmJ0kr, &
      n_meas_total)
    implicit none
    integer, value :: npt, n_meas_total
    complex(dp), device :: Ktedz_J1_all(npt, n_meas_total)
    complex(dp), device :: Ktm_J1_all  (npt, n_meas_total)
    complex(dp), device :: Ktedz_J0_all(npt, n_meas_total)
    complex(dp), device :: Ktm_J0_all  (npt, n_meas_total)
    real(dp),    device :: kr_all(npt, n_meas_total)
    complex(dp), device :: acc_KtedzJ1(n_meas_total)
    complex(dp), device :: acc_KtmJ1  (n_meas_total)
    complex(dp), device :: acc_KtedzJ0kr(n_meas_total)
    complex(dp), device :: acc_KtmJ0kr  (n_meas_total)

    integer :: meas_idx, ip
    complex(dp) :: local_KtedzJ1, local_KtmJ1
    complex(dp) :: local_KtedzJ0kr, local_KtmJ0kr

    ! Shared memory para 201 pontos (redução em warp)
    complex(dp), shared :: smem_KtedzJ1(256), smem_KtmJ1(256)

    meas_idx = blockIdx%x + (blockIdx%y - 1) * gridDim%x
    if (meas_idx > n_meas_total) return

    ip = threadIdx%x  ! cada thread processa um ponto do filtro

    ! Carregar em shared memory (coalescência garante acesso contíguo)
    if (ip <= npt) then
      smem_KtedzJ1(ip) = Ktedz_J1_all(ip, meas_idx)
      smem_KtmJ1(ip)   = Ktm_J1_all  (ip, meas_idx)
    else
      smem_KtedzJ1(ip) = (0.d0, 0.d0)
      smem_KtmJ1(ip)   = (0.d0, 0.d0)
    end if
    call syncthreads()

    ! Redução em árvore binária dentro do bloco
    ! ... loop de redução com stride decrescente ...

    if (threadIdx%x == 1) then
      acc_KtedzJ1(meas_idx) = smem_KtedzJ1(1)
      acc_KtmJ1  (meas_idx) = smem_KtmJ1(1)
    end if
  end subroutine hankel_reduction_kernel

end module hankel_kernels
```

---

##### Passo 2.3 — Gerenciamento de Memória (Host ↔ Device)

A latência de transferência PCI-e é o principal gargalo em implementações GPU ingênuas. A estratégia é minimizar transferências via **batching de modelos**.

```
┌────────────────────────────────────────────────────────────────────────┐
│  ESTRATÉGIA DE MEMÓRIA — BATCH DE MODELOS NA GPU                       │
│                                                                        │
│  Host (CPU RAM)                    Device (GPU VRAM)                   │
│  ─────────────────                 ─────────────────────               │
│  resist(n,2, N_batch)  ──H2D──▶   resist_d(n,2,N_batch)               │
│  esp(n, N_batch)       ──H2D──▶   esp_d(n, N_batch)                   │
│                                                                        │
│  Arrays invariantes (transferidos UMA VEZ):                            │
│  krwJ0J1(201,3)        ──H2D──▶   krwJ0J1_d   (reside em GPU)         │
│  wJ0(201), wJ1(201)    ──H2D──▶   wJ0_d, wJ1_d (reside em GPU)        │
│                                                                        │
│  Resultados:                                                           │
│  cH1(nt,600,nf,9)      ◀─D2H──   cH1_d(nt,600,nf,9,N_batch)          │
│  zrho1(nt,600,nf,3)    ◀─D2H──   zrho1_d(...)                         │
│                                                                        │
│  N_batch ≈ 1000 modelos (VRAM 40 GB A100 suporta ~50.000 modelos)     │
└────────────────────────────────────────────────────────────────────────┘
```

**Estimativa de memória por modelo (n=10 camadas, nf=2, nmed=600):**

| Array | Dimensão | Tipo | Memória |
|:------|:---------|:-----|:--------|
| `cH1` | (1, 600, 2, 9) | `complex(dp)` | ~86 KB |
| `zrho1` | (1, 600, 2, 3) | `real(dp)` | ~29 KB |
| `u, s, uh, sh` | (201, 10, 2) | `complex(dp)` | ~51 KB |
| `RTEdw/up, RTMdw/up` | (201, 10, 2) × 4 | `complex(dp)` | ~205 KB |
| **Total por modelo** | — | — | **~371 KB** |

Com GPU A100 (40 GB VRAM): suporte a ~100.000 modelos simultâneos.

---

##### Passo 2.4 — Multi-Model Batching na GPU

```fortran
! RunAnisoOmp.f08 — modificação para batch GPU
! ANTES: loop serial sobre modelos, um de cada vez
do modelm = 1, nmaxmodel
  call perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ...)
end do

! DEPOIS: batch de N_batch modelos por chamada GPU
integer, parameter :: N_BATCH = 1000
integer :: batch_start, batch_end, nb

do batch_start = 1, nmaxmodel, N_BATCH
  batch_end = min(batch_start + N_BATCH - 1, nmaxmodel)
  nb = batch_end - batch_start + 1

  ! Transferir batch de modelos para GPU
  call transfer_models_to_gpu(resist_batch, esp_batch, nb, ...)

  ! Lançar kernel para nb modelos em paralelo
  call launch_perfila_gpu_kernel(nb, nf, freq, ntheta, theta, ...)

  ! Sincronizar e recuperar resultados
  call cudaDeviceSynchronize()
  call transfer_results_from_gpu(cH1_batch, zrho1_batch, nb, ...)

  ! Escrever resultados do batch
  do modelm = batch_start, batch_end
    call writes_files(modelm, nmaxmodel, mypath, ...)
  end do
end do
```

---

##### Passo 2.5 — Profiling e Otimização (Ocupância e Coalescência)

**Ferramentas recomendadas:**

```bash
# NVIDIA Nsight Systems — visão geral do pipeline
nsys profile --stats=true ./PerfilaAnisoOmp_gpu input.namelist

# NVIDIA Nsight Compute — análise de kernel
ncu --set full --kernel-name "hankel_reduction_kernel" \
    ./PerfilaAnisoOmp_gpu input.namelist

# Métricas-chave a monitorar:
#   sm__warps_active.avg.pct_of_peak_sustained_active  (ocupância)
#   l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_cycle_elapsed (largura de banda)
#   sm__sass_thread_inst_executed_op_ffma_pred_on.sum  (FLOP/s)
```

**Critérios de aceitação de desempenho GPU:**

| Métrica | Meta |
|:--------|:-----|
| Ocupância de SM | > 60% |
| Eficiência de coalescência L1 | > 80% |
| Razão aritmética/memória | > 10 FLOP/byte |
| Throughput em A100 | > 400 modelos/s (1 ângulo, 2 freq, 600 med.) |

---

#### 12.7.4 Fase 3 — Validação e Benchmarking

##### Protocolo de Validação

A validade física dos resultados otimizados é verificada contra o código Fortran original usando os dados de referência em `validacao.dat` e `infovalidacao.out`.

```
┌──────────────────────────────────────────────────────────────────────┐
│  PROTOCOLO DE VALIDAÇÃO — FORTRAN ORIGINAL vs. OTIMIZADO             │
│                                                                      │
│  1. Modelo de referência: arquivo validacao.dat existente            │
│     (n=5 camadas, rho_h=[1,10,1,10,1] Ω·m, rho_v=[2,20,2,20,2])    │
│                                                                      │
│  2. Comparar saídas component a component:                           │
│     Re(Hxx), Im(Hxx), Re(Hyy), Im(Hyy), Re(Hzz), Im(Hzz)           │
│     Re(Hxy), Im(Hxy), Re(Hxz), Im(Hxz) + demais off-diagonal        │
│                                                                      │
│  3. Critério de aceitação por componente:                            │
│     |H_otimizado - H_original| / |H_original| < 1×10⁻¹⁰ (CPU)      │
│     |H_gpu - H_original| / |H_original| < 1×10⁻⁶  (GPU float32)    │
│     |H_gpu - H_original| / |H_original| < 1×10⁻¹⁰ (GPU float64)    │
│                                                                      │
│  4. Teste de estresse: 10.000 modelos geológicos aleatórios          │
│     com rho_h em [0.1, 10.000] Ω·m, n em [2, 20] camadas            │
└──────────────────────────────────────────────────────────────────────┘
```

**Script de validação (Python, usando a saída binária do simulador):**

```python
import numpy as np
import struct

def read_fortran_binary(filename, n_records, n_cols=22):
    """Lê arquivo binário unformatted stream do simulador Fortran."""
    data = []
    with open(filename, 'rb') as f:
        for _ in range(n_records):
            record = struct.unpack(f'i{n_cols-1}d', f.read(4 + (n_cols-1)*8))
            data.append(record)
    return np.array(data)

# Comparar original vs. otimizado
data_orig = read_fortran_binary('validacao_original.dat', n_records=600)
data_opt  = read_fortran_binary('validacao_otimizado.dat', n_records=600)

# Erro relativo por componente (colunas 6–22 são Re/Im dos 9 componentes)
for col_idx, comp_name in enumerate(['Re(Hxx)', 'Im(Hxx)', 'Re(Hxy)', 'Im(Hxy)',
                                      'Re(Hxz)', 'Im(Hxz)', 'Re(Hyx)', 'Im(Hyx)',
                                      'Re(Hyy)', 'Im(Hyy)', 'Re(Hyz)', 'Im(Hyz)',
                                      'Re(Hzx)', 'Im(Hzx)', 'Re(Hzy)', 'Im(Hzy)',
                                      'Re(Hzz)', 'Im(Hzz)'], start=4):
    ref  = data_orig[:, col_idx]
    opt  = data_opt [:, col_idx]
    mask = np.abs(ref) > 1e-30
    err  = np.max(np.abs(opt[mask] - ref[mask]) / np.abs(ref[mask]))
    status = "PASS" if err < 1e-10 else "FAIL"
    print(f"  {comp_name:12s}: max_rel_err = {err:.2e}  [{status}]")
```

##### Metodologia de Benchmark

```bash
#!/bin/bash
# benchmark_pipeline_a.sh — medir speedup em cada fase

N_MODELS=1000
N_THREADS_LIST="1 2 4 8 16"
BASELINE_TIME=0

echo "=== Benchmark Pipeline A ==="
echo "Modelos: ${N_MODELS}, CPU: $(nproc) cores"
echo ""

for VERSION in "original" "fase1_workspace" "fase1_cache" "fase1_simd" "fase1_all" "fase2_acc" "fase2_cuda"; do
  for NTHREADS in $N_THREADS_LIST; do
    export OMP_NUM_THREADS=$NTHREADS
    TIME=$( { time ./PerfilaAnisoOmp_${VERSION} config_benchmark.namelist; } 2>&1 | grep real | awk '{print $2}' )
    echo "  ${VERSION} | threads=${NTHREADS} | time=${TIME}"
  done
done
```

##### Tabela de Desempenho Esperado (Fase 1 + Fase 2)

| Versão | Threads/Dispositivo | Tempo/modelo | Speedup vs. original | Throughput (mod/h) |
|:-------|:-------------------:|:------------:|:--------------------:|:------------------:|
| Original (baseline) | 16 CPU | 2,40 s | 1,0× | 24.000 |
| Fase 1 — workspace | 16 CPU | 1,50 s | 1,6× | 38.400 |
| Fase 1 — +cache | 16 CPU | 0,90 s | 2,7× | 64.000 |
| Fase 1 — +SIMD | 16 CPU | 0,70 s | 3,4× | 82.300 |
| Fase 1 — completa | 16 CPU | 0,50 s | 4,8× | 115.000 |
| Fase 2 — OpenACC | A100 GPU | 0,12 s | 20× | 480.000 |
| Fase 2 — CUDA batch | A100 GPU | 0,048 s | 50× | 1.200.000 |

---

#### 12.7.5 Cronograma e Métricas de Sucesso

```
┌───────────────────────────────────────────────────────────────────────┐
│  CRONOGRAMA — PIPELINE A                                              │
│                                                                       │
│  Semana  1: Passo 1.1 (workspace) + 1.5 (escalonador) + testes       │
│  Semana  2: Passo 1.2 (cache commonarraysMD) + benchmark parcial      │
│  Semana  3: Passo 1.3 (collapse) + 1.4 (SIMD) + 1.6 (camadT cache)  │
│  Semana  4: Benchmark completo Fase 1 + validação + commit            │
│  Semanas 5–6: Protótipo OpenACC (Passo 2.1) + testes GPU             │
│  Semanas 7–9: CUDA Fortran kernels (Passos 2.2–2.4)                  │
│  Semana 10: Profiling Nsight + ajuste ocupância (Passo 2.5)           │
│  Semanas 11–12: Validação completa + benchmark final (Fase 3)         │
└───────────────────────────────────────────────────────────────────────┘
```

**KPIs (Key Performance Indicators):**

| KPI | Métrica | Meta | Prazo |
|:----|:--------|:----:|:-----:|
| K1 | Speedup Fase 1 (16 threads, 1000 modelos) | ≥ 4× | Semana 4 |
| K2 | Erro de validação CPU (max relativo) | < 1×10⁻¹⁰ | Semana 4 |
| K3 | Speedup Fase 2 OpenACC (A100, batch=1000) | ≥ 15× | Semana 6 |
| K4 | Speedup Fase 2 CUDA (A100, batch=1000) | ≥ 40× | Semana 10 |
| K5 | Erro de validação GPU (max relativo) | < 1×10⁻⁶ | Semana 12 |
| K6 | Escalabilidade CPU (efficiency Amdahl) | ≥ 70% | Semana 4 |
| K7 | Ocupância GPU (Nsight Compute) | ≥ 60% | Semana 10 |
| K8 | Reprodutibilidade (mesmo resultado em 10 execuções) | 100% | Semana 12 |

---

### 12.8 Pipeline B — Roteiro de Novos Recursos para o Simulador Fortran

#### 12.8.1 Fase 1 — Melhorias Imediatas (1–2 semanas)

Esta fase adiciona recursos de alta utilidade prática com impacto mínimo na estrutura do código existente. Todos os itens podem ser implementados sem alterar as sub-rotinas físicas principais (`commonarraysMD`, `commonfactorsMD`, `hmd_TIV_optimized`, `vmd_optimized`).

---

##### B1.1 — Seleção Adaptativa de Filtro de Hankel

**Motivação:** O código atual usa exclusivamente o filtro de Werthmuller de 201 pontos (`J0J1Wer` em `filtersv2.f08`). Para geometrias de longo espaçamento (dTR > 5 m) ou frequências muito baixas (< 1 kHz), os 201 pontos são excessivos — o filtro Kong de 61 pontos oferece precisão equivalente (erro < 0,1%) com apenas 30% do custo computacional. Para frequências muito altas (> 100 kHz) com resistividade baixa (< 0,1 Ω·m), o filtro de Anderson de 801 pontos pode ser necessário para precisão numérica adequada.

**O que implementar:** Seleção automática em `perfila1DanisoOMP` baseada em critério de qualidade (produto `r × k_max`, onde `k_max = kr_max / r` é o wavenumber máximo do filtro).

**Impacto:** Redução de 30–70% no tempo de cálculo para configurações de baixa frequência / longo espaçamento.

**Implementação — seleção adaptativa em `perfila1DanisoOMP`:**

```fortran
subroutine select_hankel_filter(freq, r, rho_min, npt_out, krJ0J1, wJ0, wJ1)
  ! Seleciona o filtro de Hankel ótimo para a geometria e frequência.
  !
  ! Critério de seleção baseado no parâmetro adimensional:
  !   kappa = r * sqrt(omega * mu * sigma_max)  (skin-depth normalizado)
  !
  ! kappa < 0.1  → Kong 61 pts  (resposta quase-estática, alta condutividade)
  ! 0.1 ≤ kappa ≤ 10 → Werthmuller 201 pts  (regime intermediário padrão)
  ! kappa > 10   → Anderson 801 pts  (alta frequência, resposta de onda)
  implicit none
  real(dp), intent(in) :: freq, r, rho_min
  integer, intent(out) :: npt_out
  real(dp), dimension(:), allocatable, intent(out) :: krJ0J1, wJ0, wJ1

  real(dp) :: omega, sigma_max, kappa

  omega     = 2.d0 * pi * freq
  sigma_max = 1.d0 / rho_min            ! condutividade máxima do modelo
  kappa     = r * sqrt(omega * mu * sigma_max)

  if (kappa < 0.1d0) then
    npt_out = 61
    call J0J1Kong(61, krJ0J1, wJ0, wJ1)
  else if (kappa <= 10.d0) then
    npt_out = 201
    call J0J1Wer(201, krJ0J1, wJ0, wJ1)
  else
    npt_out = 801
    ! call J0J1Anderson(801, krJ0J1, wJ0, wJ1)  ! a implementar
    write(*,'(A,F8.3,A)') ' AVISO: kappa=', kappa, &
      '. Filtro 801pts recomendado (Anderson). Usando 201pts.'
    npt_out = 201
    call J0J1Wer(201, krJ0J1, wJ0, wJ1)
  end if

  write(*,'(A,F6.2,A,I3,A)') ' Filtro Hankel selecionado: kappa=', kappa, &
    ', npt=', npt_out, ' pontos'
end subroutine select_hankel_filter
```

**Tabela de seleção de filtro:**

```
┌──────────────────┬───────────────┬──────────────────────────────────────┐
│  Regime          │  Filtro       │  Casos típicos                       │
├──────────────────┼───────────────┼──────────────────────────────────────┤
│  Quasi-estático  │  Kong 61pt    │  f < 1 kHz, dTR = 1m, rho > 10 Ω·m │
│  Intermediário   │  Werthmuller  │  f = 20 kHz, dTR = 1m (padrão LWD)  │
│                  │  201pt ★      │                                      │
│  Alta frequência │  Anderson     │  f > 100 kHz, rho < 0.1 Ω·m         │
│                  │  801pt        │  (a implementar)                     │
└──────────────────┴───────────────┴──────────────────────────────────────┘
★ = filtro atual do código
```

---

##### B1.2 — Processamento em Paralelo de Múltiplas Frequências

**Motivação:** Atualmente, o laço `do i = 1, nf` dentro de `fieldsinfreqs` é serial. Para simulações com `nf > 2` frequências (por exemplo, `nf = 4` para sistemas LWD de múltipla frequência como SCOPE/ARC), paralelizar sobre frequências oferece speedup proporcional a `nf`.

**Implementação:**

```fortran
! Em fieldsinfreqs: paralelizar o laço de frequências com OpenMP
! (quando chamado sem laço paralelo externo ativo — caso de threading simples)

! Pré-computar eta fora do laço (invariante para todas as frequências)
do i_lay = 1, n
  eta(i_lay,1) = 1.d0 / resist(i_lay,1)
  eta(i_lay,2) = 1.d0 / resist(i_lay,2)
end do

! Arrays de resultados por frequência (pré-alocados fora do laço)
complex(dp) :: u_f(npt,n,nf), s_f(npt,n,nf), uh_f(npt,n,nf), sh_f(npt,n,nf)
complex(dp) :: RTEdw_f(npt,n,nf), RTEup_f(npt,n,nf)
complex(dp) :: RTMdw_f(npt,n,nf), RTMup_f(npt,n,nf)
complex(dp) :: AdmInt_f(npt,n,nf)

!$omp parallel do schedule(static) num_threads(min(nf, omp_get_max_threads())) &
!$omp   private(i, freq, omega, zeta)
do i = 1, nf
  freq  = freqs(i)
  omega = 2.d0 * pi * freq
  zeta  = cmplx(0.d0, 1.d0, kind=dp) * omega * mu
  zrho(i,:) = (/zobs, resist(layerObs,1), resist(layerObs,2)/)

  call commonarraysMD(n, npt, r, krJ0J1, zeta, h, eta, &
                      u_f(:,:,i), s_f(:,:,i), uh_f(:,:,i), sh_f(:,:,i), &
                      RTEdw_f(:,:,i), RTEup_f(:,:,i), &
                      RTMdw_f(:,:,i), RTMup_f(:,:,i), AdmInt_f(:,:,i))
end do
!$omp end parallel do

! Fase 2: commonfactors + hmd/vmd (dependem de camadT — potencialmente serial)
do i = 1, nf
  call commonfactorsMD(n, npt, Tz, h, prof, camadT, &
                       u_f(:,:,i), s_f(:,:,i), uh_f(:,:,i), sh_f(:,:,i), &
                       RTEdw_f(:,:,i), RTEup_f(:,:,i), &
                       RTMdw_f(:,:,i), RTMup_f(:,:,i), &
                       Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)
  ! ... hmd e vmd ...
end do
```

---

##### B1.3 — Validação de Entradas e Tratamento de Erros

**Motivação:** Atualmente, entradas inválidas (frequências fora do range, espaçamentos negativos, resistividades zero) podem causar comportamento indefinido, `NaN`/`Inf` silenciosos ou `STOP` sem mensagem diagnóstica. A adição de uma sub-rotina de validação melhora a usabilidade e facilita depuração.

```fortran
subroutine validate_inputs(nf, freq, n, resist, esp, dTR, h1, tj, p_med, ierr)
  ! Valida todos os parâmetros de entrada do simulador.
  ! ierr = 0: tudo válido; ierr > 0: número de erros encontrados.
  implicit none
  integer, intent(in) :: nf, n
  real(dp), intent(in) :: freq(nf), resist(n,2), esp(n), dTR, h1, tj, p_med
  integer, intent(out) :: ierr

  integer :: i
  ierr = 0

  ! Validar frequências (range LWD típico: 100 Hz – 2 MHz)
  do i = 1, nf
    if (freq(i) < 1.d2 .or. freq(i) > 2.d6) then
      write(*,'(A,I3,A,ES12.4,A)') &
        ' ERRO: freq(', i, ') = ', freq(i), ' Hz fora do range [100, 2e6] Hz'
      ierr = ierr + 1
    end if
  end do

  ! Validar resistividades (evitar divisão por zero em eta = 1/rho)
  do i = 1, n
    if (resist(i,1) <= 0.d0 .or. resist(i,2) <= 0.d0) then
      write(*,'(A,I3,A,2ES12.4)') &
        ' ERRO: resist(', i, ',:) <= 0:', resist(i,1), resist(i,2)
      ierr = ierr + 1
    end if
    ! Anisotropia fisicamente razoável: rho_v >= rho_h
    if (resist(i,2) < resist(i,1)) then
      write(*,'(A,I3,A)') &
        ' AVISO: camada ', i, ': rho_v < rho_h (anisotropia invertida)'
    end if
  end do

  ! Validar espaçamento (dTR deve ser positivo)
  if (dTR <= 0.d0) then
    write(*,'(A,ES12.4)') ' ERRO: dTR <= 0: ', dTR
    ierr = ierr + 1
  end if

  ! Validar janela de perfilagem
  if (tj <= 0.d0) then
    write(*,'(A,ES12.4)') ' ERRO: tj <= 0 (comprimento de janela): ', tj
    ierr = ierr + 1
  end if

  if (ierr > 0) then
    write(*,'(A,I3,A)') ' Simulação abortada: ', ierr, ' erro(s) de validação.'
    stop 1
  end if
end subroutine validate_inputs
```

---

##### B1.4 — Logging e Relatório de Progresso

**Motivação:** Simulações de 1.000+ modelos levam horas. O código atual não emite nenhuma indicação de progresso, dificultando a estimativa de tempo restante e a detecção de travamentos.

```fortran
! Em RunAnisoOmp.f08 — adicionar relatório de progresso
real(dp) :: t_start, t_now, t_per_model, t_remaining
integer :: modelm, n_done, report_interval

t_start = omp_get_wtime()
report_interval = max(1, nmaxmodel / 20)  ! reportar a cada 5% de progresso

do modelm = 1, nmaxmodel
  call perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ...)

  n_done = modelm
  if (mod(n_done, report_interval) == 0 .or. n_done == nmaxmodel) then
    t_now       = omp_get_wtime()
    t_per_model = (t_now - t_start) / real(n_done, dp)
    t_remaining = t_per_model * real(nmaxmodel - n_done, dp)
    write(*,'(A,I6,A,I6,A,F5.1,A,F7.1,A,F7.1,A)') &
      ' Progresso: ', n_done, '/', nmaxmodel, &
      ' (', real(n_done,dp)/real(nmaxmodel,dp)*100.d0, '%)' , &
      ' | ', t_per_model, ' s/mod | ETA: ', t_remaining/60.d0, ' min'
  end if
end do

t_now = omp_get_wtime()
write(*,'(A,F8.2,A,F6.3,A)') &
  ' Concluído! Tempo total: ', (t_now-t_start)/60.d0, ' min | ', &
  (t_now-t_start)/real(nmaxmodel,dp), ' s/modelo (média)'
```

---

#### 12.8.2 Fase 2 — Extensões de Médio Prazo (2–4 semanas)

##### B2.1 — Suporte a Múltiplos Receptores (Vários Valores de `dTR`)

**Motivação:** Ferramentas LWD comerciais modernas (Baker Hughes AziTrak, Schlumberger arcVISION) possuem 2–5 arranjos TR com espaçamentos diferentes (por exemplo, dTR = 0.5, 1.0, 2.0, 4.0 m). Cada espaçamento fornece investigação em profundidade diferente, aumentando o conteúdo de informação para inversão.

**O que implementar:** Generalizar `perfila1DanisoOMP` para aceitar um array `dTR_arr(n_receivers)` em vez de um escalar `dTR`.

```fortran
subroutine perfila1DanisoOMP_multiRx(modelm, nmaxmodel, mypath, nf, freq, &
                                      ntheta, theta, h1, tj, &
                                      n_receivers, dTR_arr, &
                                      p_med, n, resist, esp, filename)
  implicit none
  integer, intent(in) :: modelm, nmaxmodel, nf, ntheta, n, n_receivers
  real(dp), intent(in) :: freq(nf), theta(ntheta), h1, tj
  real(dp), intent(in) :: dTR_arr(n_receivers)  ! array de espaçamentos
  real(dp), intent(in) :: p_med, resist(n,2), esp(n)
  character(*), intent(in) :: mypath, filename

  ! Arrays de saída agora têm dimensão extra: n_receivers
  complex(dp), allocatable :: cH1_all(:,:,:,:,:)    ! (nt, nmmax, nf, 9, n_recv)
  real(dp),    allocatable :: zrho1_all(:,:,:,:,:)  ! (nt, nmmax, nf, 3, n_recv)

  integer :: ir

  allocate(cH1_all (ntheta, nmmax, nf, 9, n_receivers))
  allocate(zrho1_all(ntheta, nmmax, nf, 3, n_receivers))

  ! Laço sobre receptores (potencialmente paralelizável com OpenMP)
  !$omp parallel do schedule(static) private(ir)
  do ir = 1, n_receivers
    ! Reutilizar commonarraysMD para cada receptor (r muda com dTR_arr(ir))
    ! ... corpo da simulação para dTR = dTR_arr(ir) ...
    cH1_all (:,:,:,:,ir) = ...
    zrho1_all(:,:,:,:,ir) = ...
  end do
  !$omp end parallel do

  ! Escrever arquivo com coluna extra identificando o receptor
  call writes_files_multiRx(modelm, nmaxmodel, mypath, &
                             zrho1_all, cH1_all, n_receivers, dTR_arr, &
                             ntheta, theta, nf, freq, nmed, filename)
  deallocate(cH1_all, zrho1_all)
end subroutine perfila1DanisoOMP_multiRx
```

**Impacto no arquivo de saída:** O formato binário `stream` existente é mantido; adiciona-se apenas um registro de cabeçalho com `n_receivers` e `dTR_arr` no arquivo `.out`.

---

##### B2.2 — Gradientes por Diferenças Finitas para PINNs (∂H/∂ρ)

**Motivação:** O treinamento de PINNs (Physics-Informed Neural Networks) com o simulador Fortran como `forward model` requer gradientes ∂H/∂ρ_h e ∂H/∂ρ_v para cada camada. A abordagem mais simples (e suficiente para treinamento offline de dados de inversão) são diferenças finitas centradas com passo `δρ` ótimo.

**O que implementar:** Sub-rotina `compute_gradients_fd` que chama `perfila1DanisoOMP` com perturbações de resistividade.

```fortran
subroutine compute_gradients_fd(nf, freq, ntheta, theta, h1, tj, dTR, p_med, &
                                 n, resist, esp, mypath, filename, &
                                 dHdRho_h, dHdRho_v)
  ! Calcula os gradientes ∂H_{ij}/∂ρ_{h,k} e ∂H_{ij}/∂ρ_{v,k}
  ! para todas as n camadas usando diferenças finitas centradas.
  !
  ! Custo: 2*n chamadas adicionais ao simulador (uma por perturbação).
  ! Alternativa mais eficiente: equações adjuntas (ver B3.1).
  implicit none
  integer, intent(in) :: nf, ntheta, n
  real(dp), intent(in) :: freq(nf), theta(ntheta), h1, tj, dTR, p_med
  real(dp), intent(in) :: resist(n,2), esp(n)
  character(*), intent(in) :: mypath, filename
  ! Gradientes de saída: (nmed, nf, 9, n, 2) — última dim: h=1, v=2
  complex(dp), intent(out) :: dHdRho_h(nf,9,n), dHdRho_v(nf,9,n)

  integer :: k_lay
  real(dp) :: resist_plus(n,2), resist_minus(n,2)
  real(dp) :: delta_rho, rho_ref
  complex(dp) :: cH_plus(nf,9), cH_minus(nf,9)

  ! Passo ótimo para diferenças finitas: δρ = sqrt(eps_mach) * ρ
  real(dp), parameter :: FD_STEP_REL = 1.d-4  ! 0.01% de perturbação relativa

  do k_lay = 1, n
    resist_plus  = resist
    resist_minus = resist

    ! Perturbação em rho_h (componente horizontal)
    rho_ref = resist(k_lay, 1)
    delta_rho = max(FD_STEP_REL * rho_ref, 1.d-6)  ! passo mínimo de 1e-6 Ω·m
    resist_plus (k_lay, 1) = rho_ref + delta_rho
    resist_minus(k_lay, 1) = rho_ref - delta_rho

    call fieldsinfreqs_single(ang_ref, nf, freq, posTR_ref, dipolo, &
                              npt, krwJ0J1, n, h, prof, resist_plus, &
                              zrho_dummy, cH_plus)
    call fieldsinfreqs_single(ang_ref, nf, freq, posTR_ref, dipolo, &
                              npt, krwJ0J1, n, h, prof, resist_minus, &
                              zrho_dummy, cH_minus)

    dHdRho_h(:,:,k_lay) = (cH_plus - cH_minus) / (2.d0 * delta_rho)

    ! Perturbação em rho_v (componente vertical)
    resist_plus  = resist
    resist_minus = resist
    rho_ref = resist(k_lay, 2)
    delta_rho = max(FD_STEP_REL * rho_ref, 1.d-6)
    resist_plus (k_lay, 2) = rho_ref + delta_rho
    resist_minus(k_lay, 2) = rho_ref - delta_rho

    call fieldsinfreqs_single(ang_ref, nf, freq, posTR_ref, dipolo, &
                              npt, krwJ0J1, n, h, prof, resist_plus, &
                              zrho_dummy, cH_plus)
    call fieldsinfreqs_single(ang_ref, nf, freq, posTR_ref, dipolo, &
                              npt, krwJ0J1, n, h, prof, resist_minus, &
                              zrho_dummy, cH_minus)

    dHdRho_v(:,:,k_lay) = (cH_plus - cH_minus) / (2.d0 * delta_rho)
  end do
end subroutine compute_gradients_fd
```

**Custo computacional:** Para `n = 10` camadas e 1 medição de referência: 20 chamadas adicionais ao simulador forward, ou seja, overhead de 20× sobre uma simulação única. Para geração offline de conjunto de treinamento (1.000 modelos × 20 perturbações = 20.000 simulações extras), o custo total com Pipeline A CPU ≈ 20.000 × 0,5 s / 600 ≈ 17 segundos por modelo, ou ~4,7 horas para 1.000 modelos com gradientes completos.

---

##### B2.3 — Formato de Saída Configurável (Binário Stream vs. Texto Formatado)

**Motivação:** O formato binário `unformatted stream` atual é eficiente para grandes volumes de dados mas dificulta depuração, inspeção manual e integração com ferramentas externas (MATLAB, Julia, ParaView). Adicionar um modo de saída texto formatado facilita o desenvolvimento.

```fortran
! Em writes_files: adicionar parâmetro de formato de saída
subroutine writes_files(modelm, nmaxmodel, mypath, zrho, cH, &
                        ntheta, theta, nf, freq, nmeds, filename, &
                        output_format)
  implicit none
  character(10), intent(in) :: output_format  ! 'binary' | 'text' | 'both'

  select case(trim(output_format))
    case('binary')
      ! Formato atual: binário stream (padrão de produção)
      open(unit=1000, file=fileTR, form='unformatted', access='stream', &
           status='unknown', position='append')
      ! ... write binário existente ...

    case('text')
      ! Formato texto para depuração e integração com scripts Python
      open(unit=1000, file=fileTR//'.txt', form='formatted', &
           status='unknown', position='append')
      ! Cabeçalho com metadados
      write(1000,'(A)') '# i  z_obs  rho_h  rho_v  Re(Hxx) Im(Hxx) ...'
      do k = 1, ntheta
        do j = 1, nf
          do i = 1, nmeds(k)
            write(1000,'(I7, 2(1X,ES14.6), 18(1X,ES18.10))') &
              i, zrho(k,i,j,1), zrho(k,i,j,2), &
              real(cH(k,i,j,1)), aimag(cH(k,i,j,1)), &
              ! ... demais componentes ...
              real(cH(k,i,j,9)), aimag(cH(k,i,j,9))
          end do
        end do
      end do

    case('both')
      ! Escrever ambos os formatos simultaneamente
      ! ... chamadas recursivas ou código duplicado ...
  end select
end subroutine writes_files
```

---

##### B2.4 — Aproximação de Camadas Inclinadas (Dip ≠ 0°)

**Motivação:** O simulador atual modela camadas estritamente horizontais (TIV). Para geosteering em poços direcionais perfurando formações com dip estratigráfico real (geralmente 2–10°), a aproximação de "camadas inclinadas" melhora a fidelidade do modelo sem recorrer a modelagem 2D completa. A aproximação consiste em decompor a geometria em componentes horizontal e vertical considerando o ângulo de dip, e ajustar as profundidades de interface em função da posição horizontal do arranjo TR.

```fortran
! Aproximação de camadas inclinadas: ajustar prof() em função da posição x
subroutine apply_dip_correction(n, prof_ref, esp_ref, x_position, dip_angle_deg, &
                                 prof_corrected, esp_corrected)
  ! Desloca as interfaces de camada lateralmente com base no dip da formação.
  ! Válido para dip < 30° (aproximação de primeira ordem).
  !
  ! Geometria:
  !   Interface original em z = prof_ref(k) (coordenada vertical, x=0)
  !   Interface corrigida: z = prof_ref(k) + x_position * tan(dip_rad)
  implicit none
  integer, intent(in) :: n
  real(dp), intent(in) :: prof_ref(0:n), esp_ref(n), x_position, dip_angle_deg
  real(dp), intent(out) :: prof_corrected(0:n), esp_corrected(n)

  real(dp) :: dip_rad, dz_per_dx, dz_shift
  integer :: k

  dip_rad    = dip_angle_deg * pi / 180.d0
  dz_per_dx  = tan(dip_rad)
  dz_shift   = x_position * dz_per_dx

  prof_corrected(0) = prof_ref(0)  ! semiespaço superior: inalterado
  do k = 1, n
    prof_corrected(k) = prof_ref(k) + dz_shift
  end do

  ! Recalcular espessuras a partir das profundidades corrigidas
  esp_corrected(1) = 0.d0  ! primeira interface: posição de referência
  do k = 2, n-1
    esp_corrected(k) = prof_corrected(k) - prof_corrected(k-1)
  end do
  esp_corrected(n) = 0.d0  ! semiespaço inferior

  if (dip_angle_deg > 15.d0) then
    write(*,'(A,F5.1,A)') ' AVISO: dip=', dip_angle_deg, &
      '° > 15°. Aproximação pode ter erro > 5% em Hzz.'
  end if
end subroutine apply_dip_correction
```

Esta sub-rotina é chamada dentro do laço `j` de medições em `perfila1DanisoOMP`, passando `x_position = (j-1) * px` como posição horizontal relativa.

---

#### 12.8.3 Fase 3 — Extensões de Longo Prazo (1–3 meses)

##### B3.1 — Equações Adjuntas para Gradientes Exatos

**Motivação:** Diferenças finitas (B2.2) têm custo computacional de 2n chamadas forward por ponto de medição, e sofrem de cancelamento numérico para passos muito pequenos. As equações adjuntas permitem calcular **todos os gradientes** ∂H/∂ρ_k (para todas as `n` camadas simultaneamente) com o custo de apenas **2 soluções de problema adjunto** (forward + adjoint), independentemente de `n`.

**Princípio:** Para a resposta escalar `J = <H, W>` (H = campo, W = vetor de pesos), o gradiente em relação a `m` (vetor de parâmetros do modelo) é:

```
∂J/∂m = Re( λ^H ∂A/∂m u )
```

onde `u` é a solução forward (`A u = s`), `λ` é a solução adjunta (`A^H λ = W`), e `A` é a matriz do sistema linear EM (coeficientes de reflexão acoplados).

**Roadmap de implementação:**

```
┌──────────────────────────────────────────────────────────────────────┐
│  IMPLEMENTAÇÃO EQUAÇÕES ADJUNTAS (3 etapas)                          │
│                                                                      │
│  Etapa 3.1.a (semana 1–2): Derivar equações adjuntas analíticas      │
│    para os coeficientes de reflexão RTEdw, RTEup, RTMdw, RTMup       │
│    em função de ρ_h e ρ_v (ver Habashy & Groom, 2004)               │
│                                                                      │
│  Etapa 3.1.b (semana 3–5): Implementar sub-rotina adjoint_solve()   │
│    que resolve o problema transposto-conjugado usando os mesmos      │
│    arrays de propagação (u, s, uh, sh) calculados no forward         │
│                                                                      │
│  Etapa 3.1.c (semana 6–8): Validar gradientes adjuntos contra        │
│    diferenças finitas centradas para modelos simples (n ≤ 5)         │
│    Critério: |grad_adj - grad_fd| / |grad_fd| < 0.1%                 │
└──────────────────────────────────────────────────────────────────────┘
```

**Assinatura prevista:**

```fortran
subroutine adjoint_solve(n, npt, nf, camadT, camadR, &
                         u_all, s_all, uh_all, sh_all, &
                         RTEdw_all, RTEup_all, RTMdw_all, RTMup_all, &
                         weight_H, eta, &
                         dJdRho_h, dJdRho_v)
  ! Calcula gradientes de funcional J = <H, weight_H> em relação a
  ! rho_h(1:n) e rho_v(1:n) via solução adjunta.
  ! Custo: equivalente a 2 avaliações forward (independente de n).
  implicit none
  integer, intent(in) :: n, npt, nf, camadT, camadR
  complex(dp), intent(in) :: u_all(npt,n,nf), s_all(npt,n,nf)
  complex(dp), intent(in) :: uh_all(npt,n,nf), sh_all(npt,n,nf)
  complex(dp), intent(in) :: RTEdw_all(npt,n,nf), RTEup_all(npt,n,nf)
  complex(dp), intent(in) :: RTMdw_all(npt,n,nf), RTMup_all(npt,n,nf)
  complex(dp), intent(in) :: weight_H(nf,9)   ! pesos da funcional
  real(dp),    intent(in) :: eta(n,2)
  real(dp), intent(out) :: dJdRho_h(n), dJdRho_v(n)
  ! ... implementação a desenvolver ...
end subroutine adjoint_solve
```

---

##### B3.2 — Aproximação de Born para Perturbações 2D

**Motivação:** A aproximação de Born de primeira ordem permite modelar perturbações de resistividade em 2D (variação lateral) como integral de volume sobre o campo de background 1D. Esta extensão é fundamental para geosteering look-ahead — onde heterogeneidades laterais (falhas, bordas de camada) precisam ser detectadas antes de serem alcançadas pela broca.

**Formulação:**

```
H_Born(r) = H_background(r) + ∫∫ G(r, r') · δσ(r') · E_background(r') d²r'
```

onde `G(r, r')` é o tensor Green EM 1D (calculado pelo simulador atual) e `δσ(r')` é a perturbação de condutividade 2D.

**Implementação (esboço):**

```fortran
subroutine born_approximation_2D(nf, freq, n_background, resist_bg, esp_bg, &
                                   dTR, h1, tj, p_med, &
                                   n_pert, x_pert, z_pert, delta_sigma, &
                                   H_born)
  ! Calcula resposta EM de perturbação 2D usando aproximação de Born.
  ! background = modelo 1D definido por resist_bg, esp_bg
  ! perturbação = n_pert células 2D em posições (x_pert, z_pert)
  !              com variação de condutividade delta_sigma
  implicit none
  integer, intent(in) :: nf, n_background, n_pert
  real(dp), intent(in) :: freq(nf), resist_bg(n_background,2), esp_bg(n_background)
  real(dp), intent(in) :: dTR, h1, tj, p_med
  real(dp), intent(in) :: x_pert(n_pert), z_pert(n_pert), delta_sigma(n_pert)
  complex(dp), intent(out) :: H_born(nf,9)

  ! 1. Calcular campo E no background usando campo de campo elétrico
  !    (requer extensão de hmd_TIV para calcular E além de H)
  ! 2. Calcular tensor Green G(r_obs, r') para cada célula de perturbação
  ! 3. Integrar: H_born = H_bg + sum_j G(r_obs, r'_j) * delta_sigma_j * E_bg(r'_j)
  ! ... implementação progressiva em 4–6 semanas ...
end subroutine born_approximation_2D
```

---

##### B3.3 — Anisotropia Biaxial (σ_x ≠ σ_y ≠ σ_z)

**Motivação:** O código atual modela anisotropia TIV (transversalmente isotrópica na vertical), com σ_h = σ_x = σ_y e σ_v = σ_z. Formações geológicas com acamamento inclinado ou estruturas sedimentares cruzadas podem exibir anisotropia ortorrômbica completa (biaxial): σ_x ≠ σ_y ≠ σ_z. Isso altera os coeficientes de reflexão TE/TM e a estrutura da transformada de Hankel.

**Impacto na física:** Os modos TE e TM acoplam-se quando σ_x ≠ σ_y (o sistema deixa de ser diagonal no espaço de Hankel), exigindo solução de sistema 4×4 em vez do sistema 2×2 atual.

**Estimativa de esforço:** 3–6 semanas para implementação completa, incluindo derivação das equações de propagação e validação analítica.

---

##### B3.4 — Paralelismo MPI para Execução em Cluster

**Motivação:** Para a geração de conjuntos de dados de treinamento com >100.000 modelos geológicos, a execução em cluster de computadores (HPC) oferece speedup quase-linear com o número de nós. MPI divide os modelos entre nós, com cada nó usando OpenMP para paralelismo interno.

**Estratégia híbrida MPI + OpenMP:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  ESTRATÉGIA HÍBRIDA MPI + OpenMP                                    │
│                                                                     │
│  Nível 1 (MPI): cada processo MPI recebe um subconjunto de modelos  │
│    Processo 0: modelos 1–1000                                       │
│    Processo 1: modelos 1001–2000                                    │
│    ...                                                              │
│    Processo P-1: modelos (P-1)*1000 – P*1000                        │
│                                                                     │
│  Nível 2 (OpenMP): cada processo usa N threads para paralelizar     │
│    ângulos × medições (código existente em perfila1DanisoOMP)        │
│                                                                     │
│  Coleta: MPI_Reduce ou escrita independente por processo            │
│  (sem comunicação MPI durante cálculo EM — zero dependência)        │
│                                                                     │
│  Escalabilidade esperada: > 95% efficiency até 100 nós              │
│  (problema embaraçosamente paralelo por modelo)                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Implementação mínima em `RunAnisoOmp.f08`:**

```fortran
program RunAnisoOmpMPI
  use mpi_f08  ! Fortran 2008 MPI interface (requer MPICH ou OpenMPI)
  use DManisoTIV
  implicit none

  integer :: rank, nprocs, ierr
  integer :: model_start, model_end, n_local

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

  ! Distribuição de modelos: divisão estática por processo
  n_local    = (nmaxmodel + nprocs - 1) / nprocs
  model_start = rank * n_local + 1
  model_end   = min(model_start + n_local - 1, nmaxmodel)

  ! Cada processo simula seu subconjunto de modelos de forma independente
  do modelm = model_start, model_end
    call perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ...)
  end do

  ! Concatenar arquivos de saída (opcional: usar MPI-IO para escrita coletiva)
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  if (rank == 0) call concatenate_output_files(nprocs, mypath, filename)

  call MPI_Finalize(ierr)
end program RunAnisoOmpMPI
```

**Compilação com MPI + OpenMP:**

```bash
mpif90 -O3 -fopenmp -ffast-math \
       parameters.f08 utils.f08 filtersv2.f08 \
       magneticdipoles.f08 PerfilaAnisoOmp.f08 RunAnisoOmpMPI.f08 \
       -o PerfilaAnisoOmp_mpi

# Execução em cluster (SLURM):
# srun --nodes=10 --ntasks-per-node=1 --cpus-per-task=16 \
#      ./PerfilaAnisoOmp_mpi
```

---

#### 12.8.4 Cronograma Integrado Pipelines A+B

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  CRONOGRAMA INTEGRADO — PIPELINES A (OTIMIZAÇÃO) + B (NOVOS RECURSOS)           │
│                                                                                  │
│  Semana   Pipeline A (Otimização)            Pipeline B (Novos Recursos)         │
│  ───────  ─────────────────────────────────  ─────────────────────────────────   │
│     1     A1.1 workspace + A1.5 scheduler    B1.3 validação de entradas          │
│     2     A1.2 cache commonarraysMD          B1.4 logging de progresso           │
│     3     A1.3 collapse + A1.4 SIMD          B1.1 filtro adaptativo              │
│     4     A1.6 cache camadT + benchmark F1   B1.2 multi-freq paralelo            │
│     5     A2.1 protótipo OpenACC             B2.3 formato saída configurável     │
│     6     A2.1 testes + validação GPU        B2.1 múltiplos receptores (dTR)     │
│     7     A2.2 CUDA kernels Hankel           B2.2 gradientes FD (B2.2)           │
│     8     A2.2 CUDA kernels coef. reflexão   B2.4 dip approximation              │
│     9     A2.3 gerenciamento memória GPU     B2.4 testes dip + validação         │
│    10     A2.4 batch models + A2.5 profiling  —                                  │
│    11     A3 validação completa              B3.1 equações adjuntas (início)     │
│    12     A3 benchmark final + documentação  B3.1 adjuntas (continuação)         │
│    13–16  —                                  B3.2 Born 2D + B3.3 biaxial        │
│    17–20  —                                  B3.4 MPI cluster                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Tabela de prioridade e dependências:**

| Item | Fase | Prioridade | Depende de | Impacto para IA |
|:-----|:----:|:----------:|:----------:|:----------------|
| A1.1 Workspace pré-alocado | A1 | Alta | — | Reduz tempo de geração de dados |
| A1.2 Cache commonarraysMD | A1 | Alta | — | Idem |
| A1.4 SIMD Hankel | A1 | Alta | — | Idem |
| B1.3 Validação entradas | B1 | Alta | — | Evita dados corrompidos no dataset |
| B1.4 Logging progresso | B1 | Média | — | Monitoramento de simulações longas |
| B1.1 Filtro adaptativo | B1 | Alta | — | Reduz custo em freq. baixas |
| B2.1 Multi-receptor | B2 | Alta | B1.3 | Mais features para inversão |
| B2.2 Gradientes FD | B2 | Alta | B1.3, B1.1 | Treinamento de PINNs |
| A2.1 OpenACC | A2 | Média | A1 completo | 15–20× speedup no Colab GPU |
| A2.2 CUDA Fortran | A2 | Média | A2.1 | 40–50× speedup no Colab GPU |
| B2.4 Dip approximation | B2 | Média | B2.1 | Geosteering em poços direcionais |
| B3.1 Adjuntas | B3 | Alta | B2.2 | Treinamento PINNs 20× mais rápido |
| B3.4 MPI cluster | B3 | Média | A1 completo | Dataset >1M modelos viável |
| B3.2 Born 2D | B3 | Baixa | B3.1 | Look-ahead geosteering |
| B3.3 Biaxial | B3 | Baixa | B3.1 | Extensão física futura |

**Estimativa de ROI (Retorno por Esforço) para geração de dataset de IA:**

```
┌────────────────────────────────────────────────────────────────────┐
│  ROI ESTIMADO — IMPACTO NO PIPELINE DE GERAÇÃO DE DADOS PARA IA    │
│                                                                    │
│  Cenário base:  1.000 modelos, 1 ângulo, 2 freq, 600 med           │
│  Tempo atual:   ~40 minutos (16 threads, 2.4 s/mod)                │
│                                                                    │
│  Após Pipeline A Fase 1 (4 semanas de trabalho):                   │
│    Tempo: ~8 min (0.5 s/mod, 16 threads)                           │
│    Speedup: 4,8× | Dataset de 10.000 modelos: ~80 min             │
│                                                                    │
│  Após Pipeline A Fase 2 + B1 + B2 (12 semanas):                   │
│    Tempo GPU: ~1.5 min (0.09 s/mod, A100 batch=1000)               │
│    Speedup: 26× | Dataset de 100.000 modelos: ~2.5 horas          │
│    + Gradientes FD: dataset com ∂H/∂ρ para PINNs                  │
│                                                                    │
│  Após Pipeline A Fase 2 + B3 MPI (20 semanas):                    │
│    Cluster 10 nós × A100: ~1.000 modelos/min                       │
│    Dataset de 1.000.000 modelos: ~16 horas                         │
└────────────────────────────────────────────────────────────────────┘
```

---

**Documentos de referência para as implementações acima:**

- Werthmuller, D. (2017). "An open-source full 3D electromagnetic modeler for 1D VTI media in Python: empymod." *Geophysics*, 82(6), WB9–WB19. — filtro de 201 pontos e critérios de precisão.
- Kong, F.N. (2007). "Hankel transform filters for dipole antenna radiation in a conductive medium." *Geophysical Prospecting*, 55(1), 83–89. — filtro de 61 pontos.
- Anderson, W.L. (1979). "Numerical integration of related Hankel transforms of orders 0 and 1 by adaptive digital filtering." *Geophysics*, 44(7), 1287–1305. — filtro de 801 pontos.
- Habashy, T.M. & Groom, R. (2004). "Adjoint method for computing electromagnetic sensitivities in layered anisotropic media." *Geophysical Journal International*, 159(2), 698–712. — base teórica para B3.1.
- OpenMP Architecture Review Board (2023). "OpenMP 5.2 Specification." — `collapse`, `simd`, `schedule(guided)`.
- NVIDIA Corporation (2023). "CUDA Fortran Programming Guide." — kernels CUDA para Fortran via NVHPC.

---

## 13. Análise de Reimplementação em Python Otimizado

### 13.1 Motivação

Uma versão Python do simulador permitiria:
- Integração direta com o pipeline `geosteering_ai` (sem subprocess)
- Geração de dados on-the-fly durante o treinamento
- Eliminação da dependência de compilação Fortran
- Facilidade de extensão (novos modelos de ferramenta, multi-frequência)
- Potencial implementação de backpropagation through physics (PINNs)

### 13.2 Bibliotecas para Computação de Alto Desempenho em Python

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

### 13.3 Estratégia de Implementação Python

#### 13.3.1 Fase 1: Protótipo NumPy Vetorizado

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

#### 13.3.2 Fase 2: Compilação JIT com Numba

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

#### 13.3.3 Fase 3: GPU via Numba CUDA ou CuPy

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

### 13.4 Referência: empymod

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

### 13.5 Diferenciação Automática com JAX para PINNs

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

### 13.6 Estimativa de Desempenho Python

| Implementação | Tempo/modelo (est.) | vs Fortran | Notas |
|:-------------|:-------------------|:-----------|:------|
| Python puro | ~60 s | 150x mais lento | Inviável |
| NumPy vetorizado | ~4 s | 10x mais lento | Aceitável para protótipo |
| **Numba CPU (JIT)** | **~0.5 s** | **~1.2x mais lento** | **Viável para produção** |
| **Numba CUDA** | **~0.02 s** | **~20x mais rápido** | **Ideal para treinamento** |
| CuPy | ~0.1 s | ~4x mais rápido | Bom com batching |
| JAX (XLA) | ~0.05 s | ~8x mais rápido | Permite gradientes |

### 13.7 Roteiro de Implementação Python

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

### 13.8 Arquitetura Proposta para o Módulo Python

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


### 13.9 Pipeline A — Conversão Fortran→Python Otimizado (Numba JIT)


#### 13.9.1 Visão Geral do Pipeline A

O Pipeline A cobre a conversão sistemática do código Fortran `PerfilaAnisoOmp` para Python otimizado com Numba JIT (CPU) e Numba CUDA (GPU). O objetivo é igualar ou superar o desempenho do Fortran compilado com OpenMP, mantendo exatamente a mesma física implementada — sem simplificações matemáticas — e expondo uma interface unificada compatível com `PipelineConfig`.

**Diagrama de conversão:**

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE A — CONVERSÃO FORTRAN → PYTHON                       │
│                                                                                  │
│  ┌─────────────────────────┐                                                     │
│  │  Fortran (PerfilaAnisoOmp)│                                                   │
│  │  commonarraysMD          │                                                     │
│  │  hmd_TIV                 │◄── subroutines → Python functions                 │
│  │  vmd_TIV                 │                                                     │
│  │  convol1D (Hankel 201pt) │                                                     │
│  │  rotate_tensor           │                                                     │
│  └─────────────────────────┘                                                     │
│             │                                                                    │
│             ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐              │
│  │  FASE 1 — Protótipo NumPy                                      │              │
│  │  Validação funcional: erro relativo < 1e-10 vs Fortran         │              │
│  │  Tempo/modelo: ~4 s (10x mais lento que Fortran OpenMP)        │              │
│  └────────────────────────────────────────────────────────────────┘              │
│             │                                                                    │
│             ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐              │
│  │  FASE 2 — Numba JIT CPU                                        │              │
│  │  @njit(parallel=True) + prange sobre 201 pontos do filtro      │              │
│  │  Tempo/modelo: ~0.4 s (equivalente ao Fortran OMP 8 cores)     │              │
│  └────────────────────────────────────────────────────────────────┘              │
│             │                                                                    │
│             ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐              │
│  │  FASE 3 — Numba CUDA GPU                                       │              │
│  │  CUDA kernels + batching de múltiplos modelos                  │              │
│  │  Throughput: ~50 modelos/s (RTX 3090) vs ~2.5 modelos/s (OMP) │              │
│  └────────────────────────────────────────────────────────────────┘              │
│             │                                                                    │
│             ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐              │
│  │  ForwardSimulator(config, backend='cpu'|'gpu')                 │              │
│  │  Interface unificada — seleção via PipelineConfig              │              │
│  └────────────────────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Metas de desempenho (modelo típico: 20 camadas, 600 medições, 1 frequência):**

| Backend | Tempo/modelo | Throughput | Quando usar |
|:--------|:------------|:-----------|:------------|
| Fortran (single-core) | ~2.4 s | ~0.4 mod/s | Referência |
| Fortran (OMP 8 cores) | ~0.4 s | ~2.5 mod/s | Referência |
| Numba CPU (8 cores) | ~0.4 s | ~2.5 mod/s | Sem GPU disponível |
| Numba CUDA (batch=32) | ~0.02 s | ~50 mod/s | Colab Pro+ GPU |

**Mecanismo de seleção CPU vs GPU via `PipelineConfig`:**

```python
from geosteering_ai.config import PipelineConfig

# Seleção via preset:
config = PipelineConfig.geosinais_p4()      # default: backend='cpu'
config = PipelineConfig.robusto()           # default: backend='cpu'

# Seleção explícita no YAML:
# configs/gpu_training.yaml:
#   simulation_backend: "gpu"
#   simulation_batch_size: 32
#   simulation_fallback_to_cpu: true

config = PipelineConfig.from_yaml("configs/gpu_training.yaml")
```

---

#### 13.9.2 Fase 1 — Protótipo NumPy Vetorizado (Validação Funcional)

A Fase 1 converte cada sub-rotina Fortran para uma função Python usando NumPy puro, vetorizando sobre as 201 posições do filtro de Hankel. O objetivo não é desempenho, mas **correção física verificável** contra a saída de referência do Fortran.

**Tabela de mapeamento Fortran → Python:**

| Sub-rotina Fortran | Função Python | Módulo | Operação Principal |
|:-------------------|:--------------|:-------|:-------------------|
| `commonarraysMD` | `compute_propagation_constants()` | `propagation.py` | Constantes u, s; coeficientes TE/TM por recursão |
| `hmd_TIV` | `compute_hmd_fields()` | `dipoles.py` | Montagem dos campos HMD (6 componentes do tensor H) |
| `vmd_TIV` | `compute_vmd_fields()` | `dipoles.py` | Montagem dos campos VMD (3 componentes diagonais) |
| `convol1D` | `hankel_transform()` | `hankel.py` | Convolução discreta com filtro de 201 pontos (J0/J1) |
| `rotate_tensor` | `rotate_tensor_tiv()` | `rotation.py` | Rotação do tensor H(3×3) para coordenadas do poço |
| `geometry_setup` | `build_tr_geometry()` | `geometry.py` | Posições T-R para 600 medições |
| *(carregamento de filtros)* | `load_filter_coefficients()` | `filters.py` | Tabela de 201 coeficientes (Werthmuller 2006) |

**Protocolo de validação contra a saída Fortran:**

```python
# validation.py — protocolo de validação Fase 1
import numpy as np
from geosteering_ai.simulation.forward import ForwardSimulator
from geosteering_ai.simulation.validation import load_fortran_reference

def validate_phase1(config, reference_dat_path: str) -> dict:
    """Compara saída Python (NumPy) contra arquivo .dat do Fortran.

    Critério de aceitação:
        max_rel_error < 1e-10   (Re e Im de todas as componentes)
        mean_rel_error < 1e-12  (valor esperado: ~1e-14 com float64)

    Args:
        config: PipelineConfig com parâmetros do modelo de teste.
        reference_dat_path: Caminho para o arquivo .dat gerado pelo Fortran.

    Returns:
        Dicionário com métricas de validação por componente do tensor H.
    """
    sim = ForwardSimulator(config, backend='numpy')
    H_python = sim.simulate_model(config.test_model)   # (600, 9) complexo

    H_fortran = load_fortran_reference(reference_dat_path)  # (600, 9) complexo

    results = {}
    component_names = ['Hxx','Hxy','Hxz','Hyx','Hyy','Hyz','Hzx','Hzy','Hzz']
    for k, name in enumerate(component_names):
        ref = H_fortran[:, k]
        pred = H_python[:, k]
        # Erro relativo: |pred - ref| / (|ref| + eps) para evitar divisão por zero
        rel_err = np.abs(pred - ref) / (np.abs(ref) + 1e-30)
        results[name] = {
            'max_rel_error':  float(rel_err.max()),
            'mean_rel_error': float(rel_err.mean()),
            'passed': bool(rel_err.max() < 1e-10),
        }
    return results
```

**Estratégia de vetorização NumPy para os 201 pontos do filtro:**

Os coeficientes do filtro de Hankel de 201 pontos — que no Fortran são iterados em um laço `do i=1,npt` (com `npt=201`) — são totalmente vetorizados em NumPy usando broadcasting:

```python
import numpy as np

def hankel_transform(f_kr: np.ndarray, kr: np.ndarray,
                     filter_j0: np.ndarray, filter_j1: np.ndarray,
                     r: float) -> tuple:
    """Transformada de Hankel discreta via filtro de 201 pontos.

    Implementação vetorizada NumPy do laço Fortran convol1D.

    Física:
        F_J0(r) = (1/r) * sum_i[ f(kr_i) * J0(kr_i * r) * w_i ]
        F_J1(r) = (1/r) * sum_i[ f(kr_i) * J1(kr_i * r) * w_i ]

        Os pesos w_i estão pré-incorporados nos coeficientes do filtro
        (Werthmuller 2006), então a convolução é simplesmente um produto
        interno entre f(kr) e os coeficientes interpolados na posição r.

    Args:
        f_kr: Array (201,) complexo — integrando avaliado em cada kr_i.
        kr: Array (201,) float64 — nós de wavenumber do filtro.
        filter_j0: Array (201,) float64 — coeficientes para J0.
        filter_j1: Array (201,) float64 — coeficientes para J1.
        r: float — distância radial T-R (metros), igual a spacing_meters.

    Returns:
        Tupla (F_J0, F_J1): escalares complexos — integrais de Hankel.

    Note:
        Ref: Werthmuller D. (2006) — filtros de 201 pontos J0/J1.
             No Fortran: loop `do i=1,npt; soma += f(i)*coef(i); end do`.
             Aqui substituído por produto interno NumPy (sem loop Python).
    """
    # ── Vetorização do laço de 201 pontos ──────────────────────────────────
    # No Fortran: `do i=1,201; field_j0 += f_kr(i)*fj0(i); enddo`
    # Aqui: produto interno vetorizado — equivalente mas sem laço Python.
    # Shape: f_kr=(201,) complexo, filter_j0=(201,) real → escalar complexo.
    F_J0 = np.dot(filter_j0, f_kr) / r
    F_J1 = np.dot(filter_j1, f_kr) / r
    return F_J0, F_J1
```

**Modelos canônicos de teste para validação (10 modelos):**

| Modelo | Camadas | rho_h (Ohm.m) | rho_v (Ohm.m) | Espessura (m) | Dificuldade |
|:-------|:-------:|:-------------|:-------------|:-------------|:------------|
| M01 — Homogêneo isotropico | 1 | 10.0 | 10.0 | ∞ | Trivial |
| M02 — Homogêneo TIV | 1 | 5.0 | 20.0 | ∞ | Básica |
| M03 — 3 camadas simples | 3 | 1/10/1 | 1/10/1 | 20/20 m | Básica |
| M04 — 5 camadas TIV | 5 | misto | misto | 10 m cada | Média |
| M05 — Camadas finas (2 m) | 10 | alternado 1/100 | alternado | 2 m cada | Difícil |
| M06 — Contraste extremo | 3 | 0.1/10000/0.1 | igual | 30/20 m | Difícil |
| M07 — Anisotropia alta | 5 | 1–10 | 10–100 | 15 m cada | Difícil |
| M08 — Muitas camadas | 40 | amostrado | amostrado | 1–5 m | Alta |
| M09 — Camadas muito finas | 20 | alternado | alternado | 0.5 m | Extrema |
| M10 — Máxima complexidade | 80 | Sobol | Sobol | Sobol | Extrema |

---

#### 13.9.3 Fase 2 — Numba JIT CPU (Performance Igual ou Superior ao Fortran)

A Fase 2 reescreve as funções críticas com decoradores Numba `@njit`, habilitando compilação JIT ahead-of-time e paralelismo automático sobre os 201 pontos do filtro. O resultado esperado é desempenho equivalente ao Fortran com 8 cores OpenMP em um único processo Python.

**Decoradores e configurações Numba:**

```python
from numba import njit, prange
import numpy as np

# ── Configuração padrão para kernels críticos ──────────────────────────────
# parallel=True: habilita paralelismo automático com prange (equivalente a omp parallel do)
# cache=True: salva bytecode compilado em disco — evita recompilação entre runs
# fastmath=True: permite reordenação de FP para maior throughput (seguro aqui pois
#                resultados são comparados com rtol=1e-6 apenas para desempenho)
# nogil=True: libera GIL Python — permite paralelismo externo via threading
_NJIT_OPTS = dict(parallel=True, cache=True, fastmath=True, nogil=True)
```

**Estratégia de pré-alocação de memória (evitar pressão no GC):**

```python
@njit(**_NJIT_OPTS)
def compute_propagation_constants(
    kr: np.ndarray,          # (npt=201,) float64 — wavenumbers do filtro
    eta_h: np.ndarray,       # (n_layers,) complex128 — admitância horizontal por camada
    eta_v: np.ndarray,       # (n_layers,) complex128 — admitância vertical por camada
    zeta: np.ndarray,        # (n_layers,) complex128 — impedância magnética por camada
    h: np.ndarray,           # (n_layers-1,) float64 — espessuras (m)
    workspace: np.ndarray,   # (npt, n_layers, 6) complex128 — workspace pré-alocado
) -> tuple:
    """Calcula constantes de propagação u, s e coeficientes TE/TM por recursão.

    Implementação Numba JIT do Fortran commonarraysMD, com paralelismo
    sobre os 201 pontos do filtro de Hankel via prange.

    Física:
        Para cada wavenumber kr_i (i=1..201) e cada camada j (j=1..n):
            u_j = sqrt(kr_i^2 - zeta_j * eta_h_j)   [modo TE, componente z]
            s_j = sqrt(kr_i^2 - zeta_j * eta_v_j)   [modo TM, componente z]

        Recursão de cima para baixo (interface n-1 → n-2 → ... → 1):
            RTE_dw_j = (u_j - u_{j+1}) / (u_j + u_{j+1}) * exp(-2*u_{j+1}*h_j)
            RTM_dw_j = analogamente com s e eta

        A recursão é SEQUENCIAL em j (dependência de dados) mas PARALELA em i.
        Isso mapeia diretamente para prange(npt) com loop interno sequencial.

    Args:
        kr: Wavenumbers do filtro de 201 pontos (espaçados logaritmicamente).
        eta_h: Admitância horizontal: eta_h_j = sigma_h_j / (i*omega*mu0).
        eta_v: Admitância vertical: eta_v_j = sigma_v_j / (i*omega*mu0).
        zeta: Impedância magnética: zeta_j = i*omega*mu0 (homogêneo em mu).
        h: Espessuras das camadas em metros (n_layers - 1 valores).
        workspace: Array pré-alocado para u, s, RTE_dw, RTM_dw, RTE_up, RTM_up.
                   Shape (201, n_layers, 6) — evita alocação dentro do loop JIT.

    Returns:
        Tupla (u, s, RTE_dw, RTM_dw, RTE_up, RTM_up): vistas do workspace.

    Note:
        Ref: Chew W.C. (1995) §2.4 — recursão de coeficientes de reflexão.
        O workspace pré-alocado elimina malloc/free dentro do loop JIT,
        reduzindo overhead de GC e fragmentação de cache L1/L2.
    """
    npt = kr.shape[0]
    n   = eta_h.shape[0]

    # ── Vistas nomeadas do workspace pré-alocado ──────────────────────────
    # workspace[:,:,0] = u   | workspace[:,:,1] = s
    # workspace[:,:,2] = RTE_dw | workspace[:,:,3] = RTM_dw
    # workspace[:,:,4] = RTE_up | workspace[:,:,5] = RTM_up
    u       = workspace[:, :, 0]
    s       = workspace[:, :, 1]
    RTE_dw  = workspace[:, :, 2]
    RTM_dw  = workspace[:, :, 3]
    RTE_up  = workspace[:, :, 4]
    RTM_up  = workspace[:, :, 5]

    # ── Loop paralelo sobre os 201 pontos do filtro (equivalente a omp parallel do) ──
    # Cada thread processa um kr_i independentemente.
    # O loop interno sobre camadas (j) é sequencial por dependência de dados.
    for i in prange(npt):
        kri2 = kr[i] * kr[i]

        # ── Constantes de propagação por camada ────────────────────────
        # u_j (modo TE): raiz do wavenumber vertical em cada camada.
        # s_j (modo TM): idêntico mas com eta_v em vez de eta_h.
        # Convenção de branch cut: Re(u) >= 0 para garantir decaimento físico.
        for j in range(n):
            u[i, j] = _csqrt_positive_real(kri2 - zeta[j] * eta_h[j])
            s[i, j] = _csqrt_positive_real(kri2 - zeta[j] * eta_v[j])

        # ── Recursão de baixo para cima: coeficientes de reflexão ──────
        # Camada mais profunda: sem reflexão (meio semi-infinito).
        RTE_dw[i, n-1] = 0.0 + 0.0j
        RTM_dw[i, n-1] = 0.0 + 0.0j

        for j in range(n-2, -1, -1):
            exp_term = np.exp(-2.0 * u[i, j+1] * h[j])
            denom_TE = u[i,j] + u[i,j+1] + (u[i,j] - u[i,j+1]) * RTE_dw[i,j+1] * exp_term
            RTE_dw[i,j] = ((u[i,j] - u[i,j+1]) + (u[i,j] + u[i,j+1]) * RTE_dw[i,j+1] * exp_term) / denom_TE

            # ── Modo TM: substitui u→s e eta_h→eta_v (Liu 2017, eq. 2.38) ──
            exp_s = np.exp(-2.0 * s[i, j+1] * h[j])
            r_s   = eta_h[j+1] * s[i,j] - eta_h[j] * s[i,j+1]
            r_s_d = eta_h[j+1] * s[i,j] + eta_h[j] * s[i,j+1]
            RTM_dw[i,j] = (r_s + r_s_d * RTM_dw[i,j+1] * exp_s) / (r_s_d + r_s * RTM_dw[i,j+1] * exp_s)

    return u, s, RTE_dw, RTM_dw, RTE_up, RTM_up
```

**Exemplo de transformada de Hankel com Numba (função crítica):**

```python
@njit(parallel=True, cache=True, fastmath=True)
def hankel_transform_batch(
    f_kr_batch: np.ndarray,   # (n_meas, npt) complex128 — integrando para cada medição
    filter_j0:  np.ndarray,   # (npt=201,) float64 — coeficientes J0
    filter_j1:  np.ndarray,   # (npt=201,) float64 — coeficientes J1
    r: float,                 # float64 — distância T-R em metros
    out_j0: np.ndarray,       # (n_meas,) complex128 — resultado J0 (pré-alocado)
    out_j1: np.ndarray,       # (n_meas,) complex128 — resultado J1 (pré-alocado)
) -> None:
    """Transformada de Hankel em lote para todas as medições (Numba JIT paralelo).

    Vetoriza o laço Fortran convol1D sobre todas as n_meas posições T-R de forma
    paralela. Cada thread processa uma medição independentemente (prange outer),
    e o produto interno de 201 pontos é executado sequencialmente (inner loop).

    Note:
        In-place: resultados escritos em out_j0 e out_j1 (sem retorno).
        O uso de arrays pré-alocados evita alocação dentro do kernel JIT.
    """
    n_meas = f_kr_batch.shape[0]
    inv_r  = 1.0 / r

    # ── Paralelismo externo: cada medição processada por uma thread ──────
    for m in prange(n_meas):
        acc_j0 = 0.0 + 0.0j
        acc_j1 = 0.0 + 0.0j
        # ── Loop interno sequencial: produto interno com filtro de 201 pts ──
        for i in range(201):
            acc_j0 += filter_j0[i] * f_kr_batch[m, i]
            acc_j1 += filter_j1[i] * f_kr_batch[m, i]
        out_j0[m] = acc_j0 * inv_r
        out_j1[m] = acc_j1 * inv_r
```

**Padrão de workspace thread-local no Numba:**

```python
# ── Criação de workspace thread-local (evita race conditions) ────────────
# No Numba, arrays criados dentro de prange são thread-local automaticamente.
# Para arrays grandes, pré-alocar fora do loop e usar slices por thread.

def build_workspace(n_layers: int, npt: int = 201) -> np.ndarray:
    """Aloca workspace pré-inicializado para o kernel de propagação.

    O workspace de shape (npt, n_layers, 6) representa 6 arrays complexos
    [u, s, RTE_dw, RTM_dw, RTE_up, RTM_up], todos de shape (npt, n_layers).
    A pré-alocação antes do loop JIT elimina malloc dentro do kernel.

    Returns:
        np.ndarray de shape (201, n_layers, 6), dtype=complex128, zerado.
    """
    return np.zeros((npt, n_layers, 6), dtype=np.complex128)
```

---

#### 13.9.4 Fase 3 — Numba CUDA GPU (Alta Performance)

A Fase 3 reescreve os kernels críticos como CUDA kernels via `numba.cuda.jit`, explorando o paralelismo massivo da GPU para processar múltiplos modelos geológicos simultaneamente. O mapeamento principal é: **uma thread por ponto do filtro × uma warp por camada × um block por medição × um grid por modelo**.

**Design de kernels CUDA:**

```python
import numba.cuda as cuda
import numpy as np
import math

# ── Kernel CUDA: constantes de propagação para batch de modelos ──────────
# Grid: (n_models, n_meas_per_model)
# Block: (npt=201,)  — 201 threads por block, uma por ponto do filtro
# Constraint: npt=201 ≤ 1024 (max threads/block) — satisfeito.
@cuda.jit
def kernel_propagation_constants(
    kr,           # (npt,) float64 — wavenumbers do filtro (constante)
    eta_h_batch,  # (n_models, n_layers) complex128
    eta_v_batch,  # (n_models, n_layers) complex128
    zeta_batch,   # (n_models, n_layers) complex128
    h_batch,      # (n_models, n_layers-1) float64
    n_layers_arr, # (n_models,) int32 — número de camadas por modelo
    u_out,        # (n_models, npt, n_layers_max) complex128 — saída u
    s_out,        # (n_models, npt, n_layers_max) complex128 — saída s
    RTE_dw_out,   # (n_models, npt, n_layers_max) complex128
    RTM_dw_out,   # (n_models, npt, n_layers_max) complex128
):
    """CUDA kernel: constantes de propagação para batch de modelos.

    Mapeamento thread → dado:
        thread_x (0..200) → ponto do filtro kr_i
        block_y           → índice do modelo geológico
    A recursão sobre camadas permanece sequencial dentro de cada thread
    (dependência de dados impossibilita paralelismo adicional nessa dimensão).
    """
    i     = cuda.threadIdx.x    # ponto do filtro (0..200)
    m_idx = cuda.blockIdx.y     # índice do modelo

    if i >= 201:
        return
    n = n_layers_arr[m_idx]
    kri2 = kr[i] * kr[i]

    # ── Constantes de propagação (u, s) para cada camada ─────────────
    for j in range(n):
        arg_u = kri2 - zeta_batch[m_idx, j] * eta_h_batch[m_idx, j]
        arg_s = kri2 - zeta_batch[m_idx, j] * eta_v_batch[m_idx, j]
        # Raiz quadrada com branch cut em Re >= 0 (física)
        u_out[m_idx, i, j]  = _cuda_csqrt(arg_u)
        s_out[m_idx, i, j]  = _cuda_csqrt(arg_s)

    # ── Recursão de baixo para cima: coeficientes de reflexão ────────
    RTE_dw_out[m_idx, i, n-1] = 0.0 + 0.0j
    RTM_dw_out[m_idx, i, n-1] = 0.0 + 0.0j

    for j in range(n-2, -1, -1):
        u_j   = u_out[m_idx, i, j]
        u_jp1 = u_out[m_idx, i, j+1]
        exp_u = cuda.libdevice.cexp(-2.0 * u_jp1 * h_batch[m_idx, j])
        num   = (u_j - u_jp1) + (u_j + u_jp1) * RTE_dw_out[m_idx, i, j+1] * exp_u
        den   = (u_j + u_jp1) + (u_j - u_jp1) * RTE_dw_out[m_idx, i, j+1] * exp_u
        RTE_dw_out[m_idx, i, j] = num / den
        # ── Modo TM: análogo com s e eta_h ──────────────────────────
        # (omitido por brevidade — estrutura idêntica ao modo TE)
```

**Estratégia de gerenciamento de memória GPU:**

```python
class CudaSimulationContext:
    """Contexto CUDA: pré-aloca arrays no device para batch de modelos.

    Evita transferências Host→Device a cada chamada, mantendo buffers
    persistentes no device para o tamanho máximo de batch configurado.

    Atributos:
        batch_size: Número máximo de modelos por batch (default: 32).
        n_layers_max: Número máximo de camadas (default: 80).
        n_meas: Número de medições por modelo (= config.sequence_length).
        npt: Pontos do filtro de Hankel (= 201).

    Uso:
        ctx = CudaSimulationContext(config)
        with ctx:
            H_tensor = ctx.simulate_batch(models_list)
    """

    def __init__(self, config):
        self.batch_size   = config.simulation_batch_size    # default: 32
        self.n_layers_max = config.max_layers               # default: 80
        self.n_meas       = config.sequence_length          # default: 600
        self.npt          = 201
        self._allocate_device_buffers()

    def _allocate_device_buffers(self):
        """Pré-aloca buffers persistentes no device (GPU DRAM)."""
        B, L, M, P = self.batch_size, self.n_layers_max, self.n_meas, self.npt
        # ── Arrays de entrada (parâmetros geológicos por modelo) ──────
        self.d_eta_h   = cuda.device_array((B, L), dtype=np.complex128)
        self.d_eta_v   = cuda.device_array((B, L), dtype=np.complex128)
        self.d_zeta    = cuda.device_array((B, L), dtype=np.complex128)
        self.d_h       = cuda.device_array((B, L-1), dtype=np.float64)
        self.d_nlayers = cuda.device_array((B,), dtype=np.int32)
        # ── Arrays intermediários (propagação) ─────────────────────
        self.d_u       = cuda.device_array((B, P, L), dtype=np.complex128)
        self.d_s       = cuda.device_array((B, P, L), dtype=np.complex128)
        self.d_RTE_dw  = cuda.device_array((B, P, L), dtype=np.complex128)
        self.d_RTM_dw  = cuda.device_array((B, P, L), dtype=np.complex128)
        # ── Array de saída (tensor H por modelo × medição) ─────────
        self.d_H_tensor = cuda.device_array((B, M, 9), dtype=np.complex128)

    def simulate_batch(self, models: list) -> np.ndarray:
        """Simula batch de modelos geológicos na GPU.

        Args:
            models: Lista de dicionários com chaves 'rho_h', 'rho_v',
                    'thicknesses', 'n_layers'. Comprimento <= batch_size.

        Returns:
            np.ndarray de shape (len(models), n_meas, 9) complex128 —
            tensor H completo para cada modelo e medição.
        """
        n = len(models)
        # ── Transferência Host→Device (apenas parâmetros geológicos) ──
        self._upload_models(models)
        # ── Lançamento dos kernels CUDA em sequência ──────────────────
        threads_per_block = (201, 1)      # 201 threads: um por ponto do filtro
        blocks_per_grid   = (1, n)        # n blocks: um por modelo no batch
        kernel_propagation_constants[blocks_per_grid, threads_per_block](
            self.d_kr, self.d_eta_h, self.d_eta_v, self.d_zeta,
            self.d_h, self.d_nlayers, self.d_u, self.d_s,
            self.d_RTE_dw, self.d_RTM_dw
        )
        # ... lançamento dos kernels HMD, VMD, Hankel, Rotation ...
        cuda.synchronize()
        # ── Transferência Device→Host (somente tensor H final) ────────
        return self.d_H_tensor[:n].copy_to_host()
```

**Otimização de ocupância CUDA:**

```python
# ── Análise de ocupância para o kernel de propagação ─────────────────────
# threads_per_block = 201  → PROBLEMA: não é múltiplo de 32 (warp size)
# Recomendação: padear para 224 (7×32) com guard `if i >= 201: return`
# Resultado: 7 warps × 32 threads = 224 threads/block → melhor coalescing

# Para o kernel de Hankel (redução):
# Usar 256 threads/block com shared memory para redução em árvore
# Cada block processa uma medição, cada thread acumula 201/256 ≈ 1 ponto

@cuda.jit
def kernel_hankel_reduction(f_kr_batch, filter_j0, filter_j1, r, out_j0, out_j1):
    """CUDA kernel de Hankel com redução em shared memory.

    Usa shared memory para acumular parcialmente o produto interno de 201
    pontos, reduzindo conflitos de banco e acessos a global memory.
    Block: (256,) — shared: (256,) complex128 × 2 = 4 KB (dentro do limite).
    """
    shared_j0 = cuda.shared.array(shape=256, dtype=numba.complex128)
    shared_j1 = cuda.shared.array(shape=256, dtype=numba.complex128)
    tid   = cuda.threadIdx.x
    m_idx = cuda.blockIdx.x    # uma medição por block

    # ── Acumulação parcial: cada thread cobre i=tid (201 < 256) ──────
    acc0 = 0.0 + 0.0j
    acc1 = 0.0 + 0.0j
    if tid < 201:
        acc0 = filter_j0[tid] * f_kr_batch[m_idx, tid]
        acc1 = filter_j1[tid] * f_kr_batch[m_idx, tid]
    shared_j0[tid] = acc0
    shared_j1[tid] = acc1
    cuda.syncthreads()

    # ── Redução em árvore (tree reduction) ───────────────────────────
    stride = 128
    while stride > 0:
        if tid < stride:
            shared_j0[tid] += shared_j0[tid + stride]
            shared_j1[tid] += shared_j1[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out_j0[m_idx] = shared_j0[0] / r
        out_j1[m_idx] = shared_j1[0] / r
```

---

#### 13.9.5 Interface Unificada CPU/GPU

A classe `ForwardSimulator` expõe uma API idêntica independentemente do backend, com fallback automático de GPU para CPU quando CUDA não está disponível.

```python
# geosteering_ai/simulation/forward.py

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Literal
import numpy as np

from geosteering_ai.config import PipelineConfig

logger = logging.getLogger("geosteering_ai.simulation.forward")


@dataclass
class SimulationResult:
    """Resultado de uma simulação EM 1D para um único modelo geológico.

    Atributos:
        H_tensor: Array (n_meas, 9) complex128 — tensor H completo rotacionado.
                  Colunas: [Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz]
                  em coordenadas do poço (após rotação).
        z_obs: Array (n_meas,) float64 — profundidades de medição (m).
        backend_used: 'numpy' | 'numba_cpu' | 'numba_cuda' — backend efetivo.
        time_seconds: float — tempo de simulação em segundos.
    """
    H_tensor:     np.ndarray
    z_obs:        np.ndarray
    backend_used: str
    time_seconds: float


class ForwardSimulator:
    """Simulador EM 1D TIV com interface unificada CPU/GPU.

    Encapsula os três backends (NumPy, Numba CPU, Numba CUDA) e expõe
    uma API consistente integrada com PipelineConfig. A seleção de backend
    é feita via config.simulation_backend, com fallback automático.

    Estrutura interna:
      ┌─────────────────────────────────────────────────────────────────┐
      │  ForwardSimulator(config)                                        │
      │       │                                                          │
      │       ├── backend='numpy'      → _NumPyBackend                  │
      │       ├── backend='cpu'        → _NumbaCPUBackend               │
      │       └── backend='gpu'        → _NumbaCUDABackend              │
      │                                    │                             │
      │                               Fallback automático               │
      │                               se CUDA indisponível              │
      │                               → _NumbaCPUBackend                │
      └─────────────────────────────────────────────────────────────────┘

    Exemplo de uso:
        config = PipelineConfig.from_yaml("configs/gpu_training.yaml")
        sim = ForwardSimulator(config)
        result = sim.simulate(model_params)    # simulação única
        H_batch = sim.simulate_batch(models)   # batch de modelos

    Note:
        Ref: Seção 13.9 de documentacao_simulador_fortran.md.
        Integração com tf.data: usar simulate_batch em tf.py_function
        dentro do map_fn do DataPipeline (Seção 13.10.1).
    """

    def __init__(
        self,
        config: PipelineConfig,
        backend: Literal['auto', 'numpy', 'cpu', 'gpu'] = 'auto',
    ):
        self.config = config
        # ── Seleção de backend com fallback automático ─────────────────
        # 1. Se backend='auto', usa config.simulation_backend (default: 'cpu').
        # 2. Se backend='gpu' mas CUDA indisponível, faz fallback para 'cpu'.
        # 3. Se backend='numpy', usa NumPy puro (apenas para validação/debug).
        requested = backend if backend != 'auto' else config.simulation_backend
        self._backend = self._resolve_backend(requested)
        self._impl    = self._build_impl(self._backend, config)
        logger.info("ForwardSimulator iniciado: backend=%s", self._backend)

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        """Resolve backend com fallback: gpu → cpu se CUDA indisponível."""
        if requested == 'gpu':
            try:
                from numba import cuda as numba_cuda
                if not numba_cuda.is_available():
                    raise RuntimeError("CUDA não disponível")
                return 'gpu'
            except Exception as e:
                logger.warning(
                    "Fallback GPU→CPU: %s. Usando Numba JIT CPU.", e
                )
                return 'cpu'
        return requested

    def simulate(self, model_params: dict) -> SimulationResult:
        """Simula um único modelo geológico.

        Args:
            model_params: Dicionário com chaves obrigatórias:
                'rho_h'       : np.ndarray (n_layers,) — resistividade horizontal (Ohm.m)
                'rho_v'       : np.ndarray (n_layers,) — resistividade vertical (Ohm.m)
                'thicknesses' : np.ndarray (n_layers-1,) — espessuras das camadas (m)
                'dip_deg'     : float — ângulo de mergulho em graus (default: 0.0)

        Returns:
            SimulationResult com tensor H e metadados.
        """
        return self._impl.simulate_single(model_params)

    def simulate_batch(self, models: list[dict]) -> np.ndarray:
        """Simula batch de modelos geológicos.

        Args:
            models: Lista de dicionários (mesmo formato de simulate()).
                    Comprimento recomendado: múltiplo de config.simulation_batch_size.

        Returns:
            np.ndarray de shape (len(models), config.sequence_length, 9) complex128.
        """
        return self._impl.simulate_batch(models)

    @property
    def backend(self) -> str:
        """Backend efetivo em uso ('numpy', 'cpu' ou 'gpu')."""
        return self._backend
```

**Integração com `PipelineConfig` (campos a adicionar):**

```python
# Em geosteering_ai/config.py — novos campos para o módulo simulation/

@dataclass
class PipelineConfig:
    # ... campos existentes ...

    # ── Configuração do Simulador Python (Seção 13) ───────────────────────
    # simulation_backend: backend padrão para ForwardSimulator.
    #   'cpu': Numba JIT com paralelismo OpenMP-equivalente (default).
    #   'gpu': Numba CUDA — requer GPU com CUDA 11.x+.
    #   'numpy': NumPy puro — apenas para validação, ~10x mais lento.
    simulation_backend: str = "cpu"

    # simulation_batch_size: número de modelos por batch na GPU.
    #   Valor ótimo depende da VRAM disponível e n_layers_max.
    #   GPU 8 GB (T4/Colab): batch_size=32 (modelo típico 20 camadas).
    #   GPU 24 GB (RTX 3090): batch_size=128.
    simulation_batch_size: int = 32

    # simulation_fallback_to_cpu: se True, faz fallback automático GPU→CPU.
    #   Recomendado: True (para compatibilidade em ambientes sem CUDA).
    simulation_fallback_to_cpu: bool = True

    # simulation_n_workers: threads Numba para backend CPU.
    #   None = auto-detectar (usa todos os cores físicos disponíveis).
    simulation_n_workers: int | None = None

    # simulation_validate_on_init: se True, valida backend contra NumPy
    #   durante __init__ do ForwardSimulator (aumenta tempo de inicialização ~2s).
    simulation_validate_on_init: bool = False
```

---

### 13.10 Pipeline B — Novos Recursos da Versão Python

A versão Python do simulador desbloqueia capacidades fundamentalmente impossíveis com o binário Fortran externo: geração on-the-fly durante treinamento, diferenciação automática para PINNs, augmentation física e substituição por rede neural surrogada. Esta seção detalha cada um desses recursos.

---

#### 13.10.1 Geração On-the-Fly durante Treinamento

Com o simulador Python disponível em processo, o `DataPipeline` pode gerar novos modelos geológicos e suas respostas EM a cada epoch — eliminando o conjunto de treinamento fixo e criando um **dataset virtualmente infinito**.

**Diagrama do fluxo on-the-fly:**

```
┌────────────────────────────────────────────────────────────────────────────┐
│              GERAÇÃO ON-THE-FLY COM SIMULADOR PYTHON                       │
│                                                                            │
│  Epoch N                                                                   │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │  GeologicModelSampler (Sobol/Monte Carlo)               │               │
│  │      ↓                                                  │               │
│  │  ForwardSimulator.simulate_batch(batch_size=32)  [GPU]  │               │
│  │      ↓                                                  │               │
│  │  H_tensor (32, 600, 9) ← campos EM brutos               │               │
│  │      ↓                                                  │               │
│  │  NoiseFn (on-the-fly, σ em A/m)                         │               │
│  │      ↓                                                  │               │
│  │  FeatureView_tf (FV computada sobre campos ruidosos)     │               │
│  │      ↓                                                  │               │
│  │  GeoSignal_tf (GS computado sobre campos ruidosos)       │               │
│  │      ↓                                                  │               │
│  │  Scaler → (X_scaled, y_scaled)                          │               │
│  │      ↓                                                  │               │
│  │  tf.data.Dataset.from_generator → model.fit()           │               │
│  └─────────────────────────────────────────────────────────┘               │
│                                                                            │
│  Epoch N+1: NOVOS modelos geológicos gerados (seed → seed+1)              │
│             → dataset estatisticamente diferente a cada epoch              │
└────────────────────────────────────────────────────────────────────────────┘
```

**Integração com `tf.data.Dataset`:**

```python
# geosteering_ai/data/pipeline.py — extensão para on-the-fly simulation

import tensorflow as tf
import numpy as np
from geosteering_ai.config import PipelineConfig
from geosteering_ai.simulation.forward import ForwardSimulator
from geosteering_ai.data.sampling import GeologicModelSampler

def build_onthefly_dataset(
    config: PipelineConfig,
    simulator: ForwardSimulator,
    sampler: GeologicModelSampler,
    *,
    steps_per_epoch: int = 200,
    seed: int = 42,
) -> tf.data.Dataset:
    """Constrói tf.data.Dataset com geração on-the-fly via simulador Python.

    O dataset é virtualmente infinito (tf.data.experimental.INFINITE_CARDINALITY):
    a cada step, novos modelos geológicos são amostrados e simulados.

    Fluxo:
        generator() → simulate_batch() → noise → FV → GS → scale → (X, y)

    Args:
        config: PipelineConfig com todos os parâmetros do pipeline.
        simulator: ForwardSimulator inicializado (CPU ou GPU).
        sampler: GeologicModelSampler configurado com distribuição Sobol.
        steps_per_epoch: Número de batches por epoch.
        seed: Semente para reprodutibilidade da amostragem.

    Returns:
        tf.data.Dataset que emite tuplas (X_scaled, y_scaled) com shape
        (batch_size, seq_len, n_features) e (batch_size, seq_len, 2).

    Note:
        O simulador é chamado via tf.py_function para compatibilidade com
        o grafo TF. O overhead do py_function é amortizado pelo batch_size.
        Em GPU, simulate_batch(32) leva ~0.02s — custo negligível vs I/O.
    """
    batch_size = config.simulation_batch_size
    rng = np.random.default_rng(seed)

    def generator():
        """Gerador Python: amostrar → simular → processar → yield."""
        while True:
            # ── Amostragem de modelos geológicos (Sobol quasi-MC) ────────
            # Cada batch tem modelos completamente novos — jamais repetidos.
            models = sampler.sample(batch_size, rng=rng)

            # ── Simulação EM (CPU ou GPU via ForwardSimulator) ────────────
            # Retorna (batch_size, seq_len, 9) complex128
            H_batch = simulator.simulate_batch(models)

            # ── Conversão para float32 (Re e Im empilhados) ──────────────
            # Shape final: (batch_size, seq_len, 18) — 9 componentes × Re+Im
            H_real = np.concatenate([H_batch.real, H_batch.imag], axis=-1).astype(np.float32)

            # ── Targets: rho_h, rho_v em log10 ───────────────────────────
            rho_h = np.stack([m['rho_h_profile'] for m in models], axis=0)
            rho_v = np.stack([m['rho_v_profile'] for m in models], axis=0)
            y_log10 = np.log10(np.stack([rho_h, rho_v], axis=-1)).astype(np.float32)

            yield H_real, y_log10

    output_sig = (
        tf.TensorSpec(shape=(batch_size, config.sequence_length, 18), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, config.sequence_length, 2),  dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_sig)

    # ── Aplicar noise + FV + GS + scale (mesmo map_fn do DataPipeline) ──
    map_fn = _build_simulation_map_fn(config)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
```

**Conceito de dataset infinito:**

```python
# O dataset não tem cardinalidade finita — nunca esgota.
# O treinamento é controlado por steps_per_epoch no model.fit():

history = model.fit(
    train_ds,                        # Dataset infinito (on-the-fly)
    steps_per_epoch=200,             # 200 batches × 32 modelos = 6400 modelos/epoch
    epochs=100,
    validation_data=val_ds,          # Dataset offline fixo (validação)
    callbacks=build_callbacks(config, model, noise_var),
)
# Cada epoch vê 6.400 modelos DIFERENTES — dataset efetivo: 640.000 modelos/run
# Com Fortran externo: geração prévia de ~100.000 modelos em ~40h de CPU.
# Com simulador Python GPU: os mesmos 100.000 modelos em ~2.000s (33 min).
```

---

#### 13.10.2 Diferenciação Automática via JAX

O simulador Python em JAX permite calcular gradientes exatos do forward model em relação aos parâmetros do meio, habilitando o treinamento de PINNs com termo de regularização físico sem necessidade de diferenças finitas.

**Forward model diferenciável em JAX:**

```python
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=(4,))
def forward_em_1d_jax(
    rho_h: jnp.ndarray,         # (n_layers,) float32 — resistividade horizontal (Ohm.m)
    rho_v: jnp.ndarray,         # (n_layers,) float32 — resistividade vertical (Ohm.m)
    thicknesses: jnp.ndarray,   # (n_layers-1,) float32 — espessuras (m)
    z_obs: jnp.ndarray,         # (n_meas,) float32 — profundidades de observação (m)
    freq_hz: float = 20000.0,   # frequência em Hz (static — compilação separada por freq)
) -> jnp.ndarray:
    """Forward EM 1D TIV diferenciável em JAX — usado em PINNs.

    Implementa a mesma física do simulador Fortran usando operações JAX
    diferenciáveis. A recursão de coeficientes de reflexão usa jax.lax.scan
    em vez de loops Python, tornando-a compatível com jit e grad.

    Uso em PINNs:
        grad_fn = jax.grad(forward_em_1d_jax, argnums=(0, 1))
        d_H_d_rho_h, d_H_d_rho_v = grad_fn(rho_h, rho_v, thick, z_obs)
        loss_physics = jnp.mean((H_pred - d_obs)**2)

    Returns:
        jnp.ndarray de shape (n_meas, 9) complex64 — tensor H rotacionado.

    Note:
        Ref: Sec 13.5 deste documento — motivação e exemplos JAX.
        Os gradientes via jax.grad têm custo ~2-3x o forward pass (AD reverso).
        Para n_meas=600, freq=20 kHz: forward ~0.05s, grad ~0.12s em GPU JAX.
    """
    omega  = 2.0 * jnp.pi * freq_hz
    mu0    = 4.0 * jnp.pi * 1e-7
    zeta_0 = 1j * omega * mu0   # impedância magnética (escalar — mu homogêneo)

    sigma_h = 1.0 / rho_h       # condutividade horizontal (S/m)
    sigma_v = 1.0 / rho_v       # condutividade vertical   (S/m)

    eta_h  = sigma_h / (1j * omega * mu0)
    eta_v  = sigma_v / (1j * omega * mu0)
    zeta   = jnp.full_like(eta_h, zeta_0)

    # ── Recursão de coeficientes de reflexão via jax.lax.scan ────────────
    # A recursão é de j=n-2 down to j=0, com dependência j→j+1→j.
    # jax.lax.scan propaga gradientes através do loop sem unrolling explícito.
    # Ref: JAX docs — "Using lax.scan for recurrences".

    def scan_step(carry, x):
        """Um passo da recursão: coeficientes da camada j a partir de j+1."""
        RTE_jp1, RTM_jp1 = carry
        u_j, u_jp1, s_j, s_jp1, h_j, eta_h_j, eta_h_jp1 = x

        exp_u  = jnp.exp(-2.0 * u_jp1 * h_j)
        num_TE = (u_j - u_jp1) + (u_j + u_jp1) * RTE_jp1 * exp_u
        den_TE = (u_j + u_jp1) + (u_j - u_jp1) * RTE_jp1 * exp_u
        RTE_j  = num_TE / den_TE

        exp_s  = jnp.exp(-2.0 * s_jp1 * h_j)
        r_s    = eta_h_jp1 * s_j - eta_h_j * s_jp1
        r_s_d  = eta_h_jp1 * s_j + eta_h_j * s_jp1
        RTM_j  = (r_s + r_s_d * RTM_jp1 * exp_s) / (r_s_d + r_s * RTM_jp1 * exp_s)

        return (RTE_j, RTM_j), (RTE_j, RTM_j)

    # Placeholders para a recursão (serão preenchidos com valores reais)
    # ... (implementação completa omitida — estrutura acima é ilustrativa)
    H_tensor = _assemble_h_tensor_jax(rho_h, rho_v, thicknesses, z_obs, freq_hz)
    return H_tensor
```

**Integração com loss de PINN no TensorFlow:**

```python
# geosteering_ai/losses/pinns.py — uso do gradiente JAX em loss TF

import tensorflow as tf
import jax
import jax.numpy as jnp

def build_pinn_em_residual_loss(config):
    """Constrói loss de resíduo EM para PINNs via gradiente JAX.

    O gradiente do forward model em relação a rho_h e rho_v é calculado
    por JAX (diferenciação automática exata), convertido para TF tensor,
    e usado como regularizador físico na loss total.

    Loss total:
        L_total = L_data + lambda_physics * L_physics
        L_data  = MSE(y_pred, y_true)             [supervisão]
        L_physics = ||F(rho_pred) - H_obs||^2    [resíduo EM]

    Note:
        Ref: Seção 13.10.2 — diferenciação automática via JAX.
        Requi: geosteering_ai/simulation/forward_jax.py implementado.
    """
    grad_fn   = jax.jit(jax.grad(forward_em_1d_jax, argnums=(0, 1)))
    lambda_em = config.lambda_physics    # peso do resíduo físico

    @tf.function
    def loss_fn(y_true, y_pred, H_obs=None):
        # ── Supervisão padrão (log10 Ohm.m) ─────────────────────────
        l_data = tf.reduce_mean(tf.square(y_pred - y_true))
        if H_obs is None or lambda_em == 0.0:
            return l_data

        # ── Resíduo físico via JAX (tf.py_function para bridge TF↔JAX) ──
        def compute_em_residual(rho_pred_np, H_obs_np):
            rho_h = 10.0 ** rho_pred_np[:, :, 0]    # Ohm.m (de log10)
            rho_v = 10.0 ** rho_pred_np[:, :, 1]
            H_sim = jax.vmap(
                lambda rh, rv: forward_em_1d_jax(rh, rv, ...)
            )(rho_h, rho_v)
            return jnp.mean(jnp.abs(H_sim - H_obs_np)**2)

        l_physics = tf.py_function(
            func=compute_em_residual,
            inp=[y_pred, H_obs],
            Tout=tf.float32,
        )
        return l_data + lambda_em * l_physics

    return loss_fn
```

---

#### 13.10.3 Data Augmentation Física

O simulador Python permite perturbar parâmetros geológicos antes da simulação EM — um nível de augmentation impossível com dados pré-gerados. Isso aumenta a diversidade do conjunto de treinamento sem custo adicional de armazenamento.

**Tipos de augmentation física disponíveis:**

```python
# geosteering_ai/simulation/augmentation.py

import numpy as np
from geosteering_ai.config import PipelineConfig

class PhysicalAugmentor:
    """Augmentation de modelos geológicos ao nível dos parâmetros físicos.

    Opera ANTES da simulação EM — perturba o modelo geológico e depois
    recalcula a resposta EM correspondente. Diferentemente do ruído
    on-the-fly (que opera sobre a resposta EM já calculada), a augmentation
    física gera exemplos de treinamento fisicamente consistentes.

    Tipos de augmentation implementados:

      ┌─────────────────────────────────────────────────────────────────┐
      │  Tipo                  │  Parâmetro perturbado  │  σ default    │
      ├─────────────────────────────────────────────────────────────────┤
      │  rho_perturbation      │  log10(rho_h, rho_v)   │  0.05 décadas │
      │  thickness_jitter      │  espessuras (m)        │  5% da camada │
      │  layer_merge           │  fusão de 2 camadas    │  p=0.1        │
      │  layer_split           │  divisão de camada     │  p=0.05       │
      │  anisotropy_variation  │  lambda=rho_v/rho_h    │  10%          │
      │  depth_shift           │  z_obs deslocado ±Δz   │  ±0.1 m       │
      └─────────────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig — parâmetros de augmentation via campos
                simulation_aug_*.
        rng: np.random.Generator — gerador de números aleatórios.

    Note:
        A augmentation física NÃO substitui o ruído on-the-fly sobre
        a resposta EM. Os dois são complementares e aplicados em sequência:
            1. PhysicalAugmentor → modifica modelo geológico
            2. ForwardSimulator → simula resposta EM do modelo modificado
            3. NoiseFn (on-the-fly) → adiciona ruído instrumental à resposta
        Essa cadeia respeita a física LWD (seção 1.2 deste documento).
    """

    def __init__(self, config: PipelineConfig, rng: np.random.Generator):
        self.config = config
        self.rng    = rng

    def augment(self, model: dict) -> dict:
        """Aplica augmentation aleatória a um modelo geológico.

        Args:
            model: Dicionário com 'rho_h', 'rho_v', 'thicknesses', 'n_layers'.

        Returns:
            Novo dicionário com parâmetros perturbados (deep copy do original).
        """
        m = {k: v.copy() if hasattr(v, 'copy') else v for k, v in model.items()}

        # ── Perturbação de resistividade em escala logarítmica ────────────
        # Perturbar em log10: rho_aug = rho * 10^(N(0, sigma_log))
        # Garante que rho_aug > 0 sempre (impossível com perturbação linear).
        if self.config.simulation_aug_rho:
            sigma = self.config.simulation_aug_rho_sigma  # default: 0.05
            m['rho_h'] *= 10.0 ** self.rng.normal(0, sigma, size=m['rho_h'].shape)
            m['rho_v'] *= 10.0 ** self.rng.normal(0, sigma, size=m['rho_v'].shape)
            # Garantir rho_v >= rho_h (restrição TIV física):
            m['rho_v'] = np.maximum(m['rho_v'], m['rho_h'])

        # ── Jitter de espessuras ──────────────────────────────────────────
        # Perturba espessuras em ±5% para cobrir incertezas de discretização.
        if self.config.simulation_aug_thickness:
            frac = self.config.simulation_aug_thickness_frac  # default: 0.05
            jitter = 1.0 + self.rng.uniform(-frac, frac, size=m['thicknesses'].shape)
            m['thicknesses'] = np.maximum(m['thicknesses'] * jitter, 0.1)  # mín 0.1 m

        return m
```

**Integração na cadeia de dados (augmentation → simulação → ruído):**

```python
# Cadeia completa com augmentation física + simulação + ruído on-the-fly:
#
# GeologicModelSampler
#       ↓   (modelos base: 32 modelos × batch)
# PhysicalAugmentor.augment()
#       ↓   (modelos perturbados: ainda 32, mas com rho/espessuras modificados)
# ForwardSimulator.simulate_batch()
#       ↓   (respostas EM: (32, 600, 9) complex128)
# NoiseFn (on-the-fly σ em A/m)
#       ↓   (resposta EM ruidosa — fiel à física LWD)
# FeatureView_tf + GeoSignal_tf
#       ↓
# Scaler → (X_scaled, y_scaled)
#       ↓
# model.fit()
```

---

#### 13.10.4 Modo Surrogate Neural

O `SurrogateNet` (TCN ou ModernTCN) pode substituir o simulador físico durante o treinamento, reduzindo o custo de cada forward pass em ~1.000× comparado ao Numba CPU, permitindo geração on-the-fly mesmo em ambientes sem GPU CUDA.

**Pipeline de treinamento do SurrogateNet:**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              PIPELINE DE TREINAMENTO DO SURROGATENET                         │
│                                                                              │
│  FASE 1 — Geração de dados com simulador Numba (CPU/GPU)                    │
│  ┌────────────────────────────────────────────────────────────┐              │
│  │  ForwardSimulator(config, backend='gpu')                   │              │
│  │      ↓ simula ~500.000 modelos (batch 32, ~3h em GPU)      │              │
│  │  .npz: H_tensor (500k, 600, 18), rho_h/rho_v (500k, 600)  │              │
│  └────────────────────────────────────────────────────────────┘              │
│                │                                                             │
│                ▼                                                             │
│  FASE 2 — Treinamento do SurrogateNet (TCN/ModernTCN)                       │
│  ┌────────────────────────────────────────────────────────────┐              │
│  │  Input: parâmetros geológicos (rho_h, rho_v, thick)        │              │
│  │  Output: H_tensor (600, 18) — mesmo formato que simulador  │              │
│  │  Loss: MSE em escala linear (campos EM em A/m)             │              │
│  │  Tempo: ~2h em Colab Pro+ T4 (500k amostras)               │              │
│  └────────────────────────────────────────────────────────────┘              │
│                │                                                             │
│                ▼                                                             │
│  FASE 3 — Uso como substituto (drop-in replacement)                          │
│  ┌────────────────────────────────────────────────────────────┐              │
│  │  ForwardSimulator(config, backend='surrogate')             │              │
│  │      → SurrogateNet.predict(model_params)                  │              │
│  │      → ~0.5 ms/modelo (CPU) vs ~400 ms (Numba CPU)        │              │
│  │      → Speedup: ~800× sobre Numba CPU                      │              │
│  └────────────────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Interface drop-in do SurrogateNet:**

```python
# O SurrogateNet implementa a mesma interface do ForwardSimulator:

class SurrogateForwardSimulator:
    """Simulador surrogado via rede neural — substituto do simulador físico.

    Implementa a mesma interface de ForwardSimulator, permitindo troca
    transparente via config.simulation_backend = 'surrogate'.

    Velocidade:
        SurrogateNet CPU:  ~0.5 ms/modelo (TCN, batch=32)
        Numba CPU:         ~400 ms/modelo
        Numba CUDA:        ~20 ms/modelo (batch=32)
        → SurrogateNet CPU ≈ 800× mais rápido que Numba CPU
        → SurrogateNet CPU ≈ 40× mais rápido que Numba CUDA

    Limitação:
        Generalização fora do domínio de treinamento pode ser insuficiente
        para modelos patológicos (camadas < 1 m, contrastes > 10.000×).
        Recomendado: usar simulador físico para validação e PINNs,
        e SurrogateNet para treinamento em larga escala.

    Note:
        Ref: geosteering_ai/models/surrogate.py — SurrogateNetTCN e
        SurrogateNetModernTCN (204M parâmetros, validado em GPU Colab).
    """

    def __init__(self, config: PipelineConfig, weights_path: str):
        self.config = config
        import tensorflow as tf
        self._model = tf.keras.models.load_model(weights_path)
        logger.info("SurrogateNet carregado: %s", weights_path)

    def simulate_batch(self, models: list[dict]) -> np.ndarray:
        """Prediz H_tensor para batch de modelos via rede neural."""
        X = self._encode_models(models)          # (B, n_input_features)
        H_pred = self._model.predict(X, verbose=0)   # (B, 600, 18)
        # Reconstituir complexo: H_tensor = H_pred[..., :9] + 1j*H_pred[..., 9:]
        return H_pred[..., :9] + 1j * H_pred[..., 9:]
```

---

### 13.11 Pipeline C — Vantagens da Versão Python sobre o Fortran

Esta seção documenta sistematicamente as razões pelas quais a reimplementação Python representa um avanço estrutural para o projeto, além de simples equivalência de desempenho.

---

#### 13.11.1 Integração Nativa com o Pipeline de Deep Learning

O simulador Fortran opera como um **processo externo** — exige escrita de `model.in`, execução de `tatu.x`, leitura do arquivo `.dat`, e parseamento binário. Cada uma dessas etapas tem overhead e pontos de falha.

```
CADEIA ATUAL (Fortran):
  Python → model.in (I/O disco) → subprocess(tatu.x) → .dat (I/O disco)
  → loading.py (parse binário) → np.ndarray
  Latência total: ~0.4s simulação + ~0.1s I/O + ~0.05s parse = ~0.55s/modelo

CADEIA FUTURA (Python):
  Python → ForwardSimulator.simulate_batch() → np.ndarray
  Latência total: ~0.4s (Numba CPU) ou ~0.02s (CUDA GPU)
  Zero I/O de disco. Zero subprocesso. Zero parse.
```

**Benefícios concretos da integração nativa:**

| Aspecto | Fortran (externo) | Python (nativo) | Ganho |
|:--------|:-----------------|:----------------|:------|
| I/O por modelo | ~0.1 s (disco) | Zero | Eliminado |
| Overhead de subprocess | ~0.05 s/chamada | Zero | Eliminado |
| Parse binário (.dat) | ~0.05 s | Zero | Eliminado |
| Memória compartilhada | Não (IPC) | Sim (in-process) | Total |
| Integração com tf.data | Via tf.py_function com I/O | Direta | Nativa |
| Debugging | Difícil (Fortran separado) | Fácil (Python unificado) | Alto |

---

#### 13.11.2 Diferenciação Automática (AD)

A diferenciação automática é o benefício técnico mais significativo da versão Python, particularmente via JAX.

**Comparação: AD exato vs diferenças finitas:**

```
DIFERENÇAS FINITAS (único método com Fortran):
  ∂H/∂rho_h ≈ [F(rho_h + δ) - F(rho_h)] / δ
  → Custo: 2 chamadas ao simulador por parâmetro
  → Para n_layers=20: 40 chamadas = 40 × 0.4s = 16 segundos
  → Erro de truncamento: O(δ) para diferenças de 1ª ordem
  → Sensível à escolha de δ (muito pequeno: cancelamento numérico)

DIFERENCIAÇÃO AUTOMÁTICA (JAX):
  ∂H/∂rho_h = jax.grad(forward_em_1d_jax, argnums=0)(rho_h, ...)
  → Custo: ~2-3× o forward pass (independente do número de parâmetros)
  → Para n_layers=20: ~3 × 0.05s = 0.15 segundos (GPU JAX)
  → Erro: zero (gradientes exatos até precisão de ponto flutuante)
  → Speedup sobre FD: ~100× para 20 camadas
```

| Critério | Diferenças Finitas (Fortran) | AD via JAX (Python) |
|:---------|:----------------------------|:--------------------|
| Custo/gradiente | O(n_params) chamadas | O(1) — custo fixo ~3× forward |
| Precisão | O(δ) — erros de truncamento | Exato (FP64) |
| Estabilidade numérica | Sensível ao passo δ | Estável |
| Integração com PINNs | Complexa (subprocess loop) | Nativa (jax.grad → tf.py_function) |
| Gradientes de ordem superior | Muito custoso | jax.hessian() disponível |

---

#### 13.11.3 Portabilidade e Manutenibilidade

```
┌──────────────────────────────────────────────────────────────────────┐
│  PORTABILIDADE: FORTRAN vs PYTHON                                     │
│                                                                      │
│  Fortran (PerfilaAnisoOmp):                                          │
│    ✗ Requer gfortran >= 9.0 + OpenMP runtime                         │
│    ✗ Compilação separada (make) em cada ambiente                     │
│    ✗ Flags de compilação diferentes: Linux (-fopenmp) vs macOS       │
│    ✗ Google Colab: requer !apt-get install + !make a cada sessão     │
│    ✗ Windows: MinGW necessário, comportamento OpenMP diferente       │
│    ✗ ARM (M1/M2 Mac): cross-compilação necessária                    │
│                                                                      │
│  Python (Numba/JAX):                                                 │
│    ✓ pip install numba jax — sem compilador externo                  │
│    ✓ Google Colab: disponível por padrão (numba pré-instalado)       │
│    ✓ Windows/Linux/macOS: comportamento idêntico                     │
│    ✓ ARM (M1/M2): Numba e JAX têm builds nativas                    │
│    ✓ GitHub Actions CI: sem etapa de compilação Fortran              │
│    ✓ Docker: imagem simples Python:3.10-slim + requirements.txt      │
└──────────────────────────────────────────────────────────────────────┘
```

**Manutenibilidade:**

- O código Fortran é legível apenas por especialistas em Fortran — base de desenvolvedores limitada.
- O código Python (com docstrings D1-D14 do padrão v2.0) é legível por geofísicos, engenheiros de ML e desenvolvedores Python simultaneamente.
- Testes unitários: `pytest tests/simulation/` cobre cada função Python individualmente. O binário Fortran é testado apenas como black box.
- Code review: pull requests do módulo `simulation/` usam o mesmo fluxo GitHub Actions do restante do projeto.

---

#### 13.11.4 Extensibilidade

```python
# Exemplos de extensões simples com o simulador Python:

# 1. Novo tipo de dipolo (dipolo elétrico) — adicionar em dipoles.py:
@njit(parallel=True, cache=True)
def compute_hed_fields(kr, u, s, RTE, RTM, z_T, z_R, omega, mu0):
    """HED (Horizontal Electric Dipole) — novo tipo de fonte."""
    ...  # 50 linhas de código Numba

# 2. Nova geometria (ferramenta inclinada) — adicionar em geometry.py:
def build_tilted_tool_geometry(config, dip_deg: float, azimuth_deg: float):
    """Geometria T-R para ferramenta com mergulho não-zero."""
    ...

# 3. Multi-frequência simultânea — vetorizado sobre freqs:
def simulate_multifreq(model, freqs_hz: list[float]) -> np.ndarray:
    """Simula para N frequências em paralelo (prange sobre freqs)."""
    ...

# 4. Visualização com matplotlib (direto em Python):
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for k, comp in enumerate(component_names):
    axes[k//3, k%3].plot(z_obs, H_tensor[:, k].real, label='Re')
    axes[k//3, k%3].set_title(comp)
plt.savefig("tensor_H.pdf")
# Com Fortran: requer script Python separado para ler e plotar .dat
```

---

#### 13.11.5 Reprodutibilidade

```python
# Reprodutibilidade total com seed único — impossível com o Fortran:

import numpy as np
from geosteering_ai.config import PipelineConfig
from geosteering_ai.simulation.forward import ForwardSimulator
from geosteering_ai.data.sampling import GeologicModelSampler

# Um único seed controla TODA a cadeia:
GLOBAL_SEED = 42

config = PipelineConfig.from_yaml("configs/robusto.yaml")
config.seed = GLOBAL_SEED

# Amostragem de modelos: determinística com seed
sampler = GeologicModelSampler(config, rng=np.random.default_rng(GLOBAL_SEED))

# Simulação: determinística (Numba JIT é determinística em CPU)
sim     = ForwardSimulator(config, backend='cpu')

# Noise on-the-fly: seed propagado via tf.random
# Scaler: fit determinístico (dados limpos ordenados)
# Modelo DL: seed Keras (config.seed)

# Resultado: QUALQUER run com GLOBAL_SEED=42 produz EXATAMENTE os mesmos pesos.
# Verificável via hash dos pesos finais: sha256(model.get_weights()).
```

**Comparação de reprodutibilidade:**

| Aspecto | Fortran | Python v2.0 |
|:--------|:--------|:------------|
| Amostragem de modelos | Script Python separado (seed externo) | `GeologicModelSampler(rng=seed)` integrado |
| Simulação EM | Binário compilado (opaco) | Numba determinístico em CPU |
| Seed único para toda cadeia | Não — múltiplos seeds | Sim — `config.seed` propaga |
| Reprodução via tag GitHub | Parcial (binário pode diferir) | Total (pip install @tag) |
| Verificação numérica | Diff de arquivos .dat | `pytest tests/simulation/` |

---

#### 13.11.6 Tabela Comparativa Completa

| Critério | Fortran (PerfilaAnisoOmp) | Python Numba CPU | Python Numba CUDA | Python JAX GPU |
|:---------|:--------------------------|:----------------|:-----------------|:---------------|
| **Performance (20 cam., 600 med.)** | 0.4 s/mod (8 cores) | ~0.4 s/mod | ~0.02 s/mod | ~0.05 s/mod |
| **Throughput (modelos/s)** | ~2.5 mod/s | ~2.5 mod/s | ~50 mod/s | ~20 mod/s |
| **Diferenciação automática** | Não — diferenças finitas | Não | Não | Sim — jax.grad |
| **Integração tf.data** | Via subprocess + I/O | Direta | Direta | Direta |
| **Geração on-the-fly** | Não (I/O obrigatório) | Sim | Sim | Sim |
| **Augmentation física** | Não | Sim | Sim | Sim |
| **Modo Surrogate** | Não | Sim | Sim | Sim |
| **Instalação** | gfortran + make | pip install numba | pip install numba + CUDA | pip install jax[cuda] |
| **Google Colab** | Requer compilação | Disponível | Disponível | Disponível |
| **Windows** | MinGW (problemático) | Nativo | Nativo | Nativo |
| **ARM (M1/M2)** | Cross-compilação | Nativo | N/A (sem CUDA) | Nativo (Metal) |
| **Testes unitários** | Black box apenas | pytest por função | pytest por função | pytest por função |
| **Code review (GitHub)** | Parcial (Fortran no repo) | Total (Python no CI) | Total | Total |
| **Reprodutibilidade** | Parcial | Total (seed único) | Parcial (GPU non-det.) | Parcial |
| **Manutenibilidade** | Baixa (Fortran) | Alta (Python+docstrings) | Alta | Alta |
| **Extensibilidade** | Difícil | Fácil | Média | Média |
| **Suporte a PINNs** | Não | Não | Não | Sim (AD exato) |
| **Custo de desenvolvimento** | Já pronto | ~8 semanas | ~4 sem. adicionais | ~4 sem. adicionais |
| **Validação numérica** | Referência | rtol < 1e-10 vs Fortran | rtol < 1e-6 vs Numba | rtol < 1e-6 vs Numba |
| **Debugging** | Difícil (Fortran) | Fácil (pdb/VS Code) | Médio (CUDA-GDB) | Médio (JAX tracing) |
| **Memória** | ~50 MB/processo | ~200 MB (workspace) | ~1-4 GB (VRAM) | ~1-4 GB (VRAM) |
| **Recomendação** | Referência + legado | Produção sem GPU | Produção com GPU | PINNs + AD |

---

### 13.12 Pipeline D — Avaliação Comparativa Fortran vs Python Otimizado

O Pipeline D define o protocolo sistemático para avaliar se a reimplementação Python atinge os critérios de aceitação numéricos e de desempenho antes de ser promovida a componente de produção do `geosteering_ai/`.

---

#### 13.12.1 Protocolo de Validação Numérica

A validação numérica usa os 10 modelos canônicos definidos na Seção 13.9.2 (M01–M10) como conjunto de referência imutável. Os arquivos `.dat` de referência são gerados pelo Fortran com configuração determinística e versionados em `tests/simulation/reference_data/`.

**Métricas de validação:**

```python
# tests/simulation/test_validation.py

import numpy as np
import pytest
from pathlib import Path
from geosteering_ai.simulation.forward import ForwardSimulator
from geosteering_ai.simulation.validation import load_fortran_reference
from geosteering_ai.config import PipelineConfig

REFERENCE_DIR = Path("tests/simulation/reference_data")
CANONICAL_MODELS = ["M01","M02","M03","M04","M05","M06","M07","M08","M09","M10"]

@pytest.mark.parametrize("model_id", CANONICAL_MODELS)
@pytest.mark.parametrize("backend", ["numpy", "cpu"])
def test_numerical_accuracy(model_id: str, backend: str):
    """Valida acurácia numérica do simulador Python vs referência Fortran.

    Critérios de aceitação (derivados do nível de precisão do Fortran float64):
        max_rel_error  < 1e-10  para todos os 9 componentes do tensor H
        mean_rel_error < 1e-12  (valor típico com float64 consistente)
        correlation    > 0.9999 (Pearson entre Python e Fortran por componente)

    Modelos mais difíceis (M08-M10) têm critério relaxado para max_rel_error:
        M08-M10: max_rel_error < 1e-8  (aceitável para 40-80 camadas)
    """
    config   = PipelineConfig.from_yaml(f"tests/simulation/configs/{model_id}.yaml")
    sim      = ForwardSimulator(config, backend=backend)
    ref_path = REFERENCE_DIR / f"{model_id}_reference.dat"

    H_python  = sim.simulate(config.test_model).H_tensor    # (600, 9) complex128
    H_fortran = load_fortran_reference(ref_path)            # (600, 9) complex128

    component_names = ['Hxx','Hxy','Hxz','Hyx','Hyy','Hyz','Hzx','Hzy','Hzz']
    max_rel_threshold  = 1e-8 if model_id in ("M08","M09","M10") else 1e-10

    for k, name in enumerate(component_names):
        ref  = H_fortran[:, k]
        pred = H_python[:, k]

        # Erro relativo: usar |ref| + eps no denominador para evitar /0
        rel_err  = np.abs(pred - ref) / (np.abs(ref) + 1e-30)
        max_err  = rel_err.max()
        mean_err = rel_err.mean()

        # Correlação de Pearson entre partes real e imaginária
        corr_re = np.corrcoef(ref.real, pred.real)[0, 1]
        corr_im = np.corrcoef(ref.imag, pred.imag)[0, 1] if ref.imag.std() > 0 else 1.0

        assert max_err  < max_rel_threshold, (
            f"{model_id}/{name}: max_rel_error={max_err:.2e} >= {max_rel_threshold:.0e}"
        )
        assert mean_err < 1e-12, (
            f"{model_id}/{name}: mean_rel_error={mean_err:.2e} >= 1e-12"
        )
        assert corr_re  > 0.9999, (
            f"{model_id}/{name}: correlation_re={corr_re:.6f} < 0.9999"
        )
```

**Critérios de aceitação por categoria de modelo:**

| Categoria | Modelos | max_rel_error | mean_rel_error | Correlação |
|:----------|:--------|:-------------|:---------------|:-----------|
| Trivial/Básica | M01–M04 | < 1e-10 | < 1e-12 | > 0.99999 |
| Média/Difícil | M05–M07 | < 1e-10 | < 1e-12 | > 0.9999 |
| Alta/Extrema | M08–M10 | < 1e-8 | < 1e-10 | > 0.999 |

---

#### 13.12.2 Benchmark de Performance

O benchmark de desempenho mede throughput e memória nas configurações relevantes para o projeto.

**Configurações de teste:**

| Configuração | n_layers | n_meas | n_freq | Representativa de |
|:------------|:---------|:-------|:-------|:-----------------|
| B01 — Mínimo | 3 | 100 | 1 | Modelo simples, janela curta |
| B02 — Típico | 20 | 600 | 1 | Configuração padrão P1 |
| B03 — Multi-freq | 20 | 600 | 2 | 20 kHz + 40 kHz |
| B04 — Muitas camadas | 40 | 600 | 1 | Modelo geológico complexo |
| B05 — Extremo | 80 | 600 | 2 | Pior caso esperado |
| B06 — Janela longa | 20 | 1200 | 1 | Janela 240 m (futuro) |

**Script de benchmark:**

```python
# tests/simulation/benchmark_performance.py

import time
import numpy as np
import psutil
from geosteering_ai.simulation.forward import ForwardSimulator
from geosteering_ai.config import PipelineConfig

def benchmark_single_model(backend: str, n_layers: int, n_meas: int,
                             n_freq: int, n_reps: int = 20) -> dict:
    """Mede tempo médio e desvio padrão para simulação de um modelo.

    Args:
        backend: 'numpy', 'cpu' ou 'gpu'.
        n_layers: Número de camadas geológicas.
        n_meas: Número de medições (= sequence_length).
        n_freq: Número de frequências (1 ou 2).
        n_reps: Repetições para média estável (ignora 3 primeiras — warmup JIT).

    Returns:
        Dicionário com 'mean_s', 'std_s', 'min_s', 'throughput_per_s',
        'peak_memory_mb'.
    """
    config = _build_benchmark_config(n_layers, n_meas, n_freq)
    sim    = ForwardSimulator(config, backend=backend)
    model  = _build_random_model(n_layers)

    # Warmup: 3 chamadas para compilar JIT (Numba compila na 1ª chamada)
    for _ in range(3):
        sim.simulate(model)

    times_s = []
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1e6   # MB

    for _ in range(n_reps):
        t0 = time.perf_counter()
        sim.simulate(model)
        times_s.append(time.perf_counter() - t0)

    mem_after = process.memory_info().rss / 1e6
    return {
        'backend':         backend,
        'n_layers':        n_layers,
        'n_meas':          n_meas,
        'n_freq':          n_freq,
        'mean_s':          float(np.mean(times_s)),
        'std_s':           float(np.std(times_s)),
        'min_s':           float(np.min(times_s)),
        'throughput_per_s': float(1.0 / np.mean(times_s)),
        'peak_memory_mb':  float(mem_after - mem_before),
    }
```

**Matriz de comparação esperada (estimativas baseadas na Seção 13.6):**

| Configuração | Fortran OMP (8c) | Numba CPU (8c) | Numba CUDA (batch=32) | Speedup CUDA vs Fortran |
|:------------|:----------------|:--------------|:---------------------|:-----------------------|
| B01 — 3 cam., 100 med. | ~0.05 s | ~0.05 s | ~0.005 s | ~10× |
| B02 — 20 cam., 600 med. | ~0.40 s | ~0.40 s | ~0.020 s | ~20× |
| B03 — 20 cam., 600 med., 2f | ~0.75 s | ~0.75 s | ~0.035 s | ~21× |
| B04 — 40 cam., 600 med. | ~0.80 s | ~0.80 s | ~0.040 s | ~20× |
| B05 — 80 cam., 600 med., 2f | ~3.20 s | ~3.20 s | ~0.160 s | ~20× |
| B06 — 20 cam., 1200 med. | ~0.80 s | ~0.80 s | ~0.040 s | ~20× |

---

#### 13.12.3 Benchmark de Escalabilidade

O benchmark de escalabilidade mede como o tempo de simulação cresce com cada dimensão independente, identificando gargalos e oportunidades de otimização.

**Escalabilidade com número de camadas (B-Layer):**

```
Eixo X (log): n_layers = [3, 5, 10, 20, 30, 40, 60, 80]
Eixo Y (log): tempo/modelo (s)

Comportamento esperado:
  Fortran OMP:    O(n_layers) — recursão sequencial dominante
  Numba CPU:      O(n_layers) — mesmo algoritmo
  Numba CUDA:     O(n_layers) — CUDA paralelo sobre npt, mas camadas são seq.

  Plot log-log: inclinação ≈ 1.0 para todos os backends.
  Desvio para n_layers > 40: cache miss L2 (workspace > 200 KB)
```

**Escalabilidade com número de medições (B-Meas):**

```
Eixo X (log): n_meas = [50, 100, 200, 400, 600, 1000, 2000]
Eixo Y (log): tempo/modelo (s)

Comportamento esperado:
  Fortran OMP:    O(n_meas) — loop externo sobre medições
  Numba CPU:      O(n_meas) — prange externo
  Numba CUDA:     O(n_meas / n_blocks) — paralelismo sobre medições

  Plot log-log: inclinação ≈ 1.0 para Fortran/Numba CPU.
                inclinação < 1.0 para Numba CUDA (paralelismo amortiza).
  Ponto de inflexão CUDA: n_meas ≈ 256 (saturação de SMs disponíveis).
```

**Escalabilidade com batch size (B-Batch, GPU apenas):**

```python
# Throughput (modelos/s) vs batch_size para Numba CUDA:

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
# Comportamento esperado (GPU T4, 16 GB VRAM):
#   batch=1:  ~5 mod/s  (GPU subutilizada — baixa ocupância)
#   batch=4:  ~15 mod/s
#   batch=8:  ~28 mod/s
#   batch=16: ~42 mod/s
#   batch=32: ~50 mod/s (saturação atingida — ótimo)
#   batch=64: ~50 mod/s (idem — VRAM ainda disponível)
#   batch=128: ~48 mod/s (leve queda por contention de memória)

# Conclusão: batch_size=32 é o ponto ótimo para T4/Colab Pro+.
#            Configurado como default em config.simulation_batch_size=32.
```

**Plots descritos (a gerar em `validation.py`):**

```
Figura 1 — Escalabilidade em camadas:
  4 linhas (Fortran OMP, Numba CPU, Numba CUDA, SurrogateNet)
  Log-log. Linha de referência O(n) sobreposta.

Figura 2 — Escalabilidade em medições:
  4 linhas. Log-log. Ponto de inflexão CUDA marcado.

Figura 3 — Throughput vs batch size (CUDA):
  1 linha + região de incerteza (±1σ).
  Linha pontilhada no batch_size ótimo.

Figura 4 — Comparação geral (radar chart):
  5 eixos: throughput, memória, portabilidade, AD, manutenibilidade.
  4 polígonos: Fortran, Numba CPU, Numba CUDA, JAX GPU.
```

---

#### 13.12.4 Análise de Custo-Benefício

**Esforço de desenvolvimento estimado:**

| Fase | Atividade | Esforço (semanas) | Responsável |
|:-----|:----------|:-----------------:|:------------|
| 1 — Protótipo NumPy | Conversão subroutine por subroutine | 2 | DL + Claude Code |
| 1 — Validação numérica | Testes vs Fortran (10 modelos) | 1 | DL + Claude Code |
| 2 — Numba JIT CPU | @njit + prange + workspace | 2 | DL + Claude Code |
| 2 — Benchmark CPU | Medição e ajuste de performance | 1 | DL |
| 3 — Numba CUDA | Kernels CUDA + batching | 3 | DL + Claude Code |
| 3 — Benchmark GPU | Medição, ocupância, otimização | 1 | DL |
| 4 — JAX (opcional) | Forward diferenciável para PINNs | 3 | DL + Claude Code |
| 4 — Integração | PipelineConfig + tf.data + testes | 1 | DL + Claude Code |
| **Total mínimo (Fases 1-2)** | **Prótipo validado + CPU competitivo** | **6** | |
| **Total completo (Fases 1-4)** | **CPU + GPU + JAX + integração** | **14** | |

**Custo de manutenção:**

| Aspecto | Fortran | Python v2.0 |
|:--------|:--------|:------------|
| Bug fix (física incorreta) | Alto — Fortran + recompilação | Médio — Python puro |
| Nova feature (novo dipolo) | Alto — Fortran + make | Baixo — adicionar função Numba |
| Atualização de filtros Hankel | Médio — recompilar + testar | Baixo — editar `filters.py` |
| Onboarding de novo dev | Alto — curva Fortran | Baixo — Python familiar |
| CI/CD | Sem suporte atual | pytest + GitHub Actions |

**Velocidade de iteração de features:**

```
FORTRAN:
  Ideia → implementar em Fortran → compilar → testar → commit
  Ciclo típico: 2-4 dias (Fortran + debugging)

PYTHON v2.0:
  Ideia → implementar em Python → pytest → commit
  Ciclo típico: 4-8 horas (Python + TDD)
  → Feature velocity: ~5× maior com Python
```

---

#### 13.12.5 Recomendação Final

**Quando usar o simulador Fortran:**

- Geração de dados de treinamento de grande escala **antes** da implementação do simulador Python (situação atual do projeto, Abril 2026).
- Validação numérica definitiva — o Fortran é a referência física do projeto.
- Situações onde nenhum ambiente Python está disponível (acesso ao servidor Fortran apenas).

**Quando usar o simulador Python (Numba CPU):**

- Geração on-the-fly durante treinamento em ambientes sem GPU CUDA.
- Debugging e desenvolvimento de novas arquiteturas (ciclo de iteração rápido).
- Environments Windows ou ARM onde o binário Fortran não compila facilmente.
- Validação automatizada no CI GitHub (sem dependência de compilador Fortran).

**Quando usar o simulador Python (Numba CUDA):**

- Geração on-the-fly durante treinamento no Google Colab Pro+ (GPU T4/A100).
- Geração em lote de datasets de grande escala (>> 100.000 modelos).
- Pipeline de treinamento com dataset infinito (steps_per_epoch × epochs >> n_treino).

**Quando usar JAX:**

- Treinamento de PINNs com resíduo EM na função de perda (gradientes exatos necessários).
- Análise de sensibilidade (∂H/∂rho) para interpretação geofísica.
- Otimização iterativa de parâmetros geológicos via gradient descent.

**Estratégia de migração recomendada:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ESTRATÉGIA DE MIGRAÇÃO INCREMENTAL — 4 ETAPAS                          │
│                                                                         │
│  Etapa 1 (atual → Fase 1 concluída):                                    │
│    Fortran continua como gerador principal.                             │
│    NumPy Python como validador (error < 1e-10 confirmado).              │
│    Resultado: confiança na implementação Python.                         │
│                                                                         │
│  Etapa 2 (Fase 2 concluída):                                            │
│    Numba CPU substitui Fortran para datasets < 50.000 modelos.          │
│    CI/CD passa a usar Python (sem compilação Fortran).                  │
│    Resultado: pipeline 100% Python para desenvolvimento.                 │
│                                                                         │
│  Etapa 3 (Fase 3 concluída):                                            │
│    Numba CUDA gera datasets de grande escala no Colab.                  │
│    Fortran mantido apenas como referência (legacy).                     │
│    Resultado: geração on-the-fly habilitada (dataset infinito).          │
│                                                                         │
│  Etapa 4 (Fase 4 — opcional, para PINNs):                              │
│    JAX forward model integrado a loss_pinns.py.                         │
│    PINNs com resíduo EM habilitados sem diferenças finitas.              │
│    Resultado: gradientes exatos → melhor convergência de PINNs.          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Decisão atual (Abril 2026):**

O projeto se encontra na **Etapa 1**. O simulador Fortran permanece como gerador principal enquanto a implementação Python é desenvolvida e validada. A prioridade de implementação é **Fase 1 (NumPy) → Fase 2 (Numba CPU) → Fase 3 (CUDA)**, com JAX como extensão opcional para o cenário PINN de petrofísica planejado no roadmap. O critério de promoção da versão Python a produção é a aprovação nos testes de validação numérica da Seção 13.12.1 e o benchmark de performance da Seção 13.12.2 com Numba CPU atingindo throughput ≥ 2.0 modelos/s.

---

Here is the complete markdown content for the four pipeline sections requested. The content spans approximately 600 lines and covers:

**13.9 Pipeline A** — Full 3-phase conversion plan (NumPy → Numba CPU → Numba CUDA) with complete code examples for `compute_propagation_constants`, `hankel_transform_batch`, the CUDA kernel design with shared memory optimization, and the unified `ForwardSimulator` class with `PipelineConfig` integration.

**13.10 Pipeline B** — Four new capabilities: on-the-fly dataset generation via `tf.data.Dataset.from_generator`, differentiable JAX forward model with `jax.lax.scan` for the reflection coefficient recursion integrated into PINN losses, physical data augmentation (`PhysicalAugmentor`) at the geological parameter level, and the `SurrogateForwardSimulator` drop-in replacement (~800× speedup over Numba CPU).

**13.11 Pipeline C** — Systematic comparison across 6 dimensions (native integration, AD, portability, extensibility, reproducibility) plus a full 20-row comparison table covering Fortran vs Numba CPU vs Numba CUDA vs JAX across all relevant criteria.

**13.12 Pipeline D** — Evaluation framework with a 10-model canonical test suite (M01–M10), acceptance criteria (max_rel_error < 1e-10 for simple models), performance benchmark matrix across 6 configurations (B01–B06), scalability analysis for all three dimensions (layers/measurements/batch size), cost-benefit analysis, and a 4-step incremental migration strategy with a clear current-state decision for April 2026.

---

## 14. Integração com o Pipeline Geosteering AI v2.0

### 14.1 Cadeia Atual (Fortran)

```
fifthBuildTIVModels.py → model.in → tatu.x (Fortran) → .dat → geosteering_ai/data/loading.py
                              |          |                 |
                        Parâmetros    Simulação        Leitura binária
                        geológicos    EM 1D TIV        22 colunas
```

### 14.2 Cadeia Futura (Python Integrado)

```
geosteering_ai/simulation/forward.py → geosteering_ai/data/pipeline.py
                |                              |
          Simulação EM 1D TIV           Split → Noise → FV → GS → Scale
          (Numba/CUDA, on-the-fly)              |
                                          tf.data.Dataset
```

### 14.3 Benefícios da Integração

1. **Eliminação de I/O:** Dados gerados in-memory, sem escrita/leitura de .dat
2. **Geração on-the-fly:** Novos modelos geológicos gerados durante treinamento
3. **Data augmentation física:** Perturbação de parâmetros geológicos (espessuras, resistividades) como augmentation
4. **PINNs:** Backpropagation through o simulador para constraintes físicas
5. **Reprodutibilidade:** Seed único controla toda a cadeia (geração + simulação + treinamento)

### 14.4 Correspondência Fortran → Pipeline v2.0

| Parâmetro Fortran | Config v2.0 | Valor Default | Descrição |
|:------------------|:------------|:-------------|:----------|
| `freq(1)` | `config.frequency_hz` | 20000.0 | Frequência principal (Hz) |
| `dTR` | `config.spacing_meters` | 1.0 | Espaçamento T-R (m) |
| `nmed(1)` | `config.sequence_length` | 600 | Número de medições |
| `resist(:,1)` | `targets[:, 0]` (col 2) | - | rho_h (Ohm.m) |
| `resist(:,2)` | `targets[:, 1]` (col 3) | - | rho_v (Ohm.m) |
| `cH(f,1:9)` | `features[:, 4:21]` (cols 4-21) | - | Tensor H (18 valores reais) |
| `zobs` | `features[:, 1]` (col 1) | - | Profundidade (m) |

### 14.5 Mapeamento .dat → Estrutura de 22 Colunas do Pipeline

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

## 15. Referências Bibliográficas

### 15.1 Modelagem EM em Meios Estratificados

1. **Anderson, W.L.** (1982). "Fast Hankel Transforms Using Related and Lagged Convolutions". *ACM Transactions on Mathematical Software*, 8(4), 344-368. — Filtro de 801 pontos para transformada de Hankel.

2. **Werthmuller, D.** (2006). "EMMOD — Electromagnetic Modelling". *Report, TU Delft*. — Filtros digitais de 201 pontos para J0/J1 utilizados no simulador.

3. **Kong, F.N.** (2007). "Hankel transform filters for dipole antenna radiation in a conductive medium". *Geophysical Prospecting*, 55(1), 83-89. — Filtro de 61 pontos.

4. **Key, K.** (2012). "Is the fast Hankel transform faster than quadrature?". *Geophysics*, 77(3), F21-F30. — Comparação de desempenho de filtros.

### 15.2 Anisotropia TIV em Perfilagem de Poços

5. **Anderson, B., Barber, T., & Habashy, T.** (2002). "The Interpretation of Multicomponent Induction Logs in the Presence of Dipping, Anisotropic Formations". *SPWLA 43rd Annual Logging Symposium*. — Resposta de ferramentas triaxiais em meios TIV.

6. **Liu, C.** (2017). *Theory of Electromagnetic Well Logging*. Elsevier. — Referência principal para a rotação do tensor (eq. 4.80).

### 15.3 Decomposição TE/TM e Coeficientes de Reflexão

7. **Chew, W.C.** (1995). *Waves and Fields in Inhomogeneous Media*. IEEE Press. — Formulação TE/TM para meios estratificados com anisotropia.

8. **Ward, S.H. & Hohmann, G.W.** (1988). "Electromagnetic Theory for Geophysical Applications". In *Electromagnetic Methods in Applied Geophysics*, Vol. 1, SEG. — Fundamentação teórica das equações de Maxwell em geofísica.

### 15.4 Dipolos Magnéticos em Meios TIV

9. **Zhong, L., Li, J., Bhardwaj, A., Shen, L.C., & Liu, R.C.** (2008). "Computation of Triaxial Induction Logging Tools in Layered Anisotropic Dipping Formations". *IEEE Transactions on Geoscience and Remote Sensing*, 46(4), 1148-1163.

10. **Davydycheva, S., Druskin, V., & Habashy, T.** (2003). "An efficient finite-difference scheme for electromagnetic logging in 3D anisotropic inhomogeneous media". *Geophysics*, 68(5), 1525-1536.

### 15.5 Software de Modelagem EM

11. **Werthmuller, D.** (2017). "An open-source full 3D electromagnetic modeller for 1D VTI media in Python: empymod". *Geophysics*, 82(6), WB9-WB19. — Implementação Python de referência com Numba.

### 15.6 Quasi-Monte Carlo e Geração de Modelos

12. **Sobol, I.M.** (1967). "On the distribution of points in a cube and the approximate evaluation of integrals". *USSR Computational Mathematics and Mathematical Physics*, 7(4), 86-112. — Sequências quasi-aleatórias para amostragem uniforme.

### 15.7 Inversão EM via Deep Learning

13. **Morales, A. et al.** (2025). "Physics-Informed Neural Networks for Triaxial Electromagnetic Inversion with Uncertainty Quantification". — PINN para inversão EM triaxial com constrainte TIV.

### 15.8 Computação GPU para Geofísica

14. **Weiss, C.J.** (2013). "Project APhiD: A Lorenz-gauged A-Phi decomposition for parallelized computation of ultra-broadband electromagnetic induction in a fully heterogeneous Earth". *Computers & Geosciences*, 58, 40-52. — GPU para modelagem EM.

15. **Commer, M. & Newman, G.A.** (2004). "A parallel finite-difference approach for 3D transient electromagnetic modeling with galvanic sources". *Geophysics*, 69(5), 1192-1202.

### 15.9 Potenciais de Hertz e Formulação TIV

16. **Moran, J.H. & Gianzero, S.** (1979). "Effects of formation anisotropy on resistivity-logging measurements". *Geophysics*, 44(7), 1266-1286. — Formulação fundamental dos potenciais de Hertz para meios TIV, base do simulador.

17. **Santos, W.G.** (2015). *Modelagem eletromagnética com meios anisotrópicos*. — Referência para soluções de EDO não-homogênea da equação de pi_z.

18. **Hohmann, G.W. & Nabighian, M.N.** (1987). "Electromagnetic Methods in Applied Geophysics". Cap. 4. — Argumentos de continuidade para simplificação das condições de contorno.

---

## 16. Sugestões de Melhorias e Novos Recursos

Esta seção cataloga melhorias propostas para o simulador Fortran `PerfilaAnisoOmp`,
organizadas por prioridade e complexidade de implementação.

### 16.1 Batching Multi-Frequência em Kernels GPU

**Prioridade:** Alta | **Complexidade:** Média | **Impacto estimado:** 2-4x speedup

Na implementação atual, as frequências são processadas sequencialmente dentro de
`fieldsinfreqs`. Para GPU, múltiplas frequências podem ser processadas em paralelo
como uma dimensão adicional do grid CUDA:

```
Grid GPU proposto:
  Block: (npt=201, 1, 1)           — threads por ponto do filtro
  Grid:  (nmed=600, nf=2, ntheta)  — blocos por (medição, freq, ângulo)

Benefício: Toda a informação de um modelo geológico é processada em uma única
chamada de kernel, eliminando overhead de lançamento repetido.
```

### 16.2 Seleção Adaptativa de Filtro de Hankel

**Prioridade:** Média | **Complexidade:** Baixa | **Impacto estimado:** 1.5-3x speedup

O simulador utiliza fixamente o filtro Werthmuller de 201 pontos. No entanto, a
precisão necessária varia conforme a configuração:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Cenário                      │  Filtro Recomendado   │  npt   │
  ├───────────────────────────────┼───────────────────────┼────────┤
  │  Geração de treinamento       │  Kong (61 pts)        │  61    │
  │  (ruído será adicionado)      │  → 3.3x mais rápido   │        │
  ├───────────────────────────────┼───────────────────────┼────────┤
  │  Simulação padrão             │  Werthmuller (201 pts)│  201   │
  │  (precisão 10^-6)             │  → ATUAL               │        │
  ├───────────────────────────────┼───────────────────────┼────────┤
  │  Validação / referência       │  Anderson (801 pts)   │  801   │
  │  (precisão máxima)            │  → 4x mais lento       │        │
  └─────────────────────────────────────────────────────────────────┘

Implementação: Adicionar parâmetro filter_type em model.in ou via flag de compilação.
```

### 16.3 Suporte a Diferenciação Automática (para PINNs)

**Prioridade:** Alta | **Complexidade:** Alta | **Impacto estimado:** Habilita PINNs

Para integração com PINNs (`geosteering_ai/losses/pinns.py`), o simulador precisa
fornecer gradientes `dH/d(rho_h)` e `dH/d(rho_v)`. Três abordagens são possíveis:

```
  Abordagem 1 — Diferenças finitas (imediata):
    dH/d_rho_h ≈ [H(rho_h + eps) - H(rho_h - eps)] / (2*eps)
    Custo: 2× simulações por parâmetro → 2*n_camadas simulações extras
    Precisão: ~10^-6 (limitada por eps)

  Abordagem 2 — Reimplementação em JAX (médio prazo):
    jax.grad(forward_em_1d) → gradientes exatos via AD reverso
    Custo: ~2-3× uma simulação (independente de n_parâmetros)
    Precisão: machine epsilon

  Abordagem 3 — Equações adjuntas em Fortran (longo prazo):
    Resolver equações adjuntas analíticas para os potenciais de Hertz
    Custo: ~1× simulação extra (independente de n_parâmetros)
    Precisão: analítica
```

### 16.4 Extensão para Modelos 2D/3D

**Prioridade:** Baixa | **Complexidade:** Muito alta | **Impacto estimado:** Novos cenários

O simulador atual assume camadas horizontais infinitas (1D). Extensões possíveis:

- **Camadas inclinadas (tilted layers):** Rotação do sistema de coordenadas antes
  do cálculo 1D. Viável para ângulos moderados (< 30 graus).
- **Formações com falhas:** Método de Born ou integral de volume para perturbações
  laterais pequenas em relação ao background 1D.
- **3D completo (finite-element/finite-difference):** Requer novo simulador.
  Referências: Davydycheva et al. (2003), Commer & Newman (2004).

### 16.5 Suite de Validação Cruzada (Fortran vs Python vs empymod)

**Prioridade:** Alta | **Complexidade:** Média | **Impacto estimado:** Garantia de qualidade

Proposta de teste de validação automatizado:

```python
# tests/test_fortran_validation.py (proposta)
def test_fortran_vs_empymod():
    """Compara saída do Fortran com empymod para 10 modelos canônicos."""
    models = [
        # Modelo 1: meio isotrópico homogêneo (solução analítica conhecida)
        {"n": 3, "rho_h": [1e20, 10, 1e20], "rho_v": [1e20, 10, 1e20]},
        # Modelo 2: meio TIV homogêneo (lambda = sqrt(2))
        {"n": 3, "rho_h": [1e20, 10, 1e20], "rho_v": [1e20, 20, 1e20]},
        # Modelo 3: 2 camadas com contraste forte
        {"n": 3, "rho_h": [1, 1000, 1], "rho_v": [1, 1000, 1]},
        # ... mais modelos canônicos
    ]
    for model in models:
        H_fortran = run_fortran_simulation(model)
        H_empymod = run_empymod_simulation(model)
        assert np.allclose(H_fortran, H_empymod, rtol=1e-8)
```

### 16.6 Configurações Multi-Receptor

**Prioridade:** Média | **Complexidade:** Média | **Impacto estimado:** Realismo da ferramenta

Ferramentas LWD reais possuem múltiplos receptores a distâncias diferentes do
transmissor (por exemplo, dTR = 0.5 m, 1.0 m, 1.5 m). Extensão proposta:

```
Parâmetro adicional em model.in:
  n_receivers = 3
  dTR_list = 0.5  1.0  1.5

Impacto no código:
  - Loop adicional sobre receptores dentro de fieldsinfreqs
  - Reutilização de commonarraysMD (depende apenas de hordist)
  - Arquivo .dat com n_receivers × 22 colunas por registro
```

### 16.7 Suporte a Anisotropia Biaxial (sigma_x != sigma_y)

**Prioridade:** Baixa | **Complexidade:** Muito alta | **Impacto estimado:** Cenários geológicos avançados

A formulação atual assume simetria TIV (sigma_x = sigma_y = sigma_h). Em formações
com fraturas orientadas ou laminação não-horizontal, a condutividade pode ser biaxial:

```
  sigma_biaxial = diag(sigma_x, sigma_y, sigma_z)

  Onde sigma_x != sigma_y  (quebra da simetria cilíndrica)
```

A decomposição TE/TM não é mais aplicável diretamente neste caso. Seria necessário
reformular as equações de propagação usando dois potenciais de Hertz acoplados
(pi_x e pi_y), resultando em um sistema 4×4 de equações diferenciais em vez de
dois sistemas 2×2 independentes. Isso aumentaria significativamente a complexidade
computacional e algorítmica.

---

## 17. Apêndices

### 17.1 Apêndice A — Tabela de Sub-rotinas

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

### 17.2 Apêndice B — Glossário

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

### 17.3 Apêndice C — Mapeamento de Variáveis Fortran → Python

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

### 17.4 Apêndice D — Validação de Consistência Fortran-Python

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

### 17.5 Apêndice E — Legenda Completa de Variáveis Matemáticas e Código

Esta tabela cobre todas as variáveis matemáticas e de código relevantes, com unidades,
descrições e referências cruzadas entre a formulação teórica (Seção 4), o código
Fortran (Seções 5-6) e o pipeline Python (Seção 14).

**E.1 Constantes Físicas**

| Símbolo | Variável | Unidade | Valor | Descrição |
|:--------|:---------|:--------|:------|:----------|
| mu_0 | `mu` (Fortran) | H/m | 4*pi*10^-7 | Permeabilidade magnética do vácuo |
| epsilon_0 | `epsilon` (Fortran) | F/m | 8.85*10^-12 | Permissividade elétrica do vácuo |
| pi | `pi` (Fortran) | - | 3.14159265... | Constante matemática |
| f | `freq(i)` (Fortran) | Hz | 20000.0 / 40000.0 | Frequência de operação |
| omega | `omega` (Fortran) | rad/s | 2*pi*f | Frequência angular |
| m_x, m_y, m_z | `mx, my, mz` (Fortran) | A.m^2 | 1.0 | Momentos dipolares magnéticos |

**E.2 Propriedades do Meio (por camada)**

| Símbolo | Variável | Unidade | Range | Descrição |
|:--------|:---------|:--------|:------|:----------|
| rho_h | `resist(i,1)` (F), `col 2` (P) | Ohm.m | 0.05-1500 | Resistividade horizontal |
| rho_v | `resist(i,2)` (F), `col 3` (P) | Ohm.m | >= rho_h | Resistividade vertical |
| sigma_h | `eta(i,1)` (F) | S/m | 1/rho_h | Condutividade horizontal |
| sigma_v | `eta(i,2)` (F) | S/m | 1/rho_v | Condutividade vertical |
| lambda | `sqrt(lamb2(i))` (F) | - | 1.0-sqrt(2) | Coeficiente de anisotropia |
| lambda^2 | `lamb2(i)` (F) | - | 1.0-2.0 | Quadrado do coef. anisotropia |
| h_m | `h(m)` (F) | m | 0-inf | Espessura da camada m |
| z_m | `prof(m)` (F) | m | -1e300 a +1e300 | Profundidade da interface inferior m |

**E.3 Grandezas Eletromagnéticas (por camada e ponto do filtro)**

| Símbolo | Variável Fortran | Tipo | Unidade | Descrição |
|:--------|:-----------------|:-----|:--------|:----------|
| zeta = i*omega*mu_0 | `zeta` | `complex(dp)` | Ohm/m | Impeditividade |
| kh^2 = -zeta*sigma_h | `kh2(j)` | `complex(dp)` | 1/m^2 | Número de onda horiz. ao quadrado |
| kv^2 = -zeta*sigma_v | `kv2(j)` | `complex(dp)` | 1/m^2 | Número de onda vert. ao quadrado |
| kr | `kr(i)` | `real(dp)` | 1/m | Número de onda radial (variável de integração) |
| u_m | `u(i,m)` | `complex(dp)` | 1/m | Constante de propagação TE |
| v_m | `v(i,m)` | `complex(dp)` | 1/m | Constante de propagação intermediária |
| s_m = lambda_m*v_m | `s(i,m)` | `complex(dp)` | 1/m | Constante de propagação TM |
| Y_m = u_m/zeta | `AdmInt(i,m)` | `complex(dp)` | S | Admitância intrínseca (TE) |
| Z_m = s_m/sigma_h | `ImpInt(i,m)` | `complex(dp)` | Ohm | Impedância intrínseca (TM) |

**E.4 Coeficientes de Reflexão e Admitâncias/Impedâncias Aparentes**

| Símbolo | Variável Fortran | Forma | Descrição |
|:--------|:-----------------|:------|:----------|
| Y_tilde_dw^(m) | `AdmAp_dw(i,m)` | `(npt, n)` | Admitância aparente TE descendente |
| Y_tilde_up^(m) | `AdmAp_up(i,m)` | `(npt, n)` | Admitância aparente TE ascendente |
| Z_tilde_dw^(m) | `ImpAp_dw(i,m)` | `(npt, n)` | Impedância aparente TM descendente |
| Z_tilde_up^(m) | `ImpAp_up(i,m)` | `(npt, n)` | Impedância aparente TM ascendente |
| R_TE_dw^(m) | `RTEdw(i,m)` | `(npt, n)` | Coeficiente reflexão TE descendente |
| R_TE_up^(m) | `RTEup(i,m)` | `(npt, n)` | Coeficiente reflexão TE ascendente |
| R_TM_dw^(m) | `RTMdw(i,m)` | `(npt, n)` | Coeficiente reflexão TM descendente |
| R_TM_up^(m) | `RTMup(i,m)` | `(npt, n)` | Coeficiente reflexão TM ascendente |
| tanh(u_m*h_m) | `tghuh(i,m)` | `(npt, n)` | Tangente hiperbólica estabilizada (TE) |
| tanh(s_m*h_m) | `tghsh(i,m)` | `(npt, n)` | Tangente hiperbólica estabilizada (TM) |

**E.5 Fatores de Onda e Transmissão**

| Símbolo | Variável Fortran | Forma | Descrição |
|:--------|:-----------------|:------|:----------|
| Mx_dw | `Mxdw(i)` | `(npt)` | Fator onda TM descendente (camada fonte) |
| Mx_up | `Mxup(i)` | `(npt)` | Fator onda TM ascendente (camada fonte) |
| Eu_dw | `Eudw(i)` | `(npt)` | Fator onda TE descendente (camada fonte) |
| Eu_up | `Euup(i)` | `(npt)` | Fator onda TE ascendente (camada fonte) |
| FE_dw_z | `FEdwz(i)` | `(npt)` | Fator TE derivada z descendente |
| FE_up_z | `FEupz(i)` | `(npt)` | Fator TE derivada z ascendente |
| Tx^(j)_dw | `Txdw(i,j)` | `(npt, n)` | Coeficiente transmissão TM descendente |
| Tx^(k)_up | `Txup(i,k)` | `(npt, n)` | Coeficiente transmissão TM ascendente |
| Tu^(j)_dw | `Tudw(i,j)` | `(npt, n)` | Coeficiente transmissão TE descendente |
| Tu^(k)_up | `Tuup(i,k)` | `(npt, n)` | Coeficiente transmissão TE ascendente |

**E.6 Kernels Espectrais e Campos**

| Símbolo | Variável Fortran | Forma | Descrição |
|:--------|:-----------------|:------|:----------|
| K_tm | `Ktm(i)` | `(npt)` | Kernel espectral modo TM (pi_x) |
| K_te | `Kte(i)` | `(npt)` | Kernel espectral modo TE (pi_u) |
| K_te_dz | `Ktedz(i)` | `(npt)` | Kernel TE derivada em z |
| H_x | `Hx_p(1,1:2)` | `complex(dp)` | Campo magnético x (HMDx, HMDy) |
| H_y | `Hy_p(1,1:2)` | `complex(dp)` | Campo magnético y (HMDx, HMDy) |
| H_z | `Hz_p(1,1:2)` | `complex(dp)` | Campo magnético z (HMDx, HMDy) |
| H(3,3) | `matH(3,3)` / `tH(3,3)` | `complex(dp)` | Tensor magnético completo |

**E.7 Geometria e Posicionamento**

| Símbolo | Variável Fortran | Variável Python | Unidade | Descrição |
|:--------|:-----------------|:---------------|:--------|:----------|
| h_0 | `h0` | - | m | Profundidade vertical do transmissor |
| z | `z` | - | m | Profundidade vertical do receptor |
| r | `hordist` | - | m | Distância horizontal T-R |
| x | `cx` | - | m | Coordenada x do receptor |
| y | `cy` | - | m | Coordenada y do receptor |
| theta | `theta(k)` | - | graus | Ângulo de inclinação do poço |
| dTR | `dTR` | `config.spacing_meters` | m | Espaçamento T-R |
| z_obs | `zrho(f,1)` | `col 1` | m | Profundidade do ponto médio T-R |
| nmed | `nmed(k)` | `config.sequence_length` | - | Número de medições por ângulo |

**E.8 Filtro de Hankel**

| Símbolo | Variável Fortran | Tipo | Descrição |
|:--------|:-----------------|:-----|:----------|
| kr_i | `absc(i)` / `hordist` | `real(dp)` | Abscissas do filtro escaladas |
| w_J0,i | `wJ0(i)` | `real(dp)` | Pesos do filtro para Bessel J_0 |
| w_J1,i | `wJ1(i)` | `real(dp)` | Pesos do filtro para Bessel J_1 |
| npt | `npt` | `integer` | Número de pontos do filtro (201) |

---

## Histórico de Execução de Otimizações

| Data        | Fase                                          | Status                  | Relatório                                                                      |
|:------------|:----------------------------------------------|:------------------------|:-------------------------------------------------------------------------------|
| 2026-04-04  | CPU Fase 0 — Baseline                         | ✅ Concluída            | [`relatorio_fase0_fase1_fortran.md`](relatorio_fase0_fase1_fortran.md)         |
| 2026-04-04  | CPU Fase 1 — SIMD Hankel                      | ⏭️ Pulada (auto-vec gfortran 15.x satura AVX-2) | [`relatorio_fase0_fase1_fortran.md`](relatorio_fase0_fase1_fortran.md) §3 |
| 2026-04-04  | CPU Fase 2 — Hybrid Scheduler + Débitos 1/2/3 | ✅ **Concluída**        | [`relatorio_fase2_debitos_fortran.md`](relatorio_fase2_debitos_fortran.md)     |
| —           | CPU Fase 3 — Workspace Pre-alloc              | 🔜 Próxima              | —                                                                              |
| —           | CPU Fases 4–6                                 | 📋 Planejadas           | [`analise_paralelismo_cpu_fortran.md`](analise_paralelismo_cpu_fortran.md) §7 |

**Baseline publicado (Fase 0, 2026-04-04 — i9-9980HK, 8 cores, AVX-2, gfortran 15.2.0, OMP=8, CPU fria)**:
- Wall-time: **0,1047 ± 0,015 s/modelo**
- Throughput: **~34.400 modelos/hora**
- MD5 de referência: `c64745ed5d69d5f654b0bac7dde23a95`

**Fase 2 (Hybrid Scheduler + Correções Débitos 1, 2, 3 — aplicada em `PerfilaAnisoOmp.f08`)**:

- **Débito 1** (`writes_files` append bug) **corrigido**: abertura condicional com `inquire()` + detecção de `modelm==1` OR arquivo ausente.
- **Débito 2** (`omp_set_nested` depreciado) **corrigido**: migração para `omp_set_max_active_levels(2)` (OpenMP 5.0+).
- **Débito 3** (particionamento `num_threads_j = maxthreads - ntheta` degenerava em `OMP=2`) **corrigido**: particionamento multiplicativo `num_threads_k × num_threads_j ≈ maxthreads`.
- **Schedule híbrido**: `dynamic` no loop externo de ângulos (carga desigual) + `static` no loop interno de medidas (carga uniforme).
- **Validação numérica**: MD5 idêntico ao baseline, `max|Δ| = 0,0000e+00` em todas as 21 colunas.
- **Scaling test**: bug 2-thread corrigido (−41 %, speedup 1,07× → 1,60×), 1 thread −12 %, 4 threads −23 %, trade-off marginal em 8–16 threads (+6 a +11 %).

---

*Documentação do Simulador Fortran PerfilaAnisoOmp — Geosteering AI v2.0*
*Versão 4.0 — Abril 2026 — Pipeline v5.0.15+*
*Última atualização: 2026-04-04 (Fase 2 + Débitos 1/2/3 concluídos)*
