# Documentação Completa do Simulador Fortran — PerfilaAnisoOmp

## Simulação Eletromagnética 1D para Meios Anisotrópicos TIV com Paralelismo OpenMP

**Projeto:** Geosteering AI v2.0 — Inversão 1D de Resistividade via Deep Learning
**Autor do Simulador:** Daniel Leal
**Linguagem:** Fortran 2008 (gfortran) com extensões OpenMP
**Localização:** `Fortran_Gerador/`
**Versão do Documento:** 9.0 (Abril 2026) — Fases 0-5b + 2b + Multi-TR + f2py wrapper + batch parallel + análise novos recursos (1.5D, 2D, compensação, antenas inclinadas, invasão, sensibilidades)

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
12. [Histórico de Execução de Otimizacoes (Fases 0-5b)](#12-histórico-de-execução-de-otimizacoes-fases-0-5b)
13. [Análise de Viabilidade CUDA (GPU)](#13-análise-de-viabilidade-cuda-gpu)
    - 13.7 [Pipeline A — Roteiro de Otimização do Código Fortran](#137-pipeline-a--roteiro-de-otimização-do-código-fortran)
    - 13.8 [Pipeline B — Roteiro de Novos Recursos para o Simulador Fortran](#138-pipeline-b--roteiro-de-novos-recursos-para-o-simulador-fortran)
14. [Análise de Reimplementação em Python Otimizado](#14-análise-de-reimplementação-em-python-otimizado)
    - 14.9 [Pipeline A — Conversão Fortran->Python Otimizado](#149-pipeline-a--conversão-fortranpython-otimizado-numba-jit)
    - 14.10 [Pipeline B — Novos Recursos da Versão Python](#1410-pipeline-b--novos-recursos-da-versão-python)
    - 14.11 [Pipeline C — Vantagens sobre a Versão Fortran](#1411-pipeline-c--vantagens-da-versão-python-sobre-o-fortran)
    - 14.12 [Pipeline D — Avaliacao Comparativa Fortran vs Python](#1412-pipeline-d--avaliação-comparativa-fortran-vs-python-otimizado)
15. [Feature 1 — Multi-TR (Múltiplos Pares Transmissor-Receptor)](#15-feature-1--multi-tr-múltiplos-pares-transmissor-receptor)
16. [Feature 2 — f2py Wrapper (Interface Python Nativa)](#16-feature-2--f2py-wrapper-interface-python-nativa)
17. [Feature 5 — Batch Parallel Runner](#17-feature-5--batch-parallel-runner)
18. [Integração com o Pipeline Geosteering AI v2.0](#18-integração-com-o-pipeline-geosteering-ai-v20)
19. [Referências Bibliográficas](#19-referências-bibliográficas)
20. [Apêndices](#20-apêndices)
21. [Roadmap de Novos Recursos — Estratégias Avançadas (v9.0+)](#21-roadmap-de-novos-recursos--estratégias-avançadas-v90)

---

## 1. Introdução e Motivação

### 1.1 Contexto do Problema

O simulador `PerfilaAnisoOmp` resolve o *problema direto* (forward modeling) da resposta eletromagnética (EM) de uma ferramenta de perfilagem triaxial em meios estratificados horizontalmente com anisotropia TIV (Transversalmente Isotropica Vertical). Este tipo de modelagem e fundamental para a geração de dados sintéticos de treinamento para redes neurais de inversão.

A cadeia completa do projeto Geosteering AI segue o fluxo:

```
Gerador Python          Simulador Fortran         Pipeline DL (Python/TF)
(modelos geologicos) --> (forward EM 1D TIV)  --> (inversao EM via rede neural)
  fifthBuildTIV            PerfilaAnisoOmp           geosteering_ai/
  Models.py                   tatu.x
       |                        |                        |
  Parametros               Campos EM               Perfis de rho
  geologicos               tensoriais              estimados
  (rho_h, rho_v,           (tensor H 3x3)          (rho_h, rho_v)
   espessuras)                                      log10 scale
```

### 1.2 O que e Forward Modeling (Modelagem Direta)

O **forward modeling** (modelagem direta ou problema direto) e o processo de calcular a resposta
física de um instrumento de medição dado um modelo conhecido do meio. Em termos concretos,
o simulador recebe como entrada um modelo geológico completamente específicado — isto e,
o número de camadas, suas espessuras, e as resistividades horizontal e vertical de cada
camada — e calcula como uma ferramenta EM triaxial "veria" esse modelo. O resultado e o
tensor completo de campo magnético H(3x3) em cada posição de medição ao longo do poco.

O forward modeling e o passo complementar a **inversão** (problema inverso), onde se deseja
determinar as propriedades do meio a partir das medições. Na abordagem do projeto Geosteering AI,
o forward modeling e utilizado para gerar milhares de pares (modelo geológico, resposta EM)
que servem como dados de treinamento supervisionado para redes neurais. A rede neural aprende
o mapeamento inverso: dada uma resposta EM, estimar as resistividades do meio.

Matemáticamente, se $\mathbf{d}$ representa os dados observados e $\mathbf{m}$ o modelo
geológico, o forward modeling computa a função:

```
d = F(m)    (forward: modelo -> dados)

A rede neural aprende a funcao inversa aproximada:
m_est = G(d)  ~=  F^{-1}(d)    (inversao: dados -> modelo estimado)
```

A qualidade da inversão via rede neural depende diretamente da fidelidade e diversidade
dos dados de treinamento gerados pelo forward modeling. Por isso, o simulador Fortran deve
ser físicamente preciso e capaz de gerar grandes volumes de dados eficientemente.

### 1.3 Geração de Dados de Treinamento

O conceito central do projeto e substituir a inversão EM tradicional (iterativa, lenta,
computacionalmente custosa) por uma rede neural treinada em dados sintéticos. O processo
de geração de dados de treinamento segue estas etapas:

1. **Geração de modelos geológicos aleatorios:** O script Python `fifthBuildTIVModels.py`
   gera milhares de modelos geológicos 1D com parâmetros amostrados via Sobol quasi-Monte Carlo.
   Cada modelo específica número de camadas (3-80), espessuras, e resistividades
   horizontais/verticais. Os cenarios cobrem desde modelos "amigaveis" (camadas grossas,
   contrastes moderados) até "patologicos" (camadas finas, contrastes extremos).

2. **Simulação EM (forward modeling):** Para cada modelo geológico, o simulador Fortran
   calcula a resposta EM completa ao longo de uma janela de 120 metros com passo de 0.2 m,
   resultando em 600 posições de medição. Em cada posição, o tensor H(3x3) completo e
   calculado para 1-2 frequências (20 kHz e 40 kHz).

3. **Armazenamento em formato binario:** Os resultados são salvos em arquivos `.dat`
   no formato binario Fortran stream, com 22 colunas por registro (1 índice int32 +
   21 float64: profundidade, rho_h, rho_v, e 18 componentes reais/imaginarias do tensor H).

4. **Consumo pelo pipeline Python:** O módulo `geosteering_ai/data/loading.py` le os
   arquivos `.dat` e organiza os dados em arrays NumPy de forma (n_modelos, seq_len, n_features),
   prontos para alimentar o pipeline de treinamento da rede neural.

### 1.4 Objetivo do Simulador

Dado um modelo geológico 1D (camadas horizontais com resistividades anisotrópicas), o simulador calcula o tensor completo de campo magnético `H(3x3)` medido por uma ferramenta triaxial LWD (Logging While Drilling) em cada posição de medição ao longo do poco. O resultado e armazenado em formato binario `.dat` para consumo pelo pipeline de Deep Learning.

### 1.5 Escopo da Documentação

Este documento apresenta:
- Fundamentos físicos completos (equações de Maxwell em meios TIV)
- Formulação matemática detalhada (decomposição TE/TM, transformada de Hankel, coeficientes de reflexão recursivos)
- Análise linha a linha do código Fortran
- Estrutura de dados de entrada e saída
- Análise completa de paralelismo e otimização
- Viabilidade de portabilidade para CUDA e Python
- **Novidades v8.0:** Multi-TR (múltiplos pares T-R simultaneos), f2py wrapper, batch parallel runner
- **Novidades v9.0:** Análise e roadmap de novos recursos: estratégia 1.5D (relative dip), 2D (Born approximation), compensação de poço (midpoint multi-TR), frequências arbitrárias, antenas inclinadas, efeito de invasão, sensibilidades ∂H/∂ρ, otimização OpenMP avançada (AVX-512, NUMA, tasks)

---

## 2. Fundamentos Físicos e Geofísicos

### 2.1 O Problema Eletromagnético em Meios Estratificados

A perfilagem EM de pocos utiliza fontes dipolares magnéticas operando em baixa frequência (tipicamente 20 kHz a 2 MHz) para sondar as propriedades elétricas das formacoes rochosas ao redor do poco. O campo eletromagnético emitido pelo transmissor penetra no meio geológico e e modificado pelas propriedades de resistividade de cada camada. Os receptores medem o campo resultante, que contém informação sobre a distribuição de resistividade.

### 2.2 Anisotropia TIV (Transversalmente Isotropica Vertical)

Rochas sedimentares apresentam anisotropia elétrica intrinseca devida a laminacao deposicional. No modelo TIV, a condutividade e descrita por um tensor diagonal:

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

Constraint fisico: rho_v >= rho_h (SEMPRE em rochas sedimentares TIV)

Coeficiente de anisotropia:
  lambda = sqrt(sigma_h / sigma_v) = sqrt(rho_v / rho_h)
  Range no dataset do projeto: 1.0 <= lambda <= sqrt(2) ~ 1.414
  (Em formacoes reais, lambda pode exceder sqrt(2); o range acima
   reflete a distribuicao de treinamento do gerador fifthBuildTIVModels.py)
```

**Por que rho_v >= rho_h em rochas sedimentares?**

A anisotropia elétrica TIV em rochas sedimentares tem origem na **microestrutura laminada**
dos sedimentos. Quando sedimentos de granulometrias diferentes se depositam alternadamente
(por exemplo, laminas de areia e folhelho), cada lamina possui uma resistividade diferente.
Em escala macroscopica, esse empilhamento de laminas finas se comporta como um material
anisotrópico equivalente:

```
  Corrente horizontal (paralela as laminas):
    ┌────────────────────────────────────────┐
    │  Areia (rho_a alto)   ───►  I_h        │  Corrente flui por AMBAS
    │  Folhelho (rho_f baixo) ───►  I_h      │  as laminas em PARALELO
    └────────────────────────────────────────┘
    -> Resistencia equivalente: media harmonica (dominada por rho_f baixo)
    -> Resultado: rho_h BAIXO

  Corrente vertical (perpendicular as laminas):
    ┌────────────────────────────────────────┐
    │  Areia (rho_a alto)     ↓ I_v          │  Corrente cruza TODAS
    │  ─────────────────────────────────     │  as laminas em SERIE
    │  Folhelho (rho_f baixo)  ↓ I_v         │
    └────────────────────────────────────────┘
    -> Resistencia equivalente: media aritmetica (influenciada por rho_a alto)
    -> Resultado: rho_v ALTO
```

Matemáticamente, para N laminas de espessura $h_i$ e resistividade $\rho_i$:

```
  1/rho_h = sum(h_i / rho_i) / sum(h_i)    (media harmonica -> dominada por valores BAIXOS)
  rho_v   = sum(h_i * rho_i) / sum(h_i)    (media aritmetica -> influenciada por valores ALTOS)
```

Como a media aritmetica e sempre >= media harmonica (desigualdade AM-HM), segue que
`rho_v >= rho_h` sempre. A igualdade so ocorre quando todas as laminas tem a mesma
resistividade (meio isotropico, lambda = 1).

Em formacoes com alternancia de areia e folhelho, o coeficiente de anisotropia lambda
pode atingir valores de 2 a 10 ou mais. No dataset de treinamento do projeto, o range
e conservador (1.0 a sqrt(2) ~ 1.414) para focar na faixa mais comum em reservatorios.

**Implicação no código Fortran:** Cada camada `i` possui dois valores de resistividade `resist(i,1) = rho_h` e `resist(i,2) = rho_v`, e as condutividades são calculadas como `eta(i,1) = 1/rho_h`, `eta(i,2) = 1/rho_v`.

### 2.3 Geometria do Meio Estratificado

O meio e composto por `n` camadas horizontais planas, empilhadas verticalmente:

```
    z = -infinity    ┌──────────────────────────────────────┐
                     │  Camada 1 (semi-espaco superior)     │
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
                     │  Camada n (semi-espaco inferior)      │
                     │  rho_h(n), rho_v(n)                  │
                     │  espessura: infinita                  │
    z = +infinity    └──────────────────────────────────────┘
```

**Convenção de fronteiras no código:**
- `prof(0) = -1e300` (sentinela para semi-espaço superior)
- `prof(1) = h(1) = 0` (primeira interface, atribuido diretamente)
- `prof(k) = prof(k-1) + h(k)` para `k = 2, ..., n-1`
- `prof(n) = +1e300` (sentinela para semi-espaço inferior)

A utilização de sentinelas `+/-1e300` elimina condicionais nos limites das exponenciais, evitando divisoes por zero e garantindo estabilidade numérica.

### 2.4 Ferramenta Triaxial LWD

A ferramenta simulada e composta por:
- **1 transmissor** com 3 dipolos magnéticos ortogonais (Mx, My, Mz)
- **1 receptor** com 3 receptores ortogonais (Hx, Hy, Hz)
- **Espacamento T-R:** `dTR` metros (default: 1.0 m; **v8.0: múltiplos dTR simultaneos**)

O tensor de campo magnético medido pela ferramenta triaxial completa e:

```
       ┌                ┐
       │ Hxx  Hxy  Hxz  │
  H =  │ Hyx  Hyy  Hyz  │    (9 componentes complexas)
       │ Hzx  Hzy  Hzz  │
       └                ┘

Onde H_ij = campo na direcao j devido ao dipolo na direcao i.

Linha 1: campo do DMH-x (Horizontal Magnetic Dipole, direcao x)
Linha 2: campo do DMH-y (Horizontal Magnetic Dipole, direcao y)
Linha 3: campo do DMV   (Vertical Magnetic Dipole, direcao z)
```

**Por que 9 componentes importam?**

Em um meio isotropico com poco vertical, o tensor H possui simetrias que reduzem o
número de componentes independentes. Por exemplo, Hxy = Hyx, Hxz = Hzx = 0, Hyy = Hxx,
e Hz* = 0 para dipolos horizontais. Porem, em meios TIV com poco inclinado (theta != 0),
essas simetrias se quebram e **todas as 9 componentes** se tornam independentes e
carregam informação distinta sobre a formacao:

```
  Poco vertical (theta = 0) em TIV:     Poco inclinado em TIV:
  ┌              ┐                       ┌                ┐
  │ Hxx  0    0  │                       │ Hxx  Hxy  Hxz  │  <- 9 independentes
  │ 0    Hxx  0  │  <- 2 independentes   │ Hyx  Hyy  Hyz  │     (tensor cheio)
  │ 0    0   Hzz │                       │ Hzx  Hzy  Hzz  │
  └              ┘                       └                ┘
```

As componentes diagonais (Hxx, Hyy, Hzz) são sensiveis principalmente a resistividade
na direção do respectivo dipolo. As componentes off-diagonal (Hxy, Hxz, etc.) sao
especialmente sensiveis a presenca de interfaces entre camadas e a inclinacao do poco,
fornecendo informação sobre a geometria da formacao que não esta disponível nos
componentes diagonais isoladamente.

No pipeline Geosteering AI, o modo P1 (baseline) utiliza apenas 5 features selecionadas
das 22 colunas: zobs, Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz). Porem, modos futuros (Modo C)
podem explorar o tensor completo de 9 componentes (18 valores reais) para melhorar a
resolução da inversão em cenarios com poco inclinado.

**Antenas inclinadas (tilted coils):**

Ferramentas LWD modernas (como a Schlumberger Rt Scanner ou a Halliburton EarthStar)
utilizam antenas inclinadas em relação ao eixo do poco, além das antenas axiais e
transversais tradicionais. Antenas inclinadas geram combinacoes lineares dos 9 componentes
do tensor, o que permite medir a anisotropia e a geometria das camadas com maior robustez.
O simulador `PerfilaAnisoOmp` calcula o tensor completo H(3x3), o que permite simular
qualquer configuração de antenas inclinadas via combinação linear pos-cálculo.

### 2.5 Arranjo T-R e Geometria de Perfilagem

O poco possui inclinacao `theta` em relação a vertical. A ferramenta percorre o poco com passo `p_med` metros, cobrindo uma janela de investigação `tj` metros:

```
                         theta (angulo de inclinacao)
                        /
                       /
    z1 = -h1   ◯ R  /  (receptor, inicio da janela, acima de T)
                  |/
                  ◯ T  (transmissor, dTR metros abaixo do R)
                 /|
                / |
               /  |  pz = p_med * cos(theta)  (passo vertical)
              /   |  px = p_med * sin(theta)  (passo horizontal)
             /    |
    ◯ R+1  /     |
          ◯ T+1  |
         .       |
         .       |
         .       |
    z1 + tj ◯ R_final
              ◯ T_final

  nmed = ceil(tj / pz)  (numero de medições por angulo)
```

**Posições do Transmissor e Receptor para a j-esima medição:**

```
Receptor (variavel `x`, `z` no código):
  x = 0 + (j-1) * px - Lsen/2
  z = z1 + (j-1) * pz - Lcos/2

Transmissor (variavel `Tx`, `Tz` no código):
  Tx = 0 + (j-1) * px + Lsen/2
  Tz = z1 + (j-1) * pz + Lcos/2

Onde:
  Lsen = dTR * sin(theta)
  Lcos = dTR * cos(theta)
  px = p_med * sin(theta)
  pz = p_med * cos(theta)
```

**Nota de configuração:** O transmissor esta **abaixo** dos receptores, conforme configuração padrao dos arranjos da Petrobras.

**Novidade v8.0 — Múltiplos pares T-R:** A versão 7.0+ do simulador suporta `nTR` esspacamentos T-R simultaneos via loop externo `do itr = 1, nTR`. Cada par produz `Lsen(itr) = dTR(itr) * sin(theta)` e `Lcos(itr) = dTR(itr) * cos(theta)`, e a distância horizontal `r_k = dTR(itr) * |sin(theta)|` e recalculada para cada par, exigindo recomputo do cache Fase 4 (commonarraysMD).

### 2.6 Constantes Físicas do Simulador

| Constante | Simbolo | Valor | Unidade | Significado |
|:----------|:--------|:------|:--------|:------------|
| Permeabilidade magnética | mu | 4pi x 10^-7 | H/m | Permeabilidade do vacuo |
| Permissividade elétrica | epsilon | 8.85 x 10^-12 | F/m | Permissividade do vacuo |
| Tolerancia numérica | eps | 10^-9 | - | Limiar para singularidades |
| Tolerancia angular | del | 0.1 | graus | Resolução angular minima |
| Corrente do dipolo | Iw | 1.0 | A | Corrente do transmissor |
| Momento dipolar | mx, my, mz | 1.0 | A.m^2 | Momento do dipolo magnético |

### 2.7 Skin Depth e Profundidade de Investigacao (DOI)

O **skin depth** (profundidade pelicular ou de penetracao) e a distância na qual a
amplitude da onda EM decai por um fator de 1/e (~37%) em relação ao valor na fonte.
E o parâmetro fundamental que determina o alcance da investigação da ferramenta:

```
  delta = sqrt(2 / (omega * mu * sigma))

Onde:
  omega = 2*pi*f  (frequencia angular, rad/s)
  mu = 4*pi*10^-7 H/m  (permeabilidade magnetica)
  sigma = condutividade do meio (S/m)

Valores tipicos para f = 20 kHz (FREQUENCY_HZ do projeto):

  ┌──────────────────────────────────────────────────────────────┐
  │  Formacao              │  rho (Ohm.m)  │  delta (m)          │
  ├────────────────────────┼───────────────┼─────────────────────┤
  │  Folhelho salino       │  0.3          │  0.62               │
  │  Folhelho              │  1.0          │  1.13               │
  │  Arenito saturado      │  10           │  3.56               │
  │  Arenito com oleo      │  100          │  11.25              │
  │  Carbonato compacto    │  1000         │  35.59              │
  │  Sal/anidrita          │  10000        │  112.54             │
  └──────────────────────────────────────────────────────────────┘
```

**Implicacoes para a profundidade de investigação (DOI):**

A DOI real da ferramenta e tipicamente 2-3 vezes o skin depth, dependendo da
relação sinal-ruido do instrumento. Para a configuração padrao do projeto
(f = 20 kHz, dTR = 1.0 m):

- Em meios condutivos (rho < 1 Ohm.m): DOI ~ 1-2 m. A ferramenta "ve"
  apenas a camada imediata.
- Em meios resistivos (rho > 100 Ohm.m): DOI ~ 20-30 m. A ferramenta
  detecta interfaces distantes.
- Frequências mais baixas aumentam o skin depth (e a DOI), mas reduzem
  a resolução vertical.

O skin depth também explica por que o simulador usa frequências na faixa de
kHz (20 kHz, 40 kHz): frequências muito altas (MHz) teriam skin depth de
centimetros em meios condutivos, insuficiente para sondar além do poco.
Frequências muito baixas (Hz) teriam skin depth de quilometros, mas com
resolução vertical inadequada para geosteering.

### 2.8 Decoupling — Remocao do Acoplamento Direto

O campo magnético medido pelo receptor e composto por duas contribuicoes:
o **acoplamento direto** (campo do transmissor que chega ao receptor sem
interagir com o meio) e o **campo secundario** (campo refletido/refratado
pelas interfaces das camadas). Para a inversão, interessa apenas o campo
secundario, pois ele carrega a informação sobre as propriedades do meio.

O **decoupling** e a operação de subtrair o acoplamento direto do campo
total medido. Para dipolos magnéticos em espaço livre, o acoplamento
direto analitico e:

```
Campo do dipolo magnetico em espaco livre (r = distancia T-R):

  Para componentes PLANARES (Hxx, Hyy):
    AC_p = -1 / (4 * pi * L^3)
    AC_p ~= -0.079577  para L = 1.0 m

  Para componente AXIAL (Hzz):
    AC_x = +1 / (2 * pi * L^3)
    AC_x ~= +0.159155  para L = 1.0 m

Onde L = espacamento T-R (SPACING_METERS = 1.0 m no projeto).
```

No pipeline Geosteering AI, o decoupling e implementado pelas **views de geosinais (GS)**
no módulo `geosteering_ai/data/geosignals.py`. As views subtraem os valores AC_p e AC_x
das componentes correspondentes do tensor H antes de alimentar a rede neural.

**Nota importante:** O simulador Fortran NAO realiza o decoupling — ele armazena o campo
total (acoplamento direto + campo secundario). O decoupling e feito exclusivamente
no pipeline Python, permitindo que diferentes estrategias de decoupling sejam testadas
sem recomputar a simulação.

```
  Campo total       =  Acoplamento direto  +  Campo secundario
  (armazenado .dat)    (analitico, AC)        (contem info geologica)

  No pipeline:
    campo_decoupled = campo_total - AC
    -> Usado como input da rede neural
```

---

## 3. Formulação Matemática Completa

### 3.1 Equações de Maxwell em Meios TIV

O campo EM em meios com anisotropia TIV obedece as equações de Maxwell no dominio da frequência (convenção `exp(-i*omega*t)`):

```
nabla x E = i*omega*mu*H           (Lei de Faraday)
nabla x H = sigma_tensor * E + J   (Lei de Ampere)

Onde:
  omega = 2*pi*f  (frequencia angular, rad/s)
  mu = 4*pi*10^-7 H/m  (permeabilidade, assumida isotropica)
  sigma_tensor = diag(sigma_h, sigma_h, sigma_v)  (condutividade TIV)
  J = fonte dipolar (corrente do transmissor)
```

**Sobre a corrente de deslocamento:**

A lei de Ampere completa inclui o termo de corrente de deslocamento:

```
nabla x H = sigma * E + J + dD/dt

No dominio da frequencia:
nabla x H = (sigma - i*omega*epsilon) * E + J
```

O simulador **desconsidera** a corrente de deslocamento (`-i*omega*epsilon*E`).
Esta simplificacao e válida quando a corrente de condução domina sobre a corrente
de deslocamento, ou seja, quando:

```
  sigma >> omega * epsilon

Para f = 20 kHz e rho = 1000 Ohm.m (caso MAIS resistivo do dataset):
  sigma = 0.001 S/m
  omega * epsilon = 2*pi*20000 * 8.85e-12 ~= 1.11e-6 S/m

  Razao: sigma / (omega*epsilon) = 0.001 / 1.11e-6 ~= 900

Mesmo no caso mais desfavoravel, a corrente de conducao e 900 vezes maior
que a corrente de deslocamento. Para formacoes tipicas (rho < 100 Ohm.m),
a razao excede 10^4, tornando a simplificacao extremamente precisa.
```

Esta e a chamada **apróximação quasi-estática**, válida para frequências
na faixa de kHz a MHz em formacoes geológicas. A apróximação falha apenas
para frequências muito altas (> 100 MHz) ou meios extremamente resistivos
(rho > 10^6 Ohm.m), ambos fora do escopo da perfilagem LWD.

### 3.2 Impeditividade (zeta)

```
zeta = i * omega * mu

No código (Fortran):
  zeta = cmplx(0, 1.d0, kind=dp) * omega * mu

Significado fisico: zeta relaciona o campo eletrico ao campo magnetico
nas equações de Maxwell. E a impedancia intrinseca do meio multiplicada
pelo numero de onda.
```

### 3.3 Números de Onda e Constantes de Propagacao

Para cada camada `i` com condutividades `eta_h = 1/rho_h` e `eta_v = 1/rho_v`:

```
Numero de onda horizontal ao quadrado:
  kh^2(i) = -zeta * eta_h(i) = -i*omega*mu*sigma_h(i)

Numero de onda vertical ao quadrado:
  kv^2(i) = -zeta * eta_v(i) = -i*omega*mu*sigma_v(i)

Constantes de propagacao (dependem do parametro espectral kr):
  u(i) = sqrt(kr^2 - kh^2(i))   (propagacao horizontal, modo TE)
  v(i) = sqrt(kr^2 - kv^2(i))   (propagacao vertical)
  s(i) = lambda(i) * v(i)       (propagacao TIV, modo TM)

Onde:
  lambda(i) = sqrt(sigma_h(i) / sigma_v(i))  (coeficiente de anisotropia)
  kr = variavel de integracao no dominio espectral (Hankel)
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

O campo EM em meios estratificados TIV e decomposto em dois modos independentes:

**Modo TE (Transverse Electric):** Componente do campo elétrico tangencial as interfaces.
- Governado pela constante de propagação `u` (horizontal)
- Associado a admitancia intrinseca `AdmInt = u / zeta`

**Modo TM (Transverse Magnetic):** Componente do campo magnético tangencial as interfaces.
- Governado pela constante de propagação `s` (TIV)
- Associado a impedancia intrinseca `ImpInt = s / eta_h`

```
                       Campo EM total
                      /               \
                   Modo TE            Modo TM
                  (u, AdmInt)        (s, ImpInt)
                  /      \           /      \
              RTEup   RTEdw      RTMup   RTMdw
           (reflexao) (reflexao)(reflexao)(reflexao)
```

**Intuicao física para a decomposição TE/TM:**

A decomposição TE/TM e a separação do campo EM em duas polarizacoes independentes
relativas as interfaces horizontais das camadas. Essa decomposição e possível porque,
em meios estratificados 1D com anisotropia TIV, as interfaces são planas e horizontais,
e o tensor de condutividade e diagonal com simetria cilindrica em torno do eixo z.

- **Modo TE:** O campo elétrico e inteiramente tangencial (horizontal) as interfaces.
  Como a corrente flui paralelamente as laminas, o modo TE "ve" apenas a condutividade
  horizontal sigma_h. Sua constante de propagação `u` depende somente de kh^2.
  O modo TE e **insensivel a anisotropia** em poco vertical — a resposta TE de um
  meio TIV e identica a de um meio isotropico com sigma = sigma_h.

- **Modo TM:** O campo magnético e inteiramente tangencial as interfaces, e o campo
  elétrico possui componente vertical (perpendicular as laminas). Por isso, o modo TM
  "sente" tanto sigma_h quanto sigma_v. Sua constante de propagação `s = lambda*v`
  depende da razao sigma_h/sigma_v (anisotropia). O modo TM e o **único canal
  de informação sobre rho_v** em configurações com poco vertical.

Esta separação e analoga a decomposição de ondas opticas em polarização s e p
na reflexão em uma interface plana. A grande vantagem e que cada modo pode ser
resolvido independentemente (equação escalar de 2a ordem), reduzindo o problema
vetorial 3D a dois problemas escalares 1D.

### 3.5 Coeficientes de Reflexao Recursivos

Os coeficientes de reflexão são calculados recursivamente das fronteiras mais externas para a camada do transmissor, utilizando a formula de estabilidade numérica baseada em `tanh`:

**Direcao descendente (de cima para baixo):**

```
Admitancia aparente descendente:
  AdmAp_dw(n) = AdmInt(n)    (semi-espaco inferior: sem reflexao)

  AdmAp_dw(i) = AdmInt(i) * [AdmAp_dw(i+1) + AdmInt(i) * tanh(u(i)*h(i))]
                              / [AdmInt(i) + AdmAp_dw(i+1) * tanh(u(i)*h(i))]

Coeficiente de reflexao TE descendente:
  RTEdw(n) = 0    (sem reflexao na ultima camada)
  RTEdw(i) = [AdmInt(i) - AdmAp_dw(i+1)] / [AdmInt(i) + AdmAp_dw(i+1)]
```

**Formula `tanh` no código (estabilidade numérica):**

```fortran
! Em vez de tanh(x) diretamente, o código usa:
tghuh(:,i) = (1.d0 - exp(-2.d0 * uh(:,i))) / (1.d0 + exp(-2.d0 * uh(:,i)))

! Que e matematicamente equivalente a tanh(uh) mas evita overflow
! quando uh e grande (camadas espessas ou alta frequencia).
! Para uh >> 1: tanh(uh) -> 1.0 (correto)
! Para uh << 1: tanh(uh) -> uh (correto)
```

**Direcao ascendente (de baixo para cima):**

```
Admitancia aparente ascendente:
  AdmAp_up(1) = AdmInt(1)    (semi-espaco superior: sem reflexao)

  AdmAp_up(i) = AdmInt(i) * [AdmAp_up(i-1) + AdmInt(i) * tanh(u(i)*h(i))]
                              / [AdmInt(i) + AdmAp_up(i-1) * tanh(u(i)*h(i))]

Coeficiente de reflexao TE ascendente:
  RTEup(1) = 0    (sem reflexao na primeira camada)
  RTEup(i) = [AdmInt(i) - AdmAp_up(i-1)] / [AdmInt(i) + AdmAp_up(i-1)]
```

As mesmas formulas se aplicam ao modo TM substituindo `AdmInt` por `ImpInt`, `u` por `s`, e `uh` por `sh`.

### 3.6 Fatores de Onda Refletida na Camada do Transmissor

A sub-rotina `commonfactorsMD` calcula os fatores de onda refletida específicamente para a camada onde o transmissor esta localizado. Estes fatores encapsulam as multiplas reflexoes dentro da camada fonte:

**Modo TM (fatores Mxdw, Mxup):**

```
den_TM = 1 - RTMdw(T) * RTMup(T) * exp(-2*s*h(T))

Mxdw = [exp(-s*(prof(T) - h0)) + RTMup(T) * exp(s*(prof(T-1) - h0 - h(T)))] / den_TM
Mxup = [exp(s*(prof(T-1) - h0)) + RTMdw(T) * exp(-s*(prof(T) - h0 + h(T)))] / den_TM

Onde:
  T = camadT (indice da camada do transmissor)
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

Quando o receptor esta em uma camada diferente do transmissor, os campos são calculados via coeficientes de transmissão recursivos:

**Receptor abaixo do transmissor (camadR > camadT):**

```
Txdw(T) = mx / (2*s(T))    (termo fonte na camada T)
Tudw(T) = -mx / 2          (termo fonte TE)

Para j = T+1 ate camadR:
  Txdw(j) = s(j-1) * Txdw(j-1) * (...exponenciais...) / [(1 - RTMdw(j)*exp(-2*sh(j))) * s(j)]
  Tudw(j) = u(j-1) * Tudw(j-1) * (...exponenciais...) / [(1 - RTEdw(j)*exp(-2*uh(j))) * u(j)]
```

**Receptor acima do transmissor (camadR < camadT):** Formulas analogas, com direção ascendente e coeficientes RTMup/RTEup.

### 3.8 Transformada de Hankel via Filtros Digitais

A passagem do dominio espectral (kr) para o dominio espacial (r) e realizada pela transformada de Hankel, implementada via filtro digital:

```
f(r) = integral_0^inf F(kr) * Jn(kr*r) * kr * d(kr)

Onde Jn e a funcao de Bessel de primeira especie de ordem n.

Implementacao via filtro digital (Werthmuller, 201 pontos):

  f(r) ~= (1/r) * sum_{i=1}^{npt} F(kr_i/r) * w_i

Onde:
  kr_i = abscissas do filtro (tabeladas)
  w_i = pesos do filtro (tabelados, diferentes para J0 e J1)
  npt = 201 pontos (filtro Werthmuller)
```

**Por que a transformada de Hankel funciona (simetria cilindrica):**

A transformada de Hankel e a ferramenta matemática natural para problemas com
**simetria cilindrica**. Em meios estratificados 1D, as interfaces são planas e
horizontais, e as fontes dipolares geram campos com dependência azimutal simples
(cos(phi) ou sin(phi) para HMD, e nenhuma dependência azimutal para VMD).

Quando aplicamos a transformada de Fourier 2D nas coordenadas horizontais (x, y)
das equações de Maxwell, obtemos integrais no espaço (kx, ky). Devido a simetria
cilindrica do meio, podemos converter essas integrais duplas em uma integral simples
sobre o número de onda radial kr = sqrt(kx^2 + ky^2), usando a identidade:

```
  integral_{-inf}^{+inf} integral_{-inf}^{+inf} F(kx,ky) * exp(i*(kx*x+ky*y)) dkx dky
  = 2*pi * integral_0^{inf} F(kr) * J_n(kr*r) * kr * d(kr)

  Onde r = sqrt(x^2 + y^2) e J_n e a funcao de Bessel de ordem n.
  n = 0 para VMD (simetria axial completa)
  n = 0 e n = 1 para HMD (dependencia cos(phi))
```

Esta redução de integral dupla para simples e a razao pela qual o cálculo e
computacionalmente tratavel. Em vez de avaliar uma integral 2D (O(N^2) pontos),
avaliamos uma soma 1D de 201 termos — uma economia de várias ordens de magnitude.

O filtro digital substitui a integração numérica (quadratura) por uma soma
ponderada com coeficientes pre-calculados, tornando a avaliação extremamente
eficiente: ~201 multiplicacoes complexas e uma soma por componente de campo.

**Filtros disponiveis no código (`filtersv2.f08`):**

| Filtro | Pontos | Referencia | Uso |
|:-------|:------:|:-----------|:----|
| Kong | 61 | Kong (2007) | Rapido, precisão moderada |
| Key | - | Key (2012) | Precisão padrao |
| **Werthmuller** | **201** | **Werthmuller (2006)** | **Usado no simulador** |
| Anderson | 801 | Anderson (1982) | Alta precisão |

O simulador utiliza o filtro Werthmuller de 201 pontos (`npt = 201`), que oferece excelente balanco entre precisão e desempenho.

### 3.9 Campos do Dipolo Magnetico Horizontal (HMD) em Meio TIV

O campo magnético do HMD e calculado a partir dos kernels espectrais Ktm (modo TM) e Kte (modo TE), convolvidos com funções de Bessel J0 e J1:

```
Para o HMD na direcao x (hmdx):

Hx = [(2x^2/r^2 - 1) * sum(Ktedz * wJ1)/r - kh^2*(2y^2/r^2 - 1) * sum(Ktm * wJ1)/r
      - x^2/r^2 * sum(Ktedz * wJ0 * kr) + kh^2*y^2/r^2 * sum(Ktm * wJ0 * kr)] / (2*pi*r)

Hy = xy/r^2 * [sum(Ktedz * wJ1 + kh^2 * Ktm * wJ1)/r
               - sum((Ktedz * wJ0 + kh^2 * Ktm * wJ0) * kr)/2] / (pi*r)

Hz = -x * sum(Kte * wJ1 * kr^2) / (r * 2*pi*r)
```

**Propriedade de simetria do HMDy:** O campo do dipolo y e obtido por rotação de 90 graus do HMDx:

```
HMDy: x -> y, y -> -x
Hx(hmdy) = Hy(hmdx)
Hy(hmdy) = expressao com (2y^2/r^2 - 1) e x^2/r^2 (permutados)
Hz(hmdy) = -y * sum(Kte * wJ1 * kr^2) / (r * 2*pi*r)
```

O modo `'hmdxy'` calcula ambos simultaneamente, evitando recomputação dos kernels.

### 3.10 Campos do Dipolo Magnetico Vertical (VMD) em Meio TIV

O VMD excita apenas o modo TE (por simetria axial, não ha acoplamento TM):

```
Hx = -x * sum(KtedzzJ1 * kr^2) / (2*pi*r) / r

Hy = -y * sum(KtedzzJ1 * kr^2) / (2*pi*r) / r

Hz = sum(KtezJ0 * kr^3) / (2*pi*zeta) / r

Onde:
  KtezJ0 = fac * wJ0     (kernel z com Bessel J0)
  KtezJ1 = fac * wJ1     (kernel z com Bessel J1)
  KtedzzJ1 = AdmInt * KtezJ1  (kernel derivada z com J1)
```

### 3.11 Rotacao do Tensor para Orientacao Arbitraria da Ferramenta

O tensor `H` calculado no sistema de coordenadas geológico (x, y, z fixos) deve ser rotacionado para o sistema de coordenadas da ferramenta. A rotação e definida por tres ângulos de Euler (alpha, beta, gamma):

```
H_ferramenta = R^T * H_geologico * R

Onde R e a matriz de rotacao (Liu, 2017, eq. 4.80):

R = [  cos(a)*cos(b)*cos(g) - sin(b)*sin(g)    -cos(a)*cos(b)*sin(g) - sin(b)*cos(g)    sin(a)*cos(b) ]
    [  cos(a)*sin(b)*cos(g) + cos(b)*sin(g)    -cos(a)*sin(b)*sin(g) + cos(b)*cos(g)    sin(a)*sin(b) ]
    [ -sin(a)*cos(g)                             sin(a)*sin(g)                            cos(a)        ]

Na perfilagem LWD:
  alpha = theta (inclinacao do poco)
  beta = 0      (azimute, assumido zero)
  gamma = 0     (rotacao da ferramenta, assumida zero)
```

---

## 4. Formulação Teórica via Potenciais de Hertz

Esta seção apresenta a derivacao teórica completa da formulação de potenciais de Hertz
para dipolos magnéticos em meios estratificados com anisotropia TIV. A formulação segue
Moran & Gianzero (1979) e esta documentada em detalhe no documento TeX do projeto
(`Tex_Projects/TatuAniso/FormulaçãoTatuAnisoTIV.tex`). As equações aqui apresentadas
são a fundamentacao matemática direta do código Fortran em `magneticdipoles.f08` e
`utils.f08`.

### 4.1 Equações de Maxwell para Meios TIV no Dominio da Frequência

O simulador adota a convenção temporal `exp(+i*omega*t)` (engenharia), de modo que as
transformadas direta e inversa de Fourier sao:

```
Transformada direta:
  f_hat(omega) = integral_{-inf}^{+inf} f(t) * exp(-i*omega*t) dt

Transformada inversa:
  f(t) = (1/2*pi) * integral_{-inf}^{+inf} f_hat(omega) * exp(+i*omega*t) d_omega

Dependencia temporal resultante: exp(+i*omega*t)
```

As equações de Maxwell no dominio da frequência, para esta convenção, sao:

```
(i)    div(epsilon * E) = rho_V                  (Lei de Gauss)
(ii)   rot(H) - y * E = J_e                      (Lei de Ampere)
(iii)  div(mu * H) = 0                           (Lei de Coulomb magnetica)
(iv)   rot(E) + zeta * H = J_m                   (Lei de Faraday)

Onde:
  epsilon = permissividade dieletrica
  rho_V   = densidade volumetrica de carga eletrica
  y       = sigma + i*omega*epsilon_0  (admitividade, tensor para TIV)
  mu      = permeabilidade magnetica (isotropica, mu_0)
  zeta    = i*omega*mu_0              (impeditividade)
  J_e     = vetor densidade de corrente eletrica na fonte
  J_m     = vetor densidade de "corrente magnetica" na fonte
  E       = campo vetorial eletrico
  H       = campo vetorial magnetico
```

**Fonte de dipolo magnético:**

Para um dipolo magnético com momento `m`, as fontes sao:

```
J_m = -zeta * m * delta(x) * delta(y) * delta(z)
J_e = 0

Onde m = (m_x, m_y, m_z) e o vetor momento do dipolo.
No código Fortran: mx = my = mz = 1.0 A.m^2.
```

**Tensor admitividade para meios TIV:**

Em meios com anisotropia TIV na condutividade, a admitividade e um tensor diagonal:

```
         ┌                              ┐
         │ sigma_h + i*omega*eps_0    0                      0                 │
  y  =   │ 0                      sigma_h + i*omega*eps_0    0                 │
         │ 0                      0                      sigma_v + i*omega*eps_0│
         └                              ┘

Onde sigma_h e sigma_v sao as condutividades horizontal e vertical, respectivamente.
```

**Apróximacao quasi-estática:**

Sob o regime quasi-estático (sigma >> omega*epsilon_0), o tensor simplifica para
`y ~= diag(sigma_h, sigma_h, sigma_v)`. Esta e a apróximação adotada pelo simulador
(válidada na Seção 3.1).

**Densidade volumétrica de carga em meios TIV:**

Uma consequência direta da anisotropia TIV e a existência de uma densidade volumétrica
de carga elétrica mesmo em regime estacionario. A partir da equação da continuidade
`div(J) = 0` e da lei de Ohm `J = y * E`, obtem-se:

```
div(y * E) = 0

Expandindo com o tensor TIV e usando a Lei de Gauss:

  sigma_h * (dE_x/dx + dE_y/dy) + sigma_v * dE_z/dz = 0

Definindo lambda^2 = sigma_h / sigma_v e substituindo div(E) = rho_V / epsilon_0:

  rho_V = (epsilon_0 / lambda^2) * (lambda^2 - 1) * dE_z/dz

Significado: Em meios anisotropicos (lambda != 1), a descontinuidade de sigma
na direcao z induz uma separacao de cargas proporcional ao gradiente vertical
do campo eletrico. Essa carga desaparece no caso isotropico (lambda = 1).
```

### 4.2 Potenciais de Hertz (pi_x, pi_u, pi_z)

O potêncial de Hertz `pi` e um potêncial vetorial auxiliar que permite desacoplar as
equações de Maxwell em equações escalares independentes para cada modo de propagação.

**Definicao (Moran & Gianzero, 1979):**

```
y * E = -y_h * zeta * rot(pi)

Onde:
  pi = (pi_x, pi_y, pi_z) e o vetor potencial de Hertz
  y_h = sigma_h (admitividade horizontal escalar)
  zeta = i*omega*mu_0 (impeditividade)
```

A escolha de `y_h` no membro direito e deliberada: simplifica a forma final das
equações diferenciais para os potenciais.

**Condicao de calibre (gauge):**

Seguindo Moran & Gianzero (1979), a condição de calibre adotada e:

```
div(y * pi) = sigma_v * Phi

Onde Phi e o potencial escalar associado.
```

**Campo H em termos dos potenciais:**

Substituindo a definicao do potêncial de Hertz na lei de Ampere e aplicando a
condição de calibre, obtem-se:

```
H = -zeta * y_h * pi + (1/sigma_v) * grad(div(y * pi))
```

As componentes do campo magnético são entao:

```
H_x = kh^2 * pi_x + lambda^2 * (d^2 pi_x/dx^2 + d^2 pi_y/dxdy) + d^2 pi_z/dzdx
H_y = kh^2 * pi_y + lambda^2 * (d^2 pi_x/dxdy + d^2 pi_y/dy^2) + d^2 pi_z/dzdy
H_z = kh^2 * pi_z + lambda^2 * (d^2 pi_x/dzdx + d^2 pi_y/dzdy) + d^2 pi_z/dz^2

Onde kh^2 = -i*omega*mu_0*sigma_h = -zeta*sigma_h
```

**Simplificacao para fonte sem componente y:**

Para um dipolo magnético horizontal na direção x (DMH_x), não ha necessidade de
componente `pi_y`. O potêncial de Hertz reduz-se a `pi = (pi_x, 0, pi_z)`.

### 4.3 Equações de Onda Desacopladas para os Potenciais

Substituindo as expressoes de H e E em função de `pi` na lei de Faraday, e apos
manipulacao algebrica extensa (detalhada no TeX), obtem-se equações de onda
desacopladas para cada componente do potêncial:

**Equacao para pi_x (modo TM):**

```
d^2 pi_x/dx^2 + d^2 pi_x/dy^2 + (1/lambda^2) * d^2 pi_x/dz^2
  - i*omega*mu_0*sigma_v * pi_x = -(m_x / lambda^2) * delta(x)*delta(y)*delta(z)
```

**Equacao para pi_y (modo TM, analoga):**

```
d^2 pi_y/dx^2 + d^2 pi_y/dy^2 + (1/lambda^2) * d^2 pi_y/dz^2
  - i*omega*mu_0*sigma_v * pi_y = -(m_y / lambda^2) * delta(x)*delta(y)*delta(z)
```

**Equacao para pi_z (modo misto, acoplada com pi_x e pi_y):**

```
d^2 pi_z/dx^2 + d^2 pi_z/dy^2 + d^2 pi_z/dz^2
  - i*omega*mu_0*sigma_h * pi_z = (1-lambda^2) * d/dz(dpi_x/dx + dpi_y/dy)
                                    - m_z * lambda^2 * delta(x)*delta(y)*delta(z)
```

**Observacoes fundamentais:**

1. As equações para `pi_x` e `pi_y` são **desacopladas** entre si e são identicas
   em forma. Basta resolver uma delas; a outra e obtida por analogia.

2. A equação para `pi_z` e **acoplada** com `pi_x` e `pi_y` (via o termo com
   derivadas cruzadas). No entanto, como primeiro resolvemos `pi_x` e `pi_y`,
   o membro direito de `pi_z` e conhecido, tornando-a uma equação forcada.

3. O operador em `pi_x` e `pi_y` e anisotrópico: o fator `1/lambda^2` multiplica
   a derivada em z. Isso reflete a propagação mais lenta na direção vertical em
   meios com `sigma_v < sigma_h`.

4. O operador em `pi_z` e **isotropico** (todas as derivadas segundas com coeficiente 1),
   mas com `kh^2` em vez de `kv^2`. Isso e consequência da escolha de calibre.

### 4.4 Solucoes Espectrais no Meio Ilimitado

A solução das equações de onda e obtida aplicando a transformada tripla de Fourier
para passar do dominio `(x, y, z)` para `(k_x, k_y, k_z)`.

**Solucao para pi_x (transformada tripla):**

Definindo `v^2 = k_x^2 + k_y^2 - kv^2` e `kv^2 = -i*omega*mu_0*sigma_v`:

```
pi_x_hat_hat_hat(k_x, k_y, k_z) = m_x / ((lambda*v)^2 + k_z^2)
```

Aplicando a transformada inversa em `k_z`:

```
pi_x_hat_hat(k_x, k_y, z) = m_x * exp(-lambda*v*|z - h_0|) / (2*lambda*v)

Onde h_0 e a posicao vertical da fonte (transmissor).

Definindo s = lambda*v (constante de propagacao TM):
  pi_x_hat_hat = m_x * exp(-s*|z - h_0|) / (2*s)

Para z > h_0: pi_x_hat_hat = m_x * exp(-s*(z - h_0)) / (2*s)
Para z < h_0: pi_x_hat_hat = m_x * exp(+s*(z - h_0)) / (2*s)
```

**Solucao para pi_z (via pi_x e pi_u):**

A solução de `pi_z` e obtida a partir de `pi_x` e de um potêncial auxiliar `pi_u`
associado ao modo TE:

```
pi_z_hat_hat = -(i*k_x / k_r^2) * d(pi_x_hat_hat)/dz + (i*k_x / k_r^2) * pi_u_hat_hat

Onde k_r^2 = k_x^2 + k_y^2 (numero de onda radial ao quadrado).
```

O potêncial `pi_u` e a parte de `pi_z` associada ao modo TE. Suas solucoes no
meio ilimitado sao:

```
pi_u_hat_hat = (m_x/2) * exp(-u*|z - h_0|)

Onde u = sqrt(k_r^2 - kh^2) e a constante de propagacao TE.
```

**Definicoes-chave:**

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Grandeza            │  Expressao              │  Modo       │
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

No modelo de camadas horizontais com fonte (transmissor) na camada `l`, as solucoes
espectrais dos potenciais `pi_x` e `pi_u` assumem formas distintas em 6 zonas
geométricas. Cada zona corresponde a uma combinação diferente de ondas transmitidas e
refletidas nas interfaces.

**Diagrama do modelo de camadas:**

```
  z = -inf     ┌────────────────────────────────────────┐
               │  Zona 0: Semi-espaco superior          │  pi_x^(0) = Tx^(0) * exp(s_0*z)
               │  (sigma_h0, sigma_v0)                  │  -> Apenas onda transmitida ascendente
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
               │  Zona n: Semi-espaco inferior           │  pi_x^(n) = Tx^(n) * exp(-s_n*(z-z_{n-1}))
               │  (sigma_hn, sigma_vn)                  │  -> Apenas onda transmitida descendente
  z = +inf     └────────────────────────────────────────┘
```

**As 6 zonas e suas expressoes:**

| Zona | Condicao | Descricao | Forma de `pi_x` |
|:-----|:---------|:----------|:----------------|
| 0 | z < z_0 | Semi-espaço superior | `Tx^(0) * exp(s_0*z)` |
| k | z_{k-1} <= z < z_k, k < l | Camadas acima da fonte | Transmitida + refletida superiormente |
| l (z < h_0) | z_{l-1} <= z < h_0 | Camada fonte, receptor acima | Transmitida + 2 reflexoes (Mx_up, Mx_dw) |
| l (z > h_0) | h_0 <= z < z_l | Camada fonte, receptor abaixo | Transmitida + 2 reflexoes (Mx_up, Mx_dw) |
| j | z_{j-1} <= z < z_j, j > l | Camadas abaixo da fonte | Transmitida + refletida inferiormente |
| n | z >= z_{n-1} | Semi-espaço inferior | `Tx^(n) * exp(-s_n*(z-z_{n-1}))` |

**Correspondencia com os 6 casos em `hmd_TIV_optimized`:**

Estas 6 zonas correspondem **diretamente** aos 6 blocos `if/elseif` na sub-rotina
`hmd_TIV_optimized` em `magneticdipoles.f08`:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Zona Teorica         │  Caso no Código        │  Condicao Fortran │
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

As expressoes para `pi_u` são inteiramente analogas, substituindo `s` por `u`,
`R_TM` por `R_TE`, e os fatores `Mx` por `Eu`.

**Coeficientes de onda transmitida na camada fonte:**

Na camada `l` que contém a fonte, os coeficientes de transmissão são derivados
diretamente das solucoes do meio ilimitado:

```
Tx_up^(l) = Tx_dw^(l) = m_x / (2*s_l)       (modo TM)
Tu_up^(l) = +m_x / 2                          (modo TE, direcao ascendente)
Tu_dw^(l) = -m_x / 2                          (modo TE, direcao descendente)
```

O sinal oposto em `Tu_up` e `Tu_dw` reflete a antissimetria do modo TE em relação
a fonte. No código Fortran, estes valores aparecem na inicializacao dos arrays
`Txdw`, `Txup`, `Tudw`, `Tuup` na sub-rotina `hmd_TIV_optimized`.

### 4.6 Correspondencia Potenciais <-> Modos TE/TM

O campo eletromagnético, expresso nas equações espectrais, pode ser escrito em
função apenas de `pi_x` e `pi_u`. A separação em modos de propagação e:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Potencial  │  Modo   │  Governa         │  Sensivel a          │
  ├─────────────┼─────────┼──────────────────┼──────────────────────┤
  │  pi_x       │  TM     │  E_z, H_x, H_y  │  sigma_h E sigma_v  │
  │  pi_u       │  TE     │  H_z, E_x, E_y  │  Apenas sigma_h     │
  └─────────────┴─────────┴──────────────────┴──────────────────────┘

Justificativa:
  - E_z = zeta * lambda^2 * (i*k_y) * pi_x   -> depende SOMENTE de pi_x
  - H_z = i*k_x * pi_u                       -> depende SOMENTE de pi_u
```

**Impedancia e admitancia intrinsecas:**

Associadas a cada modo, definem-se as grandezas intrinsecas de cada camada `m`:

```
Impedancia intrinseca (modo TM):
  Z_m = s_m / sigma_h,m     onde s_m = lambda_m * v_m

Admitancia intrinseca (modo TE):
  Y_m = u_m / zeta          onde u_m = sqrt(kr^2 - kh^2_m)
```

**Coeficientes de reflexão:**

Os coeficientes de reflexão em cada interface são definidos pela diferença entre a
grandeza intrinseca da camada e a grandeza aparente vista atraves das camadas adjacentes:

```
Modo TM (reflexao descendente na interface inferior da camada m):
  R_TM_dw^(m) = (Z_m - Z_tilde_dw^(m+1)) / (Z_m + Z_tilde_dw^(m+1))

Modo TM (reflexao ascendente na interface superior da camada m):
  R_TM_up^(m) = (Z_m - Z_tilde_up^(m-1)) / (Z_m + Z_tilde_up^(m-1))

Modo TE (reflexao descendente):
  R_TE_dw^(m) = (Y_m - Y_tilde_dw^(m+1)) / (Y_m + Y_tilde_dw^(m+1))

Modo TE (reflexao ascendente):
  R_TE_up^(m) = (Y_m - Y_tilde_up^(m-1)) / (Y_m + Y_tilde_up^(m-1))
```

**Formula recursiva para impedancia/admitancia aparente (tanh):**

```
Impedancia aparente descendente:
  Z_tilde_dw^(n) = Z_n                               (semi-espaco inferior)
  Z_tilde_dw^(m) = Z_m * [Z_tilde_dw^(m+1) + Z_m * tanh(s_m*h_m)]
                         / [Z_m + Z_tilde_dw^(m+1) * tanh(s_m*h_m)]

Impedancia aparente ascendente:
  Z_tilde_up^(0) = Z_0                               (semi-espaco superior)
  Z_tilde_up^(m) = Z_m * [Z_tilde_up^(m-1) + Z_m * tanh(s_m*h_m)]
                         / [Z_m + Z_tilde_up^(m-1) * tanh(s_m*h_m)]

As admitancias aparentes seguem formulas identicas com Y_m, u_m em lugar de Z_m, s_m.
```

Para as camadas extremas (semi-espacos 0 e n), os coeficientes de reflexão são nulos:
`R_TM_up^(0) = R_TM_dw^(n) = R_TE_up^(0) = R_TE_dw^(n) = 0`.

### 4.7 Fatores de Onda na Camada do Transmissor

Na camada `l` que contém a fonte, as reflexoes multiplas entre as interfaces superior
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
reflete as condições de contorno distintas: a componente `E_z` (TM) e continua na
interface, enquanto `H_z` (TE) sofre uma descontinuidade proporcional a corrente
magnética superficial equivalente.

### 4.8 Coeficientes de Transmissão entre Camadas

Quando o receptor esta em uma camada diferente da fonte, os coeficientes de transmissão
são calculados recursivamente a partir da camada da fonte, utilizando as condições de
continuidade dos campos tangenciais nas interfaces.

**Condições de continuidade (no dominio espectral):**

```
Em cada interface z = z_j, para meios nao-magneticos:

  (a) zeta * pi_z^(j)|_{z_j} = zeta * pi_z^(j+1)     -> pi_z continuo
  (b) d(pi_z)/dz|_{z_j} continuo
  (c) zeta * d(pi_x)/dz|_{z_j} continuo               -> d(pi_x)/dz continuo
  (d) sigma_h,j * pi_x^(j)|_{z_j} = sigma_h,j+1 * pi_x^(j+1)
  (e) pi_u^(j)|_{z_j} = pi_u^(j+1)                    -> pi_u continuo
  (f) d(pi_u)/dz|_{z_j} continuo
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

**Casos especiais para semi-espacos:**

```
Semi-espaco superior (zona 0):
  Tx^(0) = (s_1/s_0) * Tx^(1) * exp(-s_1*h_1) * (1 - R_TM_up^(1))
  Tu^(0) = Tu^(1) * exp(-u_1*h_1) * (1 - R_TE_up^(1))

Semi-espaco inferior (zona n):
  Tx^(n) = (s_{n-1}/s_n) * Tx^(n-1) * exp(-s_{n-1}*h_{n-1}) * (1 - R_TM_dw^(n-1))
  Tu^(n) = Tu^(n-1) * exp(-u_{n-1}*h_{n-1}) * (1 + R_TE_dw^(n-1))
```

No código Fortran (`hmd_TIV_optimized`), esses coeficientes são computados nos arrays
`Txdw(:,j)`, `Txup(:,k)`, `Tudw(:,j)`, `Tuup(:,k)` via loops recursivos.

### 4.9 Campo no Dominio Espacial via Transformada de Hankel

As expressoes finais do campo magnético H no dominio espacial são obtidas aplicando a
transformada inversa de Fourier dupla (em `k_x, k_y`) as expressoes espectrais. Devido
a simetria cilindrica do meio, essa transformada dupla reduz-se a integrais de Hankel
com funções de Bessel `J_0` e `J_1`.

**Campo espectral (resultado da Seção 4.2):**

```
H_x_hat_hat = -(k_y^2/kr^2) * kh^2 * pi_x + (k_x^2/kr^2) * d(pi_u)/dz
H_y_hat_hat = (k_x*k_y/kr^2) * kh^2 * pi_x + (k_x*k_y/kr^2) * d(pi_u)/dz
H_z_hat_hat = i*k_x * pi_u
```

**Transformada de Hankel — DMH_x (Dipolo Magnetico Horizontal x):**

Para o DMH_x, as componentes do campo no dominio espacial envolvem convolucoes com
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
  Ktm = kernel modo TM (depende de pi_x, Tx, reflexoes TM)
  Kte = kernel modo TE (depende de pi_u, Tu, reflexoes TE)
  Kte_dz = derivada do kernel TE em z
  wJ0, wJ1 = pesos do filtro de Hankel para J_0 e J_1
  r = sqrt(x^2 + y^2) = distancia horizontal T-R
  kr = abscissa do filtro / r
```

**DMH_y (Dipolo Magnetico Horizontal y):**

O campo do dipolo y e obtido por rotação de 90 graus do DMH_x:

```
Regra de rotacao: x -> y, y -> -x

H_x(DMH_y) = H_y(DMH_x)  (com a substituicao de variaveis)
H_y(DMH_y) = expressao com (2*y^2/r^2 - 1) e x^2/r^2 permutados
H_z(DMH_y) = -y * sum(Kte * wJ1 * kr^2) / (r * 2*pi*r)
```

No código, o modo `'hmdxy'` calcula ambos os dipolos simultaneamente,
reutilizando os kernels espectrais.

**DMV (Dipolo Magnetico Vertical):**

O DMV excita apenas o modo TE (simetria axial elimina o acoplamento TM):

```
H_x = -x * sum(Kte_dzz * J1 * kr^2) / (2*pi*r) / r
H_y = -y * sum(Kte_dzz * J1 * kr^2) / (2*pi*r) / r
H_z = sum(Kte_z * J0 * kr^3) / (2*pi*zeta) / r

Onde Kte_dzz = AdmInt * Kte_z  (kernel segunda derivada em z com J1)
```

### 4.10 Mapeamento Formulação Teórica <-> Código Fortran

A tabela abaixo estabelece a correspondencia direta entre as variáveis da formulação
teórica (conforme o documento TeX) e as variáveis do código Fortran:

| Teoria (TeX) | Fortran | Sub-rotina | Tipo Fortran | Descricao |
|:-------------|:--------|:-----------|:-------------|:----------|
| `zeta = i*omega*mu_0` | `zeta` | `fieldsinfreqs` | `complex(dp)` | Impeditividade |
| `sigma_h, sigma_v` | `eta(i,1), eta(i,2)` | `fieldsinfreqs` | `real(dp)` | Condutividades (1/rho) |
| `kh^2 = -zeta*sigma_h` | `kh2(j)` | `commonarraysMD` | `complex(dp)` | Número de onda horiz. ao quadrado |
| `kv^2 = -zeta*sigma_v` | `kv2(j)` | `commonarraysMD` | `complex(dp)` | Número de onda vert. ao quadrado |
| `lambda^2 = sigma_h/sigma_v` | `lamb2(j)` | `commonarraysMD` | `real(dp)` | Coeficiente de anisotropia ao quadrado |
| `u_m = sqrt(kr^2 - kh^2)` | `u(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Constante de propagação TE |
| `v_m = sqrt(kr^2 - kv^2)` | `v(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Constante de propagação intermediaria |
| `s_m = lambda_m * v_m` | `s(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Constante de propagação TM |
| `Y_m = u_m / zeta` | `AdmInt(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Admitancia intrinseca (TE) |
| `Z_m = s_m / sigma_h,m` | `ImpInt(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Impedancia intrinseca (TM) |
| `tanh(u_m * h_m)` | `tghuh(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Tanh estabilizada (TE) |
| `tanh(s_m * h_m)` | `tghsh(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Tanh estabilizada (TM) |
| `R_TE_up^(m)` | `RTEup(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Coeficiente reflexão TE ascendente |
| `R_TE_dw^(m)` | `RTEdw(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Coeficiente reflexão TE descendente |
| `R_TM_up^(m)` | `RTMup(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Coeficiente reflexão TM ascendente |
| `R_TM_dw^(m)` | `RTMdw(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Coeficiente reflexão TM descendente |
| `Y_tilde_dw^(m)` | `AdmAp_dw(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Admitancia aparente descendente |
| `Y_tilde_up^(m)` | `AdmAp_up(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Admitancia aparente ascendente |
| `Z_tilde_dw^(m)` | `ImpAp_dw(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Impedancia aparente descendente |
| `Z_tilde_up^(m)` | `ImpAp_up(:,m)` | `commonarraysMD` | `complex(dp), (npt,n)` | Impedancia aparente ascendente |
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
| `R^T * H * R` | `RtHR` | `utils.f08` | Sub-rotina | Rotacao do tensor para frame da ferramenta |
| `h_0` (posição da fonte) | `h0` | `fieldsinfreqs` | `real(dp)` | Profundidade do transmissor |
| `kr` (número de onda radial) | `kr(:)` | `commonarraysMD` | `real(dp), (npt)` | `absc(i) / hordist` |
| `h_m` (espessura camada m) | `h(m)` | `sanitize_hprof_well` | `real(dp)` | Espessura da camada m |
| `z_m` (interface inferior m) | `prof(m)` | `sanitize_hprof_well` | `real(dp)` | Profundidade da interface m |

---

## 5. Arquitetura do Software

### 5.1 Diagrama de Módulos (v8.0)

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                      Simulador PerfilaAnisoOmp v8.0                        │
  │                                                                             │
  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
  │  │  parameters.f08 │    │  filtersv2.f08  │    │  utils.f08              │ │
  │  │  (constantes,   │    │  (filtros       │    │  (sanitize, RtHR,      │ │
  │  │   tipos dp,     │    │   Hankel J0/J1, │    │   findlayers, thread_  │ │
  │  │   thread_       │    │   Werthmuller   │    │   workspace type)      │ │
  │  │   workspace)    │    │   201 pontos)   │    │                         │ │
  │  └────────┬────────┘    └────────┬────────┘    └────────────┬────────────┘ │
  │           │                      │                          │              │
  │           └──────────────────────┼──────────────────────────┘              │
  │                                  │                                         │
  │                    ┌─────────────┴───────────────┐                        │
  │                    │  magneticdipoles.f08         │                        │
  │                    │  (commonarraysMD,            │                        │
  │                    │   commonfactorsMD,           │                        │
  │                    │   hmd_TIV_optimized[_ws],   │                        │
  │                    │   vmd_optimized[_ws])        │                        │
  │                    └─────────────┬───────────────┘                        │
  │                                  │                                         │
  │           ┌──────────────────────┼──────────────────────────┐              │
  │           │                      │                          │              │
  │  ┌────────┴────────┐   ┌────────┴────────┐   ┌─────────────┴───────────┐ │
  │  │PerfilaAniso     │   │ RunAnisoOmp.f08 │   │ tatu_f2py_wrapper.f08  │ │
  │  │ Omp.f08         │   │ (main program,  │   │ (f2py interface,       │ │
  │  │(perfila1Daniso  │   │  leitura        │   │  simulate() retorna    │ │
  │  │ OMP, fieldsin   │   │  model.in,      │   │  arrays NumPy direto,  │ │
  │  │ freqs_cached_ws │   │  despacho)      │   │  sem I/O de disco)     │ │
  │  │ writes_files)   │   │                 │   │                         │ │
  │  └────────┬────────┘   └────────┬────────┘   └─────────────┬───────────┘ │
  │           │                      │                          │              │
  │           └──────────────────────┼──────────────────────────┘              │
  │                                  │                                         │
  │                           ┌──────┴──────┐                                 │
  │                           │   tatu.x    │   <- Executavel (make)          │
  │                           │ tatu_f2py.so│   <- Modulo Python (make f2py)  │
  │                           └──────┬──────┘                                 │
  │                                  │                                         │
  └──────────────────────────────────┼─────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
           ┌───────┴──────┐  ┌──────┴──────┐  ┌─────┴──────────┐
           │fifthBuildTIV │  │  model.in   │  │ batch_runner.py│
           │Models.py     │  │  (entrada)  │  │ (multi-process │
           │(gerador,     │  │             │  │  paralelo)     │
           │ subprocess)  │  │             │  │                │
           └──────────────┘  └─────────────┘  └────────────────┘
```

### 5.2 Fluxo de Execução (v8.0 — Multi-TR)

```
fifthBuildTIVModels.py
  │
  ├── Gera modelo geologico i (Sobol QMC)
  ├── Escreve model.in (com nTR e dTR(1:nTR))
  ├── subprocess.run(['./tatu.x'])
  │     │
  │     └── RunAnisoOmp.f08 (main)
  │           ├── Le model.in (nf, freq, ntheta, theta, h1, tj, p_med, nTR, dTR, ...)
  │           └── call perfila1DanisoOMP(...)
  │                 │
  │                 ├── [1x] J0J1Wer(npt=201) -> filtros Hankel
  │                 ├── [1x] sanitize_hprof_well -> h(:), prof(0:n)
  │                 ├── [1x] Aloca ws_pool(0:maxthreads-1)  [Fase 3]
  │                 ├── [1x] Aloca 9 caches (npt,n,nf)      [Fase 4]
  │                 ├── [1x] Computa eta_shared(n,2)         [Fase 4/B2]
  │                 │
  │                 └── do itr = 1, nTR    ← NOVO: loop Multi-TR
  │                       │
  │                       ├── [parallel if ntheta>1] do k = 1, ntheta
  │                       │     │
  │                       │     ├── Lsen = dTR(itr) * sin(theta_k)
  │                       │     ├── Lcos = dTR(itr) * cos(theta_k)
  │                       │     ├── r_k  = dTR(itr) * |sin(theta_k)|
  │                       │     │
  │                       │     ├── [serial, 1x/freq] commonarraysMD -> 9 caches
  │                       │     │
  │                       │     └── [parallel guided,16] do j = 1, nmed(k)
  │                       │           ├── Calcula posições T/R para dTR(itr)
  │                       │           ├── fieldsinfreqs_cached_ws(ws_pool(tid), caches, ...)
  │                       │           │     ├── findlayersTR2well -> camadT, camadR
  │                       │           │     └── do i = 1, nf
  │                       │           │           ├── commonfactorsMD(caches_i)
  │                       │           │           ├── hmd_TIV_optimized_ws(ws, caches_i)
  │                       │           │           ├── vmd_optimized_ws(ws, caches_i)
  │                       │           │           └── RtHR(ang, matH) -> cH(i,:)
  │                       │           └── z_rho1(j) = zrho; c_H1(j) = cH
  │                       │
  │                       └── writes_files(..., itr, nTR)
  │                             │
  │                             ├── nTR==1: filename.dat       (sem sufixo)
  │                             └── nTR>1:  filename_TR{itr}.dat (com sufixo)
  │
  └── Valida .dat gerados (NaN/Inf check por par T-R)
```

### 5.3 Sub-rotinas Principais (v8.0)

| Sub-rotina | Módulo | Descricao |
|:-----------|:-------|:----------|
| `perfila1DanisoOMP` | PerfilaAnisoOmp.f08 | Orquestrador principal: loop Multi-TR x theta x medidas |
| `fieldsinfreqs` | PerfilaAnisoOmp.f08 | Versão original (preservada para referência) |
| `fieldsinfreqs_ws` | PerfilaAnisoOmp.f08 | Versão Fase 3 com workspace (preservada para rollback) |
| **`fieldsinfreqs_cached_ws`** | PerfilaAnisoOmp.f08 | **Versão produção** Fase 4: caches + workspace |
| `writes_files` | PerfilaAnisoOmp.f08 | Escrita .dat/.out com logica Multi-TR (sufixo _TR{itr}) |
| `commonarraysMD` | magneticdipoles.f08 | Pre-computa u, s, RTEdw/up, RTMdw/up, AdmInt |
| `commonfactorsMD` | magneticdipoles.f08 | Fatores de onda na camada do transmissor |
| `hmd_TIV_optimized_ws` | magneticdipoles.f08 | HMD com workspace pré-alocado |
| `vmd_optimized_ws` | magneticdipoles.f08 | VMD com workspace pré-alocado |
| `sanitize_hprof_well` | utils.f08 | Sanitizacao de espessuras e profundidades |
| `findlayersTR2well` | utils.f08 | Determinacao das camadas do T e R |
| `RtHR` | utils.f08 | Rotacao do tensor H para frame da ferramenta |
| `J0J1Wer` | filtersv2.f08 | Filtro Werthmuller 201 pontos (J0 e J1) |
| `simulate` | tatu_f2py_wrapper.f08 | Interface f2py: retorna arrays NumPy direto |

---

## 6. Módulos Fortran — Análise Detalhada

### 6.1 parameters.f08 — Constantes e Tipos

Define as constantes físicas e o tipo de precisão dupla:

```fortran
integer, parameter :: dp = selected_real_kind(15, 307)  ! double precision
real(dp), parameter :: pi = 3.14159265358979323846d0
real(dp), parameter :: mu = 4.d0 * pi * 1.d-7          ! permeabilidade magnetica
real(dp), parameter :: eps = 1.d-9                      ! tolerancia numerica
real(dp), parameter :: del = 0.1d0                      ! tolerancia angular (graus)
```

Tambem define o `type :: thread_workspace` (Fase 3 + 3b) com 12 campos pré-alocados por thread:

```fortran
type :: thread_workspace
  ! Fase 3 — arrays de transmissao (npt x n)
  complex(dp), allocatable :: Tudw(:,:), Txdw(:,:)
  complex(dp), allocatable :: Tuup(:,:), Txup(:,:)
  complex(dp), allocatable :: TEdwz(:,:), TEupz(:,:)
  ! Fase 3b — fatores de onda de commonfactorsMD (npt)
  complex(dp), allocatable :: Mxdw(:), Mxup(:)
  complex(dp), allocatable :: Eudw(:), Euup(:)
  complex(dp), allocatable :: FEdwz(:), FEupz(:)
  ! Sentinel para cache de commonfactorsMD
  integer :: last_camadT = -1
end type
```

### 6.2 filtersv2.f08 — Filtros de Hankel

Contem os coeficientes tabelados dos filtros de Hankel para as funções de Bessel J0 e J1. O filtro Werthmuller de 201 pontos (`npt = 201`) e utilizado em produção:

```fortran
subroutine J0J1Wer(npt, absc, wJ0, wJ1)
  ! Retorna abscissas e pesos para J0 e J1
  ! npt = 201 pontos (filtro Werthmuller)
```

### 6.3 utils.f08 — Utilidades

| Sub-rotina | Descricao |
|:-----------|:----------|
| `sanitize_hprof_well` | Converte espessuras em profundidades com sentinelas |
| `findlayersTR2well` | Encontra camadas do T e R dado (Tz, z) e prof(0:n) |
| `RtHR` | Rotacao R^T * H * R para frame da ferramenta |

### 6.4 magneticdipoles.f08 — Dipolos Magneticos

Módulo central que implementa o cálculo do campo EM. Contem:

| Sub-rotina | Linhas | Descricao |
|:-----------|:------:|:----------|
| `commonarraysMD` | ~200 | Pre-computa arrays comuns (u, s, uh, sh, RTE, RTM, AdmInt) |
| `commonfactorsMD` | ~100 | Fatores de onda na camada do transmissor (Mx, Eu, FE) |
| `hmd_TIV_optimized` | ~400 | Campo HMD completo (6 zonas, kernels TM+TE, Hankel) |
| `hmd_TIV_optimized_ws` | ~400 | Versão com workspace (Fase 3) |
| `vmd_optimized` | ~200 | Campo VMD (modo TE puro, Hankel) |
| `vmd_optimized_ws` | ~200 | Versão com workspace (Fase 3) |

### 6.5 PerfilaAnisoOmp.f08 — Módulo Principal

#### 6.5.1 perfila1DanisoOMP — Orquestrador

**Assinatura (v8.0 — Multi-TR):**

```fortran
subroutine perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ntheta, theta, h1, tj, &
                             nTR, dTR, p_med, n, resist, esp, filename)
  integer, intent(in) :: modelm, nmaxmodel, nf, ntheta, n, nTR
  real(dp), intent(in) :: freq(nf), theta(ntheta), h1, tj, dTR(nTR), p_med, resist(n,2), esp(n)
  character(*), intent(in) :: mypath, filename
```

**Parâmetros novos (v8.0):**
- `nTR` (integer): número de pares T-R (1 para backward-compatible)
- `dTR(nTR)` (real64 array): espassamentos T-R em metros

**Fluxo de execução detalhado:**

1. Calcula `nmed(k)` para cada ângulo theta(k)
2. Inicializa filtro Hankel (`J0J1Wer`)
3. Sanitiza espessuras (`sanitize_hprof_well`)
4. Configura OpenMP: `omp_set_max_active_levels(2)`, particionamento multiplicativo
5. Aloca `ws_pool(0:maxthreads-1)` com 12 campos por thread (Fase 3 + 3b)
6. Aloca 9 caches `(npt, n, nf)` para commonarraysMD (Fase 4)
7. Computa `eta_shared(n, 2) = 1/resist` uma única vez (debito B2)
8. **Loop Multi-TR:** `do itr = 1, nTR`
   - Loop ângulos: `do k = 1, ntheta` (parallel se ntheta > 1)
     - Computa `r_k = dTR(itr) * |sin(theta_k)|`
     - Pre-computa `commonarraysMD` serial, 1x por frequência -> caches
     - Loop medidas: `do j = 1, nmed(k)` (parallel guided,16)
       - Calcula posições T/R com `Lsen = dTR(itr) * sin(theta_k)`
       - Chama `fieldsinfreqs_cached_ws(ws_pool(tid), caches, ...)`
   - Chama `writes_files(..., itr, nTR)` apos cada par T-R

#### 6.5.2 fieldsinfreqs — Versão Original (preservada)

Sub-rotina original que calcula campos EM para todas as frequências em uma posição de medição. Fluxo:

```
fieldsinfreqs(ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, n, h, prof, resist, zrho, cH)
  │
  ├── findlayersTR2well → camadT, camadR, layerObs
  │
  └── do i = 1, nf
        ├── eta = 1/resist (recomputada a cada chamada — ineficiente, corrigido em B2)
        ├── commonarraysMD(n, npt, r, kr, zeta, h, eta) → u, s, ...
        ├── commonfactorsMD(Tz, camadT, u, s, ...) → Mxdw, Mxup, ...
        ├── hmd_TIV_optimized(x, y, z, camadR, camadT, ...) → matH(1:2,:)
        ├── vmd_optimized(x, y, z, camadR, camadT, ...) → matH(3,:)
        └── RtHR(ang, 0, 0, matH) → cH(f,:) = [tH(1,1)..tH(3,3)]
```

#### 6.5.2b fieldsinfreqs_ws — Versão com Workspace (Fase 3, preservada)

Copia de `fieldsinfreqs` que recebe `type(thread_workspace), intent(inout) :: ws` como primeiro argumento e delega para `hmd_TIV_optimized_ws` e `vmd_optimized_ws` em vez das rotinas originais. **Preservada para rollback** — substituida por `fieldsinfreqs_cached_ws` na Fase 4.

#### 6.5.2c fieldsinfreqs_cached_ws — Versão com Cache (Fase 4, produção)

Sub-rotina **ativa em produção** pos-Fase 4. Recebe os 9 caches de `commonarraysMD` como `intent(in)` e **não chama** `commonarraysMD` internamente — os arrays `u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt` são pré-computados pelo caller (`perfila1DanisoOMP`, uma vez por ângulo `k` e par T-R `itr`).

**Assinatura:**

```fortran
subroutine fieldsinfreqs_cached_ws(ws, ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, &
                                    n, h, prof, resist, eta_in,                      &
                                    u_c, s_c, uh_c, sh_c,                            &
                                    RTEdw_c, RTEup_c, RTMdw_c, RTMup_c, AdmInt_c,    &
                                    zrho, cH)
  type(thread_workspace), intent(inout) :: ws
  ! ... (21 argumentos intent(in) + 2 intent(out))
  complex(dp), dimension(npt,n,nf), intent(in) :: u_c, s_c, uh_c, sh_c, ...
```

**Cadeia de chamadas (produção pos-Fase 4):**

```
perfila1DanisoOMP
  │
  ├── [serial, 1x por (itr, k)] commonarraysMD → u_cache, s_cache, ...
  │
  └── [parallel, nmed iteracoes] fieldsinfreqs_cached_ws(ws_pool(tid), ...)
        │
        ├── findlayersTR2well → camadT, camadR
        │
        └── do i = 1, nf
              ├── commonfactorsMD(Tz, camadT, u_c(:,:,i), ...) → ws%Mxdw, ...
              ├── hmd_TIV_optimized_ws(ws, ..., u_c(:,:,i), ..., ws%Mxdw, ...)
              ├── vmd_optimized_ws(ws, ..., u_c(:,:,i), ..., ws%FEdwz, ...)
              └── RtHR(ang, 0, 0, matH) → cH(i,:)
```

**Diferencas vs `fieldsinfreqs` original:**
1. `commonarraysMD` **eliminada** — caches passados como argumentos `intent(in)`
2. `eta` não recomputada — recebida como `eta_in` (debito B2 resolvido)
3. Slices `krwJ0J1(:,1)`, `krwJ0J1(:,2)`, `krwJ0J1(:,3)` passadas diretamente (debito B1)
4. Fatores de onda `Mxdw..FEupz` usam `ws%Mxdw` etc. (Fase 3b — heap, não stack)
5. `ws` (thread_workspace) contém 12 campos pré-alocados (Fases 3 + 3b)

**Custo por chamada (Fase 4 + 3b + 2b):**
- `commonfactorsMD`: 14 `exp()` complexos x 201 pontos = 2.814 operações transcendentais
- `hmd_TIV_optimized_ws`: ~31 KB de kernel com 9 somas de Hankel
- `vmd_optimized_ws`: ~15 KB de kernel com 3 somas de Hankel
- **Total**: ~68 KB de computação por chamada x 600 medidas x 2 frequências = ~81 MB/modelo

#### 6.5.3 writes_files — Escrita de Saída (v8.0 — Multi-TR)

**Assinatura (v8.0):**

```fortran
subroutine writes_files(modelm, nmaxmodel, mypath, zrho, cH, nt, theta, nf, freq, &
                        nmeds, filename, itr, nTR)
  integer, intent(in) :: modelm, nmaxmodel, nt, nf, itr, nTR
```

**Parâmetros novos (v8.0):**
- `itr` (integer): índice do par T-R atual (1..nTR)
- `nTR` (integer): número total de pares T-R

**Logica de nomeacao de arquivos (Multi-TR):**

```fortran
if (nTR > 1) then
  write(tr_suffix, '(A,I0)') '_TR', itr
  fileTR = mypath // trim(adjustl(filename)) // trim(tr_suffix) // '.dat'
else
  fileTR = mypath // trim(adjustl(filename)) // '.dat'
end if
```

**Tabela de decisão para nomeacao:**

```
  ┌──────────┬──────────┬──────────────────────────────────────────┐
  │  nTR     │  itr     │  Nome do arquivo .dat                    │
  ├──────────┼──────────┼──────────────────────────────────────────┤
  │  1       │  1       │  filename.dat             (sem sufixo)   │
  │  3       │  1       │  filename_TR1.dat                        │
  │  3       │  2       │  filename_TR2.dat                        │
  │  3       │  3       │  filename_TR3.dat                        │
  └──────────┴──────────┴──────────────────────────────────────────┘
```

**Arquivo .out (metadados):** Escrito apenas quando `modelm == nmaxmodel` (último modelo):

```
Linha 1: nt nf nmaxmodel     (num. angulos, num. frequencias, num. modelos)
Linha 2: theta(1) ... theta(nt)   (angulos)
Linha 3: freq(1) ... freq(nf)    (frequencias)
Linha 4: nmeds(1) ... nmeds(nt)  (num. medições por angulo)
```

**Arquivo .dat (dados binarios):** Escrito em modo `stream` (sem registros fixos). A abertura usa logica condicional (debito B1 corrigido):

```
  ┌─────────────────────┬─────────────────────┬──────────────────────────┐
  │ modelm              │ arquivo existe?     │ acao                      │
  ├─────────────────────┼─────────────────────┼──────────────────────────┤
  │ == 1 (1o do lote)   │ não importa         │ status='replace' (wipe)   │
  │ > 1 (subsequente)   │ sim                 │ status='old' + append     │
  │ > 1 (subsequente)   │ não                 │ status='replace' (safe)   │
  └─────────────────────┴─────────────────────┴──────────────────────────┘
```

### 6.6 RunAnisoOmp.f08 — Programa Principal

Programa sequêncial que:
1. Obtem o diretorio corrente via `getcwd`
2. Le o arquivo `model.in` sequêncialmente (formato v8.0 com nTR)
3. Aloca arrays de resistividade, espessura e `dTR_arr(nTR)`
4. Chama `perfila1DanisoOMP(...)` com todos os parâmetros incluindo `nTR` e `dTR_arr`

**Trecho da leitura Multi-TR (v8.0):**

```fortran
! Leitura de multiplos pares T-R (Feature 1):
read(11,*) nTR
allocate(dTR_arr(nTR))
do i = 1, nTR
  read(11,*) dTR_arr(i)
end do
```

### 6.7 tatu_f2py_wrapper.f08 — Interface f2py (v8.0)

Módulo wrapper que expoe o simulador EM 1D TIV a Python via f2py (NumPy). Retorna os arrays de saída diretamente (sem I/O de disco).

**Assinatura:**

```fortran
subroutine simulate(nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med, &
                    n, resist, esp, nmmax, zrho_out, cH_out)
  ! OUTPUT:
  !   zrho_out(nTR, ntheta, nmmax, nf, 3) : resistividades aparentes (real64)
  !   cH_out(nTR, ntheta, nmmax, nf, 9)   : tensor EM completo (complex128)
```

**Uso em Python:**

```python
import tatu_f2py
zrho, cH = tatu_f2py.simulate(nf, freq, ntheta, theta, h1, tj,
                               nTR, dTR, p_med, n, resist, esp, nmmax)
```

**Caracteristicas:**
- Replica exata da logica de `perfila1DanisoOMP` (Fases 3/3b/4/5b)
- Suporta Multi-TR nativo (loop `do itr = 1, nTR`)
- Configuração OpenMP identica (workspace pool, caches, guided scheduling)
- `nmmax` deve ser pre-calculado pelo caller Python como `max(ceil(tj/(p_med*cos(theta_i*pi/180))))`
- Eliminacao total de I/O de disco — arrays retornados via f2py bindings

---

## 7. Arquivo de Entrada model.in

### 7.1 Formato v8.0 (Multi-TR)

O arquivo `model.in` e lido sequêncialmente pelo programa principal `RunAnisoOmp.f08`. A versão 8.0 introduz o campo `nTR` (número de pares T-R) seguido de `nTR` valores de `dTR`:

```
Linha  Variavel            Tipo      Descricao
─────  ─────────────────  ────────  ────────────────────────────────────────────
  1    nf                 integer   Numero de frequencias
  2    freq(1)            real(dp)  Frequencia 1 (Hz)
  ...  freq(nf)           real(dp)  Frequencia nf (Hz)
  +1   ntheta             integer   Numero de angulos de inclinacao
  +2   theta(1)           real(dp)  Angulo 1 (graus, 0-90)
  ...  theta(ntheta)      real(dp)  Angulo ntheta (graus)
  +1   h1                 real(dp)  Altura do 1o ponto-medio T-R (m, acima da 1a interface)
  +1   tj                 real(dp)  Tamanho da janela de investigacao (m)
  +1   p_med              real(dp)  Passo entre medições (m)
  +1   nTR                integer   ** NOVO v8.0: Numero de pares T-R **
  +2   dTR(1)             real(dp)  ** NOVO v8.0: Distancia T-R 1 (m) **
  ...  dTR(nTR)           real(dp)  ** NOVO v8.0: Distancia T-R nTR (m) **
  +1   filename           char      Nome base dos arquivos de saida
  +1   ncam               integer   Numero total de camadas (incluindo semi-espacos)
  +2   resist(1,1:2)      real(dp)  rho_h, rho_v da camada 1 (semi-espaco superior)
  ...  resist(ncam,1:2)   real(dp)  rho_h, rho_v da camada ncam
  +1   esp(2)             real(dp)  Espessura da camada 2 (m)
  ...  esp(ncam-1)        real(dp)  Espessura da camada ncam-1 (m)
  +1   modelm nmaxmodel   integer   Modelo atual e total de modelos
```

### 7.2 Comparacao: Formato Antigo (v7.0-) vs Novo (v8.0)

**Formato antigo (nTR=1 implícito):**

```
2                    ! nf = 2 frequencias
20000.0              ! freq(1) = 20 kHz
40000.0              ! freq(2) = 40 kHz
1                    ! ntheta = 1 angulo
0.0                  ! theta(1) = 0 graus
10.0                 ! h1 = 10 m
120.0                ! tj = 120 m
0.2                  ! p_med = 0.2 m
1.0                  ! dTR = 1.0 m  ← escalar unico
Inv0_15Dip1000_t5    ! nome dos arquivos
10                   ! ncam = 10 camadas
...
```

**Formato novo (v8.0 — nTR explícito, backward-compatible):**

```
1                    ! nf = 1 frequencia
20000.0              ! freq(1) = 20 kHz
1                    ! ntheta = 1 angulo
0.0                  ! theta(1) = 0 graus
10.0                 ! h1 = 10 m
120.0                ! tj = 120 m
0.2                  ! p_med = 0.2 m
3                    ! nTR = 3 pares T-R      ← NOVO: inteiro
1.0                  ! dTR(1) = 1.0 m         ← NOVO: 1o espacamento
2.0                  ! dTR(2) = 2.0 m         ← NOVO: 2o espacamento
3.0                  ! dTR(3) = 3.0 m         ← NOVO: 3o espacamento
Inv0_Dip3000_teste   ! nome dos arquivos
20                   ! ncam = 20 camadas
...
```

**Backward-compatibility:** Para `nTR = 1`, o formato e equivalente ao antigo — a única diferença e que o valor `1` precede o único `dTR`. O comportamento de saída (arquivo único sem sufixo) e identico.

### 7.3 Exemplo Comentado (model.in atual — 3 pares T-R)

O arquivo `model.in` atualmente no repositorio demonstra o formato Multi-TR:

```
1                 !numero de frequencias
20000.0           !frequencia 1
1                 !numero de angulos de inclinacao
0.0               !angulo 1
10.0              !altura do primeiro ponto-medio T-R, acima da primeira interface de camadas
120.0             !tamanho da janela de investigacao
0.2               !passo entre as medidas
3                 !numero de pares T-R          ← nTR = 3
1.0               !distancia T-R 1              ← dTR(1) = 1.0 m
2.0               !distancia T-R 2              ← dTR(2) = 2.0 m
3.0               !distancia T-R 3              ← dTR(3) = 3.0 m
Inv0_Dip3000_teste              !nome dos arquivos de saida
20                !numero de camadas
1346.34    1346.47     !resistividades horizontal e vertical
1466.19    1466.42
201.62    201.65
42.25    42.26
0.7    0.7
0.39    0.39
192.7    192.71
2.94    2.94
87.52    87.53
15.64    15.64
0.32    0.32
61.31    61.32
54.51    54.51
1.28    1.28
15.83    15.83
0.1    0.1
0.2    0.2
5.51    5.51
4.83    4.83
72.84    72.85
9.52              !espessuras das n-2 camadas
5.42
3.18
1.35
7.5
1.21
0.5
3.41
5.45
0.5
0.5
37.57
2.38
5.26
3.2
4.92
7.63
0.5
3000 3000         !modelo atual e o numero maximo de modelos
```

### 7.4 Observacoes Importantes

1. **Espessuras:** As camadas 1 e ncam são semi-espacos (espessura infinita), então `esp(1) = esp(ncam) = 0` e atribuido em `RunAnisoOmp.f08` antes de chamar `perfila1DanisoOMP`.

2. **Frequências:** O simulador padrao usa 1-2 frequências (20 kHz e/ou 40 kHz). A primeira e a mesma do pipeline (`FREQUENCY_HZ = 20000.0`).

3. **Número de medições:** Para `theta = 0`, `pz = p_med = 0.2`, `nmed = ceil(120/0.2) = 600`, consistente com `SEQUENCE_LENGTH = 600` do pipeline.

4. **Anisotropia neste exemplo:** O modelo mostra anisotropia muito fraca (`lambda ~ 1.0` em quase todas as camadas). Em cenarios reais, `lambda` varia de 1.0 a sqrt(2).

5. **Multi-TR e cache:** Cada par T-R exige recomputo completo dos caches Fase 4 porque `commonarraysMD` depende de `r_k = dTR(itr) * |sin(theta)|`, que varia com o espaçamento. Para `nTR = 3`, o custo total e ~3x o de `nTR = 1` (linear em nTR).

---

## 8. Arquivos de Saída (.dat e .out)

### 8.1 Arquivo .out — Metadados

Arquivo texto com 4 linhas, escrito apenas pelo último modelo da serie:

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

**Nota v8.0:** O arquivo .out e único por execução (independente de nTR), pois os metadados de ângulos/frequências/medidas são os mesmos para todos os pares T-R.

### 8.2 Arquivo .dat — Dados Binarios

Arquivo binario no formato Fortran `stream` (sem registros fixos), escrito em modo `append`.

**Convenção de nomeacao (v8.0 — Multi-TR):**

```
  ┌──────────┬───────────────────────────────────────────────────────┐
  │  nTR     │  Arquivos gerados                                     │
  ├──────────┼───────────────────────────────────────────────────────┤
  │  1       │  filename.dat                  (sem sufixo)           │
  │  2       │  filename_TR1.dat, filename_TR2.dat                   │
  │  3       │  filename_TR1.dat, filename_TR2.dat, filename_TR3.dat │
  └──────────┴───────────────────────────────────────────────────────┘

Exemplo concreto com filename = "Inv0_Dip3000_teste" e nTR = 3:
  Inv0_Dip3000_teste_TR1.dat   (dTR = 1.0 m)
  Inv0_Dip3000_teste_TR2.dat   (dTR = 2.0 m)
  Inv0_Dip3000_teste_TR3.dat   (dTR = 3.0 m)
```

**Formato por registro (22 colunas):**

| Posicao | Variavel | Tipo | Bytes | Coluna | Descricao |
|:--------|:---------|:-----|------:|:------:|:----------|
| 0 | i | int32 | 4 | 0 | Índice da medição |
| 4 | zobs | float64 | 8 | 1 | Profundidade do ponto-medio T-R (m) |
| 12 | rho_h | float64 | 8 | 2 | Resistividade horizontal verdadeira (Ohm.m) |
| 20 | rho_v | float64 | 8 | 3 | Resistividade vertical verdadeira (Ohm.m) |
| 28 | Re(Hxx) | float64 | 8 | 4 | Parte real de H(1,1) |
| 36 | Im(Hxx) | float64 | 8 | 5 | Parte imaginaria de H(1,1) |
| 44 | Re(Hxy) | float64 | 8 | 6 | Parte real de H(1,2) |
| 52 | Im(Hxy) | float64 | 8 | 7 | Parte imaginaria de H(1,2) |
| 60 | Re(Hxz) | float64 | 8 | 8 | Parte real de H(1,3) |
| 68 | Im(Hxz) | float64 | 8 | 9 | Parte imaginaria de H(1,3) |
| 76 | Re(Hyx) | float64 | 8 | 10 | Parte real de H(2,1) |
| 84 | Im(Hyx) | float64 | 8 | 11 | Parte imaginaria de H(2,1) |
| 92 | Re(Hyy) | float64 | 8 | 12 | Parte real de H(2,2) |
| 100 | Im(Hyy) | float64 | 8 | 13 | Parte imaginaria de H(2,2) |
| 108 | Re(Hyz) | float64 | 8 | 14 | Parte real de H(2,3) |
| 116 | Im(Hyz) | float64 | 8 | 15 | Parte imaginaria de H(2,3) |
| 124 | Re(Hzx) | float64 | 8 | 16 | Parte real de H(3,1) |
| 132 | Im(Hzx) | float64 | 8 | 17 | Parte imaginaria de H(3,1) |
| 140 | Re(Hzy) | float64 | 8 | 18 | Parte real de H(3,2) |
| 148 | Im(Hzy) | float64 | 8 | 19 | Parte imaginaria de H(3,2) |
| 156 | Re(Hzz) | float64 | 8 | 20 | Parte real de H(3,3) |
| 164 | Im(Hzz) | float64 | 8 | 21 | Parte imaginaria de H(3,3) |

**Total por registro:** 4 + 21 x 8 = **172 bytes** (1 int32 + 21 float64 = 22 valores)

**Mapeamento de colunas para o tensor H(3x3):**

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Tensor H(3x3)   │  Colunas .dat                                      │
  ├───────────────────┼────────────────────────────────────────────────────┤
  │  H(1,1) = Hxx    │  Col 4  (Re) + Col 5  (Im)                        │
  │  H(1,2) = Hxy    │  Col 6  (Re) + Col 7  (Im)                        │
  │  H(1,3) = Hxz    │  Col 8  (Re) + Col 9  (Im)                        │
  │  H(2,1) = Hyx    │  Col 10 (Re) + Col 11 (Im)                        │
  │  H(2,2) = Hyy    │  Col 12 (Re) + Col 13 (Im)                        │
  │  H(2,3) = Hyz    │  Col 14 (Re) + Col 15 (Im)                        │
  │  H(3,1) = Hzx    │  Col 16 (Re) + Col 17 (Im)                        │
  │  H(3,2) = Hzy    │  Col 18 (Re) + Col 19 (Im)                        │
  │  H(3,3) = Hzz    │  Col 20 (Re) + Col 21 (Im)                        │
  ├───────────────────┼────────────────────────────────────────────────────┤
  │  INPUT_FEATURES   │  [1, 4, 5, 20, 21] = zobs, Re(Hxx), Im(Hxx),     │
  │  (pipeline P1)    │                       Re(Hzz), Im(Hzz)            │
  │  OUTPUT_TARGETS   │  [2, 3] = rho_h, rho_v                            │
  └───────────────────┴────────────────────────────────────────────────────┘
```

### 8.3 Leitura no Pipeline Python (geosteering_ai)

O pipeline Python (`geosteering_ai/data/loading.py`) le o .dat interpretando os registros binarios de 172 bytes:

```python
# Pseudocódigo ilustrativo (NÃO é o código real de loading.py):
dtype = np.dtype([
    ('meds', np.int32),       # Col 0: indice de medicao
    ('values', np.float64, 21) # Cols 1-21: zobs, rho_h, rho_v, 18 EM
])
data = np.fromfile(filepath, dtype=dtype)

# Reorganizacao em 22 colunas:
# Col 0:  meds (metadata)
# Col 1:  zobs -> INPUT_FEATURE
# Col 2:  rho_h -> OUTPUT_TARGET
# Col 3:  rho_v -> OUTPUT_TARGET
# Col 4:  Re(Hxx) -> INPUT_FEATURE
# Col 5:  Im(Hxx) -> INPUT_FEATURE
# ...
# Col 20: Re(Hzz) -> INPUT_FEATURE
# Col 21: Im(Hzz) -> INPUT_FEATURE
```

### 8.4 Tamanho dos Arquivos

Para 1000 modelos, 1 ângulo, 2 frequências, 600 medições:

```
Tamanho por par T-R = nmodels x ntheta x nfreq x nmeds x 172 bytes
                    = 1000 x 1 x 2 x 600 x 172
                    = 206,400,000 bytes ~= 197 MB

Para nTR = 3 pares T-R:
  Total = 3 x 197 MB ~= 591 MB (3 arquivos .dat independentes)

Para nTR = 1 com nf = 1 (configuracao atual):
  Total = 1000 x 1 x 1 x 600 x 172 = 103,200,000 bytes ~= 98 MB
```

### 8.5 Validação dos Arquivos .dat (Multi-TR aware)

O script `fifthBuildTIVModels.py` inclui um bloco de válidação pos-geração que verifica cada arquivo .dat gerado:

```python
dtyp = np.dtype([('col0', np.int32)] +
                [('col{}'.format(i), np.float64) for i in range(1, 22)])

if isinstance(dTR, (list, tuple, np.ndarray)):
    dat_files = [
        (mypath + filename + f'_TR{itr + 1}.dat', f'TR{itr + 1} (dTR={d} m)')
        for itr, d in enumerate(dTR)
    ]
else:
    dat_files = [(mypath + filename + '.dat', f'TR1 (dTR={dTR} m)')]

for dat_path, label in dat_files:
    mydat = np.fromfile(dat_path, dtype=dtyp)
    myarr = np.array(mydat.tolist())
    has_nan = np.isnan(myarr).any()
    has_inf = np.isinf(myarr).any()
    # Reporta status [OK] ou [ATENCAO] por par T-R
```

---

## 9. Sistema de Build (Makefile)

### 9.1 Estrutura do Makefile (v8.0)

O Makefile segue a convenção padrao de projetos Fortran com secoes adicionais para f2py e válidação numérica:

| Seção | Descricao |
|:------|:----------|
| SETTINGS | Binario (`tatu.x`), extensões, diretorio de build (`./build`) |
| DEPENDENCIES | Ordem de compilação: `parameters -> filtersv2 -> utils -> magneticdipoles -> PerfilaAnisoOmp -> RunAnisoOmp` |
| LINKER WORKAROUND | Workaround para macOS 26+ / Darwin 25+ (ld-classic) |
| COMPILER | `gfortran` com 3 conjuntos de flags |
| TARGETS | `$(binary)`, `f2py_wrapper`, `debug_O0`, `run_python`, `all`, `clean`, `cleanall` |

### 9.2 Flags de Compilação

**Flags de desenvolvimento (comentadas, para debug):**

```makefile
-g             # Simbolos de debug
-fcheck=all    # Verificacao de bounds, overflow, etc.
-fbacktrace    # Stack trace em caso de erro
```

Para ativar o modo de desenvolvimento, descomente `flags = $(development_flags_gfortran)` e comente `flags = $(production_flags_gfortran)` no Makefile.

**Flags de produção (ativas):**

```makefile
-J$(build)     # Diretorio para arquivos .mod
-std=f2008     # Padrao Fortran 2008
-pedantic      # Avisos de conformidade com o padrao
-Wall -Wextra  # Todos os avisos
-Wimplicit-interface  # Erro em interfaces implicitas
-fPIC          # Posicao independente (permite linking dinamico)
-fmax-errors=1 # Para na primeira falha
-O3            # Otimizacao agressiva
-march=native  # Otimizacao para a arquitetura local
-ffast-math    # Otimizacoes matematicas agressivas
-funroll-loops # Desenrola loops para otimização
-fall-intrinsics # Habilita todas as funções intrinsecas
```

**Flags de válidação numérica (debug_O0):**

```makefile
-O0            # Sem otimização (determinismo bit-a-bit)
-g             # Simbolos de debug
-fno-fast-math # Sem reordenamento FP
-fsignaling-nans # Sinaliza NaN em operações
```

**Flags OpenMP:** `-fopenmp` adicionada tanto na compilação quanto na linkagem.

### 9.3 Workaround macOS (Darwin 25+)

O Makefile inclui um workaround para o linker do macOS 26+ (ld-1266.8) que falha com `!tapi-tbd` para x86_64:

```makefile
ifeq ($(shell uname -s),Darwin)
  _LD_CLASSIC := /Library/Developer/CommandLineTools/usr/bin/ld-classic
  ifneq ($(wildcard $(_LD_CLASSIC)),)
    $(shell printf '#!/bin/sh\nexec $(_LD_CLASSIC) "$$@"\n' > $(build)/ld && chmod +x $(build)/ld)
    LDFLAGS_EXTRA := -B$(build)/
  endif
endif
```

**Funcionamento:** Cria um wrapper `build/ld` que redireciona para `ld-classic` e usa `-B$(build)/` para que o `collect2` do gfortran encontre o wrapper antes do `ld` do sistema. Em Linux ou quando `ld-classic` não existe, `LDFLAGS_EXTRA` fica vazio (noop).

### 9.4 Targets (v8.0)

| Target | Comando | Descricao |
|:-------|:--------|:----------|
| `$(binary)` (`tatu.x`) | `make` | Compila e linka todos os .o em um executavel |
| **`f2py_wrapper`** | `make f2py_wrapper` | **NOVO v8.0:** Gera módulo `tatu_f2py.so` importavel em Python |
| **`debug_O0`** | `make debug_O0` | **NOVO v7.0:** Rebuild com `-O0 -fno-fast-math` para válidação numérica |
| `run_python` | `make run_python` | Executa `fifthBuildTIVModels.py` (gerador de modelos) |
| `all` | `make all` | Compila + executa Python |
| `clean` | `make clean` | Remove `./build/` |
| `cleanall` | `make cleanall` | Remove `./build/`, `tatu.x`, `*.dat`, `*.out` |

**Target f2py_wrapper — detalhes:**

```makefile
f2py_wrapper: $(object_files)
    cd $(build) && python3 -m numpy.f2py -c -m tatu_f2py ../tatu_f2py_wrapper.f08 \
        parameters.$(obj_ext) filtersv2.$(obj_ext) utils.$(obj_ext) \
        magneticdipoles.$(obj_ext) PerfilaAnisoOmp.$(obj_ext) \
        --f90flags="-fopenmp $(production_flags_gfortran)" \
        -lgomp $(LDFLAGS_EXTRA) \
        && mv tatu_f2py*.so ../
```

**Pre-requisitos:** f2py (parte do NumPy) e compilador Fortran com OpenMP. Os objetos `.o` devem estar compilados previamente via `make $(binary)` para gerar os `.mod` necessários.

**Resultado:** `tatu_f2py.so` (ou `.dylib` no macOS) no diretorio raiz, importavel como `import tatu_f2py`.

**Target debug_O0 — detalhes:**

```makefile
debug_O0:
    $(MAKE) clean
    $(MAKE) flags="$(debug_O0_flags_gfortran)" $(binary)
```

**Uso:** `make clean && make debug_O0`. Produz binario determinístico bit-a-bit identico entre versoes matemáticamente equivalentes do código. Qualquer divergencia de MD5 entre duas builds `debug_O0` revela um problema matematico real, não um artefato do compilador (como reordenamento associativo de ponto flutuante com `-ffast-math`).

### 9.5 Ordem de Compilação

```
parameters.f08 ──> build/parameters.o
filtersv2.f08  ──> build/filtersv2.o
utils.f08      ──> build/utils.o      (depende de parameters)
magneticdipoles.f08 ──> build/magneticdipoles.o (depende de parameters)
PerfilaAnisoOmp.f08 ──> build/PerfilaAnisoOmp.o (depende de todos acima + omp_lib)
RunAnisoOmp.f08 ──> build/RunAnisoOmp.o (depende de parameters, DManisoTIV)

Linkagem: gfortran -fopenmp [flags] $(LDFLAGS_EXTRA) -o tatu.x build/*.o

f2py (target separado):
  tatu_f2py_wrapper.f08 + build/*.o ──> tatu_f2py.so (via numpy.f2py)
```

### 9.6 Notas e Recomendacoes

- **`-ffast-math`** pode afetar a precisão de operações com NaN e infinitos. Para válidação numérica, use `make debug_O0`.
- **`-march=native`** otimiza para a CPU local, mas o binario não e portavel.
- O diretorio `./build` e criado automáticamente via `$(shell mkdir -p $(build))`.
- O workaround macOS e automático e não afeta builds em Linux.
- O target `f2py_wrapper` depende dos objetos pre-compilados — execute `make` antes de `make f2py_wrapper`.

---

## 10. Gerador de Modelos Geológicos (Python)

### 10.1 Visão Geral

O script `fifthBuildTIVModels.py` gera modelos geológicos estocásticos usando amostragem Sobol Quasi-Monte Carlo (`scipy.stats.qmc.Sobol`). Na execução direta (`__main__`), os modelos são escritos sequencialmente em `model.in` e simulados pelo Fortran via subprocess. Na execução em lote paralela — orquestrada por `batch_runner.py` —, os dicionários de modelos são gerados centralmente na memória e distribuídos entre múltiplos processos worker, cada um rodando o Fortran em sandbox isolada. A versão v8.0+ suporta geração com múltiplos pares T-R simultâneos.

### 10.2 Parâmetros dos Modelos Geológicos

| Parâmetro | Range | Distribuição | Descrição |
|:----------|:------|:------------|:----------|
| `n_layers` | 3-80 | Empírica (ponderada) ou uniforme | Número de camadas |
| `rho_h` | 0,05–1500 Ω·m | Log-uniforme (Sobol) | Resistividade horizontal |
| `lambda` | 1,0–√2 | Uniforme (Sobol), correlacionada com rho_h | Coeficiente de anisotropia TIV |
| `rho_v` | calculado | `lambda^2 * rho_h` | Resistividade vertical |
| `espessuras` | 0,1–50+ m | Sobol + stick-breaking | Espessuras das camadas internas |

### 10.3 Cenários de Geração (8 Geradores)

As proporções abaixo refletem o mix definido em `batch_runner.py` → `fabricar_universos_aleatorios()`.
A soma é 100% (com arredondamento de `int()` aplicado sobre `ntmodels`).

| Cenário | Função | Proporção | ncam | Espessuras | Contrastes | Ruído |
|:--------|:-------|:---------:|:----:|:-----------|:-----------|:------|
| **Baseline empírico** | `baseline_empirical_2` | **15 %** | 3-30 (pesos empíricos) | Standard (min 1,0 m) | Natural | Não |
| **Baseline uniforme ncam** | `baseline_ncamuniform_2` | **10 %** | 3-80 (uniforme) | Standard (min 0,5 m) | Natural | Não |
| **Camadas grossas** | `baseline_thick_thicknesses_2` | **10 %** | 3-14 | Grossas (min 10 m, p=0,7) | Forçados em grossas | Não |
| **Desfavorável empírico** | `unfriendly_empirical_2` | **12,5 %** | 3-30 (pesos) | Finas (min 0,2 m, p=0,6) | Forçados (5×, p=0,5) | Não |
| **Desfavorável ruidoso** | `unfriendly_noisy_2` | **17,5 %** | 3-30 (pesos) | Finas + ruído (3 %) | Forçados + ruído (5 %) | Sim |
| **Patológico standard** | `generate_pathological_models_2` | **10 %** | 3-28 | Muito finas (min 0,1 m) | Extremos (10×, p=0,7) | Sim (7 %) |
| **Patológico isotrópico/grosso** | `generate_pathological_models_2` (λ≈1, thick) | **12,5 %** | 3-28 | Grossas (min 10 m) | Extremos (p=0,3) | Sim (7 %) |
| **Baseline extremamente isotrópico** | `baseline_empirical_2` (λ≈1, log-uni) | **12,5 %** | 3-30 (empírico) | Standard (min 0,5 m) | Natural | Não |

**Notas sobre os cenários extremos:**

- **Patológico isotrópico/grosso (m7):** `rho_h` ∈ [0,1–1500] Ω·m, `lambda` ∈ [1, 1,001],
  `min_thickness_internal = 10 m`, `p_camfina = 0,3` — modela formações quase-isotrópicas
  com camadas espessas, frequentes em coquinas e carbonatos maciços.

- **Baseline extremamente isotrópico (m8):** `rho_h` ∈ [0,1–1500] Ω·m via `rhohdistr='loguni'`,
  `lambda` ∈ [1, 1,0001] — anisotropia praticamente nula (λ → 1). Força a rede a aprender
  que ρ_v ≈ ρ_h é um estado físico válido, evitando viés em direção a modelos anisotrópicos.

### 10.4 Funções Auxiliares e Vantagens do Sobol QMC

| Função | Descrição |
|:-------|:----------|
| `log_transform_2` | Transforma Sobol [0,1] para escala log-uniforme [min, max] |
| `uniform_transform_2` | Transforma Sobol [0,1] para escala linear [min, max] |
| `generate_thicknesses_2` | Sobol stick-breaking com espessura mínima garantida |
| `generate_thick_thicknesses_2` | Forçamento de camadas grossas (>15 m) com probabilidade `p_thick` |
| `generate_thin_thicknesses_2` | Forçamento de camadas finas (~mín) com probabilidade `p_thin` |
| `conditional_rho_h_sampling_core_2` | Força contrastes entre camadas adjacentes |
| `conditional_rho_h_with_thickness_2` | Força resistividades extremas em camadas grossas |
| `correlated_lambda_sampling_core_2` | Lambda correlacionado com resistividade (ρ alto → λ alto) |

**Sobol Quasi-Monte Carlo vs Pseudo-Aleatório:**

A escolha de sequências Sobol (quasi-aleatórias) em vez de geradores pseudo-aleatórios
tradicionais (como NumPy `np.random`) é deliberada e traz vantagens significativas para
a geração de modelos geológicos:

```
  Pseudo-aleatório (Monte Carlo):         Sobol (Quasi-Monte Carlo):
  ┌────────────────────────────┐          ┌────────────────────────────┐
  │ . . .  .    .   .  .      │          │ .   .   .   .   .   .     │
  │    .  .    . . .   .   .  │          │   .   .   .   .   .   .   │
  │  .  .  ...   .      . .   │          │ .   .   .   .   .   .     │
  │    .     .  .   ..  .     │          │   .   .   .   .   .   .   │
  │  .  .  .  .      . .  .   │          │ .   .   .   .   .   .     │
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

### 10.5 Configuração Multi-TR no Gerador (v8.0)

A variável `dTR` no gerador controla o número e os valores dos pares T-R. A lógica suporta tanto escalar (backward-compatible) quanto lista (Multi-TR):

**Definição da variável:**

```python
# Multi-TR: lista de espacamentos (nTR = len(dTR))
dTR = [1.0, 2.0, 3.0]

# Single-TR: escalar (backward-compatible, nTR = 1)
# dTR = 1.0
```

**Lógica de escrita do model.in (isinstance dispatch):**

```python
# Feature 1 (Multi-TR): escreve nTR + nTR valores de dTR
# Para nTR=1, backward-compatible com formato anterior
if isinstance(dTR, (list, tuple, np.ndarray)):
    nTR_val = len(dTR)
    f.write(str(nTR_val) + '                 ' + '!numero de pares T-R' + '\n')
    for itr_idx, dtr_val in enumerate(dTR):
        f.write(str(dtr_val) + '               ' + '!distancia T-R ' + str(itr_idx+1) + '\n')
else:
    f.write('1                 ' + '!numero de pares T-R' + '\n')
    f.write(str(dTR) + '               ' + '!distancia T-R 1' + '\n')
```

**Tabela de decisão (isinstance dispatch):**

```
  ┌──────────────────────────────┬──────────┬──────────────────────────────┐
  │  Tipo de dTR                 │  nTR     │  Linhas escritas no model.in │
  ├──────────────────────────────┼──────────┼──────────────────────────────┤
  │  float (1.0)                 │  1       │  "1\n1.0\n"                  │
  │  list ([1.0, 2.0, 3.0])     │  3       │  "3\n1.0\n2.0\n3.0\n"       │
  │  tuple ((0.5, 1.0))         │  2       │  "2\n0.5\n1.0\n"            │
  │  np.ndarray ([1.0, 2.0])    │  2       │  "2\n1.0\n2.0\n"            │
  └──────────────────────────────┴──────────┴──────────────────────────────┘
```

### 10.6 Fluxo do Gerador

```
Para cada modelo i = 1..nmodels:
  1. Sorteia ncam (numero de camadas)
  2. Gera amostra Sobol de dimensao (ncam + ncam + ncamint-1)
  3. Fatia amostra em: rho_h_portion, lambda_portion, thickness_portion
  4. Transforma rho_h (log-uniforme)
  5. Aplica forcamento condicional de contrastes
  6. Transforma lambda (uniforme, correlacionada)
  7. Calcula rho_v = lambda^2 * rho_h
  8. Gera espessuras (stick-breaking + min)
  9. Escreve model.in (com nTR e dTR(1:nTR))
  10. Executa tatu.x via subprocess
  11. Resultado: append no(s) .dat (1 por par T-R)
```

### 10.7 Loop de Execução via Subprocess

O gerador Python orquestra a execução do simulador Fortran via `subprocess.run()`.
Para cada modelo geológico gerado, o fluxo é:

```python
# Pseudocódigo do loop de execução (simplificado):
for i in range(1, nmodels + 1):
    # 1. Gera parametros do modelo i
    ncam, rho_h, rho_v, espessuras = generate_model(i, ...)

    # 2. Escreve model.in com os parametros (v8.0: inclui nTR e dTR)
    write_model_in(
        nf=1, freqs=[20000.0],
        ntheta=1, thetas=[0.0],
        h1=10.0, tj=120.0, p_med=0.2,
        dTR=[1.0, 2.0, 3.0],   # <- Multi-TR: lista
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
        timeout=300  # 5 minutos por modelo (seguranca)
    )

    # 4. Verifica sucesso (exibe info OpenMP apenas no 1o modelo)
    if i == 1 and result.stdout.strip():
        print(result.stdout.strip())

    # O resultado e appendado ao(s) .dat automaticamente pelo Fortran
```

**Detalhes importantes do subprocess:**

- O Fortran lê `model.in` do diretório corrente e escreve `.dat` e `.out` no
  mesmo diretório. O `cwd=fortran_dir` garante que os arquivos sejam encontrados.
- O modo `append` do Fortran (`position='append'`) acumula os resultados de todos
  os modelos em um único arquivo `.dat` por par T-R, sem necessidade de concatenação posterior.
- O `.out` (metadados) é escrito apenas quando `modelm == nmaxmodel`, então não é
  sobrescrito a cada modelo.
- Para 1000 modelos com ~2,4 s/modelo, o tempo total é ~40 minutos (single-thread)
  ou ~5–10 minutos com OpenMP em 8 cores.
- **Com nTR = 3**, o custo por modelo é ~3× maior (linear em nTR), resultando em
  ~7,2 s/modelo e ~120 minutos para 1000 modelos (single-thread).

### 10.8 Validação Multi-TR pos-geração

Após o loop de geração, o script coleta e valida todos os arquivos `.dat` gerados, com consciência do formato Multi-TR:

```python
# Coleta arquivos .dat gerados (Multi-TR aware)
if isinstance(dTR, (list, tuple, np.ndarray)):
    dat_files = [
        (mypath + filename + f'_TR{itr + 1}.dat', f'TR{itr + 1} (dTR={d} m)')
        for itr, d in enumerate(dTR)
    ]
else:
    dat_files = [(mypath + filename + '.dat', f'TR1 (dTR={dTR} m)')]

# Validacao: NaN, Inf, shape por par T-R
for dat_path, label in dat_files:
    mydat = np.fromfile(dat_path, dtype=dtyp)
    myarr = np.array(mydat.tolist())
    has_nan = np.isnan(myarr).any()
    has_inf = np.isinf(myarr).any()
    status = 'OK' if not has_nan and not has_inf else 'ATENCAO'
    print(f'  [{status}] {label}')
    print(f'         Arquivo  : {dat_path}')
    print(f'         Registros: {myarr.shape[0]} | NaN: {has_nan} | Inf: {has_inf}')
```

**Saída esperada (3 pares T-R, 3000 modelos, nf=1, nmed=600):**

```
Validacao dos arquivos .dat gerados (3 par(es) T-R):
-----------------------------------------------------------------------------------------------------
  [OK] TR1 (dTR=1.0 m)
         Arquivo  : .../Inv0_Dip3000_teste_TR1.dat
         Registros: 1800000 | NaN: False | Inf: False
  [OK] TR2 (dTR=2.0 m)
         Arquivo  : .../Inv0_Dip3000_teste_TR2.dat
         Registros: 1800000 | NaN: False | Inf: False
  [OK] TR3 (dTR=3.0 m)
         Arquivo  : .../Inv0_Dip3000_teste_TR3.dat
         Registros: 1800000 | NaN: False | Inf: False
-----------------------------------------------------------------------------------------------------
```

### 10.9 Orquestrador de Geração em Lote — `batch_runner.py` (v3.0)

#### 10.9.1 Visão Geral e Paradigma

O `batch_runner.py` implementa o paradigma **"Plano Central, Execução Paralela"**:
todos os modelos geológicos são gerados _na memória_ pelo processo principal (usando
`fifthBuildTIVModels`), depois distribuídos entre `N` workers via
`concurrent.futures.ProcessPoolExecutor`. Cada worker roda o Fortran em um sandbox
completamente isolado — sem compartilhamento de arquivos de entrada ou saída.

```
  ┌────────────────────────────────────────────────────────────────────┐
  │                       batch_runner.py v3.0                        │
  │                                                                    │
  │  main()                                                            │
  │    │                                                               │
  │    ├─ fabricar_universos_aleatorios(ntotal, dTR)                   │
  │    │     └─ 8 geradores → lista de nmodels dicts na memória       │
  │    │                                                               │
  │    ├─ dividir_em_batches(all_models, n_workers)                    │
  │    │     └─ [batch_0, batch_1, ..., batch_{N-1}]                  │
  │    │                                                               │
  │    └─ ProcessPoolExecutor(max_workers=N)                           │
  │          ├─ worker 0: tempdir_0 → tatu.x → w0_TR1.dat ...         │
  │          ├─ worker 1: tempdir_1 → tatu.x → w1_TR1.dat ...         │
  │          └─ worker N-1: tempdir_{N-1} → tatu.x → w{N-1}_TR1.dat  │
  │                                                                    │
  │    merge_outputs() → [TR1.dat, TR2.dat, TR3.dat] + validação      │
  └────────────────────────────────────────────────────────────────────┘
```

#### 10.9.2 Interface de Linha de Comando (CLI)

```
python batch_runner.py [OPÇÕES]

Opções obrigatórias:
  --nmodels INT         Número total de modelos a gerar (ex.: 100000)
  --output STR          Prefixo dos arquivos de saída (ex.: Inv0_Dip)

Opções de paralelismo:
  --workers INT         Número de processos worker (default: cpu_count // OMP_threads)
  --omp-threads INT     Threads OpenMP por worker (default: 2)
                        Regra: workers × omp_threads ≤ cpu_count físico

Opções de localização:
  --tatu STR            Caminho do binário tatu.x (default: ./tatu.x)
  --model-in STR        Template model.in de referência (default: ./model.in)
  --output-dir STR      Diretório de saída dos .dat (default: ./)

Opções de física:
  --dtr FLOAT [FLOAT …] Espaçamentos T-R em metros (default: 1.0)
                        Ex: --dtr 1.0 2.0 3.0  →  Multi-TR com 3 pares

Exemplo típico (8 workers, 2 threads OMP, 16 cores totais):
  python batch_runner.py \
      --nmodels 200000 \
      --output Inv0_Dip \
      --workers 8 \
      --omp-threads 2 \
      --dtr 1.0 2.0 3.0
```

#### 10.9.3 Fluxo Interno Detalhado

**Fase 1 — Geração centralizada de modelos:**

```python
# fabricar_universos_aleatorios(ntmodels, dTR)
# Retorna: lista de dicts com chaves [ncam, rho_h, rho_v, espessuras, dTR]

m1 = fbt.baseline_empirical_2(nmodels=int(ntmodels * 0.15))       # 15 %
m2 = fbt.baseline_ncamuniform_2(nmodels=int(ntmodels * 0.10))     # 10 %
m3 = fbt.baseline_thick_thicknesses_2(nmodels=int(ntmodels * 0.10)) # 10 %
m4 = fbt.unfriendly_empirical_2(nmodels=int(ntmodels * 0.125))    # 12,5 %
m5 = fbt.unfriendly_noisy_2(nmodels=int(ntmodels * 0.175))        # 17,5 %
m6 = fbt.generate_pathological_models_2(nmodels=int(ntmodels*0.10)) # 10 %
m7 = fbt.generate_pathological_models_2(                           # 12,5 %
         nmodels=int(ntmodels*0.125),
         rho_h_min=0.1, rho_h_max=1500,
         lambda_min=1, lambda_max=1.001,
         min_thickness_internal=10, p_camfina=0.3)
m8 = fbt.baseline_empirical_2(                                     # 12,5 %
         nmodels=int(ntmodels*0.125),
         rho_h_min=0.1, rho_h_max=1500, rhohdistr='loguni',
         lambda_min=1, lambda_max=1.0001, min_thickness=0.5)
all_models = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8
random.shuffle(all_models)   # embaralha para quebrar correlação entre workers
```

**Fase 2 — Execução paralela em sandbox:**

Cada worker recebe `(batch, worker_id, tatu_abs, model_in_header, output_dir, dTR)` e:

1. Cria `tempfile.TemporaryDirectory()` exclusivo
2. Copia `tatu.x` para o sandbox (`shutil.copy2`)
3. Para cada modelo no batch: reescreve `model.in` via `write_model_in()`, executa
   `subprocess.run([tatu_dst], cwd=tmpdir, timeout=120)` — timeout 120 s
   (≫60× o tempo esperado de ~0,06 s/modelo)
4. Os `.dat` gerados no sandbox são renomeados com prefixo `w{worker_id}_` e
   movidos para `output_dir`

**Fase 3 — Fusão e validação:**

```python
# merge_outputs(output_dir, n_workers, n_tr, prefix)
# Ordena por worker_id → concatena binários → valida com np.memmap

for tr_idx in range(n_tr):
    worker_files = sorted(glob(f"{output_dir}/w*_TR{tr_idx+1}.dat"))
    with open(f"{prefix}_TR{tr_idx+1}.dat", 'wb') as fout:
        for wf in worker_files:
            with open(wf, 'rb') as fin:
                shutil.copyfileobj(fin, fout)   # sem buffer extra em RAM

# Validação via memmap (O(1) RAM independente do tamanho do arquivo):
mmap = np.memmap(merged_path, dtype=dtype_22col, mode='r')
has_nan = np.isnan(mmap.view(np.float64)).any()
has_inf = np.isinf(mmap.view(np.float64)).any()
```

#### 10.9.4 Isolamento de Sandbox

Cada worker opera em um `tempfile.TemporaryDirectory` independente. Isso garante:

- **Sem condição de corrida:** workers nunca lêem/escrevem no mesmo `model.in`
- **Limpeza automática:** o diretório temporário é destruído ao sair do contexto,
  mesmo em caso de exceção no worker
- **Portabilidade:** funciona em qualquer sistema de arquivos (NFS, tmpfs, SSD)

```
  Worker 0          Worker 1          Worker N-1
  /tmp/tatu_abc/    /tmp/tatu_def/    /tmp/tatu_xyz/
  ├─ tatu.x (cópia) ├─ tatu.x (cópia) ├─ tatu.x (cópia)
  ├─ model.in       ├─ model.in       ├─ model.in
  └─ TR1.dat        └─ TR1.dat        └─ TR1.dat
       ↓                  ↓                  ↓
  w0_TR1.dat        w1_TR1.dat        w{N-1}_TR1.dat
                  (output_dir)
```

#### 10.9.5 Bugs Conhecidos e Limitações

Os itens abaixo foram identificados durante revisão de código mas **não afetam a
correção dos dados gerados** nas condições normais de uso. São registrados aqui
para orientar futuras refatorações.

| ID | Severidade | Descrição | Mitigação atual |
|:---|:----------:|:----------|:----------------|
| **B10** | Média | `run_model_batch` usa `os.getcwd()` internamente; se o diretório de trabalho mudar entre o processo principal e o worker, o caminho pode ser incorreto. | Usar `--output-dir` com caminho absoluto; o `main()` chama `os.path.abspath()` antes de passar para o pool. |
| **B11** | Alta | Ausência de `try/except` em torno de `future.result()` no loop de coleta de resultados. Uma exceção não capturada em qualquer worker propaga-se e interrompe toda a geração sem salvar os dados já produzidos. | Encerrar o processo de forma limpa e reiniciar com `--workers` reduzido; dados parciais ficam em `output_dir` com prefixo `w*_`. |
| **B18** | Baixa | `parse_model_in` usa `idx += 3` para pular a seção de ângulos — hardcoded, frágil se o header mudar de formato. Além disso, `except Exception: return None` silencia todos os erros de parsing. | Qualquer alteração no header do `model.in` requer atualização manual de `parse_model_in`. |

**Limitação de qualidade QMC (não é bug, é decisão de design):**

O `fifthBuildTIVModels.py` instancia um novo `qmc.Sobol(scramble=True)` por modelo
e sorteia apenas 1 amostra. Isso descarta a propriedade de baixa discrepância do QMC
(que só se manifesta com N amostras consecutivas do mesmo instanciador). Na prática,
o comportamento é equivalente a pseudo-aleatório com overhead adicional. Para restaurar
o QMC verdadeiro, seria necessário agrupar modelos por `ncam` (mesma dimensão) e gerar
`N` amostras de um único instanciador por grupo.

#### 10.9.6 Desempenho Observado

| Configuração | Modelos/h | Observações |
|:-------------|:---------:|:------------|
| 1 worker, 1 OMP thread (baseline) | ~24.000 | Single-thread puro |
| 8 workers, 2 OMP threads (16 cores) | **~58.856** | Medido em benchmark oficial |
| 16 workers, 1 OMP thread (16 cores) | ~52.000 | Sub-ótimo: overhead de IPC |
| 4 workers, 4 OMP threads (16 cores) | ~48.000 | OMP sobre-subscrito |

A configuração ótima é `workers × omp_threads ≈ n_cores_físicos` com
`omp_threads ∈ [2, 4]`. Valores maiores de `omp_threads` sofrem com overhead de
sincronização OpenMP; valores menores deixam núcleos ociosos entre subprocess calls.

---

---

## 11. Recursos Avançados do Simulador

> **Versão 10.0 (Abril 2026):** Esta seção documenta os oito recursos avançados do simulador
> `PerfilaAnisoOmp` para suporte ao pipeline Geosteering AI v2.0: múltiplos pares T-R (v7.0),
> tensor completo 9 componentes (v7.0), interface f2py para Python (v7.0), execução batch
> paralela de modelos (v7.0), **frequências arbitrárias F5 (v8.0)**, **antenas inclinadas F7 (v8.0)**,
> **compensação midpoint F6 e Filtro Adaptativo (v9.0)** e **sensibilidades ∂H/∂ρ F10 (v10.0)**.
> Cada recurso está validado com bit-equivalência ao baseline original quando operando em modo
> retrocompatível (todos os flags opcionais == 0).

---

### 11.1 Múltiplos Pares Transmissor-Receptor (Feature 1)

#### 11.1.1 Motivação Física

Em perfilagem LWD (Logging While Drilling), a **profundidade de investigação** (DOI) de uma ferramenta EM é proporcional ao skin depth e ao espaçamento T-R:

```
DOI ~ sqrt(rho / (2 * pi * f * mu)) * g(L / delta)
```

onde `rho` é a resistividade da formacao, `f` a frequência, `mu` a permeabilidade magnética, `L` o espaçamento transmissor-receptor e `delta` o skin depth. A função `g(L/delta)` modela a sensibilidade radial e aumenta com `L`.

Ferramentas comerciais exploram este principio com múltiplos espaçamentos:

```
  +----------------------------------------------------------------------+
  |  FERRAMENTAS LWD COM MULTIPLOS ESPACAMENTOS T-R                      |
  |                                                                      |
  |  Ferramenta       Fabricante    Espacamentos    Freq. (kHz)          |
  |  ----------------------------------------------------------------    |
  |  ARC-8            Schlumberger  5 pares         2/400/2000           |
  |  Periscope        Schlumberger  5 pares         100/400/2000         |
  |  EcoScope         Schlumberger  6 pares         2/100/400/2000       |
  |  ADR              Halliburton   4 pares         100/500/2000         |
  |  TerraVision      Baker Hughes  5 pares         125/500/2000         |
  |                                                                      |
  |  Espacamentos curtos (~0,25-0,50 m):                                 |
  |    - Alta resolucao vertical (~0,3 m)                                |
  |    - Rasa profundidade de investigacao (~0,5-1,0 m radial)           |
  |    - Sensivel a invasao de fluido de perfuracao                      |
  |                                                                      |
  |  Espacamentos longos (~1,5-2,0 m):                                   |
  |    - Baixa resolucao vertical (~1,5 m)                               |
  |    - Profunda investigacao (~2,0-5,0 m radial)                       |
  |    - Ve a resistividade virgem da formacao                           |
  +----------------------------------------------------------------------+
```

Para treinar redes de inversão que operem com dados reais, o simulador precisa gerar respostas para os mesmos conjuntos de espaçamentos das ferramentas comerciais. Com `nTR > 1`, o simulador produz dados sintéticos multi-espaçamento em uma única execução, mantendo consistência entre os pares (mesmo modelo geológico, mesma posição de poco, mesma frequência).

#### 11.1.2 Implementação

A implementação consiste em um loop externo `do itr = 1, nTR` que envolve toda a logica de simulação existente (loops theta e medidas):

```
  +----------------------------------------------------------------------+
  |  ESTRUTURA DE LOOPS — MULTI-TR (perfila1DanisoOMP)                   |
  |                                                                      |
  |  do itr = 1, nTR                    <- Feature 1: loop externo       |
  |    !$omp parallel do if(ntheta > 1) <- Fase 5b: nested adaptativo    |
  |    do k = 1, ntheta                 <- Loop de angulos               |
  |      r_k = dTR(itr) * |sin(theta_k)|  <- Distancia horiz. TR        |
  |      commonarraysMD(r_k, ...) x nf  <- Fase 4: cache por (itr, k)   |
  |      !$omp parallel do schedule(guided, 16)                          |
  |      do j = 1, nmed(k)             <- Loop de medidas               |
  |        Lsen = dTR(itr) * sin(theta_k)                               |
  |        Lcos = dTR(itr) * cos(theta_k)                               |
  |        fieldsinfreqs_cached_ws(ws_pool(tid), ...)                    |
  |      end do                                                          |
  |    end do                                                            |
  |    writes_files(..., itr, nTR)      <- Um arquivo por par T-R        |
  |  end do                                                              |
  +----------------------------------------------------------------------+
```

**Leitura do `model.in`** (em `RunAnisoOmp.f08`):

```fortran
! model.in agora inclui nTR e nTR valores de dTR:
read(11,*) nTR
allocate(dTR_arr(nTR))
do i = 1, nTR
  read(11,*) dTR_arr(i)
end do
```

Exemplo de `model.in` com 3 pares T-R:

```
2                   ! nf (numero de frequencias)
20000.0             ! freq(1) em Hz
40000.0             ! freq(2) em Hz
1                   ! ntheta
0.0                 ! theta(1) em graus
-1.0                ! h1 (altura inicial)
600.0               ! tj (tamanho da janela)
1.0                 ! p_med (passo entre medidas)
3                   ! nTR (numero de pares T-R) <-- NOVO
0.5                 ! dTR(1) = 0.5 m (espacamento curto)
1.0                 ! dTR(2) = 1.0 m (espacamento medio)
2.0                 ! dTR(3) = 2.0 m (espacamento longo)
validacao           ! filename (prefixo dos arquivos de saida)
...                 ! ncam, resistividades, espessuras, etc.
```

**Geometria T-R por par:** Para cada par `itr`, o código computa a geometria do transmissor e receptor em coordenadas de poco inclinado:

```fortran
Lsen = dTR(itr) * seno       ! Projecao horizontal do espacamento
Lcos = dTR(itr) * coss       ! Projecao vertical do espacamento
r_k  = dTR(itr) * dabs(seno) ! Distancia horizontal T-R (para Hankel)

! Posicao do receptor (j-esima medida):
x  = 0.d0 + (j-1) * px - Lsen / 2
z  = z1   + (j-1) * pz - Lcos / 2

! Posicao do transmissor (j-esima medida):
Tx = 0.d0 + (j-1) * px + Lsen / 2
Tz = z1   + (j-1) * pz + Lcos / 2
```

#### 11.1.3 Interação com o Cache Fase 4

O cache de `commonarraysMD` (9 arrays de dimensão `(npt, n, nf)`) é **recomputado para cada combinação (itr, k)** porque `r_k = dTR(itr) * |sin(theta_k)|` varia com o espaçamento T-R:

```
Chamadas a commonarraysMD por modelo:
  Antes (sem cache):  nTR * ntheta * nf * nmed = 3 * 1 * 2 * 600 = 3.600
  Com cache Fase 4:   nTR * ntheta * nf        = 3 * 1 * 2       = 6
  Reducao:            99,83 %
```

#### 11.1.4 Convenção de Saída

A sub-rotina `writes_files` gera nomes de arquivo condicionados ao número de pares T-R:

| Cenario | Arquivo de dados | Arquivo info |
|:--------|:----------------|:-------------|
| `nTR = 1` | `filename.dat` | `infofilename.out` |
| `nTR = 3, itr = 1` | `filename_TR1.dat` | `infofilename.out` |
| `nTR = 3, itr = 2` | `filename_TR2.dat` | `infofilename.out` |
| `nTR = 3, itr = 3` | `filename_TR3.dat` | `infofilename.out` |

```fortran
! Em writes_files:
if (nTR > 1) then
  write(tr_suffix, '(A,I0)') '_TR', itr
  fileTR = mypath//trim(adjustl(filename))//trim(tr_suffix)//'.dat'
else
  fileTR = mypath//trim(adjustl(filename))//'.dat'
end if
```

O arquivo `.out` de metadados é escrito uma única vez (no último modelo `modelm == nmaxmodel`) e contém o número de ângulos, frequências, modelos e número de medidas por ângulo — independente do número de pares T-R.

#### 11.1.5 Compatibilidade Retroativa

Para `nTR = 1`, o loop externo executa uma única iteração e o nome do arquivo de saída não inclui sufixo. A saída é **bit-exata** em relação ao código pre-Feature-1, válidada por MD5:

```
MD5 (nTR=1, -O0): 3d3c309fd1aa121f8b4166268552814c
```

---

### 11.2 Tensor Completo 9 Componentes (Feature 2)

#### 11.2.1 Mapeamento do Tensor H(3x3)

O simulador computa o tensor completo de inducao magnética **H** como uma matriz 3x3 complexa, resultante de tres fontes de dipolo e tres componentes de campo:

```
  +----------------------------------------------------------------------+
  |  TENSOR DE INDUCAO MAGNETICA H(3x3)                                  |
  |                                                                      |
  |  Fonte \\ Campo    Hx           Hy           Hz                       |
  |  ----------------------------------------------------------------    |
  |  HMD-x (dip. x)   Hxx          Hyx          Hzx                     |
  |  HMD-y (dip. y)   Hxy          Hyy          Hzy                     |
  |  VMD-z (dip. z)   Hxz          Hyz          Hzz                     |
  |                                                                      |
  |  Sub-rotinas:                                                        |
  |    hmd_TIV_optimized_ws(dipolo='hmdxy') -> Hx(1,1:2), Hy(1,1:2)     |
  |      1 = HMD-x (Hxx, Hyx, Hzx)                                      |
  |      2 = HMD-y (Hxy, Hyy, Hzy)                                      |
  |    vmd_optimized_ws()                   -> HxVMD, HyVMD, HzVMD       |
  |      = VMD-z  (Hxz, Hyz, Hzz)                                       |
  +----------------------------------------------------------------------+
```

A montagem do tensor e aplicação da rotação ocorrem em `fieldsinfreqs_cached_ws`:

```fortran
! Montagem do tensor no sistema de ferramenta:
matH(1,:) = (/HxHMD(1,1), HyHMD(1,1), HzHMD(1,1)/)  ! Linha HMD-x
matH(2,:) = (/HxHMD(1,2), HyHMD(1,2), HzHMD(1,2)/)  ! Linha HMD-y
matH(3,:) = (/HxVMD,      HyVMD,      HzVMD/)         ! Linha VMD-z

! Rotacao para o sistema de coordenadas do poco:
tH = RtHR(ang, 0.d0, 0.d0, matH)
```

A função `RtHR(theta, phi, psi, H)` aplica a transformacao `H' = R^T * H * R`, onde `R` é a matriz de rotação parametrizada pelo ângulo de inclinacao `theta` (dip), azimute `phi` e rotação da ferramenta `psi`.

#### 11.2.2 Armazenamento das 9 Componentes

O array de saída `cH(nf, 9)` armazena as 9 componentes em ordem **column-major** (por linha de `matH`):

```
  Indice    Componente    Fonte       Campo     Fisicamente
  ------    ----------    ---------   ------    ---------------------
  cH(i,1)   Hxx          HMD-x       Hx        Coplanar horizontal
  cH(i,2)   Hxy          HMD-y       Hx        Cross-componente
  cH(i,3)   Hxz          VMD-z       Hx        Cross-componente
  cH(i,4)   Hyx          HMD-x       Hy        Cross-componente
  cH(i,5)   Hyy          HMD-y       Hy        Coplanar horizontal
  cH(i,6)   Hyz          VMD-z       Hy        Cross-componente
  cH(i,7)   Hzx          HMD-x       Hz        Cross-componente
  cH(i,8)   Hzy          HMD-y       Hz        Cross-componente
  cH(i,9)   Hzz          VMD-z       Hz        Coaxial vertical
```

#### 11.2.3 Formato Binario de 22 Colunas

Cada registro escrito por `writes_files` em formato `stream unformatted` contém 22 colunas:

```
  Col    Tipo       Conteudo                    Unidade
  ---    --------   -----------------------     -------
   0     int32      Numero da medida (j)        adimensional
   1     real64     Frequencia                  Hz
   2     real64     Angulo theta                graus
   3     real64     zobs (profundidade T-R)     metros
   4     real64     rho_h (resistividade horiz) Ohm.m
   5     real64     rho_v (resistividade vert)  Ohm.m
   6     real64     Re(Hxx)                     A/m
   7     real64     Im(Hxx)                     A/m
   8     real64     Re(Hxy)                     A/m
   9     real64     Im(Hxy)                     A/m
  10     real64     Re(Hxz)                     A/m
  11     real64     Im(Hxz)                     A/m
  12     real64     Re(Hyx)                     A/m
  13     real64     Im(Hyx)                     A/m
  14     real64     Re(Hyy)                     A/m
  15     real64     Im(Hyy)                     A/m
  16     real64     Re(Hyz)                     A/m
  17     real64     Im(Hyz)                     A/m
  18     real64     Re(Hzx)                     A/m
  19     real64     Im(Hzx)                     A/m
  20     real64     Re(Hzy)                     A/m
  21     real64     Im(Hzy)                     A/m
  22     real64     Re(Hzz)                     A/m
  23     real64     Im(Hzz)                     A/m
```

O tamanho de cada registro é `4 + 21 * 8 = 172 bytes` (1 int32 + 21 float64). Para 600 medidas e 2 frequências, cada modelo produz `600 * 2 = 1.200 registros = 206.400 bytes` por par T-R.

**Mapeamento para o pipeline DL** (`geosteering_ai/`):

```
  INPUT_FEATURES = [1, 4, 5, 20, 21]
    Col 1 = zobs      (profundidade)
    Col 4 = Re(Hxx)   (coplanar horizontal, real)
    Col 5 = Im(Hxx)   (coplanar horizontal, imaginario)
    Col 20 = Re(Hzz)  (coaxial vertical, real)
    Col 21 = Im(Hzz)  (coaxial vertical, imaginario)

  OUTPUT_TARGETS = [2, 3]
    Col 2 = rho_h     (resistividade horizontal — target de inversao)
    Col 3 = rho_v     (resistividade vertical — target de inversao)
```

---

### 11.3 Interface f2py (Feature 4)

#### 11.3.1 Motivação

O fluxo de produção atual envolve escrita de arquivos `.dat` em disco pelo Fortran e leitura subsequente pelo Python — overhead de I/O e de parsing binario que domina em cenarios de geração interativa ou de otimização bayesiana (Optuna), onde milhares de modelos são avaliados em loop apertado. A interface f2py elimina este overhead retornando arrays NumPy diretamente da sub-rotina Fortran.

#### 11.3.2 Módulo `tatu_wrapper`

O arquivo `Fortran_Gerador/tatu_f2py_wrapper.f08` define o módulo `tatu_wrapper` com uma única sub-rotina publica `simulate`:

```fortran
module tatu_wrapper
  use parameters
  use filterscommonbase
  use utils
  use magneticdipoles
  use omp_lib
  implicit none
contains

subroutine simulate(nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med, &
                    n, resist, esp, nmmax, zrho_out, cH_out)
  ! ...
  !f2py intent(in) :: nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med
  !f2py intent(in) :: n, resist, esp, nmmax
  !f2py intent(out) :: zrho_out, cH_out
  !f2py depend(nf) :: freq
  !f2py depend(ntheta) :: theta
  !f2py depend(nTR) :: dTR
  !f2py depend(n) :: resist, esp
  !f2py depend(nTR, ntheta, nmmax, nf) :: zrho_out, cH_out

  real(dp), intent(out) :: zrho_out(nTR, ntheta, nmmax, nf, 3)
  complex(dp), intent(out) :: cH_out(nTR, ntheta, nmmax, nf, 9)
  ! ...
end subroutine simulate
end module tatu_wrapper
```

#### 11.3.3 Arrays de Saída

| Array | Shape | Tipo | Conteudo |
|:------|:------|:-----|:---------|
| `zrho_out` | `(nTR, ntheta, nmmax, nf, 3)` | `float64` | Resistividades aparentes: `zobs`, `rho_h`, `rho_v` |
| `cH_out` | `(nTR, ntheta, nmmax, nf, 9)` | `complex128` | Tensor EM completo: 9 componentes `Hxx..Hzz` |

**Dimensões:**
- `nTR`: número de pares T-R (1 a ~6)
- `ntheta`: número de ângulos de inclinacao (1 a ~7)
- `nmmax`: número máximo de medidas por ângulo, pre-calculado pelo caller
- `nf`: número de frequências (tipicamente 2)
- `3` ou `9`: componentes escalares ou tensoriais

**Calculo de `nmmax` em Python:**

```python
import math
nmmax = max(
    math.ceil(tj / (p_med * math.cos(theta_i * math.pi / 180)))
    for theta_i in theta
)
```

#### 11.3.4 Compilação e Uso

**Compilação com f2py (NumPy):**

```bash
# Compilacao do modulo compartilhado:
python3 -m numpy.f2py -c \\
    --fcompiler=gnu95 \\
    --f90flags=\"-O3 -march=native -fopenmp -std=f2008\" \\
    -lgomp \\
    parameters.f08 utils.f08 filtersv2.f08 \\
    magneticdipoles.f08 tatu_f2py_wrapper.f08 \\
    -m tatu_f2py

# Resultado: tatu_f2py.cpython-310-darwin.so (ou .so em Linux)
```

**Uso em Python:**

```python
import numpy as np
import tatu_f2py

# Parametros do modelo
freq = np.array([20000.0, 40000.0])          # 2 frequencias (Hz)
theta = np.array([0.0])                       # 1 angulo (graus)
dTR = np.array([0.5, 1.0, 2.0])              # 3 pares T-R (metros)
resist = np.array([[10.0, 20.0], ...])        # (n, 2) resistividades
esp = np.array([0.0, 5.0, 3.0, ...])         # (n,) espessuras

# Calculo de nmmax
nmmax = max(
    int(np.ceil(tj / (p_med * np.cos(th * np.pi / 180))))
    for th in theta
)

# Simulacao direta (sem I/O de disco)
zrho, cH = tatu_f2py.tatu_wrapper.simulate(
    nf=2, freq=freq, ntheta=1, theta=theta,
    h1=1.0, tj=600.0, nTR=3, dTR=dTR,
    p_med=1.0, n=len(resist), resist=resist, esp=esp, nmmax=nmmax
)

# zrho.shape = (3, 1, nmmax, 2, 3)  — [nTR, ntheta, medidas, freq, 3]
# cH.shape   = (3, 1, nmmax, 2, 9)  — [nTR, ntheta, medidas, freq, 9]

# Acessar Hzz (componente 9) para o 2o par T-R, freq 1:
Hzz = cH[1, 0, :, 0, 8]  # complex128 array
```

#### 11.3.5 Compatibilidade com OpenMP

O wrapper preserva toda a infraestrutura de paralelismo do simulador: workspace pool, caches Fase 4, paralelismo adaptativo Fase 5b. O número de threads é controlado pela variável de ambiente `OMP_NUM_THREADS` normalmente.

```python
import os
os.environ['OMP_NUM_THREADS'] = '8'
import tatu_f2py  # Threads configuradas na importacao
```

---

### 11.4 Batch Paralelo de Modelos (Feature 5)

#### 11.4.1 Motivação

O simulador Fortran processa um modelo geológico por vez. Para gerar datasets de treinamento com milhares ou milhoes de modelos, o script gerador `fifthBuildTIVModels.py` invoca o executavel `tatu.x` sequêncialmente. O batch runner (`Fortran_Gerador/batch_runner.py`) introduz paralelismo de **nivel de modelo** usando `ProcessPoolExecutor`, complementando o paralelismo OpenMP intra-modelo.

#### 11.4.2 Arquitetura

```
  +----------------------------------------------------------------------+
  |  BATCH RUNNER — PARALELISMO MULTI-NIVEL                              |
  |                                                                      |
  |  Nivel 1: Workers Python (ProcessPoolExecutor)                       |
  |    Worker 0: modelos 1-250      Worker 1: modelos 251-500            |
  |    Worker 2: modelos 501-750    Worker 3: modelos 751-1000           |
  |                                                                      |
  |  Nivel 2: Threads OpenMP (por worker, intra-modelo)                  |
  |    Cada worker: OMP_NUM_THREADS = omp_threads                        |
  |                                                                      |
  |  Restricao: workers * omp_threads <= nucleos fisicos                 |
  |    Exemplo: 4 workers * 2 OMP = 8 threads total (i9-9980HK)         |
  +----------------------------------------------------------------------+
```

#### 11.4.3 Isolamento por Diretorio Temporario

Cada worker opera em um diretorio temporario independente para evitar conflitos de I/O:

```python
def run_model_batch(args):
    worker_id, model_start, model_end, tatu_path, model_in_path, omp_threads = args
    env = {**os.environ, 'OMP_NUM_THREADS': str(omp_threads)}

    with tempfile.TemporaryDirectory(prefix=f'tatu_w{worker_id}_') as tmpdir:
        # Copiar executavel e model.in para diretorio isolado
        shutil.copy2(tatu_path, os.path.join(tmpdir, 'tatu.x'))
        shutil.copy2(model_in_path, os.path.join(tmpdir, 'model.in'))

        # Executar modelos sequencialmente dentro do worker
        for i in range(model_start, model_end + 1):
            # Atualizar model.in com o modelo atual
            subprocess.run([tatu_dst], cwd=tmpdir, env=env, check=True)

        # Coletar arquivos .dat e .out gerados
        return worker_id, model_start, model_end, results
```

#### 11.4.4 Interface CLI

```bash
python3 batch_runner.py --models 3000 --workers 4 --omp-threads 2
```

| Flag | Default | Descricao |
|:-----|:-------:|:----------|
| `--models` | 1000 | Número total de modelos a gerar |
| `--workers` | 4 | Número de workers paralelos (processos Python) |
| `--omp-threads` | 2 | Threads OpenMP por worker |
| `--tatu` | `./tatu.x` | Caminho para o executavel compilado |
| `--model-in` | `./model.in` | Caminho para o arquivo `model.in` base |

#### 11.4.5 Estrategia de Threading

A regra fundamental é:

```
workers * omp_threads <= nucleos_fisicos
```

Configurações recomendadas por hardware:

| Hardware | Cores Físicos | Workers | OMP Threads | Total | Throughput Estimado |
|:---------|:-------------:|:-------:|:-----------:|:-----:|:-------------------:|
| Laptop i9-9980HK | 8 | 4 | 2 | 8 | ~120.000 mod/h |
| Workstation Ryzen 5950X | 16 | 4 | 4 | 16 | ~240.000 mod/h |
| Servidor Dual Xeon 8280 | 56 | 8 | 7 | 56 | ~800.000 mod/h |
| Google Colab (T4) | 2 | 2 | 1 | 2 | ~30.000 mod/h |

A distribuição de modelos entre workers usa chunks iguais com remainder distribuido:

```python
chunk = total // n_workers
remainder = total % n_workers
# Worker w recebe (chunk + 1) modelos se w < remainder, senao chunk
```

---

### 11.5 Frequências Arbitrárias — F5 (v8.0)

#### 11.5.1 Motivação Física

Ferramentas LWD reais operam com 4-8 frequências simultâneas (10, 20, 40, 100, 200, 400 kHz)
para investigar diferentes profundidades. A profundidade de penetração (skin depth) é
inversamente proporcional à raiz quadrada da frequência:

```
δ = √(ρ / (π × f × μ₀))

Tabela de skin depth (ρ = 10 Ω·m):
  ┌──────────────┬──────────────┬───────────────┐
  │ Frequência   │ Skin depth   │ Investigação  │
  ├──────────────┼──────────────┼───────────────┤
  │  10 kHz      │  5,03 m      │ Profunda      │
  │  20 kHz (★)  │  3,56 m      │ Média         │
  │  40 kHz (★)  │  2,52 m      │ Intermediária │
  │ 100 kHz      │  1,59 m      │ Rasa          │
  │ 200 kHz      │  1,13 m      │ Muito rasa    │
  │ 400 kHz      │  0,80 m      │ Sub-metro     │
  └──────────────┴──────────────┴───────────────┘
  (★) = frequências do baseline v7.0 (nf = 2)
```

Múltiplas frequências fornecem resolução multi-escala essencial para detecção de fronteiras
e caracterização de formação em tempo real.

#### 11.5.2 Implementação

**Escopo:** Mínimo — o código já suporta `nf` arbitrário via caches Phase 4 `(npt, n, nf, ntheta)`.

**Flag de ativação (model.in v8.0):**
```
0                 !F5: use_arbitrary_freq (0=desabilitado, 1=habilitado)
```

**Comportamento:**

| `use_arb_freq` | nf | Ação |
|:--------------:|:--:|:-----|
| 0 (padrão) | 1-2 | Normal, sem aviso |
| 0 (padrão) | >2 | Aviso no console: "[F5 AVISO] nf > 2 com use_arbitrary_freq desabilitado" |
| 1 | 1-16 | Validado, sem restrição |
| 1 | >16 | Erro fatal: "[F5 ERRO] nf fora do intervalo [1, 16]" |

**Validação no Fortran (perfila1DanisoOMP):**
```fortran
if (use_arb_freq == 1) then
  if (nf < 1 .or. nf > 16) stop '[F5] nf deve estar entre 1 e 16'
  ! Exibe todas as frequências configuradas
end if
```

#### 11.5.3 Impacto de Performance (F5)

O custo computacional escala **linearmente** com `nf`:

| nf | Custo relativo | Throughput estimado (8t) | Memória cache Phase 4 |
|:--:|:--------------:|:------------------------:|:---------------------:|
| 2 (baseline) | 1× | ~58.856 mod/h | ~3,4 MB |
| 4 | ~2× | ~30.000 mod/h | ~6,8 MB |
| 6 | ~3× | ~20.000 mod/h | ~10,2 MB |
| 8 | ~4× | ~15.000 mod/h | ~13,6 MB |

**Backward compatibility:** `nf = 2` com `use_arb_freq = 0` produz resultado **bit-exato** com v7.0.

---

### 11.6 Antenas Inclinadas — F7 (v8.0)

#### 11.6.1 Motivação Física

Ferramentas LWD modernas (GeoSphere HD, Rt Scanner, EarthStar) utilizam antenas inclinadas
a 15-45 graus do eixo do poço para melhorar a detecção de fronteiras de camada e fornecer
informação direcional (azimuth).

#### 11.6.2 Formulação Matemática

A resposta de uma antena receptora com eixo inclinado n̂ = (sinβ·cosφ, sinβ·sinφ, cosβ)
medindo o campo de um transmissor axial (ẑ) é uma combinação linear do tensor H(3×3):

```
H_tilted(β, φ) = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]

Onde:
  β  = ângulo de inclinação (0°-90°)
     β = 0° → antena axial (Hzz puro)
     β = 90° → antena transversal (Hxz, Hyz puros)
  φ  = ângulo azimutal (0°-360°)
     φ = 0° → componente x
     φ = 90° → componente y

Mapeamento do tensor cH(1:9):
  cH(1)=Hxx  cH(2)=Hxy  cH(3)=Hxz
  cH(4)=Hyx  cH(5)=Hyy  cH(6)=Hyz
  cH(7)=Hzx  cH(8)=Hzy  cH(9)=Hzz
```

#### 11.6.3 Implementação

**Escopo:** Pós-processamento puro — **zero modificação no core EM**. O tensor completo
H(3×3) já é computado pelo forward model; a combinação linear é calculada após o loop
paralelo de medidas.

**Flag de ativação (model.in v8.0):**
```
1                 !F7: use_tilted_antennas (0=desabilitado, 1=habilitado)
2                 !F7: n_tilted
45.0  0.0         !F7: beta(1)=45° phi(1)=0°
30.0  90.0        !F7: beta(2)=30° phi(2)=90°
```

**Cálculo (perfila1DanisoOMP, serial pós-loop paralelo):**
```fortran
do it = 1, n_tilted
  beta_rad = beta_tilt(it) * pi / 180.d0
  phi_rad  = phi_tilt(it)  * pi / 180.d0
  do k = 1, ntheta
    do j = 1, nmed(k)
      do i = 1, nf
        cH_tilted(k, j, i, it) = &
          cos(beta_rad) * cH1(k, j, i, 9) + &
          sin(beta_rad) * (cos(phi_rad) * cH1(k, j, i, 3) + &
                           sin(phi_rad) * cH1(k, j, i, 6))
      end do
    end do
  end do
end do
```

**Custo:** 5 multiplicações + 2 adições por ponto × ntheta × nmed × nf × n_tilted.
Para n_tilted=2, nf=2, nmed=600: ~2.400 operações — **negligível** (~24 μs).

#### 11.6.4 Formato de Saída Estendido

Quando F7 está habilitado, cada registro binário é estendido com `2 × n_tilted` floats64
adicionais (Re e Im de cada H_tilted):

```
Registro padrão (22 colunas, 172 bytes):
  int32 | zobs | rho_h | rho_v | Re(Hxx) Im(Hxx) ... Re(Hzz) Im(Hzz)

Extensão F7 (+2×n_tilted colunas):
  ... | Re(H_tilted_1) Im(H_tilted_1) | Re(H_tilted_2) Im(H_tilted_2) | ...

Total: 172 + n_tilted × 16 bytes por registro.
```

O arquivo `.out` inclui metadados F7 (linhas 5-7):
```
use_tilted  n_tilted            ! linha 5: flags F7
beta(1) beta(2) ...             ! linha 6: ângulos de inclinação
phi(1)  phi(2)  ...             ! linha 7: ângulos azimutais
```

#### 11.6.5 Interface f2py (simulate_v8)

A sub-rotina `simulate_v8` no wrapper f2py aceita os parâmetros F5/F7 e retorna
um array adicional `cH_tilted_out(nTR, ntheta, nmmax, nf, n_tilted)`:

```python
zrho, cH, cH_tilted = tatu_f2py.simulate_v8(
    nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med,
    n, resist, esp, nmmax,
    use_arb_freq=1, use_tilted=1, n_tilted=2,
    beta_tilt=[45., 30.], phi_tilt=[0., 90.])
```

**Backward compatibility:** `use_tilted = 0` produz resultado **idêntico** a `simulate()`.

---

### 11.9 F10 — Sensibilidades ∂H/∂ρ (Jacobiano) — v10.0

#### 11.9.1 Motivação Física

A matriz Jacobiana **J = ∂H/∂ρ** é o operador central de três aplicações críticas do projeto:

1. **PINNs (Physics-Informed Neural Networks):** a physics loss envolve o resíduo
   das equações de Maxwell `∂/∂ρ(L[H]) = 0`, que requer J explicitamente.
2. **Inversão determinística (Gauss-Newton):** atualização do modelo
   `Δρ = (JᵀJ)⁻¹·JᵀΔd` requer J a cada iteração.
3. **Quantificação de incerteza:** `Σ_post ≈ (JᵀC_d⁻¹J + C_m⁻¹)⁻¹` — matriz de
   covariância posterior aproximada a partir de J.

Ref: [`docs/reference/relatorio_vantagens_jacobiano.md`](relatorio_vantagens_jacobiano.md) §1–3.

#### 11.9.2 Formulação Matemática

Diferenças finitas **centradas** (erro O(δ²)):

```
J_{i,k,h} = (H_i(ρ_h,k + δ) − H_i(ρ_h,k − δ)) / (2·δ)
J_{i,k,v} = (H_i(ρ_v,k + δ) − H_i(ρ_v,k − δ)) / (2·δ)

δ = max(fd_step × |ρ_ref|, 1e-6)     (default fd_step = 1e-4 = 0,01%)
```

Dimensões do Jacobiano:
- **Input:** modelo com `n` camadas → `2n` parâmetros (ρ_h e ρ_v)
- **Output:** `(nTR, ntheta, nmmax, nf, 9, n)` complex(dp) — uma derivada por camada
- **Memória:** para `nTR=1, ntheta=1, nmmax=600, nf=2, n=10` ≈ **3,5 MB** (h + v)

#### 11.9.3 Duas Estratégias de Paralelização (B e C)

**Ambas implementadas em v10.0 — selecionáveis via `jacobian_method`:**

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  ESTRATÉGIA B — Python Workers (ProcessPoolExecutor)                     │
  │                                                                          │
  │  Princípio: expandir cada modelo em (1 + 4n) sub-modelos perturbados.    │
  │  Cada sub-modelo é uma execução normal do simulador (sem conhecimento    │
  │  do Jacobiano). Pós-processamento em Python reagrupa por parent_id.      │
  │                                                                          │
  │  Fluxo:                                                                  │
  │    master → expand_models_with_perturbations(models, δ_rel=1e-4)         │
  │    workers → run_model_batch (cada um roda seus sub-modelos)             │
  │    master → merge .dat → compute_jacobian_from_perturbations             │
  │    → salva .jac.npz (formato NumPy compactado)                           │
  │                                                                          │
  │  Vantagens:                                                              │
  │    • Zero modificação no Fortran                                         │
  │    • Reutiliza 100% da infraestrutura de batch existente                 │
  │    • Escala linearmente com workers (limitado apenas por I/O disco)      │
  │                                                                          │
  │  Desvantagens:                                                           │
  │    • 61× mais arquivos intermediários (para n=15 camadas)                │
  │    • Throughput ~1.930 mod+J/h (relatório §9.2)                          │
  │                                                                          │
  │  Uso: --use-jacobian 1 --jacobian-method 0                               │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  ESTRATÉGIA C — OpenMP Interno ao Fortran (compute_jacobian_fd)          │
  │                                                                          │
  │  Princípio: mover o loop de perturbações PARA DENTRO do Fortran,         │
  │  reutilizando caches Phase 4 e ws_pool existente. Loop paralelo sobre    │
  │  2n perturbações (n camadas × 2 componentes h/v) via !$omp parallel do.  │
  │                                                                          │
  │  !$omp parallel do schedule(dynamic, 1) default(shared)                  │
  │    private(kk, layer, comp, resist_p, resist_m, cH_p, cH_m, ...)         │
  │  do kk = 1, 2*n                                                          │
  │    layer = (kk-1)/2 + 1                                                  │
  │    comp  = mod(kk-1, 2) + 1    ! 1=h, 2=v                                │
  │    ! Aloca caches Phase 4 privados por thread                            │
  │    ! Chama fieldsinfreqs_cached_ws para +δ e −δ                          │
  │    ! Calcula J = (H_p - H_m) / (2δ)                                      │
  │  end do                                                                  │
  │  !$omp end parallel do                                                   │
  │                                                                          │
  │  Vantagens:                                                              │
  │    • Zero I/O: J fica em memória                                         │
  │    • Reaproveita commonarraysMD e ws_pool existentes                     │
  │    • ~13× mais rápido que Estratégia B (relatório §9.3)                  │
  │    • Throughput ~12.900 mod+J/h                                          │
  │                                                                          │
  │  Saída: arquivos .jac (binário stream) com shape                         │
  │         (ntheta, nmmax, nf, 9, n) complex(dp) para h e v.                │
  │                                                                          │
  │  Uso: --use-jacobian 1 --jacobian-method 1                               │
  └──────────────────────────────────────────────────────────────────────────┘
```

#### 11.9.4 Tabela Comparativa B × C × JAX

| Estratégia | Throughput (mod+J/h) | LOC Fortran | LOC Python | I/O disco | Risco | Uso |
|:-----------|:--------------------:|:-----------:|:----------:|:---------:|:-----:|:----|
| **A** (Sequencial intra-worker) | ~968 | 0 | ~50 | 61× | Baixo | Prototipagem |
| **B** (Python Workers) | ~1.930 | 0 | ~280 | 61× | Médio | **Datasets ≤ 5k** (v10.0) |
| **C** (OpenMP Fortran) | **~12.900** | ~430 | ~50 | Mínimo | Alto | **Produção** (v10.0) |
| **JAX auto-diff** (referência) | ~133.000 | 0 | ~0 | Zero | Médio | Track 2 (futuro) |

**Recomendação:** usar Estratégia C (método 1) para datasets de produção (throughput 13× maior).
Estratégia B (método 0) é útil para validação cruzada e ambientes onde não é possível recompilar o Fortran.

#### 11.9.5 Formato do arquivo `.jac` (Estratégia C, binário stream)

```
Header (int32): nt, nmmax, nf, 9, n_layers, itr, nTR     (28 bytes)
Payload:
  Re/Im de dH_dRho_h(k, j, i, ic, layer) × nt × nmmax × nf × 9 × n_layers
  Re/Im de dH_dRho_v(k, j, i, ic, layer) × nt × nmmax × nf × 9 × n_layers
```

Nome: `{filename}[_TR{itr}].jac` (sufixo `_TR{itr}` apenas se nTR > 1).

#### 11.9.6 Formato do arquivo `.jac.npz` (Estratégia B, NumPy compactado)

Arrays salvos via `np.savez_compressed`:
- `dH_dRho_h`: complex128 `(N_models, nTR, ntheta, nmmax, nf, 9, max_n)`
- `dH_dRho_v`: mesma shape
- `n_layers_per_model`: int32 `(N_models,)`
- `deltas`: float64 `(N_models, max_n, 2)` — δ efetivo por (layer, componente)
- `parent_ids`: int32 `(N_models,)` — validação do reagrupamento

#### 11.9.7 Integração com simulate_v10_jacobian (f2py)

```python
import tatu_f2py
zrho, cH, cH_tilted, dJ_h, dJ_v = tatu_f2py.simulate_v10_jacobian(
    nf=2, freq=np.array([20000., 40000.]),
    ntheta=1, theta=np.array([0.]),
    h1=10., tj=120., nTR=1, dTR=np.array([1.0]), p_med=0.2,
    n=3, resist=np.array([[1.,1.],[10.,10.],[100.,100.]]),
    esp=np.array([0.,50.,0.]), nmmax=600,
    use_arb_freq=0, use_tilted=0, n_tilted=0, n_tilted_sz=1,
    beta_tilt=np.zeros(1), phi_tilt=np.zeros(1),
    filter_type_in=0, use_jacobian_in=1, jacobian_fd_step_in=1e-4)

assert dJ_h.shape == (1, 1, 600, 2, 9, 3)
assert dJ_v.shape == (1, 1, 600, 2, 9, 3)
```

---

### 11.7 Impacto Computacional dos Novos Recursos

| Recurso | Custo Adicional | Throughput (8t, n=15) | Compatibilidade |
|:--------|:----------------|:---------------------:|:---------------:|
| **Multi-TR (nTR=1)** | 0 % (loop executa 1×) | 58.856 mod/h | Bit-exato |
| **Multi-TR (nTR=3)** | ~3× (linear em nTR) | ~20.000 mod/h | N/A (novo cenário) |
| **Tensor 9 comp.** | 0 % (já era computado) | 58.856 mod/h | Bit-exato |
| **f2py wrapper** | ~0 % (elimina I/O disco) | ~60.000+ mod/h | Numericamente idêntico |
| **Batch runner (4w × 2t)** | Overhead de processo ~2-5% | ~120.000 mod/h | Idêntico por modelo |
| **F5: nf arbitrário (nf=6)** | ~3× (linear em nf) | ~20.000 mod/h | Bit-exato p/ nf=2 |
| **F7: Tilted (n_tilted=2)** | <0,01% (pós-processamento) | ~58.856 mod/h | Zero impacto no core |
| **F10-B: Jacobiano Python** | 1 + 4n× (n camadas) | ~1.930 mod+J/h (n=10) | Opt-in, desabilitado default |
| **F10-C: Jacobiano Fortran OpenMP** | 2n× com paralelismo | **~12.900 mod+J/h** (n=10) | Opt-in, bit-exato se off |

**Nota:** O custo do Multi-TR é linear porque o cache Fase 4 deve ser recomputado para cada valor de `r_k = dTR(itr) * |sin(theta)|`, e o loop completo de medidas é executado `nTR` vezes. Não há possibilidade de reutilização de cache entre pares T-R distintos.

---

### 11.8 Bases Físicas e Geofísicas

#### 11.8.1 Por que Múltiplos Espaçamentos?

Em um meio de camadas com invasão de fluido de perfuracao, a resistividade varia radialmente:

```
  Distancia radial do poco (m)
  0.0     0.5     1.0     1.5     2.0     3.0
  |-------|-------|-------|-------|-------|
  |  mud  | invaded zone  |  virgin formation |
  |  Rm   |     Rxo       |       Rt          |
  |~0.1   |   ~1-5        |   ~10-100 Ohm.m   |
```

Espacamentos curtos (0,25-0,50 m) são dominados pela zona invadida (`Rxo`). Espacamentos longos (1,5-2,0 m) penetram até a formacao virgem (`Rt`). A combinação de múltiplos espaçamentos permite **desconvolucao radial** — separar `Rxo` de `Rt` — informação critica para cálculo de saturacao de agua (`Sw`) via equação de Archie.

#### 11.8.2 Por que o Tensor Completo?

Em meios TIV (eixo de simetria vertical), as componentes diagonais `Hxx`, `Hyy`, `Hzz` são sensíveis a:
- `Hzz` (coaxial): dominada pela resistividade horizontal `rho_h`
- `Hxx`, `Hyy` (coplanares): sensíveis ao contraste `rho_v / rho_h` (anisotropia)

As componentes off-diagonal (`Hxy`, `Hxz`, etc.) são **zero em meios 1D isotropicos** mas tornam-se nao-nulas quando:
- O poco tem inclinacao (`dip > 0`)
- A ferramenta tem rotação azimutal
- Existem contrastes laterais (geometria 2D/3D)

Para treinar redes de inversão que operem em cenarios reais multi-dip, as componentes off-diagonal contém informação direcional que as diagonais não capturam.

#### 11.8.3 Por que a Interface f2py?

A integração direta elimina tres gargalos:

1. **Latencia de I/O**: Escrita binaria `stream unformatted` + leitura Python via `struct.unpack` adiciona ~0,01-0,05 s/modelo — até 50% do tempo de simulação pos-Fase 4.

2. **Memoria intermediaria**: O arquivo `.dat` armazena dados que já existem em arrays Fortran; a leitura Python cria uma copia NumPy. O f2py retorna o array diretamente — zero copia intermediaria.

3. **Otimização bayesiana**: Em loops Optuna, a função objetivo avalia modelos individuais em millisegundos. A latência de I/O domina completamente. Com f2py, a função objetivo chama `tatu_f2py.simulate()` diretamente.

#### 11.8.4 Por que Batch Paralelo?

O paralelismo intra-modelo (OpenMP) escala até os núcleos físicos mas tem retornos decrescentes apos ~8 threads (eficiência de Amdahl ~52% a 8 threads no baseline). O batch runner explora **paralelismo inter-modelo** — completamente independente e linear — atingindo eficiência próxima a 100% até saturar a largura de banda de memoria.

A combinação dos dois niveis (inter-modelo via `ProcessPoolExecutor` + intra-modelo via OpenMP) é a estrategia otima para CPUs multi-core modernas com hierarquia de cache NUMA.

---

---

## 12. Paralelismo OpenMP — Análise Completa

> **NOTA (v7.0):** Esta seção consolida toda a documentação de paralelismo do simulador:
> a estrutura original (pre-otimização), o diagnóstico de gargalos, as 12 fases de
> otimização implementadas (Fases 0-6b), resultados empiricos detalhados e a referência
> completa de diretivas OpenMP. Os resultados finais mostram speedup de **5,62x** e
> throughput de **58.856 modelos/hora** (8 threads, i9-9980HK).
>
> **Estrutura atual do paralelismo:**
> ```
> !$omp parallel do if(ntheta > 1) schedule(dynamic) num_threads(num_threads_k)
> do k = 1, ntheta          <- Fase 5b: nested apenas quando ntheta > 1
>   [pre-computo serial: commonarraysMD -> 9 caches por freq]  <- Fase 4
>   [reset sentinel ws_pool(t)%last_camadT = -1]
>   !$omp parallel do schedule(guided, 16) num_threads(merge(maxthreads, nj, ntheta==1))
>   do j = 1, nmed(k)       <- Fase 2b: guided chunk tuning
>     tid = [adaptativo]     <- Fase 5b: single-level ou nested
>     call fieldsinfreqs_cached_ws(ws_pool(tid), ..., caches, ...)  <- Fases 3/3b/4
>   end do
> end do
> ```
>
> Throughput final: **58.856 mod/h** (8 threads, `schedule(guided, 16)`, Fase 2b).

---

### 12.1 Estrutura Original (pre-otimização)

O simulador **originalmente** utilizava paralelismo OpenMP aninhado em 2 niveis:

```
Nivel 1 (externo): Angulos theta
  !$omp parallel do schedule(dynamic) num_threads(num_threads_k)
  do k = 1, ntheta   <- tipicamente 1-2 threads

    Nivel 2 (interno): Medicoes por angulo
    !$omp parallel do schedule(dynamic) num_threads(num_threads_j)
    do j = 1, nmed(k)   <- 600+ threads disponiveis
      call fieldsinfreqs(ang, nf, freq, posTR, dipolo, npt, krwJ0J1,
                         n, h, prof, resist, zrho, cH)
    end do
    !$omp end parallel do

  end do
  !$omp end parallel do
```

**Distribuição de threads (código original):**

```fortran
maxthreads    = omp_get_max_threads()      ! Total de threads disponiveis
num_threads_k = ntheta                     ! Threads para angulos (1-2)
num_threads_j = maxthreads - ntheta        ! Restante para medições
```

No caso mais frequente (`ntheta = 1`), o nivel 1 usa exatamente **1 thread** e o nivel 2 usa `maxthreads - 1` threads. O fork/join aninhado ocorre sem qualquer ganho de paralelismo no nivel externo.

**Hierarquia de chamadas dentro de cada iteração j:**

```
  +-------------------------------------------------------------------+
  |  HIERARQUIA DE CHAMADAS — loop j = 1, nmed(k)                     |
  |                                                                   |
  |  fieldsinfreqs(ang, nf, freq, posTR, ...)                        |
  |    |                                                              |
  |    +-- commonarraysMD(r, freq, n, h, resist, ...)   <- 45% CPU   |
  |    |     Calcula: u, s, uh, sh, RTEdw, RTEup,                    |
  |    |              RTMdw, RTMup, AdmInt, ImpInt                   |
  |    |     Chamado: nf x nmed = 2 x 600 = 1.200 vezes/modelo      |
  |    |                                                              |
  |    +-- commonfactorsMD(camadT, freq, ...)            <- 15% CPU  |
  |    |     Calcula: Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz          |
  |    |     Depende: (camadT, freq, h0=Tz)                          |
  |    |                                                              |
  |    +-- hmd_TIV_optimized(...)                        <- 20% CPU  |
  |    |     Calcula: Hx, Hy, Hz para dipolo magnetico horizontal    |
  |    |     Aloca: Tudw, Txdw, Tuup, Txup, TEdwz, TEupz           |
  |    |            -> 600 allocate/deallocate por modelo por freq   |
  |    |                                                              |
  |    +-- vmd_optimized(...)                            <- 10% CPU  |
  |          Calcula: Hz para dipolo magnetico vertical               |
  |          Aloca: mesmos arrays de hmd                              |
  |                 -> 600 allocate/deallocate por modelo por freq   |
  +-------------------------------------------------------------------+
```

---

### 12.2 Diagnostico de Gargalos (4 Causas-Raiz)

#### Causa-Raiz 1 — Alocacao Dinamica Dentro do Laco Paralelo (Impacto: Alto)

As sub-rotinas `hmd_TIV_optimized` e `vmd_optimized` alocam dinamicamente os arrays `Tudw`, `Txdw`, `Tuup`, `Txup`, `TEdwz`, `TEupz` a cada chamada. Com 600 medições x 2 frequências, isso resulta em:

```
2 sub-rotinas x 6 arrays x 600 medições x 2 frequencias = 7.200 alocações/modelo
```

Em regiões OpenMP paralelas, multiplas threads competem simultaneamente pelo alocador de heap do sistema operacional — gerando **contenção de mutex**, **fragmentação de memoria** e latência de alocação variável.

#### Causa-Raiz 2 — Redundancia Computacional em `commonarraysMD` (Impacto: Muito Alto)

A sub-rotina `commonarraysMD` calcula os coeficientes de reflexão TE/TM e as constantes de propagação, que dependem de:

```
f(r, freq, eta(n,2), zeta) -> u(npt,n), s(npt,n), RTEdw(npt,n), RTEup(npt,n), ...
```

Em perfilagem de poco com passo fixo `p_med`, o espaçamento T-R `r = dTR * |sin(theta)|` é **invariante** por construcao. Com `nf = 2` frequências, `commonarraysMD` produz resultados identicos para todas as 600 medições do mesmo ângulo e frequência. O código original a chama **1.200 vezes por modelo** com argumentos identicos — **trabalho 100% redundante**.

#### Causa-Raiz 3 — Escalonador `dynamic` para Carga Uniforme (Impacto: Baixo)

O escalonador `dynamic` distribui iterações do loop as threads sob demanda, mantendo uma fila gerenciada por mutex. Para 600 medições com custo computacional semelhante, o escalonador `static` distribui as iterações em tempo de compilação — eliminando o overhead de sincronizacao.

#### Causa-Raiz 4 — Redundancia em `commonfactorsMD` por `camadT` Invariante (Impacto: Medio)

A sub-rotina `commonfactorsMD` calcula os fatores `Mxdw`, `Mxup`, `Eudw`, `Euup`, `FEdwz`, `FEupz` que dependem de `(camadT, freq, h0)`. Em formacoes com camadas espessas (> 2 m), medições consecutivas posicionam o transmissor na mesma camada (`camadT` constante) mas em profundidades **diferentes** (`h0` varia com `j`). Conforme descoberto na Fase 6, o caching por `camadT` sozinho é **insuficiente** — `h0` impede reutilização direta.

---

### 12.3 Custo Computacional por Sub-rotina

#### 12.3.1 Complexidade por Modelo

```
Computacoes por modelo:
  ntheta x nfreq x nmed x custo(commonarraysMD + commonfactorsMD + hmd + vmd + rotacao)
  = 1 x 2 x 600 x (~201 x n operações complexas)
  = 1.200 x (~201 x 10 camadas = 2.010 FLOPs complexos)
  ~ 2,4 x 10^6 operações complexas por modelo

Total para 1.000 modelos: ~ 2,4 x 10^9 operações complexas
Tempo estimado (1 core, ~1 GFLOP/s complexo): ~2,4 segundos por modelo
```

#### 12.3.2 Distribuição de Tempo por Sub-rotina

```
  +--------------------------------------------------------------------+
  |  DISTRIBUICAO DE TEMPO — BASELINE (16 threads, 1 angulo, 2 freq)    |
  |                                                                    |
  |  commonarraysMD       ######################## 45%  (gargalo 1)   |
  |  hmd_TIV_optimized    ################ 30%          (gargalo 2)   |
  |    - reducao Hankel     ###### ~25% do total                      |
  |    - overhead malloc    #### ~5% do total                         |
  |  commonfactorsMD      ######## 15%                                |
  |  vmd_optimized        ##### ~7%                                   |
  |  rotacao tensorial    ## ~3%                                      |
  |                                                                    |
  |  commonarraysMD + hmd = ~75% do tempo -> alvos prioritarios       |
  +--------------------------------------------------------------------+
```

#### 12.3.3 O Nucleo Computacional: Filtro de Hankel de Werthmuller

O núcleo de `hmd_TIV_optimized` e `vmd_optimized` são reduções sobre os 201 pontos do filtro de Hankel:

```fortran
! Reducao critica:
kernelHxJ1 = (twox2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twoy2_r2m1 * sum(Ktm_J1)) / r
kernelHxJ0 = x2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * y2_r2 * sum(Ktm_J0 * kr)
```

Com `npt = 201` pontos e aritmetica complexa de precisão dupla (complex(8)), cada `sum()` é uma redução sobre vetores complexos de 201 elementos de 16 bytes. O `npt = 201 = 25 x 8 + 1`, significando 25 iterações vetorizadas com AVX-512 (8 doubles por ciclo) + 1 iteração escalar residual.

---

### 12.4 Mapa de Oportunidades de Otimização

#### 12.4.1 Otimizacoes de Tempo de Execução

| # | Otimização | Impacto Estimado | Sub-rotina Alvo | Natureza |
|:-:|:-----------|:----------------:|:----------------|:--------:|
| 1 | Cache de `commonarraysMD` por `(r, freq)` | **30-50% redução** | `fieldsinfreqs` | Redundancia |
| 2 | Pre-alocação de workspace por thread | **Alto** | `hmd/vmd_optimized` | Memoria |
| 3 | SIMD para convolucao Hankel | 10-30% | `hmd/vmd_optimized` | Vetorizacao |
| 4 | Colapso de lacos `collapse(2/3)` | 10-20% adicional | `perfila1DanisoOMP` | Estrutura |
| 5 | Escalonador hibrido static/dynamic | 5-10% | `perfila1DanisoOMP` | Overhead |
| 6 | Cache de `commonfactorsMD` por `camadT` | 15-25% | `fieldsinfreqs` | Redundancia |
| 7 | Paralelismo sobre frequências | 2x para `nf=2` | `perfila1DanisoOMP` | Estrutura |

#### 12.4.2 Otimizacoes de Memoria

| Otimização | Impacto | Complexidade |
|:-----------|:-------:|:------------:|
| Pre-alocar arrays de trabalho por thread | **Alto** — elimina `malloc` no laco | Media |
| Mover `commonarraysMD` para fora do laco `j` | **Alto** — computa uma vez por `(angulo, freq)` | Media |
| `firstprivate` para arrays somente-leitura | Medio — evita copias desnecessárias | Baixa |
| Pool de memoria por thread (memory pooling) | **Alto** — elimina fragmentação | Alta |
| Arrays pequenos (H 3x3) em stack | Medio — elimina `allocate` desnecessário | Baixa |
| Alinhamento de 64 bytes para SIMD | Medio — habilita AVX-512 | Baixa |

---

### 12.5 Arquitetura Paralela Otimizada

#### 12.5.1 Arquitetura-Alvo

```
  +------------------------------------------------------------------------+
  |  ARQUITETURA-ALVO — PARALELISMO OTIMIZADO                              |
  |                                                                        |
  |  ANTES DA REGIAO PARALELA (serial, uma vez por modelo):               |
  |    1 precompute_common_arrays(nf=2, r=dTR*|sin(theta)|)               |
  |       -> elimina 1.198 das 1.200 chamadas a commonarraysMD            |
  |                                                                        |
  |  ANTES DO LACO PARALELO (alocacao unica):                             |
  |    2 allocate(ws_pool(0:maxthreads-1))                                |
  |       -> ~720 KB x 8 threads = ~5,6 MB total (cabe na L3 cache)      |
  |                                                                        |
  |  LACO PARALELO (adaptativo):                                          |
  |    3 if (ntheta > 1) fork outer parallel do                           |
  |       do k = 1, ntheta                                                |
  |         commonarraysMD x nf [serial, dentro de cada k]                |
  |         !$omp parallel do schedule(guided, 16)                        |
  |         do j = 1, nmed(k)                                             |
  |           tid = [adaptativo: direto ou ancestor-based]                |
  |           call fieldsinfreqs_cached_ws(ws_pool(tid), caches, ...)     |
  |         end do                                                        |
  |       end do                                                          |
  +------------------------------------------------------------------------+
```

#### 12.5.2 Tipo `thread_workspace` (Fase 3 + 3b)

A estrutura de workspace por thread agrega todos os arrays que antes eram alocados dinamicamente a cada chamada:

```fortran
type :: thread_workspace
    ! -- Fase 3 -- arrays de transmissao/potencial (npt x n) --
    complex(dp), allocatable :: Tudw(:,:)   ! (npt, 1:n) coef. transmissao TE desc.
    complex(dp), allocatable :: Txdw(:,:)   ! (npt, 1:n) coef. transmissao TM desc.
    complex(dp), allocatable :: Tuup(:,:)   ! (npt, 1:n) coef. transmissao TE asc.
    complex(dp), allocatable :: Txup(:,:)   ! (npt, 1:n) coef. transmissao TM asc.
    complex(dp), allocatable :: TEdwz(:,:)  ! (npt, 1:n) potencial VMD TE z desc.
    complex(dp), allocatable :: TEupz(:,:)  ! (npt, 1:n) potencial VMD TE z asc.

    ! -- Fase 3b -- fatores de onda de commonfactorsMD (npt) --
    complex(dp), allocatable :: Mxdw(:)     ! (npt) fator reflexao TM desc.
    complex(dp), allocatable :: Mxup(:)     ! (npt) fator reflexao TM asc.
    complex(dp), allocatable :: Eudw(:)     ! (npt) fator reflexao TE desc.
    complex(dp), allocatable :: Euup(:)     ! (npt) fator reflexao TE asc.
    complex(dp), allocatable :: FEdwz(:)    ! (npt) fator TE z-potencial desc. (VMD)
    complex(dp), allocatable :: FEupz(:)    ! (npt) fator TE z-potencial asc. (VMD)
end type thread_workspace
```

**Memoria estimada por thread** (n=15, npt=201):
- 6 arrays (npt x n) x 16 bytes = 6 x 201 x 15 x 16 = ~289 KB
- 6 arrays (npt) x 16 bytes = 6 x 201 x 16 = ~19 KB
- **Total por thread: ~308 KB** (~5,6 MB para 8 threads em n=15)

#### 12.5.3 Caches de `commonarraysMD` (Fase 4)

9 arrays de dimensão `(npt, n, nf)` alocados no heap uma única vez por modelo:

```fortran
! Alocacao (fora do loop paralelo):
allocate(u_cache     (npt, n, nf))
allocate(s_cache     (npt, n, nf))
allocate(uh_cache    (npt, n, nf))
allocate(sh_cache    (npt, n, nf))
allocate(RTEdw_cache (npt, n, nf))
allocate(RTEup_cache (npt, n, nf))
allocate(RTMdw_cache (npt, n, nf))
allocate(RTMup_cache (npt, n, nf))
allocate(AdmInt_cache(npt, n, nf))
```

**Tamanho total:** 9 x (npt x n x nf) x 16 bytes = 9 x 201 x 15 x 2 x 16 = ~870 KB (para n=15, nf=2).

Os caches são preenchidos serialmente dentro do loop `k` (ângulos) e lidos por todas as threads do inner parallel via `shared` clause — read-only, sem race conditions, sem locks.

#### 12.5.4 Stack vs Heap para Arrays Pequenos

Para arrays de tamanho fixo (tensor H 3x3, vetores de componentes), alocação em stack é preferivel:

```fortran
! CORRETO: stack allocation (tamanho fixo, conhecido em compilacao)
complex(dp) :: matH(3,3)
complex(dp) :: Hx_p(1,2), Hy_p(1,2), Hz_p(1,2)

! EVITAR: heap allocation para arrays pequenos
complex(dp), allocatable :: matH(:,:)   ! overhead desnecessario
allocate(matH(3,3))
```

---

### 12.6 Escalabilidade Multi-Core e Lei de Amdahl

#### 12.6.1 Tarefas Disponiveis por Configuração

| Configuração | ntheta | nf | nmed total | Tarefas Totais | Eficiencia (8 cores) |
|:-------------|:------:|:--:|:----------:|:--------------:|:--------------------:|
| Baseline (ntheta=1) | 1 | 2 | 600 | **1.200** | ~95% (150 tarefas/core) |
| Multi-ângulo (ntheta=2) | 2 | 2 | 600+622 | **2.444** | ~98% (305 tarefas/core) |
| Multi-freq extenso | 1 | 4 | 600 | **2.400** | ~97% (300 tarefas/core) |

#### 12.6.2 Afinidade NUMA

```bash
# Single-socket (laptop, workstation, Colab):
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
# Distribui threads uniformemente entre cores para balanceamento termico
# e evita contencao de hyperthreading em operações FP intensivas.

# Dual-socket (servidor HPC com 2 CPUs):
export OMP_PLACES=cores
export OMP_PROC_BIND=close
# Mantem threads no mesmo socket NUMA, reduzindo latencia de
# acesso a memoria de ~300 ns (inter-socket) para ~100 ns (intra-socket).
```

#### 12.6.3 Lei de Amdahl Aplicada

```
P = 1 - fracao_serial

Fracao serial estimada (operações intrinsecamente sequenciais):
  - Leitura/escrita de arquivos: ~1%
  - Calculo de nmed, carga do filtro: ~1%
  - Pre-processamento serial (precompute_common_arrays): ~0,5%
  Total serial: ~2,5%

P ~ 0,975

Speedup maximo teorico (Lei de Amdahl, N=16 cores):
  S_max = 1 / (1 - P + P/N) = 1 / (0,025 + 0,975/16) = 1 / 0,086 ~ 11,6x

Speedup esperado realista (com overhead OpenMP ~5%):
  S_realista ~ 8-10x sobre código single-thread
```

---

### 12.7 Fase 0 — Benchmark de Baseline

**Objetivo:** Estabelecer medidas reproduziveis antes de qualquer alteracao.

**Ambiente de medição:**
- Hardware: Intel Core i9-9980HK (8 cores físicos, 16 logicos), 32 GB RAM
- SO: macOS Darwin 25.4 (macOS 26)
- Compilador: gfortran 15.2.0 Homebrew
- Linker: `ld-classic` (workaround macOS 26)
- Config: `model.in` com 15 camadas, 2 frequências (20/40 kHz), `ntheta=1`, 600 medidas
- Compilação: `-O3 -march=native -ffast-math -funroll-loops -fopenmp -std=f2008`

**Resultados (60 iterações apos 3 warmups, 8 threads):**

| Métrica | Valor |
|:--------|:------|
| Wall-time medio | **0,1047 +/- 0,0147 s/modelo** |
| Mediana | 0,1000 s |
| Throughput | **~34.400 modelos/hora** |
| MD5 de referência | `c64745ed5d69d5f654b0bac7dde23a95` |

**Escalabilidade multi-thread observada:**

| Threads | Wall-time (s) | Speedup | Eficiencia |
|:-------:|:-------------:|:-------:|:----------:|
| 1 | 0,50 | 1,00x | 100 % |
| 2 | 0,53 | 0,94x | 47 % (BUG) |
| 4 | 0,19 | 2,63x | 66 % |
| **8** | **0,12** | **4,17x** | **52 %** |
| 16 | 0,08 | 6,25x | 39 % |

**Anomalia em 2 threads:** O speedup de 0,94x (anti-escalabilidade) decorre de `num_threads_j = maxthreads - ntheta = 2 - 1 = 1`, degenerando o loop interno a uma única thread. Este bug foi o alvo primario da Fase 2.

**Ajuste de premissa:** O baseline previsto no roteiro era 2,40 s/modelo (24.000 mod/h). O medido foi 0,1047 s/modelo (34.400 mod/h) — 23x mais rapido. A discrepancia decorre de: (a) `ntheta=1` vs `ntheta > 1` assumido; (b) hardware mais recente; (c) gfortran 15.2.0 auto-vetoriza melhor.

**Commit:** `43709bf`
**Relatorio:** `docs/reference/relatorio_fase0_fase1_fortran.md`

---

### 12.8 Fase 1 — SIMD Hankel (Pulada)

**Objetivo:** Substituir os `sum()` intrinsecos sobre arrays complexos de 201 elementos por lacos explícitos com diretiva `!$omp simd`, habilitando vetorizacao AVX-2/AVX-512.

**Implementação testada:** Refatoracao de `commonarraysMD` substituindo array-syntax por loops `do ipt=1,npt` explícitos com `!$omp simd`, com hoisting de invariantes.

**Benchmark interleaved (60 iterações alternadas):**

| Métrica | Baseline | Fase 1 | Delta |
|:--------|:---------|:-------|:------|
| Wall-time medio (s) | 0,1047 +/- 0,015 | 0,1057 +/- 0,011 | +0,96 % |
| Welch *t*-statistic | — | +0,425 | nao-signif. |

**Resultado:** Delta estatísticamente insignificante (`|t| < 2`). A Fase 1 **não entregou o ganho previsto** de +15 a +30%.

**Causa raiz:** gfortran 15.2.0 **já auto-vetoriza** os loops de `commonarraysMD` e `magneticdipoles.f08` com vetores de 32 bytes (máximo AVX-2 = 4 doubles). Não ha margem para SIMD explícito nesta classe de CPU. Confirmado por `-fopt-info-vec=vec.txt`.

**Validação numérica:** `max|Delta| = 1,93 x 10^-13` (muito abaixo de `atol=1e-10`). Código numéricamente equivalente.

**Decisao:** Experimento **rejeitado** — arquivado em `Fortran_Gerador/bench/attic/phase1_simd.patch`. Re-tentar apenas em hardware AVX-512 (Xeon Scalable, Ice Lake-SP).

**Commit:** `43709bf` (não aplicado a produção)
**Relatorio:** `docs/reference/relatorio_fase0_fase1_fortran.md` S3

---

### 12.9 Fase 2 — Hybrid Scheduler + Debitos 1/2/3

**Objetivo:** Corrigir o bug de particionamento de threads e migrar para APIs OpenMP modernas.

**Intervencoes aplicadas:**

1. **Debito 1** (`writes_files` append bug): Abertura condicional com `inquire()` + deteccao de `modelm==1` OR arquivo ausente. Eliminacao do bug de concatenacao silenciosa.

2. **Debito 2** (`omp_set_nested` depreciado): Migracao para `omp_set_max_active_levels(2)` (OpenMP 5.0+).

3. **Debito 3 + Fase 2** (particionamento de threads): Substituicao do cálculo subtrativo buggado `num_threads_j = maxthreads - ntheta` por particionamento MULTIPLICATIVO:

```fortran
! ANTES (buggado):
num_threads_k = ntheta                     ! = 1
num_threads_j = maxthreads - ntheta        ! = 1 quando OMP=2 !!

! DEPOIS (Fase 2 + Debito 3):
call omp_set_max_active_levels(2)          ! OpenMP 5.0+
num_threads_k = max(1, min(ntheta, maxthreads))
num_threads_j = max(1, maxthreads / num_threads_k)
```

**Tabela de distribuição válidada:**

| maxthreads | ntheta | n_k | n_j | n_k x n_j | Nota |
|:----------:|:------:|:---:|:---:|:---------:|:-----|
| 2 | 1 | 1 | 2 | 2 | FIX: antes era 1 |
| 2 | 2 | 2 | 1 | 2 | |
| 8 | 1 | 1 | 8 | 8 | |
| 8 | 2 | 2 | 4 | 8 | |
| 8 | 7 | 7 | 1 | 7 | |
| 16 | 1 | 1 | 16 | 16 | |
| 16 | 7 | 7 | 2 | 14 | |

**Schedule hibrido:** `dynamic` no loop externo de ângulos (carga desigual) + `static` no loop interno de medidas (carga uniforme).

**Scaling Test (2 warmups + 5 medições/ponto):**

| OMP_NUM_THREADS | Baseline (s) | Fase 2 (s) | Delta% | Speedup Baseline | Speedup Fase 2 | Avaliacao |
|:---------------:|:------------:|:----------:|:------:|:----------------:|:--------------:|:----------|
| 1 | 1,432 | 1,254 | **-12,4 %** | 1,00x | 1,00x | Fase 2 melhora |
| **2** | **1,340** | **0,786** | **-41,3 %** | **1,07x** | **1,60x** | **Bug corrigido** |
| 4 | 0,522 | 0,400 | **-23,4 %** | 2,74x | 3,13x | Fase 2 melhora |
| 8 | 0,276 | 0,306 | +10,9 % | 5,19x | 4,10x | Trade-off marginal |
| 16 | 0,226 | 0,240 | +6,2 % | 6,34x | 5,23x | Trade-off marginal |

**Validação numérica:** `max|Delta| = 0,0000e+00` em todas as 21 colunas. MD5 identico entre baseline e Fase 2 (`c64745ed5d69d5f654b0bac7dde23a95`). Reprodutibilidade **bit-a-bit exata**.

**Trade-off em 8-16 threads:** A substituicao de `schedule(dynamic)` por `schedule(static)` no loop interno causou regressão marginal de 6-11% em alta concorrencia — mitigado posteriormente na Fase 2b com `schedule(guided, 16)`.

**Commit:** `6ac51ca`
**Relatorio:** `docs/reference/relatorio_fase2_debitos_fortran.md`

---

### 12.10 Fase 3 — Workspace Pre-allocation

**Objetivo:** Eliminar todas as chamadas a `allocate/deallocate` dentro do laco paralelo, pre-alocando um workspace exclusivo por thread antes do laco.

**Arquivos modificados:**
- `magneticdipoles.f08`: tipo `thread_workspace` (6 campos), sub-rotinas `hmd_TIV_optimized_ws` e `vmd_optimized_ws`
- `PerfilaAnisoOmp.f08`: alocação de `ws_pool`, sub-rotina `fieldsinfreqs_ws`

**Estrategia:** Criar versoes com sufixo `_ws` das sub-rotinas (preservando originais para rollback). O `type :: thread_workspace` foi adicionado com 6 componentes (`Tudw, Txdw, Tuup, Txup, TEdwz, TEupz`), focando exclusivamente nos arrays dinamicamente alocados no hot path.

**Resultados a 8 threads (60 iterações, model.in n=29):**

| Métrica | Fase 2 (HEAD) | Fase 3 | Delta |
|:--------|:-------------:|:------:|:-----:|
| Tempo medio (s/modelo) | 0,3830 | **0,3433** | -10,4 % |
| Desvio-padrao (s) | 0,0355 | 0,0278 | -21,7 % |
| Throughput (modelos/h) | 9.399 | **10.485** | **+11,5 %** |

**Escalabilidade 1 -> 8 threads (30 iterações por ponto):**

| Threads | Fase 2 tempo (s) | Fase 3 tempo (s) | Fase 2 thput | Fase 3 thput | Speedup 3 vs 2 |
|:-------:|:----------------:|:----------------:|:------------:|:------------:|:--------------:|
| **1** | 1,7800 | **1,3690** | 2.023 | **2.630** | **1,30x** |
| **2** | 0,8073 | **0,7500** | 4.459 | **4.800** | 1,08x |
| **4** | 0,5873 | **0,4617** | 6.129 | **7.798** | **1,27x** |
| **8** | 0,3830 | **0,3433** | 9.399 | **10.485** | 1,12x |

**Validação numérica:**
- Fase 3 vs HEAD com `-O0`: **MD5 binario identico** (`8aa4aeee...`) — equivalencia matemática bit-exata.
- Fase 3 vs HEAD com `-O3 -ffast-math`: `max|Delta| = 3,4 x 10^-14`, `RMS(Delta) = 1,7 x 10^-15`, `max rel Delta = 1,2 x 10^-9` — quatro ordens de magnitude abaixo do criterio `1 x 10^-10`.

**Contagem de malloc eliminados:**

| Localização | Chamadas antes | Chamadas depois | Reducao |
|:------------|:--------------:|:---------------:|:-------:|
| `hmd_TIV_optimized` (4 arrays) | ~4.800/modelo | 0 | 100 % |
| `vmd_optimized` (2 arrays) | ~2.400/modelo | 0 | 100 % |
| **Total no hot path** | **~7.200/modelo** | **6 (1 vez por modelo)** | **99,92 %** |

**Análise vs meta:** O plano projetava +40% a +80%. O obtido foi **+30,1% em serial** (onde so conta overhead de malloc) e **+11,5% em 8 threads** (contenção mitigada mas custo matematico domina). A diferença decorre de `n=29` camadas (vs `n=10` calibrado no plano): o custo matematico cresce linearmente enquanto o custo de malloc permanece constante.

**Commit:** `c213b66`
**Relatorio:** `docs/reference/relatorio_fase3_fortran.md`

---

### 12.11 PR1 Hygiene — Debitos D4/D5/D6

**Objetivo:** Corrigir debitos tecnicos OpenMP identificados durante a Fase 3.

| # | Descricao | Correcao |
|:-:|:----------|:---------|
| **D4** | `private(z_rho1, c_H1)` com `allocatable`: copias privadas tem status de alocação indefinido por spec OpenMP 5.x | `firstprivate(z_rho1, c_H1)` — copias herdam alocação+valores do master. Custo: ~32 KB copiados por thread 1x por região paralela |
| **D5** | `!$omp barrier` orfao fora de região paralela: ignorado por gfortran, erro em outros compiladores | Linha removida — barreira implícita do `end parallel do` é suficiente |
| **D6** | `omp_get_thread_num()` dentro do inner team retorna tid local `[0, num_threads_j-1]`, não global: race em `ws_pool(0)` se `num_threads_k > 1` | `tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()` — índice global `[0, maxthreads-1]`. Backward-compatible: com `num_threads_k=1`, `ancestor(1)=0` |

**Impacto em runtime:** Zero para `ntheta=1` (produção atual). Validado por MD5 bit-exato em 1/2/4/8 threads (`aadbc86be2af5e1fd300f535d7e80e3b`). Pre-requisito estrutural para ativar multi-ângulo.

**Commit:** `db997d2`

---

### 12.12 Fase 4 — Cache `commonarraysMD`

**Objetivo:** Pre-computar os resultados de `commonarraysMD` uma única vez por frequência (2 chamadas por modelo) em vez de 1.200 chamadas por modelo.

**Observacao-chave:** `commonarraysMD(n, npt, r, krJ0J1, zeta, h, eta, ...)` depende apenas de `(r, freq, n, h, eta)` — todos **invariantes em `j`** (medidas). O valor `r = dTR * |sin(theta_k)|` é constante por ângulo (translacao rigida da ferramenta LWD).

**Implementação:**
1. 9 arrays de cache `(npt, n, nf)` alocados no heap uma vez por modelo
2. Pre-computo serial dentro do loop `k`: `nf` chamadas a `commonarraysMD` por ângulo
3. Nova sub-rotina `fieldsinfreqs_cached_ws` recebe caches como `intent(in)`
4. `eta_shared(n, 2)` hoisted para escopo de `perfila1DanisoOMP` (Debito B2)

```fortran
! Pre-computo serial (uma vez por angulo k):
r_k = dTR(itr) * dabs(seno)
do ii = 1, nf
  omega_i = 2.d0 * pi * freq(ii)
  zeta_i  = cmplx(0.d0, 1.d0, kind=dp) * omega_i * mu
  call commonarraysMD(n, npt, r_k, krwJ0J1(:,1), zeta_i, h, eta_shared, &
                      u_cache(:,:,ii), s_cache(:,:,ii), ...)
end do
```

**Resultados a 8 threads (60 iterações, model.in n=15):**

| Métrica | Fase 3 | **Fase 4** | Delta |
|:--------|:------:|:----------:|:-----:|
| Tempo medio (s/modelo) | 0,3433 | **0,0620** | **-81,9 %** |
| Throughput (modelos/h) | 10.485 | **58.064** | **+453,8 %** |
| Speedup | 1,00x (ref.) | **5,54x** | — |

**Escalabilidade 1 -> 8 threads (30 iterações por ponto):**

| Threads | Fase 3 tempo (s) | Fase 4 tempo (s) | Fase 4 thput (mod/h) | Speedup 4 vs 3 |
|:-------:|:----------------:|:----------------:|:--------------------:|:--------------:|
| **1** | 1,3690 | **0,2393** | 15.042 | **5,72x** |
| **2** | 0,7500 | **0,1390** | 25.899 | 5,40x |
| **4** | 0,4617 | **0,0820** | 43.902 | **5,63x** |
| **8** | 0,3433 | **0,0620** | **58.064** | 5,54x |

**Contagem de chamadas a commonarraysMD:**

| Momento | Chamadas/modelo | Reducao |
|:--------|:---------------:|:-------:|
| Fase 3 | `nf x nmed = 1.200` | — |
| Fase 4 | `ntheta x nf = 2` | **99,83 %** |

**Validação numérica:**
- `-O3 -ffast-math` vs Fase 3: `max|Delta| = 3,97 x 10^-13`, `RMS(Delta) = 2,02 x 10^-14`, `max rel Delta = 1,01 x 10^-8`. Tres ordens de magnitude abaixo do criterio `1 x 10^-10`.
- MD5 determinístico em 1/2/4/8 threads: `1e4f36fa8f0bcd21f3700dc445c2894d`.

**Análise vs meta:** O plano projetava speedup de 1,6-2,2x. O obtido foi **5,54x**, muito acima. A explicação é que o profile gprof inicial subestimava `commonarraysMD` em ~45% quando na verdade dominava ~82% do tempo.

**Novo gargalo:** Com `commonarraysMD` eliminada, `commonfactorsMD` passa a dominar ~40-50% do tempo restante — alvo da Fase 6.

**Commit:** `44acf2e`
**Relatorio:** `docs/reference/relatorio_fase4_fortran.md`

---

### 12.13 Validação Final — Testes Bit-Exatos Fases 0->4

**Objetivo:** Fechar lacunas de verificação de fidelidade numérica das Fases 3/PR1/4.

**Infraestrutura construida:**
- Target `debug_O0` no Makefile (`-O0 -g -fno-fast-math -fsignaling-nans`)
- Worktree git isolado em `6ac51ca` (Fase 2) para compilação de referência
- `model.in.n10_synthetic` para testes em configuração alternativa (10 camadas)
- Script `bench/validate_numeric_extensive.py`: parser de formato binario, reporta `max|Delta|`, `RMS(Delta)`, NaN/Inf

**Matriz de resultados:**

| Teste | Flags | Modelo | Threads | max\\|Delta\\| | Status |
|:------|:------|:------:|:-------:|:--------:|:------:|
| Fase 4 vs Fase 2 (bit-exato) | `-O0 -fno-fast-math` | n=15 prod | 1 | **0** | BIT-EXATO |
| Fase 4 vs Fase 2 (bit-exato) | `-O0 -fno-fast-math` | n=10 sint | 1 | **0** | BIT-EXATO |
| Fase 4 vs Fase 2 (sub-ULP) | `-O3 -ffast-math` | n=15 prod | 1 | 1,96e-13 | PASS |
| Fase 4 vs Fase 2 (sub-ULP) | `-O3 -ffast-math` | n=10 sint | 1 | 6,11e-14 | PASS |
| Determinismo Fase 4 | `-O3 -ffast-math` | n=15 prod | {1,2,4,8} | — | MD5 identico |

**Conclusões:**
1. Equivalencia matemática entre Fase 2 e Fase 4 é **bit-a-bit exata** em ausencia de reordenamento FP (`-O0 -fno-fast-math`).
2. O `max|Delta|` em produção (~2 x 10^-13) esta **tres ordens de magnitude** abaixo do criterio `1 x 10^-10`, e **nove ordens** abaixo do ruido LWD físico tipico (~1 A/m).
3. Zero NaN e zero Inf em todas as saídas — fidelidade física total.
4. Determinismo entre thread counts (1/2/4/8) confirmado.
5. Revisão de código: zero `allocate/deallocate` dentro de regiões `!$omp parallel do`; D4, D5, D6 corretamente aplicados.

**Commit:** `195b69f`
**Relatorio:** `docs/reference/relatorio_validacao_final_fortran.md`

---

### 12.14 Fase 5 / PR2 — Single-Level + Debitos B1/B3/B5/B7

#### 12.14.1 Debitos Corrigidos

| ID | Correcao | Impacto |
|:--:|:---------|:--------|
| B1 | Copia redundante `krJ0J1/wJ0/wJ1 = krwJ0J1(:,1..3)` eliminada em `fieldsinfreqs_cached_ws`. Slices passadas diretamente | -2,9 MB/modelo de copias desnecessárias |
| B3/D7 | `private(zrho, cH)` migrado para `firstprivate(zrho, cH)` no inner parallel | Portabilidade OpenMP allocatable |
| B5 | `if (allocated(krwJ0J1)) deallocate(krwJ0J1)` adicionado | Eliminacao de leak ~4,8 KB/modelo |
| B7 | Atributo `contiguous` adicionado a dummy arguments em `hmd/vmd_optimized_ws` | Previne copia temporaria pelo compilador |

**Validação:** MD5 bit-exato @ `-O0` vs Fase 4, `max|Delta|=0` (correcoes semanticamente neutras).

#### 12.14.2 Fase 5 — Single-Level Parallel

Eliminacao do nested parallelism (outer `!$omp parallel do` + inner `!$omp parallel do`) em favor de loop serial `k` + single `!$omp parallel do` com `maxthreads`:

- `tid = omp_get_thread_num()` direto (sem `omp_get_ancestor_thread_num`)
- Para `ntheta=1`: mesmo efeito que o nested (gfortran 15.2 já otimizava)
- Para `ntheta>1`: loop serial em `k` — restaurar nested planejado como Fase 5b

**Benchmark (i9-9980HK, model.in n=15, 60 iterações):**
- 8 threads: **0,0693 s/modelo** (51.923 mod/h)

**Validação numérica:**
- Fase 5 vs Fase 4 @ `-O0`: `max|Delta| = 0` (bit-exato)
- Fase 5 @ `-O3` vs referência: `max|Delta| = 4,26 x 10^-13` (sub-ULP)
- Determinismo T=1,2,4,8: MD5 `3d3c309fd1aa121f8b4166268552814c`

**Commit:** `fc208c1`
**Relatorio:** `docs/reference/relatorio_fase5_debitos_fortran.md`

---

### 12.15 Fase 3b — Workspace Estendido (12 campos)

Expansão do tipo `thread_workspace` de 6 para **12 campos**, adicionando os 6 arrays de fatores de onda de `commonfactorsMD`:

```
Novos campos (Fase 3b):
  ws%Mxdw (npt)    ws%Mxup (npt)
  ws%Eudw (npt)    ws%Euup (npt)
  ws%FEdwz(npt)    ws%FEupz(npt)
```

Antes eram automatic arrays no stack de `fieldsinfreqs_cached_ws` (~19 KB/thread). Para `n >= 30` camadas com muitos threads, a pressão de stack se acumula. Movidos para heap via workspace.

**Custo:** +19 KB x 8 threads = 155 KB de heap adicional (irrelevante).
**Validação:** bit-exato @ `-O0` vs Fase 5. Zero impacto em performance.
**Commit:** `0aa0aa9`

---

### 12.16 Fase 5b — Paralelismo Adaptativo `if(ntheta > 1)`

Restauracao do outer `!$omp parallel do` com clausula `if(ntheta > 1)`:

```fortran
!$omp parallel do schedule(dynamic) num_threads(num_threads_k) &
!$omp&        if(ntheta > 1) &
!$omp&        private(k,ang,seno,coss,px,pz,Lsen,Lcos,r_k,omega_i,zeta_i,ii) &
!$omp&        firstprivate(z_rho1,c_H1)
do k = 1, ntheta
  ! ...
  !$omp parallel do schedule(guided, 16) &
  !$omp&        num_threads(merge(maxthreads, num_threads_j, ntheta == 1)) &
  !$omp&        default(shared) &
  !$omp&        private(j, x, y, z, Tx, Ty, Tz, posTR, tid) &
  !$omp&        firstprivate(zrho, cH)
  do j = 1, nmed(k)
    if (ntheta == 1) then
      tid = omp_get_thread_num()
    else
      tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()
    end if
    ! ...
  end do
  !$omp end parallel do
end do
!$omp end parallel do
```

**Comportamento em runtime:**

```
  +----------------------------------------------------------------------+
  |  ntheta = 1 (producao: perfilagem a 0 graus)                        |
  |    -> Loop k serial (1 iteracao), inner parallel j com maxthreads   |
  |    -> tid = omp_get_thread_num() direto                             |
  |    -> Sem overhead de nested fork/join                              |
  |                                                                      |
  |  ntheta > 1 (multi-angulo: geosteering)                             |
  |    -> Outer parallel k com num_threads_k threads (schedule dynamic) |
  |    -> Inner parallel j com num_threads_j threads (schedule guided)  |
  |    -> tid = ancestor(1) * num_threads_j + thread_num()              |
  |    -> Pre-computo commonarraysMD serial dentro de cada k            |
  +----------------------------------------------------------------------+
```

**Benchmark (i9-9980HK, n=15, 60 iterações, 8 threads):**
- Mean: **0,0668 s/modelo** (53.865 mod/h) — +3,7% vs Fase 5

**Validação:** bit-exato @ `-O0`; determinismo 1/2/4/8 threads.
**Commit:** `0aa0aa9`

---

### 12.17 Fase 2b — `schedule(guided, 16)`

Migracao do inner `!$omp parallel do` de `schedule(static)` para `schedule(guided, 16)`.

**Justificativa:** `guided` atribui chunks decrescentes (inicial ~ nmed/nthreads, mínimo 16 iterações), melhorando balanceamento em:
- Regimes degradados (poucos threads, nmed grande)
- Multi-ângulo futuro com `nmed(k)` variável
- Custo nao-uniforme por iteração (`commonfactorsMD` varia com posição na camada)

Chunk mínimo 16 preserva localidade de cache L1 (~16 x ~19 KB = ~300 KB/chunk).

**Benchmark (i9-9980HK, n=15, 60 iterações, 8 threads):**
- Mean: **0,0612 s/modelo** (58.856 mod/h) — **+9,3%** vs Fase 5b

**Validação:**
- Bit-exato @ `-O0`: `97123697a2e4db34c77cd1d84077b083` (identico ao baseline Fase 2)
- Determinismo 1/2/4/8 threads

**Commit:** `8de722e`

---

### 12.18 Fase 6 / 6b — Cache `commonfactorsMD` (Erro + Descartada)

#### 12.18.1 Fase 6 — Erro Conceitual Identificado

A proposta original de cache por sentinela (`if camadT == camadT_prev then copy`) continha um **erro conceitual fundamental**: `commonfactorsMD` depende de `h0 = Tz` (profundidade do transmissor), que **varia a cada medida `j`**, não apenas de `camadT`. Os termos exponenciais `exp(-s(:,cT) * (prof(cT) - h0))`, `exp(s(:,cT) * (prof(cT-1) - h0))`, etc., são **todos funções de `h0`**. Copiar o resultado da medida `j-1` para `j` quando `camadT` não muda é **matemáticamente incorreto**.

**Status:** Não implementada.

#### 12.18.2 Fase 6b — Fatorizacao de Invariantes `h0` (Descartada por NaN)

**Tentativa:** Fatorar os 14 `exp()` de `commonfactorsMD` em 10 coeficientes invariantes (dependem de `camadT`) + 4 `exp()` variantes (dependem de `h0`), com sentinel por thread.

**Resultado: INSTABILIDADE NUMÉRICA FATAL.** A separação `exp(-s*(prof(cT) - h0))` em `exp(-s*prof(cT)) * exp(s*h0)` causa overflow individual quando `s` é grande e `prof(cT)` é profundo, mesmo que o argumento original `(-s*(prof(cT) - h0))` sejá finito por cancelamento (`h0 ~ prof(cT)` quando o transmissor esta próximo da interface).

```
Resultado: 21.600 NaN em 25.200 saidas do .dat
```

**Licao aprendida:** Exponenciais com argumentos de sinal oposto que se cancelam (`exp(-s*a) * exp(s*b)` com `a ~ b`) **NAO** podem ser separadas sem tratamento especial (e.g., log-sum-exp scaling). A versão original `exp(-s*(a - b))` mantem o cancelamento **ANTES** do `exp()`, preservando magnitude finita.

**Status:** `commonfactorsMD` permanece como hot path inline sem caching. O ganho seria ~30-50% se implementavel, mas a **fidelidade física é nao-negociavel**.

---

### 12.19 Tabela Consolidada de Performance — Todas as Fases

| Fase | Commit | Tempo/modelo (8t) | Throughput (mod/h) | Speedup vs Baseline | Validação @ -O0 |
|:----:|:------:|:------------------:|:------------------:|:-------------------:|:---------------:|
| **0 (Baseline)** | `43709bf` | 0,1047 s | 34.400 | 1,0x | `c64745ed...` |
| **1 (SIMD)** | — | — | — | Pulada | — |
| **2 (Hybrid)** | `6ac51ca` | ~0,100 s | ~36.000 | ~1,05x | MD5 = Fase 0 |
| **3 (Workspace)** | `c213b66` | 0,343 s* | 10.485* | — | `max\\|Delta\\|=3,4e-14` |
| **PR1 (D4/D5/D6)** | `db997d2` | = Fase 3 | = Fase 3 | — | MD5 = Fase 3 |
| **4 (Cache)** | `44acf2e` | **0,062 s** | **58.064** | **5,54x** | `97123697...` |
| **Validação** | `195b69f` | — | — | — | bit-exato confirmado |
| **PR2 (B1-B7)** | `fc208c1` | = Fase 4 | = Fase 4 | — | MD5 = Fase 4 |
| **5 (Single)** | `fc208c1` | 0,069 s | 51.923 | ~5,0x | `ffd13177...` |
| **3b (WS ext.)** | `0aa0aa9` | = Fase 5 | = Fase 5 | — | MD5 = Fase 5 |
| **5b (Adaptativo)** | `0aa0aa9` | 0,067 s | 53.865 | ~5,1x | MD5 = Fase 5 |
| **2b (Guided)** | `8de722e` | **0,061 s** | **58.856** | **5,62x** | `97123697...` |
| **6 (Erro)** | — | — | — | — | Não implementada |
| **6b (NaN)** | — | — | — | — | 21.600 NaN |

*\\* Fase 3 medida com `model.in` diferente (n=29 vs n=15 das demais); valores não diretamente comparaveis.*

**Estado final:** 0,061 s/modelo, **58.856 mod/h**, **245% da meta original** (24.000 mod/h). Fidelidade numérica: bit-exato @ `-O0` vs código original (Fase 2), `max|Delta| ~ 4 x 10^-13` @ `-O3 -ffast-math`.

---

### 12.20 Diretivas OpenMP Detalhadas

Referencia completa de todas as diretivas OpenMP utilizadas no código de produção atual (`PerfilaAnisoOmp.f08`, versão pos-Fase 2b).

#### 12.20.1 Configuração Global

```fortran
! Habilitacao de nested parallelism (Fase 2, Debito 2):
call omp_set_max_active_levels(2)
nested_enabled = (omp_get_max_active_levels() >= 2)

! Particionamento multiplicativo de threads (Fase 2 + Debito 3):
maxthreads = omp_get_max_threads()
num_threads_k = max(1, min(ntheta, maxthreads))
num_threads_j = max(1, maxthreads / num_threads_k)
```

#### 12.20.2 Outer Parallel Do (Loop de Ângulos)

```fortran
!$omp parallel do schedule(dynamic) num_threads(num_threads_k) &
!$omp&        if(ntheta > 1) &
!$omp&        private(k,ang,seno,coss,px,pz,Lsen,Lcos,r_k,omega_i,zeta_i,ii) &
!$omp&        firstprivate(z_rho1,c_H1)
do k = 1, ntheta
```

| Clausula | Valor | Justificativa |
|:---------|:------|:-------------|
| `schedule(dynamic)` | chunk=1 implícito | Carga desigual: `nmed(k)` varia com `theta(k)` |
| `num_threads(num_threads_k)` | `min(ntheta, maxthreads)` | Não mais threads que ângulos |
| `if(ntheta > 1)` | Condicional | Fase 5b: evita fork/join quando ntheta=1 |
| `private(...)` | Escalares geométricos | Cada thread computa seus proprios ângulos |
| `firstprivate(z_rho1,c_H1)` | Arrays allocatable | D4: heranca de alocação do master |

#### 12.20.3 Inner Parallel Do (Loop de Medidas)

```fortran
!$omp parallel do schedule(guided, 16) &
!$omp&        num_threads(merge(maxthreads, num_threads_j, ntheta == 1)) &
!$omp&        default(shared) &
!$omp&        private(j, x, y, z, Tx, Ty, Tz, posTR, tid) &
!$omp&        firstprivate(zrho, cH)
do j = 1, nmed(k)
```

| Clausula | Valor | Justificativa |
|:---------|:------|:-------------|
| `schedule(guided, 16)` | Chunks decrescentes, min=16 | Fase 2b: balanceamento adaptativo preservando localidade L1 |
| `num_threads(merge(...))` | `maxthreads` (ntheta=1) ou `num_threads_j` (ntheta>1) | Fase 5b: adaptativo |
| `default(shared)` | Todos os arrays são shared por default | Caches Fase 4 são read-only |
| `private(j, x, y, z, ...)` | Coordenadas geométricas | Cada thread computa posições independentes |
| `firstprivate(zrho, cH)` | Arrays de saída por medida | B3/D7: heranca de alocação allocatable |

#### 12.20.4 Calculo de Thread ID

```fortran
! Fase 5b: tid adaptativo (dentro do inner parallel do):
if (ntheta == 1) then
  tid = omp_get_thread_num()
  ! Single-level: tid direto, range [0, maxthreads-1]
else
  tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()
  ! Nested: tid global, range [0, num_threads_k * num_threads_j - 1]
  !   ancestor_thread_num(1) = tid do outer team (nivel 1)
  !   omp_get_thread_num()   = tid do inner team (nivel 2)
end if
```

#### 12.20.5 APIs OpenMP Utilizadas

| API | Localização | Uso |
|:----|:-----------|:----|
| `omp_set_max_active_levels(2)` | `perfila1DanisoOMP` | Habilitar até 2 niveis de nested parallelism |
| `omp_get_max_active_levels()` | `perfila1DanisoOMP` | Verificar se nested esta habilitado |
| `omp_get_max_threads()` | `perfila1DanisoOMP` | Obter número total de threads disponiveis |
| `omp_get_thread_num()` | Inner parallel do | Obter tid local (single-level) |
| `omp_get_ancestor_thread_num(1)` | Inner parallel do | Obter tid do outer team (nested) |
| `omp_lib` | `use omp_lib` | Módulo com interfaces de todas as rotinas OMP |

#### 12.20.6 Clausulas de Compartilhamento de Dados

```
  +----------------------------------------------------------------------+
  |  RESUMO DE CLAUSULAS DE COMPARTILHAMENTO                             |
  |                                                                      |
  |  Clausula           Variaveis                  Justificativa         |
  |  -----------------------------------------------------------------  |
  |  shared (default)   u_cache, s_cache, ...      Caches Fase 4 (RO)  |
  |                     krwJ0J1, h, prof, resist   Modelo geologico (RO)|
  |                     eta_shared, freq, theta    Parametros (RO)      |
  |                     ws_pool                    Indexado por tid      |
  |                     z_rho1, c_H1 [outer]       Escritos por k       |
  |                                                                      |
  |  private            j, x, y, z, Tx, Ty, Tz    Geometria por medida |
  |                     posTR, tid                 Posicao + thread id  |
  |                     k, ang, seno, coss, ...    Geometria por angulo |
  |                                                                      |
  |  firstprivate       zrho, cH [inner]           D7: herda alocacao  |
  |                     z_rho1, c_H1 [outer]       D4: herda alocacao  |
  +----------------------------------------------------------------------+
```

#### 12.20.7 Variaveis de Ambiente Recomendadas

```bash
# Producao (8 threads, single-socket):
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_STACKSIZE=64M        # Para n >= 30 camadas com automatic arrays

# Debug / validacao:
export OMP_NUM_THREADS=1        # Serial para comparacao bit-exata
export OMP_DISPLAY_ENV=true     # Imprimir config OMP no inicio
```

#### 12.20.8 Flags de Compilação para OpenMP

```makefile
# Producao:
FFLAGS = -O3 -march=native -ffast-math -funroll-loops -fopenmp -std=f2008

# Debug / validacao bit-exata:
FFLAGS_DEBUG = -O0 -g -fno-fast-math -fsignaling-nans -fopenmp -std=f2008 \\
               -fcheck=all -fbacktrace
```

---

### Documentos Relacionados

| Documento | Escopo |
|:----------|:-------|
| [`analise_paralelismo_cpu_fortran.md`](analise_paralelismo_cpu_fortran.md) | Roteiro de 6 fases de otimização CPU, análise de gargalos, resultados empiricos |
| [`analise_evolucao_simulador_fortran_python.md`](analise_evolucao_simulador_fortran_python.md) | Viabilidade de conversão Fortran para Python/JAX/Numba |
| [`relatorio_fase0_fase1_fortran.md`](relatorio_fase0_fase1_fortran.md) | Baseline + Fase 1 SIMD (pulada) |
| [`relatorio_fase2_debitos_fortran.md`](relatorio_fase2_debitos_fortran.md) | Fase 2 + Debitos 1/2/3 |
| [`relatorio_fase3_fortran.md`](relatorio_fase3_fortran.md) | Fase 3 + PR1 Hygiene D4/D5/D6 |
| [`relatorio_fase4_fortran.md`](relatorio_fase4_fortran.md) | Fase 4 Cache commonarraysMD |
| [`relatorio_validacao_final_fortran.md`](relatorio_válidação_final_fortran.md) | Testes bit-exatos Fases 0-4 |
| [`relatorio_fase5_debitos_fortran.md`](relatorio_fase5_debitos_fortran.md) | Fase 5 + Debitos B1-B7 + Fase 6 erro |

---

### Histórico de Execução de Otimizacoes

| Data | Fase | Status | Relatorio / Commit |
|:-----|:-----|:-------|:-------------------|
| 2026-04-04 | CPU Fase 0 — Baseline | Concluida | `relatorio_fase0_fase1_fortran.md` / `43709bf` |
| 2026-04-04 | CPU Fase 1 — SIMD Hankel | Pulada | `relatorio_fase0_fase1_fortran.md` S3 / `43709bf` |
| 2026-04-04 | CPU Fase 2 — Hybrid Scheduler + Debitos 1/2/3 | Concluida | `relatorio_fase2_debitos_fortran.md` / `6ac51ca` |
| 2026-04-05 | CPU Fase 3 — Workspace Pre-allocation | Concluida | `relatorio_fase3_fortran.md` / `c213b66` |
| 2026-04-05 | PR1 Hygiene — Debitos D4/D5/D6 | Concluida | `relatorio_fase3_fortran.md` S5 / `db997d2` |
| 2026-04-05 | CPU Fase 4 — Cache `commonarraysMD` | Concluida | `relatorio_fase4_fortran.md` / `44acf2e` |
| 2026-04-05 | Validação Final — Testes bit-exatos Fases 0-4 | Concluida | `relatorio_validacao_final_fortran.md` / `195b69f` |
| 2026-04-05 | PR2 Debitos B1/B3/B5/B7 + Fase 5 | Concluida | `relatorio_fase5_debitos_fortran.md` / `fc208c1` |
| 2026-04-05 | CPU Fase 3b — Workspace estendido (12 campos) | Concluida | `0aa0aa9` |
| 2026-04-05 | CPU Fase 5b — Paralelismo adaptativo `if(ntheta>1)` | Concluida | `0aa0aa9` |
| 2026-04-05 | CPU Fase 2b — `schedule(guided, 16)` | Concluida | `8de722e` |
| 2026-04-05 | CPU Fase 6 — Cache `commonfactorsMD` por `camadT` | Erro conceitual | `analise_paralelismo_cpu_fortran.md` S7.7 |
| 2026-04-05 | CPU Fase 6b — Fatorizacao invariantes `h0` | Descartada (NaN) | `analise_paralelismo_cpu_fortran.md` S7.9 |

**Baseline publicado (Fase 0, 2026-04-04 — i9-9980HK, 8 cores, AVX-2, gfortran 15.2.0, OMP=8, CPU fria):**
- Wall-time: **0,1047 +/- 0,015 s/modelo**
- Throughput: **~34.400 modelos/hora**
- MD5 de referência: `c64745ed5d69d5f654b0bac7dde23a95`

**Estado final (Fase 2b, 2026-04-05):**
- Wall-time: **0,061 s/modelo**
- Throughput: **58.856 modelos/hora**
- Speedup: **5,62x** vs baseline
- Fidelidade: bit-exato @ `-O0`, `max|Delta| ~ 4e-13` @ `-O3`

---


---

## 13. Análise de Viabilidade CUDA (GPU)

### 13.1 Identificação de Kernels Paralelizáveis

| Kernel | Dimensão | Paralelismo | Adequação GPU |
|:-------|:---------|:------------|:-------------|
| `commonarraysMD` | npt × n | Independente por ponto do filtro | **Alta** — 201 threads independentes |
| `commonfactorsMD` | npt | Independente por ponto do filtro | **Alta** — 201 threads |
| Convolução Hankel (sum) | npt | Redução paralela | **Alta** — redução clássica |
| Loop medições | nmed | Independente por medição | **Alta** — 600+ threads |
| Loop frequências | nf | Independente por frequência | **Média** — apenas 2 frequências |
| Recursão reflexão | n (sequêncial) | **Dependência de dados** | **Baixa** — n camadas sequênciais |

### 13.2 Estratégia de Implementação CUDA

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

### 13.3 Desafios CUDA

1. **Recursão dos coeficientes de reflexão:** Os coeficientes são calculados de forma recursiva (camada n para 1, e 1 para n). Cada passo depende do anterior, impedindo paralelismo nesta dimensão. Porém, os 201 pontos do filtro são independentes, permitindo 201 threads executando a recursão simultaneamente.

2. **Alocação dinâmica:** O código Fortran aloca arrays de tamanho variável (`Tudw`, `Txdw`) dependendo da posição relativa T-R. Na GPU, estes arrays devem ser pré-alocados com tamanho máximo.

3. **Divergência de branch:** Os 6 casos geométricos em `hmd_TIV_optimized` causam divergência de warp em GPUs NVIDIA. Estratégia: agrupar medições por caso geométrico para minimizar divergência.

4. **Memória:** Para n=80 camadas, npt=201: arrays de tamanho (201, 80) x complex64 = 201 x 80 x 16 bytes = ~257 KB por medição. Caberiam ~800 medições simultâneas em 200 MB de memória global.

### 13.4 Análise Detalhada de Memória GPU

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
pois o overhead de transferência CPU->GPU é amortizado e a ocupação dos SMs é maximizada.

### 13.5 Estimativa de Speedup

| Componente | CPU (1 core) | GPU (estimado) | Speedup |
|:-----------|:-------------|:---------------|:--------|
| `commonarraysMD` | ~0.5 ms/call | ~0.01 ms/call | 50x |
| Convolução Hankel | ~0.2 ms/sum | ~0.002 ms/sum | 100x |
| Loop medições (600) | 600 x serial | 600 x paralelo | ~100x |
| Overhead transferência | 0 | ~1 ms/modelo | - |
| **Total por modelo** | ~400 ms | ~10 ms | **~40x** |

### 13.6 Frameworks CUDA Recomendados

| Framework | Linguagem | Vantagem | Desvantagem |
|:----------|:----------|:---------|:------------|
| **CUDA Fortran (PGI/NVHPC)** | Fortran | Reuso máximo do código | Compilador proprietário |
| **CUDA C/C++** | C/C++ | Ecossistema maduro, cuBLAS | Reescrita completa |
| **OpenACC** | Fortran | Mínimo de alteração no código | Desempenho menor que CUDA nativo |
| **hipSYCL** | C++ | Portabilidade AMD/NVIDIA | Complexidade adicional |

**Recomendação:** Para prototipagem rápida, **OpenACC** com `nvfortran` (NVIDIA HPC SDK). Para produção, **CUDA C** com wrappers Fortran.


### 13.7 Pipeline A — Roteiro de Otimização do Código Fortran

#### 13.7.1 Visão Geral do Pipeline A

O Pipeline A concentra-se em extrair o máximo desempenho do código Fortran existente sem alterar a física implementada. A estratégia é dividida em três fases progressivas: otimizações CPU com OpenMP (retorno imediato, risco mínimo), implementação GPU via OpenACC ou CUDA (aceleração de 10-50x, risco moderado) e válidação sistemática com benchmarking quantitativo.

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
| Erro relativo vs. código original | -- | < 1x10^-12 | < 1x10^-6 |
| Uso de memória por thread | ~150 MB | < 80 MB | ~500 MB (GPU) |
| Escalabilidade (efficiency Amdahl) | -- | > 70% | > 80% |

---

#### 13.7.2 Fase 1 — Otimizações CPU com OpenMP (Prioridade Inicial)

Esta fase ataca os gargalos identificados na análise OpenMP (Seção 11) sem introduzir dependências externas ou risco de divergência numérica. Todas as modificações são feitas no módulo `DManisoTIV` (`PerfilaAnisoOmp.f08`) e no módulo `magneticdipoles` (`magneticdipoles.f08`).

---

##### Passo 1.1 — Pré-alocação de Workspace por Thread (Eliminar `allocate` no Laço Paralelo)

**Problema identificado:** A sub-rotina `fieldsinfreqs` é chamada dentro do laço paralelo `!$omp parallel do` sobre medições (`j = 1, nmed(k)`). Internamente, `hmd_TIV_optimized` e `vmd_optimized` alocam dinamicamente arrays `Tudw`, `Txdw`, `Tuup`, `Txup`, `TEdwz`, `TEupz` em cada chamada. Com 600 medições x 2 frequências, isso resulta em até **2.400 chamadas a `allocate/deallocate`** por modelo no laço interno, gerando contenção de heap e fragmentação de memória.

**O que modificar:** Sub-rotinas `hmd_TIV_optimized` e `vmd_optimized` em `magneticdipoles.f08`; laço externo em `perfila1DanisoOMP` em `PerfilaAnisoOmp.f08`.

**Speedup esperado:** 1,4-1,8x (redução de 30-40% no tempo de execução medido por `gprof`).

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

**Speedup esperado:** 1,6-2,2x (a sub-rotina `commonarraysMD` representa ~45% do tempo total, segundo análise de profiling com `gprof -l`).

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

##### Passo 1.3 — Collapse de Laços com `collapse(3)` para Ângulos x Frequências x Medições

**Problema identificado:** O laço paralelo externo (`k = 1, ntheta`) tipicamente tem `ntheta = 1` em simulações de ângulo único (caso 0 graus), o que impede qualquer benefício do paralelismo de nível 1. O laço interno (`j = 1, nmed(k)`) tem `nmed = 600`, que é paralelizado com `num_threads_j = maxthreads - ntheta` threads. A combinação aninhada gera overhead de fork/join para apenas uma iteração do laço externo.

**O que modificar:** Estrutura de laços em `perfila1DanisoOMP`; requer linearização do índice de medições.

**Speedup esperado:** 1,2-1,5x para `ntheta = 1`; sem ganho para `ntheta > 4` (já paralelizado de forma eficiente).

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

Com `npt = 201` pontos e aritmética complexa de precisão dupla, cada `sum()` é uma redução sobre vetores complexos de 201 elementos. O compilador `gfortran` com `-O3 -march=native` pode vetorizar automáticamente, mas a presença de `sum()` intrínseco sobre arrays complexos frequentemente impede a geração de instruções AVX-512 ou AVX2.

**O que modificar:** Sub-rotinas `hmd_TIV_optimized` e `vmd_optimized` em `magneticdipoles.f08`; adicionar diretivas `!DIR$ VECTOR` ou usar laço explícito com `!$omp simd`.

**Speedup esperado:** 1,3-2,0x nas reduções de Hankel (que representam ~25% do tempo total).

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
gfortran -O3 -march=native -fopenmp -ffast-math \\
         -fopt-info-vec-optimized \\    # relatorio de vetorizacao
         -funroll-loops \\              # desenrolar laços de 201 pts
         -fprefetch-loop-arrays \\      # prefetch de arrays grandes
         -o PerfilaAnisoOmp *.f08
```

---

##### Passo 1.5 — Escalonador Híbrido (Static para Uniforme, Dynamic para Variável)

**Problema identificado:** O código atual usa `schedule(dynamic)` para ambos os laços — externo (ângulos `k`) e interno (medições `j`). Para `ntheta = 1` e `nmed` uniforme, o escalonador `dynamic` introduz overhead de sincronização desnecessário. Para casos multi-ângulo com `nmed(k)` variável (o número de medições pode diferir entre ângulos), `static` causaria desbalanceamento.

**O que modificar:** Diretivas `!$omp parallel do schedule(...)` em `perfila1DanisoOMP`.

**Speedup esperado:** 1,05-1,15x (ganho marginal, mas cumulativo com outros passos).

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

**Speedup esperado:** 1,2-1,5x em modelos com camadas espessas (> 5 m); menor impacto em modelos de camadas finas.

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

#### 13.7.3 Fase 2 — Implementação GPU (CUDA/OpenACC)

A Fase 2 aproveita o paralelismo massivo de GPUs modernas (NVIDIA A100: 6.912 CUDA cores; RTX 4090: 16.384 CUDA cores) para acelerar o simulador em 20-50x sobre o código CPU original. A estratégia preferida é **OpenACC como protótipo rápido** (mínimas mudanças no código), seguida de **CUDA Fortran via NVHPC** para maximizar ocupância e coalescência de memória.

---

##### Passo 2.1 — Protótipo OpenACC (Mudanças Mínimas no Código)

OpenACC permite acelerar o código existente com diretivas de compilador sem reescrever o algoritmo. A curva de aprendizado é baixa e a portabilidade é preservada (o mesmo código compila para CPU sem OpenACC).

**Requisito:** Compilador NVHPC (NVIDIA HPC SDK) ou GCC 10+ com suporte a OpenACC.

```bash
# Instalação NVHPC (Ubuntu/Colab):
apt-get install -y nvhpc-23-11

# Compilação com OpenACC:
nvfortran -O3 -acc=gpu -gpu=cc80,managed -Minfo=accel \\
          parameters.f08 utils.f08 filtersv2.f08 \\
          magneticdipoles.f08 PerfilaAnisoOmp.f08 RunAnisoOmp.f08 \\
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

**Limitação do protótipo OpenACC:** A estrutura de chamadas entre sub-rotinas (`fieldsinfreqs -> commonarraysMD -> commonfactorsMD -> hmd/vmd`) exige que todas as sub-rotinas chamadas dentro de regiões `!$acc parallel` sejam marcadas com `!$acc routine`. Isso pode requerer modificações nas assinaturas.

---

##### Passo 2.2 — Kernels CUDA Fortran via NVHPC

Para maximizar o desempenho, o laço mais interno (201 pontos x `n` camadas de `commonarraysMD`) é convertido em kernels CUDA explícitos.

**Mapeamento CPU -> GPU:**

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

##### Passo 2.3 — Gerenciamento de Memória (Host <-> Device)

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
| `RTEdw/up, RTMdw/up` | (201, 10, 2) x 4 | `complex(dp)` | ~205 KB |
| **Total por modelo** | -- | -- | **~371 KB** |

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
ncu --set full --kernel-name \"hankel_reduction_kernel\" \\
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

#### 13.7.4 Fase 3 — Validação e Benchmarking

##### Protocolo de Validação

A válidade física dos resultados otimizados é verificada contra o código Fortran original usando os dados de referência em `validacao.dat` e `infovalidacao.out`.

```
┌──────────────────────────────────────────────────────────────────────┐
│  PROTOCOLO DE VALIDAÇÃO — FORTRAN ORIGINAL vs. OTIMIZADO             │
│                                                                      │
│  1. Modelo de referência: arquivo validacao.dat existente            │
│     (n=5 camadas, rho_h=[1,10,1,10,1] Ohm.m, rho_v=[2,20,2,20,2])  │
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
│     com rho_h em [0.1, 10.000] Ohm.m, n em [2, 20] camadas          │
└──────────────────────────────────────────────────────────────────────┘
```

**Script de válidação (Python, usando a saída binária do simulador):**

```python
import numpy as np
import struct

def read_fortran_binary(filename, n_records, n_cols=22):
    \"\"\"Lê arquivo binário unformatted stream do simulador Fortran.\"\"\"
    data = []
    with open(filename, 'rb') as f:
        for _ in range(n_records):
            record = struct.unpack(f'i{n_cols-1}d', f.read(4 + (n_cols-1)*8))
            data.append(record)
    return np.array(data)

# Comparar original vs. otimizado
data_orig = read_fortran_binary('validacao_original.dat', n_records=600)
data_opt  = read_fortran_binary('validacao_otimizado.dat', n_records=600)

# Erro relativo por componente (colunas 6-22 são Re/Im dos 9 componentes)
for col_idx, comp_name in enumerate(['Re(Hxx)', 'Im(Hxx)', 'Re(Hxy)', 'Im(Hxy)',
                                      'Re(Hxz)', 'Im(Hxz)', 'Re(Hyx)', 'Im(Hyx)',
                                      'Re(Hyy)', 'Im(Hyy)', 'Re(Hyz)', 'Im(Hyz)',
                                      'Re(Hzx)', 'Im(Hzx)', 'Re(Hzy)', 'Im(Hzy)',
                                      'Re(Hzz)', 'Im(Hzz)'], start=4):
    ref  = data_orig[:, col_idx]
    opt  = data_opt [:, col_idx]
    mask = np.abs(ref) > 1e-30
    err  = np.max(np.abs(opt[mask] - ref[mask]) / np.abs(ref[mask]))
    status = \"PASS\" if err < 1e-10 else \"FAIL\"
    print(f\"  {comp_name:12s}: max_rel_err = {err:.2e}  [{status}]\")
```

##### Metodologia de Benchmark

```bash
#!/bin/bash
# benchmark_pipeline_a.sh — medir speedup em cada fase

N_MODELS=1000
N_THREADS_LIST=\"1 2 4 8 16\"
BASELINE_TIME=0

echo \"=== Benchmark Pipeline A ===\"
echo \"Modelos: ${N_MODELS}, CPU: $(nproc) cores\"
echo \"\"

for VERSION in \"original\" \"fase1_workspace\" \"fase1_cache\" \"fase1_simd\" \"fase1_all\" \"fase2_acc\" \"fase2_cuda\"; do
  for NTHREADS in $N_THREADS_LIST; do
    export OMP_NUM_THREADS=$NTHREADS
    TIME=$( { time ./PerfilaAnisoOmp_${VERSION} config_benchmark.namelist; } 2>&1 | grep real | awk '{print $2}' )
    echo \"  ${VERSION} | threads=${NTHREADS} | time=${TIME}\"
  done
done
```

##### Tabela de Desempenho Esperado (Fase 1 + Fase 2)

| Versão | Threads/Dispositivo | Tempo/modelo | Speedup vs. original | Throughput (mod/h) |
|:-------|:-------------------:|:------------:|:--------------------:|:------------------:|
| Original (baseline) | 16 CPU | 2,40 s | 1,0x | 24.000 |
| Fase 1 — workspace | 16 CPU | 1,50 s | 1,6x | 38.400 |
| Fase 1 — +cache | 16 CPU | 0,90 s | 2,7x | 64.000 |
| Fase 1 — +SIMD | 16 CPU | 0,70 s | 3,4x | 82.300 |
| Fase 1 — completa | 16 CPU | 0,50 s | 4,8x | 115.000 |
| Fase 2 — OpenACC | A100 GPU | 0,12 s | 20x | 480.000 |
| Fase 2 — CUDA batch | A100 GPU | 0,048 s | 50x | 1.200.000 |

---

#### 13.7.5 Cronograma e Métricas de Sucesso

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
| K1 | Speedup Fase 1 (16 threads, 1000 modelos) | >= 4x | Semana 4 |
| K2 | Erro de válidação CPU (max relativo) | < 1x10^-10 | Semana 4 |
| K3 | Speedup Fase 2 OpenACC (A100, batch=1000) | >= 15x | Semana 6 |
| K4 | Speedup Fase 2 CUDA (A100, batch=1000) | >= 40x | Semana 10 |
| K5 | Erro de válidação GPU (max relativo) | < 1x10^-6 | Semana 12 |
| K6 | Escalabilidade CPU (efficiency Amdahl) | >= 70% | Semana 4 |
| K7 | Ocupância GPU (Nsight Compute) | >= 60% | Semana 10 |
| K8 | Reprodutibilidade (mesmo resultado em 10 execuções) | 100% | Semana 12 |

---

### 13.8 Pipeline B — Roteiro de Novos Recursos para o Simulador Fortran

#### 13.8.1 Fase 1 — Melhorias Imediatas (1-2 semanas)

Esta fase adiciona recursos de alta utilidade prática com impacto mínimo na estrutura do código existente. Todos os itens podem ser implementados sem alterar as sub-rotinas físicas principais (`commonarraysMD`, `commonfactorsMD`, `hmd_TIV_optimized`, `vmd_optimized`).

---

##### B1.1 — Seleção Adaptativa de Filtro de Hankel

**Motivação:** O código atual usa exclusivamente o filtro de Werthmuller de 201 pontos (`J0J1Wer` em `filtersv2.f08`). Para geometrias de longo espaçamento (dTR > 5 m) ou frequências muito baixas (< 1 kHz), os 201 pontos são excessivos — o filtro Kong de 61 pontos oferece precisão equivalente (erro < 0,1%) com apenas 30% do custo computacional. Para frequências muito altas (> 100 kHz) com resistividade baixa (< 0,1 Ohm.m), o filtro de Anderson de 801 pontos pode ser necessário para precisão numérica adequada.

**O que implementar:** Seleção automática em `perfila1DanisoOMP` baseada em critério de qualidade (produto `r x k_max`, onde `k_max = kr_max / r` é o wavenumber máximo do filtro).

**Impacto:** Redução de 30-70% no tempo de cálculo para configurações de baixa frequência / longo espaçamento.

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
│                  │  201pt (*)    │                                      │
│  Alta frequência │  Anderson     │  f > 100 kHz, rho < 0.1 Ω·m         │
│                  │  801pt        │  (a implementar)                     │
└──────────────────┴───────────────┴──────────────────────────────────────┘
(*) = filtro atual do código
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
```

---

##### B1.3 — Validação de Entradas e Tratamento de Erros

**Motivação:** Atualmente, entradas inválidas (frequências fora do range, espaçamentos negativos, resistividades zero) podem causar comportamento indefinido, `NaN`/`Inf` silenciosos ou `STOP` sem mensagem diagnóstica. A adição de uma sub-rotina de válidação melhora a usabilidade e facilita depuração.

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

  ! Validar frequências (range LWD típico: 100 Hz - 2 MHz)
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

#### 13.8.2 Fase 2 — Extensões de Médio Prazo (2-4 semanas)

##### B2.1 — Suporte a Múltiplos Receptores (Vários Valores de `dTR`)

**Motivação:** Ferramentas LWD comerciais modernas (Baker Hughes AziTrak, Schlumberger arcVISION) possuem 2-5 arranjos TR com espaçamentos diferentes (por exemplo, dTR = 0.5, 1.0, 2.0, 4.0 m). Cada espaçamento fornece investigação em profundidade diferente, aumentando o conteúdo de informação para inversão.

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

##### B2.2 — Gradientes por Diferenças Finitas para PINNs (dH/drho)

**Motivação:** O treinamento de PINNs (Physics-Informed Neural Networks) com o simulador Fortran como `forward model` requer gradientes dH/drho_h e dH/drho_v para cada camada. A abordagem mais simples (e suficiente para treinamento offline de dados de inversão) são diferenças finitas centradas com passo delta_rho ótimo.

**O que implementar:** Sub-rotina `compute_gradients_fd` que chama `perfila1DanisoOMP` com perturbações de resistividade.

```fortran
subroutine compute_gradients_fd(nf, freq, ntheta, theta, h1, tj, dTR, p_med, &
                                 n, resist, esp, mypath, filename, &
                                 dHdRho_h, dHdRho_v)
  ! Calcula os gradientes dH_{ij}/drho_{h,k} e dH_{ij}/drho_{v,k}
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

  ! Passo ótimo para diferenças finitas: delta_rho = sqrt(eps_mach) * rho
  real(dp), parameter :: FD_STEP_REL = 1.d-4  ! 0.01% de perturbação relativa

  do k_lay = 1, n
    resist_plus  = resist
    resist_minus = resist

    ! Perturbação em rho_h (componente horizontal)
    rho_ref = resist(k_lay, 1)
    delta_rho = max(FD_STEP_REL * rho_ref, 1.d-6)  ! passo mínimo de 1e-6 Ohm.m
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

**Custo computacional:** Para `n = 10` camadas e 1 medição de referência: 20 chamadas adicionais ao simulador forward, ou seja, overhead de 20x sobre uma simulação única. Para geração offline de conjunto de treinamento (1.000 modelos x 20 perturbações = 20.000 simulações extras), o custo total com Pipeline A CPU = 20.000 x 0,5 s / 600 = 17 segundos por modelo, ou ~4,7 horas para 1.000 modelos com gradientes completos.

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

##### B2.4 — Apróximação de Camadas Inclinadas (Dip != 0 graus)

**Motivação:** O simulador atual modela camadas estritamente horizontais (TIV). Para geosteering em poços direcionais perfurando formações com dip estratigráfico real (geralmente 2-10 graus), a apróximação de \"camadas inclinadas\" melhora a fidelidade do modelo sem recorrer a modelagem 2D completa. A apróximação consiste em decompor a geometria em componentes horizontal e vertical considerando o ângulo de dip, e ajustar as profundidades de interface em função da posição horizontal do arranjo TR.

```fortran
! Aproximação de camadas inclinadas: ajustar prof() em função da posição x
subroutine apply_dip_correction(n, prof_ref, esp_ref, x_position, dip_angle_deg, &
                                 prof_corrected, esp_corrected)
  ! Desloca as interfaces de camada lateralmente com base no dip da formação.
  ! Válido para dip < 30 graus (aproximação de primeira ordem).
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
      ' graus > 15 graus. Aproximação pode ter erro > 5% em Hzz.'
  end if
end subroutine apply_dip_correction
```

Esta sub-rotina é chamada dentro do laço `j` de medições em `perfila1DanisoOMP`, passando `x_position = (j-1) * px` como posição horizontal relativa.

---

#### 13.8.3 Fase 3 — Extensões de Longo Prazo (1-3 meses)

##### B3.1 — Equações Adjuntas para Gradientes Exatos

**Motivação:** Diferenças finitas (B2.2) têm custo computacional de 2n chamadas forward por ponto de medição, e sofrem de cancelamento numérico para passos muito pequenos. As equações adjuntas permitem calcular **todos os gradientes** dH/drho_k (para todas as `n` camadas simultaneamente) com o custo de apenas **2 soluções de problema adjunto** (forward + adjoint), independentemente de `n`.

**Princípio:** Para a resposta escalar `J = <H, W>` (H = campo, W = vetor de pesos), o gradiente em relação a `m` (vetor de parâmetros do modelo) é:

```
dJ/dm = Re( lambda^H dA/dm u )
```

onde `u` é a solução forward (`A u = s`), `lambda` é a solução adjunta (`A^H lambda = W`), e `A` é a matriz do sistema linear EM (coeficientes de reflexão acoplados).

**Roadmap de implementação:**

```
┌──────────────────────────────────────────────────────────────────────┐
│  IMPLEMENTAÇÃO EQUAÇÕES ADJUNTAS (3 etapas)                          │
│                                                                      │
│  Etapa 3.1.a (semana 1–2): Derivar equações adjuntas analíticas      │
│    para os coeficientes de reflexão RTEdw, RTEup, RTMdw, RTMup       │
│    em função de rho_h e rho_v (ver Habashy & Groom, 2004)           │
│                                                                      │
│  Etapa 3.1.b (semana 3–5): Implementar sub-rotina adjoint_solve()   │
│    que resolve o problema transposto-conjugado usando os mesmos      │
│    arrays de propagação (u, s, uh, sh) calculados no forward         │
│                                                                      │
│  Etapa 3.1.c (semana 6–8): Validar gradientes adjuntos contra        │
│    diferenças finitas centradas para modelos simples (n <= 5)        │
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

##### B3.2 — Apróximação de Born para Perturbações 2D

**Motivação:** A apróximação de Born de primeira ordem permite modelar perturbações de resistividade em 2D (variação lateral) como integral de volume sobre o campo de background 1D. Esta extensão é fundamental para geosteering look-ahead — onde heterogeneidades laterais (falhas, bordas de camada) precisam ser detectadas antes de serem alcançadas pela broca.

**Formulação:**

```
H_Born(r) = H_background(r) + integral_integral G(r, r') . delta_sigma(r') . E_background(r') d^2r'
```

onde `G(r, r')` é o tensor Green EM 1D (calculado pelo simulador atual) e `delta_sigma(r')` é a perturbação de condutividade 2D.

---

##### B3.3 — Anisotropia Biaxial (sigma_x != sigma_y != sigma_z)

**Motivação:** O código atual modela anisotropia TIV (transversalmente isotrópica na vertical), com sigma_h = sigma_x = sigma_y e sigma_v = sigma_z. Formações geológicas com acamamento inclinado ou estruturas sedimentares cruzadas podem exibir anisotropia ortorrômbica completa (biaxial): sigma_x != sigma_y != sigma_z. Isso altera os coeficientes de reflexão TE/TM e a estrutura da transformada de Hankel.

**Impacto na física:** Os modos TE e TM acoplam-se quando sigma_x != sigma_y (o sistema deixa de ser diagonal no espaço de Hankel), exigindo solução de sistema 4x4 em vez do sistema 2x2 atual.

**Estimativa de esforço:** 3-6 semanas para implementação completa, incluindo derivação das equações de propagação e válidação analítica.

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
mpif90 -O3 -fopenmp -ffast-math \\
       parameters.f08 utils.f08 filtersv2.f08 \\
       magneticdipoles.f08 PerfilaAnisoOmp.f08 RunAnisoOmpMPI.f08 \\
       -o PerfilaAnisoOmp_mpi

# Execução em cluster (SLURM):
# srun --nodes=10 --ntasks-per-node=1 --cpus-per-task=16 \\
#      ./PerfilaAnisoOmp_mpi
```

---

#### 13.8.4 Cronograma Integrado Pipelines A+B

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
| A1.1 Workspace pré-alocado | A1 | Alta | -- | Reduz tempo de geração de dados |
| A1.2 Cache commonarraysMD | A1 | Alta | -- | Idem |
| A1.4 SIMD Hankel | A1 | Alta | -- | Idem |
| B1.3 Validação entradas | B1 | Alta | -- | Evita dados corrompidos no dataset |
| B1.4 Logging progresso | B1 | Média | -- | Monitoramento de simulações longas |
| B1.1 Filtro adaptativo | B1 | Alta | -- | Reduz custo em freq. baixas |
| B2.1 Multi-receptor | B2 | Alta | B1.3 | Mais features para inversão |
| B2.2 Gradientes FD | B2 | Alta | B1.3, B1.1 | Treinamento de PINNs |
| A2.1 OpenACC | A2 | Média | A1 completo | 15-20x speedup no Colab GPU |
| A2.2 CUDA Fortran | A2 | Média | A2.1 | 40-50x speedup no Colab GPU |
| B2.4 Dip appróximation | B2 | Média | B2.1 | Geosteering em poços direcionais |
| B3.1 Adjuntas | B3 | Alta | B2.2 | Treinamento PINNs 20x mais rápido |
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
│    + Gradientes FD: dataset com dH/drho para PINNs                │
│                                                                    │
│  Após Pipeline A Fase 2 + B3 MPI (20 semanas):                    │
│    Cluster 10 nós × A100: ~1.000 modelos/min                       │
│    Dataset de 1.000.000 modelos: ~16 horas                         │
└────────────────────────────────────────────────────────────────────┘
```

---

**Documentos de referência para as implementações acima:**

- Werthmuller, D. (2017). \"An open-source full 3D electromagnetic modeler for 1D VTI media in Python: empymod.\" *Geophysics*, 82(6), WB9-WB19. -- filtro de 201 pontos e critérios de precisão.
- Kong, F.N. (2007). \"Hankel transform filters for dipole antenna radiation in a conductive medium.\" *Geophysical Prospecting*, 55(1), 83-89. -- filtro de 61 pontos.
- Anderson, W.L. (1979). \"Numerical integration of related Hankel transforms of orders 0 and 1 by adaptive digital filtering.\" *Geophysics*, 44(7), 1287-1305. -- filtro de 801 pontos.
- Habashy, T.M. & Groom, R. (2004). \"Adjoint method for computing electromagnetic sensitivities in layered anisotropic media.\" *Geophysical Journal International*, 159(2), 698-712. -- base teórica para B3.1.
- OpenMP Architecture Review Board (2023). \"OpenMP 5.2 Specification.\" -- `collapse`, `simd`, `schedule(guided)`.
- NVIDIA Corporation (2023). \"CUDA Fortran Programming Guide.\" -- kernels CUDA para Fortran via NVHPC.

---

## 14. Reimplementação em Python Otimizado

### 14.1 Mapeamento Estrutural Fortran -> Python

O código Fortran otimizado possui propriedades que facilitam a conversão:

**Propriedades favoráveis à portabilidade:**

| Propriedade do Fortran | Equivalente Python/Numba | Ref. |
|:------------------------|:-------------------------|:----:|
| Loops `do j = 1, nmed` sem dependências entre iterações | `@njit` com `prange(nmed)` | [11] |
| `type :: thread_workspace` com arrays pré-alocados | `numpy.ndarray` pré-alocados ou `numba.typed.List` | [12] |
| Cache de `commonarraysMD` por `(r, freq)` | Mesmo padrão: pré-computar em array global antes do `prange` | -- |
| Aritmética sobre `complex(dp)` (complex128) | `numpy.complex128` ou `jax.numpy.complex128` | [13] |
| `exp()`, `sqrt()`, `tanh()` sobre arrays | Compilados por LLVM (Numba) ou XLA (JAX) com qualidade similar a GCC/GFortran | [14, 15] |
| `schedule(guided, 16)` no OpenMP | `numba.prange` com particionamento automático; JAX `vmap` para batch | [11, 16] |

**Mapeamento detalhado das subrotinas:**

```
Fortran                              Python/Numba/JAX
──────────────────────────────────────────────────────────────
perfila1DanisoOMP                 →  @jax.jit ou @numba.njit (orquestrador)
  ├── omp_set_max_active_levels   →  (não necessário — JAX gerencia paralelismo)
  ├── allocate(ws_pool)           →  np.zeros((maxthreads, npt, n), dtype=complex128)
  ├── commonarraysMD (cache)      →  jax.jit(commonarraysMD)(r_k, freq, h, eta)
  ├── !$omp parallel do           →  prange(nmed) ou jax.vmap
  │   └── fieldsinfreqs_cached_ws →  @jit function (hot path)
  │       ├── commonfactorsMD     →  @jit function (14 exp())
  │       ├── hmd_TIV_optimized   →  @jit function (kernel HMD)
  │       └── vmd_optimized       →  @jit function (kernel VMD)
  └── writes_files                →  np.save() ou struct.pack()

thread_workspace (12 campos)      →  Tuple de arrays ou NamedTuple
  ├── Tudw, Txdw, ..., TEupz     →  arrays (npt, n) complex128
  └── Mxdw, Mxup, ..., FEupz     →  arrays (npt,) complex128
```

**Tabela de mapeamento Fortran -> Python por sub-rotina:**

| Sub-rotina Fortran | Função Python | Módulo | Operação Principal |
|:-------------------|:--------------|:-------|:-------------------|
| `commonarraysMD` | `compute_propagation_constants()` | `propagation.py` | Constantes u, s; coeficientes TE/TM por recursão |
| `hmd_TIV` | `compute_hmd_fields()` | `dipoles.py` | Montagem dos campos HMD (6 componentes do tensor H) |
| `vmd_TIV` | `compute_vmd_fields()` | `dipoles.py` | Montagem dos campos VMD (3 componentes diagonais) |
| `convol1D` | `hankel_transform()` | `hankel.py` | Convolução discreta com filtro de 201 pontos (J0/J1) |
| `rotate_tensor` | `rotate_tensor_tiv()` | `rotation.py` | Rotação do tensor H(3x3) para coordenadas do poço |
| `geometry_setup` | `build_tr_geometry()` | `geometry.py` | Posições T-R para 600 medições |
| *(carregamento de filtros)* | `load_filter_coefficients()` | `filters.py` | Tabela de 201 coeficientes (Werthmuller 2006) |

**Arquitetura proposta para o módulo Python:**

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

### 14.2 Numba CPU (`@njit` + `prange`) — Análise Aprofundada

Numba [11] é um compilador JIT baseado em LLVM que traduz funções Python anotadas para código de máquina nativo. Para o simulador EM:

**Vantagens:**
- Compilação transparente de loops numéricos com performance ~95% do GFortran `-O3`
- Suporte nativo a `complex128`, `numpy.exp`, `numpy.sqrt`
- `prange` para paralelismo multi-thread (equivalente a `!$omp parallel do`)
- Zero dependência de compilador Fortran para distribuição
- Integração direta com o pipeline `geosteering_ai/` existente

**Limitações:**
- Subconjunto restrito de Python (sem dicionários, sem classes com herança no modo `@njit`)
- Tempo de compilação JIT na primeira execução (~5-15 s para o simulador completo)
- Sem auto-diferenciação (requer implementação manual do backward pass)
- Debug complexo: erros de tipo em tempo de compilação JIT são crípticos

**Fidelidade numérica:** O LLVM compila `exp()`, `sqrt()`, `tanh()` sobre `complex128` com a mesma qualidade que o GCC/GFortran — as funções transcendentais são mapeadas para as mesmas implementações de libm. A diferença esperada entre Numba e Fortran para operações numéricas puras é <5-10%, vinda do overhead de gerenciamento de arrays NumPy e do JIT warmup.

**Exemplo de conversão completa (commonfactorsMD):**

```python
@numba.njit(cache=True)
def commonfactorsMD(n, npt, h0, h, prof, camadT, u, s, uh, sh,
                     RTEdw, RTEup, RTMdw, RTMup):
    \"\"\"Fatores de onda refletida da camada do transmissor.

    Equivalente Fortran: utils.f08:243-297 (commonfactorsMD)
    Fatoração: modos TM (s, RTMdw/RTMup) e TE (u, RTEdw/RTEup)
    separados, cada um com denominador de ressonância de camada.

    Args:
        n, npt: número de camadas e pontos do filtro Hankel
        h0: profundidade do transmissor (variável em j)
        h, prof: espessuras e profundidades das interfaces
        camadT: índice da camada do transmissor
        u, s, uh, sh: constantes de propagação (npt x n)
        RTEdw..RTMup: coeficientes de reflexão (npt x n)

    Returns:
        Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz: fatores de onda (npt,)
    \"\"\"
    cT = camadT
    # Modo TM
    den_TM = 1.0 - RTMdw[:, cT] * RTMup[:, cT] * np.exp(-2.0 * sh[:, cT])
    Mxdw = (np.exp(-s[:, cT] * (prof[cT] - h0)) +
            RTMup[:, cT] * np.exp(s[:, cT] * (prof[cT-1] - h0 - h[cT]))) / den_TM
    Mxup = (np.exp(s[:, cT] * (prof[cT-1] - h0)) +
            RTMdw[:, cT] * np.exp(-s[:, cT] * (prof[cT] - h0 + h[cT]))) / den_TM
    # Modo TE
    den_TE = 1.0 - RTEdw[:, cT] * RTEup[:, cT] * np.exp(-2.0 * uh[:, cT])
    Eudw = (np.exp(-u[:, cT] * (prof[cT] - h0)) -
            RTEup[:, cT] * np.exp(u[:, cT] * (prof[cT-1] - h0 - h[cT]))) / den_TE
    Euup = (np.exp(u[:, cT] * (prof[cT-1] - h0)) -
            RTEdw[:, cT] * np.exp(-u[:, cT] * (prof[cT] - h0 + h[cT]))) / den_TE
    # Modo TE z-potencial (VMD)
    FEdwz = (np.exp(-u[:, cT] * (prof[cT] - h0)) +
             RTEup[:, cT] * np.exp(u[:, cT] * (prof[cT-1] - h[cT] - h0))) / den_TE
    FEupz = (np.exp(u[:, cT] * (prof[cT-1] - h0)) +
             RTEdw[:, cT] * np.exp(-u[:, cT] * (prof[cT] + h[cT] - h0))) / den_TE
    return Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz
```

**Projeção de throughput:** ~50.000-55.000 mod/h em 8 cores (i9-9980HK), equivalente a ~90-95% do Fortran otimizado.

---

### 14.3 f2py Wrapper — Produção Imediata

A solução de menor risco e implementação mais rápida é o wrapper `f2py` do Fortran otimizado existente. O código `tatu_f2py_wrapper.f08` já foi implementado e válidado.

**Características:**
- **Zero reescrita:** O wrapper chama diretamente o código Fortran compilado via interface Python gerada automáticamente por `f2py`
- **Performance idêntica ao Fortran:** Não há overhead de tradução — é o Fortran original executando
- **Integração com `geosteering_ai/`:** Import direto em Python: `from tatu_f2py_wrapper import perfila1DanisoOMP`
- **Já implementado:** O wrapper `tatu_f2py_wrapper.f08` e o runner paralelo `batch_runner.py` (com `ProcessPoolExecutor`) já existem no repositório

**Limitações:**
- Sem GPU (permanece CPU-only)
- Sem auto-diferenciação (impossível diferenciar através de código Fortran compilado)
- Dependência de compilador Fortran (gfortran) para compilação do wrapper

**Projeção de throughput:** ~58.000 mod/h (idêntico ao Fortran otimizado, pois **é** o Fortran).

---

### 14.4 JAX CPU (8 cores) — Análise Aprofundada

JAX [16] é a recomendação principal para este projeto, por três razões fundamentais: auto-diferenciação nativa, vetorização via `vmap`, e compilação XLA com kernel fusion.

**Compilação XLA para CPU:**

O compilador XLA [22] funde operações consecutivas em um único kernel, eliminando alocações intermediárias:

```
Sem XLA:
  exp(-s*a) → temp1 (201 × 30 × 16 bytes allocation)
  RTMup * exp(s*b) → temp2 (allocation)
  temp1 + temp2 → temp3 (allocation)
  temp3 / den → result (allocation)
  Total: 4 kernel launches + 4 allocations

Com XLA:
  (exp(-s*a) + RTMup * exp(s*b)) / den → result
  Total: 1 fused kernel + 1 allocation
```

Para `commonfactorsMD` com 14 `exp()` e 8 operações aritméticas, o XLA funde tudo em ~2-3 kernels vs ~22 sem fusão. Referência: Abadi et al. [22] reportam 2-5x speedup por kernel fusion em operações de álgebra linear.

**Threading via XLA (não OpenMP):** JAX CPU usa XLA's internal thread pool, não OpenMP. A paralelização é automática sobre operações vetorizáveis.

**`vmap` para batch de modelos:**

```python
# Simular 10.000 modelos simultaneamente
batch_simulate = jax.vmap(simulate, in_axes=(0, 0, None, 0, 0, None))
H_batch = batch_simulate(rho_h_batch, rho_v_batch, freq, h_batch, prof_batch, geom)
# H_batch.shape = (10000, nf, 9)
```

**Auto-diferenciação nativa (dH/drho):**

```python
# Jacobiano dH/drho — automático, custo ~3x forward
jacobian_fn = jax.jacobian(simulate, argnums=(0, 1))
J_h, J_v = jacobian_fn(rho_h, rho_v, freq, h, prof, geom)
# J_h.shape = (nf, 9, n_layers) — sensibilidade a cada camada
```

**Projeção de throughput:** ~45.000-55.000 mod/h em 8 cores CPU. A diferença em relação ao Numba é que o XLA pode perder eficiência na recursão sequêncial de impedâncias (requer `jax.lax.scan`), mas ganha em kernel fusion para as exponenciais complexas.

---

### 14.5 Numba CUDA — Análise Aprofundada

Numba CUDA [17] permite escrever kernels GPU diretamente em Python com controle fino de shared memory, thread blocks, e coalesced access.

**Estratégia de paralelização GPU (3 níveis):**

```
┌─────────────────────────────────────────────────────────────────┐
│  Nível 1 — Batch de modelos (grid dimension x)                 │
│    Cada bloco CUDA processa um modelo geológico completo        │
│                                                                 │
│  Nível 2 — Medidas j (grid dimension y)                        │
│    Cada warp processa uma medida (posição T-R na janela)        │
│                                                                 │
│  Nível 3 — Pontos de Hankel ipt (threads dentro do warp)       │
│    Cada thread processa um ponto do filtro (ipt in [1, 201])   │
│    → 201 threads ≈ 6,3 warps → boa ocupação de SM              │
│                                                                 │
│  Shared memory:                                                 │
│    u(:, camadT), s(:, camadT) — constantes de propagação       │
│    RTEdw(:, camadT), RTMup(:, camadT) — coeficientes           │
│    → ~6,4 KB por bloco (cabe nos 48–164 KB de shared)          │
│                                                                 │
│  Redução:                                                       │
│    Somas de Hankel (sum_{ipt} kernel * weight) via warp shuffle │
│    → 1 warp reduce por componente de H                          │
└─────────────────────────────────────────────────────────────────┘
```

**Workaround para complex128:** `complex128` não é tipo nativo em CUDA — requer emulação via `struct {double re, im}` ou uso de operações separadas real/imag. O overhead é ~10-20% comparado a complex64, mas a precisão dupla é mandatória para o simulador EM (ver Seção 14.9).

**Shared memory para coeficientes:** ~6,4 KB por bloco para os arrays de constantes de propagação da camada do transmissor, bem dentro do limite de 48-164 KB disponíveis nos SMs modernos.

**Warp shuffle para reduções de Hankel:** As somas sobre 201 pontos são computadas via tree reduction em shared memory com warp shuffle primitives, atingindo ~95% de eficiência de pico na redução.

**Projeção de throughput:**
- Tesla T4 (16 GB): ~200.000-500.000 mod/h
- A100 (80 GB): ~800.000-2.000.000 mod/h

---

### 14.6 JAX GPU — Análise Aprofundada

JAX GPU combina compilação XLA com execução em GPU NVIDIA, oferecendo fusão automática de kernels sem necessidade de escrever CUDA manualmente.

**Compilação XLA para GPU:** O XLA gera kernels CUDA otimizados automáticamente, incluindo:
- Fusão de operações consecutivas (14 `exp()` em `commonfactorsMD` -> 2-3 kernels)
- Eliminação de alocações intermediárias
- Scheduling automático de kernels no stream CUDA

**Vantagem sobre Numba CUDA:** Zero código CUDA manual. O mesmo código Python que roda em CPU roda em GPU sem modificação (apenas `jax.devices(\"gpu\")`).

**Limitação:** Menor controle sobre shared memory e coalescing comparado a Numba CUDA manual. Para o caso do simulador EM com branches condicionais (6 casos em `hmd_TIV`), o XLA pode gerar código menos eficiente que kernels CUDA hand-tuned.

**Projeção de throughput:**
- Tesla T4: ~150.000-400.000 mod/h
- A100: ~500.000-1.500.000 mod/h

---

### 14.7 JAX `vmap` Batch — Análise Aprofundada

A transformação `jax.vmap` permite vetorizar automáticamente o simulador sobre batches de modelos geológicos, explorando o paralelismo massivo da GPU sem loop explícito.

```python
# Simular batch de 1024 modelos simultaneamente em GPU
batch_simulate = jax.vmap(simulate, in_axes=(0, 0, None, 0, 0, None))
rho_h_batch = jnp.array(...)  # (1024, n_layers)
H_batch = batch_simulate(rho_h_batch, rho_v_batch, freq, h_batch, prof_batch, geom)
# H_batch.shape = (1024, nmed, nf, 9) — todos os modelos de uma vez
```

O `vmap` transforma automáticamente o simulador escalar em versão batched, explorando paralelismo sobre modelos sem reescrever o kernel. Em GPU A100 com batch=1024, a ocupação dos SMs é maximizada.

**Throughput massivo:**
- A100 (batch=1024): ~2.000.000-5.000.000 mod/h

**Casos de uso principais:**
- Geração massiva de datasets de treinamento (100k-1M modelos)
- PINNs com avaliação do forward model no training loop (batch de hipóteses)
- Inversão em tempo real com múltiplas hipóteses simultâneas

---

### 14.8 Estratégias Híbridas

Para maximizar desempenho e flexibilidade, diferentes componentes do pipeline podem usar diferentes backends simultaneamente.

#### 14.8.1 JAX GPU + Numba CPU

**Cenário:** JAX para o training loop (auto-diferenciação de PINNs), Numba CPU para geração de datasets offline.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ESTRATÉGIA: JAX GPU + NUMBA CPU                                      │
│                                                                      │
│  Geração de dados (offline):                                         │
│    Numba CPU (@njit + prange) → datasets .npz                       │
│    Throughput: ~55k mod/h (8 cores)                                  │
│    Vantagem: sem GPU necessária, determinístico                      │
│                                                                      │
│  Treinamento PINN (online):                                          │
│    JAX GPU (jax.grad + jax.vmap) → gradientes dH/drho               │
│    Throughput: ~500k mod/h (T4)                                      │
│    Vantagem: auto-diff nativa, integração com loss TF               │
└──────────────────────────────────────────────────────────────────────┘
```

#### 14.8.2 f2py + ProcessPoolExecutor

**Cenário:** Uso imediato em produção. Já implementado em `batch_runner.py`.

```python
from concurrent.futures import ProcessPoolExecutor
from tatu_f2py_wrapper import perfila1DanisoOMP

def simulate_model(model_params):
    return perfila1DanisoOMP(**model_params)

with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(simulate_model, models_list))
```

**Throughput:** ~58.000 mod/h (idêntico ao Fortran, pois é o Fortran).

**Vantagem:** Zero risco, zero reescrita, disponível agora.

#### 14.8.3 JAX + TensorFlow

**Cenário:** Integração nativa para PINNs. Ambos usam backend XLA, permitindo interoperabilidade sem cópia de dados.

```python
# JAX calcula gradientes do forward model
dH_drho = jax.grad(simulate_jax)(rho_h, rho_v, ...)

# TensorFlow computa a loss total (dados + física)
loss = tf_loss_data + lambda_phys * jax2tf(dH_drho)
```

**Vantagem:** Elimina ponte `tf.py_function` para operações JAX, reduzindo overhead de serialização.

#### 14.8.4 CuPy

**Cenário:** Prototipagem rápida em GPU. Drop-in replacement para NumPy com `complex128` suportado nativamente.

**Limitação:** Kernels customizados (`RawKernel`) necessários para lógica condicional de `hmd/vmd`. Overhead de lançamento de kernel (~10-50 microssegundos) domina para arrays pequenos (npt=201).

**Recomendação:** Útil para válidação rápida de conceitos GPU, não para produção.

#### 14.8.5 Taichi Lang

**Cenário:** DSL para computação paralela com backend CPU + GPU e auto-diferenciação parcial.

**Status:** Ecossistema pequeno, menos maduro que JAX/Numba. Auto-diferenciação limitada a tipos escalares.

**Recomendação:** Monitorar evolução, não adotar atualmente.

#### 14.8.6 PyTorch + custom CUDA

**NOTA: PyTorch é PROIBIDO neste projeto** conforme CLAUDE.md (framework exclusivo: TensorFlow 2.x / Keras). Esta opção é documentada apenas para completude da análise comparativa. Qualquer implementação DEVE usar TensorFlow/Keras + JAX.

#### 14.8.7 Tabela Comparativa de Estratégias Híbridas

| Estratégia | Throughput | Auto-diff | Complexidade | Disponibilidade | Recomendação |
|:-----------|:----------|:----------|:------------|:---------------|:-------------|
| **f2py + ProcessPool** | 58k mod/h | Não | Muito baixa | **Agora** | **Track 1 (produção)** |
| **Numba CPU** | 55k mod/h | Não | Baixa | 6-8 semanas | Substituição gradual |
| **JAX CPU** | 50k mod/h | **Sim** | Média | 8-12 semanas | PINNs CPU |
| **JAX GPU (T4)** | 400k mod/h | **Sim** | Média | 10-14 semanas | **Track 2 (principal)** |
| **Numba CUDA (T4)** | 500k mod/h | Não | Alta | 12-16 semanas | Track 3 (performance) |
| **JAX vmap (A100)** | 5M mod/h | **Sim** | Média | 10-14 semanas | Geração massiva |
| CuPy | 200k mod/h | Não | Média | 4-6 semanas | Prototipagem |
| Taichi | Desconhecido | Parcial | Alta | N/A | Monitorar |
| PyTorch | N/A | N/A | N/A | **PROIBIDO** | N/A |

---

### 14.9 Fidelidade Física como Critério Máximo

A fidelidade numérica é o critério de aceitação mais importante para qualquer reimplementação do simulador. Velocidade sem precisão é inútil — dados de treinamento com erros numéricos produzem redes neurais que aprendem artefatos, não física.

#### 14.9.1 Por que a Fidelidade é Prioridade Absoluta

O simulador calcula campos eletromagnéticos `H` em unidades de A/m, com magnitudes que variam de `~10^-3` (componente axial Hzz) a `~10^-8` (componentes off-diagonal em camadas espessas). Uma imprecisão de `10^-6` em valor relativo pode representar ruído artificial de `~10^-14` A/m nas componentes fracas — da mesma ordem que o ruído instrumental real de ferramentas LWD. Portanto, **a reimplementação Python não pode introduzir erros comparáveis ao ruído do instrumento físico**.

#### 14.9.2 Precisão Dupla Mandatória (complex128)

A aritmética em `complex128` (64 bits por parte real e imaginária) é mandatória por três razões:

1. **Exponenciais com argumentos grandes:** `exp(-s * h)` com `|s*h| > 50` resulta em underflow em `float32` mas permanece representável em `float64`.
2. **Cancelamento subtractivo:** Em `commonfactorsMD`, os termos `exp(-s*(prof - h0))` envolvem subtração de números de magnitude similar (`prof ~ h0`), amplificando erros relativos.
3. **Recursão de impedâncias:** A recursão bottom-up de `RTEdw` acumula erros de arredondamento proporcionalmente ao número de camadas `n`. Para `n=80` camadas, `float32` pode acumular erros de `~10^-3` (inaceitável), enquanto `float64` mantém `~10^-11`.

#### 14.9.3 Protocolo de Validação Componente-a-Componente

Cada componente do tensor H deve ser válidada individualmente:

```
Critério de aceitação por componente:
  max|H_python(i) - H_fortran(i)| / |H_fortran(i)| < 1×10⁻¹⁰

  Para todas as 9 componentes: Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz
  Para todas as 600 medições
  Para ambas as frequências (20 kHz e 40 kHz)
```

**Suite de 10 modelos canônicos (M01-M10):** De trivial (homogêneo isotrópico) a extremo (80 camadas, Sobol sampling). Os arquivos `.dat` de referência são gerados pelo Fortran com configuração determinística e versionados em `tests/simulation/reference_data/`.

#### 14.9.4 Testes de Regressão Automatizados

- **Bit-exato (MD5):** Para compilação `-O0` (sem otimização agressiva), a saída Python deve ser bit-por-bit idêntica à do Fortran. Verificação via hash MD5 do arquivo `.dat`.
- **Componentwise (para `-O3`):** Com otimizações agressivas (`-ffast-math`), a reordenação de operações FP produz diferenças de ~`10^-13`. O teste componentwise verifica `max|Delta| < 1x10^-10` para cada componente individualmente.

#### 14.9.5 Garantias de Estabilidade Numérica

Três padrões de instabilidade foram identificados no simulador e devem ser preservados na reimplementação:

1. **Overflow de exponencial:** `exp(s * h)` com `Re(s*h) > 709` causa `Inf` em `float64`. O código Fortran evita isso via `tanh` estabilizada: `tanh(x) = 1.0` para `Re(x) > 25`.

2. **Saturação de tanh:** Para `|x| > 25`, `tanh(x) = sign(x)` com erro < `10^-22`. A função `tghuh/tghsh` no Fortran usa este cutoff explícitamente.

3. **Divisão por denominador pequeno:** O denominador `den_TE = 1 - RTEdw * RTEup * exp(-2*uh)` pode se apróximar de zero em condições de ressonância de camada. O Fortran não trata explícitamente (confia na aritmética IEEE 754), mas a reimplementação Python deve monitorar e reportar estes casos.

#### 14.9.6 Lição da Fase 6b: Fatoração de Exponenciais Causa NaN

A tentativa de fatorar `exp(-s*(prof - h0))` em `exp(-s*prof) * exp(s*h0)` para semi-cache por `camadT` foi **descartada com instabilidade numérica fatal**:

```
CAUSA: exp(-s*prof) × exp(s*h0) com |s*prof| >> 1 e |s*h0| >> 1
       → overflow nos termos separados, mesmo quando o produto original
         exp(-s*(prof-h0)) é finito (pois prof ≈ h0, cancelamento interno).

RESULTADO: 21.600 NaN em 25.200 saídas.

LIÇÃO: Exponenciais com argumentos de sinais opostos que se cancelam
       NÃO podem ser separadas sem tratamento especial (log-sum-exp scaling).
       A versão original exp(-s*(a-b)) mantém o cancelamento ANTES do exp(),
       preservando magnitude finita.
```

**Regra para reimplementação Python:** Qualquer otimização que separe exponenciais acopladas DEVE ser válidada contra o modelo M10 (80 camadas, pior caso de profundidade) antes de ser aceita.

#### 14.9.7 Requisito de Validação

**Qualquer reimplementação Python DEVE passar a mesma suite de válidação que o Fortran otimizado** (Seção 13.7.4), incluindo:
- 10 modelos canônicos com critérios por categoria (Seção 14.14)
- Teste bit-exato em `-O0` (quando aplicável)
- Teste componentwise em `-O3` com `max|Delta| < 1x10^-10`
- Verificação de zero NaN/Inf em 10.000 modelos aleatórios

---

### 14.10 Auto-diferenciação (dH/drho) para PINNs

#### 14.10.1 Fundamentação Teórica

A sensibilidade do campo eletromagnético às propriedades das camadas (Jacobiano de Frechet [37]) é fundamental para:

1. **Inversão determinística (Gauss-Newton):** `Delta_rho = (J^T W_d J + lambda W_m)^{-1} J^T W_d (d_obs - d_pred)`
2. **PINNs com physical loss:** `L = L_data + lambda_phys * L_physics(dH_pred/drho, dH_sim/drho)`
3. **Análise de resolução:** Eigenvalues de `J^T J` revelam a resolução vertical da inversão.

#### 14.10.2 Três Métodos de Cálculo

**Método 1 — Diferenças finitas (Fortran e Python):**

```python
def jacobian_fd(simulate, rho, delta=1e-4):
    \"\"\"Jacobiano via diferenças finitas centradas.\"\"\"
    J = np.zeros((n_obs, n_params))
    for i in range(n_params):
        rho_plus = rho.copy(); rho_plus[i] += delta * rho[i]
        rho_minus = rho.copy(); rho_minus[i] -= delta * rho[i]
        J[:, i] = (simulate(rho_plus) - simulate(rho_minus)) / (2 * delta * rho[i])
    return J
```

Custo: `2 x n_params` execuções do simulador. Para n=15, 2 params/camada: 60 execuções -> ~3,7 s/modelo. **Inaceitável para treinamento.**

**Método 2 — Adjoint method (Fortran, implementação manual):**

Custo: ~3x forward pass (1 forward + 1 adjoint + 1 combinação). Para o simulador atual: ~0,18 s/modelo -> ~20.000 mod/h. **Aceitável para PINNs.** Complexidade: ~500-800 LOC adicionais em Fortran.

**Método 3 — Auto-diferenciação (JAX, automática):**

```python
@jax.jit
def loss_with_physics(rho_h, rho_v, H_obs, ...):
    H_pred = simulate_jax(rho_h, rho_v, ...)
    loss_data = jnp.mean((H_pred - H_obs)**2)
    dH_drho_h = jax.jacobian(simulate_jax, argnums=0)(rho_h, rho_v, ...)
    loss_physics = jnp.mean(dH_drho_h**2)
    return loss_data + lambda_phys * loss_physics

grad_fn = jax.grad(loss_with_physics, argnums=(0, 1))
grads = grad_fn(rho_h, rho_v, H_obs, ...)
```

Custo: ~3x forward pass (auto-diff reverse mode [45]). **Idêntico ao adjoint method, mas sem implementação manual.**

#### 14.10.3 Comparação dos Métodos

| Método | Custo/modelo | LOC adicionais | Auto-diff 2a ordem | GPU | Ref. |
|:-------|:------------|:--------------:|:-------------------:|:---:|:----:|
| Diferenças finitas | 60x forward | ~50 | Não | Sim | -- |
| Adjoint method (Fortran) | 3x forward | ~500-800 | Manual | Não | [42, 43] |
| **JAX auto-diff** | **3x forward** | **~0 (automático)** | **Sim** | **Sim** | [16, 45] |

**Recomendação:** JAX auto-diff é a abordagem ótima — custo equivalente ao adjoint, zero implementação adicional, suporte a derivadas de ordem superior (Hessiano para análise de incerteza [46]).

---

### 14.11 Projeção Comparativa de Throughput

| Plataforma | Throughput (mod/h) | vs Fortran | Hardware | Custo/modelo |
|:-----------|:------------------:|:----------:|:--------:|:------------:|
| **Fortran otimizado (atual)** | **58.856** | **1,0x** | i9-9980HK 8c | 0,061 s |
| Numba CPU (8 threads) | ~50.000-55.000 | ~0,9x | i9-9980HK 8c | 0,065-0,072 s |
| f2py wrapper | ~58.000 | ~1,0x | i9-9980HK 8c | 0,062 s |
| JAX CPU (8 cores) | ~45.000-55.000 | ~0,8-0,9x | i9-9980HK 8c | 0,065-0,080 s |
| **Numba CUDA (T4)** | **~200.000-500.000** | **~4-8x** | Tesla T4 16GB | 7-18 ms |
| **JAX GPU (T4)** | **~150.000-400.000** | **~3-7x** | Tesla T4 16GB | 9-24 ms |
| JAX GPU (A100) | ~500.000-1.500.000 | ~8-25x | A100 80GB | 2,4-7,2 ms |
| Numba CUDA (A100) | ~800.000-2.000.000 | ~14-34x | A100 80GB | 1,8-4,5 ms |
| JAX `vmap` batch (A100) | ~2.000.000-5.000.000 | ~34-85x | A100 80GB (batch=1024) | 0,7-1,8 ms |

**Notas métodológicas:** As estimativas de GPU são baseadas em:
- Speedup teórico: razão de FLOPS (A100: 19,5 TFLOPS FP64 vs i9-9980HK: ~0,3 TFLOPS FP64) corrigido por eficiência de utilização (~15-30% para carga com branches) [29]
- Overhead de transfer CPU<->GPU amortizado pelo batch (latência ~10 microsegundos por transfer, negligível para batch >= 100)
- Referências empíricas: Commer & Newman [20] reportaram 10-50x para EM 3D em GPU; modelagem 1D é mais favorável por menor carga de memória

**Conclusão:** Em CPU puro, Python/Numba atinge ~90-100% do Fortran. Em GPU, Python supera o Fortran CPU em 4-85x, dependendo do hardware e do grau de batching.

---

### 14.12 Recomendação Estratégica (Track 1/2/3)

```
┌─────────────────────────────────────────────────────────────────┐
│  Track 1 — Produção imediata (curto prazo, 2–4 semanas)        │
│    f2py wrapper do Fortran otimizado existente                  │
│    → Zero reescrita, performance máxima CPU                     │
│    → Integra com geosteering_ai/ via import direto              │
│    → Limitação: sem GPU, sem auto-diferenciação                 │
│                                                                 │
│  Track 2 — Evolução principal (médio prazo, 2–3 meses)         │
│    Reimplementação em JAX                                       │
│    → Auto-diferenciação nativa (dH/drho para PINNs)            │
│    → GPU via XLA (sem reescrita CUDA manual)                    │
│    → jax.vmap para batch de modelos (throughput massivo)        │
│    → Compatível com TensorFlow/Keras                            │
│                                                                 │
│  Track 3 — Performance máxima (longo prazo, 3–6 meses)         │
│    Numba CUDA para hot path específicos                         │
│    → Máxima performance em GPU NVIDIA para geração de dados     │
│    → Shared memory + warp shuffle para somas de Hankel          │
│    → Complementar ao JAX (pré-geração de datasets)              │
└─────────────────────────────────────────────────────────────────┘
```

**Decisão atual (Abril 2026):** O Track 1 (f2py wrapper) está implementado e em uso. O Track 2 (JAX) é a próxima prioridade para habilitar PINNs e GPU. O Track 3 (Numba CUDA) é reservado para quando o throughput de geração de dados se tornar gargalo.

---

### 14.13 Vantagens Python sobre Fortran

#### 14.13.1 Integração Nativa com o Pipeline de Deep Learning

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

| Aspecto | Fortran (externo) | Python (nativo) | Ganho |
|:--------|:-----------------|:----------------|:------|
| I/O por modelo | ~0.1 s (disco) | Zero | Eliminado |
| Overhead de subprocess | ~0.05 s/chamada | Zero | Eliminado |
| Parse binário (.dat) | ~0.05 s | Zero | Eliminado |
| Memória compartilhada | Não (IPC) | Sim (in-process) | Total |
| Integração com tf.data | Via tf.py_function com I/O | Direta | Nativa |
| Debugging | Difícil (Fortran separado) | Fácil (Python unificado) | Alto |

#### 14.13.2 Diferenciação Automática (AD)

| Critério | Diferenças Finitas (Fortran) | AD via JAX (Python) |
|:---------|:----------------------------|:--------------------|
| Custo/gradiente | O(n_params) chamadas | O(1) — custo fixo ~3x forward |
| Precisão | O(delta) — erros de truncamento | Exato (FP64) |
| Estabilidade numérica | Sensível ao passo delta | Estável |
| Integração com PINNs | Complexa (subprocess loop) | Nativa (jax.grad -> tf.py_function) |
| Gradientes de ordem superior | Muito custoso | jax.hessian() disponível |

#### 14.13.3 Portabilidade e Manutenibilidade

```
┌──────────────────────────────────────────────────────────────────────┐
│  PORTABILIDADE: FORTRAN vs PYTHON                                     │
│                                                                      │
│  Fortran (PerfilaAnisoOmp):                                          │
│    x Requer gfortran >= 9.0 + OpenMP runtime                        │
│    x Compilação separada (make) em cada ambiente                     │
│    x Google Colab: requer !apt-get install + !make a cada sessão     │
│    x Windows: MinGW necessário, comportamento OpenMP diferente       │
│    x ARM (M1/M2 Mac): cross-compilação necessária                    │
│                                                                      │
│  Python (Numba/JAX):                                                 │
│    ok pip install numba jax — sem compilador externo                 │
│    ok Google Colab: disponível por padrão (numba pré-instalado)      │
│    ok Windows/Linux/macOS: comportamento idêntico                    │
│    ok ARM (M1/M2): Numba e JAX têm builds nativas                   │
│    ok GitHub Actions CI: sem etapa de compilação Fortran             │
│    ok Docker: imagem simples Python:3.10-slim + requirements.txt     │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 14.14 Avaliação Comparativa Fortran vs Python

#### 14.14.1 Tabela Comparativa Completa

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
| **Reprodutibilidade** | Parcial | Total (seed único) | Parcial (GPU non-det.) | Parcial |
| **Manutenibilidade** | Baixa (Fortran) | Alta (Python+docstrings) | Alta | Alta |
| **Suporte a PINNs** | Não | Não | Não | Sim (AD exato) |
| **Custo de desenvolvimento** | Já pronto | ~8 semanas | ~4 sem. adicionais | ~4 sem. adicionais |
| **Validação numérica** | Referência | rtol < 1e-10 vs Fortran | rtol < 1e-6 vs Numba | rtol < 1e-6 vs Numba |
| **Recomendação** | Referência + legado | Produção sem GPU | Produção com GPU | PINNs + AD |

#### 14.14.2 Estratégia de Migração Incremental

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

---

## 15. Evolução em Geofísica, Petrofísica e IA

### 15.1 Evolução do Simulador Fortran (Curto Prazo)

| # | Recurso | Descrição | Impacto | Complexidade | Prioridade | Ref. |
|:-:|:--------|:----------|:--------|:-------------|:----------:|:----:|
| 1 | **Múltiplos pares T-R** | Loop sobre `nTR` espaçamentos com cache Fase 4 indexado por par | Datasets multi-escala -> modelos DL mais robustos | ~200 LOC | **Alta** | [31, 36] |
| 2 | **Tensor completo (9 componentes)** | Saída de todas as componentes de H, não apenas diagonais | Treinamento com tensor completo -> inversão 3D | ~50 LOC (já computado) | **Alta** | [47] |
| 3 | **Batch paralelo de modelos** | Paralelismo MPI ou fork/exec sobre modelos independentes | Throughput massivo para geração de datasets | ~150 LOC | **Alta** | -- |
| 4 | **Frequências arbitrárias (>2)** | Suporte a nf = 4-8 (10, 20, 40, 100, 400 kHz) | Inversão multi-frequência com DOI variável | Baixa (configuração) | Média | [48] |
| 5 | **f2py wrapper** | Interface Python automática para o Fortran existente | Integração imediata com pipeline DL | ~100 LOC wrapper | **Alta** | [24] |

### 15.2 Versão Python/JAX (Médio Prazo)

| # | Recurso | Descrição | Impacto | Prioridade | Ref. |
|:-:|:--------|:----------|:--------|:----------:|:----:|
| 1 | **Reimplementação em JAX** | Tradução completa do simulador para JAX com `@jit` | Auto-diff + GPU + integração DL | **Crítica** | [16] |
| 2 | **`jax.vmap` batch** | Vetorização sobre batch de 100-10.000 modelos | Throughput 10-100x em GPU | **Alta** | [16] |
| 3 | **`jax.jacobian` para dH/drho** | Jacobiano automático para PINNs e inversão | Treinamento physics-informed | **Crítica** | [7, 8, 45] |
| 4 | **Surrogaté diferenciável** | JAX simulator como teacher -> SurrogateNet herda differentiability | Inferência rápida + gradientes | Alta | [49] |
| 5 | **Mixed-precision** | float32 forward, float64 backward | 2x throughput GPU no forward | Média | [50] |

### 15.3 Avanços em Geofísica e Petrofísica

#### 15.3.1 Modelo de Rocha Integrado (Rock Physics)

Conectar resistividade elétrica a propriedades petrofísicas via equação de Archie [51] e extensões:

```
sigma_w = f(T, salinity)            — condutividade da água de formação
sigma_bulk = sigma_w * phi^m * S_w^n / a  — equação de Archie modificada

onde:
  phi = porosidade
  m   = expoente de cimentação (~1,8-2,2 para arenitos)
  n   = expoente de saturação (~1,8-2,0)
  Sw  = saturação de água
  a   = fator de tortuosidade (~0,6-1,0)
```

O simulador passaria a receber `(phi, Sw, salinidade, litologia)` em vez de `(rho_h, rho_v)`, e computaria internamente `rho_h(phi, Sw)` e `rho_v(phi, Sw)` via relações empíricas de anisotropia (Klein et al. [52], Clavaud et al. [53]).

**Impacto para PINNs:** A rede inverteria diretamente para `(phi, Sw)` — propriedades com significado petrofísico direto — em vez de resistividades que requerem pós-interpretação.

#### 15.3.2 Efeito de Invasão

Em poços perfurados com lama base água, o filtrado da lama invade a formação criando uma zona com resistividade diferente da formação virgem [54]:

```
┌──────────────────────────────────────────────┐
│  Poço   │  Zona lavada  │ Zona transição │ Formação │
│ (lama)  │  (Rxo, Sxo)   │  (blend)       │ (Rt, Sw)  │
│   ←───────  ri  ──────→                              │
└──────────────────────────────────────────────┘
```

A modelagem requer perfil radial de resistividade, implementável como camadas cilíndricas concêntricas (extensão de Anderson [55]).

#### 15.3.3 Modelos 1.5D (Camadas Inclinadas)

O modelo 1D assume que o poço é perpendicular às camadas. Em geosteering, o poço cruza as camadas com ângulo relativo (relative dip) — criando efeitos de boundary (horns) nas transições [56, 57]:

```
                          poço (70 graus do vertical)
                         /
   ─────────────────────/──────── interface 1
                       /
   ───────────────────/────────── interface 2
                     /
```

A extensão 1.5D (Li & Wang [58]) modifica os coeficientes de reflexão para incluir o ângulo de cruzamento. Impacto: essencial para geosteering real — o poço quase nunca é perpendicular.

#### 15.3.4 Inversão Conjunta EM + Sônica

Combinar resistividade (EM) com velocidade compressional/cisalhante (sônica) para reduzir a ambiguidade de inversão, especialmente em carbonatos onde resistividade != porosidade (efeito vugular) [59, 60]:

```python
# PINN multi-física
def loss_conjunta(params, H_obs, T_obs):
    rho, Vp, Vs, phi = params  # 4 propriedades por camada
    H_pred = simulate_EM(rho, ...)
    T_pred = simulate_sonic(Vp, Vs, ...)
    loss_EM = mse(H_pred, H_obs)
    loss_sonic = mse(T_pred, T_obs)
    # Constraint petrofísica: Archie + Gassmann
    loss_physics = archie_constraint(rho, phi) + gassmann_constraint(Vp, Vs, phi)
    return loss_EM + loss_sonic + lambda * loss_physics
```

#### 15.3.5 Geosteering em Tempo Real

Pipeline fechado para tomada de decisão na perfuração [9, 10, 61]:

```
┌────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│ Medição │ →  │ Inversão │ →  │ Modelo   │ →  │ Decisão   │
│ LWD    │    │ PINN     │    │ geológico│    │ trajetória│
│ (~1 Hz)│    │ (<50 ms) │    │ atualiz. │    │ (<100 ms) │
└────────┘    └──────────┘    └──────────┘    └───────────┘
```

**Requisitos de latência:**
- Forward model: <10 ms (JAX GPU, batch=1)
- Inversão PINN: <50 ms (inferência WaveNet/ResNet otimizada)
- Atualização modelo + decisão: <40 ms
- **Total: <100 ms** — viável com JAX em GPU

**Quantificação de incerteza:** Essencial para decisões seguras. INN (Invertible Neural Networks [62]) ou MC Dropout [63] fornecem intervalos de confiança nas propriedades invertidas.

---

## 16. Integração com o Pipeline Geosteering AI v2.0

### 16.1 Cadeia Atual (Fortran)

```
fifthBuildTIVModels.py → model.in → tatu.x (Fortran) → .dat → geosteering_ai/data/loading.py
                              |          |                 |
                        Parâmetros    Simulação        Leitura binária
                        geológicos    EM 1D TIV        22 colunas
```

### 16.2 Cadeia Futura (Python Integrado)

```
geosteering_ai/simulation/forward.py → geosteering_ai/data/pipeline.py
                |                              |
          Simulação EM 1D TIV           Split → Noise → FV → GS → Scale
          (Numba/CUDA, on-the-fly)              |
                                          tf.data.Dataset
```

### 16.3 Benefícios da Integração

1. **Eliminação de I/O:** Dados gerados in-memory, sem escrita/leitura de .dat
2. **Geração on-the-fly:** Novos modelos geológicos gerados durante treinamento
3. **Data augmentation física:** Perturbação de parâmetros geológicos (espessuras, resistividades) como augmentation
4. **PINNs:** Backpropagation through o simulador para constraintes físicas
5. **Reprodutibilidade:** Seed único controla toda a cadeia (geração + simulação + treinamento)

### 16.4 Correspondência Fortran -> Pipeline v2.0

| Parâmetro Fortran | Config v2.0 | Valor Default | Descrição |
|:------------------|:------------|:-------------|:----------|
| `freq(1)` | `config.frequency_hz` | 20000.0 | Frequência principal (Hz) |
| `dTR` | `config.spacing_meters` | 1.0 | Espaçamento T-R (m) |
| `nmed(1)` | `config.sequence_length` | 600 | Número de medições |
| `resist(:,1)` | `targets[:, 0]` (col 2) | - | rho_h (Ohm.m) |
| `resist(:,2)` | `targets[:, 1]` (col 3) | - | rho_v (Ohm.m) |
| `cH(f,1:9)` | `features[:, 4:21]` (cols 4-21) | - | Tensor H (18 valores reais) |
| `zobs` | `features[:, 1]` (col 1) | - | Profundidade (m) |

### 16.5 Mapeamento .dat -> Estrutura de 22 Colunas do Pipeline

O pipeline Python (`geosteering_ai/data/loading.py`) interpreta os registros binários de 172 bytes do .dat e os reorganiza em uma estrutura de **22 colunas** que é a representação canônica dos dados ao longo de todo o pipeline:

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
    # → 2 targets em escala log10 (TARGET_SCALING = \"log10\")
```

O modo P1 seleciona apenas Hxx (componente planar) e Hzz (componente axial) porque:
- Hxx é sensível à resistividade horizontal (via modo TE + TM)
- Hzz é sensível à resistividade vertical (via modo TE apenas, em poço vertical)
- Juntos, Hxx e Hzz fornecem informação complementar para resolver rho_h e rho_v

As 14 colunas restantes (cols 6-19) contêm as componentes off-diagonal e a segunda componente diagonal (Hyy = Hxx em poço vertical), que ficam disponíveis para modos futuros (P2 tensor parcial, Modo C tensor completo).

---

## 17. Roadmap Integrado

### 17.1 Visão Temporal

```
2026-Q2 (Abril–Junho):
  ├── f2py wrapper do Fortran → integrar com geosteering_ai/
  ├── Múltiplos pares T-R no Fortran
  ├── Saída tensor completo (9 componentes)
  ├── Batch paralelo de modelos (geração datasets)
  └── Início reimplementação JAX (core: commonarraysMD + commonfactorsMD)

2026-Q3 (Julho–Setembro):
  ├── JAX simulator completo com jax.jit + jax.vmap
  ├── jax.jacobian dH/drho integrado com PINNs
  ├── Modelo de rocha (Archie) integrado
  ├── Treinamento PINN com gradientes do simulador
  └── Benchmark GPU (T4/A100) vs Fortran CPU

2026-Q4 (Outubro–Dezembro):
  ├── Numba CUDA para máxima performance (geração datasets)
  ├── Modelos 1.5D (camadas inclinadas para geosteering)
  ├── Efeito de invasão (zona lavada)
  ├── Geosteering em tempo real (pipeline <100 ms)
  └── Inversão conjunta EM + sônica (protótipo)

2027-Q1 (Janeiro–Março):
  ├── Anisotropia azimutal (ortorômbica)
  ├── Múltiplas ferramentas LWD simultâneas
  ├── Quantificação de incerteza (INN + Ensemble)
  └── Publicação: artigo sobre inversão DL com simulador diferenciável
```

### 17.2 Métricas de Sucesso

| Marco | Métrica | Meta | Prazo |
|:------|:--------|:-----|:------|
| f2py wrapper | Integração com pipeline DL | Funcional | Q2 2026 |
| JAX reimplementação | Throughput GPU | >= 200k mod/h (T4) | Q3 2026 |
| Auto-diff dH/drho | Custo do Jacobiano | <= 3x forward | Q3 2026 |
| PINN com gradientes | Val loss com physics term | <= 1,5x baseline PINN sem physics | Q3 2026 |
| Geosteering real-time | Latência end-to-end | <= 100 ms | Q4 2026 |
| Inversão conjunta | RMSE de (rho, Vp) invertidos | <= 0,7x inversão EM-only | Q1 2027 |

---

## 18. Referências Bibliográficas

### 18.1 Modelagem EM em Meios Estratificados

[1] Anderson, B. I. (2001). **Modeling and inversion methods for the interpretation of resistivity logging tool response.** Ph.D. Thesis, Delft University of Technology. -- Formulação da modelagem 1D EM para ferramentas LWD com anisotropia TIV.

[2] Loseth, L. O., & Ursin, B. (2007). **Electromagnetic fields in planarly layered anisotropic media.** Geophysical Journal International, 170(1), 44-80. doi:10.1111/j.1365-246X.2007.03390.x -- Derivação completa dos campos EM em meios TIV com decomposição TE/TM.

[3] Werthmuller, D. (2017). **An open-source full 3D electromagnetic modeler for 1D VTI media in Python: empymod.** Geophysics, 82(6), WB9-WB19. doi:10.1190/geo2016-0626.1 -- Filtros digitais de Hankel de 201 pontos usados no simulador; implementação Python de referência com Numba.

[4] Bonner, S., et al. (1996). **Resistivity while drilling — images from the string.** Oilfield Review, 8(1), 4-19. -- Descrição das ferramentas LWD de resistividade e princípios de operação.

[5] Ellis, D. V., & Singer, J. M. (2007). **Well Logging for Earth Scientists.** Springer, 2nd edition. ISBN: 978-1-4020-3738-2. -- Referência completa sobre princípios de perfilagem, incluindo ferramentas EM.

[6] Anderson, W. L. (1979). **Numerical integration of related Hankel transforms of orders 0 and 1 by adaptive digital filtering.** Geophysics, 44(7), 1287-1305. doi:10.1190/1.1441007 -- Algoritmo base para transformadas de Hankel digitais; filtro de 801 pontos.

[7] Anderson, W. L. (1982). **Fast Hankel Transforms Using Related and Lagged Convolutions.** ACM Transactions on Mathematical Software, 8(4), 344-368. -- Filtro de 801 pontos para transformada de Hankel.

[8] Kong, F. N. (2007). **Hankel transform filters for dipole antenna radiation in a conductive medium.** Geophysical Prospecting, 55(1), 83-89. -- Filtro de 61 pontos.

[9] Key, K. (2012). **Is the fast Hankel transform faster than quadrature?** Geophysics, 77(3), F21-F30. -- Comparação de desempenho de filtros.

### 18.2 Anisotropia TIV e Dipolos Magnéticos

[10] Anderson, B., Barber, T., & Habashy, T. (2002). **The Interpretation of Multicomponent Induction Logs in the Presence of Dipping, Anisotropic Formations.** SPWLA 43rd Annual Logging Symposium. -- Resposta de ferramentas triaxiais em meios TIV.

[11] Liu, C. (2017). *Theory of Electromagnetic Well Logging.* Elsevier. -- Referência principal para a rotação do tensor (eq. 4.80).

[12] Chew, W. C. (1995). *Waves and Fields in Inhomogeneous Media.* IEEE Press. -- Formulação TE/TM para meios estratificados com anisotropia.

[13] Ward, S. H. & Hohmann, G. W. (1988). **Electromagnetic Theory for Geophysical Applications.** In *Electromagnetic Methods in Applied Geophysics*, Vol. 1, SEG. -- Fundamentação teórica das equações de Maxwell em geofísica.

[14] Zhong, L., et al. (2008). **Computation of Triaxial Induction Logging Tools in Layered Anisotropic Dipping Formations.** IEEE Transactions on Geoscience and Remote Sensing, 46(4), 1148-1163.

[15] Davydycheva, S., Druskin, V., & Habashy, T. (2003). **An efficient finite-difference scheme for electromagnetic logging in 3D anisotropic inhomogeneous media.** Geophysics, 68(5), 1525-1536.

[16] Moran, J. H. & Gianzero, S. (1979). **Effects of formation anisotropy on resistivity-logging measurements.** Geophysics, 44(7), 1266-1286. -- Formulação fundamental dos potenciais de Hertz para meios TIV.

[17] Santos, W. G. (2015). *Modelagem eletromagnética com meios anisotrópicos.* -- Referência para soluções de EDO não-homogênea da equação de pi_z.

[18] Hohmann, G. W. & Nabighian, M. N. (1987). **Electromagnetic Methods in Applied Geophysics.** Cap. 4. -- Argumentos de continuidade para simplificação das condições de contorno.

### 18.3 Propagação em Meios Estratificados

[19] Kennett, B. L. N. (1983). *Seismic Wave Propagation in Stratified Media.* Cambridge University Press. ISBN: 978-0521239332. -- Formulação matricial de propagação em meios estratificados (base da recursão de impedâncias do simulador).

[20] Chave, A. D., & Cox, C. S. (1982). **Controlled electromagnetic sources for measuring electrical conductivity beneath the oceans: 1. Forward problem and model study.** Journal of Geophysical Research, 87(B7), 5327-5338. doi:10.1029/JB087iB07p05327 -- Análise computacional das transformadas de Hankel para EM em meios planares.

### 18.4 Deep Learning e PINNs

[21] He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep residual learning for image recognition.** In Proceedings of the IEEE CVPR, pp. 770-778. doi:10.1109/CVPR.2016.90 -- Demonstração de que redes profundas requerem datasets grandes para generalização.

[22] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.** Journal of Computational Physics, 378, 686-707. doi:10.1016/j.jcp.2018.10.045 -- Framework original de PINNs para problemas governados por EDPs.

[23] Karniadakis, G. E., et al. (2021). **Physics-informed machine learning.** Nature Reviews Physics, 3(6), 422-440. doi:10.1038/s42254-021-00314-5 -- Review abrangente de ML informado por física, incluindo PINNs e operadores.

[24] Lu, L., et al. (2021). **DeepXDE: A deep learning library for solving differential equations.** SIAM Review, 63(1), 208-228. doi:10.1137/19M1274067 -- Biblioteca de PINNs com suporte a operadores diferenciais.

[25] Wang, S., Teng, Y., & Perdikaris, P. (2021). **Understanding and mitigating gradient flow pathologies in physics-informed neural networks.** SIAM Journal on Scientific Computing, 43(5), A3055-A3081. doi:10.1137/20M1318043 -- Análise de patologias de gradiente em PINNs e soluções.

[26] Morales, A. et al. (2025). **Physics-Informed Neural Networks for Triaxial Electromagnetic Inversion with Uncertainty Quantification.** -- PINN para inversão EM triaxial com constrainte TIV.

### 18.5 Geosteering e Inversão EM via Deep Learning

[27] Sato, T., Dupuis, C., & Omeragic, D. (2021). **Real-time geosteering with deep-learning-based electromagnetic inversion.** SPE Journal, 26(05), 2793-2809. doi:10.2118/205345-PA -- Inversão DL em tempo real para geosteering com ferramenta LWD.

[28] Alyaev, S., et al. (2019). **A decision support system for multi-target geosteering.** Journal of Petroleum Science and Engineering, 183, 106381. doi:10.1016/j.petrol.2019.106381 -- Sistema de suporte à decisão para geosteering multi-alvo.

[29] Dupuis, C., & Denichou, J.-M. (2015). **Automatic inversion of deep-directional-resistivity measurements for well placement and navigation.** In SPE Annual Technical Conference and Exhibition. doi:10.2118/178076-MS -- Inversão automática para geosteering.

[30] Shahriari, M., Pardo, D., & Torres-Verdin, C. (2020). **A deep learning approach to design of electromagnetic well-logging responses.** Geophysics, 85(4), E167-E177. doi:10.1190/geo2019-0631.1 -- Demonstração de que DL para inversão EM se beneficia de dados multi-espaçamento.

[31] Puzyrev, V., & Swidinsky, A. (2021). **Deep learning electromagnetic inversion with convolutional neural networks.** Geophysical Journal International, 224(1), 656-667. doi:10.1093/gji/ggaa423 -- Surrogaté neural network para inversão EM com simulador como teacher.

### 18.6 Inversão Geofísica e Sensibilidades

[32] McGillivray, P. R., & Oldenburg, D. W. (1990). **Methods for calculating Frechet derivatives and sensitivities for the non-linear inverse problem.** Geophysical Prospecting, 38(5), 499-524. doi:10.1111/j.1365-2478.1990.tb01859.x -- Métodos de cálculo de sensibilidades para inversão geofísica.

[33] Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model Parameter Estimation.* SIAM. ISBN: 978-0898715729. -- Formulação de Gauss-Newton e regularização para problemas inversos.

[34] Alumbaugh, D. L., & Newman, G. A. (2000). **Image appraisal for 2-D and 3-D electromagnetic inversion.** Geophysics, 65(5), 1455-1467. doi:10.1190/1.1444834 -- Análise de resolução via eigenvalues do Jacobiano.

[35] Plessix, R.-E. (2006). **A review of the adjoint-staté method for computing the gradient of a functional with geophysical applications.** Geophysical Journal International, 167(2), 495-503. doi:10.1111/j.1365-246X.2006.02978.x -- Review do método adjunto para gradientes em geofísica.

[36] Haber, E. (2014). *Computational Methods in Geophysical Electromagnetics.* SIAM. ISBN: 978-1611973792. -- Métodos computacionais para EM incluindo adjoint method.

[37] Pardo, D., Torres-Verdin, C., & Paszynski, M. (2008). **Simulations of 3D DC borehole resistivity measurements with a goal-oriented hp finite-element method. Part 2.** Computational Geosciences, 12(1), 83-89. doi:10.1007/s10596-007-9059-0 -- Derivação de sensibilidades para instrumentos EM de poço.

[38] Key, K. (2009). **1D inversion of multicomponent, multifrequency marine CSEM data.** Geophysics, 74(2), F9-F20. doi:10.1190/1.3058434 -- Inversão 1D de dados EM com sensibilidades analíticas.

[39] Martin, J., et al. (2012). **A stochastic Newton MCMC method for large-scale statistical inverse problems.** SIAM Journal on Scientific Computing, 34(3), A1460-A1487. doi:10.1137/110845598 -- Uso de Hessiano para quantificação de incerteza em inversão.

### 18.7 Ferramentas LWD e Multi-Espaçamento

[40] Schlumberger. (2013). **ARC — Array Resistivity Compensated Tool.** Product datasheet. -- Especificações da ferramenta ARC com 5 espaçamentos.

[41] Li, Q., et al. (2005). **New directional electromagnetic tool for proactive geosteering and accuraté formation evaluation while drilling.** In SPWLA 46th Annual Logging Symposium, Paper UU. -- Ferramenta Periscope com azimuthal EM.

[42] Schlumberger. (2009). **EcoScope — Multifunction LWD Service.** Product datasheet. -- Ferramenta multi-função com 6 espaçamentos de resistividade.

[43] Baker Hughes. (2015). **OnTrak — Multicomponent Induction While Drilling Service.** Technical documentation. -- Ferramenta de indução multi-componente durante perfuração.

[44] Zhang, Z., et al. (2018). **Interpretation of multi-spacing, multi-frequency electromagnetic measurements for the characterization of oil-based mud-filtraté invasion.** Petrophysics, 59(4), 500-513. -- Uso de múltiplos espaçamentos para resolução de invasão.

### 18.8 Modelagem Avançada e Extensões Físicas

[45] Hou, J., et al. (2006). **Finite-difference simulation of borehole EM measurements in 3D anisotropic media.** Geophysics, 71(5), G225-G233. doi:10.1190/1.2245467 -- Modelagem com tensor completo de condutividade.

[46] Wang, T., & Signorelli, J. (2004). **Finite-difference modeling of electromagnetic tool response for logging while drilling.** Geophysics, 69(1), 152-160. doi:10.1190/1.1649383 -- Modelagem multi-frequência para ferramentas LWD.

[47] Rabinovich, M. B., et al. (2004). **Effect of relative dip angle on electromagnetic measurements and formation boundary detection.** Petrophysics, 45(6), 518-532. -- Efeito de dip relativo nas medições EM.

[48] Wang, T., & Fang, S. (2001). **3-D electromagnetic anisotropy modeling using finite differences.** Geophysics, 66(5), 1386-1398. doi:10.1190/1.1487086 -- Modelagem 3D com anisotropia e efeitos de boundary.

[49] Li, H., & Wang, H. (2016). **Investigation of eccentricity effect on induction response in horizontal wells using 3D FEM.** Journal of Petroleum Science and Engineering, 143, 211-225. doi:10.1016/j.petrol.2016.02.030 -- Extensão 1.5D para poços horizontais.

[50] Anderson, B. I., & Barber, T. D. (1997). **Deconvolution and boosting parameters for obsolete Schlumberger resistivity tools.** The Log Analyst, 38(3), 7-14. -- Modelagem com perfis radiais de resistividade.

### 18.9 Petrofísica e Rock Physics

[51] Archie, G. E. (1942). **The electrical resistivity log as an aid in determining some reservoir characteristics.** Transactions of the AIME, 146(01), 54-62. doi:10.2118/942054-G -- Equação fundamental de resistividade-porosidade.

[52] Klein, J. D., et al. (1997). **The petrophysics of electrically anisotropic reservoirs.** The Log Analyst, 38(3), 25-36. -- Relações de anisotropia elétrica em formações estratificadas.

[53] Clavaud, J.-B., et al. (2008). **Pore pressure prediction in hydrocarbon formations.** Petrophysics, 49(3), 226-241. -- Modelos petrofísicos para meios anisotrópicos.

[54] Schlumberger. (2005). *Log Interpretation Charts.* Schlumberger Educational Services. -- Gráficos de interpretação incluindo correção de invasão.

### 18.10 Inversão Conjunta e Métodos Avançados

[55] Hoversten, G. M., et al. (2006). **Direct reservoir parameter estimation using joint inversion of marine seismic AVA and CSEM data.** Geophysics, 71(3), C1-C13. doi:10.1190/1.2194510 -- Inversão conjunta EM + sísmica.

[56] Abubakar, A., et al. (2008). **2.5D forward and inverse modeling for interpreting low-frequency electromagnetic measurements.** Geophysics, 73(4), F165-F177. doi:10.1190/1.2937466 -- Inversão conjunta multi-física.

### 18.11 Quantificação de Incerteza

[57] Ardizzone, L., et al. (2019). **Guided image generation with conditional invertible neural networks.** arXiv:1907.02392. -- INNs para problemas inversos com quantificação de incerteza.

[58] Gal, Y., & Ghahramani, Z. (2016). **Dropout as a Bayesian appróximation: Representing model uncertainty in deep learning.** In ICML, pp. 1050-1059. -- MC Dropout para quantificação de incerteza.

### 18.12 Computação de Alto Desempenho e Frameworks

[59] Lam, S. K., Pitrou, A., & Seibert, S. (2015). **Numba: A LLVM-based Python JIT compiler.** In Proc. LLVM-HPC Workshop, 1-6. doi:10.1145/2833157.2833162 -- Compilador JIT baseado em LLVM para Python numérico.

[60] Harris, C. R., et al. (2020). **Array programming with NumPy.** Nature, 585(7825), 357-362. doi:10.1038/s41586-020-2649-2 -- Fundamentos do ecossistema NumPy para computação numérica.

[61] Bradbury, J., et al. (2018). **JAX: composable transformations of Python+NumPy programs.** GitHub repository. -- Framework de computação diferenciável com compilação XLA.

[62] Abadi, M., et al. (2016). **TensorFlow: A system for large-scale machine learning.** In 12th USENIX OSDI, pp. 265-283. -- Compilador XLA para fusão de kernels e otimização de grafo computacional.

[63] Baydin, A. G., et al. (2018). **Automatic differentiation in machine learning: a survey.** Journal of Machine Learning Research, 18(153), 1-43. -- Survey completo de auto-diferenciação, incluindo forward e reverse mode.

[64] Lattner, C., & Adve, V. (2004). **LLVM: A compilation framework for lifelong program analysis & transformation.** In Proceedings of the IEEE/ACM CGO, pp. 75-86. -- Infraestrutura de compilação LLVM usada por Numba e JAX.

[65] GCC Team. (2023). **GCC Optimization Options.** GCC 13.x Manual. -- Documentação das otimizações do GFortran (-O3, -ffast-math, -march=native).

[66] NVIDIA Corporation. (2023). **Numba CUDA Documentation.** Numba 0.58.x. -- Documentação oficial de kernels CUDA em Numba.

[67] NVIDIA Corporation. (2023). **CUDA C++ Programming Guide: Complex Number Support.** CUDA Toolkit 12.x. -- Limitações e workarounds para aritmética complexa em CUDA.

[68] NVIDIA Corporation. (2023). **A100 Tensor Core GPU Architecture Whitepaper.** -- Especificações de performance: 19,5 TFLOPS FP64, 312 TFLOPS TF32.

[69] Kirk, D. B., & Hwu, W. W. (2016). *Programming Massively Parallel Processors: A Hands-on Approach.* Morgan Kaufmann, 3rd edition. ISBN: 978-0128119860. -- Técnicas de otimização GPU incluindo eliminação de divergência por especialização de kernel.

[70] Kouatchou, J. (2018). **Comparison of Fortran, C, and Python for Scientific Computing.** NASA GSFC Technical Report. -- Benchmark comparativo de linguagens para computação científica.

[71] Commer, M., & Newman, G. A. (2008). **New advances in three-dimensional controlled-source electromagnetic inversion.** Geophysical Journal International, 172(2), 513-535. doi:10.1111/j.1365-246X.2007.03663.x -- Modelagem 3D EM em GPU com speedups de 10-50x sobre CPU.

[72] Okuta, R., et al. (2017). **CuPy: A NumPy-compatible library for NVIDIA GPU calculations.** In Proceedings of the Workshop on ML Systems at NIPS 2017. -- Biblioteca NumPy-compatible para GPU.

[73] Peterson, P. (2009). **F2PY: a tool for connecting Fortran and Python programs.** International Journal of Computational Science and Engineering, 4(4), 296-305. doi:10.1504/IJCSE.2009.029164 -- Ferramenta de wrapping Fortran->Python.

[74] Hu, Y., et al. (2019). **Taichi: a language for high-performance computation on spatially sparse data structures.** ACM Transactions on Graphics, 38(6), 201. doi:10.1145/3355089.3356506 -- DSL para computação paralela com auto-diff parcial.

[75] Micikevicius, P., et al. (2018). **Mixed precision training.** In ICLR 2018. -- Treinamento em precisão mista para aceleração em GPU.

[76] Sobol, I. M. (1967). **On the distribution of points in a cube and the appróximaté evaluation of integrals.** USSR Computational Mathematics and Mathematical Physics, 7(4), 86-112. -- Sequências quasi-aleatórias para amostragem uniforme.

[77] Weiss, C. J. (2013). **Project APhiD: A Lorenz-gauged A-Phi decomposition for parallelized computation of ultra-broadband electromagnetic induction.** Computers & Geosciences, 58, 40-52. -- GPU para modelagem EM.

[78] Commer, M. & Newman, G. A. (2004). **A parallel finite-difference approach for 3D transient electromagnetic modeling with galvanic sources.** Geophysics, 69(5), 1192-1202.

[79] The SciPy Community. (2023). **NumPy complex number support.** NumPy Documentation, v1.26. -- Documentação oficial do suporte a complex128 em NumPy.

---

## 19. Sugestões de Melhorias e Novos Recursos

Esta seção cataloga melhorias propostas para o simulador Fortran `PerfilaAnisoOmp`,
organizadas por prioridade e complexidade de implementação.

### 19.1 Batching Multi-Frequência em Kernels GPU

**Prioridade:** Alta | **Complexidade:** Média | **Impacto estimado:** 2-4x speedup

Na implementação atual, as frequências são processadas sequêncialmente dentro de `fieldsinfreqs`. Para GPU, múltiplas frequências podem ser processadas em paralelo como uma dimensão adicional do grid CUDA:

```
Grid GPU proposto:
  Block: (npt=201, 1, 1)           — threads por ponto do filtro
  Grid:  (nmed=600, nf=2, ntheta)  — blocos por (medição, freq, ângulo)

Benefício: Toda a informação de um modelo geológico é processada em uma única
chamada de kernel, eliminando overhead de lançamento repetido.
```

### 19.2 Seleção Adaptativa de Filtro de Hankel

**Prioridade:** Média | **Complexidade:** Baixa | **Impacto estimado:** 1.5-3x speedup

O simulador utiliza fixamente o filtro Werthmuller de 201 pontos. No entanto, a precisão necessária varia conforme a configuração:

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

### 19.3 Suporte a Diferenciação Automática (para PINNs)

**Prioridade:** Alta | **Complexidade:** Alta | **Impacto estimado:** Habilita PINNs

Para integração com PINNs (`geosteering_ai/losses/pinns.py`), o simulador precisa fornecer gradientes `dH/d(rho_h)` e `dH/d(rho_v)`. Três abordagens são possíveis:

```
Abordagem 1 — Diferenças finitas (imediata):
  dH/d_rho_h = [H(rho_h + eps) - H(rho_h - eps)] / (2*eps)
  Custo: 2x simulações por parâmetro → 2*n_camadas simulações extras
  Precisão: ~10^-6 (limitada por eps)

Abordagem 2 — Reimplementação em JAX (médio prazo):
  jax.grad(forward_em_1d) → gradientes exatos via AD reverso
  Custo: ~2-3x uma simulação (independente de n_parâmetros)
  Precisão: machine epsilon

Abordagem 3 — Equações adjuntas em Fortran (longo prazo):
  Resolver equações adjuntas analíticas para os potenciais de Hertz
  Custo: ~1x simulação extra (independente de n_parâmetros)
  Precisão: analítica
```

### 19.4 Extensão para Modelos 2D/3D

**Prioridade:** Baixa | **Complexidade:** Muito alta | **Impacto estimado:** Novos cenários

O simulador atual assume camadas horizontais infinitas (1D). Extensões possíveis:

- **Camadas inclinadas (tilted layers):** Rotação do sistema de coordenadas antes do cálculo 1D. Viável para ângulos moderados (< 30 graus).
- **Formações com falhas:** Método de Born ou integral de volume para perturbações laterais pequenas em relação ao background 1D.
- **3D completo (finite-element/finite-difference):** Requer novo simulador. Referências: Davydycheva et al. (2003), Commer & Newman (2004).

### 19.5 Suite de Validação Cruzada (Fortran vs Python vs empymod)

**Prioridade:** Alta | **Complexidade:** Média | **Impacto estimado:** Garantia de qualidade

Proposta de teste de válidação automatizado:

```python
# tests/test_fortran_validation.py (proposta)
def test_fortran_vs_empymod():
    \"\"\"Compara saída do Fortran com empymod para 10 modelos canônicos.\"\"\"
    models = [
        # Modelo 1: meio isotrópico homogêneo (solução analítica conhecida)
        {\"n\": 3, \"rho_h\": [1e20, 10, 1e20], \"rho_v\": [1e20, 10, 1e20]},
        # Modelo 2: meio TIV homogêneo (lambda = sqrt(2))
        {\"n\": 3, \"rho_h\": [1e20, 10, 1e20], \"rho_v\": [1e20, 20, 1e20]},
        # Modelo 3: 2 camadas com contraste forte
        {\"n\": 3, \"rho_h\": [1, 1000, 1], \"rho_v\": [1, 1000, 1]},
        # ... mais modelos canônicos
    ]
    for model in models:
        H_fortran = run_fortran_simulation(model)
        H_empymod = run_empymod_simulation(model)
        assert np.allclose(H_fortran, H_empymod, rtol=1e-8)
```

### 19.6 Configurações Multi-Receptor

**Prioridade:** Média | **Complexidade:** Média | **Impacto estimado:** Realismo da ferramenta

Ferramentas LWD reais possuem múltiplos receptores a distâncias diferentes do transmissor (por exemplo, dTR = 0.5 m, 1.0 m, 1.5 m). Extensão proposta:

```
Parâmetro adicional em model.in:
  n_receivers = 3
  dTR_list = 0.5  1.0  1.5

Impacto no código:
  - Loop adicional sobre receptores dentro de fieldsinfreqs
  - Reutilização de commonarraysMD (depende apenas de hordist)
  - Arquivo .dat com n_receivers × 22 colunas por registro
```

### 19.7 Suporte a Anisotropia Biaxial (sigma_x != sigma_y)

**Prioridade:** Baixa | **Complexidade:** Muito alta | **Impacto estimado:** Cenários geológicos avançados

A formulação atual assume simetria TIV (sigma_x = sigma_y = sigma_h). Em formações com fraturas orientadas ou laminação não-horizontal, a condutividade pode ser biaxial:

```
sigma_biaxial = diag(sigma_x, sigma_y, sigma_z)

Onde sigma_x != sigma_y  (quebra da simetria cilíndrica)
```

A decomposição TE/TM não é mais aplicável diretamente neste caso. Seria necessário reformular as equações de propagação usando dois potenciais de Hertz acoplados (pi_x e pi_y), resultando em um sistema 4x4 de equações diferenciais em vez de dois sistemas 2x2 independentes. Isso aumentaria significativamente a complexidade computacional e algorítmica.

---

## 20. Apêndices

### 20.1 Apêndice A — Tabela de Sub-rotinas

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

### 20.2 Apêndice B — Glossário

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
| **PINN** | Physics-Informed Neural Network — rede neural com regularização baseada em equações físicas |
| **AD** | Automatic Differentiation — diferenciação automática via grafo computacional |
| **XLA** | Accelerated Linear Algebra — compilador de grafo para otimização de operações numéricas |

### 20.3 Apêndice C — Mapeamento de Variáveis Fortran -> Python

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

### 20.4 Apêndice D — Validação de Consistência Fortran-Python

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

### 20.5 Apêndice E — Legenda Completa de Variáveis Matemáticas e Código

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
| s_m | `s(i,m)` | `complex(dp)` | 1/m | Constante de propagação TM |
| Y_m = u_m/zeta | `AdmInt(i,m)` | `complex(dp)` | S | Admitância intrínseca (TE) |
| Z_m = s_m/sigma_h | `ImpInt(i,m)` | `complex(dp)` | Ohm | Impedância intrínseca (TM) |

**E.4 Coeficientes de Reflexão**

| Símbolo | Variável Fortran | Forma | Descrição |
|:--------|:-----------------|:------|:----------|
| R_TE_dw^(m) | `RTEdw(i,m)` | `(npt, n)` | Coeficiente reflexão TE descendente |
| R_TE_up^(m) | `RTEup(i,m)` | `(npt, n)` | Coeficiente reflexão TE ascendente |
| R_TM_dw^(m) | `RTMdw(i,m)` | `(npt, n)` | Coeficiente reflexão TM descendente |
| R_TM_up^(m) | `RTMup(i,m)` | `(npt, n)` | Coeficiente reflexão TM ascendente |
| tanh(u_m*h_m) | `tghuh(i,m)` | `(npt, n)` | Tangente hiperbólica estabilizada (TE) |
| tanh(s_m*h_m) | `tghsh(i,m)` | `(npt, n)` | Tangente hiperbólica estabilizada (TM) |

**E.5 Fatores de Onda e Campos**

| Símbolo | Variável Fortran | Forma | Descrição |
|:--------|:-----------------|:------|:----------|
| Mx_dw | `Mxdw(i)` | `(npt)` | Fator onda TM descendente (camada fonte) |
| Mx_up | `Mxup(i)` | `(npt)` | Fator onda TM ascendente (camada fonte) |
| Eu_dw | `Eudw(i)` | `(npt)` | Fator onda TE descendente (camada fonte) |
| Eu_up | `Euup(i)` | `(npt)` | Fator onda TE ascendente (camada fonte) |
| H(3,3) | `matH(3,3)` / `tH(3,3)` | `complex(dp)` | Tensor magnético completo |

**E.6 Geometria e Filtro de Hankel**

| Símbolo | Variável Fortran | Variável Python | Unidade | Descrição |
|:--------|:-----------------|:---------------|:--------|:----------|
| h_0 | `h0` | - | m | Profundidade vertical do transmissor |
| z | `z` | - | m | Profundidade vertical do receptor |
| r | `hordist` | - | m | Distância horizontal T-R |
| theta | `theta(k)` | - | graus | Ângulo de inclinação do poço |
| dTR | `dTR` | `config.spacing_meters` | m | Espaçamento T-R |
| z_obs | `zrho(f,1)` | `col 1` | m | Profundidade do ponto médio T-R |
| nmed | `nmed(k)` | `config.sequence_length` | - | Número de medições por ângulo |
| kr_i | `absc(i)` | - | 1/m | Abscissas do filtro escaladas |
| w_J0,i | `wJ0(i)` | - | - | Pesos do filtro para Bessel J_0 |
| w_J1,i | `wJ1(i)` | - | - | Pesos do filtro para Bessel J_1 |
| npt | `npt` | - | - | Número de pontos do filtro (201) |

---

### 20.6 Histórico de Execução de Otimizações

| Data | Fase | Status | Relatório / Commit |
|:-----|:-----|:-------|:-------------------|
| 2026-04-04 | CPU Fase 0 — Baseline | Concluída | `relatorio_fase0_fase1_fortran.md` / `43709bf` |
| 2026-04-04 | CPU Fase 1 — SIMD Hankel | Pulada | `relatorio_fase0_fase1_fortran.md` S3 / `43709bf` |
| 2026-04-04 | CPU Fase 2 — Hybrid Scheduler + Débitos 1/2/3 | Concluída | `relatorio_fase2_debitos_fortran.md` / `6ac51ca` |
| 2026-04-05 | CPU Fase 3 — Workspace Pre-allocation | Concluída | `relatorio_fase3_fortran.md` / `c213b66` |
| 2026-04-05 | PR1 Hygiene — Débitos D4/D5/D6 | Concluída | `relatorio_fase3_fortran.md` S5 / `db997d2` |
| 2026-04-05 | CPU Fase 4 — Cache `commonarraysMD` | Concluída | `relatorio_fase4_fortran.md` / `44acf2e` |
| 2026-04-05 | Validação Final — Testes bit-exatos Fases 0->4 | Concluída | `relatorio_validacao_final_fortran.md` / `195b69f` |
| 2026-04-05 | PR Débitos B1/B3/B5/B7 + Fase 5 | Concluída | `relatorio_fase5_debitos_fortran.md` / `fc208c1` |
| 2026-04-05 | CPU Fase 3b — Workspace estendido (12 campos) | Concluída | `0aa0aa9` |
| 2026-04-05 | CPU Fase 5b — Paralelismo adaptativo `if(ntheta>1)` | Concluída | `0aa0aa9` |
| 2026-04-05 | CPU Fase 2b — `schedule(guided, 16)` | Concluída | `8de722e` |
| 2026-04-05 | CPU Fase 6 — Cache `commonfactorsMD` por `camadT` | Erro conceitual | `analise_paralelismo_cpu_fortran.md` S7.7 |
| 2026-04-05 | CPU Fase 6b — Fatoração invariantes `h0` | Descartada (NaN) | `analise_paralelismo_cpu_fortran.md` S7.9 |
| 2026-04-05 | Análise de Evolução Fortran -> Python/JAX/Numba | Documento publicado | `analise_evolucao_simulador_fortran_python.md` |
| 2026-04-06 | Multi-TR, f2py wrapper, batch parallel | Implementados | `931e486` |

**Baseline publicado (Fase 0, 2026-04-04 — i9-9980HK, 8 cores, AVX-2, gfortran 15.2.0, OMP=8, CPU fria):**
- Wall-time: **0,1047 +/- 0,015 s/modelo**
- Throughput: **~34.400 modelos/hora**
- MD5 de referência: `c64745ed5d69d5f654b0bac7dde23a95`

**Tabela Consolidada de Performance — Todas as Fases:**

| Fase | Commit | Tempo/modelo (8t) | Throughput (mod/h) | Speedup vs Baseline | Validação @ -O0 |
|:----:|:------:|:------------------:|:------------------:|:-------------------:|:---------------:|
| **0 (Baseline)** | `43709bf` | 0,1047 s | 34.400 | 1,0x | `c64745ed...` |
| **2 (Hybrid)** | `6ac51ca` | ~0,100 s | ~36.000 | ~1,05x | MD5 = Fase 0 |
| **3 (Workspace)** | `c213b66` | 0,343 s* | 10.485* | -- | `max\\|Delta\\|=3,4e-14` |
| **PR1 (D4/D5/D6)** | `db997d2` | = Fase 3 | = Fase 3 | -- | MD5 = Fase 3 |
| **4 (Cache)** | `44acf2e` | 0,062 s | 58.064 | **5,54x** | `97123697...` |
| **Validação** | `195b69f` | -- | -- | -- | bit-exato confirmado |
| **PR2 (B1-B7)** | `fc208c1` | = Fase 4 | = Fase 4 | -- | MD5 = Fase 4 |
| **5 (Single)** | `fc208c1` | 0,069 s | 51.923 | ~5,0x | `ffd13177...` |
| **3b (WS ext.)** | `0aa0aa9` | = Fase 5 | = Fase 5 | -- | MD5 = Fase 5 |
| **5b (Adaptativo)** | `0aa0aa9` | 0,067 s | 53.865 | ~5,1x | MD5 = Fase 5 |
| **2b (Guided)** | `8de722e` | **0,061 s** | **58.856** | **5,62x** | `97123697...` |
| **6 (Erro)** | -- | -- | -- | -- | Não implementada |
| **6b (NaN)** | -- | -- | -- | -- | 21.600 NaN |

*\\* Fase 3 medida com `model.in` diferente (n=29 vs n=15 das demais); valores não diretamente comparáveis.*

**Estado final:** 0,061 s/modelo, 58.856 mod/h, **245% da meta original** (24k mod/h). Fidelidade numérica: bit-exato @ `-O0` vs código original (Fase 2), `max|Delta| ~ 4 x 10^-13` @ `-O3 -ffast-math`.

---

### 20.7 Documentos Relacionados

| Documento | Escopo |
|:----------|:-------|
| [`analise_paralelismo_cpu_fortran.md`](analise_paralelismo_cpu_fortran.md) | Roteiro de 6 fases de otimização CPU, análise de gargalos, resultados empíricos S7.1-S7.9 |
| [`analise_evolucao_simulador_fortran_python.md`](analise_evolucao_simulador_fortran_python.md) | Viabilidade de conversão Fortran -> Python/JAX/Numba; múltiplos pares T-R; derivadas dH/drho; roadmap de evolução com 63 referências bibliográficas |
| [`relatorio_validacao_final_fortran.md`](relatorio_válidação_final_fortran.md) | Matriz de testes bit-exatos Fases 0->4; revisão de código; infraestrutura de válidação |
| [`relatorio_fase4_fortran.md`](relatorio_fase4_fortran.md) | Relatório executivo da Fase 4 (cache `commonarraysMD`); speedup 5,54x; válidação pós-execução S9 |
| [`relatorio_fase5_debitos_fortran.md`](relatorio_fase5_debitos_fortran.md) | Fase 5 + PR Débitos B1/B3/B5/B7; Fase 6 erro conceitual descoberto |
| [`analise_novos_recursos_simulador_fortran.md`](analise_novos_recursos_simulador_fortran.md) | Análise detalhada de estratégias 1.5D, 2D, compensação de poço, antenas inclinadas, invasão, sensibilidades, e otimização OpenMP avançada. Abril 2026 |

---

## 21. Roadmap de Novos Recursos — Estratégias Avançadas (v9.0+)

Esta seção documenta as propostas de novos recursos planejados para as próximas versões do
simulador, com fundamentos físicos, estimativas de implementação e prioridades.

Documento detalhado: [`analise_novos_recursos_simulador_fortran.md`](analise_novos_recursos_simulador_fortran.md)

### 21.1 Visão Geral dos Novos Recursos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ROADMAP DE NOVOS RECURSOS — SIMULADOR FORTRAN v9.0+                       │
│                                                                             │
│  Curto Prazo (Q2 2026):                                                    │
│  ├── Feature 5: Frequências arbitrárias (nf > 2)         [~50 LOC]         │
│  ├── Feature 6: Compensação de poço (midpoint multi-TR)  [~300 LOC]        │
│  ├── Feature 7: Antenas inclinadas (tilted coils)        [~50 LOC]         │
│  └── Feature 10: Sensibilidades ∂H/∂ρ (diff. finitas)   [~800 LOC]        │
│                                                                             │
│  Médio Prazo (Q3-Q4 2026):                                                 │
│  ├── Feature 8: Correção 1.5D (relative dip)             [~500 LOC]        │
│  ├── Feature 9: Efeito de invasão (mud filtrate)         [~400 LOC]        │
│  └── Feature 11: Modelo de rocha (Archie no gerador)     [~200 LOC]        │
│                                                                             │
│  Longo Prazo (2027+):                                                      │
│  ├── Born 2D: Aproximação de espalhamento 2D             [~1200 LOC]       │
│  └── Anisotropia ortorrômbica (σx ≠ σy ≠ σz)            [Reformulação]    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 21.2 Feature 5 — Frequências Arbitrárias (nf > 2)

**Prioridade:** Alta | **Complexidade:** Baixa | **LOC:** ~50

O simulador atual suporta `nf = 2` frequências (20 kHz e 40 kHz). Ferramentas LWD reais
operam com 4-8 frequências (10, 20, 40, 100, 200, 400 kHz) para investigar diferentes
profundidades (DOI). A profundidade de investigação é inversamente proporcional à frequência:

```
DOI ∝ √(ρ / (2πfμ))

Frequências típicas e DOI para ρ = 10 Ohm.m:
  ┌─────────────────────────────────────────────────────────────┐
  │  Frequência (kHz)  │  Skin depth (m)  │  DOI aprox. (m)    │
  ├────────────────────┼──────────────────┼────────────────────┤
  │  10                │  5.03            │  10-15             │
  │  20                │  3.56            │  7-10              │
  │  40                │  2.52            │  5-7               │
  │  100               │  1.59            │  3-5               │
  │  200               │  1.13            │  2-3               │
  │  400               │  0.80            │  1.5-2             │
  └─────────────────────────────────────────────────────────────┘
```

**Implementação:** Extensão do cache Fase 4 para `nf × nTR` combinações. O cache
`commonarraysMD` já suporta a dimensão `nf` — basta aumentar o range de `nf` no `model.in`.
A saída `.dat` escala linearmente com `nf`.

### 21.3 Feature 6 — Compensação de Poço (Midpoint Multi-TR)

**Prioridade:** Alta | **Complexidade:** Média | **LOC:** ~300

A compensação de poço elimina efeitos ambientais (rugosidade, excentricidade, invasão
de lama) via simetrização de medições de múltiplos pares T-R.

**Princípio físico:** Um arranjo simétrico com dois transmissores (T1 acima e T2 abaixo
dos receptores) produz medições compensadas:

```
                    T1
                    │
            R1 ────┤──── R2       ← Receptores
                    │
                    T2

  Compensação por fase:
    φ_comp = (φ_T1R + φ_T2R) / 2

  Compensação por atenuação:
    α_comp = (α_T1R + α_T2R) / 2

  Simetrização (geométrica):
    H_sym = √(H_T1R1 × H_T2R2 / (H_T1R2 × H_T2R1))
```

**O ponto médio** (midpoint) é a posição geométrica central entre os dois transmissores,
à qual a medição compensada é atribuída. Este ponto é o referencial de profundidade
para dados compensados.

**Canais derivados:**
- Phase difference: `Δφ = φ_near - φ_far` (sensível a resistividade)
- Attenuation: `Δα = 20×log10(|V_near|/|V_far|)` (sensível a resistividade)
- Symmetrized ratio: elimina erros de ganho e efeitos ambientais

**Implementação:** Pós-processamento sobre a saída Multi-TR (Feature 1). Nova sub-rotina
`borehole_compensation()` ou processamento no pipeline Python (`data/compensation.py`).

### 21.4 Feature 7 — Antenas Inclinadas (Tilted Coils)

**Prioridade:** Média | **Complexidade:** Baixa | **LOC:** ~50

Ferramentas LWD modernas (Rt Scanner, EarthStar, GeoSphere) utilizam antenas inclinadas
a 15°-45° do eixo do poço. A resposta de uma antena inclinada é uma combinação linear
dos componentes do tensor H(3×3):

```
H_tilted(β, φ) = cos(β) × H_axial + sin(β) × [cos(φ) × H_x + sin(φ) × H_y]

Onde:
  β = ângulo de inclinação da antena (relativo ao eixo do poço)
  φ = ângulo azimutal da antena (rotação no plano transversal)
```

**Implementação:** O simulador já calcula o tensor completo H(3×3). A combinação linear
é pós-processamento — sem modificação do core do simulador. Nova sub-rotina em `utils.f08`
ou processamento no pipeline Python.

### 21.5 Feature 8 — Correção 1.5D (Relative Dip)

**Prioridade:** Alta | **Complexidade:** Alta | **LOC:** ~500

O modelo 1D assume que o poço é perpendicular às camadas. Em geosteering real, o poço
cruza as camadas com um ângulo relativo (relative dip), produzindo efeitos de "horn"
nas transições de camadas.

```
Modelo 1D (poço vertical, camadas horizontais):

  ─────────────────────────────  interface
         │ poço
  ─────────────────────────────  interface

Modelo 1.5D (poço inclinado, camadas inclinadas):

  ─────────────────/──────────  interface inclinada
                  / poço
  ──────────────/─────────────  interface inclinada
               /
```

**Formulação:** A extensão 1.5D modifica os coeficientes de reflexão TE/TM para incluir
o ângulo de cruzamento entre poço e camadas (relative dip = dip_camadas - dip_poço):

```
Coeficientes modificados:
  u_eff(i) = u(i) × cos(θ_rel)
  s_eff(i) = s(i) × cos(θ_rel)

  Onde θ_rel = ângulo relativo poço-camadas
```

**Impacto no treinamento DL:** Essencial para datasets realistas. Modelos treinados
apenas com relative dip = 0° não generalizam para poços horizontais.

### 21.6 Feature 9 — Efeito de Invasão (Mud Filtrate)

**Prioridade:** Média | **Complexidade:** Média | **LOC:** ~400

Em poços perfurados com lama base água, o filtrado da lama invade a formação criando
uma zona lavada (flushed zone) com resistividade diferente da formação virgem:

```
  Poço    Zona lavada    Zona transição    Formação virgem
  (Rm)    (Rxo, Sxo)     (blend)           (Rt, Sw)
  ←─── ri (raio invasão) ───→
```

**Modelo step:** Duas camadas radiais concêntricas por camada geológica.
**Modelo gradient:** Perfil radial contínuo Rxo → Rt.

### 21.7 Feature 10 — Sensibilidades ∂H/∂ρ (Jacobiano)

**Prioridade:** Alta | **Complexidade:** Alta | **LOC:** ~800

O Jacobiano ∂H/∂ρ é fundamental para PINNs e inversão determinística. Três abordagens:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Método              │  Custo/modelo    │  LOC    │  Plataforma     │
├──────────────────────┼──────────────────┼─────────┼─────────────────┤
│  Diferenças finitas  │  2×n_params × fw │  ~800   │  Fortran/f2py   │
│  Adjoint method      │  ~3× forward     │  ~1200  │  Fortran        │
│  Auto-diferenciação  │  ~3× forward     │  ~0     │  JAX (Python)   │
└──────────────────────────────────────────────────────────────────────┘
```

**Recomendação:** Diferenças finitas via f2py (imediato) + JAX auto-diff (médio prazo).

### 21.8 Otimização OpenMP Avançada

Próximas otimizações de paralelismo após Phase 2b:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Otimização                  │  Ganho estimado  │  Pré-requisito    │
├──────────────────────────────┼──────────────────┼───────────────────┤
│  AVX-512 vectorization       │  10-20%          │  Hardware AVX-512 │
│  NUMA-aware allocation       │  Significativo   │  Multi-socket     │
│  Task-based parallelism      │  5-15%           │  OpenMP 4.5+      │
│  Seleção adaptativa filtro   │  1.5-3×          │  Nenhum           │
└──────────────────────────────────────────────────────────────────────┘
```

### 21.9 Integração com Pipeline DL v2.0

Os novos recursos do simulador se integram ao pipeline via:

| Recurso Fortran | Módulo Python v2.0 | Impacto |
|:----------------|:-------------------|:--------|
| Freq. arbitrárias | `config.py` (nf field) | Multi-DOI training |
| Compensação | `data/compensation.py` (novo) | Realistic features |
| Antenas inclinadas | `data/feature_views.py` | Tilted-coil features |
| 1.5D correction | `data/loading.py` | Dip-aware training |
| Invasão | `fifthBuildTIVModels.py` | More realistic data |
| ∂H/∂ρ | `losses/pinns.py` | Physics-informed loss |

---

*Documentação do Simulador Fortran PerfilaAnisoOmp — Geosteering AI v2.0*
*Versão 9.0 — Abril 2026 — Pipeline v5.0.15+*
*Última atualização: 2026-04-08 (Análise de novos recursos: 1.5D, 2D, compensação, antenas inclinadas, invasão, sensibilidades)*
