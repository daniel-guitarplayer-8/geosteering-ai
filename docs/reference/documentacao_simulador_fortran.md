# Documentacao Completa do Simulador Fortran — PerfilaAnisoOmp

## Simulacao Eletromagnetica 1D para Meios Anisotrópicos TIV com Paralelismo OpenMP

**Projeto:** Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning
**Autor do Simulador:** Daniel Leal
**Linguagem:** Fortran 2008 (gfortran) com extensoes OpenMP
**Localizacao:** `Fortran_Gerador/`
**Versao do Documento:** 1.0 (Abril 2026)

---

## Sumario

1. [Introducao e Motivacao](#1-introducao-e-motivacao)
2. [Fundamentos Fisicos e Geofisicos](#2-fundamentos-fisicos-e-geofisicos)
3. [Formulacao Matematica Completa](#3-formulacao-matematica-completa)
4. [Arquitetura do Software](#4-arquitetura-do-software)
5. [Modulos Fortran — Analise Detalhada](#5-modulos-fortran--analise-detalhada)
6. [Arquivo de Entrada model.in](#6-arquivo-de-entrada-modelin)
7. [Arquivos de Saida (.dat e .out)](#7-arquivos-de-saida-dat-e-out)
8. [Sistema de Build (Makefile)](#8-sistema-de-build-makefile)
9. [Gerador de Modelos Geologicos (Python)](#9-gerador-de-modelos-geologicos-python)
10. [Paralelismo OpenMP — Analise e Otimizacao](#10-paralelismo-openmp--analise-e-otimizacao)
11. [Analise de Viabilidade CUDA (GPU)](#11-analise-de-viabilidade-cuda-gpu)
12. [Analise de Reimplementacao em Python Otimizado](#12-analise-de-reimplementacao-em-python-otimizado)
13. [Integracao com o Pipeline Geosteering AI v2.0](#13-integracao-com-o-pipeline-geosteering-ai-v20)
14. [Referencias Bibliograficas](#14-referencias-bibliograficas)
15. [Apendices](#15-apendices)

---

## 1. Introducao e Motivacao

### 1.1 Contexto do Problema

O simulador `PerfilaAnisoOmp` resolve o *problema direto* (forward modeling) da resposta eletromagnetica (EM) de uma ferramenta de perfilagem triaxial em meios estratificados horizontalmente com anisotropia TIV (Transversalmente Isotropica Vertical). Este tipo de modelagem e fundamental para a geracao de dados sinteticos de treinamento para redes neurais de inversao.

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

### 1.2 Objetivo do Simulador

Dado um modelo geologico 1D (camadas horizontais com resistividades anisotropicas), o simulador calcula o tensor completo de campo magnetico `H(3x3)` medido por uma ferramenta triaxial LWD (Logging While Drilling) em cada posicao de medicao ao longo do poco. O resultado e armazenado em formato binario `.dat` para consumo pelo pipeline de Deep Learning.

### 1.3 Escopo da Documentacao

Este documento apresenta:
- Fundamentos fisicos completos (equacoes de Maxwell em meios TIV)
- Formulacao matematica detalhada (decomposicao TE/TM, transformada de Hankel, coeficientes de reflexao recursivos)
- Analise linha a linha do codigo Fortran
- Estrutura de dados de entrada e saida
- Analise completa de paralelismo e otimizacao
- Viabilidade de portabilidade para CUDA e Python

---

## 2. Fundamentos Fisicos e Geofisicos

### 2.1 O Problema Eletromagnetico em Meios Estratificados

A perfilagem EM de pocos utiliza fontes dipolares magneticas operando em baixa frequencia (tipicamente 20 kHz a 2 MHz) para sondar as propriedades eletricas das formacoes rochosas ao redor do poco. O campo eletromagnetico emitido pelo transmissor penetra no meio geologico e e modificado pelas propriedades de resistividade de cada camada. Os receptores medem o campo resultante, que contem informacao sobre a distribuicao de resistividade.

### 2.2 Anisotropia TIV (Transversalmente Isotropica Vertical)

Rochas sedimentares apresentam anisotropia eletrica intrinseca devida a laminacao deposicional. No modelo TIV, a condutividade e descrita por um tensor diagonal:

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
  Range tipico: 1.0 <= lambda <= sqrt(2) ~ 1.414
```

**Implicacao no codigo Fortran:** Cada camada `i` possui dois valores de resistividade `resist(i,1) = rho_h` e `resist(i,2) = rho_v`, e as condutividades sao calculadas como `eta(i,1) = 1/rho_h`, `eta(i,2) = 1/rho_v`.

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

**Convencao de fronteiras no codigo:**
- `prof(0) = -1e300` (sentinela para semi-espaco superior)
- `prof(k) = prof(k-1) + h(k)` para `k = 1, ..., n-1`
- `prof(n) = +1e300` (sentinela para semi-espaco inferior)

A utilizacao de sentinelas `+/-1e300` elimina condicionais nos limites das exponenciais, evitando divisoes por zero e garantindo estabilidade numerica.

### 2.4 Ferramenta Triaxial LWD

A ferramenta simulada e composta por:
- **1 transmissor** com 3 dipolos magneticos ortogonais (Mx, My, Mz)
- **1 receptor** com 3 receptores ortogonais (Hx, Hy, Hz)
- **Espacamento T-R:** `dTR` metros (default: 1.0 m)

O tensor de campo magnetico medido pela ferramenta triaxial completa e:

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

### 2.5 Arranjo T-R e Geometria de Perfilagem

O poco possui inclinacao `theta` em relacao a vertical. A ferramenta percorre o poco com passo `p_med` metros, cobrindo uma janela de investigacao `tj` metros:

```
                         θ (angulo de inclinacao)
                        /
                       /
    z1 = -h1   ◯ T  /  (transmissor, inicio da janela)
                  |/
                  ◯ R  (receptor, dTR metros abaixo do T)
                 /|
                / |
               /  |  pz = p_med * cos(θ)  (passo vertical)
              /   |  px = p_med * sin(θ)  (passo horizontal)
             /    |
    ◯ T+1  /     |
          ◯ R+1  |
         .       |
         .       |
         .       |
    z1 + tj ◯ T_final
              ◯ R_final

  nmed = ceil(tj / pz)  (numero de medicoes por angulo)
```

**Posicoes do Transmissor e Receptor para a j-esima medicao:**

```
Transmissor:
  Tx = 0 + (j-1) * px + Lsen/2
  Tz = z1 + (j-1) * pz + Lcos/2

Receptor:
  Rx = 0 + (j-1) * px - Lsen/2
  Rz = z1 + (j-1) * pz - Lcos/2

Onde:
  Lsen = dTR * sin(theta)
  Lcos = dTR * cos(theta)
  px = p_med * sin(theta)
  pz = p_med * cos(theta)
```

**Nota de configuracao:** O transmissor esta **abaixo** dos receptores, conforme configuracao padrao dos arranjos da Petrobras.

### 2.6 Constantes Fisicas do Simulador

| Constante | Simbolo | Valor | Unidade | Significado |
|:----------|:--------|:------|:--------|:------------|
| Permeabilidade magnetica | mu | 4pi x 10^-7 | H/m | Permeabilidade do vacuo |
| Permissividade eletrica | epsilon | 8.85 x 10^-12 | F/m | Permissividade do vacuo |
| Tolerancia numerica | eps | 10^-9 | - | Limiar para singularidades |
| Tolerancia angular | del | 0.1 | graus | Resolucao angular minima |
| Corrente do dipolo | Iw | 1.0 | A | Corrente do transmissor |
| Momento dipolar | mx, my, mz | 1.0 | A.m^2 | Momento do dipolo magnetico |

---

## 3. Formulacao Matematica Completa

### 3.1 Equacoes de Maxwell em Meios TIV

O campo EM em meios com anisotropia TIV obedece as equacoes de Maxwell no dominio da frequencia (convencao `exp(-i*omega*t)`):

```
∇ × E = i*omega*mu*H           (Lei de Faraday)
∇ × H = sigma_tensor * E + J   (Lei de Ampere)

Onde:
  omega = 2*pi*f  (frequencia angular, rad/s)
  mu = 4*pi*10^-7 H/m  (permeabilidade, assumida isotropica)
  sigma_tensor = diag(sigma_h, sigma_h, sigma_v)  (condutividade TIV)
  J = fonte dipolar (corrente do transmissor)
```

### 3.2 Impeditividade (zeta)

```
zeta = i * omega * mu

No codigo (Fortran):
  zeta = cmplx(0, 1.d0, kind=dp) * omega * mu

Significado fisico: zeta relaciona o campo eletrico ao campo magnetico
nas equacoes de Maxwell. E a impedancia intriseca do meio multiplicada
pelo numero de onda.
```

### 3.3 Numeros de Onda e Constantes de Propagacao

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

**No codigo Fortran (`commonarraysMD`):**

```fortran
kh2(i) = -zeta * eta(i,1)           ! kh^2 = -i*omega*mu*sigma_h
kv2(i) = -zeta * eta(i,2)           ! kv^2 = -i*omega*mu*sigma_v
lamb2(i) = eta(i,1) / eta(i,2)      ! (sigma_h/sigma_v) = lambda^2
u(:,i) = sqrt(kr*kr - kh2(i))       ! constante horiz (TE)
v(:,i) = sqrt(kr*kr - kv2(i))       ! constante vert
s(:,i) = sqrt(lamb2(i)) * v(:,i)    ! lambda * v (TM)
```

### 3.4 Decomposicao TE/TM para Meios TIV

O campo EM em meios estratificados TIV e decomposto em dois modos independentes:

**Modo TE (Transverse Electric):** Componente do campo eletrico tangencial as interfaces.
- Governado pela constante de propagacao `u` (horizontal)
- Associado a admitancia intrinseca `AdmInt = u / zeta`

**Modo TM (Transverse Magnetic):** Componente do campo magnetico tangencial as interfaces.
- Governado pela constante de propagacao `s` (TIV)
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

### 3.5 Coeficientes de Reflexao Recursivos

Os coeficientes de reflexao sao calculados recursivamente das fronteiras mais externas para a camada do transmissor, utilizando a formula de estabilidade numerica baseada em `tanh`:

**Direcao descendente (de cima para baixo):**

```
Admitancia aparente descendente:
  AdmAp_dw(n) = AdmInt(n)    (semi-espaco inferior: sem reflexao)

  AdmAp_dw(i) = AdmInt(i) * [AdmAp_dw(i+1) + AdmInt(i) * tanh(u*h)]
                              / [AdmInt(i) + AdmAp_dw(i+1) * tanh(u*h)]

Coeficiente de reflexao TE descendente:
  RTEdw(n) = 0    (sem reflexao na ultima camada)
  RTEdw(i) = [AdmInt(i) - AdmAp_dw(i+1)] / [AdmInt(i) + AdmAp_dw(i+1)]
```

**Formula `tanh` no codigo (estabilidade numerica):**

```fortran
! Em vez de tanh(x) diretamente, o codigo usa:
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

  AdmAp_up(i) = AdmInt(i) * [AdmAp_up(i-1) + AdmInt(i) * tanh(u*h)]
                              / [AdmInt(i) + AdmAp_up(i-1) * tanh(u*h)]

Coeficiente de reflexao TE ascendente:
  RTEup(1) = 0    (sem reflexao na primeira camada)
  RTEup(i) = [AdmInt(i) - AdmAp_up(i-1)] / [AdmInt(i) + AdmAp_up(i-1)]
```

As mesmas formulas se aplicam ao modo TM substituindo `AdmInt` por `ImpInt`, `u` por `s`, e `uh` por `sh`.

### 3.6 Fatores de Onda Refletida na Camada do Transmissor

A sub-rotina `commonfactorsMD` calcula os fatores de onda refletida especificamente para a camada onde o transmissor esta localizado. Estes fatores encapsulam as multiplas reflexoes dentro da camada fonte:

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

**Nota sobre sinais:** Os fatores `Eu` (modo TE) possuem sinais opostos aos fatores `Mx` (modo TM) nos termos de reflexao. Isso reflete a diferenca de condicoes de contorno entre os modos TE (continuidade de E tangencial) e TM (continuidade de H tangencial) nas interfaces.

### 3.7 Coeficientes de Transmissao entre Camadas

Quando o receptor esta em uma camada diferente do transmissor, os campos sao calculados via coeficientes de transmissao recursivos:

**Receptor abaixo do transmissor (camadR > camadT):**

```
Txdw(T) = mx / (2*s(T))    (termo fonte na camada T)
Tudw(T) = -mx / 2          (termo fonte TE)

Para j = T+1 ate camadR:
  Txdw(j) = s(j-1) * Txdw(j-1) * (...exponenciais...) / [(1 - RTMdw(j)*exp(-2*sh(j))) * s(j)]
  Tudw(j) = u(j-1) * Tudw(j-1) * (...exponenciais...) / [(1 - RTEdw(j)*exp(-2*uh(j))) * u(j)]
```

**Receptor acima do transmissor (camadR < camadT):** Formulas analogas, com direcao ascendente e coeficientes RTMup/RTEup.

### 3.8 Transformada de Hankel via Filtros Digitais

A passagem do dominio espectral (kr) para o dominio espacial (r) e realizada pela transformada de Hankel, implementada via filtro digital:

```
f(r) = integral_0^inf F(kr) * Jn(kr*r) * kr * d(kr)

Onde Jn e a funcao de Bessel de primeira especie de ordem n.

Implementacao via filtro digital (Werthmuller, 201 pontos):

  f(r) ≈ (1/r) * sum_{i=1}^{npt} F(kr_i/r) * w_i

Onde:
  kr_i = abscissas do filtro (tabeladas)
  w_i = pesos do filtro (tabelados, diferentes para J0 e J1)
  npt = 201 pontos (filtro Werthmuller)
```

**Filtros disponiveis no codigo (`filtersv2.f08`):**

| Filtro | Pontos | Referencia | Uso |
|:-------|:------:|:-----------|:----|
| Kong | 61 | Kong (2007) | Rapido, precisao moderada |
| Key | - | Key (2012) | Precisao padrao |
| **Werthmuller** | **201** | **Werthmuller (2006)** | **Usado no simulador** |
| Anderson | 801 | Anderson (1982) | Alta precisao |

O simulador utiliza o filtro Werthmuller de 201 pontos (`npt = 201`), que oferece excelente balanco entre precisao e desempenho.

### 3.9 Campos do Dipolo Magnetico Horizontal (HMD) em Meio TIV

O campo magnetico do HMD e calculado a partir dos kernels espectrais Ktm (modo TM) e Kte (modo TE), convolvidos com funcoes de Bessel J0 e J1:

```
Para o HMD na direcao x (hmdx):

Hx = [(2x^2/r^2 - 1) * sum(Ktedz * wJ1)/r - kh^2*(2y^2/r^2 - 1) * sum(Ktm * wJ1)/r
      - x^2/r^2 * sum(Ktedz * wJ0 * kr) + kh^2*y^2/r^2 * sum(Ktm * wJ0 * kr)] / (2*pi*r)

Hy = xy/r^2 * [sum(Ktedz * wJ1 + kh^2 * Ktm * wJ1)/r
               - sum((Ktedz * wJ0 + kh^2 * Ktm * wJ0) * kr)/2] / (pi*r)

Hz = -x * sum(Kte * wJ1 * kr^2) / (r * 2*pi*r)
```

**Propriedade de simetria do HMDy:** O campo do dipolo y e obtido por rotacao de 90 graus do HMDx:

```
HMDy: x -> y, y -> -x
Hx(hmdy) = Hy(hmdx)
Hy(hmdy) = expressao com (2y^2/r^2 - 1) e x^2/r^2 (permutados)
Hz(hmdy) = -y * sum(Kte * wJ1 * kr^2) / (r * 2*pi*r)
```

O modo `'hmdxy'` calcula ambos simultaneamente, evitando recomputacao dos kernels.

### 3.10 Campos do Dipolo Magnetico Vertical (VMD) em Meio TIV

O VMD excita apenas o modo TE (por simetria axial, nao ha acoplamento TM):

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

O tensor `H` calculado no sistema de coordenadas geologico (x, y, z fixos) deve ser rotacionado para o sistema de coordenadas da ferramenta. A rotacao e definida por tres angulos de Euler (alpha, beta, gamma):

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

## 4. Arquitetura do Software

### 4.1 Estrutura de Modulos

O simulador e composto por 6 arquivos-fonte Fortran organizados em modulos com dependencias hierarquicas:

```
  ┌─────────────────────────────────────────────────────────┐
  │  RunAnisoOmp.f08 (Programa Principal)                   │
  │    ├── Lê model.in                                      │
  │    └── Chama perfila1DanisoOMP(...)                      │
  ├─────────────────────────────────────────────────────────┤
  │  PerfilaAnisoOmp.f08 (Modulo DManisoTIV)                │
  │    ├── perfila1DanisoOMP (loop principal + OpenMP)       │
  │    ├── fieldsinfreqs (campos EM por frequencia)          │
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
  │    ├── commonarraysMD (constantes de propagacao)         │
  │    ├── commonfactorsMD (fatores de onda refletida)       │
  │    ├── RtHR (rotacao tensor)                            │
  │    └── layer2z_inwell, int2str, real2str, etc.          │
  ├─────────────────────────────────────────────────────────┤
  │  filtersv2.f08 (Filtros Digitais)                       │
  │    ├── J0J1Kong (61 pontos)                             │
  │    ├── J0J1Key                                          │
  │    ├── J0J1Wer (201 pontos) ← USADO                    │
  │    └── J0J1And (801 pontos)                             │
  ├─────────────────────────────────────────────────────────┤
  │  parameters.f08 (Constantes Fisicas)                    │
  │    └── dp, pi, mu, epsilon, eps, del, Iw, mx, my, mz   │
  └─────────────────────────────────────────────────────────┘
```

### 4.2 Grafo de Dependencias

```
parameters.f08
      │
      ├──────────────────────┐
      v                      v
filtersv2.f08          magneticdipoles.f08
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

### 4.3 Fluxo de Execucao Completo

```
1. RunAnisoOmp lê model.in
      │
2. Chama perfila1DanisoOMP(...)
      │
      ├── 2a. Calcula nmed(theta) para cada angulo
      ├── 2b. Carrega filtro Werthmuller (201 pontos)
      ├── 2c. Monta geometria (sanitize_hprof_well)
      ├── 2d. Configura OpenMP (nested parallelism)
      │
      ├── 2e. LOOP PARALELO sobre angulos k = 1..ntheta
      │       │
      │       ├── LOOP PARALELO sobre medicoes j = 1..nmed(k)
      │       │       │
      │       │       ├── Calcula posicoes T-R
      │       │       └── Chama fieldsinfreqs(...)
      │       │               │
      │       │               ├── Identifica camadas T, R
      │       │               ├── Para cada frequencia f:
      │       │               │     ├── commonarraysMD (u,s,RTEdw,RTEup,RTMdw,RTMup)
      │       │               │     ├── commonfactorsMD (Mxdw,Mxup,Eudw,Euup)
      │       │               │     ├── hmd_TIV_optimized (HMD x,y)
      │       │               │     ├── vmd_optimized (VMD z)
      │       │               │     ├── Monta tensor H(3x3)
      │       │               │     └── RtHR (rotacao)
      │       │               └── Armazena zrho, cH
      │       └── Coleta resultados do loop j
      └── Coleta resultados do loop k
      │
3. writes_files(...)
      │
      ├── Escreve .out (metadata)
      └── Escreve .dat (binario, stream)
```

### 4.4 Inventario de Linhas de Codigo

| Arquivo | LOC | Proposito |
|:--------|----:|:----------|
| `parameters.f08` | 19 | Constantes fisicas e parametros numericos |
| `filtersv2.f08` | 5559 | Coeficientes tabelados dos filtros digitais |
| `utils.f08` | 383 | Utilidades: geometria, propagacao, reflexao, rotacao |
| `magneticdipoles.f08` | 540 | Campos HMD e VMD em meios TIV |
| `PerfilaAnisoOmp.f08` | 304 | Modulo principal: loop de perfilagem + I/O |
| `RunAnisoOmp.f08` | 54 | Programa principal: leitura de model.in |
| **Total Fortran** | **6859** | |
| `fifthBuildTIVModels.py` | ~900 | Gerador de modelos geologicos |
| `Makefile` | 83 | Sistema de build |
| **Total do projeto** | **~7842** | |

---

## 5. Modulos Fortran — Analise Detalhada

### 5.1 parameters.f08 — Constantes Fisicas

Modulo minimo que define constantes fundamentais com precisao dupla (`dp = kind(1.d0)`):

| Variavel | Valor | Tipo | Descricao |
|:---------|:------|:-----|:----------|
| `dp` | `kind(1.d0)` | `integer, parameter` | Precisao dupla (~15 digitos) |
| `qp` | `selected_real_kind(30)` | `integer, parameter` | Precisao quádrupla (disponivel, nao usado) |
| `pi` | 3.14159265... | `real(dp), parameter` | Pi com 37 digitos decimais |
| `mu` | 4e-7 * pi | `real(dp), parameter` | Permeabilidade do vacuo (H/m) |
| `epsilon` | 8.85e-12 | `real(dp), parameter` | Permissividade do vacuo (F/m) |
| `eps` | 1e-9 | `real(dp), parameter` | Tolerancia numerica para singularidades |
| `del` | 0.1 | `real(dp), parameter` | Tolerancia angular (graus) |
| `Iw` | 1.0 | `real(dp), parameter` | Corrente do dipolo (A) |
| `mx, my, mz` | 1.0 | `real(dp), parameter` | Momentos dipolares (A.m^2) |

### 5.2 filtersv2.f08 — Filtros Digitais para Transformada de Hankel

Modulo que armazena os coeficientes tabelados de 4 filtros digitais para transformadas de Hankel. Cada filtro fornece:
- `absc(npt)`: Abscissas (pontos de amostragem no dominio espectral)
- `wJ0(npt)`: Pesos para convolucao com funcao de Bessel J0
- `wJ1(npt)`: Pesos para convolucao com funcao de Bessel J1

**Filtro utilizado no simulador:** `J0J1Wer` com `npt = 201` (Werthmuller).

A transformada de Hankel via filtro digital e uma tecnica classica em geofisica EM que substitui a integracao numerica direta (quadratura) por uma soma ponderada. Os coeficientes sao pre-calculados offline e tabelados, tornando a avaliacao extremamente eficiente (~201 multiplicacoes por ponto).

### 5.3 utils.f08 — Funcoes Utilitarias

#### 5.3.1 sanitize_hprof_well

Converte espessuras de camadas em arrays de profundidade com sentinelas:

```
Input:  esp(1:n) = [0, h2, h3, ..., h(n-1), 0]
                    ^                         ^
              semi-espaco              semi-espaco
              superior                 inferior

Output: h(1:n) = esp  (espessuras, com h(1)=h(n)=0)
        prof(0:n) = [-1e300, 0, h2, h2+h3, ..., +1e300]
                      ^                           ^
               sentinela                    sentinela
               (evita overflow)             (evita overflow)
```

As sentinelas `+/-1e300` eliminam condicionais nos calculos de exponenciais nas camadas extremas, simplificando o codigo de propagacao.

#### 5.3.2 findlayersTR2well

Identifica em qual camada estao o transmissor (camadT) e o receptor (camadR), dado o array de profundidades das interfaces. A busca e feita de baixo para cima (`do i = n-1, 2, -1`) para eficiencia em cenarios onde T e R estao em camadas profundas.

#### 5.3.3 commonarraysMD — Constantes de Propagacao e Reflexao

Esta e a sub-rotina mais importante do ponto de vista fisico. Calcula, para cada ponto do filtro (`npt = 201`) e cada camada (`n`):

1. Numeros de onda: `kh^2`, `kv^2`
2. Constantes de propagacao: `u`, `v`, `s = lambda*v`
3. Admitancias/impedancias intrinsecas: `AdmInt`, `ImpInt`
4. Produtos espessura × constante: `uh = u*h`, `sh = s*h`
5. Tanh estabilizado: `tghuh`, `tghsh`
6. Coeficientes de reflexao recursivos (TE e TM, ascendentes e descendentes)

**Complexidade computacional:** O(npt * n) por chamada, onde `npt = 201` e `n = numero de camadas`.

#### 5.3.4 commonfactorsMD — Fatores de Onda da Camada Fonte

Calcula os 6 fatores de onda refletida (Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz) para a camada do transmissor. Estes fatores encapsulam as reflexoes multiplas dentro da camada fonte e sao reutilizados para todos os dipolos.

**Otimizacao:** Esta sub-rotina so precisa ser recalculada quando a distancia horizontal `r` entre T e R muda, ou quando a camada do transmissor muda. Em configuracoes coaxiais (r constante), o custo e amortizado.

#### 5.3.5 RtHR — Rotacao do Tensor Magnetico

Implementa a rotacao `R^T * H * R` conforme Liu (2017), equacao 4.80. Recebe os tres angulos de Euler (alpha, beta, gamma) e o tensor H(3x3) complexo, retornando o tensor rotacionado no sistema de coordenadas da ferramenta.

Na perfilagem inclinada, `alpha = theta` (inclinacao do poco), `beta = gamma = 0`.

### 5.4 magneticdipoles.f08 — Campos Dipolares

#### 5.4.1 hmd_TIV_optimized — Dipolo Magnetico Horizontal

Sub-rotina principal para o calculo do campo do HMD em meio TIV. Trata 6 configuracoes geometricas:

| Caso | Condicao | Descricao |
|:-----|:---------|:----------|
| 1 | `camadR == 1 .and. camadT /= 1` | Receptor no semi-espaco superior |
| 2 | `camadR < camadT` | Receptor acima do transmissor (camada intermediaria) |
| 3 | `camadR == camadT .and. z <= h0` | Mesma camada, receptor acima de T |
| 4 | `camadR == camadT .and. z > h0` | Mesma camada, receptor abaixo de T |
| 5 | `camadR > camadT .and. camadR /= n` | Receptor abaixo do transmissor (camada intermediaria) |
| 6 | `camadR == n` | Receptor no semi-espaco inferior |

Para cada caso, os kernels espectrais `Ktm` (modo TM) e `Kte` (modo TE) sao calculados usando os coeficientes de transmissao e reflexao apropriados, e entao convolvidos com os pesos do filtro de Hankel.

**Modos de dipolo suportados:**
- `'hmdx'`: Apenas dipolo horizontal x (saida: Hx(1,1), Hy(1,1), Hz(1,1))
- `'hmdy'`: Apenas dipolo horizontal y
- `'hmdxy'`: Ambos simultaneamente (saida: Hx(1,2), Hy(1,2), Hz(1,2))

O modo `'hmdxy'` e o utilizado no simulador, pois calcula ambos os dipolos compartilhando os kernels.

#### 5.4.2 vmd_optimized — Dipolo Magnetico Vertical

O VMD excita apenas o modo TE (simetria axial). A estrutura e similar ao HMD, com 6 casos geometricos. Os coeficientes de transmissao `TEdwz`/`TEupz` sao calculados recursivamente.

### 5.5 PerfilaAnisoOmp.f08 — Modulo Principal

#### 5.5.1 perfila1DanisoOMP — Loop de Perfilagem

Funcao principal do simulador. Etapas:

1. **Calculo de nmed:** Numero de medicoes por angulo baseado na janela `tj` e passo `p_med`
2. **Carregamento do filtro:** `J0J1Wer(201, ...)` — filtro de Werthmuller
3. **Preparacao da geometria:** `sanitize_hprof_well(n, esp, h, prof)`
4. **Configuracao OpenMP:** Paralelismo aninhado com threads dinamicos
5. **Loop duplo paralelo:**
   - Loop externo: angulos (`k = 1..ntheta`), `num_threads_k = ntheta`
   - Loop interno: medicoes (`j = 1..nmed(k)`), `num_threads_j = maxthreads - ntheta`
6. **Coleta e escrita:** `writes_files(...)` escreve .dat e .out

#### 5.5.2 fieldsinfreqs — Campos em Todas as Frequencias

Para cada posicao T-R, calcula o tensor H completo em todas as frequencias. Cadeia de chamadas:

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

#### 5.5.3 writes_files — Escrita de Saida

**Arquivo .out (metadados):** Escrito apenas quando `modelm == nmaxmodel` (ultimo modelo):

```
Linha 1: nt nf nmaxmodel     (num. angulos, num. frequencias, num. modelos)
Linha 2: theta(1) ... theta(nt)   (angulos)
Linha 3: freq(1) ... freq(nf)    (frequencias)
Linha 4: nmeds(1) ... nmeds(nt)  (num. medicoes por angulo)
```

**Arquivo .dat (dados binarios):** Escrito em modo `stream` (sem registros fixos), `append`:

```
Para cada angulo k, frequencia j, medicao i:
  write(1000) i, zobs, rho_h, rho_v, Re(H11), Im(H11), ..., Re(H33), Im(H33)
              ^    ^      ^      ^     ^                                    ^
           int32  real64  real64  real64   9 × (2 × real64 = 16 bytes)
           4 bytes 8 bytes 8 bytes 8 bytes    = 144 bytes
                                                                    
  Total por registro: 4 + 3×8 + 18×8 = 4 + 24 + 144 = 172 bytes
  Formato: 1 int32 + 21 float64 = 22 valores por registro
```

### 5.6 RunAnisoOmp.f08 — Programa Principal

Programa sequencial que:
1. Obtem o diretorio corrente via `getcwd`
2. Le o arquivo `model.in` sequencialmente
3. Aloca arrays de resistividade e espessura
4. Chama `perfila1DanisoOMP(...)` com todos os parametros

---

## 6. Arquivo de Entrada model.in

### 6.1 Estrutura Completa

O arquivo `model.in` e lido sequencialmente pelo programa principal. Cada linha contem um ou mais valores:

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
  +1   p_med              real(dp)  Passo entre medicoes (m)
  +1   dTR                real(dp)  Distancia Transmissor-Receptor (m)
  +1   filename           char      Nome base dos arquivos de saida
  +1   ncam               integer   Numero total de camadas (incluindo semi-espacos)
  +2   resist(1,1:2)      real(dp)  rho_h, rho_v da camada 1 (semi-espaco superior)
  ...  resist(ncam,1:2)   real(dp)  rho_h, rho_v da camada ncam
  +1   esp(2)             real(dp)  Espessura da camada 2 (m)
  ...  esp(ncam-1)        real(dp)  Espessura da camada ncam-1 (m)
  +1   modelm nmaxmodel   integer   Modelo atual e total de modelos
```

### 6.2 Exemplo Comentado (model.in atual)

```
2                    ! nf = 2 frequencias
20000.0              ! freq(1) = 20 kHz
40000.0              ! freq(2) = 40 kHz
1                    ! ntheta = 1 angulo
0.0                  ! theta(1) = 0 graus (poco vertical)
10.0                 ! h1 = 10 m acima da 1a interface
120.0                ! tj = 120 m de janela de investigacao
0.2                  ! p_med = 0.2 m entre medicoes
1.0                  ! dTR = 1.0 m de espacamento T-R
Inv0_15Dip1000_t5    ! nome dos arquivos de saida
10                   ! ncam = 10 camadas
1.38    1.38         ! Camada 1 (semi-espaco): rho_h=1.38, rho_v=1.38
1.76    1.76         ! Camada 2
107.07  107.08       ! Camada 3 (alta resistividade — possivel reservatorio)
102.19  102.2        ! Camada 4
1474.11 1474.26      ! Camada 5 (resistividade muito alta — possivel sal/carbonato)
2.58    2.58         ! Camada 6
9.48    9.48         ! Camada 7
0.35    0.35         ! Camada 8 (baixissima — agua salgada)
582.74  582.82       ! Camada 9
0.3     0.3          ! Camada 10 (semi-espaco inferior)
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

### 6.3 Observacoes Importantes

1. **Espessuras:** As camadas 1 e ncam sao semi-espacos (espessura infinita), entao `esp(1) = esp(ncam) = 0` e atribuido automaticamente no codigo.

2. **Frequencias:** O simulador padrao usa 2 frequencias (20 kHz e 40 kHz). A primeira e a mesma do pipeline (`FREQUENCY_HZ = 20000.0`).

3. **Numero de medicoes:** Para `theta = 0`, `pz = p_med = 0.2`, `nmed = ceil(120/0.2) = 600`, consistente com `SEQUENCE_LENGTH = 600` do pipeline.

4. **Anisotropia neste exemplo:** O modelo mostra anisotropia muito fraca (`lambda ~ 1.0` em quase todas as camadas). Em cenarios reais, `lambda` varia de 1.0 a sqrt(2).

---

## 7. Arquivos de Saida (.dat e .out)

### 7.1 Arquivo .out — Metadados

Arquivo texto com 4 linhas, escrito apenas pelo ultimo modelo da serie:

```
Formato:
  Linha 1: nt  nf  nmaxmodel
  Linha 2: theta(1)  theta(2) ... theta(nt)
  Linha 3: freq(1)  freq(2)  ... freq(nf)
  Linha 4: nmeds(1)  nmeds(2) ... nmeds(nt)
```

**Exemplos de .out existentes:**

| Arquivo | nt | nf | nmodels | Angulos | Frequencias | nmeds |
|:--------|:--:|:--:|:-------:|:--------|:------------|:------|
| t2 | 2 | 1 | 1000 | 0, 15 | 20000 | 600, 622 |
| t4 | 2 | 2 | 1000 | 0, 15 | 20000, 40000 | 600, 622 |
| t5 | 1 | 2 | 1000 | 0 | 20000, 40000 | 600 |

### 7.2 Arquivo .dat — Dados Binarios

Arquivo binario no formato Fortran `stream` (sem registros fixos), escrito em modo `append`.

**Formato por registro:**

| Posicao | Variavel | Tipo | Bytes | Descricao |
|:--------|:---------|:-----|------:|:----------|
| 0 | i | int32 | 4 | Indice da medicao |
| 4 | zobs | float64 | 8 | Profundidade do ponto-medio T-R (m) |
| 12 | rho_h | float64 | 8 | Resistividade horizontal verdadeira (Ohm.m) |
| 20 | rho_v | float64 | 8 | Resistividade vertical verdadeira (Ohm.m) |
| 28 | Re(Hxx) | float64 | 8 | Parte real de H(1,1) |
| 36 | Im(Hxx) | float64 | 8 | Parte imaginaria de H(1,1) |
| 44 | Re(Hxy) | float64 | 8 | Parte real de H(1,2) |
| 52 | Im(Hxy) | float64 | 8 | Parte imaginaria de H(1,2) |
| 60 | Re(Hxz) | float64 | 8 | Parte real de H(1,3) |
| 68 | Im(Hxz) | float64 | 8 | Parte imaginaria de H(1,3) |
| 76 | Re(Hyx) | float64 | 8 | Parte real de H(2,1) |
| 84 | Im(Hyx) | float64 | 8 | Parte imaginaria de H(2,1) |
| 92 | Re(Hyy) | float64 | 8 | Parte real de H(2,2) |
| 100 | Im(Hyy) | float64 | 8 | Parte imaginaria de H(2,2) |
| 108 | Re(Hyz) | float64 | 8 | Parte real de H(2,3) |
| 116 | Im(Hyz) | float64 | 8 | Parte imaginaria de H(2,3) |
| 124 | Re(Hzx) | float64 | 8 | Parte real de H(3,1) |
| 132 | Im(Hzx) | float64 | 8 | Parte imaginaria de H(3,1) |
| 140 | Re(Hzy) | float64 | 8 | Parte real de H(3,2) |
| 148 | Im(Hzy) | float64 | 8 | Parte imaginaria de H(3,2) |
| 156 | Re(Hzz) | float64 | 8 | Parte real de H(3,3) |
| 164 | Im(Hzz) | float64 | 8 | Parte imaginaria de H(3,3) |

**Total por registro:** 4 + 21 x 8 = **172 bytes**

### 7.3 Leitura no Pipeline Python (geosteering_ai)

O pipeline Python (`geosteering_ai/data/loading.py`) le o .dat usando numpy:

```python
# Formato de leitura do .dat:
dtype = np.dtype([
    ('meds', np.int32),       # Col 0: indice de medicao
    ('values', np.float64, 21) # Cols 1-21: zobs, rho_h, rho_v, 18 EM
])
data = np.fromfile(filepath, dtype=dtype)

# Reorganizacao em 22 colunas:
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

Para 1000 modelos, 1 angulo, 2 frequencias, 600 medicoes:

```
Tamanho = nmodels × ntheta × nfreq × nmeds × 172 bytes
        = 1000 × 1 × 2 × 600 × 172
        = 206,400,000 bytes ≈ 197 MB
```

---

## 8. Sistema de Build (Makefile)

### 8.1 Estrutura do Makefile

O Makefile segue a convencao padrao de projetos Fortran com:

| Secao | Descricao |
|:------|:----------|
| SETTINGS | Binario (`tatu.x`), extensoes, diretorio de build (`./build`) |
| DEPENDENCIES | Ordem de compilacao: `parameters → filtersv2 → utils → magneticdipoles → PerfilaAnisoOmp → RunAnisoOmp` |
| COMPILER | `gfortran` com flags de producao |
| TARGETS | `$(binary)`, `run_python`, `all`, `clean`, `cleanall` |

### 8.2 Flags de Compilacao

**Flags de producao (ativas):**

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
-funroll-loops # Desenrola loops para otimizacao
-fall-intrinsics # Habilita todas as funcoes intrinsecas
```

**Flags OpenMP:** `-fopenmp` adicionada tanto na compilacao quanto na linkagem.

### 8.3 Targets

| Target | Descricao |
|:-------|:----------|
| `$(binary)` (`tatu.x`) | Compila e linka todos os .o em um executavel |
| `run_python` | Executa `fifthBuildTIVModels.py` (gerador de modelos) |
| `all` | Compila + executa Python |
| `clean` | Remove `./build/` |
| `cleanall` | Remove `./build/`, `tatu.x`, `*.dat`, `*.out` |

### 8.4 Ordem de Compilacao

```
parameters.f08 ──> build/parameters.o
filtersv2.f08  ──> build/filtersv2.o
utils.f08      ──> build/utils.o      (depende de parameters)
magneticdipoles.f08 ──> build/magneticdipoles.o (depende de parameters)
PerfilaAnisoOmp.f08 ──> build/PerfilaAnisoOmp.o (depende de todos acima + omp_lib)
RunAnisoOmp.f08 ──> build/RunAnisoOmp.o (depende de parameters, DManisoTIV)

Linkagem: gfortran -fopenmp [flags] -o tatu.x build/*.o
```

### 8.5 Notas e Recomendacoes

- **`-ffast-math`** pode afetar a precisao de operacoes com NaN e infinitos. Para validacao numerica, recomenda-se compilar sem esta flag.
- **`-march=native`** otimiza para a CPU local, mas o binario nao e portavel.
- O diretorio `./build` e criado automaticamente via `$(shell mkdir -p $(build))`.

---

## 9. Gerador de Modelos Geologicos (Python)

### 9.1 Visao Geral

O script `fifthBuildTIVModels.py` gera modelos geologicos aleatorios usando amostragem Sobol Quasi-Monte Carlo (`scipy.stats.qmc.Sobol`). Os modelos sao escritos em `model.in` e simulados pelo Fortran em sequencia.

### 9.2 Parametros dos Modelos Geologicos

| Parametro | Range | Distribuicao | Descricao |
|:----------|:------|:------------|:----------|
| `n_layers` | 3-80 | Empirica (ponderada) ou uniforme | Numero de camadas |
| `rho_h` | 0.05-1500 Ohm.m | Log-uniforme (Sobol) | Resistividade horizontal |
| `lambda` | 1.0-sqrt(2) | Uniforme (Sobol), correlacionada com rho_h | Coeficiente de anisotropia |
| `rho_v` | calculado | `lambda^2 * rho_h` | Resistividade vertical |
| `espessuras` | 0.1-50+ m | Sobol + stick-breaking | Espessuras das camadas internas |

### 9.3 Cenarios de Geracao (6 Geradores)

| Cenario | Funcao | nmodels | ncam | Espessuras | Contrastes | Ruido |
|:--------|:-------|:-------:|:----:|:-----------|:-----------|:------|
| **Baseline empirico** | `baseline_empirical_2` | 18000 | 3-30 (pesos empiricos) | Standard (min 1.0 m) | Natural | Nao |
| **Baseline uniforme** | `baseline_ncamuniform_2` | 9000 | 3-80 (uniforme) | Standard (min 0.5 m) | Natural | Nao |
| **Camadas grossas** | `baseline_thick_thicknesses_2` | 9000 | 3-14 | Grossas (min 10 m, p=0.7) | Forcados em grossas | Nao |
| **Desfavoravel empirico** | `unfriendly_empirical_2` | 12000 | 3-30 (pesos) | Finas (min 0.2 m, p=0.6) | Forcados (5x, p=0.5) | Nao |
| **Desfavoravel ruidoso** | `unfriendly_noisy_2` | 12000 | 3-30 (pesos) | Finas + ruido (3%) | Forcados + ruido (5%) | Sim |
| **Patologico** | `generate_pathological_models_2` | 4500 | 3-28 | Muito finas (min 0.1 m) | Extremos (10x, p=0.7) | Sim (7%) |

### 9.4 Funcoes Auxiliares

| Funcao | Descricao |
|:-------|:----------|
| `log_transform_2` | Transforma Sobol [0,1] para escala log-uniforme [min, max] |
| `uniform_transform_2` | Transforma Sobol [0,1] para escala linear [min, max] |
| `generate_thicknesses_2` | Sobol stick-breaking com espessura minima garantida |
| `generate_thick_thicknesses_2` | Forcamento de camadas grossas (>15 m) com probabilidade `p_thick` |
| `generate_thin_thicknesses_2` | Forcamento de camadas finas (~min) com probabilidade `p_thin` |
| `conditional_rho_h_sampling_core_2` | Forca contrastes entre camadas adjacentes |
| `conditional_rho_h_with_thickness_2` | Forca resistividades extremas em camadas grossas |
| `correlated_lambda_sampling_core_2` | Lambda correlacionado com resistividade (rho alto → lambda alto) |

### 9.5 Fluxo do Gerador

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
  9. Escreve model.in
  10. Executa tatu.x via subprocess
  11. Resultado: append no .dat
```

---

## 10. Paralelismo OpenMP — Analise e Otimizacao

### 10.1 Estrutura Atual de Paralelismo

O simulador utiliza **paralelismo OpenMP aninhado em 2 niveis**:

```
Nivel 1 (externo): Angulos theta
  !$omp parallel do schedule(dynamic) num_threads(num_threads_k)
  do k = 1, ntheta   ← tipicamente 1-2 threads

    Nivel 2 (interno): Medicoes por angulo
    !$omp parallel do schedule(dynamic) num_threads(num_threads_j)
    do j = 1, nmed(k)   ← 600+ threads disponiveis
      ...
    end do
    !$omp end parallel do

  end do
  !$omp end parallel do
```

**Distribuicao de threads:**

```fortran
maxthreads = omp_get_max_threads()      ! Total de threads disponíveis
num_threads_k = ntheta                  ! Threads para angulos (1-2)
num_threads_j = maxthreads - ntheta     ! Restante para medicoes
```

### 10.2 Analise de Desempenho

**Problemas identificados:**

1. **Desbalanceamento de carga no nivel externo:** Com `ntheta = 1` (comum), o loop externo usa apenas 1 thread, desperdicando o segundo nivel aninhado.

2. **Alocacao de memoria dentro do loop paralelo:** Cada chamada a `commonarraysMD` e `hmd_TIV_optimized` aloca arrays temporarios. Isso causa:
   - Contencao no alocador de memoria (malloc)
   - Fragmentacao de heap
   - Overhead de alocacao/desalocacao repetida

3. **Escalonamento `dynamic` para loop regular:** Se todas as medicoes tem custo computacional similar (mesmo modelo), `static` seria mais eficiente (sem overhead de despacho).

4. **Redundancia computacional:** `commonarraysMD` calcula constantes de propagacao e coeficientes de reflexao que dependem apenas do modelo (n, resist, freq), nao da posicao T-R. Em medicoes onde T e R estao na mesma camada (maioria dos casos em pocos verticais), `commonarraysMD` produz o mesmo resultado para todos os `j`.

### 10.3 Pontos de Otimizacao — Memoria

| Otimizacao | Impacto | Complexidade |
|:-----------|:--------|:-------------|
| **Pre-alocar arrays de trabalho por thread** | Alto — elimina malloc dentro do loop | Media |
| **Mover `commonarraysMD` para fora do loop j** | Alto — computa uma vez por (angulo, freq) quando r e constante | Media |
| **Usar `firstprivate` para arrays somente-leitura** | Medio — evita copias desnecessarias | Baixa |
| **Reduzir arrays temporarios em hmd_TIV** | Medio — Tudw/Txdw/etc. alocados com tamanho minimo | Media |
| **Pool de memoria por thread** | Alto — elimina fragmentacao | Alta |

### 10.4 Pontos de Otimizacao — Tempo de Execucao

| Otimizacao | Impacto Estimado | Descricao |
|:-----------|:----------------|:----------|
| **Cache de `commonarraysMD`** | 30-50% reducao | Computa uma vez por (camadT, freq), reutiliza para todos os j com mesmo camadT |
| **Paralelismo sobre frequencias** | 2x para nf=2 | Adicionar nivel de paralelismo sobre frequencias |
| **Colapso de loops** | 10-20% | `!$omp parallel do collapse(2)` para angulos × medicoes |
| **SIMD para convolucao Hankel** | 10-30% | `!$omp simd` nos sums de kernels × pesos do filtro |
| **Scheduler hibrido** | 5-10% | `static` quando nmed uniforme, `dynamic` quando variavel |
| **Batch de medicoes** | 15-25% | Agrupar medicoes por camadT para reutilizar coeficientes de reflexao |

### 10.5 Proposta de Paralelismo Otimizado

```fortran
! PROPOSTA: Colapso de loops + pre-alocacao + cache de coeficientes
!
! 1. Pre-alocar workspace por thread:
!$omp parallel private(tid, ws)
tid = omp_get_thread_num()
ws = workspace_pool(tid)  ! Arrays pre-alocados

! 2. Loop colapsado (angulos × medicoes):
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
! Identificar camadT para cada posicao j (depende apenas de Tz)
! Agrupar por camadT e computar commonarraysMD uma vez por grupo
```

### 10.6 Metricas de Escalabilidade

Para 1000 modelos, 1 angulo, 2 frequencias, 600 medicoes:

```
Computacoes por modelo:
  = ntheta × nfreq × nmed × (commonarraysMD + commonfactorsMD + hmd + vmd + rotacao)
  = 1 × 2 × 600 × (~201*n operacoes complexas)
  = 1200 × (~201*10 = 2010 FLOPs complexos)
  ≈ 2.4 × 10^6 operacoes complexas por modelo

Total para 1000 modelos: ≈ 2.4 × 10^9 operacoes
Tempo estimado (single core, ~1 GFLOP/s complex): ~2.4 segundos por modelo
```

---

## 11. Analise de Viabilidade CUDA (GPU)

### 11.1 Identificacao de Kernels Paralelizaveis

| Kernel | Dimensao | Paralelismo | Adequacao GPU |
|:-------|:---------|:------------|:-------------|
| `commonarraysMD` | npt × n | Independente por ponto do filtro | **Alta** — 201 threads independentes |
| `commonfactorsMD` | npt | Independente por ponto do filtro | **Alta** — 201 threads |
| Convolucao Hankel (sum) | npt | Reducao paralela | **Alta** — reducao classica |
| Loop medicoes | nmed | Independente por medicao | **Alta** — 600+ threads |
| Loop frequencias | nf | Independente por frequencia | **Media** — apenas 2 frequencias |
| Recursao reflexao | n (sequencial) | **Dependencia de dados** | **Baixa** — n camadas sequenciais |

### 11.2 Estrategia de Implementacao CUDA

```
GPU Grid:
  Block dimension: (npt=201 threads, 1, 1) — um thread por ponto do filtro
  Grid dimension:  (nmed, nf, ntheta) — uma thread-block por medicao/freq/angulo

Kernel 1: compute_propagation_constants
  Input: kr(201), eta(n,2), zeta, h(n)
  Output: u(201,n), s(201,n), uh(201,n), sh(201,n)
  Paralelismo: cada thread computa para um kr_i

Kernel 2: compute_reflection_coefficients
  Input: u, s, uh, sh, AdmInt, ImpInt
  Output: RTEdw, RTEup, RTMdw, RTMup
  ATENCAO: recursao sequencial sobre camadas (n)
  Estrategia: cada thread processa um kr_i independentemente,
              mas faz a recursao sobre n camadas sequencialmente

Kernel 3: compute_wave_factors
  Input: RTEdw, RTEup, RTMdw, RTMup, camadT, h0
  Output: Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz
  Paralelismo: independente por kr_i

Kernel 4: compute_hmd_vmd
  Input: todos os fatores + geometria
  Output: Hx, Hy, Hz para HMD e VMD
  Inclui reducao paralela para convolucao Hankel

Kernel 5: assemble_tensor_and_rotate
  Input: H_hmd, H_vmd, angulos
  Output: tensor H(3,3) rotacionado
  Paralelismo: independente por medicao
```

### 11.3 Desafios CUDA

1. **Recursao dos coeficientes de reflexao:** Os coeficientes sao calculados de forma recursiva (camada n para 1, e 1 para n). Cada passo depende do anterior, impedindo paralelismo nesta dimensao. Porem, os 201 pontos do filtro sao independentes, permitindo 201 threads executando a recursao simultaneamente.

2. **Alocacao dinamica:** O codigo Fortran aloca arrays de tamanho variavel (`Tudw`, `Txdw`) dependendo da posicao relativa T-R. Na GPU, estes arrays devem ser pre-alocados com tamanho maximo.

3. **Divergencia de branch:** Os 6 casos geometricos em `hmd_TIV_optimized` causam divergencia de warp em GPUs NVIDIA. Estrategia: agrupar medicoes por caso geometrico para minimizar divergencia.

4. **Memoria:** Para n=80 camadas, npt=201: arrays de tamanho (201, 80) × complex64 = 201 × 80 × 16 bytes = ~257 KB por medicao. Caberiam ~800 medicoes simultaneas em 200 MB de memoria global.

### 11.4 Estimativa de Speedup

| Componente | CPU (1 core) | GPU (estimado) | Speedup |
|:-----------|:-------------|:---------------|:--------|
| `commonarraysMD` | ~0.5 ms/call | ~0.01 ms/call | 50x |
| Convolucao Hankel | ~0.2 ms/sum | ~0.002 ms/sum | 100x |
| Loop medicoes (600) | 600 × serial | 600 × paralelo | ~100x |
| Overhead transferencia | 0 | ~1 ms/modelo | - |
| **Total por modelo** | ~400 ms | ~10 ms | **~40x** |

### 11.5 Frameworks CUDA Recomendados

| Framework | Linguagem | Vantagem | Desvantagem |
|:----------|:----------|:---------|:------------|
| **CUDA Fortran (PGI/NVHPC)** | Fortran | Reuso maximo do codigo | Compilador proprietario |
| **CUDA C/C++** | C/C++ | Ecossistema maduro, cuBLAS | Reescrita completa |
| **OpenACC** | Fortran | Minimo de alteracao no codigo | Desempenho menor que CUDA nativo |
| **hipSYCL** | C++ | Portabilidade AMD/NVIDIA | Complexidade adicional |

**Recomendacao:** Para prototipagem rapida, **OpenACC** com `nvfortran` (NVIDIA HPC SDK). Para producao, **CUDA C** com wrappers Fortran.

---

## 12. Analise de Reimplementacao em Python Otimizado

### 12.1 Motivacao

Uma versao Python do simulador permitiria:
- Integracao direta com o pipeline `geosteering_ai` (sem subprocess)
- Geracao de dados on-the-fly durante o treinamento
- Eliminacao da dependencia de compilacao Fortran
- Facilidade de extensao (novos modelos de ferramenta, multi-frequencia)
- Potencial implementacao de backpropagation through physics (PINNs)

### 12.2 Bibliotecas para Computacao de Alto Desempenho em Python

| Biblioteca | Tipo | Paralelismo | Adequacao |
|:-----------|:-----|:------------|:----------|
| **NumPy** | CPU, vetorial | Multi-threaded (BLAS) | Baseline — boa para prototipagem |
| **Numba** | CPU, JIT | Multi-threaded, SIMD | **Excelente** — compilacao JIT com tipagem |
| **Numba CUDA** | GPU | CUDA kernels | **Alta** — acesso direto a GPU |
| **CuPy** | GPU, vetorial | CUDA automatico | **Alta** — drop-in NumPy para GPU |
| **JAX** | CPU/GPU/TPU | XLA compilation | **Alta** — gradientes automaticos |
| **TensorFlow** | CPU/GPU/TPU | XLA compilation | **Media** — overhead de framework |
| **PyTorch** | CPU/GPU | CUDA | **PROIBIDO** no projeto |
| **empymod** | CPU | Numba JIT | **Referencia** — simulador EM 1D existente |

### 12.3 Estrategia de Implementacao Python

#### 12.3.1 Fase 1: Prototipo NumPy Vetorizado

```python
# Vetorizacao do loop de medicoes:
# Em vez de iterar sobre j=1..nmed, computar todas as medicoes de uma vez

# Posicoes T-R (vetorizado para todas as medicoes):
j = np.arange(nmed)
Tz = z1 + j * pz + Lcos / 2    # shape: (nmed,)
Rz = z1 + j * pz - Lcos / 2    # shape: (nmed,)

# commonarraysMD vetorizado:
# kr: (npt,), u: (npt, n), => broadcast automatico
# Resultado: arrays de shape (npt, n) para cada frequencia
```

#### 12.3.2 Fase 2: Compilacao JIT com Numba

```python
@numba.njit(parallel=True, cache=True)
def compute_reflection_coefficients(kr, eta, zeta, h, n, npt):
    """Calcula coeficientes de reflexao recursivos (Numba JIT)."""
    u = np.empty((npt, n), dtype=np.complex128)
    s = np.empty((npt, n), dtype=np.complex128)
    RTEdw = np.empty((npt, n), dtype=np.complex128)
    RTMdw = np.empty((npt, n), dtype=np.complex128)
    
    for i in numba.prange(npt):  # Paralelo sobre pontos do filtro
        for j in range(n):
            kh2 = -zeta * eta[j, 0]
            u[i, j] = np.sqrt(kr[i]**2 - kh2)
            # ... recursao sequencial sobre camadas
    
    return u, s, RTEdw, RTMdw
```

#### 12.3.3 Fase 3: GPU via Numba CUDA ou CuPy

```python
@numba.cuda.jit
def kernel_commonarraysMD(kr, eta, zeta, h, n, u, s, RTEdw, RTMdw):
    """CUDA kernel para constantes de propagacao."""
    i = numba.cuda.grid(1)  # Thread index = ponto do filtro
    if i < npt:
        for j in range(n):
            kh2 = -zeta * eta[j, 0]
            u[i, j] = cmath.sqrt(kr[i]**2 - kh2)
            # ... recursao sobre camadas
```

### 12.4 Referencia: empymod

O pacote Python `empymod` (Werthmuller, 2017) ja implementa modelagem EM 1D/3D para meios anisotropicos usando Numba. Principais caracteristicas:

- Transformadas de Hankel via filtros digitais (mesmos filtros do simulador Fortran)
- Suporte a meios TIV e anisotropia geral
- Compilacao JIT via Numba
- Interface NumPy-compativel

**Limitacoes do empymod para o projeto:**
- Nao suporta diretamente o arranjo triaxial com rotacao
- Nao otimizado para loop de perfilagem (muitas posicoes T-R)
- Nao integrado com TensorFlow para backpropagation

**Recomendacao:** Usar empymod como referencia e validacao, mas implementar versao customizada otimizada para o caso de uso especifico do pipeline.

### 12.5 Estimativa de Desempenho Python

| Implementacao | Tempo/modelo (est.) | vs Fortran | Notas |
|:-------------|:-------------------|:-----------|:------|
| Python puro | ~60 s | 150x mais lento | Inviavel |
| NumPy vetorizado | ~4 s | 10x mais lento | Aceitavel para prototipo |
| **Numba CPU (JIT)** | **~0.5 s** | **~1.2x mais lento** | **Viável para produção** |
| **Numba CUDA** | **~0.02 s** | **~20x mais rápido** | **Ideal para treinamento** |
| CuPy | ~0.1 s | ~4x mais rapido | Bom com batching |
| JAX (XLA) | ~0.05 s | ~8x mais rapido | Permite gradientes |

### 12.6 Roteiro de Implementacao Python

```
Fase 1 (Prototipo — 2-3 semanas):
  ├── Implementar commonarraysMD em NumPy vetorizado
  ├── Implementar hmd_TIV e vmd em NumPy
  ├── Validar contra Fortran (erro relativo < 1e-10)
  └── Benchmark: tempo/modelo NumPy vs Fortran

Fase 2 (Otimizacao CPU — 2-3 semanas):
  ├── Reescrever kernels criticos com Numba @njit
  ├── Paralelismo sobre pontos do filtro (prange)
  ├── Cache de coeficientes de reflexao por modelo
  └── Benchmark: tempo/modelo Numba vs Fortran

Fase 3 (GPU — 3-4 semanas):
  ├── CUDA kernels para commonarraysMD + Hankel
  ├── Batching de multiplos modelos na GPU
  ├── Integracao com pipeline tf.data
  └── Benchmark: throughput (modelos/segundo)

Fase 4 (Integracao — 2 semanas):
  ├── Modulo geosteering_ai/simulation/forward.py
  ├── Interface com PipelineConfig
  ├── Modo surrogado: rede neural substitui simulador
  └── Testes de integracao com pipeline completo
```

### 12.7 Arquitetura Proposta para o Modulo Python

```
geosteering_ai/simulation/
├── __init__.py
├── forward.py        ← ForwardSimulator: interface principal
├── propagation.py    ← Constantes de propagacao, reflexao (Numba)
├── dipoles.py        ← HMD e VMD para meios TIV (Numba)
├── hankel.py         ← Transformada de Hankel via filtros (Numba)
├── rotation.py       ← Rotacao do tensor (Numba)
├── filters.py        ← Coeficientes tabelados (Werthmuller, etc.)
├── geometry.py       ← Geometria do poco e arranjo T-R
├── cuda_kernels.py   ← Kernels CUDA (Numba CUDA) [opcional]
└── validation.py     ← Comparacao com Fortran
```

---

## 13. Integracao com o Pipeline Geosteering AI v2.0

### 13.1 Cadeia Atual (Fortran)

```
fifthBuildTIVModels.py → model.in → tatu.x (Fortran) → .dat → geosteering_ai/data/loading.py
                              |          |                 |
                        Parametros    Simulacao        Leitura binaria
                        geologicos    EM 1D TIV        22 colunas
```

### 13.2 Cadeia Futura (Python Integrado)

```
geosteering_ai/simulation/forward.py → geosteering_ai/data/pipeline.py
                |                              |
          Simulacao EM 1D TIV           Split → Noise → FV → GS → Scale
          (Numba/CUDA, on-the-fly)              |
                                          tf.data.Dataset
```

### 13.3 Beneficios da Integracao

1. **Eliminacao de I/O:** Dados gerados in-memory, sem escrita/leitura de .dat
2. **Geracao on-the-fly:** Novos modelos geologicos gerados durante treinamento
3. **Data augmentation fisica:** Perturbacao de parametros geologicos (espessuras, resistividades) como augmentation
4. **PINNs:** Backpropagation through o simulador para constraintes fisicas
5. **Reproducibilidade:** Seed unico controla toda a cadeia (geracao + simulacao + treinamento)

### 13.4 Correspondencia Fortran → Pipeline v2.0

| Parametro Fortran | Config v2.0 | Valor Default | Descricao |
|:------------------|:------------|:-------------|:----------|
| `freq(1)` | `config.frequency_hz` | 20000.0 | Frequencia principal (Hz) |
| `dTR` | `config.spacing_meters` | 1.0 | Espacamento T-R (m) |
| `nmed(1)` | `config.sequence_length` | 600 | Numero de medicoes |
| `resist(:,1)` | `targets[:, 0]` (col 2) | - | rho_h (Ohm.m) |
| `resist(:,2)` | `targets[:, 1]` (col 3) | - | rho_v (Ohm.m) |
| `cH(f,1:9)` | `features[:, 4:21]` (cols 4-21) | - | Tensor H (18 valores reais) |
| `zobs` | `features[:, 1]` (col 1) | - | Profundidade (m) |

---

## 14. Referencias Bibliograficas

### 14.1 Modelagem EM em Meios Estratificados

1. **Anderson, W.L.** (1982). "Fast Hankel Transforms Using Related and Lagged Convolutions". *ACM Transactions on Mathematical Software*, 8(4), 344-368. — Filtro de 801 pontos para transformada de Hankel.

2. **Werthmuller, D.** (2006). "EMMOD — Electromagnetic Modelling". *Report, TU Delft*. — Filtros digitais de 201 pontos para J0/J1 utilizados no simulador.

3. **Kong, F.N.** (2007). "Hankel transform filters for dipole antenna radiation in a conductive medium". *Geophysical Prospecting*, 55(1), 83-89. — Filtro de 61 pontos.

4. **Key, K.** (2012). "Is the fast Hankel transform faster than quadrature?". *Geophysics*, 77(3), F21-F30. — Comparacao de desempenho de filtros.

### 14.2 Anisotropia TIV em Perfilagem de Pocos

5. **Anderson, B., Barber, T., & Habashy, T.** (2002). "The Interpretation of Multicomponent Induction Logs in the Presence of Dipping, Anisotropic Formations". *SPWLA 43rd Annual Logging Symposium*. — Resposta de ferramentas triaxiais em meios TIV.

6. **Liu, C.** (2017). *Theory of Electromagnetic Well Logging*. Elsevier. — Referencia principal para a rotacao do tensor (eq. 4.80).

### 14.3 Decomposicao TE/TM e Coeficientes de Reflexao

7. **Chew, W.C.** (1995). *Waves and Fields in Inhomogeneous Media*. IEEE Press. — Formulacao TE/TM para meios estratificados com anisotropia.

8. **Ward, S.H. & Hohmann, G.W.** (1988). "Electromagnetic Theory for Geophysical Applications". In *Electromagnetic Methods in Applied Geophysics*, Vol. 1, SEG. — Fundamentacao teorica das equacoes de Maxwell em geofisica.

### 14.4 Dipolos Magneticos em Meios TIV

9. **Zhong, L., Li, J., Bhardwaj, A., Shen, L.C., & Liu, R.C.** (2008). "Computation of Triaxial Induction Logging Tools in Layered Anisotropic Dipping Formations". *IEEE Transactions on Geoscience and Remote Sensing*, 46(4), 1148-1163.

10. **Davydycheva, S., Druskin, V., & Habashy, T.** (2003). "An efficient finite-difference scheme for electromagnetic logging in 3D anisotropic inhomogeneous media". *Geophysics*, 68(5), 1525-1536.

### 14.5 Software de Modelagem EM

11. **Werthmuller, D.** (2017). "An open-source full 3D electromagnetic modeller for 1D VTI media in Python: empymod". *Geophysics*, 82(6), WB9-WB19. — Implementacao Python de referencia com Numba.

### 14.6 Quasi-Monte Carlo e Geracao de Modelos

12. **Sobol, I.M.** (1967). "On the distribution of points in a cube and the approximate evaluation of integrals". *USSR Computational Mathematics and Mathematical Physics*, 7(4), 86-112. — Sequencias quasi-aleatorias para amostragem uniforme.

### 14.7 Inversao EM via Deep Learning

13. **Morales, A. et al.** (2025). "Physics-Informed Neural Networks for Triaxial Electromagnetic Inversion with Uncertainty Quantification". — PINN para inversao EM triaxial com constrainte TIV.

### 14.8 Computacao GPU para Geofisica

14. **Weiss, C.J.** (2013). "Project APhiD: A Lorenz-gauged A-Phi decomposition for parallelized computation of ultra-broadband electromagnetic induction in a fully heterogeneous Earth". *Computers & Geosciences*, 58, 40-52. — GPU para modelagem EM.

15. **Commer, M. & Newman, G.A.** (2004). "A parallel finite-difference approach for 3D transient electromagnetic modeling with galvanic sources". *Geophysics*, 69(5), 1192-1202.

---

## 15. Apendices

### 15.1 Apendice A — Tabela de Sub-rotinas

| Sub-rotina | Modulo | Argumentos | Descricao |
|:-----------|:-------|:-----------|:----------|
| `perfila1DanisoOMP` | DManisoTIV | modelm, nmaxmodel, mypath, nf, freq, ntheta, theta, h1, tj, dTR, p_med, n, resist, esp, filename | Loop principal de perfilagem |
| `fieldsinfreqs` | DManisoTIV | ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, n, h, prof, resist, zrho, cH | Campos em todas as frequencias para um ponto T-R |
| `writes_files` | DManisoTIV | modelm, nmaxmodel, mypath, zrho, cH, nt, theta, nf, freq, nmeds, filename | Escrita de .dat e .out |
| `write_results` | DManisoTIV | results, nk, nj, ni, arq, filename | Escrita de resultados auxiliares |
| `sanitize_hprof_well` | utils | n, esp, h, prof | Prepara geometria com sentinelas |
| `findlayersTR2well` | utils | n, h0, z, prof, camadT, camad | Identifica camadas T e R |
| `sanitizedata2well` | utils | n, h0, z, esp, camadT, camad, h, prof | Versao combinada (geometria + camadas) |
| `commonarraysMD` | utils | n, npt, hordist, krJ0J1, zeta, h, eta, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt | Constantes de propagacao e reflexao |
| `commonfactorsMD` | utils | n, npt, h0, h, prof, camadT, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz | Fatores de onda da camada fonte |
| `layer2z_inwell` | utils | n, z, profs | Retorna indice da camada para profundidade z |
| `RtHR` | utils | alpha, beta, gamma, H | Rotacao do tensor H: R^T*H*R |
| `hmd_TIV_optimized` | magneticdipoles | Tx, Ty, h0, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, prof, zeta, eta, cx, cy, z, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, Mxdw, Mxup, Eudw, Euup, Hx_p, Hy_p, Hz_p, dipolo | Campo do HMD em meio TIV |
| `vmd_optimized` | magneticdipoles | Tx, Ty, h0, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, prof, zeta, cx, cy, z, u, uh, AdmInt, RTEdw, RTEup, FEdwz, FEupz, Hx_p, Hy_p, Hz_p | Campo do VMD em meio TIV |
| `J0J1Kong` | filterscommonbase | npt, absc, wJ0, wJ1 | Filtro Kong (61 pts) |
| `J0J1Key` | filterscommonbase | npt, absc, wJ0, wJ1 | Filtro Key |
| `J0J1Wer` | filterscommonbase | npt, absc, wJ0, wJ1 | Filtro Werthmuller (201 pts) |
| `J0J1And` | filterscommonbase | npt, absc, wJ0, wJ1 | Filtro Anderson (801 pts) |

### 15.2 Apendice B — Glossario

| Termo | Descricao |
|:------|:----------|
| **TIV** | Transversalmente Isotropico Vertical — modelo de anisotropia com sigma_h != sigma_v |
| **LWD** | Logging While Drilling — perfilagem durante a perfuracao |
| **HMD** | Horizontal Magnetic Dipole — fonte dipolar horizontal |
| **VMD** | Vertical Magnetic Dipole — fonte dipolar vertical |
| **TE** | Transverse Electric — modo com E tangencial as interfaces |
| **TM** | Transverse Magnetic — modo com H tangencial as interfaces |
| **Transformada de Hankel** | Integral de Bessel para passagem do dominio espectral para espacial |
| **DOI** | Depth of Investigation — profundidade de investigacao da ferramenta |
| **BHA** | Bottom Hole Assembly — conjunto de ferramentas na extremidade do drill string |
| **Skin depth** | Profundidade de penetracao da onda EM: delta = sqrt(2/(omega*mu*sigma)) |
| **Impeditividade** | zeta = i*omega*mu — constante que relaciona E e H |
| **Admitancia intrinseca** | Y = u/zeta — admitancia do modo TE por camada |
| **Impedancia intrinseca** | Z = s/eta_h — impedancia do modo TM por camada |

### 15.3 Apendice C — Mapeamento de Variaveis Fortran → Python

| Variavel Fortran | Tipo | Variavel Python (pipeline) | Descricao |
|:-----------------|:-----|:--------------------------|:----------|
| `resist(i,1)` | `real(dp)` | `y[:, :, 0]` (apos log10) | rho_h |
| `resist(i,2)` | `real(dp)` | `y[:, :, 1]` (apos log10) | rho_v |
| `cH(f,1)` | `complex(dp)` | `x[:, :, 1]` (Re) + `x[:, :, 2]` (Im) | Hxx |
| `cH(f,9)` | `complex(dp)` | `x[:, :, 19]` (Re) + `x[:, :, 20]` (Im) | Hzz |
| `zrho(f,1)` | `real(dp)` | `x[:, :, 0]` | z_obs (profundidade) |
| `nmed` | `integer` | `config.sequence_length` | 600 |
| `freq(1)` | `real(dp)` | `config.frequency_hz` | 20000.0 |
| `dTR` | `real(dp)` | `config.spacing_meters` | 1.0 |
| `n` | `integer` | `model['n_layers']` | Numero de camadas |
| `esp(2:n-1)` | `real(dp)` | `model['thicknesses']` | Espessuras internas |

### 15.4 Apendice D — Validacao de Consistencia Fortran-Python

Para garantir que o pipeline Python le corretamente os dados do Fortran, os seguintes testes de consistencia devem ser executados:

```python
# Teste 1: Numero de registros
n_records_expected = sum(nmed) * nf * nmaxmodel
n_records_actual = os.path.getsize(filepath) // 172
assert n_records_actual == n_records_expected

# Teste 2: Ranges fisicos
assert np.all(rho_h > 0)          # Resistividade positiva
assert np.all(rho_v >= rho_h)     # Constraint TIV
assert np.all(np.isfinite(H_real + H_imag))  # Sem NaN/Inf

# Teste 3: Simetria do tensor para theta=0 (poco vertical)
# Para poco vertical, Hxy = Hyx (simetria)
assert np.allclose(Re_Hxy, Re_Hyx, rtol=1e-10)
assert np.allclose(Im_Hxy, Im_Hyx, rtol=1e-10)
```

---

*Documentacao do Simulador Fortran PerfilaAnisoOmp — Geosteering AI v2.0*
*Gerado em Abril 2026 — Pipeline v5.0.15+*
