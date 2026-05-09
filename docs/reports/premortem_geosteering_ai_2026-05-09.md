# Pré-Mortem Geosteering AI — Análise Crítica de Falha Hipotética

**Versão**: 1.0
**Data**: 2026-05-09
**Autor**: Daniel Leal (calibrado com Claude Opus 4.7)
**Branch**: `feat/premortem-analysis-artifacts`
**Status**: Documento de referência para análises críticas futuras

---

## 1. Sumário Executivo

Este documento apresenta uma análise pré-mortem detalhada do projeto Geosteering AI, executada como exercício de governança crítica. A premissa metodológica é: "É 2028, o projeto Geosteering AI falhou. O que aconteceu?" Esse enquadramento força identificar pontos fracos antes que se tornem fatais e desafia premissas arquiteturais centrais.

A análise identifica **8 pontos fracos**, **7 premissas adversariais**, e propõe **6 blocos de melhoria**. Cada item é discutido com evidência direta do estado atual do projeto (v2.22.6, branch `feat/fase1-fundacao-multiagente`). Após a análise inicial, o usuário forneceu **5 observações de calibração** que corrigem premissas equivocadas e ajustam recomendações ao contexto real do projeto. As recomendações finais refletem essa calibração.

**Principais conclusões pós-calibração**:

- O projeto tem fundação técnica sólida (paridade Fortran <1e-12, 1 597 testes PASS, Quality Mesh 7 camadas) e está dentro da janela de prazo prevista (14–22 meses para entrega com protótipos intermediários)
- O risco principal NÃO é "ausência de produto" mas **descalibração entre velocidade de construção de infraestrutura e velocidade de validação científica**
- Datasets reais (SDAR, Volve, Teapot) devem ser incorporados como **caminho suplementar / teste antecipado**, não como bloqueio do roadmap dos simuladores
- 3 melhorias arquiteturais foram aprovadas e incorporadas ao documento de aprofundamento: §74 (métodos alternativos de inversão), §75 (Framework-Agnostic Core), §24.4 (cadência de pré-mortem)

---

## 2. Metodologia Pré-Mortem

### 2.1 Premissa Central

> "É 2028. O projeto Geosteering AI foi abandonado. Trabalhe em retrospectiva: o que deu errado?"

Diferente de uma análise de risco tradicional (que pondera probabilidades de falhas individuais), o pré-mortem **assume falha** e força raciocínio contrafactual sobre causas raiz. Essa inversão revela pontos cegos que o otimismo do dia-a-dia mascara.

### 2.2 Eixos de Análise

| Eixo | Pergunta-chave |
|:-----|:---------------|
| Dados | A validação atual generaliza para dados reais? |
| Arquitetura | A complexidade entrega valor proporcional? |
| Latência | O hardware-alvo real atinge o SLA esperado? |
| Dependências | Quais vendors são pontos únicos de falha? |
| Produto | O que o usuário final pode usar **hoje**? |
| Validação | Os testes verificam o que importa? |
| Governança | A cadência de revisão crítica existe? |
| Mercado | O produto ainda será relevante quando entregar? |

### 2.3 Análise Adversarial

Para cada premissa central, formula-se o **oposto** e busca-se evidência que o sustente. O objetivo não é convencer-se do oposto, mas testar a robustez da premissa original.

---

## 3. Pontos Fracos Identificados

### 3.1 Gap de Dados Reais — O Problema Central

O projeto inteiro foi construído sobre dados **sintéticos** gerados pelo simulador Fortran `tatu.x`. O ciclo é fechado: Fortran gera → DL aprende a inverter Fortran → DL é validado contra Fortran. Não existe saída desse loop sintético.

**O que `tatu.x` não modela**: ruído eletrônico de BHA, deriva de temperatura, vibração mecânica do trepano, efeitos de invasão de fluido de perfuração, bordas de camada com resolução finita de ferramenta, interferência de colunas metálicas adjacentes.

**Distribuição estatística divergente**: o currículo de treino usa sorteios uniformes de parâmetros geológicos. Modelos reais têm correlações espaciais, hierarquia de fácies, trends que dados sintéticos não capturam.

**Evidência direta**: `MEMORY.md` lista explicitamente "Treino SurrogateNet: re-simular Fortran multi-dip + treinar TCN/ModernTCN Colab GPU" como pendente. O modelo principal nunca foi treinado em dados multi-dip.

### 3.2 Complexidade Combinatória Sem Validação de Acurácia

| Métrica | Valor |
|:--------|------:|
| Arquiteturas | 48 |
| Loss functions | 26 |
| Tipos de ruído | 34 |
| Cenários PINN | 8 |
| Configurações combinatórias | ~42 432 |

Os testes verificam: shapes, forward pass sem erro, paridade numérica, validação de configuração. Não verificam: qual configuração produz a melhor inversão geológica, em que condições uma arquitetura supera outra, se a incerteza reportada é calibrada honestamente (calibration curve, ECE).

### 3.3 Dimensionalidade 1D em Mundo 3D

Geosteering real ocorre em trajetórias inclinadas com heterogeneidades laterais — falhas, canais fluviais, lentes de areia, cunhas. O modelo 1D-TIV assume camadas horizontais infinitas, geometria que raramente existe exatamente nas situações onde o geosteering é mais necessário.

**Estado de mercado**: Halliburton, Schlumberger e Baker Hughes já oferecem inversão 2D/2.5D em produtos comerciais.

**Mitigação prevista**: §21 do documento de aprofundamento define roadmap 1D → Born 2D → MEF 2D → 2.5D → 3D, com cronograma de ~24 meses.

### 3.4 Latência Real vs. Latência Benchmarkada

Premissa original: "o benchmark é feito em Apple M-series otimizada".

**Calibração do usuário (Obs. 2)**: o desenvolvimento ocorre em **MacBook Pro Intel Core i9 2019**, hardware mais próximo do alvo final (desktops industriais). Premissa de "M-series ARM otimizada" estava equivocada. O hardware atual é uma boa proxy para o ambiente final do **Geosteering AI Studio** e **Simulation Manager** standalone.

**Risco residual**: latência do pipeline DL completo (pré-processamento + inferência + pós-processamento) em hardware de campo ainda não foi benchmarkada como sistema integrado.

### 3.5 Meta-Complexidade da Arquitetura Multi-Agente

| Componente | Quantidade |
|:-----------|----------:|
| Agentes | 27+ |
| Skills | 41+ |
| MCP servers | 4 |
| Workflows orquestrados | 12 |

**Risco**: a sequência lógica recomendada seria (1) validar produto contra dados reais, (2) construir automação para escalar produto validado. A sequência atual é (1) construir automação completa, (2) validar produto pendente.

**Calibração do usuário (Obs. 3)**: o prazo de entrega é 14–22 meses com protótipos intermediários. O projeto está dentro da janela. Mas o pré-mortem mantém validade como instrumento de calibração.

### 3.6 Dependências Single-Vendor Críticas

| Dependência | Risco | Mitigação proposta |
|:------------|:------|:-------------------|
| TensorFlow/Keras exclusivo | PyTorch tem >70% share em DL geofísica 2024-2026 | Framework-Agnostic Core (§75 doc arquitetura) |
| Google Colab Pro+ | Cotas instáveis, latência I/O via Drive | Calibração: uso justificado por ausência de GPU local + flexibilidade |
| `tatu.x` como ground truth | Binário sem testes independentes publicados | SDAR/SPWLA RtSIG como ground truth institucional (§7.3 abaixo) |
| Anthropic API | Custo de sessões longas com 27 agentes | Caveman opcional + caching agressivo (§39, §32 doc arquitetura) |

### 3.7 Ausência de Produto Entregável (Calibrado)

Premissa original: "Após 73 módulos, ~46k LOC, 744 testes, o que o usuário final pode usar hoje?"

**Calibração do usuário (Obs. 3)**: o projeto está em janela de 14–22 meses. Protótipos serão entregues nesse período. **Não há ausência prematura de produto** — há um roadmap em execução. O ponto fraco real é diferente: o ritmo de construção de **infraestrutura de agentes** está aparentemente à frente do ritmo de **validação científica do produto**.

### 3.8 Testes Sem Validação Científica Independente

**1 597 testes PASS** verificam consistência interna (paridade, shapes, configs). O que não verificam: acurácia de inversão em casos geológicos reais, incerteza calibrada honestamente, comparação com métodos estabelecidos (Occam, Tikhonov), blind test em dados de campo.

A Quality Mesh 7 camadas é **necessária mas não suficiente** — pode criar falsa sensação de qualidade.

---

## 4. Análise Adversarial — "E se o oposto for verdade?"

### 4.1 Premissa: "DL supera inversão analítica regularizada em geosteering 1D"

**Oposto**: Para problemas com N≤6 camadas (caso típico geosteering), inversão de Occam regularizada com jacobiano `∂H/∂ρ` (já implementado!) pode ser mais rápida, mais interpretável, com incerteza calibrada matematicamente, e com garantias de convergência.

**Evidência de risco**: o projeto computa `∂H/∂ρ` em Numba — exatamente o ingrediente para inversão de segunda ordem. Mas nenhum benchmark DL versus Occam foi executado em dados reais.

**Decisão pós-análise**: implementar 3 métodos alternativos como benchmark (§74 doc arquitetura: Occam + LUT + Tikhonov).

### 4.2 Premissa: "Dados sintéticos com ruído Gaussiano on-the-fly são suficientes"

**Oposto**: Modelos geológicos reais têm correlações espaciais, hierarquias de fácies, distribuições de resistividade que refletem processos deposicionais — nada disso é capturado por sorteios uniformes. Ruído real LWD não é Gaussiano (spikes, deriva lenta, outliers).

**Calibração do usuário (Obs. 1)**: dados reais virão em etapa avançada. Aceito. Mas adapter de dados reais como **teste antecipado suplementar** (§7.6 abaixo) garante que o pipeline esteja pronto quando a etapa chegar.

### 4.3 Premissa: "Paridade Fortran <1e-12 é a métrica de qualidade física central"

**Oposto**: Paridade <1e-12 com Fortran só é relevante se o Fortran for fisicamente correto. Se `tatu.x` tem simplificações na modelagem, paridade perfeita propaga esses erros com fidelidade absoluta.

**Mitigação proposta**: complementar `tatu.x` com **SDAR/SPWLA RtSIG** (canonical models institucionais peer-reviewed, OSTI 1501648) como segunda fonte de verdade. Não substitui — adiciona.

### 4.4 Premissa: "48 arquiteturas aumentam robustez"

**Oposto**: 48 arquiteturas sem benchmark de acurácia em geologia real é 48 hipóteses não testadas. A "melhor" arquitetura selecionada em benchmark sintético pode ser inferior em campo.

**Decisão proposta**: definir critério de seleção objetivo + reduzir a 10–12 arquiteturas prioritárias documentadas.

### 4.5 Premissa: "TF/Keras exclusivo protege a coerência do projeto"

**Oposto**: A restrição isola o projeto. PyTorch é o padrão de facto em DL geofísica 2024-2026. Manter a proibição significa custo de oportunidade real (colaborações, modelos pré-treinados, integrações industriais).

**Decisão pós-análise (Obs. 5 do usuário)**: implementar **Framework-Agnostic Core** (§75 doc arquitetura). TF segue como default exclusivo no pipeline de produção; PyTorch permitido para módulos de pesquisa via adapter isolado.

### 4.6 Premissa: "Quality Mesh 7 camadas garante qualidade do produto"

**Oposto**: As 7 camadas verificam qualidade de **código**, não qualidade **científica**. Um sistema pode passar em todas as 7 camadas com 0 findings do CodeRabbit e ainda assim produzir inversões geologicamente implausíveis.

**Decisão proposta**: adicionar **Layer 8 — Validação Científica** ao Quality Mesh: comparação contra benchmarks SDAR + métodos analíticos + incerteza calibrada.

### 4.7 Premissa: "A arquitetura multi-agente acelera o desenvolvimento"

**Oposto**: A arquitetura multi-agente pode estar criando trabalho que gera mais trabalho. Cada sprint de infraestrutura produz artefatos que precisam de manutenção, relatórios, code reviews, atualizações de ROADMAP/CLAUDE.md.

**Decisão proposta**: instituir **cadência de pré-mortem trimestral** (§24.4 doc arquitetura) como mecanismo de calibração explícito entre infraestrutura e produto científico.

---

## 5. Esquema de Melhorias (6 Blocos)

### B0 — Validação com Dados Reais (Calibrado)

**Versão original**: "bloqueante; obter ≥3 datasets antes de continuar".

**Versão calibrada (após Obs. 1 do usuário)**:

- **Não bloqueante**. Prioridade primária permanece: simuladores 1D → 2D → 2.5D → 3D conforme §21
- Dados reais entram como **caminho suplementar / teste antecipado** via adapter opt-in
- Ação imediata: criar `geosteering_ai/data/loaders/real_data_adapter.py` (Sprint v2.28)
- Datasets recomendados: **SDAR primário** (institucional sintético peer-reviewed), **Volve/Teapot suplementares** (validação adicional)
- Métrica de sucesso: pipeline carrega LAS real → 22-col tensor → inferência sem erro. Não exige acurácia ainda

### B1 — Baseline Científico Rigoroso

Implementar 3 métodos alternativos de inversão em paralelo ao DL:

| Método | Arquivo previsto | Latência alvo | Uso |
|:-------|:-----------------|:-------------:|:----|
| Occam regularizado | `geosteering_ai/inversion/occam.py` | <0.1s para N≤6 camadas | Baseline analítico, incerteza calibrada |
| Look-up Table (LUT) | `geosteering_ai/inversion/lut.py` | <1ms | Hardware limitado, fallback |
| Tikhonov | `geosteering_ai/inversion/tikhonov.py` | <0.5s | Baseline clássico para paper |

Benchmark comparativo em casos canônicos SPWLA SDAR: `benchmarks/bench_inversion_methods.py`. Detalhes em §74 do documento de aprofundamento.

### B2 — Redução Controlada de Complexidade

| Item | Estado atual | Estado alvo |
|:-----|:------------:|:-----------:|
| Arquiteturas | 48 | 10–12 |
| Loss functions | 26 | ≤10 |
| Tipos de ruído | 34 | ≤15 |
| Critério de seleção | Implícito | Documentado + automatizado |

Não remover código — mover para `legacy/` com README de depreciação. Manter apenas arquiteturas com benchmark de acurácia em ≥3 casos geológicos.

### B3 — Hardware de Campo Realista (Calibrado)

**Versão calibrada (após Obs. 2 do usuário)**: hardware atual (MacBook Pro Intel i9 2019) é proxy razoável para hardware-alvo final (desktops industriais).

**Ações**:

- Definir SLA explícito: latência máxima aceitável para inversão durante perfuração
- Benchmarkar pipeline integrado (não apenas simulador) em perfil de uso típico
- Documentar mode-switching DL → Occam quando recursos limitados (já habilitado por B1)

### B4 — Expansão 2D / 2.5D / 3D (Calibrado)

**Versão calibrada (após Obs. 1 do usuário)**: já está no roadmap (§21). Pré-mortem confirma prioridade.

| Versão | Backend | Cronograma §21.6 |
|:-------|:--------|:----------------:|
| v2.30 | Backend abstrato | ~2 meses |
| v2.40 | Born 2D | ~3 meses |
| v2.50 | MEF 2D | ~5 meses |
| v2.60 | MEF 2.5D | ~3 meses |
| v3.0 | MEF 3D | ~9–12 meses |

### B5 — Framework-Agnostic Core

Detalhado em §75 do documento de aprofundamento. Decisão pós-Obs. 5 do usuário.

### B6 — Documentação Científica (Paper)

Sequência: B0 (adapter de dados reais) → B1 (3 métodos alternativos como baseline) → paper de validação. Sem B0+B1, não há paper publicável com peer-review.

---

## 6. Observações do Usuário e Respostas Calibradas

Esta seção registra as 5 observações do usuário sobre a análise pré-mortem original e as respostas calibradas que foram incorporadas às recomendações finais.

### Obs. 1 — Dados Reais e Simuladores 2D/2.5D/3D

> "Os dados reais serão obtidos no futuro. Fazem parte do planejamento. Por hora, utilizaremos dados simulados 1D e, posteriormente, simulação 2D, 2.5D e 3D (...). Podemos adiantar a obtenção de dados reais para serem usados no futuro. Lembre-se: Geosteering AI deve ser construído para lidar tanto com dados obtidos por simulador quanto dados reais."

**Resposta calibrada**:

- B0 reposicionado como **não-bloqueante**. Prioridade primária permanece simuladores
- Adapter de dados reais entra como **caminho suplementar / teste antecipado**: `geosteering_ai/data/loaders/real_data_adapter.py`
- Análise de viabilidade dos datasets executada (§7 abaixo): SDAR=VIÁVEL, Volve/Teapot/ANP=PARCIAL, NGDS/Penobscot=NÃO VIÁVEL
- Sprint v2.28 incluída no ROADMAP.md como suplementar não-bloqueante

### Obs. 2 — Hardware de Desenvolvimento

> "O desenvolvimento está sendo em uma máquina MacBook Pro Intel Core i9 de 2019, não em uma máquina M-Series. O ambiente final do software (Geosteering AI Studio, Simulation Manager) está planejado para ser executado em máquinas desktop com configurações similares (...). O uso atual de Google Colab Pro+ ocorre por: a) ausência de GPUs locais; b) flexibilidade."

**Resposta calibrada**:

- Premissa original sobre "M-series otimizada" estava equivocada — corrigida em §3.4
- Hardware atual (i9 2019) é proxy razoável para alvo final
- Justificativa Colab Pro+ aceita como racional dado contexto atual (sem GPU local)
- B3 reescrito sem assumir desconexão entre dev e produção

### Obs. 3 — Prazo de Entrega

> "O prazo de entrega é entre 14 e 22 meses, com a apresentação de protótipos e programas funcionais nesse período."

**Resposta calibrada**:

- §3.7 ("Ausência de produto entregável") recalibrado: o projeto está dentro da janela de prazo prevista
- Pré-mortem mantém validade como **instrumento de calibração**, não como julgamento de atraso
- Risco real é desbalanço entre construção de infraestrutura e validação científica — não atraso de produto

### Obs. 4 — Métodos Alternativos de Inversão

> "Inclua e incorpore na documentação da arquitetura do projeto Geosteering AI a implementação dos três métodos alternativos: Occam regularizada, LUT, Tikhonov."

**Resposta**: aprovado e incorporado como **§74** no documento de aprofundamento. ROADMAP.md atualizado com Sprint v2.29.

### Obs. 5 — Framework-Agnostic Core

> "Inclua e incorpore na documentação da arquitetura do projeto Geosteering AI o framework Framework-Agnostic Core."

**Resposta**: aprovado e incorporado como **§75** no documento de aprofundamento. ROADMAP.md atualizado com Sprint v2.30. TF segue como default exclusivo no pipeline de produção; PyTorch permitido apenas via adapter isolado em `adapters/`.

### Obs. extra — Skill de Pré-Mortem

> "CASO SEJA POSSÍVEL, UM AGENTE VOLTADO PARA ANÁLISE PRÉ-MORTEM PODERIA SER ÚTIL."

**Resposta**: aprovado. Skill `geosteering-premortem-analyst` criada em `.claude/commands/geosteering-premortem-analyst.md` (Opus 4.7, effort extra-high). Cadência trimestral instituída em §24.4 do documento de aprofundamento.

---

## 7. Análise de Viabilidade de Datasets Reais — Validação Suplementar Antecipada

**Posicionamento estratégico**: dados reais entram como **caminho suplementar / teste antecipado**, paralelo ao caminho primário (simuladores 1D → 2D → 2.5D → 3D conforme §21). Não substituem nem condicionam o avanço dos simuladores. Isso preserva a viabilidade do roadmap de 14–22 meses sem introduzir dependência crítica de aquisição de datasets externos.

### 7.1 Tabela Classificatória

| Dataset | Classe | Uso recomendado |
|:--------|:------:|:----------------|
| **SDAR / SPWLA RtSIG** | **VIÁVEL** | Canonical models institucionais — paridade Fortran <1e-12 + validação DL contra publicação peer-reviewed (OSTI 1501648) |
| **Equinor Volve** | PARCIAL | Pre-training SurrogateNet + validação petrofísica. Licença NC bloqueia uso comercial; logs LWD pós-processados (não tensor 3×3 raw) |
| **Teapot Dome / RMOTC** | PARCIAL | Petrofísica auxiliar (Archie/Picasso) — domínio público. 31 LAS com RD/RS dual-laterolog wireline single-freq |
| **ANP BDEP (Brasil)** | PARCIAL | Validação geográfica BR pós-período de confidencialidade (até 10a). Solicitação manual, não-API |
| **USGS NGDS** | NÃO VIÁVEL | Domínio ortogonal (geotermia rasa, sem skin depth/anisotropia TIV) |
| **Penobscot 3D** | NÃO VIÁVEL | Sísmico 3D — apenas 2 poços wireline pré-LWD moderno |
| **OEDI / WoGSS** | NÃO VIÁVEL | Sem datasets LWD-EM dedicados |

### 7.2 Equinor Volve — Análise Detalhada

| Atributo | Valor |
|:---------|:------|
| URL | `https://data.equinor.com/` + `https://www.equinor.com/energy/volve-data-sharing` |
| Licença | Equinor Open Data License (modelo CC BY-NC-SA 4.0) |
| Tamanho | ~5 TB total, 16,3 GB de well logs, ~40k arquivos |
| Poço relevante | **15/9-F-1 C** (suite LWD ampla) |
| Formatos | LAS (logs) + DLIS (composite/integrity/production) + TIF raster |
| Resistividade | Deep/medium induction + spherically focused (multi-DOI), pós-processada |
| Limitação crítica | Sem dados raw multi-freq de propagation tools (ARC/PeriScope/GeoSphere). Curvas pós-processadas — incompatíveis com `INPUT_FEATURES=[1,4,5,20,21]` do formato 22-col |

**Veredito**: útil para **pre-training SurrogateNet** e validação petrofísica (Archie/Picasso); inadequado como fonte primária de tensor raw 3×3.

### 7.3 SDAR / SPWLA RtSIG — Recomendação Primária

**Fonte**: SPWLA RtSIG benchmarks + caso canônico data-driven (OSTI 1501648).

**Configurações canônicas**:

- 155 ft @ 2 kHz
- 75 ft @ 6–12 kHz
- 45 ft @ 12–24 kHz
- 24 in @ 2 MHz

**Por que primário**:

- É **conjunto de modelos canônicos institucionais peer-reviewed** — alinhado naturalmente com paridade Fortran <1e-12
- Não é dataset de campo, mas serve como ground truth científico independente complementar a `tatu.x`
- Permite validar DL contra publicação peer-reviewed sem risco de licença comercial
- Adicionar como modelos canônicos em `geosteering_ai/simulation/validation/canonical_models.py` (extensão da lista atual de 7 modelos)

### 7.4 USGS NGDS — Descartado

URL: `https://data.geothermaldata.org/` (ativo em 2026, mas domínio ortogonal). Conteúdo: temperatura de poço, gradientes geotermais, faults, geoquímica. Logs rasos focados em geotermia.

**Razão de descarte**: domínio físico ortogonal — geotermia rasa, sem skin depth/anisotropia TIV/multi-DOI 24in–155ft característico de geosteering em hidrocarbonetos.

### 7.5 Teapot Dome / ANP BDEP — Suplementares Opcionais

**Teapot Dome / RMOTC** (`https://wiki.seg.org/wiki/Teapot_dome_3D_survey`):

- Domínio público DOE
- 37 poços no Tensleep, 31 com LAS (GR, RHOB, NPHI, RD, RS apenas deep+shallow)
- Resistividade: dual-laterolog wireline single-freq (não LWD propagation)
- Útil para DTB/Picasso e modelos petrofísicos auxiliares (Archie)

**ANP BDEP** (`https://www.gov.br/anp/pt-br/assuntos/exploracao-e-producao-de-oleo-e-gas/dados-tecnicos/acervo-de-dados`):

- 3,3 PB em LAS + composite + AGP
- Confidencialidade até 10a pós-aquisição
- Solicitação manual, não-API
- Vantagem: aderência geográfica BR
- Recomendação: avaliar quando houver bandwidth burocrático

### 7.6 Plano de Incorporação à Infraestrutura

**Adapter opt-in**: `geosteering_ai/data/loaders/real_data_adapter.py`

**Etapas**:

1. **Download manual** (não automatizado — datasets requerem aceite de licença individual)
2. **Conversão LAS → tensor 22-col**: usar `welly` ou `lasio`; mapear curvas disponíveis para layout INPUT_FEATURES
3. **Adapter sem fitar pipeline**: dados reais NÃO entram em `DataPipeline.fit_scaler()` (preserva fit em dados limpos sintéticos)
4. **Smoke validation**: pipeline carrega LAS → tensor → inferência sem erro
5. **Não bloqueia nem altera fluxo de produção atual**

**Princípio de design**: dados reais são **read-only consumers** do pipeline existente, não modificadores.

### 7.7 Sources

- [Volve field data set — Equinor](https://www.equinor.com/energy/volve-data-sharing)
- [Equinor Open Data Portal](https://data.equinor.com/)
- [Disclosing all Volve data — Equinor 2018](https://www.equinor.com/news/archive/14jun2018-disclosing-volve-data)
- [Well Log Suite of the Volve Oilfield](https://discovervolve.com/2023/02/12/well-log-suite-of-the-volve-oilfield/)
- [National Geothermal Data System](https://data.geothermaldata.org/)
- [NGDS Wikipedia](https://en.wikipedia.org/wiki/National_Geothermal_Data_System)
- [Open Energy Data Initiative (OEDI)](https://data.openei.org/)
- [Penobscot 3D — TerraNubis](https://terranubis.com/datainfo/Penobscot)
- [Teapot Dome 3D Survey — SEG Wiki](https://wiki.seg.org/wiki/Teapot_dome_3D_survey)
- [Teapot Dome — Data Underground](https://dataunderground.org/dataset/teapot-dome)
- [ANP — Acervo de dados técnicos](https://www.gov.br/anp/pt-br/assuntos/exploracao-e-producao-de-oleo-e-gas/dados-tecnicos/acervo-de-dados)
- [Solicitar acesso a dados técnicos E&P — Gov.br](https://www.gov.br/pt-br/servicos/acessar-dados-tecnicos-exploracao-e-producao)
- [Data-Driven Interpretation of Ultra-Deep Azimuthal Resistivity — OSTI 1501648](https://www.osti.gov/servlets/purl/1501648)
- [Towards Well Placement Automation — SPE 25MEOS](https://onepetro.org/SPEMEOS/proceedings-abstract/25MEOS/25MEOS/790207)

---

## 8. Síntese — Os 3 Problemas Raiz

Se o projeto falhar em 2028, será por uma destas razões, em ordem de probabilidade:

| # | Problema | Solução proposta |
|:-:|:---------|:-----------------|
| 1 | O modelo funciona contra Fortran mas não contra a terra real | Adapter opt-in + SDAR como ground truth institucional + B0 calibrado |
| 2 | O produto chegou tarde — inversão 2D/2.5D já é o padrão de mercado | §21 do doc arquitetura cobre o roadmap; pré-mortem confirma prioridade |
| 3 | A complexidade de infraestrutura obscureceu o objetivo científico | §24.4 (cadência pré-mortem trimestral) + B2 (redução controlada de complexidade) |

---

## 9. Pontos Fortes Reais do Projeto

A análise crítica não deve obscurecer o que é genuinamente excepcional:

| Conquista | Evidência |
|:----------|:----------|
| Paridade Numba/JAX vs Fortran <1e-12 em 1 597 testes | `pytest tests/` com 1597 PASS / 295 SKIP / 0 FAIL |
| Simulator Numba atinge >120k modelos/hora | Cenário E v2.21–v2.22 (KB-013 resolvido) |
| Quality Mesh 7 camadas | §41 do doc arquitetura, ~3 anos de defesa em profundidade |
| Jacobiano `∂H/∂ρ` implementado | Abre a porta para inversão analítica rigorosa |
| Documentação rica nível C28 (D1-D14) | CLAUDE.md §"Padroes de Documentacao" |
| Arquitetura multi-agente formalizada (27 agentes, 4 MCPs) | §3-§22 doc aprofundamento |
| Skills v2.22.5 com model+effort granular | physics-reviewer Sonnet→Opus 4.7 |

O projeto tem **fundação técnica sólida**. O que está em risco não é a qualidade do código — é a relevância científica do produto se a validação com dados reais continuar deferida indefinidamente.

---

## 10. Skill `geosteering-premortem-analyst`

**Justificativa**: precedente direto em 4 skills de revisão crítica já existentes (`geosteering-code-reviewer`, `geosteering-physics-reviewer`, `geosteering-perf-reviewer`, `geosteering-security-auditor`). A skill nova preenche o nicho de **análise crítica de premissas arquiteturais sistêmicas** — diferente de revisão de código (qualidade local) ou física (correção numérica).

**Localização**: `.claude/commands/geosteering-premortem-analyst.md`

**Configuração**:

| Atributo | Valor |
|:---------|:------|
| Modelo | `claude-opus-4-7` |
| Effort | `extra-high` |
| Tools | `Read`, `Agent`, `WebSearch`, `WebFetch` |
| Allowed paths | `docs/**`, `geosteering_ai/**`, `.claude/commands/**`, `tests/**` |
| Triggers | "pré-mortem", "premortem", "análise crítica", "pontos fracos", "what if opposite", "challenge assumptions" |

**Cadência**: trimestral (mínimo), OU sob gatilho explícito (release major, mudança de fase, decisão arquitetural significativa). Detalhes em §24.4 do documento de aprofundamento.

---

## 11. Recomendação Pós-Calibração

Após calibração com as 5 observações do usuário, a recomendação final é:

| Prioridade | Ação | Sprint | Esforço |
|:----------:|:-----|:------:|:--------|
| 1 | Manter prioridade primária: simuladores 1D → 2D → 2.5D → 3D conforme §21 | em curso | — |
| 2 | Criar adapter opt-in para dados reais (SDAR primário + Volve/Teapot suplementares) | v2.28 | ~2 sem |
| 3 | Implementar 3 métodos alternativos de inversão (Occam + LUT + Tikhonov) | v2.29 | ~3-4 sem |
| 4 | Implementar Framework-Agnostic Core (BaseInversionModel + adapters TF/PyTorch/ONNX) | v2.30 | ~4-6 sem |
| 5 | Instituir cadência de pré-mortem trimestral via skill `geosteering-premortem-analyst` | contínuo | ~3h/trimestre |
| 6 | Reduzir complexidade combinatória (48 arqs → 10–12, 26 losses → ≤10) | v2.31+ | ~6 sem |
| 7 | Adicionar Layer 8 — Validação Científica ao Quality Mesh | v2.32+ | ~4 sem |

**Total adicional ao roadmap atual**: ~6 sprints distribuídas (compatível com janela de 14–22 meses).

---

## 12. Próximos Passos Imediatos

Checklist para sessão de implementação atual:

- [x] Branch `feat/premortem-analysis-artifacts` criada a partir de `feat/fase1-fundacao-multiagente@138ab49`
- [ ] Relatório pré-mortem gravado em `docs/reports/premortem_geosteering_ai_2026-05-09.md`
- [ ] Atualização §74 (Backends de Inversão Alternativa) em documento de aprofundamento
- [ ] Atualização §75 (Framework-Agnostic Core) em documento de aprofundamento
- [ ] Atualização §24.4 (Cadência de Pré-Mortem) em documento de aprofundamento
- [ ] Atualização ROADMAP.md com 4 entradas (Sprint v2.28, v2.29, v2.30, F-cross premortem)
- [ ] Skill `geosteering-premortem-analyst.md` criada com modelo Opus 4.7 + effort extra-high
- [ ] MEMORY.md atualizado com pointer para este relatório
- [ ] Commits granulares (5)
- [ ] /code-review final via CodeRabbit CLI
- [ ] Branch permanece aberta para revisão do usuário antes de merge

**Decisão de merge**: usuário decide após revisão completa. Tag esperada se aprovado: `v2.22.7-docs` (patch de documentação, não muda código de produção).

---

## Anexo A — Mapeamento Documento × Decisões

| Decisão | Localização incorporada |
|:--------|:------------------------|
| Métodos alternativos (Occam, LUT, Tikhonov) | §74 do doc aprofundamento + Sprint v2.29 ROADMAP |
| Framework-Agnostic Core | §75 do doc aprofundamento + Sprint v2.30 ROADMAP |
| Cadência de pré-mortem trimestral | §24.4 do doc aprofundamento |
| Adapter opt-in para dados reais | Sprint v2.28 ROADMAP + `geosteering_ai/data/loaders/real_data_adapter.py` |
| Skill `geosteering-premortem-analyst` | `.claude/commands/geosteering-premortem-analyst.md` |
| Pointer no MEMORY.md | seção "Quality Mesh — Estado Atual" |

---

**Documento finalizado em**: 2026-05-09
**Próxima revisão programada**: trimestral (próxima ≈ 2026-08-09) OU sob gatilho explícito
**Autoridade de aprovação**: Daniel Leal (autor do projeto)
