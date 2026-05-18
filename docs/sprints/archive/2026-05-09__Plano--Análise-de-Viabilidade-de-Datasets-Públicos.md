# Plano — Análise de Viabilidade de Datasets Públicos para Geosteering AI

**Data:** 2026-05-09 · **Modo:** Plan (apenas leitura) · **Objetivo:** Avaliar incorporação de datasets públicos no projeto Geosteering AI v2.0 (inversão 1D EM-LWD via DL).

---

## Critério de Classificação

| Classe | Definição |
|:------|:----------|
| **VIÁVEL** | LWD multi-frequência + interpretação estratigráfica + licença permissiva + LAS/DLIS digital |
| **PARCIAL** | Útil para validação cruzada ou pre-training, mas falta característica-chave (single-freq, sem LWD propagation, formato heterogêneo) |
| **NÃO VIÁVEL** | Não atende caso de uso (sísmica pura, geotermia rasa, sem resistividade, restrito) |

---

## 1. Equinor Volve Dataset — **PARCIAL**

| Campo | Valor |
|:------|:------|
| URL oficial | `https://www.equinor.com/energy/volve-data-sharing` + portal `https://data.equinor.com/` |
| Licença | **Equinor Open Data License** (modelo CC BY-NC-SA 4.0) — uso acadêmico OK; **comercial RESTRITO** (NC) e share-alike (SA) propaga licença derivativos |
| Tamanho | ~5 TB total · 16,3 GB de well logs · ~40 mil arquivos |
| Formatos | LAS (maioria dos logs) + DLIS (composite/integrity/production) + TIF (alguns LWD em imagem raster, **não digital**) |
| Conteúdo LWD relevante | Wells de desenvolvimento usaram **MWD/LWD exclusivo**; well 15/9-F-1 C tem suite LWD ampla. Resistividade: deep induction, medium induction, spherically focused (multi-DOI) |
| **Limitação crítica** | **Não publicam dados raw multi-frequência de propagation tools** (e.g., ARC/PeriScope/GeoSphere). Resistividades são curvas pós-processadas (induction logs), não componentes Hxx/Hzz/Hxz raw que o pipeline `geosteering_ai/` consome (INPUT_FEATURES = [1,4,5,20,21] do formato 22-col). Alguns LWD vêm como TIF (sem extração digital) |

**Justificativa PARCIAL:** Excelente para **validação de plausibilidade petrofísica** (Archie, comparação com inversão clássica) e para **pre-training do SurrogateNet** em condições reais; **inadequado** como fonte primária de tensor 3×3 raw para inversão 1D direta. Licença NC bloqueia uso comercial futuro do modelo treinado em derivativos.

## 2. USGS NGDS — **NÃO VIÁVEL**

| Campo | Valor |
|:------|:------|
| URL | `https://data.geothermaldata.org/` (ativo) |
| Licença | Variável por contribuidor (federal US, geralmente domínio público) |
| Conteúdo | Borehole temperatures, gradientes geotermais, faults, geochem, alguns logs **rasos** (geotermia, não hidrocarbonetos) |
| Limitação | **Sem LWD multi-freq de propagation**. Foco em geotermia: temperatura, fluxo de calor, química. Resistividade rara e em poços rasos sem o regime físico do projeto (skin depth, anisotropia TIV, multi-DOI 24in–155ft) |

**Justificativa NÃO VIÁVEL:** Domínio físico ortogonal ao caso de uso (geosteering em reservatório de hidrocarbonetos). Mesmo logs presentes não têm tensor EM raw nem condições operacionais de LWD horizontal.

### Alternativas avaliadas
- **OEDI/NREL (`data.openei.org`):** agregador. Sem dataset LWD-EM dedicado; foco em renováveis + geotermia. **NÃO VIÁVEL** para o caso de uso primário.
- **USGS Produced Waters / WoGSS:** química de água produzida e estatísticas operacionais. Sem logs raw. **NÃO VIÁVEL.**

## 3. Datasets Adicionais

### 3.1 Penobscot 3D (CNSOPB, Nova Scotia) — **NÃO VIÁVEL**

URL: `https://terranubis.com/datainfo/Penobscot` · Licença CC BY-SA. Dataset principalmente sísmico 3D (87 km², 601 inlines × 482 crosslines). Apenas **2 poços (B-41, L-30, 1976)** com logs convencionais (wireline pré-LWD moderno). Sem propagation tools, sem multi-freq. Útil para sísmica/ML, **inadequado** para inversão EM-LWD.

### 3.2 Teapot Dome / RMOTC (DOE/Wyoming) — **PARCIAL**

URL: `https://wiki.seg.org/wiki/Teapot_dome_3D_survey` · `https://dataunderground.org/dataset/teapot-dome` · Domínio público (DOE). 37 poços no Tensleep, 31 com logs LAS (GR, RHOB, NPHI, **RD, RS** — só deep+shallow resistivity). **PARCIAL:** licença e formato excelentes, mas resistividade é **wireline single-freq dual-laterolog**, não LWD multi-freq propagation. Bom para **DTB/Picasso validation** e treino de modelos petrofísicos auxiliares (Archie).

### 3.3 ANP BDEP (Brasil) — **PARCIAL** (com fricção burocrática)

URL: `https://www.gov.br/anp/pt-br/assuntos/exploracao-e-producao-de-oleo-e-gas/dados-tecnicos/acervo-de-dados` · 3,3 PB · LAS + composite + AGP. Dados ficam confidenciais até **10 anos** após aquisição, depois públicos. **Acesso via solicitação** (pessoa física/jurídica, instituição de pesquisa gratuita). Inclui poços do pré-sal e bacias maduras. **PARCIAL:** dados disponíveis após confidencialidade, mas requer fluxo de solicitação manual (não API), e qualidade/multi-freq varia por operador. Aderência geográfica BR é vantagem para autor brasileiro.

### 3.4 SPE/SPWLA / SDAR Benchmark — **VIÁVEL** (referência sintética)

OnePetro publica papers SPE com benchmarks sintéticos: **SDAR (Standardization of Deep Azimuthal Resistivity)** do SPWLA RtSIG e o caso canônico de **155ft@2kHz / 75ft@6-12kHz / 45ft@12-24kHz / 24in@2MHz** (data-driven interpretation paper, OSTI 1501648). **VIÁVEL** como conjunto de **modelos canônicos para validação cruzada** do simulador Fortran/Numba/JAX e do pipeline DL — paridade contra publicações peer-reviewed.

---

## Recomendação Sintética

| Dataset | Classe | Uso recomendado |
|:--------|:------:|:----------------|
| SDAR / SPWLA RtSIG benchmarks | **VIÁVEL** | Canonical models para paridade Fortran <1e-12 e validação DL |
| Equinor Volve | PARCIAL | Pre-training SurrogateNet, validação petrofísica (apenas pesquisa, NC) |
| Teapot Dome RMOTC | PARCIAL | DTB/Picasso, Archie, modelos auxiliares (domínio público) |
| ANP BDEP | PARCIAL | Validação geográfica BR após solicitação (10y waiting) |
| Penobscot 3D | NÃO VIÁVEL | Sísmico, não EM-LWD |
| USGS NGDS | NÃO VIÁVEL | Geotermia, sem propagation EM |
| OEDI/NREL · USGS PW · WoGSS | NÃO VIÁVEL | Domínio ortogonal |

**Ação proposta:** priorizar **SDAR/SPWLA** como ground-truth sintético institucional, complementar com **Volve+Teapot Dome** para pre-training/validação petrofísica, e considerar **ANP BDEP** só se justificar fluxo de solicitação. NGDS/OEDI/Penobscot fora do escopo.

---

## Próximos passos (após sair do plan mode)

1. Solicitar confirmação ao usuário se quer integrar SDAR canonical models como `tests/golden/` (paridade Fortran).
2. Avaliar se cria `docs/reference/datasets_publicos.md` com este sumário.
3. Verificar se precisa adaptar `geosteering_ai/data/loading.py` para LAS/DLIS (atualmente formato 22-col proprietário).
