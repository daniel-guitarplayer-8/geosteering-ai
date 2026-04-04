# Roadmap Completo — Geosteering AI v2.0+

**Versão do documento:** 1.0 (Abril 2026)
**Autor:** Daniel Leal
**Projeto:** Inversão 1D de Resistividade via Deep Learning para Geosteering
**Framework:** TensorFlow 2.x / Keras (exclusivo)
**Repositório:** `github.com/daniel-guitarplayer-8/geosteering-ai`

---

## Sumário

1. [Status Atual do Projeto](#1-status-atual-do-projeto-abril-2026)
2. [Embasamento Científico](#2-embasamento-científico-literatura-2024-2026)
3. [Roadmap de Desenvolvimento (F1-F6)](#3-roadmap-de-desenvolvimento-fases-f1-f6)
4. [Propostas de Novos Recursos](#4-propostas-de-novos-recursos)
5. [Otimizações de Código](#5-otimizações-de-código-recomendadas)
6. [Próximos Passos](#6-próximos-passos-requisições-futuras)
7. [Referências Bibliográficas](#7-referências-bibliográficas)
8. [Arquivos Críticos](#8-arquivos-críticos-para-consulta)

---

## 1. Status Atual do Projeto (Abril 2026)

### 1.1 Métricas Globais

| Métrica | Valor |
|:--------|:------|
| Linhas de código (produção) | 44.762 LOC |
| Linhas de testes | 9.024 LOC |
| Arquivos Python (src) | 73 |
| Arquivos Python (testes) | 15 |
| Testes totais | 744 (CPU) / 1011+ (GPU) |
| Campos PipelineConfig | 246 |
| Arquiteturas (ModelRegistry) | 48 (9 famílias) |
| Funções de perda | 26 (4 categorias) |
| Tipos de ruído | 34 implementados |
| Feature Views | 7 (identity, raw, H1_logH2, logH1_logH2, 3× razão/fase, second_order) |
| Famílias de Geosinais | 5 (USD, UAD, UHR, UHA, U3DF) |
| Cenários PINN | 8 (oracle, surrogate, maxwell, smoothness, skin_depth, continuity, variational, self_adaptive) |
| Presets YAML | 7 (baseline, robusto, nstage_n2, nstage_n3, geosinais_p4, dtb_p5, realtime_causal) |
| Callbacks Keras | 17+ |
| **Simulador Fortran** | 6.859 LOC (6 módulos F08) + gerador Python (~900 LOC) |
| **Documentação Simulador** | 6.558 LOC (17 seções + 6 pipelines, v4.0 com pipelines Fortran A+B e Python A+B+C+D) |
| **Formulação Teórica TeX** | `Tex_Projects/TatuAniso/FormulaçãoTatuAnisoTIV.tex` (~960 LOC TeX) |
| Métricas customizadas | 3 (R2Score, PerComponentMetric, AnisotropyRatioError) |
| Formatos de exportação | 3 (SavedModel, TFLite, ONNX) |

### 1.2 Arquitetura do Pacote

```
geosteering_ai/                           # 73 arquivos, 44.762 LOC
├── config.py                             # PipelineConfig (246 campos, 7 presets)
├── data/ (12 módulos, ~6.700 LOC)
│   ├── loading.py                        # Carregamento .dat 22-col Fortran
│   ├── splitting.py                      # Split por modelo geológico (P1)
│   ├── feature_views.py                  # 7 Feature Views (NumPy + TF)
│   ├── geosignals.py                     # 5 famílias Geosinais (NumPy + TF)
│   ├── scaling.py                        # 8 scalers + per-group P3
│   ├── pipeline.py                       # DataPipeline on-the-fly (noise→FV→GS→scale)
│   ├── boundaries.py                     # DTB labels (P5)
│   ├── sampling.py                       # Oversampling alta rho (Estratégia B)
│   ├── second_order.py                   # Features 2º grau (Estratégia C)
│   ├── surrogate_data.py                 # Extração pares SurrogateNet (Modo A/B/C)
│   └── inspection.py                     # Diagnóstico pré-treinamento
├── noise/ (2 módulos, ~2.600 LOC)
│   ├── functions.py                      # 34 tipos de ruído
│   └── curriculum.py                     # Curriculum 3-phase (clean→ramp→stable)
├── models/ (13 módulos, ~8.800 LOC)
│   ├── blocks.py                         # 23+ blocos Keras reutilizáveis
│   ├── cnn.py                            # ResNet-18/34/50, ConvNeXt, Inception, CNN_1D, ResNeXt
│   ├── tcn.py                            # TCN, TCN_Advanced, ModernTCN
│   ├── rnn.py                            # LSTM, BiLSTM
│   ├── hybrid.py                         # CNN_LSTM, CNN_BiLSTM_ED, ResNeXt_LSTM
│   ├── unet.py                           # 14 variantes U-Net
│   ├── transformer.py                    # 6 Transformers
│   ├── decomposition.py                  # N-BEATS, N-HiTS
│   ├── advanced.py                       # DNN, FNO, DeepONet, Geophysical_Attention
│   ├── geosteering.py                    # WaveNet, Causal_Transformer, Informer, Mamba_S4
│   ├── surrogate.py                      # SurrogateNet TCN (campo receptivo ~127m)
│   └── registry.py                       # ModelRegistry — 48 entradas + build()
├── losses/ (3 módulos, ~3.600 LOC)
│   ├── catalog.py                        # 26 funções de perda
│   ├── factory.py                        # LossFactory.build_combined()
│   └── pinns.py                          # 8 cenários PINN + TIV constraint + lambda schedule
├── training/ (7 módulos, ~8.100 LOC)
│   ├── loop.py                           # TrainingLoop.run()
│   ├── callbacks.py                      # 17+ callbacks Keras
│   ├── metrics.py                        # R2Score, PerComponentMetric, AnisotropyRatioError
│   ├── nstage.py                         # NStageTrainer (N=2,3,4)
│   ├── adaptation.py                     # DomainAdapter (fine-tune + progressive)
│   └── optuna_hpo.py                     # HPO com Optuna
├── inference/ (4 módulos, ~1.800 LOC)
│   ├── pipeline.py                       # InferencePipeline serializável (save/load)
│   ├── realtime.py                       # Sliding window (causal)
│   ├── export.py                         # SavedModel, TFLite, ONNX
│   └── uncertainty.py                    # MC Dropout + Ensemble
├── evaluation/ (11 módulos, ~7.300 LOC)
│   ├── metrics.py                        # MetricsReport (R², RMSE, MAE, MBE, MAPE)
│   ├── comparison.py                     # Ranking multi-modelo
│   ├── advanced.py                       # InterfaceReport, CoherenceReport, StabilityReport
│   ├── dod.py                            # Picasso DOD analítico (6 métodos)
│   ├── geosteering_metrics.py            # DTB error, look-ahead accuracy
│   ├── realtime_comparison.py            # Offline vs Realtime
│   ├── report.py                         # Relatório Markdown automático
│   ├── manifest.py                       # Manifesto JSON reprodutibilidade
│   └── config_report.py                  # Relatório pré-treinamento
├── visualization/ (11 módulos, ~2.500 LOC)
│   ├── eda.py                            # EDA avançada (6 funções)
│   ├── holdout.py                        # True vs predicted (clean + noisy)
│   ├── training.py                       # Curvas loss, R², LR, noise
│   ├── error_maps.py                     # Heatmap 2D, barras, perfil espacial
│   ├── geosteering.py                    # Curtain, DTB, dashboard
│   ├── uncertainty.py                    # Histogramas, bandas, calibração
│   ├── optuna_viz.py                     # Optuna (4 visualizações)
│   └── export.py                         # Exportação batch (PNG, PDF)
└── utils/ (6 módulos, ~2.500 LOC)
    ├── logger.py                         # ColoredFormatter, setup_logger
    ├── timer.py                          # ProgressTracker, elapsed_since
    ├── validation.py                     # ValidationTracker, validate_shape
    ├── formatting.py                     # format_number, log_header
    ├── system.py                         # is_colab, get_gpu_info, set_seed
    └── io.py                             # load_json, save_json, safe_mkdir
```

### 1.3 Funcionalidades Implementadas

| Categoria | Funcionalidade | Status |
|:----------|:---------------|:------:|
| **Dados** | Carregamento .dat 22-col Fortran | ✅ |
| **Dados** | Split por modelo geológico (P1) — evita data leakage | ✅ |
| **Dados** | 7 Feature Views (NumPy + TF consistentes, SEMPRE log10) | ✅ |
| **Dados** | 5 famílias Geosinais — USD, UAD, UHR, UHA, U3DF (NumPy + TF) | ✅ |
| **Dados** | 8 scalers (Standard, Robust, MinMax, etc.) + per-group P3 | ✅ |
| **Dados** | DataPipeline on-the-fly — GS veem ruído (fidelidade LWD) | ✅ |
| **Dados** | DTB labels — detect_boundaries + compute_dtb (P5) | ✅ |
| **Dados** | Oversampling alta rho (Estratégia B) | ✅ |
| **Dados** | Features 2º grau — |H|², d|H|/dz, Re/Im ratio (Estratégia C) | ✅ |
| **Dados** | SurrogateDataset — extração pares Modo A/B/C (K componentes) | ✅ |
| **Ruído** | 34 tipos (9 CORE + 13 EXTENDED + 8 geofísicos + 4 geosteering) | ✅ |
| **Ruído** | Curriculum 3-phase (clean → ramp → stable) | ✅ |
| **Modelos** | 48 arquiteturas em 9 famílias (CNN, TCN, RNN, Hybrid, U-Net, Transformer, Decomposition, Advanced, Geosteering) | ✅ |
| **Modelos** | 23+ blocos Keras reutilizáveis | ✅ |
| **Modelos** | SurrogateNet TCN dilatado (6 blocos, campo receptivo ~127m) | ✅ |
| **Modelos** | SurrogateNet v2 ModernTCN (4 blocos DWConv, campo receptivo ~204m) | ✅ |
| **Modelos** | ModernTCN — convolução pura moderna para inversão seq2seq (Luo 2024) | ✅ |
| **Modelos** | INN — Invertible Neural Network para UQ probabilística (Ardizzone 2019) | ✅ |
| **Modelos** | ResNeXt — grouped convolutions 32×4d bottleneck (Xie 2017) | ✅ |
| **Modelos** | ResNeXt_LSTM — híbrido ResNeXt encoder + LSTM temporal | ✅ |
| **Inference** | UQ via INN (sampling da posterior, alternativa ao MC Dropout) | ✅ |
| **Modelos** | Static injection P2/P3 (broadcast, dual_input, FiLM) | ✅ |
| **Modelos** | TIVConstraintLayer (hard constraint rho_v ≥ rho_h via softplus) | ✅ |
| **Losses** | 26 funções de perda (13 genéricas + 4 geofísicas + 2 geosteering + 7 avançadas) | ✅ |
| **Losses** | 8 cenários PINN + 4 lambda schedules (fixed, linear, cosine, step) | ✅ |
| **Losses** | Forward analítico 1D (magnitude log10|H| + complexo Re/Im) | ✅ |
| **Losses** | TIV constraint loss (penaliza rho_h > rho_v) | ✅ |
| **Losses** | LossFactory.build_combined() — base + look_ahead + DTB + PINNs | ✅ |
| **Treinamento** | TrainingLoop.run() + N-Stage (N=2,3,4) com mini-curriculum | ✅ |
| **Treinamento** | 17+ callbacks (DualValidation, PINN schedule, Causal, LR schedules) | ✅ |
| **Treinamento** | HPO Optuna com visualizações | ✅ |
| **Treinamento** | DomainAdapter (fine-tune + progressive) | ✅ |
| **Treinamento** | GradientMonitor (normas por layer via GradientTape) | ✅ |
| **Inferência** | InferencePipeline serializável (model + scalers + config → save/load) | ✅ |
| **Inferência** | Realtime sliding window (causal, buffer FIFO) | ✅ |
| **Inferência** | Export SavedModel / TFLite (com quantização) / ONNX | ✅ |
| **Inferência** | UQ — MC Dropout (30 passes) + Ensemble + CI 95% | ✅ |
| **Avaliação** | Métricas (R², RMSE, MAE, MBE, MAPE por componente) | ✅ |
| **Avaliação** | DOD Picasso analítico (6 métodos: standard, contrast, SNR, frequency, anisotropy, dip) | ✅ |
| **Avaliação** | Relatórios Markdown automáticos + Manifesto JSON | ✅ |
| **Avaliação** | Comparação offline vs realtime + geosteering metrics | ✅ |
| **Visualização** | EDA, holdout, training curves, error maps, geosteering, UQ, Optuna | ✅ |
| **Compatibilidade** | Keras 3.x (8 fixes: KerasTensor, DepthwiseConv1D, FourierLayer, etc.) | ✅ |
| **CI/CD** | GitHub Actions (compile + pytest + mypy, Python 3.10/3.11) | ✅ |
| **Validação GPU** | 824 passed no Colab Pro+ (TF 2.19, Keras 3.x, GPU T4) | ✅ |
| **Documentação** | Padronização seq_len — remoção de hardcoded 600 em docstrings e comentários (~30 ocorrências corrigidas em 15 módulos) | ✅ |
| **Documentação** | Mega-headers atualizados com contagens corretas (8 CNN, 3 TCN, 3 Hybrid, 5 Advanced) | ✅ |
| **Documentação** | Manual técnico de ruídos — `docs/documentacao_noises.md` (34 tipos, curriculum, API) | ✅ |
| **Documentação** | Manual técnico de losses — `docs/documentacao_losses.md` (26 losses, 8 PINN, TIV) | ✅ |
| **Documentação** | Manual técnico de modelos — `docs/documentacao_models.md` (48 arqs, 9 famílias, blocos) | ✅ |
| **Documentação** | Manual técnico de inferência offline — `docs/documentacao_inferencia_offline.md` (pipeline, UQ, export) | ✅ |
| **Documentação** | Manual técnico de geosteering — `docs/documentacao_geosteering.md` (5 arqs causais, realtime) | ✅ |
| **Documentação** | Manual técnico de PINNs — `docs/documentacao_pinns.md` (8 cenários, TIV, SurrogateNet) | ✅ |
| **Documentação** | Apresentação geral do projeto — `docs/documentacao_apresentacao_geosteering_ai.md` | ✅ |

### 1.4 Lacunas Reconhecidas

| Lacuna | Impacto | Dependência |
|:-------|:--------|:------------|
| Treino do SurrogateNet com dados reais | Alto — PINNs modo surrogate neural | Dados Fortran |
| Surrogate Modo C (tensor completo 9 componentes, 18 canais) | Médio — digital twin Fortran | Fortran multi-dip |
| Notebook de treinamento end-to-end | Alto — pipeline completo validado | Dados disponíveis |
| Multi-Task Loss (Kendall et al. 2018) — placeholder | Baixo — auto-balanceamento | Implementação |
| Cenário PINN petrofísica (Archie + Klein) | Baixo — requer parâmetros reservatório | Dados de campo |
| Evidential Regression (alternativa a MC Dropout para UQ) | Baixo — MC Dropout funciona | Pesquisa |

---

## 2. Embasamento Científico (Literatura 2024-2026)

### 2.1 Artigos-Chave

| # | Referência | Contribuição para o Projeto |
|:-:|:-----------|:---------------------------|
| 1 | Morales et al. (2025) — *Anisotropic resistivity estimation and uncertainty quantification from borehole triaxial electromagnetic induction measurements: Gradient-based inversion and physics-informed neural network.* Computers & Geosciences, 196, 105786. | Base teórica dos cenários PINN. PINN estima propriedades petrofísicas em 0,5 ms com 91-99% de acurácia vs minutos com inversão baseada em gradiente. Validado com medidas triaxiais. **Já referenciado no projeto.** |
| 2 | INN-UDAR (2025) — *Invertible neural network for real-time inversion and uncertainty quantification of ultra-deep resistivity measurements.* Computers & Geosciences. | Redes invertíveis (INN) fornecem distribuição posterior completa para quantificação de incerteza probabilística em tempo real, sem necessidade de múltiplas forward passes (MC Dropout). **Implementado: INN como arquitetura #45 em `models/advanced.py` + UQ em `inference/uncertainty.py`.** |
| 3 | SPE/SPWLA (2021) — *Real-Time 2.5D Inversion of LWD Resistivity Measurements Using Deep Learning for Geosteering Applications Across Faulted Formations.* | Deep learning para inversão 2.5D com custo online desprezível. Validado com ferramenta LWD triaxial em formações anisotrópicas com falhas. **Referência para extensão futura (v3.0).** |
| 4 | Li et al. (2025) — *Self-Supervised Deep Learning Inversion Incorporating a Fast Forward Network for Transient Electromagnetic Data.* JGR: Machine Learning and Computation. | Inversão auto-supervisionada que incorpora rede forward rápida diretamente na função de perda. **Alinha com cenário "surrogate" PINN do projeto.** |
| 5 | Jiang et al. (2025) — *One-Fit-All Transformer for Multimodal Geophysical Inversion.* JGR: Machine Learning and Computation. | Framework G-Query — Transformer unificado que adapta entre múltiplas modalidades geofísicas via query tokens. **Proposta: avaliar como arquitetura #46 (G_Query_Transformer).** |
| 6 | Frontiers (2025) — *Fast forward modeling and response analysis of extra-deep azimuthal resistivity measurements in complex model.* | Modelagem direta rápida para UDAR (Ultra-Deep Azimuthal Resistivity) em modelos complexos. **Complementa o surrogate neural do projeto.** |
| 7 | ModernTCN (ICLR 2024) — *A Modern Pure Convolution Structure for General Time Series Analysis.* | Modernização do TCN clássico com patch embedding e channel mixing, superando TCN em benchmarks de séries temporais. **Implementado: ModernTCN em `models/tcn.py` + SurrogateNet v2 em `models/surrogate.py`.** |
| 8 | Physics-guided AEM inversion (GJI 2024) — *Physics-guided deep learning-based inversion for airborne electromagnetic data.* | PGNN incorpora leis físicas governantes diretamente na função de perda para inversão EM aerotransportada. **Valida a abordagem PINN adotada pelo projeto.** |
| 9 | Noh, K., Pardo, D., Torres-Verdín, C. (2023) — *Physics-guided deep-learning inversion method for the interpretation of noisy logging-while-drilling resistivity measurements.* Geophysical Journal International, 235. | Combina constraints físicas com DL para inversão de medidas LWD ruidosas. Diretamente aplicável aos cenários PINN do projeto com injeção de ruído. **PDF disponível em `PDFs/ggad217.pdf`.** |
| 10 | Puzyrev, V. (2019) — *Deep learning electromagnetic inversion with convolutional neural networks.* Geophysical Journal International, 218. | Demonstra viabilidade de CNNs para inversão EM em tempo real — base histórica para a abordagem DL do projeto. **PDF disponível em `PDFs/ggz204.pdf`.** |
| 11 | Guo, W. et al. (2024) — *Efficient 1D Modeling of Triaxial Electromagnetic Logging in Uniaxial and Biaxial Anisotropic Formations Using Virtual Boundaries and Equivalent Resistivity Schemes.* Journal of Geophysics and Engineering. | Algoritmo de propagação eficiente para forward 1D triaxial em formações TIV — relevante para o surrogate analítico e validação do SurrogateNet. **PDF disponível em `PDFs/gxag017.pdf`.** |
| 12 | Liu, W. et al. (2022) — *Physics-Driven Deep Learning Inversion with Application to Magnetotelluric.* Remote Sensing, 14, 3218. | Propõe operador físico forward integrado ao treinamento DL — valida a abordagem physics-driven do projeto. **PDF disponível em `PDFs/remotesensing-14-03218-v2.pdf`.** |
| 13 | Liu, W. et al. (2024) — *Physics-Informed Deep Learning Inversion with Application to Noisy Magnetotelluric Measurements.* Remote Sensing, 16, 62. | Estende abordagem com Swin Transformer e estratégias de noise injection para dados EM ruidosos de campo — alinha diretamente com o curriculum de ruído do projeto. **PDF disponível em `PDFs/remotesensing-16-00062-v2.pdf`.** |
| 14 | Constable, M. V. et al. (2016) — *Looking Ahead of the Bit While Drilling: From Vision to Reality.* Petrophysics, 57(5). | Apresenta a ferramenta EMLA para detecção de contrastes de resistividade à frente da broca, com DOI > 75 m. Contexto aplicado da ferramenta LWD que o projeto pretende substituir por DL. **PDF disponível em `PDFs/Constable_et_al_2016_Petrophysics.pdf`.** |
| 15 | Wang, L. et al. (2018) — *Sensitivity analysis and inversion processing of azimuthal resistivity logging-while-drilling measurements.* Journal of Geophysics and Engineering, 15, 2339. | Inversão Gauss-Newton com janela deslizante 1D para medidas ARM. Referência de algoritmo convencional que o projeto busca substituir via DL em tempo real. **PDF disponível em `PDFs/Wang_2018_J._Geophys._Eng._15_2339.pdf`.** |
| 16 | Guoyu Li et al. (2025) — *Optimization and Analysis of Sensitive Areas for Look-Ahead Electromagnetic Logging-While-Drilling Based on Geometric Factors.* Energies, 18, 3014. | Análise de áreas sensíveis para look-ahead EM LWD via fatores geométricos — relevante para design de features e avaliação DOD do projeto. **PDF disponível em `PDFs/Guoyu_et_al_2025_FG.pdf`.** |
| 17 | Bai, J. et al. (2022) — *An Introduction to Programming Physics-Informed Neural Networks for Computational Solid Mechanics.* arXiv:2210.09060. | Tutorial prático de implementação de PINNs com TensorFlow/PyTorch — referência pedagógica para os cenários PINN implementados em `losses/pinns.py`. **PDF disponível em `PDFs/2210.09060v4.pdf`.** |

### 2.2 Tendências Identificadas

1. **Redes Invertíveis (INN / Normalizing Flows)** para quantificação de incerteza probabilística — alternativa superior ao MC Dropout por fornecer a distribuição posterior completa em uma única forward pass.

2. **Auto-supervisão com forward model neural** integrado na função de perda — elimina necessidade de labels supervisionados, alinhando com o cenário "surrogate" PINN já implementado no projeto.

3. **Transformers multimodais** para inversão geofísica — framework G-Query demonstra que um único modelo pode adaptar entre EM, sísmica e gravimetria via tokens de query especializados.

4. **ModernTCN** supera o TCN clássico em benchmarks de séries temporais, mantendo a arquitetura puramente convolucional (sem atenção) com menor custo computacional.

5. **Inversão 2.5D/3D via Deep Learning** em formações complexas (falhas, intrusões salinas, dip variável) — próxima fronteira para o pipeline, atualmente limitado a 1D.

6. **Quantificação de incerteza como requisito obrigatório** para decisões operacionais em geosteering — não é mais opcional, é mandatório para uso em tempo real.

7. **Robustez a ruído como critério de avaliação primário** — Liu et al. (2024) e Noh et al. (2023) mostram que modelos treinados sem estratégias de noise injection falham em dados de campo reais, reforçando o curriculum de 34 tipos de ruído do projeto.

8. **Forward modeling eficiente como habilitador** — algoritmos rápidos para 1D triaxial (Guo et al. 2024) são pré-requisito para treinar surrogates em escala e para o ciclo de auto-supervisão.

---

## 3. Roadmap de Desenvolvimento (Fases F1-F6)

### Visão Geral

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ROADMAP GEOSTEERING AI v2.0+                                              │
│                                                                             │
│  F1 ─ Consolidação e Commit (imediato)                                     │
│  ├── Commit pendente (Passos 1+2 surrogate)                               │
│  ├── Atualização MEMORY.md                                                 │
│  └── Documentação roadmap (este documento)                                 │
│                                                                             │
│  F2 ─ Treinamento e Validação GPU (curto prazo)                           │
│  ├── Notebook train_surrogate.ipynb                                        │
│  ├── Notebook train_baseline.ipynb                                         │
│  ├── Benchmark 48 arquiteturas no Colab                                    │
│  ├── Validação end-to-end com dados Fortran                               │
│  └── Preset surrogate_mode_a.yaml                                          │
│                                                                             │
│  F3 ─ Otimização de Performance (médio prazo)                             │
│  ├── XLA/JIT compilation (2-3× speedup)                                   │
│  ├── Mixed precision float16 (1.5-2× speedup)                             │
│  ├── Profiling TensorBoard                                                 │
│  ├── Benchmark latência por arquitetura                                    │
│  └── Otimização DataPipeline (prefetch, cache)                            │
│                                                                             │
│  F4 ─ Expansão Científica (médio prazo)                                   │
│  ├── INN (Invertible Neural Network) — arquitetura #45                    │
│  ├── G-Query Transformer multimodal — arquitetura #46                     │
│  ├── ModernTCN para SurrogateNet v2                                        │
│  ├── Multi-Task Loss (Kendall et al. 2018)                                │
│  ├── Cenário PINN petrofísica (Archie)                                    │
│  └── Evidential Regression para UQ                                         │
│                                                                             │
│  F5 ─ Dados Multi-Dip e Modo C (longo prazo)                             │
│  ├── Re-simulação Fortran multi-dip (0°-90°)                             │
│  ├── Treino SurrogateNet Modo B/C                                          │
│  ├── Surrogate Transformer (se TCN insuficiente)                          │
│  └── Validação física H_surrogate vs H_Fortran                            │
│                                                                             │
│  F6 ─ Produção e Deploy (longo prazo)                                     │
│  ├── API REST (FastAPI/gRPC)                                               │
│  ├── Integração WITSML/OPC-UA                                             │
│  ├── Dashboard web (Streamlit/Grafana)                                     │
│  ├── Containerização (Docker + K8s)                                        │
│  └── Monitoramento MLOps (MLflow/Weights&Biases)                          │
│                                                                             │
│  F7 ─ Simulador Python Otimizado (médio-longo prazo)                       │
│  ├── Reimplementação do PerfilaAnisoOmp em Python (Numba JIT)             │
│  ├── Kernels CUDA via Numba para GPU (commonarraysMD + Hankel)            │
│  ├── Integração direta com geosteering_ai/simulation/                     │
│  ├── Geração on-the-fly de dados sintéticos durante treinamento           │
│  └── Validação numérica Python vs Fortran (erro < 1e-10)                  │
│                                                                             │
│  v3.0 ─ Inversão 2D/3D (futuro)                                           │
│  ├── Extensão para formações com falhas                                    │
│  ├── Múltiplas frequências simultâneas                                     │
│  └── Digital twin completo do simulador Fortran                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### F1 — Consolidação e Commit (Imediato)

**Objetivo**: Commitar trabalho pendente e estabilizar a base de código.

| Passo | Descrição | Prioridade |
|:------|:----------|:----------:|
| F1.1 | Rodar `pytest tests/ -v --tb=short` — validar 802+ testes | Alta |
| F1.2 | Commit dos Passos 1+2 (surrogate_data.py + surrogate.py + testes) | Alta |
| F1.3 | Reescrever MEMORY.md (<150 linhas, atualizado) | Média |
| F1.4 | Gerar `docs/ROADMAP.md` (este documento) | Média |

**Verificação**: `pytest` green, `git log --oneline -3` mostra commit.

---

### F2 — Treinamento e Validação GPU (Curto Prazo)

**Objetivo**: Validar o pipeline end-to-end com treinamento real no Colab Pro+ GPU.

| Passo | Descrição | Artefato | Prioridade |
|:------|:----------|:---------|:----------:|
| F2.1 | Criar `notebooks/train_surrogate.ipynb` | Notebook Colab | Alta |
| | — Carregar dados .dat (Inv0Dip 0°, existentes) | | |
| | — `extract_surrogate_pairs()` Modo A (XX, ZZ → 4 canais) | | |
| | — Split por modelo geológico (P1) | | |
| | — `build_surrogate(config)` → `model.compile()` → `model.fit()` | | |
| | — Salvar como `surrogate_mode_a.keras` | | |
| F2.2 | Criar `configs/surrogate_mode_a.yaml` | YAML | Alta |
| F2.3 | Criar `notebooks/train_baseline.ipynb` | Notebook Colab | Alta |
| | — Pipeline completo: load → split → train → eval → report | | |
| | — Preset `robusto.yaml` (curriculum 3-phase, noise 8%) | | |
| | — Gerar MetricsReport + figuras (EDA, holdout, training) | | |
| F2.4 | Criar `notebooks/benchmark_architectures.ipynb` | Notebook Colab | Média |
| | — Forward + backward pass para 44 arquiteturas no GPU | | |
| | — Medir: latência (ms), memória (MB), parâmetros (M) | | |
| | — Ranking por trade-off R² × latência × parâmetros | | |
| F2.5 | Validação surrogate analítico vs Fortran | Métricas | Alta |
| | — Comparar H_analítico vs H_Fortran para 100+ modelos | | |
| | — RMSE por componente (Re/Im × XX/ZZ) | | |

#### F2.5.1 — CPU Optimization Roadmap (Simulador Fortran PerfilaAnisoOmp)

Roteiro de 6 fases para otimização do simulador Fortran conforme [`docs/reference/analise_paralelismo_cpu_fortran.md`](reference/analise_paralelismo_cpu_fortran.md) §7. Relatório de execução das Fases 0–1 em [`docs/reference/relatorio_fase0_fase1_fortran.md`](reference/relatorio_fase0_fase1_fortran.md).

| Fase | Descrição | Status | Ganho Real / Esperado |
|:----:|:----------|:------:|:----------------------|
| **Fase 0** | Benchmark Baseline (wall-time, MD5, infra `bench/`) | ✅ **Concluída 2026-04-04** | Baseline publicado: **0,1047 s/modelo**, **~34.400 mod/h** (i9-9980HK, 8 threads, AVX-2) |
| **Fase 1** | SIMD Hankel Reduction (`!$omp simd` em `commonarraysMD`) | ⏭️ **Pulada 2026-04-04** | Δ +0,96 % (Welch *t*=+0,425, não-significativo). Causa: gfortran 15.x já auto-vetoriza em AVX-2 32-byte. Experimento arquivado em `Fortran_Gerador/bench/attic/`. Re-avaliar em AVX-512. |
| **Fase 2** | Hybrid Scheduler (static/dynamic/guided conforme `ntheta × nmed`) | 🔜 **Próxima** | +5–15 % esperado. Também corrige bug de anti-escalabilidade em 2 threads (débito técnico descoberto na Fase 0). |
| **Fase 3** | Workspace Pre-allocation (`thread_workspace` tipo) | 📋 Planejada | +40–80 % esperado. Pré-requisito estrutural para GPU. |
| **Fase 4** | Cache de `commonarraysMD` por `(r, freq)` — 1200 → 2 chamadas/modelo | 📋 Planejada | **+60–120 % — maior ganho do roteiro** |
| **Fase 5** | `collapse(3)` nos loops `theta × medidas × freq` | 📋 Planejada | +10–20 % adicional |
| **Fase 6** | Cache de `commonfactorsMD` por `camadT` | 📋 Planejada | +15–25 % adicional |

**Débitos técnicos descobertos durante Fase 0** (correção em fases futuras):

1. [`writes_files` em PerfilaAnisoOmp.f08:245-246](../Fortran_Gerador/PerfilaAnisoOmp.f08) usa `position='append'` sem limpeza — concatena em re-execuções (prioridade **Alta**).
2. [`omp_set_nested(.true.)` em PerfilaAnisoOmp.f08:74](../Fortran_Gerador/PerfilaAnisoOmp.f08) depreciado desde OpenMP 5.0 — migrar para `omp_set_max_active_levels(2)` (prioridade Média).
3. [`num_threads_j = maxthreads - ntheta` em PerfilaAnisoOmp.f08:85](../Fortran_Gerador/PerfilaAnisoOmp.f08) produz 1 thread no nível interno quando `OMP_NUM_THREADS=2` — endereçada pela **Fase 2** (prioridade Alta).

**Verificação**: Notebooks executam sem erro no Colab Pro+ (T4/A100). SurrogateNet Mode A converge com val_loss decrescente. Simulador Fortran com baseline CPU publicado e infra `bench/` operacional.

---

### F3 — Otimização de Performance (Médio Prazo)

**Objetivo**: Maximizar velocidade de treinamento e inferência.

| Passo | Descrição | Ganho Esperado | Prioridade |
|:------|:----------|:---------------|:----------:|
| F3.1 | **XLA/JIT compilation** | 2-3× speedup treinamento | Alta |
| | — `model.compile(jit_compile=True)` | | |
| | — Verificar compatibilidade com as 48 arquiteturas | | |
| | — Campo `config.use_xla` já existe no PipelineConfig | | |
| F3.2 | **Mixed precision (float16)** | 1.5-2× speedup + 50% menos memória GPU | Alta |
| | — `tf.keras.mixed_precision.set_global_policy('mixed_float16')` | | |
| | — Verificar estabilidade numérica (EPS=1e-12, gradientes, losses) | | |
| | — Campo `config.use_mixed_precision` a adicionar | | |
| F3.3 | **Profiling TensorBoard** | Identificação de gargalos | Média |
| | — `tf.profiler.experimental.start()` em sessão de treino | | |
| | — Análise de utilização GPU (kernels, memória, I/O) | | |
| F3.4 | **Benchmark latência de inferência** | Tabela por arquitetura | Média |
| | — Medir tempo/amostra (ms) para 27 arquiteturas causais | | |
| | — Identificar candidatas para realtime (<10ms) | | |
| | — Comparar TFLite vs SavedModel vs ONNX | | |
| F3.5 | **Otimização DataPipeline** | Reduzir I/O overhead | Média |
| | — `tf.data.Dataset.cache()` para datasets que cabem em RAM | | |
| | — `prefetch(tf.data.AUTOTUNE)` para sobreposição I/O-compute | | |
| | — `interleave()` para carregamento paralelo | | |
| F3.6 | **Quantização TFLite** | 4× redução de tamanho | Baixa |
| | — Post-training quantization (int8 pesos, float32 ativações) | | |
| | — Validar RMSE degradation < 5% pós-quantização | | |

**Verificação**: Speedup mensurável documentado. Latência < 10ms para top-5 arquiteturas causais.

---

### F4 — Expansão Científica (Médio Prazo)

**Objetivo**: Incorporar avanços recentes da literatura (2024-2026) ao pipeline.

#### F4.1 — Invertible Neural Network (INN) — Arquitetura #45

**Base científica**: INN-UDAR (Computers & Geosciences, 2025). Redes invertíveis fornecem distribuição posterior completa para UQ probabilística, sem necessidade de MC Dropout.

| Item | Descrição |
|:-----|:----------|
| Módulo | `models/advanced.py` — adicionar `build_inn()` |
| Registro | `registry.py` → entrada "INN" na ModelRegistry |
| Testes | `test_models.py` → forward pass + verificação de invertibilidade |
| Config | `model_type: "INN"` aceito pelo PipelineConfig |
| Referência | Ardizzone et al. (2019) + INN-UDAR (2025) |

#### F4.2 — G-Query Transformer Multimodal — Arquitetura #46

**Base científica**: Jiang et al. (2025) JGR:ML. Framework "One-Fit-All" que adapta entre modalidades geofísicas via query tokens especializados.

| Item | Descrição |
|:-----|:----------|
| Módulo | `models/transformer.py` — adicionar `build_gquery_transformer()` |
| Registro | `registry.py` → entrada "G_Query_Transformer" |
| Diferencial | Multi-head cross-attention entre componentes EM (XX, ZZ, XZ, ZX) |

#### F4.3 — ModernTCN para SurrogateNet v2

**Base científica**: ModernTCN (ICLR 2024). Modernização do TCN com patch embedding + depthwise separable + channel mixing.

| Item | Descrição |
|:-----|:----------|
| Módulo | `models/surrogate.py` — adicionar `build_surrogate_modern()` |
| Ganho | Melhor captura de dependências longas com menos parâmetros |
| Config | `surrogate_architecture: "tcn"` ou `"modern_tcn"` |

#### F4.4 — Multi-Task Loss (Kendall et al. 2018) — Completar #22

**Base científica**: "Multi-Task Learning Using Uncertainty to Weigh Losses" — pesos automáticos via variáveis treináveis log_sigma por tarefa.

| Item | Descrição |
|:-----|:----------|
| Módulo | `losses/catalog.py` → completar `make_multitask()` (atualmente placeholder) |
| Fórmula | `L = Σ (0.5/σ_i²) × L_i + 0.5 × log(σ_i²)` |
| Aplicação | Balancear rho_h + rho_v + DTB automaticamente |

#### F4.5 — Cenário PINN Petrofísica (Archie + Klein)

**Base científica**: Morales et al. (2025) + Lei de Archie (1942). Integrar constraints petrofísicas na função de perda.

| Item | Descrição |
|:-----|:----------|
| Módulo | `losses/pinns.py` → cenário #9 "petrophysics" |
| Constraint | `rho_t = a × Rw / (phi^m × Sw^n)` |
| Dependência | Parâmetros do reservatório (Rw, phi, m, n) como inputs adicionais |

#### F4.6 — Evidential Regression para UQ

**Base científica**: Amini et al. (2020) — "Deep Evidential Regression". UQ em uma única forward pass sem MC Dropout.

| Item | Descrição |
|:-----|:----------|
| Módulo | `inference/uncertainty.py` → método "evidential" |
| Output | (μ, σ², ν, α) por ponto — incerteza aleatória + epistêmica |
| Vantagem | ~30× mais rápido que MC Dropout (1 pass vs 30 passes) |

**Verificação**: Cada nova feature tem testes unitários + validação no Colab GPU.

---

### F5 — Dados Multi-Dip e Modo C (Longo Prazo)

**Objetivo**: Treinar SurrogateNet com tensor EM completo para múltiplos ângulos de dip.

**Contexto**: As componentes off-diagonais do tensor H (Hxz, Hzx, Hxy, etc.) são estruturalmente zero em dip=0° por simetria TIV. Para treinar o Modo C (9 componentes, 18 canais), é necessário re-executar o simulador Fortran com múltiplos ângulos.

| Passo | Descrição | Dependência | Prioridade |
|:------|:----------|:------------|:----------:|
| F5.1 | Re-executar PerfilaAnisoOmp com multi-dip | Fortran + HPC | Alta |
| | — dip = [0°, 5°, 10°, 15°, 20°, 30°, 45°, 60°, 75°, 90°] | | |
| | — ~40.000 modelos × 600 pontos × 22 colunas por ângulo | | |
| | — Estimativa: ~10 GB de dados brutos | | |
| F5.2 | Treinar SurrogateNet Modo B (XX, ZZ, XZ, ZX → 8 canais) | F5.1 | Alta |
| | — Requer dados com dip ≥ 5° (componentes cruzadas ≠ 0) | | |
| | — `configs/surrogate_mode_b.yaml` | | |
| F5.3 | Treinar SurrogateNet Modo C (9 comp → 18 canais) | F5.1 | Média |
| | — Avaliar se TCN atual (~2M params) é suficiente ou precisa escalar | | |
| | — Alternativa: `build_surrogate_modern()` (ModernTCN, F4.3) | | |
| | — `configs/surrogate_mode_c.yaml` | | |
| F5.4 | Validação física H_surrogate vs H_Fortran | F5.2/F5.3 | Alta |
| | — RMSE < 1% por componente diagonal, < 5% por cruzada | | |
| | — Gráficos de comparação (Re/Im × componente × dip) | | |
| F5.5 | Integrar surrogate treinado nas PINNs | F5.4 | Alta |
| | — `make_surrogate_physics_loss()` com modo `"neural_external"` | | |
| | — Carregar SavedModel do surrogate treinado | | |

**Verificação**: Surrogate Modo B/C convergem. RMSE vs Fortran < 1% para diagonais, < 5% para cruzadas.

---

### F6 — Produção e Deploy (Longo Prazo)

**Objetivo**: Levar o pipeline para ambiente de produção industrial para uso em operações de geosteering em tempo real.

| Passo | Descrição | Tecnologia | Prioridade |
|:------|:----------|:-----------|:----------:|
| F6.1 | **API REST** para inferência | FastAPI + TF Serving | Alta |
| | — `POST /predict` — inversão batch | | |
| | — `POST /predict/realtime` — streaming com sliding window | | |
| | — `POST /uncertainty` — predição com MC Dropout / Ensemble | | |
| | — Documentação automática OpenAPI/Swagger | | |
| F6.2 | **Integração WITSML/OPC-UA** | python-witsml + opcua | Média |
| | — Consumir dados LWD em tempo real de sistemas SCADA | | |
| | — Converter formato WITSML → layout 22 colunas | | |
| F6.3 | **Dashboard web** | Streamlit ou Grafana | Média |
| | — Visualização realtime (curtain plot, DTB, bandas de incerteza) | | |
| | — Seleção de presets e configurações | | |
| | — Histórico de predições e métricas | | |
| F6.4 | **Containerização** | Docker + Kubernetes | Média |
| | — Dockerfile multi-stage (build + runtime, CPU + GPU) | | |
| | — Helm chart para deploy em cluster | | |
| | — Auto-scaling horizontal por demanda | | |
| F6.5 | **MLOps** | MLflow ou W&B | Baixa |
| | — Tracking de experimentos (hiperparâmetros, métricas, artefatos) | | |
| | — Model registry (versionamento de modelos treinados) | | |
| | — Monitoramento de data drift em produção | | |
| F6.6 | **Edge deployment** | TFLite + NVIDIA Jetson | Baixa |
| | — Inferência on-rig sem conectividade cloud | | |
| | — Latência-alvo: < 5ms em hardware embarcado | | |

**Verificação**: API responde em < 100ms. Dashboard renderiza em realtime. Container passa health checks.

---

### v3.0 — Inversão 2D/3D (Futuro)

| Área | Descrição | Base Científica |
|:-----|:----------|:---------------|
| Inversão 2D | Formações com falhas + dip variável | SPE/SPWLA (2021) — "Real-Time 2.5D Inversion" |
| Multi-frequência | Múltiplas frequências simultâneas (2-96 kHz) | Perspectiva P3 + ferramentas UDAR |
| Digital twin | Surrogate neural substitui completamente o Fortran | Li et al. (2025) — self-supervised forward |
| Geomecânica | Integrar pressão de poros + estabilidade de poço | Extensão petrofísica além de resistividade |
| Transfer learning | Pré-treinar em sintéticos, fine-tune em dados de campo | DomainAdapter já implementado (v2.0) |

---

## 4. Propostas de Novos Recursos

### 4.1 Alta Prioridade

| # | Recurso | Módulo | Justificativa | Esforço |
|:-:|:--------|:-------|:-------------|:-------:|
| 1 | Notebook `train_baseline.ipynb` | `notebooks/` | Pipeline end-to-end validado em GPU | 1 sessão |
| 2 | Notebook `train_surrogate.ipynb` | `notebooks/` | Treinar SurrogateNet Modo A | 1 sessão |
| 3 | XLA compilation | `training/loop.py` | 2-3× speedup de treinamento | 1 sessão |
| 4 | Mixed precision (float16) | `training/loop.py` | 1.5-2× speedup + 50% menos memória | 1 sessão |
| 5 | Benchmark 48 arquiteturas | `notebooks/` | Ranking R² × latência × parâmetros | 2 sessões |
| 6 | Preset `surrogate_mode_a.yaml` | `configs/` | Reprodutibilidade do surrogate | 0.5 sessão |

### 4.2 Média Prioridade

| # | Recurso | Módulo | Justificativa | Esforço |
|:-:|:--------|:-------|:-------------|:-------:|
| 7 | INN — arquitetura #45 | `models/advanced.py` | UQ probabilística (state-of-art 2025) | 2 sessões |
| 8 | ModernTCN SurrogateNet v2 | `models/surrogate.py` | Supera TCN clássico em benchmarks | 1 sessão |
| 9 | Multi-Task Loss (Kendall) | `losses/catalog.py` | Auto-balanceamento rho + DTB | 1 sessão |
| 10 | Profiling TensorBoard | `training/` | Identificar gargalos GPU | 1 sessão |
| 11 | DataPipeline cache + prefetch | `data/pipeline.py` | Reduzir overhead de I/O | 0.5 sessão |
| 12 | Quantização TFLite (int8) | `inference/export.py` | 4× redução tamanho para edge | 1 sessão |

### 4.3 Baixa Prioridade (Pesquisa)

| # | Recurso | Módulo | Justificativa | Esforço |
|:-:|:--------|:-------|:-------------|:-------:|
| 13 | G-Query Transformer (#46) | `models/transformer.py` | Multimodal geofísico (2025) | 3 sessões |
| 14 | Evidential Regression | `inference/uncertainty.py` | UQ single-pass (~30× mais rápido) | 2 sessões |
| 15 | PINN Petrofísica (Archie) | `losses/pinns.py` | Constraint de reservatório | 2 sessões |
| 16 | API REST (FastAPI) | Novo módulo | Produção industrial | 3 sessões |
| 17 | Dashboard Streamlit | Novo módulo | Visualização realtime | 2 sessões |
| 18 | Containerização Docker | Infraestrutura | Deploy padronizado | 2 sessões |

---

## 5. Otimizações de Código Recomendadas

| # | Otimização | Arquivo(s) | Impacto Esperado |
|:-:|:-----------|:-----------|:-----------------|
| 1 | Lazy import consistente em TODOS os módulos | Vários (`__init__.py`) | Startup ~50% mais rápido |
| 2 | `@tf.function` em funções de loss críticas | `losses/catalog.py` | 10-30% speedup em treinamento |
| 3 | Batch processing para `predict()` via `tf.data` | `inference/pipeline.py` | Melhor utilização GPU |
| 4 | Pré-compilação de scalers em TF ops | `data/scaling.py` | Eliminar overhead NumPy→TF |
| 5 | Paralelização de Feature Views + Geosinais no `tf.data.map` | `data/pipeline.py` | Reduzir latência on-the-fly |
| 6 | Memoização de blocos Keras comuns | `models/blocks.py` | Reduzir memory footprint |
| 7 | Gradient checkpointing para variantes U-Net | `models/unet.py` | ~40% menos memória GPU |

---

## 6. Próximos Passos (Requisições Futuras)

As seguintes requisições podem ser utilizadas em futuras sessões de desenvolvimento:

### F1 — Consolidação (Imediato)
```
"Commitar as mudanças pendentes dos Passos 1+2 (surrogate) e atualizar MEMORY.md"
```

### F2 — Treinamento GPU (Curto Prazo)
```
"Criar notebook train_baseline.ipynb para treinamento completo no Colab"
"Criar notebook train_surrogate.ipynb para treinar SurrogateNet Modo A"
"Criar configs/surrogate_mode_a.yaml com hiperparâmetros otimizados"
"Criar notebook benchmark_architectures.ipynb para ranking das 48 arquiteturas"
```

### F3 — Otimização (Médio Prazo)
```
"Implementar XLA compilation e mixed precision no TrainingLoop"
"Otimizar DataPipeline com cache, prefetch e interleave"
"Criar benchmark de latência de inferência para as 27 arquiteturas causais"
```

### F4 — Expansão Científica (Médio Prazo)
```
"Implementar INN (Invertible Neural Network) como arquitetura #45"
"Implementar ModernTCN para SurrogateNet v2"
"Completar Multi-Task Loss (Kendall et al. 2018) — atualmente placeholder"
"Adicionar cenário PINN petrofísica com constraints de Archie"
"Implementar Evidential Regression como método alternativo de UQ"
```

### F5 — Dados Multi-Dip (Longo Prazo)
```
"Criar script de processamento para dados multi-dip do Fortran"
"Treinar SurrogateNet Modo B com dados de dip 5°-90°"
"Validar SurrogateNet vs Fortran — RMSE por componente e dip"
```

### F6 — Deploy (Longo Prazo)
```
"Criar API REST com FastAPI para inferência batch e realtime"
"Criar Dockerfile multi-stage para deploy CPU e GPU"
"Implementar dashboard Streamlit para visualização realtime"
```

### F7 — Simulador Python Otimizado (Médio-Longo Prazo)

**Contexto**: O simulador Fortran `PerfilaAnisoOmp` (6.859 LOC, 6 módulos) resolve o forward EM 1D TIV
via decomposição TE/TM + transformada de Hankel (filtro Werthmuller 201 pts) + coeficientes de reflexão
recursivos. A análise de viabilidade (doc v2.0) demonstra que o gargalo computacional está em
`commonarraysMD` (propagação) e na convolução de Hankel — ambos altamente paralelizáveis.

```
"F7.1 — Protótipo NumPy vetorizado: commonarraysMD + hmd_TIV + vmd — validar vs Fortran (erro < 1e-10)"
"F7.2 — Numba @njit + prange: kernels críticos com compilação JIT paralela (target: ~1.2× Fortran)"
"F7.3 — CUDA kernels via Numba: 201 threads/bloco para propagação + redução para Hankel (target: ~20× Fortran)"
"F7.4 — JAX/XLA: diferenciação automática para backpropagation through physics (PINNs cenário surrogate)"
"F7.5 — Integrar como geosteering_ai/simulation/ com interface PipelineConfig — geração on-the-fly"
"F7.6 — Benchmark throughput: NumPy vs Numba CPU vs Numba CUDA vs Fortran OpenMP"
```

**Estimativas de desempenho** (doc seção 12.6):
| Implementação | Tempo/modelo | vs Fortran |
|:-------------|:------------|:-----------|
| NumPy vetorizado | ~4 s | 10× mais lento |
| Numba CPU (JIT) | ~0.5 s | ~1.2× mais lento |
| Numba CUDA | ~0.02 s | ~20× mais rápido |
| JAX (XLA) | ~0.05 s | ~8× mais rápido |

**Módulos propostos** (doc seção 12.8):
`geosteering_ai/simulation/`: forward.py, propagation.py, dipoles.py, hankel.py, rotation.py, filters.py, geometry.py, cuda_kernels.py, validation.py

Ref: `docs/reference/documentacao_simulador_fortran.md` (6.558 LOC, v4.0) — documentação completa incluindo formulação teórica via Potenciais de Hertz (Moran & Gianzero 1979), análise avançada de paralelismo OpenMP (3-level collapse, NUMA), viabilidade CUDA/Python, **Pipeline A Fortran** (otimização CPU+GPU em 3 fases), **Pipeline B Fortran** (novos recursos), **Pipeline A Python** (conversão Numba JIT CPU+GPU), **Pipeline B Python** (novos recursos), **Pipeline C Python** (vantagens sobre Fortran), e **Pipeline D Python** (avaliação comparativa Fortran vs Python).
Ref: `Tex_Projects/TatuAniso/FormulaçãoTatuAnisoTIV.tex` — artigo com a formulação matemática fundamental do simulador.

---

## 7. Referências Bibliográficas

### 7.1 Artigos Fundamentais do Pipeline

- Morales, M. et al. (2025). Anisotropic resistivity estimation and uncertainty quantification from borehole triaxial electromagnetic induction measurements: Gradient-based inversion and physics-informed neural network. *Computers & Geosciences*, 196, 105786.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Bai, S. et al. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*.
- Oord, A. et al. (2016). WaveNet: A Generative Model for Raw Audio. *arXiv:1609.03499*.

### 7.2 Artigos Recentes (2024-2026) — Expansões Propostas

- INN-UDAR (2025). Invertible neural network for real-time inversion and uncertainty quantification of ultra-deep resistivity measurements. *Computers & Geosciences*.
- Jiang, H. et al. (2025). One-Fit-All Transformer for Multimodal Geophysical Inversion. *JGR: Machine Learning and Computation*.
- Li, X. et al. (2025). A Novel Self-Supervised Deep Learning Inversion Method Incorporating a Fast Forward Network for Transient Electromagnetic Data. *JGR: ML&C*.
- ModernTCN (2024). A Modern Pure Convolution Structure for General Time Series Analysis. *ICLR*.
- SPE/SPWLA (2021). Real-Time 2.5D Inversion of LWD Resistivity Measurements Using Deep Learning for Geosteering Applications Across Faulted Formations.

### 7.3 Referências dos PDFs do Projeto (`PDFs/`)

Os artigos abaixo estão disponíveis localmente na pasta `PDFs/` e devem ser consultados nas fases de expansão científica (F4) e validação física (F5).

#### 7.3.1 Deep Learning para Inversão EM

- **Noh, K., Pardo, D., Torres-Verdín, C.** (2023). Physics-guided deep-learning inversion method for the interpretation of noisy logging-while-drilling resistivity measurements. *Geophysical Journal International*, 235, ggad217. DOI: 10.1093/gji/ggad217
  → Arquivo: `PDFs/ggad217.pdf`
  → Uso: Validação científica dos cenários PINN (`losses/pinns.py`) com ênfase em robustez a ruído LWD.

- **Puzyrev, V.** (2019). Deep learning electromagnetic inversion with convolutional neural networks. *Geophysical Journal International*, 218, 817–832.
  → Arquivo: `PDFs/ggz204.pdf`
  → Uso: Base histórica para inversão EM via DL; comparação com arquiteturas CNN do projeto.

- **Liu, W., Wang, H., Xi, Z., Zhang, R., Huang, X.** (2022). Physics-Driven Deep Learning Inversion with Application to Magnetotelluric. *Remote Sensing*, 14(13), 3218. DOI: 10.3390/rs14133218
  → Arquivo: `PDFs/remotesensing-14-03218-v2.pdf`
  → Uso: Valida integração de operador forward físico no loop de treinamento (cenário "surrogate" PINN).

- **Liu, W., Wang, H., Xi, Z., Wang, L.** (2024). Physics-Informed Deep Learning Inversion with Application to Noisy Magnetotelluric Measurements. *Remote Sensing*, 16(1), 62. DOI: 10.3390/rs16010062
  → Arquivo: `PDFs/remotesensing-16-00062-v2.pdf`
  → Uso: Estratégias de noise injection para dados EM de campo; alinha com curriculum 3-phase do projeto.

#### 7.3.2 Forward Modeling e Ferramenta LWD

- **Guo, W., Wang, L., Wang, N., Qiao, P., Zeng, Z., Yang, K.** (2024). Efficient 1D Modeling of Triaxial Electromagnetic Logging in Uniaxial and Biaxial Anisotropic Formations Using Virtual Boundaries and Equivalent Resistivity Schemes. *Journal of Geophysics and Engineering*, gxag017. DOI: 10.1093/jge/gxag017
  → Arquivo: `PDFs/gxag017.pdf`
  → Uso: Algoritmo eficiente para forward 1D triaxial TIV — base para treinar SurrogateNet e validar surrogate analítico.

- **Wang, L., Li, H., Fan, Y., Wu, Z.** (2018). Sensitivity analysis and inversion processing of azimuthal resistivity logging-while-drilling measurements. *Journal of Geophysics and Engineering*, 15, 2339–2352. DOI: 10.1088/1742-2140/aac5b1
  → Arquivo: `PDFs/Wang_2018_J._Geophys._Eng._15_2339.pdf`
  → Uso: Algoritmo de inversão convencional Gauss-Newton + janela deslizante 1D — baseline para comparação com o pipeline DL.

- **Guoyu Li, Zhenguan Wu, Xiaoqiao Liao, Xizhou Yue, Xiang Zhang, Tianlin Liu, Yunxin Zeng** (2025). Optimization and Analysis of Sensitive Areas for Look-Ahead Electromagnetic Logging-While-Drilling Based on Geometric Factors. *Energies*, 18, 3014. DOI: 10.3390/en18123014
  → Arquivo: `PDFs/Guoyu_et_al_2025_FG.pdf`
  → Uso: Análise de sensibilidade e DOD para look-ahead EM LWD — referência para avaliação de DOD Picasso (`evaluation/dod.py`).

- **Constable, M. V., Antonsen, F., Stalheim, S. O.** et al. (2016). Looking Ahead of the Bit While Drilling: From Vision to Reality. *Petrophysics*, 57(5), 426–438.
  → Arquivo: `PDFs/Constable_et_al_2016_Petrophysics.pdf`
  → Uso: Contexto aplicado da ferramenta EMLA (look-ahead EM) com DOI > 75 m; justifica o foco em geosteering de longo alcance.

#### 7.3.3 PINNs e Métodos Físico-Informados

- **Bai, J., Jeong, H., Batuwatta-Gamage, C. P.** et al. (2022). An Introduction to Programming Physics-Informed Neural Networks for Computational Solid Mechanics. *arXiv:2210.09060*.
  → Arquivo: `PDFs/2210.09060v4.pdf`
  → Uso: Tutorial de implementação de PINNs (TensorFlow/JAX) — referência pedagógica para `losses/pinns.py`.

#### 7.3.4 Documentação Técnica de Ferramentas

- **SDAR Work Group** (2020). Benchmark Models for Look-Ahead Applications — Standardization of LWD Deep Azimuthal Resistivity. Industry Technical Document.
  → Arquivo: `PDFs/Benchmark_Look_Ahead_Models.pdf`
  → Uso: Modelos benchmark padrão para look-ahead com LWD — referência para validação de cenários de geosteering.

- **University of Houston** (2016). A Survey on Definitions of Some Key Service Specs of LWD Deep Azimuthal Resistivity Tools. Technical Report.
  → Arquivo: `PDFs/Key_Specs_Survey_UH.pdf`
  → Uso: Definições de DOI (Depth of Investigation), DOD (Depth of Detection) e especificações de ferramentas — base conceitual para `evaluation/dod.py` (Picasso plots).

- **Schlumberger** (2020). GeoSphere HD — High-Definition Reservoir Mapping-While-Drilling Service. Internal Documentation Rev. 1.0.
  → Arquivos: `PDFs/arquivo.md` (texto) e `PDFs/Schlumberger_GeoSphere HD 1.0.pdf`
  → Uso: Especificações da ferramenta GeoSphere HD — física dos geosinais USD/UAD/UHR/UHA/U3DF, configuração de antenas, interpretação de medidas, Picasso plots.

- **CNOOC/COSL** (internal). GeoSphere vs. TatuAniso1D — Convenções de Sinais. Technical Note.
  → Arquivo: `PDFs/GeoSphereXTatu.pdf`
  → Uso: Tabela de conversão de sinais USD/UAD/U3DF entre GeoSphere e simulador TatuAniso1D — essencial para validar `data/geosignals.py`.

### 7.4 Referências do Simulador Fortran

- Liu, C. (2017). *Theory of Electromagnetic Well Logging*. Elsevier. — Rotação do tensor triaxial (eq. 4.80), utilizada em `RtHR()`.
- Werthmuller, D. (2006). "EMMOD — Electromagnetic Modelling". *Report, TU Delft*. — Filtro digital de 201 pontos para transformada de Hankel.
- Werthmuller, D. (2017). "An open-source full 3D electromagnetic modeller for 1D VTI media in Python: empymod". *Geophysics*, 82(6), WB9-WB19. — Referência Python para reimplementação.
- Chew, W.C. (1995). *Waves and Fields in Inhomogeneous Media*. IEEE Press. — Formulação TE/TM para meios estratificados.
- Anderson, W.L. (1982). "Fast Hankel Transforms Using Related and Lagged Convolutions". *ACM TOMS*, 8(4), 344-368.

### 7.5 Livros-Texto

- Ellis, D. V. & Singer, J. M. (2008). *Well Logging for Earth Scientists* (2nd ed.). Springer.
  → Arquivo: `PDFs/Darwin V Ellis Julian M Singer - Well Logging for Earth Scientists 2008 Springer - libgenli.pdf`
- Zhang, W. (2013). *Principles and Applications of Well Logging*. Springer.
  → Arquivo: `PDFs/Principles and Applications of Well Logging.pdf`
- Misra, S., Li, H. & He, J. (2019). *Machine Learning for Subsurface Characterization*. Elsevier.
- Zhang, A. et al. (2023). *Dive into Deep Learning*. D2L.ai.
  → Arquivo: `PDFs/[Livro] Dive Into Deep Learning_compressed.pdf`

---

## 8. Arquivos Críticos para Consulta

| Arquivo | Propósito |
|:--------|:----------|
| `geosteering_ai/config.py` | PipelineConfig (246 campos) — ponto único de verdade |
| `geosteering_ai/models/registry.py` | ModelRegistry — 48 arquiteturas |
| `geosteering_ai/losses/factory.py` | LossFactory.build_combined() |
| `geosteering_ai/losses/pinns.py` | 8 cenários PINN + forward analítico |
| `geosteering_ai/models/surrogate.py` | SurrogateNet TCN |
| `geosteering_ai/data/surrogate_data.py` | Extração de pares para surrogate |
| `geosteering_ai/training/loop.py` | TrainingLoop |
| `geosteering_ai/training/callbacks.py` | 17+ callbacks Keras |
| `geosteering_ai/inference/pipeline.py` | InferencePipeline (save/load) |
| `docs/ARCHITECTURE_v2.md` | Especificação arquitetural completa |
| `docs/physics/errata_valores.md` | Constantes físicas imutáveis |
| `CLAUDE.md` | Regras e proibições de desenvolvimento |
| `docs/ROADMAP.md` | Este documento |
| `docs/reference/documentacao_simulador_fortran.md` | Documentação completa do simulador Fortran PerfilaAnisoOmp |
| `Fortran_Gerador/PerfilaAnisoOmp.f08` | Simulador EM 1D TIV — módulo principal |
| `Fortran_Gerador/fifthBuildTIVModels.py` | Gerador de modelos geológicos (Sobol QMC) |
| `Tex_Projects/TatuAniso/FormulaçãoTatuAnisoTIV.tex` | Formulação teórica (Potenciais de Hertz, TIV) |

---

*Documento gerado em Abril 2026. Geosteering AI v2.0.*
*Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch proibido).*
*Configuração: PipelineConfig dataclass (ponto único de verdade).*
