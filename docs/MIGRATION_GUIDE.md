# Geosteering AI — Guia de Migracao: Legado (C0-C47) → Pacote v2.0

## Mapa: Celula Legada → Modulo v2.0

### Secao 0: Infraestrutura (C0-C2)

| Celula | Funcao/Classe | Destino v2.0 | Acao |
|:------:|:-------------|:-------------|:----:|
| C0 | `logger`, `ColoredFormatter` | `utils/logger.py` | EXTRAIR |
| C0 | `C` (ANSI colors class) | `utils/logger.py` | EXTRAIR |
| C0 | `format_time`, `timer_decorator` | `utils/timer.py` | EXTRAIR |
| C0 | `ValidationTracker` | `utils/validation.py` | EXTRAIR |
| C0 | `print_header`, `format_number`, `format_bytes` | `utils/formatting.py` | EXTRAIR |
| C0 | `is_colab`, `has_gpu`, `gpu_memory_info` | `utils/system.py` | EXTRAIR |
| C0 | `safe_mkdir`, `safe_json_dump` | `utils/io.py` | EXTRAIR |
| C0 | `set_all_seeds` | `utils/__init__.py` | EXTRAIR |
| C1 | pip install | `pyproject.toml` dependencies | SUBSTITUIDO |
| C2 | imports + verificacao | Package `__init__.py` imports | SUBSTITUIDO |

### Secao 1: FLAGS (C3-C18)

| Celula | Conteudo | Destino v2.0 | Acao |
|:------:|:---------|:-------------|:----:|
| C3 | Paths, FILE_SCHEME | `config.py` campos base_dir, dataset_dir | INTEGRADO |
| C4-C18 | ~1.185 FLAGS | `config.py` PipelineConfig dataclass | INTEGRADO |
| C17 | ARCH_PARAMS dict | `config.py` campo arch_params ou YAML | INTEGRADO |
| C18 | Optuna FLAGS | `config.py` campos optuna_* | INTEGRADO |

### Secao 2: Dados (C19-C26)

| Celula | Funcao | Destino v2.0 | Acao |
|:------:|:-------|:-------------|:----:|
| C19 | `parse_out_metadata()` | `data/loader.py` | EXTRAIR |
| C19 | `load_binary_dat()` | `data/loader.py` | EXTRAIR |
| C19 | `split_by_geological_model()` | `data/splitter.py` | EXTRAIR |
| C19 | `segregate_by_angle()` | `data/loader.py` | EXTRAIR |
| C19 | `create_sliding_window_dataset()` | `data/loader.py` | REFATORAR (CC~99→3 sub-funcoes) |
| C20 | `detect_boundaries()`, `compute_dtb_labels()` | `data/loader.py` | EXTRAIR |
| C21 | `compute_coupling_correction()` | `data/decoupling.py` | EXTRAIR |
| C21 | `apply_decoupling()` | `data/decoupling.py` | EXTRAIR |
| C22 | `apply_feature_view()` numpy | `data/feature_views.py` | EXTRAIR |
| C22 | `apply_feature_view_tf()` TF | `data/feature_views.py` | EXTRAIR |
| C22 | `compute_geosignal_features()` numpy | `data/geosignals.py` | EXTRAIR |
| C22 | `compute_geosignal_tf()` TF | `data/geosignals.py` | EXTRAIR |
| C22 | `compute_expanded_input_features()` | `data/geosignals.py` | EXTRAIR |
| C23 | `apply_target_scaling()` | `data/scaling.py` | EXTRAIR |
| C23 | `inverse_target_scaling()` | `data/scaling.py` | EXTRAIR |
| C23 | `create_scaler()` | `data/scaling.py` | EXTRAIR |
| C23 | `fit_and_transform_features()` | `data/scaling.py` | REFATORAR (separar fit de transform) |
| C24 | `add_gaussian_noise()` TF | `noise/functions.py` | EXTRAIR |
| C24 | `add_multiplicative_noise()` TF | `noise/functions.py` | EXTRAIR |
| C24 | `apply_raw_em_noise()` numpy | `noise/utils.py` | EXTRAIR |
| C24 | `train_map_fn` | `data/pipeline.py` DataPipeline.build_train_map_fn() | REFATORAR |
| C24 | `_make_tf_scaler_fn()` | `data/scaling.py` make_tf_scaler() | EXTRAIR + ATIVAR |
| C24 | `Noisy3DDataGenerator` | REMOVIDO (noise offline) | DESCARTAR |
| C25 | `InferencePipeline` class | `inference/pipeline.py` | EXTRAIR |
| C25 | `optimize_dataset_pipeline()` | `data/pipeline.py` | EXTRAIR |
| C26 | Picasso DOD plots | `visualization/picasso.py` | EXTRAIR |
| C26A | Inspecao de dados | `visualization/eda.py` | EXTRAIR |

### Secao 3: Arquiteturas (C27-C39)

| Celula | Funcoes | Destino v2.0 | Acao |
|:------:|:--------|:-------------|:----:|
| C27 | 23 blocos reutilizaveis | `models/blocks.py` | EXTRAIR |
| C28 | build_resnet18/34/50, convnext, inception×2 | `models/cnn.py` | EXTRAIR |
| C29 | build_cnn1d | `models/cnn.py` | EXTRAIR |
| C30 | build_tcn, tcn_advanced | `models/tcn.py` | EXTRAIR |
| C31 | build_lstm, bilstm | `models/rnn.py` | EXTRAIR |
| C32 | build_cnn_lstm, cnn_bilstm_ed | `models/rnn.py` | EXTRAIR |
| C33 | 14 build_unet_* | `models/unet.py` | EXTRAIR |
| C34 | 6 build_transformer_* | `models/transformer.py` | EXTRAIR |
| C35 | build_nbeats, nhits | `models/decomposition.py` | EXTRAIR |
| C36 | build_dnn, fno, deeponet, geo_attention | `models/advanced.py` | EXTRAIR |
| C36A | 5 build_geosteering_* | `models/geosteering.py` | EXTRAIR |
| C37 | _MODEL_REGISTRY, build_model() | `models/registry.py` ModelRegistry | REFATORAR |
| C38 | Optuna search space | `training/optuna_hpo.py` | EXTRAIR |
| C39 | Checkpoint Secao 3 | REMOVIDO (testes unitarios substituem) | DESCARTAR |

### Secao 4: Treinamento (C40-C47)

| Celula | Funcao | Destino v2.0 | Acao |
|:------:|:-------|:-------------|:----:|
| C40 | 26 callbacks + assembly | `training/callbacks.py` build_callbacks() | REFATORAR |
| C41 | 26 losses + get_loss() | `losses/catalog.py` + `losses/factory.py` | EXTRAIR + REFATORAR |
| C42 | 3 custom metrics | `training/metrics.py` | EXTRAIR |
| C42A | Relatorio FLAGS | REMOVIDO (config.__repr__() substitui) | DESCARTAR |
| C42B | Holdout plots | `visualization/holdout.py` | REFATORAR |
| C43 | Training loop + N-Stage | `training/loop.py` + `training/nstage.py` | REFATORAR |
| C44 | Curriculum summary | Integrado em TrainingLoop | INTEGRADO |
| C45 | Optuna HPO | `training/optuna_hpo.py` | EXTRAIR |
| C46 | Model saving | `inference/export.py` | EXTRAIR |
| C47 | Training summary | Integrado em TrainingLoop.run() return | INTEGRADO |

### Secao 5: Avaliacao Avancada (C48-C57)

| Celula | Funcao | Destino v2.0 | Acao |
|:------:|:-------|:-------------|:----:|
| C48 | Predicao no teste (y_pred) | `evaluation/predict.py` predict_test() | CRIAR |
| C49 | Metricas globais + MBE por componente | `evaluation/metrics.py` (JA EXISTE) | EXPANDIR |
| C50 | Metricas por interface + sharpness | `evaluation/advanced.py` interface_metrics() | CRIAR |
| C51 | Erro por faixa de resistividade | `evaluation/advanced.py` error_by_resistivity_band() | CRIAR |
| C52 | Erro por faixa de anisotropia | `evaluation/advanced.py` error_by_anisotropy() | CRIAR |
| C53 | Erro espacial por profundidade | `evaluation/advanced.py` spatial_error_profile() | CRIAR |
| C54 | Coerencia fisica (rho_v >= rho_h) | `evaluation/advanced.py` physical_coherence_check() | CRIAR |
| C55 | Analise de estabilidade (perturbacao) | `evaluation/advanced.py` stability_analysis() | CRIAR |
| C56 | Holdout evaluation | `evaluation/metrics.py` evaluate_predictions() (JA EXISTE) | EXPANDIR |
| C57 | Comparacao multi-modelo | `evaluation/comparison.py` (JA EXISTE) | EXPANDIR |

### Secao 6: Visualizacao Avancada (C58-C65)

| Celula | Funcao | Destino v2.0 | Acao |
|:------:|:-------|:-------------|:----:|
| C58 | Perfis predito vs verdadeiro | `visualization/holdout.py` (JA EXISTE) | EXPANDIR |
| C59 | Historico de treinamento (loss, LR) | `visualization/training.py` plot_training_history() | CRIAR |
| C60 | Mapas de erro (heatmaps) | `visualization/error_maps.py` plot_error_maps() | CRIAR |
| C61 | Analise de incerteza (histogramas, CI) | `visualization/uncertainty.py` plot_uncertainty() | CRIAR |
| C62 | Visualizacoes Optuna (se HPO ativo) | `visualization/optuna_viz.py` plot_optuna_results() | CRIAR |
| C63 | Exportacao de figuras (PNG/PDF/SVG) | `visualization/export.py` export_all_figures() | CRIAR |
| C64 | Manifesto JSON do experimento | `evaluation/manifest.py` create_manifest() | CRIAR |
| C65 | Relatorio final automatizado | `evaluation/report.py` generate_report() | CRIAR |

### Secao 7: Geosteering Avancado (C66-C73)

| Celula | Funcao | Destino v2.0 | Acao |
|:------:|:-------|:-------------|:----:|
| C66 | Inference pipeline realtime | `inference/realtime.py` (JA EXISTE) | EXPANDIR |
| C67 | Quantificacao de incerteza (MC/Ensemble) | `inference/uncertainty.py` UncertaintyEstimator | CRIAR |
| C68 | Edge export (TFLite/ONNX) | `inference/export.py` (JA EXISTE) | EXPANDIR |
| C69 | Domain adaptation (fine-tune campo) | `training/adaptation.py` DomainAdapter | CRIAR |
| C70 | Comparacao offline vs realtime | `evaluation/realtime_comparison.py` compare_modes() | CRIAR |
| C71 | Metricas geosteering (DTB, look-ahead) | `evaluation/geosteering_metrics.py` GeoMetrics | CRIAR |
| C72 | Visualizacao geosteering (curtain plots) | `visualization/geosteering.py` plot_curtain() | CRIAR |
| C73 | Relatorio geosteering | `evaluation/geosteering_report.py` generate_geo_report() | CRIAR |

### Callbacks Faltantes (C40 expansao)

| Callback Legado | Destino v2.0 | Prioridade |
|:----------------|:-------------|:-----------|
| DualValidationCallback (P2) | `training/callbacks.py` | ALTA |
| PINNSLambdaScheduleCallback | `training/callbacks.py` | ALTA |
| CausalDegradationMonitorCallback | `training/callbacks.py` | ALTA |
| SlidingWindowValidationCallback | `training/callbacks.py` | ALTA |
| PeriodicCheckpointCallback | `training/callbacks.py` | MEDIA |
| MetricPlateauDetectorCallback | `training/callbacks.py` | MEDIA |
| OneCycleLRCallback | `training/callbacks.py` | MEDIA |
| CosineWarmRestartsCallback | `training/callbacks.py` | MEDIA |
| CyclicalLRCallback | `training/callbacks.py` | MEDIA |
| MemoryMonitorCallback | `training/callbacks.py` | BAIXA |
| LatencyBenchmarkCallback | `training/callbacks.py` | BAIXA |
| EpochSummaryCallback | `training/callbacks.py` | BAIXA |

### Gaps Remanescentes (identificados na auditoria de 2026-03-26)

#### Noise Catalog (C12 → noise/)

| Recurso Legado | Destino v2.0 | Status |
|:---------------|:-------------|:------:|
| NOISE_CATALOG 35 tipos | `noise/catalog.py` (CRIAR) + `noise/functions.py` (EXPANDIR) | FALTA |
| 9 CORE: varying, gaussian_local/global, speckle, drift, quantization, saturation, depth_dependent, pink | `noise/functions.py` | FALTA (4 de 35 impl.) |
| 12 EXTENDED: cross_talk, orientation, emi, freq_dependent, noise_floor, proportional, reim_diff, component_diff, gaussian_keras, motion, thermal, spikes | `noise/functions.py` | FALTA |
| 13 EXPERIMENTAL: dropouts, uniform, arma, fractal, step, mixture, phase_shift, synthetic_geological, poisson, salt_pepper, lognormal, rayleigh, rician, spectral_custom | `noise/functions.py` | FALTA |
| NOISE_APPLICATION_STAGE | `config.py` noise_application_stage | FALTA (campo) |
| NOISE_COMBINATION_MODE | `config.py` noise_combination_mode | FALTA (campo) |

#### PINNs (C13 → losses/ + config.py)

| Recurso Legado | Destino v2.0 | Status |
|:---------------|:-------------|:------:|
| 15 FLAGS PINNs (cenarios, norms, hard constraint, surrogate) | `config.py` SECAO 18 | FALTA |
| 3 cenarios L_physics (oracle, surrogate, maxwell) | `losses/pinns.py` (CRIAR) | FALTA |
| Hard constraint layer (Morales I1) | `models/blocks.py` | FALTA |
| Lambda schedule 4 metodos (linear, cosine, step, fixed) | `training/callbacks.py` PINNSLambdaScheduleCallback | PARCIAL (so linear) |
| Custom training loop com GradientTape (maxwell) | `training/loop.py` | FALTA |

#### Geological Parser + DTB (C20 → data/)

| Recurso Legado | Destino v2.0 | Status |
|:---------------|:-------------|:------:|
| parse_geological_structure() | `data/boundaries.py` (CRIAR) | FALTA |
| detect_boundaries() (3 metodos) | `data/boundaries.py` | FALTA |
| compute_dtb_labels() (3 estrategias, 3 scalings) | `data/boundaries.py` | FALTA |

#### tf.data Optimization (C24 → data/pipeline.py)

| Recurso Legado | Destino v2.0 | Status |
|:---------------|:-------------|:------:|
| .prefetch(tf.data.AUTOTUNE) | `data/pipeline.py` build_tf_dataset() | FALTA |
| .cache() para val/test | `data/pipeline.py` | FALTA |
| num_parallel_calls=AUTOTUNE no .map() | `data/pipeline.py` | FALTA |
| Mixed Precision set_global_policy | `training/loop.py` _setup_mixed_precision() | FALTA |

#### EDA + Picasso (C26 → visualization/)

| Recurso Legado | Destino v2.0 | Status |
|:---------------|:-------------|:------:|
| plot_feature_distributions() | `visualization/eda.py` | FALTA |
| plot_correlation_heatmap() | `visualization/eda.py` | FALTA |
| plot_sample_profiles() | `visualization/eda.py` | FALTA |
| plot_train_val_test_comparison() | `visualization/eda.py` | FALTA |
| plot_sensitivity_heatmap() | `visualization/eda.py` | FALTA |
| compute_picasso_dod() modelo analitico 2-layer | `visualization/picasso.py` | FALTA |
| Picasso multi-frequencia (A1) | `visualization/picasso.py` | FALTA |
| Picasso multi-espacamento (A2) | `visualization/picasso.py` | FALTA |
| Picasso multi-angulo | `visualization/picasso.py` | FALTA |
| Picasso side-by-side (B2) | `visualization/picasso.py` | FALTA |
| Picasso assimetria (B3) | `visualization/picasso.py` | FALTA |
| Export .npy + .json (B4) | `visualization/picasso.py` | FALTA |

#### LR Schedule Helpers (C40 → training/)

| Recurso Legado | Destino v2.0 | Status |
|:---------------|:-------------|:------:|
| _make_cosine_schedule() (Loshchilov 2016) | `training/callbacks.py` | FALTA |
| _make_step_schedule() | `training/callbacks.py` | FALTA |
| _make_warmup_cosine_schedule() (Vaswani 2017) | `training/callbacks.py` | FALTA |
| EpochTierCallback (P5 offline tiers) | `training/callbacks.py` | FALTA |
| GradientMonitorCallback (GradientTape real) | `training/callbacks.py` | FALTA |
