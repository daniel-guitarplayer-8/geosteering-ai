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

### Secoes 5-7: A Implementar (C48-C73)

| Secao | Destino v2.0 | Acao |
|:-----:|:-------------|:----:|
| Secao 5 (C48-C57): Avaliacao | `evaluation/` | CRIAR do zero |
| Secao 6 (C58-C65): Visualizacao | `visualization/` | CRIAR do zero |
| Secao 7 (C66-C73): Geosteering | `inference/realtime.py` + `visualization/realtime_monitor.py` | CRIAR do zero |
