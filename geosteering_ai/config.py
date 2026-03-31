# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: config.py                                                         ║
# ║  Bloco: 1 — Fundacao (Config + Estrutura)                                 ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass — PONTO UNICO DE VERDADE                ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • PipelineConfig dataclass: substitui 574 globals().get() do legado    ║
# ║    • Validacao fail-fast de Errata v4.4.5 + v5.0.15 em __post_init__     ║
# ║    • 5 presets (baseline, robusto, nstage, geosinais_p4, realtime)        ║
# ║    • Serializacao YAML para reprodutibilidade (from_yaml/to_yaml)         ║
# ║    • Propriedades derivadas (n_features, needs_onthefly_fv_gs, etc.)      ║
# ║                                                                            ║
# ║  Dependencias: nenhuma interna (modulo raiz)                              ║
# ║  Exports: ~1 classe (PipelineConfig) — ver __all__                        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 3.1                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial com 210+ atributos           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""PipelineConfig — Ponto unico de verdade para todas as FLAGS do pipeline.

Substitui 574 chamadas globals().get() por um dataclass tipado, validado,
e serializavel. Errata v4.4.5 e v5.0.15 sao verificadas no __post_init__
(fail-fast). Presets disponiveis via metodos de classe.

Example:
    >>> config = PipelineConfig.robusto()
    >>> config.noise_level_max
    0.08
    >>> config.to_yaml("configs/meu_experimento.yaml")
    >>> loaded = PipelineConfig.from_yaml("configs/meu_experimento.yaml")
    >>> assert config == loaded

Note:
    Framework: TensorFlow 2.x / Keras (EXCLUSIVO — PyTorch PROIBIDO).
    Referenciado em:
        - data/loading.py: parse_out_metadata, load_binary_dat, apply_decoupling,
          segregate_by_angle, load_dataset (parametro config)
        - data/splitting.py: split_angle_group (parametro config)
        - data/scaling.py: fit_per_group_scalers (parametro config)
        - data/pipeline.py: DataPipeline.__init__ (atributo self.config)
        - data/feature_views.py: apply_feature_view (config.feature_view)
        - data/geosignals.py: compute_expanded_features (config.input_features)
        - tests/test_config.py: TestErrata, TestPresets, TestSerialization,
          TestDerivedProperties, TestValidation, TestMutualExclusivity
        - tests/test_data_pipeline.py: todos os testes via PipelineConfig.baseline()
          ou PipelineConfig.geosinais_p4()
    Ref: docs/ARCHITECTURE_v2.md secao 3.1 (PipelineConfig) e secao 4.1
    (validacao de errata).
"""

# ──────────────────────────────────────────────────────────────────────
# Imports padrao (stdlib apenas — sem dependencias externas)
# ──────────────────────────────────────────────────────────────────────
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────
# D8: Exports publicos — agrupados semanticamente
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    # --- Classe principal ---
    "PipelineConfig",
]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuracao unica e validada do pipeline de inversao geofisica.

    Cada FLAG do pipeline e um campo tipado com default explicito.
    Validacao automatica no __post_init__ garante errata e consistencia.
    Serializavel para YAML (reprodutibilidade) e dict (logging).

    Attributes:
        frequency_hz: Frequencia EM em Hz. DEVE ser 20000.0 (Errata v4.4.5).
        spacing_meters: Espacamento T-R em metros. DEVE ser 1.0 (Errata v4.4.5).
        sequence_length: Numero de medidas por modelo. DEVE ser 600.
        input_features: Indices das colunas de entrada no formato 22-col.
        output_targets: Indices das colunas de saida no formato 22-col.
        target_scaling: Metodo de scaling dos targets. DEVE ser "log10".
        model_type: Tipo de arquitetura (44 opcoes no ModelRegistry).
        inference_mode: "offline" (acausal) ou "realtime" (causal/geosteering).
        use_noise: Ativa injecao de ruido on-the-fly.
        noise_level_max: Nivel maximo de ruido (sigma).
        use_geosignal_features: Ativa geosinais (P4) no pipeline on-the-fly.
        feature_view: Transformacao Feature View (6 opcoes).
        learning_rate: Taxa de aprendizado.
        epochs: Numero maximo de epocas de treinamento.
        use_nstage: Ativa N-Stage Training (mutuamente exclusivo com curriculum).
        use_curriculum: Ativa curriculum noise 3-phase.

    Example:
        >>> config = PipelineConfig.baseline()      # P1: sem noise
        >>> config = PipelineConfig.robusto()        # E-Robusto S21
        >>> config = PipelineConfig.nstage(n=3)      # N-Stage N=3
        >>> config = PipelineConfig.geosinais_p4()   # P4 com GS
        >>> config = PipelineConfig.realtime()        # Geosteering causal

    Note:
        Referenciado em TODOS os modulos do pacote como parametro obrigatorio.
        Ref: docs/ARCHITECTURE_v2.md secao 3.1 (design) e CLAUDE.md
        (proibicoes absolutas e valores fisicos criticos).
        Errata v4.4.5: frequency_hz, spacing_meters, sequence_length.
        Errata v5.0.15: input_features, output_targets, target_scaling, eps_tf.
        Validacao fail-fast no __post_init__: violacao gera AssertionError.
    """

    #   ┌──────────────────────────────────────────────────────────────────────┐
    #   │  PipelineConfig — 7 Secoes de Campos                                │
    #   ├──────────────────────────────────────────────────────────────────────┤
    #   │  SECAO 1: Fisica       │ frequency_hz, spacing_meters, seq_length   │
    #   │  SECAO 2: Dados/Split  │ split ratios, dual_validation, holdout     │
    #   │  SECAO 3: Noise        │ noise_level_max, curriculum, n_stage       │
    #   │  SECAO 4: FV/GS/Scale  │ feature_view, geosignals, scaler_type     │
    #   │  SECAO 5: Modelo       │ model_type, filters, kernel_size, blocks   │
    #   │  SECAO 6: Treinamento  │ LR, epochs, patience, optimizer, loss      │
    #   │  SECAO 7: Inference    │ inference_mode, export, realtime, UQ       │
    #   ├──────────────────────────────────────────────────────────────────────┤
    #   │  Errata: __post_init__ valida physics + mutual exclusivity          │
    #   │  Presets: baseline, robusto, nstage, geosinais_p4, realtime        │
    #   │  Serializacao: from_yaml/to_yaml para reprodutibilidade             │
    #   └──────────────────────────────────────────────────────────────────────┘

    # ══════════════════════════════════════════════════════════════════
    # SECAO 1: FISICA (Errata v4.4.5 + v5.0.15 — valores imutaveis)
    # Constantes do sensor LWD validadas contra Errata publicada.
    # frequency_hz = 20 kHz (operacao real do tool EM);
    # spacing_meters = 1.0 m (distancia transmissor-receptor);
    # sequence_length = 600 medidas por modelo geologico.
    # NUNCA alterar estes valores — violacao gera AssertionError.
    # ══════════════════════════════════════════════════════════════════
    frequency_hz: float = 20000.0
    spacing_meters: float = 1.0
    sequence_length: int = 600
    input_features: List[int] = field(default_factory=lambda: [1, 4, 5, 20, 21])
    output_targets: List[int] = field(default_factory=lambda: [2, 3])
    target_scaling: str = "log10"
    depth_max: float = 150.0
    n_columns: int = 22

    # ══════════════════════════════════════════════════════════════════
    # SECAO 2: DADOS E SPLIT
    # Split por modelo geologico (NUNCA por amostra) — principio P1.
    # Cada modelo geologico e um cenario unico de resistividade;
    # misturar amostras do mesmo modelo entre train/val/test causa
    # data leakage. Ratios padrao: 70/15/15.
    # use_dual_validation: P2 — valida em clean + noisy separadamente.
    # ══════════════════════════════════════════════════════════════════
    split_by_model: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    use_dual_validation: bool = True
    global_seed: int = 42

    # ══════════════════════════════════════════════════════════════════
    # SECAO 2B: PERSPECTIVAS P2/P3 — Angulo e Frequencia como Features
    # Theta (angulo de inclinacao) e frequencia da ferramenta LWD
    # injetados como colunas PREFIXO no array de features. Nao existem
    # no .dat (22-col) — vem do header (.out) e sao broadcast por seq.
    #   theta_norm = theta/90.0 → [0.0, 1.0]
    #   f_norm = log10(freq) → [3.3, 5.6] para 2kHz-400kHz
    # Layout resultante:
    #   P1:    [z, Re(H1), Im(H1), Re(H2), Im(H2)]       → 5 feat
    #   P2:    [theta_norm, z, H1, H2]                     → 6 feat
    #   P3:    [f_norm, z, H1, H2]                         → 6 feat
    #   P2+P3: [theta_norm, f_norm, z, H1, H2]            → 7 feat
    # Ref: docs/physics/perspectivas.md secoes P2, P3.
    # Nota: theta e freq sao PROTEGIDOS do noise (parametros conhecidos,
    #        nao medidos pelo sensor EM).
    # ══════════════════════════════════════════════════════════════════
    use_theta_as_feature: bool = False
    use_freq_as_feature: bool = False
    freq_normalization: str = "log10"
    # ── Modo de injecao de variaveis estaticas (theta, freq) ──────────
    # "broadcast"  → Abordagem A: colunas repetidas no array (default)
    # "dual_input" → Abordagem B: escalares separados + broadcast no modelo
    # "film"       → Abordagem C: FiLM conditioning (modulacao γ×h+β)
    # Ref: Perez et al. (2018) "FiLM: Visual Reasoning with Conditioning"
    # Nota: "film" restrito a arquiteturas compativeis (CNN, TCN, Transformer,
    #        Geosteering, Hybrid). Incompativeis: N-BEATS, N-HiTS, FNO, DeepONet.
    static_injection_mode: str = "broadcast"

    # ══════════════════════════════════════════════════════════════════
    # SECAO 3: FEATURE VIEWS E GEOSINAIS
    # Feature Views (FV) transformam componentes EM brutos em
    # representacoes mais informativas (log, razao, etc.).
    # Geosinais (GS) sao features derivadas (USD, UHR, etc.) que
    # capturam relacoes fisicas entre componentes EM.
    # Ambos devem ser computados APOS noise (on-the-fly) para
    # fidelidade fisica: GS veem ruido como LWD real.
    # ══════════════════════════════════════════════════════════════════
    feature_view: str = "identity"
    use_geosignal_features: bool = False
    geosignal_set: str = "usd_uhr"
    geosignal_families: Optional[List[str]] = None
    eps_tf: float = 1e-12

    # ══════════════════════════════════════════════════════════════════
    # SECAO 4: DECOUPLING EM
    # Remove acoplamento direto (free-space) das componentes do tensor
    # EM. Coeficientes para L = 1.0 m:
    #   ACp = -1/(4*pi*L^3) = -0.079577 (planar: Hxx, Hyy)
    #   ACx = +1/(2*pi*L^3) = +0.159155 (axial: Hzz)
    # decoupling_full_tensor: processa todos os 9 componentes (3x3).
    # ══════════════════════════════════════════════════════════════════
    decoupling_hxx: bool = True
    decoupling_hyy: bool = True
    decoupling_hzz: bool = True
    decoupling_full_tensor: bool = False

    # ══════════════════════════════════════════════════════════════════
    # SECAO 5: SCALING
    # Scaler fitado em dados LIMPOS (FV+GS clean) — principio P3.
    # Fit em dados ruidosos introduz bias estatistico no scaler.
    # use_per_group_scalers: scalers separados para EM vs GS.
    # gs_scaler_type: robust e melhor para GS (outliers frequentes).
    # smoothing_type: pos-processamento opcional das predicoes.
    # ══════════════════════════════════════════════════════════════════
    scaler_type: str = "standard"
    use_per_group_scalers: bool = True
    gs_scaler_type: str = "robust"
    use_separate_z_scaler: bool = False
    smoothing_type: str = "none"

    # ══════════════════════════════════════════════════════════════════
    # SECAO 6: NOISE (ON-THE-FLY EXCLUSIVO)
    # Ruido injetado DINAMICAMENTE a cada batch via tf.data.map.
    # On-the-fly e o UNICO path fisicamente correto (noise offline
    # com GS viola causalidade: GS veriam dados limpos).
    # Curriculum 3-phase: clean → rampa → estavel.
    # noise_level_max = 0.08 (sigma) — limite superior E-Robusto S21.
    # ══════════════════════════════════════════════════════════════════
    use_noise: bool = True
    noise_level_max: float = 0.08
    noise_types: List[str] = field(default_factory=lambda: ["gaussian"])
    noise_weights: List[float] = field(default_factory=lambda: [1.0])
    use_curriculum: bool = True
    epochs_no_noise: int = 10
    noise_ramp_epochs: int = 80

    # ══════════════════════════════════════════════════════════════════
    # SECAO 7: ARQUITETURA
    # 44 arquiteturas disponiveis (39 standard + 5 geosteering).
    # model_type seleciona via ModelRegistry (factory pattern).
    # inference_mode "realtime" auto-ativa use_causal_mode.
    # use_physical_constraint_layer: camada de saida que aplica
    # restricoes fisicas (ex: resistividade positiva via softplus).
    # arch_params: override granular dos hiperparametros da arq.
    # ══════════════════════════════════════════════════════════════════
    model_type: str = "ResNet_18"
    inference_mode: str = "offline"
    use_causal_mode: bool = False
    output_channels: int = 2
    use_physical_constraint_layer: bool = False
    constraint_activation: str = "softplus"
    arch_params: Optional[Dict[str, Any]] = None

    # ══════════════════════════════════════════════════════════════════
    # SECAO 8: SKIP CONNECTIONS E BLOCOS
    # Conexoes residuais e mecanismos de atencao para blocos conv.
    # skip_connection_type "add" (residual) ou "concat" (dense).
    # SE (Squeeze-and-Excitation) recalibra canais por atencao
    # global — melhora convergencia em arqs profundas (ResNet_50+).
    # se_reduction: fator de reducao do bottleneck SE.
    # ══════════════════════════════════════════════════════════════════
    use_skip_connections: bool = True
    skip_connection_type: str = "add"
    use_se_block: bool = False
    se_reduction: int = 16

    # ══════════════════════════════════════════════════════════════════
    # SECAO 9: TREINAMENTO
    # Hiperparametros gerais de otimizacao e treinamento.
    # LR = 1e-4 (cenario E-Robusto S21 — estavel com noise 8%).
    # AdamW com weight_decay para regularizacao implicita.
    # Patience 60 epocas para curriculum (rampa ~80 ep + estavel).
    # use_restore_best_weights = False (preserva pesos noise-trained,
    # evita reversao para pesos pre-ruido — bug S20).
    # ══════════════════════════════════════════════════════════════════
    learning_rate: float = 1e-4
    epochs: int = 400
    batch_size: int = 32
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    early_stopping_patience: int = 60
    use_restore_best_weights: bool = False
    use_gradient_clipping: bool = True
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = False
    # ── XLA (Accelerated Linear Algebra) ───────────────────────────────
    # Habilita compilacao JIT via jit_compile=True no model.compile().
    # XLA funde operacoes elementares (Conv+BN+ReLU) em kernels GPU
    # unicos, eliminando alocacoes intermediarias de VRAM e reduzindo
    # overhead de kernel launch. Ganho tipico: 20-40% em throughput
    # para modelos com muitas operacoes elementares (ResNets, Transformers).
    # Default False (opt-in): nem todas as operacoes TF suportam XLA
    # (ex: tf.py_function, custom ops). Se uma op nao-suportada for
    # encontrada, TF gera erro de compilacao em tempo de build.
    # Recomendado para: ResNet-18, ConvNeXt, Transformer, PatchTST.
    # NAO recomendado para: modelos com tf.py_function ou ops dinamicas.
    # Ref: XLA overview (tensorflow.org/xla) — fusao de ops, eliminacao
    #   de materialization, otimizacao de layout de memoria.
    # Ref: Sabne (2020) "XLA: Compiling Machine Learning for Peak
    #   Performance" — benchmarks em GPUs V100/A100.
    # Nota: v2.0.1 (2026-03 — GPU completeness, opt-in)
    use_xla: bool = False
    use_tensorboard: bool = True
    use_csv_logger: bool = True
    # ── GradientMonitor — monitoramento de gradientes reais via GradientTape
    # Habilita o GradientMonitor callback que computa normas de gradiente
    # reais (nao normas de pesos) via tf.GradientTape em um batch de amostra.
    # Util para diagnosticar gradient explosion (norma > threshold) ou
    # gradient vanishing (norma < threshold) durante o treinamento.
    # gradient_monitor_freq: frequencia de amostragem (a cada N epocas).
    # Manter freq >= 5 para evitar overhead significativo de GradientTape.
    # Ref: Goodfellow et al. (2016) Deep Learning — secao 8.2.5.
    # Nota: v2.0.2 (2026-03 — Fase E, GAP 7 corrigido)
    use_gradient_monitor: bool = False
    gradient_monitor_freq: int = 5
    gradient_explosion_threshold: float = 100.0
    gradient_vanishing_threshold: float = 1e-7

    # ══════════════════════════════════════════════════════════════════
    # SECAO 10: N-STAGE TRAINING
    # Treinamento em estagios progressivos de ruido (S21+).
    # Stage 1: convergencia clean (nstage_stage1_epochs epocas).
    # Stages 2..N: noise crescente com LR e patience auto-calculados.
    # Mutuamente exclusivo com curriculum (validado em __post_init__).
    # stage_lr_decay = 0.5 → Stage 2 LR = LR * 0.5.
    # use_stage_mini_curriculum: rampa curta dentro de cada stage.
    # ══════════════════════════════════════════════════════════════════
    use_nstage: bool = False
    n_training_stages: int = 2
    nstage_stage1_epochs: int = 15
    stage_lr_decay: float = 0.5
    nstage_base_patience: int = 30
    use_stage_mini_curriculum: bool = True
    stage_ramp_fraction: float = 0.25

    # ══════════════════════════════════════════════════════════════════
    # SECAO 11: LOSS
    # 26 funcoes de perda no LossFactory (13 gen + 4 geo + 9 adv).
    # loss_type seleciona a loss base via LossFactory.get(config).
    # Composicao: combined = base + look_ahead + DTB + PINNs.
    # TARGET_SCALING-aware: losses operam no dominio scaled (log10).
    #
    # Sub-secoes:
    #   11a — Loss base + combinacao (loss_type, look_ahead, DTB, PINNs)
    #   11b — Geofisicas (#14-#17): thresholds, gangorra, robust
    #   11c — Geosteering (#19): look-ahead decay
    #   11d — Avancadas (#20-#26): DILATE, Sobolev, Spectral, Morales
    # ══════════════════════════════════════════════════════════════════

    # ── 11a: Loss base + combinacao ──────────────────────────────────
    # loss_alpha/beta/gamma: pesos para log_scale_aware (#14) —
    #   alpha = interface, beta = oscilacao, gamma = subestimacao.
    # morales_physics_omega: peso MSE vs MAE no Morales hybrid (#26).
    loss_type: str = "rmse"
    loss_alpha: float = 1.0
    loss_beta: float = 0.5
    loss_gamma: float = 0.1
    use_look_ahead_loss: bool = False
    look_ahead_weight: float = 0.1
    use_dtb_loss: bool = False
    dtb_weight: float = 0.1
    use_pinns: bool = False
    pinns_lambda: float = 0.01
    use_morales_hybrid_loss: bool = False
    morales_physics_omega: float = 0.85

    # ── 11b: Geofisicas (#14-#17) — thresholds e pesos ──────────────
    # penalty_warmup_epochs: epocas iniciais com RMSE/Huber puro
    #   antes de ativar penalidades fisicas (rampa 0→1).
    # interface_threshold: |gradiente| em dominio scaled acima do qual
    #   um ponto e considerado transicao geologica (boundary).
    # high_rho_threshold: resistividade (Ohm.m) acima da qual zona e
    #   considerada "alta rho" (OscillationPenalty ativa). Log10 interno.
    # low_rho_threshold: resistividade (Ohm.m) abaixo da qual zona e
    #   considerada condutiva (UnderestimationPenalty ativa). Log10 interno.
    penalty_warmup_epochs: int = 10
    interface_threshold: float = 0.5
    high_rho_threshold: float = 300.0
    low_rho_threshold: float = 50.0

    # ── Gangorra noise-aware (#15) — beta varia com noise_level ──────
    # gangorra_beta_min: peso oscilacao quando noise=0 (fitting agressivo).
    # gangorra_beta_max: peso oscilacao quando noise=max (suavidade).
    # gangorra_max_noise: noise_level esperado para saturacao do ratio.
    gangorra_beta_min: float = 0.1
    gangorra_beta_max: float = 0.5
    gangorra_max_noise: float = 0.1

    # ── Robust (#16-#17) — pesos Huber + suavidade global ────────────
    # Mesmo significado que loss_alpha/beta/gamma, mas para Huber base.
    # robust_delta_smooth: penalidade TV (Total Variation) 1a ordem.
    robust_alpha: float = 0.15
    robust_beta: float = 0.1
    robust_gamma: float = 0.15
    robust_delta_smooth: float = 0.05

    # ── 11c: Geosteering (#19) — look-ahead decay ───────────────────
    # Decaimento exponencial: w[i] = exp(-decay * i / N).
    # Valores maiores concentram mais peso nos pontos proximos a broca.
    look_ahead_decay_rate: float = 10.0

    # ── 11d: Avancadas (#20-#26) — DILATE, Sobolev, etc. ────────────
    # dilate_alpha: peso Soft-DTW vs TDI (0=TDI puro, 1=DTW puro).
    # dilate_gamma: suavizacao do Soft-DTW (menor = mais DTW-like).
    # dilate_downsample: fator de reducao temporal para eficiencia.
    dilate_alpha: float = 0.5
    dilate_gamma: float = 0.01
    dilate_downsample: int = 10
    # enc_decoder_recon_weight: peso da loss de reconstrucao.
    enc_decoder_recon_weight: float = 0.1
    # sobolev_lambda: peso da penalidade de gradiente (Sobolev H1).
    sobolev_lambda: float = 0.1
    # cross_gradient_lambda: peso da regularizacao cruzada rho_h/rho_v.
    cross_gradient_lambda: float = 0.1
    # spectral_lambda: peso do MSE no espaco de frequencias (FFT).
    spectral_lambda: float = 0.5
    # Morales adaptive omega: rampa omega_initial → morales_physics_omega.
    use_adaptive_omega: bool = False
    morales_omega_initial: float = 0.15
    morales_ramp_epochs: int = 50

    # ══════════════════════════════════════════════════════════════════
    # SECAO 12: REGULARIZACAO
    # Tecnicas de regularizacao explicitas (alem de weight_decay).
    # dropout_rate: dropout global (0.0 = desativado).
    # L1 promove esparsidade; L2 penaliza pesos grandes.
    # ElasticNet (L1+L2) via ambas flags ativas.
    # use_l2 = False no cenario E-Robusto (weight_decay ja regulariza).
    # ══════════════════════════════════════════════════════════════════
    dropout_rate: float = 0.0
    use_l2_regularization: bool = False
    l2_weight: float = 1e-4
    use_l1_regularization: bool = False
    l1_weight: float = 1e-5

    # ══════════════════════════════════════════════════════════════════
    # SECAO 13: AVALIACAO E VISUALIZACAO
    # Plots de holdout: amostras clean+noisy para diagnostico visual.
    # holdout_plots_max_samples: limite de amostras exibidas.
    # holdout_plots_dpi: resolucao para publicacao (300 = print).
    # verbose: ativa logging detalhado durante treinamento.
    # ══════════════════════════════════════════════════════════════════
    use_holdout_plots: bool = True
    holdout_plots_max_samples: int = 5
    holdout_plots_dpi: int = 300
    verbose: bool = True

    # ══════════════════════════════════════════════════════════════════
    # SECAO 14: OPTUNA (OPT-IN)
    # Otimizacao de hiperparametros via Optuna (opt-in, default off).
    # TPE (Tree-structured Parzen Estimators) como sampler padrao.
    # Median pruner descarta trials com desempenho abaixo da mediana.
    # optuna_timeout: tempo maximo em segundos (1h default).
    # Weight reset entre trials para evitar path-dependency.
    # ══════════════════════════════════════════════════════════════════
    use_optuna: bool = False
    optuna_n_trials: int = 50
    optuna_timeout: int = 3600
    optuna_sampler: str = "tpe"
    optuna_pruner: str = "median"

    # ══════════════════════════════════════════════════════════════════
    # SECAO 15: PATHS
    # Caminhos de diretorio para dados, experimentos e artefatos.
    # base_dir: raiz no Google Drive (Colab Pro+).
    # dataset_dir: diretorio contendo arquivos .out e .dat.
    # experiment_dir: isolamento por experimento (subdirs automaticos).
    # experiment_tag: tag unica para rastreabilidade (ex: "s21_robusto").
    # ══════════════════════════════════════════════════════════════════
    base_dir: str = "/content/drive/MyDrive/Geosteering_AI"
    dataset_dir: Optional[str] = None
    experiment_dir: Optional[str] = None
    experiment_tag: Optional[str] = None

    # ══════════════════════════════════════════════════════════════════
    # SECAO 16: VALIDACAO (__post_init__)
    # Validacao centralizada fail-fast: errata, ranges, mutual
    # exclusivity, inference mode, enum values.
    # Executada automaticamente na criacao de qualquer instancia.
    # AssertionError com mensagem descritiva em caso de violacao.
    # ══════════════════════════════════════════════════════════════════

    def __post_init__(self):
        """Validacao centralizada — fail-fast para errata e inconsistencias.

        Note:
            Referenciado em:
                - tests/test_config.py: TestErrata (6 test cases),
                  TestMutualExclusivity (3 test cases),
                  TestValidation (9 test cases)
            Ref: CLAUDE.md secao "Valores Fisicos Criticos (Errata Imutavel)".
            Errata v4.4.5: frequency_hz=20000.0, spacing_meters=1.0,
            sequence_length=600.
            Errata v5.0.15: input_features=[1,4,5,20,21], output_targets=[2,3],
            target_scaling="log10".
            Mutual exclusivity: use_nstage e use_curriculum (NUNCA ambos True).
        """
        # ── Errata v4.4.5 (valores fisicos imutaveis) ────────────────
        assert (
            self.frequency_hz == 20000.0
        ), "Errata v4.4.5: FREQUENCY_HZ DEVE ser 20000.0 (NUNCA 2.0)"
        assert (
            self.spacing_meters == 1.0
        ), "Errata v4.4.5: SPACING_METERS DEVE ser 1.0 (NUNCA 1000.0)"
        assert self.sequence_length == 600, "SEQUENCE_LENGTH DEVE ser 600 (NUNCA 601)"
        assert (
            self.target_scaling == "log10"
        ), "TARGET_SCALING DEVE ser 'log10' (NUNCA 'log')"

        # ── Errata v5.0.15 (mapeamento 22-col) ──────────────────────
        # Validacao semantica: baseline [1,4,5,20,21] OBRIGATORIO como
        # subconjunto. Features adicionais (Hxy, Hxz, etc.) permitidas.
        # Indices do formato antigo 9-col (0,3,7,8) PROIBIDOS.
        # Ref: docs/physics/errata_valores.md, CLAUDE.md.
        #
        #   ┌──────────────────────────────────────────────────────────┐
        #   │  BASELINE OBRIGATORIO (z_obs + Hxx + Hzz):             │
        #   │    col 1  = z_obs (profundidade)                        │
        #   │    col 4,5 = Re(Hxx), Im(Hxx) (planar)                │
        #   │    col 20,21 = Re(Hzz), Im(Hzz) (axial)               │
        #   │                                                          │
        #   │  EXTENSOES PERMITIDAS (off-diagonal do tensor H):       │
        #   │    col 6,7 = Hxy   col 8,9 = Hxz   col 10,11 = Hyx   │
        #   │    col 12,13 = Hyy  col 14,15 = Hyz  col 16,17 = Hzx  │
        #   │    col 18,19 = Hzy                                      │
        #   │                                                          │
        #   │  PROIBIDO:                                               │
        #   │    col 0 = meds (contador de metadata, NUNCA feature)   │
        #   │                                                          │
        #   │  O formato antigo 9-col [0,3,4,7,8] eh rejeitado por: │
        #   │    - col 0 proibida (metadata)                           │
        #   │    - col 3 overlap com targets [2,3]                    │
        #   │    - baseline {1,4,5,20,21} obrigatorio como subconjunto│
        #   └──────────────────────────────────────────────────────────┘
        _BASELINE_REQUIRED = {1, 4, 5, 20, 21}
        _FORBIDDEN_METADATA = {0}  # col 0 = meds (contador, NUNCA feature)
        _input_set = set(self.input_features)

        _meta_found = _input_set & _FORBIDDEN_METADATA
        assert not _meta_found, (
            f"INPUT_FEATURES contem indice de metadata (col 0 = meds): "
            f"formato antigo 9-col detectado. Use formato 22-col com baseline "
            f"[1,4,5,20,21]"
        )

        _missing_baseline = _BASELINE_REQUIRED - _input_set
        assert not _missing_baseline, (
            f"INPUT_FEATURES deve conter baseline {{1,4,5,20,21}} como subconjunto. "
            f"Faltando: {sorted(_missing_baseline)}"
        )

        assert all(0 <= i < self.n_columns for i in self.input_features), (
            f"INPUT_FEATURES indices devem estar no range [0, {self.n_columns}). "
            f"Recebido: {self.input_features}"
        )

        assert self.output_targets == [2, 3], "OUTPUT_TARGETS 22-col: [2,3] (NUNCA [1,2])"
        assert _input_set & set(self.output_targets) == set(), (
            f"Features e targets NAO podem ter overlap. "
            f"Overlap: {sorted(_input_set & set(self.output_targets))}"
        )

        # ── Mutual exclusivity ───────────────────────────────────────
        if self.use_nstage:
            assert not self.use_curriculum, (
                "N-Stage e Curriculum sao mutuamente exclusivos. "
                "Use use_nstage=True OU use_curriculum=True, nao ambos."
            )

        # ── Ranges ───────────────────────────────────────────────────
        assert (
            0.0 <= self.noise_level_max <= 1.0
        ), f"noise_level_max deve estar em [0, 1], recebido: {self.noise_level_max}"
        assert self.batch_size > 0, "batch_size deve ser > 0"
        assert self.learning_rate > 0, "learning_rate deve ser > 0"
        assert self.epochs > 0, "epochs deve ser > 0"
        assert len(self.noise_types) == len(
            self.noise_weights
        ), "noise_types e noise_weights devem ter o mesmo tamanho"
        assert self.train_ratio > 0, "train_ratio deve ser > 0"
        assert self.val_ratio > 0, "val_ratio deve ser > 0"
        assert self.test_ratio >= 0, "test_ratio deve ser >= 0"
        assert (
            self.train_ratio + self.val_ratio + self.test_ratio <= 1.0 + 1e-9
        ), "Soma de train/val/test ratios deve ser <= 1.0"
        assert (
            self.eps_tf >= 1e-15
        ), "eps_tf deve ser >= 1e-15 (Errata v5.0.15: NUNCA 1e-30 para float32)"
        assert self.output_channels in (2, 4, 6), "output_channels deve ser 2, 4 ou 6"

        # ── GradientMonitor ranges ──────────────────────────────────
        # Validacao dos campos de monitoramento de gradientes (Fase E).
        # gradient_monitor_freq deve ser >= 1 para evitar divisao por zero.
        # Thresholds devem ser positivos e vanishing < explosion.
        # Ref: Review Fase E (H2) — config validation.
        if self.use_gradient_monitor:
            assert (
                self.gradient_monitor_freq >= 1
            ), f"gradient_monitor_freq deve ser >= 1, recebido: {self.gradient_monitor_freq}"
            assert (
                self.gradient_explosion_threshold > 0
            ), f"gradient_explosion_threshold deve ser > 0, recebido: {self.gradient_explosion_threshold}"
            assert (
                self.gradient_vanishing_threshold > 0
            ), f"gradient_vanishing_threshold deve ser > 0, recebido: {self.gradient_vanishing_threshold}"
            assert (
                self.gradient_vanishing_threshold < self.gradient_explosion_threshold
            ), (
                f"vanishing_threshold ({self.gradient_vanishing_threshold}) "
                f"deve ser < explosion_threshold ({self.gradient_explosion_threshold})"
            )

        # ── Inference mode valido ────────────────────────────────────
        _VALID_IM = {"offline", "realtime"}
        assert (
            self.inference_mode in _VALID_IM
        ), f"inference_mode '{self.inference_mode}' invalido. Validos: {_VALID_IM}"

        # ── Auto-derivacao: realtime → causal ────────────────────────
        if self.inference_mode == "realtime" and not self.use_causal_mode:
            object.__setattr__(self, "use_causal_mode", True)

        # ── Feature view valida ──────────────────────────────────────
        _VALID_FV = {
            "identity",
            "raw",
            "H1_logH2",
            "logH1_logH2",
            "IMH1_IMH2_razao",
            "IMH1_IMH2_lograzao",
        }
        assert (
            self.feature_view in _VALID_FV
        ), f"feature_view '{self.feature_view}' invalido. Validos: {_VALID_FV}"

        # ── Target scaling ──────────────────────────────────────────
        # Nota: target_scaling eh fixado em "log10" pela Errata v4.4.5
        # (validado acima). O enum completo (none, sqrt, cbrt, asinh, yj, pt)
        # sera habilitado quando a errata for flexibilizada em versao futura.

        # ── Scaler type valido ───────────────────────────────────────
        _VALID_SC = {
            "standard",
            "minmax",
            "robust",
            "maxabs",
            "quantile",
            "power",
            "normalizer",
            "none",
        }
        assert (
            self.scaler_type in _VALID_SC
        ), f"scaler_type '{self.scaler_type}' invalido. Validos: {_VALID_SC}"

        # ── Optimizer valido ─────────────────────────────────────────
        _VALID_OPT = {"adam", "adamw", "sgd", "rmsprop", "nadam", "adagrad"}
        assert (
            self.optimizer in _VALID_OPT
        ), f"optimizer '{self.optimizer}' invalido. Validos: {_VALID_OPT}"

        # ── Smoothing type valido ────────────────────────────────────
        _VALID_SM = {
            "none",
            "moving_average",
            "savitzky_golay",
            "gaussian",
            "median",
            "exponential",
            "lowess",
        }
        assert (
            self.smoothing_type in _VALID_SM
        ), f"smoothing_type '{self.smoothing_type}' invalido. Validos: {_VALID_SM}"

        # ── Frequencia normalizacao valida (P3) ────────────────────
        if self.use_freq_as_feature:
            _VALID_FN = {"log10", "khz", "raw"}
            assert self.freq_normalization in _VALID_FN, (
                f"freq_normalization '{self.freq_normalization}' invalido. "
                f"Validos: {_VALID_FN}"
            )

        # ── Static injection mode valido (A/B/C) ──────────────────
        _VALID_SIM = {"broadcast", "dual_input", "film"}
        assert self.static_injection_mode in _VALID_SIM, (
            f"static_injection_mode '{self.static_injection_mode}' invalido. "
            f"Validos: {_VALID_SIM}"
        )

        # ── FiLM restrito a arquiteturas compativeis ──────────────
        # FiLM requer injecao de modulacao (γ×h+β) nos blocos internos.
        # Arquiteturas com blocos auto-contidos (N-BEATS, N-HiTS) ou
        # dominio espectral (FNO) ou branch separado (DeepONet) sao
        # incompativeis por requerer adaptacao muito complexa.
        if self.static_injection_mode == "film":
            _FILM_INCOMPATIBLE = {
                "N_BEATS",
                "N_HiTS",  # Decomposition — blocks auto-contidos
                "FNO",  # Fourier — dominio espectral
                "DeepONet",  # Operador — branch separado
            }
            assert self.model_type not in _FILM_INCOMPATIBLE, (
                f"static_injection_mode='film' incompativel com model_type="
                f"'{self.model_type}'. Arquiteturas incompativeis com FiLM: "
                f"{sorted(_FILM_INCOMPATIBLE)}. Use 'dual_input' ou 'broadcast'."
            )

    # ══════════════════════════════════════════════════════════════════
    # SECAO 17: PROPRIEDADES DERIVADAS
    # Propriedades read-only calculadas a partir dos campos do config.
    # n_features: total de canais de entrada para o modelo.
    # needs_onthefly_fv_gs: indica se FV/GS devem ser computados
    # on-the-fly (apos noise) para fidelidade fisica.
    # needs_expanded_features: indica se colunas EM extras sao
    # necessarias para computar geosinais (off-diagonal).
    # ══════════════════════════════════════════════════════════════════

    @property
    def n_base_features(self) -> int:
        """Numero de features base de entrada (sem GS).

        Note:
            Referenciado em:
                - data/pipeline.py: DataPipeline.__init__ (self._n_em_features)
                - tests/test_config.py: TestDerivedProperties.test_n_base_features
            Valor: len(input_features) = 5 para [1, 4, 5, 20, 21].
        """
        return len(self.input_features)

    @property
    def n_geosignal_channels(self) -> int:
        """Numero de canais de geosinais (2 por familia: att + phase).

        Note:
            Referenciado em:
                - config.py: n_features (composicao n_base + n_gs)
                - tests/test_config.py: TestDerivedProperties.test_n_features_with_gs_usd_uhr
            Valor: 0 se GS off, 2*len(families) se GS on.
            Cada familia gera 2 canais: attenuation + phase_difference.
        """
        if not self.use_geosignal_features:
            return 0
        return 2 * len(self.resolve_families())

    @property
    def n_prefix(self) -> int:
        """Numero de colunas prefixo (theta, freq) antes de z_obs.

        Derivado automaticamente de use_theta_as_feature e use_freq_as_feature.
        Usado por feature_views.py (n_prefix), noise (separacao protegido/EM),
        e pipeline.py (offset para h1_cols/h2_cols).

        Note:
            Referenciado em:
                - data/pipeline.py: DataPipeline.__init__ (offset h1/h2)
                - data/pipeline.py: build_train_map_fn() (FV_tf n_prefix)
                - noise/functions.py: n_protected = n_prefix + 1
                - tests/test_config.py: TestThetaFreqFeatures
            Valor: 0 (P1 ou dual_input/film), 1 (P2 broadcast), 2 (P2+P3 broadcast).
            ZERO quando static_injection_mode != "broadcast" (escalares separados).
            Layout broadcast: [theta_norm?] [f_norm?] [z_obs] [EM...] [GS...]
            Ref: docs/physics/perspectivas.md secoes P2, P3.
        """
        if self.static_injection_mode != "broadcast":
            return 0  # dual_input/film: escalares separados, nao como prefixo
        return int(self.use_theta_as_feature) + int(self.use_freq_as_feature)

    @property
    def n_features(self) -> int:
        """Numero total de features (prefix + base + geosinais).

        Note:
            Referenciado em:
                - tests/test_config.py: TestDerivedProperties.test_n_features_without_gs,
                  TestDerivedProperties.test_n_features_with_gs_usd_uhr,
                  TestThetaFreqFeatures
            Valor: 5 (P1), 6 (P2 ou P3), 7 (P2+P3), 9 (P4 usd_uhr), etc.
            Composicao: n_prefix (0-2) + n_base_features (5+) + n_geosignal_channels (0+).
            Usado para validar shape do modelo Keras (input_shape).
        """
        return self.n_prefix + self.n_base_features + self.n_geosignal_channels

    @property
    def needs_onthefly_fv_gs(self) -> bool:
        """True se FV/GS devem ser computados on-the-fly (pos-noise).

        Ativo quando noise esta habilitado E (FV nao-identity OU GS ativo).
        Garante fidelidade fisica: GS computados de EM ruidoso.

        Note:
            Referenciado em:
                - data/pipeline.py: DataPipeline.is_onthefly (property)
                - data/pipeline.py: DataPipeline.prepare() (decisao offline
                  vs on-the-fly)
                - tests/test_config.py: TestDerivedProperties (3 test cases)
            Ref: docs/ARCHITECTURE_v2.md secao 4.3.
            Cadeia on-the-fly: noise → FV_tf → GS_tf → scale_tf.
            Cadeia offline: FV → GS → fit_scaler → scale (estatico).
        """
        return self.use_noise and (
            self.feature_view not in ("identity", "raw", None)
            or self.use_geosignal_features
        )

    @property
    def needs_expanded_features(self) -> bool:
        """True se precisa de colunas EM expandidas (alem de INPUT_FEATURES).

        Geosinais como USD e UHR requerem componentes off-diagonal
        (Hxz, Hzx, Hyy) que nao estao no INPUT_FEATURES base [1,4,5,20,21].

        Note:
            Referenciado em:
                - data/pipeline.py: DataPipeline.__init__ (compute_expanded_features)
            Valor: True se use_geosignal_features=True.
            Colunas expandidas: determinadas por compute_expanded_features()
            em data/geosignals.py com base nas familias ativas.
        """
        return self.use_geosignal_features

    # ══════════════════════════════════════════════════════════════════
    # SECAO 18: METODOS AUXILIARES
    # Resolucao de familias de geosinais a partir do geosignal_set
    # ou de lista explicita em geosignal_families.
    # Sets pre-definidos: usd_uhr (2), usd_uhr_uha (3),
    # full_1d (4), full_3d (5 com U3DF).
    # ══════════════════════════════════════════════════════════════════

    def resolve_families(self) -> List[str]:
        """Resolve lista de familias de geosinais ativas.

        Returns:
            Lista de nomes de familias (ex: ["USD", "UHR"]).

        Note:
            Referenciado em:
                - config.py: n_geosignal_channels (calcula 2*len(families))
                - data/pipeline.py: DataPipeline.__init__ (self._families)
                - tests/test_config.py: TestDerivedProperties.test_resolve_families_usd_uhr,
                  TestDerivedProperties.test_resolve_families_full_3d
            Ref: docs/ARCHITECTURE_v2.md secao 4.3 (principio P4).
            Sets pre-definidos: usd_uhr (2), usd_uhr_uha (3), full_1d (4),
            full_3d (5 com U3DF).
            geosignal_families override tem prioridade sobre geosignal_set.
        """
        _SET_MAP = {
            "usd_uhr": ["USD", "UHR"],
            "usd_uhr_uha": ["USD", "UHR", "UHA"],
            "full_1d": ["USD", "UAD", "UHR", "UHA"],
            "full_3d": ["USD", "UAD", "UHR", "UHA", "U3DF"],
        }
        if self.geosignal_families is not None:
            return list(self.geosignal_families)
        return _SET_MAP.get(self.geosignal_set, [])

    # ══════════════════════════════════════════════════════════════════
    # SECAO 19: PRESETS (metodos de classe)
    # Configuracoes pre-definidas para cenarios comuns.
    # baseline: P1 sem noise, para debugging e baseline.
    # robusto: E-Robusto S21, cenario padrao de producao.
    # nstage: N-Stage com estagios progressivos de noise.
    # geosinais_p4: P4 com geosinais on-the-fly.
    # realtime: geosteering causal para inferencia em tempo real.
    # ══════════════════════════════════════════════════════════════════

    @classmethod
    def baseline(cls) -> "PipelineConfig":
        """Preset P1 baseline — sem noise, sem GS, defaults conservadores.

        Cenario mais simples: 5 features EM, sem augmentation.
        Util para debugging e estabelecer baseline de performance.

        Note:
            Referenciado em:
                - tests/test_config.py: TestPresets.test_baseline
                - tests/test_data_pipeline.py: TestDecoupling, TestPipelineIntegration
                  (usado como config padrao para testes sem noise)
            Ref: docs/ARCHITECTURE_v2.md secao 3.1 (presets).
            FLAGS: use_noise=False, use_curriculum=False, use_dual_validation=False.
            Demais campos herdam defaults do PipelineConfig (E-Robusto).
        """
        return cls(
            use_noise=False,
            use_curriculum=False,
            use_dual_validation=False,
        )

    @classmethod
    def robusto(cls) -> "PipelineConfig":
        """Preset E-Robusto S21 — noise 8%, curriculum, LR 1e-4.

        Cenario padrao para producao: curriculum learning 3-phase,
        noise ate 8%, LR conservador, patience 60 epocas.
        Defaults do PipelineConfig ja correspondem ao E-Robusto.

        Note:
            Referenciado em:
                - tests/test_config.py: TestPresets.test_robusto,
                  TestSerialization.test_to_dict, test_copy_with_override,
                  test_yaml_roundtrip
                - configs/robusto.yaml: equivalente em YAML
            Ref: docs/ARCHITECTURE_v2.md secao 3.1 (presets).
            Todos os defaults do PipelineConfig = E-Robusto S21.
            Retorna cls() sem overrides (defaults sao o cenario robusto).
        """
        return cls()

    @classmethod
    def nstage(cls, n: int = 2, **kwargs) -> "PipelineConfig":
        """Preset N-Stage — treinamento em estagios progressivos de noise.

        Separa convergencia clean (Stage 1) de adaptacao a ruido
        (Stages 2..N). Cada stage tem noise, LR e patience auto-calculados.

        Args:
            n: Numero de estagios (>= 2). Default: 2.
            **kwargs: Overrides adicionais do config.

        Note:
            Referenciado em:
                - tests/test_config.py: TestPresets.test_nstage_n2,
                  TestPresets.test_nstage_n3
            Ref: docs/ARCHITECTURE_v2.md secao 3.1 (presets).
            Mutuamente exclusivo com curriculum (use_curriculum=False forcado).
            Auto-calculo por stage: noise_k, lr_k, patience_k (S21+).
        """
        return cls(
            use_nstage=True,
            n_training_stages=n,
            use_curriculum=False,
            **kwargs,
        )

    @classmethod
    def geosinais_p4(cls, geosignal_set: str = "usd_uhr", **kwargs) -> "PipelineConfig":
        """Preset P4 — geosinais on-the-fly com noise.

        Ativa geosinais computados de EM ruidoso (fisicamente correto).
        Requer cadeia on-the-fly: noise → FV → GS → scale.

        Args:
            geosignal_set: Conjunto de familias. Default: "usd_uhr".
            **kwargs: Overrides adicionais.

        Note:
            Referenciado em:
                - tests/test_config.py: TestPresets.test_geosinais_p4,
                  TestDerivedProperties.test_needs_onthefly_noise_and_gs,
                  TestDerivedProperties.test_n_features_with_gs_usd_uhr
                - tests/test_data_pipeline.py: TestPipelineIntegration.test_pipeline_init_geosinais
            Ref: docs/ARCHITECTURE_v2.md secao 3.1 (presets) e 4.3 (P4).
            needs_onthefly_fv_gs=True quando noise=True + GS=True.
            Cadeia on-the-fly: noise → FV_tf → GS_tf → scale_tf.
        """
        return cls(
            use_geosignal_features=True,
            geosignal_set=geosignal_set,
            **kwargs,
        )

    @classmethod
    def realtime(cls, model_type: str = "WaveNet", **kwargs) -> "PipelineConfig":
        """Preset geosteering realtime — modo causal.

        Ativa inferencia causal com sliding window. use_causal_mode
        e auto-derivado de inference_mode="realtime".

        Args:
            model_type: Arquitetura causal-native. Default: "WaveNet".
            **kwargs: Overrides adicionais.

        Note:
            Referenciado em:
                - tests/test_config.py: TestPresets.test_realtime,
                  TestPresets.test_realtime_custom_model
            Ref: docs/ARCHITECTURE_v2.md secao 3.1 (presets).
            inference_mode="realtime" auto-ativa use_causal_mode=True
            no __post_init__. Apenas arquiteturas causal-native sao
            compativeis (17 _CAUSAL_INCOMPATIBLE no ModelRegistry).
        """
        return cls(
            inference_mode="realtime",
            model_type=model_type,
            **kwargs,
        )

    # ══════════════════════════════════════════════════════════════════
    # SECAO 20: SERIALIZACAO (YAML + dict)
    # from_yaml/to_yaml: reproducibilidade total via arquivo YAML.
    # to_dict: conversao para dict (logging, JSON, comparacao).
    # copy: copia defensiva com overrides (pattern funcional).
    # Formato YAML preserva ordem dos campos (sort_keys=False).
    # ══════════════════════════════════════════════════════════════════

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Carrega configuracao de arquivo YAML.

        Args:
            path: Caminho para o arquivo .yaml.

        Returns:
            PipelineConfig com valores do arquivo.

        Raises:
            FileNotFoundError: Se o arquivo nao existir.
            AssertionError: Se valores violarem errata/validacao.

        Note:
            Referenciado em:
                - tests/test_config.py: TestSerialization.test_yaml_roundtrip,
                  TestSerialization.test_from_yaml_preset
            Ref: docs/ARCHITECTURE_v2.md secao 3.1.
            Requer pyyaml (lazy import — ImportError se ausente).
            Validacao completa no __post_init__ apos carregar.
            Arquivo YAML vazio resulta em defaults (E-Robusto).
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml necessario: pip install pyyaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Salva configuracao em arquivo YAML.

        Args:
            path: Caminho para o arquivo .yaml de saida.

        Note:
            Referenciado em:
                - tests/test_config.py: TestSerialization.test_yaml_roundtrip
            Ref: docs/ARCHITECTURE_v2.md secao 3.1.
            sort_keys=False preserva ordem dos campos do dataclass.
            Requer pyyaml (lazy import).
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml necessario: pip install pyyaml")
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionario Python.

        Returns:
            Dict com todos os campos e valores.

        Note:
            Referenciado em:
                - config.py: to_yaml() (serializa dict para YAML)
                - config.py: copy() (cria dict base para override)
                - config.py: inject_as_globals() (itera campos)
                - tests/test_config.py: TestSerialization.test_to_dict,
                  TestSerialization.test_yaml_roundtrip
            Usa dataclasses.asdict() para conversao recursiva.
        """
        return dataclasses.asdict(self)

    def copy(self, **overrides) -> "PipelineConfig":
        """Cria copia com overrides opcionais.

        Args:
            **overrides: Campos a sobrescrever na copia.

        Returns:
            Nova instancia de PipelineConfig.

        Example:
            >>> config_base = PipelineConfig.robusto()
            >>> config_lr = config_base.copy(learning_rate=3e-4)

        Note:
            Referenciado em:
                - tests/test_config.py: TestSerialization.test_copy_with_override
            Pattern funcional: original inalterado, nova instancia validada.
            Overrides passam por __post_init__ (validacao completa).
        """
        data = self.to_dict()
        data.update(overrides)
        return PipelineConfig(**data)

    # ══════════════════════════════════════════════════════════════════
    # SECAO 21: COMPATIBILIDADE COM NOTEBOOK LEGADO
    # Ponte de migracao: injeta campos do config como globals() no
    # namespace do notebook Colab. Permite que celulas C19-C47
    # funcionem sem modificacao durante o periodo de transicao.
    # Cada campo field_name vira FIELD_NAME no namespace global.
    # TEMPORARIO — sera removido quando migracao estiver completa.
    # ══════════════════════════════════════════════════════════════════

    def inject_as_globals(self, namespace: Optional[dict] = None) -> None:
        """Injeta FLAGS no namespace global para celulas legadas.

        Permite que celulas C19-C47 funcionem sem modificacao durante
        o periodo de migracao. Cada campo do config e injetado como
        variavel global em UPPER_CASE.

        Args:
            namespace: Dict de namespace (default: globals() do chamador).
                       Em notebook Colab, usar globals().

        Example:
            >>> config = PipelineConfig.robusto()
            >>> config.inject_as_globals(globals())
            >>> print(LEARNING_RATE)  # 0.0001

        Note:
            Referenciado em:
                - Notebook Colab (celulas legadas C19-C47)
            TEMPORARIO — sera removido quando migracao v2.0 estiver completa.
            Cada campo field_name vira FIELD_NAME (upper case) no namespace.
            Usa inspect.currentframe() para detectar namespace do chamador.
        """
        if namespace is None:
            import inspect

            frame = inspect.currentframe()
            if frame is None or frame.f_back is None:
                raise RuntimeError(
                    "inject_as_globals: nao foi possivel detectar namespace "
                    "do chamador. Passe namespace=globals() explicitamente."
                )
            namespace = frame.f_back.f_globals
        for field_name, value in self.to_dict().items():
            flag_name = field_name.upper()
            namespace[flag_name] = value

    # ══════════════════════════════════════════════════════════════════
    # SECAO 22: REPRESENTACAO (__repr__)
    # Exibe campos principais de forma legivel para logging/debug.
    # Inclui: model, mode, noise, curriculum, nstage, FV, GS, LR,
    # epochs, batch_size, loss, n_features, output_channels.
    # ══════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        """Representacao legivel com campos principais."""
        lines = [f"PipelineConfig("]
        lines.append(f"  model_type='{self.model_type}',")
        lines.append(f"  inference_mode='{self.inference_mode}',")
        lines.append(f"  use_noise={self.use_noise}, noise_max={self.noise_level_max},")
        lines.append(f"  use_curriculum={self.use_curriculum},")
        lines.append(
            f"  use_nstage={self.use_nstage}, n_stages={self.n_training_stages},"
        )
        lines.append(f"  feature_view='{self.feature_view}',")
        lines.append(f"  use_geosignal={self.use_geosignal_features},")
        lines.append(
            f"  lr={self.learning_rate}, epochs={self.epochs}, bs={self.batch_size},"
        )
        lines.append(f"  loss='{self.loss_type}',")
        lines.append(f"  n_features={self.n_features}, output_ch={self.output_channels},")
        lines.append(f")")
        return "\n".join(lines)
