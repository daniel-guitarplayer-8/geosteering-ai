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
    Referencia: docs/ARCHITECTURE_v2.md secao 4.1.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy


@dataclass
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
    """

    # ══════════════════════════════════════════════════════════════════
    # FISICA (Errata v4.4.5 + v5.0.15 — valores imutaveis)
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
    # DADOS E SPLIT
    # ══════════════════════════════════════════════════════════════════
    split_by_model: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    use_dual_validation: bool = True
    global_seed: int = 42

    # ══════════════════════════════════════════════════════════════════
    # FEATURE VIEWS E GEOSINAIS
    # ══════════════════════════════════════════════════════════════════
    feature_view: str = "identity"
    use_geosignal_features: bool = False
    geosignal_set: str = "usd_uhr"
    geosignal_families: Optional[List[str]] = None
    eps_tf: float = 1e-12

    # ══════════════════════════════════════════════════════════════════
    # DECOUPLING EM
    # ══════════════════════════════════════════════════════════════════
    decoupling_hxx: bool = True
    decoupling_hyy: bool = True
    decoupling_hzz: bool = True
    decoupling_full_tensor: bool = False

    # ══════════════════════════════════════════════════════════════════
    # SCALING
    # ══════════════════════════════════════════════════════════════════
    scaler_type: str = "standard"
    use_per_group_scalers: bool = True
    gs_scaler_type: str = "robust"
    use_separate_z_scaler: bool = False
    smoothing_type: str = "none"

    # ══════════════════════════════════════════════════════════════════
    # NOISE (ON-THE-FLY EXCLUSIVO)
    # ══════════════════════════════════════════════════════════════════
    use_noise: bool = True
    noise_level_max: float = 0.08
    noise_types: List[str] = field(default_factory=lambda: ["gaussian"])
    noise_weights: List[float] = field(default_factory=lambda: [1.0])
    use_curriculum: bool = True
    epochs_no_noise: int = 10
    noise_ramp_epochs: int = 80

    # ══════════════════════════════════════════════════════════════════
    # ARQUITETURA
    # ══════════════════════════════════════════════════════════════════
    model_type: str = "ResNet_18"
    inference_mode: str = "offline"
    use_causal_mode: bool = False
    output_channels: int = 2
    use_physical_constraint_layer: bool = False
    constraint_activation: str = "softplus"
    arch_params: Optional[Dict[str, Any]] = None

    # ══════════════════════════════════════════════════════════════════
    # SKIP CONNECTIONS E BLOCOS
    # ══════════════════════════════════════════════════════════════════
    use_skip_connections: bool = True
    skip_connection_type: str = "add"
    use_se_block: bool = False
    se_reduction: int = 16

    # ══════════════════════════════════════════════════════════════════
    # TREINAMENTO
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
    use_tensorboard: bool = True
    use_csv_logger: bool = True

    # ══════════════════════════════════════════════════════════════════
    # N-STAGE TRAINING
    # ══════════════════════════════════════════════════════════════════
    use_nstage: bool = False
    n_training_stages: int = 2
    nstage_stage1_epochs: int = 15
    stage_lr_decay: float = 0.5
    nstage_base_patience: int = 30
    use_stage_mini_curriculum: bool = True
    stage_ramp_fraction: float = 0.25

    # ══════════════════════════════════════════════════════════════════
    # LOSS
    # ══════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════
    # REGULARIZACAO
    # ══════════════════════════════════════════════════════════════════
    dropout_rate: float = 0.0
    use_l2_regularization: bool = False
    l2_weight: float = 1e-4
    use_l1_regularization: bool = False
    l1_weight: float = 1e-5

    # ══════════════════════════════════════════════════════════════════
    # AVALIACAO E VISUALIZACAO
    # ══════════════════════════════════════════════════════════════════
    use_holdout_plots: bool = True
    holdout_plots_max_samples: int = 5
    holdout_plots_dpi: int = 300
    verbose: bool = True

    # ══════════════════════════════════════════════════════════════════
    # OPTUNA (OPT-IN)
    # ══════════════════════════════════════════════════════════════════
    use_optuna: bool = False
    optuna_n_trials: int = 50
    optuna_timeout: int = 3600
    optuna_sampler: str = "tpe"
    optuna_pruner: str = "median"

    # ══════════════════════════════════════════════════════════════════
    # PATHS
    # ══════════════════════════════════════════════════════════════════
    base_dir: str = "/content/drive/MyDrive/Geosteering_AI"
    dataset_dir: Optional[str] = None
    experiment_dir: Optional[str] = None
    experiment_tag: Optional[str] = None

    # ══════════════════════════════════════════════════════════════════
    # VALIDACAO
    # ══════════════════════════════════════════════════════════════════

    def __post_init__(self):
        """Validacao centralizada — fail-fast para errata e inconsistencias."""
        # ── Errata v4.4.5 (valores fisicos imutaveis) ────────────────
        assert self.frequency_hz == 20000.0, \
            "Errata v4.4.5: FREQUENCY_HZ DEVE ser 20000.0 (NUNCA 2.0)"
        assert self.spacing_meters == 1.0, \
            "Errata v4.4.5: SPACING_METERS DEVE ser 1.0 (NUNCA 1000.0)"
        assert self.sequence_length == 600, \
            "SEQUENCE_LENGTH DEVE ser 600 (NUNCA 601)"
        assert self.target_scaling == "log10", \
            "TARGET_SCALING DEVE ser 'log10' (NUNCA 'log')"

        # ── Errata v5.0.15 (mapeamento 22-col) ──────────────────────
        assert self.input_features == [1, 4, 5, 20, 21], \
            "INPUT_FEATURES 22-col: [1,4,5,20,21] (NUNCA [0,3,4,7,8])"
        assert self.output_targets == [2, 3], \
            "OUTPUT_TARGETS 22-col: [2,3] (NUNCA [1,2])"
        assert set(self.input_features) & set(self.output_targets) == set(), \
            "Features e targets NAO podem ter overlap"

        # ── Mutual exclusivity ───────────────────────────────────────
        if self.use_nstage:
            assert not self.use_curriculum, \
                "N-Stage e Curriculum sao mutuamente exclusivos. " \
                "Use use_nstage=True OU use_curriculum=True, nao ambos."

        # ── Ranges ───────────────────────────────────────────────────
        assert 0.0 <= self.noise_level_max <= 1.0, \
            f"noise_level_max deve estar em [0, 1], recebido: {self.noise_level_max}"
        assert self.batch_size > 0, "batch_size deve ser > 0"
        assert self.learning_rate > 0, "learning_rate deve ser > 0"
        assert self.epochs > 0, "epochs deve ser > 0"
        assert len(self.noise_types) == len(self.noise_weights), \
            "noise_types e noise_weights devem ter o mesmo tamanho"
        assert self.train_ratio + self.val_ratio + self.test_ratio <= 1.0 + 1e-9, \
            "Soma de train/val/test ratios deve ser <= 1.0"
        assert self.eps_tf > 0, "eps_tf deve ser > 0 (recomendado: 1e-12)"
        assert self.output_channels in (2, 4, 6), \
            "output_channels deve ser 2, 4 ou 6"

        # ── Auto-derivacao: realtime → causal ────────────────────────
        if self.inference_mode == "realtime" and not self.use_causal_mode:
            object.__setattr__(self, "use_causal_mode", True)

        # ── Feature view valida ──────────────────────────────────────
        _VALID_FV = {"identity", "raw", "H1_logH2", "logH1_logH2",
                     "IMH1_IMH2_razao", "IMH1_IMH2_lograzao", "IMH1_IMH2_fases"}
        assert self.feature_view in _VALID_FV, \
            f"feature_view '{self.feature_view}' invalido. Validos: {_VALID_FV}"

        # ── Target scaling valido ────────────────────────────────────
        _VALID_TS = {"none", "linear", "log10", "sqrt", "cbrt", "asinh", "yj", "pt"}
        assert self.target_scaling in _VALID_TS, \
            f"target_scaling '{self.target_scaling}' invalido. Validos: {_VALID_TS}"

    # ══════════════════════════════════════════════════════════════════
    # PROPRIEDADES DERIVADAS
    # ══════════════════════════════════════════════════════════════════

    @property
    def n_base_features(self) -> int:
        """Numero de features base de entrada (sem GS)."""
        return len(self.input_features)

    @property
    def n_geosignal_channels(self) -> int:
        """Numero de canais de geosinais (2 por familia: att + phase)."""
        if not self.use_geosignal_features:
            return 0
        return 2 * len(self.resolve_families())

    @property
    def n_features(self) -> int:
        """Numero total de features (base + geosinais)."""
        return self.n_base_features + self.n_geosignal_channels

    @property
    def needs_onthefly_fv_gs(self) -> bool:
        """True se FV/GS devem ser computados on-the-fly (pos-noise).

        Ativo quando noise esta habilitado E (FV nao-identity OU GS ativo).
        Garante fidelidade fisica: GS computados de EM ruidoso.
        """
        return (
            self.use_noise
            and (self.feature_view not in ("identity", "raw", None)
                 or self.use_geosignal_features)
        )

    @property
    def needs_expanded_features(self) -> bool:
        """True se precisa de colunas EM expandidas (alem de INPUT_FEATURES).

        Geosinais como USD e UHR requerem componentes off-diagonal
        (Hxz, Hzx, Hyy) que nao estao no INPUT_FEATURES base [1,4,5,20,21].
        """
        return self.use_geosignal_features

    # ══════════════════════════════════════════════════════════════════
    # METODOS AUXILIARES
    # ══════════════════════════════════════════════════════════════════

    def resolve_families(self) -> List[str]:
        """Resolve lista de familias de geosinais ativas.

        Returns:
            Lista de nomes de familias (ex: ["USD", "UHR"]).
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
    # PRESETS
    # ══════════════════════════════════════════════════════════════════

    @classmethod
    def baseline(cls) -> "PipelineConfig":
        """Preset P1 baseline — sem noise, sem GS, defaults conservadores.

        Cenario mais simples: 5 features EM, sem augmentation.
        Util para debugging e estabelecer baseline de performance.
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
        """
        return cls(
            inference_mode="realtime",
            model_type=model_type,
            **kwargs,
        )

    # ══════════════════════════════════════════════════════════════════
    # SERIALIZACAO
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
        """
        data = self.to_dict()
        data.update(overrides)
        return PipelineConfig(**data)

    # ══════════════════════════════════════════════════════════════════
    # COMPATIBILIDADE COM NOTEBOOK LEGADO
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
        """
        if namespace is None:
            import inspect
            namespace = inspect.currentframe().f_back.f_globals
        for field_name, value in self.to_dict().items():
            flag_name = field_name.upper()
            namespace[flag_name] = value

    def __repr__(self) -> str:
        """Representacao legivel com campos principais."""
        lines = [f"PipelineConfig("]
        lines.append(f"  model_type='{self.model_type}',")
        lines.append(f"  inference_mode='{self.inference_mode}',")
        lines.append(f"  use_noise={self.use_noise}, noise_max={self.noise_level_max},")
        lines.append(f"  use_curriculum={self.use_curriculum},")
        lines.append(f"  use_nstage={self.use_nstage}, n_stages={self.n_training_stages},")
        lines.append(f"  feature_view='{self.feature_view}',")
        lines.append(f"  use_geosignal={self.use_geosignal_features},")
        lines.append(f"  lr={self.learning_rate}, epochs={self.epochs}, bs={self.batch_size},")
        lines.append(f"  loss='{self.loss_type}',")
        lines.append(f"  n_features={self.n_features}, output_ch={self.output_channels},")
        lines.append(f")")
        return "\n".join(lines)
