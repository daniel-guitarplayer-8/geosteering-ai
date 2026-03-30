# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: models/registry.py                                                ║
# ║  Bloco: 3k — ModelRegistry (44 arquiteturas)                             ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • ModelRegistry: dicionario de 44 funcoes build_* + metadados         ║
# ║    • build_model(config): factory central — ponto unico de construcao     ║
# ║    • get_model_info(name): metadados (familia, tier, causal_compat)       ║
# ║    • list_available_models(): lista todas as 44 arquiteturas              ║
# ║    • is_causal_compatible(name): verifica compatibilidade causal           ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), models/*.py (44 builders)      ║
# ║  Exports: 5 simbolos — ver __all__                                        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5.11, legado C37                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (44 entradas)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""ModelRegistry: factory central para as 44 arquiteturas.

Mapeamento model_type (str) → funcao build_*(config: PipelineConfig).
Metadados por arquitetura: familia, tier, causal_compatible.

Catalogo:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Familia       │ Count │ Arquiteturas                                │
    ├──────────────────────────────────────────────────────────────────────┤
    │  CNN           │  7    │ ResNet_18★, ResNet_34, ResNet_50,          │
    │                │       │ ConvNeXt, InceptionNet, InceptionTime, CNN_1D │
    │  TCN           │  2    │ TCN, TCN_Advanced                          │
    │  RNN           │  2    │ LSTM, BiLSTM                               │
    │  Hybrid        │  2    │ CNN_LSTM, CNN_BiLSTM_ED                   │
    │  UNet          │ 14    │ UNet_Base, UNet_Attention,                 │
    │                │       │ UNet_{ResNet18/34/50, ConvNeXt,            │
    │                │       │ Inception, EfficientNet} × {Base, Attn}   │
    │  Transformer   │  6    │ Transformer, Simple_TFT, TFT, PatchTST,   │
    │                │       │ Autoformer, iTransformer                   │
    │  Decomposition │  2    │ N_BEATS, N_HiTS                           │
    │  Advanced      │  4    │ DNN, FNO, DeepONet, Geophysical_Attention  │
    │  Geosteering   │  5    │ WaveNet, Causal_Transformer, Informer,     │
    │                │       │ Mamba_S4, Encoder_Forecaster              │
    └──────────────────────────────────────────────────────────────────────┘

Note:
    Referenciado em:
        - training/loop.py: TrainingLoop.run() — build_model(config)
        - tests/test_models.py: TestRegistry
    Legado C37 (1069 linhas).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# SECAO: CONJUNTOS DE METADADOS
# ════════════════════════════════════════════════════════════════════════════
# _CAUSAL_INCOMPATIBLE: arquiteturas que NAO podem ser usadas em realtime.
# _FAMILIES: mapeamento arquitetura → familia.
# _TIERS: classificacao por maturidade/performance.
# ──────────────────────────────────────────────────────────────────────────

# ── Arquiteturas incompativeis com modo causal ────────────────────────────
# BiLSTM usa backward pass; U-Nets usam skips do encoder inteiro;
# CNN_BiLSTM_ED usa BiLSTM bidirecional.
_CAUSAL_INCOMPATIBLE: frozenset = frozenset(
    {
        "BiLSTM",
        "CNN_BiLSTM_ED",
        "UNet_Base",
        "UNet_Attention",
        "UNet_ResNet18",
        "UNet_Attention_ResNet18",
        "UNet_ResNet34",
        "UNet_Attention_ResNet34",
        "UNet_ResNet50",
        "UNet_Attention_ResNet50",
        "UNet_ConvNeXt",
        "UNet_Attention_ConvNeXt",
        "UNet_Inception",
        "UNet_Attention_Inception",
        "UNet_EfficientNet",
        "UNet_Attention_EfficientNet",
        "N_BEATS",
        "N_HiTS",
    }
)

# ── Familias das 44 arquiteturas ──────────────────────────────────────────
_FAMILIES: Dict[str, str] = {
    # CNN
    "ResNet_18": "CNN",
    "ResNet_34": "CNN",
    "ResNet_50": "CNN",
    "ConvNeXt": "CNN",
    "InceptionNet": "CNN",
    "InceptionTime": "CNN",
    "CNN_1D": "CNN",
    # TCN
    "TCN": "TCN",
    "TCN_Advanced": "TCN",
    # RNN
    "LSTM": "RNN",
    "BiLSTM": "RNN",
    # Hybrid
    "CNN_LSTM": "Hybrid",
    "CNN_BiLSTM_ED": "Hybrid",
    # UNet (14)
    "UNet_Base": "UNet",
    "UNet_Attention": "UNet",
    "UNet_ResNet18": "UNet",
    "UNet_Attention_ResNet18": "UNet",
    "UNet_ResNet34": "UNet",
    "UNet_Attention_ResNet34": "UNet",
    "UNet_ResNet50": "UNet",
    "UNet_Attention_ResNet50": "UNet",
    "UNet_ConvNeXt": "UNet",
    "UNet_Attention_ConvNeXt": "UNet",
    "UNet_Inception": "UNet",
    "UNet_Attention_Inception": "UNet",
    "UNet_EfficientNet": "UNet",
    "UNet_Attention_EfficientNet": "UNet",
    # Transformer (6)
    "Transformer": "Transformer",
    "Simple_TFT": "Transformer",
    "TFT": "Transformer",
    "PatchTST": "Transformer",
    "Autoformer": "Transformer",
    "iTransformer": "Transformer",
    # Decomposition
    "N_BEATS": "Decomposition",
    "N_HiTS": "Decomposition",
    # Advanced
    "DNN": "Advanced",
    "FNO": "Advanced",
    "DeepONet": "Advanced",
    "Geophysical_Attention": "Advanced",
    # Geosteering (5)
    "WaveNet": "Geosteering",
    "Causal_Transformer": "Geosteering",
    "Informer": "Geosteering",
    "Mamba_S4": "Geosteering",
    "Encoder_Forecaster": "Geosteering",
}

# ── Tiers de maturidade ───────────────────────────────────────────────────
# Tier 1: validado, estavel, resultado publicado
# Tier 2: implementado, testado, performance conhecida
# Tier 3: experimental, pode precisar de tuning
_TIERS: Dict[str, int] = {
    "ResNet_18": 1,  # ★ Default validado
    "ResNet_34": 1,
    "ResNet_50": 2,
    "ConvNeXt": 1,
    "CNN_1D": 1,  # Baseline simples
    "InceptionNet": 2,
    "InceptionTime": 2,
    "TCN": 1,  # Causal nativo validado
    "TCN_Advanced": 2,
    "LSTM": 1,
    "BiLSTM": 2,
    "CNN_LSTM": 2,
    "CNN_BiLSTM_ED": 2,
    "Transformer": 2,
    "Simple_TFT": 2,
    "TFT": 3,
    "PatchTST": 2,
    "Autoformer": 3,
    "iTransformer": 3,
    "DNN": 1,  # Baseline ponto a ponto
    "FNO": 3,
    "DeepONet": 3,
    "Geophysical_Attention": 2,
    "WaveNet": 1,  # Causal geosteering validado
    "Causal_Transformer": 2,
    "Informer": 2,
    "Mamba_S4": 3,
    "Encoder_Forecaster": 2,
    "N_BEATS": 2,
    "N_HiTS": 2,
    # U-Nets: todas tier 2
    **{k: 2 for k in _FAMILIES if _FAMILIES[k] == "UNet"},
}


# ════════════════════════════════════════════════════════════════════════════
# SECAO: REGISTRY — MAPEAMENTO model_type → build_fn
# ════════════════════════════════════════════════════════════════════════════
# Lazy registry: as funcoes build_* sao importadas apenas quando chamadas.
# Evita importar TF no nivel de modulo.
# ──────────────────────────────────────────────────────────────────────────


def _get_build_fn(model_type: str) -> Callable:
    """Retorna a funcao build_* para o model_type informado.

    Imports lazy por familia para nao carregar TF no nivel de modulo.

    Args:
        model_type: Nome da arquitetura (44 opcoes).

    Returns:
        Callable: funcao build_*(config) → tf.keras.Model.

    Raises:
        ValueError: Se model_type nao esta no registry.
    """
    # ── CNN ───────────────────────────────────────────────────────────
    if model_type in (
        "ResNet_18",
        "ResNet_34",
        "ResNet_50",
        "ConvNeXt",
        "InceptionNet",
        "InceptionTime",
        "CNN_1D",
    ):
        from geosteering_ai.models.cnn import (
            build_cnn1d,
            build_convnext,
            build_inceptionnet,
            build_inceptiontime,
            build_resnet18,
            build_resnet34,
            build_resnet50,
        )

        _cnn = {
            "ResNet_18": build_resnet18,
            "ResNet_34": build_resnet34,
            "ResNet_50": build_resnet50,
            "ConvNeXt": build_convnext,
            "InceptionNet": build_inceptionnet,
            "InceptionTime": build_inceptiontime,
            "CNN_1D": build_cnn1d,
        }
        return _cnn[model_type]

    # ── TCN ───────────────────────────────────────────────────────────
    if model_type in ("TCN", "TCN_Advanced"):
        from geosteering_ai.models.tcn import build_tcn, build_tcn_advanced

        return {"TCN": build_tcn, "TCN_Advanced": build_tcn_advanced}[model_type]

    # ── RNN ───────────────────────────────────────────────────────────
    if model_type in ("LSTM", "BiLSTM"):
        from geosteering_ai.models.rnn import build_bilstm, build_lstm

        return {"LSTM": build_lstm, "BiLSTM": build_bilstm}[model_type]

    # ── Hybrid ────────────────────────────────────────────────────────
    if model_type in ("CNN_LSTM", "CNN_BiLSTM_ED"):
        from geosteering_ai.models.hybrid import build_cnn_bilstm_ed, build_cnn_lstm

        return {"CNN_LSTM": build_cnn_lstm, "CNN_BiLSTM_ED": build_cnn_bilstm_ed}[
            model_type
        ]

    # ── U-Net ─────────────────────────────────────────────────────────
    if model_type.startswith("UNet"):
        from geosteering_ai.models.unet import (
            build_unet_attention,
            build_unet_attention_convnext,
            build_unet_attention_efficientnet,
            build_unet_attention_inception,
            build_unet_attention_resnet18,
            build_unet_attention_resnet34,
            build_unet_attention_resnet50,
            build_unet_base,
            build_unet_convnext,
            build_unet_efficientnet,
            build_unet_inception,
            build_unet_resnet18,
            build_unet_resnet34,
            build_unet_resnet50,
        )

        _unet = {
            "UNet_Base": build_unet_base,
            "UNet_Attention": build_unet_attention,
            "UNet_ResNet18": build_unet_resnet18,
            "UNet_Attention_ResNet18": build_unet_attention_resnet18,
            "UNet_ResNet34": build_unet_resnet34,
            "UNet_Attention_ResNet34": build_unet_attention_resnet34,
            "UNet_ResNet50": build_unet_resnet50,
            "UNet_Attention_ResNet50": build_unet_attention_resnet50,
            "UNet_ConvNeXt": build_unet_convnext,
            "UNet_Attention_ConvNeXt": build_unet_attention_convnext,
            "UNet_Inception": build_unet_inception,
            "UNet_Attention_Inception": build_unet_attention_inception,
            "UNet_EfficientNet": build_unet_efficientnet,
            "UNet_Attention_EfficientNet": build_unet_attention_efficientnet,
        }
        if model_type in _unet:
            return _unet[model_type]

    # ── Transformer ───────────────────────────────────────────────────
    if model_type in (
        "Transformer",
        "Simple_TFT",
        "TFT",
        "PatchTST",
        "Autoformer",
        "iTransformer",
    ):
        from geosteering_ai.models.transformer import (
            build_autoformer,
            build_itransformer,
            build_patchtst,
            build_simple_tft,
            build_tft,
            build_transformer,
        )

        _tr = {
            "Transformer": build_transformer,
            "Simple_TFT": build_simple_tft,
            "TFT": build_tft,
            "PatchTST": build_patchtst,
            "Autoformer": build_autoformer,
            "iTransformer": build_itransformer,
        }
        return _tr[model_type]

    # ── Decomposition ─────────────────────────────────────────────────
    if model_type in ("N_BEATS", "N_HiTS"):
        from geosteering_ai.models.decomposition import build_nbeats, build_nhits

        return {"N_BEATS": build_nbeats, "N_HiTS": build_nhits}[model_type]

    # ── Advanced ──────────────────────────────────────────────────────
    if model_type in ("DNN", "FNO", "DeepONet", "Geophysical_Attention"):
        from geosteering_ai.models.advanced import (
            build_deeponet,
            build_dnn,
            build_fno,
            build_geophysical_attention,
        )

        _adv = {
            "DNN": build_dnn,
            "FNO": build_fno,
            "DeepONet": build_deeponet,
            "Geophysical_Attention": build_geophysical_attention,
        }
        return _adv[model_type]

    # ── Geosteering ───────────────────────────────────────────────────
    if model_type in (
        "WaveNet",
        "Causal_Transformer",
        "Informer",
        "Mamba_S4",
        "Encoder_Forecaster",
    ):
        from geosteering_ai.models.geosteering import (
            build_causal_transformer,
            build_encoder_forecaster,
            build_informer,
            build_mamba_s4,
            build_wavenet,
        )

        _gs = {
            "WaveNet": build_wavenet,
            "Causal_Transformer": build_causal_transformer,
            "Informer": build_informer,
            "Mamba_S4": build_mamba_s4,
            "Encoder_Forecaster": build_encoder_forecaster,
        }
        return _gs[model_type]

    raise ValueError(
        f"model_type '{model_type}' nao encontrado no registry. "
        f"Use list_available_models() para ver as 44 opcoes."
    )


# ════════════════════════════════════════════════════════════════════════════
# SECAO: API PUBLICA
# ════════════════════════════════════════════════════════════════════════════
# 4 funcoes publicas + ModelRegistry class.
# build_model(): factory principal (mais usada).
# get_model_info(): metadados por arquitetura.
# list_available_models(): catalogo completo.
# is_causal_compatible(): validacao de modo.
# ──────────────────────────────────────────────────────────────────────────


def build_model(config: "PipelineConfig") -> "tf.keras.Model":
    """Constroi o modelo selecionado em config.model_type.

    Factory central: ponto unico de construcao de modelos.
    Valida compatibilidade causal antes de construir.

    Args:
        config: PipelineConfig com model_type (str) e todos os
            hiperparametros necessarios.

    Returns:
        tf.keras.Model: Modelo Keras pronto para compile() e fit().
            Input shape: (batch, config.sequence_length, config.n_features)
            Output shape: (batch, config.sequence_length, config.output_channels)

    Raises:
        ValueError: Se model_type nao existe ou e incompativel com
            use_causal_mode.

    Example:
        >>> config = PipelineConfig.robusto()
        >>> model = build_model(config)
        >>> model.summary()

    Note:
        Referenciado em:
            - training/loop.py: TrainingLoop.run() — ponto de uso
            - tests/test_models.py: TestRegistry.test_build_model_default
        Valida: se use_causal_mode=True e model em _CAUSAL_INCOMPATIBLE
        → warning, mas ainda constroi (usuario pode querer testar).
        Ref: docs/ARCHITECTURE_v2.md secao 5.11.

        3 Modos de Injecao Estatica (theta/freq → modelo):

        .. code-block:: text

            ┌────────────────────────────────────────────────────────────────┐
            │  A. broadcast (default):                                       │
            │    [θ?, f?, z, EM..., GS...] → model → output                │
            │    n_prefix = 0-2 (colunas prepended ao tensor)               │
            │                                                                │
            │  B. dual_input:                                                │
            │    [z, EM..., GS...], [θ, f] → stem(broadcast+concat) → out  │
            │    n_prefix = 0 (theta/freq via input separado)               │
            │                                                                │
            │  C. film:                                                      │
            │    [z, EM..., GS...], [θ, f] → model → FiLM(γ×h+β) → out    │
            │    n_prefix = 0 (modulacao output-level por θ/f)              │
            └────────────────────────────────────────────────────────────────┘
    """
    model_type = config.model_type

    if model_type not in _FAMILIES:
        raise ValueError(
            f"model_type '{model_type}' invalido. " f"Validos: {list_available_models()}"
        )

    if config.use_causal_mode and not is_causal_compatible(model_type):
        logger.warning(
            "model_type='%s' e CAUSAL_INCOMPATIBLE mas use_causal_mode=True. "
            "Use inference_mode='offline' para esta arquitetura.",
            model_type,
        )

    logger.info(
        "build_model: model_type='%s', familia='%s', tier=%d, n_feat=%d, out_ch=%d",
        model_type,
        _FAMILIES.get(model_type, "?"),
        _TIERS.get(model_type, 0),
        config.n_features,
        config.output_channels,
    )

    # ── Construcao com wrapper para Abordagens A/B/C (P2/P3) ────────
    _n_static_vars = int(config.use_theta_as_feature) + int(config.use_freq_as_feature)
    _needs_wrapper = config.static_injection_mode != "broadcast" and _n_static_vars > 0

    if _needs_wrapper and config.static_injection_mode == "dual_input":
        # Abordagem B: core com shape expandida + stem dual-input
        import dataclasses

        _cfg_b = dataclasses.replace(config, static_injection_mode="broadcast")
        model = _wrap_dual_input(
            _get_build_fn(model_type), _cfg_b, config, _n_static_vars
        )
    elif _needs_wrapper and config.static_injection_mode == "film":
        # Abordagem C: core processa EM puro, FiLM modula output
        core_model = _get_build_fn(model_type)(config)
        model = _wrap_film(core_model, config, _n_static_vars)
    else:
        # Abordagem A (broadcast): modelo single-input padrao
        model = _get_build_fn(model_type)(config)

    logger.info(
        "Modelo '%s' construido: %d params treinaveis, injection='%s'",
        model_type,
        sum(int(v.numpy().size) for v in model.trainable_weights)
        if hasattr(model, "trainable_weights")
        else -1,
        config.static_injection_mode,
    )
    return model


def _wrap_dual_input(
    build_fn: Callable,
    cfg_broadcast: "PipelineConfig",
    config_original: "PipelineConfig",
    n_static: int,
) -> "tf.keras.Model":
    """Abordagem B: wrapper dual-input com StaticInjectionStem.

    Constroi core com shape expandida (n_em + n_static) e envolve em
    wrapper que aceita [em_input, static_input]. O stem faz broadcast +
    concat dos escalares com as features EM, produzindo tensor unico
    que o core processa normalmente. Todas as 44 arquiteturas compativeis.

    Args:
        build_fn: Funcao build_* para o model_type.
        cfg_broadcast: Config com mode="broadcast" (n_features expandido).
        config_original: Config original do usuario.
        n_static: Numero de variaveis estaticas (1 ou 2).

    Returns:
        tf.keras.Model: Dual-input [em_input, static_input] → output.

    Note:
        Referenciado em: build_model() (Abordagem B).
        core construido com cfg_broadcast.n_features = n_base + n_prefix + n_gs.
        Stem: broadcast escalares na GPU + concat com EM.
        Ref: docs/physics/perspectivas.md secoes P2, P3.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import static_injection_stem

    core = build_fn(cfg_broadcast)
    n_em = config_original.n_base_features + config_original.n_geosignal_channels
    seq_len = config_original.sequence_length

    em_input = tf.keras.Input(shape=(seq_len, n_em), name="em_features")
    static_input = tf.keras.Input(shape=(n_static,), name="static_params")
    combined = static_injection_stem(em_input, static_input)
    out = core(combined)

    return tf.keras.Model(
        inputs=[em_input, static_input],
        outputs=out,
        name=f"{core.name}_dual",
    )


def _wrap_film(
    core_model: "tf.keras.Model",
    config: "PipelineConfig",
    n_static: int,
) -> "tf.keras.Model":
    """Abordagem C: wrapper FiLM (output-level modulation).

    Core processa EM normalmente, depois theta/freq modulam os canais de
    saida via gamma*h+beta (Feature-wise Linear Modulation). Modulacao
    output-level: theta/f ajustam escala e bias das predicoes finais
    conforme condicoes de aquisicao (fisicamente interpretavel como
    calibracao da inversao para diferentes geometrias e frequencias).

    Args:
        core_model: Modelo single-input construido por build_fn.
        config: PipelineConfig com output_channels.
        n_static: Numero de variaveis estaticas (1 ou 2).

    Returns:
        tf.keras.Model: Dual-input [em_input, static_input] → output modulado.

    Note:
        Referenciado em: build_model() (Abordagem C).
        Incompativel com: N_BEATS, N_HiTS, FNO, DeepONet (validado em config).
        Ref: Perez et al. (2018) AAAI, adaptado para output-level modulation.
    """
    import tensorflow as tf

    from geosteering_ai.models.blocks import film_layer

    core_input_shape = core_model.input_shape
    n_em = core_input_shape[-1]
    seq_len = core_input_shape[-2]

    em_input = tf.keras.Input(shape=(seq_len, n_em), name="em_features")
    static_input = tf.keras.Input(shape=(n_static,), name="static_params")

    core_output = core_model(em_input)
    out = film_layer(core_output, static_input, n_channels=config.output_channels)

    return tf.keras.Model(
        inputs=[em_input, static_input],
        outputs=out,
        name=f"{core_model.name}_film",
    )


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Retorna metadados de uma arquitetura.

    Args:
        model_type: Nome da arquitetura (str).

    Returns:
        dict com:
            - name: str
            - family: str (CNN, TCN, RNN, ...)
            - tier: int (1=validado, 2=testado, 3=experimental)
            - causal_compatible: bool
            - description: str

    Raises:
        ValueError: Se model_type nao existe.

    Example:
        >>> info = get_model_info("ResNet_18")
        >>> info["tier"]
        1

    Note:
        Referenciado em:
            - training/loop.py: log de metadados pre-treinamento
            - tests/test_models.py: TestRegistry.test_get_model_info
    """
    if model_type not in _FAMILIES:
        raise ValueError(f"model_type '{model_type}' invalido.")

    _DESCRIPTIONS: Dict[str, str] = {
        "ResNet_18": "ResNet-18 1D seq2seq — default ★ (validado, estavel)",
        "ResNet_34": "ResNet-34 1D — mais profundo para datasets grandes",
        "ResNet_50": "ResNet-50 1D — bottleneck blocks, maior capacidade",
        "ConvNeXt": "ConvNeXt 1D Tiny — depthwise+LN+GELU moderno",
        "InceptionNet": "InceptionNet 1D — multi-escala (9,19,39 kernels)",
        "InceptionTime": "InceptionTime 1D — inception + residual (TSAI)",
        "CNN_1D": "CNN_1D baseline — 6 camadas simetricas [32,64,128,128,64,32]",
        "TCN": "TCN — dilation doubling causal nativo (Bai 2018)",
        "TCN_Advanced": "TCN_Advanced — multi-scale stacks + SE + atencao",
        "LSTM": "LSTM — recorrente causal nativo (3 camadas)",
        "BiLSTM": "BiLSTM — bidirecional CAUSAL_INCOMPATIBLE",
        "CNN_LSTM": "CNN_LSTM — CNN encoder + LSTM temporal",
        "CNN_BiLSTM_ED": "CNN_BiLSTM_ED — encoder-decoder CAUSAL_INCOMPATIBLE",
        "Transformer": "Transformer vanilla — pre-LN, pos. enc. aprendido",
        "Simple_TFT": "Simple_TFT — GRN + Transformer simplificado",
        "TFT": "TFT completo — VSN + GRN + gating (Lim 2021)",
        "PatchTST": "PatchTST — Transformer sobre patches temporais (Nie 2023)",
        "Autoformer": "Autoformer — decomp + autocorrelacao (Wu 2021)",
        "iTransformer": "iTransformer — atencao invertida sobre variaveis (Liu 2023)",
        "N_BEATS": "N-BEATS — basis expansion residual (Oreshkin 2020)",
        "N_HiTS": "N-HiTS — hierarquico multi-escala (Challu 2023)",
        "DNN": "DNN — MLP ponto a ponto via TimeDistributed (baseline)",
        "FNO": "FNO — Fourier Neural Operator espectral (Li 2021)",
        "DeepONet": "DeepONet — branch-trunk operator network (Lu 2021)",
        "Geophysical_Attention": "Geophysical_Attention — CNN+atencao fisica LWD",
        "WaveNet": "WaveNet — gated activation causal (Oord 2016)",
        "Causal_Transformer": "Causal_Transformer — Transformer mascara causal",
        "Informer": "Informer — sparse attention O(L log L) (Zhou 2021)",
        "Mamba_S4": "Mamba_S4 — state space model causal (Gu 2022)",
        "Encoder_Forecaster": "Encoder_Forecaster — LSTM enc + CNN dec causal",
    }
    # U-Nets
    for k in list(_FAMILIES.keys()):
        if k.startswith("UNet") and k not in _DESCRIPTIONS:
            attn = "Attn " if "Attention" in k else ""
            enc = k.replace("UNet_Attention_", "").replace("UNet_", "") or "Base"
            _DESCRIPTIONS[k] = f"UNet {attn}1D — encoder {enc} CAUSAL_INCOMPATIBLE"

    return {
        "name": model_type,
        "family": _FAMILIES.get(model_type, "Unknown"),
        "tier": _TIERS.get(model_type, 3),
        "causal_compatible": is_causal_compatible(model_type),
        "description": _DESCRIPTIONS.get(model_type, ""),
    }


def list_available_models(family: Optional[str] = None) -> List[str]:
    """Lista todas as 44 arquiteturas disponíveis.

    Args:
        family: Filtro por familia ('CNN', 'TCN', 'RNN', 'Hybrid',
            'UNet', 'Transformer', 'Decomposition', 'Advanced', 'Geosteering').
            None = todas.

    Returns:
        List[str]: Nomes das arquiteturas, ordenados por familia.

    Example:
        >>> list_available_models("CNN")
        ['ResNet_18', 'ResNet_34', 'ResNet_50', 'ConvNeXt', ...]
        >>> len(list_available_models())
        44

    Note:
        Referenciado em:
            - tests/test_models.py: TestRegistry.test_list_models_count
        Conta total: 7+2+2+2+14+6+2+4+5 = 44 arquiteturas.
    """
    if family is not None:
        return [k for k, v in _FAMILIES.items() if v == family]
    return list(_FAMILIES.keys())


def is_causal_compatible(model_type: str) -> bool:
    """Verifica se uma arquitetura e compativel com modo causal.

    Args:
        model_type: Nome da arquitetura.

    Returns:
        bool: True se compativel com inference_mode='realtime'.

    Example:
        >>> is_causal_compatible("ResNet_18")
        True
        >>> is_causal_compatible("BiLSTM")
        False

    Note:
        Referenciado em:
            - build_model(): validacao pre-construcao
            - tests/test_models.py: TestRegistry.test_causal_compat
        17 arquiteturas sao CAUSAL_INCOMPATIBLE (BiLSTM, todos UNets, N-BEATS, N-HiTS).
        27 arquiteturas sao causal-compatible.
    """
    return model_type not in _CAUSAL_INCOMPATIBLE


class ModelRegistry:
    """Fachada orientada a objeto do registry de arquiteturas.

    Wrapper sobre as funcoes do modulo para uso em contextos OO.

    Example:
        >>> registry = ModelRegistry()
        >>> model = registry.build(config)
        >>> info = registry.info("TCN")
        >>> models = registry.available(family="CNN")

    Note:
        Referenciado em:
            - training/loop.py: ModelRegistry().build(config)
            - tests/test_models.py: TestRegistry
        Todas as operacoes delegam para as funcoes do modulo.
        Ref: docs/ARCHITECTURE_v2.md secao 5.11.
    """

    def build(self, config: "PipelineConfig") -> "tf.keras.Model":
        """Constroi modelo via config.model_type. Delega a build_model()."""
        return build_model(config)

    def info(self, model_type: str) -> Dict[str, Any]:
        """Metadados de uma arquitetura. Delega a get_model_info()."""
        return get_model_info(model_type)

    def available(self, family: Optional[str] = None) -> List[str]:
        """Lista arquiteturas disponiveis. Delega a list_available_models()."""
        return list_available_models(family)

    def causal_compatible(self, model_type: str) -> bool:
        """Verifica compatibilidade causal. Delega a is_causal_compatible()."""
        return is_causal_compatible(model_type)

    @property
    def count(self) -> int:
        """Numero total de arquiteturas registradas (deve ser 44)."""
        return len(_FAMILIES)


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    "ModelRegistry",
    "build_model",
    "get_model_info",
    "list_available_models",
    "is_causal_compatible",
]
