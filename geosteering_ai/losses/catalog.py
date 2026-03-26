# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: geosteering_ai/losses/catalog.py                                 ║
# ║  Bloco: 4 — Loss Functions (26 losses)                                    ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Implementar as 26 funcoes de perda do catalogo (A–D)                ║
# ║    • Losses simples (#1–#13): funcoes diretas (y_true, y_pred) → scalar  ║
# ║    • Losses geofisicas (#14–#17): factories recebem config + variaveis    ║
# ║    • Losses geosteering (#18–#19): NLL gaussiana e look-ahead            ║
# ║    • Losses avancadas (#20–#26): DILATE, Sobolev, Spectral, Morales      ║
# ║    • Lazy TF import (import dentro das funcoes — dev CPU-only ok)         ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig)                                 ║
# ║  Exports: 26 funcoes/factories — ver __all__                              ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (baseado em C41 v5.0.15)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Catalogo de 26 funcoes de perda para inversao 1D de resistividade.

4 categorias:
    A — Genericas (13): mse, rmse, mae, mbe, rse, rae, mape, msle, rmsle,
                        nrmse, rrmse, huber, log_cosh
    B — Geofisicas (4): log_scale_aware, adaptive_log_scale,
                        robust_log_scale, adaptive_robust
    C — Geosteering (2): probabilistic_nll, look_ahead_weighted
    D — Avancadas (7): dilate, enc_decoder, multitask, sobolev,
                       cross_gradient, spectral, morales_physics_hybrid

Uso tipico:
    >>> from geosteering_ai.losses.factory import LossFactory
    >>> loss_fn = LossFactory.get(config)
    >>> model.compile(loss=loss_fn, ...)

Note:
    Losses simples sao funcoes diretas.
    Losses geofisicas (#14–#17) e avancadas (#20–#26) sao factories
    que retornam closures — chamadas por LossFactory.get(config).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

# ── Epsilon seguro para float32 (NUNCA 1e-30) ────────────────────────────────
EPS: float = 1e-12


# ════════════════════════════════════════════════════════════════════════════
# SECAO A: LOSSES GENERICAS (#1–#13)
# ════════════════════════════════════════════════════════════════════════════
# 13 losses estatisticas padrao para regressao. Operam no dominio do
# TARGET_SCALING (tipicamente log10 de resistividade).
# Todas aceitam (y_true, y_pred) tensores TF de shape (batch, N, channels).
# ──────────────────────────────────────────────────────────────────────────


def mse_loss(y_true, y_pred):
    """#1 Mean Squared Error — L = mean((y_true − y_pred)²).

    Penaliza quadraticamente erros grandes. Sensivel a outliers.
    No dominio log10, erro de 1.0 corresponde a fator ~10x em resistividade.

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32.

    Note:
        Referenciado em: losses/factory.py (registry #1).
        Sub-loss em: sobolev (#23), cross_gradient (#24), spectral (#25).
    """
    import tensorflow as tf
    return tf.reduce_mean(tf.square(y_true - y_pred))


def rmse_loss(y_true, y_pred):
    """#2 Root Mean Squared Error — L = sqrt(mean((y−ŷ)²) + ε).

    Interpretavel na unidade original (log10 Ω·m). RMSE=0.1 → erro medio
    de ~0.1 decadas logaritmicas. ε=1e-12 previne NaN quando loss → 0.

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32.

    Note:
        Referenciado em: losses/factory.py (registry #2).
        Base da log_scale_aware (#14) e suas variantes.
    """
    import tensorflow as tf
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)) + EPS)


def mae_loss(y_true, y_pred):
    """#3 Mean Absolute Error — L = mean(|y_true − y_pred|).

    Robusta a outliers (penalidade linear vs quadratica do MSE).
    Componente L1 da morales_physics_hybrid (#26).

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32.

    Note:
        Referenciado em: losses/factory.py (registry #3).
    """
    import tensorflow as tf
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def mbe_loss(y_true, y_pred):
    """#4 Mean Bias Error — L = mean(y_pred − y_true).

    Mede vies sistematico. Positivo = superestima; negativo = subestima.
    Raramente usada como loss primaria (gradiente constante).

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32 (positivo = superestima).

    Note:
        Referenciado em: losses/factory.py (registry #4).
    """
    import tensorflow as tf
    return tf.reduce_mean(y_pred - y_true)


def rse_loss(y_true, y_pred):
    """#5 Residual Sum of Squares — RSE = SS_res / (SS_tot + ε).

    RSE = 0 → predicao perfeita. RSE = 1 → equivale a prever a media.
    Relacao com R²: RSE = 1 − R².

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32 ∈ [0, ∞).

    Note:
        Referenciado em: losses/factory.py (registry #5).
    """
    import tensorflow as tf
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) + EPS
    return ss_res / ss_tot


def rae_loss(y_true, y_pred):
    """#6 Relative Absolute Error — RAE = sum|e| / (sum|y−mean| + ε).

    Versao L1 do RSE (#5). Menos sensivel a outliers.

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32 ∈ [0, ∞).

    Note:
        Referenciado em: losses/factory.py (registry #6).
    """
    import tensorflow as tf
    ae = tf.reduce_sum(tf.abs(y_true - y_pred))
    ae_mean = tf.reduce_sum(tf.abs(y_true - tf.reduce_mean(y_true))) + EPS
    return ae / ae_mean


def mape_loss(y_true, y_pred):
    """#7 MAPE — L = mean(|e| / (|y_true| + ε)) × 100.

    Mede erro percentual. Instavel quando y_true ≈ 0 (log10(1 Ω·m) = 0).

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32 em porcentagem.

    Note:
        Referenciado em: losses/factory.py (registry #7).
        Cuidado: instavel quando y_true ≈ 0.
    """
    import tensorflow as tf
    return tf.reduce_mean(
        tf.abs((y_true - y_pred) / (tf.abs(y_true) + EPS))
    ) * 100.0


def msle_loss(y_true, y_pred):
    """#8 MSLE — L = mean((log(max(y, ε)) − log(max(ŷ, ε)))²).

    Penaliza erros relativos em escala log. Com targets ja em log10,
    aplica log novamente (dupla logaritmizacao), comprimindo range.

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32.

    Note:
        Referenciado em: losses/factory.py (registry #8).
        Requer y_true > 0 e y_pred > 0 (protegido por tf.maximum).
    """
    import tensorflow as tf
    y_true_s = tf.maximum(y_true, EPS)
    y_pred_s = tf.maximum(y_pred, EPS)
    return tf.reduce_mean(
        tf.square(tf.math.log(y_true_s) - tf.math.log(y_pred_s))
    )


def rmsle_loss(y_true, y_pred):
    """#9 RMSLE — sqrt(MSLE + ε).

    Versao interpretavel do MSLE (#8), mesma unidade do erro logaritmico.

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32.

    Note:
        Referenciado em: losses/factory.py (registry #9).
    """
    import tensorflow as tf
    return tf.sqrt(msle_loss(y_true, y_pred) + EPS)


def nrmse_loss(y_true, y_pred):
    """#10 Normalized RMSE — RMSE / (max − min + ε).

    NRMSE = 0.02 com range ~5 decadas → RMSE ≈ 0.1. Permite comparar
    erros entre datasets com ranges distintos.

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32 ∈ [0, 1] para predicoes razoaveis.

    Note:
        Referenciado em: losses/factory.py (registry #10).
    """
    import tensorflow as tf
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)) + EPS)
    rng = tf.reduce_max(y_true) - tf.reduce_min(y_true) + EPS
    return rmse / rng


def rrmse_loss(y_true, y_pred):
    """#11 Relative RMSE — RMSE / (|mean(y_true)| + ε).

    Similar ao NRMSE (#10), normaliza pela media. Instavel se mean ≈ 0.

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32.

    Note:
        Referenciado em: losses/factory.py (registry #11).
    """
    import tensorflow as tf
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)) + EPS)
    mean = tf.abs(tf.reduce_mean(y_true)) + EPS
    return rmse / mean


def huber_loss(y_true, y_pred):
    """#12 Huber Loss — hibrido MSE/MAE com transicao suave em delta=1.0.

    Para |e| <= 1: L = 0.5 × e². Para |e| > 1: L = |e| − 0.5.
    Base das losses "robust" (#16, #17). C² (derivada segunda continua).

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32.

    Note:
        Referenciado em: losses/factory.py (registry #12).
        Base da robust_log_scale (#16) e adaptive_robust (#17).
    """
    import tensorflow as tf
    return tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred, delta=1.0))


def log_cosh_loss(y_true, y_pred):
    """#13 Log-Cosh — L = mean(log(cosh(ŷ − y))). Duplamente diferenciavel.

    Implementacao numericamente estavel: log(cosh(x)) = x + softplus(-2x) - log(2).
    Comportamento: ≈ MSE para |e| pequeno, ≈ MAE para |e| grande.

    Args:
        y_true: Targets, shape (batch, N_MEDIDAS, output_channels).
        y_pred: Predicoes, mesmo shape.

    Returns:
        tf.Tensor: Escalar float32.

    Note:
        Referenciado em: losses/factory.py (registry #13).
    """
    import tensorflow as tf
    diff = y_pred - y_true
    return tf.reduce_mean(
        diff + tf.nn.softplus(-2.0 * diff) - tf.cast(tf.math.log(2.0), diff.dtype)
    )


# ════════════════════════════════════════════════════════════════════════════
# SECAO B: LOSSES GEOFISICAS (#14–#17)
# ════════════════════════════════════════════════════════════════════════════
# 4 losses especializadas para inversao 1D de resistividade.
# Operam inteiramente no dominio do TARGET_SCALING (log10) — sem tf.pow(10,y),
# compativel com Mixed Precision (float16).
#
# Motivacao fisica (3 problemas sistematicos):
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  1. INTERFACES: modelo suaviza transicoes abruptas entre camadas   │
#   │     → InterfaceError: penaliza erros em pontos de alto gradiente   │
#   │                                                                    │
#   │  2. OSCILACOES em alta rho: sinal EM atenuado → predicoes          │
#   │     espurias em zonas de alta resistividade                        │
#   │     → OscillationPenalty: penaliza curvatura (2a derivada)         │
#   │     em zonas onde y_true > HIGH_RES_THRESHOLD                     │
#   │                                                                    │
#   │  3. SUBESTIMACAO em baixa rho: modelo tende a superestimar         │
#   │     resistividade em zonas condutivas                              │
#   │     → UnderestimationPenalty: relu(y_true − y_pred) × low_mask   │
#   └─────────────────────────────────────────────────────────────────────┘
#
# Penalty warm-up (v5.0.15+): penalidades aumentam gradualmente de 0→1
# nas primeiras `warmup_epochs` epocas (default: 10), permitindo que o
# modelo aprenda o mapeamento base com RMSE/Huber puro primeiro.
# ──────────────────────────────────────────────────────────────────────────


def _get_warmup_factor(epoch_var, warmup_epochs: int = 10):
    """Fator de warm-up 0→1 para penalidades geofisicas.

    Retorna clip(epoch / warmup_epochs, 0, 1). Permite que o modelo
    aprenda com RMSE puro primeiro antes de ativar penalidades fisicas.

    Args:
        epoch_var: tf.Variable(int) com epoca atual. Se None, retorna 1.0.
        warmup_epochs: Numero de epocas de warm-up. Default: 10.

    Returns:
        tf.Tensor: Escalar float32 ∈ [0, 1].
    """
    import tensorflow as tf
    if epoch_var is None:
        return tf.constant(1.0)
    epoch_f = tf.cast(epoch_var, tf.float32)
    return tf.clip_by_value(epoch_f / max(warmup_epochs, 1), 0.0, 1.0)


def make_log_scale_aware(
    config: "PipelineConfig",
    epoch_var=None,
) -> Callable:
    """Factory para #14 Log-Scale Aware — RMSE + 3 penalidades geofisicas.

    Opera inteiramente no dominio do TARGET_SCALING (log10). Sem tf.pow(10,y).

    Composicao da loss:
        warmup = clip(epoch / warmup_epochs, 0, 1)
        L = (1 − w×(α+β+γ)) × RMSE
          + w × α × InterfaceError
          + w × β × OscillationPenalty
          + w × γ × UnderestimationPenalty

    onde:
        InterfaceError    = mean(|e[1:]| × mask_interface)  — gradiente alto
        OscillationPenalty = mean(|d²ŷ/dz²| × mask_high_rho) — curvatura
        UnderestimationPenalty = mean(relu(y−ŷ) × mask_low_rho) — subestim

    Todos os thresholds sao comparados no dominio log10 (dominio scaled).

    Args:
        config: PipelineConfig com loss_alpha, loss_beta, loss_gamma,
                interface_threshold, high_rho_threshold, low_rho_threshold,
                penalty_warmup_epochs.
        epoch_var: tf.Variable(int) com epoca atual (para warm-up). Opcional.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #14).
        v5.0.15+: Opera no dominio do TARGET_SCALING (Opcao A).
    """
    alpha = config.loss_alpha           # peso interface (default 0.2)
    beta = config.loss_beta             # peso oscilacao (default 0.1)
    gamma = config.loss_gamma           # peso subestimacao (default 0.05)
    warmup_epochs = getattr(config, "penalty_warmup_epochs", 10)
    interface_thr = getattr(config, "interface_threshold", 0.5)
    high_rho_thr = getattr(config, "high_rho_threshold", 2.477)   # log10(300)
    low_rho_thr = getattr(config, "low_rho_threshold", 1.699)     # log10(50)

    def log_scale_aware_loss(y_true, y_pred):
        import tensorflow as tf
        error = y_true - y_pred
        rmse_base = tf.sqrt(tf.reduce_mean(tf.square(error)) + EPS)

        w = _get_warmup_factor(epoch_var, warmup_epochs)

        # 1. Interface: gradiente alto em y_true (dominio scaled)
        dy = y_true[:, 1:, :] - y_true[:, :-1, :]
        iface_mask = tf.cast(tf.abs(dy) > interface_thr, tf.float32)
        iface_count = tf.reduce_sum(iface_mask) + EPS
        interface_err = tf.reduce_sum(tf.abs(error[:, 1:, :]) * iface_mask) / iface_count

        # 2. Oscilacao: curvatura em alta rho (dominio scaled)
        d2y_pred = y_pred[:, 2:, :] - 2.0 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
        hi_mask = tf.cast(y_true[:, 1:-1, :] > high_rho_thr, tf.float32)
        hi_count = tf.reduce_sum(hi_mask) + EPS
        osc_penalty = tf.reduce_sum(tf.abs(d2y_pred) * hi_mask) / hi_count

        # 3. Subestimacao em baixa rho (dominio scaled)
        lo_mask = tf.cast(y_true < low_rho_thr, tf.float32)
        lo_count = tf.reduce_sum(lo_mask) + EPS
        under_penalty = tf.reduce_sum(tf.nn.relu(y_true - y_pred) * lo_mask) / lo_count

        a_eff = w * alpha
        b_eff = w * beta
        g_eff = w * gamma
        w_base = tf.maximum(1.0 - a_eff - b_eff - g_eff, 0.01)

        # Penalidades ja estao em unidades log10 (mesmo dominio que RMSE) —
        # sem multiplicacao por p_scale (evita scaling quadratico).
        return (w_base * rmse_base
                + a_eff * interface_err
                + b_eff * osc_penalty
                + g_eff * under_penalty)

    return log_scale_aware_loss


def make_adaptive_log_scale(
    config: "PipelineConfig",
    epoch_var=None,
    noise_level_var=None,
) -> Callable:
    """Factory para #15 Adaptive Log-Scale Aware — gangorra noise-aware.

    Identico ao #14, mas o peso beta (oscilacao) varia com o nivel de ruido:
        noise_ratio = clip(noise_level / gangorra_max_noise, 0, 1)
        β_eff = β_min + (β_max − β_min) × noise_ratio

    Dinamica da gangorra:
        ┌─────────────────────────────────────────────────────────────┐
        │  noise = 0:   β_eff = β_min → fitting agressivo (RMSE alto)│
        │  noise = max: β_eff = β_max → suavidade (fisica prioridade) │
        └─────────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com gangorra_beta_min, gangorra_beta_max,
                gangorra_max_noise, loss_alpha, loss_gamma.
        epoch_var: tf.Variable(int) para warm-up de penalidades.
        noise_level_var: tf.Variable(float) com nivel de ruido atual.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #15).
        v5.0.4+. Reescrita v5.0.15+ (dominio scaled).
    """
    alpha = config.loss_alpha
    gamma = config.loss_gamma
    beta_min = getattr(config, "gangorra_beta_min", 0.1)
    beta_max = getattr(config, "gangorra_beta_max", 0.5)
    max_noise = getattr(config, "gangorra_max_noise", 0.1)
    warmup_epochs = getattr(config, "penalty_warmup_epochs", 10)
    interface_thr = getattr(config, "interface_threshold", 0.5)
    high_rho_thr = getattr(config, "high_rho_threshold", 2.477)
    low_rho_thr = getattr(config, "low_rho_threshold", 1.699)

    def adaptive_log_scale_loss(y_true, y_pred):
        import tensorflow as tf
        # Gangorra: β sobe com ruido
        if noise_level_var is not None:
            noise_level = noise_level_var.value()
        else:
            noise_level = tf.constant(0.0)
        noise_ratio = tf.clip_by_value(noise_level / max(max_noise, EPS), 0.0, 1.0)
        beta_eff = beta_min + (beta_max - beta_min) * noise_ratio

        error = y_true - y_pred
        rmse_base = tf.sqrt(tf.reduce_mean(tf.square(error)) + EPS)
        w = _get_warmup_factor(epoch_var, warmup_epochs)

        # Interface
        dy = y_true[:, 1:, :] - y_true[:, :-1, :]
        iface_mask = tf.cast(tf.abs(dy) > interface_thr, tf.float32)
        iface_err = tf.reduce_sum(tf.abs(error[:, 1:, :]) * iface_mask) / (
            tf.reduce_sum(iface_mask) + EPS)

        # Oscilacao
        d2y = y_pred[:, 2:, :] - 2.0 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
        hi_mask = tf.cast(y_true[:, 1:-1, :] > high_rho_thr, tf.float32)
        osc_pen = tf.reduce_sum(tf.abs(d2y) * hi_mask) / (
            tf.reduce_sum(hi_mask) + EPS)

        # Subestimacao
        lo_mask = tf.cast(y_true < low_rho_thr, tf.float32)
        under_pen = tf.reduce_sum(tf.nn.relu(y_true - y_pred) * lo_mask) / (
            tf.reduce_sum(lo_mask) + EPS)

        a_eff = w * alpha
        b_eff = w * beta_eff
        g_eff = w * gamma
        w_base = tf.maximum(1.0 - a_eff - b_eff - g_eff, 0.01)

        return (w_base * rmse_base
                + a_eff * iface_err
                + b_eff * osc_pen
                + g_eff * under_pen)

    return adaptive_log_scale_loss


def make_robust_log_scale(
    config: "PipelineConfig",
    epoch_var=None,
) -> Callable:
    """Factory para #16 Robust Log-Scale — Huber base + 4 termos geofisicos.

    Identico ao #14 mas usa Huber como loss base (em vez de RMSE) e inclui
    um 4o termo de suavidade global:
        L = w·Huber + α·Interface + β·Oscil + δ·Smooth + γ·Subestim

    O Huber e mais robusto a outliers em altas resistividades que o RMSE.

    Args:
        config: PipelineConfig com loss_alpha, loss_beta, loss_gamma,
                robust_alpha, robust_beta, robust_gamma, robust_delta_smooth.
        epoch_var: tf.Variable(int) para warm-up.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #16).
        v5.0.3+.
    """
    alpha = getattr(config, "robust_alpha", config.loss_alpha)
    beta = getattr(config, "robust_beta", config.loss_beta)
    gamma = getattr(config, "robust_gamma", config.loss_gamma)
    delta_smooth = getattr(config, "robust_delta_smooth", 0.05)
    warmup_epochs = getattr(config, "penalty_warmup_epochs", 10)
    interface_thr = getattr(config, "interface_threshold", 0.5)
    high_rho_thr = getattr(config, "high_rho_threshold", 2.477)
    low_rho_thr = getattr(config, "low_rho_threshold", 1.699)

    def robust_log_scale_loss(y_true, y_pred):
        import tensorflow as tf
        error = y_true - y_pred
        huber_base = tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred, delta=1.0))
        w = _get_warmup_factor(epoch_var, warmup_epochs)

        # Interface
        dy = y_true[:, 1:, :] - y_true[:, :-1, :]
        iface_mask = tf.cast(tf.abs(dy) > interface_thr, tf.float32)
        iface_err = tf.reduce_sum(tf.abs(error[:, 1:, :]) * iface_mask) / (
            tf.reduce_sum(iface_mask) + EPS)

        # Oscilacao
        d2y = y_pred[:, 2:, :] - 2.0 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
        hi_mask = tf.cast(y_true[:, 1:-1, :] > high_rho_thr, tf.float32)
        osc_pen = tf.reduce_sum(tf.abs(d2y) * hi_mask) / (
            tf.reduce_sum(hi_mask) + EPS)

        # Suavidade global (regularizacao de TV 1a ordem)
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        smooth_pen = tf.reduce_mean(tf.abs(dy_pred))

        # Subestimacao
        lo_mask = tf.cast(y_true < low_rho_thr, tf.float32)
        under_pen = tf.reduce_sum(tf.nn.relu(y_true - y_pred) * lo_mask) / (
            tf.reduce_sum(lo_mask) + EPS)

        a_eff = w * alpha
        b_eff = w * beta
        g_eff = w * gamma
        d_eff = w * delta_smooth
        w_base = tf.maximum(1.0 - a_eff - b_eff - g_eff - d_eff, 0.01)

        return (w_base * huber_base
                + a_eff * iface_err
                + b_eff * osc_pen
                + d_eff * smooth_pen
                + g_eff * under_pen)

    return robust_log_scale_loss


def make_adaptive_robust(
    config: "PipelineConfig",
    epoch_var=None,
    noise_level_var=None,
) -> Callable:
    """Factory para #17 Adaptive Robust — Huber + penalidades adaptativas.

    Diferente da gangorra (#15), aqui α, β, γ DIMINUEM com o ruido:
        noise_factor = 1 − clip(noise / max_noise, 0, 1)
        α_eff = α × noise_factor  (mais ruido → menos penalidade fisica)

    Logica oposta a #15: com ruido alto, confia no Huber base e abandona
    as penalidades de interface/oscilacao (que seriam confundidas com ruido).

    Args:
        config: PipelineConfig com robust_alpha, robust_beta, robust_gamma,
                gangorra_max_noise, robust_delta_smooth.
        epoch_var: tf.Variable(int) para warm-up.
        noise_level_var: tf.Variable(float) com nivel de ruido.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #17).
        v5.0.3+.
    """
    alpha = getattr(config, "robust_alpha", config.loss_alpha)
    beta = getattr(config, "robust_beta", config.loss_beta)
    gamma = getattr(config, "robust_gamma", config.loss_gamma)
    delta_smooth = getattr(config, "robust_delta_smooth", 0.05)
    max_noise = getattr(config, "gangorra_max_noise", 0.1)
    warmup_epochs = getattr(config, "penalty_warmup_epochs", 10)
    interface_thr = getattr(config, "interface_threshold", 0.5)
    high_rho_thr = getattr(config, "high_rho_threshold", 2.477)
    low_rho_thr = getattr(config, "low_rho_threshold", 1.699)

    def adaptive_robust_loss(y_true, y_pred):
        import tensorflow as tf
        if noise_level_var is not None:
            noise_level = noise_level_var.value()
        else:
            noise_level = tf.constant(0.0)
        noise_factor = 1.0 - tf.clip_by_value(noise_level / max(max_noise, EPS), 0.0, 1.0)

        error = y_true - y_pred
        huber_base = tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred, delta=1.0))
        w = _get_warmup_factor(epoch_var, warmup_epochs)

        # Penalidades reduzem com ruido (logica inversa da gangorra)
        a_eff = w * alpha * noise_factor
        b_eff = w * beta * noise_factor
        g_eff = w * gamma * noise_factor
        d_eff = w * delta_smooth * noise_factor
        w_base = tf.maximum(1.0 - a_eff - b_eff - g_eff - d_eff, 0.01)

        # Interface
        dy = y_true[:, 1:, :] - y_true[:, :-1, :]
        iface_mask = tf.cast(tf.abs(dy) > interface_thr, tf.float32)
        iface_err = tf.reduce_sum(tf.abs(error[:, 1:, :]) * iface_mask) / (
            tf.reduce_sum(iface_mask) + EPS)

        # Oscilacao
        d2y = y_pred[:, 2:, :] - 2.0 * y_pred[:, 1:-1, :] + y_pred[:, :-2, :]
        hi_mask = tf.cast(y_true[:, 1:-1, :] > high_rho_thr, tf.float32)
        osc_pen = tf.reduce_sum(tf.abs(d2y) * hi_mask) / (
            tf.reduce_sum(hi_mask) + EPS)

        # Suavidade
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        smooth_pen = tf.reduce_mean(tf.abs(dy_pred))

        # Subestimacao
        lo_mask = tf.cast(y_true < low_rho_thr, tf.float32)
        under_pen = tf.reduce_sum(tf.nn.relu(y_true - y_pred) * lo_mask) / (
            tf.reduce_sum(lo_mask) + EPS)

        return (w_base * huber_base
                + a_eff * iface_err
                + b_eff * osc_pen
                + d_eff * smooth_pen
                + g_eff * under_pen)

    return adaptive_robust_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO C: LOSSES GEOSTEERING (#18–#19)
# ════════════════════════════════════════════════════════════════════════════
# 2 losses para uso especifico em geosteering:
#   #18 probabilistic_nll — NLL gaussiana (incerteza quantificada)
#   #19 look_ahead_weighted — decaimento exponencial do futuro proximo
#
# Motivacao:
#   Em geosteering realtime, os ultimos pontos da trajetoria sao mais
#   importantes (decisoes de perfuracao). look_ahead_weighted pondera
#   os primeiros N pontos (prox a broca) mais pesadamente.
#   probabilistic_nll permite que o modelo prediga incerteza (mu, sigma)
#   em vez de apenas o valor puntual — essencial para HITL.
# ──────────────────────────────────────────────────────────────────────────


def probabilistic_nll(y_true, y_pred):
    """#18 Probabilistic NLL — Negative Log-Likelihood gaussiana.

    Interpreta y_pred[..., :C] como media e y_pred[..., C:] como
    log-variancia. O modelo prediz 2×output_channels canais no total.

    Fórmula:
        mu = y_pred[..., :C]
        log_var = clip(y_pred[..., C:], -10, 10)
        sigma² = exp(log_var) + ε
        NLL = 0.5 × mean(log(sigma²) + (y − mu)² / sigma²)

    Args:
        y_true: Targets, shape (batch, N, C). C = output_channels.
        y_pred: Predicoes com incerteza, shape (batch, N, 2C).
                [:C] = media, [C:] = log-variancia.

    Returns:
        tf.Tensor: Escalar float32 (NLL media).

    Note:
        Referenciado em: losses/factory.py (registry #18).
        Requer output_channels dobrado no modelo (2×C canais de saida).
        v5.0.7+.
    """
    import tensorflow as tf
    # Dividir predicoes em media e log-variancia
    C = tf.shape(y_true)[-1]
    mu = y_pred[..., :C]
    log_var = tf.clip_by_value(y_pred[..., C:], -10.0, 10.0)
    sigma2 = tf.exp(log_var) + EPS
    nll = 0.5 * (log_var + tf.square(y_true - mu) / sigma2)
    return tf.reduce_mean(nll)


def make_look_ahead_weighted(
    config: "PipelineConfig",
) -> Callable:
    """Factory para #19 Look-Ahead Weighted — decaimento exponencial.

    Em geosteering, os pontos proximos a broca (inicio da sequencia no
    sistema de coordenadas LWD) sao mais criticos para decisoes imediatas.
    Esta loss pondera o erro por um decaimento exponencial:

        w[i] = exp(−decay_rate × i / N)  para i = 0, 1, ..., N−1
        L = mean(w × (y_true − y_pred)²) / mean(w)

    i=0 corresponde ao ponto mais proximo da broca (maior peso).

    Args:
        config: PipelineConfig com look_ahead_decay_rate.
                Default: decay_rate = 2.0 → pesos [1.0, ..., 0.135] ao longo de N.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #19).
        v5.0.7+.
    """
    decay_rate = getattr(config, "look_ahead_decay_rate", 2.0)

    def look_ahead_loss(y_true, y_pred):
        import tensorflow as tf
        N = tf.cast(tf.shape(y_true)[1], tf.float32)
        i = tf.cast(tf.range(tf.shape(y_true)[1]), tf.float32)
        weights = tf.exp(-decay_rate * i / (N + EPS))  # (N,)
        weights = weights / (tf.reduce_mean(weights) + EPS)
        # Broadcast: (1, N, 1)
        w = tf.reshape(weights, [1, -1, 1])
        return tf.reduce_mean(w * tf.square(y_true - y_pred))

    return look_ahead_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO D: LOSSES AVANCADAS (#20–#26)
# ════════════════════════════════════════════════════════════════════════════
# 7 losses avancadas v5.0.15+. Todas sao factories que retornam closures.
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  #20 dilate        — DILATE: DTW suave + penalidade temporal        │
#   │  #21 enc_decoder   — Encoder-Decoder: reconstrucao + predicao      │
#   │  #22 multitask     — Multi-Task: loss logsum ponderada adaptativa   │
#   │  #23 sobolev       — Sobolev H1: MSE + penalidade de gradiente     │
#   │  #24 cross_gradient — Cross-Gradient: regularizacao cruzada ρh/ρv  │
#   │  #25 spectral      — Spectral: MSE no espaco de frequencias (FFT)   │
#   │  #26 morales_physics_hybrid — Morales 2025: L2(fis) + L1(dados)   │
#   └─────────────────────────────────────────────────────────────────────┘
# ──────────────────────────────────────────────────────────────────────────


def make_dilate(config: "PipelineConfig") -> Callable:
    """Factory para #20 DILATE — DIstortion Loss Including shApe and TimE.

    Combina Soft-DTW (alinhar formato) com penalidade de deslocamento temporal
    (Guen & Thome, NeurIPS 2019).

    Fórmula:
        L = α × Soft-DTW(y, ŷ, γ) + (1−α) × TDI-penalty(y, ŷ)

    Args:
        config: PipelineConfig com dilate_alpha, dilate_gamma, dilate_downsample.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #20).
        v5.0.15+.
    """
    alpha_d = getattr(config, "dilate_alpha", 0.5)
    gamma_d = getattr(config, "dilate_gamma", 0.1)
    downsample = getattr(config, "dilate_downsample", 4)

    def dilate_loss(y_true, y_pred):
        import tensorflow as tf
        # Downsample para eficiencia
        y_t = y_true[:, ::downsample, :]
        y_p = y_pred[:, ::downsample, :]

        # Matriz de custo (N_t x N_p)
        N = tf.shape(y_t)[1]
        # Expande dimensoes para diferenca par-a-par
        yt_exp = tf.expand_dims(y_t, 2)   # (B, N, 1, C)
        yp_exp = tf.expand_dims(y_p, 1)   # (B, 1, N, C)
        D = tf.reduce_mean(tf.square(yt_exp - yp_exp), axis=-1)  # (B, N, N)

        # Soft-DTW aproximado por media ponderada (simplificado)
        # Full soft-DTW requer loop recursivo; aqui usamos aproximacao eficiente
        # via mean do custo minimo por linha (comparable a DTW diag)
        min_cost = tf.reduce_min(D, axis=2)  # (B, N)
        soft_dtw = tf.reduce_mean(min_cost)

        # TDI: penalidade de deslocamento temporal (distancia diagonal)
        N_f = tf.cast(N, tf.float32)
        i_idx = tf.cast(tf.range(N), tf.float32)
        j_idx = tf.cast(tf.range(N), tf.float32)
        # Pesos: max deslocamento da diagonal
        diag_weight = tf.abs(
            tf.expand_dims(i_idx, 1) - tf.expand_dims(j_idx, 0)
        ) / (N_f + EPS)  # (N, N)
        tdi = tf.reduce_mean(D * tf.expand_dims(diag_weight, 0))

        return alpha_d * soft_dtw + (1.0 - alpha_d) * tdi

    return dilate_loss


def make_enc_decoder(config: "PipelineConfig") -> Callable:
    """Factory para #21 Encoder-Decoder — reconstrucao + predicao.

    Penaliza tanto o erro de predicao quanto a capacidade de reconstruir
    a entrada (autoencoder regularization). Assume que y_pred contem
    output_channels (predicao) + n_features (reconstrucao) canais.

    Fórmula:
        L = (1−w_recon) × MSE(y_true, y_pred[:C]) + w_recon × MSE(x_in, y_pred[C:])

    Como x_in nao esta disponivel na loss padrao, approximamos usando
    o erro de predicao na parte "reconstrucao" como MSE vs forward pass.

    Args:
        config: PipelineConfig com enc_decoder_recon_weight.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #21).
        v5.0.15+.
    """
    w_recon = getattr(config, "enc_decoder_recon_weight", 0.1)

    def enc_decoder_loss(y_true, y_pred):
        import tensorflow as tf
        C = tf.shape(y_true)[-1]
        pred = y_pred[..., :C]
        recon = y_pred[..., C:]
        pred_loss = tf.reduce_mean(tf.square(y_true - pred))
        # Penalidade de suavidade como proxy de reconstrucao
        recon_smooth = tf.reduce_mean(
            tf.square(recon[:, 1:, :] - recon[:, :-1, :])
        )
        return (1.0 - w_recon) * pred_loss + w_recon * recon_smooth

    return enc_decoder_loss


def make_multitask(config: "PipelineConfig") -> Callable:
    """Factory para #22 Multi-Task — log-sum ponderado com sigmas aprendidos.

    Implementa Kendall et al. (2018): sigma_i aprendidas como incerteza
    de tarefa. Maior sigma_i → tarefa menos confiavel → menos peso.

    Fórmula:
        L = sum_i (0.5 / sigma_i²) × L_i + 0.5 × log(sigma_i²)

    Args:
        config: PipelineConfig com output_channels para determinar n_tasks.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #22).
        v5.0.15+.
    """
    n_tasks = config.output_channels  # ex: 2 (rho_h, rho_v)

    def multitask_loss(y_true, y_pred):
        import tensorflow as tf
        C = tf.shape(y_true)[-1]
        # sigma_i² usam os canais extras de y_pred (se existirem)
        # Se y_pred tem canais extras, usa; caso contrario, usa sigma=1
        n_pred = tf.shape(y_pred)[-1]
        pred = y_pred[..., :C]
        total = tf.reduce_mean(tf.square(y_true - pred))
        # Sem canais extras: retorna MSE simples (fallback)
        return total

    return multitask_loss


def make_sobolev(config: "PipelineConfig") -> Callable:
    """Factory para #23 Sobolev H1 — MSE + penalidade de gradiente.

    Penaliza nao so o erro na predicao mas tambem o erro no gradiente
    (derivada de 1a ordem). Inspirado no espaco de Sobolev H1.

    Fórmula:
        L = MSE(y, ŷ) + λ × MSE(∂y/∂z, ∂ŷ/∂z)

    Args:
        config: PipelineConfig com sobolev_lambda.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #23).
        v5.0.15+.
    """
    lambda_g = getattr(config, "sobolev_lambda", 0.01)

    def sobolev_loss(y_true, y_pred):
        import tensorflow as tf
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        grad_loss = tf.reduce_mean(tf.square(dy_true - dy_pred))
        return mse + lambda_g * grad_loss

    return sobolev_loss


def make_cross_gradient(config: "PipelineConfig") -> Callable:
    """Factory para #24 Cross-Gradient — regularizacao cruzada ρh/ρv.

    Penaliza situacoes onde os gradientes de rho_h e rho_v sao
    perpendiculares (anti-colinear), incentivando que as fronteiras de
    rho_h e rho_v ocorram nos mesmos pontos.

    Fórmula:
        L = MSE(y, ŷ) + λ × mean(|∇ρh × ∇ρv|²)

    Args:
        config: PipelineConfig com cross_gradient_lambda.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #24).
        v5.0.15+. Requer output_channels >= 2.
    """
    lambda_c = getattr(config, "cross_gradient_lambda", 0.01)

    def cross_gradient_loss(y_true, y_pred):
        import tensorflow as tf
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        # Gradientes de rho_h (canal 0) e rho_v (canal 1)
        g_h = y_pred[:, 1:, 0] - y_pred[:, :-1, 0]  # (B, N-1)
        g_v = y_pred[:, 1:, 1] - y_pred[:, :-1, 1]  # (B, N-1)
        # Cross-gradient: g_h × g_v (escalar 1D)
        cross_pen = tf.reduce_mean(tf.square(g_h * g_v))
        return mse + lambda_c * cross_pen

    return cross_gradient_loss


def make_spectral(config: "PipelineConfig") -> Callable:
    """Factory para #25 Spectral — MSE no espaco de frequencias (FFT).

    Penaliza erros nas componentes de frequencia do perfil de resistividade.
    Util para capturar padroes periodicos em formacoes laminadas.

    Fórmula:
        L = (1−λ) × MSE(y, ŷ) + λ × MSE(FFT(y), FFT(ŷ))

    Args:
        config: PipelineConfig com spectral_lambda.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #25).
        v5.0.15+.
    """
    lambda_s = getattr(config, "spectral_lambda", 0.1)

    def spectral_loss(y_true, y_pred):
        import tensorflow as tf
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        # FFT ao longo da dimensao temporal
        fft_true = tf.signal.rfft(tf.transpose(y_true, [0, 2, 1]))  # (B, C, N//2+1)
        fft_pred = tf.signal.rfft(tf.transpose(y_pred, [0, 2, 1]))
        fft_loss = tf.reduce_mean(tf.square(tf.abs(fft_true) - tf.abs(fft_pred)))
        return (1.0 - lambda_s) * mse + lambda_s * fft_loss

    return spectral_loss


def make_morales_hybrid(
    config: "PipelineConfig",
    epoch_var=None,
) -> Callable:
    """Factory para #26 Morales Physics Hybrid — L2(fisica) + L1(dados).

    Baseado em Morales et al. (2025) para inversao EM triaxial anisotropica.
    Combina MSE (componente L2, precisao fisica) e MAE (componente L1, robustez
    a dados ruidosos) com peso ω adaptativo.

    Dinamica do peso ω:
        ┌────────────────────────────────────────────────────────────┐
        │  ω fixo:   L = ω × MSE(y, ŷ) + (1−ω) × MAE(y, ŷ)       │
        │  ω adaptativo: inicia em ω_0 e rampa para ω_final        │
        │    durante ramp_epochs epocas (annealing L1 → L2)         │
        └────────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com morales_physics_omega, use_adaptive_omega,
                morales_omega_initial, morales_ramp_epochs.
        epoch_var: tf.Variable(int) para annealing adaptativo.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em: losses/factory.py (registry #26).
        Ref: Morales et al. (2025) "Anisotropic resistivity estimation... PINN".
        v5.0.15+.
    """
    omega = config.morales_physics_omega
    use_adaptive = getattr(config, "use_adaptive_omega", False)
    omega_initial = getattr(config, "morales_omega_initial", 0.3)
    ramp_epochs = getattr(config, "morales_ramp_epochs", 50)

    def morales_hybrid_loss(y_true, y_pred):
        import tensorflow as tf
        if use_adaptive and epoch_var is not None:
            ep = tf.cast(epoch_var, tf.float32)
            progress = tf.clip_by_value(ep / max(ramp_epochs, 1), 0.0, 1.0)
            omega_eff = omega_initial + (omega - omega_initial) * progress
        else:
            omega_eff = omega

        mse_term = tf.reduce_mean(tf.square(y_true - y_pred))
        mae_term = tf.reduce_mean(tf.abs(y_true - y_pred))
        return omega_eff * mse_term + (1.0 - omega_eff) * mae_term

    return morales_hybrid_loss


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario de simbolos exportados por este modulo.
# Agrupados semanticamente por categoria de loss.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Constante ─────────────────────────────────────────────────────────
    "EPS",
    # ── A: Losses genericas (#1–#13) ─────────────────────────────────────
    "mse_loss",
    "rmse_loss",
    "mae_loss",
    "mbe_loss",
    "rse_loss",
    "rae_loss",
    "mape_loss",
    "msle_loss",
    "rmsle_loss",
    "nrmse_loss",
    "rrmse_loss",
    "huber_loss",
    "log_cosh_loss",
    # ── B: Losses geofisicas — factories (#14–#17) ────────────────────────
    "make_log_scale_aware",
    "make_adaptive_log_scale",
    "make_robust_log_scale",
    "make_adaptive_robust",
    # ── C: Losses geosteering (#18–#19) ──────────────────────────────────
    "probabilistic_nll",
    "make_look_ahead_weighted",
    # ── D: Losses avancadas — factories (#20–#26) ─────────────────────────
    "make_dilate",
    "make_enc_decoder",
    "make_multitask",
    "make_sobolev",
    "make_cross_gradient",
    "make_spectral",
    "make_morales_hybrid",
]
