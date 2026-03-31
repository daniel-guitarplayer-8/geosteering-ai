# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: geosteering_ai/losses/factory.py                                 ║
# ║  Bloco: 4 — Loss Factory (LossFactory + build_combined)                   ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • LossFactory.get(config) — retorna loss base compilada                ║
# ║    • LossFactory.build_combined() — combina base + look_ahead + DTB + PINNs║
# ║    • _LOSS_REGISTRY — mapa de 26 nomes → funcoes/factories               ║
# ║    • VALID_LOSS_TYPES — lista de 26 nomes validos                        ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), losses/catalog.py              ║
# ║  Exports: LossFactory, build_loss_fn, VALID_LOSS_TYPES — ver __all__     ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 6                                    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Factory de loss functions para o pipeline de inversao.

Fluxo de uso:
    >>> from geosteering_ai.losses.factory import LossFactory
    >>> loss_fn = LossFactory.get(config)               # loss base
    >>> loss_fn = LossFactory.build_combined(config)    # combinada

Fluxo de decisao para build_combined:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  use_morales_hybrid_loss = True?                                    │
    │    SIM → loss_fn = morales_physics_hybrid (#26)                     │
    │    NAO ↓                                                            │
    │  loss_fn = get(config) — loss base conforme loss_type               │
    │    use_pinns or use_tiv_constraint?                                 │
    │      SIM → build_pinns_loss() (lambda_var × cenario + TIV)          │
    │    use_look_ahead_loss or use_dtb_loss?                             │
    │      SIM → combined (base + extras ponderados + PINNs)              │
    │      NAO → base simples (+ PINNs se ativas)                         │
    └──────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

from geosteering_ai.losses.catalog import (  # A: genericas; B: geofisicas (factories); C: geosteering; D: avancadas (factories)
    EPS,
    huber_loss,
    log_cosh_loss,
    mae_loss,
    make_adaptive_log_scale,
    make_adaptive_robust,
    make_cross_gradient,
    make_dilate,
    make_enc_decoder,
    make_log_scale_aware,
    make_look_ahead_weighted,
    make_morales_hybrid,
    make_multitask,
    make_robust_log_scale,
    make_sobolev,
    make_spectral,
    mape_loss,
    mbe_loss,
    mse_loss,
    msle_loss,
    nrmse_loss,
    probabilistic_nll,
    rae_loss,
    rmse_loss,
    rmsle_loss,
    rrmse_loss,
    rse_loss,
)
from geosteering_ai.losses.pinns import build_pinns_loss

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# SECAO: TIPOS VALIDOS
# ════════════════════════════════════════════════════════════════════════════
# Lista das 26 losses validas. Usada em PipelineConfig.__post_init__()
# para validacao e em list_available_losses() para catalogo.
# ──────────────────────────────────────────────────────────────────────────

VALID_LOSS_TYPES: list[str] = [
    # A: genericas
    "mse",
    "rmse",
    "mae",
    "mbe",
    "rse",
    "rae",
    "mape",
    "msle",
    "rmsle",
    "nrmse",
    "rrmse",
    "huber",
    "log_cosh",
    # B: geofisicas
    "log_scale_aware",
    "adaptive_log_scale",
    "robust_log_scale",
    "adaptive_robust",
    # C: geosteering
    "probabilistic_nll",
    "look_ahead_weighted",
    # D: avancadas
    "dilate",
    "enc_decoder",
    "multitask",
    "sobolev",
    "cross_gradient",
    "spectral",
    "morales_physics_hybrid",
]

# ════════════════════════════════════════════════════════════════════════════
# SECAO: LOSS REGISTRY
# ════════════════════════════════════════════════════════════════════════════
# Mapa de nome → (tipo, callable_or_factory).
# tipo = "direct" → funcao direta (y_true, y_pred)
# tipo = "factory" → funcao que recebe (config, epoch_var, noise_level_var)
#                    e retorna a closure compilada
# ──────────────────────────────────────────────────────────────────────────

_DIRECT = "direct"
_FACTORY = "factory"

_LOSS_REGISTRY: dict[str, tuple[str, object]] = {
    # ── A: Genericas — funcoes diretas ────────────────────────────────────
    "mse": (_DIRECT, mse_loss),
    "rmse": (_DIRECT, rmse_loss),
    "mae": (_DIRECT, mae_loss),
    "mbe": (_DIRECT, mbe_loss),
    "rse": (_DIRECT, rse_loss),
    "rae": (_DIRECT, rae_loss),
    "mape": (_DIRECT, mape_loss),
    "msle": (_DIRECT, msle_loss),
    "rmsle": (_DIRECT, rmsle_loss),
    "nrmse": (_DIRECT, nrmse_loss),
    "rrmse": (_DIRECT, rrmse_loss),
    "huber": (_DIRECT, huber_loss),
    "log_cosh": (_DIRECT, log_cosh_loss),
    # ── B: Geofisicas — factories ─────────────────────────────────────────
    "log_scale_aware": (_FACTORY, make_log_scale_aware),
    "adaptive_log_scale": (_FACTORY, make_adaptive_log_scale),
    "robust_log_scale": (_FACTORY, make_robust_log_scale),
    "adaptive_robust": (_FACTORY, make_adaptive_robust),
    # ── C: Geosteering — misto ────────────────────────────────────────────
    "probabilistic_nll": (_DIRECT, probabilistic_nll),
    "look_ahead_weighted": (_FACTORY, make_look_ahead_weighted),
    # ── D: Avancadas — factories ──────────────────────────────────────────
    "dilate": (_FACTORY, make_dilate),
    "enc_decoder": (_FACTORY, make_enc_decoder),
    "multitask": (_FACTORY, make_multitask),
    "sobolev": (_FACTORY, make_sobolev),
    "cross_gradient": (_FACTORY, make_cross_gradient),
    "spectral": (_FACTORY, make_spectral),
    "morales_physics_hybrid": (_FACTORY, make_morales_hybrid),
}

assert (
    len(_LOSS_REGISTRY) == 26
), f"Registry deve ter 26 losses, encontrado {len(_LOSS_REGISTRY)}"


# ════════════════════════════════════════════════════════════════════════════
# SECAO: LOSS FACTORY
# ════════════════════════════════════════════════════════════════════════════
# LossFactory: fachada OO para construcao de loss functions.
# Aceita config: PipelineConfig e variaveis opcionais de runtime
# (epoch_var, noise_level_var) para losses adaptativas.
# ──────────────────────────────────────────────────────────────────────────


class LossFactory:
    """Fabrica de loss functions para o pipeline de inversao de resistividade.

    Interface unica para obter qualquer uma das 26 losses do catalogo,
    compilar a loss combinada (base + geosteering + DTB + PINNs) e listar
    as losses disponiveis.

    Attributes:
        N_LOSSES (int): Total de losses no registry (26).

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> from geosteering_ai.losses.factory import LossFactory
        >>> config = PipelineConfig.baseline()
        >>> loss_fn = LossFactory.get(config)          # rmse (default)
        >>> combined = LossFactory.build_combined(config)
        >>> model.compile(loss=combined, optimizer="adam")

    Note:
        Utilizado em: training/loop.py (TrainingLoop.run()).
        Ref: docs/ARCHITECTURE_v2.md secao 6.
    """

    N_LOSSES: int = 26

    @classmethod
    def get(
        cls,
        config: "PipelineConfig",
        epoch_var=None,
        noise_level_var=None,
    ) -> Callable:
        """Retorna a loss base compilada para model.compile().

        Resolve a loss a partir de config.loss_type. Losses do tipo "direct"
        sao retornadas diretamente; losses do tipo "factory" sao compiladas
        com os parametros do config e das variaveis de runtime.

        Args:
            config: PipelineConfig com loss_type e hiperparametros da loss.
            epoch_var: tf.Variable(int) com epoca atual. Opcional.
                       Necessario para: log_scale_aware, adaptive_log_scale,
                       robust_log_scale, adaptive_robust, morales_physics_hybrid.
            noise_level_var: tf.Variable(float) com nivel de ruido atual.
                             Opcional. Necessario para: adaptive_log_scale,
                             adaptive_robust.

        Returns:
            Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar float32.

        Raises:
            ValueError: Se config.loss_type nao esta em VALID_LOSS_TYPES.

        Example:
            >>> config = PipelineConfig(loss_type="log_scale_aware")
            >>> loss_fn = LossFactory.get(config, epoch_var=my_var)

        Note:
            Referenciado em: training/loop.py (TrainingLoop.run()).
            Para loss combinada (base + extras), usar build_combined().
        """
        loss_type = config.loss_type
        if loss_type not in _LOSS_REGISTRY:
            raise ValueError(
                f"LossFactory.get: loss_type='{loss_type}' invalida. "
                f"Opcoes: {VALID_LOSS_TYPES}"
            )

        kind, fn = _LOSS_REGISTRY[loss_type]

        if kind == _DIRECT:
            logger.debug("LossFactory.get: loss '%s' (direct)", loss_type)
            return fn

        # Factory: chamar com config + variaveis de runtime
        logger.debug("LossFactory.get: compilando factory '%s'", loss_type)

        # Losses de ruido precisam de noise_level_var
        if loss_type in ("adaptive_log_scale", "adaptive_robust"):
            return fn(config, epoch_var=epoch_var, noise_level_var=noise_level_var)

        # Losses de epoch precisam de epoch_var
        if loss_type in ("log_scale_aware", "robust_log_scale", "morales_physics_hybrid"):
            return fn(config, epoch_var=epoch_var)

        # Factories simples (sem variaveis de runtime)
        return fn(config)

    @classmethod
    def build_combined(
        cls,
        config: "PipelineConfig",
        epoch_var=None,
        noise_level_var=None,
        pinns_lambda_var=None,
    ) -> Callable:
        """Constroi a loss combinada final para model.compile().

        Fluxo de decisao:
            1. use_morales_hybrid_loss → morales_physics_hybrid (#26)
            2. loss base = LossFactory.get(config)
            3. use_look_ahead_loss → adiciona look_ahead_weighted
            4. use_dtb_loss → adiciona MSE nos canais DTB [2:4] (DTB_up, DTB_down)
            5. pinns_lambda_var → adiciona termo PINNs (via callback externo)

        A loss combinada e:
            L = (1−w_la−w_dtb) × L_base
              + w_la × L_look_ahead
              + w_dtb × L_dtb
              (PINNs sao adicionadas externamente via GradientTape no loop)

        Args:
            config: PipelineConfig com use_morales_hybrid_loss,
                    use_look_ahead_loss, look_ahead_weight,
                    use_dtb_loss, dtb_weight.
            epoch_var: tf.Variable(int). Opcional.
            noise_level_var: tf.Variable(float). Opcional.
            pinns_lambda_var: tf.Variable(float). Ignorado aqui (PINNs
                              adicionadas no loop de treinamento externo).

        Returns:
            Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

        Example:
            >>> loss_fn = LossFactory.build_combined(config, epoch_var=ep)
            >>> model.compile(loss=loss_fn, optimizer="adam")

        Note:
            Referenciado em: training/loop.py (TrainingLoop.run()).
            DTB usa canais [2:4] de y_pred (DTB_up, DTB_down).
            Base loss opera APENAS em canais rho [0:2] quando DTB ativo.
        """
        # ── Caso 1: Morales hybrid substitui a loss base ──────────────────
        if config.use_morales_hybrid_loss:
            fn = make_morales_hybrid(config, epoch_var=epoch_var)
            logger.info("LossFactory: usando morales_physics_hybrid (#26)")
            return fn

        # ── Caso base: loss do loss_type ──────────────────────────────────
        base_fn = cls.get(config, epoch_var=epoch_var, noise_level_var=noise_level_var)
        # ── Construir loss PINNs (cenario + TIV constraint) ──────────────
        # build_pinns_loss retorna loss nula se use_pinns=False e
        # use_tiv_constraint=False. Caso contrario, combina cenario
        # PINN + TIV com lambda schedule.
        _pinns_active = config.use_pinns or config.use_tiv_constraint
        pinns_fn = None
        if _pinns_active:
            pinns_fn = build_pinns_loss(config, epoch_var=epoch_var)

        logger.info(
            "LossFactory: loss_type='%s', look_ahead=%s, dtb=%s, pinns=%s",
            config.loss_type,
            config.use_look_ahead_loss,
            config.use_dtb_loss,
            _pinns_active,
        )

        needs_combined = (
            config.use_look_ahead_loss or config.use_dtb_loss or _pinns_active
        )
        if not needs_combined:
            return base_fn

        # ── Caso combinado: base + extras ─────────────────────────────────
        w_la = config.look_ahead_weight if config.use_look_ahead_loss else 0.0
        w_dtb = config.dtb_weight if config.use_dtb_loss else 0.0
        w_base = max(1.0 - w_la - w_dtb, 0.01)

        look_ahead_fn = (
            make_look_ahead_weighted(config) if config.use_look_ahead_loss else None
        )

        # ── Pre-computar flag DTB para uso na closure ─────────────────
        _dtb_active = config.use_dtb_as_target

        def combined_loss_fn(y_true, y_pred):
            import tensorflow as tf

            # ── Loss base: APENAS canais de resistividade ─────────────
            # Quando DTB ativo (6 canais), a loss base opera SOMENTE
            # nos canais rho (0,1) para evitar mistura de dominios:
            #   canais 0,1: rho_h, rho_v (log10-scaled)
            #   canais 2,3: DTB_up, DTB_down (DTB-scaled, dominio diferente)
            #   canais 4,5: rho_up, rho_down (log10-scaled)
            # Se DTB inativo, loss base opera sobre todos os canais.
            if _dtb_active:
                y_true_base = y_true[..., :2]
                y_pred_base = y_pred[..., :2]
            else:
                n_target = tf.shape(y_true)[-1]
                y_pred_base = y_pred[..., :n_target]
                y_true_base = y_true
            total = w_base * base_fn(y_true_base, y_pred_base)

            # Look-ahead geosteering (opera sobre canais rho)
            if look_ahead_fn is not None:
                total = total + w_la * look_ahead_fn(y_true_base, y_pred_base)

            # ── DTB: canais 2,3 de y — DTB_up e DTB_down ─────────────
            # Quando use_dtb_as_target=True e output_channels>=4,
            # canais [2, 3] contem DTB_up e DTB_down (ja scaled).
            # Loss: MSE com clipping em dtb_max (DOD fisico).
            # Ref: docs/physics/perspectivas.md secao P5 (L_DTB).
            if config.use_dtb_loss:
                dtb_pred = y_pred[..., 2:4]
                dtb_true = y_true[..., 2:4]
                dtb_loss = tf.reduce_mean(tf.square(dtb_true - dtb_pred))
                total = total + w_dtb * dtb_loss

            # ── PINNs: cenario + TIV constraint ──────────────────────
            # PINNs adicionadas ADITIVAMENTE a loss total (nao ponderadas
            # pelo w_base). Lambda schedule controla o peso internamente.
            # Ref: losses/pinns.py — build_pinns_loss().
            if pinns_fn is not None:
                total = total + pinns_fn(y_true, y_pred)

            return total

        return combined_loss_fn

    @classmethod
    def list_available(cls) -> list[str]:
        """Retorna lista das 26 losses disponiveis.

        Returns:
            list[str]: Nomes ordenados de todas as losses no registry.

        Example:
            >>> LossFactory.list_available()
            ['adaptive_log_scale', 'adaptive_robust', ...]
        """
        return sorted(_LOSS_REGISTRY.keys())


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCAO CONVENIENTE
# ════════════════════════════════════════════════════════════════════════════
# build_loss_fn() e a API funcional equivalente a LossFactory.build_combined().
# Ponto de entrada preferido para uso no loop de treinamento.
# ──────────────────────────────────────────────────────────────────────────


def build_loss_fn(
    config: "PipelineConfig",
    epoch_var=None,
    noise_level_var=None,
    pinns_lambda_var=None,
) -> Callable:
    """Constroi a loss function final a partir de config.

    Wrapper conveniente sobre LossFactory.build_combined(). Ponto de uso
    preferido em training/loop.py.

    Args:
        config: PipelineConfig com todos os parametros de loss.
        epoch_var: tf.Variable(int) com epoca atual. Opcional.
        noise_level_var: tf.Variable(float) com nivel de ruido. Opcional.
        pinns_lambda_var: tf.Variable(float) para PINNs. Opcional (ignorado).

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar float32.

    Example:
        >>> from geosteering_ai.losses.factory import build_loss_fn
        >>> loss_fn = build_loss_fn(config)
        >>> model.compile(loss=loss_fn, optimizer="adam")

    Note:
        Referenciado em: training/loop.py (TrainingLoop.run()).
        Ref: docs/ARCHITECTURE_v2.md secao 6.
    """
    return LossFactory.build_combined(
        config,
        epoch_var=epoch_var,
        noise_level_var=noise_level_var,
        pinns_lambda_var=pinns_lambda_var,
    )


def list_available_losses() -> list[str]:
    """Retorna lista das 26 losses disponiveis no catalogo.

    Returns:
        list[str]: Nomes das 26 losses em ordem alfabetica.

    Example:
        >>> losses = list_available_losses()
        >>> len(losses)
        26
    """
    return LossFactory.list_available()


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    "VALID_LOSS_TYPES",
    "LossFactory",
    "build_loss_fn",
    "list_available_losses",
]
