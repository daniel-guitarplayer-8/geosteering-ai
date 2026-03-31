# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: losses/pinns.py                                                  ║
# ║  Bloco: 4b — Physics-Informed Neural Network Losses (PINNs)              ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (ponto unico de verdade)                ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • 3 cenarios de PINN loss: oracle, surrogate, maxwell                  ║
# ║    • TIVConstraintLoss: penaliza violacoes de rho_v >= rho_h              ║
# ║    • PINNsLambdaSchedule: 4 estrategias de annealing para lambda          ║
# ║    • build_pinns_loss(): factory central para loss PINNs                   ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig)                                 ║
# ║  Exports: ~7 funcoes/classes — ver __all__                                ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 18 (PINNs)                           ║
# ║       Morales et al. (2025) GJI — PINN para inversao EM triaxial         ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""Physics-Informed Neural Network (PINN) losses para inversao EM 1D.

Integra constraintes fisicas na funcao de perda do pipeline de inversao,
regularizando o modelo para produzir predicoes fisicamente consistentes.

Motivacao fisica:
    Em inversao EM, a rede aprende somente a partir de dados (L_data).
    Sem constraintes fisicas, ela pode produzir:
      - rho_v < rho_h (viola anisotropia TIV)
      - Oscilacoes espurias em zonas homogeneas
      - Residuos grandes na equacao de Maxwell

    PINNs adicionam um termo L_physics a funcao de perda total:
        L_total = L_data + lambda(t) × L_physics

    O peso lambda(t) segue um schedule com warmup (lambda=0 nas primeiras
    epocas, permitindo que a rede aprenda o mapeamento basico) e ramp
    (lambda cresce gradualmente para o valor alvo).

    ┌──────────────────────────────────────────────────────────────────────┐
    │  3 CENARIOS DE PINN LOSS                                             │
    │                                                                      │
    │  Cenario    │ Formula L_physics        │ Custo  │ Precisao          │
    │  ───────────┼──────────────────────────┼────────┼───────────────────│
    │  oracle     │ ||rho - rho_ref||        │ Baixo  │ Depende de ref    │
    │  surrogate  │ ||H_meas - F(rho_pred)|| │ Medio  │ Alta (end-to-end) │
    │  maxwell    │ ||nabla^2E + k^2E||      │ Alto   │ Maxima (PDE)      │
    │                                                                      │
    │  + TIV Constraint: max(0, rho_h - rho_v) (sempre ativo se habilitado)│
    │                                                                      │
    │  Lambda Schedules:                                                   │
    │    fixed  : lambda = target (apos warmup)                            │
    │    linear : lambda = target × progress                               │
    │    cosine : lambda = target × 0.5 × (1 - cos(pi × progress))        │
    │    step   : lambda = target (after warmup + ramp/2)                   │
    └──────────────────────────────────────────────────────────────────────┘

Ref: Morales et al. (2025) "Anisotropic resistivity estimation... PINN",
     Geophysical Journal International, ggaf101.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ── Epsilon float32-safe (NUNCA 1e-30) ────────────────────────────────
EPS = 1e-12

# ── Cenarios e schedules validos ──────────────────────────────────────
VALID_PINNS_SCENARIOS = {"oracle", "surrogate", "maxwell"}
VALID_LAMBDA_SCHEDULES = {"fixed", "linear", "cosine", "step"}


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente: schedules, cenarios, constraints, factory.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Constantes ────────────────────────────────────────────────────────
    "VALID_PINNS_SCENARIOS",
    "VALID_LAMBDA_SCHEDULES",
    # ── Lambda schedule ───────────────────────────────────────────────────
    "compute_lambda_schedule",
    # ── Cenarios PINN ─────────────────────────────────────────────────────
    "make_oracle_physics_loss",
    "make_surrogate_physics_loss",
    "make_maxwell_physics_loss",
    # ── TIV constraint ────────────────────────────────────────────────────
    "make_tiv_constraint_loss",
    # ── Factory central ───────────────────────────────────────────────────
    "build_pinns_loss",
]


# ════════════════════════════════════════════════════════════════════════════
# SECAO: LAMBDA SCHEDULE
# ════════════════════════════════════════════════════════════════════════════
# Controla a evolucao temporal do peso lambda_physics durante o treinamento.
# Lambda = 0 durante warmup (rede aprende mapeamento limpo).
# Lambda cresce de 0 ate pinns_lambda durante ramp (4 estrategias).
# Lambda = pinns_lambda constante apos warmup + ramp.
#
# Ref: Morales et al. (2025) — cosine annealing para lambda.
#      Bai et al. (2022) arXiv:2210.09060 — warmup + ramp para PINNs.
# ──────────────────────────────────────────────────────────────────────────


def compute_lambda_schedule(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    lambda_target: float,
    schedule: str = "linear",
) -> float:
    """Computa o valor de lambda_physics para uma dada epoca.

    O schedule implementa 3 fases:
      1. Warmup (epoch < warmup_epochs): lambda = 0
         A rede aprende o mapeamento data-driven sem constraintes fisicas.
      2. Ramp (warmup_epochs <= epoch < warmup_epochs + ramp_epochs):
         lambda cresce de 0 ate lambda_target segundo a estrategia.
      3. Hold (epoch >= warmup_epochs + ramp_epochs): lambda = lambda_target

    Diagrama:
      ┌──────────────────────────────────────────────────────┐
      │  lambda                                               │
      │  ▲        ╱── linear                                  │
      │  │       ╱                                            │
      │  │ target ─────────── hold                            │
      │  │     ╱                                              │
      │  │   ╱  cosine (S-curve)                              │
      │  │  ╱                                                 │
      │  │ 0────── warmup ──── ramp ──────── hold             │
      │  └───────────────────────────── epoch                 │
      └──────────────────────────────────────────────────────┘

    Args:
        epoch: Epoca atual (0-indexed).
        warmup_epochs: Numero de epocas com lambda=0.
        ramp_epochs: Numero de epocas de rampa (0→target).
        lambda_target: Valor alvo de lambda apos ramp.
        schedule: Estrategia de rampa. Opcoes:
            - "fixed": lambda = target (imediatamente apos warmup)
            - "linear": lambda = target × progress (rampa reta)
            - "cosine": lambda = target × 0.5 × (1 - cos(pi × progress))
              (Morales I2 — curva S suave, atrasa inicio e final)
            - "step": lambda = target quando progress >= 0.5, senao 0
              (degrau no meio do ramp)

    Returns:
        float: Valor de lambda em [0, lambda_target].

    Example:
        >>> compute_lambda_schedule(0, warmup_epochs=10, ramp_epochs=20,
        ...                         lambda_target=0.01, schedule="linear")
        0.0  # warmup phase
        >>> compute_lambda_schedule(20, warmup_epochs=10, ramp_epochs=20,
        ...                         lambda_target=0.01, schedule="linear")
        0.005  # mid-ramp
        >>> compute_lambda_schedule(50, warmup_epochs=10, ramp_epochs=20,
        ...                         lambda_target=0.01, schedule="linear")
        0.01  # hold phase

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (schedule interno)
            - training/callbacks.py: PINNsLambdaCallback (atualiza tf.Variable)
            - tests/test_pinns.py: TestLambdaSchedule
        Ref: Morales et al. (2025) — cosine annealing (I2).
             Bai et al. (2022) — warmup + ramp para estabilidade PINN.
    """
    # ── Fase 1: Warmup (lambda = 0) ──────────────────────────────────
    # Rede aprende mapeamento basico sem constrainte fisica.
    # Sem warmup, PINNs podem dominar a loss inicial e impedir
    # convergencia do termo de dados.
    if epoch < warmup_epochs:
        return 0.0

    # ── Fase 3: Hold (lambda = target) ────────────────────────────────
    # Apos ramp, lambda permanece constante no valor alvo.
    if ramp_epochs <= 0 or epoch >= warmup_epochs + ramp_epochs:
        return lambda_target

    # ── Fase 2: Ramp (0 → target) ────────────────────────────────────
    # progress ∈ [0, 1] indica fracao do ramp concluida.
    progress = (epoch - warmup_epochs) / ramp_epochs

    if schedule == "fixed":
        # ── Fixed: salto imediato para target apos warmup ────────────
        # Uso: quando se quer ativar PINNs de forma brusca.
        return lambda_target

    elif schedule == "linear":
        # ── Linear: rampa reta proporcional a progress ───────────────
        # lambda = target × (epoch - warmup) / ramp
        # Simples e previsivel. Default recomendado para primeiros testes.
        return lambda_target * progress

    elif schedule == "cosine":
        # ── Cosine: curva S suave (Morales I2) ───────────────────────
        # lambda = target × 0.5 × (1 - cos(pi × progress))
        # Vantagem: inicio e final suaves evitam descontinuidades
        # no gradiente da loss total. Recomendado para producao.
        return lambda_target * 0.5 * (1.0 - math.cos(math.pi * progress))

    elif schedule == "step":
        # ── Step: degrau no meio do ramp ─────────────────────────────
        # lambda = 0 para progress < 0.5, target para progress >= 0.5.
        # Uso: quando se deseja transicao abrupta controlada.
        return lambda_target if progress >= 0.5 else 0.0

    else:
        raise ValueError(
            f"Lambda schedule '{schedule}' invalido. "
            f"Validos: {VALID_LAMBDA_SCHEDULES}"
        )


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CENARIO 1 — ORACLE PHYSICS LOSS
# ════════════════════════════════════════════════════════════════════════════
# Compara predicao com valores de referencia pre-computados (Fortran).
# Cenario mais simples: requer dados de referencia acessiveis.
# Custo computacional baixo (apenas norma de diferenca).
#
# Formula:
#   L_oracle = ||rho_pred - rho_ref||_norm
#
# Ref: ARCHITECTURE_v2.md secao 18.3.1
# ──────────────────────────────────────────────────────────────────────────


def make_oracle_physics_loss(
    config: "PipelineConfig",
) -> Callable:
    """Factory para Oracle Physics Loss — comparacao com referencia Fortran.

    Compara a predicao da rede com valores de referencia pre-computados
    pelo simulador Fortran (forward model). Este eh o cenario mais simples
    de PINN: nao requer diferenciacao do forward model, apenas acesso
    aos dados de referencia.

    Formula:
        L_oracle = ||rho_pred - rho_ref||_norm

        Onde norm pode ser:
          - "l2": MSE(rho_pred, rho_ref) — penaliza grandes erros
          - "l1": MAE(rho_pred, rho_ref) — robusto a outliers
          - "huber": Huber(rho_pred, rho_ref) — hibrido L1/L2

    Uso pratico:
        O oracle loss opera como dupla supervisao: alem do target real
        (y_true = rho do dataset), a rede tambem eh comparada com a
        saida do simulador Fortran. Util quando:
          - Dados sinteticos de alta qualidade disponiveis
          - Forward model confiavel mas nao diferenciavel
          - Regularizacao adicional desejada para zonas dificeis

    Args:
        config: PipelineConfig com pinns_physics_norm.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.
            y_true: (batch, seq_len, 2) — rho_h, rho_v de referencia.
            y_pred: (batch, seq_len, 2) — rho_h, rho_v preditos.

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (cenario "oracle")
            - losses/factory.py: _LOSS_REGISTRY (se registrado)
            - tests/test_pinns.py: TestOraclePhysicsLoss
        Ref: ARCHITECTURE_v2.md secao 18.3.1.
        Neste cenario, y_true funciona como rho_ref (dados sinteticos
        do Fortran). Para dados de campo, rho_ref deve ser fornecido
        separadamente via callback ou dataset auxiliar.
    """
    norm = config.pinns_physics_norm

    def oracle_physics_loss(y_true, y_pred):
        """Oracle loss: compara predicao com referencia.

        Opera sobre canais de resistividade (0:2) em log10 scale.
        """
        import tensorflow as tf

        # ── Canais de resistividade (rho_h, rho_v) ──────────────
        # Em modo DTB (output_channels >= 4), opera apenas nos canais rho.
        # y_true ja contem rho_ref; y_pred contem rho_pred.
        n_target = tf.minimum(tf.shape(y_true)[-1], 2)
        y_t = y_true[..., :n_target]
        y_p = y_pred[..., :n_target]

        if norm == "l2":
            # ── MSE: penaliza grandes erros quadraticamente ──────
            return tf.reduce_mean(tf.square(y_t - y_p))
        elif norm == "l1":
            # ── MAE: robusto a outliers em dados de campo ────────
            return tf.reduce_mean(tf.abs(y_t - y_p))
        else:
            # ── Huber: transicao suave L2→L1 em delta=1.0 ───────
            # Combina precisao do MSE (erros pequenos) com
            # robustez do MAE (erros grandes/outliers).
            return tf.reduce_mean(tf.keras.losses.huber(y_t, y_p, delta=1.0))

    return oracle_physics_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CENARIO 2 — SURROGATE PHYSICS LOSS
# ════════════════════════════════════════════════════════════════════════════
# Usa um modelo neural substituto (surrogate) do forward model Fortran.
# O surrogate mapeia rho → H_EM (campo EM predito), permitindo
# comparacao end-to-end diferenciavel com os dados medidos.
#
# Formula:
#   L_surrogate = ||H_meas - F_surrogate(rho_pred)||_norm
#
# Ref: ARCHITECTURE_v2.md secao 18.3.2
#      Morales et al. (2025) — Klein equations como forward model
# ──────────────────────────────────────────────────────────────────────────


def make_surrogate_physics_loss(
    config: "PipelineConfig",
) -> Callable:
    """Factory para Surrogate Physics Loss — forward model neural.

    Usa um modelo neural pre-treinado (surrogate) que aproxima o
    forward model Fortran: rho → H_EM. A loss compara os campos EM
    medidos com os preditos pelo surrogate a partir de rho_pred.

    Fluxo:
      ┌──────────────────────────────────────────────────────────┐
      │  H_measured (dados reais)                                 │
      │    ↓                                                      │
      │  Inverse Network: H_measured → rho_pred                  │
      │    ↓                                                      │
      │  Surrogate Forward: rho_pred → H_predicted               │
      │    ↓                                                      │
      │  L_surrogate = ||H_measured - H_predicted||_norm          │
      │                                                           │
      │  End-to-end diferenciavel: gradientes fluem              │
      │  do H_measured ate rho_pred via surrogate.                │
      └──────────────────────────────────────────────────────────┘

    NOTA IMPORTANTE:
        Esta implementacao eh um placeholder estrutural. O surrogate
        model deve ser treinado separadamente e carregado via
        config.surrogate_model_path. Ate que o surrogate esteja
        disponivel, esta loss retorna 0.0 (sem efeito).

    Args:
        config: PipelineConfig com pinns_physics_norm,
                pinns_use_forward_surrogate, surrogate_model_path.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (cenario "surrogate")
            - tests/test_pinns.py: TestSurrogatePhysicsLoss
        Ref: ARCHITECTURE_v2.md secao 18.3.2.
             Morales et al. (2025) — Klein equations como forward model
             analitico diferenciavel (2 equacoes algebricas TI).
        Status: Placeholder — surrogate model a ser treinado em v2.1.
    """
    use_surrogate = config.pinns_use_forward_surrogate
    model_path = config.surrogate_model_path

    # ── Tentativa de carregar surrogate model ─────────────────────────
    # Se path nao existe ou flag desativada, loss retorna 0.0.
    _surrogate_model = None
    if use_surrogate and model_path:
        try:
            import tensorflow as tf

            _surrogate_model = tf.keras.models.load_model(model_path)
            logger.info(
                "Surrogate model carregado: %s (%d params)",
                model_path,
                _surrogate_model.count_params(),
            )
        except Exception as e:
            logger.warning(
                "Surrogate model nao carregado (%s): %s. " "L_surrogate retornara 0.0.",
                model_path,
                e,
            )

    norm = config.pinns_physics_norm

    def surrogate_physics_loss(y_true, y_pred):
        """Surrogate loss: forward model neural.

        Compara H_measured (via y_true como proxy) com F(rho_pred).
        """
        import tensorflow as tf

        # ── Placeholder: sem surrogate → loss nula ────────────────
        # Quando o surrogate nao esta disponivel, retorna 0.0
        # para nao afetar o treinamento. Lambda schedule cuidara
        # de desativar o termo PINNs automaticamente (lambda=0).
        if _surrogate_model is None:
            logger.debug("Surrogate nao disponivel — L_surrogate = 0.0")
            return tf.constant(0.0, dtype=tf.float32)

        # ── Forward pass pelo surrogate ───────────────────────────
        # rho_pred (batch, seq_len, 2) → H_pred (batch, seq_len, 4)
        # Canais rho: opera apenas em [0:2]
        rho_pred = y_pred[..., :2]
        h_pred = _surrogate_model(rho_pred, training=False)

        # ── Comparacao com H_measured ─────────────────────────────
        # y_true contem rho verdadeiro, NAO H_measured diretamente.
        # Em implementacao completa, H_measured viria do pipeline.
        # Por ora, compara rho → rho roundtrip como proxy.
        # TODO(v2.1): receber H_measured via dataset auxiliar.
        h_ref = _surrogate_model(y_true[..., :2], training=False)

        if norm == "l2":
            return tf.reduce_mean(tf.square(h_ref - h_pred))
        elif norm == "l1":
            return tf.reduce_mean(tf.abs(h_ref - h_pred))
        else:
            return tf.reduce_mean(tf.keras.losses.huber(h_ref, h_pred, delta=1.0))

    return surrogate_physics_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CENARIO 3 — MAXWELL PHYSICS LOSS (PDE RESIDUAL)
# ════════════════════════════════════════════════════════════════════════════
# Computa o residuo da equacao de Helmholtz (difusao EM) diretamente
# a partir de rho_pred usando diferencas finitas para a 2a derivada.
#
# Equacao de Helmholtz 1D (regime quase-estacionario):
#   d^2E/dz^2 + k^2(z) × E(z) = 0
#   k^2 = -i × omega × mu × sigma(z) = -i × omega × mu / rho(z)
#
# Residuo (formulacao simplificada em magnitude):
#   R(z) = d^2(rho)/dz^2 / (1 + k^2(z))
#   L_maxwell = mean(|R(z)|^2)
#
# Ref: ARCHITECTURE_v2.md secao 18.3.3
#      Bai et al. (2022) — PINNs para mecanica computacional
# ──────────────────────────────────────────────────────────────────────────


def make_maxwell_physics_loss(
    config: "PipelineConfig",
) -> Callable:
    """Factory para Maxwell Physics Loss — residuo PDE via diferencas finitas.

    Computa o residuo da equacao de Helmholtz 1D para verificar se
    o perfil de resistividade predito eh fisicamente plausivel.
    Penaliza curvatura excessiva do perfil, normalizada pela
    condutividade local (meios condutivos permitem mais resolucao).

    Equacao de Helmholtz 1D:
        d^2E/dz^2 + k^2(z) × E(z) = 0

        k^2(z) = omega × mu × sigma(z) (magnitude)
        sigma(z) = 1 / rho(z)
        omega = 2 × pi × frequency_hz
        mu = 4 × pi × 1e-7  (permeabilidade do vacuo, H/m)

    Formulacao simplificada (magnitude real):
        Como operamos em log10(rho), usamos uma proxy baseada na
        suavidade do perfil de resistividade ponderada pela frequencia.
        O residuo penaliza transicoes nao-fisicas (mais abruptas que
        o skin depth permitiria para a frequencia dada).

    Diagrama:
      ┌──────────────────────────────────────────────────────────┐
      │  RESIDUO DE MAXWELL (SIMPLIFICADO)                        │
      │                                                           │
      │  rho_pred (log10 scale) → sigma = 10^(-rho_pred)         │
      │                                                           │
      │  k^2 = omega × mu / 10^(rho_pred)  (magnitude)           │
      │  delta = sqrt(2 / (omega × mu × sigma)) = skin depth     │
      │                                                           │
      │  Residuo:                                                 │
      │    R = d^2(rho_pred)/dz^2  (curvatura do perfil)         │
      │    Penalidade = mean(R^2 / (1 + k^2))                    │
      │    Normalizado por (1+k^2): curvatura permitida cresce    │
      │    com condutividade (meios condutivos → mais resolucao)  │
      └──────────────────────────────────────────────────────────┘

    NOTA: Esta eh uma formulacao simplificada do residuo de Maxwell.
    A implementacao completa com campo E complexo e derivadas de 2a
    ordem sera desenvolvida em v2.1 quando o surrogate forward
    model estiver disponivel para fornecer E(z).

    Args:
        config: PipelineConfig com frequency_hz, spacing_meters,
                pinns_physics_norm.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (cenario "maxwell")
            - tests/test_pinns.py: TestMaxwellPhysicsLoss
        Ref: ARCHITECTURE_v2.md secao 18.3.3.
             Bai et al. (2022) arXiv:2210.09060 — PINNs para PDEs.
        Fisica: omega = 2*pi*f, mu = 4*pi*1e-7 H/m.
        Custo: derivada de 2a ordem via diferencas finitas (eficiente).
    """
    # ── Constantes EM ─────────────────────────────────────────────────
    # omega: frequencia angular (rad/s). Default: 2*pi*20000 ≈ 125664 rad/s.
    # mu: permeabilidade magnetica do vacuo (H/m). Constante universal.
    # omega_mu: produto pre-computado para eficiencia na closure.
    omega = 2.0 * math.pi * config.frequency_hz
    mu = 4.0 * math.pi * 1e-7
    omega_mu = omega * mu
    # ── dz^2: quadrado do espacamento entre pontos de medicao ─────
    # Necessario para normalizar a derivada de 2a ordem:
    # d^2f/dz^2 ≈ (f[i+1] - 2f[i] + f[i-1]) / dz^2
    # Com SPACING_METERS=1.0 (default), dz_sq=1.0 (no-op).
    # Para spacings diferentes (0.5, 2.0), corrige a escala fisica.
    dz_sq = config.spacing_meters**2

    def maxwell_physics_loss(y_true, y_pred):
        """Maxwell residual loss (simplificada via diferencas finitas).

        Penaliza curvatura excessiva do perfil de resistividade,
        normalizada pela condutividade local (skin depth constraint).
        """
        import tensorflow as tf

        # ── Extrair canais rho (log10 scale) ──────────────────────
        # Canal 0: log10(rho_h), Canal 1: log10(rho_v)
        rho_log = y_pred[..., :2]  # (B, N, 2)

        # ── Condutividade (S/m) a partir de log10(rho) ───────────
        # sigma = 1/rho = 10^(-log10(rho))
        # Clamp para evitar overflow: log10(rho) em [-2, 5]
        # corresponde a rho em [0.01, 100000] Ohm.m
        rho_log_clamped = tf.clip_by_value(rho_log, -2.0, 5.0)
        sigma = tf.pow(10.0, -rho_log_clamped)  # (B, N, 2)

        # ── k^2 = omega × mu × sigma (numero de onda ao quadrado) ─
        # k^2 grande → meio condutivo → mais resolucao → mais curvatura permitida.
        k_sq = omega_mu * sigma  # (B, N, 2)

        # ── Curvatura d^2(rho)/dz^2 via diferencas finitas centrais ─
        # d^2f/dz^2 ≈ (f[i+1] - 2*f[i] + f[i-1]) / dz^2
        # Interior: indices [1:-1] para ter vizinhos em ambos os lados.
        # dz_sq normaliza pela distancia real entre pontos de medicao.
        d2_rho = (
            rho_log[:, 2:, :] - 2.0 * rho_log[:, 1:-1, :] + rho_log[:, :-2, :]
        ) / dz_sq  # (B, N-2, 2)

        # ── Normalizacao pela condutividade local ─────────────────
        # Divisor: (1 + k^2) nos pontos interiores.
        # Meios condutivos (k^2 grande) permitem mais curvatura.
        # Meios resistivos (k^2 pequeno) penalizam curvatura forte.
        # +1 evita divisao por zero em rho muito alto (sigma → 0).
        k_sq_interior = k_sq[:, 1:-1, :]  # (B, N-2, 2)
        normalizer = 1.0 + k_sq_interior

        # ── Residuo normalizado ───────────────────────────────────
        # R = d^2(rho)/dz^2 / (1 + k^2)
        # L_maxwell = mean(R^2)
        residual = d2_rho / (normalizer + EPS)
        loss = tf.reduce_mean(tf.square(residual))

        return loss

    return maxwell_physics_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO: TIV CONSTRAINT LOSS
# ════════════════════════════════════════════════════════════════════════════
# Soft constraint que penaliza violacoes da anisotropia TIV:
#   rho_v >= rho_h (SEMPRE em meios TIV — Transversalmente Isotropicos)
#
# Em log10 scale: log10(rho_v) >= log10(rho_h)
#   → penalidade = max(0, log10(rho_h) - log10(rho_v))^2
#
# Motivacao fisica:
#   Em rochas sedimentares, a corrente flui mais facilmente ao longo
#   das camadas (horizontal) do que atraves delas (vertical).
#   Isso resulta em rho_v >= rho_h (ratio tipico 1.5-10x).
#   Violar esta constrainte indica predicao fisicamente invalida.
#
# Ref: Morales et al. (2025) — hard constraint via sigmoid/ReLU.
#      Neste pipeline: soft constraint via penalidade quadratica.
# ──────────────────────────────────────────────────────────────────────────


def make_tiv_constraint_loss(
    config: "PipelineConfig",
) -> Callable:
    """Factory para TIV Constraint Loss — penaliza rho_v < rho_h.

    Implementa soft constraint para anisotropia TIV (Transversalmente
    Isotropica Vertical). Em meios sedimentares, SEMPRE rho_v >= rho_h.

    Formula:
        violation = max(0, log10(rho_h) - log10(rho_v))
        L_tiv = mean(violation^2)

    Diagrama:
      ┌──────────────────────────────────────────────────────────┐
      │  TIV CONSTRAINT: rho_v >= rho_h                           │
      │                                                           │
      │  Cenario valido (sem penalidade):                         │
      │    rho_h = 10 Ohm.m, rho_v = 30 Ohm.m → ratio 3.0      │
      │    log10(rho_h) = 1.0, log10(rho_v) = 1.477              │
      │    violation = max(0, 1.0 - 1.477) = 0 ✓                │
      │                                                           │
      │  Cenario invalido (penalidade ativa):                     │
      │    rho_h = 100 Ohm.m, rho_v = 50 Ohm.m → ratio 0.5     │
      │    log10(rho_h) = 2.0, log10(rho_v) = 1.699              │
      │    violation = max(0, 2.0 - 1.699) = 0.301               │
      │    L_tiv = 0.301^2 = 0.0907                              │
      │                                                           │
      │  Tipico em TIV:                                           │
      │    Folhelho: rho_v/rho_h ∈ [3, 5]                       │
      │    Arenito argiloso: rho_v/rho_h ∈ [2, 5]               │
      │    Arenito limpo: rho_v/rho_h ≈ 1                        │
      │    NUNCA rho_v/rho_h < 1 em TIV                          │
      └──────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig. Utiliza apenas output_channels para
            validar que canais rho existem (>= 2).

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.
            Retorna 0.0 quando rho_v >= rho_h em todos os pontos.
            Retorna > 0 proporcional ao grau de violacao.

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (constraint TIV)
            - tests/test_pinns.py: TestTIVConstraintLoss
        Ref: Morales et al. (2025) — hard constraint via sigmoid/ReLU.
        Fisica: rho_v >= rho_h SEMPRE em meios TIV sedimentares.
            Ratio tipico 1.5-10x (folhelho 3-5x, arenito argiloso 2-5x).
            Excecao: arenito limpo tem ratio ≈ 1 (mas NUNCA < 1).
    """

    def tiv_constraint_loss(y_true, y_pred):
        """Soft constraint: penaliza rho_v < rho_h em log10 scale."""
        import tensorflow as tf

        # ── Extrair rho_h (canal 0) e rho_v (canal 1) ────────────
        # Em log10 scale: valores tipicos em [-1, 4] (0.1-10000 Ohm.m)
        rho_h_log = y_pred[..., 0]  # (B, N) — log10(rho_h)
        rho_v_log = y_pred[..., 1]  # (B, N) — log10(rho_v)

        # ── Violacao: max(0, rho_h - rho_v) em log10 ─────────────
        # Se rho_h > rho_v → violation > 0 → penalidade
        # Se rho_h <= rho_v → violation = 0 → sem penalidade
        violation = tf.nn.relu(rho_h_log - rho_v_log)

        # ── Loss: MSE da violacao ─────────────────────────────────
        # Quadratica: penaliza violacoes grandes mais fortemente.
        return tf.reduce_mean(tf.square(violation))

    return tiv_constraint_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FACTORY CENTRAL — build_pinns_loss()
# ════════════════════════════════════════════════════════════════════════════
# Constroi a loss PINN completa combinando:
#   1. Cenario (oracle/surrogate/maxwell) com lambda schedule
#   2. TIV constraint (opcional, sempre aditivo)
#
# Formula final:
#   L_pinns = lambda(t) × L_scenario + lambda_tiv × L_tiv
#
# A loss retornada eh um callable que recebe epoch_var (tf.Variable)
# para computar lambda(t) dinamicamente.
#
# Ref: ARCHITECTURE_v2.md secao 18.4
# ──────────────────────────────────────────────────────────────────────────


def build_pinns_loss(
    config: "PipelineConfig",
    epoch_var=None,
    pinns_lambda_var=None,
) -> Callable:
    """Factory central para a loss PINN completa.

    Combina o cenario PINN selecionado (oracle/surrogate/maxwell)
    com a TIV constraint loss. Lambda controlado via tf.Variable
    (graph-safe — NUNCA usar epoch_var.numpy() dentro da closure).

    Fluxo de decisao:
      ┌──────────────────────────────────────────────────────────┐
      │  config.use_pinns ou use_tiv_constraint = True?           │
      │    NAO → retorna loss nula (0.0)                          │
      │    SIM ↓                                                  │
      │  config.use_pinns = True?                                 │
      │    SIM → cenario (oracle/surrogate/maxwell)               │
      │  config.use_tiv_constraint = True?                        │
      │    SIM → + lambda_tiv × make_tiv_constraint_loss(config)  │
      │                                                           │
      │  Lambda via tf.Variable (graph-safe):                     │
      │    pinns_lambda_var atualizado por callback externo.       │
      │    Se None, cria variavel interna com valor fixo.          │
      │                                                           │
      │  L_pinns = lambda_var × L_scenario + w_tiv × L_tiv       │
      └──────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com use_pinns, pinns_scenario,
                pinns_lambda, use_tiv_constraint, tiv_constraint_weight.
        epoch_var: tf.Variable(int) com epoca atual. Opcional.
                   Usado apenas para logging, NAO para computar lambda.
        pinns_lambda_var: tf.Variable(float) com lambda atual.
                   Graph-safe: lido diretamente no forward pass sem
                   chamar .numpy(). Atualizado externamente por callback
                   (PINNsLambdaCallback) usando compute_lambda_schedule().
                   Se None, cria variavel interna com valor fixo.

    Returns:
        Callable: Funcao loss(y_true, y_pred) → tf.Tensor scalar.
            Retorna 0.0 se use_pinns=False e use_tiv_constraint=False.

    Example:
        >>> config = PipelineConfig(
        ...     use_pinns=True,
        ...     pinns_scenario="oracle",
        ...     pinns_lambda=0.01,
        ...     use_tiv_constraint=True,
        ...     tiv_constraint_weight=0.1,
        ... )
        >>> import tensorflow as tf
        >>> lam_var = tf.Variable(0.0, dtype=tf.float32)
        >>> pinns_loss = build_pinns_loss(config, pinns_lambda_var=lam_var)
        >>> loss_val = pinns_loss(y_true, y_pred)

    Note:
        Referenciado em:
            - losses/factory.py: build_combined() — integrado via slot PINNs
            - training/loop.py: TrainingLoop.run() — adiciona ao total loss
            - tests/test_pinns.py: TestBuildPINNsLoss
        Ref: ARCHITECTURE_v2.md secao 18.4.
        IMPORTANTE: lambda controlado via tf.Variable (graph-safe).
        NUNCA usar epoch_var.numpy() dentro da closure — quebra graph mode.
        O callback PINNsLambdaCallback atualiza pinns_lambda_var a cada
        epoca usando compute_lambda_schedule().
    """
    _has_pinns = config.use_pinns
    _has_tiv = config.use_tiv_constraint

    # ── Nenhuma constrainte ativa → loss nula ─────────────────────────
    if not _has_pinns and not _has_tiv:
        logger.debug("PINNs e TIV desativadas")

        def _null_loss(y_true, y_pred):
            import tensorflow as tf

            return tf.constant(0.0, dtype=tf.float32)

        return _null_loss

    # ── TIV constraint (pode ser ativada sem cenario PINN) ────────────
    tiv_fn = None
    tiv_weight = 0.0
    if _has_tiv:
        tiv_fn = make_tiv_constraint_loss(config)
        tiv_weight = config.tiv_constraint_weight
        logger.info(
            "TIV constraint ativada: weight=%.4f",
            tiv_weight,
        )

    # ── Cenario PINN (opcional — pode ter apenas TIV) ─────────────────
    scenario_fn = None
    if _has_pinns:
        scenario = config.pinns_scenario
        if scenario == "oracle":
            scenario_fn = make_oracle_physics_loss(config)
        elif scenario == "surrogate":
            scenario_fn = make_surrogate_physics_loss(config)
        elif scenario == "maxwell":
            scenario_fn = make_maxwell_physics_loss(config)
        else:
            raise ValueError(
                f"pinns_scenario='{scenario}' invalido. "
                f"Validos: {VALID_PINNS_SCENARIOS}"
            )

        logger.info(
            "PINNs ativadas: scenario='%s', lambda=%.4f, tiv=%s",
            scenario,
            config.pinns_lambda,
            _has_tiv,
        )

    # ── Lambda variable (graph-safe) ──────────────────────────────────
    # Se pinns_lambda_var nao fornecido, cria variavel interna com
    # valor fixo = pinns_lambda. Callback externo (PINNsLambdaCallback)
    # atualiza esta variavel a cada epoca seguindo o schedule.
    # NUNCA usar epoch_var.numpy() dentro da closure — quebra graph mode.
    _lambda_var = pinns_lambda_var
    if _lambda_var is None and _has_pinns:
        import tensorflow as tf

        _lambda_var = tf.Variable(
            config.pinns_lambda,
            dtype=tf.float32,
            trainable=False,
            name="pinns_lambda",
        )

    def pinns_combined_loss(y_true, y_pred):
        """Loss PINN combinada com lambda variable e TIV constraint.

        Graph-safe: usa tf.Variable para lambda (nao epoch_var.numpy()).
        """
        import tensorflow as tf

        total = tf.constant(0.0, dtype=tf.float32)

        # ── Loss do cenario (ponderada por lambda_var) ────────────
        # _lambda_var eh tf.Variable — leitura segura em graph mode.
        if scenario_fn is not None:
            total = total + _lambda_var * scenario_fn(y_true, y_pred)

        # ── TIV constraint ────────────────────────────────────────
        if tiv_fn is not None:
            total = total + tiv_weight * tiv_fn(y_true, y_pred)

        return total

    return pinns_combined_loss
