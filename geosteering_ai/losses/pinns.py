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
# ║    • 8 cenarios de PINN loss: oracle, surrogate, maxwell,                 ║
# ║      smoothness, skin_depth, continuity, variational, self_adaptive       ║
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
    │  8 CENARIOS DE PINN LOSS                                             │
    │                                                                      │
    │  Cenario       │ Formula L_physics            │ Custo  │ Tipo       │
    │  ──────────────┼─────────────────────────────-┼────────┼────────────│
    │  oracle        │ ||rho - rho_ref||            │ Baixo  │ Supervisao │
    │  surrogate     │ ||H_meas - F(rho_pred)||     │ Medio  │ End-to-end │
    │  maxwell       │ ||d^2rho/dz^2 / (1+k^2)||   │ Alto   │ PDE forte  │
    │  smoothness    │ L2(grad) + TV(grad)          │ Baixo  │ Tikhonov   │
    │  skin_depth    │ max(0, |grad| - 1/delta)^2   │ Baixo  │ Fisico     │
    │  continuity    │ ||d(rho)/dz|| (L1)           │ Baixo  │ Sparse     │
    │  variational   │ |grad|^2 / (1+k^2)          │ Medio  │ PDE fraca  │
    │  self_adaptive │ w(z) × R(z)^2 / mean(w)     │ Alto   │ Adaptativo │
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

# ── ln(10) pre-computado para conversao log10 → linear ──────────────
# Usado em skin_depth_loss para converter 1/delta para log10 space:
#   d(log10 rho)/dz = (d rho/dz) / (rho × ln(10))
_LN10 = math.log(10.0)  # ≈ 2.302585

# ── Cenarios e schedules validos ──────────────────────────────────────
VALID_PINNS_SCENARIOS = {
    "oracle",
    "surrogate",
    "maxwell",
    "smoothness",
    "skin_depth",
    "continuity",
    "variational",
    "self_adaptive",
}
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
    "make_smoothness_loss",
    "make_skin_depth_loss",
    "make_continuity_loss",
    "make_variational_loss",
    "make_self_adaptive_loss",
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
        # Fix CR#2: slice estatico [:2] preserva shape em tf.function.
        y_t = y_true[..., :2]
        y_p = y_pred[..., :2]

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
        # Fix CR#3: sem logger.debug dentro de closure traced — side effect.
        if _surrogate_model is None:
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

        # ── Fix CR#4: guard para seq_len < 3 (d2 requer 3+ pontos) ─
        seq_len = tf.shape(rho_log)[1]
        if_short = tf.less(seq_len, 3)
        if tf.executing_eagerly():
            if if_short:
                return tf.constant(0.0, dtype=tf.float32)

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
        # Fix CR#5: usar rho_log_clamped para consistencia com k_sq.
        d2_rho = (
            rho_log_clamped[:, 2:, :]
            - 2.0 * rho_log_clamped[:, 1:-1, :]
            + rho_log_clamped[:, :-2, :]
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
# SECAO: CENARIO 4 — SMOOTHNESS LOSS (TIKHONOV + TV)
# ════════════════════════════════════════════════════════════════════════════
# Regularizacao classica do perfil de resistividade combinando:
#   - Tikhonov (L2): penaliza gradientes suaves — estabiliza inversao
#   - Total Variation (L1): preserva descontinuidades geologicas reais
#
# Formula:
#   L_smooth = alpha_l2 × mean(|d(rho)/dz|^2)
#            + alpha_tv × mean(|d(rho)/dz|)
#
# Custo: MUITO baixo (apenas derivadas de 1a ordem via diferencas finitas).
# Ref: Tikhonov (1963) — regularizacao L2 para problemas mal-postos.
#      Rudin, Osher & Fatemi (1992) — TV preserva bordas.
# ──────────────────────────────────────────────────────────────────────────


def make_smoothness_loss(
    config: "PipelineConfig",
) -> Callable:
    """Factory para Smoothness Loss — Tikhonov + Total Variation.

    Regulariza o perfil de resistividade predito combinando penalidade
    L2 (Tikhonov) que estabiliza a inversao e penalidade L1 (TV) que
    preserva descontinuidades geologicas reais (limites de camada).

    Formula:
        d(rho)/dz = (rho[i+1] - rho[i]) / dz   (diferencas finitas 1a ordem)
        L_smooth = alpha_l2 × mean(|d(rho)/dz|^2) + alpha_tv × mean(|d(rho)/dz|)

        alpha_l2 = 0.7 (peso do termo L2, estabilidade)
        alpha_tv = 0.3 (peso do termo L1, preservacao de bordas)

    Diagrama:
      ┌──────────────────────────────────────────────────────────┐
      │  SMOOTHNESS LOSS: TIKHONOV + TOTAL VARIATION              │
      │                                                           │
      │  Perfil rho(z):                                           │
      │    ─────┐                                                 │
      │         │ ← borda real (TV preserva)                      │
      │         └─────────────┐                                   │
      │                       │ ← borda real (TV preserva)        │
      │    ~~~~ oscilacao ~~~~ ← Tikhonov penaliza               │
      │                                                           │
      │  Termo L2 (Tikhonov): suprime oscilacoes espurias        │
      │  Termo L1 (TV): permite transicoes geologicas abruptas    │
      │  Ratio 70/30: prioriza estabilidade, bordas ok se reais   │
      └──────────────────────────────────────────────────────────┘

    Motivacao fisica:
        Em inversao EM 1D, o perfil de resistividade deve ser suave
        dentro das camadas geologicas (rho constante), com transicoes
        abruptas apenas nos limites entre camadas. A regularizacao
        Tikhonov pura (L2) suaviza demais, borrando bordas reais.
        A TV (L1) mantem bordas afiadas. A combinacao 70/30 produz
        perfis geologicamente realistas.

    Args:
        config: PipelineConfig com spacing_meters.
            spacing_meters: Distancia entre pontos de medicao (m).
                Default: 1.0 m. Usado para normalizar d(rho)/dz.

    Returns:
        Callable: Funcao loss(y_true, y_pred) -> tf.Tensor scalar.
            Opera sobre os canais de resistividade (0:2) de y_pred.

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (cenario "smoothness")
            - tests/test_pinns.py: TestSmoothnessLoss
        Ref: Tikhonov (1963) "Solution of incorrectly formulated problems
             and the regularization method" — regularizacao L2.
             Rudin, Osher & Fatemi (1992) "Nonlinear total variation
             based noise removal algorithms" — TV preserva bordas.
        Custo: O(N) por amostra. Cenario PINN mais barato.
    """
    # ── Pesos internos da combinacao L2+TV ────────────────────────────
    # Ratio fixo 70/30: prioriza estabilidade (Tikhonov) sobre
    # preservacao de bordas (TV). Este ratio funciona bem para
    # perfis de resistividade 1D em frequencias LWD tipicas.
    alpha_l2 = 0.7
    alpha_tv = 0.3

    # ── dz: espacamento entre pontos de medicao ──────────────────────
    # Normaliza o gradiente para unidades fisicas (Ohm.m / m em log10).
    # Com SPACING_METERS=1.0 (default), dz=1.0 (no-op).
    dz = config.spacing_meters

    def smoothness_loss(y_true, y_pred):
        """Smoothness loss: Tikhonov (L2) + Total Variation (L1).

        Penaliza oscilacoes espurias no perfil de resistividade
        preservando transicoes geologicas reais via componente TV.
        """
        import tensorflow as tf

        # ── Extrair canais de resistividade (log10 scale) ─────────
        # Canal 0: log10(rho_h), Canal 1: log10(rho_v)
        # Fix CR#2: slice estatico [:2] preserva shape em tf.function.
        rho_log = y_pred[..., :2]  # (B, N, 2) — shape estatica

        # ── Derivada de 1a ordem via diferencas finitas ───────────
        # d(rho)/dz ≈ (rho[i+1] - rho[i]) / dz
        # Shape: (B, N-1, C) — perde ultimo ponto da sequencia.
        # dz normaliza para gradiente em unidades fisicas.
        d_rho = (rho_log[:, 1:, :] - rho_log[:, :-1, :]) / (dz + EPS)

        # ── Termo L2 (Tikhonov): penaliza gradientes ao quadrado ──
        # Suprime oscilacoes espurias em zonas homogeneas.
        # mean(|d(rho)/dz|^2) — custo quadratico, difuso.
        l2_term = tf.reduce_mean(tf.square(d_rho))

        # ── Termo L1 (Total Variation): penaliza gradientes em norma 1
        # Preserva bordas afiadas — penalidade linear permite
        # transicoes abruptas com custo proporcional (nao quadratico).
        # mean(|d(rho)/dz|) — custo linear, esparso.
        tv_term = tf.reduce_mean(tf.abs(d_rho))

        # ── Combinacao: 70% L2 + 30% TV ──────────────────────────
        return alpha_l2 * l2_term + alpha_tv * tv_term

    return smoothness_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CENARIO 5 — SKIN DEPTH LOSS
# ════════════════════════════════════════════════════════════════════════════
# Constrainte fisica baseada no skin depth EM.
# A ferramenta LWD a 20 kHz nao pode resolver features menores que delta.
# Penaliza gradientes de resistividade que excedem 1/delta(z).
#
# Formula:
#   delta(z) = sqrt(2 / (omega × mu × sigma(z)))
#   L_skin = mean(max(0, |d(rho)/dz| - 1/delta(z))^2)
#
# Ref: Ward & Hohmann (1988) "Electromagnetic Theory for Geophysical
#      Applications" — definicao classica de skin depth.
# ──────────────────────────────────────────────────────────────────────────


def make_skin_depth_loss(
    config: "PipelineConfig",
) -> Callable:
    """Factory para Skin Depth Loss — resolucao maxima da ferramenta LWD.

    Penaliza gradientes do perfil de resistividade que excedem o limite
    imposto pelo skin depth EM. A uma frequencia de 20 kHz, a onda EM
    penetra uma distancia finita (skin depth) que depende da condutividade
    local. Features menores que delta nao sao resolvidas pela ferramenta.

    Formula:
        sigma(z) = 10^(-rho_pred(z))   (condutividade em S/m)
        delta(z) = sqrt(2 / (omega × mu × sigma(z)))  (skin depth em m)
        L_skin = mean(max(0, |d(rho)/dz| - 1/delta(z))^2)

    Diagrama:
      ┌──────────────────────────────────────────────────────────┐
      │  SKIN DEPTH CONSTRAINT                                     │
      │                                                           │
      │  Frequencia: f = 20 kHz (default)                         │
      │  omega = 2 × pi × f ≈ 125664 rad/s                       │
      │  mu = 4 × pi × 1e-7 H/m                                  │
      │                                                           │
      │  Exemplos de skin depth:                                   │
      │    rho = 1 Ohm.m   → delta ≈ 3.6 m   → 1/delta ≈ 0.28  │
      │    rho = 10 Ohm.m  → delta ≈ 11.3 m  → 1/delta ≈ 0.09  │
      │    rho = 100 Ohm.m → delta ≈ 35.6 m  → 1/delta ≈ 0.03  │
      │                                                           │
      │  Gradiente |d(rho)/dz| > 1/delta → penalidade ativa      │
      │  Gradiente |d(rho)/dz| <= 1/delta → sem penalidade       │
      │                                                           │
      │  Meios condutivos: delta pequeno → limite apertado        │
      │  Meios resistivos: delta grande → limite frouxo           │
      └──────────────────────────────────────────────────────────┘

    Motivacao fisica:
        O skin depth define a resolucao vertical maxima de uma
        ferramenta EM. Transicoes de resistividade mais abruptas
        que 1/delta sao artefatos da inversao, nao features reais.
        Esta loss atua como regularizacao informada pela fisica,
        adaptada localmente a condutividade do meio.

    Args:
        config: PipelineConfig com:
            frequency_hz: Frequencia da ferramenta LWD (Hz).
                Default: 20000.0 Hz (20 kHz).
            spacing_meters: Distancia entre pontos de medicao (m).
                Default: 1.0 m.

    Returns:
        Callable: Funcao loss(y_true, y_pred) -> tf.Tensor scalar.
            Opera sobre os canais de resistividade (0:2) de y_pred.

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (cenario "skin_depth")
            - tests/test_pinns.py: TestSkinDepthLoss
        Ref: Ward & Hohmann (1988) "Electromagnetic Theory for
             Geophysical Applications" — skin depth delta = sqrt(2/(omega*mu*sigma)).
        Fisica: delta depende de sqrt(rho) — meios resistivos tem skin
            depth maior (mais penetracao, menos resolucao vertical).
    """
    # ── Constantes EM pre-computadas ──────────────────────────────────
    # omega: frequencia angular (rad/s). Default: 2*pi*20000 ≈ 125664 rad/s.
    # mu: permeabilidade magnetica do vacuo (4*pi*1e-7 H/m).
    # omega_mu: produto pre-computado — aparece no denominador de delta^2.
    omega = 2.0 * math.pi * config.frequency_hz
    mu = 4.0 * math.pi * 1e-7
    omega_mu = omega * mu

    # ── dz: espacamento entre pontos de medicao ──────────────────────
    dz = config.spacing_meters

    def skin_depth_loss(y_true, y_pred):
        """Skin depth loss: penaliza gradientes acima da resolucao EM.

        Gradientes maiores que 1/delta(z) indicam features nao
        resolvidas pela ferramenta LWD na frequencia configurada.
        """
        import tensorflow as tf

        # ── Extrair canais rho (log10 scale) ──────────────────────
        # Fix CR#2: slice estatico [:2] preserva shape em tf.function.
        rho_log = y_pred[..., :2]  # (B, N, 2) — shape estatica

        # ── Clamp para evitar overflow: rho em [0.01, 100000] Ohm.m
        # Fix CR#5: usar rho_log_clamped tanto para sigma quanto para
        # o gradiente, garantindo consistencia numerica.
        rho_log_clamped = tf.clip_by_value(rho_log, -2.0, 5.0)

        # ── Condutividade sigma(z) = 10^(-rho_log) ───────────────
        sigma = tf.pow(10.0, -rho_log_clamped)  # (B, N, 2)

        # ── Skin depth: delta(z) = sqrt(2 / (omega × mu × sigma))
        # Em meios condutivos (sigma grande), delta eh pequeno.
        # Em meios resistivos (sigma pequeno), delta eh grande.
        # +EPS evita divisao por zero quando sigma → 0.
        delta = tf.sqrt(2.0 / (omega_mu * sigma + EPS))  # (B, N, 2)

        # ── Limite de gradiente em log10 space ───────────────────
        # Fix CR#1: o gradiente d_rho esta em log10(Ohm.m)/m.
        # Converter 1/delta para log10 space:
        #   d(log10 rho)/dz = (d rho/dz) / (rho × ln(10))
        # Portanto o limiar em log10 space eh:
        #   inv_delta_log = 1 / (delta × rho × ln(10))
        inv_delta = 1.0 / (delta + EPS)  # (B, N, 2)
        rho_linear = tf.pow(10.0, rho_log_clamped)  # rho em Ohm.m
        inv_delta_log = inv_delta / (rho_linear * _LN10 + EPS)  # (B, N, 2)

        # ── Derivada de 1a ordem |d(log10 rho)/dz| ───────────────
        # Fix CR#5: usar rho_log_clamped para consistencia.
        d_rho = tf.abs(
            (rho_log_clamped[:, 1:, :] - rho_log_clamped[:, :-1, :]) / (dz + EPS)
        )  # (B, N-1, 2)

        # ── Alinhar inv_delta_log com d_rho (midpoints) ──────────
        # inv_delta_log tem shape (B, N, 2), d_rho tem (B, N-1, 2).
        inv_delta_mid = 0.5 * (
            inv_delta_log[:, 1:, :] + inv_delta_log[:, :-1, :]
        )  # (B, N-1, 2)

        # ── Excesso: max(0, |d(log10 rho)/dz| - limiar_log) ─────
        # Penaliza apenas gradientes que EXCEDEM o limite fisico
        # em log10 space. Gradientes dentro do skin depth sao
        # permitidos (sem custo).
        excess = tf.nn.relu(d_rho - inv_delta_mid)

        # ── Loss: MSE do excesso ──────────────────────────────────
        # Penalidade quadratica: grandes violacoes sao fortemente
        # penalizadas, pequenas violacoes quase ignoradas.
        return tf.reduce_mean(tf.square(excess))

    return skin_depth_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CENARIO 6 — CONTINUITY LOSS (L1 SPATIAL SMOOTHNESS)
# ════════════════════════════════════════════════════════════════════════════
# Regularizacao L1 pura do gradiente espacial de resistividade.
# Diferente do smoothness (L2+TV), usa apenas norma L1 que produz
# gradientes esparsos — mantendo o perfil constante com transicoes
# afiadas apenas onde geologicamente justificado.
#
# Formula:
#   L_cont = mean(|d(rho_pred)/dz|)
#
# Projetado para modo realtime (geosteering causal): predicoes devem
# variar lentamente ao longo da trajetoria do poco.
#
# Ref: Tibshirani (1996) — Lasso (L1 regularization) produz
#      solucoes esparsas.
# ──────────────────────────────────────────────────────────────────────────


def make_continuity_loss(
    config: "PipelineConfig",
) -> Callable:
    """Factory para Continuity Loss — L1 spatial smoothness esparsa.

    Enforces que predicoes de resistividade variem lentamente ao longo
    da trajetoria do poco (dimensao z). Usa norma L1 pura, que produz
    gradientes esparsos — o perfil tende a ser constante por trechos
    com transicoes afiadas apenas nos limites de camada.

    Formula:
        d(rho)/dz = (rho_pred[i+1] - rho_pred[i]) / dz
        L_cont = mean(|d(rho)/dz|)

    Diagrama:
      ┌──────────────────────────────────────────────────────────┐
      │  CONTINUITY LOSS: L1 SPATIAL SMOOTHNESS                    │
      │                                                           │
      │  Comparacao com smoothness (cenario 4):                   │
      │                                                           │
      │  smoothness: alpha_l2 × ||grad||^2 + alpha_tv × ||grad|| │
      │  continuity: ||grad|| (L1 puro)                           │
      │                                                           │
      │  L1 puro → gradientes ESPARSOS:                           │
      │    ────────┐                                              │
      │            │ ← unica transicao permitida (custo linear)   │
      │            └──────── (constante)                          │
      │                                                           │
      │  L2 → gradientes SUAVES:                                  │
      │    ───────╲                                               │
      │           ╲── transicao gradual (custo quadratico)       │
      │                                                           │
      │  Uso: realtime geosteering (modo causal, sliding window)  │
      └──────────────────────────────────────────────────────────┘

    Motivacao fisica:
        No geosteering em tempo real, o perfil de resistividade ao
        longo do poco nao deve oscilar rapidamente — as formacoes
        geologicas sao lateralmente continuas por dezenas de metros.
        A regularizacao L1 penaliza QUALQUER variacao, mas com custo
        linear (nao quadratico), permitindo transicoes reais sem
        custo excessivo. Ideal para sliding window causal.

    Args:
        config: PipelineConfig com:
            spacing_meters: Distancia entre pontos de medicao (m).
                Default: 1.0 m.

    Returns:
        Callable: Funcao loss(y_true, y_pred) -> tf.Tensor scalar.
            Opera sobre os canais de resistividade (0:2) de y_pred.

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (cenario "continuity")
            - tests/test_pinns.py: TestContinuityLoss
        Ref: Tibshirani (1996) "Regression Shrinkage and Selection via
             the Lasso" — L1 regularization produz solucoes esparsas.
        Diferenca chave do smoothness: L1 puro (sem componente L2).
            Produz perfis "blocky" (constantes por trechos) que sao
            mais realistas geologicamente que perfis suavizados.
    """
    # ── dz: espacamento entre pontos de medicao ──────────────────────
    dz = config.spacing_meters

    def continuity_loss(y_true, y_pred):
        """Continuity loss: L1 pura do gradiente espacial.

        Produz perfis de resistividade esparsos (constantes por trechos)
        com transicoes afiadas apenas onde geologicamente justificado.
        """
        import tensorflow as tf

        # ── Extrair canais de resistividade (log10 scale) ─────────
        # Fix CR#2: slice estatico [:2] preserva shape em tf.function.
        rho_log = y_pred[..., :2]  # (B, N, 2) — shape estatica

        # ── Derivada de 1a ordem via diferencas finitas ───────────
        # d(rho)/dz = (rho[i+1] - rho[i]) / dz
        # Norma L1: penaliza magnitude do gradiente linearmente.
        # Isso favorece perfis constantes por trechos (sparsity).
        d_rho = (rho_log[:, 1:, :] - rho_log[:, :-1, :]) / (dz + EPS)

        # ── Loss: MAE do gradiente (L1 pura) ─────────────────────
        # Custo linear: transicoes reais pagam custo proporcional,
        # NAO quadratico (como Tikhonov). Isso preserva bordas
        # geofisicas reais enquanto suprime oscilacoes.
        return tf.reduce_mean(tf.abs(d_rho))

    return continuity_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CENARIO 7 — VARIATIONAL LOSS (DEEP RITZ METHOD)
# ════════════════════════════════════════════════════════════════════════════
# Forma fraca da equacao de Helmholtz: minimiza funcional de energia.
# Usa apenas derivada de 1a ordem (nao 2a), ~40% mais leve que maxwell.
# Mais estavel numericamente que a forma forte (PDE residual).
#
# Formula:
#   k^2(z) = omega × mu × sigma(z) = omega × mu × 10^(-rho_pred(z))
#   L_var = mean(|d(rho)/dz|^2 / (1 + k^2(z)))
#
# Ref: Weinan & Yu (2018) "The Deep Ritz Method" — formulacao
#      variacional de PDEs com redes neurais.
# ──────────────────────────────────────────────────────────────────────────


def make_variational_loss(
    config: "PipelineConfig",
) -> Callable:
    """Factory para Variational Loss — forma fraca de Helmholtz (Deep Ritz).

    Minimiza o funcional de energia associado a equacao de Helmholtz 1D
    usando apenas derivadas de 1a ordem. Diferente do maxwell (cenario 3)
    que usa derivada de 2a ordem (forma forte), a formulacao variacional
    eh ~40% mais leve computacionalmente e mais estavel numericamente.

    Formula:
        sigma(z) = 10^(-rho_pred(z))   (condutividade)
        k^2(z) = omega × mu × sigma(z) (numero de onda ao quadrado)
        L_var = mean(|d(rho)/dz|^2 / (1 + k^2(z)))

    Diagrama:
      ┌──────────────────────────────────────────────────────────┐
      │  VARIATIONAL LOSS: FORMA FRACA DE HELMHOLTZ               │
      │                                                           │
      │  Comparacao maxwell vs variational:                       │
      │                                                           │
      │  maxwell (forma forte):                                   │
      │    Residuo = d^2(rho)/dz^2 / (1 + k^2)                   │
      │    L = mean(R^2)                                          │
      │    Custo: derivada de 2a ordem (3 pontos por stencil)     │
      │    Perde 2 pontos nas bordas                              │
      │                                                           │
      │  variational (forma fraca):                               │
      │    Energia = |d(rho)/dz|^2 / (1 + k^2)                   │
      │    L = mean(E)                                            │
      │    Custo: derivada de 1a ordem (2 pontos por stencil)     │
      │    Perde 1 ponto nas bordas                               │
      │    ~40% mais leve que maxwell                              │
      │                                                           │
      │  Normalizacao (1 + k^2):                                  │
      │    k^2 grande (condutivo) → denominador grande            │
      │      → menos penalidade (curvatura permitida)             │
      │    k^2 pequeno (resistivo) → denominador ≈ 1              │
      │      → penalidade total (suavidade exigida)               │
      └──────────────────────────────────────────────────────────┘

    Motivacao fisica:
        A formulacao variacional (fraca) de uma PDE minimiza o funcional
        de energia em vez de impor o residuo pontual (forma forte).
        Para a equacao de Helmholtz, o funcional eh proporcional ao
        quadrado do gradiente normalizado por k^2. Vantagens:
        - Nao requer derivada de 2a ordem (mais estavel)
        - Admite descontinuidades fracas (L^2 vs H^2)
        - Converge mais rápido em redes profundas (Weinan & Yu 2018)

    Args:
        config: PipelineConfig com:
            frequency_hz: Frequencia da ferramenta LWD (Hz).
                Default: 20000.0 Hz. omega = 2*pi*f.
            spacing_meters: Distancia entre pontos de medicao (m).
                Default: 1.0 m.

    Returns:
        Callable: Funcao loss(y_true, y_pred) -> tf.Tensor scalar.

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (cenario "variational")
            - tests/test_pinns.py: TestVariationalLoss
        Ref: Weinan & Yu (2018) "The Deep Ritz Method: A Deep Learning-
             Based Numerical Method for Solving Variational Problems"
             — formulacao variacional de PDEs com redes neurais.
        Custo: ~60% do maxwell (1a derivada vs 2a derivada).
        Estabilidade: melhor que maxwell em perfis com descontinuidades.
    """
    # ── Constantes EM pre-computadas ──────────────────────────────────
    # omega_mu: produto pre-computado para k^2 = omega*mu*sigma.
    omega = 2.0 * math.pi * config.frequency_hz
    mu = 4.0 * math.pi * 1e-7
    omega_mu = omega * mu

    # ── dz: espacamento entre pontos de medicao ──────────────────────
    dz = config.spacing_meters

    def variational_loss(y_true, y_pred):
        """Variational loss: funcional de energia (forma fraca Helmholtz).

        Minimiza |d(rho)/dz|^2 / (1 + k^2) — derivada de 1a ordem
        normalizada pela condutividade local.
        """
        import tensorflow as tf

        # ── Extrair canais rho (log10 scale) ──────────────────────
        # Fix CR#2: slice estatico [:2] preserva shape em tf.function.
        rho_log = y_pred[..., :2]  # (B, N, 2) — shape estatica

        # ── Condutividade sigma(z) = 10^(-rho_log) ───────────────
        # Clamp para evitar overflow: rho em [0.01, 100000] Ohm.m
        rho_log_clamped = tf.clip_by_value(rho_log, -2.0, 5.0)
        sigma = tf.pow(10.0, -rho_log_clamped)  # (B, N, C)

        # ── k^2(z) = omega × mu × sigma (numero de onda) ─────────
        # k^2 modula a penalidade: meios condutivos (k^2 grande)
        # permitem mais gradiente, meios resistivos (k^2 pequeno)
        # exigem suavidade.
        k_sq = omega_mu * sigma  # (B, N, C)

        # ── Derivada de 1a ordem d(rho)/dz ────────────────────────
        # Fix CR#5: usar rho_log_clamped para consistencia com k_sq.
        d_rho = (rho_log_clamped[:, 1:, :] - rho_log_clamped[:, :-1, :]) / (dz + EPS)
        # (B, N-1, C)

        # ── Alinhar k^2 com d_rho (midpoints) ────────────────────
        # k_sq tem shape (B, N, C), d_rho tem (B, N-1, C).
        # Media dos pontos adjacentes para alinhar indices.
        k_sq_mid = 0.5 * (k_sq[:, 1:, :] + k_sq[:, :-1, :])
        # (B, N-1, C)

        # ── Funcional de energia: |grad|^2 / (1 + k^2) ──────────
        # Numerador: quadrado do gradiente (energia cinetica).
        # Denominador: 1 + k^2 (normalizacao por condutividade).
        # +1 evita divisao por zero; +EPS protecao extra float32.
        energy = tf.square(d_rho) / (1.0 + k_sq_mid + EPS)

        # ── Loss: media do funcional de energia ───────────────────
        return tf.reduce_mean(energy)

    return variational_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO: CENARIO 8 — SELF-ADAPTIVE LOSS (ATTENTION POR GRADIENTE)
# ════════════════════════════════════════════════════════════════════════════
# Residuo PDE com pesos adaptativos derivados do proprio gradiente.
# Regioes com alto gradiente (bordas de camada) recebem mais atencao.
# Nao requer variaveis treinaveis extras — a atencao eh self-derived.
#
# Formula:
#   R(z) = d^2(rho)/dz^2 / (1 + k^2(z))  (residuo maxwell)
#   w(z) = softplus(|d(rho)/dz|)           (peso por gradiente)
#   L_adaptive = mean(w(z) × R(z)^2) / mean(w(z))
#
# Ref: McClenny & Braga-Neto (2023) "Self-Adaptive PINNs" —
#      pesos adaptativos para concentrar esforco nas regioes dificeis.
# ──────────────────────────────────────────────────────────────────────────


def make_self_adaptive_loss(
    config: "PipelineConfig",
) -> Callable:
    """Factory para Self-Adaptive Loss — residuo PDE com atencao por gradiente.

    Computa o residuo de Maxwell (como cenario 3) mas aplica pesos
    adaptativos derivados do gradiente local de resistividade. Regioes
    com alto gradiente (limites de camada) recebem mais atencao,
    concentrando o esforco de regularizacao nas zonas mais dificeis.

    Formula:
        R(z) = d^2(rho)/dz^2 / (1 + k^2(z))   (residuo Helmholtz)
        w(z) = softplus(|d(rho)/dz|)             (peso adaptativo)
        L_adaptive = mean(w(z) × R(z)^2) / mean(w(z))

    Diagrama:
      ┌──────────────────────────────────────────────────────────┐
      │  SELF-ADAPTIVE LOSS: ATTENTION POR GRADIENTE              │
      │                                                           │
      │  Perfil rho(z):                                           │
      │    ─────┐         ┌──────                                 │
      │         │         │                                       │
      │         └─────────┘                                       │
      │    w(z): 0.7   5.2       4.8   0.7                       │
      │         (baixo) (alto)   (alto) (baixo)                   │
      │                                                           │
      │  Zonas homogeneas (grad ≈ 0):                             │
      │    w = softplus(0) ≈ 0.69 → peso base (baixa atencao)    │
      │                                                           │
      │  Limites de camada (grad alto):                           │
      │    w = softplus(5) ≈ 5.0 → peso alto (alta atencao)      │
      │                                                           │
      │  Normalizacao: L / mean(w) → escala invariante            │
      │  Sem variaveis treinaveis extras (self-derived)           │
      └──────────────────────────────────────────────────────────┘

    Motivacao fisica:
        O residuo de Maxwell eh mais critico nos limites de camada
        (transicoes de resistividade) do que em zonas homogeneas.
        Sem pesos adaptativos, a loss media trata todos os pontos
        igualmente, diluindo o sinal das bordas. Com self-attention,
        a rede concentra esforco nas regioes onde erros fisicos sao
        mais provaveis e mais impactantes para geosteering.

    Args:
        config: PipelineConfig com:
            frequency_hz: Frequencia da ferramenta LWD (Hz).
                Default: 20000.0 Hz.
            spacing_meters: Distancia entre pontos de medicao (m).
                Default: 1.0 m.

    Returns:
        Callable: Funcao loss(y_true, y_pred) -> tf.Tensor scalar.

    Note:
        Referenciado em:
            - losses/pinns.py: build_pinns_loss() (cenario "self_adaptive")
            - tests/test_pinns.py: TestSelfAdaptiveLoss
        Ref: McClenny & Braga-Neto (2023) "Self-Adaptive Physics-Informed
             Neural Networks" — pesos adaptativos para PINNs, concentrando
             esforco computacional nas regioes de maior residuo.
        Custo: ~120% do maxwell (derivada de 2a ordem + softplus weights).
        Sem variaveis treinaveis extras — atencao derivada do gradiente.
    """
    # ── Constantes EM pre-computadas ──────────────────────────────────
    omega = 2.0 * math.pi * config.frequency_hz
    mu = 4.0 * math.pi * 1e-7
    omega_mu = omega * mu

    # ── dz e dz^2: espacamento e seu quadrado ─────────────────────────
    dz = config.spacing_meters
    dz_sq = dz**2

    def self_adaptive_loss(y_true, y_pred):
        """Self-adaptive loss: residuo PDE ponderado por gradiente local.

        Combina residuo de Helmholtz (derivada de 2a ordem) com
        pesos softplus derivados do gradiente de 1a ordem,
        concentrando atencao nos limites de camada.
        """
        import tensorflow as tf

        # ── Extrair canais rho (log10 scale) ──────────────────────
        # Fix CR#2: slice estatico [:2] preserva shape em tf.function.
        rho_log = y_pred[..., :2]  # (B, N, 2) — shape estatica

        # ── Fix CR#4: guard para seq_len < 3 (d2 requer 3+ pontos) ─
        seq_len = tf.shape(rho_log)[1]
        if_short = tf.less(seq_len, 3)
        if tf.executing_eagerly():
            if if_short:
                return tf.constant(0.0, dtype=tf.float32)

        # ── Condutividade sigma(z) = 10^(-rho_log) ───────────────
        rho_log_clamped = tf.clip_by_value(rho_log, -2.0, 5.0)
        sigma = tf.pow(10.0, -rho_log_clamped)  # (B, N, C)

        # ── k^2(z) = omega × mu × sigma ──────────────────────────
        k_sq = omega_mu * sigma  # (B, N, C)

        # ── Derivada de 2a ordem d^2(rho)/dz^2 (pontos interiores)
        # Fix CR#5: usar rho_log_clamped para consistencia com k_sq.
        # Stencil central: (f[i+1] - 2f[i] + f[i-1]) / dz^2
        # Shape resultante: (B, N-2, C)
        d2_rho = (
            rho_log_clamped[:, 2:, :]
            - 2.0 * rho_log_clamped[:, 1:-1, :]
            + rho_log_clamped[:, :-2, :]
        ) / (dz_sq + EPS)

        # ── Normalizar residuo por (1 + k^2) nos pontos interiores
        k_sq_interior = k_sq[:, 1:-1, :]
        residual = d2_rho / (1.0 + k_sq_interior + EPS)
        residual_sq = tf.square(residual)  # (B, N-2, C)

        # ── Derivada de 1a ordem |d(rho)/dz| (para pesos) ────────
        # Fix CR#5: usar rho_log_clamped para consistencia.
        d1_rho = tf.abs(
            (rho_log_clamped[:, 1:, :] - rho_log_clamped[:, :-1, :]) / (dz + EPS)
        )

        # ── Alinhar d1_rho com pontos interiores (N-2) ───────────
        # Media dos gradientes adjacentes para ter shape (B, N-2, C).
        # d1_rho[i] e d1_rho[i+1] cobrem o ponto interior [i+1].
        d1_rho_interior = 0.5 * (d1_rho[:, 1:, :] + d1_rho[:, :-1, :])
        # (B, N-2, C)

        # ── Pesos adaptativos: w(z) = softplus(|grad|) ───────────
        # softplus(x) = log(1 + exp(x)):
        #   - x ≈ 0 (zona homogenea): w ≈ 0.69 (peso base)
        #   - x >> 0 (borda de camada): w ≈ x (peso alto)
        # Suave e diferenciavel em todo o dominio.
        weights = tf.math.softplus(d1_rho_interior)  # (B, N-2, C)

        # ── Loss ponderada: mean(w × R^2) / mean(w) ──────────────
        # Numerador: residuo quadratico ponderado pela atencao.
        # Denominador: normalizacao para escala invariante.
        # Sem denominador, a loss cresceria com o gradiente medio,
        # criando instabilidade numerica.
        weighted_residual = tf.reduce_mean(weights * residual_sq)
        normalizer = tf.reduce_mean(weights) + EPS

        return weighted_residual / normalizer

    return self_adaptive_loss


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FACTORY CENTRAL — build_pinns_loss()
# ════════════════════════════════════════════════════════════════════════════
# Constroi a loss PINN completa combinando:
#   1. Cenario (oracle/surrogate/maxwell/smoothness/skin_depth/
#      continuity/variational/self_adaptive) com lambda schedule
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
        elif scenario == "smoothness":
            scenario_fn = make_smoothness_loss(config)
        elif scenario == "skin_depth":
            scenario_fn = make_skin_depth_loss(config)
        elif scenario == "continuity":
            scenario_fn = make_continuity_loss(config)
        elif scenario == "variational":
            scenario_fn = make_variational_loss(config)
        elif scenario == "self_adaptive":
            scenario_fn = make_self_adaptive_loss(config)
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
