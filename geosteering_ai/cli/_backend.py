# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/_backend.py                                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Resolução do backend de simulação da CLI (numba ⇄ jax)     ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.53 — backend JAX GPU selecionável)      ║
# ║  Versão      : v2.53                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-02                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : stdlib + dispatch._jax_gpu_available (auto-detect GPU)      ║
# ║  Dependências: geosteering_ai.simulation.dispatch (lazy, p/ detect GPU)    ║
# ║  Padrão      : função pura testável (mock de _jax_gpu_available)           ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Mapeia o backend SOLICITADO pelo usuário (``--backend numba|jax``)     ║
# ║    para o backend EFETIVO + device, com fallback gracioso quando o        ║
# ║    usuário pede ``jax`` mas não há GPU JAX visível (CI/CPU): emite aviso   ║
# ║    e cai para ``numba``/``cpu`` (o caminho rápido com pool de workers).   ║
# ║                                                                           ║
# ║  ÁRVORE DE RESOLUÇÃO                                                      ║
# ║    ┌────────────────────────────────────────────────────────────────┐    ║
# ║    │  numba              → (numba, cpu)                              │    ║
# ║    │  jax  + GPU JAX     → (jax,   gpu)                              │    ║
# ║    │  jax  sem GPU JAX   → warn + (numba, cpu)   [fallback gracioso] │    ║
# ║    └────────────────────────────────────────────────────────────────┘    ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    resolve_backend: (requested) → (backend_efetivo, device)              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Resolução do backend de simulação da CLI (Sprint v2.53).

:func:`resolve_backend` traduz a escolha do usuário (``numba`` default ou
``jax`` GPU) no backend efetivo + device, com fallback gracioso para Numba
CPU quando o JAX não enxerga uma GPU. Pura e testável (mocka-se
``_jax_gpu_available``).
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

__all__ = ["resolve_backend"]


def resolve_backend(requested: str, *, quiet: bool = False) -> Tuple[str, str]:
    """Resolve o backend efetivo + device a partir do solicitado.

    Args:
        requested: ``"numba"`` (padrão da CLI, CPU) ou ``"jax"`` (GPU). Outros
            valores levantam ``ValueError`` (argparse já restringe via
            ``choices``, mas a função valida defensivamente).
        quiet: Se ``True``, suprime o ``logger.warning`` do fallback (mantém
            a resolução). Default ``False``.

    Returns:
        Tupla ``(backend_efetivo, device)``:
          - ``("numba", "cpu")`` — sempre, para ``requested="numba"``;
          - ``("jax", "gpu")`` — ``requested="jax"`` com GPU JAX visível;
          - ``("numba", "cpu")`` — ``requested="jax"`` SEM GPU (fallback + warn).

    Raises:
        ValueError: ``requested`` não é ``"numba"`` nem ``"jax"``.

    Note:
        A detecção de GPU reusa
        :func:`geosteering_ai.simulation.dispatch._jax_gpu_available`
        (lazy ``import jax``; ``False`` em qualquer falha → rota Numba segura).
        Um futuro ``--strict-backend`` (proposto no relatório v2.53) poderá
        transformar o fallback em erro fatal; hoje o default é gracioso para
        manter a CLI usável em CI/CPU.

    Example:
        >>> resolve_backend("numba")
        ('numba', 'cpu')
    """
    req = (requested or "").strip().lower()
    if req == "numba":
        return "numba", "cpu"
    if req != "jax":
        raise ValueError(f"backend inválido: {requested!r}. Use 'numba' ou 'jax'.")

    # ── requested == "jax" — verifica GPU JAX ────────────────────────────
    # Import lazy do dispatch: mantém ``--help`` rápido e permite mock em teste.
    from geosteering_ai.simulation.dispatch import _jax_gpu_available

    if _jax_gpu_available():
        return "jax", "gpu"

    # Sem GPU JAX → fallback gracioso para o caminho Numba CPU (pool de workers).
    if not quiet:
        logger.warning(
            "backend=jax solicitado, mas nenhuma GPU JAX foi detectada — "
            "usando 'numba' (CPU). Para execução JAX em GPU, ative o ambiente "
            "CUDA (conda Geosteering_AI + jax[cuda12])."
        )
    return "numba", "cpu"
