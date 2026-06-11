# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/_jax/__init__.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Backend JAX (Sprint 3.1)               ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-12                                                 ║
# ║  Status      : Em construção (Sprint 3.1 — fundação CPU)                 ║
# ║  Framework   : JAX 0.4.30+                                                ║
# ║  Dependências: jax, jaxlib (opcional — dual-mode)                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Backend JAX (CPU/GPU/TPU) do simulador Python. Na Sprint 3.1          ║
# ║    implementa os módulos fundamentais (hankel, rotation) que não têm     ║
# ║    recursões complexas. A paridade numérica com o backend Numba é        ║
# ║    validada em tolerância < 1e-12 (float64) via                          ║
# ║    `validation/compare_backends.py`.                                      ║
# ║                                                                           ║
# ║  DUAL-MODE                                                                ║
# ║    Exporta `HAS_JAX: Final[bool]`. Quando JAX não está instalado,       ║
# ║    `HAS_JAX=False` e as funções ``integrate_j0/j1`` levantam           ║
# ║    :class:`ImportError` ao serem chamadas — diferente do backend         ║
# ║    Numba (que tem fallback no-op), JAX é inerentemente substituto        ║
# ║    do Numba (não complementar). Se o usuário pediu JAX, deve instalar.   ║
# ║                                                                           ║
# ║  float64 OBRIGATÓRIO                                                      ║
# ║    Chamamos `jax.config.update("jax_enable_x64", True)` no import do     ║
# ║    submódulo para garantir paridade com Numba. JAX default é float32    ║
# ║    para compatibilidade com TPU, mas o simulador EM exige complex128.   ║
# ║                                                                           ║
# ║  MÓDULOS (Sprint 3.1)                                                    ║
# ║    • hankel.py    — integrate_j0/j1 via jnp.einsum + @jax.jit           ║
# ║    • rotation.py  — build_rotation_matrix + rotate_tensor diferenciáveis║
# ║                                                                           ║
# ║  MÓDULOS FUTUROS (Sprints 3.2-3.4)                                      ║
# ║    • propagation.py — common_arrays + common_factors com jax.lax.scan   ║
# ║    • dipoles.py     — hmd_tiv + vmd com @jit(static_argnames=...)       ║
# ║    • kernel.py      — fields_in_freqs_jax (orquestrador)                 ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • docs/reference/plano_simulador_python_jax_numba.md §4-6            ║
# ║    • .claude/commands/geosteering-simulator-python.md (Seção 3.1)       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Backend JAX do simulador Python — fundação CPU (Sprint 3.1).

Este subpacote implementa as funções fundamentais do simulador forward
EM 1D TIV usando JAX como backend numérico. JAX proporciona:

  - **Auto-diferenciação** via ``jax.grad`` / ``jax.jacfwd`` — essencial
    para treino de PINNs com gradientes ∂H/∂ρ.
  - **Aceleração XLA** em CPU/GPU/TPU sem reescrita de código.
  - **Vetorização automática** via ``jax.vmap`` — substitui loops de
    posições/ângulos por operações matriciais contíguas (sem GIL).

Example:
    Carregamento do filtro Hankel e cálculo de uma integral J₀::

        >>> import jax.numpy as jnp
        >>> from geosteering_ai.simulation.filters import FilterLoader
        >>> from geosteering_ai.simulation._jax import integrate_j0
        >>> filt = FilterLoader().load("werthmuller_201pt")
        >>> # Valores da função a integrar em cada kr
        >>> values = jnp.ones(filt.npt, dtype=jnp.complex128)
        >>> result = integrate_j0(values, jnp.asarray(filt.weights_j0))

Note:
    Paridade com Numba é de < 1e-12 em float64 para todos os módulos
    da Sprint 3.1 (hankel, rotation). Diferenças residuais vêm do
    reordenamento XLA das operações de ponto flutuante.
"""

import logging as _logging
import os as _os
from pathlib import Path as _Path
from typing import Final

_logger = _logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sprint O1 (v2.43) — Setup de ambiente XLA (Quick Win #3 + #4)
# ──────────────────────────────────────────────────────────────────────────────
# DEVE rodar ANTES do primeiro ``import jax`` para que XLA capture as flags
# durante a inicialização do jaxlib. Como este ``__init__.py`` é executado
# antes de qualquer submódulo ``_jax/xxx.py`` (que importa ``jax``), este é
# o ponto correto desde que nenhum outro módulo do projeto importe ``jax``
# antes — auditado em 2026-05-24 (commit Sprint O1): nenhum ``import jax``
# fora de ``geosteering_ai/simulation/_jax/``.
#
# Quick Win #3 — Persistent XLA Compilation Cache
#   Sem cache persistente, cada sessão (Colab, notebook, pytest) recompila
#   os JITs (cold-start 30–180 s). Com cache persistente em disco, sessões
#   subsequentes reutilizam o XLA HLO compilado (~3 s).
#
# Quick Win #4 — XLA flags otimizadas para A100
#   * latency_hiding_scheduler  → melhor escalonamento de kernels CUDA.
#   * triton_softmax_fusion     → fusão de softmax (sem impacto em EM 1D, mas
#                                  defensivo para sub-rotinas futuras).
#   * XLA_PYTHON_CLIENT_PREALLOCATE=false → evita alocação eager de 75%
#     da VRAM (BFC allocator), só ativado em GPU (não em CPU dev).
#   * XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 → deixa 15% para CUDA runtime
#     e cuDNN handles.
#
# Backward compatibility
#   * Opt-out via ``GEOSTEERING_JAX_NO_XLA_SETUP=1`` (preserva env vars
#     pré-existentes do usuário sem modificação).
#   * Cache dir: criado se não existir; em caso de OSError, faz log
#     warning e segue sem cache (degradação graciosa).
#   * Em CPU-only (macOS dev), env vars de GPU não causam efeito —
#     XLA CPU backend ignora-as silenciosamente.
# ──────────────────────────────────────────────────────────────────────────────


def _setup_xla_environment() -> None:
    """Configura env vars XLA + cache de compilação persistente.

    Idempotente: usa ``os.environ.setdefault`` para preservar overrides
    do usuário. Gracioso: cache dir criado on-demand; falhas só geram
    warning. Pode ser desabilitado globalmente via env var
    ``GEOSTEERING_JAX_NO_XLA_SETUP=1``.
    """
    if _os.environ.get("GEOSTEERING_JAX_NO_XLA_SETUP", "0") == "1":
        _logger.debug("XLA setup desabilitado via GEOSTEERING_JAX_NO_XLA_SETUP=1")
        return

    # ── Detecção GPU pré-JAX (via nvidia-smi) ─────────────────────────────
    # Patch O1-fix-2 (v2.43.1): flags A100-tuned causaram regressão em T4
    # nas cenários A/B (-11.8%/-15.7%). Detecção via nvidia-smi (subprocess
    # 2s timeout) decide se aplica flags otimistas. Falha gracioso → assume
    # não-A100 (conservador, sem regressão garantida).
    _is_a100_class = False
    try:
        import subprocess as _sp

        _gpu_name = _sp.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.lower()
        _is_a100_class = "a100" in _gpu_name or "h100" in _gpu_name
    except (FileNotFoundError, _sp.TimeoutExpired, _sp.SubprocessError, OSError):
        # nvidia-smi ausente (CPU dev) ou timeout — assume não-A100
        pass

    # ── XLA flags (combina com pré-existentes, sem duplicar) ──────────────
    # xla_gpu_enable_triton_softmax_fusion foi removida em JAX ≥ 0.5.x (XLA 0.7+).
    # Setar flag desconhecida causa FATAL em parse_flags_from_env.cc — mata o kernel.
    existing_flags = _os.environ.get("XLA_FLAGS", "")
    candidate_flags: list[str] = []
    if _is_a100_class:
        # latency_hiding_scheduler é tunado para A100/H100 (deep async copy
        # engines, 108 SMs). Em T4 adiciona compile-time e reordena kernels
        # pequenos de forma que prejudica cenário A (launch-bound).
        candidate_flags.append("--xla_gpu_enable_latency_hiding_scheduler=true")
    new_flag_str = " ".join(
        [existing_flags] + [f for f in candidate_flags if f not in existing_flags]
    ).strip()
    if new_flag_str:
        _os.environ["XLA_FLAGS"] = new_flag_str

    # ── VRAM allocation (só relevante em GPU; CPU XLA ignora) ─────────────
    # setdefault preserva qualquer override explícito do usuário.
    # MEM_FRACTION: A100 (40 GB) suporta 0.85 confortavelmente. T4 (15 GB)
    # com 0.85 = 12.75 GB; somado a cuDNN handles + cache HLO acumulado,
    # fragmenta o BFC allocator e causa OOM em cenário E (5.6 GiB alloc).
    # 0.75 = 11.25 GB em T4 deixa ~3.75 GB de folga para drivers.
    _os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85" if _is_a100_class else "0.75"
    )

    # ── Compilation cache dir (PERSISTENTE — sobrevive reboot) ────────────
    # Recuperado do WIP v2.51 (warmup): default ESTÁVEL em ~/.cache (sobrevive
    # reboot → cache-hits XLA cross-run REAIS), em vez de /tmp (efêmero). Prioridade:
    #   1) override explícito JAX_COMPILATION_CACHE_DIR (respeitado sempre);
    #   2) ~/.cache/geosteering/jax_compilation_cache (estável);
    #   3) fallback $TMPDIR/jax_compilation_cache_geosteering (efêmero, mas
    #      válido na sessão) se o home não for gravável (CI/containers).
    cache_dir = _os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir:
        # Override do usuário: cria on-demand, gracioso.
        try:
            _Path(cache_dir).mkdir(parents=True, exist_ok=True)
            _logger.info("JAX compilation cache (override): %s", cache_dir)
        except OSError as exc:
            _logger.warning(
                "Falha ao criar JAX cache dir %s: %s — sem cache persistente",
                cache_dir,
                exc,
            )
    else:
        _stable = _Path.home() / ".cache" / "geosteering" / "jax_compilation_cache"
        _tmp = (
            _Path(_os.environ.get("TMPDIR", "/tmp"))
            / "jax_compilation_cache_geosteering"
        )
        for _candidate in (_stable, _tmp):
            try:
                _candidate.mkdir(parents=True, exist_ok=True)
                cache_dir = str(_candidate)
                _os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
                _logger.info("JAX compilation cache configurado: %s", cache_dir)
                break
            except OSError as exc:
                _logger.warning(
                    "JAX cache dir %s não-gravável (%s) — tentando fallback",
                    _candidate,
                    exc,
                )
        if not cache_dir:
            _logger.warning(
                "Nenhum JAX cache dir gravável — seguindo sem cache persistente"
            )


# Executa setup ANTES do primeiro ``import jax`` abaixo.
_setup_xla_environment()


# ──────────────────────────────────────────────────────────────────────────────
# Detecção dual-mode
# ──────────────────────────────────────────────────────────────────────────────
try:
    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # noqa: F401

    # Habilita float64 globalmente — CRÍTICO para paridade com Numba.
    # JAX default é float32 para TPU; o simulador EM requer complex128.
    jax.config.update("jax_enable_x64", True)

    # Sprint O1 — Persistent compilation cache via jax.config (API moderna,
    # JAX ≥ 0.4.25). Algumas versões antigas não têm a chave; degrada
    # gracioso. A env var JAX_COMPILATION_CACHE_DIR (setada acima) também
    # é honrada por jaxlib internamente.
    _cache_dir = _os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if _cache_dir:
        try:
            jax.config.update("jax_compilation_cache_dir", _cache_dir)
            _logger.debug(
                "jax.config.jax_compilation_cache_dir aplicado: %s", _cache_dir
            )
        except (AttributeError, KeyError, ValueError) as _exc:
            _logger.warning(
                "jax.config persistent cache não suportado nesta versão (%s): %s",
                getattr(jax, "__version__", "?"),
                _exc,
            )

    # Recuperado do WIP v2.51 (warmup) — persiste TODO compile (mesmo rápido) no
    # cache de disco. Default do JAX é ~1.0s → compiles de bucket abaixo do limiar
    # NÃO persistiam, anulando o cache cross-run. 0.0 = persiste tudo. Version-guarded
    # (chave ausente em versões antigas → degrada gracioso, como o bloco acima).
    try:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        _logger.debug("jax_persistent_cache_min_compile_time_secs=0.0 aplicado")
    except (AttributeError, KeyError, ValueError) as _exc:
        _logger.warning(
            "jax_persistent_cache_min_compile_time_secs não suportado (%s): %s",
            getattr(jax, "__version__", "?"),
            _exc,
        )

    HAS_JAX: Final[bool] = True
except ImportError:
    HAS_JAX = False  # type: ignore[no-redef,misc]


# ──────────────────────────────────────────────────────────────────────────────
# Re-exports públicos
# ──────────────────────────────────────────────────────────────────────────────
if HAS_JAX:
    from geosteering_ai.simulation._jax.dipoles_native import (
        IMPLEMENTATION_STATUS,
        _hmd_tiv_native_jax_unified,
        _vmd_native_jax_unified,
        decoupling_factors_jax,
        native_dipoles_full_jax_unified,
    )
    from geosteering_ai.simulation._jax.dipoles_unified import (
        _hmd_tiv_propagation_unified,
        _vmd_propagation_unified,
    )
    from geosteering_ai.simulation._jax.forward_pure import (
        clear_unified_jit_cache,
        count_compiled_xla_programs,
    )
    from geosteering_ai.simulation._jax.geometry_jax import (
        find_layers_tr_jax,
        find_layers_tr_jax_vmap,
    )
    from geosteering_ai.simulation._jax.hankel import (
        integrate_j0,
        integrate_j0_j1,
        integrate_j1,
    )
    from geosteering_ai.simulation._jax.rotation import (
        build_rotation_matrix,
        rotate_tensor,
    )

    __all__ = [
        "HAS_JAX",
        "integrate_j0",
        "integrate_j1",
        "integrate_j0_j1",
        "build_rotation_matrix",
        "rotate_tensor",
        "decoupling_factors_jax",
        "IMPLEMENTATION_STATUS",
        # Sprint 10 (PR #23 Phase 1 + PR #24-part1 Phase 2)
        "_hmd_tiv_propagation_unified",
        "_vmd_propagation_unified",
        # Sprint 10 Phase 2 final (PR #24-part2): wrappers unified cabeados
        "_hmd_tiv_native_jax_unified",
        "_vmd_native_jax_unified",
        "native_dipoles_full_jax_unified",
        "count_compiled_xla_programs",
        "clear_unified_jit_cache",
        # Sprint 12 (PR #25): geometria tracer-safe para vmap real multi-TR/multi-ang
        "find_layers_tr_jax",
        "find_layers_tr_jax_vmap",
    ]
else:
    __all__ = ["HAS_JAX"]
