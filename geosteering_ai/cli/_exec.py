# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/_exec.py                                             ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Núcleo de execução de simulação da CLI (numba ⇄ jax)       ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.56 — geometria/chunk JAX + tempo)        ║
# ║  Versão      : v2.56                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-02                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : numpy + simulate_multi (numba) / simulate_batch (jax)      ║
# ║  Dependências: geosteering_ai.simulation (lazy dentro das funções)         ║
# ║  Padrão      : helpers stateless reusados por simulate.py + benchmark.py   ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Centraliza (DRY) a execução de um batch de modelos nos dois backends   ║
# ║    SEM regressão do caminho Numba (mantém ``simulate_multi(models=...)``  ║
# ║    EXATAMENTE como antes — preserva o pool de workers) e roteando o JAX   ║
# ║    pelo dispatcher parity-tested ``simulate_batch``. Também provê:        ║
# ║    finitude (NaN/Inf), paridade max|Δ|, mapeamento z→camada (res_h/res_v  ║
# ║    no ponto de observação) e warmup best-effort.                         ║
# ║                                                                           ║
# ║  INVIOLÁVEL                                                               ║
# ║    O ramo Numba chama ``simulate_multi(positions_z=, models=, cfg=        ║
# ║    SimulationConfig(n_workers, threads_per_worker), ...)`` byte-a-byte    ║
# ║    como a CLI legada — NÃO roteia pelo dispatcher (não perde o pool).     ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    sample_geometry, models_to_batch, rho_at_obs_from_batch,               ║
# ║    finitude_stats, parity_max_abs_diff, warmup_backend, run_once,         ║
# ║    run_compare_backends                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Núcleo de execução da CLI — numba (pool) ⇄ jax (dispatcher) (Sprint v2.54).

Helpers reusados por ``simulate`` e ``benchmark``: empacotamento de modelos,
mapeamento z→camada (res_h/res_v no ponto de observação para o `.dat`),
estatística de finitude (NaN/Inf), paridade max|Δ| entre tensores, warmup
best-effort e a execução cronometrada de UMA rodada (``run_once``).
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "models_to_batch",
    "rho_at_obs_from_batch",
    "finitude_stats",
    "parity_max_abs_diff",
    "warmup_backend",
    "run_once",
    "run_compare_backends",
    "sample_geometry",
    "resolve_requested_backend",
    "resolve_backend_preflight",
    "resolve_jax_chunk_size",
]

# Spec 0003 — versão-alvo em que o default de ``--backend`` muda numba→auto
# (próxima minor após o bump de semver da spec 0002). Citada no DeprecationWarning.
_BACKEND_DEFAULT_CHANGE_VERSION: str = "v2.57.0"

# Spec 0003 — constantes da árvore de decisão ``auto`` ESPELHADAS de
# ``geosteering_ai.simulation.dispatch`` (mesmo padrão de _HIGH_CONFIG_MIN_CONFIGS
# abaixo). Mantidas LOCAIS de propósito: importar o pacote ``simulation`` carrega
# os módulos ``_jax`` (``import jax``), o que violaria a disciplina jax-free do
# caminho Numba do pré-voo (v2.55). A não-divergência vs o dispatcher é garantida
# por teste de drift (``tests/test_cli_backend_auto.py``).
_AUTO_GROUPABLE_RATIO_MAX: float = 0.5  # espelha dispatch._GROUPABLE_RATIO_MAX
_AUTO_N_MODELS_GPU_THRESHOLD: int = 32  # espelha dispatch._N_MODELS_GPU_THRESHOLD

# Faixa de espessura das camadas internas (m) dos modelos sintéticos da CLI —
# alinhada a ``_build_models``/``_build_random_models`` (rng.uniform(2, 10)).
_CLI_THICKNESS_RANGE: Tuple[float, float] = (2.0, 10.0)

# Teto de geometrias distintas no modo ``templates`` (v2.56). POUCOS grupos
# grandes minimizam o re-tracing Python por-grupo do JAX-grouped (medido — ver
# relatório v2.56) E saturam melhor a GPU (1 vmap grande). Mantém alguma
# diversidade (>1) sem explodir o nº de shapes traçados.
_TEMPLATE_GEOMETRIES_CAP: int = 4


def sample_geometry(
    rng: "np.random.Generator",
    n_models: int,
    n_esp: int,
    *,
    mode: str = "per-model",
    n_geometries: Optional[int] = None,
    quantize_step: Optional[float] = None,
    thickness_range: Tuple[float, float] = _CLI_THICKNESS_RANGE,
) -> np.ndarray:
    """Amostra ``esp`` ``(n_models, n_esp)`` controlando a BATCHABILIDADE no JAX.

    Reconcilia diversidade geológica × ocupação da GPU, espelhando os 3 modos de
    produção em :func:`geosteering_ai.data.synthetic_generator` (§ geometry_mode):

      ┌──────────────┬──────────────────────────────────────────────────────┐
      │ per-model    │ esp único por modelo → cada modelo = 1 grupo de       │
      │  (default)   │   geometria → no JAX cai p/ Numba (não-agrupável)     │
      │ templates    │ K = n_geometries ou max(1,min(n//256,4)) geometrias   │
      │              │   DISTINTAS replicadas round-robin → POUCOS grupos    │
      │              │   GRANDES → vmap satura a GPU + warm bate Numba (v2.56)│
      │ quantized    │ esp contínuo arredondado a quantize_step → muitos     │
      │              │   modelos compartilham esp → agrupável (parcial)      │
      └──────────────┴──────────────────────────────────────────────────────┘

    Args:
        rng: gerador NumPy (reprodutibilidade via seed do caller).
        n_models: nº de modelos a gerar.
        n_esp: nº de espessuras internas (``n_layers - 2``).
        mode: ``"per-model"`` (default) | ``"templates"`` | ``"quantized"``.
        n_geometries: (só ``templates``) K geometrias distintas; default
            ``max(1, min(n_models//256, 4))`` (v2.56 — POUCOS grupos grandes
            minimizam o re-tracing Python por-grupo do JAX). Clampeado a
            ``[1, n_models]``.
        quantize_step: (só ``quantized``) passo em m; default
            ``(hi-lo)/8``. Deve ser > 0.
        thickness_range: faixa ``(lo, hi)`` das espessuras (m). Default (2, 10).

    Returns:
        ``np.ndarray`` ``(n_models, n_esp)`` float64 — espessuras internas.

    Raises:
        ValueError: ``mode`` inválido ou ``quantize_step <= 0``.

    Note:
        Espessuras de camada em LWD têm resolução finita — ``templates``/
        ``quantized`` refletem essa incerteza geológica realista (não é "esp
        fixo único"). O Numba é INDIFERENTE ao modo (cada modelo é independente
        no pool) — só o JAX-grouped se beneficia do compartilhamento. Reusa a
        mesma lógica de ``synthetic_generator.py`` (geometry_mode de produção).
    """
    lo, hi = thickness_range
    if n_esp <= 0:
        return np.zeros((n_models, 0), dtype=np.float64)

    if mode == "per-model":
        return rng.uniform(lo, hi, size=(n_models, n_esp)).astype(np.float64)

    if mode == "templates":
        # Default K (v2.56): POUCOS grupos grandes. A medição no A6000 (relatório
        # v2.56) mostrou que o custo do JAX-grouped é re-tracing Python POR-GRUPO
        # (não compilação — o cache XLA está cheio mas o 2º run é idêntico). Menos
        # grupos = menos tracing + 1 vmap maior satura melhor a GPU → mais rápido E
        # +13% throughput. `n//256` (cap 4) em vez de `n//32` (era 31 grupos a
        # n=1000; agora 3). Numba é INDIFERENTE (cada modelo é independente).
        n_geo = (
            n_geometries
            if n_geometries is not None
            else max(1, min(n_models // 256, _TEMPLATE_GEOMETRIES_CAP))
        )
        n_geo = max(1, min(int(n_geo), n_models))
        templates = rng.uniform(lo, hi, size=(n_geo, n_esp))
        # Réplica round-robin: modelos i e i+K compartilham esp bit-idêntico →
        # group_by_geometry forma K grupos batcháveis. (Indexação avançada já
        # retorna cópia fresca; np.asarray fixa o dtype + a tipagem.)
        return np.asarray(templates[np.arange(n_models) % n_geo], dtype=np.float64)

    if mode == "quantized":
        step = float(quantize_step) if quantize_step is not None else (hi - lo) / 8.0
        if step <= 0.0:
            raise ValueError(f"quantize_step deve ser > 0 (recebido {step}).")
        raw = rng.uniform(lo, hi, size=(n_models, n_esp))
        return np.asarray(
            np.clip(np.round(raw / step) * step, lo, hi), dtype=np.float64
        )

    raise ValueError(
        f"geometry inválido: {mode!r}. Use 'per-model' | 'templates' | 'quantized'."
    )


def _count_geometry_groups(esp_batch: np.ndarray) -> int:
    """Conta grupos de geometria DISTINTA (chave ``esp.tobytes()``) — NumPy PURO.

    Réplica **jax-free** da lógica de
    :func:`geosteering_ai.simulation._jax.multi_forward.group_by_geometry`. Permite
    à CLI decidir a AGRUPABILIDADE da geometria **ANTES** de importar/inicializar o
    JAX — crítico para evitar o crash de TLS estático (Sprint v2.55): se a geometria
    é não-agrupável (o JAX cairia p/ Numba), a CLI roda Numba SEM nunca chamar
    ``jax.devices()`` (que inicializaria o CUDA e consumiria o surplus de TLS,
    fazendo o init do threading-layer do Numba estourar com ``_dl_allocate_tls_init``).

    Args:
        esp_batch: ``(n_models, n_esp)`` float — espessuras internas por modelo.

    Returns:
        int: nº de geometrias DISTINTAS (≥ 1). ``n_esp == 0`` (≤2 camadas) → 1
        (todos compartilham ``b''``). Batch vazio → 1.

    Note:
        Equivalência bit-exata com ``group_by_geometry`` garantida por teste
        (``tests/test_cli_jax_crash_guard.py``). NÃO importa ``geosteering_ai.
        simulation._jax.*`` (que carregaria o módulo JAX).
    """
    esp_np = np.asarray(esp_batch, dtype=np.float64)
    if esp_np.ndim != 2:
        return 1
    return len({esp_np[i].tobytes() for i in range(esp_np.shape[0])}) or 1


def models_to_batch(
    models: Sequence[Mapping[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empilha a lista de modelos ``{rho_h, rho_v, esp}`` em arrays batched.

    Args:
        models: Lista de dicts ``{"rho_h": (n,), "rho_v": (n,), "esp": (n-2,)}``.

    Returns:
        Tupla ``(rho_h_batch, rho_v_batch, esp_batch)`` com shapes
        ``(n_models, n)`` / ``(n_models, n)`` / ``(n_models, n-2)`` float64.

    Raises:
        ValueError: ``models`` vazio (não há o que empilhar).
    """
    if not models:
        raise ValueError("models vazio — nada a empilhar para o batch.")
    rho_h_batch = np.stack([np.asarray(m["rho_h"], dtype=np.float64) for m in models])
    rho_v_batch = np.stack([np.asarray(m["rho_v"], dtype=np.float64) for m in models])
    esp_batch = np.stack([np.asarray(m["esp"], dtype=np.float64) for m in models])
    return rho_h_batch, rho_v_batch, esp_batch


def resolve_requested_backend(args: argparse.Namespace) -> str:
    """Normaliza ``args.backend`` e avisa quando o default IMPLÍCITO é usado (spec 0003).

    O flag ``--backend`` usa ``default=None`` (sentinela): ``None`` indica que o
    usuário NÃO escolheu explicitamente. Nesse caso, mantém o comportamento atual
    (``"numba"``) e emite um ``DeprecationWarning`` anunciando que o default mudará
    para ``"auto"`` em :data:`_BACKEND_DEFAULT_CHANGE_VERSION` (PEP 387 — evita
    regressão silenciosa em scripts/CI existentes).

    Args:
        args: ``Namespace`` do argparse com o atributo ``backend`` (``None`` |
            ``"numba"`` | ``"jax"`` | ``"auto"``).

    Returns:
        str: backend SOLICITADO concreto — ``"numba"`` (default implícito, com
        aviso) ou o valor explícito ``"numba"`` | ``"jax"`` | ``"auto"``.

    Note:
        Escolha explícita (mesmo ``--backend numba``) NÃO emite o aviso — o
        usuário fixou o comportamento conscientemente. ``stacklevel=2`` aponta o
        aviso para o chamador (``handle_simulate``/``handle_benchmark``).
        **Visibilidade**: o ``DeprecationWarning`` é o contrato Pythonic (capturável
        por ``pytest.warns`` / ``-W``), mas o Python o OCULTA por padrão fora de
        ``__main__``; por isso a mensagem é TAMBÉM emitida via ``logger.warning``
        (sempre visível ao usuário da CLI, sem depender de filtro de mensagem).
    """
    requested: Optional[str] = getattr(args, "backend", None)
    if requested is not None:
        return requested

    msg = (
        "O default de --backend mudará de 'numba' para 'auto' em "
        f"{_BACKEND_DEFAULT_CHANGE_VERSION} (seleção automática numba/jax pela "
        "árvore medida do dispatcher). Passe --backend explicitamente "
        "(numba|jax|auto) para silenciar este aviso e fixar o comportamento."
    )
    # Contrato Pythonic (pytest.warns / -W). Oculto por padrão fora de __main__.
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # Canal SEMPRE visível ao usuário da CLI (robusto, sem regex de mensagem).
    logger.warning("deprecação: %s", msg)
    return "numba"


def resolve_backend_preflight(
    requested: str,
    models: Sequence[dict],
    *,
    quiet: bool = False,
) -> Tuple[str, str, Optional[str]]:
    """Resolve o backend com PRÉ-VOO de geometria — evita o crash TLS (v2.55).

    Para ``requested == "jax"``, conta os grupos de geometria do batch **com NumPy
    puro** (:func:`_count_geometry_groups`, sem importar/inicializar JAX). Se a
    geometria é NÃO-AGRUPÁVEL (``n_groups > 0.5·n_models`` — o mesmo gate do
    dispatcher), o JAX-grouped degeneraria e o dispatcher cairia p/ Numba de
    qualquer forma; então retornamos ``("numba", "cpu", reason)`` SEM chamar
    :func:`resolve_backend` (que faria ``jax.devices()`` → init do CUDA). Assim o
    Numba roda com o **surplus de TLS estático íntegro** → o init do threading-layer
    libgomp NÃO estoura (``_dl_allocate_tls_init``). Se agrupável, delega a
    ``resolve_backend`` (que sonda a GPU normalmente).

    Args:
        requested: ``"numba"`` ou ``"jax"`` (escolha do usuário).
        models: lista de dicts ``{rho_h, rho_v, esp}`` (para contar geometrias).
        quiet: suprime o ``logger.warning`` do pré-voo.

    Returns:
        Tupla ``(backend_efetivo, device, reason)``:
          - ``("numba", "cpu", <motivo>)`` — jax + geometria não-agrupável
            (resolvido SEM tocar JAX);
          - ``(backend, device, None)`` — caso contrário (delegado a
            ``resolve_backend``: numba→cpu; jax+GPU→gpu; jax sem GPU→numba/cpu).

    Note:
        Crítico: para o caminho que vai rodar Numba, JAX NUNCA é importado/
        inicializado → sem CUDA no PID → sem pressão de TLS. See Also:
        :func:`geosteering_ai.cli._backend.resolve_backend`.
    """
    from geosteering_ai.cli._backend import resolve_backend

    req = (requested or "").strip().lower()

    # ── auto (spec 0003): árvore medida do dispatcher, em ordem TLS-safe ──
    if req == "auto":
        return _resolve_backend_auto(models, quiet=quiet)

    if req != "jax":
        backend, device = resolve_backend(requested, quiet=quiet)
        return backend, device, None

    # jax solicitado — pré-voo de agrupabilidade ANTES de tocar JAX.
    _rho_h, _rho_v, esp_batch = models_to_batch(models)
    n_models = len(models)
    n_groups = _count_geometry_groups(esp_batch)
    if n_groups > 0.5 * n_models:
        reason = (
            f"--backend jax + geometria não-agrupável ({n_groups}/{n_models} grupos) "
            f"→ Numba (JAX-grouped degeneraria; use --geometry templates)"
        )
        if not quiet:
            logger.warning("backend: %s", reason)
        return "numba", "cpu", reason

    backend, device = resolve_backend(requested, quiet=quiet)
    return backend, device, None


def _resolve_backend_auto(
    models: Sequence[dict], *, quiet: bool = False
) -> Tuple[str, str, Optional[str]]:
    """Resolve ``--backend auto`` replicando a árvore do dispatcher, em ordem TLS-safe.

    Produz a MESMA decisão de
    :func:`geosteering_ai.simulation.dispatch._resolve_backend` (modo ``"auto"``) —
    a conjunção ``GPU ∧ n_models≥limiar ∧ agrupável`` — porém com os
    disqualificadores **jax-free** avaliados PRIMEIRO (limiar de modelos e
    agrupabilidade via :func:`_count_geometry_groups`, NumPy puro) e a sonda de GPU
    (``jax.devices()``) por ÚLTIMO. Assim, batches pequenos ou geometria
    não-agrupável resolvem para Numba **sem nunca sondar a GPU** — ou seja, sem
    chamar ``jax.devices()`` / inicializar o CUDA, que é o GATILHO real do crash de
    TLS estático (Sprint v2.55). (O *módulo* ``jax`` pode já estar carregado pelo
    import do pacote no startup da CLI; o que importa para o crash é não INICIAR o
    CUDA no caminho Numba — e isso é garantido aqui.) Como a decisão é uma
    conjunção, a reordenação preserva o backend final.

    Os limiares são as constantes LOCAIS :data:`_AUTO_GROUPABLE_RATIO_MAX` /
    :data:`_AUTO_N_MODELS_GPU_THRESHOLD` — espelhos de
    ``dispatch._GROUPABLE_RATIO_MAX`` / ``dispatch._N_MODELS_GPU_THRESHOLD`` (mesmo
    padrão de :data:`_HIGH_CONFIG_MIN_CONFIGS`). São LOCAIS de propósito: importar
    ``dispatch`` carrega o pacote ``simulation`` (cujo ``__init__`` importa os
    módulos ``_jax`` → ``import jax``), o que violaria a disciplina jax-free do
    caminho Numba.

    Args:
        models: lista de dicts ``{rho_h, rho_v, esp}`` (para contar geometrias).
        quiet: suprime o ``logger.info`` do motivo da resolução.

    Returns:
        Tupla ``(backend_efetivo, device, reason)`` — backend SEMPRE concreto
        (``"numba"`` | ``"jax"``), nunca ``"auto"``.

    Note:
        NÃO chama ``dispatch._resolve_backend``. ``_jax_gpu_available`` é importado
        LAZY, só no passo final (quando a GPU efetivamente seria usada — único
        ponto que toca o JAX). A não-divergência vs o dispatcher é garantida por
        teste (``tests/test_cli_backend_auto.py`` — guard de drift das constantes +
        teste de consistência de decisão); a equivalência de contagem de grupos vs
        ``group_by_geometry`` é garantida por ``tests/test_cli_jax_crash_guard.py``.
    """
    _rho_h, _rho_v, esp_batch = models_to_batch(models)
    n_models = len(models)
    n_groups = _count_geometry_groups(esp_batch)
    groupable = n_groups <= _AUTO_GROUPABLE_RATIO_MAX * n_models

    # ── Disqualificadores JAX-FREE primeiro: NUNCA importam nem inicializam o
    #    jax → o caminho Numba preserva o surplus de TLS estático íntegro (v2.55). ──
    if n_models < _AUTO_N_MODELS_GPU_THRESHOLD:
        reason = (
            f"auto: n_models={n_models} < {_AUTO_N_MODELS_GPU_THRESHOLD} "
            "(GPU subocupada) → Numba CPU"
        )
        if not quiet:
            logger.info("backend: %s", reason)
        return "numba", "cpu", reason

    if not groupable:
        reason = (
            f"auto: geometria não-agrupável ({n_groups}/{n_models} grupos) → "
            "Numba CPU (JAX-grouped degeneraria; use --geometry templates)"
        )
        if not quiet:
            logger.info("backend: %s", reason)
        return "numba", "cpu", reason

    # ── Lado-CPU favorece a GPU → só AGORA importa+sonda a GPU. Import lazy de
    #    _jax_gpu_available (que carrega o jax) é aceitável aqui: se há GPU, o jax
    #    de fato roda (sem conflito com pool numba); em máquina CPU, devices()=cpu
    #    sem init de CUDA → cai p/ Numba. Este é o ÚNICO ponto que toca o JAX. ──
    from geosteering_ai.simulation.dispatch import _jax_gpu_available

    if not _jax_gpu_available():
        reason = "auto: nenhuma GPU JAX visível → Numba CPU"
        if not quiet:
            logger.info("backend: %s", reason)
        return "numba", "cpu", reason

    reason = (
        f"auto: GPU + n_models≥{_AUTO_N_MODELS_GPU_THRESHOLD} + agrupável "
        f"({n_groups}/{n_models} grupos) → JAX bucketed"
    )
    if not quiet:
        logger.info("backend: %s", reason)
    return "jax", "gpu", reason


# Limiar de high-config (espelha ``dispatch._HIGH_CONFIG_MIN_CONFIGS``): acima
# disso o tensor (n_models, nTR, nAng, n_pos, nf, 9) é grande o bastante para
# estourar VRAM com grupos GRANDES (v2.56) → fragmenta-se o eixo de modelos.
_HIGH_CONFIG_MIN_CONFIGS: int = 9
_JAX_AUTO_CHUNK_MODELS: int = 64


def resolve_jax_chunk_size(
    backend: str,
    n_configs: int,
    *,
    explicit: Optional[int] = None,
) -> Optional[int]:
    """Resolve o chunk de modelos do path JAX (anti-OOM em high-config) — v2.56.

    Com o default v2.56 de POUCOS grupos GRANDES (deliverable A), o ``jax.vmap``
    sobre o eixo de modelos cresce; em cenários **high-config** (``nf·nTR·nAng ≥ 9``,
    e.g. G/H) o tensor ``(n_models, nTR, nAng, n_pos, nf, 9)`` pode estourar a VRAM.
    Fragmentar o eixo de modelos (``jax_chunk_size_models``) evita o OOM sem afetar
    a numérica. Cenário E (``nf=1``) → ``None`` (vmap cheio, máxima velocidade).

    Args:
        backend: backend efetivo (só ``"jax"`` é afetado).
        n_configs: ``nf · nTR · nAng`` (nº de configurações por modelo).
        explicit: valor de ``--jax-chunk-size`` (vence o auto); None = auto.

    Returns:
        Optional[int]: chunk de modelos (``explicit`` se dado; senão
        ``_JAX_AUTO_CHUNK_MODELS`` em high-config jax; ``None`` caso contrário).

    Note:
        Reusa ``simulate_batch(jax_chunk_size_models=…)`` (dispatch). A medição de
        VRAM (3.2/48 GB) foi só no cenário E (nf=1); G/H (512 cfg) precisa do chunk.
    """
    if explicit is not None:
        return int(explicit)
    if backend == "jax" and n_configs >= _HIGH_CONFIG_MIN_CONFIGS:
        return _JAX_AUTO_CHUNK_MODELS
    return None


def rho_at_obs_from_batch(
    positions_z: np.ndarray,
    rho_h_batch: np.ndarray,
    rho_v_batch: np.ndarray,
    esp_batch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resistividade (h, v) NO ponto de observação — para o `.dat` 22-col.

    Mapeia cada profundidade ``z`` à camada que a contém e devolve a
    resistividade correspondente, preenchendo as colunas ``res_h``/``res_v``
    (col 2/3) do formato 22-col de ``geosteering-physics.md`` §4. Sem este
    mapeamento as colunas ficariam zeradas (não-conformes).

    Args:
        positions_z: ``(n_pos,)`` float — profundidades de observação (m).
        rho_h_batch: ``(n_models, n_layers)`` float — resistividade horizontal.
        rho_v_batch: ``(n_models, n_layers)`` float — resistividade vertical.
        esp_batch: ``(n_models, n_layers-2)`` float — espessuras internas (m).

    Returns:
        Tupla ``(rho_h_at_obs, rho_v_at_obs)`` com shape ``(n_models, n_pos)``.

    Note:
        Reusa o mapeamento z→camada VETORIZADO e bit-exato
        :func:`geosteering_ai.data.synthetic_generator._layer_at_batch`
        (twin de ``_find_layer_for_z``; equivalência garantida por teste).
        O caller faz o broadcast ``[:, None, :]`` para o eixo de dip antes de
        passar a ``write_dat_from_tensor`` (que espera ``(n_models, nAng, n_pos)``).
    """
    # Import lazy do helper canônico (mantém --help rápido; reuso DRY do
    # mapeamento já validado contra o laço escalar).
    from geosteering_ai.data.synthetic_generator import _layer_at_batch

    n_layers = int(rho_h_batch.shape[1])
    pos = np.asarray(positions_z, dtype=np.float64)
    layer = _layer_at_batch(pos, np.asarray(esp_batch, dtype=np.float64), n_layers)
    rho_h_at_obs = np.take_along_axis(rho_h_batch, layer, axis=1)
    rho_v_at_obs = np.take_along_axis(rho_v_batch, layer, axis=1)
    return rho_h_at_obs, rho_v_at_obs


def finitude_stats(H6: np.ndarray) -> dict:
    """Conta NaNs/Infs e reporta finitude total do tensor H.

    Args:
        H6: Tensor complexo (qualquer shape) — saída do simulador.

    Returns:
        dict ``{"nan_count": int, "inf_count": int, "all_finite": bool}``.
        Conta partes real E imaginária (um NaN em qualquer parte conta).

    Note:
        ``np.isinf``/``np.isnan`` em complexo testam ambas as partes. Usado
        pela tabela de resultados (saúde numérica) — um valor não-finito
        sinaliza modelo degenerado (e.g. resistividade ~0, dip 90°).
    """
    arr = np.asarray(H6)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    all_finite = bool(np.isfinite(arr).all())
    return {"nan_count": nan_count, "inf_count": inf_count, "all_finite": all_finite}


def parity_max_abs_diff(h_a: np.ndarray, h_b: np.ndarray) -> float:
    """Máxima diferença absoluta elemento-a-elemento entre dois tensores.

    Args:
        h_a: Primeiro tensor (e.g. saída Numba).
        h_b: Segundo tensor (e.g. saída JAX) — mesmo shape de ``h_a``.

    Returns:
        float: ``max|h_a - h_b|`` (0.0 se shapes vazios; ``inf`` se shapes
        incompatíveis não-broadcastáveis viram NaN→guarda).

    Raises:
        ValueError: shapes incompatíveis (não-broadcastáveis).

    Note:
        Usado por ``--compare-backends`` para guardar a fidelidade física
        (numba=referência <1e-6 Fortran; jax bucketed <1e-13 c128).
    """
    a = np.asarray(h_a)
    b = np.asarray(h_b)
    if a.shape != b.shape:
        raise ValueError(f"shapes incompatíveis para paridade: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def warmup_backend(
    backend: str,
    *,
    n_pos: int,
    n_layers: int = 5,
    dtype: str = "complex128",
    jax_strategy: str = "bucketed",
    dip_degs: Sequence[float] = (0.0,),
    tr_spacings_m: Sequence[float] = (1.0,),
    freqs_hz: Sequence[float] = (20000.0,),
) -> None:
    """Aquece o backend escolhido (JIT/XLA) antes da medição (best-effort).

    Args:
        backend: ``"numba"`` ou ``"jax"``.
        n_pos: Posições do grid (deve bater com produção p/ warmup pleno).
        n_layers: Camadas do modelo sintético de warmup (default 5).
        dtype: dtype complexo (path JAX — cada dtype é um cache key).
        jax_strategy: estratégia JAX a aquecer (default bucketed).
        dip_degs / tr_spacings_m / freqs_hz: geometria a pré-compilar.

    Returns:
        None. Falhas são engolidas (warmup é best-effort; ``logger.debug``).

    Note:
        Reusa ``_numba/warmup.warmup_numba_simulator`` (kernels prange) e
        ``_jax/warmup.warmup_jax_simulator`` (buckets XLA). Nunca propaga —
        a simulação real roda de qualquer forma (apenas sem cache quente).
    """
    try:
        if backend == "jax":
            from geosteering_ai.simulation._jax.warmup import warmup_jax_simulator

            warmup_jax_simulator(
                n_layers=n_layers,
                n_positions=n_pos,
                complex_dtype=dtype,
                jax_strategy=jax_strategy,
                dip_degs=tuple(dip_degs),
                tr_spacings_m=tuple(tr_spacings_m),
                freqs_hz=tuple(freqs_hz),
                verbose=False,
            )
        else:
            from geosteering_ai.simulation._numba.warmup import warmup_numba_simulator

            warmup_numba_simulator(n_layers=n_layers, n_positions=n_pos, verbose=False)
    except Exception as exc:  # noqa: BLE001 — warmup é best-effort
        logger.debug("warmup do backend %s falhou (ignorado): %s", backend, exc)


def run_once(
    backend: str,
    models: Sequence[dict],
    positions_z: np.ndarray,
    *,
    frequencies_hz: Sequence[float],
    dip_degs: Sequence[float],
    tr_spacings_m: Sequence[float],
    workers: Optional[int] = None,
    threads: Optional[int] = None,
    dtype: str = "complex128",
    jax_strategy: str = "bucketed",
    hankel_filter: str = "werthmuller_201pt",
    numba_fallback: bool = True,
    jax_chunk_size_models: Optional[int] = None,
) -> Tuple[np.ndarray, float, Optional[int], str, Optional[str]]:
    """Executa UMA rodada cronometrada no backend escolhido.

    Estrutura:
      ┌────────────────────────────────────────────────────────────────────┐
      │  numba → simulate_multi(models=…, cfg=SimulationConfig(workers,     │
      │          threads)) → result.H_stack  [POOL preservado — sem regress.]│
      │  jax   → simulate_batch(backend="jax", numba_fallback, dtype,       │
      │          jax_strategy) → (H6, info)   [dispatcher parity-tested]     │
      └────────────────────────────────────────────────────────────────────┘

    Args:
        backend: ``"numba"`` (pool de workers) ou ``"jax"`` (dispatcher GPU).
        models: lista de dicts ``{rho_h, rho_v, esp}``.
        positions_z: ``(n_pos,)`` profundidades (m).
        frequencies_hz / dip_degs / tr_spacings_m: eixos de geometria.
        workers / threads: paralelismo Numba (None = auto-detect). Ignorados no JAX.
        dtype: dtype complexo do path JAX. jax_strategy: estratégia JAX.
        hankel_filter: nome do filtro Hankel.
        numba_fallback: (só path jax) se ``True`` (DEFAULT), geometria
            NÃO-agrupável (n_grupos > 0.5·n_models) cai para Numba via o gate do
            dispatcher ([dispatch.py:147]) — evita o grouped degenerado (1
            modelo/grupo, sem batching). ``--compare-backends`` passa ``False``
            para FORÇAR o JAX e medir paridade real.
        jax_chunk_size_models: (só path jax) fragmenta o eixo de modelos do vmap
            em chunks (anti-OOM em high-config). None = vmap cheio. Ver
            :func:`resolve_jax_chunk_size`.

    Returns:
        Tupla ``(H6, elapsed_s, n_geometry_groups, effective_backend, reason)``:
          - ``H6``: ``(n_models, nTR, nAng, n_pos, nf, 9)`` complex;
          - ``elapsed_s``: tempo de parede da rodada (s);
          - ``n_geometry_groups``: nº de grupos de geometria (JAX) ou None (numba);
          - ``effective_backend``: backend REALMENTE executado (``"jax"`` pode
            virar ``"numba"`` por fallback de agrupabilidade);
          - ``reason``: motivo do roteamento do dispatcher (path jax) ou None.

    Raises:
        ValueError/RuntimeError/OSError: propagados do simulador (o handler
            da CLI captura e converte em exit code 1).

    Note:
        O ramo Numba replica EXATAMENTE a chamada da CLI legada
        (``simulate_multi(models=...)``) — qualquer roteamento pelo dispatcher
        perderia o ``ProcessPoolExecutor`` (regressão de throughput).
    """
    if backend == "jax":
        from geosteering_ai.simulation.dispatch import simulate_batch

        rho_h_batch, rho_v_batch, esp_batch = models_to_batch(models)
        t0 = time.perf_counter()
        H6, info = simulate_batch(
            rho_h_batch,
            rho_v_batch,
            esp_batch,
            np.asarray(positions_z, dtype=np.float64),
            frequencies_hz=list(frequencies_hz),
            tr_spacings_m=list(tr_spacings_m),
            dip_degs=list(dip_degs),
            backend="jax",
            numba_fallback=numba_fallback,
            dtype=dtype,
            jax_strategy=jax_strategy,
            jax_chunk_size_models=jax_chunk_size_models,
            hankel_filter=hankel_filter,
        )
        elapsed = time.perf_counter() - t0
        # info["backend"] reflete o backend EFETIVO (jax pode virar numba por
        # fallback de geometria não-agrupável); reportar p/ a tabela ser honesta.
        effective = str(info.get("backend", "jax"))
        reason = info.get("reason")
        n_groups = info.get("n_geometry_groups")
        return np.asarray(H6), elapsed, n_groups, effective, reason

    # ── Numba — caminho legado INALTERADO (pool de workers preservado) ───
    from geosteering_ai.simulation import simulate_multi
    from geosteering_ai.simulation.config import SimulationConfig

    cfg = SimulationConfig(n_workers=workers, threads_per_worker=threads)
    t0 = time.perf_counter()
    result = simulate_multi(
        positions_z=np.asarray(positions_z, dtype=np.float64),
        models=list(models),
        cfg=cfg,
        frequencies_hz=list(frequencies_hz),
        dip_degs=list(dip_degs),
        tr_spacings_m=list(tr_spacings_m),
    )
    elapsed = time.perf_counter() - t0
    # ``models=`` → MultiSimulationResultBatch (tem .H_stack) na union de retorno;
    # ``getattr`` evita o erro union-attr do mypy + valida em runtime (espelha
    # o padrão de dispatch._simulate_batch_numba com H_tensor).
    h_stack = getattr(result, "H_stack", None)
    if h_stack is None:  # pragma: no cover — guarda defensiva
        raise TypeError("simulate_multi(models=...) retornou tipo sem H_stack.")
    return np.asarray(h_stack), elapsed, None, "numba", None


def run_compare_backends(
    *,
    models: Sequence[dict],
    positions_z: np.ndarray,
    frequencies_hz: Sequence[float],
    dip_degs: Sequence[float],
    tr_spacings_m: Sequence[float],
    n_pos: int,
    workers: Optional[int],
    threads: Optional[int],
    dtype: str,
    jax_strategy: str,
    warmup: bool,
    as_json: bool,
    quiet: bool,
    title: str,
) -> int:
    """Roda numba E jax lado-a-lado: throughput, speedup e paridade max|Δ|.

    Força AMBOS os backends (o JAX roda mesmo sem GPU — em CPU, lento — para
    produzir um número de paridade real vs. a referência Numba/Fortran). Cada
    backend é isolado em try/except: se um falhar (e.g. JAX ausente), reporta
    N/A e segue. Compartilhado por ``simulate`` e ``benchmark`` (DRY).

    Args:
        models: lista de dicts ``{rho_h, rho_v, esp}``.
        positions_z: ``(n_pos,)`` profundidades (m).
        frequencies_hz/dip_degs/tr_spacings_m: eixos de geometria.
        n_pos: nº de posições (para o warmup).
        workers/threads: paralelismo Numba (None=auto). dtype/jax_strategy: path JAX.
        warmup: aquece cada backend antes de medir.
        as_json: emite o resultado da comparação em JSON.
        quiet: suprime a tabela (mantém o JSON se ``as_json``).
        title: título da tabela de comparação.

    Returns:
        Exit code: 0 (comparação concluída, mesmo com um backend N/A) | 1
        (ambos falharam).

    Note:
        Guarda a FIDELIDADE: ``max|Δ|`` entre numba (ref. Fortran <1e-6) e jax
        (bucketed <1e-13 c128) deve ser ínfimo. Ver
        :func:`parity_max_abs_diff`.
    """
    from geosteering_ai.cli._hwinfo import collect_hardware_info
    from geosteering_ai.cli._table import render_kv_table

    n_models = len(models)
    outcomes: dict[str, dict] = {}
    for be in ("numba", "jax"):
        try:
            if warmup:
                warmup_backend(
                    be,
                    n_pos=n_pos,
                    dtype=dtype,
                    jax_strategy=jax_strategy,
                    dip_degs=dip_degs,
                    tr_spacings_m=tr_spacings_m,
                    freqs_hz=frequencies_hz,
                )
            # numba_fallback=False: o compare DEVE forçar o JAX (mesmo com
            # geometria não-agrupável) para medir paridade REAL vs Numba —
            # caso contrário a coluna "jax" mediria Numba (speedup≈1.0× espúrio,
            # paridade≈0). FURO 1 da verificação adversarial: este site precisa
            # passar False EXPLÍCITO (run_once agora tem default True).
            H6, elapsed, groups, _eff, _reason = run_once(
                be,
                models,
                positions_z,
                frequencies_hz=frequencies_hz,
                dip_degs=dip_degs,
                tr_spacings_m=tr_spacings_m,
                workers=workers,
                threads=threads,
                dtype=dtype,
                jax_strategy=jax_strategy,
                numba_fallback=False,
            )
            thr = (n_models / elapsed) * 3600.0 if elapsed > 0 else 0.0
            outcomes[be] = {
                "ok": True,
                "H6": H6,
                "elapsed": elapsed,
                "thr": thr,
                "groups": groups,
            }
        except Exception as exc:  # noqa: BLE001 — backend pode estar indisponível
            logger.warning("backend %s indisponível para comparação: %s", be, exc)
            outcomes[be] = {"ok": False, "error": str(exc)}

    if not outcomes["numba"]["ok"] and not outcomes["jax"]["ok"]:
        logger.error("Ambos os backends falharam — nada a comparar.")
        return 1

    # ── Paridade + speedup (só se ambos rodaram) ─────────────────────────
    parity: Optional[float] = None
    speedup: Optional[float] = None
    if outcomes["numba"]["ok"] and outcomes["jax"]["ok"]:
        try:
            parity = parity_max_abs_diff(outcomes["numba"]["H6"], outcomes["jax"]["H6"])
        except ValueError as exc:
            logger.warning("paridade não computável: %s", exc)
        e_num = outcomes["numba"]["elapsed"]
        e_jax = outcomes["jax"]["elapsed"]
        if e_jax > 0:
            speedup = e_num / e_jax

    def _thr_txt(be: str) -> str:
        o = outcomes[be]
        return f"{o['thr']:,.0f} mod/h" if o["ok"] else f"N/A ({o['error'][:40]})"

    rows: List[Tuple[str, str]] = [
        ("n_models", f"{n_models:,}"),
        ("numba throughput", _thr_txt("numba")),
        ("jax throughput", _thr_txt("jax")),
        ("speedup (numba/jax)", f"{speedup:.2f}×" if speedup is not None else "—"),
        ("paridade max|Δ|", f"{parity:.2e}" if parity is not None else "—"),
    ]
    for be in ("numba", "jax"):
        if outcomes[be]["ok"]:
            finite = finitude_stats(outcomes[be]["H6"])["all_finite"]
            rows.append((f"{be} finito", "sim ✓" if finite else "NÃO ✗"))

    if as_json:
        import json

        payload = {
            "compare": {
                be: ({k: v for k, v in o.items() if k != "H6"} if o["ok"] else o)
                for be, o in outcomes.items()
            },
            "parity_max_abs_diff": parity,
            "speedup_numba_over_jax": speedup,
        }
        print(json.dumps(payload, ensure_ascii=False, default=str))
        return 0
    if not quiet:
        hw = collect_hardware_info(want_gpu=True)
        rows.append(("GPU", str(hw.get("gpu_name", "desconhecido"))))
        print(render_kv_table(title, rows))
    return 0
