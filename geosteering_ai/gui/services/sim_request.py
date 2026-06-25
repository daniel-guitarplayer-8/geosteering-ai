# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/gui/services/sim_request.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : SimRequest + construção de batch + chamada simulate_batch  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : GUI — services (spec 0011a)                                ║
# ║  Versão      : v0.1                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção — fundação (walking skeleton)                     ║
# ║  Framework   : stdlib + numpy PURO — NÃO importa Qt (Princípio X)          ║
# ║  Dependências: numpy; geosteering_ai.simulation (lazy)                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    A parte PURA (sem Qt) da camada de simulação: o ``SimRequest`` (que o   ║
# ║    ViewModel PURO monta) e ``_run_simulation`` (que o ``Worker`` roda).    ║
# ║    Separado de ``simulation_service.py`` (que importa Qt via BaseService)  ║
# ║    para que o ViewModel possa importar ``SimRequest`` SEM puxar Qt.        ║
# ║                                                                           ║
# ║  FIDELIDADE (inviolável)                                                  ║
# ║    ``_run_simulation`` só chama ``simulate_batch`` — não copia kernel nem  ║
# ║    altera ordem de ops (paridade Fortran <1e-12 preservada). Geometria     ║
# ║    FIXA + ``backend="numba"`` no skeleton → JAX não inicializa (TLS-safe). ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    SimRequest · compute_n_pos                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``SimRequest`` + construção de batch + ``_run_simulation`` — parte PURA (sem Qt)."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["SimRequest", "compute_n_pos"]

logger = logging.getLogger(__name__)

# Tipo do callback de progresso: ``progress_callback(done, total)`` (Fatia 6a). É
# injetado pelo ``Worker`` (emite ``signals.progress`` → QueuedConnection → VM). No
# caminho jax/subprocesso fica ``None`` (não cruza o limite de processo no v1).
_ProgressCb = Optional[Callable[[int, int], None]]


def _await_resume_or_cancel(cancel_event: Any, pause_event: Any) -> bool:
    """Coopera com pause/cancel ENTRE grupos (Fatia 6a). Retorna ``True`` se cancelado.

    Bloqueia (sleep cooperativo) enquanto ``pause_event`` estiver LIMPO (pausado),
    saindo imediatamente se ``cancel_event`` for setado. NÃO interrompe o kernel
    (a chamada ``simulate_batch`` é atômica) — só atua nas fronteiras de grupo, o
    que preserva a fidelidade (um grupo simulado é sempre completo ou descartado).

    Args:
        cancel_event: ``threading.Event`` (set = cancelar) ou ``None``.
        pause_event: ``threading.Event`` (set = rodando / clear = pausado) ou ``None``.

    Returns:
        ``True`` se o cancelamento foi solicitado; ``False`` caso contrário.
    """
    if cancel_event is not None and cancel_event.is_set():
        return True
    if pause_event is not None:
        while not pause_event.is_set():
            if cancel_event is not None and cancel_event.is_set():
                return True
            time.sleep(0.05)  # espera cooperativa (worker thread; não trava a UI)
    return False


def compute_n_pos(tj: float, p_med: float, dip0_deg: float) -> int:
    """Nº de pontos de medição pela CONVENÇÃO FORTRAN (fonte única da fórmula).

    ``n_pos = max(1, ceil(tj / (p_med · cos(dip0))))`` — o ``cos(dip0)`` projeta o
    passo no eixo vertical (guard ``1e-6`` evita ÷0 em 90°). Usado por
    :func:`_compute_positions_z` e pela property ``n_pos`` do ViewModel.

    Note:
        ``n_pos`` deriva SÓ do ``dip0`` (1º ângulo). Num run multi-dip a grade TVD
        (``positions_z``) é ÚNICA e COMPARTILHADA por todos os ângulos — o dip é
        parâmetro da resposta EM, não da amostragem. Consequência: a ORDEM dos dips
        muda ``n_pos`` (``[0,15]`` usa a grade do 0°; ``[15,0]`` a do 15°). NÃO há
        perda de precisão (cada ponto é um forward solve exato) — só a densidade de
        amostragem dos ângulos secundários difere da "nativa" deles (sub-cm,
        inofensivo: a resposta difusiva está super-amostrada). Convenção Fortran
        (espelha o monólito); o CLI passa ``n_pos`` direto.

    Args:
        tj: janela de investigação (m).
        p_med: passo entre medidas (m) — DEVE ser > 0 (é o denominador).
        dip0_deg: 1º ângulo de mergulho (graus).

    Returns:
        int ``≥ 1`` — nº de posições de medição.

    Raises:
        ValueError: se ``p_med <= 0`` (divisão por zero). Hardening de API
            pública — os chamadores internos (ViewModel/View) já guardam, mas a
            função é exportada e não confia no chamador.
    """
    if p_med <= 0.0:
        raise ValueError(f"p_med deve ser > 0 (got {p_med}).")
    cos_d = max(1e-6, math.cos(math.radians(abs(dip0_deg))))
    return max(1, int(math.ceil(tj / (p_med * cos_d))))


def _jax_pool_initializer() -> None:
    """``initializer`` do pool JAX PERSISTENTE — roda no INÍCIO do subprocesso filho
    (ANTES de qualquer import pesado do filho). Definido AQUI (módulo Qt-FREE, já
    importado pelo filho via :func:`_run_simulation`) e NÃO em ``gui/services/base.py``,
    para que o ``spawn`` resolva esta referência sem importar ``base.py`` — que puxa Qt
    (mantém o filho enxuto no cold-start; revisão adversarial Turn 7 op c).

    Fixa a ORDEM do env de pré-alocação de VRAM ANTES de o XLA inicializar:
    ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (não agarra ~75% da VRAM de cara) via
    ``os.environ.setdefault`` (respeita um override exportado pelo usuário). NUNCA
    importa ``jax`` (TLS-safe — só mexe em ``os.environ``; mantenha-a NÃO-levantante).

    Defense-in-depth: a ordem JÁ é correta hoje porque o filho carrega
    ``simulation/_jax/__init__::_setup_xla_environment`` (mesmo ``setdefault``) ANTES do
    ``import jax``. Este initializer garante a ordem mesmo se o grafo de imports do filho
    mudar no futuro. O ``XLA_PYTHON_CLIENT_MEM_FRACTION`` por-tier de GPU continua SÓ em
    ``_setup_xla_environment`` (fonte única do tuning por GPU) — NÃO o tocamos aqui.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# Chunk FIXO do eixo de modelos no path JAX do SM (Turn 7, op d). 64 casa o
# ``_SM_CANON_N_MODELS`` do ``cli/warmup.py`` (dim líder do HLO vmapado) → a forma
# de 64 do warmup bate a maioria das fatias do runtime (uma fatia-cauda < 64, ou um
# n_models forçado ≠ 64, paga 1 compile XLA único e não-aquecido — só cold-start). Limita
# a VRAM de pico fatiando ``n_models`` (ex.: 2000) e concatenando em CPU. Paridade vs
# chunk=None: <1e-13 (bit-exato em GPU; ~1e-14 ULP em CPU por reordenação de reduções do
# XLA conforme o tamanho do batch — fisicamente idêntico, MUITO abaixo do 1e-12 Fortran).
_SM_JAX_CHUNK_SIZE_MODELS = 64


@dataclass(frozen=True)
class SimRequest:
    """Requisição de simulação (Fatia 2 params + Fatia 3 geologia estocástica).

    Attributes:
        frequencies_hz: frequências em Hz (range do simulador [10, 2e6]; default 20 kHz).
        tr_spacings_m: espaçamentos transmissor-receptor em m (range [0.1, 50]; default 1.0).
        dip_degs: ângulos de mergulho em graus (range [0, 90]; default 0°).
        h1: altura do 1º ponto-médio acima da interface (m) — convenção Fortran.
        tj: janela de investigação (m) — extensão da varredura de profundidade; também
            é o ``total_depth`` da geologia estocástica (Σ espessuras = tj).
        p_med: passo entre medidas (m).
        n_models: nº de modelos no batch (default 2).
        backend: ``"numba"`` (evita JAX/TLS); auto/jax = Fatia 5.
        geology_mode: ``"fixed"`` (geologia 3-camadas determinística da Fatia 2) ou
            ``"stochastic"`` (gera N modelos TIV via gui.services.stochastic_geology).
        n_layers_min/max: range de camadas amostradas (incl/excl) quando não-fixo.
        n_layers_fixed: se ≥3, força todos os modelos a esse nº de camadas.
        rho_h_min/max: range de ρₕ (Ω·m) da geração estocástica.
        rho_h_distribution: ``"loguni"`` (log-uniforme) ou ``"uniform"``.
        anisotropic: se ``True``, ρᵥ=λ²·ρₕ com λ∈[lambda_min,lambda_max]; senão ρᵥ=ρₕ.
        lambda_min/max: range do fator de anisotropia λ (≥1, TIV física).
        min_thickness: piso de espessura por camada (m).
        generator: gerador estocástico (sobol/halton/niederreiter/mersenne_twister/
            uniform/normal/box_muller — ver ``GENERATORS_AVAILABLE``).
        normal_mu_log/sigma_log: parâmetros log-normal quando ``generator="normal"``.
        rng_seed: ``None`` (semente aleatória a cada run) ou int (reprodutível).

    Note:
        ``h1``/``tj``/``p_med`` (+ ``dip_degs[0]``) determinam ``positions_z`` pela
        convenção Fortran (ver :func:`_compute_positions_z`). Os campos de geologia só
        têm efeito quando ``geology_mode="stochastic"``.

        Os defaults deste dataclass são LEVES (objeto-fio / teste: tj=10, p_med=1 ⇒
        n_pos=10; n_models=2) e DIFEREM **de propósito** dos defaults de PRODUÇÃO do
        ``SimulationViewModel`` (tj=120/p_med=0.2 ⇒ n_pos=600; n_models=2000 — paridade
        c/ o monólito). A View/VM SEMPRE passa valores explícitos; estes defaults são só
        o fallback do objeto-fio. NÃO construa ``SimRequest()`` esperando a forma de
        produção.
    """

    frequencies_hz: Tuple[float, ...] = (20000.0,)
    tr_spacings_m: Tuple[float, ...] = (1.0,)
    dip_degs: Tuple[float, ...] = (0.0,)
    h1: float = 1.0
    tj: float = 10.0
    p_med: float = 1.0
    n_models: int = 2
    backend: str = "numba"
    # ── Fatia 3 — geologia estocástica (ver gui/services/stochastic_geology) ──
    geology_mode: str = "fixed"  # "fixed" (determinístico) | "stochastic"
    n_layers_min: int = 3
    n_layers_max: int = 11
    n_layers_fixed: Optional[int] = None
    rho_h_min: float = 1.0
    rho_h_max: float = 1000.0
    rho_h_distribution: str = "loguni"  # "loguni" | "uniform"
    anisotropic: bool = True
    lambda_min: float = 1.0
    lambda_max: float = math.sqrt(2.0)
    min_thickness: float = 1.0
    generator: str = "sobol"
    normal_mu_log: float = 2.0
    normal_sigma_log: float = 1.0
    rng_seed: Optional[int] = None
    # ── Fatia 6b — filtro de Hankel + geologia manual ────────────────────────
    # Filtro de Hankel (catálogo em simulation/filters/loader.py). simulate_batch
    # já aceita o nome (paridade <1e-12 preservada — só troca os pesos do filtro).
    hankel_filter: str = "werthmuller_201pt"
    # Geologia MANUAL (geology_mode="manual"): N camadas arbitrárias (editor de
    # camadas / perfil canônico). Tuplas (picklable/imutável); replicadas n_models×.
    manual_n_layers: int = 0
    manual_thicknesses: Tuple[float, ...] = ()
    manual_rho_h: Tuple[float, ...] = ()
    manual_rho_v: Tuple[float, ...] = ()
    # ── Lote 1 — paralelismo + saída Fortran-compat ──────────────────────────
    # threads_per_worker tem EFEITO REAL: numba.set_num_threads(...) antes do
    # simulate_batch (knob de threading → resultados bit-idênticos; clampado a
    # NUMBA_NUM_THREADS). ``0`` = não toca o numba (default p/ chamadas diretas em
    # teste — a View/VM sempre passa ≥1). n_workers é transportado (pool = Fatia 5).
    n_workers: int = 1
    threads_per_worker: int = 0
    # ── Saída — artefatos .dat/.out (22-col, formato do tatu.x; LIDOS pelo
    #    Visualizador .dat da Fatia 6h). É um FORMATO de arquivo do simulador
    #    Python, NÃO executa Fortran. Se save_fortran_artifacts e output_dir,
    #    grava .dat (22-col binário) + .out ASCII via write_dat_from_tensor.
    output_dir: str = ""
    save_fortran_artifacts: bool = False
    # ── PR-1 — diversidade de geometria p/ batchabilidade no JAX GPU ──────────
    # No modo estocástico cada modelo tem espessuras ÚNICAS → o dispatcher vê
    # n_grupos=n_models (agrupa por esp.tobytes() em _jax/multi_forward) e o
    # JAX-grouped degenera em per-model → fallback p/ Numba (lento). "templates"
    # colapsa as espessuras a K geometrias por n_layers (round-robin) → POUCOS
    # grupos → o JAX batela na GPU. ρ_h/ρ_v/λ continuam por-modelo (diversidade
    # petrofísica preservada). "auto" = templates p/ backend jax/auto; per_model
    # p/ numba (que é INDIFERENTE à agrupabilidade — cada modelo é independente).
    # NÃO toca a física: cada modelo é computado com seu próprio esp/ρ/λ.
    geometry_diversity: str = "auto"  # "auto" | "templates" | "per_model"
    n_geometries: Optional[int] = (
        None  # K templates (só templates); None = auto (cap 4)
    )
    # ── Turn 7 (op d) — chunk FIXO do eixo de modelos no path JAX (VRAM) ──────
    # Limita a VRAM de pico fatiando n_models em pedaços de K p/ o vmap do XLA
    # (mesma compilação por fatia, concatena em CPU — multi_forward.py:1094-1139);
    # paridade <1e-13 vs chunk=None (GPU bit-exato; CPU ~1e-14 ULP). Só afeta o backend
    # JAX (numba ignora). None = vmap monolítico. Ver _SM_JAX_CHUNK_SIZE_MODELS.
    jax_chunk_size_models: Optional[int] = _SM_JAX_CHUNK_SIZE_MODELS


# Modelo TIV de referência (3 camadas): ρₕ por camada + λ²=2 (anisotropia branda,
# λ=√2≈1.414 ∈ [1,5]) + 1 espessura interna de 8 m. Valores dentro da errata física
# (ρ ∈ [0.01, 1e6]). Geometria FIXA no skeleton — geração estocástica é fatia futura.
_BASE_RHO_H = np.array([1.0, 10.0, 100.0], dtype=np.float64)  # (n_layers=3,)
_LAMBDA_SQ = 2.0
_INNER_THICKNESS_M = 8.0


def _compute_positions_z(request: SimRequest) -> np.ndarray:
    """Posições de medição ``positions_z`` pela CONVENÇÃO FORTRAN (Fatia 2).

    Replica EXATAMENTE o cálculo do monólito (``simulation_manager.py:~8221``):
    o nº de pontos é ``ceil(tj / (p_med · cos(dip0)))`` e ``z_obs`` vai de ``-h1``
    (acima da 1ª interface) a ``tj - h1`` (abaixo), com as interfaces em ``z=0``.
    O ``cos(dip0)`` projeta o passo no eixo vertical (guard ``1e-6`` evita ÷0 em 90°).

    Args:
        request: a requisição (usa ``h1``, ``tj``, ``p_med`` e ``dip_degs[0]``).

    Returns:
        ``positions_z`` shape ``(n_pos,)`` float64 — ``linspace(-h1, tj-h1, n_pos)``.
    """
    # n_pos via :func:`compute_n_pos` (FONTE ÚNICA da fórmula — não reinlinar).
    n_pos = compute_n_pos(request.tj, request.p_med, request.dip_degs[0])
    return np.linspace(-request.h1, request.tj - request.h1, n_pos, dtype=np.float64)


def _build_batch(
    request: SimRequest,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constrói um batch TIV pequeno e VÁLIDO para ``simulate_batch``.

    Geologia FIXA (3-camadas TIV; ρₕ varia ×(1+0.1·i) por modelo para não degenerar;
    esp = n_layers−2 internas) — geração estocástica é a Fatia 3. ``positions_z`` vem
    da convenção Fortran (:func:`_compute_positions_z`, a partir de h1/tj/p_med/dip).

    Args:
        request: a requisição (usa ``n_models`` + h1/tj/p_med/dip via positions_z).

    Returns:
        ``(rho_h, rho_v, esp, positions_z)`` — shapes ``(n,3)``, ``(n,3)``,
        ``(n,1)``, ``(n_pos,)`` float64.
    """
    n = max(1, int(request.n_models))
    rho_h = np.stack([_BASE_RHO_H * (1.0 + 0.1 * i) for i in range(n)])  # (n, 3)
    rho_v = rho_h * _LAMBDA_SQ  # ρᵥ = ρₕ·λ²  (λ=√2 → anisotropia TIV branda)
    esp = np.full((n, 1), _INNER_THICKNESS_M, dtype=np.float64)  # n_layers−2 = 1
    positions_z = _compute_positions_z(request)
    return rho_h, rho_v, esp, positions_z


# ──────────────────────────────────────────────────────────────────────────
# Fatia 3 — geração estocástica (gerador PURO compartilhado com o monólito)
# ──────────────────────────────────────────────────────────────────────────


def _genconfig_from_request(request: SimRequest) -> Any:
    """Constrói um ``GenConfig`` (gerador puro) a partir do ``SimRequest``.

    ``total_depth`` = ``request.tj`` (a Σ das espessuras casa a janela de
    investigação; ``positions_z`` cobre ``[-h1, tj-h1]``, interfaces em ``[0, tj]``).

    Args:
        request: a requisição (campos de geologia).

    Returns:
        ``GenConfig`` (de :mod:`gui.services.stochastic_geology`).
    """
    from geosteering_ai.gui.services.stochastic_geology import GenConfig

    return GenConfig(
        total_depth=request.tj,
        n_layers_min=request.n_layers_min,
        n_layers_max=request.n_layers_max,
        n_layers_fixed=request.n_layers_fixed,
        rho_h_min=request.rho_h_min,
        rho_h_max=request.rho_h_max,
        rho_h_distribution=request.rho_h_distribution,
        anisotropic=request.anisotropic,
        lambda_min=request.lambda_min,
        lambda_max=request.lambda_max,
        min_thickness=request.min_thickness,
        generator=request.generator,
        normal_mu_log=request.normal_mu_log,
        normal_sigma_log=request.normal_sigma_log,
    )


def _generate_stochastic_models(request: SimRequest) -> List[Dict[str, Any]]:
    """Gera ``n_models`` perfis TIV estocásticos (lista de dicts ``MODEL_KEYS``).

    Usa o gerador PURO compartilhado (extraído de ``sm_model_gen`` em 0011c).
    Pode ser RAGGED (``n_layers`` variável) quando ``n_layers_fixed is None`` —
    o agrupamento por ``n_layers`` é feito em :func:`_simulate_grouped`.

    Args:
        request: a requisição (campos de geologia + ``n_models``/``rng_seed``).

    Returns:
        Lista de dicts ``{"n_layers", "rho_h", "rho_v", "lambda", "thicknesses"}``.
    """
    from geosteering_ai.gui.services.stochastic_geology import generate_models

    cfg = _genconfig_from_request(request)
    # ``generate_models`` é sem anotação de retorno (Any) → tipa local p/ mypy.
    models: List[Dict[str, Any]] = generate_models(
        cfg, max(1, int(request.n_models)), rng_seed=request.rng_seed
    )
    return models


def _manual_models(request: SimRequest) -> List[Dict[str, Any]]:
    """Replica a geologia MANUAL (N camadas) ``n_models`` vezes (Fatia 6b).

    O modo manual simula UM perfil (editor de camadas ou perfil canônico)
    replicado em ``n_models`` cópias idênticas — paridade com o monólito
    (``simulation_manager.py:~8281``, ``for _ in range(n_models)``). Como todos
    têm o mesmo ``n_layers``, formam 1 grupo em :func:`_simulate_grouped`.

    Args:
        request: a requisição (campos ``manual_*`` + ``n_models``).

    Returns:
        Lista de ``n_models`` dicts ``{"n_layers", "rho_h", "rho_v", "thicknesses"}``.
    """
    model: Dict[str, Any] = {
        "n_layers": int(request.manual_n_layers),
        "rho_h": list(request.manual_rho_h),
        "rho_v": list(request.manual_rho_v),
        "thicknesses": list(request.manual_thicknesses),
    }
    return [dict(model) for _ in range(max(1, int(request.n_models)))]


# Teto default de geometrias distintas POR grupo de n_layers no modo templates.
# Inspirado no CLI (`_exec._TEMPLATE_GEOMETRIES_CAP=4`), mas ADAPTADO ao agrupamento
# por-n_layers do SM: aqui o teto é por GRUPO (clampado a g//2 p/ garantir n_grupos ≤
# 0.5·g), não pela divisão por-batch do CLI (n//256). Poucas geometrias por grupo
# maximizam a ocupação da GPU/minimizam re-tracing, mantendo alguma diversidade (>1).
#
# NOTA DE OCUPAÇÃO (honestidade): o colapso resolve a AGRUPABILIDADE (esp.tobytes()),
# mas NÃO a ocupação da GPU, que o dispatcher mede POR grupo de n_layers (≥32 modelos
# = `_N_MODELS_GPU_THRESHOLD`). Com geologia RAGGED (n_layers variável) e N pequeno,
# grupos com <32 modelos ainda caem p/ Numba. Para batelar TODO o ensemble na GPU:
# N grande (cada grupo ≥32 — ex.: 1000 modelos / faixa 3-11 ⇒ ~125/grupo) OU n_layers
# FIXO (1 grupo grande). Não há como elevar a ocupação sem mexer no n_layers (quebraria
# ρ por-camada por-modelo) ou no dispatcher (física sagrada) — por isso é documentado.
_TEMPLATE_GEOMETRIES_CAP: int = 4


def _templates_active(request: SimRequest) -> bool:
    """Decide se as espessuras devem ser colapsadas a templates (batchável no JAX).

    ``"templates"`` sempre; ``"per_model"`` nunca; ``"auto"`` → só p/ backend
    jax/auto (o Numba é INDIFERENTE à agrupabilidade — cada modelo é independente
    no pool, então não há ganho em colapsar e preserva-se a diversidade total).

    Args:
        request: a requisição (campos ``geometry_diversity`` + ``backend``).

    Returns:
        bool: ``True`` se o colapso a templates deve ser aplicado.
    """
    mode = request.geometry_diversity
    if mode == "templates":
        return True
    if mode == "per_model":
        return False
    return request.backend in ("jax", "jax_gpu", "auto")  # "auto"


def _collapse_geometry_to_templates(
    models: List[Dict[str, Any]], n_geometries: Optional[int]
) -> List[Dict[str, Any]]:
    """Colapsa as espessuras a K templates POR n_layers (round-robin) — batchável.

    Mantém as espessuras REAIS geradas (um subconjunto vira template) — NÃO inventa
    geometria uniforme, preservando a riqueza do gerador estocástico. Dentro de cada
    grupo de ``n_layers`` (``g`` modelos), escolhe ``K = min(K_req, g//2)`` templates
    (garante a agrupabilidade do dispatcher: ``n_grupos ≤ 0.5·g``, ``dispatch.py:127``)
    e reatribui ``thicknesses`` round-robin. ``rho_h``/``rho_v``/``lambda`` ficam
    INTOCADOS (diversidade petrofísica por-modelo). NÃO toca a física — só prepara o
    batch para o JAX-grouped saturar a GPU (cada modelo é computado com seu esp/ρ/λ).

    Args:
        models: lista de dicts ``{"n_layers","rho_h","rho_v","thicknesses",...}``.
        n_geometries: K templates por grupo (``None`` → :data:`_TEMPLATE_GEOMETRIES_CAP`).

    Returns:
        A MESMA lista (mutada in-place) com ``thicknesses`` colapsadas a ≤K por n_layers.
    """
    from collections import defaultdict

    groups: Dict[int, List[int]] = defaultdict(list)
    for i, m in enumerate(models):
        groups[int(m["n_layers"])].append(i)
    for _n_layers, idxs in groups.items():
        g = len(idxs)
        if g <= 1:
            continue  # 1 modelo no grupo → nada a compartilhar
        k_req = (
            int(n_geometries) if n_geometries is not None else _TEMPLATE_GEOMETRIES_CAP
        )
        k = max(1, min(k_req, g // 2))  # garante K ≤ 0.5·g (agrupável)
        templates = [list(models[idxs[j]]["thicknesses"]) for j in range(k)]
        for local, original_idx in enumerate(idxs):
            models[original_idx]["thicknesses"] = list(templates[local % k])
    return models


def _simulate_grouped(
    models: List[Dict[str, Any]],
    positions_z: np.ndarray,
    request: SimRequest,
    progress_callback: _ProgressCb = None,
    cancel_event: Any = None,
    pause_event: Any = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Simula modelos RAGGED agrupando por ``n_layers`` e reassembla ``H6``.

    ``simulate_batch`` exige batch RETANGULAR (``n_layers`` uniforme). Modelos com
    ``n_layers`` diferentes vão em grupos distintos: cada grupo é um
    ``simulate_batch`` (caminho Numba serial da Fatia 2 — SEM pool, adiado p/
    Fatia 5), e o ``H6`` de cada grupo é colocado de volta no índice ORIGINAL do
    modelo (a ordem do batch final == ordem de ``models``).

    Feedback (Fatia 6a): emite progresso POR-GRUPO via ``progress_callback`` e
    coopera com ``cancel_event``/``pause_event`` ENTRE grupos (a chamada do kernel
    é atômica — o cancel descarta o resultado parcial, nunca o corrompe).

    Args:
        models: lista de dicts ``MODEL_KEYS`` (possivelmente ragged).
        positions_z: ``(n_pos,)`` float64 — compartilhado por todos (convenção Fortran).
        request: a requisição (freqs/dips/TRs/backend).
        progress_callback: ``f(done, total)`` chamado por grupo concluído (ou ``None``).
        cancel_event/pause_event: ``threading.Event`` cooperativos (ou ``None``).

    Returns:
        ``(H6, info)`` — ``H6`` shape ``(n_models, nTR, nAng, n_pos, nf, 9)`` na
        ordem original; ``info`` agrega os dispatchers por grupo. Se cancelado,
        ``(None, {"cancelled": True})`` (resultado parcial descartado).
    """
    from collections import defaultdict

    from geosteering_ai.simulation import simulate_batch  # lazy (Numba pesado)

    n_models = len(models)
    freqs = list(request.frequencies_hz)
    trs = list(request.tr_spacings_m)
    dips = list(request.dip_degs)

    # Agrupa índices ORIGINAIS por n_layers (sorted p/ determinismo do dict order).
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, m in enumerate(models):
        groups[int(m["n_layers"])].append(i)

    if progress_callback is not None:
        progress_callback(0, n_models)  # estado inicial (0%)

    h6_out: Optional[np.ndarray] = None
    info_by_group: Dict[str, Any] = {}
    models_done = 0
    for n_layers, idxs in sorted(groups.items()):
        # Pause/cancel cooperativo ANTES de cada grupo (kernel é atômico).
        if _await_resume_or_cancel(cancel_event, pause_event):
            return None, {"cancelled": True}
        # Empilha o grupo em arrays RETANGULARES (mesmo n_layers).
        rho_h = np.array([models[i]["rho_h"] for i in idxs], dtype=np.float64)  # (g, L)
        rho_v = np.array([models[i]["rho_v"] for i in idxs], dtype=np.float64)  # (g, L)
        esp = np.array(
            [models[i]["thicknesses"] for i in idxs], dtype=np.float64
        )  # (g, L-2)
        h6_g, info_g = simulate_batch(
            rho_h,
            rho_v,
            esp,
            positions_z,
            frequencies_hz=freqs,
            tr_spacings_m=trs,
            dip_degs=dips,
            backend=request.backend,
            hankel_filter=request.hankel_filter,
            jax_chunk_size_models=request.jax_chunk_size_models,  # VRAM (op d); no-op p/ numba
        )
        if h6_out is None:
            # 1º grupo define o shape downstream (nTR, nAng, n_pos, nf, 9).
            h6_out = np.empty((n_models,) + h6_g.shape[1:], dtype=h6_g.dtype)
        for local, original_idx in enumerate(idxs):
            h6_out[original_idx] = h6_g[local]  # reassembla na ORDEM original
        info_by_group[str(n_layers)] = info_g
        models_done += len(idxs)
        if progress_callback is not None:
            progress_callback(models_done, n_models)  # progresso por grupo concluído

    assert h6_out is not None  # n_models ≥ 1 garante ≥ 1 grupo
    return h6_out, {"groups": info_by_group, "n_groups": len(groups)}


def _apply_thread_count(threads_per_worker: int) -> None:
    """Aplica ``numba.set_num_threads`` (knob de threading — resultados bit-idênticos).

    Clampa a ``NUMBA_NUM_THREADS`` (``set_num_threads`` levanta se exceder). ``≤0`` =
    não toca (default p/ chamadas diretas em teste). Best-effort: qualquer falha
    (numba ausente, valor inválido) é silenciosa — NUNCA afeta a física nem derruba a
    simulação.
    """
    try:
        n = int(threads_per_worker)
        if n < 1:
            return
        import numba  # lazy

        cap = int(numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(min(n, cap))
    except Exception:  # noqa: BLE001 — best-effort; threading não é física
        pass


def _write_fortran_artifacts(
    out_dir: str,
    h6: np.ndarray,
    positions_z: np.ndarray,
    geology: List[Dict[str, Any]],
    request: SimRequest,
) -> str:
    """Grava ``.dat`` (22-col) + ``.out`` (ASCII) Fortran-compat; retorna o path do .dat.

    ``col2/col3`` (ρₕ/ρᵥ no ponto de observação) vêm de :func:`derived.rho_profile`
    (byte-fiel ao perfil — convenção do monólito). NÃO toca a física: só serializa o
    H6 já computado.
    """
    import os

    from geosteering_ai.gui.services.derived import rho_profile
    from geosteering_ai.simulation.io.tensor_dat import (
        write_dat_from_tensor,
        write_out_file,
    )

    os.makedirs(out_dir, exist_ok=True)
    n_models, _n_tr, n_ang, n_pos, n_f, _ = h6.shape
    z = np.asarray(positions_z, dtype=np.float64).reshape(-1)
    # ρ no ponto de observação (n_models, n_pos) → broadcast no eixo de ângulos.
    rho_h_obs = np.stack(
        [rho_profile(z, g["rho_h"], g["thicknesses"]) for g in geology]
    )
    rho_v_obs = np.stack(
        [rho_profile(z, g["rho_v"], g["thicknesses"]) for g in geology]
    )
    rho_h_obs = np.broadcast_to(rho_h_obs[:, None, :], (n_models, n_ang, n_pos))
    rho_v_obs = np.broadcast_to(rho_v_obs[:, None, :], (n_models, n_ang, n_pos))
    dat_path = os.path.join(out_dir, "sm_output.dat")
    write_dat_from_tensor(
        dat_path, h6, z, rho_h_at_obs=rho_h_obs, rho_v_at_obs=rho_v_obs
    )
    write_out_file(
        os.path.join(out_dir, "sm_output.out"),
        n_dips=n_ang,
        n_freqs=n_f,
        nmaxmodel=n_models,
        angles=list(request.dip_degs),
        freqs_hz=list(request.frequencies_hz),
        nmeds_per_angle=[n_pos] * n_ang,
    )
    return dat_path


def _run_simulation(
    request: SimRequest,
    progress_callback: _ProgressCb = None,
    cancel_event: Any = None,
    pause_event: Any = None,
) -> Dict[str, Any]:
    """Roda ``simulate_batch`` (na worker thread) e empacota o resultado.

    PURO/picklable (módulo-nível) — chamável direto em teste de fidelidade SEM Qt.
    Importa ``simulate_batch`` lazy (não pesa o import deste módulo).

    Args:
        request: a requisição validada.
        progress_callback: ``f(done, total)`` p/ feedback de progresso (Fatia 6a;
            ``None`` no caminho jax/subprocesso — não cruza o limite de processo).
        cancel_event/pause_event: ``threading.Event`` cooperativos (numba in-thread;
            ``None`` no subprocesso). Cancel entre grupos → resultado ``cancelled``.

    Returns:
        dict com ``H6`` (n_models, nTR, nAng, n_pos, nf, 9) complexo, ``positions_z``,
        ``info`` (dispatcher) e ``backend`` efetivo. Se cancelado, ``{"cancelled":
        True, "backend": ...}`` (sem ``H6`` — resultado parcial descartado).

    Modos (``request.geology_mode``):
      ┌──────────────┬─────────────────────────────────────────────────────────┐
      │ "fixed"      │ geologia 3-camadas determinística (Fatia 2) — 1 batch    │
      │ "stochastic" │ N modelos TIV (gerador puro) → agrupa por n_layers →     │
      │              │ simulate_batch por grupo → reassembla (Fatia 3)          │
      └──────────────┴─────────────────────────────────────────────────────────┘

    Note:
        Cold-start por backend (ambos pagam um custo de 1ª chamada, mas só o Numba
        é pré-aquecido no SM/CI → o JAX "parece" mais lento na 1ª vez):
          - **Numba**: a 1ª chamada dispara o JIT warmup (~1-30 s); as seguintes são
            rápidas (cache ``.nbc``). O SM/CI pré-aquecem via ``geosteering-warmup``.
          - **JAX GPU**: a 1ª execução de CADA geometria/tamanho COMPILA os kernels
            XLA (dezenas de s por geometria distinta = por template K, cresce com o
            tamanho do grupo; custo ÚNICO). As seguintes reusam o cache de disco
            persistente (``~/.cache/geosteering/jax_compilation_cache``, ~30 s — mais
            rápido que o Numba, p/ qualquer K). A chave do cache embute o batch-dim por
            grupo, então mudar ``n_models``/``n_geometries`` (K) re-dispara 1 compilação.
            Menos K ⇒ menos compilações ⇒ cold-start mais curto (medido no A6000, 1000
            modelos/20 camadas/600 pos: K=1 ~108 s, K=4 ~173 s; warm ~29 s em ambos).
            NÃO é regressão nem fallback p/ CPU — ver
            ``docs/reports`` (investigação 2026-06-17).
        DÍVIDA (Fatia 5): o Numba ``prange`` (parallel, nogil) usa pool OMP próprio —
        rodar dentro de uma ``QThread`` funciona (pools distintos), mas a arquitetura
        manda ``ProcessPoolExecutor`` para CPU-bound em produção (isola o pool Numba,
        melhor p/ multi-sim concorrente).
    """
    # Threads do Numba (efeito REAL; resultados bit-idênticos) — antes do kernel.
    _apply_thread_count(request.threads_per_worker)
    if request.geology_mode in ("stochastic", "manual"):
        # Geração estocástica (ragged) OU geologia MANUAL N-camadas (replicada
        # n_models×). Ambas reusam ``_simulate_grouped`` (agrupa por n_layers).
        positions_z = _compute_positions_z(request)
        if request.geology_mode == "manual":
            models = _manual_models(request)
        else:
            models = _generate_stochastic_models(request)
            # #4 — colapsa a geometria a K templates (batchável no JAX GPU) quando
            # ativo (auto p/ jax/auto). Estocástico-único explodia n_grupos=N →
            # fallback p/ Numba; com templates o JAX-grouped satura a GPU. ρ/λ
            # ficam por-modelo; manual já é groupable (replicado), então não entra.
            if _templates_active(request):
                models = _collapse_geometry_to_templates(models, request.n_geometries)
        h6, info = _simulate_grouped(
            models, positions_z, request, progress_callback, cancel_event, pause_event
        )
        if info.get("cancelled"):
            return {"cancelled": True, "backend": request.backend}
        # Geologia por modelo (Fatia 6d — perfis ρ/λ). Já gerada; só expõe (ragged OK).
        geology = [
            {
                "rho_h": np.asarray(m["rho_h"], dtype=np.float64),
                "rho_v": np.asarray(m["rho_v"], dtype=np.float64),
                "thicknesses": np.asarray(m["thicknesses"], dtype=np.float64),
            }
            for m in models
        ]
    else:
        # Geologia FIXA (Fatia 2) — batch retangular, 1 chamada. ``_build_batch``
        # já deriva ``positions_z`` (convenção Fortran) — não recomputa.
        from geosteering_ai.simulation import simulate_batch  # lazy (Numba pesado)

        # Cancel/pause cooperativo antes da única chamada (kernel é atômico).
        if _await_resume_or_cancel(cancel_event, pause_event):
            return {"cancelled": True, "backend": request.backend}
        rho_h, rho_v, esp, positions_z = _build_batch(request)
        h6, info = simulate_batch(
            rho_h,
            rho_v,
            esp,
            positions_z,
            frequencies_hz=list(request.frequencies_hz),
            tr_spacings_m=list(request.tr_spacings_m),
            dip_degs=list(request.dip_degs),
            backend=request.backend,
            hankel_filter=request.hankel_filter,
            jax_chunk_size_models=request.jax_chunk_size_models,  # VRAM (op d); no-op p/ numba
        )
        if progress_callback is not None:
            progress_callback(request.n_models, request.n_models)  # 100%
        # Geologia por modelo (Fatia 6d) — extraída do batch fixo (n, n_layers).
        geology = [
            {"rho_h": rho_h[i], "rho_v": rho_v[i], "thicknesses": esp[i]}
            for i in range(rho_h.shape[0])
        ]
    # ── Lote 1 — artefatos Fortran-compat (.dat/.out), best-effort ───────────
    # Serialização do H6 JÁ computado (zero física). Falha (permissão/disco) NÃO
    # derruba o resultado: registra ``artifacts_error`` p/ a View exibir.
    artifacts_path: Optional[str] = None
    artifacts_error: Optional[str] = None
    if request.save_fortran_artifacts and request.output_dir and h6 is not None:
        try:
            artifacts_path = _write_fortran_artifacts(
                request.output_dir, h6, positions_z, geology, request
            )
        except Exception as exc:  # noqa: BLE001 — I/O best-effort; sim não falha
            # Loga o traceback completo (debug) + propaga a msg via artifacts_error
            # (a View a exibe no log — ver viewmodel._on_sim_finished).
            logger.warning("Falha ao gravar artefatos Fortran: %s", exc, exc_info=True)
            artifacts_error = str(exc)
    return {
        "H6": h6,
        "positions_z": positions_z,
        "info": info,
        "backend": request.backend,
        # ── Fatia 6d — geologia por modelo (perfis ρ/λ) + nº modelos ─────────
        "geology": geology,
        "n_models": len(geology),
        # ── Lote 1 — artefatos Fortran-compat (path ou erro; None se desligado) ──
        "artifacts_path": artifacts_path,
        "artifacts_error": artifacts_error,
    }
