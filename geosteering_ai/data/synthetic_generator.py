# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/data/synthetic_generator.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : DataPipeline — Gerador sintético Python puro              ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13 (Sprint 6.2 — PR #14c)                         ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy + simulação Python otimizada                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Substitui ``Fortran_Gerador/batch_runner.py`` (ProcessPoolExecutor    ║
# ║    chamando tatu.x via subprocess) por um gerador 100% Python que       ║
# ║    respeita ``config.simulator_backend`` (Sprint 6.1) para rotear       ║
# ║    entre Numba / JAX híbrido / JAX nativo / Fortran legacy.             ║
# ║                                                                           ║
# ║    Compatibilidade de saída 22-col preservada (ver io/binary_dat.py)   ║
# ║    — os .dat gerados por este módulo são lidos por ``loading.py`` do  ║
# ║    mesmo jeito que os do Fortran.                                       ║
# ║                                                                           ║
# ║  DIFERENÇAS VS BATCH_RUNNER FORTRAN                                       ║
# ║    ┌─────────────────────────┬───────────────┬────────────────────────┐  ║
# ║    │  Aspecto                │  Fortran      │  SyntheticDataGenerator│  ║
# ║    ├─────────────────────────┼───────────────┼────────────────────────┤  ║
# ║    │  Subprocess             │  sim (tatu.x) │  não                   │  ║
# ║    │  I/O binário 22-col     │  sim          │  sim (paridade)        │  ║
# ║    │  Paralelização          │  ProcessPool  │  prange Numba ou vmap │  ║
# ║    │  GPU opcional           │  não          │  sim (backend='jax')  │  ║
# ║    │  In-memory batching     │  não          │  sim (batch tensor)   │  ║
# ║    └─────────────────────────┴───────────────┴────────────────────────┘  ║
# ║                                                                           ║
# ║  CORRELAÇÃO COM CLAUDE.md                                                 ║
# ║    • SPACING_METERS = 1.0, FREQUENCY_HZ = 20000 por default               ║
# ║    • Scaler fit em dados LIMPOS (este gerador produz dados limpos;      ║
# ║      ruído é injetado on-the-fly pelo DataPipeline).                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Gerador sintético Python puro — substituto do batch_runner Fortran.

Expõe :class:`SyntheticDataGenerator` com `generate_batch()` que
retorna um :class:`GeneratedBatch` imutável contendo:

- ``H_tensor`` complex128 de shape ``(n_models, n_pos, nf, 9)``
- ``rho_h`` / ``rho_v`` / ``esp`` do modelo geológico amostrado
- ``dat_22col`` estruturado com ``DTYPE_22COL`` para ``.dat`` binário

Example:
    Geração de 100 modelos via Numba (CPU local)::

        >>> from geosteering_ai.config import PipelineConfig
        >>> from geosteering_ai.data.synthetic_generator import (
        ...     SyntheticDataGenerator,
        ... )
        >>> cfg = PipelineConfig(simulator_backend="numba")
        >>> gen = SyntheticDataGenerator(cfg)
        >>> batch = gen.generate_batch(n_models=100, seed=42)
        >>> batch.H_tensor.shape
        (100, 100, 1, 9)

Note:
    Estratégias de amostragem (``"sobol"``, ``"uniform"``, ``"mixed"``)
    replicam o comportamento de ``fifthBuildTIVModels.py``, mas em Python
    puro. Seed é deterministicamente propagada.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# GeneratedBatch — container imutável de resultado
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeneratedBatch:
    """Batch de modelos sintetizados com tensor H correspondente.

    Attributes:
        H_tensor: (n_models, n_pos, nf, 9) complex128 — tensor EM.
        rho_h: (n_models, n_layers) float64 — resistividades horizontais.
        rho_v: (n_models, n_layers) float64 — resistividades verticais.
        esp: (n_models, n_layers-2) float64 — espessuras internas.
        positions_z: (n_pos,) float64 — profundidades ponto-médio.
        freqs_hz: (nf,) float64 — frequências usadas.
        dat_22col: (n_models·n_pos·nf,) estruturado ``DTYPE_22COL``.
        metadata: dict — tempo total, backend usado, seed.
        tr_spacings_m: (nTR,) float64 — espaçamentos T-R (self-describing p/
            extração de features por-config). Default array vazio (compat).
        dip_degs: (nAng,) float64 — ângulos dip usados. Default array vazio.
    """

    H_tensor: np.ndarray
    rho_h: np.ndarray
    rho_v: np.ndarray
    esp: np.ndarray
    positions_z: np.ndarray
    freqs_hz: np.ndarray
    dat_22col: np.ndarray
    metadata: dict = field(default_factory=dict)
    tr_spacings_m: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    dip_degs: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    @property
    def n_models(self) -> int:
        return int(self.H_tensor.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.rho_h.shape[1])


@dataclass(frozen=True)
class FeatureDataset:
    """Dataset COMPACTO de features extraído de um :class:`GeneratedBatch`.

    Reduz o ``H_tensor`` (9 componentes complexas = 18 floats/posição) a um array
    enxuto de input features (``input_features``) + targets (``output_targets``),
    pronto p/ o ``DataPipeline``/treino.

    Attributes:
        X: ``(n_models, [n_config,] n_pos, n_features)`` float32 — input da rede.
        y: ``(n_models, [n_config,] n_pos, n_targets)`` float32 — alvos (ρₕ/ρᵥ obs).
        positions_z: ``(n_pos,)`` float64 — profundidades de medição.
        feature_names: nomes das colunas de ``X``.
        metadata: dict — input_features, output_targets, feature_view, GS, etc.
    """

    X: np.ndarray
    y: np.ndarray
    positions_z: np.ndarray
    feature_names: list
    metadata: dict = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return int(self.X.shape[0])


# ──────────────────────────────────────────────────────────────────────────────
# SyntheticDataGenerator
# ──────────────────────────────────────────────────────────────────────────────


class SyntheticDataGenerator:
    """Gerador sintético Python para dados de treino do geosteering.

    Substitui o pipeline Fortran ``batch_runner.py`` → ``tatu.x`` por um
    gerador in-process que usa o simulador Python otimizado. O backend
    é selecionado via ``config.simulator_backend``.

    Attributes:
        cfg: Instância de ``PipelineConfig`` que orquestra o gerador.

    Example:
        >>> cfg = PipelineConfig(simulator_backend="numba")
        >>> gen = SyntheticDataGenerator(cfg)
        >>> batch = gen.generate_batch(n_models=10, seed=42)
        >>> batch.metadata["backend"]
        'numba'

    Note:
        Este gerador usa perfis geológicos amostrados uniformemente
        em ``[rho_min, rho_max]`` com escala log. Para estratégias mais
        sofisticadas (Sobol, mixed), use ``fifthBuildTIVModels.py``
        como referência; implementação Python em Sprint 6.3 (futuro).
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Inicializa o gerador.

        Args:
            config: ``PipelineConfig`` com ``simulator_backend`` validado.

        Raises:
            ValueError: Se ``simulator_backend='fortran_f2py'`` for pedido
                (o gerador legacy é ``Fortran_Gerador/batch_runner.py``).
        """
        if config.simulator_backend == "fortran_f2py":
            raise ValueError(
                "SyntheticDataGenerator não suporta simulator_backend='fortran_f2py'. "
                "Use Fortran_Gerador/batch_runner.py para o caminho legacy, "
                "ou altere para 'numba' / 'jax'."
            )
        self.cfg = config

    def generate_batch(
        self,
        n_models: int,
        n_positions: int = 100,
        n_layers: int = 3,
        rho_h_range: tuple[float, float] = (1.0, 1000.0),
        rho_v_range: tuple[float, float] = (1.0, 1000.0),
        thickness_range: tuple[float, float] = (1.0, 10.0),
        strategy: Literal["uniform", "log_uniform"] = "log_uniform",
        seed: int = 42,
        frequencies_hz: Optional[Sequence[float]] = None,
        dip_degs: Optional[Sequence[float]] = None,
        tr_spacings_m: Optional[Sequence[float]] = None,
        n_geometries: Optional[int] = None,
        jax_chunk_size_models: Optional[int] = None,
        geometry_mode: Literal["templates", "quantize", "per_model"] = "templates",
        quantize_step: Optional[float] = None,
        numba_fallback: bool = True,
    ) -> GeneratedBatch:
        """Gera ``n_models`` modelos sintéticos com tensor H correspondente.

        Backend ``jax`` usa o caminho BATCHED + agrupado
        (:func:`simulate_multi_jax_batched_grouped` — vmap sobre modelos +
        agrupamento por geometria, 1.5–1.9× Numba a n≥32 quando há geometria
        compartilhada). Backend ``numba`` itera por modelo via
        :func:`simulate_multi` (paralelização interna por prange — ótimo p/ CPU).
        Ambos suportam multi-config (frequência × dip × TR).

        Args:
            n_models: Número de modelos a gerar.
            n_positions: Posições de medição por modelo. Default 100.
            n_layers: Camadas por modelo (inclui 2 semi-espaços). Default 3.
            rho_h_range: ``(min, max)`` para ρₕ (Ω·m).
            rho_v_range: ``(min, max)`` para ρᵥ.
            thickness_range: ``(min, max)`` para espessuras internas (m).
            strategy: Distribuição de amostragem. ``"log_uniform"`` (default)
                é mais fiel ao comportamento físico em subsuperfície.
            seed: Semente para reprodutibilidade.
            frequencies_hz: Lista de frequências (Hz). Default ``[cfg.frequency_hz]``.
            dip_degs: Lista de ângulos dip (°). Default ``[0.0]``.
            tr_spacings_m: Lista de espaçamentos T-R (m). Default
                ``[cfg.spacing_meters]``.
            n_geometries: Se definido (≤ ``n_models``), amostra K geometrias
                (``esp``) DISTINTAS e as replica entre os modelos (round-robin),
                fazendo modelos COMPARTILHAREM geometria → ativa o caminho
                bucketed rápido no backend JAX. ``None`` (default) = geometria
                única por modelo (cada modelo é um grupo → correto, mas sem
                ganho de batch; ainda evita o kernel unified/OOM via agrupamento).
            jax_chunk_size_models: (Só backend JAX) fatia o eixo de modelos em
                grupos grandes p/ limitar VRAM (ex.: 32–64). Recomendado quando
                um grupo de geometria tem muitos modelos (ex.: ``n_geometries=1``
                + ``n_models`` grande). ``None`` = batch monolítico por grupo.
            geometry_mode: Como amostrar ``esp`` (reconcilia diversidade × GPU):
                ``"templates"`` (DEFAULT, Opção A) — K geometrias distintas
                replicadas (K=``n_geometries`` ou ``max(1, n_models//32)``); grupos
                grandes batcháveis → bucketed rápido (1.87× Numba). ``"quantize"``
                (Opção b) — ``esp`` contínuo arredondado a ``quantize_step``; mais
                diverso, ainda agrupável. ``"per_model"`` — único por modelo (máx
                diversidade; no JAX dispara ``numba_fallback``).
            quantize_step: (Só ``geometry_mode="quantize"``) passo de quantização
                de ``esp`` em metros. ``None`` → ``(thick_max-thick_min)/8``.
            numba_fallback: Se ``True`` (default) e backend=jax com geometria
                MAL-AGRUPÁVEL (>50% dos modelos em grupos próprios → JAX degenera),
                cai para Numba 16w×4t automaticamente (Opção c automática). ``False``
                força o caminho JAX-grouped mesmo degenerado.

        Returns:
            :class:`GeneratedBatch`. ``H_tensor`` tem shape
            ``(n_models, n_pos, nf, 9)`` quando ``nTR==nAng==1`` (compat
            retroativa), ou ``(n_models, nTR, nAng, n_pos, nf, 9)`` em
            multi-config. ``dat_22col`` cobre a config de referência
            (TR₀, dip₀, freq₀).

        Note:
            Grid ``positions_z`` compartilhado entre modelos (batched exige).
            ``metadata["n_geometry_groups"]`` indica quão batchável era a
            geometria (1 = ótimo; ``n_models`` = sem compartilhamento).

            **OOM-fix por design**: o backend JAX SEMPRE usa
            :func:`simulate_multi_jax_batched_grouped` (cada grupo é homogêneo →
            kernel BUCKETED). NUNCA roteia para o kernel ``unified`` (~7× lento +
            OOM 80 GB a 18cfg/600pos). VRAM controlada por ``jax_chunk_size_models``
            no eixo de modelos (não precisa de chunk de configs/posições).
        """
        from geosteering_ai.simulation import simulate_batch
        from geosteering_ai.simulation.io.binary_dat import DTYPE_22COL

        rng = np.random.default_rng(seed)

        # ── Amostragem de resistividades (ρₕ, ρᵥ) ────────────────────────────
        if strategy == "log_uniform":
            log_h = rng.uniform(
                np.log10(rho_h_range[0]),
                np.log10(rho_h_range[1]),
                size=(n_models, n_layers),
            )
            log_v = rng.uniform(
                np.log10(rho_v_range[0]),
                np.log10(rho_v_range[1]),
                size=(n_models, n_layers),
            )
            rho_h_all = 10.0**log_h
            rho_v_all = 10.0**log_v
        else:
            rho_h_all = rng.uniform(
                rho_h_range[0], rho_h_range[1], size=(n_models, n_layers)
            )
            rho_v_all = rng.uniform(
                rho_v_range[0], rho_v_range[1], size=(n_models, n_layers)
            )

        # ── Amostragem de geometria (esp) — modo de batchabilidade ───────────
        # geometry_mode reconcilia diversidade geológica × velocidade GPU:
        #   "templates" (DEFAULT, Opção A): K geometrias DISTINTAS replicadas →
        #       grupos grandes batcháveis (bucketed rápido). K = n_geometries ou
        #       max(1, n_models//32) (grupos ~32 ≈ crossover de ocupação GPU).
        #   "quantize" (Opção b): esp contínuo arredondado a quantize_step → muitos
        #       modelos compartilham esp quantizado → agrupável (mais diverso que
        #       templates, menos que per_model).
        #   "per_model": esp único por modelo (máx diversidade; cada modelo = 1
        #       grupo → no JAX dispara fallback Numba, ver abaixo).
        n_esp = max(0, n_layers - 2)
        if n_esp == 0:
            esp_all = np.zeros((n_models, 0), dtype=np.float64)
        elif geometry_mode == "templates":
            n_geo = n_geometries if n_geometries is not None else max(1, n_models // 32)
            n_geo = max(1, min(int(n_geo), n_models))
            templates = rng.uniform(
                thickness_range[0], thickness_range[1], size=(n_geo, n_esp)
            )
            esp_all = templates[np.arange(n_models) % n_geo].copy()
        elif geometry_mode == "quantize":
            step = (
                float(quantize_step)
                if quantize_step is not None
                else (thickness_range[1] - thickness_range[0]) / 8.0
            )
            if step <= 0.0:
                raise ValueError(f"quantize_step deve ser > 0 (recebido {step}).")
            raw = rng.uniform(
                thickness_range[0], thickness_range[1], size=(n_models, n_esp)
            )
            esp_all = np.clip(
                np.round(raw / step) * step, thickness_range[0], thickness_range[1]
            )
        elif geometry_mode == "per_model":
            esp_all = rng.uniform(
                thickness_range[0], thickness_range[1], size=(n_models, n_esp)
            )
        else:
            raise ValueError(
                f"geometry_mode inválido: {geometry_mode!r}. "
                "Use 'templates' | 'quantize' | 'per_model'."
            )

        # ── Grid compartilhado + configs multi-dim ───────────────────────────
        # Grid cobre o modelo MAIS ESPESSO do batch (não só o modelo 0) — com
        # geometria heterogênea/round-robin outros modelos podem ser mais espessos.
        total_thick = (
            float(esp_all.sum(axis=1).max())
            if esp_all.shape[1] > 0
            else (thickness_range[0] + thickness_range[1]) / 2.0
        )
        positions_z = np.linspace(-1.0, total_thick + 1.0, n_positions)
        freqs = np.asarray(
            frequencies_hz if frequencies_hz is not None else [self.cfg.frequency_hz],
            dtype=np.float64,
        )
        trs = list(
            tr_spacings_m if tr_spacings_m is not None else [self.cfg.spacing_meters]
        )
        dips = list(dip_degs if dip_degs is not None else [0.0])
        nf, nTR, nAng = len(freqs), len(trs), len(dips)
        freqs_list = [float(f) for f in freqs]

        # Aviso anti-pegadinha: em multi-config, o .dat 22-col cobre só a config
        # de referência (TR₀, dip₀, freq₀) — o tensor H completo preserva tudo.
        if nf * nTR * nAng > 1:
            logger.warning(
                "generate_batch multi-config (%d configs): dat_22col cobre apenas "
                "(TR0, dip0, freq0); o H_tensor completo retém todas as configs.",
                nf * nTR * nAng,
            )

        # ── Simulação batched via DISPATCHER (Sprint B) ──────────────────────
        # Reusa `simulate_batch` (árvore de decisão JAX GPU ⇄ Numba 16w×4t). Para
        # `simulator_backend` ∈ {"jax","numba","auto"}. "jax" preserva o
        # comportamento Sprint A (forçado + `numba_fallback` p/ geometria
        # não-agrupável + warn n<32); "auto" aplica a árvore estrita. H6 canônico:
        # (n_models, nTR, nAng, n_pos, nf, 9).
        H6, disp_info = simulate_batch(
            rho_h_all,
            rho_v_all,
            esp_all,
            positions_z,
            frequencies_hz=freqs_list,
            tr_spacings_m=trs,
            dip_degs=dips,
            backend=self.cfg.simulator_backend,
            numba_fallback=numba_fallback,
            dtype="complex128",
            jax_chunk_size_models=jax_chunk_size_models,
        )
        elapsed = disp_info["elapsed_s"]
        backend = disp_info["backend"]
        geometry_n_groups = disp_info["n_geometry_groups"]
        geom_info = {"n_groups": geometry_n_groups}
        # "fallback" só quando o backend pedido (jax/auto) caiu p/ Numba.
        fallback_reason = (
            disp_info["reason"]
            if (self.cfg.simulator_backend in ("jax", "auto") and backend == "numba")
            else None
        )

        # ── Squeeze p/ compat retroativa (single-TR & single-dip) ────────────
        if nTR == 1 and nAng == 1:
            H_out = H6[:, 0, 0]  # (n_models, n_pos, nf, 9)
        else:
            H_out = H6  # (n_models, nTR, nAng, n_pos, nf, 9)

        # ── .dat 22-col — config de referência (TR₀, dip₀, freq₀) ────────────
        H_ref = H6[:, 0, 0, :, 0, :]  # (n_models, n_pos, 9)
        total_recs = n_models * n_positions
        dat = np.zeros(total_recs, dtype=DTYPE_22COL)
        field_pairs = [
            ("Re_Hxx", "Im_Hxx"),
            ("Re_Hxy", "Im_Hxy"),
            ("Re_Hxz", "Im_Hxz"),
            ("Re_Hyx", "Im_Hyx"),
            ("Re_Hyy", "Im_Hyy"),
            ("Re_Hyz", "Im_Hyz"),
            ("Re_Hzx", "Im_Hzx"),
            ("Re_Hzy", "Im_Hzy"),
            ("Re_Hzz", "Im_Hzz"),
        ]
        idx = 0
        for m in range(n_models):
            for p in range(n_positions):
                dat[idx]["i"] = idx + 1
                dat[idx]["z_obs"] = positions_z[p]
                # Receptor = camada cujo topo é mais próximo de z_obs (simplificação).
                layer_idx = _find_layer_for_z(positions_z[p], esp_all[m], n_layers)
                dat[idx]["rho_h"] = rho_h_all[m, layer_idx]
                dat[idx]["rho_v"] = rho_v_all[m, layer_idx]
                for c, (re_n, im_n) in enumerate(field_pairs):
                    dat[idx][re_n] = H_ref[m, p, c].real
                    dat[idx][im_n] = H_ref[m, p, c].imag
                idx += 1

        return GeneratedBatch(
            H_tensor=H_out,
            rho_h=rho_h_all,
            rho_v=rho_v_all,
            esp=esp_all,
            positions_z=positions_z,
            freqs_hz=freqs,
            dat_22col=dat,
            tr_spacings_m=np.asarray(trs, dtype=np.float64),
            dip_degs=np.asarray(dips, dtype=np.float64),
            metadata={
                "backend": backend,
                "jax_mode": (self.cfg.simulator_jax_mode if backend == "jax" else None),
                "elapsed_s": elapsed,
                "seed": seed,
                "throughput_mod_h": 3600.0 * n_models / elapsed if elapsed > 0 else 0.0,
                "n_configs": nf * nTR * nAng,
                "config_shape": {"nf": nf, "nTR": nTR, "nAng": nAng},
                "geometry_mode": geometry_mode,
                "n_geometry_groups": (
                    geom_info.get("n_groups")
                    if geom_info.get("n_groups") is not None
                    else geometry_n_groups
                ),
                "fallback": fallback_reason,
            },
        )

    def to_feature_dataset(
        self,
        batch: GeneratedBatch,
        *,
        apply_transforms: bool = False,
        feature_view: Optional[str] = None,
        geosignal_families: Optional[Sequence[str]] = None,
        decouple: bool = True,
        dtype: type = np.float32,
    ) -> FeatureDataset:
        """Extrai um dataset COMPACTO ``(X, y)`` de um :class:`GeneratedBatch`.

        Compactação: ``H_tensor`` (9 componentes complexas = 18 floats/posição) →
        ``input_features`` (default 5: ``[z, Re/Im Hxx, Re/Im Hzz]``) + ``output_targets``
        (2: ρₕ/ρᵥ na profundidade de obs). Reusa as funções VALIDADAS da pipeline
        (``apply_feature_view``/``compute_geosignals``/``apply_decoupling``).

        Modos:
          - ``apply_transforms=False`` (DEFAULT): ``X`` = ``input_features`` CRU →
            alimenta o ``DataPipeline``, que aplica ruído→decouple→FV→GS on-the-fly
            (caminho FÍSICO correto do treino — ruído é regerado por época).
          - ``apply_transforms=True``: aplica decouple→FV→GS em CLEAN (dataset
            "pronto" p/ inspeção/scaler em dados limpos; NÃO substitui o ruído
            on-the-fly de treino).

        Args:
            batch: :class:`GeneratedBatch` de :meth:`generate_batch`.
            apply_transforms: aplica decouple→FV→GS (default ``False`` = cru).
            feature_view: nome da FV (default ``cfg.feature_view``). Só com transforms.
            geosignal_families: famílias GS (ex.: ``["USD","UHR"]``). Só com transforms.
            decouple: subtrai campo direto (ACp/ACx) antes de FV/GS (default ``True``).
            dtype: dtype de saída (default ``float32`` — compacto p/ treino).

        Returns:
            :class:`FeatureDataset` com ``X``/``y`` compactos (config achatado se 1).

        Note:
            Mapa de componentes (``io/binary_dat.py``): Hxx=idx0, Hzz=idx8 no eixo
            de 9; 22-col col(4+2c)=Re, col(5+2c)=Im. ``input_features``/``output_targets``
            são índices 22-col (default ``[1,4,5,20,21]`` / ``[2,3]``).
        """
        from geosteering_ai.data.feature_views import apply_feature_view
        from geosteering_ai.data.geosignals import compute_geosignals
        from geosteering_ai.data.loading import apply_decoupling

        cfg = self.cfg
        input_features = list(cfg.input_features)
        output_targets = list(cfg.output_targets)
        view = feature_view if feature_view is not None else cfg.feature_view

        positions_z = np.asarray(batch.positions_z, dtype=np.float64)
        n_pos = positions_z.shape[0]

        # ── Normaliza H p/ (n_models, n_config, n_pos, 9) ────────────────────
        H = batch.H_tensor
        if H.ndim == 4:  # (n_models, n_pos, nf, 9) — single TR/dip
            n_models, _, nf, _ = H.shape
            Hc = np.transpose(H, (0, 2, 1, 3)).reshape(n_models, nf, n_pos, 9)
        elif H.ndim == 6:  # (n_models, nTR, nAng, n_pos, nf, 9)
            n_models, nTR, nAng, _, nf, _ = H.shape
            Hc = np.transpose(H, (0, 1, 2, 4, 3, 5)).reshape(
                n_models, nTR * nAng * nf, n_pos, 9
            )
        else:
            raise ValueError(f"H_tensor com ndim inesperado: {H.ndim}")
        n_config = Hc.shape[1]

        # ── ρ nos obs (model,pos) — config-independente (geom igual entre configs) ──
        n_layers = batch.n_layers
        rho_h_obs = np.empty((n_models, n_pos), dtype=np.float64)
        rho_v_obs = np.empty((n_models, n_pos), dtype=np.float64)
        for m in range(n_models):
            for p in range(n_pos):
                li = _find_layer_for_z(float(positions_z[p]), batch.esp[m], n_layers)
                rho_h_obs[m, p] = batch.rho_h[m, li]
                rho_v_obs[m, p] = batch.rho_v[m, li]

        # ── Bloco 22-col (n_models, n_config, n_pos, 22) ─────────────────────
        block = np.zeros((n_models, n_config, n_pos, 22), dtype=np.float64)
        block[..., 1] = positions_z[None, None, :]
        block[..., 2] = rho_h_obs[:, None, :]
        block[..., 3] = rho_v_obs[:, None, :]
        for c in range(9):
            block[..., 4 + 2 * c] = Hc[..., c].real
            block[..., 5 + 2 * c] = Hc[..., c].imag

        # ── (opcional) decouple → FV → GS em CLEAN ───────────────────────────
        if apply_transforms and decouple:
            block = apply_decoupling(block.reshape(-1, 22), cfg).reshape(
                n_models, n_config, n_pos, 22
            )

        x_base = block[..., input_features]  # (n_models, n_config, n_pos, n_in)
        y = block[..., output_targets]
        feature_names = [f"col{c}" for c in input_features]

        if apply_transforms:
            n_in = x_base.shape[-1]
            x3 = x_base.reshape(n_models * n_config, n_pos, n_in)
            h1 = (
                (input_features.index(4), input_features.index(5))
                if 4 in input_features and 5 in input_features
                else None
            )
            h2 = (
                (input_features.index(20), input_features.index(21))
                if 20 in input_features and 21 in input_features
                else None
            )
            if h1 is not None and h2 is not None:
                x3 = apply_feature_view(x3, view, h1_cols=h1, h2_cols=h2)
            x_base = x3.reshape(n_models, n_config, n_pos, x3.shape[-1])
            if geosignal_families:
                gs2 = compute_geosignals(
                    block.reshape(-1, 22), list(geosignal_families), n_columns=22
                )
                gs = gs2.reshape(n_models, n_config, n_pos, -1)
                x_base = np.concatenate([x_base, gs], axis=-1)
                for fam in geosignal_families:
                    feature_names += [f"{fam}_att", f"{fam}_phase"]

        # ── Squeeze do eixo de config se single-config ──────────────────────
        if n_config == 1:
            x_out, y_out = x_base[:, 0], y[:, 0]
        else:
            x_out, y_out = x_base, y

        return FeatureDataset(
            X=np.asarray(x_out, dtype=dtype),
            y=np.asarray(y_out, dtype=dtype),
            positions_z=positions_z,
            feature_names=feature_names,
            metadata={
                "input_features": input_features,
                "output_targets": output_targets,
                "feature_view": (view if apply_transforms else "raw"),
                "geosignal_families": (
                    list(geosignal_families)
                    if (apply_transforms and geosignal_families)
                    else None
                ),
                "decoupled": bool(apply_transforms and decouple),
                "n_config": n_config,
                "compaction": f"18 floats/pos -> {int(x_out.shape[-1])} features",
            },
        )


def _find_layer_for_z(z: float, esp: np.ndarray, n_layers: int) -> int:
    """Determina a camada que contém uma profundidade z.

    Args:
        z: Profundidade em metros.
        esp: Espessuras internas (n_layers-2,).
        n_layers: Número total de camadas.

    Returns:
        Índice 0-based da camada. Camada 0 é semi-espaço superior (z<0).
    """
    if n_layers <= 1 or esp.shape[0] == 0:
        return 0
    if z < 0:
        return 0
    acc = 0.0
    for i, thickness in enumerate(esp):
        acc += thickness
        if z < acc:
            return i + 1
    return n_layers - 1


def _layer_at_batch(
    positions_z: np.ndarray, esp_batch: np.ndarray, n_layers: int
) -> np.ndarray:
    """Twin VETORIZADO de :func:`_find_layer_for_z` — mapeia z→camada p/ um BATCH.

    Equivalência BIT-EXATA com o laço escalar (mesma convenção de fronteira
    ``z < acc`` estrita): ``z < 0`` → camada 0 (semi-espaço superior); senão a
    camada é ``(nº de fronteiras cumsum(esp) ≤ z) + 1``, clampada a ``n_layers-1``.
    Usado pela CLI (``_exec.rho_at_obs_from_batch``) p/ preencher col2/col3 (res_h/
    res_v) do ``.dat`` 22-col sem o laço Python por-modelo.

    Args:
        positions_z: ``(n_pos,)`` profundidades de observação (m).
        esp_batch: ``(n_models, n_layers-2)`` espessuras internas por modelo (m).
        n_layers: número total de camadas (inclui 2 semi-espaços).

    Returns:
        ``np.ndarray`` ``(n_models, n_pos)`` ``intp`` — índice 0-based da camada
        de cada ``z`` por modelo (consumível por ``np.take_along_axis``).

    Note:
        ``n_layers ≤ 1`` ou ``n_esp == 0`` → tudo na camada 0 (igual ao escalar).
        A fronteira em ``z`` exato vai para a camada SEGUINTE (``≤`` em searchsorted
        espelha o ``<`` estrito do escalar).
    """
    z = np.asarray(positions_z, dtype=np.float64).reshape(-1)  # (n_pos,)
    esp = np.atleast_2d(np.asarray(esp_batch, dtype=np.float64))  # (n_models, n_esp)
    n_models, n_esp = esp.shape
    if n_layers <= 1 or n_esp == 0:
        return np.zeros((n_models, z.shape[0]), dtype=np.intp)
    bnd = np.cumsum(esp, axis=1)  # (n_models, n_esp) — fronteiras acumuladas
    # nº de fronteiras ≤ z → a 1ª fronteira > z está nesse índice; camada = +1.
    counts = (bnd[:, :, None] <= z[None, None, :]).sum(axis=1)  # (n_models, n_pos)
    layer = np.where(z[None, :] < 0.0, 0, counts + 1)
    return layer.astype(np.intp)


__all__ = ["GeneratedBatch", "SyntheticDataGenerator"]
