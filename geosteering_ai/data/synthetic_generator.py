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
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

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
    """

    H_tensor: np.ndarray
    rho_h: np.ndarray
    rho_v: np.ndarray
    esp: np.ndarray
    positions_z: np.ndarray
    freqs_hz: np.ndarray
    dat_22col: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        return int(self.H_tensor.shape[0])

    @property
    def n_layers(self) -> int:
        return int(self.rho_h.shape[1])


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
    ) -> GeneratedBatch:
        """Gera ``n_models`` modelos sintéticos com tensor H correspondente.

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

        Returns:
            :class:`GeneratedBatch` com tensor H e metadados.

        Note:
            Todos os modelos usam o mesmo grid ``positions_z`` uniforme
            entre 0 m e ``(n_layers-1) * thickness_mean`` m.
            Frequência única lida de ``cfg.frequency_hz``.
        """
        from geosteering_ai.simulation import SimulationConfig, simulate
        from geosteering_ai.simulation.io.binary_dat import DTYPE_22COL

        rng = np.random.default_rng(seed)

        # Amostragem
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
        if n_layers >= 3:
            esp_all = rng.uniform(
                thickness_range[0], thickness_range[1], size=(n_models, n_layers - 2)
            )
        else:
            esp_all = np.zeros((n_models, 0), dtype=np.float64)

        # Grid comum
        total_thick = (
            float(esp_all[0].sum())
            if esp_all.shape[1] > 0
            else float((thickness_range[0] + thickness_range[1]) / 2)
        )
        positions_z = np.linspace(-1.0, total_thick + 1.0, n_positions)
        freqs = np.array([self.cfg.frequency_hz], dtype=np.float64)

        # Backend selection
        if self.cfg.simulator_backend == "numba":
            sim_cfg = SimulationConfig(
                frequency_hz=self.cfg.frequency_hz,
                tr_spacing_m=self.cfg.spacing_meters,
                backend="numba",
                parallel=True,
            )
        elif self.cfg.simulator_backend == "jax":
            use_native = self.cfg.simulator_jax_mode == "native"
            sim_cfg = SimulationConfig(
                frequency_hz=self.cfg.frequency_hz,
                tr_spacing_m=self.cfg.spacing_meters,
                backend="jax",
                use_native_dipoles=use_native,
            )
        else:
            raise ValueError(f"Backend inesperado: {self.cfg.simulator_backend}")

        # Loop por modelo (prange via backend; paralelização interna)
        t0 = time.perf_counter()
        H_batch = np.empty((n_models, n_positions, 1, 9), dtype=np.complex128)
        for i in range(n_models):
            rho_h_i = rho_h_all[i]
            rho_v_i = rho_v_all[i]
            esp_i = esp_all[i]
            res = simulate(
                rho_h=rho_h_i,
                rho_v=rho_v_i,
                esp=esp_i,
                positions_z=positions_z,
                cfg=sim_cfg,
            )
            H = res.H_tensor
            if H.ndim == 2:
                H = H[:, np.newaxis, :]
            H_batch[i] = H

        elapsed = time.perf_counter() - t0

        # Monta .dat 22-col compatível
        total_recs = n_models * n_positions
        dat = np.zeros(total_recs, dtype=DTYPE_22COL)
        idx = 0
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
        for m in range(n_models):
            for p in range(n_positions):
                dat[idx]["i"] = idx + 1
                dat[idx]["z_obs"] = positions_z[p]
                # Encontra camada do receptor (simplificação: posição -> índice)
                # Usamos receptor = camada com topo mais próximo de z_obs.
                z = positions_z[p]
                layer_idx = _find_layer_for_z(z, esp_all[m], n_layers)
                dat[idx]["rho_h"] = rho_h_all[m, layer_idx]
                dat[idx]["rho_v"] = rho_v_all[m, layer_idx]
                for c, (re_n, im_n) in enumerate(field_pairs):
                    dat[idx][re_n] = H_batch[m, p, 0, c].real
                    dat[idx][im_n] = H_batch[m, p, 0, c].imag
                idx += 1

        return GeneratedBatch(
            H_tensor=H_batch,
            rho_h=rho_h_all,
            rho_v=rho_v_all,
            esp=esp_all,
            positions_z=positions_z,
            freqs_hz=freqs,
            dat_22col=dat,
            metadata={
                "backend": self.cfg.simulator_backend,
                "jax_mode": (
                    self.cfg.simulator_jax_mode
                    if self.cfg.simulator_backend == "jax"
                    else None
                ),
                "elapsed_s": elapsed,
                "seed": seed,
                "throughput_mod_h": 3600.0 * n_models / elapsed if elapsed > 0 else 0.0,
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


__all__ = ["GeneratedBatch", "SyntheticDataGenerator"]
