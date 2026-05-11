# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/simulate.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Subcomando `simulate` da CLI                               ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.30 — multi-dim)                         ║
# ║  Versão      : v2.30                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-10                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : argparse + simulate_multi (geosteering_ai.simulation)      ║
# ║  Dependências: numpy, geosteering_ai.simulation (lazy import dentro       ║
# ║                de handle_simulate para acelerar `--help`)                 ║
# ║  Padrão      : Hexagonal — adapter externo do simulate_multi              ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementa o handler do subcomando ``simulate``: gera N modelos       ║
# ║    sintéticos 1D 5-camada com perfis randomizados (rho_h, rho_v, esp)    ║
# ║    e executa ``simulate_multi`` em batch. Reutiliza a infraestrutura      ║
# ║    de auto-detect de paralelismo da Sprint v2.23 A.2                      ║
# ║    (``recommend_default_parallelism``).                                    ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    handle_simulate: handler do subcomando, recebe argparse.Namespace     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Subcomando ``simulate`` da CLI Geosteering AI (Sprint v2.24 — I2.6).

Gera modelos sintéticos com perfis 1D randomizados e executa
``simulate_multi`` em batch. Reutiliza a infraestrutura de auto-detect
de paralelismo da Sprint v2.23 A.2 (``recommend_default_parallelism``).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _parse_float_list(text: str | None, default: list[float]) -> list[float]:
    """Interpreta string CSV de floats; retorna ``default`` se None ou inválido.

    Aceita vírgula ou ponto-e-vírgula como separador. Espaços são ignorados.

    Args:
        text: string no formato ``"1.0,2.0,3.0"`` ou ``None``.
        default: lista retornada quando ``text`` é None, vazio ou inválido.

    Returns:
        Lista de floats parseados, ou ``default`` em caso de erro.

    Example:
        >>> _parse_float_list("2000, 20000, 100000", [20000.0])
        [2000.0, 20000.0, 100000.0]
        >>> _parse_float_list(None, [20000.0])
        [20000.0]
    """
    if not text or not text.strip():
        return default
    try:
        parsed = [
            float(v.strip()) for v in text.replace(";", ",").split(",") if v.strip()
        ]
        return parsed if parsed else default
    except ValueError:
        logger.warning(
            "Valor inválido em lista de floats: %r — usando default %s", text, default
        )
        return default


def _build_random_models(n_models: int, seed: int) -> list[dict]:
    """Gera N modelos sintéticos 1D 5-camada para smoke/benchmark.

    Cada modelo é um dict ``{rho_h, rho_v, esp}`` com:
      - 5 camadas (3 internas + 2 semi-espaços)
      - resistividades em [1.0, 100.0] Ω·m
      - razão TIV ``rho_v / rho_h`` em [1.0, 3.0]
      - espessuras das camadas internas em [2.0, 10.0] m

    Args:
        n_models: número de modelos a gerar.
        seed: semente do gerador (reprodutibilidade).

    Returns:
        Lista de dicts ``[{rho_h, rho_v, esp}, ...]``, com cada array
        em ``np.float64`` (float32 não é suportado pelo simulador
        Fortran-equivalente).

    Raises:
        ValueError: se ``n_models <= 0`` (numpy.random.default_rng
            aceita 0 mas retorna lista vazia, sem erro).

    Note:
        Range de resistividades alinhado aos modelos canônicos do
        projeto (oklahoma_3, devine_8, etc.) — facilita comparação
        com benchmarks históricos.

    Example:
        >>> models = _build_random_models(2, seed=42)
        >>> len(models)
        2
        >>> sorted(models[0].keys())
        ['esp', 'rho_h', 'rho_v']
    """
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(n_models):
        rho_h = rng.uniform(1.0, 100.0, size=5).astype(np.float64)
        ratio = rng.uniform(1.0, 3.0, size=5).astype(np.float64)
        rho_v = rho_h * ratio
        esp = rng.uniform(2.0, 10.0, size=3).astype(np.float64)
        models.append({"rho_h": rho_h, "rho_v": rho_v, "esp": esp})
    return models


def handle_simulate(args: argparse.Namespace) -> int:
    """Handler do subcomando ``simulate``.

    Constrói N modelos randomizados, executa ``simulate_multi`` em
    batch (com paralelismo auto-detectado quando args não especificam),
    e opcionalmente grava o tensor H em ``.npz``.

    Args:
        args: ``argparse.Namespace`` com campos:

            - ``models`` (int): número de modelos a gerar/simular
            - ``n_pos`` (int): pontos de profundidade por modelo
            - ``workers`` (int | None): workers paralelos
            - ``threads`` (int | None): threads Numba por worker
            - ``seed`` (int): semente do gerador
            - ``frequencies`` (str | None): CSV de Hz (ex: ``"2000,20000"``)
            - ``dips`` (str | None): CSV de graus (ex: ``"0,15,30"``)
            - ``tr_spacings`` (str | None): CSV de metros (ex: ``"0.5,1.0"``)
            - ``out`` (str | None): diretório de saída (None = memória)
            - ``quiet`` (bool): suprime logs informativos

    Returns:
        Exit code:

        - 0: sucesso (todos os modelos simulados e gravados)
        - 1: erro tratado (ValueError, RuntimeError, OSError)

    Raises:
        Não propaga exceções — todas as falhas viram exit code 1
        com ``logger.error(..., exc_info=True)``.

    Note:
        Logging é configurado em ``main.py`` (centralizado). Aqui
        apenas obtemos o logger do módulo. Imports pesados de
        simulação são lazy para preservar ``--help`` rápido.

    Example:
        Em produção via CLI::

            $ geosteering-cli simulate --models 10 --n-pos 100 --quiet
            $ geosteering-cli simulate --frequencies 2000,20000 --dips 0,15,30
            $ geosteering-cli simulate --tr-spacings 0.5,1.0,1.5 --models 50
    """
    # Lazy imports — Sprint v2.24 I2.6 — evita carregar numba em --help
    from geosteering_ai.simulation import simulate_multi
    from geosteering_ai.simulation.config import SimulationConfig

    # Parsear parâmetros multi-dim — Sprint v2.30
    frequencies_hz = _parse_float_list(getattr(args, "frequencies", None), [20000.0])
    dip_degs = _parse_float_list(getattr(args, "dips", None), [0.0])
    tr_spacings_m = _parse_float_list(getattr(args, "tr_spacings", None), [1.0])

    # Constrói perfil de posições — N pontos uniformes em [-5, +5] m
    positions_z = np.linspace(-5.0, 5.0, args.n_pos).astype(np.float64)

    # Gera modelos sintéticos reprodutíveis
    if not args.quiet:
        logger.info(
            "Gerando %d modelos sintéticos (seed=%d, n_pos=%d)...",
            args.models,
            args.seed,
            args.n_pos,
        )
    models = _build_random_models(args.models, args.seed)

    # SimulationConfig deriva auto-detect quando ambos None (Sprint v2.23 A.2)
    cfg = SimulationConfig(
        n_workers=args.workers,
        threads_per_worker=args.threads,
    )
    if not args.quiet:
        logger.info(
            "Configuração: %d modelos × %d pos × %d freq × %d dips × %d TR — %dw × %dt",
            args.models,
            args.n_pos,
            len(frequencies_hz),
            len(dip_degs),
            len(tr_spacings_m),
            cfg.n_workers or 1,
            cfg.threads_per_worker or 1,
        )

    # Executa simulação — passa parâmetros multi-dim (Sprint v2.30)
    t0 = time.perf_counter()
    try:
        result = simulate_multi(
            positions_z=positions_z,
            models=models,
            cfg=cfg,
            frequencies_hz=frequencies_hz,
            dip_degs=dip_degs,
            tr_spacings_m=tr_spacings_m,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Erro durante simulação: %s", exc, exc_info=True)
        return 1
    dt = time.perf_counter() - t0

    if not args.quiet:
        rate = (args.models / dt) * 3600 if dt > 0 else 0.0
        logger.info(
            "OK: %d modelos simulados em %.2fs (%.0f mod/h)",
            args.models,
            dt,
            rate,
        )

    # Grava resultados se --out fornecido
    if args.out is not None:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "simulate_results.npz"
        try:
            # MultiSimulationResultBatch ou MultiSimulationResult — ambos têm .H
            # mas a tipagem varia. Aceita ambos via getattr seguro.
            H = getattr(result, "H", None)
            if H is None:
                logger.warning(
                    "Resultado não tem atributo 'H' — usando repr() como fallback"
                )
                np.savez(out_file, repr=str(result))
            else:
                np.savez(out_file, H=H)
            if not args.quiet:
                logger.info("Resultados gravados em: %s", out_file)
        except OSError as exc:
            logger.error("Erro ao gravar arquivo de saída: %s", exc)
            return 1

    return 0
