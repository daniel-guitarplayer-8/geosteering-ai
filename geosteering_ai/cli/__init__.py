# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/__init__.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : CLI MVP — fachada pública                                  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Interface de Linha de Comando (Sprint v2.32 — warmup ep)   ║
# ║  Versão      : v2.32                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-10                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : argparse (alinhado a benchmarks/bench_v214_numba.py)       ║
# ║  Dependências: geosteering_ai.simulation (lazy import nos handlers)       ║
# ║  Padrão      : Hexagonal (CLI = adapter externo do simulador)             ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Reexporta `main` do dispatcher principal, permitindo uso via:          ║
# ║                                                                           ║
# ║      • Entry point pip: `geosteering-cli ...`                             ║
# ║      • Módulo Python: `python -m geosteering_ai.cli ...`                  ║
# ║      • Import direto: `from geosteering_ai.cli import main`               ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    main: ponto de entrada da CLI (argparse dispatcher)                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""CLI MVP do Geosteering AI v2.0 (Sprint v2.24 — I2.6).

Interface de linha de comando para tornar o simulador acessível sem
escrever código Python. Subcomandos:

- ``geosteering-cli simulate``  — gera modelos sintéticos via ``simulate_multi``
- ``geosteering-cli benchmark`` — executa cenários A/B/C/D/E/F/G e reporta mod/h
- ``geosteering-cli version``   — exibe versão atual do Simulation Manager

Entry point relacionado (Sprint v2.32, módulo separado):

- ``geosteering-warmup`` — aquece sincronicamente o cache JIT/LLVM. Útil em
  CI/notebooks antes de medir throughput. Ver ``geosteering_ai/cli/warmup.py``.

Uso típico::

    $ geosteering-cli version
    Geosteering AI Simulation Manager v2.32

    $ geosteering-cli simulate --models 100 --workers 4 --out /tmp/sim
    Simulando 100 modelos com 4 workers...
    OK: 100 modelos gravados em /tmp/sim/

    $ geosteering-cli benchmark --scenario A --n 1000
    Cenário A — 1.18M mod/h

Padrão de design:
    - argparse (alinhado ao padrão do projeto em ``benchmarks/``)
    - lazy imports dentro de cada handler (evita overhead em ``--help``)
    - reutiliza ``simulate_multi`` e ``recommend_default_parallelism``
"""

from geosteering_ai.cli._main import main

__all__ = ["main"]
