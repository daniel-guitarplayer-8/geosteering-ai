# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/__main__.py                                           ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Entry point para `python -m geosteering_ai.cli`            ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.24 — I2.6)                              ║
# ║  Versão      : v2.24                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-05-10                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : argparse (Python stdlib)                                   ║
# ║  Dependências: geosteering_ai.cli.main                                    ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Permite invocar a CLI via `python -m geosteering_ai.cli ...`           ║
# ║    quando o entry point pip-installable não está disponível (ex.:         ║
# ║    ambiente sem `pip install -e .` aplicado).                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Entry point para ``python -m geosteering_ai.cli``."""

import sys

from geosteering_ai.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
