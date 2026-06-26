# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/preferences/service.py                     ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : PreferencesService — persistência de preferências (JSON)   ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app — perspectiva Preferências (Fatia 6e)               ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: gui.persistence.atomic (atomic_write_text), json, pathlib  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Lê/grava as preferências do SM (tema, paths, backend de plot, limites  ║
# ║    de cache LRU) num JSON ATÔMICO em ``~/.config/geosteering_ai/``. NÃO    ║
# ║    importa Qt — é um serviço puro de I/O (testável headless). Degrada      ║
# ║    gracioso: arquivo ausente/corrompido → defaults (nunca levanta no load).║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    PreferencesService · DEFAULT_PREFERENCES                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``PreferencesService`` — persistência atômica das preferências (Fatia 6e)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from geosteering_ai.gui.persistence.atomic import atomic_write_text

logger = logging.getLogger(__name__)

__all__ = ["PreferencesService", "DEFAULT_PREFERENCES"]


# ──────────────────────────────────────────────────────────────────────────
# Defaults — única fonte de verdade dos valores-padrão (usada no 1º boot e no
# "Restaurar padrões"). Espelha o subconjunto da PreferencesPage do monólito
# relevante ao MVVM (Fatia 6e): tema, 4 paths, backend de plot, cache LRU.
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_PREFERENCES: Dict[str, Any] = {
    "theme": "antigravity_dark",  # único tema disponível (spec 0013)
    "plot_backend": "matplotlib",  # default conservador (sempre disponível)
    "cache_max_mb": 256,  # limite de bytes do LRU de plots (MB)
    "cache_max_snapshots": 12,  # limite de entradas (maxlen) do LRU
    # Aquecer o worker JAX GPU no boot (async): tira o cold-start XLA da 1ª sim. Opt-in
    # (False) — não gasta GPU/VRAM no boot de quem só usa Numba (nem do CI/Studio).
    "jax_boot_warmup": False,
    # Aquecer o JAX GPU com as formas EXATAS da config ao SELECIONAR backend jax/auto
    # (async, em background): a 1ª sim real fica cache-hit (190 s → ~12 s). Default ON —
    # não-bloqueante, cancelável e no-op p/ Numba/sem-jax (find_spec). Ver jax_warmup_spec.
    "jax_auto_warmup": True,
    "paths": {  # paridade com load_paths()/save_paths() do monólito
        "output_dir": "",
        "tatu_binary": "",
        "python_binary": "",
        "geosteering_ai": "",
    },
}


class PreferencesService:
    """Persiste/recupera as preferências do SM num JSON atômico (sem Qt).

    Attributes:
        path: caminho do arquivo de preferências (``preferences.json``).

    Example:
        >>> svc = PreferencesService(path="/tmp/prefs.json")
        >>> svc.save({"theme": "antigravity_dark", "cache_max_mb": 512})
        >>> svc.load()["cache_max_mb"]
        512

    Note:
        ``load`` SEMPRE devolve um dict completo (merge sobre
        :data:`DEFAULT_PREFERENCES`) — chaves ausentes/arquivo corrompido caem
        para o default (forward-compat: chaves extras desconhecidas são
        preservadas no round-trip). NÃO importa Qt → testável headless.
    """

    def __init__(self, path: Optional[str | os.PathLike[str]] = None) -> None:
        """Inicializa o serviço.

        Args:
            path: caminho do JSON de preferências. Default:
                ``~/.config/geosteering_ai/preferences.json``.
        """
        if path is None:
            path = Path.home() / ".config" / "geosteering_ai" / "preferences.json"
        self.path: Path = Path(path)

    def defaults(self) -> Dict[str, Any]:
        """Cópia profunda dos valores-padrão (segura para mutação)."""
        import copy

        return copy.deepcopy(DEFAULT_PREFERENCES)

    def load(self) -> Dict[str, Any]:
        """Carrega as preferências do disco (merge sobre os defaults).

        Returns:
            Dict completo de preferências. Arquivo ausente ou JSON inválido →
            devolve os defaults (degradação graciosa, nunca levanta).
        """
        data = self.defaults()
        try:
            raw = self.path.read_text(encoding="utf-8")
            stored = json.loads(raw)
        except (FileNotFoundError, OSError):
            return data  # 1º boot — sem arquivo
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("preferences.json corrompido (%s) — usando defaults", exc)
            return data
        if not isinstance(stored, dict):
            logger.warning("preferences.json não é um objeto — usando defaults")
            return data
        # Merge raso preservando chaves extras (forward-compat); 'paths' é
        # mergeado num nível adicional para não perder paths default ausentes.
        merged = {**data, **stored}
        if isinstance(stored.get("paths"), dict):
            merged["paths"] = {**data["paths"], **stored["paths"]}
        return merged

    def save(self, data: Dict[str, Any]) -> None:
        """Grava as preferências no disco de forma ATÔMICA (crash-safe).

        Args:
            data: dict de preferências (tipicamente ``viewmodel.to_session_dict()``).

        Note:
            Cria o diretório-pai se necessário. Usa
            :func:`geosteering_ai.gui.persistence.atomic.atomic_write_text`
            (write-temp → fsync → ``os.replace``).
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(data, indent=2, ensure_ascii=False))
        logger.debug("Preferências salvas em %s", self.path)
